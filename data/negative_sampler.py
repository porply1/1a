"""
data/negative_sampler.py
------------------------
动态负采样器（Dynamic Negative Sampler）。

为深度学习推荐模型（BST、PLE 等）提供每轮 Epoch 随机重采样负例的能力。
采用后台线程 + 有界队列（Buffer）实现预取，避免负采样阻塞 GPU 训练。

工作原理
--------

    ┌──────────────────────────────┐
    │   主线程（GPU 训练）            │
    │                              │          ┌ Queue(buffer_size) ┐
    │  for epoch in 1..T:          │  get()   │  (X, y) epoch_N+2  │
    │    X_ep, y_ep = sampler ─────┼← block ─│  (X, y) epoch_N+1  │
    │                  .get()      │  if      │  (X, y) epoch_N    │  ← 后台线程 put()
    │    train(X_ep, y_ep)         │  empty   └────────────────────┘
    └──────────────────────────────┘               ↑
                                        ┌──────────────────────┐
                                        │  后台线程（CPU 采样）  │
                                        │  while running:      │
                                        │    neg = sample()    │
                                        │    mixed = pos+neg   │
                                        │    queue.put(mixed)  │
                                        └──────────────────────┘

采样策略
--------
  每个 Epoch：
    1. 保留全部正例（y == 1）
    2. 从负例池中随机无放回抽取 min(n_pos × neg_ratio, |neg_pool|) 条
    3. 合并打乱后推入队列

  注意：本模块仅做池内重采样（Population-level Negative Sampling）。
  物品级别的随机负对（User × Random-Item）需要在数据加载阶段
  通过 DataLoader 的 in-batch negative 策略实现（见 two_tower_wrapper.py）。

YAML 配置示例
-------------
  models:
    - name: "bst"
      type: "BSTModel"
      negative_sampling:
        enable:      true
        neg_ratio:   4          # 正:负 ≈ 1:4
        buffer_size: 3          # 预取缓冲 epoch 数（越大越不 block，内存稍大）
        n_workers:   1          # 后台采样线程数（通常 1 就足够）
        seed:        42
        verbose:     false      # true → debug 级别输出每轮采样信息
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------

@dataclass
class NegativeSamplerConfig:
    """
    动态负采样器配置。

    Parameters
    ----------
    enable : bool
        False → 不启用采样，模型 fit() 原样使用传入数据。

    neg_ratio : int
        每个正例匹配的负例数量（1 : neg_ratio）。
        若负例池不足，则取全部可用负例。

    buffer_size : int
        后台线程预取的 Epoch 数量（有界队列容量）。
        建议 2~5：太小会让 GPU 频繁等待；太大占用更多内存。

    n_workers : int
        后台采样线程数。数据量 < 10M 行时 1 足够；超过可设 2。

    seed : int
        随机种子（控制每个 worker 的 RandomState）。

    verbose : bool
        True → 以 DEBUG 级别记录每 Epoch 的采样统计。
    """
    enable:      bool  = False
    neg_ratio:   int   = 4
    buffer_size: int   = 3
    n_workers:   int   = 1
    seed:        int   = 42
    verbose:     bool  = False

    @classmethod
    def from_dict(cls, d: dict) -> "NegativeSamplerConfig":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# 核心采样引擎
# ---------------------------------------------------------------------------

class DynamicNegativeSampler:
    """
    后台线程 + 有界队列 动态负采样器。

    生命周期
    --------
    ::

        sampler = DynamicNegativeSampler(NegativeSamplerConfig(neg_ratio=4))
        sampler.fit(X_train, y_train)   # 切分正负样本池
        sampler.start()                  # 启动后台线程，开始预取
        for epoch in range(n_epochs):
            X_ep, y_ep = sampler.get_next_epoch()   # blocking
            train(X_ep, y_ep)
        sampler.stop()                   # 回收线程

    线程安全
    --------
    - `_stop_event`（threading.Event）控制后台线程退出
    - `queue.Queue` 保证生产者/消费者线程安全
    - `get_next_epoch()` 设置 60 s timeout，避免主线程死锁
    """

    def __init__(self, config: NegativeSamplerConfig):
        self.config   = config
        self._pos_X:  Optional[pd.DataFrame] = None
        self._pos_y:  Optional[np.ndarray]   = None
        self._neg_X:  Optional[pd.DataFrame] = None
        self._neg_y:  Optional[np.ndarray]   = None

        self._queue:       Optional[queue.Queue]  = None
        self._threads:     list[threading.Thread] = []
        self._stop_event   = threading.Event()

    # ------------------------------------------------------------------
    # fit：切分正负样本池
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> "DynamicNegativeSampler":
        """
        从训练集切分正例（y==1）与负例（y==0）采样池。

        Parameters
        ----------
        X : pd.DataFrame  训练特征（可含标签列；整行原样保留，不做列过滤）
        y : array-like    二值标签，0=负例，1=正例
        """
        y_arr    = np.asarray(y, dtype=np.float32).ravel()
        pos_mask = y_arr == 1
        neg_mask = y_arr == 0

        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        if n_pos == 0:
            raise ValueError(
                "DynamicNegativeSampler.fit(): 训练集中没有正例（y==1）。"
                "请检查标签列——确认正例标签值确实为 1。"
            )
        if n_neg == 0:
            warnings.warn(
                "DynamicNegativeSampler: 训练集中没有负例（y==0），"
                "负采样将退化为无操作，直接返回原始数据。",
                UserWarning, stacklevel=2,
            )

        self._pos_X = X.loc[pos_mask].reset_index(drop=True)
        self._pos_y = y_arr[pos_mask]
        self._neg_X = X.loc[neg_mask].reset_index(drop=True)
        self._neg_y = y_arr[neg_mask]

        n_neg_target = min(n_pos * self.config.neg_ratio, n_neg)
        logger.info(
            f"[NegSampler] fit 完成 | "
            f"pos={n_pos:,} | neg_pool={n_neg:,} | "
            f"per_epoch_neg≈{n_neg_target:,} | "
            f"ratio≈1:{self.config.neg_ratio}"
        )
        return self

    # ------------------------------------------------------------------
    # start / stop：线程生命周期
    # ------------------------------------------------------------------

    def start(self) -> "DynamicNegativeSampler":
        """启动后台预取线程，开始异步填充缓冲队列。"""
        if self._pos_X is None:
            raise RuntimeError("请先调用 fit() 再调用 start()。")
        if self._neg_X is None or len(self._neg_X) == 0:
            logger.warning("[NegSampler] 无负例，跳过后台线程启动。")
            return self

        self._queue      = queue.Queue(maxsize=self.config.buffer_size)
        self._stop_event.clear()
        self._threads    = []

        n_workers = max(1, self.config.n_workers)
        for wid in range(n_workers):
            t = threading.Thread(
                target = self._producer_loop,
                args   = (wid,),
                daemon = True,
                name   = f"NegSampler-W{wid}",
            )
            t.start()
            self._threads.append(t)

        logger.info(
            f"[NegSampler] {n_workers} 个后台线程已启动 | "
            f"buffer_size={self.config.buffer_size}"
        )
        return self

    def stop(self) -> None:
        """
        终止后台线程并清空队列。

        在模型 fit() 结束后（无论正常结束还是异常）调用。
        """
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()
        # 清空队列，防止 producer 在 put() 处永久阻塞
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    # ------------------------------------------------------------------
    # 主线程接口
    # ------------------------------------------------------------------

    def get_next_epoch(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        从缓冲队列取一个 Epoch 的混合训练数据（blocking，超时 60 s 后降级）。

        若无负例池（fit 时全为正例），直接返回正例原始数据。

        Returns
        -------
        X_epoch : pd.DataFrame  已打乱的混合数据（正例 + 采样负例）
        y_epoch : np.ndarray    对应二值标签，dtype=float32
        """
        if self._queue is None or (self._neg_X is not None and len(self._neg_X) == 0):
            return self._pos_X.copy(), self._pos_y.copy()

        try:
            X_ep, y_ep = self._queue.get(timeout=60.0)
            return X_ep, y_ep
        except queue.Empty:
            warnings.warn(
                "DynamicNegativeSampler.get_next_epoch(): "
                "等待 60 s 未收到数据，降级为返回正例原始数据。"
                "请检查后台线程是否正常运行。",
                RuntimeWarning, stacklevel=2,
            )
            return self._pos_X.copy(), self._pos_y.copy()

    # ------------------------------------------------------------------
    # 后台生产者
    # ------------------------------------------------------------------

    def _producer_loop(self, worker_id: int) -> None:
        """后台线程持续采样并推入队列。"""
        rng = np.random.RandomState(self.config.seed + worker_id + 1)

        while not self._stop_event.is_set():
            try:
                X_ep, y_ep = self._sample_one_epoch(rng)
                # timeout=2s：stop() 后最多 2 s 退出
                self._queue.put((X_ep, y_ep), timeout=2.0)
            except queue.Full:
                # 队列满，稍后重试（不 sleep，直接循环检测 stop_event）
                continue
            except Exception as exc:
                logger.warning(f"[NegSampler W{worker_id}] 采样异常: {exc}")
                time.sleep(0.5)

    def _sample_one_epoch(
        self, rng: np.random.RandomState
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        生成单 Epoch 的混合数据集。

        1. 取全量正例
        2. 从负例池无放回抽取 min(n_pos × neg_ratio, |neg_pool|) 条
        3. 合并 + shuffle
        """
        n_pos        = len(self._pos_X)
        n_neg_avail  = len(self._neg_X)
        n_neg_sample = min(n_pos * self.config.neg_ratio, n_neg_avail)

        neg_idx      = rng.choice(n_neg_avail, size=n_neg_sample, replace=False)
        neg_X_sample = self._neg_X.iloc[neg_idx]
        neg_y_sample = self._neg_y[neg_idx]

        X_mixed = pd.concat(
            [self._pos_X, neg_X_sample],
            axis=0, ignore_index=True,
        )
        y_mixed = np.concatenate([self._pos_y, neg_y_sample])

        perm    = rng.permutation(len(X_mixed))
        X_out   = X_mixed.iloc[perm].reset_index(drop=True)
        y_out   = y_mixed[perm].astype(np.float32)

        if self.config.verbose:
            logger.debug(
                f"[NegSampler] 采样完成 | "
                f"pos={n_pos} | neg={n_neg_sample} | "
                f"total={len(X_out)} | ratio=1:{n_neg_sample / max(n_pos, 1):.1f}"
            )
        return X_out, y_out

    # ------------------------------------------------------------------
    # 属性 / repr
    # ------------------------------------------------------------------

    @property
    def n_positives(self) -> int:
        return len(self._pos_X) if self._pos_X is not None else 0

    @property
    def n_negatives(self) -> int:
        return len(self._neg_X) if self._neg_X is not None else 0

    @property
    def is_running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def __repr__(self) -> str:
        return (
            f"DynamicNegativeSampler("
            f"pos={self.n_positives:,}, neg_pool={self.n_negatives:,}, "
            f"ratio=1:{self.config.neg_ratio}, "
            f"buffer={self.config.buffer_size}, "
            f"running={self.is_running})"
        )
