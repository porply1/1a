"""
data/splitter.py
----------------
交叉验证切分策略工厂。

设计目标：
  1. 工厂模式：一个 get_cv() 入口，按名称返回对应切分器
  2. 覆盖竞赛三大赛道：
       - 结构化数据  → KFold / StratifiedKFold / GroupKFold / StratifiedGroupKFold
       - 时间序列    → TimeSeriesSplit / PurgedGroupTimeSeriesSplit（防泄露）
       - 推荐系统    → UserGroupKFold（按用户切分，防泄露）
  3. 所有切分器统一返回 List[Tuple[np.ndarray, np.ndarray]] 格式（eager 求值）
  4. 提供 validate_no_leakage() 辅助函数，检查 train/val 是否有 group 泄露

使用示例
--------
>>> from data.splitter import get_cv, CVConfig

>>> # 结构化分类任务
>>> cv = get_cv(CVConfig(strategy="stratified_kfold", n_splits=5))
>>> folds = cv.split(X, y)

>>> # 时序任务（带 gap）
>>> cv = get_cv(CVConfig(strategy="purged_timeseries", n_splits=5, gap=7))
>>> folds = cv.split(X, y, groups=time_index)

>>> # 推荐系统（按用户）
>>> cv = get_cv(CVConfig(strategy="user_group_kfold", n_splits=5))
>>> folds = cv.split(X, y, groups=user_ids)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    StratifiedGroupKFold,
)

# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------

@dataclass
class CVConfig:
    """
    交叉验证配置，传入 get_cv() 工厂。

    Parameters
    ----------
    strategy : str
        切分策略名称，见 STRATEGY_REGISTRY。
    n_splits : int
        折数，默认 5。
    shuffle : bool
        KFold / StratifiedKFold 是否打乱，默认 True。
    random_state : int
        随机种子，默认 42。
    gap : int
        时序切分中 train/val 之间的间隔样本数（防近期数据泄露），默认 0。
    max_train_size : int | None
        时序切分中训练集最大样本数（滑动窗口），None 表示不限。
    purge_pct : float
        PurgedGroupTimeSeriesSplit 中，按 group 时间排序后，
        从 val 前沿向前清除的比例，默认 0.01（1%）。
    """
    strategy: str = "stratified_kfold"
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    gap: int = 0
    max_train_size: Optional[int] = None
    purge_pct: float = 0.01
    # ---- TimeSeriesSplitter 专用 ----
    mode: str = "expanding"           # "expanding" | "sliding"
    test_size: Optional[int] = None   # 验证集固定大小；None 时自动均分
    min_train_size: Optional[int] = None  # 最小训练集大小；不足时跳过该折


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseCVSplitter:
    """所有切分器的统一接口。"""

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Returns
        -------
        list of (train_idx, val_idx) numpy arrays
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# 策略实现
# ---------------------------------------------------------------------------

class KFoldSplitter(BaseCVSplitter):
    """标准 K 折，适用于回归任务或类别均衡的分类任务。"""

    def __init__(self, cfg: CVConfig):
        self._kf = KFold(
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state if cfg.shuffle else None,
        )

    def split(self, X, y=None, groups=None):
        return list(self._kf.split(X, y))

    def __repr__(self):
        return f"KFoldSplitter(n_splits={self._kf.n_splits})"


class StratifiedKFoldSplitter(BaseCVSplitter):
    """分层 K 折，保证每折目标分布一致，适用于类别不均衡分类。"""

    def __init__(self, cfg: CVConfig):
        self._skf = StratifiedKFold(
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state if cfg.shuffle else None,
        )

    def split(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("StratifiedKFoldSplitter 需要传入 y（目标列）")
        return list(self._skf.split(X, y))

    def __repr__(self):
        return f"StratifiedKFoldSplitter(n_splits={self._skf.n_splits})"


class GroupKFoldSplitter(BaseCVSplitter):
    """
    按 group 划分，同一 group 不会同时出现在 train 和 val 中。
    适用场景：同一用户/同一店铺的数据需要防泄露。
    """

    def __init__(self, cfg: CVConfig):
        self._gkf = GroupKFold(n_splits=cfg.n_splits)

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("GroupKFoldSplitter 需要传入 groups")
        return list(self._gkf.split(X, y, groups=groups))

    def __repr__(self):
        return f"GroupKFoldSplitter(n_splits={self._gkf.n_splits})"


class StratifiedGroupKFoldSplitter(BaseCVSplitter):
    """
    分层 + Group 双重约束，兼顾类别分布和 group 不泄露。
    sklearn >= 1.1 原生支持。
    """

    def __init__(self, cfg: CVConfig):
        self._sgkf = StratifiedGroupKFold(
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state if cfg.shuffle else None,
        )

    def split(self, X, y=None, groups=None):
        if y is None or groups is None:
            raise ValueError("StratifiedGroupKFoldSplitter 需要 y 和 groups")
        return list(self._sgkf.split(X, y, groups=groups))

    def __repr__(self):
        return f"StratifiedGroupKFoldSplitter(n_splits={self._sgkf.n_splits})"


class TimeSeriesSplitter(BaseCVSplitter):
    """
    冠军级时间序列切分器。

    同时支持 Expanding Window（扩张窗口）和 Sliding Window（滑动窗口）策略，
    内置 gap_size 防近期数据泄露，兼容 Pandas / Polars / NumPy。

    Parameters
    ----------
    cfg : CVConfig
        关键字段：
        n_splits       : 折数，默认 5。
        mode           : "expanding"（默认）或 "sliding"。
                         expanding — 每折训练集向历史累积（ExpandingWindow）。
                         sliding   — 训练集大小固定，窗口向前滑动（SlidingWindow），
                                     需配合 max_train_size 使用。
        gap            : 训练末尾到验证首端之间跳过的样本数（gap_size 防泄露），默认 0。
                         例：训练到位置 t，gap=7，则验证集从 t+8 开始。
        test_size      : 验证集固定大小；None 时自动均分为 n_samples//(n_splits+1)。
        min_train_size : 最小训练集样本数；不足时跳过该折并发出警告。
        max_train_size : 训练集最大样本数（sliding 模式的窗口宽度）。

    Notes
    -----
    - X 必须已按时间升序排列（splitter 本身不排序）。
    - split() 只依赖 len(X)，天然兼容 pandas.DataFrame / polars.DataFrame / np.ndarray。
    - 返回整数位置索引 np.ndarray，使用方式：X.iloc[train_idx] 或 X[train_idx]。

    Examples
    --------
    >>> # Expanding Window，gap=7 天防泄露
    >>> cfg = CVConfig(strategy="timeseries", n_splits=5, gap=7, mode="expanding")
    >>> ts = TimeSeriesSplitter(cfg)
    >>> for tr, va in ts.split(df):
    ...     assert tr.max() + 7 < va.min()

    >>> # Sliding Window，固定 200 个样本训练窗口
    >>> cfg = CVConfig(strategy="timeseries", n_splits=5, gap=0,
    ...                mode="sliding", max_train_size=200)
    >>> ts = TimeSeriesSplitter(cfg)
    """

    VALID_MODES = {"expanding", "sliding"}

    def __init__(self, cfg: CVConfig):
        if cfg.mode not in self.VALID_MODES:
            raise ValueError(
                f"mode 必须是 {self.VALID_MODES}，得到 '{cfg.mode}'"
            )
        if cfg.gap < 0:
            raise ValueError(f"gap_size 不能为负数，得到 {cfg.gap}")
        if cfg.mode == "sliding" and cfg.max_train_size is None:
            warnings.warn(
                "sliding 模式下未设置 max_train_size，行为退化为 expanding 模式。",
                UserWarning,
                stacklevel=2,
            )

        self.n_splits       = cfg.n_splits
        self.mode           = cfg.mode
        self.gap_size       = cfg.gap
        self.test_size      = cfg.test_size
        self.min_train_size = cfg.min_train_size
        self.max_train_size = cfg.max_train_size

    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        X : pd.DataFrame | pl.DataFrame | np.ndarray
            只使用 len(X)，不访问内容，天然兼容多种后端。
        y, groups : 忽略（保持接口统一性）

        Returns
        -------
        list of (train_idx, val_idx) — 整数位置索引 np.ndarray
        """
        n_samples = len(X)

        # 每折验证集大小（使用纯整数运算，避免浮点误差）
        test_sz = (
            self.test_size
            if self.test_size is not None
            else n_samples // (self.n_splits + 1)
        )
        if test_sz <= 0:
            raise ValueError(
                f"样本数 {n_samples} 不足以支撑 {self.n_splits} 折切分，"
                f"请减少 n_splits 或增大 test_size。"
            )

        folds: list[tuple[np.ndarray, np.ndarray]] = []

        for i in range(self.n_splits):
            # --- 验证集定位（从右往左，第 n_splits-1-i 个 test_sz 窗口） ---
            val_end   = n_samples - (self.n_splits - 1 - i) * test_sz
            val_start = val_end - test_sz

            # --- gap 隔离：训练集不超过 val_start - gap_size ---
            train_end = val_start - self.gap_size

            if train_end <= 0:
                warnings.warn(
                    f"第 {i} 折 train_end={train_end}≤0（gap_size={self.gap_size} 过大），"
                    "已跳过。",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            # --- 训练集起点（expanding vs sliding） ---
            if self.mode == "sliding" and self.max_train_size is not None:
                # 滑动窗口：固定宽度，窗口向前移动
                train_start = max(0, train_end - self.max_train_size)
            else:
                # 扩张窗口：从 0 开始累积
                train_start = 0

            # --- 最小训练集约束 ---
            actual_train_size = train_end - train_start
            if self.min_train_size and actual_train_size < self.min_train_size:
                warnings.warn(
                    f"第 {i} 折训练集大小 {actual_train_size} < "
                    f"min_train_size {self.min_train_size}，已跳过。",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            # --- 使用 np.arange 纯索引切片，零内存拷贝 ---
            train_idx = np.arange(train_start, train_end)
            val_idx   = np.arange(val_start,   val_end)
            folds.append((train_idx, val_idx))

        return folds

    def __repr__(self) -> str:
        return (
            f"TimeSeriesSplitter("
            f"n_splits={self.n_splits}, "
            f"mode={self.mode!r}, "
            f"gap_size={self.gap_size})"
        )


class PurgedGroupTimeSeriesSplitter(BaseCVSplitter):
    """
    冠军级时序 Group 切分器，带 Purge + Gap 双重防泄露机制。

    适用场景
    --------
    - 多城市、多股票等带实体 group 属性的时间序列
    - 同一实体 group 的时间跨度可能与 val 窗口发生"时间重叠"
      （即某只股票的训练数据和验证期高度相邻，导致 entity-level leakage）

    核心防泄露逻辑（参考 Lopez de Prado, AFML Ch.7）
    -----------------------------------------------
    1. 以时间戳将样本均分为 n_splits+1 段，第 i+1 段为 val；
    2. 计算 purge_boundary = min(val_timestamps) - gap_size；
    3. 对每个实体 group g：
         若 max(g 的所有时间戳) >= purge_boundary → 整组从训练集剔除；
    4. 此机制确保即使同一 group 的数据横跨 gap 区间，也会被完整清除。

    split() 签名
    ------------
    split(X, y=None, groups=entity_labels, timestamps=time_values)

    - groups     : 实体标签（城市名 / 股票代码 / 用户 ID 等），shape (n,)
    - timestamps : 每个样本的时间值（int/float/可排序类型），shape (n,)
                   若为 None，则回退到"以 groups 本身作为时间轴"的兼容模式。

    兼容模式（timestamps=None）
    --------------------------
    与原接口保持一致：groups 既是实体也是时间排序键。
    每个唯一 group 视为一个独立时间槽，按 group 值排序后切分。
    purge_boundary = 距 val_start 左边 gap_size 个 group 处。

    Parameters
    ----------
    cfg : CVConfig
        n_splits  : 折数
        gap       : gap_size（时间戳单位），默认 0
        purge_pct : 兼容模式下 purge 比例，默认 0.01（新模式不使用此参数）

    References
    ----------
    Marcos Lopez de Prado, "Advances in Financial Machine Learning", Ch.7
    """

    def __init__(self, cfg: CVConfig):
        self.n_splits  = cfg.n_splits
        self.gap_size  = cfg.gap
        self.purge_pct = cfg.purge_pct

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def split(
        self,
        X,
        y=None,
        groups=None,
        timestamps=None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        X          : array-like，仅用 len(X)，兼容 pandas/polars/numpy
        groups     : 实体标签，shape (n,)，不能为 None
        timestamps : 时间戳，shape (n,)；None 时启用兼容模式
        """
        if groups is None:
            raise ValueError(
                "PurgedGroupTimeSeriesSplitter.split() 需要 groups 参数。\n"
                "  新模式：groups=实体标签（股票/城市），timestamps=时间值\n"
                "  兼容模式：groups=时间 group id（原接口）"
            )

        groups_arr = np.asarray(groups)

        if timestamps is not None:
            # ---- 新模式：实体 group + 独立时间戳，支持时间重叠 purge ----
            ts = np.asarray(timestamps, dtype=float)
            if len(ts) != len(groups_arr):
                raise ValueError(
                    f"timestamps 长度 ({len(ts)}) 与 groups 长度 ({len(groups_arr)}) 不一致"
                )
            return self._split_entity_mode(groups_arr, ts)
        else:
            # ---- 兼容模式：groups 即时间 group（原行为，略有改进） ----
            return self._split_legacy_mode(groups_arr)

    # ------------------------------------------------------------------
    # 新模式：实体 group + 时间戳
    # ------------------------------------------------------------------

    def _split_entity_mode(
        self,
        groups_arr: np.ndarray,
        ts: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        行级 purge：以时间戳轴切分，移除所有 ts >= purge_boundary 的训练样本。

        设计原则
        --------
        金融 / 多城市场景中，同一实体（如某只股票）在整个时间轴上均有数据，
        不能因"实体出现在 val 期"就整组剔除——那会清空所有训练数据。
        正确做法是：
          - 训练集 = 所有 ts < purge_boundary 的行（跨实体）
          - 验证集 = 所有 ts 在 val 桶内的行（跨实体）
          - 同一实体可同时出现在 train（早期）和 val（晚期），这是正常的时序切分
          - gap_size 保证训练集最晚时间戳 < val 开始时间 - gap（防止近边界标签泄露）

        时间重叠处理
        ------------
        当不同实体的时间范围不连续 / 存在重叠时，本方法以 **唯一时间戳** 为轴
        均匀划分 val 桶，不依赖行号顺序，天然处理多实体时间重叠问题。
        """
        sorted_unique_ts = np.unique(ts)
        n_unique_ts = len(sorted_unique_ts)

        if n_unique_ts < self.n_splits + 2:
            raise ValueError(
                f"唯一时间戳数量 ({n_unique_ts}) 不足以切 {self.n_splits} 折，"
                f"至少需要 {self.n_splits + 2} 个唯一时间戳。"
            )

        # 将时间轴均分为 n_splits+1 桶（按时间值，非行号）
        ts_buckets = np.array_split(sorted_unique_ts, self.n_splits + 1)

        folds: list[tuple[np.ndarray, np.ndarray]] = []

        for fold_idx in range(self.n_splits):
            val_bucket     = ts_buckets[fold_idx + 1]
            val_start_time = float(val_bucket.min())
            # purge_boundary：训练集样本的时间上界（严格小于）
            purge_boundary = val_start_time - self.gap_size

            # 行级 purge：保留 ts < purge_boundary 的所有行（所有实体均可参与）
            train_idx = np.where(ts < purge_boundary)[0]
            # 验证集：时间戳落在 val 桶内（所有实体）
            val_idx   = np.where(np.isin(ts, val_bucket))[0]

            if len(train_idx) == 0 or len(val_idx) == 0:
                warnings.warn(
                    f"第 {fold_idx} 折 train/val 为空"
                    f"（gap_size={self.gap_size} 可能过大），已跳过。",
                    RuntimeWarning,
                    stacklevel=3,
                )
                continue

            folds.append((train_idx, val_idx))

        return folds

    # ------------------------------------------------------------------
    # 兼容模式：groups 即时间 group（原接口）
    # ------------------------------------------------------------------

    def _split_legacy_mode(
        self,
        groups_arr: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        原接口兼容实现：groups = 时间 group id，值越大越晚。

        purge_boundary 定义：距 val 起始 group 向左 max(gap_size, purge_n) 处。
        """
        unique_groups = np.unique(groups_arr)
        n_groups = len(unique_groups)

        if n_groups < self.n_splits + 2:
            raise ValueError(
                f"group 数量（{n_groups}）不足以切 {self.n_splits} 折，"
                f"至少需要 {self.n_splits + 2} 个 group。"
            )

        # purge 区间大小：取 purge_pct 与 gap_size 中的较大值（至少 1）
        purge_n = max(1, int(n_groups * self.purge_pct), self.gap_size)

        # 将 unique_groups 均分为 n_splits+1 段
        group_splits = np.array_split(unique_groups, self.n_splits + 1)

        folds: list[tuple[np.ndarray, np.ndarray]] = []

        for fold_idx in range(self.n_splits):
            val_groups      = set(group_splits[fold_idx + 1].tolist())
            train_groups_all = np.concatenate(group_splits[: fold_idx + 1])
            sorted_train    = np.sort(train_groups_all)
            val_min_group   = min(val_groups)

            # 紧靠 val 左边的 purge_n 个 group
            boundary = sorted_train[sorted_train < val_min_group]
            excluded: set = set(boundary[-purge_n:].tolist()) if len(boundary) > 0 else set()

            final_train_groups = set(train_groups_all.tolist()) - excluded
            train_idx = np.where(np.isin(groups_arr, list(final_train_groups)))[0]
            val_idx   = np.where(np.isin(groups_arr, list(val_groups)))[0]

            if len(train_idx) == 0 or len(val_idx) == 0:
                warnings.warn(
                    f"第 {fold_idx} 折 train/val 为空，已跳过。"
                    "请检查 n_splits 与 purge_pct/gap_size 设置。",
                    RuntimeWarning,
                    stacklevel=3,
                )
                continue

            folds.append((train_idx, val_idx))

        return folds

    def __repr__(self) -> str:
        return (
            f"PurgedGroupTimeSeriesSplitter("
            f"n_splits={self.n_splits}, "
            f"gap_size={self.gap_size})"
        )


class UserGroupKFoldSplitter(BaseCVSplitter):
    """
    推荐系统专用：按用户 ID 分折，同一用户的交互数据不会跨 train/val。

    本质上是 GroupKFold 的语义化封装，增加了用户分布均衡检查。
    """

    def __init__(self, cfg: CVConfig):
        self._gkf = GroupKFold(n_splits=cfg.n_splits)
        self.n_splits = cfg.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups: Optional[pd.Series] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("UserGroupKFoldSplitter 需要 groups（用户 ID 列）")

        n_users = len(np.unique(groups))
        if n_users < self.n_splits:
            raise ValueError(
                f"用户数（{n_users}）少于折数（{self.n_splits}），"
                "请减少 n_splits。"
            )

        folds = list(self._gkf.split(X, y, groups=groups))
        # 简单校验：每折 val 用户数应相近
        val_sizes = [len(np.unique(groups.iloc[val] if hasattr(groups, 'iloc') else groups[val]))
                     for _, val in folds]
        cv_ratio = np.std(val_sizes) / max(np.mean(val_sizes), 1)
        if cv_ratio > 0.3:
            warnings.warn(
                f"各折用户分布不均（变异系数={cv_ratio:.2f}），"
                "建议检查用户交互数据是否存在长尾分布。",
                RuntimeWarning,
                stacklevel=2,
            )
        return folds

    def __repr__(self):
        return f"UserGroupKFoldSplitter(n_splits={self._gkf.n_splits})"


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[BaseCVSplitter]] = {
    "kfold":                      KFoldSplitter,
    "stratified_kfold":           StratifiedKFoldSplitter,
    "group_kfold":                GroupKFoldSplitter,
    "stratified_group_kfold":     StratifiedGroupKFoldSplitter,
    "timeseries":                 TimeSeriesSplitter,
    "purged_timeseries":          PurgedGroupTimeSeriesSplitter,
    "user_group_kfold":           UserGroupKFoldSplitter,
}


def get_cv(cfg: CVConfig) -> BaseCVSplitter:
    """
    工厂函数：根据 CVConfig 返回对应切分器实例。

    Parameters
    ----------
    cfg : CVConfig

    Returns
    -------
    BaseCVSplitter

    Raises
    ------
    ValueError
        当 strategy 不在注册表中时。

    Examples
    --------
    >>> cv = get_cv(CVConfig(strategy="stratified_kfold", n_splits=5))
    >>> folds = cv.split(X_train, y_train)
    >>> for fold_i, (tr_idx, val_idx) in enumerate(folds):
    ...     X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    """
    strategy = cfg.strategy.lower()
    if strategy not in STRATEGY_REGISTRY:
        valid = list(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"未知的 CV 策略：'{strategy}'。\n"
            f"可用策略：{valid}"
        )
    return STRATEGY_REGISTRY[strategy](cfg)


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def validate_no_leakage(
    folds: list[tuple[np.ndarray, np.ndarray]],
    groups: pd.Series,
) -> bool:
    """
    检查每一折的 train/val 之间是否存在 group 泄露。

    Parameters
    ----------
    folds : list of (train_idx, val_idx)
    groups : pd.Series
        与原始数据行对齐的 group 标识列。

    Returns
    -------
    bool
        True 表示无泄露，False 表示存在泄露（同时打印详情）。
    """
    groups_arr = np.asarray(groups)
    has_leakage = False

    for i, (tr_idx, val_idx) in enumerate(folds):
        train_groups = set(groups_arr[tr_idx])
        val_groups   = set(groups_arr[val_idx])
        leaked = train_groups & val_groups
        if leaked:
            has_leakage = True
            print(
                f"[leakage] 第 {i} 折发现泄露！"
                f"共 {len(leaked)} 个 group 同时出现在 train 和 val 中。"
                f"示例：{list(leaked)[:5]}"
            )

    if not has_leakage:
        print(f"[leakage] 所有 {len(folds)} 折均无 group 泄露。")
    return not has_leakage


def fold_statistics(
    folds: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
) -> pd.DataFrame:
    """
    打印每折的样本数和（分类任务下的）目标分布，便于验证分层效果。

    Parameters
    ----------
    folds : list of (train_idx, val_idx)
    y : pd.Series
        目标列。

    Returns
    -------
    pd.DataFrame
        每折统计摘要。
    """
    y_arr = np.asarray(y)
    is_clf = len(np.unique(y_arr)) <= 50  # 粗略判断是否为分类

    rows = []
    for i, (tr_idx, val_idx) in enumerate(folds):
        row: dict = {
            "fold":      i,
            "train_n":   len(tr_idx),
            "val_n":     len(val_idx),
        }
        if is_clf:
            val_pos_rate = y_arr[val_idx].mean()
            tr_pos_rate  = y_arr[tr_idx].mean()
            row["train_pos%"] = round(tr_pos_rate * 100, 2)
            row["val_pos%"]   = round(val_pos_rate * 100, 2)
        rows.append(row)

    return pd.DataFrame(rows).set_index("fold")
