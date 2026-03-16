"""
core/base_trainer.py
--------------------
交叉验证训练器 CrossValidatorTrainer。

职责：
  - 编排完整的 CV 训练流程（fold 切分 → 训练 → OOF 预测 → 评估 → 日志）
  - 保存每折模型，支持后续 Stacking/Ensemble
  - 汇总平均特征重要性
  - 生成测试集预测（对每折推理后取均值）

关键设计：
  - 与 BaseModel / BaseCVSplitter 解耦：通过接口而非具体类型依赖
  - 指标函数统一签名：metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float
  - 所有中间结果暴露为实例属性，方便 Notebook 内省
  - 线程安全日志：每个 Trainer 实例使用独立 Logger（含 name 隔离）

使用示例
--------
>>> from core.base_trainer import CrossValidatorTrainer, MetricConfig
>>> from data.splitter import get_cv, CVConfig
>>> from sklearn.metrics import roc_auc_score

>>> cv = get_cv(CVConfig(strategy="stratified_kfold", n_splits=5))
>>> trainer = CrossValidatorTrainer(
...     model=lgbm_model,
...     cv=cv,
...     metrics=[MetricConfig("auc", roc_auc_score, higher_is_better=True)],
...     task="binary",
... )
>>> result = trainer.fit(X_train, y_train, X_test=X_test)
>>> print(result.cv_scores)
>>> print(result.oof_score)
"""

from __future__ import annotations

import gc
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from core.base_model import BaseModel
from data.splitter import BaseCVSplitter
from utils.memory import get_process_memory_mb

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

def _build_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# 指标配置
# ---------------------------------------------------------------------------

@dataclass
class MetricConfig:
    """
    封装一个评估指标。

    Parameters
    ----------
    name : str
        指标名称，用于日志和结果字典的键。
    fn : Callable
        签名：fn(y_true, y_pred) -> float。
    higher_is_better : bool
        True 表示越高越好（AUC/F1），False 表示越低越好（RMSE/MAE）。
    primary : bool
        是否为主指标（用于 early stopping 和最佳折判断）。
        同时只允许一个 MetricConfig.primary=True。
    use_proba : bool
        True 时传入 predict_proba 的输出，False 时传入 predict 的输出。
        AUC 等需要概率的指标应设为 True。
    """
    name:             str
    fn:               Callable[[np.ndarray, np.ndarray], float]
    higher_is_better: bool  = True
    primary:          bool  = True
    use_proba:        bool  = False


# ---------------------------------------------------------------------------
# 训练结果
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """
    CrossValidatorTrainer.fit() 的返回值，包含所有训练产物。

    Attributes
    ----------
    oof_predictions : np.ndarray
        形状 (n_train_samples,)，训练集的 OOF 预测值。
    test_predictions : np.ndarray | None
        形状 (n_test_samples,)，测试集各折预测值的均值。
        未传入 X_test 时为 None。
    fold_models : list[BaseModel]
        每折训练好的模型实例。
    cv_scores : dict[str, list[float]]
        {metric_name: [fold_0_score, fold_1_score, ...]}。
    oof_score : dict[str, float]
        {metric_name: 全量 OOF 计算的得分}（比 cv_scores 均值更准确）。
    feature_importance : pd.DataFrame
        列：["feature", "importance_mean", "importance_std"]，按均值降序。
    fold_times : list[float]
        每折训练耗时（秒）。
    fold_memory_deltas : list[float]
        每折内存增量（MB）。
    n_folds : int
        实际完成的折数。
    """
    oof_predictions:    np.ndarray
    test_predictions:   Optional[np.ndarray]
    fold_models:        list
    cv_scores:          dict[str, list[float]]
    oof_score:          dict[str, float]
    feature_importance: pd.DataFrame
    fold_times:         list[float]
    fold_memory_deltas: list[float]
    n_folds:            int

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    def primary_cv_mean(self, metric_name: str) -> float:
        scores = self.cv_scores.get(metric_name, [])
        return float(np.mean(scores)) if scores else float("nan")

    def primary_cv_std(self, metric_name: str) -> float:
        scores = self.cv_scores.get(metric_name, [])
        return float(np.std(scores)) if scores else float("nan")

    def summary(self) -> str:
        lines = [
            "=" * 62,
            f"  CV 结果摘要（{self.n_folds} 折）",
            "=" * 62,
        ]
        for name, scores in self.cv_scores.items():
            mean, std = np.mean(scores), np.std(scores)
            oof_s = self.oof_score.get(name, float("nan"))
            lines.append(
                f"  {name:<20} CV={mean:.6f} ± {std:.6f}   OOF={oof_s:.6f}"
            )
        lines += [
            f"  平均耗时   : {np.mean(self.fold_times):.2f} s/fold",
            f"  总耗时     : {sum(self.fold_times):.2f} s",
            f"  平均内存增量: {np.mean(self.fold_memory_deltas):.1f} MB/fold",
            "=" * 62,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主训练器
# ---------------------------------------------------------------------------

class CrossValidatorTrainer:
    """
    通用交叉验证训练器。

    Parameters
    ----------
    model : BaseModel
        未训练的模型实例（每折自动 clone）。
    cv : BaseCVSplitter
        切分器实例，需实现 split(X, y, groups) -> list[(tr_idx, val_idx)]。
    metrics : list[MetricConfig]
        评估指标列表，至少一个 primary=True。
    task : str
        任务类型：'binary' | 'multiclass' | 'regression'。
        影响默认预测方法（predict_proba vs predict）。
    groups : pd.Series | None
        样本 group 标识（GroupKFold/时序切分时使用）。
    save_dir : str | Path | None
        模型保存目录。None 表示不落盘。
    verbose : bool
        是否在控制台打印进度日志。
    log_level : int
        logging 级别，默认 INFO。
    """

    def __init__(
        self,
        model:     BaseModel,
        cv:        BaseCVSplitter,
        metrics:   list[MetricConfig],
        task:      str = "regression",
        groups:    Optional[pd.Series] = None,
        save_dir:  Optional[Union[str, Path]] = None,
        verbose:   bool = True,
        log_level: int  = logging.INFO,
    ):
        self._validate_metrics(metrics)

        self.model    = model
        self.cv       = cv
        self.metrics  = metrics
        self.task     = task.lower()
        self.groups   = groups
        self.save_dir = Path(save_dir) if save_dir else None
        self.verbose  = verbose

        self._logger  = _build_logger(
            f"trainer.{model.name}", level=log_level
        )

        # fit() 后填充
        self._result: Optional[CVResult] = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        fit_kwargs: Optional[dict] = None,
    ) -> CVResult:
        """
        执行完整 CV 训练流程。

        Parameters
        ----------
        X : pd.DataFrame
            训练特征（行顺序须与 y 对齐）。
        y : pd.Series
            训练标签。
        X_test : pd.DataFrame | None
            测试集特征；若提供则每折推理后平均。
        fit_kwargs : dict | None
            透传给每折 model.fit() 的额外参数（如 callbacks）。

        Returns
        -------
        CVResult
        """
        fit_kwargs   = fit_kwargs or {}
        folds        = self.cv.split(X, y, groups=self.groups)
        n_folds      = len(folds)
        primary_cfg  = self._primary_metric()

        self._log(f"开始 CV 训练 | 模型={self.model.name} | "
                  f"折数={n_folds} | 任务={self.task}")
        self._log(f"切分器={self.cv!r}")
        self._log(f"数据形状: X={X.shape}, y={y.shape}")

        # 初始化收集容器
        oof_preds    = np.full(len(X), np.nan, dtype=np.float64)
        test_preds   = (
            np.zeros((n_folds, len(X_test)), dtype=np.float64)
            if X_test is not None else None
        )
        fold_models: list[BaseModel] = []
        cv_scores:   dict[str, list[float]] = {m.name: [] for m in self.metrics}
        fold_times:  list[float]  = []
        fold_mem_deltas: list[float] = []
        importance_frames: list[pd.Series] = []

        # ------ 折循环 ------
        for fold_i, (tr_idx, val_idx) in enumerate(folds):
            fold_label = f"Fold {fold_i + 1}/{n_folds}"
            self._log(f"{'─' * 55}")
            self._log(f"{fold_label} | train={len(tr_idx):,}  val={len(val_idx):,}")

            X_tr,  y_tr  = X.iloc[tr_idx],  y.iloc[tr_idx]
            X_val, y_val = X.iloc[val_idx],  y.iloc[val_idx]

            mem_before = get_process_memory_mb()
            t_fold_start = time.perf_counter()

            # ---- 训练 ----
            fold_model = self.model.clone()
            fold_model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, **fit_kwargs)
            fold_time = time.perf_counter() - t_fold_start
            mem_after  = get_process_memory_mb()
            mem_delta  = mem_after - mem_before

            fold_times.append(fold_time)
            fold_mem_deltas.append(mem_delta)

            # ---- OOF 预测 ----
            val_pred = self._predict(fold_model, X_val)
            oof_preds[val_idx] = val_pred

            # ---- 评估 ----
            fold_score_strs = []
            for metric_cfg in self.metrics:
                pred_for_metric = (
                    self._predict_proba(fold_model, X_val)
                    if metric_cfg.use_proba else val_pred
                )
                score = metric_cfg.fn(y_val.values, pred_for_metric)
                cv_scores[metric_cfg.name].append(score)
                fold_score_strs.append(f"{metric_cfg.name}={score:.6f}")

            self._log(
                f"{fold_label} | "
                + " | ".join(fold_score_strs)
                + f" | 耗时={fold_time:.2f}s | 内存Δ={mem_delta:+.1f}MB"
            )

            # ---- 测试集预测 ----
            if test_preds is not None:
                test_preds[fold_i] = self._predict(fold_model, X_test)

            # ---- 特征重要性 ----
            fi = fold_model.feature_importance
            if len(fi) > 0:
                importance_frames.append(fi)

            # ---- 保存模型 ----
            fold_models.append(fold_model)
            if self.save_dir:
                self._save_fold_model(fold_model, fold_i)

            # 显式 GC 释放 train/val 子集（大数据场景）
            del X_tr, y_tr, X_val, y_val
            gc.collect()

        # ------ 全局 OOF 评估 ------
        oof_score: dict[str, float] = {}
        oof_valid_mask = ~np.isnan(oof_preds)
        for metric_cfg in self.metrics:
            oof_pred_for_metric = oof_preds[oof_valid_mask]
            oof_true = y.values[oof_valid_mask]
            score = metric_cfg.fn(oof_true, oof_pred_for_metric)
            oof_score[metric_cfg.name] = score

        # ------ 汇总特征重要性 ------
        feat_importance_df = self._aggregate_importance(importance_frames, X.columns)

        # ------ 测试集均值预测 ------
        final_test_preds = (
            test_preds.mean(axis=0) if test_preds is not None else None
        )

        # ------ 构建结果 ------
        result = CVResult(
            oof_predictions    = oof_preds,
            test_predictions   = final_test_preds,
            fold_models        = fold_models,
            cv_scores          = cv_scores,
            oof_score          = oof_score,
            feature_importance = feat_importance_df,
            fold_times         = fold_times,
            fold_memory_deltas = fold_mem_deltas,
            n_folds            = n_folds,
        )
        self._result = result

        # ------ 最终汇总日志 ------
        self._log("\n" + result.summary())
        primary_scores = cv_scores[primary_cfg.name]
        direction      = "↑" if primary_cfg.higher_is_better else "↓"
        self._log(
            f"主指标 [{primary_cfg.name}] {direction} "
            f"CV={np.mean(primary_scores):.6f} ± {np.std(primary_scores):.6f} | "
            f"OOF={oof_score[primary_cfg.name]:.6f}"
        )

        return result

    # ------------------------------------------------------------------
    # 推理辅助
    # ------------------------------------------------------------------

    def _predict(self, model: BaseModel, X: pd.DataFrame) -> np.ndarray:
        """
        根据任务类型选择 predict / predict_proba 的正确输出。
        - regression   → predict()
        - binary       → predict_proba()[:, 1] 或标量输出
        - multiclass   → predict_proba()（返回矩阵时 OOF 存为 argmax）
        """
        if self.task == "regression":
            return model.predict(X)
        elif self.task == "binary":
            proba = model.predict_proba(X)
            # 兼容返回 (n,) 或 (n, 2) 两种形式
            if proba.ndim == 2:
                return proba[:, 1]
            return proba
        elif self.task == "multiclass":
            proba = model.predict_proba(X)
            # OOF 存 argmax，原始 proba 留给指标函数另取
            return np.argmax(proba, axis=1) if proba.ndim == 2 else proba
        else:
            raise ValueError(
                f"未知 task='{self.task}'，请使用 regression/binary/multiclass。"
            )

    def _predict_proba(self, model: BaseModel, X: pd.DataFrame) -> np.ndarray:
        """为 use_proba=True 的指标提供完整概率输出。"""
        return model.predict_proba(X)

    # ------------------------------------------------------------------
    # 特征重要性汇总
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_importance(
        frames: list[pd.Series],
        feature_names,
    ) -> pd.DataFrame:
        """
        将多折的 feature_importance Series 合并为带均值/标准差的 DataFrame。
        """
        if not frames:
            return pd.DataFrame(
                columns=["feature", "importance_mean", "importance_std"]
            )

        df = pd.DataFrame(frames).T  # shape: (n_features, n_folds)
        df.index.name = "feature"

        # 归一化每折（sum=1），再计算统计
        normalized = df.div(df.sum(axis=0).replace(0, 1), axis=1)
        result = pd.DataFrame({
            "feature":          normalized.index,
            "importance_mean":  normalized.mean(axis=1).values,
            "importance_std":   normalized.std(axis=1).values,
        })
        return result.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 模型保存
    # ------------------------------------------------------------------

    def _save_fold_model(self, model: BaseModel, fold_i: int) -> None:
        if self.save_dir is None:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"{model.name}_fold{fold_i}.pkl"
        try:
            model.save(path)
            self._log(f"模型已保存 → {path}", level=logging.DEBUG)
        except Exception as e:
            warnings.warn(f"模型保存失败（{path}）：{e}", RuntimeWarning)

    # ------------------------------------------------------------------
    # 验证 & 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_metrics(metrics: list[MetricConfig]) -> None:
        if not metrics:
            raise ValueError("metrics 列表不能为空，至少需要一个 MetricConfig。")
        primary_count = sum(1 for m in metrics if m.primary)
        if primary_count == 0:
            # 自动将第一个设为 primary
            metrics[0] = MetricConfig(
                name             = metrics[0].name,
                fn               = metrics[0].fn,
                higher_is_better = metrics[0].higher_is_better,
                primary          = True,
                use_proba        = metrics[0].use_proba,
            )
        elif primary_count > 1:
            raise ValueError(
                f"只能有一个 primary=True 的 MetricConfig，"
                f"当前有 {primary_count} 个。"
            )

    def _primary_metric(self) -> MetricConfig:
        for m in self.metrics:
            if m.primary:
                return m
        return self.metrics[0]

    def _log(self, msg: str, level: int = logging.INFO) -> None:
        if self.verbose:
            self._logger.log(level, msg)

    # ------------------------------------------------------------------
    # 结果访问
    # ------------------------------------------------------------------

    @property
    def result(self) -> CVResult:
        if self._result is None:
            raise RuntimeError("请先调用 fit() 后再访问 result。")
        return self._result

    @property
    def oof_predictions(self) -> np.ndarray:
        return self.result.oof_predictions

    @property
    def feature_importance(self) -> pd.DataFrame:
        return self.result.feature_importance

    def __repr__(self) -> str:
        fitted = self._result is not None
        return (
            f"CrossValidatorTrainer("
            f"model={self.model.name!r}, "
            f"cv={self.cv!r}, "
            f"task={self.task!r}, "
            f"fitted={fitted})"
        )
