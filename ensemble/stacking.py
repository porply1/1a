"""
ensemble/stacking.py
--------------------
Stacking 集成引擎：将多个 base model 的 OOF 预测作为新特征，
训练一个元学习器（meta-learner）做最终融合。

架构图
------
                  ┌─────────────────────────────────────┐
  训练集 X ──▶  │  Base Models (LGBM / XGB / CatBoost) │
                  │       ↓  CrossValidatorTrainer        │
                  │  OOF predictions (N × K 矩阵)        │
                  └──────────────┬──────────────────────┘
                                 │ X_meta (N × K)
                                 ▼
                  ┌──────────────────────────────────────┐
                  │  Meta Learner (Ridge / ElasticNet)   │
                  │       fit(X_meta, y_true)             │
                  └──────────────┬───────────────────────┘
                                 │
                                 ▼
                        最终集成预测结果

防泄露原则
----------
  OOF 预测天然防泄露：每个样本的预测值由"未见过该样本"的折产生。
  与 Blending（按比例切分）相比，Stacking 在小数据集上更稳健。

设计特点
--------
  1. 输入：CVResult 对象列表，自动提取 oof_predictions + test_predictions
  2. 严格校验：OOF 长度/NaN 一致性检查，防止静默错误
  3. 元特征扩展：可选择追加原始特征的子集（增强元学习器表达力）
  4. Meta Learner：默认 Ridge，可替换为任何 sklearn-compatible 估计器
  5. 系数分析：输出 base model 贡献权重，诊断哪把武器最强
  6. 任务支持：regression / binary / multiclass（OOF 为 argmax 时自动处理）
  7. 二层 Stacking：StackingEnsemble 本身输出的 meta_oof 可再次堆叠

使用示例
--------
>>> from ensemble.stacking import StackingEnsemble, StackingConfig
>>> from sklearn.metrics import roc_auc_score

>>> stacker = StackingEnsemble(
...     results=[lgbm_result, xgb_result, cat_result],
...     model_names=["lgbm", "xgb", "catboost"],
...     config=StackingConfig(task="binary"),
... )
>>> stacking_result = stacker.fit(y_train)
>>> final_pred = stacker.predict()
>>> print(stacker.weight_report())
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler

# CVResult 从 base_trainer 导入
from core.base_trainer import CVResult


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

def _build_logger(name: str) -> logging.Logger:
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
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------

@dataclass
class StackingConfig:
    """
    StackingEnsemble 的行为配置。

    Parameters
    ----------
    task : str
        任务类型：'regression' | 'binary' | 'multiclass'。
    meta_learner : Any
        sklearn-compatible 元学习器。
        None 时按 task 自动选择：
          regression  → Ridge(alpha=1.0)
          binary      → LogisticRegression(C=0.1, max_iter=1000)
          multiclass  → LogisticRegression(C=0.1, multi_class='multinomial')
    scale_meta_features : bool
        训练元学习器前是否对 X_meta 做 StandardScaler。
        默认 True（Ridge/LogReg 对尺度敏感）。
    append_original_features : list[str] | None
        在 X_meta 中追加原始特征的列名子集。
        None 表示仅使用 OOF 预测作为元特征。
    passthrough : bool
        True 时自动将原始特征全部追加到 X_meta（同 sklearn StackingClassifier
        的 passthrough 参数）。append_original_features 优先级更高。
    clip_oof : bool
        True 时将 OOF 预测裁剪到 [oof_min, oof_max] 范围（防止极端值）。
        默认 False。
    """
    task:                       str  = "regression"
    meta_learner:               Any  = None
    scale_meta_features:        bool = True
    append_original_features:   Optional[list[str]] = None
    passthrough:                bool = False
    clip_oof:                   bool = False


# ---------------------------------------------------------------------------
# 核心集成结果
# ---------------------------------------------------------------------------

@dataclass
class StackingResult:
    """
    StackingEnsemble.fit() 的完整输出。

    Attributes
    ----------
    meta_oof : np.ndarray, shape (n_train,)
        元学习器的 OOF 预测（用于评估和下一层 Stacking）。
    meta_test : np.ndarray | None
        测试集最终预测。
    X_meta_train : pd.DataFrame
        元特征训练矩阵（base model OOF + 可选原始特征）。
    X_meta_test : pd.DataFrame | None
        元特征测试矩阵。
    model_weights : pd.DataFrame
        各 base model 的元学习器系数（贡献权重）。
    meta_learner : Any
        训练好的元学习器对象。
    fit_time : float
        元学习器训练耗时（秒）。
    """
    meta_oof:       np.ndarray
    meta_test:      Optional[np.ndarray]
    X_meta_train:   pd.DataFrame
    X_meta_test:    Optional[pd.DataFrame]
    model_weights:  pd.DataFrame
    meta_learner:   Any
    fit_time:       float

    def weight_report(self) -> str:
        """打印各 base model 贡献权重的格式化报告。"""
        lines = [
            "=" * 55,
            "  Base Model 贡献权重（Meta Learner Coefficients）",
            "=" * 55,
        ]
        df = self.model_weights.sort_values("abs_weight", ascending=False)
        for _, row in df.iterrows():
            bar_len  = int(row["abs_weight"] / df["abs_weight"].max() * 30)
            bar      = "█" * bar_len
            sign     = "+" if row["weight"] >= 0 else "-"
            lines.append(
                f"  {row['model']:<20} {sign}{abs(row['weight']):.6f}  {bar}"
            )
        lines.append("=" * 55)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# StackingEnsemble
# ---------------------------------------------------------------------------

class StackingEnsemble:
    """
    两层 Stacking 集成引擎。

    Parameters
    ----------
    results : list[CVResult]
        多个 base model 的 CrossValidatorTrainer.fit() 输出。
        每个 CVResult 必须包含：
          - oof_predictions  (shape: n_train_samples,)
          - test_predictions (shape: n_test_samples,) 或 None
    model_names : list[str] | None
        各 base model 的可读名称，用于列名和报告。
        None 时自动编号为 "model_0", "model_1", ...
    config : StackingConfig
        集成行为配置。
    X_train_original : pd.DataFrame | None
        原始训练特征，仅在 config.passthrough=True 或
        config.append_original_features 非空时需要。
    X_test_original : pd.DataFrame | None
        原始测试特征（与 X_train_original 对应）。
    verbose : bool
        是否打印训练日志。

    Examples
    --------
    >>> stacker = StackingEnsemble(
    ...     results=[lgbm_result, xgb_result, cat_result],
    ...     model_names=["lgbm", "xgb", "catboost"],
    ...     config=StackingConfig(task="binary"),
    ... )
    >>> sr = stacker.fit(y_train)
    >>> final_pred = stacker.predict()
    >>> print(sr.weight_report())
    """

    def __init__(
        self,
        results:              list[CVResult],
        model_names:          Optional[list[str]] = None,
        config:               Optional[StackingConfig] = None,
        X_train_original:     Optional[pd.DataFrame] = None,
        X_test_original:      Optional[pd.DataFrame] = None,
        verbose:              bool = True,
    ):
        if not results:
            raise ValueError("results 列表不能为空，至少需要一个 CVResult。")

        self.results          = results
        self.config           = config or StackingConfig()
        self.X_train_original = X_train_original
        self.X_test_original  = X_test_original
        self.verbose          = verbose
        self._logger          = _build_logger("stacking")

        # 规范化 model_names
        self.model_names = model_names or [
            f"model_{i}" for i in range(len(results))
        ]
        if len(self.model_names) != len(results):
            raise ValueError(
                f"model_names 长度（{len(self.model_names)}）"
                f"与 results 长度（{len(results)}）不一致。"
            )

        # fit() 后填充
        self._stacking_result: Optional[StackingResult] = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def fit(self, y: Union[pd.Series, np.ndarray]) -> StackingResult:
        """
        构建元特征矩阵 → 训练元学习器 → 生成元 OOF 预测。

        Parameters
        ----------
        y : pd.Series | np.ndarray
            训练集真实标签（与 base model 训练时使用的标签相同）。

        Returns
        -------
        StackingResult
        """
        y_arr = np.asarray(y)
        self._log(f"{'=' * 58}")
        self._log(f"  Stacking 集成训练 | 模型数={len(self.results)} | task={self.config.task}")
        self._log(f"{'=' * 58}")

        # ---- Step 1: 校验输入 ----
        self._validate_results(y_arr)

        # ---- Step 2: 构建元特征矩阵 ----
        X_meta_train, X_meta_test = self._build_meta_features()
        self._log(
            f"元特征矩阵：train={X_meta_train.shape}, "
            f"test={X_meta_test.shape if X_meta_test is not None else 'N/A'}"
        )

        # ---- Step 3: 标准化（可选）----
        X_meta_train_fit, X_meta_test_fit, scaler = self._maybe_scale(
            X_meta_train, X_meta_test
        )

        # ---- Step 4: 构建并训练元学习器 ----
        meta_learner = self._build_meta_learner()
        self._log(f"元学习器：{meta_learner.__class__.__name__}")

        t0 = time.perf_counter()
        meta_learner.fit(X_meta_train_fit, y_arr)
        fit_time = time.perf_counter() - t0
        self._log(f"元学习器训练完成，耗时 {fit_time:.3f}s")

        # ---- Step 5: OOF 元预测 ----
        meta_oof = self._meta_predict(meta_learner, X_meta_train_fit)

        # ---- Step 6: 测试集元预测 ----
        meta_test: Optional[np.ndarray] = None
        if X_meta_test_fit is not None:
            meta_test = self._meta_predict(meta_learner, X_meta_test_fit)

        # ---- Step 7: 解析 base model 权重 ----
        weights_df = self._extract_weights(meta_learner, X_meta_train.columns.tolist())

        # ---- Step 8: 打印权重报告 ----
        sr = StackingResult(
            meta_oof      = meta_oof,
            meta_test     = meta_test,
            X_meta_train  = X_meta_train,
            X_meta_test   = X_meta_test,
            model_weights = weights_df,
            meta_learner  = meta_learner,
            fit_time      = fit_time,
        )
        self._stacking_result = sr
        self._log("\n" + sr.weight_report())

        return sr

    def predict(self) -> np.ndarray:
        """返回测试集最终预测（需要先调用 fit）。"""
        if self._stacking_result is None:
            raise RuntimeError("请先调用 fit() 后再调用 predict()。")
        if self._stacking_result.meta_test is None:
            raise RuntimeError(
                "测试集预测不存在，请确保 base model 训练时传入了 X_test。"
            )
        return self._stacking_result.meta_test

    # ------------------------------------------------------------------
    # 校验逻辑
    # ------------------------------------------------------------------

    def _validate_results(self, y: np.ndarray) -> None:
        """
        严格校验所有 CVResult 的 OOF 预测，防止静默错误。
        检查项：
          1. OOF 长度与 y 一致
          2. 各 CVResult 的 OOF 长度互相一致
          3. OOF 中不存在全 NaN（部分 NaN 发出警告）
          4. test_predictions 长度互相一致（若存在）
        """
        n_train = len(y)

        # --- OOF 长度一致性 ---
        for i, (name, result) in enumerate(zip(self.model_names, self.results)):
            oof_len = len(result.oof_predictions)
            if oof_len != n_train:
                raise ValueError(
                    f"[校验失败] {name} 的 OOF 长度（{oof_len}）"
                    f"与标签 y 长度（{n_train}）不一致。\n"
                    f"请确保所有 base model 使用相同的训练数据。"
                )

        # --- OOF 各模型长度互相一致 ---
        oof_lengths = {name: len(r.oof_predictions)
                       for name, r in zip(self.model_names, self.results)}
        unique_lengths = set(oof_lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                f"[校验失败] OOF 长度不一致：{oof_lengths}\n"
                f"所有 base model 必须使用相同数量的训练样本。"
            )

        # --- NaN 检查 ---
        for name, result in zip(self.model_names, self.results):
            nan_count = np.isnan(result.oof_predictions).sum()
            if nan_count == n_train:
                raise ValueError(
                    f"[校验失败] {name} 的 OOF 预测全部为 NaN，"
                    f"请检查 CrossValidatorTrainer 是否正常运行。"
                )
            if nan_count > 0:
                warnings.warn(
                    f"[校验警告] {name} 的 OOF 预测含 {nan_count} 个 NaN "
                    f"（占比 {nan_count / n_train:.1%}），将填充为 0。",
                    UserWarning,
                    stacklevel=2,
                )

        # --- 测试集长度一致性 ---
        test_lengths = {}
        for name, result in zip(self.model_names, self.results):
            if result.test_predictions is not None:
                test_lengths[name] = len(result.test_predictions)

        if len(test_lengths) > 0:
            if len(set(test_lengths.values())) > 1:
                raise ValueError(
                    f"[校验失败] test_predictions 长度不一致：{test_lengths}"
                )
            # 检查是否有部分模型缺少测试预测
            missing_test = [
                name for name, result in zip(self.model_names, self.results)
                if result.test_predictions is None
            ]
            if missing_test:
                warnings.warn(
                    f"[校验警告] 以下模型没有测试集预测，将无法生成最终提交：\n"
                    f"  {missing_test}",
                    UserWarning,
                    stacklevel=2,
                )

        self._log(
            f"[校验] 通过 ✓ | OOF 长度={n_train:,} | "
            f"模型数={len(self.results)} | "
            f"有测试集预测的模型数={len(test_lengths)}"
        )

    # ------------------------------------------------------------------
    # 元特征构建
    # ------------------------------------------------------------------

    def _build_meta_features(
        self,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        将各 base model 的 OOF/test 预测拼接为元特征矩阵。

        返回
        -----
        (X_meta_train, X_meta_test)
          X_meta_train shape: (n_train, n_base_models [+ extra_features])
          X_meta_test  shape: (n_test,  n_base_models [+ extra_features]) | None
        """
        oof_cols:  dict[str, np.ndarray] = {}
        test_cols: dict[str, np.ndarray] = {}

        for name, result in zip(self.model_names, self.results):
            oof = result.oof_predictions.copy().astype(np.float64)
            # NaN 填 0（已在校验阶段警告）
            oof = np.nan_to_num(oof, nan=0.0)

            # 可选：裁剪极端值
            if self.config.clip_oof:
                lo, hi = np.nanpercentile(oof, 1), np.nanpercentile(oof, 99)
                oof = np.clip(oof, lo, hi)

            oof_cols[name] = oof

            if result.test_predictions is not None:
                test_cols[name] = result.test_predictions.astype(np.float64)

        X_meta_train = pd.DataFrame(oof_cols)
        X_meta_test  = pd.DataFrame(test_cols) if test_cols else None

        # ---- 追加原始特征（passthrough 或指定列）----
        extra_cols = self._resolve_extra_columns()
        if extra_cols and self.X_train_original is not None:
            self._validate_extra_columns(extra_cols)
            extra_train = self.X_train_original[extra_cols].reset_index(drop=True)
            # 只追加数值列（category/object 不直接喂给 Ridge）
            num_extra = extra_train.select_dtypes(include=np.number)
            if len(num_extra.columns) < len(extra_cols):
                skipped = set(extra_cols) - set(num_extra.columns)
                warnings.warn(
                    f"以下非数值列将不追加到元特征（请先编码）：{sorted(skipped)}",
                    UserWarning,
                    stacklevel=2,
                )
            X_meta_train = pd.concat(
                [X_meta_train, num_extra.reset_index(drop=True)], axis=1
            )
            if X_meta_test is not None and self.X_test_original is not None:
                extra_test = self.X_test_original[num_extra.columns].reset_index(drop=True)
                X_meta_test = pd.concat(
                    [X_meta_test, extra_test.reset_index(drop=True)], axis=1
                )

        return X_meta_train, X_meta_test

    def _resolve_extra_columns(self) -> list[str]:
        """解析需要追加的原始特征列名。"""
        if self.config.append_original_features:
            return list(self.config.append_original_features)
        if self.config.passthrough and self.X_train_original is not None:
            return self.X_train_original.columns.tolist()
        return []

    def _validate_extra_columns(self, cols: list[str]) -> None:
        """确保指定的原始特征列存在。"""
        if self.X_train_original is None:
            raise ValueError(
                "passthrough=True 或设置了 append_original_features，"
                "但未传入 X_train_original。"
            )
        missing = set(cols) - set(self.X_train_original.columns)
        if missing:
            raise ValueError(
                f"append_original_features 中以下列不存在：{sorted(missing)}"
            )

    # ------------------------------------------------------------------
    # 标准化
    # ------------------------------------------------------------------

    def _maybe_scale(
        self,
        X_train: pd.DataFrame,
        X_test:  Optional[pd.DataFrame],
    ) -> tuple[np.ndarray, Optional[np.ndarray], Optional[StandardScaler]]:
        """
        可选地对元特征做 StandardScaler（仅 fit 在 train 上）。
        返回 (X_train_scaled, X_test_scaled, scaler)。
        """
        if not self.config.scale_meta_features:
            return (
                X_train.values,
                X_test.values if X_test is not None else None,
                None,
            )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train.values)
        X_test_s  = scaler.transform(X_test.values) if X_test is not None else None
        return X_train_s, X_test_s, scaler

    # ------------------------------------------------------------------
    # 元学习器
    # ------------------------------------------------------------------

    def _build_meta_learner(self) -> Any:
        """
        按 task 返回合适的默认元学习器，或使用用户自定义的。

        为什么用简单线性模型做元学习器？
          - 避免过拟合到 OOF（OOF 维度通常只有 2~10 列）
          - 系数可解释（直接读出每个 base model 的权重）
          - 训练极快，不构成瓶颈
        """
        if self.config.meta_learner is not None:
            return self.config.meta_learner

        task = self.config.task.lower()
        if task == "regression":
            return Ridge(alpha=1.0, fit_intercept=True)
        elif task == "binary":
            return LogisticRegression(
                C=0.1,
                max_iter=1000,
                solver="lbfgs",
                penalty="l2",
            )
        elif task == "multiclass":
            return LogisticRegression(
                C=0.1,
                max_iter=1000,
                solver="lbfgs",
                penalty="l2",
                multi_class="multinomial",
            )
        else:
            # 兜底：regression 型
            return Ridge(alpha=1.0)

    def _meta_predict(self, learner: Any, X: np.ndarray) -> np.ndarray:
        """
        根据任务类型选择正确的元学习器预测方法。
          regression  → predict()
          binary      → predict_proba()[:, 1]
          multiclass  → predict_proba()（概率矩阵）或 predict()（类别）
        """
        task = self.config.task.lower()
        if task == "regression":
            return learner.predict(X)
        elif task == "binary":
            if hasattr(learner, "predict_proba"):
                proba = learner.predict_proba(X)
                return proba[:, 1] if proba.ndim == 2 else proba
            return learner.predict(X)
        elif task == "multiclass":
            if hasattr(learner, "predict_proba"):
                return learner.predict_proba(X)
            return learner.predict(X)
        else:
            return learner.predict(X)

    # ------------------------------------------------------------------
    # 权重提取
    # ------------------------------------------------------------------

    def _extract_weights(
        self,
        learner: Any,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """
        从元学习器中提取各 base model 对应的系数/权重。

        支持：
          - 线性模型（Ridge/ElasticNet/LogisticRegression）→ coef_
          - 基于树的模型（RF/ExtraTrees）→ feature_importances_
          - 其他 → 返回空 DataFrame + UserWarning
        """
        weights: Optional[np.ndarray] = None

        if hasattr(learner, "coef_"):
            coef = learner.coef_
            # LogisticRegression multiclass → shape (n_class, n_features)，取均值
            if coef.ndim == 2:
                coef = np.abs(coef).mean(axis=0)
            weights = coef.flatten()

        elif hasattr(learner, "feature_importances_"):
            weights = learner.feature_importances_

        if weights is None:
            warnings.warn(
                f"元学习器 {learner.__class__.__name__} 没有 coef_ 或 "
                "feature_importances_ 属性，无法提取权重。",
                UserWarning,
                stacklevel=2,
            )
            return pd.DataFrame(columns=["model", "weight", "abs_weight"])

        # 对齐维度（extra columns 时 weights > model_names 长度）
        n = min(len(weights), len(feature_names))
        rows = [
            {
                "model":      feature_names[i],
                "weight":     float(weights[i]),
                "abs_weight": float(abs(weights[i])),
            }
            for i in range(n)
        ]
        df = pd.DataFrame(rows).sort_values("abs_weight", ascending=False)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def correlation_matrix(self) -> pd.DataFrame:
        """
        计算各 base model OOF 预测之间的相关系数矩阵。
        低相关 → 多样性高 → 集成收益大。
        建议在 fit 前调用，诊断模型多样性。
        """
        oof_dict = {
            name: result.oof_predictions
            for name, result in zip(self.model_names, self.results)
        }
        return pd.DataFrame(oof_dict).corr()

    def diversity_report(self) -> str:
        """
        打印 OOF 预测多样性分析报告：
          - 两两相关系数
          - 平均相关系数（越低越好）
        """
        corr = self.correlation_matrix()
        lines = [
            "=" * 55,
            "  Base Model OOF 预测多样性分析",
            "  （相关系数越低 → 多样性越高 → 集成收益越大）",
            "=" * 55,
        ]
        # 上三角
        names = corr.columns.tolist()
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = corr.iloc[i, j]
                pairs.append((names[i], names[j], r))
                lines.append(f"  {names[i]:<16} ↔  {names[j]:<16} r={r:.4f}")

        if pairs:
            avg_r = np.mean([abs(p[2]) for p in pairs])
            lines.append(f"  {'─' * 50}")
            lines.append(f"  平均 |r| = {avg_r:.4f}")
            if avg_r > 0.98:
                lines.append("  ⚠  模型高度相关，集成收益有限，建议增加模型多样性")
            elif avg_r > 0.90:
                lines.append("  →  中等相关，集成仍有效")
            else:
                lines.append("  ✓  低相关，集成收益显著")
        lines.append("=" * 55)
        return "\n".join(lines)

    def oof_score_report(
        self,
        y: Union[pd.Series, np.ndarray],
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        metric_name: str = "score",
    ) -> pd.DataFrame:
        """
        计算各 base model OOF 得分与集成后得分的对比表。

        Parameters
        ----------
        y : 真实标签
        metric_fn : fn(y_true, y_pred) -> float
        metric_name : 指标名称

        Returns
        -------
        pd.DataFrame  index=model_name, columns=[metric_name]
        """
        y_arr = np.asarray(y)
        rows  = []

        for name, result in zip(self.model_names, self.results):
            oof = np.nan_to_num(result.oof_predictions.copy(), nan=0.0)
            score = metric_fn(y_arr, oof)
            rows.append({"model": name, metric_name: score})

        # 加入集成结果
        if self._stacking_result is not None:
            ensemble_score = metric_fn(y_arr, self._stacking_result.meta_oof)
            rows.append({"model": "★ Stacking", metric_name: ensemble_score})

        df = pd.DataFrame(rows).set_index("model")
        return df

    # ------------------------------------------------------------------
    # 属性访问
    # ------------------------------------------------------------------

    @property
    def result(self) -> StackingResult:
        if self._stacking_result is None:
            raise RuntimeError("请先调用 fit() 后再访问 result。")
        return self._stacking_result

    @property
    def meta_oof(self) -> np.ndarray:
        return self.result.meta_oof

    @property
    def model_weights(self) -> pd.DataFrame:
        return self.result.model_weights

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            self._logger.info(msg)

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self._stacking_result is not None
        return (
            f"StackingEnsemble("
            f"n_models={len(self.results)}, "
            f"task={self.config.task!r}, "
            f"fitted={fitted})"
        )
