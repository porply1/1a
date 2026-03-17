"""
post_process/optimizer.py
--------------------------
竞赛后处理三件套：概率校准 + 阈值优化 + 加权融合。

模块概览
--------
ProbabilityCalibrator
    使用 Isotonic Regression（非参数单调回归）或 Platt Scaling（logistic sigmoid）
    校准 OOF 概率，缓解 GBM 模型输出概率过于极端的问题。
    支持 K-Fold CV 内校准，防止校准器自身过拟合。

ThresholdOptimizer
    通过 Optuna TPE 搜索最优二分类决策阈值（默认最大化 F1）。
    支持 F1 / MCC / Fbeta / 任意 Callable 目标函数。
    直接接受 OOF 概率，无需重新推理。

WeightedEnsembleOptimizer
    通过 Optuna 搜索多模型 OOF 融合的最优权重。
    权重参数化为 softmax(raw_weights)，保证非负且和为 1。
    提供 from_cv_results() 工厂方法，直接从 CVResult 列表构建。

接口约定
--------
所有类均实现：
    fit(y_true, oof_predictions, ...)  → self
    transform(predictions)             → np.ndarray
    fit_transform(...)                 → np.ndarray

与 CVResult 兼容
-----------------
>>> from core.base_trainer import CVResult
>>> calibrator = ProbabilityCalibrator()
>>> calibrator.fit(y_train, result.oof_predictions)
>>> calibrated_test = calibrator.transform(result.test_predictions)
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, matthews_corrcoef

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


# ===========================================================================
# ProbabilityCalibrator
# ===========================================================================

class ProbabilityCalibrator:
    """
    OOF 概率校准器（Isotonic Regression / Platt Scaling）。

    问题背景
    --------
    GBM 模型（LightGBM/XGBoost/CatBoost）输出的原始概率往往过于"自信"，
    分布趋向于两端（0 附近和 1 附近），实际校准度（calibration）较差。
    校准后的概率更接近真实概率，有助于：
      - 阈值优化更准确
      - 模型融合权重更合理
      - 风险决策更可靠

    两种校准方法
    ------------
    isotonic（默认）:
        单调保序回归，非参数方法，拟合能力强，适合样本量 ≥ 1000 的场景。
    platt:
        对数几率回归（sigmoid fit），参数少，适合小样本或需要平滑输出的场景。

    防过拟合
    --------
    使用 n_folds 折 CV 内校准：
      对每一折 (tr, va)，在 tr 上拟合校准器，对 va 执行 transform。
    测试集使用全量 OOF 拟合一次的全局校准器 transform。

    Parameters
    ----------
    method : str
        校准方法：'isotonic'（默认）| 'platt'。
    n_folds : int
        内部 CV 折数，默认 5。设为 1 时退化为全局校准（可能轻微过拟合）。
    clip_eps : float
        输出概率截断到 [clip_eps, 1 - clip_eps]，防止概率为 0/1，默认 1e-6。
    random_state : int
        随机种子，默认 42。
    verbose : bool
        是否打印日志，默认 True。

    Examples
    --------
    >>> cal = ProbabilityCalibrator(method="isotonic", n_folds=5)
    >>> oof_calibrated = cal.fit_transform(y_true, oof_proba)
    >>> test_calibrated = cal.transform(test_proba)
    """

    _VALID_METHODS = {"isotonic", "platt"}

    def __init__(
        self,
        method:       str   = "isotonic",
        n_folds:      int   = 5,
        clip_eps:     float = 1e-6,
        random_state: int   = 42,
        verbose:      bool  = True,
    ):
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method 必须是 {self._VALID_METHODS}，得到 '{method}'"
            )
        self.method       = method
        self.n_folds      = n_folds
        self.clip_eps     = clip_eps
        self.random_state = random_state
        self.verbose      = verbose

        self._global_calibrator = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def fit(
        self,
        y_true:          Union[np.ndarray, pd.Series],
        oof_predictions: Union[np.ndarray, pd.Series],
    ) -> "ProbabilityCalibrator":
        """
        在 OOF 预测上拟合全局校准器（供后续 transform 测试集使用）。

        Parameters
        ----------
        y_true : array-like, shape (n,)
            真实标签（0/1 或连续值）。
        oof_predictions : array-like, shape (n,)
            训练集 OOF 预测概率。

        Returns
        -------
        self
        """
        y    = np.asarray(y_true, dtype=float)
        pred = np.asarray(oof_predictions, dtype=float)

        self._global_calibrator = self._fit_one(pred, y)
        self._is_fitted = True

        if self.verbose:
            # 诊断：校准前后 brier score
            brier_before = float(np.mean((pred - y) ** 2))
            pred_cal     = self._apply_one(self._global_calibrator, pred)
            brier_after  = float(np.mean((pred_cal - y) ** 2))
            print(
                f"[calibrator] method={self.method} | "
                f"Brier: {brier_before:.5f} → {brier_after:.5f} "
                f"({'改善' if brier_after < brier_before else '未改善'})"
            )
        return self

    def fit_transform(
        self,
        y_true:          Union[np.ndarray, pd.Series],
        oof_predictions: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """
        K-Fold CV 内校准 OOF 预测（防校准器自身过拟合），
        同时拟合全局校准器供后续 transform 测试集使用。

        Returns
        -------
        np.ndarray, shape (n,) — 校准后的 OOF 概率
        """
        y    = np.asarray(y_true, dtype=float)
        pred = np.asarray(oof_predictions, dtype=float)
        n    = len(y)

        if self.n_folds <= 1:
            # 不做 CV 内校准，直接全局校准
            self.fit(y, pred)
            return self.transform(pred)

        # K-Fold CV 内校准
        oof_calibrated = np.empty(n, dtype=np.float32)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for tr_idx, va_idx in kf.split(pred):
            cal = self._fit_one(pred[tr_idx], y[tr_idx])
            oof_calibrated[va_idx] = self._apply_one(cal, pred[va_idx])

        # 拟合全局校准器（测试集用）
        self.fit(y, pred)

        return np.clip(oof_calibrated, self.clip_eps, 1.0 - self.clip_eps)

    def transform(
        self,
        predictions: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """
        对预测概率应用全局校准器。

        Parameters
        ----------
        predictions : array-like, shape (n,)
            待校准的概率（通常为测试集预测值）。

        Returns
        -------
        np.ndarray, shape (n,) — 校准后的概率
        """
        self._assert_fitted()
        pred = np.asarray(predictions, dtype=float)
        calibrated = self._apply_one(self._global_calibrator, pred)
        return np.clip(calibrated, self.clip_eps, 1.0 - self.clip_eps).astype(np.float32)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _fit_one(self, pred: np.ndarray, y: np.ndarray):
        """拟合一个校准器实例。"""
        if self.method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(pred, y)
        else:  # platt
            cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            cal.fit(pred.reshape(-1, 1), y)
        return cal

    def _apply_one(self, cal, pred: np.ndarray) -> np.ndarray:
        """应用校准器。"""
        if self.method == "isotonic":
            return cal.predict(pred).astype(np.float32)
        else:
            return cal.predict_proba(pred.reshape(-1, 1))[:, 1].astype(np.float32)

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "ProbabilityCalibrator 尚未 fit，请先调用 fit() 或 fit_transform()。"
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"ProbabilityCalibrator("
            f"method={self.method!r}, "
            f"n_folds={self.n_folds}, "
            f"status={status})"
        )


# ===========================================================================
# ThresholdOptimizer
# ===========================================================================

class ThresholdOptimizer:
    """
    二分类决策阈值优化器（Optuna TPE 搜索）。

    问题背景
    --------
    二分类模型输出概率后，默认以 0.5 作为阈值，但在类别不均衡场景下
    最优阈值往往偏离 0.5。通过在 OOF 概率上直接搜索最优阈值，可以
    在不重新训练模型的前提下提升 F1 / MCC 等指标。

    优化过程
    --------
    1. 使用 Optuna TPE（贝叶斯优化）在 [search_range[0], search_range[1]] 搜索阈值
    2. 对每个候选阈值计算 metric_fn(y_true, (oof_proba >= threshold).astype(int))
    3. 返回使 metric 最大化（或最小化）的阈值

    Parameters
    ----------
    metric : str | Callable
        优化目标：
        - 'f1'    : sklearn.metrics.f1_score（默认）
        - 'mcc'   : matthews_corrcoef
        - 'fbeta' : 需配合 beta 参数
        - Callable: fn(y_true, y_pred_binary) -> float
    beta : float
        Fbeta 中的 beta 值（仅 metric='fbeta' 时有效），默认 1.0（等同 F1）。
    n_trials : int
        Optuna 搜索次数，默认 200。越多越精确，但 >500 后边际收益递减。
    search_range : tuple[float, float]
        阈值搜索范围，默认 (0.01, 0.99)。
    direction : str
        优化方向：'maximize'（默认）| 'minimize'。
        通常指标越高越好，选 maximize。
    random_state : int
        随机种子，默认 42。
    verbose : bool
        是否打印 Optuna 搜索日志，默认 False（Optuna 自身日志已关闭）。

    Attributes
    ----------
    best_threshold_ : float
        搜索到的最优阈值。
    best_score_ : float
        最优阈值对应的 OOF 指标分数。

    Examples
    --------
    >>> opt = ThresholdOptimizer(metric="f1", n_trials=300)
    >>> opt.fit(y_true, oof_proba)
    >>> print(opt.best_threshold_)  # e.g. 0.38
    >>> y_pred = opt.predict(test_proba)
    """

    def __init__(
        self,
        metric:       Union[str, Callable] = "f1",
        beta:         float                = 1.0,
        n_trials:     int                  = 200,
        search_range: tuple[float, float]  = (0.01, 0.99),
        direction:    str                  = "maximize",
        random_state: int                  = 42,
        verbose:      bool                 = False,
    ):
        self.metric       = metric
        self.beta         = beta
        self.n_trials     = n_trials
        self.search_range = search_range
        self.direction    = direction
        self.random_state = random_state
        self.verbose      = verbose

        self.best_threshold_: float = 0.5
        self.best_score_:     float = float("nan")
        self._is_fitted:      bool  = False

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def fit(
        self,
        y_true:          Union[np.ndarray, pd.Series],
        oof_predictions: Union[np.ndarray, pd.Series],
    ) -> "ThresholdOptimizer":
        """
        在 OOF 预测上搜索最优阈值。

        Parameters
        ----------
        y_true : array-like, shape (n,)
            真实二值标签（0/1）。
        oof_predictions : array-like, shape (n,)
            训练集 OOF 预测概率。

        Returns
        -------
        self
        """
        if not _HAS_OPTUNA:
            warnings.warn(
                "ThresholdOptimizer 需要 optuna，未安装时回退到网格搜索。"
            )
            self._grid_search(
                np.asarray(y_true, dtype=float),
                np.asarray(oof_predictions, dtype=float),
            )
            return self

        y    = np.asarray(y_true, dtype=float)
        pred = np.asarray(oof_predictions, dtype=float)
        metric_fn = self._resolve_metric()

        def objective(trial: optuna.Trial) -> float:
            thr = trial.suggest_float("threshold", *self.search_range)
            y_bin = (pred >= thr).astype(int)
            return float(metric_fn(y, y_bin))

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study   = optuna.create_study(direction=self.direction, sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_threshold_ = float(study.best_params["threshold"])
        self.best_score_     = float(study.best_value)
        self._is_fitted      = True

        if self.verbose:
            default_score = float(
                metric_fn(y, (pred >= 0.5).astype(int))
            )
            print(
                f"[threshold_opt] best_thr={self.best_threshold_:.4f} | "
                f"best_score={self.best_score_:.5f} | "
                f"default(0.5)={default_score:.5f} | "
                f"delta={self.best_score_ - default_score:+.5f}"
            )
        return self

    def predict(
        self,
        predictions: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """
        用最优阈值将概率转为 0/1 预测。

        Parameters
        ----------
        predictions : array-like, shape (n,)
            待预测的概率（OOF 或测试集）。

        Returns
        -------
        np.ndarray, shape (n,) — 0/1 二值数组
        """
        self._assert_fitted()
        pred = np.asarray(predictions, dtype=float)
        return (pred >= self.best_threshold_).astype(np.int8)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _resolve_metric(self) -> Callable:
        """将 metric 字符串解析为可调用函数。"""
        if callable(self.metric):
            return self.metric
        if self.metric == "f1":
            return lambda yt, yp: f1_score(yt, yp, zero_division=0)
        if self.metric == "mcc":
            return matthews_corrcoef
        if self.metric == "fbeta":
            from sklearn.metrics import fbeta_score
            beta = self.beta
            return lambda yt, yp: fbeta_score(yt, yp, beta=beta, zero_division=0)
        raise ValueError(
            f"未知 metric '{self.metric}'。"
            "支持: 'f1', 'mcc', 'fbeta' 或 Callable。"
        )

    def _grid_search(
        self,
        y:    np.ndarray,
        pred: np.ndarray,
    ) -> None:
        """Optuna 不可用时的简单网格搜索后备。"""
        metric_fn  = self._resolve_metric()
        candidates = np.linspace(self.search_range[0], self.search_range[1], 200)
        best_score = float("-inf") if self.direction == "maximize" else float("inf")
        best_thr   = 0.5

        for thr in candidates:
            score = float(metric_fn(y, (pred >= thr).astype(int)))
            if (self.direction == "maximize" and score > best_score) or \
               (self.direction == "minimize" and score < best_score):
                best_score = score
                best_thr   = float(thr)

        self.best_threshold_ = best_thr
        self.best_score_     = best_score
        self._is_fitted      = True

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "ThresholdOptimizer 尚未 fit，请先调用 fit()。"
            )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        thr    = f"{self.best_threshold_:.4f}" if self._is_fitted else "?"
        return (
            f"ThresholdOptimizer("
            f"metric={self.metric!r}, "
            f"n_trials={self.n_trials}, "
            f"best_threshold={thr}, "
            f"status={status})"
        )


# ===========================================================================
# WeightedEnsembleOptimizer
# ===========================================================================

class WeightedEnsembleOptimizer:
    """
    多模型 OOF 加权融合权重优化器（Optuna + Softmax 参数化）。

    算法
    ----
    1. 将每个模型的 OOF 预测组成矩阵 P，shape (n_samples, n_models)
    2. 使用 Optuna 搜索 raw_weights（无约束实数向量）
    3. 权重参数化：w = softmax(raw_weights)，保证 w_i ≥ 0 且 sum(w) = 1
    4. 候选融合：pred_ensemble = P @ w
    5. 最大化 metric_fn(y_true, pred_ensemble)

    Softmax 参数化优势
    -----------------
    - 避免线性约束（0 ≤ w_i，sum=1），大幅简化搜索空间
    - 梯度平滑，TPE 更容易找到全局最优
    - 自动保证权重和为 1

    Parameters
    ----------
    metric : str | Callable
        优化目标：'auc'（默认）| 'rmse' | 'logloss' | Callable。
        Callable 签名：fn(y_true, y_pred_ensemble) -> float。
    n_trials : int
        Optuna 搜索次数，默认 500。
    direction : str
        'maximize'（默认，适合 AUC/F1）| 'minimize'（适合 RMSE/logloss）。
    random_state : int
        随机种子，默认 42。
    verbose : bool
        是否打印最优权重，默认 True。

    Attributes
    ----------
    best_weights_ : np.ndarray, shape (n_models,)
        最优融合权重（softmax 归一化后）。
    best_score_ : float
        最优权重对应的 OOF 指标分数。
    model_names_ : list[str]
        模型名称列表（fit 时提供）。

    Examples
    --------
    >>> ens = WeightedEnsembleOptimizer(metric="auc", n_trials=500)
    >>> ens.fit(y_true, oof_matrix, model_names=["lgbm", "xgb", "catb"])
    >>> print(ens.best_weights_)   # e.g. [0.45, 0.35, 0.20]
    >>> test_ensemble = ens.transform(test_matrix)

    # 从 CVResult 列表直接构建
    >>> ens = WeightedEnsembleOptimizer.from_cv_results(results, y_true)
    """

    _METRIC_FN_MAP: dict[str, tuple[Callable, str]] = {}  # 动态填充

    def __init__(
        self,
        metric:       Union[str, Callable] = "auc",
        n_trials:     int                  = 500,
        direction:    str                  = "maximize",
        random_state: int                  = 42,
        verbose:      bool                 = True,
    ):
        self.metric       = metric
        self.n_trials     = n_trials
        self.direction    = direction
        self.random_state = random_state
        self.verbose      = verbose

        self.best_weights_: np.ndarray = np.array([])
        self.best_score_:   float      = float("nan")
        self.model_names_:  list[str]  = []
        self._is_fitted:    bool       = False

    # ------------------------------------------------------------------
    # 工厂方法：从 CVResult 列表构建
    # ------------------------------------------------------------------

    @classmethod
    def from_cv_results(
        cls,
        cv_results: list,          # list[CVResult]
        y_true:     Union[np.ndarray, pd.Series],
        metric:     Union[str, Callable] = "auc",
        n_trials:   int                  = 500,
        direction:  str                  = "maximize",
        model_names: Optional[list[str]] = None,
        **kwargs,
    ) -> "WeightedEnsembleOptimizer":
        """
        从 CVResult 列表提取 OOF 预测并自动执行 fit。

        Parameters
        ----------
        cv_results : list[CVResult]
            CrossValidatorTrainer.fit() 返回值列表，每个元素对应一个模型。
        y_true : array-like
            真实标签。
        model_names : list[str] | None
            模型名称列表，长度须与 cv_results 相同。
            None 时自动生成 ["model_0", "model_1", ...]。

        Returns
        -------
        WeightedEnsembleOptimizer (已 fit)
        """
        oof_matrix = np.column_stack(
            [r.oof_predictions for r in cv_results]
        )
        names = model_names or [f"model_{i}" for i in range(len(cv_results))]
        obj   = cls(metric=metric, n_trials=n_trials, direction=direction, **kwargs)
        obj.fit(y_true, oof_matrix, model_names=names)
        return obj

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def fit(
        self,
        y_true:      Union[np.ndarray, pd.Series],
        oof_matrix:  Union[np.ndarray, pd.DataFrame],
        model_names: Optional[list[str]] = None,
    ) -> "WeightedEnsembleOptimizer":
        """
        在 OOF 矩阵上搜索最优融合权重。

        Parameters
        ----------
        y_true : array-like, shape (n,)
            真实标签。
        oof_matrix : array-like, shape (n, n_models)
            每列为一个模型的 OOF 预测值。
            支持 pd.DataFrame（自动提取列名作为 model_names）。
        model_names : list[str] | None
            模型名称，None 时从 oof_matrix 列名提取（若为 DataFrame）
            或自动生成。

        Returns
        -------
        self
        """
        y = np.asarray(y_true, dtype=float)

        if isinstance(oof_matrix, pd.DataFrame):
            if model_names is None:
                model_names = list(oof_matrix.columns)
            P = oof_matrix.values.astype(np.float64)
        else:
            P = np.asarray(oof_matrix, dtype=np.float64)

        n_models = P.shape[1]
        if model_names is None:
            model_names = [f"model_{i}" for i in range(n_models)]
        self.model_names_ = model_names

        # 检查 NaN/Inf
        P = np.where(np.isfinite(P), P, 0.0)

        metric_fn = self._resolve_metric()

        if not _HAS_OPTUNA:
            warnings.warn(
                "WeightedEnsembleOptimizer 需要 optuna，回退到均值融合。"
            )
            self.best_weights_ = np.ones(n_models) / n_models
            pred_avg = P.mean(axis=1)
            self.best_score_ = float(metric_fn(y, pred_avg))
            self._is_fitted  = True
            return self

        def _softmax(raw: np.ndarray) -> np.ndarray:
            e = np.exp(raw - raw.max())
            return e / e.sum()

        def objective(trial: optuna.Trial) -> float:
            raw = np.array([
                trial.suggest_float(f"w{i}", -5.0, 5.0)
                for i in range(n_models)
            ])
            w    = _softmax(raw)
            pred = P @ w
            return float(metric_fn(y, pred))

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study   = optuna.create_study(direction=self.direction, sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        raw_best = np.array([study.best_params[f"w{i}"] for i in range(n_models)])
        self.best_weights_ = _softmax(raw_best).astype(np.float32)
        self.best_score_   = float(study.best_value)
        self._is_fitted    = True

        if self.verbose:
            print(f"\n[ensemble_opt] best_score={self.best_score_:.5f} | 最优权重:")
            for name, w in zip(model_names, self.best_weights_):
                bar = "█" * int(w * 40)
                print(f"  {name:<30} {w:.4f}  {bar}")

            # 与均值融合对比
            avg_score = float(metric_fn(y, P.mean(axis=1)))
            print(
                f"  均值融合分数: {avg_score:.5f} | "
                f"加权融合增益: {self.best_score_ - avg_score:+.5f}"
            )

        return self

    def transform(
        self,
        predictions_matrix: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        用最优权重对预测矩阵加权融合。

        Parameters
        ----------
        predictions_matrix : array-like, shape (n, n_models)
            每列为一个模型的预测值（OOF 或测试集）。

        Returns
        -------
        np.ndarray, shape (n,) — 加权融合后的预测值
        """
        self._assert_fitted()
        if isinstance(predictions_matrix, pd.DataFrame):
            P = predictions_matrix.values.astype(np.float64)
        else:
            P = np.asarray(predictions_matrix, dtype=np.float64)

        P = np.where(np.isfinite(P), P, 0.0)
        return (P @ self.best_weights_).astype(np.float32)

    def fit_transform(
        self,
        y_true:      Union[np.ndarray, pd.Series],
        oof_matrix:  Union[np.ndarray, pd.DataFrame],
        model_names: Optional[list[str]] = None,
    ) -> np.ndarray:
        """fit + transform 便捷方法，返回加权融合的 OOF 预测。"""
        self.fit(y_true, oof_matrix, model_names=model_names)
        return self.transform(oof_matrix)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _resolve_metric(self) -> Callable:
        """将 metric 字符串解析为可调用函数。"""
        if callable(self.metric):
            return self.metric
        if self.metric == "auc":
            from sklearn.metrics import roc_auc_score
            return roc_auc_score
        if self.metric == "rmse":
            return lambda yt, yp: float(np.sqrt(np.mean((yt - yp) ** 2)))
        if self.metric == "logloss":
            from sklearn.metrics import log_loss
            return log_loss
        if self.metric == "f1":
            return lambda yt, yp: f1_score(
                yt, (yp >= 0.5).astype(int), zero_division=0
            )
        if self.metric == "mcc":
            return lambda yt, yp: matthews_corrcoef(yt, (yp >= 0.5).astype(int))
        raise ValueError(
            f"未知 metric '{self.metric}'。"
            "支持: 'auc', 'rmse', 'logloss', 'f1', 'mcc' 或 Callable。"
        )

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "WeightedEnsembleOptimizer 尚未 fit，请先调用 fit()。"
            )

    def get_weight_table(self) -> pd.DataFrame:
        """返回模型权重汇总 DataFrame。"""
        self._assert_fitted()
        return pd.DataFrame({
            "model":  self.model_names_,
            "weight": self.best_weights_,
        }).sort_values("weight", ascending=False).reset_index(drop=True)

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"WeightedEnsembleOptimizer("
            f"metric={self.metric!r}, "
            f"n_trials={self.n_trials}, "
            f"n_models={len(self.model_names_)}, "
            f"status={status})"
        )
