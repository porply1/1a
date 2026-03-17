"""
features/selector.py
--------------------
特征选择模块：Null Importance（置换检验）特征筛选器。

核心算法（Kaggle GrandMaster Olivier 方法）
-----------------------------------------
1. 在真实标签上训练轻量级模型（LightGBM），记录每个特征的真实重要性。
2. 将标签随机打乱 n_iterations 次，每次重新训练，记录"零假设"下的特征重要性分布。
3. 对每个特征：
     null_score = percentile(null_importances, percentile_threshold)
     keep = real_importance > null_score
4. 可选：通过 gain_threshold 进一步过滤总体贡献过低的特征。

防信息泄露
----------
- Null 模型使用打乱后的标签，特征-标签关联被完全切断。
- 真实模型仅在 fit(X, y) 时执行一次，与 Null 模型独立。
- 支持 max_samples 对大数据集采样，加速 Null 迭代。

用法
----
>>> sel = NullImportanceSelector(n_iterations=80, percentile_threshold=75)
>>> sel.fit(X_train, y_train)
>>> X_selected = sel.transform(X_train)
>>> print(sel.get_scores())          # 查看每个特征的分数
>>> print(sel.selected_features_)   # 最终保留的特征名列表
"""

from __future__ import annotations

import time
import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

# LightGBM 为主后端，RandomForest 为备用
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ---------------------------------------------------------------------------
# NullImportanceSelector
# ---------------------------------------------------------------------------

class NullImportanceSelector:
    """
    Null Importance 置换检验特征筛选器（Kaggle GrandMaster 级方法）。

    算法摘要
    --------
    对于每个特征 f：
        real_score(f)  = 真实标签训练的模型对 f 的特征重要性
        null_scores(f) = [打乱标签后训练模型对 f 的重要性] × n_iterations
        threshold_f    = percentile(null_scores(f), percentile_threshold)
        keep_f         = real_score(f) > threshold_f

    支持自定义重要性函数，默认使用 LightGBM `gain`（与 split 相比更稳健）。

    Parameters
    ----------
    n_iterations : int
        Null 模型训练次数，默认 80。越多越稳定，但耗时线性增长。
    percentile_threshold : float
        保留特征的置信度阈值，默认 75.0。
        75.0 表示"真实重要性必须超过 75% 的随机重要性"。
        建议范围：50（宽松）~ 95（严格）。
    task : str
        任务类型：'binary'（二分类）| 'multiclass' | 'regression'。
        影响 LightGBM objective 和随机森林类型选择。
    importance_type : str
        LightGBM 特征重要性类型：'gain'（默认，绝对增益）| 'split'。
    lgb_params : dict | None
        LightGBM 训练参数覆盖。None 时使用内置默认参数（极简快速）。
    max_samples : int | None
        训练时对样本采样的最大数量（加速 Null 迭代）。
        None 时使用全量数据。大数据集（>100K 行）建议设为 50_000。
    gain_threshold : float
        额外过滤：删除真实重要性占比低于此值的特征（绝对 0 贡献列）。
        默认 0.0（不额外过滤）。
    n_jobs : int
        LightGBM 并行线程数，默认 -1（全部 CPU）。
    random_state : int
        随机种子，控制标签打乱和模型随机性，默认 42。
    verbose : bool
        是否打印进度日志，默认 True。

    Attributes
    ----------
    selected_features_ : list[str]
        fit 后保留的特征名列表。
    real_importances_ : pd.Series
        真实模型的特征重要性（index=feature_name, values=importance）。
    null_importances_ : pd.DataFrame
        Null 模型重要性矩阵，shape (n_features, n_iterations)。
    feature_scores_ : pd.DataFrame
        每个特征的诊断分数（见 get_scores()）。
    fit_time_ : float
        fit() 总耗时（秒）。

    Examples
    --------
    >>> sel = NullImportanceSelector(n_iterations=80, percentile_threshold=75)
    >>> sel.fit(X_train, y_train)
    >>> X_new = sel.transform(X_train)
    >>> sel.get_scores().head(10)
    """

    _DEFAULT_LGB_PARAMS_BASE = {
        "n_estimators":    200,
        "learning_rate":   0.05,
        "max_depth":       5,
        "num_leaves":      31,
        "min_child_samples": 20,
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":       0.1,
        "reg_lambda":      0.1,
        "verbose":         -1,
        "n_jobs":          -1,
    }

    _TASK_OBJECTIVE = {
        "binary":     "binary",
        "multiclass": "multiclass",
        "regression": "regression",
    }

    def __init__(
        self,
        n_iterations:         int            = 80,
        percentile_threshold: float          = 75.0,
        task:                 str            = "binary",
        importance_type:      str            = "gain",
        lgb_params:           Optional[dict] = None,
        max_samples:          Optional[int]  = None,
        gain_threshold:       float          = 0.0,
        n_jobs:               int            = -1,
        random_state:         int            = 42,
        verbose:              bool           = True,
    ):
        if task not in self._TASK_OBJECTIVE:
            raise ValueError(
                f"task 必须是 {list(self._TASK_OBJECTIVE)}, 得到 '{task}'"
            )
        if not (0.0 <= percentile_threshold <= 100.0):
            raise ValueError("percentile_threshold 必须在 [0, 100] 范围内。")

        self.n_iterations         = n_iterations
        self.percentile_threshold = percentile_threshold
        self.task                 = task
        self.importance_type      = importance_type
        self.lgb_params           = lgb_params
        self.max_samples          = max_samples
        self.gain_threshold       = gain_threshold
        self.n_jobs               = n_jobs
        self.random_state         = random_state
        self.verbose              = verbose

        # fit 后填充
        self.selected_features_: list[str]    = []
        self.real_importances_:  pd.Series    = pd.Series(dtype=float)
        self.null_importances_:  pd.DataFrame = pd.DataFrame()
        self.feature_scores_:    pd.DataFrame = pd.DataFrame()
        self.fit_time_:          float        = 0.0
        self._is_fitted:         bool         = False

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> "NullImportanceSelector":
        """
        执行 Null Importance 置换检验，确定保留的特征集合。

        Parameters
        ----------
        X : pd.DataFrame
            特征矩阵（仅数值列参与选择，非数值列自动忽略）。
        y : pd.Series | np.ndarray
            目标标签。

        Returns
        -------
        self
        """
        t0 = time.perf_counter()

        # 预处理：仅保留数值列，Inf → NaN
        X_num, feature_names = self._preprocess_X(X)
        y_arr = np.asarray(y)

        if len(feature_names) == 0:
            warnings.warn("NullImportanceSelector: 没有可用的数值特征，fit 被跳过。")
            self._is_fitted = True
            return self

        # 可选采样（加速大数据集）
        X_sub, y_sub = self._maybe_subsample(X_num, y_arr)

        self._log(
            f"NullImportanceSelector | task={self.task} | "
            f"features={len(feature_names)} | "
            f"n_rows={len(X_sub):,} | n_iterations={self.n_iterations}"
        )

        # 1. 真实模型重要性
        self._log("  >> 训练真实模型...")
        real_imp = self._train_and_get_importance(X_sub, y_sub, feature_names)
        self.real_importances_ = pd.Series(real_imp, index=feature_names, name="real")

        # 2. Null 模型重要性（打乱标签）
        rng = np.random.default_rng(self.random_state)
        null_matrix = np.zeros((len(feature_names), self.n_iterations), dtype=np.float32)

        for i in range(self.n_iterations):
            y_shuffled = rng.permutation(y_sub)
            null_imp = self._train_and_get_importance(X_sub, y_shuffled, feature_names)
            null_matrix[:, i] = null_imp.astype(np.float32)

            if self.verbose and (i + 1) % 20 == 0:
                elapsed = time.perf_counter() - t0
                self._log(f"  Null 迭代 {i + 1}/{self.n_iterations} | 耗时 {elapsed:.1f}s")

        self.null_importances_ = pd.DataFrame(
            null_matrix,
            index=feature_names,
            columns=[f"null_{i}" for i in range(self.n_iterations)],
        )

        # 3. 计算分数并决定保留/丢弃
        self._compute_scores_and_select(real_imp, null_matrix, feature_names)

        self.fit_time_ = time.perf_counter() - t0
        self._is_fitted = True

        n_sel = len(self.selected_features_)
        n_tot = len(feature_names)
        self._log(
            f"  选择结果: {n_sel}/{n_tot} 特征保留 "
            f"({n_sel/max(n_tot,1)*100:.1f}%) | "
            f"总耗时 {self.fit_time_:.1f}s"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        返回只包含 selected_features_ 的 DataFrame。

        Parameters
        ----------
        X : pd.DataFrame
            原始特征矩阵（行数任意，可与 fit 时不同）。

        Returns
        -------
        pd.DataFrame，列为 selected_features_ 中存在于 X 的特征。
        """
        self._assert_fitted()
        keep = [c for c in self.selected_features_ if c in X.columns]
        if len(keep) < len(self.selected_features_):
            missing = set(self.selected_features_) - set(X.columns)
            warnings.warn(
                f"NullImportanceSelector.transform: {len(missing)} 个选中特征在 X 中缺失，已忽略。"
            )
        return X[keep].copy()

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """fit + transform 便捷方法。"""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # 诊断接口
    # ------------------------------------------------------------------

    def get_scores(self) -> pd.DataFrame:
        """
        返回每个特征的诊断分数 DataFrame，按 score 降序排列。

        列说明
        ------
        feature          : 特征名
        real_importance  : 真实模型特征重要性
        null_mean        : Null 重要性均值
        null_std         : Null 重要性标准差
        null_percentile  : Null 重要性在 percentile_threshold 处的分位值
        score            : real / (null_percentile + eps)，越大越好
        selected         : 是否被选中
        """
        self._assert_fitted()
        return self.feature_scores_.copy()

    def plot_importance_distribution(self, top_n: int = 20) -> None:
        """
        打印 top_n 特征的文本条形图（真实 vs Null 均值对比）。
        不依赖 matplotlib，在 Notebook/终端均可用。
        """
        self._assert_fitted()
        df = self.feature_scores_.head(top_n)
        max_real = float(df["real_importance"].max()) or 1.0

        print(f"\n  {'特征名':<40} {'真实':>8} {'Null均值':>9} {'选中':>5}")
        print("  " + "-" * 68)
        for _, row in df.iterrows():
            bar_len = int(row["real_importance"] / max_real * 30)
            bar = "█" * bar_len
            sel_mark = "YES" if row["selected"] else "---"
            print(
                f"  {row['feature']:<40} "
                f"{row['real_importance']:>8.2f} "
                f"{row['null_mean']:>9.2f} "
                f"{sel_mark:>5}  {bar}"
            )

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _preprocess_X(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, list[str]]:
        """保留数值列，Inf → NaN，返回 (numpy array, feature_names)。"""
        X_num = X.select_dtypes(include=np.number)
        X_clean = X_num.replace([np.inf, -np.inf], np.nan)
        return X_clean.values.astype(np.float32), list(X_clean.columns)

    def _maybe_subsample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """对大数据集按 max_samples 随机采样。"""
        if self.max_samples is None or len(X) <= self.max_samples:
            return X, y
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=self.max_samples, replace=False)
        return X[idx], y[idx]

    def _build_lgb_params(self) -> dict:
        """构建 LightGBM 参数字典（合并默认与用户覆盖）。"""
        params = dict(self._DEFAULT_LGB_PARAMS_BASE)
        params["objective"] = self._TASK_OBJECTIVE[self.task]
        params["n_jobs"]    = self.n_jobs

        if self.lgb_params:
            params.update(self.lgb_params)
        return params

    def _train_and_get_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        """
        训练一次模型，返回 shape (n_features,) 的重要性向量。
        LightGBM 优先，不可用时退回 RandomForest。
        """
        if _HAS_LGB:
            return self._lgb_importance(X, y, feature_names)
        return self._rf_importance(X, y, feature_names)

    def _lgb_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        """使用 LightGBM sklearn API 训练并提取 gain 重要性。"""
        params = self._build_lgb_params()

        if self.task in ("binary", "multiclass"):
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y, feature_name=feature_names)

        imp = model.booster_.feature_importance(importance_type=self.importance_type)
        return imp.astype(np.float64)

    def _rf_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> np.ndarray:
        """RandomForest 后备（无 LightGBM 时使用）。"""
        params = dict(
            n_estimators=100,
            max_depth=5,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        if self.task in ("binary", "multiclass"):
            model = RandomForestClassifier(**params)
        else:
            model = RandomForestRegressor(**params)

        # 用 NaN 均值填充（RandomForest 不支持 NaN）
        X_filled = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_filled, y)

        return model.feature_importances_.astype(np.float64)

    def _compute_scores_and_select(
        self,
        real_imp:      np.ndarray,
        null_matrix:   np.ndarray,
        feature_names: list[str],
    ) -> None:
        """
        基于真实重要性和 Null 分布，计算分数并填充 selected_features_。
        """
        null_pct = np.percentile(null_matrix, self.percentile_threshold, axis=1)
        null_mean = null_matrix.mean(axis=1)
        null_std  = null_matrix.std(axis=1)
        eps = 1e-8

        # 综合分数：real / (null_percentile + eps)，越大越好
        scores = real_imp / (null_pct + eps)

        # gain_threshold 额外过滤：相对贡献 < gain_threshold 的列直接删除
        total_real = float(real_imp.sum()) or 1.0
        real_ratio = real_imp / total_real

        rows = []
        selected = []
        for i, feat in enumerate(feature_names):
            keep = bool(real_imp[i] > null_pct[i]) and bool(real_ratio[i] >= self.gain_threshold)
            rows.append({
                "feature":         feat,
                "real_importance": float(real_imp[i]),
                "null_mean":       float(null_mean[i]),
                "null_std":        float(null_std[i]),
                "null_percentile": float(null_pct[i]),
                "score":           float(scores[i]),
                "selected":        keep,
            })
            if keep:
                selected.append(feat)

        self.feature_scores_ = (
            pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        self.selected_features_ = selected

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "NullImportanceSelector 尚未 fit，请先调用 fit()。"
            )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[selector] {msg}")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        n_sel  = len(self.selected_features_) if self._is_fitted else "?"
        return (
            f"NullImportanceSelector("
            f"n_iterations={self.n_iterations}, "
            f"percentile_threshold={self.percentile_threshold}, "
            f"task={self.task!r}, "
            f"selected={n_sel}, "
            f"status={status})"
        )
