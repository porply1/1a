"""
models/gbm/xgb_wrapper.py
--------------------------
XGBoost 原生 API Wrapper，继承 BaseModel。

设计要点：
  1. 全程使用 xgb.train + xgb.DMatrix（原生 API），严禁 sklearn 接口
     → 完整支持自定义 feval/fobj、DART booster、GPU hist 等高级特性
  2. 原生类别特征支持：DMatrix(enable_categorical=True)，配合 category dtype
     → XGBoost >= 1.7 原生支持，自动感知 pd.Categorical，无需手工编码
  3. Early Stopping：xgb.callback.EarlyStopping，支持 maximize 方向自动判断
  4. 序列化：booster.save_model(.json/.ubj) + .meta.json 元数据双轨并行
     → .json 可读可版本控制，.ubj（UBJSON）体积更小，推理更快
  5. feature_importance 支持四种 importance_type：
     weight / gain / cover / total_gain（接口与 LGBM 对称）
  6. _align_features 在推理时严格校验特征顺序，防止静默数值错位
  7. 内置 Optuna 搜索空间 suggest_params()，与调参层无缝对接

与 LGBMModel 的对称关系
-----------------------
  接口层面完全镜像，行为差异仅在底层 API 调用：
    lgb.Dataset       ←→  xgb.DMatrix
    lgb.train         ←→  xgb.train
    lgb.early_stopping←→  xgb.callback.EarlyStopping
    booster.save_model←→  booster.save_model
    feature_importance←→  booster.get_score

典型用法
--------
>>> from models.gbm.xgb_wrapper import XGBModel
>>> from core import CrossValidatorTrainer, MetricConfig
>>> from data.splitter import get_cv, CVConfig
>>> from sklearn.metrics import roc_auc_score

>>> model = XGBModel(
...     params={"max_depth": 8, "eta": 0.03},
...     task="binary",
...     num_boost_round=2000,
...     early_stopping_rounds=100,
... )
>>> cv = get_cv(CVConfig(strategy="stratified_kfold", n_splits=5))
>>> trainer = CrossValidatorTrainer(
...     model=model, cv=cv, task="binary",
...     metrics=[MetricConfig("auc", roc_auc_score, use_proba=True)],
...     save_dir="outputs/models",
... )
>>> result = trainer.fit(X_train, y_train, X_test=X_test)
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    import xgboost as xgb

    _XGB_VERSION = tuple(int(x) for x in xgb.__version__.split(".")[:2])
except ImportError as e:
    raise ImportError(
        "XGBoost 未安装，请执行：pip install xgboost"
    ) from e

from core.base_model import BaseModel, ModelState


# ---------------------------------------------------------------------------
# 版本守卫
# ---------------------------------------------------------------------------

_SUPPORTS_NATIVE_CATEGORICAL = _XGB_VERSION >= (1, 7)

if not _SUPPORTS_NATIVE_CATEGORICAL:
    warnings.warn(
        f"当前 XGBoost 版本 {xgb.__version__} < 1.7，"
        "原生类别特征支持（enable_categorical）不可用。"
        "建议升级：pip install -U xgboost",
        UserWarning,
        stacklevel=1,
    )


# ---------------------------------------------------------------------------
# 任务映射表
# ---------------------------------------------------------------------------

# XGBoost objective 命名与竞赛任务的映射
_TASK_TO_OBJECTIVE: dict[str, str] = {
    "binary":        "binary:logistic",    # → 概率输出
    "multiclass":    "multi:softprob",     # → 每类概率（num_class 必须设置）
    "regression":    "reg:squarederror",   # → MSE 回归
    "regression_l1": "reg:absoluteerror",  # → MAE 回归
    "huber":         "reg:pseudohubererror",
    "logistic":      "reg:logistic",       # → logistic 回归（输出概率，无阈值）
    "poisson":       "count:poisson",      # → 计数回归
    "tweedie":       "reg:tweedie",        # → Tweedie 回归（保险/销量场景）
    "rank":          "rank:pairwise",      # → 排序（Learning to Rank）
    "rank_ndcg":     "rank:ndcg",          # → NDCG 排序
}

# 分类任务集合（predict_proba 时返回概率）
_CLF_TASKS = {"binary", "multiclass", "logistic", "rank", "rank_ndcg"}

# 有效 importance_type
_FI_TYPES = {"weight", "gain", "cover", "total_gain", "total_cover"}

# eval_metric 中越大越好的指标（用于 EarlyStopping maximize 判断）
_HIGHER_IS_BETTER_METRICS = {
    "auc", "aucpr", "map", "ndcg", "error", "merror",
    "pre", "rec", "f1",
}


# ---------------------------------------------------------------------------
# XGBModel
# ---------------------------------------------------------------------------

class XGBModel(BaseModel):
    """
    XGBoost 原生 API 封装（xgb.train + DMatrix）。

    Parameters
    ----------
    params : dict | None
        XGBoost booster 参数。与 DEFAULT_PARAMS 合并，用户参数优先。
        不需要设置 objective，通过 task 参数自动推断注入。
    task : str
        任务类型，决定 objective 与 predict 行为。
        可选值见 _TASK_TO_OBJECTIVE 的键。默认：'regression'
    num_boost_round : int
        最大 boosting 轮数（即树的棵数上限）。默认：1000
    early_stopping_rounds : int | None
        连续多少轮 eval metric 不改善时停止。
        None 表示禁用 Early Stopping。默认：100
    eval_metric : str | list[str] | None
        xgb.train 的 eval_metric 参数。
        None 时按 task 自动推断合理默认值。
    maximize : bool | None
        Early Stopping 的优化方向。
        None 时按 eval_metric 自动判断（auc 等指标自动 maximize=True）。
    log_period : int
        每隔多少轮打印训练日志。0 = 完全静默。默认：100
    fi_type : str
        feature_importance 计算方式：
        'weight'(分裂次数) / 'gain'(平均增益) / 'cover'(平均覆盖度) /
        'total_gain'(总增益，推荐) / 'total_cover'
        默认：'total_gain'
    save_format : str
        序列化格式：'json'（可读）或 'ubj'（UBJSON，更小更快）。
        默认：'json'
    num_class : int | None
        多分类任务的类别数，task='multiclass' 时必须设置。
    name : str | None
        模型名称，用于日志和文件命名。
    seed : int
        全局随机种子。

    Attributes
    ----------
    booster : xgb.Booster | None
        训练完成后的原生 Booster 对象。
    best_iteration : int
        Early Stopping 停止时的最优迭代轮数（从 0 开始计数）。
    feature_names : list[str]
        训练时的特征名列表（按顺序）。
    categorical_features : list[str]
        自动识别到的 category dtype 列名。
    evals_result : dict
        训练过程的 eval 指标历史，格式：
        {"train": {"rmse": [...]}, "val": {"rmse": [...]}}
    """

    # ------------------------------------------------------------------
    # 竞赛高分稳健默认参数
    # ------------------------------------------------------------------
    DEFAULT_PARAMS: dict = {
        # 树结构
        "max_depth":         6,      # 树深度，竞赛常用 6~8，过深易过拟合
        "min_child_weight":  1,      # 叶子节点最小 Hessian 和（等效 LGBM min_child_samples）
        "max_delta_step":    0,      # 0 = 不限制叶子权重更新幅度（Poisson 场景设 1）
        "gamma":             0.0,    # 分裂最小增益阈值（正则化旋钮）

        # 学习率
        "eta":               0.05,   # 即 learning_rate，竞赛惯例命名 eta

        # 随机化（防过拟合三件套）
        "subsample":         0.8,    # 每棵树随机采样行比例
        "colsample_bytree":  0.8,    # 每棵树随机采样特征比例
        "colsample_bylevel": 1.0,    # 每层随机采样特征比例
        "colsample_bynode":  1.0,    # 每个分裂节点随机采样特征比例

        # 正则化
        "alpha":             0.1,    # L1 正则（稀疏化叶子权重，等效 LGBM lambda_l1）
        "lambda":            1.0,    # L2 正则（收缩叶子权重，等效 LGBM lambda_l2）

        # 训练加速
        "tree_method":       "hist", # hist = 直方图算法（快速，等效 LGBM 默认模式）
        "grow_policy":       "depthwise",  # 'lossguide' 类似 LGBM leaf-wise

        # 类别特征（XGBoost >= 1.7）
        "enable_categorical": True,  # 开启原生类别特征支持

        # 硬件
        "nthread":           -1,     # -1 = 使用全部 CPU 核
        "device":            "cpu",  # 'cuda' 需要 GPU 版 XGBoost

        # 输出控制
        "verbosity":         0,      # 0=静默，1=warning，2=info，3=debug
    }

    def __init__(
        self,
        params:                Optional[dict]       = None,
        task:                  str                  = "regression",
        num_boost_round:       int                  = 1000,
        early_stopping_rounds: Optional[int]        = 100,
        eval_metric:           Optional[Union[str, list[str]]] = None,
        maximize:              Optional[bool]        = None,
        log_period:            int                  = 100,
        fi_type:               str                  = "total_gain",
        save_format:           str                  = "json",
        num_class:             Optional[int]        = None,
        name:                  Optional[str]        = None,
        seed:                  int                  = 42,
    ):
        # ---- 校验 ----
        if task not in _TASK_TO_OBJECTIVE:
            raise ValueError(
                f"不支持的 task='{task}'。"
                f"可选：{list(_TASK_TO_OBJECTIVE.keys())}"
            )
        if fi_type not in _FI_TYPES:
            raise ValueError(
                f"fi_type 必须为 {_FI_TYPES}，收到：'{fi_type}'"
            )
        if save_format not in ("json", "ubj"):
            raise ValueError(
                f"save_format 必须为 'json' 或 'ubj'，收到：'{save_format}'"
            )
        if task == "multiclass" and num_class is None:
            warnings.warn(
                "task='multiclass' 但未设置 num_class，"
                "请确保 params 中包含 'num_class' 键。",
                UserWarning,
                stacklevel=2,
            )

        self.task                  = task
        self.num_boost_round       = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric           = eval_metric or self._default_eval_metric(task)
        self.maximize              = maximize
        self.log_period            = log_period
        self.fi_type               = fi_type
        self.save_format           = save_format
        self.num_class             = num_class

        # 运行时属性
        self.booster:             Optional[xgb.Booster] = None
        self.best_iteration:      int                   = 0
        self.feature_names:       list[str]             = []
        self.categorical_features: list[str]            = []
        self.evals_result:        dict                  = {}

        # 父类初始化（含 _merge_params + seed 注入）
        super().__init__(
            params=params,
            name=name or f"xgb_{task}",
            seed=seed,
        )

        # 合并后注入 objective 与 num_class
        self.params["objective"] = _TASK_TO_OBJECTIVE[task]
        if num_class is not None:
            self.params["num_class"] = num_class

    # ------------------------------------------------------------------
    # seed 键名
    # ------------------------------------------------------------------

    def _seed_key(self) -> str:
        # XGBoost 原生参数键为 "seed"，但文档也接受 "random_state"
        # 此处遵循项目约定，统一使用 "seed"
        return "seed"

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_tr:   pd.DataFrame,
        y_tr:   pd.Series,
        X_val:  Optional[pd.DataFrame] = None,
        y_val:  Optional[pd.Series]    = None,
        **kwargs,
    ) -> "XGBModel":
        """
        使用 xgb.train 原生 API 训练模型。

        Parameters
        ----------
        X_tr, y_tr  : 训练集特征与标签
        X_val, y_val : 验证集（用于 early stopping 和 evals 监控）
        **kwargs     : 透传给 xgb.train 的额外参数（如 fobj / feval）
        """
        t0 = time.perf_counter()

        # ---- 1. 识别 category 列 ----
        self.categorical_features = self._detect_categoricals(X_tr)
        self.feature_names        = X_tr.columns.tolist()

        # ---- 2. 构建 DMatrix ----
        dtrain = self._make_dmatrix(X_tr, y_tr)
        evals: list[tuple[xgb.DMatrix, str]] = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = self._make_dmatrix(X_val, y_val)
            evals.append((dval, "val"))
        else:
            dval = None

        # ---- 3. eval_metric 注入到 params ----
        # 注意：xgb.train 的 eval_metric 既可以在 params 里也可以独立传入
        # 独立传入优先级更高且不污染 params 字典（折间 clone 更干净）
        eval_metric = kwargs.pop("eval_metric", self.eval_metric)

        # ---- 4. 构建 callbacks ----
        callbacks, evals_result = self._build_callbacks(
            has_val     = dval is not None,
            eval_metric = eval_metric,
        )

        # ---- 5. 提取 xgb.train 专属 kwargs ----
        num_boost_round = kwargs.pop("num_boost_round", self.num_boost_round)

        # fobj / feval 可通过 kwargs 透传
        train_kwargs: dict = {}
        for key in ("fobj", "feval", "xgb_model"):
            if key in kwargs:
                train_kwargs[key] = kwargs.pop(key)

        # ---- 6. 训练 ----
        self.booster = xgb.train(
            params          = self.params,
            dtrain          = dtrain,
            num_boost_round = num_boost_round,
            evals           = evals,
            evals_result    = evals_result,
            callbacks       = callbacks,
            verbose_eval    = False,      # 日志完全由 callbacks 接管
            **train_kwargs,
            **kwargs,
        )
        self.evals_result = evals_result

        # ---- 7. 最优轮数 ----
        # XGBoost 的 best_iteration 从 0 开始，+1 后等价于 num_boost_round
        self.best_iteration = getattr(self.booster, "best_iteration", 0)
        if self.best_iteration == 0 and not evals_result:
            self.best_iteration = num_boost_round - 1  # 未启用 ES，用最大轮

        # ---- 8. 状态更新 ----
        self.fit_time = time.perf_counter() - t0
        self._fitted  = True
        self.state    = ModelState.FITTED

        return self

    # ------------------------------------------------------------------
    # predict / predict_proba
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        回归任务：返回原始预测值，形状 (n,)。
        分类任务：返回类别预测（argmax），形状 (n,)。
        需要概率时请使用 predict_proba。
        """
        self._assert_fitted()
        raw = self._raw_predict(X)
        if self.task in _CLF_TASKS:
            if raw.ndim == 2:           # multiclass → softprob 返回 (n, k)
                return np.argmax(raw, axis=1)
            return (raw >= 0.5).astype(int)   # binary → 阈值 0.5
        return raw

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        分类概率输出：
          - binary     → shape (n,)，正类概率
          - multiclass → shape (n, num_class)
          - 其他任务   → 等同 predict()
        """
        self._assert_fitted()
        return self._raw_predict(X)

    def _raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        """构建推理 DMatrix 并调用 booster.predict。"""
        X_aligned = self._align_features(X)
        dtest     = self._make_dmatrix(X_aligned, label=None)
        return self.booster.predict(
            dtest,
            iteration_range=(0, self.best_iteration + 1),
        )

    # ------------------------------------------------------------------
    # feature_importance
    # ------------------------------------------------------------------

    @property
    def feature_importance(self) -> pd.Series:
        """
        返回归一化到 [0,1] 的特征重要性 pd.Series。
        index 为特征名，name 为 fi_type。
        使用 booster.get_score()，缺失特征（未参与分裂）补 0。
        """
        self._assert_fitted()
        score_dict = self.booster.get_score(importance_type=self.fi_type)

        # 补全：未在任何树中被使用的特征补 0
        full_scores = {f: score_dict.get(f, 0.0) for f in self.feature_names}
        values      = np.array([full_scores[f] for f in self.feature_names],
                               dtype=np.float64)
        total = values.sum()
        normalized = values / total if total > 0 else values

        return pd.Series(normalized, index=self.feature_names, name=self.fi_type)

    def feature_importance_dataframe(self) -> pd.DataFrame:
        """
        返回所有 importance_type 的汇总 DataFrame，按 total_gain 降序。
        列：feature | weight | gain | cover | total_gain | total_cover
        """
        self._assert_fitted()
        rows = {}
        for fi_type in _FI_TYPES:
            score_dict = self.booster.get_score(importance_type=fi_type)
            for feat in self.feature_names:
                rows.setdefault(feat, {})["feature"] = feat
                rows[feat][fi_type] = score_dict.get(feat, 0.0)

        df = pd.DataFrame(list(rows.values()))
        return df.sort_values("total_gain", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        保存 Booster 与 Wrapper 元数据。

        生成文件：
          <path>.<save_format>    — XGBoost 原生格式（.json 可读 / .ubj 紧凑）
          <path>.meta.json        — Wrapper 状态（特征名、任务类型等）
        """
        self._assert_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # ---- 原生 Booster ----
        booster_path = path.with_suffix(f".{self.save_format}")
        self.booster.save_model(str(booster_path))

        # ---- 元数据 ----
        meta = {
            "name":                  self.name,
            "task":                  self.task,
            "num_boost_round":       self.num_boost_round,
            "best_iteration":        self.best_iteration,
            "early_stopping_rounds": self.early_stopping_rounds,
            "eval_metric":           self.eval_metric
                                     if isinstance(self.eval_metric, list)
                                     else [self.eval_metric],
            "fi_type":               self.fi_type,
            "save_format":           self.save_format,
            "num_class":             self.num_class,
            "seed":                  self.seed,
            "feature_names":         self.feature_names,
            "categorical_features":  self.categorical_features,
            "params":                self.params,
            "xgb_version":           xgb.__version__,
        }
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.state = ModelState.SAVED

    def load(self, path: Union[str, Path]) -> "XGBModel":
        """
        从磁盘恢复 Booster 及 Wrapper 元数据。

        Parameters
        ----------
        path : .json/.ubj 文件路径 或不含后缀的基础路径均可。
        """
        path = Path(path)
        meta_path = path.with_suffix(".meta.json")

        # ---- 元数据先行（获取 save_format 才能定位 booster 文件）----
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            fmt = meta.get("save_format", self.save_format)
        else:
            warnings.warn(
                f"未找到元数据文件 {meta_path}，"
                "feature_names 等属性可能不完整，将尝试从 .json 加载 Booster。",
                UserWarning,
                stacklevel=2,
            )
            meta  = {}
            fmt   = "json"

        booster_path = path.with_suffix(f".{fmt}")
        if not booster_path.exists():
            # fallback：尝试另一种格式
            alt_fmt      = "ubj" if fmt == "json" else "json"
            alt_path     = path.with_suffix(f".{alt_fmt}")
            if alt_path.exists():
                booster_path = alt_path
            else:
                raise FileNotFoundError(
                    f"未找到 Booster 文件：{booster_path} 或 {alt_path}"
                )

        # ---- 加载原生 Booster ----
        self.booster = xgb.Booster()
        self.booster.load_model(str(booster_path))

        # ---- 恢复 Wrapper 状态 ----
        if meta:
            self.name                 = meta.get("name",                 self.name)
            self.task                 = meta.get("task",                 self.task)
            self.best_iteration       = meta.get("best_iteration",       0)
            self.fi_type              = meta.get("fi_type",              self.fi_type)
            self.save_format          = meta.get("save_format",          self.save_format)
            self.num_class            = meta.get("num_class",            self.num_class)
            self.feature_names        = meta.get("feature_names",        [])
            self.categorical_features = meta.get("categorical_features", [])
            self.params               = meta.get("params",               self.params)
        else:
            # 降级：从 booster 获取特征名
            self.feature_names = self.booster.feature_names or []

        self._fitted = True
        self.state   = ModelState.FITTED
        return self

    # ------------------------------------------------------------------
    # clone（携带非 params 属性）
    # ------------------------------------------------------------------

    def clone(self) -> "XGBModel":
        return XGBModel(
            params                = self.get_params(),
            task                  = self.task,
            num_boost_round       = self.num_boost_round,
            early_stopping_rounds = self.early_stopping_rounds,
            eval_metric           = self.eval_metric,
            maximize              = self.maximize,
            log_period            = self.log_period,
            fi_type               = self.fi_type,
            save_format           = self.save_format,
            num_class             = self.num_class,
            name                  = self.name,
            seed                  = self.seed,
        )

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_categoricals(X: pd.DataFrame) -> list[str]:
        """识别 dtype == 'category' 的列，XGBoost >= 1.7 原生处理。"""
        return [col for col in X.columns if X[col].dtype.name == "category"]

    def _make_dmatrix(
        self,
        X:     pd.DataFrame,
        label: Optional[pd.Series],
    ) -> xgb.DMatrix:
        """
        构建 xgb.DMatrix，自动开启类别特征支持。

        注意事项：
          - enable_categorical=True 需要 XGBoost >= 1.7
          - category 列必须是 pd.CategoricalDtype，不能是 object
          - feature_names 显式传入，确保特征名可追溯
        """
        kwargs: dict = {
            "data":              X,
            "label":             label,
            "feature_names":     X.columns.tolist(),
            "enable_categorical": _SUPPORTS_NATIVE_CATEGORICAL,
        }
        return xgb.DMatrix(**kwargs)

    def _build_callbacks(
        self,
        has_val:     bool,
        eval_metric: Union[str, list[str]],
    ) -> tuple[list, dict]:
        """
        构建 callbacks 列表与 evals_result 容器。

        Returns
        -------
        (callbacks, evals_result)
        """
        evals_result: dict = {}
        callbacks: list    = []

        # ---- 日志 callback ----
        if self.log_period > 0:
            callbacks.append(
                xgb.callback.EvaluationMonitor(period=self.log_period)
            )

        # ---- Early Stopping callback ----
        if has_val and self.early_stopping_rounds is not None:
            # 自动判断 maximize 方向
            maximize = self._infer_maximize(eval_metric)

            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds          = self.early_stopping_rounds,
                    metric_name     = self._primary_eval_metric(eval_metric),
                    data_name       = "val",       # 监控 val 集指标
                    maximize        = maximize,
                    save_best       = True,        # 自动保留最优 iteration 的 booster
                    min_delta       = 1e-7,        # 微小改善也算（防浮点舍入漏停）
                )
            )
        elif not has_val and self.early_stopping_rounds is not None:
            warnings.warn(
                "设置了 early_stopping_rounds 但未传入 X_val/y_val，"
                "Early Stopping 已自动禁用。",
                UserWarning,
                stacklevel=3,
            )

        return callbacks, evals_result

    def _infer_maximize(self, eval_metric: Union[str, list[str]]) -> bool:
        """根据 eval_metric 名称自动推断优化方向。"""
        if self.maximize is not None:
            return self.maximize
        primary = self._primary_eval_metric(eval_metric)
        # 去掉 @k 后缀（如 "ndcg@5" → "ndcg"）
        base_metric = primary.split("@")[0].lower()
        return base_metric in _HIGHER_IS_BETTER_METRICS

    @staticmethod
    def _primary_eval_metric(eval_metric: Union[str, list[str]]) -> str:
        """取第一个（主）eval metric 的名称。"""
        if isinstance(eval_metric, list):
            return eval_metric[0] if eval_metric else "rmse"
        return eval_metric

    @staticmethod
    def _default_eval_metric(task: str) -> str:
        """按任务类型返回合理的默认 eval_metric。"""
        defaults = {
            "binary":        "logloss",
            "multiclass":    "mlogloss",
            "regression":    "rmse",
            "regression_l1": "mae",
            "huber":         "rmse",
            "logistic":      "logloss",
            "poisson":       "poisson-nloglik",
            "tweedie":       "tweedie-nloglik@1.5",
            "rank":          "map",
            "rank_ndcg":     "ndcg",
        }
        return defaults.get(task, "rmse")

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        推理时严格校验并对齐特征列顺序。
        多余列静默丢弃；缺失列抛出 ValueError（禁止静默填 0 掩盖 bug）。
        """
        if not self.feature_names:
            return X
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(
                f"推理数据缺少以下特征列：{sorted(missing)}\n"
                f"请检查特征工程流水线是否与训练时一致。"
            )
        return X[self.feature_names]

    # ------------------------------------------------------------------
    # Optuna 搜索空间建议
    # ------------------------------------------------------------------

    @staticmethod
    def suggest_params(trial) -> dict:
        """
        Optuna trial 参数建议，可直接传入 optuna_tuner。

        Usage
        -----
        >>> study.optimize(lambda t: objective(t, XGBModel.suggest_params(t)), ...)
        """
        return {
            "max_depth":          trial.suggest_int("max_depth", 3, 12),
            "eta":                trial.suggest_float("eta", 1e-3, 0.1, log=True),
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bynode":   trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 100),
            "gamma":              trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "alpha":              trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "lambda":             trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "max_delta_step":     trial.suggest_int("max_delta_step", 0, 10),
            "grow_policy":        trial.suggest_categorical(
                                      "grow_policy", ["depthwise", "lossguide"]
                                  ),
        }

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"XGBModel("
            f"name={self.name!r}, "
            f"task={self.task!r}, "
            f"status={status}, "
            f"best_iter={self.best_iteration}, "
            f"n_features={len(self.feature_names)})"
        )
