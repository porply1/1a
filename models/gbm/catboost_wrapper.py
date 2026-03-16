"""
models/gbm/catboost_wrapper.py
-------------------------------
CatBoost 原生 API Wrapper，继承 BaseModel。

设计要点：
  1. 使用 CatBoost 基类（非 Classifier/Regressor sklearn 子类）+ Pool 数据格式
     → 完整保留原生 API 特性：自定义损失、eval_metric、monotone_constraints 等
  2. 类别特征王者：Pool(cat_features=...) 接收列名，CatBoost 自动完成内部编码
     → 无需 Label Encoding，支持训练集未见过的 category 值（test-time 不报错）
  3. Early Stopping：od_type='Iter' + od_wait 原生过拟合检测，
     配合 use_best_model=True 自动保留最优迭代权重
  4. 序列化：model.save_model(.cbm/.json) + .meta.json 元数据双轨
     → .cbm 二进制（最快加载，最小体积）；.json 可读可 diff
  5. feature_importance 支持四种类型：
     PredictionValuesChange / LossFunctionChange / Interaction / ShapValues
  6. 三剑客对称原则：与 LGBMModel/XGBModel 接口完全镜像，可无缝替换
  7. 内置 Optuna 搜索空间 suggest_params()，与调参层无缝对接

与前两把武器的对称关系
----------------------
  lgb.Dataset / xgb.DMatrix   ←→  catboost.Pool
  lgb.train / xgb.train       ←→  CatBoost.fit（原生基类调用）
  lgb.early_stopping          ←→  od_type='Iter', od_wait, use_best_model=True
  booster.feature_importance  ←→  model.get_feature_importance(type=...)
  .txt / .json                ←→  .cbm / .json（可选格式）

典型用法
--------
>>> from models.gbm.catboost_wrapper import CatBoostModel
>>> from core import CrossValidatorTrainer, MetricConfig
>>> from data.splitter import get_cv, CVConfig
>>> from sklearn.metrics import roc_auc_score

>>> model = CatBoostModel(
...     params={"depth": 8, "learning_rate": 0.03},
...     task="binary",
...     iterations=2000,
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
    from catboost import CatBoost, Pool, EFstrType
    import catboost as cb

    _CB_VERSION = tuple(
        int(x) for x in cb.__version__.split(".")[:2]
        if x.isdigit()
    )
except ImportError as e:
    raise ImportError(
        "CatBoost 未安装，请执行：pip install catboost"
    ) from e

from core.base_model import BaseModel, ModelState


# ---------------------------------------------------------------------------
# 任务映射表
# ---------------------------------------------------------------------------

# 竞赛任务名 → CatBoost loss_function 名称
_TASK_TO_LOSS: dict[str, str] = {
    # 分类
    "binary":        "Logloss",          # 二分类（等效 CrossEntropy）
    "crossentropy":  "CrossEntropy",     # 二分类概率标签版本
    "multiclass":    "MultiClass",       # 多分类（softmax）
    "multilogloss":  "MultiClassOneVsAll",  # One-vs-All 多分类
    # 回归
    "regression":    "RMSE",             # 均方根误差（默认回归）
    "mae":           "MAE",              # 平均绝对误差
    "mape":          "MAPE",             # 平均绝对百分比误差
    "huber":         "Huber:delta=1.35", # Huber 损失
    "poisson":       "Poisson",          # 泊松回归（计数）
    "tweedie":       "Tweedie:variance_power=1.5",
    "quantile":      "Quantile:alpha=0.5",  # 分位数回归
    # 排序
    "rank":          "YetiRank",         # CatBoost 特色排序损失
    "rank_pairwise": "PairLogit",        # Pairwise 排序
    "rank_ndcg":     "NDCG",
}

# 分类任务集合（predict_proba 时返回概率）
_CLF_TASKS = {"binary", "crossentropy", "multiclass", "multilogloss"}

# feature_importance 合法类型
_FI_TYPES = {
    "PredictionValuesChange",  # 默认，快速，类似 split
    "LossFunctionChange",      # 更精确，类似 gain（需要 eval_set）
    "Interaction",             # 特征交互重要性
    "ShapValues",              # SHAP 值（需要额外时间）
}

# 合法序列化格式
_SAVE_FORMATS = {"cbm", "json", "onnx", "cpp", "python"}

# 越大越好的 eval_metric（用于 verbose 日志方向标注）
_HIGHER_IS_BETTER = {
    "AUC", "Accuracy", "F1", "MCC", "NDCG", "MAP",
    "R2", "PRAUC", "BalancedAccuracy",
}


# ---------------------------------------------------------------------------
# CatBoostModel
# ---------------------------------------------------------------------------

class CatBoostModel(BaseModel):
    """
    CatBoost 原生 API 封装（CatBoost 基类 + Pool）。

    Parameters
    ----------
    params : dict | None
        CatBoost 训练参数。与 DEFAULT_PARAMS 合并，用户参数优先。
        不需要设置 loss_function，通过 task 参数自动推断注入。
    task : str
        任务类型，决定 loss_function 与 predict 行为。
        可选值见 _TASK_TO_LOSS 的键。默认：'regression'
    iterations : int
        最大迭代轮数（即树的棵数上限）。默认：1000
    early_stopping_rounds : int | None
        od_wait 参数：连续多少轮 eval metric 不改善时停止。
        None 表示禁用 Early Stopping。默认：100
    eval_metric : str | None
        监控指标名称（CatBoost 格式，如 'AUC', 'RMSE'）。
        None 时按 task 自动推断。
    log_period : int
        每隔多少轮打印训练日志。0 = 完全静默。默认：100
    fi_type : str
        feature_importance 计算方式：
        'PredictionValuesChange'（快速）/ 'LossFunctionChange'（精确，需 eval_set）
        默认：'LossFunctionChange'
    save_format : str
        序列化格式：'cbm'（二进制，最快）或 'json'（可读）。
        默认：'cbm'
    num_class : int | None
        多分类类别数，task='multiclass' 时必须设置。
    name : str | None
        模型名称，用于日志和文件命名。
    seed : int
        全局随机种子。

    Attributes
    ----------
    booster : CatBoost | None
        训练完成后的原生 CatBoost 对象。
    best_iteration : int
        Early Stopping 停止时的最优迭代轮数。
    feature_names : list[str]
        训练时的特征名列表（按顺序）。
    categorical_features : list[str]
        自动识别到的 category dtype 列名。
    text_features : list[str]
        自动识别到的文本列名（CatBoost 原生文本特征支持）。
    """

    # ------------------------------------------------------------------
    # 竞赛稳健默认参数
    # ------------------------------------------------------------------
    DEFAULT_PARAMS: dict = {
        # 树结构（对称树！depth 每+1 开销翻倍，竞赛建议 6~10）
        "depth":                6,       # 对称树深度，等效于 max_depth
        "min_data_in_leaf":     1,       # 叶子最少样本数
        "max_ctr_complexity":   4,       # CTR 组合特征的最大特征数

        # 学习率
        "learning_rate":        0.05,    # CatBoost 内部有自适应学习率，但竞赛建议手设

        # 随机化（对称树的随机化机制与 LGBM/XGB 不同）
        "rsm":                  0.8,     # Random Subspace Method，等效 colsample_bytree
        "subsample":            0.8,     # 行采样比例（bootstrap_type='Bernoulli' 时生效）
        "bootstrap_type":       "Bernoulli",  # Bernoulli 允许 subsample 参数

        # 正则化
        "l2_leaf_reg":          3.0,     # L2 正则（等效 LGBM lambda_l2 / XGB lambda）
        "random_strength":      1.0,     # 分裂候选的随机扰动强度（防过拟合）

        # 类别特征处理（CatBoost 王者特性）
        "one_hot_max_size":     2,       # 唯一值 ≤ 2 的类别特征用 one-hot，其余用 CTR
        "cat_features_border_count": 128, # 数值型特征分箱数（越大越精确）

        # 输出控制
        "verbose":              0,       # 0 = 静默（由 log_period 独立控制）
        "allow_writing_files":  False,   # 禁止 CatBoost 在训练时写临时文件

        # 硬件
        "thread_count":         -1,      # -1 = 全部 CPU 核
        "task_type":            "CPU",   # 'GPU' 需要 GPU 版 CatBoost
    }

    def __init__(
        self,
        params:                Optional[dict]  = None,
        task:                  str             = "regression",
        iterations:            int             = 1000,
        early_stopping_rounds: Optional[int]   = 100,
        eval_metric:           Optional[str]   = None,
        log_period:            int             = 100,
        fi_type:               str             = "LossFunctionChange",
        save_format:           str             = "cbm",
        num_class:             Optional[int]   = None,
        name:                  Optional[str]   = None,
        seed:                  int             = 42,
    ):
        # ---- 校验 ----
        if task not in _TASK_TO_LOSS:
            raise ValueError(
                f"不支持的 task='{task}'。"
                f"可选：{list(_TASK_TO_LOSS.keys())}"
            )
        if fi_type not in _FI_TYPES:
            raise ValueError(
                f"fi_type 必须为 {_FI_TYPES}，收到：'{fi_type}'"
            )
        if save_format not in _SAVE_FORMATS:
            raise ValueError(
                f"save_format 必须为 {_SAVE_FORMATS}，收到：'{save_format}'"
            )
        if task in ("multiclass", "multilogloss") and num_class is None:
            warnings.warn(
                f"task='{task}' 但未设置 num_class，"
                "请确保 params 中包含 'classes_count' 键。",
                UserWarning,
                stacklevel=2,
            )

        self.task                  = task
        self.iterations            = iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric           = eval_metric or self._default_eval_metric(task)
        self.log_period            = log_period
        self.fi_type               = fi_type
        self.save_format           = save_format
        self.num_class             = num_class

        # 运行时属性
        self.booster:             Optional[CatBoost] = None
        self.best_iteration:      int                = 0
        self.feature_names:       list[str]          = []
        self.categorical_features: list[str]         = []
        self.text_features:       list[str]          = []
        self._train_pool:         Optional[Pool]     = None  # 存留供 LFC 计算

        # 父类初始化（含 _merge_params + seed 注入）
        super().__init__(
            params=params,
            name=name or f"catboost_{task}",
            seed=seed,
        )

        # 合并后注入 loss_function / classes_count / iterations / eval_metric
        self.params["loss_function"] = _TASK_TO_LOSS[task]
        self.params["iterations"]    = iterations
        self.params["eval_metric"]   = self.eval_metric
        if num_class is not None:
            self.params["classes_count"] = num_class

    # ------------------------------------------------------------------
    # seed 键名（CatBoost 使用 random_seed）
    # ------------------------------------------------------------------

    def _seed_key(self) -> str:
        return "random_seed"

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
    ) -> "CatBoostModel":
        """
        使用 CatBoost 原生基类的 fit() 方法训练模型。

        Parameters
        ----------
        X_tr, y_tr  : 训练集特征与标签
        X_val, y_val : 验证集（用于 early stopping 和 eval 监控）
        **kwargs     : 透传给 CatBoost.fit() 的额外参数（如 sample_weight）
        """
        t0 = time.perf_counter()

        # ---- 1. 识别特征类型 ----
        self.feature_names        = X_tr.columns.tolist()
        self.categorical_features = self._detect_categoricals(X_tr)
        self.text_features        = self._detect_text_features(X_tr)

        # ---- 2. 构建 Pool ----
        train_pool = self._make_pool(X_tr, y_tr)
        self._train_pool = train_pool   # 保留用于 LossFunctionChange 计算

        eval_set: Optional[Pool] = None
        if X_val is not None and y_val is not None:
            eval_set = self._make_pool(X_val, y_val)

        # ---- 3. 构建训练参数（含 early stopping）----
        fit_params = self._build_fit_params(has_val=(eval_set is not None))

        # ---- 4. 初始化并训练 ----
        self.booster = CatBoost(fit_params)

        fit_kwargs: dict = {
            "X":         train_pool,
            "eval_set":  eval_set,
            "verbose":   self.log_period if self.log_period > 0 else False,
            "plot":      False,
        }
        # 透传合法 kwargs（如 sample_weight、baseline 等）
        _allowed_kwargs = {"sample_weight", "baseline", "init_model"}
        for k in _allowed_kwargs:
            if k in kwargs:
                fit_kwargs[k] = kwargs[k]

        self.booster.fit(**fit_kwargs)

        # ---- 5. 最优轮数 ----
        self.best_iteration = self.booster.get_best_iteration() or (self.iterations - 1)

        # ---- 6. 状态更新 ----
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
        pool = self._make_pool(self._align_features(X), label=None)
        if self.task in _CLF_TASKS:
            raw = self.booster.predict(pool, prediction_type="Class")
            return raw.astype(int).flatten()
        return self.booster.predict(pool, prediction_type="RawFormulaVal").flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        概率输出：
          - binary / crossentropy → shape (n,)，正类概率
          - multiclass            → shape (n, num_class)
          - 其他任务              → 等同 predict()（原始预测值）
        """
        self._assert_fitted()
        pool = self._make_pool(self._align_features(X), label=None)

        if self.task in _CLF_TASKS:
            proba = self.booster.predict(pool, prediction_type="Probability")
            # binary：(n, 2) → 取正类列
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba

        # 非分类任务：返回原始预测
        return self.booster.predict(pool, prediction_type="RawFormulaVal").flatten()

    # ------------------------------------------------------------------
    # feature_importance
    # ------------------------------------------------------------------

    @property
    def feature_importance(self) -> pd.Series:
        """
        返回归一化到 [0,1] 的特征重要性 pd.Series。
        index 为特征名，name 为 fi_type。

        注意：LossFunctionChange 需要 eval_set（训练时必须传入 X_val）。
        若 fi_type='LossFunctionChange' 但无 eval_set，自动回退到
        PredictionValuesChange 并发出 UserWarning。
        """
        self._assert_fitted()
        importance = self._compute_importance(self.fi_type)
        total = importance.sum()
        normalized = importance / total if total > 0 else importance

        return pd.Series(
            normalized,
            index=self.feature_names,
            name=self.fi_type,
        )

    def feature_importance_dataframe(self) -> pd.DataFrame:
        """
        返回 PredictionValuesChange 与 LossFunctionChange 双视角 DataFrame。
        按 LossFunctionChange 降序（如不可用则按 PredictionValuesChange）。
        """
        self._assert_fitted()
        pvc = self._compute_importance("PredictionValuesChange")
        df  = pd.DataFrame({
            "feature": self.feature_names,
            "PredictionValuesChange": pvc,
        })

        # LossFunctionChange 需要有 eval_set，尝试计算
        lfc = self._try_compute_lfc()
        if lfc is not None:
            df["LossFunctionChange"] = lfc
            sort_col = "LossFunctionChange"
        else:
            sort_col = "PredictionValuesChange"

        return df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        保存 CatBoost 模型与 Wrapper 元数据。

        生成文件：
          <path>.<save_format>    — CatBoost 原生格式（.cbm 或 .json）
          <path>.meta.json        — Wrapper 状态（特征名、任务类型等）
        """
        self._assert_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # ---- 原生模型文件 ----
        model_path = path.with_suffix(f".{self.save_format}")
        self.booster.save_model(
            str(model_path),
            format=self.save_format,
        )

        # ---- 元数据 ----
        meta = {
            "name":                  self.name,
            "task":                  self.task,
            "iterations":            self.iterations,
            "best_iteration":        self.best_iteration,
            "early_stopping_rounds": self.early_stopping_rounds,
            "eval_metric":           self.eval_metric,
            "fi_type":               self.fi_type,
            "save_format":           self.save_format,
            "num_class":             self.num_class,
            "seed":                  self.seed,
            "feature_names":         self.feature_names,
            "categorical_features":  self.categorical_features,
            "text_features":         self.text_features,
            "params":                self.params,
            "catboost_version":      cb.__version__,
        }
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.state = ModelState.SAVED

    def load(self, path: Union[str, Path]) -> "CatBoostModel":
        """
        从磁盘恢复 CatBoost 模型及 Wrapper 元数据。

        Parameters
        ----------
        path : .cbm/.json 文件路径 或不含后缀的基础路径均可。
        """
        path      = Path(path)
        meta_path = path.with_suffix(".meta.json")

        # ---- 读取元数据（先获取 save_format 定位 booster 文件）----
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            fmt = meta.get("save_format", self.save_format)
        else:
            warnings.warn(
                f"未找到元数据文件 {meta_path}，"
                "feature_names 等属性可能不完整。",
                UserWarning,
                stacklevel=2,
            )
            meta = {}
            fmt  = "cbm"   # 默认尝试 .cbm

        model_path = path.with_suffix(f".{fmt}")
        if not model_path.exists():
            # fallback：尝试另一格式
            alt_fmt  = "json" if fmt == "cbm" else "cbm"
            alt_path = path.with_suffix(f".{alt_fmt}")
            if alt_path.exists():
                model_path = alt_path
                fmt        = alt_fmt
            else:
                raise FileNotFoundError(
                    f"未找到模型文件：{model_path} 或 {alt_path}"
                )

        # ---- 恢复原生模型 ----
        self.booster = CatBoost()
        self.booster.load_model(str(model_path), format=fmt)

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
            self.text_features        = meta.get("text_features",        [])
            self.params               = meta.get("params",               self.params)
        else:
            # 降级：从模型对象中获取特征名
            self.feature_names = list(
                self.booster.feature_names_ or []
            )

        self._fitted = True
        self.state   = ModelState.FITTED
        return self

    # ------------------------------------------------------------------
    # clone（携带非 params 属性）
    # ------------------------------------------------------------------

    def clone(self) -> "CatBoostModel":
        return CatBoostModel(
            params                = self.get_params(),
            task                  = self.task,
            iterations            = self.iterations,
            early_stopping_rounds = self.early_stopping_rounds,
            eval_metric           = self.eval_metric,
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
        """识别 dtype == 'category' 的列。CatBoost 接受列名（非 index）。"""
        return [col for col in X.columns if X[col].dtype.name == "category"]

    @staticmethod
    def _detect_text_features(X: pd.DataFrame) -> list[str]:
        """
        识别潜在文本列（object dtype，平均字符长度 > 50）。
        CatBoost 原生支持文本特征（内部做 bag-of-words + embedding）。
        """
        text_cols = []
        for col in X.columns:
            if X[col].dtype == object:
                mean_len = X[col].dropna().astype(str).str.len().mean()
                if mean_len and mean_len > 50:
                    text_cols.append(col)
        return text_cols

    def _make_pool(
        self,
        X:     pd.DataFrame,
        label: Optional[pd.Series],
    ) -> Pool:
        """
        构建 CatBoost Pool。

        设计细节：
          - cat_features 传列名（字符串），CatBoost 内部自动映射到索引
          - text_features 仅在有文本列时传入（CatBoost >= 0.26 支持）
          - 推理时 label=None，Pool 正常工作（CatBoost 允许无标签 Pool 用于 predict）
        """
        pool_kwargs: dict = {
            "data":          X,
            "label":         label,
            "cat_features":  self.categorical_features if self.categorical_features else None,
            "feature_names": self.feature_names if self.feature_names else None,
        }
        # 文本特征（需要 CatBoost >= 0.26）
        if self.text_features:
            try:
                pool_kwargs["text_features"] = self.text_features
            except TypeError:
                # 旧版 CatBoost 不支持 text_features 参数，静默降级
                pass

        return Pool(**pool_kwargs)

    def _build_fit_params(self, has_val: bool) -> dict:
        """
        构建传入 CatBoost() 构造函数的完整参数字典。
        CatBoost 的 early stopping 通过参数而非 callback 控制。
        """
        fit_params = dict(self.params)   # 深拷贝，防止污染 self.params

        # Early Stopping（需要有验证集）
        if has_val and self.early_stopping_rounds is not None:
            fit_params["od_type"]        = "Iter"   # 基于迭代轮数的过拟合检测
            fit_params["od_wait"]        = self.early_stopping_rounds
            fit_params["use_best_model"] = True     # 自动保留最优迭代权重（关键！）
        elif not has_val and self.early_stopping_rounds is not None:
            warnings.warn(
                "设置了 early_stopping_rounds 但未传入 X_val/y_val，"
                "Early Stopping 已自动禁用。",
                UserWarning,
                stacklevel=3,
            )
            # 清除 od 参数，防止无验证集时触发异常
            fit_params.pop("od_type", None)
            fit_params.pop("od_wait", None)
            fit_params["use_best_model"] = False

        return fit_params

    def _compute_importance(self, fi_type: str) -> np.ndarray:
        """
        调用 booster.get_feature_importance()，返回原始（未归一化）重要性数组。
        LossFunctionChange 需要 train_pool，不可用时自动回退。
        """
        if fi_type == "LossFunctionChange":
            return self._try_compute_lfc() or self._compute_pvc()
        elif fi_type == "PredictionValuesChange":
            return self._compute_pvc()
        elif fi_type == "ShapValues":
            # ShapValues 返回 (n_samples, n_features + 1)，取均值绝对值
            if self._train_pool is None:
                warnings.warn(
                    "ShapValues 需要训练集 Pool，"
                    "将回退到 PredictionValuesChange。",
                    UserWarning,
                    stacklevel=2,
                )
                return self._compute_pvc()
            shap_vals = self.booster.get_feature_importance(
                data=self._train_pool,
                type=EFstrType.ShapValues,
            )
            return np.abs(shap_vals[:, :-1]).mean(axis=0)
        else:
            return self._compute_pvc()

    def _compute_pvc(self) -> np.ndarray:
        """PredictionValuesChange：不需要 eval_set，最快。"""
        return self.booster.get_feature_importance(
            type=EFstrType.PredictionValuesChange
        )

    def _try_compute_lfc(self) -> Optional[np.ndarray]:
        """尝试计算 LossFunctionChange，若无 train_pool 则返回 None。"""
        if self._train_pool is None:
            return None
        try:
            return self.booster.get_feature_importance(
                data=self._train_pool,
                type=EFstrType.LossFunctionChange,
            )
        except Exception as e:
            warnings.warn(
                f"LossFunctionChange 计算失败（{e}），"
                "将回退到 PredictionValuesChange。",
                UserWarning,
                stacklevel=2,
            )
            return None

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """推理时严格校验并对齐特征列顺序。缺失列抛出 ValueError。"""
        if not self.feature_names:
            return X
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(
                f"推理数据缺少以下特征列：{sorted(missing)}\n"
                f"请检查特征工程流水线是否与训练时一致。"
            )
        return X[self.feature_names]

    @staticmethod
    def _default_eval_metric(task: str) -> str:
        """按任务类型返回合理的默认 eval_metric（CatBoost 格式）。"""
        defaults = {
            "binary":        "AUC",
            "crossentropy":  "AUC",
            "multiclass":    "Accuracy",
            "multilogloss":  "Accuracy",
            "regression":    "RMSE",
            "mae":           "MAE",
            "mape":          "MAPE",
            "huber":         "RMSE",
            "poisson":       "Poisson:variant=mean",
            "tweedie":       "RMSE",
            "quantile":      "MAE",
            "rank":          "NDCG",
            "rank_pairwise": "AUC",
            "rank_ndcg":     "NDCG",
        }
        return defaults.get(task, "RMSE")

    # ------------------------------------------------------------------
    # Optuna 搜索空间建议
    # ------------------------------------------------------------------

    @staticmethod
    def suggest_params(trial) -> dict:
        """
        Optuna trial 参数建议，可直接传入 optuna_tuner。

        Usage
        -----
        >>> study.optimize(lambda t: objective(t, CatBoostModel.suggest_params(t)), ...)
        """
        bootstrap_type = trial.suggest_categorical(
            "bootstrap_type", ["Bernoulli", "MVS", "Bayesian"]
        )
        params = {
            "depth":               trial.suggest_int("depth", 4, 10),
            "learning_rate":       trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            "random_strength":     trial.suggest_float("random_strength", 1e-2, 10.0, log=True),
            "rsm":                 trial.suggest_float("rsm", 0.5, 1.0),
            "min_data_in_leaf":    trial.suggest_int("min_data_in_leaf", 1, 100),
            "one_hot_max_size":    trial.suggest_int("one_hot_max_size", 2, 25),
            "max_ctr_complexity":  trial.suggest_int("max_ctr_complexity", 1, 8),
            "bootstrap_type":      bootstrap_type,
        }
        # Bayesian bootstrap 无 subsample 参数
        if bootstrap_type in ("Bernoulli", "MVS"):
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        return params

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status   = "fitted" if self._fitted else "unfitted"
        n_cat    = len(self.categorical_features)
        n_text   = len(self.text_features)
        return (
            f"CatBoostModel("
            f"name={self.name!r}, "
            f"task={self.task!r}, "
            f"status={status}, "
            f"best_iter={self.best_iteration}, "
            f"n_features={len(self.feature_names)}, "
            f"n_cat={n_cat}, "
            f"n_text={n_text})"
        )
