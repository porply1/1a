"""
models/gbm/lgbm_wrapper.py
--------------------------
LightGBM 原生 API Wrapper，继承 BaseModel。

设计要点：
  1. 全程使用 lgb.train（原生 API），不走 sklearn 接口
     → 支持原生 callbacks、自定义 eval、dart/goss 等高级特性
  2. category 列自动识别并传入 lgb.Dataset（无需手工 label encoding）
  3. Early Stopping 通过 lgb.early_stopping callback 实现
     → early_stopping_rounds 在 fit_params 或 DEFAULT_PARAMS 中控制
  4. 序列化使用 booster.save_model（文本格式，跨平台，可用 lgbm CLI 推理）
  5. feature_importance 支持三种模式：split / gain / shap（需安装 shap）
  6. 内置 log_evaluation callback 静默控制，避免训练日志刷屏
  7. num_boost_round 可在实例化时通过 params["n_estimators"] 或独立参数控制

典型用法
--------
>>> from models.gbm.lgbm_wrapper import LGBMModel
>>> from core import CrossValidatorTrainer, MetricConfig
>>> from data.splitter import get_cv, CVConfig
>>> from sklearn.metrics import roc_auc_score

>>> model = LGBMModel(
...     params={"num_leaves": 63, "learning_rate": 0.03},
...     task="binary",
... )
>>> cv = get_cv(CVConfig(strategy="stratified_kfold", n_splits=5))
>>> trainer = CrossValidatorTrainer(
...     model=model, cv=cv, task="binary",
...     metrics=[MetricConfig("auc", roc_auc_score, use_proba=True)],
... )
>>> result = trainer.fit(X_train, y_train, X_test=X_test)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "LightGBM 未安装，请执行：pip install lightgbm"
    ) from e

from core.base_model import BaseModel, ModelState


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# LightGBM 任务类型 → 对应 objective 键值
_TASK_TO_OBJECTIVE: dict[str, str] = {
    "binary":     "binary",
    "multiclass": "multiclass",
    "regression": "regression",
    "regression_l1": "regression_l1",
    "huber":      "huber",
    "mape":       "mape",
    "poisson":    "poisson",
    "tweedie":    "tweedie",
    "rank":       "lambdarank",
}

# 分类任务集合（predict 时使用 predict_proba）
_CLF_TASKS = {"binary", "multiclass", "rank"}

# feature_importance_type 合法值
_FI_TYPES = {"split", "gain"}


# ---------------------------------------------------------------------------
# LGBMModel
# ---------------------------------------------------------------------------

class LGBMModel(BaseModel):
    """
    LightGBM 原生 API 封装。

    Parameters
    ----------
    params : dict | None
        LightGBM 超参数。会与 DEFAULT_PARAMS 合并（用户参数优先）。
        不需要显式设置 objective，通过 task 参数自动推断并注入。
    task : str
        任务类型，决定 objective 和 predict 行为。
        可选：'binary' | 'multiclass' | 'regression' | 'regression_l1' |
               'huber' | 'mape' | 'poisson' | 'tweedie' | 'rank'
        默认：'regression'
    num_boost_round : int
        最大迭代轮数（即树的棵数上限）。
        若启用 early stopping，实际轮数可能更少。
        默认：1000
    early_stopping_rounds : int | None
        连续多少轮验证集指标不改善时停止。
        None 表示不启用 early stopping（不建议）。
        默认：100
    log_period : int
        每隔多少轮打印一次训练日志。0 表示完全静默。
        默认：100
    fi_type : str
        feature_importance 的计算方式：'split' 或 'gain'。
        默认：'gain'（更反映信息量，split 可能偏向高基数特征）
    num_class : int | None
        多分类任务的类别数。task='multiclass' 时必须设置。
    name : str | None
        模型名称，用于日志和文件命名。
    seed : int
        全局随机种子。

    Attributes
    ----------
    booster : lgb.Booster | None
        训练完成后的原生 Booster 对象。
    best_iteration : int
        Early stopping 停止时的最优迭代轮数。
    feature_names : list[str]
        训练时使用的特征名列表（按顺序）。
    categorical_features : list[str]
        自动识别到的 category 列名。
    """

    # ------------------------------------------------------------------
    # 竞赛常用稳健默认参数
    # ------------------------------------------------------------------
    DEFAULT_PARAMS: dict = {
        # 树结构
        "num_leaves":        31,       # 叶子数，主要复杂度控制旋钮
        "max_depth":         -1,       # -1 = 不限深度，由 num_leaves 隐式控制
        "min_child_samples": 20,       # 叶子最少样本数，防止过拟合
        "min_child_weight":  1e-3,     # 叶子最小 Hessian 和

        # 学习率与迭代
        "learning_rate":     0.05,     # 竞赛常用：粗调 0.05，细调 0.01~0.02

        # 随机化（防过拟合三件套）
        "feature_fraction":  0.8,      # 每棵树随机选 80% 特征（colsample_bytree）
        "bagging_fraction":  0.8,      # 每次迭代随机选 80% 样本（subsample）
        "bagging_freq":      1,        # 每 1 轮做一次 bagging

        # 正则化
        "lambda_l1":         0.1,      # L1 正则（稀疏化叶子权重）
        "lambda_l2":         0.1,      # L2 正则（收缩叶子权重）
        "min_gain_to_split": 0.0,      # 最小增益阈值（可调为 0.01+ 剪枝）
        "path_smooth":       0.0,      # 路径平滑（抑制深层叶子过拟合）

        # 硬件
        "n_jobs":            -1,       # 使用全部 CPU 核
        "device_type":       "cpu",    # 'gpu' 需要 GPU 版 LightGBM

        # 输出控制
        "verbose":           -1,       # 关闭 LightGBM 内部日志（由 callbacks 控制）
    }

    def __init__(
        self,
        params:                Optional[dict] = None,
        task:                  str            = "regression",
        num_boost_round:       int            = 1000,
        early_stopping_rounds: Optional[int]  = 100,
        log_period:            int            = 100,
        fi_type:               str            = "gain",
        num_class:             Optional[int]  = None,
        name:                  Optional[str]  = None,
        seed:                  int            = 42,
    ):
        # 校验
        if task not in _TASK_TO_OBJECTIVE:
            raise ValueError(
                f"不支持的 task='{task}'。"
                f"可选：{list(_TASK_TO_OBJECTIVE.keys())}"
            )
        if fi_type not in _FI_TYPES:
            raise ValueError(f"fi_type 必须为 {_FI_TYPES}，收到：'{fi_type}'")
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
        self.log_period            = log_period
        self.fi_type               = fi_type
        self.num_class             = num_class

        # 运行时属性
        self.booster:             Optional[lgb.Booster] = None
        self.best_iteration:      int                   = 0
        self.feature_names:       list[str]             = []
        self.categorical_features: list[str]            = []

        # 调用父类（执行参数合并与 seed 注入）
        super().__init__(
            params=params,
            name=name or f"lgbm_{task}",
            seed=seed,
        )

        # 合并后注入 objective 与 num_class
        self.params["objective"] = _TASK_TO_OBJECTIVE[task]
        if num_class is not None:
            self.params["num_class"] = num_class

    # ------------------------------------------------------------------
    # seed 键名（父类 _merge_params 使用）
    # ------------------------------------------------------------------

    def _seed_key(self) -> str:
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
    ) -> "LGBMModel":
        """
        使用 lgb.train 原生 API 训练模型。

        Parameters
        ----------
        X_tr, y_tr : 训练集特征与标签
        X_val, y_val : 验证集（用于 early stopping 和评估）
        **kwargs : 透传给 lgb.train 的额外参数（如 feval 自定义指标）
        """
        import time
        t0 = time.perf_counter()

        # ---- 1. 识别 category 列 ----
        self.categorical_features = self._detect_categoricals(X_tr)
        self.feature_names        = X_tr.columns.tolist()

        # ---- 2. 构建 lgb.Dataset ----
        train_set = self._make_dataset(X_tr, y_tr, reference=None)
        valid_sets  = [train_set]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_set = self._make_dataset(X_val, y_val, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("val")

        # ---- 3. 构建 callbacks ----
        callbacks = self._build_callbacks(has_val=(X_val is not None))

        # ---- 4. 提取 lgb.train 专属 kwargs，防止参数污染 ----
        train_kwargs = {
            "num_boost_round": kwargs.pop("num_boost_round", self.num_boost_round),
            "valid_sets":      valid_sets,
            "valid_names":     valid_names,
            "callbacks":       callbacks,
        }
        # feval（自定义评估函数）可通过 kwargs 传入
        if "feval" in kwargs:
            train_kwargs["feval"] = kwargs.pop("feval")

        # ---- 5. 训练 ----
        self.booster = lgb.train(
            params       = self.params,
            train_set    = train_set,
            **train_kwargs,
            **kwargs,
        )

        # ---- 6. 记录最优轮数 ----
        self.best_iteration = getattr(self.booster, "best_iteration", 0) or 0
        if self.best_iteration == 0:
            # 未启用 early stopping，直接用最大轮数
            self.best_iteration = self.num_boost_round

        # ---- 7. 更新状态 ----
        self.fit_time = time.perf_counter() - t0
        self._fitted  = True
        self.state    = ModelState.FITTED

        return self

    # ------------------------------------------------------------------
    # predict / predict_proba
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        回归任务：返回原始预测值（形状 (n,)）。
        分类任务：返回类别预测（argmax），形状 (n,)。
        注意：Stacking/AUC 计算请使用 predict_proba。
        """
        self._assert_fitted()
        raw = self._raw_predict(X)
        if self.task in _CLF_TASKS and raw.ndim == 2:
            return np.argmax(raw, axis=1)
        return raw

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        返回概率输出：
          - binary     → shape (n,)，正类概率
          - multiclass → shape (n, num_class)
          - regression → 等同 predict()
        """
        self._assert_fitted()
        raw = self._raw_predict(X)
        return raw

    def _raw_predict(self, X: pd.DataFrame) -> np.ndarray:
        """调用 booster.predict，统一处理特征对齐。"""
        X_aligned = self._align_features(X)
        return self.booster.predict(
            X_aligned,
            num_iteration=self.best_iteration,
        )

    # ------------------------------------------------------------------
    # feature_importance
    # ------------------------------------------------------------------

    @property
    def feature_importance(self) -> pd.Series:
        """
        返回归一化到 [0,1] 的特征重要性 pd.Series。
        index 为特征名，name 为 fi_type（'gain' 或 'split'）。
        """
        self._assert_fitted()
        importance = self.booster.feature_importance(importance_type=self.fi_type)
        total = importance.sum()
        if total == 0:
            normalized = importance.astype(float)
        else:
            normalized = importance / total

        return pd.Series(
            normalized,
            index=self.feature_names,
            name=self.fi_type,
        )

    def feature_importance_dataframe(self) -> pd.DataFrame:
        """
        返回按重要性降序排列的 DataFrame，含 split 和 gain 双列。
        方便 EDA 时对比两种视角。
        """
        self._assert_fitted()
        fi_split = self.booster.feature_importance(importance_type="split")
        fi_gain  = self.booster.feature_importance(importance_type="gain")
        df = pd.DataFrame({
            "feature": self.feature_names,
            "split":   fi_split,
            "gain":    fi_gain,
        })
        df["gain_normalized"]  = df["gain"]  / max(df["gain"].sum(),  1)
        df["split_normalized"] = df["split"] / max(df["split"].sum(), 1)
        return df.sort_values("gain", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        使用 LightGBM 原生文本格式保存 Booster。
        生成的 .txt 文件可被任意平台的 lgb.Booster(model_file=...) 加载，
        也兼容 lgbm CLI 推理。
        额外保存一份 .meta.json 记录 Wrapper 状态（特征名、任务类型等）。
        """
        self._assert_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 原生 booster 文件（.txt 文本格式）
        booster_path = path.with_suffix(".txt")
        self.booster.save_model(str(booster_path))

        # Wrapper 元数据（.meta.json）
        import json
        meta = {
            "name":                  self.name,
            "task":                  self.task,
            "num_boost_round":       self.num_boost_round,
            "best_iteration":        self.best_iteration,
            "early_stopping_rounds": self.early_stopping_rounds,
            "fi_type":               self.fi_type,
            "num_class":             self.num_class,
            "seed":                  self.seed,
            "feature_names":         self.feature_names,
            "categorical_features":  self.categorical_features,
            "params":                self.params,
        }
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.state = ModelState.SAVED

    def load(self, path: Union[str, Path]) -> "LGBMModel":
        """
        从 path 加载 Booster 及元数据，恢复完整 Wrapper 状态。

        Parameters
        ----------
        path : 传入 .txt 文件路径 或不含后缀的基础路径均可。
        """
        import json
        path = Path(path)
        booster_path = path.with_suffix(".txt")
        meta_path    = path.with_suffix(".meta.json")

        if not booster_path.exists():
            raise FileNotFoundError(f"Booster 文件不存在：{booster_path}")

        # 恢复原生 Booster
        self.booster = lgb.Booster(model_file=str(booster_path))

        # 恢复元数据（若存在）
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.name                  = meta.get("name", self.name)
            self.task                  = meta.get("task", self.task)
            self.best_iteration        = meta.get("best_iteration", 0)
            self.fi_type               = meta.get("fi_type", self.fi_type)
            self.num_class             = meta.get("num_class", self.num_class)
            self.feature_names         = meta.get("feature_names", [])
            self.categorical_features  = meta.get("categorical_features", [])
            self.params                = meta.get("params", self.params)
        else:
            warnings.warn(
                f"未找到元数据文件 {meta_path}，"
                "feature_names 等属性可能不完整。",
                UserWarning,
                stacklevel=2,
            )
            # 降级：从 booster 中获取特征名
            self.feature_names = self.booster.feature_name()

        self._fitted = True
        self.state   = ModelState.FITTED
        return self

    # ------------------------------------------------------------------
    # clone 覆盖（保留 task/num_boost_round 等非 params 参数）
    # ------------------------------------------------------------------

    def clone(self) -> "LGBMModel":
        return LGBMModel(
            params                = self.get_params(),
            task                  = self.task,
            num_boost_round       = self.num_boost_round,
            early_stopping_rounds = self.early_stopping_rounds,
            log_period            = self.log_period,
            fi_type               = self.fi_type,
            num_class             = self.num_class,
            name                  = self.name,
            seed                  = self.seed,
        )

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_categoricals(X: pd.DataFrame) -> list[str]:
        """
        自动识别 DataFrame 中 dtype=='category' 的列。
        LightGBM 原生支持 category 类型，无需手工 Label Encoding。
        """
        return [col for col in X.columns if X[col].dtype.name == "category"]

    def _make_dataset(
        self,
        X:         pd.DataFrame,
        y:         pd.Series,
        reference: Optional[lgb.Dataset],
    ) -> lgb.Dataset:
        """
        构建 lgb.Dataset，注入 category 列信息和特征名。

        Parameters
        ----------
        reference : lgb.Dataset | None
            验证集构建时传入训练集 Dataset 作为 reference，
            确保 bin 边界对齐（LightGBM 官方推荐）。
        """
        cat_features = self.categorical_features if self.categorical_features else "auto"
        return lgb.Dataset(
            data                = X,
            label               = y,
            categorical_feature = cat_features,
            feature_name        = X.columns.tolist(),
            reference           = reference,
            free_raw_data       = True,   # 训练后释放原始数据，节省内存
        )

    def _build_callbacks(self, has_val: bool) -> list:
        """
        构建 LightGBM callbacks 列表：
          - log_evaluation：控制日志频率（0 = 静默）
          - early_stopping：有验证集且设置了 rounds 时启用
        """
        callbacks = []

        # 日志频率控制
        if self.log_period > 0:
            callbacks.append(lgb.log_evaluation(period=self.log_period))
        else:
            callbacks.append(lgb.log_evaluation(period=0))  # 完全静默

        # Early Stopping（需要有验证集）
        if has_val and self.early_stopping_rounds is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds = self.early_stopping_rounds,
                    first_metric_only = True,   # 只看第一个 eval metric
                    verbose           = (self.log_period > 0),
                )
            )
        elif not has_val and self.early_stopping_rounds is not None:
            warnings.warn(
                "设置了 early_stopping_rounds 但未传入 X_val/y_val，"
                "Early Stopping 已自动禁用。",
                UserWarning,
                stacklevel=3,
            )

        return callbacks

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        推理时对齐特征列顺序与训练时一致。
        多余列静默丢弃，缺失列报错（而非静默填 0，避免掩盖 bug）。
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
    # 调参辅助：Optuna 搜索空间建议
    # ------------------------------------------------------------------

    @staticmethod
    def suggest_params(trial) -> dict:
        """
        Optuna trial 对象的参数建议函数，可直接传入 optuna_tuner。

        Usage
        -----
        >>> study.optimize(lambda t: objective(t, LGBMModel.suggest_params(t)), ...)
        """
        return {
            "num_leaves":        trial.suggest_int("num_leaves", 16, 255),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "path_smooth":       trial.suggest_float("path_smooth", 0.0, 1.0),
        }

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"LGBMModel("
            f"name={self.name!r}, "
            f"task={self.task!r}, "
            f"status={status}, "
            f"best_iter={self.best_iteration}, "
            f"n_features={len(self.feature_names)})"
        )
