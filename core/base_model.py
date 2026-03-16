"""
core/base_model.py
------------------
所有模型 Wrapper 的抽象基类。

设计哲学：
  - 定义"契约"而非"实现"：所有子类必须实现规定的接口
  - 参数管理标准化：params 通过 dict 注入，支持 yaml 驱动
  - 序列化内置：save/load 是一等公民，而不是事后补丁
  - 特征重要性统一格式：pd.Series(index=feature_names)

子类实现示例
-----------
>>> class LGBMModel(BaseModel):
...     def fit(self, X_tr, y_tr, X_val=None, y_val=None, **kwargs):
...         self._model = lgb.train(self.params, ...)
...         self._fitted = True
...     def predict(self, X):
...         return self._model.predict(X)
...     @property
...     def feature_importance(self):
...         return pd.Series(self._model.feature_importance(), index=X.columns)
...     def save(self, path): ...
...     def load(self, path): ...
"""

from __future__ import annotations

import abc
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 模型状态枚举
# ---------------------------------------------------------------------------

class ModelState:
    CREATED  = "created"   # 刚实例化，未训练
    FITTED   = "fitted"    # 已在某一折上训练完毕
    SAVED    = "saved"     # 已持久化到磁盘


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseModel(abc.ABC):
    """
    所有竞赛模型 Wrapper 的统一抽象基类。

    Parameters
    ----------
    params : dict | None
        模型超参数字典。None 时各子类使用自身默认值。
    name : str | None
        模型的可读名称（用于日志和文件命名）。
        None 时自动使用类名小写。
    seed : int
        随机种子，注入到 params 时子类负责映射到正确键名。

    Attributes
    ----------
    params : dict
        当前生效的超参数（深拷贝，防止外部篡改）。
    name : str
        模型名称。
    seed : int
        随机种子。
    state : str
        当前状态，取值见 ModelState。
    fit_time : float
        最近一次 fit() 的耗时（秒）。
    """

    # 子类可覆盖：模型默认超参数
    DEFAULT_PARAMS: dict = {}

    def __init__(
        self,
        params: Optional[dict] = None,
        name: Optional[str] = None,
        seed: int = 42,
    ):
        self.seed   = seed
        self.name   = name or self.__class__.__name__.lower()
        self.params = self._merge_params(params or {})
        self.state  = ModelState.CREATED

        # 运行时填充
        self.fit_time: float = 0.0
        self._fitted: bool   = False

    # ------------------------------------------------------------------
    # 必须实现的抽象接口
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series]    = None,
        **kwargs,
    ) -> "BaseModel":
        """
        在给定数据上训练模型。

        Parameters
        ----------
        X_tr : pd.DataFrame   训练特征
        y_tr : pd.Series      训练标签
        X_val : pd.DataFrame  验证特征（可选，用于 early stopping）
        y_val : pd.Series     验证标签（可选）
        **kwargs              透传给底层训练 API 的额外参数

        Returns
        -------
        self（支持链式调用）
        """
        ...

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        对输入数据进行推理。

        Returns
        -------
        np.ndarray, shape (n_samples,) 或 (n_samples, n_classes)
        """
        ...

    @abc.abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """将模型持久化到 path（子类自行选择 pickle/joblib/booster 格式）。"""
        ...

    @abc.abstractmethod
    def load(self, path: Union[str, Path]) -> "BaseModel":
        """从 path 加载模型，返回 self。"""
        ...

    # ------------------------------------------------------------------
    # 可选覆盖的接口（非 abstract，有默认兜底实现）
    # ------------------------------------------------------------------

    @property
    def feature_importance(self) -> pd.Series:
        """
        返回特征重要性。

        格式：pd.Series，index 为特征名，values 为重要性分数（已归一化到 [0,1]）。
        子类应覆盖此属性提供真实实现。
        默认实现返回空 Series（用于不支持特征重要性的模型）。
        """
        return pd.Series(dtype=float, name="importance")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        分类任务的概率输出。
        默认直接调用 predict()，子类可覆盖以返回 softmax 概率。
        """
        return self.predict(X)

    def get_params(self) -> dict:
        """返回当前超参数的深拷贝。"""
        return copy.deepcopy(self.params)

    def set_params(self, **params) -> "BaseModel":
        """更新超参数（链式调用）。"""
        self.params.update(params)
        return self

    def clone(self) -> "BaseModel":
        """
        返回一个参数相同但未训练的新实例（用于 CV 折间复用）。

        子类若有额外初始化逻辑，应覆盖此方法。
        """
        return self.__class__(
            params=self.get_params(),
            name=self.name,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # 参数管理
    # ------------------------------------------------------------------

    def _merge_params(self, user_params: dict) -> dict:
        """
        将用户传入的 params 与子类 DEFAULT_PARAMS 合并。
        用户参数优先级更高。同时将 seed 注入到正确的键（子类覆盖 _seed_key）。
        """
        merged = copy.deepcopy(self.DEFAULT_PARAMS)
        merged.update(user_params)
        # 注入随机种子（子类可覆盖 _seed_key 映射到正确键名）
        seed_key = self._seed_key()
        if seed_key and seed_key not in user_params:
            merged[seed_key] = self.seed
        return merged

    def _seed_key(self) -> Optional[str]:
        """
        子类覆盖此方法，返回超参数中随机种子的键名。
        例如 LightGBM 返回 "seed"，XGBoost 返回 "seed"，
        CatBoost 返回 "random_seed"。
        None 表示该模型不需要注入随机种子。
        """
        return None

    # ------------------------------------------------------------------
    # 参数指纹（用于缓存 key 生成）
    # ------------------------------------------------------------------

    def param_hash(self) -> str:
        """返回当前超参数的 MD5 哈希，可用于特征/模型缓存 key。"""
        serialized = json.dumps(self.params, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:8]

    # ------------------------------------------------------------------
    # 状态保护
    # ------------------------------------------------------------------

    def _assert_fitted(self):
        """在 predict/feature_importance 前调用，防止未训练即推理。"""
        if not self._fitted:
            raise RuntimeError(
                f"模型 '{self.name}' 尚未训练，请先调用 fit()。"
            )

    # ------------------------------------------------------------------
    # 计时装饰（供子类 fit 方法调用）
    # ------------------------------------------------------------------

    def _timed_fit(self, fit_fn, *args, **kwargs):
        """
        包装 fit 调用，记录耗时并更新 state。
        子类可在 fit() 中调用：return self._timed_fit(self._do_fit, ...)
        """
        t0 = time.perf_counter()
        result = fit_fn(*args, **kwargs)
        self.fit_time = time.perf_counter() - t0
        self._fitted = True
        self.state   = ModelState.FITTED
        return result

    # ------------------------------------------------------------------
    # dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"status={status}, "
            f"params={self.params})"
        )
