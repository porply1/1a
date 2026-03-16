"""
features/engine.py
------------------
特征工程引擎：抽象框架 + 防泄露流水线 + 高频原子模块。

架构层级
--------
  BaseFeatureTransformer          ← 所有特征类的契约（抽象基类）
      │
      ├── TimeFeatureTransformer  ← 时序：Sin/Cos 周期编码 + 日历特征
      ├── GroupAggTransformer     ← 结构化：多维聚合统计
      ├── DiffTransformer         ← 时序：差分/移动平均
      ├── LagTransformer          ← 时序：Lag/Lead 特征
      ├── FreqEncoderTransformer  ← 结构化：频率编码（无泄露）
      └── TargetEncoderTransformer← 结构化：目标编码（强制折内防泄露）

  FeaturePipeline                 ← 编排器，顺序/并行执行 + 自动拼接
      └── fit_transform_oof()     ← 折内防泄露变换（最重要方法！）

防泄露核心原则
--------------
  ① 全局统计（如频率、中位数）：在 fit(X_tr) 上计算，transform(X_val) 只做查表
  ② 目标相关统计（Target Encoding）：严禁在全量训练集上全局拟合
     → 必须使用 fit_transform_oof()，基于 CV fold 的折内拟合
  ③ 时序特征（Lag/Diff）：严禁使用未来时间点数据，shift 方向必须 > 0

使用示例
--------
>>> from features.engine import FeaturePipeline, GroupAggTransformer, TimeFeatureTransformer

>>> pipeline = FeaturePipeline(
...     transformers=[
...         TimeFeatureTransformer(date_col="date"),
...         GroupAggTransformer(group_keys=["store_id"], agg_col="sales",
...                             agg_funcs=["mean","std","max"]),
...     ],
...     compress=True,
... )
>>> X_train_feats = pipeline.fit_transform(X_train)
>>> X_test_feats  = pipeline.transform(X_test)
"""

from __future__ import annotations

import abc
import hashlib
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from utils.memory import compress_dataframe, get_memory_usage_mb


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseFeatureTransformer(abc.ABC):
    """
    所有特征变换器的统一契约。

    生命周期
    --------
    fit(X, y)           → 从训练集学习统计量（均值、映射表等）
    transform(X)        → 无状态变换，只使用 fit 时计算的统计量
    fit_transform(X, y) → fit + transform 的便捷组合

    Attributes
    ----------
    name : str
        变换器名称，用于日志和生成特征列的前缀。
    feature_names_out : list[str]
        fit 之后记录的输出特征名列表。
    is_fitted : bool
        是否已完成 fit。
    """

    def __init__(self, name: Optional[str] = None):
        self.name              = name or self.__class__.__name__.lower()
        self.feature_names_out: list[str] = []
        self.is_fitted:         bool      = False
        self._fit_time:         float     = 0.0

    # --- 必须实现 ---

    @abc.abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "BaseFeatureTransformer":
        """从 X（和可选的 y）学习统计量，返回 self。"""
        ...

    @abc.abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        使用 fit 阶段的统计量变换 X。

        Returns
        -------
        pd.DataFrame，列名必须写入 self.feature_names_out。
        不得包含原始 X 的列（只返回新特征）。
        """
        ...

    # --- 默认实现 ---

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        t0 = time.perf_counter()
        self.fit(X, y)
        result = self.transform(X)
        self._fit_time = time.perf_counter() - t0
        return result

    def _assert_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.name} 尚未 fit，请先调用 fit() 或 fit_transform()。"
            )

    def _mark_fitted(self, feature_names: list[str]) -> None:
        self.feature_names_out = feature_names
        self.is_fitted         = True

    def get_feature_names(self) -> list[str]:
        self._assert_fitted()
        return self.feature_names_out

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"status={status}, "
            f"n_features_out={len(self.feature_names_out)})"
        )


# ---------------------------------------------------------------------------
# FeaturePipeline — 编排器
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """
    特征工程流水线编排器。

    接受一组 BaseFeatureTransformer，支持：
      - 顺序执行（默认）
      - 并行执行（parallel=True，适用于无依赖的独立变换器）
      - 自动拼接所有变换结果
      - 防泄露的折内目标编码（fit_transform_oof）
      - 内存压缩

    Parameters
    ----------
    transformers : list[BaseFeatureTransformer]
        变换器列表，按顺序执行（parallel=False 时）。
    compress : bool
        每步变换后是否压缩新列的 dtype。默认 True。
    parallel : bool
        True 时所有变换器并行执行（线程池），适用于相互独立的变换器。
        注意：如果变换器之间有特征依赖，请保持 parallel=False。
    max_workers : int
        parallel=True 时的线程数。默认 4。
    keep_original : bool
        True 时在输出中保留原始 X 的列。默认 True。
    verbose : bool
        是否打印每步执行日志。默认 True。
    """

    def __init__(
        self,
        transformers:  list[BaseFeatureTransformer],
        compress:      bool = True,
        parallel:      bool = False,
        max_workers:   int  = 4,
        keep_original: bool = True,
        verbose:       bool = True,
    ):
        if not transformers:
            raise ValueError("transformers 列表不能为空。")
        self.transformers  = transformers
        self.compress      = compress
        self.parallel      = parallel
        self.max_workers   = max_workers
        self.keep_original = keep_original
        self.verbose       = verbose

        self._all_feature_names: list[str] = []
        self.is_fitted = False

    # ------------------------------------------------------------------
    # fit / transform / fit_transform
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> "FeaturePipeline":
        """顺序对所有变换器调用 fit。"""
        self._log(f"Pipeline fit | {len(self.transformers)} 个变换器")
        for tf in self.transformers:
            t0 = time.perf_counter()
            tf.fit(X, y)
            elapsed = time.perf_counter() - t0
            self._log(f"  ✓ {tf.name:<35} fit={elapsed:.3f}s")
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        对 X 应用所有已 fit 的变换器，返回拼接后的 DataFrame。
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline 尚未 fit，请先调用 fit() 或 fit_transform()。")

        parts: list[pd.DataFrame] = (
            [X.reset_index(drop=True)] if self.keep_original else []
        )
        mem_before = get_memory_usage_mb(X)

        if self.parallel:
            new_parts = self._transform_parallel(X)
        else:
            new_parts = self._transform_sequential(X)

        parts.extend(new_parts)
        result = pd.concat(parts, axis=1)

        if self.compress:
            result = compress_dataframe(result, verbose=False)

        mem_after = get_memory_usage_mb(result)
        self._all_feature_names = result.columns.tolist()
        self._log(
            f"Pipeline transform 完成 | "
            f"shape={result.shape} | "
            f"内存: {mem_before:.1f}→{mem_after:.1f} MB"
        )
        return result

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    # ------------------------------------------------------------------
    # 防泄露折内变换（最核心方法）
    # ------------------------------------------------------------------

    def fit_transform_oof(
        self,
        X:      pd.DataFrame,
        y:      pd.Series,
        folds:  list[tuple[np.ndarray, np.ndarray]],
        X_test: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        折内防泄露变换（Out-of-Fold Feature Generation）。

        算法
        ----
        对每一折 (tr_idx, val_idx)：
          1. 在 X[tr_idx] 上 fit 所有标签相关变换器
          2. 用 fit 后的变换器 transform X[val_idx]（无泄露）
          3. 写入 X_train_feats[val_idx]

        对测试集：
          使用全量训练集 fit 一次（最大化信息利用），transform X_test

        参数
        ----
        X      : 训练集原始特征
        y      : 训练标签
        folds  : [(tr_idx, val_idx), ...] 由 BaseCVSplitter.split() 生成
        X_test : 测试集（可选），使用全量训练集 fit 的变换器转换

        Returns
        -------
        (X_train_feats, X_test_feats)
          X_train_feats : 所有折拼接后的训练特征（行顺序与原始 X 对齐）
          X_test_feats  : 测试集特征（或 None）

        防泄露保证
        ----------
        Target Encoding / GroupAgg（依赖 y）的变换器：
          在 transform(X_val) 时只能使用 fit(X_tr) 计算的统计量。
        非标签相关变换器（Time/Diff/Lag）：
          在全量 X 上 fit 一次，不存在泄露风险。
        """
        self._log(
            f"OOF 防泄露变换 | {len(folds)} 折 | "
            f"{len(self.transformers)} 个变换器"
        )

        # 区分标签相关与无关变换器
        leaky_tfs   = [tf for tf in self.transformers if getattr(tf, "requires_y", False)]
        noleaky_tfs = [tf for tf in self.transformers if not getattr(tf, "requires_y", False)]

        # 无关变换器在全量 X 上一次性 fit（无泄露）
        for tf in noleaky_tfs:
            tf.fit(X, y)
            self._log(f"  [全量fit] {tf.name}")

        # 初始化训练特征容器（行顺序与 X 对齐）
        n_rows = len(X)
        col_placeholder: dict[str, np.ndarray] = {}

        # -------- 折循环 --------
        for fold_i, (tr_idx, val_idx) in enumerate(folds):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_val      = X.iloc[val_idx]

            # 标签相关变换器：只用 X_tr fit，transform X_val（防泄露核心）
            fold_parts_val: list[pd.DataFrame] = []
            for tf in leaky_tfs:
                cloned_tf = deepcopy(tf)         # 每折独立实例，互不污染
                cloned_tf.fit(X_tr, y_tr)
                val_feats = cloned_tf.transform(X_val).reset_index(drop=True)
                fold_parts_val.append(val_feats)

            # 无关变换器：直接 transform（已在全量 fit）
            for tf in noleaky_tfs:
                val_feats = tf.transform(X_val).reset_index(drop=True)
                fold_parts_val.append(val_feats)

            # 写入对应行
            fold_val_df = pd.concat(fold_parts_val, axis=1)
            for col in fold_val_df.columns:
                if col not in col_placeholder:
                    col_placeholder[col] = np.full(n_rows, np.nan, dtype=np.float64)
                col_placeholder[col][val_idx] = fold_val_df[col].values

            self._log(f"  Fold {fold_i + 1}/{len(folds)} OOF 完成 | val_n={len(val_idx):,}")

        # 拼接训练特征
        oof_df = pd.DataFrame(col_placeholder)
        if self.keep_original:
            oof_df = pd.concat([X.reset_index(drop=True), oof_df], axis=1)
        if self.compress:
            oof_df = compress_dataframe(oof_df, verbose=False)

        # -------- 测试集：全量 fit --------
        test_df: Optional[pd.DataFrame] = None
        if X_test is not None:
            test_parts: list[pd.DataFrame] = []

            # 标签相关：用全量训练集 fit（信息最大化）
            for tf in leaky_tfs:
                full_tf = deepcopy(tf)
                full_tf.fit(X, y)
                test_parts.append(full_tf.transform(X_test).reset_index(drop=True))

            # 无关：直接 transform
            for tf in noleaky_tfs:
                test_parts.append(tf.transform(X_test).reset_index(drop=True))

            test_df = pd.concat(test_parts, axis=1)
            if self.keep_original:
                test_df = pd.concat([X_test.reset_index(drop=True), test_df], axis=1)
            if self.compress:
                test_df = compress_dataframe(test_df, verbose=False)

        self._log(
            f"OOF 变换完成 | train_shape={oof_df.shape} | "
            f"test_shape={test_df.shape if test_df is not None else 'N/A'}"
        )
        return oof_df, test_df

    # ------------------------------------------------------------------
    # 内部执行方法
    # ------------------------------------------------------------------

    def _transform_sequential(self, X: pd.DataFrame) -> list[pd.DataFrame]:
        parts = []
        for tf in self.transformers:
            t0 = time.perf_counter()
            tf._assert_fitted()
            new_feats = tf.transform(X).reset_index(drop=True)
            elapsed = time.perf_counter() - t0
            n_cols = new_feats.shape[1]
            self._log(f"  ✓ {tf.name:<35} +{n_cols} 列 | {elapsed:.3f}s")
            if self.compress:
                new_feats = compress_dataframe(new_feats, verbose=False)
            parts.append(new_feats)
        return parts

    def _transform_parallel(self, X: pd.DataFrame) -> list[pd.DataFrame]:
        results_map: dict[int, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_one_transformer, tf, X, i
                ): i
                for i, tf in enumerate(self.transformers)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results_map[idx] = future.result()
                except Exception as e:
                    raise RuntimeError(
                        f"并行变换器 {self.transformers[idx].name} 执行失败：{e}"
                    ) from e
        # 按原始顺序拼接
        return [results_map[i] for i in range(len(self.transformers))]

    def _run_one_transformer(
        self, tf: BaseFeatureTransformer, X: pd.DataFrame, idx: int
    ) -> pd.DataFrame:
        tf._assert_fitted()
        result = tf.transform(X).reset_index(drop=True)
        if self.compress:
            result = compress_dataframe(result, verbose=False)
        return result

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        return self._all_feature_names

    def summary(self) -> pd.DataFrame:
        """返回各变换器的特征列摘要 DataFrame。"""
        rows = []
        for tf in self.transformers:
            rows.append({
                "transformer": tf.name,
                "fitted":      tf.is_fitted,
                "n_features":  len(tf.feature_names_out),
                "features":    ", ".join(tf.feature_names_out[:5])
                               + ("..." if len(tf.feature_names_out) > 5 else ""),
            })
        return pd.DataFrame(rows)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[pipeline] {msg}")

    def __repr__(self) -> str:
        names = [tf.name for tf in self.transformers]
        return f"FeaturePipeline(transformers={names}, fitted={self.is_fitted})"


# ===========================================================================
# 原子变换器（Atomic Transformers）
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. TimeFeatureTransformer — 日历 + Sin/Cos 周期编码
# ---------------------------------------------------------------------------

class TimeFeatureTransformer(BaseFeatureTransformer):
    """
    时序特征提取：从 datetime 列自动生成日历特征和 Sin/Cos 周期编码。

    Sin/Cos 编码原理
    ----------------
    将周期性时间特征（如 hour ∈ [0,23]）映射到单位圆上：
      sin_hour = sin(2π × hour / 24)
      cos_hour = cos(2π × hour / 24)
    优点：保留了连续性（hour=23 与 hour=0 的距离 = hour=0 与 hour=1 的距离），
    避免了线性编码的"人工截断"问题。

    Parameters
    ----------
    date_col : str
        datetime 列名。
    extract : list[str]
        要提取的时间粒度：
        'year','month','day','hour','minute','second',
        'dayofweek','dayofyear','weekofyear','quarter',
        'is_month_end','is_month_start','is_weekend'
    cyclic : list[str]
        对哪些时间粒度做 Sin/Cos 编码（默认：month/dayofweek/hour）。
    drop_date_col : bool
        是否在输出中删除原始 datetime 列。默认 False（由调用方决定）。
    """

    # 各时间粒度的周期长度
    _PERIOD_MAP: dict[str, int] = {
        "month":      12,
        "day":        31,
        "hour":       24,
        "minute":     60,
        "second":     60,
        "dayofweek":  7,
        "dayofyear":  365,
        "weekofyear": 52,
        "quarter":    4,
    }

    # 默认提取粒度
    _DEFAULT_EXTRACT = [
        "year", "month", "day", "hour", "dayofweek",
        "dayofyear", "weekofyear", "quarter",
        "is_month_end", "is_month_start", "is_weekend",
    ]

    def __init__(
        self,
        date_col:      str,
        extract:       Optional[list[str]] = None,
        cyclic:        Optional[list[str]] = None,
        drop_date_col: bool                = False,
        name:          Optional[str]       = None,
    ):
        super().__init__(name=name or f"time_{date_col}")
        self.date_col      = date_col
        self.extract       = extract or self._DEFAULT_EXTRACT
        self.cyclic        = cyclic  or ["month", "dayofweek", "hour"]
        self.drop_date_col = drop_date_col
        self.requires_y    = False   # 无标签依赖，无泄露风险

    def fit(self, X: pd.DataFrame, y=None) -> "TimeFeatureTransformer":
        self._validate_col(X)
        self._mark_fitted(self._compute_output_names())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._assert_fitted()
        self._validate_col(X)

        dt = pd.to_datetime(X[self.date_col], errors="coerce")
        parts: dict[str, pd.Series] = {}

        # ---- 日历特征 ----
        _extractor = {
            "year":           dt.dt.year,
            "month":          dt.dt.month,
            "day":            dt.dt.day,
            "hour":           dt.dt.hour,
            "minute":         dt.dt.minute,
            "second":         dt.dt.second,
            "dayofweek":      dt.dt.dayofweek,
            "dayofyear":      dt.dt.dayofyear,
            "weekofyear":     dt.dt.isocalendar().week.astype(int),
            "quarter":        dt.dt.quarter,
            "is_month_end":   dt.dt.is_month_end.astype(np.int8),
            "is_month_start": dt.dt.is_month_start.astype(np.int8),
            "is_weekend":     (dt.dt.dayofweek >= 5).astype(np.int8),
        }
        for granularity in self.extract:
            if granularity in _extractor:
                col_name = f"{self.name}_{granularity}"
                parts[col_name] = _extractor[granularity].values

        # ---- Sin/Cos 周期编码 ----
        for granularity in self.cyclic:
            period = self._PERIOD_MAP.get(granularity)
            if period is None:
                continue
            if granularity not in _extractor:
                continue
            raw = _extractor[granularity].values.astype(float)
            parts[f"{self.name}_{granularity}_sin"] = np.sin(2 * np.pi * raw / period)
            parts[f"{self.name}_{granularity}_cos"] = np.cos(2 * np.pi * raw / period)

        return pd.DataFrame(parts, index=X.index)

    def _validate_col(self, X: pd.DataFrame) -> None:
        if self.date_col not in X.columns:
            raise ValueError(f"列 '{self.date_col}' 不存在于 DataFrame 中。")

    def _compute_output_names(self) -> list[str]:
        names = [f"{self.name}_{g}" for g in self.extract
                 if g in self._extractor_keys()]
        for g in self.cyclic:
            if g in self._PERIOD_MAP:
                names += [f"{self.name}_{g}_sin", f"{self.name}_{g}_cos"]
        return names

    @staticmethod
    def _extractor_keys() -> list[str]:
        return [
            "year","month","day","hour","minute","second",
            "dayofweek","dayofyear","weekofyear","quarter",
            "is_month_end","is_month_start","is_weekend",
        ]


# ---------------------------------------------------------------------------
# 2. GroupAggTransformer — 多维聚合统计
# ---------------------------------------------------------------------------

class GroupAggTransformer(BaseFeatureTransformer):
    """
    GroupBy 聚合特征生成器（多 key × 多 col × 多 func 笛卡尔积）。

    无泄露保证
    ----------
    fit(X_tr) 阶段计算聚合表（group → agg_value），
    transform(X_val) 只做 map/merge（查表），不使用 X_val 本身的值。
    → requires_y=False（除非 agg_col 是目标列本身，此时由调用方负责折隔离）

    Parameters
    ----------
    group_keys : list[str] | list[list[str]]
        GroupBy 的键，支持多个分组维度。
        例：["store_id"] 或 [["store_id"], ["store_id","item_id"]]
        传入字符串列表时视为多个单键分组，各自独立聚合。
    agg_cols : str | list[str]
        被聚合的列（目标列或数值列）。
    agg_funcs : list[str | Callable]
        聚合函数：'mean','std','max','min','median','sum','count',
                  'skew','nunique' 或自定义 Callable。
    fill_na : float
        聚合特征中 NaN 的填充值（未见过的 group 用此值）。默认 -1。
    """

    def __init__(
        self,
        group_keys: Union[list[str], list[list[str]]],
        agg_cols:   Union[str, list[str]],
        agg_funcs:  list[Union[str, Callable]],
        fill_na:    float = -1.0,
        name:       Optional[str] = None,
    ):
        # 规范化 group_keys → list[list[str]]
        if isinstance(group_keys[0], str):
            group_keys = [[k] for k in group_keys]

        super().__init__(name=name or "group_agg")
        self.group_keys = group_keys
        self.agg_cols   = [agg_cols] if isinstance(agg_cols, str) else list(agg_cols)
        self.agg_funcs  = agg_funcs
        self.fill_na    = fill_na
        self.requires_y = False  # 若 agg_col 是 target，需外部保证折隔离

        # fit 后的聚合映射表：{spec_key → pd.DataFrame}
        self._agg_tables: dict[str, pd.DataFrame] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "GroupAggTransformer":
        self._agg_tables.clear()
        all_names: list[str] = []

        for keys in self.group_keys:
            for col in self.agg_cols:
                if col not in X.columns:
                    warnings.warn(f"GroupAgg: 列 '{col}' 不存在，已跳过。")
                    continue
                # 计算聚合表
                agg_df = (
                    X.groupby(keys)[col]
                    .agg(self.agg_funcs)
                    .reset_index()
                )
                # 规范化列名
                func_names = [
                    f.__name__ if callable(f) else f
                    for f in self.agg_funcs
                ]
                prefix = f"{self.name}_{'_'.join(keys)}_{col}"
                rename = {
                    fn: f"{prefix}_{fn}"
                    for fn in func_names
                }
                agg_df.rename(columns=rename, inplace=True)

                spec_key = f"{'_'.join(keys)}__{col}"
                self._agg_tables[spec_key] = agg_df
                all_names.extend(rename.values())

        self._mark_fitted(all_names)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._assert_fitted()
        result = pd.DataFrame(index=X.index)

        for spec_key, agg_df in self._agg_tables.items():
            # 从 spec_key 还原 keys（仅需 join 列名，不重新计算）
            join_cols = [c for c in agg_df.columns
                         if c in X.columns]
            if not join_cols:
                continue
            feat_cols = [c for c in agg_df.columns
                         if c not in join_cols]
            merged = X[join_cols].merge(
                agg_df[join_cols + feat_cols],
                on=join_cols,
                how="left",
            )
            for fc in feat_cols:
                result[fc] = merged[fc].fillna(self.fill_na).values

        return result


# ---------------------------------------------------------------------------
# 3. DiffTransformer — 差分 / 百分比变化 / 移动统计
# ---------------------------------------------------------------------------

class DiffTransformer(BaseFeatureTransformer):
    """
    时序数据差分、百分比变化和移动统计特征。

    防泄露保证
    ----------
    所有操作均为向后移位（shift > 0 或 periods > 0），
    绝不使用未来时间点数据。

    Parameters
    ----------
    value_cols : list[str]
        需要计算差分/移动统计的数值列。
    sort_by : str | None
        变换前按此列排序（通常是时间列）。None 时假设已排序。
    group_by : str | list[str] | None
        分组列（如按商品、用户分组后再做差分）。
    diff_periods : list[int]
        差分阶数列表。默认 [1, 7]（日数据的一阶和周差分）。
    pct_change_periods : list[int]
        百分比变化的期数。默认 [1]。
    rolling_windows : list[int]
        移动窗口大小列表。默认 [7, 14, 28]。
    rolling_funcs : list[str]
        移动统计函数：'mean','std','min','max','sum'。
    min_periods : int
        移动窗口最少有效数据点数。默认 1。
    """

    def __init__(
        self,
        value_cols:          list[str],
        sort_by:             Optional[str]                  = None,
        group_by:            Optional[Union[str, list[str]]] = None,
        diff_periods:        list[int]                       = None,
        pct_change_periods:  list[int]                       = None,
        rolling_windows:     list[int]                       = None,
        rolling_funcs:       list[str]                       = None,
        min_periods:         int                             = 1,
        name:                Optional[str]                   = None,
    ):
        super().__init__(name=name or "diff")
        self.value_cols         = value_cols
        self.sort_by            = sort_by
        self.group_by           = [group_by] if isinstance(group_by, str) else (group_by or [])
        self.diff_periods       = diff_periods        or [1, 7]
        self.pct_change_periods = pct_change_periods  or [1]
        self.rolling_windows    = rolling_windows     or [7, 14, 28]
        self.rolling_funcs      = rolling_funcs       or ["mean", "std"]
        self.min_periods        = min_periods
        self.requires_y         = False

    def fit(self, X: pd.DataFrame, y=None) -> "DiffTransformer":
        # 无状态变换器：fit 只做校验
        missing = [c for c in self.value_cols if c not in X.columns]
        if missing:
            warnings.warn(f"DiffTransformer: 以下列不存在将被跳过：{missing}")
            self.value_cols = [c for c in self.value_cols if c in X.columns]

        self._mark_fitted(self._compute_output_names())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._assert_fitted()
        df = X.copy()

        # 排序
        if self.sort_by and self.sort_by in df.columns:
            df = df.sort_values(self.sort_by)

        result = pd.DataFrame(index=df.index)

        def _apply(series: pd.Series, col: str) -> None:
            """对一列应用所有差分/滚动操作。"""
            # 差分
            for p in self.diff_periods:
                result[f"{self.name}_{col}_diff{p}"] = series.diff(p)

            # 百分比变化
            for p in self.pct_change_periods:
                result[f"{self.name}_{col}_pct{p}"] = series.pct_change(p)

            # 滚动统计（closed='left' 确保不用到当前值本身）
            for w in self.rolling_windows:
                roller = series.shift(1).rolling(
                    window=w, min_periods=self.min_periods
                )
                for fn in self.rolling_funcs:
                    result[f"{self.name}_{col}_roll{w}_{fn}"] = getattr(roller, fn)()

        # 是否分组
        if self.group_by:
            for col in self.value_cols:
                if col not in df.columns:
                    continue
                applied = df.groupby(self.group_by)[col].transform(
                    lambda s: self._apply_series(s)
                )
                # 分组后逐个操作
                for p in self.diff_periods:
                    result[f"{self.name}_{col}_diff{p}"] = (
                        df.groupby(self.group_by)[col]
                        .transform(lambda s: s.diff(p))
                    )
                for p in self.pct_change_periods:
                    result[f"{self.name}_{col}_pct{p}"] = (
                        df.groupby(self.group_by)[col]
                        .transform(lambda s: s.pct_change(p))
                    )
                for w in self.rolling_windows:
                    for fn in self.rolling_funcs:
                        result[f"{self.name}_{col}_roll{w}_{fn}"] = (
                            df.groupby(self.group_by)[col]
                            .transform(
                                lambda s, _w=w, _fn=fn: (
                                    getattr(
                                        s.shift(1).rolling(_w, min_periods=self.min_periods),
                                        _fn
                                    )()
                                )
                            )
                        )
        else:
            for col in self.value_cols:
                if col not in df.columns:
                    continue
                _apply(df[col], col)

        # 如果排过序，还原原始行顺序
        if self.sort_by and self.sort_by in df.columns:
            result = result.reindex(X.index)

        return result

    @staticmethod
    def _apply_series(s: pd.Series) -> pd.Series:
        return s  # 占位，实际操作在 lambda 中完成

    def _compute_output_names(self) -> list[str]:
        names = []
        for col in self.value_cols:
            for p in self.diff_periods:
                names.append(f"{self.name}_{col}_diff{p}")
            for p in self.pct_change_periods:
                names.append(f"{self.name}_{col}_pct{p}")
            for w in self.rolling_windows:
                for fn in self.rolling_funcs:
                    names.append(f"{self.name}_{col}_roll{w}_{fn}")
        return names


# ---------------------------------------------------------------------------
# 4. LagTransformer — Lag / Lead 特征
# ---------------------------------------------------------------------------

class LagTransformer(BaseFeatureTransformer):
    """
    时序 Lag 特征（向后移位）。

    防泄露保证：shift > 0，不使用未来数据。

    Parameters
    ----------
    value_cols : list[str]
        需要生成 Lag 的列。
    lags : list[int]
        Lag 期数列表（必须全为正整数）。
    sort_by : str | None
        排序列（时间列）。
    group_by : str | list[str] | None
        分组列。
    """

    def __init__(
        self,
        value_cols: list[str],
        lags:       list[int],
        sort_by:    Optional[str]                   = None,
        group_by:   Optional[Union[str, list[str]]] = None,
        name:       Optional[str]                   = None,
    ):
        super().__init__(name=name or "lag")
        if any(l <= 0 for l in lags):
            raise ValueError("lags 必须全为正整数（向后移位，防止使用未来数据）。")
        self.value_cols = value_cols
        self.lags       = lags
        self.sort_by    = sort_by
        self.group_by   = [group_by] if isinstance(group_by, str) else (group_by or [])
        self.requires_y = False

    def fit(self, X: pd.DataFrame, y=None) -> "LagTransformer":
        self._mark_fitted(self._compute_output_names())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._assert_fitted()
        df = X.copy()
        if self.sort_by and self.sort_by in df.columns:
            df = df.sort_values(self.sort_by)

        result = pd.DataFrame(index=df.index)
        for col in self.value_cols:
            if col not in df.columns:
                continue
            for lag in self.lags:
                col_name = f"{self.name}_{col}_lag{lag}"
                if self.group_by:
                    result[col_name] = (
                        df.groupby(self.group_by)[col]
                        .transform(lambda s, _l=lag: s.shift(_l))
                    )
                else:
                    result[col_name] = df[col].shift(lag)

        if self.sort_by and self.sort_by in df.columns:
            result = result.reindex(X.index)
        return result

    def _compute_output_names(self) -> list[str]:
        return [
            f"{self.name}_{col}_lag{lag}"
            for col in self.value_cols
            for lag in self.lags
        ]


# ---------------------------------------------------------------------------
# 5. TargetEncoderTransformer — 目标编码（强制依赖 y，折内防泄露）
# ---------------------------------------------------------------------------

class TargetEncoderTransformer(BaseFeatureTransformer):
    """
    目标编码（Target Encoding）。

    ⚠ 强制标记 requires_y=True：
    FeaturePipeline.fit_transform_oof() 会自动检测此标志，
    确保每折只在 X_tr 上 fit，在 X_val 上 transform，绝不全局 fit。

    Smoothing 公式（贝叶斯平滑，防止小样本过拟合）：
        encoded = (count × mean + smoothing × global_mean) / (count + smoothing)

    Parameters
    ----------
    cat_cols : list[str]
        需要目标编码的类别列。
    smoothing : float
        平滑系数（越大越保守，防过拟合）。默认 10.0。
    min_samples_leaf : int
        最少样本数，低于此值的 group 全部用全局均值。默认 1。
    handle_unknown : str
        未见过的类别处理方式：'mean'（全局均值）| 'nan'。
    """

    def __init__(
        self,
        cat_cols:         list[str],
        smoothing:        float = 10.0,
        min_samples_leaf: int   = 1,
        handle_unknown:   str   = "mean",
        name:             Optional[str] = None,
    ):
        super().__init__(name=name or "target_enc")
        self.cat_cols         = cat_cols
        self.smoothing        = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.handle_unknown   = handle_unknown
        self.requires_y       = True   # ← 强制标记：必须折内防泄露

        self._encoding_maps: dict[str, dict] = {}
        self._global_mean:   float           = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoderTransformer":
        if y is None:
            raise ValueError("TargetEncoderTransformer.fit() 需要传入 y（目标列）。")

        self._encoding_maps.clear()
        self._global_mean = float(y.mean())
        y_arr = y.values

        output_names = []
        for col in self.cat_cols:
            if col not in X.columns:
                warnings.warn(f"TargetEncoder: 列 '{col}' 不存在，已跳过。")
                continue
            enc_map = self._compute_encoding(X[col], y_arr)
            self._encoding_maps[col] = enc_map
            output_names.append(f"{self.name}_{col}")

        self._mark_fitted(output_names)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._assert_fitted()
        result: dict[str, np.ndarray] = {}

        for col, enc_map in self._encoding_maps.items():
            if col not in X.columns:
                continue
            unknown_val = (
                self._global_mean if self.handle_unknown == "mean"
                else np.nan
            )
            encoded = (
                X[col].map(enc_map)
                      .fillna(unknown_val)
                      .values.astype(np.float32)
            )
            result[f"{self.name}_{col}"] = encoded

        return pd.DataFrame(result, index=X.index)

    def _compute_encoding(
        self, series: pd.Series, y: np.ndarray
    ) -> dict:
        """贝叶斯平滑目标编码映射表。"""
        df_tmp = pd.DataFrame({"cat": series.values, "target": y})
        stats  = df_tmp.groupby("cat")["target"].agg(["mean", "count"])

        # 平滑公式
        smoother = 1 / (1 + np.exp(-(stats["count"] - self.min_samples_leaf) / self.smoothing))
        smoothed = smoother * stats["mean"] + (1 - smoother) * self._global_mean

        return smoothed.to_dict()


# ---------------------------------------------------------------------------
# 6. FreqEncoderTransformer — 频率编码
# ---------------------------------------------------------------------------

class FreqEncoderTransformer(BaseFeatureTransformer):
    """
    频率编码（Frequency Encoding）：将类别值替换为其在训练集中的出现频率。
    无标签依赖，requires_y=False，可在全量 X 上 fit。

    Parameters
    ----------
    cat_cols : list[str]
        需要频率编码的类别列。
    normalize : bool
        True 时输出频率（0~1），False 时输出计数。默认 True。
    """

    def __init__(
        self,
        cat_cols:  list[str],
        normalize: bool         = True,
        name:      Optional[str] = None,
    ):
        super().__init__(name=name or "freq_enc")
        self.cat_cols   = cat_cols
        self.normalize  = normalize
        self.requires_y = False

        self._freq_maps: dict[str, dict] = {}

    def fit(self, X: pd.DataFrame, y=None) -> "FreqEncoderTransformer":
        self._freq_maps.clear()
        output_names = []
        for col in self.cat_cols:
            if col not in X.columns:
                warnings.warn(f"FreqEncoder: 列 '{col}' 不存在，已跳过。")
                continue
            freq = X[col].value_counts(normalize=self.normalize)
            self._freq_maps[col] = freq.to_dict()
            output_names.append(f"{self.name}_{col}")
        self._mark_fitted(output_names)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._assert_fitted()
        result: dict[str, np.ndarray] = {}
        for col, freq_map in self._freq_maps.items():
            if col not in X.columns:
                continue
            result[f"{self.name}_{col}"] = (
                X[col].map(freq_map).fillna(0).values.astype(np.float32)
            )
        return pd.DataFrame(result, index=X.index)
