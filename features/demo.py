"""
features/demo.py
----------------
演示 RobustCategoricalEncoder、TimeSeriesFeatureGenerator、
AutoFeatureInteraction 在模拟零售销售数据上的完整用法。

模拟数据规模
-----------
20 家门店 x 50 种商品 x 365 天 = 365,000 行
足以展示性能特性，同时保持演示可快速完成。

运行方式
--------
    python features/demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 确保能导入项目模块（无论从哪个目录运行）
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from features.engine import (
    AutoFeatureInteraction,
    RobustCategoricalEncoder,
    TimeSeriesFeatureGenerator,
    _HAS_POLARS,
)


# ===========================================================================
# 工具函数
# ===========================================================================

def section(title: str) -> None:
    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def show_df(df: pd.DataFrame, label: str, n_rows: int = 4) -> None:
    print(f"\n  [{label}]")
    print(f"    shape  : {df.shape}")
    mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"    memory : {mb:.2f} MB")
    with pd.option_context("display.max_columns", 8, "display.width", 120):
        print(df.head(n_rows).to_string(index=False))


def timer(label: str):
    """简单计时上下文管理器。"""
    class _T:
        def __enter__(self):
            self._t0 = time.perf_counter()
            return self
        def __exit__(self, *_):
            print(f"    耗时: {time.perf_counter() - self._t0:.3f}s")
    print(f"\n  >> {label}")
    return _T()


# ===========================================================================
# 模拟数据生成
# ===========================================================================

def make_retail_data(
    n_stores:   int = 20,
    n_products: int = 50,
    n_days:     int = 365,
    seed:       int = 42,
) -> pd.DataFrame:
    """
    生成模拟零售时序数据。

    列说明
    ------
    date        : 日期（时间列，日频）
    store_id    : 门店编号（str 类别，20 个唯一值）
    product_id  : 商品编号（str 类别，50 个唯一值，高基数）
    sales       : 日销售量（回归目标，Poisson 分布 + 周期效应 + 噪声）
    price       : 商品基准价格（数值特征）
    promo       : 是否促销（0/1，约 10% 促销率）
    """
    rng      = np.random.default_rng(seed)
    stores   = np.arange(n_stores)
    products = np.arange(n_products)
    dates    = pd.date_range("2023-01-01", periods=n_days, freq="D")

    # 笛卡尔积（store × product × day）
    store_arr   = np.repeat(stores,   n_products * n_days)
    product_arr = np.tile(np.repeat(products, n_days), n_stores)
    date_arr    = np.tile(dates, n_stores * n_products)
    date_arr    = pd.DatetimeIndex(date_arr)   # 保留 datetime 属性
    n           = len(store_arr)

    # 基础销量 = 门店效应 × 商品效应 × 周效应 × 月份效应 + 噪声
    store_eff   = rng.normal(100, 30,  n_stores )[store_arr]
    product_eff = rng.normal(1.0, 0.3, n_products)[product_arr]
    dow_eff     = np.where(date_arr.dayofweek >= 5, 1.2, 1.0)      # 周末 +20%
    month_eff   = 1.0 + 0.1 * np.sin(2 * np.pi * date_arr.month / 12)

    sales = (store_eff * product_eff * dow_eff * month_eff
             + rng.normal(0, 10, n))
    sales = np.clip(sales, 0, None).astype(np.float32)

    price = rng.uniform(5, 200, n_products)[product_arr].astype(np.float32)
    promo = (rng.random(n) < 0.10).astype(np.int8)

    df = pd.DataFrame({
        "date":       date_arr,
        "store_id":   [f"store_{s:02d}"   for s in store_arr],   # str 类别
        "product_id": [f"prod_{p:03d}"    for p in product_arr],  # str 类别，高基数
        "sales":      sales,
        "price":      price,
        "promo":      promo,
    })
    # 按 store → product → date 排序（保证分组时序正确性）
    return df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


# ===========================================================================
# 主演示流程
# ===========================================================================

def main() -> None:
    # -----------------------------------------------------------------------
    # 0. 数据准备
    # -----------------------------------------------------------------------
    section("0. 生成模拟零售数据")
    with timer("make_retail_data"):
        df = make_retail_data(n_stores=20, n_products=50, n_days=365)

    print(f"    数据形状  : {df.shape}")
    print(f"    日期范围  : {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"    门店数    : {df['store_id'].nunique()}")
    print(f"    商品数    : {df['product_id'].nunique()}")
    print(f"    Polars 引擎: {'可用' if _HAS_POLARS else '不可用（Pandas 路径）'}")
    show_df(df, "raw data", n_rows=3)

    X = df.drop("sales", axis=1)
    y = df["sales"]

    # -----------------------------------------------------------------------
    # Module 1: RobustCategoricalEncoder
    # -----------------------------------------------------------------------
    section("Module 1: RobustCategoricalEncoder（K-Fold 目标编码）")

    enc = RobustCategoricalEncoder(
        cat_cols=["store_id", "product_id"],
        n_splits=5,
        smoothing=10.0,
        noise_std=0.01,          # 1% target_std 的 Gaussian 噪声
        cardinality_threshold=30,  # product_id（50 唯一值）→ 高基数，自动加强平滑
        high_card_multiplier=3.0,
        random_state=42,
    )
    print(f"\n  {enc}")

    with timer("fit_transform（K-Fold OOF + 噪声注入，训练集路径）"):
        X_enc_train = enc.fit_transform(X, y)

    show_df(X_enc_train, "RobustCategoricalEncoder output（训练集 OOF）", n_rows=3)

    # 测试集路径（全局编码，无噪声）
    X_test_subset = X.tail(1000)
    with timer("transform（全局编码，测试集路径）"):
        X_enc_test = enc.transform(X_test_subset)
    print(f"    测试集 shape: {X_enc_test.shape}")

    # 高基数自适应 smoothing 验证
    store_uniq   = df["store_id"].nunique()    # 20 < 30 → 普通 smoothing
    product_uniq = df["product_id"].nunique()  # 50 > 30 → 高基数，放大 smoothing
    print(f"\n  [平滑系数验证]")
    print(f"    store_id   唯一值={store_uniq}  (<30) → smoothing = {enc.smoothing:.1f}")
    print(f"    product_id 唯一值={product_uniq} (>30) → smoothing = "
          f"{enc.smoothing * enc.high_card_multiplier:.1f}（高基数放大）")

    # OOF vs 全局编码的均值差异（噪声效应）
    oof_mean  = float(X_enc_train[f"{enc.name}_store_id"].mean())
    glob_mean = float(X_enc_test[f"{enc.name}_store_id"].mean())
    print(f"    OOF store_id 均值 = {oof_mean:.4f}（含噪声），"
          f"全局 = {glob_mean:.4f}（无噪声）")

    # -----------------------------------------------------------------------
    # Module 2: TimeSeriesFeatureGenerator
    # -----------------------------------------------------------------------
    section("Module 2: TimeSeriesFeatureGenerator（多尺度时序特征）")

    ts_gen = TimeSeriesFeatureGenerator(
        value_cols=["sales", "price"],
        time_col="date",
        group_cols=["store_id", "product_id"],
        lags=[1, 7, 30],                            # t-1 / t-7 / t-30
        rolling_windows=[7, 14, 28],                # 周 / 双周 / 月窗口
        rolling_funcs=["mean", "std", "max", "min"],
        diff_periods=[1, 7],                        # 1 阶差分 + 周差分
    )
    print(f"\n  {ts_gen}")
    print(f"  预计生成特征数: {len(ts_gen.fit(df).feature_names_out)}")

    with timer("transform（分组 Lag + Rolling + Diff）"):
        X_ts = ts_gen.transform(df)

    show_df(X_ts, "TimeSeriesFeatureGenerator output", n_rows=3)

    # ---- 防泄露验证：lag_1 应 == 上一时间步的 sales ----
    print(f"\n  [防泄露验证] lag_1 应等于前一行 sales（第一行为 NaN）")
    lag1_col = f"{ts_gen.name}_sales_lag1"
    samp     = df[(df["store_id"] == "store_00") & (df["product_id"] == "prod_000")]
    verif    = samp[["date", "sales"]].head(5).copy()
    verif["lag1_expected"] = verif["sales"].shift(1).values
    verif["lag1_actual"]   = X_ts.loc[verif.index, lag1_col].values
    verif["match"]         = np.isclose(
        verif["lag1_expected"].fillna(-999),
        verif["lag1_actual"].fillna(-999)
    )
    with pd.option_context("display.width", 100):
        print(verif.to_string(index=False))

    # ---- 特征统计 ----
    lag_null_rate = X_ts[[c for c in X_ts.columns if "lag30" in c]].isna().mean().mean()
    print(f"\n  lag30 平均 NaN 率: {lag_null_rate:.1%}（前 30 行无历史数据，符合预期）")

    # -----------------------------------------------------------------------
    # Module 3: AutoFeatureInteraction
    # -----------------------------------------------------------------------
    section("Module 3: AutoFeatureInteraction（自动二阶特征交叉）")

    # 构建数值特征矩阵（编码特征 + 时序特征部分列）
    ts_key_cols = [c for c in X_ts.columns
                   if any(k in c for k in ["lag1", "roll7_mean", "diff1"])]
    X_numeric = pd.concat([
        X[["price", "promo"]].reset_index(drop=True),
        X_enc_train.reset_index(drop=True),
        X_ts[ts_key_cols].fillna(0).reset_index(drop=True),
    ], axis=1)
    print(f"  输入数值特征数: {X_numeric.shape[1]}")

    interact = AutoFeatureInteraction(
        max_features=6,           # 取与 y 相关性最高的 6 个特征
        interaction_types=["product", "ratio"],
        min_corr_with_y=0.005,    # 宽松阈值（模拟数据相关性相对低）
        max_interactions=12,      # 最多保留 12 对交叉特征
    )
    print(f"\n  {interact}")

    with timer("fit_transform（特征相关性排序 + 二阶交叉）"):
        X_interact = interact.fit_transform(X_numeric, y.reset_index(drop=True))

    show_df(X_interact, "AutoFeatureInteraction output", n_rows=3)
    print(f"\n  选出特征对（按 |corr with y| 排序的 top-{interact.max_features} 特征的组合）:")
    for i, (a, b) in enumerate(interact._selected_pairs, 1):
        print(f"    {i:2d}. {a}  ×/÷  {b}")

    # -----------------------------------------------------------------------
    # 汇总 & 最终特征矩阵
    # -----------------------------------------------------------------------
    section("汇总：最终特征矩阵")

    X_final = pd.concat([
        X.reset_index(drop=True),
        X_enc_train.reset_index(drop=True),
        X_ts.fillna(0).reset_index(drop=True),
        X_interact.reset_index(drop=True),
    ], axis=1)

    total_mb = X_final.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"  原始特征数         : {X.shape[1]}")
    print(f"  + 目标编码特征     : {X_enc_train.shape[1]}")
    print(f"  + 时序特征         : {X_ts.shape[1]}")
    print(f"  + 二阶交叉特征     : {X_interact.shape[1]}")
    print(f"  ─────────────────────────────────")
    total_feats = X.shape[1] + X_enc_train.shape[1] + X_ts.shape[1] + X_interact.shape[1]
    print(f"  合计特征数         : {total_feats}")
    print(f"  最终矩阵 shape     : {X_final.shape}")
    print(f"  内存占用           : {total_mb:.2f} MB")

    print(f"\n  [所有模块均已通过 sklearn fit/transform 规范接口运行]")
    print(f"  [可直接嵌入 FeaturePipeline 或 sklearn Pipeline 使用]\n")


if __name__ == "__main__":
    main()
