"""
scripts/run_post_process.py
----------------------------
端到端后处理演示脚本：特征筛选 → 概率校准 → 阈值优化 → 加权融合。

演示流程
--------
0. 生成模拟多模型 OOF 场景（Binary 分类，20K 样本，65 维特征）
1. NullImportanceSelector  — 从 65 维特征中筛选出信息量高的子集
2. ProbabilityCalibrator   — 校准 OOF 概率（Isotonic / Platt 对比）
3. ThresholdOptimizer      — 搜索最优二分类决策阈值（Optuna TPE）
4. WeightedEnsembleOptimizer — 搜索三模型 OOF 最优融合权重（Optuna）

运行方式
--------
    python scripts/run_post_process.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, log_loss

from features.selector import NullImportanceSelector
from post_process.optimizer import (
    ProbabilityCalibrator,
    ThresholdOptimizer,
    WeightedEnsembleOptimizer,
)


# ===========================================================================
# 工具函数
# ===========================================================================

def section(title: str) -> None:
    bar = "=" * 66
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def timer(label: str):
    class _T:
        def __enter__(self):
            self._t = time.perf_counter()
            return self
        def __exit__(self, *_):
            print(f"    耗时: {time.perf_counter() - self._t:.2f}s")
    print(f"\n  >> {label}")
    return _T()


# ===========================================================================
# 模拟数据生成
# ===========================================================================

def make_classification_data(
    n_samples:    int = 20_000,
    n_features:   int = 65,
    n_informative: int = 20,
    noise_ratio:  float = 0.3,
    pos_rate:     float = 0.25,
    seed:         int  = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    生成模拟二分类数据集。

    n_informative 个特征与标签真实相关；
    其余特征为纯噪声（Null Importance 应过滤它们）。

    Returns
    -------
    X : pd.DataFrame, shape (n_samples, n_features)
    y : np.ndarray,   shape (n_samples,)
    """
    rng = np.random.default_rng(seed)

    # 生成真实信号特征（线性组合后加噪声）
    X_info = rng.standard_normal((n_samples, n_informative)).astype(np.float32)
    coefs  = rng.uniform(-1, 1, n_informative)
    logit  = X_info @ coefs
    logit += rng.normal(0, noise_ratio, n_samples)
    # 按 pos_rate 确定正类比例
    threshold = np.percentile(logit, (1 - pos_rate) * 100)
    y = (logit >= threshold).astype(np.int8)

    # 纯噪声特征（模拟实际比赛中大量冗余列）
    n_noise  = n_features - n_informative
    X_noise  = rng.standard_normal((n_samples, n_noise)).astype(np.float32)

    # 混合并打乱列顺序
    X_all  = np.concatenate([X_info, X_noise], axis=1)
    col_shuffle = rng.permutation(n_features)
    X_all  = X_all[:, col_shuffle]

    feature_names = [f"feat_{i:03d}" for i in range(n_features)]
    X_df = pd.DataFrame(X_all, columns=feature_names)

    # 模拟 ratio 特征中常见的 Inf/NaN（约 0.5% 的值）
    inf_mask = rng.random((n_samples, n_features)) < 0.005
    X_df[inf_mask] = np.inf

    return X_df, y


def simulate_model_oof(
    X:   pd.DataFrame,
    y:   np.ndarray,
    model_id: int = 0,
    seed:     int = 42,
    test_idx: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    用 LightGBM 训练一个简单模型，返回 (oof_proba, test_proba)。
    test 集可由外部传入（保证多模型使用同一测试集进行公平比较）。
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("演示脚本需要 lightgbm，请安装：pip install lightgbm")

    from sklearn.model_selection import StratifiedKFold

    n = len(X)
    if test_idx is None:
        rng_test = np.random.default_rng(seed)  # 固定 seed 保持一致
        test_idx = rng_test.choice(n, size=n // 5, replace=False)

    train_mask = np.ones(n, dtype=bool)
    train_mask[test_idx] = False

    X_train, y_train = X[train_mask], y[train_mask]
    X_test            = X.iloc[test_idx]

    # Inf → NaN（LightGBM 可处理 NaN，但 Inf 会报错）
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan)

    n_train = len(X_train)
    oof_proba = np.zeros(n_train, dtype=np.float32)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + model_id)
    fold_test_probas = []

    params = dict(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        n_jobs=-1,
        random_state=seed + model_id,
    )

    for tr_idx, va_idx in skf.split(X_train, y_train):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train.iloc[tr_idx], y_train[tr_idx],
            eval_set=[(X_train.iloc[va_idx], y_train[va_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_proba[va_idx] = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        fold_test_probas.append(model.predict_proba(X_test)[:, 1])

    test_proba = np.mean(fold_test_probas, axis=0).astype(np.float32)
    return oof_proba, test_proba, y_train, y[test_idx]


# ===========================================================================
# 主演示流程
# ===========================================================================

def main() -> None:

    # -----------------------------------------------------------------------
    # 0. 数据准备
    # -----------------------------------------------------------------------
    section("0. 生成模拟分类数据")
    with timer("make_classification_data"):
        X, y = make_classification_data(
            n_samples=20_000,
            n_features=65,
            n_informative=20,
            pos_rate=0.25,
            seed=42,
        )
    print(f"    数据形状   : {X.shape}")
    print(f"    正样本比例 : {y.mean():.2%}")
    print(f"    Inf 值数量 : {np.isinf(X.values).sum():,}")

    # -----------------------------------------------------------------------
    # Module 1: NullImportanceSelector
    # -----------------------------------------------------------------------
    section("Module 1: NullImportanceSelector（置换检验特征筛选）")

    X_clean = X.replace([np.inf, -np.inf], np.nan)  # 预处理

    sel = NullImportanceSelector(
        n_iterations=60,              # 迭代 60 次（生产建议 80~100）
        percentile_threshold=75.0,    # 真实重要性须超过 75% 的随机值
        task="binary",
        max_samples=10_000,           # 采样 1 万行加速 Null 迭代
        random_state=42,
        verbose=True,
    )
    print(f"\n  {sel}")

    with timer("NullImportanceSelector.fit"):
        sel.fit(X_clean, y)

    scores = sel.get_scores()
    n_selected  = sel.selected_features_.__len__()
    n_informative_kept = sum(
        1 for f in sel.selected_features_ if f in scores["feature"].values
    )

    print(f"\n  [筛选结果]")
    print(f"    原始特征数   : {X.shape[1]}")
    print(f"    保留特征数   : {n_selected}")
    print(f"    top-5 保留特征:")
    for _, row in scores.head(5).iterrows():
        mark = "YES" if row["selected"] else "---"
        print(
            f"      {row['feature']:<12} real={row['real_importance']:>8.1f}  "
            f"null={row['null_mean']:>8.1f}  "
            f"score={row['score']:>8.2f}  [{mark}]"
        )

    X_selected = sel.transform(X_clean)
    print(f"\n    筛选后特征矩阵 shape: {X_selected.shape}")

    # -----------------------------------------------------------------------
    # Module 2: 模拟多模型 OOF（LightGBM 基础演示）
    # -----------------------------------------------------------------------
    section("2. 训练多模型 OOF（3 个 LGBM，不同随机种子）")

    # 固定测试集（所有模型共用，保证融合比较公平）
    shared_test_idx = np.random.default_rng(42).choice(len(X_selected), size=len(X_selected)//5, replace=False)

    model_results = []
    for mid in range(3):
        with timer(f"model_{mid} 5-Fold CV"):
            oof, test, y_tr, y_te = simulate_model_oof(
                X_selected, y, model_id=mid, seed=42, test_idx=shared_test_idx
            )
        auc = roc_auc_score(y_tr, oof)
        print(f"    model_{mid} OOF AUC = {auc:.5f}")
        model_results.append({
            "oof":   oof,
            "test":  test,
            "y_tr":  y_tr,
            "y_te":  y_te,
        })

    # 使用第一个模型的 y_tr 作为通用 y_train（各模型相同）
    y_train = model_results[0]["y_tr"]
    y_test  = model_results[0]["y_te"]

    # -----------------------------------------------------------------------
    # Module 3: ProbabilityCalibrator
    # -----------------------------------------------------------------------
    section("Module 3: ProbabilityCalibrator（OOF 概率校准）")

    oof_proba_m0 = model_results[0]["oof"]
    test_proba_m0 = model_results[0]["test"]

    # Isotonic 校准
    cal_iso = ProbabilityCalibrator(method="isotonic", n_folds=5, verbose=True)
    with timer("isotonic fit_transform"):
        oof_cal_iso = cal_iso.fit_transform(y_train, oof_proba_m0)
    test_cal_iso = cal_iso.transform(test_proba_m0)

    # Platt 校准
    cal_platt = ProbabilityCalibrator(method="platt", n_folds=5, verbose=True)
    with timer("platt fit_transform"):
        oof_cal_platt = cal_platt.fit_transform(y_train, oof_proba_m0)

    # AUC 对比（校准不影响排序，AUC 应基本不变）
    print(f"\n  [AUC 对比（校准不应影响排序）]")
    print(f"    原始 OOF AUC     : {roc_auc_score(y_train, oof_proba_m0):.5f}")
    print(f"    Isotonic AUC     : {roc_auc_score(y_train, oof_cal_iso):.5f}")
    print(f"    Platt AUC        : {roc_auc_score(y_train, oof_cal_platt):.5f}")

    # LogLoss 对比（校准应改善 logloss）
    print(f"\n  [LogLoss 对比（校准应降低 logloss）]")
    print(f"    原始 OOF LogLoss : {log_loss(y_train, oof_proba_m0):.5f}")
    print(f"    Isotonic LogLoss : {log_loss(y_train, oof_cal_iso):.5f}")
    print(f"    Platt LogLoss    : {log_loss(y_train, oof_cal_platt):.5f}")

    # -----------------------------------------------------------------------
    # Module 4: ThresholdOptimizer
    # -----------------------------------------------------------------------
    section("Module 4: ThresholdOptimizer（决策阈值优化）")

    thr_opt = ThresholdOptimizer(
        metric="f1",
        n_trials=300,
        search_range=(0.01, 0.99),
        random_state=42,
        verbose=True,
    )
    print(f"\n  {thr_opt}")

    with timer("ThresholdOptimizer.fit (Optuna TPE)"):
        thr_opt.fit(y_train, oof_cal_iso)

    # F1 对比
    f1_default  = f1_score(y_train, (oof_cal_iso >= 0.5).astype(int), zero_division=0)
    f1_optimized = f1_score(y_train, thr_opt.predict(oof_cal_iso), zero_division=0)
    print(f"\n  [F1 对比]")
    print(f"    默认阈值 0.50 F1   : {f1_default:.5f}")
    print(f"    最优阈值 {thr_opt.best_threshold_:.4f} F1 : {f1_optimized:.5f}")
    print(f"    提升幅度           : {f1_optimized - f1_default:+.5f}")

    # 测试集预测
    y_pred_test = thr_opt.predict(test_cal_iso)
    f1_test     = f1_score(y_test, y_pred_test, zero_division=0)
    print(f"    测试集 F1（最优阈值）: {f1_test:.5f}")

    # -----------------------------------------------------------------------
    # Module 5: WeightedEnsembleOptimizer
    # -----------------------------------------------------------------------
    section("Module 5: WeightedEnsembleOptimizer（多模型加权融合）")

    # 构建 OOF 矩阵（3 模型）
    oof_matrix  = np.column_stack([r["oof"]  for r in model_results])
    test_matrix = np.column_stack([r["test"] for r in model_results])

    ens_opt = WeightedEnsembleOptimizer(
        metric="auc",
        n_trials=300,
        direction="maximize",
        random_state=42,
        verbose=True,
    )
    print(f"\n  {ens_opt}")

    with timer("WeightedEnsembleOptimizer.fit (Optuna + Softmax)"):
        ens_opt.fit(y_train, oof_matrix, model_names=["lgbm_0", "lgbm_1", "lgbm_2"])

    # 权重汇总
    print(f"\n  {ens_opt.get_weight_table().to_string(index=False)}")

    # 融合测试集预测
    test_ensemble = ens_opt.transform(test_matrix)
    test_avg      = test_matrix.mean(axis=1)

    auc_avg = roc_auc_score(y_test, test_avg)
    auc_ens = roc_auc_score(y_test, test_ensemble)
    print(f"\n  [测试集 AUC 对比]")
    print(f"    均值融合 AUC  : {auc_avg:.5f}")
    print(f"    加权融合 AUC  : {auc_ens:.5f}")
    print(f"    提升幅度      : {auc_ens - auc_avg:+.5f}")

    # -----------------------------------------------------------------------
    # 汇总
    # -----------------------------------------------------------------------
    section("汇总：完整后处理流水线效果")

    print(f"""
  原始特征数         : {X.shape[1]}
  筛选后特征数       : {X_selected.shape[1]}  (-{X.shape[1]-X_selected.shape[1]} 噪声特征)

  OOF 概率（Model 0）:
    原始 AUC         : {roc_auc_score(y_train, oof_proba_m0):.5f}
    校准后 LogLoss   : {log_loss(y_train, oof_cal_iso):.5f}  (Isotonic)

  决策阈值           :
    默认 (0.5) F1    : {f1_default:.5f}
    最优 ({thr_opt.best_threshold_:.4f}) F1  : {f1_optimized:.5f}

  多模型融合（测试集）:
    均值融合 AUC     : {auc_avg:.5f}
    加权融合 AUC     : {auc_ens:.5f}

  [所有模块均实现 fit / transform / fit_transform 标准接口]
  [可直接接入 CVResult 输出，零修改嵌入训练流水线]
""")


if __name__ == "__main__":
    main()
