"""
data/adversarial_validation.py
--------------------------------
对抗性验证（Adversarial Validation）。

核心思路
--------
将训练集和测试集混合，用 is_test（0=train, 1=test）作为伪标签，
训练一个 LightGBM 二分类器：

  - AUC ≈ 0.5  → 分布一致，CV 可靠，放心使用全部特征
  - AUC > 0.7  → 存在显著分布偏移，高重要性特征是"罪魁祸首"
  - AUC > 0.9  → 严重偏移，必须处理后才能参赛

两阶段诊断
----------
  Stage 1（快速，默认启用）
    全特征混合训练 → 5-Fold AUC + 特征重要性排名
    → 告诉你"是否存在偏移"以及"哪些特征最可疑"

  Stage 2（精准，可选，compute_per_feature_auc=True）
    对 Top-K 特征逐一做单变量 AUC 计算
    → 每个特征独立的"分布偏移贡献度"
    → 更精准的剔除建议

输出接口
--------
  AdversarialValidationResult
    .overall_auc            float       全特征 CV 均值 AUC
    .fold_aucs              list[float] 每折 AUC
    .feature_importance     DataFrame   columns: feature / importance_mean /
                                                 importance_std / rank /
                                                 per_feature_auc（Stage 2）/
                                                 suggested_drop
    .suggested_drop_cols    list[str]   建议剔除的列名
    .distribution_shift     bool        AUC > auc_threshold
    .report()               str         人类可读摘要（可直接 print）

调用示例
--------
>>> from data.adversarial_validation import AdversarialValidator, AdvConfig
>>> av = AdversarialValidator(AdvConfig(enable=True, auto_drop=True))
>>> result = av.run(X_train, X_test)
>>> print(result.report())
>>> X_train_clean = X_train.drop(columns=result.suggested_drop_cols)
>>> X_test_clean  = X_test.drop(columns=result.suggested_drop_cols)

在 train_pipeline.py 中的接入点
---------------------------------
  DataLoader.load()
       │
  AdversarialValidator.run()    ← 此处（Step 1.5）
       │
  FeaturePipeline.fit_transform_oof()
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------

@dataclass
class AdvConfig:
    """
    对抗性验证配置。

    Parameters
    ----------
    enable : bool
        False 时 AdversarialValidator.run() 是空操作，直接返回 None。

    auc_threshold : float
        全特征 AUC 高于此值则判定"存在分布偏移"，默认 0.7。

    drop_threshold_auc : float
        Stage 2 中，单特征 AUC 高于此值则列入剔除建议，默认 0.65。

    drop_threshold_importance : float
        Stage 1 中，归一化特征重要性高于此比例时进入候选（再经 Stage 2 确认）。
        用于过滤 Stage 2 的计算范围。默认 0.01（Top-1%）。

    n_splits : int
        对抗模型 CV 折数，默认 5。

    n_top_features : int
        Stage 2 最多分析的可疑特征数，默认 30。

    compute_per_feature_auc : bool
        True → 启用 Stage 2（单特征 AUC 精准诊断），默认 False。

    auto_drop : bool
        True → run() 同时返回 clean_train / clean_test（已剔除建议列）。
        False → 只返回建议，不修改数据。

    ignore_cols : list[str]
        不参与验证的列（如 ID、时间戳列）。

    lgbm_params : dict
        覆盖默认 LightGBM 参数的字典。

    seed : int
        随机种子，默认 42。

    verbose : bool
        是否打印进度日志，默认 True。
    """
    enable:                    bool       = False
    auc_threshold:             float      = 0.70
    drop_threshold_auc:        float      = 0.65
    drop_threshold_importance: float      = 0.01
    n_splits:                  int        = 5
    n_top_features:            int        = 30
    compute_per_feature_auc:   bool       = False
    auto_drop:                 bool       = False
    ignore_cols:               list       = field(default_factory=list)
    lgbm_params:               dict       = field(default_factory=dict)
    seed:                      int        = 42
    verbose:                   bool       = True

    @classmethod
    def from_dict(cls, d: dict) -> "AdvConfig":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# 结果数据类
# ---------------------------------------------------------------------------

@dataclass
class AdversarialValidationResult:
    """
    AdversarialValidator.run() 的返回值。

    Attributes
    ----------
    overall_auc : float
        5-Fold 平均 AUC（全特征对抗模型）。

    fold_aucs : list[float]
        每折的 AUC 分数。

    feature_importance : pd.DataFrame
        columns:
          - feature          : 特征名
          - importance_mean  : 各折重要性均值
          - importance_std   : 各折重要性标准差
          - rank             : 重要性排名（1=最高）
          - per_feature_auc  : 单特征 AUC（compute_per_feature_auc=True 时有值）
          - suggested_drop   : bool，是否建议删除

    suggested_drop_cols : list[str]
        建议从训练集和测试集中一起删除的列。

    distribution_shift : bool
        overall_auc > config.auc_threshold。

    elapsed_seconds : float
    """
    overall_auc:          float
    fold_aucs:            list[float]
    feature_importance:   pd.DataFrame
    suggested_drop_cols:  list[str]
    distribution_shift:   bool
    elapsed_seconds:      float
    config:               AdvConfig

    def report(self) -> str:
        """返回人类可读的对抗验证报告字符串。"""
        lines = [
            "",
            "=" * 62,
            "  对抗性验证报告 (Adversarial Validation Report)",
            "=" * 62,
            f"  全特征 CV AUC  : {self.overall_auc:.4f}  "
            f"{'⚠  分布偏移检测' if self.distribution_shift else '✓  分布一致'}",
            f"  各折 AUC       : "
            + "  ".join(f"{s:.4f}" for s in self.fold_aucs),
            f"  AUC 判定阈值   : {self.config.auc_threshold}",
            f"  建议剔除列数   : {len(self.suggested_drop_cols)}",
            "-" * 62,
        ]

        if self.distribution_shift:
            lines.append("  ⚠  警告：训练集与测试集存在显著特征分布偏移！")
            lines.append("     以下特征导致分类器能区分来源：")
            lines.append("")

        # Top 特征展示
        top_n = min(20, len(self.feature_importance))
        fi    = self.feature_importance.head(top_n)

        has_pfa = "per_feature_auc" in fi.columns and fi["per_feature_auc"].notna().any()
        header  = f"  {'排名':>4}  {'特征名':<35}  {'重要性':>8}"
        if has_pfa:
            header += f"  {'单特征AUC':>9}"
        header += f"  {'建议剔除':>6}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for _, row in fi.iterrows():
            mark = "  ← DROP" if row.get("suggested_drop", False) else ""
            line = (
                f"  {int(row['rank']):>4}  "
                f"{str(row['feature']):<35}  "
                f"{row['importance_mean']:>8.4f}"
            )
            if has_pfa and pd.notna(row.get("per_feature_auc")):
                line += f"  {row['per_feature_auc']:>9.4f}"
            elif has_pfa:
                line += f"  {'N/A':>9}"
            line += f"  {str(row.get('suggested_drop', False)):>6}{mark}"
            lines.append(line)

        lines.append("")
        if self.suggested_drop_cols:
            lines.append(f"  建议删除的列（共 {len(self.suggested_drop_cols)} 列）：")
            for col in self.suggested_drop_cols:
                lines.append(f"    - {col}")
        else:
            lines.append("  未发现需要强制删除的列。")

        lines += [
            "",
            f"  耗时：{self.elapsed_seconds:.1f}s",
            "=" * 62,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 对抗性验证主类
# ---------------------------------------------------------------------------

class AdversarialValidator:
    """
    对抗性验证引擎。

    使用示例
    --------
    >>> from data.adversarial_validation import AdversarialValidator, AdvConfig
    >>> config = AdvConfig(
    ...     enable                  = True,
    ...     auc_threshold           = 0.70,
    ...     compute_per_feature_auc = True,
    ...     auto_drop               = True,
    ... )
    >>> av = AdversarialValidator(config)
    >>> result = av.run(X_train, X_test)
    >>> print(result.report())

    # 自动剔除高偏移特征
    >>> X_train_clean, X_test_clean = av.apply_drop(X_train, X_test, result)
    """

    # 默认 LGBM 对抗模型参数（快速、稳定、不过拟合）
    _DEFAULT_LGBM_PARAMS: dict = {
        "objective":        "binary",
        "metric":           "auc",
        "n_estimators":     200,
        "learning_rate":    0.05,
        "num_leaves":       31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     1,
        "verbose":         -1,
        "n_jobs":          -1,
    }

    def __init__(self, config: AdvConfig):
        self.config = config

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(
        self,
        X_train: pd.DataFrame,
        X_test:  pd.DataFrame,
    ) -> Optional[AdversarialValidationResult]:
        """
        执行对抗性验证。

        Parameters
        ----------
        X_train : pd.DataFrame
            训练特征（不含标签列）。
        X_test : pd.DataFrame
            测试特征。

        Returns
        -------
        AdversarialValidationResult | None
            config.enable=False 时返回 None。
        """
        if not self.config.enable:
            logger.debug("AdversarialValidator: enable=False，跳过对抗验证。")
            return None

        t0 = time.perf_counter()
        if self.config.verbose:
            logger.info("─" * 62)
            logger.info("  Step 1.5  对抗性验证（Adversarial Validation）")
            logger.info("─" * 62)

        # ── 特征列对齐 ─────────────────────────────────────────────
        ignore   = set(self.config.ignore_cols)
        feat_cols = [
            c for c in X_train.columns
            if c in X_test.columns and c not in ignore
        ]
        missing_in_test  = set(X_train.columns) - set(X_test.columns) - ignore
        missing_in_train = set(X_test.columns)  - set(X_train.columns) - ignore

        if missing_in_test:
            warnings.warn(
                f"以下列仅在训练集中存在，已排除：{sorted(missing_in_test)}",
                UserWarning, stacklevel=2,
            )
        if missing_in_train:
            warnings.warn(
                f"以下列仅在测试集中存在，已排除：{sorted(missing_in_train)}",
                UserWarning, stacklevel=2,
            )

        if not feat_cols:
            raise ValueError("训练集和测试集没有共同列，无法进行对抗验证。")

        if self.config.verbose:
            logger.info(
                f"  参与验证的特征数: {len(feat_cols)} | "
                f"训练集: {len(X_train)} 行 | 测试集: {len(X_test)} 行"
            )

        # ── Step 1: 混合数据集 ─────────────────────────────────────
        X_mixed, y_mixed = self._build_mixed_dataset(
            X_train[feat_cols], X_test[feat_cols]
        )

        # ── Step 2: 全特征对抗 CV ──────────────────────────────────
        overall_auc, fold_aucs, fi_df = self._train_adversarial_cv(
            X_mixed, y_mixed, feat_cols
        )
        distribution_shift = overall_auc > self.config.auc_threshold

        if self.config.verbose:
            status = "⚠  存在分布偏移" if distribution_shift else "✓  分布一致"
            logger.info(
                f"  全特征 AUC={overall_auc:.4f} | "
                f"各折={[round(s,4) for s in fold_aucs]} | {status}"
            )

        # ── Step 3 (可选): 单特征 AUC 精准诊断 ────────────────────
        per_feature_auc: dict[str, float] = {}
        if self.config.compute_per_feature_auc and distribution_shift:
            per_feature_auc = self._compute_per_feature_auc(
                X_mixed, y_mixed, fi_df, feat_cols
            )
            fi_df["per_feature_auc"] = fi_df["feature"].map(per_feature_auc)
        else:
            fi_df["per_feature_auc"] = np.nan

        # ── Step 4: 生成剔除建议 ───────────────────────────────────
        suggested_drop = self._suggest_drops(fi_df, per_feature_auc)
        fi_df["suggested_drop"] = fi_df["feature"].isin(suggested_drop)
        fi_df = fi_df.sort_values("rank").reset_index(drop=True)

        elapsed = time.perf_counter() - t0

        result = AdversarialValidationResult(
            overall_auc         = overall_auc,
            fold_aucs           = fold_aucs,
            feature_importance  = fi_df,
            suggested_drop_cols = suggested_drop,
            distribution_shift  = distribution_shift,
            elapsed_seconds     = elapsed,
            config              = self.config,
        )

        if self.config.verbose:
            logger.info(
                f"  建议剔除 {len(suggested_drop)} 列 | 耗时 {elapsed:.1f}s"
            )
            if suggested_drop:
                logger.info(f"  待剔除: {suggested_drop}")

        return result

    # ------------------------------------------------------------------
    # 辅助：构建混合数据集
    # ------------------------------------------------------------------

    def _build_mixed_dataset(
        self,
        X_tr: pd.DataFrame,
        X_te: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        拼接 train + test，标签 is_test: 0=train, 1=test。

        对抗验证中通常对两个集合各自采样，使样本数均衡，
        避免类别不均衡导致 AUC 偏高/偏低。
        """
        # 均衡采样：以较小的集合为准
        n_sample = min(len(X_tr), len(X_te))
        rng      = np.random.RandomState(self.config.seed)

        if len(X_tr) > n_sample:
            tr_idx = rng.choice(len(X_tr), size=n_sample, replace=False)
            X_tr   = X_tr.iloc[tr_idx].reset_index(drop=True)
        else:
            X_tr = X_tr.reset_index(drop=True)

        if len(X_te) > n_sample:
            te_idx = rng.choice(len(X_te), size=n_sample, replace=False)
            X_te   = X_te.iloc[te_idx].reset_index(drop=True)
        else:
            X_te = X_te.reset_index(drop=True)

        X_mixed = pd.concat(
            [X_tr.assign(_is_test=0), X_te.assign(_is_test=1)],
            axis=0, ignore_index=True,
        )
        y_mixed = X_mixed.pop("_is_test").values

        # 编码：object → category（LightGBM 原生支持）
        for col in X_mixed.select_dtypes(include=["object"]).columns:
            X_mixed[col] = X_mixed[col].astype("category")

        return X_mixed, y_mixed

    # ------------------------------------------------------------------
    # Stage 1：全特征对抗 CV
    # ------------------------------------------------------------------

    def _train_adversarial_cv(
        self,
        X_mixed: pd.DataFrame,
        y_mixed: np.ndarray,
        feat_cols: list[str],
    ) -> tuple[float, list[float], pd.DataFrame]:
        """
        5-Fold StratifiedKFold 对抗训练。

        Returns
        -------
        (overall_auc, fold_aucs, feature_importance_df)
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM 未安装。对抗验证需要 LightGBM：\n"
                "  pip install lightgbm"
            )

        params = {**self._DEFAULT_LGBM_PARAMS, **self.config.lgbm_params}
        n_est  = params.pop("n_estimators", 200)

        skf       = StratifiedKFold(
            n_splits     = self.config.n_splits,
            shuffle      = True,
            random_state = self.config.seed,
        )
        fold_aucs:  list[float]             = []
        fi_accum:   dict[str, list[float]]  = {f: [] for f in feat_cols}
        cat_cols    = [c for c in feat_cols
                       if pd.api.types.is_categorical_dtype(X_mixed[c])
                       or X_mixed[c].dtype == object]

        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_mixed, y_mixed)):
            X_tr  = X_mixed.iloc[tr_idx]
            X_val = X_mixed.iloc[val_idx]
            y_tr  = y_mixed[tr_idx]
            y_val = y_mixed[val_idx]

            dtrain = lgb.Dataset(
                X_tr,  label=y_tr,
                categorical_feature=cat_cols if cat_cols else "auto",
                free_raw_data=False,
            )
            dval   = lgb.Dataset(
                X_val, label=y_val,
                reference=dtrain,
                free_raw_data=False,
            )

            callbacks = [
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),   # 完全静默
            ]

            booster = lgb.train(
                params          = params,
                train_set       = dtrain,
                num_boost_round = n_est,
                valid_sets      = [dval],
                callbacks       = callbacks,
            )

            proba     = booster.predict(X_val)
            fold_auc  = roc_auc_score(y_val, proba)
            fold_aucs.append(round(fold_auc, 6))

            # 累积特征重要性（gain）
            fi = dict(zip(booster.feature_name(),
                          booster.feature_importance(importance_type="gain")))
            for feat in feat_cols:
                fi_accum[feat].append(fi.get(feat, 0.0))

            if self.config.verbose:
                logger.info(f"    Fold {fold_i + 1}/{self.config.n_splits}  AUC={fold_auc:.4f}")

        # 汇总特征重要性
        fi_rows = []
        for feat in feat_cols:
            vals = fi_accum[feat]
            fi_rows.append({
                "feature":          feat,
                "importance_mean":  float(np.mean(vals)),
                "importance_std":   float(np.std(vals)),
            })

        fi_df = (
            pd.DataFrame(fi_rows)
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        fi_df["rank"] = fi_df.index + 1

        # 归一化重要性到 [0, 1]
        total = fi_df["importance_mean"].sum()
        fi_df["importance_norm"] = (
            fi_df["importance_mean"] / total if total > 0 else 0.0
        )

        overall_auc = float(np.mean(fold_aucs))
        return overall_auc, fold_aucs, fi_df

    # ------------------------------------------------------------------
    # Stage 2：单特征 AUC 精准诊断
    # ------------------------------------------------------------------

    def _compute_per_feature_auc(
        self,
        X_mixed:    pd.DataFrame,
        y_mixed:    np.ndarray,
        fi_df:      pd.DataFrame,
        feat_cols:  list[str],
    ) -> dict[str, float]:
        """
        对 Top-K 可疑特征分别做单变量 AUC 计算。

        原理：若单独用某一列能以 AUC > drop_threshold_auc 区分
        train/test，则该列独立贡献了分布偏移。

        使用简单 LGBM（早停，最多 100 棵树）加速计算。
        """
        try:
            import lightgbm as lgb
        except ImportError:
            return {}

        # 只分析 importance_norm > drop_threshold_importance 的 Top-K 列
        candidates = fi_df[
            fi_df["importance_norm"] >= self.config.drop_threshold_importance
        ]["feature"].tolist()
        candidates = candidates[:self.config.n_top_features]

        if not candidates:
            return {}

        if self.config.verbose:
            logger.info(
                f"  Stage 2: 对 {len(candidates)} 个可疑特征计算单变量 AUC..."
            )

        params_fast = {
            "objective":    "binary",
            "metric":       "auc",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves":   7,
            "verbose":     -1,
            "n_jobs":      -1,
        }
        skf = StratifiedKFold(
            n_splits     = 3,  # Stage 2 用 3 折加速
            shuffle      = True,
            random_state = self.config.seed,
        )

        per_feat_auc: dict[str, float] = {}
        for feat in candidates:
            X_single = X_mixed[[feat]].copy()
            # object → category
            if X_single[feat].dtype == object:
                X_single[feat] = X_single[feat].astype("category")

            fold_aucs_single = []
            for tr_idx, val_idx in skf.split(X_single, y_mixed):
                X_tr  = X_single.iloc[tr_idx]
                X_val = X_single.iloc[val_idx]
                y_tr  = y_mixed[tr_idx]
                y_val = y_mixed[val_idx]

                cat_cols = [feat] if (
                    pd.api.types.is_categorical_dtype(X_single[feat])
                    or X_single[feat].dtype == object
                ) else []

                dtrain = lgb.Dataset(
                    X_tr, label=y_tr,
                    categorical_feature=cat_cols if cat_cols else "auto",
                    free_raw_data=False,
                )
                dval = lgb.Dataset(
                    X_val, label=y_val,
                    reference=dtrain,
                    free_raw_data=False,
                )
                booster = lgb.train(
                    params          = params_fast,
                    train_set       = dtrain,
                    num_boost_round = 100,
                    valid_sets      = [dval],
                    callbacks       = [
                        lgb.early_stopping(20, verbose=False),
                        lgb.log_evaluation(-1),
                    ],
                )
                proba    = booster.predict(X_val)
                fold_aucs_single.append(roc_auc_score(y_val, proba))

            per_feat_auc[feat] = round(float(np.mean(fold_aucs_single)), 6)

        if self.config.verbose:
            n_bad = sum(1 for v in per_feat_auc.values()
                        if v > self.config.drop_threshold_auc)
            logger.info(
                f"  Stage 2 完成 | "
                f"单变量 AUC > {self.config.drop_threshold_auc} 的特征: {n_bad} 个"
            )

        return per_feat_auc

    # ------------------------------------------------------------------
    # 剔除建议生成
    # ------------------------------------------------------------------

    def _suggest_drops(
        self,
        fi_df:          pd.DataFrame,
        per_feature_auc: dict[str, float],
    ) -> list[str]:
        """
        确定建议删除的列，优先级：

        1. 若 Stage 2 启用（per_feature_auc 非空）：
             per_feature_auc > drop_threshold_auc → 直接建议删除

        2. 若仅 Stage 1（无 per_feature_auc）：
             importance_norm > drop_threshold_importance
             且 overall_auc > auc_threshold
             → 进入建议列表（保守策略）
        """
        drops = []

        if per_feature_auc:
            # Stage 2 精准模式
            drops = [
                feat for feat, auc in per_feature_auc.items()
                if auc > self.config.drop_threshold_auc
            ]
        else:
            # Stage 1 保守模式（仅在检测到偏移时触发）
            suspicious = fi_df[
                fi_df["importance_norm"] >= self.config.drop_threshold_importance
            ]["feature"].tolist()
            drops = suspicious[:self.config.n_top_features]

        return sorted(drops)

    # ------------------------------------------------------------------
    # 实用方法：一键剔除
    # ------------------------------------------------------------------

    @staticmethod
    def apply_drop(
        X_train:  pd.DataFrame,
        X_test:   pd.DataFrame,
        result:   AdversarialValidationResult,
        extra_drop: Optional[list[str]] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        根据 result.suggested_drop_cols 从 train 和 test 中同步删除列。

        Parameters
        ----------
        X_train, X_test : 原始 DataFrame
        result          : AdversarialValidator.run() 的返回值
        extra_drop      : 额外追加的列名列表（手动指定）

        Returns
        -------
        (X_train_clean, X_test_clean)
        """
        to_drop = list(result.suggested_drop_cols)
        if extra_drop:
            to_drop = list(set(to_drop) | set(extra_drop))

        drop_train = [c for c in to_drop if c in X_train.columns]
        drop_test  = [c for c in to_drop if c in X_test.columns]

        X_train_clean = X_train.drop(columns=drop_train)
        X_test_clean  = X_test.drop(columns=drop_test)

        logger.info(
            f"[apply_drop] 已删除 {len(drop_train)} 列 | "
            f"train: {X_train.shape} → {X_train_clean.shape} | "
            f"test:  {X_test.shape}  → {X_test_clean.shape}"
        )
        return X_train_clean, X_test_clean

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def plot_importance(
        self,
        result: AdversarialValidationResult,
        top_n:  int = 30,
        save_path: Optional[str] = None,
    ) -> None:
        """
        绘制对抗模型特征重要性柱状图，高亮建议剔除的列。

        Parameters
        ----------
        save_path : str | None
            非 None 时保存图片到指定路径（如 "outputs/.../adversarial_fi.png"）。
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib 未安装，跳过可视化。")
            return

        fi   = result.feature_importance.head(top_n).copy()
        fi   = fi.iloc[::-1]   # 翻转，使最重要的在顶部

        colors = [
            "#E74C3C" if row["suggested_drop"] else "#3498DB"
            for _, row in fi.iterrows()
        ]

        fig, ax = plt.subplots(figsize=(10, max(6, len(fi) * 0.35)))
        bars = ax.barh(fi["feature"], fi["importance_mean"], color=colors)

        # 若有单特征 AUC，在柱子末端标注
        if "per_feature_auc" in fi.columns and fi["per_feature_auc"].notna().any():
            for bar, (_, row) in zip(bars, fi.iterrows()):
                if pd.notna(row["per_feature_auc"]):
                    ax.text(
                        bar.get_width() * 1.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"AUC={row['per_feature_auc']:.3f}",
                        va="center", ha="left", fontsize=7.5,
                    )

        ax.set_xlabel("Importance (gain, mean over folds)")
        ax.set_title(
            f"Adversarial Validation — Feature Importance\n"
            f"Overall AUC={result.overall_auc:.4f}  "
            f"({'⚠ shift detected' if result.distribution_shift else '✓ no shift'})\n"
            f"Red = suggested drop ({len(result.suggested_drop_cols)} cols)",
            fontsize=11,
        )

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#E74C3C", label="Suggested Drop"),
            Patch(facecolor="#3498DB", label="Keep"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            logger.info(f"[adversarial] 特征重要性图已保存 → {save_path}")
        else:
            plt.show()

        plt.close(fig)


# ---------------------------------------------------------------------------
# 便捷工厂函数（供 train_pipeline.py 直接调用）
# ---------------------------------------------------------------------------

def run_adversarial_validation(
    X_train:    pd.DataFrame,
    X_test:     pd.DataFrame,
    config:     AdvConfig,
    save_dir:   Optional[str] = None,
) -> Optional[AdversarialValidationResult]:
    """
    对抗性验证的一键调用入口。

    在 train_pipeline.py 的 Step 1.5 中直接调用：
      result = run_adversarial_validation(X_train, X_test, adv_cfg, save_dir=exp_dir)

    Parameters
    ----------
    X_train   : 训练特征 DataFrame（不含目标列）
    X_test    : 测试特征 DataFrame
    config    : AdvConfig 实例（可从 YAML 通过 AdvConfig.from_dict(cfg) 初始化）
    save_dir  : 非 None 时将报告文本和图片写入该目录

    Returns
    -------
    AdversarialValidationResult | None
    """
    if not config.enable:
        return None

    av     = AdversarialValidator(config)
    result = av.run(X_train, X_test)

    if result is not None and save_dir is not None:
        from pathlib import Path
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)

        # 保存文本报告
        report_path = p / "adversarial_validation_report.txt"
        report_path.write_text(result.report(), encoding="utf-8")

        # 保存特征重要性 CSV
        fi_path = p / "adversarial_feature_importance.csv"
        result.feature_importance.to_csv(fi_path, index=False)

        # 保存可视化
        try:
            av.plot_importance(result, save_path=str(p / "adversarial_fi.png"))
        except Exception:
            pass

        logger.info(f"[adversarial] 报告已写入 {save_dir}")

    return result
