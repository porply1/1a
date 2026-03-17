"""
scripts/train_pipeline.py
--------------------------
军火库自动化全流程指挥部。

调用链路
--------
  YAML Config
      │
      ▼
  ① DataLoader          → 加载 + dtype 压缩
      │
      ▼
  ② FeaturePipeline     → fit_transform_oof（防泄露）
      │
      ▼
  ③ [OptunaTuner]       → 可选：自动超参搜索（per model）
      │
      ▼
  ④ CrossValidatorTrainer × N models
      │  OOF predictions / test predictions / feature importance
      ▼
  ⑤ [StackingEnsemble]  → 可选：元学习器融合
      │
      ▼
  ⑥ 持久化              → models / OOF / submission / report.json

命令行用法
----------
  python scripts/train_pipeline.py --config configs/template.yaml
  python scripts/train_pipeline.py --config configs/my_comp.yaml --no-tune
  python scripts/train_pipeline.py --config configs/my_comp.yaml --fold-only 0 1 2
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ── 将项目根目录加入 sys.path（scripts/ 运行时需要）─────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML 未安装，请执行：pip install pyyaml")

import sklearn.metrics as skm

from data.loader    import DataLoader
from data.splitter  import BaseCVSplitter, CVConfig, get_cv
from data.adversarial_validation import (
    AdversarialValidator,
    AdvConfig,
    run_adversarial_validation,
)
from data.negative_sampler import NegativeSamplerConfig
from core.base_model   import BaseModel
from core.base_trainer import CrossValidatorTrainer, MetricConfig, CVResult
from features.engine   import (
    FeaturePipeline,
    TimeFeatureTransformer,
    GroupAggTransformer,
    DiffTransformer,
    LagTransformer,
    TargetEncoderTransformer,
    FreqEncoderTransformer,
)
from features.selector import NullImportanceSelector
from post_process.optimizer import (
    ProbabilityCalibrator,
    ThresholdOptimizer,
)
from models.gbm.lgbm_wrapper      import LGBMModel
from models.gbm.xgb_wrapper       import XGBModel
from models.gbm.catboost_wrapper  import CatBoostModel
# ── 深度学习模型（可选，torch/deepctr-torch 未安装时降级为警告）────────────
try:
    from models.deep.deepfm_wrapper    import DeepFMModel
    from models.deep.din_wrapper       import DINModel, SeqFeatureConfig
    from models.deep.two_tower_wrapper import TwoTowerModel
    from models.deep.esmm_wrapper      import ESMMModel
    from models.deep.ple_wrapper       import PLEModel
    from models.deep.bst_wrapper       import BSTModel, BSTSeqConfig
    _DEEP_MODELS_AVAILABLE = True
except ImportError as _deep_err:
    _DEEP_MODELS_AVAILABLE = False
    _DEEP_IMPORT_ERROR     = str(_deep_err)

from ensemble.stacking            import StackingEnsemble, StackingConfig, HillClimbingConfig
from optimization.optuna_tuner    import OptunaTuner, TunerConfig
from utils.memory                 import memory_monitor


# ---------------------------------------------------------------------------
# 注册表（供 YAML 通过字符串反射实例化）
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    # ── GBM 三剑客 ────────────────────────────────────────────────────────
    "LGBMModel":     LGBMModel,
    "XGBModel":      XGBModel,
    "CatBoostModel": CatBoostModel,
}

# 深度模型在 torch/deepctr-torch 可用时动态注入注册表
if _DEEP_MODELS_AVAILABLE:
    _MODEL_REGISTRY.update({
        "DeepFMModel":    DeepFMModel,
        "DINModel":       DINModel,
        "TwoTowerModel":  TwoTowerModel,
        "ESMMModel":      ESMMModel,
        "PLEModel":       PLEModel,
        "BSTModel":       BSTModel,
    })
else:
    # 首次加载时打印一次警告，之后通过 _warn_deep_unavailable 按需触发
    import warnings as _w
    _w.warn(
        f"深度学习模型不可用（torch/deepctr-torch 未安装）：{_DEEP_IMPORT_ERROR}\n"
        "如需使用 DeepFM/DIN/TwoTower/ESMM/PLE，请执行：\n"
        "  pip install torch deepctr-torch",
        ImportWarning,
        stacklevel=2,
    )

_TRANSFORMER_REGISTRY = {
    "TimeFeatureTransformer":   TimeFeatureTransformer,
    "GroupAggTransformer":      GroupAggTransformer,
    "DiffTransformer":          DiffTransformer,
    "LagTransformer":           LagTransformer,
    "TargetEncoderTransformer": TargetEncoderTransformer,
    "FreqEncoderTransformer":   FreqEncoderTransformer,
}

_METRIC_REGISTRY: dict[str, Any] = {
    "roc_auc_score":      skm.roc_auc_score,
    "log_loss":           skm.log_loss,
    "accuracy_score":     skm.accuracy_score,
    "f1_score":           skm.f1_score,
    "mean_squared_error": skm.mean_squared_error,
    "mean_absolute_error":skm.mean_absolute_error,
    "r2_score":           skm.r2_score,
}


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

def _build_logger(exp_dir: Path) -> logging.Logger:
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # 文件
    fh = logging.FileHandler(exp_dir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# 实验目录管理
# ---------------------------------------------------------------------------

def _create_exp_dir(base_dir: str, exp_name: str) -> Path:
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{exp_name}_{ts}"
    (exp_dir / "models").mkdir(parents=True, exist_ok=True)
    (exp_dir / "oof").mkdir(parents=True, exist_ok=True)
    (exp_dir / "submissions").mkdir(parents=True, exist_ok=True)
    (exp_dir / "importance").mkdir(parents=True, exist_ok=True)
    (exp_dir / "feature_selection").mkdir(parents=True, exist_ok=True)
    (exp_dir / "post_process").mkdir(parents=True, exist_ok=True)
    return exp_dir


# ---------------------------------------------------------------------------
# YAML 配置解析
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def _resolve_metric(metric_cfg: dict) -> MetricConfig:
    fn_name = metric_cfg["func"]
    if fn_name not in _METRIC_REGISTRY:
        raise ValueError(
            f"未知指标函数 '{fn_name}'。\n"
            f"可选：{list(_METRIC_REGISTRY.keys())}"
        )
    return MetricConfig(
        name             = metric_cfg["name"],
        fn               = _METRIC_REGISTRY[fn_name],
        higher_is_better = metric_cfg.get("higher_is_better", True),
        primary          = True,
        use_proba        = metric_cfg.get("use_proba", False),
    )


def _build_transformers(feat_cfg: dict) -> list:
    tfs = []
    for tf_cfg in feat_cfg.get("transformers", []):
        if not tf_cfg.get("enabled", True):
            continue
        cls_name = tf_cfg["type"]
        if cls_name not in _TRANSFORMER_REGISTRY:
            raise ValueError(
                f"未知 Transformer 类型：'{cls_name}'。\n"
                f"可选：{list(_TRANSFORMER_REGISTRY.keys())}"
            )
        params = tf_cfg.get("params", {})
        tfs.append(_TRANSFORMER_REGISTRY[cls_name](**params))
    return tfs


def _build_model(model_cfg: dict, task: str, seed: int) -> BaseModel:
    cls_name = model_cfg["type"]
    if cls_name not in _MODEL_REGISTRY:
        # 深度模型被配置但环境不支持时给出明确错误
        _deep_names = {"DeepFMModel", "DINModel", "TwoTowerModel", "ESMMModel", "PLEModel", "BSTModel"}
        if cls_name in _deep_names and not _DEEP_MODELS_AVAILABLE:
            raise RuntimeError(
                f"模型 '{cls_name}' 需要 PyTorch 和 DeepCTR-Torch，但当前环境不可用。\n"
                f"原始错误：{_DEEP_IMPORT_ERROR}\n"
                "请执行：pip install torch deepctr-torch"
            )
        raise ValueError(
            f"未知模型类型：'{cls_name}'。\n"
            f"可选：{list(_MODEL_REGISTRY.keys())}"
        )

    cls    = _MODEL_REGISTRY[cls_name]
    params = model_cfg.get("params", {})
    kwargs: dict = {
        "params": params,
        "task":   task,
        "name":   model_cfg.get("name", cls_name.lower()),
        "seed":   seed,
    }

    # ── GBM 三剑客专属参数 ─────────────────────────────────────────────
    if cls_name == "LGBMModel":
        kwargs["num_boost_round"]       = model_cfg.get("num_boost_round", 1000)
        kwargs["early_stopping_rounds"] = model_cfg.get("early_stopping_rounds", 100)
        kwargs["log_period"]            = model_cfg.get("log_period", 200)
        kwargs["fi_type"]               = model_cfg.get("fi_type", "gain")

    elif cls_name == "XGBModel":
        kwargs["num_boost_round"]       = model_cfg.get("num_boost_round", 1000)
        kwargs["early_stopping_rounds"] = model_cfg.get("early_stopping_rounds", 100)
        kwargs["log_period"]            = model_cfg.get("log_period", 200)
        kwargs["fi_type"]               = model_cfg.get("fi_type", "total_gain")

    elif cls_name == "CatBoostModel":
        kwargs["iterations"]            = model_cfg.get("iterations", 1000)
        kwargs["early_stopping_rounds"] = model_cfg.get("early_stopping_rounds", 100)
        kwargs["log_period"]            = model_cfg.get("log_period", 200)
        kwargs["fi_type"]               = model_cfg.get("fi_type", "LossFunctionChange")

    # ── 深度模型通用参数 ───────────────────────────────────────────────
    # DeepFM / DIN / TwoTower / ESMM 共享：无额外必填构造参数，
    # 全部超参统一通过 params 字典注入，Early Stopping 由各 Wrapper 内部管理。

    elif cls_name == "DINModel":
        # DIN 专属：序列特征配置（YAML 中以列表形式声明）
        # 示例 YAML：
        #   seq_configs:
        #     - seq_col: hist_item_id_list
        #       target_col: item_id
        #       maxlen: 50
        raw_seq = model_cfg.get("seq_configs") or []
        kwargs["seq_configs"] = [
            SeqFeatureConfig(**c) if isinstance(c, dict) else c
            for c in raw_seq
        ]

    elif cls_name == "TwoTowerModel":
        # TwoTower 专属：必须显式声明 user/item 列分组
        # 示例 YAML：
        #   user_cols: [user_id, age, gender]
        #   item_cols: [item_id, cate_id, price]
        user_cols = model_cfg.get("user_cols")
        item_cols = model_cfg.get("item_cols")
        if not user_cols or not item_cols:
            raise ValueError(
                "TwoTowerModel 必须在 YAML 中声明 user_cols 和 item_cols。\n"
                "示例：\n"
                "  user_cols: [user_id, age, gender]\n"
                "  item_cols: [item_id, cate_id, price]"
            )
        kwargs["user_cols"] = user_cols
        kwargs["item_cols"] = item_cols

    # ESMMModel 无额外必填参数，label_click/label_conversion 列名
    # 通过 params 传入（已在 ESMMParams 中有默认值）。

    elif cls_name == "BSTModel":
        # BST 专属：序列特征配置（与 DINModel 格式完全一致）
        # 示例 YAML：
        #   seq_configs:
        #     - seq_col:     hist_item_id_list
        #       target_col:  item_id
        #       timegap_col: hist_timegap_list   # 可选
        #       maxlen:      50
        raw_seq = model_cfg.get("seq_configs") or []
        kwargs["seq_configs"] = [
            BSTSeqConfig(**c) if isinstance(c, dict) else c
            for c in raw_seq
        ]

    return cls(**kwargs)


def _build_cv(cv_cfg: dict) -> tuple[BaseCVSplitter, Optional[str]]:
    """返回 (splitter, group_col)。"""
    group_col = cv_cfg.get("group_col")
    splitter  = get_cv(CVConfig(
        strategy     = cv_cfg.get("strategy", "stratified_kfold"),
        n_splits     = cv_cfg.get("n_splits", 5),
        shuffle      = cv_cfg.get("shuffle", True),
        random_state = cv_cfg.get("random_state", 42),
        gap          = cv_cfg.get("gap", 0),
    ))
    return splitter, group_col


# ---------------------------------------------------------------------------
# 持久化工具
# ---------------------------------------------------------------------------

def _save_oof(oof: np.ndarray, name: str, exp_dir: Path) -> None:
    path = exp_dir / "oof" / f"{name}_oof.npy"
    np.save(str(path), oof)

def _save_feature_importance(fi_df: pd.DataFrame, name: str, exp_dir: Path) -> None:
    csv_path = exp_dir / "importance" / f"{name}_importance.csv"
    fi_df.to_csv(csv_path, index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = fi_df.head(30)
        fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.3)))
        ax.barh(top["feature"][::-1], top["importance_mean"][::-1], color="#4C72B0")
        ax.set_xlabel("Importance (normalized)")
        ax.set_title(f"{name} — Top-{len(top)} Feature Importance")
        plt.tight_layout()
        fig.savefig(exp_dir / "importance" / f"{name}_importance.png", dpi=120)
        plt.close(fig)
    except Exception:
        pass   # matplotlib 不可用时静默跳过图表

def _save_submission(
    pred:        np.ndarray,
    test_ids:    Optional[pd.Series],
    id_col:      Optional[str],
    pred_col:    str,
    exp_dir:     Path,
    suffix:      str = "final",
) -> None:
    df = pd.DataFrame({pred_col: pred})
    if test_ids is not None and id_col:
        df.insert(0, id_col, test_ids.values)
    path = exp_dir / "submissions" / f"submission_{suffix}.csv"
    df.to_csv(path, index=(id_col is None))

def _save_report(report: dict, exp_dir: Path) -> None:
    with open(exp_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# 主流水线类
# ---------------------------------------------------------------------------

class TrainPipeline:
    """
    全链路训练流水线，由 YAML 配置驱动。

    使用方法
    --------
    >>> pipeline = TrainPipeline("configs/template.yaml")
    >>> pipeline.run()
    """

    def __init__(
        self,
        config_path:  str,
        no_tune:      bool       = False,
        fold_only:    Optional[list[int]] = None,
    ):
        self.cfg        = load_config(config_path)
        self.config_path = config_path
        self.no_tune    = no_tune
        self.fold_only  = fold_only

        # 实验目录
        exp_cfg     = self.cfg["experiment"]
        output_cfg  = self.cfg.get("output", {})
        self.exp_dir = _create_exp_dir(
            base_dir = output_cfg.get("base_dir", "outputs"),
            exp_name = exp_cfg["name"],
        )
        self.logger = _build_logger(self.exp_dir)

        # 保存配置副本到实验目录
        import shutil
        shutil.copy(config_path, self.exp_dir / "config.yaml")

        self.report: dict = {
            "experiment":  exp_cfg["name"],
            "config_path": str(config_path),
            "exp_dir":     str(self.exp_dir),
            "started_at":  datetime.now().isoformat(),
            "models":      {},
            "ensemble":    {},
            "timing":      {},
        }
        # 存储 Step 1.5 的 AV AUC，供 Step 3.5 二阶对比使用
        self._pre_fe_av_auc: Optional[float] = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(self) -> dict:
        t_total = time.perf_counter()
        cfg     = self.cfg
        self._banner("🚀  军火库流水线启动")

        # ── Step 1: 加载数据 ───────────────────────────────────────────
        train_df, test_df, y, test_ids = self._step_load_data()

        # ── Step 1.5: 对抗性验证（可选）──────────────────────────────
        train_df, test_df = self._step_adversarial_validation(train_df, test_df, y)

        # ── Step 2: 构建 CV ────────────────────────────────────────────
        cv, group_col = _build_cv(cfg["cv"])
        groups = train_df[group_col] if group_col and group_col in train_df.columns else None
        X_raw  = train_df.drop(columns=[cfg["data"]["target_col"]] +
                               ([group_col] if group_col else []))
        X_test_raw = test_df.drop(
            columns=[c for c in [cfg["data"].get("id_col")] if c and c in test_df.columns],
            errors="ignore",
        ) if test_df is not None else None

        # ── Step 3: 特征工程 ───────────────────────────────────────────
        X_train, X_test = self._step_feature_engineering(
            X_raw, y, X_test_raw, cv, groups
        )

        # ── Step 3.5: 特征工程后二阶对抗验证 ─────────────────────────
        self._step_adversarial_validation_post_fe(X_train, X_test)

        # ── Step 3.7: Null Importance 特征筛选（可选）────────────────
        X_train, X_test = self._step_feature_selection(X_train, y, X_test)

        # ── Step 4: 构建指标 ───────────────────────────────────────────
        primary_metric, all_metrics = self._build_metrics()

        # ── Step 5: 多模型训练 ─────────────────────────────────────────
        cv_results: dict[str, CVResult] = {}
        model_names: list[str]          = []
        enabled_models = [m for m in cfg["models"] if m.get("enabled", True)]

        for model_cfg in enabled_models:
            name, result = self._step_train_model(
                model_cfg, X_train, y, X_test, cv, groups,
                all_metrics, primary_metric,
            )
            cv_results[name] = result
            model_names.append(name)
            self._save_model_artifacts(name, result)

        # ── Step 5.5: 后处理（概率校准 + 阈值优化，可选）──────────────
        cv_results = self._step_post_process(cv_results, y)

        # ── Step 6: Stacking 集成 ──────────────────────────────────────
        final_pred = self._step_ensemble(
            cv_results, model_names, y, X_train, X_test, primary_metric
        )

        # ── Step 7: 保存提交文件 ───────────────────────────────────────
        if final_pred is not None:
            output_cfg = cfg.get("output", {})
            _save_submission(
                pred     = final_pred,
                test_ids = test_ids,
                id_col   = cfg["data"].get("id_col"),
                pred_col = output_cfg.get("submission_col", "target"),
                exp_dir  = self.exp_dir,
                suffix   = "final",
            )
            self.logger.info(f"提交文件已保存 → {self.exp_dir}/submissions/")

        # ── Step 8: 报告 ───────────────────────────────────────────────
        self.report["finished_at"] = datetime.now().isoformat()
        self.report["timing"]["total_seconds"] = round(
            time.perf_counter() - t_total, 1
        )
        _save_report(self.report, self.exp_dir)

        self._print_final_report()
        self.logger.info(f"实验目录：{self.exp_dir}")
        return self.report

    # ------------------------------------------------------------------
    # Step 1: 数据加载
    # ------------------------------------------------------------------

    def _step_load_data(self):
        self._banner("Step 1 / 7  数据加载")
        cfg      = self.cfg
        data_cfg = cfg["data"]
        t0       = time.perf_counter()

        loader = DataLoader(
            compress     = True,
            parse_dates  = data_cfg.get("parse_dates") or [],
            verbose      = True,
        )
        train_df = loader.load(data_cfg["train_path"])
        test_df  = None
        if data_cfg.get("test_path"):
            test_df = loader.load(data_cfg["test_path"])

        # 删除指定列
        drop_cols = data_cfg.get("drop_cols") or []
        for col in drop_cols:
            for df in [train_df, test_df]:
                if df is not None and col in df.columns:
                    df.drop(columns=[col], inplace=True)

        # 提取标签 & ID
        target_col = data_cfg["target_col"]
        id_col     = data_cfg.get("id_col")

        y        = train_df[target_col]
        test_ids = test_df[id_col] if (test_df is not None and id_col
                                       and id_col in test_df.columns) else None

        elapsed = time.perf_counter() - t0
        self.report["timing"]["data_load"] = round(elapsed, 1)
        self.logger.info(
            f"训练集: {train_df.shape} | 测试集: "
            f"{test_df.shape if test_df is not None else 'N/A'} | "
            f"耗时 {elapsed:.1f}s"
        )
        return train_df, test_df, y, test_ids

    # ------------------------------------------------------------------
    # Step 1.5: 对抗性验证
    # ------------------------------------------------------------------

    def _step_adversarial_validation(
        self,
        train_df: pd.DataFrame,
        test_df:  Optional[pd.DataFrame],
        y:        pd.Series,
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        对抗性验证入口。

        执行时机：数据加载后、特征工程前。
        行为：
          - config.adversarial_validation.enable=False → 直接透传，零开销
          - enable=True，auto_drop=False              → 打印报告，不修改数据
          - enable=True，auto_drop=True               → 自动剔除建议列，返回干净数据
        """
        adv_cfg_dict = self.cfg.get("adversarial_validation", {})
        if not adv_cfg_dict.get("enable", False):
            return train_df, test_df
        if test_df is None:
            self.logger.warning(
                "[adversarial] 未提供测试集（test_path=null），跳过对抗验证。"
            )
            return train_df, test_df

        adv_cfg = AdvConfig.from_dict(adv_cfg_dict)
        cfg_data = self.cfg["data"]

        # 从 train_df 中去除标签列，得到纯特征 DataFrame
        target_col = cfg_data["target_col"]
        id_col     = cfg_data.get("id_col")
        group_col  = self.cfg.get("cv", {}).get("group_col")

        meta_cols  = [c for c in [target_col, id_col, group_col]
                      if c and c in train_df.columns]
        X_tr_feat  = train_df.drop(columns=meta_cols)
        X_te_feat  = test_df.drop(
            columns=[c for c in [id_col] if c and c in test_df.columns],
            errors="ignore",
        )

        # 合并 ignore_cols：用户配置 + 自动识别的元列
        auto_ignore = [c for c in meta_cols if c in X_tr_feat.columns]
        adv_cfg.ignore_cols = list(
            set(adv_cfg.ignore_cols) | set(auto_ignore)
        )

        result = run_adversarial_validation(
            X_train  = X_tr_feat,
            X_test   = X_te_feat,
            config   = adv_cfg,
            save_dir = str(self.exp_dir),
        )

        if result is None:
            return train_df, test_df

        # 写入 report
        self.report["adversarial_validation"] = {
            "overall_auc":         result.overall_auc,
            "distribution_shift":  result.distribution_shift,
            "suggested_drop_cols": result.suggested_drop_cols,
            "elapsed_s":           round(result.elapsed_seconds, 1),
        }
        # 缓存供 Step 3.5 二阶对比
        self._pre_fe_av_auc = result.overall_auc

        self.logger.info(result.report())

        if adv_cfg.auto_drop and result.suggested_drop_cols:
            train_df, test_df = AdversarialValidator.apply_drop(
                train_df, test_df, result
            )
            self.logger.info(
                f"[adversarial] auto_drop=True | "
                f"已从训练集和测试集删除 {len(result.suggested_drop_cols)} 列"
            )

        return train_df, test_df

    # ------------------------------------------------------------------
    # Step 3.5: 二阶对抗验证（特征工程后）
    # ------------------------------------------------------------------

    def _step_adversarial_validation_post_fe(
        self,
        X_train: pd.DataFrame,
        X_test:  Optional[pd.DataFrame],
    ) -> None:
        """
        在特征工程结束后再次执行对抗验证。

        逻辑：
          - adversarial_validation.enable=False → 静默跳过
          - X_test 为 None → 跳过
          - _pre_fe_av_auc 未设置（Step 1.5 未运行） → 跳过
          - post_FE_AUC - pre_FE_AUC > 0.05 → ⚠ 强制打印 Top-5 漂移特征
            并建议指挥官检查特征逻辑
        """
        adv_cfg_dict = self.cfg.get("adversarial_validation", {})
        if not adv_cfg_dict.get("enable", False):
            return
        if X_test is None:
            return
        if self._pre_fe_av_auc is None:
            return

        self._banner("Step 3.5 / 7  二阶对抗验证（特征工程后）")
        t0 = time.perf_counter()

        adv_cfg          = AdvConfig.from_dict(adv_cfg_dict)
        adv_cfg.auto_drop            = False   # 诊断用途，不修改数据
        adv_cfg.compute_per_feature_auc = False  # 加速（只需 Stage 1 判断）

        # 只保留数值列（排除序列列、object 列等无法直接对比的列）
        num_cols_tr = X_train.select_dtypes(include=np.number).columns.tolist()
        num_cols_te = X_test.select_dtypes(include=np.number).columns.tolist()
        shared_num  = [c for c in num_cols_tr if c in num_cols_te]

        if not shared_num:
            self.logger.warning(
                "[Step 3.5] 特征工程后无共同数值列可比较，跳过二阶对抗验证。"
            )
            return

        result_post = run_adversarial_validation(
            X_train  = X_train[shared_num],
            X_test   = X_test[shared_num],
            config   = adv_cfg,
            save_dir = None,   # 不重复写文件
        )
        if result_post is None:
            return

        post_auc = result_post.overall_auc
        pre_auc  = self._pre_fe_av_auc
        delta    = post_auc - pre_auc
        elapsed  = time.perf_counter() - t0

        # 写入 report
        self.report.setdefault("adversarial_validation", {}).update({
            "post_fe_auc":   round(post_auc, 6),
            "fe_auc_delta":  round(delta, 6),
        })

        self.logger.info(
            f"[Step 3.5] AV 对比 | "
            f"原始数据 AUC={pre_auc:.4f} | "
            f"特征工程后 AUC={post_auc:.4f} | "
            f"Δ={delta:+.4f} | 耗时 {elapsed:.1f}s"
        )

        DRIFT_THRESHOLD = 0.05
        if delta > DRIFT_THRESHOLD:
            top5 = result_post.feature_importance.head(5)

            self.logger.warning(
                "\n" + "⚠ " * 25 + "\n"
                "  [二阶对抗验证 — 漂移警告]\n"
                f"  特征工程后 AUC 提升 {delta:+.4f}，超过阈值 {DRIFT_THRESHOLD}！\n"
                "  特征工程可能向模型引入了与测试集分布差异高度相关的信号。\n"
                "  建议指挥官执行以下回滚/检查操作：\n"
                "    1. 确认 use_oof_transform: true（目标编码等必须折内计算）\n"
                "    2. 检查 Lag / Diff Transformer 是否使用了未来数据\n"
                "    3. 在 features.transformers 中逐一禁用可疑 Transformer 后重跑\n"
                + "⚠ " * 25
            )
            self.logger.warning("  漂移最严重的 Top-5 特征（重点检查）：")
            for rank, (_, row) in enumerate(top5.iterrows(), 1):
                self.logger.warning(
                    f"    {rank}. {str(row['feature']):<40} "
                    f"importance={row['importance_mean']:.6f}"
                )
        else:
            self.logger.info(
                "[Step 3.5] ✓ 特征工程后漂移增量在可接受范围内，无需回滚。"
            )

    # ------------------------------------------------------------------
    # Step 3: 特征工程
    # ------------------------------------------------------------------

    def _step_feature_engineering(self, X_raw, y, X_test_raw, cv, groups):
        feat_cfg = self.cfg.get("features", {})
        if not feat_cfg.get("enable", False):
            self.logger.info("特征工程已禁用，使用原始特征。")
            return X_raw, X_test_raw

        self._banner("Step 3 / 7  特征工程")
        t0 = time.perf_counter()

        transformers = _build_transformers(feat_cfg)
        if not transformers:
            self.logger.info("未配置任何 Transformer，跳过特征工程。")
            return X_raw, X_test_raw

        pipeline = FeaturePipeline(
            transformers  = transformers,
            compress      = True,
            parallel      = False,
            keep_original = True,
            verbose       = True,
        )

        if feat_cfg.get("use_oof_transform", True):
            folds = cv.split(X_raw, y, groups=groups)
            X_train, X_test = pipeline.fit_transform_oof(
                X_raw, y, folds=folds, X_test=X_test_raw
            )
        else:
            X_train = pipeline.fit_transform(X_raw, y)
            X_test  = pipeline.transform(X_test_raw) if X_test_raw is not None else None

        elapsed = time.perf_counter() - t0
        self.report["timing"]["feature_engineering"] = round(elapsed, 1)
        self.logger.info(
            f"特征工程完成 | train={X_train.shape} | "
            f"test={X_test.shape if X_test is not None else 'N/A'} | "
            f"耗时 {elapsed:.1f}s"
        )
        return X_train, X_test

    # ------------------------------------------------------------------
    # Step 3.7: Null Importance 特征筛选（可选，在特征工程后、训练前）
    # ------------------------------------------------------------------

    def _step_feature_selection(
        self,
        X_train: pd.DataFrame,
        y:       pd.Series,
        X_test:  Optional[pd.DataFrame],
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Null Importance 置换检验特征筛选。

        配置开关：feature_selection.enable=true
        执行时机：特征工程结束后、CV 训练开始前。
        产出归档：{exp_dir}/feature_selection/
          - selected_features.json  : 保留特征名列表 + 统计摘要
          - null_importance_scores.csv : 每个特征的诊断分数
        """
        fs_cfg = self.cfg.get("feature_selection", {})
        if not fs_cfg.get("enable", False):
            return X_train, X_test

        self._banner("Step 3.7  Null Importance 特征筛选")
        t0   = time.perf_counter()
        seed = fs_cfg.get("random_state") or self.cfg["experiment"].get("seed", 42)

        # task: null → 继承 experiment.task
        task = fs_cfg.get("task") or self.cfg["experiment"]["task"]
        # 三值映射（multiclass 直接透传，其余不变）
        task = task if task in ("binary", "multiclass", "regression") else "binary"

        sel = NullImportanceSelector(
            n_iterations         = int(fs_cfg.get("n_iterations", 80)),
            percentile_threshold = float(fs_cfg.get("percentile_threshold", 75.0)),
            task                 = task,
            max_samples          = fs_cfg.get("max_samples"),
            gain_threshold       = float(fs_cfg.get("gain_threshold", 0.0)),
            random_state         = int(seed),
            verbose              = True,
        )

        # Inf → NaN 预处理（不改变原 DataFrame，只做 fit 用）
        X_clean = X_train.replace([np.inf, -np.inf], np.nan)
        sel.fit(X_clean, y)

        selected = sel.selected_features_
        n_orig   = X_train.shape[1]
        n_sel    = len(selected)

        # ── 持久化产物 ──────────────────────────────────────────────
        fs_dir = self.exp_dir / "feature_selection"

        # 1. 保留特征名单（JSON）
        json.dump(
            {
                "n_original":       n_orig,
                "n_selected":       n_sel,
                "drop_ratio":       round(1 - n_sel / max(n_orig, 1), 4),
                "selected_features": selected,
                "dropped_features": [
                    c for c in X_train.columns if c not in selected
                ],
                "selector_params": {
                    "n_iterations":         int(fs_cfg.get("n_iterations", 80)),
                    "percentile_threshold": float(fs_cfg.get("percentile_threshold", 75.0)),
                    "task":                 task,
                    "max_samples":          fs_cfg.get("max_samples"),
                    "gain_threshold":       float(fs_cfg.get("gain_threshold", 0.0)),
                },
            },
            open(fs_dir / "selected_features.json", "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )

        # 2. 特征分数诊断表（CSV）
        sel.get_scores().to_csv(
            fs_dir / "null_importance_scores.csv", index=False
        )

        # 3. 可选：重要性条形图（matplotlib 不可用时静默跳过）
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            scores_df = sel.get_scores()
            top = scores_df.head(40)
            fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.28)))
            colors = ["#4C72B0" if row["selected"] else "#DD8452"
                      for _, row in top.iterrows()]
            ax.barh(
                top["feature"][::-1],
                top["real_importance"][::-1],
                color=colors[::-1],
            )
            ax.set_xlabel("Real Importance (gain)")
            ax.set_title(
                f"Null Importance — Top-40  "
                f"[蓝=保留 {n_sel}  橙=丢弃 {n_orig - n_sel}]"
            )
            plt.tight_layout()
            fig.savefig(fs_dir / "null_importance_plot.png", dpi=120)
            plt.close(fig)
        except Exception:
            pass

        # ── 应用筛选 ──────────────────────────────────────────────
        keep = [c for c in selected if c in X_train.columns]
        X_train_out = X_train[keep]
        X_test_out  = X_test[keep] if X_test is not None else None

        elapsed = time.perf_counter() - t0
        self.report["feature_selection"] = {
            "n_original":     n_orig,
            "n_selected":     n_sel,
            "drop_ratio":     round(1 - n_sel / max(n_orig, 1), 4),
            "timing_s":       round(elapsed, 1),
        }
        self.report["timing"]["feature_selection"] = round(elapsed, 1)
        self.logger.info(
            f"特征筛选完成 | {n_orig} → {n_sel} 特征 "
            f"(丢弃 {n_orig - n_sel}) | 耗时 {elapsed:.1f}s"
        )
        return X_train_out, X_test_out

    # ------------------------------------------------------------------
    # Step 5.5: 后处理（概率校准 + 阈值优化，可选）
    # ------------------------------------------------------------------

    def _step_post_process(
        self,
        cv_results: dict,
        y:          pd.Series,
    ) -> dict:
        """
        对每个模型的 OOF / 测试集预测进行后处理：
          - ProbabilityCalibrator : Isotonic / Platt 概率校准
          - ThresholdOptimizer    : Optuna 搜索最优决策阈值（binary 任务）

        执行时机：所有模型 CV 训练完成后、Stacking 集成之前。
        注意事项：
          - oof_predictions 不被修改（保留原值供 Stacking 使用，
            防止元学习器在已校准空间上再次学习导致信息损失）
          - test_predictions 如 apply_to_test=True 则替换为校准后的值
          - 所有参数和报告自动归档到 {exp_dir}/post_process/

        产出归档：{exp_dir}/post_process/{model_name}/
          - calibrator.pkl         : 已拟合的校准器（可用于线上服务）
          - threshold.json         : 最优阈值 + OOF 指标
          - calibration_report.json: Brier/LogLoss 前后对比
        """
        pp_cfg = self.cfg.get("post_process", {})
        if not pp_cfg.get("enable", False):
            return cv_results

        self._banner("Step 5.5  后处理（概率校准 + 阈值优化）")

        import pickle

        task          = self.cfg["experiment"]["task"]
        apply_to_test = pp_cfg.get("apply_to_test", True)
        cal_cfg       = pp_cfg.get("calibration", {})
        thr_cfg       = pp_cfg.get("threshold", {})
        pp_dir        = self.exp_dir / "post_process"
        y_arr         = y.values if hasattr(y, "values") else np.asarray(y)

        self.report["post_process"] = {}

        for model_name, result in cv_results.items():
            model_pp_dir = pp_dir / model_name
            model_pp_dir.mkdir(parents=True, exist_ok=True)

            oof_orig  = result.oof_predictions.copy()
            test_orig = (result.test_predictions.copy()
                         if result.test_predictions is not None else None)

            cal_oof  = oof_orig.copy()
            cal_test = test_orig.copy() if test_orig is not None else None

            model_report: dict = {
                "model":               model_name,
                "task":                task,
                "calibration_method":  "none",
                "best_threshold":      0.5,
                "best_threshold_score": float("nan"),
                "threshold_metric":    "f1",
            }

            # ── 概率校准 ────────────────────────────────────────────
            if cal_cfg.get("enable", True) and task in ("binary", "multiclass"):
                cal = ProbabilityCalibrator(
                    method       = cal_cfg.get("method", "isotonic"),
                    n_folds      = int(cal_cfg.get("n_folds", 5)),
                    verbose      = True,
                )
                try:
                    cal_oof = cal.fit_transform(y_arr, oof_orig)
                    if apply_to_test and cal_test is not None:
                        cal_test = cal.transform(cal_test)

                    # 持久化校准器
                    with open(model_pp_dir / "calibrator.pkl", "wb") as fh:
                        pickle.dump(cal, fh)

                    model_report["calibration_method"] = cal_cfg.get("method", "isotonic")
                    model_report["brier_before"] = float(
                        np.mean((oof_orig - y_arr) ** 2)
                    )
                    model_report["brier_after"] = float(
                        np.mean((cal_oof - y_arr) ** 2)
                    )
                    model_report["brier_improvement"] = round(
                        model_report["brier_before"] - model_report["brier_after"], 6
                    )
                    self.logger.info(
                        f"  [{model_name}] 校准完成 | "
                        f"Brier: {model_report['brier_before']:.5f} → "
                        f"{model_report['brier_after']:.5f} "
                        f"(Δ={-model_report['brier_improvement']:+.5f})"
                    )
                except Exception as exc:
                    warnings.warn(
                        f"[{model_name}] 概率校准失败（{exc}），使用原始预测。"
                    )

            # ── 阈值优化（仅 binary）────────────────────────────────
            if thr_cfg.get("enable", True) and task == "binary":
                thr = ThresholdOptimizer(
                    metric       = thr_cfg.get("metric", "f1"),
                    beta         = float(thr_cfg.get("beta", 1.0)),
                    n_trials     = int(thr_cfg.get("n_trials", 300)),
                    search_range = tuple(thr_cfg.get("search_range", [0.01, 0.99])),
                    verbose      = True,
                )
                try:
                    thr.fit(y_arr, cal_oof)
                    model_report["best_threshold"]       = float(thr.best_threshold_)
                    model_report["best_threshold_score"] = float(thr.best_score_)
                    model_report["threshold_metric"]     = str(thr_cfg.get("metric", "f1"))

                    # 持久化阈值
                    json.dump(
                        {
                            "best_threshold":       float(thr.best_threshold_),
                            "best_score":           float(thr.best_score_),
                            "metric":               str(thr_cfg.get("metric", "f1")),
                            "default_threshold":    0.5,
                            "calibration_method":   model_report["calibration_method"],
                        },
                        open(model_pp_dir / "threshold.json", "w", encoding="utf-8"),
                        indent=2,
                    )
                    self.logger.info(
                        f"  [{model_name}] 阈值优化完成 | "
                        f"最优阈值={thr.best_threshold_:.4f} | "
                        f"{thr_cfg.get('metric','f1')}={thr.best_score_:.5f}"
                    )
                except Exception as exc:
                    warnings.warn(
                        f"[{model_name}] 阈值优化失败（{exc}），使用默认阈值 0.5。"
                    )

            # ── 持久化校准报告 ───────────────────────────────────────
            json.dump(
                model_report,
                open(model_pp_dir / "calibration_report.json", "w", encoding="utf-8"),
                indent=2,
                default=str,
            )

            # ── 保存校准后的 OOF（独立存储，不覆盖原始 OOF）────────
            np.save(
                str(self.exp_dir / "oof" / f"{model_name}_oof_calibrated.npy"),
                cal_oof,
            )

            # ── 更新 test_predictions（用于最终提交）────────────────
            # oof_predictions 保持不变，供 Stacking 元学习器使用
            if apply_to_test and cal_test is not None and result.test_predictions is not None:
                result.test_predictions[:] = cal_test
                self.logger.info(
                    f"  [{model_name}] 测试集预测已替换为校准后的概率"
                )

            self.report["post_process"][model_name] = model_report

        return cv_results

    # ------------------------------------------------------------------
    # Step 5: 单模型训练（含可选调参）
    # ------------------------------------------------------------------

    def _step_train_model(
        self, model_cfg, X_train, y, X_test, cv, groups,
        all_metrics, primary_metric,
    ) -> tuple[str, CVResult]:
        name = model_cfg.get("name", model_cfg["type"].lower())
        self._banner(f"Step 5 / 7  模型训练 [{name}]")

        task = self.cfg["experiment"]["task"]
        seed = self.cfg["experiment"].get("seed", 42)
        model = _build_model(model_cfg, task, seed)

        # ── 动态负采样配置注入（深度模型专属）────────────────────────
        ns_dict = model_cfg.get("negative_sampling")
        if ns_dict and ns_dict.get("enable", False):
            if hasattr(model, "negative_sampler_config"):
                model.negative_sampler_config = NegativeSamplerConfig.from_dict(ns_dict)
                self.logger.info(
                    f"  → [{name}] 动态负采样已启用 | "
                    f"neg_ratio={ns_dict.get('neg_ratio', 4)} | "
                    f"buffer={ns_dict.get('buffer_size', 3)}"
                )
            else:
                self.logger.warning(
                    f"  → [{name}] negative_sampling 在 YAML 中已配置，"
                    f"但该模型不支持动态负采样（未实现 negative_sampler_config 接口）。"
                )

        # ── 可选：Optuna 调参 ────────────────────────────────────────
        tune_cfg = self.cfg.get("tuning", {})
        should_tune = (
            tune_cfg.get("enable", False)
            and not self.no_tune
            and name in tune_cfg.get("models_to_tune", [name])
        )
        if should_tune:
            model = self._step_tune(model, X_train, y, cv, groups, primary_metric, tune_cfg)

        # ── CrossValidatorTrainer ────────────────────────────────────
        t0 = time.perf_counter()
        trainer = CrossValidatorTrainer(
            model   = model,
            cv      = cv,
            metrics = all_metrics,
            task    = task,
            groups  = groups,
            save_dir = self.exp_dir / "models" / name,
            verbose  = True,
        )
        result  = trainer.fit(X_train, y, X_test=X_test)
        elapsed = time.perf_counter() - t0

        # ── 记录到 report ────────────────────────────────────────────
        self.report["models"][name] = {
            "cv_scores":  result.cv_scores,
            "oof_scores": result.oof_score,
            "n_folds":    result.n_folds,
            "timing_s":   round(elapsed, 1),
        }
        self.report["timing"][f"train_{name}"] = round(elapsed, 1)

        self.logger.info(
            f"[{name}] 完成 | "
            + " | ".join(
                f"OOF {k}={v:.6f}"
                for k, v in result.oof_score.items()
            )
            + f" | 耗时 {elapsed:.1f}s"
        )
        return name, result

    # ------------------------------------------------------------------
    # 可选：Optuna 调参
    # ------------------------------------------------------------------

    def _step_tune(self, model, X_train, y, cv, groups, primary_metric, tune_cfg) -> BaseModel:
        self.logger.info(f"  → 开始 Optuna 调参 | n_trials={tune_cfg['n_trials']}")
        t0 = time.perf_counter()

        storage = tune_cfg.get("storage")
        if storage and "{exp_dir}" in storage:
            storage = storage.replace("{exp_dir}", str(self.exp_dir))

        tuner = OptunaTuner(
            model   = model,
            cv      = cv,
            metric  = primary_metric,
            config  = TunerConfig(
                n_trials         = tune_cfg.get("n_trials", 50),
                timeout          = tune_cfg.get("timeout"),
                study_name       = f"{model.name}_{self.cfg['experiment']['name']}",
                storage          = storage,
                sampler          = tune_cfg.get("sampler", "tpe"),
                pruner           = tune_cfg.get("pruner", "median"),
                task             = self.cfg["experiment"]["task"],
                groups           = groups,
                fit_best_at_end  = False,
            ),
        )
        tuner.optimize(X_train, y)
        tuner.apply_best_params()   # 原地注入最优参数

        elapsed = time.perf_counter() - t0
        self.report["timing"][f"tune_{model.name}"] = round(elapsed, 1)
        self.logger.info(
            f"  调参完成 | best={tuner.result.best_value:.6f} | 耗时 {elapsed:.1f}s"
        )
        return model   # 参数已更新

    # ------------------------------------------------------------------
    # Step 6: 集成
    # ------------------------------------------------------------------

    def _step_ensemble(
        self, cv_results, model_names, y, X_train, X_test, primary_metric
    ) -> Optional[np.ndarray]:
        ens_cfg = self.cfg.get("ensemble", {})
        if not ens_cfg.get("enable", True) or len(cv_results) < 2:
            # 只有一个模型时，直接返回其测试预测
            if cv_results:
                single = list(cv_results.values())[0]
                return single.test_predictions
            return None

        self._banner("Step 6 / 7  Stacking 集成")
        t0 = time.perf_counter()

        results_list = [cv_results[n] for n in model_names]

        # ── 解析 Hill Climbing 专属配置（可选）─────────────────────
        hc_dict = ens_cfg.get("hill_climbing")
        hc_cfg  = HillClimbingConfig.from_dict(hc_dict) if hc_dict else None

        stacker = StackingEnsemble(
            results           = results_list,
            model_names       = model_names,
            config            = StackingConfig(
                task                 = ens_cfg.get("task", self.cfg["experiment"]["task"]),
                method               = ens_cfg.get("method", "stacking"),
                scale_meta_features  = ens_cfg.get("scale_meta_features", True),
                passthrough          = ens_cfg.get("passthrough", False),
                clip_oof             = ens_cfg.get("clip_oof", False),
                hill_climbing        = hc_cfg,
            ),
            X_train_original  = X_train if ens_cfg.get("passthrough") else None,
            X_test_original   = X_test  if ens_cfg.get("passthrough") else None,
            verbose           = True,
        )

        # 多样性诊断
        self.logger.info("\n" + stacker.diversity_report())

        sr = stacker.fit(y)

        # OOF 得分对比
        score_df = stacker.oof_score_report(y, primary_metric.fn, primary_metric.name)
        self.logger.info("\n" + score_df.to_string())

        elapsed = time.perf_counter() - t0
        oof_score = primary_metric.fn(y.values, sr.meta_oof)
        self.report["ensemble"] = {
            "oof_score":    {primary_metric.name: oof_score},
            "model_weights": sr.model_weights.to_dict(orient="records"),
            "timing_s":     round(elapsed, 1),
        }
        self.report["timing"]["stacking"] = round(elapsed, 1)
        self.logger.info(
            f"集成完成 | OOF {primary_metric.name}={oof_score:.6f} | 耗时 {elapsed:.1f}s"
        )

        # 保存 meta OOF
        np.save(str(self.exp_dir / "oof" / "meta_oof.npy"), sr.meta_oof)

        return sr.meta_test

    # ------------------------------------------------------------------
    # 持久化：模型 artifacts
    # ------------------------------------------------------------------

    def _save_model_artifacts(self, name: str, result: CVResult) -> None:
        output_cfg = self.cfg.get("output", {})

        # OOF 预测
        if output_cfg.get("save_oof", True):
            _save_oof(result.oof_predictions, name, self.exp_dir)

        # 特征重要性
        if output_cfg.get("save_feature_importance", True) and len(result.feature_importance) > 0:
            _save_feature_importance(result.feature_importance, name, self.exp_dir)

        # 单模型提交（备用）
        if result.test_predictions is not None:
            output_cfg2 = self.cfg.get("output", {})
            _save_submission(
                pred     = result.test_predictions,
                test_ids = None,
                id_col   = None,
                pred_col = output_cfg2.get("submission_col", "target"),
                exp_dir  = self.exp_dir,
                suffix   = name,
            )

    # ------------------------------------------------------------------
    # 指标构建
    # ------------------------------------------------------------------

    def _build_metrics(self) -> tuple[MetricConfig, list[MetricConfig]]:
        metrics_cfg = self.cfg.get("metrics", {})
        primary = _resolve_metric(metrics_cfg["primary"])
        primary = MetricConfig(
            name=primary.name, fn=primary.fn,
            higher_is_better=primary.higher_is_better,
            primary=True,
            use_proba=primary.use_proba,
        )
        secondary_list = [
            MetricConfig(
                name             = m["name"],
                fn               = _METRIC_REGISTRY[m["func"]],
                higher_is_better = m.get("higher_is_better", True),
                primary          = False,
                use_proba        = m.get("use_proba", False),
            )
            for m in metrics_cfg.get("secondary", [])
        ]
        return primary, [primary] + secondary_list

    # ------------------------------------------------------------------
    # 最终报告打印
    # ------------------------------------------------------------------

    def _print_final_report(self) -> None:
        self._banner("实验完成 — 最终报告")
        for model_name, info in self.report["models"].items():
            for metric, score in info["oof_scores"].items():
                self.logger.info(
                    f"  [{model_name}]  OOF {metric} = {score:.6f}  "
                    f"(耗时 {info['timing_s']}s)"
                )
        ens = self.report.get("ensemble", {})
        if ens.get("oof_score"):
            for metric, score in ens["oof_score"].items():
                self.logger.info(
                    f"  [★ Stacking] OOF {metric} = {score:.6f}"
                )
        self.logger.info(
            f"  总耗时: {self.report['timing'].get('total_seconds', '?')}s"
        )

    def _banner(self, text: str) -> None:
        line = "─" * 62
        self.logger.info(f"\n{line}\n  {text}\n{line}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="军火库全链路训练流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python scripts/train_pipeline.py --config configs/template.yaml
  python scripts/train_pipeline.py --config configs/my_comp.yaml --no-tune
  python scripts/train_pipeline.py --config configs/my_comp.yaml --fold-only 0 1 2
        """,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        default=False,
        help="强制跳过 Optuna 调参（即使 config 中 enable=true）",
    )
    parser.add_argument(
        "--fold-only",
        nargs="+",
        type=int,
        default=None,
        metavar="FOLD_IDX",
        help="只训练指定折（快速调试用），如 --fold-only 0 1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="只解析配置并打印摘要，不执行实际训练",
    )
    return parser.parse_args()


def _dry_run(config_path: str) -> None:
    """不训练，只打印配置解析摘要。"""
    cfg = load_config(config_path)
    print("\n配置文件解析成功：")
    print(f"  实验名称   : {cfg['experiment']['name']}")
    print(f"  任务类型   : {cfg['experiment']['task']}")
    print(f"  训练数据   : {cfg['data']['train_path']}")
    print(f"  CV 策略    : {cfg['cv']['strategy']} × {cfg['cv']['n_splits']} 折")

    enabled_models = [m for m in cfg.get("models", []) if m.get("enabled", True)]
    print(f"  启用模型   : {[m['name'] for m in enabled_models]}")

    feat_cfg = cfg.get("features", {})
    tfs = [t['type'] for t in feat_cfg.get("transformers", []) if t.get("enabled", True)]
    print(f"  特征变换器 : {tfs}")

    fs_cfg = cfg.get("feature_selection", {})
    print(f"  特征筛选   : {'启用' if fs_cfg.get('enable') else '禁用'}"
          + (f" (iterations={fs_cfg.get('n_iterations',80)}, "
             f"pct={fs_cfg.get('percentile_threshold',75.0)})"
             if fs_cfg.get("enable") else ""))

    tune_cfg = cfg.get("tuning", {})
    print(f"  Optuna 调参: {'启用' if tune_cfg.get('enable') else '禁用'}")
    print(f"  Stacking   : {'启用' if cfg.get('ensemble', {}).get('enable', True) else '禁用'}")

    pp_cfg = cfg.get("post_process", {})
    print(f"  后处理     : {'启用' if pp_cfg.get('enable') else '禁用'}"
          + (f" (校准={pp_cfg.get('calibration',{}).get('method','isotonic')}, "
             f"阈值优化={pp_cfg.get('threshold',{}).get('enable',True)})"
             if pp_cfg.get("enable") else ""))

    print(f"\n[dry-run] 配置合法，可以正式运行。")


def main() -> None:
    args = parse_args()

    if args.dry_run:
        _dry_run(args.config)
        return

    pipeline = TrainPipeline(
        config_path = args.config,
        no_tune     = args.no_tune,
        fold_only   = args.fold_only,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
