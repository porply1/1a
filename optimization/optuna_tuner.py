"""
optimization/optuna_tuner.py
-----------------------------
Optuna 超参数自动搜索引擎，与军火库 core 层无缝集成。

架构原理
--------
                    ┌──────────────────────────────────────┐
  study.optimize()  │           Objective Function         │
       ↓            │  ① trial.suggest → suggest_params()  │
  trial #0          │  ② 克隆 BaseModel，注入新参数        │
  trial #1   ──────▶│  ③ 逐折训练（BaseCVSplitter.split）  │
   ...              │  ④ 每折后 report 中间值 → 剪枝判断   │
  trial #N          │  ⑤ 返回全局 OOF 得分               │
                    └──────────────────────────────────────┘
                                   │ best_params
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  CrossValidatorTrainer（最终验证）    │
                    │  用最优参数跑完整 CV，生成 OOF 预测   │
                    └──────────────────────────────────────┘

关键设计决策
-----------
  1. 目标函数内部使用轻量折循环（非完整 CrossValidatorTrainer），
     原因：剪枝需要在折间插入 trial.report / should_prune，
     CrossValidatorTrainer 是黑盒无法中断。
  2. 搜索空间来自各模型的 suggest_params(trial) 静态方法，
     与模型本身耦合，不需要在调参层维护重复的搜索空间字典。
  3. 最优参数确认后，调用 CrossValidatorTrainer 做一次完整评估，
     产出可用于 Stacking 的 OOF/test 预测。
  4. study 持久化到 SQLite/PostgreSQL，支持断点续传和多进程并行。
  5. 方向（maximize/minimize）由 MetricConfig.higher_is_better 自动推断。

使用示例
--------
>>> from optimization.optuna_tuner import OptunaTuner, TunerConfig
>>> from data.splitter import get_cv, CVConfig
>>> from core.base_trainer import MetricConfig
>>> from models.gbm import LGBMModel
>>> from sklearn.metrics import roc_auc_score

>>> model = LGBMModel(task="binary")
>>> cv    = get_cv(CVConfig(strategy="stratified_kfold", n_splits=5))
>>> metric = MetricConfig("auc", roc_auc_score, higher_is_better=True, use_proba=True)

>>> tuner = OptunaTuner(
...     model   = model,
...     cv      = cv,
...     metric  = metric,
...     config  = TunerConfig(n_trials=100, study_name="lgbm_binary_v1"),
... )
>>> best_result = tuner.optimize(X_train, y_train, X_test=X_test)
>>> print(best_result.best_params)
>>> print(best_result.best_cv_result.oof_score)
"""

from __future__ import annotations

import gc
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.pruners  import MedianPruner, NopPruner, HyperbandPruner
    from optuna.samplers import TPESampler, CmaEsSampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)   # 默认静默 optuna 内部日志
except ImportError as e:
    raise ImportError("Optuna 未安装，请执行：pip install optuna") from e

from core.base_model   import BaseModel
from core.base_trainer import CrossValidatorTrainer, MetricConfig, CVResult
from data.splitter     import BaseCVSplitter


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class TunerConfig:
    """
    OptunaTuner 的完整行为配置。

    Parameters
    ----------
    n_trials : int
        最大搜索次数（trials）。默认 100。
    timeout : float | None
        最长搜索秒数（优先级高于 n_trials）。None 表示不限时。
    study_name : str
        Study 名称，用于持久化和恢复。
    storage : str | None
        持久化存储 URI（SQLite: "sqlite:///tuning.db"，
        PostgreSQL: "postgresql://..."）。
        None 时使用内存 study（不可恢复）。
    sampler : str
        搜索采样器：'tpe'（默认，贝叶斯）| 'cmaes'（连续空间更优）| 'random'
    pruner : str
        剪枝器：'median'（默认）| 'hyperband' | 'none'
    pruner_startup_trials : int
        MedianPruner 在前 N 次 trial 不剪枝（积累基准值）。默认 10。
    pruner_warmup_steps : int
        每个 trial 前 N 折不报告中间值（warm-up）。默认 1。
    n_jobs : int
        并行 trial 数（需要 storage 支持并发写入）。默认 1。
    show_progress_bar : bool
        是否显示 tqdm 进度条。默认 True。
    fit_best_at_end : bool
        True 时，搜索结束后用最优参数跑完整 CrossValidatorTrainer，
        生成可用于 Stacking 的 OOF/test 预测。默认 True。
    fix_params : dict
        在 suggest_params 返回的搜索空间之外，
        额外固定不参与搜索的超参数（如 num_leaves=63）。
    task : str
        任务类型（透传给 CrossValidatorTrainer）。
    groups : pd.Series | None
        Group 标识列（GroupKFold/时序切分时使用）。
    """
    n_trials:             int             = 100
    timeout:              Optional[float] = None
    study_name:           str             = "optuna_study"
    storage:              Optional[str]   = None
    sampler:              str             = "tpe"
    pruner:               str             = "median"
    pruner_startup_trials: int            = 10
    pruner_warmup_steps:  int             = 1
    n_jobs:               int             = 1
    show_progress_bar:    bool            = True
    fit_best_at_end:      bool            = True
    fix_params:           dict            = field(default_factory=dict)
    task:                 str             = "regression"
    groups:               Optional[pd.Series] = None


# ---------------------------------------------------------------------------
# 搜索结果
# ---------------------------------------------------------------------------

@dataclass
class TunerResult:
    """
    OptunaTuner.optimize() 的完整输出。

    Attributes
    ----------
    best_params : dict
        Optuna 找到的最优超参数字典（已与 fix_params 合并）。
    best_value : float
        最优 trial 的目标函数值。
    best_trial_number : int
        最优 trial 的编号。
    study : optuna.Study
        完整 study 对象，可用于后续分析和可视化。
    best_cv_result : CVResult | None
        用最优参数跑完整 CV 的结果（fit_best_at_end=True 时才有）。
        包含 OOF 预测，可直接用于 StackingEnsemble。
    n_completed_trials : int
        实际完成的 trial 数（未被剪枝中断的）。
    n_pruned_trials : int
        被剪枝提前终止的 trial 数。
    total_time : float
        搜索总耗时（秒）。
    param_importance : dict | None
        各超参数对目标函数的重要性（需要 optuna.importance）。
    """
    best_params:         dict
    best_value:          float
    best_trial_number:   int
    study:               Any                   # optuna.Study
    best_cv_result:      Optional[CVResult]
    n_completed_trials:  int
    n_pruned_trials:     int
    total_time:          float
    param_importance:    Optional[dict] = None

    def summary(self) -> str:
        lines = [
            "=" * 62,
            "  Optuna 搜索结果摘要",
            "=" * 62,
            f"  最优 Trial      : #{self.best_trial_number}",
            f"  最优目标值      : {self.best_value:.6f}",
            f"  完成 Trials     : {self.n_completed_trials}",
            f"  剪枝 Trials     : {self.n_pruned_trials}",
            f"  总耗时          : {self.total_time:.1f}s",
            f"  {'─' * 56}",
            "  最优超参数：",
        ]
        for k, v in sorted(self.best_params.items()):
            lines.append(f"    {k:<30} = {v}")
        if self.best_cv_result:
            lines.append(f"  {'─' * 56}")
            lines.append("  完整 CV 结果（最优参数）：")
            for name, score in self.best_cv_result.oof_score.items():
                lines.append(f"    OOF {name:<28} = {score:.6f}")
        lines.append("=" * 62)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# OptunaTuner 主类
# ---------------------------------------------------------------------------

class OptunaTuner:
    """
    基于 Optuna 的自动超参数搜索器。

    搜索流程
    --------
    optimize(X, y) 被调用时：
      1. 创建（或加载）Optuna Study
      2. 对每个 trial：
         a. 调用 model_cls.suggest_params(trial) 生成候选参数
         b. 逐折训练，每折后 report 中间分数 → 触发剪枝判断
         c. 返回全局 OOF 均值作为目标值
      3. n_trials 结束后提取最优参数
      4. 可选：用最优参数跑完整 CrossValidatorTrainer（生成 OOF/test 预测）
    """

    def __init__(
        self,
        model:    BaseModel,
        cv:       BaseCVSplitter,
        metric:   MetricConfig,
        config:   Optional[TunerConfig] = None,
    ):
        self.model   = model
        self.cv      = cv
        self.metric  = metric
        self.config  = config or TunerConfig()
        self._logger = _build_logger(f"tuner.{model.name}")

        # fit 后填充
        self._study:  Optional[Any] = None
        self._result: Optional[TunerResult] = None

        # 校验 suggest_params 是否存在
        model_cls = type(model)
        if not hasattr(model_cls, "suggest_params"):
            raise AttributeError(
                f"{model_cls.__name__} 没有 suggest_params(trial) 静态方法。\n"
                f"请在模型类中实现：\n"
                f"  @staticmethod\n"
                f"  def suggest_params(trial) -> dict: ..."
            )

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def optimize(
        self,
        X:      pd.DataFrame,
        y:      pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        fit_kwargs: Optional[dict] = None,
    ) -> TunerResult:
        """
        启动超参数搜索。

        Parameters
        ----------
        X, y        : 训练集特征与标签
        X_test      : 测试集（仅在 fit_best_at_end=True 时使用）
        fit_kwargs  : 传递给最终完整 CV 训练的额外参数

        Returns
        -------
        TunerResult
        """
        fit_kwargs = fit_kwargs or {}
        cfg        = self.config
        t_start    = time.perf_counter()

        self._log(f"{'=' * 60}")
        self._log(
            f"  Optuna 搜索启动 | 模型={self.model.name} | "
            f"trials={cfg.n_trials} | sampler={cfg.sampler} | "
            f"pruner={cfg.pruner}"
        )
        self._log(
            f"  数据形状 X={X.shape} | 指标={self.metric.name} | "
            f"方向={'maximize' if self.metric.higher_is_better else 'minimize'}"
        )
        self._log(f"{'=' * 60}")

        # ---- 1. 构建 study ----
        study = self._create_or_load_study()
        self._study = study

        # ---- 2. 构建目标函数（闭包捕获 X, y）----
        objective = self._build_objective(X, y)

        # ---- 3. 搜索 ----
        study.optimize(
            objective,
            n_trials          = cfg.n_trials,
            timeout           = cfg.timeout,
            n_jobs            = cfg.n_jobs,
            show_progress_bar = cfg.show_progress_bar,
            callbacks         = [self._trial_callback],
        )

        # ---- 4. 提取最优参数 ----
        best_trial  = study.best_trial
        best_params = {**best_trial.params, **cfg.fix_params}

        # ---- 5. 统计 trial 状态 ----
        from optuna.trial import TrialState
        n_completed = len([t for t in study.trials
                           if t.state == TrialState.COMPLETE])
        n_pruned    = len([t for t in study.trials
                           if t.state == TrialState.PRUNED])

        # ---- 6. 参数重要性 ----
        param_importance = self._compute_param_importance(study)

        # ---- 7. 可选：最优参数完整 CV ----
        best_cv_result: Optional[CVResult] = None
        if cfg.fit_best_at_end:
            best_cv_result = self._fit_best(
                best_params, X, y, X_test, fit_kwargs
            )

        total_time = time.perf_counter() - t_start

        result = TunerResult(
            best_params        = best_params,
            best_value         = best_trial.value,
            best_trial_number  = best_trial.number,
            study              = study,
            best_cv_result     = best_cv_result,
            n_completed_trials = n_completed,
            n_pruned_trials    = n_pruned,
            total_time         = total_time,
            param_importance   = param_importance,
        )
        self._result = result

        self._log("\n" + result.summary())
        if param_importance:
            self._log(self._format_param_importance(param_importance))

        return result

    # ------------------------------------------------------------------
    # 目标函数工厂
    # ------------------------------------------------------------------

    def _build_objective(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Callable:
        """
        构建 Optuna 目标函数（闭包）。

        内部使用逐折轻量训练循环（而非完整 CrossValidatorTrainer），
        原因：需要在每折结束后调用 trial.report / trial.should_prune。
        """
        cfg         = self.config
        model_cls   = type(self.model)
        cv          = self.cv
        metric      = self.metric
        groups      = cfg.groups
        warmup      = cfg.pruner_warmup_steps

        def objective(trial: optuna.Trial) -> float:
            # ---- 参数建议 ----
            suggested   = model_cls.suggest_params(trial)
            trial_params = {**suggested, **cfg.fix_params}

            # ---- 切分 ----
            folds   = cv.split(X, y, groups=groups)
            n_folds = len(folds)
            scores: list[float] = []

            for fold_i, (tr_idx, val_idx) in enumerate(folds):
                X_tr,  y_tr  = X.iloc[tr_idx],  y.iloc[tr_idx]
                X_val, y_val = X.iloc[val_idx],  y.iloc[val_idx]

                # ---- 克隆模型并注入新参数 ----
                fold_model = self.model.clone()
                fold_model.set_params(**trial_params)

                # ---- 训练（静默模式）----
                _silence_model(fold_model)
                try:
                    fold_model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
                except Exception as e:
                    # 参数组合可能导致训练失败，标记为 nan 让 Optuna 剪枝
                    self._log(
                        f"Trial #{trial.number} Fold {fold_i} 训练失败：{e}",
                        level=logging.WARNING,
                    )
                    raise optuna.exceptions.TrialPruned()

                # ---- 评估 ----
                if metric.use_proba:
                    pred = fold_model.predict_proba(X_val)
                else:
                    pred = fold_model.predict(X_val)

                score = metric.fn(y_val.values, pred)
                scores.append(score)

                # ---- 报告中间值（触发剪枝器）----
                if fold_i >= warmup:
                    # Optuna 要求 intermediate 值方向与 direction 一致
                    report_val = score if metric.higher_is_better else -score
                    trial.report(report_val, step=fold_i)

                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                # 显式释放内存
                del X_tr, y_tr, X_val, y_val, fold_model
                gc.collect()

            if not scores:
                raise optuna.exceptions.TrialPruned()

            mean_score = float(np.mean(scores))
            std_score  = float(np.std(scores))

            # 将折标准差存入 user_attrs，方便后续分析稳定性
            trial.set_user_attr("cv_mean", mean_score)
            trial.set_user_attr("cv_std",  std_score)
            trial.set_user_attr("n_folds", len(scores))

            return mean_score

        return objective

    # ------------------------------------------------------------------
    # Study 构建
    # ------------------------------------------------------------------

    def _create_or_load_study(self) -> optuna.Study:
        """
        创建新 study 或从 storage 加载已有 study（断点续传）。
        """
        cfg       = self.config
        direction = "maximize" if self.metric.higher_is_better else "minimize"

        sampler = self._build_sampler(cfg.sampler)
        pruner  = self._build_pruner(cfg.pruner, cfg.pruner_startup_trials)

        if cfg.storage:
            self._log(f"  持久化存储：{cfg.storage}")
            self._log(f"  Study 名称  ：{cfg.study_name}")

        study = optuna.create_study(
            study_name        = cfg.study_name,
            storage           = cfg.storage,
            direction         = direction,
            sampler           = sampler,
            pruner            = pruner,
            load_if_exists    = True,    # 关键：已存在则加载（断点续传）
        )

        n_existing = len(study.trials)
        if n_existing > 0:
            self._log(
                f"  ✓ 断点续传：已加载 {n_existing} 个历史 trials，"
                f"将继续搜索至 {cfg.n_trials} 个 trials。"
            )

        return study

    @staticmethod
    def _build_sampler(name: str):
        samplers = {
            "tpe":    TPESampler(seed=42, multivariate=True),
            "cmaes":  CmaEsSampler(seed=42),
            "random": optuna.samplers.RandomSampler(seed=42),
        }
        if name not in samplers:
            raise ValueError(f"未知 sampler='{name}'，可选：{list(samplers.keys())}")
        return samplers[name]

    def _build_pruner(self, name: str, startup_trials: int):
        if name == "median":
            return MedianPruner(
                n_startup_trials  = startup_trials,
                n_warmup_steps    = self.config.pruner_warmup_steps,
                interval_steps    = 1,
            )
        elif name == "hyperband":
            return HyperbandPruner(
                min_resource     = 1,
                max_resource     = self.cv._kf.n_splits if hasattr(self.cv, "_kf") else 5,
                reduction_factor = 3,
            )
        elif name == "none":
            return NopPruner()
        else:
            raise ValueError(f"未知 pruner='{name}'，可选：median | hyperband | none")

    # ------------------------------------------------------------------
    # 最优参数完整 CV
    # ------------------------------------------------------------------

    def _fit_best(
        self,
        best_params: dict,
        X:           pd.DataFrame,
        y:           pd.Series,
        X_test:      Optional[pd.DataFrame],
        fit_kwargs:  dict,
    ) -> CVResult:
        """
        用最优参数克隆模型，调用完整 CrossValidatorTrainer。
        生成 OOF/test 预测，可直接喂入 StackingEnsemble。
        """
        self._log("─" * 60)
        self._log("  用最优参数跑完整 CrossValidatorTrainer...")

        best_model = self.model.clone()
        best_model.set_params(**best_params)
        best_model.name = f"{self.model.name}_best"

        trainer = CrossValidatorTrainer(
            model   = best_model,
            cv      = self.cv,
            metrics = [self.metric],
            task    = self.config.task,
            groups  = self.config.groups,
            verbose = True,
        )
        return trainer.fit(X, y, X_test=X_test, fit_kwargs=fit_kwargs)

    # ------------------------------------------------------------------
    # 参数重要性
    # ------------------------------------------------------------------

    def _compute_param_importance(
        self, study: optuna.Study
    ) -> Optional[dict]:
        """
        使用 optuna.importance 计算各超参数对目标函数的重要性。
        需要足够多的 completed trials（至少 2 个）。
        """
        from optuna.trial import TrialState
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(completed) < 2:
            return None
        try:
            importance = optuna.importance.get_param_importances(study)
            return dict(importance)
        except Exception as e:
            warnings.warn(f"参数重要性计算失败：{e}", UserWarning)
            return None

    @staticmethod
    def _format_param_importance(importance: dict) -> str:
        lines = [
            "─" * 55,
            "  超参数重要性（对目标函数的贡献度）",
            "─" * 55,
        ]
        max_val = max(importance.values()) if importance else 1.0
        for param, val in sorted(importance.items(), key=lambda x: -x[1]):
            bar_len = int(val / max_val * 28)
            bar     = "█" * bar_len
            lines.append(f"  {param:<28} {val:.4f}  {bar}")
        lines.append("─" * 55)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 可视化建议
    # ------------------------------------------------------------------

    def plot_optimization_history(self) -> None:
        """
        绘制搜索历史（目标值随 trial 变化趋势）。
        需要安装 optuna[visualization] 和 plotly。
        """
        self._check_study()
        try:
            from optuna.visualization import plot_optimization_history
            fig = plot_optimization_history(self._study)
            fig.show()
        except ImportError:
            self._log(
                "可视化需要安装 plotly：pip install plotly",
                level=logging.WARNING,
            )

    def plot_param_importances(self) -> None:
        """绘制超参数重要性柱状图。"""
        self._check_study()
        try:
            from optuna.visualization import plot_param_importances
            fig = plot_param_importances(self._study)
            fig.show()
        except ImportError:
            self._log(
                "可视化需要安装 plotly：pip install plotly",
                level=logging.WARNING,
            )

    def plot_contour(self, params: Optional[list[str]] = None) -> None:
        """绘制参数对之间的等高线图（探索参数交互）。"""
        self._check_study()
        try:
            from optuna.visualization import plot_contour
            fig = plot_contour(self._study, params=params)
            fig.show()
        except ImportError:
            self._log(
                "可视化需要安装 plotly：pip install plotly",
                level=logging.WARNING,
            )

    def plot_parallel_coordinate(self) -> None:
        """绘制平行坐标图（总览所有 trial 的参数分布）。"""
        self._check_study()
        try:
            from optuna.visualization import plot_parallel_coordinate
            fig = plot_parallel_coordinate(self._study)
            fig.show()
        except ImportError:
            self._log(
                "可视化需要安装 plotly：pip install plotly",
                level=logging.WARNING,
            )

    # ------------------------------------------------------------------
    # Trials 分析
    # ------------------------------------------------------------------

    def trials_dataframe(self) -> pd.DataFrame:
        """
        返回所有 trials 的参数与得分 DataFrame，便于 Notebook 分析。

        Returns
        -------
        pd.DataFrame  index=trial_number, 含 value / params / user_attrs
        """
        self._check_study()
        df = self._study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
        return df.sort_values("value", ascending=not self.metric.higher_is_better)

    def top_k_params(self, k: int = 5) -> list[dict]:
        """
        返回得分最高的 k 组参数（含 cv_std），
        可用于构建多种子集成。
        """
        self._check_study()
        from optuna.trial import TrialState
        completed = [
            t for t in self._study.trials
            if t.state == TrialState.COMPLETE
        ]
        if not completed:
            return []

        reverse = self.metric.higher_is_better
        sorted_trials = sorted(
            completed,
            key=lambda t: t.value,
            reverse=reverse,
        )[:k]

        results = []
        for t in sorted_trials:
            entry = {
                "trial":    t.number,
                "value":    t.value,
                "cv_mean":  t.user_attrs.get("cv_mean", t.value),
                "cv_std":   t.user_attrs.get("cv_std", float("nan")),
                "params":   {**t.params, **self.config.fix_params},
            }
            results.append(entry)
        return results

    def apply_best_params(self) -> BaseModel:
        """
        将最优参数注入到原始 model 实例并返回（原地修改）。

        Returns
        -------
        self.model（已更新参数）
        """
        self._check_result()
        self.model.set_params(**self._result.best_params)
        self._log(
            f"最优参数已注入 {self.model.name}，"
            f"param_hash={self.model.param_hash()}"
        )
        return self.model

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------

    def _trial_callback(
        self,
        study:  optuna.Study,
        trial:  optuna.trial.FrozenTrial,
    ) -> None:
        """每个 trial 结束后打印简要进度。"""
        from optuna.trial import TrialState
        n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        n_pruned   = len([t for t in study.trials if t.state == TrialState.PRUNED])

        if trial.state == TrialState.COMPLETE:
            is_best = trial.number == study.best_trial.number
            marker  = " ★ NEW BEST" if is_best else ""
            cv_std  = trial.user_attrs.get("cv_std", float("nan"))
            self._log(
                f"  Trial #{trial.number:>4} | "
                f"{self.metric.name}={trial.value:.6f} ± {cv_std:.6f}{marker} | "
                f"best={study.best_value:.6f} | "
                f"done={n_complete} pruned={n_pruned}"
            )
        elif trial.state == TrialState.PRUNED:
            self._log(
                f"  Trial #{trial.number:>4} | [PRUNED] | "
                f"done={n_complete} pruned={n_pruned}",
                level=logging.DEBUG,
            )

    # ------------------------------------------------------------------
    # 属性访问
    # ------------------------------------------------------------------

    @property
    def result(self) -> TunerResult:
        self._check_result()
        return self._result

    @property
    def best_params(self) -> dict:
        self._check_result()
        return self._result.best_params

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _check_study(self) -> None:
        if self._study is None:
            raise RuntimeError("请先调用 optimize() 后再访问 study。")

    def _check_result(self) -> None:
        if self._result is None:
            raise RuntimeError("请先调用 optimize() 后再访问结果。")

    def _log(self, msg: str, level: int = logging.INFO) -> None:
        self._logger.log(level, msg)

    def __repr__(self) -> str:
        n_trials = len(self._study.trials) if self._study else 0
        return (
            f"OptunaTuner("
            f"model={self.model.name!r}, "
            f"metric={self.metric.name!r}, "
            f"trials_done={n_trials})"
        )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _silence_model(model: BaseModel) -> None:
    """
    在调参搜索时静默模型的训练日志，避免刷屏。
    通过修改 params 中的 verbose/verbosity 键实现。
    """
    silence_map = {
        # LGBM
        "verbose":     -1,
        # XGB
        "verbosity":   0,
        # CatBoost
        "allow_writing_files": False,
    }
    for k, v in silence_map.items():
        if k in model.params:
            model.params[k] = v

    # 静默 log_period（CatBoost / LGBMModel 的实例属性）
    if hasattr(model, "log_period"):
        model.log_period = 0
