"""
ensemble/stacking.py
--------------------
集成引擎：提供两种集成策略，统一由 StackingEnsemble 门面类调度。

策略 A — Stacking（默认）
  传统两层 Stacking：base model OOF → 元特征矩阵 → Ridge / LogReg 元学习器。
  优点：可解释，权重即系数；缺点：元学习器本身也需要调参。

策略 B — Hill Climbing（Caruana Ensemble Selection）
  Caruana et al., ICML 2004, "Ensemble Selection from Libraries of Models"
  贪心迭代算法，每轮将使 OOF 指标提升最大的模型加入集成池。
  同一模型可被多次选入，最终权重 = 选中次数 / 总轮数。

  关键优势（相比 Ridge Stacking）：
    1. 无负权重：所有 base model 贡献 ≥ 0（不会出现 Ridge 的反向系数）
    2. 天然稀疏：只保留真正有贡献的模型，忽略噪音模型
    3. 非参数：无元学习器超参，搜索空间为整数（轮数 T）
    4. 允许重复采样：间接实现 Boosting-like 加权
    5. 随机扰动：random_shuffle_pct 防止陷入局部最优

  算法伪码
  --------
    E = []         # 选中模型列表（允许重复），初始为空
    S_best = None  # 当前最优集成分数

    for t in 1..T:
      if random() < ε:               ← ε = random_shuffle_pct（噪声防过拟合）
        best = random_choice(pool)
      else:
        best = argmax_{m ∈ pool} metric(y, mean(oof[i] for i in E + [m]))
      E.append(best)
      S_best = metric(y, mean(oof[i] for i in E))

    weight[k] = count(k in E) / T   ← 归一化权重

  Bagging 变体（n_bags > 1）
  --------------------------
    在 N 个 bootstrap 样本上独立运行上述算法，权重取平均。
    减少对特定 OOF 分布的过拟合，等价于给算法加了 Bootstrap 集成。

接口稳定性保证
--------------
  - StackingEnsemble.fit() 返回 StackingResult（与旧版完全相同）
  - train_pipeline.py 无需修改，通过 YAML 的 ensemble.method 切换策略
  - StackingResult.weight_report() / diversity_report() 对两种策略均适用

YAML 切换示例
-------------
  ensemble:
    enable: true
    method: "hill_climbing"          # 或 "stacking"（默认）
    task: "binary"
    hill_climbing:
      n_iterations:        100
      higher_is_better:    true
      random_shuffle_pct:  0.05
      n_bags:              5
      seed:                42
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler

from core.base_trainer import CVResult


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

def _build_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Hill Climbing 专属配置
# ---------------------------------------------------------------------------

@dataclass
class HillClimbingConfig:
    """
    Caruana Ensemble Selection 超参数。

    Parameters
    ----------
    n_iterations : int
        贪心迭代总轮数 T（每轮选一个模型加入集成）。
        建议：50 ~ 500。轮数越多权重分布越稳定，但边际收益递减。

    metric_fn : Callable | None
        评估函数 fn(y_true, y_pred) -> float。
        None 时由 StackingConfig.task 自动选择：
          binary      → roc_auc_score
          regression  → -mean_squared_error（取负使得越大越好）
          multiclass  → accuracy_score

    higher_is_better : bool
        True  → 最大化 metric（AUC、Accuracy）
        False → 最小化 metric（MSE、RMSE）

    random_shuffle_pct : float
        每轮以此概率随机选择模型（而非贪心选择），防止陷入局部最优。
        典型值：0.0（纯贪心）~ 0.1（10% 随机扰动）。

    n_bags : int
        Bootstrap 袋数。
        > 1 时：在 N 个 bootstrap 样本上独立运行 Hill Climbing，
        权重取均值。减少过拟合，提升稳定性。
        建议：5 ~ 20。1 = 不 Bagging。

    bag_fraction : float
        每个 bag 使用的样本比例（0,1]，默认 1.0（有放回抽样）。

    seed : int
        随机种子（控制 bootstrap 采样 + 随机扰动）。

    warm_start_with_best : bool
        True  → 第 0 步以单模型最优 OOF 得分为起点加入 E（加速收敛）。
        False → E 从空集开始。
    """
    n_iterations:         int   = 100
    metric_fn:            Optional[Callable] = None
    higher_is_better:     bool  = True
    random_shuffle_pct:   float = 0.05
    n_bags:               int   = 1
    bag_fraction:         float = 1.0
    seed:                 int   = 42
    warm_start_with_best: bool  = True

    @classmethod
    def from_dict(cls, d: dict) -> "HillClimbingConfig":
        d = dict(d)  # 不修改调用方的 dict
        # bagging_runs 是 n_bags 的语义别名（YAML 可用任意一个）
        if "bagging_runs" in d:
            if "n_bags" not in d:
                d["n_bags"] = d.pop("bagging_runs")
            else:
                d.pop("bagging_runs")  # n_bags 优先
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> dict:
        import dataclasses
        return {
            k: v for k, v in dataclasses.asdict(self).items()
            if k != "metric_fn"
        }


# ---------------------------------------------------------------------------
# Hill Climbing 追踪记录
# ---------------------------------------------------------------------------

@dataclass
class HillClimbingTrace:
    """
    记录 Hill Climbing 每一轮的搜索历史。

    Attributes
    ----------
    round_idx      : 轮次（0-indexed）
    model_added    : 本轮被选入的模型名称
    score_after    : 加入后的集成 OOF 分数
    delta          : 本轮的分数增量（相对于加入前）
    was_random     : 本轮是否为随机选择（非贪心）
    bag_id         : 来自哪个 bootstrap bag（n_bags > 1 时）
    """
    round_idx:   int
    model_added: str
    score_after: float
    delta:       float
    was_random:  bool
    bag_id:      int = 0

    def __repr__(self) -> str:
        rand_mark = " [rand]" if self.was_random else ""
        sign      = "+" if self.delta >= 0 else ""
        return (
            f"Round {self.round_idx + 1:>3d} | "
            f"Added: {self.model_added:<20} | "
            f"Score: {self.score_after:.6f} | "
            f"Δ: {sign}{self.delta:.6f}{rand_mark}"
        )


# ---------------------------------------------------------------------------
# Stacking 配置（新增 method + hill_climbing 字段）
# ---------------------------------------------------------------------------

@dataclass
class StackingConfig:
    """
    StackingEnsemble 的行为配置。

    新增字段（相对旧版）
    --------------------
    method : str
        集成策略：'stacking'（默认）| 'hill_climbing'
    hill_climbing : HillClimbingConfig | None
        Hill Climbing 专属配置。None 时使用默认值。
        也可通过 YAML 的 ensemble.hill_climbing 节点注入。
    """
    task:                       str   = "regression"
    method:                     str   = "stacking"       # ← 新增
    meta_learner:               Any   = None
    scale_meta_features:        bool  = True
    append_original_features:   Optional[list[str]] = None
    passthrough:                bool  = False
    clip_oof:                   bool  = False
    hill_climbing:              Optional[HillClimbingConfig] = None  # ← 新增


# ---------------------------------------------------------------------------
# 集成结果（与旧版完全兼容，新增 hill_climbing_trace）
# ---------------------------------------------------------------------------

@dataclass
class StackingResult:
    """
    StackingEnsemble.fit() 的完整输出。

    新增字段
    --------
    hill_climbing_trace : list[HillClimbingTrace] | None
        Hill Climbing 策略专属：每轮的选择记录。
        Stacking 策略时为 None。
    """
    meta_oof:              np.ndarray
    meta_test:             Optional[np.ndarray]
    X_meta_train:          pd.DataFrame
    X_meta_test:           Optional[pd.DataFrame]
    model_weights:         pd.DataFrame
    meta_learner:          Any
    fit_time:              float
    hill_climbing_trace:   Optional[list[HillClimbingTrace]] = None

    def weight_report(self) -> str:
        """打印各 base model 贡献权重的格式化报告。"""
        lines = [
            "=" * 58,
            "  Base Model 贡献权重（Method: "
            + ("Hill Climbing" if self.hill_climbing_trace else "Stacking")
            + "）",
            "=" * 58,
        ]
        if self.model_weights.empty:
            lines.append("  （无权重信息）")
        else:
            df     = self.model_weights.sort_values("abs_weight", ascending=False)
            max_aw = df["abs_weight"].max()
            for _, row in df.iterrows():
                bar_len = int(row["abs_weight"] / (max_aw + 1e-12) * 30)
                bar     = "█" * bar_len
                sign    = "+" if row["weight"] >= 0 else "-"
                lines.append(
                    f"  {str(row['model']):<22} "
                    f"{sign}{abs(row['weight']):.6f}  {bar}"
                )
        lines.append("=" * 58)
        return "\n".join(lines)

    def hill_climbing_report(self) -> str:
        """
        Hill Climbing 专属：输出顺序追踪报告（每轮选了哪个模型、增量多少）。
        """
        if not self.hill_climbing_trace:
            return "（当前集成策略不是 Hill Climbing，无追踪记录）"

        lines = [
            "=" * 70,
            "  Hill Climbing 迭代追踪报告（Caruana Ensemble Selection）",
            "=" * 70,
        ]

        # 按 bag 分组
        bags = {}
        for t in self.hill_climbing_trace:
            bags.setdefault(t.bag_id, []).append(t)

        for bag_id, traces in sorted(bags.items()):
            if len(bags) > 1:
                lines.append(f"\n  ── Bag {bag_id + 1} / {len(bags)} ──")
            for t in traces:
                lines.append(f"  {t}")

        # 摘要：最终权重
        lines += ["", "  最终模型选择频次（权重）："]
        wdf = self.model_weights.sort_values("weight", ascending=False)
        for _, row in wdf.iterrows():
            bar = "■" * max(1, int(row["weight"] * 40))
            lines.append(
                f"    {str(row['model']):<22} "
                f"weight={row['weight']:.4f}  {bar}"
            )

        # 收益统计
        all_deltas = [t.delta for t in self.hill_climbing_trace if not t.was_random]
        pos_rounds = sum(1 for d in all_deltas if d > 1e-9)
        zero_rounds = sum(1 for d in all_deltas if abs(d) <= 1e-9)
        neg_rounds  = sum(1 for d in all_deltas if d < -1e-9)
        total_gain  = sum(all_deltas)

        lines += [
            "",
            f"  总轮数   : {len(self.hill_climbing_trace)}",
            f"  有效提升 : {pos_rounds} 轮",
            f"  无变化   : {zero_rounds} 轮",
            f"  负增益   : {neg_rounds} 轮",
            f"  总增益   : {total_gain:+.6f}",
            "=" * 70,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hill Climbing 核心引擎
# ---------------------------------------------------------------------------

class HillClimbingEnsemble:
    """
    Caruana Ensemble Selection 引擎。
    通常不直接使用，而是通过 StackingEnsemble（method='hill_climbing'）调用。
    """

    def __init__(
        self,
        oof_matrix:   np.ndarray,          # (n_samples, n_models)
        test_matrix:  Optional[np.ndarray], # (n_test, n_models) | None
        model_names:  list[str],
        config:       HillClimbingConfig,
        task:         str = "binary",
    ):
        """
        Parameters
        ----------
        oof_matrix   : 每列是一个 base model 的 OOF 预测，shape (N, K)
        test_matrix  : 每列是一个 base model 的测试集预测，shape (M, K) | None
        model_names  : K 个模型的名称
        config       : HillClimbingConfig
        task         : 任务类型，用于自动选择默认 metric
        """
        self.oof_matrix   = oof_matrix.astype(np.float64)
        self.test_matrix  = test_matrix.astype(np.float64) if test_matrix is not None else None
        self.model_names  = model_names
        self.config       = config
        self.task         = task
        self._n_models    = len(model_names)
        self._logger      = _build_logger("hill_climbing")

        # 解析 metric
        self._metric_fn        = self._resolve_metric(config.metric_fn, task)
        self._higher_is_better = config.higher_is_better

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def fit(self, y: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray],
                                          pd.DataFrame, list[HillClimbingTrace]]:
        """
        运行 Hill Climbing 集成选择。

        Returns
        -------
        (oof_pred, test_pred, weight_df, all_traces)
          oof_pred   : (N,) 最终集成 OOF 预测
          test_pred  : (M,) 最终集成测试预测（或 None）
          weight_df  : pd.DataFrame  columns=[model, weight, abs_weight, count]
          all_traces : 所有 bag 的轮次记录
        """
        t0         = time.perf_counter()
        rng        = np.random.RandomState(self.config.seed)
        n_samples  = len(y)
        n_bags     = max(1, self.config.n_bags)
        all_traces: list[HillClimbingTrace] = []

        # 汇总每个 bag 的模型计数
        total_counts = np.zeros(self._n_models, dtype=np.float64)

        for bag_id in range(n_bags):
            bag_rng = np.random.RandomState(self.config.seed + bag_id)

            # Bootstrap 采样（有放回）
            if n_bags > 1:
                n_select = max(1, int(n_samples * self.config.bag_fraction))
                bag_idx  = bag_rng.choice(n_samples, size=n_select, replace=True)
                y_bag    = y[bag_idx]
                oof_bag  = self.oof_matrix[bag_idx]
            else:
                y_bag   = y
                oof_bag = self.oof_matrix

            counts, traces = self._run_single_pass(
                y_bag, oof_bag, bag_rng, bag_id
            )
            total_counts += counts
            all_traces.extend(traces)

        # 平均权重（跨 bag）
        total_iters  = self.config.n_iterations * n_bags
        final_counts = total_counts / n_bags  # 每 bag 平均选中次数
        weights      = final_counts / (self.config.n_iterations + 1e-12)

        # 生成最终 OOF 预测（加权平均）
        oof_pred  = self._weighted_predict(self.oof_matrix, weights)
        test_pred = (
            self._weighted_predict(self.test_matrix, weights)
            if self.test_matrix is not None
            else None
        )

        # 构建权重 DataFrame
        weight_df = pd.DataFrame({
            "model":      self.model_names,
            "weight":     weights,
            "abs_weight": np.abs(weights),
            "count":      np.round(final_counts).astype(int),
        }).sort_values("weight", ascending=False).reset_index(drop=True)

        elapsed = time.perf_counter() - t0
        final_score = self._metric_fn(y, oof_pred)
        self._logger.info(
            f"Hill Climbing 完成 | "
            f"T={self.config.n_iterations} × bags={n_bags} | "
            f"最终 OOF score={final_score:.6f} | 耗时 {elapsed:.2f}s"
        )
        self._logger.info(
            f"非零权重模型：{(weights > 1e-6).sum()} / {self._n_models}"
        )

        return oof_pred, test_pred, weight_df, all_traces

    # ------------------------------------------------------------------
    # 单次 Hill Climbing 迭代（单个 bag）
    # ------------------------------------------------------------------

    def _run_single_pass(
        self,
        y:       np.ndarray,          # (N_bag,)
        oof:     np.ndarray,          # (N_bag, K)
        rng:     np.random.RandomState,
        bag_id:  int,
    ) -> tuple[np.ndarray, list[HillClimbingTrace]]:
        """
        在单个 bootstrap 样本上执行贪心 Hill Climbing。

        优化技巧
        --------
        维护一个累积和数组 `running_sum`（shape: N_bag），
        每轮加入模型 k 后：
          running_sum += oof[:, k]
          ensemble_score = metric(y, running_sum / (t + 1))
        这样每轮的集成预测无需重新累加，O(N) 而不是 O(N×K)。
        """
        T        = self.config.n_iterations
        eps      = self.config.random_shuffle_pct
        n_models = self._n_models

        selected_indices:  list[int]              = []
        traces:            list[HillClimbingTrace] = []

        # running_sum: 累积选中模型 OOF 之和
        running_sum = np.zeros(len(y), dtype=np.float64)

        # warm start：以最优单模型为起点
        if self.config.warm_start_with_best and n_models > 0:
            scores_single = np.array([
                self._metric_fn(y, oof[:, k]) for k in range(n_models)
            ])
            best_k = int(np.argmax(scores_single)
                         if self._higher_is_better
                         else np.argmin(scores_single))
            selected_indices.append(best_k)
            running_sum += oof[:, best_k]
            initial_score = scores_single[best_k]
        else:
            initial_score = (
                float("-inf") if self._higher_is_better else float("inf")
            )

        prev_score = initial_score

        for t in range(T):
            is_random = rng.random() < eps

            if is_random:
                # 随机扰动：随机选一个模型
                chosen_k = int(rng.randint(0, n_models))
            else:
                # 贪心搜索：找加入后分数最优的模型
                best_k     = -1
                best_score = float("-inf") if self._higher_is_better else float("inf")
                n_in_set   = len(selected_indices)  # 加入新模型后总数 = n_in_set + 1

                for k in range(n_models):
                    # candidate ensemble = (running_sum + oof[:, k]) / (n_in_set + 1)
                    candidate_pred = (running_sum + oof[:, k]) / (n_in_set + 1)
                    score_k        = self._metric_fn(y, candidate_pred)

                    if (
                        (self._higher_is_better and score_k > best_score) or
                        (not self._higher_is_better and score_k < best_score)
                    ):
                        best_score = score_k
                        best_k     = k

                chosen_k = best_k if best_k >= 0 else 0

            # 更新集成状态
            selected_indices.append(chosen_k)
            running_sum += oof[:, chosen_k]
            current_pred  = running_sum / len(selected_indices)
            current_score = self._metric_fn(y, current_pred)
            delta         = current_score - prev_score
            prev_score    = current_score

            traces.append(HillClimbingTrace(
                round_idx   = t,
                model_added = self.model_names[chosen_k],
                score_after = current_score,
                delta       = delta,
                was_random  = is_random,
                bag_id      = bag_id,
            ))

        # 统计每个模型被选中的次数
        counts = np.zeros(n_models, dtype=np.float64)
        for idx in selected_indices:
            counts[idx] += 1

        return counts, traces

    # ------------------------------------------------------------------
    # 加权预测
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_predict(
        pred_matrix: np.ndarray,   # (N, K)
        weights:     np.ndarray,   # (K,)
    ) -> np.ndarray:
        """加权平均：忽略零权重模型（数值稳定）。"""
        mask = weights > 1e-12
        if not mask.any():
            # 退化：所有权重为零，均匀平均
            return pred_matrix.mean(axis=1)
        w = weights[mask]
        p = pred_matrix[:, mask]
        return (p * w[np.newaxis, :]).sum(axis=1) / w.sum()

    # ------------------------------------------------------------------
    # 自动 Metric 选择
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_metric(
        metric_fn: Optional[Callable],
        task:      str,
    ) -> Callable:
        if metric_fn is not None:
            return metric_fn

        try:
            from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
        except ImportError:
            raise ImportError("sklearn 未安装，请执行：pip install scikit-learn")

        task = task.lower()
        if task == "binary":
            return roc_auc_score          # 越大越好
        elif task == "regression":
            # 返回负 MSE（越大越好，与 higher_is_better=True 保持一致）
            def neg_mse(y_true, y_pred):
                return -float(mean_squared_error(y_true, y_pred))
            return neg_mse
        elif task == "multiclass":
            return accuracy_score
        else:
            # 默认用负 MSE
            def neg_mse_default(y_true, y_pred):
                return -float(np.mean((y_true - y_pred) ** 2))
            return neg_mse_default


# ---------------------------------------------------------------------------
# StackingEnsemble — 统一门面类（向后兼容）
# ---------------------------------------------------------------------------

class StackingEnsemble:
    """
    两层集成门面类，兼容 Stacking 和 Hill Climbing 两种策略。

    通过 StackingConfig.method 选择策略：
      method="stacking"      → Ridge/LogReg 元学习器（默认，与旧版一致）
      method="hill_climbing" → Caruana Ensemble Selection（新）

    旧版调用方式完全不变：
    >>> stacker = StackingEnsemble(results=[...], model_names=[...], config=StackingConfig())
    >>> sr = stacker.fit(y_train)
    >>> print(sr.weight_report())
    """

    def __init__(
        self,
        results:          list[CVResult],
        model_names:      Optional[list[str]]    = None,
        config:           Optional[StackingConfig] = None,
        X_train_original: Optional[pd.DataFrame] = None,
        X_test_original:  Optional[pd.DataFrame] = None,
        verbose:          bool = True,
    ):
        if not results:
            raise ValueError("results 列表不能为空。")

        self.results          = results
        self.config           = config or StackingConfig()
        self.X_train_original = X_train_original
        self.X_test_original  = X_test_original
        self.verbose          = verbose
        self._logger          = _build_logger("stacking")

        self.model_names = model_names or [
            f"model_{i}" for i in range(len(results))
        ]
        if len(self.model_names) != len(results):
            raise ValueError(
                f"model_names 长度（{len(self.model_names)}）"
                f"与 results 长度（{len(results)}）不一致。"
            )

        self._stacking_result: Optional[StackingResult] = None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def fit(self, y: Union[pd.Series, np.ndarray]) -> StackingResult:
        """
        按 config.method 路由到对应策略。
        两种策略均返回 StackingResult（接口完全一致）。
        """
        y_arr = np.asarray(y)
        method = (self.config.method or "stacking").lower()

        self._log(f"{'=' * 62}")
        self._log(
            f"  集成训练 | 策略={method.upper()} | "
            f"模型数={len(self.results)} | task={self.config.task}"
        )
        self._log(f"{'=' * 62}")

        self._validate_results(y_arr)

        if method == "hill_climbing":
            result = self._fit_hill_climbing(y_arr)
        else:
            result = self._fit_stacking(y_arr)

        self._stacking_result = result
        self._log("\n" + result.weight_report())
        if result.hill_climbing_trace:
            self._log("\n" + result.hill_climbing_report())
        return result

    # ------------------------------------------------------------------
    # 策略 A：Stacking
    # ------------------------------------------------------------------

    def _fit_stacking(self, y_arr: np.ndarray) -> StackingResult:
        X_meta_train, X_meta_test = self._build_meta_features()
        self._log(
            f"元特征矩阵：train={X_meta_train.shape} | "
            f"test={X_meta_test.shape if X_meta_test is not None else 'N/A'}"
        )

        X_tr_fit, X_te_fit, _ = self._maybe_scale(X_meta_train, X_meta_test)
        meta_learner           = self._build_meta_learner()
        self._log(f"元学习器：{meta_learner.__class__.__name__}")

        t0 = time.perf_counter()
        meta_learner.fit(X_tr_fit, y_arr)
        fit_time = time.perf_counter() - t0
        self._log(f"元学习器训练完成，耗时 {fit_time:.3f}s")

        meta_oof  = self._meta_predict(meta_learner, X_tr_fit)
        meta_test = (
            self._meta_predict(meta_learner, X_te_fit)
            if X_te_fit is not None else None
        )
        weights_df = self._extract_weights(
            meta_learner, X_meta_train.columns.tolist()
        )

        return StackingResult(
            meta_oof            = meta_oof,
            meta_test           = meta_test,
            X_meta_train        = X_meta_train,
            X_meta_test         = X_meta_test,
            model_weights       = weights_df,
            meta_learner        = meta_learner,
            fit_time            = fit_time,
            hill_climbing_trace = None,
        )

    # ------------------------------------------------------------------
    # 策略 B：Hill Climbing
    # ------------------------------------------------------------------

    def _fit_hill_climbing(self, y_arr: np.ndarray) -> StackingResult:
        t0 = time.perf_counter()

        # ── 构建 OOF / Test 矩阵 ──────────────────────────────────
        oof_cols  = []
        test_cols = []
        for result in self.results:
            oof = np.nan_to_num(result.oof_predictions.astype(np.float64), nan=0.0)
            if self.config.clip_oof:
                lo, hi = np.nanpercentile(oof, 1), np.nanpercentile(oof, 99)
                oof    = np.clip(oof, lo, hi)
            oof_cols.append(oof)

            if result.test_predictions is not None:
                test_cols.append(result.test_predictions.astype(np.float64))

        oof_matrix  = np.column_stack(oof_cols)                                # (N, K)
        test_matrix = np.column_stack(test_cols) if test_cols else None        # (M, K)

        # ── 解析 HillClimbingConfig ───────────────────────────────
        hc_cfg = self.config.hill_climbing or HillClimbingConfig()
        if hc_cfg.metric_fn is None:
            # higher_is_better 自动对齐任务类型
            task = self.config.task.lower()
            hc_cfg.higher_is_better = (task != "regression")

        self._log(
            f"Hill Climbing | T={hc_cfg.n_iterations} | "
            f"bags={hc_cfg.n_bags} | "
            f"random_pct={hc_cfg.random_shuffle_pct:.2%} | "
            f"warm_start={hc_cfg.warm_start_with_best}"
        )

        # ── 执行 Hill Climbing ────────────────────────────────────
        hce = HillClimbingEnsemble(
            oof_matrix  = oof_matrix,
            test_matrix = test_matrix,
            model_names = self.model_names,
            config      = hc_cfg,
            task        = self.config.task,
        )
        oof_pred, test_pred, weight_df, traces = hce.fit(y_arr)

        # ── 构建元特征 DataFrame（供下层 Stacking 复用）──────────
        X_meta_train = pd.DataFrame(
            oof_matrix, columns=self.model_names
        )
        X_meta_test  = (
            pd.DataFrame(test_matrix, columns=self.model_names)
            if test_matrix is not None else None
        )

        fit_time = time.perf_counter() - t0

        # 把 Hill Climbing 引擎本身作为 meta_learner 槽位传出（便于序列化）
        return StackingResult(
            meta_oof            = oof_pred,
            meta_test           = test_pred,
            X_meta_train        = X_meta_train,
            X_meta_test         = X_meta_test,
            model_weights       = weight_df,
            meta_learner        = hce,
            fit_time            = fit_time,
            hill_climbing_trace = traces,
        )

    # ------------------------------------------------------------------
    # 校验（与旧版一致）
    # ------------------------------------------------------------------

    def _validate_results(self, y: np.ndarray) -> None:
        n_train = len(y)
        for name, result in zip(self.model_names, self.results):
            oof_len = len(result.oof_predictions)
            if oof_len != n_train:
                raise ValueError(
                    f"[校验失败] {name} 的 OOF 长度（{oof_len}）"
                    f"与标签 y 长度（{n_train}）不一致。"
                )

        oof_lengths = {n: len(r.oof_predictions)
                       for n, r in zip(self.model_names, self.results)}
        if len(set(oof_lengths.values())) > 1:
            raise ValueError(f"[校验失败] OOF 长度不一致：{oof_lengths}")

        for name, result in zip(self.model_names, self.results):
            nan_cnt = np.isnan(result.oof_predictions).sum()
            if nan_cnt == n_train:
                raise ValueError(f"[校验失败] {name} 的 OOF 全为 NaN。")
            if nan_cnt > 0:
                warnings.warn(
                    f"[校验警告] {name} OOF 含 {nan_cnt} 个 NaN，将填 0。",
                    UserWarning, stacklevel=2,
                )

        test_lengths = {
            n: len(r.test_predictions)
            for n, r in zip(self.model_names, self.results)
            if r.test_predictions is not None
        }
        if test_lengths and len(set(test_lengths.values())) > 1:
            raise ValueError(
                f"[校验失败] test_predictions 长度不一致：{test_lengths}"
            )

        missing_test = [
            n for n, r in zip(self.model_names, self.results)
            if r.test_predictions is None
        ]
        if missing_test:
            warnings.warn(
                f"[校验警告] 以下模型无测试集预测：{missing_test}",
                UserWarning, stacklevel=2,
            )

        self._log(
            f"[校验] ✓ | OOF N={n_train:,} | "
            f"有测试集预测={len(test_lengths)}/{len(self.results)}"
        )

    # ------------------------------------------------------------------
    # 元特征构建（Stacking 策略专用）
    # ------------------------------------------------------------------

    def _build_meta_features(self) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        oof_cols, test_cols = {}, {}
        for name, result in zip(self.model_names, self.results):
            oof = np.nan_to_num(result.oof_predictions.astype(np.float64), nan=0.0)
            if self.config.clip_oof:
                lo, hi = np.nanpercentile(oof, 1), np.nanpercentile(oof, 99)
                oof    = np.clip(oof, lo, hi)
            oof_cols[name] = oof
            if result.test_predictions is not None:
                test_cols[name] = result.test_predictions.astype(np.float64)

        X_meta_train = pd.DataFrame(oof_cols)
        X_meta_test  = pd.DataFrame(test_cols) if test_cols else None

        extra_cols = (
            self.config.append_original_features
            or (self.X_train_original.columns.tolist()
                if self.config.passthrough and self.X_train_original is not None
                else [])
        )
        if extra_cols and self.X_train_original is not None:
            extra_tr = self.X_train_original[extra_cols].select_dtypes(include=np.number)
            X_meta_train = pd.concat(
                [X_meta_train, extra_tr.reset_index(drop=True)], axis=1
            )
            if X_meta_test is not None and self.X_test_original is not None:
                X_meta_test = pd.concat(
                    [X_meta_test,
                     self.X_test_original[extra_tr.columns].reset_index(drop=True)],
                    axis=1,
                )
        return X_meta_train, X_meta_test

    def _maybe_scale(self, X_train, X_test):
        if not self.config.scale_meta_features:
            return (
                X_train.values,
                X_test.values if X_test is not None else None,
                None,
            )
        scaler = StandardScaler()
        return (
            scaler.fit_transform(X_train.values),
            scaler.transform(X_test.values) if X_test is not None else None,
            scaler,
        )

    def _build_meta_learner(self):
        if self.config.meta_learner is not None:
            return self.config.meta_learner
        task = self.config.task.lower()
        if task == "regression":
            return Ridge(alpha=1.0)
        elif task in ("binary", "multiclass"):
            multi = "multinomial" if task == "multiclass" else "ovr"
            return LogisticRegression(
                C=0.1, max_iter=1000, solver="lbfgs", multi_class=multi
            )
        return Ridge(alpha=1.0)

    def _meta_predict(self, learner, X) -> np.ndarray:
        task = self.config.task.lower()
        if task == "regression":
            return learner.predict(X)
        if hasattr(learner, "predict_proba"):
            proba = learner.predict_proba(X)
            return proba[:, 1] if (task == "binary" and proba.ndim == 2) else proba
        return learner.predict(X)

    def _extract_weights(self, learner, feature_names) -> pd.DataFrame:
        weights = None
        if hasattr(learner, "coef_"):
            coef    = learner.coef_
            coef    = np.abs(coef).mean(axis=0) if coef.ndim == 2 else coef.flatten()
            weights = coef
        elif hasattr(learner, "feature_importances_"):
            weights = learner.feature_importances_

        if weights is None:
            return pd.DataFrame(columns=["model", "weight", "abs_weight"])

        n    = min(len(weights), len(feature_names))
        rows = [
            {"model": feature_names[i],
             "weight": float(weights[i]),
             "abs_weight": float(abs(weights[i]))}
            for i in range(n)
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("abs_weight", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # 公共工具方法（与旧版保持相同签名）
    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        if self._stacking_result is None:
            raise RuntimeError("请先调用 fit()。")
        if self._stacking_result.meta_test is None:
            raise RuntimeError("测试集预测不存在，请确保 base model 传入了 X_test。")
        return self._stacking_result.meta_test

    def correlation_matrix(self) -> pd.DataFrame:
        oof_dict = {
            n: r.oof_predictions
            for n, r in zip(self.model_names, self.results)
        }
        return pd.DataFrame(oof_dict).corr()

    def diversity_report(self) -> str:
        corr  = self.correlation_matrix()
        names = corr.columns.tolist()
        lines = [
            "=" * 58,
            "  Base Model OOF 预测多样性分析",
            "  （相关系数越低 → 多样性越高 → 集成收益越大）",
            "=" * 58,
        ]
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = corr.iloc[i, j]
                pairs.append((names[i], names[j], r))
                lines.append(f"  {names[i]:<18} ↔  {names[j]:<18} r={r:.4f}")
        if pairs:
            avg_r = np.mean([abs(p[2]) for p in pairs])
            lines += [
                f"  {'─' * 52}",
                f"  平均 |r| = {avg_r:.4f}",
                "  ⚠  高度相关，集成收益有限" if avg_r > 0.98 else
                ("  →  中等相关，集成仍有效" if avg_r > 0.90 else
                 "  ✓  低相关，集成收益显著"),
            ]
        lines.append("=" * 58)
        return "\n".join(lines)

    def oof_score_report(
        self,
        y:           Union[pd.Series, np.ndarray],
        metric_fn:   Callable[[np.ndarray, np.ndarray], float],
        metric_name: str = "score",
    ) -> pd.DataFrame:
        y_arr = np.asarray(y)
        rows  = []
        for name, result in zip(self.model_names, self.results):
            oof   = np.nan_to_num(result.oof_predictions.copy(), nan=0.0)
            rows.append({"model": name, metric_name: metric_fn(y_arr, oof)})
        if self._stacking_result is not None:
            ens_score = metric_fn(y_arr, self._stacking_result.meta_oof)
            method    = (self.config.method or "stacking").upper()
            rows.append({"model": f"★ {method}", metric_name: ens_score})
        return pd.DataFrame(rows).set_index("model")

    @property
    def result(self) -> StackingResult:
        if self._stacking_result is None:
            raise RuntimeError("请先调用 fit()。")
        return self._stacking_result

    @property
    def meta_oof(self) -> np.ndarray:
        return self.result.meta_oof

    @property
    def model_weights(self) -> pd.DataFrame:
        return self.result.model_weights

    def _log(self, msg: str) -> None:
        if self.verbose:
            self._logger.info(msg)

    def __repr__(self) -> str:
        return (
            f"StackingEnsemble("
            f"n_models={len(self.results)}, "
            f"method={self.config.method!r}, "
            f"task={self.config.task!r}, "
            f"fitted={self._stacking_result is not None})"
        )
