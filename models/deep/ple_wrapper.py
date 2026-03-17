"""
models/deep/ple_wrapper.py
---------------------------
PLE (Progressive Layered Extraction) 多任务学习模型。

论文
----
  "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL)
   Model for Personalized Recommendations"
  Tang et al., RecSys 2020 — Tencent

架构简介
--------
  传统 MMOE 的缺点：所有任务共用同一组专家池，门控网络容易产生跷跷板效应
  （seesaw phenomenon），当两个任务存在负迁移时，模型倾向于牺牲一个任务。

  PLE 改进：
    - 每个任务拥有自己的 Specific Experts（避免任务间干扰）
    - 保留 Shared Experts（促进知识共享）
    - 多层渐进提取：每一层的专家以上一层的门控输出为输入，
      让高层专家能聚焦于更抽象的任务特有表征

  可视化结构（2-level, 2-task 为例）：

    Input ──────────────────────────────────────────
               │
               ├── Task-1 Specific Experts × K₁  ──┐
               ├── Task-2 Specific Experts × K₂  ──┤ Level-1
               └── Shared Experts          × Ks  ──┘
                         │
               ┌──────────────────────────┐
               │  Gate₁(task-1 + shared)  │ → h₁⁽¹⁾
               │  Gate₂(task-2 + shared)  │ → h₂⁽¹⁾
               │  GateS(all experts)      │ → hS⁽¹⁾  （中间层才有）
               └──────────────────────────┘
                         │
               ├── Task-1 Specific Experts × K₁  ──┐
               ├── Task-2 Specific Experts × K₂  ──┤ Level-2（最后一层）
               └── Shared Experts          × Ks  ──┘
                         │
               ┌──────────────────────────┐
               │  Gate₁(task-1 + shared)  │ → h₁⁽²⁾
               │  Gate₂(task-2 + shared)  │ → h₂⁽²⁾
               └──────────────────────────┘
                         │
               ┌──────────────────┐
               │  CTR Tower DNN   │ → sigmoid → pCTR
               │  CVR Tower DNN   │ → sigmoid → pCVR
               └──────────────────┘
                         │
               pCTCVR = pCTR × pCVR（ESMM 全空间训练思想）

依赖
----
  pip install torch scikit-learn numpy pandas
  （无需 deepctr-torch，完全原生 PyTorch 实现）
"""

from __future__ import annotations

import copy
import io
import json
import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.base_model import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 依赖守卫
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise ImportError(
            "PyTorch 未安装。请执行：\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
            "  # CPU 版本：pip install torch"
        )


# ---------------------------------------------------------------------------
# 超参数容器
# ---------------------------------------------------------------------------

@dataclass
class PLEParams:
    """
    PLE 全量超参数，支持从 dict 初始化（与 YAML 无缝对接）。

    关键参数说明
    ------------
    n_levels        : PLE 提取层数。1 层退化为 MMOE；2~3 层是竞赛最佳实践
    n_specific      : 每个任务的专用专家数量
    n_shared        : 共享专家数量（建议 ≥ n_specific）
    expert_units    : 每个专家网络的隐层维度列表
    tower_units     : 任务塔 DNN 隐层维度列表
    ctr_loss_weight : CTR 任务的损失权重
    cvr_loss_weight : CVR 任务的损失权重（pCTCVR = pCTR × pCVR 的监督信号）
    label_click     : DataFrame 中点击标签的列名
    label_conversion: DataFrame 中转化标签的列名（全空间标签，非点击后标签）
    """

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_dim: int = 8

    # ── PLE 架构 ───────────────────────────────────────────────────────────
    n_levels:   int  = 2               # 提取层数
    n_specific: int  = 2               # 每任务专用专家数
    n_shared:   int  = 2               # 共享专家数
    expert_units:  tuple = (128, 64)   # 每个专家 MLP 隐层
    expert_activation: str = "relu"
    expert_dropout:    float = 0.0
    gate_activation:   str = "softmax"

    # ── 任务塔 ─────────────────────────────────────────────────────────────
    tower_units:     tuple = (64, 32)
    tower_activation: str  = "relu"
    tower_dropout:   float = 0.1

    # ── 正则化 ─────────────────────────────────────────────────────────────
    l2_reg_embedding: float = 1e-6
    l2_reg_expert:    float = 0.0
    l2_reg_tower:     float = 0.0

    # ── 优化器 ─────────────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    optimizer: str = "adam"           # adam | adagrad | rmsprop
    weight_decay: float = 0.0

    # ── 多目标损失权重 ──────────────────────────────────────────────────────
    ctr_loss_weight:  float = 1.0     # w₁·Loss_CTR
    cvr_loss_weight:  float = 1.0     # w₂·Loss_CVR（通过 pCTCVR 监督）

    # ── 标签列名 ───────────────────────────────────────────────────────────
    label_click:      str = "label_click"
    label_conversion: str = "label_conversion"

    # ── 训练控制 ───────────────────────────────────────────────────────────
    batch_size: int  = 4096
    epochs:     int  = 50
    device:     str  = "auto"

    # ── Early Stopping ─────────────────────────────────────────────────────
    patience:     int   = 5
    min_delta:    float = 1e-5
    monitor_mode: str   = "min"   # "min"（val_loss）

    @classmethod
    def from_dict(cls, d: dict) -> "PLEParams":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        obj   = cls(**valid)
        for attr in ("expert_units", "tower_units"):
            if isinstance(getattr(obj, attr), list):
                setattr(obj, attr, tuple(getattr(obj, attr)))
        return obj

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Early Stopping（与 deepfm_wrapper 保持一致的实现）
# ---------------------------------------------------------------------------

class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 1e-5, mode: str = "min"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self._counter   = 0
        self._best      = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0

    def step(self, value: float, epoch: int) -> bool:
        improved = (
            value < self._best - self.min_delta
            if self.mode == "min"
            else value > self._best + self.min_delta
        )
        if improved:
            self._best      = value
            self._counter   = 0
            self.best_epoch = epoch
        else:
            self._counter += 1
        return self._counter >= self.patience

    @property
    def best_value(self) -> float:
        return self._best


# ---------------------------------------------------------------------------
# 特征编码器（与 deepfm_wrapper 完全一致）
# ---------------------------------------------------------------------------

class FeatureEncoder:
    """Label Encoding + 词表管理，保证 train/val/test 编码一致。"""

    def __init__(self):
        self._encoders:         dict[str, dict] = {}
        self._vocabulary_sizes: dict[str, int]  = {}

    def fit(self, X: pd.DataFrame, cat_cols: list[str]) -> "FeatureEncoder":
        for col in cat_cols:
            uniques = X[col].dropna().unique().tolist()
            mapping = {v: i + 1 for i, v in enumerate(sorted(map(str, uniques)))}
            self._encoders[col]         = mapping
            self._vocabulary_sizes[col] = len(mapping) + 1  # +1 for unknown
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self._encoders.items():
            if col not in X.columns:
                X[col] = 0
                continue
            X[col] = (
                X[col].astype(str).map(mapping).fillna(0).astype(np.int32)
            )
        return X

    @property
    def vocabulary_sizes(self) -> dict[str, int]:
        return self._vocabulary_sizes


# ---------------------------------------------------------------------------
# 原生 PyTorch 神经网络组件
# ---------------------------------------------------------------------------

def _get_activation(name: str):
    """返回 nn.Module 激活函数实例。"""
    import torch.nn as nn
    _map = {
        "relu":    nn.ReLU(),
        "gelu":    nn.GELU(),
        "tanh":    nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "dice":    None,  # Dice 见下方
        "prelu":   nn.PReLU(),
        "selu":    nn.SELU(),
        "leaky":   nn.LeakyReLU(0.1),
    }
    act = _map.get(name.lower())
    if act is None and name.lower() != "dice":
        raise ValueError(f"未知激活函数：{name}。支持：{list(_map.keys())}")
    return act


class DiceActivation:
    """
    Dice 激活函数（DIN/PLE 论文推荐）。
    公式：Dice(x) = p(x) · x + (1 - p(x)) · αx
          p(x)   = sigmoid((x - E[x]) / sqrt(Var[x] + ε))
    """
    @staticmethod
    def build(input_dim: int):
        import torch
        import torch.nn as nn

        class _Dice(nn.Module):
            def __init__(self, dim: int, eps: float = 1e-9):
                super().__init__()
                self.alpha   = nn.Parameter(torch.zeros(dim))
                self.eps     = eps
                self.bn      = nn.BatchNorm1d(dim, affine=False)

            def forward(self, x):
                normed = self.bn(x)
                p = torch.sigmoid(normed)
                return p * x + (1 - p) * self.alpha * x

        return _Dice(input_dim)


def _build_mlp(
    input_dim:  int,
    hidden_units: tuple,
    activation: str  = "relu",
    dropout:    float = 0.0,
    use_bn:     bool  = False,
    output_dim: Optional[int] = None,
):
    """构建标准 MLP，返回 nn.Sequential。"""
    import torch.nn as nn

    layers = []
    dims   = [input_dim] + list(hidden_units)

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if use_bn:
            layers.append(nn.BatchNorm1d(dims[i + 1]))
        if activation.lower() == "dice":
            layers.append(DiceActivation.build(dims[i + 1]))
        else:
            layers.append(_get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    if output_dim is not None:
        layers.append(nn.Linear(dims[-1], output_dim))

    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# PLE 核心网络组件
# ---------------------------------------------------------------------------

class ExpertNetwork:
    """
    单个专家 MLP 网络。
    input_dim  → expert_units → output_dim (= expert_units[-1])
    """
    @staticmethod
    def build(input_dim: int, units: tuple, activation: str, dropout: float):
        return _build_mlp(input_dim, units, activation=activation, dropout=dropout)


class PLEGatingNetwork:
    """
    门控网络：对 n_experts 个专家输出做 softmax 加权求和。

    forward(gate_input, expert_outputs):
      logits  = Linear(gate_input)  shape: (B, n_experts)
      weights = softmax(logits)
      output  = sum_k(weights[:, k] · expert_outputs[k])  shape: (B, expert_dim)
    """
    @staticmethod
    def build(gate_input_dim: int, n_experts: int):
        import torch.nn as nn
        return nn.Linear(gate_input_dim, n_experts, bias=False)


# ---------------------------------------------------------------------------
# PLE 完整网络（nn.Module）
# ---------------------------------------------------------------------------

class _PLENetwork:
    """
    工厂类：构建整个 PLE nn.Module。

    参数
    ----
    input_dim   : 特征输入维度（Embedding 拼接 + Dense）
    hp          : PLEParams
    n_tasks     : 当前固定为 2（CTR + CVR）
    """

    @staticmethod
    def build(input_dim: int, hp: PLEParams, n_tasks: int = 2):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class PLEModule(nn.Module):

            def __init__(self):
                super().__init__()
                self.hp      = hp
                self.n_tasks = n_tasks
                n_lv  = hp.n_levels
                n_sp  = hp.n_specific
                n_sh  = hp.n_shared
                e_dim = hp.expert_units[-1]   # 专家输出维度

                # ── 专家网络（每层 × 每任务 / 共享）─────────────────────────
                # self.specific[l][t]  = ModuleList(n_specific experts)
                # self.shared[l]       = ModuleList(n_shared experts)
                self.specific = nn.ModuleList()
                self.shared   = nn.ModuleList()

                for lv in range(n_lv):
                    # 输入维度：第 0 层 = input_dim，之后 = e_dim
                    in_dim = input_dim if lv == 0 else e_dim
                    # 每个任务的专用专家组
                    task_group = nn.ModuleList([
                        nn.ModuleList([
                            _build_mlp(in_dim, hp.expert_units,
                                       activation=hp.expert_activation,
                                       dropout=hp.expert_dropout)
                            for _ in range(n_sp)
                        ])
                        for _ in range(n_tasks)
                    ])
                    self.specific.append(task_group)

                    # 共享专家组
                    self.shared.append(nn.ModuleList([
                        _build_mlp(in_dim, hp.expert_units,
                                   activation=hp.expert_activation,
                                   dropout=hp.expert_dropout)
                        for _ in range(n_sh)
                    ]))

                # ── 门控网络──────────────────────────────────────────────────
                # task gate at level lv:  uses n_specific + n_shared experts
                # shared gate at level lv (not final): uses n_specific*n_tasks + n_shared
                self.task_gates   = nn.ModuleList()  # [lv × n_tasks]
                self.shared_gates = nn.ModuleList()  # [lv - 1]（中间层）

                for lv in range(n_lv):
                    gate_in = input_dim if lv == 0 else e_dim
                    lv_task_gates = nn.ModuleList([
                        PLEGatingNetwork.build(gate_in, n_sp + n_sh)
                        for _ in range(n_tasks)
                    ])
                    self.task_gates.append(lv_task_gates)

                    # 中间层的 shared gate（最后一层不需要）
                    if lv < n_lv - 1:
                        self.shared_gates.append(
                            PLEGatingNetwork.build(gate_in, n_sp * n_tasks + n_sh)
                        )

                # ── 任务塔 ──────────────────────────────────────────────────
                self.ctr_tower = _build_mlp(
                    e_dim, hp.tower_units,
                    activation=hp.tower_activation,
                    dropout=hp.tower_dropout,
                    output_dim=1,
                )
                self.cvr_tower = _build_mlp(
                    e_dim, hp.tower_units,
                    activation=hp.tower_activation,
                    dropout=hp.tower_dropout,
                    output_dim=1,
                )

            def forward(self, x: "torch.Tensor"):
                """
                x : (B, input_dim)
                Returns
                -------
                p_ctr   : (B,)
                p_cvr   : (B,)
                p_ctcvr : (B,)   = p_ctr × p_cvr
                """
                # task_hiddens[t] = 上一层 gate 对任务 t 的输出，初始 = x
                task_hiddens   = [x for _ in range(self.n_tasks)]
                shared_hidden  = x

                n_lv = self.hp.n_levels
                n_sp = self.hp.n_specific
                n_sh = self.hp.n_shared

                for lv in range(n_lv):
                    is_last = (lv == n_lv - 1)

                    # ── 专家前向 ─────────────────────────────────────────────
                    # specific_out[t][k]: (B, e_dim)
                    specific_out = [
                        [expert(task_hiddens[t]) for expert in self.specific[lv][t]]
                        for t in range(self.n_tasks)
                    ]
                    # shared_out[k]: (B, e_dim)
                    shared_out = [exp(shared_hidden) for exp in self.shared[lv]]

                    # ── 任务门控 ──────────────────────────────────────────────
                    new_task_hiddens = []
                    for t in range(self.n_tasks):
                        gate_in  = task_hiddens[t]
                        # 候选专家 = 该任务专用 + 共享
                        candidates = specific_out[t] + shared_out      # list of (B, e_dim)
                        experts_stack = torch.stack(candidates, dim=1)  # (B, n_sp+n_sh, e_dim)

                        logits  = self.task_gates[lv][t](gate_in)       # (B, n_sp+n_sh)
                        weights = F.softmax(logits, dim=-1).unsqueeze(2) # (B, n_sp+n_sh, 1)
                        h = (weights * experts_stack).sum(dim=1)         # (B, e_dim)
                        new_task_hiddens.append(h)

                    # ── 共享门控（中间层）────────────────────────────────────
                    if not is_last:
                        # 候选专家 = 所有专用 + 共享
                        all_sp = [e for t_out in specific_out for e in t_out]
                        all_candidates = all_sp + shared_out
                        experts_stack  = torch.stack(all_candidates, dim=1)

                        logits  = self.shared_gates[lv](shared_hidden)
                        weights = F.softmax(logits, dim=-1).unsqueeze(2)
                        shared_hidden = (weights * experts_stack).sum(dim=1)

                    task_hiddens = new_task_hiddens

                # ── 任务塔输出 ───────────────────────────────────────────────
                p_ctr   = torch.sigmoid(self.ctr_tower(task_hiddens[0])).squeeze(1)
                p_cvr   = torch.sigmoid(self.cvr_tower(task_hiddens[1])).squeeze(1)
                p_ctcvr = p_ctr * p_cvr

                return p_ctr, p_cvr, p_ctcvr

        return PLEModule()


# ---------------------------------------------------------------------------
# PLEModel：BaseModel 子类
# ---------------------------------------------------------------------------

class PLEModel(BaseModel):
    """
    PLE 多任务推荐模型 Wrapper。

    功能
    ----
    - 继承 BaseModel，与 CrossValidatorTrainer / StackingEnsemble 完全兼容
    - 原生 PyTorch 实现，无 DeepCTR-Torch 依赖
    - 自动特征分拣：category → Embedding，numeric → Dense
    - Early Stopping + Best Checkpoint
    - 多目标：同时优化 CTR 和 CVR（pCTCVR = pCTR × pCVR）
    - predict_proba 默认返回 pCTCVR（与 CrossValidatorTrainer 兼容）

    用法
    ----
    >>> model = PLEModel(task="binary", seed=42)
    >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    >>> proba = model.predict_proba(X_test)   # pCTCVR，shape=(N,)
    >>> p_ctr, p_cvr = model.predict_ctr(X_test), model.predict_cvr(X_test)
    """

    DEFAULT_PARAMS: dict = {
        "embedding_dim":    8,
        "n_levels":         2,
        "n_specific":       2,
        "n_shared":         2,
        "expert_units":     [128, 64],
        "expert_activation": "relu",
        "expert_dropout":   0.0,
        "tower_units":      [64, 32],
        "tower_dropout":    0.1,
        "ctr_loss_weight":  1.0,
        "cvr_loss_weight":  1.0,
        "label_click":      "label_click",
        "label_conversion": "label_conversion",
        "learning_rate":    1e-3,
        "batch_size":       4096,
        "epochs":           50,
        "patience":         5,
        "device":           "auto",
    }

    def __init__(
        self,
        params:    Optional[dict] = None,
        task:      str  = "binary",
        name:      str  = "ple",
        seed:      int  = 42,
    ):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged, task=task, name=name, seed=seed)

        self.hp: PLEParams = PLEParams.from_dict(merged)
        self._device_str: Optional[str] = None

        # 运行时属性
        self._net:               Optional[object] = None   # PLEModule
        self._encoder:           FeatureEncoder   = FeatureEncoder()
        self._cat_cols:          list[str]         = []
        self._num_cols:          list[str]         = []
        self._feature_cols:      list[str]         = []
        self._input_dim:         int               = 0
        self._best_weights:      Optional[bytes]   = None  # state_dict buffer
        self._feature_importance_cache: Optional[pd.DataFrame] = None

        # 动态负采样配置（由 train_pipeline.py 注入，None = 不启用）
        self.negative_sampler_config: Optional[object] = None

    # ------------------------------------------------------------------
    # 设备解析
    # ------------------------------------------------------------------

    def _resolve_device(self) -> "torch.device":
        torch, _ = _require_torch()
        setting = self.hp.device
        if setting == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(setting)

    # ------------------------------------------------------------------
    # 特征列分拣
    # ------------------------------------------------------------------

    def _identify_feature_cols(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        """
        返回 (cat_cols, num_cols)。
        - category dtype → 稀疏特征（Embedding）
        - int / float / bool → 稠密特征（归一化后直接输入）
        """
        label_cols = {self.hp.label_click, self.hp.label_conversion}
        cat_cols, num_cols = [], []
        for col in X.columns:
            if col in label_cols:
                continue
            if pd.api.types.is_categorical_dtype(X[col]) or X[col].dtype == object:
                cat_cols.append(col)
            elif pd.api.types.is_numeric_dtype(X[col]):
                num_cols.append(col)
        return cat_cols, num_cols

    def _compute_input_dim(self) -> int:
        """Embedding 拼接 + Dense 拼接的总维度。"""
        emb_dim = len(self._cat_cols) * self.hp.embedding_dim
        den_dim = len(self._num_cols)
        return emb_dim + den_dim

    # ------------------------------------------------------------------
    # 数据预处理 → Tensor
    # ------------------------------------------------------------------

    def _to_tensor(self, X: pd.DataFrame) -> "torch.Tensor":
        torch, _ = _require_torch()
        X_enc    = self._encoder.transform(X)
        device   = torch.device(self._device_str)

        parts = []
        if self._cat_cols:
            emb_dim = self.hp.embedding_dim
            for col in self._cat_cols:
                idx  = torch.tensor(X_enc[col].values, dtype=torch.long, device=device)
                emb  = self._net.embeddings[col](idx)        # (B, emb_dim)
                parts.append(emb)

        if self._num_cols:
            num_arr = X_enc[self._num_cols].values.astype(np.float32)
            parts.append(torch.tensor(num_arr, device=device))

        return torch.cat(parts, dim=1)  # (B, input_dim)

    # ------------------------------------------------------------------
    # 训练标签提取
    # ------------------------------------------------------------------

    def _extract_labels(self, X: pd.DataFrame):
        """从 DataFrame 提取多目标标签，返回 (y_click, y_conv) numpy arrays。"""
        if self.hp.label_click not in X.columns:
            raise ValueError(
                f"未找到点击标签列 '{self.hp.label_click}'。\n"
                f"请确保 DataFrame 包含该列，或在 params 中修改 label_click。"
            )
        if self.hp.label_conversion not in X.columns:
            raise ValueError(
                f"未找到转化标签列 '{self.hp.label_conversion}'。\n"
                f"请确保 DataFrame 包含该列，或在 params 中修改 label_conversion。"
            )
        return (
            X[self.hp.label_click].values.astype(np.float32),
            X[self.hp.label_conversion].values.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # 构建网络（含 Embedding Table）
    # ------------------------------------------------------------------

    def _build_network(self):
        torch, nn = _require_torch()

        class FullPLENet(nn.Module):
            """在 PLEModule 外层包装 Embedding Table。"""

            def __init__(self_inner, encoder: FeatureEncoder, cat_cols, emb_dim, input_dim, hp):
                super().__init__()
                self_inner.cat_cols = cat_cols
                self_inner.emb_dim  = emb_dim

                # Embedding 层（每列一个独立 Embedding Table）
                self_inner.embeddings = nn.ModuleDict({
                    col: nn.Embedding(
                        vocab_size,
                        emb_dim,
                        padding_idx=0,
                    )
                    for col, vocab_size in encoder.vocabulary_sizes.items()
                    if col in cat_cols
                })

                # PLE 主体
                self_inner.ple = _PLENetwork.build(input_dim, hp, n_tasks=2)

            def forward(self_inner, x_cat_dict: dict, x_dense: Optional["torch.Tensor"]):
                parts = []
                for col in self_inner.cat_cols:
                    emb = self_inner.embeddings[col](x_cat_dict[col])
                    parts.append(emb)
                if x_dense is not None:
                    parts.append(x_dense)
                x_in = torch.cat(parts, dim=1)
                return self_inner.ple(x_in)   # p_ctr, p_cvr, p_ctcvr

        import torch
        net = FullPLENet(
            encoder   = self._encoder,
            cat_cols  = self._cat_cols,
            emb_dim   = self.hp.embedding_dim,
            input_dim = self._input_dim,
            hp        = self.hp,
        )
        return net

    # ------------------------------------------------------------------
    # Tensor 构建（分离 cat / dense）
    # ------------------------------------------------------------------

    def _prepare_batch(self, X_enc: pd.DataFrame, device: "torch.device"):
        torch, _ = _require_torch()
        cat_dict = {}
        for col in self._cat_cols:
            cat_dict[col] = torch.tensor(X_enc[col].values, dtype=torch.long, device=device)

        x_dense = None
        if self._num_cols:
            arr     = X_enc[self._num_cols].values.astype(np.float32)
            x_dense = torch.tensor(arr, device=device)

        return cat_dict, x_dense

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train:   pd.DataFrame,
        y_train:   Optional[pd.Series] = None,
        X_val:     Optional[pd.DataFrame] = None,
        y_val:     Optional[pd.Series]    = None,
        **kwargs,
    ) -> "PLEModel":
        """
        训练 PLE 模型。

        Parameters
        ----------
        X_train : 含特征列 + label_click + label_conversion 的 DataFrame
        y_train : 忽略（标签从 X_train 的标签列提取，保持接口统一）
        X_val   : 验证集（同上），用于 Early Stopping
        y_val   : 忽略

        Notes
        -----
        如果你不想把标签放进 X_train，可以在 X_train 里手动添加两列：
            X_train["label_click"]      = y_click_train
            X_train["label_conversion"] = y_conv_train
        """
        torch, nn = _require_torch()
        import torch

        t0     = time.perf_counter()
        device = self._resolve_device()
        self._device_str = str(device)
        self._set_seed(self.seed)

        # ── 特征分拣 ──────────────────────────────────────────────────
        self._cat_cols, self._num_cols = self._identify_feature_cols(X_train)
        self._feature_cols = self._cat_cols + self._num_cols
        self._encoder.fit(X_train, self._cat_cols)
        self._input_dim = self._compute_input_dim()

        logger.info(
            f"[{self.name}] 特征: {len(self._cat_cols)} 类别列 "
            f"+ {len(self._num_cols)} 数值列 → 输入维度={self._input_dim}"
        )

        # ── 提取多任务标签 ─────────────────────────────────────────────
        y_click_train, y_conv_train = self._extract_labels(X_train)
        has_val = X_val is not None
        if has_val:
            y_click_val, y_conv_val = self._extract_labels(X_val)

        # ── 构建网络 ───────────────────────────────────────────────────
        self._net = self._build_network().to(device)
        n_params  = sum(p.numel() for p in self._net.parameters() if p.requires_grad)
        logger.info(f"[{self.name}] 可训练参数: {n_params:,}")

        # ── 优化器 ────────────────────────────────────────────────────
        opt_map = {
            "adam":    torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
            "rmsprop": torch.optim.RMSprop,
        }
        opt_cls   = opt_map.get(self.hp.optimizer.lower(), torch.optim.Adam)
        optimizer = opt_cls(
            self._net.parameters(),
            lr           = self.hp.learning_rate,
            weight_decay = self.hp.weight_decay,
        )

        # ── 损失函数：BCE（CTR） + BCE（pCTCVR → CVR 全空间） ─────────
        bce = torch.nn.BCELoss()

        def _weighted_loss(p_ctr, p_cvr, p_ctcvr, y_click, y_conv):
            """
            ESMM 思想的全空间多任务损失：
              L = w₁ · BCE(p_ctr, y_click) + w₂ · BCE(p_ctcvr, y_conv)
            CVR 通过 pCTCVR 在全样本空间监督，解决样本选择偏差。
            """
            loss_ctr   = bce(p_ctr,   y_click)
            loss_ctcvr = bce(p_ctcvr, y_conv)
            return (
                self.hp.ctr_loss_weight  * loss_ctr +
                self.hp.cvr_loss_weight  * loss_ctcvr
            )

        # ── DataLoader ─────────────────────────────────────────────────
        X_enc_train = self._encoder.transform(X_train)
        if has_val:
            X_enc_val = self._encoder.transform(X_val)

        n_train = len(X_train)
        bs      = self.hp.batch_size

        # ── 动态负采样器（可选）──────────────────────────────────────
        # PLE 为多任务模型，标签存储在 X_train 列中（label_click / label_conversion）。
        # 负采样以 label_click 作为正例判据：点击样本(=1)为正，曝光未点击(=0)为负。
        # 采样后整行（含所有标签列）传入 _extract_labels() 重新提取。
        _sampler = None
        ns_cfg = self.negative_sampler_config
        if ns_cfg is not None and getattr(ns_cfg, "enable", False):
            from data.negative_sampler import DynamicNegativeSampler
            # 用 label_click 列定义正负（1=正例，0=负例）
            _y_click_for_sampler = y_click_train.copy()
            _sampler = DynamicNegativeSampler(ns_cfg)
            _sampler.fit(X_train, _y_click_for_sampler)
            _sampler.start()
            logger.info(
                f"[{self.name}] 动态负采样器启动 | "
                f"pos(click=1)={_sampler.n_positives:,} | "
                f"neg_pool={_sampler.n_negatives:,} | "
                f"neg_ratio=1:{ns_cfg.neg_ratio}"
            )

        # ── Early Stopping ─────────────────────────────────────────────
        stopper = EarlyStopper(
            patience  = self.hp.patience,
            min_delta = self.hp.min_delta,
            mode      = "min",
        )

        # ── 训练循环 ───────────────────────────────────────────────────
        for epoch in range(1, self.hp.epochs + 1):
            # 每 Epoch 取新采样数据（负采样启用时 blocking pull from buffer）
            if _sampler is not None:
                X_ep_raw, _ = _sampler.get_next_epoch()
                y_click_ep, y_conv_ep = self._extract_labels(X_ep_raw)
                X_enc_ep = self._encoder.transform(X_ep_raw)
                n_ep     = len(X_ep_raw)
            else:
                X_enc_ep   = X_enc_train
                y_click_ep = y_click_train
                y_conv_ep  = y_conv_train
                n_ep       = n_train

            self._net.train()
            indices  = np.random.permutation(n_ep)
            ep_loss  = 0.0
            n_batches = 0

            for start in range(0, n_ep, bs):
                idx      = indices[start:start + bs]
                X_batch  = X_enc_ep.iloc[idx]
                cat_d, x_dense = self._prepare_batch(X_batch, device)

                y_c = torch.tensor(y_click_ep[idx],   device=device)
                y_v = torch.tensor(y_conv_ep[idx],    device=device)

                optimizer.zero_grad()
                p_ctr, p_cvr, p_ctcvr = self._net(cat_d, x_dense)
                loss = _weighted_loss(p_ctr, p_cvr, p_ctcvr, y_c, y_v)
                loss.backward()
                optimizer.step()

                ep_loss   += loss.item()
                n_batches += 1

            ep_loss /= n_batches

            # ── 验证 ──────────────────────────────────────────────────
            if has_val:
                self._net.eval()
                val_loss = 0.0
                n_vb     = 0
                with torch.no_grad():
                    n_val = len(X_val)
                    for vstart in range(0, n_val, bs):
                        X_vb   = X_enc_val.iloc[vstart:vstart + bs]
                        cat_d, x_dense = self._prepare_batch(X_vb, device)
                        y_vc = torch.tensor(
                            y_click_val[vstart:vstart + bs], device=device)
                        y_vv = torch.tensor(
                            y_conv_val[vstart:vstart + bs],  device=device)
                        pc, pv, pctcvr = self._net(cat_d, x_dense)
                        vl = _weighted_loss(pc, pv, pctcvr, y_vc, y_vv)
                        val_loss += vl.item()
                        n_vb += 1
                val_loss /= n_vb

                monitor_val = val_loss
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"[{self.name}] Epoch {epoch:3d}/{self.hp.epochs} | "
                        f"train_loss={ep_loss:.5f} | val_loss={val_loss:.5f}"
                    )

                # 保存最优 checkpoint
                if stopper.best_epoch == epoch - 1 or stopper._counter == 0:
                    buf = io.BytesIO()
                    torch.save(self._net.state_dict(), buf)
                    self._best_weights = buf.getvalue()

                if stopper.step(monitor_val, epoch):
                    logger.info(
                        f"[{self.name}] Early Stopping 触发 | "
                        f"最优 epoch={stopper.best_epoch} | "
                        f"最优 val_loss={stopper.best_value:.5f}"
                    )
                    break
            else:
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"[{self.name}] Epoch {epoch:3d}/{self.hp.epochs} | "
                        f"train_loss={ep_loss:.5f}"
                    )

        # ── 负采样后台线程回收 ─────────────────────────────────────────
        if _sampler is not None:
            _sampler.stop()

        # ── 恢复最优权重 ───────────────────────────────────────────────
        if self._best_weights is not None:
            buf = io.BytesIO(self._best_weights)
            self._net.load_state_dict(torch.load(buf, map_location=device))
            logger.info(
                f"[{self.name}] 已恢复最优权重 | 最优 epoch={stopper.best_epoch}"
            )

        elapsed = time.perf_counter() - t0
        logger.info(f"[{self.name}] 训练完成 | 耗时 {elapsed:.1f}s")
        return self

    # ------------------------------------------------------------------
    # 推理（内部）
    # ------------------------------------------------------------------

    def _predict_raw(self, X: pd.DataFrame) -> tuple:
        """返回 (p_ctr, p_cvr, p_ctcvr) 三个 np.ndarray，shape=(N,)。"""
        torch, _ = _require_torch()
        import torch

        if self._net is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

        self._net.eval()
        device  = torch.device(self._device_str)
        X_enc   = self._encoder.transform(X)
        bs      = self.hp.batch_size

        ctr_list, cvr_list, ctcvr_list = [], [], []
        with torch.no_grad():
            for start in range(0, len(X), bs):
                X_b        = X_enc.iloc[start:start + bs]
                cat_d, x_dense = self._prepare_batch(X_b, device)
                pc, pv, pctcvr = self._net(cat_d, x_dense)
                ctr_list.append(pc.cpu().numpy())
                cvr_list.append(pv.cpu().numpy())
                ctcvr_list.append(pctcvr.cpu().numpy())

        return (
            np.concatenate(ctr_list).astype(np.float64),
            np.concatenate(cvr_list).astype(np.float64),
            np.concatenate(ctcvr_list).astype(np.float64),
        )

    # ------------------------------------------------------------------
    # 公开预测接口
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        返回 pCTCVR = pCTR × pCVR，shape=(N,)。
        默认接口，与 CrossValidatorTrainer / StackingEnsemble 兼容。
        """
        _, _, p_ctcvr = self._predict_raw(X)
        return p_ctcvr

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """返回 pCTCVR（与 predict_proba 相同）。"""
        return self.predict_proba(X, **kwargs)

    def predict_ctr(self, X: pd.DataFrame) -> np.ndarray:
        """返回点击率 pCTR，shape=(N,)。"""
        p_ctr, _, _ = self._predict_raw(X)
        return p_ctr

    def predict_cvr(self, X: pd.DataFrame) -> np.ndarray:
        """
        返回转化率 pCVR，shape=(N,)。
        注意：pCVR = pCTCVR / (pCTR + ε)，是 CVR 的无偏估计。
        """
        p_ctr, _, p_ctcvr = self._predict_raw(X)
        return p_ctcvr / (p_ctr + 1e-9)

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        返回三列 DataFrame：pCTR | pCVR | pCTCVR。
        适合多目标排序融合。
        """
        p_ctr, p_cvr, p_ctcvr = self._predict_raw(X)
        return pd.DataFrame({
            "pCTR":   p_ctr,
            "pCVR":   p_cvr,
            "pCTCVR": p_ctcvr,
        }, index=X.index)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        保存到目录 path/：
          - ple_weights.pt   : 模型权重（state_dict）
          - meta.json        : 超参数 + 特征列名 + encoder
        """
        import torch, pickle
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        # 权重
        torch.save(self._net.state_dict(), str(p / "ple_weights.pt"))

        # 元信息
        meta = {
            "hp":          self.hp.to_dict(),
            "task":        self.task,
            "name":        self.name,
            "seed":        self.seed,
            "cat_cols":    self._cat_cols,
            "num_cols":    self._num_cols,
            "input_dim":   self._input_dim,
        }
        with open(p / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # FeatureEncoder（pickle）
        with open(p / "encoder.pkl", "wb") as f:
            pickle.dump(self._encoder, f)

        logger.info(f"[{self.name}] 模型已保存 → {path}")

    def load(self, path: str) -> "PLEModel":
        """从目录 path/ 加载模型。"""
        import torch, pickle
        p = Path(path)

        with open(p / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        self.hp          = PLEParams.from_dict(meta["hp"])
        self.task        = meta["task"]
        self.name        = meta["name"]
        self.seed        = meta["seed"]
        self._cat_cols   = meta["cat_cols"]
        self._num_cols   = meta["num_cols"]
        self._input_dim  = meta["input_dim"]

        with open(p / "encoder.pkl", "rb") as f:
            self._encoder = pickle.load(f)

        device       = self._resolve_device()
        self._device_str = str(device)
        self._net    = self._build_network().to(device)
        state_dict   = torch.load(str(p / "ple_weights.pt"), map_location=device)
        self._net.load_state_dict(state_dict)
        self._net.eval()

        logger.info(f"[{self.name}] 模型已加载 ← {path}")
        return self

    # ------------------------------------------------------------------
    # 特征重要性（Embedding 范数归因）
    # ------------------------------------------------------------------

    @property
    def feature_importance(self) -> pd.DataFrame:
        """
        特征重要性基于 Embedding L2 范数均值：
          importance(f) = mean(||emb_f||₂) / Σ mean(||emb_k||₂)
        数值特征使用第一层专家权重的均值绝对值作为代理。

        注意：这是对深度模型贡献度的近似估计，适合快速排查无效特征。
        """
        if self._feature_importance_cache is not None:
            return self._feature_importance_cache

        if self._net is None:
            return pd.DataFrame(columns=["feature", "importance_mean"])

        import torch
        records = []

        # 类别特征 → Embedding 范数
        for col, emb_module in self._net.embeddings.items():
            weight = emb_module.weight.data     # (vocab_size, emb_dim)
            # 去掉 padding（index=0）
            norms  = weight[1:].norm(dim=1).mean().item()
            records.append({"feature": col, "importance_mean": norms})

        # 数值特征 → 第一层专家 FC 层权重均值绝对值
        if self._num_cols and hasattr(self._net, "ple"):
            offset  = len(self._cat_cols) * self.hp.embedding_dim
            try:
                # 取第一层第一个共享专家的首层权重（近似）
                first_expert = self._net.ple.shared[0][0]
                fc_weight    = list(first_expert.parameters())[0].data
                # fc_weight shape: (hidden_dim, input_dim)
                # 数值列的权重切片
                num_weights = fc_weight[:, offset:offset + len(self._num_cols)]
                for i, col in enumerate(self._num_cols):
                    imp = num_weights[:, i].abs().mean().item()
                    records.append({"feature": col, "importance_mean": imp})
            except (IndexError, StopIteration):
                for col in self._num_cols:
                    records.append({"feature": col, "importance_mean": 0.0})

        if not records:
            return pd.DataFrame(columns=["feature", "importance_mean"])

        fi = pd.DataFrame(records)

        # 归一化到 [0, 1]
        total = fi["importance_mean"].sum()
        if total > 0:
            fi["importance_mean"] = fi["importance_mean"] / total

        fi = fi.sort_values("importance_mean", ascending=False).reset_index(drop=True)
        self._feature_importance_cache = fi
        return fi

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        import random
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _seed_key() -> str:
        return "seed"

    # ------------------------------------------------------------------
    # Optuna 超参数搜索空间
    # ------------------------------------------------------------------

    @staticmethod
    def suggest_params(trial) -> dict:
        """
        Optuna 搜索空间。
        涵盖 PLE 专属参数（专家数、层数）+ 通用训练参数。
        """
        n_levels   = trial.suggest_int("n_levels",   1, 3)
        n_specific = trial.suggest_int("n_specific", 1, 4)
        n_shared   = trial.suggest_int("n_shared",   1, 4)

        # 专家 MLP 维度
        expert_dim = trial.suggest_categorical(
            "expert_dim", [32, 64, 128, 256]
        )
        n_expert_layers = trial.suggest_int("n_expert_layers", 1, 3)
        expert_units    = tuple([expert_dim] * n_expert_layers)

        # 任务塔维度
        tower_dim     = trial.suggest_categorical("tower_dim", [32, 64, 128])
        n_tower_layers = trial.suggest_int("n_tower_layers", 1, 2)
        tower_units    = tuple([tower_dim] * n_tower_layers)

        return {
            "n_levels":          n_levels,
            "n_specific":        n_specific,
            "n_shared":          n_shared,
            "expert_units":      list(expert_units),
            "tower_units":       list(tower_units),
            "expert_dropout":    trial.suggest_float("expert_dropout", 0.0, 0.3),
            "tower_dropout":     trial.suggest_float("tower_dropout",  0.0, 0.3),
            "embedding_dim":     trial.suggest_categorical(
                                     "embedding_dim", [4, 8, 16, 32]
                                 ),
            "learning_rate":     trial.suggest_float(
                                     "learning_rate", 1e-4, 1e-2, log=True
                                 ),
            "ctr_loss_weight":   trial.suggest_float("ctr_loss_weight",  0.3, 2.0),
            "cvr_loss_weight":   trial.suggest_float("cvr_loss_weight",  0.3, 2.0),
            "l2_reg_embedding":  trial.suggest_float(
                                     "l2_reg_embedding", 1e-7, 1e-4, log=True
                                 ),
        }

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PLEModel(name={self.name!r}, task={self.task!r}, "
            f"n_levels={self.hp.n_levels}, "
            f"n_specific={self.hp.n_specific}, n_shared={self.hp.n_shared}, "
            f"expert_units={self.hp.expert_units})"
        )
