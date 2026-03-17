"""
models/deep/bst_wrapper.py
---------------------------
BST (Behavior Sequence Transformer) — 行为序列 Transformer 推荐模型。

论文
----
  "Behavior Sequence Transformer for E-commerce Recommendation in Alibaba"
  Qiwei Chen et al., DLP-KDD 2019
  https://arxiv.org/abs/1905.06874

架构核心思想
------------
  DIN 的局限：用 Attention 计算目标物品与历史行为的相关性，
  但**忽略了历史行为内部的时序依赖**（如"先买鞋垫再买跑鞋"的顺序语义）。

  BST 改进：将用户历史序列（含目标物品追加在末尾）
  整体送入 Transformer，利用 Multi-Head Self-Attention
  同时建模以下三类关系：
    ① 每个历史物品与目标物品的相关性  （类似 DIN）
    ② 历史物品之间的协同依赖          （超越 DIN）
    ③ 行为顺序的位置信息              （Positional Encoding）

  额外创新（本实现）：
    Time-Gap Embedding：将相邻行为的时间间隔离散化为桶，
    与物品 Embedding 拼接后投影，让 Transformer 感知行为的"紧迫度"。

完整数据流
----------

  稀疏特征（user/context）
       │  Embedding
       ▼
  ─────────────────────────────────────────────────────────
  行为序列 + 目标物品                时间间隔序列
  [i₁, i₂, ..., iₙ, i_target]    [Δt₁, Δt₂, ..., Δtₙ, 0]
       │  Item Embedding                 │  TimeGap Embedding
       └───────────── concat ────────────┘
                      │  Linear → d_model
                      ▼
              + Positional Encoding (learned / sinusoidal)
                      │
              ┌───────────────┐
              │ Transformer   │ × n_layers
              │  - LayerNorm  │
              │  - MHA + Mask │  ← padding 位置不参与注意力
              │  - Residual   │
              │  - FFN        │
              │  - Residual   │
              └───────────────┘
                      │
              取目标物品位置（末尾）的输出 h_target
                      │
  ─────────────────────────────────────────────────────────
  concat([h_target,  用户/场景稀疏 Embedding,  稠密特征])
                      │
              DNN → sigmoid → pCTR
  ─────────────────────────────────────────────────────────

数据格式约定
------------
  seq_col     : DataFrame 列，每行为 list 或空格分隔字符串，如 [101, 32, 7]
  timegap_col : DataFrame 列，每行为对应的时间间隔（秒），长度 = len(seq)
                末位为 0（代表"目标物品刚发生"）
  target_col  : DataFrame 列，当前候选物品 ID（必须与 seq_col 共享词表）

YAML 快速配置示例
-----------------
  models:
    - name: "bst"
      type: "BSTModel"
      enabled: true
      seq_configs:
        - seq_col:     hist_item_id_list
          target_col:  item_id
          timegap_col: hist_timegap_list   # 可选
          maxlen:      50
      params:
        d_model:              64
        head_num:             4
        n_transformer_layers: 2
        hidden_size:          256
        dropout:              0.1
        dnn_hidden_units:     [256, 128]
        learning_rate:        0.001
        batch_size:           2048
        patience:             5

依赖
----
  pip install torch scikit-learn numpy pandas
  （纯原生 PyTorch，无需 deepctr-torch）
"""

from __future__ import annotations

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
            "  pip install torch --index-url https://download.pytorch.org/whl/cu118"
        )


# ---------------------------------------------------------------------------
# 序列特征配置（与 DINModel 对齐）
# ---------------------------------------------------------------------------

@dataclass
class BSTSeqConfig:
    """
    一组序列特征的完整描述。

    Parameters
    ----------
    seq_col     : 历史行为序列列名（list of item ids）
    target_col  : 当前目标物品列名（与 seq 共享词表和 Embedding）
    timegap_col : 时间间隔序列列名（可选），单位秒，长度须与 seq_col 一致
    maxlen      : 序列截断长度（保留最近 maxlen 个行为）+ 1 个目标位 = maxlen+1
    """
    seq_col:     str
    target_col:  str
    timegap_col: Optional[str] = None
    maxlen:      int = 50


# ---------------------------------------------------------------------------
# 超参数容器
# ---------------------------------------------------------------------------

@dataclass
class BSTParams:
    """
    BST 全量超参数，支持从 dict 初始化（YAML 无缝对接）。

    Transformer 参数说明
    --------------------
    d_model              : 序列 Embedding 映射后的维度（= Transformer 宽度）
    head_num             : Multi-Head Self-Attention 的头数
                           须满足 d_model % head_num == 0
    n_transformer_layers : Transformer Block 堆叠层数（2~4 为最佳实践）
    hidden_size          : Transformer FFN 中间层维度（建议 4 × d_model）
    dropout              : Transformer + DNN 的 Dropout 概率
    pos_encoding         : "learned"（可训练位置嵌入）| "sinusoidal"（固定正弦编码）

    Time-Gap 参数
    ------------
    n_timegap_buckets : 时间间隔离散桶数量
    timegap_dim       : 时间间隔 Embedding 维度
    """

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_dim: int    = 32    # 物品 / 稀疏特征 Embedding 维度

    # ── Transformer ────────────────────────────────────────────────────────
    d_model:              int   = 64      # 序列投影维度（Transformer 宽度）
    head_num:             int   = 4       # 注意力头数（d_model 须整除）
    n_transformer_layers: int   = 2       # Block 堆叠数
    hidden_size:          int   = 256     # FFN 隐层维度
    dropout:              float = 0.1
    pos_encoding:         str   = "learned"  # "learned" | "sinusoidal"
    use_layer_norm_before: bool = True    # Pre-LN（更稳定）vs Post-LN

    # ── Time-Gap ───────────────────────────────────────────────────────────
    n_timegap_buckets: int  = 128   # 时间间隔离散桶数
    timegap_dim:       int  = 16    # 时间间隔 Embedding 维度

    # ── DNN 输出塔 ─────────────────────────────────────────────────────────
    dnn_hidden_units: tuple = (256, 128)
    dnn_activation:   str   = "relu"
    dnn_dropout:      float = 0.1

    # ── 正则化 ─────────────────────────────────────────────────────────────
    l2_reg_embedding: float = 1e-6
    weight_decay:     float = 0.0

    # ── 优化器 ─────────────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    optimizer:     str   = "adam"

    # ── 训练控制 ───────────────────────────────────────────────────────────
    batch_size: int = 2048
    epochs:     int = 50
    device:     str = "auto"

    # ── Early Stopping ─────────────────────────────────────────────────────
    patience:     int   = 5
    min_delta:    float = 1e-5
    monitor_mode: str   = "min"

    @classmethod
    def from_dict(cls, d: dict) -> "BSTParams":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        obj   = cls(**valid)
        if isinstance(obj.dnn_hidden_units, list):
            obj.dnn_hidden_units = tuple(obj.dnn_hidden_units)
        return obj

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Early Stopping（与 ple_wrapper 对齐）
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
# 特征编码器（与 ple_wrapper 完全一致）
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
            self._vocabulary_sizes[col] = len(mapping) + 1
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
# 序列编码器（BST 专属）
# ---------------------------------------------------------------------------

class BSTSeqEncoder:
    """
    负责：
      1. 解析 seq_col（list 或字符串 → list of int）
      2. LabelEncoding：序列物品 & 目标物品共享同一编码器
      3. Padding 序列到固定长度，生成 attention mask
      4. 时间间隔 → 对数均匀桶（log-uniform buckets）
      5. 将目标物品追加到序列末尾（BST 核心：target 作为序列最后一个 token）
    """

    # 时间桶边界（秒）：从 0 到 ~1 年，对数均匀分布
    _TIMEGAP_BOUNDARIES = np.array([
        0, 60, 300, 900, 1800, 3600,        # 0~1h
        10800, 21600, 43200, 86400,          # 1h~1d
        172800, 432000, 604800,              # 1d~1w
        1209600, 2592000, 5184000,           # 1w~2m
        7776000, 15552000, 31536000,         # 2m~1y
        float("inf"),
    ], dtype=np.float64)

    def __init__(self, config: BSTSeqConfig, n_timegap_buckets: int = 128):
        self.config            = config
        self.n_timegap_buckets = n_timegap_buckets

        self._item_encoder = FeatureEncoder()   # seq + target 共享词表
        self._vocab_size: int = 0

    def fit(self, X: pd.DataFrame) -> "BSTSeqEncoder":
        """从训练集 fit 物品词表（同时涵盖历史序列和目标物品）。"""
        # 收集所有物品 ID（序列 + 目标）
        all_items = []
        for row_val in X[self.config.seq_col].dropna():
            all_items.extend(self._parse_seq(row_val))
        for item in X[self.config.target_col].dropna():
            all_items.append(str(item))

        uniques = sorted(set(map(str, all_items)))
        mapping = {v: i + 1 for i, v in enumerate(uniques)}  # 0 = padding
        self._item_encoder._encoders[self.config.seq_col]         = mapping
        self._item_encoder._vocabulary_sizes[self.config.seq_col] = len(mapping) + 1
        self._vocab_size = len(mapping) + 1
        return self

    def transform(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        seq_encoded  : (N, maxlen + 1)  LongArray  — padded seq + target at end
        timegap_enc  : (N, maxlen + 1)  LongArray  — bucketed timegaps
        mask         : (N, maxlen + 1)  BoolArray  — True = padding (忽略位置)
        """
        mapping = self._item_encoder._encoders[self.config.seq_col]
        maxlen  = self.config.maxlen
        n       = len(X)
        total   = maxlen + 1   # maxlen 历史 + 1 目标

        seq_out  = np.zeros((n, total), dtype=np.int64)   # 0 = PAD
        tg_out   = np.zeros((n, total), dtype=np.int64)
        mask     = np.ones((n, total), dtype=bool)         # True = PAD

        for idx, (_, row) in enumerate(X.iterrows()):
            # ── 解析序列 ──────────────────────────────────────────
            raw_seq = row.get(self.config.seq_col, None)
            seq     = self._parse_seq(raw_seq) if raw_seq is not None else []
            seq     = seq[-maxlen:]             # 保留最近 maxlen 个行为
            seq_enc = [mapping.get(str(s), 0) for s in seq]

            # ── 解析目标物品 ────────────────────────────────────────
            target     = row[self.config.target_col]
            target_enc = mapping.get(str(target), 0)

            # ── 右填充 + 目标追加在末尾 ─────────────────────────────
            start  = maxlen - len(seq_enc)  # 左侧 padding 起点
            for j, v in enumerate(seq_enc):
                seq_out[idx, start + j] = v
                mask[idx, start + j]    = False
            seq_out[idx, -1] = target_enc
            mask[idx, -1]    = False            # 目标位永远不 mask

            # ── 解析时间间隔 ────────────────────────────────────────
            if self.config.timegap_col and self.config.timegap_col in X.columns:
                raw_tg = row.get(self.config.timegap_col, None)
                tg_list = self._parse_seq(raw_tg, as_float=True) if raw_tg else []
                tg_list = tg_list[-maxlen:]     # 与序列等长
                for j, tg in enumerate(tg_list):
                    tg_out[idx, start + j] = self._bucket_timegap(float(tg))
                # 目标物品的时间间隔设为 0（最近一次行为与目标之间）
                tg_out[idx, -1] = 0

        return seq_out, tg_out, mask

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_seq(value, as_float: bool = False) -> list:
        """将 list / 字符串 / numpy array 解析为 Python list。"""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return []
        if isinstance(value, (list, np.ndarray)):
            return [float(v) if as_float else v for v in value]
        if isinstance(value, str):
            parts = value.strip().split()
            return [float(p) if as_float else p for p in parts if p]
        return [float(value) if as_float else value]

    def _bucket_timegap(self, seconds: float) -> int:
        """将时间间隔（秒）映射到 [1, n_timegap_buckets] 中的整数桶 ID。"""
        # 先用预定义边界找大类，再线性细分到 n_buckets
        idx = np.searchsorted(self._TIMEGAP_BOUNDARIES, seconds, side="right") - 1
        idx = int(np.clip(idx, 0, len(self._TIMEGAP_BOUNDARIES) - 2))
        # 归一化到 [0, n_timegap_buckets-1]
        bucket = min(
            int(idx / (len(self._TIMEGAP_BOUNDARIES) - 1) * self.n_timegap_buckets),
            self.n_timegap_buckets - 1,
        )
        return bucket + 1   # +1 保留 0 为 padding


# ---------------------------------------------------------------------------
# Transformer 网络组件（纯 PyTorch）
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding:
    """
    标准正弦位置编码（固定，不可训练）。
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    @staticmethod
    def build(max_len: int, d_model: int):
        import torch
        import torch.nn as nn

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])

        class _SinPE(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

            def forward(self, x):
                return x + self.pe[:, :x.size(1), :]

        return _SinPE()


class LearnedPositionalEncoding:
    """可训练位置 Embedding（效果通常优于正弦编码，适合推荐场景）。"""
    @staticmethod
    def build(max_len: int, d_model: int):
        import torch.nn as nn

        class _LearnedPE(nn.Module):
            def __init__(self):
                super().__init__()
                self.pe = nn.Embedding(max_len, d_model)

            def forward(self, x):
                import torch
                positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
                return x + self.pe(positions)

        return _LearnedPE()


class TransformerBlock:
    """
    单个 Transformer Block。
    支持 Pre-LN（默认，更稳定）和 Post-LN 两种结构。

    Pre-LN:
      x → LN → MHA → + x → LN → FFN → + x
    Post-LN (原 Transformer):
      x → MHA → + x → LN → FFN → + x → LN
    """
    @staticmethod
    def build(
        d_model:    int,
        head_num:   int,
        hidden_size: int,
        dropout:    float,
        pre_ln:     bool = True,
    ):
        import torch.nn as nn

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn    = nn.MultiheadAttention(
                    embed_dim   = d_model,
                    num_heads   = head_num,
                    dropout     = dropout,
                    batch_first = True,   # (B, S, d_model)
                )
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, d_model),
                    nn.Dropout(dropout),
                )
                self.ln1    = nn.LayerNorm(d_model)
                self.ln2    = nn.LayerNorm(d_model)
                self.pre_ln = pre_ln

            def forward(self, x, key_padding_mask=None):
                """
                x                : (B, S, d_model)
                key_padding_mask : (B, S)  True = 忽略该位置
                """
                if self.pre_ln:
                    # ── Pre-LN MHA ────────────────────────────────────
                    residual = x
                    x = self.ln1(x)
                    x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
                    x = x + residual
                    # ── Pre-LN FFN ────────────────────────────────────
                    residual = x
                    x = self.ln2(x)
                    x = self.ffn(x)
                    x = x + residual
                else:
                    # ── Post-LN MHA ───────────────────────────────────
                    residual = x
                    x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
                    x = self.ln1(x + residual)
                    # ── Post-LN FFN ───────────────────────────────────
                    residual = x
                    x = self.ffn(x)
                    x = self.ln2(x + residual)
                return x   # (B, S, d_model)

        return _Block()


# ---------------------------------------------------------------------------
# BST 完整 PyTorch 网络
# ---------------------------------------------------------------------------

def _build_bst_network(
    hp:              BSTParams,
    cat_vocab_sizes: dict[str, int],   # 非序列稀疏列的词表大小
    cat_cols:        list[str],         # 非序列稀疏列名
    num_cols:        list[str],         # 数值列名
    seq_configs:     list[BSTSeqConfig],
    seq_encoders:    list[BSTSeqEncoder],
):
    """工厂函数：返回完整 BST nn.Module。"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class BSTNetwork(nn.Module):

        def __init__(self):
            super().__init__()
            hp_ref    = hp
            total_len = hp_ref.d_model  # Transformer 输入维度

            # ── 非序列稀疏特征 Embedding ──────────────────────────────
            self.cat_embeddings = nn.ModuleDict({
                col: nn.Embedding(vocab_size, hp_ref.embedding_dim, padding_idx=0)
                for col, vocab_size in cat_vocab_sizes.items()
            })

            # ── 序列 Embedding（每组序列独立）────────────────────────
            self.seq_item_embeddings = nn.ModuleList([
                nn.Embedding(enc._vocab_size, hp_ref.embedding_dim, padding_idx=0)
                for enc in seq_encoders
            ])

            # ── 时间间隔 Embedding（每组序列独立）────────────────────
            self.timegap_embeddings = nn.ModuleList([
                nn.Embedding(
                    hp_ref.n_timegap_buckets + 1,
                    hp_ref.timegap_dim,
                    padding_idx=0,
                )
                if sc.timegap_col else None
                for sc in seq_configs
            ])

            # ── 序列投影：(item_emb [+ timegap_emb]) → d_model ───────
            has_timegap = [sc.timegap_col is not None for sc in seq_configs]
            seq_in_dims = [
                hp_ref.embedding_dim + (hp_ref.timegap_dim if ht else 0)
                for ht in has_timegap
            ]
            self.seq_projections = nn.ModuleList([
                nn.Linear(in_d, hp_ref.d_model)
                for in_d in seq_in_dims
            ])

            # ── 位置编码 ─────────────────────────────────────────────
            max_total = max(sc.maxlen for sc in seq_configs) + 1
            if hp_ref.pos_encoding == "sinusoidal":
                self.positional_encoding = nn.ModuleList([
                    SinusoidalPositionalEncoding.build(max_total, hp_ref.d_model)
                    for _ in seq_configs
                ])
            else:
                self.positional_encoding = nn.ModuleList([
                    LearnedPositionalEncoding.build(max_total, hp_ref.d_model)
                    for _ in seq_configs
                ])

            # ── Transformer Blocks（每组序列共享一套 Blocks）─────────
            self.transformer_layers = nn.ModuleList([
                TransformerBlock.build(
                    d_model     = hp_ref.d_model,
                    head_num    = hp_ref.head_num,
                    hidden_size = hp_ref.hidden_size,
                    dropout     = hp_ref.dropout,
                    pre_ln      = hp_ref.use_layer_norm_before,
                )
                for _ in range(hp_ref.n_transformer_layers)
            ])

            # ── 计算 DNN 输入维度 ─────────────────────────────────────
            # 每组序列贡献 d_model（目标位置 h_target）
            seq_out_dim = len(seq_configs) * hp_ref.d_model
            # 非序列稀疏特征
            cat_out_dim = len(cat_cols) * hp_ref.embedding_dim
            # 数值特征
            num_out_dim = len(num_cols)
            dnn_in_dim  = seq_out_dim + cat_out_dim + num_out_dim

            # ── DNN 输出塔 ────────────────────────────────────────────
            dnn_layers: list = []
            prev_dim = dnn_in_dim
            for h in hp_ref.dnn_hidden_units:
                dnn_layers += [
                    nn.Linear(prev_dim, h),
                    nn.ReLU() if hp_ref.dnn_activation == "relu" else nn.GELU(),
                    nn.Dropout(hp_ref.dnn_dropout),
                ]
                prev_dim = h
            dnn_layers.append(nn.Linear(prev_dim, 1))
            self.dnn = nn.Sequential(*dnn_layers)

            # 存储配置供 forward 使用
            self._seq_configs  = seq_configs
            self._cat_cols     = cat_cols
            self._num_cols     = num_cols
            self._hp           = hp_ref
            self._has_timegap  = has_timegap

        def forward(
            self,
            cat_dict:      dict,                  # {col: (B,) LongTensor}
            x_dense:       Optional[torch.Tensor], # (B, num_dim)
            seq_batch:     list[torch.Tensor],     # [(B, S+1) LongTensor]
            timegap_batch: list[Optional[torch.Tensor]],  # [(B, S+1) LongTensor | None]
            mask_batch:    list[torch.Tensor],     # [(B, S+1) BoolTensor]
        ) -> torch.Tensor:
            """Returns logit tensor of shape (B,)."""

            all_parts = []

            # ── 序列 Transformer 前向 ─────────────────────────────────
            for k, (seq_cfg, seq_enc, item_emb, tg_emb, proj, pos_enc) in enumerate(
                zip(
                    self._seq_configs,
                    [None] * len(self._seq_configs),  # placeholder
                    self.seq_item_embeddings,
                    self.timegap_embeddings,
                    self.seq_projections,
                    self.positional_encoding,
                )
            ):
                seq_ids = seq_batch[k]          # (B, S+1)
                mask    = mask_batch[k]         # (B, S+1)  True = PAD

                # Item Embedding
                x = item_emb(seq_ids)           # (B, S+1, emb_dim)

                # Time-Gap Embedding（拼接）
                if self._has_timegap[k] and tg_emb is not None:
                    tg_ids = timegap_batch[k]   # (B, S+1)
                    tg_x   = tg_emb(tg_ids)     # (B, S+1, tg_dim)
                    x      = torch.cat([x, tg_x], dim=-1)  # (B, S+1, emb+tg)

                # 投影到 d_model
                x = proj(x)                     # (B, S+1, d_model)
                x = F.dropout(x, p=self._hp.dropout, training=self.training)

                # 位置编码
                x = pos_enc(x)                  # (B, S+1, d_model)

                # Transformer Blocks
                for block in self.transformer_layers:
                    x = block(x, key_padding_mask=mask)   # (B, S+1, d_model)

                # 提取目标物品位置（末位）的隐向量
                h_target = x[:, -1, :]          # (B, d_model)
                all_parts.append(h_target)

            # ── 非序列稀疏特征 Embedding ──────────────────────────────
            for col in self._cat_cols:
                emb = self.cat_embeddings[col](cat_dict[col])  # (B, emb_dim)
                all_parts.append(emb)

            # ── 数值特征 ─────────────────────────────────────────────
            if x_dense is not None:
                all_parts.append(x_dense)

            # ── DNN ──────────────────────────────────────────────────
            z      = torch.cat(all_parts, dim=-1)  # (B, dnn_in_dim)
            logit  = self.dnn(z).squeeze(1)         # (B,)
            return logit

    return BSTNetwork()


# ---------------------------------------------------------------------------
# BSTModel：BaseModel 子类
# ---------------------------------------------------------------------------

class BSTModel(BaseModel):
    """
    BST Wrapper — 与 ple_wrapper.py 完全对齐的接口规范。

    核心差异点（相比 PLE）
    ----------------------
    - 需要在 __init__ 中传入 seq_configs（声明序列特征）
    - fit() 从 X_train 中自动提取序列、时间间隔、非序列特征
    - predict_proba() 返回 pCTR，shape=(N,)，与 CrossValidatorTrainer 兼容

    YAML 示例
    ---------
    models:
      - name: "bst"
        type: "BSTModel"
        enabled: true
        seq_configs:
          - seq_col:     hist_item_id_list
            target_col:  item_id
            timegap_col: hist_timegap_list
            maxlen:      50
        params:
          d_model: 64
          head_num: 4
          n_transformer_layers: 2
          hidden_size: 256
          dropout: 0.1
    """

    DEFAULT_PARAMS: dict = {
        "embedding_dim":        32,
        "d_model":              64,
        "head_num":             4,
        "n_transformer_layers": 2,
        "hidden_size":          256,
        "dropout":              0.1,
        "pos_encoding":         "learned",
        "use_layer_norm_before": True,
        "n_timegap_buckets":    128,
        "timegap_dim":          16,
        "dnn_hidden_units":     [256, 128],
        "dnn_activation":       "relu",
        "dnn_dropout":          0.1,
        "learning_rate":        1e-3,
        "batch_size":           2048,
        "epochs":               50,
        "patience":             5,
        "weight_decay":         0.0,
        "optimizer":            "adam",
        "device":               "auto",
    }

    def __init__(
        self,
        params:      Optional[dict]              = None,
        seq_configs: Optional[list]              = None,
        task:        str                         = "binary",
        name:        str                         = "bst",
        seed:        int                         = 42,
    ):
        """
        Parameters
        ----------
        seq_configs : list of BSTSeqConfig 或 list of dict（供 YAML 驱动）
        """
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged, task=task, name=name, seed=seed)

        self.hp: BSTParams = BSTParams.from_dict(merged)

        # 序列配置：dict → BSTSeqConfig
        raw_seqs = seq_configs or []
        self.seq_configs: list[BSTSeqConfig] = [
            BSTSeqConfig(**c) if isinstance(c, dict) else c
            for c in raw_seqs
        ]
        if not self.seq_configs:
            raise ValueError(
                "BSTModel 必须至少声明一组序列特征（seq_configs）。\n"
                "示例：seq_configs=[BSTSeqConfig(seq_col='hist_item', target_col='item_id')]"
            )

        # 运行时属性
        self._net:            Optional[object]    = None
        self._encoder:        FeatureEncoder      = FeatureEncoder()
        self._seq_encoders:   list[BSTSeqEncoder] = []
        self._cat_cols:       list[str]           = []
        self._num_cols:       list[str]           = []
        self._device_str:     Optional[str]       = None
        self._best_weights:   Optional[bytes]     = None
        self._fi_cache:       Optional[pd.DataFrame] = None

        # 动态负采样配置（由 train_pipeline.py 注入，None = 不启用）
        self.negative_sampler_config: Optional[object] = None

        # 序列相关列（排除在普通特征处理之外）
        self._seq_related_cols: set[str] = set()
        for sc in self.seq_configs:
            self._seq_related_cols.update(
                [sc.seq_col, sc.target_col]
                + ([sc.timegap_col] if sc.timegap_col else [])
            )

    # ------------------------------------------------------------------
    # 设备解析
    # ------------------------------------------------------------------

    def _resolve_device(self):
        torch, _ = _require_torch()
        s = self.hp.device
        return torch.device("cuda" if (s == "auto" and torch.cuda.is_available()) else
                            ("cpu" if s == "auto" else s))

    # ------------------------------------------------------------------
    # 特征列分拣（排除序列相关列）
    # ------------------------------------------------------------------

    def _identify_feature_cols(self, X: pd.DataFrame) -> tuple[list, list]:
        cat_cols, num_cols = [], []
        for col in X.columns:
            if col in self._seq_related_cols:
                continue
            if pd.api.types.is_categorical_dtype(X[col]) or X[col].dtype == object:
                cat_cols.append(col)
            elif pd.api.types.is_numeric_dtype(X[col]):
                num_cols.append(col)
        return cat_cols, num_cols

    # ------------------------------------------------------------------
    # Tensor 准备
    # ------------------------------------------------------------------

    def _prepare_batch(self, X_enc: pd.DataFrame, device):
        torch, _ = _require_torch()

        # 非序列稀疏特征
        cat_dict = {
            col: torch.tensor(X_enc[col].values, dtype=torch.long, device=device)
            for col in self._cat_cols
        }

        # 数值特征
        x_dense = None
        if self._num_cols:
            arr     = X_enc[self._num_cols].values.astype(np.float32)
            x_dense = torch.tensor(arr, device=device)

        # 序列特征
        seq_batch: list     = []
        tg_batch:  list     = []
        mask_batch: list    = []

        for enc in self._seq_encoders:
            seq_arr, tg_arr, mask_arr = enc.transform(X_enc)
            seq_batch.append(torch.tensor(seq_arr, dtype=torch.long, device=device))
            tg_batch.append(torch.tensor(tg_arr, dtype=torch.long, device=device))
            mask_batch.append(torch.tensor(mask_arr, dtype=torch.bool, device=device))

        return cat_dict, x_dense, seq_batch, tg_batch, mask_batch

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train:  pd.DataFrame,
        y_train:  Optional[pd.Series] = None,
        X_val:    Optional[pd.DataFrame] = None,
        y_val:    Optional[pd.Series]    = None,
        **kwargs,
    ) -> "BSTModel":
        torch, nn = _require_torch()
        import torch

        t0     = time.perf_counter()
        device = self._resolve_device()
        self._device_str = str(device)
        self._set_seed(self.seed)

        y_tr = (
            y_train.values.astype(np.float32) if y_train is not None
            else X_train.get("label", pd.Series(np.zeros(len(X_train)))).values.astype(np.float32)
        )
        has_val = X_val is not None and y_val is not None
        if has_val:
            y_va = y_val.values.astype(np.float32)

        # ── 特征分拣 ──────────────────────────────────────────────────
        self._cat_cols, self._num_cols = self._identify_feature_cols(X_train)
        self._encoder.fit(X_train, self._cat_cols)

        # ── 序列编码器 fit ─────────────────────────────────────────────
        self._seq_encoders = []
        for sc in self.seq_configs:
            enc = BSTSeqEncoder(sc, n_timegap_buckets=self.hp.n_timegap_buckets)
            enc.fit(X_train)
            self._seq_encoders.append(enc)

        logger.info(
            f"[{self.name}] 稀疏特征={len(self._cat_cols)} | "
            f"数值特征={len(self._num_cols)} | "
            f"序列组数={len(self.seq_configs)} | 设备={device}"
        )

        # ── 构建网络 ───────────────────────────────────────────────────
        cat_vocab_sizes = {
            col: self._encoder.vocabulary_sizes[col]
            for col in self._cat_cols
        }
        self._net = _build_bst_network(
            hp              = self.hp,
            cat_vocab_sizes = cat_vocab_sizes,
            cat_cols        = self._cat_cols,
            num_cols        = self._num_cols,
            seq_configs     = self.seq_configs,
            seq_encoders    = self._seq_encoders,
        ).to(device)

        n_params = sum(p.numel() for p in self._net.parameters() if p.requires_grad)
        logger.info(f"[{self.name}] 可训练参数: {n_params:,}")

        # ── 优化器 ────────────────────────────────────────────────────
        opt_map = {
            "adam":    torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
            "rmsprop": torch.optim.RMSprop,
        }
        optimizer = opt_map.get(self.hp.optimizer, torch.optim.Adam)(
            self._net.parameters(),
            lr           = self.hp.learning_rate,
            weight_decay = self.hp.weight_decay,
        )
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # ── Early Stopping ─────────────────────────────────────────────
        stopper = EarlyStopper(
            patience  = self.hp.patience,
            min_delta = self.hp.min_delta,
            mode      = "min",
        )

        # ── 预编码（避免每 epoch 重复编码）────────────────────────────
        X_enc_tr = self._encoder.transform(X_train)
        if has_val:
            X_enc_va = self._encoder.transform(X_val)

        n_train = len(X_train)
        bs      = self.hp.batch_size

        # ── 动态负采样器（可选）──────────────────────────────────────
        _sampler = None
        ns_cfg = self.negative_sampler_config
        if ns_cfg is not None and getattr(ns_cfg, "enable", False):
            from data.negative_sampler import DynamicNegativeSampler
            _sampler = DynamicNegativeSampler(ns_cfg)
            _sampler.fit(X_train, y_tr)
            _sampler.start()
            logger.info(
                f"[{self.name}] 动态负采样器启动 | "
                f"pos={_sampler.n_positives:,} | neg_pool={_sampler.n_negatives:,} | "
                f"neg_ratio=1:{ns_cfg.neg_ratio}"
            )

        # ── 训练循环 ───────────────────────────────────────────────────
        for epoch in range(1, self.hp.epochs + 1):
            # 每 Epoch 取新采样数据（负采样启用时 blocking pull from buffer）
            if _sampler is not None:
                X_ep_raw, y_ep_arr = _sampler.get_next_epoch()
                y_ep     = y_ep_arr.astype(np.float32)
                X_enc_ep = self._encoder.transform(X_ep_raw)
                n_ep     = len(X_ep_raw)
            else:
                X_enc_ep = X_enc_tr
                y_ep     = y_tr
                n_ep     = n_train
            self._net.train()
            perm     = np.random.permutation(n_ep)
            ep_loss  = 0.0
            n_batch  = 0

            for start in range(0, n_ep, bs):
                idx     = perm[start:start + bs]
                X_b     = X_enc_ep.iloc[idx]
                y_b     = torch.tensor(y_ep[idx], device=device)

                cat_d, x_dense, seq_b, tg_b, mask_b = self._prepare_batch(X_b, device)
                optimizer.zero_grad()
                logit = self._net(cat_d, x_dense, seq_b, tg_b, mask_b)
                loss  = loss_fn(logit, y_b)
                loss.backward()
                optimizer.step()

                ep_loss += loss.item()
                n_batch += 1

            ep_loss /= n_batch

            # ── 验证 ──────────────────────────────────────────────────
            if has_val:
                self._net.eval()
                val_loss = 0.0
                n_vb     = 0
                with torch.no_grad():
                    for vstart in range(0, len(X_val), bs):
                        X_vb  = X_enc_va.iloc[vstart:vstart + bs]
                        y_vb  = torch.tensor(y_va[vstart:vstart + bs], device=device)
                        c, d, s, tg, mk = self._prepare_batch(X_vb, device)
                        vlogit = self._net(c, d, s, tg, mk)
                        val_loss += loss_fn(vlogit, y_vb).item()
                        n_vb += 1
                val_loss /= n_vb

                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"[{self.name}] Epoch {epoch:3d}/{self.hp.epochs} | "
                        f"train={ep_loss:.5f} | val={val_loss:.5f}"
                    )

                # Checkpoint：保存最优权重
                if stopper._counter == 0:
                    buf = io.BytesIO()
                    torch.save(self._net.state_dict(), buf)
                    self._best_weights = buf.getvalue()

                if stopper.step(val_loss, epoch):
                    logger.info(
                        f"[{self.name}] Early Stopping | "
                        f"最优 epoch={stopper.best_epoch} | "
                        f"val_loss={stopper.best_value:.5f}"
                    )
                    break
            else:
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"[{self.name}] Epoch {epoch:3d}/{self.hp.epochs} | "
                        f"train={ep_loss:.5f}"
                    )

        # ── 恢复最优权重 ───────────────────────────────────────────────
        # 负采样后台线程回收（daemon 线程会随进程退出，此处主动 stop 更干净）
        if _sampler is not None:
            _sampler.stop()

        if self._best_weights is not None:
            buf = io.BytesIO(self._best_weights)
            self._net.load_state_dict(torch.load(buf, map_location=device))
            logger.info(f"[{self.name}] 已恢复最优权重（epoch {stopper.best_epoch}）")

        elapsed = time.perf_counter() - t0
        logger.info(f"[{self.name}] 训练完成 | 耗时 {elapsed:.1f}s")
        return self

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def _predict_logits(self, X: pd.DataFrame) -> np.ndarray:
        torch, _ = _require_torch()
        import torch

        if self._net is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

        self._net.eval()
        device  = torch.device(self._device_str)
        X_enc   = self._encoder.transform(X)
        bs      = self.hp.batch_size
        logits  = []

        with torch.no_grad():
            for start in range(0, len(X), bs):
                X_b = X_enc.iloc[start:start + bs]
                cat_d, x_d, seq_b, tg_b, mask_b = self._prepare_batch(X_b, device)
                logit = self._net(cat_d, x_d, seq_b, tg_b, mask_b)
                logits.append(logit.cpu().numpy())

        return np.concatenate(logits).astype(np.float64)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """返回 pCTR（sigmoid(logit)），shape=(N,)，与 BaseTrainer 兼容。"""
        logits = self._predict_logits(X)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.predict_proba(X, **kwargs)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import torch, pickle
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        torch.save(self._net.state_dict(), str(p / "bst_weights.pt"))

        seq_cfg_dicts = [
            {
                "seq_col":     sc.seq_col,
                "target_col":  sc.target_col,
                "timegap_col": sc.timegap_col,
                "maxlen":      sc.maxlen,
            }
            for sc in self.seq_configs
        ]
        meta = {
            "hp":          self.hp.to_dict(),
            "task":        self.task,
            "name":        self.name,
            "seed":        self.seed,
            "cat_cols":    self._cat_cols,
            "num_cols":    self._num_cols,
            "seq_configs": seq_cfg_dicts,
        }
        with open(p / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with open(p / "encoder.pkl", "wb") as f:
            pickle.dump(self._encoder, f)

        with open(p / "seq_encoders.pkl", "wb") as f:
            pickle.dump(self._seq_encoders, f)

        logger.info(f"[{self.name}] 模型已保存 → {path}")

    def load(self, path: str) -> "BSTModel":
        import torch, pickle
        p = Path(path)

        with open(p / "meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        self.hp          = BSTParams.from_dict(meta["hp"])
        self.task        = meta["task"]
        self.name        = meta["name"]
        self.seed        = meta["seed"]
        self._cat_cols   = meta["cat_cols"]
        self._num_cols   = meta["num_cols"]
        self.seq_configs = [BSTSeqConfig(**c) for c in meta["seq_configs"]]
        self._seq_related_cols = set()
        for sc in self.seq_configs:
            self._seq_related_cols.update(
                [sc.seq_col, sc.target_col]
                + ([sc.timegap_col] if sc.timegap_col else [])
            )

        with open(p / "encoder.pkl", "rb") as f:
            self._encoder = pickle.load(f)
        with open(p / "seq_encoders.pkl", "rb") as f:
            self._seq_encoders = pickle.load(f)

        device = self._resolve_device()
        self._device_str = str(device)

        cat_vocab_sizes = {
            col: self._encoder.vocabulary_sizes[col]
            for col in self._cat_cols
        }
        self._net = _build_bst_network(
            hp              = self.hp,
            cat_vocab_sizes = cat_vocab_sizes,
            cat_cols        = self._cat_cols,
            num_cols        = self._num_cols,
            seq_configs     = self.seq_configs,
            seq_encoders    = self._seq_encoders,
        ).to(device)

        state = torch.load(str(p / "bst_weights.pt"), map_location=device)
        self._net.load_state_dict(state)
        self._net.eval()

        logger.info(f"[{self.name}] 模型已加载 ← {path}")
        return self

    # ------------------------------------------------------------------
    # 特征重要性（注意力权重均值归因）
    # ------------------------------------------------------------------

    @property
    def feature_importance(self) -> pd.DataFrame:
        """
        BST 特征重要性估算：

        - 序列特征：用 Embedding 权重 L2 范数（整体词表）反映表征能力
        - 非序列稀疏特征：同 PLEModel，Embedding L2 范数
        - 数值特征：DNN 第一层权重列均值绝对值（近似代理）

        返回 DataFrame: columns=[feature, importance_mean, importance_type]
        """
        if self._fi_cache is not None:
            return self._fi_cache

        if self._net is None:
            return pd.DataFrame(columns=["feature", "importance_mean"])

        records = []

        # 序列特征 Embedding 范数（取词表中所有词向量的均值范数）
        for i, (sc, emb_module) in enumerate(
            zip(self.seq_configs, self._net.seq_item_embeddings)
        ):
            w = emb_module.weight.data[1:]  # 去掉 padding(0)
            norm_mean = w.norm(dim=1).mean().item()
            records.append({
                "feature":        f"[SEQ] {sc.seq_col} → {sc.target_col}",
                "importance_mean": norm_mean,
                "importance_type": "seq_embedding_norm",
            })
            if self._net.timegap_embeddings[i] is not None:
                tg_w    = self._net.timegap_embeddings[i].weight.data[1:]
                tg_norm = tg_w.norm(dim=1).mean().item()
                records.append({
                    "feature":        f"[TIMEGAP] {sc.timegap_col}",
                    "importance_mean": tg_norm,
                    "importance_type": "timegap_embedding_norm",
                })

        # 非序列稀疏特征 Embedding 范数
        for col, emb_module in self._net.cat_embeddings.items():
            w    = emb_module.weight.data[1:]
            norm = w.norm(dim=1).mean().item()
            records.append({
                "feature":        col,
                "importance_mean": norm,
                "importance_type": "cat_embedding_norm",
            })

        # 数值特征：DNN 第一层权重
        if self._num_cols:
            try:
                # DNN 第一层 Linear
                first_linear = [m for m in self._net.dnn.modules()
                                 if hasattr(m, 'weight') and len(m.weight.shape) == 2][0]
                num_start    = len(self.seq_configs) * self.hp.d_model \
                             + len(self._cat_cols) * self.hp.embedding_dim
                for i, col in enumerate(self._num_cols):
                    w = first_linear.weight[:, num_start + i]
                    records.append({
                        "feature":        col,
                        "importance_mean": w.abs().mean().item(),
                        "importance_type": "dnn_weight_proxy",
                    })
            except (IndexError, AttributeError):
                for col in self._num_cols:
                    records.append({
                        "feature": col,
                        "importance_mean": 0.0,
                        "importance_type": "dnn_weight_proxy",
                    })

        if not records:
            return pd.DataFrame(columns=["feature", "importance_mean"])

        fi = pd.DataFrame(records)
        total = fi["importance_mean"].sum()
        if total > 0:
            fi["importance_mean"] = fi["importance_mean"] / total
        fi = fi.sort_values("importance_mean", ascending=False).reset_index(drop=True)

        self._fi_cache = fi
        return fi

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        import random, torch
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
        涵盖 Transformer 三大核心超参 + 通用训练参数。
        """
        d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])
        # head_num 须整除 d_model
        valid_heads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
        head_num    = trial.suggest_categorical("head_num", valid_heads)

        return {
            "d_model":              d_model,
            "head_num":             head_num,
            "n_transformer_layers": trial.suggest_int(
                                        "n_transformer_layers", 1, 4
                                    ),
            "hidden_size":          trial.suggest_categorical(
                                        "hidden_size", [64, 128, 256, 512]
                                    ),
            "dropout":              trial.suggest_float("dropout", 0.0, 0.4),
            "embedding_dim":        trial.suggest_categorical(
                                        "embedding_dim", [8, 16, 32, 64]
                                    ),
            "timegap_dim":          trial.suggest_categorical(
                                        "timegap_dim", [8, 16, 32]
                                    ),
            "pos_encoding":         trial.suggest_categorical(
                                        "pos_encoding", ["learned", "sinusoidal"]
                                    ),
            "learning_rate":        trial.suggest_float(
                                        "learning_rate", 1e-4, 5e-3, log=True
                                    ),
            "dnn_hidden_units":     trial.suggest_categorical(
                                        "dnn_hidden_units",
                                        [[256, 128], [512, 256], [256, 128, 64]],
                                    ),
            "l2_reg_embedding":     trial.suggest_float(
                                        "l2_reg_embedding", 1e-7, 1e-4, log=True
                                    ),
        }

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        seq_names = [f"{sc.seq_col}→{sc.target_col}" for sc in self.seq_configs]
        return (
            f"BSTModel(name={self.name!r}, "
            f"d_model={self.hp.d_model}, "
            f"heads={self.hp.head_num}, "
            f"layers={self.hp.n_transformer_layers}, "
            f"seqs={seq_names})"
        )
