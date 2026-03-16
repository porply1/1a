"""
models/deep/din_wrapper.py
---------------------------
生产级 DIN (Deep Interest Network) Wrapper，基于 DeepCTR-Torch。

论文参考
--------
  《Deep Interest Network for Click-Through Rate Prediction》
  Zhou et al., KDD 2018 — https://arxiv.org/abs/1706.06978

核心思想
--------
  普通 CTR 模型对用户历史行为序列做 sum/mean pooling → 丢失序列内部的差异信息。
  DIN 针对每条历史行为与「目标商品」计算 attention score，
  用加权 sum 生成「当前候选物品相关的兴趣向量」。

  用户历史序列  [e_1, e_2, ..., e_k]
                     │
               Activation Unit
               (target_emb 作为 query)
                     │
              [a_1, a_2, ..., a_k]   ← attention weights
                     │
              Σ a_i * e_i            ← 用户兴趣表示

依赖
----
  pip install deepctr-torch torch

与 DeepFMModel 的设计对称性
---------------------------
- 相同的 FeatureEncoder / EarlyStopper / _to_tensor_dataset（直接 import 复用）
- 相同的 fit/predict_proba/save/load/feature_importance 接口
- 新增：VarLenSparseFeat 列的自动识别 + Padding/Masking 处理
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from core.base_model import BaseModel

# 复用 DeepFM 中已经写好的基础组件，保持 DRY 原则
from models.deep.deepfm_wrapper import (
    EarlyStopper,
    FeatureEncoder,
    _require_deepctr,
    _require_torch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DIN 专用：VarLenSparseFeat 守卫
# ---------------------------------------------------------------------------

def _require_varlen():
    """加载 DeepCTR-Torch 的变长序列 / 注意力相关类。"""
    try:
        from deepctr_torch.models import DIN
        from deepctr_torch.inputs import (
            SparseFeat,
            DenseFeat,
            VarLenSparseFeat,
            get_feature_names,
        )
        return DIN, SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
    except ImportError:
        raise ImportError(
            "DeepCTR-Torch 未安装或版本不支持 DIN。请执行：\n"
            "  pip install deepctr-torch>=0.9.0"
        )


# ---------------------------------------------------------------------------
# 序列列配置：描述一对 (序列列 ↔ 目标商品列) 的关系
# ---------------------------------------------------------------------------

@dataclass
class SeqFeatureConfig:
    """
    描述一组用户历史行为序列与其对应目标物品列的配置。

    Parameters
    ----------
    seq_col       : DataFrame 中的序列列名（如 "hist_item_id_list"）
    target_col    : 该序列对应的目标商品列名（如 "item_id"）
                    DIN attention 以 target_col 的 embedding 作为 query。
    maxlen        : 序列截断/填充长度
    padding_value : 填充值（0 代表 unknown/pad，与 FeatureEncoder 约定一致）
    """
    seq_col:       str
    target_col:    str
    maxlen:        int  = 50
    padding_value: int  = 0


# ---------------------------------------------------------------------------
# 序列 Padding 工具
# ---------------------------------------------------------------------------

class SequencePadder:
    """
    将变长序列列（list / 空格分隔字符串）转换为填充后的定长 numpy 数组。

    支持格式
    --------
    - Python list:   [101, 32, 7]
    - 字符串:         "101 32 7"
    - numpy array:   array([101, 32, 7])
    - 空值 / NaN:    自动替换为全 0 序列
    """

    def __init__(self, encoder: FeatureEncoder, cfg: SeqFeatureConfig):
        self.cfg     = cfg
        self.encoder = encoder   # 与目标商品列共享同一个词表

    def pad_column(self, series: pd.Series) -> np.ndarray:
        """
        输入：  pd.Series，每个元素是行为序列
        输出：  shape=(N, maxlen) 的 int32 ndarray（后填充 / 前截断）
        """
        maxlen  = self.cfg.maxlen
        pad_val = self.cfg.padding_value
        target  = self.cfg.target_col
        mapping = self.encoder._encoders.get(target, {})  # 共享词表

        result = np.full((len(series), maxlen), pad_val, dtype=np.int32)

        for i, raw in enumerate(series):
            seq = self._parse_raw(raw)
            if not seq:
                continue
            # 前截断保留最近的行为（more recent = more relevant）
            seq = seq[-maxlen:]
            # 编码（未见词 → 0）
            encoded = [mapping.get(str(v), 0) for v in seq]
            result[i, :len(encoded)] = encoded

        return result   # shape: (N, maxlen)

    @staticmethod
    def _parse_raw(raw) -> list:
        """将各种格式的原始序列解析为 Python list。"""
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return []
        if isinstance(raw, (list, np.ndarray)):
            return list(raw)
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return []
            # 支持空格、逗号、竖线三种分隔符
            for sep in (" ", ",", "|"):
                if sep in raw:
                    return [x.strip() for x in raw.split(sep) if x.strip()]
            return [raw]
        return [raw]


# ---------------------------------------------------------------------------
# DIN 超参数容器
# ---------------------------------------------------------------------------

@dataclass
class DINParams:
    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_dim: int       = 8

    # ── DNN ────────────────────────────────────────────────────────────────
    dnn_hidden_units: tuple  = (256, 128)
    dnn_activation:   str    = "relu"
    dnn_dropout:      float  = 0.0
    dnn_use_bn:       bool   = False

    # ── DIN Attention ──────────────────────────────────────────────────────
    att_hidden_size:           tuple = (80, 40)
    att_activation:            str   = "dice"    # 论文推荐；也可用 "relu" / "prelu"
    att_weight_normalization:  bool  = False      # True → softmax, False → sigmoid

    # ── 正则化 ─────────────────────────────────────────────────────────────
    l2_reg_linear:     float = 1e-5
    l2_reg_embedding:  float = 1e-6
    l2_reg_dnn:        float = 0.0

    # ── 优化器 ─────────────────────────────────────────────────────────────
    learning_rate: float     = 1e-3
    optimizer:     str       = "adam"

    # ── 训练控制 ───────────────────────────────────────────────────────────
    batch_size: int          = 4096
    epochs:     int          = 50
    device:     str          = "auto"

    # ── Early Stopping ─────────────────────────────────────────────────────
    patience:     int        = 5
    min_delta:    float      = 1e-5
    monitor:      str        = "val_loss"
    monitor_mode: str        = "min"

    @classmethod
    def from_dict(cls, d: dict) -> "DINParams":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        obj = cls(**valid)
        if isinstance(obj.dnn_hidden_units, list):
            obj.dnn_hidden_units = tuple(obj.dnn_hidden_units)
        if isinstance(obj.att_hidden_size, list):
            obj.att_hidden_size = tuple(obj.att_hidden_size)
        return obj

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# DIN Wrapper 主体
# ---------------------------------------------------------------------------

class DINModel(BaseModel):
    """
    DIN (Deep Interest Network) Wrapper。

    Parameters
    ----------
    params       : 超参数字典，与 DEFAULT_PARAMS 合并。
    task         : "binary" | "regression"
    name         : 模型标识
    seed         : 随机种子
    seq_configs  : 序列特征配置列表，每个元素是一个 dict 或 SeqFeatureConfig。
                   示例：
                     [{"seq_col": "hist_item_id_list",
                       "target_col": "item_id",
                       "maxlen": 50}]

    使用示例
    --------
    >>> model = DINModel(
    ...     task="binary",
    ...     seed=42,
    ...     seq_configs=[{
    ...         "seq_col": "hist_item_id_list",
    ...         "target_col": "item_id",
    ...         "maxlen": 50,
    ...     }],
    ... )
    >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    >>> proba = model.predict_proba(X_test)
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "embedding_dim":           8,
        "dnn_hidden_units":        [256, 128],
        "dnn_activation":          "relu",
        "dnn_dropout":             0.0,
        "dnn_use_bn":              False,
        "att_hidden_size":         [80, 40],
        "att_activation":          "dice",
        "att_weight_normalization": False,
        "l2_reg_linear":           1e-5,
        "l2_reg_embedding":        1e-6,
        "l2_reg_dnn":              0.0,
        "learning_rate":           1e-3,
        "optimizer":               "adam",
        "batch_size":              4096,
        "epochs":                  50,
        "device":                  "auto",
        "patience":                5,
        "min_delta":               1e-5,
        "monitor":                 "val_loss",
        "monitor_mode":            "min",
    }

    def __init__(
        self,
        params:      Optional[dict]                    = None,
        task:        str                               = "binary",
        name:        str                               = "din",
        seed:        int                               = 42,
        seq_configs: Optional[list[Union[dict, SeqFeatureConfig]]] = None,
    ):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged, task=task, name=name, seed=seed)

        self._hp: DINParams = DINParams.from_dict(self.params)

        # 序列配置：统一转为 SeqFeatureConfig
        raw_cfgs = seq_configs or []
        self._seq_configs: list[SeqFeatureConfig] = [
            SeqFeatureConfig(**c) if isinstance(c, dict) else c
            for c in raw_cfgs
        ]

        # 列名集合（方便快速查找）
        self._seq_col_names:    set[str] = {c.seq_col    for c in self._seq_configs}
        self._target_col_names: set[str] = {c.target_col for c in self._seq_configs}

        # 运行时状态
        self._model                                   = None
        self._encoder:   Optional[FeatureEncoder]     = None
        self._padders:   dict[str, SequencePadder]    = {}
        self._sparse_cols:       list[str]            = []
        self._dense_cols:        list[str]            = []
        self._varlen_cols:       list[str]            = []    # seq_col names
        self._linear_feature_columns                  = None
        self._dnn_feature_columns                     = None
        self._feature_names:     list[str]            = []
        self._fi:                pd.DataFrame         = pd.DataFrame()
        self._best_epoch:        int                  = 0
        self._device_str:        str                  = ""

        self.history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # 1. 特征列构建
    # ------------------------------------------------------------------

    def _build_feature_columns(
        self,
        X: pd.DataFrame,
        fit_encoder: bool = True,
    ) -> tuple[list, list]:
        """
        自动分拣三类特征：

        ┌──────────────────────────────────────────────────────────────┐
        │  seq_col (如 hist_item_id_list)  → VarLenSparseFeat          │
        │  category 非序列列              → SparseFeat                 │
        │  数值列                          → DenseFeat                  │
        └──────────────────────────────────────────────────────────────┘

        重要约定
        --------
        - seq_col 与其对应的 target_col（如 item_id）**共享同一个 Embedding**，
          DeepCTR-Torch 通过 VarLenSparseFeat.sparsefeat 实现这一机制。
        - FeatureEncoder 对 target_col 进行 LabelEncoding，
          SequencePadder 复用相同词表编码序列 token。
        """
        (DIN, SparseFeat, DenseFeat,
         VarLenSparseFeat, get_feature_names) = _require_varlen()
        hp = self._hp

        # ── 排除序列列，对剩余列做分类 ────────────────────────────────
        non_seq_cols = [c for c in X.columns if c not in self._seq_col_names]
        self._sparse_cols = [
            c for c in non_seq_cols
            if X[c].dtype.name == "category"
        ]
        self._dense_cols = [
            c for c in non_seq_cols
            if c not in self._sparse_cols
            and np.issubdtype(X[c].dtype, np.number)
        ]
        self._varlen_cols = [cfg.seq_col for cfg in self._seq_configs
                             if cfg.seq_col in X.columns]

        # ── FeatureEncoder：同时编码 SparseFeat + target 列 ──────────
        cols_to_encode = list(self._sparse_cols)
        # target_col 可能已经在 sparse_cols 中（如果被标记了 category dtype）
        # 确保每个 target_col 都纳入编码范围
        for cfg in self._seq_configs:
            if cfg.target_col not in cols_to_encode and cfg.target_col in X.columns:
                cols_to_encode.append(cfg.target_col)

        if fit_encoder:
            self._encoder = FeatureEncoder()
            self._encoder.fit(X, cols_to_encode)

        vocab = self._encoder.vocabulary_sizes

        # ── SparseFeat ─────────────────────────────────────────────────
        sparse_feat_list = [
            SparseFeat(col, vocabulary_size=vocab[col], embedding_dim=hp.embedding_dim)
            for col in self._sparse_cols
        ]

        # ── VarLenSparseFeat（共享 target_col 的 SparseFeat embedding）──
        # DeepCTR-Torch 约定：VarLenSparseFeat.sparsefeat 指向共享的 SparseFeat 定义
        varlen_feat_list = []
        shared_sparse_for_seq = []   # 目标商品对应的 SparseFeat（若不在 sparse_cols 里则补充）

        for cfg in self._seq_configs:
            if cfg.seq_col not in X.columns:
                logger.warning(f"[{self.name}] 序列列 '{cfg.seq_col}' 不存在于 DataFrame，跳过。")
                continue

            target_vocab_size = vocab.get(cfg.target_col, 2)

            # 目标商品 SparseFeat（作为 attention query 的 embedding 来源）
            target_sparse = SparseFeat(
                cfg.target_col,
                vocabulary_size = target_vocab_size,
                embedding_dim   = hp.embedding_dim,
            )
            # 避免重复添加（target_col 可能已经在 sparse_cols 里）
            if cfg.target_col not in self._sparse_cols:
                shared_sparse_for_seq.append(target_sparse)

            # VarLenSparseFeat
            varlen_feat = VarLenSparseFeat(
                sparsefeat  = target_sparse,    # 共享 embedding
                maxlen      = cfg.maxlen,
                combiner    = "mean",           # 默认 mean-pooling；DIN 会用 attention 覆盖
                length_name = None,             # 不使用显式 length 列，依赖 mask
            )
            varlen_feat_list.append(varlen_feat)

            # 初始化 Padder
            self._padders[cfg.seq_col] = SequencePadder(self._encoder, cfg)

        # ── DenseFeat ──────────────────────────────────────────────────
        dense_feat_list = [DenseFeat(col, 1) for col in self._dense_cols]

        # ── 组装最终列表 ────────────────────────────────────────────────
        linear_feature_columns = (
            sparse_feat_list
            + shared_sparse_for_seq
            + varlen_feat_list
            + dense_feat_list
        )
        dnn_feature_columns = linear_feature_columns   # DIN 共享所有列

        return linear_feature_columns, dnn_feature_columns

    # ------------------------------------------------------------------
    # 2. 设备选择
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        torch = _require_torch()
        if self._hp.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._hp.device

    # ------------------------------------------------------------------
    # 3. DataFrame → 模型输入字典
    # ------------------------------------------------------------------

    def _to_model_input(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        将 DataFrame 转为 DeepCTR-Torch 所需的 {feature_name: ndarray} 格式。

        序列列 → 经 SequencePadder 填充后的 (N, maxlen) 数组展开为模型 key
        普通列 → 标准 LabelEncoding（sparse）或直接转 float32（dense）
        """
        (_, SparseFeat, DenseFeat,
         VarLenSparseFeat, get_feature_names) = _require_varlen()

        feature_names = get_feature_names(self._linear_feature_columns)
        self._feature_names = feature_names

        X_enc = self._encoder.transform(X)
        result: dict[str, np.ndarray] = {}

        # ── 普通 Sparse / Dense 列 ────────────────────────────────────
        for name in feature_names:
            if name in self._varlen_cols:
                continue   # 序列列单独处理
            if name in X_enc.columns:
                if name in self._dense_cols:
                    result[name] = X_enc[name].values.astype(np.float32)
                else:
                    result[name] = X_enc[name].values.astype(np.int32)

        # ── 变长序列列 → Padding ──────────────────────────────────────
        for seq_col, padder in self._padders.items():
            if seq_col not in X.columns:
                cfg = padder.cfg
                result[seq_col] = np.zeros(
                    (len(X), cfg.maxlen), dtype=np.int32
                )
                continue

            padded = padder.pad_column(X[seq_col])   # (N, maxlen)
            result[seq_col] = padded

        return result

    # ------------------------------------------------------------------
    # 4. fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val:   Optional[pd.DataFrame] = None,
        y_val:   Optional[pd.Series]    = None,
        **kwargs,
    ) -> "DINModel":
        torch = _require_torch()
        (DIN, SparseFeat, DenseFeat,
         VarLenSparseFeat, get_feature_names) = _require_varlen()
        hp = self._hp

        # ── 随机种子 ────────────────────────────────────────────────────
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # ── 设备 ────────────────────────────────────────────────────────
        self._device_str = self._resolve_device()
        logger.info(f"[{self.name}] 设备: {self._device_str}")

        # ── 序列配置校验 ────────────────────────────────────────────────
        if not self._seq_configs:
            logger.warning(
                f"[{self.name}] 未配置 seq_configs，DIN 将退化为普通 DNN。"
                "请通过构造函数传入序列特征配置。"
            )

        # ── 构建特征列 ──────────────────────────────────────────────────
        lin_cols, dnn_cols = self._build_feature_columns(X_train, fit_encoder=True)
        self._linear_feature_columns = lin_cols
        self._dnn_feature_columns    = dnn_cols

        logger.info(
            f"[{self.name}] SparseFeat={len(self._sparse_cols)}, "
            f"DenseFeat={len(self._dense_cols)}, "
            f"VarLenSparseFeat={len(self._varlen_cols)}"
        )

        # ── 提取 history_feature_list（告诉 DIN 哪些是行为序列）────────
        history_feature_list = [cfg.target_col for cfg in self._seq_configs
                                if cfg.seq_col in X_train.columns]

        # ── 任务映射 ────────────────────────────────────────────────────
        deepctr_task = "binary" if self.task in ("binary", "multiclass") else "regression"

        # ── 构建 DIN 模型 ───────────────────────────────────────────────
        self._model = DIN(
            dnn_feature_columns  = dnn_cols,
            history_feature_list = history_feature_list,
            dnn_use_bn           = hp.dnn_use_bn,
            dnn_hidden_units     = hp.dnn_hidden_units,
            dnn_activation       = hp.dnn_activation,
            att_hidden_size      = hp.att_hidden_size,
            att_activation       = hp.att_activation,
            att_weight_normalization = hp.att_weight_normalization,
            l2_reg_embedding     = hp.l2_reg_embedding,
            l2_reg_dnn           = hp.l2_reg_dnn,
            dnn_dropout          = hp.dnn_dropout,
            init_std             = 0.0001,
            seed                 = self.seed,
            task                 = deepctr_task,
            device               = self._device_str,
        )

        # ── 优化器 & 损失函数 ───────────────────────────────────────────
        opt_map = {
            "adam":    torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
            "rmsprop": torch.optim.RMSprop,
            "sgd":     torch.optim.SGD,
        }
        optimizer = opt_map.get(hp.optimizer.lower(), torch.optim.Adam)(
            self._model.parameters(), lr=hp.learning_rate
        )
        loss_fn = (
            torch.nn.BCELoss() if deepctr_task == "binary"
            else torch.nn.MSELoss()
        )

        # ── 准备 DataLoader ─────────────────────────────────────────────
        train_input = self._to_model_input(X_train)
        train_loader = _make_loader(
            train_input, y_train.values.astype(np.float32),
            self._feature_names, hp.batch_size, shuffle=True
        )

        has_val     = X_val is not None and y_val is not None
        val_loader  = None
        if has_val:
            val_input  = self._to_model_input(X_val)
            val_loader = _make_loader(
                val_input, y_val.values.astype(np.float32),
                self._feature_names, hp.batch_size * 4, shuffle=False
            )

        # ── Early Stopping + Checkpoint ─────────────────────────────────
        stopper    = EarlyStopper(hp.patience, hp.min_delta, hp.monitor_mode)
        best_state = copy.deepcopy(self._model.state_dict())

        self.history = {"train_loss": [], "val_loss": []}
        t_start = time.perf_counter()

        for epoch in range(1, hp.epochs + 1):
            train_loss = _train_epoch(
                self._model, train_loader, loss_fn, optimizer,
                deepctr_task, self._device_str
            )
            self.history["train_loss"].append(train_loss)

            val_loss = (
                _eval_epoch(self._model, val_loader, loss_fn, deepctr_task, self._device_str)
                if val_loader else float("nan")
            )
            self.history["val_loss"].append(val_loss)

            monitor_val = val_loss if has_val else train_loss
            should_stop = stopper.step(monitor_val, epoch)

            if stopper._counter == 0:   # 刚改善
                best_state      = copy.deepcopy(self._model.state_dict())
                self._best_epoch = epoch

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"[{self.name}] Epoch {epoch:3d}/{hp.epochs} | "
                    f"train_loss={train_loss:.5f}"
                    + (f" | val_loss={val_loss:.5f}" if has_val else "")
                    + (" ★" if stopper._counter == 0 else "")
                )

            if should_stop:
                logger.info(
                    f"[{self.name}] Early stopping @ epoch {epoch} | "
                    f"best={self._best_epoch} | {hp.monitor}={stopper.best_value:.5f}"
                )
                break

        # ── 恢复最优权重 ────────────────────────────────────────────────
        self._model.load_state_dict(best_state)
        elapsed = time.perf_counter() - t_start
        logger.info(
            f"[{self.name}] 训练完成 | best epoch={self._best_epoch} | 耗时={elapsed:.1f}s"
        )

        self._fi = self._compute_feature_importance()
        return self

    # ------------------------------------------------------------------
    # 5. predict / predict_proba
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.predict_proba(X, **kwargs)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        输出正例概率（或回归值）。
        与 CrossValidatorTrainer / StackingEnsemble 完全兼容。
        """
        torch = _require_torch()
        if self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

        model_input = self._to_model_input(X)
        loader = _make_loader(
            model_input,
            np.zeros(len(X), dtype=np.float32),
            self._feature_names,
            self._hp.batch_size * 4,
            shuffle=False,
        )

        preds = []
        self._model.eval()
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = {k: v.to(self._device_str) for k, v in x_batch.items()}
                out = self._model(x_batch).squeeze(-1)
                if self.task == "binary":
                    out = torch.sigmoid(out)
                preds.append(out.cpu().numpy())

        return np.concatenate(preds, axis=0).astype(np.float64)

    # ------------------------------------------------------------------
    # 6. 特征重要性
    # ------------------------------------------------------------------

    def _compute_feature_importance(self) -> pd.DataFrame:
        """
        特征重要性来源：
        - SparseFeat  → Embedding 矩阵 L2 范数均值
        - VarLenSparse→ 序列 token Embedding L2 范数均值（体现序列整体重要性）
        - DenseFeat   → 线性层权重绝对值均值
        """
        if self._model is None:
            return pd.DataFrame()

        records: list[dict] = []
        state = self._model.state_dict()

        # ── Sparse Embedding 范数 ────────────────────────────────────────
        all_emb_cols = list(self._sparse_cols) + [
            cfg.target_col for cfg in self._seq_configs
        ]
        for col in set(all_emb_cols):
            key = f"embedding_dict.{col}.weight"
            if key in state:
                norm = state[key].float().norm(dim=1).mean().item()
                feat_type = "sequence_target" if col in self._target_col_names else "sparse"
                records.append({"feature": col, "importance": norm, "type": feat_type})

        # ── VarLen 序列列（用 target_col 范数代表序列重要性）─────────────
        for cfg in self._seq_configs:
            records.append({
                "feature":   f"{cfg.seq_col} (seq→{cfg.target_col})",
                "importance": next(
                    (r["importance"] for r in records if r["feature"] == cfg.target_col), 0.0
                ),
                "type": "varlen_sequence",
            })

        # ── Dense 线性权重 ───────────────────────────────────────────────
        for i, col in enumerate(self._dense_cols):
            key = "linear_model.dense_weight"
            if key in state:
                w = state[key].float()
                if w.shape[1] > i:
                    records.append({
                        "feature":    col,
                        "importance": w[:, i].abs().mean().item(),
                        "type":       "dense",
                    })

        if not records:
            return pd.DataFrame(columns=["feature", "importance_mean", "type"])

        fi = pd.DataFrame(records)
        max_imp = fi["importance"].max()
        if max_imp > 0:
            fi["importance"] = fi["importance"] / max_imp
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        fi.rename(columns={"importance": "importance_mean"}, inplace=True)
        return fi

    @property
    def feature_importance(self) -> pd.DataFrame:
        return self._fi

    # ------------------------------------------------------------------
    # 7. 序列化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        保存内容
        --------
        <path>/din_weights.pt      → PyTorch state_dict
        <path>/din_encoder.json    → FeatureEncoder（词表）
        <path>/din_meta.json       → 超参数 + 列名 + 序列配置 + best_epoch
        """
        torch = _require_torch()
        if self._model is None:
            raise RuntimeError("模型尚未训练，无法保存。")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.state_dict(), save_dir / "din_weights.pt")

        if self._encoder is not None:
            with open(save_dir / "din_encoder.json", "w", encoding="utf-8") as f:
                json.dump(self._encoder.to_dict(), f, ensure_ascii=False)

        seq_cfg_list = [
            {
                "seq_col":       c.seq_col,
                "target_col":    c.target_col,
                "maxlen":        c.maxlen,
                "padding_value": c.padding_value,
            }
            for c in self._seq_configs
        ]
        meta = {
            "params":       self._hp.to_dict(),
            "task":         self.task,
            "name":         self.name,
            "seed":         self.seed,
            "sparse_cols":  self._sparse_cols,
            "dense_cols":   self._dense_cols,
            "varlen_cols":  self._varlen_cols,
            "feature_names": self._feature_names,
            "seq_configs":  seq_cfg_list,
            "best_epoch":   self._best_epoch,
            "device":       self._device_str,
        }
        with open(save_dir / "din_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"[{self.name}] 模型已保存 → {save_dir}")

    def load(self, path: str) -> "DINModel":
        """从目录加载模型（须先 save 过）。"""
        torch = _require_torch()
        (DIN, SparseFeat, DenseFeat,
         VarLenSparseFeat, get_feature_names) = _require_varlen()

        load_dir = Path(path)

        with open(load_dir / "din_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._hp             = DINParams.from_dict(meta["params"])
        self.task            = meta["task"]
        self.name            = meta["name"]
        self.seed            = meta["seed"]
        self._sparse_cols    = meta["sparse_cols"]
        self._dense_cols     = meta["dense_cols"]
        self._varlen_cols    = meta["varlen_cols"]
        self._feature_names  = meta["feature_names"]
        self._best_epoch     = meta["best_epoch"]
        self._device_str     = meta.get("device", "cpu")

        # 重建序列配置
        self._seq_configs = [SeqFeatureConfig(**c) for c in meta["seq_configs"]]
        self._seq_col_names    = {c.seq_col    for c in self._seq_configs}
        self._target_col_names = {c.target_col for c in self._seq_configs}

        enc_path = load_dir / "din_encoder.json"
        if enc_path.exists():
            with open(enc_path, "r", encoding="utf-8") as f:
                self._encoder = FeatureEncoder.from_dict(json.load(f))

            # 重建 Padder
            for cfg in self._seq_configs:
                self._padders[cfg.seq_col] = SequencePadder(self._encoder, cfg)

        # 重建模型结构
        vocab = self._encoder.vocabulary_sizes if self._encoder else {}
        hp    = self._hp

        sparse_feats = [
            SparseFeat(c, vocabulary_size=vocab[c], embedding_dim=hp.embedding_dim)
            for c in self._sparse_cols
        ]
        shared_sparse = []
        varlen_feats  = []
        for cfg in self._seq_configs:
            target_sparse = SparseFeat(
                cfg.target_col,
                vocabulary_size=vocab.get(cfg.target_col, 2),
                embedding_dim=hp.embedding_dim,
            )
            if cfg.target_col not in self._sparse_cols:
                shared_sparse.append(target_sparse)
            varlen_feats.append(
                VarLenSparseFeat(sparsefeat=target_sparse, maxlen=cfg.maxlen,
                                 combiner="mean", length_name=None)
            )
        dense_feats = [DenseFeat(c, 1) for c in self._dense_cols]

        dnn_cols = sparse_feats + shared_sparse + varlen_feats + dense_feats
        self._linear_feature_columns = dnn_cols
        self._dnn_feature_columns    = dnn_cols

        history_feature_list = [c.target_col for c in self._seq_configs]
        deepctr_task = "binary" if self.task in ("binary", "multiclass") else "regression"

        self._model = DIN(
            dnn_feature_columns      = dnn_cols,
            history_feature_list     = history_feature_list,
            dnn_use_bn               = hp.dnn_use_bn,
            dnn_hidden_units         = hp.dnn_hidden_units,
            dnn_activation           = hp.dnn_activation,
            att_hidden_size          = hp.att_hidden_size,
            att_activation           = hp.att_activation,
            att_weight_normalization = hp.att_weight_normalization,
            l2_reg_embedding         = hp.l2_reg_embedding,
            l2_reg_dnn               = hp.l2_reg_dnn,
            dnn_dropout              = hp.dnn_dropout,
            seed                     = self.seed,
            task                     = deepctr_task,
            device                   = self._device_str,
        )
        state = torch.load(
            load_dir / "din_weights.pt",
            map_location=self._device_str,
        )
        self._model.load_state_dict(state)
        self._model.eval()

        logger.info(f"[{self.name}] 模型已加载 ← {load_dir}")
        return self

    # ------------------------------------------------------------------
    # 8. 种子注入 / Optuna 搜索空间
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_key() -> str:
        return "seed"

    @staticmethod
    def suggest_params(trial) -> dict:
        """Optuna 搜索空间。"""
        hidden_choices = ["[512, 256]", "[256, 128]", "[256, 128, 64]", "[128, 64]"]
        att_choices    = ["[80, 40]",   "[64, 32]",   "[128, 64]"]
        return {
            "embedding_dim":    trial.suggest_categorical("embedding_dim", [4, 8, 16, 32]),
            "dnn_hidden_units": trial.suggest_categorical("dnn_hidden_units", hidden_choices),
            "att_hidden_size":  trial.suggest_categorical("att_hidden_size", att_choices),
            "att_activation":   trial.suggest_categorical("att_activation", ["dice", "relu", "prelu"]),
            "dnn_dropout":      trial.suggest_float("dnn_dropout", 0.0, 0.5, step=0.1),
            "l2_reg_embedding": trial.suggest_float("l2_reg_embedding", 1e-7, 1e-4, log=True),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size":       trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192]),
        }

    # ------------------------------------------------------------------
    # 9. __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        hp  = self._hp
        seq = [f"{c.seq_col}→{c.target_col}(maxlen={c.maxlen})" for c in self._seq_configs]
        return (
            f"DINModel("
            f"task={self.task}, "
            f"dnn={hp.dnn_hidden_units}, "
            f"att={hp.att_hidden_size}[{hp.att_activation}], "
            f"emb={hp.embedding_dim}, "
            f"seqs={seq})"
        )


# ---------------------------------------------------------------------------
# 内部工具：训练 / 评估 epoch（与 DeepFM 逻辑对称，但不导入避免循环依赖）
# ---------------------------------------------------------------------------

def _make_loader(
    model_input: dict[str, np.ndarray],
    y: np.ndarray,
    feature_names: list[str],
    batch_size: int,
    shuffle: bool,
):
    """构建支持序列输入（2D array）的 DataLoader。"""
    import torch

    class _DS(torch.utils.data.Dataset):
        def __init__(self, xi, yi, names):
            self.names = names
            self.tensors = {
                k: torch.tensor(v)      # 支持 (N,) 和 (N, maxlen) 两种形状
                for k, v in xi.items()
            }
            self.y = torch.tensor(yi, dtype=torch.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return {k: self.tensors[k][idx] for k in self.names}, self.y[idx]

    ds = _DS(model_input, y, feature_names)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _train_epoch(model, loader, loss_fn, optimizer, task, device) -> float:
    import torch
    model.train()
    total = 0.0
    for x_batch, y_batch in loader:
        x_batch = {k: v.to(device) for k, v in x_batch.items()}
        y_batch  = y_batch.to(device)
        optimizer.zero_grad()
        pred = model(x_batch).squeeze(-1)
        if task == "binary":
            pred = torch.sigmoid(pred)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y_batch)
    return total / len(loader.dataset)


def _eval_epoch(model, loader, loss_fn, task, device) -> float:
    import torch
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = {k: v.to(device) for k, v in x_batch.items()}
            y_batch  = y_batch.to(device)
            pred = model(x_batch).squeeze(-1)
            if task == "binary":
                pred = torch.sigmoid(pred)
            loss = loss_fn(pred, y_batch)
            total += loss.item() * len(y_batch)
    return total / len(loader.dataset)
