"""
models/deep/two_tower_wrapper.py
---------------------------------
生产级双塔召回模型 (Two-Tower / Dual Encoder) Wrapper。

架构总览
--------
                     ┌──────────────────────────────────────────┐
  User Features ──►  │  User Tower DNN  │ → user_emb (dim=D)  │
                     │  [BN → FC → Act] │                       │
                     └──────────────────────────────────────────┘
                                                  │
                                          In-batch Softmax Loss
                                          (InfoNCE / Temperature)
                                                  │
                     ┌──────────────────────────────────────────┐
  Item Features ──►  │  Item Tower DNN  │ → item_emb (dim=D)  │
                     │  [BN → FC → Act] │                       │
                     └──────────────────────────────────────────┘

训练目标
--------
  最大化 正样本对 (u, v+) 的内积，同时最小化 (u, v-) 的内积。
  核心 Loss：**In-batch Negative Sampling + Temperature-scaled InfoNCE**

    L = -1/B * Σ_i log( exp(u_i · v_i / τ)
                        / Σ_j exp(u_i · v_j / τ) )

  其中 batch 内其他物品自动成为负样本，无需额外采样开销。

向量导出
--------
  model.get_user_embedding(X_user)  → np.ndarray (N, D)
  model.get_item_embedding(X_item)  → np.ndarray (N, D)

  导出的向量可直接写入 Faiss IndexFlatIP 进行 ANN 检索。

依赖
----
  pip install torch deepctr-torch  （仅用 deepctr-torch 的 FeatureEncoder 工具）

接口一致性
----------
  与 DeepFMModel / DINModel 保持完全对称：
  fit / predict / predict_proba / save / load /
  feature_importance / suggest_params / _seed_key
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.base_model import BaseModel
from models.deep.deepfm_wrapper import (
    EarlyStopper,
    FeatureEncoder,
    _require_torch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 可选依赖守卫
# ---------------------------------------------------------------------------

def _require_torch_nn():
    torch = _require_torch()
    return torch, torch.nn


# ---------------------------------------------------------------------------
# 超参数容器
# ---------------------------------------------------------------------------

@dataclass
class TwoTowerParams:
    # ── 塔结构 ─────────────────────────────────────────────────────────────
    user_tower_units: tuple = (256, 128, 64)   # User Tower 各层维度
    item_tower_units: tuple = (256, 128, 64)   # Item Tower 各层维度
    embedding_dim:    int   = 8                # 类别特征 Embedding 维度
    output_dim:       int   = 64               # 最终输出隐向量维度（两塔必须一致）
    activation:       str   = "relu"
    dropout:          float = 0.0
    use_bn:           bool  = True

    # ── 相似度 ─────────────────────────────────────────────────────────────
    similarity:    str   = "dot"       # "dot" | "cosine"
    temperature:   float = 0.05        # InfoNCE 温度系数 τ（越小越"锐利"）
    l2_normalize:  bool  = True        # 输出向量是否 L2 归一化

    # ── 正则化 ─────────────────────────────────────────────────────────────
    l2_reg:   float = 1e-5

    # ── 优化器 ─────────────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    optimizer:     str   = "adam"

    # ── 训练控制 ───────────────────────────────────────────────────────────
    batch_size: int  = 2048
    epochs:     int  = 30
    device:     str  = "auto"

    # ── Early Stopping ─────────────────────────────────────────────────────
    patience:     int   = 5
    min_delta:    float = 1e-5
    monitor_mode: str   = "min"

    @classmethod
    def from_dict(cls, d: dict) -> "TwoTowerParams":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        obj = cls(**valid)
        for attr in ("user_tower_units", "item_tower_units"):
            if isinstance(getattr(obj, attr), list):
                setattr(obj, attr, tuple(getattr(obj, attr)))
        return obj

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# 单塔 DNN（PyTorch 原生，无 DeepCTR 依赖）
# ---------------------------------------------------------------------------

class _TowerDNN:
    """
    构造一个 N 层 DNN：
      输入层维度 → hidden_units[0] → ... → output_dim

    返回 torch.nn.Sequential 实例。
    """

    @staticmethod
    def build(
        input_dim:    int,
        hidden_units: tuple,
        output_dim:   int,
        activation:   str   = "relu",
        dropout:      float = 0.0,
        use_bn:       bool  = True,
    ):
        torch, nn = _require_torch_nn()

        act_map = {
            "relu":    nn.ReLU(),
            "prelu":   nn.PReLU(),
            "dice":    nn.PReLU(),   # Dice 近似用 PReLU（无外部依赖）
            "gelu":    nn.GELU(),
            "tanh":    nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }

        layers: list = []
        prev_dim = input_dim

        for units in hidden_units:
            layers.append(nn.Linear(prev_dim, units))
            if use_bn:
                layers.append(nn.BatchNorm1d(units))
            layers.append(act_map.get(activation, nn.ReLU()))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = units

        # 输出投影层（无激活，保持线性）
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# 特征嵌入层：将 DataFrame 编码为稠密向量
# ---------------------------------------------------------------------------

class FeatureEmbeddingLayer:
    """
    将一列 DataFrame 特征转为稠密向量矩阵（即塔的输入）。

    sparse cols → Embedding → concat
    dense  cols → 直接拼接
    最终输出 shape: (N, total_input_dim)
    """

    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = embedding_dim
        self._encoder: Optional[FeatureEncoder] = None
        self._sparse_cols: list[str] = []
        self._dense_cols:  list[str] = []
        self._embedding_layer = None   # torch.nn.EmbeddingBag
        self.input_dim: int  = 0       # 塔的最终输入维度

    def fit(self, X: pd.DataFrame) -> "FeatureEmbeddingLayer":
        self._sparse_cols = [c for c in X.columns if X[c].dtype.name == "category"]
        self._dense_cols  = [
            c for c in X.columns
            if c not in self._sparse_cols and np.issubdtype(X[c].dtype, np.number)
        ]
        self._encoder = FeatureEncoder()
        self._encoder.fit(X, self._sparse_cols)
        # 输入维度 = sparse Embedding 总维度 + dense 列数
        self.input_dim = (
            len(self._sparse_cols) * self.embedding_dim
            + len(self._dense_cols)
        )
        return self

    def build_embedding_layer(self, device: str):
        """返回 torch.nn.ModuleDict（每个 sparse 列一个 Embedding 表）。"""
        torch, nn = _require_torch_nn()
        vocab = self._encoder.vocabulary_sizes
        emb_dict = nn.ModuleDict({
            col: nn.Embedding(
                num_embeddings = vocab[col],
                embedding_dim  = self.embedding_dim,
                padding_idx    = 0,
            )
            for col in self._sparse_cols
        })
        return emb_dict.to(device)

    def transform_to_numpy(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        返回 {"sparse": {col: int_array}, "dense": float_matrix}
        供 DataLoader 使用。
        """
        X_enc = self._encoder.transform(X)
        sparse_arrays = {
            col: X_enc[col].values.astype(np.int64)
            for col in self._sparse_cols
        }
        dense_array = (
            X[[c for c in self._dense_cols if c in X.columns]]
            .fillna(0.0)
            .values.astype(np.float32)
            if self._dense_cols else np.zeros((len(X), 0), dtype=np.float32)
        )
        return {"sparse": sparse_arrays, "dense": dense_array}

    def to_dict(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "sparse_cols":   self._sparse_cols,
            "dense_cols":    self._dense_cols,
            "encoder":       self._encoder.to_dict() if self._encoder else {},
            "input_dim":     self.input_dim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureEmbeddingLayer":
        obj = cls(embedding_dim=d["embedding_dim"])
        obj._sparse_cols = d["sparse_cols"]
        obj._dense_cols  = d["dense_cols"]
        obj._encoder     = FeatureEncoder.from_dict(d["encoder"])
        obj.input_dim    = d["input_dim"]
        return obj


# ---------------------------------------------------------------------------
# 双塔 PyTorch 模型（nn.Module）
# ---------------------------------------------------------------------------

class _TwoTowerModule:
    """
    工厂函数，返回包含 user_tower / item_tower / emb_dicts 的 nn.Module。
    这样整个模型可以用一个 state_dict 序列化。
    """

    @staticmethod
    def build(
        user_feat_layer: FeatureEmbeddingLayer,
        item_feat_layer: FeatureEmbeddingLayer,
        hp:              TwoTowerParams,
        device:          str,
    ):
        torch, nn = _require_torch_nn()

        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                # Embedding 表
                self.user_emb_dict = user_feat_layer.build_embedding_layer(device)
                self.item_emb_dict = item_feat_layer.build_embedding_layer(device)

                # 双塔 DNN
                self.user_tower = _TowerDNN.build(
                    user_feat_layer.input_dim, hp.user_tower_units,
                    hp.output_dim, hp.activation, hp.dropout, hp.use_bn,
                )
                self.item_tower = _TowerDNN.build(
                    item_feat_layer.input_dim, hp.item_tower_units,
                    hp.output_dim, hp.activation, hp.dropout, hp.use_bn,
                )

            def _embed(self, feat_dict: dict, emb_dict: nn.ModuleDict,
                       dense_tensor) -> "torch.Tensor":
                """将 sparse + dense 特征拼接为单一稠密向量。"""
                parts = []
                for col, emb in emb_dict.items():
                    parts.append(emb(feat_dict[col]))   # (B, emb_dim)
                if dense_tensor.shape[1] > 0:
                    parts.append(dense_tensor)
                return torch.cat(parts, dim=-1)         # (B, input_dim)

            def encode_user(self, user_sparse: dict, user_dense) -> "torch.Tensor":
                x = self._embed(user_sparse, self.user_emb_dict, user_dense)
                return self.user_tower(x)

            def encode_item(self, item_sparse: dict, item_dense) -> "torch.Tensor":
                x = self._embed(item_sparse, self.item_emb_dict, item_dense)
                return self.item_tower(x)

            def forward(self, user_sparse, user_dense, item_sparse, item_dense):
                u = self.encode_user(user_sparse, user_dense)
                v = self.encode_item(item_sparse, item_dense)
                return u, v

        return _Module().to(device)


# ---------------------------------------------------------------------------
# In-batch Negative Sampling Loss（InfoNCE）
# ---------------------------------------------------------------------------

class InBatchSoftmaxLoss:
    """
    In-batch Negative Sampling Loss（温度缩放 InfoNCE）。

    给定 batch B 对正样本 (u_i, v_i)：
      ① 对角线 = 正样本相似度
      ② 非对角线 = batch 内负样本相似度

    L = CrossEntropy(SimMatrix / τ, target=eye(B))

    对称版本同时考虑 user-side 和 item-side loss，取均值。
    """

    def __init__(self, temperature: float = 0.05, similarity: str = "dot",
                 l2_normalize: bool = True):
        self.temperature   = temperature
        self.similarity    = similarity
        self.l2_normalize  = l2_normalize

    def __call__(self, u: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
        torch, nn = _require_torch_nn()

        if self.l2_normalize:
            u = nn.functional.normalize(u, p=2, dim=-1)
            v = nn.functional.normalize(v, p=2, dim=-1)

        # 相似度矩阵 (B, B)
        sim = torch.matmul(u, v.T) / self.temperature   # (B, B)

        # 正样本标签 = 对角线 index
        B      = u.size(0)
        labels = torch.arange(B, device=u.device)

        # 对称 InfoNCE Loss
        loss_u = nn.functional.cross_entropy(sim,   labels)   # user → item
        loss_v = nn.functional.cross_entropy(sim.T, labels)   # item → user
        return (loss_u + loss_v) / 2.0


# ---------------------------------------------------------------------------
# 主体：TwoTowerModel
# ---------------------------------------------------------------------------

class TwoTowerModel(BaseModel):
    """
    双塔召回模型 Wrapper。

    Parameters
    ----------
    params        : 超参数字典（与 DEFAULT_PARAMS 合并）
    task          : "binary"（召回场景通常二分类）| "regression"
    name          : 模型标识
    seed          : 随机种子
    user_cols     : 用户塔使用的列名列表
    item_cols     : 物品塔使用的列名列表

    使用示例
    --------
    >>> model = TwoTowerModel(
    ...     task="binary",
    ...     user_cols=["user_id", "age", "gender", "city"],
    ...     item_cols=["item_id", "cate_id", "price", "brand"],
    ... )
    >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    >>>
    >>> # 推断相似度分数（排序召回）
    >>> scores = model.predict_proba(X_test)
    >>>
    >>> # 导出向量 → 写入 Faiss
    >>> item_embs = model.get_item_embedding(X_items)  # (N_items, 64)
    >>> user_embs = model.get_user_embedding(X_users)  # (N_users, 64)
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "user_tower_units": [256, 128, 64],
        "item_tower_units": [256, 128, 64],
        "embedding_dim":    8,
        "output_dim":       64,
        "activation":       "relu",
        "dropout":          0.0,
        "use_bn":           True,
        "similarity":       "dot",
        "temperature":      0.05,
        "l2_normalize":     True,
        "l2_reg":           1e-5,
        "learning_rate":    1e-3,
        "optimizer":        "adam",
        "batch_size":       2048,
        "epochs":           30,
        "device":           "auto",
        "patience":         5,
        "min_delta":        1e-5,
        "monitor_mode":     "min",
    }

    def __init__(
        self,
        params:    Optional[dict]      = None,
        task:      str                 = "binary",
        name:      str                 = "two_tower",
        seed:      int                 = 42,
        user_cols: Optional[list[str]] = None,
        item_cols: Optional[list[str]] = None,
    ):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged, task=task, name=name, seed=seed)

        self._hp: TwoTowerParams = TwoTowerParams.from_dict(self.params)

        self.user_cols: list[str] = list(user_cols or [])
        self.item_cols: list[str] = list(item_cols or [])

        # 运行时状态
        self._module                               = None
        self._user_feat_layer: Optional[FeatureEmbeddingLayer] = None
        self._item_feat_layer: Optional[FeatureEmbeddingLayer] = None
        self._loss_fn: Optional[InBatchSoftmaxLoss]            = None
        self._fi:      pd.DataFrame                            = pd.DataFrame()
        self._best_epoch: int  = 0
        self._device_str: str  = ""

        self.history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # 1. 列分配校验
    # ------------------------------------------------------------------

    def _split_columns(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """将 DataFrame 拆分为 user_df 和 item_df。"""
        if not self.user_cols or not self.item_cols:
            raise ValueError(
                "必须通过构造函数指定 user_cols 和 item_cols。\n"
                "示例：TwoTowerModel(user_cols=['user_id','age'], item_cols=['item_id','price'])"
            )
        missing_u = [c for c in self.user_cols if c not in X.columns]
        missing_i = [c for c in self.item_cols if c not in X.columns]
        if missing_u:
            raise ValueError(f"user_cols 中以下列不存在：{missing_u}")
        if missing_i:
            raise ValueError(f"item_cols 中以下列不存在：{missing_i}")

        return X[self.user_cols].copy(), X[self.item_cols].copy()

    # ------------------------------------------------------------------
    # 2. 设备选择
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        torch = _require_torch()
        if self._hp.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._hp.device

    # ------------------------------------------------------------------
    # 3. 将 DataFrame 转为 Tensor 字典
    # ------------------------------------------------------------------

    def _to_tensors(
        self,
        X_user: pd.DataFrame,
        X_item: pd.DataFrame,
        device: str,
    ) -> tuple[dict, "torch.Tensor", dict, "torch.Tensor"]:
        torch = _require_torch()

        def _convert(feat_data: dict, device: str):
            sparse = {
                k: torch.tensor(v, dtype=torch.long).to(device)
                for k, v in feat_data["sparse"].items()
            }
            dense = torch.tensor(feat_data["dense"], dtype=torch.float32).to(device)
            return sparse, dense

        user_data = self._user_feat_layer.transform_to_numpy(X_user)
        item_data = self._item_feat_layer.transform_to_numpy(X_item)

        u_sparse, u_dense = _convert(user_data, device)
        i_sparse, i_dense = _convert(item_data, device)
        return u_sparse, u_dense, i_sparse, i_dense

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
    ) -> "TwoTowerModel":
        torch = _require_torch()
        hp    = self._hp

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._device_str = self._resolve_device()
        logger.info(f"[{self.name}] 设备: {self._device_str}")

        # ── 列拆分 ──────────────────────────────────────────────────────
        X_user_train, X_item_train = self._split_columns(X_train)
        y_np = y_train.values.astype(np.float32)

        # ── 特征嵌入层 fit ──────────────────────────────────────────────
        self._user_feat_layer = FeatureEmbeddingLayer(hp.embedding_dim).fit(X_user_train)
        self._item_feat_layer = FeatureEmbeddingLayer(hp.embedding_dim).fit(X_item_train)

        logger.info(
            f"[{self.name}] UserTower input_dim={self._user_feat_layer.input_dim} | "
            f"ItemTower input_dim={self._item_feat_layer.input_dim}"
        )

        # ── 构建双塔模型 ────────────────────────────────────────────────
        self._module = _TwoTowerModule.build(
            self._user_feat_layer,
            self._item_feat_layer,
            hp,
            self._device_str,
        )

        # ── 损失函数 ────────────────────────────────────────────────────
        self._loss_fn = InBatchSoftmaxLoss(
            temperature  = hp.temperature,
            similarity   = hp.similarity,
            l2_normalize = hp.l2_normalize,
        )

        # ── 优化器 + L2 正则（weight_decay）────────────────────────────
        opt_map = {
            "adam":    torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
            "rmsprop": torch.optim.RMSprop,
            "sgd":     torch.optim.SGD,
        }
        optimizer = opt_map.get(hp.optimizer.lower(), torch.optim.Adam)(
            self._module.parameters(),
            lr           = hp.learning_rate,
            weight_decay = hp.l2_reg,
        )

        # ── DataLoader ──────────────────────────────────────────────────
        train_loader = _make_pair_loader(
            self._user_feat_layer.transform_to_numpy(X_user_train),
            self._item_feat_layer.transform_to_numpy(X_item_train),
            y_np,
            self.user_cols, self.item_cols,
            hp.batch_size, shuffle=True,
        )

        has_val    = X_val is not None and y_val is not None
        val_loader = None
        if has_val:
            X_user_val, X_item_val = self._split_columns(X_val)
            val_loader = _make_pair_loader(
                self._user_feat_layer.transform_to_numpy(X_user_val),
                self._item_feat_layer.transform_to_numpy(X_item_val),
                y_val.values.astype(np.float32),
                self.user_cols, self.item_cols,
                hp.batch_size * 4, shuffle=False,
            )

        # ── Early Stopping + Checkpoint ─────────────────────────────────
        stopper    = EarlyStopper(hp.patience, hp.min_delta, hp.monitor_mode)
        best_state = copy.deepcopy(self._module.state_dict())

        self.history = {"train_loss": [], "val_loss": []}
        t_start = time.perf_counter()

        for epoch in range(1, hp.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer)
            self.history["train_loss"].append(train_loss)

            val_loss = (
                self._eval_epoch(val_loader)
                if val_loader else float("nan")
            )
            self.history["val_loss"].append(val_loss)

            monitor_val = val_loss if has_val else train_loss
            should_stop = stopper.step(monitor_val, epoch)

            if stopper._counter == 0:
                best_state      = copy.deepcopy(self._module.state_dict())
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
                    f"best epoch={self._best_epoch} | loss={stopper.best_value:.5f}"
                )
                break

        self._module.load_state_dict(best_state)
        elapsed = time.perf_counter() - t_start
        logger.info(
            f"[{self.name}] 训练完成 | best epoch={self._best_epoch} | 耗时={elapsed:.1f}s"
        )

        self._fi = self._compute_feature_importance()
        return self

    # ------------------------------------------------------------------
    # 4a / 4b. 训练 & 评估 Epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, loader, optimizer) -> float:
        self._module.train()
        total = 0.0
        for batch in loader:
            u_sparse, u_dense, i_sparse, i_dense, _ = _unpack_batch(
                batch, self.user_cols, self.item_cols, self._device_str
            )
            optimizer.zero_grad()
            u_emb, i_emb = self._module(u_sparse, u_dense, i_sparse, i_dense)
            loss = self._loss_fn(u_emb, i_emb)
            loss.backward()
            optimizer.step()
            total += loss.item() * u_emb.size(0)
        return total / len(loader.dataset)

    def _eval_epoch(self, loader) -> float:
        torch = _require_torch()
        self._module.eval()
        total = 0.0
        with torch.no_grad():
            for batch in loader:
                u_sparse, u_dense, i_sparse, i_dense, _ = _unpack_batch(
                    batch, self.user_cols, self.item_cols, self._device_str
                )
                u_emb, i_emb = self._module(u_sparse, u_dense, i_sparse, i_dense)
                loss = self._loss_fn(u_emb, i_emb)
                total += loss.item() * u_emb.size(0)
        return total / len(loader.dataset)

    # ------------------------------------------------------------------
    # 5. predict_proba（兼容 CrossValidatorTrainer）
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.predict_proba(X, **kwargs)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        计算每个 (user, item) 样本对的相似度分数。

        输出：float64 一维数组，shape=(N,)
        兼容 CrossValidatorTrainer 的 OOF 流程与 StackingEnsemble。
        """
        torch = _require_torch()
        if self._module is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

        X_user, X_item = self._split_columns(X)
        hp = self._hp

        loader = _make_pair_loader(
            self._user_feat_layer.transform_to_numpy(X_user),
            self._item_feat_layer.transform_to_numpy(X_item),
            np.zeros(len(X), dtype=np.float32),
            self.user_cols, self.item_cols,
            hp.batch_size * 4, shuffle=False,
        )

        scores = []
        self._module.eval()
        with torch.no_grad():
            for batch in loader:
                u_sparse, u_dense, i_sparse, i_dense, _ = _unpack_batch(
                    batch, self.user_cols, self.item_cols, self._device_str
                )
                u_emb, i_emb = self._module(u_sparse, u_dense, i_sparse, i_dense)

                if hp.l2_normalize:
                    import torch.nn.functional as F
                    u_emb = F.normalize(u_emb, p=2, dim=-1)
                    i_emb = F.normalize(i_emb, p=2, dim=-1)

                # 逐样本点积 → (B,)
                sim = (u_emb * i_emb).sum(dim=-1)
                scores.append(sim.cpu().numpy())

        return np.concatenate(scores, axis=0).astype(np.float64)

    # ------------------------------------------------------------------
    # 6. 向量导出接口（接入 Faiss 的核心）
    # ------------------------------------------------------------------

    def get_user_embedding(
        self,
        X_user: pd.DataFrame,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        批量提取用户向量。

        Parameters
        ----------
        X_user    : 包含 user_cols 列的 DataFrame
        normalize : True → L2 归一化；None → 使用 hp.l2_normalize 设置

        Returns
        -------
        np.ndarray  shape=(N, output_dim)，可直接写入 Faiss IndexFlatIP
        """
        return self._extract_embedding(X_user, tower="user", normalize=normalize)

    def get_item_embedding(
        self,
        X_item: pd.DataFrame,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        批量提取物品向量。

        Parameters
        ----------
        X_item    : 包含 item_cols 列的 DataFrame
        normalize : True → L2 归一化；None → 使用 hp.l2_normalize 设置

        Returns
        -------
        np.ndarray  shape=(N, output_dim)，可直接写入 Faiss IndexFlatIP

        典型使用
        --------
        >>> import faiss
        >>> item_embs = model.get_item_embedding(X_items).astype("float32")
        >>> index = faiss.IndexFlatIP(item_embs.shape[1])
        >>> index.add(item_embs)
        >>> D, I = index.search(user_embs, k=100)   # Top-100 召回
        """
        return self._extract_embedding(X_item, tower="item", normalize=normalize)

    def _extract_embedding(
        self,
        X: pd.DataFrame,
        tower: str,
        normalize: Optional[bool],
    ) -> np.ndarray:
        torch = _require_torch()
        if self._module is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

        hp             = self._hp
        do_normalize   = hp.l2_normalize if normalize is None else normalize
        feat_layer     = (self._user_feat_layer if tower == "user"
                          else self._item_feat_layer)
        feat_data      = feat_layer.transform_to_numpy(X)

        # 构建只含单塔特征的 DataLoader
        loader = _make_single_tower_loader(feat_data, hp.batch_size * 4)

        embs = []
        self._module.eval()
        with torch.no_grad():
            for sparse_dict, dense_t in loader:
                sparse_t = {k: v.to(self._device_str) for k, v in sparse_dict.items()}
                dense_t  = dense_t.to(self._device_str)

                if tower == "user":
                    out = self._module.encode_user(sparse_t, dense_t)
                else:
                    out = self._module.encode_item(sparse_t, dense_t)

                if do_normalize:
                    import torch.nn.functional as F
                    out = F.normalize(out, p=2, dim=-1)

                embs.append(out.cpu().numpy())

        return np.concatenate(embs, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # 7. 特征重要性
    # ------------------------------------------------------------------

    def _compute_feature_importance(self) -> pd.DataFrame:
        """
        Embedding L2 范数：反映各特征在向量空间中的"激活强度"。
        分塔标注 type = user_sparse / user_dense / item_sparse / item_dense。
        """
        if self._module is None:
            return pd.DataFrame()

        records: list[dict] = []
        state = self._module.state_dict()

        def _parse_emb(prefix: str, feat_layer: FeatureEmbeddingLayer, tag: str):
            for col in feat_layer._sparse_cols:
                key = f"{prefix}.{col}.weight"
                if key in state:
                    norm = state[key].float().norm(dim=1).mean().item()
                    records.append({"feature": col, "importance": norm, "type": tag})
            for col in feat_layer._dense_cols:
                records.append({"feature": col, "importance": 0.0, "type": f"{tag}_dense"})

        _parse_emb("user_emb_dict", self._user_feat_layer, "user")
        _parse_emb("item_emb_dict", self._item_feat_layer, "item")

        if not records:
            return pd.DataFrame(columns=["feature", "importance_mean", "type"])

        fi = pd.DataFrame(records)
        max_i = fi["importance"].max()
        if max_i > 0:
            fi["importance"] /= max_i
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        fi.rename(columns={"importance": "importance_mean"}, inplace=True)
        return fi

    @property
    def feature_importance(self) -> pd.DataFrame:
        return self._fi

    # ------------------------------------------------------------------
    # 8. 序列化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        保存内容
        --------
        <path>/two_tower_weights.pt  → PyTorch state_dict
        <path>/user_feat_layer.json  → User 侧特征编码器
        <path>/item_feat_layer.json  → Item 侧特征编码器
        <path>/two_tower_meta.json   → 超参数 + 列名 + best_epoch
        """
        torch = _require_torch()
        if self._module is None:
            raise RuntimeError("模型尚未训练，无法保存。")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self._module.state_dict(), save_dir / "two_tower_weights.pt")

        with open(save_dir / "user_feat_layer.json", "w", encoding="utf-8") as f:
            json.dump(self._user_feat_layer.to_dict(), f, ensure_ascii=False)
        with open(save_dir / "item_feat_layer.json", "w", encoding="utf-8") as f:
            json.dump(self._item_feat_layer.to_dict(), f, ensure_ascii=False)

        meta = {
            "params":     self._hp.to_dict(),
            "task":       self.task,
            "name":       self.name,
            "seed":       self.seed,
            "user_cols":  self.user_cols,
            "item_cols":  self.item_cols,
            "best_epoch": self._best_epoch,
            "device":     self._device_str,
        }
        with open(save_dir / "two_tower_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"[{self.name}] 模型已保存 → {save_dir}")

    def load(self, path: str) -> "TwoTowerModel":
        """从目录加载模型。"""
        torch = _require_torch()
        load_dir = Path(path)

        with open(load_dir / "two_tower_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._hp          = TwoTowerParams.from_dict(meta["params"])
        self.task         = meta["task"]
        self.name         = meta["name"]
        self.seed         = meta["seed"]
        self.user_cols    = meta["user_cols"]
        self.item_cols    = meta["item_cols"]
        self._best_epoch  = meta["best_epoch"]
        self._device_str  = meta.get("device", "cpu")

        with open(load_dir / "user_feat_layer.json", "r", encoding="utf-8") as f:
            self._user_feat_layer = FeatureEmbeddingLayer.from_dict(json.load(f))
        with open(load_dir / "item_feat_layer.json", "r", encoding="utf-8") as f:
            self._item_feat_layer = FeatureEmbeddingLayer.from_dict(json.load(f))

        self._module = _TwoTowerModule.build(
            self._user_feat_layer,
            self._item_feat_layer,
            self._hp,
            self._device_str,
        )
        state = torch.load(
            load_dir / "two_tower_weights.pt",
            map_location=self._device_str,
        )
        self._module.load_state_dict(state)
        self._module.eval()

        self._loss_fn = InBatchSoftmaxLoss(
            temperature  = self._hp.temperature,
            similarity   = self._hp.similarity,
            l2_normalize = self._hp.l2_normalize,
        )

        logger.info(f"[{self.name}] 模型已加载 ← {load_dir}")
        return self

    # ------------------------------------------------------------------
    # 9. 种子注入 / Optuna 搜索空间
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_key() -> str:
        return "seed"

    @staticmethod
    def suggest_params(trial) -> dict:
        tower_choices = ["[256, 128, 64]", "[512, 256, 64]", "[128, 64]", "[256, 64]"]
        return {
            "embedding_dim":    trial.suggest_categorical("embedding_dim", [4, 8, 16, 32]),
            "user_tower_units": trial.suggest_categorical("user_tower_units", tower_choices),
            "item_tower_units": trial.suggest_categorical("item_tower_units", tower_choices),
            "output_dim":       trial.suggest_categorical("output_dim", [32, 64, 128, 256]),
            "temperature":      trial.suggest_float("temperature", 0.01, 0.2, log=True),
            "dropout":          trial.suggest_float("dropout", 0.0, 0.4, step=0.1),
            "l2_reg":           trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size":       trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
        }

    # ------------------------------------------------------------------
    # 10. __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        hp = self._hp
        return (
            f"TwoTowerModel("
            f"user_tower={hp.user_tower_units}, "
            f"item_tower={hp.item_tower_units}, "
            f"output_dim={hp.output_dim}, "
            f"τ={hp.temperature}, "
            f"sim={hp.similarity})"
        )


# ---------------------------------------------------------------------------
# DataLoader 工具函数
# ---------------------------------------------------------------------------

def _make_pair_loader(
    user_data:   dict,
    item_data:   dict,
    y:           np.ndarray,
    user_cols:   list[str],
    item_cols:   list[str],
    batch_size:  int,
    shuffle:     bool,
):
    """构建 (user_features, item_features, label) 三元组 DataLoader。"""
    import torch

    class _PairDS(torch.utils.data.Dataset):
        def __init__(self, ud, id_, labels):
            self.u_sparse = {k: torch.tensor(v, dtype=torch.long)  for k, v in ud["sparse"].items()}
            self.u_dense  = torch.tensor(ud["dense"], dtype=torch.float32)
            self.i_sparse = {k: torch.tensor(v, dtype=torch.long)  for k, v in id_["sparse"].items()}
            self.i_dense  = torch.tensor(id_["dense"], dtype=torch.float32)
            self.labels   = torch.tensor(labels, dtype=torch.float32)
            self._u_keys  = list(ud["sparse"].keys())
            self._i_keys  = list(id_["sparse"].keys())

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return (
                {k: self.u_sparse[k][idx] for k in self._u_keys},
                self.u_dense[idx],
                {k: self.i_sparse[k][idx] for k in self._i_keys},
                self.i_dense[idx],
                self.labels[idx],
            )

    ds = _PairDS(user_data, item_data, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _make_single_tower_loader(feat_data: dict, batch_size: int):
    """单塔特征 DataLoader（用于向量导出）。"""
    import torch

    class _SingleDS(torch.utils.data.Dataset):
        def __init__(self, fd):
            self.sparse = {k: torch.tensor(v, dtype=torch.long)  for k, v in fd["sparse"].items()}
            self.dense  = torch.tensor(fd["dense"], dtype=torch.float32)
            self._keys  = list(fd["sparse"].keys())
            self._n     = len(self.dense)

        def __len__(self):
            return self._n

        def __getitem__(self, idx):
            return {k: self.sparse[k][idx] for k in self._keys}, self.dense[idx]

    ds = _SingleDS(feat_data)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def _unpack_batch(batch, user_cols: list, item_cols: list, device: str):
    """将 DataLoader 的 batch 元组拆包并发送到指定设备。"""
    u_sparse_d, u_dense, i_sparse_d, i_dense, labels = batch
    u_sparse = {k: v.to(device) for k, v in u_sparse_d.items()}
    i_sparse = {k: v.to(device) for k, v in i_sparse_d.items()}
    return u_sparse, u_dense.to(device), i_sparse, i_dense.to(device), labels.to(device)
