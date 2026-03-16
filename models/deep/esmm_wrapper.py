"""
models/deep/esmm_wrapper.py
----------------------------
生产级 ESMM (Entire Space Multi-Task Model) Wrapper，基于 DeepCTR-Torch。

论文参考
--------
  《Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click
  Conversion Rate》  Ma et al., SIGIR 2018 — https://arxiv.org/abs/1804.07931

核心问题：CVR 的样本选择偏差 (Sample Selection Bias)
-------------------------------------------------------
  传统 CVR 模型只在「已点击样本」上训练：
    TRAINING SPACE: {点击样本}    → 仅占全量曝光的 1‰ ~ 5%
    SERVING SPACE:  {全量曝光样本}
  ⇒ 训练/预测空间不一致，模型严重低估真实 CVR。

ESMM 的解法（全空间训练）
--------------------------
  定义辅助任务 pCTCVR（点击且转化的概率）：
    pCTCVR = pCTR × pCVR

  Loss = BCE(pCTR,   label_click)        # 曝光空间，样本充足
       + BCE(pCTCVR, label_conversion)   # 曝光空间，标签稀疏但不偏

  ⇒ pVCR 塔通过 pCTCVR 的梯度间接在全空间更新，彻底解决偏差问题。

模型架构
--------
  ┌─────────────────────────────────────────────────────────┐
  │               Shared Embedding Layer                     │
  │  cat_col_1 → Emb_1 ┐                                    │
  │  cat_col_2 → Emb_2 ├─── [concat] ─┬── CTR DNN → pCTR   │
  │  num_col_1 ────────┘               └── CVR DNN → pCVR   │
  │                                                          │
  │                         pCTR × pCVR = pCTCVR            │
  └─────────────────────────────────────────────────────────┘

依赖
----
  pip install deepctr-torch>=0.9.0 torch

接口一致性
----------
  与 DeepFMModel / DINModel 完全对称：
  fit / predict / predict_proba / save / load /
  feature_importance / suggest_params / _seed_key

  predict_proba() 默认返回 pCTCVR（可通过 output 参数切换）
  额外提供：predict_ctr() / predict_cvr() 单独获取两个子任务分数
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

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

def _require_esmm():
    """加载 DeepCTR-Torch 的 ESMM 及相关输入工具。"""
    try:
        from deepctr_torch.models import ESMM
        from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
        return ESMM, SparseFeat, DenseFeat, get_feature_names
    except ImportError:
        raise ImportError(
            "DeepCTR-Torch 未安装或版本过低。请执行：\n"
            "  pip install deepctr-torch>=0.9.0"
        )


# ---------------------------------------------------------------------------
# 超参数容器
# ---------------------------------------------------------------------------

@dataclass
class ESMMParams:
    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_dim: int     = 8

    # ── 双塔 DNN（CTR 和 CVR 各自的隐层，共享 Embedding）─────────────────
    dnn_hidden_units:  tuple = (256, 128)
    dnn_activation:    str   = "relu"
    dnn_dropout:       float = 0.0
    dnn_use_bn:        bool  = False

    # ── 损失权重（允许用户调整 CTR / CTCVR 的贡献比例）──────────────────
    ctr_loss_weight:   float = 1.0
    ctcvr_loss_weight: float = 1.0

    # ── 正则化 ─────────────────────────────────────────────────────────────
    l2_reg_embedding: float  = 1e-6
    l2_reg_dnn:       float  = 0.0

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
    monitor:      str        = "val_loss"   # "val_loss" | "val_ctr_loss" | "val_ctcvr_loss"
    monitor_mode: str        = "min"

    # ── 标签列名（从 DataFrame 中读取双标签）─────────────────────────────
    label_click:      str    = "label_click"       # 点击标签
    label_conversion: str    = "label_conversion"  # 转化标签（全空间，非仅点击样本）

    @classmethod
    def from_dict(cls, d: dict) -> "ESMMParams":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        obj   = cls(**valid)
        if isinstance(obj.dnn_hidden_units, list):
            obj.dnn_hidden_units = tuple(obj.dnn_hidden_units)
        return obj

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# 双任务 DataLoader
# ---------------------------------------------------------------------------

def _make_dual_label_loader(
    model_input:    dict[str, np.ndarray],
    y_ctr:          np.ndarray,
    y_conversion:   np.ndarray,
    feature_names:  list[str],
    batch_size:     int,
    shuffle:        bool,
):
    """
    构建同时携带 CTR 标签和 Conversion 标签的 DataLoader。

    每个 batch 返回：({feature: tensor}, y_ctr_batch, y_conversion_batch)
    """
    import torch

    class _DualLabelDS(torch.utils.data.Dataset):
        def __init__(self, xi, yc, yv, names):
            self.names   = names
            self.tensors = {k: torch.tensor(v) for k, v in xi.items()}
            self.y_ctr   = torch.tensor(yc, dtype=torch.float32)
            self.y_conv  = torch.tensor(yv, dtype=torch.float32)

        def __len__(self):
            return len(self.y_ctr)

        def __getitem__(self, idx):
            return (
                {k: self.tensors[k][idx] for k in self.names},
                self.y_ctr[idx],
                self.y_conv[idx],
            )

    ds = _DualLabelDS(model_input, y_ctr, y_conversion, feature_names)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# 主体：ESMMModel
# ---------------------------------------------------------------------------

class ESMMModel(BaseModel):
    """
    ESMM (Entire Space Multi-Task Model) Wrapper。

    训练数据格式要求
    ----------------
    X 中必须包含两列双标签：
      - label_click      (默认列名，可通过 params 覆盖)
      - label_conversion (默认列名，可通过 params 覆盖)

    label_conversion 的语义：**全空间转化标签**
      - 1 = 该样本被曝光后，最终产生了转化（不管有没有点击）
      - 0 = 未转化
      ⚠️  注意：不是「仅点击样本中是否转化」，而是曝光空间上的 CTCVR 标签。
      ⚠️  即：label_conversion = label_click × label_cvr_in_click_space

    Parameters
    ----------
    params    : 超参数字典（与 DEFAULT_PARAMS 合并）
    task      : "binary"（固定，ESMM 是二分类多任务模型）
    name      : 模型标识
    seed      : 随机种子

    使用示例
    --------
    >>> model = ESMMModel(params={"dnn_hidden_units": [256, 128]}, seed=42)
    >>> model.fit(X_train, y_dummy, X_val=X_val, y_val=y_dummy_val)
    >>> # y_dummy 可传任意值，真正的标签从 X 的 label_click/label_conversion 列读取
    >>>
    >>> pctr    = model.predict_ctr(X_test)        # pCTR
    >>> pcvr    = model.predict_cvr(X_test)        # pCVR
    >>> pctcvr  = model.predict_proba(X_test)      # pCTCVR（默认）
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "embedding_dim":     8,
        "dnn_hidden_units":  [256, 128],
        "dnn_activation":    "relu",
        "dnn_dropout":       0.0,
        "dnn_use_bn":        False,
        "ctr_loss_weight":   1.0,
        "ctcvr_loss_weight": 1.0,
        "l2_reg_embedding":  1e-6,
        "l2_reg_dnn":        0.0,
        "learning_rate":     1e-3,
        "optimizer":         "adam",
        "batch_size":        4096,
        "epochs":            50,
        "device":            "auto",
        "patience":          5,
        "min_delta":         1e-5,
        "monitor":           "val_loss",
        "monitor_mode":      "min",
        "label_click":       "label_click",
        "label_conversion":  "label_conversion",
    }

    def __init__(
        self,
        params: Optional[dict] = None,
        task:   str             = "binary",
        name:   str             = "esmm",
        seed:   int             = 42,
    ):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged, task=task, name=name, seed=seed)

        self._hp: ESMMParams = ESMMParams.from_dict(self.params)

        # 运行时状态
        self._model                             = None
        self._encoder: Optional[FeatureEncoder] = None
        self._sparse_cols: list[str]            = []
        self._dense_cols:  list[str]            = []
        self._feature_cols: list[str]           = []   # 纯特征列（排除双标签列）
        self._feature_names: list[str]          = []
        self._linear_feature_columns            = None
        self._dnn_feature_columns               = None
        self._fi:  pd.DataFrame                 = pd.DataFrame()
        self._best_epoch: int                   = 0
        self._device_str: str                   = ""

        # 每 epoch 的双任务 loss 记录
        self.history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # 1. 特征列构建（排除双标签列）
    # ------------------------------------------------------------------

    def _build_feature_columns(
        self,
        X: pd.DataFrame,
        fit_encoder: bool = True,
    ) -> tuple[list, list]:
        """
        自动分拣特征列。

        ⚠️  label_click 和 label_conversion 列会被自动排除，不进入模型特征。
        """
        ESMM, SparseFeat, DenseFeat, get_feature_names = _require_esmm()
        hp = self._hp

        # 排除标签列
        exclude = {hp.label_click, hp.label_conversion}
        self._feature_cols = [c for c in X.columns if c not in exclude]

        X_feat = X[self._feature_cols]

        self._sparse_cols = [c for c in X_feat.columns if X_feat[c].dtype.name == "category"]
        self._dense_cols  = [
            c for c in X_feat.columns
            if c not in self._sparse_cols
            and np.issubdtype(X_feat[c].dtype, np.number)
        ]

        if fit_encoder:
            self._encoder = FeatureEncoder()
            self._encoder.fit(X_feat, self._sparse_cols)

        vocab = self._encoder.vocabulary_sizes

        # ESMM 使用共享特征列（CTR 和 CVR 共享同一套 Embedding）
        feature_columns = (
            [
                SparseFeat(col, vocabulary_size=vocab[col], embedding_dim=hp.embedding_dim)
                for col in self._sparse_cols
            ]
            + [DenseFeat(col, 1) for col in self._dense_cols]
        )

        # ESMM 的 linear 和 dnn 特征列相同
        return feature_columns, feature_columns

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
        ESMM, SparseFeat, DenseFeat, get_feature_names = _require_esmm()
        feature_names = get_feature_names(self._linear_feature_columns)
        self._feature_names = feature_names

        X_enc = self._encoder.transform(X[self._feature_cols])
        return {
            name: (
                X_enc[name].values.astype(np.float32)
                if name in self._dense_cols
                else X_enc[name].values.astype(np.int32)
            )
            for name in feature_names
            if name in X_enc.columns
        }

    # ------------------------------------------------------------------
    # 4. fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,          # 兼容接口：实际标签从 X 的双标签列读取
        X_val:   Optional[pd.DataFrame] = None,
        y_val:   Optional[pd.Series]    = None,
        **kwargs,
    ) -> "ESMMModel":
        torch = _require_torch()
        ESMM, SparseFeat, DenseFeat, get_feature_names = _require_esmm()
        hp = self._hp

        # ── 随机种子 ────────────────────────────────────────────────────
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # ── 设备 ────────────────────────────────────────────────────────
        self._device_str = self._resolve_device()
        logger.info(f"[{self.name}] 设备: {self._device_str}")

        # ── 双标签校验 ──────────────────────────────────────────────────
        self._validate_labels(X_train)

        # ── 提取双标签 ──────────────────────────────────────────────────
        y_ctr_train   = X_train[hp.label_click].values.astype(np.float32)
        y_conv_train  = X_train[hp.label_conversion].values.astype(np.float32)

        self._log_label_stats(y_ctr_train, y_conv_train)

        # ── 特征列构建 ──────────────────────────────────────────────────
        lin_cols, dnn_cols = self._build_feature_columns(X_train, fit_encoder=True)
        self._linear_feature_columns = lin_cols
        self._dnn_feature_columns    = dnn_cols

        logger.info(
            f"[{self.name}] SparseFeat={len(self._sparse_cols)}, "
            f"DenseFeat={len(self._dense_cols)} | "
            f"共享 Embedding 参数量={sum(self._encoder.vocabulary_sizes.values()) * hp.embedding_dim:,}"
        )

        # ── 构建 ESMM 模型 ──────────────────────────────────────────────
        self._model = ESMM(
            dnn_feature_columns = dnn_cols,
            tower_dnn_hidden_units_esmm = hp.dnn_hidden_units,
            l2_reg_embedding    = hp.l2_reg_embedding,
            l2_reg_dnn          = hp.l2_reg_dnn,
            dnn_dropout         = hp.dnn_dropout,
            dnn_activation      = hp.dnn_activation,
            task_dnn_use        = hp.dnn_use_bn,
            init_std            = 0.0001,
            seed                = self.seed,
            device              = self._device_str,
        )

        # ── 优化器 ──────────────────────────────────────────────────────
        opt_map = {
            "adam":    torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
            "rmsprop": torch.optim.RMSprop,
            "sgd":     torch.optim.SGD,
        }
        optimizer = opt_map.get(hp.optimizer.lower(), torch.optim.Adam)(
            self._model.parameters(), lr=hp.learning_rate
        )
        bce_loss = torch.nn.BCELoss()

        # ── DataLoader ──────────────────────────────────────────────────
        train_input  = self._to_model_input(X_train)
        train_loader = _make_dual_label_loader(
            train_input, y_ctr_train, y_conv_train,
            self._feature_names, hp.batch_size, shuffle=True,
        )

        has_val    = X_val is not None
        val_loader = None
        if has_val:
            self._validate_labels(X_val)
            val_input  = self._to_model_input(X_val)
            y_ctr_val  = X_val[hp.label_click].values.astype(np.float32)
            y_conv_val = X_val[hp.label_conversion].values.astype(np.float32)
            val_loader = _make_dual_label_loader(
                val_input, y_ctr_val, y_conv_val,
                self._feature_names, hp.batch_size * 4, shuffle=False,
            )

        # ── Early Stopping + Checkpoint ─────────────────────────────────
        stopper    = EarlyStopper(hp.patience, hp.min_delta, hp.monitor_mode)
        best_state = copy.deepcopy(self._model.state_dict())

        self.history = {
            "train_loss": [], "train_ctr_loss": [], "train_ctcvr_loss": [],
            "val_loss":   [], "val_ctr_loss":   [], "val_ctcvr_loss":   [],
        }
        t_start = time.perf_counter()

        # ── 训练主循环 ──────────────────────────────────────────────────
        for epoch in range(1, hp.epochs + 1):
            tr_loss, tr_ctr, tr_ctcvr = self._train_epoch(
                train_loader, optimizer, bce_loss, hp
            )
            self.history["train_loss"].append(tr_loss)
            self.history["train_ctr_loss"].append(tr_ctr)
            self.history["train_ctcvr_loss"].append(tr_ctcvr)

            if val_loader:
                vl_loss, vl_ctr, vl_ctcvr = self._eval_epoch(val_loader, bce_loss, hp)
            else:
                vl_loss, vl_ctr, vl_ctcvr = float("nan"), float("nan"), float("nan")

            self.history["val_loss"].append(vl_loss)
            self.history["val_ctr_loss"].append(vl_ctr)
            self.history["val_ctcvr_loss"].append(vl_ctcvr)

            # 按 monitor 参数决定监控哪条曲线
            monitor_val = self._pick_monitor(
                hp.monitor, vl_loss, vl_ctr, vl_ctcvr,
                tr_loss, tr_ctr, tr_ctcvr, has_val
            )
            should_stop = stopper.step(monitor_val, epoch)

            if stopper._counter == 0:
                best_state      = copy.deepcopy(self._model.state_dict())
                self._best_epoch = epoch

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"[{self.name}] Epoch {epoch:3d}/{hp.epochs} | "
                    f"train[loss={tr_loss:.4f} ctr={tr_ctr:.4f} ctcvr={tr_ctcvr:.4f}]"
                    + (
                        f" | val[loss={vl_loss:.4f} ctr={vl_ctr:.4f} ctcvr={vl_ctcvr:.4f}]"
                        if has_val else ""
                    )
                    + (" ★" if stopper._counter == 0 else "")
                )

            if should_stop:
                logger.info(
                    f"[{self.name}] Early stopping @ epoch {epoch} | "
                    f"best epoch={self._best_epoch} | {hp.monitor}={stopper.best_value:.5f}"
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
    # 4a / 4b. 单 Epoch 训练 & 评估
    # ------------------------------------------------------------------

    def _train_epoch(
        self, loader, optimizer, bce_loss, hp: ESMMParams
    ) -> tuple[float, float, float]:
        self._model.train()
        total, total_ctr, total_ctcvr, n = 0.0, 0.0, 0.0, 0

        for x_batch, y_ctr_b, y_conv_b in loader:
            x_batch  = {k: v.to(self._device_str) for k, v in x_batch.items()}
            y_ctr_b  = y_ctr_b.to(self._device_str)
            y_conv_b = y_conv_b.to(self._device_str)

            optimizer.zero_grad()

            # ESMM 输出 [pCTR, pCTCVR]（shape: (B, 2)）
            pred = self._model(x_batch)   # (B, 2)
            p_ctr, p_ctcvr = self._split_output(pred)

            # 双任务加权损失
            loss_ctr   = bce_loss(p_ctr,   y_ctr_b)
            loss_ctcvr = bce_loss(p_ctcvr, y_conv_b)
            loss       = (hp.ctr_loss_weight   * loss_ctr
                        + hp.ctcvr_loss_weight * loss_ctcvr)

            loss.backward()
            optimizer.step()

            bs          = y_ctr_b.size(0)
            total      += loss.item()       * bs
            total_ctr  += loss_ctr.item()   * bs
            total_ctcvr+= loss_ctcvr.item() * bs
            n          += bs

        return total / n, total_ctr / n, total_ctcvr / n

    def _eval_epoch(
        self, loader, bce_loss, hp: ESMMParams
    ) -> tuple[float, float, float]:
        import torch
        self._model.eval()
        total, total_ctr, total_ctcvr, n = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for x_batch, y_ctr_b, y_conv_b in loader:
                x_batch  = {k: v.to(self._device_str) for k, v in x_batch.items()}
                y_ctr_b  = y_ctr_b.to(self._device_str)
                y_conv_b = y_conv_b.to(self._device_str)

                pred = self._model(x_batch)
                p_ctr, p_ctcvr = self._split_output(pred)

                loss_ctr   = bce_loss(p_ctr,   y_ctr_b)
                loss_ctcvr = bce_loss(p_ctcvr, y_conv_b)
                loss       = (hp.ctr_loss_weight   * loss_ctr
                            + hp.ctcvr_loss_weight * loss_ctcvr)

                bs          = y_ctr_b.size(0)
                total      += loss.item()       * bs
                total_ctr  += loss_ctr.item()   * bs
                total_ctcvr+= loss_ctcvr.item() * bs
                n          += bs

        return total / n, total_ctr / n, total_ctcvr / n

    @staticmethod
    def _split_output(pred) -> tuple:
        """
        兼容 DeepCTR-Torch ESMM 的输出格式。
        - 若输出 shape=(B,2)：第 0 列=pCTR，第 1 列=pCTCVR
        - 若输出为 list/tuple：直接解包
        """
        if isinstance(pred, (list, tuple)):
            return pred[0].squeeze(-1), pred[1].squeeze(-1)
        if pred.dim() == 2 and pred.shape[1] == 2:
            return pred[:, 0], pred[:, 1]
        # 单输出时（pCTCVR only）
        return pred.squeeze(-1), pred.squeeze(-1)

    # ------------------------------------------------------------------
    # 5. 三种预测接口
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.predict_proba(X, **kwargs)

    def predict_proba(
        self,
        X:      pd.DataFrame,
        output: Literal["pctcvr", "pctr", "pcvr"] = "pctcvr",
        **kwargs,
    ) -> np.ndarray:
        """
        默认返回 pCTCVR，兼容 CrossValidatorTrainer 和 StackingEnsemble。

        Parameters
        ----------
        output : "pctcvr" (default) | "pctr" | "pcvr"
        """
        p_ctr, p_ctcvr = self._forward_all(X)
        if output == "pctr":
            return p_ctr
        if output == "pcvr":
            # pCVR = pCTCVR / pCTR（数值稳定）
            return np.where(p_ctr > 1e-7, p_ctcvr / np.clip(p_ctr, 1e-7, 1.0), 0.0)
        return p_ctcvr   # default: pCTCVR

    def predict_ctr(self, X: pd.DataFrame) -> np.ndarray:
        """专项接口：返回 pCTR。"""
        return self.predict_proba(X, output="pctr")

    def predict_cvr(self, X: pd.DataFrame) -> np.ndarray:
        """
        专项接口：返回 pCVR（点击后转化率）。
        pCVR = pCTCVR / pCTR
        """
        return self.predict_proba(X, output="pcvr")

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        一次性返回三个分值，方便多目标排序融合。

        Returns
        -------
        pd.DataFrame with columns: ["pCTR", "pCVR", "pCTCVR"]
        """
        p_ctr, p_ctcvr = self._forward_all(X)
        p_cvr = np.where(p_ctr > 1e-7, p_ctcvr / np.clip(p_ctr, 1e-7, 1.0), 0.0)
        return pd.DataFrame({"pCTR": p_ctr, "pCVR": p_cvr, "pCTCVR": p_ctcvr})

    def _forward_all(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """批量推断，返回 (p_ctr, p_ctcvr) numpy 数组。"""
        torch = _require_torch()
        if self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

        model_input = self._to_model_input(X)
        loader      = _make_dual_label_loader(
            model_input,
            np.zeros(len(X), dtype=np.float32),
            np.zeros(len(X), dtype=np.float32),
            self._feature_names,
            self._hp.batch_size * 4,
            shuffle=False,
        )

        all_ctr, all_ctcvr = [], []
        self._model.eval()
        with torch.no_grad():
            for x_batch, _, _ in loader:
                x_batch = {k: v.to(self._device_str) for k, v in x_batch.items()}
                pred    = self._model(x_batch)
                p_ctr, p_ctcvr = self._split_output(pred)
                all_ctr.append(p_ctr.cpu().numpy())
                all_ctcvr.append(p_ctcvr.cpu().numpy())

        return (
            np.concatenate(all_ctr,   axis=0).astype(np.float64),
            np.concatenate(all_ctcvr, axis=0).astype(np.float64),
        )

    # ------------------------------------------------------------------
    # 6. 特征重要性：共享 Embedding L2 范数
    # ------------------------------------------------------------------

    def _compute_feature_importance(self) -> pd.DataFrame:
        """
        ESMM 两个任务共享 Embedding，故特征重要性即 Embedding L2 范数。
        高 L2 范数 → 该特征的表示向量更"活跃"，对双任务贡献更大。
        """
        if self._model is None:
            return pd.DataFrame()

        state   = self._model.state_dict()
        records = []

        for col in self._sparse_cols:
            key = f"embedding_dict.{col}.weight"
            if key in state:
                norm = state[key].float().norm(dim=1).mean().item()
                records.append({"feature": col, "importance": norm, "type": "sparse_shared"})

        for col in self._dense_cols:
            records.append({"feature": col, "importance": 0.0, "type": "dense"})

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
    # 7. 序列化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        保存内容
        --------
        <path>/esmm_weights.pt     → PyTorch state_dict
        <path>/esmm_encoder.json   → FeatureEncoder（词表）
        <path>/esmm_meta.json      → 超参数 + 列名 + best_epoch
        """
        torch = _require_torch()
        if self._model is None:
            raise RuntimeError("模型尚未训练，无法保存。")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.state_dict(), save_dir / "esmm_weights.pt")

        if self._encoder is not None:
            with open(save_dir / "esmm_encoder.json", "w", encoding="utf-8") as f:
                json.dump(self._encoder.to_dict(), f, ensure_ascii=False)

        meta = {
            "params":        self._hp.to_dict(),
            "task":          self.task,
            "name":          self.name,
            "seed":          self.seed,
            "sparse_cols":   self._sparse_cols,
            "dense_cols":    self._dense_cols,
            "feature_cols":  self._feature_cols,
            "feature_names": self._feature_names,
            "best_epoch":    self._best_epoch,
            "device":        self._device_str,
        }
        with open(save_dir / "esmm_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"[{self.name}] 模型已保存 → {save_dir}")

    def load(self, path: str) -> "ESMMModel":
        """从目录加载模型（须先 save 过）。"""
        torch = _require_torch()
        ESMM, SparseFeat, DenseFeat, get_feature_names = _require_esmm()

        load_dir = Path(path)

        with open(load_dir / "esmm_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._hp            = ESMMParams.from_dict(meta["params"])
        self.task           = meta["task"]
        self.name           = meta["name"]
        self.seed           = meta["seed"]
        self._sparse_cols   = meta["sparse_cols"]
        self._dense_cols    = meta["dense_cols"]
        self._feature_cols  = meta["feature_cols"]
        self._feature_names = meta["feature_names"]
        self._best_epoch    = meta["best_epoch"]
        self._device_str    = meta.get("device", "cpu")

        enc_path = load_dir / "esmm_encoder.json"
        if enc_path.exists():
            with open(enc_path, "r", encoding="utf-8") as f:
                self._encoder = FeatureEncoder.from_dict(json.load(f))

        vocab = self._encoder.vocabulary_sizes if self._encoder else {}
        hp    = self._hp

        feature_columns = (
            [SparseFeat(c, vocabulary_size=vocab[c], embedding_dim=hp.embedding_dim)
             for c in self._sparse_cols]
            + [DenseFeat(c, 1) for c in self._dense_cols]
        )
        self._linear_feature_columns = feature_columns
        self._dnn_feature_columns    = feature_columns

        self._model = ESMM(
            dnn_feature_columns            = feature_columns,
            tower_dnn_hidden_units_esmm    = hp.dnn_hidden_units,
            l2_reg_embedding               = hp.l2_reg_embedding,
            l2_reg_dnn                     = hp.l2_reg_dnn,
            dnn_dropout                    = hp.dnn_dropout,
            dnn_activation                 = hp.dnn_activation,
            task_dnn_use                   = hp.dnn_use_bn,
            seed                           = self.seed,
            device                         = self._device_str,
        )
        state = torch.load(
            load_dir / "esmm_weights.pt",
            map_location=self._device_str,
        )
        self._model.load_state_dict(state)
        self._model.eval()

        logger.info(f"[{self.name}] 模型已加载 ← {load_dir}")
        return self

    # ------------------------------------------------------------------
    # 8. 辅助方法
    # ------------------------------------------------------------------

    def _validate_labels(self, X: pd.DataFrame) -> None:
        """校验双标签列是否存在，并打印样本量统计。"""
        hp = self._hp
        missing = [c for c in [hp.label_click, hp.label_conversion] if c not in X.columns]
        if missing:
            raise ValueError(
                f"DataFrame 中缺少以下标签列：{missing}\n"
                f"ESMM 需要同时包含 '{hp.label_click}' 和 '{hp.label_conversion}'。\n"
                f"当前列：{list(X.columns)}"
            )

    def _log_label_stats(self, y_ctr: np.ndarray, y_conv: np.ndarray) -> None:
        """打印标签分布统计，帮助诊断样本选择偏差程度。"""
        n         = len(y_ctr)
        n_click   = y_ctr.sum()
        n_convert = y_conv.sum()
        ctr       = n_click  / n  if n > 0 else 0.0
        ctcvr     = n_convert / n if n > 0 else 0.0
        cvr       = n_convert / n_click if n_click > 0 else 0.0

        logger.info(
            f"[{self.name}] 标签统计 | "
            f"样本量={n:,} | "
            f"点击={int(n_click):,}({ctr:.2%}) | "
            f"转化={int(n_convert):,}(CTCVR={ctcvr:.4%}, CVR={cvr:.2%})"
        )

    @staticmethod
    def _pick_monitor(
        monitor: str,
        vl_total: float, vl_ctr: float, vl_ctcvr: float,
        tr_total: float, tr_ctr: float, tr_ctcvr: float,
        has_val:  bool,
    ) -> float:
        """根据 monitor 参数选择早停监控的值。"""
        if has_val:
            return {
                "val_loss":       vl_total,
                "val_ctr_loss":   vl_ctr,
                "val_ctcvr_loss": vl_ctcvr,
            }.get(monitor, vl_total)
        return {
            "val_loss":       tr_total,
            "val_ctr_loss":   tr_ctr,
            "val_ctcvr_loss": tr_ctcvr,
        }.get(monitor, tr_total)

    # ------------------------------------------------------------------
    # 9. 种子注入 / Optuna 搜索空间
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_key() -> str:
        return "seed"

    @staticmethod
    def suggest_params(trial) -> dict:
        """Optuna 搜索空间。"""
        hidden_choices = ["[512, 256]", "[256, 128]", "[256, 128, 64]", "[128, 64]"]
        return {
            "embedding_dim":     trial.suggest_categorical("embedding_dim", [4, 8, 16, 32]),
            "dnn_hidden_units":  trial.suggest_categorical("dnn_hidden_units", hidden_choices),
            "dnn_dropout":       trial.suggest_float("dnn_dropout", 0.0, 0.5, step=0.1),
            "l2_reg_embedding":  trial.suggest_float("l2_reg_embedding", 1e-7, 1e-4, log=True),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "ctr_loss_weight":   trial.suggest_float("ctr_loss_weight", 0.5, 2.0, step=0.5),
            "ctcvr_loss_weight": trial.suggest_float("ctcvr_loss_weight", 0.5, 2.0, step=0.5),
            "batch_size":        trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192]),
        }

    # ------------------------------------------------------------------
    # 10. __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        hp = self._hp
        return (
            f"ESMMModel("
            f"dnn={hp.dnn_hidden_units}, "
            f"emb={hp.embedding_dim}, "
            f"loss=[ctr×{hp.ctr_loss_weight}+ctcvr×{hp.ctcvr_loss_weight}], "
            f"τ={hp.temperature if hasattr(hp, 'temperature') else 'N/A'}, "
            f"device={hp.device})"
        )
