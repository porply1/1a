"""
models/deep/deepfm_wrapper.py
------------------------------
生产级 DeepFM Wrapper，基于 DeepCTR-Torch。

依赖
----
  pip install deepctr-torch torch

特性
----
- 继承 BaseModel，与 CrossValidatorTrainer / StackingEnsemble 完全兼容
- 自动特征分拣：category → SparseFeat(Embedding)，numeric → DenseFeat
- Label Encoding 持久化，保证 train/val/test 编码一致
- Early Stopping + Model Checkpoint（保存最优 val 分数的权重）
- GPU 自动检测 / 手动指定
- 特征重要性：Embedding L2 范数 + 线性层梯度归因
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.base_model import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 可选依赖守卫
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch 未安装。请执行：\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
            "  # CPU 版本：pip install torch"
        )

def _require_deepctr():
    try:
        from deepctr_torch.models import DeepFM
        from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
        return DeepFM, SparseFeat, DenseFeat, get_feature_names
    except ImportError:
        raise ImportError(
            "DeepCTR-Torch 未安装。请执行：\n"
            "  pip install deepctr-torch"
        )


# ---------------------------------------------------------------------------
# 超参数容器
# ---------------------------------------------------------------------------

@dataclass
class DeepFMParams:
    """DeepFM 全量超参数，支持 dict 初始化。"""

    # ── Embedding ──────────────────────────────────────────────────────────
    embedding_dim: int   = 8        # SparseFeat 统一 Embedding 维度

    # ── DNN 结构 ───────────────────────────────────────────────────────────
    dnn_hidden_units: tuple  = (256, 128)
    dnn_activation: str      = "relu"
    dnn_dropout: float       = 0.0
    dnn_use_bn: bool         = False

    # ── 正则化 ─────────────────────────────────────────────────────────────
    l2_reg_linear:    float  = 1e-5
    l2_reg_embedding: float  = 1e-6
    l2_reg_dnn:       float  = 0.0

    # ── 优化器 ─────────────────────────────────────────────────────────────
    learning_rate: float     = 1e-3
    optimizer:     str       = "adam"   # adam | adagrad | rmsprop | sgd

    # ── 训练控制 ───────────────────────────────────────────────────────────
    batch_size: int     = 4096
    epochs:     int     = 50
    device:     str     = "auto"     # "auto" | "cpu" | "cuda" | "cuda:0"

    # ── Early Stopping ─────────────────────────────────────────────────────
    patience:        int    = 5
    min_delta:       float  = 1e-5
    monitor:         str    = "val_loss"  # 监控指标键名
    monitor_mode:    str    = "min"       # "min" | "max"

    @classmethod
    def from_dict(cls, d: dict) -> "DeepFMParams":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        obj = cls(**valid)
        if isinstance(obj.dnn_hidden_units, list):
            obj.dnn_hidden_units = tuple(obj.dnn_hidden_units)
        return obj

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Early Stopping 与 Checkpoint 逻辑
# ---------------------------------------------------------------------------

class EarlyStopper:
    """
    标准 Early Stopping。

    Parameters
    ----------
    patience   : 允许没有改善的 epoch 数
    min_delta  : 最小有效改善量
    mode       : "min"（越小越好）或 "max"（越大越好）
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-5, mode: str = "min"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self._counter   = 0
        self._best      = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0

    def step(self, value: float, epoch: int) -> bool:
        """返回 True 表示应当停止训练。"""
        improved = (
            value < self._best - self.min_delta
            if self.mode == "min"
            else value > self._best + self.min_delta
        )
        if improved:
            self._best    = value
            self._counter = 0
            self.best_epoch = epoch
        else:
            self._counter += 1

        return self._counter >= self.patience

    @property
    def best_value(self) -> float:
        return self._best


# ---------------------------------------------------------------------------
# 特征编码器（Label Encoding + 词表管理）
# ---------------------------------------------------------------------------

class FeatureEncoder:
    """
    管理 SparseFeat 列的 LabelEncoding 和词表大小。

    - fit 时记录每列的唯一值映射
    - transform 时将类别转换为 int index（未见词 → 0，从 1 开始编码）
    - vocabulary_sizes 字典供 SparseFeat 初始化
    """

    def __init__(self):
        self._encoders:        dict[str, dict] = {}
        self._vocabulary_sizes: dict[str, int]  = {}

    def fit(self, X: pd.DataFrame, cat_cols: list[str]) -> "FeatureEncoder":
        for col in cat_cols:
            uniques = X[col].dropna().unique().tolist()
            mapping = {v: i + 1 for i, v in enumerate(sorted(map(str, uniques)))}
            self._encoders[col]         = mapping
            self._vocabulary_sizes[col] = len(mapping) + 1   # +1 for unknown
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self._encoders.items():
            if col not in X.columns:
                X[col] = 0
                continue
            X[col] = (
                X[col]
                .astype(str)
                .map(mapping)
                .fillna(0)
                .astype(np.int32)
            )
        return X

    def fit_transform(self, X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
        return self.fit(X, cat_cols).transform(X)

    @property
    def vocabulary_sizes(self) -> dict[str, int]:
        return self._vocabulary_sizes

    def to_dict(self) -> dict:
        return {
            "encoders":        self._encoders,
            "vocabulary_sizes": self._vocabulary_sizes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureEncoder":
        obj = cls()
        obj._encoders         = d["encoders"]
        obj._vocabulary_sizes = d["vocabulary_sizes"]
        return obj


# ---------------------------------------------------------------------------
# 主体：DeepFMModel
# ---------------------------------------------------------------------------

class DeepFMModel(BaseModel):
    """
    DeepFM Wrapper。

    Parameters
    ----------
    params : dict
        超参数字典，会与 DEFAULT_PARAMS 合并。
    task   : str
        "binary" | "regression" | "multiclass"（DeepFM 暂不支持多分类，建议用 binary）
    name   : str
        模型标识，用于日志和保存文件名。
    seed   : int
        全局随机种子。

    使用示例
    --------
    >>> model = DeepFMModel(task="binary", seed=42)
    >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    >>> proba = model.predict_proba(X_test)
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "embedding_dim":    8,
        "dnn_hidden_units": [256, 128],
        "dnn_activation":   "relu",
        "dnn_dropout":      0.0,
        "dnn_use_bn":       False,
        "l2_reg_linear":    1e-5,
        "l2_reg_embedding": 1e-6,
        "l2_reg_dnn":       0.0,
        "learning_rate":    1e-3,
        "optimizer":        "adam",
        "batch_size":       4096,
        "epochs":           50,
        "device":           "auto",
        "patience":         5,
        "min_delta":        1e-5,
        "monitor":          "val_loss",
        "monitor_mode":     "min",
    }

    def __init__(
        self,
        params: Optional[dict] = None,
        task:   str             = "binary",
        name:   str             = "deepfm",
        seed:   int             = 42,
    ):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        super().__init__(params=merged, task=task, name=name, seed=seed)

        self._hp: DeepFMParams = DeepFMParams.from_dict(self.params)

        # 运行时状态（fit 后填充）
        self._model                         = None
        self._encoder: Optional[FeatureEncoder] = None
        self._sparse_cols:  list[str]       = []
        self._dense_cols:   list[str]       = []
        self._linear_feature_columns        = None
        self._dnn_feature_columns           = None
        self._feature_names:  list[str]     = []
        self._fi: pd.DataFrame              = pd.DataFrame()
        self._best_epoch: int               = 0
        self._device_str: str               = ""

        # 训练历史
        self.history: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # 1. 特征分拣
    # ------------------------------------------------------------------

    def _build_feature_columns(
        self,
        X: pd.DataFrame,
        fit_encoder: bool = True,
    ) -> tuple[list, list]:
        """
        自动分拣特征列，返回 (linear_feature_columns, dnn_feature_columns)。

        Rules
        -----
        - dtype == category  → SparseFeat（整数编码 + Embedding）
        - 数值型 (int/float) → DenseFeat（dimension=1）
        """
        _, SparseFeat, DenseFeat, _ = _require_deepctr()
        hp = self._hp

        self._sparse_cols = [c for c in X.columns if X[c].dtype.name == "category"]
        self._dense_cols  = [
            c for c in X.columns
            if c not in self._sparse_cols
            and np.issubdtype(X[c].dtype, np.number)
        ]

        # Label Encoding
        if fit_encoder:
            self._encoder = FeatureEncoder()
            self._encoder.fit(X, self._sparse_cols)

        vocab_sizes = self._encoder.vocabulary_sizes

        linear_cols = (
            [
                SparseFeat(
                    col,
                    vocabulary_size = vocab_sizes[col],
                    embedding_dim   = hp.embedding_dim,
                )
                for col in self._sparse_cols
            ]
            + [DenseFeat(col, 1) for col in self._dense_cols]
        )
        dnn_cols = linear_cols   # DeepFM 的 linear 与 DNN 共享特征列

        return linear_cols, dnn_cols

    # ------------------------------------------------------------------
    # 2. 设备选择
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        torch = _require_torch()
        if self._hp.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self._hp.device

    # ------------------------------------------------------------------
    # 3. 将 DataFrame 转为 DeepCTR 所需输入格式
    # ------------------------------------------------------------------

    def _to_model_input(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """将 encoded DataFrame 转为 {feature_name: np.array} 字典。"""
        _, _, _, get_feature_names = _require_deepctr()
        feature_names = get_feature_names(self._linear_feature_columns)
        self._feature_names = feature_names

        X_enc = self._encoder.transform(X)
        return {
            name: X_enc[name].values.astype(np.float32)
                  if name in self._dense_cols
                  else X_enc[name].values.astype(np.int32)
            for name in feature_names
        }

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
    ) -> "DeepFMModel":
        torch = _require_torch()
        DeepFM, SparseFeat, DenseFeat, get_feature_names = _require_deepctr()
        hp = self._hp

        # ── 随机种子 ────────────────────────────────────────────────────
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # ── 设备 ────────────────────────────────────────────────────────
        self._device_str = self._resolve_device()
        logger.info(f"[{self.name}] 使用设备: {self._device_str}")

        # ── 特征列构建 + 编码 ───────────────────────────────────────────
        lin_cols, dnn_cols = self._build_feature_columns(X_train, fit_encoder=True)
        self._linear_feature_columns = lin_cols
        self._dnn_feature_columns    = dnn_cols

        logger.info(
            f"[{self.name}] SparseFeat={len(self._sparse_cols)}, "
            f"DenseFeat={len(self._dense_cols)}"
        )

        # ── 任务映射 ────────────────────────────────────────────────────
        deepctr_task = {
            "binary":     "binary",
            "regression": "regression",
            "multiclass": "binary",   # DeepFM 暂不支持多分类，保留兼容
        }.get(self.task, "binary")

        # ── 构建模型 ────────────────────────────────────────────────────
        model_kwargs = dict(
            linear_feature_columns = lin_cols,
            dnn_feature_columns    = dnn_cols,
            dnn_hidden_units       = hp.dnn_hidden_units,
            dnn_activation         = hp.dnn_activation,
            dnn_dropout            = hp.dnn_dropout,
            dnn_use_bn             = hp.dnn_use_bn,
            l2_reg_linear          = hp.l2_reg_linear,
            l2_reg_embedding       = hp.l2_reg_embedding,
            l2_reg_dnn             = hp.l2_reg_dnn,
            init_std               = 0.0001,
            seed                   = self.seed,
            task                   = deepctr_task,
            device                 = self._device_str,
        )
        self._model = DeepFM(**model_kwargs)

        # ── 优化器 ──────────────────────────────────────────────────────
        opt_map = {
            "adam":    torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
            "rmsprop": torch.optim.RMSprop,
            "sgd":     torch.optim.SGD,
        }
        OptimizerCls = opt_map.get(hp.optimizer.lower(), torch.optim.Adam)
        optimizer = OptimizerCls(self._model.parameters(), lr=hp.learning_rate)

        # ── Loss ────────────────────────────────────────────────────────
        loss_fn = (
            torch.nn.BCELoss()
            if deepctr_task == "binary"
            else torch.nn.MSELoss()
        )

        # ── 准备数据 ────────────────────────────────────────────────────
        model_input_train = self._to_model_input(X_train)
        y_train_np = y_train.values.astype(np.float32)
        train_ds   = _to_tensor_dataset(model_input_train, y_train_np, self._feature_names)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size = hp.batch_size,
            shuffle    = True,
        )

        has_val = X_val is not None and y_val is not None
        if has_val:
            model_input_val = self._to_model_input(X_val)
            y_val_np        = y_val.values.astype(np.float32)
            val_ds          = _to_tensor_dataset(model_input_val, y_val_np, self._feature_names)
            val_loader      = torch.utils.data.DataLoader(
                val_ds,
                batch_size = hp.batch_size * 4,
                shuffle    = False,
            )
        else:
            val_loader = None

        # ── Early Stopping + Checkpoint ─────────────────────────────────
        stopper    = EarlyStopper(
            patience  = hp.patience,
            min_delta = hp.min_delta,
            mode      = hp.monitor_mode,
        )
        best_state = copy.deepcopy(self._model.state_dict())

        self.history = {"train_loss": [], "val_loss": []}
        t_start = time.perf_counter()

        # ── 训练主循环 ──────────────────────────────────────────────────
        for epoch in range(1, hp.epochs + 1):
            train_loss = self._train_epoch(
                self._model, train_loader, loss_fn, optimizer,
                deepctr_task, self._device_str
            )
            self.history["train_loss"].append(train_loss)

            val_loss = (
                self._eval_epoch(self._model, val_loader, loss_fn,
                                 deepctr_task, self._device_str)
                if val_loader else float("nan")
            )
            self.history["val_loss"].append(val_loss)

            monitor_val = val_loss if has_val else train_loss
            should_stop = stopper.step(monitor_val, epoch)

            # 更新最优 checkpoint
            if stopper._counter == 0:          # 刚刚改善
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
                    f"[{self.name}] Early stopping at epoch {epoch} | "
                    f"best epoch={self._best_epoch} | "
                    f"best {hp.monitor}={stopper.best_value:.5f}"
                )
                break

        # ── 恢复最优权重 ────────────────────────────────────────────────
        self._model.load_state_dict(best_state)
        elapsed = time.perf_counter() - t_start
        logger.info(
            f"[{self.name}] 训练完成 | 最优 Epoch={self._best_epoch} | 耗时={elapsed:.1f}s"
        )

        # ── 计算特征重要性 ───────────────────────────────────────────────
        self._fi = self._compute_feature_importance()
        return self

    # ------------------------------------------------------------------
    # 4a. 单 Epoch 训练
    # ------------------------------------------------------------------

    @staticmethod
    def _train_epoch(model, loader, loss_fn, optimizer, task, device) -> float:
        import torch
        model.train()
        total_loss = 0.0
        for batch in loader:
            x_batch, y_batch = batch
            x_batch = {k: v.to(device) for k, v in x_batch.items()}
            y_batch  = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch).squeeze(-1)
            if task == "binary":
                pred = torch.sigmoid(pred)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)

        return total_loss / len(loader.dataset)

    @staticmethod
    def _eval_epoch(model, loader, loss_fn, task, device) -> float:
        import torch
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                x_batch, y_batch = batch
                x_batch = {k: v.to(device) for k, v in x_batch.items()}
                y_batch  = y_batch.to(device)
                pred = model(x_batch).squeeze(-1)
                if task == "binary":
                    pred = torch.sigmoid(pred)
                loss = loss_fn(pred, y_batch)
                total_loss += loss.item() * len(y_batch)
        return total_loss / len(loader.dataset)

    # ------------------------------------------------------------------
    # 5. predict / predict_proba
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        返回最终预测值。
        - binary     → sigmoid 概率
        - regression → 原始输出
        """
        return self.predict_proba(X, **kwargs)

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        返回正例概率（或回归值）。
        与 CrossValidatorTrainer 的 OOF 逻辑完全兼容。
        """
        torch = _require_torch()
        if self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

        model_input = self._to_model_input(X)
        ds = _to_tensor_dataset(
            model_input,
            np.zeros(len(X), dtype=np.float32),
            self._feature_names,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=self._hp.batch_size * 4, shuffle=False
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
    # 6. 特征重要性：Embedding L2 范数 + 线性层权重归因
    # ------------------------------------------------------------------

    def _compute_feature_importance(self) -> pd.DataFrame:
        """
        Sparse → Embedding 矩阵的 L2 范数（越大说明该特征的表示越"活跃"）
        Dense  → 线性层对应权重的绝对值
        """
        torch = _require_torch()
        if self._model is None:
            return pd.DataFrame()

        records: list[dict] = []

        # ── Sparse: Embedding L2 norm ───────────────────────────────────
        state = self._model.state_dict()
        for feat_name in self._sparse_cols:
            key = f"embedding_dict.{feat_name}.weight"
            if key in state:
                emb_weight = state[key]                    # (vocab, emb_dim)
                norm = emb_weight.float().norm(dim=1).mean().item()
                records.append({"feature": feat_name, "importance": norm, "type": "sparse"})

        # ── Dense: 线性层第一层权重绝对均值 ─────────────────────────────
        # DeepFM linear_model 对 DenseFeat 有单独线性权重
        for i, feat_name in enumerate(self._dense_cols):
            key = f"linear_model.dense_weight"
            if key in state:
                w = state[key].float()
                if w.shape[1] > i:
                    records.append({
                        "feature":    feat_name,
                        "importance": w[:, i].abs().mean().item(),
                        "type":       "dense",
                    })
            else:
                records.append({"feature": feat_name, "importance": 0.0, "type": "dense"})

        if not records:
            return pd.DataFrame(columns=["feature", "importance", "type"])

        fi = pd.DataFrame(records)

        # ── 归一化到 [0, 1] ──────────────────────────────────────────────
        max_imp = fi["importance"].max()
        if max_imp > 0:
            fi["importance"] = fi["importance"] / max_imp
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        fi.rename(columns={"importance": "importance_mean"}, inplace=True)
        return fi

    @property
    def feature_importance(self) -> pd.DataFrame:
        """返回归一化后的特征重要性 DataFrame（兼容 BaseTrainer）。"""
        return self._fi

    # ------------------------------------------------------------------
    # 7. 序列化
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        保存内容：
        - <path>/deepfm_weights.pt      → PyTorch state_dict
        - <path>/deepfm_encoder.json    → FeatureEncoder（词表）
        - <path>/deepfm_meta.json       → 超参数 + 列名 + 最优 epoch
        """
        torch = _require_torch()
        if self._model is None:
            raise RuntimeError("模型尚未训练，无法保存。")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 权重
        torch.save(self._model.state_dict(), save_dir / "deepfm_weights.pt")

        # 编码器
        if self._encoder is not None:
            with open(save_dir / "deepfm_encoder.json", "w", encoding="utf-8") as f:
                json.dump(self._encoder.to_dict(), f, ensure_ascii=False)

        # 元数据
        meta = {
            "params":       self._hp.to_dict(),
            "task":         self.task,
            "name":         self.name,
            "seed":         self.seed,
            "sparse_cols":  self._sparse_cols,
            "dense_cols":   self._dense_cols,
            "feature_names": self._feature_names,
            "best_epoch":   self._best_epoch,
            "device":       self._device_str,
        }
        with open(save_dir / "deepfm_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"[{self.name}] 模型已保存 → {save_dir}")

    def load(self, path: str) -> "DeepFMModel":
        """从目录加载模型（须先 save 过）。"""
        torch = _require_torch()
        DeepFM, SparseFeat, DenseFeat, get_feature_names = _require_deepctr()

        load_dir = Path(path)

        # ── 元数据 ──────────────────────────────────────────────────────
        with open(load_dir / "deepfm_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        self._hp            = DeepFMParams.from_dict(meta["params"])
        self.task           = meta["task"]
        self.name           = meta["name"]
        self.seed           = meta["seed"]
        self._sparse_cols   = meta["sparse_cols"]
        self._dense_cols    = meta["dense_cols"]
        self._feature_names = meta["feature_names"]
        self._best_epoch    = meta["best_epoch"]
        self._device_str    = meta.get("device", "cpu")

        # ── 编码器 ──────────────────────────────────────────────────────
        enc_path = load_dir / "deepfm_encoder.json"
        if enc_path.exists():
            with open(enc_path, "r", encoding="utf-8") as f:
                self._encoder = FeatureEncoder.from_dict(json.load(f))

        # ── 重建模型结构 ────────────────────────────────────────────────
        vocab_sizes = self._encoder.vocabulary_sizes if self._encoder else {}
        lin_cols = (
            [
                SparseFeat(col, vocabulary_size=vocab_sizes[col],
                           embedding_dim=self._hp.embedding_dim)
                for col in self._sparse_cols
            ]
            + [DenseFeat(col, 1) for col in self._dense_cols]
        )
        self._linear_feature_columns = lin_cols
        self._dnn_feature_columns    = lin_cols

        deepctr_task = "binary" if self.task in ("binary", "multiclass") else "regression"
        self._model = DeepFM(
            linear_feature_columns = lin_cols,
            dnn_feature_columns    = lin_cols,
            dnn_hidden_units       = self._hp.dnn_hidden_units,
            dnn_activation         = self._hp.dnn_activation,
            dnn_dropout            = self._hp.dnn_dropout,
            dnn_use_bn             = self._hp.dnn_use_bn,
            l2_reg_linear          = self._hp.l2_reg_linear,
            l2_reg_embedding       = self._hp.l2_reg_embedding,
            l2_reg_dnn             = self._hp.l2_reg_dnn,
            seed                   = self.seed,
            task                   = deepctr_task,
            device                 = self._device_str,
        )

        # ── 权重 ─────────────────────────────────────────────────────────
        state = torch.load(
            load_dir / "deepfm_weights.pt",
            map_location=self._device_str,
        )
        self._model.load_state_dict(state)
        self._model.eval()

        logger.info(f"[{self.name}] 模型已加载 ← {load_dir}")
        return self

    # ------------------------------------------------------------------
    # 8. 种子注入（BaseModel 约定）
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_key() -> str:
        return "seed"   # DeepCTR-Torch 构造函数中的 seed 参数名

    # ------------------------------------------------------------------
    # 9. Optuna 搜索空间（供 OptunaTuner 调用）
    # ------------------------------------------------------------------

    @staticmethod
    def suggest_params(trial) -> dict:
        """
        Optuna 超参数搜索空间。
        由 OptunaTuner 在 objective 函数内调用。
        """
        hidden_choices = [
            [512, 256],
            [256, 128],
            [256, 128, 64],
            [128, 64],
        ]
        return {
            "embedding_dim": trial.suggest_categorical("embedding_dim", [4, 8, 16, 32]),
            "dnn_hidden_units": trial.suggest_categorical(
                "dnn_hidden_units",
                [str(c) for c in hidden_choices],
            ),
            "dnn_dropout":     trial.suggest_float("dnn_dropout", 0.0, 0.5, step=0.1),
            "dnn_use_bn":      trial.suggest_categorical("dnn_use_bn", [True, False]),
            "l2_reg_linear":   trial.suggest_float("l2_reg_linear", 1e-6, 1e-3, log=True),
            "l2_reg_embedding":trial.suggest_float("l2_reg_embedding", 1e-7, 1e-4, log=True),
            "learning_rate":   trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size":      trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192]),
        }

    # ------------------------------------------------------------------
    # 10. __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        hp = self._hp
        return (
            f"DeepFMModel("
            f"task={self.task}, "
            f"dnn={hp.dnn_hidden_units}, "
            f"emb_dim={hp.embedding_dim}, "
            f"lr={hp.learning_rate}, "
            f"device={hp.device})"
        )


# ---------------------------------------------------------------------------
# 工具函数：DataFrame → TensorDataset
# ---------------------------------------------------------------------------

def _to_tensor_dataset(
    model_input: dict[str, np.ndarray],
    y: np.ndarray,
    feature_names: list[str],
):
    """
    将 {feature: array} 字典 + 标签数组 转为 PyTorch TensorDataset。

    返回 Dataset，每个 item 是 ({feature: tensor}, label_tensor)。
    底层用 NamedTensorDataset 避免 key 顺序依赖。
    """
    import torch

    class NamedTensorDataset(torch.utils.data.Dataset):
        def __init__(self, x_dict: dict, y: np.ndarray, names: list[str]):
            self.names = names
            self.x_tensors = {
                k: torch.tensor(v) for k, v in x_dict.items()
            }
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return {k: self.x_tensors[k][idx] for k in self.names}, self.y[idx]

    return NamedTensorDataset(model_input, y, feature_names)
