"""
data/splitter.py
----------------
交叉验证切分策略工厂。

设计目标：
  1. 工厂模式：一个 get_cv() 入口，按名称返回对应切分器
  2. 覆盖竞赛三大赛道：
       - 结构化数据  → KFold / StratifiedKFold / GroupKFold / StratifiedGroupKFold
       - 时间序列    → TimeSeriesSplit / PurgedGroupTimeSeriesSplit（防泄露）
       - 推荐系统    → UserGroupKFold（按用户切分，防泄露）
  3. 所有切分器统一返回 List[Tuple[np.ndarray, np.ndarray]] 格式（eager 求值）
  4. 提供 validate_no_leakage() 辅助函数，检查 train/val 是否有 group 泄露

使用示例
--------
>>> from data.splitter import get_cv, CVConfig

>>> # 结构化分类任务
>>> cv = get_cv(CVConfig(strategy="stratified_kfold", n_splits=5))
>>> folds = cv.split(X, y)

>>> # 时序任务（带 gap）
>>> cv = get_cv(CVConfig(strategy="purged_timeseries", n_splits=5, gap=7))
>>> folds = cv.split(X, y, groups=time_index)

>>> # 推荐系统（按用户）
>>> cv = get_cv(CVConfig(strategy="user_group_kfold", n_splits=5))
>>> folds = cv.split(X, y, groups=user_ids)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    StratifiedGroupKFold,
    TimeSeriesSplit,
)

# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------

@dataclass
class CVConfig:
    """
    交叉验证配置，传入 get_cv() 工厂。

    Parameters
    ----------
    strategy : str
        切分策略名称，见 STRATEGY_REGISTRY。
    n_splits : int
        折数，默认 5。
    shuffle : bool
        KFold / StratifiedKFold 是否打乱，默认 True。
    random_state : int
        随机种子，默认 42。
    gap : int
        时序切分中 train/val 之间的间隔样本数（防近期数据泄露），默认 0。
    max_train_size : int | None
        时序切分中训练集最大样本数（滑动窗口），None 表示不限。
    purge_pct : float
        PurgedGroupTimeSeriesSplit 中，按 group 时间排序后，
        从 val 前沿向前清除的比例，默认 0.01（1%）。
    """
    strategy: str = "stratified_kfold"
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    gap: int = 0
    max_train_size: Optional[int] = None
    purge_pct: float = 0.01


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class BaseCVSplitter:
    """所有切分器的统一接口。"""

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Returns
        -------
        list of (train_idx, val_idx) numpy arrays
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# 策略实现
# ---------------------------------------------------------------------------

class KFoldSplitter(BaseCVSplitter):
    """标准 K 折，适用于回归任务或类别均衡的分类任务。"""

    def __init__(self, cfg: CVConfig):
        self._kf = KFold(
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state if cfg.shuffle else None,
        )

    def split(self, X, y=None, groups=None):
        return list(self._kf.split(X, y))

    def __repr__(self):
        return f"KFoldSplitter(n_splits={self._kf.n_splits})"


class StratifiedKFoldSplitter(BaseCVSplitter):
    """分层 K 折，保证每折目标分布一致，适用于类别不均衡分类。"""

    def __init__(self, cfg: CVConfig):
        self._skf = StratifiedKFold(
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state if cfg.shuffle else None,
        )

    def split(self, X, y=None, groups=None):
        if y is None:
            raise ValueError("StratifiedKFoldSplitter 需要传入 y（目标列）")
        return list(self._skf.split(X, y))

    def __repr__(self):
        return f"StratifiedKFoldSplitter(n_splits={self._skf.n_splits})"


class GroupKFoldSplitter(BaseCVSplitter):
    """
    按 group 划分，同一 group 不会同时出现在 train 和 val 中。
    适用场景：同一用户/同一店铺的数据需要防泄露。
    """

    def __init__(self, cfg: CVConfig):
        self._gkf = GroupKFold(n_splits=cfg.n_splits)

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("GroupKFoldSplitter 需要传入 groups")
        return list(self._gkf.split(X, y, groups=groups))

    def __repr__(self):
        return f"GroupKFoldSplitter(n_splits={self._gkf.n_splits})"


class StratifiedGroupKFoldSplitter(BaseCVSplitter):
    """
    分层 + Group 双重约束，兼顾类别分布和 group 不泄露。
    sklearn >= 1.1 原生支持。
    """

    def __init__(self, cfg: CVConfig):
        self._sgkf = StratifiedGroupKFold(
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state if cfg.shuffle else None,
        )

    def split(self, X, y=None, groups=None):
        if y is None or groups is None:
            raise ValueError("StratifiedGroupKFoldSplitter 需要 y 和 groups")
        return list(self._sgkf.split(X, y, groups=groups))

    def __repr__(self):
        return f"StratifiedGroupKFoldSplitter(n_splits={self._sgkf.n_splits})"


class TimeSeriesSplitter(BaseCVSplitter):
    """
    标准时序切分（expanding window），不打乱顺序。
    val 集始终在 train 集之后，支持 gap 参数。

    注意：X 必须已按时间升序排列。
    """

    def __init__(self, cfg: CVConfig):
        self._tss = TimeSeriesSplit(
            n_splits=cfg.n_splits,
            gap=cfg.gap,
            max_train_size=cfg.max_train_size,
        )

    def split(self, X, y=None, groups=None):
        return list(self._tss.split(X))

    def __repr__(self):
        return (
            f"TimeSeriesSplitter(n_splits={self._tss.n_splits}, "
            f"gap={self._tss.gap})"
        )


class PurgedGroupTimeSeriesSplitter(BaseCVSplitter):
    """
    带 purge 的时序 Group 切分，Kaggle 时序竞赛防泄露核武器。

    算法：
      1. 按 groups（时间戳或时间 group id）对样本排序
      2. 将 group 按时间顺序分为 n_splits + 1 段
      3. 第 i 折：前 i 段为候选训练集，第 i+1 段为验证集
      4. 从验证集左边界向左清除 purge_n_groups 个 group（防止近邻泄露）
      5. 若设 gap，额外跳过 gap 个 group

    Parameters
    ----------
    groups : array-like
        每行对应的时间 group 标识（整数或可比较类型），
        值越大代表越晚。

    References
    ----------
    Marcos Lopez de Prado, "Advances in Financial Machine Learning", Ch.7
    """

    def __init__(self, cfg: CVConfig):
        self.n_splits = cfg.n_splits
        self.gap = cfg.gap
        self.purge_pct = cfg.purge_pct

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups: Optional[pd.Series] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("PurgedGroupTimeSeriesSplitter 需要 groups（时间 group）")

        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        if n_groups < self.n_splits + 2:
            raise ValueError(
                f"group 数量（{n_groups}）不足以切 {self.n_splits} 折，"
                f"至少需要 {self.n_splits + 2} 个 group。"
            )

        # purge 的 group 数量（至少 1）
        purge_n = max(1, int(n_groups * self.purge_pct))

        # 将 unique_groups 均匀分为 n_splits + 1 段
        group_splits = np.array_split(unique_groups, self.n_splits + 1)

        folds = []
        for fold_idx in range(self.n_splits):
            # 验证集：第 fold_idx+1 段的所有 group
            val_groups = set(group_splits[fold_idx + 1])

            # 训练集：fold_idx+1 段之前的所有 group
            train_groups_all = np.concatenate(group_splits[: fold_idx + 1])

            # 排除 purge 区域（val 左边 purge_n 个 group）
            val_min_group = min(val_groups)
            purge_groups: set = set()
            sorted_train = np.sort(train_groups_all)
            # 找出紧靠 val 左边的 purge_n 个 group
            boundary_groups = sorted_train[sorted_train < val_min_group]
            if len(boundary_groups) > 0:
                purge_groups = set(boundary_groups[-purge_n:])

            # 排除 gap 区域（purge 区域再向左 gap 个 group）
            gap_groups: set = set()
            if self.gap > 0 and len(boundary_groups) > purge_n:
                remaining = boundary_groups[: -purge_n]
                if len(remaining) > 0:
                    gap_groups = set(remaining[-self.gap:])

            excluded = purge_groups | gap_groups
            final_train_groups = set(train_groups_all) - excluded

            train_idx = np.where(np.isin(groups, list(final_train_groups)))[0]
            val_idx   = np.where(np.isin(groups, list(val_groups)))[0]

            if len(train_idx) == 0 or len(val_idx) == 0:
                warnings.warn(
                    f"第 {fold_idx} 折训练集或验证集为空，已跳过。"
                    "请检查 n_splits 与 purge_pct 设置。",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            folds.append((train_idx, val_idx))

        return folds

    def __repr__(self):
        return (
            f"PurgedGroupTimeSeriesSplitter("
            f"n_splits={self.n_splits}, "
            f"gap={self.gap}, "
            f"purge_pct={self.purge_pct})"
        )


class UserGroupKFoldSplitter(BaseCVSplitter):
    """
    推荐系统专用：按用户 ID 分折，同一用户的交互数据不会跨 train/val。

    本质上是 GroupKFold 的语义化封装，增加了用户分布均衡检查。
    """

    def __init__(self, cfg: CVConfig):
        self._gkf = GroupKFold(n_splits=cfg.n_splits)
        self.n_splits = cfg.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups: Optional[pd.Series] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("UserGroupKFoldSplitter 需要 groups（用户 ID 列）")

        n_users = len(np.unique(groups))
        if n_users < self.n_splits:
            raise ValueError(
                f"用户数（{n_users}）少于折数（{self.n_splits}），"
                "请减少 n_splits。"
            )

        folds = list(self._gkf.split(X, y, groups=groups))
        # 简单校验：每折 val 用户数应相近
        val_sizes = [len(np.unique(groups.iloc[val] if hasattr(groups, 'iloc') else groups[val]))
                     for _, val in folds]
        cv_ratio = np.std(val_sizes) / max(np.mean(val_sizes), 1)
        if cv_ratio > 0.3:
            warnings.warn(
                f"各折用户分布不均（变异系数={cv_ratio:.2f}），"
                "建议检查用户交互数据是否存在长尾分布。",
                RuntimeWarning,
                stacklevel=2,
            )
        return folds

    def __repr__(self):
        return f"UserGroupKFoldSplitter(n_splits={self._gkf.n_splits})"


# ---------------------------------------------------------------------------
# 工厂注册表
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[BaseCVSplitter]] = {
    "kfold":                      KFoldSplitter,
    "stratified_kfold":           StratifiedKFoldSplitter,
    "group_kfold":                GroupKFoldSplitter,
    "stratified_group_kfold":     StratifiedGroupKFoldSplitter,
    "timeseries":                 TimeSeriesSplitter,
    "purged_timeseries":          PurgedGroupTimeSeriesSplitter,
    "user_group_kfold":           UserGroupKFoldSplitter,
}


def get_cv(cfg: CVConfig) -> BaseCVSplitter:
    """
    工厂函数：根据 CVConfig 返回对应切分器实例。

    Parameters
    ----------
    cfg : CVConfig

    Returns
    -------
    BaseCVSplitter

    Raises
    ------
    ValueError
        当 strategy 不在注册表中时。

    Examples
    --------
    >>> cv = get_cv(CVConfig(strategy="stratified_kfold", n_splits=5))
    >>> folds = cv.split(X_train, y_train)
    >>> for fold_i, (tr_idx, val_idx) in enumerate(folds):
    ...     X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    """
    strategy = cfg.strategy.lower()
    if strategy not in STRATEGY_REGISTRY:
        valid = list(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"未知的 CV 策略：'{strategy}'。\n"
            f"可用策略：{valid}"
        )
    return STRATEGY_REGISTRY[strategy](cfg)


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def validate_no_leakage(
    folds: list[tuple[np.ndarray, np.ndarray]],
    groups: pd.Series,
) -> bool:
    """
    检查每一折的 train/val 之间是否存在 group 泄露。

    Parameters
    ----------
    folds : list of (train_idx, val_idx)
    groups : pd.Series
        与原始数据行对齐的 group 标识列。

    Returns
    -------
    bool
        True 表示无泄露，False 表示存在泄露（同时打印详情）。
    """
    groups_arr = np.asarray(groups)
    has_leakage = False

    for i, (tr_idx, val_idx) in enumerate(folds):
        train_groups = set(groups_arr[tr_idx])
        val_groups   = set(groups_arr[val_idx])
        leaked = train_groups & val_groups
        if leaked:
            has_leakage = True
            print(
                f"[leakage] 第 {i} 折发现泄露！"
                f"共 {len(leaked)} 个 group 同时出现在 train 和 val 中。"
                f"示例：{list(leaked)[:5]}"
            )

    if not has_leakage:
        print(f"[leakage] 所有 {len(folds)} 折均无 group 泄露。")
    return not has_leakage


def fold_statistics(
    folds: list[tuple[np.ndarray, np.ndarray]],
    y: pd.Series,
) -> pd.DataFrame:
    """
    打印每折的样本数和（分类任务下的）目标分布，便于验证分层效果。

    Parameters
    ----------
    folds : list of (train_idx, val_idx)
    y : pd.Series
        目标列。

    Returns
    -------
    pd.DataFrame
        每折统计摘要。
    """
    y_arr = np.asarray(y)
    is_clf = len(np.unique(y_arr)) <= 50  # 粗略判断是否为分类

    rows = []
    for i, (tr_idx, val_idx) in enumerate(folds):
        row: dict = {
            "fold":      i,
            "train_n":   len(tr_idx),
            "val_n":     len(val_idx),
        }
        if is_clf:
            val_pos_rate = y_arr[val_idx].mean()
            tr_pos_rate  = y_arr[tr_idx].mean()
            row["train_pos%"] = round(tr_pos_rate * 100, 2)
            row["val_pos%"]   = round(val_pos_rate * 100, 2)
        rows.append(row)

    return pd.DataFrame(rows).set_index("fold")
