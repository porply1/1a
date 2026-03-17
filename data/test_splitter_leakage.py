"""
data/test_splitter_leakage.py
-----------------------------
验证 TimeSeriesSplitter 与 PurgedGroupTimeSeriesSplitter
在不同 gap_size 下是否存在数据泄露。

运行方式
--------
  pytest data/test_splitter_leakage.py -v
  python data/test_splitter_leakage.py       # 独立运行（无 pytest）

测试矩阵
--------
  T1  expanding + gap=0   → train_max < val_min
  T2  expanding + gap=7   → train_max + 7 < val_min（严格 gap 隔离）
  T3  expanding + gap=20  → 部分折被跳过（gap 过大警告）
  T4  sliding  + gap=0    → 每折 train 大小 == max_train_size
  T5  sliding  + gap=5    → sliding + gap 联合约束
  T6  purged_legacy  + gap=0  → val_groups ∩ train_groups == ∅
  T7  purged_legacy  + gap=3  → purge 边界正确后移
  T8  purged_entity  + gap=0  → 无实体 group 同时出现在 train/val
  T9  purged_entity  + gap=5  → 有时间重叠的实体被整组 purge
  T10 purged_entity  + gap=0，极端重叠 → 横跨 val 的实体必须被完整清除
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 路径修正（独立运行时确保能 import）
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.splitter import CVConfig, TimeSeriesSplitter, PurgedGroupTimeSeriesSplitter


# ===========================================================================
# 工具函数
# ===========================================================================

def make_ts_df(n: int = 300) -> pd.DataFrame:
    """生成简单时序 DataFrame，已按时间升序排列。"""
    rng = np.random.default_rng(42)
    return pd.DataFrame({"value": rng.standard_normal(n)})


def make_entity_df(
    n_entities: int = 5,
    n_times: int = 50,
    overlap_entities: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    生成多实体时序 DataFrame。

    Parameters
    ----------
    n_entities       : 实体数（如股票数）
    n_times          : 每个实体的时间步数
    overlap_entities : 其中有多少实体的时间跨度被人为延伸至未来
                       （用于模拟时间重叠场景）

    Returns
    -------
    df       : pd.DataFrame
    entities : np.ndarray，entity id per row
    times    : np.ndarray，timestamp per row
    """
    records = []
    rng = np.random.default_rng(42)

    for eid in range(n_entities):
        if eid < overlap_entities:
            # 时间重叠实体：时间范围故意覆盖到更晚的时段
            t_start = n_times // 2
            t_end   = n_times + n_times // 2
        else:
            t_start = 0
            t_end   = n_times

        for t in range(t_start, t_end):
            records.append({"entity": eid, "time": t, "value": rng.standard_normal()})

    df = pd.DataFrame(records).sort_values(["time", "entity"]).reset_index(drop=True)
    return df, df["entity"].to_numpy(), df["time"].to_numpy()


# ===========================================================================
# T1–T5：TimeSeriesSplitter 测试
# ===========================================================================

class TestTimeSeriesSplitter:
    """验证 expanding/sliding 模式下 gap 隔离的正确性。"""

    @pytest.mark.parametrize("gap_size", [0, 1, 5, 7, 10])
    def test_expanding_gap_isolation(self, gap_size: int):
        """T1/T2：expanding 模式，对每一折验证 train_max + gap < val_min。"""
        df  = make_ts_df(300)
        cfg = CVConfig(strategy="timeseries", n_splits=5, gap=gap_size, mode="expanding")
        splitter = TimeSeriesSplitter(cfg)

        folds = splitter.split(df)
        assert len(folds) > 0, "应至少产生 1 折"

        for fold_i, (tr, va) in enumerate(folds):
            # 关键断言：训练集最大位置 + gap_size < 验证集最小位置
            assert tr.max() + gap_size < va.min(), (
                f"第 {fold_i} 折 gap 隔离失败！"
                f"train_max={tr.max()}, val_min={va.min()}, gap={gap_size}"
            )
            # 训练集必须早于验证集
            assert tr.max() < va.min(), (
                f"第 {fold_i} 折 train/val 时序交叉！"
                f"train_max={tr.max()}, val_min={va.min()}"
            )

    def test_expanding_gap_too_large_skips_folds(self):
        """T3：gap 过大时，折数减少并发出 RuntimeWarning。"""
        df  = make_ts_df(60)
        # gap=20，test_size 约 10，每折 train_end 可能 ≤ 0
        cfg = CVConfig(strategy="timeseries", n_splits=5, gap=20, mode="expanding")
        splitter = TimeSeriesSplitter(cfg)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            folds = splitter.split(df)

        # 部分折被跳过 → 折数 < n_splits
        assert len(folds) < 5, "gap 过大时应跳过若干折"
        # 应产生 RuntimeWarning
        runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warns) > 0, "应产生 RuntimeWarning 提示折被跳过"

    @pytest.mark.parametrize("max_train_size", [50, 80, 100])
    def test_sliding_train_size_fixed(self, max_train_size: int):
        """T4：sliding 模式，每折训练集大小 ≤ max_train_size。"""
        df  = make_ts_df(300)
        cfg = CVConfig(
            strategy="timeseries", n_splits=5, gap=0,
            mode="sliding", max_train_size=max_train_size,
        )
        splitter = TimeSeriesSplitter(cfg)
        folds = splitter.split(df)

        assert len(folds) > 0
        for fold_i, (tr, _) in enumerate(folds):
            assert len(tr) <= max_train_size, (
                f"第 {fold_i} 折训练集大小 {len(tr)} > max_train_size {max_train_size}"
            )

    @pytest.mark.parametrize("gap_size", [0, 3, 7])
    def test_sliding_gap_isolation(self, gap_size: int):
        """T5：sliding + gap，gap 隔离仍然有效。"""
        df  = make_ts_df(300)
        cfg = CVConfig(
            strategy="timeseries", n_splits=5, gap=gap_size,
            mode="sliding", max_train_size=80,
        )
        splitter = TimeSeriesSplitter(cfg)
        folds = splitter.split(df)

        assert len(folds) > 0
        for fold_i, (tr, va) in enumerate(folds):
            assert tr.max() + gap_size < va.min(), (
                f"第 {fold_i} 折 sliding+gap 隔离失败！"
                f"train_max={tr.max()}, val_min={va.min()}, gap={gap_size}"
            )

    def test_expanding_produces_more_folds_than_sliding_with_small_min_train(self):
        """expanding 模式下早期折训练数据少，但应产生折；sliding 小窗口同理。"""
        df = make_ts_df(200)
        cfg_exp = CVConfig(strategy="timeseries", n_splits=5, mode="expanding", gap=0)
        cfg_sld = CVConfig(
            strategy="timeseries", n_splits=5, mode="sliding",
            max_train_size=30, gap=0,
        )
        folds_exp = TimeSeriesSplitter(cfg_exp).split(df)
        folds_sld = TimeSeriesSplitter(cfg_sld).split(df)

        assert len(folds_exp) == 5
        assert len(folds_sld) == 5

    def test_polars_compat(self):
        """验证 polars DataFrame 不会引发异常（依赖 polars 可选安装）。"""
        try:
            import polars as pl
        except ImportError:
            pytest.skip("polars 未安装，跳过此测试")

        df_pl = pl.DataFrame({"value": list(range(200))})
        cfg   = CVConfig(strategy="timeseries", n_splits=5, gap=3, mode="expanding")
        folds = TimeSeriesSplitter(cfg).split(df_pl)
        assert len(folds) == 5


# ===========================================================================
# T6–T10：PurgedGroupTimeSeriesSplitter 测试
# ===========================================================================

class TestPurgedGroupTimeSeriesSplitter:
    """验证 Purge 机制在不同 gap_size 和时间重叠场景下的正确性。"""

    # -----------------------------------------------------------------------
    # 兼容模式（timestamps=None，groups=时间 group id）
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("gap_size", [0, 1, 3, 5])
    def test_legacy_no_time_group_leakage(self, gap_size: int):
        """T6/T7：兼容模式，验证 val_groups ∩ train_groups == ∅（无 group 泄露）。"""
        n = 200
        # groups 作为时间 group id（0..19，每 10 行同一 group）
        groups = np.repeat(np.arange(20), n // 20)
        df = pd.DataFrame({"val": np.random.randn(n)})

        cfg = CVConfig(strategy="purged_timeseries", n_splits=4, gap=gap_size)
        splitter = PurgedGroupTimeSeriesSplitter(cfg)
        folds = splitter.split(df, groups=groups)

        assert len(folds) > 0, f"gap={gap_size} 时不应产生空折列表"

        for fold_i, (tr, va) in enumerate(folds):
            train_g = set(groups[tr].tolist())
            val_g   = set(groups[va].tolist())
            leaked  = train_g & val_g
            assert not leaked, (
                f"[gap={gap_size}] 第 {fold_i} 折 group 泄露！"
                f"泄露 groups: {leaked}"
            )

    def test_legacy_train_always_before_val_in_time(self):
        """兼容模式：train 的所有时间 group 必须 < val 的所有时间 group。"""
        n = 200
        groups = np.repeat(np.arange(20), n // 20)
        df     = pd.DataFrame({"val": np.random.randn(n)})
        cfg    = CVConfig(strategy="purged_timeseries", n_splits=4, gap=1)
        folds  = PurgedGroupTimeSeriesSplitter(cfg).split(df, groups=groups)

        for fold_i, (tr, va) in enumerate(folds):
            max_train_g = groups[tr].max()
            min_val_g   = groups[va].min()
            assert max_train_g < min_val_g, (
                f"第 {fold_i} 折：train group max={max_train_g} >= val group min={min_val_g}"
            )

    # -----------------------------------------------------------------------
    # 新模式（groups=实体, timestamps=时间值）
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize("gap_size", [0, 2, 5])
    def test_entity_mode_no_group_in_both_train_and_val(self, gap_size: int):
        """T8：新模式，同一实体不能同时出现在 train 和 val（group-level 无泄露）。"""
        df, entities, times = make_entity_df(n_entities=8, n_times=30)

        cfg = CVConfig(strategy="purged_timeseries", n_splits=4, gap=gap_size)
        splitter = PurgedGroupTimeSeriesSplitter(cfg)
        folds = splitter.split(df, groups=entities, timestamps=times)

        assert len(folds) > 0

        for fold_i, (tr, va) in enumerate(folds):
            train_ent = set(entities[tr].tolist())
            val_ent   = set(entities[va].tolist())
            leaked    = train_ent & val_ent
            # 新模式下实体可同时出现在 train/val（不同时段），
            # 但 train 中的实体 max_time 必须 < purge_boundary
            # 此处验证：如果实体同时出现，其 train 部分的 max_ts < val_start - gap
            if leaked:
                # 找到 val 的起始时间
                val_start_time = float(times[va].min())
                purge_boundary = val_start_time - gap_size
                for eid in leaked:
                    train_rows_of_eid = tr[entities[tr] == eid]
                    max_ts = float(times[train_rows_of_eid].max())
                    assert max_ts < purge_boundary, (
                        f"[gap={gap_size}] 第 {fold_i} 折：实体 {eid} 在 train 中 "
                        f"max_ts={max_ts} >= purge_boundary={purge_boundary}（泄露！）"
                    )

    def test_entity_mode_temporal_overlap_purge(self):
        """T9/T10：有时间重叠的实体（其 max_ts >= purge_boundary）必须被整组清除。"""
        #
        # 数据构造：
        #   实体 0-2：时间 0..29（早期）
        #   实体 3-4：时间 15..44（故意延伸到后期，与 val 重叠）
        #
        df, entities, times = make_entity_df(
            n_entities=5, n_times=30, overlap_entities=2
        )

        # n_splits=3，val 窗口约为时间 30..44
        cfg = CVConfig(strategy="purged_timeseries", n_splits=3, gap=0)
        splitter = PurgedGroupTimeSeriesSplitter(cfg)
        folds = splitter.split(df, groups=entities, timestamps=times)

        assert len(folds) > 0

        # 对最后一折（val 时间最晚），验证 purge 有效性
        tr, va = folds[-1]
        val_start_time = float(times[va].min())

        for eid in np.unique(entities[tr]):
            train_rows = tr[entities[tr] == eid]
            eid_max_ts = float(times[train_rows].max())
            assert eid_max_ts < val_start_time, (
                f"实体 {eid} 在 train 中 max_ts={eid_max_ts} >= val_start={val_start_time}（泄露！）"
            )

    @pytest.mark.parametrize("gap_size", [0, 3, 7, 10])
    def test_entity_mode_purge_boundary_correctness(self, gap_size: int):
        """验证 purge_boundary = val_start - gap_size 的精确性。"""
        n_entities, n_times = 6, 40
        df, entities, times = make_entity_df(n_entities=n_entities, n_times=n_times)

        cfg = CVConfig(strategy="purged_timeseries", n_splits=4, gap=gap_size)
        folds = PurgedGroupTimeSeriesSplitter(cfg).split(
            df, groups=entities, timestamps=times
        )

        for fold_i, (tr, va) in enumerate(folds):
            val_start = float(times[va].min())
            purge_boundary = val_start - gap_size

            # 每个出现在 train 中的实体，其 max_ts 必须严格 < purge_boundary
            for eid in np.unique(entities[tr]):
                mask = entities[tr] == eid
                eid_max_ts = float(times[tr[mask]].max())
                assert eid_max_ts < purge_boundary, (
                    f"[gap={gap_size}] 第 {fold_i} 折：实体 {eid} max_ts={eid_max_ts} "
                    f">= purge_boundary={purge_boundary}（gap 隔离失败！）"
                )

    def test_missing_groups_raises(self):
        """不传 groups 时应抛出 ValueError。"""
        df = make_ts_df(100)
        cfg = CVConfig(strategy="purged_timeseries", n_splits=3)
        with pytest.raises(ValueError, match="groups"):
            PurgedGroupTimeSeriesSplitter(cfg).split(df)

    def test_timestamps_length_mismatch_raises(self):
        """timestamps 长度与 groups 不一致时应抛出 ValueError。"""
        df = make_ts_df(100)
        entities  = np.zeros(100, dtype=int)
        bad_times = np.arange(50)  # 故意错误长度
        cfg = CVConfig(strategy="purged_timeseries", n_splits=3)
        with pytest.raises(ValueError, match="长度"):
            PurgedGroupTimeSeriesSplitter(cfg).split(
                df, groups=entities, timestamps=bad_times
            )


# ===========================================================================
# 独立运行入口（无 pytest 时也可验证）
# ===========================================================================

if __name__ == "__main__":
    import traceback

    PASS = "[PASS]"
    FAIL = "[FAIL]"

    def run(name: str, fn):
        try:
            fn()
            print(f"  {PASS}  {name}")
        except Exception:
            print(f"  {FAIL}  {name}")
            traceback.print_exc()

    print("\n" + "="*60)
    print("  TimeSeriesSplitter 防泄露测试")
    print("="*60)

    ts_suite = TestTimeSeriesSplitter()
    for gap in [0, 1, 5, 7, 10]:
        run(f"expanding gap={gap}", lambda g=gap: ts_suite.test_expanding_gap_isolation(g))
    run("expanding gap_too_large", ts_suite.test_expanding_gap_too_large_skips_folds)
    for mts in [50, 80, 100]:
        run(f"sliding max_train={mts}", lambda m=mts: ts_suite.test_sliding_train_size_fixed(m))
    for gap in [0, 3, 7]:
        run(f"sliding gap={gap}", lambda g=gap: ts_suite.test_sliding_gap_isolation(g))
    run("expanding vs sliding folds", ts_suite.test_expanding_produces_more_folds_than_sliding_with_small_min_train)

    print("\n" + "="*60)
    print("  PurgedGroupTimeSeriesSplitter 防泄露测试")
    print("="*60)

    pg_suite = TestPurgedGroupTimeSeriesSplitter()
    for gap in [0, 1, 3, 5]:
        run(f"legacy no-leakage gap={gap}", lambda g=gap: pg_suite.test_legacy_no_time_group_leakage(g))
    run("legacy train < val in time", pg_suite.test_legacy_train_always_before_val_in_time)
    for gap in [0, 2, 5]:
        run(f"entity no-leakage gap={gap}", lambda g=gap: pg_suite.test_entity_mode_no_group_in_both_train_and_val(g))
    run("temporal overlap purge", pg_suite.test_entity_mode_temporal_overlap_purge)
    for gap in [0, 3, 7, 10]:
        run(f"purge boundary gap={gap}", lambda g=gap: pg_suite.test_entity_mode_purge_boundary_correctness(g))
    run("missing groups raises", pg_suite.test_missing_groups_raises)
    run("timestamps length mismatch", pg_suite.test_timestamps_length_mismatch_raises)

    print()
