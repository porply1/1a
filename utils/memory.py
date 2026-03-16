"""
utils/memory.py
---------------
内存优化工具：dtype 自动压缩、内存用量报告、分块处理辅助。

设计原则：
  - 加载即压缩，数据不以 float64/int64 形式在内存中长期存活
  - 所有压缩操作保证数值无损（不改变语义）
  - 提供上下文管理器方便监控局部内存增量
"""

from __future__ import annotations

import gc
import os
import tracemalloc
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd
import psutil


# ---------------------------------------------------------------------------
# dtype 压缩
# ---------------------------------------------------------------------------

# 整型降级映射：按值域选最小类型
_INT_RANGES: list[tuple[int, int, str]] = [
    (np.iinfo(np.int8).min,  np.iinfo(np.int8).max,  "int8"),
    (np.iinfo(np.int16).min, np.iinfo(np.int16).max, "int16"),
    (np.iinfo(np.int32).min, np.iinfo(np.int32).max, "int32"),
]

# 浮点降级：尝试 float32，溢出则保留 float64
_FLOAT32_MAX = np.finfo(np.float32).max


def compress_dataframe(
    df: pd.DataFrame,
    *,
    compress_int: bool = True,
    compress_float: bool = True,
    compress_object: bool = True,
    category_threshold: float = 0.5,
    inplace: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    自动压缩 DataFrame 各列的 dtype 以减少内存占用。

    Parameters
    ----------
    df : pd.DataFrame
        待压缩的 DataFrame。
    compress_int : bool
        是否压缩整型列（int64 → int8/16/32）。
    compress_float : bool
        是否压缩浮点列（float64 → float32）。
    compress_object : bool
        是否将低基数 object 列转为 category。
    category_threshold : float
        唯一值比例低于此阈值时，object 列转为 category（默认 0.5）。
    inplace : bool
        是否原地修改，默认返回副本。
    verbose : bool
        打印压缩前后内存对比。

    Returns
    -------
    pd.DataFrame
        压缩后的 DataFrame。
    """
    if not inplace:
        df = df.copy()

    mem_before = df.memory_usage(deep=True).sum() / 1024 ** 2  # MB

    for col in df.columns:
        col_dtype = df[col].dtype

        # --- 整型 ---
        if compress_int and pd.api.types.is_integer_dtype(col_dtype):
            col_min = df[col].min()
            col_max = df[col].max()
            target = "int64"  # 默认兜底
            for lo, hi, dtype_name in _INT_RANGES:
                if col_min >= lo and col_max <= hi:
                    target = dtype_name
                    break
            df[col] = df[col].astype(target)

        # --- 浮点 ---
        elif compress_float and pd.api.types.is_float_dtype(col_dtype):
            col_abs_max = df[col].abs().max()
            if col_abs_max <= _FLOAT32_MAX:
                df[col] = df[col].astype(np.float32)

        # --- 字符串/object → category ---
        elif compress_object and col_dtype == object:
            n_unique = df[col].nunique(dropna=False)
            ratio = n_unique / max(len(df), 1)
            if ratio < category_threshold:
                df[col] = df[col].astype("category")

    mem_after = df.memory_usage(deep=True).sum() / 1024 ** 2

    if verbose:
        reduction = (1 - mem_after / mem_before) * 100 if mem_before > 0 else 0
        print(
            f"[memory] 压缩完成：{mem_before:.2f} MB → {mem_after:.2f} MB "
            f"（节省 {reduction:.1f}%）"
        )

    return df


def get_memory_usage_mb(df: pd.DataFrame) -> float:
    """返回 DataFrame 的深度内存用量（MB）。"""
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def get_process_memory_mb() -> float:
    """返回当前进程的 RSS 内存用量（MB）。"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2


# ---------------------------------------------------------------------------
# 内存监控上下文管理器
# ---------------------------------------------------------------------------

@contextmanager
def memory_monitor(tag: str = "block"):
    """
    监控代码块的内存增量（基于 tracemalloc）。

    Usage
    -----
    >>> with memory_monitor("feature engineering"):
    ...     df = build_features(df)
    """
    gc.collect()
    mem_before = get_process_memory_mb()
    tracemalloc.start()
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        gc.collect()
        mem_after = get_process_memory_mb()
        delta = mem_after - mem_before
        sign = "+" if delta >= 0 else ""
        print(
            f"[memory:{tag}] RSS Δ={sign}{delta:.1f} MB | "
            f"tracemalloc peak={peak / 1024 ** 2:.1f} MB"
        )


# ---------------------------------------------------------------------------
# 分块处理辅助
# ---------------------------------------------------------------------------

def iter_chunks(
    df: pd.DataFrame,
    chunk_size: int,
) -> "Generator[pd.DataFrame, None, None]":
    """
    将 DataFrame 按行分块迭代，避免一次性操作大表。

    Parameters
    ----------
    df : pd.DataFrame
    chunk_size : int
        每块的行数。

    Yields
    ------
    pd.DataFrame
        当前块（slice，非拷贝）。
    """
    n = len(df)
    for start in range(0, n, chunk_size):
        yield df.iloc[start: start + chunk_size]


def reduce_mem_usage_report(df: pd.DataFrame) -> dict:
    """
    返回每列的内存占用详情（用于 EDA 阶段诊断）。

    Returns
    -------
    dict
        {col: {"dtype": str, "mem_mb": float, "n_unique": int}}
    """
    report = {}
    for col in df.columns:
        mem_mb = df[col].memory_usage(deep=True) / 1024 ** 2
        try:
            n_unique = df[col].nunique(dropna=False)
        except TypeError:
            n_unique = -1
        report[col] = {
            "dtype": str(df[col].dtype),
            "mem_mb": round(mem_mb, 4),
            "n_unique": n_unique,
        }
    return report
