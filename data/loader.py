"""
data/loader.py
--------------
内存优化数据加载器。

核心能力：
  1. 自动 dtype 推断与压缩（加载时即压缩，峰值内存最小化）
  2. 分块流式读取（大文件 CSV/Parquet 不撑爆内存）
  3. 统一接口：load() 屏蔽底层格式细节
  4. 特征类型自动分类（numeric / categorical / datetime / text）
  5. 轻量数据质量报告（缺失率、唯一值、分布）

支持格式：CSV, Parquet, Feather, ORC, JSON Lines
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from utils.memory import compress_dataframe, get_memory_usage_mb

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# 列类型分析
# ---------------------------------------------------------------------------

class ColumnTypeAnalyzer:
    """
    将 DataFrame 各列归类为：numeric / categorical / datetime / text / unknown。
    用于下游特征工程自动分派。
    """

    # 文本列判断：平均字符数超过此阈值视为 text
    TEXT_MEAN_LEN_THRESHOLD: int = 50

    def analyze(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """
        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        dict with keys: "numeric", "categorical", "datetime", "text", "unknown"
        """
        result: dict[str, list[str]] = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "text": [],
            "unknown": [],
        }
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                result["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                result["datetime"].append(col)
            elif dtype == "category" or dtype == object:
                # 区分短字符串（categorical）与长文本（text）
                sample = df[col].dropna().head(1000).astype(str)
                mean_len = sample.str.len().mean() if len(sample) > 0 else 0
                if mean_len >= self.TEXT_MEAN_LEN_THRESHOLD:
                    result["text"].append(col)
                else:
                    result["categorical"].append(col)
            else:
                result["unknown"].append(col)
        return result


# ---------------------------------------------------------------------------
# 主加载器
# ---------------------------------------------------------------------------

class DataLoader:
    """
    统一数据加载入口，屏蔽格式差异并自动完成内存优化。

    Parameters
    ----------
    compress : bool
        加载后是否立即执行 dtype 压缩，默认 True。
    parse_dates : list[str] | None
        需要解析为 datetime 的列名。
    chunk_size : int | None
        分块读取行数（仅对 CSV 有效）。None 表示一次性加载。
    verbose : bool
        是否打印加载摘要。
    low_memory_csv : bool
        CSV 加载时是否启用 low_memory 模式（减少峰值内存，
        但可能导致 dtype 推断不准，建议配合 compress=True 使用）。

    Examples
    --------
    >>> loader = DataLoader(compress=True, verbose=True)
    >>> train = loader.load("data/train.csv")
    >>> test  = loader.load("data/test.parquet")
    >>> print(loader.column_types)
    """

    def __init__(
        self,
        compress: bool = True,
        parse_dates: Optional[list[str]] = None,
        chunk_size: Optional[int] = None,
        verbose: bool = True,
        low_memory_csv: bool = True,
    ):
        self.compress = compress
        self.parse_dates = parse_dates or []
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.low_memory_csv = low_memory_csv

        self._analyzer = ColumnTypeAnalyzer()
        # 上次加载的列类型分析结果（供外部查询）
        self.column_types: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def load(
        self,
        path: PathLike,
        **kwargs,
    ) -> pd.DataFrame:
        """
        根据文件扩展名自动选择读取方式并返回压缩后的 DataFrame。

        Parameters
        ----------
        path : str | Path
            数据文件路径。
        **kwargs :
            透传给底层 pandas 读取函数的额外参数。

        Returns
        -------
        pd.DataFrame
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"数据文件不存在：{path}")

        t0 = time.perf_counter()
        suffix = path.suffix.lower()

        dispatch = {
            ".csv":     self._load_csv,
            ".tsv":     self._load_csv,
            ".parquet": self._load_parquet,
            ".feather": self._load_feather,
            ".ftr":     self._load_feather,
            ".orc":     self._load_orc,
            ".jsonl":   self._load_jsonl,
            ".json":    self._load_jsonl,
        }

        loader_fn = dispatch.get(suffix)
        if loader_fn is None:
            raise ValueError(
                f"不支持的文件格式：{suffix}。"
                f"支持：{list(dispatch.keys())}"
            )

        df = loader_fn(path, **kwargs)

        # dtype 压缩
        if self.compress:
            df = compress_dataframe(df, verbose=self.verbose)

        # 列类型分析
        self.column_types = self._analyzer.analyze(df)

        elapsed = time.perf_counter() - t0
        if self.verbose:
            self._print_summary(path, df, elapsed)

        return df

    def load_multiple(
        self,
        paths: list[PathLike],
        concat: bool = True,
        **kwargs,
    ) -> Union[pd.DataFrame, list[pd.DataFrame]]:
        """
        批量加载多个文件，可选拼接。

        Parameters
        ----------
        paths : list[PathLike]
        concat : bool
            True → pd.concat 返回单个 DataFrame；
            False → 返回 list[DataFrame]。
        """
        dfs = [self.load(p, **kwargs) for p in paths]
        if concat:
            result = pd.concat(dfs, ignore_index=True)
            if self.compress:
                result = compress_dataframe(result, verbose=self.verbose)
            return result
        return dfs

    def load_chunks(
        self,
        path: PathLike,
        process_fn=None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        分块加载 CSV 并逐块处理后拼接，适用于超大文件。

        Parameters
        ----------
        path : str | Path
        process_fn : callable | None
            对每个 chunk (pd.DataFrame) 执行的处理函数，
            例如特征提取。None 表示直接拼接原始块。
        **kwargs :
            透传给 pd.read_csv。

        Returns
        -------
        pd.DataFrame
        """
        path = Path(path)
        chunk_size = self.chunk_size or 100_000
        chunks = []

        reader = pd.read_csv(
            path,
            chunksize=chunk_size,
            low_memory=self.low_memory_csv,
            parse_dates=self.parse_dates or False,
            **kwargs,
        )
        for i, chunk in enumerate(reader):
            if self.compress:
                chunk = compress_dataframe(chunk, verbose=False)
            if process_fn is not None:
                chunk = process_fn(chunk)
            chunks.append(chunk)
            if self.verbose:
                print(f"\r[loader] 已处理 {(i + 1) * chunk_size:,} 行...", end="")

        if self.verbose:
            print()

        return pd.concat(chunks, ignore_index=True)

    # ------------------------------------------------------------------
    # 格式专属加载（私有）
    # ------------------------------------------------------------------

    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        defaults = dict(
            low_memory=self.low_memory_csv,
            parse_dates=self.parse_dates or False,
        )
        defaults.update(kwargs)
        return pd.read_csv(path, **defaults)

    def _load_parquet(self, path: Path, **kwargs) -> pd.DataFrame:
        # engine 优先 pyarrow，回退 fastparquet
        try:
            return pd.read_parquet(path, engine="pyarrow", **kwargs)
        except ImportError:
            return pd.read_parquet(path, engine="fastparquet", **kwargs)

    def _load_feather(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_feather(path, **kwargs)

    def _load_orc(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_orc(path, **kwargs)

    def _load_jsonl(self, path: Path, **kwargs) -> pd.DataFrame:
        defaults = dict(lines=True)
        defaults.update(kwargs)
        return pd.read_json(path, **defaults)

    # ------------------------------------------------------------------
    # 摘要打印
    # ------------------------------------------------------------------

    def _print_summary(self, path: Path, df: pd.DataFrame, elapsed: float):
        mem_mb = get_memory_usage_mb(df)
        n_rows, n_cols = df.shape
        missing_pct = df.isnull().mean().mean() * 100

        print(
            f"\n{'=' * 60}\n"
            f"[loader] 文件   : {path.name}\n"
            f"[loader] 形状   : {n_rows:,} 行 × {n_cols} 列\n"
            f"[loader] 内存   : {mem_mb:.2f} MB\n"
            f"[loader] 缺失率 : {missing_pct:.2f}%（全表均值）\n"
            f"[loader] 耗时   : {elapsed:.3f} s\n"
            f"[loader] 列类型 : {self._format_col_types()}\n"
            f"{'=' * 60}\n"
        )

    def _format_col_types(self) -> str:
        parts = []
        for kind, cols in self.column_types.items():
            if cols:
                parts.append(f"{kind}({len(cols)})")
        return " | ".join(parts) if parts else "—"


# ---------------------------------------------------------------------------
# 便捷函数（无需实例化 DataLoader）
# ---------------------------------------------------------------------------

def load_data(
    path: PathLike,
    compress: bool = True,
    parse_dates: Optional[list[str]] = None,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    快捷加载函数，适合脚本/Notebook 单次使用。

    Examples
    --------
    >>> train = load_data("train.csv", parse_dates=["date"])
    >>> test  = load_data("test.parquet")
    """
    loader = DataLoader(compress=compress, parse_dates=parse_dates, verbose=verbose)
    return loader.load(path, **kwargs)


def quick_eda(df: pd.DataFrame, max_cols: int = 50) -> pd.DataFrame:
    """
    生成轻量 EDA 报告：dtype、缺失率、唯一值数、均值/中位数（数值列）。

    Parameters
    ----------
    df : pd.DataFrame
    max_cols : int
        最多展示列数，防止超宽表格刷屏。

    Returns
    -------
    pd.DataFrame
        每行对应一列的统计摘要，可直接 display()。
    """
    cols = df.columns[:max_cols]
    rows = []
    for col in cols:
        s = df[col]
        row: dict = {
            "column":    col,
            "dtype":     str(s.dtype),
            "missing_%": round(s.isnull().mean() * 100, 2),
            "n_unique":  s.nunique(dropna=False),
        }
        if pd.api.types.is_numeric_dtype(s):
            row["mean"]   = round(s.mean(), 4) if not s.isnull().all() else np.nan
            row["median"] = round(s.median(), 4) if not s.isnull().all() else np.nan
            row["std"]    = round(s.std(), 4) if not s.isnull().all() else np.nan
        else:
            row["mean"]   = None
            row["median"] = None
            row["std"]    = None
        rows.append(row)

    return pd.DataFrame(rows).set_index("column")
