"""
data/__init__.py
公开 data 层的核心接口。
"""

from data.loader import DataLoader, load_data, quick_eda
from data.splitter import (
    CVConfig,
    BaseCVSplitter,
    get_cv,
    validate_no_leakage,
    fold_statistics,
    STRATEGY_REGISTRY,
)

__all__ = [
    # loader
    "DataLoader",
    "load_data",
    "quick_eda",
    # splitter
    "CVConfig",
    "BaseCVSplitter",
    "get_cv",
    "validate_no_leakage",
    "fold_statistics",
    "STRATEGY_REGISTRY",
]
