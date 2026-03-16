"""
core/__init__.py
对外公开 core 层的核心接口。
"""

from core.base_model import BaseModel, ModelState
from core.base_trainer import (
    CrossValidatorTrainer,
    CVResult,
    MetricConfig,
)

__all__ = [
    "BaseModel",
    "ModelState",
    "CrossValidatorTrainer",
    "CVResult",
    "MetricConfig",
]
