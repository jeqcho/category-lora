"""Stub for category_lora wrapper utilities.

Implementation in plan v3. NotImplementedError raised on use.
"""

from __future__ import annotations

from dataclasses import dataclass
from torch import nn


@dataclass
class CategoryLoRAConfig:
    """Stub — see reports/PLAN.md for the spec."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.0


def wrap_in_place(
    model: nn.Module,
    config: CategoryLoRAConfig,
    target_class_names: list[str] | None = None,
    target_classes: list[type] | None = None,
) -> int:
    """Stub. Returns number of replaced modules."""
    raise NotImplementedError


def unload_adapters(model: nn.Module) -> int:
    """Stub. Returns number of unloaded adapters."""
    raise NotImplementedError
