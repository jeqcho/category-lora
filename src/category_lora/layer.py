"""Stub for category_lora.CategoryLoRALinear.

Implementation in plan v3. This stub exists so test imports succeed; the
test-writer subagent runs against this, expecting tests to FAIL initially.
"""

from __future__ import annotations

import torch
from torch import nn


class CategoryLoRALinear(nn.Module):
    """Stub — see reports/PLAN.md for the spec. NotImplementedError raised on use."""

    def __init__(self, base_layer: nn.Module, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError("CategoryLoRALinear not yet implemented; see reports/PLAN.md")

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def merge_adapter(self) -> None:
        raise NotImplementedError

    def unmerge_adapter(self) -> None:
        raise NotImplementedError

    def merge_and_unload(self) -> nn.Module:
        raise NotImplementedError
