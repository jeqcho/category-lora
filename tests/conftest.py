"""Shared pytest fixtures for the category-lora test suite.

The synthetic ``CategorySpecificLinear`` defined here mirrors the GR00T module
that ``CategoryLoRALinear`` wraps. It is defined locally (not imported from
GR00T) so the tests run with only ``torch`` and ``pytest`` installed.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from category_lora import CategoryLoRAConfig


C = 4
IN = 8
OUT = 16
R = 4


class SyntheticCategoryLinear(nn.Module):
    """Tiny ``CategorySpecificLinear``-shaped module for tests.

    The weight is a 3D tensor of shape ``(C, in, out)`` and the bias is
    ``(C, out)``. ``forward`` takes ``(x, cat_ids)`` and computes
    ``y[i] = x[i] @ W[cat_ids[i]] + b[cat_ids[i]]``.

    Matches the API of NVIDIA GR00T's ``CategorySpecificLinear`` so the same
    adapter that wraps GR00T can wrap this in tests.
    """

    def __init__(self, num_categories: int = C, in_features: int = IN, out_features: int = OUT):
        super().__init__()
        self.num_categories = num_categories
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.randn(num_categories, in_features, out_features))
        self.b = nn.Parameter(torch.zeros(num_categories, out_features))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, in), cat_ids: (B,) -> y: (B, out)
        W_sel = self.W[cat_ids]  # (B, in, out)
        b_sel = self.b[cat_ids]  # (B, out)
        y = torch.einsum("bi,bio->bo", x, W_sel) + b_sel
        return y


@pytest.fixture(autouse=True)
def manual_seed_fixture():
    """Seed torch to 42 at the start of every test for determinism."""
    torch.manual_seed(42)
    yield


@pytest.fixture
def synthetic_cat_linear() -> SyntheticCategoryLinear:
    """A fresh ``SyntheticCategoryLinear`` with ``C=4, in=8, out=16``."""
    return SyntheticCategoryLinear(num_categories=C, in_features=IN, out_features=OUT)


@pytest.fixture
def category_lora_config_default() -> CategoryLoRAConfig:
    """Default ``CategoryLoRAConfig(r=4, alpha=8, dropout=0.0)`` for tests."""
    return CategoryLoRAConfig(r=R, alpha=8, dropout=0.0)
