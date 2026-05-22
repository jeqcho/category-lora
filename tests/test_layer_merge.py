"""Merge / unmerge / merge_and_unload tests (acceptance criteria 3, 4)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from category_lora import CategoryLoRALinear

from tests.conftest import SyntheticCategoryLinear, C, IN, OUT, R


def _randomize_B(adapter: CategoryLoRALinear) -> None:
    with torch.no_grad():
        adapter.B.data = torch.randn_like(adapter.B.data)


def test_merge_adapter_then_forward_matches_unmerged(synthetic_cat_linear):
    """``merge_adapter()`` then forward equals the unmerged forward.

    Acceptance criterion 3: the merged path is mathematically equivalent to
    the additive path within precision tolerance. fp32 atol=1e-6, bf16
    atol=1e-3.
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    _randomize_B(adapter)
    x = torch.randn(6, IN)
    cat_ids = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)

    y_unmerged = adapter(x, cat_ids).detach().clone()
    adapter.merge_adapter()
    y_merged = adapter(x, cat_ids)

    torch.testing.assert_close(y_merged, y_unmerged, atol=1e-6, rtol=1e-5)

    # bf16 path
    base_bf16 = SyntheticCategoryLinear(C, IN, OUT).to(torch.bfloat16)
    with torch.no_grad():
        base_bf16.W.copy_(synthetic_cat_linear.W.to(torch.bfloat16))
        base_bf16.b.copy_(synthetic_cat_linear.b.to(torch.bfloat16))
    adapter_bf16 = CategoryLoRALinear(base_bf16, r=R, alpha=8).to(torch.bfloat16)
    with torch.no_grad():
        adapter_bf16.A.data.copy_(adapter.A.to(torch.bfloat16))
        adapter_bf16.B.data.copy_(adapter.B.to(torch.bfloat16))
    x_bf16 = x.to(torch.bfloat16)
    y_unmerged_bf16 = adapter_bf16(x_bf16, cat_ids).detach().clone()
    adapter_bf16.merge_adapter()
    y_merged_bf16 = adapter_bf16(x_bf16, cat_ids)
    torch.testing.assert_close(y_merged_bf16, y_unmerged_bf16, atol=1e-3, rtol=1e-2)


def test_merge_unmerge_roundtrip_fp32(synthetic_cat_linear):
    """``merge → unmerge`` is bit-exact in fp32.

    Acceptance criterion 4: round-trip restores the base weight exactly when
    the math is done in fp32.
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    _randomize_B(adapter)
    W_orig = synthetic_cat_linear.W.detach().clone()

    adapter.merge_adapter()
    adapter.unmerge_adapter()

    assert torch.equal(synthetic_cat_linear.W, W_orig), "fp32 round-trip must be bit-exact"


def test_merge_unmerge_roundtrip_bf16():
    """``merge → unmerge`` in bf16: within ``atol=1e-3, rtol=1e-2`` (peft tol)."""
    base = SyntheticCategoryLinear(C, IN, OUT).to(torch.bfloat16)
    adapter = CategoryLoRALinear(base, r=R, alpha=8).to(torch.bfloat16)
    with torch.no_grad():
        adapter.B.data = torch.randn_like(adapter.B.data)
    W_orig = base.W.detach().clone()

    adapter.merge_adapter()
    adapter.unmerge_adapter()

    torch.testing.assert_close(base.W, W_orig, atol=1e-3, rtol=1e-2)


def test_merge_and_unload_returns_base(synthetic_cat_linear):
    """``merge_and_unload`` returns the base layer (not the adapter) and is one-shot.

    Acceptance criterion 4: terminal operation; second call raises.
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    _randomize_B(adapter)
    W_orig = synthetic_cat_linear.W.detach().clone()

    returned = adapter.merge_and_unload()

    # Must be the original base class, NOT a CategoryLoRALinear.
    assert isinstance(returned, nn.Module)
    assert not isinstance(returned, CategoryLoRALinear)
    assert isinstance(returned, SyntheticCategoryLinear)

    # The returned module's W has the merged values (differs from the original).
    assert not torch.equal(returned.W, W_orig), "W should be mutated by merge"

    # Calling a second time must raise.
    with pytest.raises(RuntimeError):
        adapter.merge_and_unload()


def test_merge_and_unload_forward_matches_unmerged_adapter(synthetic_cat_linear):
    """After ``merge_and_unload``, ``base.forward`` matches the unmerged adapter forward.

    Acceptance criterion 3 (variant): exercises the post-unload base layer.
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    _randomize_B(adapter)
    x = torch.randn(5, IN)
    cat_ids = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)

    y_unmerged = adapter(x, cat_ids).detach().clone()
    base = adapter.merge_and_unload()
    y_base = base(x, cat_ids)

    torch.testing.assert_close(y_base, y_unmerged, atol=1e-6, rtol=1e-5)
