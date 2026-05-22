"""State-dict save/load tests (acceptance criterion 5)."""

from __future__ import annotations

import io

import torch

from category_lora import CategoryLoRALinear

from tests.conftest import SyntheticCategoryLinear, C, IN, OUT, R


def test_state_dict_roundtrip_fp32(synthetic_cat_linear):
    """Save then load reproduces forward outputs bit-exactly in fp32.

    Acceptance criterion 5: ``torch.save(adapter.state_dict()) → fresh
    adapter → load_state_dict`` forward equals the original forward exactly.
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    with torch.no_grad():
        adapter.B.data = torch.randn_like(adapter.B.data)

    x = torch.randn(4, IN)
    cat_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    y_orig = adapter(x, cat_ids).detach().clone()

    buf = io.BytesIO()
    torch.save(adapter.state_dict(), buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=True)

    # Fresh base must hold the SAME weights as the original (the adapter's
    # state_dict does not store the frozen base).
    fresh_base = SyntheticCategoryLinear(C, IN, OUT)
    with torch.no_grad():
        fresh_base.W.copy_(synthetic_cat_linear.W)
        fresh_base.b.copy_(synthetic_cat_linear.b)

    fresh = CategoryLoRALinear(fresh_base, r=R, alpha=8)
    missing, unexpected = fresh.load_state_dict(loaded, strict=False)
    # We allow missing-base-keys but unexpected must be empty.
    assert unexpected == [], f"unexpected keys: {unexpected}"

    y_fresh = fresh(x, cat_ids)
    assert torch.equal(y_fresh, y_orig), "fp32 state-dict round-trip must be bit-exact"


def test_state_dict_contains_expected_keys(synthetic_cat_linear):
    """State-dict has the adapter ``A``/``B`` and excludes the frozen base ``W``.

    Acceptance criterion 5: adapter state must be the deltas only — the frozen
    base layer's weight should not appear in the adapter checkpoint.
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    sd = adapter.state_dict()
    keys = set(sd.keys())

    # Must contain A and B in some namespacing.
    assert any(k.endswith("A") for k in keys), f"missing A; got {keys}"
    assert any(k.endswith("B") for k in keys), f"missing B; got {keys}"

    # Must NOT contain the base layer's W (or b) — base is frozen and excluded.
    for k in keys:
        assert "base_layer.W" not in k, f"base W must not be in adapter state_dict; got {keys}"
        assert "base_layer.b" not in k, f"base b must not be in adapter state_dict; got {keys}"
