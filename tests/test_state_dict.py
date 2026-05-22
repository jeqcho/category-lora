"""State-dict save/load tests (acceptance criterion 5).

v0.1.1: ``state_dict()`` now returns the FULL state (base + adapter) for
HF-Trainer / standard PyTorch save-load compatibility.
``adapter_state_dict()`` is the opt-in API for adapter-only saves.
"""

from __future__ import annotations

import io

import torch

from category_lora import CategoryLoRALinear

from tests.conftest import SyntheticCategoryLinear, C, IN, OUT, R


def test_state_dict_roundtrip_fp32(synthetic_cat_linear):
    """Full state_dict round-trip is bit-exact in fp32 (v0.1.1).

    Acceptance criterion 5: ``torch.save(adapter.state_dict()) → fresh
    adapter (with fresh base, since state_dict now includes base) →
    load_state_dict → forward equals original forward exactly.
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

    # Fresh adapter wraps a *fresh* base (zero-init); state_dict now includes
    # the base layer, so load_state_dict will populate fresh.base_layer too.
    fresh_base = SyntheticCategoryLinear(C, IN, OUT)
    with torch.no_grad():
        fresh_base.W.zero_()
        fresh_base.b.zero_()

    fresh = CategoryLoRALinear(fresh_base, r=R, alpha=8)
    incompat = fresh.load_state_dict(loaded, strict=True)
    assert incompat.missing_keys == [], f"missing: {incompat.missing_keys}"
    assert incompat.unexpected_keys == [], f"unexpected: {incompat.unexpected_keys}"

    y_fresh = fresh(x, cat_ids)
    assert torch.equal(y_fresh, y_orig), "fp32 state-dict round-trip must be bit-exact"


def test_state_dict_includes_base_for_hf_trainer_compat(synthetic_cat_linear):
    """v0.1.1: default ``state_dict()`` includes the frozen base layer.

    This is the HF-Trainer-compatible behavior. Use ``adapter_state_dict()``
    (separate method) for the v0.1.0-style adapter-only save.
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    sd = adapter.state_dict()
    keys = set(sd.keys())

    # Adapter A/B present.
    assert any(k.endswith("A") for k in keys), f"missing A; got {keys}"
    assert any(k.endswith("B") for k in keys), f"missing B; got {keys}"

    # Base W and b PRESENT (full state for HF Trainer compat).
    assert any("base_layer.W" in k for k in keys), f"base W expected; got {keys}"
    assert any("base_layer.b" in k for k in keys), f"base b expected; got {keys}"


def test_adapter_state_dict_excludes_base(synthetic_cat_linear):
    """``adapter_state_dict()`` returns ONLY ``A`` and ``B``.

    For users who want to ship a small adapter-only fine-tune. Round-trippable
    via ``load_adapter_state_dict``.
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    with torch.no_grad():
        adapter.B.data = torch.randn_like(adapter.B.data)

    asd = adapter.adapter_state_dict()
    assert set(asd.keys()) == {"A", "B"}, f"expected {{A,B}}; got {set(asd.keys())}"

    x = torch.randn(3, IN)
    cat_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    y_orig = adapter(x, cat_ids).detach().clone()

    # Fresh adapter wrapping the SAME base layer (in real life the caller
    # reconstructs the base before applying the adapter delta).
    fresh = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    fresh.load_adapter_state_dict(asd)

    y_fresh = fresh(x, cat_ids)
    assert torch.equal(y_fresh, y_orig), "adapter_state_dict round-trip must be bit-exact"
