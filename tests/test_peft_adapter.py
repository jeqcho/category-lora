"""peft adapter tests (acceptance criterion 8).

Skipped entirely if ``peft`` is not installed.
"""

from __future__ import annotations

import pytest

peft = pytest.importorskip("peft")

import torch
from torch import nn

from tests.conftest import SyntheticCategoryLinear, C, IN, OUT, R


def _register():
    """Import + call ``register_with_peft`` (path lives under ``peft_adapter``)."""
    from category_lora.peft_adapter import register_with_peft

    return register_with_peft


def test_register_with_peft_idempotent():
    """Calling ``register_with_peft()`` twice must not raise.

    Acceptance criterion 8: registration is idempotent.
    """
    register_with_peft = _register()
    register_with_peft()
    register_with_peft()  # second call should be a no-op


class _SingleInputCategoryLinear(nn.Module):
    """A category-indexed Linear whose forward takes only ``x``.

    The category id is stored as a buffer/attribute on the module, set by the
    surrounding model before each forward. This shape is the only one peft can
    wrap in v0.1.
    """

    def __init__(self, num_categories: int = C, in_features: int = IN, out_features: int = OUT):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_categories, in_features, out_features))
        self.b = nn.Parameter(torch.zeros(num_categories, out_features))
        self.register_buffer("active_cat", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = int(self.active_cat.item())
        return x @ self.W[c] + self.b[c]


class _SingleInputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat = _SingleInputCategoryLinear()

    def forward(self, x):
        return self.cat(x)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "v0.1: peft path is a stub. peft's LoraConfig wraps nn.Linear/Conv/"
        "Embedding submodules, not nn.Parameter tensors like our category "
        "weight `W`. Full peft integration requires a custom tuner subclass "
        "(deferred to v0.2). The supported v0.1 path is the standalone API "
        "via wrap_in_place / unload_adapters."
    ),
)
def test_register_with_peft_single_input_layer():
    """``get_peft_model`` would produce a model with peft keys for a Linear-shaped layer.

    Acceptance criterion 8 (v0.1 limit documented): peft cannot wrap 3D
    parameter tensors natively. Marked xfail; will be promoted to a real
    pass in v0.2 if/when we implement a peft tuner subclass.
    """
    register_with_peft = _register()
    register_with_peft()

    model = _SingleInputModel()
    lora_cfg = peft.LoraConfig(r=R, lora_alpha=8, target_modules=["W"])
    peft_model = peft.get_peft_model(model, lora_cfg)

    sd_keys = list(peft_model.state_dict().keys())
    assert any("lora" in k.lower() for k in sd_keys), (
        f"expected peft lora keys, got: {sd_keys}"
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "v0.1 behavior on multi-input category layers (forward(x, cat_ids)) "
        "is implementer's choice: skip or raise. Remove this xfail once the "
        "behavior is decided and assert the chosen contract."
    ),
)
def test_register_with_peft_multi_input_layer_documented_v01_behavior():
    """v0.1: peft path must either skip multi-input layers or raise clearly.

    Acceptance criterion 8: the realistic GR00T case (forward takes
    ``cat_ids``) is not supported by the peft path in v0.1. This test
    documents that — the implementer picks ``skip`` vs ``raise`` and removes
    the ``xfail``.
    """
    register_with_peft = _register()
    register_with_peft()

    class _MultiInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cat = SyntheticCategoryLinear(C, IN, OUT)

        def forward(self, x, cat_ids):
            return self.cat(x, cat_ids)

    model = _MultiInputModel()
    lora_cfg = peft.LoraConfig(r=R, lora_alpha=8, target_modules=["W"])

    # One of these must hold; the implementer picks and removes xfail.
    # (a) RAISE: peft.get_peft_model raises a clear error
    # (b) SKIP: peft.get_peft_model succeeds but injects NO lora keys for the
    #          multi-input layer.
    raised = False
    try:
        peft_model = peft.get_peft_model(model, lora_cfg)
    except Exception:
        raised = True
        peft_model = None

    if raised:
        assert True  # behavior (a)
    else:
        sd_keys = list(peft_model.state_dict().keys())
        assert not any("lora" in k.lower() and "cat" in k for k in sd_keys), (
            f"multi-input layer should not be wrapped by peft path in v0.1; got: {sd_keys}"
        )
