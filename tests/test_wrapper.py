"""``wrap_in_place`` / ``unload_adapters`` tests (acceptance criteria 6, 7)."""

from __future__ import annotations

import torch
from torch import nn

from category_lora import (
    CategoryLoRAConfig,
    CategoryLoRALinear,
    unload_adapters,
    wrap_in_place,
)

from tests.conftest import SyntheticCategoryLinear, C, IN, OUT, R


class _ToyModel(nn.Module):
    """Toy model with two category-linear children + one ``nn.Linear``."""

    def __init__(self):
        super().__init__()
        self.cat_linear_1 = SyntheticCategoryLinear(C, IN, OUT)
        self.cat_linear_2 = SyntheticCategoryLinear(C, IN, OUT)
        self.plain_linear = nn.Linear(OUT, OUT)


class _NestedModel(nn.Module):
    """Toy model with category-linears inside ``nn.Sequential`` and ``ModuleList``."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(SyntheticCategoryLinear(C, IN, OUT))
        self.list = nn.ModuleList([SyntheticCategoryLinear(C, IN, OUT)])
        self.top = SyntheticCategoryLinear(C, IN, OUT)


def test_wrap_in_place_by_class_name(category_lora_config_default):
    """``wrap_in_place(..., target_class_names=[...])`` replaces matching children.

    Acceptance criterion 6: matches by class name, returns the count, and
    leaves non-matching modules untouched.
    """
    model = _ToyModel()
    n = wrap_in_place(
        model,
        config=category_lora_config_default,
        target_class_names=["SyntheticCategoryLinear"],
    )
    assert n == 2
    assert isinstance(model.cat_linear_1, CategoryLoRALinear)
    assert isinstance(model.cat_linear_2, CategoryLoRALinear)
    assert isinstance(model.plain_linear, nn.Linear)
    assert not isinstance(model.plain_linear, CategoryLoRALinear)


def test_wrap_in_place_by_class_object(category_lora_config_default):
    """``wrap_in_place(..., target_classes=[...])`` replaces matching children.

    Acceptance criterion 6: matches by class object (the non-fragile variant).
    """
    model = _ToyModel()
    n = wrap_in_place(
        model,
        config=category_lora_config_default,
        target_classes=[SyntheticCategoryLinear],
    )
    assert n == 2
    assert isinstance(model.cat_linear_1, CategoryLoRALinear)
    assert isinstance(model.cat_linear_2, CategoryLoRALinear)
    assert isinstance(model.plain_linear, nn.Linear)


def test_wrap_in_place_freezes_base(category_lora_config_default):
    """Base params get ``requires_grad=False``; adapter ``A``/``B`` stay trainable.

    Acceptance criterion 6: freezing contract.
    """
    model = _ToyModel()
    wrap_in_place(
        model,
        config=category_lora_config_default,
        target_classes=[SyntheticCategoryLinear],
    )
    wrapped = model.cat_linear_1
    assert isinstance(wrapped, CategoryLoRALinear)
    assert wrapped.base_layer.W.requires_grad is False
    assert wrapped.base_layer.b.requires_grad is False
    assert wrapped.A.requires_grad is True
    assert wrapped.B.requires_grad is True


def test_wrap_in_place_nested(category_lora_config_default):
    """Nested category-linears (Sequential / ModuleList) are also replaced.

    Acceptance criterion 6: the walker must recurse, not just look at direct
    children of the root model.
    """
    model = _NestedModel()
    n = wrap_in_place(
        model,
        config=category_lora_config_default,
        target_classes=[SyntheticCategoryLinear],
    )
    assert n == 3
    assert isinstance(model.seq[0], CategoryLoRALinear)
    assert isinstance(model.list[0], CategoryLoRALinear)
    assert isinstance(model.top, CategoryLoRALinear)


def test_unload_adapters_swaps_children(category_lora_config_default):
    """``unload_adapters(model)`` swaps each adapter back to its merged base.

    Acceptance criterion 7: after unload, the children of the model are the
    original base-class instances (with weights merged), not adapters.
    """
    model = _ToyModel()
    wrap_in_place(
        model,
        config=category_lora_config_default,
        target_classes=[SyntheticCategoryLinear],
    )
    # Randomize each adapter's B so the merge produces a non-trivial delta.
    with torch.no_grad():
        model.cat_linear_1.B.data = torch.randn_like(model.cat_linear_1.B.data)
        model.cat_linear_2.B.data = torch.randn_like(model.cat_linear_2.B.data)

    # Sanity check: snapshot the unmerged forward we should reproduce post-unload.
    x = torch.randn(3, IN)
    cat_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    y_before = model.cat_linear_1(x, cat_ids).detach().clone()

    unload_adapters(model)

    assert not isinstance(model.cat_linear_1, CategoryLoRALinear)
    assert not isinstance(model.cat_linear_2, CategoryLoRALinear)
    assert isinstance(model.cat_linear_1, SyntheticCategoryLinear)
    assert isinstance(model.cat_linear_2, SyntheticCategoryLinear)

    y_after = model.cat_linear_1(x, cat_ids)
    torch.testing.assert_close(y_after, y_before, atol=1e-6, rtol=1e-5)
