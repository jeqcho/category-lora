"""Forward-pass tests for ``CategoryLoRALinear`` (acceptance criteria 1, 2).

Also covers the DDP grad-mask scenario documented in the v3 plan: with
unbalanced category sampling, present-category slices receive non-zero
gradients while absent-category slices receive zero/``None`` gradients.
"""

from __future__ import annotations

import torch
from torch import nn

from category_lora import CategoryLoRALinear

from tests.conftest import SyntheticCategoryLinear, C, IN, OUT, R


def test_forward_shape(synthetic_cat_linear):
    """Forward returns the expected ``(B, out)`` shape.

    Acceptance criterion 1: forward must match the math, starting with shape.
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    x = torch.randn(2, IN)
    cat_ids = torch.tensor([0, 1], dtype=torch.long)
    y = adapter(x, cat_ids)
    assert y.shape == (2, OUT)


def test_forward_init_identity(synthetic_cat_linear):
    """At init (``B = 0``), adapter forward is bit-exact equal to base forward.

    Acceptance criterion 2: ``B`` is zero-initialized so ``A @ B = 0`` and the
    adapter starts as identity over the base layer. fp32 only; uses
    ``torch.equal`` (not ``allclose``).
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    x = torch.randn(5, IN)
    cat_ids = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
    y_adapter = adapter(x, cat_ids)
    y_base = synthetic_cat_linear(x, cat_ids)
    assert torch.equal(y_adapter, y_base), "B=0 init must yield bit-exact base output"


def test_forward_matches_manual_reference(synthetic_cat_linear):
    """Adapter forward equals a manual einsum reference impl.

    Acceptance criterion 1: matches the math in PLAN.md Math section. We
    randomize ``B`` so the LoRA branch is non-trivial, then compare.
    Tolerances follow the plan's precision contract.
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    # Randomize B so the LoRA path contributes (B=0 at init).
    with torch.no_grad():
        adapter.B.data = torch.randn_like(adapter.B.data)

    B = 6
    x = torch.randn(B, IN)
    cat_ids = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.long)

    y = adapter(x, cat_ids)

    # Manual reference per PLAN.md Math (Forward additive form).
    W = synthetic_cat_linear.W  # (C, in, out)
    b = synthetic_cat_linear.b  # (C, out)
    A = adapter.A                 # (C, in, r)
    Bp = adapter.B                # (C, r, out)
    scale = 8 / R

    base_out = torch.einsum("bi,bio->bo", x, W[cat_ids])
    lora_mid = torch.einsum("bi,bir->br", x, A[cat_ids])
    lora_out = torch.einsum("br,bro->bo", lora_mid, Bp[cat_ids])
    y_ref = base_out + b[cat_ids] + scale * lora_out

    torch.testing.assert_close(y, y_ref, atol=1e-6, rtol=1e-5)

    # bf16 tolerance check
    base_bf16 = SyntheticCategoryLinear(C, IN, OUT).to(torch.bfloat16)
    with torch.no_grad():
        base_bf16.W.copy_(W.to(torch.bfloat16))
        base_bf16.b.copy_(b.to(torch.bfloat16))
    adapter_bf16 = CategoryLoRALinear(base_bf16, r=R, alpha=8).to(torch.bfloat16)
    with torch.no_grad():
        adapter_bf16.A.data.copy_(A.to(torch.bfloat16))
        adapter_bf16.B.data.copy_(Bp.to(torch.bfloat16))
    x_bf16 = x.to(torch.bfloat16)
    y_bf16 = adapter_bf16(x_bf16, cat_ids)
    torch.testing.assert_close(y_bf16.float(), y_ref, atol=1e-3, rtol=1e-2)


def test_forward_multi_category_batch(synthetic_cat_linear):
    """Mixed-category batch yields per-element correct outputs.

    Acceptance criterion 1: indexing by ``cat_ids`` must work elementwise,
    not blend categories.
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    with torch.no_grad():
        adapter.B.data = torch.randn_like(adapter.B.data)

    cat_ids = torch.tensor([0, 1, 0, 2, 3], dtype=torch.long)
    x = torch.randn(5, IN)

    y = adapter(x, cat_ids)

    # Per-element check: each row computed via single-category forward.
    for i, c in enumerate(cat_ids.tolist()):
        xi = x[i : i + 1]
        ci = torch.tensor([c], dtype=torch.long)
        yi = adapter(xi, ci)
        torch.testing.assert_close(y[i : i + 1], yi, atol=1e-6, rtol=1e-5)


def test_ddp_grad_mask(synthetic_cat_linear):
    """Single-rank DDP simulation: absent categories produce zero/None grads.

    Documents the DDP ``find_unused_parameters`` failure mode from PLAN.md.
    When category ``c`` is absent from a batch, ``A[c]`` and ``B[c]`` slices
    receive no gradient signal; under DDP's default this hangs at the
    allreduce barrier. Asserts grads are zero for absent categories and
    non-zero for present ones, which is exactly the condition that triggers
    the DDP hang.
    """
    adapter = CategoryLoRALinear(synthetic_cat_linear, r=R, alpha=8)
    # Randomize B so the LoRA gradients are non-trivial.
    with torch.no_grad():
        adapter.B.data = torch.randn_like(adapter.B.data)

    # Batch only uses categories {0, 2}; categories {1, 3} are absent.
    present = {0, 2}
    absent = {1, 3}
    cat_ids = torch.tensor([0, 2, 0, 2], dtype=torch.long)
    x = torch.randn(4, IN)

    y = adapter(x, cat_ids)
    loss = y.sum()
    loss.backward()

    for c in present:
        # A and B must have non-trivial gradients for present categories.
        A_grad_c = adapter.A.grad[c]
        B_grad_c = adapter.B.grad[c]
        assert A_grad_c is not None
        assert B_grad_c is not None
        assert (A_grad_c.abs().sum() > 0).item(), f"A[{c}] grad should be non-zero"
        assert (B_grad_c.abs().sum() > 0).item(), f"B[{c}] grad should be non-zero"

    for c in absent:
        # Absent categories: grads must be zero (and may be None per param).
        A_grad_c = adapter.A.grad[c]
        B_grad_c = adapter.B.grad[c]
        assert (A_grad_c.abs().sum() == 0).item(), f"A[{c}] grad should be zero"
        assert (B_grad_c.abs().sum() == 0).item(), f"B[{c}] grad should be zero"
