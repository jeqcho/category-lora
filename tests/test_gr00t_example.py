"""Synthetic GR00T-shaped end-to-end test (acceptance criteria 10, 11).

Builds a model with three category-linear children (mirroring the GR00T
action head: state_encoder, action_encoder, action_decoder), wraps it with
``wrap_in_place``, runs one forward + backward + optimizer step, and asserts
loss decreased and stayed finite. Runs in <5 s on CPU.
"""

from __future__ import annotations

import time

import torch
from torch import nn

from category_lora import CategoryLoRAConfig, CategoryLoRALinear, wrap_in_place

from tests.conftest import SyntheticCategoryLinear


class _TinyActionHead(nn.Module):
    """Mirrors GR00T's action head shape with three category-linear children."""

    def __init__(self, num_embodiments: int = 4, hidden: int = 16, action_dim: int = 8):
        super().__init__()
        self.state_encoder = SyntheticCategoryLinear(num_embodiments, hidden, hidden)
        self.action_encoder = SyntheticCategoryLinear(num_embodiments, action_dim, hidden)
        self.action_decoder = SyntheticCategoryLinear(num_embodiments, hidden, action_dim)

    def forward(self, state, action, cat_ids):
        s = self.state_encoder(state, cat_ids)
        a = self.action_encoder(action, cat_ids)
        h = torch.relu(s + a)
        return self.action_decoder(h, cat_ids)


def test_synthetic_gr00t_action_head_e2e():
    """End-to-end wrap + train step on a synthetic GR00T-shaped model.

    Acceptance criteria 10, 11: a runnable example wraps a multi-child model,
    runs one forward + backward + optimizer step, loss is finite, and loss
    decreased after the step. Must finish in <5 s.
    """
    t0 = time.time()
    torch.manual_seed(42)

    num_emb = 4
    hidden = 16
    action_dim = 8
    batch = 8

    model = _TinyActionHead(num_emb, hidden, action_dim)
    cfg = CategoryLoRAConfig(r=4, alpha=8, dropout=0.0)
    n = wrap_in_place(
        model,
        config=cfg,
        target_classes=[SyntheticCategoryLinear],
    )
    assert n == 3, "all three category-linear children should be wrapped"

    # All three children are adapters; bases are frozen.
    for child in (model.state_encoder, model.action_encoder, model.action_decoder):
        assert isinstance(child, CategoryLoRALinear)
        assert child.base_layer.W.requires_grad is False

    state = torch.randn(batch, hidden)
    action = torch.randn(batch, action_dim)
    cat_ids = torch.randint(0, num_emb, (batch,))
    target = torch.randn(batch, action_dim)

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0, "wrap_in_place should leave A/B trainable"
    opt = torch.optim.SGD(trainable, lr=1e-1)

    pred = model(state, action, cat_ids)
    loss_before = torch.nn.functional.mse_loss(pred, target)
    assert torch.isfinite(loss_before), "initial loss must be finite"

    opt.zero_grad()
    loss_before.backward()
    opt.step()

    pred2 = model(state, action, cat_ids)
    loss_after = torch.nn.functional.mse_loss(pred2, target)
    assert torch.isfinite(loss_after), "post-step loss must be finite"
    assert loss_after.item() < loss_before.item(), (
        f"loss should decrease after a step; before={loss_before.item()}, "
        f"after={loss_after.item()}"
    )

    elapsed = time.time() - t0
    assert elapsed < 5.0, f"test must run in <5s; took {elapsed:.2f}s"
