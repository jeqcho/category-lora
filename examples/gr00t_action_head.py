"""End-to-end example: LoRA-ify a synthetic GR00T-shaped action head.

Mirrors the structure of NVIDIA Isaac-GR00T's action head (3 category-indexed
projector layers: state_encoder, action_encoder, action_decoder). Uses
synthetic weights so the example runs in ~1s on CPU without downloading any
GR00T checkpoints.

Run:
    python examples/gr00t_action_head.py
"""

from __future__ import annotations

import torch
from torch import nn

from category_lora import (
    CategoryLoRAConfig,
    CategoryLoRALinear,
    unload_adapters,
    wrap_in_place,
)


class CategorySpecificLinear(nn.Module):
    """Mirror of NVIDIA Isaac-GR00T's `CategorySpecificLinear`.

    Real source: `gr00t/model/modules/embodiment_conditioned_mlp.py`.
    """

    def __init__(self, num_categories: int, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, in_features, out_features))
        self.b = nn.Parameter(torch.zeros(num_categories, out_features))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,bio->bo", x, self.W[cat_ids]) + self.b[cat_ids]


class SyntheticGR00TActionHead(nn.Module):
    """Tiny model mirroring GR00T's projector layout (3 category-Linears)."""

    def __init__(
        self,
        num_categories: int = 32,
        state_dim: int = 14,
        action_dim: int = 14,
        hidden_dim: int = 256,
    ):
        super().__init__()
        # state -> hidden
        self.state_encoder = CategorySpecificLinear(num_categories, state_dim, hidden_dim)
        # action -> hidden (in the real GR00T it's an MLP; we use one layer for brevity)
        self.action_encoder = CategorySpecificLinear(num_categories, action_dim, hidden_dim)
        # hidden -> action
        self.action_decoder = CategorySpecificLinear(num_categories, hidden_dim, action_dim)

    def forward(
        self,
        state: torch.Tensor,        # (B, state_dim)
        action: torch.Tensor,       # (B, action_dim)
        cat_ids: torch.Tensor,      # (B,)
    ) -> torch.Tensor:              # -> (B, action_dim)
        s = torch.relu(self.state_encoder(state, cat_ids))
        a = torch.relu(self.action_encoder(action, cat_ids))
        h = s + a
        return self.action_decoder(h, cat_ids)


def main() -> None:
    torch.manual_seed(42)
    model = SyntheticGR00TActionHead(num_categories=32, state_dim=14, action_dim=14, hidden_dim=256)

    # Param accounting BEFORE wrapping (all trainable).
    n_params_total = sum(p.numel() for p in model.parameters())
    n_params_trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Before wrap: {n_params_trainable_before:,} / {n_params_total:,} trainable "
          f"({100 * n_params_trainable_before / n_params_total:.1f}%)")

    # Wrap every CategorySpecificLinear with LoRA r=16.
    n_wrapped = wrap_in_place(
        model,
        config=CategoryLoRAConfig(r=16, alpha=32, dropout=0.0),
        target_classes=[CategorySpecificLinear],
    )
    print(f"Wrapped {n_wrapped} layers with LoRA r=16")

    # Param accounting AFTER wrapping (base frozen, only A/B trainable).
    n_params_total_after = sum(p.numel() for p in model.parameters())
    n_params_trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"After wrap : {n_params_trainable_after:,} / {n_params_total_after:,} trainable "
          f"({100 * n_params_trainable_after / n_params_total_after:.1f}%)")
    print(f"Compression: {n_params_trainable_before / n_params_trainable_after:.1f}× cheaper to train")

    # One step of "training" on a random batch.
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-3,
    )
    state = torch.randn(8, 14)
    action = torch.randn(8, 14)
    cat_ids = torch.randint(0, 32, (8,))
    target = torch.randn(8, 14)

    pred = model(state, action, cat_ids)
    loss_before = nn.functional.mse_loss(pred, target)

    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()

    with torch.no_grad():
        pred_after = model(state, action, cat_ids)
        loss_after = nn.functional.mse_loss(pred_after, target)

    print(f"Loss before step: {loss_before.item():.5f}")
    print(f"Loss after step:  {loss_after.item():.5f}")
    assert loss_after.item() < loss_before.item(), "Loss should decrease after one step"
    print("[OK] LoRA training works — loss decreased")

    # At inference time, unload: merge each adapter back into its base.
    unload_adapters(model)
    for name, child in model.named_children():
        assert not isinstance(child, CategoryLoRALinear), \
            f"{name} should no longer be a CategoryLoRALinear"
    print("[OK] unload_adapters: adapters merged into base layers")

    # The merged model produces the same output as the pre-unload model.
    pred_unloaded = model(state, action, cat_ids)
    diff = (pred_after - pred_unloaded).abs().max().item()
    print(f"Max abs diff between adapter-mode and unloaded-mode forward: {diff:.2e}")


if __name__ == "__main__":
    main()
