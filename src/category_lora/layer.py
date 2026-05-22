"""``CategoryLoRALinear``: LoRA adapter for category-indexed Linear layers.

Wraps a module whose weight is a 3D tensor ``(C, in, out)`` and whose forward
takes ``(x, cat_ids)``. Adds per-category low-rank adapters
``A: (C, in, r)``, ``B: (C, r, out)`` and computes the additive forward
``y = base(x, cat_ids) + scale * (x @ A[cat_ids]) @ B[cat_ids]``.

See ``reports/PLAN.md`` for the full math and acceptance criteria.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class CategoryLoRALinear(nn.Module):
    """LoRA adapter for a category-indexed Linear layer.

    Args:
        base_layer: A module with ``.W`` shape ``(C, in, out)``, ``.b`` shape
            ``(C, out)``, and ``forward(x, cat_ids)`` matching
            ``y[i] = x[i] @ W[cat_ids[i]] + b[cat_ids[i]]``.
        r: LoRA rank.
        alpha: LoRA alpha. Scaling = ``alpha / r``.
        dropout: Dropout probability applied to ``x`` on the LoRA path
            (matches peft's convention).

    Attributes:
        A: ``nn.Parameter`` of shape ``(C, in, r)``. Kaiming-uniform init.
        B: ``nn.Parameter`` of shape ``(C, r, out)``. Zero init so the adapter
            starts as identity (``A @ B = 0``).
        base_layer: The wrapped base module. Its ``W`` and ``b`` get
            ``requires_grad=False`` in ``__init__``.
        scaling: Cached ``alpha / r``.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        r: int,
        alpha: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"r must be > 0, got {r}")
        if not hasattr(base_layer, "W") or base_layer.W.ndim != 3:
            raise ValueError(
                "base_layer must have a 3D weight attribute named 'W' of shape "
                "(C, in, out); got "
                f"{type(base_layer).__name__} with W shape "
                f"{getattr(base_layer.W, 'shape', None) if hasattr(base_layer, 'W') else None}"
            )

        C, in_dim, out_dim = base_layer.W.shape
        self.base_layer = base_layer
        # Freeze base params — the adapter learns the corrections.
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # Adapter parameters in fp32 by default; .to() will cast appropriately.
        dtype = base_layer.W.dtype
        device = base_layer.W.device
        self.A = nn.Parameter(torch.empty(C, in_dim, r, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.zeros(C, r, out_dim, dtype=dtype, device=device))
        # peft-compatible init: kaiming_uniform on A, zeros on B → A @ B = 0 at init.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout)

        # Internal state guards.
        self._merged = False
        self._unloaded = False

    # ----- forward -----

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """Compute ``base(x, cat_ids) + scale * (x @ A[c]) @ B[c]``.

        Args:
            x: Shape ``(B, in)``.
            cat_ids: Long tensor of shape ``(B,)``, values in ``[0, C)``.

        Returns:
            Tensor of shape ``(B, out)``.
        """
        if self._unloaded:
            raise RuntimeError(
                "This adapter has been unloaded via merge_and_unload(); call the "
                "returned base layer directly instead."
            )

        base_out = self.base_layer(x, cat_ids)
        if self._merged:
            # Base weights already contain the merged delta; do not double-count.
            return base_out

        # LoRA additive path:
        # lora_mid = einsum("bi,bir->br", x, A[cat_ids])
        # lora_out = einsum("br,bro->bo", lora_mid, B[cat_ids])
        x_dropped = self.dropout(x)
        A_sel = self.A[cat_ids]  # (B, in, r)
        B_sel = self.B[cat_ids]  # (B, r, out)
        lora_mid = torch.einsum("bi,bir->br", x_dropped, A_sel)
        lora_out = torch.einsum("br,bro->bo", lora_mid, B_sel)
        return base_out + self.scaling * lora_out

    # ----- merge / unmerge -----

    def _compute_delta(self) -> torch.Tensor:
        """Return the per-category delta ``scaling * (A @ B)`` of shape ``(C, in, out)``.

        Done in the adapter's native dtype; casting to the base dtype happens
        at the in-place add/sub site.
        """
        return self.scaling * torch.bmm(self.A, self.B)

    def merge_adapter(self) -> None:
        """Add ``scaling * A @ B`` into ``base_layer.W`` in-place. Reversible.

        After this, ``forward`` returns the base output directly (the LoRA
        path is skipped). Call :meth:`unmerge_adapter` to undo.

        Raises:
            RuntimeError: If already merged.
        """
        if self._merged:
            raise RuntimeError("Adapter is already merged. Call unmerge_adapter() first.")
        with torch.no_grad():
            delta = self._compute_delta()
            self.base_layer.W.data.add_(delta.to(self.base_layer.W.dtype))
        self._merged = True

    def unmerge_adapter(self) -> None:
        """Subtract ``scaling * A @ B`` from ``base_layer.W`` in-place.

        Round-trip ``merge → unmerge`` is bit-exact in fp32; ``atol=1e-3,
        rtol=1e-2`` in bf16 (matches peft's tolerance).

        Raises:
            RuntimeError: If not currently merged.
        """
        if not self._merged:
            raise RuntimeError("Adapter is not merged; nothing to unmerge.")
        with torch.no_grad():
            delta = self._compute_delta()
            self.base_layer.W.data.sub_(delta.to(self.base_layer.W.dtype))
        self._merged = False

    # ----- adapter-only state_dict (opt-in) -----

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only the adapter's ``A``/``B`` tensors, without the frozen base.

        Standard ``state_dict()`` (since v0.1.1) returns the full state including
        the frozen base layer — this matches HF Trainer / standard PyTorch
        save/load conventions and is what you want for checkpoint-resume.

        Call ``adapter_state_dict()`` when you only need the small (~MB) adapter
        delta — e.g. to ship a fine-tune as a tiny artifact that's applied on
        top of an existing base model checkpoint.

        Returns:
            ``{"A": tensor, "B": tensor}``. The keys are unprefixed; the caller
            can prefix them if needed for nested model loads.
        """
        return {
            "A": self.A.detach().clone(),
            "B": self.B.detach().clone(),
        }

    def load_adapter_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """Load adapter weights produced by :meth:`adapter_state_dict`.

        Args:
            sd: A dict with ``"A"`` and ``"B"`` keys.

        Raises:
            KeyError: If either ``"A"`` or ``"B"`` is missing.
            RuntimeError: If shapes mismatch.
        """
        for k in ("A", "B"):
            if k not in sd:
                raise KeyError(f"adapter state dict missing key: {k!r}")
        with torch.no_grad():
            self.A.data.copy_(sd["A"])
            self.B.data.copy_(sd["B"])

    # ----- merge_and_unload -----

    def merge_and_unload(self) -> nn.Module:
        """Merge and return the (now-mutated) base layer. Terminal operation.

        After this call, :meth:`forward` raises and the returned base layer
        should be used directly. Calling :meth:`merge_and_unload` a second
        time raises ``RuntimeError``.

        Returns:
            ``self.base_layer`` with its ``W`` mutated in-place.
        """
        if self._unloaded:
            raise RuntimeError("Adapter has already been unloaded; cannot unload again.")
        if not self._merged:
            self.merge_adapter()
        # Re-enable grad on the base now that the caller owns it (LoRA is gone).
        for p in self.base_layer.parameters():
            p.requires_grad = True
        self._unloaded = True
        return self.base_layer
