"""Model-level utilities: ``CategoryLoRAConfig``, ``wrap_in_place``, ``unload_adapters``."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from category_lora.layer import CategoryLoRALinear


@dataclass
class CategoryLoRAConfig:
    """Configuration for a category-LoRA wrap operation.

    Attributes:
        r: LoRA rank.
        alpha: LoRA alpha; scaling factor in forward is ``alpha / r``.
        dropout: Dropout probability applied to ``x`` on the LoRA path.
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.0


def wrap_in_place(
    model: nn.Module,
    config: CategoryLoRAConfig,
    target_class_names: list[str] | None = None,
    target_classes: list[type] | None = None,
) -> int:
    """Walk ``model`` and replace matching category-indexed Linear children with adapters.

    A child is matched if either:
    - ``type(child).__name__`` is in ``target_class_names``, or
    - ``isinstance(child, target_classes)``.

    At least one of the two must be provided. Both can be provided
    simultaneously — a child matching either is wrapped.

    Args:
        model: The model to mutate in place.
        config: ``CategoryLoRAConfig`` controlling rank, alpha, dropout.
        target_class_names: Match by class name (string). Useful when you
            can't import the original class (e.g., to avoid circular deps).
        target_classes: Match by class object. Safer when imports are
            available — robust to alias / submodule shadowing.

    Returns:
        The number of modules replaced.
    """
    if target_class_names is None and target_classes is None:
        raise ValueError(
            "Must specify at least one of target_class_names or target_classes"
        )
    names = set(target_class_names or [])
    classes = tuple(target_classes or [])

    # Snapshot (parent, child_name, child) before mutating to avoid
    # iteration-during-modification weirdness.
    pairs = []
    for parent in model.modules():
        for child_name, child in parent.named_children():
            cls_name = type(child).__name__
            if cls_name in names or (classes and isinstance(child, classes)):
                pairs.append((parent, child_name, child))

    for parent, child_name, child in pairs:
        adapter = CategoryLoRALinear(
            base_layer=child,
            r=config.r,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        setattr(parent, child_name, adapter)

    return len(pairs)


def unload_adapters(model: nn.Module) -> int:
    """Walk ``model`` and replace every ``CategoryLoRALinear`` with its merged base.

    Each adapter is merged via :meth:`CategoryLoRALinear.merge_and_unload` and
    the parent's child reference is set to the (now-mutated) base layer. After
    this call the model contains no ``CategoryLoRALinear`` instances.

    Args:
        model: The model to mutate in place.

    Returns:
        The number of adapters unloaded.
    """
    pairs = []
    for parent in model.modules():
        for child_name, child in parent.named_children():
            if isinstance(child, CategoryLoRALinear):
                pairs.append((parent, child_name, child))

    for parent, child_name, adapter in pairs:
        base = adapter.merge_and_unload()
        setattr(parent, child_name, base)

    return len(pairs)
