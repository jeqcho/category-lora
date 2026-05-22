"""Optional peft integration for ``category-lora``.

``register_with_peft()`` is idempotent and ensures peft is importable.

**v0.1 limitation:** the realistic GR00T case (``forward(x, cat_ids)``) is NOT
supported via the peft path. peft's ``LoraConfig`` targets ``nn.Linear``/
``nn.Conv*``/``nn.Embedding`` submodules and assumes single-input forward.
Category-indexed layers have a 3D ``nn.Parameter`` (not a wrapped Linear) and
a two-arg forward, neither of which peft natively understands. Extending
peft to handle this requires subclassing ``peft.tuners.tuners_utils`` —
outside v0.1 scope.

**The supported path** for category-indexed layers is the standalone API:

.. code-block:: python

    from category_lora import wrap_in_place, CategoryLoRAConfig
    wrap_in_place(model, CategoryLoRAConfig(r=16, alpha=32),
                  target_classes=[CategorySpecificLinear])
    # train as normal
    from category_lora import unload_adapters
    unload_adapters(model)

This module exists primarily so users who already drive everything through
peft can call ``register_with_peft()`` for forward-compat with future
versions of category-lora that may add deeper peft integration.
"""

from __future__ import annotations

_REGISTERED = False


def register_with_peft() -> None:
    """Idempotent. v0.1: ensures peft is importable; no peft mutation yet.

    Raises:
        ImportError: If peft is not installed.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    import peft  # noqa: F401  — fail fast if peft is not installed

    # v0.1: no peft mutation. Future versions may inject a custom tuner here.
    _REGISTERED = True
