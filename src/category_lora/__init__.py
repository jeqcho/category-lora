"""category-lora — LoRA for category-indexed Linear layers.

See https://github.com/jeqcho/category-lora and reports/PLAN.md for details.
"""

__version__ = "0.1.2"

from category_lora.layer import CategoryLoRALinear
from category_lora.wrapper import CategoryLoRAConfig, wrap_in_place, unload_adapters

__all__ = [
    "CategoryLoRALinear",
    "CategoryLoRAConfig",
    "wrap_in_place",
    "unload_adapters",
    "__version__",
]
