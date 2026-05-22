# category-lora

LoRA for **category-indexed Linear layers** — weight tensors of shape `(num_categories, in_features, out_features)` selected at runtime by per-input category id.

Standard peft can't wrap these layers because their weight is a 3D `nn.Parameter` (not `nn.Linear`/`nn.Conv*`/`nn.Embedding`), and their forward takes two args `(x, cat_ids)`. `category-lora` fills that gap.

## Where this shows up

- **NVIDIA GR00T** action heads (`CategorySpecificLinear`, `MultiEmbodimentActionEncoder`, `CategorySpecificMLP`) — large multi-embodiment VLA models where projector layers dominate the trainable budget.
- **Mixture-of-experts** with hard expert selection (each expert is a slice of a 3D weight tensor).
- Any "one weight matrix per task/category" Linear pattern.

## Install

```bash
pip install category-lora
```

(Or from source: `pip install -e .` after cloning.)

## 10-line example

```python
import torch
from category_lora import wrap_in_place, unload_adapters, CategoryLoRAConfig

# Suppose your model contains category-indexed Linear children:
# e.g. class CategorySpecificLinear(nn.Module): self.W: (C, in, out); forward(x, cat_ids)
n = wrap_in_place(
    model,
    CategoryLoRAConfig(r=16, alpha=32, dropout=0.0),
    target_class_names=["CategorySpecificLinear"],
)
print(f"wrapped {n} layers")
# train as usual: only adapter A/B are trainable; base W is frozen
# ...
unload_adapters(model)  # merges adapters into base; model now adapter-free
```

## Math

For a base layer with `W ∈ ℝ^(C × in × out)` and the standard forward `y[i] = x[i] @ W[cat_ids[i]] + b[cat_ids[i]]`, the LoRA-wrapped layer adds per-category low-rank adapters `A ∈ ℝ^(C × in × r)` and `B ∈ ℝ^(C × r × out)`:

```
y[i] = x[i] @ W[c] + b[c] + (alpha/r) · (x[i] @ A[c]) @ B[c]    where c = cat_ids[i]
```

Init: `A` kaiming-uniform, `B` zero — so at init the adapter is identity. Standard peft convention, scaled by `alpha/r`.

For `(C, in, out) = (32, 1536, 1536)` and `r=16`:
- Base: ~75M params
- LoRA adapters: ~1.6M params (~2% of base; 50× cheaper to train)

## API

```python
from category_lora import (
    CategoryLoRALinear,       # the core adapter class
    CategoryLoRAConfig,       # dataclass: r, alpha, dropout
    wrap_in_place,            # find + replace matching children with adapters
    unload_adapters,          # merge each adapter, swap parent's child back
)
```

### Standalone (single layer)

```python
adapter = CategoryLoRALinear(base_layer, r=16, alpha=32, dropout=0.0)
y = adapter(x, cat_ids)               # additive forward
adapter.merge_adapter()               # in-place merge (reversible)
adapter.unmerge_adapter()             # restore (≤1 ULP fp32 drift)
base = adapter.merge_and_unload()     # terminal merge; returns (mutated) base layer
```

### Whole-model wrap

```python
n = wrap_in_place(model, CategoryLoRAConfig(r=16, alpha=32),
                  target_class_names=["CategorySpecificLinear"])
# OR by class:
n = wrap_in_place(model, CategoryLoRAConfig(r=16, alpha=32),
                  target_classes=[CategorySpecificLinear])

unload_adapters(model)                # walks tree, swaps each adapter to its merged base
```

### State-dict

`adapter.state_dict()` contains **only** the adapter's `A` and `B` — the frozen base is excluded. Save the adapter alone; reconstruct by instantiating a `CategoryLoRALinear` over a fresh base and calling `load_state_dict(..., strict=False)`.

## DDP / distributed training

Category-indexed adapters have a quirk: per-category slices of `A` and `B` only receive gradients when their category appears in the current batch. Under PyTorch DDP's default `find_unused_parameters=False`, **absent categories cause a hang** at the allreduce barrier.

**Two correct usage patterns (one MUST be applied):**

1. **`find_unused_parameters=True`** in the DDP constructor. Small overhead; simplest fix.
2. **Balanced category sampling** so every category is hit on every rank every step.

A post-backward hook cannot fix this — DDP hangs INSIDE `loss.backward()`, before any post-hook fires.

## peft integration (v0.1: stub)

`category_lora.peft_adapter.register_with_peft()` exists but is currently a no-op stub. peft's `LoraConfig` wraps `nn.Linear`/`Conv*`/`Embedding` submodules; category-indexed layers expose a 3D `nn.Parameter` with a two-arg forward, which requires extending peft's tuner machinery (planned for v0.2). For now, **use the standalone API** (`wrap_in_place` / `unload_adapters`) — it's the primary supported path.

## Status & versioning

- **v0.1.0** — initial release. Core layer + wrappers + tests. Single-input peft integration deferred to v0.2.
- Semver. Breaking changes bump major; new features bump minor.
- CI: GitHub Actions runs `pytest` on Python 3.10+ with `torch>=2.1`.

## License

MIT. See [LICENSE](LICENSE).
