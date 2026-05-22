# category-lora — Implementation Plan

**Status:** v1 (pre-subagent review)
**Date:** 2026-05-21

## What this package is

A small PyTorch library that adds LoRA-style low-rank adapters to **category-indexed Linear layers** — layers whose weight is a 3D tensor `(num_categories, in_features, out_features)`, where at runtime a per-input category id selects one slice of the weight to compute `y = x @ W[cat] + b[cat]`.

These layers appear in **multi-task / multi-embodiment VLA models** (NVIDIA's GR00T `CategorySpecificLinear`, `MultiEmbodimentActionEncoder`, `CategorySpecificMLP`) and in **mixture-of-experts** settings with hard expert selection.

Standard peft LoRA doesn't wrap these layers because:
- They're not `nn.Linear` / `nn.Conv*` / `nn.Embedding` instances (peft's default target classes).
- Their weights are 3D, so a vanilla rank-`r` LoRA `W += B@A` with 2D `A`/`B` doesn't preserve the per-category structure.

This package provides:
1. A `CategoryLoRALinear` adapter module that wraps an existing 3D-weight Linear and adds **per-category** low-rank updates `W_c += B_c @ A_c` while keeping the base 3D weight frozen.
2. A drop-in `wrap_in_place(model, ...)` utility that finds matching modules in a `nn.Module` tree and replaces them with adapter-wrapped versions.
3. An optional peft adapter (`peft_adapter.py`) so users can register `CategorySpecificLinear`-like classes with peft's `get_peft_model` machinery and use the standard peft `LoraConfig` API.
4. Merge / unmerge primitives (analogous to peft's `merge_and_unload` and `merge_adapter`/`unmerge_adapter`) for inference and post-training checkpoint export.
5. A working example wrapping GR00T's `CategorySpecificLinear`.

## Why this exists

GR00T's "projector" submodule (`state_encoder`, `action_encoder`, `action_decoder` inside the action head) uses `CategorySpecificLinear` extensively. At ~200-400M params, it dominates the trainable budget of LoRA recipes that keep the projector full-FT. Freezing the projector hurts task adaptation (val MSE 3-4× worse in our exp 8 H-recipe). LoRA-ing the projector is the missing recipe.

Beyond GR00T, this primitive is identical to what mixture-of-experts adapter frameworks like [FLoRA (Wen+ 2023)](https://arxiv.org/pdf/2312.05677) and S-LoRA's MoE-mode use internally. Shipping it as a standalone, well-tested package lets others reuse it without copy-pasting code from a research repo.

## API surface (target)

### Standalone path

```python
from category_lora import CategoryLoRALinear

# Wrap an existing category-indexed module
adapter = CategoryLoRALinear(
    base_layer=original_layer,   # has .W shape (C, in, out) and .b shape (C, out)
    r=16,
    alpha=32,
    dropout=0.0,
)

# Forward signature unchanged — adapter delegates to base + adds LoRA delta
y = adapter(x, cat_ids)

# Merge into base for inference (irreversible)
merged_layer = adapter.merge_and_unload()

# Or merge reversibly + unmerge
adapter.merge_adapter()
y = adapter(x, cat_ids)   # base now contains merged weights
adapter.unmerge_adapter() # restores
```

### Drop-in path

```python
from category_lora import wrap_in_place, LoRAConfig

cfg = LoRAConfig(r=16, alpha=32, dropout=0.0)
n_wrapped = wrap_in_place(
    model,
    target_class_names=["CategorySpecificLinear", "MultiEmbodimentActionEncoder"],
    config=cfg,
)
# Now model contains adapter modules in place; base params are frozen,
# adapter params are trainable. Standard PyTorch training loop works.
```

### peft integration path (optional)

```python
import peft
from category_lora.peft_adapter import register_with_peft

register_with_peft()   # idempotent; teaches peft about CategoryLoRA

# Now standard peft API works:
peft_model = peft.get_peft_model(
    model,
    peft.LoraConfig(r=16, lora_alpha=32, target_modules=["W"]),
)
```

## Math

Given a base `CategoryLoRALinear`-style module with weight `W ∈ ℝ^(C × in × out)` and bias `b ∈ ℝ^(C × out)`, and an input batch `(x ∈ ℝ^(B × in), cat_ids ∈ ℤ^B)`, the base forward computes:

```
y[i] = x[i] @ W[cat_ids[i]] + b[cat_ids[i]]
```

The LoRA update introduces per-category low-rank matrices `A ∈ ℝ^(C × r × in)` and `B ∈ ℝ^(C × out × r)`, plus a scalar scale `s = alpha / r`. The adapted forward is:

```
y[i] = x[i] @ W[cat_ids[i]] + b[cat_ids[i]] + s · x[i] @ A[cat_ids[i]]ᵀ @ B[cat_ids[i]]ᵀ
```

(Standard LoRA init: `A` ~ N(0, σ²) small, `B` = 0 → adapter starts at identity transform.)

**Param counts** for `(C, in, out) = (32, 1536, 1536)`:
- Base: 32 × 1536 × 1536 ≈ **75M params**
- LoRA adapter at r=16: 32 × (16·1536 + 16·1536) ≈ **1.6M params** (~2% of base)
- 50× reduction on the category-Linear's trainable cost.

**Merge math** (in-place, reversible):
```
W'[c] := W[c] + s · A[c]ᵀ @ B[c]ᵀ   (shape: (in, out) per category c)
```

**Unmerge:** subtract the same delta. Two notes on precision:
- Both operations are computed in `float32` then cast back to `W`'s dtype; this matches peft's convention and minimizes bf16 round-trip drift.
- Round-trip merge → unmerge is bit-exact in fp32 and `O(ulp)` drift in bf16/fp16; we mark `unmerge_adapter()` as approximately reversible (same caveat as peft for non-DoRA LoRA).

## File layout

```
category-lora/
├── README.md                       # short pitch + 30-second usage
├── LICENSE (MIT)
├── pyproject.toml                  # minimal, torch + peft (optional) as deps
├── src/category_lora/
│   ├── __init__.py                 # public API: CategoryLoRALinear, wrap_in_place, LoRAConfig
│   ├── layer.py                    # CategoryLoRALinear (the core class)
│   ├── wrapper.py                  # wrap_in_place, helper utilities
│   └── peft_adapter.py             # register_with_peft (optional integration)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # shared fixtures (synthetic CategorySpecificLinear)
│   ├── test_layer_forward.py
│   ├── test_layer_merge.py
│   ├── test_wrapper.py
│   └── test_peft_adapter.py        # marked optional; skipped if peft not installed
├── examples/
│   └── gr00t_action_head.py        # walk-through: load GR00T action head, LoRA-ify
└── reports/
    ├── PLAN.md                     # this file
    └── RUN_LOG.md                  # appended as we go
```

## Acceptance criteria

1. `CategoryLoRALinear` forward matches the math above (verified via tests against a hand-rolled reference impl).
2. `merge_and_unload()` + a re-eval gives the same output as the wrapped forward, within bf16 ULP tolerance.
3. `merge_adapter()` → forward → `unmerge_adapter()` round-trip is bit-exact in fp32, ≤1 ULP drift in bf16.
4. `wrap_in_place(model, target_class_names=...)` replaces matching modules without touching others; base params end with `requires_grad=False`, adapter params with `requires_grad=True`.
5. Optional peft adapter registers our class and a `get_peft_model(model, LoraConfig(target_modules=[...]))` call wraps GR00T's `CategorySpecificLinear` correctly.
6. ≥90% test coverage on `layer.py` and `wrapper.py`.
7. README has a runnable 10-line usage snippet.
8. CI: `pytest -v` passes on Python 3.10 with `torch==2.5` and `peft==0.x`.

## Out of scope (v1)

- DoRA / weight-decomposed LoRA variants.
- AdaLoRA-style rank scheduling.
- Distributed FSDP-specific shard handling (should "just work" but not explicitly tested).
- Triton kernels for batched per-cat matmul (defer; default impl is a fancy-indexed bmm + standard torch ops).
- Saving/loading peft-style adapter-only checkpoints (defer; v1 ships merge + full state_dict).

## Risks and open questions

1. **`forward(x, cat_ids)` signature.** GR00T's `CategorySpecificLinear.forward` takes two args. peft's `target_modules` mechanism assumes a single-input `nn.Linear`-like forward. Going via the standalone API is straightforward; the peft adapter has to monkey-patch or subclass to handle the extra `cat_ids` arg. Possible solution: capture `cat_ids` via a forward pre-hook on a containing module and pass to the adapter via a thread-local. Ugly. Acceptable v1: peft integration limited to single-input layers; multi-input layers MUST use the standalone API.

2. **DDP gradient sync.** Per-category adapters are trainable, but a given batch only updates the slices of `A` and `B` for the categories present in that batch (others get zero gradient). DDP all-reduce will still send zero gradients for those slices, which is correct but wasteful. Out of scope to optimize; v1 documents as a known inefficiency.

3. **Initialization.** Standard LoRA inits `B = 0`, `A ~ N(0, σ²)`. For per-category, we apply the same init independently per category. Confirmed safe (identity at init, learns per-category corrections).

4. **`alpha / r` scaling convention.** Match peft's convention. Document that the effective LR per adapter param is `s × (base_lr)` so users know to consider that when comparing to full FT.

5. **Naming conflict with peft's `LoraConfig`.** We expose `LoRAConfig` from our own package; users who want peft's class still import it from peft. Resolved by naming our class differently (e.g. `CategoryLoRAConfig`).

## Implementation plan (sequenced)

1. Write `pyproject.toml`, `__init__.py`, and a stub `layer.py` so `pip install -e .` works.
2. Subagent #1 — test plan: write `tests/` covering each acceptance criterion. Tests should FAIL initially (red).
3. Implement `CategoryLoRALinear` forward + init in `layer.py`.
4. Implement `merge_and_unload`, `merge_adapter`, `unmerge_adapter` on the class.
5. Implement `wrap_in_place` + `CategoryLoRAConfig` in `wrapper.py`.
6. Run tests; iterate until green.
7. Write `peft_adapter.py` with `register_with_peft()`. Tests for it skip if peft not installed.
8. Write `examples/gr00t_action_head.py` (loadable; doesn't actually require downloading the 6 GB GR00T weights — uses a tiny synthetic model with the same structure).
9. Polish README with a 10-line usage example.
10. Tag v0.1.0 in the submodule repo.

## Subagent expectations

The reviewer should:
- Confirm the math is right (the per-category LoRA decomposition).
- Flag any API issues (the `forward(x, cat_ids)` signature problem with peft integration is the obvious one).
- Sanity-check the acceptance criteria.
- Catch any out-of-scope claims that should actually be in v1.

The test-writer should:
- Produce failing tests for each acceptance criterion FIRST.
- Use small synthetic models (no GR00T-scale weights) so tests run in <10 s total.
- Cover: shape correctness, gradient flow, merge precision, in-place wrapping idempotency, dtype handling (bf16 + fp32).
