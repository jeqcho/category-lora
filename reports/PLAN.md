# category-lora — Implementation Plan

**Status:** v3 (post-subagent-review-round-2; addresses 1 HIGH)
**Date:** 2026-05-21

## v3 changelog (vs v2)

Round-2 reviewer flagged 1 HIGH:

- **[HIGH] DDP workaround #3 was wrong.** The plan proposed a post-backward hook to zero unused gradients, but DDP hangs *inside* `loss.backward()` at the allreduce barrier — too early for a post-backward hook to help. v3: removed option 3. Users with unbalanced category sampling MUST use `find_unused_parameters=True`. Also dropped the `fix_unused_grads` utility from the API; it can't do what its name implied.

## v2 changelog (vs v1)

Round-1 reviewer flagged 2 BLOCKERs, 4 HIGH, 4 MEDIUM, 2 LOW. All addressed:

- **[BLOCKER] Math notation transposed/inconsistent.** v2: one canonical layout — `A: (C, in, r)`, `B: (C, r, out)`, `delta = A @ B` shape `(in, out)`. All forward / merge equations rewritten in this layout with explicit shape annotations.
- **[BLOCKER] "≤1 ULP drift in bf16" unachievable.** v2: replaced with `atol=1e-3, rtol=1e-2` in bf16 (peft's convention) and `atol=1e-6` in fp32.
- **[HIGH] Thread-local peft shim is broken under DP / `torch.compile`.** v2: dropped from plan. peft path is registration-only for single-input layers; multi-input layers use standalone API. Documented as v0.1 limit.
- **[HIGH] Name inconsistency `LoRAConfig` vs `CategoryLoRAConfig`.** v2: unified on `CategoryLoRAConfig` across API examples, `__init__.py` exports, and Risks.
- **[HIGH] DDP `find_unused_parameters=False` will hang on unused category slices.** v2: explicit "DDP usage" section that documents this and the workarounds.
- **[HIGH] Missing save/load round-trip acceptance criterion.** v2: added as criterion #5.
- **[MEDIUM] `merge_and_unload()` return semantics ambiguous.** v2: explicit return type contract — mutates `self.base_layer` in-place AND returns it for caller convenience; a model-level `unload_adapters(model)` walks the tree and swaps adapters back to their base layers in the parent's children dict.
- **[MEDIUM] σ for `A` init unspecified.** v2: matches peft — `nn.init.kaiming_uniform_(A, a=math.sqrt(5))` and `B = 0` (so `B@A = 0` at init regardless of A's scale).
- **[MEDIUM] `peft==0.x` invalid pip spec.** v2: `peft>=0.10,<1.0`.
- **[MEDIUM] String-only `target_class_names` is fragile.** v2: `wrap_in_place` accepts both `target_class_names: list[str]` AND `target_classes: list[type]`.
- **[LOW] Library hygiene.** v2: added explicit steps for `pyproject.toml` URLs/version, `__version__`, `CHANGELOG.md`, GitHub Actions CI, Google-style docstrings.
- **[LOW] Examples in CI.** v2: synthetic GR00T-shaped example is included in pytest, runs in <5s.

## What this package is

A small PyTorch library that adds LoRA-style low-rank adapters to **category-indexed Linear layers** — layers whose weight is a 3D tensor `(num_categories, in_features, out_features)`, where at runtime a per-input category id selects one slice of the weight to compute `y = x @ W[cat] + b[cat]`.

Used in **multi-task / multi-embodiment VLA models** (NVIDIA GR00T's `CategorySpecificLinear`, `MultiEmbodimentActionEncoder`, `CategorySpecificMLP`) and **mixture-of-experts** with hard expert selection. Standard peft can't wrap them because the weight is 3D and the forward takes a second `cat_ids` arg.

## API surface (v2 canonical)

### Standalone path (primary)

```python
from category_lora import CategoryLoRALinear

adapter = CategoryLoRALinear(
    base_layer=original_layer,   # .W shape (C, in, out); .b shape (C, out); forward(x, cat_ids)
    r=16,
    alpha=32,
    dropout=0.0,
)

# Forward — same signature as base
y = adapter(x, cat_ids)            # x: (B, in), cat_ids: (B,) → y: (B, out)

# Reversible merge
adapter.merge_adapter()             # in-place; sets self._merged=True
y2 = adapter(x, cat_ids)            # uses merged base; LoRA path skipped
adapter.unmerge_adapter()           # restores base weights (approximately, in bf16)

# Irreversible merge + unwrap
base_returned = adapter.merge_and_unload()
# mutates self.base_layer.W in-place AND returns self.base_layer for caller convenience.
# After this, adapter.merge_and_unload() raises if called again (state guard).
```

### Drop-in path (recommended for whole-model wrapping)

```python
from category_lora import wrap_in_place, CategoryLoRAConfig
# Match either by class name (string) or by class object — both supported.
cfg = CategoryLoRAConfig(r=16, alpha=32, dropout=0.0)
n_wrapped = wrap_in_place(
    model,
    target_class_names=["CategorySpecificLinear", "MultiEmbodimentActionEncoder"],
    # OR: target_classes=[CategorySpecificLinear, MultiEmbodimentActionEncoder],
    config=cfg,
)
# Returns the number of replaced modules. Base params have requires_grad=False;
# adapter A/B params have requires_grad=True. Pass the model to your optimizer as usual.

# At training end:
from category_lora import unload_adapters
unload_adapters(model)              # merges each adapter and swaps the parent's child reference
                                    # back to the (now-merged-in-place) base layer.
```

### peft path (optional, single-input layers only in v0.1)

```python
import peft
from category_lora.peft_adapter import register_with_peft

register_with_peft()                # idempotent; teaches peft about our adapter
peft_model = peft.get_peft_model(
    model,
    peft.LoraConfig(r=16, lora_alpha=32, target_modules=["W"]),
)
```

v0.1 limit: peft path only works for category-indexed layers whose `forward` takes a single tensor input (rare). The standard GR00T case has `forward(x, cat_ids)` and **must use the standalone or drop-in path**. This is documented in the README and enforced by `register_with_peft` raising a clear error if the user tries to use it for a multi-input layer.

## Math (v2 canonical layout)

### Notation

For a base `CategorySpecificLinear` module:
- Base weight: `W ∈ ℝ^(C × in × out)`
- Base bias:   `b ∈ ℝ^(C × out)`
- Input batch: `x ∈ ℝ^(B × in)`, `cat_ids ∈ ℤ^B`
- Base forward: `y[i] = x[i] @ W[cat_ids[i]] + b[cat_ids[i]]` → `y ∈ ℝ^(B × out)`

For the LoRA adapter:
- `A ∈ ℝ^(C × in × r)` (one matrix per category)
- `B ∈ ℝ^(C × r × out)` (one matrix per category)
- Scale: `s = alpha / r` (scalar)

### Forward (additive form, no merge)

For each batch element `i` with category `c = cat_ids[i]`:
```
y[i] = x[i] @ W[c] + b[c] + s · (x[i] @ A[c]) @ B[c]
                              └── shape (r,) ──┘
        └── shape (out,) ────┘  └── shape (out,) ────┘
```

Batched (einsum):
```python
base_out  = einsum("bi,bio->bo", x, W[cat_ids])    # (B, out)
lora_mid  = einsum("bi,bir->br", x, A[cat_ids])    # (B, r)
lora_out  = einsum("br,bro->bo", lora_mid, B[cat_ids])  # (B, out)
y = base_out + b[cat_ids] + scale * lora_out
```

### Merge (in-place, reversible)

For each category `c`:
```
W[c] := W[c] + s · (A[c] @ B[c])
       ↑         ↑
       (in, out)  (in, r) @ (r, out) = (in, out)
```

Implementation (vectorized over categories):
```python
delta = scale * torch.bmm(A, B)                    # (C, in, out)
self.base_layer.W.data.add_(delta.to(self.base_layer.W.dtype))
```

### Unmerge

```python
delta = scale * torch.bmm(A, B)
self.base_layer.W.data.sub_(delta.to(self.base_layer.W.dtype))
```

**Precision contract:**
- Merge/unmerge done in `A`/`B`'s native dtype (typically fp32 for training), then `.to(W.dtype)` for the in-place op.
- Round-trip `merge_adapter → unmerge_adapter` is bit-exact in fp32, ≤`atol=1e-3, rtol=1e-2` in bf16 (matches peft's testing tolerance).

### Initialization (v2: matches peft)

```python
nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # same as peft.lora_A init
nn.init.zeros_(self.B)                              # ensures delta = 0 at init
```

`B = 0` guarantees `A @ B = 0` at init regardless of `A`'s distribution, so the adapter starts as identity (`y = base output exactly`). First test verifies this.

## Parameter counts (for typical GR00T projector layer)

Base `CategorySpecificLinear` with `C=32, in=1536, out=1536`:
- Base: 32 × 1536 × 1536 = **75.5M params**

LoRA adapter at r=16:
- `A`: 32 × 1536 × 16 = 786 432
- `B`: 32 × 16 × 1536 = 786 432
- Total adapter: **~1.57M params** (~2.1% of base)

50× reduction on the category-Linear's trainable cost. With ~6 such layers in a GR00T action head (~450M base), full-projector LoRA → ~10M adapter params.

## DDP usage (v2: explicit)

When training under `torch.nn.parallel.DistributedDataParallel`, the per-category `A[c]` and `B[c]` slices are only touched if category `c` appears in a given batch on a given rank. DDP's default `find_unused_parameters=False` requires every parameter with `requires_grad=True` to receive a non-None gradient each step — categories absent from a rank's batch will produce `None` gradient slices and the reducer will hang.

**Two correct usage patterns** (documented in README; one MUST be applied for distributed training):
1. Set `find_unused_parameters=True` in DDP constructor. Adds a small overhead but is the simplest fix.
2. Use balanced category sampling across ranks/batches so every category is hit every step.

(A third option — post-backward gradient zeroing — was considered and rejected in plan v3 round-2 review: it can't work because DDP hangs INSIDE `loss.backward()` at the allreduce barrier, before any post-backward hook would fire. The only correct mechanism to "always produce a gradient" would be a forward-time ghost contribution like `output += 0 * (A.sum() + B.sum())` that forces every slice into the autograd graph. This adds compute overhead, regresses sparse efficiency, and is brittle — easier to just enable `find_unused_parameters=True`.)

A test in `test_layer_forward.py` simulates a single-rank "minibatch missing some categories" case and asserts that the present-category slices receive non-zero gradients while absent-category slices receive zero/`None` gradients — documenting the DDP-incompatible failure pattern.

## File layout

```
category-lora/
├── README.md                       # short pitch + 10-line usage
├── CHANGELOG.md                    # v2: added
├── LICENSE (MIT)
├── pyproject.toml                  # v2: full metadata + version + URLs + deps
├── .github/workflows/ci.yml        # v2: added (pytest on py3.10 with torch 2.5 + peft 0.10+)
├── src/category_lora/
│   ├── __init__.py                 # exports: CategoryLoRALinear, CategoryLoRAConfig,
│   │                               #          wrap_in_place, unload_adapters, __version__
│   ├── layer.py                    # CategoryLoRALinear class
│   ├── wrapper.py                  # CategoryLoRAConfig, wrap_in_place, unload_adapters
│   └── peft_adapter.py             # register_with_peft (optional)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # synthetic CategorySpecificLinear fixture
│   ├── test_layer_forward.py       # shape, init, identity-at-init, multi-batch, DDP grad mask
│   ├── test_layer_merge.py         # merge/unmerge round-trip, merge_and_unload, dtype handling
│   ├── test_state_dict.py          # save/load round-trip (v2: new)
│   ├── test_wrapper.py             # wrap_in_place by name AND by class, unload_adapters
│   ├── test_peft_adapter.py        # marked optional; skipped if peft not installed
│   └── test_gr00t_example.py       # synthetic GR00T-shaped end-to-end (v2: in CI)
├── examples/
│   └── gr00t_action_head.py        # human-readable walkthrough (test_gr00t_example covers same code)
└── reports/
    ├── PLAN.md                     # this file
    └── RUN_LOG.md                  # appended as we go
```

## pyproject.toml content (v2: explicit)

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "category-lora"
version = "0.1.0"
description = "LoRA for category-indexed Linear layers (3D weight tensors)"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [{name = "jeqcho"}]
dependencies = [
    "torch>=2.1",
]

[project.optional-dependencies]
peft = ["peft>=0.10,<1.0"]
dev  = ["pytest>=7", "pytest-cov"]

[project.urls]
Homepage = "https://github.com/jeqcho/category-lora"
Repository = "https://github.com/jeqcho/category-lora"
Issues = "https://github.com/jeqcho/category-lora/issues"

[tool.setuptools.packages.find]
where = ["src"]
```

## Acceptance criteria (v2)

1. `CategoryLoRALinear.forward(x, cat_ids)` matches the math above: verified against a manual reference impl in `test_layer_forward.py`, atol=1e-6 fp32, atol=1e-3 bf16.
2. At init, `CategoryLoRALinear.forward(x, cat_ids)` is bit-exact equal to `base_layer.forward(x, cat_ids)` (because `B = 0`).
3. `merge_and_unload()` then `base.forward(x, cat_ids)` equals the additive `adapter.forward(x, cat_ids)` within precision tolerance.
4. `merge_adapter() → unmerge_adapter()` round-trip: bit-exact in fp32, ≤`atol=1e-3, rtol=1e-2` in bf16. `merge_and_unload()` is a one-shot terminal operation; calling twice raises.
5. **(v2 new)** Save/load round-trip: `torch.save(adapter.state_dict(), p); fresh = CategoryLoRALinear(base); fresh.load_state_dict(torch.load(p))` → forward matches original exactly in fp32.
6. `wrap_in_place(model, target_class_names=[...])` and `wrap_in_place(model, target_classes=[...])` both replace matching modules; base params end with `requires_grad=False`, adapter params with `requires_grad=True`; non-matching modules untouched.
7. `unload_adapters(model)` walks the model tree, calls `merge_and_unload()` on each `CategoryLoRALinear`, swaps the parent's child reference back to the base layer.
8. peft adapter: `register_with_peft()` is idempotent. For single-input category-indexed layers, `get_peft_model(model, LoraConfig(...))` produces a model whose state_dict has the expected keys. For multi-input layers, `register_with_peft` either skips or raises a clear error (TBD; chosen during implementation based on what's least surprising).
9. ≥90% line coverage on `layer.py` and `wrapper.py` (`pytest --cov`).
10. README: a runnable 10-line example that imports the package, wraps a synthetic `CategorySpecificLinear`, does one forward + backward, and shows the loss decreased.
11. CI (GitHub Actions): on push/PR to `main`, runs `pytest -v --cov=category_lora` on Python 3.10 with `torch>=2.1` and `peft>=0.10`. Passes.

## Out of scope (v0.1)

- DoRA / weight-decomposed LoRA.
- AdaLoRA-style rank scheduling.
- Distributed FSDP-specific testing (should "just work" given parameter shapes, but not in CI matrix).
- Triton kernels for batched per-category matmul (default is `torch.bmm` + fancy indexing).
- Saving peft-style adapter-only checkpoints (v0.1 ships full-model `state_dict` round-trip only).
- Multi-input layers via the peft path (standalone path covers this).

## Risks (v2)

1. **peft API stability.** `peft>=0.10,<1.0` is the v0.1 pin; peft's `register_layer_pattern` API has changed across minor versions. v0.2 may need to bump the upper pin if peft's API breaks compatibility. Acceptable risk for v0.1.
2. **`merge_and_unload` swap semantics.** The plan specifies in-place mutation + return of `base_layer`. Tests verify the parent's child reference (after `unload_adapters(model)`) points to the merged base, not the adapter. If a user calls `adapter.merge_and_unload()` standalone (without using `unload_adapters`), they get a merged base but the model still has the adapter in its tree — documented as a footgun.
3. **DDP `find_unused_parameters`.** Documented above; users must pick one of three patterns. Out of v0.1 scope to autodetect/fix.

## Implementation plan (sequenced)

1. Create `pyproject.toml`, `src/category_lora/__init__.py` (with `__version__`), `CHANGELOG.md`, `.github/workflows/ci.yml`. Verify `pip install -e .` works.
2. **Subagent #2 (test-writer):** writes `tests/` covering every acceptance criterion. Tests should FAIL initially (red); test_state_dict, test_layer_merge, test_wrapper, test_gr00t_example must produce concrete failures. Tests use synthetic fixtures in `conftest.py` (no large model downloads).
3. Implement `CategoryLoRALinear` (forward + init) in `layer.py`. Run forward tests → green.
4. Add `merge_adapter`, `unmerge_adapter`, `merge_and_unload` to the class. Run merge tests → green.
5. Implement `CategoryLoRAConfig`, `wrap_in_place`, `unload_adapters` in `wrapper.py`. Run wrapper tests → green.
6. Implement `peft_adapter.py::register_with_peft()`. Run peft tests → green (skip if peft unavailable in CI matrix).
7. Write `examples/gr00t_action_head.py` (synthetic GR00T-shaped model). `test_gr00t_example.py` covers it.
8. README polish (10-line snippet, install instructions, DDP note).
9. CHANGELOG entry: "v0.1.0 — initial release with CategoryLoRALinear, wrap_in_place, peft integration (single-input only)."
10. Tag v0.1.0 on the submodule repo. Push.

Commit + push cadence: after each step that finishes a logical unit (after pyproject works, after each test file lands, after each implementation file passes tests).

## Subagent expectations

The reviewer (this round) should:
- Confirm the v2 math is now consistent and shape-correct.
- Confirm the BLOCKER/HIGH/MEDIUM/LOW findings are all resolved (or rebuke the ones that aren't).
- Catch any new issues introduced by the v2 edits.

The test-writer (next subagent) should:
- Produce concrete failing tests for every acceptance criterion BEFORE I implement.
- Use small synthetic modules (no GR00T-scale weights); whole test suite runs <10 s.
- Cover dtype handling (bf16 + fp32), gradient flow, in-place wrapping idempotency, DDP grad-mask scenario.
- Tests are deterministic (`torch.manual_seed(42)` in conftest).
