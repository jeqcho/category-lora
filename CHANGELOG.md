# Changelog

All notable changes to category-lora will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial PLAN.md (v3, GREEN after 2 rounds of subagent review).
- Repository scaffold: `pyproject.toml`, stub modules, test directory.

## [0.1.2] - 2026-05-22

### Fixed
- ``CategoryLoRALinear.forward`` now accepts arbitrary leading dims
  (``(B, *, in)``). v0.1.1's einsum was 2D-only and crashed at
  training step 1 when wrapping GR00T's projector layers, which see
  ``(B, T, state_dim)``. The base path already supported it; the LoRA
  branch now matches.

### Added
- Tests for 3D and 4D input shapes in ``tests/test_layer_forward.py``.
- The synthetic ``SyntheticCategoryLinear`` fixture now supports
  arbitrary leading dims, matching GR00T's real shape contract.

## [0.1.1] - 2026-05-21

### Changed
- ``state_dict()`` now returns the **full state** (base + adapter), making the
  package compatible with HF Trainer save/load and ``model.save_pretrained``
  flows. Previously the auto-applied state_dict hook stripped the base which
  broke checkpoint resumption.

### Added
- ``CategoryLoRALinear.adapter_state_dict()`` — opt-in API for the v0.1.0-style
  adapter-only save (returns ``{"A": ..., "B": ...}``).
- ``CategoryLoRALinear.load_adapter_state_dict(sd)`` — companion loader.

### Migration
- If you were calling ``adapter.state_dict()`` and expecting adapter-only
  output, switch to ``adapter.adapter_state_dict()``. The default state_dict
  now includes the base layer.

## [0.1.0] - 2026-05-21

### Added
- `CategoryLoRALinear` — LoRA adapter for category-indexed Linear layers (3D weight tensors).
- `wrap_in_place(model, config, target_class_names=..., target_classes=...)` — drop-in wrapper for whole models.
- `unload_adapters(model)` — model-level merge + tree-swap utility.
- Optional peft integration via `register_with_peft()` (single-input layers only).
- Example: `examples/gr00t_action_head.py` (synthetic GR00T action head).
- CI: GitHub Actions running `pytest` on Python 3.10 with `torch>=2.1`.
