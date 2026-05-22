# Changelog

All notable changes to category-lora will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial PLAN.md (v3, GREEN after 2 rounds of subagent review).
- Repository scaffold: `pyproject.toml`, stub modules, test directory.

## [0.1.0] - TBD

### Added
- `CategoryLoRALinear` — LoRA adapter for category-indexed Linear layers (3D weight tensors).
- `wrap_in_place(model, config, target_class_names=..., target_classes=...)` — drop-in wrapper for whole models.
- `unload_adapters(model)` — model-level merge + tree-swap utility.
- Optional peft integration via `register_with_peft()` (single-input layers only).
- Example: `examples/gr00t_action_head.py` (synthetic GR00T action head).
- CI: GitHub Actions running `pytest` on Python 3.10 with `torch>=2.1`.
