---
title: History — Conventions & Version Summary
---

# History — Conventions & Version Summary

This directory tracks the version history of the AlphaTafl project. Each major iteration of the codebase is documented in a separate page.

## When to Create a New History Page

A new page should be created in this directory when:

1. **A major version is completed** — a significant set of coordinated changes that alter the architecture, performance characteristics, or API surface of the system. Examples: v0 (initial implementation), v1 (performance optimization), v2 (future feature work).
2. **A new development phase begins** — if work is organized into phases (as with the implementation plan), each phase that changes the system's capabilities or structure warrants a history entry.
3. **A breaking change is introduced** — anything that changes the public API, the data flow between components, or the expected behavior of the system.

Pages do NOT need to be created for:
- Bug fixes
- Documentation updates
- Minor refactoring that doesn't change behavior
- Dependency updates

## Page Naming Convention

Use descriptive names with a version prefix:

- `initial_v0.md` — the baseline implementation
- `optimization_v1.md` — first optimization round
- Future: `feature_v2.md`, `refactor_v3.md`, etc.

All pages should have YAML frontmatter with a `title` field.

## High-Level Version Summary

### v0 — Initial Implementation (Baseline)

The original AlphaTafl system. A fully functional AlphaZero-style self-play system for Hnefatafl:

- **C++ Game Engine** — Complete move generation, capture logic (custodian, edge, throne-assisted, King), win conditions, threefold repetition, Pybind11 bindings.
- **Neural Network** — 14-channel ResNet (10 blocks × 128 filters), dual policy/value heads, Adam optimizer, replay buffer.
- **Self-Play** — Functional MCTS-guided game generation with Dirichlet noise and temperature-based action selection.
- **GUI/CLI** — Pygame board visualization and training management interface.
- **Key bottlenecks**: Batch size 1 GPU inference, Python↔C++ boundary per leaf, Python `state_to_tensor` loops, per-file disk I/O replay buffer, O(4840) `select_child` iteration, `GameState.clone()` per node, O(N) threefold scan.

### v1 — Performance Optimization (Phases 1–4)

Complete rewrite of the hot-path performance architecture. No game rule or network changes:

- **Zero-copy C++ tensor generation** — `to_tensor()`, `batch_to_tensor()`, `get_legal_moves_mask()` in C++ with `py::array_t<float>` zero-copy views.
- **GPU Inference Server** — Single GPU process batches up to 256 states or 50ms. Workers CPU-only.
- **Batched MCTS** — Virtual loss (`visit += 3, value -= 3`), pending-node semantics, stall guard, backward-compatible single-leaf fallback.
- **Undo-stack MCTS** — `apply_move_inplace()`/`undo_move()` replaces `clone()`. `MCTSNode::state` removed. O(1) threefold via `unordered_map`. Ring buffer for board history.
- **Queue-based replay buffer** — No disk I/O on hot path. `torch.multiprocessing.Queue` streaming with backpressure.
- **Model hot-reload** — Inference Server watches `current_best.pt` mtime. Async CPU-cloned checkpointing.
- **Async GUI AI** — MCTS in background thread, responsive window.
- **torch.compile** — `mode="reduce-overhead"` on Inference Server model.
- **DataLoader** — `ReplayDataset` + `DataLoader(batch_size=128)` for training.
- **50 passing tests** across 16 test classes.
- See [[optimization_v1_benchmarks]] for full quantitative results.
