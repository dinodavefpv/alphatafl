# AlphaTafl Wiki Index

## History
- [Version Summary](./docs/history/readme.md): Conventions and high-level overview of all versions.
- [v0 — Initial Implementation](./docs/history/initial_v0.md): Baseline implementation and known bottlenecks.
- [v1 — Performance Optimization](./docs/history/optimization_v1.md): Phases 1–4 optimization round.
- [v1 — Phase 5 Optimization](./docs/history/optimization_v1_phase5.md): Phase 5 adaptive timeout, SHM transport, C++ MCTS callback, and sparse priors.
- [v1 — Benchmarks](./docs/history/optimization_v1_benchmarks.md): Compiled quantitative benchmarks from all optimization phases.

## Core Documentation
- [Architecture](./docs/architecture/README.md): High-level system design.

## Components
### C++ Engine
- [Game State](./docs/engine/game_state.md): Core board logic and move generation.
- [MCTS](./docs/engine/mcts.md): Monte Carlo Tree Search implementation.

### Python / Machine Learning
- [Neural Network](./docs/model/resnet.md): ResNet architecture and heads.
- [Training Loop](./docs/training/training_loop.md): The training process.
- [Self-Play](./docs/training/self_play.md): Generating training data.
- [Inference Server](./docs/training/inference_server.md): GPU batch inference server.

### Integration
- [Python Bindings](./docs/api/bindings.md): Pybind11 bridge between C++ and Python.
- [Interface Contract](./docs/api/interface_contract.md): API contract between C++ engine and Python inference pipeline.
- [GUI](./docs/api/gui.md): Pygame-based graphical interface.
- [CLI](./docs/api/cli.md): Command-line interface.

## Benchmark & Verification Tools
- [bench_fp16.py](../benchmarks/bench_fp16.py): FP32 vs FP16 vs AMP inference comparison.
- [bench_multi_worker.py](../benchmarks/bench_multi_worker.py): Multi-worker SHM vs Queue throughput.
- [test_profile.cpp](../tests/cpp/test_profile.cpp): C++ profiling tests (`NodeMemoryFootprint` and `MctsMemoryGrowth`).

## Recent Updates
- **2026-11-12**: Fixed history ring-buffer corruption on multi-ply undo (`UndoInfo.overwritten_history`); MCTS leaves now see correct T-1..T-3 history planes. Updated [Game State](./docs/engine/game_state.md); added regression test `UndoMultiPlyRestoresHistory`.
- **2026-06-01**: Lint pass (round 2). Read every source file (`src/`, `cli.py`, `gui.py`, `self_play.py`, `tests/`, `CMakeLists.txt`) and cross-referenced the current-state docs. Fixed: `is_restricted_square` (claimed Pybind11-bound but not in `bind.cpp`), `to_tensor` (claimed zero-copy but copies into a capsule-managed buffer), `check_threefold` (claimed never called on capture but is called on every non-terminal move), `self_play.md` Batched Path step 4 (claimed the worker applies legal masks / dirichlet — moved to C++ in Phase 5D), `training_loop.md` DataLoader (claimed unconditional `pin_memory=True` — actually conditional on CUDA), `architecture/README.md` (removed two zero-copy claims), and the `mcts.md` notes on `dirichlet_epsilon` defaults and the abandoned `src/training/mcts.py` file.
- **2026-06-01**: Added `AGENTS.md` at the project root so future LLM agents have a single orientation file that points to the canonical wiki at `alphatafl/` and states the hard rules (no root-level `wiki/`, current-state pages only, update index + log on every change).
- **2026-06-01**: Wiki merge + lint pass. Consolidated a stale `wiki/` duplicate at the project root into the canonical Obsidian wiki at `alphatafl/` (the previous `alphatafl/WIKI.md` note pointing to `wiki/` was the source of the split). Audited every page in `alphatafl/docs/` against the live code: documented the Phase 5F `InferenceEngine` (C++ ONNX Runtime), added the `dirichlet_alpha`/`dirichlet_epsilon` constructor parameters to MCTS and bindings, fixed the `ALPHATAFL_INFERENCE_TIMEOUT_MS` default to `1ms` everywhere it was wrong, and reconciled the test-count rows (now 62 Python / 36 C++).
- **2026-05-31**: Step 3 (ONNX C++ Inference) completed. Migrated the C++ `InferenceEngine` from LibTorch to standalone ONNX Runtime, removing PyTorch/LibTorch as a build dependency. Synthetic MCTS throughput rose to **81,439 states/sec** (62% over baseline). E2E turn time is **245.5 ms / turn** with zero Python IPC.
- **2026-05-28**: Step 2 (Sparse Priors) completed. Reduced node memory footprint by 18x to **1.078 KB / node** (aggregate tree memory down to **1.56 MB**). Yielded a **22% C++ MCTS throughput increase** (to 61,072 states/sec) due to cache-friendly sequential selection. Turn time reduced to **207 ms / turn**.
- **2026-05-28**: Step 1 (C++ Callback) completed. Moved legal move masking, softmax normalization, and Dirichlet noise generation into native C++. Eliminated Python callback list build overhead (**6.5 ms -> 0.0 ms**), reducing 800-sim E2E Turn Time to **216 - 244 ms / turn**.
- **2026-05-09**: Callback micro-profile confirms bottleneck is Python process boundary. mp.Queue.get 13.9ms (58%) + list build 6.5ms (27%) + pybind11 wrap 3.0ms (12%) = 23.8ms/batch. 84% eliminable via C++ native inference (libtorch/ONNX).
- **2026-05-09**: Deep-tree profile + single-game timing corrects bottleneck analysis. C++ MCTS is fast (23µs/sim, flat with tree depth). Real bottleneck is Python callback per batch: mp.Queue pipes + list construction = ~42ms/batch (65% of 346ms/turn). libtorch/ONNX Runtime reprioritized as next step.
- **2026-05-09**: Batch size sweep confirms redundancy (bs ≥ sims = identical throughput) but disproves IPC-only projection model.
- **2026-05-09**: Shared memory IPC implemented — `ALPHATAFL_SHM=1` eliminates mp.Queue bottleneck. Peak throughput 275 st/s at 20 workers.
- **2026-05-09**: Multi-worker throughput benchmarks completed — 0ms timeout causes server starvation, 5ms timeout scales super-linearly to 12 workers, GPU saturates at 20 workers (133 st/s peak). Shared memory IPC identified as required next step.
- **2026-05-08**: v1 optimization complete — batched inference, undo-stack MCTS, queue-based replay buffer, hot-reload, async GUI, torch.compile, 50 passing tests.
- **2026-05-08**: Phase 3 delivered — undo-stack, ring buffer, O(1) threefold, MCTSNode state removal, model hot-reload, async GUI AI.
- **2026-05-08**: Phase 2 delivered — GPU Inference Server, batched MCTS with virtual loss, per-worker response queues.
- **2026-05-08**: Phase 1 delivered — C++ zero-copy tensors, sparse select_child, interface contract, test framework.
- **2026-05-08**: Completed comprehensive documentation of the entire AlphaTafl system.
