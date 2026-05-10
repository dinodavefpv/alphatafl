# AlphaTafl Wiki Index

## History
- [Version Summary](./docs/history/readme.md): Conventions and high-level overview of all versions.
- [v0 — Initial Implementation](./docs/history/initial_v0.md): Baseline implementation and known bottlenecks.
- [v1 — Performance Optimization](./docs/history/optimization_v1.md): Phases 1–4 optimization round.
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

## Recent Updates
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
