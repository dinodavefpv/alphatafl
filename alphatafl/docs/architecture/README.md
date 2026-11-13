---
title: System Architecture
---

# AlphaTafl System Architecture

AlphaTafl is an AI system designed to play Hnefatafl at a superhuman level using Deep Reinforcement Learning and Monte Carlo Tree Search (MCTS), following the AlphaZero paradigm.

## High-Level Overview

The system is a hybrid application combining C++ for performance-critical simulations and Python for neural network training and high-level orchestration.

### Components

1.  **C++ Engine (`src/engine/`)**:
    *   **Game State**: Implements the rules of Hnefatafl, move generation, and state transitions. Generates board tensors in C++ (`to_tensor()`, `batch_to_tensor()`) and hands them to Python via capsule-managed numpy buffers.
    *   **MCTS**: Implements Monte Carlo Tree Search with both single-leaf and batched evaluation paths, virtual loss for parallelism, and undo-based traversal.

2.  **Neural Network (`src/model/`)**:
    *   A Convolutional Residual Neural Network (ResNet) with dual heads:
        *   **Policy Head**: Predicts the probability distribution over 4840 moves.
        *   **Value Head**: Predicts the expected game outcome from the current state.

3.  **Inference Server (`src/training/inference_server.py`, v1) & C++ ONNX Backend**:
    *   **Inference Server**: Single GPU process that batches leaf states from workers, runs `model.forward()`, and routes results via per-worker `response_queues`. Supports hot-reload and optional `torch.compile`.
    *   **C++ ONNX Inference**: In-process inference engine (`InferenceEngine`) running ONNX Runtime on CUDA directly inside worker threads. Eliminates background server processes and IPC queue overhead for single-worker or low-worker workloads.

4.  **Training Loop (`src/training/`)**:
    *   **Self-Play**: Workers (CPU-only) run C++ MCTS and stream game data via `replay_queue`.
    *   **Optimization**: `ReplayDataset` + `DataLoader`, Adam optimizer, minimizes policy + value loss.

5.  **Bindings (`src/bindings/`)**:
    *   Uses **Pybind11** to expose the C++ engine to Python. Tensors are returned as `py::array_t<float>` backed by capsule-managed C++ buffers (single C++→Python copy, no Python list allocation).

6.  **Interface**:
    *   **GUI (`gui.py`)**: Pygame-based interface with async AI search (background thread, responsive window).
    *   **CLI (`cli.py`)**: Command-line interface for managing training, workers, and inference server.

## Data Flow (v1 — Batched Mode)

1. **Self-Play**: CPU workers run C++ MCTS (undo-based traversal). Leaf states are batched and sent to the Inference Server via `inference_queue`.
2. **Inference**: The GPU Inference Server batches up to 256 states (or 1ms timeout — see Phase 5A's timeout sweep in `docs/history/optimization_v1_phase5.md`), runs a single `model.forward()`, returns raw policies/values via per-worker `response_queues`.
3. **Worker-Side Pre-Processing**: Workers obtain the legal-move mask from `GameState::get_legal_moves_mask()` and return both the raw softmaxed policy and the mask to C++ MCTS. The actual legal-move masking, softmax renormalization, and root-only Dirichlet mixing happen **inside the C++ `MCTSNode::expand` callback** (Phase 5D) — see `docs/api/interface_contract.md` §7. The C++ `MCTS` constructor is initialized with `dirichlet_alpha=0.3, dirichlet_epsilon=0.25` from `self_play_game_batched`, so the noise is generated in C++ and never crosses the Python boundary on the hot path.
4. **Replay Buffer**: Completed game data is pushed to `replay_queue` (maxsize=50000). Trainer background receiver thread appends to in-memory deque.
5. **Training**: `DataLoader` samples batches of 128 from the deque. Adam optimizer updates network weights on GPU.
6. **Checkpointing**: Every 500 batches, model is saved asynchronously (CPU-cloned state dict, background thread) with an atomic temp-file + `os.replace` write. Inference Server detects the mtime change and hot-reloads.

> [!NOTE]
> **Standalone C++ ONNX Inference Mode (Phase 5F / Step 3)**:
> A C++ `InferenceEngine` (ONNX Runtime on CUDA, reading `models/alphatafl.onnx`) is built and bound to Python, reaching 81K states/sec in synthetic benchmarks. It is not wired into the self-play pipeline: `self_play_game_batched` still evaluates through the Python Inference Server, and the ONNX file is only refreshed by running `scripts/export_onnx.py` manually.

## Hardware

*   **CPU**: i9-14900K (24 cores) used for parallel MCTS simulations.
*   **GPU**: RTX 5080 (16GB VRAM) used for neural network inference and training.
*   **RAM**: 128GB used to house the massive replay buffer in memory for zero-latency sampling.

---
See also:
- [Game Rules](../engine/game_rules.md)
- [C++ Engine: Game State](../engine/game_state.md)
- [C++ Engine: MCTS](../engine/mcts.md)
- [Neural Network](../model/resnet.md)
- [Inference Server](../training/inference_server.md)
- [History](../history/readme.md)
