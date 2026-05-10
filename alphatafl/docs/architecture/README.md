---
title: System Architecture
---

# AlphaTafl System Architecture

AlphaTafl is an AI system designed to play Hnefatafl at a superhuman level using Deep Reinforcement Learning and Monte Carlo Tree Search (MCTS), following the AlphaZero paradigm.

## High-Level Overview

The system is a hybrid application combining C++ for performance-critical simulations and Python for neural network training and high-level orchestration.

### Components

1.  **C++ Engine (`src/engine/`)**:
    *   **Game State**: Implements the rules of Hnefatafl, move generation, and state transitions. Includes zero-copy tensor generation (`to_tensor()`, `batch_to_tensor()`).
    *   **MCTS**: Implements Monte Carlo Tree Search with both single-leaf and batched evaluation paths, virtual loss for parallelism, and undo-based traversal.

2.  **Neural Network (`src/model/`)**:
    *   A Convolutional Residual Neural Network (ResNet) with dual heads:
        *   **Policy Head**: Predicts the probability distribution over 4840 moves.
        *   **Value Head**: Predicts the expected game outcome from the current state.

3.  **Inference Server (`src/training/inference_server.py`, v1)**:
    *   Single GPU process that batches leaf states from workers, runs `model.forward()`, and routes results via per-worker `response_queues`.
    *   Supports hot-reload (watching `current_best.pt` mtime) and optional `torch.compile`.

4.  **Training Loop (`src/training/`)**:
    *   **Self-Play**: Workers (CPU-only) run C++ MCTS and stream game data via `replay_queue`.
    *   **Optimization**: `ReplayDataset` + `DataLoader`, Adam optimizer, minimizes policy + value loss.

5.  **Bindings (`src/bindings/`)**:
    *   Uses **Pybind11** to expose the C++ engine to Python. Zero-copy tensor views.

6.  **Interface**:
    *   **GUI (`gui.py`)**: Pygame-based interface with async AI search (background thread, responsive window).
    *   **CLI (`cli.py`)**: Command-line interface for managing training, workers, and inference server.

## Data Flow (v1 — Batched Mode)

1. **Self-Play**: CPU workers run C++ MCTS (undo-based traversal). Leaf states are batched and sent to the Inference Server via `inference_queue`.
2. **Inference**: The GPU Inference Server batches up to 256 states (or 5ms timeout), runs a single `model.forward()`, returns raw policies/values via per-worker `response_queues`.
3. **Worker-Side Post-Processing**: Workers apply legal move masks and Dirichlet noise (root-only), then return results to C++ MCTS for expansion and backpropagation.
4. **Replay Buffer**: Completed game data is pushed to `replay_queue` (maxsize=50000). Trainer background receiver thread appends to in-memory deque.
5. **Training**: `DataLoader` samples batches of 128 from the deque. Adam optimizer updates network weights on GPU.
6. **Checkpointing**: Every 500 batches, model is saved asynchronously (CPU-cloned state dict, background thread). Inference Server detects the mtime change and hot-reloads.

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
