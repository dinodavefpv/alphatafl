---
title: Self-Play Data Generation
---

# Self-Play Data Generation

Self-play is the process by which AlphaTafl generates high-quality training data by playing against itself.

## Two Self-Play Paths

### Batched Path (v1, default)
Used when `ALPHATAFL_BATCHED=1`. Function: `self_play_game_batched()`.

1. **C++ MCTS** accumulates leaf states until batch is full.
2. **Python callback** calls `GameState.batch_to_tensor()` (one C++ boundary crossing for entire batch), converts to `torch.from_numpy()`, sends to Inference Server via `inference_queue`.
3. **Inference Server** evaluates on GPU, returns `(policy, value)` via per-worker `response_queue`.
4. **Worker** applies legal move masks and Dirichlet noise (root-only), returns results to C++ MCTS.
5. Completed game data pushed to `replay_queue` (no disk I/O on hot path).

### Legacy Single-Leaf Path (fallback)
Used when `ALPHATAFL_BATCHED=0` or by GUI. Function: `self_play_game()`.

1. **MCTS Search**: For each turn, run MCTS guided by the current neural network.
2. **Noise**: Add **Dirichlet noise** to the root node's prior probabilities.
3. **Action Selection**: Sample proportionally for first 30 moves, argmax thereafter.
4. **Recording**: Store `(state tensor, search probabilities, current player)`.
5. **Storage**: Save as `.pt` files in `data/` directory.

## Action Selection
- **Early Game (< 30 moves)**: Sample actions proportionally to visit counts (Temperature $\tau = 1$).
- **Late Game**: Select the action with the maximum visit count ($\tau \to 0$).
- Max 200 turns before forced cutoff (draw).

## Worker Architecture (K3)
- Python `multiprocessing.Pool` processes run on CPU cores.
- Workers import `alphatafl_engine`, `numpy`, `torch` (CPU-only).
- `torch.cuda.is_initialized() == False` in all workers.
- No per-worker CUDA contexts — one GPU Inference Server handles all evaluations.
- Workers use `torch.set_num_threads(1)` to avoid OpenMP contention.
- The `Orchestrator` in `src/training/orchestrator.py` manages the pool, inference server, and queue lifecycle.

## Source Files
- [self_play.py](../../../self_play.py)
- [orchestrator.py](../../../src/training/orchestrator.py)
- [inference_server.py](../../../src/training/inference_server.py)
