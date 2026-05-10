---
title: The Training Loop
---

# The Training Loop

The training loop optimizes the neural network using data generated from self-play.

## Components

### Replay Buffer (K7)
- An in-memory buffer (`collections.deque` with `maxlen=200000`) that stores `(state, policy, value)` tuples.
- **Queue-based streaming** (v1): Workers push CPU tensors directly to `torch.multiprocessing.Queue` with `maxsize=50000`. Trainer runs a background receiver thread that appends to the deque.
- **Legacy disk fallback** (v0): When `ALPHATAFL_BATCHED=0`, data is loaded from `.pt` files in the `data/` directory. Loaded files are tracked by name to avoid duplicates.
- **Backpressure**: When the trainer falls behind, workers block on `Queue.put()` instead of consuming unbounded memory.
- **Crash recovery**: On restart, the trainer loads `current_best.pt` and begins generating data anew. No periodic disk checkpoint of the buffer.

### Trainer
- **Initialization**: Loads `current_best.pt` if it exists, creates Adam optimizer (`lr=0.001`, `weight_decay=1e-4`).
- **DataLoader (K9)**: `ReplayDataset` wrapper + `torch.utils.data.DataLoader(batch_size=128, shuffle=True, num_workers=0, pin_memory=True)`. `num_workers=0` avoids pickling overhead since the deque is in the main process.
- **Training step**: Samples batch, computes combined policy cross-entropy + value MSE loss, backpropagates, steps optimizer.

## Loss Function

The network is trained to minimize:
$L = (z - v)^2 - \pi \log p$

1.  **Value Loss**: Mean Squared Error between predicted value $v$ and actual outcome $z$.
2.  **Policy Loss**: Cross-Entropy between predicted policy $p$ and MCTS search probabilities $\pi$.
3.  **Regularization**: L2 regularization applied via Adam's `weight_decay` parameter.

## Checkpointing (K5)

- **Async save**: `save_model(sync=False)` clones state dict to CPU (`{k: v.cpu().clone() for k, v ...}`) and saves in a background thread. Training continues without stalls on GPU tensors.
- **Checkpoint frequency**: `current_best.pt` every 500 training batches, versioned `alphatafl_vXXXXX.pt` every 5000 batches.
- Versioned checkpoint numbering scans existing files on init to resume correctly.
- Training loss is printed every 100 batches.

## Source Files
- [trainer.py](../../../src/training/trainer.py)
- [replay_buffer.py](../../../src/training/replay_buffer.py)
