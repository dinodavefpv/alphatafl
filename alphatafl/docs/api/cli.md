---
title: Command Line Interface
---

# Command Line Interface (CLI)

The `cli.py` script is the main management tool for training the AlphaTafl AI.

## Commands

| Command | Description |
|---|---|
| `workers [N]` | Get/set parallel CPU worker count |
| `simulations [N]` | Get/set MCTS simulations per move |
| `batch_size [N]` | Get/set MCTS batch size (v1) |
| `start` | Start continuous background self-play generation (parallel workers + inference server) |
| `stop` | Stop generation (parallel or loop) |
| `train` | Start training loop (creates Trainer, starts queue receiver or loads from disk) |
| `stop_train` | Stop training loop |
| `auto` | Start both generation and training simultaneously |
| `loop` | Start sequential GPU self-play (one game at a time, reloading weights between games) |
| `auto_loop` | Start sequential GPU self-play and training simultaneously |
| `stop_auto` | Stop both generation and training |
| `reset_model confirm` | Delete `models/current_best.pt` |
| `play_one [cpu\|gpu]` | Run a single synchronous self-play game |
| `status` | View generation and training statistics |
| `exit` | Clean shutdown of all components |

## Key Features (v1)

- **Batched mode detection**: Reads `ALPHATAFL_BATCHED` env var at init. Displays mode in `start` and `auto` output.
- **Inference server lifecycle**: Server started as `multiprocessing.Process` in `do_start`, stopped via `stop_event` in `do_stop`.
- **Queue-based training**: When batched mode is active, `do_train` passes `replay_queue` to trainer for streaming buffer.
- **Legacy disk fallback**: When `ALPHATAFL_BATCHED=0`, training loop loads `.pt` files from `data/` every 50 batches.
- **Async checkpointing**: Calls `trainer.maybe_checkpoint(batch_count)` every batch in training loop.

## Architecture

The CLI uses an `Orchestrator` to manage a pool of worker processes and optionally the GPU inference server. In batched mode, workers stream data through queues. In legacy mode, workers save `.pt` files to disk.

## Source Files
- [cli.py](../../../cli.py)
