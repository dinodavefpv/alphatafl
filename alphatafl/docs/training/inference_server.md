---
title: Inference Server
---

# Inference Server

The Inference Server is a dedicated GPU process introduced in v1 (Phase 2, K2) that replaces per-worker GPU inference with a single batched evaluation pipeline.

> **Note on Phase 5F (Standalone C++ ONNX inference):** As of Phase 5F, the project also supports a **second inference backend** — a standalone C++ `InferenceEngine` that uses ONNX Runtime directly inside the C++ MCTS search thread, bypassing the Python Inference Server, `mp.Queue`, and SHM regions entirely. The two backends are mutually exclusive: when the C++ `InferenceEngine` path is active (via the exported `models/alphatafl.onnx` model and the `InferenceEngine` Pybind11 class — see `docs/api/bindings.md` and `docs/api/interface_contract.md` §6), no Inference Server is started. The rest of this page documents the **Python Inference Server** path; for the C++ path see `docs/history/optimization_v1_phase5.md` §5F.

## Architecture

The server runs as a `multiprocessing.Process` spawned by `cli.py`. It owns the `AlphaTaflNet` model on `cuda:0` in `eval()` + `inference_mode()`. Workers send batched state tensors via `inference_queue` and receive results via per-worker `response_queues`.

### Main Loop

1. Dequeue messages from `inference_queue` in a loop.
2. Accumulate `(worker_id, request_id, states_tensor)` tuples up to `MAX_BATCH` or `TIMEOUT_MS`.
3. Run `model.forward(states_tensor)` once for the entire batch.
4. Apply softmax to policy logits.
5. Route `(request_id, policy_batch, value_batch)` to the correct `response_queues[worker_id]`.
6. Repeat.

### Configuration

| Parameter | Default | Env Variable |
|---|---|---|
| Max batch size | 256 | `ALPHATAFL_INFERENCE_MAX_BATCH` |
| Timeout | 1 ms | `ALPHATAFL_INFERENCE_TIMEOUT_MS` |
| torch.compile | disabled | `ALPHATAFL_COMPILE=1` to enable |
| SHM transport | disabled | `ALPHATAFL_SHM=1` to enable (Phase 5A) |

### Hot-Reload (K5)

Every 10 batches, the server checks `os.path.getmtime("models/current_best.pt")`. If modified, it calls `model.load_state_dict(torch.load(...))` to pick up new weights. Workers never touch model files.

### torch.compile (K8)

When `ALPHATAFL_COMPILE=1` and PyTorch >= 2.0, the model is wrapped with `torch.compile(model, mode="reduce-overhead")`. Verified to produce outputs within 1e-5 of the uncompiled model.

## Lifecycle

- **Start**: CLI creates queues, spawns server process, waits for confirmation, then spawns workers.
- **Stop**: CLI signals `stop_event`. Server exits main loop. Workers receive `EOFError` on response queues and exit.
- **Error handling**: On forward pass failure, zero tensors are returned to workers as fallback.

## Source Files
- [inference_server.py](../../../src/training/inference_server.py)
