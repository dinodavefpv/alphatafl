---
title: AlphaTafl Inference Interface Contract
---

# AlphaTafl Inference Interface Contract

> **Version:** v1.2  
> **Status:** Phase 5F — Complete (Phase 5D/5E/5F evolved legal-move handling and added the C++ ONNX inference path)  
> **Owners:** Deepseek (C++ Engine) · Kimi (Python Architecture)

This document is the single source of truth for the contract between the C++ Hnefatafl engine and the Python inference / training pipeline. It covers the two inference backends the project supports today: the Python Inference Server (the historical default) and the standalone C++ ONNX Runtime `InferenceEngine` (Phase 5F, no IPC).

---

## 1. Data Types

### Input Tensor
- **Format:** `numpy.ndarray` / `torch.Tensor`
- **Shape:** `(N, 14, 11, 11)`
- **Dtype:** `float32`
- **Source:** `GameState::batch_to_tensor()` (C++ → `py::array_t<float>` → `torch.from_numpy()`)
- **Channel Layout:**
  - `0–2`:   Attackers, Defenders, King (Current Turn, T)
  - `3–5`:   Attackers, Defenders, King (T‑1)
  - `6–8`:   Attackers, Defenders, King (T‑2)
  - `9–11`:  Attackers, Defenders, King (T‑3)
  - `12`:    Turn flag (`1.0` = Attacker, `0.0` = Defender)
  - `13`:    Restricted squares (`1.0` at corners and throne)

### Output Tensors
- **Policy:** `torch.Tensor` shape `(N, 4840)`, `float32` (Inference Server) — softmax probabilities; legal-move masking now happens inside the C++ callback (see §7).
- **Value:** `torch.Tensor` shape `(N,)`, `float32`.

The Inference Server returns **softmax probabilities** over all 4840 actions. As of Phase 5D, legal-move masking and softmax renormalization are performed **inside the C++ `MCTSNode::expand` callback** (see `src/engine/mcts.cpp`). When the C++ `InferenceEngine` path is used instead of the Inference Server, softmax, legal masking, and Dirichlet noise all happen synchronously in the same thread.

---

## 2. Batch Bounds & Timeout

| Parameter | Default | Env Variable |
|---|---|---|
| Max batch size | 256 | `ALPHATAFL_INFERENCE_MAX_BATCH` |
| Partial-batch timeout | 1 ms | `ALPHATAFL_INFERENCE_TIMEOUT_MS` |
| Min batch size | 1 | — |

The Inference Server dequeues states until either `MAX_BATCH` are collected or `TIMEOUT_MS` elapse, then runs `model.forward()` once. The default of **1 ms** (lowered from 5 ms in Phase 5A — see `docs/history/optimization_v1_phase5.md`) was identified by sweep as the optimal window for cross-worker batching. The C++ `InferenceEngine` path is not affected by this timeout; it runs synchronously.

---

## 3. Virtual Loss

Standard AlphaZero formulation used during batched tree traversal:

```cpp
node.visit_count += 3;
node.value_sum   -= 3;
node.is_pending   = true;
```

When the batch result arrives:

```cpp
node.visit_count -= 3;   // remove virtual visits
node.value_sum   += 3;    // remove virtual penalty
node.visit_count += 1;    // count real visit
node.value_sum   += actual_value;
node.is_pending   = false;
node.expand(policy);
```

---

## 4. Batched MCTS Algorithm (Single-Threaded)

The C++ `MCTS::search()` runs in a **single thread**; parallelism is achieved via virtual loss, which makes pending nodes unattractive to subsequent traversals.

### Pseudocode

```
1. Evaluate root state, expand root.
2. For i in 0..num_simulations-1:
   a. Select from root using PUCT (skip nodes where is_pending == true).
   b. When reaching an unexpanded, non-terminal node:
      - If node is NOT pending:
        * Clone GameState from traversal state.
        * Set is_pending = true.
        * Add virtual loss (visit += 3, value -= 3).
        * Add node to pending batch.
        * Start next simulation from root.
      - If node IS pending:
        * Increment consecutive_all_pending_stall.
        * Start next simulation from root.
   c. When reaching a terminal node:
      * Use game outcome as leaf value.
      * Backpropagate immediately.
   d. If all children of current node are pending:
      * Increment consecutive_all_pending_stall.
      * If counter > 10: force-flush pending batch immediately.
      * Otherwise: start new simulation from root.
      * Reset counter on any successful expansion.
3. When pending batch is full (batch_size) or all simulations queued or force-flushed:
   a. Call eval_fn_batched(pending_states) via Python callback.
   b. For each result (policy, value):
      * Remove virtual loss.
      * Expand node with policy.
      * Set is_pending = false.
      * Backpropagate value via parent-pointer chain.
4. Repeat from step 2 if simulations remain after batch evaluation.
```

### Pending-Node Double-Queue Guard
A node that is already pending must **never** be added to the batch a second time. Traversals that reach a pending node increment `consecutive_all_pending_stall` and restart from root.

### All-Pending Stall Guard
If a traversal finds that **all children** of the current node are pending, `consecutive_all_pending_stall` increments. If the counter exceeds 10, the pending batch is force-flushed immediately, regardless of size.

### Parent-Pointer Backpropagation
Backpropagation uses the existing `MCTSNode::parent` pointer chain. No separate path stack is required.

---

## 5. Transport Definitions (Inference Server path only)

The Inference Server supports two transport modes for tensors, both of which use `torch.multiprocessing.Queue` for lightweight control messages.

### 5.1 Queue Mode (default)

#### `inference_queue` (Workers → Inference Server)
- **Message:** `(worker_id: int, request_id: int, states_tensor: torch.Tensor)`
- **states_tensor shape:** `(N, 14, 11, 11)` on CPU
- **Purpose:** Submit batches of leaf states for GPU evaluation.

#### `response_queues` (Inference Server → Workers)
- **Type:** `List[torch.multiprocessing.Queue]` — one per worker.
- **Message:** `(request_id: int, policy_batch: torch.Tensor, value_batch: torch.Tensor)`
- **policy_batch shape:** `(N, 4840)`
- **value_batch shape:** `(N,)`
- **Purpose:** Return evaluated policies and values to the correct worker.
- **Rationale:** Per-worker response queues prevent **cross-worker message stealing**.

#### `replay_queue` (Workers → Trainer)
- **Message:** `(state_tensor: torch.Tensor, policy_tensor: torch.Tensor, value_tensor: torch.Tensor)`
- **Max size:** `maxsize=50000`
- **Purpose:** Stream completed game data to the training loop.
- **Backpressure:** When the trainer falls behind, workers block on `put()` instead of consuming unbounded memory.

### 5.2 SHM Mode (`ALPHATAFL_SHM=1`)

Phase 5A adds `multiprocessing.shared_memory` to bypass mp.Queue tensor serialization:
- **Tensors** are written/read through named SHM regions (`alphatafl_shm_{input,policy,value}_{worker_id}`) via raw `memcpy` (~32 µs for 64 states).
- **Control messages** on `inference_queue` carry `(worker_id, request_id, n_states)` only; results are returned on the per-worker `response_queue` as `(request_id, n_states)`.
- Yields ~12.5× faster transport vs. queue tensors. The Inference Server handles both modes transparently (`inference_server.py:_get_msg_tensor` autodetects the wire format).

### Batch Ordering
Results returned by the Inference Server are in the **same order** as the input `std::vector<GameState>` submitted by the worker.

---

## 6. C++ API Surface

### GameState Methods
- `to_tensor()` → `np.ndarray[float32, shape (14, 11, 11)]`
- `batch_to_tensor(states)` (static) → `np.ndarray[float32, shape (N, 14, 11, 11)]`
- `get_legal_moves_mask()` → `np.ndarray[float32, shape (4840,)]`
- `get_piece(r, c)` → `Piece` (lightweight accessor for GUI)

> Note: `GameState::is_restricted_square(r, c)` exists on the C++ class (used internally by `get_legal_moves()` and capture logic) but is **not currently bound to Python**; if a Python caller needs it, wrap the call in a C++ helper.

### MCTSNode Fields (C++ internal)
```cpp
MCTSNode* parent;
int visit_count;
double value_sum;
float prior;
bool is_expanded;
bool is_pending;
std::unordered_map<int, std::unique_ptr<MCTSNode>> children;
std::vector<float> child_priors;        // Phase 5E: stored sparsely, parallel to legal_action_indices
std::vector<int> legal_action_indices;  // sorted ascending for binary search

float get_child_prior(int action) const;  // Phase 5E binary-search lookup
```

### MCTS Interface
```cpp
using EvalFn = std::function<std::pair<std::vector<float>, float>(const GameState&)>;
using EvalFnBatched = std::function<
    std::pair<std::vector<float>, std::vector<float>>(
        const std::vector<GameState>&
    )>;

MCTS(EvalFn eval_fn, double c_puct = 1.4,
     double dirichlet_alpha = 0.3, double dirichlet_epsilon = 0.0);
MCTS(EvalFn eval_fn, EvalFnBatched eval_fn_batched, double c_puct = 1.4,
     double dirichlet_alpha = 0.3, double dirichlet_epsilon = 0.0);

std::vector<float> search(const GameState&, int num_simulations);
std::vector<float> search(const GameState&, int num_simulations, int batch_size);
```

If `eval_fn_batched` is null and `batch_size > 0` is passed, falls back to `eval_fn` per leaf.

### InferenceEngine (Phase 5F — ONNX Runtime C++)
```cpp
class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& model_path,
                             const std::string& device = "cuda");
    std::pair<std::vector<std::vector<float>>, std::vector<float>>
        evaluate(const std::vector<GameState>& states);
    std::pair<std::vector<float>, float>
        evaluate_single(const GameState& state);
};
```

Exposed to Python as `alphatafl_engine.InferenceEngine`. Loads an exported ONNX model (see `scripts/export_onnx.py`) and runs inference with the ORT CUDA Execution Provider (with CPU fallback). The C++ MCTS call site uses this class directly via an internal `eval_fn` adapter, so no Inference Server, mp.Queue, or SHM regions are involved on this path. See `docs/training/inference_server.md` and `docs/history/optimization_v1_phase5.md` §5F for the throughput and E2E numbers.

---

## 7. C++ Callback Legal-Move Handling (Phase 5D)

Since Phase 5D the legal-move mask, numerically stable softmax normalization, and Dirichlet noise are applied **inside C++** during `MCTSNode::expand`. The Python callback receives raw softmax probabilities; downstream code does not need to re-apply a legal mask. The pre-Phase-5D worker-side responsibilities (mask, normalize, Dirichlet noise) are retained as fallback when `MCTSNode::expand` is not on the hot path (e.g. GUI single-leaf mode); see `src/training/self_play.py` and `src/training/inference_server.py` for the actual call sites.

---

## 8. Feature Flags

| Env Variable | Default | Description |
|---|---|---|
| `ALPHATAFL_BATCHED` | `"1"` | Enable batched MCTS path |
| `ALPHATAFL_COMPILE` | `"0"` | Enable `torch.compile` on Inference Server |
| `ALPHATAFL_SHM` | `"0"` | Use OS shared memory for tensor transport (Phase 5A) |
| `ALPHATAFL_INFERENCE_TIMEOUT_MS` | `"1"` | Inference server partial-batch timeout (Phase 5A) |
| `ALPHATAFL_INFERENCE_MAX_BATCH` | `"256"` | Inference server max batch size |

Setting `ALPHATAFL_BATCHED=0` falls back to the legacy single-leaf path for A/B parity testing. The `ALPHATAFL_INFERENCE_*` flags are no-ops when the C++ `InferenceEngine` path is used (no server is started).

---

## 9. Inference Server Lifecycle

### Startup
1. `cli.py` creates `inference_queue`, per-worker `response_queues`, and `replay_queue`.
2. `cli.py` starts the Inference Server as a `multiprocessing.Process` with `daemon=False`.
3. The server loads `AlphaTaflNet` onto `cuda:0` in `eval()` + `inference_mode()`.
4. Only after the server is confirmed running does `cli.py` spawn worker processes.

### Hot-Reload (K5)
The Inference Server checks `os.path.getmtime("models/current_best.pt")` every 10 batches. Calls `model.load_state_dict(torch.load(...))` only when mtime changes. Workers never touch model files.

### Shutdown
- `cli.py` signals the `stop_event` or sends a `None` sentinel on `inference_queue`. The server exits its main loop and terminates cleanly.
- Workers blocked on `response_queues[worker_id].get()` receive an `EOFError`/`None` and exit.

### Windows Note
Windows uses `spawn` (not `fork`) for `multiprocessing`. All worker initialization happens inside `worker_task(params)` after spawn. Queues are passed via `params`. `if __name__ == "__main__":` guards are required around process creation.
