---
title: v0 — Initial Implementation
---

# v0 — Initial Implementation (Baseline)

## Current Version: v0 (Initial Implementation)

AlphaTafl v0 is the baseline implementation — a fully functional AlphaZero-style self-play system for Hnefatafl. It demonstrates the core architecture: C++ game engine + PyTorch neural network + Python training pipeline. All correctness features work; performance is severely limited by the bottlenecks documented in [Performance Analysis](./performance.md).

### What Works (v0)

- **C++ Game Engine** — Complete move generation, capture logic (custodian, edge, throne-assisted, King), win conditions, threefold repetition, Pybind11 bindings.
- **Neural Network** — 14-channel ResNet (10 blocks × 128 filters), dual policy/value heads, training loop with Adam optimizer, replay buffer.
- **Self-Play** — Functional MCTS-guided game generation, Dirichlet noise, temperature-based action selection, training data collection.
- **GUI** — Pygame board visualization, two-player mode, AI-vs-human play with single-leaf MCTS inference.
- **CLI** — Training loop control, self-play orchestration, checkpoint management.

### Known Shortcomings (v0)

These are the performance bottlenecks identified by the five-model analysis and documented in [Performance Analysis](./performance.md):

| Bottleneck | Impact | Status |
|---|---|---|
| Batch Size 1 GPU Inference | GPU ~95% idle; ~5000 serial forward passes per game | **Critical** — Highest priority |
| Python↔C++ Boundary Per Leaf | Millions of crossing per minute; GIL contention | **Critical** |
| `state_to_tensor` Python Loops | ~1,000 Python bytecode instructions per evaluation | **Critical** |
| Workers Load Stale Model | Off-policy training data; model weights never hot-reloaded | **Major** |
| `.pt` File-Per-Game Replay Buffer | Disk I/O on every game; 1000-file batch load | **Major** |
| `select_child` O(4840) Iteration | 90%+ of loop is wasted computation on illegal actions | **Moderate** |
| `GameState.clone()` Per Node | Deep copies of board, history, hashes on every simulation | **Moderate** |
| O(N) Threefold Repetition Scan | Linear scan over history_hashes vector | **Minor** |
| No Transposition Table | Identical positions searched independently | **Minor** — Skipped |

### Why These Were Not Fixed in v0

The v0 implementation prioritized **correctness and architectural clarity** over performance. The bottlenecks are structural — batch size 1 requires redesigning the Python↔C++ callback contract; eliminating clone() requires an undo-stack that changes MCTSNode fundamentally. These are Phase 2–4 optimizations, not bug fixes.

---

## Next Version: v1 (Optimization Roadmap)

v1 is defined by the [Implementation Plan](../.agent/implementation_plan.md). It is a **performance-only upgrade** — game rules, network architecture, and training objectives are unchanged. v1 targets 5,000–8,000 states/sec (vs. v0's estimated ~100–500 states/sec).

### High-Level Summary of v1 Changes

#### Phase 1 — Foundation (Parallel, No Dependencies)

**Deepseek (C++ Engine):**

- **D1: Zero-Copy Tensor Generation** — `GameState::to_tensor()`, `batch_to_tensor()`, and `get_legal_moves_mask()` implemented in C++ returning `py::array_t<float>` with zero-copy memory views. Eliminates Python loops, string comparisons (`p.name == "ATTACKER"`), and Pybind11 object churn from the hot path.
- **D2: Remove bind.cpp Heap Churn** — Delete the `board` and `get_historical_board` property lambdas that allocate `std::vector<std::vector<Piece>>` on every read. The hot path uses D1's `to_tensor()` exclusively.
- **D3: Sparse select_child** — `MCTSNode` stores only legal action indices (`std::vector<int>`). `select_child()` iterates ~100–300 legal actions instead of all 4,840. ~15–50× speedup per selection step.

**Kimi (Python Architecture):**

- **K1: Interface Contract + Feature Flag** — Define the contract between C++ MCTS and Python Inference Server: `EvalFnBatched` typedef, batched MCTS algorithm (virtual loss, pending-node queue, all-pending stall guard), per-worker response queues, `USE_BATCHED_INFERENCE` feature flag (default `"0"` until parity test passes).
- **K0: Test Framework** — pytest + GoogleTest, CI commands `pytest tests/python/` and `ctest --test-dir build`.

Phase 1 Milestone: `state.to_tensor()` produces `(14, 11, 11)` tensor in under 1ms. `batch_to_tensor()` crosses Pybind11 once per batch. Sparse `select_child` produces identical results to full iteration within `1e-4`.

---

#### Phase 2 — Batched Inference Server (Depends on Phase 1)

**Kimi:**

- **K2: GPU Inference Server** — Single Python process owns `AlphaTaflNet` on `cuda:0`. Workers send batched state tensors via `torch.multiprocessing.Queue`. Server dequeus up to 256 states (or 50ms timeout), runs one forward pass, returns policy/value via per-worker response queues. Eliminates per-worker CUDA contexts.
- **K3: Remove Per-Worker CUDA Contexts** — Workers are CPU-only Python processes. `torch.cuda.is_initialized() == False` in all workers.
- **K4: Batched eval_fn** — C++ MCTS accumulates leaf states for batch evaluation. Worker-side applies legal move masks and Dirichlet noise after receiving raw policies.

**Deepseek:**

- **D4: Batched MCTS + Backward Compatibility** — New `MCTS::search(state, sims, batch_size)` overload. Original single-leaf `search(state, sims)` retained for GUI. `is_pending` field on `MCTSNode`.
- **D5: Virtual Loss** — When leaf queued: `visit_count += 3, value_sum -= 3, is_pending = true`. On batch result: reverse virtual loss, expand node. Skip `is_pending == true` nodes during selection. All-pending stall guard (force-flush after 10 consecutive stalls).

Phase 2 Milestone: Single self-play game runs end-to-end with batched inference. `nvidia-smi` shows exactly one Python process on GPU. GUI works with single-leaf fallback. 64 states processed in under 10ms. KL divergence < 0.05 vs. sequential MCTS.

---

#### Phase 3 — Memory and Pipeline Optimization (Parallel after Phase 2)

**Deepseek:**

- **D6: Undo-Stack + Remove MCTSNode::state** — `GameState::apply_move_inplace()` and `undo_move()` replace `clone()`. Single mutable `GameState` applies on descent, undoes on backtrack. **Atomically** with D7/D8: remove `GameState state` field from `MCTSNode` — nodes become pure tree structure.
- **D7: O(1) Threefold Repetition** — Replace `std::vector<uint64_t>` linear scan with `std::unordered_map<uint64_t, uint8_t> hash_counts`. Increment on apply, decrement on undo. Never cleared on capture.
- **D8: Ring Buffer for board_history** — Replace `std::deque<std::array<...>>` with `std::array<std::array<std::array<Piece, 11>, 11>, 4>` + index. No heap allocation.
- **D9: constexpr inline helpers** — `get_action_index()` and `get_move_from_index()` marked `constexpr inline`. Optional 12KB lookup table if profiling shows it matters.

**Kimi:**

- **K5: Model Hot-Reload** — Inference Server checks `current_best.pt` mtime between batches. Workers never touch model files. Async checkpointing with CPU-cloned state dicts.
- **K6: Async GUI AI** — `mcts.search()` moved to `threading.Thread` in GUI. Main render loop polls for completion. GUI uses local CPU model copy (not Inference Server) for ~1 inference/move, avoiding IPC latency.

Phase 3 Milestone: `clone()` calls eliminated from MCTS hot path. `MCTSNode::state` removed. `hash_counts` never cleared. GUI remains responsive during AI search. Hot-reload picks up new weights within one batch cycle.

---

#### Phase 4 — Replay Buffer, Compilation, and Validation

**Kimi:**

- **K7: Queue-Based Replay Buffer** — Replace `.pt` file I/O with `torch.multiprocessing.Queue`-based streaming. Workers push CPU tensors directly to `replay_queue`. Trainer receiver thread appends to in-memory deque. `maxsize=50000` for backpressure. No disk I/O on hot path.
- **K8: torch.compile()** — `torch.compile(mode="reduce-overhead")` on Inference Server model. Verify PyTorch >= 2.0. One-line change once batching is stable.
- **K9: DataLoader for Training** — `ReplayDataset` wrapper + `torch.utils.data.DataLoader` with `pin_memory=True`, `num_workers=0`.

**Deepseek:**

- **D10: Full Regression Suite** — All D1–D9 tests plus game completion, capture rules, threefold repetition.
- **D11: Profiling** — Profile `select_child()`, `to_tensor()`, `batch_to_tensor()`, `check_captures()`, `compute_hash()`. Target 5,000–8,000 states/sec, 10,000 as stretch.

Phase 4 Milestone: 5,000–8,000 states/sec. Single game with 800 simulations under 5 seconds. No regression in game quality. `USE_BATCHED_INFERENCE=0` path parity within KL < 0.05.

---

### v1 Design Decisions

| Decision | Rationale |
|---|---|
| **Per-worker response queues** (not shared FIFO) | Prevents cross-worker message stealing. A single shared queue would cause worker A to consume responses meant for worker B, causing deadlocks. |
| **Workers CPU-only, no CUDA** | Each worker creating its own CUDA context fragments VRAM and causes driver-level context switching on 20+ workers. One GPU process is the correct architecture. |
| **Virtual loss formula: `visit += 3, value -= 3`** | Standard AlphaZero. Makes pending nodes unattractive for parallel explorers without corrupting visit counts irreparably. |
| **`USE_BATCHED_INFERENCE` defaults to `"0"`** | Old single-leaf path must remain functional for A/B parity testing and GUI fallback. Switch to `"1"` only after `test_batched_self_play_parity` confirms KL < 0.05. |
| **Memory-only replay buffer (no LMDB yet)** | With 128GB RAM and 200K buffer, data fits entirely in memory. LMDB is a future migration if Queue proves to be a bottleneck. |
| **`hash_counts` never cleared on capture** | v0's `history_hashes.clear()` on capture was a bug — it destroyed repetition tracking for the entire game history. v1 maintains the map for the full game and clears it only on game reset. |
| **GUI uses local CPU model copy** | GUI needs ~1 inference/move; sending to Inference Server adds IPC latency. A local CPU model is faster for this use case and avoids coupling the GUI to the training pipeline. |
| **D6 + D7 + D8 + MCTSNode state removal as atomic commit** | All four touch the same data structures. Partial delivery would create merge conflicts and correctness regressions. Bundle-commit ensures consistent state. |
| **`history_write_idx_` increment-then-write convention** | Ensures `get_historical_board(0)` returns the current board (not stored in ring buffer), and historical depths map correctly via modular arithmetic. |

---

## Version Index

| Version | Status | Description |
|---|---|---|
| **v0** | Complete | Baseline implementation. Correct, complete, but performance-limited by structural bottlenecks. |
| **v1** | Planned | Optimization roadmap per [Implementation Plan](../.agent/implementation_plan.md). Phases 1–4. |

---

## Changelog

### v0 → v1 (Planned)

See the [Implementation Plan](../.agent/implementation_plan.md) for the full phase-by-phase specification. Summary of changes:

- **D1–D3**: C++ zero-copy tensor generation, bind.cpp heap churn elimination, sparse select_child iteration.
- **K1**: Interface contract with `EvalFnBatched`, virtual loss algorithm, per-worker response queues, `USE_BATCHED_INFERENCE` feature flag.
- **K2–K4**: GPU Inference Server with batched evaluation. Workers CPU-only.
- **D4–D5**: Batched MCTS with virtual loss, backward-compatible `search()` overload.
- **D6–D8**: Undo-stack `apply_move_inplace()`/`undo_move()`, O(1) repetition map, ring buffer `board_history`, removal of `MCTSNode::state`.
- **K5–K6**: Model hot-reload via mtime watching, async GUI AI.
- **K7–K9**: Queue-based replay buffer, `torch.compile()`, DataLoader for training.
- **D10–D11**: Full regression suite, profiling, benchmark.

Phase 5 candidate (pipelined GPU evaluation) documented but out of scope for v1.
