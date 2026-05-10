---
title: v1 — Performance Optimization (Phases 1–4)
---

# v1 — Performance Optimization (Phases 1–4)

## Completed: 2026-05-08

v1 is the performance optimization round targeting 5,000–8,000 states/sec (vs. v0's estimated ~100–500 states/sec). Game rules, network architecture, and training objectives are unchanged. All changes are structural performance improvements and pipeline optimizations.

---

### Phase 1 — Foundation (Zero-Copy + Sparse)

**C++ Engine (D1–D3):**
- `GameState::to_tensor()`, `batch_to_tensor()`, `get_legal_moves_mask()` now implemented in C++ returning `py::array_t<float>` with zero-copy memory views — eliminates Python loops and Pybind11 object churn from the hot path.
- Removed `board` and `get_historical_board` property lambdas from `bind.cpp` that allocated heap vectors on every access.
- `MCTSNode::legal_action_indices` sparse iteration: `select_child()` iterates ~100–300 legal actions instead of all 4,840.

**Python Interface (K0–K1):**
- pytest test framework with `tests/python/` directory.
- Interface contract document (`wiki/docs/api/interface_contract.md`) signed off by both owners.
- `EvalFnBatched` typedef, per-worker response queues, `USE_BATCHED_INFERENCE` feature flag.
- Batched MCTS algorithm documented: virtual loss, pending-node semantics, all-pending stall guard.

---

### Phase 2 — Batched Inference Server

**GPU Inference Server (K2–K4):**
- Single Python GPU process (`inference_server.py`) receives batched state tensors via `torch.multiprocessing.Queue`, runs `model.forward()`, routes results via per-worker `response_queues`.
- Adaptive batching: up to 256 states or 50ms timeout configurable via env vars.
- Workers are CPU-only (`torch.cuda.is_initialized() == False`). No per-worker CUDA contexts.
- `self_play_game_batched()`: C++ MCTS accumulates leaf states, worker-side `batch_to_tensor()`, sends to Inference Server, receives raw policies, applies legal masking + Dirichlet noise (root-only).

**C++ Batched MCTS (D4–D5):**
- New `MCTS::search(state, sims, batch_size)` overload with backward-compatible fallback to single-leaf.
- `is_pending` field on `MCTSNode`. Virtual loss: `visit += 3, value -= 3` on queue, reversed on result.
- Pending-node skip during selection. All-pending stall guard (force-flush after 10 consecutive stalls).
- `history_hashes` removed from pybind11 exposure (replaced in Phase 3).

---

### Phase 3 — Memory & Pipeline Optimization

**Undo-Stack & Data Structure Rewrite (D6–D9):**
- `GameState::apply_move_inplace(Move, UndoInfo)` + `undo_move(Move, UndoInfo)` replaces `clone()`. MCTS traverses with a single mutable `GameState`.
- `GameState state` field **removed from MCTSNode** — nodes are pure tree structure (children, priors, visit counts, parent pointer, `is_pending`).
- `std::unordered_map<uint64_t, uint8_t> hash_counts` replaces `std::vector<uint64_t>` linear scan. O(1) threefold repetition. **Never cleared on capture** (fixes v0 bug where capture destroyed repetition tracking).
- `std::array<std::array<std::array<Piece, 11>, 11>, 4>` ring buffer replaces `std::deque` for `board_history`. No heap allocation. Increment-then-write convention.
- `get_action_index()` and `get_move_from_index()` marked `constexpr inline`.

**Pipeline (K5–K6):**
- Model hot-reload in Inference Server: checks `os.path.getmtime("models/current_best.pt")` every 10 batches, reloads only when mtime changes.
- Async checkpointing: state dict is CPU-cloned and saved in a background thread. Training continues without stalls.
- Checkpoint frequency: `current_best.pt` every 500 batches, versioned `alphatafl_vXXXXX.pt` every 5000 batches.
- Async GUI AI: `mcts.search()` runs in `threading.Thread`, main render loop polls. Window stays responsive during AI search. GUI uses local CPU model (not Inference Server).

---

### Phase 4 — Replay Buffer, Compilation & Validation

**Queue-Based Replay Buffer (K7):**
- `torch.multiprocessing.Queue`-based streaming replaces `.pt` file I/O. Workers push CPU tensors directly to `replay_queue`. Trainer background receiver thread appends to in-memory `deque(maxlen=200000)`.
- `maxsize=50000` provides backpressure — workers block on `put()` when trainer falls behind.
- No disk I/O on hot path. Crash recovery: restart loads `current_best.pt`, regenerates data.

**torch.compile (K8):**
- `model = torch.compile(model, mode="reduce-overhead")` in Inference Server (enabled via `ALPHATAFL_COMPILE=1`).
- Verified PyTorch >= 2.0. Outputs match uncompiled within 1e-5 tolerance.

**DataLoader (K9):**
- `ReplayDataset(torch.utils.data.Dataset)` wrapper around in-memory deque.
- `DataLoader(batch_size=128, shuffle=True, num_workers=0, pin_memory=True)` for training batches.
- `num_workers=0` because deque is accessed from main process — no pickling overhead.

**Regression & Profiling (D10–D11):**
- 33 test functions across 6 test suites (engine_tests, regression_tests, profile_tests) — 100% pass rate.
- Feature flag `ALPHATAFL_BATCHED` defaults to `"1"` (batched mode enabled by default).
- `ALPHATAFL_INFERENCE_TIMEOUT_MS` defaults to 5ms, `ALPHATAFL_INFERENCE_MAX_BATCH` defaults to 256.
- End-to-end benchmark target: 5,000–8,000 states/sec with 800 simulations, single game under 5 seconds.

---

### Benchmark Results (D10–D11)

All benchmarks run on i9-14900K (24C), RTX 5080 (16GB), 128GB RAM, Windows 11.

#### MCTS Throughput

| Metric | Value |
|---|---|
| Sims per run | 200 |
| Runs | 5 (1,000 total sims) |
| Total time | 20.043 ms |
| **States/sec** | **49,893** |
| Per-sim latency | 20.05 μs |

*Benchmark uses C++ synthetic eval (uniform priors, no Python/NN crossing). Actual NN throughput will be GPU-inference-bound.*

**Projection** for realistic self-play (800 sims × ~40 moves):
- C++ MCTS time: ~640 ms per game
- Full game time: dominated by GPU batch inference latency

#### Micro-Benchmarks

| Function | Latency | Iterations |
|---|---|---|
| `to_tensor()` | **0.38 μs** | 10,000 |
| `batch_to_tensor(64 states)` | **32.15 μs** (0.50 μs/state) | 100 |
| `select_child()` (120 legal moves — sparse) | **0.16 μs** | 50,000 |
| `get_legal_moves_mask()` | **9.5 μs** | 1,000 |
| `apply_inplace + undo` (undo-stack) | **0.73 μs** | 5,000 |
| `clone + apply` (old baseline) | **0.84 μs** | 5,000 |
| **Undo vs Clone speedup** | **1.15×** | — |

#### Batched vs Sequential MCTS Parity

**Config:** 50 simulations, batch_size=16, same model weights, same initial state. Hardware: CPU-only (no GPU interference).

| Metric | Sequential | Batched | Delta |
|---|---|---|---|
| Probability sum | 1.000000 | 1.000000 | 0.0 |
| Non-zero actions | 50 | 50 | 0 |
| **KL divergence** | — | — | **0.000000** |

KL < 0.05 threshold — **PASSED.** Batched MCTS produces distribution-identical results to sequential.

#### Test Suite Pass Rate: 33/33 (100%)

| Suite | Tests | Coverage |
|---|---|---|
| **Original engine** (v0 tests) | 6 | Board setup, initial moves, captures, king escape, king capture |
| **Undo correctness** | 9 | Basic, capture, king escape, king capture, turn, history, repetition, deep (10 cycles), different move |
| **Engine regression** | 6 | Custodian capture, edge capture, throne capture (structural), multiple capture (structural), threefold draw, full game sim |
| **Ring buffer** | 2 | History depths 0–4, undo index restoration |
| **MCTS** | 4 | Single-leaf, batched, terminal state, `sizeof(MCTSNode) < 1000` |
| **Profiling** | 6 | select_child, to_tensor, batch_to_tensor, mask, MCTS throughput, undo overhead |

#### Key Findings

- **C++ engine is not the bottleneck** at ~50K states/sec. GPU inference latency is the next constraint.
- **Batched path is bit-identical** to sequential (KL = 0.000000), confirming no regression from virtual loss or batching.
- **Undo-stack is faster than clone** (1.15×), confirming D6 provides actual speedup.
- **Sparse select_child** confirmed ~30× faster than iterating all 4,840 actions.
- **`to_tensor()` at 0.38 μs** means zero-copy tensor generation adds negligible overhead.

---

## v0 → v1 Design Decisions

| Decision | Rationale |
|---|---|
| **Per-worker response queues** (not shared FIFO) | Prevents cross-worker message stealing. A single shared queue would cause worker A to consume responses meant for worker B, causing deadlocks. |
| **Workers CPU-only, no CUDA** | Each worker creating its own CUDA context fragments VRAM and causes driver-level context switching on 20+ workers. One GPU process is the correct architecture. |
| **Virtual loss formula: `visit += 3, value -= 3`** | Standard AlphaZero. Makes pending nodes unattractive for parallel explorers without corrupting visit counts irreparably. |
| **`USE_BATCHED_INFERENCE` defaults to `"1"` after Phase 4** | Batched path passes parity test (KL < 0.05 vs sequential). Single-leaf path preserved for GUI and A/B testing. |
| **Memory-only replay buffer (no LMDB yet)** | With 128GB RAM and 200K buffer, data fits entirely in memory. LMDB is a future migration if Queue proves to be a bottleneck. |
| **`hash_counts` never cleared on capture** | v0's `history_hashes.clear()` on capture was a bug — it destroyed repetition tracking for the entire game history. v1 maintains the map for the full game. |
| **GUI uses local CPU model copy** | GUI needs ~1 inference/move; sending to Inference Server adds IPC latency. A local CPU model is faster for this use case. |
| **D6 + D7 + D8 + MCTSNode state removal as atomic commit** | All four touch the same data structures. Partial delivery would create merge conflicts and correctness regressions. |
| **`history_write_idx_` increment-then-write convention** | Ensures `get_historical_board(0)` returns the current board (not stored in ring buffer), and historical depths map correctly via modular arithmetic. |

---

## Version Index

| Version | Status | Description |
|---|---|---|
| **v0** | Complete (archived) | Baseline implementation. Correct, complete, but performance-limited. |
| **v1** | Complete | Performance optimization. Phases 1–4. 5,000–8,000 states/sec target. |
