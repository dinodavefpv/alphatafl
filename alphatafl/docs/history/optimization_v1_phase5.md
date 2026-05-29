# Phase 5 Optimization

Details on the Phase 5A through Phase 5E optimization efforts for AlphaTafl v1.

---

## Phase 5A: Adaptive Timeout + Shared Memory IPC

### What Changed

- **`ALPHATAFL_INFERENCE_TIMEOUT_MS`** env var added (default **5ms**, was hard-coded 50ms)
- **`ALPHATAFL_INFERENCE_MAX_BATCH`** env var added (default 256)
- **Adaptive 1ms polling loop**: collects stragglers without dead time when queue is empty
- **Shared-memory `torch.Tensor` responses**: `share_memory_()` on CPU tensors before queue put

### Files Modified

- `src/training/inference_server.py`
- `self_play.py`
- `tests/python/test_inference.py`
- `tests/python/test_phase5.py` (new)
- `benchmarks/bench_phase5.py` (new)

### Test Results

**59 passed, 3 skipped** (2 torch.compile skipped for no MSVC, 1 e2e skipped for Windows cleanup hang). No regressions.

### Benchmark Results

| Config | Queue Latency | E2E Game Time |
|--------|--------------|---------------|
| 50ms timeout | 15.65 ms/batch | 9.56 s |
| 5ms timeout | 15.67 ms/batch | 9.56 s |
| **0ms timeout** | **6.63 ms/batch** | **3.44 s** |

### Key Insight

For single-worker self-play, **5ms vs 50ms makes no difference** because the inference server receives one batch, polls for stragglers, and the queue is empty — it waits the full timeout every time. **0ms (immediate flush) yields a 2.8x speedup** because the server processes the batch instantly without polling for more.

---

## Phase 5B: C++ libtorch Native Inference (Attempted)

### Goal

Move GPU inference into the C++ extension to eliminate Python `mp.Queue` round-trips entirely.

### What Was Built

1. **TorchScript export**: `scripts/export_torchscript.py` exports `AlphaTaflNet` to `models/alphatafl_scripted.pt`
2. **C++ InferenceEngine**: `src/inference/inference_engine.h` / `.cpp` — loads TorchScript model, runs batch inference, applies legal masks and softmax
3. **CMake integration**: Links PyTorch pip C++ libraries (`torch.lib`, `torch_cpu.lib`, etc.)
4. **Pybind11 binding**: `alphatafl_engine.InferenceEngine` class exposed to Python

### Blockers

| Blocker | Symptom | Root Cause |
|---------|---------|------------|
| **CUDA operators not registered** | `Could not run 'aten::empty.memory_format' with arguments from the 'CUDA' backend` | PyTorch pip packages use lazy/dynamic loading for CUDA kernels. They only register when Python's `torch` module initializes. From standalone C++, only CPU operators are available. |
| **C10 registry conflict** | `Key already registered with the same priority: C10` | Both Python `torch` and C++ `libtorch` try to register the C10 dispatcher in the same process. |

### Result

- **CPU inference**: 175 ms/batch (28x slower than mp.Queue)
- **CUDA inference**: Blocked
- **Accuracy**: 1.86e-09 max diff vs Python (numerically identical)

**Status**: Not viable on Windows with pip-installed PyTorch.

---

## Phase 5C: torch.compile (Attempted)

### What Happened

- Installed MSVC BuildTools, OpenMP headers, Windows SDK
- Installed `triton-windows` package
- `torch.compile` now compiles successfully in isolation

### Why It Didn't Help

- First-run Triton kernel compilation: **>5 minutes** (benchmark timeout)
- The model forward is already **<1 ms** on RTX 5080
- The queue overhead is **15 ms/batch** — 15x larger than the forward pass
- torch.compile optimizes the wrong thing

---

## Phase 5D: Step 1 MCTS C++ Callback Optimization

### What Changed

- **C++ Legal Masking & Softmax Normalization**: Moved legal move masking and numerically stable softmax normalization into the native C++ `MCTSNode::expand` routine.
- **C++ Dirichlet Noise Generation**: Root-node Dirichlet noise generation moved into native C++ using `std::gamma_distribution` and `std::mt19937`.
- **Zero-copy Array Buffer Access**: Adjusted Pybind11 wrappers in `bind.cpp` to extract direct pointers to raw flat 1D and 2D arrays, bypassing Python list allocation completely.
- **Robust sampler normalization**: Handled double-precision variance by re-normalizing `probs` in float64 before `np.random.choice`.
- **Process Cleanup Guard**: Wrapped benchmarks in a `try...finally` block to ensure background neural net server subprocesses are cleaned up cleanly even on failure.

### Files Modified

- `src/engine/mcts.h`
- `src/engine/mcts.cpp`
- `src/bindings/bind.cpp`
- `self_play.py`
- `benchmarks/time_single_game.py`

### Test Results

- All 60 Python tests pass (`pytest tests/python/`).
- All 9 profile tests and 21 regression tests pass (`build/Release/profile_tests.exe` & `regression_tests.exe`).

### Benchmark Results

- **Python overhead**: Reduced from **6.5 ms / batch** to **0.0 ms / batch**.
- **Turn Time**: Reduced from **346 ms / turn** to **216 - 244 ms / turn** (up to **37.5% speedup**).
- **Game Time**: Reduced from **63.1 s** to **43.1 - 48.8 s**.

---

## Phase 5E: Step 2 Sparse Priors Memory Optimization

### What Changed

- **Sparse prior mapping**: Altered `MCTSNode::child_priors` vector to store values parallel to `legal_action_indices`.
- **Binary search prior extraction**: Added `MCTSNode::get_child_prior` to lookup priors during new child node expansions using `std::lower_bound` on sorted indices.
- **Cache-friendly sequential selection**: Modified MCTS node selection to iterate sequentially over `legal_action_indices` and access `child_priors[i]` sequentially, resulting in cache-friendly operations.
- **Dirichlet noise mapping**: Enabled root Dirichlet noise generation to map directly to sequentially aligned elements.

### Files Modified

- `src/engine/mcts.h`
- `src/engine/mcts.cpp`
- `tests/cpp/test_profile.cpp`

### Test Results

- All 21 C++ regression tests and 9 profile tests pass.
- Node Memory Footprint: **1.078 KB / node** (down from 19.5 KB, a **18x reduction**).
- Aggregate 800-sim MCTS tree memory: **1.56 MB** (down from 15.6 MB).
- Synthetic C++ MCTS throughput: **61,072 states/sec** (up from 50,000 states/sec, a **22% throughput speedup**).

### E2E Benchmark Results

- **Turn Time**: Reduced from 216 - 244 ms/turn to **207 ms / turn** (40% speedup compared to 346 ms baseline).
- **Game Time**: Reduced from 43.1 - 48.8 s to **41.4 s**.

---

## Recommendations

| Priority | Action | Expected Speedup | Status |
|----------|--------|-----------------|--------|
| **Immediate** | Set `ALPHATAFL_INFERENCE_TIMEOUT_MS=1` + `ALPHATAFL_SHM=1` | **2-4x** | **Implemented** (5A) |
| **P1** | Move masking/noise into C++ eval callback | **3-5x** (GIL bypass) | **Completed** (5D/Step 1) |
| **P2** | Sparse child_priors storage (18x mem/node) | Memory only | **Completed** (5E/Step 2) |
| Medium-term | ONNX Runtime / TensorRT inference | **5-20x** | Planned (Step 3) |
| Platform | Re-test on Linux where `fork` + native Triton work | Potentially large | Planned |

### D12 Update: 1ms Timeout Is the Single Biggest Lever

Multi-worker throughput benchmark (50 sims/turn, batch_size=16, 30s window):

**Queue transport — 5ms vs 1ms**

| Workers | 5ms (st/s) | 1ms (st/s) | Speedup |
|---------|-----------|-----------|---------|
| 1 | 9 | 9 | 1.0x |
| 4 | 42 | 56 | 1.3x |
| 8 | 76 | 80 | 1.1x |
| 20 | 42 | 173 | **4.1x** |

**SHM transport — 5ms vs 1ms**

| Workers | 5ms (st/s) | 1ms (st/s) | Speedup |
|---------|-----------|-----------|---------|
| 1 | 9 | 9 | 1.0x |
| 4 | 26 | 107 | **4.1x** |
| 8 | 75 | 172 | 2.3x |
| 20 | 178 | 195 | 1.1x |

1ms timeout eliminates the batch-collection dead time that dominated 5ms. Queue mode alone gains 4.1x at 20 workers. SHM at 4 workers recovers from the 5ms penalty entirely (26→107, 4.1x). At 20 workers SHM, the gain is only 1.1x (178→195) — the GPU inference server hits a ~195 st/s ceiling.

**Recommendation: `ALPHATAFL_INFERENCE_TIMEOUT_MS=1` + `ALPHATAFL_SHM=1` as production defaults.**

---

*Last updated: 2026-05-28*
