# AlphaTafl Wiki Log

## [2026-11-13] Fix | Tests no longer delete real versioned checkpoints

- **Bug:** `tests/python/test_phase3.py::test_checkpoint_frequency` cleaned up with `glob("models/alphatafl_v*.pt")` + `os.remove`, deleting the user's real versioned checkpoints. It also wrote `alphatafl_v00001/2.pt` into `models/`, because `Trainer` hardcoded `models/` for versioned checkpoints regardless of `model_path`.
- **Fix:** `Trainer.checkpoint_dir` = directory of `model_path`; versioned writes and the resume scan use it (`src/training/trainer.py:36`, `:65`, `:228`). `test_async_checkpoint_cpu_clone` and `test_checkpoint_frequency` now use pytest `tmp_path`; the latter asserts `models/alphatafl_v*.pt` is unchanged. The async-save timing bound was loosened from 50 ms to 500 ms and the fixed sleep replaced by polling (it failed under full-suite load).
- **Related:** `Trainer.__init__` now writes the fresh model synchronously (it was async, so the file could be missing right after construction).
- **Tests:** two new `TestVersionedCheckpointLocation` tests in `tests/python/test_checkpoint.py`. Full Python suite: 74 passed, 2 skipped, 1 flaky (`test_hot_reload_no_reload`, server-startup timing; passes on rerun). No file in `models/` changed during the run.
- **Docs:** `docs/training/training_loop.md`.

## [2026-11-12] Fix | Checkpoint handling (atomic writes, safe init, compiled hot-reload)

- **Hot-reload under `ALPHATAFL_COMPILE=1`**: loading a plain state dict into the `torch.compile` wrapper fails (`Missing key(s)... "_orig_mod.*"`, verified), so the server silently kept stale weights. `inference_server_main` now keeps `base_model` and loads into it; the wrapper shares its parameters.
- **Non-atomic saves**: added `src/training/checkpoint.py` with `atomic_save()` (temp file + `os.replace`, retried on Windows `PermissionError`). Trainer `save_model()` and versioned checkpoints use it. Async saves now carry a sequence number and are serialized so an older snapshot cannot overwrite a newer one.
- **Trainer overwrote `current_best.pt` on any load error**: it now moves an unloadable file to `current_best.unloadable_<timestamp>.pt` via `backup_unloadable()` before writing a fresh model; `OSError` is retried and raised instead.
- **Tests**: `tests/python/test_checkpoint.py` (10 tests). Full Python suite: 72 passed, 2 skipped, 1 failed (`test_async_checkpoint_cpu_clone` under full-suite load; passes alone and in 3 consecutive runs of `test_phase3.py`).
- **Docs**: `docs/training/training_loop.md`, `docs/training/inference_server.md`, `docs/architecture/README.md`. The architecture note also no longer claims MCTS calls the C++ `InferenceEngine` directly — it is bound but not wired into self-play.

## [2026-11-12] Fix | History ring buffer restored on undo

- **Bug:** `GameState::apply_move_inplace` overwrote a slot of the 4-slot `board_history_` ring buffer, and `undo_move` only rewound `history_write_idx_` without restoring the old contents. Any MCTS traversal two or more plies deep therefore clobbered the root's T-1..T-3 boards, so later leaves were evaluated with wrong history planes (channels 3–11) — a train/inference mismatch, since training tensors come from the real, never-undone game state.
- **Fix:** `UndoInfo` gains `overwritten_history`; `apply_move_inplace` saves the slot before writing (`src/engine/game_state.cpp:132`) and `undo_move` restores it (`src/engine/game_state.cpp:198`).
- **Test:** added `UndoTest.UndoMultiPlyRestoresHistory` in `tests/cpp/test_regression.cpp` (6-ply apply/undo, checks board, ring buffer, write index, and `to_tensor()`). Verified it fails without the restore and passes with it. C++: 22/22 regression + 6/6 engine tests pass. Python: 62 passed, 2 skipped, 1 flaky inference-server timing test (a different one per run; each passes in isolation).
- **Docs:** `docs/engine/game_state.md` — documented `overwritten_history` and the `post_hash` / `hash_counted` undo fields, updated the `apply_move_inplace` and `undo_move` steps.

## [2026-06-01] Lint | Documentation accuracy pass (round 2)

Cross-referenced every page in `alphatafl/docs/` against the live code (`src/`, `cli.py`, `gui.py`, `self_play.py`, `tests/`, `CMakeLists.txt`). The previous round mostly caught stale env-var defaults and missing APIs; this round caught a few subtle doc/code mismatches that had been left in place.

Fixed the following (current-state pages only — historical pages were not rewritten):

- **`docs/api/bindings.md`**: removed the `is_restricted_square(r, c)` line from the Python `GameState` API list — the C++ method exists (`src/engine/game_state.h:82`) but is not bound in `src/bindings/bind.cpp:39-77`. The doc is the binding contract, not the C++ surface. Also reworded the `to_tensor()` description: the binding copies the C++ `std::vector<float>` into a capsule-managed buffer and exposes a numpy view of that buffer, so it is "single C++→Python copy with capsule-managed lifetime", not "zero-copy".
- **`docs/api/interface_contract.md`**: the C++ API surface section listed `is_restricted_square(r, c) → bool (Pybind11-bound)` — replaced with an explicit note that the C++ method exists but is not bound, so callers must wrap the call in a C++ helper if they need it from Python.
- **`docs/architecture/README.md`**: removed the two "zero-copy tensor generation / zero-copy tensor views" claims in the High-Level Overview (same reason as `to_tensor()` above). Reworded data-flow step 3 ("Worker-Side Pre-Processing") so it no longer reads as if the worker applies the legal mask or Dirichlet noise; the worker just obtains the mask and forwards the raw softmaxed policy, and the C++ `MCTSNode::expand` does the masking and Dirichlet mixing (Phase 5D). Also added the explicit `dirichlet_alpha=0.3, dirichlet_epsilon=0.25` from `self_play_game_batched` so the data flow matches the actual call site.
- **`docs/engine/game_state.md`**: `check_threefold()` was documented as "Never called on capture". The code (`src/engine/game_state.cpp:113-119`, called from `apply_move_inplace` line 158) shows that captures alone do not end the game, so the threefold check still runs after a capture when no king capture or stalemate has occurred. Reworded to describe the actual condition: called only when `winner == Player::NONE`.
- **`docs/engine/mcts.md`**: the `src/training/mcts.py` source link was described as a "Python fallback implementation, deprecated". It is not a fallback — the C++ `MCTS` in single-leaf mode (no `eval_fn_batched`) is the fallback. The pure-Python file is an abandoned prototype not imported by any module. Replaced with a note to that effect. Also fixed the `dirichlet_epsilon` default description, which incorrectly said "the GUI and CLI typically leave it at 0 and apply noise from Python": the GUI uses the default and adds no noise, the CLI batched self-play passes `0.25` to enable C++-side mixing, and only the legacy single-leaf self-play leaves the ctor default and mixes noise from inside its Python `eval_fn` callback.
- **`docs/training/self_play.md`**: the Batched Path step 4 said "Worker applies legal move masks and Dirichlet noise (root-only), returns results to C++ MCTS". This contradicts `docs/api/interface_contract.md` §7. The worker only forwards the raw softmaxed `(policy, value)`; the C++ `MCTSNode::expand` does the masking, softmax renormalization, and Dirichlet mixing. Step 3 was also tightened to make the server's role (apply softmax to the policy logits) explicit.
- **`docs/training/training_loop.md`**: the DataLoader description said `pin_memory=True` unconditionally. The code (`src/training/trainer.py:71`) uses `pin_memory=(self.device == "cuda")` so the doc now says the same.

No code was changed in this pass — documentation only. The only remaining `is_restricted_square` mention in the current-state wiki is the explicit "not currently bound" note in `docs/api/interface_contract.md`, which is intentional. The only remaining "zero-copy" / "pin_memory=True" / "deprecated" mentions live in `docs/history/`, where they are accurate historical records.

## [2026-06-01] Orientation | Added project-root `AGENTS.md`

- The schema document (`alphatafl/WIKI.md`) was buried inside the wiki folder, so a fresh LLM agent visiting the project had no obvious entry point. Added `AGENTS.md` at the project root as the canonical orientation file.
- `AGENTS.md` is a thin pointer: it names the canonical wiki location (`alphatafl/`), lists the entry-point files (`index.md`, `log.md`, `docs/`, `WIKI.md`), and states the three hard rules (no root-level `wiki/`, current-state pages reflect only the current code, every change updates `index.md` and `log.md`).
- Full schema, conventions, and workflows still live in `alphatafl/WIKI.md`; `AGENTS.md` is intentionally a short orientation, not a duplicate.
- Logged here per the rule that every wiki-affecting change updates `log.md`.

## [2026-06-01] Merge | Consolidated stale `wiki/` split into the canonical Obsidian wiki at `alphatafl/`

- Investigation: a root-level `wiki/` directory had been created in parallel with the proper Obsidian wiki at `alphatafl/`. The two had diverged: `alphatafl/docs/history/optimization_v1_phase5.md` and `alphatafl/docs/history/optimization_v1_benchmarks.md` (Sections 11–13) already contain the same content as `wiki/phase5_optimization.md` and `wiki/benchmarks.md`, and the `alphatafl/` index/log include earlier entries (back to 2026-05-08) missing from `wiki/`.
- Resolution: the `wiki/` folder was a stale duplicate with no unique content beyond what `alphatafl/` already had. Deleted it.
- Schema fix: updated `alphatafl/WIKI.md` to state the wiki lives at `alphatafl/` (not `wiki/`), since the old note was the root cause of the split.
- Followed by a lint pass against the live codebase to repair stale references and outdated API descriptions. See entries below.

## [2026-06-01] Lint | Documentation accuracy pass

Audited every page in `alphatafl/docs/` against the current code. Fixed the following:

- **`docs/api/interface_contract.md`**: documented the `InferenceEngine` (Phase 5F) C++ ONNX inference class, added the `dirichlet_alpha` and `dirichlet_epsilon` constructor parameters that the C++ `MCTS` class has had since v1, removed the "raw softmax probabilities, worker-side legal masking" requirement that Phase 5D moved into C++, updated `ALPHATAFL_INFERENCE_TIMEOUT_MS` default from `"5"` to `"1"`, and added the `ALPHATAFL_SHM` flag.
- **`docs/api/bindings.md`**: added the `InferenceEngine` class, added the `dirichlet_alpha` and `dirichlet_epsilon` constructor parameters for `MCTS`.
- **`docs/engine/mcts.md`**: added the `dirichlet_alpha` and `dirichlet_epsilon` parameters to the constructor, added the `MCTSNode::get_child_prior` method (Phase 5E).
- **`docs/architecture/README.md`**: data flow now says `1ms timeout` (matches the 1ms default swept in Phase 5A); the existing ONNX-C++ note is retained.
- **`docs/training/inference_server.md`**: explicitly notes that the standalone C++ ONNX path (`InferenceEngine`) is the alternative post-Phase-5F and that the Python Inference Server and the C++ `InferenceEngine` are mutually exclusive inference backends.
- **`docs/history/optimization_v1.md`**: fixed test counts. The Phase 1–4 section's "50 passing tests" figure and the historical "60 Python tests" line in `optimization_v1_phase5.md` are now followed by clarifying notes; the current count is 62 Python test methods (27 + 14 + 7 + 8 + 6) across 5 files (typically 59 pass, 3 skip on Windows). The C++ test count was bumped from 33 to 36 (6 + 21 + 9) across 3 files (the 9 figure reflects the `NodeMemoryFootprint` and `MctsMemoryGrowth` profile tests added in D12).
- **`docs/history/optimization_v1_benchmarks.md`**: Section 7's `test_interface.py` row updated from 21 to 27 and the total from 56 to 62 (59 pass + 3 skip); profile_tests row updated from 6 to 9 to match the D12 additions.
- **`docs/history/optimization_v1.md`**, **`docs/history/initial_v0.md`**: replaced `wiki/docs/...` path references with `alphatafl/docs/...` to match the actual layout.

## [2026-05-31] Inference | Step 3 ONNX Runtime Standalone C++ Inference
- Transitioned model inference backend of `InferenceEngine` from PyTorch C++ (LibTorch) to standalone ONNX Runtime on CUDA.
- Patched PyTorch model export script (`scripts/export_onnx.py`) to flatten policy logits using `p.flatten(1)` instead of dynamic reshape `p.reshape(p.size(0), -1)` to prevent division-by-batch-size bugs in ONNX Runtime.
- Removed LibTorch detection and dependencies from CMakeLists.txt and added automatic DLL copy steps post-build.
- Implemented C++ ONNX Runtime API session, tensor wrapping, execution on CUDA, and CPU post-processing with dynamic device fallback.
- Validated numerical parity against Python model, yielding near-zero difference (policy diff **1.21e-08**, value diff **4.40e-07**).
- Achieved **81,439 states/second** synthetic MCTS search throughput (62% increase vs baseline) and **245.5 ms / turn** synchronously on GPU in a single process, completely eliminating multiprocessing and SHM.
- Updated `alphatafl/docs/history/optimization_v1_phase5.md` and `alphatafl/docs/history/optimization_v1_benchmarks.md`. Pre-existing PyTorch scripts are also updated to support ONNX model evaluation.
- Documented Phase 5F ONNX Runtime standalone C++ implementation details and updated recommendations status table in `alphatafl/docs/history/optimization_v1_phase5.md`.
- Appended Section 13 detailing E2E game time, batch latency (4.91 ms), and synthetic MCTS throughput (81.4K states/sec) to `alphatafl/docs/history/optimization_v1_benchmarks.md`.

## [2026-05-28] Memory | Step 2 Sparse Priors Optimization
- Stored prior probabilities sparsely parallel to `legal_action_indices`. Added binary search lookup helper `MCTSNode::get_child_prior()` for node creation.
- Updated MCTS child selection to iterate sequentially and access parallel `child_priors[i]` sequentially, resulting in cache-friendly operations.
- Re-routed root Dirichlet noise generation to map directly to sequential indices.
- Node memory footprint dropped from 19.5 KB to **1.078 KB / node** (a **18x reduction**). Aggregate 800-sim tree memory dropped to **1.56 MB** (down from 15.6 MB).
- C++ MCTS synthetic throughput increased by **22%** (to 61,072 states/second) due to sequential cache friendliness.
- E2E Turn Time reduced to **207 ms / turn** (41.4s total game time), yielding a **40% speedup** vs baseline.
- Created `alphatafl/docs/history/optimization_v1_phase5.md` to document the entire Phase 5 optimization progress. Updated `alphatafl/docs/history/optimization_v1_benchmarks.md` with Section 12.
- Updated `alphatafl/log.md` and `alphatafl/index.md`.
- Documented Phase 5E Sparse Priors implementation details and marked P2 as Completed in the recommendations status table in `alphatafl/docs/history/optimization_v1_phase5.md`.
- Appended Section 12 detailing the sparse memory metrics (1.078 KB / node) and the unexpected 22% C++ MCTS throughput speedup (61,072 states/sec) to `alphatafl/docs/history/optimization_v1_benchmarks.md`.

## [2026-05-28] Add | Step 1 C++ MCTS Callback Optimization
- Reimplemented legal move masking, softmax normalization, and Dirichlet noise in C++ using zero-copy memory views.
- Modified Pybind11 wrapper and `self_play.py` to extract raw flat policy/value arrays directly from NumPy buffers, bypassing Python list marshalling.
- Eliminated Python callback list build overhead (**6.5 ms -> 0.0 ms**), reducing E2E Turn Time to **216 - 244 ms / turn** (up to 37.5% speedup vs 346 ms baseline).
- Resolved NumPy `np.random.choice` probability sum ValueError by re-normalizing MCTS search outputs in float64.
- Wrapped benchmarks in a `try...finally` block to guarantee subprocess/SHM cleanup on error or cancellation.
- Updated `alphatafl/docs/history/optimization_v1_benchmarks.md`: Added Section 11 (Phase 5D Step 1 results), resolved python masking loop unknowns.
- Updated `alphatafl/log.md` and `alphatafl/index.md`.
- Documented Phase 5D C++ Callback implementation details and updated recommendations status table in `alphatafl/docs/history/optimization_v1_phase5.md`.
- Appended Section 11 detailing 800-sim E2E timing results and resolved Python masking overhead unknowns to `alphatafl/docs/history/optimization_v1_benchmarks.md`.

## [2026-05-09] Add | Python Callback Micro-Profile
- Instrumented `eval_fn_batched` in `benchmarks/time_single_game.py` with per-step `time.perf_counter()` timers. Collected 1755 batch samples across one 200-turn 800-sim game.
- Measured per-batch breakdown: mp.Queue.get (await server) = 13.9ms (58%), Python list `.tolist()` build = 6.5ms (27%), C++ `batch_to_tensor` + pybind11 wrap = 3.0ms (12%), SHM memcpy = 0.5ms (2%), mp.Queue.put = negligible.
- Confirmed 84% of per-batch Python overhead (23.4ms) is eliminable by moving NN inference into C++ (removes mp.Queue pipes, list construction, pybind11 tensor wrap).
- Updated `wiki/docs/history/optimization_v1_benchmarks.md`: Added Section 8.8 (Callback Micro-Profile) and 8.9 (Revised Bottleneck Summary with measured data). Replaced estimated bottleneck numbers with instrumented data throughout 8.9-8.11 and conclusion.
- Answered open question #7. Added #8 (pybind11 game loop overhead profiling).
- Updated `wiki/log.md` and `wiki/index.md`.

## [2026-05-09] Benchmark | Compile all benchmarks into wiki
- Created initial wiki structure and populated with complete benchmark compilation in `alphatafl/docs/history/optimization_v1_benchmarks.md` and `alphatafl/docs/history/optimization_v1_phase5.md`.
- Key findings documented:
  - Phase 5A: 0ms timeout gives 2.8x speedup (9.56s -> 3.44s/game).
  - Phase 5B: C++ libtorch blocked on Windows (CUDA operators not registered).
  - Phase 5C: torch.compile works but compilation time >5 minutes, not viable.

## [2026-05-09] Correct | Bottleneck Analysis After Deep-Tree Profile & Single-Game Timing
- Ran C++ deep-tree profile (`ProfileTest.DeepTreeTraversal`): per-sim MCTS cost is flat at 22-24 µs regardless of 100/400/800 sims. Tree depth has no measurable effect — sparse `select_child` is O(legal moves), not O(tree depth). C++ is definitively NOT the bottleneck.
- Timed single 800-sim game end-to-end: 63.1s for 182 turns (346ms/turn). Turn time is stable at 320-380ms after first turn.
- Corrected bottleneck analysis: the per-batch Python callback overhead (~42ms: mp.Queue pipe round-trips + Python list construction) dominates at 65% of turn time. C++ MCTS is only 5% (18ms). The ~90µs/sim estimate and "C++ dominates" conclusion from earlier wiki were wrong.
- Replaced Sections 8.6.2–8.8 in `wiki/docs/history/optimization_v1_benchmarks.md` with corrected data, deep-tree profile results, single-game timing, revised bottleneck summary, and updated cumulative improvement table.
- Answered open question #6 (C++ tree traversal is not the issue). Added #7 (Python callback elimination). Reprioritized libtorch/ONNX Runtime as highest-impact next step.
- Updated `wiki/log.md` and `wiki/index.md`.

## [2026-05-09] Add | Batch Size Sweep & Bottleneck Analysis
- Ran batch size sweep at 50/256/800 sims across batch sizes 8–512 (8 workers, SHM + 1ms, max_batch=2048). Confirmed redundancy: batch_size ≥ sims produces identical throughput — plateau at all three sim counts.
- Key finding: IPC-only projection model disproven. At 800 sims, C++ MCTS tree traversal (72ms/turn) dominates over IPC (42ms/turn). Best 800-sim throughput: 7 st/s (25s/game) — 5× off target. Bottleneck permanently shifted from Python mp.Queue to C++ MCTS.
- Updated `wiki/docs/history/optimization_v1_benchmarks.md`: Replaced Section 8.5 projection with actual sweep data (new 8.6). Added per-sim-count analysis, bottleneck breakdown table, updated cumulative improvement, and revised conclusion. Answered open question #5; added #6 (C++ MCTS optimization).
- Updated `wiki/index.md` with latest benchmark milestone.

## [2026-05-09] Add | Shared Memory Transport & Timeout Optimization
- Implemented `multiprocessing.shared_memory` transport in `inference_server.py`, `self_play.py`, and `orchestrator.py` (feature flag `ALPHATAFL_SHM=1`). Eliminates mp.Queue tensor serialization — workers write tensor data to OS shared memory via raw memcpy; inference server reads directly; only lightweight `(worker_id, request_id, n_states)` metadata goes through queues.
- SHM vs Queue comparison at 20 workers: 275 st/s vs 148 st/s (1.9× speedup). SHM scales near-perfectly (99.6% efficiency) while Queue degrades at high concurrency.
- Timeout sweep 0–5ms identified **1ms as the optimal batching window** for both transports. Default `ALPHATAFL_INFERENCE_TIMEOUT_MS` changed 5→1ms. SHM + 1ms = 275 st/s peak.
- Batch_size=64 projects to 2.6s/game at 800 sims within the 3–5s target.
- Updated `wiki/docs/history/optimization_v1_benchmarks.md`: Section 2 (SHM latency), Section 5 (ceiling analysis with SHM), Section 6 (new flags), Section 8 (full multi-worker + SHM + timeout sweep data, projections), open questions #1 & #4 answered, #5 added.
- Updated `wiki/index.md` with latest benchmark milestone.

## [2026-05-09] Add | Multi-Worker Throughput Benchmarks
- Added `benchmarks/bench_multi_worker.py` and `benchmarks/_bench_multi_worker_helper.py`: multi-worker self-play throughput benchmark measuring states/sec across 1–20 concurrent workers at different timeout settings (0ms, 5ms).
- Updated `wiki/docs/history/optimization_v1_benchmarks.md` with Section 8 multi-worker results: 0ms timeout causes server starvation (4 st/s at 20 workers vs 133 st/s at 5ms), 5ms gives super-linear scaling up to 12 workers, GPU saturation at 20 workers.
- Answered open question #1: 0ms timeout is confirmed catastrophic for multi-worker setups.
- Added new open question #4: can shared memory IPC eliminate the queue bottleneck?
- Updated benchmarks page status, raw data files, and changelog.

## [2026-05-08] Initialize | Wiki setup
- Initialized wiki structure and schema.
- Created `index.md` and `log.md`.
- Planned initial set of documentation pages.

## [2026-05-08] Ingest | Core System Documentation
- Documented High-level Architecture.
- Documented Hnefatafl Game Rules.
- Documented C++ Engine (`GameState`, `MCTS`).
- Documented Neural Network (ResNet).
- Documented Python Bindings (Pybind11).
- Documented Training Workflow (Self-play, Training loop).
- Documented User Interfaces (GUI, CLI).

## [2026-05-08] Add | Version History & Changelog Page
- Created `initial_v0.md` documenting v0 baseline, known bottlenecks, and v1 roadmap.
- Updated `wiki/index.md` to include the new page.

## [2026-05-08] Fix | Wiki Accuracy Audit
- Fixed `resnet.md`: Corrected policy head, value head, input channels.
- Fixed `training_loop.md`: Adam optimizer only, versioned checkpoint, batch timing, buffer capacity.
- Fixed `cli.md`: Added missing `loop`, `auto_loop`, `stop_auto` commands.
- Fixed `game_state.md`: Added board_history, history_hashes, compute_hash, get_historical_board.
- Fixed `bindings.md`: Added history_hashes and get_historical_board(depth) to exposed API.
- Fixed `architecture/README.md`: Clarified MCTS runs in Python worker processes, split engine links.
- Fixed `mcts.md`: Added reference to Python MCTS implementation.

## [2026-05-08] Phase 1 | Interface Contract & Test Framework
- Created `tests/python/` with pytest framework.
- Created `interface_contract.md` documenting data types, batch bounds, virtual loss, batched MCTS algorithm, queue definitions, C++ API additions, worker-side responsibilities, feature flag, inference server lifecycle.
- Updated `wiki/index.md` to include Interface Contract.

## [2026-05-08] Phase 2 | GPU Inference Server & Batched Integration
- Created `inference_server.py`: GPU process with adaptive batching, per-worker response queues.
- Rewrote `orchestrator.py`: Queue-based worker spawning, CPU-only workers.
- Added `self_play_game_batched()`: batched MCTS with Python callback, legal masking, Dirichlet noise.
- Updated `cli.py`: Inference server lifecycle, queue-based replay buffer wiring.
- Integrated `replay_queue` into `replay_buffer.py` and `trainer.py`.
- Created `test_inference.py`: K2–K4 validation tests.
- Deepseek D4–D5: EvalFnBatched, batched search overload, virtual loss, stall guard.

## [2026-05-08] Phase 3 | Memory & Pipeline Optimization
- Added model hot-reload to `inference_server.py`: mtime polling every 10 batches.
- Implemented async checkpointing in `trainer.py`: CPU-cloned state dict saved in background thread.
- Updated `cli.py` training loop: calls `maybe_checkpoint()` instead of synchronous save.
- Implemented async GUI AI in `gui.py`: MCTS in threading.Thread, responsive window.
- Created `test_phase3.py`: 7 validation tests for hot-reload, checkpointing, async GUI.
- Deepseek D6–D9: undo-stack, ring buffer, O(1) threefold, constexpr helpers, MCTSNode state removal.

## [2026-05-08] Phase 4 | Replay Buffer, Compilation & Validation
- Queue-based replay buffer streaming: workers push to `replay_queue`, trainer receiver thread.
- `torch.compile(mode="reduce-overhead")` on Inference Server (env flag `ALPHATAFL_COMPILE=1`).
- `ReplayDataset` + `DataLoader(batch_size=128, shuffle=True, num_workers=0, pin_memory=True)`.
- 50 test functions across 16 test classes in `tests/python/`.
- Phase 5 tests added for adaptive timeout and shared memory IPC.
- Feature flag defaults: `ALPHATAFL_BATCHED=1`, `ALPHATAFL_COMPILE=0`, `TIMEOUT_MS=5`, `MAX_BATCH=256`.

## [2026-05-08] Restructure | History Directory
- Moved `version_history.md` → `history/initial_v0.md` with updated title.
- Created `history/optimization_v1.md` documenting completed v1 changes.
- Created `history/readme.md` with conventions for when to create new history pages and high-level version summary.
- Updated `wiki/index.md`, `log.md`, and `WIKI.md` to reflect new structure.

## [2026-05-08] Update | Component Pages for v1
- Updated `game_state.md`: hash_counts map, board_history_ ring buffer, history_write_idx_, UndoInfo, apply_move_inplace(), undo_move(), check_threefold().
- Updated `mcts.md`: EvalFnBatched, batched search overload, is_pending, virtual loss, undo-based traversal, constexpr action index helpers.
- Updated `bindings.md`: batch_to_tensor, get_legal_moves_mask, EvalFnBatched constructor, batched search overload, removed board/history_hashes/get_historical_board.
- Updated `interface_contract.md`: Updated to reflect completed Phase 2–4, removed sign-off section.
- Updated `training_loop.md`: Queue-based replay buffer, async checkpointing, DataLoader.
- Updated `self_play.md`: Batched self-play path, inference server, queue-based storage.
- Updated `cli.md`: batch_size command, inference server lifecycle, queue-based training, auto mode.
- Updated `gui.md`: Async AI search, AI thinking indicator.
- Updated `architecture/README.md`: Inference Server component, updated data flow.
- Created `inference_server.md`: New page documenting the GPU inference server.

## [2026-05-08] Add | Phase 4 Benchmark Results
- Added benchmark results section to `optimization_v1.md`: MCTS throughput (49,893 states/sec), micro-benchmarks (to_tensor 0.38μs, batch_to_tensor 32μs for 64 states, select_child 0.16μs), KL parity (0.000000), full test pass rate (33/33), key findings.
- Updated `index.md` to reference the benchmark data.

## [2026-05-08] Update | Phase 4 C++ Benchmarks on Benchmarks Page
- Added C++ regression test results (33/33) to `optimization_v1_benchmarks.md` — including undo, engine, ring buffer, MCTS test breakdowns.
- Added missing micro-benchmarks: `get_legal_moves_mask()` (9.5 µs), undo overhead comparison (0.73 µs vs 0.84 µs, 1.15× speedup).
- Updated changelog on benchmarks page.
- Created `optimization_v1_benchmarks.md` in `wiki/docs/history/`: compiled all quantitative benchmarks from Phases 1–5A (C++ engine, bridge, inference pipeline, end-to-end self-play, architectural ceiling analysis).
- Added `benchmarks/bench_phase5.py` and `benchmarks/_bench_e2e_helper.py` for reproducible benchmarking.
- Updated `wiki/index.md` and `wiki/docs/history/readme.md` to reference the new benchmarks page.
- Added agent note to `wiki/WIKI.md`: all new wiki pages and directories must be created inside `wiki/`.
