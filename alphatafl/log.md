# AlphaTafl Wiki Log

## [2026-05-09] Add | Python Callback Micro-Profile
- Instrumented `eval_fn_batched` in `benchmarks/time_single_game.py` with per-step `time.perf_counter()` timers. Collected 1755 batch samples across one 200-turn 800-sim game.
- Measured per-batch breakdown: mp.Queue.get (await server) = 13.9ms (58%), Python list `.tolist()` build = 6.5ms (27%), C++ `batch_to_tensor` + pybind11 wrap = 3.0ms (12%), SHM memcpy = 0.5ms (2%), mp.Queue.put = negligible.
- Confirmed 84% of per-batch Python overhead (23.4ms) is eliminable by moving NN inference into C++ (removes mp.Queue pipes, list construction, pybind11 tensor wrap).
- Updated `wiki/docs/history/optimization_v1_benchmarks.md`: Added Section 8.8 (Callback Micro-Profile) and 8.9 (Revised Bottleneck Summary with measured data). Replaced estimated bottleneck numbers with instrumented data throughout 8.9-8.11 and conclusion.
- Answered open question #7. Added #8 (pybind11 game loop overhead profiling).
- Updated `wiki/log.md` and `wiki/index.md`.

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
