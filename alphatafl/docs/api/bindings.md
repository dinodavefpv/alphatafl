---
title: Python Bindings
---

# Python Bindings (Pybind11)

The C++ engine is exposed to Python as the `alphatafl_engine` module using Pybind11. This allows the Python training loop and GUI to interact with the high-performance C++ simulator.

## Exposed Classes

### `Piece` (Enum)
- `EMPTY`, `ATTACKER`, `DEFENDER`, `KING`

### `Player` (Enum)
- `NONE`, `ATTACKER`, `DEFENDER`, `DRAW`

### `Move`
- `from_row`, `from_col`, `to_row`, `to_col`
- Constructor: `Move(fr, fc, tr, tc)`
- Equality operator: `move1 == move2`

### `GameState`
- `reset()`: Reset to starting position.
- `get_legal_moves()`: Returns a list of `Move` objects.
- `apply_move(move)`: Applies the move to the state.
- `get_piece(r, c)`: Returns the `Piece` at the given coordinates.
- `clone()`: Returns a deep copy of the state.
- `current_turn`: Current player (read/write).
- `winner`: Current winner (read/write, `NONE` if ongoing).
- `to_tensor()`: Returns `np.ndarray[float32, shape (14, 11, 11)]`. C++ allocates a buffer, copies the tensor into it, and hands ownership to numpy via a capsule (no second copy on the Python side; the original `std::vector<float>` is freed normally).
- `batch_to_tensor(states)`: Static method. Returns `np.ndarray[float32, shape (N, 14, 11, 11)]`. Single Pybind11 boundary crossing.
- `get_legal_moves_mask()`: Returns `np.ndarray[float32, shape (4840,)]`. Binary mask of legal moves.

### `MCTS`
**Constructors:**
- `MCTS(eval_fn, c_puct=1.4, dirichlet_alpha=0.3, dirichlet_epsilon=0.0)`: Single-leaf mode.
- `MCTS(eval_fn, eval_fn_batched, c_puct=1.4, dirichlet_alpha=0.3, dirichlet_epsilon=0.0)`: Dual-mode with batched callback.

**Search methods:**
- `search(initial_state, num_simulations)`: Single-leaf MCTS. Returns `list[float]` length 4840.
- `search(initial_state, num_simulations, batch_size)`: Batched MCTS. Returns `list[float]` length 4840. Falls back to single-leaf if `batch_size <= 0` or no batched callback.

**GIL:** `search` holds the GIL (`src/bindings/bind.cpp:169-175`). Every eval callback must hold it anyway, and the C++ work between callbacks is microseconds, so other Python threads (e.g. the GUI loop) still get the GIL at each callback (measured max main-thread stall < 2 ms). Releasing it was measured to slow search down because each callback then has to re-acquire it.

`dirichlet_alpha` and `dirichlet_epsilon` control the root-only Dirichlet noise that is mixed into the prior on the first expansion (standard AlphaZero: `policy = (1 - ε) * policy + ε * Dirichlet(α)`). They are surfaced in the Pybind11 constructor so the C++ MCTS can mix Dirichlet noise into the root priors right after the root expansion in `MCTS::search` (`src/engine/mcts.cpp:101-120`, `:210-229`), without crossing the boundary.

### `InferenceEngine` (Phase 5F)
A standalone C++ class that loads an ONNX-exported model and runs inference with the ONNX Runtime CUDA Execution Provider (with CPU fallback). It is not used by the self-play pipeline (`self_play_game_batched` evaluates through the Python Inference Server); it is exercised by benchmarks and debug scripts.

**Constructor:**
- `InferenceEngine(model_path, device="cuda")`: `model_path` is the path to an `.onnx` model exported via `scripts/export_onnx.py`. `device` selects the ORT execution provider; `"cpu"` forces CPU execution.

**Methods:**
- `evaluate(states)`: Batch inference. Returns `(policies, values)` where `policies` is shape `(N, 4840)` (list of lists of float) and `values` is shape `(N,)`.
- `evaluate_single(state)`: Single-state inference. Returns `(policy [4840], value)`.

**GIL:** both methods release the GIL for the C++ call (`py::call_guard<py::gil_scoped_release>`). Converting the returned `vector<vector<float>>` into nested Python lists still needs the GIL and dominates the stall (≈50 ms for N=64 on CPU).

Used as the `eval_fn` of an `MCTS` instance when running the C++ MCTS in self-contained mode (no inference server process). See `docs/api/interface_contract.md` §6 and `docs/history/optimization_v1_phase5.md` §5F.

## Evaluation Function Interfaces

### Single-Leaf `eval_fn`
Python callable: `Callable[[GameState], tuple[np.ndarray[float32, shape (4840,)], float]]`
- Input: single `GameState`
- Returns: `(policy_probs [4840], value)`
- Used by GUI, as the root evaluator in batched mode, and as the C++ `InferenceEngine` adapter.
- May also return a Python `list[float]` of length 4840; the C++ binding will convert it via numpy automatically.

### Batched `eval_fn_batched`
Python callable: `Callable[[list[GameState]], tuple[np.ndarray[float32, shape (N, 4840)], np.ndarray[float32, shape (N,)]]]`
- Input: list of `GameState` objects
- Returns: `(policies [N×4840], values [N])`
- Used by batched `search()` for leaf evaluation.
- A flat `np.ndarray` is preferred (single buffer); passing nested Python lists incurs a copy on the C++ side.

## Build Process
The bindings are compiled as part of the CMake project into a shared library (e.g., `alphatafl_engine.cp313-win_amd64.pyd` on Windows) and placed in the `build/Release` directory. The build links against the standalone ONNX Runtime under `lib/onnxruntime/` — LibTorch/PyTorch is no longer a build dependency after Phase 5F.

## Source Files
- [bind.cpp](../../../src/bindings/bind.cpp)
