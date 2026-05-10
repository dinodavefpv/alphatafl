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
- `to_tensor()`: Returns `np.ndarray[float32, shape (14, 11, 11)]`. Zero-copy memory view.
- `batch_to_tensor(states)`: Static method. Returns `np.ndarray[float32, shape (N, 14, 11, 11)]`. Single Pybind11 boundary crossing.
- `get_legal_moves_mask()`: Returns `np.ndarray[float32, shape (4840,)]`. Binary mask of legal moves.

### `MCTS`
**Constructors:**
- `MCTS(eval_fn, c_puct=1.4)`: Single-leaf mode.
- `MCTS(eval_fn, eval_fn_batched, c_puct=1.4)`: Dual-mode with batched callback.

**Search methods:**
- `search(initial_state, num_simulations)`: Single-leaf MCTS. Returns `list[float]` length 4840.
- `search(initial_state, num_simulations, batch_size)`: Batched MCTS. Returns `list[float]` length 4840. Falls back to single-leaf if `batch_size <= 0` or no batched callback.

## Evaluation Function Interfaces

### Single-Leaf `eval_fn`
Python callable: `Callable[[GameState], tuple[list[float], float]]`
- Input: single `GameState`
- Returns: `(policy_probs [4840], value)`
- Used by GUI and as root evaluator in batched mode.

### Batched `eval_fn_batched`
Python callable: `Callable[[list[GameState]], tuple[list[list[float]], list[float]]]`
- Input: list of `GameState` objects
- Returns: `(policies [N×4840], values [N])`
- Used by batched `search()` for leaf evaluation.

## Build Process
The bindings are compiled as part of the CMake project into a shared library (e.g., `alphatafl_engine.pyd` on Windows) and placed in the `build/Release` directory.

## Source Files
- [bind.cpp](../../../src/bindings/bind.cpp)
