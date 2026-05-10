---
title: Game State Engine
---

# GameState Engine

The `GameState` class is the core of the Hnefatafl simulation. It handles board representation, move validation, and game rules.

## Data Structures

### `Piece` Enum
- `EMPTY (0)`
- `ATTACKER (1)`
- `DEFENDER (2)`
- `KING (3)`

### `Player` Enum
- `NONE (0)`: Game ongoing.
- `ATTACKER (1)`
- `DEFENDER (2)`
- `DRAW (3)`

### `Move` Struct
Represents a move from `(from_row, from_col)` to `(to_row, to_col)`. Has `constexpr` constructors and equality operator.

### `UndoInfo` Struct
Stores data needed to reverse a move:
- `from_row, from_col, to_row, to_col` — the move coordinates
- `moved_piece` — the piece that was moved
- `captured_pieces` — vector of `(row, col, Piece)` tuples for each captured piece
- `prev_turn` — player before the move
- `prev_winner` — winner before the move

### Internal State Tracking (v1)
- `hash_counts`: `std::unordered_map<uint64_t, uint8_t>` mapping board hashes to occurrence counts. Used for O(1) threefold repetition detection. **Never cleared on capture.**
- `board_history_`: `std::array<std::array<std::array<Piece, 11>, 11>, 4>` ring buffer storing the last 3 board states. No heap allocation.
- `history_write_idx_`: Current write position in the ring buffer (increment-then-write convention).
- `compute_hash()`: Computes an FNV-1a hash of the current board + turn for repetition checking.
- `get_historical_board(depth)`: Returns the board state `depth` moves ago (0 = current, 1 = T-1, etc.). Depth 0 returns board directly; depth 1-3 reads from ring buffer via modular arithmetic; depth > 3 returns an empty board.

## Core Logic

### `get_legal_moves()`
Generates all legal moves for the current player.
- Moves are orthogonal (Rook-like).
- Pieces cannot jump over others.
- Only the King can land on restricted squares (Corners and Throne).
- Pieces CAN pass over the empty Throne.

### `apply_move(Move)` / `apply_move_inplace(Move, UndoInfo)`
Updates the game state by applying a move. `apply_move()` calls `apply_move_inplace()` with a dummy `UndoInfo` for external use. `apply_move_inplace()` is used internally by MCTS with real undo tracking.

1. Saves current board to ring buffer at `history_write_idx_`, advances index.
2. Records undo info (position, piece, turn, winner).
3. Moves the piece.
4. Checks for victory (King reaching a corner).
5. Checks for captures (Custodian + King capture).
6. Swaps the turn.
7. Checks for loss by no legal moves.
8. Checks for threefold repetition via `hash_counts`.

### `undo_move(Move, UndoInfo)`
Reverses a move applied by `apply_move_inplace()`:
1. Restores all captured pieces from `UndoInfo.captured_pieces`.
2. Moves the piece back to its original position.
3. Restores `current_turn` and `winner`.
4. Decrements `hash_counts[hash]` (removes key if count reaches 0).
5. Decrements `history_write_idx_` (unwinds ring buffer).

### `check_threefold()`
O(1) implementation: increments `hash_counts[compute_hash()]`. If count >= 3, sets `winner = Player::DRAW`. Never called on capture.

### `check_captures(row, col, UndoInfo*)`
Implemented in `game_state.cpp`. Uses the "sandwich" rule.
- A piece is captured if it is between the moved piece and another friendly piece or a hostile square (Corner/Throne).
- Captured pieces are recorded in `UndoInfo.captured_pieces` if undo pointer is non-null.
- Capture logic for the King is specialized: requires 4-way surrounding (or 3-way on edges).

### `is_hostile(row, col, player)`
Determines if a square is hostile to the given player.
- Corners are always hostile.
- The empty Throne is hostile.
- Friendly pieces are hostile to enemies (helping in capture).

### Tensor Methods (D1)
- `to_tensor()`: Returns `std::vector<float>` of size 1694. 14-channel encoding: 0-2 current board, 3-5 T-1, 6-8 T-2, 9-11 T-3, 12 turn indicator, 13 restricted squares.
- `batch_to_tensor(states)`: Static method returning flat `std::vector<float>` of `N * 1694` elements.
- `get_legal_moves_mask()`: Returns `std::vector<float>(4840)` with 1.0 at legal move indices.

### Action Index Helpers (in mcts.h)
- `get_action_index(from_r, from_c, to_r, to_c)`: Encodes a move as flat index in [0, 4840). `constexpr inline`.
- `get_move_from_index(index)`: Decodes a flat index back to `Move`. `constexpr inline`.
- Encoding: `piece_index * 40 + action_type * 10 + distance - 1`, where action types are 0=right, 1=left, 2=down, 3=up.

## Source Files
- [game_state.h](../../../src/engine/game_state.h)
- [game_state.cpp](../../../src/engine/game_state.cpp)
