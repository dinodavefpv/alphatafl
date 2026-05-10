---
title: Monte Carlo Tree Search
---

# Monte Carlo Tree Search (MCTS)

AlphaTafl uses a neural-network-guided MCTS to search for optimal moves.

## Components

### `MCTSNode`
Represents a state in the search tree.
- `visit_count`: Number of times the node has been visited.
- `value_sum`: Total value accumulated from leaf evaluations.
- `prior`: The prior probability from the neural network.
- `is_expanded`: Whether the node has been expanded with a policy.
- `is_pending`: Whether the node is awaiting batch evaluation (Phase 2, virtual loss).
- `children`: A map of action indices to child nodes.
- `child_priors`: The full policy vector from expansion.
- `legal_action_indices`: Sparse list of action indices with non-zero prior (Phase 1, D3).
- Note: `GameState state` field was **removed** in Phase 3 (D6). The MCTS traversal uses a single mutable `GameState` with `apply_move_inplace()`/`undo_move()`.

### `MCTS` Class
The main search orchestrator.
- `c_puct`: The constant controlling exploration vs. exploitation (default: 1.4).
- `eval_fn`: Single-state callback: `(GameState) -> (vector<double> policy, double value)`.
- `eval_fn_batched`: Batched callback: `(vector<GameState>) -> (vector<vector<double>> policies, vector<double> values)`. May be null for single-leaf fallback.

### Constructors
- `MCTS(eval_fn, c_puct=1.4)`: Single-leaf mode (used by GUI).
- `MCTS(eval_fn, eval_fn_batched, c_puct=1.4)`: Dual-mode (allows batched search).

## Search Algorithms

### Single-Leaf Search (`search(state, sims)`)
Standard sequential MCTS. Used by GUI and for backward compatibility / A/B parity testing.

1. Evaluate root with `eval_fn`, expand root.
2. For each simulation:
   a. **Select**: Traverse tree using PUCT (skip pending nodes).
   b. **Expand**: When reaching unexpanded leaf, evaluate with `eval_fn`, expand.
   c. **Backpropagate**: Propagate value up the tree (negated at each level).
   d. **Undo**: Reverse all moves made during traversal.
3. Terminal nodes get hard-coded values based on winner/turn.
4. Returns visit-count-normalized policy vector (size 4840).

### Batched Search (`search(state, sims, batch_size)`)
Uses virtual loss for parallelism within a single thread. Falls back to single-leaf if `batch_size <= 0` or `eval_fn_batched` is null.

1. Evaluate root with `eval_fn`, expand root.
2. For each simulation:
   a. Select from root using PUCT (skip `is_pending == true` nodes).
   b. When reaching unexpanded, non-terminal node:
      - Set `is_pending = true`, add virtual loss (`visit += 3, value -= 3`).
      - Clone `GameState` and add to pending batch.
   c. When reaching terminal node: backpropagate immediately.
   d. If all children pending: increment stall counter; if > 10, force-flush.
3. When batch is full or all simulations queued:
   a. Call `eval_fn_batched` with all pending states.
   b. For each result: remove virtual loss, expand node, backpropagate via parent pointer.
4. Repeat if simulations remain.

## Virtual Loss (D5)
Standard AlphaZero formulation:
- On queue: `visit_count += 3, value_sum -= 3, is_pending = true`
- On result: remove virtual visits/penalty, then `visit_count += 1, value_sum += actual_value`
- Makes pending nodes unattractive to subsequent traversals.
- Pending nodes are never added to the batch twice (double-queue guard).

## PUCT Formula
`score = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`
- For unvisited children: `score = c_puct * P * sqrt(N_parent + 1e-8)`

## Action Indexing
To map moves to neural network outputs, a flat index is used:
- Total actions: 11 × 11 × 40 = 4840.
- For each of the 121 squares, there are 40 possible moves: 4 directions × 10 distances.

### Helper Functions (constexpr inline, D9)
- `get_action_index(from_r, from_c, to_r, to_c, board_size=11)`
- `get_move_from_index(index, board_size=11)`

## Source Files
- [mcts.h](../../../src/engine/mcts.h)
- [mcts.cpp](../../../src/engine/mcts.cpp)
- [mcts.py](../../../src/training/mcts.py) (Python fallback implementation, deprecated)
