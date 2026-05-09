#pragma once
#include "game_state.h"
#include <vector>
#include <memory>
#include <unordered_map>
#include <cmath>
#include <functional>

namespace alphatafl {

class MCTSNode {
public:
    MCTSNode* parent;
    int visit_count;
    double value_sum;
    double prior;
    bool is_expanded;
    bool is_pending;

    std::unordered_map<int, std::unique_ptr<MCTSNode>> children;
    std::vector<double> child_priors;
    std::vector<int> legal_action_indices;

    MCTSNode(MCTSNode* parent = nullptr, double prior = 0.0);

    double get_value() const;
    int select_child(double c_puct) const;
    void expand(const std::vector<double>& action_probs);
};

class MCTS {
public:
    double c_puct;
    using EvalFn = std::function<std::pair<std::vector<double>, double>(const GameState&)>;
    using EvalFnBatched = std::function<
        std::pair<std::vector<std::vector<double>>, std::vector<double>>(
            const std::vector<GameState>&
        )>;
    EvalFn eval_fn;
    EvalFnBatched eval_fn_batched;

    MCTS(EvalFn eval_fn, double c_puct = 1.4);
    MCTS(EvalFn eval_fn, EvalFnBatched eval_fn_batched, double c_puct = 1.4);

    std::vector<double> search(const GameState& initial_state, int num_simulations);
    std::vector<double> search(const GameState& initial_state, int num_simulations, int batch_size);
};

constexpr inline int get_action_index(int from_row, int from_col, int to_row, int to_col, int board_size = 11) {
    int action_type = 0;
    int action_val = 0;
    if (to_row == from_row) {
        int dist = to_col - from_col;
        if (dist > 0) { action_type = 0; action_val = dist - 1; }
        else { action_type = 1; action_val = -dist - 1; }
    } else {
        int dist = to_row - from_row;
        if (dist > 0) { action_type = 2; action_val = dist - 1; }
        else { action_type = 3; action_val = -dist - 1; }
    }
    return (from_row * board_size + from_col) * 40 + (action_type * 10 + action_val);
}

constexpr inline Move get_move_from_index(int index, int board_size = 11) {
    int piece_idx = index / 40;
    int action_idx = index % 40;
    
    int from_r = piece_idx / board_size;
    int from_c = piece_idx % board_size;
    
    int action_type = action_idx / 10;
    int dist = (action_idx % 10) + 1;
    
    int to_r = 0, to_c = 0;
    if (action_type == 0) { to_r = from_r; to_c = from_c + dist; }
    else if (action_type == 1) { to_r = from_r; to_c = from_c - dist; }
    else if (action_type == 2) { to_r = from_r + dist; to_c = from_c; }
    else { to_r = from_r - dist; to_c = from_c; }
    
    return Move(from_r, from_c, to_r, to_c);
}

} // namespace alphatafl
