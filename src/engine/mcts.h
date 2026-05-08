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
    GameState state;
    MCTSNode* parent;
    int visit_count;
    double value_sum;
    double prior;
    bool is_expanded;
    
    std::unordered_map<int, std::unique_ptr<MCTSNode>> children;
    std::vector<double> child_priors;

    MCTSNode(const GameState& state, MCTSNode* parent = nullptr, double prior = 0.0);

    double get_value() const;
    int select_child(double c_puct) const;
    void expand(const std::vector<double>& action_probs);
};

class MCTS {
public:
    double c_puct;
    // Eval function: takes GameState, returns pair of (action_probs[4840], value)
    using EvalFn = std::function<std::pair<std::vector<double>, double>(const GameState&)>;
    EvalFn eval_fn;

    MCTS(EvalFn eval_fn, double c_puct = 1.4);

    std::vector<double> search(const GameState& initial_state, int num_simulations);
};

// Helper functions for action index conversions
int get_action_index(int from_row, int from_col, int to_row, int to_col, int board_size = 11);
Move get_move_from_index(int index, int board_size = 11);

} // namespace alphatafl
