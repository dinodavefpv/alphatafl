#include "mcts.h"
#include <limits>

namespace alphatafl {

int get_action_index(int from_row, int from_col, int to_row, int to_col, int board_size) {
    int action_type = 0;
    int action_val = 0;
    if (to_row == from_row) { // Horizontal
        int dist = to_col - from_col;
        if (dist > 0) { action_type = 0; action_val = dist - 1; }
        else { action_type = 1; action_val = -dist - 1; }
    } else { // Vertical
        int dist = to_row - from_row;
        if (dist > 0) { action_type = 2; action_val = dist - 1; }
        else { action_type = 3; action_val = -dist - 1; }
    }
    return (from_row * board_size + from_col) * 40 + (action_type * 10 + action_val);
}

Move get_move_from_index(int index, int board_size) {
    int piece_idx = index / 40;
    int action_idx = index % 40;
    
    int from_r = piece_idx / board_size;
    int from_c = piece_idx % board_size;
    
    int action_type = action_idx / 10;
    int dist = (action_idx % 10) + 1;
    
    int to_r, to_c;
    if (action_type == 0) { to_r = from_r; to_c = from_c + dist; }
    else if (action_type == 1) { to_r = from_r; to_c = from_c - dist; }
    else if (action_type == 2) { to_r = from_r + dist; to_c = from_c; }
    else { to_r = from_r - dist; to_c = from_c; }
    
    return Move(from_r, from_c, to_r, to_c);
}

MCTSNode::MCTSNode(const GameState& state, MCTSNode* parent, double prior)
    : state(state), parent(parent), visit_count(0), value_sum(0.0), prior(prior), is_expanded(false) {}

double MCTSNode::get_value() const {
    if (visit_count == 0) return 0.0;
    return value_sum / visit_count;
}

int MCTSNode::select_child(double c_puct) const {
    double best_score = -std::numeric_limits<double>::infinity();
    int best_action = -1;
    
    for (size_t action = 0; action < child_priors.size(); ++action) {
        double prob = child_priors[action];
        if (prob == 0.0) continue;
        
        double score = 0.0;
        auto it = children.find(static_cast<int>(action));
        if (it != children.end()) {
            const MCTSNode* child = it->second.get();
            score = child->get_value() + c_puct * prob * (std::sqrt(static_cast<double>(visit_count)) / (1.0 + child->visit_count));
        } else {
            score = c_puct * prob * (std::sqrt(static_cast<double>(visit_count) + 1e-8) / 1.0);
        }
        
        if (score > best_score) {
            best_score = score;
            best_action = static_cast<int>(action);
        }
    }
    return best_action;
}

void MCTSNode::expand(const std::vector<double>& action_probs) {
    is_expanded = true;
    child_priors = action_probs;
}

MCTS::MCTS(EvalFn eval_fn, double c_puct) : eval_fn(eval_fn), c_puct(c_puct) {}

std::vector<double> MCTS::search(const GameState& initial_state, int num_simulations) {
    MCTSNode root(initial_state);
    
    auto eval_result = eval_fn(initial_state);
    root.expand(eval_result.first);

    for (int i = 0; i < num_simulations; ++i) {
        MCTSNode* node = &root;
        std::vector<MCTSNode*> search_path;
        search_path.push_back(node);

        while (node->is_expanded) {
            int action = node->select_child(c_puct);
            if (action == -1) break; // Should not happen if there are legal moves

            auto it = node->children.find(action);
            if (it != node->children.end()) {
                node = it->second.get();
                search_path.push_back(node);
            } else {
                Move move = get_move_from_index(action, BOARD_SIZE);
                GameState new_state = node->state.clone();
                new_state.apply_move(move);
                
                auto child = std::make_unique<MCTSNode>(new_state, node, node->child_priors[action]);
                MCTSNode* child_ptr = child.get();
                node->children[action] = std::move(child);
                node = child_ptr;
                search_path.push_back(node);
                break;
            }
        }

        double leaf_value = 0.0;
        if (node->state.winner == Player::NONE) {
            auto child_eval = eval_fn(node->state);
            leaf_value = child_eval.second;
            node->expand(child_eval.first);
        } else {
            if (node->state.winner == Player::DRAW) {
                leaf_value = 0.0;
            } else if (node->state.winner == Player::ATTACKER) {
                leaf_value = (node->state.current_turn == Player::ATTACKER) ? 1.0 : -1.0;
            } else if (node->state.winner == Player::DEFENDER) {
                leaf_value = (node->state.current_turn == Player::DEFENDER) ? 1.0 : -1.0;
            }
        }

        double curr_value = leaf_value;
        for (auto it = search_path.rbegin(); it != search_path.rend(); ++it) {
            (*it)->value_sum += curr_value;
            (*it)->visit_count += 1;
            curr_value = -curr_value; // Assuming turns alternate strictly
        }
    }

    std::vector<double> probs(BOARD_SIZE * BOARD_SIZE * 40, 0.0);
    double sum = 0.0;
    for (const auto& kv : root.children) {
        int action = kv.first;
        const MCTSNode* child = kv.second.get();
        probs[action] = child->visit_count;
        sum += child->visit_count;
    }
    if (sum > 0.0) {
        for (double& p : probs) {
            p /= sum;
        }
    }
    return probs;
}

} // namespace alphatafl
