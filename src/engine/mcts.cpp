#include "mcts.h"
#include <limits>

namespace alphatafl {

MCTSNode::MCTSNode(MCTSNode* parent, double prior)
    : parent(parent), visit_count(0), value_sum(0.0), prior(prior), is_expanded(false), is_pending(false) {}

double MCTSNode::get_value() const {
    if (visit_count == 0) return 0.0;
    return value_sum / visit_count;
}

int MCTSNode::select_child(double c_puct) const {
    double best_score = -std::numeric_limits<double>::infinity();
    int best_action = -1;
    
    for (int action : legal_action_indices) {
        double prob = child_priors[action];
        
        double score = 0.0;
        auto it = children.find(action);
        if (it != children.end()) {
            const MCTSNode* child = it->second.get();
            if (child->is_pending) continue;
            score = child->get_value() + c_puct * prob * (std::sqrt(static_cast<double>(visit_count)) / (1.0 + child->visit_count));
        } else {
            score = c_puct * prob * (std::sqrt(static_cast<double>(visit_count) + 1e-8) / 1.0);
        }
        
        if (score > best_score) {
            best_score = score;
            best_action = action;
        }
    }
    return best_action;
}

void MCTSNode::expand(const std::vector<double>& action_probs) {
    is_expanded = true;
    child_priors = action_probs;
    legal_action_indices.clear();
    for (size_t i = 0; i < action_probs.size(); ++i) {
        if (action_probs[i] > 0.0)
            legal_action_indices.push_back(static_cast<int>(i));
    }
}

MCTS::MCTS(EvalFn eval_fn, double c_puct)
    : eval_fn(eval_fn), eval_fn_batched(nullptr), c_puct(c_puct) {}

MCTS::MCTS(EvalFn eval_fn, EvalFnBatched eval_fn_batched, double c_puct)
    : eval_fn(eval_fn), eval_fn_batched(eval_fn_batched), c_puct(c_puct) {}

std::vector<double> MCTS::search(const GameState& initial_state, int num_simulations) {
    MCTSNode root;
    GameState traversal = initial_state.clone();

    auto eval_result = eval_fn(traversal);
    root.expand(eval_result.first);

    for (int i = 0; i < num_simulations; ++i) {
        MCTSNode* node = &root;
        std::vector<MCTSNode*> search_path;
        std::vector<UndoInfo> undo_stack;
        search_path.push_back(node);

        while (node->is_expanded) {
            int action = node->select_child(c_puct);
            if (action == -1) break;

            auto it = node->children.find(action);
            if (it != node->children.end()) {
                Move move = get_move_from_index(action, BOARD_SIZE);
                UndoInfo undo;
                traversal.apply_move_inplace(move, undo);
                undo_stack.push_back(undo);
                node = it->second.get();
                search_path.push_back(node);
            } else {
                Move move = get_move_from_index(action, BOARD_SIZE);
                UndoInfo undo;
                traversal.apply_move_inplace(move, undo);
                undo_stack.push_back(undo);

                auto child = std::make_unique<MCTSNode>(node, node->child_priors[action]);
                MCTSNode* child_ptr = child.get();
                node->children[action] = std::move(child);
                node = child_ptr;
                search_path.push_back(node);
                break;
            }
        }

        double leaf_value = 0.0;
        if (traversal.winner == Player::NONE) {
            auto child_eval = eval_fn(traversal);
            leaf_value = child_eval.second;
            node->expand(child_eval.first);
        } else {
            if (traversal.winner == Player::ATTACKER) {
                leaf_value = (traversal.current_turn == Player::ATTACKER) ? 1.0 : -1.0;
            } else if (traversal.winner == Player::DEFENDER) {
                leaf_value = (traversal.current_turn == Player::DEFENDER) ? 1.0 : -1.0;
            } else {
                leaf_value = 0.0;
            }
        }

        double curr_value = leaf_value;
        for (auto it = search_path.rbegin(); it != search_path.rend(); ++it) {
            (*it)->value_sum += curr_value;
            (*it)->visit_count += 1;
            curr_value = -curr_value;
        }

        // Undo all moves in reverse order
        for (auto it = undo_stack.rbegin(); it != undo_stack.rend(); ++it) {
            Move m{it->from_row, it->from_col, it->to_row, it->to_col};
            traversal.undo_move(m, *it);
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

std::vector<double> MCTS::search(const GameState& initial_state, int num_simulations, int batch_size) {
    if (batch_size <= 0 || !eval_fn_batched) {
        return search(initial_state, num_simulations);
    }

    MCTSNode root;
    GameState traversal = initial_state.clone();

    auto eval_result = eval_fn(traversal);
    root.expand(eval_result.first);

    int simulations_done = 0;
    std::vector<MCTSNode*> pending_nodes;
    std::vector<GameState> pending_states;
    int consecutive_all_pending_stall = 0;

    auto flush_batch = [&]() {
        if (pending_nodes.empty()) return;
        auto result = eval_fn_batched(pending_states);
        const auto& policies = result.first;
        const auto& values = result.second;

        for (size_t i = 0; i < pending_nodes.size(); ++i) {
            MCTSNode* node = pending_nodes[i];

            node->visit_count -= 3;
            node->value_sum += 3;
            node->visit_count += 1;
            node->value_sum += values[i];
            node->is_pending = false;
            node->expand(policies[i]);

            double back_val = values[i];
            MCTSNode* p = node->parent;
            while (p) {
                p->visit_count += 1;
                p->value_sum += back_val;
                back_val = -back_val;
                p = p->parent;
            }
        }

        pending_nodes.clear();
        pending_states.clear();
        consecutive_all_pending_stall = 0;
    };

    while (simulations_done < num_simulations) {
        MCTSNode* node = &root;
        std::vector<UndoInfo> undo_stack;

        while (node->is_expanded) {
            int action = node->select_child(c_puct);
            if (action == -1) {
                bool all_pending = true;
                bool any_child = false;
                for (int a : node->legal_action_indices) {
                    any_child = true;
                    auto it = node->children.find(a);
                    if (it != node->children.end() && !it->second->is_pending) {
                        all_pending = false;
                        break;
                    }
                }
                if (any_child && all_pending) {
                    consecutive_all_pending_stall++;
                    if (consecutive_all_pending_stall > 10) {
                        flush_batch();
                    }
                }
                break;
            }

            auto it = node->children.find(action);
            if (it != node->children.end()) {
                MCTSNode* child = it->second.get();
                if (child->is_pending) {
                    consecutive_all_pending_stall++;
                    break;
                }
                Move move = get_move_from_index(action, BOARD_SIZE);
                UndoInfo undo;
                traversal.apply_move_inplace(move, undo);
                undo_stack.push_back(undo);
                node = child;
            } else {
                Move move = get_move_from_index(action, BOARD_SIZE);
                UndoInfo undo;
                traversal.apply_move_inplace(move, undo);
                undo_stack.push_back(undo);

                auto child = std::make_unique<MCTSNode>(node, node->child_priors[action]);
                MCTSNode* child_ptr = child.get();
                node->children[action] = std::move(child);
                node = child_ptr;
                break;
            }
        }

        if (node == &root && root.is_expanded) {
            // Undo any moves made before breaking
            for (auto it = undo_stack.rbegin(); it != undo_stack.rend(); ++it) {
                Move m{it->from_row, it->from_col, it->to_row, it->to_col};
                traversal.undo_move(m, *it);
            }
            continue;
        }

        if (!node->is_expanded && traversal.winner == Player::NONE) {
            if (node->is_pending) {
                consecutive_all_pending_stall++;
                // Undo before continuing
                for (auto it = undo_stack.rbegin(); it != undo_stack.rend(); ++it) {
                    Move m{it->from_row, it->from_col, it->to_row, it->to_col};
                    traversal.undo_move(m, *it);
                }
                continue;
            }
            node->visit_count += 3;
            node->value_sum -= 3;
            node->is_pending = true;

            pending_nodes.push_back(node);
            pending_states.push_back(traversal.clone());
            consecutive_all_pending_stall = 0;

            if (static_cast<int>(pending_nodes.size()) >= batch_size) {
                // Undo before flush (traversal is at leaf, flush uses cloned states)
                for (auto it = undo_stack.rbegin(); it != undo_stack.rend(); ++it) {
                    Move m{it->from_row, it->from_col, it->to_row, it->to_col};
                    traversal.undo_move(m, *it);
                }
                undo_stack.clear();
                flush_batch();
            } else {
                for (auto it = undo_stack.rbegin(); it != undo_stack.rend(); ++it) {
                    Move m{it->from_row, it->from_col, it->to_row, it->to_col};
                    traversal.undo_move(m, *it);
                }
            }
        } else if (!node->is_expanded) {
            double leaf_value = 0.0;
            Player winner = traversal.winner;
            Player turn = traversal.current_turn;
            if (winner == Player::ATTACKER) {
                leaf_value = (turn == Player::ATTACKER) ? 1.0 : -1.0;
            } else if (winner == Player::DEFENDER) {
                leaf_value = (turn == Player::DEFENDER) ? 1.0 : -1.0;
            }

            double back_val = leaf_value;
            MCTSNode* p = node;
            while (p) {
                p->visit_count += 1;
                p->value_sum += back_val;
                back_val = -back_val;
                p = p->parent;
            }
            consecutive_all_pending_stall = 0;

            // Undo all moves
            for (auto it = undo_stack.rbegin(); it != undo_stack.rend(); ++it) {
                Move m{it->from_row, it->from_col, it->to_row, it->to_col};
                traversal.undo_move(m, *it);
            }
        }

        simulations_done++;
    }

    flush_batch();

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
