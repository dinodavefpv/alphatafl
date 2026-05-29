#include <gtest/gtest.h>
#include "game_state.h"
#include "mcts.h"
#include <chrono>
#include <iostream>
#include <random>

using namespace alphatafl;

template<typename F>
double measure_us(F&& fn, int iterations = 10000) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return static_cast<double>(duration) / iterations;
}

TEST(ProfileTest, SelectChildSpeed) {
    GameState state;
    // Play some moves to get a mid-game position
    std::mt19937 rng(42);
    for (int i = 0; i < 10; ++i) {
        auto moves = state.get_legal_moves();
        if (moves.empty()) break;
        state.apply_move(moves[rng() % moves.size()]);
    }

    auto moves = state.get_legal_moves();
    int num_legal = static_cast<int>(moves.size());

    // Build a node with these legal moves as priors
    MCTSNode node;
    std::vector<float> priors(121 * 40, 0.0f);
    std::vector<float> mask(121 * 40, 0.0f);
    for (const auto& m : moves) {
        int idx = get_action_index(m.from_row, m.from_col, m.to_row, m.to_col);
        mask[idx] = 1.0f;
    }
    node.expand(priors, mask);

    double avg_us = measure_us([&]() {
        node.select_child(1.4);
    }, 50000);

    std::cout << "[Profile] select_child on " << num_legal << "-move position: "
              << avg_us << " us/call" << std::endl;

    // Verify it's fast enough: the sparse iteration should be well under 10us
    // (old full-4840 was O(4840), new is O(num_legal))
    EXPECT_LT(avg_us, 50.0);
}

TEST(ProfileTest, ToTensorSpeed) {
    GameState state;
    // Play some moves to populate board_history
    for (int i = 0; i < 5; ++i) {
        auto moves = state.get_legal_moves();
        if (moves.empty()) break;
        state.apply_move(moves[0]);
    }

    double avg_us = measure_us([&]() {
        volatile auto t = state.to_tensor();
        (void)t;
    }, 10000);

    std::cout << "[Profile] to_tensor: " << avg_us << " us/call" << std::endl;

    EXPECT_LT(avg_us, 100.0); // Under 100us
}

TEST(ProfileTest, BatchToTensorSpeed) {
    GameState state;
    std::vector<GameState> states;
    for (int i = 0; i < 64; ++i) {
        GameState s;
        for (int j = 0; j < (i % 5); ++j) {
            auto moves = s.get_legal_moves();
            if (moves.empty()) break;
            s.apply_move(moves[0]);
        }
        states.push_back(s);
    }

    double avg_us = measure_us([&]() {
        volatile auto t = GameState::batch_to_tensor(states);
        (void)t;
    }, 100);

    std::cout << "[Profile] batch_to_tensor(64 states): " << avg_us << " us/call" << std::endl;

    EXPECT_LT(avg_us, 5000.0); // Under 5ms for 64
}

TEST(ProfileTest, LegalMovesMaskSpeed) {
    GameState state;

    double avg_us = measure_us([&]() {
        volatile auto m = state.get_legal_moves_mask();
        (void)m;
    }, 1000);

    std::cout << "[Profile] get_legal_moves_mask: " << avg_us << " us/call" << std::endl;
}

TEST(ProfileTest, MctsThroughput) {
    GameState state;

    MCTS::EvalFn eval = [](const GameState& s) -> std::pair<std::vector<float>, float> {
        auto legal = s.get_legal_moves();
        std::vector<float> probs(121 * 40, 0.0f);
        for (const auto& m : legal) {
            int idx = get_action_index(m.from_row, m.from_col, m.to_row, m.to_col);
            probs[idx] = 1.0f / legal.size();
        }
        return {probs, 0.0f};
    };

    MCTS mcts(eval, 1.4);

    auto start = std::chrono::high_resolution_clock::now();
    int total_sims = 0;
    int runs = 5;

    for (int i = 0; i < runs; ++i) {
        auto r = mcts.search(state, 200);
        total_sims += 200;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double states_per_sec = (total_sims * 1000000.0) / total_us;

    std::cout << "[Profile] MCTS throughput: " << states_per_sec << " states/sec ("
              << total_sims << " sims in " << (total_us / 1000.0) << " ms)" << std::endl;

    // This is a relative check — just verify it completes
    EXPECT_GT(states_per_sec, 0.0);
}

TEST(ProfileTest, UndoOverhead) {
    // Measure time to apply+undo a move vs just applying
    GameState state;

    auto m = state.get_legal_moves()[0];

    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        state.apply_move(m);
        // Reset state each time... this is tricky
    }
    auto end1 = std::chrono::high_resolution_clock::now();

    // More meaningful: measure apply_inplace + undo vs clone + apply
    state = GameState();
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5000; ++i) {
        UndoInfo undo;
        state.apply_move_inplace(m, undo);
        state.undo_move(m, undo);
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    auto apply_undo_us = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() / 5000.0;

    state = GameState();
    auto start3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5000; ++i) {
        GameState copy = state.clone();
        copy.apply_move(m);
    }
    auto end3 = std::chrono::high_resolution_clock::now();
    auto clone_apply_us = std::chrono::duration_cast<std::chrono::microseconds>(end3 - start3).count() / 5000.0;

    std::cout << "[Profile] apply_inplace+undo: " << apply_undo_us << " us"
              << " | clone+apply: " << clone_apply_us << " us"
              << " | speedup: " << (clone_apply_us / apply_undo_us) << "x" << std::endl;

    EXPECT_LT(apply_undo_us, clone_apply_us * 2.0);
}

TEST(ProfileTest, DeepTreeTraversal) {
    // Profile MCTS search at increasing simulation depths to measure
    // how per-sim latency scales with tree size (deep trees are more
    // expensive because PUCT select_child traverses from root).

    GameState state;
    std::mt19937 rng(42);
    // Play 20 random moves to get a complex mid-game position
    for (int i = 0; i < 20; ++i) {
        auto moves = state.get_legal_moves();
        if (moves.empty()) break;
        state.apply_move(moves[rng() % moves.size()]);
    }
    if (state.winner != Player::NONE || state.get_legal_moves().empty()) {
        state = GameState(); // reset if terminal
    }

    MCTS::EvalFn eval = [](const GameState& s) -> std::pair<std::vector<float>, float> {
        auto legal = s.get_legal_moves();
        std::vector<float> probs(121 * 40, 0.0f);
        for (const auto& m : legal) {
            int idx = get_action_index(m.from_row, m.from_col, m.to_row, m.to_col);
            probs[idx] = 1.0f / legal.size();
        }
        return {probs, 0.0f};
    };

    std::cout << "\n[Profile] Deep Tree Traversal (mid-game position, 20 random moves in):"
              << std::endl;

    for (int sims : {100, 400, 800}) {
        MCTS mcts(eval, 1.4);
        auto start = std::chrono::high_resolution_clock::now();
        auto result = mcts.search(state, sims);
        auto end = std::chrono::high_resolution_clock::now();
        auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double us_per_sim = static_cast<double>(total_us) / sims;
        double states_per_sec = (sims * 1000000.0) / total_us;

        std::cout << "  " << sims << " sims: " << (total_us / 1000.0) << " ms total, "
                  << us_per_sim << " us/sim, "
                  << static_cast<int>(states_per_sec) << " states/sec" << std::endl;
    }

    // Also profile just the per-sim overhead: how much time does a single
    // simulation (select + expand + backprop) take in a deep tree?
    std::cout << "\n[Profile] Per-simulation latency in deep tree (800 sims, 10 repeats):"
              << std::endl;
    {
        MCTS mcts(eval, 1.4);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            mcts.search(state, 800);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double us_per_sim = static_cast<double>(total_us) / (800 * 10);
        std::cout << "  " << us_per_sim << " us/sim (synthetic eval, no NN)"
                  << std::endl;
    }

    // Just verify it completes
    SUCCEED();
}

TEST(ProfileTest, NodeMemoryFootprint) {
    std::cout << "\n[Profile] Memory Footprint Analysis" << std::endl;
    std::cout << "  sizeof(Piece) = " << sizeof(Piece) << std::endl;
    std::cout << "  sizeof(Move) = " << sizeof(Move) << std::endl;
    std::cout << "  sizeof(UndoInfo) = " << sizeof(UndoInfo) << std::endl;
    std::cout << "  sizeof(GameState) = " << sizeof(GameState) << std::endl;
    std::cout << "  sizeof(std::array<Piece,11x11>) = " << sizeof(std::array<std::array<Piece, BOARD_SIZE>, BOARD_SIZE>) << std::endl;
    std::cout << "  sizeof(MCTSNode) (empty) = " << sizeof(MCTSNode) << std::endl;

    // Measure child_priors capacity impact
    MCTSNode node;
    int action_space = GameState::ACTION_SPACE;
    std::vector<float> priors(action_space, 0.0f);
    std::vector<float> mask(action_space, 0.0f);
    for (int i = 0; i < 120; ++i) mask[i] = 1.0f;
    node.expand(priors, mask);

    size_t priors_bytes = node.child_priors.capacity() * sizeof(float);
    size_t indices_bytes = node.legal_action_indices.capacity() * sizeof(int);
    size_t children_overhead = node.children.size() * (sizeof(int) + sizeof(std::unique_ptr<MCTSNode>) + sizeof(void*) * 2);
    size_t total_node_bytes = sizeof(MCTSNode) + priors_bytes + indices_bytes + children_overhead;

    std::cout << "  Expanded node breakdown:" << std::endl;
    std::cout << "    child_priors capacity: " << node.child_priors.capacity() << " floats = " << priors_bytes << " bytes (" << (priors_bytes / 1024.0) << " KB)" << std::endl;
    std::cout << "    legal_action_indices capacity: " << node.legal_action_indices.capacity() << " ints = " << indices_bytes << " bytes" << std::endl;
    std::cout << "    children overestimate: ~" << children_overhead << " bytes" << std::endl;
    std::cout << "    total per expanded node: ~" << total_node_bytes << " bytes (" << (total_node_bytes / 1024.0) << " KB)" << std::endl;

    std::cout << "    Non-zero priors: 120 / " << action_space << " (" << (100.0 * 120 / action_space) << "%)" << std::endl;
    std::cout << "    waste if storing dense: " << ((priors_bytes - 120 * sizeof(float)) / 1024.0) << " KB of zeros per node" << std::endl;

    EXPECT_GT(total_node_bytes, 0);
}

TEST(ProfileTest, MctsMemoryGrowth) {
    GameState state;
    std::mt19937 rng(42);
    for (int i = 0; i < 20; ++i) {
        auto moves = state.get_legal_moves();
        if (moves.empty()) break;
        state.apply_move(moves[rng() % moves.size()]);
    }
    if (state.winner != Player::NONE || state.get_legal_moves().empty()) {
        state = GameState();
    }

    int action_space = GameState::ACTION_SPACE;
    MCTS::EvalFn eval = [action_space](const GameState& s) -> std::pair<std::vector<float>, float> {
        auto legal = s.get_legal_moves();
        std::vector<float> probs(action_space, 0.0f);
        for (const auto& m : legal) {
            int idx = get_action_index(m.from_row, m.from_col, m.to_row, m.to_col);
            probs[idx] = 1.0f / legal.size();
        }
        return {probs, 0.0f};
    };

    static constexpr size_t EST_NODE_BYTES = 2000;

    std::cout << "\n[Profile] MCTS Tree Memory Growth vs Sims (synthetic eval, no NN)" << std::endl;
    std::cout << "  Sims   | Nodes | Est Mem (KB) | KB/Sim | Sims/sec | Time(ms)" << std::endl;
    std::cout << "  " << std::string(72, '-') << std::endl;

    for (int sims : {100, 400, 800, 1600}) {
        MCTSNode root_node;
        auto res = eval(state);
        root_node.expand(res.first, state.get_legal_moves_mask());

        GameState traversal = state.clone();
        int total_nodes = 1;

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < sims; ++i) {
            MCTSNode* node = &root_node;
            std::vector<UndoInfo> undo_stack;

            while (node->is_expanded) {
                int action = node->select_child(1.4);
                if (action == -1) break;
                auto it = node->children.find(action);
                if (it != node->children.end()) {
                    Move move = get_move_from_index(action, BOARD_SIZE);
                    UndoInfo undo;
                    traversal.apply_move_inplace(move, undo);
                    undo_stack.push_back(undo);
                    node = it->second.get();
                } else {
                    Move move = get_move_from_index(action, BOARD_SIZE);
                    UndoInfo undo;
                    traversal.apply_move_inplace(move, undo);
                    undo_stack.push_back(undo);

                    auto child = std::make_unique<MCTSNode>(node, node->get_child_prior(action));
                    MCTSNode* child_ptr = child.get();
                    node->children[action] = std::move(child);
                    node = child_ptr;
                    total_nodes++;
                    break;
                }
            }

            if (traversal.winner == Player::NONE) {
                auto child_eval = eval(traversal);
                node->expand(child_eval.first, traversal.get_legal_moves_mask());
            }

            double leaf_value = 0.0;
            if (traversal.winner == Player::ATTACKER) {
                leaf_value = (traversal.current_turn == Player::ATTACKER) ? 1.0 : -1.0;
            } else if (traversal.winner == Player::DEFENDER) {
                leaf_value = (traversal.current_turn == Player::DEFENDER) ? 1.0 : -1.0;
            }

            MCTSNode* p = node;
            while (p) {
                p->visit_count += 1;
                p->value_sum += leaf_value;
                leaf_value = -leaf_value;
                p = p->parent;
            }

            for (auto it = undo_stack.rbegin(); it != undo_stack.rend(); ++it) {
                Move m{it->from_row, it->from_col, it->to_row, it->to_col};
                traversal.undo_move(m, *it);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double sims_per_sec = (sims * 1000000.0) / total_us;
        double total_ms = total_us / 1000.0;

        size_t est_mem = total_nodes * EST_NODE_BYTES;
        double kb_per_sim = (double)est_mem / sims;

        std::cout << "  " << sims << "    | " << total_nodes << "    | "
                  << (est_mem / 1024) << "        | " << kb_per_sim << "  | "
                  << static_cast<int>(sims_per_sec) << "     | " << total_ms << std::endl;
    }

    EXPECT_GT(sizeof(MCTSNode), 0);
}
