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
    std::vector<double> priors(121 * 40, 0.0);
    for (const auto& m : moves) {
        int idx = get_action_index(m.from_row, m.from_col, m.to_row, m.to_col);
        priors[idx] = 1.0 / num_legal;
    }
    node.expand(priors);

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

    MCTS::EvalFn eval = [](const GameState& s) -> std::pair<std::vector<double>, double> {
        auto legal = s.get_legal_moves();
        std::vector<double> probs(121 * 40, 0.0);
        for (const auto& m : legal) {
            int idx = get_action_index(m.from_row, m.from_col, m.to_row, m.to_col);
            probs[idx] = 1.0 / legal.size();
        }
        return {probs, 0.0};
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
