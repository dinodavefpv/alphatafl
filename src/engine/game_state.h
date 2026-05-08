#pragma once

#include <vector>
#include <array>
#include <deque>
#include <cstdint>
#include <iostream>

namespace alphatafl {

constexpr int BOARD_SIZE = 11;

enum class Piece : uint8_t {
    EMPTY = 0,
    ATTACKER = 1,
    DEFENDER = 2,
    KING = 3
};

enum class Player : uint8_t {
    NONE = 0,
    ATTACKER = 1,
    DEFENDER = 2,
    DRAW = 3
};

struct Move {
    int from_row;
    int from_col;
    int to_row;
    int to_col;

    Move() : from_row(0), from_col(0), to_row(0), to_col(0) {}
    Move(int fr, int fc, int tr, int tc) : from_row(fr), from_col(fc), to_row(tr), to_col(tc) {}

    bool operator==(const Move& other) const {
        return from_row == other.from_row && from_col == other.from_col &&
               to_row == other.to_row && to_col == other.to_col;
    }
};

class GameState {
public:
    std::array<std::array<Piece, BOARD_SIZE>, BOARD_SIZE> board;
    Player current_turn;
    Player winner; // NONE if game is ongoing, ATTACKER, DEFENDER, or DRAW if finished
    std::vector<uint64_t> history_hashes;
    std::deque<std::array<std::array<Piece, BOARD_SIZE>, BOARD_SIZE>> board_history;

    GameState();

    // Reset to the initial starting position
    void reset();

    // Get all legal moves for the current player
    std::vector<Move> get_legal_moves() const;

    // Apply a move and update the board state (including captures and turn change)
    void apply_move(const Move& move);

    GameState clone() const;

    uint64_t compute_hash() const;

    // Zero-copy tensor generation (Phase 1 D1)
    static constexpr int TENSOR_CHANNELS = 14;
    static constexpr int TENSOR_SIZE = BOARD_SIZE * BOARD_SIZE * TENSOR_CHANNELS;
    static constexpr int ACTION_SPACE = BOARD_SIZE * BOARD_SIZE * 40;

    std::vector<float> to_tensor() const;
    static std::vector<float> batch_to_tensor(const std::vector<GameState>& states);
    std::vector<float> get_legal_moves_mask() const;

    // Helper functions
    bool is_restricted_square(int row, int col) const;
    Piece get_piece(int row, int col) const { return board[row][col]; }
    std::array<std::array<Piece, BOARD_SIZE>, BOARD_SIZE> get_historical_board(int depth) const;
    
private:
    void check_captures(int row, int col);
    void check_win_condition(int row, int col);
    bool is_hostile(int row, int col, Player moving_player) const;
    Player get_piece_owner(Piece p) const;
};

} // namespace alphatafl
