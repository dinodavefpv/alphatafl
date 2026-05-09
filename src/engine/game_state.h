#pragma once

#include <vector>
#include <array>
#include <tuple>
#include <cstdint>
#include <unordered_map>

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
    int from_row = 0;
    int from_col = 0;
    int to_row = 0;
    int to_col = 0;

    constexpr Move() = default;
    constexpr Move(int fr, int fc, int tr, int tc)
        : from_row(fr), from_col(fc), to_row(tr), to_col(tc) {}

    constexpr bool operator==(const Move& other) const {
        return from_row == other.from_row && from_col == other.from_col &&
               to_row == other.to_row && to_col == other.to_col;
    }
};

struct UndoInfo {
    int from_row, from_col, to_row, to_col;
    Piece moved_piece;
    std::vector<std::tuple<int, int, Piece>> captured_pieces;
    Player prev_turn;
    Player prev_winner;
};

class GameState {
public:
    std::array<std::array<Piece, BOARD_SIZE>, BOARD_SIZE> board;
    Player current_turn;
    Player winner;
    std::unordered_map<uint64_t, uint8_t> hash_counts;
    std::array<std::array<std::array<Piece, BOARD_SIZE>, BOARD_SIZE>, 4> board_history_;
    int history_write_idx_;

    GameState();

    void reset();

    std::vector<Move> get_legal_moves() const;

    void apply_move(const Move& move);
    void apply_move_inplace(const Move& move, UndoInfo& undo);
    void undo_move(const Move& move, const UndoInfo& undo);

    GameState clone() const;

    uint64_t compute_hash() const;

    static constexpr int TENSOR_CHANNELS = 14;
    static constexpr int TENSOR_SIZE = BOARD_SIZE * BOARD_SIZE * TENSOR_CHANNELS;
    static constexpr int ACTION_SPACE = BOARD_SIZE * BOARD_SIZE * 40;

    std::vector<float> to_tensor() const;
    static std::vector<float> batch_to_tensor(const std::vector<GameState>& states);
    std::vector<float> get_legal_moves_mask() const;

    bool is_restricted_square(int row, int col) const;
    Piece get_piece(int row, int col) const { return board[row][col]; }
    std::array<std::array<Piece, BOARD_SIZE>, BOARD_SIZE> get_historical_board(int depth) const;

private:
    void check_captures(int row, int col, UndoInfo* undo);
    void check_win_condition(int row, int col);
    bool is_hostile(int row, int col, Player moving_player) const;
    Player get_piece_owner(Piece p) const;
    void check_threefold();
};

} // namespace alphatafl
