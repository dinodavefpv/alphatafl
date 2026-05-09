#include "game_state.h"
#include "mcts.h"

namespace alphatafl {

GameState::GameState() {
    reset();
}

void GameState::reset() {
    const Piece init[BOARD_SIZE][BOARD_SIZE] = {
        {Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::ATTACKER, Piece::ATTACKER, Piece::ATTACKER, Piece::ATTACKER, Piece::ATTACKER, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY},
        {Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::ATTACKER, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY},
        {Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY},
        {Piece::ATTACKER, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::DEFENDER, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::ATTACKER},
        {Piece::ATTACKER, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::DEFENDER, Piece::DEFENDER, Piece::DEFENDER, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::ATTACKER},
        {Piece::ATTACKER, Piece::ATTACKER, Piece::EMPTY, Piece::DEFENDER, Piece::DEFENDER, Piece::KING, Piece::DEFENDER, Piece::DEFENDER, Piece::EMPTY, Piece::ATTACKER, Piece::ATTACKER},
        {Piece::ATTACKER, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::DEFENDER, Piece::DEFENDER, Piece::DEFENDER, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::ATTACKER},
        {Piece::ATTACKER, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::DEFENDER, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::ATTACKER},
        {Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY},
        {Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::ATTACKER, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY},
        {Piece::EMPTY, Piece::EMPTY, Piece::EMPTY, Piece::ATTACKER, Piece::ATTACKER, Piece::ATTACKER, Piece::ATTACKER, Piece::ATTACKER, Piece::EMPTY, Piece::EMPTY, Piece::EMPTY}
    };

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            board[r][c] = init[r][c];
        }
    }
    current_turn = Player::ATTACKER;
    winner = Player::NONE;

    hash_counts.clear();
    hash_counts[compute_hash()] = 1;

    for (int i = 0; i < 4; ++i) {
        for (int r = 0; r < BOARD_SIZE; ++r)
            for (int c = 0; c < BOARD_SIZE; ++c)
                board_history_[i][r][c] = Piece::EMPTY;
    }
    history_write_idx_ = 0;
}

uint64_t GameState::compute_hash() const {
    uint64_t hash = 14695981039346656037ULL;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            hash ^= static_cast<uint64_t>(board[r][c]);
            hash *= 1099511628211ULL;
        }
    }
    hash ^= static_cast<uint64_t>(current_turn);
    hash *= 1099511628211ULL;
    return hash;
}

bool GameState::is_restricted_square(int row, int col) const {
    if (row == 0 && col == 0) return true;
    if (row == 0 && col == BOARD_SIZE - 1) return true;
    if (row == BOARD_SIZE - 1 && col == 0) return true;
    if (row == BOARD_SIZE - 1 && col == BOARD_SIZE - 1) return true;
    if (row == 5 && col == 5) return true;
    return false;
}

Player GameState::get_piece_owner(Piece p) const {
    if (p == Piece::ATTACKER) return Player::ATTACKER;
    if (p == Piece::DEFENDER || p == Piece::KING) return Player::DEFENDER;
    return Player::NONE;
}

std::vector<Move> GameState::get_legal_moves() const {
    std::vector<Move> moves;
    if (winner != Player::NONE) return moves;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            Piece p = board[r][c];
            if (p == Piece::EMPTY) continue;
            
            Player owner = get_piece_owner(p);
            if (owner != current_turn) continue;

            int dr[] = {-1, 1, 0, 0};
            int dc[] = {0, 0, -1, 1};

            for (int i = 0; i < 4; ++i) {
                int curr_r = r + dr[i];
                int curr_c = c + dc[i];

                while (curr_r >= 0 && curr_r < BOARD_SIZE && curr_c >= 0 && curr_c < BOARD_SIZE) {
                    if (board[curr_r][curr_c] != Piece::EMPTY) {
                        break;
                    }
                    
                    bool restricted = is_restricted_square(curr_r, curr_c);
                    if (restricted && p != Piece::KING) {
                        if (!(curr_r == 5 && curr_c == 5)) {
                        }
                    } else {
                        moves.push_back({r, c, curr_r, curr_c});
                    }

                    curr_r += dr[i];
                    curr_c += dc[i];
                }
            }
        }
    }
    return moves;
}

void GameState::check_threefold() {
    uint64_t hash = compute_hash();
    hash_counts[hash]++;
    if (hash_counts[hash] >= 3) {
        winner = Player::DRAW;
    }
}

void GameState::apply_move(const Move& move) {
    UndoInfo dummy;
    apply_move_inplace(move, dummy);
}

void GameState::apply_move_inplace(const Move& move, UndoInfo& undo) {
    if (winner != Player::NONE) return;

    // Ring buffer: save current board before mutating
    board_history_[history_write_idx_] = board;
    history_write_idx_ = (history_write_idx_ + 1) % 4;

    // Store undo info
    undo.from_row = move.from_row;
    undo.from_col = move.from_col;
    undo.to_row = move.to_row;
    undo.to_col = move.to_col;
    undo.moved_piece = board[move.from_row][move.from_col];
    undo.captured_pieces.clear();
    undo.prev_turn = current_turn;
    undo.prev_winner = winner;

    // Move the piece
    Piece p = board[move.from_row][move.from_col];
    board[move.from_row][move.from_col] = Piece::EMPTY;
    board[move.to_row][move.to_col] = p;

    check_win_condition(move.to_row, move.to_col);
    if (winner == Player::NONE) {
        check_captures(move.to_row, move.to_col, &undo);
    }

    if (winner == Player::NONE) {
        current_turn = (current_turn == Player::ATTACKER) ? Player::DEFENDER : Player::ATTACKER;
        if (get_legal_moves().empty()) {
            winner = (current_turn == Player::ATTACKER) ? Player::DEFENDER : Player::ATTACKER;
        } else {
            check_threefold();
        }
    }
}

void GameState::undo_move(const Move& move, const UndoInfo& undo) {
    // Restore captured pieces
    for (const auto& cap : undo.captured_pieces) {
        int r, c;
        Piece piece;
        std::tie(r, c, piece) = cap;
        board[r][c] = piece;
    }

    // Move the piece back
    board[move.to_row][move.to_col] = Piece::EMPTY;
    board[move.from_row][move.from_col] = undo.moved_piece;

    // Restore turn and winner
    current_turn = undo.prev_turn;
    winner = undo.prev_winner;

    // Undo threefold hash count
    uint64_t hash = compute_hash();
    if (hash_counts.count(hash)) {
        hash_counts[hash]--;
        if (hash_counts[hash] == 0) {
            hash_counts.erase(hash);
        }
    }

    // Undo ring buffer write
    history_write_idx_ = (history_write_idx_ - 1 + 4) % 4;
}

bool GameState::is_hostile(int row, int col, Player moving_player) const {
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return false;

    if ((row == 0 || row == BOARD_SIZE - 1) && (col == 0 || col == BOARD_SIZE - 1)) return true;

    if (row == 5 && col == 5 && board[row][col] == Piece::EMPTY) return true;

    Piece p = board[row][col];
    if (p == Piece::EMPTY) return false;

    return get_piece_owner(p) == moving_player;
}

void GameState::check_captures(int row, int col, UndoInfo* undo) {
    Player moving_player = current_turn;
    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};

    for (int i = 0; i < 4; ++i) {
        int adj_r = row + dr[i];
        int adj_c = col + dc[i];
        int far_r = row + 2 * dr[i];
        int far_c = col + 2 * dc[i];

        if (adj_r >= 0 && adj_r < BOARD_SIZE && adj_c >= 0 && adj_c < BOARD_SIZE) {
            Piece adj_piece = board[adj_r][adj_c];
            if (adj_piece != Piece::EMPTY && adj_piece != Piece::KING) {
                Player adj_owner = get_piece_owner(adj_piece);
                if (adj_owner != moving_player) {
                    if (is_hostile(far_r, far_c, moving_player)) {
                        if (undo) {
                            undo->captured_pieces.push_back(std::make_tuple(adj_r, adj_c, adj_piece));
                        }
                        board[adj_r][adj_c] = Piece::EMPTY;
                    }
                }
            } else if (adj_piece == Piece::KING && moving_player == Player::ATTACKER) {
                int hostile_count = 0;
                int req_hostile = 4;
                
                if (adj_r == 0 || adj_r == BOARD_SIZE - 1 || adj_c == 0 || adj_c == BOARD_SIZE - 1) {
                    req_hostile = 3;
                }
                
                for (int j = 0; j < 4; ++j) {
                    int k_r = adj_r + dr[j];
                    int k_c = adj_c + dc[j];
                    if (k_r < 0 || k_r >= BOARD_SIZE || k_c < 0 || k_c >= BOARD_SIZE) {
                        continue;
                    }
                    if (k_r == 5 && k_c == 5) {
                        hostile_count++;
                    } else if (board[k_r][k_c] == Piece::ATTACKER) {
                        hostile_count++;
                    }
                }
                
                if (hostile_count >= req_hostile) {
                    winner = Player::ATTACKER;
                }
            }
        }
    }
}

GameState GameState::clone() const {
    GameState copy;
    copy.board = this->board;
    copy.current_turn = this->current_turn;
    copy.winner = this->winner;
    copy.hash_counts = this->hash_counts;
    copy.board_history_ = this->board_history_;
    copy.history_write_idx_ = this->history_write_idx_;
    return copy;
}

std::array<std::array<Piece, BOARD_SIZE>, BOARD_SIZE> GameState::get_historical_board(int depth) const {
    if (depth == 0) {
        return board;
    }
    if (depth < 1 || depth > 4) {
        std::array<std::array<Piece, BOARD_SIZE>, BOARD_SIZE> empty = {};
        return empty;
    }
    int idx = (history_write_idx_ - depth + 4) % 4;
    return board_history_[idx];
}

void GameState::check_win_condition(int row, int col) {
    if (board[row][col] == Piece::KING) {
        if ((row == 0 || row == BOARD_SIZE - 1) && (col == 0 || col == BOARD_SIZE - 1)) {
            winner = Player::DEFENDER;
        }
    }
}

std::vector<float> GameState::to_tensor() const {
    std::vector<float> tensor(TENSOR_SIZE, 0.0f);
    constexpr int CH_STRIDE = BOARD_SIZE * BOARD_SIZE;

    for (int depth = 0; depth < 4; ++depth) {
        const auto& hist_board = get_historical_board(depth);
        int base_ch = depth * 3;
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                Piece p = hist_board[r][c];
                int offset = r * BOARD_SIZE + c;
                if (p == Piece::ATTACKER)
                    tensor[base_ch * CH_STRIDE + offset] = 1.0f;
                else if (p == Piece::DEFENDER)
                    tensor[(base_ch + 1) * CH_STRIDE + offset] = 1.0f;
                else if (p == Piece::KING)
                    tensor[(base_ch + 2) * CH_STRIDE + offset] = 1.0f;
            }
        }
    }

    float turn_val = (current_turn == Player::ATTACKER) ? 1.0f : 0.0f;
    int ch12_base = 12 * CH_STRIDE;
    for (int i = 0; i < CH_STRIDE; ++i)
        tensor[ch12_base + i] = turn_val;

    int ch13_base = 13 * CH_STRIDE;
    tensor[ch13_base + 0 * BOARD_SIZE + 0] = 1.0f;
    tensor[ch13_base + 0 * BOARD_SIZE + (BOARD_SIZE - 1)] = 1.0f;
    tensor[ch13_base + (BOARD_SIZE - 1) * BOARD_SIZE + 0] = 1.0f;
    tensor[ch13_base + (BOARD_SIZE - 1) * BOARD_SIZE + (BOARD_SIZE - 1)] = 1.0f;
    tensor[ch13_base + 5 * BOARD_SIZE + 5] = 1.0f;

    return tensor;
}

std::vector<float> GameState::batch_to_tensor(const std::vector<GameState>& states) {
    size_t N = states.size();
    std::vector<float> tensor(N * TENSOR_SIZE, 0.0f);
    for (size_t i = 0; i < N; ++i) {
        auto single = states[i].to_tensor();
        std::copy(single.begin(), single.end(), tensor.begin() + i * TENSOR_SIZE);
    }
    return tensor;
}

std::vector<float> GameState::get_legal_moves_mask() const {
    std::vector<float> mask(ACTION_SPACE, 0.0f);
    auto moves = get_legal_moves();
    for (const auto& m : moves) {
        int idx = get_action_index(m.from_row, m.from_col, m.to_row, m.to_col, BOARD_SIZE);
        if (idx >= 0 && idx < ACTION_SPACE)
            mask[idx] = 1.0f;
    }
    return mask;
}

} // namespace alphatafl
