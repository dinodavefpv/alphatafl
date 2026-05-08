#include "game_state.h"

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
    history_hashes.clear();
    history_hashes.push_back(compute_hash());
    board_history.clear();
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

            // Check 4 directions
            int dr[] = {-1, 1, 0, 0};
            int dc[] = {0, 0, -1, 1};

            for (int i = 0; i < 4; ++i) {
                int curr_r = r + dr[i];
                int curr_c = c + dc[i];

                while (curr_r >= 0 && curr_r < BOARD_SIZE && curr_c >= 0 && curr_c < BOARD_SIZE) {
                    if (board[curr_r][curr_c] != Piece::EMPTY) {
                        break; // Blocked by another piece
                    }
                    
                    // Cannot land on restricted squares unless it's the King
                    bool restricted = is_restricted_square(curr_r, curr_c);
                    if (restricted && p != Piece::KING) {
                        // Regular pieces cannot land here, but they CAN pass over the empty throne
                        if (!(curr_r == 5 && curr_c == 5)) {
                            // If it's a corner, they can't even pass over since it's the end of the board
                            // but technically loop will terminate soon anyway.
                            // We just don't add it as a valid move to land on.
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

void GameState::apply_move(const Move& move) {
    if (winner != Player::NONE) return;

    // Push current board before mutating
    board_history.push_front(board);
    if (board_history.size() > 3)
        board_history.pop_back();

    Piece p = board[move.from_row][move.from_col];
    board[move.from_row][move.from_col] = Piece::EMPTY;
    board[move.to_row][move.to_col] = p;

    check_win_condition(move.to_row, move.to_col);
    if (winner == Player::NONE) {
        check_captures(move.to_row, move.to_col);
        // After captures, check win condition again (e.g. king captured)
        // check_captures might set winner if the king is captured
    }

    if (winner == Player::NONE) {
        current_turn = (current_turn == Player::ATTACKER) ? Player::DEFENDER : Player::ATTACKER;
        // Check if next player has no legal moves (they lose)
        if (get_legal_moves().empty()) {
            winner = (current_turn == Player::ATTACKER) ? Player::DEFENDER : Player::ATTACKER;
        } else {
            // Check threefold repetition
            uint64_t hash = compute_hash();
            history_hashes.push_back(hash);
            int count = 0;
            for (uint64_t h : history_hashes) {
                if (h == hash) count++;
            }
            if (count >= 3) {
                winner = Player::DRAW;
            }
        }
    }
}

bool GameState::is_hostile(int row, int col, Player moving_player) const {
    // Out of bounds is not hostile
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return false;

    // Corners are always hostile
    if ((row == 0 || row == BOARD_SIZE - 1) && (col == 0 || col == BOARD_SIZE - 1)) return true;

    // The empty Throne is hostile
    if (row == 5 && col == 5 && board[row][col] == Piece::EMPTY) return true;

    // Friendly pieces (of the moving player) help capture enemy pieces
    Piece p = board[row][col];
    if (p == Piece::EMPTY) return false;

    return get_piece_owner(p) == moving_player;
}

void GameState::check_captures(int row, int col) {
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
                    // Check if there is a hostile piece or square on the other side
                    if (is_hostile(far_r, far_c, moving_player)) {
                        // Capture!
                        board[adj_r][adj_c] = Piece::EMPTY;
                        
                        // We also must clear history_hashes on a capture because the board state 
                        // can never be repeated after pieces are permanently removed.
                        history_hashes.clear();
                    }
                }
            } else if (adj_piece == Piece::KING && moving_player == Player::ATTACKER) {
                // Special King capture logic
                // Check if king is surrounded on 4 sides (or 3 if against edge/throne)
                int hostile_count = 0;
                int req_hostile = 4;
                
                // If King is on the edge of the board, the out-of-bounds side doesn't count.
                // We only have 3 valid adjacent squares. So we need 3 hostiles.
                if (adj_r == 0 || adj_r == BOARD_SIZE - 1 || adj_c == 0 || adj_c == BOARD_SIZE - 1) {
                    req_hostile = 3;
                }
                // Note: If King is adjacent to the throne, req_hostile remains 4.
                // The throne itself will be counted as 1 hostile square below, 
                // meaning we correctly need exactly 3 attackers + 1 throne.
                
                for (int j = 0; j < 4; ++j) {
                    int k_r = adj_r + dr[j];
                    int k_c = adj_c + dc[j];
                    if (k_r < 0 || k_r >= BOARD_SIZE || k_c < 0 || k_c >= BOARD_SIZE) {
                        continue; // Edge of board
                    }
                    if (k_r == 5 && k_c == 5) {
                        hostile_count++; // Throne is hostile
                    } else if (board[k_r][k_c] == Piece::ATTACKER) {
                        hostile_count++;
                    }
                }
                
                if (hostile_count >= req_hostile) {
                    // King is captured!
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
    copy.history_hashes = this->history_hashes;
    copy.board_history = this->board_history;
    return copy;
}

std::array<std::array<Piece, BOARD_SIZE>, BOARD_SIZE> GameState::get_historical_board(int depth) const {
    if (depth == 0) {
        return board;
    }
    int history_index = depth - 1;
    if (history_index >= 0 && history_index < static_cast<int>(board_history.size())) {
        return board_history[history_index];
    }
    // Return empty board if depth exceeds history
    std::array<std::array<Piece, BOARD_SIZE>, BOARD_SIZE> empty_board = {};
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            empty_board[r][c] = Piece::EMPTY;
        }
    }
    return empty_board;
}

void GameState::check_win_condition(int row, int col) {
    if (board[row][col] == Piece::KING) {
        if ((row == 0 || row == BOARD_SIZE - 1) && (col == 0 || col == BOARD_SIZE - 1)) {
            winner = Player::DEFENDER; // King escaped
        }
    }
}

} // namespace alphatafl
