#include <gtest/gtest.h>
#include "game_state.h"

using namespace alphatafl;

TEST(GameStateTest, InitialState) {
    GameState state;
    EXPECT_EQ(state.current_turn, Player::ATTACKER);
    EXPECT_EQ(state.winner, Player::NONE);
    EXPECT_EQ(state.get_piece(5, 5), Piece::KING);
    EXPECT_EQ(state.get_piece(0, 3), Piece::ATTACKER);
    EXPECT_EQ(state.get_piece(5, 4), Piece::DEFENDER);
    EXPECT_EQ(state.get_piece(0, 0), Piece::EMPTY);
}

TEST(GameStateTest, LegalMovesFromInitialState) {
    GameState state;
    auto moves = state.get_legal_moves();
    // Attackers have 24 pieces. 
    // Top group: (0,3) to (0,7), (1,5)
    // Left group: (3,0) to (7,0), (5,1)
    // Right group: (3,10) to (7,10), (5,9)
    // Bottom group: (10,3) to (10,7), (9,5)
    // There are many moves. We just verify the list is not empty.
    EXPECT_FALSE(moves.empty());
    
    // Check specific move logic
    // (1,5) can move to (1,1), (1,2), (1,3), (1,4), (1,6), (1,7), (1,8), (1,9)
    // (1,5) can also move to (2,5)
    bool found_1_5_to_2_5 = false;
    for (const auto& m : moves) {
        if (m.from_row == 1 && m.from_col == 5 && m.to_row == 2 && m.to_col == 5) {
            found_1_5_to_2_5 = true;
            break;
        }
    }
    EXPECT_TRUE(found_1_5_to_2_5);
}

TEST(GameStateTest, CapturePiece) {
    GameState state;
    // Set up a custom board for capturing
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            state.board[r][c] = Piece::EMPTY;
        }
    }
    state.board[2][2] = Piece::ATTACKER;
    state.board[2][3] = Piece::DEFENDER;
    state.board[2][5] = Piece::ATTACKER; // Will move to 2,4
    
    state.current_turn = Player::ATTACKER;
    state.apply_move({2, 5, 2, 4});
    
    // Defender at 2,3 should be captured
    EXPECT_EQ(state.get_piece(2, 3), Piece::EMPTY);
    EXPECT_EQ(state.get_piece(2, 2), Piece::ATTACKER);
    EXPECT_EQ(state.get_piece(2, 4), Piece::ATTACKER);
}

TEST(GameStateTest, HostileSquareCapture) {
    GameState state;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            state.board[r][c] = Piece::EMPTY;
        }
    }
    // Corner is at (0,0)
    state.board[0][1] = Piece::DEFENDER;
    state.board[0][3] = Piece::ATTACKER; // Move to 0,2
    
    state.current_turn = Player::ATTACKER;
    state.apply_move({0, 3, 0, 2});
    
    // Defender at 0,1 should be captured against the corner (0,0)
    EXPECT_EQ(state.get_piece(0, 1), Piece::EMPTY);
    EXPECT_EQ(state.get_piece(0, 2), Piece::ATTACKER);
}

TEST(GameStateTest, KingEscape) {
    GameState state;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            state.board[r][c] = Piece::EMPTY;
        }
    }
    state.board[0][1] = Piece::KING;
    state.current_turn = Player::DEFENDER;
    
    state.apply_move({0, 1, 0, 0}); // Move to corner
    
    EXPECT_EQ(state.winner, Player::DEFENDER);
}

TEST(GameStateTest, KingCaptureOpen) {
    GameState state;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            state.board[r][c] = Piece::EMPTY;
        }
    }
    state.board[3][3] = Piece::KING;
    state.board[2][3] = Piece::ATTACKER;
    state.board[4][3] = Piece::ATTACKER;
    state.board[3][2] = Piece::ATTACKER;
    state.board[3][5] = Piece::ATTACKER;
    
    state.current_turn = Player::ATTACKER;
    state.apply_move({3, 5, 3, 4}); // Complete the surround
    
    EXPECT_EQ(state.winner, Player::ATTACKER);
}
