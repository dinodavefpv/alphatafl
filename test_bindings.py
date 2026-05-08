import sys
import os

# Add the build directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'build', 'Release'))

import alphatafl_engine as engine

def test_bindings():
    state = engine.GameState()
    print(f"Initial turn: {state.current_turn}")
    print(f"Winner: {state.winner}")
    
    moves = state.get_legal_moves()
    print(f"Number of legal moves: {len(moves)}")
    
    if moves:
        move = moves[0]
        print(f"Applying move: ({move.from_row}, {move.from_col}) -> ({move.to_row}, {move.to_col})")
        state.apply_move(move)
        print(f"New turn: {state.current_turn}")
        
    tensor = state.to_tensor()
    print(f"Tensor shape: {tensor.shape}")
    print(f"Piece at (5, 5): {state.get_piece(5, 5)}")
    print(f"Legal moves mask sum: {state.get_legal_moves_mask().sum()}")

if __name__ == "__main__":
    test_bindings()
