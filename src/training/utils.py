import torch
import numpy as np

def state_to_tensor(state):
    """
    Converts a GameState object to a 14-channel tensor.
    Channels:
    0-2: Attackers, Defenders, King (Current Turn, T)
    3-5: Attackers, Defenders, King (T-1)
    6-8: Attackers, Defenders, King (T-2)
    9-11: Attackers, Defenders, King (T-3)
    12: Turn (1 if Attacker, 0 if Defender)
    13: Restricted squares
    """
    board_size = 11
    tensor = np.zeros((14, board_size, board_size), dtype=np.float32)
    
    # Fill channels 0-11
    for depth in range(4):
        hist_board = state.get_historical_board(depth)
        for r in range(board_size):
            for c in range(board_size):
                p = hist_board[r][c]
                if p.name == "ATTACKER":
                    tensor[depth * 3 + 0, r, c] = 1.0
                elif p.name == "DEFENDER":
                    tensor[depth * 3 + 1, r, c] = 1.0
                elif p.name == "KING":
                    tensor[depth * 3 + 2, r, c] = 1.0
                
    # Fill channel 12 (Turn)
    if state.current_turn.name == "ATTACKER":
        tensor[12, :, :] = 1.0
    else:
        tensor[12, :, :] = 0.0
        
    # Fill channel 13 (Restricted squares)
    for r, c in [(0,0), (0,10), (10,0), (10,10), (5,5)]:
        tensor[13, r, c] = 1.0
        
    return torch.from_numpy(tensor)

def get_legal_moves_mask(state, board_size=11):
    """
    Returns a mask (1.0 for legal, 0.0 for illegal) of shape (4840,)
    """
    from src.model.network import get_action_index
    mask = np.zeros(board_size * board_size * 40, dtype=np.float32)
    legal_moves = state.get_legal_moves()
    
    for m in legal_moves:
        idx = get_action_index(m.from_row, m.from_col, m.to_row, m.to_col, board_size)
        mask[idx] = 1.0
        
    return torch.from_numpy(mask)
