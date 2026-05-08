import torch
import numpy as np

def state_to_tensor(state):
    """Converts a GameState to a (14, 11, 11) tensor using C++ zero-copy to_tensor()."""
    return torch.from_numpy(state.to_tensor()).clone()

def get_legal_moves_mask(state, board_size=11):
    """Returns a mask (1.0 for legal, 0.0 for illegal) of shape (4840,) using C++ method."""
    return torch.from_numpy(state.get_legal_moves_mask()).clone()
