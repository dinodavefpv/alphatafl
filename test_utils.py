import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'build', 'Release'))

import alphatafl_engine as engine
from src.training.utils import state_to_tensor, get_legal_moves_mask

def test_utils():
    state = engine.GameState()
    
    tensor = state_to_tensor(state)
    print(f"Tensor shape: {tensor.shape}")
    
    mask = get_legal_moves_mask(state)
    print(f"Mask sum: {mask.sum()}")
    
    # Check some values
    # Initial state should have 116 legal moves
    assert mask.sum() == 116
    
    # Check king position in channel 2
    assert tensor[2, 5, 5] == 1.0
    
    print("Utils test passed!")

if __name__ == "__main__":
    test_utils()
