import torch
from src.model.network import AlphaTaflNet

def test_network():
    model = AlphaTaflNet()
    # Batch size 2, 14 channels, 11x11 board
    dummy_input = torch.randn(2, 14, 11, 11)
    
    p, v = model(dummy_input)
    
    print(f"Policy shape: {p.shape}") # Expected: (2, 4840)
    print(f"Value shape: {v.shape}")   # Expected: (2, 1)
    
    assert p.shape == (2, 4840)
    assert v.shape == (2, 1)
    print("Network test passed!")

if __name__ == "__main__":
    test_network()
