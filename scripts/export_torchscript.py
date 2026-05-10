"""
Export AlphaTaflNet to TorchScript for C++ inference.

Usage:
    python scripts/export_torchscript.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from src.model.network import AlphaTaflNet


def main():
    model_path = "models/current_best.pt"
    script_path = "models/alphatafl_scripted.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AlphaTaflNet().to(device).eval()

    # Load weights if available
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights from {model_path}")
    else:
        print(f"No weights found at {model_path}, using random init")

    # Create sample input for tracing
    # Use eval mode and no grad
    model.eval()
    example_input = torch.randn(1, 14, 11, 11, device=device)

    # Trace the model
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)

    # Verify outputs match
    print("Verifying traced model outputs...")
    with torch.no_grad():
        orig_policy, orig_value = model(example_input)
        script_policy, script_value = traced(example_input)

    policy_diff = torch.abs(orig_policy - script_policy).max().item()
    value_diff = torch.abs(orig_value - script_value).max().item()

    print(f"Policy max diff: {policy_diff:.2e}")
    print(f"Value max diff:  {value_diff:.2e}")

    if policy_diff < 1e-5 and value_diff < 1e-5:
        print("[OK] Output verification PASSED")
    else:
        print("[FAIL] Output verification FAILED")
        return 1

    # Save scripted model
    os.makedirs("models", exist_ok=True)
    traced.save(script_path)
    print(f"Saved TorchScript model to {script_path}")

    # Also test batched inference
    batch_input = torch.randn(64, 14, 11, 11, device=device)
    with torch.no_grad():
        batch_policy, batch_value = traced(batch_input)
    print(f"Batch inference test: policy shape={batch_policy.shape}, value shape={batch_value.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
