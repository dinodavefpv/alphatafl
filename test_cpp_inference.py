"""
Quick test of C++ InferenceEngine.
"""
import sys
import os
sys.path.insert(0, 'build/Release')

import torch
import numpy as np
from src.model.network import AlphaTaflNet
import alphatafl_engine as engine

def test_cpp_inference():
    print("Loading C++ inference engine...")
    inf = engine.InferenceEngine("models/alphatafl_scripted.pt", "cuda")

    print("Creating test states...")
    states = [engine.GameState() for _ in range(4)]

    print("Running C++ inference...")
    policies, values = inf.evaluate(states)

    print(f"Policies: {len(policies)} states, each {len(policies[0])} floats")
    print(f"Values: {len(values)} floats")

    # Verify probabilities sum to ~1.0
    for i, p in enumerate(policies):
        s = sum(p)
        print(f"  State {i}: policy sum={s:.6f}, value={values[i]:.4f}")
        assert abs(s - 1.0) < 0.01, f"Policy should sum to 1.0, got {s}"

    # Compare with Python model
    print("\nComparing with Python model...")
    device = "cuda"
    model = AlphaTaflNet().to(device).eval()
    model_path = "models/current_best.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    batch_tensor = torch.from_numpy(engine.GameState.batch_to_tensor(states)).to(device)
    with torch.no_grad():
        py_policy_logits, py_values = model(batch_tensor)
        py_policies = torch.softmax(py_policy_logits, dim=1).cpu().numpy()
        py_values = py_values.squeeze(1).cpu().numpy()

    # Apply legal masks in Python for fair comparison
    for i, s in enumerate(states):
        mask = s.get_legal_moves_mask()
        py_p = py_policies[i] * mask
        py_p /= py_p.sum()

        cpp_p = np.array(policies[i])
        diff = np.abs(py_p - cpp_p).max()
        v_diff = abs(py_values[i] - values[i])
        print(f"  State {i}: policy max diff={diff:.2e}, value diff={v_diff:.2e}")

    print("\n[OK] C++ inference test PASSED")

if __name__ == "__main__":
    test_cpp_inference()
