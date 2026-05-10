"""
Debug benchmark for C++ inference performance.
"""
import os
import sys
import time

sys.path.insert(0, 'build/Release')
sys.path.insert(0, '.')

import alphatafl_engine as engine
import torch

print("Loading C++ inference engine...")
inf = engine.InferenceEngine("models/alphatafl_scripted.pt", "cuda")

# Create states once
states = [engine.GameState() for _ in range(64)]

print("\n--- Warmup ---")
inf.evaluate(states)
print("Warmup done")

print("\n--- C++ Inference: 50 batches of 64 ---")
t0 = time.time()
for i in range(50):
    policies, values = inf.evaluate(states)
t1 = time.time()
print(f"Total: {t1-t0:.3f}s, per batch: {(t1-t0)/50*1000:.2f}ms")

# Compare with Python model
print("\n--- Python Model: 50 batches of 64 ---")
from src.model.network import AlphaTaflNet
model = AlphaTaflNet().to("cuda").eval()
if os.path.exists("models/current_best.pt"):
    model.load_state_dict(torch.load("models/current_best.pt", map_location="cuda"))

batch_tensor = torch.from_numpy(engine.GameState.batch_to_tensor(states)).to("cuda")

# Warmup
with torch.no_grad():
    _ = model(batch_tensor)

t0 = time.time()
for i in range(50):
    with torch.no_grad():
        _ = model(batch_tensor)
    torch.cuda.synchronize()
t1 = time.time()
print(f"Total: {t1-t0:.3f}s, per batch: {(t1-t0)/50*1000:.2f}ms")

# Compare with Python model + queue round-trip
print("\n--- Python Model + numpy round-trip: 50 batches ---")
t0 = time.time()
for i in range(50):
    tensor = torch.from_numpy(engine.GameState.batch_to_tensor(states)).to("cuda")
    with torch.no_grad():
        policy_logits, values = model(tensor)
        policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        values = values.squeeze(1).cpu().numpy()
    torch.cuda.synchronize()
t1 = time.time()
print(f"Total: {t1-t0:.3f}s, per batch: {(t1-t0)/50*1000:.2f}ms")
