"""
Debug C++ inference without importing torch in Python.
"""
import os
import sys
import time

sys.path.insert(0, 'build/Release')

import alphatafl_engine as engine

print("Loading C++ inference engine...")
inf = engine.InferenceEngine("models/alphatafl_scripted.pt", "cuda")

states = [engine.GameState() for _ in range(64)]

print("Warmup...")
inf.evaluate(states)
print("Done")

print("\n--- 50 batches of 64 states ---")
t0 = time.time()
for i in range(50):
    policies, values = inf.evaluate(states)
t1 = time.time()
print(f"Total: {t1-t0:.3f}s")
print(f"Per batch: {(t1-t0)/50*1000:.2f}ms")
print(f"Per state: {(t1-t0)/50/64*1000:.2f}ms")
