import sys
sys.path.insert(0, 'build/Release')
import alphatafl_engine as engine

# Try CPU first
print("Testing CPU inference...")
inf_cpu = engine.InferenceEngine("models/alphatafl_scripted.pt", "cpu")
states = [engine.GameState() for _ in range(64)]
import time
t0 = time.time()
inf_cpu.evaluate(states)
t1 = time.time()
print(f"CPU batch: {(t1-t0)*1000:.2f}ms")

# Try CUDA
print("\nTesting CUDA inference...")
try:
    inf_cuda = engine.InferenceEngine("models/alphatafl_scripted.pt", "cuda")
    t0 = time.time()
    inf_cuda.evaluate(states)
    t1 = time.time()
    print(f"CUDA batch: {(t1-t0)*1000:.2f}ms")
except Exception as e:
    print(f"CUDA failed: {e}")
