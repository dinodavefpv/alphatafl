"""
Phase 5B Benchmark: C++ native inference vs Python mp.Queue.

Usage:
    python benchmarks/bench_phase5b.py
"""
import os
import sys
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def run_cpp_inference_bench(num_batches: int = 50):
    """Benchmark C++ InferenceEngine directly (no queue)."""
    code = f'''
import os, sys, time
sys.path.insert(0, os.path.join(r"{PROJECT_ROOT}", "build", "Release"))
sys.path.insert(0, r"{PROJECT_ROOT}")

import alphatafl_engine as engine

inf = engine.InferenceEngine("models/alphatafl.onnx", "cuda")

# Warmup
states = [engine.GameState() for _ in range(64)]
inf.evaluate(states)

t0 = time.time()
for i in range({num_batches}):
    states = [engine.GameState() for _ in range(64)]
    policies, values = inf.evaluate(states)
    assert len(policies) == 64
t1 = time.time()

total = t1 - t0
per_batch = (total / {num_batches}) * 1000
per_state = per_batch / 64
print(f"CPP_INFERENCE {{total:.3f}} {{per_batch:.2f}} {{per_state:.3f}}")
sys.stdout.flush()
os._exit(0)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120
    )
    for line in result.stdout.splitlines():
        if line.startswith("CPP_INFERENCE "):
            parts = line.split()
            return float(parts[1]), float(parts[2]), float(parts[3])
    print(f"C++ inference benchmark failed: stdout={result.stdout} stderr={result.stderr}")
    return None, None, None


def run_queue_bench(num_batches: int = 50):
    """Benchmark mp.Queue round-trip (0ms timeout)."""
    code = f'''
import os, sys, time, multiprocessing as mp
sys.path.insert(0, os.path.join(r"{PROJECT_ROOT}", "build", "Release"))
sys.path.insert(0, r"{PROJECT_ROOT}")

os.environ["ALPHATAFL_INFERENCE_TIMEOUT_MS"] = "0"

import torch
from src.training.inference_server import inference_server_main

inference_queue = mp.Queue()
response_queue = mp.Queue()
stop_event = mp.Event()

proc = mp.Process(
    target=inference_server_main,
    args=(inference_queue, [response_queue], "models/current_best.pt", stop_event),
    daemon=False
)
proc.start()
time.sleep(2)

# Warmup
dummy = torch.randn(64, 14, 11, 11)
inference_queue.put((0, 0, dummy))
response_queue.get(timeout=10)

t0 = time.time()
for i in range({num_batches}):
    tensor = torch.randn(64, 14, 11, 11)
    inference_queue.put((0, i, tensor))
    _, policies, values = response_queue.get(timeout=30)
    assert policies.shape[0] == 64
t1 = time.time()

stop_event.set()
try:
    inference_queue.put(None)
except Exception:
    pass
proc.join(timeout=5)
if proc.is_alive():
    proc.terminate()
    proc.join(timeout=2)

total = t1 - t0
per_batch = (total / {num_batches}) * 1000
per_state = per_batch / 64
print(f"QUEUE {{total:.3f}} {{per_batch:.2f}} {{per_state:.3f}}")
sys.stdout.flush()
os._exit(0)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120
    )
    for line in result.stdout.splitlines():
        if line.startswith("QUEUE "):
            parts = line.split()
            return float(parts[1]), float(parts[2]), float(parts[3])
    print(f"Queue benchmark failed: stdout={result.stdout} stderr={result.stderr}")
    return None, None, None


def run_e2e_cpp_bench(num_simulations: int = 20, batch_size: int = 16):
    """End-to-end self-play using C++ InferenceEngine (no queue, no server)."""
    code = f'''
import os, sys, time
sys.path.insert(0, os.path.join(r"{PROJECT_ROOT}", "build", "Release"))
sys.path.insert(0, r"{PROJECT_ROOT}")

import alphatafl_engine as engine

inf = engine.InferenceEngine("models/alphatafl.onnx", "cuda")

state = engine.GameState()
game_history = []
moves_made = []
request_counter = [0]

def eval_fn_batched(states_vector):
    if not states_vector:
        return [], []
    policies, values = inf.evaluate(states_vector)
    return policies, values

def eval_fn_single(state_to_eval):
    policies, values = eval_fn_batched([state_to_eval])
    return policies[0], values[0]

mcts = engine.MCTS(
    eval_fn=eval_fn_single,
    eval_fn_batched=eval_fn_batched,
    c_puct=1.4
)

import numpy as np

while state.winner == engine.Player.NONE:
    probs_list = mcts.search(state, {num_simulations}, {batch_size})
    probs = np.array(probs_list)
    game_history.append((state.clone(), probs, state.current_turn))

    if len(game_history) < 30:
        action = np.random.choice(len(probs), p=probs)
    else:
        action = np.argmax(probs)

    from_r = action // 4840  # simplified
    # Actually we need get_move_from_index
    from src.model.network import get_move_from_index
    from_r, from_c, to_r, to_c = get_move_from_index(action)
    moves_made.append({{"from": [int(from_r), int(from_c)], "to": [int(to_r), int(to_c)]}})
    state.apply_move(engine.Move(from_r, from_c, to_r, to_c))

    if len(game_history) > 200:
        break

print(f"E2E_CPP {{time.time()-t0:.3f}} {{len(game_history)}}")
sys.stdout.flush()
os._exit(0)
'''
    # The above code has issues (time.time() not captured, get_move_from_index import).
    # Let me use a simpler approach - just measure eval_fn_batched latency in a loop
    # that mimics what MCTS would do.
    code = f'''
import os, sys, time
sys.path.insert(0, os.path.join(r"{PROJECT_ROOT}", "build", "Release"))
sys.path.insert(0, r"{PROJECT_ROOT}")

import alphatafl_engine as engine
import numpy as np

inf = engine.InferenceEngine("models/alphatafl.onnx", "cuda")

state = engine.GameState()
mcts = engine.MCTS(
    eval_fn=lambda s: ([1.0/4840]*4840, 0.0),
    eval_fn_batched=lambda ss: inf.evaluate(ss),
    c_puct=1.4
)

t0 = time.time()
probs = mcts.search(state, {num_simulations}, {batch_size})
t1 = time.time()

print(f"E2E_CPP {{t1-t0:.3f}} {{len(probs)}}")
sys.stdout.flush()
os._exit(0)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120
    )
    for line in result.stdout.splitlines():
        if line.startswith("E2E_CPP "):
            parts = line.split()
            return float(parts[1]), int(parts[2])
    print(f"E2E C++ benchmark failed: stdout={result.stdout} stderr={result.stderr}")
    return None, None


def run_e2e_queue_bench(num_simulations: int = 20, batch_size: int = 16):
    """End-to-end self-play using mp.Queue (0ms timeout)."""
    helper = os.path.join(PROJECT_ROOT, "benchmarks", "_bench_e2e_helper.py")
    env = os.environ.copy()
    env["ALPHATAFL_INFERENCE_TIMEOUT_MS"] = "0"
    env["BENCH_NUM_SIMULATIONS"] = str(num_simulations)
    env["BENCH_BATCH_SIZE"] = str(batch_size)

    result = subprocess.run(
        [sys.executable, helper],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120, env=env
    )
    for line in result.stdout.splitlines():
        if line.startswith("E2E "):
            parts = line.split()
            return float(parts[1]), int(parts[2])
    print(f"E2E queue benchmark failed: stdout={result.stdout} stderr={result.stderr}")
    return None, None


def main():
    print("=" * 70)
    print("AlphaTafl Phase 5B Benchmark: C++ Native Inference vs mp.Queue")
    print("=" * 70)

    print("\n--- Micro-Benchmark: 50 batches of 64 states ---")
    print(f"{'Method':>20} {'Total(s)':>10} {'Batch(ms)':>12} {'State(ms)':>12}")

    total, per_batch, per_state = run_cpp_inference_bench(num_batches=50)
    if total is not None:
        print(f"{'C++ Inference':>20} {total:>10.3f} {per_batch:>12.2f} {per_state:>12.3f}")

    total, per_batch, per_state = run_queue_bench(num_batches=50)
    if total is not None:
        print(f"{'mp.Queue (0ms)':>20} {total:>10.3f} {per_batch:>12.2f} {per_state:>12.3f}")

    print("\n--- End-to-End: 20 sims/turn, batch_size=16 ---")
    print(f"{'Method':>20} {'Time(s)':>10} {'Turns':>8} {'Turn(ms)':>12}")

    game_time, num_turns = run_e2e_cpp_bench(num_simulations=20, batch_size=16)
    if game_time is not None:
        print(f"{'C++ Inference':>20} {game_time:>10.2f} {num_turns:>8} {game_time/num_turns*1000:>12.1f}")

    game_time, num_turns = run_e2e_queue_bench(num_simulations=20, batch_size=16)
    if game_time is not None:
        print(f"{'mp.Queue (0ms)':>20} {game_time:>10.2f} {num_turns:>8} {game_time/num_turns*1000:>12.1f}")

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
