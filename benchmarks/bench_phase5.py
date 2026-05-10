"""
Phase 5 IPC Benchmark: measure mp.Queue round-trip latency with different
timeout settings, and end-to-end self-play throughput.

Usage:
    python benchmarks/bench_phase5.py
"""
import os
import sys
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def run_latency_bench(timeout_ms: int, num_batches: int = 50, use_compile: bool = False):
    """Run queue latency benchmark in a subprocess to avoid Windows cleanup hangs."""
    compile_flag = "1" if use_compile else "0"
    code = f'''
import os, sys, time, multiprocessing as mp
sys.path.insert(0, os.path.join(r"{PROJECT_ROOT}", "build", "Release"))
sys.path.insert(0, r"{PROJECT_ROOT}")

os.environ["ALPHATAFL_INFERENCE_TIMEOUT_MS"] = "{timeout_ms}"
os.environ["ALPHATAFL_INFERENCE_MAX_BATCH"] = "256"
os.environ["ALPHATAFL_COMPILE"] = "{compile_flag}"

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
print(f"LATENCY {{total:.3f}} {{per_batch:.2f}} {{per_state:.3f}}")
sys.stdout.flush()
os._exit(0)
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120,
        env={**os.environ, "ALPHATAFL_COMPILE": compile_flag}
    )
    for line in result.stdout.splitlines():
        if line.startswith("LATENCY "):
            parts = line.split()
            return float(parts[1]), float(parts[2]), float(parts[3])
    print(f"Latency benchmark failed: stdout={result.stdout} stderr={result.stderr}")
    return None, None, None


def run_e2e_bench(timeout_ms: int, num_simulations: int = 20, batch_size: int = 16, use_compile: bool = False):
    """Run end-to-end self-play benchmark in a subprocess."""
    helper = os.path.join(PROJECT_ROOT, "benchmarks", "_bench_e2e_helper.py")
    env = os.environ.copy()
    env["ALPHATAFL_INFERENCE_TIMEOUT_MS"] = str(timeout_ms)
    env["ALPHATAFL_INFERENCE_MAX_BATCH"] = "256"
    env["BENCH_NUM_SIMULATIONS"] = str(num_simulations)
    env["BENCH_BATCH_SIZE"] = str(batch_size)
    env["ALPHATAFL_COMPILE"] = "1" if use_compile else "0"

    result = subprocess.run(
        [sys.executable, helper],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=180, env=env
    )
    for line in result.stdout.splitlines():
        if line.startswith("E2E "):
            parts = line.split()
            return float(parts[1]), int(parts[2])
    print(f"E2E benchmark failed: stdout={result.stdout} stderr={result.stderr}")
    return None, None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on inference server")
    args = parser.parse_args()

    compile_label = " (torch.compile ON)" if args.compile else ""
    print("=" * 70)
    print(f"AlphaTafl Phase 5 IPC Benchmark{compile_label}")
    print("=" * 70)

    print("\n--- Queue Round-Trip Latency (50 batches of 64 states) ---")
    print(f"{'Timeout':>10} {'Total(s)':>10} {'Batch(ms)':>12} {'State(ms)':>12}")
    for timeout_ms in [50, 5, 0]:
        total, per_batch, per_state = run_latency_bench(timeout_ms, num_batches=50, use_compile=args.compile)
        if total is not None:
            print(f"{timeout_ms:>10}ms {total:>10.3f} {per_batch:>12.2f} {per_state:>12.3f}")

    print("\n--- End-to-End Self-Play (20 sims/turn, batch_size=16) ---")
    print(f"{'Timeout':>10} {'Game(s)':>10} {'Turns':>8} {'Turn(ms)':>12}")
    for timeout_ms in [50, 5, 0]:
        game_time, num_turns = run_e2e_bench(timeout_ms, num_simulations=20, batch_size=16, use_compile=args.compile)
        if game_time is not None:
            print(f"{timeout_ms:>10}ms {game_time:>10.2f} {num_turns:>8} {game_time/num_turns*1000:>12.1f}")

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
