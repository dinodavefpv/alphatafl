"""
Multi-worker ONNX standalone benchmark.
Measures concurrent self-play throughput using native C++ ONNX inference.

Usage:
    python benchmarks/bench_multi_worker_onnx.py [--workers 1,2,4,8,12,20] [--duration 30] [--sims 50] [--batch-size 16]
"""
import os
import sys
import time
import argparse
import multiprocessing as mp
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'Release'))
sys.path.insert(0, PROJECT_ROOT)

import alphatafl_engine as engine
from src.model.network import get_move_from_index


def onnx_worker(worker_id, num_simulations, batch_size, results_queue, stop_flag):
    # Enable CUDA for each process. On Windows/CUDA, multiple processes can share the GPU context.
    try:
        inf = engine.InferenceEngine("models/alphatafl.onnx", "cuda")
    except Exception as e:
        print(f"[Worker {worker_id}] Failed to load ONNX: {e}", file=sys.stderr)
        return

    # Define MCTS callbacks
    def eval_fn_batched(states_vector):
        if not states_vector:
            return [], []
        policies, values = inf.evaluate(states_vector)
        return policies, values

    def eval_fn_single(state_to_eval):
        policy, value = inf.evaluate_single(state_to_eval)
        return policy, value

    games = 0
    states = 0

    while stop_flag.value == 0:
        state = engine.GameState()
        turns = 0
        game_history = []

        mcts = engine.MCTS(
            eval_fn=eval_fn_single,
            eval_fn_batched=eval_fn_batched,
            c_puct=1.4,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25
        )

        while state.winner == engine.Player.NONE and stop_flag.value == 0:
            probs_list = mcts.search(state, num_simulations, batch_size)
            probs = np.array(probs_list, dtype=np.float64)
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
            else:
                probs = np.full(len(probs), 1.0 / len(probs))

            if turns < 30:
                action = np.random.choice(len(probs), p=probs)
            else:
                action = np.argmax(probs)

            from_r, from_c, to_r, to_c = get_move_from_index(action)
            state.apply_move(engine.Move(from_r, from_c, to_r, to_c))
            game_history.append(action)
            turns += 1

            if turns >= 200:
                break

        if stop_flag.value == 0 or len(game_history) > 10:
            games += 1
            states += turns

    results_queue.put((worker_id, games, states))
    # Ensure queue data is flushed before exiting
    time.sleep(0.5)
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="ONNX Multi-worker self-play benchmark")
    parser.add_argument('--workers', type=str, default='1,2,4,8,12,20',
                        help='Comma-separated worker counts to test')
    parser.add_argument('--duration', type=int, default=30,
                        help='Benchmark duration per config (seconds)')
    parser.add_argument('--sims', type=int, default=50,
                        help='MCTS simulations per turn')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='MCTS batch size')
    args = parser.parse_args()

    worker_counts = [int(w) for w in args.workers.split(',')]

    print("=" * 75)
    print("AlphaTafl ONNX Standalone C++ Multi-Worker Benchmark")
    print("=" * 75)
    print(f"Config: {args.sims} sims/turn, batch_size={args.batch_size}, {args.duration}s/window")
    print(f"Workers to test: {worker_counts}")
    print("=" * 75)

    for nw in worker_counts:
        print(f"\n[{nw} workers] Starting...", flush=True)

        results_queue = mp.Queue()
        stop_flag = mp.Value('i', 0)

        procs = []
        for i in range(nw):
            p = mp.Process(
                target=onnx_worker,
                args=(i, args.sims, args.batch_size, results_queue, stop_flag),
                daemon=False
            )
            p.start()
            procs.append(p)

        time.sleep(0.5)  # Let them start
        t_start = time.perf_counter()

        time.sleep(args.duration)
        stop_flag.value = 1

        # Let workers wrap up
        time.sleep(5)

        # Terminate if still alive
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=2)

        t_end = time.perf_counter()
        actual_duration = t_end - t_start

        # Collect results
        total_games = 0
        total_states = 0
        collected = 0
        while collected < nw:
            try:
                _, g, s = results_queue.get(timeout=1)
                total_games += g
                total_states += s
                collected += 1
            except Exception:
                break

        states_per_sec = total_states / actual_duration if actual_duration > 0 else 0
        games_per_hour = (total_games / actual_duration) * 3600 if actual_duration > 0 else 0

        print(f"Results for {nw} workers:")
        print(f"  Total Games:      {total_games}")
        print(f"  Total States:     {total_states}")
        print(f"  Duration:         {actual_duration:.2f} s")
        print(f"  Throughput:       {states_per_sec:.1f} states/sec")
        print(f"  Games/Hour:       {games_per_hour:.1f}")


if __name__ == '__main__':
    mp.freeze_support()
    main()
