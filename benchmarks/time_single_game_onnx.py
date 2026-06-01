"""
ONNX Standalone C++ Inference E2E Self-Play Benchmark.

Measures actual end-to-end game time using the native C++ InferenceEngine
with ONNX Runtime on CUDA. Bypasses the multiprocessing server, queues, and SHM.

Usage:
    python benchmarks/time_single_game_onnx.py [--batch-size 128] [--sims 800]
"""
import os
import sys
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'Release'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import alphatafl_engine as engine
from src.model.network import get_move_from_index


def main():
    parser = argparse.ArgumentParser(description="Time a single ONNX self-play game")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sims', type=int, default=800)
    args = parser.parse_args()

    print("=" * 60)
    print(f"ONNX Standalone C++ E2E Game: {args.sims} sims/turn, batch_size={args.batch_size}")
    print("Backend: C++ ONNX Runtime (CUDA)")
    print("=" * 60)

    print("Loading ONNX model in C++...")
    inf = engine.InferenceEngine("models/alphatafl.onnx", "cuda")
    print("Model loaded successfully.")

    state = engine.GameState()
    turns = 0
    t_start = time.time()

    # Define callbacks
    def eval_fn_batched(states_vector):
        if not states_vector:
            return [], []
        policies, values = inf.evaluate(states_vector)
        # Flatten policies list of lists for C++ compatibility if needed, 
        # but pybind11 handle conversion automatically.
        return policies, values

    def eval_fn_single(state_to_eval):
        policy, value = inf.evaluate_single(state_to_eval)
        return policy, value

    # Create MCTS
    mcts = engine.MCTS(
        eval_fn=eval_fn_single,
        eval_fn_batched=eval_fn_batched,
        c_puct=1.4,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25
    )

    print("\nStarting 800-sim self-play game...")
    t_search_total = 0.0
    
    while state.winner == engine.Player.NONE:
        t0 = time.perf_counter()
        probs_list = mcts.search(state, args.sims, args.batch_size)
        t1 = time.perf_counter()
        
        t_search = t1 - t0
        t_search_total += t_search
        
        probs = np.array(probs_list, dtype=np.float64)
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.full(len(probs), 1.0 / len(probs))

        # Action selection (temperature logic matching self_play.py)
        if turns < 30:
            action = np.random.choice(len(probs), p=probs)
        else:
            action = np.argmax(probs)

        from_r, from_c, to_r, to_c = get_move_from_index(action)
        move = engine.Move(from_r, from_c, to_r, to_c)
        state.apply_move(move)
        
        turns += 1
        print(f"Turn {turns:3d}: MCTS search={t_search * 1000:6.1f} ms | move=({from_r},{from_c})->({to_r},{to_c})")

        if turns >= 200:
            print("Reached turn limit.")
            break

    total_time = time.time() - t_start
    print("=" * 60)
    print(f"Game finished! Winner: {state.winner.name}")
    print(f"Total Game Time:      {total_time:.2f} s")
    print(f"Total MCTS Time:      {t_search_total:.2f} s ({(t_search_total / total_time) * 100:.1f}%)")
    print(f"Total Turns:          {turns}")
    print(f"Average Turn Time:    {(t_search_total / turns) * 1000:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
