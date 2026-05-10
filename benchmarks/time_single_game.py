"""
Single-game wall-clock timing for 800-sim self-play.

Measures actual end-to-end game time (start to terminal state) at production
settings: SHM + 1ms timeout + batch_size=128, 800 simulations/turn.

Usage:
    python benchmarks/time_single_game.py [--batch-size 128] [--sims 800]
"""
import os
import sys
import time
import argparse
import multiprocessing as mp
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'Release'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import alphatafl_engine as engine
from src.training.inference_server import inference_server_main
from src.training.utils import state_to_tensor
from src.model.network import get_move_from_index


def drain_replay(replay_queue, stop_event):
    while not stop_event.is_set():
        try:
            replay_queue.get(timeout=0.5)
        except Exception:
            pass


def create_shm_regions(num_workers, max_batch):
    """Create shared memory regions for all workers."""
    from multiprocessing import shared_memory
    SHM_PREFIX = "alphatafl_shm"
    input_size = max_batch * 14 * 11 * 11 * 4
    policy_size = max_batch * 4840 * 4
    value_size = max_batch * 4
    shms = []
    for i in range(num_workers):
        shms.append(shared_memory.SharedMemory(
            create=True, size=input_size, name=f"{SHM_PREFIX}_input_{i}"))
        shms.append(shared_memory.SharedMemory(
            create=True, size=policy_size, name=f"{SHM_PREFIX}_policy_{i}"))
        shms.append(shared_memory.SharedMemory(
            create=True, size=value_size, name=f"{SHM_PREFIX}_value_{i}"))
    return shms


def cleanup_shm(shms):
    for shm in shms:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Time a single self-play game")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sims', type=int, default=800)
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of workers (game runs on worker 0)')
    args = parser.parse_args()

    # Environment
    os.environ['ALPHATAFL_BATCHED'] = '1'
    os.environ['ALPHATAFL_SHM'] = '1'
    os.environ['ALPHATAFL_INFERENCE_TIMEOUT_MS'] = '1'
    os.environ['ALPHATAFL_INFERENCE_MAX_BATCH'] = '2048'

    print("=" * 60)
    print(f"Single-Game Timer: {args.sims} sims/turn, batch_size={args.batch_size}")
    print(f"Transport: SHM, 1ms timeout, {args.num_workers} workers")
    print("=" * 60)

    # Create shared memory
    num_workers = max(args.num_workers, 1)
    shms = create_shm_regions(num_workers, max_batch=2048)
    print(f"SHM regions created ({len(shms)} segments)")

    # Start inference server
    inference_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(num_workers)]
    replay_queue = mp.Queue(maxsize=50000)

    drain_stop = threading.Event()
    drain_thread = threading.Thread(target=drain_replay, args=(replay_queue, drain_stop), daemon=True)
    drain_thread.start()

    server_stop = mp.Event()
    server_proc = mp.Process(
        target=inference_server_main,
        args=(inference_queue, response_queues, "models/current_best.pt", server_stop),
        daemon=False
    )
    server_proc.start()
    print("Inference server starting...")
    time.sleep(3)

    worker_id = 0
    num_simulations = args.sims
    batch_size = args.batch_size

    # Initialize SHM handles in this process
    from multiprocessing import shared_memory
    SHM_PREFIX = "alphatafl_shm"
    shm_input = shared_memory.SharedMemory(name=f"{SHM_PREFIX}_input_{worker_id}")
    shm_policy = shared_memory.SharedMemory(name=f"{SHM_PREFIX}_policy_{worker_id}")
    shm_value = shared_memory.SharedMemory(name=f"{SHM_PREFIX}_value_{worker_id}")

    state = engine.GameState()
    request_counter = [0]

    # Timing accumulators for profiling eval_fn_batched steps
    timing = {
        'batch_to_tensor': 0.0,
        'shm_write': 0.0,
        'queue_put': 0.0,
        'queue_get': 0.0,
        'shm_read': 0.0,
        'list_build': 0.0,
        'masking': 0.0,
        'calls': 0,
    }

    def eval_fn_batched(states_vector):
        if not states_vector:
            return [], []

        n_states = len(states_vector)

        # Step 1: C++ batch_to_tensor
        t0 = time.perf_counter()
        tensor_np = engine.GameState.batch_to_tensor(states_vector)
        t1 = time.perf_counter()

        # Step 2: SHM write (memcpy numpy → shared memory)
        dest = np.ndarray((n_states, 14, 11, 11), dtype=np.float32, buffer=shm_input.buf)
        np.copyto(dest, tensor_np)
        t2 = time.perf_counter()

        # Step 3: mp.Queue.put() — send metadata to server
        req_id = request_counter[0]
        request_counter[0] += 1
        inference_queue.put((worker_id, req_id, n_states))
        t3 = time.perf_counter()

        # Step 4: mp.Queue.get() — await response
        try:
            rid, n_out = response_queues[worker_id].get(timeout=120)
        except Exception as e:
            print(f"Inference timeout: {e}")
            fallback_p = [([1.0 / 4840] * 4840) for _ in states_vector]
            fallback_v = [0.0] * len(states_vector)
            return fallback_p, fallback_v
        t4 = time.perf_counter()

        # Step 5: SHM read (memcpy shared memory → numpy)
        policies = np.ndarray((n_out, 4840), dtype=np.float32, buffer=shm_policy.buf).copy()
        values = np.ndarray(n_out, dtype=np.float32, buffer=shm_value.buf).copy()
        t5 = time.perf_counter()

        # Step 6: Legal masking + Dirichlet noise + list construction
        results_policies = []
        results_values = []
        is_root = (request_counter[0] == 1)

        for i, s in enumerate(states_vector):
            mask = s.get_legal_moves_mask()
            p = policies[i] * mask
            if is_root:
                num_legal = int(mask.sum())
                if num_legal > 0:
                    noise = np.random.dirichlet([0.3] * num_legal)
                    noise_full = np.zeros(4840, dtype=np.float32)
                    noise_full[mask > 0] = noise
                    p = 0.75 * p + 0.25 * noise_full
            p_sum = p.sum()
            if p_sum > 0:
                p /= p_sum
            results_policies.append(p.tolist())
            results_values.append(float(values[i]))

        t6 = time.perf_counter()

        # Accumulate timing
        timing['batch_to_tensor'] += (t1 - t0) * 1000
        timing['shm_write'] += (t2 - t1) * 1000
        timing['queue_put'] += (t3 - t2) * 1000
        timing['queue_get'] += (t4 - t3) * 1000
        timing['shm_read'] += (t5 - t4) * 1000
        timing['list_build'] += (t6 - t5) * 1000
        timing['calls'] += 1

        return results_policies, results_values

    def eval_fn_single(state_to_eval):
        policies, values = eval_fn_batched([state_to_eval])
        return policies[0], values[0]

    mcts = engine.MCTS(
        eval_fn=eval_fn_single,
        eval_fn_batched=eval_fn_batched,
        c_puct=1.4
    )

    print(f"\nStarting 800-sim self-play game...")
    total_turns = 0
    batch_count = 0
    t0 = time.time()

    while state.winner == engine.Player.NONE:
        turn_start = time.time()
        probs_list = mcts.search(state, num_simulations, batch_size)
        turn_time = time.time() - turn_start

        probs = np.array(probs_list)

        if total_turns < 30:
            action = np.random.choice(len(probs), p=probs)
        else:
            action = np.argmax(probs)

        from_r, from_c, to_r, to_c = get_move_from_index(action)
        state.apply_move(engine.Move(from_r, from_c, to_r, to_c))
        total_turns += 1
        batch_count += 1

        if total_turns <= 5 or total_turns % 20 == 0:
            print(f"  Turn {total_turns:>3}: {turn_time:.2f}s "
                  f"({state.current_turn.name} moves ({from_r},{from_c})->({to_r},{to_c}))")

        if total_turns >= 200:
            break

    t1 = time.time()
    total_time = t1 - t0

    # Shutdown
    server_stop.set()
    try:
        inference_queue.put(None)
    except Exception:
        pass
    server_proc.join(timeout=5)
    if server_proc.is_alive():
        server_proc.terminate()
        server_proc.join(timeout=2)

    drain_stop.set()
    drain_thread.join(timeout=1)
    cleanup_shm(shms)

    # Report
    print(f"\n{'='*60}")
    print(f"GAME COMPLETE")
    print(f"{'='*60}")
    print(f"  Winner:     {state.winner.name}")
    print(f"  Turns:      {total_turns}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg/turn:   {total_time/total_turns*1000:.0f}ms")
    print(f"  Sim count:  {args.sims}/turn")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batches:    ~{total_turns * max(1, args.sims // args.batch_size)} (estimated)")
    print(f"  Effective:  {total_turns/total_time:.1f} turns/sec")

    # Per-batch timing breakdown
    if timing['calls'] > 0:
        n = timing['calls']
        total_per_batch = sum(v for k, v in timing.items() if k != 'calls')
        print(f"\n{'='*60}")
        print(f"PER-BATCH TIMING BREAKDOWN ({n} batches)")
        print(f"{'='*60}")
        print(f"  {'Step':<30} {'Total(ms)':>10} {'Per batch(ms)':>14} {'%':>6}")
        print(f"  {'-'*30} {'-'*10} {'-'*14} {'-'*6}")
        for label, key in [
            ('1. C++ batch_to_tensor', 'batch_to_tensor'),
            ('2. SHM memcpy write (input)', 'shm_write'),
            ('3. mp.Queue.put (send)', 'queue_put'),
            ('4. mp.Queue.get (recv)', 'queue_get'),
            ('5. SHM memcpy read (output)', 'shm_read'),
            ('6. Mask + noise + list build', 'list_build'),
        ]:
            total_ms = timing[key]
            per_batch = total_ms / n
            pct = total_ms / total_per_batch * 100
            print(f"  {label:<30} {total_ms:>10.1f} {per_batch:>14.1f} {pct:>5.1f}%")
        print(f"  {'Total':<30} {total_per_batch:>10.1f} {total_per_batch/n:>14.1f} {'100.0%':>6}")
        print(f"\n  Queue round-trip (put + get): {(timing['queue_put'] + timing['queue_get']) / n:.1f}ms/batch")
        print(f"  Python processing (list + mask): {timing['list_build'] / n:.1f}ms/batch")
        print(f"  Data movement (SHM write+read): {(timing['shm_write'] + timing['shm_read']) / n:.1f}ms/batch")


if __name__ == '__main__':
    main()
