"""
Multi-worker benchmark helper. Spawned via bench_multi_worker.py
to avoid Windows CUDA cleanup hangs (uses os._exit on completion).
"""
import os
import sys
import time
import signal
import threading
import multiprocessing as mp

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'Release'))
sys.path.insert(0, PROJECT_ROOT)

import torch
from src.training.inference_server import inference_server_main


def drain_replay(replay_queue, stop_event):
    """Background thread that drains the replay queue so workers don't block."""
    while not stop_event.is_set():
        try:
            replay_queue.get(timeout=0.5)
        except Exception:
            pass


def bench_worker(worker_id, num_simulations, batch_size,
                 inference_queue, response_queue, replay_queue,
                 results_queue, stop_flag):
    """Worker process: run self-play games in a loop until stopped."""
    import alphatafl_engine as engine
    from self_play import self_play_game_batched

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)

    # Drain any stale responses in the queue from previous runs
    while True:
        try:
            response_queue.get_nowait()
        except Exception:
            break

    games = 0
    states = 0

    while stop_flag.value == 0:
        try:
            history = self_play_game_batched(
                worker_id=worker_id,
                num_simulations=num_simulations,
                batch_size=batch_size,
                inference_queue=inference_queue,
                response_queue=response_queue,
                replay_queue=replay_queue,
                verbose=False
            )
            if history:
                games += 1
                states += len(history)
        except Exception as e:
            print(f"[W{worker_id}] Error: {e}", file=sys.stderr)
            break

    results_queue.put((worker_id, games, states))


def main():
    num_workers = int(os.environ['BENCH_NUM_WORKERS'])
    num_simulations = int(os.environ.get('BENCH_NUM_SIMULATIONS', '50'))
    batch_size = int(os.environ.get('BENCH_BATCH_SIZE', '16'))
    duration = int(os.environ.get('BENCH_DURATION', '60'))
    timeout_ms = os.environ.get('ALPHATAFL_INFERENCE_TIMEOUT_MS', '5')

    # Create queues (must be in parent before spawn on Windows)
    inference_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(num_workers)]
    replay_queue = mp.Queue(maxsize=50000)
    results_queue = mp.Queue()

    # Background drainer for replay queue (prevents workers from blocking on put)
    drain_stop = threading.Event()
    drain_thread = threading.Thread(target=drain_replay, args=(replay_queue, drain_stop), daemon=True)
    drain_thread.start()

    # Create shared memory regions if enabled
    use_shm = os.environ.get("ALPHATAFL_SHM", "0") == "1"
    shm_regions = []
    if use_shm:
        try:
            from multiprocessing import shared_memory
            max_batch = int(os.environ.get("ALPHATAFL_INFERENCE_MAX_BATCH", "256"))
            input_size = max_batch * 14 * 11 * 11 * 4
            policy_size = max_batch * 4840 * 4
            value_size = max_batch * 4
            SHM_PREFIX = "alphatafl_shm"
            for i in range(num_workers):
                shm_in = shared_memory.SharedMemory(
                    create=True, size=input_size, name=f"{SHM_PREFIX}_input_{i}")
                shm_pol = shared_memory.SharedMemory(
                    create=True, size=policy_size, name=f"{SHM_PREFIX}_policy_{i}")
                shm_val = shared_memory.SharedMemory(
                    create=True, size=value_size, name=f"{SHM_PREFIX}_value_{i}")
                shm_regions.extend([shm_in, shm_pol, shm_val])
            print(f"[Helper] Created {num_workers}x SHM regions "
                  f"({(input_size + policy_size + value_size) * num_workers / (1024*1024):.1f} MB)", file=sys.stderr)
        except Exception as e:
            print(f"[Helper] SHM creation failed: {e}", file=sys.stderr)
            use_shm = False

    # Start inference server process
    server_stop_event = mp.Event()
    server_proc = mp.Process(
        target=inference_server_main,
        args=(inference_queue, response_queues, "models/current_best.pt", server_stop_event),
        daemon=False
    )
    server_proc.start()
    print(f"[Helper] Inference server PID={server_proc.pid}", file=sys.stderr)
    time.sleep(3)  # model load + warmup

    # Shared stop flag
    stop_flag = mp.Value('i', 0)

    # Start workers
    worker_procs = []
    for i in range(num_workers):
        p = mp.Process(
            target=bench_worker,
            args=(i, num_simulations, batch_size,
                  inference_queue, response_queues[i], replay_queue,
                  results_queue, stop_flag),
            daemon=False
        )
        p.start()
        worker_procs.append(p)

    # Small delay to let all workers initialize, then start timing
    time.sleep(0.5)
    t_start = time.perf_counter()
    print(f"[Helper] {num_workers} workers running for {duration}s (timeout={timeout_ms}ms)", file=sys.stderr)

    time.sleep(duration)

    # Signal stop
    stop_flag.value = 1

    # Drain period: let workers finish their current game
    drain_time = max(10, min(30, duration // 2))
    time.sleep(drain_time)

    # Terminate any remaining workers
    for p in worker_procs:
        if p.is_alive():
            p.terminate()
        p.join(timeout=3)

    t_end = time.perf_counter()
    actual_duration = t_end - t_start

    # Collect results from queue
    total_games = 0
    total_states = 0
    worker_results = 0
    while worker_results < num_workers:
        try:
            _, g, s = results_queue.get(timeout=2)
            total_games += g
            total_states += s
            worker_results += 1
        except Exception:
            break

    # Shut down inference server
    server_stop_event.set()
    try:
        inference_queue.put(None)
    except Exception:
        pass
    server_proc.join(timeout=5)
    if server_proc.is_alive():
        server_proc.terminate()
        server_proc.join(timeout=2)

    # Clean up shared memory regions
    for shm in shm_regions:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass

    drain_stop.set()
    drain_thread.join(timeout=1)

    shm_flag = "1" if use_shm else "0"
    print(f"RESULT {total_games} {total_states} {actual_duration:.1f} {num_workers} {timeout_ms} {shm_flag}")
    sys.stdout.flush()
    # Force exit to avoid Windows CUDA cleanup hang
    os._exit(0)


if __name__ == '__main__':
    main()
