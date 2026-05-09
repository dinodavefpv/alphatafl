"""
Helper for Phase 5 end-to-end benchmark. Run via bench_phase5.py.
"""
import os
import sys
import time
import threading
import multiprocessing as mp

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'Release'))
sys.path.insert(0, PROJECT_ROOT)

from src.training.inference_server import inference_server_main
from self_play import self_play_game_batched


def main():
    timeout_ms = int(os.environ.get("ALPHATAFL_INFERENCE_TIMEOUT_MS", "5"))
    num_simulations = int(os.environ.get("BENCH_NUM_SIMULATIONS", "20"))
    batch_size = int(os.environ.get("BENCH_BATCH_SIZE", "16"))

    inference_queue = mp.Queue()
    response_queue = mp.Queue()
    replay_queue = mp.Queue()
    stop_event = threading.Event()

    server_thread = threading.Thread(
        target=inference_server_main,
        args=(inference_queue, [response_queue], "models/current_best.pt", stop_event),
        daemon=True
    )
    server_thread.start()
    time.sleep(2)

    t0 = time.time()
    history = self_play_game_batched(
        worker_id=0,
        num_simulations=num_simulations,
        batch_size=batch_size,
        inference_queue=inference_queue,
        response_queue=response_queue,
        replay_queue=replay_queue,
        verbose=False
    )
    t1 = time.time()

    stop_event.set()
    try:
        inference_queue.put(None)
    except Exception:
        pass
    server_thread.join(timeout=5)

    print(f"E2E {t1-t0:.3f} {len(history)}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
    # Force exit to avoid Windows CUDA cleanup hang
    os._exit(0)
