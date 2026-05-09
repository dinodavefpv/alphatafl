"""
Helper script for Phase 5 end-to-end test.
Run via: python tests/python/_phase5_e2e_helper.py
"""
import os
import sys
import time

os.environ['ALPHATAFL_INFERENCE_TIMEOUT_MS'] = '1'
os.environ['ALPHATAFL_INFERENCE_MAX_BATCH'] = '64'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'build', 'Release'))
sys.path.insert(0, PROJECT_ROOT)

from src.training.inference_server import inference_server_main
from self_play import self_play_game_batched
import torch
import multiprocessing as mp


def main():
    inference_queue = mp.Queue()
    response_queue = mp.Queue()
    replay_queue = mp.Queue()

    stop_event = mp.Event()
    proc = mp.Process(
        target=inference_server_main,
        args=(inference_queue, [response_queue], "models/current_best.pt", stop_event),
        daemon=False
    )
    proc.start()
    time.sleep(2)

    try:
        print("[E2E] Starting self_play_game_batched...")
        t0 = time.time()
        history = self_play_game_batched(
            worker_id=0,
            num_simulations=5,
            batch_size=4,
            inference_queue=inference_queue,
            response_queue=response_queue,
            replay_queue=replay_queue,
            verbose=False
        )
        t1 = time.time()
        print(f"[E2E] Game finished in {t1-t0:.2f}s, {len(history)} states")
        assert len(history) > 0
        print("PASS")
    finally:
        print("[E2E] Shutting down inference server...")
        stop_event.set()
        try:
            inference_queue.put(None)
        except Exception:
            pass
        proc.join(timeout=10)
        if proc.is_alive():
            print("[E2E] Terminating inference server...")
            proc.terminate()
            proc.join(timeout=5)
        print("[E2E] Cleanup done")


if __name__ == '__main__':
    main()
    # Force exit to avoid PyTorch CUDA cleanup hangs on Windows spawn
    sys.exit(0)
