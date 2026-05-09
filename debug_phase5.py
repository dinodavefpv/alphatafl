import os
import sys
import time

os.environ['ALPHATAFL_INFERENCE_TIMEOUT_MS'] = '1'
os.environ['ALPHATAFL_INFERENCE_MAX_BATCH'] = '64'

sys.path.insert(0, 'build/Release')
from src.training.inference_server import inference_server_main
from self_play import self_play_game_batched
import torch
import multiprocessing as mp

print("Starting inference server...")
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

print("Running self_play_game_batched...")
try:
    history = self_play_game_batched(
        worker_id=0,
        num_simulations=10,
        batch_size=8,
        inference_queue=inference_queue,
        response_queue=response_queue,
        replay_queue=replay_queue,
        verbose=True
    )
    print(f"PASS: {len(history)} states")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("Shutting down...")
    stop_event.set()
    try:
        inference_queue.put(None)
    except Exception:
        pass
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
    print("Done")
