import os
import sys
import time
import torch
import numpy as np
from queue import Empty

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.network import AlphaTaflNet

def inference_server_main(inference_queue, response_queues, model_path, stop_event):
    """
    GPU Inference Server main loop.
    
    Args:
        inference_queue: mp.Queue receiving (worker_id, request_id, states_tensor)
        response_queues: list[mp.Queue] indexed by worker_id
        model_path: path to current_best.pt
        stop_event: mp.Event to signal shutdown
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlphaTaflNet().to(device).eval()
    torch.set_grad_enabled(False)
    
    last_mtime = 0
    if os.path.exists(model_path):
        last_mtime = os.path.getmtime(model_path)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[InferenceServer] Loaded model from {model_path} on {device}")
        except Exception as e:
            print(f"[InferenceServer] Warning: could not load model: {e}")
    else:
        print(f"[InferenceServer] No model found at {model_path}, using random weights")
    
    batch_count = 0
    while not stop_event.is_set():
        try:
            msg = inference_queue.get(timeout=0.1)
        except Empty:
            continue
        
        if msg is None:
            print("[InferenceServer] Received shutdown sentinel")
            break
        
        worker_id, request_id, states_tensor = msg
        batch = [states_tensor]
        batch_info = [(worker_id, request_id, states_tensor.shape[0])]
        
        # Collect more states with 50ms timeout
        deadline = time.time() + 0.050
        while len(batch) < 256 and time.time() < deadline and not stop_event.is_set():
            try:
                msg = inference_queue.get(timeout=max(0.0, deadline - time.time()))
                if msg is None:
                    break
                wid, rid, tensor = msg
                batch.append(tensor)
                batch_info.append((wid, rid, tensor.shape[0]))
            except Empty:
                break
        
        if stop_event.is_set():
            break
        
        # Concatenate and move to GPU
        try:
            batch_tensor = torch.cat(batch, dim=0).to(device)
            with torch.inference_mode():
                policy_logits, values = model(batch_tensor)
                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                values = values.squeeze(1).cpu().numpy()
        except Exception as e:
            print(f"[InferenceServer] Forward error: {e}")
            total_states = sum(n for _, _, n in batch_info)
            policies = np.zeros((total_states, 4840), dtype=np.float32)
            values = np.zeros(total_states, dtype=np.float32)
        
        batch_count += 1
        
        # K5: Hot-reload model if file changed
        if batch_count % 10 == 0 and os.path.exists(model_path):
            current_mtime = os.path.getmtime(model_path)
            if current_mtime != last_mtime:
                try:
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    last_mtime = current_mtime
                    print(f"[InferenceServer] Hot-reloaded model from {model_path}")
                except Exception as e:
                    print(f"[InferenceServer] Hot-reload failed: {e}")
        
        # Route results back: each response contains the exact number of states
        # that were in the original request
        offset = 0
        for wid, rid, n_states in batch_info:
            if wid < len(response_queues):
                resp_policies = policies[offset:offset + n_states]
                resp_values = values[offset:offset + n_states]
                response_queues[wid].put((rid, resp_policies, resp_values))
                offset += n_states
    
    print("[InferenceServer] Shutting down")
