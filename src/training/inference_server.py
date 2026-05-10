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

# Shared memory naming convention: alphatafl_shm_{type}_{worker_id}
SHM_PREFIX = "alphatafl_shm"


def _open_shm(name):
    """Open a SharedMemory object by name. Returns None if unavailable."""
    try:
        from multiprocessing import shared_memory
        return shared_memory.SharedMemory(name=name)
    except Exception:
        return None


def _get_msg_tensor(msg, shm_inputs):
    """Extract a CPU torch.Tensor from an inference queue message.

    Supports two transport modes, autodetected:
      - Queue mode:  msg = (worker_id, request_id, torch.Tensor)
      - SHM mode:    msg = (worker_id, request_id, n_states)
        Tensor is read from shared memory alphatafl_shm_input_{worker_id}.
    """
    wid = msg[0]
    data = msg[2]

    if isinstance(data, torch.Tensor):
        return data, data.shape[0]

    # SHM mode: data is n_states (int)
    n_states = int(data)
    if wid not in shm_inputs:
        shm_inputs[wid] = _open_shm(f"{SHM_PREFIX}_input_{wid}")
    shm = shm_inputs[wid]
    if shm is None:
        raise RuntimeError(f"SHM transport enabled but shm_input_{wid} not found")
    arr = np.ndarray((n_states, 14, 11, 11), dtype=np.float32, buffer=shm.buf)
    # Copy out of shared memory into a standalone CPU tensor for GPU transfer
    tensor = torch.from_numpy(arr.copy())
    return tensor, n_states


def _write_shm_results(wid, n_states, policies_slice, values_slice, shm_policies, shm_values):
    """Write inference results to shared memory for a worker."""
    if wid not in shm_policies:
        shm_policies[wid] = _open_shm(f"{SHM_PREFIX}_policy_{wid}")
    if wid not in shm_values:
        shm_values[wid] = _open_shm(f"{SHM_PREFIX}_value_{wid}")

    shm_p = shm_policies[wid]
    shm_v = shm_values[wid]
    if shm_p is None or shm_v is None:
        return False

    policy_arr = np.ndarray((n_states, 4840), dtype=np.float32, buffer=shm_p.buf)
    value_arr = np.ndarray(n_states, dtype=np.float32, buffer=shm_v.buf)
    np.copyto(policy_arr, policies_slice)
    np.copyto(value_arr, values_slice)
    return True


def inference_server_main(inference_queue, response_queues, model_path, stop_event):
    """
    GPU Inference Server main loop.

    Args:
        inference_queue: mp.Queue receiving inference requests.
            Queue mode:  (worker_id, request_id, states_tensor)
            SHM mode:    (worker_id, request_id, n_states)
        response_queues: list[mp.Queue] indexed by worker_id
        model_path: path to current_best.pt
        stop_event: mp.Event to signal shutdown
    """
    timeout_ms = float(os.environ.get("ALPHATAFL_INFERENCE_TIMEOUT_MS", "1"))
    max_batch = int(os.environ.get("ALPHATAFL_INFERENCE_MAX_BATCH", "256"))
    use_shm = os.environ.get("ALPHATAFL_SHM", "0") == "1"

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

    # K8: torch.compile (opt-in via ALPHATAFL_COMPILE=1)
    use_compile = os.environ.get("ALPHATAFL_COMPILE", "0") == "1"
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[InferenceServer] torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"[InferenceServer] torch.compile failed: {e}")

    if use_shm:
        print("[InferenceServer] Shared memory transport enabled")

    # Lazy-init caches for SHM handles
    shm_inputs = {}
    shm_policies = {}
    shm_values = {}

    batch_count = 0
    while not stop_event.is_set():
        try:
            msg = inference_queue.get(timeout=0.001)  # 1ms poll, responsive to single-worker
        except Empty:
            continue

        if msg is None:
            print("[InferenceServer] Received shutdown sentinel")
            break

        wid, rid = msg[0], msg[1]
        tensor, n_states = _get_msg_tensor(msg, shm_inputs)
        batch = [tensor]
        batch_info = [(wid, rid, n_states)]
        use_shm_for_batch = use_shm and not isinstance(msg[2], torch.Tensor)

        # Adaptive batching
        if timeout_ms > 0 and len(batch) < max_batch:
            deadline = time.time() + (timeout_ms / 1000.0)
            while len(batch) < max_batch and time.time() < deadline and not stop_event.is_set():
                try:
                    remaining = max(0.0, deadline - time.time())
                    msg = inference_queue.get(timeout=min(0.001, remaining))
                    if msg is None:
                        break
                    t, n = _get_msg_tensor(msg, shm_inputs)
                    batch.append(t)
                    batch_info.append((msg[0], msg[1], n))
                except Empty:
                    break

        if stop_event.is_set():
            break

        # Forward pass
        try:
            batch_tensor = torch.cat(batch, dim=0).to(device)
            with torch.inference_mode():
                policy_logits, values = model(batch_tensor)
                policies = torch.softmax(policy_logits, dim=1).cpu()
                values = values.squeeze(1).cpu()
                if not use_shm_for_batch:
                    policies.share_memory_()
                    values.share_memory_()
        except Exception as e:
            print(f"[InferenceServer] Forward error: {e}")
            total_states = sum(n for _, _, n in batch_info)
            policies = torch.zeros((total_states, 4840), dtype=torch.float32)
            values = torch.zeros(total_states, dtype=torch.float32)
            if not use_shm_for_batch:
                policies.share_memory_()
                values.share_memory_()

        batch_count += 1

        # Hot-reload
        if batch_count % 10 == 0 and os.path.exists(model_path):
            current_mtime = os.path.getmtime(model_path)
            if current_mtime != last_mtime:
                try:
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    last_mtime = current_mtime
                    print(f"[InferenceServer] Hot-reloaded model from {model_path}")
                except Exception as e:
                    print(f"[InferenceServer] Hot-reload failed: {e}")

        # Route results
        offset = 0
        for wid, rid, n in batch_info:
            if wid < len(response_queues):
                if use_shm_for_batch:
                    p_slice = policies[offset:offset + n].numpy()
                    v_slice = values[offset:offset + n].numpy()
                    _write_shm_results(wid, n, p_slice, v_slice, shm_policies, shm_values)
                    response_queues[wid].put((rid, n))
                else:
                    response_queues[wid].put((
                        rid,
                        policies[offset:offset + n],
                        values[offset:offset + n]
                    ))
                offset += n

    # Clean up SHM handles
    for d in [shm_inputs, shm_policies, shm_values]:
        for shm in d.values():
            try:
                shm.close()
            except Exception:
                pass

    print("[InferenceServer] Shutting down")
