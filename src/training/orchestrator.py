import multiprocessing as mp
import time
import os
import signal
import sys
import torch

# Add the root directory to sys.path so we can import self_play
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.training.inference_server import inference_server_main

# On Windows, spawn is default. Queues must be created in parent before spawning.

def init_worker():
    # Ignore keyboard interrupt in workers, let parent CLI handle it cleanly
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Critical: Prevent PyTorch from spawning dozens of threads per worker
    torch.set_num_threads(1)


def worker_task(params):
    """
    Worker process that runs self-play using the C++ batched MCTS.
    
    Args:
        params: tuple of (
            worker_id, num_simulations, batch_size,
            inference_queue, response_queue, replay_queue,
            use_batched
        )
    """
    worker_id, num_simulations, batch_size, inference_queue, response_queue, replay_queue, use_batched = params
    
    # CPU-only worker
    torch.set_num_threads(1)
    
    from self_play import self_play_game, self_play_game_batched
    
    if use_batched:
        history = self_play_game_batched(
            worker_id=worker_id,
            num_simulations=num_simulations,
            batch_size=batch_size,
            inference_queue=inference_queue,
            response_queue=response_queue,
            replay_queue=replay_queue,
            verbose=False
        )
    else:
        # Fallback: load model locally (old behavior)
        import torch
        from src.model.network import AlphaTaflNet
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AlphaTaflNet()
        model_path = "models/current_best.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        history, save_path = self_play_game(model, num_simulations=num_simulations, verbose=False)
    
    return worker_id, len(history)


class Orchestrator:
    def __init__(self, num_workers=20, num_simulations=50, batch_size=64):
        self.num_workers = num_workers
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.model_path = "models/current_best.pt"
        self.is_running = False
        self.pool = None
        self.games_completed = 0
        self.states_generated = 0
        self.active_tasks = 0
        
        self.use_batched = os.environ.get("ALPHATAFL_BATCHED", "1") == "1"
        self.use_shm = os.environ.get("ALPHATAFL_SHM", "0") == "1"
        
        # Queues (created in parent, passed to children)
        self.inference_queue = None
        self.response_queues = None
        self.replay_queue = None
        
        # Shared memory regions (Phase 5A: created in parent, opened by children via name)
        self._shm_regions = []
        
        # Inference server process
        self.inference_server_process = None
        self.inference_stop_event = None
        
        if not os.path.exists("models"):
            os.makedirs("models")
            
    def _start_inference_server(self):
        """Start the GPU inference server process."""
        self.inference_stop_event = mp.Event()
        self.inference_queue = mp.Queue()
        self.response_queues = [mp.Queue() for _ in range(self.num_workers)]
        self.replay_queue = mp.Queue(maxsize=50000)
        
        # Create shared memory regions for zero-copy tensor transport
        if self.use_shm:
            try:
                from multiprocessing import shared_memory
                max_batch = int(os.environ.get("ALPHATAFL_INFERENCE_MAX_BATCH", "256"))
                input_size = max_batch * 14 * 11 * 11 * 4  # float32 bytes
                policy_size = max_batch * 4840 * 4
                value_size = max_batch * 4
                SHM_PREFIX = "alphatafl_shm"
                for i in range(self.num_workers):
                    shm_in = shared_memory.SharedMemory(
                        create=True, size=input_size, name=f"{SHM_PREFIX}_input_{i}"
                    )
                    shm_pol = shared_memory.SharedMemory(
                        create=True, size=policy_size, name=f"{SHM_PREFIX}_policy_{i}"
                    )
                    shm_val = shared_memory.SharedMemory(
                        create=True, size=value_size, name=f"{SHM_PREFIX}_value_{i}"
                    )
                    self._shm_regions.extend([shm_in, shm_pol, shm_val])
                print(f"[Orchestrator] Created {self.num_workers}x SHM regions "
                      f"({(input_size + policy_size + value_size) * self.num_workers / (1024*1024):.1f} MB total)")
            except Exception as e:
                print(f"[Orchestrator] SHM creation failed: {e}, disabling")
                self.use_shm = False
        
        self.inference_server_process = mp.Process(
            target=inference_server_main,
            args=(self.inference_queue, self.response_queues, self.model_path, self.inference_stop_event),
            daemon=False
        )
        self.inference_server_process.start()
        print(f"[Orchestrator] Inference server started (PID {self.inference_server_process.pid})")
        # Brief delay to let server initialize model
        time.sleep(2)
            
    def start(self):
        if not self.is_running:
            self.is_running = True
            if self.use_batched:
                self._start_inference_server()
            self.pool = mp.Pool(processes=self.num_workers, initializer=init_worker)
            
    def stop(self):
        if self.is_running:
            self.is_running = False
            if self.pool:
                self.pool.terminate()
                self.pool.join()
                self.pool = None
            
            # Shutdown inference server
            if self.inference_queue:
                try:
                    self.inference_queue.put(None)  # sentinel
                except Exception:
                    pass
            if self.inference_stop_event:
                self.inference_stop_event.set()
            if self.inference_server_process:
                self.inference_server_process.join(timeout=5)
                if self.inference_server_process.is_alive():
                    self.inference_server_process.terminate()
                    self.inference_server_process.join(timeout=2)
                self.inference_server_process = None
                print("[Orchestrator] Inference server stopped")
            
            # Clean up shared memory regions
            for shm in self._shm_regions:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
            self._shm_regions.clear()
            if self.use_shm:
                self.use_shm = False
                
    def dispatch_game(self, callback=None):
        if not self.is_running:
            return
            
        self.active_tasks += 1
        
        # Round-robin assign worker_id
        worker_id = self.games_completed % self.num_workers
        
        params = (
            worker_id, self.num_simulations, self.batch_size,
            self.inference_queue, self.response_queues[worker_id] if self.response_queues else None,
            self.replay_queue,
            self.use_batched
        )
        
        def internal_callback(result):
            worker_id, num_states = result
            self.games_completed += 1
            self.states_generated += num_states
            self.active_tasks -= 1
            print(f"\n[Worker] Game finished! Generated {num_states} states. (Total Games: {self.games_completed})")
            if callback:
                callback(result)
                
        def internal_error(err):
            self.active_tasks -= 1
            print(f"Worker error: {err}")

        self.pool.apply_async(
            worker_task, 
            args=(params,),
            callback=internal_callback,
            error_callback=internal_error
        )
        
    def get_replay_queue(self):
        """Return the replay queue for the trainer to read from."""
        return self.replay_queue
