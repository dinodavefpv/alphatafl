import multiprocessing as mp
import time
import os
import signal
import sys
import torch

# Add the root directory to sys.path so we can import self_play
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from self_play import self_play_game
from src.model.network import AlphaTaflNet

def init_worker():
    # Ignore keyboard interrupt in workers, let parent CLI handle it cleanly
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Critical: Prevent PyTorch from spawning dozens of threads per worker
    # which causes massive CPU thrashing on high-core machines.
    torch.set_num_threads(1)

def worker_task(params):
    worker_id, num_simulations, model_path = params
    
    # Initialize model locally for this process
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlphaTaflNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Run the game quietly
    history, save_path = self_play_game(model, num_simulations=num_simulations, verbose=False)
    
    return worker_id, len(history), save_path

class Orchestrator:
    def __init__(self, num_workers=4, num_simulations=50):
        self.num_workers = num_workers
        self.num_simulations = num_simulations
        self.model_path = "models/current_best.pt"
        self.is_running = False
        self.pool = None
        self.games_completed = 0
        self.states_generated = 0
        self.active_tasks = 0
        
        if not os.path.exists("models"):
            os.makedirs("models")
            
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.pool = mp.Pool(processes=self.num_workers, initializer=init_worker)
            
    def stop(self):
        if self.is_running:
            self.is_running = False
            if self.pool:
                self.pool.terminate()
                self.pool.join()
                self.pool = None

    def dispatch_game(self, callback=None):
        if not self.is_running:
            return
            
        self.active_tasks += 1
        
        def internal_callback(result):
            worker_id, num_states, path = result
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
            args=((0, self.num_simulations, self.model_path),),
            callback=internal_callback,
            error_callback=internal_error
        )
