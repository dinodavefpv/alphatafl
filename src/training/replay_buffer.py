import os
import torch
import random
import threading
import time
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.loaded_files = set()
        self._lock = threading.Lock()
        
    def save_game(self, game_data):
        with self._lock:
            for state, policy, value in game_data:
                self.buffer.append((state, policy, value))
    
    def push(self, state, policy, value):
        """Thread-safe single push."""
        with self._lock:
            self.buffer.append((state, policy, value))
            
    def load_from_directory(self, data_dir="data"):
        if not os.path.exists(data_dir):
            return
            
        files = [f for f in os.listdir(data_dir) if f.endswith(".pt")]
        files.sort()
        
        new_files = [f for f in files if f not in self.loaded_files]
        
        for f in new_files[-1000:]:
            filepath = os.path.join(data_dir, f)
            try:
                game_data = torch.load(filepath, weights_only=False, map_location='cpu')
                self.save_game(game_data)
                self.loaded_files.add(f)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                
    def sample(self, batch_size, device='cpu'):
        with self._lock:
            batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)
        states = torch.stack([s.to('cpu') for s in states]).to(device)
        policies = torch.stack([p.to('cpu') for p in policies]).to(device)
        values = torch.stack([v.to('cpu') for v in values]).to(device)
        return states, policies, values
        
    def __len__(self):
        with self._lock:
            return len(self.buffer)
