import torch
import torch.nn as nn
import torch.optim as optim
import glob
import re
import os
import sys
import threading
import time

# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.model.network import AlphaTaflNet
from src.training.replay_buffer import ReplayBuffer

class Trainer:
    def __init__(self, model_path="models/current_best.pt", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaTaflNet().to(self.device)
        self.model_path = model_path
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded existing model from {model_path} onto {self.device}")
            except RuntimeError as e:
                print(f"Warning: Could not load existing model from {model_path}. It may be from an older architecture version.")
                print(f"Error details: {e}")
                print(f"Initializing a fresh model instead. You may want to delete the old models/ and data/ directories.")
                self.save_model()
        else:
            print(f"Initialized new model on {self.device}")
            self.save_model()
        
        # Scan for existing versioned checkpoints to resume numbering
        highest = 0
        for f in glob.glob("models/alphatafl_v*.pt"):
            m = re.search(r'alphatafl_v(\d+)\.pt', f)
            if m:
                highest = max(highest, int(m.group(1)))
        self.save_count = highest * 50
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.buffer = ReplayBuffer(capacity=200000)
        
        self._queue_receiver_thread = None
        self._queue_receiver_running = False
        
    def start_queue_receiver(self, replay_queue):
        """Start a background thread that pulls from replay_queue into buffer."""
        if self._queue_receiver_running:
            return
        self._queue_receiver_running = True
        self._queue_receiver_thread = threading.Thread(
            target=self._queue_receiver_loop,
            args=(replay_queue,),
            daemon=True
        )
        self._queue_receiver_thread.start()
        print("[Trainer] Queue receiver started")
        
    def stop_queue_receiver(self):
        self._queue_receiver_running = False
        if self._queue_receiver_thread:
            self._queue_receiver_thread.join(timeout=2)
            
    def _queue_receiver_loop(self, replay_queue):
        while self._queue_receiver_running:
            try:
                item = replay_queue.get(timeout=0.5)
                if item is None:
                    continue
                state, policy, value = item
                self.buffer.push(state, policy, value)
            except Exception:
                time.sleep(0.1)
        
    def train_step(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return None
            
        self.model.train()
        states, target_policies, target_values = self.buffer.sample(batch_size, self.device)
        
        if states.shape[1] != 14:
            raise ValueError(f"Legacy data detected: tensor has {states.shape[1]} channels instead of 14. "
                             "Please delete the contents of the 'data/' and 'saves/' directories.")
        
        states = states.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)
        
        self.optimizer.zero_grad()
        
        out_policies, out_values = self.model(states)
        
        # Policy Loss: Cross Entropy
        log_probs = torch.log_softmax(out_policies, dim=1)
        policy_loss = -torch.sum(target_policies * log_probs, dim=1).mean()
        
        # Value Loss: Mean Squared Error
        value_loss = nn.MSELoss()(out_values, target_values)
        
        # Total Loss
        total_loss = policy_loss + value_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item()
        }
        
    def save_model(self, sync=False):
        """Save model. If sync=False, saves asynchronously in background thread."""
        if not os.path.exists("models"):
            os.makedirs("models")
        
        # Clone state dict to CPU to avoid corrupting GPU tensors during concurrent training
        state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        
        if sync:
            torch.save(state_dict, self.model_path)
        else:
            # Async save: training continues immediately
            def _save():
                try:
                    torch.save(state_dict, self.model_path)
                except Exception as e:
                    print(f"[Trainer] Async save failed: {e}")
            threading.Thread(target=_save, daemon=True).start()
        
    def maybe_checkpoint(self, batch_count):
        """Check if we should save a checkpoint based on batch count."""
        # Every 500 batches: current_best.pt
        if batch_count % 500 == 0 and batch_count > 0:
            self.save_model(sync=False)
            if batch_count % 5000 == 0:
                # Every 5000 batches: versioned checkpoint
                version = batch_count // 5000
                version_path = f"models/alphatafl_v{version:05d}.pt"
                state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                def _save_version():
                    try:
                        torch.save(state_dict, version_path)
                        print(f"[Trainer] Saved versioned checkpoint to {version_path}")
                    except Exception as e:
                        print(f"[Trainer] Versioned save failed: {e}")
                threading.Thread(target=_save_version, daemon=True).start()
