import cmd
import threading
import time
import sys
import os
from src.training.orchestrator import Orchestrator

class AlphaTaflCLI(cmd.Cmd):
    intro = '\n=== AlphaTafl Training CLI ===\nType help or ? to list commands.\n'
    prompt = '(alphatafl) '
    
    def __init__(self):
        super().__init__()
        # Initialize with settings appropriate for an i9-14900K
        self.orchestrator = Orchestrator(num_workers=20, num_simulations=50)
        self.continuous_generation = False
        self.gen_thread = None
        self.gen_start_time = None
        
        self.loop_generation = False
        self.loop_thread = None
        self.loop_games_completed = 0
        self.loop_states_generated = 0
        self.loop_start_time = None
        
        self.trainer = None
        self.training = False
        self.train_thread = None

    def do_workers(self, arg):
        """Set or get the number of parallel CPU workers. Usage: workers [N]"""
        if arg:
            if self.orchestrator.is_running:
                print("Cannot change workers while running. Stop first.")
                return
            try:
                self.orchestrator.num_workers = int(arg)
                print(f"Set workers to {self.orchestrator.num_workers}")
            except ValueError:
                print("Invalid number.")
        else:
            print(f"Current workers: {self.orchestrator.num_workers}")

    def do_simulations(self, arg):
        """Set or get the number of MCTS simulations per move. Usage: simulations [N]"""
        if arg:
            try:
                self.orchestrator.num_simulations = int(arg)
                print(f"Set MCTS simulations to {self.orchestrator.num_simulations}")
            except ValueError:
                print("Invalid number.")
        else:
            print(f"Current simulations: {self.orchestrator.num_simulations}")

    def do_start(self, arg):
        """Start continuous self-play generation in the background."""
        if self.continuous_generation:
            print("Already generating.")
            return
            
        self.orchestrator.start()
        self.continuous_generation = True
        self.gen_start_time = time.time()
        self.gen_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.gen_thread.start()
        print(f"Started background self-play on {self.orchestrator.num_workers} cores.")
        print("Use 'status' to monitor, 'stop' to pause.")

    def do_stop(self, arg):
        """Stop background generation (loop or parallel)."""
        if self.loop_generation:
            print("Stopping loop...")
            self.loop_generation = False
            if self.loop_thread:
                self.loop_thread.join()
            self._print_loop_stats()
            self.loop_start_time = None
            print("Stopped loop.")
            return
            
        if not self.continuous_generation:
            print("Not currently generating.")
            return
            
        print("Stopping generation... (waiting for active games to finish)")
        self.continuous_generation = False
        if self.gen_thread:
            self.gen_thread.join()
        self.orchestrator.stop()
        self._print_gen_stats()
        self.gen_start_time = None
        print("Stopped generation.")
        
    def _print_gen_stats(self):
        if not self.gen_start_time:
            return
        elapsed = time.time() - self.gen_start_time
        games = self.orchestrator.games_completed
        games_per_sec = games / elapsed if elapsed > 0 else 0
        print(f"\n=== Generation Stats ===")
        print(f"Total Games:     {games}")
        print(f"Total Time:      {elapsed:.1f}s ({elapsed/60:.1f}m)")
        print(f"Games/Second:   {games_per_sec:.2f}")
        
    def do_train(self, arg):
        """Start the training loop."""
        if self.training:
            print("Already training.")
            return
            
        if not self.trainer:
            from src.training.trainer import Trainer
            self.trainer = Trainer()
            print("Loading existing data into replay buffer...")
            self.trainer.buffer.load_from_directory("data")
            print(f"Buffer size: {len(self.trainer.buffer)} states")
            
        self.training = True
        self.train_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.train_thread.start()
        print("Started training loop.")
        
    def do_stop_train(self, arg):
        """Stop the training loop."""
        if not self.training:
            print("Not currently training.")
            return
            
        print("Stopping training...")
        self.training = False
        if self.train_thread:
            self.train_thread.join()
        print("Stopped training.")

    def do_auto(self, arg):
        """Start both continuous self-play generation and the training loop simultaneously."""
        self.do_start("")
        # Give workers a brief moment to initialize before starting the trainer
        time.sleep(1)
        self.do_train("")
        print("\n=== Auto Mode Active ===")
        print("CPU workers are generating data, and the GPU is training.")

    def do_loop(self, arg):
        """Start sequential GPU self-play loop (one game at a time)."""
        if self.loop_generation:
            print("Already looping.")
            return
        self.loop_generation = True
        self.loop_games_completed = 0
        self.loop_states_generated = 0
        self.loop_start_time = time.time()
        self.loop_thread = threading.Thread(target=self._loop_play, daemon=True)
        self.loop_thread.start()
        print("Started sequential self-play loop on GPU.")
        print("Use 'status' to monitor, 'stop' to pause.")

    def do_auto_loop(self, arg):
        """Start sequential GPU self-play + training simultaneously."""
        self.do_loop("")
        time.sleep(1)
        self.do_train("")
        print("\n=== Auto Loop Mode Active ===")
        print("Sequential GPU self-play + training.")

    def do_stop_auto(self, arg):
        """Stop both generation and training."""
        self.do_stop("")
        self.do_stop_train("")
        print("Auto Mode stopped.")

    def do_reset_model(self, arg):
        """Delete the current model and start training from scratch. Usage: reset_model [confirm]"""
        if arg == "confirm":
            model_path = "models/current_best.pt"
            if os.path.exists(model_path):
                os.remove(model_path)
                print("Model deleted. Restart CLI to initialize a fresh random brain.")
            else:
                print("No model found to delete.")
        else:
            print("To prevent accidents, type 'reset_model confirm' to wipe the brain.")

    def do_play_one(self, arg):
        """Run a single self-play game synchronously. Usage: play_one [cpu|gpu]"""
        import torch
        import os
        from self_play import self_play_game
        from src.model.network import AlphaTaflNet
        
        if arg.strip().lower() in ("cpu", "gpu"):
            force = arg.strip().lower()
        else:
            force = None
        
        model = AlphaTaflNet()
        if force == "cpu":
            device = "cpu"
        elif force == "gpu":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_path = "models/current_best.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        print(f"Running on: {device}")
        t0 = time.time()
        history, saved_path = self_play_game(model, num_simulations=self.orchestrator.num_simulations, verbose=True)
        elapsed = time.time() - t0
        print(f"Game finished! Collected {len(history)} states.")
        print(f"Time: {elapsed:.1f}s for {len(history)} turns ({elapsed/len(history):.1f}s/turn)")
        print(f"Saved to: {saved_path}")

    def do_status(self, arg):
        """View generation and training statistics."""
        train_status = "Running" if self.training else "Stopped"
        
        if self.loop_generation:
            print(f"\n--- Loop Generation Status ---")
            print(f"State:            Running")
            print(f"Games Completed:  {self.loop_games_completed}")
            print(f"States Generated: {self.loop_states_generated}")
        elif self.continuous_generation:
            print(f"\n--- Generation Status (Parallel) ---")
            print(f"State:            Running")
            print(f"Workers:          {self.orchestrator.num_workers}")
            print(f"Active Tasks:     {self.orchestrator.active_tasks}")
            print(f"Games Completed:  {self.orchestrator.games_completed}")
            print(f"States Generated: {self.orchestrator.states_generated}")
        else:
            print(f"\n--- Generation Status ---")
            print(f"State:            Stopped")
        
        print(f"\n--- Training Status ---")
        print(f"State:            {train_status}")
        buffer_size = len(self.trainer.buffer) if self.trainer else 0
        print(f"Buffer Size:      {buffer_size} states")
        print("-----------------------\n")

    def do_exit(self, arg):
        """Exit the CLI."""
        if self.loop_generation:
            self.do_stop("")
        if self.continuous_generation:
            self.do_stop("")
        if self.training:
            self.do_stop_train("")
        print("Exiting...")
        return True

    def _generation_loop(self):
        max_games = 10000
        while self.continuous_generation and self.orchestrator.games_completed < max_games:
            # Keep the queue populated with tasks to avoid starvation
            target_tasks = self.orchestrator.num_workers * 2
            while self.orchestrator.active_tasks < target_tasks and self.continuous_generation and self.orchestrator.games_completed < max_games:
                self.orchestrator.dispatch_game()
            time.sleep(1)
        if self.orchestrator.games_completed >= max_games:
            print(f"\nReached {max_games} games, stopping generation.")
            self.continuous_generation = False
            self._print_gen_stats()
            self.gen_start_time = None
            
    def _loop_play(self):
        import torch
        import os
        from self_play import self_play_game
        from src.model.network import AlphaTaflNet
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AlphaTaflNet()
        model_path = "models/current_best.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"[Loop] Model loaded on {device}")
        
        max_games = 10000
        while self.loop_generation and self.loop_games_completed < max_games:
            # Reload weights from disk to pick up training updates
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            t0 = time.time()
            history, save_path = self_play_game(model, num_simulations=self.orchestrator.num_simulations, verbose=False)
            elapsed = time.time() - t0
            
            self.loop_games_completed += 1
            self.loop_states_generated += len(history)
            print(f"\n[Loop] Game {self.loop_games_completed} | {len(history)} turns | {elapsed:.1f}s ({elapsed/len(history):.1f}s/turn) | {save_path}")
        
        if self.loop_games_completed >= max_games:
            print(f"\nReached {max_games} games, stopping loop.")
            self.loop_generation = False
            self._print_loop_stats()
            self.loop_start_time = None
            
    def _print_loop_stats(self):
        if not self.loop_start_time:
            return
        elapsed = time.time() - self.loop_start_time
        games = self.loop_games_completed
        games_per_sec = games / elapsed if elapsed > 0 else 0
        print(f"\n=== Loop Stats ===")
        print(f"Total Games:     {games}")
        print(f"Total States:    {self.loop_states_generated}")
        print(f"Total Time:      {elapsed:.1f}s ({elapsed/60:.1f}m)")
        print(f"Games/Second:   {games_per_sec:.2f}")
            
    def _training_loop(self):
        batch_count = 0
        while self.training:
            # Periodically load new data generated by workers
            if batch_count % 50 == 0:
                self.trainer.buffer.load_from_directory("data")
                
            loss = self.trainer.train_step(batch_size=128)
            if loss:
                batch_count += 1
                if batch_count % 1000 == 0:
                    print(f"\n[Train] Batch {batch_count} | Policy Loss: {loss['policy_loss']:.4f} | Value Loss: {loss['value_loss']:.4f}")
                if batch_count % 100 == 0:
                    self.trainer.save_model()
            else:
                # Not enough data in buffer
                time.sleep(2)

if __name__ == '__main__':
    try:
        AlphaTaflCLI().cmdloop()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
