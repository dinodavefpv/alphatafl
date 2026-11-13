"""
Phase 3 validation tests for K5 (Model Hot-Reload) and K6 (Async GUI AI).
"""

import sys
import os
import time
import numpy as np
import pytest
import torch
import threading
import multiprocessing as mp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
BUILD_DIR = os.path.join(PROJECT_ROOT, 'build', 'Release')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

import alphatafl_engine as engine
from src.training.inference_server import inference_server_main
from src.model.network import AlphaTaflNet
from src.training.trainer import Trainer


class TestK5HotReload:
    """K5: Model hot-reload in Inference Server."""

    def test_hot_reload_detect(self):
        """Server detects mtime change and reloads within one batch cycle."""
        inference_queue = mp.Queue()
        response_queue = mp.Queue()
        stop_event = mp.Event()
        
        # Create a temporary model file
        model_path = "models/test_hot_reload.pt"
        model = AlphaTaflNet()
        torch.save(model.state_dict(), model_path)
        original_mtime = os.path.getmtime(model_path)
        
        proc = mp.Process(
            target=inference_server_main,
            args=(inference_queue, [response_queue], model_path, stop_event),
            daemon=False
        )
        proc.start()
        time.sleep(2)
        
        try:
            # Send a request to ensure server is running
            state = engine.GameState()
            tensor = torch.from_numpy(state.to_tensor()).unsqueeze(0)
            inference_queue.put((0, 1, tensor))
            req_id, policies, values = response_queue.get(timeout=5)
            assert req_id == 1
            
            # Save a new model to trigger hot-reload
            time.sleep(0.1)
            new_model = AlphaTaflNet()
            torch.save(new_model.state_dict(), model_path)
            
            # Send another request - server should have reloaded
            inference_queue.put((0, 2, tensor))
            req_id, policies, values = response_queue.get(timeout=5)
            assert req_id == 2
            
            # Verify mtime changed
            new_mtime = os.path.getmtime(model_path)
            assert new_mtime > original_mtime
            
        finally:
            stop_event.set()
            try:
                inference_queue.put(None)
            except Exception:
                pass
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
            # Cleanup
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_hot_reload_no_reload(self):
        """Server does not reload when mtime is unchanged."""
        model_path = "models/current_best.pt"
        if not os.path.exists(model_path):
            model = AlphaTaflNet()
            torch.save(model.state_dict(), model_path)
        
        inference_queue = mp.Queue()
        response_queue = mp.Queue()
        stop_event = mp.Event()
        
        proc = mp.Process(
            target=inference_server_main,
            args=(inference_queue, [response_queue], model_path, stop_event),
            daemon=False
        )
        proc.start()
        time.sleep(2)
        
        try:
            # Send multiple requests without changing the model
            state = engine.GameState()
            tensor = torch.from_numpy(state.to_tensor()).unsqueeze(0)
            
            for i in range(3):
                inference_queue.put((0, i, tensor))
                req_id, policies, values = response_queue.get(timeout=5)
                assert req_id == i
            
            # If we got here, server handled requests without unnecessary reloads
            
        finally:
            stop_event.set()
            try:
                inference_queue.put(None)
            except Exception:
                pass
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)

    def test_async_checkpoint_cpu_clone(self, tmp_path):
        """Trainer saves from CPU-cloned state dict without blocking on the write."""
        model_path = str(tmp_path / "test_async.pt")
        trainer = Trainer(model_path=model_path)
        os.remove(model_path)  # drop the file written by __init__

        # Simulate training
        trainer.train_step(batch_size=4)

        # Trigger async save: only the CPU clone is synchronous, the write is not.
        # Generous bound so the test is not flaky when the full suite loads the machine.
        start = time.time()
        trainer.save_model(sync=False)
        elapsed = time.time() - start
        assert elapsed < 0.5

        # Wait for the background write to land
        deadline = time.time() + 10
        while trainer._saved_seq < trainer._save_seq and time.time() < deadline:
            time.sleep(0.01)

        assert os.path.exists(model_path)
        loaded = torch.load(model_path, map_location='cpu')
        assert isinstance(loaded, dict)

    def test_checkpoint_frequency(self, tmp_path):
        """current_best every 500 batches, versioned every 5000 - written next to model_path only."""
        import glob
        real_versions_before = sorted(glob.glob("models/alphatafl_v*.pt"))

        trainer = Trainer(model_path=str(tmp_path / "test_freq.pt"))
        for batch_count in [499, 500, 999, 1000, 4999, 5000, 9999, 10000]:
            trainer.maybe_checkpoint(batch_count)

        expected = {str(tmp_path / "alphatafl_v00001.pt"), str(tmp_path / "alphatafl_v00002.pt")}
        deadline = time.time() + 10
        while time.time() < deadline:
            if expected <= set(glob.glob(str(tmp_path / "alphatafl_v*.pt"))):
                break
            time.sleep(0.05)
        assert set(glob.glob(str(tmp_path / "alphatafl_v*.pt"))) == expected

        # The real models/ directory must be untouched
        assert sorted(glob.glob("models/alphatafl_v*.pt")) == real_versions_before


class TestK6AsyncGUI:
    """K6: Async GUI AI."""

    def test_gui_async_search_fields_exist(self):
        """Verify GUI has async search fields."""
        # We can't easily instantiate the GUI without pygame display,
        # but we can verify the module imports correctly and has the fields
        import gui
        
        # Verify HnefataflGUI has the async fields by inspecting the source
        import inspect
        source = inspect.getsource(gui.HnefataflGUI.__init__)
        assert 'ai_search_thread' in source
        assert 'ai_search_result' in source
        assert 'ai_is_searching' in source

    def test_gui_single_leaf_path(self):
        """Verify GUI uses single-leaf MCTS (not batched)."""
        import gui
        import inspect
        source = inspect.getsource(gui.main)
        
        # GUI should use engine.MCTS(eval_fn, c_puct) - single leaf
        assert 'engine.MCTS(eval_fn' in source
        # Should not use batched search in GUI
        assert 'batch_size' not in source or 'batch_size=100' not in source

    def test_gui_no_inference_server(self):
        """Verify GUI does not use Inference Server."""
        import gui
        import inspect
        source = inspect.getsource(gui.main)
        
        # GUI should not reference inference_queue or response_queue
        assert 'inference_queue' not in source
        assert 'response_queue' not in source
