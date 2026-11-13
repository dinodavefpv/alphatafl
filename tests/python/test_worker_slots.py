"""
F3 regression tests: unique worker slot assignment in Orchestrator.dispatch_game.

Concurrently running games must never share a worker_id, because inference
response queues (and SHM regions) are per-worker.
"""

import sys
import os
import threading
import time

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.orchestrator import Orchestrator


class StubPool:
    """Stand-in for mp.Pool that tracks concurrently active worker_ids."""

    def __init__(self, fail=False, task_duration=0.2):
        self.fail = fail
        self.task_duration = task_duration
        self.submitted = []
        self.active_worker_ids = set()
        self.collisions = []
        self.max_concurrent = 0
        self.lock = threading.Lock()

    def apply_async(self, func, args=(), callback=None, error_callback=None):
        worker_id = args[0][0]
        with self.lock:
            self.submitted.append(worker_id)
            if worker_id in self.active_worker_ids:
                self.collisions.append(worker_id)
            self.active_worker_ids.add(worker_id)
            self.max_concurrent = max(self.max_concurrent, len(self.active_worker_ids))

        def run():
            time.sleep(self.task_duration)
            with self.lock:
                self.active_worker_ids.discard(worker_id)
            if self.fail:
                if error_callback:
                    error_callback(RuntimeError("stub worker failure"))
            elif callback:
                callback((worker_id, 1))

        threading.Thread(target=run, daemon=True).start()


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.02)
    return predicate()


class TestWorkerSlotUniqueness:
    """dispatch_game must hand out unique slots and skip dispatch when full."""

    def test_concurrent_worker_ids_unique(self):
        num_workers = 4
        orch = Orchestrator(num_workers=num_workers, num_simulations=1, batch_size=1)
        pool = StubPool()
        orch.pool = pool
        orch.is_running = True

        # Dispatch 2x num_workers games at once: only num_workers slots exist,
        # so the excess must be skipped rather than double-booked.
        for _ in range(2 * num_workers):
            orch.dispatch_game()

        assert len(pool.submitted) == num_workers
        assert orch.active_tasks == num_workers

        assert _wait_for(lambda: orch.active_tasks == 0)
        assert orch.games_completed == num_workers
        assert pool.collisions == []
        assert pool.max_concurrent <= num_workers

        # Freed slots must be reusable for the next wave.
        for _ in range(num_workers):
            orch.dispatch_game()

        assert _wait_for(lambda: orch.active_tasks == 0)
        assert orch.games_completed == 2 * num_workers
        assert pool.collisions == []

    def test_slot_released_on_worker_error(self):
        num_workers = 2
        orch = Orchestrator(num_workers=num_workers, num_simulations=1, batch_size=1)
        pool = StubPool(fail=True)
        orch.pool = pool
        orch.is_running = True

        for _ in range(num_workers):
            orch.dispatch_game()
        assert orch.active_tasks == num_workers

        # Errors must release the slots as well.
        assert _wait_for(lambda: orch.active_tasks == 0)

        for _ in range(num_workers):
            orch.dispatch_game()
        assert len(pool.submitted) == 2 * num_workers
        assert pool.collisions == []
        assert _wait_for(lambda: orch.active_tasks == 0)
