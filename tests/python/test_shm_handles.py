"""Self-play must not leak per-game shared-memory handles."""

import os
import uuid
from multiprocessing import shared_memory

import pytest

import self_play


@pytest.fixture
def shm_slot(monkeypatch):
    """Create input/policy/value regions under a unique prefix; yield the worker id."""
    monkeypatch.setattr(self_play, "SHM_PREFIX", f"alphatafl_test_{uuid.uuid4().hex[:8]}")
    worker_id = 0
    owned = []

    def create(kinds=("input", "policy", "value")):
        for kind in kinds:
            owned.append(shared_memory.SharedMemory(
                create=True, size=4096, name=f"{self_play.SHM_PREFIX}_{kind}_{worker_id}"))
        return worker_id

    yield create
    for shm in owned:
        shm.close()
        shm.unlink()


@pytest.fixture
def track_opened(monkeypatch):
    """Record every SharedMemory handle opened by self_play helpers."""
    opened = []
    real = shared_memory.SharedMemory

    def recording(*args, **kwargs):
        shm = real(*args, **kwargs)
        if not kwargs.get("create", False):  # ignore regions the test itself creates
            opened.append(shm)
        return shm

    monkeypatch.setattr(shared_memory, "SharedMemory", recording)
    return opened


def _is_closed(shm):
    return shm.buf is None


class TestOpenCloseHelpers:
    def test_open_returns_three_handles_and_close_closes_them(self, shm_slot):
        wid = shm_slot()
        handles = self_play.open_worker_shm(wid)
        assert handles is not None and len(handles) == 3
        assert all(not _is_closed(h) for h in handles)
        self_play.close_worker_shm(handles)
        assert all(_is_closed(h) for h in handles)

    def test_partial_open_failure_closes_already_opened(self, shm_slot, track_opened):
        wid = shm_slot(kinds=("input", "policy"))  # value region missing
        assert self_play.open_worker_shm(wid) is None
        assert len(track_opened) == 2
        assert all(_is_closed(h) for h in track_opened)

    def test_close_tolerates_none(self):
        self_play.close_worker_shm((None, None, None))


class TestSelfPlayClosesHandles:
    def test_handles_closed_even_when_search_raises(self, shm_slot, track_opened, monkeypatch):
        wid = shm_slot()
        monkeypatch.setenv("ALPHATAFL_SHM", "1")

        class ExplodingMCTS:
            def __init__(self, *args, **kwargs):
                pass

            def search(self, *args, **kwargs):
                raise RuntimeError("search failed")

        monkeypatch.setattr(self_play.engine, "MCTS", ExplodingMCTS)

        with pytest.raises(RuntimeError, match="search failed"):
            self_play.self_play_game_batched(
                worker_id=wid, num_simulations=1, batch_size=1,
                inference_queue=None, response_queue=None, replay_queue=None,
                verbose=False)

        assert len(track_opened) == 3
        assert all(_is_closed(h) for h in track_opened)
