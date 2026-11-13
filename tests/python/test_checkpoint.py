"""Checkpoint I/O safety: atomic writes, ordered async saves, no overwrite of unloadable files."""

import glob
import os
import threading
import time

import pytest
import torch

from src.model.network import AlphaTaflNet
from src.training import checkpoint
from src.training.checkpoint import atomic_save, backup_unloadable
from src.training.trainer import Trainer


def _leftover_tmp(directory):
    return glob.glob(os.path.join(directory, "*.tmp"))


class TestAtomicSave:
    def test_writes_loadable_file_without_leftovers(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        atomic_save({"x": torch.arange(3)}, path)
        assert torch.equal(torch.load(path)["x"], torch.arange(3))
        assert _leftover_tmp(str(tmp_path)) == []

    def test_replaces_existing_file(self, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        atomic_save({"v": 1}, path)
        atomic_save({"v": 2}, path)
        assert torch.load(path)["v"] == 2

    def test_retries_when_target_locked(self, tmp_path, monkeypatch):
        path = str(tmp_path / "ckpt.pt")
        real_replace = os.replace
        calls = {"n": 0}

        def flaky_replace(src, dst):
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError("locked by reader")
            return real_replace(src, dst)

        monkeypatch.setattr(checkpoint.os, "replace", flaky_replace)
        atomic_save({"v": 7}, path, retry_delay=0.001)
        assert calls["n"] == 3
        assert torch.load(path)["v"] == 7
        assert _leftover_tmp(str(tmp_path)) == []

    def test_gives_up_and_cleans_tmp(self, tmp_path, monkeypatch):
        path = str(tmp_path / "ckpt.pt")

        def always_locked(src, dst):
            raise PermissionError("locked")

        monkeypatch.setattr(checkpoint.os, "replace", always_locked)
        with pytest.raises(PermissionError):
            atomic_save({"v": 1}, path, retries=3, retry_delay=0.001)
        assert not os.path.exists(path)
        assert _leftover_tmp(str(tmp_path)) == []


class TestBackupUnloadable:
    def test_missing_file_returns_none(self, tmp_path):
        assert backup_unloadable(str(tmp_path / "nope.pt")) is None

    def test_moves_file_and_avoids_collisions(self, tmp_path):
        path = tmp_path / "ckpt.pt"
        path.write_bytes(b"first")
        b1 = backup_unloadable(str(path))
        path.write_bytes(b"second")
        b2 = backup_unloadable(str(path))
        assert b1 != b2
        assert not path.exists()
        assert open(b1, "rb").read() == b"first"
        assert open(b2, "rb").read() == b"second"


class TestTrainerCheckpointSafety:
    def test_unloadable_checkpoint_is_backed_up_not_overwritten(self, tmp_path):
        path = tmp_path / "current_best.pt"
        path.write_bytes(b"not a torch checkpoint")

        trainer = Trainer(model_path=str(path), device="cpu")

        backups = glob.glob(str(tmp_path / "current_best.unloadable_*.pt"))
        assert len(backups) == 1
        assert open(backups[0], "rb").read() == b"not a torch checkpoint"
        # A fresh, loadable model was written in its place
        AlphaTaflNet().load_state_dict(torch.load(str(path)))
        assert trainer.model is not None

    def test_incompatible_architecture_is_backed_up(self, tmp_path):
        path = str(tmp_path / "current_best.pt")
        torch.save(AlphaTaflNet(num_res_blocks=1, num_filters=8).state_dict(), path)

        Trainer(model_path=path, device="cpu")

        assert len(glob.glob(str(tmp_path / "current_best.unloadable_*.pt"))) == 1
        AlphaTaflNet().load_state_dict(torch.load(path))

    def test_valid_checkpoint_is_loaded_untouched(self, tmp_path):
        path = str(tmp_path / "current_best.pt")
        ref = AlphaTaflNet()
        torch.save(ref.state_dict(), path)
        mtime = os.path.getmtime(path)

        trainer = Trainer(model_path=path, device="cpu")

        assert os.path.getmtime(path) == mtime
        assert glob.glob(str(tmp_path / "*.unloadable_*")) == []
        k = "conv_block.0.weight"
        assert torch.equal(trainer.model.state_dict()[k], ref.state_dict()[k])

    def test_newer_async_save_is_never_overwritten_by_older(self, tmp_path):
        path = str(tmp_path / "current_best.pt")
        trainer = Trainer(model_path=path, device="cpu")
        k = "conv_block.0.weight"

        with torch.no_grad():
            trainer.model.conv_block[0].weight.fill_(1.0)
        # Hold the writer lock so both saves queue up and may run in either order
        trainer._save_lock.acquire()
        try:
            trainer.save_model(sync=False)  # older snapshot (1.0)
            with torch.no_grad():
                trainer.model.conv_block[0].weight.fill_(2.0)
            trainer.save_model(sync=False)  # newer snapshot (2.0)
        finally:
            trainer._save_lock.release()

        deadline = time.time() + 5
        while trainer._saved_seq < trainer._save_seq and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)  # let any stale writer finish (it should no-op)

        saved = torch.load(path)[k]
        assert torch.all(saved == 2.0)
        assert _leftover_tmp(str(tmp_path)) == []
