"""Crash-safe checkpoint I/O shared by the trainer and inference server."""

import os
import threading
import time

import torch


def atomic_save(obj, path, retries=10, retry_delay=0.05):
    """Save ``obj`` to ``path`` so readers never observe a partial file.

    Writes to a unique temporary file in the same directory, then swaps it
    into place with ``os.replace``. On Windows the swap fails while another
    process has ``path`` open (e.g. the inference server mid-load), so it is
    retried with a short backoff before giving up.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        torch.save(obj, tmp_path)
        for attempt in range(retries):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError:
                if attempt == retries - 1:
                    raise
                time.sleep(retry_delay * (attempt + 1))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def backup_unloadable(path):
    """Move an unloadable checkpoint aside instead of overwriting it.

    Returns the backup path, or None if ``path`` does not exist.
    """
    if not os.path.exists(path):
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root, ext = os.path.splitext(path)
    backup = f"{root}.unloadable_{stamp}{ext}"
    n = 1
    while os.path.exists(backup):
        backup = f"{root}.unloadable_{stamp}_{n}{ext}"
        n += 1
    os.replace(path, backup)
    return backup
