"""
Model registry on the cloud server: the cloud is where models are trained, the gateway
is where they run.

  GET  /api/models               manifest: name, size, sha256, modified - for every file the gateway uses
  GET  /api/models/<name>        download one file (only the names listed below)
  POST /api/qtable               the gateway backs up its ONLINE-learned Q-table here
  GET  /api/qtable/latest        restore it (e.g. after the SD card is replaced)

tools/sync_models.py on the gateway compares the manifest with its own files (sha256)
and downloads only what changed, so re-training in the cloud (python -m sim.train_q,
python -m vision.train) reaches the gateway with one command.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import threading
import time

import numpy as np

# files the gateway needs; the big cloud model stays on the server
GATEWAY_FILES = ("gateway_model.onnx", "edge_tiny.bin", "model_card.json", "q_table.npz", "rule_params.json", "probe.jpg")
_DEVICE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
MAX_QTABLE_BYTES = 2 * 1024 * 1024


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class ModelRegistry:
    def __init__(self, models_dir: str, backup_dir: str, keep_backups: int = 20):
        self.models_dir = models_dir
        self.backup_dir = backup_dir
        self.keep = keep_backups
        self._hash_cache: dict[str, tuple[float, int, str]] = {}   # name -> (mtime, size, sha256)
        self._lock = threading.Lock()
        os.makedirs(backup_dir, exist_ok=True)

    # ------------------------------------------------------------------ manifest
    def _sha(self, name: str, path: str, st) -> str:
        """sha256 is re-computed only when the file changed (mtime/size), not on every request."""
        with self._lock:
            c = self._hash_cache.get(name)
            if c and c[0] == st.st_mtime and c[1] == st.st_size:
                return c[2]
        digest = sha256_file(path)
        with self._lock:
            self._hash_cache[name] = (st.st_mtime, st.st_size, digest)
        return digest

    def manifest(self) -> dict:
        files = {}
        for name in GATEWAY_FILES:
            p = os.path.join(self.models_dir, name)
            if os.path.isfile(p):
                st = os.stat(p)
                files[name] = {"size": st.st_size, "sha256": self._sha(name, p, st),
                               "modified": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))}
        return {"files": files, "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def path(self, name: str) -> str | None:
        if name not in GATEWAY_FILES:            # whitelist: no path tricks possible
            return None
        p = os.path.join(self.models_dir, name)
        return p if os.path.isfile(p) else None

    # ------------------------------------------------------------------ Q-table backups
    @staticmethod
    def valid_device(device: str) -> bool:
        return bool(_DEVICE_RE.match(device or ""))

    def save_qtable(self, device: str, data: bytes) -> dict:
        if not self.valid_device(device):
            raise ValueError("bad device id")
        if not data or len(data) > MAX_QTABLE_BYTES:
            raise ValueError("empty or too large")
        z = np.load(io.BytesIO(data), allow_pickle=False)     # must be a real npz with a Q-table
        if "q" not in z.files or z["q"].ndim != 2:
            raise ValueError("not a Q-table (.npz with array 'q')")
        shape = list(z["q"].shape)
        d = os.path.join(self.backup_dir, device)
        os.makedirs(d, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        dated = os.path.join(d, f"q_online_{stamp}.npz")
        for p in (dated, os.path.join(d, "latest.npz")):
            tmp = p + ".tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
        olds = sorted(f for f in os.listdir(d) if f.startswith("q_online_"))
        for f in olds[:-self.keep]:
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
        return {"device": device, "shape": shape, "file": os.path.basename(dated), "bytes": len(data)}

    def latest_qtable(self, device: str) -> str | None:
        if not self.valid_device(device):
            return None
        p = os.path.join(self.backup_dir, device, "latest.npz")
        return p if os.path.isfile(p) else None

    def backups(self) -> list[str]:
        out = []
        for dev in sorted(os.listdir(self.backup_dir)) if os.path.isdir(self.backup_dir) else []:
            p = os.path.join(self.backup_dir, dev, "latest.npz")
            if os.path.isfile(p):
                out.append(p)
        return out
