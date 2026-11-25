"""
Pull the latest models from the cloud server to the gateway - only the files that changed.

    python -m tools.sync_models                      # server from config / env CONVEYOR_SERVER
    python -m tools.sync_models --server http://<ec2-ip>:8000 --dry-run
    python -m tools.sync_models --restore-qtable     # also fetch the last online Q-table backup

How it works: GET /api/models returns name + sha256 of every file the gateway uses.
The gateway hashes its own copies and downloads only the files whose hash differs,
writes each one to a temporary file, checks its sha256, then swaps it in
atomically (os.replace) - a broken download can never replace a good model.

Typical loop:  re-train in the cloud (python -m sim.train_q / vision.train on the EC2
instance) -> run this on the gateway -> restart gateway_node.main.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

from common.config import load_config, project_path
from common.net import api_key, device_id, make_session


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="download changed models from the cloud server")
    ap.add_argument("--config", default=None)
    ap.add_argument("--server", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore-qtable", action="store_true",
                    help="also download this device's last online Q-table backup to models/q_table_online.npz")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    server = (a.server or os.environ.get("CONVEYOR_SERVER") or cfg["server"]["cloud_url"]).rstrip("/")
    if "<" in server:
        raise SystemExit("give the server address: --server http://<ec2-ip>:8000 (or env CONVEYOR_SERVER)")
    s = make_session(api_key(cfg, a.api_key), device_id(cfg))
    r = s.get(f"{server}/api/models", timeout=10)
    if r.status_code == 401:
        raise SystemExit("401 from the server: set CONVEYOR_API_KEY (or --api-key)")
    r.raise_for_status()
    files = r.json().get("files", {})
    models_dir = project_path("models")
    changed = 0
    for name, info in sorted(files.items()):
        local = os.path.join(models_dir, name)
        same = os.path.exists(local) and sha256_file(local) == info["sha256"]
        print(f"  {name:20s} {'up to date' if same else 'CHANGED'}  ({info['size'] / 1024:.0f} kB, {info['modified']})")
        if same or a.dry_run:
            continue
        tmp = local + ".download"
        h = hashlib.sha256()
        with s.get(f"{server}/api/models/{name}", timeout=60, stream=True) as d:
            d.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in d.iter_content(1 << 16):
                    f.write(chunk)
                    h.update(chunk)
        if h.hexdigest() != info["sha256"]:
            os.remove(tmp)
            print(f"  !! {name}: checksum mismatch - kept the old file")
            continue
        os.replace(tmp, local)        # atomic swap
        changed += 1
    if a.restore_qtable and not a.dry_run:
        r = s.get(f"{server}/api/qtable/latest", timeout=20)
        if r.ok:
            out = project_path(cfg["agent"]["q_table"]).replace(".npz", "_online.npz")
            with open(out + ".download", "wb") as f:
                f.write(r.content)
            os.replace(out + ".download", out)
            print(f"  restored online Q-table -> {os.path.relpath(out)}  (start with --resume to use it)")
        else:
            print(f"  no Q-table backup on the server for this device ({r.status_code})")
    print(f"{changed} file(s) updated" + (" (dry run)" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
