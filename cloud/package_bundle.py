"""
Build the code bundle that the EC2 instance downloads from S3.

    python cloud/package_bundle.py              -> dist/server_bundle.zip

Contains only what the cloud side needs: the server, the shared code, the
simulator (so the agent can be re-trained in the cloud), the config, the
deployment scripts and the ONNX models. No PyTorch weights, no data sets,
no secrets (the API key lives in AWS SSM Parameter Store, never in the bundle).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INCLUDE_DIRS = ["common", "server", "sim", "cloud"]
INCLUDE_FILES = ["config/system.yaml", "models/cloud_model.onnx", "models/gateway_model.onnx",
                 "models/model_card.json", "models/q_table.npz", "models/rule_params.json", "models/probe.jpg",
                 "tools/__init__.py", "tools/check_server.py"]
OPTIONAL_FILES = ["config/calibration.json", "models/edge_tiny.bin", "results/E1_steady.csv", "results/E2_changes.csv",
                  "results/E3_pretraining_value.json", "results/E5_mismatch.csv", "results/E6_sensitivity.json",
                  "results/tradeoff.json", "results/STFG_summary.json", "data/public/SOURCE.md"]
SKIP_PARTS = {"__pycache__", ".secrets", "dist", ".pytest_cache"}
SKIP_SUFFIX = (".pyc", ".pt", ".tmp", ".download")


def files_to_pack():
    out = []
    for d in INCLUDE_DIRS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, d)):
            dirnames[:] = [x for x in dirnames if x not in SKIP_PARTS]
            for f in filenames:
                if not f.endswith(SKIP_SUFFIX):
                    out.append(os.path.relpath(os.path.join(dirpath, f), ROOT))
    for f in INCLUDE_FILES:
        if not os.path.exists(os.path.join(ROOT, f)):
            raise SystemExit(f"missing {f} - train/export the models first")
        out.append(f)
    out += [f for f in OPTIONAL_FILES if os.path.exists(os.path.join(ROOT, f))]
    return sorted(set(p.replace(os.sep, "/") for p in out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dist", "server_bundle.zip"))
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    files = files_to_pack()
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in files:
            info = zipfile.ZipInfo.from_file(os.path.join(ROOT, rel), rel)   # keeps each file's real date
            info.compress_type = zipfile.ZIP_DEFLATED
            if rel.endswith(".sh"):
                info.external_attr = 0o100755 << 16           # keep scripts executable on Linux
            else:
                info.external_attr = 0o100644 << 16
            with open(os.path.join(ROOT, rel), "rb") as f:
                z.writestr(info, f.read())
    with open(a.out, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"{a.out}: {len(files)} files, {os.path.getsize(a.out) / 1e6:.1f} MB, sha256 {digest[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
