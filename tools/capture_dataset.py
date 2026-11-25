"""
Collect YOUR OWN training photos on the conveyor (run on the gateway).

    python -m tools.capture_dataset --label good    --n 120
    python -m tools.capture_dataset --label damaged --n 120

Run the belt, place boxes one at a time; every IR trigger saves the 256x256 crop
to data/real/raw/<label>/. Vary position, rotation, lighting and the kind of damage.
Then split into train/val/test:

    python -m tools.capture_dataset --split          (70% / 15% / 15%)

and train on the LAPTOP (see docs/BUILD_GUIDE.md, Day 2):
    python -m vision.train --role edge  --data data/real --init imagenet --epochs 15
    python -m vision.train --role cloud --data data/real --init imagenet --epochs 12
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import time

from common.config import load_config, project_path


def capture(label: str, n: int):
    from gateway_node.camera import Camera
    from gateway_node.hardware import RealHardware
    cfg = load_config()
    out = project_path(f"data/real/raw/{label}")
    os.makedirs(out, exist_ok=True)
    cam, hw = Camera(cfg), RealHardware(cfg)
    count = {"n": 0}
    last = {"t": 0.0}

    def on_cam():
        now = time.time()
        if now - last["t"] < cfg["conveyor"]["min_box_gap_s"]:
            return
        last["t"] = now
        img = cam.capture_box()
        name = time.strftime("%Y%m%d_%H%M%S") + f"_{int(now*1000)%1000:03d}.jpg"
        img.save(os.path.join(out, name), quality=92)
        count["n"] += 1
        print(f"  saved {label} #{count['n']}: {name}")

    hw.bind(on_cam, lambda: None)
    hw.start_belt()
    print(f"belt running - place {n} {label.upper()} boxes one by one (Ctrl+C to stop early)")
    try:
        while count["n"] < n:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    hw.stop_belt(); hw.close(); cam.close()
    print(f"done: {count['n']} photos in {out}")


def split(seed=0):
    rnd = random.Random(seed)
    raw = project_path("data/real/raw")
    for lab in ("good", "damaged"):
        files = sorted(os.listdir(os.path.join(raw, lab)))
        rnd.shuffle(files)
        n = len(files)
        parts = {"train": files[: int(0.7 * n)], "val": files[int(0.7 * n): int(0.85 * n)], "test": files[int(0.85 * n):]}
        for sp, fs in parts.items():
            d = project_path(f"data/real/{sp}/{lab}")
            os.makedirs(d, exist_ok=True)
            for f in fs:
                shutil.copy2(os.path.join(raw, lab, f), os.path.join(d, f))
        print(f"{lab}: " + ", ".join(f"{k} {len(v)}" for k, v in parts.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["good", "damaged"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", action="store_true")
    a = ap.parse_args()
    if a.split:
        split()
    elif a.label:
        capture(a.label, a.n)
    else:
        ap.print_help()
