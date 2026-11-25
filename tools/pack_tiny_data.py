"""
Pack a photo folder into ONE small file of shrunken grey photos (for developing the ESP32's tiny model).

    python -m tools.pack_tiny_data --data data/public      ->  data/public_tiny.npz

For every photo in <data>/train|val|test/good|damaged it stores the pixels exactly as the gateway would send
them to the ESP32 (32x32, 48x48 grey; 48x48 colour for comparison), the label, and what the gateway and
cloud models say about the photo (useful as a teacher when training the tiny model).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from common import tiny
from common.config import load_config, project_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/public")
    a = ap.parse_args()
    cfg = load_config()
    data = a.data if os.path.isabs(a.data) else project_path(a.data)
    from common.vision_io import OnnxInspector
    teachers = {}
    for role, key in (("gateway", "gateway_onnx"), ("cloud", "cloud_onnx")):
        p = project_path(cfg["models"][key])
        if os.path.exists(p):
            teachers[role] = OnnxInspector(p, threads=0, providers=["CPUExecutionProvider"])
    out = {}
    for split in ("train", "val", "test"):
        g32, g48, c48, ys, names = [], [], [], [], []
        tp = {r: [] for r in teachers}
        for label, sub in ((0, "good"), (1, "damaged")):
            d = os.path.join(data, split, sub)
            if not os.path.isdir(d):
                continue
            for f in sorted(x for x in os.listdir(d) if x.lower().endswith((".jpg", ".jpeg", ".png"))):
                with Image.open(os.path.join(d, f)) as im:
                    im = im.convert("RGB")
                    if im.size != (256, 256):
                        im = im.resize((256, 256), Image.BILINEAR)
                    g32.append(np.frombuffer(tiny.to_pixels(im, 32, 1), np.uint8))
                    g48.append(np.frombuffer(tiny.to_pixels(im, 48, 1), np.uint8))
                    c48.append(np.frombuffer(tiny.to_pixels(im, 48, 3), np.uint8))
                    for r, m in teachers.items():
                        tp[r].append(m.predict(im)[0])
                ys.append(label)
                names.append(f)
        out[f"{split}_g32"] = np.stack(g32).reshape(-1, 1, 32, 32)
        out[f"{split}_g48"] = np.stack(g48).reshape(-1, 1, 48, 48)
        out[f"{split}_c48"] = np.stack(c48).reshape(-1, 3, 48, 48)
        out[f"{split}_y"] = np.asarray(ys, np.uint8)
        out[f"{split}_names"] = np.asarray(names)
        for r in teachers:
            out[f"{split}_p_{r}"] = np.asarray(tp[r], np.float32)
        print(f"{split}: {len(ys)} photos", flush=True)
    dst = data.rstrip("/\\") + "_tiny.npz"
    np.savez_compressed(dst, **out)
    print(f"saved {dst}  ({os.path.getsize(dst) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
