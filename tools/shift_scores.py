"""
The same photos, made harder on purpose: darker, blurred, or both. Scored again with all three models.

    python -m tools.shift_scores                 # photos = the data the models were trained on
    python -m tools.shift_scores --data data/public

Why: the models and the confidence levels were tuned on clean photos. A real line does not stay like
that: a lamp fails, the belt runs faster and the photo smears. This tool measures what the SAME models
say about the SAME boxes when the photo gets worse, so the simulator can replay "the photos changed"
with real verdicts instead of made-up ones.

  dim   : 40 % of the light (a lamp off, evening)
  blur  : the box moved during the exposure (smear along the belt, about 4 % of the picture width)
  both  : dim + blur + a little sensor noise

Writes  models/photo_scores_dim.npz, models/photo_scores_blur.npz, models/photo_scores_both.npz
        results/shift_scores.json   (how correct each model still is on the changed test photos)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from common.cascade import PhotoSplit, PhotoTable
from common.config import load_config, project_path
from common.rl import load_card
from sim.sensitivity import model_scores, tiny_scores

SHIFTS = ("dim", "blur", "both")
SHIFT_NAMES = {"dim": "darker (40 % light)", "blur": "blurred (box moving)", "both": "darker + blurred + noise"}


def _dim(a: np.ndarray) -> np.ndarray:
    return a * 0.4


def _blur(a: np.ndarray) -> np.ndarray:
    """Smear along the belt direction (x): the average of k neighbouring pixels in each row."""
    k = max(3, int(round(a.shape[1] * 0.04)) | 1)          # odd, about 4 % of the width
    pad = k // 2
    b = np.pad(a, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    c = np.cumsum(np.concatenate([np.zeros((b.shape[0], 1, b.shape[2])), b], axis=1), axis=1)
    return (c[:, k:] - c[:, :-k]) / k


def make_transform(kind: str):
    """photo (PIL, RGB) -> the changed photo. The noise is fixed by the photo's own pixels, so every
    model sees exactly the same changed photo."""
    from PIL import Image

    def fn(im):
        a = np.asarray(im, np.float64)
        if kind in ("blur", "both"):
            a = _blur(a)
        if kind in ("dim", "both"):
            a = _dim(a)
        if kind == "both":
            rng = np.random.default_rng(int(a[::7, ::7].sum()) % (2 ** 32))
            a = a + rng.normal(0, 6.0, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None, help="folder with val/ and test/ (each with good/ and damaged/)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    card = load_card(project_path(cfg["models"]["card"]))
    data = a.data or str((card.get("gateway") or {}).get("data") or "") or "data/synthetic"
    data = data if os.path.isabs(data) else project_path(data)
    for part in ("val", "test"):
        if not os.path.isdir(os.path.join(data, part, "good")):
            raise SystemExit(f"[shift] {os.path.join(data, part)} has no good/ and damaged/ folders.")
    clean_path = project_path(cfg["models"].get("photo_scores", "models/photo_scores.npz"))
    if not os.path.exists(clean_path):
        raise SystemExit("[shift] models/photo_scores.npz is missing - run tools.photo_scores first.")
    clean = PhotoTable.load(clean_path)
    thr = dict(clean.val.thr)                       # same cut-offs as on the clean photos: nobody re-tuned anything
    tiny_path = project_path(cfg["models"].get("edge_tiny", "models/edge_tiny.bin"))
    paths = {"gateway": project_path(cfg["models"]["gateway_onnx"]), "cloud": project_path(cfg["models"]["cloud_onnx"])}

    def acc(split, role):
        return float((split.verdict[role] == split.y).mean())

    t0 = time.time()
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "photos": os.path.relpath(data, project_path(".")),
           "n_val": len(clean.val), "n_test": len(clean.test), "shifts": {},
           "clean": {r: acc(clean.test, r) for r in clean.test.p}}
    for kind in SHIFTS:
        fn = make_transform(kind)
        splits = {}
        for part in ("val", "test"):
            folder = os.path.join(data, part)
            scores, y = {}, None
            if "edge" in thr and os.path.exists(tiny_path):
                y, scores["edge"] = tiny_scores(tiny_path, folder, fn)
            for role in ("gateway", "cloud"):
                y, scores[role] = model_scores(paths[role], folder, 0, fn)
            splits[part] = (y, scores)
        use = {r: thr[r] for r in splits["val"][1]}
        val = PhotoSplit(splits["val"][0], splits["val"][1], use, clean.val.edges)
        test = PhotoSplit(splits["test"][0], splits["test"][1], use, clean.val.edges)
        if len(val) != len(clean.val) or len(test) != len(clean.test):
            raise SystemExit("[shift] the photo folders changed since the clean scores were made - run tools.photo_scores again.")
        table = PhotoTable(val, test, f"{SHIFT_NAMES[kind]}: {os.path.relpath(data, project_path('.'))}")
        table.save(project_path(f"models/photo_scores_{kind}.npz"), {"made": out["generated"], "shift": kind})
        vg, vc, yy = test.verdict["gateway"], test.verdict["cloud"], test.y
        out["shifts"][kind] = {"name": SHIFT_NAMES[kind], **{r: acc(test, r) for r in test.p},
                               "cloud_fixes_gateway": float(((vg != yy) & (vc == yy)).mean()),
                               "cloud_makes_it_worse": float(((vg == yy) & (vc != yy)).mean()),
                               "level_share_old_edges": [float((test.conf == c).mean()) for c in range(1, 5)]}
        s = out["shifts"][kind]
        print(f"[shift] {kind:<5} " + "  ".join(f"{r} {100*s[r]:.1f}%" for r in test.p)
              + f"   (clean: " + "  ".join(f"{r} {100*out['clean'][r]:.1f}%" for r in test.p) + ")", flush=True)
    os.makedirs(project_path("results"), exist_ok=True)
    with open(project_path("results/shift_scores.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"[shift] saved models/photo_scores_dim|blur|both.npz and results/shift_scores.json ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
