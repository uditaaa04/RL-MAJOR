"""
Turn a downloaded public photo set into this project's training layout.

Input : a ZIP in the Roboflow "folder" export format
            train/<class>/*.jpg   valid/<class>/*.jpg   test/<class>/*.jpg
        (default: the newest *boxcondition*.zip in your Downloads folder)
Output: data/public/{train,val,test}/{good,damaged}/*.jpg   (256 x 256)
        data/public/SOURCE.md                               (where the photos came from, licence, counts)

    python -m tools.prepare_public_dataset
    python -m tools.prepare_public_dataset --zip "C:/Users/me/Downloads/BoxCondition.v2i.folder.zip"

Why the split is re-made here instead of using the ZIP's own train/valid/test:
public sets often contain several copies of the same source photo (re-uploads, rotated or resized
versions). If one copy is in "train" and another in "test", the test score is inflated. So all photos are
pooled, copies of one source photo are grouped (same original file name, or identical picture), and whole
GROUPS go to train (70%), val (15%) or test (15%).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import os
import random
import re
import shutil
import sys
import time
import zipfile

from PIL import Image

from common.config import project_path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
KNOWN = {
    "boxcondition": {
        "name": "BoxCondition (Roboflow Universe, workspace 'Robot Arm')",
        "url": "https://universe.roboflow.com/robot-arm-xlohb/boxcondition",
        "licence": "CC BY 4.0",
        "cite": "Robot Arm, \"BoxCondition Dataset\", Roboflow Universe, 2025. "
                "https://universe.roboflow.com/robot-arm-xlohb/boxcondition",
    },
}


def label_of(class_dir: str) -> str | None:
    c = class_dir.lower().replace("_", "").replace("-", "").replace(" ", "")
    if any(k in c for k in ("undamag", "intact", "good", "normal", "nodamage", "ok")):
        return "good"                      # checked first: "undamaged" also contains "damag"
    if any(k in c for k in ("damag", "defect", "broken", "bad", "torn", "crush")):
        return "damaged"
    return None


def source_name(member: str) -> str:
    """'train/undamagedpackages/undamaged190_jpg.rf.3fa9c1.jpg' -> 'undamaged190'"""
    base = os.path.basename(member)
    base = re.sub(r"\.rf\.[0-9a-zA-Z]+\.[a-zA-Z]+$", "", base)      # Roboflow's ".rf.<hash>.jpg"
    base = re.sub(r"_(jpe?g|png|webp|bmp|avif)$", "", base, flags=re.I)
    base = os.path.splitext(base)[0]
    return base.lower()


def tiny_hash(img: Image.Image) -> str:
    """Same picture (also after re-saving or resizing) -> same code. 16x16 grey, 1 bit per pixel."""
    g = img.convert("L").resize((16, 16), Image.BILINEAR)
    px = list(g.tobytes())
    avg = sum(px) / len(px)
    bits = "".join("1" if p > avg else "0" for p in px)
    return hashlib.md5(bits.encode()).hexdigest()[:16]


def find_zip(pattern: str) -> str | None:
    home = os.path.expanduser("~")
    cands = []
    for d in (os.path.join(home, "Downloads"), project_path("data"), project_path(".")):
        for p in glob.glob(os.path.join(d, "*.zip")):
            if pattern.lower() in os.path.basename(p).lower().replace(" ", "").replace("-", "").replace("_", ""):
                cands.append(p)
    return max(cands, key=os.path.getmtime) if cands else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="path to the downloaded ZIP (default: newest *boxcondition*.zip in Downloads)")
    ap.add_argument("--name", default="boxcondition", help="which known dataset this is (for the SOURCE.md text)")
    ap.add_argument("--out", default="data/public")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-class", type=int, default=0, help="use at most this many photos per class (0 = all)")
    a = ap.parse_args(argv)

    zpath = a.zip or find_zip(a.name)
    if not zpath or not os.path.exists(zpath):
        print(f"No ZIP found. Download the dataset first (its file name must contain '{a.name}'), "
              f"or pass --zip <path>.")
        return 2
    out = a.out if os.path.isabs(a.out) else project_path(a.out)
    print(f"[data] reading {zpath}  ({os.path.getsize(zpath) / 1e6:.0f} MB)", flush=True)

    items = []                      # (member, label)
    skipped = {}
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if m.endswith("/") or not m.lower().endswith(IMG_EXT):
                continue
            parts = m.replace("\\", "/").split("/")
            if len(parts) < 2:
                continue
            lab = label_of(parts[-2])
            if lab is None:
                skipped[parts[-2]] = skipped.get(parts[-2], 0) + 1
                continue
            items.append((m, lab))
        if not items:
            print("[data] no class folders recognised. Folders seen:", sorted(skipped)[:20])
            return 2
        if skipped:
            print("[data] ignored folders (not a good/damaged class):", skipped)

        # ---- group copies of the same source photo
        rng = random.Random(a.seed)
        groups: dict = {}            # key -> list of (member, label, image bytes)
        by_hash: dict = {}
        bad = 0
        t0 = time.time()
        for i, (m, lab) in enumerate(items):
            try:
                img = Image.open(io.BytesIO(z.read(m))).convert("RGB")
            except Exception:
                bad += 1
                continue
            h = tiny_hash(img)
            key = by_hash.get((lab, h)) or f"{lab}/{source_name(m)}"
            by_hash[(lab, h)] = key
            buf = io.BytesIO()
            img.resize((a.size, a.size), Image.BILINEAR).save(buf, format="JPEG", quality=90)
            groups.setdefault(key, []).append((m, lab, buf.getvalue()))
            if (i + 1) % 500 == 0:
                print(f"[data]   {i + 1}/{len(items)} photos read ({time.time() - t0:.0f} s)", flush=True)

    # ---- split by group, per class
    if os.path.isdir(out):
        shutil.rmtree(out)
    counts = {sp: {"good": 0, "damaged": 0} for sp in ("train", "val", "test")}
    n_groups = {"good": 0, "damaged": 0}
    for lab in ("good", "damaged"):
        keys = sorted(k for k in groups if k.startswith(lab + "/"))
        rng.shuffle(keys)
        if a.max_per_class:
            kept, n = [], 0
            for k in keys:
                if n >= a.max_per_class:
                    break
                kept.append(k)
                n += len(groups[k])
            keys = kept
        n_groups[lab] = len(keys)
        total = sum(len(groups[k]) for k in keys)
        cut_tr, cut_va = 0.70 * total, 0.85 * total
        seen = 0
        for k in keys:
            sp = "train" if seen < cut_tr else ("val" if seen < cut_va else "test")
            d = os.path.join(out, sp, lab)
            os.makedirs(d, exist_ok=True)
            for _, _, data in groups[k]:
                with open(os.path.join(d, f"{lab}_{counts[sp][lab]:05d}.jpg"), "wb") as f:
                    f.write(data)
                counts[sp][lab] += 1
            seen += len(groups[k])

    info = KNOWN.get(a.name, {"name": a.name, "url": "", "licence": "see the dataset page", "cite": ""})
    total = sum(sum(c.values()) for c in counts.values())
    lines = [f"# Photo source: {info['name']}", "",
             f"- Page: {info['url']}", f"- Licence: {info['licence']}",
             f"- Cite as: {info['cite']}",
             f"- Downloaded file: {os.path.basename(zpath)} ({os.path.getsize(zpath) / 1e6:.0f} MB), "
             f"prepared {time.strftime('%Y-%m-%d %H:%M')}",
             "- These are photos collected by other people, NOT photos from this project's conveyor.", "",
             "| split | good | damaged |", "|---|---|---|"]
    lines += [f"| {sp} | {counts[sp]['good']} | {counts[sp]['damaged']} |" for sp in ("train", "val", "test")]
    lines += ["", f"{total} photos, resized to {a.size} x {a.size}. Source-photo groups: "
              f"{n_groups['good']} good, {n_groups['damaged']} damaged. Copies of one source photo are kept in the "
              f"same split, so the test photos were never seen in training. Unreadable files skipped: {bad}."]
    with open(os.path.join(out, "SOURCE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[data] " + "  ".join(f"{sp}: {counts[sp]['good']} good / {counts[sp]['damaged']} damaged" for sp in counts))
    print(f"[data] written to {out}  ({total} photos, {n_groups['good'] + n_groups['damaged']} source groups, {bad} unreadable)")
    if min(counts["train"].values()) < 40 or min(counts["val"].values()) < 10:
        print("[data] too few photos in one class to train on")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
