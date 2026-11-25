"""
Train the ESP32's TINY model and write it to models/edge_tiny.bin.

    python -m vision.train_tiny --data data/public        (real photos)
    python -m vision.train_tiny --data data/synth         (computer-drawn boxes)

Why a separate, much smaller model
----------------------------------
An ESP32 has about 0.3 MB of free memory. The gateway's small model (MobileNetV3-Small) needs
many megabytes, so it cannot run there. This model is three or four 3x3 convolutions on a 32x32 or
48x48 grey photo: 6,000 to 12,000 numbers, 24 to 47 kB, a few tens of milliseconds on the ESP32. It is less accurate
on purpose: that is the edge end of the speed-accuracy trade-off.

What the script does
--------------------
  1. shrinks every photo exactly the way the gateway does before sending it to the ESP32
  2. trains two candidates (32 px with three layers, 48 px with four) and keeps the cheaper one for the
     ESP32, unless the bigger one is clearly better on the validation photos
  3. folds batch-norm into the convolutions and writes the file the firmware reads ("TNY1" format)
  4. re-computes every score with the same arithmetic as the firmware (common/tiny.py) and
     checks it against PyTorch, so the numbers in the model card are what the ESP32 will give
  5. picks the cost-based decision threshold on the validation photos, measures on the test photos,
     and writes the "edge" entry of models/model_card.json

Folder layout: <data>/train|val|test/good|damaged/*.jpg  (same as vision/train.py)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from common import tiny
from common.algos import cost_optimal_threshold

# candidates, cheapest for the ESP32 first: (input size, channels of each convolution, stride of each)
# Chosen after trying eight shapes on the public photos: more width or colour did not help, a fourth
# layer did, and the 48 px grey input matters more than anything else (see docs/ALGORITHM.md).
CANDIDATES = [(32, (8, 16, 32), (2, 2, 2)), (48, (8, 16, 24, 32), (2, 2, 2, 1))]


def load_split(root: str, split: str, sizes, ch: int, limit: int = 0):
    """-> {size: uint8 [N, ch, size, size]}, labels (1 = damaged)"""
    xs, ys = {s: [] for s in sizes}, []
    for label, sub in ((0, "good"), (1, "damaged")):
        d = os.path.join(root, split, sub)
        if not os.path.isdir(d):
            continue
        names = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if limit:
            names = names[:limit]
        for f in names:
            with Image.open(os.path.join(d, f)) as im:
                im = im.convert("RGB")
                if im.size != (256, 256):                 # the gateway always hands over a 256x256 box photo
                    im = im.resize((256, 256), Image.BILINEAR)
                for s in sizes:
                    xs[s].append(np.frombuffer(tiny.to_pixels(im, s, ch), np.uint8).reshape(ch, s, s))
            ys.append(label)
    return {s: np.stack(v) if v else np.zeros((0, ch, s, s), np.uint8) for s, v in xs.items()}, np.asarray(ys)


class TinyNet(nn.Module):
    def __init__(self, ch=1, widths=(8, 16, 32), strides=(2, 2, 2)):
        super().__init__()
        layers, cin = [], ch
        for w, s in zip(widths, strides):
            layers += [nn.Conv2d(cin, w, 3, s, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            cin = w
        self.features = nn.Sequential(*layers)
        self.strides = strides
        self.fc = nn.Linear(cin, 1)

    def forward(self, x):
        return self.fc(self.features(x).mean((2, 3))).squeeze(1)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """x: float [B, C, H, W] in 0..1. Flips, quarter turns and brightness changes. (Shifts and noise were
    tried too: at this tiny size they erase the little detail there is and made the result worse.)"""
    b = x.shape[0]
    flip = torch.rand(b, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    flip = torch.rand(b, generator=gen) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(2), x)
    if torch.rand(1, generator=gen).item() < 0.5:
        x = x.transpose(2, 3)
    gain = 0.85 + 0.3 * torch.rand(b, 1, 1, 1, generator=gen)
    return (x * gain).clamp(0, 1)


def export(model: TinyNet, size: int, ch: int, mean: float, std: float) -> bytes:
    """Fold each batch-norm into its convolution and write the firmware's file format."""
    convs, mods = [], list(model.features)
    for i in range(0, len(mods), 3):
        conv, bn = mods[i], mods[i + 1]
        k = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
        w = (conv.weight.detach() * k[:, None, None, None]).numpy()
        b = (bn.bias.detach() - bn.running_mean * k).numpy()
        convs.append((w, b, conv.stride[0]))
    return tiny.pack(size, ch, mean, std, convs, model.fc.weight.detach().numpy()[0], float(model.fc.bias.detach()[0]))


def metrics(ys, ps, thr: float) -> dict:
    pred = ps >= thr
    dam, good = ys == 1, ys == 0
    return {"n": int(len(ys)), "accuracy": float((pred == dam).mean()),
            "recall_damaged": float(pred[dam].mean()) if dam.any() else 0.0,
            "false_reject_rate": float(pred[good].mean()) if good.any() else 0.0}


def expected_cost(m: dict, p_def, c_esc, c_fr) -> float:
    return p_def * (1 - m["recall_damaged"]) * c_esc + (1 - p_def) * m["false_reject_rate"] * c_fr


def train_one(size, widths, strides, ch, xtr, ytr, xva, yva, epochs, seed, costs):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    mean, std = float(xtr.mean() / 255.0), float(max(1e-3, xtr.std() / 255.0))
    model = TinyNet(ch, widths, strides)
    xt = torch.from_numpy(xtr).float() / 255.0
    yt = torch.from_numpy(ytr).float()
    xv = (torch.from_numpy(xva).float() / 255.0 - mean) / std
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    bs = 64
    steps = epochs * max(1, len(xt) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=steps, pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state = None, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        tot = 0.0
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            x = (augment(xt[idx], gen) - mean) / std
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv)).numpy()
        thr = cost_optimal_threshold(yva, pv, *costs)
        m = metrics(yva, pv, thr)
        cost = expected_cost(m, *costs)
        if best is None or cost < best - 1e-9:
            best, best_state = cost, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            print(f"    {size}x{size} {widths}  round {ep+1:2d}/{epochs}  loss {tot / max(1, len(perm) // bs):.3f}  "
                  f"val correct {100*m['accuracy']:.1f}%  val cost {cost:.2f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    blob = export(model, size, ch, mean, std)
    with torch.no_grad():
        pt = torch.sigmoid(model(xv)).numpy()
    err = float(np.abs(tiny.forward(tiny.parse(blob), xva) - pt).max()) if len(xva) else 0.0
    return blob, best, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--out", default="models")
    ap.add_argument("--quick", action="store_true", help="only the smallest candidate (code test)")
    ap.add_argument("--channels", type=int, default=1, choices=(1, 3))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--packed", default=None, help="train from a file made by tools/pack_tiny_data.py instead of the photo folders")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    try:
        from common.config import load_config
        rc = load_config()
        costs = (rc["conveyor"]["defect_rate_assumed"], rc["reward"]["cost_escape"], rc["reward"]["cost_false_reject"])
    except Exception:
        costs = (0.3, 100.0, 10.0)
    plan = CANDIDATES[:1] if a.quick else CANDIDATES
    sizes = sorted({c[0] for c in plan})
    t0 = time.time()
    if a.packed:                                  # same pixels, already shrunk (tools/pack_tiny_data.py)
        z = np.load(a.packed)
        xtr, xva, xte = ({s: z[f"{sp}_g{s}"] for s in sizes} for sp in ("train", "val", "test"))
        ytr, yva, yte = (z[f"{sp}_y"].astype(int) for sp in ("train", "val", "test"))
    else:
        xtr, ytr = load_split(a.data, "train", sizes, a.channels, a.max_train)
        xva, yva = load_split(a.data, "val", sizes, a.channels)
        xte, yte = load_split(a.data, "test", sizes, a.channels)
    print(f"[edge] tiny model for the ESP32: train={len(ytr)} val={len(yva)} test={len(yte)} photos "
          f"(loaded in {time.time()-t0:.0f}s)", flush=True)
    if len(ytr) == 0 or len(yva) == 0:
        raise SystemExit("No images found. Check the --data folder layout (see top of this file).")

    cands = []
    for s, widths, strides in plan:
        blob, cost, err = train_one(s, widths, strides, a.channels, xtr[s], ytr, xva[s], yva, a.epochs, a.seed, costs)
        mults = tiny.mults(tiny.parse(blob))
        cands.append({"size": s, "widths": list(widths), "strides": list(strides), "val_cost": cost, "bytes": len(blob),
                      "mults": mults, "blob": blob, "err": err})
        print(f"  {s}x{s} {widths}: validation cost {cost:.2f}, file {len(blob)} bytes, {mults} multiplies per photo, "
              f"firmware-vs-PyTorch difference {err:.1e}", flush=True)
    # cheapest for the ESP32 first; a costlier one must be clearly better (2 %) to be worth its extra time
    cands.sort(key=lambda c: (c["size"], c["mults"]))
    pick = cands[0]
    for c in cands[1:]:
        if c["val_cost"] < pick["val_cost"] * 0.98:
            pick = c
    blob, size = pick["blob"], pick["size"]
    m = tiny.parse(blob)
    pv = tiny.forward(m, xva[size])
    thr = float(cost_optimal_threshold(yva, pv, *costs))
    pv5 = pv >= 0.5
    result = {"role": "edge", "arch": f"TinyCNN: {len(pick['widths'])} x 3x3 conv {tuple(pick['widths'])}, average, 1 output",
              "img": size, "channels": a.channels, "params": tiny.n_params(m), "mults": tiny.mults(m),
              "bytes": len(blob), "file": "edge_tiny.bin", "init": "scratch", "data": a.data, "epochs": a.epochs,
              "threshold": thr, "val": metrics(yva, pv, thr), "classes": ["good", "damaged"],
              "firmware_vs_pytorch_max_abs_err": pick["err"],
              "candidates": [{k: v for k, v in c.items() if k != "blob"} for c in cands],
              "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    result["val_accuracy_at_half"] = float((pv5 == (yva == 1)).mean())      # with a plain 50 % cut-off, for comparison
    if len(yte):
        pt = tiny.forward(m, xte[size])
        result["test"] = metrics(yte, pt, thr)
        result["test_accuracy_at_half"] = float(((pt >= 0.5) == (yte == 1)).mean())
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "edge_tiny.bin"), "wb") as f:
        f.write(blob)
    from common.rl import load_card
    card_p = os.path.join(a.out, "model_card.json")
    card = load_card(card_p)
    card["edge"] = result
    with open(card_p, "w") as f:
        json.dump(card, f, indent=2)
    k = "test" if "test" in result else "val"
    print(f"[edge] saved {os.path.join(a.out, 'edge_tiny.bin')}  {size}x{size} grey, {result['params']} numbers, "
          f"{len(blob)} bytes  threshold={thr:.2f}  {k}: acc={result[k]['accuracy']:.4f} "
          f"recall={result[k]['recall_damaged']:.3f} false_reject={result[k]['false_reject_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
