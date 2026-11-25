"""
Train the GOOD / DAMAGED box inspector and export it to ONNX.

The network ("DefectNet")
-------------------------
  photo -> backbone CNN -> 1x1 conv -> DAMAGE MAP (one score per image region)
  box score = the highest region score   ->   P(damaged) = sigmoid(box score)

Taking the MAXIMUM over regions means one small tear anywhere is enough to
flag the box (averaging over the whole photo would wash a small defect out).
The damage map is also exported, so the dashboard can show WHERE the damage is.

Two of the three models are trained with this script (the ESP32's tiny one: vision/train_tiny.py):
  * GATEWAY (laptop at the belt) : MobileNetV3-Small backbone, 128x128 input
  * CLOUD   (AWS server)         : MobileNetV3-Large backbone, 224x224 input (default; the shipped
                                   models use 160x160 to train on a CPU). On a GPU you can use --arch resnet50.

Training signal
---------------
  * box label (good / damaged)                         - always
  * damage masks (where the damage is), if available   - synthetic data has them
    good boxes always get an all-zero map target, even for real photos.

Examples
--------
  # what produced the shipped models (synthetic data, from scratch, 2-core CPU):
  python -m vision.train --role gateway  --data data/synth --init scratch --epochs 10
  python -m vision.train --role gateway  --data data/synth --init models/gateway_model.pt --epochs 10 --lr 1e-3
  python -m vision.train --role cloud --data data/synth --init scratch --epochs 16 --img 160

  # YOUR real photos (on your laptop - downloads ImageNet weights automatically):
  python -m vision.train --role gateway  --data data/real --init imagenet --epochs 15
  python -m vision.train --role cloud --data data/real --init imagenet --epochs 12
  # or continue from the synthetic-trained weights:
  python -m vision.train --role cloud --data data/real --init models/cloud_model.pt

Folder layout (same for synthetic and real data):
  <data>/train/good/*.jpg   <data>/train/damaged/*.jpg   [<data>/train/masks/<name>.png]
  <data>/val/...            <data>/test/...  (test optional)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from common.algos import cost_optimal_threshold
from common.vision_io import random_hard_photo

CLASSES =["good", "damaged"]           # index 0 = good, 1 = damaged
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

ROLES = {
    "gateway": {"arch": "mobilenet_v3_small", "img": 128},
    "cloud": {"arch": "mobilenet_v3_large", "img": 224},
}


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class DefectNet(nn.Module):
    def __init__(self, arch: str, imagenet: bool = False):
        super().__init__()
        self.arch = arch
        if arch == "mobilenet_v3_small":
            bb = models.mobilenet_v3_small(weights="IMAGENET1K_V1" if imagenet else None).features
            ch = 576
        elif arch == "mobilenet_v3_large":
            bb = models.mobilenet_v3_large(weights="IMAGENET1K_V2" if imagenet else None).features
            ch = 960
        elif arch in ("resnet18", "resnet34", "resnet50"):
            ctor = getattr(models, arch)
            wname = {"resnet18": "IMAGENET1K_V1", "resnet34": "IMAGENET1K_V1", "resnet50": "IMAGENET1K_V2"}[arch]
            r = ctor(weights=wname if imagenet else None)
            bb = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
            ch = 2048 if arch == "resnet50" else 512
        else:
            raise ValueError(f"unknown arch {arch}")
        self.backbone = bb
        # torchvision's MobileNetV3 uses BatchNorm momentum 0.01: its running statistics then lag
        # far behind during short trainings and the model is wrong in eval mode. 0.1 fixes that.
        for mod in self.backbone.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.momentum = 0.1
        self.head = nn.Sequential(nn.Conv2d(ch, 128, 1), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
                                  nn.Conv2d(128, 1, 1))

    def forward(self, x):
        heat = self.head(self.backbone(x))                 # B x 1 x h x w   (damage score per region)
        z = heat.flatten(1).max(1).values                  # B               (box score)
        logits = torch.stack([torch.zeros_like(z), z], 1)  # [good, damaged] -> softmax = sigmoid(z)
        return logits, heat


class ExportWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        logits, heat = self.m(x)
        return logits, torch.sigmoid(heat)


def build_model(arch: str, init: str) -> DefectNet:
    m = DefectNet(arch, imagenet=(init == "imagenet"))
    if init not in ("scratch", "imagenet"):
        sd = torch.load(init, map_location="cpu")
        m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)
        print(f"loaded weights from {init}")
    return m


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def to_tensor(img: Image.Image) -> torch.Tensor:
    a = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


class BoxDS(Dataset):
    def __init__(self, root, split, img, train: bool, hard_aug: float = 0.0):
        self.hard_aug = float(hard_aug)        # share of training photos made darker / smeared (see random_hard_photo)
        self.items = []
        for ci, c in enumerate(CLASSES):
            d = os.path.join(root, split, c)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    mpath = os.path.join(root, split, "masks", os.path.splitext(f)[0] + ".png")
                    self.items.append((os.path.join(d, f), ci, f"{c}/{f}", mpath if os.path.exists(mpath) else None))
        self.img, self.train = img, train
        self.has_masks = any(it[3] for it in self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y, key, mpath = self.items[i]
        img = Image.open(path).convert("RGB")
        if mpath:
            mask = Image.open(mpath).convert("L")
            mvalid = 1.0
        else:
            mask = Image.new("L", img.size, 0)
            mvalid = 1.0 if y == 0 else 0.0        # good box: we KNOW there is no damage anywhere
        if self.train:
            if random.random() < 0.5:
                img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                img, mask = img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
            ang = random.uniform(-8, 8)
            img = img.rotate(ang, resample=Image.BILINEAR)
            mask = mask.rotate(ang, resample=Image.NEAREST)
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
            img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
            if self.hard_aug > 0 and random.random() < self.hard_aug:
                img = random_hard_photo(img, random)
        img = img.resize((self.img, self.img), Image.BILINEAR)
        mask = mask.resize((self.img, self.img), Image.BILINEAR)
        m = torch.from_numpy(np.asarray(mask, np.float32)[None] / 255.0)
        return to_tensor(img), y, m, mvalid, key


def mask_targets(mask: torch.Tensor, hw) -> torch.Tensor:
    """fraction of damaged pixels per heat-map cell -> 1 if > 1%"""
    frac = F.adaptive_avg_pool2d(mask, hw)
    return (frac > 0.01).float()


# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    ys, probs, keys = [], [], []
    for x, y, _, _, k in loader:
        logits, _ = model(x)
        p = torch.softmax(logits, 1)[:, 1]
        ys += y.tolist(); probs += p.tolist(); keys += list(k)
    ys, probs = np.array(ys), np.array(probs)
    ps = (probs > threshold).astype(int)
    tp = int(((ys == 1) & (ps == 1)).sum()); fn = int(((ys == 1) & (ps == 0)).sum())
    tn = int(((ys == 0) & (ps == 0)).sum()); fp = int(((ys == 0) & (ps == 1)).sum())
    return {
        "accuracy": float((ys == ps).mean()) if len(ys) else float("nan"),
        "recall_damaged": tp / max(1, tp + fn),        # share of damaged boxes caught
        "false_reject_rate": fp / max(1, fp + tn),     # share of good boxes wrongly rejected
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "n": int(len(ys)),
    }, (ys, ps, probs, keys)


def cost_threshold(ys, probs, p_def=0.30, c_esc=100.0, c_fr=10.0):
    """Choose the P(damaged) cut-off that minimises the SAME expected cost the RL reward uses:
       cost = p_def * miss_rate_damaged * c_esc + (1 - p_def) * false_reject_rate * c_fr"""
    # sort the scores once, then each candidate cut-off is two binary searches
    # (O(n log n) total instead of a full pass over the data for each of the 91 candidates)
    return cost_optimal_threshold(ys, probs, p_def, c_esc, c_fr)


def severity_breakdown(data_root, split, ys, ps, keys):
    meta_p = os.path.join(data_root, f"{split}_meta.csv")
    if not os.path.exists(meta_p):
        return None
    sev, kinds = {}, {}
    with open(meta_p) as f:
        for r in csv.DictReader(f):
            sev[r["file"]] = float(r["min_severity"] or 0)
            kinds[r["file"]] = [d.split(":")[0] for d in r["defects"].split(";") if d]
    out = {"by_severity": {}, "by_defect": {}}
    bands = [(0.25, 0.45, "subtle"), (0.45, 0.70, "medium"), (0.70, 1.01, "obvious")]
    # ONE pass over the damaged test images, grouping as we go
    # (before: one full pass per severity band and per defect type)
    by_sev: dict = {}
    by_def: dict = {}
    for i, k in enumerate(keys):
        if ys[i] != 1:
            continue
        hit = int(ps[i] == 1)
        s = sev.get(k, -1)
        for lo, hi, name in bands:
            if lo <= s < hi:
                by_sev.setdefault(name, []).append(hit)
                break
        kk = kinds.get(k)
        if kk is not None and len(kk) == 1:          # images with exactly one defect type
            by_def.setdefault(kk[0], []).append(hit)
    for _, _, name in bands:
        if name in by_sev:
            out["by_severity"][name] = {"recall": float(np.mean(by_sev[name])), "n": len(by_sev[name])}
    for t in sorted(by_def):
        out["by_defect"][t] = {"recall": float(np.mean(by_def[t])), "n": len(by_def[t])}
    return out


def export_onnx(model, img, path):
    model.eval()
    w = ExportWrapper(model).eval()
    dummy = torch.randn(1, 3, img, img)
    kw = dict(input_names=["input"], output_names=["logits", "heatmap"],
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "heatmap": {0: "batch"}},
              opset_version=17)
    try:
        torch.onnx.export(w, dummy, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(w, dummy, path, **kw)
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = dummy.numpy()
    with torch.no_grad():
        ref = w(dummy)[0].numpy()
    out = sess.run(None, {"input": x})[0]
    err = float(np.abs(out - ref).max())
    for _ in range(5):
        sess.run(None, {"input": x})
    t = time.perf_counter()
    for _ in range(30):
        sess.run(None, {"input": x})
    return err, (time.perf_counter() - t) / 30 * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=list(ROLES), required=True)
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--arch", default=None, help="override backbone (mobilenet_v3_small/large, resnet18/34/50)")
    ap.add_argument("--img", type=int, default=None, help="override input size")
    ap.add_argument("--init", default="scratch", help="scratch | imagenet | path/to/weights.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seg-weight", type=float, default=1.0, help="weight of the damage-map loss")
    ap.add_argument("--pos-weight", type=float, default=10.0, help="extra weight for (rare) damaged map cells")
    ap.add_argument("--hard-aug", type=float, default=0.3,
                    help="share of training photos made darker and / or smeared, so the model copes with a "
                         "worse day on the line (0 = off, as before)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arch = a.arch or ROLES[a.role]["arch"]
    img = a.img or ROLES[a.role]["img"]
    lr = a.lr or (2e-3 if a.init == "scratch" else 5e-4)
    os.makedirs(a.out, exist_ok=True)

    tr = BoxDS(a.data, "train", img, train=True, hard_aug=a.hard_aug)
    if a.max_train:
        random.shuffle(tr.items); tr.items = tr.items[: a.max_train]
    va = BoxDS(a.data, "val", img, train=False)
    te = BoxDS(a.data, "test", img, train=False)
    print(f"[{a.role}] arch={arch} img={img} init={a.init} device={dev} train={len(tr)} val={len(va)} "
          f"test={len(te)} masks={'yes' if tr.has_masks else 'no'}", flush=True)
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("No images found. Check the --data folder layout (see top of this file).")

    dl_tr = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
    dl_va = DataLoader(va, batch_size=64, num_workers=a.workers)
    dl_te = DataLoader(te, batch_size=64, num_workers=a.workers) if len(te) else None

    model = build_model(arch, a.init).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = a.epochs * max(1, len(dl_tr))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    bce_seg = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(a.pos_weight, device=dev))

    best, best_state, history = -1.0, None, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y, m, mvalid, _ in dl_tr:
            x, y, m, mvalid = x.to(dev), y.float().to(dev), m.to(dev), mvalid.float().to(dev)
            opt.zero_grad(set_to_none=True)
            logits, heat = model(x)
            z = logits[:, 1]
            loss_cls = bce(z, y).mean()
            tgt = mask_targets(m, heat.shape[-2:])
            per = bce_seg(heat, tgt).flatten(1).mean(1)
            loss_seg = (per * mvalid).sum() / mvalid.sum().clamp(min=1)
            loss = loss_cls + a.seg_weight * loss_seg
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        vm, _ = evaluate(model.cpu(), dl_va)
        model.to(dev)
        history.append({"epoch": ep + 1, "train_loss": tot / n, "val_acc": vm["accuracy"]})
        print(f"  epoch {ep+1:2d}/{a.epochs}  loss {tot/n:.4f}  val_acc {vm['accuracy']:.4f}  "
              f"recall {vm['recall_damaged']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        if vm["accuracy"] > best:
            best = vm["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model = model.cpu()
    model.load_state_dict(best_state)
    # cost-based decision threshold, chosen on the VALIDATION set (costs from config/system.yaml)
    try:
        from common.config import load_config
        rc = load_config()
        p_def, c_esc, c_fr = (rc["conveyor"]["defect_rate_assumed"], rc["reward"]["cost_escape"],
                              rc["reward"]["cost_false_reject"])
    except Exception:
        p_def, c_esc, c_fr = 0.3, 100.0, 10.0
    _, (vy, _, vprob, _) = evaluate(model, dl_va)
    thr = cost_threshold(vy, vprob, p_def, c_esc, c_fr)
    print(f"  cost-based threshold (from validation set): P(damaged) >= {thr:.2f}")
    vm, _ = evaluate(model, dl_va, threshold=thr)
    result = {"role": a.role, "arch": arch, "img": img, "init": a.init, "data": a.data, "epochs": a.epochs,
              "threshold": thr, "val": vm, "history": history, "classes": CLASSES, "mean": MEAN.tolist(), "std": STD.tolist(),
              "outputs": ["logits [good, damaged]", "heatmap (sigmoid damage score per region)"],
              "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if dl_te is not None:
        tm, (ys, ps, probs, keys) = evaluate(model, dl_te, threshold=thr)
        result["test"] = tm
        sb = severity_breakdown(a.data, "test", ys, ps, keys)
        if sb:
            result["test_breakdown"] = sb
    torch.save({"state_dict": model.state_dict(), "arch": arch, "img": img},
               os.path.join(a.out, f"{a.role}_model.pt"))
    onnx_p = os.path.join(a.out, f"{a.role}_model.onnx")
    err, ms = export_onnx(model, img, onnx_p)
    result["onnx"] = {"file": os.path.basename(onnx_p), "max_abs_err_vs_torch": err,
                      "cpu_latency_ms_training_machine": round(ms, 2)}
    card_p = os.path.join(a.out, "model_card.json")
    from common.rl import load_card
    card = load_card(card_p)
    card[a.role] = result
    with open(card_p, "w") as f:
        json.dump(card, f, indent=2)
    k = "test" if "test" in result else "val"
    print(f"[{a.role}] saved {onnx_p}  threshold={thr:.2f}  {k}: acc={result[k]['accuracy']:.4f} "
          f"recall={result[k]['recall_damaged']:.3f} false_reject={result[k]['false_reject_rate']:.3f}  "
          f"onnx_err={err:.1e}  cpu={ms:.1f}ms", flush=True)


if __name__ == "__main__":
    main()
