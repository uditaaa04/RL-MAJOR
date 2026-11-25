"""
Speed against accuracy: the SAME photos checked at all three places, timed and scored.

    python -m tools.tradeoff                         # pretend ESP32 if no board, cloud = server on this PC
    python -m tools.tradeoff --esp32 auto            # real ESP32 on USB
    python -m tools.tradeoff --server http://<ec2-ip>:8000 --esp32 auto     # the full real chain

For every photo:
  ESP32    shrink it, send it over the USB cable, tiny model, answer back      (time = all of that)
  gateway  small model on this laptop                                          (time = model only)
  cloud    JPEG, HTTP to the server, big model, answer back                    (time = all of that)
Each place uses its own cut-off from models/model_card.json. No belt, no deadline, no agent:
this is the raw trade-off the agent has to work with.

Writes results/tradeoff.json, results/TRADEOFF.md, results/figures/tradeoff.png.
Every row says how it was measured (real board or emulated, real Internet or same PC).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
from PIL import Image

from common import tiny
from common.config import load_config, project_path
from common.net import api_key, device_id
from common.rl import load_card
from common.vision_io import OnnxInspector, to_jpeg


def pick_photos(folder: str, n: int, seed: int = 0):
    """-> [(path, 'good' | 'damaged')] in a fixed shuffled order. n = 0: every photo in the folder (the same
    photos and the same good/damaged mix the model card was measured on); n > 0: a random n of them."""
    out = []
    rng = random.Random(seed)
    for lab in ("good", "damaged"):
        d = os.path.join(folder, lab)
        if not os.path.isdir(d):
            continue
        out += [(os.path.join(d, f), lab) for f in sorted(os.listdir(d)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    rng.shuffle(out)
    return out[:n] if n and n > 0 else out


def score(ys, ps, thr):
    ys, ps = np.asarray(ys), np.asarray(ps)
    pred = ps >= thr
    dam, good = ys == 1, ys == 0
    return {"accuracy": float((pred == dam).mean()), "recall_damaged": float(pred[dam].mean()) if dam.any() else None,
            "false_reject_rate": float(pred[good].mean()) if good.any() else None,
            "accuracy_at_half": float(((ps >= 0.5) == dam).mean())}      # with a plain 50 % cut-off, for comparison


def times(ms):
    a = np.asarray(ms, float)
    return {"ms_median": float(np.median(a)), "ms_p95": float(np.percentile(a, 95)), "ms_mean": float(a.mean()),
            "ms_min": float(a.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--server", default=None, help="cloud server URL (default: env CONVEYOR_SERVER or this PC)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--esp32", default="auto", help="auto | mock | a port such as COM5")
    ap.add_argument("--data", default=None, help="folder with good/ and damaged/ photos (default: the test photos "
                                                 "of the data the models were trained on)")
    ap.add_argument("--n", type=int, default=0, help="how many photos (0 = all test photos)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    card = load_card(project_path(cfg["models"]["card"]))

    # ---------------- photos
    data = a.data or os.environ.get("CONVEYOR_MOCK_FEED")
    if not data:
        d = str((card.get("gateway") or {}).get("data") or "")
        cand = project_path(os.path.join(d, "test")) if d else ""
        data = cand if cand and os.path.isdir(cand) else project_path(cfg["mock"]["feed_dir"])
    data = data if os.path.isabs(data) else project_path(data)
    photos = pick_photos(data, a.n, a.seed)
    if len(photos) < 10:
        raise SystemExit(f"Not enough photos in {data} (need good/ and damaged/ folders).")
    kind = "own" if "real" in data.replace("\\", "/") else ("public" if "public" in data.replace("\\", "/") else "synthetic")
    print(f"[tradeoff] {len(photos)} photos from {data}")
    imgs = []
    for path, lab in photos:
        with Image.open(path) as im:
            imgs.append((im.convert("RGB").resize((256, 256), Image.BILINEAR), 1 if lab == "damaged" else 0))
    ys = [y for _, y in imgs]
    thr = {r: float((card.get(r) or {}).get("threshold", cfg["models"]["damaged_threshold"])) for r in ("edge", "gateway", "cloud")}
    tiers, notes = [], []

    # ---------------- ESP32
    per_photo = {}                               # each photo's score at each place (for the cascade analysis)
    tiny_path = project_path(cfg["models"].get("edge_tiny", "models/edge_tiny.bin"))
    if os.path.exists(tiny_path):
        from gateway_node.esp32_link import EdgeTiny, Esp32Error, connect
        model = tiny.TinyModel.load(tiny_path)
        link = None
        if a.esp32.lower() not in ("mock", "off", "none"):
            try:
                link = connect(cfg, model.blob, None if a.esp32.lower() == "auto" else a.esp32, verbose=True)
            except Esp32Error as e:
                print(f"[tradeoff] no ESP32 board ({e}) - its time will be EMULATED")
        edge = EdgeTiny(model, link, timeout_s=1.0, baud=int(cfg.get("esp32", {}).get("baud", 460800)))
        edge.measure()
        ps, ms, esp, fails = [], [], [], 0
        for img, _ in imgs:
            r = edge.check(img)
            if not r.get("ok"):
                fails += 1
                r = {"p_damaged": model.predict(img), "compute_s": float("nan"), "esp_ms": float("nan")}
            ps.append(r["p_damaged"]); ms.append(r["compute_s"] * 1000); esp.append(r["esp_ms"])
        per_photo["edge"] = [float(x) for x in ps]
        ok = [m for m in ms if m == m]
        if link is not None:
            link.close()
        real = link is not None and len(ok) >= len(ms) * 0.9
        tiers.append({"key": "edge", "place": "ESP32 (edge)", "model": (card.get("edge") or {}).get("arch", "tiny CNN"),
                      "input": f"{model.size}x{model.size} grey", "params": model.params, "threshold": thr["edge"],
                      **score(ys, ps, thr["edge"]), **times(ok or [edge.ewma_ms]),
                      "compute_ms_median": float(np.nanmedian(esp)) if esp else None,
                      "measured": "real ESP32 over USB" if real else "EMULATED (no board connected): cable time from the baud "
                      "rate, compute time from 40 ns per multiply",
                      "real": bool(real), "failed": fails,
                      "time_includes": "shrinking the photo, USB cable both ways, tiny model on the ESP32"})
        if not real:
            notes.append("The ESP32 time is an estimate until this is run with the board plugged in. Its accuracy is exact: "
                         "the same arithmetic as the firmware.")
    else:
        notes.append("No tiny model yet (models/edge_tiny.bin), so the ESP32 row is missing.")

    # ---------------- gateway
    gw_path = project_path(cfg["models"]["gateway_onnx"])
    gw = OnnxInspector(gw_path, threads=cfg["models"]["gateway_threads"], providers=["CPUExecutionProvider"])
    for _ in range(3):
        gw.predict(imgs[0][0])
    ps, ms = [], []
    for img, _ in imgs:
        t = time.perf_counter()
        p, _, _ = gw.predict(img)
        ms.append((time.perf_counter() - t) * 1000); ps.append(p)
    per_photo["gateway"] = [float(x) for x in ps]
    tiers.append({"key": "gateway", "place": "Gateway (laptop)", "model": (card.get("gateway") or {}).get("arch", "small CNN"),
                  "input": f"{gw.size}x{gw.size} colour", "threshold": thr["gateway"], **score(ys, ps, thr["gateway"]), **times(ms),
                  "measured": "this laptop, CPU", "real": True, "time_includes": "small model on the laptop (no network)"})

    # ---------------- cloud
    server = (a.server or os.environ.get("CONVEYOR_SERVER") or "http://127.0.0.1:%d" % cfg["server"]["port"]).rstrip("/")
    local = any(h in server for h in ("127.0.0.1", "localhost", "[::1]"))
    ps, ms, srv_ms, ok_server = [], [], [], False
    try:
        from common.net import make_session
        sess = make_session(api_key(cfg, a.api_key), device_id(cfg))
        h = sess.get(server + "/health", timeout=4).json()
        ok_server = bool(h.get("ok"))
    except Exception as e:
        print(f"[tradeoff] cloud server {server} not reachable ({type(e).__name__})")
    if ok_server:
        q = int(cfg["camera"]["jpeg_quality"])
        for _ in range(3):
            sess.post(server + "/infer", params={"probe": 1}, data=to_jpeg(imgs[0][0], q),
                      headers={"Content-Type": "image/jpeg"}, timeout=10)
        fails = 0
        for img, _ in imgs:
            t = time.perf_counter()
            try:
                r = sess.post(server + "/infer", params={"probe": 1}, data=to_jpeg(img, q),
                              headers={"Content-Type": "image/jpeg"}, timeout=10)
                r.raise_for_status()
                j = r.json()
            except Exception:
                fails += 1
                continue
            ms.append((time.perf_counter() - t) * 1000); ps.append(j["p_damaged"]); srv_ms.append(j.get("server_ms") or 0)
        if fails or len(ps) != len(imgs):
            raise SystemExit(f"[tradeoff] {fails} cloud requests failed - check the server and the API key, then run again.")
        per_photo["cloud"] = [float(x) for x in ps]
        on_aws = not local                      # a server on this same PC is never called "AWS"
        measured = (f"real Internet to {server}" if not local else
                    "server on THIS PC: no Internet in the path, so the real cloud will be slower")
        tiers.append({"key": "cloud", "place": "Cloud (AWS)" if on_aws else "Cloud server (on this PC for now)",
                      "model": (card.get("cloud") or {}).get("arch", "big CNN"),
                      "input": f"{(card.get('cloud') or {}).get('img', '?')}x{(card.get('cloud') or {}).get('img', '?')} colour",
                      "threshold": thr["cloud"], **score(ys, ps, thr["cloud"]), **times(ms),
                      "compute_ms_median": float(np.median(srv_ms)) if srv_ms else None, "measured": measured,
                      "real": bool(not local), "time_includes": "JPEG, upload, big model on the server, answer back"})
        if local:
            notes.append("The cloud server ran on this same PC, so its time has no Internet in it. Run this again with "
                         "--server http://<AWS address>:8000 for the real number.")
    else:
        big = OnnxInspector(project_path(cfg["models"]["cloud_onnx"]), threads=0, providers=["CPUExecutionProvider"])
        for _ in range(2):
            big.predict(imgs[0][0])
        for img, _ in imgs:
            t = time.perf_counter()
            p, _, _ = big.predict(img)
            ms.append((time.perf_counter() - t) * 1000); ps.append(p)
        per_photo["cloud"] = [float(x) for x in ps]
        tiers.append({"key": "cloud", "place": "Cloud model (run on this PC, no server)",
                      "model": (card.get("cloud") or {}).get("arch", "big CNN"), "input": "colour", "threshold": thr["cloud"],
                      **score(ys, ps, thr["cloud"]), **times(ms), "measured": "MODEL ONLY on this laptop: no server, no network",
                      "real": False, "time_includes": "big model only"})
        notes.append("No cloud server was reachable, so the cloud row is the big model's compute time only. "
                     "Start the server (START_DEMO or the AWS one) and run this again.")

    # ---------------- plain reading of the numbers
    by = {t["key"]: t for t in tiers}
    read = []
    order_acc = sorted(tiers, key=lambda t: -t["accuracy"])
    order_ms = sorted(tiers, key=lambda t: t["ms_median"])
    read.append("Most accurate first: " + ", ".join(f"{t['place']} {100*t['accuracy']:.1f}%" for t in order_acc) + ".")
    read.append("Fastest first: " + ", ".join(f"{t['place']} {t['ms_median']:.0f} ms" for t in order_ms) + ".")
    if "edge" in by and "gateway" in by:
        e, g = by["edge"], by["gateway"]
        if e["ms_median"] > g["ms_median"] and e["accuracy"] <= g["accuracy"]:
            read.append(f"The ESP32 is both slower ({e['ms_median']:.0f} ms against {g['ms_median']:.0f} ms) and less accurate than "
                        f"the gateway, because the photo has to be sent to it over the cable first. It is not there to win this "
                        f"table. It is there for when the gateway is too busy and the Internet is too slow: then it is the only "
                        f"place that still answers before the box reaches the arm.")
        elif e["ms_median"] <= g["ms_median"]:
            read.append("The ESP32 answers faster than the gateway here, at lower accuracy.")
    if "gateway" in by and "cloud" in by:
        g, c = by["gateway"], by["cloud"]
        d_acc, d_ms = 100 * (c["accuracy"] - g["accuracy"]), c["ms_median"] - g["ms_median"]
        if d_acc > 0 and d_ms > 0:
            read.append(f"Going from the gateway to the cloud buys {d_acc:.1f} points of accuracy for {d_ms:.0f} ms more per box.")
        elif d_acc <= 0:
            read.append("On these photos the cloud model is NOT more accurate than the gateway model.")

    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "n": len(imgs), "photos": os.path.relpath(data, project_path(".")),
           "photo_kind": kind, "server": server, "tiers": tiers, "reading": read, "notes": notes}
    res = project_path("results")
    os.makedirs(os.path.join(res, "figures"), exist_ok=True)
    with open(os.path.join(res, "tradeoff.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    keys = [k for k in ("edge", "gateway", "cloud") if len(per_photo.get(k, [])) == len(ys)]
    with open(os.path.join(res, "tradeoff_scores.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("damaged," + ",".join("p_" + k for k in keys) + "\n")
        for i in range(len(ys)):
            f.write(str(int(ys[i])) + "," + ",".join("%.5f" % per_photo[k][i] for k in keys) + "\n")
    pc = lambda v: "n/a" if v is None else f"{100*v:.1f}%"
    L = ["# Speed against accuracy: ESP32, gateway, cloud", "",
         f"Generated {out['generated']}. The same {len(imgs)} photos ({kind} photos, `{out['photos']}`) were checked at all "
         "three places. Each place used its own cut-off from the model card.", "",
         "| place | model | correct verdicts | damaged caught | good wrongly rejected | correct with a plain 50% cut-off | time per photo (median) | 95% under | how it was measured |",
         "|---|---|---|---|---|---|---|---|---|"]
    for t in tiers:
        L.append(f"| {t['place']} | {t['model']} ({t['input']}) | {pc(t['accuracy'])} | {pc(t['recall_damaged'])} | "
                 f"{pc(t['false_reject_rate'])} | {pc(t.get('accuracy_at_half'))} | {t['ms_median']:.1f} ms | {t['ms_p95']:.1f} ms | {t['measured']} |")
    L += ["", "The first three columns use each model's cost-based cut-off (a shipped defect costs ten times a wasted good "
          "box, so every cut-off leans toward rejecting; the weaker the model, the more good boxes that costs). "
          "The 50% column shows the same model with a neutral cut-off."]
    L += ["", "What each time includes:"] + [f"- {t['place']}: {t['time_includes']}." for t in tiers]
    L += ["", "## What the numbers say", ""] + [f"- {r}" for r in read]
    if notes:
        L += ["", "## Read with care", ""] + [f"- {n}" for n in notes]
    L += ["", "![](figures/tradeoff.png)", ""]
    with open(os.path.join(res, "TRADEOFF.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for t in tiers:
            col = {"edge": "#111827", "gateway": "#6b7280", "cloud": "#06b6d4"}[t["key"]]
            ax.errorbar(t["ms_median"], 100 * t["accuracy"], xerr=[[max(0.0, t["ms_median"] - t["ms_min"])], [max(0.0, t["ms_p95"] - t["ms_median"])]],
                        fmt="o" if t["real"] else "s", ms=9, color=col, mfc=col if t["real"] else "white", capsize=3, lw=1)
            ax.annotate(t["place"] + ("" if t["real"] else "\n(not the final measurement)"), (t["ms_median"], 100 * t["accuracy"]),
                        textcoords="offset points", xytext=(10, -4), fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("time per photo, ms (median; bar = fastest to 95th percentile)")
        ax.set_ylabel("correct verdicts, %")
        ax.set_title("Speed against accuracy at the three places", loc="left", fontweight="bold", fontsize=11)
        ax.grid(alpha=0.3, which="both")
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo / 1.6, hi * 4)
        fig.tight_layout()
        fig.savefig(os.path.join(res, "figures", "tradeoff.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("figure skipped:", e)
    print("\n".join(L[:len(L) - 2]))
    print("\nsaved results/TRADEOFF.md, results/tradeoff.json, results/figures/tradeoff.png")


if __name__ == "__main__":
    main()
