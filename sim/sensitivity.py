"""
E6 - Does the conclusion survive other assumptions?  (answers two fair objections)

  "30% damaged boxes is unrealistic"      -> repeat at 1% and 5% damaged
  "you chose the costs 100 / 10 / 5"      -> repeat with other cost ratios

For EVERY setting the whole chain is redone, exactly as a careful engineer would:
  1. each vision model picks a new decision threshold for that damage rate and those costs
     (on a validation image set), and its recall / false-reject rate is measured on a test set;
  2. the Q-agent is pre-trained again with the new reward;
  3. the baseline rules are tuned again on the same random conditions;
  4. all policies run on the network / device situations of E1.

    python -m sim.sensitivity --data data/synth          (val/ and test/ folders with good/ damaged/)
    python -m sim.sensitivity --data data/synth --quick  (fewer runs, ~3 min)

Outputs: results/E6_sensitivity.csv, results/E6_sensitivity.json, results/E6_SENSITIVITY.md,
         results/figures/E6_sensitivity.png
Everything here is SIMULATED.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time

import numpy as np

from common.algos import cost_optimal_threshold
from common.config import load_config, project_path
from common.rl import RewardModel
from .env import summarize
from .experiments import factories, with_budget
from .scenarios import LOSSY, STALLS, STEADY
from .train_q import attach_photos, build, make_sim, pretrain, tune_rules

# name, damaged share, cost of an escaped defect, cost of a false reject, cost of an unchecked box (handling)
SETTINGS = [
    ("1% damaged", 0.01, 100.0, 10.0, 5.0),
    ("5% damaged", 0.05, 100.0, 10.0, 5.0),
    ("30% damaged (base)", 0.30, 100.0, 10.0, 5.0),
    ("base, wasted good box costs 3x", 0.30, 100.0, 30.0, 5.0),
    ("base, escaped defect costs 3x", 0.30, 300.0, 10.0, 5.0),
]


def model_scores(onnx_path: str, folder: str, limit: int = 0):
    """P(damaged) of every image in folder/good and folder/damaged."""
    from PIL import Image
    from common.vision_io import OnnxInspector
    m = OnnxInspector(onnx_path, threads=1, providers=["CPUExecutionProvider"])
    ys, ps = [], []
    for label, sub in ((0, "good"), (1, "damaged")):
        d = os.path.join(folder, sub)
        names = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if limit:
            names = names[:limit]
        for f in names:
            p, _, _ = m.predict(Image.open(os.path.join(d, f)).convert("RGB"))
            ys.append(label)
            ps.append(p)
    return np.asarray(ys), np.asarray(ps)


def tiny_scores(path: str, folder: str):
    """P(damaged) from the ESP32's tiny model (same arithmetic as the firmware) for folder/good and folder/damaged."""
    from PIL import Image
    from common import tiny
    m = tiny.TinyModel.load(path)
    ys, ps = [], []
    for label, sub in ((0, "good"), (1, "damaged")):
        d = os.path.join(folder, sub)
        for f in sorted(x for x in os.listdir(d) if x.lower().endswith((".jpg", ".jpeg", ".png"))):
            with Image.open(os.path.join(d, f)) as im:
                im = im.convert("RGB")
                if im.size != (256, 256):
                    im = im.resize((256, 256), Image.BILINEAR)
                ps.append(m.predict(im))
            ys.append(label)
    return np.asarray(ys), np.asarray(ps)


def metrics_at(ys, ps, thr: float) -> dict:
    pred = ps >= thr
    dam, good = ys == 1, ys == 0
    return {"recall_damaged": float(pred[dam].mean()), "false_reject_rate": float(pred[good].mean()),
            "accuracy": float((pred == dam).mean()), "threshold": float(thr)}


def auc(ys, ps) -> float:
    order = np.argsort(ps)
    ranks = np.empty(len(ps))
    ranks[order] = np.arange(1, len(ps) + 1)
    n1, n0 = int((ys == 1).sum()), int((ys == 0).sum())
    return float((ranks[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run_setting(args):
    (name, p_def, c_esc, c_fr, c_un), cfg, scores, episodes, seeds, boxes, fixed = args
    t0 = time.time()
    cfg = copy.deepcopy(cfg)
    cfg["reward"].update({"cost_escape": c_esc, "cost_false_reject": c_fr, "cost_unchecked": c_un})
    cfg["conveyor"]["defect_rate_assumed"] = p_def
    met = {}
    for role in ("edge", "gateway", "cloud"):
        if role not in scores:                                          # no tiny model file: keep the card's numbers
            met[role] = dict(fixed[role], threshold=float("nan"))
            continue
        vy, vp, ty, tp = scores[role]
        thr = cost_optimal_threshold(vy, vp, p_def, c_esc, c_fr)       # picked on validation images
        met[role] = metrics_at(ty, tp, thr)                             # measured on test images
    params, base_reward, space = build(cfg)
    params.defect_rate = p_def
    reward = RewardModel(cfg["reward"], p_def, met["gateway"], met["cloud"], met["edge"])
    tbl = getattr(base_reward, "photos", None)
    if tbl is not None and not tbl.source.startswith("made-up"):       # real photo scores: use this setting's cut-offs
        tbl = tbl.with_thresholds({r: m["threshold"] for r, m in met.items() if m["threshold"] == m["threshold"]})
        attach_photos(cfg, reward, tbl)
    else:
        attach_photos(cfg, reward)
    agent, _ = pretrain(params, reward, space, episodes, 150, 0, gamma=cfg["agent"]["gamma"], log_every=10 ** 9)
    agent.alpha, agent.epsilon = cfg["agent"]["alpha"], cfg["agent"]["epsilon"]
    agent.alpha_mode = "constant"
    agent.explore_gap = cfg["agent"].get("explore_gap")
    agent.forget = float(cfg["agent"].get("forget", 0.0))
    agent.q0 = agent.q.copy()
    rp, _ = tune_rules(params, reward, n_eval=120)
    facs = factories(cfg, agent, space, rp)
    conds = [with_budget(cfg, c) for c in STEADY + [LOSSY, STALLS]]
    rows = []
    for pol, mk in facs.items():
        per = []
        for c in conds:
            for sd in range(seeds):
                log = make_sim(params, reward, np.random.default_rng(1000 + sd)).run(mk(sd), [(0, c)], boxes,
                                                                                    log_conditions=False)
                per.append(summarize(log))
        rows.append({"setting": name, "p_damaged": p_def, "cost_escape": c_esc, "cost_false_reject": c_fr,
                     "cost_unchecked": c_un, "policy": pol,
                     "reward_mean": float(np.mean([s["reward_mean"] for s in per])),
                     "miss_rate": float(np.mean([s["miss_rate"] for s in per])),
                     "escape_rate": float(np.mean([s["escape_rate"] for s in per])),
                     "false_reject_rate": float(np.mean([s["false_reject_rate"] for s in per])),
                     "cloud_share": float(np.mean([s["cloud_share"] for s in per]))})
    d = reward.describe()
    info = {"setting": name, "p_damaged": p_def, "cost_escape": c_esc, "cost_false_reject": c_fr,
            "cost_unchecked": c_un, "edge": met["edge"], "gateway": met["gateway"], "cloud": met["cloud"],
            "reward_edge_on_time": d["edge_on_time"], "reward_gateway_on_time": d["gateway_on_time"],
            "reward_cloud_on_time": d["cloud_on_time"],
            "reward_missed": d["missed"], "seconds": round(time.time() - t0)}
    print(f"[E6] {name}: done in {info['seconds']} s", flush=True)
    return info, rows


def write_outputs(infos, rows, aucs, n_val, n_test):
    res = project_path("results")
    os.makedirs(os.path.join(res, "figures"), exist_ok=True)
    with open(os.path.join(res, "E6_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = []
    for info in infos:
        rs = sorted((r for r in rows if r["setting"] == info["setting"]), key=lambda r: -r["reward_mean"])
        q = next(r for r in rs if r["policy"] == "Q-learning (online)")
        rule = max((r for r in rs if not r["policy"].startswith("Q-learning")), key=lambda r: r["reward_mean"])
        summary.append({**info, "best_policy": rs[0]["policy"], "best_reward": rs[0]["reward_mean"],
                        "q_online_reward": q["reward_mean"], "q_online_miss": q["miss_rate"],
                        "best_rule": rule["policy"], "best_rule_reward": rule["reward_mean"],
                        "best_rule_miss": rule["miss_rate"],
                        "q_minus_best_rule": q["reward_mean"] - rule["reward_mean"]})
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "auc": aucs, "n_val": n_val, "n_test": n_test,
           "settings": summary, "rows": rows}
    with open(os.path.join(res, "E6_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=1)

    pols = [r["policy"] for r in rows if r["setting"] == infos[0]["setting"]]
    L = ["# E6 - Does the conclusion survive other assumptions? (simulated)", "",
         f"Generated: {out['generated']}. For every setting the model thresholds were re-picked, the agent re-trained "
         "and the rules re-tuned. Scores are average reward per box over the situations of E1 "
         "(higher is better; compare within a row only, the cost scale changes between rows).", "",
         f"Image sets used for the thresholds: {n_val} validation, {n_test} test images. AUC (how well each model "
         f"separates good from damaged, 1.0 = perfectly, 0.5 = guessing): " +
         ", ".join(f"{k} {v:.3f}" for k, v in aucs.items()) + ".", "",
         "| setting | " + " | ".join(pols) + " | best | Q online minus best rule |",
         "|---|" + "---|" * (len(pols) + 2)]
    for s in summary:
        rs = {r["policy"]: r for r in rows if r["setting"] == s["setting"]}
        best = max(rs.values(), key=lambda r: r["reward_mean"])["policy"]
        cells = [("**%.2f**" if p == best else "%.2f") % rs[p]["reward_mean"] for p in pols]
        L.append(f"| {s['setting']} | " + " | ".join(cells) + f" | {s['best_policy']} | {s['q_minus_best_rule']:+.2f} |")
    L += ["", "**Model operating points picked for each setting** (threshold / damaged caught / good wrongly rejected)", "",
          "| setting | ESP32 model | gateway model | cloud model | reward: ESP32 in time | gateway in time | cloud in time | missed |",
          "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        cell = lambda m: f"{m['threshold']:.2f} / {100*m['recall_damaged']:.1f}% / {100*m['false_reject_rate']:.1f}%"
        L.append(f"| {s['setting']} | {cell(s['edge'])} | {cell(s['gateway'])} | {cell(s['cloud'])} "
                 f"| {s['reward_edge_on_time']:.2f} | {s['reward_gateway_on_time']:.2f} | {s['reward_cloud_on_time']:.2f} "
                 f"| {s['reward_missed']:.2f} |")
    L += ["", "**Missed boxes % (average over the situations)**", "",
          "| setting | " + " | ".join(pols) + " |", "|---|" + "---|" * len(pols)]
    for s in summary:
        rs = {r["policy"]: r for r in rows if r["setting"] == s["setting"]}
        L.append(f"| {s['setting']} | " + " | ".join("%.1f" % (100 * rs[p]["miss_rate"]) for p in pols) + " |")
    with open(os.path.join(res, "E6_SENSITIVITY.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        show = ["Always-Edge", "Always-Gateway", "Always-Cloud", "Adaptive rule", "Q-learning (online)"]
        col = {"Always-Edge": "#d1d5db", "Always-Gateway": "#9ca3af", "Always-Cloud": "#4b5563", "Adaptive rule": "#111827",
               "Q-learning (online)": "#06b6d4"}
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(len(summary))
        wid = 0.8 / len(show)
        for j, p in enumerate(show):
            vals = [next(r["reward_mean"] for r in rows if r["setting"] == s["setting"] and r["policy"] == p) for s in summary]
            ax.bar(x + j * wid, vals, wid, label=p, color=col[p])
        ax.set_xticks(x + wid * (len(show) - 1) / 2)
        ax.set_xticklabels([s["setting"] for s in summary], fontsize=8)
        ax.set_ylabel("reward per box (higher is better)")
        ax.set_title("E6: the comparison under other damage rates and costs (simulated)", loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(res, "figures", "E6_sensitivity.png"), dpi=150)
        plt.close(fig)
    except Exception as e:                      # the tables are what matters
        print("figure skipped:", e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default="data/synth", help="folder with val/ and test/ (each: good/ damaged/)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cfg = load_config(a.config)
    data = a.data if os.path.isabs(a.data) else project_path(a.data)
    print("[E6] scoring the validation and test images with both models ...", flush=True)
    scores, aucs = {}, {}
    from common.rl import load_card
    fixed = RewardModel.metrics_from_card(load_card(project_path(cfg["models"]["card"])))
    tiny_path = project_path(cfg["models"].get("edge_tiny", "models/edge_tiny.bin"))
    if os.path.exists(tiny_path):
        vy, vp = tiny_scores(tiny_path, os.path.join(data, "val"))
        ty, tp = tiny_scores(tiny_path, os.path.join(data, "test"))
        scores["edge"] = (vy, vp, ty, tp)
        aucs["esp32"] = auc(ty, tp)
        print(f"  esp32: {len(vy)} val + {len(ty)} test images, test AUC {aucs['esp32']:.3f}", flush=True)
    for role, key in (("gateway", "gateway_onnx"), ("cloud", "cloud_onnx")):
        vy, vp = model_scores(project_path(cfg["models"][key]), os.path.join(data, "val"))
        ty, tp = model_scores(project_path(cfg["models"][key]), os.path.join(data, "test"))
        scores[role] = (vy, vp, ty, tp)
        aucs[role] = auc(ty, tp)
        print(f"  {role}: {len(vy)} val + {len(ty)} test images, test AUC {aucs[role]:.3f}", flush=True)
    episodes, seeds, boxes = (2500, 5, 300) if a.quick else (6000, 10, 400)
    jobs = [(s, cfg, scores, episodes, seeds, boxes, fixed) for s in SETTINGS]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            done = pool.map(run_setting, jobs, chunksize=1)
    else:
        done = [run_setting(j) for j in jobs]
    infos = [d[0] for d in done]
    rows = [r for d in done for r in d[1]]
    out = write_outputs(infos, rows, aucs, len(scores["gateway"][0]), len(scores["gateway"][2]))
    for s in out["settings"]:
        print(f"  {s['setting']:34s} best: {s['best_policy']:22s} Q online {s['q_online_reward']:.2f}  "
              f"best rule ({s['best_rule']}) {s['best_rule_reward']:.2f}  diff {s['q_minus_best_rule']:+.2f}")
    print("saved results/E6_sensitivity.csv, .json, E6_SENSITIVITY.md, figures/E6_sensitivity.png")


if __name__ == "__main__":
    main()
