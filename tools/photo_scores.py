"""
Score every validation and test photo with all three models and keep the scores.

    python -m tools.photo_scores                 # photos = the data the models were trained on
    python -m tools.photo_scores --data data/public

Why: the gateway's model checks every box first. Its score tells how sure it is, and the cloud is only
worth asking when it is unsure. To train and test that honestly the simulator replays REAL photos:
each simulated box is one real photo, and every model's verdict on it is that model's real verdict.

Writes
  models/photo_scores.npz      the scores (validation photos train the agent, test photos judge it)
  models/model_card.json       section "cascade": confidence levels and how often each model is wrong per level
  results/cascade.json         on the TEST photos: accuracy and cost when only the least-sure X % go to the cloud
  results/CASCADE.md           the same as a table
"""
from __future__ import annotations

import argparse
import json
import os
import time


from common.cascade import CONF_NAMES, PhotoSplit, PhotoTable, escalation_curve, level_costs
from common.config import load_config, project_path
from common.rl import load_card
from sim.sensitivity import model_scores, tiny_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", default=None, help="folder with val/ and test/ (each with good/ and damaged/)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    card_path = project_path(cfg["models"]["card"])
    card = load_card(card_path)
    data = a.data or str((card.get("gateway") or {}).get("data") or "") or "data/synthetic"
    data = data if os.path.isabs(data) else project_path(data)
    for part in ("val", "test"):
        if not os.path.isdir(os.path.join(data, part, "good")):
            raise SystemExit(f"[scores] {os.path.join(data, part)} has no good/ and damaged/ folders.")
    thr = {r: float((card.get(r) or {}).get("threshold", cfg["models"]["damaged_threshold"])) for r in ("edge", "gateway", "cloud")}
    tiny_path = project_path(cfg["models"].get("edge_tiny", "models/edge_tiny.bin"))
    paths = {"gateway": project_path(cfg["models"]["gateway_onnx"]), "cloud": project_path(cfg["models"]["cloud_onnx"])}

    t0 = time.time()
    splits = {}
    for part in ("val", "test"):
        folder = os.path.join(data, part)
        scores, y = {}, None
        if os.path.exists(tiny_path):
            y, scores["edge"] = tiny_scores(tiny_path, folder)
        for role in ("gateway", "cloud"):
            yy, scores[role] = model_scores(paths[role], folder)
            if y is not None and len(yy) != len(y):
                raise SystemExit("[scores] the models saw a different number of photos - check the photo folders.")
            y = yy
        splits[part] = (y, scores)
        print(f"[scores] {part}: {len(y)} photos scored by {', '.join(scores)}", flush=True)
    use = {r: thr[r] for r in splits["val"][1]}
    val = PhotoSplit(splits["val"][0], splits["val"][1], use)
    test = PhotoSplit(splits["test"][0], splits["test"][1], use, val.edges)
    kind = "own" if "real" in data.replace("\\", "/") else ("public" if "public" in data.replace("\\", "/") else "synthetic")
    table = PhotoTable(val, test, f"{kind} photos, {os.path.relpath(data, project_path('.'))}")
    out_npz = project_path(cfg["models"].get("photo_scores", "models/photo_scores.npz"))
    table.save(out_npz, {"made": time.strftime("%Y-%m-%d %H:%M"), "data": os.path.relpath(data, project_path("."))})

    # ---- model card: confidence levels (from the validation photos only)
    st = val.stats()
    st["made"] = time.strftime("%Y-%m-%d %H:%M")
    st["photo_kind"] = kind
    card["cascade"] = st
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=1)

    # ---- what the TEST photos say
    p_def = float(cfg["conveyor"]["defect_rate_assumed"])
    c_esc, c_fr = float(cfg["reward"]["cost_escape"]), float(cfg["reward"]["cost_false_reject"])
    curve = escalation_curve(test, p_def, c_esc, c_fr)
    vg, vc, y = test.verdict["gateway"], test.verdict["cloud"], test.y
    g_ok, c_ok = vg == y, vc == y
    agree = {"same_verdict": float((vg == vc).mean()), "cloud_fixes_gateway": float((~g_ok & c_ok).mean()),
             "cloud_makes_it_worse": float((g_ok & ~c_ok).mean()), "both_wrong": float((~g_ok & ~c_ok).mean())}
    lv_test = level_costs(test.stats(), p_def, c_esc, c_fr)
    lv_val = level_costs(st, p_def, c_esc, c_fr)
    levels = []
    for c in range(1, len(CONF_NAMES)):
        mask = test.conf == c
        levels.append({"level": c, "name": CONF_NAMES[c], "n": int(mask.sum()), "share": lv_test["share"][c],
                       "gateway_right": float(g_ok[mask].mean()) if mask.any() else None,
                       "cloud_right": float(c_ok[mask].mean()) if mask.any() else None,
                       "cost_gateway": lv_test["gateway"][c], "cost_cloud": lv_test["cloud"][c],
                       "gain_per_call": lv_test["gateway"][c] - lv_test["cloud"][c],
                       "gain_per_call_val": lv_val["gateway"][c] - lv_val["cloud"][c]})
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "photos": os.path.relpath(data, project_path(".")),
           "photo_kind": kind, "n_val": len(val), "n_test": len(test), "p_damaged_assumed": p_def,
           "edges": val.edges, "thresholds": use, "agreement": agree, "levels": levels, "curve": curve}
    rdir = project_path("results")
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "cascade.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    L = ["# Second opinion: which boxes are worth a cloud call", "",
         f"Generated {res['generated']}. {len(test)} test photos ({kind}), never used for training or for choosing "
         f"the confidence levels (those come from the {len(val)} validation photos). Costs assume {100*p_def:.0f}% "
         f"damaged boxes, {c_esc:.0f} for a shipped damaged box and {c_fr:.0f} for a wasted good one.", "",
         f"- The gateway's and the cloud's model give the same verdict on {100*agree['same_verdict']:.1f}% of photos.",
         f"- The cloud corrects the gateway on {100*agree['cloud_fixes_gateway']:.1f}% and makes it worse on "
         f"{100*agree['cloud_makes_it_worse']:.1f}%. Both are wrong on {100*agree['both_wrong']:.1f}%.", "",
         "## How sure the gateway is, and what a cloud call is worth", "",
         "| Gateway is | Share of boxes | Gateway right | Cloud right | Cost if kept | Cost if cloud asked | Gain per cloud call |",
         "|---|---|---|---|---|---|---|"]
    for lv in reversed(levels):
        L.append(f"| {lv['name']} | {100*lv['share']:.0f}% | {100*(lv['gateway_right'] or 0):.1f}% | "
                 f"{100*(lv['cloud_right'] or 0):.1f}% | {lv['cost_gateway']:.2f} | {lv['cost_cloud']:.2f} | "
                 f"{lv['gain_per_call']:+.2f} |")
    L += ["", "## Send only the least-sure boxes to the cloud", "",
          "| Sent to the cloud | Correct | Cost per box | Same share picked at random: correct | cost |", "|---|---|---|---|---|"]
    for r in curve:
        L.append(f"| {r['share_pct']}% | {100*r['accuracy']:.1f}% | {r['cost']:.2f} | {100*r['accuracy_random']:.1f}% | "
                 f"{r['cost_random']:.2f} |")
    L += ["", f"With {len(test)} photos a difference of about 1.5 points in 'correct' is within chance."]
    with open(os.path.join(rdir, "CASCADE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[4:8]))
    for lv in reversed(levels):
        print(f"  {lv['name']:<12} share {100*lv['share']:4.0f}%  gateway right {100*(lv['gateway_right'] or 0):5.1f}%  "
              f"cloud right {100*(lv['cloud_right'] or 0):5.1f}%  gain per call {lv['gain_per_call']:+.2f}")
    print(f"[scores] saved {os.path.relpath(out_npz, project_path('.'))}, results/cascade.json, results/CASCADE.md "
          f"({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
