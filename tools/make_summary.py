"""
Write the results summary into README.md and docs/PROJECT_OVERVIEW.md straight from the
result files (so the numbers in the docs are never typed by hand).

    python -m tools.make_summary        (after python -m sim.experiments)
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from common.config import project_path


def pct(x):
    return f"{100 * x:.1f}%"


def build() -> tuple[str, str]:
    from common.rl import load_card
    card = load_card(project_path("models/model_card.json"))
    e1 = pd.read_csv(project_path("results/E1_steady.csv"))
    e2 = pd.read_csv(project_path("results/E2_changes.csv"))
    e5p = project_path("results/E5_mismatch.csv")
    e5 = pd.read_csv(e5p) if os.path.exists(e5p) else None

    # ---- vision
    vrows = []
    for role, label in (("edge", "ESP32 (edge)"), ("gateway", "Gateway laptop"), ("cloud", "Cloud server")):
        c = card.get(role, {})
        t = c.get("test") or c.get("val") or {}
        vrows.append(f"| {label} | {c.get('arch','?')} @ {c.get('img','?')} px | {pct(t.get('accuracy', 0))} | "
                     f"{pct(t.get('recall_damaged', 0))} | {pct(t.get('false_reject_rate', 0))} | {c.get('threshold', 0.5):.2f} |")
    data = str((card.get("gateway") or {}).get("data", "")).replace("\\", "/")
    kind = ("own-belt photos" if "real" in data else "real photos from a public set, not this project's belt" if "public" in data
            else "computer-drawn boxes")
    vis = (f"**The three vision models** (held-out test photos: {kind})\n\n"
           "| Model | Backbone | Accuracy | Damaged caught | Good wrongly rejected | Threshold |\n|---|---|---|---|---|---|\n"
           + "\n".join(vrows))

    # ---- E1
    avg = e1.groupby("policy")["reward_mean"].mean().sort_values(ascending=False)
    miss = e1.groupby("policy")["miss_rate"].mean()
    esc = e1.groupby("policy")["escape_rate"].mean()
    frr = e1.groupby("policy")["false_reject_rate"].mean()
    rows = "\n".join(f"| {p} | {avg[p]:.2f} | {pct(miss[p])} | {pct(esc[p])} | {pct(frr[p])} |" for p in avg.index)
    calls = e1.groupby("policy")["cloud_share"].mean()
    rows = "\n".join(f"| {p} | {avg[p]:.2f} | {pct(miss[p])} | {pct(calls[p])} | {pct(esc[p])} | {pct(frr[p])} |" for p in avg.index)
    e1_md = (f"**E1 — {e1['scenario'].nunique()} network/device situations, 20 runs × 400 boxes each (simulation; every box "
             "is a real test photo, reward = minus the real cost of what happened to it, cloud calls limited)**\n\n"
             "| Policy | Avg reward / box (higher = better) | Boxes missed | Boxes sent to the cloud | Damaged got through | "
             "Good boxes wrongly rejected |\n|---|---|---|---|---|---|\n" + rows)

    # ---- E2
    pick = e2[e2["change"].str.contains("stalls|overloaded|bandwidth", case=False)]
    rows = "\n".join(f"| {r['change']} | {r['policy']} | {r['missed_during_%']:.1f}% | {r['boxes_to_recover']} |"
                     for _, r in pick.iterrows()
                     if r["policy"] in ("Threshold rule", "Deadline rule", "Adaptive rule", "Confidence rule + reserve",
                                        "Q-learning (online)"))
    e2_md = ("**E2 — problem appears in the middle (boxes 301–600), nothing re-tuned**\n\n"
             "| Change | Policy | Missed during the problem | Boxes to recover |\n|---|---|---|---|\n" + rows)

    e5_md = ""
    if e5 is not None:
        r = {row["policy"]: row for _, row in e5.iterrows()}
        keys = [k for k in r if not k.endswith("(reference)")]
        e5_md = ("**E5 — 'real' hardware 2–3× slower than the simulator assumed:** "
                 + ", ".join(f"{k} {r[k]['reward_all']:.2f}" for k in keys)
                 + " (reward/box). Measured features (ping, test-image time, time left) absorb such differences, "
                   "so no method needs re-tuning; the fixed rules are hurt most.")

    best = avg.index[0]
    q_on = avg.get("Q-learning (online)")
    rules = avg[[p for p in avg.index if not p.startswith("Q-learning")]]
    n_s = e1["scenario"].nunique()
    if q_on is not None and len(rules):
        d = q_on - rules.iloc[0]
        verdict = (f"Q-learning (online) {q_on:.2f} against the best hand-written rule ({rules.index[0]}) {rules.iloc[0]:.2f}: "
                   + ("**level**, no real difference." if abs(d) < 0.05 else
                      f"**Q-learning ahead by {d:.2f}**: real, but small." if d > 0 else f"**the rule is ahead by {-d:.2f}**."))
        if "Adaptive rule" in avg.index and rules.index[0] != "Adaptive rule":
            verdict += (f" Against the best rule that picks ONE place per box (Adaptive rule, {avg['Adaptive rule']:.2f}) "
                        f"Q-learning is ahead by {q_on - avg['Adaptive rule']:.2f}.")
    else:
        verdict = f"Best: **{best} ({avg.iloc[0]:.2f})**."
    head = (f"Average over all {n_s} situations: {verdict} "
            f"Figures: `results/figures/E1_steady_comparison.png`, `E2_change_in_middle.png`, "
            f"`E3_pretraining_value.png`, `E4_policy_map.png`, `E5_sim_to_real_mismatch.png`.")
    trade = ""
    tp = project_path("results/tradeoff.json")
    if os.path.exists(tp):
        t = json.load(open(tp, encoding="utf-8"))
        rows = "\n".join(f"| {x['place']} | {pct(x['accuracy'])} | {x['ms_median']:.1f} ms | {x['measured']} |" for x in t["tiers"])
        trade = (f"**Speed against accuracy** (the same {t['n']} photos at all three places, measured {t['generated']})\n\n"
                 "| Place | Correct verdicts | Time per photo (median) | How it was measured |\n|---|---|---|---|\n" + rows
                 + "\n\n" + "\n".join(f"- {r}" for r in t.get("reading", []) + t.get("notes", [])))
    casc = ""
    cp = project_path("results/cascade.json")
    if os.path.exists(cp):
        c = json.load(open(cp, encoding="utf-8"))
        ag, cv = c.get("agreement", {}), {r["share_pct"]: r for r in c.get("curve", [])}
        lv = "\n".join(f"| {x['name']} | {pct(x['share'])} | {pct(x['gateway_right'] or 0)} | {pct(x['cloud_right'] or 0)} | "
                       f"{x['gain_per_call']:+.2f} |" for x in reversed(c.get("levels", [])))
        casc = (f"**Second opinion on real photos** ({c['n_test']} test photos, {c.get('photo_kind', '?')}; measured {c['generated']})\n\n"
                f"- The gateway's and the cloud's model give the same verdict on {pct(ag.get('same_verdict', 0))} of photos. "
                f"The cloud corrects the gateway on {pct(ag.get('cloud_fixes_gateway', 0))} and makes it worse on "
                f"{pct(ag.get('cloud_makes_it_worse', 0))}.\n"
                + (f"- Sending only the least-sure 25% of boxes to the cloud: {pct(cv[25]['accuracy'])} correct "
                   f"(all boxes: {pct(cv[100]['accuracy'])}, none: {pct(cv[0]['accuracy'])}; the same 25% picked at random: "
                   f"{pct(cv[25]['accuracy_random'])}).\n" if 25 in cv and 100 in cv and 0 in cv else "")
                + "\n| Gateway's model is | Share of boxes | Gateway right | Cloud right | Cost saved per cloud call |\n"
                  "|---|---|---|---|---|\n" + lv)
    e7_md = ""
    e7p = project_path("results/E7_budget.csv")
    if os.path.exists(e7p):
        e7 = pd.read_csv(e7p)
        g = e7[e7["network"] == "good"]
        buds = list(dict.fromkeys(g["cloud_calls_per_min"]))
        hdr = "| Policy | " + " | ".join("no limit" if b == 0 else f"{int(b)}/min" for b in buds) + " |"
        body = "\n".join("| " + p + " | " + " | ".join(
            f"{-float(g[(g['policy'] == p) & (g['cloud_calls_per_min'] == b)]['reward_mean'].iloc[0]):.2f}" for b in buds) + " |"
            for p in dict.fromkeys(g["policy"]))
        e7_md = ("**E7 — cost per box with fewer cloud calls allowed (good network, simulation, lower = better)**\n\n"
                 + hdr + "\n|---" * (len(buds) + 1) + "|\n" + body)
    parts = [x for x in (trade, casc, vis, e1_md, e7_md, e2_md) if x]
    readme = "\n\n".join(parts + [head, "Full tables: [results/TRADEOFF.md](results/TRADEOFF.md), "
                                          "[results/SIM_RESULTS.md](results/SIM_RESULTS.md)"])
    overview = "\n\n".join(parts + [e5_md, head])
    return readme, overview


def inject(path, marker, text):
    s = open(path, encoding="utf-8").read()
    start = f"<!-- {marker} -->"
    end = f"<!-- /{marker} -->"
    block = f"{start}\n{text}\n{end}"
    if end in s:
        s = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda m: block, s, flags=re.S)
    else:
        s = s.replace(start, block)
    open(path, "w", encoding="utf-8").write(s)


def main():
    readme, overview = build()
    inject(project_path("README.md"), "README_RESULTS", readme)
    inject(project_path("docs/PROJECT_OVERVIEW.md"), "RESULTS_SUMMARY", overview)
    print(readme)


if __name__ == "__main__":
    main()
