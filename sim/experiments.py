"""
Run every simulation experiment for the report and save tables + graphs.

    python -m sim.experiments            (takes ~2-4 minutes)

E1  Steady conditions    : 10 situations x 8 policies x 20 runs
E2  Change in the middle : normal -> problem -> normal (900 boxes) - THE key graph
E3  Pre-training value   : pre-trained agent vs agent learning from zero
E4  Policy map           : what the trained agent decided, as a colour map
E5  Simulator was wrong  : hardware slower than assumed, nothing re-tuned
E7  Cloud-call budget    : the same line with no limit, 30, 15, 8 and 4 cloud calls a minute
E8  The photos change    : after 500 boxes every photo gets darker / blurred (the same real photos, scored
                           again by the same models). Who copes, and does the agent correct itself from spot checks?
E9  How sure is a gap    : the comparison repeated on 200 re-drawn test sets -> a range for every difference

Every simulated box is one REAL test photo (see common/cascade.py), and "reward" is minus the real
cost of what happened to that box. All policies see the same boxes in the same order.
The second-opinion policies (ask always, confidence rules, Q-learning) let the gateway's model check
every box first; the others pick one place per box, as before. E1 and E2 run with the cloud-call
limit from config/system.yaml (cloud_budget).

Everything here is SIMULATED. The same graphs are produced from real hardware
logs by tools/report_real.py once the conveyor is built.
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np

from common.algos import first_index_staying_below, first_window_above, rolling_mean
from common.config import load_config, project_path
from dataclasses import replace as _replace

from common.policies import (AdaptiveRule, AlwaysCloud, AlwaysEdge, AlwaysGateway, AskAlways, ConfidenceReserveRule,
                             ConfidenceRule, DeadlineRule, OraclePolicy, QPolicy, ThresholdRule)
from common.cascade import N_CONF, LiveGain
from common.rl import CLOUD, EDGE, GATEWAY, QAgent, StateSpace
from .env import random_schedule, summarize
from .scenarios import BUSY, CHANGES, GOOD, LOSSY, STALLS, STEADY
from .train_q import build, load_shift_tables, make_sim

COLORS = {
    "Always-Edge": "#d1d5db", "Always-Gateway": "#9ca3af", "Always-Cloud": "#6b7280", "Threshold rule": "#f59e0b",
    "Deadline rule": "#8b5cf6", "Adaptive rule": "#0ea5e9",
    "Ask the cloud always": "#fb7185", "Confidence rule": "#0f766e", "Confidence rule + reserve": "#115e59",
    "Best possible (knows the answers)": "#111827",
    "Q-learning (frozen)": "#86efac", "Q-learning (online)": "#16a34a", "Q-learning (from zero)": "#ef4444",
    "Q-learning (online + spot checks)": "#1d4ed8", "Q-learning (online + 1 box in 5 checked)": "#60a5fa",
}
SHIFT_TITLES = {"blur": "blurred (box moving)", "dim": "darker (40 % light)", "both": "darker + blurred + noise"}
FIG = project_path("results/figures")
RES = project_path("results")


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def factories(cfg, agent: QAgent, space: StateSpace, rp: dict):
    def q_online(seed):
        ag = agent.copy(seed)
        ag.alpha, ag.epsilon, ag.alpha_mode = cfg["agent"]["alpha"], cfg["agent"]["epsilon"], "constant"
        ag.explore_gap = cfg["agent"]["explore_gap"]
        ag.forget = cfg["agent"]["forget"]
        return QPolicy(ag, space, learn=True, explore=True, name="Q-learning (online)")

    return {
        "Always-Edge": lambda s: AlwaysEdge(),
        "Always-Gateway": lambda s: AlwaysGateway(),
        "Always-Cloud": lambda s: AlwaysCloud(),
        "Threshold rule": lambda s: ThresholdRule(**rp["threshold"]),
        "Deadline rule": lambda s: DeadlineRule(**rp["deadline"]),
        "Adaptive rule": lambda s: AdaptiveRule(**rp["adaptive"]),
        "Ask the cloud always": lambda s: AskAlways(**rp["adaptive"]),
        "Confidence rule": lambda s: ConfidenceRule(**rp.get("confidence", rp["adaptive"])),
        "Confidence rule + reserve": lambda s: ConfidenceReserveRule(**rp.get("confidence_reserve", rp["adaptive"])),
        "Q-learning (frozen)": lambda s: QPolicy(agent.copy(s), space, learn=False, explore=False,
                                                 name="Q-learning (frozen)"),
        "Q-learning (online)": q_online,
    }


def with_budget(cfg, c):
    """The condition with the cloud-call limit from config/system.yaml."""
    b = cfg.get("cloud_budget") or {}
    return _replace(c, budget_per_min=float(b.get("per_min", 0) or 0), budget_burst=float(b.get("burst", 5) or 5))


# ----------------------------------------------------------------------------
def e1_steady(plt, params, reward, facs, seeds=20, boxes=400, cfg=None):
    conds = STEADY + [LOSSY, STALLS]
    if cfg is not None:
        conds = [with_budget(cfg, c) for c in conds]
    rows = []
    for c in conds:
        for name, mk in facs.items():
            ss = []
            for sd in range(seeds):
                log = make_sim(params, reward, np.random.default_rng(1000 + sd)).run(mk(sd), [(0, c)], boxes,
                                                                                    log_conditions=False)
                ss.append(summarize(log))
            agg = {k: float(np.mean([s[k] for s in ss])) for k in ss[0] if k != "n"}
            agg.update({f"{k}_std": float(np.std([s[k] for s in ss])) for k in ("reward_mean", "miss_rate")})
            rows.append({"scenario": c.name, "policy": name, **agg})
    with open(os.path.join(RES, "E1_steady.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    names = list(facs)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5))
    x = np.arange(len(conds))
    wid = 0.8 / len(names)
    for j, n in enumerate(names):
        sub = [r for r in rows if r["policy"] == n]
        axes[0].bar(x + j * wid, [r["reward_mean"] for r in sub], wid, label=n, color=COLORS[n])
        axes[1].bar(x + j * wid, [100 * r["miss_rate"] for r in sub], wid, color=COLORS[n])
        axes[2].bar(x + j * wid, [100 * r["escape_rate"] for r in sub], wid, color=COLORS[n])
    for a, t in zip(axes, ["Reward per box (higher is better)", "Boxes missed - answer too late (%)",
                           "Damaged boxes that got through (%)"]):
        a.set_xticks(x + wid * (len(names) - 1) / 2)
        a.set_xticklabels([c.name for c in conds])
        a.set_title(t, loc="left", fontweight="bold")
        a.grid(axis="y", alpha=0.3)
    axes[1].set_yscale("symlog", linthresh=1)
    axes[0].legend(ncol=4, fontsize=8, loc="lower left")
    axes[0].set_ylim(bottom=max(-36, axes[0].get_ylim()[0]))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E1_steady_comparison.png"), dpi=150)
    plt.close(fig)
    return rows


# ----------------------------------------------------------------------------
def rolling(x, w):
    """Moving average with prefix sums: O(n) instead of the O(n*w) convolution."""
    return rolling_mean(x, w)


def e2_changes(plt, params, reward, facs, seeds=30, phase=300, win=25, cfg=None):
    names = ["Threshold rule", "Deadline rule", "Adaptive rule", "Confidence rule + reserve", "Q-learning (frozen)",
             "Q-learning (online)"]
    summary = []
    fig, axes = plt.subplots(3, len(CHANGES), figsize=(3.7 * len(CHANGES), 8.6), sharex=True)
    for ci, (title, base, changed) in enumerate(CHANGES):
        if cfg is not None:
            base, changed = with_budget(cfg, base), with_budget(cfg, changed)
        sched = [(0, base), (phase, changed), (2 * phase, base)]
        n = 3 * phase
        for name in names:
            miss = np.zeros(n); cloud = np.zeros(n); rew = np.zeros(n); esp = np.zeros(n)
            for sd in range(seeds):
                log = make_sim(params, reward, np.random.default_rng(5000 + sd)).run(
                    facs[name](sd), sched, n, log_conditions=False).arrays()
                miss += 1 - log["on_time"]; cloud += (log["action"] == CLOUD); rew += log["reward"]
                esp += (log["action"] == EDGE)
            cloud_count = cloud.astype(np.int64)        # boxes sent to the cloud, summed over seeds (exact)
            miss /= seeds; cloud /= seeds; rew /= seeds; esp /= seeds
            xs = np.arange(win - 1, n)
            ls = (0, (5, 3)) if name == "Threshold rule" else "-"     # dashed: it often overlaps the deadline rule
            lw = 2.2 if "online" in name else (1.8 if name == "Threshold rule" else 1.2)
            axes[0, ci].plot(xs, 100 * rolling(miss, win), color=COLORS[name], lw=lw, ls=ls, label=name)
            axes[1, ci].plot(xs, 100 * rolling(cloud, win), color=COLORS[name], lw=lw, ls=ls)
            axes[2, ci].plot(xs, 100 * rolling(esp, win), color=COLORS[name], lw=lw, ls=ls)
            during = miss[phase:2 * phase]
            # boxes after the change until the rolling miss rate stays under 5%
            rm = rolling(during, win)
            # first i from which the rolling miss rate stays below 5%: one backward scan, O(n)
            i0 = first_index_staying_below(rm, 0.05)
            rec = i0 + win if i0 is not None else None
            # first window where >80% of boxes go to the laptop again: prefix sums, O(n)
            after_count = cloud_count[2 * phase:]
            back = first_window_above(after_count, win, 0.8 * seeds, n_starts=len(after_count) - win)
            summary.append({
                "change": title, "policy": name,
                "missed_before_%": 100 * miss[:phase].mean(),
                "missed_during_%": 100 * during.mean(),
                "missed_after_%": 100 * miss[2 * phase:].mean(),
                "boxes_missed_during(of 300)": float(during.sum()),
                "boxes_to_recover": rec if rec is not None else ">300",
                "boxes_to_trust_laptop_again": back if back is not None else ">300",
                "esp32_share_during_%": 100 * esp[phase:2 * phase].mean(),
                "reward_before": rew[:phase].mean(), "reward_during": rew[phase:2 * phase].mean(),
                "reward_after": rew[2 * phase:].mean(),
            })
        for r_ in (0, 1, 2):
            axes[r_, ci].axvspan(phase, 2 * phase, color="#fee2e2", alpha=0.6, lw=0)
            axes[r_, ci].grid(alpha=0.3)
            axes[r_, ci].set_ylim(-2, 102)
        axes[0, ci].set_title(title, fontsize=9, fontweight="bold")
        axes[2, ci].set_xlabel("box number")
    axes[0, 0].set_ylabel("missed boxes, rolling %")
    axes[1, 0].set_ylabel("sent to cloud, rolling %")
    axes[2, 0].set_ylabel("checked on ESP32, rolling %")
    axes[0, 0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Change in the middle (red band = problem present). Nothing is re-tuned by hand.", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E2_change_in_middle.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(RES, "E2_changes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    return summary


# ----------------------------------------------------------------------------
def e3_pretrain_value(plt, cfg, params, reward, agent, space, seeds=30, boxes=3000, win=100):
    rng = np.random.default_rng(777)
    scheds = [random_schedule(rng, boxes, change_prob=0.6) for _ in range(seeds)]
    curves = {}
    for name in ("Q-learning (online)", "Q-learning (from zero)"):
        acc = np.zeros(boxes); mis = np.zeros(boxes)
        for sd in range(seeds):
            if name == "Q-learning (online)":
                ag = agent.copy(sd)
            else:
                ag = QAgent(space.n_states, 3, seed=sd, prefer=GATEWAY)
                d0, g0 = reward.describe(), 1 - cfg["agent"]["gamma"]
                ag.init_prior(d0["gateway_on_time"] / g0, 1.0, edge_value=d0["edge_on_time"] / g0)
            ag.alpha, ag.gamma, ag.epsilon, ag.alpha_mode = cfg["agent"]["alpha"], cfg["agent"]["gamma"], \
                cfg["agent"]["epsilon"], "constant"
            ag.explore_gap = cfg["agent"]["explore_gap"]
            ag.forget = cfg["agent"]["forget"] if name == "Q-learning (online)" else 0.0
            log = make_sim(params, reward, np.random.default_rng(9000 + sd)).run(
                QPolicy(ag, space, True, True), scheds[sd], boxes, log_conditions=False).arrays()
            acc += log["reward"]; mis += 1 - log["on_time"]
        curves[name] = (acc / seeds, mis / seeds)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for name, (rw, ms) in curves.items():
        xs = np.arange(win - 1, boxes)
        ax[0].plot(xs, rolling(rw, win), color=COLORS[name], label=name)
        ax[1].plot(xs, 100 * rolling(ms, win), color=COLORS[name])
    ax[0].set_title("Reward per box (rolling 100)", loc="left", fontweight="bold")
    ax[1].set_title("Missed boxes % (rolling 100)", loc="left", fontweight="bold")
    for a in ax:
        a.set_xlabel("boxes on the conveyor"); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E3_pretraining_value.png"), dpi=150)
    plt.close(fig)
    out = {}
    for name, (rw, ms) in curves.items():
        out[name] = {"reward_first_500": float(rw[:500].mean()), "reward_last_500": float(rw[-500:].mean()),
                     "missed_first_500_%": float(100 * ms[:500].mean()),
                     "missed_last_500_%": float(100 * ms[-500:].mean())}
    with open(os.path.join(RES, "E3_pretraining_value.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# ----------------------------------------------------------------------------
def e4_policy_map(plt, agent: QAgent, space: StateSpace):
    """What the trained agent does. Three panels for the normal case (the gateway's model has checked the box):
    rows = how sure the gateway is, columns = cloud calls left. One panel for 'gateway too slow'."""
    from matplotlib.colors import ListedColormap
    lab = space.labels()
    ns, nc = space.dims[0], space.dims[1]
    n_conf, n_bud = space.dims[5], space.dims[6]
    names = {EDGE: "ESP32", GATEWAY: "keep\n(gateway)", CLOUD: "ask the\ncloud"}
    cmap = ListedColormap(["#fde68a", "#e5e7eb", "#67e8f9"])       # ESP32, gateway, cloud
    cases = [("Good network, lots of spare time", ns - 1, 0, 0), ("Busy network, some spare time", min(5, ns - 1), 0, 1),
             ("Little spare time", min(3, ns - 1), 1, 1)]
    fig, axes = plt.subplots(1, len(cases) + 1, figsize=(4.3 * (len(cases) + 1), 4.4))
    for ax, (title, slack_b, cloud_b, rtt_b) in zip(axes, cases):
        m = np.zeros((n_conf - 1, n_bud)); seen = np.zeros_like(m)
        for i in range(1, n_conf):
            for j in range(n_bud):
                s = int(np.ravel_multi_index((slack_b, cloud_b, rtt_b, 0, 0, i, j), space.dims))
                m[i - 1, j] = agent.greedy(s, (GATEWAY, CLOUD))
                seen[i - 1, j] = agent.visits[s].sum()
        ax.imshow(m, cmap=cmap, vmin=0, vmax=2, aspect="auto", origin="lower")
        for i in range(n_conf - 1):
            for j in range(n_bud):
                txt = names[int(m[i, j])] + ("\n(never seen)" if seen[i, j] == 0 else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5)
        ax.set_xticks(range(n_bud)); ax.set_xticklabels(lab["budget"], fontsize=8)
        ax.set_yticks(range(n_conf - 1)); ax.set_yticklabels(lab["conf"][1:], fontsize=8)
        ax.set_xlabel("cloud calls left"); ax.set_ylabel("gateway's model is ...")
        ax.set_title(title, fontsize=10, fontweight="bold")
    ax = axes[-1]
    m = np.zeros((ns, nc)); seen = np.zeros((ns, nc))
    for i in range(ns):
        for j in range(nc):
            s = int(np.ravel_multi_index((i, j, 1, 2, 0, 0, n_bud - 1), space.dims))
            m[i, j] = agent.greedy(s)
            seen[i, j] = agent.visits[s].sum()
    ax.imshow(m, cmap=cmap, vmin=0, vmax=2, aspect="auto", origin="lower")
    one = {EDGE: "ESP32", GATEWAY: "GATEWAY", CLOUD: "CLOUD"}
    for i in range(ns):
        for j in range(nc):
            ax.text(j, i, one[int(m[i, j])] + ("\n(never seen)" if seen[i, j] == 0 else ""), ha="center", va="center", fontsize=7)
    ax.set_xticks(range(nc)); ax.set_xticklabels(lab["cloud_ms"], fontsize=8)
    ax.set_yticks(range(ns)); ax.set_yticklabels(lab["slack_ms"], fontsize=7)
    ax.set_xlabel("measured cloud response"); ax.set_ylabel("spare time if sent to cloud")
    ax.set_title("Gateway too slow to go first (busy ping)", fontsize=10, fontweight="bold")
    fig.suptitle("What the trained agent does (grey = keep the gateway's verdict, cyan = ask the cloud, yellow = ESP32)", y=1.02)
    fig.savefig(os.path.join(FIG, "E4_policy_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def e7_budget(plt, params, reward, facs, rp, seeds=20, boxes=400):
    """The same line with different cloud-call limits. Who spends a small budget best?"""
    pols = {k: facs[k] for k in ("Adaptive rule", "Ask the cloud always", "Confidence rule", "Confidence rule + reserve",
                                 "Q-learning (frozen)", "Q-learning (online)")}
    pols["Best possible (knows the answers)"] = lambda s: OraclePolicy(**rp.get("confidence", rp["adaptive"]))
    budgets = [0, 30, 15, 8, 4]
    rows = []
    for base in (GOOD, BUSY):
        for b in budgets:
            c = _replace(base, budget_per_min=float(b), budget_burst=5.0)
            for name, mk in pols.items():
                ss = []
                for sd in range(seeds):
                    log = make_sim(params, reward, np.random.default_rng(3000 + sd)).run(mk(sd), [(0, c)], boxes,
                                                                                        log_conditions=False)
                    ss.append(summarize(log))
                rows.append({"network": base.name, "cloud_calls_per_min": b, "policy": name,
                             **{k: float(np.mean([x[k] for x in ss])) for k in
                                ("reward_mean", "accuracy", "miss_rate", "cloud_share", "ask_share", "fallback_share",
                                 "escape_rate", "false_reject_rate")}})
    with open(os.path.join(RES, "E7_budget.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, net in zip(axes, ("good", "busy")):
        for name in pols:
            sub = [r for r in rows if r["network"] == net and r["policy"] == name]
            ax.plot(range(len(budgets)), [-r["reward_mean"] for r in sub], marker="o", ms=3.5,
                    color=COLORS.get(name, "#000"), lw=2.2 if "online" in name else 1.3,
                    ls="--" if name.startswith("Best possible") else "-", label=name)
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels(["no limit" if b == 0 else str(b) for b in budgets])
        ax.set_xlabel("cloud calls allowed per minute (about 60 boxes a minute)")
        ax.set_title(f"{net} network", loc="left", fontweight="bold")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cost per box (lower is better)")
    axes[1].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E7_budget.png"), dpi=150)
    plt.close(fig)
    return rows


# ----------------------------------------------------------------------------
def spot_policy(cfg, reward, facs, name="Q-learning (online + spot checks)"):
    """The online agent plus the spot-check tally (common.cascade.LiveGain)."""
    card_gain = [0.0] + [reward.gain(c) for c in range(1, N_CONF)]
    a = cfg["agent"]

    def mk(seed):
        pol = facs["Q-learning (online)"](seed)
        pol.live = LiveGain(card_gain, float(a.get("spot_prior_n", 60)), float(a.get("spot_half_life", 300)))
        pol.name = name
        return pol
    return mk


def e8_photos_change(plt, cfg, params, reward, facs, rp, seeds=20, before=500, after=3000, win=300):
    """Box 1-500: the normal photos. From box 501 on every photo is darker / blurred: the SAME real photos,
    scored again by the SAME models (tools/shift_scores.py). Nothing is re-tuned. The policies were all
    built on clean photos; only the two 'spot checks' lines get to see the true state of boxes afterwards
    (every box, or 1 box in 5) and correct what they think a cloud answer is worth."""
    shifts = [k for k in ("blur", "dim", "both") if k in (getattr(reward, "photo_shift", None) or {})]
    if not shifts:
        print("    (skipped: no models/photo_scores_blur.npz ... run  python -m tools.shift_scores  first)")
        return None
    bud = float((cfg.get("cloud_budget") or {}).get("per_min", 15) or 15)
    pols = {k: (facs[k], 0.0) for k in ("Adaptive rule", "Ask the cloud always", "Confidence rule",
                                        "Confidence rule + reserve", "Q-learning (frozen)", "Q-learning (online)")}
    pols["Q-learning (online + spot checks)"] = (spot_policy(cfg, reward, facs), 1.0)
    pols["Q-learning (online + 1 box in 5 checked)"] = (spot_policy(cfg, reward, facs, "Q-learning (online + 1 box in 5 checked)"), 0.2)
    pols["Best possible (knows the answers)"] = (lambda s: OraclePolicy(**rp.get("confidence", rp["adaptive"])), 0.0)
    n = before + after
    half = after // 2
    rows, curves = [], {}
    for sh in shifts:
        for b in (bud, 0.0):
            base = _replace(GOOD, budget_per_min=b, budget_burst=5.0)
            sched = [(0, base), (before, _replace(base, photos=sh, name=f"{base.name}+{sh}"))]
            for name, (mk, spot) in pols.items():
                cost = np.zeros(n)
                agg = {"before": [], "first": [], "last": [], "ask_last": [], "acc_last": []}
                for sd in range(seeds):
                    log = make_sim(params, reward, np.random.default_rng(8000 + sd), spot_rate=spot).run(
                        mk(sd), sched, n, log_conditions=False)
                    cost += -np.asarray(log.reward)
                    agg["before"].append(-summarize(log, 0, before)["reward_mean"])
                    agg["first"].append(-summarize(log, before, before + 500)["reward_mean"])
                    last = summarize(log, n - half, n)
                    agg["last"].append(-last["reward_mean"])
                    agg["ask_last"].append(last["ask_share"])
                    agg["acc_last"].append(last["accuracy"])
                curves[(sh, b, name)] = cost / seeds
                rows.append({"photos": sh, "cloud_calls_per_min": b, "policy": name,
                             "cost_before": float(np.mean(agg["before"])), "cost_first_500": float(np.mean(agg["first"])),
                             "cost_last": float(np.mean(agg["last"])), "cost_last_std": float(np.std(agg["last"])),
                             "ask_share_last": float(np.mean(agg["ask_last"])), "accuracy_last": float(np.mean(agg["acc_last"]))})
    with open(os.path.join(RES, "E8_photos_change.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    show = ["Ask the cloud always", "Confidence rule", "Q-learning (frozen)", "Q-learning (online + spot checks)",
            "Q-learning (online + 1 box in 5 checked)", "Best possible (knows the answers)"]
    fig, axes = plt.subplots(len(shifts), 2, figsize=(11.5, 3.2 * len(shifts)), sharex=True, squeeze=False)
    for i, sh in enumerate(shifts):
        for j, b in enumerate((bud, 0.0)):
            ax = axes[i, j]
            for name in show:
                ax.plot(np.arange(win - 1, n), rolling(curves[(sh, b, name)], win), color=COLORS.get(name, "#000"),
                        lw=2.2 if name.endswith("spot checks)") else 1.3, ls="--" if name.startswith("Best") else "-", label=name)
            ax.axvspan(before, n, color="#fee2e2", alpha=0.45, lw=0)
            ax.grid(alpha=0.3)
            ax.set_title(f"photos become {SHIFT_TITLES.get(sh, sh)}; " + ("no call limit" if b == 0 else f"{b:.0f} cloud calls a minute"),
                         fontsize=9, fontweight="bold", loc="left")
            if j == 0:
                ax.set_ylabel(f"cost per box (rolling {win})")
            if i == len(shifts) - 1:
                ax.set_xlabel("box number (red = after the photos changed)")
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=[0, 0.045 + 0.02 * (3 - min(3, len(shifts))), 1, 1])
    fig.savefig(os.path.join(FIG, "E8_photos_change.png"), dpi=150)
    plt.close(fig)
    return rows


E9_PAIRS = [("Q-learning (online)", "Adaptive rule"), ("Q-learning (online)", "Ask the cloud always"),
            ("Q-learning (online)", "Confidence rule"), ("Q-learning (frozen)", "Confidence rule"),
            ("Confidence rule", "Adaptive rule")]


def e9_how_sure(params, reward, facs, cfg, reps=200, boxes=400):
    """How much do the E1 gaps depend on WHICH 594 photos are in the test set?

    The test set is re-drawn 200 times (same size, photos picked with repeats: the bootstrap). On each
    re-drawn set every policy runs the 10 situations of E1 on the same boxes. The spread of a difference
    over the 200 sets is its uncertainty: if the 95 % range of 'A minus B' does not contain 0, the gap is real
    for photos like these; if it does, the two are level. The range also contains the simulator's own run-to-run noise,
    so it is on the careful side."""
    table = getattr(reward, "photos", None)
    if table is None:
        return None
    conds = [with_budget(cfg, c) for c in STEADY + [LOSSY, STALLS]]
    names = list(dict.fromkeys(n for pair in E9_PAIRS for n in pair))
    cost = {n: np.zeros(reps) for n in names}
    brng = np.random.default_rng(424242)
    for b in range(reps):
        photos = table.test.resample(brng)
        for name in names:
            tot = 0.0
            for ci, c in enumerate(conds):
                log = make_sim(params, reward, np.random.default_rng(20000 + 37 * b + ci), photos=photos).run(
                    facs[name](b), [(0, c)], boxes, log_conditions=False)
                tot += -float(np.mean(log.reward))
            cost[name][b] = tot / len(conds)

    def rng_of(x):
        return float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "reps": reps, "n_test": len(table.test), "situations": len(conds),
           "policies": {}, "pairs": []}
    for n in names:
        m, lo, hi = rng_of(cost[n])
        out["policies"][n] = {"cost": m, "low": lo, "high": hi}
    for a, bb in E9_PAIRS:
        d = cost[a] - cost[bb]
        m, lo, hi = rng_of(d)
        out["pairs"].append({"a": a, "b": bb, "diff": m, "low": lo, "high": hi, "a_cheaper_share": float((d < 0).mean()),
                             "real": bool(hi < 0 or lo > 0)})
    with open(os.path.join(RES, "E9_how_sure.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


# ----------------------------------------------------------------------------
REALITY_CHANGES = dict(gw_med_s=2.5, server_med_s=3.0, server_sigma=0.5, payload_kb=2.0,
                       fast_retx_prob=0.5, capture_med_s=1.4)


def reality_params(params):
    """The 'real hardware' turned out different from what the simulator assumed."""
    from dataclasses import replace
    r = replace(params)
    r.gw_med_s *= REALITY_CHANGES["gw_med_s"]        # gateway 2.5x slower than assumed
    r.server_med_s *= REALITY_CHANGES["server_med_s"]    # cloud server 3x slower ...
    r.server_sigma = REALITY_CHANGES["server_sigma"]     # ... and much less predictable
    r.payload_kb *= REALITY_CHANGES["payload_kb"]        # photos twice as big
    r.fast_retx_prob = REALITY_CHANGES["fast_retx_prob"]  # lost packets cost more
    r.capture_med_s *= REALITY_CHANGES["capture_med_s"]  # camera a bit slower
    return r


def e5_mismatch(plt, cfg, params, reward, agent, space, rp, seeds=20, boxes=2000, win=100):
    """Policies were built for the SIMULATOR; they now run on 'real' hardware that behaves differently."""
    from .train_q import pretrain, tune_rules
    real = reality_params(params)
    print("    re-tuning rules / re-training Q on 'reality' (reference only) ...")
    rp_real, _ = tune_rules(real, reward, n_eval=120)
    ag_real, _ = pretrain(real, reward, space, episodes=6000, seed=1, gamma=cfg["agent"]["gamma"], log_every=10**9)
    rng = np.random.default_rng(2024)
    scheds = [random_schedule(rng, boxes, change_prob=0.6) for _ in range(seeds)]

    def q_online(seed):
        ag = agent.copy(seed)
        ag.alpha, ag.epsilon, ag.alpha_mode = cfg["agent"]["alpha"], cfg["agent"]["epsilon"], "constant"
        ag.explore_gap = cfg["agent"]["explore_gap"]
        ag.forget = cfg["agent"]["forget"]
        return QPolicy(ag, space, True, True, name="Q-learning (online)")

    pols = {
        "Threshold rule": lambda s: ThresholdRule(**rp["threshold"]),
        "Deadline rule": lambda s: DeadlineRule(**rp["deadline"]),
        "Adaptive rule": lambda s: AdaptiveRule(**rp["adaptive"]),
        "Q-learning (frozen)": lambda s: QPolicy(agent.copy(s), space, False, False),
        "Q-learning (online)": q_online,
        "Adaptive rule RE-TUNED by hand (reference)": lambda s: AdaptiveRule(**rp_real["adaptive"]),
        "Q re-trained on real params (reference)": lambda s: QPolicy(ag_real.copy(s), space, False, False),
    }
    cols = dict(COLORS)
    cols["Adaptive rule RE-TUNED by hand (reference)"] = "#0369a1"
    cols["Q re-trained on real params (reference)"] = "#14532d"
    rows, curves = [], {}
    for name, mk in pols.items():
        rw = np.zeros(boxes); ms = np.zeros(boxes)
        for sd in range(seeds):
            log = make_sim(real, reward, np.random.default_rng(12000 + sd)).run(mk(sd), scheds[sd], boxes,
                                                                                log_conditions=False).arrays()
            rw += log["reward"]; ms += 1 - log["on_time"]
        rw /= seeds; ms /= seeds
        curves[name] = rw
        rows.append({"policy": name, "reward_all": float(rw.mean()), "reward_first_500": float(rw[:500].mean()),
                     "reward_last_500": float(rw[-500:].mean()), "missed_%": float(100 * ms.mean()),
                     "missed_last_500_%": float(100 * ms[-500:].mean())})
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, rw in curves.items():
        ls = "--" if "reference" in name else "-"
        ax.plot(np.arange(win - 1, boxes), rolling(rw, win), ls, color=cols.get(name, "#000"), label=name,
                lw=2 if "online" in name else 1.2)
    ax.set_xlabel("boxes on the (mismatched) conveyor"); ax.set_ylabel("reward per box (rolling 100)")
    ax.set_title("Simulator was wrong about the hardware - who copes without re-tuning?", loc="left",
                 fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "E5_sim_to_real_mismatch.png"), dpi=150); plt.close(fig)
    with open(os.path.join(RES, "E5_mismatch.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


# ----------------------------------------------------------------------------
def write_markdown(e1, e2, e3, rp, reward, params, agent, e5=None, e7=None, cfg=None, e8=None, e9=None):
    L = []
    L.append("# Simulation results (auto-generated by `python -m sim.experiments`)\n")
    L.append("> These are **simulated** results from the calibrated conveyor simulator. "
             "Repeat them on the real conveyor with `tools/run_experiment.py` + `tools/report_real.py`.\n")
    L.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")
    tbl = getattr(reward, "photos", None)
    bud = (cfg or {}).get("cloud_budget") or {}
    if tbl is not None:
        L.append(f"Boxes are replayed photos: **{tbl.source}** ({len(tbl.test)} test photos; the agent and the rules were "
                 f"trained and tuned on {len(tbl.val)} other photos). \"Reward\" in every table is minus the real cost of "
                 "what happened to each box (100 for a shipped damaged box, 10 for a wasted good one, 5 plus the damage "
                 "for an unchecked one). All policies see the same boxes.\n")
    if bud.get("per_min"):
        L.append(f"E1 and E2 run with a cloud-call limit of **{bud['per_min']} per minute** (about one box in four at "
                 "one box a second). A pick-one policy that wants the cloud without a call left gets the gateway instead.\n")
    L.append("## Reward per box (what the agent optimises)\n")
    d = reward.describe()
    L.append(f"- checked on the ESP32 in time: **{d['edge_on_time']:.2f}**  \n- checked on the gateway in time: "
             f"**{d['gateway_on_time']:.2f}**  \n- checked on the cloud in time: "
             f"**{d['cloud_on_time']:.2f}**  \n- missed (box passed unchecked): **{d['missed']:.2f}**\n")
    for lv in d.get("by_confidence", []):
        L.append(f"- gateway {lv['name']} ({100*lv['share']:.0f}% of boxes): keep its verdict **{lv['gateway']:.2f}**, "
                 f"cloud answer **{lv['cloud']:.2f}**")
    L.append("")
    L.append("Model quality used: " + json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()}
                                                  for k, v in {"esp32": reward.metrics[EDGE], "gateway": reward.metrics[GATEWAY],
                                                               "cloud": reward.metrics[CLOUD]}.items()}) + "\n")
    L.append("## E1 - Steady conditions (mean of 20 runs x 400 boxes)\n")
    L.append("`wifi-stalls` (20% of cloud requests freeze 0.4-0.8 s, e.g. microwave-oven interference) was "
             "NEVER part of pre-training or rule tuning.\n")
    pols = list(dict.fromkeys(r["policy"] for r in e1))
    scen = list(dict.fromkeys(r["scenario"] for r in e1))
    L.append("**Reward per box** (higher is better; best in bold)\n")
    L.append("| scenario | " + " | ".join(pols) + " |")
    L.append("|---" * (len(pols) + 1) + "|")
    for s in scen:
        vals = {r["policy"]: r["reward_mean"] for r in e1 if r["scenario"] == s}
        best = max(vals.values())
        L.append(f"| {s} | " + " | ".join(("**%.2f**" if abs(vals[p] - best) < 1e-9 else "%.2f") % vals[p] for p in pols) + " |")
    L.append("\n**Missed boxes %**\n")
    L.append("| scenario | " + " | ".join(pols) + " |")
    L.append("|---" * (len(pols) + 1) + "|")
    for s in scen:
        L.append(f"| {s} | " + " | ".join("%.1f" % (100 * next(r["miss_rate"] for r in e1 if r["scenario"] == s and r["policy"] == p)) for p in pols) + " |")
    L.append("\n**Damaged boxes that got through %**\n")
    L.append("| scenario | " + " | ".join(pols) + " |")
    L.append("|---" * (len(pols) + 1) + "|")
    for s in scen:
        L.append(f"| {s} | " + " | ".join("%.1f" % (100 * next(r["escape_rate"] for r in e1 if r["scenario"] == s and r["policy"] == p)) for p in pols) + " |")
    avg = {p: np.mean([r["reward_mean"] for r in e1 if r["policy"] == p]) for p in pols}
    L.append(f"\n**Average reward over all {len(scen)} situations:** " + ", ".join(f"{p} {v:.2f}" for p, v in sorted(avg.items(), key=lambda kv: -kv[1])) + "\n")
    L.append("![E1](figures/E1_steady_comparison.png)\n")
    L.append("## E2 - Change in the middle (box 1-300 normal, 301-600 problem, 601-900 normal; 30 runs)\n")
    L.append("| change | policy | missed % during problem | boxes missed (of 300) | boxes to recover | boxes to trust cloud again | missed % after | on ESP32 % during problem |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in e2:
        L.append(f"| {r['change']} | {r['policy']} | {r['missed_during_%']:.1f} | {r['boxes_missed_during(of 300)']:.1f} | "
                 f"{r['boxes_to_recover']} | {r['boxes_to_trust_laptop_again']} | {r['missed_after_%']:.1f} | "
                 f"{r['esp32_share_during_%']:.1f} |")
    L.append("\n![E2](figures/E2_change_in_middle.png)\n")
    L.append("## E3 - Does pre-training help? (30 runs x 3000 boxes, random changing conditions)\n")
    L.append("| agent | reward first 500 | reward last 500 | missed % first 500 | missed % last 500 |")
    L.append("|---|---|---|---|---|")
    for k, v in e3.items():
        L.append(f"| {k} | {v['reward_first_500']:.2f} | {v['reward_last_500']:.2f} | {v['missed_first_500_%']:.2f} | {v['missed_last_500_%']:.2f} |")
    L.append("\n![E3](figures/E3_pretraining_value.png)\n")
    L.append("## E4 - What the agent learned\n\n![E4](figures/E4_policy_map.png)\n")
    if e5:
        L.append("## E5 - Simulator vs reality mismatch (20 runs x 2000 boxes)\n")
        L.append("The 'real' hardware differs from the simulator: " + json.dumps(REALITY_CHANGES) +
                 " (multipliers / new values). Policies were built for the simulator and are NOT re-tuned. "
                 "The two 'reference' rows show what you get if a person re-tunes / re-trains for the new hardware.\n")
        L.append("| policy | reward (all) | reward first 500 | reward last 500 | missed % | missed % last 500 |")
        L.append("|---|---|---|---|---|---|")
        for r in e5:
            L.append(f"| {r['policy']} | {r['reward_all']:.3f} | {r['reward_first_500']:.3f} | {r['reward_last_500']:.3f} | "
                     f"{r['missed_%']:.2f} | {r['missed_last_500_%']:.2f} |")
        L.append("\n![E5](figures/E5_sim_to_real_mismatch.png)\n")
    if e7:
        L.append("## E7 - Cloud-call budget (20 runs x 400 boxes, about 60 boxes a minute)\n")
        L.append("Cost per box (lower is better). \"Best possible\" knows which boxes a cloud answer would fix; no real "
                 "policy can know that.\n")
        for net in dict.fromkeys(r["network"] for r in e7):
            buds = list(dict.fromkeys(r["cloud_calls_per_min"] for r in e7))
            pols7 = list(dict.fromkeys(r["policy"] for r in e7))
            L.append(f"**{net} network**\n")
            L.append("| policy | " + " | ".join("no limit" if b == 0 else f"{b}/min" for b in buds) + " |")
            L.append("|---" * (len(buds) + 1) + "|")
            for p_ in pols7:
                L.append(f"| {p_} | " + " | ".join("%.2f" % -next(r["reward_mean"] for r in e7 if r["network"] == net
                         and r["policy"] == p_ and r["cloud_calls_per_min"] == b) for b in buds) + " |")
            L.append("")
        L.append("![E7](figures/E7_budget.png)\n")
    if e8:
        L.append("## E8 - The photos change (20 runs; box 1-500 normal photos, then 3000 boxes of changed photos)\n")
        L.append("The same real test photos, made darker / blurred and scored again by the same three models "
                 "(`tools/shift_scores.py`). Nothing is re-tuned when the change happens. Cost per box, lower is better. "
                 "Only the two \"spot checks\" rows are told the true state of boxes afterwards (every box, or 1 box in 5); "
                 "with it they re-measure what a cloud answer is worth. \"Last 1500\" = boxes 2001-3500, after they had "
                 "time to correct themselves.\n")
        for sh in dict.fromkeys(r["photos"] for r in e8):
            for b in dict.fromkeys(r["cloud_calls_per_min"] for r in e8):
                sub = [r for r in e8 if r["photos"] == sh and r["cloud_calls_per_min"] == b]
                L.append(f"**Photos become {SHIFT_TITLES.get(sh, sh)}; " + ("no call limit" if b == 0 else f"{b:.0f} cloud calls a minute") + "**\n")
                L.append("| policy | before the change | first 500 after | last 1500 | asked the cloud (last 1500) | correct verdicts (last 1500) |")
                L.append("|---|---|---|---|---|---|")
                real = [r for r in sub if not r["policy"].startswith("Best possible")]
                best = min(r["cost_last"] for r in real)
                for r in sub:
                    c = ("**%.2f**" if (r in real and abs(r["cost_last"] - best) < 1e-9) else "%.2f") % r["cost_last"]
                    L.append(f"| {r['policy']} | {r['cost_before']:.2f} | {r['cost_first_500']:.2f} | {c} | "
                             f"{100*r['ask_share_last']:.0f}% | {100*r['accuracy_last']:.1f}% |")
                L.append("")
        L.append("![E8](figures/E8_photos_change.png)\n")
    if e9:
        L.append(f"## E9 - How sure are the gaps? ({e9['reps']} re-drawn test sets)\n")
        L.append(f"The {e9['n_test']} test photos are one draw out of many possible ones. The whole E1 comparison "
                 f"({e9['situations']} situations) was repeated on {e9['reps']} re-drawn test sets (same size, photos picked "
                 "with repeats). \"Range\" is where 95 % of the results fall. A gap is called real only when its range "
                 "does not contain 0.\n")
        L.append("| comparison (cost per box, A minus B) | gap | 95 % range | A cheaper in | verdict |")
        L.append("|---|---|---|---|---|")
        for pr in e9["pairs"]:
            L.append(f"| {pr['a']} vs {pr['b']} | {pr['diff']:+.2f} | {pr['low']:+.2f} to {pr['high']:+.2f} | "
                     f"{100*pr['a_cheaper_share']:.0f}% of sets | {'real gap' if pr['real'] else 'level (within chance)'} |")
        L.append("")
        L.append("| policy | cost per box | 95 % range |")
        L.append("|---|---|---|")
        for n, v in e9["policies"].items():
            L.append(f"| {n} | {v['cost']:.2f} | {v['low']:.2f} to {v['high']:.2f} |")
        L.append("")
    L.append("## Baseline rule settings (tuned on the same random conditions as Q pre-training)\n")
    L.append("```json\n" + json.dumps({k: v for k, v in rp.items() if not k.startswith("_")}, indent=2) + "\n```\n")
    L.append("## Simulator parameters used\n")
    L.append("```json\n" + json.dumps(params.__dict__, indent=2) + "\n```\n")
    with open(os.path.join(RES, "SIM_RESULTS.md"), "w") as f:
        f.write("\n".join(L))


def main():
    t0 = time.time()
    plt = setup()
    cfg = load_config()
    params, reward, space = build(cfg)
    load_shift_tables(cfg, reward)              # the same photos made harder (only E8 uses them)
    agent = QAgent.load(project_path(cfg["agent"]["q_table"]))
    with open(project_path(cfg["agent"]["rule_params"])) as f:
        rp = json.load(f)
    os.makedirs(FIG, exist_ok=True)
    facs = factories(cfg, agent, space, rp)
    print("E1 steady conditions ..."); e1 = e1_steady(plt, params, reward, facs, cfg=cfg)
    print("E2 change in the middle ..."); e2 = e2_changes(plt, params, reward, facs, cfg=cfg)
    print("E3 pre-training value ..."); e3 = e3_pretrain_value(plt, cfg, params, reward, agent, space)
    print("E4 policy map ..."); e4_policy_map(plt, agent, space)
    print("E5 simulator vs reality mismatch ..."); e5 = e5_mismatch(plt, cfg, params, reward, agent, space, rp)
    print("E7 cloud-call budget ..."); e7 = e7_budget(plt, params, reward, facs, rp)
    print("E8 the photos change ..."); e8 = e8_photos_change(plt, cfg, params, reward, facs, rp)
    print("E9 how sure are the gaps ..."); e9 = e9_how_sure(params, reward, facs, cfg)
    write_markdown(e1, e2, e3, rp, reward, params, agent, e5, e7, cfg, e8, e9)
    print(f"done in {time.time()-t0:.0f}s -> results/SIM_RESULTS.md, results/figures/")


if __name__ == "__main__":
    main()
