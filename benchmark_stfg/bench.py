"""
Secondary benchmark: the SAME tabular Q-learning offloading idea on the public
Synthetic Task Flow Graphs dataset (edge / hub / cloud, task dependencies).

    python -m benchmark_stfg.bench                      # uses data/stfg/ (the real dataset)
    python -m benchmark_stfg.bench --data data/stfg_standin   # code test only (stand-in graphs)

Model (simplified from the dataset's paper):
  * 3 devices: Edge (gateway), Hub (laptop), Cloud; each runs one task at a time
  * a task can start when all parents are done AND their output data has been sent
    (edge<->hub Wi-Fi link, hub<->cloud Internet link; edge<->cloud goes through the hub)
  * tasks with FixedAllocation must run where they are fixed
  * network condition per run: good / busy / bad (bandwidth + latency of both links)
  * goal: small makespan (total finish time of the application)

Policies: all-edge, all-hub, all-cloud, random, greedy earliest-finish-time (EFT, a classic
list-scheduling heuristic) and Q-learning.

Q-learning state (972 states): which device the quick estimate says finishes first, how clear
that lead is, how much data the task sends on, where its parents ran, network quality, how many
tasks wait for it, and whether the application still has tasks pinned to a device.
Reward: -(time the task finishes after its parents were done). If two choices are valued
within 10 %, it takes the one the quick estimate prefers.

Trained on the small and medium graphs (up to ~120 tasks), tested on ALL graphs; the
1000-task graphs are never seen in training. Averaged over 5 training seeds.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from common.config import project_path
from common.rl import QAgent
from .load import load_all

NETS = {  # edge-hub Mbps, hub-cloud Mbps, edge-hub latency s, hub-cloud latency s
    "good": (50.0, 100.0, 0.002, 0.020),
    "busy": (10.0, 30.0, 0.010, 0.050),
    "bad": (2.0, 8.0, 0.050, 0.150),
}
NET_NAMES = list(NETS)


def transfer_s(mbit, src, dst, net):
    if src == dst or mbit <= 0:
        return 0.0
    b_eh, b_hc, l_eh, l_hc = net
    lo, hi = min(src, dst), max(src, dst)
    if (lo, hi) == (0, 1):
        return l_eh + mbit / b_eh
    if (lo, hi) == (1, 2):
        return l_hc + mbit / b_hc
    return l_eh + mbit / b_eh + l_hc + mbit / b_hc     # edge <-> cloud via hub


def topo(tasks):
    idx = {t["id"]: i for i, t in enumerate(tasks)}
    indeg = [0] * len(tasks)
    kids = [[] for _ in tasks]
    for i, t in enumerate(tasks):
        for p in t["parents"]:
            if p in idx:
                indeg[i] += 1
                kids[idx[p]].append(i)
    order, q = [], [i for i in range(len(tasks)) if indeg[i] == 0]
    while q:
        i = q.pop(0)
        order.append(i)
        for k in kids[i]:
            indeg[k] -= 1
            if indeg[k] == 0:
                q.append(k)
    return order, idx


TIE_MARGIN = 0.10      # Q-values closer than this (relative) -> follow the quick estimate


class Features:
    def __init__(self, graphs):
        datas = [t["out_mbit"] for g in graphs for t in g["tasks"]]
        self.d_e = list(np.quantile(datas, [1 / 3, 2 / 3])) if datas else [0, 0]
        # estimate-best device, lead of that estimate, output data, parents' place, network,
        # tasks waiting for this one, pinned tasks (none / some left / all done)
        self.dims = (3, 3, 3, 4, 3, 3, 3)
        self.n = int(np.prod(self.dims))

    @staticmethod
    def _b(x, e):
        return 0 if x < e[0] else (1 if x < e[1] else 2)

    def state(self, t, est, parent_loc, net_i, n_kids, pinned):
        o = np.argsort(est)
        gap = (est[o[1]] - est[o[0]]) / max(1e-6, est[o[0]])
        lead = 0 if gap < 0.1 else (1 if gap < 0.5 else 2)
        return int(np.ravel_multi_index((int(o[0]), lead, self._b(t["out_mbit"], self.d_e), parent_loc, net_i,
                                         min(2, n_kids), pinned), self.dims))


def run_graph(g, net_name, policy, feats=None, agent=None, learn=False, rng=None):
    net = NETS[net_name]
    tasks = g["tasks"]
    order, idx = topo(tasks)
    finish = [0.0] * len(tasks)
    where = [0] * len(tasks)
    free = [0.0, 0.0, 0.0]
    energy = 0.0
    prev = None
    kids = [0] * len(tasks)
    for t in tasks:
        for p in t["parents"]:
            if p in idx:
                kids[idx[p]] += 1
    pins_left = sum(1 for t in tasks if t["fixed"] is not None)
    had_pins = pins_left > 0
    for i in order:
        t = tasks[i]
        par = [idx[p] for p in t["parents"] if p in idx]
        ready = [max([finish[p] + transfer_s(tasks[p]["out_mbit"], where[p], d, net) for p in par], default=0.0)
                 for d in range(3)]
        earliest = max([finish[p] for p in par], default=0.0)
        est = [max(ready[k], free[k]) + t["L"][k] / 1000 for k in range(3)]
        s = None
        if t["fixed"] is not None:
            d = t["fixed"]
            pins_left -= 1
        elif policy in ("edge", "hub", "cloud"):
            d = ("edge", "hub", "cloud").index(policy)
        elif policy == "random":
            d = int(rng.integers(3))
        elif policy == "eft":
            d = int(np.argmin(est))
        else:  # qlearning
            locs = [where[p] for p in par]
            ploc = 3 if not locs else int(np.bincount(locs, minlength=3).argmax())
            pinned = 0 if not had_pins else (1 if pins_left > 0 else 2)
            s = feats.state(t, np.array(est) - earliest, ploc, NET_NAMES.index(net_name), kids[i], pinned)
            d = agent.act(s, explore=learn)
            if not learn:                      # near-tie -> trust the quick estimate
                q = agent.q[s]
                b = int(np.argmin(est))
                if q[b] >= q.max() - TIE_MARGIN * abs(q.max()) - 1e-9:
                    d = b
        start = max(ready[d], free[d])
        finish[i] = start + t["L"][d] / 1000.0
        free[d] = finish[i]
        where[i] = d
        if d == 0:
            energy += t["P_edge"] * t["L"][d] / 1000.0
        if policy == "qlearning" and s is not None:
            r = -(finish[i] - earliest)
            if learn and prev is not None:
                agent.update(prev[0], prev[1], prev[2], s)
            prev = (s, d, r)
    if policy == "qlearning" and learn and prev is not None:
        agent.update(prev[0], prev[1], prev[2], None)
    return max(finish), energy, np.bincount(where, minlength=3) / len(tasks)


def size_group(n):
    return "about 10 tasks" if n <= 15 else ("about 100 tasks" if n <= 150 else "about 1000 tasks (never seen in training)")


GROUPS = ["about 10 tasks", "about 100 tasks", "about 1000 tasks (never seen in training)"]
POLS = ("edge", "hub", "cloud", "random", "eft", "qlearning")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    folder = project_path(a.data) if a.data else project_path("data/stfg")
    graphs = load_all(folder)
    if not graphs:
        raise SystemExit(f"No task graphs in {folder}. Put SyntheticTaskFlowGraphs.zip there "
                         f"(https://doi.org/10.5281/zenodo.10654551), or test with "
                         f"`python -m benchmark_stfg.make_standin` + `--data data/stfg_standin`.")
    standin = "standin" in folder
    print(f"loaded {len(graphs)} graphs from {folder}" + ("  [STAND-IN - code test only]" if standin else ""))
    train = [g for g in graphs if g["n"] <= 150] or graphs
    feats = Features(graphs)
    rows = []
    t0 = time.time()
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        agent = QAgent(feats.n, 3, gamma=0.9, epsilon=1.0, seed=seed, alpha_mode="visits", alpha_min=0.02)
        for ep in range(a.episodes):
            agent.epsilon = max(0.05, 1 - ep / (0.7 * a.episodes))
            g = train[int(rng.integers(len(train)))]
            run_graph(g, NET_NAMES[int(rng.integers(3))], "qlearning", feats, agent, learn=True, rng=rng)
        for g in graphs:
            for net in NET_NAMES:
                res = {}
                for pol in POLS:
                    ms, en, share = run_graph(g, net, pol, feats, agent, learn=False,
                                              rng=np.random.default_rng(1000 + seed))
                    res[pol] = ms
                    rows.append({"seed": seed, "graph": g["name"], "kind": g["kind"], "tasks": g["n"],
                                 "group": size_group(g["n"]), "network": net, "policy": pol, "makespan_s": ms,
                                 "edge_energy_J": en, "share_edge": share[0], "share_hub": share[1],
                                 "share_cloud": share[2]})
                for r in rows[-len(POLS):]:
                    r["vs_eft"] = r["makespan_s"] / res["eft"]
    print(f"Q-learning trained {a.seeds} times on {len(train)} small/medium graphs, "
          f"{a.episodes} rounds each ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    os.makedirs(project_path("results/figures"), exist_ok=True)
    tag = "STFG_STANDIN" if standin else "STFG"
    df.to_csv(project_path(f"results/{tag}_results.csv"), index=False)
    groups = [g for g in GROUPS if g in set(df["group"])]
    piv_g = df.groupby(["group", "policy"])["vs_eft"].mean().unstack().reindex(groups)[list(POLS)].round(2)
    piv_n = df.groupby(["network", "policy"])["vs_eft"].mean().unstack().reindex(NET_NAMES)[list(POLS)].round(2)
    q = df[df.policy == "qlearning"]
    per_seed = q.groupby("seed")["vs_eft"].mean().round(2).tolist()
    wins = {g: {"better": int((q[q.group == g].vs_eft < 0.999).sum()),
                "same": int(((q[q.group == g].vs_eft >= 0.999) & (q[q.group == g].vs_eft <= 1.001)).sum()),
                "worse": int((q[q.group == g].vs_eft > 1.001).sum())} for g in groups}
    big = groups[-1]
    v_big = float(piv_g.loc[big, "qlearning"])
    if v_big <= 0.97:
        verdict = (f"On the graphs it never saw ({big.split(' (')[0]}) Q-learning finishes "
                   f"{round((1-v_big)*100)} % sooner than the greedy scheduler.")
    elif v_big <= 1.03:
        verdict = (f"On the graphs it never saw ({big.split(' (')[0]}) Q-learning is level with the greedy "
                   f"scheduler ({v_big:.2f} of its time). It does not beat it.")
    else:
        verdict = (f"On the graphs it never saw ({big.split(' (')[0]}) Q-learning is "
                   f"{round((v_big-1)*100)} % slower than the greedy scheduler.")
    summary = {
        "standin": standin, "dataset": "Synthetic Task Flow Graphs", "doi": "10.5281/zenodo.10654551",
        "graphs": len(graphs), "train_graphs": len(train), "seeds": a.seeds, "episodes": a.episodes,
        "policies": list(POLS), "groups": groups, "verdict": verdict,
        "by_group": {g: {p_: float(piv_g.loc[g, p_]) for p_ in POLS} for g in groups},
        "by_network": {n: {p_: float(piv_n.loc[n, p_]) for p_ in POLS} for n in NET_NAMES},
        "q_cases": wins, "q_per_seed": per_seed,
    }
    import json
    with open(project_path(f"results/{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.6))
    short = piv_g.copy()
    short.index = [g.split(" (")[0] + ("\n(never seen)" if "never" in g else "") for g in short.index]
    short.plot.bar(ax=ax, rot=0, logy=True)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("finish time / greedy scheduler (log)"); ax.grid(axis="y", alpha=0.3)
    ax.set_title(("STAND-IN graphs (code test) - " if standin else "") + "Task-graph benchmark: lower is better",
                 loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(project_path(f"results/figures/{tag}_benchmark.png"), dpi=150); plt.close(fig)

    cases = pd.DataFrame(wins).T[["better", "same", "worse"]]
    md = [f"# Task-graph benchmark ({'STAND-IN data: code test only, NOT the dataset results' if standin else 'Synthetic Task Flow Graphs, doi:10.5281/zenodo.10654551'})\n",
          "A second, public test of the same idea: a table-based Q-learner decides where each task of an "
          "application runs (edge / hub / cloud).\n",
          "Numbers are finish time divided by the finish time of the greedy earliest-finish-time (EFT) "
          "scheduler, a strong classic method. 1.00 = same as EFT, lower is better. "
          f"Average of {a.seeds} training runs.\n",
          f"**Plain result:** {verdict}\n",
          "## By application size\n", piv_g.to_markdown(),
          "\n## By network condition\n", piv_n.to_markdown(),
          "\n## Q-learning against the greedy scheduler, case by case\n",
          "(one case = one graph on one network in one training run)\n", cases.to_markdown(),
          f"\nAverage per training run: {per_seed}\n",
          "## How to read this honestly\n",
          "- The about-100-task graphs were used for training, so a good number there is not a fair test. "
          "The fair test is the about-1000-task group.",
          "- The about-10-task graphs finish in a fraction of a second, so a large ratio there is a small "
          "amount of real time.",
          "- EFT is a hand-made rule that sees exact timings. Q-learning has to learn from experience with a "
          "coarse table. Being level with EFT on unseen graphs shows the learner carries over to a different "
          "problem; it is not a claim to beat the best schedulers.",
          f"\n![](figures/{tag}_benchmark.png)\n"]
    with open(project_path(f"results/{tag}_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
