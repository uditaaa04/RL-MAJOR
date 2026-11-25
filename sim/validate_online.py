"""
How the ONLINE learning settings (alpha, epsilon, explore_gap, forget) were chosen.

    python -m sim.validate_online        (~5-10 minutes)

Uses separate VALIDATION runs (random, changing conditions, own random seeds) -
never the named test scenarios of sim/experiments.py - so the test results stay honest.
Writes results/online_validation.csv
"""
from __future__ import annotations

import csv
import itertools
import json

import numpy as np

from common.config import load_config, project_path
from common.policies import AdaptiveRule, QPolicy
from common.rl import QAgent
from .env import random_schedule, summarize
from .train_q import build, make_sim


def main(boxes=1500, seeds=16):
    cfg = load_config()
    params, reward, space = build(cfg)
    base = QAgent.load(project_path(cfg["agent"]["q_table"]))
    with open(project_path(cfg["agent"]["rule_params"])) as f:
        rp = json.load(f)
    rng = np.random.default_rng(424242)
    scheds = [random_schedule(rng, boxes, change_prob=0.8) for _ in range(seeds)]

    def run(mk):
        rs, ms = [], []
        for i in range(seeds):
            log = make_sim(params, reward, np.random.default_rng(31000 + i)).run(mk(i), scheds[i], boxes,
                                                                                log_conditions=False)
            s = summarize(log)
            rs.append(s["reward_mean"]); ms.append(s["miss_rate"])
        return float(np.mean(rs)), float(np.mean(ms))

    rows = []
    r, m = run(lambda i: AdaptiveRule(**rp["adaptive"]))
    rows.append({"policy": "Adaptive rule", "alpha": "", "epsilon": "", "explore_gap": "", "forget": "", "reward": r, "missed": m})
    r, m = run(lambda i: QPolicy(base.copy(i), space, False, False))
    rows.append({"policy": "Q frozen", "alpha": "", "epsilon": "", "explore_gap": "", "forget": "", "reward": r, "missed": m})
    for alpha, eps, gap, fg in itertools.product([0.05, 0.1, 0.2], [0.02, 0.05], [None, 3.0, 6.0], [0.0, 0.02]):
        def mk(i, alpha=alpha, eps=eps, gap=gap, fg=fg):
            ag = base.copy(i)
            ag.alpha, ag.epsilon, ag.alpha_mode, ag.explore_gap, ag.forget = alpha, eps, "constant", gap, fg
            return QPolicy(ag, space, True, True)
        r, m = run(mk)
        rows.append({"policy": "Q online", "alpha": alpha, "epsilon": eps, "explore_gap": gap, "forget": fg,
                     "reward": r, "missed": m})
        print(f"alpha={alpha} eps={eps} gap={gap} forget={fg}: reward={r:.3f} missed={m:.3%}", flush=True)
    with open(project_path("results/online_validation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    best = max((r for r in rows if r["policy"] == "Q online"), key=lambda r: r["reward"])
    print("best online setting:", best)


if __name__ == "__main__":
    main()
