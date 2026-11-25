"""Re-run E2 exactly as sim.experiments does (same seeds) and save the per-item curves for paper figures."""
import json
import numpy as np
from common.config import load_config, project_path
from common.rl import QAgent
from sim.experiments import factories
from sim.scenarios import CHANGES
from sim.train_q import build, make_sim

cfg = load_config()
params, reward, space = build(cfg)
agent = QAgent.load(project_path(cfg["agent"]["q_table"]))
rp = json.load(open(project_path(cfg["agent"]["rule_params"])))
facs = factories(cfg, agent, space, rp)
names = ["Threshold rule", "Deadline rule", "Adaptive rule", "Q-learning (frozen)", "Q-learning (online)"]
out = {}
phase, seeds = 300, 30
for ci, (title, base, changed) in enumerate(CHANGES):
    sched = [(0, base), (phase, changed), (2 * phase, base)]
    n = 3 * phase
    for name in names:
        miss = np.zeros(n); cloud = np.zeros(n)
        for sd in range(seeds):
            log = make_sim(params, reward, np.random.default_rng(5000 + sd)).run(
                facs[name](sd), sched, n, log_conditions=False).arrays()
            miss += 1 - log["on_time"]; cloud += log["action"]
        out[f"{ci}|{name}|miss"] = miss / seeds
        out[f"{ci}|{name}|cloud"] = cloud / seeds
    print("done", title, flush=True)
np.savez(project_path("results/E2_curves.npz"), **out)
print("saved results/E2_curves.npz")
