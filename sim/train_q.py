"""
PRE-TRAIN the Q-learning agent in the simulator (before it ever touches the real belt),
and tune the baseline rules fairly on the SAME randomised conditions.

    python -m sim.train_q                       # defaults
    python -m sim.train_q --calib config/calibration.json   # after measuring your hardware

Outputs
  models/q_table.npz          the trained Q-table (loaded by the gateway)
  models/rule_params.json     best settings for the threshold / deadline / adaptive rules
  results/pretrain_curve.csv  learning curve
  results/figures/pretrain_curve.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from common.config import load_config, project_path
from common.cascade import load_table
from common.policies import AdaptiveRule, ConfidenceReserveRule, ConfidenceRule, DeadlineRule, QPolicy, ThresholdRule
from common.rl import GATEWAY, QAgent, RewardModel, StateSpace
from .env import ConveyorSim, SimParams, random_schedule, summarize


def build(cfg: dict, calib: str | None = None):
    calib_path = project_path(calib) if calib else project_path("config/calibration.json")
    calib_d = {}
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            calib_d = json.load(f)
        print(f"using calibration {calib_path}")
    params = SimParams.from_calibration(calib_d)
    params.cam_to_arm_m = float(calib_d.get("cam_to_arm_m", cfg["conveyor"]["cam_to_arm_m"]))
    params.guard_s = float(cfg["conveyor"]["guard_s"])
    params.defect_rate = float(cfg["conveyor"]["defect_rate_assumed"])
    params.probe_interval_s = float(cfg["network"]["probe_interval_s"])
    params.probe_timeout_s = float(cfg["network"]["probe_timeout_s"])
    params.ewma_alpha = float(cfg["network"]["ewma_alpha"])
    params.cloud_probe_interval_s = float(cfg["network"]["cloud_probe_interval_s"])
    params.cloud_probe_timeout_s = float(cfg["network"]["cloud_probe_timeout_s"])
    params.cloud_ewma_alpha = float(cfg["network"]["cloud_ewma_alpha"])
    params.p_cpu_w = float(cfg["energy"]["p_cpu_active_w"])
    params.p_wifi_w = float(cfg["energy"]["p_wifi_tx_w"])
    params.p_esp_w = float(cfg["energy"].get("p_esp32_w", params.p_esp_w))
    reward = RewardModel.from_card(cfg, project_path(cfg["models"]["card"]))
    # the ESP32's time follows from the tiny model actually in use (photo size -> cable time, multiplies ->
    # compute time), unless it was measured on the real board and put into the calibration file
    from common.rl import load_card
    e = load_card(project_path(cfg["models"]["card"])).get("edge") or {}
    if e.get("img") and e.get("mults") and "esp_link_s" not in calib_d:
        baud = float(cfg.get("esp32", {}).get("baud", 460800))
        params.esp_link_s = (int(e["img"]) ** 2 * int(e.get("channels", 1)) + 40) * 10.0 / baud + 0.002
        params.esp_med_s = float(e["mults"]) * 40e-9 + 0.001
    space = StateSpace(cfg["state_bins"])
    attach_photos(cfg, reward)
    return params, reward, space


def attach_photos(cfg: dict, reward: RewardModel, table=None):
    """Give the reward model the photo table the simulator replays (real scores from tools/photo_scores.py,
    or made-up scores that match the model metrics when that file does not exist yet). The per-confidence
    costs the agent learns from come from the table's VALIDATION photos; experiments replay its TEST photos."""
    if table is None:
        names = {0: "edge", 1: "gateway", 2: "cloud"}
        table = load_table(project_path(cfg["models"].get("photo_scores", "models/photo_scores.npz")),
                           {names[a]: m for a, m in reward.metrics.items()})
    cascade = table.val.stats()
    new = RewardModel({"cost_escape": reward.c_esc, "cost_false_reject": reward.c_fr, "cost_unchecked": reward.c_unchk,
                       "energy_weight": reward.w_e}, reward.p_def, reward.metrics[1], reward.metrics[2],
                      reward.metrics[0], cascade)
    reward.cascade, reward.level = new.cascade, new.level
    reward.photos = table
    # the gateway's model goes first only when it clearly fits (state "fits"); when it is tight or too slow the
    # agent picks one place outright, so a slow gateway can never hold a box past the arm
    reward.gw_fit_limit = float(sorted(cfg["state_bins"].get("gw_fit", [0.4, 1.0]))[0])
    return reward


def make_sim(params, reward, rng, split: str = "test"):
    """split: which photos the boxes are: "val" while training and tuning, "test" when judging."""
    table = getattr(reward, "photos", None)
    photos = None if table is None else (table.val if split == "val" else table.test)
    return ConveyorSim(params, reward, rng, photos, getattr(reward, "gw_fit_limit", 1.0))


def rule_defaults(params: SimParams):
    return dict(gw_s=params.gw_med_s, jpeg_s=params.jpeg_s, payload_kb=params.payload_kb,
                bw_mbps=10.0, server_s=params.server_med_s, esp_s=params.esp_link_s + params.esp_med_s)


def evaluate_policy(make_policy, params, reward, eval_set, boxes, split: str = "val"):
    """Mean reward / miss rate over a fixed set of (seed, schedule) pairs."""
    rs, ms = [], []
    for seed, sched in eval_set:
        rng = np.random.default_rng(seed)
        log = make_sim(params, reward, rng, split).run(make_policy(), sched, boxes, log_conditions=False)
        s = summarize(log)
        rs.append(s["reward_mean"])
        ms.append(s["miss_rate"])
    return float(np.mean(rs)), float(np.mean(ms))


def tune_rules(params, reward, n_eval=200, boxes=150, seed=12345):
    rng = np.random.default_rng(seed)
    eval_set = [(int(rng.integers(1 << 30)), random_schedule(rng, boxes)) for _ in range(n_eval)]
    base = rule_defaults(params)
    best = {}

    res = []
    for T in [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600]:
        r, m = evaluate_policy(lambda: ThresholdRule(T), params, reward, eval_set, boxes)
        res.append((r, T, m))
    r, T, m = max(res)
    best["threshold"] = {"rtt_threshold_ms": T}
    print(f"  threshold rule : best T={T} ms   reward={r:.3f} miss={m:.3%}")

    res = []
    for mg in [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30]:
        r, m = evaluate_policy(lambda: DeadlineRule(margin_s=mg, **base), params, reward, eval_set, boxes)
        res.append((r, mg, m))
    r, mg, m = max(res)
    best["deadline"] = dict(margin_s=mg, **base)
    print(f"  deadline rule  : best margin={mg}s  reward={r:.3f} miss={m:.3%}")

    res = []
    for mg in [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25]:
        r, m = evaluate_policy(lambda: AdaptiveRule(margin_s=mg, **base), params, reward, eval_set, boxes)
        res.append((r, mg, m))
    r, mg, m = max(res)
    best["adaptive"] = dict(margin_s=mg, **base)
    print(f"  adaptive rule  : best margin={mg}s  reward={r:.3f} miss={m:.3%}")

    # second-opinion rules: how unsure must the gateway be before the cloud is asked, and with what time margin
    res = []
    for mc in (2, 3, 4):
        for mg in [0.0, 0.03, 0.06, 0.10, 0.15]:
            r, m = evaluate_policy(lambda: ConfidenceRule(min_conf=mc, margin_s=mg, **base), params, reward, eval_set, boxes)
            res.append((r, mc, mg, m))
    r, mc, mg, m = max(res)
    best["confidence"] = dict(min_conf=mc, margin_s=mg, **base)
    print(f"  confidence rule: best 'ask from level {mc}', margin={mg}s  reward={r:.3f} miss={m:.3%}")
    res = []
    for mc2 in (2, 3):
        for rsv in (0.3, 0.5, 0.7):
            r, m = evaluate_policy(lambda: ConfidenceReserveRule(min_conf=mc2, reserve=rsv, margin_s=mg, **base),
                                   params, reward, eval_set, boxes)
            res.append((r, mc2, rsv, m))
    r, mc2, rsv, m = max(res)
    best["confidence_reserve"] = dict(min_conf=mc2, reserve=rsv, margin_s=mg, **base)
    print(f"  confidence rule + reserve: best 'ask from level {mc2}', reserve={rsv}  reward={r:.3f} miss={m:.3%}")
    return best, eval_set


def pretrain(params, reward, space, episodes=6000, boxes=150, seed=0, gamma=0.6,
             alpha0=0.2, alpha_min=0.03, eps_min=0.05, log_every=50):
    rng = np.random.default_rng(seed)
    agent = QAgent(space.n_states, 3, alpha=alpha0, gamma=gamma, epsilon=1.0, seed=seed,
                   alpha_mode="visits", alpha_min=0.01, prefer=GATEWAY)
    # safe prior: before learning, the gateway is assumed to be the safe choice; the ESP32 starts at
    # what its lower accuracy is worth, the cloud has to earn trust
    d = reward.describe()
    agent.init_prior(d["gateway_on_time"] / (1 - gamma), cloud_handicap=1.0, edge_value=d["edge_on_time"] / (1 - gamma))
    curve = []
    t0 = time.time()
    rw, ms, cs, es = [], [], [], []
    for ep in range(episodes):
        frac = ep / episodes
        agent.epsilon = max(eps_min, 0.5 * (1.0 - frac / 0.6))
        sched = random_schedule(rng, boxes)
        pol = QPolicy(agent, space, learn=True, explore=True)
        log = make_sim(params, reward, rng, "val").run(pol, sched, boxes, log_conditions=False)
        s = summarize(log)
        rw.append(s["reward_mean"]); ms.append(s["miss_rate"]); cs.append(s["cloud_share"]); es.append(s["edge_share"])
        if (ep + 1) % log_every == 0:
            row = {"episode": ep + 1, "boxes_seen": (ep + 1) * boxes, "epsilon": round(agent.epsilon, 3),
                   "alpha": round(agent.alpha, 3), "reward": float(np.mean(rw)),
                   "miss_rate": float(np.mean(ms)), "cloud_share": float(np.mean(cs)),
                   "edge_share": float(np.mean(es))}
            curve.append(row)
            rw, ms, cs, es = [], [], [], []
            if (ep + 1) % (log_every * 10) == 0:
                print(f"  ep {ep+1:5d}  eps={row['epsilon']:.2f}  reward={row['reward']:.3f}  "
                      f"miss={row['miss_rate']:.2%}  cloud={row['cloud_share']:.0%}  esp32={row['edge_share']:.0%}  "
                      f"({time.time()-t0:.0f}s)",
                      flush=True)
    return agent, curve


def plot_curve(curve, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [c["boxes_seen"] / 1000 for c in curve]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(x, [c["reward"] for c in curve], color="#2563eb")
    ax[0].set_xlabel("boxes simulated (thousands)"); ax[0].set_ylabel("mean reward per box")
    ax[0].set_title("Pre-training: reward goes up")
    ax[1].plot(x, [100 * c["miss_rate"] for c in curve], color="#dc2626")
    ax[1].set_xlabel("boxes simulated (thousands)"); ax[1].set_ylabel("missed boxes (%)")
    ax[1].set_title("Pre-training: missed boxes go down")
    for a in ax:
        a.grid(alpha=0.3)
    ax2 = ax[0].twiny() if False else None
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--episodes", type=int, default=16000)
    ap.add_argument("--boxes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-rules", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    params, reward, space = build(cfg, a.calib)
    d = reward.describe()
    print("reward per box:", {k: round(v, 3) for k, v in d.items() if k != "by_confidence"})
    print(f"photos replayed by the simulator: {reward.photos.source} "
          f"({len(reward.photos.val)} to learn from, {len(reward.photos.test)} to be judged on)")
    for lv in d.get("by_confidence", []):
        print(f"  gateway {lv['name']:<12} {100*lv['share']:4.0f}% of boxes: keep its verdict {lv['gateway']:6.2f}, "
              f"cloud answer {lv['cloud']:6.2f}  -> a cloud call is worth {lv['cloud'] - lv['gateway']:+.2f}")
    print(f"state space: {space.dims} = {space.n_states} states x 3 actions (ESP32 / gateway / cloud)")

    os.makedirs(project_path("results/figures"), exist_ok=True)
    print("pre-training Q-agent ...")
    agent, curve = pretrain(params, reward, space, a.episodes, a.boxes, a.seed, gamma=cfg["agent"]["gamma"])
    meta = {"trained": time.strftime("%Y-%m-%d %H:%M:%S"), "episodes": a.episodes, "boxes": a.boxes,
            "boxes_total": a.episodes * a.boxes, "state_dims": list(space.dims),
            "state_bins": cfg["state_bins"], "reward": reward.describe(), "sim_params": params.__dict__}
    agent.alpha, agent.epsilon = cfg["agent"]["alpha"], cfg["agent"]["epsilon"]
    agent.alpha_mode = "constant"          # online: keep adapting
    agent.explore_gap = cfg["agent"].get("explore_gap")
    agent.forget = float(cfg["agent"].get("forget", 0.0))
    agent.q0 = agent.q.copy()
    qpath = project_path(cfg["agent"]["q_table"])
    agent.save(qpath, meta)
    print(f"saved {qpath}  (states visited: {(agent.visits.sum(1) > 0).sum()}/{space.n_states})")

    with open(project_path("results/pretrain_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        w.writeheader(); w.writerows(curve)
    plot_curve(curve, project_path("results/figures/pretrain_curve.png"))

    if not a.skip_rules:
        print("tuning baseline rules on the same random conditions ...")
        best, eval_set = tune_rules(params, reward)
        boxes = 150
        r, m = evaluate_policy(lambda: QPolicy(agent.copy(), space, learn=False, explore=False),
                               params, reward, eval_set, boxes)
        print(f"  Q-agent (frozen, greedy) on same set: reward={r:.3f} miss={m:.3%}")
        best["_photos"] = reward.photos.source
        best["_note"] = "tuned by sim/train_q.py on randomised conditions (same as Q pre-training)"
        with open(project_path(cfg["agent"]["rule_params"]), "w") as f:
            json.dump(best, f, indent=2)
        print(f"saved {cfg['agent']['rule_params']}")


if __name__ == "__main__":
    main()
