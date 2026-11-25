"""
The reinforcement-learning core, shared by the real gateway and the simulator.

  * Obs         - what the gateway measures at the moment a box arrives
  * StateSpace  - turns an Obs into one state number (the Q-table row)
  * RewardModel - turns "what happened" into a reward number
  * QAgent      - the Q-table and the Q-learning update rule

Plain-words summary
-------------------
Three places can look at a box:
  0 = EDGE     the ESP32 on the belt: tiny model, lowest accuracy, needs no laptop CPU and no Internet
  1 = GATEWAY  the laptop next to the belt: small model, good accuracy, fast unless the laptop is overloaded
  2 = CLOUD    the AWS server: big model, best accuracy, but the photo has to cross the Internet

Second opinion (how the three are used together)
  The gateway's model checks every box first whenever it clearly fits in the time left. Its verdict is kept as a
  fallback. The agent then chooses: GATEWAY = keep that verdict, CLOUD = ask the cloud for a second
  opinion (a late cloud answer falls back to the gateway's verdict; the box is not missed).
  Only when the gateway is too slow does the agent pick one place outright, EDGE included.
  Cloud calls are limited (so many per minute), so the agent has to spend them on the boxes that need them.

State  : (spare time if sent to the cloud, cloud speed, ping, can the gateway finish in time, cloud queue,
          how sure the gateway's model is about this box, how many cloud calls are left right now)
Reward : minus the expected cost of the box, given who answered and how sure the gateway was
         (escaped defect / false reject / box missed because no answer came in time), minus a little for energy.
Update : Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from .algos import bin_index
from .cascade import CONF_NAMES, N_CONF, level_costs

EDGE, GATEWAY, CLOUD = 0, 1, 2
ACTIONS = ["EDGE", "GATEWAY", "CLOUD"]
PLACE = {EDGE: "ESP32", GATEWAY: "gateway", CLOUD: "cloud"}      # plain names for print-outs
CARD_ROLE = {EDGE: "edge", GATEWAY: "gateway", CLOUD: "cloud"}   # keys in models/model_card.json


@dataclass
class Obs:
    rtt_ms: float          # smoothed ping time to the cloud server (ms)
    budget_s: float        # seconds left until the box reaches the reject arm
    gw_q: int              # jobs already waiting/running on the gateway's own model
    cloud_q: int           # requests already in flight to the cloud
    cpu: float             # gateway CPU utilisation 0..1 (shown on the dashboard; used by the deadline rule)
    cloud_ms: float = 0.0  # smoothed FULL cloud response time (test image every 2 s + real requests)
    gw_ms: float = 0.0     # smoothed time the gateway's own model needs for one photo (0 = not measured yet)
    edge_ms: float = 0.0   # smoothed time the ESP32 needs for one photo incl. the USB link (0 = not measured)
    conf: int = 0          # how sure the gateway's model is about THIS box: 0 = it has not run, 1 sure ... 4 very unsure
    tokens: float = 1.0    # share of the cloud-call budget left (1.0 = full or unlimited)
    calls_left: float = 99.0   # cloud calls that could be made right now (99 = no limit)
    can_cloud: bool = True  # False = no cloud call left right now
    oracle: int = 0        # simulator only (for the "best possible" line): 1 = a cloud answer would fix this box

    def as_dict(self):
        return asdict(self)


def load_card(path: str | None) -> dict:
    """models/model_card.json. Cards written before the ESP32 was added call the small model "edge";
    it is the gateway's model now, so such a card is read as {"gateway": ..., "cloud": ...}."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        card = json.load(f)
    if "gateway" not in card and "edge" in card and "mobilenet" in str(card["edge"].get("arch", "")):
        card["gateway"] = card.pop("edge")
    return card


class StateSpace:
    """State = 7 numbers, each cut into a few bins (8 x 3 x 3 x 3 x 2 x 5 x 5 = 10,800 states):

      slack    : SPARE TIME if this box is sent to the cloud =
                 time left before the arm - predicted cloud answer time
                 (predicted = max(measured cloud time, ping + fixed overhead))    8 bins
      cloud    : measured cloud response time level (fast / slow / very slow)     3 bins
      rtt      : ping time level (good / busy / bad)                              3 bins
      gateway  : can the gateway's own model answer in time?
                 predicted gateway time (queue x measured time) / time left:
                 fits easily / tight / too slow                                   3 bins
      cloud_q  : cloud requests already waiting (0 / 1+)                          2 bins
      conf     : how sure the gateway's model is about this box
                 (not run / sure / fairly sure / unsure / very unsure)            5 bins
      budget   : cloud calls left right now (none / 1 / 2 / 3-4 / 5 or more;
                 "5 or more" also means "no limit": a call spent now is not missed later)    5 bins

    The agent learns for itself how much spare time is "safe enough" in each
    situation - that is exactly the safety margin a hand-made rule needs tuned.
    The ESP32 needs no state of its own: its answer time is nearly constant.
    """

    def __init__(self, bins: dict):
        self.slack_edges = sorted(bins["slack_ms"])     # sorted: required by the binary search
        self.cloud_edges = sorted(bins["cloud_ms"])
        self.rtt_edges = sorted(bins["rtt_ms"])
        self.gw_edges = sorted(bins.get("gw_fit", [0.4, 1.0]))
        self.gw_default_ms = float(bins.get("gw_default_ms", 10.0))
        self.overhead_ms = float(bins.get("cloud_overhead_ms", 30.0))
        self.budget_edges = sorted(bins.get("budget_calls", [1, 2, 3, 5]))   # calls left: none / 1 / 2 / 3-4 / 5+
        self.dims = (len(self.slack_edges) + 1, len(self.cloud_edges) + 1, len(self.rtt_edges) + 1, 3, 2,
                     N_CONF, len(self.budget_edges) + 1)
        self.n_states = int(np.prod(self.dims))

    @staticmethod
    def _bin(x, edges):
        # binary search over the sorted bin edges (O(log B)); same answer as a linear scan
        return bin_index(x, edges)

    def predicted_cloud_ms(self, o: Obs) -> float:
        return max(o.cloud_ms, o.rtt_ms + self.overhead_ms)

    def slack_ms(self, o: Obs) -> float:
        return o.budget_s * 1000.0 - self.predicted_cloud_ms(o)

    def predicted_gw_ms(self, o: Obs) -> float:
        return (o.gw_q + 1) * (o.gw_ms if o.gw_ms > 0 else self.gw_default_ms)

    def gw_load(self, o: Obs) -> int:
        fit = self.predicted_gw_ms(o) / max(1.0, o.budget_s * 1000.0)
        return 0 if fit < self.gw_edges[0] else (1 if fit < self.gw_edges[1] else 2)

    def bins_of(self, o: Obs):
        return (self._bin(self.slack_ms(o), self.slack_edges),
                self._bin(o.cloud_ms, self.cloud_edges),
                self._bin(o.rtt_ms, self.rtt_edges),
                self.gw_load(o),
                1 if o.cloud_q >= 1 else 0,
                min(max(int(o.conf), 0), N_CONF - 1),
                self._bin(o.calls_left, self.budget_edges))

    def index(self, o: Obs) -> int:
        # mixed-radix number (Horner's rule): same as np.ravel_multi_index, without the numpy call overhead
        idx = 0
        for b, d in zip(self.bins_of(o), self.dims):
            idx = idx * d + b
        return idx

    def unravel(self, idx: int):
        return np.unravel_index(idx, self.dims)

    def labels(self):
        def rng_labels(edges, unit):
            lab = [f"<{edges[0]}{unit}"]
            for lo, hi in zip(edges[:-1], edges[1:]):
                lab.append(f"{lo}..{hi}{unit}")
            lab.append(f">{edges[-1]}{unit}")
            return lab
        return {
            "slack_ms": rng_labels(self.slack_edges, "ms"),
            "cloud_ms": ["fast", "slow", "very slow"][: len(self.cloud_edges) + 1],
            "rtt_ms": ["good", "busy", "bad"][: len(self.rtt_edges) + 1],
            "gw_load": ["fits", "tight", "too slow"],
            "cloud_q": ["0", "1+"],
            "conf": list(CONF_NAMES),
            "budget": (["none", "1", "2", "3 or more"] if len(self.budget_edges) == 3 else
                       ["none", "1", "2", "3-4", "5 or more"])[: len(self.budget_edges) + 1],
        }


DEFAULT_METRICS = {     # used only when the model card has no entry for a place
    "edge": {"recall_damaged": 0.85, "false_reject_rate": 0.30, "accuracy": 0.75},
    "gateway": {"recall_damaged": 0.90, "false_reject_rate": 0.06, "accuracy": 0.93},
    "cloud": {"recall_damaged": 0.98, "false_reject_rate": 0.02, "accuracy": 0.98},
}


class RewardModel:
    """r = -(expected cost of this box) - energy_weight * joules

    Uses only things the gateway knows at run time (no need to know the true label):
    each model's measured recall / false-reject rate from models/model_card.json,
    and whether the answer arrived before the box reached the arm.
    """

    def __init__(self, reward_cfg: dict, defect_rate: float, gw_metrics: dict, cloud_metrics: dict,
                 edge_metrics: dict | None = None, cascade: dict | None = None):
        self.c_esc = float(reward_cfg["cost_escape"])
        self.c_fr = float(reward_cfg["cost_false_reject"])
        self.c_unchk = float(reward_cfg["cost_unchecked"])
        self.w_e = float(reward_cfg["energy_weight"])
        self.p_def = float(defect_rate)
        self.metrics = {EDGE: edge_metrics or dict(DEFAULT_METRICS["edge"]), GATEWAY: gw_metrics, CLOUD: cloud_metrics}
        self.cascade = cascade                                   # per-confidence-level counts (model card)
        self.level = level_costs(cascade, self.p_def, self.c_esc, self.c_fr,
                                 {"gateway": self._flat(GATEWAY), "cloud": self._flat(CLOUD)})

    def _flat(self, place: int) -> float:
        m = self.metrics[place]
        miss_rate_damaged = 1.0 - m["recall_damaged"]
        return self.p_def * miss_rate_damaged * self.c_esc + (1 - self.p_def) * m["false_reject_rate"] * self.c_fr

    def expected_cost(self, place: int, on_time: bool, conf: int = 0) -> float:
        """place = who ANSWERED (not who was asked). conf = how sure the gateway's model was (0 = it did not run)."""
        if not on_time:
            return self.p_def * self.c_esc + self.c_unchk
        if conf > 0 and place != EDGE:
            return self.level["cloud" if place == CLOUD else "gateway"][min(conf, N_CONF - 1)]
        return self._flat(place)

    def __call__(self, place: int, on_time: bool, energy_j: float, conf: int = 0) -> float:
        return -self.expected_cost(place, on_time, conf) - self.w_e * energy_j

    def gain(self, conf: int) -> float:
        """What a cloud answer is worth on a box where the gateway was this sure (cost saved per call)."""
        return self.expected_cost(GATEWAY, True, conf) - self.expected_cost(CLOUD, True, conf)

    def describe(self) -> dict:
        d = {
            "edge_on_time": -self.expected_cost(EDGE, True),
            "gateway_on_time": -self.expected_cost(GATEWAY, True),
            "cloud_on_time": -self.expected_cost(CLOUD, True),
            "missed": -self.expected_cost(GATEWAY, False),
        }
        if self.level.get("informative"):
            d["by_confidence"] = [{"level": c, "name": CONF_NAMES[c], "share": round(self.level["share"][c], 3),
                                   "gateway": round(-self.level["gateway"][c], 2), "cloud": round(-self.level["cloud"][c], 2)}
                                  for c in range(1, N_CONF)]
        return d

    @staticmethod
    def metrics_from_card(card: dict) -> dict:
        out = {}
        for role in ("edge", "gateway", "cloud"):
            d = dict(DEFAULT_METRICS[role])
            src = (card.get(role) or {}).get("test") or (card.get(role) or {}).get("val")
            if src:
                d.update({k: src[k] for k in ("recall_damaged", "false_reject_rate", "accuracy") if k in src})
            out[role] = d
        return out

    @staticmethod
    def from_card(cfg: dict, card_path: str | None):
        card = load_card(card_path)
        m = RewardModel.metrics_from_card(card)
        return RewardModel(cfg["reward"], cfg["conveyor"]["defect_rate_assumed"], m["gateway"], m["cloud"], m["edge"],
                           card.get("cascade"))


class QAgent:
    """Tabular Q-learning.

    alpha_mode = "constant": alpha stays fixed (used ONLINE so the agent keeps
                             adapting when things change)
    alpha_mode = "visits"  : alpha = max(alpha_min, 1/(1+visits)^0.6) per (state, action)
                             (used in PRE-TRAINING: averages out simulator noise)
    Exploration: with probability epsilon the agent tries ANOTHER option.
    prefer = the option taken when values are equal (the conveyor uses GATEWAY: local, fast, accurate).
    """

    def __init__(self, n_states: int, n_actions: int = 3, alpha=0.2, gamma=0.6, epsilon=0.05, seed=None,
                 alpha_mode: str = "constant", alpha_min: float = 0.01, prefer: int | None = None):
        self.q = np.zeros((n_states, n_actions), np.float64)
        self.visits = np.zeros((n_states, n_actions), np.int64)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.alpha_mode, self.alpha_min = alpha_mode, alpha_min
        self.explore_gap = None          # None = plain epsilon-greedy
        self.forget = 0.0                # online: pull values of the option NOT taken back toward q0
        self.q0 = None                   # snapshot of the pre-trained table
        self.rng = np.random.default_rng(seed)
        self.meta = {}
        self.prefer = prefer

    def init_prior(self, gateway_value: float, cloud_handicap: float = 1.0, edge_value: float | None = None):
        """Before learning: assume the gateway is the safe choice. The cloud starts slightly
        worse and has to 'earn trust' from rewards. The ESP32 starts at what its (lower)
        accuracy is worth when it answers in time."""
        self.q[:, GATEWAY] = gateway_value
        self.q[:, CLOUD] = gateway_value - cloud_handicap
        if self.q.shape[1] > 2:
            self.q[:, EDGE] = gateway_value - cloud_handicap if edge_value is None else edge_value
        self.prefer = GATEWAY

    def greedy(self, s: int, allowed=None, cloud_bonus: float = 0.0) -> int:
        """Best option in state s. allowed = the options open right now (None = all).
        cloud_bonus = added to the value of CLOUD for this one decision (what the line's own spot checks
        say a cloud answer is worth, minus what the model card said; 0 = use the table as it is)."""
        row = self.q[s]
        if cloud_bonus:
            row = row.copy()
            row[CLOUD] += cloud_bonus
        if allowed is None:
            a = int(np.argmax(row))
        else:
            a = max(allowed, key=lambda x: row[x])
        # ties -> the preferred (safe, local) choice
        if self.prefer is not None and (allowed is None or self.prefer in allowed) and row[self.prefer] >= row[a]:
            return self.prefer
        return a

    def act(self, s: int, explore: bool = True, allowed=None, cloud_bonus: float = 0.0) -> int:
        """epsilon-greedy with SAFE exploration: another option is only tried
        when its value is close to the best one (gap < explore_gap). If the
        agent is sure an option is bad, it does not gamble a real box on it."""
        g = self.greedy(s, allowed, cloud_bonus)
        if explore and self.rng.random() < self.epsilon:
            opts = list(range(self.q.shape[1])) if allowed is None else list(allowed)
            if self.explore_gap is None:
                return opts[int(self.rng.integers(len(opts)))]
            row = self.q[s]
            if cloud_bonus:
                row = row.copy()
                row[CLOUD] += cloud_bonus
            close = [a for a in opts if a != g and (row[g] - row[a]) < self.explore_gap]
            if close:
                return close[int(self.rng.integers(len(close)))]
        return g

    def fade(self, s: int, taken: int):
        """'Bad memories fade': the value of the option NOT taken drifts back toward
        what pre-training said, so the agent eventually re-checks it after a fault."""
        if self.forget > 0 and self.q0 is not None:
            for o in range(self.q.shape[1]):
                if o != taken:
                    self.q[s, o] += self.forget * (self.q0[s, o] - self.q[s, o])

    def greedy_value(self, s: int) -> float:
        return float(np.max(self.q[s]))

    def update(self, s: int, a: int, r: float, s2: int | None):
        target = r if s2 is None else r + self.gamma * self.greedy_value(s2)
        if self.alpha_mode == "visits":
            lr = max(self.alpha_min, 1.0 / (1.0 + self.visits[s, a]) ** 0.6)
        else:
            lr = self.alpha
        self.q[s, a] += lr * (target - self.q[s, a])
        self.visits[s, a] += 1

    def _arrays(self, meta: dict | None = None) -> dict:
        return dict(q=self.q, visits=self.visits, meta=json.dumps(meta or self.meta or {}),
                    alpha=self.alpha, gamma=self.gamma, epsilon=self.epsilon,
                    explore_gap=-1.0 if self.explore_gap is None else self.explore_gap,
                    forget=self.forget, q0=self.q0 if self.q0 is not None else self.q,
                    prefer=-1 if self.prefer is None else self.prefer)

    def save(self, path: str, meta: dict | None = None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez(tmp, **self._arrays(meta))
        os.replace(tmp, path)            # atomic: a power cut never leaves half a Q-table

    def to_bytes(self, meta: dict | None = None) -> bytes:
        """The same .npz file, in memory (uploaded to the cloud as a backup)."""
        buf = io.BytesIO()
        np.savez(buf, **self._arrays(meta))
        return buf.getvalue()

    @staticmethod
    def load(path: str, alpha=None, gamma=None, epsilon=None, seed=None):
        z = np.load(path, allow_pickle=False)
        ag = QAgent(z["q"].shape[0], z["q"].shape[1],
                    float(alpha if alpha is not None else z["alpha"]),
                    float(gamma if gamma is not None else z["gamma"]),
                    float(epsilon if epsilon is not None else z["epsilon"]), seed)
        ag.q = z["q"].copy()
        ag.visits = z["visits"].copy()
        ag.meta = json.loads(str(z["meta"]))
        if "explore_gap" in z.files and float(z["explore_gap"]) >= 0:
            ag.explore_gap = float(z["explore_gap"])
        if "forget" in z.files:
            ag.forget = float(z["forget"])
        ag.q0 = z["q0"].copy() if "q0" in z.files else z["q"].copy()
        if "prefer" in z.files:
            ag.prefer = None if int(z["prefer"]) < 0 else int(z["prefer"])
        return ag

    def maps(self, space: "StateSpace", state: int | None) -> list:
        """Two small pictures of what the agent would do, for the dashboard (drawn for the situation of
        the given state, normally the last box):
          1. 'when it asks the cloud': rows = how sure the gateway's model is, columns = cloud calls left
          2. 'which place when the gateway cannot go first': rows = spare time, columns = cloud speed"""
        lab = space.labels()
        d = space.dims
        cur = [int(x) for x in space.unravel(int(state))] if state is not None else [d[0] - 1, 0, 0, 0, 0, 3, d[6] - 1]

        def cell(idx, allowed, names):
            row = self.q[idx]
            best = self.greedy(idx, allowed)
            rest = [row[a] for a in allowed if a != best]
            gap = float(row[best] - max(rest)) if rest else 0.45          # nothing else to choose: shown faint
            return {"b": names[best], "t": round(max(0.0, min(1.0, gap / 3.0)), 2), "s": bool(self.visits[idx].sum() > 0)}

        ask = {"title": f"When it asks the cloud. Shown for: spare time {lab['slack_ms'][cur[0]]}, cloud {lab['cloud_ms'][cur[1]]}, "
                        f"ping {lab['rtt_ms'][cur[2]]}",
               "rows": lab["conf"][1:], "cols": [f"{b} left" for b in lab["budget"]], "ylabel": "gateway's model is",
               "cur": [cur[5] - 1, cur[6]] if cur[5] > 0 else None, "cells": []}
        for c in range(1, d[5]):
            line = []
            for b in range(d[6]):
                idx = int(np.ravel_multi_index((cur[0], cur[1], cur[2], 0, 0, c, b), d))
                line.append(cell(idx, (GATEWAY, CLOUD) if b > 0 else (GATEWAY,), {GATEWAY: "keep", CLOUD: "ask cloud"}))
            ask["cells"].append(line)
        one = {"title": f"Gateway too slow to go first. Shown for: ping {lab['rtt_ms'][cur[2]]}, "
                        f"cloud calls left {lab['budget'][cur[6]]}",
               "rows": lab["slack_ms"], "cols": ["cloud " + x for x in lab["cloud_ms"]], "ylabel": "spare time",
               "cur": [cur[0], cur[1]] if cur[5] == 0 else None, "cells": []}
        for i in range(d[0]):
            line = []
            for j in range(d[1]):
                idx = int(np.ravel_multi_index((i, j, cur[2], 2, 0, 0, cur[6]), d))
                line.append(cell(idx, (EDGE, GATEWAY, CLOUD) if cur[6] > 0 else (EDGE, GATEWAY),
                                 {EDGE: "ESP32", GATEWAY: "gateway", CLOUD: "cloud"}))
            one["cells"].append(line)
        return [ask, one]

    def value_gap(self) -> np.ndarray:
        """Q(CLOUD) - Q(GATEWAY) for every state (positive = cloud preferred over the gateway)."""
        return self.q[:, CLOUD] - self.q[:, GATEWAY]

    def choice_map(self) -> np.ndarray:
        """The place the agent would pick in every state (0 ESP32, 1 gateway, 2 cloud)."""
        return np.array([self.greedy(s) for s in range(self.q.shape[0])])

    def copy(self, seed=None):
        ag = QAgent(self.q.shape[0], self.q.shape[1], self.alpha, self.gamma, self.epsilon, seed)
        ag.q = self.q.copy()
        ag.visits = self.visits.copy()
        ag.meta = dict(self.meta)
        ag.explore_gap = self.explore_gap
        ag.alpha_mode, ag.alpha_min = self.alpha_mode, self.alpha_min
        ag.forget = self.forget
        ag.q0 = self.q0.copy() if self.q0 is not None else self.q.copy()
        ag.prefer = self.prefer
        return ag
