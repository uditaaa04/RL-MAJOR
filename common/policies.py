"""
Decision policies: the Q-learning agent and the baselines it is compared with.

Every policy has the same two methods, so the real gateway and the simulator can
swap them with one config line (agent.policy):

    decide(box_id, obs)                                  -> 0 (EDGE = ESP32), 1 (GATEWAY) or 2 (CLOUD)
    outcome(box_id, action, on_time, latency_s, reward)  -> called when the answer arrives
                                                            (or when the box passes the arm)

Baselines
---------
always_edge    : always the ESP32's tiny model
always_gateway : always the gateway's small model
always_cloud   : always the cloud's big model
threshold      : "if ping < T ms use the cloud, else the gateway"          (1 tuned number)
deadline       : predicts all three times from fixed estimates and takes the
                 most accurate place that fits in the time left (+ margin)  (hand-built, multi-factor)
adaptive       : like `deadline`, but with the MEASURED cloud, gateway and
                 ESP32 times instead of fixed estimates                     (hand-built, self-adjusting)
qlearning      : the RL agent (learns from rewards; can keep learning online)

"Pick one" and "second opinion"
-------------------------------
The rules above pick ONE place per box (gateway_first = False). The policies below work the
second-opinion way (gateway_first = True): the gateway's model has already checked the box when
decide() is called (obs.conf says how sure it was), so the choice is GATEWAY = keep that verdict or
CLOUD = also ask the cloud. A late cloud answer then falls back to the gateway's verdict.

ask_always     : ask the cloud for every box while cloud calls are left
confidence     : ask the cloud when the gateway's model is unsure enough AND the measured cloud
                 time fits in the time left                                 (2 tuned numbers)
confidence_reserve : the same, but when the cloud-call budget runs low only the least-sure boxes
                 are sent                                                   (3 tuned numbers)
qlearning      : learns all of that from rewards, including how to spend the cloud-call budget
"""
from __future__ import annotations

from .rl import CLOUD, EDGE, GATEWAY, Obs, QAgent, StateSpace


def _cpu_mult(cpu: float) -> float:
    return 1.0 if cpu < 0.5 else (1.5 if cpu < 0.85 else 3.0)


class Policy:
    name = "base"
    gateway_first = False      # True = second-opinion policy (the gateway's model runs before decide())

    def decide(self, box_id: int, obs: Obs) -> int:
        raise NotImplementedError

    def outcome(self, box_id: int, action: int, on_time: bool, latency_s: float, reward: float):
        pass

    def cloud_call(self, answered_in_time: bool):
        """Second opinion: a cloud call was made; was its answer in time? (only the learner uses this)"""

    def spot_check(self, conf: int, saving: float):
        """A box that had both verdicts was checked by hand afterwards. (only the learner uses this)"""

    def params(self) -> dict:
        return {}


class AlwaysEdge(Policy):
    name = "Always-Edge"

    def decide(self, box_id, obs):
        return EDGE


class AlwaysGateway(Policy):
    name = "Always-Gateway"

    def decide(self, box_id, obs):
        return GATEWAY


class AlwaysCloud(Policy):
    name = "Always-Cloud"

    def decide(self, box_id, obs):
        return CLOUD


class ThresholdRule(Policy):
    name = "Threshold rule"

    def __init__(self, rtt_threshold_ms: float = 100.0, **_):
        self.T = rtt_threshold_ms

    def decide(self, box_id, obs):
        return CLOUD if obs.rtt_ms < self.T else GATEWAY

    def params(self):
        return {"rtt_threshold_ms": self.T}


class DeadlineRule(Policy):
    """Predict how long each place takes using fixed (calibrated) estimates, then take the
    most accurate place that still fits: cloud, else gateway, else ESP32."""
    name = "Deadline rule"

    def __init__(self, margin_s=0.05, gw_s=0.008, jpeg_s=0.003, payload_kb=16.0,
                 bw_mbps=20.0, server_s=0.015, esp_s=0.045, **old):
        self.margin = margin_s
        self.gw_s = old.get("edge_s", gw_s)          # "edge_s": name used before the ESP32 was added
        self.jpeg_s, self.payload_kb = jpeg_s, payload_kb
        self.bw, self.server_s, self.esp_s = bw_mbps, server_s, esp_s

    def predict(self, obs: Obs):
        """-> predicted seconds for (ESP32, gateway, cloud)"""
        m = _cpu_mult(obs.cpu)
        pred_gw = (obs.gw_q + 1) * self.gw_s * m
        upload = self.payload_kb * 8e3 / (self.bw * 1e6) * (obs.cloud_q + 1)
        pred_cloud = self.jpeg_s * m + upload + obs.rtt_ms / 1000.0 + self.server_s
        return self.esp_s, pred_gw, pred_cloud

    def _pick(self, pe, pg, pc, budget_s):
        for a, t in ((CLOUD, pc), (GATEWAY, pg), (EDGE, pe)):      # most accurate first
            if t + self.margin <= budget_s:
                return a
        return min(((pg, GATEWAY), (pe, EDGE), (pc, CLOUD)))[1]     # nothing fits: the fastest

    def decide(self, box_id, obs):
        return self._pick(*self.predict(obs), obs.budget_s)

    def params(self):
        return {"margin_s": self.margin, "gw_s": self.gw_s, "payload_kb": self.payload_kb,
                "bw_mbps": self.bw, "server_s": self.server_s, "esp_s": self.esp_s}


class AdaptiveRule(DeadlineRule):
    """Deadline rule that uses the MEASURED times (cloud: test image every 2 s + real requests;
    gateway and ESP32: their own recent answers) instead of fixed estimates.

    predicted_cloud = max(measured cloud time, ping-based estimate)
    """
    name = "Adaptive rule"

    def decide(self, box_id, obs):
        pe, pg, pc_static = self.predict(obs)
        pc = max(obs.cloud_ms / 1000.0, pc_static) if obs.cloud_ms > 0 else pc_static
        if obs.gw_ms > 0:
            pg = (obs.gw_q + 1) * obs.gw_ms / 1000.0
        if obs.edge_ms > 0:
            pe = obs.edge_ms / 1000.0
        return self._pick(pe, pg, pc, obs.budget_s)


class SecondOpinion(AdaptiveRule):
    """Base of the hand-written second-opinion rules. When the gateway's model could not run first
    (obs.conf == 0: the laptop is overloaded) it picks one place like the adaptive rule."""
    name = "second opinion"
    gateway_first = True

    def cloud_fits(self, obs: Obs) -> bool:
        _, _, pc_static = self.predict(obs)
        pc = max(obs.cloud_ms / 1000.0, pc_static) if obs.cloud_ms > 0 else pc_static
        return pc + self.margin <= obs.budget_s

    def ask(self, obs: Obs) -> bool:
        raise NotImplementedError

    def decide(self, box_id, obs):
        if obs.conf <= 0:                                   # no gateway verdict: pick one place
            a = AdaptiveRule.decide(self, box_id, obs)
            if a == CLOUD and not obs.can_cloud:
                pe = obs.edge_ms / 1000.0 if obs.edge_ms > 0 else self.esp_s
                pg = (obs.gw_q + 1) * (obs.gw_ms / 1000.0 if obs.gw_ms > 0 else self.gw_s)
                return GATEWAY if pg + self.margin <= obs.budget_s or pg <= pe else EDGE
            return a
        return CLOUD if (obs.can_cloud and self.ask(obs)) else GATEWAY


class AskAlways(SecondOpinion):
    name = "Ask the cloud always"

    def ask(self, obs):
        return True


class ConfidenceRule(SecondOpinion):
    """Ask the cloud when the gateway's model is at least `min_conf` unsure and the cloud fits in time."""
    name = "Confidence rule"

    def __init__(self, min_conf: int = 3, **kw):
        super().__init__(**kw)
        self.min_conf = int(min_conf)

    def ask(self, obs):
        return obs.conf >= self.min_conf and self.cloud_fits(obs)

    def params(self):
        return {**super().params(), "min_conf": self.min_conf}


class ConfidenceReserveRule(ConfidenceRule):
    """Confidence rule that keeps a reserve: when less than `reserve` of the cloud-call budget is left,
    only the very unsure boxes are sent."""
    name = "Confidence rule + reserve"

    def __init__(self, min_conf: int = 3, reserve: float = 0.4, **kw):
        super().__init__(min_conf, **kw)
        self.reserve = float(reserve)

    def ask(self, obs):
        need = 4 if obs.tokens < self.reserve else self.min_conf
        return obs.conf >= need and self.cloud_fits(obs)

    def params(self):
        return {**super().params(), "reserve": self.reserve}


class OraclePolicy(SecondOpinion):
    """Simulator only: knows which boxes a cloud answer would fix. Shows how much there is to win at most."""
    name = "Best possible (knows the answers)"

    def ask(self, obs):
        return obs.oracle == 1 and self.cloud_fits(obs)


class QPolicy(Policy):
    """Q-learning agent with delayed updates.

    The reward of box i is only known when its answer arrives (or the box
    passes the arm). The 'next state' s' is the state seen at box i+1.
    We update Q as soon as both are known - whichever comes last.
    """
    name = "Q-learning"
    gateway_first = True

    @staticmethod
    def allowed(obs: Obs):
        """The options open for this box."""
        if obs.conf > 0:                                    # the gateway's verdict exists: keep it or ask the cloud
            return (GATEWAY, CLOUD) if obs.can_cloud else (GATEWAY,)
        return (EDGE, GATEWAY, CLOUD) if obs.can_cloud else (EDGE, GATEWAY)

    def __init__(self, agent: QAgent, space: StateSpace, learn=True, explore=True, name=None, live=None):
        self.agent, self.space = agent, space
        self.learn, self.explore = learn, explore
        self.live = live                # common.cascade.LiveGain: what a cloud answer is worth, measured on the line
        if name:
            self.name = name
        self.states: dict[int, int] = {}
        self.pending: dict[int, tuple[int, float]] = {}
        self.last_state = None
        self.updates = 0

    def decide(self, box_id, obs):
        s = self.space.index(obs)
        self.states[box_id] = s
        self.last_state = s
        bonus = self.live.delta(obs.conf) if (self.live is not None and obs.conf > 0) else 0.0
        a = self.agent.act(s, explore=self.explore, allowed=self.allowed(obs), cloud_bonus=bonus)
        if self.learn:
            self.agent.fade(s, a)
        prev = box_id - 1
        if prev in self.pending:
            a_p, r_p = self.pending.pop(prev)
            self._update(prev, a_p, r_p, s)
        return a

    def outcome(self, box_id, action, on_time, latency_s, reward):
        nxt = self.states.get(box_id + 1)
        if nxt is not None:
            self._update(box_id, action, reward, nxt)
        else:
            self.pending[box_id] = (action, reward)

    def cloud_call(self, answered_in_time: bool):
        """A cloud call was made for a second opinion; did the answer come before the arm?"""
        if self.live is not None:
            self.live.note_call(answered_in_time)

    def spot_check(self, conf: int, saving: float):
        """A box with both verdicts turned out to be good / damaged: the saving of that cloud call is known."""
        if self.live is not None and self.learn:
            self.live.update(conf, saving)

    def _update(self, box_id, a, r, s2):
        # NOTE: do not delete states[box_id] here - it is still needed as the
        # "next state" of box_id-1 if that box's answer arrives later.
        s = self.states.get(box_id)
        if s is None:
            return
        if self.learn:
            self.agent.update(s, a, r, s2)
            self.updates += 1
        if len(self.states) > 2000:            # forget the 1000 oldest boxes
            # box ids only grow and dicts keep insertion order, so the oldest keys are
            # simply the first ones: O(1) each, no sort of all 2000 keys needed
            it = iter(self.states)
            for k in [next(it) for _ in range(1000)]:
                del self.states[k]

    def params(self):
        return {"alpha": self.agent.alpha, "gamma": self.agent.gamma,
                "epsilon": self.agent.epsilon if self.explore else 0.0, "learn_online": self.learn,
                "spot_checks": self.live is not None}


def make_policy(name: str, cfg: dict, agent: QAgent | None = None, rule_params: dict | None = None):
    """Build a policy by name (used by the gateway node)."""
    rp = rule_params or {}
    if name == "always_edge":
        return AlwaysEdge()
    if name == "always_gateway":
        return AlwaysGateway()
    if name == "always_cloud":
        return AlwaysCloud()
    if name == "threshold":
        return ThresholdRule(**rp.get("threshold", {}))
    if name == "deadline":
        return DeadlineRule(**rp.get("deadline", {}))
    if name == "adaptive":
        return AdaptiveRule(**rp.get("adaptive", {}))
    if name == "ask_always":
        return AskAlways(**rp.get("adaptive", {}))
    if name == "confidence":
        return ConfidenceRule(**rp.get("confidence", rp.get("adaptive", {})))
    if name == "confidence_reserve":
        return ConfidenceReserveRule(**rp.get("confidence_reserve", rp.get("adaptive", {})))
    if name == "qlearning":
        space = StateSpace(cfg["state_bins"])
        if agent is None:
            agent = QAgent(space.n_states, 3, cfg["agent"]["alpha"], cfg["agent"]["gamma"], cfg["agent"]["epsilon"],
                           prefer=GATEWAY)
        if cfg["agent"].get("explore_gap") is not None:
            agent.explore_gap = float(cfg["agent"]["explore_gap"])
        agent.alpha, agent.epsilon = float(cfg["agent"]["alpha"]), float(cfg["agent"]["epsilon"])
        agent.alpha_mode = "constant"
        agent.forget = float(cfg["agent"].get("forget", 0.0))
        return QPolicy(agent, space, learn=cfg["agent"]["learn_online"], explore=cfg["agent"]["learn_online"])
        # (the gateway attaches the spot-check tally itself: it needs the model card, see gateway_node/main.py)
    raise ValueError(f"unknown policy {name}")
