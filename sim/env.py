"""
Conveyor + network + ESP32 + gateway + cloud simulator (event-based, one box at a time).

What it models (all numbers come from SimParams, which tools/calibrate.py can
fill with values MEASURED on your real hardware):

  box arrives under camera  -> capture photo (~25 ms)
  decision: EDGE, GATEWAY or CLOUD
    EDGE   : shrink the photo, send it over the USB cable to the ESP32 (FIFO, ~24 ms),
             the ESP32 runs its tiny model (~10 ms)
    GATEWAY: wait for the gateway's small model to be free (FIFO), run it (~8 ms x gateway load)
    CLOUD  : JPEG encode, wait for the Wi-Fi link (FIFO), upload (size / bandwidth),
             one-way delay + jitter, TCP retransmissions on packet loss,
             wait for the cloud model (FIFO), run it, reply delay
  box reaches reject arm at  cam_to_arm / belt_speed - guard  seconds
  ON TIME if the answer arrived before that, otherwise MISSED (box passes unchecked)

Second opinion (policies with gateway_first = True)
  the gateway's model runs FIRST whenever its predicted time fits; the policy then decides with the
  model's confidence in hand: GATEWAY = keep that verdict, CLOUD = also ask the cloud. A cloud answer that
  comes too late falls back to the gateway's verdict (the box is NOT missed). Cloud calls are limited by
  a token bucket (Condition.budget_per_min, budget_burst).

Real photos
  every simulated box IS one real photo from the photo table (common/cascade.py): each model's verdict
  on the box is that model's real verdict on that photo. The photo sequence depends only on the seed,
  so every policy is judged on the very same boxes.

The agent only sees what the real gateway can measure: smoothed ping time, time left,
its own queues, and how long recent answers took. Server load and Wi-Fi bandwidth are
NOT visible to it (just like in reality) - it has to learn their effect.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field, replace

import numpy as np

from common.algos import CompletionHeap
from common.cascade import PhotoSplit, synthetic_table, verdict_cost
from common.rl import CLOUD, EDGE, GATEWAY, Obs, RewardModel


@dataclass
class SimParams:
    cam_to_arm_m: float = 0.10
    guard_s: float = 0.05
    capture_med_s: float = 0.025
    capture_sd_s: float = 0.005
    gw_med_s: float = 0.008         # small model on the gateway laptop (incl. pre-processing)
    gw_sigma: float = 0.20          # log-normal spread
    gw_probe_interval_s: float = 2.0  # the gateway times its own model this often when idle
    gw_ewma_alpha: float = 0.4
    thumb_s: float = 0.001          # shrinking the photo for the ESP32 (scales with gateway load)
    esp_link_s: float = 0.024       # 32x32 grey photo over the USB cable at 460800 baud + reply
    esp_med_s: float = 0.010        # tiny model on the ESP32
    esp_sigma: float = 0.08
    p_esp_w: float = 0.5            # ESP32 at full speed
    jpeg_s: float = 0.003
    payload_kb: float = 16.0        # 256x256 JPEG
    server_med_s: float = 0.015     # big model on the AWS EC2 server
    server_sigma: float = 0.25
    pkt_bytes: int = 1448
    fast_retx_prob: float = 0.7     # lost packet recovered by fast-retransmit (else timeout)
    min_rto_s: float = 0.20         # Linux minimum TCP retransmission timeout
    probe_interval_s: float = 0.5
    probe_timeout_s: float = 0.6
    ewma_alpha: float = 0.3
    cloud_probe_interval_s: float = 2.0
    cloud_probe_timeout_s: float = 1.0
    cloud_ewma_alpha: float = 0.4
    p_cpu_w: float = 3.0
    p_wifi_w: float = 0.6
    defect_rate: float = 0.30

    @staticmethod
    def from_calibration(d: dict) -> "SimParams":
        p = SimParams()
        for k, v in (d or {}).items():
            if v is not None and hasattr(p, k):
                setattr(p, k, type(getattr(p, k))(v))
        return p


@dataclass
class Condition:
    """The 'world' at one moment. Things marked (hidden) are invisible to the agent."""
    name: str = "custom"
    belt_mps: float = 0.20          # belt speed
    gap_s: float = 1.0              # mean time between boxes
    rtt_s: float = 0.030            # base network round trip (gateway -> AWS Mumbai)
    jitter_s: float = 0.004         # random extra delay per direction (std)
    loss: float = 0.0               # packet loss probability
    bw_mbps: float = 10.0           # (hidden) Wi-Fi + ISP upload bandwidth
    server_factor: float = 1.0      # (hidden) cloud server slowdown factor
    gw_factor: float = 1.0          # gateway slowdown factor (laptop busy with other work)
    cpu: float = 0.25               # gateway CPU load reading 0..1
    link_down: bool = False
    stall_prob: float = 0.0         # (hidden) chance a packet burst stalls (e.g. microwave-oven interference)
    stall_s: tuple = (0.4, 0.8)     # (hidden) how long a stall lasts
    budget_per_min: float = 0.0     # cloud calls allowed per minute (0 = no limit)
    budget_burst: float = 5.0       # how many unused calls can be saved up
    photos: str = ""                # which photo set the boxes come from ("" = the normal one; "blur", "dim" ... =
                                    # the same photos made harder, see tools/shift_scores.py)


@dataclass
class EpisodeLog:
    action: list = field(default_factory=list)
    on_time: list = field(default_factory=list)
    latency: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    energy: list = field(default_factory=list)
    label: list = field(default_factory=list)      # 1 = damaged
    correct: list = field(default_factory=list)    # verdict correct (False if missed)
    escaped: list = field(default_factory=list)    # damaged box shipped
    false_reject: list = field(default_factory=list)
    rtt_obs: list = field(default_factory=list)
    budget: list = field(default_factory=list)
    cond: list = field(default_factory=list)
    asked: list = field(default_factory=list)      # the cloud was asked for a second opinion
    fallback: list = field(default_factory=list)   # ... and answered too late: the gateway's verdict was used
    answered: list = field(default_factory=list)   # who gave the verdict that was used (-1 = nobody in time)
    conf: list = field(default_factory=list)       # how sure the gateway's model was (0 = it did not run first)
    reward_agent: list = field(default_factory=list)   # the expected-cost reward the agent learned from

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


def _lognorm(rng, med, sigma):
    return med * math.exp(sigma * rng.standard_normal())


def summarize(log: EpisodeLog | dict, start: int = 0, end: int | None = None) -> dict:
    a = log.arrays() if isinstance(log, EpisodeLog) else log
    sl = slice(start, end)
    act, ot = a["action"][sl], a["on_time"][sl]
    lab = a["label"][sl]
    n = len(act)
    n_dam = max(1, int(lab.sum()))
    lat = a["latency"][sl][ot.astype(bool)] if n else np.array([])
    return {
        "n": n,
        "miss_rate": float(1 - ot.mean()) if n else float("nan"),
        "escape_rate": float(a["escaped"][sl].sum() / n_dam),
        "false_reject_rate": float(a["false_reject"][sl].sum() / max(1, n - int(lab.sum()))),
        "accuracy": float(a["correct"][sl].mean()) if n else float("nan"),
        "cloud_share": float((act == CLOUD).mean()) if n else float("nan"),
        "gateway_share": float((act == GATEWAY).mean()) if n else float("nan"),
        "edge_share": float((act == EDGE).mean()) if n else float("nan"),
        "mean_latency_ms": float(lat.mean() * 1000) if len(lat) else float("nan"),
        "p95_latency_ms": float(np.percentile(lat, 95) * 1000) if len(lat) else float("nan"),
        "energy_mJ_per_box": float(a["energy"][sl].mean() * 1000) if n else float("nan"),
        # reward = minus the REAL cost of what happened to each box (shipped damaged / wasted good / unchecked)
        "reward_mean": float(a["reward"][sl].mean()) if n else float("nan"),
        "ask_share": float(a["asked"][sl].mean()) if n and len(a.get("asked", [])) else 0.0,
        "fallback_share": (float(a["fallback"][sl].sum() / max(1, a["asked"][sl].sum()))
                           if n and len(a.get("asked", [])) else 0.0),
        "cloud_answer_share": float((a["answered"][sl] == CLOUD).mean()) if n and len(a.get("answered", [])) else 0.0,
        "reward_agent_mean": float(a["reward_agent"][sl].mean()) if n and len(a.get("reward_agent", [])) else float("nan"),
    }


class ConveyorSim:
    def __init__(self, params: SimParams, reward: RewardModel, rng: np.random.Generator,
                 photos: PhotoSplit | None = None, gw_fit_limit: float = 1.0, photo_seed: int | None = None):
        self.p, self.reward, self.rng = params, reward, rng
        self.recall = {a: m["recall_damaged"] for a, m in reward.metrics.items()}
        self.frr = {a: m["false_reject_rate"] for a, m in reward.metrics.items()}
        if photos is None:                      # no photo table given: made-up scores that match the model metrics
            names = {EDGE: "edge", GATEWAY: "gateway", CLOUD: "cloud"}
            photos = synthetic_table({names[a]: m for a, m in reward.metrics.items()}, seed=12345).val
        self.photos = photos
        self.gw_fit_limit = float(gw_fit_limit)   # the gateway's model runs first when predicted time < limit x time left
                                                  # (callers pass 0.4 = the "fits" border of the state)
        # which photo each box is: fixed by the seed, NOT by what the policy does
        self.photo_seed = int(rng.integers(2 ** 31)) if photo_seed is None else int(photo_seed)
        self.photo_sets: dict = {}                # name -> PhotoSplit: the same photos made harder (Condition.photos)
        self.spot_rate = 0.0                      # share of boxes whose true state becomes known afterwards (spot checks)

    def _cloud_trip(self, c: Condition, t0: float, st: dict):
        """Send one photo to the cloud at time t0. -> (time the answer is back, energy used)."""
        p, r = self.p, self.rng
        enc = p.jpeg_s * c.gw_factor
        if c.link_down:
            return math.inf, p.p_cpu_w * enc
        up_start = max(t0 + enc, st["link_free"])
        upload = p.payload_kb * 8e3 / (c.bw_mbps * 1e6)
        st["link_free"] = up_start + upload
        n_pkts = int(math.ceil(p.payload_kb * 1000 / p.pkt_bytes))
        arrive = (up_start + upload + c.rtt_s / 2 + abs(r.normal(0, c.jitter_s))
                  + self._loss_delay(c, n_pkts) + self._stall(c))
        s_start = max(arrive, st["server_free"])
        s_svc = _lognorm(r, p.server_med_s, p.server_sigma) * c.server_factor
        st["server_free"] = s_start + s_svc
        down = c.rtt_s / 2 + abs(r.normal(0, c.jitter_s)) + self._loss_delay(c, 1)
        return s_start + s_svc + down, p.p_cpu_w * enc + p.p_wifi_w * upload

    # ------------------------------------------------------------------
    def _stall(self, c: Condition) -> float:
        if c.stall_prob > 0 and self.rng.random() < c.stall_prob:
            return self.rng.uniform(*c.stall_s)
        return 0.0

    def _probe_sample(self, c: Condition):
        r = self.rng
        if c.link_down:
            return self.p.probe_timeout_s
        if r.random() < 1 - (1 - c.loss) ** 2:
            return self.p.probe_timeout_s
        return min(self.p.probe_timeout_s,
                   c.rtt_s + abs(r.normal(0, c.jitter_s)) + abs(r.normal(0, c.jitter_s)) + self._stall(c))

    def _cloud_probe_sample(self, c: Condition) -> float:
        """Full cloud round trip for the 2-second test image (no queueing)."""
        p, r = self.p, self.rng
        if c.link_down:
            return p.cloud_probe_timeout_s
        n_pkts = int(math.ceil(p.payload_kb * 1000 / p.pkt_bytes))
        lat = (p.payload_kb * 8e3 / (c.bw_mbps * 1e6) + c.rtt_s + abs(r.normal(0, c.jitter_s))
               + abs(r.normal(0, c.jitter_s)) + self._loss_delay(c, n_pkts) + self._loss_delay(c, 1)
               + _lognorm(r, p.server_med_s, p.server_sigma) * c.server_factor + self._stall(c))
        return min(lat, p.cloud_probe_timeout_s)

    def _loss_delay(self, c: Condition, n_pkts: int) -> float:
        """Extra delay from TCP retransmissions of the k lost packets.

        Each lost packet is recovered by fast retransmit (one RTT, probability
        fast_retx_prob) or by a retransmission timeout. Instead of a Python loop
        over the k packets, draw all k uniforms at once and count the fast ones:
        delay = n_fast * RTT + n_slow * RTO. Same random numbers, same order.
        """
        if c.loss <= 0:
            return 0.0
        k = int(self.rng.binomial(n_pkts, c.loss))
        if k == 0:
            return 0.0
        rtt = c.rtt_s + 2 * c.jitter_s
        n_fast = int(np.count_nonzero(self.rng.random(k) < self.p.fast_retx_prob))
        return n_fast * rtt + (k - n_fast) * max(self.p.min_rto_s, 2 * rtt)

    # ------------------------------------------------------------------
    def run(self, policy, schedule, n_boxes: int, start_box_id: int = 0, log_conditions=True) -> EpisodeLog:
        """schedule: list of (first_box_index, Condition), sorted."""
        p, r = self.p, self.rng
        log = EpisodeLog()
        t = 0.0
        rtt_ewma = None
        gw_free = 0.0
        esp_free = 0.0
        st = {"link_free": 0.0, "server_free": 0.0}
        pr = np.random.default_rng(self.photo_seed)     # which photo each box is (same for every policy)
        sr = np.random.default_rng(self.photo_seed + 7)  # which boxes get a spot check (same for every policy)
        rm = self.reward
        tokens, t_tok = None, 0.0                       # cloud-call budget (token bucket)
        gw_done = CompletionHeap()    # completion times -> queue length at any moment (min-heap)
        cloud_done = CompletionHeap()
        feedback: list = []           # heap of (t_avail, box_id, action, on_time, latency, reward)
        sched_i = 0
        cond = schedule[0][1]
        last_probe = 0.0
        last_cprobe = 0.0
        cloud_ewma = None
        last_gprobe = 0.0
        gw_ewma = None
        esp_ewma = p.esp_link_s + p.esp_med_s      # nearly constant; refined by real answers

        for i in range(n_boxes):
            while sched_i + 1 < len(schedule) and i >= schedule[sched_i + 1][0]:
                sched_i += 1
                cond = schedule[sched_i][1]
            c = cond
            # ---- box arrival
            min_gap = 0.08 / c.belt_mps
            gap = max(min_gap, c.gap_s * r.uniform(0.6, 1.4))
            t = t + gap if i > 0 else 0.5
            # ---- network probes since last decision
            while last_probe + p.probe_interval_s <= t:
                last_probe += p.probe_interval_s
                s = self._probe_sample(c)
                rtt_ewma = s if rtt_ewma is None else (1 - p.ewma_alpha) * rtt_ewma + p.ewma_alpha * s
            if rtt_ewma is None:
                rtt_ewma = self._probe_sample(c)
            while last_cprobe + p.cloud_probe_interval_s <= t:
                last_cprobe += p.cloud_probe_interval_s
                s = self._cloud_probe_sample(c)
                cloud_ewma = s if cloud_ewma is None else (1 - p.cloud_ewma_alpha) * cloud_ewma + p.cloud_ewma_alpha * s
            if cloud_ewma is None:
                cloud_ewma = self._cloud_probe_sample(c)
            while last_gprobe + p.gw_probe_interval_s <= t:      # the gateway times its own model
                last_gprobe += p.gw_probe_interval_s
                g = _lognorm(r, p.gw_med_s, p.gw_sigma) * c.gw_factor
                gw_ewma = g if gw_ewma is None else (1 - p.gw_ewma_alpha) * gw_ewma + p.gw_ewma_alpha * g
            if gw_ewma is None:
                gw_ewma = _lognorm(r, p.gw_med_s, p.gw_sigma) * c.gw_factor
            # ---- capture
            cap = max(0.008, r.normal(p.capture_med_s, p.capture_sd_s))
            t_dec = t + cap
            deadline = t + p.cam_to_arm_m / c.belt_mps - p.guard_s
            # ---- deliver feedback that is known by now
            while feedback and feedback[0][0] <= t_dec:
                _, bid, a_f, ot_f, lat_f, rew_f, m_c, c_ok, m_g, m_e, spot = heapq.heappop(feedback)
                if m_c is not None:   # real cloud answers also update the measured cloud time
                    val = m_c if c_ok else max(m_c, p.cloud_probe_timeout_s)
                    cloud_ewma = (1 - p.cloud_ewma_alpha) * cloud_ewma + p.cloud_ewma_alpha * val
                if m_g is not None:
                    gw_ewma = (1 - p.gw_ewma_alpha) * gw_ewma + p.gw_ewma_alpha * m_g
                if m_e is not None:
                    esp_ewma = 0.7 * esp_ewma + 0.3 * m_e
                policy.outcome(bid, a_f, ot_f, lat_f, rew_f)
                if spot is not None:                 # a second opinion was asked for this box
                    policy.cloud_call(spot[0])
                    if spot[2] is not None:          # ... and the box was spot-checked afterwards
                        policy.spot_check(spot[1], spot[2])
            # ---- which box this is (ground truth: never shown to the agent)
            ph = self.photo_sets.get(c.photos, self.photos) if c.photos else self.photos
            has_edge = "edge" in ph.verdict
            damaged = bool(pr.random() < p.defect_rate)
            k = ph.pick(pr, damaged)
            v_gw, v_cl = int(ph.verdict["gateway"][k]), int(ph.verdict["cloud"][k])
            if has_edge:
                v_ed = int(ph.verdict["edge"][k])
            else:
                v_ed = int(pr.random() < (self.recall[EDGE] if damaged else self.frr[EDGE]))
            level = int(ph.conf[k])                # how sure the gateway's model is about this photo (1..4)
            # ---- cloud-call budget
            if c.budget_per_min > 0:
                cap = max(1.0, c.budget_burst)
                tokens = cap if tokens is None else min(cap, tokens + c.budget_per_min / 60.0 * (t_dec - t_tok))
                t_tok = t_dec
                can_cloud, tok_frac, calls = tokens >= 1.0, tokens / cap, tokens
            else:
                tokens, can_cloud, tok_frac, calls = None, True, 1.0, 99.0
            # ---- observation
            eq = gw_done.count_after(t_dec)       # jobs not finished yet at the decision time
            cq = cloud_done.count_after(t_dec)
            budget_true = deadline - t_dec
            obs = Obs(rtt_ms=rtt_ewma * 1000.0,
                      budget_s=max(0.0, budget_true * (1 + r.normal(0, 0.03))),
                      gw_q=eq, cloud_q=cq,
                      cpu=float(np.clip(c.cpu + r.normal(0, 0.04), 0, 1)),
                      cloud_ms=cloud_ewma * 1000.0, gw_ms=gw_ewma * 1000.0, edge_ms=esp_ewma * 1000.0,
                      tokens=tok_frac, can_cloud=can_cloud, calls_left=calls)
            box_id = start_box_id + i
            m_c = m_g = m_e = None                # times measured on this box (cloud trip, gateway model, ESP32)
            c_ok, asked, fell_back, conf_used = False, False, False, 0

            if policy.gateway_first and (eq + 1) * gw_ewma < self.gw_fit_limit * max(1e-3, budget_true):
                # ---- second opinion: the gateway's model checks the box first
                g_start = max(t_dec, gw_free)
                m_g = _lognorm(r, p.gw_med_s, p.gw_sigma) * c.gw_factor
                g_done = g_start + m_g
                gw_free = g_done
                gw_done.push(g_done)
                energy = p.p_cpu_w * m_g
                left = deadline - g_done
                conf_used = level
                obs = replace(obs, budget_s=max(0.0, left * (1 + r.normal(0, 0.03))), conf=level,
                              cloud_q=cloud_done.count_after(g_done),
                              oracle=1 if (v_gw != int(damaged) and v_cl == int(damaged)) else -1)
                a = policy.decide(box_id, obs)
                if a != CLOUD or not can_cloud or left <= 0:
                    a = GATEWAY                                   # keep the gateway's verdict
                if left < 0:                                      # the gateway's model was late after all
                    on_time, answered, done = False, -1, deadline
                elif a == GATEWAY:
                    on_time, answered, done = True, GATEWAY, g_done
                else:                                             # ask the cloud; fall back if it is late
                    asked = True
                    if tokens is not None:
                        tokens -= 1.0
                    c_done, e2 = self._cloud_trip(c, g_done, st)
                    cloud_done.push(c_done)
                    energy += e2
                    m_c, c_ok = min(c_done, deadline) - g_done, c_done <= deadline
                    on_time = True
                    if c_ok:
                        answered, done = CLOUD, c_done
                    else:
                        answered, done, fell_back = GATEWAY, deadline, True
            else:
                # ---- pick one place (old way; also when the gateway is too slow to go first)
                a = policy.decide(box_id, obs)
                if a == CLOUD and not can_cloud:
                    a = GATEWAY
                if a == EDGE:
                    start = max(t_dec + p.thumb_s * c.gw_factor, esp_free)
                    m_e = p.esp_link_s + _lognorm(r, p.esp_med_s, p.esp_sigma)
                    done = start + m_e
                    esp_free = done
                    energy = p.p_esp_w * m_e
                elif a == GATEWAY:
                    start = max(t_dec, gw_free)
                    m_g = _lognorm(r, p.gw_med_s, p.gw_sigma) * c.gw_factor
                    done = start + m_g
                    gw_free = done
                    gw_done.push(done)
                    energy = p.p_cpu_w * m_g
                else:
                    if tokens is not None:
                        tokens -= 1.0
                    done, energy = self._cloud_trip(c, t_dec, st)
                    cloud_done.push(done)
                    m_c, c_ok = min(done, deadline) - t_dec, done <= deadline
                on_time = done <= deadline
                answered = a if on_time else -1

            latency = (min(done, deadline) - t_dec)
            rew_agent = rm(answered if on_time else a, on_time, energy, conf_used)   # what the agent learns from
            t_avail = min(done, deadline)
            spot = None
            if asked:                             # (answer in time?, confidence level, saving if spot-checked)
                checked = self.spot_rate > 0 and c_ok and sr.random() < self.spot_rate
                spot = (c_ok, conf_used, (verdict_cost(v_gw, damaged, rm.c_esc, rm.c_fr)
                                          - verdict_cost(v_cl, damaged, rm.c_esc, rm.c_fr)) if checked else None)
            heapq.heappush(feedback, (t_avail, box_id, a, on_time, latency, rew_agent, m_c, c_ok, m_g, m_e, spot))

            # ---- what really happened to the box (only for scoring)
            if on_time:
                verdict = v_gw if answered == GATEWAY else (v_cl if answered == CLOUD else v_ed)
                escaped, fr = damaged and verdict == 0, (not damaged) and verdict == 1
                correct = verdict == int(damaged)
                cost = rm.c_esc * escaped + rm.c_fr * fr
            else:
                correct, escaped, fr = False, damaged, False
                cost = rm.c_unchk + (rm.c_esc if damaged else 0.0)
            rew = -cost - rm.w_e * energy

            log.action.append(a); log.on_time.append(on_time); log.latency.append(latency)
            log.reward.append(rew); log.energy.append(energy); log.label.append(int(damaged))
            log.correct.append(correct); log.escaped.append(escaped); log.false_reject.append(fr)
            log.rtt_obs.append(obs.rtt_ms); log.budget.append(obs.budget_s)
            log.asked.append(asked); log.fallback.append(fell_back); log.answered.append(answered)
            log.conf.append(conf_used); log.reward_agent.append(rew_agent)
            if log_conditions:
                log.cond.append(c.name)

        # flush remaining feedback
        while feedback:
            _, bid, a_f, ot_f, lat_f, rew_f = heapq.heappop(feedback)[:6]
            policy.outcome(bid, a_f, ot_f, lat_f, rew_f)
        return log


# ----------------------------------------------------------------------------
# Domain randomisation for pre-training
# ----------------------------------------------------------------------------
def sample_condition(rng: np.random.Generator, hidden_fault_prob: float = 0.08) -> Condition:
    u = rng.random()
    if u < 0.40:   # good Wi-Fi
        net = dict(rtt_s=rng.uniform(0.015, 0.050), jitter_s=rng.uniform(0, 0.008),
                   loss=rng.uniform(0, 0.002), bw_mbps=rng.uniform(5, 20), name="rand-good")
    elif u < 0.70:  # busy Wi-Fi
        net = dict(rtt_s=rng.uniform(0.06, 0.15), jitter_s=rng.uniform(0.01, 0.06),
                   loss=rng.uniform(0, 0.02), bw_mbps=rng.uniform(2, 8), name="rand-busy")
    elif u < 0.90:  # bad Wi-Fi
        net = dict(rtt_s=rng.uniform(0.15, 0.40), jitter_s=rng.uniform(0.03, 0.15),
                   loss=rng.uniform(0.01, 0.05), bw_mbps=rng.uniform(0.5, 3), name="rand-bad")
    elif u < 0.97:  # terrible
        net = dict(rtt_s=rng.uniform(0.4, 1.0), jitter_s=rng.uniform(0.05, 0.3),
                   loss=rng.uniform(0.05, 0.3), bw_mbps=rng.uniform(0.1, 1.0), name="rand-terrible")
    else:
        net = dict(rtt_s=0.01, jitter_s=0.0, loss=0.0, bw_mbps=20, link_down=True, name="rand-down")
    c = Condition(**net)
    c.belt_mps = rng.uniform(0.12, 0.30)
    c.gap_s = rng.uniform(0.5, 2.0)
    u = rng.random()
    if u < 0.12:                    # gateway overloaded: its own model is too slow for the belt
        c.gw_factor, c.cpu = rng.uniform(25.0, 90.0), rng.uniform(0.95, 1.0)
        c.name += "+gw-overload"
    elif u < 0.30:                  # gateway under heavy load
        c.gw_factor, c.cpu = rng.uniform(2.5, 8.0), rng.uniform(0.85, 1.0)
    elif rng.random() < 0.3:
        c.gw_factor, c.cpu = rng.uniform(1.2, 2.5), rng.uniform(0.5, 0.85)
    else:
        c.gw_factor, c.cpu = rng.uniform(1.0, 1.2), rng.uniform(0.05, 0.5)
    # cloud-call budget: sometimes none, otherwise anything from very tight to generous
    if rng.random() >= 0.25:
        c.budget_per_min = float(rng.uniform(4, 45))
        c.budget_burst = float(rng.choice([3, 5, 8]))
    # hidden faults (the agent cannot see these directly)
    if rng.random() < hidden_fault_prob:
        c.server_factor = rng.uniform(8, 40)
        c.name += "+server-slow"
    if rng.random() < hidden_fault_prob:
        c.bw_mbps = min(c.bw_mbps, rng.uniform(0.15, 0.6))
        c.name += "+bw-cap"
    return c


def random_schedule(rng, n_boxes: int, change_prob=0.3, hidden_fault_prob=0.08):
    sched = [(0, sample_condition(rng, hidden_fault_prob))]
    if rng.random() < change_prob:
        sched.append((int(rng.integers(n_boxes // 4, 3 * n_boxes // 4)), sample_condition(rng, hidden_fault_prob)))
    return sched
