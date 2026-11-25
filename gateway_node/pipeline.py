"""
The box-by-box loop on the gateway.

  IR-1 fires on the ESP32 (box under camera)
    -> take photo -> measure state
    -> SECOND-OPINION policies (Q-learning, confidence rules): the gateway's own model checks the box first
       (about 2 ms). Its verdict is kept. Then the AGENT DECIDES with the model's confidence in hand:
         keep the verdict, or also ask the cloud (if a cloud call is left).
       A cloud answer that arrives before the arm replaces the verdict; a late one changes nothing.
    -> if the gateway is too slow to go first, or with a pick-one policy:
       AGENT DECIDES one place (ESP32, GATEWAY or CLOUD) -> start the check
  answer arrives
    -> if before the box reaches the arm: ON TIME -> remember verdict
  IR-2 fires on the ESP32 (box at the reject arm)
    -> verdict "damaged" -> tell the ESP32 to push the box off
    -> no verdict yet    -> MISSED: the box passes unchecked
  outcome known
    -> reward -> agent learns (Q-table update) -> log + send to dashboard

Thread-safety: belt events, the ESP32 answers, the gateway model thread and the cloud threads all
call in here, so shared state is protected by one lock.
"""
from __future__ import annotations

import base64
import csv
import io
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace

from common.cascade import CONF_NAMES, conf_level, margin, verdict_cost
from common.rl import ACTIONS, CLOUD, EDGE, GATEWAY, Obs, load_card
from .budget import CloudBudget


@dataclass
class Box:
    box_id: int
    t_seen: float                 # IR-1 time
    t_decide: float
    deadline: float               # estimated arrival at the arm (absolute time)
    action: int
    obs: Obs
    state: int | None
    truth: str | None = None
    result: dict | None = None
    t_result: float | None = None
    at_arm: bool = False
    finalized: bool = False
    on_time: bool | None = None
    rejected: bool = False
    thumb: str | None = None
    extra: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg, hw, camera, edge, gateway, cloud, netmon, policy, reward_model, space,
                 log_dir: str, reporter=None, truth_fn=None, scenario_fn=None, agent_saver=None):
        self.cfg, self.hw, self.cam, self.cloud = cfg, hw, camera, cloud
        self.edge, self.gw = edge, gateway          # ESP32 tiny model / gateway small model
        self.net, self.policy, self.reward, self.space = netmon, policy, reward_model, space
        self.reporter, self.truth_fn = reporter, truth_fn
        self.scenario_fn = scenario_fn or (lambda: "")
        self.agent_saver = agent_saver
        c = cfg["conveyor"]
        self.cam_to_arm = float(c["cam_to_arm_m"])
        self.guard = float(c["guard_s"])
        self.min_gap = float(c["min_box_gap_s"])
        # no IR sensor at the reject arm (conveyor.arm_sensor: false): the arm is fired by TIME instead,
        # camera-to-arm distance / belt speed after the box was seen under the camera
        self.arm_by_time = not bool(getattr(hw, "arm_sensor", True))
        self.thr = float(cfg["models"]["damaged_threshold"])
        self.thr_by = {"EDGE": self.thr, "GATEWAY": self.thr, "CLOUD": self.thr}   # per-model cut-offs from the model card
        self.conf_edges = None              # borders of the confidence levels (from the validation photos)
        try:
            from common.config import project_path
            card = load_card(project_path(cfg["models"]["card"]))
            for role, key in (("edge", "EDGE"), ("gateway", "GATEWAY"), ("cloud", "CLOUD")):
                if role in card and "threshold" in card[role]:
                    self.thr_by[key] = float(card[role]["threshold"])
            ed = (card.get("cascade") or {}).get("edges")
            if ed and len(ed) == 3:
                self.conf_edges = [float(x) for x in ed]
        except Exception:
            pass
        b = cfg.get("cloud_budget") or {}
        self.budget = CloudBudget(b.get("per_min", 0), b.get("burst", 5))
        # spot checks: when the true state of a box is known (test run with a known box order, or a person
        # looked), the agent measures on the line itself what a cloud answer was worth
        self.spot_checks = bool(cfg["agent"].get("spot_checks", True))
        self.e = cfg["energy"]
        self.save_every = int(cfg["agent"]["save_every"])
        self.lock = threading.RLock()
        self.boxes: dict[int, Box] = {}
        self.fifo: deque[int] = deque()     # boxes between camera and arm, in belt order
        self._imgs: dict = {}               # photos of boxes sent to the ESP32 or the cloud (for fallback)
        self.next_id = 0
        self.last_seen = 0.0
        self.transit = None                 # measured camera->arm time (EWMA)
        self.stats = {"n": 0, "edge": 0, "gateway": 0, "cloud": 0, "missed": 0, "rejected": 0, "correct": 0,
                      "known": 0, "fallback": 0,
                      "first": 0,          # boxes the gateway's model checked first (second-opinion way)
                      "asked": 0,          # ... of those: cloud asked for a second opinion
                      "cloud_late": 0,     # ... cloud answer came too late: the gateway's verdict was used
                      "changed": 0}        # ... cloud answer was different from the gateway's verdict
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "boxes.csv")
        self._csv = open(self.log_path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._csv)
        if os.path.getsize(self.log_path) == 0:
            self._w.writerow(["box_id", "time", "policy", "scenario", "action", "state", "rtt_ms", "cloud_ms",
                              "budget_s", "slack_ms", "gw_q", "cloud_q", "cpu", "latency_ms", "on_time",
                              "p_damaged", "verdict", "rejected", "reward", "energy_mJ", "truth", "error",
                              "server_ms", "power_w", "gw_ms", "edge_ms", "answered_by", "esp_ms",
                              "conf", "asked", "cloud_late", "gw_p_damaged", "calls_left"])
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    # ------------------------------------------------------------------
    def transit_estimate(self) -> float:
        if self.transit is not None and not self.arm_by_time:
            return self.transit
        return self.cam_to_arm / max(0.02, self.hw.belt_speed_mps())

    def _expect_at_arm(self, t0: float):
        """Arm fired by time: this box 'reaches the arm' one transit time after the camera saw it."""
        if not self.arm_by_time:
            return
        wait = max(0.0, t0 + self.transit_estimate() - time.perf_counter())
        tm = threading.Timer(wait, self.on_box_at_arm)
        tm.daemon = True
        tm.start()

    def on_box_at_camera(self):
        t0 = time.perf_counter()
        with self.lock:
            if t0 - self.last_seen < self.min_gap:
                return                                  # IR glitch / same box
            self.last_seen = t0
            box_id = self.next_id
            self.next_id += 1
        self._expect_at_arm(t0)
        try:
            img = self.cam.capture_box(t0)          # first frame taken after IR-1 fired
        except Exception as e:
            print(f"[pipe] camera error: {e}")
            self._add_unchecked(box_id, t0, f"camera: {e}")
            return
        truth = self.truth_fn(box_id) if self.truth_fn else None
        t_dec = time.perf_counter()
        deadline = t0 + self.transit_estimate() - self.guard
        n = self.net.snapshot()
        bud = self.budget.snapshot()
        obs = Obs(rtt_ms=n["rtt_ms"], budget_s=max(0.0, deadline - t_dec), gw_q=self.gw.queue_len,
                  cloud_q=self.cloud.inflight, cpu=n["cpu"], cloud_ms=n["cloud_ms"],
                  gw_ms=self.gw.ewma_ms, edge_ms=self.edge.ewma_ms,
                  tokens=bud["share"], can_cloud=bud["can"], calls_left=bud["calls_left"])
        if getattr(self.policy, "gateway_first", False) and self.space.gw_load(obs) == 0:
            # second opinion: the gateway's own model looks first (only when it clearly fits in the time left);
            # the decision is made when it has answered
            with self.lock:
                box = Box(box_id, t0, t_dec, deadline, GATEWAY, obs, None, truth=truth)
                box.extra["first"] = True
                self.boxes[box_id] = box
                self.fifo.append(box_id)
                self._imgs[box_id] = img              # kept in case the cloud is asked
            self.gw.submit(box_id, img, self._on_gateway_first)
            try:
                box.thumb = self._thumb(img)
            except Exception:
                pass
            return
        s = self.space.index(obs)
        with self.lock:
            a = self.policy.decide(box_id, obs)
            if a == EDGE and not self.edge.ready:     # ESP32 unplugged: the gateway does it (and the agent
                a = GATEWAY                           # learns from what was really done)
            if a == CLOUD and not self.budget.take():  # no cloud call left: the gateway does it
                a = GATEWAY
            box = Box(box_id, t0, t_dec, deadline, a, obs, s, truth=truth)
            self.boxes[box_id] = box
            self.fifo.append(box_id)
            if a != GATEWAY:
                self._imgs[box_id] = img              # kept for a possible fallback to the gateway
        if a == GATEWAY:
            self.gw.submit(box_id, img, self.on_result)
        elif a == EDGE:
            self.edge.submit(box_id, img, self.on_result)
        else:
            self.cloud.submit(box_id, img, max(0.05, deadline - time.perf_counter()), self.on_result)
        # the small dashboard picture is made AFTER the check has started: it is not on the deadline's clock
        try:
            box.thumb = self._thumb(img)
        except Exception:
            pass

    def _on_gateway_first(self, box_id: int, res: dict):
        """The gateway's own model has checked the box. Keep its verdict, then let the policy decide
        whether the cloud is asked for a second opinion."""
        now = time.perf_counter()
        ask_img = None
        with self.lock:
            box = self.boxes.get(box_id)
            img = self._imgs.pop(box_id, None)
            if box is None or box.finalized:           # the box already passed the arm: it was counted as missed
                return
            if not res.get("ok"):
                box.extra["error"] = res.get("error")
                return                                 # no verdict: the arm will count it as missed
            thr = self.thr_by.get("GATEWAY", self.thr)
            p = float(res.get("p_damaged", 0.0))
            conf = conf_level(float(margin(p, thr)), self.conf_edges) if self.conf_edges else 1
            left = box.deadline - now
            n = self.net.snapshot()
            bud = self.budget.snapshot()
            obs = replace(box.obs, budget_s=max(0.0, left), conf=conf, cloud_q=self.cloud.inflight,
                          rtt_ms=n["rtt_ms"], cloud_ms=n["cloud_ms"], gw_ms=self.gw.ewma_ms,
                          tokens=bud["share"], can_cloud=bud["can"], calls_left=bud["calls_left"])
            a = self.policy.decide(box_id, obs)
            if a != CLOUD or img is None or left <= 0.02 or not self.budget.take():
                a = GATEWAY                            # keep the gateway's verdict
            box.action, box.obs, box.state = a, obs, self.space.index(obs)
            box.extra.update(conf=conf, gw_p=p, gw_compute_s=res.get("compute_s", 0.0), t_first=now)
            box.result, box.t_result = res, now        # from now on the box HAS a verdict (the cloud may replace it)
            self.stats["first"] += 1
            if a == CLOUD:
                box.extra.update(asked=True, t_ask=now)
                ask_img = img
            elif not box.at_arm:
                self._finalize(box, on_time=True)
        if ask_img is not None:
            self.cloud.submit(box_id, ask_img, max(0.05, box.deadline - time.perf_counter()), self.on_result)

    def _add_unchecked(self, box_id: int, t0: float, why: str):
        """No photo (camera error). The box is still on the belt, so it keeps its place in the FIFO:
        otherwise IR-2 would pair every later box with the wrong verdict and push the wrong boxes.
        It passes the arm unchecked and counts as missed; the agent does not learn from it."""
        truth = self.truth_fn(box_id) if self.truth_fn else None
        obs = Obs(rtt_ms=0.0, budget_s=0.0, gw_q=0, cloud_q=0, cpu=0.0)
        with self.lock:
            box = Box(box_id, t0, t0, t0 + self.transit_estimate() - self.guard, GATEWAY, obs, None, truth=truth)
            box.extra.update(error=why, no_photo=True)
            self.boxes[box_id] = box
            self.fifo.append(box_id)

    def on_result(self, box_id: int, res: dict):
        now = time.perf_counter()
        with self.lock:
            box = self.boxes.get(box_id)
            if box is None:
                return
            where = res.get("where")
            img = self._imgs.pop(box_id, None) if where in ("CLOUD", "EDGE") else None
            if where == "CLOUD":
                t_sent = box.extra.get("t_ask", box.t_decide)
                self.net.record_cloud(now - t_sent if res.get("ok") else max(now - t_sent, self.net.cprobe_timeout))
            if box.finalized:          # too late - box already passed the arm
                return
            if box.extra.get("asked"):                 # second opinion: the gateway's verdict is already in place
                if res.get("ok"):
                    gw_verdict = self._verdict(box.result)
                    box.extra["gw_result"] = box.result
                    box.result, box.t_result = res, now
                    if self._verdict(res) != gw_verdict:
                        box.extra["changed"] = True
                else:
                    box.extra.update(error=res.get("error"), cloud_late=True)
                if not box.at_arm:
                    self._finalize(box, on_time=True)
                return
            if not res.get("ok"):
                box.extra["error"] = res.get("error")
                # cloud or ESP32 failed FAST (connection refused, cable pulled) and there is still time
                # -> the gateway checks the box itself
                if (where in ("CLOUD", "EDGE") and img is not None and "timeout" not in str(res.get("error"))
                        and box.deadline - now > 0.06):
                    box.extra["fallback"] = True
                    self.gw.submit(box_id, img, self.on_result)
                return                 # otherwise wait for the arm: it will be counted as missed
            box.result, box.t_result = res, now
            if not box.at_arm:
                self._finalize(box, on_time=True)

    def on_box_at_arm(self):
        now = time.perf_counter()
        with self.lock:
            if not self.fifo:
                return
            box_id = self.fifo.popleft()
            box = self.boxes.get(box_id)
            if box is None:
                return
            box.at_arm = True
            if not self.arm_by_time:                   # measured travel time (only a real sensor can measure it)
                t = now - box.t_seen
                self.transit = t if self.transit is None else 0.7 * self.transit + 0.3 * t
            if box.result is not None and box.t_result is not None and box.t_result <= now:
                if box.extra.get("asked") and box.result.get("where") != "CLOUD" and not box.finalized:
                    box.extra["cloud_late"] = True     # cloud not back yet: the gateway's verdict is used
                verdict = self._verdict(box.result)
                if verdict == "damaged":
                    self.hw.push()
                    box.rejected = True
                self.hw.show(verdict)
                if not box.finalized:
                    self._finalize(box, on_time=True)
            else:
                self.hw.show("none")
                self._finalize(box, on_time=False)
            self._log(box)

    def _watch(self):
        """If IR-2 misses a box, finish it anyway 1.5 s after it should have arrived."""
        while True:
            time.sleep(0.5)
            now = time.perf_counter()
            with self.lock:
                while self.fifo:
                    b = self.boxes.get(self.fifo[0])
                    if b is None or now > b.deadline + self.guard + 1.5:
                        self.fifo.popleft()
                        if b is not None and not b.at_arm:
                            b.at_arm = True
                            b.extra["error"] = (b.extra.get("error") or "") + " arm-sensor-missed"
                            if not b.finalized:
                                self._finalize(b, on_time=b.result is not None)
                            self._log(b)
                    else:
                        break
                for k in [k for k, b in self.boxes.items() if b.at_arm and b.finalized and now - b.t_seen > 30]:
                    del self.boxes[k]

    # ------------------------------------------------------------------
    def _energy_j(self, box: Box) -> float:
        r = box.result or {}
        if r.get("where") == "EDGE":
            return float(self.e.get("p_esp32_w", 0.5)) * r.get("compute_s", 0.0)
        first = self.e["p_cpu_active_w"] * box.extra.get("gw_compute_s", 0.0) if box.extra.get("first") else 0.0
        if box.extra.get("first") and r.get("where") != "CLOUD":
            return first
        if not box.extra.get("first") and (box.action == GATEWAY or r.get("where") == "GATEWAY"):
            return self.e["p_cpu_active_w"] * r.get("compute_s", 0.0)
        up = 0.0
        if r.get("jpeg_bytes"):
            lat = (box.t_result or time.perf_counter()) - box.extra.get("t_ask", box.t_decide)
            up = max(0.0, lat - (r.get("server_ms", 0) / 1000.0) - box.obs.rtt_ms / 1000.0)
        return first + self.e["p_cpu_active_w"] * r.get("jpeg_s", 0.0) + self.e["p_wifi_tx_w"] * up

    def _finalize(self, box: Box, on_time: bool):
        box.finalized = True
        box.on_time = on_time
        if box.extra.get("no_photo"):              # camera error: missed, but no decision to learn from
            box.on_time = False
            box.extra.update(reward=self.reward(GATEWAY, False, 0.0), energy=0.0, latency=0.0)
            self.stats["n"] += 1
            self.stats["missed"] += 1
            return
        energy = self._energy_j(box)
        # quality of the model that actually answered (cloud / ESP32 failed fast -> gateway answered = fallback;
        # second opinion asked but the cloud was late -> the gateway's verdict was used)
        if box.extra.get("first"):
            if box.state is None:                  # the gateway's own model never answered: no decision was made
                box.on_time = False
                box.extra.update(reward=self.reward(GATEWAY, False, 0.0), energy=0.0,
                                 latency=time.perf_counter() - box.t_decide)
                self.stats["n"] += 1
                self.stats["gateway"] += 1
                self.stats["missed"] += 1
                return
            answered_by = CLOUD if (box.result or {}).get("where") == "CLOUD" else GATEWAY
        else:
            answered_by = GATEWAY if box.extra.get("fallback") else box.action
        conf = int(box.extra.get("conf", 0))
        rew = self.reward(answered_by, on_time, energy, conf)
        lat = ((box.t_result if on_time and box.t_result else time.perf_counter()) - box.t_decide)
        box.extra.update(reward=rew, energy=energy, latency=lat)
        self.policy.outcome(box.box_id, box.action, on_time, lat, rew)
        if box.extra.get("asked"):
            self.policy.cloud_call(answered_by == CLOUD)
            gw_res = box.extra.get("gw_result")
            if (self.spot_checks and answered_by == CLOUD and gw_res is not None and box.truth in ("good", "damaged")):
                dmg = box.truth == "damaged"
                saving = (verdict_cost(self._verdict(gw_res) == "damaged", dmg, self.reward.c_esc, self.reward.c_fr)
                          - verdict_cost(self._verdict(box.result) == "damaged", dmg, self.reward.c_esc, self.reward.c_fr))
                self.policy.spot_check(conf, saving)
                box.extra["spot_saving"] = saving
        st = self.stats
        st["n"] += 1
        st[("edge", "gateway", "cloud")[box.action]] += 1
        if box.extra.get("fallback"):
            st["fallback"] += 1
        if box.extra.get("asked"):
            st["asked"] += 1
            if answered_by != CLOUD:
                box.extra["cloud_late"] = True
                st["cloud_late"] += 1
            elif box.extra.get("changed"):
                st["changed"] += 1
        if not on_time:
            st["missed"] += 1
        if self.agent_saver and st["n"] % self.save_every == 0:
            self.agent_saver()

    def _log(self, box: Box):
        r = box.result or {}
        verdict = self._verdict(r) if box.on_time and r else "unchecked"
        if box.rejected:
            self.stats["rejected"] += 1
        if box.truth in ("good", "damaged"):
            self.stats["known"] += 1
            if verdict == box.truth:
                self.stats["correct"] += 1
        o = box.obs
        action = "NONE" if box.extra.get("no_photo") else ACTIONS[box.action]
        row = [box.box_id, time.strftime("%H:%M:%S"), getattr(self.policy, "name", "?"), self.scenario_fn(),
               action, box.state, round(o.rtt_ms, 1), round(o.cloud_ms, 1), round(o.budget_s, 3),
               round(self.space.slack_ms(o), 1), o.gw_q, o.cloud_q, round(o.cpu, 2),
               round(box.extra.get("latency", 0) * 1000, 1), int(bool(box.on_time)),
               round(r.get("p_damaged", -1), 4), verdict, int(box.rejected), round(box.extra.get("reward", 0), 3),
               round(box.extra.get("energy", 0) * 1000, 2), box.truth or "", box.extra.get("error", ""),
               r.get("server_ms", ""), self.hw.power_w() or "", round(o.gw_ms, 1), round(o.edge_ms, 1),
               r.get("where", ""), r.get("esp_ms", ""),
               box.extra.get("conf", ""), int(bool(box.extra.get("asked"))), int(bool(box.extra.get("cloud_late"))),
               round(box.extra["gw_p"], 4) if "gw_p" in box.extra else "", round(min(o.calls_left, 99.0), 2)]
        self._w.writerow(row)
        self._csv.flush()
        if self.reporter:
            self.reporter.event({
                "box_id": box.box_id, "action": action, "on_time": bool(box.on_time),
                "latency_ms": round(box.extra.get("latency", 0) * 1000, 1), "verdict": verdict,
                "p_damaged": round(r.get("p_damaged", -1), 3), "rejected": box.rejected, "truth": box.truth,
                "rtt_ms": round(o.rtt_ms, 1), "cloud_ms": round(o.cloud_ms, 1), "budget_ms": round(o.budget_s * 1000),
                "slack_ms": round(self.space.slack_ms(o)), "cpu": round(o.cpu, 2), "reward": round(box.extra.get("reward", 0), 2),
                "state": box.state, "heatmap": r.get("heatmap"), "thumb": box.thumb, "policy": getattr(self.policy, "name", "?"),
                "error": box.extra.get("error"), "t": time.time(),
                "gw_ms": round(o.gw_ms, 1), "edge_ms": round(o.edge_ms, 1), "answered_by": r.get("where"),
                "esp_ms": r.get("esp_ms"), "emulated": bool(r.get("emulated")), "fallback": bool(box.extra.get("fallback")),
                # second opinion
                "first": bool(box.extra.get("first")), "conf": box.extra.get("conf"),
                "conf_name": CONF_NAMES[box.extra["conf"]] if box.extra.get("conf") else None,
                "asked": bool(box.extra.get("asked")), "cloud_late": bool(box.extra.get("cloud_late")),
                "changed": bool(box.extra.get("changed")),
                "gw_p": round(box.extra["gw_p"], 3) if "gw_p" in box.extra else None,
                "gw_verdict": (self._verdict({"where": "GATEWAY", "p_damaged": box.extra["gw_p"]})
                               if "gw_p" in box.extra else None),
                "calls_left": round(min(o.calls_left, 99.0), 1), "can_cloud": bool(o.can_cloud),
            })

    def _verdict(self, res: dict) -> str:
        thr = self.thr_by.get(res.get("where", "GATEWAY"), self.thr)
        return "damaged" if res.get("p_damaged", 0) >= thr else "good"

    @staticmethod
    def _thumb(img) -> str:
        buf = io.BytesIO()
        img.resize((96, 96)).save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()

    def close(self):
        try:
            self._csv.close()
        except Exception:
            pass
