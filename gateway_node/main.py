"""
Smart Conveyor QC - the GATEWAY program (runs on the laptop next to the belt).

It owns the webcam, talks to the ESP32 (USB cable) and to the cloud server (Internet), and its
Q-learning agent decides for every box where the photo is checked: ESP32, gateway or cloud.

    # real belt: ESP32 plugged in by USB, webcam on this laptop, cloud server on AWS
    python -m gateway_node.main --server http://<ec2-public-ip>:8000

    # no hardware at all (pretend belt, pretend ESP32, emulated Wi-Fi)
    python -m gateway_node.main --mock

    # pretend belt, but the tiny model really runs on a plugged-in ESP32
    python -m gateway_node.main --mock --esp32 auto

Useful options
    --policy qlearning|always_edge|always_gateway|always_cloud|threshold|deadline|adaptive
    --esp32 auto|mock|COM5   which ESP32 to use (default: auto on the real belt, mock with --mock)
    --boxes 300          stop after 300 boxes
    --tag myrun          log folder name (logs/<tag>/boxes.csv)
    --sequence experiments/sequences/seq_A.csv   true labels of real boxes, in order
    --scenario busy      (mock) starting network: good busy bad bwcap lossy stalls down
    --resume             continue from the online-learned Q-table instead of the pre-trained one
    --no-learn           freeze the Q-table (no online learning)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from common.config import load_config, project_path
from common.net import api_key, device_id
from common.policies import make_policy
from common.rl import QAgent, RewardModel, StateSpace


def make_live_gain(cfg, reward):
    """The spot-check tally (what a cloud answer is worth, measured on this line). None when switched off
    or when the model card has no confidence levels yet."""
    from common.cascade import N_CONF, LiveGain
    a = cfg["agent"]
    if not a.get("spot_checks", True) or not reward.level.get("informative"):
        return None
    return LiveGain([0.0] + [reward.gain(c) for c in range(1, N_CONF)], float(a.get("spot_prior_n", 60)),
                    float(a.get("spot_half_life", 300)))


def build_policy(name, cfg, space, args, live=None):
    with open(project_path(cfg["agent"]["rule_params"])) as f:
        rp = json.load(f)
    if name != "qlearning":
        return make_policy(name, cfg, rule_params=rp), None
    online_path = project_path(cfg["agent"]["q_table"]).replace(".npz", "_online.npz")
    path = online_path if (args.resume and os.path.exists(online_path)) else project_path(cfg["agent"]["q_table"])
    agent = QAgent.load(path)
    if agent.q.shape != (space.n_states, 3) and path == online_path:      # learned before the ESP32 was added
        print("[main] the saved online Q-table is from the 2-node version - starting from the pre-trained one")
        path = project_path(cfg["agent"]["q_table"])
        agent = QAgent.load(path)
    if agent.q.shape != (space.n_states, 3):
        raise SystemExit("The Q-table does not fit the 3-node setup (ESP32 / gateway / cloud). "
                         "Train it again: python -m sim.train_q   (or double-click TRAIN_ON_DOWNLOADED_PHOTOS.bat)")
    agent.prefer = 1                       # equal values -> the gateway (local, fast, accurate)
    cfg = json.loads(json.dumps(cfg))
    if args.no_learn:
        cfg["agent"]["learn_online"] = False
    pol = make_policy("qlearning", cfg, agent=agent)
    pol.live = live
    if live is not None and path == online_path:
        live.restore((agent.meta or {}).get("spot"))          # carry on with the tally of the last run
    print(f"[main] Q-table loaded from {os.path.relpath(path)}  (online learning: {pol.learn}, "
          f"spot checks: {'on' if live is not None else 'off'})")
    return pol, (agent, online_path)


class NoEdge:
    """Stands in when there is no tiny model file yet: the ESP32 is never chosen."""
    ready, emulated, ewma_ms, queue_len, failures, last_esp_ms, link = False, False, 0.0, 0, 0, 0.0, None

    def submit(self, box_id, img, callback):
        callback(box_id, {"ok": False, "error": "esp32: no model", "where": "EDGE"})

    def measure(self, n=0):
        pass


class Stress:
    """Busy-loop processes to load the gateway CPU (the 'Overload the gateway' button on the real belt)."""

    def __init__(self):
        self.procs = []

    def set(self, on: bool):
        if on and not self.procs:
            n = os.cpu_count() or 4
            self.procs = [subprocess.Popen([sys.executable, "-c", "while True: pass"]) for _ in range(n)]
        elif not on and self.procs:
            for p in self.procs:
                p.terminate()
            self.procs = []


def main():
    ap = argparse.ArgumentParser(description="Smart Conveyor QC - gateway node")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true", help="no hardware: pretend conveyor, camera and Wi-Fi")
    ap.add_argument("--esp32", default=None, help="auto | mock | a port such as COM5 (default: mock with --mock, else auto)")
    ap.add_argument("--server", default=None, help="cloud server URL, e.g. http://<ec2-public-ip>:8000")
    ap.add_argument("--api-key", default=None, help="server API key (default: env CONVEYOR_API_KEY)")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--boxes", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sequence", default=None)
    ap.add_argument("--belt", type=float, default=None, help="belt PWM duty 0..1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    server = (args.server or os.environ.get("CONVEYOR_SERVER")
              or ("http://127.0.0.1:%d" % cfg["server"]["port"] if args.mock else cfg["server"]["cloud_url"]))
    if "<" in server:
        raise SystemExit(f"set the cloud server address first: --server http://<ip>:8000, env CONVEYOR_SERVER, "
                         f"or server.cloud_url in config/system.yaml (now {server!r})")
    key, device = api_key(cfg, args.api_key), device_id(cfg)
    policy_name = args.policy or cfg["agent"]["policy"]
    tag = args.tag or time.strftime("run_%Y%m%d_%H%M%S") + f"_{policy_name}" + ("_mock" if args.mock else "")
    log_dir = project_path(os.path.join("logs", tag))
    print(f"[main] server={server}  policy={policy_name}  device={device}  "
          f"api-key={'set' if key else 'none'}  logs={os.path.relpath(log_dir)}")

    # ---------------- ESP32 edge node ----------------
    from common import tiny
    from .esp32_link import EdgeTiny, Esp32Error, Esp32Hardware, connect
    tiny_path = project_path(cfg["models"].get("edge_tiny", "models/edge_tiny.bin"))
    tiny_model = tiny.TinyModel.load(tiny_path) if os.path.exists(tiny_path) else None
    if tiny_model is None:
        print(f"[main] WARNING: no tiny model for the ESP32 yet ({os.path.relpath(tiny_path)}). "
              f"Train it: python -m vision.train_tiny --data <photos>. Until then the ESP32 only runs the belt.")
    want = (args.esp32 or ("mock" if args.mock else "auto")).strip()
    link = None
    if want.lower() not in ("mock", "off", "none"):
        try:
            link = connect(cfg, tiny_model.blob if tiny_model else None, None if want.lower() == "auto" else want)
        except Esp32Error as e:
            if not args.mock:
                raise SystemExit(f"[main] ESP32 not reachable: {e}\n"
                                 f"       No board yet? Run the pretend version: python -m gateway_node.main --mock")
            print(f"[main] ESP32 not reachable ({e}) - using the pretend ESP32")
    esp_mode = "real" if link is not None else "emulated"
    if tiny_model is None:
        edge = NoEdge()
    else:
        edge = EdgeTiny(tiny_model, link, timeout_s=float(cfg.get("esp32", {}).get("answer_timeout_s", 0.25)),
                        baud=int(cfg.get("esp32", {}).get("baud", 460800)))
        edge.measure()
        print(f"[main] ESP32 ({esp_mode}): {tiny_model.size}x{tiny_model.size} photo, {tiny_model.params} numbers, "
              f"about {edge.ewma_ms:.0f} ms per box incl. the USB cable")

    # ---------------- belt / camera ----------------
    netemu = None
    if args.mock:
        from .hardware import MockHardware
        from .netemu import NetEmu
        # CONVEYOR_MOCK_FEED: another folder of good/ damaged/ photos for the pretend belt (set by tools/go.py)
        hw = MockHardware(cfg, os.environ.get("CONVEYOR_MOCK_FEED") or project_path(cfg["mock"]["feed_dir"]),
                          seed=args.seed)
        remote = not any(h in server for h in ("127.0.0.1", "localhost", "[::1]"))
        # pretend belt + REAL remote server (e.g. the EC2 instance): the Internet is real, emulate nothing
        netemu = NetEmu(args.scenario or ("real" if remote else cfg["mock"]["scenario"]), seed=args.seed)
        scenario = {"name": netemu.name}
    else:
        hw = Esp32Hardware(cfg, link)
        scenario = {"name": args.scenario or "as-is"}
    from .camera import Camera
    cam = Camera(cfg, mock_hw=hw if args.mock else None)

    # ---------------- models / network ----------------
    from .cloud_client import CloudClient
    from .gateway_worker import GatewayWorker
    from .netmon import NetMonitor
    gw = GatewayWorker(project_path(cfg["models"]["gateway_onnx"]), threads=cfg["models"]["gateway_threads"],
                       slowdown=cfg["mock"]["gateway_slowdown"] if args.mock else 1.0)
    cloud = CloudClient(server, cfg["camera"]["jpeg_quality"], netemu=netemu, api_key=key, device=device)
    probe_path = project_path("models/probe.jpg")
    with open(probe_path, "rb") as f:
        probe_jpeg = f.read()
    net = NetMonitor(cfg, server, cloud, probe_jpeg, netemu=netemu, api_key=key, device=device)
    net.start()
    if not net.wait_ready(6.0):
        print("[main] WARNING: cloud server not answering yet - boxes will be checked on the gateway")

    # ---------------- agent ----------------
    space = StateSpace(cfg["state_bins"])
    reward = RewardModel.from_card(cfg, project_path(cfg["models"]["card"]))
    rd = reward.describe()
    print("[main] reward per box:", {k: round(v, 2) for k, v in rd.items() if k != "by_confidence"})
    if rd.get("by_confidence"):
        print("[main] a cloud second opinion is worth (cost saved per call): " +
              ", ".join(f"{lv['name']} {lv['cloud'] - lv['gateway']:+.2f}" for lv in rd["by_confidence"]))
    else:
        print("[main] NOTE: no confidence levels in the model card yet (run the photo-score step); "
              "every box counts as 'sure' until then")
    live_gain = make_live_gain(cfg, reward)
    policy, agent_info = build_policy(policy_name, cfg, space, args, live_gain)

    def save_agent():
        if agent_info:
            ag, path = agent_info
            meta = {"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "online": True}
            if live_gain is not None:
                meta["spot"] = live_gain.state()
            ag.save(path, meta)

    # ---------------- ground truth (for accuracy numbers) ----------------
    truth_fn = None
    if args.mock:
        truth_fn = lambda box_id: hw.current[1]
    elif args.sequence or cfg["conveyor"].get("box_order_file"):
        args.sequence = args.sequence or project_path(str(cfg["conveyor"]["box_order_file"]))
        with open(args.sequence) as f:
            seq = [ln.strip().split(",")[-1] for ln in f if ln.strip() and not ln.startswith("#")]
        seq = [s for s in seq if s in ("good", "damaged")]
        truth_fn = lambda box_id: seq[box_id % len(seq)] if seq else None   # boxes are re-used in a loop
        print(f"[main] ground-truth sequence: {len(seq)} boxes")

    from .pipeline import Pipeline
    from .reporter import Reporter
    stress = Stress()
    state = {"gw_stress": False, "policy": policy_name}

    def status(full_q=False):
        n = net.snapshot()
        esp = {"mode": esp_mode if tiny_model is not None else "no model", "port": getattr(link, "port", None),
               "alive": bool(edge.ready), "resets": getattr(link, "resets", 0), "failures": edge.failures,
               "ms": round(edge.ewma_ms, 1), "compute_ms": round(edge.last_esp_ms, 1),
               "img": getattr(tiny_model, "size", None), "params": getattr(tiny_model, "params", None)}
        st = {"t": time.time(), "mock": args.mock, "policy": getattr(pipe.policy, "name", "?"), "esp32": esp,
              "gw_ms": round(gw.ewma_ms, 1), "edge_ms": round(edge.ewma_ms, 1), "nodes": 3,
              "policy_key": state["policy"], "scenario": scenario["name"], "gw_stress": state["gw_stress"],
              "belt_duty": round(hw.duty, 3), "belt_mps": round(hw.belt_speed_mps(), 3),
              "transit_s": round(pipe.transit_estimate(), 3), "rtt_ms": round(n["rtt_ms"], 1),
              "cloud_ms": round(n["cloud_ms"], 1), "cpu": round(n["cpu"], 2), "server_ok": n["server_ok"],
              "gw_q": gw.queue_len, "cloud_q": cloud.inflight, "stats": dict(pipe.stats),
              "dims": list(space.dims), "labels": space.labels(),
              # second opinion: is the gateway's model going first, and how many cloud calls are left right now
              "second_opinion": bool(getattr(pipe.policy, "gateway_first", False)),
              "conf_ready": pipe.conf_edges is not None, "budget": pipe.budget.snapshot(),
              "gain_by_conf": [round(lv["cloud"] - lv["gateway"], 2) for lv in rd.get("by_confidence", [])]}
        if agent_info:
            ag = agent_info[0]
            st["epsilon"] = ag.epsilon if getattr(pipe.policy, "explore", False) else 0.0
            st["learning"] = bool(getattr(pipe.policy, "learn", False))
            st["q_updates"] = int(getattr(pipe.policy, "updates", 0))
            if getattr(pipe.policy, "live", None) is not None:
                # what a cloud answer is worth per confidence level: model card vs measured on this line
                st["gain_live"] = pipe.policy.live.table()
            if full_q and len(space.dims) >= 7:
                # two small pictures of the learned table (the whole table has 10,800 rows: too big to send every
                # few seconds over a slow link)
                with pipe.lock:
                    st["qmaps"] = ag.maps(space, getattr(pipe.policy, "last_state", None))
        return st

    def on_control(ctl: dict):
        sc = ctl.get("scenario")
        if sc and sc != scenario["name"]:
            if netemu is not None:
                try:
                    netemu.set(sc); scenario["name"] = sc
                    print(f"[control] network scenario -> {sc}")
                except ValueError as e:
                    print(f"[control] {e}")
            elif cfg.get("gateway", {}).get("allow_shaping"):
                script = project_path("tools/net_shape.sh")
                r = subprocess.run(["sudo", "-n", "bash", script, sc, cfg["gateway"].get("iface", "wlan0")],
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    scenario["name"] = sc
                    print(f"[control] tc netem scenario -> {sc}")
                else:
                    print(f"[control] net_shape failed: {r.stderr.strip()[:200]}")
        ps = bool(ctl.get("gw_stress", False))
        if ps != state["gw_stress"]:
            state["gw_stress"] = ps
            if args.mock:      # pretend overload: the gateway's own model becomes far too slow for the belt
                gw.extra_s = float(cfg["mock"].get("gateway_overload_ms", 600.0)) / 1000.0 if ps else 0.0
                net.cpu_override = 0.97 if ps else None
            else:
                stress.set(ps)
            print(f"[control] gateway overload -> {ps}")
        cpm = ctl.get("cloud_per_min")
        if cpm is not None and abs(float(cpm) - pipe.budget.per_min) > 1e-6:
            pipe.budget.set(float(cpm))
            print(f"[control] cloud calls per minute -> {'no limit' if float(cpm) <= 0 else cpm}")
        duty = ctl.get("belt_duty")
        if duty is not None and abs(float(duty) - hw.duty) > 1e-3:
            hw.set_belt(float(duty))
            pipe.transit = None            # re-measure the camera->arm time
            print(f"[control] belt duty -> {duty}")
        pk = ctl.get("policy")
        if pk and pk != state["policy"]:
            try:
                if pk == "qlearning" and agent_info:
                    newp = make_policy("qlearning", cfg, agent=agent_info[0])
                    newp.live = live_gain
                else:
                    newp, _ = build_policy(pk, cfg, space, args, live_gain)
                with pipe.lock:
                    pipe.policy = newp
                state["policy"] = pk
                print(f"[control] policy -> {pk}")
            except Exception as e:
                print(f"[control] policy switch failed: {e}")

    pipe = Pipeline(cfg, hw, cam, edge, gw, cloud, net, policy, reward, space, log_dir,
                    truth_fn=truth_fn, scenario_fn=lambda: scenario["name"], agent_saver=save_agent)

    def qtable_bytes():
        """Online-learned Q-table for the cloud backup (copied under the pipeline lock)."""
        if not agent_info:
            return None
        with pipe.lock:
            return agent_info[0].to_bytes({"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "online": True,
                                           "device": device, "boxes": pipe.stats["n"]})

    up_every = float(cfg.get("cloud", {}).get("qtable_backup_every_s", 60))
    pipe.reporter = Reporter(server, status, on_control=on_control, api_key=key, device=device,
                             qtable_fn=qtable_bytes if up_every > 0 else None, qtable_every_s=max(5.0, up_every))
    hw.bind(pipe.on_box_at_camera, pipe.on_box_at_arm)
    with open(os.path.join(log_dir, "run_info.json"), "w") as f:
        json.dump({"policy": policy_name, "mock": args.mock, "esp32": esp_mode, "server": server,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "scenario": scenario["name"], "config": cfg}, f, indent=2, default=str)

    hw.start_belt(args.belt)
    print("[main] belt running - Ctrl+C to stop")
    t_start = time.time()
    last_n = -1
    try:
        while True:
            time.sleep(0.5)
            st = pipe.stats
            if st["n"] != last_n and st["n"] % 10 == 0 and st["n"] > 0:
                last_n = st["n"]
                n = net.snapshot()
                acc = f"  acc {st['correct'] / st['known']:.1%}" if st["known"] else ""
                so = (f"asked {st['asked'] / st['n']:4.0%} (late {st['cloud_late']}, changed {st['changed']}) | "
                      if st["first"] else "")
                print(f"  box {st['n']:5d} | ESP32 {st['edge'] / st['n']:4.0%}  gateway {st['gateway'] / st['n']:4.0%}  "
                      f"cloud {st['cloud'] / st['n']:4.0%} | {so}"
                      f"missed {st['missed'] / st['n']:5.1%} | rejected {st['rejected']:4d}{acc} | "
                      f"ping {n['rtt_ms']:5.0f} ms  cloud {n['cloud_ms']:5.0f} ms  gateway {gw.ewma_ms:4.0f} ms  "
                      f"esp32 {edge.ewma_ms:3.0f} ms | {scenario['name']}",
                      flush=True)
            if args.boxes and st["n"] >= args.boxes:
                break
            if args.minutes and time.time() - t_start > args.minutes * 60:
                break
    except KeyboardInterrupt:
        print("\n[main] stopping ...")
    finally:
        if hasattr(hw, "stop_feeding"):
            hw.stop_feeding()
        t_end = time.time()                      # let boxes already on the belt reach the arm
        while pipe.fifo and time.time() - t_end < pipe.transit_estimate() + 1.5:
            time.sleep(0.05)
        hw.stop_belt()
        time.sleep(0.5)
        save_agent()
        if agent_info and pipe.stats["n"]:
            ok = pipe.reporter.upload_qtable()        # final backup of what was learned in this run
            print(f"[main] Q-table backup to the cloud: {'ok' if ok else 'failed (server unreachable?)'}")
        stress.set(False)
        pipe.reporter.stop(); net.stop(); cam.close(); cloud.close(); pipe.close(); hw.close()
        if link is not None:
            link.close()
        st = pipe.stats
        if st["n"]:
            print(f"[main] done: {st['n']} boxes | ESP32 {st['edge']} | gateway {st['gateway']} | cloud {st['cloud']} | "
                  f"missed {st['missed']} | "
                  f"rejected {st['rejected']} | log: {os.path.relpath(pipe.log_path)}")
        sys.stdout.flush()                 # os._exit skips Python's normal exit, which would flush this
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
