"""
Run the SAME experiments as the simulator, on the real conveyor (or the mock).

Start the cloud server first (AWS: cloud/deploy_aws.ps1, or locally: python -m server.app).
Then, on the gateway laptop (real belt with the ESP32 plugged in, or --mock for the pretend belt).
The API key comes from env CONVEYOR_API_KEY or --api-key:

  # E1-style: every policy in every network condition, N boxes each
  python -m tools.run_experiment steady --policies qlearning adaptive threshold always_edge always_gateway always_cloud \\
         --scenarios good busy bad --boxes 100 [--mock]

  # E2-style: normal -> problem -> normal, N boxes per phase, for each policy
  python -m tools.run_experiment change --change bad|bwcap|stalls|server_slow --boxes 100 \\
         --policies qlearning adaptive threshold [--mock]

Real conveyor: keep feeding boxes while it runs (it stops by itself after the set
number of boxes). On a Linux gateway the network conditions are switched with tools/net_shape.sh (sudo);
on Windows use a real bad network (phone hotspot far away) or the cloud-overload switch.
Logs go to logs/<exp>_<policy>_<scenario>/boxes.csv -> then: python -m tools.report_real
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from common.config import load_config, project_path
from common.net import api_key, device_id, make_session

S = None          # authenticated HTTP session to the cloud server (set in main)


def shape(scenario: str, mock: bool, server: str, iface: str):
    """Switch the network condition."""
    if scenario == "server_slow":
        S.post(f"{server}/api/control", json={"server_slow": True}, timeout=3)
        return
    S.post(f"{server}/api/control", json={"server_slow": False, "scenario": scenario}, timeout=3)
    if not mock and sys.platform.startswith("linux"):  # Linux gateway: tc netem
        subprocess.run(["sudo", "bash", project_path("tools/net_shape.sh"), scenario, iface], check=False)


def run_node(args_list, mock):
    cmd = [sys.executable, "-m", "gateway_node.main"] + args_list + (["--mock"] if mock else [])
    print("  $", " ".join(cmd[2:]))
    return subprocess.Popen(cmd, cwd=project_path("."))


def count(server):
    try:
        return S.get(f"{server}/api/state", timeout=3).json()["all"].get("n", 0)
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", choices=["steady", "change"])
    ap.add_argument("--policies", nargs="+",
                    default=["qlearning", "adaptive", "threshold", "always_edge", "always_gateway", "always_cloud"])
    ap.add_argument("--scenarios", nargs="+", default=["good", "busy", "bad"])
    ap.add_argument("--change", default="bad", help="bad | bwcap | stalls | lossy | server_slow")
    ap.add_argument("--boxes", type=int, default=100)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--server", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--name", default=time.strftime("exp%m%d_%H%M"))
    a = ap.parse_args()
    cfg = load_config()
    server = (a.server or os.environ.get("CONVEYOR_SERVER")
              or ("http://127.0.0.1:%d" % cfg["server"]["port"] if a.mock else cfg["server"]["cloud_url"])).rstrip("/")
    global S
    key = api_key(cfg, a.api_key)
    S = make_session(key, device_id(cfg))
    if key:
        os.environ["CONVEYOR_API_KEY"] = key      # the gateway_node child processes read it from here
    iface = cfg.get("pi", {}).get("iface", "wlan0")

    if a.plan == "steady":
        for sc in a.scenarios:
            for pol in a.policies:
                tag = f"{a.name}_steady_{pol}_{sc}"
                print(f"\n=== {pol} | network: {sc} | {a.boxes} boxes ===")
                S.post(f"{server}/api/reset", timeout=3)
                shape(sc, a.mock, server, iface)
                if not a.mock:
                    input("  press Enter, then keep feeding boxes ...")
                p = run_node(["--policy", pol, "--boxes", str(a.boxes), "--tag", tag, "--server", server]
                             + (["--scenario", sc] if a.mock else []), a.mock)
                p.wait()
        shape("good", a.mock, server, iface)
    else:
        for pol in a.policies:
            tag = f"{a.name}_change-{a.change}_{pol}"
            print(f"\n=== {pol} | good -> {a.change} -> good | {a.boxes} boxes per phase ===")
            S.post(f"{server}/api/reset", timeout=3)
            shape("good", a.mock, server, iface)
            if not a.mock:
                input("  press Enter, then keep feeding boxes ...")
            p = run_node(["--policy", pol, "--boxes", str(3 * a.boxes), "--tag", tag, "--server", server]
                         + (["--scenario", "good"] if a.mock else []), a.mock)
            phase = 0
            while p.poll() is None:
                n = count(server)
                if phase == 0 and n >= a.boxes:
                    print(f"  >>> box {n}: problem ON ({a.change})"); shape(a.change, a.mock, server, iface); phase = 1
                elif phase == 1 and n >= 2 * a.boxes:
                    print(f"  >>> box {n}: problem OFF"); shape("good", a.mock, server, iface); phase = 2
                time.sleep(0.3)
            shape("good", a.mock, server, iface)
    print("\nall runs finished -> python -m tools.report_real --prefix", a.name)


if __name__ == "__main__":
    main()
