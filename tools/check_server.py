"""
Check a cloud server and measure its timing from where you run this (the gateway, or any PC).

    python -m tools.check_server --server http://<ec2-ip>:8000            # key from env CONVEYOR_API_KEY
    python -m tools.check_server --server http://127.0.0.1:8000 --n 50

What it checks
  1. /ping and /health answer
  2. the API key is enforced (a request WITHOUT the key must get 401)
  3. /infer with the 8 kB test image works
What it measures (median / 95th percentile)  -> the "measured" column of the paper's Table IV
  * ping round trip                         (GET /ping)
  * full answer time for one box photo      (POST /infer, a 256x256 JPEG as the gateway sends it)
  * server inference time                   (server_ms reported by the server)
Results are also written to results/server_check.json.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import requests

from common.algos import percentile_sorted
from common.config import load_config, project_path
from common.net import api_key, device_id, make_session


def stats(xs):
    s = sorted(xs)
    return {"n": len(s), "median_ms": round(percentile_sorted(s, 50), 1), "p95_ms": round(percentile_sorted(s, 95), 1),
            "min_ms": round(s[0], 1) if s else None, "max_ms": round(s[-1], 1) if s else None}


def box_jpeg() -> bytes:
    """A 256x256 photo like the ones the gateway uploads (from the mock test set if present)."""
    from PIL import Image
    d = project_path("data/synth/test/damaged")
    img = None
    if os.path.isdir(d):
        fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".png")))
        if fs:
            img = Image.open(os.path.join(d, fs[0])).convert("RGB").resize((256, 256))
    if img is None:
        img = Image.open(project_path("models/probe.jpg")).convert("RGB").resize((256, 256))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--n", type=int, default=30, help="repetitions per measurement")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    url = (a.server or os.environ.get("CONVEYOR_SERVER") or cfg["server"]["cloud_url"]).rstrip("/")
    if "<" in url:
        raise SystemExit("give the server: --server http://<ip>:8000")
    key = api_key(cfg, a.api_key)
    s = make_session(key, device_id(cfg))
    ok = True
    report = {"server": url, "when": time.strftime("%Y-%m-%d %H:%M:%S")}

    def line(name, good, extra=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  [{'OK' if good else 'FAIL'}] {name} {extra}")

    print(f"checking {url}")
    try:
        h = s.get(f"{url}/health", timeout=5).json()
    except Exception as e:
        print(f"  [FAIL] server not reachable: {e}")
        return 2
    line("health", h.get("ok"), f"model={h.get('model')} uptime={h.get('uptime_s')}s auth={h.get('auth')} aws={h.get('aws')}")
    report["health"] = h
    if h.get("auth"):
        r = requests.get(f"{url}/api/state", timeout=5)
        line("API key enforced (no key -> 401)", r.status_code == 401, f"(got {r.status_code})")
        r = s.get(f"{url}/api/state", timeout=5)
        line("API key accepted", r.status_code == 200, f"(got {r.status_code})")
    else:
        print("  [WARN] server has NO API key - do not expose it on the Internet like this")

    # ---- ping
    pings = []
    for _ in range(a.n):
        t = time.perf_counter()
        s.get(f"{url}/ping", timeout=3)
        pings.append((time.perf_counter() - t) * 1000)
    report["ping"] = stats(pings)

    # ---- one box photo, end to end
    jpg = box_jpeg()
    full, server, last_p = [], [], None
    for i in range(a.n):
        t = time.perf_counter()
        r = s.post(f"{url}/infer", data=jpg, params={"probe": 1}, headers={"Content-Type": "image/jpeg"}, timeout=5)
        full.append((time.perf_counter() - t) * 1000)
        if r.ok:
            j = r.json()
            server.append(float(j.get("server_ms", 0)))
            last_p = j.get("p_damaged")
        elif i == 0:
            line("/infer", False, f"(HTTP {r.status_code}: {r.text[:120]})")
            break
    if server:
        line("/infer", True, f"P(damaged) of a damaged test box = {last_p:.2f}")
    report["infer_full"] = stats(full)
    report["infer_server"] = stats(server) if server else None
    report["payload_kB"] = round(len(jpg) / 1024, 1)

    print(f"\n  {'measurement':38s} {'median':>9s} {'p95':>9s}")
    for name, k in (("ping round trip", "ping"), (f"box photo answer ({report['payload_kB']} kB JPEG)", "infer_full"),
                    ("server inference (server_ms)", "infer_server")):
        if report.get(k):
            print(f"  {name:38s} {report[k]['median_ms']:7.1f}ms {report[k]['p95_ms']:7.1f}ms")
    os.makedirs(project_path("results"), exist_ok=True)
    with open(project_path("results/server_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  saved results/server_check.json   overall: {'OK' if ok else 'PROBLEMS FOUND'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
