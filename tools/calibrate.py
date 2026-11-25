"""
CALIBRATE the simulator with numbers measured on YOUR hardware (run on the gateway).

    python -m tools.calibrate --server http://<ec2-ip>:8000               # full (key: env CONVEYOR_API_KEY)
    python -m tools.calibrate --server http://<ec2-ip>:8000 --no-camera   # no camera yet

Measures
  * camera capture time and real photo size (JPEG KB)
  * small model (edge) time on the gateway        - 100 runs
  * big model ON THE PI (to show why we don't run it there) - 20 runs, optional
  * ping time and full cloud answer time     - 50 each
  * cloud model time (reported by the server)
  * belt speed: optional, put boxes on the running belt (--belt-boxes 5)

Writes config/calibration.json. Then on the laptop:
    python -m sim.train_q --calib config/calibration.json
    python -m sim.experiments
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as stt
import time

import numpy as np
from PIL import Image

from common.config import load_config, project_path
from common.net import api_key, device_id, make_session
from common.vision_io import OnnxInspector, to_jpeg


def med(xs):
    return float(stt.median(xs)) if xs else None


def lognorm_sigma(xs):
    xs = [x for x in xs if x > 0]
    return float(np.std(np.log(xs))) if len(xs) > 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None, help="cloud server URL (default: env CONVEYOR_SERVER / config)")
    ap.add_argument("--api-key", default=None, help="default: env CONVEYOR_API_KEY")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--bench-big", action="store_true", help="also time the big model on the gateway")
    ap.add_argument("--belt-boxes", type=int, default=0, help="measure belt transit with N boxes")
    a = ap.parse_args()
    cfg = load_config()
    out = {"measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    url = (a.server or os.environ.get("CONVEYOR_SERVER") or cfg["server"]["cloud_url"]).rstrip("/")
    if "<" in url:
        raise SystemExit("give the server: --server http://<ec2-ip>:8000 (or env CONVEYOR_SERVER)")

    # ---- camera
    img = Image.open(project_path("models/probe.jpg")).convert("RGB")
    if not a.no_camera:
        from gateway_node.camera import Camera
        cam = Camera(cfg)
        ts = []
        for _ in range(30):
            t = time.perf_counter(); img = cam.capture_box(); ts.append(time.perf_counter() - t)
        out["capture_med_s"] = med(ts)
        out["capture_sd_s"] = float(np.std(ts))
        cam.close()
    jpeg = to_jpeg(img, cfg["camera"]["jpeg_quality"])
    ts = []
    for _ in range(30):
        t = time.perf_counter(); to_jpeg(img, cfg["camera"]["jpeg_quality"]); ts.append(time.perf_counter() - t)
    out["jpeg_s"] = med(ts)
    out["payload_kb"] = len(jpeg) / 1000.0
    print(f"photo: {out['payload_kb']:.1f} KB, jpeg encode {out['jpeg_s']*1000:.1f} ms")

    # ---- edge model
    edge = OnnxInspector(project_path(cfg["models"]["gateway_onnx"]), threads=cfg["models"]["gateway_threads"],
                         providers=["CPUExecutionProvider"])
    for _ in range(5):
        edge.predict(img)
    ts = [edge.predict(img)[2] for _ in range(100)]
    out["gw_med_s"], out["gw_sigma"] = med(ts), lognorm_sigma(ts)
    print(f"small model on gateway: median {out['gw_med_s']*1000:.1f} ms")
    if a.bench_big:
        big = OnnxInspector(project_path(cfg["models"]["cloud_onnx"]), threads=cfg["models"]["gateway_threads"],
                            providers=["CPUExecutionProvider"])
        for _ in range(3):
            big.predict(img)
        ts = [big.predict(img)[2] for _ in range(20)]
        out["big_model_on_pi_med_s"] = med(ts)
        print(f"BIG model on gateway:   median {out['big_model_on_pi_med_s']*1000:.1f} ms  (this is why it runs in the cloud)")

    # ---- network + cloud server (gateway -> Internet -> AWS)
    s = make_session(api_key(cfg, a.api_key), device_id(cfg))
    pings, clouds, server_ms = [], [], []
    for _ in range(50):
        t = time.perf_counter(); s.get(f"{url}/ping", timeout=2); pings.append(time.perf_counter() - t)
        time.sleep(0.05)
    for _ in range(50):
        t = time.perf_counter()
        r = s.post(f"{url}/infer", data=jpeg, params={"probe": 1}, headers={"Content-Type": "image/jpeg"}, timeout=3)
        clouds.append(time.perf_counter() - t)
        server_ms.append(r.json()["server_ms"] / 1000.0)
        time.sleep(0.05)
    out["ping_med_ms"] = med(pings) * 1000
    out["cloud_med_ms"] = med(clouds) * 1000
    out["server_med_s"], out["server_sigma"] = med(server_ms), lognorm_sigma(server_ms)
    overhead = max(5.0, out["cloud_med_ms"] - out["ping_med_ms"])
    out["cloud_overhead_ms"] = overhead
    print(f"ping {out['ping_med_ms']:.1f} ms | full cloud answer {out['cloud_med_ms']:.1f} ms | "
          f"cloud model {out['server_med_s']*1000:.1f} ms")

    # ---- belt
    if a.belt_boxes:
        from gateway_node.hardware import RealHardware
        hw = RealHardware(cfg)
        t_cam, transits = [], []
        hw.bind(lambda: t_cam.append(time.perf_counter()),
                lambda: transits.append(time.perf_counter() - t_cam.pop(0)) if t_cam else None)
        hw.start_belt()
        print(f"put {a.belt_boxes} boxes on the belt one by one ...")
        while len(transits) < a.belt_boxes:
            time.sleep(0.1)
        hw.stop_belt(); hw.close()
        tr = med(transits)
        speed = cfg["conveyor"]["cam_to_arm_m"] / tr
        out["transit_med_s"] = tr
        out["speed_mps_at_full_estimate"] = speed / cfg["conveyor"]["belt_duty"]
        print(f"camera->arm {tr:.3f} s  => belt {speed:.3f} m/s at duty {cfg['conveyor']['belt_duty']}"
              f"  (put speed_mps_at_full = {out['speed_mps_at_full_estimate']:.3f} in config)")

    p = project_path("config/calibration.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {p}\nNext (on the laptop): python -m sim.train_q --calib config/calibration.json && python -m sim.experiments")
    print(f"Also set state_bins.cloud_overhead_ms: {overhead:.0f} in config/system.yaml")


if __name__ == "__main__":
    main()
