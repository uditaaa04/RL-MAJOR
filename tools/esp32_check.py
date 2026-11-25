"""
Test the ESP32 edge node on its own (no belt program, no cloud).

    python -m tools.esp32_check              # finds the board by itself
    python -m tools.esp32_check --port COM5

What it checks, in order: the board answers, the tiny model loads, 30 photos are timed, the reject arm
swings, then for 10 seconds it prints every IR sensor event so you can wave a hand
in front of each sensor. The belt runs slowly during those 10 seconds.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from common import tiny
from common.config import load_config, project_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--no-move", action="store_true", help="do not move the arm or the belt")
    a = ap.parse_args()
    cfg = load_config()
    from gateway_node.esp32_link import Esp32Error, connect, list_ports
    ports = list_ports()
    print("USB serial ports found:", ", ".join(f"{p} ({d.split(' USB VID')[0].strip()})" for p, d in ports) or "none")
    tiny_path = project_path(cfg["models"].get("edge_tiny", "models/edge_tiny.bin"))
    model = tiny.TinyModel.load(tiny_path) if os.path.exists(tiny_path) else None
    try:
        link = connect(cfg, model.blob if model else None, a.port)
    except Esp32Error as e:
        print(f"FAILED: {e}")
        print("Check: 1) firmware flashed from esp32_node/esp32_node.ino  2) Arduino Serial Monitor closed  "
              "3) a USB DATA cable  4) the CP210x / CH340 driver is installed (a COM port must appear when you plug in).")
        return 1
    ok = True
    st = link.status()
    print(f"1. board answers: {link.firmware}   free memory {st['free_bytes'] if st else '?'} bytes")
    if model is None:
        print("2. tiny model: NONE on this PC yet (models/edge_tiny.bin). Train it, then run this again.")
    elif not link.model_loaded:
        print("2. tiny model: the board REFUSED it (the reason is in the WARNING line above). "
              "Unplug the board, plug it in again and run this check again.")
        ok = False
    else:
        print(f"2. tiny model loaded: {model.size}x{model.size} grey photo, {model.params} numbers, {len(model.blob)} bytes")
        rng = np.random.default_rng(0)
        full, esp, diff = [], [], 0.0
        for _ in range(30):
            px = rng.integers(0, 256, model.size * model.size * model.ch, dtype=np.uint8).tobytes()
            t = time.perf_counter()
            try:
                r = link.infer(px, 1.0)
            except Esp32Error as e:
                print(f"   photo failed: {e}")
                ok = False
                continue
            full.append((time.perf_counter() - t) * 1000)
            esp.append(r["esp_us"] / 1000)
            diff = max(diff, abs(r["p"] - model.predict_pixels(px)))
        if full:
            print(f"3. 30 test photos: {np.median(full):.1f} ms each including the USB cable "
                  f"(model alone on the ESP32: {np.median(esp):.1f} ms). "
                  f"Answers match the PC's own calculation to {diff:.4f}" + ("  OK" if diff < 0.002 else "  MISMATCH"))
            ok = ok and diff < 0.002
    if not a.no_move:
        print("4. reject arm: swinging twice ...")
        for _ in range(2):
            link.push(int(cfg.get("esp32", {}).get("servo_hold_ms", 350)))
            time.sleep(1.0)
        seen = {"C": 0, "A": 0}

        def ev(tag, t):
            seen[tag] += 1
            print(f"   sensor event: {'box under the CAMERA (IR 1)' if tag == 'C' else 'box at the ARM (IR 2)'}")
        link.on_event = ev
        print("5. belt runs slowly for 10 seconds (it only moves with 12 V on the motor driver). "
              "Wave a hand in front of each IR sensor now ...")
        link.set_duty(round(0.5 * 255))
        time.sleep(10)
        link.set_duty(0)
        print(f"   events seen: camera sensor {seen['C']}, arm sensor {seen['A']}"
              + ("" if seen["C"] and seen["A"] else "   <- a sensor with 0 events is not wired, or wired to another pin"))
        st = link.status()
        if st and (st["ir_camera"] or st["ir_arm"]):
            print("   NOTE: a sensor reports a box right now with nothing in front of it. If so, set "
                  "esp32.ir_active_low: false in config/system.yaml.")
    link.close()
    print("ESP32 CHECK PASSED" if ok else "ESP32 CHECK FOUND PROBLEMS (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
