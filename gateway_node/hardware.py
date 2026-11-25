"""
The pretend conveyor (no hardware needed).

MockHardware - boxes appear at random gaps, "box at camera" / "box at arm" events fire at the right
               times, push() just counts. Each box is a photo from disk, so the true label is known.

The REAL belt is driven through the ESP32: see Esp32Hardware in gateway_node/esp32_link.py.
Both offer the same methods, so the rest of the gateway does not care which one it has:
    bind(on_box_at_camera, on_box_at_arm), start_belt(), set_belt(), stop_belt(),
    belt_speed_mps(), push(), show(verdict), power_w(), close()
"""
from __future__ import annotations

import os
import random
import threading
import time


class MockHardware:
    """Pretend conveyor. Each box is an image from mock.feed_dir (good/ or damaged/)."""

    def __init__(self, cfg: dict, feed_dir: str, seed: int | None = None):
        self.cfg = cfg
        m = cfg["mock"]
        self.rng = random.Random(seed)
        self.gap = tuple(m["box_gap_s"])
        self.speed = float(m["belt_speed_mps"])
        self.duty = self.speed / float(cfg["conveyor"]["speed_mps_at_full"])
        self.items = []
        for lab in ("good", "damaged"):
            d = os.path.join(feed_dir, lab)
            if os.path.isdir(d):
                self.items += [(os.path.join(d, f), lab) for f in sorted(os.listdir(d))
                               if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not self.items:
            raise SystemExit(f"mock feed folder has no images: {feed_dir}")
        self.on_cam = self.on_arm = None
        self._running = False
        self.current = None           # (path, label) of the box under the camera
        self.truth = {}               # box number -> label (for accuracy reporting)
        self.pushes = 0
        self.n_boxes = 0
        self._feeding = True

    def bind(self, on_box_at_camera, on_box_at_arm):
        self.on_cam, self.on_arm = on_box_at_camera, on_box_at_arm

    def start_belt(self, duty: float | None = None):
        if duty is not None:
            self.set_belt(duty)
        if not self._running:
            self._running = True
            threading.Thread(target=self._feeder, daemon=True).start()

    def set_belt(self, duty: float):
        self.duty = max(0.05, min(1.0, float(duty)))
        self.speed = self.duty * float(self.cfg["conveyor"]["speed_mps_at_full"])

    def stop_feeding(self):
        """No new boxes; boxes already on the belt still travel to the arm."""
        self._feeding = False

    def stop_belt(self):
        self._running = False

    def belt_speed_mps(self) -> float:
        return self.speed

    def _feeder(self):
        cam_to_arm = float(self.cfg["conveyor"]["cam_to_arm_m"])
        while self._running:
            time.sleep(self.rng.uniform(*self.gap))
            if not self._running:
                break
            if not self._feeding:
                continue
            item = self.rng.choice(self.items)
            self.current = item
            self.truth[self.n_boxes] = item[1]
            self.n_boxes += 1
            transit = cam_to_arm / max(0.02, self.speed)
            threading.Timer(transit, self._arrive_arm).start()
            if self.on_cam:
                self.on_cam()

    def _arrive_arm(self):
        if self.on_arm and self._running:
            self.on_arm()

    def push(self):
        self.pushes += 1

    def show(self, verdict: str):
        pass

    def power_w(self):
        return None

    def close(self):
        self._running = False
