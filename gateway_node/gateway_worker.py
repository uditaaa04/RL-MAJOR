"""Runs the small model on the gateway laptop in ONE background thread (first come, first served).

It also times itself: `ewma_ms` is the smoothed time one photo takes right now. The agent uses it
to tell "the gateway is fine" from "the gateway is too busy to answer before the box reaches the arm".
When no box has used the model for 2 seconds it times one blank photo, so the number stays fresh.
"""
from __future__ import annotations

import queue
import threading
import time

from common.vision_io import OnnxInspector


class GatewayWorker:
    PROBE_EVERY_S = 2.0

    def __init__(self, onnx_path: str, threads: int = 4, slowdown: float = 1.0):
        self.model = OnnxInspector(onnx_path, threads=threads, providers=["CPUExecutionProvider"])
        self.slowdown = float(slowdown)      # multiply the model time (1 = this laptop as it is)
        self.extra_s = 0.0                   # pretend overload: every photo takes this much longer
        self.q: queue.Queue = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        self.ewma_ms = 0.0
        self._last_run = 0.0
        # warm-up (first ONNX run is always slow)
        from PIL import Image
        self._blank = Image.new("RGB", (256, 256))
        for _ in range(3):
            self.model.predict(self._blank)
        self._time_once()
        threading.Thread(target=self._loop, daemon=True, name="gateway-model").start()
        threading.Thread(target=self._probe_loop, daemon=True, name="gateway-probe").start()

    @property
    def queue_len(self) -> int:
        with self._lock:
            return self._pending

    def submit(self, box_id: int, img, callback):
        with self._lock:
            self._pending += 1
        self.q.put((box_id, img, callback))

    def _run(self, img):
        t = time.perf_counter()
        p, heat, compute_s = self.model.predict(img)
        if self.slowdown > 1.0:
            time.sleep(compute_s * (self.slowdown - 1.0))
        if self.extra_s > 0:
            time.sleep(self.extra_s)
        svc = time.perf_counter() - t
        self.ewma_ms = svc * 1000.0 if self.ewma_ms <= 0 else 0.6 * self.ewma_ms + 0.4 * svc * 1000.0
        self._last_run = time.perf_counter()
        return p, heat, svc

    def _time_once(self):
        self._run(self._blank)

    def check(self, img) -> dict:
        """Blocking: one photo -> result dict (also used by the speed-accuracy measurement)."""
        p, heat, svc = self._run(img)
        return {"ok": True, "p_damaged": p, "heatmap": heat, "compute_s": svc, "where": "GATEWAY"}

    def _loop(self):
        while True:
            box_id, img, cb = self.q.get()
            try:
                res = self.check(img)
            except Exception as e:
                res = {"ok": False, "error": f"gateway model: {e}", "where": "GATEWAY"}
            with self._lock:
                self._pending -= 1
            cb(box_id, res)

    def _probe_loop(self):
        while True:
            time.sleep(0.5)
            if self.queue_len == 0 and time.perf_counter() - self._last_run > self.PROBE_EVERY_S:
                # goes through the same queue, so it never runs next to a real box
                with self._lock:
                    self._pending += 1
                self.q.put((-1, self._blank, lambda *_: None))
