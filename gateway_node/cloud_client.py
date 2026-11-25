"""
Sends a box photo to the cloud's big model and gets the verdict back.

Runs requests in a small thread pool, so the gateway keeps working while waiting.
The timeout of each request = the time left before the box reaches the arm
(an answer after that is useless anyway).
"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from common.net import make_session
from common.vision_io import to_jpeg


class CloudClient:
    def __init__(self, base_url: str, jpeg_quality: int = 85, netemu=None, workers: int = 4,
                 api_key: str = "", device: str = "pi-1"):
        self.url = base_url.rstrip("/")
        self.q = int(jpeg_quality)
        self.netemu = netemu
        self.api_key, self.device = api_key, device
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cloud")
        self._inflight = 0
        self._lock = threading.Lock()
        self._local = threading.local()
        self.auth_failed = False

    def _session(self):
        # one keep-alive session per thread: no new TCP (and TLS) handshake for every box
        s = getattr(self._local, "s", None)
        if s is None:
            s = make_session(self.api_key, self.device)
            self._local.s = s
        return s

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def post_image(self, jpeg: bytes, timeout_s: float, probe: bool = False, box_id: int | None = None):
        """Blocking call. Returns dict(ok, p_damaged, heatmap, server_ms, error)."""
        t0 = time.perf_counter()
        if self.netemu is not None:
            d = self.netemu.uplink_delay(len(jpeg))
            if d >= timeout_s or math.isinf(d):
                time.sleep(max(0.0, min(timeout_s, 5.0)))
                return {"ok": False, "error": "timeout (emulated network)"}
            time.sleep(d)
        remaining = max(0.05, timeout_s - (time.perf_counter() - t0))
        try:
            r = self._session().post(f"{self.url}/infer", data=jpeg,
                                     params={"probe": 1} if probe else ({"box": box_id} if box_id is not None else None),
                                     headers={"Content-Type": "image/jpeg"}, timeout=remaining)
            if r.status_code == 401:
                if not self.auth_failed:
                    print("[cloud] server says 401: wrong or missing API key (CONVEYOR_API_KEY)")
                self.auth_failed = True
            r.raise_for_status()
            out = r.json()
        except requests.exceptions.Timeout:
            return {"ok": False, "error": "timeout"}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
        if self.netemu is not None:
            d = self.netemu.downlink_delay()
            left = timeout_s - (time.perf_counter() - t0)
            if math.isinf(d) or d > left:
                time.sleep(max(0.0, left))
                return {"ok": False, "error": "timeout (emulated network)"}
            time.sleep(d)
        out["ok"] = True
        return out

    def submit(self, box_id: int, img, timeout_s: float, callback):
        """Encode + send in the background. callback(box_id, result_dict) is called when done."""
        with self._lock:
            self._inflight += 1

        def _job():
            t = time.perf_counter()
            jpeg = to_jpeg(img, self.q)
            enc_s = time.perf_counter() - t
            res = self.post_image(jpeg, timeout_s - enc_s, box_id=box_id)
            res["jpeg_s"] = enc_s
            res["jpeg_bytes"] = len(jpeg)
            res["where"] = "CLOUD"
            with self._lock:
                self._inflight -= 1
            callback(box_id, res)
        self.pool.submit(_job)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
