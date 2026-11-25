"""
Network + gateway monitor: everything the agent 'sees' apart from the time budget.

  rtt_ms   : ping the cloud server (GET /ping) every 0.5 s, smoothed (EWMA)
  cloud_ms : send a small TEST IMAGE to the cloud server every 2 s and time the full
             answer; real box requests also update it. Smoothed (EWMA).
             -> this is how the gateway notices a slow cloud server or a capped Wi-Fi,
                which a ping alone cannot show.
  cpu      : gateway CPU load 0..1 (psutil), smoothed
"""
from __future__ import annotations

import math
import threading
import time

from common.algos import ewma
from common.net import make_session

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class NetMonitor:
    def __init__(self, cfg: dict, base_url: str, cloud_client, probe_jpeg: bytes, netemu=None,
                 api_key: str = "", device: str = "pi-1"):
        n = cfg["network"]
        self.url = base_url.rstrip("/")
        self.cc = cloud_client
        self.probe_jpeg = probe_jpeg
        self.netemu = netemu
        self.ping_every = float(n["probe_interval_s"])
        self.ping_timeout = float(n["probe_timeout_s"])
        self.a = float(n["ewma_alpha"])
        self.cprobe_every = float(n["cloud_probe_interval_s"])
        self.cprobe_timeout = float(n["cloud_probe_timeout_s"])
        self.ca = float(n["cloud_ewma_alpha"])
        self.rtt_ms = None
        self.cloud_ms = None
        self.cpu = 0.2
        self.cpu_override = None       # mock "gateway overload" button
        self.server_ok = False
        self._lock = threading.Lock()
        self._stop = False
        self.s = make_session(api_key, device)      # keep-alive: a ping is one request, not a new TCP handshake

    def start(self):
        for fn in (self._ping_loop, self._cloud_probe_loop, self._cpu_loop):
            threading.Thread(target=fn, daemon=True).start()

    # ------------------------------------------------------------------
    @staticmethod
    def _ewma(old, new, a):
        return ewma(old, new, a)          # O(1) smoothing, no history kept

    def _ping_once(self) -> float:
        t0 = time.perf_counter()
        try:
            if self.netemu is not None:
                d = self.netemu.uplink_delay(100) + self.netemu.downlink_delay()
                if math.isinf(d) or d > self.ping_timeout:
                    time.sleep(self.ping_timeout)
                    return self.ping_timeout
                time.sleep(d)
            self.s.get(f"{self.url}/ping", timeout=self.ping_timeout)
            self.server_ok = True
            return min(self.ping_timeout, time.perf_counter() - t0)
        except Exception:
            self.server_ok = False
            return self.ping_timeout

    def _ping_loop(self):
        while not self._stop:
            t = time.perf_counter()
            v = self._ping_once()
            with self._lock:
                self.rtt_ms = self._ewma(self.rtt_ms, v * 1000, self.a)
            time.sleep(max(0.0, self.ping_every - (time.perf_counter() - t)))

    def _cloud_probe_loop(self):
        while not self._stop:
            t = time.perf_counter()
            res = self.cc.post_image(self.probe_jpeg, self.cprobe_timeout, probe=True)
            lat = time.perf_counter() - t if res.get("ok") else self.cprobe_timeout
            self.record_cloud(min(lat, self.cprobe_timeout))
            time.sleep(max(0.0, self.cprobe_every - (time.perf_counter() - t)))

    def _cpu_loop(self):
        while not self._stop:
            if psutil is not None:
                v = psutil.cpu_percent(interval=0.5) / 100.0
            else:
                time.sleep(0.5)
                v = 0.2
            with self._lock:
                self.cpu = self._ewma(self.cpu, v, 0.5)

    # ------------------------------------------------------------------
    def record_cloud(self, latency_s: float):
        """Real box requests also tell us how fast the cloud server answers."""
        with self._lock:
            self.cloud_ms = self._ewma(self.cloud_ms, latency_s * 1000, self.ca)

    def snapshot(self) -> dict:
        with self._lock:
            cpu = self.cpu_override if self.cpu_override is not None else self.cpu
            return {"rtt_ms": self.rtt_ms if self.rtt_ms is not None else self.ping_timeout * 1000,
                    "cloud_ms": self.cloud_ms if self.cloud_ms is not None else self.cprobe_timeout * 1000,
                    "cpu": cpu, "server_ok": self.server_ok}

    def wait_ready(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.rtt_ms is not None and self.cloud_ms is not None:
                return True
            time.sleep(0.1)
        return False

    def stop(self):
        self._stop = True
