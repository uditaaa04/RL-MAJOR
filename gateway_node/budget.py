"""Cloud-call budget: so many cloud calls per minute, unused ones can be saved up to `burst`.

The cloud costs money and bandwidth, so the gateway may only ask it a limited number of times.
This is a token bucket: one token = one cloud call; tokens drip in at per_min / 60 per second.
per_min = 0 means no limit.
"""
from __future__ import annotations

import threading
import time


class CloudBudget:
    def __init__(self, per_min: float = 0.0, burst: float = 5.0):
        self._lock = threading.Lock()
        self.per_min = max(0.0, float(per_min or 0.0))
        self.burst = max(1.0, float(burst or 1.0))
        self._tokens = self.burst
        self._t = time.perf_counter()
        self.used = 0                      # cloud calls made since the start

    def _refill(self):
        now = time.perf_counter()
        if self.per_min > 0:
            self._tokens = min(self.burst, self._tokens + (now - self._t) * self.per_min / 60.0)
        self._t = now

    def set(self, per_min: float, burst: float | None = None):
        with self._lock:
            self._refill()
            self.per_min = max(0.0, float(per_min or 0.0))
            if burst is not None:
                self.burst = max(1.0, float(burst))
            self._tokens = min(self._tokens, self.burst)

    def snapshot(self) -> dict:
        """limited, calls that could be made right now, share of the bucket left, can a call be made."""
        with self._lock:
            self._refill()
            if self.per_min <= 0:
                return {"limited": False, "calls_left": 99.0, "share": 1.0, "can": True, "per_min": 0.0,
                        "burst": self.burst, "used": self.used}
            return {"limited": True, "calls_left": self._tokens, "share": self._tokens / self.burst,
                    "can": self._tokens >= 1.0, "per_min": self.per_min, "burst": self.burst, "used": self.used}

    def take(self) -> bool:
        """Spend one cloud call. False = none left (the caller must not ask the cloud)."""
        with self._lock:
            self._refill()
            if self.per_min > 0:
                if self._tokens < 1.0:
                    return False
                self._tokens -= 1.0
            self.used += 1
            return True
