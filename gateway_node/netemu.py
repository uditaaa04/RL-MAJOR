"""
Software network emulator - ONLY for the cloud server demo (Windows has no `tc netem`).

On a Linux gateway, real network conditions can be created with tools/net_shape.sh
(Linux tc/netem on wlan0) and this emulator stays switched off.

Scenario names match tools/net_shape.sh and sim/scenarios.py.
"""
from __future__ import annotations

import math
import random
import threading

SCENARIOS = {
    # gateway on Wi-Fi -> Internet -> AWS EC2 ap-south-1 (Mumbai); same numbers as sim/scenarios.py
    #            rtt_s  jitter_s  loss   bw_mbps  stall_prob stall_s
    "good":    (0.030, 0.004, 0.000, 10.0, 0.00, (0.4, 0.8)),
    "busy":    (0.100, 0.030, 0.010, 4.0, 0.00, (0.4, 0.8)),
    "bad":     (0.280, 0.080, 0.030, 1.5, 0.00, (0.4, 0.8)),
    "bwcap":   (0.030, 0.004, 0.000, 0.3, 0.00, (0.4, 0.8)),
    "lossy":   (0.100, 0.030, 0.060, 4.0, 0.00, (0.4, 0.8)),
    "stalls":  (0.030, 0.004, 0.000, 10.0, 0.20, (0.4, 0.8)),
    "down":    (0.030, 0.000, 1.000, 10.0, 0.00, (0.4, 0.8)),
    # no emulation at all: the pretend belt talks to a REAL remote server (e.g. the EC2 instance)
    "real":    (0.000, 0.000, 0.000, float("inf"), 0.00, (0.4, 0.8)),
}


class NetEmu:
    def __init__(self, scenario: str = "good", seed=None):
        self.rng = random.Random(seed)
        self._lock = threading.Lock()
        self.set(scenario)

    def set(self, scenario: str):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario}; choose from {list(SCENARIOS)}")
        with self._lock:
            self.name = scenario
            (self.rtt, self.jitter, self.loss, self.bw, self.stall_p, self.stall_s) = SCENARIOS[scenario]

    def _one_way(self):
        return self.rtt / 2 + abs(self.rng.gauss(0, self.jitter))

    def _loss_penalty(self, n_pkts: int) -> float:
        if self.loss <= 0.0:              # nothing lost: skip the per-packet draws entirely
            return 0.0
        d = 0.0
        for _ in range(n_pkts):
            if self.rng.random() < self.loss:
                d += self.rtt if self.rng.random() < 0.7 else max(0.2, 2 * self.rtt)
        return d

    def uplink_delay(self, nbytes: int) -> float:
        """seconds to add before the request reaches the cloud server (None = lost for good)"""
        with self._lock:
            if self.loss >= 1.0:
                return math.inf
            n = max(1, math.ceil(nbytes / 1448))
            d = self._one_way() + nbytes * 8 / (self.bw * 1e6) + self._loss_penalty(n)
            if self.stall_p and self.rng.random() < self.stall_p:
                d += self.rng.uniform(*self.stall_s)
            return d

    def downlink_delay(self) -> float:
        with self._lock:
            if self.loss >= 1.0:
                return math.inf
            return self._one_way() + self._loss_penalty(1)
