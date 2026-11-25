"""
Talks to the cloud dashboard - never blocks the box loop.

  * event(dict)  : one finished box -> queued -> POST /api/events (batched)
  * status every 1 s: network numbers, belt speed, policy, Q-table snapshot
  * polls /api/control: scenario buttons, belt speed, policy switch, gateway overload
  * every `qtable_every_s` seconds: uploads the online-learned Q-table (POST /api/qtable)
    so the cloud keeps a backup of what the agent has learned on the line
"""
from __future__ import annotations

import queue
import threading
import time

from common.net import make_session


class Reporter:
    def __init__(self, base_url: str, status_fn, on_control=None, every_s: float = 1.0,
                 api_key: str = "", device: str = "pi-1", qtable_fn=None, qtable_every_s: float = 60.0):
        self.url = base_url.rstrip("/")
        self.status_fn = status_fn
        self.on_control = on_control
        self.every = every_s
        self.qtable_fn = qtable_fn
        self.qtable_every = qtable_every_s
        self.q: queue.Queue = queue.Queue(maxsize=5000)
        self.s = make_session(api_key, device)
        self._stop = False
        self._last_control = None
        self._warned = False
        threading.Thread(target=self._send_loop, daemon=True).start()
        threading.Thread(target=self._status_loop, daemon=True).start()

    def event(self, ev: dict):
        try:
            self.q.put_nowait(ev)
        except queue.Full:
            pass

    def _check(self, r):
        if r.status_code == 401 and not self._warned:
            self._warned = True
            print("[reporter] dashboard says 401: wrong or missing API key (CONVEYOR_API_KEY)")
        return r

    def _send_loop(self):
        while not self._stop:
            batch = [self.q.get()]
            while not self.q.empty() and len(batch) < 50:
                batch.append(self.q.get_nowait())
            try:
                self._check(self.s.post(f"{self.url}/api/events", json=batch, timeout=2.0))
            except Exception:
                time.sleep(1.0)   # dashboard not reachable - drop this batch, keep running

    def upload_qtable(self) -> bool:
        if self.qtable_fn is None:
            return False
        data = self.qtable_fn()
        if not data:
            return False
        try:
            r = self._check(self.s.post(f"{self.url}/api/qtable", data=data,
                                        headers={"Content-Type": "application/octet-stream"}, timeout=5.0))
            return r.ok
        except Exception:
            return False

    def _status_loop(self):
        n = 0
        last_q = time.time()
        while not self._stop:
            n += 1
            try:
                st = self.status_fn(full_q=(n % 5 == 1))
                self._check(self.s.post(f"{self.url}/api/status", json=st, timeout=2.0))
                if self.on_control:
                    r = self._check(self.s.get(f"{self.url}/api/control", timeout=2.0))
                    if r.ok:
                        ctl = r.json()
                        if ctl != self._last_control:
                            self._last_control = ctl
                            self.on_control(ctl)
                if self.qtable_fn and time.time() - last_q >= self.qtable_every:
                    last_q = time.time()
                    self.upload_qtable()
            except Exception:
                pass
            time.sleep(self.every)

    def stop(self):
        self._stop = True
