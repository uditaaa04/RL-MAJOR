"""
Talks to the cloud dashboard - never blocks the box loop.

  * event(dict)  : one finished box -> queued -> POST /api/events at once (batched only if several are waiting)
  * status every 1 s, and at once after nudge() (an IR sensor changed, a box finished): network numbers,
    belt speed, policy, live ESP32 readings, Q-table snapshot
  * waits on /api/control (long polling): a dashboard button reaches the gateway one network trip after
    it is pressed - scenario buttons, belt speed / on / off, policy switch, gateway overload, manual tests
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
        self._wake = threading.Event()       # nudge(): send the status now instead of at the next second
        self.cs = make_session(api_key, device)   # own connection: a waiting control request must not block status
        threading.Thread(target=self._send_loop, daemon=True).start()
        threading.Thread(target=self._status_loop, daemon=True).start()
        if self.on_control:
            threading.Thread(target=self._control_loop, daemon=True).start()

    def nudge(self):
        """Something the dashboard should show at once (an IR sensor, the arm): send the status now."""
        self._wake.set()

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
        last_full = 0.0
        while not self._stop:
            n += 1
            t0 = time.time()
            try:
                full = t0 - last_full >= 5.0              # the small Q-table pictures: every 5 s is enough
                if full:
                    last_full = t0
                st = self.status_fn(full_q=full)
                self._check(self.s.post(f"{self.url}/api/status", json=st, timeout=2.0))
                if self.qtable_fn and time.time() - last_q >= self.qtable_every:
                    last_q = time.time()
                    self.upload_qtable()
            except Exception:
                pass
            # wait for the next second, or less if nudge() is called (but never more than 10 times a second)
            self._wake.wait(max(0.0, self.every - (time.time() - t0)))
            self._wake.clear()
            time.sleep(max(0.0, 0.1 - (time.time() - t0)))

    def _control_loop(self):
        """Waits on the server for the next button press (long polling). An older server that answers at once
        is simply asked once a second, as before."""
        after = None
        while not self._stop:
            try:
                q = "" if after is None else f"?after={after}&wait=20"
                r = self._check(self.cs.get(f"{self.url}/api/control{q}", timeout=25.0))
                if not r.ok:
                    time.sleep(1.0)
                    continue
                ctl = r.json()
                cver = ctl.pop("cver", None)
                if ctl != self._last_control:
                    self._last_control = ctl
                    try:
                        self.on_control(ctl)
                    except Exception as e:
                        print(f"[reporter] control handler error: {e}")
                waited = r.headers.get("X-Long-Poll") == "1"
                if after is not None and not waited:
                    time.sleep(self.every)                 # no long polling on this server: once a second
                after = cver
            except Exception:
                time.sleep(1.0)                            # server not reachable: try again in a second

    def stop(self):
        self._stop = True
