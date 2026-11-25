"""
Box-event storage on the cloud server.

  * every finished box from the gateway is written to SQLite (survives a server restart)
  * the dashboard numbers come from RUNNING COUNTERS, updated once per event (O(1)),
    instead of re-scanning thousands of stored events on every dashboard refresh
  * "boxes in the last minute" is a sliding-window counter (deque)
  * "Reset counters" starts a new session; old sessions stay in the database
"""
from __future__ import annotations

import csv
import io
import os
import sqlite3
import threading
import time
from collections import deque

from common.algos import SlidingWindowCounter

COLUMNS = ["box_id", "t", "policy", "action", "on_time", "latency_ms", "verdict", "p_damaged", "rejected",
           "truth", "rtt_ms", "cloud_ms", "budget_ms", "slack_ms", "cpu", "reward", "state", "error",
           # second opinion: gateway's model went first / how sure it was / cloud asked / cloud late / verdict changed
           "first", "conf", "asked", "cloud_late", "changed"]
EXPORT_COLUMNS = ["session", "device"] + COLUMNS
_COL_TYPES = {"box_id": "INTEGER", "on_time": "INTEGER", "rejected": "INTEGER", "state": "INTEGER",
              "first": "INTEGER", "conf": "INTEGER", "asked": "INTEGER", "cloud_late": "INTEGER", "changed": "INTEGER",
              "policy": "TEXT", "action": "TEXT", "verdict": "TEXT", "truth": "TEXT", "error": "TEXT"}


class LiveStats:
    """Running totals of a stream of box events. add() is O(1)."""

    __slots__ = ("n", "edge", "gateway", "cloud", "missed", "rejected", "known", "correct", "damaged_known", "escaped",
                 "lat_sum", "lat_n", "reward_sum", "first", "asked", "cloud_late", "changed")

    def __init__(self):
        self.clear()

    def clear(self):
        self.n = self.edge = self.gateway = self.cloud = self.missed = self.rejected = 0
        self.known = self.correct = self.damaged_known = self.escaped = 0
        self.first = self.asked = self.cloud_late = self.changed = 0
        self.lat_sum, self.lat_n, self.reward_sum = 0.0, 0, 0.0

    def add(self, e: dict):
        self.n += 1
        if e.get("action") == "EDGE":             # checked on the ESP32
            self.edge += 1
        elif e.get("action") == "GATEWAY":
            self.gateway += 1
        elif e.get("action") == "CLOUD":
            self.cloud += 1                       # "NONE" = no photo (camera error): neither
        on_time = bool(e.get("on_time"))
        if not on_time:
            self.missed += 1
        elif e.get("latency_ms") is not None:
            self.lat_sum += float(e["latency_ms"])
            self.lat_n += 1
        if e.get("rejected"):
            self.rejected += 1
        truth = e.get("truth")
        if truth in ("good", "damaged"):
            self.known += 1
            if e.get("verdict") == truth:
                self.correct += 1
            if truth == "damaged":
                self.damaged_known += 1
                if e.get("verdict") != "damaged":
                    self.escaped += 1
        self.reward_sum += float(e.get("reward") or 0.0)
        if e.get("first"):                         # the gateway's model checked this box first
            self.first += 1
        if e.get("asked"):                         # ... and the cloud was asked for a second opinion
            self.asked += 1
            if e.get("cloud_late"):
                self.cloud_late += 1
            elif e.get("changed"):
                self.changed += 1

    def snapshot(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    def summary(self, per_min: int = 0) -> dict:
        if not self.n:
            return {"n": 0}
        return {"n": self.n, "edge": self.edge, "gateway": self.gateway, "cloud": self.cloud, "missed": self.missed,
                "per_min": per_min, "rejected": self.rejected, "known": self.known, "correct": self.correct,
                "damaged_known": self.damaged_known, "escaped": self.escaped,
                "first": self.first, "asked": self.asked, "cloud_late": self.cloud_late, "changed": self.changed,
                "avg_latency_ms": round(self.lat_sum / self.lat_n, 1) if self.lat_n else None,
                "avg_reward": round(self.reward_sum / self.n, 3)}


def _clean(e: dict) -> dict:
    """Event as the dashboard shows it (images removed from the stored copy)."""
    return {k: v for k, v in e.items() if k not in ("heatmap",)}


class EventStore:
    def __init__(self, db_path: str | None, keep: int = 5000):
        self.lock = threading.Lock()
        self.recent: deque = deque(maxlen=keep)       # newest events for the dashboard graphs
        self.last100: deque = deque(maxlen=100)
        self.stats = LiveStats()
        self.rate = SlidingWindowCounter(60.0)
        self.db = None
        self.session = 1
        if db_path:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            self.db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            cols = ", ".join(f"{c} {_COL_TYPES.get(c, 'REAL')}" for c in COLUMNS)
            self.db.execute(f"CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                            f"session INTEGER, received REAL, device TEXT, {cols})")
            have = {r[1] for r in self.db.execute("PRAGMA table_info(events)")}
            for c in COLUMNS:                      # a database made by an older version: add the new columns
                if c not in have:
                    self.db.execute(f"ALTER TABLE events ADD COLUMN {c} {_COL_TYPES.get(c, 'REAL')}")
            self.db.execute("CREATE INDEX IF NOT EXISTS events_session ON events(session)")
            self.db.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
            row = self.db.execute("SELECT v FROM meta WHERE k='session'").fetchone()
            self.session = int(row[0]) if row else 1
            self._set_meta("session", self.session)
            self._reload_session()

    # ------------------------------------------------------------------
    def _set_meta(self, k, v):
        self.db.execute("INSERT INTO meta(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, str(v)))

    def _reload_session(self):
        """After a restart: rebuild the counters of the current session from the database (one pass)."""
        cur = self.db.execute(f"SELECT {', '.join(COLUMNS)} FROM events WHERE session=? ORDER BY id", (self.session,))
        for row in cur:
            e = dict(zip(COLUMNS, row))
            e["on_time"] = bool(e["on_time"])
            e["rejected"] = bool(e["rejected"])
            self.stats.add(e)
            self.recent.append(e)
            self.last100.append(e)

    # ------------------------------------------------------------------
    def add_many(self, evs: list[dict], device: str = "pi-1") -> int:
        now = time.time()
        rows = []
        with self.lock:
            for e in evs:
                if not isinstance(e, dict):
                    continue
                self.stats.add(e)
                ce = _clean(e)
                self.recent.append(ce)
                self.last100.append(ce)
                self.rate.add(now)
                if self.db is not None:
                    rows.append([self.session, now, device] + [_sql(e.get(c)) for c in COLUMNS])
            if rows:
                ph = ", ".join("?" * len(rows[0]))
                self.db.execute("BEGIN")
                self.db.executemany(f"INSERT INTO events (session, received, device, {', '.join(COLUMNS)}) "
                                    f"VALUES ({ph})", rows)
                self.db.execute("COMMIT")
        return len(evs)

    def reset(self):
        with self.lock:
            self.stats.clear()
            self.recent.clear()
            self.last100.clear()
            self.rate.clear()
            self.session += 1
            if self.db is not None:
                self._set_meta("session", self.session)

    # ------------------------------------------------------------------
    def summaries(self) -> tuple[dict, dict]:
        with self.lock:
            per_min = self.rate.count()
            h = LiveStats()                      # last 100 boxes: bounded, constant work
            for e in self.last100:
                h.add(e)
            return self.stats.summary(per_min), h.summary(per_min)

    def tail(self, n: int) -> list[dict]:
        with self.lock:
            if n >= len(self.recent):
                return list(self.recent)
            # deque has no slicing; walk from the right end only (O(n), not O(len))
            out = [None] * n
            it = reversed(self.recent)
            for i in range(n - 1, -1, -1):
                out[i] = next(it)
            return out

    def export_csv(self, session: str | int | None = None) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(EXPORT_COLUMNS)
        if self.db is None:
            for e in self.tail(len(self.recent)):
                w.writerow([self.session, ""] + [e.get(c) for c in COLUMNS])
            return buf.getvalue()
        with self.lock:
            if session == "all":
                cur = self.db.execute(f"SELECT {', '.join(EXPORT_COLUMNS)} FROM events ORDER BY id")
            else:
                s = self.session if session in (None, "", "current") else int(session)
                cur = self.db.execute(f"SELECT {', '.join(EXPORT_COLUMNS)} FROM events WHERE session=? ORDER BY id", (s,))
            for row in cur:
                w.writerow(row)
        return buf.getvalue()

    def count_rows(self) -> int:
        if self.db is None:
            return len(self.recent)
        with self.lock:
            return int(self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def close(self):
        if self.db is not None:
            with self.lock:
                self.db.close()
                self.db = None


def _sql(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float, str)) or v is None:
        return v
    return str(v)
