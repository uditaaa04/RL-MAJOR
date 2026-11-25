"""
AWS integration of the cloud server (optional - the server runs fine without it).

When the server runs on EC2 with the IAM role from cloud/aws/cloudformation.yaml:

  * CloudWatch metrics, every 60 s, namespace "SmartConveyor":
        BoxesPerMinute, MissRatePct, CloudSharePct, AvgLatencyMs, ServerInferMs, ProbeRequests
    -> graphs and alarms in the AWS console (e.g. alarm when MissRatePct > 5)
  * S3 backups, every 5 min, to s3://<bucket>/<prefix>/:
        events/session-<n>.csv           all boxes of the current session
        qtables/<device>/latest.npz      the gateway's online-learned Q-table

Turned on by environment variables (set by the EC2 bootstrap script):
    CONVEYOR_S3_BUCKET   bucket name            (no bucket -> no S3 uploads)
    CONVEYOR_S3_PREFIX   key prefix             (default "smart-conveyor")
    CONVEYOR_METRICS     1 = publish CloudWatch metrics
    AWS_REGION           e.g. ap-south-1

Metrics are computed from the DIFFERENCE of the running counters between two
ticks, so each tick is O(1) no matter how many boxes have been seen.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time


class CloudSync:
    def __init__(self, store, registry, server_counters, line_name: str = "line-1",
                 metrics_every_s: float = 60.0, upload_every_s: float = 300.0):
        self.store, self.registry, self.server_counters = store, registry, server_counters
        self.line = line_name
        self.bucket = os.environ.get("CONVEYOR_S3_BUCKET", "").strip()
        self.prefix = os.environ.get("CONVEYOR_S3_PREFIX", "smart-conveyor").strip("/")
        self.metrics_on = os.environ.get("CONVEYOR_METRICS", "0") == "1"
        self.region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        self.metrics_every, self.upload_every = metrics_every_s, upload_every_s
        self.s3 = self.cw = None
        self.status = {"s3": False, "cloudwatch": False, "last_upload": None, "last_metrics": None, "error": None}
        self._uploaded: dict[str, str] = {}          # s3 key -> md5 of last upload (skip unchanged files)
        self._prev = None
        self._stop = threading.Event()
        if not (self.bucket or self.metrics_on):
            return
        try:
            import boto3  # only needed on AWS
            kw = {"region_name": self.region} if self.region else {}
            if self.bucket:
                self.s3 = boto3.client("s3", **kw)
                self.status["s3"] = True
            if self.metrics_on:
                self.cw = boto3.client("cloudwatch", **kw)
                self.status["cloudwatch"] = True
        except Exception as e:      # boto3 missing or no credentials: keep serving, just report it
            self.status["error"] = f"{type(e).__name__}: {e}"[:200]
            print(f"[cloud] AWS integration disabled: {self.status['error']}")

    @property
    def enabled(self) -> bool:
        return self.s3 is not None or self.cw is not None

    def start(self):
        if self.enabled:
            threading.Thread(target=self._loop, daemon=True, name="cloud-sync").start()
            print(f"[cloud] AWS sync on: s3={'s3://' + self.bucket + '/' + self.prefix if self.s3 else 'off'} "
                  f"cloudwatch={'on' if self.cw else 'off'}")

    def stop(self):
        self._stop.set()

    # ------------------------------------------------------------------
    def _loop(self):
        next_m = time.time() + self.metrics_every
        next_u = time.time() + 30.0                 # first backup soon after start
        while not self._stop.wait(1.0):
            now = time.time()
            if self.cw is not None and now >= next_m:
                next_m = now + self.metrics_every
                self._safe(self.publish_metrics)
            if self.s3 is not None and now >= next_u:
                next_u = now + self.upload_every
                self._safe(self.upload_backups)

    def _safe(self, fn):
        try:
            fn()
            self.status["error"] = None
        except Exception as e:
            self.status["error"] = f"{fn.__name__}: {type(e).__name__}: {e}"[:200]
            print(f"[cloud] {self.status['error']}")

    # ------------------------------------------------------------------ CloudWatch
    def metric_values(self) -> dict:
        """Per-interval numbers from counter differences (O(1))."""
        cur = {**self.store.stats.snapshot(), **self.server_counters()}
        prev, self._prev = self._prev, cur
        if prev is None or cur["n"] < prev["n"]:        # first tick, or counters were reset
            prev = {k: 0 for k in cur}
        d = {k: cur[k] - prev.get(k, 0) for k in cur}
        minutes = self.metrics_every / 60.0
        out = {"BoxesPerMinute": d["n"] / minutes, "ProbeRequests": d.get("probes", 0)}
        if d["n"] > 0:
            out["MissRatePct"] = 100.0 * d["missed"] / d["n"]
            out["CloudSharePct"] = 100.0 * d["cloud"] / d["n"]
            out["EdgeSharePct"] = 100.0 * d["edge"] / d["n"]
        if d["lat_n"] > 0:
            out["AvgLatencyMs"] = d["lat_sum"] / d["lat_n"]
        if d.get("served", 0) > 0:
            out["ServerInferMs"] = d["server_ms_sum"] / d["served"]
        return out

    def publish_metrics(self):
        vals = self.metric_values()
        units = {"BoxesPerMinute": "Count", "ProbeRequests": "Count", "MissRatePct": "Percent",
                 "CloudSharePct": "Percent", "EdgeSharePct": "Percent", "AvgLatencyMs": "Milliseconds",
                 "ServerInferMs": "Milliseconds"}
        data = [{"MetricName": k, "Value": float(v), "Unit": units[k],
                 "Dimensions": [{"Name": "Line", "Value": self.line}]} for k, v in vals.items()]
        if data:
            self.cw.put_metric_data(Namespace="SmartConveyor", MetricData=data)
        self.status["last_metrics"] = time.strftime("%H:%M:%S")

    # ------------------------------------------------------------------ S3
    def _put(self, key: str, body: bytes, content_type: str):
        md5 = hashlib.md5(body, usedforsecurity=False).hexdigest()     # change detection only
        if self._uploaded.get(key) == md5:
            return False                                     # unchanged since the last upload
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType=content_type,
                           ServerSideEncryption="AES256")
        self._uploaded[key] = md5
        return True

    def upload_backups(self):
        n = 0
        csv_body = self.store.export_csv().encode("utf-8")
        n += self._put(f"{self.prefix}/events/session-{self.store.session}.csv", csv_body, "text/csv")
        for p in self.registry.backups():
            device = os.path.basename(os.path.dirname(p))
            with open(p, "rb") as f:
                n += self._put(f"{self.prefix}/qtables/{device}/latest.npz", f.read(), "application/octet-stream")
        self.status["last_upload"] = time.strftime("%H:%M:%S") + f" ({n} changed)"
