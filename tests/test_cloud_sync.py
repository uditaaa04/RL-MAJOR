"""CloudWatch metrics + S3 backups, tested with fake AWS clients (no AWS account needed).
Run:  python -m pytest -q tests"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.rl import QAgent  # noqa: E402
from server.cloud_sync import CloudSync  # noqa: E402
from server.registry import ModelRegistry  # noqa: E402
from server.store import EventStore  # noqa: E402


class FakeCW:
    def __init__(self):
        self.calls = []

    def put_metric_data(self, Namespace, MetricData):
        self.calls.append((Namespace, MetricData))


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType, ServerSideEncryption):
        assert ServerSideEncryption == "AES256"
        self.objects[(Bucket, Key)] = Body


def ev(action, on_time, lat=50.0):
    return {"box_id": 0, "action": action, "on_time": on_time, "latency_ms": lat, "reward": -2.0}


def make(tmp_path, counters):
    store = EventStore(os.path.join(tmp_path, "e.sqlite"))
    reg = ModelRegistry(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
                        os.path.join(tmp_path, "qtables"))
    sync = CloudSync(store, reg, lambda: dict(counters), line_name="line-test", metrics_every_s=60)
    sync.cw, sync.s3, sync.bucket, sync.prefix = FakeCW(), FakeS3(), "bkt", "smart-conveyor/data"
    return store, reg, sync


def test_metrics_are_per_interval_differences(tmp_path):
    counters = {"served": 0, "probes": 0, "server_ms_sum": 0.0}
    store, _, sync = make(tmp_path, counters)
    store.add_many([ev("CLOUD", True, 40)] * 8 + [ev("GATEWAY", True, 10)] * 2)
    counters.update(served=8, probes=30, server_ms_sum=80.0)
    sync.publish_metrics()
    ns, data = sync.cw.calls[-1]
    vals = {d["MetricName"]: d["Value"] for d in data}
    assert ns == "SmartConveyor" and data[0]["Dimensions"] == [{"Name": "Line", "Value": "line-test"}]
    assert vals["BoxesPerMinute"] == 10 and vals["CloudSharePct"] == 80 and vals["MissRatePct"] == 0
    assert abs(vals["AvgLatencyMs"] - 34.0) < 1e-9 and vals["ServerInferMs"] == 10.0
    # next minute: only the NEW boxes count
    store.add_many([ev("GATEWAY", False)] * 4 + [ev("GATEWAY", True, 20)] * 1)
    sync.publish_metrics()
    vals = {d["MetricName"]: d["Value"] for d in sync.cw.calls[-1][1]}
    assert vals["BoxesPerMinute"] == 5 and vals["MissRatePct"] == 80 and vals["CloudSharePct"] == 0
    assert vals["AvgLatencyMs"] == 20 and "ServerInferMs" not in vals


def test_s3_backup_uploads_only_changed_files(tmp_path):
    store, reg, sync = make(tmp_path, {"served": 0, "probes": 0, "server_ms_sum": 0.0})
    store.add_many([ev("CLOUD", True)] * 3)
    reg.save_qtable("pi-1", QAgent(432, 2).to_bytes())
    sync.upload_backups()
    keys = {k for _, k in sync.s3.objects}
    assert keys == {"smart-conveyor/data/events/session-1.csv", "smart-conveyor/data/qtables/pi-1/latest.npz"}
    assert sync.status["last_upload"].endswith("(2 changed)")
    sync.upload_backups()                                   # nothing new -> nothing re-uploaded
    assert sync.status["last_upload"].endswith("(0 changed)")
    store.add_many([ev("GATEWAY", True)])
    sync.upload_backups()
    assert sync.status["last_upload"].endswith("(1 changed)")
    body = sync.s3.objects[("bkt", "smart-conveyor/data/events/session-1.csv")].decode()
    assert body.count("\n") == 1 + 4


def test_disabled_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("CONVEYOR_S3_BUCKET", raising=False)
    monkeypatch.delenv("CONVEYOR_METRICS", raising=False)
    store = EventStore(None)
    sync = CloudSync(store, None, lambda: {})
    assert not sync.enabled
    sync.start()                                            # no thread, no error
    assert sync.status["s3"] is False and sync.status["cloudwatch"] is False
