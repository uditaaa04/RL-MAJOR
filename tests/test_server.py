"""Cloud server: API key, storage, running counters, model registry, Q-table backups.
Run:  python -m pytest -q tests"""
import io
import os
import random
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import load_config  # noqa: E402
from common.rl import QAgent  # noqa: E402
from server.app import create_app  # noqa: E402
from server.store import EventStore  # noqa: E402

KEY = "test-key-123"


class FakeModel:
    name, size, provider = "fake.onnx", 160, "CPU"

    def predict(self, img):
        return 0.7, [[0.1, 0.9], [0.0, 0.2]], 0.001


@pytest.fixture()
def client(tmp_path):
    app = create_app(load_config(), FakeModel(), data_dir=str(tmp_path), api_key=KEY, enable_cloud_sync=False)
    app.testing = True
    return app.test_client()


def H(**kw):
    return {"X-API-Key": KEY, **kw}


def jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 150, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def fake_event(i, rng):
    truth = rng.choice(["good", "damaged", None])
    on_time = rng.random() > 0.1
    return {"box_id": i, "action": rng.choice(["GATEWAY", "CLOUD"]), "on_time": on_time,
            "latency_ms": rng.uniform(5, 300), "verdict": rng.choice(["good", "damaged"]) if on_time else "unchecked",
            "rejected": rng.random() < 0.3, "truth": truth, "reward": -rng.uniform(1, 40),
            "rtt_ms": 30.0, "cloud_ms": 60.0, "budget_ms": 400, "slack_ms": 300, "cpu": 0.2, "state": 5,
            "policy": "Q-learning", "thumb": "xx", "heatmap": [[0.1]], "t": 1.0}


def naive_summary(evs):
    """The server's OLD dashboard summary: a full re-scan of every event."""
    n = len(evs)
    edge = sum(1 for e in evs if e.get("action") == "EDGE")
    gateway = sum(1 for e in evs if e.get("action") == "GATEWAY")
    missed = sum(1 for e in evs if not e.get("on_time"))
    known = [e for e in evs if e.get("truth") in ("good", "damaged")]
    correct = sum(1 for e in known if e.get("verdict") == e.get("truth"))
    dam = [e for e in known if e.get("truth") == "damaged"]
    escaped = sum(1 for e in dam if e.get("verdict") != "damaged")
    lat = [e["latency_ms"] for e in evs if e.get("on_time")]
    return {"n": n, "edge": edge, "gateway": gateway, "cloud": n - edge - gateway, "missed": missed,
            "rejected": sum(1 for e in evs if e.get("rejected")), "known": len(known), "correct": correct,
            "damaged_known": len(dam), "escaped": escaped,
            "first": sum(1 for e in evs if e.get("first")), "asked": sum(1 for e in evs if e.get("asked")),
            "cloud_late": sum(1 for e in evs if e.get("asked") and e.get("cloud_late")),
            "changed": sum(1 for e in evs if e.get("asked") and not e.get("cloud_late") and e.get("changed")),
            "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
            "avg_reward": round(sum(e.get("reward", 0) for e in evs) / n, 3)}


# ---------------------------------------------------------------- auth
def test_open_and_protected_paths(client):
    assert client.get("/ping").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
    for path in ("/api/state", "/api/control", "/api/models", "/api/export.csv"):
        assert client.get(path).status_code == 401, path
        assert client.get(path, headers=H()).status_code == 200, path
    assert client.get("/api/state", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/infer", data=jpeg()).status_code == 401
    # browser download link may carry the key as ?key=
    assert client.get(f"/api/export.csv?key={KEY}").status_code == 200


def test_infer(client):
    r = client.post("/infer?box=3", data=jpeg(), headers=H(**{"Content-Type": "image/jpeg"}))
    assert r.status_code == 200
    j = r.get_json()
    assert j["label"] == "damaged" and abs(j["p_damaged"] - 0.7) < 1e-9
    assert client.post("/infer", data=b"not a jpeg", headers=H()).status_code == 400
    st = client.get("/api/state", headers=H()).get_json()
    assert st["server"]["served"] == 1 and st["last_cloud"]["box"] == "3"


# ---------------------------------------------------------------- events + counters
def test_running_counters_equal_full_rescan(client):
    rng = random.Random(1)
    evs = [fake_event(i, rng) for i in range(700)]
    for k in range(0, 700, 50):
        assert client.post("/api/events", json=evs[k:k + 50], headers=H()).status_code == 200
    st = client.get("/api/state", headers=H()).get_json()
    got = {k: v for k, v in st["all"].items() if k != "per_min"}
    assert got == naive_summary(evs)
    got100 = {k: v for k, v in st["last100"].items() if k != "per_min"}
    assert got100 == naive_summary(evs[-100:])
    assert st["all"]["per_min"] == 700
    assert len(st["series"]) == 300 and st["series"][-1]["i"] == 699
    assert st["recent"][0]["box_id"] == 699 and "heatmap" not in st["recent"][0]


def test_events_survive_restart_and_reset_starts_new_session(tmp_path):
    rng = random.Random(2)
    db = os.path.join(tmp_path, "e.sqlite")
    s = EventStore(db)
    s.add_many([fake_event(i, rng) for i in range(40)])
    s.close()
    s2 = EventStore(db)                                  # server restarted
    assert s2.stats.n == 40 and len(s2.recent) == 40
    s2.reset()
    assert s2.stats.n == 0 and s2.session == 2
    s2.add_many([fake_event(i, rng) for i in range(5)])
    assert s2.export_csv().count("\n") == 1 + 5          # header + current session
    assert s2.export_csv("all").count("\n") == 1 + 45
    assert s2.count_rows() == 45


# ---------------------------------------------------------------- control
def test_control_validation(client):
    assert client.get("/api/control", headers=H()).get_json()["scenario"] is None   # nothing requested yet
    r = client.post("/api/control", json={"scenario": "bad", "server_slow": 1, "belt_duty": 7,
                                          "policy": "rm -rf", "evil": "x"}, headers=H())
    c = r.get_json()
    assert c["scenario"] == "bad" and c["server_slow"] is True and c["belt_duty"] == 1.0
    assert c["policy"] is None and "evil" not in c


# ---------------------------------------------------------------- model registry
def test_model_manifest_and_whitelist(client):
    m = client.get("/api/models", headers=H()).get_json()["files"]
    assert "q_table.npz" in m and "gateway_model.onnx" in m and len(m["q_table.npz"]["sha256"]) == 64
    assert "cloud_model.onnx" not in m                      # the big model never leaves the server
    assert client.get("/api/models/q_table.npz", headers=H()).status_code == 200
    for bad in ("cloud_model.onnx", "..%2Fconfig%2Fsystem.yaml", "system.yaml"):
        assert client.get(f"/api/models/{bad}", headers=H()).status_code == 404


def test_qtable_backup_roundtrip(client):
    ag = QAgent(432, 2)
    ag.q[:] = np.arange(864).reshape(432, 2)
    data = ag.to_bytes({"online": True})
    r = client.post("/api/qtable", data=data, headers=H(**{"X-Device-Id": "pi-7"}))
    assert r.status_code == 200 and r.get_json()["shape"] == [432, 2]
    back = client.get("/api/qtable/latest?device=pi-7", headers=H())
    assert back.status_code == 200
    z = np.load(io.BytesIO(back.data), allow_pickle=False)
    assert np.array_equal(z["q"], ag.q)
    assert client.post("/api/qtable", data=b"garbage", headers=H()).status_code == 400
    assert client.post("/api/qtable", data=data, headers=H(**{"X-Device-Id": "../etc"})).status_code == 400
    assert client.get("/api/qtable/latest?device=nobody", headers=H()).status_code == 404


def test_no_key_configured_means_open(tmp_path):
    app = create_app(load_config(), FakeModel(), data_dir=str(tmp_path), api_key="", enable_cloud_sync=False)
    c = app.test_client()
    assert c.get("/api/state").status_code == 200


def test_results_endpoint_needs_key_and_returns_tables(client):
    assert client.get("/api/results").status_code == 401
    r = client.get("/api/results", headers=H())
    assert r.status_code == 200
    d = r.get_json()
    assert d["simulated"] is True
    assert "models" in d and set(d["models"]) <= {"edge", "gateway", "cloud"}
    if "e1" in d:                                  # shipped with the project
        assert {"scenario", "policy", "reward_mean", "miss_rate"} <= set(d["e1"][0])
