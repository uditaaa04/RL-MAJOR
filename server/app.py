"""
Cloud inference server + live dashboard.  This is the CLOUD node. Runs on AWS EC2 or Render (or on the gateway laptop itself for a local demo).

    python -m server.app            (then open http://<server>:8000 in a browser)

Endpoints                                                     key needed?
  GET  /                 dashboard page                             no (the page asks for the key)
  GET  /ping             tiny reply - the gateway measures round-trip    no
  GET  /health           model loaded, uptime, AWS sync status      no
  POST /infer            JPEG in -> {p_damaged, label, server_ms, heatmap}   (the big model)
  POST /api/events       finished boxes from the gateway (batched)
  POST /api/status       gateway status every second (network numbers, Q-table snapshot ...)
  POST /api/edge         the ESP32's own readings over its Wi-Fi (firmware 4): sensors, belt, arm, Wi-Fi
  GET  /api/state        everything the dashboard shows
  GET/POST /api/control  dashboard buttons: network scenario, server overload, gateway overload, belt, policy,
                         manual tests (belt on/off, push the arm, check a box now)
  POST /api/reset        start a new session (counters to zero; old data kept in the database)
  GET  /api/export.csv   box events as CSV  (?session=current|all|<n>)
  GET  /api/models       model manifest (sha256) for the gateway's sync tool
  GET  /api/models/<f>   download a model file
  POST /api/qtable       gateway uploads its online-learned Q-table (backup)
  GET  /api/qtable/latest

Live updates (long polling): GET /api/state?after=<ver>&wait=<s> and GET /api/control?after=<cver>&wait=<s>
answer as soon as something changes (a finished box, a gateway status, a button) instead of on a timer, so
the dashboard and the gateway hear about it one network trip later. Without "after" they answer at once.

Security: set the API key with the environment variable CONVEYOR_API_KEY (the EC2
bootstrap does this). Every request except /, /ping, /health and /static must then
carry the header  X-API-Key: <key>.  Without a key the server is open - fine on a
private lab network, never on a public EC2 address.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import os
import random
import threading
import time

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

from common.config import load_config, project_path
from common.vision_io import OnnxInspector, from_jpeg, to_jpeg

from .cloud_sync import CloudSync
from .registry import ModelRegistry
from .store import EventStore

HERE = os.path.dirname(os.path.abspath(__file__))
OPEN_PATHS = {"/", "/ping", "/health", "/favicon.ico"}
CONTROL_KEYS = ("scenario", "server_slow", "gw_stress", "belt_duty", "policy", "cloud_per_min",
                "belt_on", "belt_n", "arm_test", "check_now")
SCENARIOS = {"good", "busy", "bad", "bwcap", "lossy", "stalls", "down", "real"}
POLICIES = {"qlearning", "confidence", "confidence_reserve", "ask_always",
            "adaptive", "deadline", "threshold", "always_gateway", "always_cloud", "always_edge"}


def api_key_from(cfg: dict) -> str:
    return (os.environ.get("CONVEYOR_API_KEY") or str(cfg.get("server", {}).get("api_key") or "")).strip()


def _valid_control(upd: dict) -> dict:
    """Accept only known keys and sane values from the dashboard."""
    out = {}
    if upd.get("scenario") in SCENARIOS:
        out["scenario"] = upd["scenario"]
    for k in ("server_slow", "gw_stress"):
        if k in upd:
            out[k] = bool(upd[k])
    if "belt_duty" in upd:
        try:
            out["belt_duty"] = min(1.0, max(0.05, float(upd["belt_duty"])))
        except (TypeError, ValueError):
            pass
    if upd.get("policy") in POLICIES:
        out["policy"] = upd["policy"]
    if "cloud_per_min" in upd:                     # cloud calls allowed per minute; 0 = no limit
        try:
            out["cloud_per_min"] = min(600.0, max(0.0, float(upd["cloud_per_min"])))
        except (TypeError, ValueError):
            pass
    if "belt_on" in upd:                           # manual tests: belt motor on / off
        out["belt_on"] = bool(upd["belt_on"])
    for k in ("arm_test", "check_now", "belt_n"):  # manual tests: a new number = do it once more
        if k in upd:
            try:
                out[k] = int(upd[k]) % 1000000000
            except (TypeError, ValueError):
                pass
    return out


def create_app(cfg: dict, model=None, data_dir: str | None = None, api_key: str | None = None,
               enable_cloud_sync: bool = True) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024          # no giant uploads
    key = api_key_from(cfg) if api_key is None else api_key
    data_dir = data_dir or os.environ.get("CONVEYOR_DATA_DIR") or project_path("cloud_data")
    os.makedirs(data_dir, exist_ok=True)

    store = EventStore(os.path.join(data_dir, "events.sqlite"))
    registry = ModelRegistry(project_path("models"), os.path.join(data_dir, "qtables"))
    S = {
        "model": model, "model_lock": threading.Lock(), "info": {},
        # scenario None = no network condition requested yet: the node keeps whatever it started with
        # (a mock node talking to a real remote server must stay on the real network)
        "control": {"scenario": None, "server_slow": False, "gw_stress": False, "belt_duty": None, "policy": None,
                    "cloud_per_min": None, "belt_on": None, "belt_n": 0, "arm_test": 0, "check_now": 0},
        "ver": 0, "cver": 0,         # change counters for long polling (anything / control only)
        "edge": None, "edge_t": 0.0,  # last readings the ESP32 sent by itself over Wi-Fi
        "status": {}, "status_t": 0.0, "q": None, "visits": None,
        "served": 0, "probes": 0, "server_ms_sum": 0.0, "last_cloud": None,
        "lock": threading.Lock(), "started": time.time(),
    }
    if model is not None:
        S["info"] = {"model": getattr(model, "name", "model"), "input": getattr(model, "size", None),
                     "provider": getattr(model, "provider", None)}

    def server_counters():
        with S["lock"]:
            return {"served": S["served"], "probes": S["probes"], "server_ms_sum": S["server_ms_sum"]}

    # ------------------------------------------------------------------ live updates (long polling)
    # A waiting request holds one server thread, so only a few may wait at once; the rest answer at once.
    http_threads = int(os.environ.get("CONVEYOR_HTTP_THREADS") or cfg.get("server", {}).get("threads", 8))
    lp_max = int(os.environ.get("CONVEYOR_LONGPOLL_MAX") or max(1, min(6, http_threads - 3)))
    changed = threading.Condition()
    lp_slots = threading.BoundedSemaphore(lp_max)

    def bump(control: bool = False):
        """Something new: wake every request that is waiting for a change."""
        with changed:
            S["ver"] += 1
            if control:
                S["cver"] += 1
            changed.notify_all()

    def wait_for_change(counter: str) -> bool:
        """?after=<n>&wait=<s>: hold this request until S[counter] is no longer n (at most 25 s).
        Returns True if it really waited (long polling on), False if it answers at once."""
        after = request.args.get("after", type=int)
        wait = request.args.get("wait", default=0.0, type=float)
        if after is None or not wait or wait <= 0:
            return False
        if not lp_slots.acquire(blocking=False):
            return False                               # all waiting slots busy: answer at once
        try:
            with changed:
                changed.wait_for(lambda: S[counter] != after, timeout=min(float(wait), 25.0))
            return True
        finally:
            lp_slots.release()

    line = os.environ.get("CONVEYOR_LINE") or str(cfg.get("project", {}).get("line", "line-1"))
    sync = CloudSync(store, registry, server_counters, line_name=line) if enable_cloud_sync else None
    app.extensions["conveyor"] = {"S": S, "store": store, "registry": registry, "sync": sync, "auth": bool(key)}

    # ------------------------------------------------------------------ auth
    @app.before_request
    def _check_key():
        if not key or request.path in OPEN_PATHS or request.path.startswith("/static/"):
            return None
        got = request.headers.get("X-API-Key", "")
        if not got and request.path == "/api/export.csv":        # browser download link
            got = request.args.get("key", "")
        if not hmac.compare_digest(got.encode(), key.encode()):  # constant-time compare
            return jsonify({"error": "missing or wrong API key (header X-API-Key)"}), 401
        return None

    # ------------------------------------------------------------------ inference
    @app.post("/infer")
    def infer():
        data = request.get_data()
        if not data:
            return jsonify({"error": "send JPEG bytes"}), 400
        probe = request.args.get("probe") == "1"
        try:
            img = from_jpeg(data)
        except Exception:
            return jsonify({"error": "not a valid image"}), 400
        t = time.perf_counter()
        with S["model_lock"]:                 # one request at a time, like a real busy server
            p, heat, _ = S["model"].predict(img)
            if S["control"]["server_slow"]:   # dashboard "server overloaded" switch
                time.sleep(random.lognormvariate(-0.98, 0.25))   # ~0.375 s median extra (sim: 25x slower)
        ms = (time.perf_counter() - t) * 1000
        with S["lock"]:
            if probe:
                S["probes"] += 1
            else:
                S["served"] += 1
                S["server_ms_sum"] += ms
                S["last_cloud"] = {"jpeg": base64.b64encode(to_jpeg(img.resize((192, 192)), 75)).decode(),
                                   "heatmap": heat, "p": round(p, 3), "box": request.args.get("box"), "t": time.time()}
        return jsonify({"p_damaged": p, "label": "damaged" if p >= 0.5 else "good",
                        "server_ms": round(ms, 2), "heatmap": heat})

    @app.get("/ping")
    def ping():
        return jsonify({"ok": True, "t": time.time()})

    @app.get("/health")
    def health():
        return jsonify({"ok": S["model"] is not None, "model": S["info"].get("model"),
                        "uptime_s": round(time.time() - S["started"]), "auth": bool(key),
                        "events_stored": store.count_rows(), "session": store.session,
                        "where": os.environ.get("CONVEYOR_WHERE", "local"),
                        "aws": sync.status if sync else None})

    # ------------------------------------------------------------------ gateway -> server
    @app.post("/api/events")
    def events():
        evs = request.get_json(force=True, silent=True) or []
        if isinstance(evs, dict):
            evs = [evs]
        if not isinstance(evs, list):
            return jsonify({"error": "send a JSON list of events"}), 400
        n = store.add_many(evs[:1000], device=request.headers.get("X-Device-Id", "pi-1")[:40])
        bump()
        return jsonify({"ok": True, "n": n})

    @app.post("/api/status")
    def status():
        st = request.get_json(force=True, silent=True) or {}
        if not isinstance(st, dict):
            return jsonify({"error": "send a JSON object"}), 400
        with S["lock"]:
            if "q" in st:
                S["q"], S["visits"] = st.pop("q"), st.pop("visits", None)
            if "qmaps" in st:                      # small pictures of the learned table (sent every few seconds)
                S["qmaps"] = st.pop("qmaps")
            S["status"], S["status_t"] = st, time.time()
        bump()
        return jsonify({"ok": True})

    EDGE_KEYS = {"fw": int, "ir_camera": int, "ir_arm": int, "duty": int, "belt_on": int, "count_camera": int,
                 "count_arm": int, "pushes": int, "arm_out": int, "usb_link": int, "model": int, "uptime_ms": int,
                 "free_heap": int, "min_heap": int, "wifi": str, "ip": str, "rssi": int, "hotspot": int,
                 "hotspot_name": str, "cloud": str, "cloud_ms": int, "host": str, "ssid": str}

    @app.post("/api/edge")
    def edge():
        """The ESP32 itself (over its Wi-Fi): sensors, belt, arm, memory, Wi-Fi. Known fields only, short strings."""
        d = request.get_json(force=True, silent=True)
        if not isinstance(d, dict):
            return jsonify({"error": "send a JSON object"}), 400
        out = {}
        for k, typ in EDGE_KEYS.items():
            if k in d:
                try:
                    out[k] = typ(d[k]) if typ is int else str(d[k])[:64]
                except (TypeError, ValueError):
                    pass
        with S["lock"]:
            S["edge"], S["edge_t"] = out, time.time()
        bump()
        return jsonify({"ok": True})

    # ------------------------------------------------------------------ dashboard
    @app.route("/api/control", methods=["GET", "POST"])
    def control():
        lp = False
        if request.method == "POST":
            upd = _valid_control(request.get_json(force=True, silent=True) or {})
            with S["lock"]:
                S["control"].update(upd)
            bump(control=True)
        else:
            lp = wait_for_change("cver")             # the gateway waits here for the next button press
        with S["lock"]:
            out = dict(S["control"])
        out["cver"] = S["cver"]
        r = jsonify(out)
        r.headers["X-Long-Poll"] = "1" if lp else "0"
        return r

    @app.post("/api/reset")
    def reset():
        store.reset()
        with S["lock"]:
            S["served"] = S["probes"] = 0
            S["server_ms_sum"] = 0.0
        bump()
        return jsonify({"ok": True, "session": store.session})

    @app.get("/api/state")
    def state():
        lp = wait_for_change("ver")                  # the dashboard waits here for the next box or status
        ver = S["ver"]                               # read BEFORE the state: a change during the read is not lost
        all_s, last100 = store.summaries()          # running counters: O(1), not a re-scan
        tail = store.tail(300)
        recent = [dict(e) for e in tail[-14:]][::-1]
        for e in recent[12:]:
            e.pop("thumb", None)
        with S["lock"]:
            out = {
                "status": dict(S["status"]),
                "status_age_s": round(time.time() - S["status_t"], 1) if S["status_t"] else None,
                "control": dict(S["control"]), "all": all_s, "last100": last100, "recent": recent,
                "series": [{"i": e.get("box_id"), "a": e.get("action"), "ok": e.get("on_time"),
                            "rtt": e.get("rtt_ms"), "cl": e.get("cloud_ms"), "bud": e.get("budget_ms"),
                            "lat": e.get("latency_ms"), "r": e.get("reward"), "gw": e.get("gw_ms")} for e in tail],
                "q": S["q"], "visits": S["visits"], "qmaps": S.get("qmaps"),
                "server": {**S["info"], "served": S["served"], "probes": S["probes"],
                           "avg_ms": round(S["server_ms_sum"] / S["served"], 1) if S["served"] else None,
                           "uptime_s": round(time.time() - S["started"]), "session": store.session,
                           "where": os.environ.get("CONVEYOR_WHERE", "local"),
                           "aws": sync.status if sync else None},
                "last_cloud": S["last_cloud"],
                "edge_wifi": dict(S["edge"], age_s=round(time.time() - S["edge_t"], 1)) if S["edge"] else None,
                "ver": ver, "lp": lp, "server_t": time.time(),
            }
        return jsonify(out)

    @app.get("/api/export.csv")
    def export_csv():
        sess = request.args.get("session", "current")
        if sess not in ("current", "all") and not sess.isdigit():
            return jsonify({"error": "session must be current, all or a number"}), 400
        body = store.export_csv(sess)
        return Response(body, mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=boxes_session-{sess}.csv"})

    @app.get("/api/results")
    def results():
        """Experiment evidence for the dashboard's Results page (read from results/ and the model card)."""
        from .results import load_results
        return jsonify(load_results())

    # ------------------------------------------------------------------ model registry
    @app.get("/api/models")
    def models_manifest():
        return jsonify(registry.manifest())

    @app.get("/api/models/<name>")
    def models_file(name):
        p = registry.path(name)
        if p is None:
            abort(404)
        return send_file(p, as_attachment=True, download_name=name)

    @app.post("/api/qtable")
    def qtable_upload():
        device = request.headers.get("X-Device-Id", "pi-1")
        try:
            info = registry.save_qtable(device, request.get_data())
        except Exception as e:
            return jsonify({"error": f"rejected: {e}"[:200]}), 400
        return jsonify({"ok": True, **info})

    @app.get("/api/qtable/latest")
    def qtable_latest():
        p = registry.latest_qtable(request.args.get("device", request.headers.get("X-Device-Id", "pi-1")))
        if p is None:
            abort(404)
        return send_file(p, as_attachment=True, download_name="q_table_online.npz")

    # ------------------------------------------------------------------ static
    @app.get("/")
    def index():
        return send_from_directory(os.path.join(HERE, "dashboard"), "index.html")

    @app.get("/static/<path:p>")
    def static_files(p):
        return send_from_directory(os.path.join(HERE, "dashboard"), p)

    return app


def load_model(cfg: dict, cpu_only: bool = False):
    path = project_path(cfg["models"]["cloud_onnx"])
    prov = ["CPUExecutionProvider"] if cpu_only else None
    # CONVEYOR_ONNX_THREADS: 0 = use every core (EC2, laptop); 1 on small shared hosts such as Render Free (0.1 CPU),
    # where many threads only fight over the tiny CPU share and make every answer slower
    model = OnnxInspector(path, threads=int(os.environ.get("CONVEYOR_ONNX_THREADS") or 0), providers=prov)
    model.name = os.path.basename(path)
    from PIL import Image
    for _ in range(3):                                   # warm-up: the first ONNX runs are slow
        model.predict(Image.new("RGB", (256, 256)))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--cpu-only", action="store_true", help="do not use the GPU even if onnxruntime-gpu is installed")
    a = ap.parse_args()
    cfg = load_config(a.config)
    model = load_model(cfg, a.cpu_only)
    app = create_app(cfg, model)
    ext = app.extensions["conveyor"]
    host = a.host or cfg["server"]["host"]
    port = a.port or int(os.environ.get("CONVEYOR_PORT") or os.environ.get("PORT") or cfg["server"]["port"])  # PORT: set by Render
    print(f"[server] big model {ext['S']['info']}  ->  dashboard: http://localhost:{port}")
    if ext["auth"]:
        print("[server] API key required (X-API-Key)")
    elif host not in ("127.0.0.1", "localhost"):
        print("[server] WARNING: no API key set (CONVEYOR_API_KEY) and listening on all interfaces - "
              "anyone who can reach this port can use the controls. Fine on a private lab network only.")
    if ext["sync"] is not None:
        ext["sync"].start()
    try:
        from waitress import serve
        threads = int(os.environ.get("CONVEYOR_HTTP_THREADS") or cfg["server"]["threads"])
        serve(app, host=host, port=port, threads=threads, ident=None)
    except ImportError:
        app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
