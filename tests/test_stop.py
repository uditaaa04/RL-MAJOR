"""The launcher's polite stop: Windows can only kill a program, and a killed gateway cannot switch the belt off
or save what the agent learned. The launcher writes a file, the gateway sees it and stops by itself.

    python -m pytest -q tests/test_stop.py
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_gateway_stops_by_itself_when_the_stop_file_appears(tmp_path):
    feed = os.path.join(ROOT, "data", "synth", "test")
    if not os.path.isdir(os.path.join(feed, "good")):            # the pretend belt's photos are made on first use
        feed = str(tmp_path / "feed")
        from PIL import Image
        for lab in ("good", "damaged"):
            os.makedirs(os.path.join(feed, lab))
            Image.new("RGB", (64, 64), (120, 100, 80)).save(os.path.join(feed, lab, "a.jpg"))
    flag = str(tmp_path / "stop.flag")
    env = dict(os.environ, CONVEYOR_STOP_FILE=flag, CONVEYOR_MOCK_FEED=feed, PYTHONUNBUFFERED="1",
               CONVEYOR_SERVER="http://127.0.0.1:9", CONVEYOR_API_KEY="")
    log = open(tmp_path / "gw.log", "w")
    # --policy always_gateway: no Q-table is involved, so stopping cannot overwrite the table a real run learned
    p = subprocess.Popen([sys.executable, "-m", "gateway_node.main", "--mock", "--esp32", "mock",
                          "--policy", "always_gateway", "--tag", "test_stop_" + str(os.getpid())], cwd=ROOT, env=env,
                         stdout=log, stderr=subprocess.STDOUT)
    try:
        t_end = time.time() + 60
        while time.time() < t_end and "belt running" not in open(tmp_path / "gw.log").read():
            assert p.poll() is None, open(tmp_path / "gw.log").read()[-2000:]
            time.sleep(0.3)
        assert "belt running" in open(tmp_path / "gw.log").read()
        time.sleep(1.0)
        assert p.poll() is None                                   # still running: nothing asked it to stop
        open(flag, "w").write("stop\n")
        code = p.wait(timeout=30)
        out = open(tmp_path / "gw.log").read()
        assert code == 0 and "stop asked by the launcher" in out
    finally:
        if p.poll() is None:
            p.kill()
        log.close()
        import shutil
        shutil.rmtree(os.path.join(ROOT, "logs", "test_stop_" + str(os.getpid())), ignore_errors=True)


def test_launcher_asks_first_and_only_then_kills(tmp_path):
    sys.path.insert(0, ROOT)
    from tools import go
    flag = str(tmp_path / "stop_gateway.flag")
    polite = subprocess.Popen([sys.executable, "-c",
                               "import os, sys, time\nwhile not os.path.exists(sys.argv[1]): time.sleep(0.05)", flag])
    polite._stopfile = flag
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])     # e.g. the cloud server
    try:
        t = time.time()
        go.ask_to_stop([polite, other], seconds=10.0)
        assert polite.poll() == 0 and time.time() - t < 8          # it ended by itself, well before the limit
        assert other.poll() is None                                 # a program without a stop file is not touched here
        assert not os.path.exists(flag)                             # the flag is cleared for the next start
    finally:
        for p in (polite, other):
            if p.poll() is None:
                p.kill()


def test_opencv_install_never_compiles_and_has_a_second_try(monkeypatch):
    """A very new Python: OpenCV's package may ask for a numpy that does not exist for it. The launcher must not
    start compiling, and must try once more with OpenCV alone before it gives up."""
    sys.path.insert(0, ROOT)
    from tools import go
    calls, works = [], {"after": None}

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return 0 if works["after"] is not None and len(calls) >= works["after"] else 1

    class Stopped(Exception):
        pass

    def fake_stop(msg):
        raise Stopped(msg)

    monkeypatch.setattr(go, "run", fake_run)
    monkeypatch.setattr(go, "stop", fake_stop)
    monkeypatch.setattr(go, "say", lambda *a: None)
    # 1. OpenCV already works: nothing is installed
    monkeypatch.setattr(go, "have_cv2", lambda: True)
    go.ensure_cv2()
    assert calls == []
    # 2. the normal install fails, OpenCV alone works
    works["after"] = 2
    monkeypatch.setattr(go, "have_cv2", lambda: len(calls) >= 2)
    go.ensure_cv2()
    assert len(calls) == 2 and all("--only-binary=:all:" in c for c in calls)
    assert "--no-deps" not in calls[0] and "--no-deps" in calls[1]
    # 3. both fail: a clear stop, no third try
    calls.clear()
    works["after"] = None
    monkeypatch.setattr(go, "have_cv2", lambda: False)
    try:
        go.ensure_cv2()
        raise AssertionError("should have stopped")
    except Stopped as e:
        assert "OpenCV could not be installed" in str(e)
    assert len(calls) == 2


def test_a_step_shows_its_lines_while_it_runs_not_only_at_the_end(monkeypatch):
    """'Wave a hand in front of the sensor NOW' must appear before the ten seconds are over."""
    sys.path.insert(0, ROOT)
    from tools import go
    seen = []
    monkeypatch.setattr(go, "say", lambda msg="": seen.append((time.time(), msg)))
    code = go.run([sys.executable, "-c", "import time\nprint('wave now')\ntime.sleep(1.5)\nprint('over')"])
    assert code == 0
    t_wave = next(t for t, m in seen if m.strip() == "wave now")      # (the first line shown is the command itself)
    t_over = next(t for t, m in seen if m.strip() == "over")
    assert t_over - t_wave > 1.0                      # with held-back output both lines arrive together


def test_first_hardware_test_runs_the_steps_in_order_and_stops_with_a_clear_reason(monkeypatch):
    sys.path.insert(0, ROOT)
    from tools import go
    order, port_code = [], {"v": 0}

    class Stopped(Exception):
        pass

    def fake_run(cmd, **kw):
        name = cmd[2] if len(cmd) > 2 else "?"
        order.append(name)
        return port_code["v"] if name == "tools.ports" else 0

    def fake_stop(msg):
        raise Stopped(msg)

    monkeypatch.setattr(go, "run", fake_run)
    monkeypatch.setattr(go, "stop", fake_stop)
    monkeypatch.setattr(go, "say", lambda *a: None)
    for name in ("ensure_env", "ensure_cv2", "ensure_ready"):
        monkeypatch.setattr(go, name, lambda *a, **k: None)
    monkeypatch.setattr(go, "mode_line", lambda: order.append("line"))
    monkeypatch.setattr(go, "mode_tradeoff", lambda: order.append("tradeoff"))
    monkeypatch.setattr(go.subprocess, "call", lambda cmd, **kw: order.append(cmd[3]) or 0)
    go.mode_first()
    assert order == ["tools.ports", "tools.esp32_check", "tools.camera_check", "line", "tradeoff"]
    for code, words in ((2, "no ESP32 was found"), (3, "does not answer")):
        order.clear()
        port_code["v"] = code
        try:
            go.mode_first()
            raise AssertionError("should have stopped")
        except Stopped as e:
            assert words in str(e)
        assert order == ["tools.ports"]               # nothing else is started without the board
