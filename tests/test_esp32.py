"""ESP32 edge node: the tiny-model file format, and the REAL firmware file compiled for the PC and driven
through the gateway's own link code (needs g++ and a Unix pseudo-terminal; skipped elsewhere, e.g. on Windows).

    python -m pytest -q tests/test_esp32.py
"""
import os
import select
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import tiny  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_model(size=32, ch=1, widths=(8, 16, 32), seed=0, strides=None):
    r = np.random.default_rng(seed)
    convs, cin = [], ch
    for w, st in zip(widths, strides or [2] * len(widths)):
        convs.append(((r.standard_normal((w, cin, 3, 3)) * 0.3).astype("f4"), (r.standard_normal(w) * 0.1).astype("f4"), st))
        cin = w
    return tiny.pack(size, ch, 0.45, 0.25, convs, (r.standard_normal(cin) * 0.5).astype("f4"), 0.1)


def slow_forward(model, px):
    """Plain loops, written independently of common/tiny.py."""
    size, ch = model["size"], model["ch"]
    x = (np.frombuffer(px, np.uint8).astype(np.float32).reshape(ch, size, size) / 255.0 - model["mean"]) / model["std"]
    for w, b, s in model["convs"]:
        h = x.shape[1]
        ho = (h + 2 - 3) // s + 1
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        o = np.zeros((w.shape[0], ho, ho), np.float32)
        for oy in range(ho):
            for ox in range(ho):
                o[:, oy, ox] = (w * xp[None, :, oy * s:oy * s + 3, ox * s:ox * s + 3]).sum((1, 2, 3)) + b
        x = np.maximum(o, 0)
    return 1 / (1 + np.exp(-float(model["db"] + (model["dw"] * x.mean((1, 2))).sum())))


def test_tiny_file_roundtrip_and_arithmetic():
    blob = make_model()
    m = tiny.TinyModel(blob)
    assert (m.size, m.ch, m.params) == (32, 1, 8 * 9 + 8 + 16 * 8 * 9 + 16 + 32 * 16 * 9 + 32 + 33)
    assert len(blob) % 4 == 0 and m.mults == 16 * 16 * 8 * 9 + 8 * 8 * 16 * 8 * 9 + 4 * 4 * 32 * 16 * 9
    r = np.random.default_rng(1)
    for _ in range(5):
        px = r.integers(0, 256, 1024, dtype=np.uint8).tobytes()
        assert abs(m.predict_pixels(px) - slow_forward(m.m, px)) < 1e-5
    batch = r.integers(0, 256, (7, 1, 32, 32), dtype=np.uint8)
    assert np.allclose(tiny.forward(m.m, batch), [m.predict_pixels(b.tobytes()) for b in batch], atol=1e-6)
    with pytest.raises(ValueError):
        tiny.parse(blob[:-4])
    with pytest.raises(ValueError):
        tiny.parse(b"NOPE" + blob[4:])


def test_photo_to_pixels_is_what_the_esp32_expects():
    from PIL import Image
    img = Image.new("RGB", (256, 256), (200, 100, 50))
    px = tiny.to_pixels(img, 32, 1)
    assert len(px) == 1024 and len(set(px)) == 1          # flat colour -> flat grey
    assert len(tiny.to_pixels(img, 48, 3)) == 48 * 48 * 3
    assert tiny.checksum(b"\xff" * 300) == (255 * 300) & 0xFFFF


# ----------------------------------------------------------------------------- firmware on the PC
class FdPort:
    """Looks like a serial port to Esp32Link, but is one end of a pseudo-terminal."""

    def __init__(self, fd):
        self.fd = fd

    def read(self, n):
        if not select.select([self.fd], [], [], 0.05)[0]:
            return b""
        return os.read(self.fd, n)

    def write(self, data):
        os.write(self.fd, data)

    def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture(scope="module")
def firmware_bin(tmp_path_factory):
    if os.name != "posix" or shutil.which("g++") is None:
        pytest.skip("needs g++ and a Unix pseudo-terminal")
    out = str(tmp_path_factory.mktemp("fw") / "esp32_host")
    cmd = ["g++", "-O2", "-Wall", "-Wextra", "-Werror", "-DHOST_TEST", "-I", os.path.join(ROOT, "tests", "esp32_host"),
           "-x", "c++", os.path.join(ROOT, "esp32_node", "esp32_node.ino"), "-o", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


class Board:
    """The firmware running as a PC program: serial = pseudo-terminal, pins = control pipe, PWM = stderr."""

    def __init__(self, binary):
        import pty
        import tty
        self.binary = binary
        self.master, self.slave = pty.openpty()
        tty.setraw(self.slave)
        tty.setraw(self.master)
        self.start()

    def start(self):
        cr, self.cw = os.pipe()
        self.err = open(os.path.join(os.path.dirname(self.binary), "pins.log"), "w+")
        self.p = subprocess.Popen([self.binary], stdin=self.slave, stdout=self.slave, stderr=self.err,
                                  preexec_fn=lambda: os.dup2(cr, 3), close_fds=False)
        os.close(cr)

    def pin(self, n, v):
        os.write(self.cw, f"pin {n} {v}\n".encode())

    def nomem(self, k):
        """The next k times the firmware asks for memory for a model, it gets none."""
        os.write(self.cw, f"nomem {k}\n".encode())

    def pwm_log(self):
        self.err.flush()
        self.err.seek(0)
        return self.err.read().split("\n")

    def stop(self):
        self.p.kill()
        self.p.wait()
        os.close(self.cw)


@pytest.fixture()
def board(firmware_bin):
    b = Board(firmware_bin)
    yield b
    b.stop()


def link_to(board):
    from gateway_node.esp32_link import Esp32Link
    link = Esp32Link("pty", verbose=False, opener=lambda: FdPort(board.master))
    link.open(hello_timeout=3.0)
    return link


def test_firmware_runs_the_tiny_model_like_the_pc(board):
    link = link_to(board)
    try:
        assert "SCQC-ESP32" in link.firmware and not link.model_loaded
        m = tiny.TinyModel(make_model())
        from gateway_node.esp32_link import Esp32Error
        with pytest.raises(Esp32Error):
            link.infer(m.pixels_blank())                      # no model on the board yet
        link.upload_model(m.blob)
        r = np.random.default_rng(3)
        for _ in range(25):
            px = r.integers(0, 256, 1024, dtype=np.uint8).tobytes()
            out = link.infer(px, 1.0)
            assert abs(out["p"] - m.predict_pixels(px)) < 2e-4      # answer is sent with 4 decimals
        with pytest.raises(Esp32Error):
            link.infer(bytes(100), 1.0)                       # wrong photo size is refused, not guessed
        big = tiny.TinyModel(make_model(48, 1, (8, 16, 24, 32), seed=5, strides=(2, 2, 2, 1)))   # the bigger candidate of the trainer
        link.upload_model(big.blob)
        px = r.integers(0, 256, 48 * 48, dtype=np.uint8).tobytes()
        assert abs(link.infer(px, 1.0)["p"] - big.predict_pixels(px)) < 2e-4
        with pytest.raises(Esp32Error):
            link.upload_model(big.blob[:-8])                  # damaged file refused ...
        link.model_loaded = True
        assert abs(link.infer(px, 1.0)["p"] - big.predict_pixels(px)) < 2e-4   # ... and the old model still works
    finally:
        link.close()


def test_firmware_belt_events_arm_and_safety_stop(board):
    link = link_to(board)
    events = []
    link.on_event = lambda tag, t: events.append(tag)
    try:
        link.configure(servo_rest=0, servo_push=90, ir_low=1, debounce_ms=20)
        board.pin(32, 0); time.sleep(0.15)                    # box under the camera
        board.pin(32, 1); time.sleep(0.01); board.pin(32, 0); time.sleep(0.01); board.pin(32, 1)   # contact bounce
        time.sleep(0.1)
        board.pin(33, 0); time.sleep(0.15); board.pin(33, 1); time.sleep(0.1)                   # box at the arm
        board.pin(32, 0); time.sleep(0.15); board.pin(32, 1); time.sleep(0.1)                   # next box
        assert events == ["C", "A", "C"]
        st = link.status()
        assert st and not st["ir_camera"] and not st["ir_arm"]
        link.set_duty(180)
        link.push(120)
        link.send("L 2")                                       # the old LED command: firmware 2 and later ignore it
        time.sleep(0.4)
        log = board.pwm_log()
        assert "pwm 25 180" in log                             # belt motor
        assert "pwm 18 4915" in log and log.index("pwm 18 4915") < len(log) - 1   # arm out (90 deg) ...
        assert log[-2 if log[-1] == "" else -1].startswith("pwm 18 1638")            # ... and back to rest (0 deg)
        assert not any(x.startswith("pin16") or x.startswith("pin17") for x in log)      # GPIO 16 / 17 are never touched
        assert link.firmware.split()[2] == "3"                                           # H SCQC-ESP32 3 ...
        assert link.status()["duty"] == 180
        # the gateway goes silent: the keep-alive stops -> the board stops the belt by itself after 3 s
        link._stop = True
        time.sleep(4.2)
        link._stop = False
        import threading
        threading.Thread(target=link._reader, daemon=True).start()
        assert link.status()["duty"] == 0
        assert "pwm 25 0" in board.pwm_log()
    finally:
        link.close()


def arm_moves(board):
    """How often the servo signal changed (the first entry is the arm going to rest at start-up)."""
    return [x for x in board.pwm_log() if x.startswith("pwm 18 ")]


def test_refused_model_or_photo_is_thrown_away_not_read_as_commands(board):
    """The raw bytes that follow a refused 'W' or 'I' line must not be read as text: a stray "P" line in them
    would swing the reject arm, a stray "B 255" would run the belt at full speed."""
    from gateway_node.esp32_link import Esp32Error
    link = link_to(board)
    try:
        rest = arm_moves(board)
        junk = (b"\nP 5000\nB 255\n" + bytes(range(256))) * 260            # about 70 kB: too big for the board
        r = link.request(f"W {len(junk)} {tiny.checksum(junk)}", "W ", 8.0, payload=junk)
        assert r.startswith("W ERR size")
        st = link.status()                                                   # the board is in step again at once
        assert st is not None and st["duty"] == 0                            # ... and the belt was not started
        photo = (b"\nP 5000\n" + bytes(40)) * 200                           # a "photo" far too big
        m = tiny.TinyModel(make_model())
        link.upload_model(m.blob)
        with pytest.raises(Esp32Error):
            link.infer(photo, 2.0)
        time.sleep(0.3)
        assert link.status()["duty"] == 0 and arm_moves(board) == rest       # the arm never moved
        px = bytes(range(256)) * 4
        assert abs(link.infer(px, 1.0)["p"] - m.predict_pixels(px)) < 2e-4   # and real photos still work
    finally:
        link.close()


def test_new_model_replaces_the_old_one_when_memory_is_short(board):
    from gateway_node.esp32_link import Esp32Error
    link = link_to(board)
    try:
        rest = arm_moves(board)
        a, b = tiny.TinyModel(make_model(seed=1)), tiny.TinyModel(make_model(seed=2))
        px = bytes(range(256)) * 4
        link.upload_model(a.blob)
        board.nomem(1)                       # no room for two models at once: the old one must go first
        time.sleep(0.1)
        link.upload_model(b.blob)
        assert abs(link.infer(px, 1.0)["p"] - b.predict_pixels(px)) < 2e-4
        board.nomem(2)                       # no room at all: refused, and the model's bytes are thrown away
        time.sleep(0.1)
        with pytest.raises(Esp32Error) as e:
            link.upload_model(a.blob)
        assert "memory" in str(e.value) and not link.model_loaded
        st = link.status()
        assert st is not None and st["duty"] == 0 and arm_moves(board) == rest
        assert "SCQC-ESP32 3 0" in link.request("H", "H SCQC-ESP32", 1.0)    # the board says: no model loaded
        link.upload_model(a.blob)            # memory is back: works again
        assert abs(link.infer(px, 1.0)["p"] - a.predict_pixels(px)) < 2e-4
    finally:
        link.close()


def test_a_refused_model_does_not_stop_the_line(board, capsys):
    """connect() must come back with a working link (belt, sensors, arm) even when the board refuses the model."""
    from gateway_node import esp32_link
    from common.config import load_config
    cfg = load_config()
    real = esp32_link.Esp32Link

    class Pty(real):
        def __init__(self, port, baud=460800, verbose=True, opener=None):
            super().__init__(port, baud, verbose, opener=lambda: FdPort(board.master))

    esp32_link.Esp32Link = Pty
    try:
        board.nomem(5)
        time.sleep(0.1)
        link = esp32_link.connect(cfg, tiny.TinyModel(make_model()).blob, "pty", verbose=False)
    finally:
        esp32_link.Esp32Link = real
    try:
        assert link.connected and not link.model_loaded
        assert "WARNING" in capsys.readouterr().out
        link.set_duty(120)
        time.sleep(0.2)
        assert link.status()["duty"] == 120                   # the belt still obeys
    finally:
        link.close()


def test_board_reset_gets_its_model_back(board):
    link = link_to(board)
    try:
        m = tiny.TinyModel(make_model())
        link.configure(servo_push=70)
        link.upload_model(m.blob)
        link.set_duty(150)
        px = bytes(range(256)) * 4
        want = m.predict_pixels(px)
        assert abs(link.infer(px, 1.0)["p"] - want) < 2e-4
        board.p.kill(); board.p.wait()                         # power dip / reset button
        board.start()
        t0 = time.time()
        while time.time() - t0 < 8 and not link.reinits:
            time.sleep(0.1)
        assert link.resets == 1 and link.reinits == 1 and link.model_loaded
        assert abs(link.infer(px, 1.0)["p"] - want) < 2e-4
        assert link.status()["duty"] == 150                    # belt speed was sent again too
    finally:
        link.close()


def test_edge_worker_falls_back_cleanly_when_the_board_dies(board):
    from PIL import Image
    from gateway_node.esp32_link import EdgeTiny
    link = link_to(board)
    try:
        m = tiny.TinyModel(make_model())
        link.upload_model(m.blob)
        edge = EdgeTiny(m, link, timeout_s=0.3)
        edge.measure(3)
        img = Image.new("RGB", (256, 256), (120, 90, 60))
        r = edge.check(img)
        assert r["ok"] and r["where"] == "EDGE" and not r["emulated"] and abs(r["p_damaged"] - m.predict(img)) < 2e-4
        assert edge.ready
        board.p.kill(); board.p.wait()
        r = edge.check(img)
        assert not r["ok"] and "esp32" in r["error"] and edge.failures == 1
    finally:
        link.close()


def test_pretend_esp32_gives_the_same_answers_and_takes_cable_time():
    from PIL import Image
    from gateway_node.esp32_link import EdgeTiny
    m = tiny.TinyModel(make_model())
    edge = EdgeTiny(m, None)
    img = Image.new("RGB", (256, 256), (10, 200, 30))
    t = time.perf_counter()
    r = edge.check(img)
    took = time.perf_counter() - t
    assert r["ok"] and r["emulated"] and abs(r["p_damaged"] - m.predict(img)) < 1e-9
    assert took >= 0.9 * (edge.link_s + edge.compute_s) and edge.ready
