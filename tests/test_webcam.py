"""USB webcam path and hardware-robustness checks, with a FAKE OpenCV (no camera / gateway needed).

    python -m pytest -q tests/test_webcam.py
"""
import os
import sys
import tempfile
import threading
import time
import types

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import load_config  # noqa: E402
from gateway_node import webcam  # noqa: E402

# ----------------------------------------------------------------------------- v4l2-ctl parsing
NEW_KERNEL = """
User Controls

                     brightness 0x00980900 (int)    : min=0 max=255 step=1 default=128 value=128
        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
      white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=1 default=4000 value=4000 flags=inactive
           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=2 value=2 (60 Hz)

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
         exposure_time_absolute 0x009a0902 (int)    : min=3 max=2047 step=1 default=250 value=250 flags=inactive
     exposure_dynamic_framerate 0x009a0903 (bool)   : default=0 value=1
"""
OLD_KERNEL = """
                  exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
              exposure_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=156 value=156
                     focus_auto 0x009a090c (bool)   : default=1 value=1
                 focus_absolute 0x009a090a (int)    : min=0 max=250 step=5 default=0 value=0
"""


def test_parse_controls():
    c = webcam.parse_controls(NEW_KERNEL)
    assert c["brightness"] == {"type": "int", "min": 0, "max": 255, "step": 1, "default": 128, "value": 128}
    assert c["auto_exposure"]["type"] == "menu" and c["auto_exposure"]["value"] == 3
    assert c["exposure_time_absolute"]["flags"] == "inactive"
    assert len(c) == 7


def test_plan_controls_aliases_clamping_and_order():
    want = {"power_line_frequency": 1, "auto_exposure": 1, "exposure_time_absolute": 100000,
            "focus_automatic_continuous": 0, "focus_absolute": 40, "gain": 10}
    new = {n: v for n, v, _ in webcam.plan_controls(want, webcam.parse_controls(NEW_KERNEL))}
    assert new["auto_exposure"] == 1 and new["exposure_time_absolute"] == 2047      # clamped to max
    assert new["gain"] is None and new["focus_absolute"] is None                    # not on this webcam
    old = [(n, v) for n, v, _ in webcam.plan_controls(want, webcam.parse_controls(OLD_KERNEL))]
    names = [n for n, v in old if v is not None]
    # new spellings map onto the old kernel's names, in the order given in the config
    assert names == ["exposure_auto", "exposure_absolute", "focus_auto", "focus_absolute"]
    assert dict(old)["exposure_absolute"] == 5000


# ----------------------------------------------------------------------------- fake OpenCV
class FakeCapture:
    opened = 0
    fail_after = None          # grab() starts failing after this many grabs (simulated unplug)

    def __init__(self, dev, *a):
        FakeCapture.opened += 1
        self.grabs, self.retrieves, self.times = 0, 0, []
        self.props = {}

    def isOpened(self):
        return True

    def set(self, prop, v):
        self.props[prop] = v
        return True

    def get(self, prop):
        return {3: 640, 4: 480, 5: 30.0, 6: 0x47504A4D}.get(prop, 0)

    def grab(self):
        time.sleep(0.01)
        if FakeCapture.fail_after is not None and self.grabs >= FakeCapture.fail_after:
            return False
        self.grabs += 1
        self.times.append(time.perf_counter())
        return True

    def retrieve(self):
        self.retrieves += 1
        f = np.zeros((480, 640, 3), np.uint8)
        f[0, 0, 0] = self.grabs % 256                # which grab this picture came from
        return True, f

    def release(self):
        pass


def fake_cv2():
    m = types.ModuleType("cv2")
    m.CAP_V4L2, m.CAP_PROP_FRAME_WIDTH, m.CAP_PROP_FRAME_HEIGHT, m.CAP_PROP_FPS = 200, 3, 4, 5
    m.CAP_PROP_FOURCC, m.CAP_PROP_BUFFERSIZE, m.COLOR_BGR2RGB = 6, 38, 4
    m.CAP_DSHOW = 700                              # Windows path (DirectShow)
    m.VideoCapture = FakeCapture
    m.VideoWriter_fourcc = lambda *c: sum(ord(ch) << 8 * i for i, ch in enumerate(c))
    m.cvtColor = lambda img, code: img[..., ::-1]
    return m


def usb_camera():
    sys.modules["cv2"] = fake_cv2()
    from gateway_node.camera import Camera
    cfg = load_config(overrides={"camera": {"source": "usb", "usb_controls": {}, "usb_warmup_frames": 0}})
    return Camera(cfg, verbose=False)


def test_usb_photo_is_taken_after_the_trigger_and_decoded_only_on_demand():
    FakeCapture.fail_after = None
    cam = usb_camera()
    try:
        time.sleep(0.2)
        assert cam.cam.grabs > 5 and cam.cam.retrieves == 0        # frames pulled, none decoded
        for _ in range(5):
            t = time.perf_counter()
            img = cam.capture_box(t)
            assert img.size == (256, 256)
            k = cam._frame[0, 0, 2]                                   # (RGB after the fake BGR->RGB flip)
            assert cam.cam.times[k - 1] >= t - 0.001                  # pulled AFTER IR-1 (frames are 10 ms apart)
        assert cam.cam.retrieves <= 6                                 # one decode per photo, not per frame
    finally:
        cam.close()


def test_usb_webcam_reconnects_after_unplug():
    FakeCapture.opened, FakeCapture.fail_after = 0, 10
    cam = usb_camera()
    try:
        time.sleep(0.4)
        assert cam.cam is None or FakeCapture.opened >= 2           # dropped after 5 failed grabs
        FakeCapture.fail_after = None
        t_end = time.time() + 3.0
        while time.time() < t_end and (cam.cam is None or FakeCapture.opened < 2):
            time.sleep(0.05)
        assert FakeCapture.opened >= 2 and cam.cam is not None
        assert cam.capture_box().size == (256, 256)                   # works again after the re-connect
    finally:
        FakeCapture.fail_after = None
        cam.close()


def test_capture_raises_when_the_webcam_sends_nothing():
    FakeCapture.fail_after = 0
    cam = usb_camera()
    try:
        t = time.perf_counter()
        try:
            cam.capture_box()
            assert False, "should have raised"
        except RuntimeError:
            pass
        assert time.perf_counter() - t < 1.6
    finally:
        FakeCapture.fail_after = None
        cam.close()


# ----------------------------------------------------------------------------- pipeline FIFO
class StubHW:
    duty = 0.6

    def __init__(self):
        self.pushes = 0

    def belt_speed_mps(self):
        return 0.18

    def push(self):
        self.pushes += 1

    def show(self, v):
        pass

    def power_w(self):
        return None


class StubCam:
    def __init__(self):
        self.calls = 0

    def capture_box(self, t0=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("camera gave no picture within 1 s")
        return Image.new("RGB", (256, 256))


class StubGateway:
    queue_len = 0
    ewma_ms = 8.0

    def submit(self, box_id, img, cb):
        cb(box_id, {"ok": True, "p_damaged": 0.99, "where": "GATEWAY", "compute_s": 0.01})


class StubEsp32:
    """ESP32 whose cable is pulled after `good_for` boxes."""
    queue_len = 0
    ewma_ms = 40.0

    def __init__(self, good_for=10 ** 9, ready=True):
        self.good_for, self.ready, self.calls = good_for, ready, 0

    def submit(self, box_id, img, cb):
        self.calls += 1
        if self.calls > self.good_for:
            cb(box_id, {"ok": False, "error": "esp32: write failed", "where": "EDGE"})
        else:
            cb(box_id, {"ok": True, "p_damaged": 0.99, "where": "EDGE", "compute_s": 0.04, "esp_ms": 9.0})


class StubNet:
    cprobe_timeout = 1.0

    def snapshot(self):
        return {"rtt_ms": 10.0, "cloud_ms": 40.0, "cpu": 0.2, "server_ok": True}

    def record_cloud(self, s):
        pass


def test_camera_error_keeps_the_belt_order():
    """Box 0 has no photo, box 1 is damaged. The arm must push at box 1's arrival, not box 0's."""
    from common.policies import AlwaysGateway
    from common.rl import RewardModel, StateSpace
    from gateway_node.pipeline import Pipeline
    cfg = load_config(overrides={"conveyor": {"min_box_gap_s": 0.0}})
    hw = StubHW()
    reward = RewardModel(cfg["reward"], 0.3, {"recall_damaged": 0.9, "false_reject_rate": 0.06, "accuracy": 0.93},
                         {"recall_damaged": 0.98, "false_reject_rate": 0.02, "accuracy": 0.98})
    with tempfile.TemporaryDirectory() as d:
        pipe = Pipeline(cfg, hw, StubCam(), StubEsp32(), StubGateway(), types.SimpleNamespace(inflight=0), StubNet(),
                        AlwaysGateway(), reward, StateSpace(cfg["state_bins"]), d)
        pipe.on_box_at_camera()             # box 0: camera error
        pipe.on_box_at_camera()             # box 1: photo ok, verdict damaged
        assert list(pipe.fifo) == [0, 1]
        pipe.on_box_at_arm()                # box 0 reaches the arm: unchecked, NOT pushed
        assert hw.pushes == 0
        pipe.on_box_at_arm()                # box 1 reaches the arm: pushed
        assert hw.pushes == 1
        assert pipe.stats["n"] == 2 and pipe.stats["missed"] == 1 and pipe.stats["gateway"] == 1
        pipe.close()
        with open(os.path.join(d, "boxes.csv")) as f:
            rows = [ln.split(",") for ln in f.read().splitlines()[1:]]
        assert rows[0][4] == "NONE" and rows[0][16] == "unchecked" and "camera" in rows[0][21]
        assert rows[1][4] == "GATEWAY" and rows[1][16] == "damaged"


class GoodCam:
    def capture_box(self, t0=None):
        return Image.new("RGB", (256, 256))


def _pipe(d, esp, policy):
    from common.rl import RewardModel, StateSpace
    from gateway_node.pipeline import Pipeline
    cfg = load_config(overrides={"conveyor": {"min_box_gap_s": 0.0}})
    reward = RewardModel(cfg["reward"], 0.3, {"recall_damaged": 0.9, "false_reject_rate": 0.06, "accuracy": 0.93},
                         {"recall_damaged": 0.98, "false_reject_rate": 0.02, "accuracy": 0.98})
    hw = StubHW()
    return hw, Pipeline(cfg, hw, GoodCam(), esp, StubGateway(), types.SimpleNamespace(inflight=0), StubNet(),
                        policy, reward, StateSpace(cfg["state_bins"]), d)


def test_esp32_checks_a_box_and_the_arm_pushes_it():
    from common.policies import AlwaysEdge
    with tempfile.TemporaryDirectory() as d:
        hw, pipe = _pipe(d, StubEsp32(), AlwaysEdge())
        pipe.on_box_at_camera()
        pipe.on_box_at_arm()
        assert hw.pushes == 1 and pipe.stats["edge"] == 1 and pipe.stats["missed"] == 0
        pipe.close()
        with open(os.path.join(d, "boxes.csv")) as f:
            head, row = [ln.split(",") for ln in f.read().splitlines()[:2]]
        assert row[head.index("action")] == "EDGE" and row[head.index("answered_by")] == "EDGE"


def test_esp32_cable_pulled_gateway_takes_over():
    """The ESP32 fails fast: the gateway checks the same box, nothing is missed."""
    from common.policies import AlwaysEdge
    with tempfile.TemporaryDirectory() as d:
        hw, pipe = _pipe(d, StubEsp32(good_for=0), AlwaysEdge())
        pipe.on_box_at_camera()
        pipe.on_box_at_arm()
        assert hw.pushes == 1 and pipe.stats["missed"] == 0 and pipe.stats["fallback"] == 1
        pipe.close()
        with open(os.path.join(d, "boxes.csv")) as f:
            head, row = [ln.split(",") for ln in f.read().splitlines()[:2]]
        assert row[head.index("action")] == "EDGE" and row[head.index("answered_by")] == "GATEWAY"


def test_esp32_not_ready_is_never_chosen():
    from common.policies import AlwaysEdge
    with tempfile.TemporaryDirectory() as d:
        esp = StubEsp32(ready=False)
        hw, pipe = _pipe(d, esp, AlwaysEdge())
        pipe.on_box_at_camera()
        pipe.on_box_at_arm()
        assert esp.calls == 0 and pipe.stats["gateway"] == 1 and pipe.stats["edge"] == 0
        pipe.close()


def test_dashboard_counts_each_place_and_no_photo_as_none():
    from server.store import LiveStats
    s = LiveStats()
    for a in ("EDGE", "GATEWAY", "CLOUD", "NONE"):
        s.add({"action": a, "on_time": a != "NONE", "latency_ms": 10})
    snap = s.snapshot()
    assert s.edge == 1 and s.gateway == 1 and s.cloud == 1 and s.n == 4 and s.missed == 1
    assert snap["edge"] == 1 and snap["gateway"] == 1 and snap["cloud"] == 1
