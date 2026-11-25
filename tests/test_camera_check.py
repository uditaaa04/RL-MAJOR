"""The camera check (CHECK_CAMERA.bat): which camera the real line uses, the page, and the two settings it
writes into config/system.yaml. Uses a FAKE OpenCV with two cameras - no real camera is touched.

    python -m pytest -q tests/test_camera_check.py
"""
import builtins
import json
import os
import re
import shutil
import sys
import time
import types

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import DEFAULT_PATH  # noqa: E402
from gateway_node import webcam  # noqa: E402
from tools import camera_check as cc  # noqa: E402


class Cap:
    """Camera 0 = the laptop's own camera in a dark room, camera 1 = the webcam over the belt, well lit."""

    def __init__(self, dev, *a):
        self.dev = dev

    def isOpened(self):
        return self.dev in (0, 1)

    def set(self, prop, v):
        return True

    def get(self, prop):
        return {3: 640, 4: 480, 5: 30.0, 6: 0x47504A4D}.get(prop, 0)

    def grab(self):
        time.sleep(0.005)
        return True

    def retrieve(self):
        f = np.full((480, 640, 3), 20, np.uint8)
        if self.dev == 1:
            f[:] = 120
            f[::8, :, :] = 200                    # printed lines on a box: detail = sharp
            f[:, ::8, :] = 60
        return True, f

    def release(self):
        pass


@pytest.fixture()
def two_cameras(monkeypatch, tmp_path):
    m = types.ModuleType("cv2")
    m.CAP_V4L2, m.CAP_PROP_FRAME_WIDTH, m.CAP_PROP_FRAME_HEIGHT, m.CAP_PROP_FPS = 200, 3, 4, 5
    m.CAP_PROP_FOURCC, m.CAP_PROP_BUFFERSIZE, m.COLOR_BGR2RGB, m.CAP_DSHOW = 6, 38, 4, 700
    m.VideoCapture = Cap
    m.VideoWriter_fourcc = lambda *c: sum(ord(ch) << 8 * i for i, ch in enumerate(c))
    m.cvtColor = lambda img, code: img[..., ::-1]
    monkeypatch.setitem(sys.modules, "cv2", m)
    monkeypatch.setattr(webcam, "is_linux", lambda: False)
    monkeypatch.setattr(webcam, "is_windows", lambda: True)
    monkeypatch.setattr(cc, "OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(cc, "SETTLE_S", 0.1)
    cfg = str(tmp_path / "system.yaml")
    shutil.copy(DEFAULT_PATH, cfg)
    return cfg, tmp_path / "out"


def page_data(out):
    html = (out / "index.html").read_text(encoding="utf-8")
    return json.loads(re.search(r"const D = (\{.*?\});\n", html, re.S).group(1)), html


def test_page_shows_every_camera_and_marks_the_one_the_line_uses(two_cameras):
    cfg, out = two_cameras
    assert cc.main(["--no-ask", "--no-open", "--config", cfg]) == 0
    d, html = page_data(out)
    assert d["used"] == 1 and [c["dev"] for c in d["cams"]] == [0, 1]
    assert [c["used"] for c in d["cams"]] == [False, True]
    dark, belt = d["cams"]
    assert dark["advice"][0]["bad"] and "dark" in dark["advice"][0]["text"].lower()
    assert not belt["advice"][0]["bad"] and belt["width"] == 640 and belt["roi"] == [80, 0, 480, 480]
    for name in ("camera_0.jpg", "camera_0_crop.jpg", "camera_1.jpg", "camera_1_crop.jpg"):
        assert (out / name).stat().st_size > 500
    assert 'data-theme' in html and "Dark" in html             # light and dark, with a button
    assert open(cfg, encoding="utf-8").read() == open(DEFAULT_PATH, encoding="utf-8").read()   # --no-ask changes nothing


def test_typed_answers_are_saved_and_nothing_else_in_the_file_changes(two_cameras, monkeypatch):
    cfg, out = two_cameras
    before = open(cfg, encoding="utf-8").read().split("\n")
    answers = iter(["0", "100 20 300"])
    monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    assert cc.main(["--no-open", "--config", cfg]) == 0
    after = open(cfg, encoding="utf-8").read().split("\n")
    new = yaml.safe_load("\n".join(after))
    assert new["camera"]["usb_device"] == 0 and new["camera"]["roi"] == [100, 20, 300, 300]
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after) and len(changed) == 2
    assert all("#" in b for _, b in changed)                    # the comments on those two lines are still there
    # the next check uses the camera that was typed, with no search through the cameras
    assert cc.main(["--no-ask", "--no-open", "--config", cfg]) == 0
    d, _ = page_data(out)
    assert d["used"] == 0 and d["cams"][0]["used"] and d["cams"][0]["ok"]
    assert "camera.usb_device is set to 0" in d["why"]


def test_enter_twice_changes_nothing_and_a_wrong_answer_is_refused(two_cameras, monkeypatch):
    cfg, _ = two_cameras
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    for typed in (["", ""], ["7", "abc"], ["1", "80 0"]):
        answers = iter(typed)
        monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
        assert cc.main(["--no-open", "--config", cfg]) == 0
        want = yaml.safe_load(open(DEFAULT_PATH, encoding="utf-8"))
        got = yaml.safe_load(open(cfg, encoding="utf-8"))
        if typed[0] == "1":                                     # 'auto' becomes the number that was typed
            want["camera"]["usb_device"] = 1
        assert got == want


def test_square_is_kept_inside_the_picture():
    assert cc.fit_roi([80, 0, 480, 480], 640, 480) == [80, 0, 480, 480]
    assert cc.fit_roi([500, 300, 480, 480], 640, 480) == [160, 0, 480, 480]
    assert cc.fit_roi([0, 0, 900, 900], 640, 480) == [0, 0, 480, 480]
    assert cc.parse_square("100, 20, 300, 300", 640, 480) == [100, 20, 300, 300]
    assert cc.parse_square("600 400 300", 640, 480) == [340, 180, 300, 300]
    assert cc.parse_square("1 2", 640, 480) is None and cc.parse_square("0 0 5", 640, 480) is None


def test_setting_a_value_keeps_comments_and_refuses_unknown_keys():
    text = "a:\n  roi: [1, 2]   # note\ncamera:\n  # roi: old\n  roi: [80, 0, 480, 480]   # x, y, w, h\n  w: 1\nb:\n  roi: 5\n"
    out = cc.set_yaml_value(text, "camera", "roi", "[1, 2, 3, 3]")
    assert out == text.replace("[80, 0, 480, 480]", "[1, 2, 3, 3]")
    assert cc.set_yaml_value("camera:\n  usb_device: auto\n", "camera", "usb_device", "2") == "camera:\n  usb_device: 2\n"
    win = text.replace("\n", "\r\n")                         # the same file saved by a Windows editor
    assert cc.set_yaml_value(win, "camera", "roi", "[1, 2, 3, 3]") == win.replace("[80, 0, 480, 480]", "[1, 2, 3, 3]")
    assert cc.set_yaml_value(win, "camera", "w", "7") == win.replace("  w: 1\r\n", "  w: 7\r\n")
    with pytest.raises(ValueError):
        cc.set_yaml_value(text, "camera", "nothere", "1")
    with pytest.raises(ValueError):
        cc.set_yaml_value(text, "nosection", "roi", "1")


def test_windows_line_ends_survive_a_save(two_cameras, tmp_path):
    cfg, _ = two_cameras
    raw = open(cfg, "rb").read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    open(cfg, "wb").write(raw)
    assert cc.save_settings(cfg, device=2, roi=[10, 20, 300, 300]) == ["camera.usb_device: 2", "camera.roi: [10, 20, 300, 300]"]
    out = open(cfg, "rb").read()
    assert out.count(b"\r\n") == raw.count(b"\r\n") and out.count(b"\n") == raw.count(b"\n")
    new = yaml.safe_load(out.decode("utf-8"))
    assert new["camera"]["usb_device"] == 2 and new["camera"]["roi"] == [10, 20, 300, 300]
    assert cc.save_settings(cfg, device=2, roi=[10, 20, 300, 300]) == []       # nothing to change: nothing written


def test_no_camera_gives_a_clear_page_and_exit_code(two_cameras, monkeypatch):
    cfg, out = two_cameras
    monkeypatch.setattr(Cap, "isOpened", lambda self: False)
    assert cc.main(["--no-ask", "--no-open", "--config", cfg]) == 1
    d, _ = page_data(out)
    assert d["cams"] == [] and d["used"] is None
