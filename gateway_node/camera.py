"""
Camera: the gateway's USB webcam (OpenCV) or mock (images from disk).

capture_box(t_trigger) returns the square box photo (256x256 PIL image) cut from the ROI set in
config (camera.roi), taken from the FIRST frame that arrives after the IR-1 trigger (+ the optional
camera.capture_delay_ms). A background thread keeps pulling frames from the camera so its buffer
never holds old pictures - but it only DECODES a frame when somebody is waiting for one
(grab() is cheap, retrieve() = JPEG decode is the expensive part), so the gateway's CPU stays free for
the AI model between boxes.

If the webcam stops answering (loose USB cable), the thread closes it and re-opens it every
second; capture_box() raises an error meanwhile and the pipeline counts that box as missed.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from PIL import Image

from common.vision_io import crop_roi

from . import webcam


class Camera:
    def __init__(self, cfg: dict, mock_hw=None, verbose: bool = True):
        c = cfg["camera"]
        self.cfg = c
        self.source = c["source"] if mock_hw is None else "mock"
        self.roi = c["roi"]
        self.size = int(c["upload_size"])
        self.mock_hw = mock_hw
        self.verbose = verbose
        self.cam = None
        self.device = None
        self.info = {}                  # what the camera really delivers (size, fps, format)
        self.fps = 0.0                  # measured frames per second (updated every 2 s)
        self.frames = 0                 # frames pulled from the camera so far
        self._cond = threading.Condition()
        self._frame = None              # newest DECODED frame, RGB numpy array (never modified in place)
        self._t_frame = -1.0            # time.perf_counter() when that frame was pulled
        self._wants: list[float] = []   # waiting callers: "a frame pulled at or after this time"
        self._stop = False
        self._thread = None
        if self.source == "usb":
            self.cv2 = _import_cv2()
            self._open_usb(first=True)
            self._thread = threading.Thread(target=self._loop, args=(self._pull_usb,), daemon=True, name="camera")
            self._thread.start()
        elif self.source == "picamera2":
            from picamera2 import Picamera2  # Pi only
            self.cam = Picamera2()
            conf = self.cam.create_video_configuration(main={"size": (c["width"], c["height"]), "format": "RGB888"})
            self.cam.configure(conf)
            self.cam.start()
            self.info = {"device": "Pi camera (CSI)", "width": c["width"], "height": c["height"]}
            time.sleep(1.0)  # auto-exposure settle
            self._thread = threading.Thread(target=self._loop, args=(self._pull_picam,), daemon=True, name="camera")
            self._thread.start()
        elif self.source != "mock":
            raise SystemExit(f"camera.source must be usb, picamera2 or mock (config says {self.source!r})")

    # ------------------------------------------------------------------ USB webcam
    def _open_usb(self, first=False) -> bool:
        cv2, c = self.cv2, self.cfg
        dev = webcam.find_device(c.get("usb_device", "auto"), int(c.get("usb_index", 0)))
        if webcam.is_linux():
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        elif webcam.is_windows():                  # DirectShow opens in under a second; the default can take 10 s
            cap = cv2.VideoCapture(dev, getattr(cv2, "CAP_DSHOW", 700))
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(dev)
        else:
            cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            cap.release()
            if first:
                raise SystemExit(f"USB webcam not found at {dev}. Is it plugged in?  Check with:  "
                                 f"python -m tools.webcam_check --list")
            return False
        fourcc = str(c.get("usb_fourcc", "MJPG") or "")
        if len(fourcc) == 4:                       # set the format BEFORE the size (UVC rule)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(c["width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(c["height"]))
        cap.set(cv2.CAP_PROP_FPS, int(c.get("usb_fps", 30)))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)        # we pull every frame anyway; 2 avoids dropped frames
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        got = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)) if code > 0 else "?"
        self.info = {"device": dev, "width": w, "height": h, "fourcc": got,
                     "fps_set": round(float(cap.get(cv2.CAP_PROP_FPS)), 1)}
        if c.get("usb_controls"):
            webcam.apply_controls(dev, c["usb_controls"], verbose=self.verbose and first)
        for _ in range(int(c.get("usb_warmup_frames", 5))):   # first frames are often dark / half-exposed
            cap.grab()
        self.cam, self.device = cap, dev
        if self.verbose:
            print(f"[camera] USB webcam {dev}: {w}x{h} {got} @ {self.info['fps_set']} fps")
            if (w, h) != (int(c["width"]), int(c["height"])):
                print(f"[camera] WARNING: asked for {c['width']}x{c['height']} but the webcam gives {w}x{h} - "
                      f"set camera.width/height to a size it supports (python -m tools.webcam_check --formats) "
                      f"and re-check camera.roi")
        return True

    def _pull_usb(self):
        """-> (ok, decode_fn). grab() only takes the frame out of the driver; decode_fn() decodes it."""
        if self.cam is None:
            time.sleep(1.0)
            try:
                if self._open_usb():
                    print("[camera] webcam re-connected")
            except Exception as e:
                print(f"[camera] re-connect failed: {e}")
            return False, None
        if not self.cam.grab():
            return False, None
        return True, self._decode_usb

    def _decode_usb(self):
        ok, bgr = self.cam.retrieve()
        return self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB) if ok and bgr is not None else None

    def _drop_usb(self):
        print("[camera] webcam stopped sending frames - re-connecting (check the USB cable)")
        try:
            self.cam.release()
        except Exception:
            pass
        self.cam = None

    # ------------------------------------------------------------------ Pi camera
    def _pull_picam(self):
        arr = self.cam.capture_array()           # Picamera2 "RGB888" arrays are in B,G,R order
        return True, (lambda: np.ascontiguousarray(arr[..., ::-1]))

    # ------------------------------------------------------------------ frame thread
    def _loop(self, pull):
        fails, n, t_rate = 0, 0, time.perf_counter()
        while not self._stop:
            try:
                ok, decode = pull()
            except Exception as e:
                ok, decode = False, None
                if self.verbose:
                    print(f"[camera] read error: {e}")
            t = time.perf_counter()
            if not ok:
                fails += 1
                if self.source == "usb" and self.cam is not None and fails >= 5:
                    self._drop_usb()
                    fails = 0
                else:
                    time.sleep(0.01)
                continue
            fails = 0
            self.frames += 1
            n += 1
            if t - t_rate >= 2.0:
                self.fps, n, t_rate = n / (t - t_rate), 0, t
            with self._cond:
                needed = any(w <= t for w in self._wants)
            if needed:
                frame = decode()
                if frame is not None:
                    with self._cond:
                        self._frame, self._t_frame = frame, t
                        self._cond.notify_all()

    def frame_after(self, t_after: float, timeout: float = 1.0):
        """RGB frame pulled from the camera at or after time.perf_counter() value t_after (None on timeout)."""
        end = time.perf_counter() + timeout
        with self._cond:
            self._wants.append(t_after)
            try:
                while self._frame is None or self._t_frame < t_after:
                    left = end - time.perf_counter()
                    if left <= 0:
                        return None
                    self._cond.wait(left)
                return self._frame
            finally:
                self._wants.remove(t_after)

    def latest_frame(self, timeout: float = 2.0):
        """A fresh full RGB frame (the next one from the camera)."""
        if self.source == "mock":
            return None
        return self.frame_after(time.perf_counter(), timeout)

    def capture_box(self, t_trigger: float | None = None) -> Image.Image:
        delay = self.cfg.get("capture_delay_ms", 0) / 1000.0
        if self.source == "mock":
            if delay:
                time.sleep(delay)
            path, _ = self.mock_hw.current
            return Image.open(path).convert("RGB").resize((self.size, self.size))
        t_after = (time.perf_counter() if t_trigger is None else t_trigger) + delay
        frame = self.frame_after(t_after, timeout=1.0 + delay)
        if frame is None:
            raise RuntimeError("camera gave no picture within 1 s (USB webcam unplugged?)")
        return crop_roi(frame, self.roi, self.size)

    def close(self):
        self._stop = True
        if self._thread is not None:            # never release the camera while the thread is inside grab()
            self._thread.join(timeout=1.5)
        try:
            if self.source == "picamera2":
                self.cam.stop()
            elif self.source == "usb" and self.cam is not None:
                self.cam.release()
        except Exception:
            pass


def _import_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise SystemExit("OpenCV is missing. On the gateway laptop:  pip install opencv-python-headless\n"
                         "(the START_*.bat launchers install it by themselves)")
