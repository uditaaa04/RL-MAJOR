"""
The gateway's side of the USB cable to the ESP32 edge node.

  Esp32Link      - the serial port: finds the board, loads the tiny model into it, sends photos,
                   receives answers and belt events. Re-connects by itself if the cable is pulled
                   or the board resets.
  Esp32Hardware  - the belt as the rest of the gateway sees it (same methods as the pretend belt):
                   belt speed, reject arm, "box at camera" / "box at arm" events.
  EdgeTiny       - "check this photo on the ESP32", one photo at a time. Without a board it runs the
                   same tiny model on the gateway and waits as long as the cable and the ESP32 would
                   need (clearly marked as emulated in the logs and on the dashboard).

Protocol: see the top of esp32_node/esp32_node.ino.
"""
from __future__ import annotations

import queue
import threading
import time

from common import tiny

USB_HINTS = ("cp210", "ch340", "ch910", "ch343", "usb-serial", "usb serial", "silicon labs", "wch", "uart", "ftdi")


class Esp32Error(Exception):
    pass


def list_ports() -> list[tuple[str, str]]:
    """[(port, description)] of USB serial adapters, most likely ESP32 boards first.
    Bluetooth and built-in ports are left out: opening those can hang for many seconds on Windows."""
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        return []
    ports = [(p.device, f"{p.description} {p.hwid}") for p in lp.comports()
             if p.vid is not None or any(h in f"{p.description} {p.hwid}".lower() for h in USB_HINTS)]
    return sorted(ports, key=lambda p: (not any(h in p[1].lower() for h in USB_HINTS), p[0]))


class Esp32Link:
    def __init__(self, port: str, baud: int = 460800, verbose: bool = True, opener=None):
        self.port, self.baud, self.verbose = port, int(baud), verbose
        self._opener = opener              # tests: a function returning a file-like serial object
        self.ser = None
        self.on_event = None               # fn(tag, t_received): tag "C" (camera) or "A" (arm)
        self.on_reset = None               # fn(): the board restarted (model and settings are re-sent first)
        self._wlock = threading.Lock()
        self._rlock = threading.Lock()     # one question-and-answer at a time
        self._replies: queue.Queue = queue.Queue()
        self._pending: dict[int, list] = {}
        self._plock = threading.Lock()
        self._next_id = 1
        self._stop = False
        self._model_blob = None
        self._settings: dict[str, int] = {}
        self.duty = 0                      # belt speed 0..255, repeated in every keep-alive
        self.model_loaded = False
        self.connected = False
        self.last_rx = 0.0
        self.firmware = ""
        self.resets = 0                    # times the board restarted while we were connected
        self.reinits = 0                   # times it was set up again after that
        self._threads = []

    # ------------------------------------------------------------------ open / close
    def _open_port(self):
        if self._opener is not None:
            return self._opener()
        import serial
        s = serial.Serial()
        # write_timeout: Windows cuts a write off after this long. The tiny model is about 47 kB and needs a
        # full second on the cable at 460800 baud, so 1 s would cut it off right at the end. 6 s is safe.
        s.port, s.baudrate, s.timeout, s.write_timeout = self.port, self.baud, 0.1, 6.0
        s.dtr = False                      # do not reset the board just by opening the port
        s.rts = False
        s.open()
        return s

    def open(self, hello_timeout: float = 5.0):
        self.ser = self._open_port()
        self._stop = False
        for fn, name in ((self._reader, "esp32-rx"), (self._keepalive, "esp32-keepalive")):
            t = threading.Thread(target=fn, daemon=True, name=name)
            t.start()
            self._threads.append(t)
        if not self.hello(hello_timeout):
            self.close()
            raise Esp32Error(f"no answer from an ESP32 on {self.port} at {self.baud} baud. Is the firmware "
                             f"flashed (esp32_node/esp32_node.ino) and the Serial Monitor closed?")
        self.connected = True
        return self

    def hello(self, timeout: float = 5.0) -> bool:
        t_end = time.time() + timeout
        while time.time() < t_end:
            try:
                r = self.request("H", "H SCQC-ESP32", 0.5)
            except Esp32Error:
                continue
            self.firmware = r
            return True
        return False

    def close(self):
        self._stop = True
        try:
            if self.ser is not None:
                with self._wlock:
                    self.ser.write(b"K 0\n")               # belt off before the port closes
                    self.ser.flush()
        except Exception:
            pass
        time.sleep(0.05)
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass
        self.connected = False

    # ------------------------------------------------------------------ sending
    def send(self, line: str, payload: bytes = b""):
        if self.ser is None:
            raise Esp32Error("port closed")
        try:
            with self._wlock:              # a photo and its header must never be mixed with another line
                self.ser.write(line.encode("ascii") + b"\n" + payload)
                self.ser.flush()
        except Exception as e:
            self.connected = False
            raise Esp32Error(f"write failed: {e}") from e

    def request(self, line: str, prefix: str, timeout: float = 1.0, payload: bytes = b"") -> str:
        """Send a line and wait for the reply that starts with `prefix`."""
        with self._rlock:
            while not self._replies.empty():
                try:
                    self._replies.get_nowait()
                except queue.Empty:
                    break
            self.send(line, payload)
            t_end = time.time() + timeout
            while True:
                left = t_end - time.time()
                if left <= 0:
                    raise Esp32Error(f"no reply to '{line.split()[0]}'")
                try:
                    r = self._replies.get(timeout=left)
                except queue.Empty:
                    raise Esp32Error(f"no reply to '{line.split()[0]}'") from None
                if r.startswith(prefix):
                    return r

    def upload_model(self, blob: bytes):
        r = self.request(f"W {len(blob)} {tiny.checksum(blob)}", "W ", 4.0, payload=blob)
        if not r.startswith("W OK"):
            self.model_loaded = False
            raise Esp32Error(f"the ESP32 refused the tiny model: {r}")
        self._model_blob = blob
        self.model_loaded = True

    def configure(self, **settings):
        """servo_rest, servo_push, ir_low, debounce_ms (kept and sent again after a board reset)."""
        for k, v in settings.items():
            self._settings[k] = int(v)
            self.request(f"C {k} {int(v)}", "OK", 1.0)

    def infer(self, px: bytes, timeout: float = 0.25) -> dict:
        """One shrunken photo -> {"p": chance damaged, "esp_us": compute time on the board, "rx_ms": receive time}."""
        if not self.model_loaded:
            raise Esp32Error("no model on the board")
        with self._plock:
            rid = self._next_id
            self._next_id = self._next_id % 60000 + 1
            slot = [threading.Event(), None]
            self._pending[rid] = slot
        try:
            self.send(f"I {rid} {len(px)} {tiny.checksum(px)}", px)
            if not slot[0].wait(timeout):
                raise Esp32Error("timeout")
            parts = slot[1]
            if len(parts) < 2 or parts[0] == "ERR":
                raise Esp32Error(" ".join(parts) or "bad reply")
            return {"p": int(parts[0]) / 10000.0, "esp_us": int(parts[1]), "rx_ms": int(parts[2]) if len(parts) > 2 else 0}
        finally:
            with self._plock:
                self._pending.pop(rid, None)

    def set_duty(self, duty: int):
        self.duty = max(0, min(255, int(duty)))
        try:
            self.send(f"K {self.duty}")
        except Esp32Error:
            pass

    def push(self, hold_ms: int = 350):
        try:
            self.send(f"P {int(hold_ms)}")
        except Esp32Error as e:
            if self.verbose:
                print(f"[esp32] push not sent: {e}")

    def status(self) -> dict | None:
        try:
            p = self.request("S", "S ", 0.5).split()
            return {"ir_camera": p[1] == "1", "ir_arm": p[2] == "1", "duty": int(p[3]), "free_bytes": int(p[4]),
                    "uptime_ms": int(p[5])}
        except (Esp32Error, IndexError, ValueError):
            return None

    @property
    def alive(self) -> bool:
        return self.connected and (time.time() - self.last_rx) < 3.5

    # ------------------------------------------------------------------ background threads
    def _keepalive(self):
        """Every second: 'still here' + the belt speed. The board stops the belt after 3 s of silence."""
        while not self._stop:
            time.sleep(1.0)
            if self._stop or self.ser is None:
                break
            try:
                self.send(f"K {self.duty}")
            except Esp32Error:
                pass

    def _reader(self):
        buf = b""
        while not self._stop:
            try:
                chunk = self.ser.read(256)
            except Exception:
                chunk = None
            if chunk is None:                       # cable pulled / port vanished
                self.connected = False
                self.model_loaded = False
                if self._stop:
                    break
                time.sleep(1.0)
                try:
                    self.ser.close()
                except Exception:
                    pass
                try:
                    self.ser = self._open_port()
                    if self.verbose:
                        print(f"[esp32] port {self.port} re-opened")
                    threading.Thread(target=self._after_reset, daemon=True).start()
                except Exception:
                    pass
                continue
            if not chunk:
                continue
            self.last_rx = time.time()
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("ascii", "ignore").strip()
                if line:
                    self._dispatch(line)
            if len(buf) > 4096:                     # noise without line ends (e.g. boot messages at another speed)
                buf = b""

    def _dispatch(self, line: str):
        if line.startswith("E "):
            parts = line.split()
            if len(parts) >= 2 and parts[1] in ("C", "A") and self.on_event:
                try:
                    self.on_event(parts[1], time.perf_counter())
                except Exception as e:
                    print(f"[esp32] event handler error: {e}")
        elif line.startswith("R "):
            parts = line.split()
            try:
                rid = int(parts[1])
            except (IndexError, ValueError):
                return
            with self._plock:
                slot = self._pending.get(rid)
            if slot is not None:
                slot[1] = parts[2:]
                slot[0].set()
        elif line.startswith("BOOT SCQC-ESP32"):
            self.model_loaded = False
            if self.connected:                      # a reset in the middle of a run (power dip, button)
                self.resets += 1
                threading.Thread(target=self._after_reset, daemon=True).start()
        else:
            self._replies.put(line)

    def _after_reset(self):
        """The board lost its model and settings: send them again, then tell the owner."""
        try:
            time.sleep(0.2)
            if not self.hello(4.0):
                return
            self.connected = True
            for k, v in self._settings.items():
                self.request(f"C {k} {v}", "OK", 1.0)
            if self._model_blob is not None:
                self.upload_model(self._model_blob)
            self.send(f"K {self.duty}")
            self.reinits += 1
            if self.verbose:
                print("[esp32] board restarted - model and settings sent again")
            if self.on_reset:
                self.on_reset()
        except Esp32Error as e:
            if self.verbose:
                print(f"[esp32] could not set the board up again: {e}")


def connect(cfg: dict, model_blob: bytes | None, port: str | None = None, verbose: bool = True) -> Esp32Link:
    """Find the board (port 'auto' tries every USB serial port), set it up and load the tiny model."""
    e = cfg.get("esp32", {})
    want = port or str(e.get("port", "auto"))
    baud = int(e.get("baud", 460800))
    tried = []
    cands = [want] if want.lower() != "auto" else [p for p, _ in list_ports()]
    if not cands:
        raise Esp32Error("no USB serial port found. Is the ESP32 plugged in with a data cable? "
                         "(If it is, install the CP210x or CH340 USB driver.)")
    for p in cands:
        link = Esp32Link(p, baud, verbose)
        try:
            link.open(hello_timeout=5.0 if len(cands) == 1 else 3.0)
        except Exception as ex:
            tried.append(f"{p}: {ex}")
            continue
        link.configure(servo_rest=e.get("servo_rest_deg", 0), servo_push=e.get("servo_push_deg", 80),
                       ir_low=1 if e.get("ir_active_low", True) else 0, debounce_ms=e.get("debounce_ms", 20))
        if model_blob is not None:
            try:
                link.upload_model(model_blob)
            except Esp32Error as ex:
                # the belt, the sensors and the arm do not need the tiny model: carry on without it
                print(f"[esp32] WARNING: {ex}. The belt, sensors and arm still work; no photo will be checked on "
                      f"the ESP32 in this run. Unplug the board, plug it in again and restart to retry.")
        if verbose:
            print(f"[esp32] connected on {p}: {link.firmware}" + ("  (tiny model loaded)" if link.model_loaded else ""))
        return link
    raise Esp32Error("no ESP32 answered. Tried " + "; ".join(tried))


class Esp32Hardware:
    """The real belt, driven through the ESP32."""

    def __init__(self, cfg: dict, link: Esp32Link):
        self.cfg, self.link = cfg, link
        self.duty = float(cfg["conveyor"]["belt_duty"])
        # False = there is no IR sensor at the reject arm; the gateway fires the arm by time instead
        self.arm_sensor = bool(cfg["conveyor"].get("arm_sensor", True))
        self.hold_ms = int(cfg.get("esp32", {}).get("servo_hold_ms", 350))
        self.truth = None                  # real boxes: ground truth comes from an experiment sequence file
        self.on_cam = self.on_arm = None
        self._running = False
        link.on_event = self._event
        link.on_reset = self._resend

    def bind(self, on_box_at_camera, on_box_at_arm):
        self.on_cam, self.on_arm = on_box_at_camera, on_box_at_arm

    def _event(self, tag: str, t: float):
        # each event gets its own short thread: taking the photo must never delay the arm
        if tag != "C" and not self.arm_sensor:
            return                         # nothing is wired to the arm-sensor pin: ignore whatever it reports
        fn = self.on_cam if tag == "C" else self.on_arm
        if fn is not None:
            threading.Thread(target=fn, daemon=True, name=f"ir-{tag}").start()

    def _resend(self):
        if self._running:
            self.link.set_duty(round(self.duty * 255))

    def start_belt(self, duty: float | None = None):
        if duty is not None:
            self.duty = max(0.0, min(1.0, float(duty)))
        self._running = True
        self.link.set_duty(round(self.duty * 255))

    def set_belt(self, duty: float):
        self.start_belt(duty)

    def stop_belt(self):
        self._running = False
        self.link.set_duty(0)

    def belt_speed_mps(self) -> float:
        return self.duty * float(self.cfg["conveyor"]["speed_mps_at_full"])

    def push(self):
        self.link.push(self.hold_ms)

    def show(self, verdict: str):
        """The belt has no good / damaged lamps (firmware version 2): the dashboard shows every verdict."""

    def power_w(self):
        return None

    def close(self):
        try:
            self.stop_belt()
        except Exception:
            pass


class EdgeTiny:
    """Checks photos on the ESP32, one at a time (first come, first served)."""

    NS_PER_MULT = 40e-9                    # pretend ESP32 only: about 40 ns per multiply-add at 240 MHz

    def __init__(self, model: tiny.TinyModel, link: Esp32Link | None, timeout_s: float = 0.25, baud: int = 460800):
        self.model, self.link = model, link
        self.emulated = link is None
        self.timeout = float(timeout_s)
        # time on the cable: 10 bits per byte, header + photo up, short answer down, USB hand-over
        self.link_s = (len(model.pixels_blank()) + 40) * 10.0 / baud + 0.002
        self.compute_s = model.mults * self.NS_PER_MULT + 0.001
        self.ewma_ms = (self.link_s + self.compute_s) * 1000.0
        self.last_esp_ms = self.compute_s * 1000.0
        self.failures = 0
        self.q: queue.Queue = queue.Queue()
        self._pending = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True, name="edge-esp32").start()

    @property
    def ready(self) -> bool:
        return self.emulated or (self.link.alive and self.link.model_loaded)

    @property
    def queue_len(self) -> int:
        with self._lock:
            return self._pending

    def measure(self, n: int = 5):
        """Real board only: time a few blank photos so the first decisions already know its speed."""
        if self.emulated:
            return
        px = self.model.pixels_blank()
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            try:
                r = self.link.infer(px, 1.0)
            except Esp32Error:
                continue
            ts.append(time.perf_counter() - t)
            self.last_esp_ms = r["esp_us"] / 1000.0
        if ts:
            self.ewma_ms = 1000.0 * sorted(ts)[len(ts) // 2]
            # a board slower than estimated must not fail every photo: allow 2.5 x what was just measured
            need = 2.5 * max(ts)
            if need > self.timeout:
                print(f"[esp32] this board needs about {1000 * max(ts):.0f} ms per photo: answer time limit raised "
                      f"from {1000 * self.timeout:.0f} to {1000 * need:.0f} ms")
                self.timeout = need
        else:
            print("[esp32] WARNING: the board answered none of the test photos. It will not be used to check boxes "
                  "until it answers (the belt, sensors and arm still work).")

    def submit(self, box_id: int, img, callback):
        with self._lock:
            self._pending += 1
        self.q.put((box_id, img, callback))

    def check(self, img) -> dict:
        """Blocking: one photo -> result dict (also used by the speed-accuracy measurement)."""
        t = time.perf_counter()
        px = self.model.pixels(img)
        if self.emulated:
            p = self.model.predict_pixels(px)
            left = self.link_s + self.compute_s - (time.perf_counter() - t)
            if left > 0:
                time.sleep(left)
            esp_ms = self.compute_s * 1000.0
        else:
            try:
                r = self.link.infer(px, self.timeout)
            except Esp32Error as e:
                self.failures += 1
                return {"ok": False, "error": f"esp32: {e}", "where": "EDGE", "compute_s": time.perf_counter() - t}
            p, esp_ms = r["p"], r["esp_us"] / 1000.0
        svc = time.perf_counter() - t
        self.ewma_ms = 0.7 * self.ewma_ms + 0.3 * svc * 1000.0
        self.last_esp_ms = esp_ms
        return {"ok": True, "p_damaged": p, "heatmap": None, "compute_s": svc, "where": "EDGE",
                "esp_ms": round(esp_ms, 2), "emulated": self.emulated}

    def _loop(self):
        while True:
            box_id, img, cb = self.q.get()
            try:
                res = self.check(img)
            except Exception as e:                  # never let one bad photo stop the worker
                res = {"ok": False, "error": f"esp32: {e}", "where": "EDGE"}
            with self._lock:
                self._pending -= 1
            cb(box_id, res)
