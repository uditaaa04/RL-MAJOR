"""
USB webcam check and set-up (run on the gateway, in the project venv).

    python -m tools.webcam_check --list        # which cameras the gateway sees (USB webcam = "usb")
    python -m tools.webcam_check --formats     # sizes / formats / fps the webcam supports
    python -m tools.webcam_check --controls    # its settings (exposure, focus, white balance ...)
    python -m tools.webcam_check               # TEST: settings from config, fps, brightness, sharpness,
                                               #       saves webcam_full.jpg + webcam_crop.jpg
    python -m tools.webcam_check --serve       # LIVE VIEW in the laptop browser: http://conveyor-pi.local:8080
                                               #       aim the camera, set focus, find camera.roi

The live view draws the crop square (camera.roi) on the picture. Type other x, y, w, h values in the
page to try them, then copy the printed line into config/system.yaml.
"""
from __future__ import annotations

import argparse
import io
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw

from common.config import load_config, project_path
from gateway_node import webcam


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian (standard focus measure): higher = sharper. Vectorised, one pass."""
    g = gray.astype(np.float32)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def measure(frame: np.ndarray, roi) -> dict:
    x, y, w, h = [int(v) for v in roi]
    crop = frame[max(0, y):y + h, max(0, x):x + w]
    gray = crop @ np.array([0.299, 0.587, 0.114], np.float32)
    return {"brightness": float(gray.mean()), "sharpness": sharpness(gray),
            "clipped": float((gray > 250).mean())}


def hints(m: dict) -> list[str]:
    out = []
    if m["brightness"] < 70:
        out.append("DARK: add light (LED strip), or raise exposure_time_absolute / uncomment gain")
    elif m["brightness"] > 200 or m["clipped"] > 0.05:
        out.append("TOO BRIGHT: lower exposure_time_absolute, or move the light so it does not glare")
    if m["sharpness"] < 60:
        out.append("BLURRY (or an empty belt): put a box with printed text under the camera; change "
                   "focus_absolute (autofocus webcams), twist the lens ring (manual-focus webcams), "
                   "or raise the camera (fixed-focus webcams are sharp from ~30 cm)")
    return out or ["looks fine"]


def overlay(frame: np.ndarray, roi, text: str) -> Image.Image:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    x, y, w, h = [int(v) for v in roi]
    d.rectangle([x, y, x + w, y + h], outline=(34, 211, 238), width=3)
    d.line([x + w // 2 - 12, y + h // 2, x + w // 2 + 12, y + h // 2], fill=(34, 211, 238), width=2)
    d.line([x + w // 2, y + h // 2 - 12, x + w // 2, y + h // 2 + 12], fill=(34, 211, 238), width=2)
    d.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return img


def jpeg(img: Image.Image, q=80) -> bytes:
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=q)
    return b.getvalue()


# ----------------------------------------------------------------------------- commands
def cmd_list():
    if not webcam.is_linux():
        raise SystemExit("run this on the gateway")
    devs = webcam.list_devices()
    usb = [d for d in devs if d["usb"] and d["capture"]]
    print("video devices (the gateway's own ISP / CSI nodes also appear here - they are NOT the webcam):")
    for d in devs:
        kind = "USB  " if d["usb"] else "board"
        role = "picture " if d["capture"] else "metadata"
        mark = "   <-- webcam (use this)" if usb and d is usb[0] else ""
        print(f"  {d['dev']:13s} {kind} {role} {d['name'][:40]:40s}{mark}")
        if d["by_id"]:
            print(f"  {'':13s} fixed name: {d['by_id']}")
    if not usb:
        print("\nNO USB webcam found. Check:  lsusb   (should list the webcam, e.g. 'Logitech ... Webcam C270')\n"
              "  * plug it straight into the gateway (not through an unpowered hub), try the other USB port\n"
              "  * dmesg | tail -20   right after plugging in: look for 'uvcvideo' and 'Found UVC'")
    else:
        print(f"\nconfig camera.usb_device: auto  -> will use {webcam.find_device('auto')}")
    if webcam.have_v4l2_ctl():
        print("\n$ v4l2-ctl --list-devices")
        print(subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True).stdout)


def _dev(cfg):
    c = cfg["camera"]
    return webcam.find_device(c.get("usb_device", "auto"), int(c.get("usb_index", 0)))


def cmd_formats(cfg):
    if not webcam.have_v4l2_ctl():
        raise SystemExit("sudo apt install -y v4l-utils")
    dev = _dev(cfg)
    print(f"$ v4l2-ctl -d {dev} --list-formats-ext   (MJPG lines at 640x480 should say 30.000 fps)")
    print(subprocess.run(["v4l2-ctl", "-d", str(dev), "--list-formats-ext"], capture_output=True, text=True).stdout)


def cmd_controls(cfg):
    dev = _dev(cfg)
    ctrls = webcam.list_controls(dev)
    if not ctrls:
        raise SystemExit("no controls read - is v4l-utils installed (sudo apt install -y v4l-utils) and the webcam plugged in?")
    print(f"settings of {dev} (name  type  current  [min..max]):")
    for name, i in ctrls.items():
        rng = f"[{i.get('min')}..{i.get('max')}]" if "min" in i else ""
        print(f"  {name:30s} {i['type']:5s} {str(i.get('value', '')):>6} {rng}  {'(inactive now)' if i.get('flags') == 'inactive' else ''}")
    want = cfg["camera"].get("usb_controls") or {}
    print("\nconfig camera.usb_controls would set:")
    for name, value, note in webcam.plan_controls(want, ctrls):
        print(f"  {name:30s} -> {value if value is not None else '-':>6}  {note}")


def cmd_test(cfg, seconds: float):
    from gateway_node.camera import Camera
    cam = Camera(cfg)
    roi = cfg["camera"]["roi"]
    try:
        print(f"measuring for {seconds:.0f} s ...")
        f0, t0 = cam.frames, time.perf_counter()
        lat, t_end = [], t0 + seconds
        frame = None
        while time.perf_counter() < t_end:
            t = time.perf_counter()
            f = cam.frame_after(t, timeout=2.0)
            if f is None:
                raise SystemExit("no frames - webcam not sending (see --list)")
            lat.append(time.perf_counter() - t)
            frame = f
            time.sleep(0.05)
        fps = (cam.frames - f0) / (time.perf_counter() - t0)
        m = measure(frame, roi)
        H, W = frame.shape[:2]
        print(f"\n  device        {cam.info.get('device')}")
        print(f"  picture       {W}x{H}  format {cam.info.get('fourcc', '-')}")
        print(f"  frame rate    {fps:.1f} fps   (want ~30; 15 or less = exposure too long or YUYV format)")
        print(f"  photo wait    median {np.median(lat) * 1000:.0f} ms, max {max(lat) * 1000:.0f} ms "
              f"(time from 'IR fired' to the next frame)")
        print(f"  brightness    {m['brightness']:.0f} / 255 inside the crop  (good: 90-180)")
        print(f"  sharpness     {m['sharpness']:.0f}  (put a box with printed text in the crop; higher = sharper)")
        for h in hints(m):
            print(f"  -> {h}")
        overlay(frame, roi, "camera.roi").save(project_path("webcam_full.jpg"), quality=90)
        from common.vision_io import crop_roi
        crop_roi(frame, roi, cfg["camera"]["upload_size"]).save(project_path("webcam_crop.jpg"), quality=90)
        print("\nsaved webcam_full.jpg (whole picture, crop square in cyan) and webcam_crop.jpg (what the AI sees)")
        print("copy to the PC:  scp <user>@conveyor-pi.local:~/SmartConveyor_QL/webcam_*.jpg .")
    finally:
        cam.close()


def cmd_serve(cfg, port: int):
    from gateway_node.camera import Camera
    cam = Camera(cfg)
    state = {"roi": [int(v) for v in cfg["camera"]["roi"]]}
    lock = threading.Lock()
    page = f"""<!doctype html><title>Conveyor webcam</title>
<body style="font-family:sans-serif;background:#111;color:#eee;margin:16px">
<h3>Conveyor webcam - live</h3>
<img id=v src="/stream" style="max-width:100%;border:1px solid #444"><br><br>
camera.roi &nbsp; x <input id=x size=4> y <input id=y size=4> w <input id=w size=4> h <input id=h size=4>
<button onclick="go()">try</button>
<p id=line style="font-family:monospace"></p>
<p style="color:#aaa">Put a box on the belt where IR-1 fires. The box should sit in the middle of the cyan square,
filling most of it. When it looks right, paste the line above into config/system.yaml (camera: section).</p>
<script>
const r={state['roi']};['x','y','w','h'].forEach((k,i)=>document.getElementById(k).value=r[i]);
function go(){{const v=['x','y','w','h'].map(k=>parseInt(document.getElementById(k).value));
fetch('/roi?v='+v.join(',')).then(()=>show(v));}}
function show(v){{document.getElementById('line').textContent='  roi: ['+v.join(', ')+']';}} show(r);
</script>"""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                body = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/roi":
                try:
                    v = [int(s) for s in parse_qs(u.query)["v"][0].split(",")][:4]
                    if len(v) == 4 and v[2] > 8 and v[3] > 8:
                        with lock:
                            state["roi"] = v
                        print(f"  roi: [{v[0]}, {v[1]}, {v[2]}, {v[3]}]")
                except (KeyError, ValueError):
                    pass
                self.send_response(204)
                self.end_headers()
            elif u.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        f = cam.latest_frame()
                        if f is None:
                            continue
                        with lock:
                            roi = list(state["roi"])
                        m = measure(f, roi)
                        txt = (f"{cam.fps:4.1f} fps   brightness {m['brightness']:3.0f}   "
                               f"sharpness {m['sharpness']:5.0f}   roi {roi}")
                        data = jpeg(overlay(f, roi, txt))
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                        time.sleep(0.1)          # ~10 pictures/s is plenty for aiming; saves gateway CPU
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    srv.daemon_threads = True
    print(f"live view: open  http://conveyor-pi.local:{port}  (or http://<pi-ip>:{port}) on the PC.  Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        cam.close()


def main():
    ap = argparse.ArgumentParser(description="USB webcam check / set-up")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="list the cameras the gateway sees")
    g.add_argument("--formats", action="store_true", help="sizes / formats / frame rates of the webcam")
    g.add_argument("--controls", action="store_true", help="webcam settings and what the config would set")
    g.add_argument("--serve", action="store_true", help="live view in a browser (aim, focus, set the ROI)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()
    cfg = load_config()
    if cfg["camera"]["source"] != "usb" and not a.list:
        print(f"note: config camera.source is {cfg['camera']['source']!r}; set it to usb for the webcam")
    if a.list:
        cmd_list()
    elif a.formats:
        cmd_formats(cfg)
    elif a.controls:
        cmd_controls(cfg)
    elif a.serve:
        cmd_serve(cfg, a.port)
    else:
        cmd_test(cfg, a.seconds)


if __name__ == "__main__":
    main()
