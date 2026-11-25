"""
USB webcam helpers for the gateway (the fine controls need Linux V4L2; on Windows the webcam's own defaults are used).

    find_device("auto")        -> "/dev/video0"   the first USB camera (never the gateway 5's own
                                                  ISP / CSI nodes, which also appear as /dev/videoN)
    list_devices()             -> every /dev/videoN with its name and whether it is USB
    list_controls(dev)         -> the webcam's settings (exposure, focus, white balance, ...)
    apply_controls(dev, {...}) -> set them with v4l2-ctl (package v4l-utils), in the given order

Why the controls matter on a conveyor: a webcam left on AUTO exposure makes the exposure long in
a dim room (motion blur, and the frame rate drops from 30 to 15 fps), auto-focus "hunts" every
time a box enters, and auto white balance changes the box colour from photo to photo. All three
make the AI less accurate, so camera.usb_controls in config/system.yaml locks them.
Kernel 5.x and 6.x webcams use different names for the same setting; both are accepted.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys

# new (kernel >= 5.x) name -> old name of the same UVC setting
ALIASES = {
    "auto_exposure": "exposure_auto",
    "exposure_time_absolute": "exposure_absolute",
    "exposure_dynamic_framerate": "exposure_auto_priority",
    "white_balance_automatic": "white_balance_temperature_auto",
    "focus_automatic_continuous": "focus_auto",
}
_CTRL_RE = re.compile(r"^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*)$")


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_windows() -> bool:
    return sys.platform.startswith("win")


def list_devices() -> list[dict]:
    """All V4L2 video nodes: dev, name, usb (bool), capture (bool: index 0 = picture, 1 = metadata)."""
    out = []
    for sysdir in glob.glob("/sys/class/video4linux/video*"):
        node = os.path.basename(sysdir)
        if not node[5:].isdigit():
            continue

        def read(name, default=""):
            try:
                with open(os.path.join(sysdir, name)) as f:
                    return f.read().strip()
            except OSError:
                return default
        real = os.path.realpath(os.path.join(sysdir, "device"))
        out.append({"dev": f"/dev/{node}", "num": int(node[5:]), "name": read("name"),
                    "usb": "/usb" in real, "capture": read("index", "0") == "0"})
    out.sort(key=lambda d: d["num"])
    by_id = {os.path.realpath(p): p for p in glob.glob("/dev/v4l/by-id/*")}
    for d in out:
        d["by_id"] = by_id.get(d["dev"], "")
    return out


def find_device(spec="auto", index: int = 0):
    """What to give cv2.VideoCapture.

    spec "auto"            first USB webcam picture node (Linux), else `index`
    spec "/dev/..."        that device (a /dev/v4l/by-id/... name survives re-plugging)
    spec 0, 1, "2" ...     /dev/video<N> on Linux, camera number N elsewhere
    """
    if spec is None or str(spec).strip().lower() in ("", "auto"):
        if not is_linux():
            return int(index)
        usb = [d for d in list_devices() if d["usb"] and d["capture"]]
        if usb:
            return usb[0]["dev"]
        return f"/dev/video{int(index)}"
    s = str(spec).strip()
    if s.isdigit():
        return f"/dev/video{s}" if is_linux() else int(s)
    return os.path.realpath(s) if is_linux() else s


def have_v4l2_ctl() -> bool:
    return shutil.which("v4l2-ctl") is not None


def parse_controls(text: str) -> dict:
    """Parse `v4l2-ctl --list-ctrls` output -> {name: {"type": "int", "min": 0, "value": 128, ...}}."""
    ctrls = {}
    for line in text.splitlines():
        m = _CTRL_RE.match(line)
        if not m:
            continue
        name, typ, rest = m.groups()
        info = {"type": typ}
        for k, v in re.findall(r"(\w+)=(-?\w+)", rest):
            try:
                info[k] = int(v)
            except ValueError:
                info[k] = v
        ctrls[name] = info
    return ctrls


def list_controls(dev) -> dict:
    if not (is_linux() and have_v4l2_ctl()):
        return {}
    r = subprocess.run(["v4l2-ctl", "-d", str(dev), "--list-ctrls"], capture_output=True, text=True, timeout=5)
    return parse_controls(r.stdout)


def resolve_name(name: str, available: dict) -> str | None:
    """The name this webcam uses for a setting (new or old kernel spelling), or None."""
    if name in available:
        return name
    old = ALIASES.get(name)
    if old and old in available:
        return old
    new = next((n for n, o in ALIASES.items() if o == name), None)
    if new and new in available:
        return new
    return None


def plan_controls(wanted: dict, available: dict) -> list[tuple[str, int | None, str]]:
    """Decide what to set: [(name, value or None, note)]. Values are clamped to the webcam's range.
    Order is kept (set 'auto_exposure: 1' BEFORE 'exposure_time_absolute', and so on)."""
    plan = []
    for want_name, value in (wanted or {}).items():
        name = resolve_name(want_name, available)
        if name is None:
            plan.append((want_name, None, "not on this webcam - skipped"))
            continue
        info = available[name]
        v = int(value)
        lo, hi = info.get("min"), info.get("max")
        if info.get("type") == "bool":
            lo, hi = 0, 1
        note = ""
        if isinstance(lo, int) and v < lo:
            v, note = lo, f"(raised to min {lo})"
        if isinstance(hi, int) and v > hi:
            v, note = hi, f"(lowered to max {hi})"
        plan.append((name, v, note))
    return plan


def apply_controls(dev, wanted: dict, verbose: bool = True) -> list[str]:
    """Set the webcam controls with v4l2-ctl. Returns one report line per control."""
    if not wanted:
        return []
    if not is_linux():
        return ["(webcam controls are only set on Linux; on Windows use the webcam's own settings app)"]
    if not have_v4l2_ctl():
        return ["v4l2-ctl not found - sudo apt install -y v4l-utils (controls left on auto)"]
    report = []
    for name, value, note in plan_controls(wanted, list_controls(dev)):
        if value is None:
            report.append(f"  {name:28s} {note}")
            continue
        r = subprocess.run(["v4l2-ctl", "-d", str(dev), "-c", f"{name}={value}"],
                           capture_output=True, text=True, timeout=5)
        ok = r.returncode == 0
        report.append(f"  {name:28s} = {value:<6} {'OK' if ok else 'FAILED: ' + r.stderr.strip()[:80]} {note}")
    if verbose:
        print(f"[camera] webcam settings on {dev}:")
        print("\n".join(report))
    return report
