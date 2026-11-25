"""
See what the gateway's camera sees (double-click CHECK_CAMERA.bat; nothing to type first).

    python -m tools.camera_check             # every camera on this laptop, one picture each
    python -m tools.camera_check --no-ask    # only make the page, change nothing

What it does
  1. Finds the cameras on this laptop (the laptop's own camera and every plugged-in webcam).
  2. Takes ONE picture from each, with the same settings the real line uses.
     The pictures stay on this PC, in logs/camera_check/ (that folder is never uploaded to GitHub).
  3. Opens a page that shows every camera with its number, marks the one the real line uses,
     draws the crop square (camera.roi) on it and shows what the AI would get.
  4. Asks which camera looks at the belt and which crop square to use, and writes the answers
     into config/system.yaml (camera.usb_device and camera.roi). Just Enter = change nothing.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import webbrowser

import numpy as np
from PIL import Image

from common.config import DEFAULT_PATH, load_config, project_path
from common.vision_io import crop_roi
from gateway_node import webcam

OUT_DIR = os.path.join("logs", "camera_check")
SETTLE_S = 1.5                             # how long a camera runs before its picture is taken


def say(msg=""):
    print(msg, flush=True)                 # plain ASCII only: a Windows console may not print anything else


# ----------------------------------------------------------------------------- measuring one picture
def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian (standard focus measure): higher = sharper."""
    g = gray.astype(np.float32)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def measure(frame: np.ndarray, roi) -> dict:
    """Brightness (0-255), sharpness and the share of burnt-out pixels inside the crop square."""
    x, y, w, h = [int(v) for v in roi]
    crop = frame[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return {"brightness": 0.0, "sharpness": 0.0, "clipped": 0.0}
    gray = crop @ np.array([0.299, 0.587, 0.114], np.float32)
    return {"brightness": float(gray.mean()), "sharpness": sharpness(gray), "clipped": float((gray > 250).mean())}


def advice(m: dict) -> list[dict]:
    """Plain-word hints. bad = True is shown as an alert."""
    out = []
    if m["brightness"] < 70:
        out.append({"bad": True, "text": "Too dark. Put a lamp or an LED strip over the belt. "
                                         "A dark photo is also a blurred photo when the belt moves."})
    elif m["brightness"] > 200 or m["clipped"] > 0.05:
        out.append({"bad": True, "text": "Too bright, parts are burnt out white. Move the lamp so it does not shine "
                                         "straight back into the camera."})
    elif m["brightness"] < 90:
        out.append({"bad": False, "text": "A little dark. It works, but more light gives sharper photos when the "
                                          "belt moves."})
    if m["sharpness"] < 60:
        out.append({"bad": True, "text": "Blurred, or nothing with detail is in the square. Put a box with print on "
                                         "it under the camera. Still blurred? Move the camera further away "
                                         "(most webcams are sharp from about 30 cm) or turn its lens ring."})
    if not any(a["bad"] for a in out) and not out:
        out.append({"bad": False, "text": "Light and sharpness look fine."})
    return out


def fit_roi(roi, width: int, height: int) -> list[int]:
    """The crop square moved and shrunk so that it lies inside the picture."""
    x, y, w, h = [int(v) for v in roi]
    s = max(16, min(w, h, width, height))
    return [max(0, min(x, width - s)), max(0, min(y, height - s)), s, s]


# ----------------------------------------------------------------------------- taking the pictures
def list_cameras(cfg, cv2) -> list:
    """What can be given to the camera code: numbers on Windows / macOS, the USB webcam's node on Linux."""
    if webcam.is_linux():
        c = cfg["camera"]
        try:
            return [webcam.find_device(c.get("usb_device", "auto"), int(c.get("usb_index", 0)))]
        except Exception:
            return []
    return webcam.list_cameras(cv2)


def shoot(cfg, dev, settle_s: float | None = None):
    """One picture from camera `dev`, opened exactly like the real line opens it.
    -> (frame or None, info dict, frames per second, error text)"""
    from gateway_node.camera import Camera
    c = copy.deepcopy(cfg)
    c["camera"]["source"] = "usb"
    c["camera"]["usb_device"] = dev
    try:
        cam = Camera(c, verbose=False)
    except SystemExit as e:
        return None, {}, 0.0, str(e)
    except Exception as e:
        return None, {}, 0.0, f"could not be opened: {e}"
    try:
        f0, t0 = cam.frames, time.perf_counter()
        time.sleep(SETTLE_S if settle_s is None else settle_s)   # let the camera's automatic brightness settle
        frame = cam.latest_frame(3.0)
        fps = (cam.frames - f0) / max(1e-6, time.perf_counter() - t0)
        if frame is None:
            return None, dict(cam.info), fps, "opened, but it sent no picture (is another program using it?)"
        return frame, dict(cam.info), fps, ""
    finally:
        cam.close()


# ----------------------------------------------------------------------------- writing config/system.yaml
def set_yaml_value(text: str, section: str, key: str, value: str) -> str:
    """Change `key:` inside the top-level `section:` of a YAML text, keeping every comment and every other line.
    Raises ValueError if the line is not found."""
    lines = text.split("\n")
    inside = False
    for i, raw in enumerate(lines):
        eol = "\r" if raw.endswith("\r") else ""           # a file saved by a Windows editor: keep its line ends
        ln = raw[:-1] if eol else raw
        if re.match(r"^[A-Za-z_][\w-]*\s*:", ln):          # a top-level key
            inside = ln.split(":")[0].strip() == section
            continue
        if not inside:
            continue
        m = re.match(r"^([ \t]+" + re.escape(key) + r"[ \t]*:[ \t]*)([^#]*?)([ \t]*)(#.*)?$", ln)
        if m:
            comment = m.group(4) or ""
            lines[i] = m.group(1) + value + ((m.group(3) or " ") + comment if comment else "") + eol
            return "\n".join(lines)
    raise ValueError(f"{section}.{key} not found")


def save_settings(path: str, device=None, roi=None) -> list[str]:
    """Write camera.usb_device and / or camera.roi. The file is checked after the change and put back
    as it was if it no longer reads correctly. -> lines saying what was changed"""
    import yaml
    with open(path, encoding="utf-8", newline="") as f:      # newline="": line ends are read and written as they are
        before = f.read()
    text, done = before, []
    if device is not None:
        text = set_yaml_value(text, "camera", "usb_device", str(device))
        done.append(f"camera.usb_device: {device}")
    if roi is not None:
        text = set_yaml_value(text, "camera", "roi", "[" + ", ".join(str(int(v)) for v in roi) + "]")
        done.append("camera.roi: [" + ", ".join(str(int(v)) for v in roi) + "]")
    if not done or text == before:
        return []
    old, new = yaml.safe_load(before), yaml.safe_load(text)
    if device is not None:
        old["camera"]["usb_device"] = new["camera"]["usb_device"]
    if roi is not None:
        old["camera"]["roi"] = new["camera"]["roi"]
    ok = (old == new and (device is None or str(new["camera"]["usb_device"]) == str(device))
          and (roi is None or [int(v) for v in new["camera"]["roi"]] == [int(v) for v in roi]))
    if not ok:
        raise ValueError("the settings file would not read correctly after the change - nothing was changed")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return done


# ----------------------------------------------------------------------------- the page
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera check</title>
<style>
:root{--bg:#ffffff;--ink:#0a0a0a;--soft:#4a4f55;--line:#d9dde1;--panel:#f4f6f7;--cyan:#008ea6;--red:#d0162b}
:root[data-theme=dark]{--bg:#000000;--ink:#ffffff;--soft:#a9b0b6;--line:#2a2f34;--panel:#0d1012;--cyan:#22d3ee;--red:#ff4d5e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 "Segoe UI",system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
header{display:flex;align-items:baseline;justify-content:space-between;gap:16px;border-bottom:3px solid var(--ink);padding-bottom:12px}
h1{font-size:34px;line-height:1.1;margin:0;letter-spacing:-.5px}
h2{font-size:22px;margin:0}
p{margin:8px 0;max-width:72ch}
.when{color:var(--soft)}
button{font:inherit;color:var(--ink);background:transparent;border:2px solid var(--ink);padding:6px 14px;cursor:pointer}
button:focus-visible,input:focus-visible{outline:3px solid var(--cyan);outline-offset:2px}
.used{display:inline-block;background:var(--cyan);color:var(--bg);font-weight:600;padding:1px 10px;margin-left:10px;font-size:15px}
.cam{border-top:1px solid var(--line);padding:26px 0}
.cam.inuse{border-top:3px solid var(--cyan)}
.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,1fr);gap:22px;margin-top:14px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.shot{position:relative;line-height:0;background:var(--panel);overflow:hidden}
.shot img{width:100%;height:auto;display:block}
.sq{position:absolute;border:3px solid var(--cyan);box-shadow:0 0 0 2000px rgba(0,0,0,.35);pointer-events:none}
canvas{width:100%;max-width:256px;height:auto;border:1px solid var(--line);background:var(--panel)}
table{border-collapse:collapse;margin-top:10px}
td{padding:3px 14px 3px 0;vertical-align:top}
td:first-child{color:var(--soft)}
.alert{color:var(--red);font-weight:600}
.ok{color:var(--soft)}
.nums{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin-top:14px}
label{display:block;color:var(--soft);font-size:14px}
input{font:inherit;width:92px;padding:5px 8px;color:var(--ink);background:var(--bg);border:1px solid var(--soft)}
code{font:15px Consolas,"Courier New",monospace;background:var(--panel);padding:3px 8px;border:1px solid var(--line)}
ol{padding-left:22px;max-width:72ch}
li{margin:6px 0}
</style></head>
<body><main>
<header><div><h1>Camera check</h1><div class="when">Pictures taken __WHEN__. They are stored on this PC only.</div></div>
<button id="mode" type="button">Dark</button></header>
<p id="lead"></p>
<div id="cams"></div>
<section class="cam">
<h2>What to do now</h2>
<ol>
<li>Find the picture that shows the belt. Remember its camera number.</li>
<li>In that picture, change the three numbers until the square is filled by the box, with the whole box inside.</li>
<li>Go back to the black window. Type the camera number, press Enter. Type the three numbers, press Enter.</li>
<li>Run CHECK_CAMERA again after you move the camera or the lamp.</li>
</ol>
</section>
</main>
<script>
const D = __DATA__;
const root = document.documentElement, btn = document.getElementById('mode');
function setTheme(t){ root.dataset.theme = t; btn.textContent = t === 'dark' ? 'Light' : 'Dark'; try{ localStorage.setItem('scqc-theme', t); }catch(e){} }
let saved = null; try{ saved = localStorage.getItem('scqc-theme'); }catch(e){}
setTheme(saved || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
btn.onclick = () => setTheme(root.dataset.theme === 'dark' ? 'light' : 'dark');

const lead = document.getElementById('lead');      // (not "top": that name belongs to the browser window)
if (!D.cams.length) { lead.className = 'alert'; lead.textContent = 'No camera was found on this laptop. Plug the webcam in and run CHECK_CAMERA again.'; }
else if (D.used === null) { lead.className = 'alert'; lead.textContent = 'The real line would not find its camera with the settings as they are now.'; }
else lead.textContent = 'This laptop has ' + D.cams.length + (D.cams.length === 1 ? ' camera' : ' cameras') + '. The real line uses camera ' + D.used + ' (' + D.why + ').';

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const host = document.getElementById('cams');
D.cams.forEach(c => {
  const sec = el('section', 'cam' + (c.used ? ' inuse' : ''));
  const h = el('h2', '', 'Camera ' + c.dev);
  if (c.used) h.appendChild(el('span', 'used', 'the real line uses this one'));
  sec.appendChild(h);
  if (!c.ok) { sec.appendChild(el('p', 'alert', 'No picture: ' + c.error)); host.appendChild(sec); return; }
  const grid = el('div', 'grid'), left = el('div'), right = el('div');
  const shot = el('div', 'shot'), img = new Image(), sq = el('div', 'sq');
  img.alt = 'Full picture from camera ' + c.dev; img.src = c.file;
  shot.appendChild(img); shot.appendChild(sq); left.appendChild(shot);
  const nums = el('div', 'nums'), inp = {};
  [['x', 'from the left'], ['y', 'from the top'], ['s', 'size of the square']].forEach(([k, name], i) => {
    const lab = el('label', '', name), box = document.createElement('input');
    box.type = 'number'; box.min = 0; box.value = [c.roi[0], c.roi[1], c.roi[2]][i]; inp[k] = box;
    const wrap = el('div'); wrap.appendChild(lab); wrap.appendChild(box); nums.appendChild(wrap);
  });
  left.appendChild(nums);
  const line = el('p'); left.appendChild(line);
  right.appendChild(el('p', '', 'What the AI gets'));
  const cv = document.createElement('canvas'); cv.width = 256; cv.height = 256; right.appendChild(cv);
  const t = el('table');
  [['Picture', c.width + ' x ' + c.height + (c.fourcc ? ', ' + c.fourcc : '')],
   ['Pictures per second', c.fps + (c.fps < 20 ? ' (slow: needs more light)' : '')],
   ['Brightness in the square', c.brightness + ' of 255 (good: 90 to 180)'],
   ['Sharpness in the square', c.sharpness + ' (higher is sharper, under 60 is blurred)']].forEach(r => {
    const tr = el('tr'); tr.appendChild(el('td', '', r[0])); tr.appendChild(el('td', '', r[1])); t.appendChild(tr); });
  right.appendChild(t);
  c.advice.forEach(a => right.appendChild(el('p', a.bad ? 'alert' : 'ok', a.text)));
  grid.appendChild(left); grid.appendChild(right); sec.appendChild(grid); host.appendChild(sec);
  function draw(){
    let s = Math.max(16, Math.min(parseInt(inp.s.value) || 16, c.width, c.height));
    let x = Math.max(0, Math.min(parseInt(inp.x.value) || 0, c.width - s));
    let y = Math.max(0, Math.min(parseInt(inp.y.value) || 0, c.height - s));
    sq.style.left = (100 * x / c.width) + '%'; sq.style.top = (100 * y / c.height) + '%';
    sq.style.width = (100 * s / c.width) + '%'; sq.style.height = (100 * s / c.height) + '%';
    line.innerHTML = '';
    line.appendChild(document.createTextNode('Type this in the black window: '));
    line.appendChild(el('code', '', x + ' ' + y + ' ' + s));
    if (img.complete && img.naturalWidth) cv.getContext('2d').drawImage(img, x, y, s, s, 0, 0, 256, 256);
  }
  img.onload = draw; Object.values(inp).forEach(b => b.oninput = draw); draw();
});
</script></body></html>
"""


def write_page(out_dir: str, data: dict) -> str:
    page = PAGE.replace("__WHEN__", time.strftime("%d %b %Y, %H:%M")).replace(
        "__DATA__", json.dumps(data).replace("</", "<\\/"))
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


# ----------------------------------------------------------------------------- the questions
def ask(prompt: str) -> str:
    say(prompt)
    try:
        return input().strip()
    except EOFError:
        return ""


def parse_square(text: str, width: int, height: int):
    """'80 0 480' or '80, 0, 480, 480' -> [x, y, s, s] inside the picture, or None if it is not 3 or 4 numbers."""
    nums = re.findall(r"-?\d+", text)
    if len(nums) not in (3, 4):
        return None
    x, y, s = int(nums[0]), int(nums[1]), int(nums[2])
    if s < 16:
        return None
    return fit_roi([x, y, s, s], width, height)


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One picture from every camera, and which one the real line uses")
    ap.add_argument("--no-ask", action="store_true", help="only make the page; ask nothing, change nothing")
    ap.add_argument("--no-open", action="store_true", help="do not open the page in the browser")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg_path = a.config or DEFAULT_PATH
    cfg = load_config(cfg_path)
    c = cfg["camera"]
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")    # OpenCV's own warnings for camera numbers that do not exist
    try:
        import cv2
    except ImportError:
        say("OpenCV is missing. Double-click CHECK_CAMERA.bat (it installs it), or: pip install opencv-python-headless")
        return 1
    out_dir = project_path(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):                          # pictures of the last check are not kept
        if f.startswith("camera_") and f.endswith(".jpg"):
            try:
                os.remove(os.path.join(out_dir, f))
            except OSError:
                pass

    say("Looking for cameras on this laptop (each one is switched on for a moment) ...")
    devs = list_cameras(cfg, cv2)
    spec = str(c.get("usb_device", "auto")).strip()
    auto = spec.lower() in ("", "auto")
    if auto:
        used, why = (devs[-1] if devs else None), "the setting is 'auto': the last camera in the list"
    else:
        try:
            used = webcam.find_device(spec, int(c.get("usb_index", 0)))
        except Exception:
            used = None
        why = f"camera.usb_device is set to {spec}"
        if used is not None and used not in devs:
            devs = devs + [used]                           # a camera number the search did not reach: try it too
    if not devs:
        say("")
        say("NO CAMERA FOUND.")
        say("  - Plug the USB webcam straight into the laptop (not through a hub) and run this again.")
        say("  - Close programs that use a camera (Teams, Zoom, the Windows Camera app, a browser tab).")
        say("  - Windows: Settings > Privacy and security > Camera > 'Let desktop apps access your camera' = On.")
        write_page(out_dir, {"cams": [], "used": None, "why": why})
        return 1

    stamp = int(time.time())
    cams, shots = [], {}
    for dev in devs:
        say(f"  camera {dev}: taking one picture ...")
        frame, info, fps, err = shoot(cfg, dev)
        entry = {"dev": dev, "used": dev == used, "ok": frame is not None, "error": err}
        if frame is not None:
            h, w = frame.shape[:2]
            roi = fit_roi(c["roi"], w, h)
            m = measure(frame, roi)
            name = f"camera_{re.sub(r'[^0-9A-Za-z]+', '_', str(dev)).strip('_')}"
            Image.fromarray(frame).save(os.path.join(out_dir, name + ".jpg"), quality=90)
            crop_roi(frame, roi, int(c["upload_size"])).save(os.path.join(out_dir, name + "_crop.jpg"), quality=90)
            entry.update(file=f"{name}.jpg?t={stamp}", width=w, height=h, fourcc=info.get("fourcc", ""),
                         fps=round(fps, 1), roi=roi, brightness=round(m["brightness"]),
                         sharpness=round(m["sharpness"]), advice=advice(m))
            shots[str(dev)] = (w, h)
            say(f"     {w}x{h}, {fps:.0f} pictures per second, brightness {m['brightness']:.0f} of 255, "
                f"sharpness {m['sharpness']:.0f}")
            for adv in entry["advice"]:
                if adv["bad"]:
                    say(f"     NOTE: {adv['text']}")
        else:
            say(f"     no picture: {err}")
        cams.append(entry)
    page = write_page(out_dir, {"cams": cams, "used": used, "why": why})
    say("")
    say(f"The real line uses camera {used}  ({why}).")
    try:
        shown = os.path.relpath(page, project_path("."))
    except ValueError:                                     # Windows: the page is on another drive
        shown = page
    say(f"Page with the pictures: {shown}")
    if not a.no_open:
        try:
            webbrowser.open("file:///" + os.path.abspath(page).replace(os.sep, "/").lstrip("/"))
        except Exception:
            pass
    good = [e for e in cams if e["ok"]]
    if a.no_ask or not sys.stdin or not sys.stdin.isatty() or not good:
        return 0 if any(e["ok"] and e["used"] for e in cams) else 1

    # ---- the two questions
    say("")
    say("=" * 72)
    names = ", ".join(str(e["dev"]) for e in good)
    ans = ask(f"1. Which camera shows the BELT?  Type its number ({names}) and press Enter.\n"
              f"   Just Enter = keep camera {used}.")
    pick, new_dev = used, None
    if ans:
        match = [e["dev"] for e in good if str(e["dev"]) == ans]
        if match:
            pick = match[0]
            if auto or str(pick) != str(used):             # 'auto' is replaced by the number that was typed
                new_dev = pick
        else:
            say(f"   '{ans}' is not one of {names} - the camera stays {used}.")
    roi_now = c["roi"]
    w, h = shots.get(str(pick), (int(c["width"]), int(c["height"])))
    ans = ask(f"2. Crop square: type the three numbers from the page (from the left, from the top, size)\n"
              f"   and press Enter.  Just Enter = keep {roi_now[0]} {roi_now[1]} {roi_now[2]}.")
    new_roi = None
    if ans:
        sq = parse_square(ans, w, h)
        if sq is None:
            say("   That is not three numbers - the crop square stays as it is.")
        elif [int(v) for v in roi_now] != sq:
            new_roi = sq
    try:
        done = save_settings(cfg_path, new_dev, new_roi)
    except (OSError, ValueError) as e:
        say(f"   COULD NOT SAVE: {e}")
        return 1
    say("")
    if done:
        say("Saved in config/system.yaml:")
        for d in done:
            say("   " + d)
        say("Double-click CHECK_CAMERA.bat again to see the result.")
    else:
        say("Nothing was changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
