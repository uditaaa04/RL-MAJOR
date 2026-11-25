"""
One-click launcher - nothing to type. Used by the *.bat files in the project folder.

    START_DEMO.bat                  -> python tools/go.py demo      (no hardware: pretend belt, this PC is also the cloud)
    START_REAL_LINE.bat             -> python tools/go.py line      (REAL belt: ESP32 on USB + webcam; cloud = AWS if set up, else this PC)
    START_DEMO_ON_AWS.bat           -> python tools/go.py aws       (pretend belt against the real AWS server)
    CHECK_ESP32.bat                 -> python tools/go.py esp32     (finds the board, tests arm, sensors, tiny model)
    CHECK_CAMERA.bat                -> python tools/go.py camera    (one picture from every camera: which one the line uses, crop square)
    MEASURE_SPEED_VS_ACCURACY.bat   -> python tools/go.py tradeoff  (same photos at ESP32 / gateway / cloud: time and accuracy)
    TRAIN_ON_DOWNLOADED_PHOTOS.bat  -> python tools/go.py public    (public photo set -> all three models, agent, results)
    RETRAIN_ON_REAL_PHOTOS.bat      -> python tools/go.py retrain   (YOUR photos -> the same)
    RUN_PUBLIC_BENCHMARK.bat        -> python tools/go.py benchmark (public task-graph dataset from Zenodo)
    REMOTE_CONTROL.bat              -> python tools/go.py remote    (stays open; Claude starts/stops the steps)
    CHECK_EVERYTHING.bat            -> python tools/go.py test      (the automatic tests)

The three nodes:  ESP32 (edge, on the belt)  <-USB->  this laptop (gateway)  <-Internet->  AWS (cloud)

What it does by itself:
  * builds a private Python environment for this PC the first time (outside OneDrive, so nothing syncs),
  * moves files of the older 2-node (Raspberry Pi) version out of the way and re-trains what the 3-node
    version needs (tiny ESP32 model, 3-way agent) if it is missing,
  * starts the cloud server and the gateway program, waits until they answer, opens the dashboard,
  * writes everything to logs/go_<mode>.log (send that file if something fails),
  * stops everything when you press Enter.

Only the Python standard library is used here, so it runs before anything is installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = os.name == "nt"
ENV_HOME = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "SmartConveyorQC")
VENV = os.path.join(ENV_HOME, "venv")
PY = os.path.join(VENV, "Scripts", "python.exe") if WIN else os.path.join(VENV, "bin", "python")
LOGS = os.path.join(ROOT, "logs")
SECRETS = os.path.join(ROOT, "cloud", ".secrets")
PORT = 8000
NEEDED = "numpy, onnxruntime, flask, waitress, PIL, yaml, requests, psutil, serial"

_log = None


def say(msg=""):
    print(msg, flush=True)
    if _log:
        _log.write(msg + "\n")
        _log.flush()


def stop(msg):
    say("")
    say("STOPPED: " + msg)
    say(f"(details are in {os.path.relpath(_log.name, ROOT) if _log else 'the window above'})")
    sys.exit(1)


def run(cmd, **kw):
    """Run a command, copy its output to the screen and the log, return the exit code."""
    say("> " + " ".join(os.path.basename(c) if i == 0 else c for i, c in enumerate(cmd)))
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         errors="replace", **kw)
    for line in p.stdout:
        say("  " + line.rstrip())
    return p.wait()


# ----------------------------------------------------------------------------- python environment
def ensure_env(extra_req=None):
    if sys.version_info < (3, 10):
        stop(f"this is Python {sys.version.split()[0]}; version 3.10 or newer is needed. "
             "Install Python 3.12 from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'.")
    req = os.path.join(ROOT, extra_req or "requirements-laptop.txt")
    with open(req, "rb") as f:
        want = hashlib.sha256(f.read()).hexdigest() + " " + sys.version.split()[0]
    mark = os.path.join(ENV_HOME, "installed_" + os.path.basename(req) + ".txt")
    if not os.path.exists(PY):
        say(f"First run on this PC: building the Python environment in {ENV_HOME}")
        say("(needs the Internet once, about 2-5 minutes)")
        os.makedirs(ENV_HOME, exist_ok=True)
        if run([sys.executable, "-m", "venv", VENV]) != 0:
            stop("could not create the Python environment")
    have = open(mark).read().strip() if os.path.exists(mark) else ""
    if have != want:
        say("Installing the packages this project needs ...")
        code = run([PY, "-m", "pip", "install", "--disable-pip-version-check", "-r", req])
        if code != 0:
            stop("a package could not be installed. Check the Internet connection and run this again. "
                 f"If it fails on 'onnxruntime', this Python ({sys.version.split()[0]}) is too new for it: "
                 "install Python 3.12 from python.org and run this again.")
        with open(mark, "w") as f:
            f.write(want)
    code = subprocess.call([PY, "-c", f"import {NEEDED}"], cwd=ROOT)
    if code != 0:
        try:
            os.remove(mark)
        except OSError:
            pass
        stop("the Python environment is incomplete - run this again to repair it")
    say("Python environment: OK")


# ----------------------------------------------------------------------------- small helpers
def get_json(url, key="", timeout=3.0):
    r = urllib.request.Request(url, headers={"X-API-Key": key} if key else {})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read().decode())


def server_up(base):
    try:
        return bool(get_json(base + "/health").get("ok"))
    except Exception:
        return False


def my_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))            # no packet is sent; just picks the outgoing network card
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(a[4][0])
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith(("127.", "169.254.")))


def spawn(name, args, env=None):
    """Start a long-running program in the background; its output goes to logs/go_<name>.log."""
    path = os.path.join(LOGS, f"go_{name}.log")
    out = open(path, "w", encoding="utf-8", errors="replace")
    e = dict(os.environ)
    e["PYTHONUNBUFFERED"] = "1"
    e.update(env or {})
    stopfile = None
    if name == "gateway":
        # Windows cannot ask a program to stop politely, it can only kill it. The gateway looks for this file
        # twice a second and then stops by itself: belt off at once, what the agent learned is saved.
        stopfile = os.path.join(LOGS, "stop_gateway.flag")
        try:
            os.remove(stopfile)
        except OSError:
            pass
        e["CONVEYOR_STOP_FILE"] = stopfile
    p = subprocess.Popen([PY] + args, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, env=e)
    p._logpath = path
    p._stopfile = stopfile
    return p


def ask_to_stop(procs, seconds=8.0):
    """Let the gateway stop by itself (belt off, agent saved) before anything is killed."""
    asked = []
    for p in procs:
        f = getattr(p, "_stopfile", None)
        if f and p.poll() is None:
            try:
                with open(f, "w") as fh:
                    fh.write("stop\n")
                asked.append(p)
            except OSError:
                pass
    t_end = time.time() + seconds
    while asked and time.time() < t_end and any(p.poll() is None for p in asked):
        time.sleep(0.2)
    for p in asked:
        try:
            os.remove(p._stopfile)
        except OSError:
            pass


def tail(path, n=25):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def wait_server(base, proc, seconds=90):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if server_up(base):
            return
        if proc is not None and proc.poll() is not None:
            say(tail(proc._logpath))
            stop("the cloud server program ended by itself (its last lines are above)")
        time.sleep(0.7)
    stop("the cloud server did not answer in time")


def hold(base, procs, key=""):
    """Keep running, print a status line now and then, stop everything on Enter."""
    say("")
    say("=" * 74)
    say("  RUNNING.  Dashboard:  " + base)
    say("  Press ENTER in this window to stop everything.")
    say("=" * 74)
    import threading
    done = threading.Event()

    def waiter():
        try:
            input()
        except EOFError:
            time.sleep(10 ** 8)
        done.set()

    threading.Thread(target=waiter, daemon=True).start()
    last = 0.0
    try:
        while not done.is_set():
            for p in procs:
                if p.poll() is not None:
                    say(tail(p._logpath))
                    say(f"A program ended by itself (log: {os.path.relpath(p._logpath, ROOT)}).")
                    done.set()
            if time.time() - last > 15:
                last = time.time()
                try:
                    a = get_json(base + "/api/state", key).get("all", {})
                    n = a.get("n", 0)
                    if n:
                        say(f"  {time.strftime('%H:%M:%S')}  boxes {n}   ESP32 {100*a.get('edge',0)/n:.0f}%   "
                            f"gateway {100*a.get('gateway',0)/n:.0f}%   cloud {100*a.get('cloud',0)/n:.0f}%   "
                            f"missed {100*a.get('missed',0)/n:.1f}%")
                except Exception:
                    pass
            done.wait(1.0)
    except KeyboardInterrupt:
        pass
    say("Stopping ...")
    ask_to_stop(procs)
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=8)
        except subprocess.TimeoutExpired:
            p.kill()
    say("Stopped.")


def mock_feed_ready(d=None):
    d = d or os.path.join(ROOT, "data", "synth", "test")
    for sub in ("good", "damaged"):
        p = os.path.join(d, sub)
        if not os.path.isdir(p) or not any(f.lower().endswith((".jpg", ".png")) for f in os.listdir(p)):
            return False
    return True


mock_feed_ready_at = mock_feed_ready


# ----------------------------------------------------------------------------- from the 2-node version to 3 nodes
OLD_2NODE = ["pi_node", "deploy/pi", "START_CLOUD_ON_THIS_PC.bat", "MAKE_PI_PACKAGE.bat", "PI_CONNECTION.txt",
             "requirements-pi.txt", "tools/hw_test.py", "tools/retarget_pi5_aws.py", "tools/package.py",
             "docs/HARDWARE_STEP_BY_STEP.md", "docs/WIRING.md", "docs/BUILD_GUIDE.md", "docs/CODEX_PROMPTS.md",
             "dist/SmartConveyor_QL_for_Pi.zip", "server/dashboard/index_v1.html"]


def read_card():
    """models/model_card.json; a card from the 2-node version calls the small model "edge" - it is "gateway" now."""
    try:
        with open(os.path.join(ROOT, "models", "model_card.json"), encoding="utf-8") as f:
            card = json.load(f)
    except (OSError, ValueError):
        return {}
    if "gateway" not in card and "edge" in card and "mobilenet" in str(card["edge"].get("arch", "")):
        card["gateway"] = card.pop("edge")
    return card


def migrate():
    """Runs before every step. Safe to repeat. Nothing is deleted: old files go to _old_2node_version/."""
    import shutil
    m = os.path.join(ROOT, "models")
    changed = []
    for old, new in (("edge_model.onnx", "gateway_model.onnx"), ("edge_model.pt", "gateway_model.pt")):
        if os.path.exists(os.path.join(m, old)) and not os.path.exists(os.path.join(m, new)):
            shutil.copy2(os.path.join(m, old), os.path.join(m, new))
            changed.append(f"models/{old} -> models/{new}")
    cp = os.path.join(m, "model_card.json")
    try:
        with open(cp, encoding="utf-8") as f:
            raw = json.load(f)
        if "gateway" not in raw and "edge" in raw and "mobilenet" in str(raw["edge"].get("arch", "")):
            raw["gateway"] = raw.pop("edge")
            with open(cp, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2)
            changed.append("model card: the small model is now listed as the gateway's")
    except (OSError, ValueError):
        pass
    attic = os.path.join(ROOT, "_old_2node_version")
    for rel in OLD_2NODE + ["models/edge_model.onnx", "models/edge_model.pt"]:
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src):
            continue
        if rel.startswith("models/edge_model") and not os.path.exists(os.path.join(m, rel.split("/")[1].replace("edge_", "gateway_"))):
            continue
        dst = os.path.join(attic, rel)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
            shutil.move(src, dst)
            changed.append(f"{rel} -> _old_2node_version/")
        except OSError:
            pass                                  # in use / locked: try again next time
    if changed:
        say("Tidied up after the switch from 2 nodes (Raspberry Pi) to 3 nodes (ESP32 + gateway + cloud):")
        for c in changed:
            say("  " + c)


def photos_dir():
    """Folder the gateway and cloud models were trained on (has train/ val/ test/), or None."""
    d = str((read_card().get("gateway") or {}).get("data") or "")
    full = os.path.join(ROOT, d) if d else ""
    if not d or not os.path.isdir(full):
        return None, None
    n = count_images(os.path.join(full, "train"))
    return (d, full) if min(n.values()) >= 40 else (None, None)      # too few photos to train anything on


def ensure_tiny(must=False):
    """The ESP32's tiny model. Trains it (1-2 minutes) when it is missing or was trained on other photos."""
    rel, full = photos_dir()
    card = read_card()
    have = os.path.exists(os.path.join(ROOT, "models", "edge_tiny.bin")) and "edge" in card
    if have and (rel is None or str(card["edge"].get("data", "")).replace("\\", "/") == rel.replace("\\", "/")):
        return "ready"
    if rel is None:
        if not have:
            say("NOTE: no tiny model for the ESP32 and no training photos on this PC - the ESP32 will only run the belt.")
        return "ready" if have else "missing"
    if subprocess.call([PY, "-c", "import torch"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        if not must:
            say("NOTE: the ESP32's tiny model is not trained yet (double-click TRAIN_ON_DOWNLOADED_PHOTOS.bat).")
            return "ready" if have else "missing"
        ensure_torch()
    say("")
    say(f"Training the tiny model for the ESP32 on {rel} (1-2 minutes) ...")
    if run([PY, "-m", "vision.train_tiny", "--data", rel]) != 0:
        say("NOTE: the tiny model could not be trained (see above). The ESP32 will only run the belt.")
        return "ready" if have else "missing"
    t = (read_card().get("edge") or {}).get("test") or {}
    if t:
        say(f"  ESP32 model: correct {100*t['accuracy']:.1f}%   damaged caught {100*t['recall_damaged']:.1f}%   "
            f"good wrongly rejected {100*t['false_reject_rate']:.1f}%")
    return "trained"


def agent_fits():
    code = ("import numpy as np, json, sys; sys.path.insert(0, '.'); "
            "from common.config import load_config; from common.rl import StateSpace; "
            "n = StateSpace(load_config(None)['state_bins']).n_states; z = np.load('models/q_table.npz'); "
            "r = json.loads(str(z['meta'])).get('reward', {}); "
            "sys.exit(0 if z['q'].shape == (n, 3) and 'edge_on_time' in r else 1)")
    return subprocess.call([PY, "-c", code], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def ensure_agent(force=False):
    """The Q-table must have three choices per situation and match the models in use."""
    if not force and agent_fits():
        return
    say("")
    say("Pre-training the agent: second opinion with a cloud-call limit (about 2 minutes) ...")
    if run([PY, "-m", "sim.train_q"]) != 0:
        stop("the agent could not be pre-trained (see above)")
    for f in ("q_table_online.npz",):
        try:
            os.remove(os.path.join(ROOT, "models", f))
        except OSError:
            pass


def ensure_ready(must_tiny=False):
    """Everything the 3-node gateway needs beyond the two big models: tiny ESP32 model, 3-way agent."""
    trained_now = ensure_tiny(must_tiny) == "trained"
    scored_now = ensure_scores(force=trained_now)
    ensure_agent(force=trained_now or scored_now)   # the agent's reward depends on the models' scores


def ensure_scores(force=False):
    """Every photo's score at each place (needed for the confidence levels). True = made just now."""
    have = os.path.exists(os.path.join(ROOT, "models", "photo_scores.npz")) and "cascade" in read_card()
    if have and not force:
        return False
    rel, _ = photos_dir()
    if rel is None:
        if not have:
            say("NOTE: no training photos on this PC, so the confidence levels cannot be measured. "
                "The agent will treat every box as 'sure' and will not ask the cloud for second opinions.")
        return False
    say("")
    say("Scoring every validation and test photo with the three models (about 20 seconds) ...")
    if run([PY, "-m", "tools.photo_scores", "--data", rel]) != 0:
        say("NOTE: the photos could not be scored; the confidence levels stay as they were.")
        return False
    return True


def pretend_feed(env):
    rel, full = photos_dir()
    if rel and not rel.replace("\\", "/").endswith("data/synth") and mock_feed_ready_at(os.path.join(full, "test")):
        env["CONVEYOR_MOCK_FEED"] = os.path.join(full, "test")
        say(f"  boxes on the pretend belt: held-out test photos from {rel}")


def aws_config():
    cf = os.path.join(SECRETS, "aws_server.json")
    try:
        with open(cf) as f:
            c = json.load(f)
        return c if c.get("url") else None
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------- modes
def start_local_server(procs, env, reset=True):
    base = f"http://127.0.0.1:{PORT}"
    if server_up(base):
        say("A cloud server is already running on this PC - using it.")
        return base
    say("Starting the cloud server on this PC ...")
    # listen on this PC only (no API key is set, so do not open it to the network)
    procs.append(spawn("server", ["-m", "server.app", "--cpu-only", "--host", "127.0.0.1"], env))
    wait_server(base, procs[-1])
    if reset:
        try:                                         # a run starts from zero, not from yesterday's counters
            urllib.request.urlopen(urllib.request.Request(base + "/api/reset", data=b"{}", method="POST",
                                   headers={"Content-Type": "application/json"}), timeout=5).read()
        except Exception:
            pass
    return base


def mode_demo():
    """No hardware needed: pretend belt on this PC, this PC is also the cloud server. If an ESP32 is plugged
    in, its tiny model is really used; otherwise a pretend ESP32 takes its place."""
    ensure_env()
    if not mock_feed_ready():
        say("Making the practice box photos (once, about a minute) ...")
        if run([PY, "-m", "vision.synth_boxes", "--out", "data/synth", "--train", "2", "--val", "2", "--test", "300"]) != 0:
            stop("could not make the practice photos")
    ensure_ready()
    procs = []
    clean = {"CONVEYOR_API_KEY": "", "CONVEYOR_SERVER": "", "CONVEYOR_WHERE": "this PC"}
    base = start_local_server(procs, clean)
    say("Starting the gateway with the pretend belt ...")
    pretend_feed(clean)
    procs.append(spawn("gateway", ["-m", "gateway_node.main", "--mock", "--esp32", "auto", "--server", base], clean))
    time.sleep(4)
    webbrowser.open(f"http://localhost:{PORT}")
    say("The dashboard opened in your browser. Press 'Run the guided demo' there.")
    say("(ESP32 plugged in by USB? Then its tiny model is really used. If not, a pretend ESP32 is.)")
    hold(base, procs)


def have_cv2():
    """True when OpenCV works in the project's Python AND the packages it shares with the models still do."""
    return subprocess.call([PY, "-c", "import numpy, onnxruntime, cv2; cv2.VideoCapture"], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def ensure_cv2():
    """OpenCV reads the webcam. It is installed the first time a step needs the camera."""
    if have_cv2():
        return
    say("Installing OpenCV for the webcam (once, needs the Internet) ...")
    # ready-made packages only: nothing is ever compiled on this PC
    pip = [PY, "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:"]
    if run(pip + ["opencv-python-headless"]) == 0 and have_cv2():
        return
    # On a very new Python, OpenCV's package can ask for an older numpy than exists for that Python. numpy is
    # already installed here and OpenCV works with it, so install OpenCV alone and leave the rest untouched.
    say("Trying again without touching the other packages ...")
    if run(pip + ["--no-deps", "opencv-python-headless"]) == 0 and have_cv2():
        return
    stop("OpenCV could not be installed. Check the Internet connection and run this again. If it still fails, "
         f"this Python ({sys.version.split()[0]}) may be too new for OpenCV: send the log file named below.")


def mode_camera():
    """One picture from every camera on this laptop: shows which one the real line uses and its crop square,
    then asks for the right camera number and square and saves them. Run it by hand only (it uses the cameras)."""
    ensure_env()
    ensure_cv2()
    if server_up(f"http://127.0.0.1:{PORT}"):
        say("NOTE: a demo or the real line is still running on this PC. If the real line is running it holds the "
            "camera, and the check will get no picture from it. Stop it first (press Enter in its window).")
    say("Taking one picture from every camera on this laptop. The pictures stay on this PC (logs/camera_check).")
    code = subprocess.call([PY, "-u", "-m", "tools.camera_check"], cwd=ROOT)    # not through run(): it asks questions
    say(f"camera check ended with code {code}")
    if code != 0:
        stop("the camera check did not get a picture from the camera the real line uses (see above)")


def mode_line():
    """THE REAL LINE: ESP32 on USB runs the belt, this laptop is the gateway (webcam + agent), the cloud is the
    AWS server if it was set up (START_DEMO_ON_AWS once), otherwise a server on this PC."""
    ensure_env()
    ensure_cv2()
    ensure_ready()
    if server_up(f"http://127.0.0.1:{PORT}"):
        # a demo left open would keep sending its pretend boxes into the same dashboard, mixed with the real ones
        stop("another run is still open on this PC (the demo, or the real line in another window). "
             "Go to that window and press Enter to stop it, or close it. Then start this again.")
    procs = []
    c = aws_config()
    if c:
        env = {"CONVEYOR_API_KEY": c.get("key", ""), "CONVEYOR_SERVER": c["url"]}
        base, key = c["url"], c.get("key", "")
        say(f"Cloud node: AWS server {base}")
        if not server_up(base):
            stop(f"the AWS server {base} does not answer. Is the EC2 instance running? "
                 f"(To use this PC as the cloud instead, delete cloud/.secrets/aws_server.json.)")
    else:
        env = {"CONVEYOR_API_KEY": "", "CONVEYOR_SERVER": "", "CONVEYOR_WHERE": "this PC"}
        say("Cloud node: no AWS server is set up yet, so the big model runs on THIS PC for now.")
        base, key = start_local_server(procs, env), ""
    say("Starting the gateway: ESP32 on USB + webcam ...")
    gw = spawn("gateway", ["-m", "gateway_node.main", "--server", base], env)
    procs.append(gw)
    time.sleep(8)
    if gw.poll() is not None:
        say(tail(gw._logpath, 12))
        for p in procs:
            if p.poll() is None:
                p.terminate()
        stop("the gateway could not start (its last lines are above). Most common: the ESP32 is not plugged in, "
             "the Arduino Serial Monitor is still open, or the webcam is used by another program. "
             "Double-click CHECK_ESP32.bat to test the board alone, CHECK_CAMERA.bat to test the camera alone.")
    webbrowser.open(f"{base}/?key={key}" if key else f"http://localhost:{PORT}")
    say("The belt is running. Put boxes on it.")
    hold(base, procs, key)


def mode_esp32():
    """Finds the ESP32, loads the tiny model, times it, moves the arm, shows the two sensors."""
    ensure_env()
    ensure_tiny()
    if run([PY, "-m", "tools.esp32_check"]) != 0:
        stop("the ESP32 check did not pass (see above)")


def mode_tradeoff():
    """The same photos at all three places: accuracy and time. Uses the real ESP32 and the AWS server when
    they are there, and says so in the result when they are not."""
    ensure_env()
    ensure_ready(must_tiny=True)
    procs = []
    c = aws_config()
    if c and server_up(c["url"]):
        env = dict(os.environ, CONVEYOR_API_KEY=c.get("key", ""), CONVEYOR_SERVER=c["url"])
        server = c["url"]
        say(f"Cloud node for this measurement: AWS server {server}")
    else:
        env = dict(os.environ, CONVEYOR_API_KEY="", CONVEYOR_SERVER="")
        server = start_local_server(procs, {"CONVEYOR_API_KEY": "", "CONVEYOR_SERVER": "", "CONVEYOR_WHERE": "this PC"}, reset=False)
        say("Cloud node for this measurement: a server on THIS PC (no AWS server is set up yet).")
    busy = False
    try:
        st = get_json(server + "/api/state", (c or {}).get("key", "") if c else "")
        busy = st.get("status_age_s") is not None and st["status_age_s"] < 5
    except Exception:
        pass
    args = [PY, "-m", "tools.tradeoff", "--server", server]
    if busy:
        say("A gateway is running right now and holds the ESP32's USB port, so the ESP32 is emulated in this run.")
        say("(For the real ESP32 timing: stop the demo first, then run this again.)")
        args += ["--esp32", "mock"]
    code = run(args, env=env)
    for p in procs:
        p.terminate()
    if code != 0:
        stop("the measurement did not finish (see above)")
    subprocess.call([PY, "-m", "tools.make_summary"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    say("Done: results/TRADEOFF.md and results/figures/tradeoff.png. It is also the first table on the dashboard's Results page.")


def local_key(new=False):
    os.makedirs(SECRETS, exist_ok=True)
    kf = os.path.join(SECRETS, "api_key_local.txt")
    if new or not os.path.exists(kf):
        with open(kf, "w") as f:
            f.write(secrets.token_hex(16))
    return open(kf).read().strip(), kf


def mode_aws():
    """Pretend belt on this PC against the REAL cloud server on AWS (real Internet path). The address and key
    typed here once are also used by START_REAL_LINE and MEASURE_SPEED_VS_ACCURACY from then on."""
    ensure_env()
    ensure_ready()
    os.makedirs(SECRETS, exist_ok=True)
    cf = os.path.join(SECRETS, "aws_server.json")
    if os.path.exists(cf):
        c = json.load(open(cf))
    else:
        say("One-time set-up. Paste the two values printed at the end of the AWS deployment.")
        url = input("  Cloud server address (http://<ip>:8000): ").strip().rstrip("/")
        key = input("  API key: ").strip()
        if not url.startswith("http"):
            stop("the address must start with http://")
        c = {"url": url, "key": key}
        with open(cf, "w") as f:
            json.dump(c, f)
    env = {"CONVEYOR_API_KEY": c["key"], "CONVEYOR_SERVER": c["url"]}
    say(f"Checking {c['url']} ...")
    e = dict(os.environ)
    e.update(env)
    if run([PY, "-m", "tools.check_server", "--server", c["url"]], env=e) != 0:
        stop("the AWS server did not pass the check (is the EC2 instance running? is the key right? "
             f"delete {os.path.relpath(cf, ROOT)} to enter the values again)")
    say("Starting the gateway with the pretend belt against AWS ...")
    pretend_feed(env)
    pi = spawn("gateway", ["-m", "gateway_node.main", "--mock", "--esp32", "auto", "--server", c["url"]], env)
    time.sleep(3)
    webbrowser.open(f"{c['url']}/?key={c['key']}")
    hold(c["url"], [pi], c["key"])


def mode_test():
    ensure_env()
    subprocess.call([PY, "-m", "pip", "install", "--disable-pip-version-check", "-q", "pytest"], cwd=ROOT)
    code = run([PY, "-m", "pytest", "-q", "tests"])
    say("ALL TESTS PASSED" if code == 0 else "SOME TESTS FAILED (see above)")


def count_images(folder):
    n = {}
    for lab in ("good", "damaged"):
        d = os.path.join(folder, lab)
        n[lab] = len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))]) if os.path.isdir(d) else 0
    return n


def ensure_torch():
    if subprocess.call([PY, "-c", "import torch, torchvision, onnx"], cwd=ROOT) != 0:
        say("Installing PyTorch for training (once, a few hundred MB) ...")
        if run([PY, "-m", "pip", "install", "--disable-pip-version-check", "torch", "torchvision", "onnx"]) != 0:
            stop("PyTorch could not be installed (Internet? very new Python?)")


def train_candidate(data_rel, gw_epochs=12, cloud_epochs=8):
    """Train the gateway and cloud models into models_candidate/ (the models in use are not touched)."""
    ensure_torch()
    cand = os.path.join(ROOT, "models_candidate")
    import shutil
    if os.path.isdir(cand):
        shutil.rmtree(cand)
    init = os.environ.get("CONVEYOR_TRAIN_INIT", "imagenet")
    scale = float(os.environ.get("CONVEYOR_TRAIN_EPOCH_SCALE", "1"))
    for role, ep, extra in (("gateway", gw_epochs, []), ("cloud", cloud_epochs, ["--img", "160"])):
        ep = max(1, int(round(ep * scale)))
        say("")
        say(f"Training the {role} model on {data_rel} ({ep} rounds, start: {init}) ...")
        cmd = [PY, "-m", "vision.train", "--role", role, "--data", data_rel, "--epochs", str(ep),
               "--out", "models_candidate"] + extra
        if run(cmd + ["--init", init]) != 0:
            if init != "imagenet":
                stop(f"training the {role} model failed")
            say("Pretrained weights could not be used (download blocked?). Training from zero instead - weaker.")
            if run(cmd + ["--init", "scratch"]) != 0:
                stop(f"training the {role} model failed")
    card = json.load(open(os.path.join(cand, "model_card.json")))
    out = {}
    for role in ("gateway", "cloud"):
        t = card[role].get("test") or card[role].get("val")
        out[role] = t
        say(f"  {role:7s}: correct {100*t['accuracy']:.1f}%   damaged caught {100*t['recall_damaged']:.1f}%   "
            f"good wrongly rejected {100*t['false_reject_rate']:.1f}%   ({t.get('n', '?')} test photos)")
    return out


def already_trained_on(data_rel):
    """True when the gateway and cloud models in use were trained on exactly this photo folder."""
    card = read_card()
    same = lambda r: str((card.get(r) or {}).get("data", "")).replace("\\", "/").rstrip("/") == data_rel
    return (same("gateway") and same("cloud") and os.path.exists(os.path.join(ROOT, "models", "gateway_model.onnx"))
            and os.path.exists(os.path.join(ROOT, "models", "cloud_model.onnx")))


def rebuild_everything(data_rel, keep=None):
    """After any model changed: tiny model, agent, experiments, trade-off table, summaries, cloud bundle."""
    safe = f" Your old models are safe in {os.path.basename(keep)}." if keep else ""
    ensure_torch()
    say("")
    say(f"[1/9] tiny model for the ESP32 on {data_rel}")
    if run([PY, "-m", "vision.train_tiny", "--data", data_rel]) != 0:
        stop("step failed: tiny model for the ESP32." + safe)
    say("")
    say("[2/9] every photo's score at each place (for the second-opinion decision)")
    if run([PY, "-m", "tools.photo_scores", "--data", data_rel]) != 0:
        stop("step failed: photo scores." + safe)
    say("  ... and the same photos made darker / blurred (for the 'photos change' experiment)")
    if run([PY, "-m", "tools.shift_scores", "--data", data_rel]) != 0:
        say("NOTE: the changed photos could not be scored; experiment E8 will be skipped.")
    online = os.path.join(ROOT, "models", "q_table_online.npz")     # learned with the old models
    if os.path.exists(online):
        os.remove(online)
    steps = [
        ("agent pre-training + rule tuning (second opinion, cloud-call limit)", ["-m", "sim.train_q"]),
        ("simulation experiments E1-E5 and E7-E9", ["-m", "sim.experiments"]),
        ("other damage rates and costs (E6)", ["-m", "sim.sensitivity", "--data", data_rel]),
    ]
    for i, (what, args) in enumerate(steps, 3):
        say("")
        say(f"[{i}/9] {what}")
        if run([PY] + args) != 0:
            stop(f"step failed: {what}." + safe)
    say("")
    say("[6/9] speed against accuracy at the three places")
    try:
        mode_tradeoff()
    except SystemExit:
        say("NOTE: the speed-against-accuracy table could not be made now. Run MEASURE_SPEED_VS_ACCURACY.bat later.")
    say("")
    say("[7/9] public task-graph benchmark")
    if os.path.exists(os.path.join(ROOT, "data", "stfg", "SyntheticTaskFlowGraphs.zip")):
        if run([PY, "-m", "benchmark_stfg.bench"]) != 0:
            say("NOTE: the benchmark did not finish; the older benchmark result stays.")
    else:
        say("  skipped (dataset not downloaded; RUN_PUBLIC_BENCHMARK.bat fetches it)")
    for i, (what, args) in enumerate([("result tables in the README", ["-m", "tools.make_summary"]),
                                      ("cloud server bundle for AWS", [os.path.join("cloud", "package_bundle.py")])], 8):
        say("")
        say(f"[{i}/9] {what}")
        if run([PY] + args) != 0:
            say(f"NOTE: step failed: {what} (the models and results above are fine).")
    say("")
    say("ALL THREE MODELS, THE AGENT AND THE RESULTS ARE UP TO DATE.")
    say("Stop the running demo (Enter in its window) and start it again to use them.")


def activate_candidate(data_rel):
    """Make models_candidate/ the models in use, then redo everything that depends on them."""
    import shutil
    cand = os.path.join(ROOT, "models_candidate")
    keep = os.path.join(ROOT, "models_before_" + time.strftime("%Y%m%d_%H%M%S"))
    shutil.copytree(os.path.join(ROOT, "models"), keep)
    say(f"Old models kept in {os.path.basename(keep)}")
    for f in ("gateway_model.onnx", "gateway_model.pt", "cloud_model.onnx", "cloud_model.pt"):
        if os.path.exists(os.path.join(cand, f)):
            shutil.copy2(os.path.join(cand, f), os.path.join(ROOT, "models", f))
    card = read_card()
    new = json.load(open(os.path.join(cand, "model_card.json")))
    card.update({r: new[r] for r in ("gateway", "cloud") if r in new})
    with open(os.path.join(ROOT, "models", "model_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    rebuild_everything(data_rel, keep)


def mode_retrain():
    """YOUR photos in data/real -> all three vision models, the agent, the rules and every result table, again."""
    ensure_env()
    real = os.path.join(ROOT, "data", "real")
    raw = os.path.join(real, "raw")
    if not os.path.isdir(os.path.join(real, "train")):
        if not os.path.isdir(raw):
            stop("no real photos found. Take them with the gateway's webcam first: python -m tools.capture_dataset "
                 "--label good --n 120, then --label damaged --n 120 (they land in data/real/raw).")
        n = count_images(raw)
        say(f"Photos found: {n['good']} good, {n['damaged']} damaged")
        if min(n.values()) < 40:
            stop("too few photos: at least 40 good and 40 damaged are needed (120 of each is the target)")
        if run([PY, "-m", "tools.capture_dataset", "--split"]) != 0:
            stop("could not split the photos into train / val / test")
    n = count_images(os.path.join(real, "train"))
    say(f"Training photos: {n['good']} good, {n['damaged']} damaged")
    m = train_candidate("data/real", 15, 12)
    if m["cloud"]["accuracy"] - m["gateway"]["accuracy"] < 0.02:
        say("NOTE: on your photos the gateway model is about as good as the cloud model. Then sending photos to the")
        say("      cloud gains nothing and the agent will (correctly) keep everything on the gateway.")
    activate_candidate("data/real")


def mode_public():
    """A downloaded public photo set (ZIP in Downloads) -> train the models -> switch only if it is worth it ->
    tiny ESP32 model, agent and every result. If the gateway and cloud models are already trained on these
    photos, they are kept and only the rest is (re)built."""
    ensure_env()
    if already_trained_on("data/public") and os.path.isdir(os.path.join(ROOT, "data", "public", "train", "good")):
        say("The gateway and cloud models are already trained on data/public - keeping them.")
        m = {r: (read_card()[r].get("test") or read_card()[r].get("val")) for r in ("gateway", "cloud")}
        for r in ("gateway", "cloud"):
            say(f"  {r:7s}: correct {100*m[r]['accuracy']:.1f}%")
        rebuild_everything("data/public")
        return
    say("Preparing the downloaded photos ...")
    if run([PY, "-m", "tools.prepare_public_dataset"]) != 0:
        stop("the downloaded photo set could not be prepared (is the ZIP in the Downloads folder?)")
    m = train_candidate("data/public", 12, 8)
    gap = m["cloud"]["accuracy"] - m["gateway"]["accuracy"]
    say("")
    if m["cloud"]["accuracy"] < 0.85:
        say(f"NOT SWITCHED: the cloud model is only {100*m['cloud']['accuracy']:.1f}% correct on these photos (need 85%).")
        say("The trained models are in models_candidate/ for inspection. The old models stay in use.")
        return
    if gap < 0.02:
        say(f"NOT SWITCHED: the gateway model ({100*m['gateway']['accuracy']:.1f}%) is as good as the cloud model "
            f"({100*m['cloud']['accuracy']:.1f}%) on these photos.")
        say("With no accuracy gap there is nothing to gain from the cloud, so the project would have no point")
        say("with these models. They are in models_candidate/. The old models stay in use.")
        return
    say(f"SWITCHING: cloud {100*m['cloud']['accuracy']:.1f}% vs gateway {100*m['gateway']['accuracy']:.1f}% on real photos "
        f"(gap {100*gap:.1f} points).")
    activate_candidate("data/public")


STFG_URL = "https://zenodo.org/api/records/10654551/files/SyntheticTaskFlowGraphs.zip/content"


def mode_benchmark():
    """Public benchmark: the same Q-learning idea on the Synthetic Task Flow Graphs dataset (Zenodo, CC BY 4.0)."""
    ensure_env()
    import glob
    import shutil
    d = os.path.join(ROOT, "data", "stfg")
    os.makedirs(d, exist_ok=True)
    z = os.path.join(d, "SyntheticTaskFlowGraphs.zip")
    if not os.path.exists(z):
        found = glob.glob(os.path.join(os.path.expanduser("~"), "Downloads", "SyntheticTaskFlowGraphs*.zip"))
        if found:
            shutil.copy2(max(found, key=os.path.getmtime), z)
            say("Using the dataset ZIP from the Downloads folder.")
        else:
            say("Downloading the public dataset (1.4 MB) from Zenodo, doi:10.5281/zenodo.10654551 ...")
            try:
                req = urllib.request.Request(STFG_URL, headers={"User-Agent": "SmartConveyorQC/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r, open(z, "wb") as f:
                    f.write(r.read())
            except Exception as e:
                stop(f"the dataset could not be downloaded ({e}). Download SyntheticTaskFlowGraphs.zip from "
                     "https://doi.org/10.5281/zenodo.10654551 into the Downloads folder and run this again.")
    if run([PY, "-m", "benchmark_stfg.bench"]) != 0:
        stop("the benchmark did not finish (see above)")
    say("Done: results/STFG_RESULTS.md and results/figures/STFG_benchmark.png")


# ----------------------------------------------------------------------------- remote control
def _kill_tree(p):
    if p.poll() is not None:
        return
    if WIN:
        subprocess.call(["taskkill", "/PID", str(p.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        p.terminate()
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()


def _free_port(port):
    """Stop whatever program is listening on this port (an older demo window)."""
    if not WIN:
        return
    try:
        out = subprocess.check_output(["netstat", "-ano", "-p", "TCP"], text=True, errors="replace")
    except Exception:
        return
    pids = set()
    for ln in out.splitlines():
        parts = ln.split()
        if len(parts) >= 5 and parts[1].endswith(f":{port}") and parts[3].upper() == "LISTENING" and parts[4].isdigit():
            pids.add(parts[4])
    for pid in pids:
        if pid != "0":
            subprocess.call(["taskkill", "/PID", pid, "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


REMOTE_JOBS = {"demo", "line", "public", "retrain", "benchmark", "test", "tradeoff", "esp32", "scores", "rebuild",
               "cloudpc", "pibundle"}       # the last two: old names, see MODES at the bottom
RUNNERS = ("demo", "line", "cloudpc")       # these hold port 8000 and the ESP32


def mode_remote():
    """Stays open. Claude asks for a step by putting a small request file into logs/inbox/; this window runs
    it. Only the steps of this launcher can be asked for (start/stop the demo, training, benchmark, tests,
    packaging) - nothing else. Close this window to switch the remote control off."""
    import glob
    inbox = os.path.join(LOGS, "inbox")
    os.makedirs(inbox, exist_ok=True)
    status_path = os.path.join(LOGS, "remote_status.json")
    jobs = {}                 # name -> Popen
    history = []

    def write_status():
        st = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "alive": True,
              "running": sorted(n for n, p in jobs.items() if p.poll() is None),
              "finished": {n: p.returncode for n, p in jobs.items() if p.poll() is not None},
              "history": history[-30:]}
        tmp = status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f, indent=1)
        os.replace(tmp, status_path)

    def start(name):
        if name in jobs and jobs[name].poll() is None:
            return f"{name}: already running"
        if name in RUNNERS:
            for other in RUNNERS:
                if other in jobs:
                    _kill_tree(jobs[other])
            _free_port(PORT)
            time.sleep(1.0)
        out = open(os.path.join(LOGS, f"remote_{name}.out"), "w", encoding="utf-8", errors="replace")
        jobs[name] = subprocess.Popen([sys.executable, os.path.join(ROOT, "tools", "go.py"), name], cwd=ROOT,
                                      stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        return f"{name}: started"

    say("REMOTE CONTROL IS ON. Keep this window open.")
    say("Claude can now start and stop the project's own steps (demo, real line, training, measurements, tests).")
    say("Close this window to switch it off.")
    write_status()
    last = 0.0
    try:
        while True:
            for f in sorted(glob.glob(os.path.join(inbox, "*.json"))):
                try:
                    with open(f) as fh:
                        req = json.load(fh)
                except Exception:
                    req = {}
                try:
                    os.remove(f)
                except OSError:
                    pass
                do, what = str(req.get("do", "")), str(req.get("what", ""))
                if do == "start" and what in REMOTE_JOBS:
                    msg = start(what)
                elif do == "stop" and what in jobs:
                    _kill_tree(jobs[what])
                    if what in RUNNERS:
                        _free_port(PORT)
                    msg = f"{what}: stopped"
                elif do == "stop" and what == "port":
                    _free_port(PORT)
                    msg = "port 8000 freed"
                else:
                    msg = f"ignored request {req!r}"
                history.append(f"{time.strftime('%H:%M:%S')}  {msg}")
                say(history[-1])
                write_status()
            if time.time() - last > 5:
                last = time.time()
                write_status()
            time.sleep(1.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in jobs.values():
            _kill_tree(p)
        try:
            with open(status_path, "w") as f:
                json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "alive": False, "history": history[-30:]}, f)
        except OSError:
            pass


def mode_packtiny():
    """Developer step: one small file with the shrunken photos, for working on the ESP32's tiny model."""
    rel, _ = photos_dir()
    if rel is None:
        stop("no training photos on this PC")
    if run([PY, "-m", "tools.pack_tiny_data", "--data", rel]) != 0:
        stop("could not pack the photos")


def mode_scores():
    """Score every validation and test photo with the three models (needed by the second-opinion decision)."""
    ensure_env()
    rel, _ = photos_dir()
    if rel is None:
        stop("no training photos on this PC")
    if run([PY, "-m", "tools.photo_scores", "--data", rel]) != 0:
        stop("could not score the photos")


def mode_shift():
    """Score the same photos again after making them darker / blurred (for the 'photos changed' experiment)."""
    ensure_env()
    rel, _ = photos_dir()
    if rel is None:
        stop("no training photos on this PC")
    if run([PY, "-m", "tools.shift_scores", "--data", rel]) != 0:
        stop("could not score the changed photos")


def mode_rebuild():
    """Models unchanged: redo the scores, the agent, the experiments and the summaries."""
    ensure_env()
    rel, _ = photos_dir()
    if rel is None:
        stop("no training photos on this PC")
    rebuild_everything(rel)


def mode_job():
    """For a REMOTE_CONTROL window opened before an update: it only knows the old step names. The step to run
    is read from logs/remote_mode.txt (one of this launcher's own steps); without that file: the measurement."""
    name = "tradeoff"
    path = os.path.join(LOGS, "remote_mode.txt")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                name = f.read().strip().lower() or "tradeoff"
            os.remove(path)
        except OSError:
            pass
    if name not in JOB_STEPS:
        stop(f"'{name}' is not a step of this launcher")
    say(f"step asked for: {name}")
    MODES[name]()


JOB_STEPS = ("tradeoff", "scores", "shift", "rebuild", "public", "benchmark", "test", "esp32", "packtiny")

MODES = {"camera": mode_camera, "scores": mode_scores, "shift": mode_shift, "rebuild": mode_rebuild, "job": mode_job, "demo": mode_demo, "line": mode_line, "aws": mode_aws, "esp32": mode_esp32, "tradeoff": mode_tradeoff,
         "test": mode_test, "retrain": mode_retrain, "public": mode_public, "benchmark": mode_benchmark,
         "remote": mode_remote,
         # names from the 2-node version, kept so a REMOTE_CONTROL window opened before the update still works:
         "cloudpc": mode_line, "pibundle": mode_job, "packtiny": mode_packtiny}


def main():
    global _log
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if mode not in MODES:
        print("usage: python tools/go.py " + " | ".join(MODES))
        return 2
    os.makedirs(LOGS, exist_ok=True)
    _log = open(os.path.join(LOGS, f"go_{mode}.log"), "w", encoding="utf-8", errors="replace")
    say(f"Smart Conveyor QC - {mode}   ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    say(f"folder: {ROOT}")
    say(f"python: {sys.version.split()[0]}  ({sys.executable})")
    try:
        if mode != "remote":
            migrate()
        MODES[mode]()
    except SystemExit:
        raise
    except Exception as e:                     # never die silently: the log must say why
        import traceback
        say(traceback.format_exc())
        stop(f"unexpected error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
