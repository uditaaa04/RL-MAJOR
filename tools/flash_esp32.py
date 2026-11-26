"""
Put the project's firmware (esp32_node/esp32_node.ino, version 4 with Wi-Fi) on the ESP32 - no Arduino IDE
clicking needed. Uses the arduino-cli that comes inside Arduino IDE 2 and the ESP32 boards it already installed.

    python -m tools.flash_esp32              # find the board, compile, upload, then show what the board says
    python -m tools.flash_esp32 --port COM5  # a fixed port
    python -m tools.flash_esp32 --check      # compile only (nothing is uploaded)

Board settings used: "ESP32 Dev Module", Partition Scheme "Huge APP (3MB No OTA/1MB SPIFFS)" (Wi-Fi + https need
the room). Wi-Fi names and passwords come from esp32_node/secrets.h (copy secrets.example.h; never uploaded to GitHub).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKETCH = os.path.join(ROOT, "esp32_node")
FQBN = "esp32:esp32:esp32:PartitionScheme=huge_app"


def find_cli() -> str | None:
    """arduino-cli: the one inside Arduino IDE 2, or one on the PATH."""
    cands = []
    for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)")):
        if base:
            cands.append(os.path.join(base, "Programs", "Arduino IDE", "resources", "app", "lib", "backend",
                                      "resources", "arduino-cli.exe"))
            cands.append(os.path.join(base, "Arduino IDE", "resources", "app", "lib", "backend", "resources",
                                      "arduino-cli.exe"))
    cands += glob.glob("/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/resources/arduino-cli")
    for c in cands:
        if os.path.isfile(c):
            return c
    return shutil.which("arduino-cli")


def run(cmd: list[str], timeout: float = 900, echo: bool = True) -> tuple[int, str]:
    print("> " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    out = []
    t0 = time.time()
    for line in p.stdout:
        out.append(line)
        if echo:
            print("  " + line.rstrip(), flush=True)
        if time.time() - t0 > timeout:
            p.kill()
            break
    return p.wait(), "".join(out)


def esp32_core(cli: str) -> str | None:
    code, out = run([cli, "core", "list", "--format", "json"], 120, echo=False)   # long JSON: not printed
    if code != 0:
        print(out[-2000:], flush=True)
        return None
    data = None
    for start in [0] + [i for i in (out.find("{"), out.find("[")) if i > 0]:
        try:
            data = json.loads(out[start:])
            break
        except ValueError:
            continue
    if data is None:
        return None
    plats = data if isinstance(data, list) else (data.get("platforms") or [])
    for p in plats:
        pid = p.get("id") or ""
        if pid == "esp32:esp32":
            return p.get("installed_version") or p.get("installed") or "?"
    return None


def pick_port(want: str | None) -> str | None:
    if want:
        return want
    from gateway_node.esp32_link import list_ports
    ports = [p for p, _ in list_ports()]
    if len(ports) == 1:
        return ports[0]
    if not ports:
        print("No ESP32 found on USB. Plug it in with a DATA cable (some cables only charge).", flush=True)
    else:
        print("More than one USB serial port: " + ", ".join(ports) + ". Unplug the others or use --port.", flush=True)
    return None


def listen(port: str, seconds: float = 25.0) -> bool:
    """After the upload: ask the board who it is, then show its Wi-Fi news for a while."""
    import serial
    time.sleep(1.5)
    try:
        s = serial.Serial(port, 460800, timeout=0.2)
    except Exception as e:
        print(f"Could not open {port} to check the board: {e}", flush=True)
        return False
    ok = False
    try:
        s.reset_input_buffer()
        t_end = time.time() + seconds
        asked = 0.0
        buf = b""
        while time.time() < t_end:
            if time.time() - asked > 4:
                s.write(b"H\nN\n")
                asked = time.time()
            buf += s.read(512)
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("ascii", "ignore").strip()
                if line.startswith("H SCQC-ESP32"):
                    if not ok:
                        print(f"  board answers: {line}", flush=True)
                    ok = True
                elif line.startswith("# ") or line.startswith("N "):
                    print("  " + line, flush=True)
    finally:
        s.close()
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--check", action="store_true", help="compile only")
    a = ap.parse_args(argv)
    cli = find_cli()
    if not cli:
        print("arduino-cli was not found (it comes with Arduino IDE 2). Either install Arduino IDE 2, or flash by "
              "hand: Arduino IDE > open esp32_node/esp32_node.ino > Tools > Board 'ESP32 Dev Module' > Tools > "
              "Partition Scheme 'Huge APP (3MB No OTA/1MB SPIFFS)' > port > Upload.", flush=True)
        return 2
    print(f"arduino-cli: {cli}", flush=True)
    ver = esp32_core(cli)
    if not ver:
        print("The ESP32 boards are not installed in Arduino IDE. Arduino IDE > Boards Manager > search 'esp32' > "
              "install 'esp32 by Espressif Systems', then run this again.", flush=True)
        return 2
    print(f"ESP32 boards installed: version {ver}", flush=True)
    if not os.path.exists(os.path.join(SKETCH, "secrets.h")):
        print("NOTE: esp32_node/secrets.h is missing: the board has no Wi-Fi name to join, so it opens its setup "
              "hotspot (password in secrets.example.h). Copy secrets.example.h to secrets.h and fill it in.", flush=True)
    port = None if a.check else pick_port(a.port)
    if not a.check and not port:
        return 2
    cmd = [cli, "compile", "--fqbn", FQBN, SKETCH]
    if port:
        cmd[2:2] = ["--upload", "-p", port]
    code, out = run(cmd)
    if code != 0:
        if "Failed to connect" in out or "Wrong boot mode" in out:
            print("The board did not start its upload mode. Hold the BOOT button on the ESP32, run this again, and "
                  "let go when 'Connecting...' appears. Also close the Arduino Serial Monitor and the gateway.",
                  flush=True)
        elif "Access is denied" in out or "could not open port" in out.lower() or "busy" in out.lower():
            print(f"{port} is in use: close the Arduino Serial Monitor and any running gateway window, then run "
                  f"this again.", flush=True)
        else:
            print("Compiling or uploading failed (see the lines above).", flush=True)
        return 1
    if a.check:
        print("Compiles cleanly.", flush=True)
        return 0
    print("Uploaded. Checking the board ...", flush=True)
    if not listen(port):
        print("The board did not answer after the upload. Press its EN (reset) button and run CHECK_ESP32.", flush=True)
        return 1
    print("The ESP32 runs firmware 4. Its Wi-Fi news is shown above (lines starting with '#').", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    sys.exit(main())
