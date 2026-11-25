"""
Which USB serial ports this PC has, and whether an ESP32 with this project's program answers on one.

    python -m tools.ports            # list only: no port is opened
    python -m tools.ports --probe    # also ask each likely port "who are you?" (one short line; nothing is changed,
                                     # the belt and the arm are not moved)

Why: before the first hardware run one wants to know three things without opening the Arduino IDE:
is the board plugged in, does Windows have a driver for its USB chip, and is the project's program on it.
"""
from __future__ import annotations

import argparse
import os
import subprocess

USB_NAMES = "CP210|CH340|CH341|CH343|CH910|USB.{0,5}Ser|UART|FT232"


def windows_usb_serial_devices() -> list[str]:
    """Windows only: plugged-in USB-to-serial chips WITH their state, also the ones that have no driver yet
    (those never show up as a COM port). Read-only question to Windows; [] when it cannot be asked."""
    if os.name != "nt":
        return []
    cmd = ("Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -match '" + USB_NAMES + "' } | "
           "ForEach-Object { $_.Status + ' | ' + $_.Class + ' | ' + $_.FriendlyName }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="ask each likely port which program is on the board")
    a = ap.parse_args(argv)
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        print("pyserial is missing (the launcher installs it)", flush=True)
        return 1
    from gateway_node.esp32_link import Esp32Link, list_ports
    ports = list(lp.comports())
    print("Serial ports on this PC:", flush=True)
    for p in ports:
        print(f"  {p.device}: {p.description}".encode("ascii", "replace").decode(), flush=True)
    if not ports:
        print("  none", flush=True)
    likely = list_ports()
    print("Ports that can be an ESP32 (USB serial): " + (", ".join(p for p, _ in likely) or "none"), flush=True)
    devs = windows_usb_serial_devices()
    if devs:
        print("USB-to-serial chips Windows sees (state | kind | name):", flush=True)
        for d in devs:
            print(("  " + d).encode("ascii", "replace").decode(), flush=True)
        if not likely and any(not d.upper().startswith("OK") for d in devs):
            print("  -> a chip is plugged in but has NO DRIVER. Install the driver named after it "
                  "(CP210x driver for CP2102, CH340 driver for CH340), then plug the board in again.", flush=True)
    if not likely:
        print("NO ESP32 PORT. Is the board plugged in with a DATA cable (a charge-only cable shows nothing)?", flush=True)
        return 2
    if not a.probe:
        return 0
    found = 3
    for port, _ in likely:
        link = Esp32Link(port, verbose=False)
        try:
            link.open(hello_timeout=3.0)
            print(f"  {port}: the project's program answers: {link.firmware}", flush=True)
            found = 0
        except Exception as e:
            msg = str(e)
            if "denied" in msg.lower() or "permission" in msg.lower():
                print(f"  {port}: in use by another program (close the Arduino Serial Monitor)", flush=True)
            else:
                print(f"  {port}: no answer from the project's program (not flashed yet, or another device)", flush=True)
        finally:
            try:
                link.close()
            except Exception:
                pass
    return found


if __name__ == "__main__":
    raise SystemExit(main())
