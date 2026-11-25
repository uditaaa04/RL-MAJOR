"""HTTP sessions from the gateway to the cloud server: every request carries the API key and the device id."""
from __future__ import annotations

import os

import requests


def api_key(cfg: dict | None = None, override: str | None = None) -> str:
    """CLI option > environment variable CONVEYOR_API_KEY > config server.api_key."""
    if override:
        return override.strip()
    env = os.environ.get("CONVEYOR_API_KEY", "").strip()
    if env:
        return env
    return str(((cfg or {}).get("server") or {}).get("api_key") or "").strip()


def device_id(cfg: dict | None = None) -> str:
    return str(os.environ.get("CONVEYOR_DEVICE_ID") or ((cfg or {}).get("server") or {}).get("device_id") or "pi-1")


def make_session(key: str = "", device: str = "pi-1") -> requests.Session:
    s = requests.Session()
    if key:
        s.headers["X-API-Key"] = key
    s.headers["X-Device-Id"] = device
    s.headers["User-Agent"] = "smart-conveyor-pi/1.0"
    return s
