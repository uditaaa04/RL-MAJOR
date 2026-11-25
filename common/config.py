"""Load config/system.yaml (one config for gateway, cloud server and simulator)."""
from __future__ import annotations

import copy
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(ROOT, "config", "system.yaml")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    with open(path or DEFAULT_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _merge(cfg, overrides or {})


def project_path(p: str) -> str:
    """Resolve a path from the config relative to the project root."""
    return p if os.path.isabs(p) else os.path.join(ROOT, p)
