"""Reads the experiment result files (results/*.csv, *.json) and the model card for the dashboard's
Results page. Read-only; a missing file simply leaves that part out."""
from __future__ import annotations

import csv
import json
import os

from common.config import project_path


def _num(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in str(v) and "e" not in str(v).lower() else f
    except (TypeError, ValueError):
        return v


def _rows(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_results(results_dir: str | None = None, card_path: str | None = None) -> dict:
    res = results_dir or project_path("results")
    out = {"simulated": True, "generated": None}
    e1 = _rows(os.path.join(res, "E1_steady.csv"))
    if e1:
        keep = ("scenario", "policy", "reward_mean", "miss_rate", "escape_rate", "false_reject_rate", "cloud_share",
                "gateway_share", "edge_share", "accuracy", "ask_share", "fallback_share")
        out["e1"] = [{k: r.get(k) for k in keep} for r in e1]
        try:
            import time
            out["generated"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(os.path.join(res, "E1_steady.csv"))))
        except OSError:
            pass
    e2 = _rows(os.path.join(res, "E2_changes.csv"))
    if e2:
        out["e2"] = [{"change": r.get("change"), "policy": r.get("policy"), "missed_during": r.get("missed_during_%"),
                      "missed_after": r.get("missed_after_%"), "recover": r.get("boxes_to_recover"),
                      "esp32_during": r.get("esp32_share_during_%")} for r in e2]
    e3 = _json(os.path.join(res, "E3_pretraining_value.json"))
    if e3:
        out["e3"] = e3
    e5 = _rows(os.path.join(res, "E5_mismatch.csv"))
    if e5:
        out["e5"] = [{"policy": r.get("policy"), "reward": r.get("reward_all"), "missed": r.get("missed_%")} for r in e5]
    e6 = _json(os.path.join(res, "E6_sensitivity.json"))
    if e6:
        out["e6"] = {"generated": e6.get("generated"), "auc": e6.get("auc"), "settings": e6.get("settings"),
                     "rows": [{k: r.get(k) for k in ("setting", "policy", "reward_mean", "miss_rate", "cloud_share")}
                              for r in e6.get("rows", [])]}
    e7 = _rows(os.path.join(res, "E7_budget.csv"))
    if e7:
        out["e7"] = [{k: r.get(k) for k in ("network", "cloud_calls_per_min", "policy", "reward_mean", "accuracy",
                                            "cloud_share", "miss_rate")} for r in e7]
    e8 = _rows(os.path.join(res, "E8_photos_change.csv"))   # the photos get darker / blurred in the middle of a run
    if e8:
        out["e8"] = [{k: r.get(k) for k in ("photos", "cloud_calls_per_min", "policy", "cost_before", "cost_first_500",
                                            "cost_last", "ask_share_last", "accuracy_last")} for r in e8]
    sh = _json(os.path.join(res, "shift_scores.json"))      # how correct each model still is on the changed photos
    if sh and sh.get("shifts"):
        out["shift"] = {"clean": sh.get("clean"), "shifts": sh.get("shifts")}
    e9 = _json(os.path.join(res, "E9_how_sure.json"))       # every gap with its range over re-drawn test sets
    if e9 and e9.get("pairs"):
        out["e9"] = e9
    cas = _json(os.path.join(res, "cascade.json"))          # real photos: what a cloud second opinion is worth
    if cas and cas.get("curve"):
        out["cascade"] = cas
    tr = _json(os.path.join(res, "tradeoff.json"))
    if tr and tr.get("tiers"):
        out["tradeoff"] = tr
    stfg = _json(os.path.join(res, "STFG_summary.json"))
    if stfg and not stfg.get("standin"):
        out["stfg"] = stfg
    card = _json(card_path or project_path("models/model_card.json"))
    if card:
        if "gateway" not in card and "edge" in card and "mobilenet" in str(card["edge"].get("arch", "")):
            card["gateway"] = card.pop("edge")          # card written before the ESP32 was added
        models = {}
        for role in ("edge", "gateway", "cloud"):
            m = card.get(role)
            if not m:
                continue
            src = m.get("test") or m.get("val") or {}
            models[role] = {"arch": m.get("arch"), "img": m.get("img"), "threshold": m.get("threshold"),
                            "data": m.get("data"), "init": "pretrained" if str(m.get("init")) == "imagenet" else "from scratch",
                            "trained_at": m.get("trained_at"), "n": src.get("n"), "params": m.get("params"),
                            "bytes": m.get("bytes"), "channels": m.get("channels"),
                            "accuracy": src.get("accuracy"), "recall_damaged": src.get("recall_damaged"),
                            "false_reject_rate": src.get("false_reject_rate")}
        out["models"] = models
        data = str((card.get("gateway") or card.get("cloud") or {}).get("data", "")).replace("\\", "/")
        out["real_photos"] = "real" in data                 # photos taken on this project's own conveyor
        out["photo_source"] = "own" if "real" in data else ("public" if "public" in data else "synthetic")
        src = project_path(os.path.join(data, "SOURCE.md")) if data else ""
        if out["photo_source"] == "public" and os.path.exists(src):
            try:
                with open(src, encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f.read().splitlines()]
                out["photo_source_name"] = lines[0].replace("# Photo source:", "").strip()
                out["photo_source_cite"] = next((ln[len("- Cite as:"):].strip() for ln in lines if ln.startswith("- Cite as:")), "")
            except OSError:
                pass
    out["real_results"] = os.path.exists(os.path.join(res, "REAL_RESULTS.md"))
    return out
