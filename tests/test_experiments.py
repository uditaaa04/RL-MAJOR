"""Smoke tests: the two newest experiments run end to end on a tiny setting and write what the dashboard reads."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("matplotlib")

from common.config import load_config, project_path  # noqa: E402
from common.rl import QAgent  # noqa: E402
from sim import experiments as E  # noqa: E402
from sim.train_q import build, load_shift_tables  # noqa: E402


@pytest.fixture(scope="module")
def world():
    cfg = load_config()
    params, reward, space = build(cfg)
    agent = QAgent.load(project_path(cfg["agent"]["q_table"]))
    with open(project_path(cfg["agent"]["rule_params"])) as f:
        rp = json.load(f)
    return cfg, params, reward, space, rp, E.factories(cfg, agent, space, rp)


def test_photos_change_experiment_writes_its_table(world, tmp_path, monkeypatch):
    cfg, params, reward, space, rp, facs = world
    monkeypatch.setattr(E, "RES", str(tmp_path))
    monkeypatch.setattr(E, "FIG", str(tmp_path))
    if not load_shift_tables(cfg, reward):                 # a fresh copy without the changed-photo scores:
        reward.photo_shift = {"blur": reward.photos}       # the run itself must still work
    rows = E.e8_photos_change(E.setup(), cfg, params, reward, facs, rp, seeds=1, before=60, after=200, win=20)
    names = {r["policy"] for r in rows}
    assert {"Ask the cloud always", "Confidence rule", "Q-learning (online + spot checks)"} <= names
    assert all(r["cost_last"] >= 0 and 0 <= r["ask_share_last"] <= 1 for r in rows)
    assert os.path.exists(tmp_path / "E8_photos_change.csv") and os.path.exists(tmp_path / "E8_photos_change.png")
    from server.results import load_results
    assert load_results(str(tmp_path))["e8"][0]["policy"]


def test_how_sure_experiment_gives_a_range_for_every_gap(world, tmp_path, monkeypatch):
    cfg, params, reward, space, rp, facs = world
    monkeypatch.setattr(E, "RES", str(tmp_path))
    out = E.e9_how_sure(params, reward, facs, cfg, reps=3, boxes=40)
    assert len(out["pairs"]) == len(E.E9_PAIRS)
    for p in out["pairs"]:
        assert p["low"] <= p["diff"] <= p["high"] and 0 <= p["a_cheaper_share"] <= 1
    from server.results import load_results
    assert load_results(str(tmp_path))["e9"]["reps"] == 3
