"""Second opinion: confidence levels, the cloud-call budget, the fallback, and the photo-replay simulator."""
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import cascade  # noqa: E402
from common.config import load_config  # noqa: E402
from common.policies import AdaptiveRule, AskAlways, ConfidenceRule, QPolicy  # noqa: E402
from common.rl import CLOUD, EDGE, GATEWAY, Obs, QAgent, RewardModel, StateSpace  # noqa: E402
from gateway_node.budget import CloudBudget  # noqa: E402
from sim.env import ConveyorSim, SimParams, summarize  # noqa: E402
from sim.scenarios import BAD, GOOD, SERVER_SLOW  # noqa: E402

CFG = load_config()
SPACE = StateSpace(CFG["state_bins"])
METRICS = {"edge": {"recall_damaged": 0.95, "false_reject_rate": 0.7}, "gateway": {"recall_damaged": 0.95, "false_reject_rate": 0.2},
           "cloud": {"recall_damaged": 0.98, "false_reject_rate": 0.08}}
TABLE = cascade.synthetic_table(METRICS, n=4000, seed=1)


def reward():
    return RewardModel(CFG["reward"], 0.3, METRICS["gateway"], METRICS["cloud"], METRICS["edge"], TABLE.val.stats())


def sim(seed, split=None):
    return ConveyorSim(SimParams(), reward(), np.random.default_rng(seed), split or TABLE.test)


def test_confidence_levels_follow_the_margin():
    edges = [1.0, 2.0, 4.0]
    assert [cascade.conf_level(m, edges) for m in (0.2, 1.5, 3.0, 9.0)] == [4, 3, 2, 1]
    assert float(cascade.margin(0.5, 0.5)) == 0.0
    lv = TABLE.val.conf
    assert set(np.unique(lv)) == {1, 2, 3, 4}
    assert abs((lv == 4).mean() - 0.10) < 0.02 and abs((lv == 1).mean() - 0.50) < 0.02      # 10 / 15 / 25 / 50 %


def test_made_up_scores_match_the_model_metrics():
    v = TABLE.val
    for role, m in METRICS.items():
        recall = v.verdict[role][v.y == 1].mean()
        fr = v.verdict[role][v.y == 0].mean()
        assert abs(recall - m["recall_damaged"]) < 0.03 and abs(fr - m["false_reject_rate"]) < 0.04


def test_cloud_is_worth_most_where_the_gateway_is_least_sure():
    r = reward()
    assert r.gain(4) > r.gain(1) and r.gain(4) > 0
    assert r(CLOUD, True, 0.0, 4) > r(GATEWAY, True, 0.0, 4)          # very unsure: the cloud's answer is worth more
    assert r(GATEWAY, False, 0.0, 4) == r(GATEWAY, False, 0.0, 0)     # a missed box costs the same whatever the level
    assert r(EDGE, True, 0.0, 3) == r(EDGE, True, 0.0, 0)             # the ESP32 has no confidence levels


def test_photo_table_save_and_load(tmp_path):
    p = str(tmp_path / "scores.npz")
    TABLE.save(p)
    t = cascade.PhotoTable.load(p)
    assert np.array_equal(t.test.y, TABLE.test.y) and np.allclose(t.val.edges, TABLE.val.edges)
    assert np.array_equal(t.test.conf, TABLE.test.conf)


def test_every_policy_sees_the_same_boxes():
    a = sim(7).run(AdaptiveRule(), [(0, GOOD)], 200).arrays()
    b = sim(7).run(AskAlways(), [(0, GOOD)], 200).arrays()
    assert np.array_equal(a["label"], b["label"])


def test_late_cloud_answer_falls_back_to_the_gateway_verdict():
    """Server overloaded: the old way (pick the cloud) misses boxes, the second-opinion way does not."""
    old = summarize(sim(3).run(type("AlwaysCloud", (AdaptiveRule,), {"decide": lambda self, i, o: CLOUD})(),
                               [(0, SERVER_SLOW)], 300))
    new = summarize(sim(3).run(AskAlways(), [(0, SERVER_SLOW)], 300))
    assert old["miss_rate"] > 0.3
    assert new["miss_rate"] < 0.02 and new["fallback_share"] > 0.3


def test_cloud_call_budget_is_respected():
    c = replace(GOOD, budget_per_min=6.0, budget_burst=3.0, gap_s=1.0)
    log = sim(5).run(AskAlways(), [(0, c)], 400).arrays()
    minutes = 400 * 1.0 / 60.0
    assert log["asked"].sum() <= 6.0 * minutes * 1.15 + 3                 # refill + what was saved up
    assert log["asked"].sum() >= 6.0 * minutes * 0.6


def test_confidence_rule_asks_only_for_unsure_boxes():
    log = sim(9).run(ConfidenceRule(min_conf=3, margin_s=0.0), [(0, GOOD)], 400).arrays()
    assert log["asked"].any()
    assert (log["conf"][log["asked"]] >= 3).all()
    assert 0.1 < log["asked"].mean() < 0.6


def test_agent_never_asks_without_a_call_and_never_sends_a_checked_box_to_the_esp32():
    ag = QAgent(SPACE.n_states, 3, epsilon=1.0, seed=0)                   # explores all the time
    pol = QPolicy(ag, SPACE, learn=True, explore=True)
    for i in range(300):
        o = Obs(40, 0.4, 0, 0, 0.2, 60, 5, 70, conf=1 + i % 4, can_cloud=(i % 3 != 0), calls_left=0.0 if i % 3 == 0 else 2.0)
        a = pol.decide(i, o)
        assert a != EDGE
        assert not (a == CLOUD and not o.can_cloud)


def test_bad_network_costs_no_boxes_with_a_second_opinion():
    s = summarize(sim(11).run(AskAlways(), [(0, BAD)], 300))
    assert s["miss_rate"] < 0.02


def test_budget_bucket():
    b = CloudBudget(per_min=60, burst=2)
    assert b.take() and b.take() and not b.take()                         # two saved up, then empty
    assert b.snapshot()["can"] is False
    b.set(0)
    s = b.snapshot()
    assert s["limited"] is False and s["can"] and b.take()                # no limit
