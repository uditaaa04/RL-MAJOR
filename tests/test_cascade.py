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


# ----------------------------------------------------------------------------- spot checks / photos that change
def test_five_budget_levels_tell_plenty_from_a_few():
    lab = SPACE.labels()["budget"]
    assert lab == ["none", "1", "2", "3-4", "5 or more"]
    b = lambda calls: SPACE.bins_of(Obs(40, 0.4, 0, 0, 0.2, conf=3, calls_left=calls))[6]
    assert [b(0.4), b(1.0), b(2.5), b(3.0), b(4.9), b(5.0), b(99.0)] == [0, 1, 2, 3, 3, 4, 4]     # 99 = no limit


def test_spot_check_tally_starts_at_the_card_and_follows_the_line():
    live = cascade.LiveGain([0.0, -0.3, 0.3, 1.2, 2.8], prior_n=60, half_life=300)
    assert live.gain(3) == 1.2 and live.delta(3) == 0.0                   # nothing checked yet: the card's number
    for _ in range(300):
        live.update(2, -10.0)                                             # on this line the cloud makes level 2 WORSE
    assert live.gain(2) < -5.0 and live.delta(2) < -5.0
    assert live.gain(3) == 1.2                                            # other levels are untouched
    for _ in range(3000):
        live.update(2, +4.0)                                              # the line changes again: old boxes fade
    assert 3.0 < live.gain(2) < 4.0
    for _ in range(60):
        live.note_call(False)                                             # cloud answers come too late: a bonus counts less
    assert 0 < live.delta(2) < 0.2 * (live.gain(2) - 0.3)
    again = cascade.LiveGain([0.0, -0.3, 0.3, 1.2, 2.8])
    again.restore(live.state())
    assert abs(again.gain(2) - live.gain(2)) < 1e-9 and again.table()[1]["checked"] == 3300


def test_cloud_bonus_changes_the_choice_for_one_decision_only():
    ag = QAgent(SPACE.n_states, 3, epsilon=0.0, seed=0, prefer=GATEWAY)
    ag.q[5] = [-9.0, -2.0, -2.5]
    assert ag.greedy(5, (GATEWAY, CLOUD)) == GATEWAY
    assert ag.greedy(5, (GATEWAY, CLOUD), cloud_bonus=1.0) == CLOUD
    assert ag.greedy(5, (GATEWAY, CLOUD), cloud_bonus=-1.0) == GATEWAY
    assert list(ag.q[5]) == [-9.0, -2.0, -2.5]                            # the table itself is not touched


def test_redrawn_test_set_keeps_size_and_classes():
    r = TABLE.test.resample(np.random.default_rng(0))
    assert len(r) == len(TABLE.test) and int(r.y.sum()) == int(TABLE.test.y.sum())
    assert r.edges == TABLE.test.edges and not np.array_equal(r.p["cloud"], TABLE.test.p["cloud"])


def _worse_cloud_table():
    """The same boxes, but the cloud's model now gets many of them wrong (as on blurred photos)."""
    t = TABLE.test
    rng = np.random.default_rng(5)
    flip = rng.random(len(t)) < 0.35
    cloud = np.where(flip, np.where(t.y == 1, 0.01, 0.99), t.p["cloud"])  # wrong on purpose
    return cascade.PhotoSplit(t.y, {**t.p, "cloud": cloud}, t.thr, t.edges)


def test_photos_can_change_in_the_middle_of_a_run():
    s = sim(21)
    s.photo_sets = {"worse": _worse_cloud_table()}
    c = replace(GOOD, gap_s=1.0)
    log = s.run(AskAlways(), [(0, c), (300, replace(c, photos="worse"))], 600)
    same = sim(21).run(AskAlways(), [(0, c)], 600)
    assert np.array_equal(np.asarray(log.label), np.asarray(same.label))                 # the very same boxes
    assert summarize(log, 0, 300)["accuracy"] == summarize(same, 0, 300)["accuracy"]
    assert summarize(log, 300, 600)["accuracy"] < summarize(same, 300, 600)["accuracy"] - 0.1


def _spot_run(spot_rate, with_tally):
    r = reward()
    ag = QAgent(SPACE.n_states, 3, alpha=0.1, gamma=0.9, epsilon=0.02, seed=1, prefer=GATEWAY)
    ag.q[:, GATEWAY], ag.q[:, CLOUD], ag.q[:, EDGE] = -2.0, -1.0, -9.0                   # pre-trained belief: always ask
    ag.q0 = ag.q.copy()
    live = cascade.LiveGain([0.0] + [r.gain(c) for c in range(1, 5)], 60, 300) if with_tally else None
    ag.alpha = 0.0                                                        # the table itself stays as it is
    pol = QPolicy(ag, SPACE, learn=with_tally, explore=True, live=live)   # only the tally can change the choice
    s = ConveyorSim(SimParams(), r, np.random.default_rng(31), TABLE.test)
    s.photo_sets, s.spot_rate = {"worse": _worse_cloud_table()}, spot_rate
    c = replace(GOOD, gap_s=1.0, photos="worse")
    return summarize(s.run(pol, [(0, c)], 2500), 1500, 2500), live


def test_spot_checks_stop_the_agent_asking_a_cloud_that_got_worse():
    blind, _ = _spot_run(0.0, False)
    seeing, live = _spot_run(1.0, True)
    assert blind["ask_share"] > 0.9                                       # believes its table: keeps asking
    assert seeing["ask_share"] < 0.5 * blind["ask_share"]                 # measured that the cloud now costs money
    assert seeing["reward_mean"] > blind["reward_mean"] + 1.0             # and the boxes are cheaper for it
    assert sum(row["checked"] for row in live.table()) > 20
    assert min(row["live"] for row in live.table()) < 0


def test_without_known_truth_the_tally_never_moves():
    _, live = _spot_run(0.0, True)
    assert all(row["checked"] == 0 and row["live"] == row["card"] for row in live.table())
