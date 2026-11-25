"""Quick checks of the core logic.   Run:  python -m pytest -q tests"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import load_config  # noqa: E402
from common.policies import AdaptiveRule, DeadlineRule, QPolicy, ThresholdRule  # noqa: E402
from common.rl import CLOUD, EDGE, GATEWAY, Obs, QAgent, RewardModel, StateSpace  # noqa: E402
from sim.env import ConveyorSim, SimParams, summarize  # noqa: E402
from sim.scenarios import BAD, GOOD  # noqa: E402

CFG = load_config()
SPACE = StateSpace(CFG["state_bins"])
REWARD = RewardModel(CFG["reward"], 0.3, {"recall_damaged": 0.9, "false_reject_rate": 0.06, "accuracy": 0.93},
                     {"recall_damaged": 0.98, "false_reject_rate": 0.02, "accuracy": 0.98},
                     {"recall_damaged": 0.85, "false_reject_rate": 0.30, "accuracy": 0.75})


def obs(**kw):
    d = dict(rtt_ms=10, budget_s=0.45, gw_q=0, cloud_q=0, cpu=0.2, cloud_ms=40)
    d.update(kw)
    return Obs(**d)


def test_state_space_size_and_range():
    assert SPACE.n_states == 8 * 3 * 3 * 3 * 2 * 5 * 5          # + how sure the gateway is, + cloud calls left
    for o in [obs(), obs(rtt_ms=900, budget_s=0.05, cloud_ms=2000, cpu=1.0, gw_q=5, cloud_q=3)]:
        assert 0 <= SPACE.index(o) < SPACE.n_states


def test_slack_feature():
    o = obs(rtt_ms=10, budget_s=0.4, cloud_ms=50)
    assert abs(SPACE.slack_ms(o) - (400 - 50)) < 1e-6          # measured laptop time dominates
    o = obs(rtt_ms=300, budget_s=0.4, cloud_ms=50)
    assert abs(SPACE.slack_ms(o) - (400 - 330)) < 1e-6         # fresh bad ping dominates


def test_gateway_feature_tells_fits_from_too_slow():
    assert SPACE.gw_load(obs(gw_ms=8, budget_s=0.45)) == 0            # 8 ms of 450 ms: fits easily
    assert SPACE.gw_load(obs(gw_ms=250, budget_s=0.45)) == 1          # tight
    assert SPACE.gw_load(obs(gw_ms=480, budget_s=0.45)) == 2          # too slow
    assert SPACE.gw_load(obs(gw_ms=100, gw_q=4, budget_s=0.45)) == 2  # fast model, but four boxes are waiting
    assert SPACE.gw_load(obs(gw_ms=0, budget_s=0.45)) == 0            # not measured yet: assume the default


def test_reward_ordering():
    # better model = better reward; any answer in time beats a missed box
    assert REWARD(CLOUD, True, 0) > REWARD(GATEWAY, True, 0) > REWARD(EDGE, True, 0) > REWARD(CLOUD, False, 0)
    assert REWARD(GATEWAY, True, 1.0) < REWARD(GATEWAY, True, 0.0)   # energy costs something


def test_q_update_moves_toward_target():
    ag = QAgent(4, 3, alpha=0.5, gamma=0.0)
    ag.update(0, CLOUD, -10.0, None)
    assert ag.q[0, CLOUD] == -5.0


def test_safe_exploration_never_tries_a_clearly_bad_option():
    ag = QAgent(1, 3, epsilon=1.0, seed=0, prefer=GATEWAY)
    ag.explore_gap = 3.0
    ag.q[0, EDGE], ag.q[0, GATEWAY], ag.q[0, CLOUD] = -60.0, -1.0, -50.0    # ESP32 and cloud clearly bad
    assert all(ag.act(0) == GATEWAY for _ in range(200))
    ag.q[0, CLOUD] = -2.0                         # cloud close -> may explore it, never the ESP32
    assert {ag.act(0) for _ in range(300)} == {CLOUD}      # epsilon is 1 here: it always explores
    ag.q[0, EDGE] = -2.5                          # now both others are close -> it tries both
    assert {ag.act(0) for _ in range(300)} == {EDGE, CLOUD}
    ag.epsilon = 0.0
    assert ag.act(0) == GATEWAY


def test_equal_values_go_to_the_gateway():
    ag = QAgent(2, 3, prefer=GATEWAY)
    assert ag.greedy(0) == GATEWAY
    ag.q[1, EDGE] = 1.0
    assert ag.greedy(1) == EDGE


def test_delayed_update_order_independent():
    """Box 1's answer arriving before box 0's must not lose box 0's update."""
    ag = QAgent(SPACE.n_states, 3, alpha=1.0, gamma=0.0)
    pol = QPolicy(ag, SPACE, learn=True, explore=False)
    s0 = pol.decide(0, obs())
    pol.decide(1, obs())
    pol.decide(2, obs())
    pol.outcome(1, s0, True, 0.02, -1.0)
    pol.outcome(0, s0, False, 0.40, -35.0)
    assert pol.updates == 2


def test_rules_basic_behaviour():
    assert ThresholdRule(75).decide(0, obs(rtt_ms=10)) == CLOUD
    assert ThresholdRule(75).decide(0, obs(rtt_ms=300)) == GATEWAY
    assert DeadlineRule(margin_s=0.05).decide(0, obs(rtt_ms=10, budget_s=0.45)) == CLOUD
    assert AdaptiveRule(margin_s=0.05).decide(0, obs(cloud_ms=900, budget_s=0.45)) == GATEWAY
    # cloud too slow AND the gateway choked -> only the ESP32 still fits
    assert AdaptiveRule(margin_s=0.05).decide(0, obs(cloud_ms=900, gw_ms=700, edge_ms=40, budget_s=0.45)) == EDGE
    # the fixed-estimate rule cannot see the choked gateway
    assert DeadlineRule(margin_s=0.05).decide(0, obs(rtt_ms=900, gw_ms=700, budget_s=0.45)) == GATEWAY
    # settings saved before the ESP32 was added still load
    assert DeadlineRule(margin_s=0.1, edge_s=0.02).gw_s == 0.02


def test_simulator_good_vs_bad_network():
    sim = lambda seed: ConveyorSim(SimParams(), REWARD, np.random.default_rng(seed))
    from common.policies import AlwaysCloud
    good = summarize(sim(1).run(AlwaysCloud(), [(0, GOOD)], 300))
    bad = summarize(sim(1).run(AlwaysCloud(), [(0, BAD)], 300))
    assert good["miss_rate"] < 0.02 < 0.5 < bad["miss_rate"]


def test_simulator_esp32_saves_boxes_when_gateway_and_network_fail():
    from common.policies import AlwaysEdge, AlwaysGateway
    from sim.scenarios import GW_OVERLOAD
    sim = lambda seed: ConveyorSim(SimParams(), REWARD, np.random.default_rng(seed))
    gw = summarize(sim(2).run(AlwaysGateway(), [(0, GW_OVERLOAD)], 300))
    esp = summarize(sim(2).run(AlwaysEdge(), [(0, GW_OVERLOAD)], 300))
    assert gw["miss_rate"] > 0.5 and esp["miss_rate"] < 0.05
    assert esp["reward_mean"] > gw["reward_mean"]          # a weak answer in time beats no answer
    assert esp["edge_share"] == 1.0 and gw["gateway_share"] == 1.0


def test_shipped_q_table_matches_state_space():
    from common.config import project_path
    ag = QAgent.load(project_path(CFG["agent"]["q_table"]))
    assert ag.q.shape == (SPACE.n_states, 3) and ag.prefer == GATEWAY
