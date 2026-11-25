"""Each fast algorithm in common/algos.py must give the SAME answer as the naive
version it replaced.   Run:  python -m pytest -q tests"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.algos import (CompletionHeap, RunningMean, SlidingWindowCounter, bin_index,  # noqa: E402
                          cost_optimal_threshold, first_index_staying_below, first_window_above,
                          percentile_sorted, rolling_mean, rolling_sum)

RNG = np.random.default_rng(2026)


# ---------------------------------------------------------------- naive references
def naive_bin(x, edges):
    i = 0
    while i < len(edges) and x >= edges[i]:
        i += 1
    return i


def naive_stay_below(x, lim):
    return next((i for i in range(len(x)) if np.all(x[i:] < lim)), None)


def naive_window_above(x, w, lim):
    return next((i for i in range(len(x) - w) if x[i:i + w].mean() > lim), None)


def naive_threshold(ys, probs, p_def=0.30, c_esc=100.0, c_fr=10.0):
    best = (1e9, 0.5)
    for t in np.linspace(0.05, 0.95, 91):
        ps = probs > t
        fnr = ((ys == 1) & ~ps).sum() / max(1, (ys == 1).sum())
        fpr = ((ys == 0) & ps).sum() / max(1, (ys == 0).sum())
        c = p_def * fnr * c_esc + (1 - p_def) * fpr * c_fr
        if c < best[0] - 1e-12:
            best = (c, float(t))
    return best[1]


# ---------------------------------------------------------------- tests
def test_bin_index_matches_linear_scan():
    for edges in ([-50, 0, 50, 100, 150, 250, 400], [150, 400], [0.5, 0.85], [1.0]):
        for x in list(RNG.uniform(-200, 800, 3000)) + edges + [e - 1e-9 for e in edges]:
            assert bin_index(x, edges) == naive_bin(x, edges)


def test_rolling_mean_matches_convolution():
    for n, w in [(900, 25), (3000, 100), (40, 15), (15, 15)]:
        x = RNG.random(n)
        ref = np.convolve(x, np.ones(w) / w, mode="valid")
        got = rolling_mean(x, w)
        assert got.shape == ref.shape
        assert np.allclose(got, ref, atol=1e-12)
    # shorter than the window: no complete window -> empty
    # (np.convolve quietly swaps its arguments here and returns len w-n+1 values)
    assert rolling_mean(RNG.random(10), 15).shape == (0,)


def test_rolling_sum_integers_exact():
    x = RNG.integers(0, 31, 900)
    ref = np.array([x[i:i + 25].sum() for i in range(len(x) - 24)])
    assert np.array_equal(rolling_sum(x, 25), ref)


def test_stays_below_matches_quadratic_search():
    for _ in range(300):
        x = RNG.random(int(RNG.integers(1, 120))) * 0.1
        assert first_index_staying_below(x, 0.05) == naive_stay_below(x, 0.05)
    assert first_index_staying_below(np.array([]), 0.05) is None
    assert first_index_staying_below(np.array([0.01, 0.02]), 0.05) == 0
    assert first_index_staying_below(np.array([0.01, 0.09]), 0.05) is None


def test_window_above_matches_slicing():
    for _ in range(300):
        counts = RNG.integers(0, 31, int(RNG.integers(30, 200)))     # boxes sent, out of 30 seeds
        w, seeds = 25, 30
        ref = naive_window_above(counts / seeds, w, 0.8)
        got = first_window_above(counts, w, 0.8 * seeds, n_starts=len(counts) - w)
        assert got == ref


def test_completion_heap_matches_list_scan():
    h, lst = CompletionHeap(), []
    t = 0.0
    for _ in range(2000):
        t += RNG.uniform(0.2, 1.0)
        for _ in range(int(RNG.integers(0, 3))):
            d = t + RNG.exponential(1.0) if RNG.random() > 0.01 else float("inf")
            h.push(d)
            lst.append(d)
        ref = sum(1 for d in lst if d > t)
        lst = [d for d in lst if d > t]
        assert h.count_after(t) == ref


def test_cost_threshold_matches_grid_loop():
    for _ in range(100):
        n = int(RNG.integers(20, 600))
        ys = (RNG.random(n) < 0.3).astype(int)
        probs = np.clip(RNG.normal(0.3 + 0.4 * ys, 0.2), 0, 1)
        assert cost_optimal_threshold(ys, probs) == naive_threshold(ys, probs)
    # the exact search over every score is never worse than the 91-point grid
    ys = (RNG.random(500) < 0.3).astype(int)
    probs = np.clip(RNG.normal(0.3 + 0.4 * ys, 0.2), 0, 1)

    def cost(t):
        fnr = ((ys == 1) & ~(probs > t)).sum() / (ys == 1).sum()
        fpr = ((ys == 0) & (probs > t)).sum() / (ys == 0).sum()
        return 0.3 * fnr * 100 + 0.7 * fpr * 10
    assert cost(cost_optimal_threshold(ys, probs, grid="all")) <= cost(cost_optimal_threshold(ys, probs)) + 1e-12


def test_streaming_stats():
    rm = RunningMean()
    xs = RNG.random(1000)
    for x in xs:
        rm.add(x)
    assert abs(rm.mean - xs.mean()) < 1e-12
    assert RunningMean().mean is None

    now = [1000.0]
    c = SlidingWindowCounter(60.0, clock=lambda: now[0])
    ts = np.sort(RNG.uniform(0, 1000, 500))
    for t in ts:
        c.add(float(t))
    assert c.count(1000.0) == int((1000.0 - ts <= 60.0).sum())

    v = np.sort(RNG.random(101))
    for q in (0, 5, 50, 95, 100):
        assert abs(percentile_sorted(v, q) - np.percentile(v, q)) < 1e-12
