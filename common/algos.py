"""
Small, standard algorithms used across the project.

The first versions of several modules used the "textbook-obvious" method
(scan every element, re-scan every window, re-count every event on every
request). Each one below is the standard faster method that gives the SAME
answer. The idea is the same as using the FFT instead of a direct DFT:
identical result, far less work.

| Where                         | Before (naive)                         | Now                                   |
|-------------------------------|----------------------------------------|---------------------------------------|
| state binning (every box)     | linear scan over bin edges   O(B)      | binary search (bisect)       O(log B) |
| rolling mean (plots, E2-E5)   | np.convolve with a box kernel O(n*w)   | prefix sums (cumsum)         O(n)     |
| "stays below x from i on"     | test every suffix            O(n^2)    | one backward scan            O(n)     |
| first window with mean > x    | slice + mean per position    O(n*w)    | prefix sums                  O(n)     |
| queue length at time t (sim)  | scan + rebuild list per box  O(q)      | min-heap, pop finished jobs  O(log q) |
| TCP retransmission delay      | Python loop per lost packet  O(k)      | one vectorised draw + count  O(1)*    |
| cost-optimal model threshold  | full pass per candidate      O(n*T)    | sort once + binary search O(n log n)  |
| dashboard summary (server)    | re-scan 5,000 events / poll  O(n)      | running counters             O(1)     |
| "boxes in the last minute"    | scan all events              O(n)      | sliding-window deque     amortised O(1)|
| average latency               | sum(list)/len(list)          O(n)      | running sum                  O(1)     |

(*) same random numbers in the same order, so simulator results are unchanged.
tests/test_algos.py checks every function against the naive version.
"""
from __future__ import annotations

import heapq
import math
import threading
import time
from bisect import bisect_right
from collections import deque

import numpy as np


# ----------------------------------------------------------------------------
# 1. Binning: which bin does x fall in?   bins: (-inf,e0) [e0,e1) ... [e_last,inf)
# ----------------------------------------------------------------------------
def bin_index(x: float, edges) -> int:
    """Number of edges <= x, i.e. the bin of x. Binary search, O(log B).

    Same result as the linear scan
        i = 0
        while i < len(edges) and x >= edges[i]: i += 1
    for sorted edges.
    """
    return bisect_right(edges, x)


# ----------------------------------------------------------------------------
# 2. Rolling (moving) mean with prefix sums
# ----------------------------------------------------------------------------
def rolling_sum(x, w: int) -> np.ndarray:
    """Sum of every window of length w ('valid' windows only, length n-w+1). O(n)."""
    x = np.asarray(x)
    if w <= 0:
        raise ValueError("window must be positive")
    exact = x.dtype.kind in "biu"                     # bool / int input -> exact integer sums
    if len(x) < w:
        return np.zeros(0, np.int64 if exact else np.float64)
    c = np.cumsum(x, dtype=np.int64 if exact else np.float64)
    c = np.concatenate((np.zeros(1, c.dtype), c))
    return c[w:] - c[:-w]


def rolling_mean(x, w: int) -> np.ndarray:
    """Moving average over windows of w samples, same as np.convolve(x, ones(w)/w, 'valid'). O(n)."""
    return rolling_sum(np.asarray(x, np.float64), w) / float(w)


# ----------------------------------------------------------------------------
# 3. Recovery-time style questions on a series
# ----------------------------------------------------------------------------
def first_index_staying_below(x, limit: float):
    """Smallest i such that x[j] < limit for EVERY j >= i (None if the last value is not below).

    Naive: next(i for i in range(n) if all(x[i:] < limit))  -> O(n^2).
    Here : find the last value that is NOT below the limit, answer is the one after it -> O(n).
    """
    x = np.asarray(x)
    if len(x) == 0:
        return None
    bad = np.flatnonzero(~(x < limit))
    if len(bad) == 0:
        return 0
    i = int(bad[-1]) + 1
    return i if i < len(x) else None


def first_window_above(x, w: int, limit: float, n_starts: int | None = None):
    """Smallest i with mean(x[i:i+w]) > limit, using prefix sums. O(n).

    n_starts limits which starts are checked (range(n_starts)), to match code that
    looped over `range(len(x) - w)`.  Compares sums (sum > limit*w), so integer
    inputs give exact answers with no floating-point ties.
    """
    s = rolling_sum(x, w)
    if n_starts is not None:
        s = s[:max(0, n_starts)]
    hit = np.flatnonzero(s > limit * w)
    return int(hit[0]) if len(hit) else None


# ----------------------------------------------------------------------------
# 4. Queue length at time t:  min-heap of completion times
# ----------------------------------------------------------------------------
class CompletionHeap:
    """Jobs in a queue, stored by completion time.

    count_after(t) = how many jobs are still unfinished at time t.
    Finished jobs are popped from the heap top, so each job is removed once:
    O(log q) per job instead of scanning the whole list for every box.
    Works even when completion times are not in order (cloud requests
    can overtake each other when some packets are lost).
    """

    __slots__ = ("_h",)

    def __init__(self):
        self._h: list[float] = []

    def push(self, t_done: float):
        heapq.heappush(self._h, t_done)

    def count_after(self, t: float) -> int:
        h = self._h
        while h and h[0] <= t:
            heapq.heappop(h)
        return len(h)

    def __len__(self):
        return len(self._h)


# ----------------------------------------------------------------------------
# 5. Cost-optimal decision threshold
# ----------------------------------------------------------------------------
def cost_optimal_threshold(ys, probs, p_def=0.30, c_esc=100.0, c_fr=10.0, grid=None):
    """Cut-off t on P(damaged) minimising  p*FNR(t)*c_esc + (1-p)*FPR(t)*c_fr.
    (an item is called damaged when prob > t)

    Sort the scores of damaged and of good items once; for any t,
        #damaged missed  = #damaged scores <= t   (binary search)
        #good rejected   = #good scores    >  t   (binary search)
    -> O(n log n) total instead of one full pass over the data per candidate.

    grid=None : default 0.05, 0.06, ... 0.95 (same 91 candidates as before)
    grid="all": every distinct score - the exact optimum over all thresholds.
    Ties keep the first (smallest) threshold, as before.
    """
    ys = np.asarray(ys).astype(int)
    probs = np.asarray(probs, np.float64)
    dam = np.sort(probs[ys == 1])
    good = np.sort(probs[ys == 0])
    n_d, n_g = max(1, len(dam)), max(1, len(good))
    if grid is None:
        cand = np.linspace(0.05, 0.95, 91)
    elif isinstance(grid, str) and grid == "all":
        cand = np.unique(probs)
    else:
        cand = np.asarray(grid, np.float64)
    missed = np.searchsorted(dam, cand, side="right")                 # damaged with prob <= t
    rejected = len(good) - np.searchsorted(good, cand, side="right")  # good with prob > t
    cost = p_def * (missed / n_d) * c_esc + (1 - p_def) * (rejected / n_g) * c_fr
    best, best_t = 1e9, 0.5
    for c, t in zip(cost, cand):        # keep the first minimum (same tie rule as the old loop)
        if c < best - 1e-12:
            best, best_t = float(c), float(t)
    return best_t


# ----------------------------------------------------------------------------
# 6. Streaming statistics (server dashboard, monitors)
# ----------------------------------------------------------------------------
def ewma(old, new, a: float):
    """Exponentially weighted moving average: O(1) memory and time per sample."""
    return new if old is None else (1.0 - a) * old + a * new


class RunningMean:
    """Mean of a stream without storing it."""

    __slots__ = ("n", "total")

    def __init__(self):
        self.n, self.total = 0, 0.0

    def add(self, x: float):
        self.n += 1
        self.total += float(x)

    @property
    def mean(self):
        return self.total / self.n if self.n else None


class SlidingWindowCounter:
    """How many events happened in the last `window_s` seconds.

    Timestamps go into a deque in arrival order; old ones are dropped from the
    left when counting. Each timestamp is added and removed once -> amortised O(1).
    """

    def __init__(self, window_s: float = 60.0, clock=time.time):
        self.window = float(window_s)
        self.clock = clock
        self._q: deque[float] = deque()
        self._lock = threading.Lock()

    def add(self, t: float | None = None):
        with self._lock:
            self._q.append(self.clock() if t is None else float(t))

    def count(self, now: float | None = None) -> int:
        now = self.clock() if now is None else now
        with self._lock:
            q = self._q
            while q and now - q[0] > self.window:
                q.popleft()
            return len(q)

    def clear(self):
        with self._lock:
            self._q.clear()


def percentile_sorted(sorted_vals, q: float) -> float:
    """Linear-interpolated percentile of an already sorted list (same as numpy's default)."""
    n = len(sorted_vals)
    if n == 0:
        return math.nan
    pos = (n - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))
