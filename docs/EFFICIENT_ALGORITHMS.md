# Simple, standard algorithms — same answers, less work

**Idea.** Computing a spectrum with the direct DFT costs O(N²); the FFT gives *exactly the same numbers* in
O(N log N). Several places in this project had the same kind of "textbook-obvious but slow" method. Each one
was replaced by the standard faster method that returns the **same result**. All of them live in
[`common/algos.py`](../common/algos.py), and [`tests/test_algos.py`](../tests/test_algos.py) checks every one
against the old naive version on random inputs.

| # | Where it runs | Before (naive) | Now (standard method) | Cost before → after |
|---|---|---|---|---|
| 1 | State of every box (`common/rl.py`) | walk the bin edges one by one | **binary search** (`bisect_right`) | O(B) → O(log B) |
| 2 | State index (`common/rl.py`) | `np.ravel_multi_index` call | **Horner's rule** on the mixed-radix digits (`idx = idx·d + b`) | same O(k), no NumPy call overhead per box |
| 3 | Greedy action with 2 actions (`common/rl.py`) | `np.argmax` on a 2-element row | one comparison (ties → EDGE, as before) | same result, no NumPy call per decision |
| 4 | Queue length at each decision (`sim/env.py`) | scan the list of finish times and rebuild it, every box | **min-heap**: pop the jobs that have finished | O(q) per box → O(log q) amortised |
| 5 | TCP retransmission delay (`sim/env.py`) | Python loop over each lost packet | draw all *k* uniforms at once, count the fast ones: `n_fast·RTT + n_slow·RTO` | k interpreted steps → 1 vectorised call (same random numbers, same order) |
| 6 | Forget old boxes in the agent (`common/policies.py`) | `sorted(dict)[:1000]` | dicts keep insertion order and box ids only grow → take the first 1000 keys | O(n log n) → O(1) per key |
| 7 | Rolling averages for figures (`sim/experiments.py`, `tools/report_real.py`, dashboard) | `np.convolve` with a box kernel / inner loop per point | **prefix sums**: `S[i+w] − S[i]` (dashboard: running window sum) | O(n·w) → O(n) |
| 8 | "Boxes to recover" after a fault (E2) | for every i, test whether *all later* values are below 5 % | **one backward scan**: the answer is right after the last value ≥ 5 % | O(n²) → O(n) |
| 9 | "Boxes until the cloud is trusted again" (E2) | slice + mean for every window | prefix sums on integer counts (exact, no float ties) | O(n·w) → O(n) |
| 10 | Cost-optimal model threshold (`vision/train.py`) | one full pass over all images per candidate threshold (91 passes) | **sort scores once + binary search** per candidate (`grid="all"` gives the exact optimum over every possible threshold) | O(T·n) → O(n log n + T log n) |
| 11 | Recall per defect type / severity (`vision/train.py`) | one pass over the test set per group | **one pass**, grouping as it goes | O(G·n) → O(n) |
| 12 | Dashboard totals (`server/store.py`) | re-scan up to 5,000 stored events on every refresh (0.8 s) | **running counters** updated once per event | O(n) per refresh → O(1) |
| 13 | "Boxes in the last minute" | scan every event's time | **sliding-window deque**: drop old timestamps from the left | O(n) → amortised O(1) |
| 14 | Smoothing of ping / cloud time / CPU | (already) EWMA | kept: **EWMA**, O(1) time and memory, now one shared function | — |
| 15 | Model sync to the gateway (`tools/sync_models.py`) | (new) | compare **sha256** hashes, download only changed files; server caches each hash by (mtime, size) | hashing once per file change, not per request |
| 16 | S3 backups (`server/cloud_sync.py`) | (new) | upload only files whose MD5 changed since the last upload | no repeated uploads |
| 17 | CloudWatch metrics | (new) | per-minute values from the **difference of running counters** | O(1) per minute, no re-scan |

## Proof that nothing changed

The simulator and the agent were run before and after the changes with the same seeds
(1,500 pre-training episodes, then the full E1–E5 experiment suite):

| Check | Result |
|---|---|
| Pre-trained Q-table (864 values) and visit counts | **bit-identical** (`max |diff| = 0.0`) |
| `E1_steady.csv`, `E2_changes.csv`, `E5_mismatch.csv`, `E3_pretraining_value.json`, `pretrain_curve.csv` | **byte-identical files** |
| Pre-training time (1,500 episodes = 225,000 boxes) | 11.8 s → **8.5 s** (−28 %) |
| Full experiment suite E1–E5 | 101.7 s → **86.9 s** (−15 %) |

So every number in `results/SIM_RESULTS.md` and in the paper is still produced by the current code.
(Timing measured once on the development laptop, Python 3.14, NumPy 2.5; the remaining time is dominated by
the event simulation itself, which is already O(1) per box.)

## Why identical results matter

A faster method that changes the numbers would silently invalidate the experiments. Each replacement above was
chosen so that the output is the same by construction (same comparisons, same random numbers in the same
order), and the equivalence is tested, not assumed:

```
python -m pytest -q tests        # all tests: algorithms, RL core, server, cloud sync, USB webcam
```

## About the DFT in the old Phase-III code

The October patient-monitoring prototype (`Claude outputs/analytics.py`, `PHASE3_FINAL/pi_gateway.py`) used a
pure-Python direct DFT. That was deliberate there — it served as an artificial "heavy job" so that edge and
cloud had different costs. The conveyor project replaced that synthetic workload with **real** work (two CNNs
of different size), so it contains no DFT. If that old code is ever reused, `numpy.fft.rfft` gives the same
spectrum in O(N log N); the task sizes would then have to be re-calibrated, because the job becomes much lighter.
