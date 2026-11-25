"""
Second opinion ("cascade"): how sure is the gateway's model, and what is a cloud answer worth?

Plain words
-----------
The gateway's small model looks at every box first (about 2 ms). Its score says how sure it is:
a score far from the cut-off = sure, a score close to the cut-off = unsure. Real photos show that
the cloud's big model mostly helps on the unsure boxes, so those are the ones worth a cloud call.

  margin      = | logit(score) - logit(cut-off) |       (0 = right on the cut-off)
  conf level  : 0 unknown (the gateway's model did not run), 1 sure, 2 fairly sure, 3 unsure, 4 very unsure
  edges       : margins that separate the levels; taken from the VALIDATION photos
                (least-sure 10 % = very unsure, next 15 % = unsure, next 25 % = fairly sure, rest = sure)

PhotoTable = the score of every validation / test photo at each place. The simulator replays these
photos, so its verdicts are the real models' verdicts on real photos, not random draws.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

CONF_NAMES = ["unknown", "sure", "fairly sure", "unsure", "very unsure"]
N_CONF = len(CONF_NAMES)
QUANTILES = (0.10, 0.25, 0.50)
ROLES = ("edge", "gateway", "cloud")


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def margin(p, thr: float):
    """How far a score is from the cut-off (logit units). Works on one number or an array."""
    return np.abs(logit(p) - float(logit(thr)))


def edges_from(margins) -> list:
    return [float(x) for x in np.quantile(np.asarray(margins, dtype=np.float64), QUANTILES)]


def conf_level(m: float, edges) -> int:
    """1 sure ... 4 very unsure (0 is reserved for 'the gateway's model did not run')."""
    if m < edges[0]:
        return 4
    if m < edges[1]:
        return 3
    if m < edges[2]:
        return 2
    return 1


def conf_levels(margins, edges) -> np.ndarray:
    m = np.asarray(margins, dtype=np.float64)
    return np.where(m < edges[0], 4, np.where(m < edges[1], 3, np.where(m < edges[2], 2, 1))).astype(np.int64)


class PhotoSplit:
    """Scores of one set of photos (validation or test)."""

    def __init__(self, y, scores: dict, thr: dict, edges=None):
        self.y = np.asarray(y, dtype=np.int64)
        self.p = {k: np.asarray(v, dtype=np.float64) for k, v in scores.items()}
        self.thr = {k: float(thr[k]) for k in self.p}
        self.verdict = {k: (self.p[k] >= self.thr[k]).astype(np.int64) for k in self.p}      # 1 = "damaged"
        self.margin = margin(self.p["gateway"], self.thr["gateway"])
        self.edges = list(edges) if edges is not None else edges_from(self.margin)
        self.conf = conf_levels(self.margin, self.edges)
        self.idx = {1: np.flatnonzero(self.y == 1), 0: np.flatnonzero(self.y == 0)}

    def __len__(self):
        return len(self.y)

    def with_thresholds(self, thr: dict, edges=None) -> "PhotoSplit":
        t = dict(self.thr)
        t.update({k: float(v) for k, v in thr.items() if k in self.p})
        return PhotoSplit(self.y, self.p, t, edges)

    def pick(self, rng, damaged: bool) -> int:
        ids = self.idx[1 if damaged else 0]
        return int(ids[int(rng.integers(len(ids)))])

    def stats(self) -> dict:
        """Per confidence level: how many damaged / good photos, and how many each model gets wrong."""
        lv = []
        for c in range(N_CONF):
            mask = np.ones(len(self.y), bool) if c == 0 else self.conf == c      # level 0 = all photos
            d, g = mask & (self.y == 1), mask & (self.y == 0)
            row = {"n_damaged": int(d.sum()), "n_good": int(g.sum())}
            for role in self.p:
                row[role + "_missed"] = int(((self.verdict[role] == 0) & d).sum())
                row[role + "_false_reject"] = int(((self.verdict[role] == 1) & g).sum())
            lv.append(row)
        return {"edges": self.edges, "quantiles": list(QUANTILES), "levels": lv, "names": CONF_NAMES,
                "n_photos": int(len(self.y)), "thresholds": dict(self.thr)}


def level_costs(stats: dict | None, p_def: float, c_esc: float, c_fr: float, fallback: dict | None = None) -> dict:
    """Expected cost of a verdict per confidence level, for a line where a share p_def of boxes is damaged.

    Returns {"share": [...], "p_damaged": [...], "gateway": [...], "cloud": [...]} with one entry per level.
    Small counts are smoothed (one imaginary photo spread over right/wrong) so a level with 0 mistakes
    in a few dozen photos is not treated as perfect.
    fallback = {"gateway": cost, "cloud": cost} used for every level when there are no stats.
    """
    if not stats or not stats.get("levels"):
        fb = fallback or {"gateway": 0.0, "cloud": 0.0}
        return {"share": [1.0] + [0.25] * (N_CONF - 1), "p_damaged": [p_def] * N_CONF,
                "gateway": [fb["gateway"]] * N_CONF, "cloud": [fb["cloud"]] * N_CONF, "informative": False}
    lv = stats["levels"]
    nd_all, ng_all = max(1, lv[0]["n_damaged"]), max(1, lv[0]["n_good"])
    out = {"share": [], "p_damaged": [], "gateway": [], "cloud": [], "informative": True}
    for row in lv:
        pb_d, pb_g = row["n_damaged"] / nd_all, row["n_good"] / ng_all
        share = p_def * pb_d + (1 - p_def) * pb_g
        pd = p_def * pb_d / share if share > 0 else p_def
        out["share"].append(float(share))
        out["p_damaged"].append(float(pd))
        for role in ("gateway", "cloud"):
            base_miss = lv[0][role + "_missed"] / nd_all
            base_fr = lv[0][role + "_false_reject"] / ng_all
            miss = (row[role + "_missed"] + base_miss) / (row["n_damaged"] + 1)
            fr = (row[role + "_false_reject"] + base_fr) / (row["n_good"] + 1)
            out[role].append(float(pd * miss * c_esc + (1 - pd) * fr * c_fr))
    return out


class PhotoTable:
    def __init__(self, val: PhotoSplit, test: PhotoSplit, source: str = "real photos"):
        self.val, self.test, self.source = val, test, source

    def with_thresholds(self, thr: dict) -> "PhotoTable":
        v = self.val.with_thresholds(thr)
        return PhotoTable(v, self.test.with_thresholds(thr, v.edges), self.source)

    def save(self, path: str, meta: dict | None = None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arr = {}
        for name, s in (("val", self.val), ("test", self.test)):
            arr[name + "_y"] = s.y
            for k, v in s.p.items():
                arr[f"{name}_{k}"] = v
        for k, v in self.val.thr.items():
            arr["thr_" + k] = np.float64(v)
        arr["meta"] = json.dumps({"source": self.source, **(meta or {})})
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arr)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "PhotoTable":
        z = np.load(path, allow_pickle=False)
        roles = [r for r in ROLES if f"val_{r}" in z.files and f"test_{r}" in z.files]
        thr = {r: float(z["thr_" + r]) for r in roles}
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        val = PhotoSplit(z["val_y"], {r: z[f"val_{r}"] for r in roles}, thr)
        test = PhotoSplit(z["test_y"], {r: z[f"test_{r}"] for r in roles}, thr, val.edges)
        return PhotoTable(val, test, meta.get("source", "real photos"))


def _inv_phi(p: float) -> float:
    """Inverse of the standard normal CDF (Acklam's approximation; plenty for making test data)."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609887986040e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        return -_inv_phi(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def synthetic_table(metrics: dict, n: int = 3000, seed: int = 0) -> PhotoTable:
    """A made-up photo table that matches each model's recall and false-reject rate (used only when
    there is no models/photo_scores.npz yet, e.g. a fresh copy or the automatic tests).

    Each photo has a hidden 'how damaged it looks' number shared by the three models plus each model's
    own noise, so an unsure gateway model goes with a photo that is hard for everyone, as in reality."""
    rng = np.random.default_rng(seed)

    def split(m):
        y = (rng.random(m) < 0.5).astype(np.int64)
        shared = rng.standard_normal(m)
        scores, thr = {}, {}
        for role in ROLES:
            mt = metrics[role]
            t = _inv_phi(1 - min(max(mt["false_reject_rate"], 1e-4), 1 - 1e-4))          # good photos ~ N(0, 1)
            mu = t - _inv_phi(1 - min(max(mt["recall_damaged"], 1e-4), 1 - 1e-4))        # damaged ~ N(mu, 1)
            z = 0.7 * shared + math.sqrt(1 - 0.49) * rng.standard_normal(m) + mu * y
            scores[role] = 1.0 / (1.0 + np.exp(-2.5 * (z - t)))
            thr[role] = 0.5
        return y, scores, thr

    yv, sv, thr = split(n)
    yt, st, _ = split(n)
    val = PhotoSplit(yv, sv, thr)
    return PhotoTable(val, PhotoSplit(yt, st, thr, val.edges), "made-up scores (no photo table yet)")


def load_table(path: str | None, metrics: dict | None = None, seed: int = 0) -> PhotoTable:
    if path and os.path.exists(path):
        return PhotoTable.load(path)
    from .rl import DEFAULT_METRICS
    return synthetic_table(metrics or DEFAULT_METRICS, seed=seed)


def escalation_curve(split: PhotoSplit, p_def: float, c_esc: float, c_fr: float,
                     shares=(0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100), seed: int = 0) -> list:
    """On real photos: send only the least-sure X % of boxes to the cloud. Accuracy and cost per box,
    next to the same X % picked at random (what a rule that ignores confidence would get)."""
    y, n = split.y, len(split.y)
    vg, vc = split.verdict["gateway"], split.verdict["cloud"]
    order = np.argsort(split.margin)
    nd, ng = max(1, int((y == 1).sum())), max(1, int((y == 0).sum()))

    def score(v):
        miss = float(((v == 0) & (y == 1)).sum() / nd)
        fr = float(((v == 1) & (y == 0)).sum() / ng)
        return float((v == y).mean()), p_def * miss * c_esc + (1 - p_def) * fr * c_fr

    rng = np.random.default_rng(seed)
    out = []
    for s in shares:
        k = int(round(n * s / 100.0))
        v = vg.copy()
        v[order[:k]] = vc[order[:k]]
        acc, cost = score(v)
        ra, rc = [], []
        for _ in range(200):
            idx = rng.permutation(n)[:k]
            w = vg.copy()
            w[idx] = vc[idx]
            a_, c_ = score(w)
            ra.append(a_)
            rc.append(c_)
        out.append({"share_pct": s, "accuracy": acc, "cost": cost,
                    "accuracy_random": float(np.mean(ra)), "cost_random": float(np.mean(rc))})
    return out
