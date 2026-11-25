"""
Make 18 STAND-IN task graphs with the same columns as the public dataset, ONLY to test the
benchmark code when the real files are not present.  NEVER report results from these as the
dataset's results.

    python -m benchmark_stfg.make_standin        -> data/stfg_standin/*.xlsx
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from common.config import project_path


def make(kind: str, n: int, rng) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        if i == 1:
            parents = []
        elif kind == "S":                          # serial chain
            parents = [i - 1]
        elif kind == "P":                          # wide: mostly depend on the first few tasks
            parents = [int(rng.integers(1, min(i, 4)))] if i > 2 else [1]
        else:                                      # mixed
            k = int(rng.integers(1, 4))
            lo = max(1, i - 10)
            parents = sorted(set(int(x) for x in rng.integers(lo, i, size=k)))
        le = float(rng.uniform(20, 800))
        rows.append({
            "TaskID": i, "ParentTasks": ",".join(map(str, parents)) if parents else "-",
            "LatencyEdge1 (ms)": round(le, 2), "LatencyHub (ms)": round(le / rng.uniform(3, 8), 2),
            "LatencyCloud (ms)": round(le / rng.uniform(10, 30), 2), "PowerEdge1 (W)": round(rng.uniform(3, 6), 2),
            "PowerCloud (W)": round(rng.uniform(50, 150), 1), "Memory (Mbit)": round(rng.uniform(10, 500), 1),
            "Storage (Mbit)": round(rng.uniform(10, 500), 1), "OutputData (Mbit)": round(rng.uniform(0.1, 10), 2),
            "FixedAllocation": "Edge" if rng.random() < 0.05 else ("Hub" if rng.random() < 0.05 else "-"),
        })
    return pd.DataFrame(rows)


def main():
    out = project_path("data/stfg_standin")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(5)
    for kind in "PSM":
        for n in (10, 100, 1000):
            for v in (1, 2):
                make(kind, n, rng).to_excel(os.path.join(out, f"STANDIN_{kind}{n}_v{v}.xlsx"), index=False)
    print(f"18 stand-in graphs -> {out}")


if __name__ == "__main__":
    main()
