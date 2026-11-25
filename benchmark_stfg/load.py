"""
Load the public "Synthetic Task Flow Graphs" dataset (18 Excel files).

  Kouloumpris, Stavrinides, Michael, Theocharides, "Datasets of synthetic task flow graphs for
  evaluating a latency/energy optimization task allocation framework", Zenodo, 2024,
  doi:10.5281/zenodo.10654551  (paper: Future Generation Computer Systems 155 (2024) 354-366)

Put SyntheticTaskFlowGraphs.zip (or the extracted .xlsx files) into  data/stfg/
Column names are matched loosely (case / spaces / units ignored), so small naming
differences in the files do not matter.
"""
from __future__ import annotations

import glob
import os
import re
import zipfile

import pandas as pd

from common.config import project_path

COLS = {
    "taskid": "id", "parenttasks": "parents", "parents": "parents",
    "latencyedge1ms": "L_edge", "latencyedgems": "L_edge", "latencyhubms": "L_hub", "latencycloudms": "L_cloud",
    "poweredge1w": "P_edge", "poweredgew": "P_edge", "powerhubw": "P_hub", "powercloudw": "P_cloud",
    "memorymbit": "mem", "storagembit": "storage", "outputdatambit": "out_mbit", "fixedallocation": "fixed",
}


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def _parents(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    s = str(v).strip()
    if s in ("", "-", "None", "none", "nan", "[]"):
        return []
    return [int(float(x)) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]


def _fixed(v):
    """returns None (free) or device index 0=edge 1=hub 2=cloud"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ("", "0", "no", "none", "false", "-", "nan", "0.0"):
        return None
    if "edge" in s or s in ("1", "1.0", "e"):
        return 0
    if "hub" in s or s in ("2", "2.0", "h"):
        return 1
    if "cloud" in s or s in ("3", "3.0", "c"):
        return 2
    return 0


def read_graph(df: pd.DataFrame, name: str) -> dict:
    ren = {}
    for c in df.columns:
        k = _norm(c)
        if k in COLS:
            ren[c] = COLS[k]
        elif k.startswith("fixedallocation"):          # real files: "FixedAllocation (0=no,1=e,2=h)"
            ren[c] = "fixed"
    df = df.rename(columns=ren)
    need = ["id", "parents", "L_edge", "L_hub", "L_cloud"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: missing columns {miss}; found {list(df.columns)}")
    tasks = []
    for _, r in df.iterrows():
        tasks.append({
            "id": int(r["id"]), "parents": _parents(r["parents"]),
            "L": [float(r["L_edge"]), float(r["L_hub"]), float(r["L_cloud"])],          # ms
            "P_edge": float(r["P_edge"]) if "P_edge" in df.columns and not pd.isna(r["P_edge"]) else 4.0,
            "out_mbit": float(r["out_mbit"]) if "out_mbit" in df.columns and not pd.isna(r["out_mbit"]) else 1.0,
            "fixed": _fixed(r["fixed"]) if "fixed" in df.columns else None,
        })
    m = re.search(r"([PSM])\D*?(\d+)", os.path.basename(name), re.I)
    kind = m.group(1).upper() if m else "?"
    return {"name": os.path.splitext(os.path.basename(name))[0], "kind": kind, "n": len(tasks), "tasks": tasks}


def load_all(folder: str | None = None) -> list[dict]:
    folder = folder or project_path("data/stfg")
    for z in glob.glob(os.path.join(folder, "*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(os.path.join(folder, "extracted"))
    files = sorted(glob.glob(os.path.join(folder, "**", "*.xlsx"), recursive=True)
                   + glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
    graphs = []
    for f in files:
        try:
            df = pd.read_excel(f) if f.endswith(".xlsx") else pd.read_csv(f)
            graphs.append(read_graph(df, f))
        except Exception as e:
            print(f"[stfg] skipped {os.path.basename(f)}: {e}")
    return graphs
