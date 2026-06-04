
# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------
"""Buchin 9-bin grid sweep on Arkansas — mirror of the existing 5-bin runs.

Re-runs Buchin with the original Dutch 9-multiplier grid
  {0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5}
instead of the 5-bin grid {0.5, 0.7, 1.0, 1.3, 1.5} used in the existing
paper experiments. Same seeds, same case-gen, so the only difference is
the size grid.

Generates 1200 calls total:
  Rect:  Fig 8 + Fig 9            × 20 trials × 15 p =  600 calls
  Disk:  Disk-Large + Disk-Small  × 20 trials × 15 p =  600 calls

Output:
  arkansas_io/fig8_buchin_9bin.csv, fig9_buchin_9bin.csv     (rect)
  arkansas_io/arkansas_disk_large_buchin_9bin.pkl            (disk r=0.6)
  arkansas_io/arkansas_disk_small_buchin_9bin.pkl            (disk r=0.395)

Joblib n_jobs=4, caffeinate attached externally.
"""
from __future__ import annotations

import math
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

# (DYLD_LIBRARY_PATH should be set by the user; see README)
import geopandas as gpd
import numpy as np
from joblib import Parallel, delayed
from shapely.geometry import Point, Polygon

import run_buchin_comparison as rbc       # for rect — uses run_buchin_comparison's seeds
import arkansas_disk_stress as ds         # for disk — uses arkansas_disk_stress's helpers

IO   = IO
JAVA_CWD = OUTPUTS / "Maarten-implementation"
JAVA_CP  = "build"

# 9-bin grid — Buchin's original Dutch experiment grid
GRID_9 = "0.5,0.7,0.85,0.95,1.0,1.05,1.15,1.3,1.5"

P_GRID = np.round(np.arange(0.20, 0.95, 0.05), 4)
N_TRIALS = 20
Q = 0.20

# ---- Rect targets (Fig 8, Fig 9) ----
RECT_FIGS = [
    {"fig_name": "fig8",
     "target":   Polygon([(-93.5, 34.0), (-93.5, 35.5),
                          (-91.5, 35.5), (-91.5, 34.0)]),
     "WH":       (2.0, 1.5),
     "seed_off": 0},
    {"fig_name": "fig9",
     "target":   Polygon([(-92.85, 34.40), (-92.85, 35.10),
                          (-92.15, 35.10), (-92.15, 34.40)]),
     "WH":       (0.7, 0.7),
     "seed_off": 10_000_000_000},
]

# ---- Disk targets (Disk-Large, Disk-Small) ----
DISK_FIGS = [
    {"name":      "large",
     "planted_r": 0.6,
     "out_pkl":   IO / "arkansas_disk_large_buchin_9bin.pkl",
     "seed_off":  80_000_000_000},
    {"name":      "small",
     "planted_r": 0.395,
     "out_pkl":   IO / "arkansas_disk_small_buchin_9bin.pkl",
     "seed_off":  60_000_000_000},
]

STRESS_9BIN_DIR = IO / "stress_9bin"


# ---------------------------------------------------------------------------
# Rect 9-bin job
# ---------------------------------------------------------------------------

def _rect_job(fig_meta: dict, trial: int, p: float) -> dict:
    """Generate cases at same seed as Phase 3 run_buchin_comparison; call Java with 9-bin grid."""
    gdf = gpd.read_file(rbc.SHP).to_crs("EPSG:4326").reset_index(drop=True)
    seed = fig_meta["seed_off"] + 1_000_000 * trial + int(round(p * 100000))
    counts = rbc.generate_cases(gdf, fig_meta["target"], p, seed=seed)

    cases_path = STRESS_9BIN_DIR / f"rect_{fig_meta['fig_name']}_t{trial:03d}_p{int(p*100):03d}.cas"
    out_csv    = STRESS_9BIN_DIR / f"rect_{fig_meta['fig_name']}_t{trial:03d}_p{int(p*100):03d}.csv"
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    rbc.write_cases(counts, cases_path)
    if out_csv.exists(): out_csv.unlink()

    W, H = fig_meta["WH"]
    cmd = [
        "java", "-cp", JAVA_CP, "app.ExperimentRunner",
        "--polys",        str(rbc.POLYS_FILE),
        "--names",        str(rbc.NAMES_FILE),
        "--pop",          str(rbc.POP_FILE),
        "--cases",        str(cases_path),
        "--window",       "rect",
        "--planted-size", f"{W} {H}",
        "--size-grid",    GRID_9,
        "--out",          str(out_csv),
        "--trial-id",     str(trial),
    ]
    t0 = time.time()
    subprocess.run(cmd, cwd=JAVA_CWD, check=True, stdout=subprocess.DEVNULL)
    wall = time.time() - t0

    # Parse the one-row CSV
    with open(out_csv) as f:
        lines = f.read().splitlines()
    parts = lines[1].split(",")
    size_mul = float(parts[3])
    best_size = parts[4]   # "W=...;H=..."
    Wd, Hd = (float(x.split("=")[1]) for x in best_size.split(";"))
    cx, cy = float(parts[5]), float(parts[6])
    score  = float(parts[7])
    java_wall = float(parts[8])
    return {
        "fig":      fig_meta["fig_name"], "trial": trial, "p": float(p),
        "mode":     "rect",
        "best_cx":  cx, "best_cy": cy,
        "best_W":   Wd, "best_H":  Hd,
        "size_mul": size_mul, "score": score,
        "java_wall": java_wall, "wall_sec": wall,
    }


# ---------------------------------------------------------------------------
# Disk 9-bin job (mirrors arkansas_disk_stress's _part1_job / arkansas_disk_large_sweep._job)
# ---------------------------------------------------------------------------

def _disk_job(disk_meta: dict, trial: int, p: float) -> dict:
    """Same seeds as the existing 5-bin disk sweeps, just with 9-bin Buchin grid."""
    with open(ds.POINT_DICT, "rb") as f:
        point_dict = pickle.load(f)
    target = ds._planted_disk(disk_meta["planted_r"])
    seed = disk_meta["seed_off"] + 1_000_000 * trial + int(round(p * 100000))
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed + 1)
    pts = ds._sample_per_region(point_dict, rng_np)
    pts_with_case = ds._coinflip(pts, target, p, rng_py)
    names = ds._county_names_in_order()
    counts = ds._per_region_count(pts_with_case, names=names)

    tag = f"disk_{disk_meta['name']}_t{trial:03d}_p{int(p*100):03d}"
    cases_path = STRESS_9BIN_DIR / f"{tag}.cas"
    out_csv    = STRESS_9BIN_DIR / f"{tag}.csv"
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    ds._write_cases_file(counts, cases_path)
    if out_csv.exists(): out_csv.unlink()

    cmd = [
        "java", "-cp", JAVA_CP, "app.ExperimentRunner",
        "--polys",        str(ds.POLYS_FILE),
        "--names",        str(ds.NAMES_FILE),
        "--pop",          str(ds.POP_FILE),
        "--cases",        str(cases_path),
        "--window",       "disk",
        "--planted-size", f"{disk_meta['planted_r']:.6f}",
        "--size-grid",    GRID_9,
        "--out",          str(out_csv),
        "--trial-id",     str(trial),
    ]
    t0 = time.time()
    subprocess.run(cmd, cwd=JAVA_CWD, check=True, stdout=subprocess.DEVNULL)
    wall = time.time() - t0
    with open(out_csv) as f:
        lines = f.read().splitlines()
    parts = lines[1].split(",")
    size_mul = float(parts[3])
    r_val = float(parts[4].replace("r=", ""))
    cx, cy = float(parts[5]), float(parts[6])
    score  = float(parts[7])
    java_wall = float(parts[8])
    return {
        "method": "buchin", "fig": disk_meta["name"], "trial": trial,
        "p": float(p), "planted_r": disk_meta["planted_r"],
        "disk_cx": cx, "disk_cy": cy, "disk_r": r_val,
        "size_mul": size_mul, "score": score,
        "java_wall": java_wall, "wall_sec": wall,
    }


# ---------------------------------------------------------------------------
# Pool JD post-step for disk records
# ---------------------------------------------------------------------------

def _pool_jd_for_disk_records(records, planted_r):
    all_pts = ds._load_ref_points()
    target = ds._planted_disk(planted_r)
    in_t = np.array([target.contains(Point(x, y)) for x, y in all_pts])
    for r in records:
        r["pool_jd"] = ds._pool_jd_disk(r["disk_cx"], r["disk_cy"], r["disk_r"],
                                         in_t, all_pts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    STRESS_9BIN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ----- Rect (Fig 8 + Fig 9) -----
    print("\n=== Rect 9-bin: Fig 8 + Fig 9, 20 trials × 15 p × 2 figs = 600 calls ===")
    rect_jobs = [(fm, t, float(p))
                 for fm in RECT_FIGS
                 for t in range(1, N_TRIALS + 1)
                 for p in P_GRID]
    print(f"  {len(rect_jobs)} jobs dispatched")
    t0 = time.time()
    rect_records = Parallel(n_jobs=4, verbose=10)(
        delayed(_rect_job)(fm, t, p) for (fm, t, p) in rect_jobs)
    print(f"  rect wall: {time.time()-t0:.1f}s")

    # Split rect records by fig and write to .csv (matching fig{8,9}_with_jaccard schema where possible)
    import pandas as pd
    rdf = pd.DataFrame(rect_records)
    # Add point_jaccard column — compute from discovered rect bounds via pool JD.
    # Use the same 500-pts/region pool the with_jaccard files use.
    all_pts = ds._load_ref_points()
    for fm in RECT_FIGS:
        sub = rdf[rdf["fig"] == fm["fig_name"]].copy()
        in_t = np.array([fm["target"].contains(Point(x, y)) for x, y in all_pts])
        pj = []
        for _, row in sub.iterrows():
            W, H = row["best_W"], row["best_H"]
            cx, cy = row["best_cx"], row["best_cy"]
            xs = all_pts[:, 0]; ys = all_pts[:, 1]
            in_w = ((xs >= cx - W/2) & (xs <= cx + W/2) &
                    (ys >= cy - H/2) & (ys <= cy + H/2))
            u = (in_t | in_w).sum(); n = (in_t & in_w).sum()
            pj.append(1.0 if u == 0 else float(max(0.0, min(1.0, 1.0 - n/u))))
        sub["point_jaccard"] = pj
        out = IO / f"{fm['fig_name']}_buchin_9bin.csv"
        # mirror with_jaccard schema columns we need
        sub[["trial", "mode", "p", "best_cx", "best_cy", "best_W", "best_H",
             "size_mul", "score", "wall_sec", "point_jaccard"]] \
           .rename(columns={"trial": "trial_id"}) \
           .to_csv(out, index=False)
        print(f"  wrote {out}  (n={len(sub)})")

    # ----- Disk (Disk-Large + Disk-Small) -----
    print(f"\n=== Disk 9-bin: Disk-Large + Disk-Small, 20 trials × 15 p × 2 sizes = 600 calls ===")
    disk_jobs = [(dm, t, float(p))
                 for dm in DISK_FIGS
                 for t in range(1, N_TRIALS + 1)
                 for p in P_GRID]
    print(f"  {len(disk_jobs)} jobs dispatched")
    t0 = time.time()
    disk_records = Parallel(n_jobs=4, verbose=10)(
        delayed(_disk_job)(dm, t, p) for (dm, t, p) in disk_jobs)
    print(f"  disk wall: {time.time()-t0:.1f}s")

    # Group by disk_meta, attach pool_jd, save pkl
    for dm in DISK_FIGS:
        recs = [r for r in disk_records if r["fig"] == dm["name"]]
        _pool_jd_for_disk_records(recs, dm["planted_r"])
        pkg = {"records": recs, "p_grid": P_GRID.tolist(),
               "planted_r": dm["planted_r"], "planted_center": ds.PLANTED_CENTER,
               "q": Q, "n_trials": N_TRIALS,
               "size_grid": [0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5],
               "net_size": ds.NET_SIZE,
               "grid_label": "9-bin"}
        with open(dm["out_pkl"], "wb") as f:
            pickle.dump(pkg, f)
        print(f"  wrote {dm['out_pkl']}  (n={len(recs)})")

    print(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
