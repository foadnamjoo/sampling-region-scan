
# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------
"""Fair-basis runtime measurement: one discovery, scan-only, Grid(100).

For each dataset (Arkansas, USA):
  • Geom-50: sample 50 pts/region, generate cases @ p=0.6/q=0.2,
    then time ONLY pyscan.Grid(100) + max_subgrid + toRectangle.
    Repeat 5 times with 5 independent seeds; report mean.
  • Buchin rect (5-bin grid {0.5,0.7,1.0,1.3,1.5}): 1 run, Java-internal wall (scan-only).
  • Buchin rect (9-bin grid {0.5,0.7,0.85,0.95,1.0,1.05,1.15,1.3,1.5}): 1 run.

All on the same machine, serial, p=0.6, q=0.2.
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
import pyscan
from shapely.geometry import Point, Polygon

ARK_SHP = DATA / "arkansas/COUNTY_BOUNDARY.shp"
USA_SHP = DATA / "usa/cb_2018_us_county_within_cd116_500k.shp"
JAVA_CWD = OUTPUTS / "Maarten-implementation"
JAVA_CP  = "build"

# Static files (Buchin's input formats — already built earlier)
ARK_IO = IO
USA_IO = OUTPUTS / "usa_io"

P = 0.60
Q = 0.20
N_POINTS_PER_REGION = 50
GRID_RES = 100
N_GEOM_RUNS = 5
SEED_BASE = 300_000_000_000

GRID_5 = "0.5,0.7,1.0,1.3,1.5"
GRID_9 = "0.5,0.7,0.85,0.95,1.0,1.05,1.15,1.3,1.5"

DATASETS = [
    {"name": "Arkansas", "shp": ARK_SHP,
     "filter": None,
     "name_col": "COUNTY",
     "io": ARK_IO,
     "polys": ARK_IO / "arkansas-polys.txt",
     "names": ARK_IO / "arkansas-names.txt",
     "pop":   ARK_IO / "arkansas-pop.txt",
     "planted_target": Polygon([(-93.5, 34), (-93.5, 35.5),
                                (-91.5, 35.5), (-91.5, 34)]),
     "planted_W": 2.0, "planted_H": 1.5},
    {"name": "USA", "shp": USA_SHP,
     "filter": lambda g: g[g.geometry.centroid.apply(
         lambda c: -126 < c.x < -64 and 23 < c.y < 50)].reset_index(drop=True),
     "name_col": "GEOID",
     "io": USA_IO,
     "polys": USA_IO / "usa-polys.txt",
     "names": USA_IO / "usa-names.txt",
     "pop":   USA_IO / "usa-pop.txt",
     "planted_target": Polygon([(-100, 33), (-100, 40),
                                (-90, 40), (-90, 33)]),
     "planted_W": 10.0, "planted_H": 7.0},
]


def _sample_50pts_per_region(gdf, rng_py: random.Random):
    pts = []
    for _, row in gdf.iterrows():
        poly = row.geometry
        minx, miny, maxx, maxy = poly.bounds
        drawn = 0
        while drawn < N_POINTS_PER_REGION:
            x = rng_py.uniform(minx, maxx)
            y = rng_py.uniform(miny, maxy)
            if poly.contains(Point(x, y)):
                pts.append((x, y))
                drawn += 1
    return pts


def _build_cases(pts, target, p, rng_py):
    baseline, measured = [], []
    for x, y in pts:
        wp = pyscan.WPoint(1.0, x, y, 1.0)
        baseline.append(wp)
        rate = p if target.contains(Point(x, y)) else Q
        if rng_py.random() <= rate:
            measured.append(wp)
    return baseline, measured


def _build_buchin_cases_file(gdf, target, p, rng_py, name_col, cases_path):
    """Per-region case count for Buchin's input."""
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cases_path, "w") as f:
        for _, row in gdf.iterrows():
            poly = row.geometry
            minx, miny, maxx, maxy = poly.bounds
            drawn = 0; cnt = 0
            while drawn < N_POINTS_PER_REGION:
                x = rng_py.uniform(minx, maxx)
                y = rng_py.uniform(miny, maxy)
                if not poly.contains(Point(x, y)):
                    continue
                drawn += 1
                rate = p if target.contains(Point(x, y)) else Q
                if rng_py.random() <= rate:
                    cnt += 1
            f.write(f"{row[name_col]};{cnt};\n")


def time_geom50(ds, n_runs):
    print(f"\n[{ds['name']}] Geom-50 — {n_runs} runs, scan-only, Grid({GRID_RES})")
    gdf = gpd.read_file(ds["shp"]).to_crs("EPSG:4326").reset_index(drop=True)
    if ds["filter"]:
        gdf = ds["filter"](gdf)
    n_regions = len(gdf)
    print(f"  n_regions = {n_regions}")

    scan_times = []
    n_measured_seen = []
    for run_i in range(n_runs):
        seed = SEED_BASE + 1000 * run_i
        rng_py = random.Random(seed)
        pts = _sample_50pts_per_region(gdf, rng_py)
        baseline, measured = _build_cases(pts, ds["planted_target"], P, rng_py)
        n_measured_seen.append(len(measured))
        # ── timed block ─────────────────────────────────────
        t0 = time.time()
        grid = pyscan.Grid(GRID_RES, measured, baseline)
        sg   = pyscan.max_subgrid(grid, pyscan.KULLDORF)
        rect = grid.toRectangle(sg)
        t_scan = time.time() - t0
        # ── end timed block ─────────────────────────────────
        scan_times.append(t_scan)
        print(f"    run {run_i+1}: n_measured={len(measured)}, "
              f"scan_wall={t_scan:.4f}s")

    mean_t = float(np.mean(scan_times))
    std_t  = float(np.std(scan_times))
    print(f"  → mean = {mean_t:.4f} s (std {std_t:.4f}, n={n_runs})")
    return {"mean": mean_t, "std": std_t, "n_runs": n_runs,
            "all_times": scan_times, "n_regions": n_regions,
            "mean_n_measured": float(np.mean(n_measured_seen))}


def time_buchin(ds, size_grid_str, label):
    print(f"\n[{ds['name']}] Buchin rect ({label}) — 1 run, Java scan-only")
    gdf = gpd.read_file(ds["shp"]).to_crs("EPSG:4326").reset_index(drop=True)
    if ds["filter"]:
        gdf = ds["filter"](gdf)

    # Use a separate cases file per call (so the timer measurement isn't polluted
    # by a stale file). Same p, q, same RNG seed for reproducibility.
    rng_py = random.Random(SEED_BASE + 999)
    cases_path = ds["io"] / f"fair_runtime_{label}.cas"
    _build_buchin_cases_file(gdf, ds["planted_target"], P, rng_py,
                              ds["name_col"], cases_path)
    out_csv = ds["io"] / f"fair_runtime_{label}.csv"
    if out_csv.exists(): out_csv.unlink()

    planted_arg = f"{ds['planted_W']} {ds['planted_H']}"
    cmd = [
        "java", "-cp", JAVA_CP, "app.ExperimentRunner",
        "--polys",        str(ds["polys"]),
        "--names",        str(ds["names"]),
        "--pop",          str(ds["pop"]),
        "--cases",        str(cases_path),
        "--window",       "rect",
        "--planted-size", planted_arg,
        "--size-grid",    size_grid_str,
        "--out",          str(out_csv),
        "--trial-id",     "1",
    ]
    t0 = time.time()
    subprocess.run(cmd, cwd=JAVA_CWD, check=True, stdout=subprocess.DEVNULL)
    py_wall = time.time() - t0
    with open(out_csv) as f:
        line = f.read().splitlines()[1]
    parts = line.split(",")
    java_wall = float(parts[8])
    print(f"  Python wall (incl JVM startup): {py_wall:.2f}s")
    print(f"  Java scan-only wall:            {java_wall:.4f}s")
    return {"java_wall": java_wall, "py_wall": py_wall,
            "size_grid": size_grid_str, "label": label}


def main() -> None:
    print(f"=== Fair-basis runtime measurement  p={P}, q={Q}, Grid({GRID_RES}) ===")
    results = {}
    for ds in DATASETS:
        ds_res = {"geom": time_geom50(ds, N_GEOM_RUNS),
                  "buchin_5bin": time_buchin(ds, GRID_5, "5bin"),
                  "buchin_9bin": time_buchin(ds, GRID_9, "9bin")}
        results[ds["name"]] = ds_res

    print("\n" + "="*70)
    print("SUMMARY (all numbers = ONE discovery, scan-only, p=0.6, q=0.2)")
    print("="*70)
    for ds in DATASETS:
        r = results[ds["name"]]
        g, b5, b9 = r["geom"], r["buchin_5bin"], r["buchin_9bin"]
        print(f"\n{ds['name']:>8s}  (n_regions = {g['n_regions']})")
        print(f"  Geom-50         {g['mean']:.4f} s  (mean of {g['n_runs']}, std {g['std']:.4f})")
        print(f"  Buchin 5-bin    {b5['java_wall']:.4f} s")
        print(f"  Buchin 9-bin    {b9['java_wall']:.4f} s")
        print(f"  Geom-vs-Buchin5: {b5['java_wall']/g['mean']:.1f}x slower")
        print(f"  Geom-vs-Buchin9: {b9['java_wall']/g['mean']:.1f}x slower")

    # Save raw numbers
    with open(OUTPUTS / "fair_runtime_measurements.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nRaw measurements: fair_runtime_measurements.pkl")


if __name__ == "__main__":
    main()
