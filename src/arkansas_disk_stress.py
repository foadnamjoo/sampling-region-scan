"""Disk-family worst-case stress test for Buchin.

Part 1 — Small planted disk (the worst case for Buchin's 5-bin grid).
  Planted: r = 0.395° at (-92.5, 34.75) — equal-area to Fig 9 but circular.
  Methods: Geom-50 disk (pyscan.max_disk, net=400) vs Buchin disk (size grid
           {0.5, 0.7, 1.0, 1.3, 1.5} × planted_r).
  Grid:    20 trials × 15 p (p ∈ [0.20, 0.90, step 0.05]), q=0.20.
  Output:  arkansas_disk_small.pkl

Part 2 — Radius sweep at fixed p=0.60 (p−q=0.40).
  Planted radii: {0.2°, 0.3°, 0.4°, 0.6°, 0.9°} at (-92.5, 34.75).
  Trials:        20 per radius.
  Output:        arkansas_disk_radius_sweep.pkl

Both parts: same case-generation as the rest of the project (50 uniform pts/
region × Bernoulli(p inside planted disk / q outside), per-region count for
Buchin; per-point WPoints for Geom). Joblib n_jobs=4.
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

# The user is expected to have set DYLD_LIBRARY_PATH and PYTHONPATH for pyscan
# before launching this script — see the project README. If a local pyscan
# build directory needs to be on sys.path, point PYSCAN_BUILD at it.
_pyscan_build = os.environ.get("PYSCAN_BUILD")
if _pyscan_build:
    sys.path.insert(0, _pyscan_build)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO as _IO_DIR  # noqa: E402

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pyscan  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from shapely.geometry import Point, Polygon  # noqa: E402

ROOT       = REPO_ROOT
SHP        = DATA / "arkansas" / "COUNTY_BOUNDARY.shp"
POINT_DICT = OUTPUTS / "arkansas_point_dict.pkl"
IO         = _IO_DIR
JAVA_CWD   = OUTPUTS / "Maarten-implementation"
JAVA_CP    = "build"
POLYS_FILE = IO / "arkansas-polys.txt"
NAMES_FILE = IO / "arkansas-names.txt"
POP_FILE   = IO / "arkansas-pop.txt"
STRESS_DIR = IO / "stress"   # transient cases + java CSVs

N_POINTS_PER_REGION = 50
NET_SIZE            = 400
SIZE_GRID_STR       = "0.5,0.7,1.0,1.3,1.5"
Q                   = 0.20
PLANTED_CENTER      = (-92.5, 34.75)

PART1_TRIALS    = 20
PART1_P_GRID    = np.round(np.arange(0.20, 0.95, 0.05), 4)
PART1_PLANTED_R = 0.395
PART1_PKL       = IO / "arkansas_disk_small.pkl"
PART1_SEED_OFF  = 60_000_000_000

PART2_TRIALS    = 20
PART2_P         = 0.60
PART2_RADII     = [0.2, 0.3, 0.4, 0.6, 0.9]
PART2_PKL       = IO / "arkansas_disk_radius_sweep.pkl"
PART2_SEED_OFF  = 70_000_000_000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _planted_disk(r: float) -> Polygon:
    return Point(*PLANTED_CENTER).buffer(r, quad_segs=64)


def _load_county_polys(point_dict: dict) -> dict[int, Polygon]:
    """Polygon per county id (matching point_dict keys, sorted by id)."""
    gdf = gpd.read_file(SHP).to_crs("EPSG:4326").reset_index(drop=True)
    return {i: gdf.geometry.iloc[i] for i in range(len(gdf))}


def _sample_per_region(point_dict: dict, rng: np.random.Generator) -> np.ndarray:
    """50 pts/region drawn from the 500-pts pool (matches rect rerun protocol)."""
    out = []
    for i in sorted(point_dict.keys()):
        pool = point_dict[i][:, :2]
        idx  = rng.choice(pool.shape[0], size=N_POINTS_PER_REGION, replace=False)
        out.append(pool[idx])
    return np.vstack(out)


def _coinflip(pts: np.ndarray, target: Polygon, p: float, rng: random.Random
              ) -> list[tuple[float, float, bool]]:
    """Return (x, y, is_case) per point."""
    out = []
    for x, y in pts:
        rate = p if target.contains(Point(x, y)) else Q
        out.append((float(x), float(y), rng.random() <= rate))
    return out


def _pool_jd_disk(cx, cy, r, in_target_mask, all_pts) -> float:
    if r <= 0:
        return 1.0
    dx = all_pts[:, 0] - cx
    dy = all_pts[:, 1] - cy
    in_w = (dx * dx + dy * dy) <= (r * r)
    u = (in_target_mask | in_w).sum()
    n = (in_target_mask & in_w).sum()
    if u == 0:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - n / u)))


# ---------------------------------------------------------------------------
# Method-specific scans
# ---------------------------------------------------------------------------

def _geom_disk_scan(pts_with_case: list[tuple[float, float, bool]]
                    ) -> tuple[float, float, float, float, float]:
    """Run pyscan.max_disk on the per-point case set. Returns (cx, cy, r, score, wall)."""
    baseline = [pyscan.WPoint(1.0, x, y, 1.0) for x, y, _ in pts_with_case]
    measured = [pyscan.WPoint(1.0, x, y, 1.0) for x, y, c in pts_with_case if c]
    if not measured:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    net    = pyscan.my_sample(baseline, NET_SIZE)
    netpts = [pyscan.Point(b.get_coord(0), b.get_coord(1), 1.0) for b in net]
    t0 = time.time()
    disk, score = pyscan.max_disk(netpts, measured, baseline, pyscan.KULLDORF)
    wall = time.time() - t0
    o = disk.get_origin()
    return float(o.get_coord(0)), float(o.get_coord(1)), float(disk.get_radius()), \
           float(score), float(wall)


def _write_cases_file(per_region_counts: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for name, c in per_region_counts.items():
            f.write(f"{name};{c};\n")


def _buchin_disk_scan(per_region_counts: dict[str, int], planted_r: float,
                      cases_path: Path, out_csv: Path, trial_id: int
                      ) -> tuple[float, float, float, float, float, float]:
    """Run Buchin Java on the per-region case map. Returns (cx, cy, r, size_mul, score, wall)."""
    _write_cases_file(per_region_counts, cases_path)
    if out_csv.exists():
        out_csv.unlink()    # one row per call
    cmd = [
        "java", "-cp", JAVA_CP,
        "app.ExperimentRunner",
        "--polys",        str(POLYS_FILE),
        "--names",        str(NAMES_FILE),
        "--pop",          str(POP_FILE),
        "--cases",        str(cases_path),
        "--window",       "disk",
        "--planted-size", f"{planted_r:.6f}",
        "--size-grid",    SIZE_GRID_STR,
        "--out",          str(out_csv),
        "--trial-id",     str(trial_id),
    ]
    t0 = time.time()
    subprocess.run(cmd, cwd=JAVA_CWD, check=True, stdout=subprocess.DEVNULL)
    wall = time.time() - t0
    # Parse CSV row: trial_id,mode,planted,best_size_mul,best_size,best_cx,best_cy,score,wall
    with open(out_csv) as f:
        lines = f.read().splitlines()
    fields = lines[1].split(",")
    size_mul = float(fields[3])
    best_size = fields[4]   # "r=X.XXXX"
    r_val = float(best_size.replace("r=", ""))
    cx, cy = float(fields[5]), float(fields[6])
    score  = float(fields[7])
    return cx, cy, r_val, size_mul, score, wall


# ---------------------------------------------------------------------------
# Job functions (called by joblib workers)
# ---------------------------------------------------------------------------

def _county_names_in_order() -> list[str]:
    """County names ordered as they appear in arkansas-names.txt (matches point_dict key order)."""
    gdf = gpd.read_file(SHP).to_crs("EPSG:4326").reset_index(drop=True)
    return list(gdf["COUNTY"])


def _per_region_count(pts_with_case: list[tuple[float, float, bool]],
                       n_per_region: int = N_POINTS_PER_REGION,
                       names: list[str] = None) -> dict[str, int]:
    """The 50 pts of region i sit at indices [i*50, (i+1)*50)."""
    if names is None:
        names = _county_names_in_order()
    counts = {}
    for i, name in enumerate(names):
        chunk = pts_with_case[i * n_per_region:(i + 1) * n_per_region]
        counts[name] = sum(1 for _, _, c in chunk if c)
    return counts


def _part1_job(method: str, trial: int, p: float) -> dict:
    """One Part-1 cell. Returns record."""
    with open(POINT_DICT, "rb") as f:
        point_dict = pickle.load(f)
    target = _planted_disk(PART1_PLANTED_R)
    seed = PART1_SEED_OFF + 1_000_000 * trial + int(round(p * 100000))
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed + 1)
    pts = _sample_per_region(point_dict, rng_np)
    pts_with_case = _coinflip(pts, target, p, rng_py)
    base = {"part": 1, "method": method, "trial": trial, "p": float(p),
            "planted_r": PART1_PLANTED_R}
    if method == "geom":
        cx, cy, r, score, wall = _geom_disk_scan(pts_with_case)
        return {**base, "disk_cx": cx, "disk_cy": cy, "disk_r": r,
                "size_mul": float("nan"), "score": score, "wall_sec": wall}
    else:
        names = _county_names_in_order()
        counts = _per_region_count(pts_with_case, names=names)
        cases_path = STRESS_DIR / f"part1_t{trial:03d}_p{int(p*100):03d}_{method}.cas"
        out_csv    = STRESS_DIR / f"part1_t{trial:03d}_p{int(p*100):03d}_{method}.csv"
        cx, cy, r, size_mul, score, wall = _buchin_disk_scan(
            counts, PART1_PLANTED_R, cases_path, out_csv, trial)
        return {**base, "disk_cx": cx, "disk_cy": cy, "disk_r": r,
                "size_mul": size_mul, "score": score, "wall_sec": wall}


def _part2_job(method: str, trial: int, planted_r: float) -> dict:
    """One Part-2 cell. Returns record."""
    with open(POINT_DICT, "rb") as f:
        point_dict = pickle.load(f)
    target = _planted_disk(planted_r)
    p = PART2_P
    rad_idx = PART2_RADII.index(planted_r)
    seed = PART2_SEED_OFF + 1_000_000 * trial + 10_000 * rad_idx
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed + 1)
    pts = _sample_per_region(point_dict, rng_np)
    pts_with_case = _coinflip(pts, target, p, rng_py)
    base = {"part": 2, "method": method, "trial": trial, "p": p,
            "planted_r": planted_r}
    if method == "geom":
        cx, cy, r, score, wall = _geom_disk_scan(pts_with_case)
        return {**base, "disk_cx": cx, "disk_cy": cy, "disk_r": r,
                "size_mul": float("nan"), "score": score, "wall_sec": wall}
    else:
        names = _county_names_in_order()
        counts = _per_region_count(pts_with_case, names=names)
        tag = f"part2_t{trial:03d}_r{int(planted_r*1000):04d}_{method}"
        cases_path = STRESS_DIR / f"{tag}.cas"
        out_csv    = STRESS_DIR / f"{tag}.csv"
        cx, cy, r, size_mul, score, wall = _buchin_disk_scan(
            counts, planted_r, cases_path, out_csv, trial)
        return {**base, "disk_cx": cx, "disk_cy": cy, "disk_r": r,
                "size_mul": size_mul, "score": score, "wall_sec": wall}


# ---------------------------------------------------------------------------
# Pool JD post-step
# ---------------------------------------------------------------------------

def _load_ref_points() -> np.ndarray:
    with open(POINT_DICT, "rb") as f:
        d = pickle.load(f)
    return np.vstack([p[:, :2] for p in d.values()])


def _attach_pool_jd(records: list[dict], all_pts: np.ndarray,
                    planted_r_lookup) -> None:
    """In-place: add 'pool_jd' to each record using the per-record planted_r."""
    # Cache target-in masks by planted_r to avoid recomputing
    in_target_cache: dict[float, np.ndarray] = {}
    for r in records:
        pr = r["planted_r"]
        if pr not in in_target_cache:
            tgt = _planted_disk(pr)
            in_target_cache[pr] = np.array(
                [tgt.contains(Point(x, y)) for x, y in all_pts])
        r["pool_jd"] = _pool_jd_disk(r["disk_cx"], r["disk_cy"], r["disk_r"],
                                      in_target_cache[pr], all_pts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # ----- Part 1 -----
    print("\n=== Part 1: small disk stress (r=0.395, 20 trials × 15 p × 2 methods) ===")
    p1_jobs = [(m, t, float(p))
               for m in ("geom", "buchin")
               for t in range(1, PART1_TRIALS + 1)
               for p in PART1_P_GRID]
    print(f"  {len(p1_jobs)} jobs dispatched", flush=True)
    t0 = time.time()
    p1_results = Parallel(n_jobs=4, verbose=10)(
        delayed(_part1_job)(m, t, p) for (m, t, p) in p1_jobs)
    print(f"  Part 1 wall: {time.time()-t0:.1f}s")

    # ----- Part 2 -----
    print(f"\n=== Part 2: radius sweep (p={PART2_P}, 20 trials × {len(PART2_RADII)} radii × 2 methods) ===")
    p2_jobs = [(m, t, r)
               for m in ("geom", "buchin")
               for t in range(1, PART2_TRIALS + 1)
               for r in PART2_RADII]
    print(f"  {len(p2_jobs)} jobs dispatched", flush=True)
    t0 = time.time()
    p2_results = Parallel(n_jobs=4, verbose=10)(
        delayed(_part2_job)(m, t, r) for (m, t, r) in p2_jobs)
    print(f"  Part 2 wall: {time.time()-t0:.1f}s")

    # ----- Pool JD -----
    print("\n=== Computing pool JD ===", flush=True)
    all_pts = _load_ref_points()
    _attach_pool_jd(p1_results, all_pts, None)
    _attach_pool_jd(p2_results, all_pts, None)

    # ----- Save pickles -----
    pkg1 = {"records": p1_results, "p_grid": PART1_P_GRID.tolist(),
            "planted_r": PART1_PLANTED_R, "planted_center": PLANTED_CENTER,
            "q": Q, "n_trials": PART1_TRIALS,
            "size_grid": [0.5, 0.7, 1.0, 1.3, 1.5], "net_size": NET_SIZE}
    with open(PART1_PKL, "wb") as f:
        pickle.dump(pkg1, f)
    print(f"  wrote {PART1_PKL}")

    pkg2 = {"records": p2_results, "radii": PART2_RADII, "p_fixed": PART2_P,
            "planted_center": PLANTED_CENTER, "q": Q,
            "n_trials": PART2_TRIALS,
            "size_grid": [0.5, 0.7, 1.0, 1.3, 1.5], "net_size": NET_SIZE}
    with open(PART2_PKL, "wb") as f:
        pickle.dump(pkg2, f)
    print(f"  wrote {PART2_PKL}")

    print(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
