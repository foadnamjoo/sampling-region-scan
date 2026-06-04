
# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------
"""Disk-Large planted disk pq sweep: r=0.6° at (-92.5, 34.75).

Same protocol as Part 1 of arkansas_disk_stress.py (which used r=0.395),
just with planted_r = 0.6. Reuses helpers from arkansas_disk_stress so the
case-generation, Geom-50 disk scan, and Buchin disk wrapper are byte-identical.

20 trials × 15 p (p ∈ [0.20, 0.90] step 0.05), q = 0.20. Joblib n_jobs=4.
Output: arkansas_disk_large.pkl (same schema as arkansas_disk_small.pkl).
"""
from __future__ import annotations

import os
import pickle
import random
import sys
import time
from pathlib import Path

# DYLD before joblib subprocess forks
# (DYLD_LIBRARY_PATH should be set by the user; see README)
import numpy as np
from joblib import Parallel, delayed

import arkansas_disk_stress as ds   # reuses _sample_per_region, _coinflip, _geom_disk_scan, _buchin_disk_scan, _county_names_in_order, _per_region_count, _planted_disk, _load_ref_points, _attach_pool_jd

IO   = IO

PLANTED_R       = 0.6
P_GRID          = np.round(np.arange(0.20, 0.95, 0.05), 4)
N_TRIALS        = 20
SEED_OFF        = 80_000_000_000   # distinct from Part 1 (60e9), Part 2 (70e9)
OUT_PKL         = IO / "arkansas_disk_large.pkl"


def _job(method: str, trial: int, p: float) -> dict:
    """One Disk-Large cell — identical structure to ds._part1_job, planted_r=0.6."""
    with open(ds.POINT_DICT, "rb") as f:
        point_dict = pickle.load(f)
    target = ds._planted_disk(PLANTED_R)
    seed = SEED_OFF + 1_000_000 * trial + int(round(p * 100000))
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed + 1)
    pts = ds._sample_per_region(point_dict, rng_np)
    pts_with_case = ds._coinflip(pts, target, p, rng_py)
    base = {"method": method, "trial": trial, "p": float(p), "planted_r": PLANTED_R}
    if method == "geom":
        cx, cy, r, score, wall = ds._geom_disk_scan(pts_with_case)
        return {**base, "disk_cx": cx, "disk_cy": cy, "disk_r": r,
                "size_mul": float("nan"), "score": score, "wall_sec": wall}
    else:
        names = ds._county_names_in_order()
        counts = ds._per_region_count(pts_with_case, names=names)
        tag = f"large_t{trial:03d}_p{int(p*100):03d}_{method}"
        cases_path = ds.STRESS_DIR / f"{tag}.cas"
        out_csv    = ds.STRESS_DIR / f"{tag}.csv"
        cx, cy, r, size_mul, score, wall = ds._buchin_disk_scan(
            counts, PLANTED_R, cases_path, out_csv, trial)
        return {**base, "disk_cx": cx, "disk_cy": cy, "disk_r": r,
                "size_mul": size_mul, "score": score, "wall_sec": wall}


def main() -> None:
    ds.STRESS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"start: {time.strftime('%Y-%m-%d %H:%M:%S')}  planted_r={PLANTED_R}",
          flush=True)
    jobs = [(m, t, float(p))
            for m in ("geom", "buchin")
            for t in range(1, N_TRIALS + 1)
            for p in P_GRID]
    print(f"  {len(jobs)} jobs (2 methods × {N_TRIALS} trials × {len(P_GRID)} p)",
          flush=True)
    t0 = time.time()
    records = Parallel(n_jobs=4, verbose=10)(
        delayed(_job)(m, t, p) for (m, t, p) in jobs)
    print(f"  sweep wall: {time.time()-t0:.1f}s")

    print("\n=== Computing pool JD ===", flush=True)
    all_pts = ds._load_ref_points()
    ds._attach_pool_jd(records, all_pts, None)

    pkg = {"records": records, "p_grid": P_GRID.tolist(),
           "planted_r": PLANTED_R, "planted_center": ds.PLANTED_CENTER,
           "q": ds.Q, "n_trials": N_TRIALS,
           "size_grid": [0.5, 0.7, 1.0, 1.3, 1.5], "net_size": ds.NET_SIZE}
    with open(OUT_PKL, "wb") as f:
        pickle.dump(pkg, f)
    print(f"  wrote {OUT_PKL}")
    print(f"\ndone: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
