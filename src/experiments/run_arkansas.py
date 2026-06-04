from __future__ import annotations

"""Re-run McClelland's Arkansas Geom protocol on THIS machine, serial.

Matches the original Geom-50/10/5/RandomPoint/Centroid loop in
`pyscan/build/McClelland_Thesisi_Implementation_22 (1).ipynb`, lines 1117-1410,
except:

  1.  Per (trial, p) call, we ALSO save the discovered rectangle bounds and
      the wall-clock for the pyscan scan call. The original code only kept
      the JD scalar — without the bounds we cannot recompute pool-JD later
      and cannot put Buchin and Geom on the same axis.

  2.  Serial. One pyscan scan at a time, one core. This is what Tables 1/2
      will quote as the official Geom runtime, so it must be measured under
      the same conditions Buchin will be measured under in Phase 3b/4.

  3.  Seeds are deterministic per (fig, trial, p, method) so the rerun is
      reproducible. Buchin's Phase 3 uses an independent seed stream; the
      advisor accepted "same process, not byte-identical draws" earlier.

Saves intermediate per-call records into arkansas_30_rerun.pkl and
arkansas_10_rerun.pkl. Pool-JD is computed in a separate post-step
(`postprocess_rerun.py`) so we don't pay the geometry cost inside the
timed scan loop.
"""

# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyscan
from shapely.geometry import Point, Polygon

SHP        = DATA / "arkansas/COUNTY_BOUNDARY.shp"
POINT_DICT = OUTPUTS / "arkansas_point_dict.pkl"
OUT_DIR    = IO

P_GRID  = np.round(np.arange(0.20, 0.95, 0.05), 4)
Q       = 0.20
N_TRIALS = 20
METHODS  = [
    ("Centroid",     0),
    ("Random Point", 1),
    ("Geom 5",       5),
    ("Geom 10",      10),
    ("Geom 50",      50),
]

FIGS = {
    "fig8": {
        "target":   Polygon([(-93.5, 34.0), (-93.5, 35.5),
                             (-91.5, 35.5), (-91.5, 34.0)]),
        "out_pkl":  OUT_DIR / "arkansas_30_rerun.pkl",
        "label":    "Fig 8 (30% / 2.0° × 1.5°)",
        # Distinct seed stream from Buchin Phase 3 (which used fig8 offset = 0).
        "seed_off": 20_000_000_000,
    },
    "fig9": {
        "target":   Polygon([(-92.85, 34.40), (-92.85, 35.10),
                             (-92.15, 35.10), (-92.15, 34.40)]),
        "out_pkl":  OUT_DIR / "arkansas_10_rerun.pkl",
        "label":    "Fig 9 (10% / 0.7° × 0.7°)",
        "seed_off": 30_000_000_000,
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _centroids_xy(gdf: gpd.GeoDataFrame) -> np.ndarray:
    return np.array([(g.centroid.x, g.centroid.y) for g in gdf.geometry])


def _seed(fig_name: str, trial: int, p: float, method_idx: int) -> int:
    return (FIGS[fig_name]["seed_off"]
            + 1_000_000 * trial
            + 1000 * int(round(p * 100))
            + method_idx)


def _sample_points(point_dict: dict, n_geom: int, gdf: gpd.GeoDataFrame,
                   rng_np: np.random.Generator) -> np.ndarray:
    """Per-county: pick n_geom of the 500 pool points without replacement."""
    out = []
    for i in range(len(gdf)):
        pool = point_dict[i][:, :2]  # (500, 2)
        idx  = rng_np.choice(pool.shape[0], size=n_geom, replace=False)
        out.append(pool[idx])
    return np.vstack(out)


def _run_one_call(points_xy: np.ndarray, target: Polygon, p: float,
                  rng_py: random.Random) -> tuple[tuple[float, float, float, float],
                                                  float]:
    """Build baseline + measured by point-level coin flip, run the scan,
    return (discovered_rect_bounds, wall_clock_seconds_of_scan)."""
    baseline = []
    measured = []
    for x, y in points_xy:
        prob = rng_py.random()
        wp = pyscan.WPoint(1.0, float(x), float(y), 1.0)
        baseline.append(wp)
        is_inside = target.contains(Point(x, y))
        rate = p if is_inside else Q
        if prob <= rate:
            measured.append(wp)

    # Edge case: no cases generated at all → return degenerate rect, log 0 wall.
    if not measured:
        return (0.0, 0.0, 0.0, 0.0), 0.0

    t0   = time.time()
    grid = pyscan.Grid(100, measured, baseline)
    sg   = pyscan.max_subgrid(grid, pyscan.KULLDORF)
    rect = grid.toRectangle(sg)
    wall = time.time() - t0
    return (rect.lowX(), rect.lowY(), rect.upX(), rect.upY()), wall


# ---------------------------------------------------------------------------
# main loop (serial)
# ---------------------------------------------------------------------------

def run_fig(fig_name: str, gdf: gpd.GeoDataFrame, point_dict: dict,
            centroids: np.ndarray, n_trials: int) -> None:
    meta   = FIGS[fig_name]
    target = meta["target"]
    print(f"\n=== {meta['label']} ===")
    records: list[dict] = []

    t_fig = time.time()
    for trial in range(1, n_trials + 1):
        t_trial = time.time()
        for method_name, n_geom in METHODS:
            for p in P_GRID:
                seed = _seed(fig_name, trial, float(p),
                             [m[0] for m in METHODS].index(method_name))
                rng_np = np.random.default_rng(seed)
                rng_py = random.Random(seed + 1)  # separate stream for flips

                if n_geom == 0:
                    pts = centroids
                else:
                    pts = _sample_points(point_dict, n_geom, gdf, rng_np)

                bounds, wall = _run_one_call(pts, target, float(p), rng_py)
                records.append({
                    "method":    method_name,
                    "trial":     trial,
                    "p":         float(p),
                    "rect_lowX": bounds[0],
                    "rect_lowY": bounds[1],
                    "rect_upX":  bounds[2],
                    "rect_upY":  bounds[3],
                    "wall_sec":  wall,
                    "n_points":  len(pts),
                })
        print(f"  trial {trial:2d}/{n_trials} done   "
              f"({time.time() - t_trial:.1f}s)", flush=True)

    print(f"  total wall: {time.time() - t_fig:.1f}s   "
          f"({len(records)} scan calls saved)")

    # Save intermediate (no JD yet — postprocess_rerun.py computes pool-JD).
    out = meta["out_pkl"]
    pkg = {
        "fig":      fig_name,
        "records":  records,
        "p_grid":   P_GRID.tolist(),
        "q":        Q,
        "n_trials": n_trials,
        "methods":  [m for m, _ in METHODS],
        "machine":  "PROJECT/PYSCAN @ this machine, serial",
    }
    with open(out, "wb") as f:
        pickle.dump(pkg, f)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs="+", choices=["fig8", "fig9"],
                    default=["fig8", "fig9"])
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--smoke", action="store_true",
                    help="1 trial × 1 p × 1 fig × all methods (~1 minute)")
    args = ap.parse_args()

    print("[load] arkansas shapefile + 500-pts pool ...", flush=True)
    gdf = gpd.read_file(SHP).to_crs("EPSG:4326").reset_index(drop=True)
    with open(POINT_DICT, "rb") as f:
        point_dict = pickle.load(f)
    centroids = _centroids_xy(gdf)
    print(f"[load] 75 counties, "
          f"pool has {sum(p.shape[0] for p in point_dict.values())} pts")

    if args.smoke:
        # Override globals for a smoke run.
        global P_GRID
        P_GRID = np.array([0.60])
        run_fig(args.figs[0], gdf, point_dict, centroids, n_trials=1)
        return

    for fn in args.figs:
        run_fig(fn, gdf, point_dict, centroids, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
