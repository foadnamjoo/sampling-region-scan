from __future__ import annotations

"""Utah + California — paired Grid(40) vs Grid(100) accuracy check.

Mirrors the original McClelland notebook cells (Utah cell 77, California cell 104),
with the same change as the NYC check: paired RNG so both grids see identical
case-generation draws, only the grid resolution differs.

Targets and shapefiles are verbatim from the notebook:
  Utah:       Polygon((-113,38),(-113,40.5),(-110,40.5),(-110,38))
              data/utah/geo_export_964ee856-...shp
  California: rotate(Polygon((-122.35,35.5),(-122.35,40),(-118.35,40),(-118.35,35.5)), -45)
              data/california/cnty19_1.shp

Same exp_name, n_geom_list, iterations=20, p_probs = arange(0.2, 0.95, 0.05), q=0.2.
"""

# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------

import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

# (DYLD_LIBRARY_PATH should be set by the user; see README)
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyscan
from shapely import affinity
from shapely.geometry import Point, Polygon

DATASETS = [
    {"name": "Utah",
     "shp": DATA / "Utah/geo_export_964ee856-5a3f-431f-b4c6-301973ba317c.shp",
     "target": Polygon([(-113, 38), (-113, 40.5), (-110, 40.5), (-110, 38)]),
     "pkl_out": OUTPUTS / "utah_grid100.pkl",
     "png_out": OUTPUTS / "utah_grid40_vs_grid100.png",
     "seed_base": 700_000_000_000},
    {"name": "California",
     "shp": DATA / "california/cnty19_1.shp",
     # Axis-aligned target — matches the original published California
     # figure. The -45 rotation that previously sat here was added to
     # McClelland's notebook AFTER the original figure was published;
     # the published curves were against this axis-aligned rectangle.
     "target": Polygon([(-122.35, 35.5), (-122.35, 40),
                        (-118.35, 40),   (-118.35, 35.5)]),
     "pkl_out": OUTPUTS / "california_grid100.pkl",
     "png_out": OUTPUTS / "california_grid40_vs_grid100.png",
     "seed_base": 800_000_000_000},
]

EXP_NAME    = ["Centroid", "Random Point", "Geom 5", "Geom 10", "Geom 50"]
N_GEOM_LIST = [0, 1, 5, 10, 50]
ITERATIONS  = 20
P_PROBS     = np.arange(0.20, 0.95, 0.05)
Q           = 0.20
GRIDS       = [40, 100]


def _point_dict_from_shp(gdf, seed):
    rng = random.Random(seed)
    out = {}
    for i, row in gdf.iterrows():
        poly = row.geometry
        minx, miny, maxx, maxy = poly.bounds
        pts = []
        while len(pts) < 500:
            x = rng.uniform(minx, maxx); y = rng.uniform(miny, maxy)
            if Point(x, y).within(poly): pts.append((x, y))
        out[i] = np.array([(p[0], p[1], i) for p in pts])
    return out


def _centroid_cal(gdf):
    return [(g.centroid.x, g.centroid.y) for g in gdf.geometry]


def run_dataset(ds):
    print(f"\n{'='*60}\n[{ds['name']}] loading shapefile + building pool")
    gdf = gpd.read_file(ds["shp"]).to_crs("EPSG:4326").reset_index(drop=True)
    n_regions = len(gdf)
    print(f"  n_regions = {n_regions}")
    point_dict = _point_dict_from_shp(gdf, ds["seed_base"] - 1)
    centroid_xy = _centroid_cal(gdf)
    target = ds["target"]

    results = {m: {g: [] for g in GRIDS} for m in EXP_NAME}

    for trial in range(1, ITERATIONS + 1):
        t_trial = time.time()
        for exp_i, exp_name in enumerate(EXP_NAME):
            n_geom = N_GEOM_LIST[exp_i]
            seed = ds["seed_base"] + 1_000_000 * trial + 1000 * exp_i
            rng_np = np.random.default_rng(seed)
            if exp_name == "Centroid":
                centroids = np.array(centroid_xy)
            else:
                pts = []
                for i in range(n_regions):
                    pool = point_dict[i][:, :2]
                    idx = rng_np.choice(pool.shape[0], size=n_geom, replace=False)
                    pts.append(pool[idx])
                centroids = np.vstack(pts)
            jd_per_grid = {g: [] for g in GRIDS}
            for p_prob in P_PROBS:
                rng_b = random.Random(seed + 1 + int(round(p_prob * 100000)))
                baseline = []; measured = []
                for x, y in centroids:
                    prob = rng_b.random()
                    wp = pyscan.WPoint(1.0, float(x), float(y), 1.0)
                    baseline.append(wp)
                    is_in = target.contains(Point(float(x), float(y)))
                    rate = p_prob if is_in else Q
                    if prob <= rate:
                        measured.append(wp)
                for grid_res in GRIDS:
                    if not measured:
                        jd_per_grid[grid_res].append(1.0); continue
                    grid = pyscan.Grid(grid_res, measured, baseline)
                    sg   = pyscan.max_subgrid(grid, pyscan.KULLDORF)
                    rect = grid.toRectangle(sg)
                    disc = Polygon(((rect.lowX(), rect.lowY()),
                                    (rect.lowX(), rect.upY()),
                                    (rect.upX(), rect.upY()),
                                    (rect.upX(), rect.lowY())))
                    a_u_b = 0; a_n_b = 0
                    for wp in measured:
                        pt = Point(wp.get_coord(0), wp.get_coord(1))
                        in_t = target.contains(pt); in_d = disc.contains(pt)
                        if in_t or in_d: a_u_b += 1
                        if in_t and in_d: a_n_b += 1
                    if a_u_b == 0: jd = 1.0
                    else:
                        jd = 1 - (a_n_b / a_u_b)
                        jd = max(0.0, min(jd, 1.0))
                    jd_per_grid[grid_res].append(jd)
            for g in GRIDS:
                results[exp_name][g].append(jd_per_grid[g])
        print(f"  trial {trial}/{ITERATIONS}  ({time.time()-t_trial:.1f}s)", flush=True)

    # Save
    pkg = {"dataset": ds["name"], "n_regions": n_regions,
           "iterations": ITERATIONS, "p_probs": P_PROBS.tolist(), "q": Q,
           "grids": GRIDS, "methods": EXP_NAME,
           "target_wkt": target.wkt,
           "results": {m: {g: np.asarray(v).tolist() for g, v in d.items()}
                       for m, d in results.items()}}
    with open(ds["pkl_out"], "wb") as f:
        pickle.dump(pkg, f)
    print(f"  [save] {ds['pkl_out']}")

    # Plot
    pq = P_PROBS - Q
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Centroid": "#7F7F7F", "Geom 50": "#1F77B4"}
    for method in ("Centroid", "Geom 50"):
        for grid_res in GRIDS:
            arr = np.array(results[method][grid_res])
            mean = arr.mean(axis=0); std = arr.std(axis=0)
            color = colors[method]
            ls = "-" if grid_res == 100 else "--"
            ax.fill_between(pq, mean - std, mean + std, color=color, alpha=0.10, lw=0)
            ax.plot(pq, mean, color=color, ls=ls,
                    marker="o" if grid_res == 100 else "s", markersize=4,
                    linewidth=1.8, label=f"{method} — Grid({grid_res})")
    ax.set_xlabel("p − q")
    ax.set_ylabel("Point Jaccard distance (m_sample, lower = better)")
    ax.set_title(f"{ds['name']} accuracy — Grid(40) vs Grid(100), paired RNG, 20 trials")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ds["png_out"], dpi=150)
    plt.close(fig)
    print(f"  [plot] {ds['png_out']}")

    # Report
    print(f"\n  === {ds['name']} means at key signal levels ===")
    print(f"  {'method':<14} {'metric':<24} {'Grid(40)':>10} {'Grid(100)':>10}")
    for method in ("Centroid", "Geom 50"):
        a40  = np.array(results[method][40])
        a100 = np.array(results[method][100])
        mask = pq >= 0.5
        m40 = a40[:, mask].mean(); m100 = a100[:, mask].mean()
        print(f"    {method:<12} plateau (p−q ≥ 0.5) {m40:>10.4f} {m100:>10.4f}")
        idx = int(np.argmin(np.abs(pq - 0.2)))
        v40 = a40[:, idx].mean(); v100 = a100[:, idx].mean()
        print(f"    {method:<12} mid (p−q = {pq[idx]:.2f})  {v40:>10.4f} {v100:>10.4f}")

    print(f"\n  === {ds['name']} Geom-vs-Centroid separation ===")
    for region, mask in (("plateau (p−q ≥ 0.5)", pq >= 0.5),
                         ("mid (p−q ≈ 0.2)",     np.isclose(pq, 0.2, atol=0.025))):
        c40  = np.array(results["Centroid"][40])[:, mask].mean()
        c100 = np.array(results["Centroid"][100])[:, mask].mean()
        g40  = np.array(results["Geom 50"][40])[:, mask].mean()
        g100 = np.array(results["Geom 50"][100])[:, mask].mean()
        gap40  = c40  - g40; gap100 = c100 - g100
        print(f"    {region}:  Grid(40) gap = {gap40:.4f},  Grid(100) gap = {gap100:.4f}  "
              f"(Δ = {gap100 - gap40:+.4f})")


def main():
    for ds in DATASETS:
        run_dataset(ds)


if __name__ == "__main__":
    main()
