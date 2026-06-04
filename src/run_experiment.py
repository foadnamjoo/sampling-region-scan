"""Reproducible re-run of all paper experiments, seeded for determinism.

Architecture:
  * `run_methods_experiment(...)` — replicates the {Centroid, Random Point,
    Geom 5/10/50} × n_trials × pq_grid sweep that produces the curve figures.
  * `run_k_sweep(...)` — extra signal-boost experiment: fix pq, sweep k over
    a fine grid {2,3,5,7,10,15,20,30,50,75,100}.
  * `EXPERIMENTS` table — declarative config for each paper figure.
  * `main()` — runs everything in sequence, writing one pickle per experiment.

Run from project root:
    python src/run_experiment.py \
        [utah|nyc|california|usa|georgia_ablation|arkansas_30|arkansas_10|k_sweep|all]

Outputs go to outputs/cached_data/.
"""
from __future__ import annotations

import os
import pickle
import random
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
from tqdm import tqdm

# Repo-relative paths. ``PYSCAN_BUILD`` env var lets a user point at a local
# pyscan build directory if the package needs ``chdir`` at import time; if
# unset, we assume pyscan is already importable on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS  # noqa: E402
ROOT = REPO_ROOT

_pyscan_build = os.environ.get("PYSCAN_BUILD")
if _pyscan_build:
    BUILD = Path(_pyscan_build)
    sys.path.insert(0, str(BUILD))
    os.chdir(BUILD)
import pyscan  # noqa: E402

OUT_DIR = OUTPUTS / "cached_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEED = 7
DEFAULT_TRIALS = 20
DEFAULT_PQ = np.arange(0.2, 0.95, 0.05)  # 15 values: 0.20 .. 0.90
Q = 0.2

METHOD_NAMES = ["Centroid", "Random Point", "Geom 5", "Geom 10", "Geom 50"]
METHOD_K = {"Centroid": 0, "Random Point": 1, "Geom 5": 5, "Geom 10": 10, "Geom 50": 50}


# ---------- Sampling helpers ----------------------------------------------------

def sample_points_in_polygon(poly, k: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform-random points inside polygon via rejection sampling."""
    minx, miny, maxx, maxy = poly.bounds
    out = np.empty((k, 2))
    n = 0
    while n < k:
        bx = rng.uniform(minx, maxx, size=k * 3)
        by = rng.uniform(miny, maxy, size=k * 3)
        for x, y in zip(bx, by):
            if n == k: break
            if poly.contains(Point(x, y)):
                out[n] = (x, y); n += 1
    return out


def centroid_points(gdf) -> np.ndarray:
    """One point per region at the centroid."""
    out = np.empty((len(gdf), 2))
    for i, g in enumerate(gdf.geometry):
        c = g.centroid
        out[i] = (c.x, c.y)
    return out


def point_set_for_method(gdf, k: int, rng: np.random.Generator) -> np.ndarray:
    """k=0 → centroids; k>=1 → k uniform random points per region."""
    if k == 0:
        return centroid_points(gdf)
    pts = []
    for g in gdf.geometry:
        pts.append(sample_points_in_polygon(g, k, rng))
    return np.vstack(pts)


def weighted_point_set_for_method(gdf, weights: np.ndarray, k_max: int, k_min: int,
                                  rng: np.random.Generator) -> np.ndarray:
    """Weighted-by-population sampling: each region gets at least k_min points,
    remaining points distributed proportionally to `weights` (Georgia notebook uses
    population; we use aland10 as proxy here). Capped at k_max per region.
    """
    if k_max == 0:
        return centroid_points(gdf)
    n_regions = len(gdf)
    total_w = weights.sum()
    total_budget = n_regions * k_max
    pts = []
    for i, g in enumerate(gdf.geometry):
        n = int(round((weights[i] / total_w) * total_budget))
        n = max(k_min, min(n, k_max))
        pts.append(sample_points_in_polygon(g, n, rng))
    return np.vstack(pts)


def area_jaccard_distance(target: Polygon, discovered: Polygon) -> float:
    """1 - area(target ∩ discovered) / area(target ∪ discovered) — matches
    `area_jd_cal` in Second_Phase_8.ipynb cell 8."""
    try:
        u = target.union(discovered).area
        if u == 0:
            return 1.0
        i = target.intersection(discovered).area
        return 1.0 - (i / u)
    except Exception:
        return 1.0


# ---------- pyScan single-shot --------------------------------------------------

def one_trial_jaccard_with_area(pts: np.ndarray, target: Polygon, p_prob: float, q: float,
                                grid_res: int, rng: np.random.Generator) -> tuple[float, float]:
    """Same as `one_trial_jaccard` but returns (point_jd, area_jd)."""
    baseline = []
    measured = []
    inside = np.array([target.contains(Point(x, y)) for x, y in pts])
    coins = rng.random(len(pts))
    for i, (x, y) in enumerate(pts):
        baseline.append(pyscan.WPoint(1.0, float(x), float(y), 1.0))
        thresh = p_prob if inside[i] else q
        if coins[i] <= thresh:
            measured.append(pyscan.WPoint(1.0, float(x), float(y), 1.0))
    grid = pyscan.Grid(grid_res, measured, baseline)
    subgrid = pyscan.max_subgrid(grid, pyscan.KULLDORF)
    rect = grid.toRectangle(subgrid)
    discovered = Polygon([
        (rect.lowX(), rect.lowY()), (rect.lowX(), rect.upY()),
        (rect.upX(), rect.upY()), (rect.upX(), rect.lowY())
    ])
    a_u_b = a_n_b = 0
    for i, (x, y) in enumerate(pts):
        if coins[i] > (p_prob if inside[i] else q):
            continue
        p = Point(float(x), float(y))
        in_t = inside[i]
        in_d = discovered.contains(p)
        if in_t or in_d: a_u_b += 1
        if in_t and in_d: a_n_b += 1
    point_jd = ((a_u_b - a_n_b) / a_u_b) if a_u_b > 0 else 1.0
    area_jd = area_jaccard_distance(target, discovered)
    return point_jd, area_jd


def one_trial_jaccard(pts: np.ndarray, target: Polygon, p_prob: float, q: float,
                      grid_res: int, rng: np.random.Generator) -> float:
    """Generate Poisson-style measured/baseline, scan for best rect, return PJD."""
    baseline = []
    measured = []
    inside = np.array([target.contains(Point(x, y)) for x, y in pts])
    coins = rng.random(len(pts))
    for i, (x, y) in enumerate(pts):
        baseline.append(pyscan.WPoint(1.0, float(x), float(y), 1.0))
        thresh = p_prob if inside[i] else q
        if coins[i] <= thresh:
            measured.append(pyscan.WPoint(1.0, float(x), float(y), 1.0))

    grid = pyscan.Grid(grid_res, measured, baseline)
    subgrid = pyscan.max_subgrid(grid, pyscan.KULLDORF)
    rect = grid.toRectangle(subgrid)
    discovered = Polygon([
        (rect.lowX(), rect.lowY()), (rect.lowX(), rect.upY()),
        (rect.upX(), rect.upY()), (rect.upX(), rect.lowY())
    ])

    # Point-Jaccard distance over the measured set
    a_u_b = a_n_b = 0
    for wp in measured:
        p = Point(wp.get_x(), wp.get_y()) if hasattr(wp, "get_x") else Point(*pts[0])
        # Fallback: track inside flags via the loop variables instead of WPoint API
        pass
    # Simpler: rebuild Point list from measured (we kept them in same order)
    a_u_b = 0; a_n_b = 0
    for i, (x, y) in enumerate(pts):
        if coins[i] > (p_prob if inside[i] else q):
            continue
        p = Point(float(x), float(y))
        in_t = inside[i]
        in_d = discovered.contains(p)
        if in_t or in_d: a_u_b += 1
        if in_t and in_d: a_n_b += 1
    return ((a_u_b - a_n_b) / a_u_b) if a_u_b > 0 else 1.0


# ---------- Experiments --------------------------------------------------------

def run_methods_experiment(name: str, shp_path: str, target: Polygon,
                           n_trials: int, pq_grid, grid_res: int,
                           seed: int, crs_target: str = "EPSG:4326",
                           bbox_filter: tuple | None = None) -> dict:
    """Replicates the dict-of-method-lists experiment used for curve figures.

    If `bbox_filter=(minx, miny, maxx, maxy)` is given, only polygons whose
    bbox lies entirely inside that rectangle are kept (used to drop
    Alaska/Hawaii/territories from the USA county shapefile).
    """
    print(f"\n[{name}] starting (n_trials={n_trials}, pq={len(pq_grid)} values, "
          f"grid_res={grid_res}, seed={seed})", flush=True)
    gdf = gpd.read_file(shp_path)
    if str(gdf.crs) != crs_target:
        gdf = gdf.to_crs(crs_target)
    if bbox_filter is not None:
        minx, miny, maxx, maxy = bbox_filter
        b = gdf.geometry.bounds
        keep = (b["minx"] >= minx) & (b["maxx"] <= maxx) & \
               (b["miny"] >= miny) & (b["maxy"] <= maxy)
        before = len(gdf)
        gdf = gdf[keep].reset_index(drop=True)
        print(f"  [{name}] bbox filter kept {len(gdf)}/{before} regions", flush=True)

    random.seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    result = {m: [] for m in METHOD_NAMES}
    t0 = time.time()
    for trial in range(n_trials):
        for method in METHOD_NAMES:
            pts = point_set_for_method(gdf, METHOD_K[method], rng)
            row = []
            for p_prob in pq_grid:
                jd = one_trial_jaccard(pts, target, float(p_prob), Q, grid_res, rng)
                row.append(jd)
            result[method].append(row)
        elapsed = time.time() - t0
        eta = elapsed / (trial + 1) * (n_trials - trial - 1)
        print(f"  [{name}] trial {trial+1}/{n_trials} done ({elapsed:.1f}s elapsed, {eta:.1f}s ETA)", flush=True)
    pkg = {"methods": result, "pq_diff": (np.array(pq_grid) - Q).round(4).tolist(),
           "n_trials": n_trials, "seed": seed, "name": name}
    out = OUT_DIR / f"{name}.pkl"
    with open(out, "wb") as f:
        pickle.dump(pkg, f)
    print(f"  [{name}] saved -> {out}", flush=True)
    return pkg


def run_georgia_ablation_full(name: str, shp_path: str, target: Polygon,
                              n_trials: int, pq_grid, grid_res: int, seed: int,
                              weighted: bool = False,
                              weight_col: str = "aland10") -> dict:
    """Georgia 2x2 ablation: computes both Point JD and Area JD per trial, under
    Uniform or Weighted sampling (use `weighted=True` for weighted; `weight_col`
    defaults to aland10 — land area in m^2 — as a proxy for population since the
    bundled Georgia shapefile lacks a population column)."""
    print(f"\n[{name}] starting ({'WEIGHTED' if weighted else 'UNIFORM'} sampling, "
          f"weight_col={weight_col if weighted else 'N/A'}, n_trials={n_trials}, "
          f"pq={len(pq_grid)} values, grid_res={grid_res}, seed={seed})", flush=True)
    gdf = gpd.read_file(shp_path)
    if str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    if weighted:
        if weight_col not in gdf.columns:
            raise KeyError(f"weight column '{weight_col}' not in shapefile")
        weights = gdf[weight_col].astype(float).values
    random.seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    point_result = {m: [] for m in METHOD_NAMES}
    area_result  = {m: [] for m in METHOD_NAMES}
    n_geom_min = {"Centroid": 0, "Random Point": 1, "Geom 5": 1,
                  "Geom 10": 5, "Geom 50": 20}
    t0 = time.time()
    for trial in range(n_trials):
        for method in METHOD_NAMES:
            k = METHOD_K[method]
            if weighted and k > 0:
                pts = weighted_point_set_for_method(
                    gdf, weights, k_max=k, k_min=n_geom_min[method], rng=rng)
            else:
                pts = point_set_for_method(gdf, k, rng)
            pt_row = []; ar_row = []
            for p_prob in pq_grid:
                pjd, ajd = one_trial_jaccard_with_area(
                    pts, target, float(p_prob), Q, grid_res, rng)
                pt_row.append(pjd); ar_row.append(ajd)
            point_result[method].append(pt_row)
            area_result[method].append(ar_row)
        elapsed = time.time() - t0
        eta = elapsed / (trial + 1) * (n_trials - trial - 1)
        print(f"  [{name}] trial {trial+1}/{n_trials} ({elapsed:.1f}s, ETA {eta:.1f}s)",
              flush=True)
    pkg = {"point_jaccard": point_result,
           "area_jaccard":  area_result,
           "pq_diff": (np.array(pq_grid) - Q).round(4).tolist(),
           "n_trials": n_trials, "seed": seed, "name": name,
           "sampling": "weighted" if weighted else "uniform",
           "weight_col": weight_col if weighted else None}
    out = OUT_DIR / f"{name}.pkl"
    with open(out, "wb") as f:
        pickle.dump(pkg, f)
    print(f"  [{name}] saved -> {out}", flush=True)
    return pkg


def run_size_sweep(name: str, shp_path: str, x_base: float, y_base: float,
                   x_array, y_array, p_prob: float, n_trials: int,
                   grid_res: int, seed: int) -> dict:
    """Fig 7 style: vary target rectangle size at fixed pq. The target rectangle
    has its lower-left corner at (x_base, y_base) and upper-right at
    (x_array[t], y_array[t]) for t in 0..len(x_array)-1.  Reports
    PJD per method per target size, plus % of state area."""
    from shapely.ops import unary_union
    print(f"\n[{name}] starting (n_targets={len(x_array)}, pq={p_prob-Q:.2f}, "
          f"n_trials={n_trials}, grid_res={grid_res}, seed={seed})", flush=True)
    gdf = gpd.read_file(shp_path)
    if str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    state_area = unary_union(gdf.geometry).area  # in degree^2; fine for ratio
    random.seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    result = {m: [[] for _ in x_array] for m in METHOD_NAMES}
    area_pct = []
    t0 = time.time()
    for t in range(len(x_array)):
        target = Polygon([(x_base, y_base),
                          (x_base, y_array[t]),
                          (x_array[t], y_array[t]),
                          (x_array[t], y_base)])
        area_pct.append(float(target.area / state_area * 100.0))
        for trial in range(n_trials):
            for method in METHOD_NAMES:
                pts = point_set_for_method(gdf, METHOD_K[method], rng)
                pjd = one_trial_jaccard(pts, target, float(p_prob), Q, grid_res, rng)
                result[method][t].append(pjd)
        elapsed = time.time() - t0
        eta = elapsed / (t + 1) * (len(x_array) - t - 1)
        print(f"  [{name}] target {t+1}/{len(x_array)} (area={area_pct[-1]:.1f}%, "
              f"{elapsed:.1f}s, ETA {eta:.1f}s)", flush=True)
    pkg = {"methods": result, "area_pct": area_pct, "p_prob": p_prob,
           "pq_diff": round(p_prob - Q, 4),
           "n_trials": n_trials, "seed": seed, "name": name}
    out = OUT_DIR / f"{name}.pkl"
    with open(out, "wb") as f:
        pickle.dump(pkg, f)
    print(f"  [{name}] saved -> {out}", flush=True)
    return pkg


def run_k_sweep(name: str, shp_path: str, target: Polygon,
                k_values, p_prob: float, n_trials: int, grid_res: int,
                seed: int, crs_target: str = "EPSG:4326",
                bbox_filter: tuple | None = None) -> dict:
    """Signal-boost: PJD as a function of k at fixed pq."""
    print(f"\n[{name}] k-sweep: k={list(k_values)} pq={p_prob - Q:.2f} "
          f"trials={n_trials} seed={seed}", flush=True)
    gdf = gpd.read_file(shp_path)
    if str(gdf.crs) != crs_target:
        gdf = gdf.to_crs(crs_target)
    if bbox_filter is not None:
        minx, miny, maxx, maxy = bbox_filter
        b = gdf.geometry.bounds
        keep = (b["minx"] >= minx) & (b["maxx"] <= maxx) & \
               (b["miny"] >= miny) & (b["maxy"] <= maxy)
        gdf = gdf[keep].reset_index(drop=True)
        print(f"  [{name}] bbox filter kept {len(gdf)} regions", flush=True)

    random.seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    result = {int(k): [] for k in k_values}
    for trial in range(n_trials):
        for k in k_values:
            pts = point_set_for_method(gdf, int(k), rng)
            jd = one_trial_jaccard(pts, target, p_prob, Q, grid_res, rng)
            result[int(k)].append(jd)
        print(f"  [{name}] trial {trial+1}/{n_trials}", flush=True)
    pkg = {"k_values": list(map(int, k_values)),
           "p_prob": p_prob, "pq_diff": round(p_prob - Q, 4),
           "by_k": result, "n_trials": n_trials, "seed": seed, "name": name}
    out = OUT_DIR / f"{name}.pkl"
    with open(out, "wb") as f:
        pickle.dump(pkg, f)
    print(f"  [{name}] saved -> {out}", flush=True)
    return pkg


# ---------- Experiment registry ------------------------------------------------

EXPERIMENTS = {
    # Each entry → (kind, kwargs).  kind ∈ {"methods", "k_sweep"}.
    "utah": ("methods", dict(
        name="utah",
        shp_path=str(DATA / "utah" / "geo_export_964ee856-5a3f-431f-b4c6-301973ba317c.shp"),
        target=Polygon([(-113, 38), (-113, 40.5), (-110, 40.5), (-110, 38)]),
        n_trials=DEFAULT_TRIALS, pq_grid=DEFAULT_PQ, grid_res=40, seed=DEFAULT_SEED)),

    "nyc": ("methods", dict(
        name="nyc",
        shp_path=str(DATA / "nyc" / "ZIP_CODE_040114.shp"),
        target=Polygon([(-74, 40.6), (-74, 40.8), (-73.8, 40.8), (-73.8, 40.6)]),
        n_trials=DEFAULT_TRIALS, pq_grid=DEFAULT_PQ, grid_res=40, seed=DEFAULT_SEED)),

    "california": ("methods", dict(
        name="california",
        shp_path=str(DATA / "california" / "cnty19_1.shp"),
        target=Polygon([(-122.35, 35.5), (-122.35, 40), (-118.35, 40), (-118.35, 35.5)]),
        n_trials=DEFAULT_TRIALS, pq_grid=DEFAULT_PQ, grid_res=40, seed=DEFAULT_SEED)),

    "usa": ("methods", dict(
        name="usa",
        shp_path=str(DATA / "usa" / "cb_2017_us_county_500k.shp"),
        target=Polygon([(-100, 33), (-100, 40), (-90, 40), (-90, 33)]),
        # Mainland-only filter drops Alaska, Hawaii, Puerto Rico, Guam, AS, MP, VI.
        bbox_filter=(-130, 24, -65, 50),
        # grid_res=100 matches the paper's appendix Listing 5 (paper.tex).
        n_trials=DEFAULT_TRIALS, pq_grid=DEFAULT_PQ, grid_res=100, seed=DEFAULT_SEED)),

    "georgia_ablation": ("methods", dict(
        name="georgia_ablation",
        shp_path=str(DATA / "georgia" / "GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp"),
        # ~30% of state area — central rectangle (matches paper).
        target=Polygon([(-84.5, 31.5), (-84.5, 34), (-82.5, 34), (-82.5, 31.5)]),
        n_trials=DEFAULT_TRIALS, pq_grid=DEFAULT_PQ, grid_res=40, seed=DEFAULT_SEED)),

    "k_sweep": ("k_sweep", dict(
        name="k_sweep_arkansas",
        shp_path=str(DATA / "arkansas" / "COUNTY_BOUNDARY.shp"),
        target=Polygon([(-93.5, 34), (-93.5, 35.5), (-91.5, 35.5), (-91.5, 34)]),
        k_values=[2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100],
        p_prob=0.35,  # pq diff = 0.15
        n_trials=DEFAULT_TRIALS, grid_res=40, seed=DEFAULT_SEED)),

    # Per-state k-sweeps so we can overlay all 6 curves on one axes.  Same k grid
    # and same pq diff = 0.15 across every state, so the comparison is apples-to-apples.
    "k_sweep_utah": ("k_sweep", dict(
        name="k_sweep_utah",
        shp_path=str(DATA / "utah" / "geo_export_964ee856-5a3f-431f-b4c6-301973ba317c.shp"),
        target=Polygon([(-113, 38), (-113, 40.5), (-110, 40.5), (-110, 38)]),
        k_values=[2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100],
        p_prob=0.35, n_trials=DEFAULT_TRIALS, grid_res=40, seed=DEFAULT_SEED)),

    "k_sweep_california": ("k_sweep", dict(
        name="k_sweep_california",
        shp_path=str(DATA / "california" / "cnty19_1.shp"),
        target=Polygon([(-122.35, 35.5), (-122.35, 40), (-118.35, 40), (-118.35, 35.5)]),
        k_values=[2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100],
        p_prob=0.35, n_trials=DEFAULT_TRIALS, grid_res=40, seed=DEFAULT_SEED)),

    "k_sweep_nyc": ("k_sweep", dict(
        name="k_sweep_nyc",
        shp_path=str(DATA / "nyc" / "ZIP_CODE_040114.shp"),
        target=Polygon([(-74, 40.6), (-74, 40.8), (-73.8, 40.8), (-73.8, 40.6)]),
        k_values=[2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100],
        p_prob=0.35, n_trials=DEFAULT_TRIALS, grid_res=40, seed=DEFAULT_SEED)),

    "k_sweep_georgia": ("k_sweep", dict(
        name="k_sweep_georgia",
        shp_path=str(DATA / "georgia" / "GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp"),
        target=Polygon([(-85.0, 31.0), (-85.0, 32.89), (-83.61, 32.89), (-83.61, 31.0)]),
        k_values=[2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100],
        p_prob=0.35, n_trials=DEFAULT_TRIALS, grid_res=40, seed=DEFAULT_SEED)),

    "k_sweep_usa": ("k_sweep", dict(
        name="k_sweep_usa",
        shp_path=str(DATA / "usa" / "cb_2017_us_county_500k.shp"),
        target=Polygon([(-100, 33), (-100, 40), (-90, 40), (-90, 33)]),
        bbox_filter=(-130, 24, -65, 50),  # mainland only — drops AK/HI/territories
        k_values=[2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100],
        p_prob=0.35, n_trials=DEFAULT_TRIALS, grid_res=40, seed=DEFAULT_SEED)),

    # Fig 7 — Georgia size sweep at fixed pq=0.4.  Targets match the original
    # notebook: anchor (x_base,y_base)=(-85,31), expand to 10 sizes.
    "georgia_size_sweep": ("size_sweep", dict(
        name="georgia_size_sweep",
        shp_path=str(DATA / "georgia" / "GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp"),
        x_base=-85.0, y_base=31.0,
        x_array=list(np.linspace(-84.5, -82.0, 10)),
        y_array=list(np.linspace(31.5, 34.0, 10)),
        p_prob=0.6,    # pq diff = 0.4 (paper Fig 7 caption)
        n_trials=DEFAULT_TRIALS, grid_res=40, seed=DEFAULT_SEED)),

    # Fig 10 — both Uniform and Weighted Georgia ablation.  Both runs compute
    # *both* Point JD and Area JD so each fills two of the four 2x2 panels.
    "georgia_ablation_uniform": ("georgia_ablation_full", dict(
        name="georgia_ablation_uniform",
        shp_path=str(DATA / "georgia" / "GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp"),
        target=Polygon([(-85.0, 31.0), (-85.0, 32.89), (-83.61, 32.89), (-83.61, 31.0)]),
        n_trials=DEFAULT_TRIALS, pq_grid=DEFAULT_PQ, grid_res=40, seed=DEFAULT_SEED,
        weighted=False)),

    "georgia_ablation_weighted": ("georgia_ablation_full", dict(
        name="georgia_ablation_weighted",
        shp_path=str(DATA / "georgia" / "GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp"),
        target=Polygon([(-85.0, 31.0), (-85.0, 32.89), (-83.61, 32.89), (-83.61, 31.0)]),
        n_trials=DEFAULT_TRIALS, pq_grid=DEFAULT_PQ, grid_res=40, seed=DEFAULT_SEED,
        weighted=True, weight_col="aland10")),
}


def main(argv):
    # Allow overriding the seed for multi-seed reruns:
    #     python run_experiment.py SEED=31 utah nyc usa ...
    # When seed != DEFAULT_SEED, the output pickle name gets a `_seed{N}` suffix.
    seed_override = None
    keys = []
    for a in argv:
        if a.startswith("SEED="):
            seed_override = int(a.split("=", 1)[1])
        else:
            keys.append(a)
    if not keys or "all" in keys:
        keys = list(EXPERIMENTS.keys())
    for key in keys:
        if key not in EXPERIMENTS:
            print(f"unknown experiment: {key}; available: {list(EXPERIMENTS.keys())}")
            continue
        kind, kw = EXPERIMENTS[key]
        if seed_override is not None and seed_override != DEFAULT_SEED:
            kw = dict(kw)  # copy so the registry stays clean
            kw["seed"] = seed_override
            kw["name"] = f"{kw['name']}_seed{seed_override}"
        if kind == "methods":
            run_methods_experiment(**kw)
        elif kind == "k_sweep":
            run_k_sweep(**kw)
        elif kind == "georgia_ablation_full":
            run_georgia_ablation_full(**kw)
        elif kind == "size_sweep":
            run_size_sweep(**kw)


if __name__ == "__main__":
    main(sys.argv[1:])
