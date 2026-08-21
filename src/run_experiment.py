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

Evaluation
----------
Point Jaccard distance is measured on a FIXED evaluation set A: 500 uniform
points per input region, evaluation seed 42, built once per dataset and reused
across every method, k, rate contrast, trial and experiment seed. This is the
default for all four experiment kinds. A is drawn from its own generator, so
constructing it perturbs no scan location, no Bernoulli draw and no discovered
rectangle. Area Jaccard is polygon-based and is unaffected.

Setting PYSCAN_LEGACY_MEASURED_JD=1 selects an alternative mode that scores each
trial on the measured points it selected, rather than on A.
"""
from __future__ import annotations

import copy
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
from fixed_a_evaluation import (  # noqa: E402
    DEFAULT_EVAL_SEED,
    DEFAULT_POINTS_PER_REGION,
    EVALUATE_ON_FIXED_A,
    EvalSet,
    points_checksum,
    record_template,
)
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

# ---------- Fixed evaluation set A ----------------------------------------------
# The paper defines Point Jaccard distance on a fixed reference set A, and that
# is the default here. PYSCAN_LEGACY_MEASURED_JD=1 selects the alternative mode
# that scores each trial on its own measured points instead.
EVAL_SEED = DEFAULT_EVAL_SEED                      # 42
EVAL_POINTS_PER_REGION = DEFAULT_POINTS_PER_REGION  # 500
LEGACY_MEASURED_JD = os.environ.get("PYSCAN_LEGACY_MEASURED_JD", "0") == "1"
USE_FIXED_A = EVALUATE_ON_FIXED_A and not LEGACY_MEASURED_JD


def build_evaluator(gdf, name: str = "") -> EvalSet | None:
    """One fixed evaluation set A per dataset.

    A is drawn from its OWN generator seeded with ``EVAL_SEED``, so building it
    consumes nothing from the experiment RNG and therefore perturbs no scan
    location, no Bernoulli coin and no discovered rectangle.
    """
    if not USE_FIXED_A:
        print(f"  [{name}] PYSCAN_LEGACY_MEASURED_JD=1: Point Jaccard scored on "
              f"each trial's measured points", flush=True)
        return None
    ev = EvalSet.build(gdf, points_per_region=EVAL_POINTS_PER_REGION,
                       eval_seed=EVAL_SEED)
    print(f"  [{name}] fixed evaluation set A: |A|={len(ev.points)} "
          f"({EVAL_POINTS_PER_REGION} pts/region x {len(gdf)} regions, "
          f"eval seed {EVAL_SEED})", flush=True)
    return ev


def eval_provenance(ev: EvalSet | None) -> dict:
    """Evaluation-side provenance stored alongside every result pickle."""
    if ev is None:
        return {"mode": "legacy_measured_points", "eval_seed": None,
                "points_per_region": None}
    return {"mode": "fixed_A", **ev.provenance()}


def _snap(rng: np.random.Generator) -> dict:
    """Deep copy of the generator state, so a point set can be reconstructed."""
    return copy.deepcopy(rng.bit_generator.state)


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
    """Capped weighted sampling: n_i = clip(round(w_i * n * k_max / sum_j w_j),
    k_min, k_max), with locations uniform inside each region.

    IMPORTANT: this is NOT an equal-budget reallocation. Points removed by the
    upper cap are never redistributed, so the realized total is well below the
    uniform total. Measured on the Georgia ablation with real county population:

        Geom 5    380 points weighted vs   795 uniform  (-52%)
        Geom 10 1,014 points weighted vs 1,590 uniform  (-36%)
        Geom 50 4,577 points weighted vs 7,950 uniform  (-42%)
        Centroid and Random Point are identical (159 each) by construction.

    The cap also prevents the concentration the weighting is meant to produce:
    after clamping, Georgia's four largest counties receive only 4-5% of the
    points, not the 33% their population share would imply. Treat this arm as a
    sensitivity test, not as an allocation comparison at a fixed budget.

    The PUBLISHED Figure 11 weighted arm used weight_col="population" via
    src/experiments/run_georgia_ablation_population.py, not the aland10 default
    below. See that script.
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

def _store(record, meta, target, bounds, subgrid, p_prob, q, grid_res, pts,
           n_measured, state, evaluator, point_jd, area_jd=None) -> None:
    """Append one fully-provenanced row, if the caller asked for records."""
    if record is None:
        return
    try:
        f_value = float(subgrid.fValue())
    except Exception:
        f_value = None
    row = record_template(
        group=(meta or {}).get("group"), method=(meta or {}).get("method"),
        trial=(meta or {}).get("trial"), k=(meta or {}).get("k"),
        p_prob=p_prob, q=q, target=target, rect_bounds=bounds, f_value=f_value,
        old_jd=None if evaluator is not None else point_jd,
        fixed_a_jd=point_jd if evaluator is not None else None,
        experiment_seed=(meta or {}).get("experiment_seed"),
        target_index=(meta or {}).get("target_index", 0),
        weighted=(meta or {}).get("weighted"), grid_res=grid_res,
        rng_state_before_points=(meta or {}).get("rng_state_before_points"),
        rng_state_before_bernoulli=state,
        points_checksum=points_checksum(pts),
        n_points=int(len(pts)), n_measured=int(n_measured))
    row["eval_seed"] = EVAL_SEED if evaluator is not None else None
    row["points_per_region"] = EVAL_POINTS_PER_REGION if evaluator is not None else None
    row["dataset"] = (meta or {}).get("dataset")
    if area_jd is not None:
        row["area_jd"] = area_jd
    record.append(row)


def one_trial_jaccard_with_area(pts: np.ndarray, target: Polygon, p_prob: float, q: float,
                                grid_res: int, rng: np.random.Generator,
                                evaluator: EvalSet | None = None,
                                record: list | None = None,
                                meta: dict | None = None) -> tuple[float, float]:
    """Same as `one_trial_jaccard` but returns (point_jd, area_jd).

    Area Jaccard is computed from the polygons and is therefore identical under
    both evaluation modes.
    """
    baseline = []
    measured = []
    state = _snap(rng)
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
    bounds = (rect.lowX(), rect.lowY(), rect.upX(), rect.upY())
    discovered = Polygon([
        (rect.lowX(), rect.lowY()), (rect.lowX(), rect.upY()),
        (rect.upX(), rect.upY()), (rect.upX(), rect.lowY())
    ])
    if evaluator is not None:
        point_jd = evaluator.jaccard(target, bounds)
    else:
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
    _store(record, meta, target, bounds, subgrid, p_prob, q, grid_res, pts,
           len(measured), state, evaluator, point_jd, area_jd)
    return point_jd, area_jd


def one_trial_jaccard(pts: np.ndarray, target: Polygon, p_prob: float, q: float,
                      grid_res: int, rng: np.random.Generator,
                      evaluator: EvalSet | None = None,
                      record: list | None = None,
                      meta: dict | None = None) -> float:
    """Generate Poisson-style measured/baseline, scan for best rect, return PJD.

    With ``evaluator`` set (the default path), the Point Jaccard distance is
    measured on the fixed evaluation set A. With ``evaluator=None`` the overlap
    is counted only over the points this trial selected into ``measured``.
    """
    baseline = []
    measured = []
    state = _snap(rng)
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
    bounds = (rect.lowX(), rect.lowY(), rect.upX(), rect.upY())
    discovered = Polygon([
        (rect.lowX(), rect.lowY()), (rect.lowX(), rect.upY()),
        (rect.upX(), rect.upY()), (rect.upX(), rect.lowY())
    ])

    if evaluator is not None:
        # Point Jaccard on the fixed evaluation set A (same points for every
        # method, k, rate contrast, trial and seed).
        jd = evaluator.jaccard(target, bounds)
    else:
        # Alternative path: Point Jaccard over this trial's measured set only.
        a_u_b = a_n_b = 0
        for i, (x, y) in enumerate(pts):
            if coins[i] > (p_prob if inside[i] else q):
                continue
            p = Point(float(x), float(y))
            in_t = inside[i]
            in_d = discovered.contains(p)
            if in_t or in_d: a_u_b += 1
            if in_t and in_d: a_n_b += 1
        jd = ((a_u_b - a_n_b) / a_u_b) if a_u_b > 0 else 1.0

    _store(record, meta, target, bounds, subgrid, p_prob, q, grid_res, pts,
           len(measured), state, evaluator, jd)
    return jd


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

    evaluator = build_evaluator(gdf, name)

    random.seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    result = {m: [] for m in METHOD_NAMES}
    records: list = []
    t0 = time.time()
    for trial in range(n_trials):
        for method in METHOD_NAMES:
            pre = _snap(rng)
            pts = point_set_for_method(gdf, METHOD_K[method], rng)
            row = []
            for p_prob in pq_grid:
                jd = one_trial_jaccard(
                    pts, target, float(p_prob), Q, grid_res, rng,
                    evaluator=evaluator, record=records,
                    meta={"group": name, "dataset": name, "method": method,
                          "trial": trial, "k": METHOD_K[method],
                          "experiment_seed": seed,
                          "rng_state_before_points": pre})
                row.append(jd)
            result[method].append(row)
        elapsed = time.time() - t0
        eta = elapsed / (trial + 1) * (n_trials - trial - 1)
        print(f"  [{name}] trial {trial+1}/{n_trials} done ({elapsed:.1f}s elapsed, {eta:.1f}s ETA)", flush=True)
    pkg = {"methods": result, "pq_diff": (np.array(pq_grid) - Q).round(4).tolist(),
           "n_trials": n_trials, "seed": seed, "name": name,
           "evaluation": eval_provenance(evaluator), "records": records}
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
    evaluator = build_evaluator(gdf, name)
    random.seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    point_result = {m: [] for m in METHOD_NAMES}
    area_result  = {m: [] for m in METHOD_NAMES}
    records: list = []
    n_geom_min = {"Centroid": 0, "Random Point": 1, "Geom 5": 1,
                  "Geom 10": 5, "Geom 50": 20}
    t0 = time.time()
    for trial in range(n_trials):
        for method in METHOD_NAMES:
            k = METHOD_K[method]
            pre = _snap(rng)
            if weighted and k > 0:
                pts = weighted_point_set_for_method(
                    gdf, weights, k_max=k, k_min=n_geom_min[method], rng=rng)
            else:
                pts = point_set_for_method(gdf, k, rng)
            pt_row = []; ar_row = []
            for p_prob in pq_grid:
                pjd, ajd = one_trial_jaccard_with_area(
                    pts, target, float(p_prob), Q, grid_res, rng,
                    evaluator=evaluator, record=records,
                    meta={"group": name, "dataset": name, "method": method,
                          "trial": trial, "k": k, "experiment_seed": seed,
                          "weighted": bool(weighted),
                          "rng_state_before_points": pre})
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
           "weight_col": weight_col if weighted else None,
           "evaluation": eval_provenance(evaluator), "records": records}
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
    # A depends only on the regions, not on the target, so ONE evaluation set
    # serves every target size; EvalSet caches the in-target mask per target.
    evaluator = build_evaluator(gdf, name)
    random.seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    result = {m: [[] for _ in x_array] for m in METHOD_NAMES}
    records: list = []
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
                pre = _snap(rng)
                pts = point_set_for_method(gdf, METHOD_K[method], rng)
                pjd = one_trial_jaccard(
                    pts, target, float(p_prob), Q, grid_res, rng,
                    evaluator=evaluator, record=records,
                    meta={"group": name, "dataset": name, "method": method,
                          "trial": trial, "k": METHOD_K[method],
                          "experiment_seed": seed, "target_index": t,
                          "rng_state_before_points": pre})
                result[method][t].append(pjd)
        elapsed = time.time() - t0
        eta = elapsed / (t + 1) * (len(x_array) - t - 1)
        print(f"  [{name}] target {t+1}/{len(x_array)} (area={area_pct[-1]:.1f}%, "
              f"{elapsed:.1f}s, ETA {eta:.1f}s)", flush=True)
    pkg = {"methods": result, "area_pct": area_pct, "p_prob": p_prob,
           "pq_diff": round(p_prob - Q, 4),
           "n_trials": n_trials, "seed": seed, "name": name,
           "evaluation": eval_provenance(evaluator), "records": records}
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

    evaluator = build_evaluator(gdf, name)

    random.seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    result = {int(k): [] for k in k_values}
    records: list = []
    for trial in range(n_trials):
        for k in k_values:
            pre = _snap(rng)
            pts = point_set_for_method(gdf, int(k), rng)
            jd = one_trial_jaccard(
                pts, target, p_prob, Q, grid_res, rng,
                evaluator=evaluator, record=records,
                meta={"group": name, "dataset": name, "method": f"Geom {k}",
                      "trial": trial, "k": int(k), "experiment_seed": seed,
                      "rng_state_before_points": pre})
            result[int(k)].append(jd)
        print(f"  [{name}] trial {trial+1}/{n_trials}", flush=True)
    pkg = {"k_values": list(map(int, k_values)),
           "p_prob": p_prob, "pq_diff": round(p_prob - Q, 4),
           "by_k": result, "n_trials": n_trials, "seed": seed, "name": name,
           "evaluation": eval_provenance(evaluator), "records": records}
    out = OUT_DIR / f"{name}.pkl"
    with open(out, "wb") as f:
        pickle.dump(pkg, f)
    print(f"  [{name}] saved -> {out}", flush=True)
    return pkg


# ---------- Experiment registry ------------------------------------------------

# ---------- NYC planted targets --------------------------------------------
# Figure 3 and the Figure 12 k-sweep were genuinely run on two different NYC
# targets. Both are kept so each published figure stays reproducible.
#
#   NYC_TARGET_FIG3   lat 40.65-40.8, covers 32.8% of NYC land area.
#                     Produced Figure 3 (buchin_attempt/nyc_grid_resolution_check.py).
#   NYC_TARGET_KSWEEP lat 40.60-40.8, covers 40.5%.
#                     Produced the NYC curve in the Figure 12 k-sweep.
NYC_TARGET_FIG3 = Polygon([(-74.0, 40.65), (-74.0, 40.8),
                           (-73.8, 40.8), (-73.8, 40.65)])
NYC_TARGET_KSWEEP = Polygon([(-74.0, 40.6), (-74.0, 40.8),
                             (-73.8, 40.8), (-73.8, 40.6)])

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
        # Figure 3 was produced with latitude 40.65, not 40.6. This target covers
        # 32.8% of NYC land area, matching the paper's "approximately one-third".
        target=NYC_TARGET_FIG3,
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

    # Registered under both names: the output pickle is k_sweep_arkansas.pkl,
    # so "k_sweep_arkansas" keeps the CLI consistent with the other five.
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
        # Deliberately 40.6: this is the target that produced the published
        # Figure 12 k-sweep curve. Do NOT "fix" it to match Figure 3.
        target=NYC_TARGET_KSWEEP,
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

EXPERIMENTS["k_sweep_arkansas"] = EXPERIMENTS["k_sweep"]


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
