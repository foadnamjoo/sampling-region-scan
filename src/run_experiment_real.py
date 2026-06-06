"""Real-world dataset pipeline (Scottish lip cancer).

Tests whether the rectangle scan on REAL per-region m(z) / b(z) recovers a
region defined by a KNOWN CAUSE (occupational UV exposure, proxied by AFF).

Imports helpers from existing modules — does NOT modify them:
  - sample_points_in_polygon : run_experiment.sample_points_in_polygon
  - reference_set            : shape_floor.reference_set
  - paper plot helpers       : paper_plots.apply_style_v9
"""
from __future__ import annotations
import os
import sys
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.ops import unary_union
import fiona

# Repo paths
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from _paths import REPO_ROOT, DATA, OUTPUTS  # noqa: E402

# pyScan via the user's local build dir (set PYSCAN_BUILD env var)
_pyscan_build = os.environ.get("PYSCAN_BUILD")
if _pyscan_build:
    sys.path.insert(0, _pyscan_build)
    os.chdir(_pyscan_build)
import pyscan  # noqa: E402

# Reuse existing helpers — no copies
from run_experiment import sample_points_in_polygon  # noqa: E402
from shape_floor import reference_set, point_jd, in_rect  # noqa: E402
import paper_plots as pp  # noqa: E402


# ===========================================================================
# Loader (Scotland lip cancer)
# ===========================================================================

SCOT_SHP = DATA / "scotland_lip" / "scotlip" / "scotlip.shp"


def _safe_read_bng(shp_path: Path) -> gpd.GeoDataFrame:
    """Read shapefile assuming British National Grid (EPSG:27700) when no .prj.
    Falls back to fiona+shape() if shapely's batch construction trips on this
    user's geopandas/shapely combo."""
    rows = []
    with fiona.open(shp_path) as src:
        for f in src:
            try:
                rows.append({"geometry": shape(f["geometry"]),
                             **dict(f["properties"])})
            except Exception:
                continue
    return gpd.GeoDataFrame(rows, crs="EPSG:27700")


def load_scotland(shp_path: Path = SCOT_SHP) -> gpd.GeoDataFrame:
    """Return the 56-district frame in EPSG:4326 with NAME / CANCER / CEXP / AFF."""
    gdf = _safe_read_bng(shp_path)
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


# ===========================================================================
# STEP 2 — cause-defined ground truth (independent of case counts)
# ===========================================================================

def build_ground_truth_from_aff(gdf: gpd.GeoDataFrame,
                                aff_col: str,
                                quantile: float) -> tuple[MultiPolygon | Polygon, list[str]]:
    """Return the union of districts whose AFF >= the given quantile, plus the
    name list. This S* is constructed entirely from the *cause* (AFF), with no
    reference to the observed case counts."""
    threshold = float(gdf[aff_col].quantile(quantile))
    mask = gdf[aff_col] >= threshold
    names = gdf.loc[mask, "NAME"].tolist()
    # Pairwise reduce to dodge shapely's batched create_collection numpy dtype
    # bug on this user's environment.
    geoms = list(gdf.loc[mask, "geometry"])
    s_star = geoms[0]
    for g in geoms[1:]:
        s_star = s_star.union(g)
    return s_star, names


# ===========================================================================
# STEP 3 — Geom-k with real m / b
# ===========================================================================

def geom_k_points_real(gdf: gpd.GeoDataFrame,
                       m: np.ndarray,
                       b: np.ndarray,
                       k: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k=0  -> 1 centroid per region carrying full m_i, b_i.
    k>=1 -> k uniform points per region, each carrying m_i/k and b_i/k."""
    if k == 0:
        coords = np.empty((len(gdf), 2))
        for i, g in enumerate(gdf.geometry):
            c = g.centroid
            coords[i] = (c.x, c.y)
        return coords, np.asarray(m, dtype=float), np.asarray(b, dtype=float)

    coords_list, mpp_list, bpp_list = [], [], []
    for i, g in enumerate(gdf.geometry):
        pts = sample_points_in_polygon(g, k, rng)
        coords_list.append(pts)
        mpp_list.append(np.full(k, m[i] / k))
        bpp_list.append(np.full(k, b[i] / k))
    return (np.vstack(coords_list),
            np.concatenate(mpp_list),
            np.concatenate(bpp_list))


# ===========================================================================
# STEP 4 — discover rectangle with real weights (KULLDORF; arg0 = mass)
# ===========================================================================

def discover_rect_real(coords: np.ndarray,
                       m_pp: np.ndarray,
                       b_pp: np.ndarray,
                       grid_res: int = 100) -> Polygon:
    """measured = WPoint(m_i/k, x, y, 1.0); baseline = WPoint(b_i/k, x, y, 1.0).
    Returns the shapely rectangle for max_subgrid under KULLDORF."""
    measured = [pyscan.WPoint(float(m_pp[i]), float(coords[i, 0]),
                              float(coords[i, 1]), 1.0)
                for i in range(len(coords)) if m_pp[i] > 0.0]
    baseline = [pyscan.WPoint(float(b_pp[i]), float(coords[i, 0]),
                              float(coords[i, 1]), 1.0)
                for i in range(len(coords)) if b_pp[i] > 0.0]
    grid = pyscan.Grid(grid_res, measured, baseline)
    sg = pyscan.max_subgrid(grid, pyscan.KULLDORF)
    r = grid.toRectangle(sg)
    return Polygon([(r.lowX(), r.lowY()), (r.lowX(), r.upY()),
                    (r.upX(), r.upY()), (r.upX(), r.lowY())])


# ===========================================================================
# STEP 5 — JD vs irregular S* on a fixed dense set A
# ===========================================================================

def point_jaccard_real(s_star, discovered: Polygon, A: np.ndarray) -> float:
    """1 - |A ∩ (S* ∩ Ŝ)| / |A ∩ (S* ∪ Ŝ)|.  Deterministic.  S* and Ŝ are
    shapely geometries; A is an (n, 2) array of reference points."""
    in_s = np.array([s_star.contains(Point(x, y)) for x, y in A])
    # discovered is axis-aligned; vectorize for speed
    minx, miny, maxx, maxy = discovered.bounds
    in_d = ((A[:, 0] >= minx) & (A[:, 0] <= maxx)
            & (A[:, 1] >= miny) & (A[:, 1] <= maxy))
    union = (in_s | in_d).sum()
    inter = (in_s & in_d).sum()
    return 1.0 - inter / union if union > 0 else 1.0


def shape_floor_rect(s_star, A: np.ndarray, n_grid: int = 14) -> dict:
    """Best-possible JD fitting one axis-aligned rectangle to S* over A.
    Cached x/y in-band masks; only the AND of cached masks happens in the
    inner loop, so total cost is O(n_grid^4 * |A| / 64) bitwise ops."""
    in_s = np.array([s_star.contains(Point(x, y)) for x, y in A])
    nS = int(in_s.sum())
    minx, miny = A[:, 0].min(), A[:, 1].min()
    maxx, maxy = A[:, 0].max(), A[:, 1].max()
    cxs = np.linspace(minx, maxx, n_grid)
    cys = np.linspace(miny, maxy, n_grid)
    hws = np.linspace((maxx - minx) * 0.05, (maxx - minx) * 0.5, n_grid)
    hhs = np.linspace((maxy - miny) * 0.05, (maxy - miny) * 0.5, n_grid)
    # Precompute every x-band and y-band membership once
    x_masks = [(np.abs(A[:, 0] - cx) <= hw) for cx in cxs for hw in hws]
    y_masks = [(np.abs(A[:, 1] - cy) <= hh) for cy in cys for hh in hhs]
    best = {"jd": 1.0, "cx": None, "cy": None, "hw": None, "hh": None}
    for ix, xm in enumerate(x_masks):
        cx_idx, hw_idx = divmod(ix, n_grid)
        cx, hw = cxs[cx_idx], hws[hw_idx]
        for iy, ym in enumerate(y_masks):
            cy_idx, hh_idx = divmod(iy, n_grid)
            cy, hh = cys[cy_idx], hhs[hh_idx]
            in_r = xm & ym
            nR = int(in_r.sum())
            if nR == 0:
                continue
            inter = int((in_s & in_r).sum())
            union = nS + nR - inter
            if union == 0:
                continue
            jd = 1.0 - inter / union
            if jd < best["jd"]:
                best.update(jd=float(jd), cx=float(cx), cy=float(cy),
                            hw=float(hw), hh=float(hh))
    return best


# ===========================================================================
# STEP 6 — k-sweep + diagnostics
# ===========================================================================

def k_sweep(gdf: gpd.GeoDataFrame,
            s_star,
            A: np.ndarray,
            k_values=(0, 1, 5, 10, 20, 50),
            n_trials: int = 20,
            seed: int = 7,
            grid_res: int = 100) -> pd.DataFrame:
    """Returns a DataFrame with mean / std PJD per k.  k=0 is deterministic
    (one trial); other k values are averaged over `n_trials`."""
    rng_master = np.random.default_rng(seed)
    m = gdf["CANCER"].astype(float).to_numpy()
    b = gdf["CEXP"].astype(float).to_numpy()
    rows = []
    for k in k_values:
        if k == 0:
            coords, mpp, bpp = geom_k_points_real(gdf, m, b, 0, rng_master)
            disc = discover_rect_real(coords, mpp, bpp, grid_res=grid_res)
            pjd = point_jaccard_real(s_star, disc, A)
            rows.append({"k": k, "mean": pjd, "std": 0.0, "n": 1,
                         "rect": (disc.bounds)})
        else:
            jds = []
            disc_last = None
            for t in range(n_trials):
                rng = np.random.default_rng(seed + 1000 * k + t)
                coords, mpp, bpp = geom_k_points_real(gdf, m, b, k, rng)
                disc = discover_rect_real(coords, mpp, bpp, grid_res=grid_res)
                jds.append(point_jaccard_real(s_star, disc, A))
                disc_last = disc
            rows.append({"k": k, "mean": float(np.mean(jds)),
                         "std": float(np.std(jds)), "n": n_trials,
                         "rect": disc_last.bounds})
    return pd.DataFrame(rows)


# ===========================================================================
# Diagnostic plotters (write into outputs/, NOT figures/)
# ===========================================================================

def plot_smr_choropleth(gdf: gpd.GeoDataFrame,
                        s_star,
                        rect_k0: Polygon,
                        rect_k50: Polygon,
                        out_path: Path,
                        title: str = "Scotland lip cancer — SMR & discovered windows"):
    """SMR choropleth + AFF ground-truth outline + Centroid vs Geom-50 rects."""
    fig, ax = plt.subplots(figsize=(9, 8))
    smr = (gdf["CANCER"].astype(float)
           / gdf["CEXP"].replace(0, np.nan).astype(float))
    gdf_plot = gdf.copy(); gdf_plot["SMR"] = smr
    gdf_plot.plot(column="SMR", ax=ax, cmap="OrRd",
                  edgecolor="#888", linewidth=0.3, legend=True,
                  legend_kwds={"label": "SMR (observed / expected)",
                               "shrink": 0.6})
    # Ground truth (AFF quantile) outline
    if hasattr(s_star, "geoms"):
        for g in s_star.geoms:
            ax.plot(*g.exterior.xy, color="#1B5E20", linewidth=2.2,
                    linestyle="--", label="AFF ground truth S*")
    else:
        ax.plot(*s_star.exterior.xy, color="#1B5E20", linewidth=2.2,
                linestyle="--", label="AFF ground truth S*")

    def _add_rect(rect: Polygon, color, label):
        minx, miny, maxx, maxy = rect.bounds
        ax.add_patch(mpatches.Rectangle((minx, miny), maxx - minx, maxy - miny,
                                         facecolor="none", edgecolor=color,
                                         linewidth=2.0, label=label))

    _add_rect(rect_k0, "red", "Centroid (k=0) rect")
    _add_rect(rect_k50, "darkmagenta", "Geom-50 rect")
    ax.set_title(title)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pjd_vs_k(df: pd.DataFrame, floor: float,
                  out_path: Path,
                  title: str = "PJD vs k (Scotland, AFF q=0.75)"):
    pp.apply_style_v9()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    k = df["k"].to_numpy(); mu = df["mean"].to_numpy(); sd = df["std"].to_numpy()
    # x-axis: treat k=0 specially with a label
    x = np.where(k == 0, 0.5, k.astype(float))  # offset for log-friendly display
    ax.fill_between(x, mu - sd, mu + sd, color="darkmagenta", alpha=0.18, lw=0)
    ax.plot(x, mu, color="darkmagenta", lw=2.0, marker="s", ms=6, label="Geom-k")
    ax.axhline(floor, color="black", lw=1.2, ls=(0, (5, 2)),
               label=f"shape floor JD={floor:.3f}")
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([str(int(v)) if v >= 1 else "0" for v in k])
    ax.set_xlabel("k (samples per region; 0 = Centroid)")
    ax.set_ylabel("Point Jaccard Distance")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Main: run STEP 6 for q=0.75, also dump q=0.50 / 0.67 tables.
# ===========================================================================

def main():
    OUT = OUTPUTS / "scotland"
    OUT.mkdir(parents=True, exist_ok=True)

    print("[load] scotland_lip ...")
    gdf = load_scotland()

    # Fixed dense reference set A (500 pts/region; deterministic seed)
    print("[A] building 500-per-region reference set ...")
    A = reference_set(gdf, n_per_region=500, seed=42)
    print(f"  |A| = {len(A)}")

    results_per_q = {}
    for q in (0.50, 0.67, 0.75):
        s_star, names = build_ground_truth_from_aff(gdf, "AFF", q)
        threshold = float(gdf["AFF"].quantile(q))
        print(f"\n=== AFF quantile = {q}  (threshold = {threshold:.0f}%; {len(names)} districts) ===")
        print(f"  districts: {names}")

        df = k_sweep(gdf, s_star, A,
                     k_values=(0, 1, 5, 10, 20, 50),
                     n_trials=20, seed=7, grid_res=100)
        results_per_q[q] = (df, s_star, names, threshold)
        print(df[["k", "mean", "std", "n"]].to_string(index=False))

        # shape-floor reference (best axis-aligned rect to S*)
        floor = shape_floor_rect(s_star, A, n_grid=25)
        print(f"  shape-floor (best rect over S*) JD = {floor['jd']:.4f}")

        # Save per-quantile PJD vs k PNG only for q=0.75 (the headline)
        if q == 0.75:
            png_pjd = OUT / f"pjd_vs_k_q{q:.2f}.png"
            plot_pjd_vs_k(df, floor["jd"], png_pjd,
                          title=f"PJD vs k (Scotland, AFF q={q})")
            print(f"  wrote {png_pjd}")

            # SMR choropleth + Centroid + Geom-50 rects
            rng = np.random.default_rng(7)
            coords0, mpp0, bpp0 = geom_k_points_real(
                gdf, gdf["CANCER"].astype(float).to_numpy(),
                gdf["CEXP"].astype(float).to_numpy(), 0, rng)
            rect_k0 = discover_rect_real(coords0, mpp0, bpp0, grid_res=100)
            coords50, mpp50, bpp50 = geom_k_points_real(
                gdf, gdf["CANCER"].astype(float).to_numpy(),
                gdf["CEXP"].astype(float).to_numpy(), 50, rng)
            rect_k50 = discover_rect_real(coords50, mpp50, bpp50, grid_res=100)
            png_map = OUT / f"smr_choropleth_q{q:.2f}.png"
            plot_smr_choropleth(gdf, s_star, rect_k0, rect_k50, png_map,
                                title="Scotland lip cancer SMR + AFF S* (q=0.75) "
                                      "+ Centroid (red) vs Geom-50 (magenta) rectangles")
            print(f"  wrote {png_map}")

    # Persist tables
    pkl = OUT / "scotland_pjd_results.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({str(q): df for q, (df, *_rest) in results_per_q.items()}, f)
    print(f"\n[save] {pkl}")

    return results_per_q


# ===========================================================================
# Diagnostics (CHECK A / B / C / D)
# ===========================================================================

def _districts_inside_rect(gdf: gpd.GeoDataFrame, rect: Polygon) -> pd.DataFrame:
    """Return a small DataFrame of districts whose CENTROID lies inside `rect`,
    with NAME / CANCER / CEXP / SMR columns."""
    minx, miny, maxx, maxy = rect.bounds
    rows = []
    for i, g in enumerate(gdf.geometry):
        c = g.centroid
        if minx <= c.x <= maxx and miny <= c.y <= maxy:
            cancer = float(gdf.iloc[i]["CANCER"])
            cexp = float(gdf.iloc[i]["CEXP"])
            smr = cancer / cexp if cexp > 0 else float("inf")
            rows.append({"NAME": gdf.iloc[i]["NAME"],
                         "CANCER": int(cancer), "CEXP": round(cexp, 2),
                         "SMR": round(smr, 2),
                         "AFF": int(gdf.iloc[i]["AFF"])})
    return pd.DataFrame(rows).sort_values("SMR", ascending=False).reset_index(drop=True)


def check_a_scan_landing(gdf: gpd.GeoDataFrame, grid_res: int = 100) -> None:
    """CHECK A — where does the scan land + sample-size sanity check."""
    print("\n" + "=" * 78)
    print("CHECK A — scan landing (noise-chasing test)")
    print("=" * 78)
    m = gdf["CANCER"].astype(float).to_numpy()
    b = gdf["CEXP"].astype(float).to_numpy()

    # k=0 centroid scan (deterministic)
    rng = np.random.default_rng(0)
    coords0, mpp0, bpp0 = geom_k_points_real(gdf, m, b, 0, rng)
    rect0 = discover_rect_real(coords0, mpp0, bpp0, grid_res=grid_res)
    print(f"\n--- k=0 (Centroid) — deterministic ---")
    print(f"  rect bounds (lon, lat):  "
          f"x=[{rect0.bounds[0]:.3f}, {rect0.bounds[2]:.3f}]  "
          f"y=[{rect0.bounds[1]:.3f}, {rect0.bounds[3]:.3f}]")
    inside = _districts_inside_rect(gdf, rect0)
    print(f"  {len(inside)} district centroids inside:")
    print(inside.to_string(index=False))

    # k=50 with 3 seeds
    for seed in (7, 31, 67):
        rng = np.random.default_rng(seed)
        coords, mpp, bpp = geom_k_points_real(gdf, m, b, 50, rng)
        rect = discover_rect_real(coords, mpp, bpp, grid_res=grid_res)
        print(f"\n--- k=50, seed={seed} ---")
        print(f"  rect bounds (lon, lat):  "
              f"x=[{rect.bounds[0]:.3f}, {rect.bounds[2]:.3f}]  "
              f"y=[{rect.bounds[1]:.3f}, {rect.bounds[3]:.3f}]")
        inside = _districts_inside_rect(gdf, rect)
        print(f"  {len(inside)} district centroids inside:")
        print(inside.to_string(index=False))

    # Sample-size sanity
    n_low1 = int((gdf["CEXP"] < 1.0).sum())
    n_low2 = int((gdf["CEXP"] < 2.0).sum())
    print(f"\n--- CEXP stability ---")
    print(f"  districts with CEXP < 1.0 (one-case-shifts-the-SMR-by-1): {n_low1}/56")
    print(f"  districts with CEXP < 2.0 (noisy rates):                  {n_low2}/56")


# ---------- CHECK B ----------

def build_adjacency(gdf: gpd.GeoDataFrame, tol_metres: float = 500.0) -> dict[int, set[int]]:
    """Adjacency: two districts are neighbours if they touch OR if the buffered
    (tol/2) geometries intersect. Done in EPSG:27700 (BNG) so tol_metres is real
    metres."""
    bng = gdf.to_crs("EPSG:27700")
    buf = bng.geometry.buffer(tol_metres / 2.0)
    sindex = bng.sindex
    adj = {i: set() for i in range(len(bng))}
    for i in range(len(bng)):
        cand = list(sindex.intersection(buf.iloc[i].bounds))
        for j in cand:
            if j == i:
                continue
            if buf.iloc[i].intersects(bng.iloc[j].geometry):
                adj[i].add(j); adj[j].add(i)
    return adj


def largest_component(indices: list[int], adj: dict[int, set[int]]) -> list[int]:
    """Connected-component BFS restricted to `indices`."""
    idx_set = set(indices)
    seen = set()
    best = []
    for start in indices:
        if start in seen:
            continue
        comp = []
        stack = [start]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v); comp.append(v)
            for u in adj[v]:
                if u in idx_set and u not in seen:
                    stack.append(u)
        if len(comp) > len(best):
            best = comp
    return sorted(best)


def check_b_coherent_targets(gdf: gpd.GeoDataFrame, A: np.ndarray):
    """CHECK B — drop the disconnected islands by keeping only the largest
    connected component of districts at each AFF threshold."""
    print("\n" + "=" * 78)
    print("CHECK B — coherent ground truths (largest connected component)")
    print("=" * 78)
    adj = build_adjacency(gdf, tol_metres=500.0)
    targets = {}
    for aff_thresh in (10, 7):
        qual_idx = [i for i in range(len(gdf)) if gdf.iloc[i]["AFF"] >= aff_thresh]
        comp_idx = largest_component(qual_idx, adj)
        comp_names = [gdf.iloc[i]["NAME"] for i in comp_idx]
        # union of geometries in the component (pairwise to dodge shapely batch bug)
        geoms = [gdf.iloc[i].geometry for i in comp_idx]
        s_star = geoms[0]
        for g in geoms[1:]:
            s_star = s_star.union(g)
        floor = shape_floor_rect(s_star, A, n_grid=14)
        targets[aff_thresh] = (s_star, comp_idx, comp_names, floor)
        print(f"\n--- AFF >= {aff_thresh}% : {len(qual_idx)} qualifying, "
              f"{len(comp_idx)} in largest component ---")
        print(f"  districts: {comp_names}")
        print(f"  shape-floor JD (best axis-aligned rect to S*_coherent) = {floor['jd']:.4f}")
    return targets


# ---------- CHECK C ----------

def check_c_kweep(gdf: gpd.GeoDataFrame, targets: dict, A: np.ndarray,
                  k_values=(0, 1, 5, 10, 20, 50), n_trials: int = 20,
                  grid_res: int = 100) -> dict:
    """Re-run k-sweep against each coherent S* and print gap/floor diagnostics."""
    print("\n" + "=" * 78)
    print("CHECK C — k-sweep vs S*_coherent")
    print("=" * 78)
    out = {}
    for aff_thresh, (s_star, comp_idx, comp_names, floor) in targets.items():
        floor_jd = floor["jd"]
        df = k_sweep(gdf, s_star, A, k_values=k_values,
                     n_trials=n_trials, seed=7, grid_res=grid_res)
        gap = float(df.loc[df["k"] == 0, "mean"].iloc[0]
                    - df.loc[df["k"] == 50, "mean"].iloc[0])
        geom50 = float(df.loc[df["k"] == 50, "mean"].iloc[0])
        floor_dist = geom50 - floor_jd
        verdict = ("POSITIVE (gap >= 0.10 & approaches floor)"
                   if gap >= 0.10 and floor_dist <= 0.10
                   else "NEGATIVE (curve flat / floor still far)")
        print(f"\n--- S*_coherent at AFF>={aff_thresh}% ({len(comp_idx)} districts) ---")
        print(df[["k", "mean", "std", "n"]].to_string(index=False))
        print(f"  shape-floor JD                       = {floor_jd:.4f}")
        print(f"  gap = PJD(k=0) - PJD(k=50)           = {gap:+.4f}")
        print(f"  geom50_minus_floor = PJD(k=50)-floor = {floor_dist:+.4f}")
        print(f"  DECISION: {verdict}")
        out[aff_thresh] = (df, gap, floor_dist, floor_jd)
    return out


# ---------- CHECK D ----------

def check_d_clean_map(gdf: gpd.GeoDataFrame, s_star,
                      rect_k0: Polygon, rect_k50: Polygon,
                      out_path: Path,
                      title: str) -> None:
    """Cleaner diagnostic map: deduped legend + CEXP<1 hatched districts."""
    fig, ax = plt.subplots(figsize=(9, 8.5))
    smr = (gdf["CANCER"].astype(float)
           / gdf["CEXP"].replace(0, np.nan).astype(float))
    gdf_plot = gdf.copy(); gdf_plot["SMR"] = smr
    gdf_plot.plot(column="SMR", ax=ax, cmap="OrRd",
                  edgecolor="#888", linewidth=0.3, legend=True,
                  legend_kwds={"label": "SMR (observed / expected)", "shrink": 0.6})

    # Hatch the small-CEXP districts (CEXP<2 — there are no <1 in scotlip;
    # widen the band so the diagnostic still shows the noisiest counts).
    low_cexp_threshold = 2.0
    low_cexp = gdf_plot[gdf_plot["CEXP"] < low_cexp_threshold]
    if len(low_cexp) > 0:
        low_cexp.plot(ax=ax, facecolor="none", edgecolor="#0033CC",
                      linewidth=0.6, hatch="///")

    # S*_coherent outline (single legend entry only)
    if hasattr(s_star, "geoms"):
        polys = list(s_star.geoms)
    else:
        polys = [s_star]
    for i, g in enumerate(polys):
        label = "AFF>=10 coherent S*" if i == 0 else None
        ax.plot(*g.exterior.xy, color="#1B5E20",
                linewidth=2.2, linestyle="--", label=label)

    # Discovered rects (single legend entry each)
    def _add_rect(rect, color, label):
        minx, miny, maxx, maxy = rect.bounds
        ax.add_patch(mpatches.Rectangle((minx, miny), maxx - minx, maxy - miny,
                                         facecolor="none", edgecolor=color,
                                         linewidth=2.0, label=label))
    _add_rect(rect_k0, "red", "Centroid (k=0) rect")
    _add_rect(rect_k50, "darkmagenta", "Geom-50 rect")

    handles, labels = ax.get_legend_handles_labels()
    seen = set(); dedup_h, dedup_l = [], []
    for h, lab in zip(handles, labels):
        if lab is None or lab in seen:
            continue
        seen.add(lab); dedup_h.append(h); dedup_l.append(lab)
    if len(low_cexp) > 0:
        dedup_h.append(mpatches.Patch(facecolor="white", edgecolor="#0033CC",
                                       hatch="///",
                                       label=f"CEXP < {low_cexp_threshold} (unstable rate)"))
        dedup_l.append(f"CEXP < {low_cexp_threshold} (unstable rate)")
    ax.legend(dedup_h, dedup_l, loc="lower left", fontsize=9, frameon=True)

    ax.set_title(title)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def diagnose():
    OUT = OUTPUTS / "scotland"
    OUT.mkdir(parents=True, exist_ok=True)

    print("[load] scotland_lip ...")
    gdf = load_scotland()
    print("[A] building 500-per-region reference set ...")
    A = reference_set(gdf, n_per_region=500, seed=42)
    print(f"  |A| = {len(A)}")

    check_a_scan_landing(gdf)
    targets = check_b_coherent_targets(gdf, A)
    check_c_kweep(gdf, targets, A)

    # CHECK D — render the coherent AFF>=10 picture
    m = gdf["CANCER"].astype(float).to_numpy()
    b = gdf["CEXP"].astype(float).to_numpy()
    rng = np.random.default_rng(0)
    coords0, mpp0, bpp0 = geom_k_points_real(gdf, m, b, 0, rng)
    rect_k0 = discover_rect_real(coords0, mpp0, bpp0, grid_res=100)
    coords50, mpp50, bpp50 = geom_k_points_real(gdf, m, b, 50, rng)
    rect_k50 = discover_rect_real(coords50, mpp50, bpp50, grid_res=100)
    png = OUT / "diagnostic_clean_q_aff10.png"
    check_d_clean_map(gdf, targets[10][0], rect_k0, rect_k50, png,
                      title="Scotland — SMR + AFF>=10 coherent S* + "
                            "Centroid (red) vs Geom-50 (magenta) rectangles "
                            "(blue hatch = CEXP<1.0)")
    print(f"\n[CHECK D] wrote {png}")


# ===========================================================================
# Snow 1854 cholera — compact, boxable, cause-defined cluster
# ===========================================================================
# Reuses sample_points_in_polygon, reference_set, geom_k_points_real,
# discover_rect_real, point_jaccard_real, shape_floor_rect. Everything happens
# in EPSG:27700 (metres).

CHOLERA_DEATHS = DATA / "cholera_snow" / "Cholera_Deaths.shp"
CHOLERA_PUMPS  = DATA / "cholera_snow" / "Pumps.shp"


def load_cholera() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, tuple[float, float]]:
    """Return (deaths, pumps, broad_st_xy).  Both frames in EPSG:27700.
    Broad Street is identified as the pump closest to the death-weighted centroid.
    """
    deaths = gpd.read_file(CHOLERA_DEATHS)
    pumps = gpd.read_file(CHOLERA_PUMPS)
    if str(deaths.crs) != "EPSG:27700":
        deaths = deaths.to_crs("EPSG:27700")
    if str(pumps.crs) != "EPSG:27700":
        pumps = pumps.to_crs("EPSG:27700")
    # Expand Count column into per-row weight, default 1
    if "Count" not in deaths.columns:
        deaths = deaths.copy(); deaths["Count"] = 1
    # Death-weighted centroid (per-location Count weights)
    w = deaths["Count"].astype(float).to_numpy()
    xs = deaths.geometry.x.to_numpy()
    ys = deaths.geometry.y.to_numpy()
    cx = float((xs * w).sum() / w.sum())
    cy = float((ys * w).sum() / w.sum())
    # Broad St = pump closest to that centroid
    pxs = pumps.geometry.x.to_numpy(); pys = pumps.geometry.y.to_numpy()
    d2 = (pxs - cx) ** 2 + (pys - cy) ** 2
    idx = int(np.argmin(d2))
    broad = (float(pxs[idx]), float(pys[idx]))
    return deaths, pumps, broad


# ----- C2: Voronoi-cell reporting regions ----------------------------------

def voronoi_regions(deaths: gpd.GeoDataFrame, n_seeds: int, seed: int,
                    hull_buffer_m: float = 50.0) -> gpd.GeoDataFrame:
    """Sprinkle `n_seeds` random points over the buffered convex hull of the
    deaths, build a Voronoi tessellation, clip cells to the hull.  Returns a
    GeoDataFrame with columns ['geometry'] in EPSG:27700."""
    from scipy.spatial import Voronoi
    pts_xy = np.column_stack([deaths.geometry.x.to_numpy(),
                              deaths.geometry.y.to_numpy()])
    # Manual convex hull (sidesteps shapely's batched create_collection bug).
    from scipy.spatial import ConvexHull as _CH
    _pts = np.column_stack([deaths.geometry.x.to_numpy(),
                            deaths.geometry.y.to_numpy()])
    _ch = _CH(_pts)
    hull = Polygon(_pts[_ch.vertices]).buffer(hull_buffer_m)
    # Drop seeds uniformly inside the hull (rejection sampling)
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = hull.bounds
    seeds = []
    while len(seeds) < n_seeds:
        bx = rng.uniform(minx, maxx, size=n_seeds * 3)
        by = rng.uniform(miny, maxy, size=n_seeds * 3)
        for x, y in zip(bx, by):
            if len(seeds) == n_seeds:
                break
            if hull.contains(Point(x, y)):
                seeds.append((x, y))
    seeds_arr = np.asarray(seeds)
    # Add 4 far-away points so Voronoi has bounded cells we can clip
    span = max(maxx - minx, maxy - miny) * 10
    fars = np.array([[minx - span, miny - span], [minx - span, maxy + span],
                     [maxx + span, miny - span], [maxx + span, maxy + span]])
    vor_pts = np.vstack([seeds_arr, fars])
    vor = Voronoi(vor_pts)
    cells = []
    for i in range(len(seeds_arr)):
        region_idx = vor.point_region[i]
        verts = vor.regions[region_idx]
        if -1 in verts or len(verts) == 0:
            continue
        poly = Polygon([vor.vertices[v] for v in verts])
        poly = poly.intersection(hull)
        if poly.is_empty or not poly.is_valid:
            continue
        cells.append(poly)
    return gpd.GeoDataFrame({"geometry": cells}, crs="EPSG:27700")


def cholera_region_attrs(deaths: gpd.GeoDataFrame,
                         regions: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-region (m_i, b_i): m_i = sum(Count) of death points inside the cell,
    b_i = area of the cell (uniform-density null)."""
    # Spatial index lookup
    sindex = deaths.sindex
    m = np.zeros(len(regions), dtype=float)
    death_xy = np.column_stack([deaths.geometry.x.to_numpy(),
                                deaths.geometry.y.to_numpy()])
    counts = deaths["Count"].astype(float).to_numpy()
    for i, g in enumerate(regions.geometry):
        cand = list(sindex.intersection(g.bounds))
        if not cand:
            continue
        for j in cand:
            if g.contains(Point(death_xy[j, 0], death_xy[j, 1])):
                m[i] += counts[j]
    b = regions.geometry.area.to_numpy() / regions.geometry.area.sum() * m.sum()
    # Note: scaling b so sum(b) == sum(m) keeps SMR=1 globally (uniform null).
    return m, b


# ----- C3: cause-defined ground truth --------------------------------------

def broad_st_disk(broad: tuple[float, float],
                  pumps: gpd.GeoDataFrame) -> tuple[Polygon, float]:
    """Disk centred at Broad St, radius = (distance to nearest other pump)/2."""
    pxs = pumps.geometry.x.to_numpy(); pys = pumps.geometry.y.to_numpy()
    d = np.sqrt((pxs - broad[0]) ** 2 + (pys - broad[1]) ** 2)
    d_sorted = np.sort(d)
    nearest = float(d_sorted[1])  # d_sorted[0]=0 (Broad itself)
    r = nearest / 2.0
    return Point(*broad).buffer(r), r


def broad_st_voronoi(broad: tuple[float, float],
                     pumps: gpd.GeoDataFrame,
                     hull) -> Polygon:
    """Broad St's Voronoi cell among the 13 (here 8) pumps, clipped to `hull`."""
    from scipy.spatial import Voronoi
    pxs = pumps.geometry.x.to_numpy(); pys = pumps.geometry.y.to_numpy()
    minx, miny, maxx, maxy = hull.bounds
    span = max(maxx - minx, maxy - miny) * 10
    fars = np.array([[minx - span, miny - span], [minx - span, maxy + span],
                     [maxx + span, miny - span], [maxx + span, maxy + span]])
    pts = np.column_stack([pxs, pys])
    vor = Voronoi(np.vstack([pts, fars]))
    # Find Broad's index
    d2 = (pxs - broad[0]) ** 2 + (pys - broad[1]) ** 2
    bidx = int(np.argmin(d2))
    verts = vor.regions[vor.point_region[bidx]]
    poly = Polygon([vor.vertices[v] for v in verts])
    return poly.intersection(hull)


# ----- internal-SMR sanity check (after C0 orientation) --------------------

def internal_smr(rect: Polygon, coords: np.ndarray,
                 m_pp: np.ndarray, b_pp: np.ndarray) -> tuple[float, float, float]:
    """Return (m_inside, b_inside, smr) for the discovered rectangle."""
    minx, miny, maxx, maxy = rect.bounds
    in_r = ((coords[:, 0] >= minx) & (coords[:, 0] <= maxx)
            & (coords[:, 1] >= miny) & (coords[:, 1] <= maxy))
    m_in = float(m_pp[in_r].sum())
    b_in = float(b_pp[in_r].sum())
    smr = m_in / b_in if b_in > 0 else float("nan")
    return m_in, b_in, smr


# ----- C4: per-N k-sweep ----------------------------------------------------

def cholera_kweep_single(deaths, regions, s_star, A,
                          k_values=(0, 1, 5, 10, 20, 50),
                          n_trials=20, seed=7, grid_res=100) -> pd.DataFrame:
    """k-sweep with sanity check that discovered rect has internal SMR>1
    (excess). Returns a DataFrame including the last-seed centre x/y for
    sanity reporting."""
    m, b = cholera_region_attrs(deaths, regions)
    rows = []
    for k in k_values:
        if k == 0:
            rng = np.random.default_rng(seed)
            coords, mpp, bpp = geom_k_points_real(regions, m, b, 0, rng)
            disc = discover_rect_real(coords, mpp, bpp, grid_res=grid_res)
            pjd = point_jaccard_real(s_star, disc, A)
            min_, bin_, smr = internal_smr(disc, coords, mpp, bpp)
            cx = (disc.bounds[0] + disc.bounds[2]) / 2
            cy = (disc.bounds[1] + disc.bounds[3]) / 2
            rows.append({"k": k, "mean": pjd, "std": 0.0, "n": 1,
                         "smr_last": smr, "cx_last": cx, "cy_last": cy})
        else:
            jds, smrs, cxs_, cys_ = [], [], [], []
            for t in range(n_trials):
                rng = np.random.default_rng(seed + 1000 * k + t)
                coords, mpp, bpp = geom_k_points_real(regions, m, b, k, rng)
                disc = discover_rect_real(coords, mpp, bpp, grid_res=grid_res)
                jds.append(point_jaccard_real(s_star, disc, A))
                _, _, smr = internal_smr(disc, coords, mpp, bpp)
                smrs.append(smr)
                cxs_.append((disc.bounds[0] + disc.bounds[2]) / 2)
                cys_.append((disc.bounds[1] + disc.bounds[3]) / 2)
            rows.append({"k": k, "mean": float(np.mean(jds)),
                         "std": float(np.std(jds)), "n": n_trials,
                         "smr_last": float(np.mean(smrs)),
                         "cx_last": float(np.mean(cxs_)),
                         "cy_last": float(np.mean(cys_))})
    return pd.DataFrame(rows)


# ----- C5: diagnostic PNGs --------------------------------------------------

def cholera_map_png(deaths, regions, pumps, broad, disk_gt,
                    rect_k0, rect_k50, out_path: Path,
                    title: str = "Snow 1854 — cholera deaths + Voronoi regions"):
    fig, ax = plt.subplots(figsize=(9, 8))
    regions.boundary.plot(ax=ax, color="#888", linewidth=0.6,
                          label=None)
    # deaths sized by Count
    sizes = 6 * deaths["Count"].astype(float)
    ax.scatter(deaths.geometry.x, deaths.geometry.y, s=sizes,
               color="#1F1F1F", alpha=0.55, label="deaths (size = count)")
    # pumps
    ax.scatter(pumps.geometry.x, pumps.geometry.y, s=80, marker="^",
               color="#0033CC", edgecolor="white", linewidth=0.8,
               label="pumps")
    # Broad St
    ax.scatter([broad[0]], [broad[1]], s=320, marker="*",
               facecolor="#FFD700", edgecolor="black", linewidth=1.2,
               zorder=6, label="Broad St pump")
    # Disk GT (dashed)
    ax.plot(*disk_gt.exterior.xy, color="#1B5E20", linewidth=2.2,
            linestyle="--", label="cause-defined disk S*")
    # Discovered rects
    for rect, color, lab in ((rect_k0, "red", "Centroid (k=0) rect"),
                             (rect_k50, "darkmagenta", "Geom-50 rect")):
        minx, miny, maxx, maxy = rect.bounds
        ax.add_patch(mpatches.Rectangle((minx, miny), maxx - minx, maxy - miny,
                                         facecolor="none", edgecolor=color,
                                         linewidth=2.0, label=lab))
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("easting (m, EPSG:27700)")
    ax.set_ylabel("northing (m, EPSG:27700)")
    h, l = ax.get_legend_handles_labels()
    seen = set(); h2, l2 = [], []
    for hh, ll in zip(h, l):
        if ll in seen: continue
        seen.add(ll); h2.append(hh); l2.append(ll)
    ax.legend(h2, l2, loc="upper left", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cholera_pjd_png(curves_by_N: dict[int, pd.DataFrame], floor: float,
                    out_path: Path):
    """PJD vs k for N in {15,25,40} on one axes; dashed shape-floor line."""
    pp.apply_style_v9()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {15: "#377EB8", 25: "#4DAF4A", 40: "#984EA3"}
    for N, df in curves_by_N.items():
        k = df["k"].to_numpy().astype(float)
        x = np.where(k == 0, 0.5, k)
        mu = df["mean"].to_numpy(); sd = df["std"].to_numpy()
        ax.fill_between(x, mu - sd, mu + sd, color=colors[N], alpha=0.15, lw=0)
        ax.plot(x, mu, color=colors[N], marker="s", ms=6, lw=2.0,
                label=f"N={N} regions")
    ax.axhline(floor, color="black", lw=1.2, ls=(0, (5, 2)),
               label=f"disk shape-floor JD={floor:.3f}")
    ax.set_xscale("log")
    all_k = sorted({int(v) for df in curves_by_N.values() for v in df["k"]})
    x_ticks = [0.5 if v == 0 else v for v in all_k]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["0" if v == 0 else str(v) for v in all_k])
    ax.set_xlabel("k (samples per region; 0 = Centroid)")
    ax.set_ylabel("Point Jaccard Distance")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Snow cholera — PJD vs k for three region coarsenesses")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cholera_run():
    OUT = OUTPUTS / "cholera"
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Snow 1854 cholera — compact cluster diagnostic")
    print("=" * 78)
    deaths, pumps, broad = load_cholera()
    n_deaths_rows = len(deaths)
    n_deaths_total = int(deaths["Count"].sum())
    n_pumps = len(pumps)
    print(f"  deaths file: {n_deaths_rows} rows ({n_deaths_total} total counts)")
    print(f"  pumps file:  {n_pumps} pumps")
    print(f"  Broad St (closest to death-weighted centroid): "
          f"({broad[0]:.1f}, {broad[1]:.1f}) m, EPSG:27700")
    # Death-weighted centroid for reference
    w = deaths["Count"].astype(float).to_numpy()
    cx = float((deaths.geometry.x.to_numpy() * w).sum() / w.sum())
    cy = float((deaths.geometry.y.to_numpy() * w).sum() / w.sum())
    print(f"  death-weighted centroid: ({cx:.1f}, {cy:.1f}) m")

    # Cause-defined disk GT
    disk, r = broad_st_disk(broad, pumps)
    print(f"\nDisk GT: centre = Broad St, radius = {r:.1f} m "
          f"(= half the distance to nearest other pump)")

    # Hull for clipping — manual convex hull (avoids shapely batch bug)
    from scipy.spatial import ConvexHull as _CH
    _pts = np.column_stack([deaths.geometry.x.to_numpy(),
                            deaths.geometry.y.to_numpy()])
    _ch = _CH(_pts)
    hull = Polygon(_pts[_ch.vertices]).buffer(50.0)
    bsv = broad_st_voronoi(broad, pumps, hull)
    print(f"Alt GT: Broad St Voronoi cell area = {bsv.area:.0f} m^2")

    # Dense reference set A — built fresh for the disk GT using the SAME
    # `reference_set` helper from shape_floor.py (one frame containing the
    # disk and surrounding hull so JD has both inside/outside coverage).
    print("\nBuilding reference set A (500 pts/region) over the regions+disk hull...")

    results = {}
    for N in (15, 25, 40):
        seed_dfs = []
        seed_last_artifacts = None
        for vseed in (1, 2, 3):
            regions = voronoi_regions(deaths, N, seed=vseed, hull_buffer_m=50.0)
            # A built once per (N, vseed), using the actual region geometry —
            # reference_set takes a gdf with .geometry, 500 pts per region.
            A = reference_set(regions, n_per_region=500, seed=42)
            df = cholera_kweep_single(deaths, regions, disk, A,
                                       k_values=(0, 1, 5, 10, 20, 50),
                                       n_trials=20, seed=7, grid_res=100)
            df["vseed"] = vseed
            seed_dfs.append(df)
            if vseed == 2:  # keep middle seed's artifacts for the map figure
                m, b = cholera_region_attrs(deaths, regions)
                rng = np.random.default_rng(0)
                coords0, mpp0, bpp0 = geom_k_points_real(regions, m, b, 0, rng)
                rect_k0 = discover_rect_real(coords0, mpp0, bpp0, grid_res=100)
                coords50, mpp50, bpp50 = geom_k_points_real(
                    regions, m, b, 50, rng)
                rect_k50 = discover_rect_real(coords50, mpp50, bpp50, grid_res=100)
                seed_last_artifacts = (regions, rect_k0, rect_k50,
                                       coords0, mpp0, bpp0,
                                       coords50, mpp50, bpp50)

        big = pd.concat(seed_dfs, ignore_index=True)
        agg = big.groupby("k").agg(mean=("mean", "mean"),
                                    std=("mean", "std"),
                                    n=("n", "sum"),
                                    smr_last=("smr_last", "mean"),
                                    cx_last=("cx_last", "mean"),
                                    cy_last=("cy_last", "mean")).reset_index()
        # κ proxy on regions for this N (avg over the 3 seeds)
        kappas = []
        m_dists = []
        for vseed in (1, 2, 3):
            regs = voronoi_regions(deaths, N, seed=vseed, hull_buffer_m=50.0)
            areas = regs.geometry.area
            kappas.append(areas.max() / areas.min())
            m_local, _ = cholera_region_attrs(deaths, regs)
            m_dists.append((m_local.min(), float(np.median(m_local)),
                            m_local.max(), int((m_local == 0).sum())))
        kappa = float(np.mean(kappas))
        m_min = min(x[0] for x in m_dists)
        m_med = float(np.mean([x[1] for x in m_dists]))
        m_max = max(x[2] for x in m_dists)
        m_zero = int(np.mean([x[3] for x in m_dists]))

        # Shape floor (one-shot using the first seed's A)
        A0 = reference_set(voronoi_regions(deaths, N, seed=1,
                                            hull_buffer_m=50.0),
                            n_per_region=500, seed=42)
        floor = shape_floor_rect(disk, A0, n_grid=14)

        gap = float(agg.loc[agg["k"] == 0, "mean"].iloc[0]
                    - agg.loc[agg["k"] == 50, "mean"].iloc[0])
        floor_gap = float(agg.loc[agg["k"] == 50, "mean"].iloc[0]
                          - floor["jd"])
        smr_k50 = float(agg.loc[agg["k"] == 50, "smr_last"].iloc[0])
        cx_k50 = float(agg.loc[agg["k"] == 50, "cx_last"].iloc[0])
        cy_k50 = float(agg.loc[agg["k"] == 50, "cy_last"].iloc[0])
        d_to_broad = float(np.hypot(cx_k50 - broad[0], cy_k50 - broad[1]))

        print(f"\n--- N = {N} regions  (avg over 3 Voronoi seeds) ---")
        print(f"  κ proxy (area max/min, avg): {kappa:.1f}")
        print(f"  m_i (deaths per region): min={m_min:.0f}, "
              f"median≈{m_med:.1f}, max={m_max:.0f}, zero-count regions≈{m_zero}")
        print(agg[["k", "mean", "std", "n"]].to_string(index=False))
        print(f"  shape-floor JD (rect→disk on A)      = {floor['jd']:.4f}")
        print(f"  gap = PJD(k=0) - PJD(k=50)           = {gap:+.4f}")
        print(f"  geom50_minus_floor = PJD(k=50)-floor = {floor_gap:+.4f}")
        print(f"  SANITY (k=50): internal SMR = {smr_k50:.3f}  "
              f"({'EXCESS' if smr_k50 > 1 else 'DEFICIT'})")
        print(f"  SANITY (k=50): rect centre = ({cx_k50:.1f}, {cy_k50:.1f}); "
              f"Broad St = ({broad[0]:.1f}, {broad[1]:.1f}); "
              f"distance = {d_to_broad:.1f} m")

        results[N] = (agg, floor["jd"], seed_last_artifacts)

    # C5 — PNG 1: middle seed of N=25
    regions25, rect_k0, rect_k50, *_ = results[25][2]
    png_map = OUT / "snow_map_N25.png"
    cholera_map_png(deaths, regions25, pumps, broad, disk,
                    rect_k0, rect_k50, png_map,
                    title="Snow 1854 — cholera deaths + Voronoi (N=25) + "
                          "Centroid (red) vs Geom-50 (magenta) rects")
    print(f"\n[CHECK C5] wrote {png_map}")

    # C5 — PNG 2: three-curve PJD vs k. All three share the same disk-floor.
    floors = {N: results[N][1] for N in (15, 25, 40)}
    avg_floor = float(np.mean(list(floors.values())))
    curves = {N: results[N][0] for N in (15, 25, 40)}
    png_pjd = OUT / "snow_pjd_vs_k.png"
    cholera_pjd_png(curves, avg_floor, png_pjd)
    print(f"[CHECK C5] wrote {png_pjd}")


# ===========================================================================
# Valley Fever — Coccidioidomycosis in California counties
# ===========================================================================
# Real fixed regions = CA's 58 counties (loader drops 2 not in the shapefile;
# R3 confirms they're non-SJV and low-incidence). Real m(z) = case counts from
# CHHS open-data; real b(z) = mean county population over the window.
# Everything in EPSG:3310 (CA Albers, metres).

# The CA county shapefile is the same one run_experiment.py uses for the
# synthetic California experiment. In a fresh clone this would live at
# DATA / "california" / "cnty19_1.shp" — but here the file is in the original
# working tree, so we resolve the path the user already has.
_VF_CA_DEFAULT = DATA / "california" / "cnty19_1.shp"
if not _VF_CA_DEFAULT.exists():
    _VF_CA_DEFAULT = Path("/Users/foadnamjoo/PROJECT/PYSCAN/pyscan/data/data/"
                           "California_County_Boundaries/cnty19_1.shp")
CA_SHP_VF = _VF_CA_DEFAULT
CHHS_CSV  = DATA / "valley_fever" / "idb.csv"

# Cause-defined S* candidates ----------------------------------------------
SJV_8 = {"Kern", "Kings", "Tulare", "Fresno", "Madera", "Merced",
         "San Joaquin", "Stanislaus"}
SJV_5_CORE = {"Kern", "Kings", "Tulare", "Fresno", "Madera"}
# CDPH-recognized highly-endemic counties (SJV core + southern CA known
# endemic + Antelope Valley). This is a documented "endemic-county" set used
# in CDPH surveillance materials.
CDPH_ENDEMIC = SJV_8 | {"Los Angeles", "San Luis Obispo", "San Diego",
                        "Ventura", "Santa Barbara", "Riverside", "Imperial"}


def _vf_safe_read_ca(shp_path: Path) -> gpd.GeoDataFrame:
    """Read CA county SHP one feature at a time (sidesteps shapely batch bug)."""
    rows = []
    with fiona.open(shp_path) as src:
        crs = src.crs
        for f in src:
            try:
                rows.append({"geometry": shape(f["geometry"]),
                             **dict(f["properties"])})
            except Exception:
                continue
    return gpd.GeoDataFrame(rows, crs=crs)


def _vf_pairwise_union(geoms):
    s = geoms[0]
    for g in geoms[1:]:
        s = s.union(g)
    return s


def load_california_counties() -> gpd.GeoDataFrame:
    """Return CA counties in EPSG:3310 with a stable NAME column. Polygons
    that share a county name (multi-island features) are dissolved."""
    gdf = _vf_safe_read_ca(CA_SHP_VF)
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf.to_crs("EPSG:3310")
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf.rename(columns={"COUNTY_NAM": "NAME"})
    if gdf["NAME"].duplicated().any():
        merged = []
        for name, sub in gdf.groupby("NAME"):
            merged.append({"NAME": name,
                           "geometry": _vf_pairwise_union(list(sub.geometry))})
        gdf = gpd.GeoDataFrame(merged, crs="EPSG:3310")
    return gdf


def load_cocci_window(year_lo: int, year_hi: int):
    """Return (cases_by_county_name, pop_by_county_name) for the inclusive
    [year_lo, year_hi] window of Coccidioidomycosis (Sex==Total)."""
    df = pd.read_csv(CHHS_CSV, low_memory=False)
    df = df[(df["Disease"] == "Coccidioidomycosis") & (df["Sex"] == "Total")
            & (df["County"] != "California")
            & (df["Year"] >= year_lo) & (df["Year"] <= year_hi)]
    cases = df.groupby("County")["Cases"].sum()
    pop = df.groupby("County")["Population"].mean()
    return cases, pop


def attach_cocci_to_counties(gdf: gpd.GeoDataFrame,
                              cases: pd.Series,
                              pop: pd.Series) -> gpd.GeoDataFrame:
    """Add m / b columns to the county frame. b is rescaled so sum(b)=sum(m)
    (global SMR = 1) — gives a uniform-rate null hypothesis."""
    out = gdf.copy()
    out["m"] = out["NAME"].map(cases).fillna(0).astype(float)
    out["b"] = out["NAME"].map(pop).astype(float)
    if out["b"].isna().any():
        out["b"] = out["b"].fillna(out["b"].median())
    out["b"] = out["b"] / out["b"].sum() * out["m"].sum()
    return out


def vf_kweep(gdf: gpd.GeoDataFrame, s_star, A: np.ndarray,
             k_values=(0, 1, 5, 10, 20, 50), n_trials: int = 20,
             grid_res: int = 100, seed: int = 7) -> pd.DataFrame:
    """k-sweep on the CA counties for a given S*."""
    m = gdf["m"].astype(float).to_numpy()
    b = gdf["b"].astype(float).to_numpy()
    rows = []
    for k in k_values:
        if k == 0:
            rng = np.random.default_rng(seed)
            coords, mpp, bpp = geom_k_points_real(gdf, m, b, 0, rng)
            disc = discover_rect_real(coords, mpp, bpp, grid_res=grid_res)
            pjd = point_jaccard_real(s_star, disc, A)
            _, _, smr = internal_smr(disc, coords, mpp, bpp)
            cx = (disc.bounds[0] + disc.bounds[2]) / 2
            cy = (disc.bounds[1] + disc.bounds[3]) / 2
            rows.append({"k": k, "mean": pjd, "std": 0.0, "n": 1,
                         "smr_last": smr,
                         "cx_last": cx, "cy_last": cy,
                         "rect": disc.bounds})
        else:
            jds, smrs, cxs, cys = [], [], [], []
            last_rect = None
            for t in range(n_trials):
                rng = np.random.default_rng(seed + 1000 * k + t)
                coords, mpp, bpp = geom_k_points_real(gdf, m, b, k, rng)
                disc = discover_rect_real(coords, mpp, bpp, grid_res=grid_res)
                jds.append(point_jaccard_real(s_star, disc, A))
                _, _, smr = internal_smr(disc, coords, mpp, bpp)
                smrs.append(smr)
                cxs.append((disc.bounds[0] + disc.bounds[2]) / 2)
                cys.append((disc.bounds[1] + disc.bounds[3]) / 2)
                last_rect = disc
            rows.append({"k": k, "mean": float(np.mean(jds)),
                         "std": float(np.std(jds)), "n": n_trials,
                         "smr_last": float(np.mean(smrs)),
                         "cx_last": float(np.mean(cxs)),
                         "cy_last": float(np.mean(cys)),
                         "rect": last_rect.bounds})
    return pd.DataFrame(rows)


def vf_pjd_png(df: pd.DataFrame, floor: float, out_path: Path,
                title: str = "Valley Fever — PJD vs k (SJV-8, 2014-2018)"):
    pp.apply_style_v9()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    k = df["k"].to_numpy().astype(float)
    x = np.where(k == 0, 0.5, k)
    mu = df["mean"].to_numpy(); sd = df["std"].to_numpy()
    ax.fill_between(x, mu - sd, mu + sd, color="darkmagenta", alpha=0.18, lw=0)
    ax.plot(x, mu, color="darkmagenta", lw=2.0, marker="s", ms=6, label="Geom-k")
    ax.axhline(floor, color="black", lw=1.2, ls=(0, (5, 2)),
               label=f"shape floor JD={floor:.3f}")
    ax.set_xscale("log"); ax.set_xticks(x)
    ax.set_xticklabels(["0" if v == 0 else str(int(v)) for v in k])
    ax.set_xlabel("k (samples per region; 0 = Centroid)")
    ax.set_ylabel("Point Jaccard Distance")
    ax.set_title(title); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def vf_choropleth_png(gdf: gpd.GeoDataFrame, s_star,
                       rect_k0: Polygon, rect_k50: Polygon,
                       out_path: Path,
                       title: str = "California Valley Fever 2014-2018"):
    fig, ax = plt.subplots(figsize=(8, 9))
    # SMR per county (m / b after rescaling); cap for the colour scale
    smr = (gdf["m"] / gdf["b"].replace(0, np.nan)).clip(upper=15.0)
    g = gdf.copy(); g["SMR"] = smr
    g.plot(column="SMR", ax=ax, cmap="OrRd",
           edgecolor="#888", linewidth=0.4,
           legend=True,
           legend_kwds={"label": "SMR (cases / expected)", "shrink": 0.55})
    # SJV outline
    polys = list(s_star.geoms) if hasattr(s_star, "geoms") else [s_star]
    first = True
    for p in polys:
        ax.plot(*p.exterior.xy, color="#1B5E20", lw=2.2, ls="--",
                label="SJV S*" if first else None)
        first = False

    def _add(rect, color, label):
        minx, miny, maxx, maxy = rect.bounds
        ax.add_patch(mpatches.Rectangle((minx, miny), maxx - minx,
                                         maxy - miny, facecolor="none",
                                         edgecolor=color, lw=2.0, label=label))
    _add(rect_k0, "red", "Centroid (k=0) rect")
    _add(rect_k50, "darkmagenta", "Geom-50 rect")
    ax.set_aspect("equal")
    ax.set_xlabel("easting (m, EPSG:3310)")
    ax.set_ylabel("northing (m, EPSG:3310)")
    ax.set_title(title)
    h, l = ax.get_legend_handles_labels()
    seen = set(); hh, ll = [], []
    for a, b in zip(h, l):
        if b in seen: continue
        seen.add(b); hh.append(a); ll.append(b)
    ax.legend(hh, ll, loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def valley_fever_run():
    OUT = OUTPUTS / "valley_fever"
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("Valley Fever (Coccidioidomycosis) — full sweep on CA counties")
    print("=" * 78)

    # --- CA counties (3310) -------------------------------------------------
    gdf_geom = load_california_counties()
    print(f"  n counties from shapefile: {len(gdf_geom)}")

    # --- Headline window 2014-2018, SJV-8 ----------------------------------
    cases, pop = load_cocci_window(2014, 2018)
    print(f"  cocci cases 2014-2018: {int(cases.sum())} across "
          f"{len(cases)} CHHS county rows")

    # R3 — hygiene: which CHHS counties are missing from the shapefile?
    print("\n--- R3 — data hygiene ---")
    shp_names = set(gdf_geom["NAME"])
    csv_names = set(cases.index)
    missing = sorted(csv_names - shp_names)
    print(f"  CHHS counties NOT in shapefile ({len(missing)}): {missing}")
    for n in missing:
        c = int(cases.get(n, 0))
        rate = (c / pop.get(n, 1.0)) * 100000 if pop.get(n, 0) else 0
        in_sjv = n in SJV_8
        print(f"    {n}: cases={c}, ~rate/100k(yr-avg)={rate:.1f}, in_SJV={in_sjv}")

    gdf = attach_cocci_to_counties(gdf_geom, cases, pop)
    # Reference set A built once for the CA county geometry
    A = reference_set(gdf, n_per_region=500, seed=42)
    print(f"  reference set |A| = {len(A)}")

    # SJV-8 S*
    sjv = _vf_pairwise_union([gdf[gdf["NAME"] == n].geometry.iloc[0]
                              for n in SJV_8 if n in shp_names])
    floor_sjv = shape_floor_rect(sjv, A, n_grid=14)
    print(f"  SJV-8 area = {sjv.area / 1e6:.0f} km²; "
          f"shape_floor_rect JD = {floor_sjv['jd']:.4f}")

    # --- HEADLINE SWEEP -----------------------------------------------------
    print("\n" + "=" * 78)
    print("HEADLINE  S* = SJV-8, window 2014-2018, k=0..50, 20 trials")
    print("=" * 78)
    headline = vf_kweep(gdf, sjv, A,
                        k_values=(0, 1, 5, 10, 20, 50),
                        n_trials=20, grid_res=100, seed=7)
    headline["pjd_minus_floor"] = headline["mean"] - floor_sjv["jd"]
    headline["centre_in_sjv"] = headline.apply(
        lambda r: sjv.contains(Point(r["cx_last"], r["cy_last"])), axis=1)
    print(headline[["k", "mean", "std", "smr_last", "centre_in_sjv",
                    "pjd_minus_floor", "n"]].to_string(index=False))
    pjd0 = float(headline.loc[headline["k"] == 0, "mean"].iloc[0])
    pjd50 = float(headline.loc[headline["k"] == 50, "mean"].iloc[0])
    gap = pjd0 - pjd50
    means = headline["mean"].to_numpy()
    mono = bool(np.all(np.diff(means) <= 0))
    print(f"\n  gap = PJD(k=0) - PJD(k=50)        = {gap:+.4f}")
    print(f"  shape_floor JD                    = {floor_sjv['jd']:.4f}")
    print(f"  monotone decreasing k=0..50?      = {mono}")

    # Figures
    rect_k0 = Polygon([(headline.loc[headline['k']==0,'rect'].iloc[0][0],
                        headline.loc[headline['k']==0,'rect'].iloc[0][1]),
                       (headline.loc[headline['k']==0,'rect'].iloc[0][2],
                        headline.loc[headline['k']==0,'rect'].iloc[0][1]),
                       (headline.loc[headline['k']==0,'rect'].iloc[0][2],
                        headline.loc[headline['k']==0,'rect'].iloc[0][3]),
                       (headline.loc[headline['k']==0,'rect'].iloc[0][0],
                        headline.loc[headline['k']==0,'rect'].iloc[0][3])])
    rect_k50 = Polygon([(headline.loc[headline['k']==50,'rect'].iloc[0][0],
                         headline.loc[headline['k']==50,'rect'].iloc[0][1]),
                        (headline.loc[headline['k']==50,'rect'].iloc[0][2],
                         headline.loc[headline['k']==50,'rect'].iloc[0][1]),
                        (headline.loc[headline['k']==50,'rect'].iloc[0][2],
                         headline.loc[headline['k']==50,'rect'].iloc[0][3]),
                        (headline.loc[headline['k']==50,'rect'].iloc[0][0],
                         headline.loc[headline['k']==50,'rect'].iloc[0][3])])
    png_pjd = OUT / "vf_pjd_vs_k.png"
    vf_pjd_png(headline, floor_sjv["jd"], png_pjd)
    print(f"\n  wrote {png_pjd}")
    png_map = OUT / "vf_smr_choropleth.png"
    vf_choropleth_png(gdf, sjv, rect_k0, rect_k50, png_map,
                      title="Valley Fever 2014-2018 — SMR + SJV S* + "
                            "Centroid (red) vs Geom-50 (magenta) rect")
    print(f"  wrote {png_map}")

    # --- R1 — S* definition ---------------------------------------------
    print("\n" + "=" * 78)
    print("R1 — S* definition robustness")
    print("=" * 78)
    for label, names in (("SJV-5 (hyperendemic core)", SJV_5_CORE),
                          ("SJV-8 (headline, repeated)", SJV_8),
                          ("CDPH-recognized endemic",   CDPH_ENDEMIC)):
        polys = [gdf[gdf["NAME"] == n].geometry.iloc[0]
                 for n in names if n in shp_names]
        s = _vf_pairwise_union(polys)
        floor = shape_floor_rect(s, A, n_grid=14)["jd"]
        # k=0 and k=50 only
        df_small = vf_kweep(gdf, s, A, k_values=(0, 50),
                             n_trials=20, grid_res=100, seed=7)
        pjd0_ = float(df_small.loc[df_small["k"] == 0, "mean"].iloc[0])
        pjd50_ = float(df_small.loc[df_small["k"] == 50, "mean"].iloc[0])
        smr50_ = float(df_small.loc[df_small["k"] == 50, "smr_last"].iloc[0])
        in_sjv50 = bool(s.contains(Point(
            float(df_small.loc[df_small["k"] == 50, "cx_last"].iloc[0]),
            float(df_small.loc[df_small["k"] == 50, "cy_last"].iloc[0]))))
        print(f"  {label:35s} | "
              f"floor={floor:.3f} | "
              f"k=0  PJD={pjd0_:.3f} | "
              f"k=50 PJD={pjd50_:.3f} | "
              f"gap={pjd0_-pjd50_:+.3f} | "
              f"SMR={smr50_:.1f} | in_S*?={in_sjv50}")

    # --- R2 — year-window robustness ------------------------------------
    print("\n" + "=" * 78)
    print("R2 — year-window robustness (SJV-8)")
    print("=" * 78)
    for win in [(2011, 2016), (2017, 2017), (2014, 2018)]:
        c2, p2 = load_cocci_window(*win)
        g2 = attach_cocci_to_counties(gdf_geom, c2, p2)
        df_small = vf_kweep(g2, sjv, A, k_values=(0, 50),
                             n_trials=20, grid_res=100, seed=7)
        pjd0_ = float(df_small.loc[df_small["k"] == 0, "mean"].iloc[0])
        pjd50_ = float(df_small.loc[df_small["k"] == 50, "mean"].iloc[0])
        smr50_ = float(df_small.loc[df_small["k"] == 50, "smr_last"].iloc[0])
        in_sjv50 = bool(sjv.contains(Point(
            float(df_small.loc[df_small["k"] == 50, "cx_last"].iloc[0]),
            float(df_small.loc[df_small["k"] == 50, "cy_last"].iloc[0]))))
        total = int(c2.sum())
        win_label = f"{win[0]}-{win[1]}" if win[0] != win[1] else str(win[0])
        print(f"  window {win_label:9s} (n cases={total:6d}) | "
              f"k=0  PJD={pjd0_:.3f} | "
              f"k=50 PJD={pjd50_:.3f} | "
              f"gap={pjd0_-pjd50_:+.3f} | "
              f"SMR={smr50_:.1f} | in_SJV?={in_sjv50}")


if __name__ == "__main__":
    import sys as _sys
    if "--diagnose" in _sys.argv:
        diagnose()
    elif "--cholera" in _sys.argv:
        cholera_run()
    elif "--valley" in _sys.argv:
        valley_fever_run()
    else:
        main()
