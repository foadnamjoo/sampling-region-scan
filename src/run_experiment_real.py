"""Real-data pipeline: California Valley Fever (paper Section 6, Figure 13).

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
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point, Polygon, shape
import fiona

# Repo paths
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from _paths import DATA, OUTPUTS  # noqa: E402

# pyScan via the user's local build dir (set PYSCAN_BUILD env var)
_pyscan_build = os.environ.get("PYSCAN_BUILD")
if _pyscan_build:
    sys.path.insert(0, _pyscan_build)
    os.chdir(_pyscan_build)
import pyscan  # noqa: E402

# Reuse existing helpers — no copies
from run_experiment import sample_points_in_polygon  # noqa: E402
from shape_floor import reference_set  # noqa: E402
import paper_plots as pp  # noqa: E402


# ===========================================================================
# Shared helpers (region -> points, scan, Point Jaccard, shape floor)
# ===========================================================================

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


# Valley Fever — Coccidioidomycosis in California counties
# ===========================================================================
# Real fixed regions = CA's 58 counties (loader drops 2 not in the shapefile;
# R3 confirms they're non-SJV and low-incidence). Real m(z) = case counts from
# CHHS open-data; real b(z) = mean county population over the window.
# Everything in EPSG:3310 (CA Albers, metres).

# The CA county shapefile is the same one run_experiment.py uses for the
# synthetic California experiment. Expected at data/california/cnty19_1.shp
# per data/README.md; download from the source listed there if missing.
CA_SHP_VF = DATA / "california" / "cnty19_1.shp"
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
    valley_fever_run()
