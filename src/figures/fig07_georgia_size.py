
# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------
"""Render Figs 6, 7, 8, 9 with the polished academic style used for NYC/Utah/CA:
  - Title without parenthetical trial count
  - Math-typeset $p - q$ difference x-axis (or area %)
  - Map inset in top-right corner, NE-anchored, small margin
  - Frameless horizontal legend below the x-axis (markers preserved so the
    legend is colorblind-safe)
  - Paper style v9 curves with shaded ±std bands

Figures produced (in buchin_attempt/, NOT touching the paper repo):
  fig6_usa_grid100_allmethods_inset.{pdf,png}
  fig7_georgia_size_sweep_allmethods_inset.{pdf,png}
  fig8_arkansas_30_allmethods_inset.{pdf,png}
  fig9_arkansas_10_allmethods_inset.{pdf,png}
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

import paper_plots as pp

CACHED = OUTPUTS / "cached_data"
OUT = OUTPUTS

# ---------------------------------------------------------------------------
# Inset helper (mirrors render_state_grid100_with_inset)
# ---------------------------------------------------------------------------

def _draw_inset(ax, shp_path, target_polys, xticks, yticks,
                bbox=None, inset_pos=None, inset_anchor="NE",
                transparent_bg=False):
    """Draw a state map with one or more target rectangles in the top-right
    corner of the host axes.

    `bbox` (minx, miny, maxx, maxy): optional centroid-bbox filter to drop
    polygons whose centroid lies outside the box. Used to crop the USA
    shapefile to the mainland (drops AK / HI / territories).
    `inset_pos`: optional [x0, y0, w, h] axes-fraction override for the
    inset box. Default is the standard top-right corner; use this to make
    a state-specific inset larger or smaller."""
    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        keep = gdf.geometry.centroid.apply(
            lambda c: minx <= c.x <= maxx and miny <= c.y <= maxy)
        gdf = gdf[keep].reset_index(drop=True)
    inset = ax.inset_axes(inset_pos if inset_pos else [0.66, 0.45, 0.33, 0.53])
    gdf.plot(ax=inset, color="white", edgecolor="#9A9A9A", linewidth=0.45,
             rasterized=True)

    target_edge = "#C0392B"
    target_face = (0.78, 0.16, 0.16, 0.10)
    if not isinstance(target_polys, list):
        target_polys = [target_polys]
    for poly in target_polys:
        xy = list(poly.exterior.coords)
        inset.add_patch(mpatches.Polygon(xy, closed=True,
            edgecolor=target_edge, facecolor=target_face,
            lw=1.2, ls=(0, (4, 2))))

    # Tight bounds with just a hair of breathing room so the outermost
    # county borders aren't pressed against the inset frame.
    xmin, ymin, xmax, ymax = gdf.total_bounds
    xpad = (xmax - xmin) * 0.015
    ypad = (ymax - ymin) * 0.025
    inset.set_xlim(xmin - xpad, xmax + xpad)
    inset.set_ylim(ymin - ypad, ymax + ypad)
    inset.margins(0, 0)
    inset.set_xticks(xticks)
    inset.set_yticks(yticks)
    inset.tick_params(axis="both", which="major", labelsize=9,
                       length=2.5, width=0.5, pad=1.5, colors="#444444")
    inset.set_facecolor("white")
    for sp in inset.spines.values():
        sp.set_edgecolor("#BFBFBF"); sp.set_linewidth(0.6)
    inset.set_aspect("equal")
    inset.set_anchor(inset_anchor)


# ---------------------------------------------------------------------------
# Shared chrome (title, axes labels, frameless legend below the axes)
# ---------------------------------------------------------------------------

def _finish_chrome(fig, ax, title, xlabel, ncol):
    ax.set_title(title, fontsize=17, pad=10, weight="medium")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Point Jaccard distance", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    # Shorten the last x-tick "0.70" → "0.7" and right-align it.
    fig.canvas.draw()
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    if xticklabels and xticklabels[-1] in ("0.70", "0.7000"):
        xticklabels[-1] = "0.7"
        ax.set_xticklabels(xticklabels)
    last = ax.get_xticklabels()[-1]
    last.set_horizontalalignment("right")
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    # One row only. Shrink font for >5 entries so nothing overlaps.
    n = len(labels)
    legend_fontsize = 13.5 if n <= 5 else 11.5
    handle_len = 1.8 if n <= 5 else 1.4
    handle_pad = 0.45 if n <= 5 else 0.35
    ax.legend(handles, labels,
              loc="upper left",
              bbox_to_anchor=(0.0, -0.22, 1.0, 0.10),
              mode="expand", ncol=n, fontsize=legend_fontsize,
              frameon=False,
              handlelength=handle_len, handletextpad=handle_pad,
              columnspacing=0.0)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.93, bottom=0.22)


# ---------------------------------------------------------------------------
# Fig 6, Fig 8, Fig 9 — standard methods curves (with FlexScan for Arkansas)
# ---------------------------------------------------------------------------

USA_SHP = DATA / "usa/cb_2018_us_county_within_cd116_500k.shp"
ARK_SHP = DATA / "arkansas/COUNTY_BOUNDARY.shp"

USA_TARGET = Polygon([(-100, 33), (-100, 40), (-90, 40), (-90, 33)])
ARK30_TARGET = Polygon([(-93.5, 34), (-93.5, 35.5), (-91.5, 35.5), (-91.5, 34)])
ARK10_TARGET = Polygon([(-92.85, 34.40), (-92.85, 35.10),
                        (-92.15, 35.10), (-92.15, 34.40)])


def render_methods_with_inset(pkl_name, title, shp, target, xticks, yticks,
                               out_base, ncol, bbox=None, inset_pos=None,
                               inset_anchor="NE", transparent_bg=False):
    pkg = pickle.load(open(CACHED / pkl_name, "rb"))
    pq_diff = np.asarray(pkg["pq_diff"])
    data = {m: np.asarray(arr) for m, arr in pkg["methods"].items()}

    pp.apply_style_v9()
    # Reskin FlexScan so it reads as the baseline competitor rather than
    # a hero method: muted dark-red dashed line with a small down-triangle.
    pp.METHOD_STYLE["FlexScan"] = {
        "color":  "#7A3030",       # muted dark red
        "ls":     (0, (5, 2)),     # dashed
        "marker": "v",             # down-triangle
        "ms":     5.5,
        "lw":     1.3,
    }
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    pp.plot_jaccard_vs_pq(ax, data, pq_diff,
                          show_band=True, band_alpha=0.22)
    _finish_chrome(fig, ax, title, r"$p - q$ difference", ncol=ncol)
    _draw_inset(ax, shp, target, xticks, yticks, bbox=bbox,
                inset_pos=inset_pos, inset_anchor=inset_anchor,
                transparent_bg=transparent_bg)

    for ext in ("png", "pdf"):
        path = OUT / f"{out_base}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def render_fig6_usa():
    # Filter to mainland — drops Alaska, Hawaii, territories so the inset
    # actually shows the continental USA the target rectangle lives in.
    render_methods_with_inset(
        pkl_name="usa.pkl",
        title="USA Counties",
        shp=USA_SHP,
        target=USA_TARGET,
        xticks=[-120, -100, -80],
        yticks=[30, 40],
        out_base="fig6_usa_grid100_allmethods_inset",
        ncol=5,
        bbox=(-126, 24, -65, 50),
        # Big inset, but keep the same small right + top margins from the
        # original ([0.40, 0.45, 0.59, 0.53] → right edge 0.99, top 0.98).
        inset_pos=[0.25, 0.36, 0.74, 0.62],
    )


def render_fig8_arkansas_30():
    render_methods_with_inset(
        pkl_name="arkansas_30.pkl",
        title="Arkansas 30 % target rectangle",
        shp=ARK_SHP,
        target=ARK30_TARGET,
        xticks=[-94, -92, -90],
        yticks=[33.5, 35, 36.5],
        out_base="fig8_arkansas_30_allmethods_inset",
        ncol=6,                                 # adds FlexScan
        # Slightly smaller inset, same 1% right/top margins.
        inset_pos=[0.61, 0.34, 0.38, 0.65],
    )


def render_fig9_arkansas_10():
    render_methods_with_inset(
        pkl_name="arkansas_10.pkl",
        title="Arkansas 10 % target rectangle",
        shp=ARK_SHP,
        target=ARK10_TARGET,
        xticks=[],                              # no lat/lon labels
        yticks=[],
        out_base="fig9_arkansas_10_allmethods_inset",
        ncol=6,                                 # adds FlexScan
        # 10% target — small inset in the lower-LEFT, no ticks.
        inset_pos=[0.015, 0.02, 0.31, 0.40],
        inset_anchor="SW",
    )


# ---------------------------------------------------------------------------
# Fig 7 — Georgia size sweep (PJD vs target rectangle area %)
# ---------------------------------------------------------------------------

GEORGIA_SHP = DATA / "georgia/GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp"
# 10 nested target rectangles, all anchored at (-85, 31), upper-right swept
# from (-84.5, 31.5) to (-82.0, 34.0) — verbatim from run_experiment.py:477
_GA_X_BASE, _GA_Y_BASE = -85.0, 31.0
_GA_X_ARRAY = np.linspace(-84.5, -82.0, 10)
_GA_Y_ARRAY = np.linspace(31.5, 34.0, 10)
GEORGIA_TARGETS = [Polygon([(_GA_X_BASE, _GA_Y_BASE),
                            (_GA_X_BASE, _GA_Y_ARRAY[t]),
                            (_GA_X_ARRAY[t], _GA_Y_ARRAY[t]),
                            (_GA_X_ARRAY[t], _GA_Y_BASE)])
                   for t in range(10)]


def render_fig7_georgia_size_sweep():
    # Prefer the higher-trial-count re-run if it exists (smoother Geom-50);
    # fall back to the original 20-trial pkl otherwise.
    high_trial_pkl = OUT / "georgia_size_sweep_grid100_t80.pkl"
    pkg = pickle.load(open(high_trial_pkl if high_trial_pkl.exists()
                            else CACHED / "georgia_size_sweep.pkl", "rb"))
    area_pct = np.asarray(pkg["area_pct"])
    # Per-method: array shape (n_targets, n_trials). For plot_jaccard_vs_pq's
    # contract we want (n_trials, n_x_points), so transpose.
    data = {m: np.asarray(arr).T for m, arr in pkg["methods"].items()}

    pp.apply_style_v9()
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    pp.plot_jaccard_vs_pq(ax, data, area_pct,
                          show_band=True, band_alpha=0.22)
    # Different x-axis label — area %, not p-q difference
    _finish_chrome(fig, ax, "Georgia: ten target rectangles",
                   r"Target rectangle area (% of state)", ncol=5)
    # Tight x-limits ending exactly at the largest area %; sparser tick
    # set so labels don't crowd. Labels formatted to 1 decimal.
    ax.set_xlim(area_pct.min() * 0.9, area_pct.max())
    tick_idx = [0, 2, 4, 6, 7, 8, 9]   # skip dense early ticks
    xticks = area_pct[tick_idx]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:.1f}" for v in xticks])
    # Bigger Georgia inset, same right/top margins as the original.
    _draw_inset(ax, GEORGIA_SHP, GEORGIA_TARGETS,
                xticks=[-85, -83, -81], yticks=[31, 33, 35],
                inset_pos=[0.45, 0.36, 0.54, 0.62])

    for ext in ("png", "pdf"):
        path = OUT / f"fig7_georgia_size_sweep_allmethods_inset.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    render_fig6_usa()
    render_fig7_georgia_size_sweep()
    render_fig8_arkansas_30()
    render_fig9_arkansas_10()


if __name__ == "__main__":
    main()
