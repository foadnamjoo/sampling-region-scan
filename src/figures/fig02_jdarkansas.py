from __future__ import annotations

"""Fig 2: JDArkansas — 4-panel illustration of (target, discovered) rectangles
for varying pq differences on Arkansas counties using Geom 5.  Seeded for
reproducibility (seed=7), matching the rerun for every other figure.

Mirrors `main_test(5, pq_diff)` from McClelland_22 cell 100, then renders the
four panels in v9 paper style.
"""

# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------

import os
import pickle
import random
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

SHP = DATA / "arkansas" / "COUNTY_BOUNDARY.shp"

_pyscan_build = os.environ.get("PYSCAN_BUILD")
if _pyscan_build:
    sys.path.insert(0, _pyscan_build)
    os.chdir(_pyscan_build)
import pyscan  # noqa: E402

import paper_plots as pp  # noqa: E402

SEED = 35
PQ_DIFFS = [0.1, 0.2, 0.4, 0.6]
Q = 0.2
TARGET = Polygon([(-93.5, 34), (-93.5, 35.5), (-91.5, 35.5), (-91.5, 34)])
K = 5
GRID_RES = 40

OUT_DIR = OUTPUTS


def sample_points_in_polygon(poly, k: int, rng: np.random.Generator) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    out = np.empty((k, 2)); n = 0
    while n < k:
        bx = rng.uniform(minx, maxx, size=k * 3)
        by = rng.uniform(miny, maxy, size=k * 3)
        for x, y in zip(bx, by):
            if n == k: break
            if poly.contains(Point(x, y)):
                out[n] = (x, y); n += 1
    return out


def one_trial(gdf, target: Polygon, pq_diff: float, rng: np.random.Generator):
    """Returns (pts, discovered_rect, jd) for a single Geom-5 trial at pq_diff."""
    pts = np.vstack([sample_points_in_polygon(g, K, rng) for g in gdf.geometry])
    p_prob = Q + pq_diff
    inside = np.array([target.contains(Point(x, y)) for x, y in pts])
    coins = rng.random(len(pts))
    baseline, measured = [], []
    for i, (x, y) in enumerate(pts):
        baseline.append(pyscan.WPoint(1.0, float(x), float(y), 1.0))
        thresh = p_prob if inside[i] else Q
        if coins[i] <= thresh:
            measured.append(pyscan.WPoint(1.0, float(x), float(y), 1.0))
    grid = pyscan.Grid(GRID_RES, measured, baseline)
    subgrid = pyscan.max_subgrid(grid, pyscan.KULLDORF)
    rect = grid.toRectangle(subgrid)
    discovered = Polygon([
        (rect.lowX(), rect.lowY()), (rect.lowX(), rect.upY()),
        (rect.upX(), rect.upY()), (rect.upX(), rect.lowY())
    ])
    a_u_b = a_n_b = 0
    for i, (x, y) in enumerate(pts):
        if coins[i] > (p_prob if inside[i] else Q):
            continue
        in_t = inside[i]
        in_d = discovered.contains(Point(float(x), float(y)))
        if in_t or in_d: a_u_b += 1
        if in_t and in_d: a_n_b += 1
    jd = ((a_u_b - a_n_b) / a_u_b) if a_u_b > 0 else 1.0
    return pts, discovered, float(jd)


def main():
    random.seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    gdf = gpd.read_file(SHP).to_crs("EPSG:4326")

    trials = []
    for pq in PQ_DIFFS:
        pts, disc, jd = one_trial(gdf, TARGET, pq, rng)
        trials.append({"pq": pq, "pts": pts, "discovered": disc, "jd": jd})
        print(f"pq={pq:.2f}: JD={jd:.3f}, discovered=({disc.bounds[0]:.2f},{disc.bounds[1]:.2f})-"
              f"({disc.bounds[2]:.2f},{disc.bounds[3]:.2f})", flush=True)

    # 1x4 row layout — all four trials side-by-side, full text width.
    pp.apply_style_v9()
    # Override v9's sans-serif default with a serif family so figure text
    # matches the paper's body font (acmart uses Linux Libertine).
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["Linux Libertine", "Linux Libertine O",
                        "Liberation Serif", "DejaVu Serif"],
    })
    # 6-panel row: 4 rectangle-scan trials + 2 shape-floor panels.
    #   panel 5: best DISK for planted rectangle  (rectangle-target floor)
    #   panel 6: best RECTANGLE for planted disk  (disk-target floor)
    from make_shape_floor import (reference_set as _ref_set,
                                  in_rect as _in_rect,
                                  in_disk as _in_disk,
                                  best_disk_for_mask as _best_disk,
                                  best_rect_for_disk as _best_rect)
    print("computing shape-floor A: best disk for rect target ...")
    _A = _ref_set(gdf, 500, 42)
    _rect_mask = _in_rect(_A, -92.5, 34.75, 1.0, 0.75)
    _floor_rect = _best_disk(_A, _rect_mask, center_hint=(-92.5, 34.75))
    print(f"  best disk: center=({_floor_rect['cx']:.3f}, {_floor_rect['cy']:.3f}) "
          f"r={_floor_rect['r']:.3f}  JD={_floor_rect['jd']:.3f}")
    print("computing shape-floor B: best rectangle for disk target ...")
    _DISK_R = float(np.sqrt(2.0 * 1.5 / np.pi))   # equal area to 2.0 x 1.5 rect
    _DISK_CENTER = (-92.5, 34.75)
    _disk_mask = _in_disk(_A, *_DISK_CENTER, _DISK_R)
    _floor_disk = _best_rect(_A, _disk_mask)
    print(f"  best rect: center=({_floor_disk['cx']:.3f}, {_floor_disk['cy']:.3f}) "
          f"hw={_floor_disk['hw']:.3f} hh={_floor_disk['hh']:.3f}  JD={_floor_disk['jd']:.3f}")

    tx0, ty0, tx1, ty1 = TARGET.bounds
    target_color    = "#C62828"   # warm red
    discovered_color = "#1B5E20"  # dark forest green
    xmin, ymin, xmax, ymax = gdf.total_bounds

    # Subsample the 500-pts/region reference pool used to compute the
    # shape floor so the point density matches the first four panels.
    _rng = np.random.default_rng(0)
    _idx = _rng.choice(len(_A), size=min(len(_A), len(trials[0]["pts"])),
                        replace=False)
    _sub = _A[_idx]

    from matplotlib.legend_handler import HandlerPatch

    class HandlerCircle(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            # Size matches the prior Line2D marker (markersize=32) — do not
            # shrink to fit the default legend handle box; let the circle
            # overflow it the same way the Line2D marker did.
            radius = 12.8  # 80% of previous 16.0
            cx = width / 2 - xdescent
            cy = height / 2 - ydescent
            p = mpatches.Circle((cx, cy), radius=radius,
                                facecolor=orig_handle.get_facecolor(),
                                edgecolor=orig_handle.get_edgecolor(),
                                linewidth=orig_handle.get_linewidth(),
                                linestyle=orig_handle.get_linestyle())
            p.set_transform(trans)
            return [p]

    def _draw_trial(ax, t):
        ax.set_facecolor("white")
        for sp in ax.spines.values(): sp.set_visible(False)
        gdf.plot(ax=ax, color="#F4F4F4", edgecolor="#9C9C9C", linewidth=0.5,
                 rasterized=True)
        ax.scatter(t["pts"][:, 0], t["pts"][:, 1],
                   s=4.0, color="#1F1F1F", alpha=0.6, linewidths=0)
        ax.add_patch(mpatches.Rectangle(
            (tx0, ty0), tx1 - tx0, ty1 - ty0,
            edgecolor=target_color, facecolor=(0.78, 0.16, 0.16, 0.18),
            lw=2.0, ls=(0, (5, 2)),
        ))
        dx0, dy0, dx1, dy1 = t["discovered"].bounds
        ax.add_patch(mpatches.Rectangle(
            (dx0, dy0), dx1 - dx0, dy1 - dy0,
            edgecolor=discovered_color, facecolor="none", lw=2.4, ls="-",
        ))
        ax.set_title(f"JD $=$ {t['jd']:.2f}",
                     fontsize=32, fontweight="normal", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(xmin - 0.02, xmax + 0.02)
        ax.set_ylim(ymin - 0.02, ymax + 0.02)
        ax.margins(0, 0)

    def _floor_basemap(ax):
        ax.set_facecolor("white")
        for sp in ax.spines.values(): sp.set_visible(False)
        gdf.plot(ax=ax, color="#F4F4F4", edgecolor="#9C9C9C", linewidth=0.5,
                 rasterized=True)
        ax.scatter(_sub[:, 0], _sub[:, 1],
                   s=4.0, color="#1F1F1F", alpha=0.6, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(xmin - 0.02, xmax + 0.02)
        ax.set_ylim(ymin - 0.02, ymax + 0.02)
        ax.margins(0, 0)

    def _draw_floor_disk_for_rect(ax):
        """5th panel: planted rectangle + best-fit disk floor."""
        _floor_basemap(ax)
        ax.add_patch(mpatches.Rectangle(
            (tx0, ty0), tx1 - tx0, ty1 - ty0,
            edgecolor=target_color, facecolor=(0.78, 0.16, 0.16, 0.18),
            lw=2.0, ls=(0, (5, 2)),
        ))
        ax.add_patch(mpatches.Circle((_floor_rect["cx"], _floor_rect["cy"]),
            _floor_rect["r"],
            edgecolor=discovered_color, facecolor="none", lw=2.4, ls="-",
        ))
        ax.set_title(f"JD $=$ {_floor_rect['jd']:.3f}",
                     fontsize=32, fontweight="normal", pad=6)

    def _draw_floor_rect_for_disk(ax):
        """6th panel: planted disk + best-fit rectangle floor."""
        _floor_basemap(ax)
        ax.add_patch(mpatches.Circle(_DISK_CENTER, _DISK_R,
            edgecolor=target_color, facecolor=(0.78, 0.16, 0.16, 0.18),
            lw=2.0, ls=(0, (5, 2)),
        ))
        ax.add_patch(mpatches.Rectangle(
            (_floor_disk["cx"] - _floor_disk["hw"],
             _floor_disk["cy"] - _floor_disk["hh"]),
            2 * _floor_disk["hw"], 2 * _floor_disk["hh"],
            edgecolor=discovered_color, facecolor="none", lw=2.4, ls="-",
        ))
        ax.set_title(f"JD $=$ {_floor_disk['jd']:.3f}",
                     fontsize=32, fontweight="normal", pad=6)

    def _render(n_panels: int, out_name: str):
        """Render either the 4-panel (trials only) or 5-panel
        (trials + disk-target floor) version of the figure.
        The rectangle-target / best-disk floor panel (JD=0.218) is
        intentionally omitted."""
        assert n_panels in (4, 5)
        # Keep per-panel width constant ~3.5 in.
        fig, axes_2d = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4.4))
        axes = axes_2d.flatten()
        fig.patch.set_facecolor("white")

        for ax, t in zip(axes[:4], trials):
            _draw_trial(ax, t)
        if n_panels == 5:
            _draw_floor_rect_for_disk(axes[4])

        target_rect_handle = mpatches.Patch(
            facecolor=(0.78, 0.16, 0.16, 0.18),
            edgecolor=target_color, linestyle="--", linewidth=2.0,
            label="Target rectangle")
        discovered_rect_handle = mpatches.Patch(
            facecolor="none", edgecolor=discovered_color,
            linewidth=2.4, label="Discovered rectangle")
        target_disk_handle = mpatches.Circle((0, 0), 1,
            facecolor=(0.78, 0.16, 0.16, 0.18),
            edgecolor=target_color, linewidth=2.0, linestyle=(0, (5, 2)),
            label="Target disk")

        if n_panels == 4:
            handles = [target_rect_handle, discovered_rect_handle]
        else:
            # Order: target rect, target disk, discovered rect.
            handles = [target_rect_handle, target_disk_handle,
                       discovered_rect_handle]

        # 80% legend handle box (handlelength 2.0->1.6, handleheight 0.7->0.56)
        # to match the 80%-shrunk circle marker.
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=len(handles),
                   fontsize=22, frameon=False, columnspacing=2.0,
                   handlelength=1.6, handleheight=0.56,
                   handler_map={mpatches.Circle: HandlerCircle()})

        fig.subplots_adjust(left=0.005, right=0.995, top=0.91, bottom=0.18,
                            wspace=0.04)
        out = OUT_DIR / out_name
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"wrote {out}")

    _render(4, "JDArkansas.pdf")
    _render(5, "JDArkansas_6.pdf")

    # Persist the data so it can be re-rendered without re-running pyscan.
    with open(OUTPUTS / "cached_data" / "fig2_jdarkansas.pkl", "wb") as f:
        pickle.dump(trials, f)


if __name__ == "__main__":
    main()
