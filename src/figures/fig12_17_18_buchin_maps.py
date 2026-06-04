from __future__ import annotations

"""Render the four 9-bin comparison plots after arkansas_buchin_9bin_sweep finishes.

  1) rect_pq_two_sizes_9bin.{pdf,png}             — 2-panel rect pq curve, Centroid + Buchin (9-bin) + Geom-50 rect
  2) sanity_disk_stress_two_sizes_9bin.{pdf,png}  — 2-panel disk pq curve, Centroid + Buchin (9-bin) + Geom-50 disk
  3) rect_map_v3_9bin.{pdf,png}                   — 2×3 rect map (Centroid / Buchin 9-bin / Geom-50) at p−q=0.50
  4) sanity_arkansas_map_disk_v3_9bin.{pdf,png}   — 2×3 disk map  (Centroid / Buchin 9-bin / Geom-50) at p−q=0.50

All written alongside the existing 5-bin files. Existing PDFs/PNGs untouched.
"""

# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------

import math
import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerPatch
from shapely.geometry import Point, Polygon

SHP  = DATA / "arkansas/COUNTY_BOUNDARY.shp"
IO   = IO

P_GRID  = np.round(np.arange(0.20, 0.95, 0.05), 4)
PQ_DIFF = np.round(P_GRID - 0.20, 4)

CENTROID_COLOR = "red"          # matches v9 Centroid curve
BUCH_COLOR     = "#1F2D5C"      # matches Buchin override in fig 8/9 (navy)
GEOM_COLOR     = "darkmagenta"  # matches v9 Geom 50 curve
BAND_ALPHA     = 0.25

PLANTED_CENTER = (-92.5, 34.75)
P_USED_MAP     = 0.70   # p − q = 0.50
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * math.cos(math.radians(34.75))


# ===========================================================================
# Plot 1 — rect pq curve (mirror rect_pq_two_sizes.pdf with Buchin 9-bin)
# ===========================================================================

RECT_FIGS = [
    {"key": "fig8",
     "panel_label": "Larger cluster (2.0° × 1.5°)",
     "rerun_pkl":   IO / "arkansas_30_rerun.pkl",
     "buchin_csv":  IO / "fig8_buchin_9bin.csv"},
    {"key": "fig9",
     "panel_label": "Smaller cluster (0.7° × 0.7°)",
     "rerun_pkl":   IO / "arkansas_10_rerun.pkl",
     "buchin_csv":  IO / "fig9_buchin_9bin.csv"},
]


def _rect_curves(fig_meta):
    out = {}
    with open(fig_meta["rerun_pkl"], "rb") as f:
        rerun = pickle.load(f)
    df = pd.DataFrame(rerun["records"])
    for tag, src in (("centroid", "Centroid"), ("geom", "Geom 50")):
        sub = df[df["method"] == src]
        means, stds = [], []
        for p in P_GRID:
            cell = sub[np.isclose(sub["p"], p)]["pool_jd"]
            means.append(cell.mean()); stds.append(cell.std())
        out[tag] = {"mean": np.array(means), "std": np.array(stds)}
    bdf = pd.read_csv(fig_meta["buchin_csv"])
    means, stds = [], []
    for p in P_GRID:
        cell = bdf[np.isclose(bdf["p"], p)]["point_jaccard"]
        means.append(cell.mean()); stds.append(cell.std())
    out["buchin"] = {"mean": np.array(means), "std": np.array(stds)}
    return out


def render_rect_pq_two_sizes_9bin():
    curves = {fm["key"]: _rect_curves(fm) for fm in RECT_FIGS}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    for ax, fm in zip(axes, RECT_FIGS):
        c = curves[fm["key"]]
        for tag, color, label, marker in (
                ("centroid", CENTROID_COLOR, "Centroid",                "v"),
                ("buchin",   BUCH_COLOR,     "Buchin rect (9-bin)",     "o"),
                ("geom",     GEOM_COLOR,     "Geom-50 rect",            "o")):
            m, s = c[tag]["mean"], c[tag]["std"]
            ax.fill_between(PQ_DIFF, m - s, m + s, color=color, alpha=BAND_ALPHA, lw=0)
            ax.plot(PQ_DIFF, m, color=color, marker=marker, markersize=5, lw=1.8, label=label)
        ax.set_title(fm["panel_label"], fontsize=11)
        ax.set_xlabel("Signal strength (p − q)")
        ax.set_ylabel("Jaccard distance (lower = better)")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Rectangle cluster recovery — Buchin 9-bin grid", fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUTPUTS / f"rect_pq_two_sizes_9bin.{ext}"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")
    plt.close(fig)

    # Plateau report
    print("\n=== Rect plateau (p − q ≥ 0.5) — 9-bin Buchin ===")
    print(f"{'fig':<32} {'Centroid':>10} {'Buchin (9-bin)':>16} {'Geom-50 rect':>14}")
    for fm in RECT_FIGS:
        c = curves[fm["key"]]
        mask = PQ_DIFF >= 0.5
        print(f"  {fm['panel_label']:<30} "
              f"{c['centroid']['mean'][mask].mean():>10.4f} "
              f"{c['buchin']['mean'][mask].mean():>16.4f} "
              f"{c['geom']['mean'][mask].mean():>14.4f}")


# ===========================================================================
# Plot 2 — disk pq curve (mirror sanity_disk_stress_two_sizes.pdf with Buchin 9-bin)
# ===========================================================================

DISK_PKLS = {
    "large": (IO / "arkansas_disk_large.pkl",                 # geom data (existing)
              IO / "arkansas_disk_large_buchin_9bin.pkl",      # buchin 9-bin (new)
              IO / "arkansas_disk_large_centroid.pkl",         # centroid (existing)
              "Larger cluster (radius 0.6°)",
              0.6),
    "small": (IO / "arkansas_disk_small.pkl",
              IO / "arkansas_disk_small_buchin_9bin.pkl",
              IO / "arkansas_disk_small_centroid.pkl",
              "Smaller cluster (radius 0.4°)",
              0.395),
}


def _disk_curves(geom_pkl, buchin_9bin_pkl, centroid_pkl):
    out = {}
    # Geom — existing pkl
    with open(geom_pkl, "rb") as f:
        gp = pickle.load(f)
    gdf = pd.DataFrame(gp["records"])
    g = gdf[gdf["method"] == "geom"]
    means, stds = [], []
    for p in P_GRID:
        cell = g[np.isclose(g["p"], p)]["pool_jd"]
        means.append(cell.mean()); stds.append(cell.std())
    out["geom"] = {"mean": np.array(means), "std": np.array(stds)}
    # Centroid — existing pkl
    with open(centroid_pkl, "rb") as f:
        cp = pickle.load(f)
    cdf = pd.DataFrame(cp["records"])
    means, stds = [], []
    for p in P_GRID:
        cell = cdf[np.isclose(cdf["p"], p)]["pool_jd"]
        means.append(cell.mean()); stds.append(cell.std())
    out["centroid"] = {"mean": np.array(means), "std": np.array(stds)}
    # Buchin 9-bin — new pkl
    with open(buchin_9bin_pkl, "rb") as f:
        bp = pickle.load(f)
    bdf = pd.DataFrame(bp["records"])
    means, stds = [], []
    for p in P_GRID:
        cell = bdf[np.isclose(bdf["p"], p)]["pool_jd"]
        means.append(cell.mean()); stds.append(cell.std())
    out["buchin"] = {"mean": np.array(means), "std": np.array(stds)}
    return out


def render_disk_two_sizes_9bin():
    plt.rcParams.update({
        "axes.labelsize":  16,
        "axes.titlesize":  17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11.5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    plateau_rows = []
    for ax, (key, (gp, bp, cp, label, planted_r)) in zip(axes, DISK_PKLS.items()):
        curves = _disk_curves(gp, bp, cp)
        for tag, color, lbl, marker in (
                ("centroid", CENTROID_COLOR, "Centroid",               "v"),
                ("buchin",   BUCH_COLOR,     "Buchin disk (9-bin)",    "o"),
                ("geom",     GEOM_COLOR,     "Geom-50 disk",           "o")):
            m, s = curves[tag]["mean"], curves[tag]["std"]
            ax.fill_between(PQ_DIFF, m - s, m + s, color=color, alpha=BAND_ALPHA, lw=0)
            ax.plot(PQ_DIFF, m, color=color, marker=marker, markersize=5, lw=1.8, label=lbl)
        ax.set_title(label)
        ax.set_xlabel("Signal strength (p − q diff)")
        ax.set_ylabel("Jaccard distance (lower = better)")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        mask = PQ_DIFF >= 0.5
        plateau_rows.append((label,
                             curves["centroid"]["mean"][mask].mean(),
                             curves["buchin"]["mean"][mask].mean(),
                             curves["geom"]["mean"][mask].mean()))
    fig.suptitle("Disk cluster recovery — Buchin 9-bin grid", fontsize=18)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = OUTPUTS / f"sanity_disk_stress_two_sizes_9bin.{ext}"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")
    plt.close(fig)
    print("\n=== Disk plateau (p − q ≥ 0.5) — 9-bin Buchin ===")
    print(f"{'fig':<32} {'Centroid':>10} {'Buchin (9-bin)':>16} {'Geom-50 disk':>14}")
    for label, c, b, g in plateau_rows:
        print(f"  {label:<30} {c:>10.4f} {b:>16.4f} {g:>14.4f}")


# ===========================================================================
# Plot 3 — rect map v3 9-bin (2×3 with Centroid column)
# ===========================================================================

def _summarise_rects(rects, planted_WH):
    if not rects: return float("nan"), float("nan")
    pW, pH = planted_WH
    pcx, pcy = PLANTED_CENTER
    p_area = pW * pH
    area_mults = [(W * H) / p_area for _, _, W, H in rects]
    offsets_km = []
    for cx, cy, _, _ in rects:
        dx = (cx - pcx) * KM_PER_DEG_LON
        dy = (cy - pcy) * KM_PER_DEG_LAT
        offsets_km.append(math.hypot(dx, dy))
    return float(np.mean(area_mults)), float(np.mean(offsets_km))


def render_rect_map_v3_9bin():
    gdf = gpd.read_file(SHP).to_crs("EPSG:4326")

    fig_meta = {
        "fig8": {"row_label": "Fig 8 — planted 2.0° × 1.5°", "WH": (2.0, 1.5),
                 "rerun_pkl": IO / "arkansas_30_rerun.pkl",
                 "buchin_csv": IO / "fig8_buchin_9bin.csv",
                 "centroid_pkl": IO / "arkansas_rect_centroid.pkl"},
        "fig9": {"row_label": "Fig 9 — planted 0.7° × 0.7°", "WH": (0.7, 0.7),
                 "rerun_pkl": IO / "arkansas_10_rerun.pkl",
                 "buchin_csv": IO / "fig9_buchin_9bin.csv",
                 "centroid_pkl": IO / "arkansas_rect_centroid.pkl"},
    }

    def centroid_rects(fig_name):
        with open(fig_meta[fig_name]["centroid_pkl"], "rb") as f:
            pkg = pickle.load(f)
        df = pd.DataFrame(pkg["records"])
        sub = df[(df["fig"] == fig_name) & np.isclose(df["p"], P_USED_MAP)]
        out = []
        for _, r in sub.iterrows():
            lx, ly, ux, uy = r["rect_lowX"], r["rect_lowY"], r["rect_upX"], r["rect_upY"]
            out.append(((lx + ux) / 2, (ly + uy) / 2, ux - lx, uy - ly))
        return out

    def geom_rects(fig_name):
        with open(fig_meta[fig_name]["rerun_pkl"], "rb") as f:
            pkg = pickle.load(f)
        df = pd.DataFrame(pkg["records"])
        sub = df[(df["method"] == "Geom 50") & np.isclose(df["p"], P_USED_MAP)]
        return [(((r["rect_lowX"] + r["rect_upX"]) / 2),
                 ((r["rect_lowY"] + r["rect_upY"]) / 2),
                 r["rect_upX"] - r["rect_lowX"],
                 r["rect_upY"] - r["rect_lowY"]) for _, r in sub.iterrows()]

    def buchin_rects_9bin(fig_name):
        df = pd.read_csv(fig_meta[fig_name]["buchin_csv"])
        sub = df[np.isclose(df["p"], P_USED_MAP, atol=1e-4)]
        return [(r["best_cx"], r["best_cy"], r["best_W"], r["best_H"])
                for _, r in sub.iterrows()]

    fetchers = {"centroid": centroid_rects, "buchin": buchin_rects_9bin, "geom": geom_rects}
    # 3 rows x 2 cols — sized to drop into a single \columnwidth slot in
    # a 2-column ACM paper. Rows = methods (Centroid / Buchin / Geom-50),
    # columns = target size (Large / Small). Method name as a rotated row
    # label on the first column; target size as a column header on the
    # first row.
    fig, axes = plt.subplots(3, 2, figsize=(4.6, 6.6))

    col_labels = {"fig8": "Large target",
                  "fig9": "Small target"}
    method_rows = (("centroid", CENTROID_COLOR, "Centroid"),
                   ("buchin",   BUCH_COLOR,     "Buchin"),
                   ("geom",     GEOM_COLOR,     "Geom-50"))

    for r_idx, (tag, color, mlabel) in enumerate(method_rows):
        for c_idx, fig_name in enumerate(("fig8", "fig9")):
            meta = fig_meta[fig_name]
            cx0, cy0 = PLANTED_CENTER; W0, H0 = meta["WH"]
            ax = axes[r_idx][c_idx]
            gdf.boundary.plot(ax=ax, color="#999999", linewidth=0.4,
                              rasterized=True)
            planted = mpatches.Rectangle((cx0 - W0/2, cy0 - H0/2), W0, H0,
                                          facecolor=(0, 0, 0, 0.06), edgecolor="black",
                                          linewidth=2.2, linestyle="--")
            ax.add_patch(planted)
            ax.plot(cx0, cy0, marker="x", color="black", markersize=8, mew=2)
            rects = fetchers[tag](fig_name)
            for cx, cy, W, H in rects:
                ax.add_patch(mpatches.Rectangle((cx - W/2, cy - H/2), W, H,
                                                 facecolor="none", edgecolor=color,
                                                 linewidth=1.0, alpha=0.5))
                ax.plot(cx, cy, marker=".", color=color, markersize=3, alpha=0.6)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            minx, miny, maxx, maxy = gdf.total_bounds
            ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
            ax.margins(0, 0)
            if r_idx == 0:
                ax.set_title(col_labels[fig_name], fontsize=12, pad=6)
            if c_idx == 0:
                ax.text(-0.04, 0.5, mlabel,
                        transform=ax.transAxes, fontsize=12,
                        ha="right", va="center", rotation=90)

    legend = [mpatches.Patch(facecolor=(0, 0, 0, 0.06), edgecolor="black",
                              linestyle="--", linewidth=2.2, label="Planted target"),
              mpatches.Patch(facecolor="none", edgecolor=CENTROID_COLOR, linewidth=1.2,
                              label="Centroid"),
              mpatches.Patch(facecolor="none", edgecolor=BUCH_COLOR, linewidth=1.2,
                              label="Buchin"),
              mpatches.Patch(facecolor="none", edgecolor=GEOM_COLOR, linewidth=1.2,
                              label="Geom-50")]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8.5,
                frameon=False, bbox_to_anchor=(0.5, 0.04),
                handletextpad=0.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.04, right=1.0, top=0.97, bottom=0.05,
                        wspace=0.0, hspace=-0.35)
    for ext in ("png", "pdf"):
        out = OUTPUTS / f"rect_map_v3_9bin.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.0)
        print(f"wrote {out}")
    plt.close(fig)


# ===========================================================================
# Plot 4 — disk map v3 9-bin
# ===========================================================================

def _summarise_disks(disks, planted_r):
    if not disks: return float("nan"), float("nan")
    pcx, pcy = PLANTED_CENTER
    r_mults = [r / planted_r for _, _, r in disks]
    offsets_km = []
    for cx, cy, _ in disks:
        dx = (cx - pcx) * KM_PER_DEG_LON; dy = (cy - pcy) * KM_PER_DEG_LAT
        offsets_km.append(math.hypot(dx, dy))
    return float(np.mean(r_mults)), float(np.mean(offsets_km))


def render_disk_map_v3_9bin():
    gdf = gpd.read_file(SHP).to_crs("EPSG:4326")

    cfg = [
        {"row_label": "Disk-Large (r = 0.6°)",   "planted_r": 0.6,
         "geom_pkl":      IO / "arkansas_disk_large.pkl",
         "buchin_pkl":    IO / "arkansas_disk_large_buchin_9bin.pkl",
         "centroid_pkl":  IO / "arkansas_disk_large_centroid.pkl"},
        {"row_label": "Disk-Small (r = 0.395°)", "planted_r": 0.395,
         "geom_pkl":      IO / "arkansas_disk_small.pkl",
         "buchin_pkl":    IO / "arkansas_disk_small_buchin_9bin.pkl",
         "centroid_pkl":  IO / "arkansas_disk_small_centroid.pkl"},
    ]

    def disks_from(pkl, method_tag):
        with open(pkl, "rb") as f:
            pkg = pickle.load(f)
        df = pd.DataFrame(pkg["records"])
        # For Buchin 9-bin pkl, all records are buchin already. For others, filter by method.
        if "method" in df.columns and method_tag is not None:
            sub = df[(df["method"] == method_tag) & np.isclose(df["p"], P_USED_MAP)]
        else:
            sub = df[np.isclose(df["p"], P_USED_MAP)]
        return [(r["disk_cx"], r["disk_cy"], r["disk_r"]) for _, r in sub.iterrows()]

    # 3 rows x 2 cols — sized to drop into a single \columnwidth slot in
    # a 2-column ACM paper. Rows = methods (Centroid / Buchin / Geom-50),
    # columns = disk size (Large / Small). Method name as a rotated row
    # label on the first column; disk size as a column header on the
    # first row.
    fig, axes = plt.subplots(3, 2, figsize=(4.6, 6.6))
    col_labels = {0: "Large disk", 1: "Small disk"}
    method_rows = (("centroid", CENTROID_COLOR, "Centroid"),
                   ("buchin",   BUCH_COLOR,     "Buchin"),
                   ("geom",     GEOM_COLOR,     "Geom-50"))
    for r_idx, (tag, color, mlabel) in enumerate(method_rows):
        for c_idx, c in enumerate(cfg):
            ax = axes[r_idx][c_idx]
            cx0, cy0 = PLANTED_CENTER
            gdf.boundary.plot(ax=ax, color="#999999", linewidth=0.4,
                              rasterized=True)
            planted = mpatches.Circle((cx0, cy0), c["planted_r"],
                                       facecolor=(0, 0, 0, 0.06), edgecolor="black",
                                       linewidth=2.2, linestyle="--")
            ax.add_patch(planted)
            ax.plot(cx0, cy0, marker="x", color="black", markersize=8, mew=2)
            if tag == "centroid":
                disks = disks_from(c["centroid_pkl"], "centroid")
            elif tag == "buchin":
                disks = disks_from(c["buchin_pkl"], None)
            else:
                disks = disks_from(c["geom_pkl"], "geom")
            for cx, cy, r in disks:
                ax.add_patch(mpatches.Circle((cx, cy), r, facecolor="none",
                                              edgecolor=color, linewidth=1.0, alpha=0.5))
                ax.plot(cx, cy, marker=".", color=color, markersize=3, alpha=0.6)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            minx, miny, maxx, maxy = gdf.total_bounds
            ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
            ax.margins(0, 0)
            if r_idx == 0:
                ax.set_title(col_labels[c_idx], fontsize=12, pad=6)
            if c_idx == 0:
                ax.text(-0.04, 0.5, mlabel,
                        transform=ax.transAxes, fontsize=12,
                        ha="right", va="center", rotation=90)

    # Circle legend handles with dashed border on "Planted disk" matching
    # the panels; custom HandlerPatch so the marker is a true circle.
    class HandlerCircle(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            radius = min(width, height) * 0.68
            cx = width / 2 - xdescent; cy = height / 2 - ydescent
            p = mpatches.Circle((cx, cy), radius=radius,
                                 facecolor=orig_handle.get_facecolor(),
                                 edgecolor=orig_handle.get_edgecolor(),
                                 linewidth=orig_handle.get_linewidth(),
                                 linestyle=orig_handle.get_linestyle())
            p.set_transform(trans)
            return [p]

    legend = [mpatches.Circle((0, 0), 1, facecolor=(0, 0, 0, 0.06),
                              edgecolor="black", linewidth=2.0,
                              linestyle="--", label="Planted disk"),
              mpatches.Circle((0, 0), 1, facecolor="none",
                              edgecolor=CENTROID_COLOR, linewidth=1.6,
                              linestyle="-", label="Centroid"),
              mpatches.Circle((0, 0), 1, facecolor="none",
                              edgecolor=BUCH_COLOR, linewidth=1.6,
                              linestyle="-", label="Buchin"),
              mpatches.Circle((0, 0), 1, facecolor="none",
                              edgecolor=GEOM_COLOR, linewidth=1.6,
                              linestyle="-", label="Geom-50")]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8.5,
                frameon=False, bbox_to_anchor=(0.5, 0.04),
                handlelength=1.4, handleheight=1.4, handletextpad=0.4,
                columnspacing=1.0,
                handler_map={mpatches.Circle: HandlerCircle()})
    fig.subplots_adjust(left=0.04, right=1.0, top=0.97, bottom=0.05,
                        wspace=0.0, hspace=-0.35)
    for ext in ("png", "pdf"):
        out = OUTPUTS / f"sanity_arkansas_map_disk_v3_9bin.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    render_rect_pq_two_sizes_9bin()
    render_disk_two_sizes_9bin()
    render_rect_map_v3_9bin()
    render_disk_map_v3_9bin()
