from __future__ import annotations

"""Shape-family floor figure (Arkansas, same style as JDArkansas.pdf).

Two panels side by side:
  A. planted RECTANGLE (red dashed) + best-fit DISK (green solid)
  B. planted DISK of equal area (red dashed) + best-fit RECTANGLE (green solid)

Both Jaccard distances are computed over the same reference set A used by the
paper: 500 uniformly random points per Arkansas county, ~37 500 points total.
Independent of any Poisson draw — this is purely a shape-family floor.
"""

# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

SHP  = DATA / "arkansas" / "COUNTY_BOUNDARY.shp"
OUT_DIR = OUTPUTS
import paper_plots as pp  # noqa: E402

SEED = 42
N_PER_REGION = 500

# Panel A — pedagogical anchor: equal-area square centered on the experiment's
# rectangle. Area = 3.0 sq° → side = sqrt(3) ≈ 1.732.  Classic floor 1 - π/4.
SQUARE_CENTER = (-92.5, 34.75)
SQUARE_HALF   = float(np.sqrt(3.0) / 2.0)                 # ≈ 0.866

# Panel B — the actual experiment target: 2.0 × 1.5 rectangle (aspect 1.33).
TARGET_RECT = Polygon([(-93.5, 34), (-93.5, 35.5), (-91.5, 35.5), (-91.5, 34)])
RECT_W = 2.0
RECT_H = 1.5

# Panel C — reverse direction: equal-area disk centered at the same point.
TARGET_DISK_R = float(np.sqrt(RECT_W * RECT_H / np.pi))   # ≈ 0.977
TARGET_DISK_CENTER = (-92.5, 34.75)


# ---- 500-per-region uniform reference set ------------------------------------

def reference_set(gdf, n_per_region: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for geom in gdf.geometry:
        minx, miny, maxx, maxy = geom.bounds
        accepted = []
        while len(accepted) < n_per_region:
            bx = rng.uniform(minx, maxx, size=n_per_region * 3)
            by = rng.uniform(miny, maxy, size=n_per_region * 3)
            for x, y in zip(bx, by):
                if len(accepted) == n_per_region:
                    break
                if geom.contains(Point(x, y)):
                    accepted.append((x, y))
        out.append(np.asarray(accepted))
    return np.vstack(out)


# ---- vectorized indicator helpers --------------------------------------------

def in_rect(pts: np.ndarray, cx: float, cy: float, hw: float, hh: float) -> np.ndarray:
    """axis-aligned rectangle centered at (cx,cy), half-width hw, half-height hh."""
    return (np.abs(pts[:, 0] - cx) <= hw) & (np.abs(pts[:, 1] - cy) <= hh)


def in_disk(pts: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    return (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2 <= r ** 2


def point_jd(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = (mask_a | mask_b).sum()
    inter = (mask_a & mask_b).sum()
    return 1.0 - inter / union if union > 0 else 1.0


# ---- best-fit searches -------------------------------------------------------

def best_disk_for_mask(pts: np.ndarray, target_mask: np.ndarray,
                       center_hint: tuple[float, float],
                       r_range: tuple[float, float] = (0.55, 1.35)) -> dict:
    """Sweep disk (cx, cy, r) to minimize JD against any target mask."""
    cx0, cy0 = center_hint
    cxs = np.linspace(cx0 - 0.4, cx0 + 0.4, 21)
    cys = np.linspace(cy0 - 0.4, cy0 + 0.4, 21)
    rs  = np.linspace(r_range[0], r_range[1], 36)
    best = {"jd": 1.0, "cx": None, "cy": None, "r": None}
    for cx in cxs:
        for cy in cys:
            d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
            for r in rs:
                mask = d2 <= r * r
                jd = point_jd(target_mask, mask)
                if jd < best["jd"]:
                    best.update(jd=jd, cx=float(cx), cy=float(cy), r=float(r))
    return best


def best_rect_for_disk(pts: np.ndarray, target_disk_mask: np.ndarray) -> dict:
    """Sweep axis-aligned rectangle (cx, cy, hw, hh) to minimize JD against
    the planted disk of equal area."""
    cxs = np.linspace(-92.9, -92.1, 17)
    cys = np.linspace(34.35, 35.15, 17)
    hws = np.linspace(0.45, 1.30, 22)
    hhs = np.linspace(0.45, 1.30, 22)
    best = {"jd": 1.0, "cx": None, "cy": None, "hw": None, "hh": None}
    for cx in cxs:
        for cy in cys:
            dx = np.abs(pts[:, 0] - cx)
            dy = np.abs(pts[:, 1] - cy)
            for hw in hws:
                mx = dx <= hw
                for hh in hhs:
                    mask = mx & (dy <= hh)
                    jd = point_jd(target_disk_mask, mask)
                    if jd < best["jd"]:
                        best.update(jd=jd, cx=float(cx), cy=float(cy),
                                    hw=float(hw), hh=float(hh))
    return best


# ---- render ------------------------------------------------------------------

def render():
    pp.apply_style_v9()
    plt.rcParams.update({  # match JDArkansas serif tweak
        "font.family":      "serif",
        "font.serif":       ["Linux Libertine", "Liberation Serif", "DejaVu Serif"],
        "axes.facecolor":   "white",
        "figure.facecolor": "white",
        "axes.grid":        False,
    })

    gdf = gpd.read_file(SHP).to_crs("EPSG:4326")
    print(f"loaded {len(gdf)} Arkansas counties")
    print(f"sampling {N_PER_REGION} points per region (seed={SEED}) ...")
    A = reference_set(gdf, N_PER_REGION, SEED)
    print(f"  reference set |A| = {len(A)}")

    # masks
    sq_mask   = in_rect(A, *SQUARE_CENTER, SQUARE_HALF, SQUARE_HALF)
    rect_mask = in_rect(A, -92.5, 34.75, RECT_W / 2, RECT_H / 2)
    disk_mask = in_disk(A, *TARGET_DISK_CENTER, TARGET_DISK_R)
    print(f"  |A ∩ square| = {sq_mask.sum()}    (equal-area square, side={2*SQUARE_HALF:.3f})")
    print(f"  |A ∩ rect|   = {rect_mask.sum()}  (2.0 × 1.5 rectangle)")
    print(f"  |A ∩ disk|   = {disk_mask.sum()}  (equal-area disk, r={TARGET_DISK_R:.3f})")

    print("\n[panel A] best-fit disk for the SQUARE (textbook anchor) ...")
    a = best_disk_for_mask(A, sq_mask, SQUARE_CENTER, r_range=(0.65, 1.10))
    print(f"  best disk: center=({a['cx']:.3f}, {a['cy']:.3f}) r={a['r']:.3f}")
    print(f"  Jaccard distance = {a['jd']:.4f}    "
          f"(theory: 1 - π/4 ≈ 0.2146)")

    print("\n[panel B] best-fit disk for the planted RECTANGLE ...")
    b = best_disk_for_mask(A, rect_mask, (-92.5, 34.75), r_range=(0.60, 1.30))
    print(f"  best disk: center=({b['cx']:.3f}, {b['cy']:.3f}) r={b['r']:.3f}")
    print(f"  Jaccard distance = {b['jd']:.4f}")

    print("\n[panel C] best-fit rectangle for the planted DISK ...")
    c = best_rect_for_disk(A, disk_mask)
    print(f"  best rect: center=({c['cx']:.3f}, {c['cy']:.3f}) hw={c['hw']:.3f} hh={c['hh']:.3f}")
    print(f"  Jaccard distance = {c['jd']:.4f}")

    # sanity-check
    for label, jd in [("square-vs-disk", a["jd"]),
                      ("rect-vs-disk",   b["jd"]),
                      ("disk-vs-rect",   c["jd"])]:
        if jd < 0.10 or jd > 0.45:
            print(f"  ⚠ {label} floor {jd:.3f} is outside the 0.10–0.45 sanity band "
                  f"— double-check search bounds")

    # Single panel — only the central RECTANGLE + best-fit-disk experiment
    # (JD ≈ 0.218). The square anchor and reverse disk panels are dropped.
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 4.6))
    target_color  = "#C62828"
    discovered_color = "#1B5E20"

    # subsample dots for visual readability (500/region is too many to render)
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(A), size=min(len(A), 3000), replace=False)
    sub_pts = A[sub_idx]

    def _draw_basemap(ax):
        gdf.plot(ax=ax, color="#FAFAFA", edgecolor="#9C9C9C", linewidth=0.5,
                 rasterized=True)
        ax.scatter(sub_pts[:, 0], sub_pts[:, 1], s=2.0, color="#1F1F1F",
                   alpha=0.45, linewidths=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        for sp in ax.spines.values(): sp.set_visible(False)

    _draw_basemap(ax)
    ax.add_patch(mpatches.Rectangle((-93.5, 34), RECT_W, RECT_H,
                                    edgecolor=target_color,
                                    facecolor=(0.78, 0.16, 0.16, 0.18),
                                    lw=2.0, ls=(0, (5, 2))))
    ax.add_patch(mpatches.Circle((b["cx"], b["cy"]), b["r"],
                                 edgecolor=discovered_color, facecolor="none",
                                 lw=2.0, ls="-"))
    ax.set_title("Best disk to target rectangle",
                 fontsize=11, pad=6)

    # ---- shared legend at bottom ------------------------------------------
    # Target: DASHED red rectangle (Patch supports linestyle).
    # Best-fit: SOLID green circle (Line2D with circle marker).
    from matplotlib.lines import Line2D
    handles = [
        mpatches.Patch(facecolor=(0.78, 0.16, 0.16, 0.18),
                       edgecolor=target_color,
                       linestyle="--", linewidth=2.0,
                       label="Target rectangle"),
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor="none",
               markeredgecolor=discovered_color,
               markeredgewidth=2.0, markersize=14,
               label="Best-fit disk"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.06),
               ncols=2, fontsize=10, frameon=False)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.14)

    out_pdf = OUT_DIR / "shape_floor.pdf"
    out_png = OUT_DIR / "shape_floor.png"
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"\nwrote {out_pdf}")
    print(f"wrote {out_png}")
    return a["jd"], b["jd"]


if __name__ == "__main__":
    render()
