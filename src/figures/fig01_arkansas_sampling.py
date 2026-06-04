
# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------
"""Regenerate paper Fig 1: the Arkansas region-to-points sampling visualization.

Design goals:
  * Reader instantly sees: "each region is replaced with k uniform random points".
  * One highlighted county (Pulaski, FIPS 119 — Little Rock) acts as the key
    example: heavier outline + its samples drawn larger, so the eye lands there
    and the concept transfers to the rest of the map.
  * Tab-10 cycled colors per county (deterministic) so adjacent regions contrast
    cleanly — no rainbow, no random palette.
  * Corner badge announces `k`. No axis ticks; lat/lon clutter removed.
  * Style harmonized with v8: serif font, thin axes, white background.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from shapely.geometry import Point

SHP = DATA / "arkansas" / "COUNTY_BOUNDARY.shp"
OUT_DIR = OUTPUTS

HIGHLIGHT_FIPS = "007"  # Benton County (Bentonville — Walmart HQ) — globally recognizable

# Bentonville's three contiguous neighbors share ONE unified warm-sand color, so the
# eye reads "Bentonville (deep navy, magenta outline) surrounded by a warm ring."
NEIGHBOR_FIPS = {"143", "087", "015"}   # Washington, Madison, Carroll
NEIGHBOR_COLOR = "#C9A876"  # warm sandstone — complements deep navy, refined academic feel

# Curated 20-color academic palette — Tol Bright + Tol Vibrant + ColorBrewer Set1
# + Dark2 picks. Hand-checked to keep good perceptual separation while avoiding
# the deep-navy of Bentonville and the warm-sandstone of its neighbour ring.
ACADEMIC_PALETTE = [
    "#E41A1C",  # vivid red          (Set1)
    "#377EB8",  # mid blue           (Set1)
    "#4DAF4A",  # green              (Set1)
    "#984EA3",  # purple             (Set1)
    "#FF7F00",  # orange             (Set1)
    "#FFD700",  # gold               (custom)
    "#A65628",  # brown              (Set1)
    "#F781BF",  # pink               (Set1)
    "#1B9E77",  # teal-green         (Dark2)
    "#D95F02",  # burnt orange       (Dark2)
    "#7570B3",  # indigo             (Dark2)
    "#E7298A",  # magenta            (Dark2)
    "#66A61E",  # lime               (Dark2)
    "#A6761D",  # bronze             (Dark2)
    "#1F78B4",  # ocean blue         (Paired)
    "#33A02C",  # dark green         (Paired)
    "#FB9A99",  # salmon             (Paired)
    "#FDBF6F",  # peach              (Paired)
    "#CAB2D6",  # lavender           (Paired)
    "#66CCEE",  # cyan               (Tol Bright)
]


def greedy_map_coloring(gdf, fixed: dict[int, str], palette: list[str]) -> dict[int, str]:
    """Greedy map coloring on the polygon-adjacency graph.

    `fixed[i]` pre-assigns a color to county i (e.g. highlighted + its neighbors).
    Returns a {county_index: color} dict where no two adjacent counties share a
    color. By the four-colour theorem 4 hues suffice; greedy may use a few more,
    but our 12-colour palette gives plenty of headroom.
    """
    n = len(gdf)
    # Build adjacency via buffered intersect — `touches` alone misses pairs whose
    # shared border has sub-degree floating-point gaps in the shapefile. Buffering
    # both sides by ~0.005° (≈ 500 m at Arkansas latitude) catches those without
    # falsely linking truly disjoint counties.
    BUFFER_DEG = 0.005
    bufs = [g.buffer(BUFFER_DEG) for g in gdf.geometry]
    sindex = gdf.sindex
    adj = {i: set() for i in range(n)}
    for i, buf in enumerate(bufs):
        for j in sindex.intersection(buf.bounds):
            j = int(j)
            if j == i:
                continue
            if buf.intersects(gdf.geometry.iloc[j]):
                adj[i].add(j)

    # Process highest-degree counties first (DSATUR-lite). For each, pick the
    # *least-used-so-far* palette color that no neighbour has — this spreads
    # usage across the full 20-colour palette while still respecting the
    # no-adjacent-duplicates constraint.
    from collections import Counter
    order = sorted(range(n), key=lambda i: -len(adj[i]))
    color_of: dict[int, str] = dict(fixed)
    counts: Counter = Counter(color_of.values())
    for i in order:
        if i in color_of:
            continue
        used = {color_of[j] for j in adj[i] if j in color_of}
        available = [c for c in palette if c not in used]
        if not available:
            # Palette exhausted by neighbours — fall back to least-used overall.
            available = palette
        chosen = min(available, key=lambda c: (counts[c], palette.index(c)))
        color_of[i] = chosen
        counts[chosen] += 1
    return color_of


def sample_points(poly, k: int, rng: np.random.Generator) -> np.ndarray:
    """Rejection-sample k uniform points inside a polygon's geometry."""
    minx, miny, maxx, maxy = poly.bounds
    out = np.empty((k, 2))
    n = 0
    while n < k:
        cand_x = rng.uniform(minx, maxx, size=k * 2)
        cand_y = rng.uniform(miny, maxy, size=k * 2)
        for x, y in zip(cand_x, cand_y):
            if n == k:
                break
            if poly.contains(Point(x, y)):
                out[n] = (x, y)
                n += 1
    return out


def _apply_style() -> None:
    import paper_plots as pp
    pp.apply_style_v8()


def _draw_panel(ax, df, k: int, seed: int) -> None:
    """Render a single Arkansas-sampling panel on the given axes."""
    rng = np.random.default_rng(seed)

    df.plot(ax=ax, facecolor="#F4F4F4", edgecolor="#9C9C9C", linewidth=0.5,
            rasterized=True)

    highlight_idx = df.index[df["COUNTYFIPS"] == HIGHLIGHT_FIPS]
    highlight_idx = int(highlight_idx[0]) if len(highlight_idx) else -1

    # Same size for k=10 and k=50, big enough to read clearly when the
    # combined figure scales down in single-column LaTeX.
    pt_size_normal = 6.0
    pt_size_hl     = 8.5
    pt_edge_w      = 0.45
    HL_COLOR       = "#C2185B"
    HL_POINT_COLOR = "#0D47A1"

    fixed: dict[int, str] = {}
    if highlight_idx >= 0:
        fixed[highlight_idx] = HL_POINT_COLOR
    color_of = greedy_map_coloring(df, fixed=fixed, palette=ACADEMIC_PALETTE)

    for i, geom in enumerate(df.geometry):
        pts = sample_points(geom, k, rng)
        is_hl = i == highlight_idx
        color = color_of.get(i, ACADEMIC_PALETTE[0])
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=pt_size_hl if is_hl else pt_size_normal,
            c=[color], edgecolors="white", linewidths=pt_edge_w,
            alpha=0.95,
            zorder=4 if is_hl else 3,
        )

    if highlight_idx >= 0:
        hl = df.iloc[[highlight_idx]]
        hl.boundary.plot(ax=ax, color=HL_COLOR, linewidth=1.2, zorder=5)
        cx, cy = hl.geometry.iloc[0].centroid.coords[0]
        ax.annotate(
            "Bentonville",
            xy=(cx, cy),
            xytext=(cx + 0.65, cy - 0.55),
            fontsize=9, color="black", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=HL_COLOR, lw=0.8,
                            shrinkA=0, shrinkB=2),
            path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
            zorder=6,
        )

    # Tight bounds: drop the default 5% auto-padding so panels can sit edge to edge.
    xmin, ymin, xmax, ymax = df.total_bounds
    ax.set_xlim(xmin - 0.02, xmax + 0.02)
    ax.set_ylim(ymin - 0.02, ymax + 0.02)
    ax.set_aspect("equal")
    ax.margins(0, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render(k: int, seed: int = 42) -> None:
    """Single-panel PDF — kept for backwards compatibility."""
    _apply_style()
    df = gpd.read_file(SHP).to_crs("EPSG:4326").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(2.45, 1.75))
    _draw_panel(ax, df, k, seed)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    out = OUT_DIR / f"arkansas_Geom{k}.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def render_combined(seed: int = 42) -> None:
    """Single PDF with both k=10 (left) and k=50 (right) panels.

    Bigger panels + halved horizontal gap; "N points per county" labels sit
    above each panel at body-text-plus size.
    """
    _apply_style()
    df = gpd.read_file(SHP).to_crs("EPSG:4326").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    for ax, k in zip(axes, (10, 50)):
        _draw_panel(ax, df, k, seed)
        # Slightly lower (closer to the panel top) — matching the more-relaxed
        # spacing the JDArkansas legend now uses below its panels.
        ax.text(0.5, 1.005, f"Geom {k}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=19, color="black", fontweight="normal")
    fig.subplots_adjust(left=0.002, right=0.998, top=0.90, bottom=0.01,
                        wspace=0.0)
    out = OUT_DIR / "arkansas_sampling_combined.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    for k in (10, 50):
        render(k)
    render_combined()
