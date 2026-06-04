"""Style module for SIGSPATIAL '26 paper figures.

Two style variants for user selection in Phase 2:
  apply_style_v1(): minimalist — no gridlines, generous whitespace
  apply_style_v2(): subtle grid — light horizontal gridlines for read-off accuracy

After Phase 2 selection, the unused variant is dropped.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- Per-style method dictionaries -------------------------------------------
# v3 "All solid, large markers" — Tableau-10-ish, distinguish by color + marker
STYLE_V3 = {
    "Centroid":     {"color": "#D62728", "ls": "-", "marker": "o", "ms": 6.5, "lw": 1.7},
    "Random Point": {"color": "#9467BD", "ls": "-", "marker": "D", "ms": 5.5, "lw": 1.7},
    "Geom 5":       {"color": "#2CA02C", "ls": "-", "marker": "^", "ms": 6.5, "lw": 1.7},
    "Geom 10":      {"color": "#FF7F0E", "ls": "-", "marker": "X", "ms": 7.0, "lw": 1.7},
    "Geom 50":      {"color": "#1F77B4", "ls": "-", "marker": "s", "ms": 6.0, "lw": 2.4},
    "FlexScan":     {"color": "#17BECF", "ls": "-", "marker": "*", "ms": 9.0, "lw": 1.7},
}

# v4 "Notebook-faithful" — colors close to the original seaborn output the user is used to,
# all solid lines, larger figure & fonts; let color do the work.
STYLE_V4 = {
    "Centroid":     {"color": "#E24A33", "ls": "-", "marker": "o", "ms": 7.0, "lw": 1.8},  # red
    "Random Point": {"color": "#348ABD", "ls": "-", "marker": "D", "ms": 6.0, "lw": 1.8},  # blue
    "Geom 5":       {"color": "#467821", "ls": "-", "marker": "v", "ms": 7.0, "lw": 1.8},  # olive green
    "Geom 10":      {"color": "#D9A23B", "ls": "-", "marker": "X", "ms": 7.5, "lw": 1.8},  # ochre/orange
    "Geom 50":      {"color": "#7A378B", "ls": "-", "marker": "s", "ms": 6.5, "lw": 2.4},  # purple
    "FlexScan":     {"color": "#1E1E1E", "ls": "-", "marker": "*", "ms": 9.5, "lw": 1.8},  # near-black
}

# v5 "Ours vs Theirs" — story-driven: dashed for baselines/competitors, solid for our Geom-k methods.
# Distinct marker per method within each group. Saturated, accessible at small print.
STYLE_V5 = {
    "Centroid":     {"color": "#888888", "ls": (0, (5, 2)),   "marker": "o", "ms": 6.0, "lw": 1.6},  # grey dashed
    "Random Point": {"color": "#888888", "ls": (0, (1, 1.5)), "marker": "D", "ms": 5.0, "lw": 1.5},  # grey dotted
    "FlexScan":     {"color": "#B30000", "ls": (0, (5, 2)),   "marker": "*", "ms": 9.0, "lw": 1.7},  # dark red dashed
    "Geom 5":       {"color": "#9ECAE1", "ls": "-",           "marker": "^", "ms": 6.0, "lw": 1.9},  # light blue solid
    "Geom 10":      {"color": "#4292C6", "ls": "-",           "marker": "X", "ms": 6.5, "lw": 1.9},  # mid blue solid
    "Geom 50":      {"color": "#08306B", "ls": "-",           "marker": "s", "ms": 6.0, "lw": 2.4},  # deep blue solid (hero)
}

# v6 — refinement of v5 per feedback:
#   * narrower lines
#   * smaller markers
#   * Centroid != Random Point (distinct colors)
#   * Geom 5/10/50 share a green family; Geom 50 = vivid green with star
STYLE_V6 = {
    "Centroid":     {"color": "#5A3E2B", "ls": (0, (5, 2)),   "marker": "o", "ms": 4.5, "lw": 1.1},  # warm brown, dashed
    "Random Point": {"color": "#9C9C9C", "ls": (0, (1, 1.6)), "marker": "D", "ms": 3.8, "lw": 1.0},  # light grey, dotted
    "FlexScan":     {"color": "#B30000", "ls": (0, (5, 2)),   "marker": "v", "ms": 4.8, "lw": 1.1},  # dark red, dashed, down-triangle
    "Geom 5":       {"color": "#A1D99B", "ls": "-",           "marker": "^", "ms": 4.5, "lw": 1.2},  # light green
    "Geom 10":      {"color": "#41AB5D", "ls": "-",           "marker": "s", "ms": 4.0, "lw": 1.3},  # mid green, square
    "Geom 50":      {"color": "#006D2C", "ls": "-",           "marker": "*", "ms": 6.5, "lw": 1.6},  # vivid green, star
}

# v7 — fixes v6 band-blending: greens spread across hue (teal / lime / emerald)
# so std-dev fills stay distinguishable; baselines moved off warm-neutral palette.
STYLE_V7 = {
    "Centroid":     {"color": "#4A148C", "ls": (0, (5, 2)),   "marker": "o", "ms": 4.5, "lw": 1.1},  # dark indigo, dashed
    "Random Point": {"color": "#8C7853", "ls": (0, (1, 1.6)), "marker": "D", "ms": 3.8, "lw": 1.0},  # warm beige, dotted
    "FlexScan":     {"color": "#B30000", "ls": (0, (5, 2)),   "marker": "v", "ms": 4.8, "lw": 1.1},  # dark red, dashed
    "Geom 5":       {"color": "#1F9E89", "ls": "-",           "marker": "^", "ms": 4.5, "lw": 1.2},  # teal-green
    "Geom 10":      {"color": "#7CB342", "ls": "-",           "marker": "s", "ms": 4.0, "lw": 1.3},  # lime-green
    "Geom 50":      {"color": "#005A32", "ls": "-",           "marker": "*", "ms": 6.5, "lw": 1.6},  # deep emerald, star
}

# v8 — high-saturation rework so the tiny Geom 5 / Geom 10 std bands still read at low alpha.
# Bigger hue gaps inside the green family (kelly → teal → forest) and cleaner contrasting baselines.
STYLE_V8 = {
    "Centroid":     {"color": "#E76F51", "ls": (0, (5, 2)),   "marker": "o", "ms": 4.5, "lw": 1.1},  # warm coral, dashed
    "Random Point": {"color": "#6C757D", "ls": (0, (1, 1.6)), "marker": "D", "ms": 3.8, "lw": 1.0},  # cool grey, dotted
    "FlexScan":     {"color": "#C1121F", "ls": (0, (5, 2)),   "marker": "v", "ms": 4.8, "lw": 1.1},  # crimson, dashed
    "Geom 5":       {"color": "#4CB050", "ls": "-",           "marker": "^", "ms": 4.5, "lw": 1.2},  # kelly green
    "Geom 10":      {"color": "#0E8E81", "ls": "-",           "marker": "s", "ms": 4.0, "lw": 1.3},  # teal-green
    "Geom 50":      {"color": "#00875A", "ls": "-",           "marker": "*", "ms": 8.0, "lw": 1.9,
                     "mew": 0.6, "mec": "white"},  # vivid jewel-emerald (Atlassian green), white-outlined star — the hero
}

# v9 "Paper original" — replicates the seaborn-themed look in the published Fig 10:
#   Centroid = red dashed circle, Random Point = deepskyblue dash-dot diamond,
#   Geom 5 = olivedrab dotted triangle, Geom 10 = darkorange dashed X,
#   Geom 50 = darkmagenta solid square, FlexScan = blue solid star.
# Used when the user wants the figures to visually match the original paper.
STYLE_V9 = {
    "Centroid":     {"color": "red",         "ls": (0, (3, 5, 1, 5, 1, 5)), "marker": "o", "ms": 6.5, "lw": 1.8},
    "Random Point": {"color": "deepskyblue", "ls": "-.",                    "marker": "D", "ms": 5.5, "lw": 1.8},
    "Geom 5":       {"color": "olivedrab",   "ls": ":",                     "marker": "^", "ms": 6.5, "lw": 1.8},
    "Geom 10":      {"color": "darkorange",  "ls": "--",                    "marker": "X", "ms": 7.5, "lw": 2.4},
    "Geom 50":      {"color": "darkmagenta", "ls": "-",                     "marker": "s", "ms": 5.5, "lw": 1.8},
    "FlexScan":     {"color": "blue",        "ls": "-",                     "marker": "*", "ms": 9.0, "lw": 1.8},
}

# v8 — drops the all-green constraint. ColorBrewer Set1, designed for maximum
# categorical hue separation; bands stay readable at alpha 0.10.
# Baselines keep dashed/dotted (less reliable); Geom-family is all solid;
# Geom 50 = vivid green + star (hero).
STYLE_V8 = {
    "Centroid":     {"color": "#A65628", "ls": (0, (5, 2)),   "marker": "o", "ms": 4.5, "lw": 1.1},  # brown, dashed
    "Random Point": {"color": "#984EA3", "ls": (0, (1, 1.6)), "marker": "D", "ms": 3.8, "lw": 1.0},  # purple, dotted
    "FlexScan":     {"color": "#E41A1C", "ls": (0, (5, 2)),   "marker": "v", "ms": 4.8, "lw": 1.1},  # red, dashed
    "Geom 5":       {"color": "#377EB8", "ls": "-",           "marker": "^", "ms": 4.5, "lw": 1.3},  # blue
    "Geom 10":      {"color": "#FF7F00", "ls": "-",           "marker": "s", "ms": 4.0, "lw": 1.3},  # orange
    "Geom 50":      {"color": "#4DAF4A", "ls": "-",           "marker": "*", "ms": 7.0, "lw": 1.7},  # green, star (hero)
}

# Default = v3 (overwritten by apply_style_vN)
METHOD_STYLE: dict[str, dict] = dict(STYLE_V3)

# Plot order — Geom 50 last so it sits on top of less-accurate methods.
METHOD_ORDER = ["Centroid", "Random Point", "FlexScan", "Geom 5", "Geom 10", "Geom 50"]

# acmart sigconf widths (inches)
ONE_COL = 3.33
TWO_COL = 7.00


def _base_rc(font_size: float = 9.5) -> dict:
    return {
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":         font_size,
        "axes.labelsize":    font_size + 0.5,
        "axes.titlesize":    font_size + 1,
        "xtick.labelsize":   font_size - 0.5,
        "ytick.labelsize":   font_size - 0.5,
        "legend.fontsize":   font_size - 0.5,
        "legend.frameon":    False,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size":  3.5,
        "ytick.major.size":  3.5,
        "lines.solid_capstyle":  "round",
        "lines.dash_capstyle":   "round",
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.04,
    }


def _apply(rc_extra: dict, style_dict: dict) -> None:
    global METHOD_STYLE
    METHOD_STYLE = dict(style_dict)
    mpl.rcdefaults()
    rc = _base_rc()
    rc.update(rc_extra)
    mpl.rcParams.update(rc)


def apply_style_v3() -> None:
    """All-solid, larger markers, no grid. Tableau-ish."""
    _apply({"axes.grid": False}, STYLE_V3)


def apply_style_v4() -> None:
    """Notebook-faithful palette, generous fonts, faint horizontal grid."""
    _apply({
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#E0E0E0", "grid.linewidth": 0.5,
        "grid.linestyle": "-", "axes.grid.axis": "y",
    }, STYLE_V4)


def apply_style_v5() -> None:
    """Story-driven: dashed = baselines/competitors, solid = our methods (blue ramp)."""
    _apply({"axes.grid": False}, STYLE_V5)


def apply_style_v6() -> None:
    """v5 refined: narrower lines, smaller markers, distinct Centroid color,
    Geom-family green ramp with vivid green + star for Geom 50."""
    _apply({"axes.grid": False}, STYLE_V6)


def apply_style_v7() -> None:
    """v6 with chromatic spread inside the green family so std-dev bands stay
    distinguishable; lighter band alpha set in draft script."""
    _apply({"axes.grid": False}, STYLE_V7)


def apply_style_v8() -> None:
    """v7 with stronger saturation so the small Geom-5/10 bands still read at
    low alpha; Geom 50 is the visual hero — vivid jewel-emerald with a thicker
    line, larger star marker, and a thin white outline to pop against its band."""
    _apply({"axes.grid": False}, STYLE_V8)


def apply_style_v9() -> None:
    """Paper-original look (matches Fig 10 screenshot): seaborn-themed gridded
    background, sans-serif, red/cyan/olive/orange/magenta/blue palette."""
    global METHOD_STYLE
    METHOD_STYLE = dict(STYLE_V9)
    mpl.rcdefaults()
    # Seaborn whitegrid-like theme: light grey grid on white bg, sans-serif.
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size":         10,
        "axes.labelsize":    11,
        "axes.titlesize":    12,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.frameon":    True,
        "legend.framealpha": 0.85,
        "legend.edgecolor":  "#CCCCCC",
        "axes.facecolor":    "#EAEAF2",          # seaborn default panel grey
        "axes.edgecolor":    "white",
        "axes.linewidth":    1.0,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.color":        "white",
        "grid.linewidth":    1.0,
        "grid.linestyle":    "-",
        "xtick.major.width": 0,
        "ytick.major.width": 0,
        "xtick.major.size":  0,
        "ytick.major.size":  0,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.spines.left":  False,
        "axes.spines.bottom":False,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,
    })


def plot_jaccard_vs_pq(
    ax: plt.Axes,
    data: Mapping[str, Sequence[Sequence[float]] | Sequence[float]],
    pq_diff: Sequence[float],
    *,
    show_band: bool = True,
    band_alpha: float = 0.13,
    legend_loc: str = "upper right",
) -> None:
    """Draw mean ± 1 std band for each method against pq difference.

    `data[method]` may be either:
      - 2-D (n_trials × n_pq) — band drawn from std across trials
      - 1-D (n_pq,)           — single curve, no band (e.g. FlexScan)
    """
    x = np.asarray(pq_diff)
    for method in METHOD_ORDER:
        if method not in data:
            continue
        style = METHOD_STYLE[method]
        arr = np.asarray(data[method])
        if arr.ndim == 2 and arr.shape[0] > 1:
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            if show_band:
                ax.fill_between(x, mean - std, mean + std,
                                color=style["color"], alpha=band_alpha, lw=0)
            y = mean
        else:
            y = arr.ravel()
        ax.plot(x, y,
                color=style["color"], ls=style["ls"],
                marker=style["marker"], ms=style["ms"], lw=style["lw"],
                mew=style.get("mew", 0.0),
                mec=style.get("mec", style["color"]),
                label=method)

    ax.set_xlabel("pq Difference")
    ax.set_ylabel("Point Jaccard Distance")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.02, 1.02)
    # Match the paper screenshot: show every pq tick, slightly rotated.
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in x], rotation=0)
    ax.legend(loc=legend_loc)
