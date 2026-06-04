from __future__ import annotations

"""Two single-panel Arkansas rect figures with map insets and all
7 methods (Centroid, Random Point, FlexScan, Geom 5/10/50, Buchin).

Same paper_plots v9 styling as fig 8 / fig 9, single-panel with
Arkansas county-map inset in the top-right (matches fig 8 layout for 30%,
bottom-left for 10% since the curves stay HIGH there for a long stretch).

Outputs:
  fig8_arkansas_30_all_7_methods_inset.{pdf,png}
  fig9_arkansas_10_all_7_methods_inset.{pdf,png}
"""

# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------

import pickle
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

IO   = IO
CACHED = OUTPUTS / "cached_data"
OUT  = OUTPUTS
import paper_plots as pp

# Share the same _draw_inset helper from the existing renderer.
from render_paper_figs_inset import _draw_inset, ARK_SHP

P_GRID  = np.round(np.arange(0.20, 0.95, 0.05), 4)
PQ_DIFF = np.round(P_GRID - 0.20, 4)

SAMPLING_METHODS = ["Centroid", "Random Point", "Geom 5", "Geom 10", "Geom 50"]

ARK30_TARGET = Polygon([(-93.5, 34), (-93.5, 35.5), (-91.5, 35.5), (-91.5, 34)])
ARK10_TARGET = Polygon([(-92.85, 34.40), (-92.85, 35.10),
                        (-92.15, 35.10), (-92.15, 34.40)])

CONFIGS = [
    {"label":      "Arkansas — 30 % target rectangle",
     "rerun_pkl":  IO / "arkansas_30_rerun.pkl",
     "buchin_csv": IO / "fig8_with_jaccard.csv",
     "flex_pkl":   CACHED / "arkansas_30.pkl",
     "target":     ARK30_TARGET,
     "out_stem":   "fig8_arkansas_30_all_7_methods_inset",
     "inset_pos":  [0.66, 0.41, 0.33, 0.58],
     "inset_anchor": "NE",
     "transparent": False},
    {"label":      "Arkansas — 10 % target rectangle",
     "rerun_pkl":  IO / "arkansas_10_rerun.pkl",
     "buchin_csv": IO / "fig9_with_jaccard.csv",
     "flex_pkl":   CACHED / "arkansas_10.pkl",
     "target":     ARK10_TARGET,
     "out_stem":   "fig9_arkansas_10_all_7_methods_inset",
     "inset_pos":  [0.015, 0.02, 0.31, 0.40],
     "inset_anchor": "SW",
     "transparent": False},
]


def _curves_for(cfg: dict) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    with open(cfg["rerun_pkl"], "rb") as f:
        rerun = pickle.load(f)
    df = pd.DataFrame(rerun["records"])
    for m in SAMPLING_METHODS:
        sub = df[df["method"] == m]
        mat = (sub.pivot_table(index="trial", columns="p",
                                values="pool_jd", aggfunc="mean")
                  .reindex(columns=P_GRID))
        out[m] = mat.to_numpy(dtype=float)

    bdf = pd.read_csv(cfg["buchin_csv"])
    bdf = bdf[bdf["mode"] == "rect"]
    mat = (bdf.pivot_table(index="trial_id", columns="p",
                            values="point_jaccard", aggfunc="mean")
              .reindex(columns=P_GRID))
    out["Buchin"] = mat.to_numpy(dtype=float)

    with open(cfg["flex_pkl"], "rb") as f:
        flex_pkg = pickle.load(f)
    out["FlexScan"] = np.asarray(flex_pkg["methods"]["FlexScan"]).reshape(-1)
    return out


def _restyle():
    pp.apply_style_v9()
    # Named colours so the text can refer to them directly:
    #   FlexScan = brown,  Buchin = navy.
    pp.METHOD_STYLE["Buchin"] = {
        "color":  "#1F2D5C",            # navy
        "ls":     (0, (5, 2)),
        "marker": "o",
        "ms":     5.0,
        "lw":     1.5,
    }
    pp.METHOD_STYLE["FlexScan"] = {
        "color":  "#8B4513",            # saddle brown
        "ls":     (0, (5, 2)),
        "marker": "v",
        "ms":     5.5,
        "lw":     1.3,
    }
    pp.METHOD_ORDER = SAMPLING_METHODS + ["FlexScan", "Buchin"]
    plt.rcParams.update({
        "axes.labelsize":  16,
        "axes.titlesize":  17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11.5,
    })


def render_one(cfg: dict) -> None:
    _restyle()
    data = _curves_for(cfg)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    pp.plot_jaccard_vs_pq(ax, data, PQ_DIFF,
                          show_band=True, band_alpha=0.22)
    ax.set_title(cfg["label"], pad=10, weight="medium")
    ax.set_xlabel(r"$p - q$ difference")
    ax.set_ylabel("Point Jaccard distance")

    # Shorten the last x-tick "0.70" → "0.7" and right-align it.
    fig.canvas.draw()
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    if xticklabels and xticklabels[-1] in ("0.70", "0.7000"):
        xticklabels[-1] = "0.7"
        ax.set_xticklabels(xticklabels)
    last = ax.get_xticklabels()[-1]
    last.set_horizontalalignment("right")

    # Single-row legend, expand-mode, frameless.
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    n = len(labels)
    # Stretch the legend a bit beyond the axes on both sides so it lines
    # up with the outermost x-tick labels (0.00 and 0.7), and a touch
    # further left as the user prefers.
    ax.legend(handles, labels,
              loc="upper left",
              bbox_to_anchor=(-0.05, -0.22, 1.05, 0.10),
              mode="expand", ncol=n, fontsize=11.0, frameon=False,
              handlelength=1.4, handletextpad=0.35,
              columnspacing=0.0)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.93, bottom=0.22)

    # Arkansas county-map inset.
    xticks = [-94, -92, -90] if cfg["inset_anchor"] == "NE" else []
    yticks = [33.5, 35, 36.5] if cfg["inset_anchor"] == "NE" else []
    _draw_inset(ax, ARK_SHP, cfg["target"],
                xticks=xticks, yticks=yticks,
                inset_pos=cfg["inset_pos"],
                inset_anchor=cfg["inset_anchor"])

    for ext in ("pdf", "png"):
        path = OUT / f"{cfg['out_stem']}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main():
    for cfg in CONFIGS:
        render_one(cfg)


if __name__ == "__main__":
    main()
