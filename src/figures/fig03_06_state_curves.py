
# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------
"""Render Grid(100) all-methods accuracy figures with state-map insets.

Same data as render_state_grid100_allmethods.py — adds a state-shape
inset in the upper-right of each curve plot, with the target rectangle
highlighted; legend moves outside the axes (right of the plot) so it
doesn't collide with the inset map.

Pattern mirrors render_all.render_nyc_with_inset() exactly.

Outputs:
  nyc_grid100_allmethods_inset.{pdf,png}
  utah_grid100_allmethods_inset.{pdf,png}
  california_grid100_allmethods_inset.{pdf,png}
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon

import paper_plots as pp

BU   = OUTPUTS

NYC_TARGET = Polygon([(-74.0, 40.65), (-74.0, 40.8), (-73.8, 40.8), (-73.8, 40.65)])
UTAH_TARGET = Polygon([(-113, 38), (-113, 40.5), (-110, 40.5), (-110, 38)])
CA_TARGET = Polygon([(-122.35, 35.5), (-122.35, 40),
                     (-118.35, 40),   (-118.35, 35.5)])

STATES = [
    {"name": "NYC",
     "pkl":   BU / "nyc_grid100_check.pkl",
     "title": "New York City Zip Codes",
     "out":   BU / "nyc_grid100_allmethods_inset",
     "shp":   DATA / "nyc/ZIP_CODE_040114.shp",
     "target": NYC_TARGET,
     # x-ticks / y-ticks for the inset
     "xticks": [-74.2, -74.0, -73.8],
     "yticks": [40.5, 40.7, 40.9]},
    {"name": "Utah",
     "pkl":   BU / "utah_grid100.pkl",
     "title": "Utah Counties",
     "out":   BU / "utah_grid100_allmethods_inset",
     "shp":   DATA / "Utah/geo_export_964ee856-5a3f-431f-b4c6-301973ba317c.shp",
     "target": UTAH_TARGET,
     "xticks": [-114, -112, -110],
     "yticks": [37, 39, 41]},
    {"name": "California",
     "pkl":   BU / "california_grid100.pkl",
     "title": "California Counties",
     "out":   BU / "california_grid100_allmethods_inset",
     "shp":   DATA / "california/cnty19_1.shp",
     "target": CA_TARGET,
     "xticks": [-124, -120, -116],
     "yticks": [34, 38, 42]},
]

METHODS = ["Centroid", "Random Point", "Geom 5", "Geom 10", "Geom 50"]


def _add_inset(ax, state):
    """Inset map — restrained, publication style (academic / NeurIPS feel):
    clean white background, hairline neutral-grey border, compact ticks."""
    gdf = gpd.read_file(state["shp"]).to_crs("EPSG:4326")

    inset_pos = [0.55, 0.38, 0.44, 0.60]
    inset = ax.inset_axes(inset_pos)
    gdf.plot(ax=inset, color="white", edgecolor="#9A9A9A", linewidth=0.45,
             rasterized=True)

    # Planted target — restrained red dashed outline, very light fill.
    tgt = state["target"]
    target_edge = "#C0392B"
    target_face = (0.78, 0.16, 0.16, 0.10)
    tx0, ty0, tx1, ty1 = tgt.bounds
    inset.add_patch(mpatches.Rectangle(
        (tx0, ty0), tx1 - tx0, ty1 - ty0,
        edgecolor=target_edge, facecolor=target_face,
        lw=1.2, ls=(0, (4, 2))))

    inset.set_xticks(state["xticks"])
    inset.set_yticks(state["yticks"])
    inset.tick_params(axis="both", which="major", labelsize=9,
                       length=2.5, width=0.5, pad=1.5, colors="#444444")
    inset.set_facecolor("white")
    for sp in inset.spines.values():
        sp.set_edgecolor("#BFBFBF"); sp.set_linewidth(0.6)
    inset.set_aspect("equal")
    inset.set_anchor("NE")


def render_state(state):
    with open(state["pkl"], "rb") as f:
        pkg = pickle.load(f)
    p_probs = np.asarray(pkg["p_probs"])
    pq_diff = p_probs - pkg["q"]
    data = {m: np.asarray(pkg["results"][m][100]) for m in METHODS}

    pp.apply_style_v9()
    plt.rcParams.update({
        "axes.labelsize":  16.0,
        "axes.titlesize":  17.0,
        "xtick.labelsize": 13.0,
        "ytick.labelsize": 13.0,
        "legend.fontsize": 13.5,
    })
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    pp.plot_jaccard_vs_pq(ax, data, pq_diff,
                          show_band=True, band_alpha=0.22)
    ax.set_title(state["title"], pad=10, weight="medium")
    ax.set_xlabel(r"$p - q$ difference")
    ax.set_ylabel("Point Jaccard distance")
    # Shorten the last x-tick label "0.70" → "0.7" and right-align it
    # so it sits comfortably inside the axes.
    fig.canvas.draw()
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    if xticklabels and xticklabels[-1] in ("0.70", "0.7000"):
        xticklabels[-1] = "0.7"
        ax.set_xticklabels(xticklabels)
    last = ax.get_xticklabels()[-1]
    last.set_horizontalalignment("right")

    # Legend stretched to span exactly the data x-range: first entry
    # aligns with x=0.00, last entry aligns with x=0.70.
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.legend(handles, labels,
              loc="upper left",
              bbox_to_anchor=(0.0, -0.22, 1.0, 0.10),
              mode="expand", ncol=len(labels), frameon=False,
              handlelength=1.8, handletextpad=0.45,
              columnspacing=0.0)

    _add_inset(ax, state)

    fig.subplots_adjust(left=0.085, right=0.975, top=0.93, bottom=0.22)
    for ext in ("png", "pdf"):
        out = state["out"].with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)
    plt.rcdefaults()


def main():
    for s in STATES:
        render_state(s)


if __name__ == "__main__":
    main()
