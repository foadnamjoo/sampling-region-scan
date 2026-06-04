from __future__ import annotations

"""Render Fig 14 (= Fig 10 in render_all.py) as a 2x2 from the new Grid(100)
Georgia ablation pkls, plus a Geom-50 4-way comparison plot.

Mirrors render_all.render_fig10_2x2() exactly except:
  - loads pkls from buchin_attempt/
  - writes a NEW file in buchin_attempt/
  - no suptitle; single figure-level legend at the top

Outputs:
  fig14_georgia_ablation_2x2_grid100.{pdf,png}
  fig14_geom50_4way_grid100.{pdf,png}
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

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from shapely.geometry import Polygon

import paper_plots as pp

BU = OUTPUTS

GA_SHP = DATA / ("georgia/"
                 "GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp")
GA_TARGET = Polygon([(-85.0, 31.0), (-85.0, 32.89),
                     (-83.61, 32.89), (-83.61, 31.0)])


def _draw_georgia_inset(ax, inset_pos=(0.66, 0.45, 0.33, 0.53)):
    """Georgia map inset in the top-right corner of `ax` with the planted
    target rectangle highlighted (muted brick-red, thin dashed). Mirrors
    the style used in render_state_grid100_with_inset.py."""
    gdf = gpd.read_file(GA_SHP).to_crs("EPSG:4326")
    inset = ax.inset_axes(list(inset_pos))
    gdf.plot(ax=inset, color="#FAFAFA", edgecolor="#5A5A5A", linewidth=0.25,
             rasterized=True)

    tx0, ty0, tx1, ty1 = GA_TARGET.bounds
    inset.add_patch(mpatches.Rectangle(
        (tx0, ty0), tx1 - tx0, ty1 - ty0,
        edgecolor="#9C2A2A", facecolor=(0.72, 0.20, 0.20, 0.10),
        lw=1.0, ls=(0, (4, 2))))

    inset.set_xticks([-85, -83, -81])
    inset.set_yticks([31, 33, 35])
    inset.tick_params(axis="both", which="major", labelsize=6,
                       length=2.0, width=0.4, pad=1.2, colors="#555555")
    inset.set_facecolor("white")
    for sp in inset.spines.values():
        sp.set_edgecolor("#AAAAAA"); sp.set_linewidth(0.5)
    inset.set_aspect("equal")
    inset.set_anchor("NE")


def _load(name: str) -> dict:
    with open(BU / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 2x2 ablation panel — one shared figure-level legend
# ---------------------------------------------------------------------------

def render_fig14_grid100_2x2() -> None:
    pp.apply_style_v9()
    # Larger fonts so the 2x2 panel reads at the same scale as the
    # Geom-50 4-way figure when both are placed in the same figure*.
    plt.rcParams.update({
        "axes.labelsize":  14.5,
        "axes.titlesize":  15.5,
        "axes.titlepad":   10,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "legend.fontsize": 12.0,
    })
    u = _load("georgia_ablation_uniform_grid100")
    w = _load("georgia_ablation_weighted_grid100")
    pq = np.asarray(u["pq_diff"])
    assert np.allclose(pq, np.asarray(w["pq_diff"]))

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.6),
                              sharex=True, sharey=True)
    panels = [
        (axes[0, 0], u["point_jaccard"], "Uniform Point Sampling",  "Point Jaccard Distance"),
        (axes[0, 1], w["point_jaccard"], "Weighted Point Sampling", "Point Jaccard Distance"),
        (axes[1, 0], u["area_jaccard"],  None,                      "Area Jaccard Distance"),
        (axes[1, 1], w["area_jaccard"],  None,                      "Area Jaccard Distance"),
    ]
    for ax, data, title, ylabel in panels:
        pp.plot_jaccard_vs_pq(ax, data, pq, show_band=True, band_alpha=0.22)
        if title:
            ax.set_title(title)
        ax.set_ylabel(ylabel if ax in (axes[0, 0], axes[1, 0]) else "")
        ax.set_xlabel(r"$p - q$ difference" if ax in (axes[1, 0], axes[1, 1]) else "")
        # Sparser x-ticks for readability (every 0.10 instead of 0.05).
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        ax.set_xticklabels([f"{v:.1f}" for v in
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    # Hide inner tick labels so the panels sit flush.
    for a in (axes[0, 0], axes[0, 1]):
        plt.setp(a.get_xticklabels(), visible=False)
    for a in (axes[0, 1], axes[1, 1]):
        plt.setp(a.get_yticklabels(), visible=False)

    # Single legend INSIDE the top-right panel only. Drop the per-axes
    # legends that plot_jaccard_vs_pq added on the other three.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax in axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    axes[0, 1].legend(handles, labels,
                      loc="upper right",
                      fontsize=11.5, frameon=True,
                      framealpha=0.92, edgecolor="#CCCCCC",
                      handlelength=2.4, handletextpad=0.55,
                      labelspacing=0.45,
                      borderpad=0.5)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.075,
                        wspace=0.06, hspace=0.07)

    for ext in ("pdf", "png"):
        out = BU / f"fig14_georgia_ablation_2x2_grid100.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Geom-50 4-way comparison: {uniform, weighted} × {point, area}
# ---------------------------------------------------------------------------

def render_geom50_4way(
    zoom_x: tuple[float, float] = (0.15, 0.50),
    band_alpha_main: float = 0.14,
    band_alpha_inset: float = 0.09,
    y_pad_frac: float = 0.06,
) -> None:
    """Geom-50 across {uniform, weighted} sampling × {Point, Area} Jaccard.

    Publication-style figure. The main panel shows the full p−q sweep
    for context; an inset magnifies the post-transition regime where
    the four curves separate, with translucent bands and emphasised
    mean lines so the ordering is unambiguous.

    Parameters
    ----------
    zoom_x
        p−q window for the inset.
    band_alpha_main / band_alpha_inset
        Confidence-band opacity in the main / inset panel respectively.
        The inset uses a lower alpha so mean lines stay legible.
    y_pad_frac
        Fractional padding added to the data-driven inset y-limits.
    """
    u = _load("georgia_ablation_uniform_grid100")
    w = _load("georgia_ablation_weighted_grid100")
    pq = np.asarray(u["pq_diff"])

    # --- Publication style matching paper Fig 14 (seaborn-darkgrid
    # background: lavender panel, white gridlines, no tick marks). ---
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset":   "stixsans",
        "axes.facecolor":     "#EAEAF2",
        "figure.facecolor":   "white",
        "axes.edgecolor":     "white",
        "axes.linewidth":     1.0,
        "axes.labelcolor":    "#222222",
        "axes.labelsize":     14.5,
        "axes.titlesize":     15.5,
        "axes.titlepad":      10,
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "grid.color":         "white",
        "grid.linewidth":     1.0,
        "grid.linestyle":     "-",
        "xtick.color":        "#222222",
        "ytick.color":        "#222222",
        "xtick.labelsize":    12.0,
        "ytick.labelsize":    12.0,
        "xtick.major.size":   0,
        "ytick.major.size":   0,
        "xtick.major.width":  0,
        "ytick.major.width":  0,
        "legend.frameon":     False,
        "savefig.facecolor":  "white",
    })

    # Original purple/magenta palette: sampling by hue (Uniform = purples,
    # Weighted = pinks); metric by linestyle + marker fill (Point JD =
    # solid filled, Area JD = dashed open).
    curves = [
        ("Uniform",  "Point Jaccard", u["point_jaccard"]["Geom 50"],
            dict(color="#311B92", marker="s", ls="-",  ms=6.5, fill=True)),
        ("Uniform",  "Area Jaccard",  u["area_jaccard"]["Geom 50"],
            dict(color="#7B1FA2", marker="^", ls="--", ms=7.0, fill=True)),
        ("Weighted", "Point Jaccard", w["point_jaccard"]["Geom 50"],
            dict(color="#C2185B", marker="D", ls="-",  ms=6.0, fill=True)),
        ("Weighted", "Area Jaccard",  w["area_jaccard"]["Geom 50"],
            dict(color="#E91E63", marker="o", ls="--", ms=6.5, fill=True)),
    ]

    fig, ax = plt.subplots(figsize=(14.5, 5.8))

    # --- Inset axes for the zoom magnifier. Positioned so the card
    # OVERLAPS the right portion of the main plot and extends past the
    # main plot's right edge (axes-fraction x > 1.0). High zorder so it
    # sits on top in the overlap region.
    ax2 = ax.inset_axes([0.40, 0.25, 0.75, 0.72])
    ax2.set_zorder(10)
    ax2.patch.set_zorder(10)

    def _plot(axx: plt.Axes, alpha: float, is_inset: bool):
        """Draw the 4 mean curves + bands onto `axx`."""
        for _samp, _metr, arr, style in curves:
            arr = np.asarray(arr)
            mean = arr.mean(axis=0); std = arr.std(axis=0)
            c = style["color"]
            # Bands sit beneath everything else; mean lines on top.
            axx.fill_between(pq, mean - std, mean + std,
                             color=c, alpha=alpha, lw=0, zorder=2)
            mfc = c if style["fill"] else "white"
            axx.plot(pq, mean,
                     color=c, ls=style["ls"], marker=style["marker"],
                     markersize=style["ms"], mfc=mfc, mec=c, mew=1.3,
                     lw=1.9, zorder=5,
                     label=None if is_inset
                            else f"{_samp} {_metr}")

    _plot(ax,  alpha=band_alpha_main,  is_inset=False)
    _plot(ax2, alpha=band_alpha_inset, is_inset=True)

    # --- Main axes chrome (seaborn-darkgrid look, matches paper Fig 14). ---
    ax.set_xlabel(r"$p - q$ difference")
    ax.set_ylabel("Jaccard distance")
    ax.set_title("Georgia ablation — Geom-50 across sampling × metric")
    ax.set_xlim(pq.min(), pq.max())
    ax.set_ylim(-0.02, 1.02)
    ax.set_facecolor("#EAEAF2")

    # --- Inset chrome. Data-driven y-limits over the zoom window so the
    # curves fill the panel; tick labels stay legible. ---
    zx0, zx1 = zoom_x
    mask = (pq >= zx0) & (pq <= zx1)
    means_in_window = np.array([
        np.asarray(arr).mean(axis=0)[mask]
        for _, _, arr, _ in curves])
    y_lo = float(means_in_window.min())
    y_hi = float(means_in_window.max())
    y_pad = (y_hi - y_lo) * y_pad_frac
    # Pad x a touch beyond the data endpoints so the leftmost/rightmost
    # marker icons sit fully inside the inset instead of being clipped.
    x_pad = (zx1 - zx0) * 0.06
    ax2.set_xlim(zx0 - x_pad, zx1 + x_pad)
    ax2.set_ylim(y_lo - y_pad, y_hi + y_pad)

    # Round ticks to round numbers within the data-driven range.
    def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
        span = hi - lo
        candidates = [0.005, 0.01, 0.02, 0.025, 0.05, 0.1]
        step = next(c for c in candidates if span / c <= n + 1)
        start = np.ceil(lo / step) * step
        ticks = []
        v = start
        while v <= hi + 1e-9:
            ticks.append(round(v, 4)); v += step
        return ticks

    ax2.set_xticks(_nice_ticks(zx0, zx1, n=5))
    ax2.set_yticks(_nice_ticks(y_lo - y_pad, y_hi + y_pad, n=5))
    ax2.tick_params(axis="both", labelsize=11.5, length=0, width=0,
                    pad=2)
    # Opaque whiter-grey card. The card axes is given a high zorder so
    # it draws on top of the main axes (including any connector lines
    # parented to the main axes), giving "card sits on top of everything"
    # behaviour even where lines would otherwise cross.
    ax2.set_facecolor("#F6F6F9")
    ax2.patch.set_alpha(1.0)
    ax2.patch.set_zorder(3)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#444444")
        sp.set_linewidth(1.6)
        sp.set_linestyle((0, (4, 2.5)))
    ax2.grid(True, color="white", lw=1.0, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_zorder(20)   # draw on top of main axes (and its connectors)
    ax2.text(0.975, 0.965, "Magnified view of the marked region",
             transform=ax2.transAxes, fontsize=11.5, color="#555555",
             ha="right", va="top", style="italic")

    # --- Source rectangle on the main panel + connector lines to the
    # inset. Source rectangle: NO fill, dashed grey outline only — a
    # restrained "highlight" against the lavender panel. Connectors stay
    # solid so the link reads cleanly.
    rect_patch, conn_lines = ax.indicate_inset_zoom(
        ax2, edgecolor="#666666", alpha=1.0, linewidth=1.0)
    if rect_patch is not None:
        # Translucent darker-grey fill on the source rectangle, with a
        # thicker dashed border so it matches the magnified card's frame.
        rect_patch.set_alpha(None)
        rect_patch.set_facecolor((0.35, 0.35, 0.40, 0.22))
        rect_patch.set_edgecolor("#444444")
        rect_patch.set_linewidth(1.6)
        rect_patch.set_linestyle((0, (4, 2.5)))
    # Hide the auto-connectors — we draw our own four, one per corner.
    for line in conn_lines:
        if line is not None:
            line.set_visible(False)

    # No connector lines — the dashed source-region rectangle and the
    # matching dashed border around the magnified card carry the visual
    # link on their own (reinforced by the "Magnified view of the marked
    # region" caption inside the card).
    src_x0, src_x1 = ax2.get_xlim()
    src_y0, src_y1 = ax2.get_ylim()


    # --- Single frameless legend below the axes, one row. ---
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(handles, labels,
              loc="upper left", bbox_to_anchor=(0.0, -0.14),
              ncol=4, fontsize=12, frameon=False,
              handlelength=2.8, handletextpad=0.6, columnspacing=2.2)
    # Reserve room on the right; trim left margin so everything shifts
    # left and the figure uses its width more efficiently.
    fig.subplots_adjust(left=0.035, right=0.55, top=0.92, bottom=0.16)

    # --- Georgia map in the right-side white space, with the planted
    # target rectangle highlighted. Sits in the figure margin, clear of
    # the main plot and the zoomed card. ---
    # Match the main plot's vertical extent (bottom 0.16, top 0.92).
    # Wide bounding box so Georgia (aspect ~0.89) fills the full height.
    ga_ax = fig.add_axes([0.60, 0.16, 0.39, 0.76])
    ga_ax.set_zorder(15)
    ga_gdf = gpd.read_file(GA_SHP).to_crs("EPSG:4326")
    ga_gdf.plot(ax=ga_ax, color="#FAFAFA", edgecolor="#5A5A5A",
                linewidth=0.3, rasterized=True)
    tx0, ty0, tx1, ty1 = GA_TARGET.bounds
    ga_ax.add_patch(mpatches.Rectangle(
        (tx0, ty0), tx1 - tx0, ty1 - ty0,
        edgecolor="#9C2A2A", facecolor=(0.72, 0.20, 0.20, 0.12),
        lw=1.2, ls=(0, (4, 2))))
    ga_ax.set_xticks([-85, -83, -81])
    ga_ax.set_yticks([31, 33, 35])
    # Match the main plot's tick style: same labelsize, same colour,
    # same (zero) tick mark length.
    ga_ax.tick_params(axis="both", which="major", labelsize=12.0,
                       length=0, width=0, pad=2, colors="#222222")
    ga_ax.set_facecolor("white")
    for sp in ga_ax.spines.values():
        sp.set_edgecolor("#AAAAAA"); sp.set_linewidth(0.6)
    ga_ax.set_aspect("equal")
    ga_ax.set_anchor("C")
    # Move latitude tick labels to the RIGHT edge of the map so they
    # don't fall into the inset zoom card area on the left.
    ga_ax.yaxis.tick_right()
    ga_ax.yaxis.set_label_position("right")
    ga_ax.text(0.5, 1.04, "Target region",
               transform=ga_ax.transAxes, fontsize=12.5,
               color="#222222", ha="center", va="bottom",
               style="italic")

    for ext in ("pdf", "png"):
        out = BU / f"fig14_geom50_4way_grid100.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)
    plt.rcdefaults()


if __name__ == "__main__":
    render_fig14_grid100_2x2()
    render_geom50_4way()
