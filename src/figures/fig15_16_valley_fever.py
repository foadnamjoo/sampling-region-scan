from __future__ import annotations

"""Figs 15 & 16 — California Valley Fever (Appendix C, real-data validation).

  Fig 15  outputs/valley_fever/vf_pjd_vs_k.pdf
          Point Jaccard distance to the San Joaquin Valley ground truth as a
          function of k.  Centroid (deterministic, n=1) plotted as a single
          red point; Geom-k (k ∈ {1, 5, 10, 20, 50}, n=20 each) plotted as a
          continuous purple curve with a ±1 std band; the irregular-shape
          floor at JD = 0.412 is shown as a dashed black reference line.

  Fig 16  outputs/valley_fever/vf_smr_choropleth.pdf
          Standardised morbidity ratio (observed / expected cases) by California
          county for 2014-2018.  SJV-8 ground truth outlined dashed green; the
          discovered Centroid (k=0) and Geom-50 rectangles overlaid.  Analysis
          stays in EPSG:3310; only the display is reprojected to EPSG:4326 so
          axes read in degrees.

Headline numbers (hard-coded for Fig 15 — no recompute, matches the headline
sweep cached in valley_fever_run()):
    Centroid                 PJD = 0.7165   (n = 1, deterministic)
    Geom k=1, 5, 10, 20, 50  PJD mean ± std over 20 trials
    shape-family floor       JD  = 0.4119

Fig 16 does a single-pass recompute (one k=0 + one Geom-50 trial, seed = 7)
to obtain the two discovered rectangles drawn on the choropleth.
"""

# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

import paper_plots as pp  # noqa: E402
from paper_plots import STYLE_V9  # noqa: E402
from run_experiment_real import (  # noqa: E402
    load_california_counties, load_cocci_window, attach_cocci_to_counties,
    geom_k_points_real, discover_rect_real, internal_smr,
    _vf_pairwise_union, SJV_8,
)

OUT_DIR = OUTPUTS / "valley_fever"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CENTROID_COLOR = STYLE_V9["Centroid"]["color"]      # "red"
GEOM50_COLOR   = STYLE_V9["Geom 50"]["color"]       # "darkmagenta"
GEOM50_MARK    = STYLE_V9["Geom 50"]["marker"]      # "s"
GEOM50_MS      = STYLE_V9["Geom 50"]["ms"]
GEOM50_LW      = STYLE_V9["Geom 50"]["lw"]
# Brighter purple for the Geom-50 rectangle in Fig 16 — darkmagenta gets
# muddled against the warm YlOrRd fill; darkviolet pops over the deepest crimson.
GEOM50_BOX_COLOR = "#9400D3"

BASE_EDGE = "#9C9C9C"
SJV_GREEN = "#1B5E20"


def render_pjd_vs_k():
    """Fig 15 — PJD vs k on California, SJV-8 ground truth."""
    k_geom  = np.array([1, 5, 10, 20, 50], dtype=float)
    mu_geom = np.array([0.7089, 0.5440, 0.5062, 0.4784, 0.4481])
    sd_geom = np.array([0.0839, 0.0506, 0.0501, 0.0350, 0.0105])
    centroid_pjd = 0.7165
    floor = 0.4119
    k0_x = 0.5

    pp.apply_style_v9()
    plt.rcParams.update({
        "axes.labelsize":  14,
        "axes.titlesize":  14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    ax.fill_between(k_geom, mu_geom - sd_geom, mu_geom + sd_geom,
                    color=GEOM50_COLOR, alpha=0.18, lw=0)
    ax.plot(k_geom, mu_geom,
            color=GEOM50_COLOR, ls="-",
            marker=GEOM50_MARK, ms=GEOM50_MS, lw=GEOM50_LW,
            label="Geom-$k$ ($n = 20$)",
            zorder=3)

    ax.plot([k0_x], [centroid_pjd],
            color=CENTROID_COLOR, ls="none",
            marker="o", ms=10,
            mec=CENTROID_COLOR, mfc=CENTROID_COLOR, mew=1.4,
            label="Centroid (deterministic, $n = 1$)",
            zorder=5)

    ax.axhline(floor, color="black", lw=1.4, ls=(0, (5, 2)),
               label=f"shape floor (JD = {floor:.3f})")

    point_labels = [
        (k0_x,       centroid_pjd, "Centroid", CENTROID_COLOR),
        (k_geom[0],  mu_geom[0],   "Geom 1",   GEOM50_COLOR),
        (k_geom[1],  mu_geom[1],   "Geom 5",   GEOM50_COLOR),
        (k_geom[2],  mu_geom[2],   "Geom 10",  GEOM50_COLOR),
        (k_geom[3],  mu_geom[3],   "Geom 20",  GEOM50_COLOR),
        (k_geom[4],  mu_geom[4],   "Geom 50",  GEOM50_COLOR),
    ]
    for x, y, txt, color in point_labels:
        ax.annotate(txt, xy=(x, y),
                    xytext=(0, 12), textcoords="offset points",
                    color=color, fontsize=10, fontweight="bold",
                    ha="center", va="bottom",
                    path_effects=[pe.withStroke(linewidth=2.0,
                                                foreground="white")],
                    zorder=6)

    ax.set_xscale("log")
    xticks      = np.concatenate(([k0_x], k_geom))
    xticklabels = ["0"] + [str(int(v)) for v in k_geom]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("$k$ (sampled points per region)")
    ax.set_ylabel("Point Jaccard Distance")
    ax.set_ylim(0.35, 0.88)
    ax.set_title("Valley Fever (2014–2018), 20 trials")

    handles, labels = ax.get_legend_handles_labels()
    order_pref = ["Centroid (deterministic, $n = 1$)",
                  "Geom-$k$ ($n = 20$)",
                  f"shape floor (JD = {floor:.3f})"]
    ordered = [(h, l) for lp in order_pref
               for h, l in zip(handles, labels) if l == lp]
    hh, ll = zip(*ordered)
    ax.legend(hh, ll, loc="upper right")

    fig.tight_layout()
    out = OUT_DIR / "vf_pjd_vs_k.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def render_choropleth():
    """Fig 16 — California SMR choropleth with SJV-8 outline and discovered rects."""
    import geopandas as gpd
    print("[setup] CA counties + cocci 2014-2018 + SJV-8 ...", flush=True)
    gdf_geom = load_california_counties()
    cases, pop = load_cocci_window(2014, 2018)
    gdf = attach_cocci_to_counties(gdf_geom, cases, pop)
    shp_names = set(gdf["NAME"])
    sjv_polys = [gdf[gdf["NAME"] == n].geometry.iloc[0]
                 for n in SJV_8 if n in shp_names]
    sjv = _vf_pairwise_union(sjv_polys)

    m = gdf["m"].astype(float).to_numpy()
    b = gdf["b"].astype(float).to_numpy()

    rng = np.random.default_rng(7)
    c0, mpp0, bpp0 = geom_k_points_real(gdf, m, b, 0, rng)
    rect_k0 = discover_rect_real(c0, mpp0, bpp0, grid_res=100)

    rng50 = np.random.default_rng(7 + 1000 * 50 + 0)
    c50, mpp50, bpp50 = geom_k_points_real(gdf, m, b, 50, rng50)
    rect_k50 = discover_rect_real(c50, mpp50, bpp50, grid_res=100)

    _, _, smr_k0  = internal_smr(rect_k0,  c0,  mpp0,  bpp0)
    _, _, smr_k50 = internal_smr(rect_k50, c50, mpp50, bpp50)
    print(f"  k=0  rect internal SMR = {smr_k0:.2f}")
    print(f"  k=50 rect internal SMR = {smr_k50:.2f}")

    gdf_disp = gdf.to_crs("EPSG:4326")
    sjv_disp = (gpd.GeoSeries([sjv], crs="EPSG:3310")
                  .to_crs("EPSG:4326")
                  .iloc[0])
    rect_k0_disp = (gpd.GeoSeries([rect_k0], crs="EPSG:3310")
                      .to_crs("EPSG:4326")
                      .iloc[0])
    rect_k50_disp = (gpd.GeoSeries([rect_k50], crs="EPSG:3310")
                       .to_crs("EPSG:4326")
                       .iloc[0])

    pp.apply_style_v9()
    mpl.rcParams.update({
        "axes.facecolor": "white",
        "axes.grid":      False,
    })

    fig, ax = plt.subplots(figsize=(8.0, 9.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)

    smr = (gdf["m"] / gdf["b"].replace(0, np.nan)).fillna(0.0)
    g = gdf_disp.copy(); g["SMR"] = smr.to_numpy()

    # Two-stage SMR ramp: low signal stays in a mild yellow gradient; a
    # deliberate visual break at SMR = 2.66 separates "background" from
    # "genuine excess", and the three high-end bins (orange-red, deep red,
    # near-black crimson) keep the SJV high-incidence counties distinguishable
    # rather than collapsing into three near-identical dark reds.
    import matplotlib.colors as _mcolors
    _class_colors = [
        "#FFFFE5",   # 0.00 – 0.55
        "#FFF7BC",   # 0.55 – 1.64
        "#FEE391",   # 1.64 – 2.66
        "#FC4E2A",   # 2.66 – 3.95
        "#B10026",   # 3.95 – 8.83
        "#4D0011",   # 8.83 – 17.29
    ]
    custom_cmap = _mcolors.ListedColormap(_class_colors, name="VF_YlOrRd_v2")

    g.plot(column="SMR", ax=ax, cmap=custom_cmap,
           scheme="NaturalBreaks", k=6,
           edgecolor=BASE_EDGE, linewidth=0.4,
           rasterized=True,
           legend=True,
           legend_kwds={
               "loc":            "lower left",
               "title":          "SMR  (observed / expected cases)",
               "fontsize":       9,
               "title_fontsize": 10,
               "frameon":        True,
               "framealpha":     0.92,
               "edgecolor":      "#888888",
           })
    smr_legend = ax.get_legend()
    smr_class_handles = list(smr_legend.legend_handles)
    smr_class_labels  = [t.get_text() for t in smr_legend.get_texts()]
    smr_legend.remove()

    polys = list(sjv_disp.geoms) if hasattr(sjv_disp, "geoms") else [sjv_disp]
    sjv_handle_label = "San Joaquin Valley (ground truth $S^*$)"
    first = True
    for p in polys:
        ax.plot(*p.exterior.xy, color=SJV_GREEN, lw=2.2, ls="--",
                label=sjv_handle_label if first else None)
        first = False

    def _add(rect, color, label, lw=2.4, ls="-", halo=False):
        minx, miny, maxx, maxy = rect.bounds
        patch = mpatches.Rectangle(
            (minx, miny), maxx - minx, maxy - miny,
            facecolor="none", edgecolor=color, lw=lw, ls=ls, label=label)
        if halo:
            patch.set_path_effects([
                pe.withStroke(linewidth=lw + 1.6, foreground="white"),
            ])
        ax.add_patch(patch)

    _add(rect_k0_disp,  CENTROID_COLOR,   "Centroid", halo=True)
    _add(rect_k50_disp, GEOM50_BOX_COLOR, "Geom-50",  halo=False)

    minx, miny, maxx, maxy = gdf_disp.total_bounds
    span_x = maxx - minx; span_y = maxy - miny
    ax.set_xlim(minx - 0.02 * span_x, maxx + 0.02 * span_x)
    ax.set_ylim(miny - 0.02 * span_y, maxy + 0.02 * span_y)
    ax.set_aspect("equal")

    lon_ticks = np.arange(int(np.ceil(minx)), int(np.floor(maxx)) + 1, 2)
    lat_ticks = np.arange(int(np.ceil(miny)), int(np.floor(maxy)) + 1, 2)
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)
    ax.set_xticklabels([f"{int(v)}°" for v in lon_ticks])
    ax.set_yticklabels([f"{int(v)}°" for v in lat_ticks])
    ax.tick_params(axis="both", which="major", labelsize=10,
                   length=2.5, width=0.5, pad=1.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.margins(0, 0)

    method_handles = [
        mpatches.Patch(facecolor="none", edgecolor=SJV_GREEN,        lw=2.2,
                       ls="--", label=sjv_handle_label),
        mpatches.Patch(facecolor="none", edgecolor=CENTROID_COLOR,   lw=2.4,
                       label="Centroid"),
        mpatches.Patch(facecolor="none", edgecolor=GEOM50_BOX_COLOR, lw=2.4,
                       label="Geom-50"),
    ]
    spacer = mpatches.Patch(facecolor="none", edgecolor="none", label="")
    smr_title_handle = mpatches.Patch(facecolor="none", edgecolor="none",
                                       label="SMR (observed / expected cases)")
    merged_handles = method_handles + [spacer, smr_title_handle] + smr_class_handles
    merged_labels  = [h.get_label() for h in method_handles] + \
                     ["", smr_title_handle.get_label()] + smr_class_labels
    leg = ax.legend(merged_handles, merged_labels,
                    loc="upper right", frameon=True,
                    framealpha=0.92, edgecolor="#888888",
                    fontsize=9.5, handlelength=2.0, labelspacing=0.45)
    smr_title_idx = len(method_handles) + 1
    leg.get_texts()[smr_title_idx].set_fontweight("bold")

    fig.tight_layout()
    out = OUT_DIR / "vf_smr_choropleth.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    render_pjd_vs_k()
    render_choropleth()
