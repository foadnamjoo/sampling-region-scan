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
GEOM50_COLOR   = STYLE_V9["Geom 50"]["color"]       # "darkmagenta"  (#8B008B)
GEOM50_MARK    = STYLE_V9["Geom 50"]["marker"]      # "s"
GEOM50_MS      = STYLE_V9["Geom 50"]["ms"]
GEOM50_LW      = STYLE_V9["Geom 50"]["lw"]
# The discovered-rectangle on the choropleth uses pure magenta (#FF00FF) so it
# pops against the dark-red Kern fill.  The Valley-Fever curve / band uses a
# midpoint between STYLE_V9's darkmagenta (#8B008B) and the box's magenta —
# so the curve ties visually to its Geom-50 rectangle on the map.
GEOM50_BOX_COLOR = "magenta"          # rectangle color  → #FF00FF
CURVE_PURPLE     = "#C500C5"          # midpoint #8B008B ↔ #FF00FF

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


def render_combined():
    """Combined Fig 13 — PJD curve spanning the full figure with the California
    SMR choropleth overlaid as an OPAQUE inset in the upper-right empty space
    above the descending curve.  Single legend row at the bottom.  Designed
    to render at \\columnwidth in the SIGSPATIAL two-column layout.
    """
    import geopandas as gpd
    from matplotlib.lines import Line2D

    # ---- shared setup ----------------------------------------------------
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
                  .to_crs("EPSG:4326").iloc[0])
    rect_k0_disp = (gpd.GeoSeries([rect_k0], crs="EPSG:3310")
                      .to_crs("EPSG:4326").iloc[0])
    rect_k50_disp = (gpd.GeoSeries([rect_k50], crs="EPSG:3310")
                       .to_crs("EPSG:4326").iloc[0])

    # ---- figure designed for single-column (~3.3in) inclusion -----------
    # Font sizes match the Utah / California methods curves (v9 defaults:
    # label 11, ticks 9, legend 9) so the paper's plots read consistently.
    pp.apply_style_v9()

    # Wider + shorter aspect to save vertical page space; the methods legend
    # lives INSIDE the axes (no bottom strip needed) and California sits as
    # an inset overlay flush with the top-right corner of the curve plot.
    fig = plt.figure(figsize=(7.6, 3.4))
    ax_c = fig.add_subplot(111)
    # Inset (California): top-right corner of the inset is FLUSH with the
    # top-right corner of the curve plot (x0+width=1.0, y0+height=1.0).
    # Aspect chosen so California (≈1:1 in EPSG:4326 with equal-aspect) fills
    # the inset with minimal padding — the inset looks bigger because the map
    # itself fills the box rather than sitting in a sea of whitespace.
    # Lower edge stays above Geom 5's PJD (≈0.544 → axes-y 0.37).
    ax_m = ax_c.inset_axes([0.19, 0.10, 0.81, 0.90])

    # ===== LEFT — Geom-k curve ============================================
    k_geom  = np.array([1, 5, 10, 20, 50], dtype=float)
    mu_geom = np.array([0.7089, 0.5440, 0.5062, 0.4784, 0.4481])
    sd_geom = np.array([0.0839, 0.0506, 0.0501, 0.0350, 0.0105])
    centroid_pjd = 0.7165
    floor = 0.4119

    # ±std band over Geom-k (k=1..50).  CURVE_PURPLE is a midpoint between
    # darkmagenta and the box's magenta so the curve visually links to the
    # Geom-50 rectangle on the inset map.  zorder=2 keeps the band ABOVE the
    # grid-masking rectangle so the shaded band is visible across the full
    # k range (the mask only hides the white grid lines, not the band).
    ax_c.fill_between(k_geom, mu_geom - sd_geom, mu_geom + sd_geom,
                      color=CURVE_PURPLE, alpha=0.18, lw=0, zorder=2)
    ax_c.plot(k_geom, mu_geom,
              color=CURVE_PURPLE, ls="-",
              marker=GEOM50_MARK, ms=5, lw=1.5,
              label="Geom-$k$", zorder=3)
    # Centroid sits AT k=1, slightly offset so its red circle and Geom 1's
    # purple square don't sit exactly on top of each other.
    cent_x = 0.86
    ax_c.plot([cent_x], [centroid_pjd],
              color=CENTROID_COLOR, ls="none",
              marker="o", ms=8,
              mec=CENTROID_COLOR, mfc=CENTROID_COLOR, mew=1.0,
              label="Centroid", zorder=5)
    # Shape-floor line — drawn ONLY from the y-axis up to the Geom 50 point
    # (k=50) so the dashed line does not extend underneath the California map.
    ax_c.plot([0.7, k_geom[-1]], [floor, floor],
              color="black", lw=1.0, ls=(0, (5, 2)),
              zorder=3, clip_on=True)
    ax_c.text(0.75, floor + 0.012,
              f"shape floor (JD = {floor:.3f})",
              color="black", fontsize=9, ha="left", va="bottom",
              path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
              zorder=4)

    # On-point labels: Centroid sits ABOVE its own marker (anchored at cent_x,
    # not at k_geom[0]); Geom 1 sits to the right of its marker.  Both labels
    # stay clear of the y-axis ticks.
    # Geom 5..50 placed BELOW their markers (the inset overlay covers the
    # area above the descending curve, so above-marker labels would be hidden).
    for x, y, txt, color, dx, dy, ha, va in [
        (cent_x,    centroid_pjd, "Centroid", CENTROID_COLOR,  0,  8,  "center", "bottom"),
        (k_geom[0], mu_geom[0],   "Geom 1",   CURVE_PURPLE,    6,  0,  "left",   "center"),
        (k_geom[1], mu_geom[1],   "Geom 5",   CURVE_PURPLE,    0, -7,  "center", "top"),
        (k_geom[2], mu_geom[2],   "Geom 10",  CURVE_PURPLE,    0, -7,  "center", "top"),
        (k_geom[3], mu_geom[3],   "Geom 20",  CURVE_PURPLE,    0, -7,  "center", "top"),
        (k_geom[4], mu_geom[4],   "Geom 50",  CURVE_PURPLE,    6, -3,  "left",   "top"),
    ]:
        ax_c.annotate(txt, xy=(x, y),
                      xytext=(dx, dy), textcoords="offset points",
                      color=color, fontsize=9, fontweight="bold",
                      ha=ha, va=va,
                      path_effects=[pe.withStroke(linewidth=1.8,
                                                  foreground="white")],
                      zorder=6)

    ax_c.set_xscale("log")
    ax_c.set_xticks(list(k_geom))
    ax_c.set_xticklabels([str(int(v)) for v in k_geom])
    ax_c.minorticks_off()
    ax_c.set_xlim(0.7, 600)
    # Keep the white grid in the main curve area, but blank it out around
    # California so the grid lines stop where they meet the ±std band.
    # A single panel-colored polygon mask follows the band's upper edge from
    # Geom 10 → Geom 20 → Geom 50, then covers everything to the right and
    # above (California's footprint + the strip past x=50 → hides the y=0.4
    # line there).  Vertical grid lines at x=10/20/50 stay visible from the
    # bottom of the plot up to where they meet the band.
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    mask_verts = [
        (0.36,  1.00),     # top-left
        (1.00,  1.00),     # top-right
        (1.00,  0.00),     # bottom-right
        (0.632, 0.00),     # bottom edge at x=50 (axes-x of Geom 50)
        (0.632, 0.157),    # up to Geom 50's band upper edge
        (0.497, 0.267),    # Geom 20's band upper edge
        (0.394, 0.353),    # Geom 10's band upper edge
        (0.36,  0.378),    # left edge, just at the band where it crosses axes-x 0.36
    ]
    codes = ([Path.MOVETO]
             + [Path.LINETO] * (len(mask_verts) - 1)
             + [Path.CLOSEPOLY])
    grid_mask_main = PathPatch(
        Path(mask_verts + [mask_verts[0]], codes),
        facecolor="#EAEAF2", edgecolor="none",
        transform=ax_c.transAxes, zorder=1.5)
    ax_c.add_patch(grid_mask_main)
    # Small strip around x=5 above PJD≈0.73 — trims the top portion of the
    # x=5 vertical grid line where it would cut through the upper curve area
    # near the Centroid / Geom 1 labels.
    grid_mask_C = mpatches.Rectangle(
        (0.27, 0.70), 0.05, 0.30,
        transform=ax_c.transAxes,
        facecolor="#EAEAF2",
        edgecolor="none",
        zorder=1.5)
    ax_c.add_patch(grid_mask_C)
    ax_c.set_xlabel("$k$ (sampled points per region)")
    ax_c.set_ylabel("Point Jaccard Distance")
    ax_c.set_ylim(0.38, 0.88)
    ax_c.set_title("California Valley Fever")
    # Curve legend lives at the BOTTOM of the figure, single row, alongside
    # the SJV entry — built once at the end after the inset is populated.

    # ===== INSET OVERLAY — SMR choropleth (transparent background) ========
    # Transparent inset background so the curve underneath remains visible
    # through the empty (non-California) area of the inset box.
    ax_m.set_facecolor("none")
    ax_m.patch.set_alpha(0)
    ax_m.set_zorder(20)
    # No inset border — the inset blends into the curve plot.
    for sp in ax_m.spines.values():
        sp.set_visible(False)

    smr = (gdf["m"] / gdf["b"].replace(0, np.nan)).fillna(0.0)
    g = gdf_disp.copy(); g["SMR"] = smr.to_numpy()

    import matplotlib.colors as _mcolors
    _class_colors = [
        "#FFFFE5", "#FFF7BC", "#FEE391",
        "#FC4E2A", "#B10026", "#4D0011",
    ]
    custom_cmap = _mcolors.ListedColormap(_class_colors, name="VF_YlOrRd_v2")

    # geopandas draws + creates an auto-legend; we capture handles and discard
    # the auto-legend, then build one merged frameless legend in the top-right.
    g.plot(column="SMR", ax=ax_m, cmap=custom_cmap,
           scheme="NaturalBreaks", k=6,
           edgecolor=BASE_EDGE, linewidth=0.35,
           rasterized=True,
           legend=True,
           legend_kwds={"loc": "lower left", "fontsize": 0.1,
                        "frameon": False})
    smr_legend = ax_m.get_legend()
    smr_class_handles = list(smr_legend.legend_handles)
    smr_class_labels  = [t.get_text() for t in smr_legend.get_texts()]
    smr_legend.remove()

    polys = list(sjv_disp.geoms) if hasattr(sjv_disp, "geoms") else [sjv_disp]
    sjv_handle_label = "San Joaquin Valley ($S^*$)"
    first = True
    for p in polys:
        ax_m.plot(*p.exterior.xy, color=SJV_GREEN, lw=1.8, ls="--",
                  label=sjv_handle_label if first else None)
        first = False

    def _add(rect, color, label, lw=2.0, halo=False):
        minx, miny, maxx, maxy = rect.bounds
        patch = mpatches.Rectangle(
            (minx, miny), maxx - minx, maxy - miny,
            facecolor="none", edgecolor=color, lw=lw, label=label)
        if halo:
            patch.set_path_effects([
                pe.withStroke(linewidth=lw + 1.2, foreground="white"),
            ])
        ax_m.add_patch(patch)

    _add(rect_k0_disp,  CENTROID_COLOR,   "Centroid", halo=True)
    _add(rect_k50_disp, GEOM50_BOX_COLOR, "Geom-50")

    minx, miny, maxx, maxy = gdf_disp.total_bounds
    span_x = maxx - minx; span_y = maxy - miny
    # Pad left and bottom only — California's right and top edges sit flush
    # against the inset's right/top so the state hugs the curve plot's
    # right edge.
    ax_m.set_xlim(minx - 0.02 * span_x, maxx)
    ax_m.set_ylim(miny, maxy)
    ax_m.set_aspect("equal")
    # Anchor California to the EAST (right) side of the inset box so its
    # right edge sits flush against the curve plot's right edge — no
    # horizontal padding on the right.
    ax_m.set_anchor("E")

    # No axis ticks, labels, or grid on the inset — California floats on the
    # transparent inset over the curve's grey background unencumbered.
    ax_m.set_xticks([]); ax_m.set_xticklabels([])
    ax_m.set_yticks([]); ax_m.set_yticklabels([])
    ax_m.set_xlabel(""); ax_m.set_ylabel("")
    ax_m.grid(False)
    ax_m.margins(0, 0)

    # ----- SMR class legend stays ON the inset map -------------------------
    # markerfirst=False puts the colour swatch on the RIGHT side of each row
    # and the bin range on the LEFT; alignment="right" right-aligns the title
    # over the column.  Combined with borderaxespad=0, the whole block hugs
    # the inset's right edge.
    smr_leg = ax_m.legend(handles=smr_class_handles, labels=smr_class_labels,
                          loc="upper right",
                          bbox_to_anchor=(1.0, 1.0),
                          title="Morbidity ratio (obs / exp)",
                          title_fontsize=9, fontsize=9,
                          frameon=False,
                          handlelength=1.4, handletextpad=0.5,
                          labelspacing=0.55,
                          borderaxespad=0.0,
                          markerfirst=False,
                          alignment="right")
    smr_leg.get_title().set_fontweight("bold")
    ax_m.add_artist(smr_leg)

    # ===== SINGLE-ROW legend at the bottom of the figure ===================
    # Legend handles are RECTANGLE OUTLINES — representing the discovered
    # rectangles drawn on the California map (Centroid k=0 rect in red,
    # Geom-50 rect in darkviolet, SJV S* outline in green dashed).
    centroid_rect_handle = mpatches.Patch(facecolor="none",
                                           edgecolor=CENTROID_COLOR,
                                           linewidth=2.0,
                                           label="Centroid")
    geom50_rect_handle = mpatches.Patch(facecolor="none",
                                         edgecolor=GEOM50_BOX_COLOR,
                                         linewidth=2.0,
                                         label="Geom-50")
    sjv_handle = mpatches.Patch(facecolor="none",
                                 edgecolor=SJV_GREEN, linewidth=1.8,
                                 linestyle="--",
                                 label="Ground truth ($S^*$)")
    # Methods legend INSIDE the curve axes — lower-left, in the empty area
    # between the descending band and the shape-floor line (above the floor
    # so it doesn't collide with the inline "shape floor" text label).
    ax_c.legend(handles=[centroid_rect_handle, geom50_rect_handle, sjv_handle],
                loc="lower left", bbox_to_anchor=(0.0, 0.17),
                ncol=3, frameon=False, fontsize=9,
                handlelength=1.6, columnspacing=1.6,
                handletextpad=0.5, borderaxespad=0.4)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.13)
    out = OUT_DIR / "vf_combined.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    render_pjd_vs_k()
    render_choropleth()
    render_combined()
