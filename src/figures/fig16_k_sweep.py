
# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------
"""Render every paper figure in v8 style from the cached pickles.

Produces in new_PlotsPDF/:
  - nyc_v8.pdf, utah_v8.pdf, california_v8.pdf, usa_v8.pdf      (Figs 3, 4, 5, 6)
  - georgia_ablation_v8.pdf                                     (Fig 10, single panel)
  - arkansas_10_v8.pdf                                           (Fig 9; pairs with arkansas_30_v8.pdf)
  - k_sweep_v8.pdf                                               (new signal-boost: PJD vs k)
  - geom50_cross_cut_v8.pdf                                      (Fig 11)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(ROOT / "scripts"))
import paper_plots as pp  # noqa: E402

CACHE = ROOT / "cached_data"
OUT = ROOT / "new_PlotsPDF"

FIGSIZE = (3.45, 2.65)  # tight two-col-half size (v8)
PAPER_FIGSIZE = (8.0, 5.0)  # matches the seaborn screenshot proportion (v9)

# Multi-seed merge: seeds we re-run with to smooth out single-draw noise.
# The original run uses seed 7 and saves as `{name}.pkl`; rerun seeds use
# `{name}_seed{N}.pkl`.
MULTI_SEEDS = (7, 31, 67)


def _load_methods_multi_seed(base_name: str, seeds=MULTI_SEEDS) -> dict:
    """Concatenate per-method trials across seed pickles for `base_name`.

    Falls back gracefully: if only some seed pickles exist, uses what's there.
    The combined `methods[name]` list is a (sum-of-trials × n_pq) matrix.
    """
    pkgs = []
    for s in seeds:
        p = CACHE / (f"{base_name}.pkl" if s == 7 else f"{base_name}_seed{s}.pkl")
        if p.exists():
            with open(p, "rb") as f:
                pkgs.append(pickle.load(f))
    if not pkgs:
        raise FileNotFoundError(f"no pickles found for {base_name}")
    out = {m: [] for m in pkgs[0]["methods"]}
    for pkg in pkgs:
        for m, trials in pkg["methods"].items():
            out[m].extend(trials)
    return {
        "methods":  out,
        "pq_diff":  pkgs[0]["pq_diff"],
        "n_trials": sum(p.get("n_trials", len(p["methods"]["Centroid"])) for p in pkgs),
        "seeds":    [p.get("seed") for p in pkgs],
    }


def _load_k_sweep_multi_seed(base_name: str, seeds=MULTI_SEEDS) -> dict:
    """Concatenate per-k trials across k-sweep seed pickles."""
    pkgs = []
    for s in seeds:
        p = CACHE / (f"{base_name}.pkl" if s == 7 else f"{base_name}_seed{s}.pkl")
        if p.exists():
            with open(p, "rb") as f:
                pkgs.append(pickle.load(f))
    if not pkgs:
        raise FileNotFoundError(f"no k-sweep pickles found for {base_name}")
    by_k = {k: [] for k in pkgs[0]["by_k"]}
    for pkg in pkgs:
        for k, trials in pkg["by_k"].items():
            by_k[k].extend(trials)
    return {
        "k_values": pkgs[0]["k_values"],
        "p_prob":   pkgs[0]["p_prob"],
        "pq_diff":  pkgs[0]["pq_diff"],
        "by_k":     by_k,
        "n_trials": sum(p.get("n_trials", 0) for p in pkgs),
        "seeds":    [p.get("seed") for p in pkgs],
    }


def _load_size_sweep_multi_seed(base_name: str, seeds=MULTI_SEEDS) -> dict:
    """Concatenate per-(target, method) trials across size-sweep seed pickles."""
    pkgs = []
    for s in seeds:
        p = CACHE / (f"{base_name}.pkl" if s == 7 else f"{base_name}_seed{s}.pkl")
        if p.exists():
            with open(p, "rb") as f:
                pkgs.append(pickle.load(f))
    if not pkgs:
        raise FileNotFoundError(f"no size-sweep pickles found for {base_name}")
    out = {m: [[] for _ in pkgs[0]["methods"][m]] for m in pkgs[0]["methods"]}
    for pkg in pkgs:
        for m, per_target_lists in pkg["methods"].items():
            for t_idx, trials in enumerate(per_target_lists):
                out[m][t_idx].extend(trials)
    return {
        "methods":  out,
        "area_pct": pkgs[0]["area_pct"],
        "p_prob":   pkgs[0]["p_prob"],
        "pq_diff":  pkgs[0]["pq_diff"],
        "n_trials": sum(p.get("n_trials", 0) for p in pkgs),
        "seeds":    [p.get("seed") for p in pkgs],
    }


def _load_georgia_ablation_multi_seed(base_name: str, seeds=MULTI_SEEDS) -> dict:
    """Concatenate point/area Jaccard trials across georgia-ablation seed pickles."""
    pkgs = []
    for s in seeds:
        p = CACHE / (f"{base_name}.pkl" if s == 7 else f"{base_name}_seed{s}.pkl")
        if p.exists():
            with open(p, "rb") as f:
                pkgs.append(pickle.load(f))
    if not pkgs:
        raise FileNotFoundError(f"no ablation pickles found for {base_name}")
    point_out = {m: [] for m in pkgs[0]["point_jaccard"]}
    area_out  = {m: [] for m in pkgs[0]["area_jaccard"]}
    for pkg in pkgs:
        for m, trials in pkg["point_jaccard"].items():
            point_out[m].extend(trials)
        for m, trials in pkg["area_jaccard"].items():
            area_out[m].extend(trials)
    return {
        "point_jaccard": point_out,
        "area_jaccard":  area_out,
        "pq_diff": pkgs[0]["pq_diff"],
        "n_trials": sum(p.get("n_trials", 0) for p in pkgs),
        "seeds": [p.get("seed") for p in pkgs],
        "sampling": pkgs[0].get("sampling"),
    }


def render_methods_curve(pkl_name: str, out_name: str, band_alpha: float = 0.25,
                          style: str = "v9", figsize: tuple = PAPER_FIGSIZE,
                          title: str | None = None) -> None:
    """Render a Jaccard-vs-pq curve figure. Default style v9 = paper original.
    Loads and merges trials across all available seeds in MULTI_SEEDS."""
    {"v8": pp.apply_style_v8, "v9": pp.apply_style_v9}[style]()
    base = pkl_name[:-4] if pkl_name.endswith(".pkl") else pkl_name
    pkg = _load_methods_multi_seed(base)
    fig, ax = plt.subplots(figsize=figsize)
    pp.plot_jaccard_vs_pq(ax, pkg["methods"], pkg["pq_diff"],
                          show_band=True, band_alpha=band_alpha)
    if title:
        ax.set_title(title)
    out = OUT / out_name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  (n_trials={pkg['n_trials']}, seeds={pkg['seeds']})")


def render_k_sweep() -> None:
    """PJD as a function of k at fixed pq, with one curve per dataset, merged
    across MULTI_SEEDS."""
    pp.apply_style_v9()
    datasets = [
        ("Arkansas",   "k_sweep_arkansas",   "red"),
        ("Utah",       "k_sweep_utah",       "darkorange"),
        ("California", "k_sweep_california", "olivedrab"),
        ("Georgia",    "k_sweep_georgia",    "darkmagenta"),
        ("NYC",        "k_sweep_nyc",        "deepskyblue"),
        ("USA",        "k_sweep_usa",        "blue"),
    ]
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    pq_label = None
    plotted = 0
    ark_ks = None
    ark_d0 = None
    for label, base, color in datasets:
        try:
            pkg = _load_k_sweep_multi_seed(base)
        except FileNotFoundError:
            print(f"  skipping {label}: no pickles")
            continue
        if pq_label is None:
            pq_label = pkg["pq_diff"]
        ks = pkg["k_values"]
        by_k = pkg["by_k"]
        means = np.array([np.mean(by_k[k]) for k in ks])
        stds  = np.array([np.std(by_k[k]) for k in ks])
        ax.fill_between(ks, means - stds, means + stds,
                        color=color, alpha=0.18, lw=0)
        ax.plot(ks, means, color=color, lw=2.0, marker="o", ms=5.5,
                label=label)
        if label == "Arkansas":
            # Capture for Theorem 1 overlay (calibrated at the smallest k).
            ark_ks = np.asarray(ks, dtype=float)
            ark_d0 = float(means[0])
        plotted += 1
    if plotted == 0:
        print("k_sweep: no pickles found; skipping render")
        plt.close(fig); return
    # Overlay Theorem 1's predicted asymptote: d_Jac(k) ~ C / sqrt(k).
    # Calibrate C using Arkansas at the smallest sampled k (so the curve
    # passes through the empirical point at k_min) and plot dashed black.
    if ark_d0 is not None and ark_ks is not None:
        k0 = float(ark_ks[0])
        pred = ark_d0 * np.sqrt(k0 / ark_ks)
        ax.plot(ark_ks, pred, color="black", lw=1.4, ls=(0, (5, 2)),
                label=r"Theorem 1: $\propto 1/\sqrt{k}$")
    ax.set_xlabel("$k$ (samples per region)")
    ax.set_ylabel("Point Jaccard Distance")
    ax.set_xscale("log")
    ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"PJD vs $k$ across datasets, pq difference = {pq_label:.2f}")
    ax.legend(loc="upper right")
    out = OUT / "k_sweep_v9.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def render_geom50_cross_cut() -> None:
    """Geom-50 curve from every state's methods experiment on one axes."""
    pp.apply_style_v9()
    sources = [
        ("Arkansas (30%)", "arkansas_30", "#1F77B4"),
        ("Utah",           "utah",        "#FF7F0E"),
        ("California",     "california",  "#2CA02C"),
        ("NYC",            "nyc",         "#D62728"),
        ("Georgia",        "georgia_ablation", "#9467BD"),
        ("USA",            "usa",         "#8C564B"),
    ]
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    for name, base, color in sources:
        try:
            pkg = _load_methods_multi_seed(base)
        except FileNotFoundError:
            continue
        arr = np.asarray(pkg["methods"]["Geom 50"])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.asarray(pkg["pq_diff"])
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.20, lw=0)
        ax.plot(x, mean, color=color, lw=2.0, marker="o", ms=5.5, label=name)
    ax.set_xlabel("pq Difference")
    ax.set_ylabel("Point Jaccard Distance")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Geom 50 across datasets")
    ax.legend(loc="upper right")
    out = OUT / "geom50_cross_cut_v9.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def render_nyc_with_inset() -> None:
    """NYC method-comparison curve with an inset map of NYC zip codes
    (target rectangle highlighted) and the legend pushed outside the plot.

    Layout:
        +-----------------------------------+   +---------+
        |  curves           [ NYC inset ]   |   |  legend |
        |                                   |   |         |
        +-----------------------------------+   +---------+
    """
    import geopandas as gpd
    import matplotlib.patches as mpatches
    from shapely.geometry import Polygon

    pp.apply_style_v9()
    # Bigger fonts so the figure reads at body-text size in Overleaf
    # (it gets scaled down to fit \textwidth).
    plt.rcParams.update({
        "axes.labelsize":  16,
        "axes.titlesize":  17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 14,
    })
    pkg = _load_methods_multi_seed("nyc")

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    pp.plot_jaccard_vs_pq(ax, pkg["methods"], pkg["pq_diff"],
                          show_band=True, band_alpha=0.25)
    ax.set_xlabel(r"$p - q$ difference")
    ax.set_ylabel("Point Jaccard distance")
    # Polished legend outside the plot on the right.
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    leg = ax.legend(handles, labels, loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    title="Method", title_fontsize=15,
                    fontsize=14, frameon=True, framealpha=0.95,
                    edgecolor="#888888", fancybox=True,
                    handlelength=2.4, handletextpad=0.7,
                    borderpad=0.8, labelspacing=0.8,
                    borderaxespad=0.6)
    leg.get_frame().set_linewidth(0.8)
    leg.get_title().set_fontweight("bold")

    # Inset map in the upper-right empty space of the plot, with lat/lon ticks.
    nyc_shp = DATA / "nyc" / "ZIP_CODE_040114.shp"
    nyc_gdf = gpd.read_file(nyc_shp).to_crs("EPSG:4326")
    target = Polygon([(-74, 40.6), (-74, 40.8), (-73.8, 40.8), (-73.8, 40.6)])

    inset = ax.inset_axes([0.62, 0.42, 0.36, 0.55])
    nyc_gdf.plot(ax=inset, color="white", edgecolor="#444444", linewidth=0.35,
                 rasterized=True)
    tx0, ty0, tx1, ty1 = target.bounds
    inset.add_patch(mpatches.Rectangle(
        (tx0, ty0), tx1 - tx0, ty1 - ty0,
        edgecolor="#C62828", facecolor=(0.78, 0.16, 0.16, 0.18),
        lw=1.4, ls=(0, (5, 2)),
    ))
    # Lat/lon ticks at sparse "round" coordinates so labels stay legible.
    inset.set_xticks([-74.2, -74.0, -73.8])
    inset.set_yticks([40.5, 40.7, 40.9])
    inset.tick_params(axis="both", which="major", labelsize=10,
                       length=2.5, width=0.5, pad=1.5)
    inset.set_facecolor("white")
    for sp in inset.spines.values():
        sp.set_edgecolor("#888888"); sp.set_linewidth(0.6)
    inset.set_aspect("equal")

    fig.subplots_adjust(left=0.06, right=0.84, top=0.97, bottom=0.12)
    out = OUT / "nyc_v9.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def render_size_sweep() -> None:
    """Fig 7: PJD vs target rectangle area % at fixed pq=0.4."""
    pp.apply_style_v9()
    pkg = _load_size_sweep_multi_seed("georgia_size_sweep")
    area_pct = np.array(pkg["area_pct"])
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    for method in pp.METHOD_ORDER:
        if method not in pkg["methods"]:
            continue
        style = pp.METHOD_STYLE[method]
        trials = np.array(pkg["methods"][method])  # (n_targets, n_trials)
        means = trials.mean(axis=1)
        stds  = trials.std(axis=1)
        ax.fill_between(area_pct, means - stds, means + stds,
                        color=style["color"], alpha=0.25, lw=0)
        ax.plot(area_pct, means, color=style["color"], ls=style["ls"],
                marker=style["marker"], ms=style["ms"], lw=style["lw"],
                label=method)
    ax.set_xlabel("Target Rectangle Percentage of Entire State")
    ax.set_ylabel("Point Jaccard Distance")
    ax.set_xlim(area_pct.min() * 0.9, area_pct.max() * 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right")
    out = OUT / "fig7_georgia_size_sweep_v9.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def render_fig10_2x2() -> None:
    """Fig 10: 2x2 grid — {uniform, weighted} sampling × {point, area} Jaccard."""
    pp.apply_style_v9()
    u = _load_georgia_ablation_multi_seed("georgia_ablation_uniform")
    w = _load_georgia_ablation_multi_seed("georgia_ablation_weighted")
    pq = np.asarray(u["pq_diff"])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    panels = [
        (axes[0, 0], u["point_jaccard"], "Uniform Point Sampling",  "Point Jaccard Distance"),
        (axes[0, 1], w["point_jaccard"], "Weighted Point Sampling", "Point Jaccard Distance"),
        (axes[1, 0], u["area_jaccard"],  None,                      "Area Jaccard Distance"),
        (axes[1, 1], w["area_jaccard"],  None,                      "Area Jaccard Distance"),
    ]
    for ax, data, title, ylabel in panels:
        pp.plot_jaccard_vs_pq(ax, data, pq, show_band=True, band_alpha=0.25,
                              legend_loc="upper right")
        if title:
            ax.set_title(title)
        ax.set_ylabel(ylabel if ax in (axes[0, 0], axes[1, 0]) else "")
        ax.set_xlabel("pq Difference" if ax in (axes[1, 0], axes[1, 1]) else "")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.08,
                        wspace=0.16, hspace=0.20)
    out = OUT / "fig10_georgia_ablation_2x2_v9.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    # All curve figures rendered in style v9 (paper-original look).
    render_methods_curve("nyc.pkl",         "nyc_v9.pdf",         title="New York City Zip Codes (20 Trials)")
    render_methods_curve("utah.pkl",        "utah_v9.pdf",        title="Utah Counties (20 Trials)")
    render_methods_curve("california.pkl",  "california_v9.pdf",  title="California Counties (20 Trials)")
    render_methods_curve("usa.pkl",         "usa_v9.pdf",         title="USA Counties (20 Trials)")
    render_methods_curve("arkansas_10.pkl", "arkansas_10_v9.pdf", title="Arkansas 10% Target Rectangle (20 Trials)")
    render_methods_curve("arkansas_30.pkl", "arkansas_30_v9.pdf", title="Arkansas 30% Target Rectangle (20 Trials)")
    # Signal-boost
    render_k_sweep()
    render_geom50_cross_cut()
    # Fig 7 + Fig 10 multi-panel
    render_size_sweep()
    render_fig10_2x2()
