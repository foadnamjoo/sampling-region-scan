
# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------
"""Georgia ablation — Grid(40) vs Grid(100) paired check.

Mirrors run_experiment.py's `georgia_ablation_uniform` + `georgia_ablation_weighted`
configs EXACTLY (same shapefile, target, n_trials=20, pq_grid, seed=7,
weight_col='aland10' for the weighted variant) — the ONLY change is
grid_res=40 -> grid_res=100.

Writes to NEW files in buchin_attempt/:
  georgia_ablation_uniform_grid100.pkl
  georgia_ablation_weighted_grid100.pkl

The existing Grid(40) pkls in cached_data/ are left untouched and used as
the baseline for the comparison report at the end.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

# Importing run_experiment runs its top-level chdir to pyscan/build and
# imports pyscan — exactly what the pipeline does. We then reuse its
# run_georgia_ablation_full() to guarantee identical semantics.
import run_experiment as RE  # noqa: E402

import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

OUT = OUTPUTS
CACHED = OUTPUTS / "cached_data"

GA_SHP = str(RE.DATA / "georgia/"
             "GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp")
GA_TARGET = RE.Polygon([(-85.0, 31.0), (-85.0, 32.89),
                        (-83.61, 32.89), (-83.61, 31.0)])


def run_grid100(weighted: bool) -> dict:
    """Re-run with grid_res=100, saving to buchin_attempt/ (NOT cached_data)."""
    name = ("georgia_ablation_weighted_grid100" if weighted
            else "georgia_ablation_uniform_grid100")
    # Temporarily redirect OUT_DIR inside run_experiment so its save() writes
    # into buchin_attempt/ instead of cached_data/. Restore after.
    old_out = RE.OUT_DIR
    RE.OUT_DIR = OUT
    try:
        pkg = RE.run_georgia_ablation_full(
            name=name,
            shp_path=GA_SHP,
            target=GA_TARGET,
            n_trials=RE.DEFAULT_TRIALS,
            pq_grid=RE.DEFAULT_PQ,
            grid_res=100,
            seed=RE.DEFAULT_SEED,
            weighted=weighted,
            weight_col="aland10" if weighted else "aland10",
        )
    finally:
        RE.OUT_DIR = old_out
    return pkg


def load_grid40(name: str) -> dict:
    with open(CACHED / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def report(pkg40: dict, pkg100: dict, sampling: str):
    """Compare Grid(40) vs Grid(100) on point_jaccard for `sampling`."""
    pq = np.asarray(pkg40["pq_diff"])
    assert np.allclose(pq, np.asarray(pkg100["pq_diff"])), \
        "pq grids must match"

    methods = ["Centroid", "Geom 5", "Geom 10", "Geom 50"]
    # Locate mid-range columns (pq ~= 0.15, 0.20, 0.25) and plateau mask.
    def col(target):
        return int(np.argmin(np.abs(pq - target)))
    mid_cols = {0.15: col(0.15), 0.20: col(0.20), 0.25: col(0.25)}
    plateau_mask = pq >= 0.5

    def arr(pkg, method):
        return np.asarray(pkg["point_jaccard"][method])

    print(f"\n{'='*78}")
    print(f"  GEORGIA ABLATION — {sampling.upper()} sampling — Point Jaccard")
    print(f"  Grid(40) baseline  vs  Grid(100)  (paired, seed=7, 20 trials)")
    print(f"{'='*78}")
    header = (f"  {'method':<10}  "
              + "  ".join(f"{'pq='+f'{t:.2f}':>10}" for t in mid_cols)
              + f"  {'plateau (pq>=0.5)':>20}")
    print(header)
    for grid_label, pkg in (("Grid(40)", pkg40), ("Grid(100)", pkg100)):
        print(f"  --- {grid_label} ---")
        for m in methods:
            a = arr(pkg, m)
            row = f"  {m:<10}  "
            for t, c in mid_cols.items():
                row += f"{a[:, c].mean():>10.4f}  "
            row += f"{a[:, plateau_mask].mean():>20.4f}"
            print(row)

    print(f"\n  --- Geom-vs-Centroid gap (Centroid mean − Geom mean) ---")
    print(f"  {'method':<10}  "
          + "  ".join(f"{'pq='+f'{t:.2f}':>10}" for t in mid_cols)
          + f"  {'plateau':>10}")
    for m in ("Geom 5", "Geom 10", "Geom 50"):
        print(f"  --- {m} ---")
        for grid_label, pkg in (("Grid(40)", pkg40), ("Grid(100)", pkg100)):
            c_arr = arr(pkg, "Centroid")
            m_arr = arr(pkg, m)
            row = f"  {grid_label:<10}  "
            for t, c in mid_cols.items():
                gap = c_arr[:, c].mean() - m_arr[:, c].mean()
                row += f"{gap:>+10.4f}  "
            gap_pl = c_arr[:, plateau_mask].mean() - m_arr[:, plateau_mask].mean()
            row += f"{gap_pl:>+10.4f}"
            print(row)


def plot_overlay(pkg40: dict, pkg100: dict, out_png: Path):
    """Overlay Grid(40) dashed vs Grid(100) solid, Point Jaccard, uniform."""
    pq = np.asarray(pkg40["pq_diff"])
    methods = ["Centroid", "Geom 5", "Geom 10", "Geom 50"]
    colors = {"Centroid": "#7F7F7F",
              "Geom 5":   "#1F77B4",
              "Geom 10":  "#2CA02C",
              "Geom 50":  "#D62728"}
    markers = {"Centroid": "o", "Geom 5": "s",
               "Geom 10": "^", "Geom 50": "D"}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for m in methods:
        a40  = np.asarray(pkg40["point_jaccard"][m])
        a100 = np.asarray(pkg100["point_jaccard"][m])
        mean40, std40 = a40.mean(0),  a40.std(0)
        mean100, std100 = a100.mean(0), a100.std(0)
        c = colors[m]; mk = markers[m]
        ax.fill_between(pq, mean100 - std100, mean100 + std100,
                        color=c, alpha=0.10, lw=0)
        ax.plot(pq, mean40,  color=c, ls="--", marker=mk, markersize=4,
                lw=1.4, label=f"{m} — Grid(40)")
        ax.plot(pq, mean100, color=c, ls="-",  marker=mk, markersize=5,
                lw=2.0, label=f"{m} — Grid(100)")
    ax.set_xlabel(r"$p - q$ difference")
    ax.set_ylabel("Point Jaccard distance")
    ax.set_title("Georgia ablation (UNIFORM sampling) — "
                 "Grid(40) dashed vs Grid(100) solid")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [plot] {out_png}")


def main():
    t0 = time.time()
    print(f"\n[grid100] starting Georgia ablation Grid(100) re-runs", flush=True)
    pkg_uni_100  = run_grid100(weighted=False)
    pkg_wgt_100  = run_grid100(weighted=True)
    print(f"\n[grid100] both Grid(100) runs done in {time.time()-t0:.0f}s")

    pkg_uni_40 = load_grid40("georgia_ablation_uniform")
    pkg_wgt_40 = load_grid40("georgia_ablation_weighted")

    report(pkg_uni_40, pkg_uni_100, sampling="uniform")
    report(pkg_wgt_40, pkg_wgt_100, sampling="weighted")

    plot_overlay(pkg_uni_40, pkg_uni_100,
                 OUT / "georgia_ablation_uniform_grid40_vs_grid100.png")


if __name__ == "__main__":
    main()
