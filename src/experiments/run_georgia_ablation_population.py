"""Georgia ablation, weighted arm re-run with REAL POPULATION at Grid(100).

Background
----------
The paper describes Weighted Sampling as allocating points proportionally to
each county's baseline POPULATION, and explains the result via Atlanta's
population concentration.  An earlier version was produced with `weight_col="aland10"` (county land
area), used as a proxy because the bundled Georgia shapefile carries no
population column. THIS script produced the arm shown in the paper.

This script restores the intended experiment: it joins a real population column
onto the same Georgia geometry and re-runs ONLY the weighted arm, holding every
other setting identical to georgia_ablation_grid100_check.py (same shapefile,
same target rectangle, n_trials=20, same pq grid, seed=7, grid_res=100).

The uniform arm is unaffected by the weighting choice, so the existing
georgia_ablation_uniform_grid100.pkl is reused as the comparison baseline.

Writes:
  buchin_attempt/GeorgiaCounties_pop/ga_counties_pop.shp   (joined geometry)
  buchin_attempt/georgia_ablation_weighted_pop_grid100.pkl (results)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "src"
sys.path.insert(0, str(PIPELINE))

import run_experiment as RE  # noqa: E402  (chdirs to pyscan/build, imports pyscan)

import geopandas as gpd  # noqa: E402
import numpy as np       # noqa: E402

OUT = ROOT / "outputs" / "cached_data"

GA_SHP = str(RE.DATA / "georgia" / "GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp")
POP_SHP = str(RE.DATA / "georgia_population" / "GU_CountyOrEquivalent.shp")
GA_TARGET = RE.Polygon([(-85.0, 31.0), (-85.0, 32.89),
                        (-83.61, 32.89), (-83.61, 31.0)])

JOINED_DIR = OUT / "GeorgiaCounties_pop"
JOINED_SHP = JOINED_DIR / "ga_counties_pop.shp"


def build_joined_shapefile() -> str:
    """Attach a real `population` column to the paper's Georgia geometry."""
    ga = gpd.read_file(GA_SHP)
    pop = gpd.read_file(POP_SHP)

    key_ga = ga["name10"].str.strip().str.upper()
    key_pop = pop["county_nam"].str.strip().str.upper()

    merged = ga.merge(pop[["county_nam", "population"]].assign(_k=key_pop),
                      left_on=key_ga, right_on="_k", how="left")
    assert len(merged) == len(ga), f"join changed row count: {len(ga)} -> {len(merged)}"
    assert merged["population"].isna().sum() == 0, "unmatched counties in population join"

    merged = merged.drop(columns=["_k", "county_nam"], errors="ignore")
    merged["population"] = merged["population"].astype(float)

    JOINED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_file(JOINED_SHP)

    print(f"[join] {len(merged)} counties, population "
          f"total={merged['population'].sum():,.0f} "
          f"min={merged['population'].min():,.0f} "
          f"max={merged['population'].max():,.0f}", flush=True)
    print(f"[join] concentration: population spread "
          f"{merged['population'].max()/merged['population'].min():.0f}x  vs  "
          f"area spread {merged['aland10'].max()/merged['aland10'].min():.0f}x", flush=True)
    return str(JOINED_SHP)


def main() -> None:
    shp = build_joined_shapefile()

    old_out = RE.OUT_DIR
    RE.OUT_DIR = OUT
    try:
        pkg = RE.run_georgia_ablation_full(
            name="georgia_ablation_weighted_pop_grid100",
            shp_path=shp,
            target=GA_TARGET,
            n_trials=RE.DEFAULT_TRIALS,
            pq_grid=RE.DEFAULT_PQ,
            grid_res=100,
            seed=RE.DEFAULT_SEED,
            weighted=True,
            weight_col="population",
        )
    finally:
        RE.OUT_DIR = old_out

    # ---- comparison against the published area-weighted arm -----------------
    def load(name):
        with open(OUT / f"{name}.pkl", "rb") as f:
            return pickle.load(f)

    uni = load("georgia_ablation_uniform_grid100")
    area = load("georgia_ablation_weighted_grid100")

    pq = np.array(pkg["pq_diff"])
    print("\n" + "=" * 78)
    print("Point Jaccard, mean over 20 trials.  UNI = uniform, AREA = area-weighted,")
    print("POP = population-weighted (this run).  Lower is better.")
    print("=" * 78)
    for method in ("Geom 50", "Geom 10", "Geom 5", "Centroid"):
        print(f"\n{method}")
        print(f"  {'pq':>6} {'UNI':>8} {'AREA':>8} {'POP':>8}   {'POP-UNI':>8}")
        for src, lab in ((uni, "u"), (area, "a"), (pkg, "p")):
            src["_m"] = None
        mu = np.array(uni["point_jaccard"][method]).mean(axis=0)
        ma = np.array(area["point_jaccard"][method]).mean(axis=0)
        mp = np.array(pkg["point_jaccard"][method]).mean(axis=0)
        for i, x in enumerate(pq):
            if round(float(x), 2) in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
                print(f"  {x:6.2f} {mu[i]:8.3f} {ma[i]:8.3f} {mp[i]:8.3f}   {mp[i]-mu[i]:+8.3f}")
    print("\nIf POP > UNI, weighted sampling is WORSE than uniform "
          "-- which is what the paper claims.")


if __name__ == "__main__":
    main()
