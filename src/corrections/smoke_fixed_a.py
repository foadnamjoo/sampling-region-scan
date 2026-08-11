"""Smoke tests for the fixed-A wiring in run_experiment.py.

Tiny runs only -- 1-2 trials, 2 rate contrasts, few targets. This never touches
the published experiment suite.

Checks:
  1. fixed-A is the DEFAULT for all four experiment kinds
  2. every result pickle carries evaluation provenance and per-scan records
  3. every required provenance field is populated on every record
  4. PYSCAN_LEGACY_MEASURED_JD=1 restores the historical evaluation
  5. building A perturbs NO scan location, NO Bernoulli draw and NO discovered
     rectangle -- proved by comparing discovered rectangles between the two modes
  6. Area Jaccard is identical between the two modes
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_experiment as RX  # noqa: E402

# Smoke pickles go to a throwaway directory, never to outputs/cached_data/.
RX.OUT_DIR = Path(tempfile.mkdtemp(prefix="fixedA_smoke_"))
print(f"smoke outputs -> {RX.OUT_DIR}")

def _georgia_shapefile() -> str:
    """The repo ships no county shapefiles, so allow a local data root.

    Order: repo data/ -> $PYSCAN_DATA_ROOT -> give up with a clear message.
    """
    import os
    name = "GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp"
    cands = [RX.DATA / "georgia" / name]
    root = os.environ.get("PYSCAN_DATA_ROOT")
    if root:
        cands += [Path(root) / "GeorgiaCounties" / name, Path(root) / "georgia" / name]
    for c in cands:
        if c.exists():
            return str(c)
    raise SystemExit(
        "Georgia shapefile not found. Set PYSCAN_DATA_ROOT to a directory "
        f"containing GeorgiaCounties/{name}. Tried:\n  "
        + "\n  ".join(str(c) for c in cands))


GA = _georgia_shapefile()
TARGET = Polygon([(-85.0, 31.0), (-85.0, 32.89), (-83.61, 32.89), (-83.61, 31.0)])
PQ = [0.35, 0.45]           # two rate contrasts
REQUIRED = ["group", "dataset", "method", "trial", "k", "p_prob", "q",
            "experiment_seed", "target_bounds", "rect", "fValue", "fixedA_jd",
            "eval_seed", "points_per_region", "grid_res",
            "rng_state_before_points", "rng_state_before_bernoulli",
            "points_checksum", "n_points", "n_measured"]

ok, fail = [], []


def check(cond, msg):
    (ok if cond else fail).append(msg)
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}", flush=True)


def rects(pkg):
    return [tuple(np.round(r["rect"], 12)) for r in pkg["records"]]


def main():
    print("\n=== 1. methods experiment (fixed A is the default) ===", flush=True)
    m = RX.run_methods_experiment("smoke_methods", GA, TARGET, n_trials=1,
                                  pq_grid=PQ, grid_res=40, seed=7)
    ev = m["evaluation"]
    check(ev["mode"] == "fixed_A", f"default mode is fixed_A (got {ev['mode']})")
    check(ev["eval_seed"] == 42, f"eval seed 42 (got {ev['eval_seed']})")
    check(ev["points_per_region"] == 500,
          f"500 points/region (got {ev['points_per_region']})")
    check(ev["n_eval_points"] == 500 * ev["n_regions"],
          f"|A| = 500n = {ev['n_eval_points']} over {ev['n_regions']} regions")
    check(len(m["records"]) == 5 * len(PQ),
          f"one record per (method, contrast): {len(m['records'])}")
    missing = {f for r in m["records"] for f in REQUIRED if r.get(f) is None}
    check(not missing, f"all provenance fields populated (missing: {sorted(missing)})")
    check(all(r["fValue"] is not None for r in m["records"]), "pyScan fValue stored")
    check(all(0.0 <= r["fixedA_jd"] <= 1.0 for r in m["records"]),
          "every fixed-A Point Jaccard in [0,1]")

    print("\n=== 2. k-sweep ===", flush=True)
    k = RX.run_k_sweep("smoke_ksweep", GA, TARGET, k_values=[2, 5],
                       p_prob=0.35, n_trials=1, grid_res=40, seed=7)
    check(k["evaluation"]["mode"] == "fixed_A", "k-sweep uses fixed A by default")
    check({r["k"] for r in k["records"]} == {2, 5}, "k recorded per scan")

    print("\n=== 3. size sweep (target VARIES) ===", flush=True)
    s = RX.run_size_sweep("smoke_size", GA, x_base=-85.0, y_base=31.0,
                          x_array=[-84.5, -83.0], y_array=[31.5, 33.0],
                          p_prob=0.6, n_trials=1, grid_res=40, seed=7)
    check(s["evaluation"]["mode"] == "fixed_A", "size sweep uses fixed A by default")
    tb = {tuple(r["target_bounds"]) for r in s["records"]}
    check(len(tb) == 2, f"two distinct targets scored on the SAME A ({len(tb)})")
    check({r["target_index"] for r in s["records"]} == {0, 1}, "target index recorded")

    print("\n=== 4. Georgia ablation (Point + Area) ===", flush=True)
    a = RX.run_georgia_ablation_full("smoke_ablation", GA, TARGET, n_trials=1,
                                     pq_grid=PQ, grid_res=40, seed=7,
                                     weighted=False)
    check(a["evaluation"]["mode"] == "fixed_A", "ablation uses fixed A by default")
    check(all("area_jd" in r for r in a["records"]), "area JD stored per record")

    print("\n=== 5. legacy compatibility flag ===", flush=True)
    RX.USE_FIXED_A = False                       # what PYSCAN_LEGACY_MEASURED_JD=1 does
    try:
        m_leg = RX.run_methods_experiment("smoke_methods_legacy", GA, TARGET,
                                          n_trials=1, pq_grid=PQ, grid_res=40, seed=7)
        a_leg = RX.run_georgia_ablation_full("smoke_ablation_legacy", GA, TARGET,
                                             n_trials=1, pq_grid=PQ, grid_res=40,
                                             seed=7, weighted=False)
    finally:
        RX.USE_FIXED_A = True
    check(m_leg["evaluation"]["mode"] == "legacy_measured_points",
          "flag restores the historical evaluation")
    check(all(r["old_jd"] is not None and r["fixedA_jd"] is None
              for r in m_leg["records"]), "legacy records store old_jd, not fixedA_jd")

    print("\n=== 6. building A perturbs NOTHING ===", flush=True)
    check(rects(m) == rects(m_leg),
          "discovered rectangles byte-identical between fixed-A and legacy runs")
    check([r["points_checksum"] for r in m["records"]]
          == [r["points_checksum"] for r in m_leg["records"]],
          "generated point sets identical (same SHA-1 per scan)")
    check([r["n_measured"] for r in m["records"]]
          == [r["n_measured"] for r in m_leg["records"]],
          "Bernoulli draws identical (same measured count per scan)")
    check([r["fValue"] for r in m["records"]] == [r["fValue"] for r in m_leg["records"]],
          "pyScan scan statistic identical")

    print("\n=== 7. Area Jaccard unaffected by the evaluation change ===", flush=True)
    same = all(np.array_equal(np.asarray(a["area_jaccard"][mn]),
                              np.asarray(a_leg["area_jaccard"][mn]))
               for mn in RX.METHOD_NAMES)
    check(same, "Area Jaccard arrays exactly equal in both modes")
    changed = any(not np.array_equal(np.asarray(a["point_jaccard"][mn]),
                                     np.asarray(a_leg["point_jaccard"][mn]))
                  for mn in RX.METHOD_NAMES)
    check(changed, "Point Jaccard DID change (the correction is actually active)")

    print(f"\n{'='*64}\n{len(ok)} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
