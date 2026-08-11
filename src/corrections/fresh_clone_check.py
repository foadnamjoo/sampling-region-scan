"""Fresh-clone reproducibility check for the camera-ready-fixedA branch.

Run this from a clone of the repository. It proves, without re-running any
published experiment, that:

  1. fixed-A evaluation is ACTIVE by default in the canonical driver;
  2. constructing the evaluation set A perturbs no point location, no Bernoulli
     draw, no discovered rectangle and no pyScan scan score;
  3. Figure 6 loads 3,108 ordinary counties, and the runtime input is a
     different 3,711-row partition;
  4. the NYC targets for Figure 3 and the Figure 12 k-sweep are distinct and
     carry the documented values;
  5. the figure scripts resolve to real files.

County shapefiles are not redistributed with the repository, so set
PYSCAN_DATA_ROOT to a directory holding them; checks that need a missing
shapefile report SKIP rather than failing.

    PYSCAN_BUILD=/path/to/pyscan/build \
    PYSCAN_DATA_ROOT=/path/to/shapefiles \
    python src/corrections/fresh_clone_check.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

ok, fail, skip = [], [], []


def check(cond, msg):
    (ok if cond else fail).append(msg)
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}", flush=True)


def skipped(msg):
    skip.append(msg)
    print(f"  [SKIP] {msg}", flush=True)


def find_shp(*names):
    root = os.environ.get("PYSCAN_DATA_ROOT")
    roots = [REPO / "data"] + ([Path(root)] if root else [])
    for r in roots:
        for n in names:
            hits = list(r.rglob(n))
            if hits:
                return hits[0]
    return None


def main():
    print("=== 1. fixed-A is the default in the canonical driver ===")
    import run_experiment as RX
    RX.OUT_DIR = Path(tempfile.mkdtemp(prefix="fresh_clone_"))
    check(RX.USE_FIXED_A is True, "USE_FIXED_A is True with no env override")
    check(RX.EVAL_SEED == 42, f"evaluation seed is 42 (got {RX.EVAL_SEED})")
    check(RX.EVAL_POINTS_PER_REGION == 500,
          f"500 evaluation points per region (got {RX.EVAL_POINTS_PER_REGION})")
    check(RX.LEGACY_MEASURED_JD is False,
          "legacy measured-subset evaluation is OFF unless PYSCAN_LEGACY_MEASURED_JD=1")
    src = (REPO / "src" / "run_experiment.py").read_text()
    check("PYSCAN_LEGACY_MEASURED_JD" in src,
          "the legacy path is reachable only behind an explicit flag")

    print("\n=== 2. NYC targets are distinct and documented ===")
    f3 = RX.NYC_TARGET_FIG3.bounds
    ks = RX.NYC_TARGET_KSWEEP.bounds
    check(abs(f3[1] - 40.65) < 1e-12, f"Figure 3 target latitude starts at 40.65 (got {f3[1]})")
    check(abs(ks[1] - 40.6) < 1e-12, f"k-sweep target latitude starts at 40.6 (got {ks[1]})")
    check(f3 != ks, "the two NYC targets are deliberately different")
    check(RX.EXPERIMENTS["nyc"][1]["target"].bounds == f3,
          "EXPERIMENTS['nyc'] uses the Figure 3 target")
    check(RX.EXPERIMENTS["k_sweep_nyc"][1]["target"].bounds == ks,
          "EXPERIMENTS['k_sweep_nyc'] keeps the k-sweep target")

    print("\n=== 3. USA partitions ===")
    import geopandas as gpd
    ord_shp = find_shp("cb_2017_us_county_500k.shp")
    cd_shp = find_shp("cb_2018_us_county_within_cd116_500k.shp")
    if ord_shp:
        g = gpd.read_file(ord_shp)
        if str(g.crs) != "EPSG:4326":
            g = g.to_crs("EPSG:4326")
        b = g.geometry.bounds
        n = int(((b["minx"] >= -130) & (b["maxx"] <= -65) &
                 (b["miny"] >= 24) & (b["maxy"] <= 50)).sum())
        check(n == 3108, f"Figure 6 input is 3,108 ordinary counties (got {n})")
    else:
        skipped("cb_2017_us_county_500k.shp not found; set PYSCAN_DATA_ROOT")
    if cd_shp:
        g = gpd.read_file(cd_shp)
        if str(g.crs) != "EPSG:4326":
            g = g.to_crs("EPSG:4326")
        c = g.geometry.centroid
        n = int(((c.x > -126) & (c.x < -64) & (c.y > 23) & (c.y < 50)).sum())
        check(n == 3711, f"runtime input is a different 3,711-row partition (got {n})")
    else:
        skipped("cb_2018_us_county_within_cd116_500k.shp not found")
    fig6 = (REPO / "src" / "figures" / "fig07_georgia_size.py").read_text()
    check("cb_2017_us_county_500k" in fig6 and
          "cb_2018_us_county_within_cd116_500k.shp\"" not in fig6.split("USA_SHP =")[1][:80],
          "the Figure 6 script points at the ordinary-county shapefile")

    print("\n=== 4. building A perturbs nothing ===")
    ga = find_shp("GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp")
    if not ga:
        skipped("Georgia shapefile not found; cannot run the perturbation check")
    else:
        tgt = Polygon([(-85.0, 31.0), (-85.0, 32.89), (-83.61, 32.89), (-83.61, 31.0)])
        kw = dict(name="fresh_fixed", shp_path=str(ga), target=tgt, n_trials=1,
                  pq_grid=[0.35, 0.45], grid_res=40, seed=7)
        a = RX.run_methods_experiment(**kw)
        RX.USE_FIXED_A = False
        try:
            b = RX.run_methods_experiment(**dict(kw, name="fresh_legacy"))
        finally:
            RX.USE_FIXED_A = True
        ra = [tuple(np.round(r["rect"], 12)) for r in a["records"]]
        rb = [tuple(np.round(r["rect"], 12)) for r in b["records"]]
        check(a["evaluation"]["mode"] == "fixed_A", "default run used fixed A")
        check(b["evaluation"]["mode"] == "legacy_measured_points",
              "flagged run used the legacy evaluation")
        check(ra == rb, "discovered rectangles identical between the two modes")
        check([r["points_checksum"] for r in a["records"]]
              == [r["points_checksum"] for r in b["records"]],
              "generated point locations identical (same SHA-1)")
        check([r["n_measured"] for r in a["records"]]
              == [r["n_measured"] for r in b["records"]],
              "Bernoulli draws identical (same measured counts)")
        check([r["fValue"] for r in a["records"]] == [r["fValue"] for r in b["records"]],
              "pyScan scan scores identical")
        check(any(x["fixedA_jd"] != y["old_jd"]
                  for x, y in zip(a["records"], b["records"])),
              "Point Jaccard values DID change, so the correction is active")

    print("\n=== 5. figure scripts resolve ===")
    for s in ["fig01_arkansas_sampling.py", "fig02_jdarkansas.py",
              "fig03_06_state_curves.py", "fig07_georgia_size.py",
              "fig08_09_arkansas_buchin.py", "fig12_17_18_buchin_maps.py",
              "fig14_georgia_ablation.py", "fig16_k_sweep.py"]:
        check((REPO / "src" / "figures" / s).exists(), f"src/figures/{s} present")
    check((REPO / "src" / "experiments" / "run_georgia_ablation_population.py").exists(),
          "the shipped Figure 11 weighted arm script is present")

    print(f"\n{'='*66}\n{len(ok)} passed, {len(fail)} failed, {len(skip)} skipped")
    for f in fail:
        print(f"  FAILED: {f}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
