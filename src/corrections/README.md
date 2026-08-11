# Point Jaccard evaluation-set correction (fixed reference set A)

## What was wrong

The paper defines Point Jaccard distance on a **fixed reference set A**: 500 points
sampled uniformly inside every input region, built once and reused. Several
synthetic-experiment pipelines instead evaluated the overlap on **that trial's
Bernoulli-selected measured points**, whose size and location depend on the
method, on `p`, and on the trial. So each method was scored on a different set.

Affected figures (all confirmed by tracing PDF → renderer → pickle → producing script):

| figure | dataset | producing script | scored on |
|---|---|---|---|
| 3 | NYC | `buchin_attempt/nyc_grid_resolution_check.py` | measured points |
| 4 | Utah | `buchin_attempt/utah_cal_grid_resolution_check.py` | measured points |
| 5 | California | `buchin_attempt/utah_cal_grid_resolution_check.py` | measured points |
| 6 | continental USA | `SIGSPATIAL_2026_figures/scripts/run_experiment.py` | measured points |
| 7 | Georgia size sweep | `buchin_attempt/rerun_georgia_size_sweep_more_trials.py` | measured points |
| 11 | Georgia ablation (Point panels) | `run_experiment.py` ablation entry point | measured points |
| 12 | k-sweep, 6 datasets × 3 seeds | `run_experiment.py` k-sweep entry point | measured points |

**Not affected** — these already used a fixed reference pool and were left alone:
Figures 8–10 (Arkansas, via `postprocess_rerun.py` and the 500-pts/region
`arkansas_point_dict.pkl`), Figure 13/14 (Valley Fever, via
`run_experiment_real.py` and `reference_set(..., seed=42)`), Figure 15 (disk
appendix, via `arkansas_disk_stress.py`).

## What was done

Every affected experiment was **re-run with the data generation completely
unchanged** — same shapefiles, same targets, same Bernoulli inclusion, same
methods, same trial counts, same experiment seeds, same grid resolution, same
pyScan calls, therefore the same discovered rectangles. The **only** change is
where the Jaccard is evaluated.

The evaluation set A:

* exactly **500 uniform points per polygon**
* deterministic **evaluation seed 42**, via `shape_floor.reference_set`
* built **once per dataset** and reused across every method, k, p−q value,
  trial and experiment seed
* uses its **own RNG**, so constructing it changes no scan location and no
  Bernoulli draw
* is **evaluation-only** — distinct from the 500-point *sampling reservoir*
  that some scripts draw the k scan points from

## Fidelity gates

Each group's reproduced **old** Jaccard values were compared against the exact
published pickle before any fixed-A result was accepted. The gate checks nested
shape first, then `max|diff| < 1e-9`, and **raises** on failure so a failed
reproduction cannot be silently used.

**All 25 gates passed at exactly 0.0e+00**, including both Georgia Area Jaccard
gates, which proves Area Jaccard is unchanged by this correction.

The k-sweep pre-flights all 18 expected pickles (6 datasets × seeds 7/31/67) and
refuses to run if any is missing; all 18 were present.

## Contents

```
results/   fixed-A outputs, one pickle per group, plus report_fixedA.json
logs/      full run logs including every gate line
scripts/   the rerun implementation and the report builder
figures/   regenerated figures (suffixed _fixedA)
```

Every record carries: group, method, trial, k, p, q, target index, target
bounds, weighted flag, grid resolution, discovered rectangle bounds, pyScan
`fValue`, old Jaccard, fixed-A Jaccard, the RNG state before
`point_set_for_method`, the RNG state before the Bernoulli draw, and a SHA-1
checksum of the generated point coordinates.

## Reproducing

```bash
B=/Users/foadnamjoo/PROJECT/PYSCAN/pyscan/build
export DYLD_FALLBACK_LIBRARY_PATH="$B/thirdparty/discrepancy:$B/thirdparty/kernel/coreset:$B/thirdparty/kernel/ANN"
python scripts/fixedA_v2.py usa gasize gaablation ksweep
python scripts/report_fixedA.py
```

The `DYLD_FALLBACK_LIBRARY_PATH` export is required because the compiled
`libpyscan` carries an rpath pointing at an older project location
(`~/Desktop/PYSCAN`). Add `--smoke` to run one record per group and print the
provenance fields without touching the gates.
