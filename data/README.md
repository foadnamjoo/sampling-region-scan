# Data sources

This directory is empty by design — we do not redistribute shapefiles. Download them yourself from the original public sources below and place them where the scripts expect them.

## Shapefiles used in the paper

| Dataset | Source | Vintage / accession | License | Notes |
|---|---|---|---|---|
| **Arkansas counties** | US Census TIGER/Line — [Arkansas counties, 2020](https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip) | 2020 vintage, accessed 2026-05 | Public domain (US Census) | 75 counties; used in Figs 1, 2, 8, 9, 10 |
| **Utah counties** | US Census TIGER/Line — Utah counties (filter STATEFP=49 from national 2020 file above) | 2020 vintage, accessed 2026-05 | Public domain (US Census) | 29 counties; used in Fig 4 |
| **California counties** | [California DOJ / DFG cnty19_1](https://data.ca.gov/dataset/ca-geographic-boundaries) | 2019 boundaries, accessed 2026-05 | CC-BY 4.0 (CA Open Data) | 58 counties represented by 69 polygons (three coastal counties split by offshore islands); used in Figs 5, 13 |
| **Georgia counties** | US Census TIGER/Line — Georgia counties (filter STATEFP=13 from national 2020 file) | 2020 vintage, accessed 2026-05 | Public domain (US Census) | Used in Figs 7, 11 |
| **NYC zip codes** | [NYC OpenData — Modified Zip Code Tabulation Areas](https://data.cityofnewyork.us/Business/Modified-Zip-Code-Tabulation-Areas-MODZCTA-/pri4-ifjk) | Accessed 2026-05 | NYC OpenData terms (attribution) | 263 polygons / 248 unique ZIPs (ten ZIPs split by water); used in Fig 3 |
| **US counties (contiguous)** | US Census TIGER/Line — [`cb_2017_us_county_500k`](https://www2.census.gov/geo/tiger/GENZ2017/shp/cb_2017_us_county_500k.zip) | 2017 vintage, one row per county, accessed 2026-05 | Public domain (US Census) | **3,108** counties after the mainland bounds filter. This is the scan input for **Fig 6** and for the USA k-sweep curve in Fig 12. |
| **US counties × congressional district** | US Census TIGER/Line — [`cb_2018_us_county_within_cd116_500k`](https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_within_cd116_500k.zip) | 2018 vintage (county × 116th-Congress-district polygons), accessed 2026-05 | Public domain (US Census) | **3,711** rows after the mainland centroid filter — the same 3,108 counties split by congressional district. Used **only** by the runtime table (`src/experiments/run_runtime.py`). Do not use it for Fig 6. |
| **California Valley Fever cases** | [CHHS Open Data Portal — Infectious Diseases by Disease, County, Year, and Sex](https://data.chhs.ca.gov/dataset/03e61434-7db8-4a53-a3e2-1d4d36d6848d) | Snapshot 2026-06-05 (dataset is updated ~annually) | CC-BY 4.0 (CDPH) | Used in §6 (Fig 13). Download the CSV and save as `data/valley_fever/idb.csv`. |
| **CA county populations (SMR denominator)** | US Census ACS 5-year, county-level population 2014–2018 | Accessed 2026-06 | Public domain (US Census) | Denominator for standardized morbidity ratio in Fig 13. Provide as `data/valley_fever/population.csv` with columns `county, year, population`. |

## Expected layout

After downloading, place each shapefile under `data/` like so:

```
data/
  arkansas/      COUNTY_BOUNDARY.shp + .shx + .dbf + .prj
  utah/          geo_export_*.shp + ...
  california/    cnty19_1.shp + ...
  georgia/       ... .shp
  nyc/           ZIP_CODE_*.shp
  usa/           cb_2017_us_county_500k.shp                # Fig 6, USA k-sweep (3,108 counties)
                 cb_2018_us_county_within_cd116_500k.shp   # runtime table only (3,711 rows)
  valley_fever/  idb.csv                      # CHHS Coccidioidomycosis CSV
                 population.csv               # ACS county-year population
```

Each experiment script under `src/experiments/` documents the exact filename it expects. If you put files elsewhere, edit the `SHP` path constant at the top of each script.

## Coordinate reference system

All scripts internally re-project everything to **EPSG:4326** (lon/lat degrees). You can supply any input CRS — the scripts call `to_crs("EPSG:4326")` on load. The Valley Fever pipeline in `src/run_experiment_real.py` uses **EPSG:3310** (California Albers, metres) internally.

## Supplementary datasets (not used in the paper)

Two side-experiment folders may appear locally in `data/` if you're exploring the codebase:


These are **not referenced by any figure or table in the paper**. They are gitignored (see `.gitignore` at the repo root) and included in the codebase only as auxiliary sanity checks against classic published clusters. If you don't need them, ignore both; the paper's results are fully reproducible without them.


## Two USA partitions, on purpose

The paper uses two different partitions of the same 3,108 contiguous-US counties:

| input | rows after the mainland filter | used by |
|---|---|---|
| `cb_2017_us_county_500k` | **3,108** ordinary counties | Figure 6, USA k-sweep |
| `cb_2018_us_county_within_cd116_500k` | **3,711** county × congressional-district rows | runtime table only |

They are not interchangeable. `n = 3,711` in the runtime discussion and
`n = 3,108` in the Figure 6 discussion are both correct.

## Two NYC planted targets, also on purpose

| target | latitude | share of NYC area | produced |
|---|---|---|---|
| `NYC_TARGET_FIG3` | 40.65 – 40.8 | 32.8% | Figure 3 |
| `NYC_TARGET_KSWEEP` | 40.60 – 40.8 | 40.5% | the NYC curve in the Figure 12 k-sweep |

Both are defined in `src/run_experiment.py`. Keep them distinct so each
published figure stays reproducible.

## Georgia ablation weighting

The weighted arm shown in Figure 11 uses **real county population**, joined onto
the Georgia geometry by `src/experiments/run_georgia_ablation_population.py`
(`weight_col="population"`). The `aland10` land-area default in
`run_experiment.py` produced an earlier, superseded version.

The allocation caps every county at `k_max` and never redistributes the removed
surplus, so the weighted arm uses **36–52% fewer points** than the uniform arm
(Geom 5: 380 vs 795; Geom 10: 1,014 vs 1,590; Geom 50: 4,577 vs 7,950). It is a
sensitivity test, not an equal-budget allocation comparison.
