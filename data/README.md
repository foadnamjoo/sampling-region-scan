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
| **US counties (continental)** | US Census TIGER/Line — [`cb_2018_us_county_within_cd116_500k`](https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_within_cd116_500k.zip) | 2018 vintage (county × 116th-Congress-district polygons), accessed 2026-05 | Public domain (US Census) | 3,711 polygons spanning ~3,108 continental counties after longitude/latitude restriction; used in Fig 6 |
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
  usa/           cb_2018_us_county_within_cd116_500k.shp
  valley_fever/  idb.csv                      # CHHS Coccidioidomycosis CSV
                 population.csv               # ACS county-year population
```

Each experiment script under `src/experiments/` documents the exact filename it expects. If you put files elsewhere, edit the `SHP` path constant at the top of each script.

## Coordinate reference system

All scripts internally re-project everything to **EPSG:4326** (lon/lat degrees). You can supply any input CRS — the scripts call `to_crs("EPSG:4326")` on load. The Valley Fever pipeline in `src/run_experiment_real.py` uses **EPSG:3310** (California Albers, metres) internally.

## Supplementary datasets (not used in the paper)

Two side-experiment folders may appear locally in `data/` if you're exploring the codebase:

- `data/cholera_snow/` — John Snow's 1854 Broad Street cholera outbreak (deaths + pump locations). Used by `src/run_experiment_real.py --cholera` for a historical validation demo. Source: [Robin Wilson's georeferenced Snow shapefiles](http://blog.rtwilson.com/john-snows-cholera-data-in-more-formats/).
- `data/scotland_lip/` — 1975–1980 Scottish lip-cancer incidence per health district. Used by `src/run_experiment_real.py --scotland`. Source: [SpatialEpi R package](https://cran.r-project.org/web/packages/SpatialEpi/index.html) or the original Clayton & Kaldor (1987) dataset.

These are **not referenced by any figure or table in the paper**. They are gitignored (see `.gitignore` at the repo root) and included in the codebase only as auxiliary sanity checks against classic published clusters. If you don't need them, ignore both; the paper's results are fully reproducible without them.
