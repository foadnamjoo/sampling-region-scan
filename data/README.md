# Data sources

This directory is empty by design — we do not redistribute shapefiles. Download them yourself from the original public sources below and place them where the scripts expect them.

## Shapefiles used in the paper

| Dataset | Source | Notes |
|---|---|---|
| **Arkansas counties** | US Census TIGER/Line — [Arkansas counties](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | 75 counties; used in Figs 1, 2, 8, 9, 12, 17, 18 |
| **Utah counties** | US Census TIGER/Line — Utah counties | 29 counties; used in Fig 4 |
| **California counties** | US Census TIGER/Line — California counties OR [California Open Data Portal](https://data.ca.gov/) | 69 counties; used in Fig 5 |
| **Georgia counties** | US Census TIGER/Line — Georgia counties | Used in Figs 7, 14 |
| **NYC zip codes** | [NYC OpenData — Modified Zip Code Tabulation Areas](https://data.cityofnewyork.us/) | 263 zip codes; used in Fig 3 |
| **US counties (continental)** | US Census TIGER/Line — counties (national) | 3,711 counties (continental); used in Fig 6 |
| **California Valley Fever cases / population** | [CHHS Open Data Portal — Infectious Diseases by Disease, County, Year, and Sex](https://data.chhs.ca.gov/dataset/03e61434-7db8-4a53-a3e2-1d4d36d6848d) | Used in Appendix C (Figs 15, 16). Download the CSV and save as `data/valley_fever/idb.csv`. |

## Expected layout

After downloading, place each shapefile under `data/` like so:

```
data/
  arkansas/      COUNTY_BOUNDARY.shp + .shx + .dbf + .prj
  utah/          geo_export_*.shp + ...
  california/    cnty19_1.shp + ...
  georgia/       ... .shp
  nyc/           ZIP_CODE_*.shp
  usa/           cb_*_us_county_*.shp
  valley_fever/  idb.csv                      # CHHS Coccidioidomycosis CSV
```

Each experiment script under `src/experiments/` documents the exact filename it expects. If you put files elsewhere, edit the `SHP` path constant at the top of each script.

## Coordinate reference system

All scripts internally re-project everything to **EPSG:4326** (lon/lat degrees). You can supply any input CRS — the scripts call `to_crs("EPSG:4326")` on load.
