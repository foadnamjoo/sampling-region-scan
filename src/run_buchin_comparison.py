"""Phase-2 driver: Buchin Arkansas head-to-head.

Generates per-region case files matching the Geom-50 protocol used by the
existing Figs 8/9 (McClelland notebook lines 1117-1410), invokes the patched
ExperimentRunner Java binary one subprocess per (trial, p, mode), and appends
the wall-clock + discovered-window results to a per-figure CSV.

Static files (polys / names / pop) are built once from the Arkansas county
shapefile in EPSG:4326. Buchin therefore runs in lon-lat degree units — same
frame the pyscan Geom curves use — so the planted-rect sizes given here are
the literal (W,H) in degrees.

Per-region b(z): uniform 50 (matches Geom-50's implicit baseline, and Kulldorff
is invariant to uniform scale).
Per-region case-gen: 50 fresh uniform interior points per (trial, p), each
coin-flipped at rate p inside the planted target / q=0.20 outside, yeses summed
to give that region's count (variant ii of the original plan).
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from joblib import Parallel, delayed
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO as _IO_DIR  # noqa: E402

ROOT      = REPO_ROOT
SHP       = DATA / "arkansas" / "COUNTY_BOUNDARY.shp"
IO_DIR    = _IO_DIR
JAVA_CWD  = OUTPUTS / "Maarten-implementation"
JAVA_CP   = "build"

POLYS_FILE = IO_DIR / "arkansas-polys.txt"
NAMES_FILE = IO_DIR / "arkansas-names.txt"
POP_FILE   = IO_DIR / "arkansas-pop.txt"

# --- Experiment constants matched to McClelland notebook (Figs 8 & 9) ---------
N_POINTS_PER_REGION = 50
Q                   = 0.20
P_GRID              = np.round(np.arange(0.20, 0.95, 0.05), 4)  # 15 values

# Planted targets (lon, lat degrees). Verbatim from McClelland.
FIG8 = {
    "name":     "fig8",
    "target":   Polygon([(-93.5, 34.0), (-93.5, 35.5),
                         (-91.5, 35.5), (-91.5, 34.0)]),
    "rect_WH":  (2.0, 1.5),
    "center":   (-92.5, 34.75),
}
FIG9 = {
    "name":     "fig9",
    "target":   Polygon([(-92.85, 34.40), (-92.85, 35.10),
                         (-92.15, 35.10), (-92.15, 34.40)]),
    "rect_WH":  (0.7, 0.7),
    "center":   (-92.5, 34.75),
}

# Same multipliers we use for the Java size-grid sweep (subset of Buchin's
# original 9-multiplier grid that the advisor selected for the head-to-head).
SIZE_GRID = "0.5,0.7,1.0,1.3,1.5"

# ---------------------------------------------------------------------------
# Static file generation
# ---------------------------------------------------------------------------

def _load_arkansas() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(SHP).to_crs("EPSG:4326").reset_index(drop=True)
    if gdf["COUNTY"].str.contains(";").any():
        raise ValueError("County names contain ';' — would break Buchin loader")
    return gdf


def build_static_files(gdf: gpd.GeoDataFrame) -> None:
    """Write polys / names / pop files in Buchin's TEXT format."""
    IO_DIR.mkdir(parents=True, exist_ok=True)

    # ----- polygons file ----------------------------------------------------
    # Format: <int id>\n' x1 y1 x2 y2 ... '\n  (one ' ... ' block per ring,
    # multiple blocks allowed before the next id token).
    with open(POLYS_FILE, "w") as f:
        for i, geom in enumerate(gdf.geometry, start=1):
            rings = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
            f.write(f"{i}\n")
            for ring in rings:
                coords = list(ring.exterior.coords)
                parts  = " ".join(f"{x:.6f} {y:.6f}" for x, y in coords)
                f.write(f"' {parts} '\n")
            f.write("\n")

    # ----- names file -------------------------------------------------------
    with open(NAMES_FILE, "w") as f:
        for i, name in enumerate(gdf["COUNTY"], start=1):
            f.write(f"{i};{name};\n")

    # ----- pop file: uniform 50 per county ---------------------------------
    with open(POP_FILE, "w") as f:
        for name in gdf["COUNTY"]:
            f.write(f"{name};{N_POINTS_PER_REGION};\n")

    print(f"[static] polys -> {POLYS_FILE}")
    print(f"[static] names -> {NAMES_FILE}  ({len(gdf)} regions)")
    print(f"[static] pop   -> {POP_FILE}    (uniform {N_POINTS_PER_REGION})")


# ---------------------------------------------------------------------------
# Per-trial case generation (THE region-aggregated step)
# ---------------------------------------------------------------------------

def generate_cases(gdf: gpd.GeoDataFrame, target: Polygon, p: float,
                   seed: int) -> dict[str, int]:
    """Region-aggregated case map matching McClelland's per-point process.

    For each county: draw N_POINTS_PER_REGION uniform interior points; each
    point becomes a 'case' with probability p if inside `target` else q.
    Per-region case count = number of yes-points (the aggregation step).
    The Java side never sees individual points — only this {name: int} map.
    """
    rng    = random.Random(seed)
    counts = {}
    for _, row in gdf.iterrows():
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds
        drawn = 0
        cnt   = 0
        # rejection-sample to N_POINTS_PER_REGION uniform interior points
        while drawn < N_POINTS_PER_REGION:
            x = rng.uniform(minx, maxx)
            y = rng.uniform(miny, maxy)
            pt = Point(x, y)
            if not geom.contains(pt):
                continue
            drawn += 1
            rate = p if target.contains(pt) else Q
            if rng.random() <= rate:
                cnt += 1
        counts[row["COUNTY"]] = cnt
    return counts


def write_cases(counts: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for name, c in counts.items():
            f.write(f"{name};{c};\n")


# ---------------------------------------------------------------------------
# Java invocation
# ---------------------------------------------------------------------------

def call_buchin(cases_file: Path, window_mode: str, planted_size: str,
                out_csv: Path, trial_id: int, quiet: bool = False) -> None:
    cmd = [
        "java", "-cp", JAVA_CP,
        "app.ExperimentRunner",
        "--polys",        str(POLYS_FILE),
        "--names",        str(NAMES_FILE),
        "--pop",          str(POP_FILE),
        "--cases",        str(cases_file),
        "--window",       window_mode,
        "--planted-size", planted_size,
        "--size-grid",    SIZE_GRID,
        "--out",          str(out_csv),
        "--trial-id",     str(trial_id),
    ]
    if not quiet:
        print(f"[java] trial={trial_id} mode={window_mode} planted={planted_size}")
    stdout = subprocess.DEVNULL if quiet else None
    subprocess.run(cmd, cwd=JAVA_CWD, check=True, stdout=stdout)


# ---------------------------------------------------------------------------
# Round-trip driver (Phase-2 sanity test)
# ---------------------------------------------------------------------------

def run_roundtrip(fig: dict, p: float, trial: int) -> Path:
    """One trial, one p, both window modes — for Phase-2 sanity check."""
    gdf = _load_arkansas()
    fig_dir = IO_DIR / fig["name"]
    cases_dir = fig_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    p_tag = f"p{int(round(p*100)):03d}"
    cases_file = cases_dir / f"trial_{trial:03d}_{p_tag}.cas"
    out_csv    = fig_dir / f"roundtrip_{fig['name']}.csv"

    # Fresh seed per (trial, p) — same process as McClelland, independent stream.
    seed = 1_000_000 * trial + int(round(p * 1000))
    counts = generate_cases(gdf, fig["target"], p, seed=seed)
    write_cases(counts, cases_file)

    total_cases  = sum(counts.values())
    inside_total = sum(c for name, c in counts.items()
                       if fig["target"].contains(gdf.loc[gdf["COUNTY"] == name,
                                                          "geometry"].iloc[0].centroid))
    print(f"[cases] {cases_file.name}: total={total_cases}, "
          f"sum_in_centroid={inside_total}")

    # rect mode — Java parses "<W> <H>" (space-separated, bare numbers)
    W, H = fig["rect_WH"]
    call_buchin(cases_file, "rect", f"{W} {H}", out_csv, trial)

    # disk mode (equal-area disk) — Java parses bare "<r>"
    r = math.sqrt(W * H / math.pi)
    call_buchin(cases_file, "disk", f"{r:.6f}", out_csv, trial)

    return out_csv


# ---------------------------------------------------------------------------
# Phase 3: parallel accuracy sweep (wall-clock from this phase is NOT a
# reported runtime number — the per-call timing column is meaningless under
# joblib parallelism. Table-2 runtime comes from a separate serial pass.)
# ---------------------------------------------------------------------------

FIGS = {"fig8": FIG8, "fig9": FIG9}
N_TRIALS_DEFAULT = 20


def _seed_for(fig_name: str, trial: int, p: float) -> int:
    fig_offset = 0 if fig_name == "fig8" else 10_000_000_000
    return fig_offset + 1_000_000 * trial + int(round(p * 100000))


def _cases_path(fig_name: str, trial: int, p: float) -> Path:
    p_tag = f"p{int(round(p * 100)):03d}"
    return IO_DIR / fig_name / "cases" / f"trial_{trial:03d}_{p_tag}.cas"


def _per_call_csv(fig_name: str, trial: int, p: float, mode: str) -> Path:
    p_tag = f"p{int(round(p * 100)):03d}"
    return (IO_DIR / fig_name / "results"
            / f"trial_{trial:03d}_{p_tag}_{mode}.csv")


def _pregenerate_all_cases(gdf: gpd.GeoDataFrame, fig_names: list[str],
                            trials: list[int], p_values: list[float]) -> None:
    """Build every cases file up front, deterministically, in serial.

    Fast — case-gen is ~75ms per file. Doing it before the parallel sweep
    keeps the Java workers IO-only and avoids any per-process Python state.
    """
    todo = []
    for fig_name in fig_names:
        for t in trials:
            for p in p_values:
                path = _cases_path(fig_name, t, p)
                if path.exists() and path.stat().st_size > 0:
                    continue
                todo.append((fig_name, t, p, path))
    print(f"[cases] {len(todo)} files to generate "
          f"({len(fig_names)} figs × {len(trials)} trials × {len(p_values)} p)")
    t0 = time.time()
    for fig_name, t, p, path in todo:
        seed = _seed_for(fig_name, t, p)
        counts = generate_cases(gdf, FIGS[fig_name]["target"], p, seed=seed)
        write_cases(counts, path)
    if todo:
        print(f"[cases] done in {time.time()-t0:.1f}s")


def _one_java_job(fig_name: str, trial: int, p: float, mode: str) -> str:
    """Single Java subprocess; writes its own one-row CSV. Returns status str."""
    out = _per_call_csv(fig_name, trial, p, mode)
    if out.exists() and out.stat().st_size > 0:
        return f"skip {fig_name}/{out.name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = _cases_path(fig_name, trial, p)

    fig = FIGS[fig_name]
    if mode == "rect":
        W, H = fig["rect_WH"]
        planted = f"{W} {H}"
    else:
        W, H = fig["rect_WH"]
        planted = f"{math.sqrt(W * H / math.pi):.6f}"

    t0 = time.time()
    call_buchin(cases, mode, planted, out, trial, quiet=True)
    return f"ok   {fig_name}/{out.name}  ({time.time()-t0:.1f}s)"


def run_phase3(fig_names: list[str], n_trials: int, p_values: list[float],
                n_jobs: int) -> None:
    print(f"=== Phase 3: parallel accuracy sweep ===")
    print(f"figs={fig_names}  trials=1..{n_trials}  "
          f"p={[round(p,2) for p in p_values]}  n_jobs={n_jobs}")

    gdf    = _load_arkansas()
    if not POLYS_FILE.exists():
        build_static_files(gdf)

    trials = list(range(1, n_trials + 1))
    _pregenerate_all_cases(gdf, fig_names, trials, p_values)

    jobs = [(fn, t, p, m)
            for fn in fig_names
            for t in trials
            for p in p_values
            for m in ("rect", "disk")]
    print(f"[java] dispatching {len(jobs)} jobs to joblib (loky, n_jobs={n_jobs})")
    print(f"[java] NOTE: per-call wall-clock under parallelism is NOT the "
          f"Table-2 runtime number")

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_one_java_job)(fn, t, p, m) for (fn, t, p, m) in jobs
    )
    print(f"[java] all jobs done in {time.time()-t0:.1f}s")

    # Combine per-call CSVs into one per-fig CSV (accuracy data only).
    for fig_name in fig_names:
        combined = IO_DIR / f"{fig_name}_accuracy.csv"
        rows = []
        header = None
        for t in trials:
            for p in p_values:
                for m in ("rect", "disk"):
                    f = _per_call_csv(fig_name, t, p, m)
                    if not f.exists() or f.stat().st_size == 0:
                        print(f"[combine] MISSING: {f}")
                        continue
                    with open(f) as fp:
                        lines = fp.read().splitlines()
                    if header is None:
                        header = lines[0] + ",fig,p"
                    # one data row per file
                    rows.append(f"{lines[1]},{fig_name},{p:.4f}")
        with open(combined, "w") as fp:
            fp.write(header + "\n")
            for r in rows:
                fp.write(r + "\n")
        print(f"[combine] {combined}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-static", action="store_true",
                    help="(re)build polys/names/pop files from the shapefile")
    ap.add_argument("--roundtrip", action="store_true",
                    help="run Phase-2 sanity: 1 trial × 1 p on Fig 8 (rect + disk)")
    ap.add_argument("--fig", choices=["fig8", "fig9"], default="fig8")
    ap.add_argument("--p", type=float, default=0.60,
                    help="single p value for round-trip (default 0.60 → p-q=0.40)")
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--phase3", action="store_true",
                    help="Phase 3 parallel accuracy sweep (both figs, all trials × p × modes)")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="joblib n_jobs for Phase 3 (-1 = all cores)")
    ap.add_argument("--figs", nargs="+", choices=["fig8", "fig9"],
                    default=["fig8", "fig9"])
    args = ap.parse_args()

    if args.build_static:
        gdf = _load_arkansas()
        build_static_files(gdf)

    if args.roundtrip:
        if not POLYS_FILE.exists():
            print("[info] static files missing — building first")
            gdf = _load_arkansas()
            build_static_files(gdf)
        fig = FIG8 if args.fig == "fig8" else FIG9
        out = run_roundtrip(fig, p=args.p, trial=args.trial)
        print(f"[done] results -> {out}")
        # Pretty-print the CSV
        print("---- CSV ----")
        with open(out) as f:
            sys.stdout.write(f.read())

    if args.phase3:
        run_phase3(fig_names=args.figs, n_trials=args.n_trials,
                   p_values=list(P_GRID), n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
