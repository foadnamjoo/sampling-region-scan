"""Fixed-A rerun, v2 -- full provenance, for the four paused groups.

Usage:  python fixedA_v2.py <group>       groups: usa gasize gaablation ksweep

WHY THE LOOPS ARE REIMPLEMENTED RATHER THAN PATCHED
run_experiment.py seeds numpy ONCE per experiment and then draws sequentially,
so there is no per-call seed to record. To satisfy full provenance we replicate
each driver loop verbatim (identical RNG call order: point_set_for_method draws
first, then the single rng.random(len(pts)) inside the scoring step) and, right
before each scoring call, snapshot rng.bit_generator.state. That snapshot plus
the loop indices reproduces any single call exactly.

EVERY RECORD CARRIES: group, method, trial, p_prob, q, k, target_index,
target_bounds, weighted, grid_res, rect, fValue, old_jd, fixedA_jd,
plus rng_state_before, call_index, experiment_seed, n_pts, n_measured.

GATES: reproduced old Jaccard must match the published pickle to < 1e-9 or the
group is REJECTED and no fixed-A array is accepted. For Georgia ablation the
reproduced Area Jaccard must match exactly as well.

Nothing existing is overwritten; outputs are <group>_fixedA_v2.pkl.
"""
from __future__ import annotations

import copy
import pickle
import random
import sys
import time
from pathlib import Path

ROOT = Path("/Users/foadnamjoo/PROJECT/PYSCAN")
# ORDER MATTERS: the figures copy of run_experiment.py is the one that produced
# the published pickles (its DATA root is pyscan/data/data). The staging copy has
# a different DATA root and must NOT shadow it; it is only needed for shape_floor.
sys.path.append("/Users/foadnamjoo/PyScan-Paper-staging/src")
sys.path.insert(0, str(ROOT / "pyscan/build"))
sys.path.insert(0, str(ROOT / "Paper_files/SIGSPATIAL_2026_figures/scripts"))

import numpy as np
import geopandas as gpd
import pyscan
from shapely.geometry import Point, Polygon

import run_experiment as RE
from shape_floor import reference_set

OUT = Path(__file__).resolve().parent
CACHED = ROOT / "Paper_files/SIGSPATIAL_2026_figures/cached_data"
BU = ROOT / "buchin_attempt"
EVAL_SEED, A_PER_REGION, TOL = 42, 500, 1e-9
SMOKE = False   # --smoke: 1 trial, 1 target, no gate; dumps one record

_A = None
_MASKS: dict = {}
RECORDS: list = []
_CALL = {"i": 0}


def build_A(gdf, label):
    global _A
    _A = np.asarray(reference_set(gdf, n_per_region=A_PER_REGION, seed=EVAL_SEED))[:, :2].astype(float)
    _MASKS.clear()
    print(f"[A:{label}] |A|={len(_A)} ({A_PER_REGION}/region, seed {EVAL_SEED})", flush=True)


def mask_for(target):
    key = tuple(np.round(target.bounds, 10))
    m = _MASKS.get(key)
    if m is None:
        x0, y0, x1, y1 = target.bounds
        if target.equals(Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])):
            m = ((_A[:, 0] >= x0) & (_A[:, 0] <= x1) & (_A[:, 1] >= y0) & (_A[:, 1] <= y1))
        else:
            m = np.fromiter((target.contains(Point(x, y)) for x, y in _A), bool, len(_A))
        _MASKS[key] = m
    return m


def pts_checksum(pts):
    """Stable checksum of the generated point coordinates."""
    import hashlib
    a = np.ascontiguousarray(np.asarray(pts, dtype=np.float64))
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]


def snapshot(rng):
    """RNG state immediately before point_set_for_method()."""
    return copy.deepcopy(rng.bit_generator.state)


def score(pts, target, p_prob, q, grid_res, rng, meta, want_area=False):
    """Verbatim RNG behaviour of run_experiment.one_trial_jaccard[_with_area]."""
    state = copy.deepcopy(rng.bit_generator.state)      # snapshot BEFORE the draw
    baseline, measured = [], []
    inside = np.array([target.contains(Point(x, y)) for x, y in pts])
    coins = rng.random(len(pts))                        # the single RNG call
    for i, (x, y) in enumerate(pts):
        baseline.append(pyscan.WPoint(1.0, float(x), float(y), 1.0))
        if coins[i] <= (p_prob if inside[i] else q):
            measured.append(pyscan.WPoint(1.0, float(x), float(y), 1.0))
    grid = pyscan.Grid(grid_res, measured, baseline)
    sg = pyscan.max_subgrid(grid, pyscan.KULLDORF)
    r = grid.toRectangle(sg)
    b = (r.lowX(), r.lowY(), r.upX(), r.upY())
    disc = Polygon([(b[0], b[1]), (b[0], b[3]), (b[2], b[3]), (b[2], b[1])])
    a_u_b = a_n_b = 0
    for i, (x, y) in enumerate(pts):
        if coins[i] > (p_prob if inside[i] else q):
            continue
        in_d = disc.contains(Point(float(x), float(y)))
        if inside[i] or in_d: a_u_b += 1
        if inside[i] and in_d: a_n_b += 1
    old_jd = ((a_u_b - a_n_b) / a_u_b) if a_u_b > 0 else 1.0
    inT = mask_for(target)
    d = (_A[:, 0] >= b[0]) & (_A[:, 0] <= b[2]) & (_A[:, 1] >= b[1]) & (_A[:, 1] <= b[3])
    u = int((inT | d).sum())
    new_jd = (u - int((inT & d).sum())) / u if u else 1.0
    try: fval = float(sg.fValue())
    except Exception: fval = float("nan")
    rec = {"call_index": _CALL["i"], "p_prob": float(p_prob), "q": float(q),
           "grid_res": int(grid_res), "n_pts": int(len(pts)),
           "n_measured": int(len(measured)),
           "target_bounds": [float(v) for v in target.bounds],
           "rect": [float(v) for v in b], "fValue": fval,
           "old_jd": float(old_jd), "fixedA_jd": float(new_jd),
           "rng_state_before_bernoulli": state,
           "pts_checksum": pts_checksum(pts)}
    rec.update(meta)
    RECORDS.append(rec); _CALL["i"] += 1
    area = RE.area_jaccard_distance(target, disc) if want_area else None
    return old_jd, new_jd, area


def load_gdf(shp, crs="EPSG:4326", bbox=None, name=""):
    gdf = gpd.read_file(shp)
    if str(gdf.crs) != crs: gdf = gdf.to_crs(crs)
    if bbox is not None:
        mnx, mny, mxx, mxy = bbox
        bb = gdf.geometry.bounds
        gdf = gdf[(bb["minx"] >= mnx) & (bb["maxx"] <= mxx) &
                  (bb["miny"] >= mny) & (bb["maxy"] <= mxy)].reset_index(drop=True)
        print(f"  [{name}] bbox filter kept {len(gdf)} regions", flush=True)
    return gdf


class GateFailure(RuntimeError):
    pass


def _flat(x):
    """Ragged-safe flatten to a 1-D float array, preserving order."""
    out = []
    def rec(v):
        if isinstance(v, (list, tuple)):
            for e in v: rec(e)
        elif isinstance(v, np.ndarray):
            for e in v.ravel(): out.append(float(e))
        else:
            out.append(float(v))
    rec(x)
    return np.asarray(out, dtype=float)


def _shape(x):
    """Nested length signature, ragged-safe."""
    if isinstance(x, (list, tuple)):
        return [len(x)] + (_shape(x[0]) if len(x) else [])
    if isinstance(x, np.ndarray):
        return list(x.shape)
    return []


def gate(label, repro, stored):
    """FAIL CLOSED: shape mismatch or diff >= TOL raises immediately."""
    sr, ss = _shape(repro), _shape(stored)
    if sr != ss:
        raise GateFailure(f"[gate:{label}] SHAPE MISMATCH reproduced={sr} published={ss}")
    a, b = _flat(repro), _flat(stored)
    if a.size != b.size:
        raise GateFailure(f"[gate:{label}] SIZE MISMATCH {a.size} vs {b.size}")
    d = float(np.abs(a - b).max()) if a.size else float("inf")
    if not (d < TOL):
        raise GateFailure(f"[gate:{label}] FAILED max|diff|={d:.3e} >= tol {TOL:.0e} "
                          f"-- fixed-A result NOT accepted")
    print(f"[gate:{label}] max |reproduced - published| = {d:.3e}  PASS "
          f"(shape {sr})", flush=True)
    return True, d


def save(name, extra):
    pkg = {"records": RECORDS, "n_records": len(RECORDS),
           "eval_seed": EVAL_SEED, "a_per_region": A_PER_REGION, "tol": TOL}
    pkg.update(extra)
    p = OUT / f"{name}_fixedA_v2.pkl"
    pickle.dump(pkg, open(p, "wb"))
    print(f"[save] {p}  ({len(RECORDS)} records)", flush=True)


# ------------------------------------------------------------------ USA ------
def g_usa():
    cfg = dict(RE.EXPERIMENTS["usa"][1])
    gdf = load_gdf(cfg["shp_path"], cfg.get("crs_target", "EPSG:4326"),
                   cfg.get("bbox_filter"), "usa")
    build_A(gdf, "usa")
    tgt, pq_grid, seed = cfg["target"], cfg["pq_grid"], cfg["seed"]
    nt, gr = cfg["n_trials"], cfg["grid_res"]
    if SMOKE: nt, pq_grid = 1, list(pq_grid)[:1]
    random.seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    old = {m: [] for m in RE.METHOD_NAMES}; new = {m: [] for m in RE.METHOD_NAMES}
    t0 = time.time()
    for trial in range(nt):
        for method in RE.METHOD_NAMES:
            k = RE.METHOD_K[method]
            st_pts = snapshot(rng); pts = RE.point_set_for_method(gdf, k, rng)
            ro, rn = [], []
            for pi, p in enumerate(pq_grid):
                o, n, _ = score(pts, tgt, float(p), RE.Q, gr, rng,
                    {"group": "usa", "method": method, "trial": trial, "k": k,
                     "target_index": 0, "weighted": None, "p_index": pi,
                     "experiment_seed": seed, "rng_state_before_points": st_pts})
                ro.append(o); rn.append(n)
            old[method].append(ro); new[method].append(rn)
        print(f"  [usa] trial {trial+1}/{nt} ({time.time()-t0:.0f}s)", flush=True)
    if SMOKE: return smoke_report("usa")
    pub = pickle.load(open(CACHED / "usa.pkl", "rb"))
    ok, d = gate("usa", [old[m] for m in RE.METHOD_NAMES],
                        [pub["methods"][m] for m in RE.METHOD_NAMES])
    save("usa", {"old": old, "fixed_A": new, "methods": RE.METHOD_NAMES,
                 "pq_diff": pub["pq_diff"], "n_trials": nt, "grid_res": gr,
                 "experiment_seed": seed, "gate_pass": ok, "gate_max_diff": d})


# ------------------------------------------------------------- size sweep ----
def g_gasize():
    shp = str(RE.DATA / "GeorgiaCounties" / "GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp")
    gdf = load_gdf(shp, name="gasize")
    build_A(gdf, "gasize")
    from shapely.ops import unary_union
    state_area = unary_union(gdf.geometry).area
    x_base, y_base = -85.0, 31.0
    xs = list(np.linspace(-84.5, -82.0, 10)); ys = list(np.linspace(31.5, 34.0, 10))
    p_prob, nt, gr, seed = 0.6, 80, 100, RE.DEFAULT_SEED
    if SMOKE: nt, xs, ys = 1, xs[:1], ys[:1]
    random.seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    old = {m: [[] for _ in xs] for m in RE.METHOD_NAMES}
    new = {m: [[] for _ in xs] for m in RE.METHOD_NAMES}
    area_pct = []; t0 = time.time()
    for t in range(len(xs)):
        tgt = Polygon([(x_base, y_base), (x_base, ys[t]), (xs[t], ys[t]), (xs[t], y_base)])
        area_pct.append(float(tgt.area / state_area * 100.0))
        for trial in range(nt):
            for method in RE.METHOD_NAMES:
                k = RE.METHOD_K[method]
                st_pts = snapshot(rng); pts = RE.point_set_for_method(gdf, k, rng)
                o, n, _ = score(pts, tgt, float(p_prob), RE.Q, gr, rng,
                    {"group": "gasize", "method": method, "trial": trial, "k": k,
                     "target_index": t, "weighted": None, "p_index": 0,
                     "experiment_seed": seed, "rng_state_before_points": st_pts})
                old[method][t].append(o); new[method][t].append(n)
        print(f"  [gasize] target {t+1}/{len(xs)} ({time.time()-t0:.0f}s)", flush=True)
    if SMOKE: return smoke_report("gasize")
    pub = pickle.load(open(BU / "georgia_size_sweep_grid100_t80.pkl", "rb"))
    ok, d = gate("gasize", [old[m] for m in RE.METHOD_NAMES],
                           [pub["methods"][m] for m in RE.METHOD_NAMES])
    save("gasize", {"old": old, "fixed_A": new, "methods": RE.METHOD_NAMES,
                    "area_pct": area_pct, "p_prob": p_prob, "n_trials": nt,
                    "grid_res": gr, "experiment_seed": seed,
                    "gate_pass": ok, "gate_max_diff": d})


# --------------------------------------------------------- georgia ablation --
def g_gaablation():
    shp = str(RE.DATA / "GeorgiaCounties" / "GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp")
    tgt = Polygon([(-85.0, 31.0), (-85.0, 32.89), (-83.61, 32.89), (-83.61, 31.0)])
    gdf = load_gdf(shp, name="gaablation")
    build_A(gdf, "gaablation")
    gr, seed, nt = 100, RE.DEFAULT_SEED, RE.DEFAULT_TRIALS
    pq_grid = RE.DEFAULT_PQ
    if SMOKE: nt, pq_grid = 1, list(pq_grid)[:1]
    n_geom_min = {"Centroid": 0, "Random Point": 1, "Geom 5": 1, "Geom 10": 5, "Geom 50": 20}
    results = {}
    for weighted, pubfile in ((False, "georgia_ablation_uniform_grid100.pkl"),
                              (True,  "georgia_ablation_weighted_pop_grid100.pkl")):
        shp_use = shp if not weighted else str(BU / "GeorgiaCounties_pop/ga_counties_pop.shp")
        g2 = load_gdf(shp_use, name="ga")
        wcol = "population" if weighted else None
        weights = g2[wcol].astype(float).values if weighted else None
        random.seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
        po = {m: [] for m in RE.METHOD_NAMES}; pn = {m: [] for m in RE.METHOD_NAMES}
        ao = {m: [] for m in RE.METHOD_NAMES}
        for trial in range(nt):
            for method in RE.METHOD_NAMES:
                k = RE.METHOD_K[method]
                if weighted and k > 0:
                    st_pts = snapshot(rng); pts = RE.weighted_point_set_for_method(g2, weights, k_max=k,
                                                           k_min=n_geom_min[method], rng=rng)
                else:
                    st_pts = snapshot(rng); pts = RE.point_set_for_method(g2, k, rng)
                ro, rn, ra = [], [], []
                for pi, p in enumerate(pq_grid):
                    o, n, a = score(pts, tgt, float(p), RE.Q, gr, rng,
                        {"group": f"gaablation_{'w' if weighted else 'u'}",
                         "method": method, "trial": trial, "k": k, "target_index": 0,
                         "weighted": bool(weighted), "p_index": pi,
                         "experiment_seed": seed, "rng_state_before_points": st_pts}, want_area=True)
                    ro.append(o); rn.append(n); ra.append(a)
                po[method].append(ro); pn[method].append(rn); ao[method].append(ra)
            print(f"  [ga {'w' if weighted else 'u'}] trial {trial+1}/{nt}", flush=True)
        if SMOKE:
            smoke_report(f"ga_{'weighted' if weighted else 'uniform'}")
            RECORDS.clear(); continue
        pub = pickle.load(open(BU / pubfile, "rb"))
        okp, dp = gate(f"ga_point_{'w' if weighted else 'u'}",
                       [po[m] for m in RE.METHOD_NAMES],
                       [pub["point_jaccard"][m] for m in RE.METHOD_NAMES])
        oka, da = gate(f"ga_AREA_{'w' if weighted else 'u'}",
                       [ao[m] for m in RE.METHOD_NAMES],
                       [pub["area_jaccard"][m] for m in RE.METHOD_NAMES])
        results[("weighted" if weighted else "uniform")] = {
            "old_point": po, "fixed_A_point": pn, "area_reproduced": ao,
            "gate_point_pass": okp, "gate_point_diff": dp,
            "gate_area_pass": oka, "gate_area_diff": da}
    if SMOKE: return True
    save("gaablation", {"arms": results, "methods": RE.METHOD_NAMES,
                        "pq_diff": (np.array(pq_grid) - RE.Q).round(4).tolist(),
                        "n_trials": nt, "grid_res": gr, "experiment_seed": seed,
                        "note": "Area Jaccard recomputed only to verify it is unchanged"})


# ---------------------------------------------------------------- k-sweep ----
MULTI_SEEDS = (7, 31, 67)      # matches render_all.MULTI_SEEDS


def g_ksweep():
    """All three seed batches per dataset, gated separately, then concatenated
    exactly as render_all._load_k_sweep_multi_seed does (by_k[k].extend)."""
    keys = ["k_sweep", "k_sweep_utah", "k_sweep_california",
            "k_sweep_nyc", "k_sweep_georgia", "k_sweep_usa"]

    # ---- preflight: which of the 18 published pickles actually exist ----
    if SMOKE: print("[smoke] preflight skipped")
    print("\n[preflight] expected 6 datasets x 3 seeds = 18 published pickles")
    expected, missing = [], []
    for key in keys:
        base = RE.EXPERIMENTS[key][1]["name"]
        for sd in MULTI_SEEDS:
            pn = base if sd == 7 else f"{base}_seed{sd}"
            ok = (CACHED / f"{pn}.pkl").exists()
            expected.append((base, sd, ok))
            if not ok: missing.append(pn)
            print(f"    {pn:34s} seed {sd:2d}  {'FOUND' if ok else 'MISSING'}")
    print(f"[preflight] {len(expected)-len(missing)}/18 present", flush=True)
    if missing and not SMOKE:
        raise GateFailure(f"[preflight] {len(missing)} expected seed pickle(s) missing: "
                          f"{missing} -- refusing to run an incomplete k-sweep")

    allout, all_pass = {}, True
    for key in keys:
        cfg = dict(RE.EXPERIMENTS[key][1])
        base = cfg["name"]                       # e.g. k_sweep_arkansas
        gdf = load_gdf(cfg["shp_path"], cfg.get("crs_target", "EPSG:4326"),
                       cfg.get("bbox_filter"), key)
        build_A(gdf, key)                        # A depends only on geometry
        tgt, ks, p_prob = cfg["target"], cfg["k_values"], cfg["p_prob"]
        nt, gr = cfg["n_trials"], cfg["grid_res"]
        if SMOKE: nt, ks = 1, list(ks)[:1]

        per_seed, comb_old, comb_new = {}, {int(k): [] for k in ks}, {int(k): [] for k in ks}
        for sd in MULTI_SEEDS:
            pubname = base if sd == 7 else f"{base}_seed{sd}"
            pubpath = CACHED / f"{pubname}.pkl"
            RECORDS.clear(); _CALL["i"] = 0
            random.seed(sd); np.random.seed(sd); rng = np.random.default_rng(sd)
            old = {int(k): [] for k in ks}; new = {int(k): [] for k in ks}
            for trial in range(nt):
                for k in ks:
                    st_pts = snapshot(rng)
                    pts = RE.point_set_for_method(gdf, int(k), rng)
                    o, n, _ = score(pts, tgt, float(p_prob), RE.Q, gr, rng,
                        {"group": base, "seed_batch": sd, "method": f"Geom {k}",
                         "trial": trial, "k": int(k), "target_index": 0,
                         "weighted": None, "p_index": 0, "experiment_seed": sd,
                         "rng_state_before_points": st_pts})
                    old[int(k)].append(o); new[int(k)].append(n)
            if SMOKE:
                smoke_report(f"ksweep_{base}_s{sd}"); return True
            pub = pickle.load(open(pubpath, "rb"))
            ok, d = gate(f"{pubname}", [old[int(k)] for k in ks],
                                       [pub["by_k"][int(k)] for k in ks])
            all_pass &= ok            # gate() raises on failure, so this is PASS-only
            per_seed[sd] = {"old": old, "fixed_A": new, "gate_pass": ok,
                            "gate_max_diff": d, "published": str(pubpath),
                            "records": list(RECORDS)}
            for k in ks:                          # renderer-identical concatenation
                comb_old[int(k)].extend(old[int(k)])
                comb_new[int(k)].extend(new[int(k)])
            print(f"  [{base}] seed {sd} done, gate {'PASS' if ok else 'FAIL'}", flush=True)

        allout[base] = {"per_seed": per_seed,
                        "combined_old": comb_old, "combined_fixed_A": comb_new,
                        "k_values": [int(k) for k in ks], "p_prob": p_prob,
                        "pq_diff": round(p_prob - RE.Q, 4),
                        "n_trials_total": sum(len(v) for v in comb_new.values()) // max(1, len(ks)),
                        "seeds_used": sorted(per_seed), "grid_res": gr,
                        "combination": "by_k[k].extend(trials) over seeds 7,31,67 "
                                       "-- identical to render_all._load_k_sweep_multi_seed"}
    RECORDS.clear()
    save("ksweep", {"datasets": allout, "multi_seeds": list(MULTI_SEEDS),
                    "all_gates_pass": bool(all_pass)})


REQUIRED_FIELDS = ["group", "method", "trial", "k", "target_bounds", "weighted",
                   "rng_state_before_points", "rng_state_before_bernoulli",
                   "pts_checksum", "rect", "fValue", "old_jd", "fixedA_jd"]


def smoke_report(group):
    if not RECORDS:
        print(f"[smoke:{group}] NO RECORDS"); return False
    r = RECORDS[0]
    miss = [f for f in REQUIRED_FIELDS if f not in r]
    print(f"[smoke:{group}] {len(RECORDS)} record(s); missing fields: {miss or 'NONE'}")
    if not miss:
        print(f"    method={r['method']} trial={r['trial']} k={r['k']} "
              f"tgt={[round(v,3) for v in r['target_bounds']]} weighted={r['weighted']}")
        print(f"    rect={[round(v,4) for v in r['rect']]} fValue={r['fValue']:.4f} "
              f"old={r['old_jd']:.4f} fixedA={r['fixedA_jd']:.4f}")
        print(f"    chk={r['pts_checksum']} "
              f"state_pts={r['rng_state_before_points']['bit_generator']} "
              f"state_bern={r['rng_state_before_bernoulli']['bit_generator']}")
    return not miss


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--smoke" in args:
        SMOKE = True
        args = [a for a in args if a != "--smoke"]
    fns = {"usa": g_usa, "gasize": g_gasize,
           "gaablation": g_gaablation, "ksweep": g_ksweep}
    for g in (args or list(fns)):
        print(f"\n{'='*30} {g} {'SMOKE' if SMOKE else ''} {'='*30}", flush=True)
        fns[g]()
