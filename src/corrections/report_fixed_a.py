"""Build the old-vs-fixed-A comparison report across all seven groups.

Reports, per dataset and method:
  * mean and max Point Jaccard shift (fixed-A minus old)
  * first p-q (or k) at which the mean curve reaches the 0.2 threshold, both metrics
  * method ranking under both metrics, and whether it changes
  * for the k-sweep: whether "diminishing returns beyond k~20" still holds
Prints a consolidated table; writes report_fixedA.json.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
METHODS = ["Centroid", "Random Point", "Geom 5", "Geom 10", "Geom 50"]
OUTPUT: dict = {}


def arr(x):
    return np.asarray(x, dtype=float)


def cross(mean_curve, xs, thr=0.2):
    idx = np.where(arr(mean_curve) <= thr)[0]
    return round(float(xs[idx[0]]), 3) if len(idx) else None


def shifts(old, new):
    o, n = arr(old).mean(axis=0), arr(new).mean(axis=0)
    d = n - o
    return float(d.mean()), float(np.abs(d).max())


def ranking(data, xs, floor=0.10):
    sel = arr(xs) >= floor
    return sorted(METHODS, key=lambda m: arr(data[m]).mean(axis=0)[sel].mean())


def curve_group(label, old, new, xs, note=""):
    print(f"\n{'='*80}\n{label}   {note}\n{'='*80}")
    print(f"{'method':16s} {'mean shift':>11} {'max shift':>10} "
          f"{'0.2 old':>9} {'0.2 fixA':>9}")
    rows = {}
    for m in METHODS:
        ms, mx = shifts(old[m], new[m])
        co = cross(arr(old[m]).mean(axis=0), xs)
        cn = cross(arr(new[m]).mean(axis=0), xs)
        rows[m] = {"mean_shift": ms, "max_shift": mx,
                   "cross_old": co, "cross_fixedA": cn,
                   "cross_changed": co != cn}
        print(f"{m:16s} {ms:+11.3f} {mx:10.3f} {str(co):>9} {str(cn):>9}")
    ro, rn = ranking(old, xs), ranking(new, xs)
    same = ro == rn
    print(f"\n  ranking old    : {'  <  '.join(ro)}")
    print(f"  ranking fixed A: {'  <  '.join(rn)}")
    print(f"  ranking changed: {'NO' if same else 'YES'}")
    OUTPUT[label] = {"methods": rows, "ranking_old": ro, "ranking_fixedA": rn,
                     "ranking_changed": not same}


def load(name):
    p = OUT / name
    return pickle.load(open(p, "rb")) if p.exists() else None


def main():
    # ---- NYC (pilot) ----
    d = load("nyc_fixedA_pilot.pkl")
    if d:
        curve_group("Figure 3 - NYC", d["old"], d["fixed_A"], d["pq_diff"],
                    f"(fidelity {d['fidelity_max_abs_diff']:.1e})")

    # ---- Utah / California ----
    for nm, lab in (("utah", "Figure 4 - Utah"), ("california", "Figure 5 - California")):
        d = load(f"{nm}_fixedA.pkl")
        if d:
            curve_group(lab, d["old"], d["fixed_A"], d["pq_diff"],
                        f"(fidelity {d['fidelity_max_abs_diff']:.1e})")

    # ---- USA ----
    d = load("usa_fixedA_v2.pkl")
    if d:
        curve_group("Figure 6 - USA", d["old"], d["fixed_A"], d["pq_diff"],
                    f"(gate {d['gate_max_diff']:.1e})")

    # ---- Georgia size sweep (x axis = target area %) ----
    d = load("gasize_fixedA_v2.pkl")
    if d:
        xs = d["area_pct"]
        old = {m: arr(d["old"][m]).T for m in METHODS}     # (trials, targets)
        new = {m: arr(d["fixed_A"][m]).T for m in METHODS}
        curve_group("Figure 7 - Georgia size sweep", old, new, xs,
                    f"(gate {d['gate_max_diff']:.1e}; x = target area %)")

    # ---- Georgia ablation (point only) ----
    d = load("gaablation_fixedA_v2.pkl")
    if d:
        for arm in ("uniform", "weighted"):
            a = d["arms"][arm]
            curve_group(f"Figure 11 - Georgia ablation ({arm}, Point JD)",
                        a["old_point"], a["fixed_A_point"], d["pq_diff"],
                        f"(point gate {a['gate_point_diff']:.1e}, "
                        f"AREA gate {a['gate_area_diff']:.1e} -> Area unchanged)")

    # ---- k-sweep ----
    d = load("ksweep_fixedA_v2.pkl")
    if d:
        print(f"\n{'='*80}\nFigure 12 - k-sweep (all gates pass: {d.get('all_gates_pass')})\n{'='*80}")
        ks_out = {}
        for base, g in d["datasets"].items():
            ks = g["k_values"]
            o = np.array([np.mean(g["combined_old"][k]) for k in ks])
            n = np.array([np.mean(g["combined_fixed_A"][k]) for k in ks])
            co, cn = cross(o, ks), cross(n, ks)
            # diminishing returns beyond k~20: improvement after k=20 vs before
            i20 = int(np.argmin(np.abs(np.array(ks) - 20)))
            gain_pre_o = float(o[0] - o[i20]); gain_post_o = float(o[i20] - o[-1])
            gain_pre_n = float(n[0] - n[i20]); gain_post_n = float(n[i20] - n[-1])
            print(f"\n  {base}  (seeds {g['seeds_used']}, {g['n_trials_total']} trials/k)")
            print(f"    {'k':>5} {'old':>8} {'fixedA':>9} {'shift':>8}")
            for j, k in enumerate(ks):
                print(f"    {k:5d} {o[j]:8.3f} {n[j]:9.3f} {n[j]-o[j]:+8.3f}")
            print(f"    reaches 0.2 at k: old={co}  fixedA={cn}")
            print(f"    gain k_min->20: old {gain_pre_o:.3f} / fixA {gain_pre_n:.3f} | "
                  f"gain 20->k_max: old {gain_post_o:.3f} / fixA {gain_post_n:.3f}")
            ks_out[base] = {"k_values": ks, "mean_old": o.tolist(), "mean_fixedA": n.tolist(),
                            "cross02_old": co, "cross02_fixedA": cn,
                            "gain_pre20_old": gain_pre_o, "gain_post20_old": gain_post_o,
                            "gain_pre20_fixedA": gain_pre_n, "gain_post20_fixedA": gain_post_n,
                            "seeds": g["seeds_used"], "n_trials_total": g["n_trials_total"]}
        OUTPUT["Figure 12 - k-sweep"] = ks_out

    json.dump(OUTPUT, open(OUT / "report_fixedA.json", "w"), indent=2, default=str)
    print(f"\n[report] wrote {OUT/'report_fixedA.json'}")


if __name__ == "__main__":
    main()
