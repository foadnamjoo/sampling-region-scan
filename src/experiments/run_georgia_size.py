from __future__ import annotations

"""Re-run the Georgia size sweep with more trials so the Geom-50 mean
curve is smooth in fig 7 (~20 trials → ~80 trials reduces noise by 2x).

Saves to buchin_attempt/georgia_size_sweep_grid100_t80.pkl (does NOT
overwrite the paper's georgia_size_sweep.pkl).
"""

# --- repo paths (injected by transform) ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
from _paths import REPO_ROOT, DATA, OUTPUTS, IO  # noqa: E402
ROOT = REPO_ROOT  # backward compatibility for scripts that reference ROOT
# --------------------------------------------

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# DYLD_LIBRARY_PATH for pyScan native libs should be set by the user; see README.
import run_experiment as RE  # noqa: E402

BU = OUTPUTS  # cached experiment results go here


def main():
    name = "georgia_size_sweep_grid100_t80"
    print(f"[{name}] re-running size sweep at 80 trials, grid 100...", flush=True)
    t0 = time.time()

    old_out = RE.OUT_DIR
    RE.OUT_DIR = BU
    try:
        pkg = RE.run_size_sweep(
            name=name,
            shp_path=str(RE.DATA /
                         "georgia" /
                         "GISPORTAL_GISOWNER01_GACOUNTIES10Polygon.shp"),
            x_base=-85.0, y_base=31.0,
            x_array=list(np.linspace(-84.5, -82.0, 10)),
            y_array=list(np.linspace(31.5, 34.0, 10)),
            p_prob=0.6,
            n_trials=80,
            grid_res=100,
            seed=RE.DEFAULT_SEED,
        )
    finally:
        RE.OUT_DIR = old_out

    print(f"\n[{name}] done in {time.time() - t0:.0f}s")
    print(f"  saved: {BU / (name + '.pkl')}")


if __name__ == "__main__":
    main()
