"""Repo-relative path constants used by all paper scripts.

Computes paths from this file's location so the scripts work regardless of
where the repo is cloned. Outputs and cached experiment results go under
``outputs/``; shapefiles users download themselves live under ``data/``.

Usage:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from _paths import REPO_ROOT, DATA, OUTPUTS, IO

The ``pyscan`` package itself must already be importable (either built into
this venv or on ``PYTHONPATH``); see the project README for build steps.
"""
from __future__ import annotations
from pathlib import Path
import sys

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
SRC: Path = REPO_ROOT / "src"
DATA: Path = REPO_ROOT / "data"
OUTPUTS: Path = REPO_ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Cached experiment results live here (replaces the original
# ``buchin_attempt/arkansas_io`` directory in the authors' working tree).
IO: Path = OUTPUTS / "arkansas_io"
IO.mkdir(parents=True, exist_ok=True)

# Make ``src/`` importable so figure scripts can do ``import paper_plots``.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
