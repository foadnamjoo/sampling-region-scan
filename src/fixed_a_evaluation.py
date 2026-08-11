"""Fixed reference-set evaluation for Point Jaccard distance.

The paper defines Point Jaccard distance on a fixed reference set A: 500 points
sampled uniformly inside every input region, constructed once and reused across
every method, k, p-q value, trial and experiment seed.

Earlier synthetic pipelines instead evaluated the overlap on each trial's
Bernoulli-selected measured points, so every method was scored on a different
support. This module provides the fixed-A evaluation and is the default for new
synthetic experiments; the historical behaviour remains available so previously
published numbers can still be reproduced exactly.

Typical use
-----------
    from fixed_a_evaluation import EvalSet

    ev = EvalSet.build(gdf)                 # once per dataset
    ...
    jd = ev.jaccard(target, rect_bounds)    # per discovered rectangle

Design notes
------------
* A is built with its OWN generator, seeded by ``eval_seed`` (default 42), so
  constructing it perturbs no scan location and no Bernoulli draw.
* A is an EVALUATION set. It is deliberately distinct from any 500-point
  *sampling reservoir* that scan points may be drawn from.
* In-target masks are cached per target, so sweeps that vary the target
  (e.g. the Georgia size sweep) stay correct and fast.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from shapely.geometry import Point, Polygon

DEFAULT_EVAL_SEED = 42
DEFAULT_POINTS_PER_REGION = 500

# Set False only to reproduce pre-correction published numbers.
EVALUATE_ON_FIXED_A = True


def sample_uniform_in_polygon(poly, n: int, rng: np.random.Generator) -> np.ndarray:
    """n uniform points strictly inside poly, by rejection from its bbox."""
    minx, miny, maxx, maxy = poly.bounds
    out = np.empty((n, 2), dtype=float)
    filled = 0
    while filled < n:
        bx = rng.uniform(minx, maxx, size=n * 3)
        by = rng.uniform(miny, maxy, size=n * 3)
        for x, y in zip(bx, by):
            if poly.contains(Point(x, y)):
                out[filled] = (x, y)
                filled += 1
                if filled == n:
                    break
    return out


@dataclass
class EvalSet:
    """A fixed, reusable evaluation set for Point Jaccard distance."""

    points: np.ndarray                     # (N, 2)
    eval_seed: int = DEFAULT_EVAL_SEED
    points_per_region: int = DEFAULT_POINTS_PER_REGION
    n_regions: int = 0
    _masks: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- build --
    @classmethod
    def build(cls, gdf, points_per_region: int = DEFAULT_POINTS_PER_REGION,
              eval_seed: int = DEFAULT_EVAL_SEED) -> "EvalSet":
        rng = np.random.default_rng(eval_seed)      # independent of experiment RNG
        pts = np.vstack([sample_uniform_in_polygon(g, points_per_region, rng)
                         for g in gdf.geometry])
        return cls(points=pts, eval_seed=eval_seed,
                   points_per_region=points_per_region, n_regions=len(gdf))

    # ----------------------------------------------------------------- use ---
    def mask(self, target: Polygon) -> np.ndarray:
        """Cached boolean mask of which evaluation points fall inside target."""
        key = tuple(np.round(target.bounds, 10))
        m = self._masks.get(key)
        if m is None:
            x0, y0, x1, y1 = target.bounds
            if target.equals(Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])):
                m = ((self.points[:, 0] >= x0) & (self.points[:, 0] <= x1) &
                     (self.points[:, 1] >= y0) & (self.points[:, 1] <= y1))
            else:
                m = np.fromiter((target.contains(Point(x, y)) for x, y in self.points),
                                bool, len(self.points))
            self._masks[key] = m
        return m

    def jaccard(self, target: Polygon, rect_bounds: Iterable[float]) -> float:
        """Point Jaccard distance between target and an axis-aligned rectangle."""
        x0, y0, x1, y1 = rect_bounds
        in_t = self.mask(target)
        in_d = ((self.points[:, 0] >= x0) & (self.points[:, 0] <= x1) &
                (self.points[:, 1] >= y0) & (self.points[:, 1] <= y1))
        union = int((in_t | in_d).sum())
        if union == 0:
            return 1.0
        return (union - int((in_t & in_d).sum())) / union

    def jaccard_polygon(self, target: Polygon, discovered: Polygon) -> float:
        """Same, for a non-rectangular discovered region."""
        in_t = self.mask(target)
        in_d = np.fromiter((discovered.contains(Point(x, y)) for x, y in self.points),
                           bool, len(self.points))
        union = int((in_t | in_d).sum())
        if union == 0:
            return 1.0
        return (union - int((in_t & in_d).sum())) / union

    # ------------------------------------------------------------ provenance --
    def provenance(self) -> dict:
        return {"eval_seed": self.eval_seed,
                "points_per_region": self.points_per_region,
                "n_regions": self.n_regions,
                "n_eval_points": int(len(self.points)),
                "reused_across": ["method", "k", "p_minus_q", "trial", "experiment_seed"]}


def record_template(*, group, method, trial, k, p_prob, q, target, rect_bounds,
                    f_value, old_jd, fixed_a_jd, experiment_seed,
                    target_index=0, weighted=None, grid_res=None,
                    rng_state_before_points=None, rng_state_before_bernoulli=None,
                    points_checksum=None, n_points=None, n_measured=None) -> dict:
    """Canonical per-scan provenance row. Keep every field populated."""
    return {"group": group, "method": method, "trial": trial, "k": k,
            "p_prob": float(p_prob), "q": float(q),
            "target_index": target_index,
            "target_bounds": [float(v) for v in target.bounds],
            "weighted": weighted, "grid_res": grid_res,
            "rect": [float(v) for v in rect_bounds], "fValue": f_value,
            "old_jd": old_jd, "fixedA_jd": fixed_a_jd,
            "experiment_seed": experiment_seed,
            "rng_state_before_points": rng_state_before_points,
            "rng_state_before_bernoulli": rng_state_before_bernoulli,
            "points_checksum": points_checksum,
            "n_points": n_points, "n_measured": n_measured}


def points_checksum(pts) -> str:
    """Stable SHA-1 of generated point coordinates, for reconstruction checks."""
    import hashlib
    a = np.ascontiguousarray(np.asarray(pts, dtype=np.float64))
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]
