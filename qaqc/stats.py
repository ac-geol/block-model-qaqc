"""Weighted statistics (spec 7.1).

All functions ignore pairs where the value or the weight is missing, and pairs with a weight of
zero or less. All sums are done in float64.

Weighted quantiles
------------------
Values are sorted and the cumulative weight ``C_i`` is worked out. Each value sits at the
plotting position ``p_i = (C_i - 0.5 * w_i) / sum(w)``, the middle of its own share of the total
weight. The quantile at a target ``p`` is read off by straight-line interpolation between these
positions. Below the first position the smallest value is returned, and above the last position
the largest. With equal weights this gives the usual "Hazen" percentiles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike


def clean_pairs(
    values: ArrayLike, weights: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 values and weights with missing values and non-positive weights removed.

    With no weights, every valid value gets a weight of 1.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if weights is None:
        w = np.ones_like(v)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.shape != v.shape:
            raise ValueError("values and weights must have the same length")
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    return v[keep], w[keep]


def weighted_mean(values: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Weighted mean, sum(w * z) / sum(w). NaN if there are no valid pairs."""
    v, w = clean_pairs(values, weights)
    if v.size == 0:
        return float("nan")
    return float(np.sum(w * v) / np.sum(w))


def weighted_sd(values: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Weighted population standard deviation. NaN if there are no valid pairs."""
    v, w = clean_pairs(values, weights)
    if v.size == 0:
        return float("nan")
    mean = np.sum(w * v) / np.sum(w)
    return float(np.sqrt(np.sum(w * (v - mean) ** 2) / np.sum(w)))


def weighted_cv(values: ArrayLike, weights: ArrayLike | None = None) -> float:
    """Weighted coefficient of variation (SD / mean). NaN if the mean is zero or less."""
    mean = weighted_mean(values, weights)
    if not np.isfinite(mean) or mean <= 0:
        return float("nan")
    return weighted_sd(values, weights) / mean


def weighted_quantile(
    values: ArrayLike, weights: ArrayLike | None, q: float | ArrayLike
) -> float | np.ndarray:
    """Weighted quantile(s) at probability ``q`` (0 to 1), using the method in the module notes.

    Returns a float for a single ``q`` and an array for several. NaN if there are no valid pairs.
    """
    qa = np.asarray(q, dtype=np.float64)
    if np.any((qa < 0) | (qa > 1)):
        raise ValueError("quantile probabilities must be between 0 and 1")
    v, w = clean_pairs(values, weights)
    if v.size == 0:
        out = np.full(qa.shape, np.nan)
    else:
        order = np.argsort(v, kind="mergesort")
        v, w = v[order], w[order]
        cum = np.cumsum(w)
        pos = (cum - 0.5 * w) / cum[-1]
        out = np.interp(qa, pos, v)
    return float(out) if out.ndim == 0 else out


def weighted_summary(
    values: ArrayLike,
    weights: ArrayLike | None = None,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[str, Any]:
    """Count, total weight, mean, SD, CV, min, max and the given quantiles (keys ``p10`` etc.)."""
    v, w = clean_pairs(values, weights)
    out: dict[str, Any] = {
        "count": int(v.size),
        "weight": float(np.sum(w)) if v.size else 0.0,
        "mean": weighted_mean(v, w),
        "sd": weighted_sd(v, w),
        "cv": weighted_cv(v, w),
        "min": float(v.min()) if v.size else float("nan"),
        "max": float(v.max()) if v.size else float("nan"),
    }
    qs = weighted_quantile(v, w, list(quantiles)) if quantiles else []
    for p, val in zip(quantiles, np.atleast_1d(qs), strict=True):
        out[f"p{round(p * 100):g}"] = float(val)
    return out
