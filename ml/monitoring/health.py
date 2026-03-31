# -*- coding: utf-8 -*-
"""
ml/monitoring/health.py — Statistical health primitives
=======================================================

Standalone functions for distribution-shift detection.
Used by FeatureDriftMonitor and ModelHealthMonitor.
"""

from __future__ import annotations

import warnings

import numpy as np


def psi_score(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI) between two distributions.

    Formula:
        PSI = Σ (actual% − expected%) × ln(actual% / expected%)

    Interpretation:
        PSI < 0.10  : no significant drift
        0.10–0.25   : moderate drift; monitor
        PSI > 0.25  : significant drift; consider retraining

    Edge cases:
        - Bins with zero current or reference count are handled with a small
          epsilon (1e-4) to avoid division-by-zero and log(0).
        - Returns NaN if fewer than 10 finite samples are available in either
          distribution.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if len(ref) < 10 or len(cur) < 10:
        return float("nan")

    # Use reference to define bin edges so both distributions use the same grid
    bin_edges = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    # Ensure edges are unique (can fail for constant distributions)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return float("nan")

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    eps = 1e-4
    ref_frac = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
    cur_frac = (cur_counts + eps) / (cur_counts.sum() + eps * len(cur_counts))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        psi = float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))

    return psi if np.isfinite(psi) else float("nan")


def kolmogorov_smirnov_drift(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    """
    KS test statistic and p-value for distribution shift.

    Returns (statistic, p_value). Returns (NaN, NaN) if scipy is unavailable
    or either array has fewer than 5 finite samples.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]

    if len(ref) < 5 or len(cur) < 5:
        return float("nan"), float("nan")

    try:
        from scipy.stats import ks_2samp
        result = ks_2samp(ref, cur)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        # Fallback: approximate KS statistic without scipy
        stat = _ks_statistic(ref, cur)
        return stat, float("nan")
    except Exception:
        return float("nan"), float("nan")


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate 2-sample KS statistic without scipy."""
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)

    all_vals = np.concatenate([a_sorted, b_sorted])
    all_vals = np.unique(all_vals)

    cdf_a = np.searchsorted(a_sorted, all_vals, side="right") / len(a_sorted)
    cdf_b = np.searchsorted(b_sorted, all_vals, side="right") / len(b_sorted)

    return float(np.max(np.abs(cdf_a - cdf_b)))
