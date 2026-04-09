# -*- coding: utf-8 -*-
# core/false_discovery.py
"""Multiple-Comparison Correction for Pairs Trading Research
=============================================================

Institutional-grade false-discovery control when testing multiple pairs,
multiple parameter sets, or multiple strategies against the same dataset.

Implements
----------
1. **Bonferroni correction** — conservative; FWER control.
2. **Benjamini-Hochberg (BH)** — less conservative; FDR control at level q.
3. **Deflated Sharpe Ratio (DSR) via trial count** — already in
   ``core.walk_forward_engine``; here we expose a *pair-level* wrapper.
4. **MinTRL (Minimum Track-Record Length)** — Bailey & Lopez de Prado (2014):
   minimum OOS history needed for a Sharpe to be significant given n_trials.
5. **FDR summary table** — pass in a list of (pair_id, sharpe, n_obs) tuples
   and get back a DataFrame with adjusted p-values and pass/fail flags.

References
----------
Bailey, D.H., and M. López de Prado (2014):
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
    Overfitting and Non-Normality."  *Journal of Portfolio Management*, 40(5).

Benjamini, Y., and Y. Hochberg (1995):
    "Controlling the False Discovery Rate: A Practical and Powerful Approach
    to Multiple Testing."  *Journal of the Royal Statistical Society*, 57(1).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PairTestResult:
    """Input record for a single pair / strategy."""
    pair_id: str
    # Observed annualised Sharpe ratio (OOS or IS)
    sharpe: float
    # Number of return observations (trading days in OOS window)
    n_obs: int
    # Optional: number of trials / pairs tested simultaneously
    n_trials: int = 1
    # Optional: skewness and kurtosis of returns (for SR p-value)
    skewness: float = 0.0
    kurtosis: float = 3.0


@dataclass
class MultipleTestingReport:
    """Results after multiple-comparison correction."""
    # Raw results (one row per pair)
    details: pd.DataFrame
    # Number of pairs passing each correction at significance level alpha
    n_bonferroni: int
    n_bh_fdr: int
    n_dsr_gate: int
    # Global warning if any signals pass raw but fail adjusted tests
    has_spurious_discoveries: bool
    # Summary narrative
    narrative: str


# ---------------------------------------------------------------------------
# Core p-value computation
# ---------------------------------------------------------------------------

def sharpe_pvalue(
    sharpe: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute one-sided p-value for H0: true Sharpe ≤ 0.

    Uses the López de Prado (2014) corrected SR standard error that accounts
    for non-normality (skewness, excess kurtosis):

        SE(SR) = sqrt( (1 - skew×SR + ((kurt-3)/4)×SR²) / (T-1) )

    Parameters
    ----------
    sharpe : float
        Annualised (or period) Sharpe ratio.
    n_obs : int
        Number of return observations.
    skewness : float
        Third standardised moment of returns.
    kurtosis : float
        Fourth standardised moment (NOT excess kurtosis).

    Returns
    -------
    float
        One-sided p-value for H0: SR ≤ 0.  Smaller = stronger evidence.
    """
    if not np.isfinite(sharpe) or n_obs < 4:
        return 1.0

    excess_kurt = kurtosis - 3.0
    var_sr = (
        1.0
        - skewness * sharpe
        + (excess_kurt / 4.0) * sharpe ** 2
    ) / max(n_obs - 1, 1)

    if var_sr <= 0:
        var_sr = 1.0 / max(n_obs - 1, 1)

    se_sr = math.sqrt(var_sr)
    if se_sr < 1e-12:
        return 0.0 if sharpe > 0 else 1.0

    z = sharpe / se_sr
    # One-sided: P(Z > z) under H0
    p = float(1.0 - stats.norm.cdf(z))
    return float(np.clip(p, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Bonferroni correction
# ---------------------------------------------------------------------------

def bonferroni_adjust(p_values: Sequence[float], alpha: float = 0.05) -> List[float]:
    """Return Bonferroni-corrected p-values.

    Bonferroni adjusted p_i = min(p_i × n_tests, 1.0).
    """
    n = len(p_values)
    if n == 0:
        return []
    return [float(min(p * n, 1.0)) for p in p_values]


def bonferroni_reject(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Return boolean reject array under Bonferroni correction."""
    adj = bonferroni_adjust(p_values, alpha)
    return [p <= alpha for p in adj]


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction
# ---------------------------------------------------------------------------

def bh_adjust(p_values: Sequence[float], fdr_level: float = 0.05) -> List[float]:
    """Benjamini-Hochberg adjusted p-values (step-up procedure).

    Returns adjusted p-values in the SAME ORDER as the input.
    Formula: adjusted_p_i = p_(i) × n / i  (for sorted rank i).
    The minimum of all subsequent adjusted values is taken to ensure
    monotonicity (Yekutieli & Benjamini 2001).
    """
    n = len(p_values)
    if n == 0:
        return []

    p_arr = np.array(p_values, dtype=float)
    order = np.argsort(p_arr)
    ranks = np.arange(1, n + 1)

    # Adjusted values in sorted order
    sorted_adjusted = p_arr[order] * n / ranks

    # Enforce monotonicity from right to left (cumulative minimum)
    for i in range(n - 2, -1, -1):
        sorted_adjusted[i] = min(sorted_adjusted[i], sorted_adjusted[i + 1])
    sorted_adjusted = np.clip(sorted_adjusted, 0.0, 1.0)

    # Map back to original order
    result = np.empty(n, dtype=float)
    result[order] = sorted_adjusted
    return list(result)


def bh_reject(p_values: Sequence[float], fdr_level: float = 0.05) -> List[bool]:
    """Return boolean reject array under BH FDR correction at level q."""
    adj = bh_adjust(p_values, fdr_level)
    return [p <= fdr_level for p in adj]


# ---------------------------------------------------------------------------
# MinTRL: minimum track-record length
# ---------------------------------------------------------------------------

def min_track_record_length(
    sharpe: float,
    n_trials: int,
    alpha: float = 0.05,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Minimum number of OOS observations required for SR to be significant.

    Derived from Bailey & López de Prado (2014), eq. (11):

        MinTRL = 1 + (1 - skew×SR + ((kurt-3)/4)×SR²) × (z_{1-alpha/n_trials} / SR)²

    Parameters
    ----------
    sharpe : float
        Observed annualised Sharpe.
    n_trials : int
        Number of trials / strategies tested (for Bonferroni correction inside DSR).
    alpha : float
        Significance level (default 0.05).
    skewness, kurtosis : float
        Return distribution moments.

    Returns
    -------
    float
        Minimum number of observations.  If the strategy does not have this
        many OOS observations, it is NOT yet validated.
    """
    if not np.isfinite(sharpe) or abs(sharpe) < 1e-9:
        return float("inf")

    n_trials = max(int(n_trials), 1)
    adjusted_alpha = alpha / n_trials  # Bonferroni-corrected significance level
    z = float(stats.norm.ppf(1.0 - adjusted_alpha))

    excess_kurt = kurtosis - 3.0
    variance_term = (
        1.0
        - skewness * sharpe
        + (excess_kurt / 4.0) * sharpe ** 2
    )
    if variance_term <= 0:
        variance_term = 1.0

    min_trl = 1.0 + variance_term * (z / sharpe) ** 2
    return float(max(min_trl, 1.0))


# ---------------------------------------------------------------------------
# DSR per pair (thin wrapper around walk_forward_engine)
# ---------------------------------------------------------------------------

def _get_dsr(sharpe: float, n_trials: int, n_obs: int) -> float:
    """Compute DSR for a single pair, with graceful fallback."""
    try:
        from core.walk_forward_engine import deflated_sharpe_ratio
        return float(deflated_sharpe_ratio(sr_obs=sharpe, n_trials=n_trials, n_obs=n_obs))
    except Exception as exc:
        logger.debug("DSR computation failed: %s", exc)
        # Rough fallback: Sharpe / sqrt(n_trials) normalised to [0, 1]
        if not np.isfinite(sharpe) or n_obs < 4:
            return 0.0
        sr_adjusted = sharpe - (abs(sharpe) / max(n_trials, 1)) * 0.3
        return float(min(max(stats.norm.cdf(sr_adjusted * math.sqrt(n_obs)) * 2 - 1, 0.0), 1.0))


# ---------------------------------------------------------------------------
# Main API: FDR summary table
# ---------------------------------------------------------------------------

def run_multiple_testing_correction(
    test_results: Sequence[PairTestResult],
    alpha: float = 0.05,
    fdr_level: float = 0.05,
    dsr_threshold: float = 0.65,
) -> MultipleTestingReport:
    """Apply Bonferroni, BH-FDR, and DSR corrections to a list of pair results.

    Parameters
    ----------
    test_results:
        List of PairTestResult objects (one per pair/strategy).
    alpha:
        FWER significance level for Bonferroni correction (default 0.05).
    fdr_level:
        FDR target for Benjamini-Hochberg correction (default 0.05).
    dsr_threshold:
        DSR gate level (default 0.65).

    Returns
    -------
    MultipleTestingReport
        Contains a DataFrame with per-pair corrections and aggregate summary.
    """
    if not test_results:
        empty_df = pd.DataFrame(columns=[
            "pair_id", "sharpe", "n_obs", "n_trials",
            "raw_pvalue", "bonferroni_pvalue", "bh_pvalue", "dsr",
            "reject_bonferroni", "reject_bh", "pass_dsr",
            "min_trl", "has_sufficient_history",
        ])
        return MultipleTestingReport(
            details=empty_df,
            n_bonferroni=0,
            n_bh_fdr=0,
            n_dsr_gate=0,
            has_spurious_discoveries=False,
            narrative="No test results provided.",
        )

    n_tests = len(test_results)

    rows = []
    raw_pvals = []
    for r in test_results:
        p_raw = sharpe_pvalue(r.sharpe, r.n_obs, r.skewness, r.kurtosis)
        raw_pvals.append(p_raw)

        dsr = _get_dsr(r.sharpe, max(r.n_trials, n_tests), r.n_obs)
        min_trl = min_track_record_length(
            sharpe=r.sharpe,
            n_trials=max(r.n_trials, n_tests),
            alpha=alpha,
            skewness=r.skewness,
            kurtosis=r.kurtosis,
        )

        rows.append({
            "pair_id": r.pair_id,
            "sharpe": r.sharpe,
            "n_obs": r.n_obs,
            "n_trials": r.n_trials,
            "raw_pvalue": p_raw,
            "dsr": dsr,
            "min_trl": min_trl,
            "has_sufficient_history": bool(r.n_obs >= min_trl),
        })

    bon_pvals = bonferroni_adjust(raw_pvals, alpha=alpha)
    bh_pvals = bh_adjust(raw_pvals, fdr_level=fdr_level)

    for i, row in enumerate(rows):
        row["bonferroni_pvalue"] = bon_pvals[i]
        row["bh_pvalue"] = bh_pvals[i]
        row["reject_bonferroni"] = bool(bon_pvals[i] <= alpha)
        row["reject_bh"] = bool(bh_pvals[i] <= fdr_level)
        row["pass_dsr"] = bool(row["dsr"] >= dsr_threshold)

    df = pd.DataFrame(rows)

    n_bonferroni = int(df["reject_bonferroni"].sum())
    n_bh = int(df["reject_bh"].sum())
    n_dsr = int(df["pass_dsr"].sum())

    # "Spurious discoveries" = pass raw but fail all adjusted tests
    if "raw_pvalue" in df.columns:
        raw_reject = df["raw_pvalue"] <= alpha
        any_adjusted = df["reject_bonferroni"] | df["reject_bh"] | df["pass_dsr"]
        n_spurious = int((raw_reject & ~any_adjusted).sum())
        has_spurious = n_spurious > 0
    else:
        has_spurious = False
        n_spurious = 0

    # Narrative
    parts = [
        f"Tested {n_tests} pairs/strategies.",
        f"Bonferroni rejects: {n_bonferroni}/{n_tests}.",
        f"BH-FDR rejects: {n_bh}/{n_tests} (q={fdr_level}).",
        f"DSR gate passed: {n_dsr}/{n_tests} (threshold={dsr_threshold}).",
    ]
    if has_spurious:
        parts.append(
            f"WARNING: {n_spurious} pairs pass raw p-value but FAIL all adjusted tests "
            "— likely spurious discoveries. DO NOT promote."
        )
    else:
        parts.append("No spurious discoveries detected after multiple-comparison correction.")
    narrative = " ".join(parts)

    return MultipleTestingReport(
        details=df,
        n_bonferroni=n_bonferroni,
        n_bh_fdr=n_bh,
        n_dsr_gate=n_dsr,
        has_spurious_discoveries=has_spurious,
        narrative=narrative,
    )


# ---------------------------------------------------------------------------
# Convenience: quick FDR check on optimizer DataFrame
# ---------------------------------------------------------------------------

def fdr_check_optimizer_results(
    results_df: pd.DataFrame,
    n_obs: int,
    sharpe_col: str = "sharpe",
    fdr_level: float = 0.05,
    dsr_threshold: float = 0.65,
) -> MultipleTestingReport:
    """One-liner FDR check on the output of ``run_optimization()``.

    Treats every trial as a separate test hypothesis (each trial = one pair
    of parameter settings tested against the same dataset).

    Parameters
    ----------
    results_df:
        Output DataFrame of ``run_optimization()``.
    n_obs:
        Number of OOS return observations (use length of test window).
    sharpe_col:
        Sharpe column name in results_df.
    fdr_level:
        BH FDR target.
    dsr_threshold:
        DSR gate threshold.

    Returns
    -------
    MultipleTestingReport
    """
    df = results_df.dropna(subset=[sharpe_col]).copy()
    if df.empty:
        return run_multiple_testing_correction([], fdr_level=fdr_level, dsr_threshold=dsr_threshold)

    n_trials = len(df)
    test_results = [
        PairTestResult(
            pair_id=str(row.get("trial_id", i)),
            sharpe=float(row[sharpe_col]),
            n_obs=n_obs,
            n_trials=n_trials,
        )
        for i, row in df.iterrows()
        if np.isfinite(float(row[sharpe_col]))
    ]
    return run_multiple_testing_correction(
        test_results,
        fdr_level=fdr_level,
        dsr_threshold=dsr_threshold,
    )
