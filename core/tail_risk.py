# -*- coding: utf-8 -*-
"""
core/tail_risk.py — Tail Risk Analysis Engine
==============================================

Ported from srv_quant_system with pairs-trading adaptations.

Features:
- Expected Shortfall via 3 methods (parametric, historical, Cornish-Fisher)
- Parametric correlation stress testing
- Tail correlation diagnostic (panic coupling detection)
- VaR backtesting with Kupiec POF test
- Christoffersen independence test for clustered exceptions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger("core.tail_risk")


# ── dataclasses ───────────────────────────────────────────────────

@dataclass
class ESReport:
    """Expected Shortfall report."""
    confidence: float = 0.99
    parametric_es: float = np.nan
    historical_es: float = np.nan
    cornish_fisher_es: float = np.nan
    best_estimate: float = np.nan
    marginal_es: dict = field(default_factory=dict)  # per-pair contribution


@dataclass
class StressTestResult:
    """Correlation stress test result."""
    stress_level: str = "normal"
    stress_eta: float = 0.0
    base_portfolio_vol: float = np.nan
    stressed_portfolio_vol: float = np.nan
    vol_multiplier: float = 1.0
    stressed_var_99: float = np.nan


@dataclass
class TailCorrelationDiag:
    """Tail vs normal correlation diagnostic."""
    tail_corr: float = np.nan
    normal_corr: float = np.nan
    ratio: float = np.nan
    panic_coupling: bool = False
    pair_label: str = ""


@dataclass
class VaRBacktestResult:
    """VaR model backtest results."""
    var_level: float = 0.01
    expected_exceptions: float = np.nan
    actual_exceptions: int = 0
    exception_rate: float = np.nan
    kupiec_stat: float = np.nan
    kupiec_pvalue: float = np.nan
    kupiec_pass: bool = True
    christoffersen_stat: float = np.nan
    christoffersen_pvalue: float = np.nan
    christoffersen_pass: bool = True


# ── Expected Shortfall ────────────────────────────────────────────

def compute_parametric_es(
    returns: np.ndarray,
    confidence: float = 0.99,
) -> float:
    """Parametric (Gaussian) Expected Shortfall."""
    mu = np.nanmean(returns)
    sigma = np.nanstd(returns)
    if sigma < 1e-10:
        return 0.0
    alpha = 1 - confidence
    z_alpha = scipy_stats.norm.ppf(alpha)
    es = mu - sigma * scipy_stats.norm.pdf(z_alpha) / alpha
    return float(es)


def compute_historical_es(
    returns: np.ndarray,
    confidence: float = 0.99,
) -> float:
    """Historical simulation Expected Shortfall."""
    alpha = 1 - confidence
    threshold = np.nanpercentile(returns, alpha * 100)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return float(threshold)
    return float(np.nanmean(tail))


def compute_cornish_fisher_es(
    returns: np.ndarray,
    confidence: float = 0.99,
) -> float:
    """Cornish-Fisher Expected Shortfall.

    Adjusts for skewness and kurtosis beyond Gaussian assumption.
    Superior for pairs trading which has non-normal return distributions.
    """
    mu = np.nanmean(returns)
    sigma = np.nanstd(returns)
    if sigma < 1e-10:
        return 0.0

    skew = float(scipy_stats.skew(returns[~np.isnan(returns)]))
    kurt = float(scipy_stats.kurtosis(returns[~np.isnan(returns)]))  # excess

    alpha = 1 - confidence
    z = scipy_stats.norm.ppf(alpha)

    # Cornish-Fisher expansion
    z_cf = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * kurt / 24
        - (2 * z**3 - 5 * z) * (skew**2) / 36
    )

    var_cf = mu + sigma * z_cf

    # ES approximation: integrate tail of CF distribution
    # Use numerical integration of the CF-adjusted quantile
    n_points = 100
    alphas = np.linspace(0.0001, alpha, n_points)
    z_points = scipy_stats.norm.ppf(alphas)
    z_cf_points = (
        z_points
        + (z_points**2 - 1) * skew / 6
        + (z_points**3 - 3 * z_points) * kurt / 24
        - (2 * z_points**3 - 5 * z_points) * (skew**2) / 36
    )
    es_cf = mu + sigma * np.mean(z_cf_points)

    return float(es_cf)


def compute_expected_shortfall(
    returns: np.ndarray | pd.Series,
    confidence: float = 0.99,
) -> ESReport:
    """Compute ES via all three methods and select best estimate."""
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values

    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]

    report = ESReport(confidence=confidence)

    if len(returns) < 20:
        return report

    report.parametric_es = compute_parametric_es(returns, confidence)
    report.historical_es = compute_historical_es(returns, confidence)
    report.cornish_fisher_es = compute_cornish_fisher_es(returns, confidence)

    # best estimate: Cornish-Fisher if enough data, else historical
    if len(returns) >= 100:
        report.best_estimate = report.cornish_fisher_es
    else:
        report.best_estimate = report.historical_es

    return report


# ── Correlation Stress Testing ────────────────────────────────────

def parametric_correlation_stress(
    corr_matrix: np.ndarray,
    weights: np.ndarray,
    volatilities: np.ndarray,
    stress_eta: float = 0.5,
) -> StressTestResult:
    """Parametric correlation stress test.

    C_stress = (1 - eta) * C + eta * ones_matrix
    Simulates crisis where all correlations increase.
    """
    n = len(weights)
    ones = np.ones((n, n))
    c_stress = (1 - stress_eta) * corr_matrix + stress_eta * ones

    # ensure valid correlation matrix
    np.fill_diagonal(c_stress, 1.0)
    c_stress = np.clip(c_stress, -1.0, 1.0)

    # base portfolio vol
    vol_diag = np.diag(volatilities)
    cov_base = vol_diag @ corr_matrix @ vol_diag
    base_vol = np.sqrt(weights @ cov_base @ weights)

    # stressed portfolio vol
    cov_stress = vol_diag @ c_stress @ vol_diag
    stressed_vol = np.sqrt(weights @ cov_stress @ weights)

    # stress level labeling
    levels = {0.0: "normal", 0.25: "mild", 0.50: "moderate", 0.75: "severe", 1.0: "extreme"}
    label = "custom"
    for threshold, name in sorted(levels.items()):
        if stress_eta <= threshold:
            label = name
            break

    return StressTestResult(
        stress_level=label,
        stress_eta=stress_eta,
        base_portfolio_vol=float(base_vol),
        stressed_portfolio_vol=float(stressed_vol),
        vol_multiplier=float(stressed_vol / max(base_vol, 1e-10)),
        stressed_var_99=float(stressed_vol * 2.326),  # 99% VaR
    )


def run_stress_scenarios(
    corr_matrix: np.ndarray,
    weights: np.ndarray,
    volatilities: np.ndarray,
) -> list[StressTestResult]:
    """Run standard stress scenarios."""
    etas = [0.0, 0.25, 0.50, 0.75, 1.0]
    return [
        parametric_correlation_stress(corr_matrix, weights, volatilities, eta)
        for eta in etas
    ]


# ── Tail Correlation ─────────────────────────────────────────────

def tail_correlation_diagnostic(
    returns_x: np.ndarray | pd.Series,
    returns_y: np.ndarray | pd.Series,
    *,
    tail_pct: float = 0.05,
    pair_label: str = "",
) -> TailCorrelationDiag:
    """Compare correlation during tail events vs normal days.

    ratio > 1.5 indicates panic coupling.
    """
    if isinstance(returns_x, pd.Series):
        returns_x = returns_x.dropna().values
    if isinstance(returns_y, pd.Series):
        returns_y = returns_y.dropna().values

    x = np.asarray(returns_x, dtype=float)
    y = np.asarray(returns_y, dtype=float)

    # align lengths
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if n < 30:
        return TailCorrelationDiag(pair_label=pair_label)

    # normal correlation
    normal_corr = float(np.corrcoef(x, y)[0, 1])

    # tail: days where either stock has extreme loss
    threshold_x = np.percentile(x, tail_pct * 100)
    threshold_y = np.percentile(y, tail_pct * 100)
    tail_mask = (x <= threshold_x) | (y <= threshold_y)

    tail_x = x[tail_mask]
    tail_y = y[tail_mask]

    if len(tail_x) < 5:
        return TailCorrelationDiag(
            normal_corr=normal_corr,
            pair_label=pair_label,
        )

    tail_corr = float(np.corrcoef(tail_x, tail_y)[0, 1])
    ratio = tail_corr / max(abs(normal_corr), 0.001)

    return TailCorrelationDiag(
        tail_corr=tail_corr,
        normal_corr=normal_corr,
        ratio=ratio,
        panic_coupling=ratio > 1.5,
        pair_label=pair_label,
    )


# ── VaR Backtesting ──────────────────────────────────────────────

def kupiec_var_backtest(
    returns: np.ndarray,
    var_estimates: np.ndarray,
    var_level: float = 0.01,
) -> VaRBacktestResult:
    """Kupiec Proportion-of-Failures test for VaR model validation."""
    n = len(returns)
    exceptions = returns < var_estimates
    x = int(exceptions.sum())

    result = VaRBacktestResult(var_level=var_level)
    result.expected_exceptions = n * var_level
    result.actual_exceptions = x
    result.exception_rate = x / max(n, 1)

    if n < 10 or x == 0 or x == n:
        return result

    # Kupiec POF statistic
    p = var_level
    p_hat = x / n
    try:
        lr = -2 * (
            x * np.log(p / p_hat) + (n - x) * np.log((1 - p) / (1 - p_hat))
        )
        result.kupiec_stat = float(lr)
        result.kupiec_pvalue = float(1 - scipy_stats.chi2.cdf(lr, 1))
        result.kupiec_pass = result.kupiec_pvalue > 0.05
    except (ValueError, ZeroDivisionError):
        pass

    return result


def christoffersen_independence_test(
    returns: np.ndarray,
    var_estimates: np.ndarray,
) -> Tuple[float, float, bool]:
    """Christoffersen independence test for clustered VaR exceptions.

    Tests whether exceptions are independent (not clustered).
    Returns (stat, pvalue, pass_flag).
    """
    exceptions = (returns < var_estimates).astype(int)
    n = len(exceptions)

    if n < 10:
        return (np.nan, np.nan, True)

    # transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev, curr = exceptions[i-1], exceptions[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    # probabilities
    total0 = n00 + n01
    total1 = n10 + n11

    if total0 == 0 or total1 == 0 or (n01 + n11) == 0:
        return (0.0, 1.0, True)

    p01 = n01 / total0
    p11 = n11 / total1
    p = (n01 + n11) / (total0 + total1)

    try:
        lr = -2 * (
            _safe_log(1-p, n00 + n10) + _safe_log(p, n01 + n11)
            - _safe_log(1-p01, n00) - _safe_log(p01, n01)
            - _safe_log(1-p11, n10) - _safe_log(p11, n11)
        )
        pvalue = float(1 - scipy_stats.chi2.cdf(lr, 1))
        return (float(lr), pvalue, pvalue > 0.05)
    except (ValueError, ZeroDivisionError):
        return (np.nan, np.nan, True)


def _safe_log(p: float, n: int) -> float:
    if n == 0 or p <= 0 or p >= 1:
        return 0.0
    return n * np.log(p)


# ── Portfolio-level tail risk ─────────────────────────────────────

class TailRiskEngine:
    """Portfolio-level tail risk analysis for pairs trading."""

    def __init__(self, confidence: float = 0.99):
        self.confidence = confidence

    def analyze_portfolio(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray | None = None,
    ) -> dict:
        """Comprehensive tail risk analysis for the portfolio.

        returns_df: columns = pair labels, rows = dates, values = returns
        """
        if returns_df.empty:
            return {}

        n_pairs = returns_df.shape[1]
        if weights is None:
            weights = np.ones(n_pairs) / n_pairs

        # portfolio returns
        port_returns = (returns_df * weights).sum(axis=1).values

        # ES report
        es = compute_expected_shortfall(port_returns, self.confidence)

        # marginal ES per pair
        for i, col in enumerate(returns_df.columns):
            pair_ret = returns_df[col].values
            es_pair = compute_expected_shortfall(pair_ret, self.confidence)
            es.marginal_es[col] = es_pair.best_estimate

        # correlation matrix
        corr = returns_df.corr().values
        vols = returns_df.std().values * np.sqrt(252)

        # stress scenarios
        stress = run_stress_scenarios(corr, weights, vols)

        # tail correlations for adjacent pairs
        tail_diags = []
        cols = list(returns_df.columns)
        for i in range(min(len(cols), 10)):
            for j in range(i+1, min(len(cols), 10)):
                diag = tail_correlation_diagnostic(
                    returns_df[cols[i]], returns_df[cols[j]],
                    pair_label=f"{cols[i]}/{cols[j]}",
                )
                if diag.panic_coupling:
                    tail_diags.append(diag)

        return {
            "es_report": es,
            "stress_scenarios": stress,
            "panic_couplings": tail_diags,
            "portfolio_vol_annual": float(np.std(port_returns) * np.sqrt(252)),
            "max_drawdown": float(_max_drawdown(port_returns)),
            "skewness": float(scipy_stats.skew(port_returns)),
            "kurtosis": float(scipy_stats.kurtosis(port_returns)),
        }


def _max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from return series."""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    return float(np.min(drawdowns))
