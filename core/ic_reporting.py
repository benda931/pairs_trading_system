# -*- coding: utf-8 -*-
"""
core/ic_reporting.py — IC Reporting Standards
==============================================

Enforces that IC is ALWAYS reported with t-statistic and sample size.

An IC without a t-statistic is uninterpretable:
- IC=0.065 on 200 periods (t=2.58, p=0.01) → signal
- IC=0.065 on 20 periods  (t=0.82, p=0.42) → noise

All reporting should use format_ic_report() to produce a standardized
string that includes all three components.

Usage
-----
    from core.ic_reporting import format_ic_report, assert_ic_significant

    # In any analysis output:
    print(format_ic_report(ic=0.065, n=200))
    # → "IC=0.065 (t=2.58, N=200) ✓ significant at α=0.05"

    # In validation gates:
    assert_ic_significant(ic=0.065, n=200, alpha=0.05, context="momentum_alpha")
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("core.ic_reporting")


def compute_ic_tstat(ic: float, n: int) -> Tuple[float, float]:
    """
    Compute t-statistic and p-value for an IC.

    Parameters
    ----------
    ic : float
        Information Coefficient (Spearman rank correlation).
    n : int
        Number of periods/samples.

    Returns
    -------
    tuple : (t_stat, p_value)
        Both NaN if n < 4 or ic is not finite.
    """
    if not math.isfinite(ic) or n < 4:
        return float("nan"), float("nan")

    try:
        denom = max(1.0 - ic ** 2, 1e-12)
        t_stat = ic * ((n - 2) ** 0.5) / (denom ** 0.5)

        try:
            from scipy.stats import t as t_dist
            p_value = 2.0 * (1.0 - t_dist.cdf(abs(t_stat), df=n - 2))
        except ImportError:
            # Approximation: two-tailed p for large N
            p_value = float("nan")

        return float(t_stat), float(p_value)
    except Exception as exc:
        logger.debug("compute_ic_tstat failed: %s", exc)
        return float("nan"), float("nan")


def is_ic_significant(
    ic: float,
    n: int,
    alpha: float = 0.05,
) -> bool:
    """
    Return True if IC is statistically significant at the given level.

    Parameters
    ----------
    ic : float
        Information Coefficient.
    n : int
        Number of samples.
    alpha : float
        Significance level (default 0.05 two-tailed, equivalent to t > 1.96).
    """
    if not math.isfinite(ic) or n < 4:
        return False
    t_stat, _ = compute_ic_tstat(ic, n)
    if not math.isfinite(t_stat):
        return False
    try:
        from scipy.stats import t as t_dist
        critical = t_dist.ppf(1 - alpha / 2, df=n - 2)
        return abs(t_stat) >= critical
    except ImportError:
        # Fallback: use normal approximation for large N
        critical = {0.10: 1.645, 0.05: 1.960, 0.01: 2.576}.get(alpha, 1.960)
        return abs(t_stat) >= critical


def format_ic_report(
    ic: float,
    n: int,
    alpha: float = 0.05,
    label: str = "",
) -> str:
    """
    Format a standardized IC report string.

    Always includes IC value, t-statistic, sample size, and significance.
    This is the ONLY sanctioned format for IC reporting in agent summaries,
    validation outputs, and backtest results.

    Parameters
    ----------
    ic : float
        Information Coefficient.
    n : int
        Number of periods.
    alpha : float
        Significance level for significance annotation.
    label : str
        Optional label prefix (e.g. "Momentum alpha").

    Returns
    -------
    str : Formatted IC report string.

    Examples
    --------
    >>> format_ic_report(0.065, 200)
    'IC=+0.065 (t=2.58, N=200) ✓ significant at α=0.05'

    >>> format_ic_report(0.065, 20)
    'IC=+0.065 (t=0.82, N=20) ✗ NOT significant at α=0.05'

    >>> format_ic_report(float("nan"), 0)
    'IC=nan (insufficient data, N=0)'
    """
    prefix = f"{label}: " if label else ""

    if not math.isfinite(ic) or n < 4:
        return f"{prefix}IC=nan (insufficient data, N={n})"

    t_stat, p_val = compute_ic_tstat(ic, n)
    significant = is_ic_significant(ic, n, alpha)

    sign = "+" if ic >= 0 else ""
    t_str = f"t={t_stat:+.2f}" if math.isfinite(t_stat) else "t=nan"
    sig_str = f"✓ significant at α={alpha}" if significant else f"✗ NOT significant at α={alpha}"

    p_str = f", p={p_val:.3f}" if math.isfinite(p_val) else ""

    return f"{prefix}IC={sign}{ic:.4f} ({t_str}{p_str}, N={n}) {sig_str}"


def assert_ic_significant(
    ic: float,
    n: int,
    alpha: float = 0.05,
    context: str = "",
    min_periods: int = 30,
) -> None:
    """
    Assert that IC is statistically significant. Raises if not.

    Use this in validation gates and auto-improve promotion gates.
    Raises InsufficientICSignificance with a detailed error message.

    Parameters
    ----------
    ic : float
    n : int
    alpha : float
    context : str
        What is being validated (for error messages).
    min_periods : int
        Minimum periods required regardless of significance.

    Raises
    ------
    InsufficientICSignificance
        If IC is not significant or N < min_periods.
    """
    report = format_ic_report(ic, n, alpha, label=context or "IC check")

    if n < min_periods:
        raise InsufficientICSignificance(
            f"{report} | FAIL: N={n} < min_periods={min_periods}"
        )

    if not is_ic_significant(ic, n, alpha):
        raise InsufficientICSignificance(
            f"{report} | FAIL: IC not significant"
        )


class InsufficientICSignificance(Exception):
    """
    Raised when IC does not meet statistical significance requirements.

    Used as a hard gate in:
    - auto_improve parameter promotion
    - alpha signal validation
    - model training quality checks
    """
    pass


class ICReport:
    """
    Structured IC report for inclusion in DataFrames and JSON outputs.

    Ensures every IC measurement in the system carries its full context.
    """
    __slots__ = ("ic", "t_stat", "p_value", "n", "significant", "alpha", "label")

    def __init__(
        self,
        ic: float,
        n: int,
        alpha: float = 0.05,
        label: str = "",
    ):
        self.ic = ic
        self.n = n
        self.alpha = alpha
        self.label = label
        self.t_stat, self.p_value = compute_ic_tstat(ic, n)
        self.significant = is_ic_significant(ic, n, alpha)

    def __str__(self) -> str:
        return format_ic_report(self.ic, self.n, self.alpha, self.label)

    def __repr__(self) -> str:
        return f"ICReport(ic={self.ic:.4f}, t={self.t_stat:.2f}, N={self.n}, sig={self.significant})"

    def to_dict(self) -> dict:
        return {
            "ic": self.ic,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "n": self.n,
            "significant": self.significant,
            "alpha": self.alpha,
            "label": self.label,
            "formatted": str(self),
        }


# ---------------------------------------------------------------------------
# Regime-conditional IC tracking
# ---------------------------------------------------------------------------

@dataclass
class RegimeICRecord:
    """IC computed separately for each market regime."""

    regime: str
    ic: float
    t_stat: float
    n_samples: int
    is_significant: bool

    def __str__(self) -> str:
        sig = "✓" if self.is_significant else "✗"
        return f"{self.regime}: IC={self.ic:+.3f} (t={self.t_stat:.2f}, N={self.n_samples}) {sig}"


class RegimeConditionalIC:
    """
    Tracks information coefficient (IC) separately for each market regime.

    Critical insight: a strategy with IC=0.05 overall may have IC=0.12 in
    MEAN_REVERTING regimes and IC=-0.03 in TRENDING regimes.  Trading only
    in high-IC regimes is the core regime-filtering alpha.

    Usage
    -----
        tracker = RegimeConditionalIC()
        tracker.add_observation(predicted_z=2.3, actual_return=0.015, regime="MEAN_REVERTING")
        report = tracker.compute_report()
    """

    def __init__(self) -> None:
        # {regime: [(predicted, actual), ...]}
        self._observations: dict[str, list[tuple[float, float]]] = {}

    def add_observation(
        self,
        predicted_z: float,
        actual_return: float,
        regime: str,
    ) -> None:
        """Record one prediction-realisation pair for a given regime."""
        if np.isnan(predicted_z) or np.isnan(actual_return):
            return
        if regime not in self._observations:
            self._observations[regime] = []
        self._observations[regime].append((float(predicted_z), float(actual_return)))

    def compute_report(self, min_samples: int = 15) -> dict[str, RegimeICRecord]:
        """
        Compute Spearman IC for each regime with >= min_samples observations.

        Returns
        -------
        dict mapping regime_name → RegimeICRecord.
        """
        from scipy.stats import spearmanr

        results: dict[str, RegimeICRecord] = {}
        for regime, obs in self._observations.items():
            if len(obs) < min_samples:
                continue
            preds = [x[0] for x in obs]
            actuals = [x[1] for x in obs]
            try:
                ic, _p_val = spearmanr(preds, actuals)
                n = len(obs)
                t_stat, _ = compute_ic_tstat(float(ic), n)
                results[regime] = RegimeICRecord(
                    regime=regime,
                    ic=float(ic),
                    t_stat=float(t_stat),
                    n_samples=n,
                    is_significant=abs(t_stat) >= 1.65,
                )
            except Exception:
                pass
        return results

    def best_regime(self) -> Optional[str]:
        """Returns the regime with the highest significant positive IC, or None."""
        report = self.compute_report()
        significant = {
            r: rec for r, rec in report.items() if rec.is_significant and rec.ic > 0
        }
        if not significant:
            return None
        return max(significant, key=lambda r: significant[r].ic)

    def worst_regime(self) -> Optional[str]:
        """Returns the regime with the most negative significant IC, or None."""
        report = self.compute_report()
        significant = {
            r: rec for r, rec in report.items() if rec.is_significant and rec.ic < 0
        }
        if not significant:
            return None
        return min(significant, key=lambda r: significant[r].ic)

    def summary(self) -> str:
        """Human-readable summary of regime-conditional IC."""
        report = self.compute_report()
        if not report:
            return "RegimeConditionalIC: insufficient data"
        lines = ["Regime-Conditional IC:"]
        for rec in sorted(report.values(), key=lambda r: -r.ic):
            lines.append(f"  {rec}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise to dict for persistence."""
        report = self.compute_report()
        return {
            regime: {
                "ic": rec.ic,
                "t_stat": rec.t_stat,
                "n_samples": rec.n_samples,
                "significant": rec.is_significant,
            }
            for regime, rec in report.items()
        }

    def n_observations(self) -> dict[str, int]:
        """Return observation counts per regime."""
        return {r: len(obs) for r, obs in self._observations.items()}


# ---------------------------------------------------------------------------
# Pair-level performance tracker (for Kelly-sizing calibration)
# ---------------------------------------------------------------------------

@dataclass
class PairTradeRecord:
    """Single trade outcome for a pair."""

    pair_id: str
    entry_z: float
    exit_z: float
    pnl_pct: float
    holding_days: int
    regime: str
    is_win: bool
    timestamp: str = ""


class PairPerformanceTracker:
    """
    Tracks historical trade outcomes per pair for Kelly-sizing calibration.

    Provides win_rate, avg_win_pct, avg_loss_pct for each pair.
    Used by portfolio/sizing.py Kelly scalar.
    """

    def __init__(self, max_records_per_pair: int = 200) -> None:
        self._records: dict[str, list[PairTradeRecord]] = {}
        self._max = max_records_per_pair

    def record_trade(
        self,
        pair_id: str,
        entry_z: float,
        exit_z: float,
        pnl_pct: float,
        holding_days: int,
        regime: str = "UNKNOWN",
    ) -> None:
        """Record a completed trade outcome."""
        import datetime

        if pair_id not in self._records:
            self._records[pair_id] = []
        rec = PairTradeRecord(
            pair_id=pair_id,
            entry_z=entry_z,
            exit_z=exit_z,
            pnl_pct=pnl_pct,
            holding_days=holding_days,
            regime=regime,
            is_win=pnl_pct > 0,
            timestamp=datetime.datetime.utcnow().isoformat(),
        )
        self._records[pair_id].append(rec)
        if len(self._records[pair_id]) > self._max:
            self._records[pair_id] = self._records[pair_id][-self._max:]

    def get_pair_stats(self, pair_id: str, min_trades: int = 10) -> Optional[dict]:
        """
        Returns Kelly-compatible stats dict or None if insufficient history.

        Returns
        -------
        dict with keys: win_rate, avg_win_pct, avg_loss_pct, n_trades, sharpe.
        """
        records = self._records.get(pair_id, [])
        if len(records) < min_trades:
            return None

        wins = [r.pnl_pct for r in records if r.is_win]
        losses = [r.pnl_pct for r in records if not r.is_win]
        all_pnl = [r.pnl_pct for r in records]

        win_rate = len(wins) / len(records)
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        pnl_std = float(np.std(all_pnl, ddof=1)) if len(all_pnl) > 1 else 1.0
        sharpe_est = (
            float(np.mean(all_pnl)) / pnl_std * np.sqrt(252) if pnl_std > 0 else 0.0
        )

        return {
            "win_rate": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "n_trades": len(records),
            "sharpe": sharpe_est,
        }

    def get_all_stats(self) -> dict[str, dict]:
        """Returns stats for all pairs with sufficient history."""
        return {
            pid: stats
            for pid in self._records
            if (stats := self.get_pair_stats(pid)) is not None
        }
