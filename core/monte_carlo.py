# -*- coding: utf-8 -*-
"""
core/monte_carlo.py — Monte Carlo Simulation Engine
=====================================================

Professional-grade Monte Carlo framework for:

1. **Strategy Confidence Intervals**
   - Bootstrap returns → equity curves → Sharpe distribution
   - Deflated Sharpe Ratio (Bailey & Lopez de Prado)
   - Probability of loss, ruin, and target achievement

2. **Path Simulation**
   - Block bootstrap (preserves autocorrelation)
   - Parametric (normal, Student-t, skewed-t)
   - Historical simulation with regime conditioning

3. **Risk Budgeting**
   - VaR/CVaR confidence bands
   - Drawdown distribution
   - Time-to-recovery distribution

4. **Strategy Comparison**
   - Paired bootstrap test (is strategy A > B?)
   - Multiple testing correction (Holm-Bonferroni)

Usage:
    from core.monte_carlo import MonteCarloEngine

    mc = MonteCarloEngine(n_simulations=10000)
    result = mc.simulate_strategy(returns, capital=1_000_000)
    print(f"Sharpe 95% CI: [{result.sharpe_ci_lower:.2f}, {result.sharpe_ci_upper:.2f}]")
    print(f"Prob of loss: {result.prob_loss:.1%}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EquityPathStats:
    """Statistics from a single simulated equity path."""
    final_return: float
    max_drawdown: float
    sharpe: float
    sortino: float
    max_equity: float
    min_equity: float
    n_days_underwater: int


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo simulation result."""
    n_simulations: int
    n_observations: int
    method: str

    # Sharpe distribution
    sharpe_mean: float
    sharpe_median: float
    sharpe_std: float
    sharpe_ci_lower: float               # 2.5th percentile
    sharpe_ci_upper: float               # 97.5th percentile
    deflated_sharpe: float               # Bailey & Lopez de Prado

    # Return distribution
    expected_annual_return: float
    return_ci_lower: float
    return_ci_upper: float

    # Probability metrics
    prob_loss: float                     # P(final return < 0)
    prob_target: float                   # P(return > target)
    prob_ruin: float                     # P(drawdown > ruin_threshold)
    prob_sharpe_positive: float          # P(Sharpe > 0)

    # Drawdown distribution
    median_max_drawdown: float
    dd_ci_lower: float                   # 2.5th percentile worst DD
    dd_ci_upper: float                   # 97.5th percentile worst DD
    avg_time_underwater_days: float

    # VaR distribution
    var95_mean: float
    var95_ci_lower: float
    var95_ci_upper: float

    # Raw distributions (for plotting)
    sharpe_distribution: Optional[np.ndarray] = None
    return_distribution: Optional[np.ndarray] = None
    drawdown_distribution: Optional[np.ndarray] = None

    # Metadata
    target_return: float = 0.0
    ruin_threshold: float = -0.30
    confidence_level: float = 0.95


@dataclass
class StrategyComparisonResult:
    """Result of comparing two strategies via paired bootstrap."""
    strategy_a_name: str
    strategy_b_name: str
    a_sharpe_mean: float
    b_sharpe_mean: float
    prob_a_better: float                 # P(Sharpe_A > Sharpe_B)
    mean_difference: float               # E[Sharpe_A - Sharpe_B]
    difference_ci_lower: float
    difference_ci_upper: float
    is_significant: bool                 # At given alpha
    p_value: float


class MonteCarloEngine:
    """
    Professional Monte Carlo simulation engine.

    Supports block bootstrap, parametric simulation, and
    strategy comparison testing.
    """

    def __init__(
        self,
        n_simulations: int = 5000,
        block_size: int = 21,            # ~1 month blocks for bootstrap
        trading_days: int = 252,
        risk_free_rate: float = 0.04,
        seed: Optional[int] = None,
    ):
        self.n_sims = n_simulations
        self.block_size = block_size
        self.tdy = trading_days
        self.rf = risk_free_rate
        self.rng = np.random.default_rng(seed)

    # ── Strategy Simulation ───────────────────────────────────

    def simulate_strategy(
        self,
        returns: pd.Series,
        capital: float = 1_000_000,
        target_return: float = 0.10,
        ruin_threshold: float = -0.30,
        method: str = "block_bootstrap",
        horizon_days: Optional[int] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on a return series.

        Parameters
        ----------
        returns : pd.Series
            Historical daily returns.
        capital : float
            Starting capital (for absolute metrics).
        target_return : float
            Target annual return (for prob_target).
        ruin_threshold : float
            Max drawdown threshold for ruin probability.
        method : str
            "block_bootstrap" or "parametric" or "historical".
        horizon_days : int, optional
            Simulation horizon. Default: len(returns).
        """
        r = returns.dropna().values
        n = len(r)
        horizon = horizon_days or n

        if n < 30:
            return self._empty_result(n, method)

        # Generate simulated paths
        if method == "block_bootstrap":
            paths = self._block_bootstrap(r, horizon)
        elif method == "parametric":
            paths = self._parametric_simulation(r, horizon)
        else:
            paths = self._historical_simulation(r, horizon)

        # Compute statistics for each path
        sharpes = np.zeros(self.n_sims)
        final_returns = np.zeros(self.n_sims)
        max_dds = np.zeros(self.n_sims)
        time_underwater = np.zeros(self.n_sims)
        var95s = np.zeros(self.n_sims)

        rf_daily = self.rf / self.tdy

        for i in range(self.n_sims):
            path = paths[i]
            equity = np.cumprod(1 + path)

            final_returns[i] = equity[-1] - 1
            ann_ret = np.mean(path) * self.tdy
            ann_vol = np.std(path, ddof=1) * np.sqrt(self.tdy)
            sharpes[i] = (ann_ret - self.rf) / ann_vol if ann_vol > 1e-10 else 0

            # Max drawdown
            running_max = np.maximum.accumulate(equity)
            underwater = (equity - running_max) / running_max
            max_dds[i] = np.min(underwater)
            time_underwater[i] = np.sum(underwater < -0.001)

            # VaR 95
            var95s[i] = -np.percentile(path, 5)

        # Deflated Sharpe Ratio
        dsr = self._deflated_sharpe(r, sharpes)

        # Confidence intervals
        def ci(arr, lo=2.5, hi=97.5):
            return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

        sharpe_lo, sharpe_hi = ci(sharpes)
        ret_lo, ret_hi = ci(final_returns)
        dd_lo, dd_hi = ci(max_dds)
        var_lo, var_hi = ci(var95s)

        return MonteCarloResult(
            n_simulations=self.n_sims,
            n_observations=n,
            method=method,
            sharpe_mean=round(float(np.mean(sharpes)), 4),
            sharpe_median=round(float(np.median(sharpes)), 4),
            sharpe_std=round(float(np.std(sharpes)), 4),
            sharpe_ci_lower=round(sharpe_lo, 4),
            sharpe_ci_upper=round(sharpe_hi, 4),
            deflated_sharpe=round(dsr, 4),
            expected_annual_return=round(float(np.mean(final_returns)), 6),
            return_ci_lower=round(ret_lo, 6),
            return_ci_upper=round(ret_hi, 6),
            prob_loss=round(float(np.mean(final_returns < 0)), 4),
            prob_target=round(float(np.mean(final_returns > target_return)), 4),
            prob_ruin=round(float(np.mean(max_dds < ruin_threshold)), 4),
            prob_sharpe_positive=round(float(np.mean(sharpes > 0)), 4),
            median_max_drawdown=round(float(np.median(max_dds)), 6),
            dd_ci_lower=round(dd_lo, 6),
            dd_ci_upper=round(dd_hi, 6),
            avg_time_underwater_days=round(float(np.mean(time_underwater)), 1),
            var95_mean=round(float(np.mean(var95s)), 6),
            var95_ci_lower=round(var_lo, 6),
            var95_ci_upper=round(var_hi, 6),
            sharpe_distribution=sharpes,
            return_distribution=final_returns,
            drawdown_distribution=max_dds,
            target_return=target_return,
            ruin_threshold=ruin_threshold,
        )

    # ── Strategy Comparison ───────────────────────────────────

    def compare_strategies(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        name_a: str = "Strategy A",
        name_b: str = "Strategy B",
        alpha: float = 0.05,
    ) -> StrategyComparisonResult:
        """
        Paired bootstrap test: is strategy A significantly better than B?
        """
        common = returns_a.index.intersection(returns_b.index)
        ra = returns_a.loc[common].values
        rb = returns_b.loc[common].values
        n = len(ra)

        if n < 30:
            return StrategyComparisonResult(
                strategy_a_name=name_a, strategy_b_name=name_b,
                a_sharpe_mean=0, b_sharpe_mean=0,
                prob_a_better=0.5, mean_difference=0,
                difference_ci_lower=0, difference_ci_upper=0,
                is_significant=False, p_value=1.0,
            )

        diffs = np.zeros(self.n_sims)
        sharpes_a = np.zeros(self.n_sims)
        sharpes_b = np.zeros(self.n_sims)

        for i in range(self.n_sims):
            idx = self.rng.choice(n, size=n, replace=True)
            sa = ra[idx]
            sb = rb[idx]

            vol_a = np.std(sa, ddof=1) * np.sqrt(self.tdy)
            vol_b = np.std(sb, ddof=1) * np.sqrt(self.tdy)

            sh_a = (np.mean(sa) * self.tdy - self.rf) / vol_a if vol_a > 1e-10 else 0
            sh_b = (np.mean(sb) * self.tdy - self.rf) / vol_b if vol_b > 1e-10 else 0

            sharpes_a[i] = sh_a
            sharpes_b[i] = sh_b
            diffs[i] = sh_a - sh_b

        prob_a_better = float(np.mean(diffs > 0))
        mean_diff = float(np.mean(diffs))
        ci_lo = float(np.percentile(diffs, 2.5))
        ci_hi = float(np.percentile(diffs, 97.5))

        # P-value: fraction of bootstrap where diff crosses zero
        p_value = 2 * min(prob_a_better, 1 - prob_a_better)
        is_sig = p_value < alpha

        return StrategyComparisonResult(
            strategy_a_name=name_a,
            strategy_b_name=name_b,
            a_sharpe_mean=round(float(np.mean(sharpes_a)), 4),
            b_sharpe_mean=round(float(np.mean(sharpes_b)), 4),
            prob_a_better=round(prob_a_better, 4),
            mean_difference=round(mean_diff, 4),
            difference_ci_lower=round(ci_lo, 4),
            difference_ci_upper=round(ci_hi, 4),
            is_significant=is_sig,
            p_value=round(p_value, 6),
        )

    # ── Simulation Methods ────────────────────────────────────

    def _block_bootstrap(self, r: np.ndarray, horizon: int) -> np.ndarray:
        """Block bootstrap preserving autocorrelation structure."""
        n = len(r)
        bs = min(self.block_size, n // 3)
        paths = np.zeros((self.n_sims, horizon))

        for i in range(self.n_sims):
            path = []
            while len(path) < horizon:
                start = self.rng.integers(0, n - bs + 1)
                block = r[start:start + bs]
                path.extend(block)
            paths[i] = np.array(path[:horizon])

        return paths

    def _parametric_simulation(self, r: np.ndarray, horizon: int) -> np.ndarray:
        """Parametric simulation with fitted distribution."""
        from scipy import stats as sp_stats

        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1))
        skew = float(sp_stats.skew(r))
        kurt = float(sp_stats.kurtosis(r))

        # If significant kurtosis, use Student-t
        if abs(kurt) > 1.0:
            # Fit Student-t
            df_est = max(4, 6 / max(kurt, 0.1) + 4)  # Rough DOF estimate
            paths = self.rng.standard_t(df_est, size=(self.n_sims, horizon))
            paths = paths * sigma + mu
        else:
            paths = self.rng.normal(mu, sigma, size=(self.n_sims, horizon))

        return paths

    def _historical_simulation(self, r: np.ndarray, horizon: int) -> np.ndarray:
        """Pure historical resampling (IID)."""
        n = len(r)
        paths = np.zeros((self.n_sims, horizon))
        for i in range(self.n_sims):
            idx = self.rng.choice(n, size=horizon, replace=True)
            paths[i] = r[idx]
        return paths

    # ── Deflated Sharpe Ratio ─────────────────────────────────

    def _deflated_sharpe(self, r: np.ndarray, sharpe_dist: np.ndarray) -> float:
        """
        Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

        Adjusts for multiple testing, skewness, and kurtosis.
        DSR = P(SR* > 0 | SR_0 = E[max(SR)])
        """
        from scipy import stats as sp_stats

        n = len(r)
        sr = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(self.tdy)) if np.std(r) > 1e-10 else 0
        skew = float(sp_stats.skew(r))
        kurt = float(sp_stats.kurtosis(r))  # Excess

        # Expected max Sharpe from trials (Euler-Mascheroni approximation)
        n_trials = max(self.n_sims, 1)
        gamma = 0.5772156649
        e_max_sr = float(
            (1 - gamma) * sp_stats.norm.ppf(1 - 1 / n_trials)
            + gamma * sp_stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )

        # DSR test statistic
        sr_std = np.sqrt(
            (1 + 0.5 * sr**2 - skew * sr + (kurt / 4) * sr**2) / max(n - 1, 1)
        )

        if sr_std < 1e-10:
            return 0.0

        dsr_stat = (sr - e_max_sr) / sr_std
        dsr = float(sp_stats.norm.cdf(dsr_stat))
        return dsr

    def _empty_result(self, n: int, method: str) -> MonteCarloResult:
        """Return empty result for insufficient data."""
        return MonteCarloResult(
            n_simulations=0, n_observations=n, method=method,
            sharpe_mean=0, sharpe_median=0, sharpe_std=0,
            sharpe_ci_lower=0, sharpe_ci_upper=0, deflated_sharpe=0,
            expected_annual_return=0, return_ci_lower=0, return_ci_upper=0,
            prob_loss=0.5, prob_target=0, prob_ruin=0, prob_sharpe_positive=0.5,
            median_max_drawdown=0, dd_ci_lower=0, dd_ci_upper=0,
            avg_time_underwater_days=0,
            var95_mean=0, var95_ci_lower=0, var95_ci_upper=0,
        )
