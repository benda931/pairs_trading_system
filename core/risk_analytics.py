# -*- coding: utf-8 -*-
"""
core/risk_analytics.py — Institutional Risk Analytics Engine
==============================================================

Professional-grade risk measurement and stress testing:

1. **Value at Risk (VaR)**
   - Historical simulation
   - Parametric (Gaussian)
   - Cornish-Fisher (skew/kurtosis adjusted)
   - Conditional VaR (Expected Shortfall / CVaR)

2. **Drawdown Analysis**
   - Maximum drawdown with duration
   - Underwater analysis
   - Calmar ratio, Sterling ratio
   - Recovery time estimation

3. **Tail Risk Metrics**
   - Skewness, Kurtosis
   - Omega ratio
   - Gain-to-Pain ratio
   - Tail ratio (95th/5th percentile)

4. **Stress Testing**
   - Historical scenario replay
   - Factor shock propagation
   - Correlation stress (regime breaks)
   - VIX-conditional risk adjustment

5. **Rolling Risk Monitors**
   - Rolling VaR / CVaR
   - Rolling volatility (EWMA + realized)
   - Rolling Sharpe / Sortino
   - Regime-conditional risk metrics

Usage:
    from core.risk_analytics import RiskAnalytics

    ra = RiskAnalytics()
    report = ra.full_risk_report(returns_series)
    print(report.var_95, report.cvar_95, report.max_drawdown)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class VaRResult:
    """Value at Risk computation result."""
    confidence: float                    # e.g. 0.95, 0.99
    historical_var: float                # Historical simulation
    parametric_var: float                # Gaussian
    cornish_fisher_var: float            # Skew/kurtosis adjusted
    cvar: float                          # Expected Shortfall (CVaR)
    method_used: str = "cornish_fisher"  # Recommended method


@dataclass
class DrawdownResult:
    """Drawdown analysis result."""
    max_drawdown: float                  # Maximum peak-to-trough
    max_dd_duration_days: int            # Duration of worst drawdown
    max_dd_start: Optional[str] = None   # Start date of worst DD
    max_dd_end: Optional[str] = None     # End date of worst DD
    max_dd_recovery: Optional[str] = None  # Recovery date (or None if not recovered)
    current_drawdown: float = 0.0        # Current DD from peak
    avg_drawdown: float = 0.0            # Average DD
    calmar_ratio: float = 0.0            # CAGR / MaxDD
    sterling_ratio: float = 0.0          # CAGR / Avg(top5 DD)
    underwater_series: Optional[pd.Series] = None


@dataclass
class TailRiskMetrics:
    """Tail risk statistics."""
    skewness: float
    kurtosis: float                      # Excess kurtosis
    jarque_bera_stat: float
    jarque_bera_p: float
    tail_ratio: float                    # |95th pct| / |5th pct|
    omega_ratio: float                   # Prob-weighted gain / loss
    gain_to_pain: float                  # Sum(gains) / Sum(|losses|)
    positive_days_pct: float
    worst_day: float
    best_day: float
    worst_week: float
    worst_month: float


@dataclass
class StressScenario:
    """Result of a single stress scenario."""
    name: str
    description: str
    portfolio_impact_pct: float          # Estimated portfolio loss
    var_multiplier: float                # How much VaR increases
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollingRiskSnapshot:
    """Rolling risk metrics at a point in time."""
    as_of: str
    rolling_vol_21d: float
    rolling_vol_63d: float
    ewma_vol: float
    rolling_sharpe_63d: float
    rolling_sortino_63d: float
    rolling_var95: float
    rolling_cvar95: float
    rolling_max_dd_63d: float


@dataclass
class RiskReport:
    """Complete risk analysis report."""
    n_observations: int
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    sortino_ratio: float

    # VaR
    var_95: Optional[VaRResult] = None
    var_99: Optional[VaRResult] = None

    # Drawdown
    drawdown: Optional[DrawdownResult] = None

    # Tail risk
    tail_risk: Optional[TailRiskMetrics] = None

    # Stress tests
    stress_scenarios: List[StressScenario] = field(default_factory=list)

    # Rolling
    rolling_snapshots: List[RollingRiskSnapshot] = field(default_factory=list)


# =====================================================================
# Core engine
# =====================================================================

class RiskAnalytics:
    """
    Institutional-grade risk analytics engine.

    Computes VaR, CVaR, drawdown, tail risk, stress tests,
    and rolling risk monitors for any return series.
    """

    def __init__(
        self,
        trading_days_per_year: int = 252,
        risk_free_rate: float = 0.04,
        ewma_lambda: float = 0.94,
    ):
        self.tdy = trading_days_per_year
        self.rf = risk_free_rate
        self.ewma_lambda = ewma_lambda

    # ── Public API ────────────────────────────────────────────

    def full_risk_report(
        self,
        returns: pd.Series,
        name: str = "Portfolio",
    ) -> RiskReport:
        """Compute complete risk report from a return series."""
        r = returns.dropna()
        n = len(r)

        ann_ret = float(r.mean() * self.tdy)
        ann_vol = float(r.std() * np.sqrt(self.tdy))
        excess = ann_ret - self.rf
        sharpe = excess / ann_vol if ann_vol > 1e-10 else 0.0
        sortino = self._sortino_ratio(r)

        report = RiskReport(
            n_observations=n,
            annualized_return=round(ann_ret, 6),
            annualized_vol=round(ann_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
        )

        # VaR
        report.var_95 = self.compute_var(r, confidence=0.95)
        report.var_99 = self.compute_var(r, confidence=0.99)

        # Drawdown
        report.drawdown = self.compute_drawdown(r)

        # Tail risk
        report.tail_risk = self.compute_tail_risk(r)

        # Stress tests
        report.stress_scenarios = self.run_stress_tests(r)

        # Rolling risk (last 20 snapshots)
        report.rolling_snapshots = self.compute_rolling_risk(r, n_snapshots=20)

        return report

    # ── Value at Risk ─────────────────────────────────────────

    def compute_var(self, returns: pd.Series, confidence: float = 0.95) -> VaRResult:
        """Compute VaR using 3 methods + CVaR."""
        r = returns.dropna().values
        alpha = 1 - confidence

        # 1. Historical VaR
        hist_var = float(-np.percentile(r, alpha * 100))

        # 2. Parametric (Gaussian) VaR
        mu, sigma = float(np.mean(r)), float(np.std(r, ddof=1))
        z = float(sp_stats.norm.ppf(alpha))
        param_var = float(-(mu + z * sigma))

        # 3. Cornish-Fisher VaR (adjusted for skew/kurtosis)
        s = float(sp_stats.skew(r))
        k = float(sp_stats.kurtosis(r))  # Excess kurtosis
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        cf_var = float(-(mu + z_cf * sigma))

        # 4. CVaR (Expected Shortfall)
        threshold = -hist_var
        tail_returns = r[r <= threshold]
        cvar = float(-np.mean(tail_returns)) if len(tail_returns) > 0 else hist_var * 1.5

        return VaRResult(
            confidence=confidence,
            historical_var=round(hist_var, 6),
            parametric_var=round(param_var, 6),
            cornish_fisher_var=round(cf_var, 6),
            cvar=round(cvar, 6),
        )

    # ── Drawdown Analysis ─────────────────────────────────────

    def compute_drawdown(self, returns: pd.Series) -> DrawdownResult:
        """Compute drawdown analysis from return series."""
        r = returns.dropna()
        equity = (1 + r).cumprod()
        running_max = equity.cummax()
        underwater = (equity - running_max) / running_max

        # Max drawdown
        max_dd = float(underwater.min())
        max_dd_idx = underwater.idxmin()

        # DD start: last peak before max DD point
        peak_before = running_max.loc[:max_dd_idx]
        dd_start = peak_before.idxmax() if len(peak_before) > 0 else None

        # DD end / recovery
        after_trough = equity.loc[max_dd_idx:]
        peak_val = float(running_max.loc[max_dd_idx])
        recovered = after_trough[after_trough >= peak_val]
        recovery_date = str(recovered.index[0]) if len(recovered) > 0 else None

        # Duration
        if dd_start is not None:
            dd_duration = len(r.loc[dd_start:max_dd_idx])
        else:
            dd_duration = 0

        # Current drawdown
        current_dd = float(underwater.iloc[-1]) if len(underwater) > 0 else 0.0

        # Average drawdown
        avg_dd = float(underwater[underwater < 0].mean()) if (underwater < 0).any() else 0.0

        # CAGR
        n_years = len(r) / self.tdy
        total_ret = float(equity.iloc[-1]) if len(equity) > 0 else 1.0
        cagr = total_ret ** (1 / max(n_years, 0.01)) - 1

        # Calmar ratio
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

        # Sterling ratio (CAGR / avg of 5 worst drawdowns)
        dd_periods = self._find_drawdown_periods(underwater)
        worst_5 = sorted([d["depth"] for d in dd_periods])[:5]
        avg_worst_5 = float(np.mean(worst_5)) if worst_5 else max_dd
        sterling = cagr / abs(avg_worst_5) if abs(avg_worst_5) > 1e-10 else 0.0

        return DrawdownResult(
            max_drawdown=round(max_dd, 6),
            max_dd_duration_days=dd_duration,
            max_dd_start=str(dd_start) if dd_start else None,
            max_dd_end=str(max_dd_idx),
            max_dd_recovery=recovery_date,
            current_drawdown=round(current_dd, 6),
            avg_drawdown=round(avg_dd, 6),
            calmar_ratio=round(calmar, 4),
            sterling_ratio=round(sterling, 4),
            underwater_series=underwater,
        )

    @staticmethod
    def _find_drawdown_periods(underwater: pd.Series) -> List[Dict[str, Any]]:
        """Find distinct drawdown periods."""
        periods = []
        in_dd = False
        dd_start = None
        dd_min = 0.0

        for i, val in enumerate(underwater.values):
            if val < -1e-6:
                if not in_dd:
                    in_dd = True
                    dd_start = i
                    dd_min = val
                else:
                    dd_min = min(dd_min, val)
            else:
                if in_dd:
                    periods.append({"start": dd_start, "end": i, "depth": dd_min})
                    in_dd = False
                    dd_min = 0.0

        if in_dd:
            periods.append({"start": dd_start, "end": len(underwater), "depth": dd_min})

        return periods

    # ── Tail Risk ─────────────────────────────────────────────

    def compute_tail_risk(self, returns: pd.Series) -> TailRiskMetrics:
        """Compute tail risk statistics."""
        r = returns.dropna().values

        # Moments
        skew = float(sp_stats.skew(r))
        kurt = float(sp_stats.kurtosis(r))

        # Jarque-Bera
        jb_stat, jb_p = sp_stats.jarque_bera(r)

        # Tail ratio
        pct_95 = float(np.percentile(r, 95))
        pct_5 = float(np.percentile(r, 5))
        tail_ratio = abs(pct_95 / pct_5) if abs(pct_5) > 1e-12 else 1.0

        # Omega ratio (threshold = 0)
        gains = r[r > 0]
        losses = r[r <= 0]
        omega = float(np.sum(gains) / abs(np.sum(losses))) if len(losses) > 0 and abs(np.sum(losses)) > 1e-12 else 999.0

        # Gain to pain
        gtp = float(np.sum(gains) / np.sum(np.abs(losses))) if len(losses) > 0 and np.sum(np.abs(losses)) > 1e-12 else 999.0

        # Win rate
        pos_pct = float(np.sum(r > 0) / len(r)) if len(r) > 0 else 0.0

        # Worst/best
        worst_day = float(np.min(r))
        best_day = float(np.max(r))

        # Worst week/month
        r_series = returns.dropna()
        worst_week = float(r_series.rolling(5).sum().min()) if len(r_series) >= 5 else worst_day * 5
        worst_month = float(r_series.rolling(21).sum().min()) if len(r_series) >= 21 else worst_day * 21

        return TailRiskMetrics(
            skewness=round(skew, 4),
            kurtosis=round(kurt, 4),
            jarque_bera_stat=round(float(jb_stat), 4),
            jarque_bera_p=round(float(jb_p), 6),
            tail_ratio=round(tail_ratio, 4),
            omega_ratio=round(omega, 4),
            gain_to_pain=round(gtp, 4),
            positive_days_pct=round(pos_pct, 4),
            worst_day=round(worst_day, 6),
            best_day=round(best_day, 6),
            worst_week=round(worst_week, 6),
            worst_month=round(worst_month, 6),
        )

    # ── Stress Testing ────────────────────────────────────────

    def run_stress_tests(self, returns: pd.Series) -> List[StressScenario]:
        """Run standard stress scenarios."""
        r = returns.dropna().values
        vol = float(np.std(r) * np.sqrt(self.tdy))
        scenarios = []

        # 1. 2-sigma move
        sigma2 = vol * 2 / np.sqrt(self.tdy)
        scenarios.append(StressScenario(
            name="2σ Daily Move",
            description="Two standard deviation adverse move in one day",
            portfolio_impact_pct=round(-sigma2 * 100, 2),
            var_multiplier=2.0,
        ))

        # 2. 3-sigma move
        sigma3 = vol * 3 / np.sqrt(self.tdy)
        scenarios.append(StressScenario(
            name="3σ Daily Move",
            description="Three standard deviation adverse move (fat tail event)",
            portfolio_impact_pct=round(-sigma3 * 100, 2),
            var_multiplier=3.0,
        ))

        # 3. Correlation break (pairs diverge)
        scenarios.append(StressScenario(
            name="Correlation Break",
            description="Pair correlation drops to zero; spread doubles",
            portfolio_impact_pct=round(-vol * 100 * 0.5, 2),
            var_multiplier=2.5,
            details={"assumed_spread_multiplier": 2.0},
        ))

        # 4. Liquidity crisis (5 worst days replay)
        worst_5 = float(np.sum(np.sort(r)[:5]))
        scenarios.append(StressScenario(
            name="Liquidity Crisis (5-day)",
            description="Replay of 5 worst historical days consecutively",
            portfolio_impact_pct=round(worst_5 * 100, 2),
            var_multiplier=5.0,
        ))

        # 5. VIX spike (vol doubles)
        scenarios.append(StressScenario(
            name="VIX Spike (+100%)",
            description="Volatility doubles; positions sized at current vol hit stop-losses",
            portfolio_impact_pct=round(-vol * 100 * 0.3, 2),
            var_multiplier=2.0,
            details={"vol_multiplier": 2.0},
        ))

        # 6. Regime shift (trending market for 20 days)
        drift_20d = vol / np.sqrt(self.tdy) * np.sqrt(20)
        scenarios.append(StressScenario(
            name="20-Day Trending Regime",
            description="Mean-reversion fails; spread trends for 20 consecutive days",
            portfolio_impact_pct=round(-drift_20d * 100, 2),
            var_multiplier=3.0,
        ))

        return scenarios

    # ── Rolling Risk ──────────────────────────────────────────

    def compute_rolling_risk(
        self, returns: pd.Series, n_snapshots: int = 20,
    ) -> List[RollingRiskSnapshot]:
        """Compute rolling risk metrics at evenly spaced points."""
        r = returns.dropna()
        n = len(r)
        if n < 63:
            return []

        # EWMA volatility
        ewma_var = pd.Series(0.0, index=r.index, dtype=float)
        ewma_var.iloc[0] = r.iloc[0] ** 2
        for i in range(1, n):
            ewma_var.iloc[i] = self.ewma_lambda * ewma_var.iloc[i - 1] + (1 - self.ewma_lambda) * r.iloc[i] ** 2
        ewma_vol = np.sqrt(ewma_var) * np.sqrt(self.tdy)

        # Rolling metrics
        roll_21 = r.rolling(21).std() * np.sqrt(self.tdy)
        roll_63 = r.rolling(63).std() * np.sqrt(self.tdy)
        roll_sharpe = (r.rolling(63).mean() * self.tdy - self.rf) / (r.rolling(63).std() * np.sqrt(self.tdy))

        # Downside deviation for Sortino
        downside = r.copy()
        downside[downside > 0] = 0
        roll_downside = downside.rolling(63).std() * np.sqrt(self.tdy)
        roll_sortino = (r.rolling(63).mean() * self.tdy - self.rf) / roll_downside.replace(0, np.nan)

        # Rolling VaR/CVaR
        def rolling_var_cvar(window_returns):
            v = window_returns.dropna().values
            if len(v) < 10:
                return float("nan"), float("nan")
            var95 = float(-np.percentile(v, 5))
            tail = v[v <= -var95]
            cvar95 = float(-np.mean(tail)) if len(tail) > 0 else var95 * 1.5
            return var95, cvar95

        # Sample points
        indices = np.linspace(63, n - 1, min(n_snapshots, n - 63), dtype=int)
        snapshots = []

        for idx in indices:
            try:
                window = r.iloc[max(0, idx - 63):idx + 1]
                var95, cvar95 = rolling_var_cvar(window)

                # Rolling max DD (63d)
                eq = (1 + window).cumprod()
                underwater = (eq - eq.cummax()) / eq.cummax()
                roll_dd = float(underwater.min())

                snapshots.append(RollingRiskSnapshot(
                    as_of=str(r.index[idx]),
                    rolling_vol_21d=round(float(roll_21.iloc[idx]), 6) if not np.isnan(roll_21.iloc[idx]) else 0.0,
                    rolling_vol_63d=round(float(roll_63.iloc[idx]), 6) if not np.isnan(roll_63.iloc[idx]) else 0.0,
                    ewma_vol=round(float(ewma_vol.iloc[idx]), 6),
                    rolling_sharpe_63d=round(float(roll_sharpe.iloc[idx]), 4) if not np.isnan(roll_sharpe.iloc[idx]) else 0.0,
                    rolling_sortino_63d=round(float(roll_sortino.iloc[idx]), 4) if not np.isnan(roll_sortino.iloc[idx]) else 0.0,
                    rolling_var95=round(var95, 6),
                    rolling_cvar95=round(cvar95, 6),
                    rolling_max_dd_63d=round(roll_dd, 6),
                ))
            except Exception:
                continue

        return snapshots

    # ── Helper ────────────────────────────────────────────────

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Sortino ratio: excess return / downside deviation."""
        r = returns.dropna().values
        ann_ret = float(np.mean(r) * self.tdy)
        downside = r[r < 0]
        if len(downside) == 0:
            return 999.0
        dd = float(np.std(downside, ddof=1) * np.sqrt(self.tdy))
        return round((ann_ret - self.rf) / dd, 4) if dd > 1e-10 else 0.0
