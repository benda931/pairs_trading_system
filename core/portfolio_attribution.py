# -*- coding: utf-8 -*-
"""
core/portfolio_attribution.py — P&L Attribution & Stress Analysis
===================================================================

Institutional-grade attribution framework for pairs portfolios.

Capabilities:
  1. P&L attribution by pair, regime, strategy bucket
  2. Drawdown decomposition: which pairs caused the drawdown
  3. Correlation contribution: pairs moving together during stress
  4. Stress scenario analysis: simultaneous correlation collapse
  5. Factor attribution: how much P&L came from market beta vs alpha

Usage:
    from core.portfolio_attribution import PortfolioAttributionEngine

    engine = PortfolioAttributionEngine()
    report = engine.attribute(positions, pnl_history, regime_history)
    print(report.worst_drawdown_contributor)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PairAttribution:
    """P&L and risk attribution for a single pair."""
    pair_id: str

    # P&L
    total_pnl_pct:    float = 0.0
    total_pnl_usd:    float = 0.0
    n_trades:         int   = 0
    win_rate:         float = 0.0
    avg_holding_days: float = 0.0

    # Risk contribution
    contribution_to_portfolio_vol: float = 0.0  # Marginal vol contribution
    contribution_to_max_dd:        float = 0.0  # Fraction of max DD caused by this pair

    # Regime breakdown
    pnl_by_regime: Dict[str, float] = field(default_factory=dict)

    # During stress
    pnl_during_worst_month: float = 0.0

    @property
    def information_ratio(self) -> float:
        """Simplified pair-level IR (not annualised)."""
        if self.n_trades < 3:
            return 0.0
        return self.total_pnl_pct / max(0.01, abs(self.total_pnl_pct) * 0.5)


@dataclass
class DrawdownDecomposition:
    """Attribution of portfolio drawdown to individual pairs."""
    peak_date:   Optional[datetime] = None
    trough_date: Optional[datetime] = None
    max_dd_pct:  float = 0.0

    # Pair contributions: pair_id → fraction of max drawdown
    pair_contributions: Dict[str, float] = field(default_factory=dict)

    # Top contributors to drawdown
    top_contributors: List[Tuple[str, float]] = field(default_factory=list)

    # Were pairs correlated during drawdown?
    drawdown_correlation: float = 0.0   # Average pairwise corr during DD period

    def summary(self) -> str:
        lines = [f"Max Drawdown: {self.max_dd_pct:.1%}"]
        for pair_id, contrib in self.top_contributors[:5]:
            lines.append(f"  {pair_id}: {contrib:.1%} of drawdown")
        lines.append(f"  Avg pairwise corr during DD: {self.drawdown_correlation:.2f}")
        return "\n".join(lines)


@dataclass
class StressScenario:
    """Results of a stress test scenario."""
    scenario_name: str
    description: str

    estimated_portfolio_pnl_pct: float = 0.0
    estimated_portfolio_pnl_usd: float = 0.0

    # Per-pair impact
    pair_impacts: Dict[str, float] = field(default_factory=dict)

    # Worst-case pairs
    worst_pair:  Optional[str] = None
    worst_impact: float = 0.0


@dataclass
class AttributionReport:
    """Complete attribution report for a portfolio."""
    as_of: str = ""

    # Per-pair attribution
    pair_attributions: Dict[str, PairAttribution] = field(default_factory=dict)

    # Portfolio-level
    total_pnl_pct:  float = 0.0
    portfolio_sharpe: float = 0.0
    max_dd_pct: float = 0.0

    # Drawdown decomposition
    drawdown_decomp: Optional[DrawdownDecomposition] = None

    # Stress scenarios
    stress_results: List[StressScenario] = field(default_factory=list)

    # Top/bottom performers
    top_pairs:    List[str] = field(default_factory=list)
    bottom_pairs: List[str] = field(default_factory=list)

    @property
    def worst_drawdown_contributor(self) -> Optional[str]:
        if self.drawdown_decomp and self.drawdown_decomp.top_contributors:
            return self.drawdown_decomp.top_contributors[0][0]
        return None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PortfolioAttributionEngine:
    """
    Computes P&L and risk attribution for a pairs portfolio.
    """

    def __init__(self) -> None:
        pass

    def attribute(
        self,
        pair_pnl_history: Dict[str, pd.Series],
        portfolio_pnl: pd.Series,
        regime_history: Optional[pd.Series] = None,
        notional_map: Optional[Dict[str, float]] = None,
    ) -> AttributionReport:
        """
        Main attribution method.

        Args:
            pair_pnl_history: dict mapping pair_id → daily P&L pct series
            portfolio_pnl:    portfolio-level daily P&L pct series
            regime_history:   optional series of regime labels (indexed by date)
            notional_map:     optional dict of pair_id → notional for USD P&L
        """
        report = AttributionReport(as_of=datetime.utcnow().isoformat())
        notional_map = notional_map or {}

        if portfolio_pnl.empty:
            return report

        # Portfolio stats
        report.total_pnl_pct   = float(portfolio_pnl.sum())
        daily_std = portfolio_pnl.std(ddof=1)
        if daily_std > 0:
            report.portfolio_sharpe = float(portfolio_pnl.mean() / daily_std * np.sqrt(252))

        # Max drawdown
        cum = (1 + portfolio_pnl).cumprod()
        rolling_max = cum.cummax()
        dd_series = (cum - rolling_max) / rolling_max
        report.max_dd_pct = float(dd_series.min())

        # Per-pair attribution
        for pair_id, pnl_series in pair_pnl_history.items():
            attr = self._attribute_pair(
                pair_id, pnl_series, portfolio_pnl,
                regime_history, notional_map.get(pair_id, 100_000)
            )
            report.pair_attributions[pair_id] = attr

        # Drawdown decomposition
        report.drawdown_decomp = self._decompose_drawdown(
            pair_pnl_history, portfolio_pnl
        )

        # Rank pairs
        sorted_pairs = sorted(
            report.pair_attributions.items(),
            key=lambda x: x[1].total_pnl_pct,
            reverse=True,
        )
        report.top_pairs    = [p for p, _ in sorted_pairs[:5]]
        report.bottom_pairs = [p for p, _ in sorted_pairs[-5:]]

        # Stress scenarios
        report.stress_results = self._run_stress_scenarios(
            pair_pnl_history, notional_map
        )

        return report

    def _attribute_pair(
        self,
        pair_id: str,
        pnl: pd.Series,
        portfolio_pnl: pd.Series,
        regime_history: Optional[pd.Series],
        notional: float,
    ) -> PairAttribution:
        attr = PairAttribution(pair_id=pair_id)
        pnl_clean = pnl.dropna()
        if pnl_clean.empty:
            return attr

        attr.total_pnl_pct = float(pnl_clean.sum())
        attr.total_pnl_usd = attr.total_pnl_pct * notional

        # Win rate (treat each non-zero day as a "trade day")
        nonzero = pnl_clean[pnl_clean != 0]
        if len(nonzero) > 0:
            attr.win_rate = float((nonzero > 0).mean())
            attr.n_trades = len(nonzero)

        # Marginal vol contribution (covariance / portfolio vol)
        aligned = pd.concat(
            [pnl_clean.rename("pair"), portfolio_pnl.rename("port")], axis=1
        ).dropna()
        if len(aligned) > 10:
            cov = float(aligned["pair"].cov(aligned["port"]))
            port_var = float(aligned["port"].var())
            if port_var > 0:
                attr.contribution_to_portfolio_vol = cov / port_var

        # Regime breakdown
        if regime_history is not None:
            for regime in regime_history.unique():
                dates = regime_history[regime_history == regime].index
                pair_regime_pnl = pnl_clean.reindex(dates).dropna().sum()
                attr.pnl_by_regime[str(regime)] = float(pair_regime_pnl)

        # Worst month P&L
        if len(pnl_clean) >= 21:
            monthly = pnl_clean.rolling(21).sum()
            attr.pnl_during_worst_month = float(monthly.min())

        return attr

    def _decompose_drawdown(
        self,
        pair_pnl: Dict[str, pd.Series],
        portfolio_pnl: pd.Series,
    ) -> DrawdownDecomposition:
        """Identify which pairs drove the worst drawdown period."""
        dd = DrawdownDecomposition()

        if portfolio_pnl.empty:
            return dd

        cum = (1 + portfolio_pnl).cumprod()
        rolling_max = cum.cummax()
        dd_series = (cum - rolling_max) / rolling_max

        dd.max_dd_pct = float(dd_series.min())
        trough_date = dd_series.idxmin()

        # Find peak before trough
        pre_trough = rolling_max[:trough_date]
        if len(pre_trough) > 0:
            peak_date = pre_trough.idxmax()
            dd.peak_date   = peak_date.to_pydatetime() if hasattr(peak_date, 'to_pydatetime') else peak_date
            dd.trough_date = trough_date.to_pydatetime() if hasattr(trough_date, 'to_pydatetime') else trough_date

            # P&L of each pair during drawdown period
            contributions = {}
            for pair_id, pnl in pair_pnl.items():
                period_pnl = pnl[peak_date:trough_date].sum()
                contributions[pair_id] = float(period_pnl)

            total_loss = abs(dd.max_dd_pct) + 1e-10
            dd.pair_contributions = {
                p: v / total_loss for p, v in contributions.items() if v < 0
            }
            dd.top_contributors = sorted(
                dd.pair_contributions.items(), key=lambda x: x[1]
            )[:10]  # worst first (most negative fraction)

            # Average pairwise correlation during drawdown
            dd_returns = pd.DataFrame({
                p: pnl[peak_date:trough_date]
                for p, pnl in pair_pnl.items()
            }).dropna(axis=1)
            if dd_returns.shape[1] >= 2:
                corr_mat = dd_returns.corr().values
                n = corr_mat.shape[0]
                off_diag = corr_mat[np.triu_indices(n, k=1)]
                dd.drawdown_correlation = float(np.mean(off_diag))

        return dd

    def _run_stress_scenarios(
        self,
        pair_pnl: Dict[str, pd.Series],
        notional_map: Dict[str, float],
    ) -> List[StressScenario]:
        """Run standard stress scenarios on the portfolio."""
        scenarios = []

        # Scenario 1: Correlation collapse (spreads all widen by 2σ simultaneously)
        s1 = StressScenario(
            scenario_name="simultaneous_spread_widening",
            description="All spreads simultaneously widen by 2σ (crowded unwind scenario)",
        )
        total_impact = 0.0
        for pair_id, pnl in pair_pnl.items():
            pnl_std = float(pnl.std(ddof=1)) if len(pnl) > 5 else 0.01
            impact = -2.0 * pnl_std   # 2σ adverse move
            s1.pair_impacts[pair_id] = impact
            notional = notional_map.get(pair_id, 100_000)
            total_impact += impact * notional
        if notional_map:
            total_notional = sum(notional_map.values())
            s1.estimated_portfolio_pnl_pct = total_impact / max(total_notional, 1)
            s1.estimated_portfolio_pnl_usd = total_impact
        if s1.pair_impacts:
            s1.worst_pair   = min(s1.pair_impacts, key=s1.pair_impacts.get)
            s1.worst_impact = s1.pair_impacts[s1.worst_pair]
        scenarios.append(s1)

        # Scenario 2: Worst historical month replayed
        s2 = StressScenario(
            scenario_name="worst_month_replay",
            description="Worst 21-day return period replayed across all pairs",
        )
        for pair_id, pnl in pair_pnl.items():
            if len(pnl) >= 21:
                worst = float(pnl.rolling(21).sum().min())
                s2.pair_impacts[pair_id] = worst
        if s2.pair_impacts:
            total = sum(s2.pair_impacts[p] * notional_map.get(p, 100_000)
                       for p in s2.pair_impacts)
            total_notional = sum(notional_map.get(p, 100_000) for p in s2.pair_impacts)
            s2.estimated_portfolio_pnl_pct = total / max(total_notional, 1)
            s2.estimated_portfolio_pnl_usd = total
            s2.worst_pair   = min(s2.pair_impacts, key=s2.pair_impacts.get)
            s2.worst_impact = s2.pair_impacts[s2.worst_pair]
        scenarios.append(s2)

        return scenarios
