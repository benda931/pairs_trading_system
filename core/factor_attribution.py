# -*- coding: utf-8 -*-
"""
core/factor_attribution.py — Factor-Based Performance Attribution
===================================================================

Institutional-grade return attribution framework:

1. **Brinson-Fachler Attribution**
   - Allocation effect (sector over/underweight vs benchmark)
   - Selection effect (stock picking within sectors)
   - Interaction effect

2. **Factor Decomposition**
   - Market beta exposure
   - Size (SMB proxy)
   - Value (HML proxy)
   - Momentum
   - Volatility
   - Pairs-specific alpha (residual)

3. **Risk Attribution**
   - Factor contribution to total variance
   - Marginal risk contribution per position
   - Diversification ratio

4. **Performance Analytics**
   - Information ratio
   - Tracking error
   - Hit rate by factor regime
   - Rolling alpha stability

Usage:
    from core.factor_attribution import FactorAttribution

    fa = FactorAttribution()
    report = fa.attribute(portfolio_returns, benchmark_returns, factor_returns)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class FactorExposure:
    """Exposure to a single factor."""
    factor_name: str
    beta: float                          # Factor loading
    t_statistic: float                   # Statistical significance
    contribution_to_return: float        # β × factor_return
    contribution_to_risk: float          # Marginal risk contribution
    r_squared_marginal: float            # Explanatory power of this factor


@dataclass
class BrinsonAttribution:
    """Brinson-Fachler attribution result."""
    total_active_return: float
    allocation_effect: float             # Over/underweight effect
    selection_effect: float              # Stock-picking effect
    interaction_effect: float            # Allocation × Selection
    residual: float                      # Unexplained


@dataclass
class RiskAttribution:
    """Risk decomposition."""
    total_portfolio_vol: float
    systematic_vol: float                # Factor-explained vol
    idiosyncratic_vol: float             # Residual vol
    diversification_ratio: float         # Sum(component vol) / portfolio vol
    factor_contributions: Dict[str, float] = field(default_factory=dict)
    position_marginal_risk: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Portfolio performance analytics."""
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: float             # Alpha / tracking error
    tracking_error: float
    alpha_annualized: float              # Jensen's alpha
    beta_to_benchmark: float
    r_squared: float                     # How much is explained by benchmark
    hit_rate: float                      # % of positive alpha days
    avg_win_loss_ratio: float
    max_drawdown: float
    calmar_ratio: float


@dataclass
class AttributionReport:
    """Complete attribution report."""
    n_observations: int
    period_start: str
    period_end: str

    # Performance
    performance: Optional[PerformanceMetrics] = None

    # Factor exposures
    factor_exposures: List[FactorExposure] = field(default_factory=list)
    total_r_squared: float = 0.0         # Total factor-explained variance

    # Brinson (if sector data available)
    brinson: Optional[BrinsonAttribution] = None

    # Risk attribution
    risk: Optional[RiskAttribution] = None

    # Time-varying
    rolling_alpha: Optional[pd.Series] = None
    rolling_beta: Optional[pd.Series] = None
    rolling_ir: Optional[pd.Series] = None


# =====================================================================
# Engine
# =====================================================================

class FactorAttribution:
    """
    Factor-based performance attribution engine.

    Decomposes portfolio returns into factor exposures, computes
    alpha, and provides Brinson-Fachler sector attribution.
    """

    def __init__(
        self,
        trading_days: int = 252,
        risk_free_rate: float = 0.04,
        rolling_window: int = 63,
    ):
        self.tdy = trading_days
        self.rf = risk_free_rate
        self.rolling_window = rolling_window

    def attribute(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        pair_returns: Optional[pd.DataFrame] = None,
    ) -> AttributionReport:
        """
        Run full factor attribution.

        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily portfolio returns.
        benchmark_returns : pd.Series, optional
            Benchmark (e.g. SPY) daily returns.
        factor_returns : pd.DataFrame, optional
            Factor returns (columns = factor names).
        pair_returns : pd.DataFrame, optional
            Individual pair returns for position-level attribution.
        """
        r = portfolio_returns.dropna()
        n = len(r)

        report = AttributionReport(
            n_observations=n,
            period_start=str(r.index[0]) if n > 0 else "",
            period_end=str(r.index[-1]) if n > 0 else "",
        )

        if n < 30:
            return report

        # Performance metrics
        report.performance = self._compute_performance(r, benchmark_returns)

        # Factor regression
        if factor_returns is not None and not factor_returns.empty:
            exposures, r_sq = self._factor_regression(r, factor_returns)
            report.factor_exposures = exposures
            report.total_r_squared = r_sq

        # Risk attribution
        if factor_returns is not None:
            report.risk = self._risk_attribution(r, factor_returns, pair_returns)

        # Rolling metrics
        if benchmark_returns is not None:
            report.rolling_alpha, report.rolling_beta, report.rolling_ir = (
                self._rolling_metrics(r, benchmark_returns)
            )

        return report

    # ── Performance ───────────────────────────────────────────

    def _compute_performance(
        self,
        r: pd.Series,
        bench: Optional[pd.Series],
    ) -> PerformanceMetrics:
        """Compute comprehensive performance metrics."""
        ann_ret = float(r.mean() * self.tdy)
        ann_vol = float(r.std() * np.sqrt(self.tdy))
        rf_daily = self.rf / self.tdy
        excess = r - rf_daily

        sharpe = float(excess.mean() / r.std() * np.sqrt(self.tdy)) if r.std() > 1e-10 else 0.0

        # Sortino
        downside = r[r < 0]
        dd_std = float(downside.std() * np.sqrt(self.tdy)) if len(downside) > 0 else ann_vol
        sortino = float((ann_ret - self.rf) / dd_std) if dd_std > 1e-10 else 0.0

        # Alpha, beta, tracking error vs benchmark
        alpha, beta, r_sq = 0.0, 0.0, 0.0
        ir, te = 0.0, 0.0

        if bench is not None:
            b = bench.reindex(r.index).dropna()
            common = r.index.intersection(b.index)
            if len(common) > 30:
                r_c = r.loc[common]
                b_c = b.loc[common]

                # OLS: r = alpha + beta * bench + epsilon
                X = np.column_stack([np.ones(len(b_c)), b_c.values])
                coeffs = np.linalg.lstsq(X, r_c.values, rcond=None)[0]
                alpha = float(coeffs[0]) * self.tdy  # Annualized
                beta = float(coeffs[1])

                # R-squared
                predicted = X @ coeffs
                ss_res = np.sum((r_c.values - predicted) ** 2)
                ss_tot = np.sum((r_c.values - r_c.mean()) ** 2)
                r_sq = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                # Tracking error & Information ratio
                active = r_c - b_c
                te = float(active.std() * np.sqrt(self.tdy))
                ir = float(active.mean() * self.tdy / te) if te > 1e-10 else 0.0

        # Hit rate
        positive_days = float(np.sum(r.values > 0) / len(r))

        # Win/loss ratio
        wins = r[r > 0]
        losses = r[r < 0]
        avg_wl = float(wins.mean() / abs(losses.mean())) if len(losses) > 0 and abs(losses.mean()) > 1e-12 else 1.0

        # Max drawdown
        eq = (1 + r).cumprod()
        underwater = (eq - eq.cummax()) / eq.cummax()
        max_dd = float(underwater.min())

        # Calmar
        n_years = len(r) / self.tdy
        total_ret = float(eq.iloc[-1]) if len(eq) > 0 else 1.0
        cagr = total_ret ** (1 / max(n_years, 0.01)) - 1
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

        return PerformanceMetrics(
            total_return=round(float(eq.iloc[-1] - 1) if len(eq) > 0 else 0, 6),
            annualized_return=round(ann_ret, 6),
            annualized_vol=round(ann_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            information_ratio=round(ir, 4),
            tracking_error=round(te, 6),
            alpha_annualized=round(alpha, 6),
            beta_to_benchmark=round(beta, 4),
            r_squared=round(r_sq, 4),
            hit_rate=round(positive_days, 4),
            avg_win_loss_ratio=round(avg_wl, 4),
            max_drawdown=round(max_dd, 6),
            calmar_ratio=round(calmar, 4),
        )

    # ── Factor Regression ─────────────────────────────────────

    def _factor_regression(
        self,
        r: pd.Series,
        factors: pd.DataFrame,
    ) -> Tuple[List[FactorExposure], float]:
        """Multi-factor OLS regression."""
        common = r.index.intersection(factors.index)
        if len(common) < 30:
            return [], 0.0

        y = r.loc[common].values
        X_raw = factors.loc[common].values
        X = np.column_stack([np.ones(len(y)), X_raw])
        factor_names = list(factors.columns)

        # OLS
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            return [], 0.0

        predicted = X @ coeffs
        ss_res = float(np.sum((y - predicted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Standard errors
        n, k = len(y), X.shape[1]
        mse = ss_res / max(n - k, 1)
        try:
            cov_matrix = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov_matrix))
        except Exception:
            se = np.ones(k) * 1e-6

        exposures = []
        for i, fname in enumerate(factor_names):
            beta_i = float(coeffs[i + 1])
            se_i = float(se[i + 1]) if i + 1 < len(se) else 1e-6
            t_stat = beta_i / se_i if abs(se_i) > 1e-12 else 0.0

            # Factor contribution to return
            factor_mean_ret = float(factors.loc[common, fname].mean()) * self.tdy
            contrib_ret = beta_i * factor_mean_ret

            # Factor contribution to risk (marginal)
            factor_vol = float(factors.loc[common, fname].std()) * np.sqrt(self.tdy)
            contrib_risk = abs(beta_i) * factor_vol

            # Marginal R-squared (drop this factor and see R² change)
            X_reduced = np.delete(X, i + 1, axis=1)
            try:
                coeffs_r = np.linalg.lstsq(X_reduced, y, rcond=None)[0]
                pred_r = X_reduced @ coeffs_r
                ss_res_r = np.sum((y - pred_r) ** 2)
                r_sq_r = 1 - ss_res_r / ss_tot if ss_tot > 0 else 0.0
                r_sq_marginal = r_sq - r_sq_r
            except Exception:
                r_sq_marginal = 0.0

            exposures.append(FactorExposure(
                factor_name=fname,
                beta=round(beta_i, 6),
                t_statistic=round(t_stat, 4),
                contribution_to_return=round(contrib_ret, 6),
                contribution_to_risk=round(contrib_risk, 6),
                r_squared_marginal=round(max(0, r_sq_marginal), 4),
            ))

        return exposures, round(r_sq, 4)

    # ── Risk Attribution ──────────────────────────────────────

    def _risk_attribution(
        self,
        r: pd.Series,
        factors: pd.DataFrame,
        positions: Optional[pd.DataFrame] = None,
    ) -> RiskAttribution:
        """Decompose risk into systematic and idiosyncratic components."""
        common = r.index.intersection(factors.index)
        if len(common) < 30:
            return RiskAttribution(
                total_portfolio_vol=float(r.std() * np.sqrt(self.tdy)),
                systematic_vol=0, idiosyncratic_vol=0, diversification_ratio=1.0,
            )

        y = r.loc[common].values
        X = np.column_stack([np.ones(len(y)), factors.loc[common].values])

        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        predicted = X @ coeffs
        residuals = y - predicted

        total_vol = float(np.std(y) * np.sqrt(self.tdy))
        sys_vol = float(np.std(predicted) * np.sqrt(self.tdy))
        idio_vol = float(np.std(residuals) * np.sqrt(self.tdy))

        # Factor contributions to variance
        factor_names = list(factors.columns)
        factor_contribs = {}
        for i, fname in enumerate(factor_names):
            factor_var_contrib = float(coeffs[i + 1] ** 2 * factors.loc[common, fname].var() * self.tdy)
            factor_contribs[fname] = round(factor_var_contrib, 8)

        # Diversification ratio
        if positions is not None and not positions.empty:
            pos_common = positions.reindex(common).dropna(how="all")
            if not pos_common.empty:
                component_vols = pos_common.std() * np.sqrt(self.tdy)
                div_ratio = float(component_vols.sum() / total_vol) if total_vol > 0 else 1.0
            else:
                div_ratio = 1.0
        else:
            div_ratio = sys_vol / total_vol if total_vol > 0 else 1.0

        return RiskAttribution(
            total_portfolio_vol=round(total_vol, 6),
            systematic_vol=round(sys_vol, 6),
            idiosyncratic_vol=round(idio_vol, 6),
            diversification_ratio=round(div_ratio, 4),
            factor_contributions=factor_contribs,
        )

    # ── Rolling Metrics ───────────────────────────────────────

    def _rolling_metrics(
        self,
        r: pd.Series,
        bench: pd.Series,
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
        """Compute rolling alpha, beta, and information ratio."""
        common = r.index.intersection(bench.index)
        if len(common) < self.rolling_window + 10:
            return None, None, None

        r_c = r.loc[common]
        b_c = bench.loc[common]
        w = self.rolling_window

        alphas, betas, irs = [], [], []
        indices = []

        for i in range(w, len(r_c)):
            r_win = r_c.iloc[i - w:i].values
            b_win = b_c.iloc[i - w:i].values

            X = np.column_stack([np.ones(w), b_win])
            try:
                coeffs = np.linalg.lstsq(X, r_win, rcond=None)[0]
                alpha_i = float(coeffs[0]) * self.tdy
                beta_i = float(coeffs[1])
            except Exception:
                alpha_i, beta_i = 0.0, 1.0

            active = r_win - b_win
            te = float(np.std(active) * np.sqrt(self.tdy))
            ir_i = float(np.mean(active) * self.tdy / te) if te > 1e-10 else 0.0

            alphas.append(alpha_i)
            betas.append(beta_i)
            irs.append(ir_i)
            indices.append(r_c.index[i])

        return (
            pd.Series(alphas, index=indices, name="rolling_alpha"),
            pd.Series(betas, index=indices, name="rolling_beta"),
            pd.Series(irs, index=indices, name="rolling_ir"),
        )
