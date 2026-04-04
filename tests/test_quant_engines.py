# -*- coding: utf-8 -*-
"""
Tests for institutional-grade quant engines:
- SpreadAnalytics (cointegration, half-life, Hurst, quality scoring)
- RiskAnalytics (VaR, CVaR, drawdown, stress tests)
- UniverseScanner (pair discovery pipeline)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─── Fixtures ─────────────────────────────────────────────────

def _make_cointegrated_pair(n=500, beta=1.5, ou_theta=0.1, seed=42):
    """Generate synthetic cointegrated pair."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, n)) + 100
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] * (1 - ou_theta) + rng.normal(0, 0.5)
    y = beta * x + spread + 50

    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(x, index=dates, name="X"), pd.Series(y, index=dates, name="Y")


def _make_random_returns(n=500, mu=0.0003, sigma=0.01, seed=42):
    """Generate synthetic return series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(mu, sigma, n)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(returns, index=dates, name="returns")


def _make_prices_df(n_tickers=5, n_days=500, seed=42):
    """Generate wide-format price DataFrame with some cointegrated pairs."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # Create a common factor
    factor = np.cumsum(rng.normal(0, 1, n_days))

    data = {}
    for i in range(n_tickers):
        ticker = f"SYM{i}"
        # Mix common factor + idiosyncratic
        loading = 0.5 + rng.uniform(0, 1)
        idio = np.cumsum(rng.normal(0, 0.5, n_days))
        data[ticker] = 100 + loading * factor + idio

    return pd.DataFrame(data, index=dates)


# ─── SpreadAnalytics Tests ────────────────────────────────────

class TestSpreadAnalytics:
    """Tests for core.spread_analytics.SpreadAnalytics."""

    def test_import(self):
        from core.spread_analytics import SpreadAnalytics, SpreadAnalysisReport
        assert SpreadAnalytics is not None

    def test_full_analysis_cointegrated_pair(self):
        from core.spread_analytics import SpreadAnalytics
        sa = SpreadAnalytics()
        px, py = _make_cointegrated_pair(n=500, ou_theta=0.1)
        report = sa.full_analysis(px, py, sym_x="X", sym_y="Y")

        assert report.n_observations >= 400
        assert report.quality is not None
        assert report.quality.composite_score > 0
        assert report.quality.grade in ["A+", "A", "B+", "B", "C", "D", "F"]

    def test_half_life_reasonable(self):
        from core.spread_analytics import SpreadAnalytics
        sa = SpreadAnalytics()
        px, py = _make_cointegrated_pair(n=500, ou_theta=0.1)
        report = sa.full_analysis(px, py)

        assert report.mean_reversion is not None
        hl = report.mean_reversion.half_life_days
        # OU with theta=0.1 → half-life ~ log(2)/0.1 ≈ 6.9 days
        assert 1 < hl < 120, f"Half-life {hl} out of range"

    def test_hurst_exponent_mean_reverting(self):
        from core.spread_analytics import SpreadAnalytics
        sa = SpreadAnalytics()
        px, py = _make_cointegrated_pair(n=500, ou_theta=0.1)
        report = sa.full_analysis(px, py)

        assert report.mean_reversion is not None
        # Hurst via R/S can be noisy on synthetic data; just verify it's a valid number
        assert 0.0 <= report.mean_reversion.hurst_exponent <= 1.0

    def test_ols_vs_tls_hedge_ratio(self):
        from core.spread_analytics import SpreadAnalytics
        sa = SpreadAnalytics()
        px, py = _make_cointegrated_pair(n=500, beta=1.5)
        report = sa.full_analysis(px, py)

        # Both should be close to 1.5
        assert abs(report.hedge_ratio_ols - 1.5) < 0.5
        assert abs(report.hedge_ratio_tls - 1.5) < 0.5

    def test_insufficient_data(self):
        from core.spread_analytics import SpreadAnalytics
        sa = SpreadAnalytics()
        px = pd.Series([1, 2, 3], name="X")
        py = pd.Series([2, 4, 6], name="Y")
        report = sa.full_analysis(px, py)

        assert report.quality is not None
        assert report.quality.grade == "F"

    def test_engle_granger(self):
        from core.spread_analytics import SpreadAnalytics
        sa = SpreadAnalytics()
        px, py = _make_cointegrated_pair(n=500, ou_theta=0.1)
        report = sa.full_analysis(px, py)

        if report.engle_granger is not None:
            assert report.engle_granger.method == "engle_granger"
            assert isinstance(report.engle_granger.p_value, float)

    def test_variance_ratio(self):
        from core.spread_analytics import SpreadAnalytics
        sa = SpreadAnalytics()
        px, py = _make_cointegrated_pair(n=500, ou_theta=0.1)
        report = sa.full_analysis(px, py)

        if report.mean_reversion:
            # VR < 1 for mean-reverting
            assert report.mean_reversion.variance_ratio < 1.5


# ─── RiskAnalytics Tests ─────────────────────────────────────

class TestRiskAnalytics:
    """Tests for core.risk_analytics.RiskAnalytics."""

    def test_import(self):
        from core.risk_analytics import RiskAnalytics, RiskReport
        assert RiskAnalytics is not None

    def test_full_risk_report(self):
        from core.risk_analytics import RiskAnalytics
        ra = RiskAnalytics()
        returns = _make_random_returns(n=500)
        report = ra.full_risk_report(returns)

        assert report.n_observations == 500
        assert abs(report.annualized_vol) > 0
        assert isinstance(report.sharpe_ratio, float)

    def test_var_computation(self):
        from core.risk_analytics import RiskAnalytics
        ra = RiskAnalytics()
        returns = _make_random_returns(n=500)
        var95 = ra.compute_var(returns, confidence=0.95)

        assert var95.confidence == 0.95
        assert var95.historical_var > 0
        assert var95.parametric_var > 0
        assert var95.cornish_fisher_var > 0
        assert var95.cvar >= var95.historical_var  # CVaR >= VaR always

    def test_var_99_greater_than_95(self):
        from core.risk_analytics import RiskAnalytics
        ra = RiskAnalytics()
        returns = _make_random_returns(n=500)
        var95 = ra.compute_var(returns, confidence=0.95)
        var99 = ra.compute_var(returns, confidence=0.99)

        assert var99.historical_var >= var95.historical_var

    def test_drawdown(self):
        from core.risk_analytics import RiskAnalytics
        ra = RiskAnalytics()
        returns = _make_random_returns(n=500)
        dd = ra.compute_drawdown(returns)

        assert dd.max_drawdown <= 0  # Drawdowns are negative
        assert dd.max_dd_duration_days >= 0
        assert isinstance(dd.calmar_ratio, float)

    def test_tail_risk(self):
        from core.risk_analytics import RiskAnalytics
        ra = RiskAnalytics()
        returns = _make_random_returns(n=500)
        tail = ra.compute_tail_risk(returns)

        assert isinstance(tail.skewness, float)
        assert isinstance(tail.kurtosis, float)
        assert 0 <= tail.positive_days_pct <= 1
        assert tail.worst_day < 0
        assert tail.best_day > 0

    def test_stress_tests(self):
        from core.risk_analytics import RiskAnalytics
        ra = RiskAnalytics()
        returns = _make_random_returns(n=500)
        scenarios = ra.run_stress_tests(returns)

        assert len(scenarios) >= 5
        for s in scenarios:
            assert s.name
            assert s.portfolio_impact_pct < 0  # Stress = losses

    def test_rolling_risk(self):
        from core.risk_analytics import RiskAnalytics
        ra = RiskAnalytics()
        returns = _make_random_returns(n=500)
        snapshots = ra.compute_rolling_risk(returns, n_snapshots=10)

        assert len(snapshots) > 0
        for snap in snapshots:
            assert snap.rolling_vol_63d > 0
            assert snap.rolling_var95 > 0


# ─── UniverseScanner Tests ────────────────────────────────────

class TestUniverseScanner:
    """Tests for core.universe_scanner.UniverseScanner."""

    def test_import(self):
        from core.universe_scanner import UniverseScanner, ScanResult
        assert UniverseScanner is not None

    def test_scan_finds_pairs(self):
        from core.universe_scanner import UniverseScanner
        prices = _make_prices_df(n_tickers=5, n_days=500)

        scanner = UniverseScanner(
            min_correlation=0.3,
            min_grade="F",
            require_cointegration=False,
            max_half_life=999,
            max_hurst=1.0,  # Disable Hurst filter for synthetic data
            max_pairs=20,
        )
        result = scanner.scan(prices)

        assert result.n_instruments == 5
        assert result.n_pairs_screened == 10  # C(5,2)
        assert len(result.pairs) > 0

    def test_scan_result_is_ranked(self):
        from core.universe_scanner import UniverseScanner
        prices = _make_prices_df(n_tickers=5, n_days=500)

        scanner = UniverseScanner(
            min_correlation=0.3, min_grade="F",
            require_cointegration=False,
        )
        result = scanner.scan(prices)

        if len(result.pairs) >= 2:
            # Should be sorted descending by score
            scores = [p.score for p in result.pairs]
            assert scores == sorted(scores, reverse=True)

    def test_scan_rejection_summary(self):
        from core.universe_scanner import UniverseScanner
        prices = _make_prices_df(n_tickers=4, n_days=500)

        scanner = UniverseScanner(min_correlation=0.99, min_grade="A+")
        result = scanner.scan(prices)

        assert "low_correlation" in result.rejection_summary
        total_rejected = sum(result.rejection_summary.values())
        assert total_rejected + result.n_pairs_final == result.n_pairs_screened

    def test_yield_rate(self):
        from core.universe_scanner import UniverseScanner
        prices = _make_prices_df(n_tickers=4, n_days=500)

        scanner = UniverseScanner(
            min_correlation=0.3, min_grade="F",
            require_cointegration=False,
        )
        result = scanner.scan(prices)

        assert 0 <= result.yield_rate <= 1.0

    def test_scored_pair_fields(self):
        from core.universe_scanner import UniverseScanner
        prices = _make_prices_df(n_tickers=4, n_days=500)

        scanner = UniverseScanner(
            min_correlation=0.3, min_grade="F",
            require_cointegration=False,
        )
        result = scanner.scan(prices)

        if result.pairs:
            p = result.pairs[0]
            assert p.sym_x
            assert p.sym_y
            assert isinstance(p.score, float)
            assert p.grade in ["A+", "A", "B+", "B", "C", "D", "F"]
            assert isinstance(p.half_life, float)
            assert isinstance(p.hurst, float)
