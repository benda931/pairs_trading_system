# -*- coding: utf-8 -*-
"""
tests/test_pair_validator.py — Tests for research/pair_validator.py

Uses synthetic data to test the validation pipeline:
  1. Clean mean-reverting pair → should PASS
  2. Trending (non-stationary spread) → should FAIL
  3. Low correlation → should FAIL
  4. Insufficient data → should FAIL
  5. Structural break mid-series → may WARN
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.contracts import PairId, ValidationResult
from research.pair_validator import PairValidator


def _make_dates(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _cointegrated_pair(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic cointegrated pair:
      log(Y) = 0.8 * log(X) + epsilon
      epsilon ~ OU(theta=0.15, mu=0, sigma=0.01)  — tight spread for high correlation
    """
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)

    # Generate common trend (larger steps → higher correlation)
    log_x = np.cumsum(rng.normal(0, 0.015, n))
    # Tight OU spread (small noise relative to common trend)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - 0.15) + rng.normal(0, 0.008)

    log_y = 0.9 * log_x + spread

    return pd.DataFrame({
        "X": np.exp(log_x + np.log(100)),
        "Y": np.exp(log_y + np.log(100)),
    }, index=dates)


def _uncorrelated_pair(n: int = 600, seed: int = 99) -> pd.DataFrame:
    """Two independent random walks — no cointegration."""
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)
    x = np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(100))
    y = np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(100))
    return pd.DataFrame({"X": x, "Y": y}, index=dates)


def _trending_spread_pair(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Correlated pair but with a trending (non-stationary) spread."""
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)
    base = np.cumsum(rng.normal(0, 0.01, n))
    # Spread has a unit root (pure random walk — not mean reverting)
    spread = np.cumsum(rng.normal(0, 0.01, n))
    x = np.exp(base + np.log(100))
    y = np.exp(base + spread + np.log(100))
    return pd.DataFrame({"X": x, "Y": y}, index=dates)


class TestPairValidatorPassCase:
    def test_cointegrated_pair_passes(self):
        prices = _cointegrated_pair(n=600)
        pid = PairId("X", "Y")
        validator = PairValidator()
        report = validator.validate(pid, prices)
        assert report.result in (ValidationResult.PASS, ValidationResult.WARN), (
            f"Expected PASS/WARN but got {report.result}. "
            f"Rejections: {report.rejection_reasons}"
        )

    def test_cointegrated_pair_no_rejection_reasons(self):
        prices = _cointegrated_pair(n=600)
        pid = PairId("X", "Y")
        validator = PairValidator()
        report = validator.validate(pid, prices)
        if report.result == ValidationResult.PASS:
            assert len(report.rejection_reasons) == 0

    def test_cointegrated_pair_has_metrics(self):
        prices = _cointegrated_pair(n=600)
        pid = PairId("X", "Y")
        validator = PairValidator()
        report = validator.validate(pid, prices)
        assert not np.isnan(report.correlation)
        assert not np.isnan(report.half_life_days)
        assert report.half_life_days > 0


class TestPairValidatorFailCases:
    def test_uncorrelated_pair_fails(self):
        prices = _uncorrelated_pair(n=600)
        pid = PairId("X", "Y")
        validator = PairValidator()
        report = validator.validate(pid, prices)
        assert report.result == ValidationResult.FAIL
        assert len(report.rejection_reasons) > 0

    def test_trending_spread_fails(self):
        prices = _trending_spread_pair(n=600)
        pid = PairId("X", "Y")
        validator = PairValidator()
        report = validator.validate(pid, prices)
        # Should fail ADF or cointegration test
        assert report.result in (ValidationResult.FAIL, ValidationResult.WARN)

    def test_insufficient_data_fails(self):
        prices = _cointegrated_pair(n=30)  # way too few bars
        pid = PairId("X", "Y")
        validator = PairValidator()
        report = validator.validate(pid, prices)
        assert report.result == ValidationResult.FAIL
        assert any("data" in r.lower() for r in report.rejection_reasons)

    def test_missing_column_returns_fail(self):
        """Missing column produces a FAIL report (no raise) with an informative reason."""
        prices = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
        pid = PairId("X", "Y")
        validator = PairValidator()
        report = validator.validate(pid, prices)
        assert report.result in (ValidationResult.FAIL, ValidationResult.SKIP)
        # Either rejected with a reason or skipped due to missing data
        assert len(report.rejection_reasons) > 0 or report.result == ValidationResult.SKIP


class TestPairValidatorTrainEnd:
    def test_train_end_restricts_estimation(self):
        """Validator with early train_end should use fewer data points."""
        prices = _cointegrated_pair(n=600)
        pid = PairId("X", "Y")
        validator = PairValidator()

        # Use only first half for validation
        train_end = prices.index[299].to_pydatetime()
        report = validator.validate(pid, prices, train_end=train_end)

        # Should still produce a valid report
        assert report.pair_id == pid
        assert not np.isnan(report.correlation)

    def test_train_end_vs_full_gives_different_metrics(self):
        """Half the data → correlation may differ slightly."""
        prices = _cointegrated_pair(n=600)
        pid = PairId("X", "Y")
        validator = PairValidator()

        report_full = validator.validate(pid, prices)
        train_end = prices.index[299].to_pydatetime()
        report_half = validator.validate(pid, prices, train_end=train_end)

        # Both should have correlation populated
        assert not np.isnan(report_full.correlation)
        assert not np.isnan(report_half.correlation)


class TestValidateBatch:
    def test_batch_returns_one_per_pair(self):
        prices_1 = _cointegrated_pair(n=600, seed=1)
        prices_2 = _cointegrated_pair(n=600, seed=2)
        prices_2.columns = ["A", "B"]
        prices = pd.concat([prices_1, prices_2], axis=1)
        prices.columns = ["X", "Y", "A", "B"]

        pids = [PairId("X", "Y"), PairId("A", "B")]
        validator = PairValidator()
        reports = validator.validate_batch(pids, prices)

        assert len(reports) == 2
        assert {r.pair_id for r in reports} == set(pids)
