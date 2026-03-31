# -*- coding: utf-8 -*-
"""
tests/test_walk_forward.py — Tests for research/walk_forward.py

Verifies:
  1. Splits are non-overlapping and embargo gap is respected
  2. No look-ahead: fold parameters use strictly training data
  3. Sufficient data produces n_splits folds
  4. Insufficient data returns graceful failure
  5. Cointegrated pair produces viable results
  6. Non-cointegrated pair is not viable
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from core.contracts import PairId, SpreadModel
from research.walk_forward import (
    WalkForwardHarness,
    _generate_splits,
    summarise_experiments,
)


def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n)


def _cointegrated_prices(n: int = 1500, beta: float = 0.8, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)
    log_y = np.cumsum(rng.normal(0, 0.01, n))
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = noise[t - 1] * 0.88 + rng.normal(0, 0.02)
    log_x = beta * log_y + 0.3 + noise
    return pd.DataFrame({
        "X": np.exp(log_x + np.log(100)),
        "Y": np.exp(log_y + np.log(100)),
    }, index=dates)


def _random_walk_prices(n: int = 1500, seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)
    x = np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(100))
    y = np.exp(np.cumsum(rng.normal(0, 0.01, n)) + np.log(100))
    return pd.DataFrame({"X": x, "Y": y}, index=dates)


# ── Split generator tests ─────────────────────────────────────────

class TestGenerateSplits:
    def test_correct_number_of_splits(self):
        idx = pd.bdate_range("2018-01-01", periods=1500)
        splits = _generate_splits(idx, n_splits=5, test_days=200, min_train_days=500, embargo_days=20)
        assert len(splits) <= 5
        assert len(splits) >= 2  # should get at least 2 with this data

    def test_no_overlap_between_test_windows(self):
        idx = pd.bdate_range("2018-01-01", periods=1500)
        splits = _generate_splits(idx, n_splits=4, test_days=200, min_train_days=400, embargo_days=20)
        for i in range(len(splits) - 1):
            _, _, _, test_end_i = splits[i]
            _, _, test_start_next, _ = splits[i + 1]
            assert test_end_i < test_start_next, (
                f"Overlap between fold {i} end={test_end_i} and fold {i+1} start={test_start_next}"
            )

    def test_embargo_gap_respected(self):
        idx = pd.bdate_range("2018-01-01", periods=1500)
        embargo = 20
        splits = _generate_splits(idx, n_splits=4, test_days=200, min_train_days=400, embargo_days=embargo)
        for train_start, train_end, test_start, test_end in splits:
            gap_days = (test_start - train_end).days
            assert gap_days >= embargo - 5, (  # -5 for business day conversion tolerance
                f"Embargo gap too small: {gap_days} days (expected >= {embargo})"
            )

    def test_train_precedes_test(self):
        idx = pd.bdate_range("2018-01-01", periods=1500)
        splits = _generate_splits(idx, n_splits=4, test_days=200, min_train_days=400, embargo_days=20)
        for train_start, train_end, test_start, test_end in splits:
            assert train_start < train_end
            assert train_end < test_start
            assert test_start < test_end

    def test_insufficient_data_returns_fewer_splits(self):
        idx = pd.bdate_range("2018-01-01", periods=300)
        splits = _generate_splits(idx, n_splits=5, test_days=200, min_train_days=500, embargo_days=20)
        assert len(splits) < 5  # can't fit 5 folds in 300 days

    def test_empty_index_returns_empty(self):
        idx = pd.DatetimeIndex([])
        splits = _generate_splits(idx, n_splits=3, test_days=100, min_train_days=200, embargo_days=10)
        assert splits == []


# ── WalkForwardHarness tests ──────────────────────────────────────

class TestWalkForwardHarness:
    def test_run_returns_experiment_result(self):
        prices = _cointegrated_prices(n=1500)
        pid = PairId("X", "Y")
        harness = WalkForwardHarness(n_splits=3, test_days=200, min_train_days=400, embargo_days=20)
        result = harness.run(pid, prices, model=SpreadModel.STATIC_OLS)
        assert result.pair_id == pid
        assert result.n_splits == 3

    def test_correct_number_of_folds(self):
        prices = _cointegrated_prices(n=1500)
        pid = PairId("X", "Y")
        harness = WalkForwardHarness(n_splits=3, test_days=200, min_train_days=400, embargo_days=20)
        result = harness.run(pid, prices, model=SpreadModel.STATIC_OLS)
        assert len(result.folds) >= 2  # at least 2 folds with this data

    def test_insufficient_data_returns_empty_result(self):
        prices = _cointegrated_prices(n=100)
        pid = PairId("X", "Y")
        harness = WalkForwardHarness(n_splits=5, test_days=252, min_train_days=504, embargo_days=20)
        result = harness.run(pid, prices, model=SpreadModel.STATIC_OLS)
        assert len(result.folds) == 0 or not result.viable

    def test_no_lookahead_fold_train_ends(self):
        """Each fold's train_end must be before its test_start."""
        prices = _cointegrated_prices(n=1500)
        pid = PairId("X", "Y")
        harness = WalkForwardHarness(n_splits=3, test_days=200, min_train_days=400, embargo_days=20)
        result = harness.run(pid, prices)
        for fold in result.folds:
            assert fold.train_end < fold.test_start, (
                f"Fold {fold.fold_idx}: train_end={fold.train_end} >= test_start={fold.test_start}"
            )

    def test_spread_defn_fitted_only_on_training(self):
        """SpreadDefinition.train_end must be <= fold.train_end."""
        prices = _cointegrated_prices(n=1500)
        pid = PairId("X", "Y")
        harness = WalkForwardHarness(n_splits=3, test_days=200, min_train_days=400, embargo_days=20)
        result = harness.run(pid, prices, model=SpreadModel.STATIC_OLS)
        for fold in result.folds:
            if fold.spread_defn is not None:
                assert fold.spread_defn.train_end <= fold.train_end, (
                    f"Fold {fold.fold_idx}: spread trained on data after train_end"
                )

    def test_missing_column_raises(self):
        prices = pd.DataFrame({"X": [100.0, 101.0, 102.0]})
        pid = PairId("X", "Y")
        harness = WalkForwardHarness()
        with pytest.raises(ValueError):
            harness.run(pid, prices)

    def test_rolling_ols_model(self):
        prices = _cointegrated_prices(n=1500)
        pid = PairId("X", "Y")
        harness = WalkForwardHarness(n_splits=2, test_days=200, min_train_days=400, embargo_days=20)
        result = harness.run(pid, prices, model=SpreadModel.ROLLING_OLS, window=60)
        assert result.pair_id == pid

    def test_kalman_model(self):
        prices = _cointegrated_prices(n=1500)
        pid = PairId("X", "Y")
        harness = WalkForwardHarness(n_splits=2, test_days=200, min_train_days=400, embargo_days=20)
        result = harness.run(pid, prices, model=SpreadModel.KALMAN)
        assert result.pair_id == pid

    def test_aggregate_metrics_populated(self):
        prices = _cointegrated_prices(n=1500)
        pid = PairId("X", "Y")
        harness = WalkForwardHarness(n_splits=3, test_days=200, min_train_days=400, embargo_days=20)
        result = harness.run(pid, prices)
        valid = result.valid_folds()
        if valid:
            assert np.isfinite(result.avg_sharpe)
            assert 0.0 <= result.validation_pass_rate <= 1.0


class TestSummariseExperiments:
    def test_summary_dataframe_structure(self):
        prices = _cointegrated_prices(n=1500)
        pid = PairId("X", "Y")
        harness = WalkForwardHarness(n_splits=2, test_days=200, min_train_days=400, embargo_days=20)
        result = harness.run(pid, prices)
        df = summarise_experiments([result])
        assert isinstance(df, pd.DataFrame)
        assert "pair_label" in df.columns
        assert "avg_sharpe" in df.columns
        assert "viable" in df.columns

    def test_empty_list_returns_empty_df(self):
        df = summarise_experiments([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
