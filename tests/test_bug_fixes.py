# -*- coding: utf-8 -*-
"""Tests verifying critical bug fixes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestDrawdownGate:
    """Verify DrawdownGate handles edge cases."""

    def test_negative_equity_trips_gate(self):
        from common.risk_helpers import DrawdownGate

        gate = DrawdownGate(threshold=0.1, cooloff=3)
        tripped, dd = gate.update(-50.0)
        assert tripped is True
        assert dd == 1.0

    def test_zero_equity_trips_gate(self):
        from common.risk_helpers import DrawdownGate

        gate = DrawdownGate(threshold=0.1, cooloff=3)
        tripped, dd = gate.update(0.0)
        assert tripped is True
        assert dd == 1.0

    def test_normal_drawdown_works(self):
        from common.risk_helpers import DrawdownGate

        gate = DrawdownGate(threshold=0.2, cooloff=2)
        # Start at 1.0, rise to 1.5, drop to 1.1
        gate.update(1.0)
        gate.update(1.5)
        tripped, dd = gate.update(1.1)
        # dd = 1 - 1.1/1.5 ≈ 0.267 > threshold 0.2
        assert tripped is True
        assert abs(dd - (1.0 - 1.1 / 1.5)) < 1e-6

    def test_nan_equity_not_tripped(self):
        from common.risk_helpers import DrawdownGate

        gate = DrawdownGate(threshold=0.1, cooloff=3)
        tripped, dd = gate.update(float("nan"))
        assert tripped is False
        assert dd == 0.0


class TestApplyCaps:
    """Verify apply_caps handles zero-weight groups."""

    def test_zero_weight_group(self):
        from common.risk_helpers import apply_caps

        weights = pd.Series({"A": 0.0, "B": 0.0, "C": 1.0})
        caps = {"g1": 0.5}
        groups = {"A": "g1", "B": "g1", "C": "g2"}
        result = apply_caps(weights, caps, groups)
        assert result.sum() > 0
        assert not result.isna().any()


class TestSignalGeneratorGuards:
    """Verify signal generators handle constant series."""

    def test_zscore_constant_series(self):
        from common.signal_generator import zscore_signals, ZScoreConfig

        price = pd.Series([100.0] * 50, index=pd.bdate_range("2023-01-01", periods=50))
        cfg = ZScoreConfig(window=20, entry_threshold=2.0, exit_threshold=0.5)
        result = zscore_signals(price, cfg)
        assert not result["zscore"].isna().any()
        assert (result["entry"] == 0).all()

    def test_bollinger_constant_series(self):
        from common.signal_generator import bollinger_signals, BollingerConfig

        price = pd.Series([50.0] * 50, index=pd.bdate_range("2023-01-01", periods=50))
        cfg = BollingerConfig(window=20, num_std=2.0)
        result = bollinger_signals(price, cfg)
        assert (result["entry"] == 0).all()

    def test_rsi_constant_series(self):
        from common.signal_generator import rsi_signals, RSIConfig

        price = pd.Series([75.0] * 50, index=pd.bdate_range("2023-01-01", periods=50))
        cfg = RSIConfig(window=14, lower=30, upper=70)
        result = rsi_signals(price, cfg)
        assert (result["entry"] == 0).all()


class TestRiskBreaches:
    """Verify check_risk_breaches returns correct breach list."""

    def test_no_breaches_clean_state(self):
        from core.risk_engine import check_risk_breaches, RiskState, RiskLimits

        state = RiskState()
        limits = RiskLimits()
        breaches = check_risk_breaches(state, limits)
        assert isinstance(breaches, list)

    def test_check_risk_breaches_exists(self):
        from core.risk_engine import check_risk_breaches

        assert callable(check_risk_breaches)


class TestConfigDateValidation:
    """Verify config date validation logs warnings."""

    def test_garbage_date_falls_back(self):
        from common.config_manager import DashboardConfig

        cfg = DashboardConfig(backtest_start="banana")
        assert cfg.backtest_start == "2020-01-01"

    def test_future_date_accepted_within_range(self):
        from common.config_manager import DashboardConfig

        cfg = DashboardConfig(backtest_start="2025-06-01")
        assert cfg.backtest_start == "2025-06-01"
