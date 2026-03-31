# -*- coding: utf-8 -*-
"""Tests for common.config_manager."""
from __future__ import annotations


def test_config_loads_defaults():
    from common.config_manager import DashboardConfig

    cfg = DashboardConfig()
    assert cfg.theme in ("light", "dark")
    assert isinstance(cfg.max_trials, int)
    assert cfg.max_trials > 0


def test_config_validates_backtest_dates():
    from common.config_manager import DashboardConfig

    cfg = DashboardConfig()
    assert cfg.backtest_start < cfg.backtest_end


def test_config_roundtrip_dict():
    from common.config_manager import DashboardConfig

    cfg = DashboardConfig()
    d = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()
    assert isinstance(d, dict)
    assert "theme" in d
