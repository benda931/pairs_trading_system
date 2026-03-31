# -*- coding: utf-8 -*-
"""Smoke tests: verify all critical modules can be imported without errors."""
from __future__ import annotations

import pytest


CORE_MODULES = [
    "core.app_context",
    "core.sql_store",
    "core.optimizer",
    "core.risk_engine",
    "core.signals_engine",
    "core.fair_value_engine",
    "core.macro_engine",
    "core.dashboard_models",
    "core.data_quality",
    "core.distributions",
    "core.meta_optimizer",
    "core.params",
    "core.ranges",
]

COMMON_MODULES = [
    "common.config_manager",
    "common.risk_helpers",
    "common.data_loader",
    "common.signal_generator",
    "common.helpers",
    "common.macro_factors",
    "common.matrix_helpers",
    "common.json_safe",
    "common.advanced_metrics",
    "common.macro_adjustments",
    "common.utils",
]


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_import_core(module_name):
    __import__(module_name)


@pytest.mark.parametrize("module_name", COMMON_MODULES)
def test_import_common(module_name):
    __import__(module_name)
