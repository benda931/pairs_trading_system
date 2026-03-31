# -*- coding: utf-8 -*-
"""Shared fixtures for the pairs trading test suite."""
from __future__ import annotations

import pytest

# CRITICAL: duckdb_engine must be imported before pandas to avoid segfault
# on Python 3.13 + DuckDB 1.3.x.
import duckdb_engine  # noqa: F401

import numpy as np
import pandas as pd


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate synthetic daily prices for two correlated assets."""
    np.random.seed(42)
    n = 252
    dates = pd.bdate_range("2023-01-01", periods=n)
    base = 100 + np.cumsum(np.random.normal(0.0005, 0.01, n))
    noise = np.random.normal(0, 0.005, n)
    return pd.DataFrame(
        {"A": base, "B": base * 1.1 + np.cumsum(noise)},
        index=dates,
    )


@pytest.fixture
def sample_returns(sample_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily returns derived from sample_prices."""
    return sample_prices.pct_change().dropna()


@pytest.fixture
def in_memory_store():
    """Create a SqlStore backed by in-memory DuckDB."""
    from core.sql_store import SqlStore

    return SqlStore("duckdb:///:memory:")
