# -*- coding: utf-8 -*-
"""Tests for core.sql_store — persistence layer."""
from __future__ import annotations

import pandas as pd


def test_store_creates_tables(in_memory_store):
    tables = in_memory_store.list_tables()
    assert isinstance(tables, list)
    assert "kv_store" in tables
    assert "prices" in tables


def test_save_and_read_price_history(in_memory_store):
    import numpy as np
    import pandas as pd

    store = in_memory_store
    dates = pd.bdate_range("2023-01-01", periods=10)
    df = pd.DataFrame({
        "open": np.random.uniform(99, 101, 10),
        "high": np.random.uniform(101, 103, 10),
        "low": np.random.uniform(97, 99, 10),
        "close": np.random.uniform(99, 101, 10),
        "volume": np.random.randint(1000, 10000, 10),
    }, index=dates)
    store.save_price_history("TEST_A", df)
    result = store.read_table("prices")
    assert len(result) > 0


def test_engine_info(in_memory_store):
    info = in_memory_store.get_engine_info()
    assert "dialect" in info
    assert info["dialect"] == "duckdb"


def test_raw_query(in_memory_store):
    result = in_memory_store.raw_query("SELECT 42 AS answer")
    assert len(result) == 1


def test_save_json_roundtrip(in_memory_store):
    store = in_memory_store
    store.save_json("test_ns", "greeting", {"msg": "hello"})
    # Verify it was saved by reading the kv_store table
    df = store.read_table("kv_store")
    assert len(df) > 0
