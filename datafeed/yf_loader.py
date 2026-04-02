"""
datafeed/yf_loader.py — Price loader (FMP-first, Yahoo fallback)

Canonical usage: FMP via DataService.
Yahoo Finance fallback retained for callers that need it explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# duckdb_engine must load before pandas (Python 3.13 + DuckDB segfault guard)
try:
    import duckdb_engine  # noqa: F401
except ImportError:
    pass


def load_prices(
    tickers: Sequence[str] = ("SPY", "QQQ"),
    start: str = "2020-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """
    Return a wide close-price DataFrame (index=date, columns=ticker).

    Routes through FMP (canonical provider) first.
    Falls back to yfinance if FMP returns empty and yfinance is installed.
    """
    tickers = [t.upper() for t in tickers]

    # Primary: FMP via DataService
    try:
        from common.data_service import get_data_service
        svc = get_data_service()
        df = svc.get_prices(tickers, start=start, end=end)
        if not df.empty:
            return df.ffill()
    except Exception:
        pass

    # Fallback: canonical loader (handles yfinance + caching + MultiIndex fix)
    try:
        from common.data_loader import load_price_data  # noqa: PLC0415
        frames = {}
        for t in tickers:
            pdf = load_price_data(t, start_date=start, end_date=end)
            if not pdf.empty:
                col = next((c for c in ("adj_close", "close", "Adj Close", "Close") if c in pdf.columns), None)
                if col:
                    frames[t] = pdf[col]
        if frames:
            return pd.DataFrame(frames).ffill()
    except Exception:
        pass

    return pd.DataFrame()
