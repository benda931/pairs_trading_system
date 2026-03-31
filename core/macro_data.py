# -*- coding: utf-8 -*-
"""
core/macro_data.py ? ???? ???? ????? ?? Cache/TTL ?-Adapters ??????
====================================================================

????
-----
????? ????? ???? ?? **????? ??? ?????** (???????????) ??????? ?????:

- ???? CSV ???????.
- DuckDB (?????? / ???????).
- SQL (?? ???? ???? ?"? SQLAlchemy).
- yfinance (ETF/Index/FX ???').

????? ????:

- `pandas.DataFrame` ?? ?????? ??? (DatetimeIndex).
- ????? ??? ??? `"value"` (float).
- ????????? (freq = "D"/"W"/"M"/"Q"/"B" ???').
- Cache ?? TTL ??? ?? ????? ???????? ??-Rate Limits.

URI ????? (sources mapping)
----------------------------
?-MacroDataClient.sources ????? ????? `indicator_id` ? URI:

- `local:/path/to/file.csv`
- `duckdb:/path/to/db.duckdb|table_name` ?? `duckdb:/path/to/db.duckdb|SELECT ...`
- `sql:CONNSTR|SELECT * FROM ...`
- `yf:SPY` (????? ????: period=10y, interval=1d, field=Close)
- `yf:SPY|period=5y,interval=1wk,field=Adj Close` (?????? ??????)

Public API (?????? ??????)
---------------------------
- class MacroDataClient:
    def get(
        self,
        indicator_id: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        freq: str = "D",
        *,
        lag_policy: str = "ffill",
    ) -> pd.DataFrame

?????:
    DataFrame ??:
      - DatetimeIndex (??????, ?? ????????? "date").
      - ????? ???: "value" (float).

??????:
    - ????: pandas, numpy.
    - ????????? (Guarded): duckdb, sqlalchemy, yfinance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import logging
import time

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("core.macro_data")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


# ============================================================
# Utilities
# ============================================================

def _ensure_ts_df(df: pd.DataFrame, *, value_col: Optional[str] = None) -> pd.DataFrame:
    """
    ????? DataFrame ????? ???? ??? ????????:
        - DatetimeIndex.
        - ????? ???: 'value' ???? float (??? NaN).

    ??????:
      1. ?? df ??? ? ????? DataFrame ??? ?? index DatetimeIndex ?????? 'value'.
      2. ?? ??????? ???? DatetimeIndex:
         - ???? ????? ????? ??? "date"/"time"/"timestamp" (case-insensitive).
         - ?? ?? ???? ? ????? ?????? ??????? ?????? ?????.
         - ???? ???? ?-Datetime ?????? ???? ?-index.
      3. ???? ?? ????? ????:
         - ?? value_col ???? ???? ???? ? ????? ??.
         - ????: ???? ?????? ??????? ???????.
         - ?? ??? ??? ????? ?????? ? ???? ?? ?????? ??????? ????? ?????.
    """
    if df is None or df.empty:
        return pd.DataFrame(
            {"value": []},
            index=pd.DatetimeIndex([], name="date"),
        )

    # ?????? ?????? ???
    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = None
        for c in df.columns:
            lc = str(c).lower()
            if lc in ("date", "time", "timestamp"):
                date_col = c
                break
        if date_col is None:
            # fallback: ?????? ???????
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)

    df = df.sort_index()

    # ????? ????? value
    if value_col and value_col in df.columns:
        ser = pd.to_numeric(df[value_col], errors="coerce")
    else:
        # ???? ?? ?????? ??????? ???????
        num_cols = [
            c for c in df.columns if np.issubdtype(df[c].dtype, np.number)
        ]
        if not num_cols:
            # fallback: ???? ??? ?? ?????? ???????
            num_cols = list(df.columns)
        ser = pd.to_numeric(df[num_cols[0]], errors="coerce")

    return pd.DataFrame({"value": ser.astype(float)}).dropna()


def _resample(df: pd.DataFrame, freq: str, *, method: str = "ffill") -> pd.DataFrame:
    """
    ???? ????????? ?? ????? ??????? ?????:

    freq:
        "D", "B", "W", "M", "Q", "A", "H", "MIN" ???' (???? pandas).
    method:
        "ffill"   ? ???? ?? ???? ?????? ?????? ???? ?????.
        "bfill"   ? ???? ?? ???? ?????? ????? ?????.
        "truncate" ? ?? ????? ??????? ??? ????? (???? ????? NaN).
    """
    if df.empty:
        return df

    freq = (freq or "D").upper()

    try:
        if method == "ffill":
            out = df.resample(freq).last().ffill()
        elif method == "bfill":
            out = df.resample(freq).first().bfill()
        elif method == "truncate":
            out = df.resample(freq).last()
        else:
            out = df.resample(freq).last().ffill()
        return out
    except Exception as exc:
        LOGGER.warning("resample failed for freq=%s, method=%s: %s", freq, method, exc)
        return df


# ============================================================
# Source adapters (guarded)
# ============================================================

class _Loader:
    """
    Source adapters for loading macro data from various sources.

    URI schemes:
      - local:/path/to/file.csv
      - duckdb:/path/to/db.duckdb|table_or_query
      - sql:CONNSTR|QUERY
      - yf:SPY
      - yf:SPY|period=5y,interval=1wk,field=Adj Close
      - fmp:INDICATOR_NAME (e.g., fmp:GDP, fmp:CPI, fmp:unemploymentRate)
      - fmp_price:TICKER (e.g., fmp_price:^VIX, fmp_price:SPY)
    """

    @staticmethod
    def local_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def duckdb(spec: str) -> pd.DataFrame:
        """
        spec: "DB_PATH|TABLE_OR_QUERY"
        ?? ??????? ???? duckdb: ?? ????? '|', ????? ValueError.
        """
        try:
            import duckdb  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ImportError("duckdb ?? ????? ??????") from e

        if "|" not in spec:
            raise ValueError("duckdb: ?? ?????? ?????? DB_PATH|TABLE_OR_QUERY")

        db_path, table_or_query = spec.split("|", 1)
        con = duckdb.connect(database=db_path)
        try:
            if table_or_query.strip().lower().startswith("select "):
                df = con.execute(table_or_query).fetchdf()
            else:
                df = con.execute(f"SELECT * FROM {table_or_query}").fetchdf()
        finally:
            con.close()
        return df

    @staticmethod
    def sql(spec: str) -> pd.DataFrame:
        """
        spec: "CONNSTR|QUERY"
        ??????:
            "postgresql://user:pass@host/db|SELECT date, value FROM macro_table"
        """
        if "|" not in spec:
            raise ValueError("sql: ?? ?????? ?????? CONNSTR|QUERY")

        conn_str, query = spec.split("|", 1)
        try:
            import sqlalchemy as sa  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ImportError("sqlalchemy ?? ????? ??????") from e

        engine = sa.create_engine(conn_str)
        try:
            df = pd.read_sql_query(query, engine)
        finally:
            engine.dispose()
        return df

    @staticmethod
    def yfinance(
        ticker_spec: str,
        *,
        period: str = "10y",
        interval: str = "1d",
        field: str = "Close",
    ) -> pd.DataFrame:
        """
        ???? ???? ??? ?-yfinance.

        ticker_spec:
            - "SPY" ? ????? ???? (period=10y, interval=1d, field=Close)
            - "SPY|period=5y,interval=1wk,field=Adj Close" ? override ???????

        ?????:
            DataFrame ?? ??????:
                - DatetimeIndex (index ?-yfinance)
                - "value" ? ???? ????? (field)
        """
        try:
            import yfinance as yf  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ImportError("yfinance ?? ????? ??????") from e

        # Parse ticker_spec (ticker|key=val,...)
        ticker = ticker_spec
        local_period = period
        local_interval = interval
        local_field = field

        if "|" in ticker_spec:
            ticker, opts = ticker_spec.split("|", 1)
            for part in opts.split(","):
                part = part.strip()
                if not part or "=" not in part:
                    continue
                k, v = part.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                if k == "period":
                    local_period = v
                elif k == "interval":
                    local_interval = v
                elif k == "field":
                    local_field = v

        LOGGER.info(
            "Downloading macro series from yfinance: ticker=%s, period=%s, interval=%s, field=%s",
            ticker,
            local_period,
            local_interval,
            local_field,
        )

        try:
            data = yf.download(
                ticker,
                period=local_period,
                interval=local_interval,
                auto_adjust=False,
                progress=False,
            )
        except TypeError as exc:  # ????? ?????? API
            raise RuntimeError(
                f"yfinance.download ???? ???????? (ticker={ticker}, period={local_period}, interval={local_interval})"
            ) from exc

        if data is None or data.empty:
            raise RuntimeError(f"yfinance ????? DataFrame ??? ?-ticker={ticker!r}")

        if local_field not in data.columns:
            raise KeyError(
                f"??? {local_field!r} ?? ???? ????? ?? yfinance. ?????? ??????: {list(data.columns)}"
            )

        data = data.rename(columns={local_field: "value"})
        return data[["value"]].reset_index()

    @staticmethod
    def fmp_economic(indicator: str) -> pd.DataFrame:
        """Fetch economic indicator from FMP.

        indicator: FMP indicator name (GDP, CPI, unemploymentRate, etc.)
        """
        try:
            from common.fmp_client import get_fmp_client
        except Exception as e:
            raise ImportError("fmp_client not available") from e

        client = get_fmp_client()
        df = client.get_economic_indicator(indicator, start="2000-01-01")
        if df.empty:
            raise RuntimeError(f"FMP returned empty data for indicator={indicator!r}")

        # normalize to date + value
        if "value" not in df.columns:
            num_cols = [c for c in df.columns if c not in ("date", "country", "currency")]
            if num_cols:
                df = df.rename(columns={num_cols[0]: "value"})
        return df

    @staticmethod
    def fmp_price(ticker: str) -> pd.DataFrame:
        """Fetch price series from FMP for use as macro indicator."""
        try:
            from common.fmp_client import get_fmp_client
        except Exception as e:
            raise ImportError("fmp_client not available") from e

        client = get_fmp_client()
        df = client.get_historical_prices(ticker, start="2010-01-01")
        if df.empty:
            raise RuntimeError(f"FMP returned empty prices for ticker={ticker!r}")

        df = df.rename(columns={"close": "value", "datetime": "date"})
        return df[["date", "value"]].dropna()


DEFAULT_MACRO_SOURCES = {
    # ── FMP economic calendar-based indicators ──
    "CPI_USA": "fmp:CPI",
    "GDP_USA": "fmp:GDP",
    "NONFARM_PAYROLLS": "fmp:Nonfarm",
    "UNEMPLOYMENT_USA": "fmp:Unemployment Rate",
    "FEDERAL_FUNDS_RATE": "fmp:Fed Interest Rate",
    "CONSUMER_SENTIMENT": "fmp:Consumer Sentiment",
    "RETAIL_SALES": "fmp:Retail Sales",
    "INDUSTRIAL_PRODUCTION": "fmp:Industrial Production",

    # ── FMP price-based macro proxies (these all work) ──
    "VIX_INDEX": "fmp_price:^VIX",
    "SPX_INDEX": "fmp_price:SPY",
    "DXY_INDEX": "fmp_price:DX-Y.NYB",
    "GOLD": "fmp_price:GC=F",
    "OIL": "fmp_price:CL=F",
    "US10Y": "fmp_price:^TNX",
    "HYG": "fmp_price:HYG",
    "LQD": "fmp_price:LQD",
    "TLT": "fmp_price:TLT",

    # ── yfinance fallbacks ──
    "MOVE_INDEX": "yf:^MOVE|period=10y,interval=1d,field=Close",
}

# ============================================================
# Cache layer ? TTL cache
# ============================================================

@dataclass
class _CacheItem:
    df: pd.DataFrame
    ts: float


class _TTLCache:
    """
    Cache ???? ?? TTL:

    - key: Tuple[indicator_id, config_key] (???? (GDP_USA, "D|ffill")).
    - value: DataFrame ?????? ????????.
    - TTL: ??? ?????? ?? ??????? ???? "?? ????".

    ??????:
        get(key) -> DataFrame | None
        set(key, df)
        clear()
        stats() -> dict
    """

    def __init__(self, ttl_seconds: int = 3600, max_items: int = 128) -> None:
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._store: Dict[Tuple[str, str], _CacheItem] = {}

    def get(self, key: Tuple[str, str]) -> Optional[pd.DataFrame]:
        item = self._store.get(key)
        if not item:
            return None
        if (time.time() - item.ts) > self.ttl:
            # ?? ????
            self._store.pop(key, None)
            return None
        return item.df

    def set(self, key: Tuple[str, str], df: pd.DataFrame) -> None:
        if len(self._store) >= self.max_items:
            # ????????? ????? ?????: ????? ?????? ??? ???
            oldest = min(self._store.items(), key=lambda kv: kv[1].ts)[0]
            self._store.pop(oldest, None)
        self._store[key] = _CacheItem(df=df.copy(), ts=time.time())

    def clear(self) -> None:
        """???? ?? ?? ?-cache."""
        self._store.clear()

    def stats(self) -> Dict[str, Any]:
        """????? ????? ?????? ?????/?????."""
        now = time.time()
        alive = {
            k: v for k, v in self._store.items()
            if (now - v.ts) <= self.ttl
        }
        return {
            "size": len(self._store),
            "alive": len(alive),
            "expired": len(self._store) - len(alive),
            "ttl_seconds": self.ttl,
            "max_items": self.max_items,
        }


# ============================================================
# Client
# ============================================================

@dataclass
class MacroDataClient:
    """
    MacroDataClient ? ???? ???? ????? ?????.

    ????:
        sources:
            dict {indicator_id: uri} ??? ?????:
                "local:..." / "duckdb:..." / "sql:..." / "yf:...".
        allow_duckdb:
            ??? ????? ???? ?-duckdb (?????? / ?????).
        allow_sql:
            ??? ????? ???? ??????? SQL (??????).
        allow_yf:
            ??? ????? ???? ?-yfinance (???? ???????).
        cache_ttl_seconds:
            ??? ??? (??????) ????? ????? ?-cache.
        cache_max_items:
            ??? ??????????? ???? ????? ?-cache.

    ????? ??????:
        client = MacroDataClient(
            sources={
                "CPI_USA": "local:data/macro/cpi_usa.csv",
                "SPX_INDEX": "yf:^GSPC|period=20y,interval=1d,field=Adj Close",
                "GDP_USA": "duckdb:data/macro.duckdb|gdp_usa_quarterly",
            },
            allow_duckdb=True,
            allow_sql=False,
            allow_yf=True,
        )

        df = client.get("CPI_USA", start="2010-01-01", end="2025-01-01", freq="M")
    """

    sources: Dict[str, str]
    allow_duckdb: bool = True
    allow_sql: bool = False
    allow_yf: bool = True
    allow_fmp: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_items: int = 128

    def __post_init__(self) -> None:
        self._cache = _TTLCache(self.cache_ttl_seconds, self.cache_max_items)

    # ---------- Internal helpers ----------

    def _resolve_source(self, indicator_id: str) -> Optional[str]:
        return self.sources.get(indicator_id)

    def _load(self, uri: str) -> pd.DataFrame:
        """
        ???? DataFrame ????? ??? ?-URI ??? ?????????/??????.
        """
        if uri.startswith("local:"):
            return _Loader.local_csv(uri.split("local:", 1)[1])

        if uri.startswith("duckdb:"):
            if not self.allow_duckdb:
                raise RuntimeError("duckdb ????? ???????????? (allow_duckdb=False).")
            return _Loader.duckdb(uri.split("duckdb:", 1)[1])

        if uri.startswith("sql:"):
            if not self.allow_sql:
                raise RuntimeError("SQL ????? ???????????? (allow_sql=False).")
            return _Loader.sql(uri.split("sql:", 1)[1])

        if uri.startswith("yf:"):
            if not self.allow_yf:
                raise RuntimeError("yfinance not enabled (allow_yf=False).")
            return _Loader.yfinance(uri.split("yf:", 1)[1])

        if uri.startswith("fmp_price:"):
            if not self.allow_fmp:
                raise RuntimeError("FMP not enabled (allow_fmp=False).")
            return _Loader.fmp_price(uri.split("fmp_price:", 1)[1])

        if uri.startswith("fmp:"):
            if not self.allow_fmp:
                raise RuntimeError("FMP not enabled (allow_fmp=False).")
            return _Loader.fmp_economic(uri.split("fmp:", 1)[1])

        # fallback: try as local CSV
        return _Loader.local_csv(uri)

    # ---------- Public API ----------

    def get(
        self,
        indicator_id: str,
        start: Optional[str | pd.Timestamp] = None,
        end: Optional[str | pd.Timestamp] = None,
        freq: str = "D",
        *,
        lag_policy: str = "ffill",
    ) -> pd.DataFrame:
        """
        ????? ????????? ????? ???????? TimeSeries ??????? (DataFrame):

        ???????
        --------
        indicator_id : str
            ?? ?????????? ??? ?????? ?-sources.
        start, end : str | Timestamp | None
            ???? ??????? ????. ?? None ? ????? ?? ?? ?????.
        freq : str
            ?????? ????? ("D", "W", "M", "Q", "B", ...).
        lag_policy : str
            'ffill' / 'bfill' / 'truncate' ? ??????? ??? ?????????.

        ?????
        -----
        DataFrame ??:
          - DatetimeIndex ??????.
          - ????? ???: "value" (float).
        """
        uri = self._resolve_source(indicator_id)
        if not uri:
            raise KeyError(f"MacroDataClient: missing source for indicator: {indicator_id!r}")

        cache_key = (indicator_id, f"{freq}|{lag_policy}")
        cached = self._cache.get(cache_key)
        if cached is not None:
            df = cached
        else:
            raw = self._load(uri)
            df = _ensure_ts_df(raw)
            df = _resample(df, freq, method=lag_policy)
            self._cache.set(cache_key, df)

        # ????? ????? ??? ?? ????
        if start is not None:
            df = df.loc[pd.to_datetime(start) :]
        if end is not None:
            df = df.loc[: pd.to_datetime(end)]

        return df.copy()

    # Utility methods ????? ?-cache (?? ???? ??????)
    def clear_cache(self) -> None:
        """???? ?? ?-cache ?????? (???? ?? ????? ????? ???)."""
        self._cache.clear()

    def cache_stats(self) -> Dict[str, Any]:
        """???? ????? ????? ?? ?-cache (?????? debug/monitoring)."""
        return self._cache.stats()


def get_macro_client(sources: Optional[Dict[str, str]] = None) -> MacroDataClient:
    """Get a macro client with FMP-powered defaults."""
    return MacroDataClient(
        sources=sources or DEFAULT_MACRO_SOURCES,
        allow_duckdb=True,
        allow_sql=False,
        allow_yf=True,
        allow_fmp=True,
        cache_ttl_seconds=3600,
        cache_max_items=256,
    )


__all__ = ["MacroDataClient", "DEFAULT_MACRO_SOURCES", "get_macro_client"]
