# -*- coding: utf-8 -*-
"""
common/data_service.py — SQL-First Market Data Access Service
=============================================================

Single entry point for all market data access in the system.

Architecture
------------
1. SQL (DuckDB/SQLite) — canonical source of truth (`pt_prices` table)
2. FMP — canonical live provider, populates SQL when data is missing/stale
3. IBKR — live/paper trading data (if connected)
4. Yahoo — last-resort legacy fallback (disabled by default)

Usage
-----
    from common.data_service import DataService, get_data_service

    svc = get_data_service()

    # Returns close-price pivot: index=date, columns=symbol
    prices = svc.get_prices(["AAPL", "MSFT"], start="2022-01-01")

    # Wide OHLCV DataFrame with symbol column
    ohlcv = svc.get_ohlcv(["AAPL"], start="2022-01-01", end="2023-12-31")

    # Force refresh from FMP even if SQL has data
    prices = svc.get_prices(["AAPL"], force_refresh=True)

Design principles
-----------------
- All callers use this service; no module should import yfinance directly.
- SQL is read first; network fetch only happens when data is missing or stale.
- Every fetch populates SQL so subsequent calls are fast.
- Staleness threshold: if most recent row is > `stale_days` old, refresh.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger("DataService")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DataServiceConfig:
    """Runtime configuration for DataService."""

    # Staleness: if newest SQL row is older than this, re-fetch from provider
    stale_days: int = 2

    # Lookback used when no start date is given (days)
    default_lookback_days: int = 730  # 2 years

    # SQL table name (no prefix — SqlStore adds its own prefix)
    prices_table: str = "prices"

    # Whether to write fetched data back to SQL automatically
    write_through: bool = True

    # FMP API key override (None = read from env / config)
    fmp_api_key: Optional[str] = None

    # Include Yahoo as last-resort fallback
    use_yahoo_fallback: bool = False


# ---------------------------------------------------------------------------
# DataService
# ---------------------------------------------------------------------------

class DataService:
    """
    SQL-first market data access layer.

    Thread-safe singleton (use `get_data_service()` for the global instance).
    """

    def __init__(
        self,
        sql_store: Any = None,
        config: DataServiceConfig | None = None,
        ib: Any = None,
    ) -> None:
        """
        Parameters
        ----------
        sql_store : SqlStore | None
            Injected SqlStore instance. If None, a default one is created.
        config : DataServiceConfig | None
            Service config. Uses defaults if None.
        ib : IB | None
            Connected ib_insync.IB instance for live data.
        """
        self._config = config or DataServiceConfig()
        self._sql = sql_store
        self._ib = ib
        self._router = None
        self._lock = threading.Lock()
        self._initialized = False

    # ------------------------------------------------------------------
    # Lazy initialization (avoid heavy imports at module load)
    # ------------------------------------------------------------------

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self._do_init()
            self._initialized = True

    def _do_init(self) -> None:
        # Build SQL store if not injected
        if self._sql is None:
            try:
                from core.sql_store import SqlStore
                self._sql = SqlStore.from_settings({})
                logger.info("DataService: created default SqlStore.")
            except Exception as exc:
                logger.warning("DataService: could not create SqlStore: %s", exc)
                self._sql = None

        # Build market data router (FMP-first)
        try:
            from common.market_data_router import build_default_router
            self._router = build_default_router(
                ib=self._ib,
                use_fmp=True,
                fmp_api_key=self._config.fmp_api_key,
                use_yahoo=self._config.use_yahoo_fallback,
            )
            logger.info("DataService: market data router initialized.")
        except Exception as exc:
            logger.warning("DataService: could not build router: %s", exc)
            self._router = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prices(
        self,
        symbols: Sequence[str],
        *,
        start: str | None = None,
        end: str | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return a close-price pivot table.

        Returns
        -------
        pd.DataFrame
            Index: DatetimeIndex (date), columns: symbol, values: close price.
        """
        ohlcv = self.get_ohlcv(
            symbols, start=start, end=end, force_refresh=force_refresh
        )
        if ohlcv.empty:
            return pd.DataFrame()

        dt_col = "datetime" if "datetime" in ohlcv.columns else "date"
        return (
            ohlcv[["symbol", dt_col, "close"]]
            .drop_duplicates(subset=["symbol", dt_col])
            .pivot(index=dt_col, columns="symbol", values="close")
            .sort_index()
        )

    def get_ohlcv(
        self,
        symbols: Sequence[str],
        *,
        start: str | None = None,
        end: str | None = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return wide OHLCV DataFrame with a `symbol` column.

        Flow:  SQL → (stale check) → FMP fetch → SQL write → return
        """
        self._ensure_init()
        syms = [s.upper().strip() for s in symbols if s and str(s).strip()]
        if not syms:
            return pd.DataFrame()

        start_dt = self._resolve_start(start)
        end_dt = datetime.now(tz=timezone.utc) if end is None else pd.Timestamp(end).tz_localize("UTC")

        # 1. Try SQL first
        if not force_refresh:
            cached = self._read_from_sql(syms, start_dt, end_dt)
            if not cached.empty and not self._is_stale(cached):
                logger.debug("DataService: returning %d rows from SQL cache.", len(cached))
                return cached

        # 2. Fetch from provider
        fresh = self._fetch_from_provider(syms, start_dt, end_dt)

        # 3. Write through to SQL
        if not fresh.empty and self._config.write_through:
            self._write_to_sql(fresh)

        # 4. If provider failed, return whatever SQL had
        if fresh.empty:
            cached = self._read_from_sql(syms, start_dt, end_dt)
            if not cached.empty:
                logger.warning(
                    "DataService: provider returned no data; using SQL cache for %s.", syms
                )
                return cached
            logger.error("DataService: no data available for %s.", syms)
            return pd.DataFrame()

        return fresh

    def is_healthy(self) -> bool:
        """Quick liveness check: SQL + router."""
        self._ensure_init()
        router_ok = self._router is not None
        sql_ok = self._sql is not None
        return router_ok or sql_ok

    def describe(self) -> dict:
        """Return a summary dict for dashboards/health checks."""
        self._ensure_init()
        result: dict = {
            "sql_available": self._sql is not None,
            "router_available": self._router is not None,
            "stale_days": self._config.stale_days,
            "write_through": self._config.write_through,
        }
        if self._router is not None:
            try:
                result["providers"] = list(self._router.providers.keys())
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # SQL helpers
    # ------------------------------------------------------------------

    def _read_from_sql(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        if self._sql is None:
            return pd.DataFrame()
        try:
            tbl = self._sql._tbl(self._config.prices_table)
            syms_sql = ", ".join(f"'{s}'" for s in symbols)
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")
            query = (
                f"SELECT symbol, date AS datetime, open, high, low, close, volume "
                f"FROM {tbl} "
                f"WHERE symbol IN ({syms_sql}) "
                f"  AND date >= '{start_str}' "
                f"  AND date <= '{end_str}' "
                f"ORDER BY symbol, date"
            )
            from sqlalchemy import text
            with self._sql.engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
            if not df.empty:
                df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        except Exception as exc:
            logger.warning("DataService._read_from_sql failed: %s", exc)
            return pd.DataFrame()

    def _write_to_sql(self, df: pd.DataFrame) -> None:
        if self._sql is None:
            return
        try:
            self._sql._ensure_prices_schema()
            write_df = df.copy()
            # Normalize: SQL table uses 'date' column
            if "datetime" in write_df.columns and "date" not in write_df.columns:
                write_df = write_df.rename(columns={"datetime": "date"})
            write_df["ts_utc"] = datetime.now(tz=timezone.utc).isoformat()
            write_df["env"] = "prod"

            tbl = self._sql._tbl(self._config.prices_table)
            cols = ["symbol", "date", "open", "high", "low", "close", "volume", "ts_utc", "env"]
            write_df = write_df[[c for c in cols if c in write_df.columns]]

            # Delete existing rows for these symbols+dates, then insert fresh
            syms = write_df["symbol"].unique().tolist()
            syms_sql = ", ".join(f"'{s}'" for s in syms)
            if "date" in write_df.columns:
                min_date = write_df["date"].min()
                max_date = write_df["date"].max()
                del_sql = (
                    f"DELETE FROM {tbl} "
                    f"WHERE symbol IN ({syms_sql}) "
                    f"  AND date >= '{min_date}' "
                    f"  AND date <= '{max_date}'"
                )
                from sqlalchemy import text
                with self._sql.engine.begin() as conn:
                    conn.execute(text(del_sql))

            write_df.to_sql(
                self._sql._tbl(self._config.prices_table),
                con=self._sql.engine,
                if_exists="append",
                index=False,
            )
            logger.info(
                "DataService: wrote %d rows to SQL (%s).", len(write_df), tbl
            )
        except Exception as exc:
            logger.warning("DataService._write_to_sql failed: %s", exc)

    def _is_stale(self, df: pd.DataFrame) -> bool:
        """Return True if the newest data point is older than stale_days."""
        dt_col = "datetime" if "datetime" in df.columns else "date"
        if dt_col not in df.columns or df.empty:
            return True
        newest = pd.to_datetime(df[dt_col]).max()
        if pd.isnull(newest):
            return True
        cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=self._config.stale_days)
        # strip tz for comparison
        newest_naive = newest.tz_localize(None) if newest.tzinfo else newest
        return newest_naive < cutoff

    # ------------------------------------------------------------------
    # Provider helpers
    # ------------------------------------------------------------------

    def _fetch_from_provider(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        if self._router is None:
            logger.warning("DataService: no router available for fetch.")
            return pd.DataFrame()
        try:
            start_str = start.strftime("%Y-%m-%d")
            end_str = end.strftime("%Y-%m-%d")
            result = self._router.get_history(
                symbols=symbols,
                start=start_str,
                end=end_str,
            )
            if result.ok:
                logger.info(
                    "DataService: fetched %d rows from '%s' for %s.",
                    len(result.df), result.source, symbols,
                )
                return result.df
            else:
                logger.warning(
                    "DataService: router returned no data for %s. Errors: %s",
                    symbols, result.errors,
                )
                return pd.DataFrame()
        except Exception as exc:
            logger.error("DataService._fetch_from_provider failed: %s", exc)
            return pd.DataFrame()

    def _resolve_start(self, start: str | None) -> datetime:
        if start is not None:
            return pd.Timestamp(start).tz_localize("UTC").to_pydatetime()
        return (
            datetime.now(tz=timezone.utc)
            - timedelta(days=self._config.default_lookback_days)
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_service: DataService | None = None
_singleton_lock = threading.Lock()


def get_data_service(
    *,
    sql_store: Any = None,
    config: DataServiceConfig | None = None,
    ib: Any = None,
    reset: bool = False,
) -> DataService:
    """
    Return the module-level DataService singleton.

    Parameters
    ----------
    sql_store : SqlStore | None
        Override the SQL store (used in tests or custom deployments).
    config : DataServiceConfig | None
        Override config (used in tests).
    ib : IB | None
        Connected IB instance for live data.
    reset : bool
        If True, discard the existing singleton and create a fresh one.
    """
    global _default_service
    with _singleton_lock:
        if reset or _default_service is None:
            _default_service = DataService(
                sql_store=sql_store, config=config, ib=ib
            )
    return _default_service
