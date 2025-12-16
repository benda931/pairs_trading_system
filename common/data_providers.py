# -*- coding: utf-8 -*-
"""
common/data_providers.py ג€” Unified Market-Data Providers Layer (HF-grade)
========================================================================

׳׳˜׳¨׳× ׳”׳§׳•׳‘׳¥:
-----------

1. ׳׳¡׳₪׳§ ׳©׳›׳‘׳” ׳׳—׳™׳“׳” ׳׳›׳ ׳׳§׳•׳¨׳•׳× ׳”׳“׳׳˜׳” (IBKR, Yahoo, ׳¡׳₪׳§׳™׳ ׳׳§׳¦׳•׳¢׳™׳™׳ ׳ ׳•׳¡׳₪׳™׳).
2. ׳׳”׳¡׳×׳™׳¨ ׳׳× ׳”׳₪׳¨׳˜׳™׳ ׳©׳ ׳›׳ ׳¡׳₪׳§ (API / Rate Limits / ׳₪׳•׳¨׳׳˜), ׳•׳׳”׳—׳–׳™׳¨ ׳×׳׳™׳“ DataFrame ׳׳—׳™׳“.
3. ׳׳׳₪׳©׳¨ ׳‘׳—׳™׳¨׳” ׳“׳™׳ ׳׳™׳× ׳©׳ ׳¡׳₪׳§ (׳¢"׳™ ׳”׳׳©׳×׳׳© ׳׳• ׳׳•׳˜׳•׳׳˜׳™׳× ׳¢"׳™ Router ׳—׳™׳¦׳•׳ ׳™).
4. ׳׳׳₪׳©׳¨ ׳”׳¨׳—׳‘׳” ׳¢׳×׳™׳“׳™׳× (Polygon, Tiingo, Alpaca ׳•׳›׳•') ׳‘׳׳™ ׳׳’׳¢׳× ׳‘׳§׳•׳“ ׳©׳ ׳”׳˜׳׳‘׳™׳/׳”׳׳•׳’׳™׳§׳”.

׳₪׳•׳¨׳׳˜ ׳׳—׳™׳“ ׳©׳ DataFrame (wide "standardized"):
-----------------------------------------------

׳›׳ Provider ׳׳—׳–׳™׳¨ DataFrame ׳‘׳₪׳•׳¨׳׳˜ ׳”׳‘׳ (׳׳₪׳—׳•׳×):
    - columns:
        symbol      : str   ג€” ׳¡׳™׳׳‘׳•׳ (QQQ, XLY, ׳•׳›׳•')
        datetime    : datetime64[ns]
        open        : float
        high        : float
        low         : float
        close       : float
        volume      : float | int | NaN

    - index: RangeIndex (׳׳ ׳—׳•׳‘׳” MultiIndex)

׳©׳׳¨ ׳”׳¢׳׳•׳“׳•׳× (dividends, average_price, etc) ׳”׳ ׳׳•׳₪׳¦׳™׳•׳ ׳׳™׳•׳×.

׳©׳™׳׳•׳© ׳‘׳¡׳™׳¡׳™:
------------

    from common.data_providers import IBKRProvider, YahooProvider, normalize_symbols

    ib = ...  # IB instance from ibkr_connection.get_ib()
    ib_provider = IBKRProvider(ib)

    df = ib_provider.get_history(["XLY", "XLC"], period="6mo", bar_size="1d")

Router ׳—׳›׳ (׳‘׳—׳™׳¨׳” ׳‘׳™׳ ׳¡׳₪׳§׳™׳) ׳׳•׳׳׳¥ ׳׳™׳™׳©׳ ׳‘׳§׳•׳‘׳¥ ׳ ׳₪׳¨׳“:
    common/market_data_router.py
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Literal, Mapping, Sequence

import pandas as pd

# ========================= Logging setup =========================

logger = logging.getLogger("MarketData")
if not logger.handlers:
    # ׳׳ ׳×׳”׳¨׳•׳¡ ׳§׳•׳ ׳₪׳™׳’ ׳’׳׳•׳‘׳׳™, ׳¨׳§ ׳“׳™׳₪׳•׳׳˜ ׳¢׳“׳™׳
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ========================= Type aliases ==========================

BarSize = Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
WhatToShow = Literal["TRADES", "MIDPOINT", "BID", "ASK"]


# ========================= Base abstraction ======================


class MarketDataProvider(ABC):
    """
    Abstract base class for all market-data providers.

    Each provider:
    --------------
    - Implements `get_history` and returns a standardized OHLCV DataFrame.
    - Has a unique `name` used by routers / configs.
    - May implement a `healthcheck` to indicate availability.
    """

    name: str = "abstract"
    priority: int = 100  # ׳ ׳׳•׳ ׳™׳•׳×׳¨ = ׳¢׳“׳™׳₪׳•׳× ׳’׳‘׳•׳”׳” ׳™׳•׳×׳¨

    @abstractmethod
    def get_history(
        self,
        symbols: Sequence[str],
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        period: str | None = "6mo",
        bar_size: BarSize = "1d",
        what_to_show: WhatToShow = "TRADES",
        use_rth: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for the requested symbols.

        Parameters
        ----------
        symbols : Sequence[str]
            List of tickers.
        start, end : str | datetime | None
            Optional explicit boundaries. If not given, `period` is used.
        period : str | None
            Rolling lookback, like '6mo', '1y', '90d'.
        bar_size : BarSize
            Candle aggregation size.
        what_to_show : WhatToShow
            TRADES/MIDPOINT/etc. Some providers may ignore this.
        use_rth : bool
            Use regular trading hours only (if applicable).

        Returns
        -------
        pd.DataFrame
            Standardized OHLCV with 'symbol' + 'datetime' columns.
        """
        raise NotImplementedError

    def healthcheck(self) -> bool:
        """
        Simple liveness check. Can be overridden to ping the provider.
        By default returns True (optimistic).
        """
        return True


# ========================= Symbol normalization ==================


def normalize_symbols(raw: Any) -> list[str]:
    """Normalize any messy symbol input into a clean ``list[str]``.

    Supports ׳‘׳™׳ ׳”׳™׳×׳¨:
    --------------------
    - "QQQ"
    - ["QQQ", "XLY"]
    - {"SYMBOLS": ["QQQ", "XLY"]}
    - "['QQQ', 'XLY']"      (list ׳׳׳•׳¨׳©׳¨)
    - "{'SYMBOLS': ['QQQ', 'XLY']}" (dict ׳׳׳•׳¨׳©׳¨)

    ׳•׳›׳ ׳׳—׳¨׳•׳–׳•׳× ׳׳¨׳•׳‘׳•׳× ׳˜׳™׳§׳¨׳™׳ ׳›׳׳•:
    - "XLY XLC"
    - "XLY,XLC"

    ׳©׳“׳¨׳•׳’ ׳—׳©׳•׳‘: ׳”׳₪׳•׳ ׳§׳¦׳™׳” ׳׳ ׳¡׳” ׳׳×׳§׳ ׳’׳ ׳׳§׳¨׳™׳ "׳׳׳•׳›׳׳›׳™׳" ׳›׳׳•:
    - ["{'SYMBOLS': ['XLY', 'XLC']}"]  ג† list ׳¢׳ ׳׳—׳¨׳•׳–׳× ׳׳—׳× ׳©׳ ׳¨׳׳™׳× ׳›׳׳• dict

    ׳›׳ ׳”׳˜׳™׳§׳¨׳™׳ ׳׳•׳›׳ ׳¡׳™׳ ׳-UPPERCASE ׳•׳׳ ׳•׳§׳™׳ ׳׳¨׳•׳•׳—׳™׳.
    """

    import ast

    # None ג†’ empty list
    if raw is None:
        return []

    # Mapping ׳¢׳ ׳׳₪׳×׳— SYMBOLS / symbols (׳§׳•׳ ׳₪׳™׳’׳™׳ ׳‘׳¡׳’׳ ׳•׳ {"SYMBOLS": [...]} ׳׳• {"symbols": [...]})
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            if str(k).upper() == "SYMBOLS":
                # ׳ ׳–׳¨׳•׳§ ׳׳× ׳”׳¢׳¨׳ ׳—׳–׳¨׳” ׳׳ ׳•׳¨׳׳׳™׳–׳¦׳™׳” ׳”׳¨׳’׳™׳׳”
                return normalize_symbols(v)

    # Iterable (list/tuple/set ...) ׳׳‘׳ ׳׳ str/bytes/Mapping
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes, Mapping)):
        seq = list(raw)
        if not seq:
            return []

        # ׳׳§׳¨׳” ׳׳׳•׳›׳׳ ׳ ׳₪׳•׳¥: ["{'SYMBOLS': ['XLY', 'XLC']}"] ׳׳• "['XLY','XLC']"
        if len(seq) == 1 and isinstance(seq[0], str):
            cleaned = seq[0].strip()
            if cleaned.startswith(("{", "[")):
                try:
                    parsed = ast.literal_eval(cleaned)
                    return normalize_symbols(parsed)
                except Exception:
                    logger.debug(
                        "normalize_symbols: failed to parse stringified container %r",
                        cleaned,
                    )
                    return [cleaned.upper()]

        # ׳׳§׳¨׳” ׳¨׳’׳™׳: ׳›׳ ׳׳™׳‘׳¨ ׳”׳•׳ ׳˜׳™׳§׳¨ ׳‘׳₪׳ ׳™ ׳¢׳¦׳׳•
        out: list[str] = []
        for s in seq:
            text = str(s).strip().upper()
            if text:
                out.append(text)
        # ׳”׳¡׳¨׳× ׳›׳₪׳•׳׳™׳ ׳×׳•׳ ׳©׳׳™׳¨׳” ׳¢׳ ׳¡׳“׳¨
        seen: set[str] = set()
        uniq: list[str] = []
        for t in out:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    # String input ג€” ׳™׳›׳•׳ ׳׳”׳™׳•׳× ׳’׳ ׳˜׳™׳§׳¨ ׳‘׳•׳“׳“ ׳•׳’׳ list/dict ׳׳׳•׳¨׳©׳¨
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return []

        # ׳׳ ׳–׳” ׳ ׳¨׳׳” ׳›׳׳• list/dict ג†’ ׳ ׳ ׳¡׳” literal_eval
        if cleaned.startswith(("{", "[")):
            try:
                parsed = ast.literal_eval(cleaned)
                return normalize_symbols(parsed)
            except Exception:
                logger.debug("normalize_symbols: literal_eval failed for %r", cleaned)
                return [cleaned.upper()]

        # ׳×׳׳™׳›׳” ׳‘׳׳₪׳¨׳™׳“׳™׳: ׳₪׳¡׳™׳§׳™׳, ׳ ׳§׳•׳“׳”-׳₪׳¡׳™׳§, ׳¨׳•׳•׳—׳™׳
        tmp = cleaned.replace(";", ",")
        if "," in tmp:
            parts = [p.strip().upper() for p in tmp.split(",") if p.strip()]
            if len(parts) > 1:
                # ׳”׳¡׳¨׳× ׳›׳₪׳•׳׳™׳ ׳×׳•׳ ׳©׳׳™׳¨׳” ׳¢׳ ׳¡׳“׳¨
                seen: set[str] = set()
                uniq: list[str] = []
                for t in parts:
                    if t not in seen:
                        seen.add(t)
                        uniq.append(t)
                return uniq

        # ׳׳ ׳׳™׳ ׳₪׳¡׳™׳§׳™׳ ג€“ ׳ ׳‘׳“׳•׳§ ׳¨׳•׳•׳—׳™׳ ("XLY XLC")
        if " " in cleaned:
            parts = [p.strip().upper() for p in cleaned.split() if p.strip()]
            if len(parts) > 1:
                seen: set[str] = set()
                uniq: list[str] = []
                for t in parts:
                    if t not in seen:
                        seen.add(t)
                        uniq.append(t)
                return uniq

        # ׳˜׳™׳§׳¨ ׳‘׳•׳“׳“
        return [cleaned.upper()]

    # Last resort: stringify ׳›׳ ׳“׳‘׳¨ ׳׳—׳¨
    s = str(raw).strip()
    return [s.upper()] if s else []


# ========================= IBKR Provider =========================
 

try:
    # ib_insync ׳”׳•׳ ׳׳•׳₪׳¦׳™׳•׳ ׳׳™ ג€“ ׳׳ ׳׳ ׳׳•׳×׳§׳, IBKRProvider ׳׳ ׳™׳”׳™׳” ׳₪׳¢׳™׳
    from ib_insync import IB, Stock, Future, util as ib_util  # type: ignore[import]
except Exception:  # pragma: no cover - optional dep
    IB = None          # type: ignore[assignment]
    Stock = None       # type: ignore[assignment]
    Future = None      # type: ignore[assignment]
    ib_util = None     # type: ignore[assignment]
    logger.warning("ib_insync is not available ג€” IBKRProvider will be disabled.")


@dataclass
class IBKRContractConfig:
    """
    ׳”׳’׳“׳¨׳•׳× ׳׳‘׳ ׳™׳™׳× ׳—׳•׳–׳” IBKR ׳׳×׳•׳ ׳¡׳™׳׳‘׳•׳.

    ׳׳₪׳©׳¨ ׳׳”׳¨׳—׳™׳‘ ׳‘׳¢׳×׳™׳“ ׳:
    - Futures
    - FX
    - Options
    - ׳”׳’׳“׳¨׳•׳× ׳©׳•׳ ׳•׳× ׳ײ¾exchange/currency ׳•׳›׳•'.
    """

    sec_type: Literal["STK", "FUT"] = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    primary_exchange: str | None = None  # ׳׳“׳•׳’׳׳” "NASDAQ"


class IBKRProvider(MarketDataProvider):
    """
    Interactive Brokers market-data provider.

    ׳“׳¨׳™׳©׳•׳×:
    --------
    - ׳׳•׳‘׳™׳™׳§׳˜ IB (ib_insync.IB) ׳׳—׳•׳‘׳¨ ׳›׳‘׳¨ ׳ײ¾TWS / Gateway.
    - ׳”׳×׳§׳ ׳× ib_insync.

    ׳”׳”׳—׳–׳¨׳”:
    -------
    DataFrame ׳׳—׳™׳“ ׳¢׳ ׳¢׳׳•׳“׳•׳×:
    symbol, datetime, open, high, low, close, volume
    """

    name: str = "ibkr"
    priority: int = 10  # ׳¢׳“׳™׳₪׳•׳× ׳’׳‘׳•׳”׳” (׳׳¡׳₪׳¨ ׳ ׳׳•׳ ׳™׳•׳×׳¨ ׳˜׳•׳‘)

    def __init__(
        self,
        ib: Any,  # ׳‘׳׳§׳•׳ "IB" ׳›׳“׳™ ׳׳ ׳׳¢׳¦׳‘׳ ׳׳× Pylance
        *,
        contract_config: IBKRContractConfig | None = None,
        max_lookback_days: int = 365,
        ignore_empty: bool = True,
    ) -> None:
        if ib_util is None:
            # ׳׳™׳ ib_insync ׳׳•׳×׳§׳
            raise RuntimeError(
                "IBKRProvider cannot be used because ib_insync is not installed."
            )

        self.ib = ib
        self.contract_config = contract_config or IBKRContractConfig()
        self.max_lookback_days = max_lookback_days
        self.ignore_empty = ignore_empty

    # ---------- Public API ----------

    def get_history(
        self,
        symbols: Sequence[str],
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        period: str | None = "6mo",
        bar_size: BarSize = "1d",
        what_to_show: WhatToShow = "TRADES",
        use_rth: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()

        norm_syms = normalize_symbols(symbols)
        frames: list[pd.DataFrame] = []

        for sym in norm_syms:
            contract = self._build_contract(sym)
            duration_str = self._to_ib_duration(period=period, start=start, end=end)
            bar_size_str = self._to_ib_bar_size(bar_size)

            logger.info(
                "IBKRProvider.get_history: symbol=%s duration=%s bar_size=%s what=%s",
                sym,
                duration_str,
                bar_size_str,
                what_to_show,
            )

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration_str,
                barSizeSetting=bar_size_str,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
                keepUpToDate=False,
            )

            if not bars:
                logger.warning("IBKRProvider: no data for %s", sym)
                if not self.ignore_empty:
                    frames.append(
                        pd.DataFrame(
                            {
                                "symbol": [sym],
                                "datetime": [pd.NaT],
                                "open": [pd.NA],
                                "high": [pd.NA],
                                "low": [pd.NA],
                                "close": [pd.NA],
                                "volume": [pd.NA],
                            }
                        )
                    )
                continue

            df = ib_util.df(bars)  # type: ignore[arg-type]
            if df.empty:
                continue

            df["symbol"] = sym
            df = df.rename(
                columns={
                    "date": "datetime",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )

            cols = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
            df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]

            frames.append(df)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], utc=False)

        return out

    def healthcheck(self) -> bool:
        try:
            _ = self.ib.reqCurrentTime()
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("IBKRProvider.healthcheck failed: %s", exc)
            return False

    # ---------- Helpers ----------

    def _build_contract(self, symbol: str):
        cfg = self.contract_config
        if cfg.sec_type == "STK":
            c = Stock(
                symbol,
                cfg.exchange,
                cfg.currency,
                primaryExchange=cfg.primary_exchange,
            )
        elif cfg.sec_type == "FUT":
            c = Future(symbol, cfg.exchange, cfg.currency)
        else:
            raise ValueError(f"Unsupported sec_type for IBKR: {cfg.sec_type}")
        return c

    def _to_ib_bar_size(self, bar_size: BarSize) -> str:
        mapping: dict[BarSize, str] = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day",
            "1w": "1 week",
        }
        return mapping.get(bar_size, "1 day")

    def _to_ib_duration(
        self,
        *,
        period: str | None,
        start: str | datetime | None,
        end: str | datetime | None,
    ) -> str:
        if period:
            p = period.lower()
            if p.endswith("mo"):
                return f"{p[:-2]} M"
            if p.endswith("y"):
                return f"{p[:-1]} Y"
            if p.endswith("d"):
                return f"{p[:-1]} D"
            if p.endswith("w"):
                return f"{int(p[:-1]) * 7} D"
            return "6 M"

        if isinstance(start, datetime) and isinstance(end, datetime):
            days = (end - start).days
            days = max(1, min(days, self.max_lookback_days))
            return f"{days} D"

        return "6 M"

    # ---------- Public API ----------

    def get_history(
        self,
        symbols: Sequence[str],
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        period: str | None = "6mo",
        bar_size: BarSize = "1d",
        what_to_show: WhatToShow = "TRADES",
        use_rth: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()

        norm_syms = normalize_symbols(symbols)
        frames: list[pd.DataFrame] = []

        for sym in norm_syms:
            contract = self._build_contract(sym)
            duration_str = self._to_ib_duration(period=period, start=start, end=end)
            bar_size_str = self._to_ib_bar_size(bar_size)

            logger.info(
                "IBKRProvider.get_history: symbol=%s duration=%s bar_size=%s what=%s",
                sym,
                duration_str,
                bar_size_str,
                what_to_show,
            )

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration_str,
                barSizeSetting=bar_size_str,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
                keepUpToDate=False,
            )

            if not bars:
                logger.warning("IBKRProvider: no data for %s", sym)
                if not self.ignore_empty:
                    # ׳ ׳—׳–׳™׳¨ ׳©׳•׳¨׳” ׳¨׳™׳§׳” ׳¨׳§ ׳¢׳ ׳¡׳׳
                    frames.append(
                        pd.DataFrame(
                            {
                                "symbol": [sym],
                                "datetime": [pd.NaT],
                                "open": [pd.NA],
                                "high": [pd.NA],
                                "low": [pd.NA],
                                "close": [pd.NA],
                                "volume": [pd.NA],
                            }
                        )
                    )
                continue

            df = ib_util.df(bars)  # type: ignore[arg-type]
            if df.empty:
                continue

            df["symbol"] = sym
            df = df.rename(
                columns={
                    "date": "datetime",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )

            cols = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
            # ׳©׳׳™׳¨׳” ׳¢׳ ׳¡׳“׳¨ ׳§׳‘׳•׳¢, ׳’׳ ׳׳ ׳™׳© ׳¢׳׳•׳“׳•׳× ׳ ׳•׳¡׳₪׳•׳×
            df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]

            frames.append(df)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        # ׳•׳“׳ ׳© datetime ׳‘׳₪׳•׳¨׳׳˜ ׳“׳™׳™׳˜-׳˜׳™׳™׳
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], utc=False)

        return out

    def healthcheck(self) -> bool:
        try:
            # ׳ ׳™׳¡׳™׳•׳ ׳§׳˜׳ ׳׳§׳‘׳ server time
            _ = self.ib.reqCurrentTime()
            return True
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("IBKRProvider.healthcheck failed: %s", exc)
            return False

    # ---------- Helpers ----------

    def _build_contract(self, symbol: str):
        cfg = self.contract_config
        if cfg.sec_type == "STK":
            c = Stock(
                symbol,
                cfg.exchange,
                cfg.currency,
                primaryExchange=cfg.primary_exchange,
            )
        elif cfg.sec_type == "FUT":
            # future example ג€“ ׳¦׳¨׳™׳ ׳׳”׳×׳׳™׳ ׳ײ¾symbol/expiry ׳©׳׳
            c = Future(symbol, cfg.exchange, cfg.currency)
        else:
            raise ValueError(f"Unsupported sec_type for IBKR: {cfg.sec_type}")
        return c

    def _to_ib_bar_size(self, bar_size: BarSize) -> str:
        mapping: dict[BarSize, str] = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day",
            "1w": "1 week",
        }
        return mapping.get(bar_size, "1 day")

    def _to_ib_duration(
        self,
        *,
        period: str | None,
        start: str | datetime | None,
        end: str | datetime | None,
    ) -> str:
        """
        Convert (period / start/end) into IB durationStr.

        Simplified logic:
        -----------------
        - If explicit start/end given, prefer a days-based duration with cap.
        - Else use `period` like '6mo' ג†’ '6 M', '1y' ג†’ '1 Y', etc.
        """
        # TODO: ׳׳₪׳©׳¨ ׳׳”׳¨׳—׳™׳‘ ׳₪׳” ׳׳•׳’׳™׳§׳” ׳׳“׳•׳™׳™׳§׳× ׳™׳•׳×׳¨ ׳׳₪׳™ ׳”׳¦׳•׳¨׳
        if period:
            p = period.lower()
            if p.endswith("mo"):
                return f"{p[:-2]} M"
            if p.endswith("y"):
                return f"{p[:-1]} Y"
            if p.endswith("d"):
                return f"{p[:-1]} D"
            if p.endswith("w"):
                return f"{int(p[:-1]) * 7} D"
            # default fallback
            return "6 M"

        # ׳׳ ׳׳™׳ period ׳׳‘׳ ׳™׳© start/end ג€“ estimated days
        if isinstance(start, datetime) and isinstance(end, datetime):
            days = (end - start).days
            days = max(1, min(days, self.max_lookback_days))
            return f"{days} D"

        # last resort
        return "6 M"


# ========================= Yahoo Finance Provider =================

try:
    import yfinance as yf  # type: ignore[import]
except Exception:  # pragma: no cover - optional dep
    yf = None
    logger.warning("yfinance is not available ג€” YahooProvider will be disabled.")


class YahooProvider(MarketDataProvider):
    """
    Yahoo Finance provider via `yfinance`.

    Use cases:
    ----------
    - Secondary / fallback provider.
    - Quick prototypes when IBKR is unavailable.
    """

    name: str = "yahoo"
    priority: int = 50  # ׳ ׳׳•׳ ׳™׳•׳×׳¨ ׳ײ¾IBKR, ׳’׳‘׳•׳” ׳׳¡׳₪׳§׳™׳ "׳©׳•׳׳™׳™׳"

    def __init__(
        self,
        *,
        auto_adjust: bool = False,
        session: Any | None = None,
    ) -> None:
        if yf is None:
            raise RuntimeError(
                "YahooProvider cannot be used because yfinance is not installed."
            )
        self.auto_adjust = auto_adjust
        self.session = session

    def get_history(
        self,
        symbols: Sequence[str],
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        period: str | None = "6mo",
        bar_size: BarSize = "1d",
        what_to_show: WhatToShow = "TRADES",
        use_rth: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()

        tickers = normalize_symbols(symbols)
        if not tickers:
            return pd.DataFrame()

        # yfinance ׳׳ ׳×׳•׳׳ ׳‘׳‘׳¨ ׳’׳“׳׳™׳ ׳›׳׳• IB, ׳׳– ׳ ׳×׳¢׳׳ ׳ײ¾bar_size ׳›׳׳ (׳׳• ׳ ׳׳₪׳” ׳ײ¾interval ׳‘׳”׳¨׳—׳‘׳” ׳¢׳×׳™׳“׳™׳×)
        if start is not None or end is not None:
            df = yf.download(
                " ".join(tickers),
                start=start,
                end=end,
                auto_adjust=self.auto_adjust,
                progress=False,
                session=self.session,
            )
        else:
            df = yf.download(
                " ".join(tickers),
                period=period or "6mo",
                auto_adjust=self.auto_adjust,
                progress=False,
                session=self.session,
            )

        if df is None or df.empty:
            logger.warning("YahooProvider: empty data for %s", tickers)
            return pd.DataFrame()

        # ׳׳ ׳™׳© ׳›׳׳” ׳˜׳™׳§׳¨׳™׳ ג€“ yfinance ׳׳—׳–׳™׳¨ MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            frames: list[pd.DataFrame] = []
            for sym in tickers:
                if (sym, "Close") not in df.columns:
                    continue
                sub = df.xs(sym, axis=1, level=1, drop_level=False)
                # columns like ('Open','QQQ') ג†’ ׳ ׳™׳§׳•׳™ ׳׳©׳׳•׳× ׳¡׳˜׳ ׳“׳¨׳˜׳™׳™׳
                sub = sub.droplevel(1, axis=1)
                sub = sub.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "adj_close",
                        "Volume": "volume",
                    }
                )
                sub = sub.reset_index().rename(columns={"Date": "datetime"})
                sub["symbol"] = sym
                frames.append(sub)
            if not frames:
                return pd.DataFrame()
            out = pd.concat(frames, ignore_index=True)
        else:
            # Single ticker
            sym = tickers[0]
            out = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            ).reset_index().rename(columns={"Date": "datetime"})
            out["symbol"] = sym

        # ׳©׳׳™׳¨׳” ׳¢׳ ׳₪׳•׳¨׳׳˜ ׳׳—׳™׳“
        cols = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
        out = out[[c for c in cols if c in out.columns] + [c for c in out.columns if c not in cols]]

        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"], utc=False)

        return out

    def healthcheck(self) -> bool:
        # ׳׳₪׳©׳¨ ׳׳¢׳©׳•׳× ping ׳§׳¦׳¨ ׳¢׳ ׳˜׳™׳§׳¨ ׳™׳“׳•׳¢, ׳׳‘׳ ׳–׳” ׳¢׳׳•׳ ׳׳”׳™׳•׳× ׳׳™׳˜׳™/׳׳™׳•׳×׳¨
        return True


# ========================= Placeholders for pro feeds ============

class PolygonProvider(MarketDataProvider):
    """
    Placeholder for Polygon.io integration.

    Intended usage:
    ---------------
    - High-quality intraday / tick data (US equities, options, crypto).
    - Requires API key & proper account.

    Currently this is only a skeleton; implement when you connect Polygon.
    """

    name: str = "polygon"
    priority: int = 30  # ׳ ׳׳•׳ ׳ײ¾IBKR, ׳’׳‘׳•׳” ׳ײ¾Yahoo (׳׳₪׳™ ׳”׳”׳¢׳“׳₪׳” ׳©׳׳)

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        # TODO: build client here

    def get_history(
        self,
        symbols: Sequence[str],
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        period: str | None = "6mo",
        bar_size: BarSize = "1d",
        what_to_show: WhatToShow = "TRADES",
        use_rth: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "PolygonProvider.get_history is not implemented yet. "
            "Hook this up when you add Polygon to your infra."
        )


class TiingoProvider(MarketDataProvider):
    """
    Placeholder for Tiingo integration.

    Tiingo offers:
    --------------
    - Daily prices for equities / ETFs.
    - Fundamental data.
    - Crypto, FX (depends on plan).

    Add actual implementation when you have a Tiingo API key.
    """

    name: str = "tiingo"
    priority: int = 40

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        # TODO: init client

    def get_history(
        self,
        symbols: Sequence[str],
        *,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        period: str | None = "6mo",
        bar_size: BarSize = "1d",
        what_to_show: WhatToShow = "TRADES",
        use_rth: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "TiingoProvider.get_history is not implemented yet. "
            "Implement this when Tiingo is added as a data source."
        )


# ========================= Factory helper ========================

def available_providers_summary() -> pd.DataFrame:
    """
    Utility: return a small DataFrame describing which providers are available
    in the current runtime (based on installed deps).

    This ׳™׳›׳•׳ ׳׳”׳™׳•׳× ׳©׳™׳׳•׳©׳™ ׳׳‘׳“׳™׳§׳•׳× / ׳˜׳׳‘ ׳§׳•׳ ׳₪׳™׳’ ׳‘׳“׳©׳‘׳•׳¨׳“.
    """
    rows: list[dict[str, Any]] = []

    rows.append(
        {
            "name": "ibkr",
            "class": "IBKRProvider",
            "available": IB is not None and ib_util is not None,
            "priority_default": IBKRProvider.priority,
            "notes": "Requires ib_insync and active IB connection",
        }
    )

    rows.append(
        {
            "name": "yahoo",
            "class": "YahooProvider",
            "available": yf is not None,
            "priority_default": YahooProvider.priority,
            "notes": "Requires yfinance; public Yahoo data (no guarantees)",
        }
    )

    rows.append(
        {
            "name": "polygon",
            "class": "PolygonProvider",
            "available": False,
            "priority_default": PolygonProvider.priority,
            "notes": "Skeleton only ג€” implement when Polygon API is configured",
        }
    )

    rows.append(
        {
            "name": "tiingo",
            "class": "TiingoProvider",
            "available": False,
            "priority_default": TiingoProvider.priority,
            "notes": "Skeleton only ג€” implement when Tiingo API is configured",
        }
    )

    return pd.DataFrame(rows)
