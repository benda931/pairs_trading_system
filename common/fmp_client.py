# -*- coding: utf-8 -*-
"""
common/fmp_client.py — FMP (FinancialModelingPrep) Unified Client
=================================================================

Central HTTP client for all FMP API calls. Features:
- Exponential backoff with jitter on 429/5xx
- Fail-fast on 401/403
- Connection pooling via requests.Session
- TTL cache for repeated calls
- Thread-safe batch fetching via ThreadPoolExecutor
- Stable-first URL strategy with v3 fallback
"""

from __future__ import annotations

import logging
import os
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("fmp_client")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ── defaults ──────────────────────────────────────────────────────

def _load_fmp_key_from_config() -> str | None:
    """Read FMP_API_KEY from config.json in the project root."""
    try:
        root = Path(__file__).resolve().parent.parent
        cfg_path = root / "config.json"
        if cfg_path.exists():
            import json as _json
            with open(cfg_path, "r", encoding="utf-8") as fh:
                cfg = _json.load(fh)
            return cfg.get("FMP_API_KEY") or None
    except Exception:
        pass
    return None


def _load_fmp_key_from_env_file() -> str | None:
    """Read FMP_API_KEY from .env file in the project root."""
    try:
        root = Path(__file__).resolve().parent.parent
        env_path = root / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("FMP_API_KEY="):
                    return line.split("=", 1)[1].strip() or None
    except Exception:
        pass
    return None

_DEFAULT_API_KEY: str | None = (
    os.environ.get("FMP_API_KEY")
    or _load_fmp_key_from_env_file()
    or _load_fmp_key_from_config()
)
_BASE = "https://financialmodelingprep.com"
_STABLE = f"{_BASE}/stable"
_V3 = f"{_BASE}/api/v3"
_V4 = f"{_BASE}/api/v4"


# ── cache ─────────────────────────────────────────────────────────
@dataclass
class _CacheEntry:
    data: Any
    ts: float


class _TTLCache:
    def __init__(self, ttl: int = 1800, max_items: int = 512):
        self._store: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self.ttl = ttl
        self.max_items = max_items

    def get(self, key: str) -> Any:
        with self._lock:
            e = self._store.get(key)
            if e is None:
                return None
            if time.time() - e.ts > self.ttl:
                self._store.pop(key, None)
                return None
            return e.data

    def put(self, key: str, data: Any) -> None:
        with self._lock:
            if len(self._store) >= self.max_items:
                oldest = min(self._store, key=lambda k: self._store[k].ts)
                self._store.pop(oldest, None)
            self._store[key] = _CacheEntry(data=data, ts=time.time())

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# ── client ────────────────────────────────────────────────────────
class FMPClient:
    """Thread-safe FMP API client with retry, caching, and batch support."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_ttl: int = 1800,
        max_workers: int = 4,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.api_key = api_key or _DEFAULT_API_KEY
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._cache = _TTLCache(ttl=cache_ttl)

    # ── low-level request ─────────────────────────────────────────

    def _request(self, url: str, params: dict | None = None) -> Any:
        """HTTP GET with retry + backoff. Returns parsed JSON."""
        params = dict(params or {})
        params["apikey"] = self.api_key

        cache_key = f"{url}|{sorted(params.items())}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                if resp.status_code in (401, 403):
                    raise PermissionError(
                        f"FMP auth error {resp.status_code}: {resp.text[:200]}"
                    )
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = (2 ** attempt) + (np.random.random() * 0.5)
                    logger.warning(
                        "FMP %s (attempt %d/%d), waiting %.1fs",
                        resp.status_code, attempt + 1, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                self._cache.put(cache_key, data)
                return data
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                wait = (2 ** attempt) + (np.random.random() * 0.5)
                logger.warning("FMP connection error (attempt %d): %s", attempt + 1, e)
                time.sleep(wait)

        raise ConnectionError(
            f"FMP request failed after {self.max_retries} attempts: {last_exc}"
        )

    def _request_stable(
        self, endpoint: str, params: dict | None = None, *, fallback_v3: str | None = None
    ) -> Any:
        """Try stable API first, fall back to v3 if available."""
        try:
            return self._request(f"{_STABLE}/{endpoint}", params)
        except Exception as e:
            if fallback_v3:
                logger.info("Stable endpoint failed, trying v3 fallback: %s", e)
                return self._request(f"{_V3}/{fallback_v3}", params)
            raise

    # ── prices ────────────────────────────────────────────────────

    def get_historical_prices(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV for a single symbol."""
        params: dict[str, str] = {"symbol": symbol}
        if start:
            params["from"] = start
        if end:
            params["to"] = end

        data = self._request_stable(
            "historical-price-eod/full",
            params,
            fallback_v3=f"historical-price-full/{symbol}",
        )

        if not data:
            return pd.DataFrame()

        # stable returns list of dicts, v3 wraps in {"historical": [...]}
        rows = data if isinstance(data, list) else data.get("historical", data)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # normalize columns
        col_map = {}
        for target, candidates in {
            "datetime": ["date", "Date", "datetime"],
            "open": ["open", "Open"],
            "high": ["high", "High"],
            "low": ["low", "Low"],
            "close": ["close", "Close", "adjClose"],
            "volume": ["volume", "Volume"],
        }.items():
            for c in candidates:
                if c in df.columns:
                    col_map[c] = target
                    break

        df = df.rename(columns=col_map)
        df["symbol"] = symbol.upper()

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

        cols = ["symbol", "datetime", "open", "high", "low", "close", "volume"]
        return df[[c for c in cols if c in df.columns]]

    def get_batch_prices(
        self,
        symbols: Sequence[str],
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch prices for multiple symbols concurrently."""
        frames: list[pd.DataFrame] = []

        def _fetch(sym: str) -> pd.DataFrame:
            try:
                return self.get_historical_prices(sym, start=start, end=end)
            except Exception as e:
                logger.warning("FMP price fetch failed for %s: %s", sym, e)
                return pd.DataFrame()

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_fetch, s): s for s in symbols}
            for fut in as_completed(futures):
                df = fut.result()
                if not df.empty:
                    frames.append(df)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ── quotes (real-time / delayed) ──────────────────────────────

    def get_quotes(self, symbols: Sequence[str]) -> pd.DataFrame:
        """Batch quotes for multiple symbols."""
        chunk_size = 50
        frames = []
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            joined = ",".join(chunk)
            data = self._request_stable(
                f"batch-quote/{joined}",
                fallback_v3=f"quote/{joined}",
            )
            if data:
                frames.append(pd.DataFrame(data if isinstance(data, list) else [data]))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ── fundamentals ──────────────────────────────────────────────

    def get_ratios_ttm(self, symbol: str) -> dict:
        data = self._request_stable("ratios-ttm", {"symbol": symbol})
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else {}

    def get_key_metrics_ttm(self, symbol: str) -> dict:
        data = self._request_stable("key-metrics-ttm", {"symbol": symbol})
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else {}

    def get_enterprise_values(self, symbol: str, limit: int = 4) -> list[dict]:
        data = self._request_stable(
            "enterprise-values", {"symbol": symbol, "limit": str(limit)}
        )
        return data if isinstance(data, list) else []

    def get_financial_growth(self, symbol: str, limit: int = 4) -> list[dict]:
        data = self._request_stable(
            "income-statement-growth", {"symbol": symbol, "limit": str(limit)}
        )
        return data if isinstance(data, list) else []

    def get_analyst_estimates(self, symbol: str, limit: int = 4) -> list[dict]:
        # analyst-estimates requires period=annual/quarter param
        try:
            data = self._request_stable(
                "analyst-estimates",
                {"symbol": symbol, "limit": str(limit), "period": "annual"},
            )
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def get_earnings_surprises(self, symbol: str) -> list[dict]:
        """Fetch earnings data (actual vs estimate)."""
        data = self._request_stable("earnings", {"symbol": symbol})
        return data if isinstance(data, list) else []

    def get_cashflow_statement(self, symbol: str, limit: int = 4) -> list[dict]:
        data = self._request_stable(
            "cash-flow-statement", {"symbol": symbol, "limit": str(limit)}
        )
        return data if isinstance(data, list) else []

    def get_income_statement(self, symbol: str, limit: int = 4) -> list[dict]:
        data = self._request_stable(
            "income-statement", {"symbol": symbol, "limit": str(limit)}
        )
        return data if isinstance(data, list) else []

    def get_balance_sheet(self, symbol: str, limit: int = 4) -> list[dict]:
        data = self._request_stable(
            "balance-sheet-statement", {"symbol": symbol, "limit": str(limit)}
        )
        return data if isinstance(data, list) else []

    def get_extended_fundamentals(self, symbol: str) -> dict:
        """Fetch all fundamental data for a symbol in one call."""
        result: dict[str, Any] = {"symbol": symbol}

        def _safe(fn, key):
            try:
                result[key] = fn(symbol)
            except Exception as e:
                logger.warning("FMP %s failed for %s: %s", key, symbol, e)
                result[key] = {} if "ttm" in key else []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            pool.submit(_safe, self.get_ratios_ttm, "ratios_ttm")
            pool.submit(_safe, self.get_key_metrics_ttm, "key_metrics_ttm")
            pool.submit(_safe, self.get_enterprise_values, "enterprise_values")
            pool.submit(_safe, self.get_financial_growth, "financial_growth")
            pool.submit(_safe, self.get_analyst_estimates, "analyst_estimates")
            pool.submit(_safe, self.get_earnings_surprises, "earnings_surprises")
            pool.submit(_safe, self.get_cashflow_statement, "cashflow")
            pool.shutdown(wait=True)

        return result

    def get_batch_fundamentals(self, symbols: Sequence[str]) -> dict[str, dict]:
        """Fetch fundamentals for multiple symbols concurrently."""
        results: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self.get_extended_fundamentals, s): s for s in symbols
            }
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    results[sym] = fut.result()
                except Exception as e:
                    logger.warning("Batch fundamentals failed for %s: %s", sym, e)
        return results

    # ── macro / economic ──────────────────────────────────────────

    def get_economic_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch economic calendar events."""
        params: dict[str, str] = {}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        data = self._request_stable("economic-calendar", params, fallback_v3="economic_calendar")
        return pd.DataFrame(data) if data else pd.DataFrame()

    def get_treasury_rates(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch US treasury rates."""
        params: dict[str, str] = {}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        if not start:
            params["from"] = "2020-01-01"
        data = self._request_stable("treasury-rates", params)
        return pd.DataFrame(data) if data else pd.DataFrame()

    def get_economic_indicator(
        self,
        indicator: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch specific economic indicator.

        Extracts from the economic calendar events matching the indicator name.
        """
        # use economic calendar and filter by event name
        params: dict[str, str] = {}
        if start:
            params["from"] = start
        if end:
            params["to"] = end
        if not start:
            params["from"] = "2000-01-01"

        data = self._request_stable("economic-calendar", params, fallback_v3="economic_calendar")
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if df.empty:
            return df

        # filter by indicator name (case-insensitive partial match)
        indicator_lower = indicator.lower()
        mask = df["event"].str.lower().str.contains(indicator_lower, na=False)
        # also check "country" = US
        if "country" in df.columns:
            mask = mask & (df["country"].str.upper() == "US")

        filtered = df[mask].copy()
        if filtered.empty:
            return pd.DataFrame()

        if "date" in filtered.columns:
            filtered["date"] = pd.to_datetime(filtered["date"])
            filtered = filtered.sort_values("date").reset_index(drop=True)

        # try to extract numeric value
        if "actual" in filtered.columns:
            filtered["value"] = pd.to_numeric(filtered["actual"], errors="coerce")

        return filtered

    def get_sector_performance(self) -> pd.DataFrame:
        """Fetch sector performance data."""
        data = self._request_stable(
            "sector-performance", fallback_v3="sector-performance"
        )
        return pd.DataFrame(data) if data else pd.DataFrame()

    def get_stock_screener(self, **kwargs) -> pd.DataFrame:
        """Screener for finding tradeable pairs candidates."""
        data = self._request(f"{_V3}/stock-screener", kwargs)
        return pd.DataFrame(data) if data else pd.DataFrame()

    # ── ETF data ──────────────────────────────────────────────────

    def get_etf_holdings(self, symbol: str) -> pd.DataFrame:
        """Fetch ETF holdings/constituents."""
        data = self._request_stable(
            "etf/holdings",
            {"symbol": symbol},
            fallback_v3=f"etf-holder/{symbol}",
        )
        return pd.DataFrame(data) if data else pd.DataFrame()

    def get_index_constituents(self, index: str = "sp500") -> pd.DataFrame:
        """Fetch index constituents (sp500, nasdaq, dowjones)."""
        endpoint_map = {
            "sp500": "sp500_constituent",
            "nasdaq": "nasdaq_constituent",
            "dowjones": "dowjones_constituent",
        }
        ep = endpoint_map.get(index.lower(), f"{index}_constituent")
        data = self._request(f"{_V3}/{ep}")
        return pd.DataFrame(data) if data else pd.DataFrame()

    # ── sector / industry classification ──────────────────────────

    def get_stock_profile(self, symbol: str) -> dict:
        """Fetch company profile (sector, industry, market cap, etc.)."""
        data = self._request(f"{_V3}/profile/{symbol}")
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else {}

    def get_batch_profiles(self, symbols: Sequence[str]) -> pd.DataFrame:
        """Fetch profiles for multiple symbols."""
        chunk_size = 50
        frames = []
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            joined = ",".join(chunk)
            data = self._request(f"{_V3}/profile/{joined}")
            if data:
                frames.append(pd.DataFrame(data if isinstance(data, list) else [data]))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ── news / sentiment ──────────────────────────────────────────

    def get_stock_news(self, symbol: str, limit: int = 20) -> list[dict]:
        """Fetch recent news for a stock."""
        data = self._request(f"{_V3}/stock_news", {"tickers": symbol, "limit": str(limit)})
        return data if isinstance(data, list) else []

    # ── utility ───────────────────────────────────────────────────

    def healthcheck(self) -> bool:
        """Quick API health check."""
        try:
            data = self._request(f"{_V3}/quote/AAPL")
            return bool(data)
        except Exception:
            return False

    def clear_cache(self) -> None:
        self._cache.clear()


# ── singleton ─────────────────────────────────────────────────────
_default_client: FMPClient | None = None
_client_lock = threading.Lock()


def get_fmp_client(api_key: str | None = None) -> FMPClient:
    """Get or create the default FMP client singleton."""
    global _default_client
    with _client_lock:
        if _default_client is None:
            _default_client = FMPClient(api_key=api_key)
        return _default_client
