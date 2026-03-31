# -*- coding: utf-8 -*-
"""
core/fmp_macro.py — FMP-powered Macro Data & Calendar
=====================================================

Auto-fetches macro indicators, economic calendar, treasury rates,
and sector performance from FMP. Integrates with existing MacroDataClient
via the 'fmp:' URI scheme.

Features:
- Economic calendar with event proximity scoring
- FOMC/CPI/NFP blackout detection
- Treasury yield curve analysis
- VIX/credit/dollar macro features for regime detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common.fmp_client import FMPClient, get_fmp_client

logger = logging.getLogger("core.fmp_macro")

# ── Economic indicators available via FMP ─────────────────────────

FMP_MACRO_INDICATORS = {
    # identifier → (FMP indicator name, description)
    "GDP_USA": ("GDP", "US GDP"),
    "CPI_USA": ("CPI", "US Consumer Price Index"),
    "UNEMPLOYMENT": ("unemploymentRate", "US Unemployment Rate"),
    "FEDERAL_FUNDS_RATE": ("federalFundsRate", "Federal Funds Rate"),
    "INFLATION_RATE": ("inflationRate", "US Inflation Rate"),
    "RETAIL_SALES": ("retailSales", "Retail Sales"),
    "INDUSTRIAL_PRODUCTION": ("industrialProductionTotalIndex", "Industrial Production"),
    "NONFARM_PAYROLLS": ("nonFarmPayrolls", "Nonfarm Payrolls"),
    "CONSUMER_SENTIMENT": ("consumerSentiment", "Consumer Sentiment"),
    "HOUSING_STARTS": ("housingStarts", "Housing Starts"),
    "DURABLE_GOODS": ("durableGoods", "Durable Goods Orders"),
}

# Tickers for market-based macro proxies (fetched via prices)
FMP_MACRO_TICKERS = {
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "GOLD": "GC=F",
    "OIL": "CL=F",
    "US10Y": "^TNX",
    "US2Y": "^IRX",
    "HYG": "HYG",
    "LQD": "LQD",
    "TLT": "TLT",
    "SPY": "SPY",
}


# ── dataclasses ───────────────────────────────────────────────────

@dataclass
class MacroEvent:
    """Single economic calendar event."""
    date: datetime
    event: str
    country: str
    actual: float | None = None
    estimate: float | None = None
    previous: float | None = None
    impact: str = "medium"  # low, medium, high
    surprise: float | None = None  # actual - estimate

    @property
    def is_high_impact(self) -> bool:
        return self.impact.lower() == "high"


@dataclass
class EventProximity:
    """Proximity features for macro events."""
    days_to_next: int = 999
    days_since_last: int = 999
    next_event: str = ""
    in_blackout: bool = False
    in_aftermath: bool = False
    event_density_7d: int = 0
    proximity_score: float = 0.0  # exponential decay, 3-day half-life


@dataclass
class YieldCurveSnapshot:
    """Treasury yield curve data."""
    date: datetime | None = None
    us_1m: float = np.nan
    us_3m: float = np.nan
    us_6m: float = np.nan
    us_1y: float = np.nan
    us_2y: float = np.nan
    us_5y: float = np.nan
    us_10y: float = np.nan
    us_30y: float = np.nan
    spread_2s10s: float = np.nan
    spread_3m10y: float = np.nan
    is_inverted: bool = False


@dataclass
class MacroSnapshot:
    """Complete macro environment snapshot."""
    timestamp: datetime | None = None
    yield_curve: YieldCurveSnapshot | None = None
    events: list[MacroEvent] = field(default_factory=list)
    proximity: EventProximity | None = None
    indicators: dict[str, float] = field(default_factory=dict)
    regime_features: dict[str, float] = field(default_factory=dict)


# ── engine ────────────────────────────────────────────────────────

class FMPMacroEngine:
    """Macro data engine powered by FMP API."""

    def __init__(self, client: FMPClient | None = None):
        self.client = client or get_fmp_client()

    # ── economic calendar ─────────────────────────────────────────

    def get_economic_calendar(
        self,
        start: str | None = None,
        end: str | None = None,
        country: str = "US",
    ) -> list[MacroEvent]:
        """Fetch economic calendar events."""
        if not start:
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end:
            end = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        df = self.client.get_economic_calendar(start=start, end=end)
        if df.empty:
            return []

        events = []
        for _, row in df.iterrows():
            if country and str(row.get("country", "")).upper() != country.upper():
                continue
            actual = row.get("actual")
            estimate = row.get("estimate")
            surprise = None
            if actual is not None and estimate is not None:
                try:
                    surprise = float(actual) - float(estimate)
                except (ValueError, TypeError):
                    pass

            events.append(MacroEvent(
                date=pd.to_datetime(row.get("date", "")),
                event=str(row.get("event", "")),
                country=str(row.get("country", "")),
                actual=_safe_num(actual),
                estimate=_safe_num(estimate),
                previous=_safe_num(row.get("previous")),
                impact=str(row.get("impact", "medium")).lower(),
                surprise=surprise,
            ))

        return sorted(events, key=lambda e: e.date)

    def compute_event_proximity(
        self,
        as_of: datetime | None = None,
        events: list[MacroEvent] | None = None,
    ) -> EventProximity:
        """Compute event proximity features for signal adjustment."""
        now = as_of or datetime.now()
        if events is None:
            events = self.get_economic_calendar()

        high_impact = [e for e in events if e.is_high_impact]
        if not high_impact:
            high_impact = events  # fallback to all events

        prox = EventProximity()

        # upcoming
        future = [e for e in high_impact if e.date > now]
        past = [e for e in high_impact if e.date <= now]

        if future:
            next_evt = future[0]
            prox.days_to_next = max(0, (next_evt.date - now).days)
            prox.next_event = next_evt.event
            prox.in_blackout = prox.days_to_next <= 2

        if past:
            last_evt = past[-1]
            prox.days_since_last = max(0, (now - last_evt.date).days)
            prox.in_aftermath = prox.days_since_last <= 2

        # event density in 7-day window
        window_start = now - timedelta(days=3)
        window_end = now + timedelta(days=4)
        prox.event_density_7d = sum(
            1 for e in high_impact if window_start <= e.date <= window_end
        )

        # proximity score: exponential decay with 3-day half-life
        if prox.days_to_next < 999:
            half_life = 3.0
            prox.proximity_score = np.exp(-0.693 * prox.days_to_next / half_life)

        return prox

    # ── yield curve ───────────────────────────────────────────────

    def get_yield_curve(self) -> YieldCurveSnapshot:
        """Fetch current yield curve from treasury rates."""
        try:
            df = self.client.get_treasury_rates(start="2025-01-01")
        except Exception as e:
            logger.warning("Failed to fetch treasury rates: %s", e)
            return YieldCurveSnapshot()

        if df.empty:
            return YieldCurveSnapshot()

        # take most recent row
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date", ascending=False)

        row = df.iloc[0]
        snap = YieldCurveSnapshot(
            date=pd.to_datetime(row.get("date")) if "date" in row else None,
            us_1m=_safe_num(row.get("month1")),
            us_3m=_safe_num(row.get("month3")),
            us_6m=_safe_num(row.get("month6")),
            us_1y=_safe_num(row.get("year1")),
            us_2y=_safe_num(row.get("year2")),
            us_5y=_safe_num(row.get("year5")),
            us_10y=_safe_num(row.get("year10")),
            us_30y=_safe_num(row.get("year30")),
        )

        if not np.isnan(snap.us_2y) and not np.isnan(snap.us_10y):
            snap.spread_2s10s = snap.us_10y - snap.us_2y
            snap.is_inverted = snap.spread_2s10s < 0

        if not np.isnan(snap.us_3m) and not np.isnan(snap.us_10y):
            snap.spread_3m10y = snap.us_10y - snap.us_3m

        return snap

    # ── macro price series ────────────────────────────────────────

    def get_macro_price_series(
        self,
        indicators: Sequence[str] | None = None,
        lookback_years: int = 5,
    ) -> pd.DataFrame:
        """Fetch price-based macro series (VIX, DXY, yields, credit, etc.)."""
        if indicators is None:
            indicators = list(FMP_MACRO_TICKERS.keys())

        tickers = [FMP_MACRO_TICKERS[ind] for ind in indicators if ind in FMP_MACRO_TICKERS]
        if not tickers:
            return pd.DataFrame()

        start = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
        frames = {}

        for ind in indicators:
            ticker = FMP_MACRO_TICKERS.get(ind)
            if not ticker:
                continue
            try:
                df = self.client.get_historical_prices(ticker, start=start)
                if not df.empty and "close" in df.columns:
                    sub = df[["datetime", "close"]].copy()
                    sub = sub.rename(columns={"close": ind})
                    sub = sub.set_index("datetime")
                    frames[ind] = sub[ind]
            except Exception as e:
                logger.warning("Failed to fetch macro series %s (%s): %s", ind, ticker, e)

        if not frames:
            return pd.DataFrame()

        result = pd.DataFrame(frames)
        result.index = pd.to_datetime(result.index)
        return result.sort_index()

    # ── economic indicators ───────────────────────────────────────

    def get_economic_indicators(
        self,
        indicators: Sequence[str] | None = None,
        start: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch FMP economic indicators."""
        if indicators is None:
            indicators = list(FMP_MACRO_INDICATORS.keys())

        if not start:
            start = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

        results = {}
        for ind_id in indicators:
            fmp_name = FMP_MACRO_INDICATORS.get(ind_id, (ind_id,))[0]
            try:
                df = self.client.get_economic_indicator(fmp_name, start=start)
                if not df.empty:
                    results[ind_id] = df
            except Exception as e:
                logger.warning("Failed to fetch indicator %s: %s", ind_id, e)

        return results

    # ── regime features ───────────────────────────────────────────

    def compute_regime_features(
        self,
        macro_df: pd.DataFrame | None = None,
        lookback: int = 60,
    ) -> dict[str, float]:
        """Compute regime detection features from macro data."""
        if macro_df is None:
            macro_df = self.get_macro_price_series()

        if macro_df.empty:
            return {}

        features = {}
        tail = macro_df.tail(lookback)

        # VIX features
        if "VIX" in tail.columns:
            vix = tail["VIX"].dropna()
            if len(vix) > 10:
                features["vix_level"] = vix.iloc[-1]
                features["vix_zscore"] = (
                    (vix.iloc[-1] - vix.mean()) / max(vix.std(), 0.01)
                )
                features["vix_5d_change"] = vix.iloc[-1] - vix.iloc[-5] if len(vix) > 5 else 0
                features["vix_percentile"] = (vix < vix.iloc[-1]).mean()

        # credit spread (HYG - LQD)
        if "HYG" in tail.columns and "LQD" in tail.columns:
            hyg = tail["HYG"].dropna()
            lqd = tail["LQD"].dropna()
            if len(hyg) > 10 and len(lqd) > 10:
                spread = (lqd / hyg).dropna()
                if len(spread) > 10:
                    features["credit_spread_level"] = spread.iloc[-1]
                    features["credit_spread_zscore"] = (
                        (spread.iloc[-1] - spread.mean()) / max(spread.std(), 0.001)
                    )

        # dollar strength
        if "DXY" in tail.columns:
            dxy = tail["DXY"].dropna()
            if len(dxy) > 20:
                features["dxy_momentum_20d"] = (dxy.iloc[-1] / dxy.iloc[-20] - 1) * 100
                features["dxy_zscore"] = (
                    (dxy.iloc[-1] - dxy.mean()) / max(dxy.std(), 0.01)
                )

        # yield curve
        if "US10Y" in tail.columns and "US2Y" in tail.columns:
            t10 = tail["US10Y"].dropna()
            t2 = tail["US2Y"].dropna()
            if len(t10) > 5 and len(t2) > 5:
                spread = t10.iloc[-1] - t2.iloc[-1]
                features["yield_curve_2s10s"] = spread
                features["yield_curve_inverted"] = float(spread < 0)

        # equity momentum
        if "SPY" in tail.columns:
            spy = tail["SPY"].dropna()
            if len(spy) > 20:
                features["spy_momentum_20d"] = (spy.iloc[-1] / spy.iloc[-20] - 1) * 100
                features["spy_drawdown"] = (spy.iloc[-1] / spy.max() - 1) * 100

        return features

    # ── full snapshot ─────────────────────────────────────────────

    def get_snapshot(self) -> MacroSnapshot:
        """Get complete macro environment snapshot."""
        snap = MacroSnapshot(timestamp=datetime.now())

        try:
            snap.yield_curve = self.get_yield_curve()
        except Exception as e:
            logger.warning("Yield curve fetch failed: %s", e)

        try:
            snap.events = self.get_economic_calendar()
            snap.proximity = self.compute_event_proximity(events=snap.events)
        except Exception as e:
            logger.warning("Calendar fetch failed: %s", e)

        try:
            macro_df = self.get_macro_price_series(lookback_years=1)
            snap.regime_features = self.compute_regime_features(macro_df)
        except Exception as e:
            logger.warning("Regime features failed: %s", e)

        return snap


# ── helpers ───────────────────────────────────────────────────────

def _safe_num(val: Any) -> float:
    if val is None:
        return np.nan
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (ValueError, TypeError):
        return np.nan
