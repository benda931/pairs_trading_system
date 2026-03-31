# -*- coding: utf-8 -*-
"""
research/universe.py — Universe Construction Layer
===================================================

Implements:
  1. InstrumentRegistry — stores and queries instrument metadata
  2. EligibilityFilter — applies eligibility rules to an instrument set
  3. UniverseBuilder — constructs a UniverseSnapshot from prices + metadata
  4. BuiltinUniverses — pre-defined research universes (sector ETFs, SP100, etc.)
  5. UniverseAnalytics — composition and diagnostic reports

Design doctrine:
  - Universe construction is always reproducible: the same config + prices
    applied at the same train_end date produces the same snapshot.
  - Eligibility is transparent: every exclusion is logged with a typed reason.
  - Universe composition is tracked at every snapshot: sector, asset class,
    region, and liquidity-tier breakdowns.
  - Universe segmentation is used to constrain search-space (pairs within
    same sector first, cross-sector with justification).

Usage:
    builder = UniverseBuilder()
    snapshot = builder.build(
        definition=BuiltinUniverses.sp_sector_etfs(),
        prices=prices_df,
        train_end=datetime(2024, 1, 1),
    )
    print(snapshot.composition_summary())
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from research.discovery_contracts import (
    EligibilityDecision,
    InstrumentMetadata,
    RejectionReason,
    UniverseDefinition,
    UniverseSnapshot,
)

logger = logging.getLogger("research.universe")


# ── Built-in universe definitions ─────────────────────────────────

class BuiltinUniverses:
    """
    Pre-defined research universe definitions.

    These are seed lists and metadata for common research universes.
    The actual eligible set is always determined at runtime by applying
    eligibility filters to the available price data.
    """

    @staticmethod
    def sp_sector_etfs() -> UniverseDefinition:
        """US equity sector ETFs (SPDR / iShares families)."""
        symbols = [
            "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE",
            "IYW", "IYF", "IYH", "IYE", "IYJ", "IYC", "IYK", "IYM", "IDU",
            "VGT", "VFH", "VHT", "VDE", "VIS", "VCR", "VDC", "VAW", "VPU",
            "FNCL", "FHLC", "FENY",
        ]
        return UniverseDefinition(
            name="sp_sector_etfs",
            description="US equity sector ETFs (SPDR, iShares, Vanguard families)",
            symbols=symbols,
            min_history_days=252,
            min_dollar_volume=1_000_000,
            min_price=5.0,
            max_missing_data_pct=0.03,
            allowed_asset_classes=["etf"],
        )

    @staticmethod
    def us_large_cap_equities() -> UniverseDefinition:
        """Illustrative large-cap equity universe (subset)."""
        # Representative large-cap equities across sectors
        symbols = [
            # Technology
            "AAPL", "MSFT", "NVDA", "GOOGL", "META", "ORCL", "CSCO", "IBM", "INTC", "AMD",
            "QCOM", "TXN", "AMAT", "MU", "NOW", "CRM", "ADBE", "INTU", "AVGO", "KLAC",
            # Financials
            "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "BK",
            "USB", "PNC", "TFC", "FITB", "COF", "DFS", "SYF",
            # Healthcare
            "JNJ", "UNH", "PFE", "ABBV", "LLY", "BMY", "MRK", "CVS", "CI", "AMGN",
            "GILD", "BIIB", "VRTX", "REGN", "ISRG", "ZTS", "DXCM",
            # Consumer
            "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG", "ABNB",
            "PG", "KO", "PEP", "WMT", "COST", "TGT", "CL",
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "OXY",
            # Industrials
            "RTX", "HON", "UPS", "CAT", "GE", "MMM", "BA", "LMT", "NOC", "DE",
            "EMR", "ETN", "CARR",
            # Utilities
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL",
        ]
        return UniverseDefinition(
            name="us_large_cap",
            description="US large-cap equities (~100 names across sectors)",
            symbols=symbols,
            min_history_days=504,
            min_dollar_volume=50_000_000,
            min_price=5.0,
            max_missing_data_pct=0.02,
        )

    @staticmethod
    def us_factor_etfs() -> UniverseDefinition:
        """Smart-beta / factor ETFs."""
        symbols = [
            "IWD", "IWF", "IWM", "IWB",           # iShares value/growth/small/large
            "MTUM", "QUAL", "VLUE", "SIZE",         # iShares factor
            "USMV", "EFAV",                          # min vol
            "VTV", "VUG", "VBR", "VBK",            # Vanguard value/growth
            "SPHQ", "SPLV", "XMMO",                 # other factor
            "QQQ", "SPY", "IVV", "VOO",             # core market
            "DIA", "RSP",                            # equal weight / Dow
        ]
        return UniverseDefinition(
            name="us_factor_etfs",
            description="US smart-beta and factor ETFs",
            symbols=symbols,
            min_history_days=252,
            min_dollar_volume=5_000_000,
            min_price=5.0,
            max_missing_data_pct=0.03,
            allowed_asset_classes=["etf"],
        )

    @staticmethod
    def commodity_etfs() -> UniverseDefinition:
        """Commodity and commodity-producer ETFs."""
        symbols = [
            "GLD", "SLV", "IAU", "PPLT",            # precious metals
            "USO", "BNO", "DBO",                    # crude oil
            "UNG", "GAZ",                            # natural gas
            "DBA", "CORN", "WEAT", "SOYB",          # agriculture
            "PDBC", "GSG", "DJP", "COMT",           # broad commodity
            "COPX", "PICK", "XME",                  # metals producers
            "XOP", "OIH", "DRIP",                   # energy producers
            "MOO", "SOIL",                           # agriculture producers
        ]
        return UniverseDefinition(
            name="commodity_etfs",
            description="Commodity and commodity-producer ETFs",
            symbols=symbols,
            min_history_days=252,
            min_dollar_volume=1_000_000,
            min_price=1.0,
            max_missing_data_pct=0.05,
            cross_sector_allowed=True,
        )

    @staticmethod
    def country_etfs() -> UniverseDefinition:
        """Country and regional ETFs."""
        symbols = [
            "EWJ", "EWZ", "EWG", "EWU", "EWC", "EWA",  # country
            "FXI", "MCHI", "KWEB",                       # China
            "EWY", "EWT", "EPHE",                        # Asia
            "EZU", "EWQ", "EWI", "EWP", "EWD",          # Europe
            "EEM", "VWO",                                  # broad EM
            "ILF", "EWW",                                  # LatAm
        ]
        return UniverseDefinition(
            name="country_etfs",
            description="Country and regional ETFs",
            symbols=symbols,
            min_history_days=504,
            min_dollar_volume=5_000_000,
            min_price=5.0,
            max_missing_data_pct=0.05,
            cross_sector_allowed=True,
        )

    @staticmethod
    def custom(
        name: str,
        symbols: list[str],
        description: str = "",
        **kwargs,
    ) -> UniverseDefinition:
        """Create a custom universe definition."""
        return UniverseDefinition(
            name=name,
            description=description,
            symbols=symbols,
            **kwargs,
        )


# ── Instrument registry ────────────────────────────────────────────

class InstrumentRegistry:
    """
    In-memory registry of instrument metadata.

    Stores metadata by symbol and supports:
    - batch ingestion from DataFrames or dicts
    - auto-inference of missing fields from symbol patterns
    - sector/industry lookup
    - eligibility status queries
    """

    # Hardcoded sector map for common ETFs and large-caps
    _SECTOR_ETF_MAP: dict[str, str] = {
        "XLK": "Technology", "XLF": "Financials", "XLV": "Healthcare",
        "XLE": "Energy", "XLI": "Industrials", "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples", "XLB": "Materials", "XLU": "Utilities",
        "XLRE": "Real Estate", "IYW": "Technology", "IYF": "Financials",
        "IYH": "Healthcare", "IYE": "Energy", "IYJ": "Industrials",
        "IYC": "Consumer Discretionary", "IYK": "Consumer Staples",
        "IYM": "Materials", "IDU": "Utilities",
        "VGT": "Technology", "VFH": "Financials", "VHT": "Healthcare",
        "VDE": "Energy", "VIS": "Industrials", "VCR": "Consumer Discretionary",
        "VDC": "Consumer Staples", "VAW": "Materials", "VPU": "Utilities",
    }

    _EQUITY_SECTOR_MAP: dict[str, str] = {
        # Technology
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
        "GOOGL": "Technology", "META": "Technology", "ORCL": "Technology",
        "CSCO": "Technology", "IBM": "Technology", "INTC": "Technology",
        "AMD": "Technology", "QCOM": "Technology", "TXN": "Technology",
        "AMAT": "Technology", "MU": "Technology", "NOW": "Technology",
        "CRM": "Technology", "ADBE": "Technology", "INTU": "Technology",
        "AVGO": "Technology", "KLAC": "Technology",
        # Financials
        "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
        "GS": "Financials", "MS": "Financials", "C": "Financials",
        "AXP": "Financials", "BLK": "Financials", "SCHW": "Financials",
        "BK": "Financials", "USB": "Financials", "PNC": "Financials",
        "TFC": "Financials", "COF": "Financials", "DFS": "Financials",
        # Healthcare
        "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
        "ABBV": "Healthcare", "LLY": "Healthcare", "BMY": "Healthcare",
        "MRK": "Healthcare", "CVS": "Healthcare", "CI": "Healthcare",
        "AMGN": "Healthcare", "GILD": "Healthcare", "BIIB": "Healthcare",
        "VRTX": "Healthcare", "REGN": "Healthcare", "ISRG": "Healthcare",
        "ZTS": "Healthcare", "DXCM": "Healthcare",
        # Consumer
        "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
        "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
        "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
        "TJX": "Consumer Discretionary", "LOW": "Consumer Discretionary",
        "BKNG": "Consumer Discretionary", "ABNB": "Consumer Discretionary",
        "PG": "Consumer Staples", "KO": "Consumer Staples",
        "PEP": "Consumer Staples", "WMT": "Consumer Staples",
        "COST": "Consumer Staples", "TGT": "Consumer Staples",
        "CL": "Consumer Staples",
        # Energy
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
        "EOG": "Energy", "MPC": "Energy", "VLO": "Energy", "PSX": "Energy",
        "OXY": "Energy",
        # Industrials
        "RTX": "Industrials", "HON": "Industrials", "UPS": "Industrials",
        "CAT": "Industrials", "GE": "Industrials", "MMM": "Industrials",
        "BA": "Industrials", "LMT": "Industrials", "NOC": "Industrials",
        "DE": "Industrials", "EMR": "Industrials", "ETN": "Industrials",
        # Utilities
        "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
        "D": "Utilities", "AEP": "Utilities", "EXC": "Utilities",
        "XEL": "Utilities",
        # Real Estate
        "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate",
        "EQIX": "Real Estate", "SPG": "Real Estate",
        # Materials
        "LIN": "Materials", "APD": "Materials", "FCX": "Materials",
        "NEM": "Materials", "DOW": "Materials",
    }

    def __init__(self):
        self._registry: dict[str, InstrumentMetadata] = {}

    def register(self, meta: InstrumentMetadata) -> None:
        self._registry[meta.symbol] = meta

    def get(self, symbol: str) -> Optional[InstrumentMetadata]:
        return self._registry.get(symbol)

    def get_or_infer(self, symbol: str) -> InstrumentMetadata:
        """Get existing metadata or auto-infer from symbol patterns."""
        if symbol in self._registry:
            return self._registry[symbol]
        meta = self._infer_metadata(symbol)
        self._registry[symbol] = meta
        return meta

    def _infer_metadata(self, symbol: str) -> InstrumentMetadata:
        """Best-effort metadata inference from symbol and known maps."""
        is_etf = symbol in self._SECTOR_ETF_MAP or symbol in {
            "SPY", "QQQ", "IWM", "IVV", "VOO", "GLD", "SLV", "USO",
            "EEM", "VWO", "TLT", "HYG", "LQD",
        }
        sector = (
            self._SECTOR_ETF_MAP.get(symbol)
            or self._EQUITY_SECTOR_MAP.get(symbol)
            or ("ETF" if is_etf else "")
        )
        return InstrumentMetadata(
            symbol=symbol,
            is_etf=is_etf,
            asset_class="etf" if is_etf else "equity",
            sector=sector,
            country="US",
            region="us",
        )

    def bulk_register_from_prices(
        self,
        prices: pd.DataFrame,
        train_end: Optional[datetime] = None,
    ) -> None:
        """
        Register metadata inferred from a prices DataFrame.

        Computes: history_days, missing_data_pct, avg_price, volatility.
        """
        cutoff = pd.Timestamp(train_end) if train_end else prices.index[-1]
        px = prices[prices.index <= cutoff]

        for sym in px.columns:
            series = px[sym].dropna()
            if len(series) == 0:
                continue

            full_len = len(px)
            history_days = len(series)
            missing_pct = 1.0 - history_days / max(full_len, 1)

            meta = self.get_or_infer(sym)
            meta.history_start = series.index[0].to_pydatetime()
            meta.history_end = series.index[-1].to_pydatetime()
            meta.history_days = history_days
            meta.missing_data_pct = missing_pct
            meta.avg_price = float(series.mean())

            log_returns = np.log(series.clip(lower=1e-8)).diff().dropna()
            if len(log_returns) > 10:
                meta.volatility_ann = float(log_returns.std() * np.sqrt(252))

            self._registry[sym] = meta

    def list_symbols(
        self,
        sector: Optional[str] = None,
        asset_class: Optional[str] = None,
        eligible_only: bool = False,
    ) -> list[str]:
        metas = self._registry.values()
        if sector:
            metas = [m for m in metas if m.sector == sector]
        if asset_class:
            metas = [m for m in metas if m.asset_class == asset_class]
        if eligible_only:
            metas = [m for m in metas if m.is_eligible]
        return [m.symbol for m in metas]

    def sector_map(self) -> dict[str, list[str]]:
        """Return {sector: [symbols]} for all registered instruments."""
        result: dict[str, list[str]] = defaultdict(list)
        for sym, meta in self._registry.items():
            result[meta.sector or "Unknown"].append(sym)
        return dict(result)


# ── Eligibility filter ─────────────────────────────────────────────

class EligibilityFilter:
    """
    Applies eligibility rules to instruments.

    Each rule is a separate check; failures are logged individually
    so the exclusion audit clearly shows WHY each instrument was excluded.

    R-004 — Survivorship Bias Mitigation:
    These eligibility filters collectively address survivorship bias risk by
    excluding instruments that are present in the data solely because they
    survived long enough to be included. Specifically:
      - min_history_days: prevents inclusion of instruments that only recently
        appeared (IPOs, spinoffs). Short histories are a marker of selection bias.
      - min_dollar_volume: prevents penny stock / illiquid name inclusion.
        Delisted stocks are typically illiquid before delisting — this filter
        partially captures pre-delisting deterioration.
      - min_price: explicitly excludes penny stocks. Stocks trading below
        threshold are near-delisting candidates and overrepresent distress.
      - max_missing_data_pct + stale_data: missing or frozen prices are a
        leading indicator of data vendor survivorship bias in historical feeds.
    RESIDUAL RISK: These filters are applied to the current universe snapshot.
    They do NOT retroactively remove stocks that were eligible in the past but
    have since delisted. For true point-in-time survivorship-bias-free data,
    a dedicated point-in-time database (e.g. CRSP) is required.
    """

    def __init__(self, definition: UniverseDefinition):
        self.definition = definition

    def evaluate(
        self,
        meta: InstrumentMetadata,
    ) -> EligibilityDecision:
        """
        Evaluate eligibility for one instrument.

        Returns EligibilityDecision with full reason logging.
        """
        rejections: list[RejectionReason] = []
        human_reasons: list[str] = []
        metrics: dict[str, float] = {}

        d = self.definition

        # Survivorship bias mitigation: exclude instruments with insufficient history.
        # Short-lived instruments (recent IPOs, spinoffs) create selection bias.
        # R-004: min_history_days must be >= 252 (1 trading year) to be meaningful.
        metrics["history_days"] = float(meta.history_days)
        if meta.history_days < d.min_history_days:
            rejections.append(RejectionReason.INSUFFICIENT_HISTORY)
            human_reasons.append(
                f"history_days={meta.history_days} < required={d.min_history_days}"
            )

        # Survivorship bias mitigation: exclude low-liquidity instruments.
        # Illiquid instruments are over-represented in historical data feeds because
        # vendors tend to include only stocks that maintained enough volume to be tracked.
        # Pre-delisting deterioration is partially captured by this ADV filter.
        if not np.isnan(meta.avg_dollar_volume):
            metrics["avg_dollar_volume"] = meta.avg_dollar_volume
            if meta.avg_dollar_volume < d.min_dollar_volume:
                rejections.append(RejectionReason.LOW_DOLLAR_VOLUME)
                human_reasons.append(
                    f"avg_dollar_volume=${meta.avg_dollar_volume:,.0f} < required=${d.min_dollar_volume:,.0f}"
                )

        # Survivorship bias mitigation: exclude penny stocks / near-delisting names.
        # Stocks trading below min_price threshold are high-distress candidates.
        # Excluding them prevents the strategy from "benefiting" from historical
        # data that was only available because the stock survived.
        if not np.isnan(meta.avg_price):
            metrics["avg_price"] = meta.avg_price
            if meta.avg_price < d.min_price:
                rejections.append(RejectionReason.LOW_PRICE)
                human_reasons.append(
                    f"avg_price={meta.avg_price:.2f} < required={d.min_price:.2f}"
                )

        # Missing data
        metrics["missing_data_pct"] = meta.missing_data_pct
        if meta.missing_data_pct > d.max_missing_data_pct:
            rejections.append(RejectionReason.EXCESSIVE_MISSING_DATA)
            human_reasons.append(
                f"missing_data={meta.missing_data_pct:.1%} > allowed={d.max_missing_data_pct:.1%}"
            )

        # Stale data
        if meta.has_stale_periods:
            rejections.append(RejectionReason.STALE_DATA)
            human_reasons.append("Instrument has stale/frozen price periods")

        # Price anomalies
        if meta.has_price_anomalies:
            rejections.append(RejectionReason.PRICE_ANOMALY)
            human_reasons.append("Instrument has detected price anomalies")

        # Sector filter
        if d.allowed_sectors and meta.sector not in d.allowed_sectors:
            rejections.append(RejectionReason.UNIVERSE_EXCLUDED)
            human_reasons.append(
                f"sector={meta.sector!r} not in allowed_sectors={d.allowed_sectors}"
            )

        # Excluded sectors
        if d.excluded_sectors and meta.sector in d.excluded_sectors:
            rejections.append(RejectionReason.UNIVERSE_EXCLUDED)
            human_reasons.append(f"sector={meta.sector!r} is excluded")

        # Asset class filter
        if d.allowed_asset_classes and meta.asset_class not in d.allowed_asset_classes:
            rejections.append(RejectionReason.UNIVERSE_EXCLUDED)
            human_reasons.append(
                f"asset_class={meta.asset_class!r} not in allowed={d.allowed_asset_classes}"
            )

        return EligibilityDecision(
            symbol=meta.symbol,
            eligible=len(rejections) == 0,
            rejection_reasons=rejections,
            human_reasons=human_reasons,
            metrics=metrics,
        )

    def filter_batch(
        self,
        metas: list[InstrumentMetadata],
    ) -> tuple[list[InstrumentMetadata], list[EligibilityDecision]]:
        """
        Apply eligibility filter to a list of instruments.

        Returns (eligible_instruments, all_decisions).
        """
        eligible = []
        decisions = []
        for meta in metas:
            dec = self.evaluate(meta)
            if dec.eligible:
                eligible.append(meta)
                meta.is_eligible = True
            else:
                meta.is_eligible = False
                meta.ineligibility_reasons = dec.rejection_reasons
                meta.ineligibility_notes = "; ".join(dec.human_reasons)
            decisions.append(dec)
        return eligible, decisions


# ── Universe builder ───────────────────────────────────────────────

class UniverseBuilder:
    """
    Builds a UniverseSnapshot from a UniverseDefinition + price data.

    The snapshot captures:
    - which instruments are eligible at train_end
    - why excluded instruments were excluded
    - universe composition (sector, asset class, liquidity tier)

    Always reproducible: same definition + prices + train_end → same snapshot.
    """

    def __init__(self, registry: Optional[InstrumentRegistry] = None):
        self.registry = registry or InstrumentRegistry()

    def build(
        self,
        definition: UniverseDefinition,
        prices: pd.DataFrame,
        *,
        train_end: Optional[datetime] = None,
    ) -> UniverseSnapshot:
        """
        Build a universe snapshot.

        Parameters
        ----------
        definition : UniverseDefinition
        prices : pd.DataFrame with symbol columns, DatetimeIndex
        train_end : Only use price data up to this date for eligibility checks

        Returns
        -------
        UniverseSnapshot
        """
        cutoff = pd.Timestamp(train_end) if train_end else prices.index[-1]
        px = prices[prices.index <= cutoff]

        # Register metadata from prices
        self.registry.bulk_register_from_prices(px, train_end=cutoff)

        # Focus on symbols in the definition
        symbols = [s for s in definition.symbols if s in px.columns]
        missing_from_prices = [s for s in definition.symbols if s not in px.columns]

        if missing_from_prices:
            logger.debug(
                "%s: %d symbols in definition not found in prices: %s...",
                definition.name,
                len(missing_from_prices),
                missing_from_prices[:5],
            )

        # Get metadata for all candidate symbols
        metas = [self.registry.get_or_infer(s) for s in symbols]

        # Apply data-quality checks via prices
        for meta in metas:
            series = px.get(meta.symbol)
            if series is None or series.dropna().__len__() == 0:
                meta.history_days = 0
                continue
            series_clean = series.dropna()
            meta.history_days = len(series_clean)
            meta.missing_data_pct = 1.0 - len(series_clean) / max(len(px), 1)
            meta.avg_price = float(series_clean.mean())
            log_ret = np.log(series_clean.clip(lower=1e-8)).diff().dropna()
            if len(log_ret) > 20:
                meta.volatility_ann = float(log_ret.std() * np.sqrt(252))
                # Price anomaly: detect large spikes (>10 sigma)
                z = (log_ret - log_ret.mean()) / (log_ret.std() + 1e-10)
                if (z.abs() > 10).any():
                    meta.has_price_anomalies = True

        # Apply eligibility filters
        ef = EligibilityFilter(definition)
        eligible_metas, decisions = ef.filter_batch(metas)

        eligible_symbols = [m.symbol for m in eligible_metas]
        excluded_symbols: dict[str, list[str]] = {}

        for m in metas:
            if not m.is_eligible:
                excluded_symbols[m.symbol] = [r.value for r in m.ineligibility_reasons]

        # Add symbols missing from prices as excluded
        for sym in missing_from_prices:
            excluded_symbols[sym] = [RejectionReason.INSUFFICIENT_HISTORY.value]

        # Compute composition
        sector_counts: dict[str, int] = defaultdict(int)
        asset_class_counts: dict[str, int] = defaultdict(int)
        region_counts: dict[str, int] = defaultdict(int)
        liquidity_tier_counts: dict[str, int] = defaultdict(int)

        for m in eligible_metas:
            sector_counts[m.sector or "Unknown"] += 1
            asset_class_counts[m.asset_class] += 1
            region_counts[m.region or "us"] += 1
            liquidity_tier_counts[m.liquidity_tier] += 1

        snapshot = UniverseSnapshot(
            universe_name=definition.name,
            snapshot_date=cutoff.to_pydatetime() if hasattr(cutoff, 'to_pydatetime') else cutoff,
            eligible_symbols=eligible_symbols,
            metadata={m.symbol: m for m in eligible_metas},
            excluded_symbols=excluded_symbols,
            sector_counts=dict(sector_counts),
            asset_class_counts=dict(asset_class_counts),
            region_counts=dict(region_counts),
            liquidity_tier_counts=dict(liquidity_tier_counts),
            definition_id=definition.universe_id,
        )

        logger.info(
            "Universe '%s': %d eligible, %d excluded (from %d candidates)",
            definition.name,
            snapshot.n_eligible,
            snapshot.n_excluded,
            len(definition.symbols),
        )

        return snapshot


# ── Universe analytics ─────────────────────────────────────────────

class UniverseAnalytics:
    """
    Diagnostic analytics for a universe snapshot or set of snapshots.

    Reports on:
    - Composition (sector/asset/region/liquidity diversification)
    - Coverage (how many symbols in definition have data?)
    - Exclusion breakdown (what are the most common reasons for exclusion?)
    - Historical evolution (if multiple snapshots provided)
    """

    @staticmethod
    def composition_report(snapshot: UniverseSnapshot) -> dict:
        """Full composition diagnostic for one snapshot."""
        total = snapshot.n_eligible
        if total == 0:
            return {"n_eligible": 0, "n_excluded": snapshot.n_excluded}

        return {
            "n_eligible": total,
            "n_excluded": snapshot.n_excluded,
            "coverage_rate": total / max(total + snapshot.n_excluded, 1),
            "snapshot_date": snapshot.snapshot_date.date().isoformat(),
            "sector_breakdown": {
                k: {"count": v, "pct": v / total}
                for k, v in sorted(snapshot.sector_counts.items(), key=lambda x: -x[1])
            },
            "asset_class_breakdown": {
                k: {"count": v, "pct": v / total}
                for k, v in sorted(snapshot.asset_class_counts.items(), key=lambda x: -x[1])
            },
            "liquidity_tier_breakdown": {
                k: {"count": v, "pct": v / total}
                for k, v in sorted(snapshot.liquidity_tier_counts.items(), key=lambda x: -x[1])
            },
            "exclusion_by_reason": snapshot.exclusion_breakdown(),
        }

    @staticmethod
    def liquidity_distribution(snapshot: UniverseSnapshot) -> dict:
        """ADV and price statistics for eligible instruments."""
        metas = list(snapshot.metadata.values())
        advs = [m.avg_dollar_volume for m in metas if not np.isnan(m.avg_dollar_volume)]
        vols = [m.volatility_ann for m in metas if not np.isnan(m.volatility_ann)]

        if not advs:
            return {}

        return {
            "adv_median": float(np.median(advs)),
            "adv_p25": float(np.percentile(advs, 25)),
            "adv_p75": float(np.percentile(advs, 75)),
            "vol_ann_median": float(np.median(vols)) if vols else np.nan,
            "vol_ann_p25": float(np.percentile(vols, 25)) if vols else np.nan,
            "vol_ann_p75": float(np.percentile(vols, 75)) if vols else np.nan,
        }

    @staticmethod
    def sector_pair_counts(snapshot: UniverseSnapshot) -> dict[str, int]:
        """
        Count how many within-sector pairs are possible from this universe.
        Useful for understanding search space by sector.
        """
        sector_map: dict[str, list[str]] = defaultdict(list)
        for sym in snapshot.eligible_symbols:
            meta = snapshot.metadata.get(sym)
            sector = meta.sector if meta else "Unknown"
            sector_map[sector].append(sym)

        return {
            sector: len(syms) * (len(syms) - 1) // 2
            for sector, syms in sector_map.items()
            if len(syms) >= 2
        }

    @staticmethod
    def search_space_size(snapshot: UniverseSnapshot) -> dict:
        """Estimate the search space for this universe."""
        n = snapshot.n_eligible
        total_pairs = n * (n - 1) // 2
        sector_pairs = sum(UniverseAnalytics.sector_pair_counts(snapshot).values())

        return {
            "n_eligible": n,
            "total_pairs_possible": total_pairs,
            "same_sector_pairs": sector_pairs,
            "cross_sector_pairs": total_pairs - sector_pairs,
            "sector_pair_pct": sector_pairs / max(total_pairs, 1),
        }
