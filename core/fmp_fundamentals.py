# -*- coding: utf-8 -*-
"""
core/fmp_fundamentals.py — Fundamentals Engine powered by FMP
=============================================================

Fetches and processes fundamental data for pair constituents:
- Valuation multiples (PE, EV/EBITDA, P/B, FCF yield)
- Growth metrics (revenue, EPS, earnings revisions)
- Quality metrics (ROIC, margins, accrual ratio)
- Earnings surprises
- Analyst estimates

Produces a FundamentalProfile per symbol and a PairFundamentalComparison
that quantifies relative value between pair legs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from common.fmp_client import FMPClient, get_fmp_client

logger = logging.getLogger("core.fmp_fundamentals")


# ── dataclasses ───────────────────────────────────────────────────

@dataclass
class FundamentalProfile:
    """Processed fundamental snapshot for a single symbol."""
    symbol: str

    # valuation
    forward_pe: float = np.nan
    trailing_pe: float = np.nan
    ev_ebitda: float = np.nan
    price_to_book: float = np.nan
    price_to_sales: float = np.nan
    fcf_yield: float = np.nan
    dividend_yield: float = np.nan
    owner_earnings_yield: float = np.nan

    # growth
    revenue_growth_yoy: float = np.nan
    eps_growth_yoy: float = np.nan
    revenue_growth_ntm: float = np.nan
    eps_growth_ntm: float = np.nan

    # quality
    roic: float = np.nan
    roe: float = np.nan
    gross_margin: float = np.nan
    operating_margin: float = np.nan
    net_margin: float = np.nan
    accrual_ratio: float = np.nan  # CFO/NI — >1 = high quality

    # earnings
    last_surprise_pct: float = np.nan
    avg_surprise_pct: float = np.nan
    beat_rate: float = np.nan  # % of beats in recent quarters

    # market
    market_cap: float = np.nan
    sector: str = ""
    industry: str = ""

    # coverage
    data_coverage: float = 0.0  # 0-1, % of fields populated

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class PairFundamentalComparison:
    """Relative fundamental comparison between two pair legs."""
    pair_label: str
    sym_x: str
    sym_y: str
    profile_x: FundamentalProfile
    profile_y: FundamentalProfile

    # relative scores (positive = X is cheaper/better)
    relative_value_score: float = 0.0  # composite cheapness
    relative_growth_score: float = 0.0  # growth advantage
    relative_quality_score: float = 0.0  # quality advantage
    fundamental_justification: float = 0.0  # 0-1, how much of spread is explained
    action_bias: str = "NEUTRAL"  # LEAN_X, LEAN_Y, NEUTRAL

    def to_dict(self) -> dict:
        return {
            "pair_label": self.pair_label,
            "sym_x": self.sym_x,
            "sym_y": self.sym_y,
            "relative_value_score": self.relative_value_score,
            "relative_growth_score": self.relative_growth_score,
            "relative_quality_score": self.relative_quality_score,
            "fundamental_justification": self.fundamental_justification,
            "action_bias": self.action_bias,
            **{f"x_{k}": v for k, v in self.profile_x.to_dict().items() if k != "symbol"},
            **{f"y_{k}": v for k, v in self.profile_y.to_dict().items() if k != "symbol"},
        }


# ── helpers ───────────────────────────────────────────────────────

def _safe_float(d: dict, *keys: str) -> float:
    """Extract first available float from dict keys."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                f = float(v)
                if np.isfinite(f):
                    return f
            except (ValueError, TypeError):
                pass
    return np.nan


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _z_relative(a: float, b: float) -> float:
    """Relative z-score: (a - b) / avg, bounded to [-3, 3]."""
    if np.isnan(a) or np.isnan(b):
        return 0.0
    avg = (abs(a) + abs(b)) / 2
    if avg < 1e-8:
        return 0.0
    return max(-3.0, min(3.0, (a - b) / avg))


# ── engine ────────────────────────────────────────────────────────

class FundamentalsEngine:
    """Fetches and processes FMP fundamentals for pairs trading."""

    def __init__(self, client: FMPClient | None = None):
        self.client = client or get_fmp_client()

    def build_profile(self, symbol: str) -> FundamentalProfile:
        """Build a complete fundamental profile for one symbol."""
        profile = FundamentalProfile(symbol=symbol)
        raw = self.client.get_extended_fundamentals(symbol)

        # ── ratios TTM ──
        ratios = raw.get("ratios_ttm", {})
        if ratios:
            profile.trailing_pe = _safe_float(ratios, "priceToEarningsRatioTTM", "peRatioTTM")
            profile.price_to_book = _safe_float(ratios, "priceToBookRatioTTM")
            profile.price_to_sales = _safe_float(ratios, "priceToSalesRatioTTM")
            profile.dividend_yield = _safe_float(ratios, "dividendYieldTTM", "dividendYielTTM")
            profile.fcf_yield = _safe_float(ratios, "freeCashFlowYieldTTM")
            # FCF yield from price ratio: 1/P/FCF
            if np.isnan(profile.fcf_yield):
                pfcf = _safe_float(ratios, "priceToFreeCashFlowRatioTTM")
                if not np.isnan(pfcf) and pfcf > 0:
                    profile.fcf_yield = 1.0 / pfcf
            profile.roe = _safe_float(ratios, "returnOnEquityTTM")
            profile.roic = _safe_float(ratios, "returnOnCapitalEmployedTTM", "roicTTM")
            profile.gross_margin = _safe_float(ratios, "grossProfitMarginTTM")
            profile.operating_margin = _safe_float(ratios, "operatingProfitMarginTTM")
            profile.net_margin = _safe_float(ratios, "netProfitMarginTTM", "continuousOperationsProfitMarginTTM")

        # ── key metrics TTM ──
        km = raw.get("key_metrics_ttm", {})
        if km:
            profile.ev_ebitda = _safe_float(km, "evToEBITDATTM", "enterpriseValueOverEBITDATTM")
            if np.isnan(profile.fcf_yield):
                profile.fcf_yield = _safe_float(km, "freeCashFlowYieldTTM")
            # accrual ratio / income quality: CFO / NI
            iq = _safe_float(km, "incomeQualityTTM")
            if not np.isnan(iq):
                profile.accrual_ratio = iq
            else:
                cfo = _safe_float(km, "operatingCashFlowPerShareTTM")
                eps = _safe_float(km, "netIncomePerShareTTM")
                if not np.isnan(cfo) and not np.isnan(eps) and abs(eps) > 0.01:
                    profile.accrual_ratio = cfo / eps
            profile.market_cap = _safe_float(km, "marketCap", "marketCapTTM")

        # ── enterprise values ──
        ev_list = raw.get("enterprise_values", [])
        if ev_list and isinstance(ev_list, list) and len(ev_list) > 0:
            ev0 = ev_list[0]
            if np.isnan(profile.ev_ebitda):
                ev_val = _safe_float(ev0, "enterpriseValue")
                ebitda = _safe_float(ev0, "addTotalOtherIncomeExpenseNet")
                # fallback approach if needed

        # ── financial growth ──
        growth = raw.get("financial_growth", [])
        if growth and isinstance(growth, list) and len(growth) > 0:
            g0 = growth[0]
            profile.revenue_growth_yoy = _safe_float(g0, "revenueGrowth")
            profile.eps_growth_yoy = _safe_float(g0, "epsgrowth", "epsGrowth")

        # ── analyst estimates ──
        estimates = raw.get("analyst_estimates", [])
        if estimates and isinstance(estimates, list) and len(estimates) > 0:
            e0 = estimates[0]
            est_eps = _safe_float(e0, "estimatedEpsAvg")
            est_rev = _safe_float(e0, "estimatedRevenueAvg")
            if not np.isnan(est_eps) and not np.isnan(profile.trailing_pe):
                # forward PE = price / est_eps (approximate via trailing_pe)
                pass
            profile.eps_growth_ntm = _safe_float(e0, "estimatedEpsAvg")
            profile.revenue_growth_ntm = _safe_float(e0, "estimatedRevenueAvg")

        # ── earnings surprises ──
        surprises = raw.get("earnings_surprises", [])
        if surprises and isinstance(surprises, list):
            surprise_pcts = []
            for s in surprises[:8]:
                actual = _safe_float(s, "epsActual", "actualEarningResult")
                est = _safe_float(s, "epsEstimated", "estimatedEarning")
                if not np.isnan(actual) and not np.isnan(est) and abs(est) > 0.001:
                    surprise_pcts.append((actual - est) / abs(est) * 100)

            if surprise_pcts:
                profile.last_surprise_pct = surprise_pcts[0]
                profile.avg_surprise_pct = np.mean(surprise_pcts)
                profile.beat_rate = sum(1 for s in surprise_pcts if s > 0) / len(surprise_pcts)

        # ── cashflow for owner earnings ──
        cf = raw.get("cashflow", [])
        if cf and isinstance(cf, list) and len(cf) > 0:
            cf0 = cf[0]
            ocf = _safe_float(cf0, "operatingCashFlow")
            capex = _safe_float(cf0, "capitalExpenditure")
            if not np.isnan(ocf) and not np.isnan(capex) and not np.isnan(profile.market_cap):
                owner_earnings = ocf + capex  # capex is negative
                if profile.market_cap > 0:
                    profile.owner_earnings_yield = owner_earnings / profile.market_cap

        # ── coverage score ──
        fields = [
            profile.trailing_pe, profile.ev_ebitda, profile.price_to_book,
            profile.fcf_yield, profile.revenue_growth_yoy, profile.eps_growth_yoy,
            profile.roic, profile.gross_margin, profile.operating_margin,
            profile.accrual_ratio, profile.beat_rate,
        ]
        populated = sum(1 for f in fields if not np.isnan(f))
        profile.data_coverage = populated / len(fields)

        return profile

    def build_batch_profiles(self, symbols: Sequence[str]) -> dict[str, FundamentalProfile]:
        """Build profiles for multiple symbols."""
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.build_profile(sym)
            except Exception as e:
                logger.warning("Failed to build profile for %s: %s", sym, e)
                results[sym] = FundamentalProfile(symbol=sym)
        return results

    def compare_pair(
        self,
        sym_x: str,
        sym_y: str,
        profile_x: FundamentalProfile | None = None,
        profile_y: FundamentalProfile | None = None,
    ) -> PairFundamentalComparison:
        """Compare fundamentals between two pair legs."""
        px = profile_x or self.build_profile(sym_x)
        py = profile_y or self.build_profile(sym_y)

        pair_label = f"{sym_x}/{sym_y}"
        comp = PairFundamentalComparison(
            pair_label=pair_label,
            sym_x=sym_x, sym_y=sym_y,
            profile_x=px, profile_y=py,
        )

        # ── relative value (negative PE z = X is cheaper) ──
        val_scores = []
        # For PE: lower is cheaper, so negate
        pe_z = -_z_relative(px.trailing_pe, py.trailing_pe)
        if pe_z != 0:
            val_scores.append(pe_z)
        eveb_z = -_z_relative(px.ev_ebitda, py.ev_ebitda)
        if eveb_z != 0:
            val_scores.append(eveb_z)
        pb_z = -_z_relative(px.price_to_book, py.price_to_book)
        if pb_z != 0:
            val_scores.append(pb_z)
        # For yields: higher is cheaper, so positive
        fcf_z = _z_relative(px.fcf_yield, py.fcf_yield)
        if fcf_z != 0:
            val_scores.append(fcf_z)

        comp.relative_value_score = np.mean(val_scores) if val_scores else 0.0

        # ── relative growth ──
        growth_scores = []
        rev_z = _z_relative(px.revenue_growth_yoy, py.revenue_growth_yoy)
        if rev_z != 0:
            growth_scores.append(rev_z)
        eps_z = _z_relative(px.eps_growth_yoy, py.eps_growth_yoy)
        if eps_z != 0:
            growth_scores.append(eps_z)

        comp.relative_growth_score = np.mean(growth_scores) if growth_scores else 0.0

        # ── relative quality ──
        quality_scores = []
        roic_z = _z_relative(px.roic, py.roic)
        if roic_z != 0:
            quality_scores.append(roic_z)
        margin_z = _z_relative(px.operating_margin, py.operating_margin)
        if margin_z != 0:
            quality_scores.append(margin_z)
        accrual_z = _z_relative(px.accrual_ratio, py.accrual_ratio)
        if accrual_z != 0:
            quality_scores.append(accrual_z)

        comp.relative_quality_score = np.mean(quality_scores) if quality_scores else 0.0

        # ── fundamental justification (0-1) ──
        # If fundamentals strongly favor one side, the spread may be justified
        total_bias = abs(comp.relative_value_score) + abs(comp.relative_growth_score)
        comp.fundamental_justification = _clip01(total_bias / 4.0)

        # ── action bias ──
        net = (
            comp.relative_value_score * 0.4
            + comp.relative_growth_score * 0.3
            + comp.relative_quality_score * 0.3
        )
        if net > 0.5:
            comp.action_bias = "LEAN_X"
        elif net < -0.5:
            comp.action_bias = "LEAN_Y"
        else:
            comp.action_bias = "NEUTRAL"

        return comp

    def compare_batch(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[PairFundamentalComparison]:
        """Compare fundamentals for multiple pairs."""
        # collect unique symbols
        all_syms = set()
        for x, y in pairs:
            all_syms.add(x)
            all_syms.add(y)

        # batch fetch profiles
        profiles = self.build_batch_profiles(list(all_syms))

        # compare each pair
        results = []
        for x, y in pairs:
            try:
                comp = self.compare_pair(x, y, profiles.get(x), profiles.get(y))
                results.append(comp)
            except Exception as e:
                logger.warning("Failed to compare %s/%s: %s", x, y, e)
        return results
