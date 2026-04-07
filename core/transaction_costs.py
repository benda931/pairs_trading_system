# -*- coding: utf-8 -*-
"""
core/transaction_costs.py — Transaction Cost Model
====================================================

Institutional-grade transaction cost model for pairs trading.

Components:
  1. Bid-ask spread cost     — half-spread per leg
  2. Market impact (Almgren-Chriss) — price impact proportional to trade size / ADV
  3. Borrow cost             — short-leg financing cost
  4. Commission              — broker fee per share/notional

Usage:
    from core.transaction_costs import TransactionCostModel, TradeCostEstimate

    tc = TransactionCostModel()
    cost = tc.estimate(
        notional_x=50_000, notional_y=50_000,
        price_x=100.0, price_y=150.0,
        adv_x=5_000_000, adv_y=8_000_000,
        spread_bps_x=5.0, spread_bps_y=6.0,
    )
    print(cost.total_bps)   # round-trip cost in basis points
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TCConfig:
    """Transaction cost model configuration."""
    # Market impact: Almgren-Chriss eta coefficient
    # impact_bps = eta * (trade_size_usd / adv_usd)^0.5
    impact_eta: float = 0.10          # Typical equity pairs range 0.05–0.20
    impact_exponent: float = 0.5      # Square-root law (Almgren-Chriss 2001)

    # Commission: fixed per-notional rate
    commission_bps: float = 0.50      # 0.5 bps per side (institutional rate)

    # Short borrow cost (annualised bps, amortised per holding day)
    default_borrow_cost_bps_annual: float = 50.0   # 50 bps/year = ~0.2 bps/day

    # Spread cost: fraction of bid-ask spread paid per trade
    spread_cost_fraction: float = 0.5   # Pay half-spread on entry and exit

    # Default spread estimates when not provided (by instrument type)
    default_spread_bps_liquid:   float = 3.0    # Large-cap ETF / liquid stock
    default_spread_bps_standard: float = 7.0    # Standard mid-cap
    default_spread_bps_illiquid: float = 20.0   # Illiquid / small-cap

    # ADV thresholds for liquidity classification
    adv_liquid_threshold:   float = 50_000_000   # $50M+ ADV = liquid
    adv_standard_threshold: float = 5_000_000    # $5M+ ADV = standard


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TradeCostEstimate:
    """Round-trip transaction cost breakdown for a pairs trade."""
    spread_cost_bps:   float = 0.0   # Bid-ask spread cost (both legs)
    impact_cost_bps:   float = 0.0   # Market impact (both legs)
    commission_bps:    float = 0.0   # Commission (both legs)
    borrow_cost_bps:   float = 0.0   # Short-leg borrow (amortised for holding_days)

    # Derived
    total_bps:          float = 0.0   # Sum of all components (round-trip)
    total_pct:          float = 0.0   # As fraction (not bps)
    notional:           float = 0.0   # Gross notional of trade
    breakeven_z:        float = 0.0   # Min z-score needed to cover costs

    holding_days_assumed: int = 1

    def is_executable(self, min_z: float = 1.5, target_sharpe: float = 0.5) -> bool:
        """Returns True if expected P&L exceeds transaction costs at given entry z."""
        # Rough check: cost must be < z_score * spread_vol equivalent
        return self.breakeven_z < min_z * 0.5  # Can cover costs at half the entry threshold


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TransactionCostModel:
    """
    Almgren-Chriss inspired transaction cost model for equity pairs.

    Estimates round-trip costs (entry + exit) for a single pairs trade.
    All costs expressed in basis points of gross notional.
    """

    def __init__(self, config: Optional[TCConfig] = None) -> None:
        self.cfg = config or TCConfig()

    def estimate(
        self,
        notional_x: float,
        notional_y: float,
        price_x: float,
        price_y: float,
        *,
        adv_x: Optional[float] = None,
        adv_y: Optional[float] = None,
        spread_bps_x: Optional[float] = None,
        spread_bps_y: Optional[float] = None,
        holding_days: int = 20,
        short_leg: str = "y",   # "x" or "y" — which leg is short
        spread_vol_pct: float = 0.02,  # spread daily vol for breakeven calculation
    ) -> TradeCostEstimate:
        """
        Estimate round-trip transaction costs for a pairs trade.

        All costs include BOTH entry AND exit legs (round-trip).
        """
        gross_notional = abs(notional_x) + abs(notional_y)
        if gross_notional < 1.0:
            return TradeCostEstimate()

        cfg = self.cfg

        # ── 1. Bid-Ask Spread Cost ──────────────────────────────────
        s_x = spread_bps_x or self._default_spread(adv_x)
        s_y = spread_bps_y or self._default_spread(adv_y)
        # Pay half-spread on entry + half-spread on exit = full spread round-trip per leg
        spread_cost_x = s_x * cfg.spread_cost_fraction * 2.0  # round-trip
        spread_cost_y = s_y * cfg.spread_cost_fraction * 2.0
        # Weight by notional fraction
        wx = abs(notional_x) / gross_notional
        wy = abs(notional_y) / gross_notional
        spread_cost_bps = wx * spread_cost_x + wy * spread_cost_y

        # ── 2. Market Impact ───────────────────────────────────────
        impact_x = self._market_impact(abs(notional_x), adv_x)
        impact_y = self._market_impact(abs(notional_y), adv_y)
        impact_cost_bps = (wx * impact_x + wy * impact_y) * 2.0  # round-trip

        # ── 3. Commission ──────────────────────────────────────────
        commission_bps = cfg.commission_bps * 4.0  # 4 sides: 2 legs × 2 (entry+exit)

        # ── 4. Borrow Cost (amortised) ─────────────────────────────
        borrow_notional = abs(notional_x) if short_leg == "x" else abs(notional_y)
        borrow_fraction = borrow_notional / gross_notional
        daily_borrow_bps = cfg.default_borrow_cost_bps_annual / 252.0
        borrow_cost_bps = daily_borrow_bps * holding_days * borrow_fraction

        # ── 5. Totals ──────────────────────────────────────────────
        total_bps = spread_cost_bps + impact_cost_bps + commission_bps + borrow_cost_bps
        total_pct = total_bps / 10_000.0

        # Breakeven z: how large must entry z be to cover round-trip costs?
        # Expected P&L ≈ z * spread_vol * notional; need P&L > total_cost
        # breakeven_z = total_cost_pct / spread_vol_pct
        breakeven_z = total_pct / max(spread_vol_pct, 1e-6)

        return TradeCostEstimate(
            spread_cost_bps=round(spread_cost_bps, 3),
            impact_cost_bps=round(impact_cost_bps, 3),
            commission_bps=round(commission_bps, 3),
            borrow_cost_bps=round(borrow_cost_bps, 3),
            total_bps=round(total_bps, 3),
            total_pct=round(total_pct, 6),
            notional=gross_notional,
            breakeven_z=round(breakeven_z, 4),
            holding_days_assumed=holding_days,
        )

    def is_worth_trading(
        self,
        entry_z: float,
        spread_vol_pct: float,
        cost_estimate: TradeCostEstimate,
        min_net_sharpe: float = 0.3,
    ) -> tuple[bool, str]:
        """
        Returns (worth_trading, reason).

        A trade is worth executing if expected P&L after costs justifies the signal.
        """
        expected_gross_pnl_pct = entry_z * spread_vol_pct
        expected_net_pnl_pct   = expected_gross_pnl_pct - cost_estimate.total_pct

        if expected_net_pnl_pct <= 0:
            return False, f"Costs ({cost_estimate.total_bps:.1f}bps) exceed expected P&L ({expected_gross_pnl_pct*100:.2f}%)"

        # Rough net Sharpe: net_pnl / spread_vol
        net_sharpe_approx = expected_net_pnl_pct / max(spread_vol_pct, 1e-6)
        if net_sharpe_approx < min_net_sharpe:
            return False, f"Net Sharpe {net_sharpe_approx:.2f} below threshold {min_net_sharpe}"

        return True, f"Net expected: {expected_net_pnl_pct*100:.2f}% ({net_sharpe_approx:.2f} net Sharpe)"

    def _default_spread(self, adv: Optional[float]) -> float:
        """Infer bid-ask spread from ADV when not provided."""
        cfg = self.cfg
        if adv is None:
            return cfg.default_spread_bps_standard
        if adv >= cfg.adv_liquid_threshold:
            return cfg.default_spread_bps_liquid
        if adv >= cfg.adv_standard_threshold:
            return cfg.default_spread_bps_standard
        return cfg.default_spread_bps_illiquid

    def _market_impact(self, trade_size: float, adv: Optional[float]) -> float:
        """Almgren-Chriss square-root market impact in bps."""
        if adv is None or adv <= 0:
            adv = trade_size * 20   # Assume trade is 5% of ADV if unknown
        participation = trade_size / adv
        impact = self.cfg.impact_eta * (participation ** self.cfg.impact_exponent) * 10_000
        return float(impact)  # in bps


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_default_model = TransactionCostModel()

def estimate_trade_cost(
    notional: float,
    spread_vol_pct: float = 0.02,
    holding_days: int = 20,
    adv_min: Optional[float] = None,
) -> TradeCostEstimate:
    """Quick estimate when only gross notional is known."""
    return _default_model.estimate(
        notional_x=notional / 2,
        notional_y=notional / 2,
        price_x=100.0,
        price_y=100.0,
        adv_x=adv_min,
        adv_y=adv_min,
        holding_days=holding_days,
        spread_vol_pct=spread_vol_pct,
    )
