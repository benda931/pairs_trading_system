# -*- coding: utf-8 -*-
"""
core/leverage_engine.py — Leverage & Position Sizing Engine
===========================================================

Ported from srv_quant_system with pairs-trading adaptations.

Features:
- Half-Kelly position sizing
- Multiplicative leverage combining 5 factors
- Progressive drawdown deleveraging
- VIX-based dampening
- Vol-targeting
- Risk parity across pairs
- Per-pair and portfolio-level caps
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("core.leverage_engine")


# ── dataclasses ───────────────────────────────────────────────────

@dataclass
class LeverageResult:
    """Output of leverage computation."""
    target_leverage: float = 1.0
    base_leverage: float = 1.0
    regime_multiplier: float = 1.0
    vix_dampening: float = 1.0
    drawdown_factor: float = 1.0
    margin_headroom: float = 1.0
    components: dict = field(default_factory=dict)


@dataclass
class PositionSize:
    """Position size for a single pair."""
    pair_label: str
    raw_weight: float = 0.0
    conviction_weight: float = 0.0
    risk_parity_weight: float = 0.0
    final_weight: float = 0.0
    notional: float = 0.0


# ── core functions ────────────────────────────────────────────────

def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    *,
    fraction: float = 0.5,
) -> float:
    """Half-Kelly (or configurable fraction) position sizing.

    f* = fraction * (p/a - q/b)
    where p=win_rate, q=1-p, a=avg_loss, b=avg_win
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    p = max(0.0, min(1.0, win_rate))
    q = 1.0 - p
    kelly = p / avg_loss - q / avg_win
    return max(0.0, fraction * kelly)


def vol_target_leverage(
    realized_vol: float,
    target_vol: float = 0.12,
    cap: float = 3.0,
) -> float:
    """Compute leverage to achieve target volatility."""
    if realized_vol <= 0.001:
        return cap
    return min(cap, target_vol / realized_vol)


def drawdown_deleverage(
    current_drawdown: float,
    dd_start: float = 0.05,
    dd_full_stop: float = 0.20,
) -> float:
    """Linear ramp from 100% to 0% between dd_start and dd_full_stop.

    Returns a multiplier in [0, 1]:
    - Above dd_start: starts reducing
    - At dd_full_stop: fully deleveraged (0)
    - Below dd_start: no reduction (1)
    """
    dd = abs(current_drawdown)
    if dd <= dd_start:
        return 1.0
    if dd >= dd_full_stop:
        return 0.0
    return 1.0 - (dd - dd_start) / (dd_full_stop - dd_start)


def vix_dampening(
    vix_level: float,
    vix_low: float = 15.0,
    vix_high: float = 35.0,
    min_factor: float = 0.3,
) -> float:
    """VIX-based leverage dampening.

    Piecewise linear: full at vix_low, min_factor at vix_high.
    """
    if vix_level <= vix_low:
        return 1.0
    if vix_level >= vix_high:
        return min_factor
    return 1.0 - (1.0 - min_factor) * (vix_level - vix_low) / (vix_high - vix_low)


def regime_leverage_multiplier(regime: str) -> float:
    """Regime-conditional leverage multiplier."""
    return {
        "CALM": 1.2,
        "NORMAL": 1.0,
        "TENSION": 0.6,
        "CRISIS": 0.2,
    }.get(regime.upper(), 1.0)


# ── risk parity ───────────────────────────────────────────────────

def risk_parity_weights(
    volatilities: np.ndarray,
    *,
    max_single: float = 0.25,
) -> np.ndarray:
    """Inverse-volatility risk parity weights.

    For pairs trading: each pair's weight is inversely proportional
    to its spread volatility.
    """
    if len(volatilities) == 0:
        return np.array([])

    vol = np.array(volatilities, dtype=float)
    vol = np.maximum(vol, 1e-8)  # avoid division by zero

    inv_vol = 1.0 / vol
    weights = inv_vol / inv_vol.sum()

    # cap single positions
    weights = np.minimum(weights, max_single)
    weights = weights / weights.sum()  # renormalize

    return weights


def conviction_weighted_sizes(
    convictions: np.ndarray,
    directions: np.ndarray,
    *,
    max_single: float = 0.20,
    dead_zone: float = 0.3,
) -> np.ndarray:
    """Conviction-weighted position sizes.

    convictions: 0-1 signal strength per pair
    directions: +1 (long spread) or -1 (short spread)
    """
    conv = np.array(convictions, dtype=float)
    dirs = np.array(directions, dtype=float)

    # zero out low-conviction
    conv[conv < dead_zone] = 0.0

    raw = conv * dirs
    total = np.abs(raw).sum()
    if total < 1e-8:
        return np.zeros_like(raw)

    weights = raw / total

    # enforce caps
    weights = np.clip(weights, -max_single, max_single)
    total = np.abs(weights).sum()
    if total > 1.0:
        weights = weights / total

    return weights


# ── main engine ───────────────────────────────────────────────────

class LeverageEngine:
    """Computes target leverage and position sizes for pairs portfolio.

    Combines 5 multiplicative factors:
    1. Sharpe-scaled base leverage
    2. Regime multiplier
    3. VIX dampening
    4. Drawdown deleveraging
    5. Margin headroom

    Each factor can independently reduce leverage to zero.
    """

    def __init__(
        self,
        target_vol: float = 0.12,
        max_leverage: float = 5.0,
        max_single_pair: float = 0.20,
        max_sector_concentration: float = 0.40,
        dd_start: float = 0.05,
        dd_full_stop: float = 0.20,
    ):
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.max_single_pair = max_single_pair
        self.max_sector_concentration = max_sector_concentration
        self.dd_start = dd_start
        self.dd_full_stop = dd_full_stop

    def compute_target_leverage(
        self,
        realized_vol: float,
        current_drawdown: float = 0.0,
        vix_level: float = 20.0,
        regime: str = "NORMAL",
        margin_utilization: float = 0.5,
    ) -> LeverageResult:
        """Compute target leverage from all factors."""
        result = LeverageResult()

        # 1. Vol-targeted base
        base = vol_target_leverage(realized_vol, self.target_vol, self.max_leverage)
        result.base_leverage = base

        # 2. Regime
        regime_mult = regime_leverage_multiplier(regime)
        result.regime_multiplier = regime_mult

        # 3. VIX dampening
        vix_damp = vix_dampening(vix_level)
        result.vix_dampening = vix_damp

        # 4. Drawdown deleverage
        dd_factor = drawdown_deleverage(current_drawdown, self.dd_start, self.dd_full_stop)
        result.drawdown_factor = dd_factor

        # 5. Margin headroom
        margin_room = max(0.0, 1.0 - margin_utilization)
        margin_factor = min(1.0, margin_room / 0.3)  # start reducing at 70% margin use
        result.margin_headroom = margin_factor

        # Multiplicative combination
        target = base * regime_mult * vix_damp * dd_factor * margin_factor
        target = max(0.0, min(target, self.max_leverage))
        result.target_leverage = target

        result.components = {
            "base": base,
            "regime": regime_mult,
            "vix": vix_damp,
            "drawdown": dd_factor,
            "margin": margin_factor,
            "final": target,
        }

        return result

    def compute_position_sizes(
        self,
        pair_labels: Sequence[str],
        convictions: np.ndarray,
        directions: np.ndarray,
        volatilities: np.ndarray,
        *,
        total_capital: float = 1_000_000,
        leverage: float = 1.0,
    ) -> list[PositionSize]:
        """Compute position sizes blending conviction + risk parity."""
        n = len(pair_labels)
        if n == 0:
            return []

        # risk parity weights
        rp_weights = risk_parity_weights(volatilities, max_single=self.max_single_pair)

        # conviction weights
        conv_weights = conviction_weighted_sizes(
            convictions, directions, max_single=self.max_single_pair,
        )

        # blend: 60% risk-parity + 40% conviction
        blended = 0.6 * rp_weights * np.sign(conv_weights) + 0.4 * conv_weights

        # normalize
        total = np.abs(blended).sum()
        if total > 0:
            blended = blended / total

        # enforce single-pair cap
        blended = np.clip(blended, -self.max_single_pair, self.max_single_pair)

        # apply leverage and capital
        effective_capital = total_capital * leverage

        results = []
        for i, label in enumerate(pair_labels):
            results.append(PositionSize(
                pair_label=label,
                raw_weight=float(conv_weights[i]) if i < len(conv_weights) else 0,
                conviction_weight=float(conv_weights[i]) if i < len(conv_weights) else 0,
                risk_parity_weight=float(rp_weights[i]) if i < len(rp_weights) else 0,
                final_weight=float(blended[i]),
                notional=float(blended[i]) * effective_capital,
            ))

        return results
