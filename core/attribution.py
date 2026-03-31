# -*- coding: utf-8 -*-
"""
core/attribution.py — Pair Trade Attribution & Mispricing Confidence
====================================================================

Ported from srv_quant_system with pairs-trading adaptations.

Computes 4 scoring blocks per pair:
- SDS (Statistical Dislocation Score): spread z-score strength + half-life quality
- FJS (Fundamental Justification Score): how much of the dislocation is explained
- MSS (Macro Shift Score): macro risk exposure
- STF (Structural Trend Filter): adverse trend detection

MC (Mispricing Confidence) = SDS * (1 - FJS) * (1 - MSS) * (1 - STF)
Each risk factor independently reduces confidence via multiplicative decay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger("core.attribution")


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ── dataclasses ───────────────────────────────────────────────────

@dataclass
class AttributionRow:
    """Full attribution for a single pair."""
    pair_label: str

    # Statistical Dislocation Score (0-1)
    sds: float = 0.0
    sds_z_component: float = 0.0
    sds_half_life_component: float = 0.0
    sds_dispersion_component: float = 0.0

    # Fundamental Justification Score (0-1)
    fjs: float = 0.0
    fjs_valuation_gap: float = 0.0
    fjs_growth_diff: float = 0.0

    # Macro Shift Score (0-1)
    mss: float = 0.0
    mss_rate_beta: float = 0.0
    mss_fx_beta: float = 0.0
    mss_credit_risk: float = 0.0
    mss_vix_exposure: float = 0.0

    # Structural Trend Filter (0-1)
    stf: float = 0.0
    stf_trend_ratio: float = 0.0
    stf_momentum: float = 0.0

    # Mispricing Confidence (0-1)
    mc: float = 0.0

    # labels
    action_bias: str = "NEUTRAL"  # LEAN_IN, SELECTIVE, SMALL_SIZE, AVOID
    explanation: str = ""
    tags: list[str] = field(default_factory=list)


# ── computation ───────────────────────────────────────────────────

def compute_sds(
    current_z: float,
    half_life: float,
    *,
    z_entry: float = 2.0,
    hl_ideal_min: float = 3.0,
    hl_ideal_max: float = 40.0,
    spread_vol: float = 0.0,
    spread_vol_median: float = 0.0,
) -> tuple[float, float, float, float]:
    """Statistical Dislocation Score.

    Higher = stronger statistical case for mean-reversion.
    """
    # z-score component: strength of dislocation
    z_strength = _clip01((abs(current_z) - 1.0) / 3.0)

    # half-life quality: peaks in ideal range
    if half_life <= 0:
        hl_quality = 0.0
    elif hl_ideal_min <= half_life <= hl_ideal_max:
        hl_quality = 1.0
    elif half_life < hl_ideal_min:
        hl_quality = max(0.0, half_life / hl_ideal_min)
    else:
        hl_quality = max(0.0, 1.0 - (half_life - hl_ideal_max) / 60.0)

    # dispersion component
    if spread_vol > 0 and spread_vol_median > 0:
        dispersion = _clip01(spread_vol / spread_vol_median - 0.5)
    else:
        dispersion = 0.5  # neutral

    sds = (z_strength * 0.5 + hl_quality * 0.35 + dispersion * 0.15)
    return _clip01(sds), z_strength, hl_quality, dispersion


def compute_fjs(
    fundamental_justification: float = 0.0,
    valuation_gap: float = 0.0,
    growth_diff: float = 0.0,
) -> tuple[float, float, float]:
    """Fundamental Justification Score.

    Higher = more of the spread is explained by fundamentals (less alpha).
    """
    val_component = _clip01(abs(valuation_gap) / 2.0)
    growth_component = _clip01(abs(growth_diff) / 2.0)
    fjs = max(fundamental_justification, val_component * 0.6 + growth_component * 0.4)
    return _clip01(fjs), val_component, growth_component


def compute_mss(
    rate_beta: float = 0.0,
    fx_beta: float = 0.0,
    credit_spread_z: float = 0.0,
    vix_z: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Macro Shift Score.

    Higher = more macro risk to the pair trade.
    """
    rate_component = _clip01(abs(rate_beta) / 0.5)
    fx_component = _clip01(abs(fx_beta) / 0.3)
    credit_component = _clip01(max(0, credit_spread_z) / 2.0)
    vix_component = _clip01(max(0, vix_z) / 2.0)

    mss = (rate_component * 0.3 + fx_component * 0.2 + credit_component * 0.3 + vix_component * 0.2)
    return _clip01(mss), rate_component, fx_component, credit_component, vix_component


def compute_stf(
    trend_ratio: float = 0.0,
    momentum_5d: float = 0.0,
    momentum_20d: float = 0.0,
    direction: int = 1,  # +1 for long spread, -1 for short spread
) -> tuple[float, float, float]:
    """Structural Trend Filter.

    Higher = stronger adverse trend working against the trade.
    """
    # trend ratio: ratio of trending component to mean-reverting component
    trend_component = _clip01(abs(trend_ratio))

    # momentum: is spread moving against us?
    momentum = momentum_20d if abs(momentum_20d) > abs(momentum_5d) else momentum_5d
    adverse_momentum = _clip01(abs(momentum) * direction * -1 / 0.05)

    stf = trend_component * 0.6 + adverse_momentum * 0.4
    return _clip01(stf), trend_component, adverse_momentum


def compute_attribution(
    pair_label: str,
    current_z: float,
    half_life: float,
    *,
    # fundamental inputs
    fundamental_justification: float = 0.0,
    valuation_gap: float = 0.0,
    growth_diff: float = 0.0,
    # macro inputs
    rate_beta: float = 0.0,
    fx_beta: float = 0.0,
    credit_spread_z: float = 0.0,
    vix_z: float = 0.0,
    # trend inputs
    trend_ratio: float = 0.0,
    momentum_5d: float = 0.0,
    momentum_20d: float = 0.0,
    direction: int = 1,
    # spread vol
    spread_vol: float = 0.0,
    spread_vol_median: float = 0.0,
) -> AttributionRow:
    """Compute full attribution for a pair trade."""
    row = AttributionRow(pair_label=pair_label)

    # SDS
    sds, z_comp, hl_comp, disp_comp = compute_sds(
        current_z, half_life,
        spread_vol=spread_vol, spread_vol_median=spread_vol_median,
    )
    row.sds = sds
    row.sds_z_component = z_comp
    row.sds_half_life_component = hl_comp
    row.sds_dispersion_component = disp_comp

    # FJS
    fjs, val_comp, growth_comp = compute_fjs(
        fundamental_justification, valuation_gap, growth_diff,
    )
    row.fjs = fjs
    row.fjs_valuation_gap = val_comp
    row.fjs_growth_diff = growth_comp

    # MSS
    mss, rate_comp, fx_comp, credit_comp, vix_comp = compute_mss(
        rate_beta, fx_beta, credit_spread_z, vix_z,
    )
    row.mss = mss
    row.mss_rate_beta = rate_comp
    row.mss_fx_beta = fx_comp
    row.mss_credit_risk = credit_comp
    row.mss_vix_exposure = vix_comp

    # STF
    stf, trend_comp, mom_comp = compute_stf(
        trend_ratio, momentum_5d, momentum_20d, direction,
    )
    row.stf = stf
    row.stf_trend_ratio = trend_comp
    row.stf_momentum = mom_comp

    # MC = SDS * (1 - FJS) * (1 - MSS) * (1 - STF)
    row.mc = sds * (1 - fjs) * (1 - mss) * (1 - stf)

    # action bias
    if row.mc >= 0.6:
        row.action_bias = "LEAN_IN"
        row.tags.append("high_conviction")
    elif row.mc >= 0.4:
        row.action_bias = "SELECTIVE"
        row.tags.append("moderate_conviction")
    elif row.mc >= 0.2:
        row.action_bias = "SMALL_SIZE"
        row.tags.append("low_conviction")
    else:
        row.action_bias = "AVOID"
        row.tags.append("no_edge")

    # tag specific risks
    if fjs > 0.5:
        row.tags.append("fundamentally_explained")
    if mss > 0.5:
        row.tags.append("macro_headwind")
    if stf > 0.5:
        row.tags.append("adverse_trend")
    if sds > 0.7:
        row.tags.append("strong_dislocation")

    # explanation
    parts = []
    parts.append(f"SDS={sds:.2f} (z={z_comp:.2f}, HL={hl_comp:.2f})")
    parts.append(f"FJS={fjs:.2f}")
    parts.append(f"MSS={mss:.2f}")
    parts.append(f"STF={stf:.2f}")
    parts.append(f"MC={row.mc:.2f} → {row.action_bias}")
    row.explanation = " | ".join(parts)

    return row


def compute_batch_attribution(
    pairs: list[dict],
) -> list[AttributionRow]:
    """Compute attribution for multiple pairs.

    Each dict should have the keys expected by compute_attribution().
    """
    results = []
    for p in pairs:
        try:
            results.append(compute_attribution(**p))
        except Exception as e:
            logger.warning("Attribution failed for %s: %s", p.get("pair_label"), e)
    return results
