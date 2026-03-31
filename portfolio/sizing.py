# -*- coding: utf-8 -*-
"""
portfolio/sizing.py — Position Sizing Engine
============================================

Converts a RankedOpportunity into a SizingDecision.

Architecture:
  SizingEngine wires together two existing engines:
    - core.leverage_engine.LeverageEngine  (vol-targeting, drawdown deleveraging)
    - core.leverage_engine.risk_parity_weights (inverse-vol risk parity)

  Then applies a conviction/quality/regime scalar stack on top.

Sizing pipeline per position:
  1. Base weight = equal-weight / risk-parity / vol-target (configurable)
  2. × conviction_scalar     (from RankedOpportunity.conviction)
  3. × quality_scalar        (from quality grade)
  4. × regime_scalar         (from regime suitability)
  5. × drawdown_scalar       (from DrawdownState.throttle_factor)
  6. Cap to max_single_pair fraction
  7. Convert weight → notional using total_capital × target_leverage
  8. Split into leg_x and leg_y notionals using hedge_ratio

All multipliers are recorded in SizingDecision for full auditability.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from core.leverage_engine import (
    LeverageEngine,
    LeverageResult,
    risk_parity_weights,
    vol_target_leverage,
)
from core.contracts import PairId, SignalDirection
from portfolio.contracts import (
    DrawdownState,
    PortfolioHeatLevel,
    RankedOpportunity,
    SizingDecision,
    ThrottleState,
)

logger = logging.getLogger("portfolio.sizing")


# ── Sizing configuration ──────────────────────────────────────────

@dataclass
class SizingConfig:
    """Configuration for the position sizing engine."""

    # Base sizing mode
    mode: str = "vol_target"  # "equal_weight" | "risk_parity" | "vol_target"

    # Vol-targeting parameters
    target_portfolio_vol: float = 0.12    # 12% annualised
    max_leverage: float = 2.0
    min_leverage: float = 0.1

    # Per-position constraints
    max_single_pair_weight: float = 0.15  # 15% of capital
    min_single_pair_weight: float = 0.005 # 0.5% — below this: not executable
    max_gross_leverage: float = 4.0

    # Conviction scaling
    conviction_scaling_enabled: bool = True
    conviction_dead_zone: float = 0.25    # Below this: scale linearly from 0
    conviction_saturation: float = 0.90  # Above this: no further increase

    # Quality grade multipliers
    quality_multipliers: dict[str, float] = field(default_factory=lambda: {
        "A+": 1.20, "A": 1.10, "B+": 1.05, "B": 1.00,
        "C+": 0.85, "C": 0.70, "D": 0.50, "F": 0.0,
    })

    # Regime suitability multipliers
    regime_multipliers: dict[str, float] = field(default_factory=lambda: {
        "MEAN_REVERTING": 1.10,
        "CALM": 1.00,
        "NORMAL": 0.90,
        "VOLATILE": 0.65,
        "TENSION": 0.50,
        "TRENDING": 0.20,
        "CRISIS": 0.0,
        "BROKEN": 0.0,
        "UNKNOWN": 0.70,
    })

    # Drawdown throttle thresholds (applied on top of DrawdownState.throttle_factor)
    drawdown_throttle_enabled: bool = True

    # Margin / capital usage mode
    margin_fraction: float = 0.50   # Effective margin (50% → 2× leverage available)

    # Minimum executable notional
    min_executable_notional: float = 1_000.0

    # Default spread vol for positions without history
    default_spread_vol: float = 0.15


# ── Sizing helpers ────────────────────────────────────────────────

def _conviction_scalar(conviction: float, cfg: SizingConfig) -> float:
    """Map conviction [0,1] → size scalar, with dead-zone and saturation."""
    if not cfg.conviction_scaling_enabled:
        return 1.0
    c = max(0.0, min(1.0, conviction))
    dz = cfg.conviction_dead_zone
    sat = cfg.conviction_saturation
    if c < dz:
        return c / dz  # linearly scale up from 0 in dead zone
    if c >= sat:
        return 1.2     # small bonus for very high conviction
    return 1.0 + 0.2 * (c - dz) / max(1e-6, sat - dz)


def _quality_scalar(grade: str, cfg: SizingConfig) -> float:
    """Quality grade → size multiplier."""
    return cfg.quality_multipliers.get(grade.upper(), 0.7)


def _regime_scalar(regime: str, cfg: SizingConfig) -> float:
    """Regime suitability → size multiplier."""
    return cfg.regime_multipliers.get(regime.upper(), 0.7)


def _drawdown_scalar(drawdown_state: Optional[DrawdownState], cfg: SizingConfig) -> float:
    """Drawdown throttle factor from portfolio state."""
    if drawdown_state is None or not cfg.drawdown_throttle_enabled:
        return 1.0
    return max(0.0, min(1.0, drawdown_state.throttle_factor))


def _split_legs(
    gross_notional: float,
    hedge_ratio: float,
    direction: str,
) -> tuple[float, float]:
    """
    Split gross notional into (leg_x_notional, leg_y_notional).

    For a long-spread trade: long X, short Y (hedge_ratio × leg_x).
    For a short-spread trade: short X, long Y.
    """
    # leg_x = gross / (1 + hedge_ratio), leg_y = gross - leg_x
    total_legs = 1.0 + abs(hedge_ratio)
    leg_x = gross_notional / total_legs
    leg_y = gross_notional - leg_x
    return leg_x, leg_y


# ── Sizing Engine ─────────────────────────────────────────────────

class SizingEngine:
    """
    Computes SizingDecision for a single RankedOpportunity.

    Designed to be called inside the PortfolioAllocator loop.

    Parameters
    ----------
    config : SizingConfig
    leverage_engine : LeverageEngine (optional; created from config if not supplied)
    """

    def __init__(
        self,
        config: Optional[SizingConfig] = None,
        leverage_engine: Optional[LeverageEngine] = None,
    ):
        self._cfg = config or SizingConfig()
        self._leverage_engine = leverage_engine or LeverageEngine(
            target_vol=self._cfg.target_portfolio_vol,
            max_leverage=self._cfg.max_leverage,
            max_single_pair=self._cfg.max_single_pair_weight,
        )

    @property
    def config(self) -> SizingConfig:
        return self._cfg

    def size(
        self,
        opportunity: RankedOpportunity,
        *,
        total_capital: float,
        n_active_positions: int = 0,
        portfolio_vol: Optional[float] = None,
        spread_vol: Optional[float] = None,
        hedge_ratio: float = 1.0,
        drawdown_state: Optional[DrawdownState] = None,
        throttle_state: Optional[ThrottleState] = None,
        current_drawdown: float = 0.0,
        vix_level: float = 20.0,
    ) -> SizingDecision:
        """
        Compute a SizingDecision for one ranked opportunity.

        Parameters
        ----------
        opportunity : RankedOpportunity
        total_capital : float — total portfolio capital
        n_active_positions : int — existing open positions (for equal-weight denominator)
        portfolio_vol : float — current portfolio realised vol (for vol-targeting)
        spread_vol : float — spread-specific vol (for risk parity)
        hedge_ratio : float — |beta| from SpreadDefinition
        drawdown_state : DrawdownState — current drawdown context
        throttle_state : ThrottleState — current throttle settings
        current_drawdown : float — current portfolio drawdown fraction
        vix_level : float — VIX for leverage dampening
        """
        cfg = self._cfg
        pair_id = opportunity.pair_id
        regime = opportunity.regime
        quality_grade = opportunity.quality_grade
        conviction = opportunity.conviction
        direction = self._infer_direction(opportunity.z_score)

        # ── 1. Compute target leverage ────────────────────────────
        realized_vol = portfolio_vol or cfg.default_spread_vol
        leverage_result = self._leverage_engine.compute_target_leverage(
            realized_vol=realized_vol,
            current_drawdown=current_drawdown,
            vix_level=vix_level,
            regime=self._regime_for_leverage(regime),
        )
        target_leverage = leverage_result.target_leverage

        # ── 2. Base weight (equal-weight or risk-parity) ──────────
        base_weight = self._base_weight(
            spread_vol=spread_vol,
            n_positions=max(1, n_active_positions + 1),  # +1 for this new position
        )

        # ── 3. Scalar stack ───────────────────────────────────────
        conv_s  = _conviction_scalar(conviction, cfg)
        qual_s  = _quality_scalar(quality_grade, cfg)
        reg_s   = _regime_scalar(regime, cfg)
        dd_s    = _drawdown_scalar(drawdown_state, cfg)

        # ThrottleState can override dd_s if more restrictive
        throttle_size_mult = 1.0
        if throttle_state is not None:
            throttle_size_mult = max(0.0, min(1.0, throttle_state.size_multiplier))
        combined_dd_throttle = dd_s * throttle_size_mult

        # Apply vol_target scalar via leverage (already in leverage result)
        vol_s = min(2.0, target_leverage / max(0.01, cfg.min_leverage))

        # Final weight = base × all scalars
        raw_weight = base_weight * conv_s * qual_s * reg_s * combined_dd_throttle

        # ── 4. Cap at max_single_pair ─────────────────────────────
        was_capped = False
        cap_reason = ""
        final_weight = raw_weight
        if final_weight > cfg.max_single_pair_weight:
            final_weight = cfg.max_single_pair_weight
            was_capped = True
            cap_reason = f"max_single_pair:{cfg.max_single_pair_weight:.1%}"

        # ── 5. Compute notional ───────────────────────────────────
        effective_capital = total_capital * min(target_leverage, cfg.max_gross_leverage)
        gross_notional = final_weight * effective_capital

        # Risk notional = spread notional only (not both legs full)
        # Approximation: risk_notional = gross_notional × spread_vol / portfolio_target_vol
        spread_v = spread_vol or cfg.default_spread_vol
        risk_scalar = spread_v / max(0.001, cfg.target_portfolio_vol)
        risk_notional = gross_notional * min(1.0, risk_scalar)

        capital_usage = gross_notional * cfg.margin_fraction

        # ── 6. Split legs ─────────────────────────────────────────
        leg_x, leg_y = _split_legs(gross_notional, hedge_ratio, direction)

        # ── 7. Executability ──────────────────────────────────────
        is_executable = gross_notional >= cfg.min_executable_notional
        if not is_executable and not was_capped:
            cap_reason = f"below_min_executable:{cfg.min_executable_notional:.0f}"

        weight_of_portfolio = final_weight
        risk_contribution = (
            risk_notional / max(1.0, total_capital * cfg.target_portfolio_vol)
        )

        return SizingDecision(
            pair_id=pair_id,
            sleeve=opportunity.recommended_sleeve,
            gross_notional=gross_notional,
            risk_notional=risk_notional,
            capital_usage=capital_usage,
            weight_of_portfolio=weight_of_portfolio,
            risk_contribution=risk_contribution,
            leg_x_notional=leg_x,
            leg_y_notional=leg_y,
            hedge_ratio=hedge_ratio,
            direction=direction,
            base_weight=base_weight,
            conviction_scalar=conv_s,
            vol_target_scalar=vol_s,
            drawdown_scalar=combined_dd_throttle,
            quality_scalar=qual_s,
            regime_scalar=reg_s,
            was_capped=was_capped,
            cap_reason=cap_reason,
            min_executable_size=cfg.min_executable_notional,
            is_executable=is_executable,
        )

    def size_batch(
        self,
        opportunities: list[RankedOpportunity],
        *,
        total_capital: float,
        spread_vols: Optional[dict[str, float]] = None,
        hedge_ratios: Optional[dict[str, float]] = None,
        drawdown_state: Optional[DrawdownState] = None,
        throttle_state: Optional[ThrottleState] = None,
        portfolio_vol: Optional[float] = None,
        current_drawdown: float = 0.0,
        vix_level: float = 20.0,
    ) -> list[SizingDecision]:
        """
        Size all opportunities in the set using risk-parity across the batch.

        When mode = "risk_parity", spreads vols are used to compute
        inverse-vol weights before applying the scalar stack.
        """
        spread_vols = spread_vols or {}
        hedge_ratios = hedge_ratios or {}

        # If risk-parity mode: pre-compute portfolio-level weights
        if self._cfg.mode == "risk_parity" and opportunities:
            vols = np.array([
                spread_vols.get(o.pair_id.label, self._cfg.default_spread_vol)
                for o in opportunities
            ])
            rp_weights_arr = risk_parity_weights(vols, max_single=self._cfg.max_single_pair_weight)
        else:
            rp_weights_arr = None

        decisions = []
        for i, opp in enumerate(opportunities):
            # Inject pre-computed risk-parity base weight
            sv = spread_vols.get(opp.pair_id.label)
            hr = hedge_ratios.get(opp.pair_id.label, 1.0)

            decision = self.size(
                opp,
                total_capital=total_capital,
                n_active_positions=i,
                portfolio_vol=portfolio_vol,
                spread_vol=sv,
                hedge_ratio=hr,
                drawdown_state=drawdown_state,
                throttle_state=throttle_state,
                current_drawdown=current_drawdown,
                vix_level=vix_level,
            )

            # Override base weight with risk-parity weight if available
            if rp_weights_arr is not None and i < len(rp_weights_arr):
                rp_w = float(rp_weights_arr[i])
                decision.base_weight = rp_w

            decisions.append(decision)

        return decisions

    # ── Internal helpers ──────────────────────────────────────────

    def _base_weight(
        self,
        spread_vol: Optional[float],
        n_positions: int,
    ) -> float:
        """Compute base weight before scalar adjustments."""
        cfg = self._cfg
        if cfg.mode == "equal_weight":
            return min(cfg.max_single_pair_weight, 1.0 / n_positions)
        elif cfg.mode == "risk_parity" and spread_vol:
            # Single-pair risk parity: inverse vol, normalised to 1/n
            # The batch method will override this with portfolio-level weights
            return min(cfg.max_single_pair_weight, cfg.default_spread_vol / max(1e-6, spread_vol) / n_positions)
        else:
            # vol_target: base = target_vol / spread_vol, bounded
            sv = spread_vol or cfg.default_spread_vol
            w = (cfg.target_portfolio_vol / max(1e-6, sv)) / n_positions
            return min(cfg.max_single_pair_weight, max(cfg.min_single_pair_weight, w))

    @staticmethod
    def _infer_direction(z_score: float) -> str:
        """Infer trade direction from z-score sign."""
        # z > 0: spread above mean → short spread (short X, long Y)
        # z < 0: spread below mean → long spread (long X, short Y)
        if z_score > 0:
            return "SHORT_SPREAD"
        return "LONG_SPREAD"

    @staticmethod
    def _regime_for_leverage(regime: str) -> str:
        """Map regime label to leverage engine's regime vocabulary."""
        mapping = {
            "MEAN_REVERTING": "CALM",
            "CALM": "CALM",
            "NORMAL": "NORMAL",
            "VOLATILE": "TENSION",
            "TRENDING": "TENSION",
            "TENSION": "TENSION",
            "CRISIS": "CRISIS",
            "BROKEN": "CRISIS",
        }
        return mapping.get(regime.upper(), "NORMAL")
