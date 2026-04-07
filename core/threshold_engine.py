# -*- coding: utf-8 -*-
"""
core/threshold_engine.py — Adaptive Threshold Engine
=====================================================

The threshold engine replaces hard-coded z-score constants with a
principled, inspectable, and adaptable threshold framework.

Three operating modes:
  STATIC              — Fixed thresholds from config (baseline/fallback)
  VOLATILITY_SCALED   — Entry/stop widen when spread vol is elevated
  REGIME_CONDITIONED  — Different threshold sets per regime label

All three modes support hysteresis (entry_z > exit_z) to avoid whipsaw.

Every threshold decision is logged with the mode and modifier values,
so downstream consumers can explain why a specific z-level was used.

Usage:
    engine = ThresholdEngine.from_config(config)
    thresholds = engine.compute(
        regime=RegimeLabel.HIGH_VOL,
        current_spread_vol=0.05,
        baseline_spread_vol=0.03,
    )
    # thresholds.entry_z, thresholds.exit_z, thresholds.stop_z, ...
"""
from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.contracts import RegimeLabel

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# THRESHOLD MODE
# ══════════════════════════════════════════════════════════════════

class ThresholdMode(str, enum.Enum):
    STATIC             = "STATIC"
    VOLATILITY_SCALED  = "VOLATILITY_SCALED"
    REGIME_CONDITIONED = "REGIME_CONDITIONED"
    ML_PREDICTED       = "ML_PREDICTED"   # Hook for future ML overlay


# ══════════════════════════════════════════════════════════════════
# THRESHOLD SET — THE OUTPUT CONTRACT
# ══════════════════════════════════════════════════════════════════

@dataclass
class ThresholdSet:
    """Complete threshold specification for one decision context.

    All thresholds are in z-score units (standard deviations of the spread).

    Hysteresis invariant:  entry_z >= exit_z  (always maintained)
    Stop invariant:        stop_z >= entry_z  (stop is outside entry)
    """
    entry_z: float = 2.0          # Enter when |z| ≥ entry_z
    exit_z: float = 0.5           # Exit when |z| ≤ exit_z (toward mean)
    stop_z: float = 4.0           # Hard stop when |z| ≥ stop_z (diverging)
    no_trade_band: float = 1.0    # No-trade zone: |z| < no_trade_band (too close to mean)
    re_entry_z: float = 2.2       # Re-entry threshold after cooldown (slightly wider)

    # Context metadata
    mode: ThresholdMode = ThresholdMode.STATIC
    regime_applied: Optional[RegimeLabel] = None
    vol_multiplier: float = 1.0   # What vol scaling was applied (1.0 = no scaling)
    modifiers_applied: list[str] = field(default_factory=list)  # Audit log of adjustments

    def __post_init__(self) -> None:
        self._enforce_invariants()

    def _enforce_invariants(self) -> None:
        """Ensure stop_z ≥ entry_z ≥ exit_z + small gap."""
        if self.exit_z >= self.entry_z:
            self.exit_z = max(0.0, self.entry_z - 0.5)
        if self.stop_z <= self.entry_z:
            self.stop_z = self.entry_z + 1.0
        if self.re_entry_z < self.entry_z:
            self.re_entry_z = self.entry_z + 0.2

    def would_enter(self, z: float) -> bool:
        return abs(z) >= self.entry_z

    def would_exit(self, z: float) -> bool:
        return abs(z) <= self.exit_z

    def would_stop(self, z: float) -> bool:
        return abs(z) >= self.stop_z

    def in_no_trade_band(self, z: float) -> bool:
        return abs(z) < self.no_trade_band

    def would_reenter(self, z: float) -> bool:
        """Re-entry after cooldown uses a slightly wider threshold."""
        return abs(z) >= self.re_entry_z

    def to_dict(self) -> dict:
        return {
            "entry_z": round(self.entry_z, 4),
            "exit_z": round(self.exit_z, 4),
            "stop_z": round(self.stop_z, 4),
            "no_trade_band": round(self.no_trade_band, 4),
            "re_entry_z": round(self.re_entry_z, 4),
            "mode": self.mode.value,
            "regime": self.regime_applied.value if self.regime_applied else None,
            "vol_multiplier": round(self.vol_multiplier, 4),
            "modifiers": self.modifiers_applied,
        }


# ══════════════════════════════════════════════════════════════════
# THRESHOLD CONFIG
# ══════════════════════════════════════════════════════════════════

@dataclass
class RegimeThresholds:
    """Threshold overrides for a specific regime."""
    entry_z: float
    exit_z: float
    stop_z: float
    no_trade_band: float = 1.0
    entry_blocked: bool = False    # If True, no new entries in this regime

    def to_threshold_set(self, mode: ThresholdMode, regime: RegimeLabel) -> ThresholdSet:
        return ThresholdSet(
            entry_z=self.entry_z,
            exit_z=self.exit_z,
            stop_z=self.stop_z,
            no_trade_band=self.no_trade_band,
            re_entry_z=self.entry_z + 0.2,
            mode=mode,
            regime_applied=regime,
        )


@dataclass
class ThresholdConfig:
    """Full configuration for the threshold engine.

    Override defaults per regime or leave as None to use static + vol scaling.
    """
    # ── Base static thresholds ────────────────────────────────────
    base_entry_z: float = 2.0
    base_exit_z: float = 0.5
    base_stop_z: float = 4.0
    base_no_trade_band: float = 1.0

    # ── Volatility scaling ────────────────────────────────────────
    vol_scale_enabled: bool = True
    vol_scale_sensitivity: float = 0.5    # 0 = no scaling; 1 = full proportional
    vol_scale_max_multiplier: float = 2.0  # Cap on vol-scaling factor
    vol_scale_min_multiplier: float = 0.8  # Floor on vol-scaling factor
    vol_scale_only_widens: bool = True     # Only widen thresholds, never tighten via vol

    # ── Regime-conditioned overrides (None = use vol-scaled static) ─
    regime_thresholds: dict[str, RegimeThresholds] = field(default_factory=lambda: {
        RegimeLabel.MEAN_REVERTING.value: RegimeThresholds(
            entry_z=2.0, exit_z=0.4, stop_z=3.5, no_trade_band=0.8
        ),
        RegimeLabel.TRENDING.value: RegimeThresholds(
            entry_z=3.0, exit_z=0.8, stop_z=4.5, no_trade_band=1.5,
            entry_blocked=True,   # No new entries in trending regime
        ),
        RegimeLabel.HIGH_VOL.value: RegimeThresholds(
            entry_z=2.5, exit_z=0.6, stop_z=5.0, no_trade_band=1.2
        ),
        RegimeLabel.CRISIS.value: RegimeThresholds(
            entry_z=4.0, exit_z=1.0, stop_z=6.0, no_trade_band=2.0,
            entry_blocked=True,   # No new entries during crisis
        ),
        RegimeLabel.BROKEN.value: RegimeThresholds(
            entry_z=99.0, exit_z=0.0, stop_z=99.0, no_trade_band=0.0,
            entry_blocked=True,   # Never enter a broken spread
        ),
    })

    # ── Hysteresis enforcement ────────────────────────────────────
    min_entry_exit_gap: float = 0.5   # Minimum gap between entry_z and exit_z

    # ── Confidence-aware exit tightening ─────────────────────────
    confidence_exit_tightening: bool = True
    # Low confidence → tighten exit (exit sooner, don't hold through uncertainty)
    # multiplier < 1.0 means exit_z shrinks → position exits at lower z
    low_confidence_exit_z_multiplier: float = 0.75  # When confidence < 0.3, tighten exit
    low_confidence_exit_tightening_threshold: float = 0.3

    # ── ML overlay hook ──────────────────────────────────────────
    ml_threshold_override_enabled: bool = False


# ══════════════════════════════════════════════════════════════════
# THRESHOLD ENGINE
# ══════════════════════════════════════════════════════════════════

class ThresholdEngine:
    """
    Computes adaptive thresholds for signal entry and exit.

    Three modes are evaluated in order of precedence:
      1. ML override (if enabled and model available)
      2. Regime-conditioned (if regime is MEAN_REVERTING/HIGH_VOL/etc.)
      3. Volatility-scaled (if vol data available)
      4. Static fallback

    The output ThresholdSet carries the mode and modifier values so
    every threshold decision is fully inspectable.
    """

    def __init__(self, config: Optional[ThresholdConfig] = None) -> None:
        self._cfg = config or ThresholdConfig()
        self._ml_hook: Optional[object] = None   # Set via set_ml_hook()

    @classmethod
    def from_config(cls, config: ThresholdConfig) -> "ThresholdEngine":
        return cls(config)

    @classmethod
    def default(cls) -> "ThresholdEngine":
        return cls(ThresholdConfig())

    def set_ml_hook(self, hook: object) -> None:
        """Register an ML-based threshold predictor.

        The hook must implement: predict(regime, vol_ratio, ...) -> ThresholdSet
        """
        self._ml_hook = hook

    # ── Main interface ────────────────────────────────────────────

    def compute(
        self,
        *,
        regime: RegimeLabel = RegimeLabel.UNKNOWN,
        current_spread_vol: float = np.nan,
        baseline_spread_vol: float = np.nan,
        signal_confidence: float = 1.0,
        half_life_days: float = np.nan,
        is_reentry: bool = False,
        garch_vol_forecast: Optional[float] = None,   # 1-step GARCH conditional vol forecast
    ) -> ThresholdSet:
        """
        Compute thresholds for the current signal context.

        Parameters
        ----------
        regime : Current regime label
        current_spread_vol : Recent spread volatility (e.g., 20-day)
        baseline_spread_vol : Long-window baseline spread volatility (e.g., 252-day)
        signal_confidence : Signal engine confidence [0, 1]
        half_life_days : Estimated half-life (used for exit calibration)
        is_reentry : If True, use re-entry thresholds (slightly wider)
        garch_vol_forecast : Optional 1-step GARCH conditional vol forecast.
            When provided, replaces the realized vol ratio for volatility scaling.
        """
        cfg = self._cfg
        modifiers: list[str] = []

        # ── 1. ML override (if available) ─────────────────────────
        if cfg.ml_threshold_override_enabled and self._ml_hook is not None:
            try:
                result = self._ml_hook.predict(  # type: ignore[attr-defined]
                    regime=regime,
                    vol_ratio=self._vol_ratio(current_spread_vol, baseline_spread_vol),
                )
                if isinstance(result, ThresholdSet):
                    result.modifiers_applied.append("ml_override")
                    logger.debug("ThresholdEngine: ML override applied for %s", regime.value)
                    return result
            except Exception as exc:
                logger.warning("ThresholdEngine: ML hook failed, falling back: %s", exc)

        # ── 2. Regime-conditioned ─────────────────────────────────
        regime_thresh = cfg.regime_thresholds.get(regime.value)
        if regime_thresh is not None:
            ts = regime_thresh.to_threshold_set(ThresholdMode.REGIME_CONDITIONED, regime)
            modifiers.append(f"regime={regime.value}")
        else:
            # ── 3. Volatility-scaled ──────────────────────────────
            # Use GARCH conditional vol if available, else fall back to realized vol ratio
            baseline_vol = baseline_spread_vol if not math.isnan(baseline_spread_vol) else np.nan
            if garch_vol_forecast is not None and garch_vol_forecast > 0:
                vol_mult_raw = garch_vol_forecast / baseline_vol if (not math.isnan(baseline_vol) and baseline_vol > 0) else 1.0
                logger.debug(
                    "Threshold using GARCH vol forecast: %.4f (baseline: %.4f)",
                    garch_vol_forecast, baseline_vol if not math.isnan(baseline_vol) else 0.0,
                )
                # Apply sensitivity and caps consistent with _compute_vol_multiplier
                if cfg.vol_scale_enabled:
                    mult = 1.0 + cfg.vol_scale_sensitivity * (vol_mult_raw - 1.0)
                    mult = max(cfg.vol_scale_min_multiplier, min(cfg.vol_scale_max_multiplier, mult))
                    if cfg.vol_scale_only_widens and mult < 1.0:
                        mult = 1.0
                    vol_mult = mult
                else:
                    vol_mult = 1.0
                modifiers.append("garch_vol_scaled")
            else:
                vol_mult = self._compute_vol_multiplier(current_spread_vol, baseline_spread_vol)
            entry_z = cfg.base_entry_z * vol_mult
            exit_z  = cfg.base_exit_z * min(vol_mult, 1.5)   # Cap at 1.5x to prevent exit becoming unreachable
            stop_z  = cfg.base_stop_z * max(1.0, vol_mult)  # stop widens more aggressively

            if vol_mult != 1.0:
                modifiers.append(f"vol_scaled={vol_mult:.3f}")
                modifiers.append("vol_exit_scaled")

            ts = ThresholdSet(
                entry_z=entry_z,
                exit_z=exit_z,
                stop_z=stop_z,
                no_trade_band=cfg.base_no_trade_band,
                mode=ThresholdMode.VOLATILITY_SCALED if vol_mult != 1.0 else ThresholdMode.STATIC,
                regime_applied=regime,
                vol_multiplier=vol_mult,
            )

        # ── 4. Confidence-aware exit tightening ───────────────────
        if cfg.confidence_exit_tightening and signal_confidence < cfg.low_confidence_exit_tightening_threshold:
            ts.exit_z *= cfg.low_confidence_exit_z_multiplier
            # NOTE: multiplier < 1.0 → tightens exit → exits sooner under uncertainty
            modifiers.append("low_confidence_exit_tightened")

        # ── 5. Half-life calibration of exit target ───────────────
        if not math.isnan(half_life_days) and half_life_days > 0:
            # Fast mean-reverters: tighten exit (they overshoot quickly)
            # Slow mean-reverters: hold longer before exiting
            if half_life_days < 5:
                ts.exit_z = min(ts.exit_z, 0.2)
                modifiers.append("fast_hl_exit_tightened")
            elif half_life_days > 60:
                ts.exit_z = max(ts.exit_z, 0.8)
                modifiers.append("slow_hl_exit_widened")

        # ── 6. Re-entry threshold ─────────────────────────────────
        if is_reentry:
            ts.re_entry_z = ts.entry_z + 0.3
            modifiers.append("reentry_threshold")

        # ── 7. Enforce invariants and log ─────────────────────────
        ts._enforce_invariants()
        ts.modifiers_applied = modifiers

        logger.debug(
            "ThresholdEngine: entry=%.2f exit=%.2f stop=%.2f [%s]",
            ts.entry_z, ts.exit_z, ts.stop_z, ", ".join(modifiers) or "static"
        )
        return ts

    # ── Entry blocking ────────────────────────────────────────────

    def is_entry_blocked_by_regime(self, regime: RegimeLabel) -> bool:
        """Does this regime block all new entries?"""
        thresh = self._cfg.regime_thresholds.get(regime.value)
        return thresh is not None and thresh.entry_blocked

    # ── Helpers ───────────────────────────────────────────────────

    def _vol_ratio(
        self, current: float, baseline: float
    ) -> float:
        if (math.isnan(current) or math.isnan(baseline)
                or baseline <= 0 or current <= 0):
            return 1.0
        return current / baseline

    def _compute_vol_multiplier(
        self, current_vol: float, baseline_vol: float
    ) -> float:
        """Compute the volatility scaling multiplier.

        Returns 1.0 if scaling is disabled or data is unavailable.
        """
        cfg = self._cfg
        if not cfg.vol_scale_enabled:
            return 1.0

        ratio = self._vol_ratio(current_vol, baseline_vol)
        if ratio == 1.0:
            return 1.0

        # Sensitivity: 0 = no scaling, 1 = full proportional
        # mult = 1 + sensitivity * (ratio - 1)
        mult = 1.0 + cfg.vol_scale_sensitivity * (ratio - 1.0)

        # Cap and floor
        mult = max(cfg.vol_scale_min_multiplier, min(cfg.vol_scale_max_multiplier, mult))

        # Only widen if configured
        if cfg.vol_scale_only_widens and mult < 1.0:
            return 1.0

        return mult

    # ── Inspection ────────────────────────────────────────────────

    def explain(
        self,
        regime: RegimeLabel,
        current_spread_vol: float = np.nan,
        baseline_spread_vol: float = np.nan,
    ) -> dict:
        """Return a human-readable explanation of the threshold decision."""
        ts = self.compute(
            regime=regime,
            current_spread_vol=current_spread_vol,
            baseline_spread_vol=baseline_spread_vol,
        )
        vol_ratio = self._vol_ratio(current_spread_vol, baseline_spread_vol)
        return {
            "thresholds": ts.to_dict(),
            "vol_ratio": round(vol_ratio, 4),
            "entry_blocked": self.is_entry_blocked_by_regime(regime),
            "regime_override_exists": regime.value in self._cfg.regime_thresholds,
            "vol_scaling_enabled": self._cfg.vol_scale_enabled,
        }


__all__ = [
    "ThresholdMode",
    "ThresholdSet",
    "RegimeThresholds",
    "ThresholdConfig",
    "ThresholdEngine",
]
