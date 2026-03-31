# -*- coding: utf-8 -*-
"""
core/regime_engine.py — Rule-Based Regime Engine with ML Hooks
==============================================================

This module provides a deterministic, rule-based regime classifier that
operates without any ML dependencies. It extends the ML-based RegimeModel
(in regime_classifier.py) by:

1. Providing a reliable rule-based baseline that always produces a result
2. Computing structured RegimeFeatureSet from raw spread/price data
3. Translating regime classifications into tradability modifiers
4. Exposing a clean ML hook interface for supervised overlay
5. Tracking regime transitions and stability

Regime labels (from core.contracts.RegimeLabel):
  MEAN_REVERTING  — Conditions favour mean-reversion trading
  TRENDING        — Directional momentum; mean reversion hazardous
  HIGH_VOL        — Elevated volatility; reduce size
  CRISIS          — Extreme stress; no new entries
  BROKEN          — Structural break; retire relationship
  UNKNOWN         — Insufficient data

Decision modifiers emitted by the regime engine:
  entry_z_multiplier   — How much to widen entry threshold
  exit_z_multiplier    — How much to adjust exit threshold
  size_modifier        — Recommended size fraction [0, 1]
  entry_blocked        — Whether entries are completely blocked
  lifecycle_restriction — Suggested lifecycle action restriction
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from core.contracts import (
    ExitReason,
    RegimeLabel,
    TradeLifecycleState,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# REGIME FEATURE SET
# ══════════════════════════════════════════════════════════════════

@dataclass
class RegimeFeatureSet:
    """Feature set consumed by the regime engine.

    All values are scalars. NaN is allowed where a feature cannot be computed.
    Build via build_regime_features() from raw spread/price data.
    """
    pair_label: str = ""
    as_of: Optional[datetime] = None

    # ── Spread volatility ─────────────────────────────────────────
    spread_vol_20d: float = np.nan       # Rolling 20-day spread std
    spread_vol_63d: float = np.nan       # Rolling 63-day spread std
    spread_vol_252d: float = np.nan      # Long-window baseline
    vol_ratio_20_252: float = np.nan     # spread_vol_20d / spread_vol_252d
    vol_of_vol: float = np.nan           # Vol of rolling vol (vol-of-vol)

    # ── Z-score state ─────────────────────────────────────────────
    z_score: float = np.nan
    z_percentile_252d: float = np.nan    # Percentile of |z| over past year
    z_persistence: float = np.nan        # AR(1) of recent z-scores (>0 = trending)
    z_mean_shift: float = np.nan         # Recent z-mean vs. 0 (mean shift indicator)
    consecutive_days_above_2: int = 0    # Days with |z| > 2 continuously

    # ── Relationship health ───────────────────────────────────────
    rolling_corr_20d: float = np.nan
    rolling_corr_63d: float = np.nan
    corr_drift: float = np.nan           # corr_20d - corr_63d (negative = deteriorating)
    half_life_days: float = np.nan
    half_life_change_pct: float = np.nan # Change in HL vs. 63 days ago
    beta_cv: float = np.nan              # Coefficient of variation of rolling beta
    residual_var_ratio: float = np.nan   # Current residual variance vs. baseline

    # ── Break risk ────────────────────────────────────────────────
    cusum_stat: float = np.nan           # CUSUM range statistic
    break_confidence: float = np.nan     # 0-1 probability of structural break
    adf_rolling_pass_rate: float = np.nan  # Fraction of rolling windows that pass ADF

    # ── Liquidity / data quality ──────────────────────────────────
    missing_data_fraction: float = 0.0
    stale_price_days: int = 0

    def to_dict(self) -> dict:
        d = {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
             for k, v in self.__dict__.items()}
        if self.as_of:
            d["as_of"] = self.as_of.isoformat()
        return d


# ══════════════════════════════════════════════════════════════════
# REGIME TRADABILITY MODIFIERS
# ══════════════════════════════════════════════════════════════════

@dataclass
class RegimeTradabilityModifiers:
    """Decision modifiers emitted by the regime engine.

    These are RECOMMENDATIONS to the threshold engine and signal engine.
    They do not make final risk or portfolio decisions.
    """
    regime: RegimeLabel = RegimeLabel.UNKNOWN
    confidence: float = 0.0

    # Threshold adjustments
    entry_z_multiplier: float = 1.0     # Multiply entry_z by this
    exit_z_multiplier: float = 1.0
    stop_z_multiplier: float = 1.0

    # Position sizing recommendation (advisory only)
    size_modifier: float = 1.0          # [0, 1] fraction of normal size

    # Hard blocks
    entry_blocked: bool = False         # No new entries in this regime
    add_blocked: bool = False           # No scale-ins

    # Lifecycle restrictions
    lifecycle_restriction: Optional[TradeLifecycleState] = None  # Force lifecycle to this state
    suggest_exit: bool = False          # Regime recommends exiting existing positions
    suggest_reduce: bool = False        # Regime recommends partial de-risking
    suggested_exit_reason: Optional[ExitReason] = None

    # Warnings
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "entry_z_multiplier": round(self.entry_z_multiplier, 3),
            "size_modifier": round(self.size_modifier, 3),
            "entry_blocked": self.entry_blocked,
            "suggest_exit": self.suggest_exit,
            "suggest_reduce": self.suggest_reduce,
            "warnings": self.warnings,
        }


# ══════════════════════════════════════════════════════════════════
# ML CLASSIFIER HOOK PROTOCOL
# ══════════════════════════════════════════════════════════════════

@runtime_checkable
class RegimeClassifierHookProtocol(Protocol):
    """Interface for an ML-based regime classifier overlay.

    When provided, the ML hook runs alongside the rule-based engine.
    The rule-based engine's result is used when:
      - ML confidence < ml_min_confidence threshold
      - ML hook raises an exception
      - ML result is UNKNOWN and rule-based result is not

    The ML hook can upgrade or downgrade the regime label from the
    rule-based baseline, but cannot override BROKEN or CRISIS labels
    set by the rule-based engine (safety floor).
    """

    def classify(
        self,
        features: RegimeFeatureSet,
    ) -> tuple[RegimeLabel, float]:
        """Return (regime_label, confidence)."""
        ...

    @property
    def model_id(self) -> str: ...


# ══════════════════════════════════════════════════════════════════
# REGIME ENGINE CONFIG
# ══════════════════════════════════════════════════════════════════

@dataclass
class RegimeEngineConfig:
    """Configuration for the rule-based regime engine."""

    # ── Vol regime thresholds ─────────────────────────────────────
    high_vol_ratio: float = 1.8   # vol_20d / vol_252d > this → HIGH_VOL
    crisis_vol_ratio: float = 3.0

    # ── Trend detection ───────────────────────────────────────────
    trending_z_persistence: float = 0.50   # AR(1) of recent z > this → TRENDING
    trend_confirmation_bars: int = 5        # Must persist N bars

    # ── Mean shift detection ──────────────────────────────────────
    mean_shift_z_threshold: float = 1.5    # |z_mean_recent| > this → mean shifted

    # ── Break risk ────────────────────────────────────────────────
    break_risk_threshold: float = 0.60     # break_confidence > this → BROKEN
    cusum_crisis_threshold: float = 5.0

    # ── Correlation deterioration ─────────────────────────────────
    min_corr_for_tradable: float = 0.40
    corr_deterioration_for_caution: float = -0.15  # corr_drift < this → warning

    # ── ADF pass rate ─────────────────────────────────────────────
    min_adf_pass_rate: float = 0.30  # Below this → BROKEN

    # ── ML overlay ────────────────────────────────────────────────
    ml_enabled: bool = False
    ml_min_confidence: float = 0.60  # ML must have this confidence to override rule-based

    # ── Tradability modifiers by regime ──────────────────────────
    regime_modifiers: dict[str, dict] = field(default_factory=lambda: {
        RegimeLabel.MEAN_REVERTING.value: dict(
            entry_z_mult=1.0, exit_z_mult=1.0, stop_z_mult=1.0,
            size_mod=1.0, entry_blocked=False, suggest_reduce=False,
        ),
        RegimeLabel.HIGH_VOL.value: dict(
            entry_z_mult=1.3, exit_z_mult=1.1, stop_z_mult=1.2,
            size_mod=0.60, entry_blocked=False, suggest_reduce=True,
        ),
        RegimeLabel.TRENDING.value: dict(
            entry_z_mult=1.5, exit_z_mult=1.0, stop_z_mult=1.0,
            size_mod=0.0, entry_blocked=True, suggest_reduce=True,
            suggest_exit=True, exit_reason=ExitReason.REGIME_FLIP,
        ),
        RegimeLabel.CRISIS.value: dict(
            entry_z_mult=2.0, exit_z_mult=0.5, stop_z_mult=0.8,
            size_mod=0.0, entry_blocked=True, suggest_reduce=True,
            suggest_exit=True, exit_reason=ExitReason.REGIME_FLIP,
        ),
        RegimeLabel.BROKEN.value: dict(
            entry_z_mult=99.0, exit_z_mult=0.0, stop_z_mult=0.0,
            size_mod=0.0, entry_blocked=True, suggest_reduce=False,
            suggest_exit=True, exit_reason=ExitReason.STRUCTURAL_BREAK,
        ),
        RegimeLabel.UNKNOWN.value: dict(
            entry_z_mult=1.1, exit_z_mult=1.0, stop_z_mult=1.0,
            size_mod=0.80, entry_blocked=False, suggest_reduce=False,
        ),
    })


# ══════════════════════════════════════════════════════════════════
# RULE-BASED REGIME ENGINE
# ══════════════════════════════════════════════════════════════════

class RegimeEngine:
    """
    Rule-based regime classifier with ML overlay hook.

    Uses a waterfall of rule checks to classify the regime:
      1. Data quality (broken → UNKNOWN)
      2. Structural break detection (BROKEN)
      3. Crisis: extreme vol + break risk (CRISIS)
      4. Trending: z-score persistence (TRENDING)
      5. High volatility (HIGH_VOL)
      6. Default: MEAN_REVERTING

    Then applies regime-specific tradability modifiers.
    """

    def __init__(
        self,
        config: Optional[RegimeEngineConfig] = None,
        ml_hook: Optional[RegimeClassifierHookProtocol] = None,
    ) -> None:
        self._cfg = config or RegimeEngineConfig()
        self._ml_hook = ml_hook

    def set_ml_hook(self, hook: RegimeClassifierHookProtocol) -> None:
        self._ml_hook = hook

    # ── Main classification interface ─────────────────────────────

    def classify(self, features: RegimeFeatureSet) -> RegimeTradabilityModifiers:
        """Classify regime from features and return tradability modifiers."""
        cfg = self._cfg

        # 1. Rule-based classification
        regime, confidence, warnings = self._rule_classify(features)

        # 2. ML overlay (if available and regime is not BROKEN/CRISIS)
        if (cfg.ml_enabled and self._ml_hook is not None
                and regime not in (RegimeLabel.BROKEN, RegimeLabel.CRISIS)):
            regime, confidence = self._apply_ml(regime, confidence, features)

        # 3. Build modifiers
        return self._build_modifiers(regime, confidence, warnings)

    # ── Rule-based classifier ─────────────────────────────────────

    def _rule_classify(
        self, f: RegimeFeatureSet
    ) -> tuple[RegimeLabel, float, list[str]]:
        cfg = self._cfg
        warnings: list[str] = []

        # ── Data quality gate ─────────────────────────────────────
        if f.missing_data_fraction > 0.10 or f.stale_price_days > 5:
            warnings.append("Data quality concern: high missing/stale rate")
            return RegimeLabel.UNKNOWN, 0.4, warnings

        # ── Structural break → BROKEN ─────────────────────────────
        if (not math.isnan(f.break_confidence)
                and f.break_confidence >= cfg.break_risk_threshold):
            warnings.append(f"Structural break detected (confidence={f.break_confidence:.0%})")
            return RegimeLabel.BROKEN, f.break_confidence, warnings

        if (not math.isnan(f.adf_rolling_pass_rate)
                and f.adf_rolling_pass_rate < cfg.min_adf_pass_rate):
            warnings.append(f"ADF pass rate too low ({f.adf_rolling_pass_rate:.0%})")
            return RegimeLabel.BROKEN, 0.7, warnings

        # ── Correlation collapse ──────────────────────────────────
        if (not math.isnan(f.rolling_corr_20d)
                and f.rolling_corr_20d < cfg.min_corr_for_tradable):
            warnings.append(f"Correlation collapsed ({f.rolling_corr_20d:.2f})")
            return RegimeLabel.BROKEN, 0.65, warnings

        # ── Crisis: extreme vol + break risk ─────────────────────
        if (not math.isnan(f.vol_ratio_20_252)
                and f.vol_ratio_20_252 >= cfg.crisis_vol_ratio):
            if (not math.isnan(f.cusum_stat)
                    and f.cusum_stat >= cfg.cusum_crisis_threshold):
                warnings.append(f"Crisis regime: vol_ratio={f.vol_ratio_20_252:.1f}, CUSUM={f.cusum_stat:.1f}")
                return RegimeLabel.CRISIS, 0.85, warnings

        # ── Trending: z-score persistence ────────────────────────
        is_trending = (
            not math.isnan(f.z_persistence)
            and f.z_persistence >= cfg.trending_z_persistence
        )
        is_mean_shifted = (
            not math.isnan(f.z_mean_shift)
            and abs(f.z_mean_shift) >= cfg.mean_shift_z_threshold
        )
        if is_trending or is_mean_shifted:
            cause = []
            if is_trending:
                cause.append(f"z_persistence={f.z_persistence:.2f}")
            if is_mean_shifted:
                cause.append(f"z_mean_shift={f.z_mean_shift:.2f}")
            warnings.append(f"Trending regime: {', '.join(cause)}")
            return RegimeLabel.TRENDING, 0.75, warnings

        # ── High volatility ───────────────────────────────────────
        if (not math.isnan(f.vol_ratio_20_252)
                and f.vol_ratio_20_252 >= cfg.high_vol_ratio):
            warnings.append(f"High-vol regime: vol_ratio={f.vol_ratio_20_252:.1f}")
            return RegimeLabel.HIGH_VOL, 0.70, warnings

        # ── Correlation deterioration warning ─────────────────────
        if (not math.isnan(f.corr_drift)
                and f.corr_drift < cfg.corr_deterioration_for_caution):
            warnings.append(f"Correlation deteriorating (drift={f.corr_drift:.2f})")
            # Doesn't change regime label, just adds a warning

        # ── Default: mean reverting ───────────────────────────────
        base_confidence = self._estimate_mr_confidence(f)
        return RegimeLabel.MEAN_REVERTING, base_confidence, warnings

    def _estimate_mr_confidence(self, f: RegimeFeatureSet) -> float:
        """Estimate confidence that we're in a mean-reverting regime."""
        confidence_components = []

        # Low vol ratio → high confidence in MR
        if not math.isnan(f.vol_ratio_20_252):
            vol_conf = max(0.3, 1.0 - max(0.0, f.vol_ratio_20_252 - 1.0) * 0.5)
            confidence_components.append(vol_conf)

        # Low z-persistence → high confidence in MR
        if not math.isnan(f.z_persistence):
            persist_conf = max(0.3, 1.0 - f.z_persistence)
            confidence_components.append(persist_conf)

        # High ADF pass rate → high confidence
        if not math.isnan(f.adf_rolling_pass_rate):
            confidence_components.append(f.adf_rolling_pass_rate)

        return float(np.mean(confidence_components)) if confidence_components else 0.6

    # ── ML overlay ────────────────────────────────────────────────

    def _apply_ml(
        self,
        rule_regime: RegimeLabel,
        rule_confidence: float,
        features: RegimeFeatureSet,
    ) -> tuple[RegimeLabel, float]:
        cfg = self._cfg
        try:
            ml_regime, ml_conf = self._ml_hook.classify(features)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("RegimeEngine ML hook failed: %s", exc)
            return rule_regime, rule_confidence

        # Only use ML if confidence is high enough
        if ml_conf < cfg.ml_min_confidence:
            return rule_regime, rule_confidence

        # ML cannot override BROKEN or CRISIS (safety floor)
        if rule_regime in (RegimeLabel.BROKEN, RegimeLabel.CRISIS):
            return rule_regime, rule_confidence

        logger.debug(
            "RegimeEngine: ML override rule=%s → ml=%s (conf=%.2f)",
            rule_regime.value, ml_regime.value, ml_conf
        )
        return ml_regime, ml_conf

    # ── Modifier builder ──────────────────────────────────────────

    def _build_modifiers(
        self,
        regime: RegimeLabel,
        confidence: float,
        warnings: list[str],
    ) -> RegimeTradabilityModifiers:
        cfg = self._cfg
        mod_config = cfg.regime_modifiers.get(
            regime.value, cfg.regime_modifiers[RegimeLabel.UNKNOWN.value]
        )

        exit_reason = mod_config.get("exit_reason")

        return RegimeTradabilityModifiers(
            regime=regime,
            confidence=confidence,
            entry_z_multiplier=mod_config.get("entry_z_mult", 1.0),
            exit_z_multiplier=mod_config.get("exit_z_mult", 1.0),
            stop_z_multiplier=mod_config.get("stop_z_mult", 1.0),
            size_modifier=mod_config.get("size_mod", 1.0),
            entry_blocked=mod_config.get("entry_blocked", False),
            add_blocked=mod_config.get("entry_blocked", False),
            suggest_exit=mod_config.get("suggest_exit", False),
            suggest_reduce=mod_config.get("suggest_reduce", False),
            suggested_exit_reason=exit_reason,
            warnings=warnings,
        )


# ══════════════════════════════════════════════════════════════════
# FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════

def build_regime_features(
    spread: pd.Series,
    *,
    prices_x: Optional[pd.Series] = None,
    prices_y: Optional[pd.Series] = None,
    as_of: Optional[datetime] = None,
    vol_window: int = 20,
    baseline_window: int = 252,
    persistence_window: int = 20,
    mean_shift_window: int = 63,
) -> RegimeFeatureSet:
    """
    Build RegimeFeatureSet from raw spread and optional price series.

    Parameters
    ----------
    spread : Spread series (z-scored or raw)
    prices_x, prices_y : Optional price series for correlation computation
    as_of : Timestamp for the snapshot (defaults to last index)
    vol_window : Short-window for volatility
    baseline_window : Long-window baseline
    persistence_window : Window for z-score AR(1) estimation
    mean_shift_window : Window for recent mean shift detection
    """
    feat = RegimeFeatureSet(
        as_of=as_of or (spread.index[-1].to_pydatetime()
                        if hasattr(spread.index[-1], 'to_pydatetime') else None)
    )

    spread_clean = spread.dropna()
    n = len(spread_clean)

    if n < 30:
        return feat

    # ── Spread vol ────────────────────────────────────────────────
    if n >= vol_window:
        feat.spread_vol_20d = float(spread_clean.rolling(vol_window).std().iloc[-1])
    if n >= 63:
        feat.spread_vol_63d = float(spread_clean.rolling(63).std().iloc[-1])
    if n >= baseline_window:
        feat.spread_vol_252d = float(spread_clean.rolling(baseline_window).std().mean())
        if feat.spread_vol_252d > 1e-10 and not math.isnan(feat.spread_vol_20d):
            feat.vol_ratio_20_252 = feat.spread_vol_20d / feat.spread_vol_252d
        # Vol of vol
        roll_vol = spread_clean.rolling(vol_window).std().dropna()
        if len(roll_vol) >= 30:
            feat.vol_of_vol = float(roll_vol.rolling(30).std().iloc[-1])

    # ── Z-score state ─────────────────────────────────────────────
    feat.z_score = float(spread_clean.iloc[-1])

    # Percentile of |z| over past year
    if n >= 63:
        lookback = spread_clean.iloc[-min(n, baseline_window):]
        pct = float((abs(lookback) < abs(feat.z_score)).mean())
        feat.z_percentile_252d = pct

    # Z-score persistence (AR(1) of recent z-scores)
    if n >= persistence_window + 5:
        recent_z = spread_clean.iloc[-persistence_window:]
        try:
            ar1 = float(recent_z.autocorr(lag=1))
            feat.z_persistence = ar1 if math.isfinite(ar1) else np.nan
        except Exception:
            pass

    # Recent z-mean shift
    if n >= mean_shift_window + 30:
        historical_std = float(spread_clean.iloc[:-mean_shift_window].std()) + 1e-10
        recent_mean = float(spread_clean.iloc[-mean_shift_window:].mean())
        feat.z_mean_shift = recent_mean / historical_std

    # Consecutive days above |z| = 2
    above_2 = (abs(spread_clean) > 2.0)
    if len(above_2) > 0:
        # Count consecutive trailing True values
        count = 0
        for v in reversed(above_2.values):
            if v:
                count += 1
            else:
                break
        feat.consecutive_days_above_2 = count

    # ── Correlation features ──────────────────────────────────────
    if prices_x is not None and prices_y is not None:
        common = prices_x.index.intersection(prices_y.index)
        px = prices_x.loc[common].dropna()
        py = prices_y.loc[common].dropna()
        ret_x = np.log(px.clip(lower=1e-8)).diff().dropna()
        ret_y = np.log(py.clip(lower=1e-8)).diff().dropna()
        common_ret = ret_x.index.intersection(ret_y.index)
        if len(common_ret) >= 20:
            rx, ry = ret_x.loc[common_ret], ret_y.loc[common_ret]
            if len(rx) >= 20:
                feat.rolling_corr_20d = float(rx.rolling(20).corr(ry).iloc[-1])
            if len(rx) >= 63:
                feat.rolling_corr_63d = float(rx.rolling(63).corr(ry).iloc[-1])
            if not math.isnan(feat.rolling_corr_20d) and not math.isnan(feat.rolling_corr_63d):
                feat.corr_drift = feat.rolling_corr_20d - feat.rolling_corr_63d

    return feat


__all__ = [
    "RegimeFeatureSet",
    "RegimeTradabilityModifiers",
    "RegimeClassifierHookProtocol",
    "RegimeEngineConfig",
    "RegimeEngine",
    "build_regime_features",
]
