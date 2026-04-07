# -*- coding: utf-8 -*-
"""
core/signal_quality.py — Signal Quality Assessment and Meta-Labeling
=====================================================================

Signal quality is distinct from signal strength.

A large z-score is strong, but it may be:
  - late (divergence has been running for a long time with no reversion)
  - contaminated (near an earnings release or known event)
  - structurally broken (regime has shifted; mean is no longer where we think)
  - noisy (low-quality spread with poor statistical properties)
  - in a bad regime (trending or crisis)

This module implements:
  1. Rule-based quality scoring using signal stack outputs + diagnostics
  2. Formal quality grades (A+ → F)
  3. Skip/take/reduce recommendations
  4. Meta-labeling protocol for ML overlay

Design principle: quality filtering happens BEFORE position sizing.
A signal below grade C should not be acted upon regardless of z-score magnitude.

Meta-labeling (de Prado, 2018):
  A meta-label is a binary (skip=0, take=1) or probability overlay applied to
  an existing signal. ML meta-labels improve on rule-based quality by learning
  which signal setups historically succeed vs. fail. The protocol here allows
  a trained model to override the rule-based grade without bypassing it entirely
  — the rule-based floor (grade F = skip) is never overridden by ML.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from core.contracts import RegimeLabel, SignalQualityGrade
from core.diagnostics import SignalFeatures, SpreadStateSnapshot

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# SIGNAL QUALITY SCORE — THE OUTPUT CONTRACT
# ══════════════════════════════════════════════════════════════════

@dataclass
class SignalQualityScore:
    """Comprehensive quality assessment for a signal setup.

    Attributes
    ----------
    grade : Overall quality grade (A+ = best, F = skip)
    score : Numeric quality score [0, 1]
    component_scores : Dict of individual quality component scores
    skip_recommended : True if grade is F; signal should not be traded
    reduce_size_recommended : True if grade is C or D; trade smaller
    reasons : Human-readable list of quality assessment reasons
    warnings : Non-fatal quality concerns
    ml_override : Whether an ML model changed the grade
    ml_probability : ML-predicted success probability (NaN if no ML)
    """
    grade: SignalQualityGrade = SignalQualityGrade.F
    score: float = 0.0

    # Breakdown by component
    component_scores: dict[str, float] = field(default_factory=dict)

    # Recommendations
    skip_recommended: bool = True
    reduce_size_recommended: bool = False
    size_multiplier: float = 1.0     # Suggested size relative to normal (advisory only)

    # Explanations
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ML overlay
    ml_override: bool = False
    ml_probability: float = np.nan
    ml_model_id: str = ""    # Model that produced the ML override (for audit/rollback)

    def to_dict(self) -> dict:
        return {
            "grade": self.grade.value,
            "score": round(self.score, 4),
            "skip": self.skip_recommended,
            "reduce_size": self.reduce_size_recommended,
            "size_multiplier": round(self.size_multiplier, 3),
            "reasons": self.reasons,
            "warnings": self.warnings,
            "components": {k: round(v, 4) for k, v in self.component_scores.items()},
            "ml_override": self.ml_override,
            "ml_probability": None if math.isnan(self.ml_probability) else round(self.ml_probability, 4),
        }


# ══════════════════════════════════════════════════════════════════
# META-LABEL PROTOCOL — ML OVERLAY HOOK
# ══════════════════════════════════════════════════════════════════

@runtime_checkable
class MetaLabelProtocol(Protocol):
    """Interface for ML-based meta-labeling overlays.

    Implementations must:
    - Accept SignalFeatures and return a probability in [0, 1]
    - Not raise exceptions (return NaN on failure)
    - Be trained on leakage-safe datasets (DatasetBuilder.build())
    - Be auditable: every prediction must be explained or logged
    - Never override grade F (rule-based floor is inviolable)
    """

    def predict_success_probability(
        self,
        features: SignalFeatures,
    ) -> float:
        """Return probability that this signal leads to successful reversion.

        Returns
        -------
        float in [0, 1], or NaN if prediction fails.
        """
        ...

    @property
    def model_id(self) -> str:
        """Unique identifier for this model version."""
        ...


# ══════════════════════════════════════════════════════════════════
# QUALITY CONFIG
# ══════════════════════════════════════════════════════════════════

@dataclass
class QualityConfig:
    """Configuration for the quality engine."""

    # ── Conviction thresholds ─────────────────────────────────────
    min_conviction_for_c: float = 0.05   # Minimum 4-layer conviction for grade C
    min_conviction_for_b: float = 0.10
    min_conviction_for_a: float = 0.20
    min_conviction_for_aplus: float = 0.35

    # ── MR quality thresholds ─────────────────────────────────────
    min_mr_score_for_trade: float = 0.20   # Below this: skip
    strong_mr_threshold: float = 0.55

    # ── Regime quality ────────────────────────────────────────────
    blocked_regimes: frozenset = frozenset({
        RegimeLabel.CRISIS.value, RegimeLabel.BROKEN.value, RegimeLabel.TRENDING.value
    })
    caution_regimes: frozenset = frozenset({RegimeLabel.HIGH_VOL.value})

    # ── Freshness / lateness ──────────────────────────────────────
    max_time_above_threshold_days: float = 10.0   # Signals older than this are "late"
    stale_signal_days: float = 5.0                 # No z-score update in N days

    # ── Z-score extremes ──────────────────────────────────────────
    extreme_z_threshold: float = 4.5    # Very large z → potential structural break
    min_z_for_quality_signal: float = 1.5  # Below this: too small to assess quality

    # ── Historical failure filter ─────────────────────────────────
    max_failed_reversion_attempts: int = 3  # After N consecutive failures: reduce grade

    # ── Correlation / stability ───────────────────────────────────
    min_correlation_for_quality: float = 0.50
    max_beta_cv_for_quality: float = 0.40

    # ── Size multipliers by grade ─────────────────────────────────
    size_by_grade: dict[str, float] = field(default_factory=lambda: {
        SignalQualityGrade.A_PLUS.value: 1.0,
        SignalQualityGrade.A.value:      0.85,
        SignalQualityGrade.B.value:      0.70,
        SignalQualityGrade.C.value:      0.40,
        SignalQualityGrade.D.value:      0.20,
        SignalQualityGrade.F.value:      0.0,
    })

    # ── ML overlay ────────────────────────────────────────────────
    ml_enabled: bool = False
    ml_min_probability_to_upgrade: float = 0.65   # ML must predict > 65% for grade upgrade
    ml_max_grade_upgrade: int = 1                  # ML can only upgrade by 1 grade step
    ml_can_skip_d_grade: bool = True               # ML can re-enable a D-grade signal


# ══════════════════════════════════════════════════════════════════
# SIGNAL QUALITY ENGINE
# ══════════════════════════════════════════════════════════════════

class SignalQualityEngine:
    """
    Assesses the quality of a signal setup.

    Inputs:
    - Conviction scores from signal_stack.py (4-layer)
    - Regime label and confidence
    - Signal features (z-score velocity, staleness, etc.)
    - Optional ML meta-label hook

    Output:
    - SignalQualityScore with grade, score, recommendations, and full rationale

    Rule: grade F = skip, always, even if ML disagrees.
    """

    _GRADE_ORDER = [
        SignalQualityGrade.F,
        SignalQualityGrade.D,
        SignalQualityGrade.C,
        SignalQualityGrade.B,
        SignalQualityGrade.A,
        SignalQualityGrade.A_PLUS,
    ]

    def __init__(
        self,
        config: Optional[QualityConfig] = None,
        ml_hook: Optional[MetaLabelProtocol] = None,
    ) -> None:
        self._cfg = config or QualityConfig()
        self._ml_hook = ml_hook

    def set_ml_hook(self, hook: MetaLabelProtocol) -> None:
        self._ml_hook = hook

    # ── Main interface ────────────────────────────────────────────

    def assess(
        self,
        *,
        conviction: float,
        mr_score: float,
        regime: RegimeLabel,
        features: Optional[SignalFeatures] = None,
        snapshot: Optional[SpreadStateSnapshot] = None,
    ) -> SignalQualityScore:
        """
        Compute the quality grade for a signal setup.

        Parameters
        ----------
        conviction : Combined 4-layer conviction [0, 1] from signal_stack.py
        mr_score : Mean-reversion quality score [0, 1] from Layer 3
        regime : Current regime label
        features : Optional expanded feature set
        snapshot : Optional current spread state snapshot
        """
        cfg = self._cfg
        reasons: list[str] = []
        warnings: list[str] = []
        components: dict[str, float] = {}

        # ── Hard veto: regime ─────────────────────────────────────
        if regime.value in cfg.blocked_regimes:
            reasons.append(f"Regime '{regime.value}' blocks all entries")
            return SignalQualityScore(
                grade=SignalQualityGrade.F,
                score=0.0,
                component_scores={"regime": 0.0},
                skip_recommended=True,
                reasons=reasons,
            )

        # ── Score conviction ──────────────────────────────────────
        conv_score = min(1.0, conviction / max(cfg.min_conviction_for_aplus, 1e-9))
        components["conviction"] = conv_score

        # ── Score MR quality ──────────────────────────────────────
        if mr_score < cfg.min_mr_score_for_trade:
            reasons.append(f"MR quality too low ({mr_score:.2f} < {cfg.min_mr_score_for_trade})")
            return SignalQualityScore(
                grade=SignalQualityGrade.F,
                score=0.0,
                component_scores={"mr_quality": mr_score},
                skip_recommended=True,
                reasons=reasons,
            )
        mr_score_norm = min(1.0, mr_score / max(cfg.strong_mr_threshold, 1e-9))
        components["mr_quality"] = mr_score_norm

        # ── Score regime ──────────────────────────────────────────
        regime_score = 1.0
        if regime.value in cfg.caution_regimes:
            regime_score = 0.6
            warnings.append(f"Caution regime '{regime.value}': reduced quality")
        elif regime == RegimeLabel.UNKNOWN:
            regime_score = 0.7
            warnings.append("Regime unknown; quality assessment may be unreliable")
        components["regime"] = regime_score

        # ── Feature-based checks ──────────────────────────────────
        feature_score = 1.0
        if features is not None:
            feature_score, f_reasons, f_warnings = self._score_features(features)
            reasons.extend(f_reasons)
            warnings.extend(f_warnings)
        elif snapshot is not None:
            feature_score, f_reasons, f_warnings = self._score_snapshot(snapshot)
            reasons.extend(f_reasons)
            warnings.extend(f_warnings)
        components["features"] = feature_score

        # ── Composite score ───────────────────────────────────────
        composite = (
            0.40 * conv_score +
            0.30 * mr_score_norm +
            0.20 * regime_score +
            0.10 * feature_score
        )
        components["composite"] = composite

        # ── Determine grade ───────────────────────────────────────
        grade = self._grade_from_conviction_and_composite(conviction, composite, cfg)

        # ── ML overlay ────────────────────────────────────────────
        ml_override = False
        ml_prob = np.nan
        ml_model_id = ""
        if cfg.ml_enabled and self._ml_hook is not None and features is not None:
            ml_prob, grade, ml_override = self._apply_ml_overlay(grade, features)
            ml_model_id = getattr(self._ml_hook, "model_id", "") or ""

        # ── Finalize recommendations ──────────────────────────────
        skip = grade == SignalQualityGrade.F
        reduce = grade in (SignalQualityGrade.C, SignalQualityGrade.D)
        size_mult = cfg.size_by_grade.get(grade.value, 0.0)

        if not skip and not reasons:
            reasons.append(f"Grade {grade.value}: conviction={conviction:.3f}, composite={composite:.3f}")

        return SignalQualityScore(
            grade=grade,
            score=round(composite, 4),
            component_scores=components,
            skip_recommended=skip,
            reduce_size_recommended=reduce,
            size_multiplier=size_mult,
            reasons=reasons,
            warnings=warnings,
            ml_override=ml_override,
            ml_probability=ml_prob,
            ml_model_id=ml_model_id,
        )

    # ── Private helpers ───────────────────────────────────────────

    def _grade_from_conviction_and_composite(
        self,
        conviction: float,
        composite: float,
        cfg: QualityConfig,
    ) -> SignalQualityGrade:
        """Map conviction + composite score to a grade."""
        # Use the more conservative of the two
        score = min(conviction * 1.5, composite)  # conviction has less dynamic range
        if score >= cfg.min_conviction_for_aplus and composite >= 0.70:
            return SignalQualityGrade.A_PLUS
        if score >= cfg.min_conviction_for_a and composite >= 0.55:
            return SignalQualityGrade.A
        if score >= cfg.min_conviction_for_b and composite >= 0.40:
            return SignalQualityGrade.B
        if score >= cfg.min_conviction_for_c and composite >= 0.25:
            return SignalQualityGrade.C
        if composite >= 0.15:
            return SignalQualityGrade.D
        return SignalQualityGrade.F

    def _score_features(
        self, features: SignalFeatures
    ) -> tuple[float, list[str], list[str]]:
        """Score signal features; return (score, reasons, warnings)."""
        cfg = self._cfg
        score = 1.0
        reasons: list[str] = []
        warnings: list[str] = []

        # Lateness: signal has been above threshold too long
        if (not math.isnan(features.time_above_threshold_days)
                and features.time_above_threshold_days > cfg.max_time_above_threshold_days):
            score *= 0.5
            reasons.append(
                f"Late signal: z above threshold for "
                f"{features.time_above_threshold_days:.1f} days "
                f"(max {cfg.max_time_above_threshold_days})"
            )

        # Staleness: z-score not updating
        # (implies data or relationship is stale)
        if (not math.isnan(features.z_velocity)
                and abs(features.z_velocity) < 1e-6
                and not math.isnan(features.z_score)
                and abs(features.z_score) > 1.0):
            score *= 0.7
            warnings.append("Z-score appears stale (velocity ≈ 0)")

        # Extreme z (potential break)
        if (not math.isnan(features.z_score)
                and abs(features.z_score) > cfg.extreme_z_threshold):
            score *= 0.4
            reasons.append(f"Extreme z-score ({features.z_score:.1f}) may indicate structural break")

        # Failed reversions
        if features.failed_reversion_attempts >= cfg.max_failed_reversion_attempts:
            score *= 0.5
            reasons.append(
                f"{features.failed_reversion_attempts} consecutive failed reversions"
            )

        # Correlation health
        if (not math.isnan(features.corr_20d)
                and features.corr_20d < cfg.min_correlation_for_quality):
            score *= 0.6
            warnings.append(f"Rolling correlation low ({features.corr_20d:.2f})")

        # Beta instability
        if (not math.isnan(features.beta_cv)
                and features.beta_cv > cfg.max_beta_cv_for_quality):
            score *= 0.7
            warnings.append(f"Hedge ratio unstable (β_cv={features.beta_cv:.2f})")

        # Vol regime check
        if (not math.isnan(features.spread_vol_regime)
                and features.spread_vol_regime > 2.0):
            score *= 0.75
            warnings.append(f"Spread vol elevated ({features.spread_vol_regime:.1f}x baseline)")

        # Break risk
        if (not math.isnan(features.break_risk)
                and features.break_risk > 0.6):
            score *= max(0.2, 1.0 - features.break_risk)
            reasons.append(f"Break risk elevated ({features.break_risk:.0%})")

        # Spread momentum veto — penalize if spread is accelerating away from mean
        # z_velocity > 0 means |z| is growing (bad timing for mean-reversion entry)
        z_velocity = features.z_velocity if not math.isnan(features.z_velocity) else None
        z_score_val = features.z_score if not math.isnan(features.z_score) else None
        if z_velocity is not None and z_score_val is not None:
            # velocity * sign(z) > 0 means spread is moving away from mean
            escape_momentum = float(z_velocity) * (1.0 if float(z_score_val) >= 0 else -1.0)
            if escape_momentum > 0.05:       # Spread accelerating away
                score *= 0.60        # 40% penalty for bad timing
                warnings.append(
                    f"Spread momentum veto: escape_momentum={escape_momentum:.3f} "
                    f"(spread accelerating away from mean)"
                )
                # Stronger penalty if acceleration is high
                if escape_momentum > 0.15:
                    score *= 0.70    # Additional 30% penalty for strong escape
                    reasons.append(
                        f"Strong escape momentum ({escape_momentum:.3f}): poor entry timing"
                    )

        return max(0.0, min(1.0, score)), reasons, warnings

    def _score_snapshot(
        self, snapshot: SpreadStateSnapshot
    ) -> tuple[float, list[str], list[str]]:
        """Score from a SpreadStateSnapshot when full features aren't available."""
        cfg = self._cfg
        score = 1.0
        reasons: list[str] = []
        warnings: list[str] = []

        if (not math.isnan(snapshot.z_score)
                and abs(snapshot.z_score) > cfg.extreme_z_threshold):
            score *= 0.4
            reasons.append(f"Extreme z-score ({snapshot.z_score:.1f})")

        if (not math.isnan(snapshot.correlation)
                and snapshot.correlation < cfg.min_correlation_for_quality):
            score *= 0.6
            warnings.append(f"Low correlation ({snapshot.correlation:.2f})")

        if (not math.isnan(snapshot.beta_cv)
                and snapshot.beta_cv > cfg.max_beta_cv_for_quality):
            score *= 0.7
            warnings.append(f"Unstable hedge ratio (β_cv={snapshot.beta_cv:.2f})")

        return max(0.0, min(1.0, score)), reasons, warnings

    def _apply_ml_overlay(
        self,
        rule_grade: SignalQualityGrade,
        features: SignalFeatures,
    ) -> tuple[float, SignalQualityGrade, bool]:
        """Apply ML meta-label overlay. Returns (ml_prob, new_grade, was_overridden)."""
        cfg = self._cfg

        # Never upgrade grade F (rule-based floor is inviolable)
        if rule_grade == SignalQualityGrade.F:
            return np.nan, SignalQualityGrade.F, False

        try:
            prob = self._ml_hook.predict_success_probability(features)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("MetaLabel ML hook failed: %s", exc)
            return np.nan, rule_grade, False

        if math.isnan(prob):
            return np.nan, rule_grade, False

        # ML can upgrade grade D to C if probability is high enough
        if (cfg.ml_can_skip_d_grade
                and rule_grade == SignalQualityGrade.D
                and prob >= cfg.ml_min_probability_to_upgrade):
            new_grade = SignalQualityGrade.C
            logger.debug("MetaLabel: upgraded D → C (prob=%.2f)", prob)
            return prob, new_grade, True

        # ML can upgrade by at most 1 grade step
        if prob >= cfg.ml_min_probability_to_upgrade:
            idx = self._GRADE_ORDER.index(rule_grade)
            new_idx = min(idx + cfg.ml_max_grade_upgrade, len(self._GRADE_ORDER) - 1)
            new_grade = self._GRADE_ORDER[new_idx]
            if new_grade != rule_grade:
                logger.debug("MetaLabel: upgraded %s → %s (prob=%.2f)",
                             rule_grade.value, new_grade.value, prob)
                return prob, new_grade, True

        return prob, rule_grade, False


__all__ = [
    "SignalQualityScore",
    "MetaLabelProtocol",
    "QualityConfig",
    "SignalQualityEngine",
]
