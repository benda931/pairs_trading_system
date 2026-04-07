# -*- coding: utf-8 -*-
"""
core/signal_contracts.py — Canonical Signal Spine Contracts
============================================================

ADR-007: Single canonical signal spine.

This module defines the **typed intermediate contracts** that flow through the
5-layer signal pipeline and the **SignalEnvelope** — the single canonical
output contract that replaces the untyped ``metadata: dict`` in the old
``signal_pipeline.SignalDecision``.

Layer model:
    Layer 0 — RawSignalInput        (z_score, spread_series, prices, as_of)
    Layer 1 — RegimeContext         (typed regime classification result)
    Layer 2 — ThresholdContext      (typed threshold computation result)
    Layer 3 — QualityVerdict        (typed quality assessment result)
    Layer 4 — SignalEnvelope        (canonical output: intent + all layers)

Layer separation within SignalEnvelope:
    hard_blocked         — binary gates; ANY one True = intent suppressed
    SoftSignalModifiers  — continuous adjustments that scale but don't gate
    AdvisoryOverlays     — display/research enrichments; NEVER used in execution
    intent               — the proposed action (None = nothing to do)

Design rules:
    1. SignalEnvelope is frozen (immutable after construction).
    2. All fields are typed; no ``metadata: dict`` in the output contract.
    3. Hard gates are evaluated before soft modifiers.
    4. Advisory overlays are populated last and cannot alter hard/soft layers.
    5. The pipeline writes a unique ``run_id`` per evaluate() call.
    6. Backward-compat shim: ``PipelineSignalRecord = SignalEnvelope``.

Migration note:
    - ``signal_pipeline.SignalDecision`` is renamed ``SignalEnvelope`` here.
    - ``core.intents.SignalDecision`` is the *rich agent/intent* contract and
      is unchanged — it wraps an intent with full context for the agent layer.
    - ``portfolio_bridge.py`` now consumes ``SignalEnvelope`` directly with
      fully-typed field access (no more ``object.__setattr__``).
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from core.contracts import (
    PairId,
    RegimeLabel,
    SignalQualityGrade,
)
from core.intents import BaseIntent


# ══════════════════════════════════════════════════════════════════
# LAYER TAGS — for audit trail and logging
# ══════════════════════════════════════════════════════════════════

class SignalLayer(str, enum.Enum):
    """Identifies which pipeline layer a warning, block, or note originated in."""
    INPUT_VALIDATION = "INPUT_VALIDATION"
    REGIME           = "REGIME"
    THRESHOLD        = "THRESHOLD"
    QUALITY          = "QUALITY"
    LIFECYCLE        = "LIFECYCLE"
    INTENT           = "INTENT"
    ADVISORY         = "ADVISORY"


class HardBlockCode(str, enum.Enum):
    """
    Machine-readable codes for hard gates.

    Hard gates are inviolable — they suppress the intent regardless of
    conviction, quality grade, or any soft modifier.
    """
    GRADE_F_SKIP            = "GRADE_F_SKIP"          # Quality engine grade F
    REGIME_CRISIS           = "REGIME_CRISIS"          # Regime is CRISIS
    REGIME_BROKEN           = "REGIME_BROKEN"          # Spread regime is BROKEN
    HALF_LIFE_TOO_SHORT     = "HALF_LIFE_TOO_SHORT"   # HL < min_half_life_days
    HALF_LIFE_TOO_LONG      = "HALF_LIFE_TOO_LONG"    # HL > max_half_life_days
    LIFECYCLE_COOLDOWN      = "LIFECYCLE_COOLDOWN"     # Cooldown period active
    LIFECYCLE_SUSPENDED     = "LIFECYCLE_SUSPENDED"    # Pair explicitly suspended
    ML_GOVERNANCE_BLOCK     = "ML_GOVERNANCE_BLOCK"    # GovernanceRouter vetoed
    DATA_INSUFFICIENT       = "DATA_INSUFFICIENT"      # Not enough bars
    SPREAD_NAN              = "SPREAD_NAN"             # z_score is NaN


# ══════════════════════════════════════════════════════════════════
# LAYER 1 — REGIME CONTEXT
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RegimeContext:
    """
    Typed output of Layer 1: RegimeEngine.classify().

    Replaces ad-hoc ``metadata["regime_confidence"]`` and ``metadata["regime"]``.
    """
    regime:        RegimeLabel = RegimeLabel.UNKNOWN
    confidence:    float       = 0.0
    # Feature subset used in classification (populated for observability)
    spread_vol:    float       = float("nan")
    spread_trend:  float       = float("nan")
    hurst_exp:     float       = float("nan")
    # True if classification failed and defaults are in use
    fallback_used: bool        = False
    layer_error:   str         = ""   # Non-empty if exception was caught

    @property
    def is_tradable(self) -> bool:
        """False for regimes where mean-reversion does not work."""
        return self.regime not in (
            RegimeLabel.CRISIS,
            RegimeLabel.BROKEN,
            RegimeLabel.UNKNOWN,
        )


# ══════════════════════════════════════════════════════════════════
# LAYER 2 — THRESHOLD CONTEXT
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ThresholdContext:
    """
    Typed output of Layer 2: ThresholdEngine.compute().

    Replaces ad-hoc ``metadata["entry_z"]``, ``metadata["exit_z"]``, etc.
    """
    entry_z:       float = 2.0
    exit_z:        float = 0.5
    stop_z:        float = 3.5
    vol_mult:      float = 1.0    # Volatility scaling applied to entry_z
    half_life_adj: float = 1.0    # Half-life adjustment factor applied
    fallback_used: bool  = False
    layer_error:   str   = ""


# ══════════════════════════════════════════════════════════════════
# LAYER 3 — QUALITY VERDICT
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class QualityVerdict:
    """
    Typed output of Layer 3: SignalQualityEngine.assess().

    Replaces ad-hoc quality_grade string and conviction float in metadata.
    """
    grade:              SignalQualityGrade = SignalQualityGrade.F
    score:              float              = 0.0   # [0, 1]
    size_multiplier:    float              = 1.0   # Scale factor for position sizing
    mr_score:           float              = 0.5   # Mean-reversion quality [0, 1]
    regime_score:       float              = 0.5   # Regime quality contribution
    conviction:         float              = 0.5   # Combined conviction [0, 1]
    skip_recommended:   bool               = False  # True → hard block triggered
    fallback_used:      bool               = False
    layer_error:        str                = ""

    @property
    def is_grade_f(self) -> bool:
        return self.grade == SignalQualityGrade.F


# ══════════════════════════════════════════════════════════════════
# LAYER 4a — SOFT SIGNAL MODIFIERS
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SoftSignalModifiers:
    """
    Continuous adjustments that scale the signal but do NOT gate it.

    These are applied AFTER hard gates have passed. They adjust position
    sizing and threshold tightening but cannot create a hard block.

    Downstream consumers (PortfolioAllocator, SizingEngine) may use any
    of these — they are always typed and never in a metadata dict.
    """
    # Net size multiplier after all soft modifiers combined
    net_size_multiplier: float = 1.0

    # Regime transition boost (first N bars in MEAN_REVERTING)
    regime_transition_boost:  float = 0.0      # Additive boost applied
    bars_since_transition:    int   = 999      # 999 = no recent transition

    # Spread momentum veto (penalty if z-score is diverging further)
    escape_momentum_penalty:  float = 0.0      # Fraction reduced (0 = no penalty)

    # GARCH vol scaling
    garch_vol_mult:           float = 1.0

    # Notes explaining which modifiers were active (for logging)
    active_modifiers:         List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# LAYER 4b — ADVISORY OVERLAYS (research-only, never used in execution)
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AdvisoryOverlays:
    """
    Research-grade enrichments that are NEVER used in execution logic.

    These are populated for display, analytics, and model development.
    Any attempt to use these in sizing, gating, or execution is a bug.

    Populated AFTER the intent is generated (cannot influence the decision).
    """
    correlation_dislocation_score: float = float("nan")  # Cross-sectional ranking
    ou_optimal_exit_z:             float = float("nan")   # OU-calibrated exit target
    variance_ratio:                float = float("nan")   # Lo-MacKinlay VR
    z_velocity:                    float = float("nan")   # dz/dt rolling 5-bar
    factor_beta_net:               float = float("nan")   # Net SPY beta of pair
    spread_momentum_score:         float = float("nan")   # Spread trend strength
    # Free-form notes for logging/display (no execution semantics)
    notes: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# CANONICAL PIPELINE OUTPUT: SignalEnvelope
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SignalEnvelope:
    """
    Canonical single-pipeline output for one pair at one evaluation point.

    This is the SINGLE canonical contract between:
        SignalPipeline.evaluate()  →  PortfolioAllocator / RiskEngine / Backtester

    Replaces the old ``signal_pipeline.SignalDecision`` which had an untyped
    ``metadata: dict`` field.  All contextual data is now fully typed in the
    five dedicated sub-contracts below.

    Layer structure:
        regime_ctx     — how the regime was classified (Layer 1)
        threshold_ctx  — which thresholds were computed (Layer 2)
        quality        — quality grade and conviction (Layer 3)
        soft           — continuous size/threshold adjustments (Layer 4a)
        advisory       — research enrichments, NEVER used in execution (Layer 4b)
        intent         — the proposed action (None = nothing to do)
        hard_blocked   — if True, intent MUST NOT be acted on

    Point-in-time reproducibility:
        run_id  — unique UUID per evaluate() call; links to pipeline audit log
        as_of   — the exact timestamp of evaluation (UTC)

    Usage:
        env = pipeline.evaluate(...)
        if not env.hard_blocked and env.intent is not None:
            allocator.submit(env.intent, quality=env.quality, soft=env.soft)

    Backward-compat alias:
        ``PipelineSignalRecord = SignalEnvelope`` — old consumers that imported
        ``SignalDecision`` from ``core.signal_pipeline`` must be updated to
        import ``SignalEnvelope`` from ``core.signal_contracts``.
    """
    # ── Identity ──────────────────────────────────────────────────
    pair_id: PairId
    as_of:   datetime
    run_id:  str            # Unique per evaluate() call; UUID4 string

    # ── Core signal ───────────────────────────────────────────────
    z_score: float
    intent:  Optional[BaseIntent]   # None = nothing to do this bar

    # ── Hard gate layer ───────────────────────────────────────────
    hard_blocked:       bool
    hard_block_reasons: List[str]      # Always populated when hard_blocked=True
    hard_block_codes:   List[str]      # HardBlockCode.value strings

    # ── Typed context layers ──────────────────────────────────────
    regime_ctx:     RegimeContext
    threshold_ctx:  ThresholdContext
    quality:        QualityVerdict

    # ── Soft modifiers (scale, don't gate) ───────────────────────
    soft: SoftSignalModifiers

    # ── Advisory overlays (display/research only) ─────────────────
    advisory: AdvisoryOverlays

    # ── Audit / observability ─────────────────────────────────────
    warnings:         List[str]
    pipeline_version: str = "2.0"

    # ── Convenience properties ────────────────────────────────────

    @property
    def regime(self) -> str:
        """Convenience accessor: regime label string."""
        return self.regime_ctx.regime.value

    @property
    def quality_grade(self) -> str:
        """Convenience accessor: quality grade string (e.g. 'A', 'B', 'F')."""
        return self.quality.grade.value

    @property
    def size_multiplier(self) -> float:
        """Net size multiplier from soft layer."""
        return self.soft.net_size_multiplier

    @property
    def is_actionable(self) -> bool:
        """True if intent should be routed to portfolio allocator."""
        return not self.hard_blocked and self.intent is not None

    # ── Backward-compat read properties (pre-ADR-007 consumers) ──

    @property
    def blocked(self) -> bool:
        """Alias for hard_blocked — backward-compat with old SignalDecision API."""
        return self.hard_blocked

    @property
    def block_reasons(self) -> List[str]:
        """Alias for hard_block_reasons — backward-compat."""
        return self.hard_block_reasons

    @property
    def metadata(self) -> dict:
        """
        Backward-compat: return a dict view of the typed sub-contracts.

        Legacy consumers that read ``decision.metadata["entry_z"]`` etc. still
        work.  DO NOT write to this dict — it is computed, not stored.

        Deprecated: access the typed sub-contracts directly instead.
        """
        d: dict = {
            "regime_confidence":  self.regime_ctx.confidence,
            "entry_z":            self.threshold_ctx.entry_z,
            "exit_z":             self.threshold_ctx.exit_z,
            "stop_z":             self.threshold_ctx.stop_z,
            "half_life":          self.quality.mr_score,  # approximate
            "z_velocity":         self.advisory.z_velocity,
            "ou_optimal_exit_z":  self.advisory.ou_optimal_exit_z,
        }
        # Merge advisory notes as free-form key
        if self.advisory.notes:
            d["advisory_notes"] = list(self.advisory.notes)
        if self.soft.active_modifiers:
            d["active_modifiers"] = list(self.soft.active_modifiers)
        return d

    def to_dict(self) -> dict:
        """Serialize to dict for logging, dashboard, and DB persistence."""
        return {
            "pair": self.pair_id.label,
            "as_of": self.as_of.isoformat(),
            "run_id": self.run_id,
            "z_score": round(self.z_score, 4),
            "intent": self.intent.to_dict() if self.intent else None,
            "hard_blocked": self.hard_blocked,
            "hard_block_reasons": self.hard_block_reasons,
            "hard_block_codes": self.hard_block_codes,
            "regime": self.regime_ctx.regime.value,
            "regime_confidence": round(self.regime_ctx.confidence, 4),
            "entry_z": self.threshold_ctx.entry_z,
            "exit_z": self.threshold_ctx.exit_z,
            "stop_z": self.threshold_ctx.stop_z,
            "quality_grade": self.quality.grade.value,
            "quality_score": round(self.quality.score, 4),
            "conviction": round(self.quality.conviction, 4),
            "mr_score": round(self.quality.mr_score, 4),
            "size_multiplier": round(self.soft.net_size_multiplier, 4),
            "regime_transition_boost": round(self.soft.regime_transition_boost, 4),
            "ou_optimal_exit_z": round(self.advisory.ou_optimal_exit_z, 4)
                if not (self.advisory.ou_optimal_exit_z != self.advisory.ou_optimal_exit_z)  # nan check
                else None,
            "z_velocity": round(self.advisory.z_velocity, 4)
                if not (self.advisory.z_velocity != self.advisory.z_velocity)
                else None,
            "warnings": self.warnings,
            "pipeline_version": self.pipeline_version,
        }


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def make_run_id() -> str:
    """Generate a unique pipeline run ID (UUID4 string)."""
    return str(uuid.uuid4())


def make_signal_decision(
    pair_id: "PairId",
    as_of: datetime,
    z_score: float,
    *,
    intent=None,
    blocked: bool = False,
    block_reasons: Optional[List[str]] = None,
    size_multiplier: float = 1.0,
    warnings: Optional[List[str]] = None,
    regime: str = "UNKNOWN",
    quality_grade: str = "C",
    metadata: Optional[dict] = None,
    **_kwargs,
) -> "SignalEnvelope":
    """
    Factory: construct a SignalEnvelope from old-style SignalDecision kwargs.

    This is the backward-compatibility bridge for tests and legacy code that
    construct ``SignalDecision(pair_id=..., blocked=..., regime=..., ...)``
    with the pre-ADR-007 API.

    All new code should construct SignalEnvelope directly with typed sub-contracts.

    Parameters mirror the old ``core.signal_pipeline.SignalDecision`` dataclass.
    """
    from core.contracts import RegimeLabel, SignalQualityGrade

    # Parse regime string → RegimeLabel
    try:
        regime_label = RegimeLabel(regime)
    except ValueError:
        regime_label = RegimeLabel.UNKNOWN

    # Parse quality_grade string → SignalQualityGrade
    try:
        grade = SignalQualityGrade(quality_grade)
    except (ValueError, KeyError):
        grade = SignalQualityGrade.C

    # Extract optional metadata fields
    meta = metadata or {}
    entry_z = meta.get("entry_z", 2.0)
    exit_z = meta.get("exit_z", 0.5)
    stop_z = meta.get("stop_z", 3.5)
    regime_conf = meta.get("regime_confidence", 0.0)

    return SignalEnvelope(
        pair_id=pair_id,
        as_of=as_of,
        run_id=make_run_id(),
        z_score=z_score,
        intent=intent,
        hard_blocked=blocked,
        hard_block_reasons=list(block_reasons or []),
        hard_block_codes=[HardBlockCode.GRADE_F_SKIP.value] if blocked else [],
        regime_ctx=RegimeContext(
            regime=regime_label,
            confidence=regime_conf,
        ),
        threshold_ctx=ThresholdContext(
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
        ),
        quality=QualityVerdict(
            grade=grade,
            size_multiplier=size_multiplier,
            conviction=0.5,
        ),
        soft=SoftSignalModifiers(net_size_multiplier=size_multiplier),
        advisory=AdvisoryOverlays(),
        warnings=list(warnings or []),
    )


def make_blocked_envelope(
    pair_id: PairId,
    as_of: datetime,
    z_score: float,
    block_codes: list[HardBlockCode],
    block_reasons: list[str],
    *,
    regime_ctx: Optional[RegimeContext] = None,
    threshold_ctx: Optional[ThresholdContext] = None,
    quality: Optional[QualityVerdict] = None,
    warnings: Optional[list[str]] = None,
) -> SignalEnvelope:
    """
    Construct a fully-blocked SignalEnvelope with minimal context.

    Use this factory when a hard gate fires before all layers have been
    evaluated (e.g., data insufficient, spread NaN).
    """
    return SignalEnvelope(
        pair_id=pair_id,
        as_of=as_of,
        run_id=make_run_id(),
        z_score=z_score,
        intent=None,
        hard_blocked=True,
        hard_block_reasons=block_reasons,
        hard_block_codes=[c.value for c in block_codes],
        regime_ctx=regime_ctx or RegimeContext(fallback_used=True),
        threshold_ctx=threshold_ctx or ThresholdContext(fallback_used=True),
        quality=quality or QualityVerdict(fallback_used=True),
        soft=SoftSignalModifiers(),
        advisory=AdvisoryOverlays(),
        warnings=warnings or [],
    )


# ══════════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBILITY SHIM
# ══════════════════════════════════════════════════════════════════

# Old code that did:
#   from core.signal_pipeline import SignalDecision
# should be updated to:
#   from core.signal_contracts import SignalEnvelope
#
# During the transition, the alias below allows old imports to work
# WITHOUT semantic guarantees — the field names have changed.
PipelineSignalRecord = SignalEnvelope


__all__ = [
    # Enums
    "SignalLayer",
    "HardBlockCode",
    # Layer contracts
    "RegimeContext",
    "ThresholdContext",
    "QualityVerdict",
    "SoftSignalModifiers",
    "AdvisoryOverlays",
    # Canonical envelope
    "SignalEnvelope",
    "PipelineSignalRecord",
    # Helpers & factories
    "make_run_id",
    "make_blocked_envelope",
    "make_signal_decision",   # backward-compat factory for old SignalDecision API
]
