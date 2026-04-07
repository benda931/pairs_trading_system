# -*- coding: utf-8 -*-
"""
core/signal_pipeline.py
=======================
Canonical signal generation pipeline.

ADR-006 / ADR-007: This module is the canonical integration point between the
z-score computation layer (common/signal_generator.py) and the portfolio intent
layer (portfolio/allocator.py via core/intents.py).

The pipeline (5 layers):
    Layer 0 — Input validation (NaN z-score → immediate hard block)
    Layer 1 — RegimeEngine.classify()    → RegimeContext
    Layer 2 — ThresholdEngine.compute()  → ThresholdContext
    Layer 3 — SignalQualityEngine.assess() → QualityVerdict
    Layer 4 — Intent generation          → SignalEnvelope

Output: ``SignalEnvelope`` (from ``core.signal_contracts``) — a fully typed,
immutable record with NO untyped ``metadata: dict``.  All contextual data is
in typed sub-contracts: regime_ctx, threshold_ctx, quality, soft, advisory.

Layer separation:
    hard_blocked   — binary gates; ANY one True = intent suppressed
    soft           — continuous adjustments (size mult, regime boost)
    advisory       — research enrichments, NEVER used in execution

Integration Status (2026-04-07):
    - ``evaluate()``      — full evaluation for live/portfolio-intent path
    - ``evaluate_bar()``  — lightweight bar-by-bar path for backtester
    - ``SignalDecision``  — DEPRECATED alias for ``SignalEnvelope`` (removed).
      Consumers must import ``SignalEnvelope`` from ``core.signal_contracts``.

Usage (full evaluation):
    pipeline = SignalPipeline(pair_id=PairId("AAPL", "MSFT"))
    env = pipeline.evaluate(
        z_score=2.3,
        spread_series=spread,
        prices_x=px,
        prices_y=py,
        as_of=datetime.utcnow(),
    )
    if env.is_actionable:
        # Route to PortfolioAllocator.allocate([env.intent])
        pass

Usage (backtester bar-by-bar):
    decision = pipeline.evaluate_bar(
        z_score=2.3,
        current_pos=0.0,
        holding_days=0,
    )
    # decision.action: +1 (long spread), -1 (short spread), 0 (flat/exit)
    # decision.entry_z, decision.exit_z, decision.stop_z: adaptive thresholds
    # decision.regime: current regime label
    # decision.quality_grade: signal quality grade
    # decision.blocked: True if quality/regime blocks the trade
"""

from __future__ import annotations

import logging
import threading
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import pandas as pd

from core.contracts import (
    PairId, RegimeLabel, SignalDirection, ExitReason, SignalQualityGrade,
    IntentAction,
)
from core.intents import EntryIntent, ExitIntent, BaseIntent
from core.threshold_engine import ThresholdEngine, ThresholdConfig, ThresholdSet
from core.regime_engine import RegimeEngine, RegimeFeatureSet, build_regime_features
from core.signal_quality import SignalQualityEngine, QualityConfig
from core.lifecycle import TradeLifecycleStateMachine, LifecycleRegistry, CooldownPolicy
from core.signals_engine import compute_mr_score
from core.signal_contracts import (
    SignalEnvelope,
    RegimeContext,
    ThresholdContext,
    QualityVerdict,
    SoftSignalModifiers,
    AdvisoryOverlays,
    HardBlockCode,
    make_run_id,
    make_blocked_envelope,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# OUTPUT CONTRACTS
# ═══════════════════════════════════════════════════════════════════

# SignalDecision is DEPRECATED — use SignalEnvelope from core.signal_contracts.
#
# The old class had:
#   blocked: bool              → SignalEnvelope.hard_blocked
#   block_reasons: list[str]   → SignalEnvelope.hard_block_reasons
#   metadata: dict             → SignalEnvelope.regime_ctx / threshold_ctx / quality / advisory
#   size_multiplier: float     → SignalEnvelope.soft.net_size_multiplier
#   regime: str                → SignalEnvelope.regime  (property)
#   quality_grade: str         → SignalEnvelope.quality_grade  (property)
#
# Consumers that imported ``SignalDecision`` from this module must be updated:
#     OLD: from core.signal_pipeline import SignalDecision
#     NEW: from core.signal_contracts import SignalEnvelope as SignalDecision
#
# The alias below preserves import compatibility during the migration window.
SignalDecision = SignalEnvelope


@dataclass
class BarDecision:
    """
    Lightweight decision for bar-by-bar backtester usage.

    This is the bridge between the canonical pipeline and the vectorised
    backtester.  It carries enough information for the backtester to set
    position, while still passing through regime/quality/threshold logic.
    """
    action: float           # +1 = long spread, -1 = short spread, 0 = flat/exit
    entry_z: float          # Adaptive entry threshold used
    exit_z: float           # Adaptive exit threshold used
    stop_z: float           # Adaptive stop threshold used
    regime: str             # RegimeLabel.value
    quality_grade: str      # SignalQualityGrade.value
    blocked: bool           # True = quality/regime blocks this trade
    size_multiplier: float  # From signal quality engine (advisory)


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

class SignalPipeline:
    """
    Canonical signal generation pipeline (ADR-006).

    Wires: z-score -> regime -> threshold -> quality -> lifecycle -> intent

    Thread safety: NOT thread-safe. Instantiate one per pair or use a registry.
    """

    def __init__(
        self,
        pair_id: PairId,
        *,
        threshold_config: Optional[ThresholdConfig] = None,
        quality_config: Optional[QualityConfig] = None,
        cooldown_policy: Optional[CooldownPolicy] = None,
        ml_regime_hook=None,   # Optional[RegimeClassifierHookProtocol]
        ml_quality_hook=None,  # Optional meta-label hook
    ):
        self.pair_id = pair_id
        self._threshold_engine = ThresholdEngine.from_config(
            threshold_config or ThresholdConfig()
        )
        self._regime_engine = RegimeEngine(ml_hook=ml_regime_hook)
        self._quality_engine = SignalQualityEngine(
            config=quality_config or QualityConfig(),
            ml_hook=ml_quality_hook,
        )
        self._lifecycle = TradeLifecycleStateMachine(
            pair_id=pair_id, cooldown_policy=cooldown_policy,
        )
        self._last_decision: Optional[SignalEnvelope] = None

        # Cached regime state for bar-by-bar use (refreshed periodically)
        self._cached_regime: RegimeLabel = RegimeLabel.UNKNOWN
        self._cached_regime_confidence: float = 0.0
        self._cached_regime_ctx: RegimeContext = RegimeContext(fallback_used=True)
        self._cached_threshold: ThresholdSet = ThresholdSet()
        self._cached_threshold_ctx: ThresholdContext = ThresholdContext(fallback_used=True)
        self._cached_quality: QualityVerdict = QualityVerdict(fallback_used=True)
        self._cached_quality_grade: str = "UNKNOWN"
        self._cached_size_mult: float = 1.0
        self._cached_blocked: bool = False
        self._regime_bar_counter: int = 0

        # Regime transition detection — track the first N bars after a
        # transition into MEAN_REVERTING, which are the highest-IC entry window.
        self._prev_regime: Optional[RegimeLabel] = None
        self._bars_since_transition: int = 999
        self._transition_boost_window: int = 10   # bars after transition with boosted confidence

    # ──────────────────────────────────────────────────────────────
    # FULL EVALUATION (for live / portfolio-intent path)
    # ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        z_score: float,
        spread_series: pd.Series,
        prices_x: pd.Series,
        prices_y: pd.Series,
        as_of: Optional[datetime] = None,
        *,
        conviction: float = 0.5,
        mr_score: float = 0.5,
        half_life: Optional[float] = None,
        correlation: Optional[float] = None,
    ) -> SignalEnvelope:
        """
        Evaluate signal for this pair at the current moment.

        Returns a fully-typed ``SignalEnvelope`` (from ``core.signal_contracts``)
        with no untyped metadata dict.  All contextual data is in the typed
        sub-contracts: regime_ctx, threshold_ctx, quality, soft, advisory.

        Parameters
        ----------
        z_score : float
            Current spread z-score (computed externally by spread constructor)
        spread_series : pd.Series
            Recent spread values for regime feature computation
        prices_x, prices_y : pd.Series
            Price series for the two legs (used for regime features)
        as_of : datetime, optional
            Point-in-time anchor. Defaults to utcnow for live use.
        conviction : float
            Signal strength [0, 1]. From spread model or ML.
        mr_score : float
            Mean-reversion quality score [0, 1] (recomputed internally).
        half_life : float, optional
            Current estimated half-life of spread.
        correlation : float, optional
            Current rolling correlation (passed through to advisory layer).
        """
        if as_of is None:
            as_of = datetime.utcnow()

        run_id = make_run_id()
        hard_block_codes: list[str] = []
        hard_block_reasons: list[str] = []
        warnings: list[str] = []

        # ── Layer 0: Input Validation ─────────────────────────────
        if np.isnan(z_score):
            return make_blocked_envelope(
                pair_id=self.pair_id,
                as_of=as_of,
                z_score=0.0,
                block_codes=[HardBlockCode.SPREAD_NAN],
                block_reasons=["z_score is NaN — cannot evaluate signal"],
                warnings=["z_score NaN at input"],
            )

        # ── Layer 1: Regime Classification ────────────────────────
        regime_ctx = self._classify_regime_typed(
            spread_series, prices_x, prices_y, as_of, warnings,
        )

        if not regime_ctx.is_tradable:
            hard_block_codes.append(
                HardBlockCode.REGIME_CRISIS.value
                if regime_ctx.regime == RegimeLabel.CRISIS
                else HardBlockCode.REGIME_BROKEN.value
            )
            hard_block_reasons.append(
                f"Regime {regime_ctx.regime.value} is not tradable for mean-reversion"
            )

        # ── Layer 2: Threshold Computation ────────────────────────
        threshold_ctx = self._compute_thresholds_typed(
            regime_ctx.regime, conviction, half_life, warnings,
        )

        # ── Layer 3: Signal Quality Assessment ────────────────────
        # Compute proper MR score from signal properties (replaces hardcoded 0.5)
        mr_score_computed = compute_mr_score(
            spread=spread_series,
            half_life=half_life if half_life is not None else float("nan"),
        )
        quality = self._assess_quality_typed(
            conviction, mr_score_computed, regime_ctx.regime,
            hard_block_codes, hard_block_reasons, warnings,
        )

        # ── Layer 3b: Regime Transition Detection ─────────────────
        # The first _transition_boost_window bars after transitioning into
        # MEAN_REVERTING carry the highest empirical IC.
        transition_boost_additive = 0.0
        bars_since = self._bars_since_transition
        if bars_since <= self._transition_boost_window:
            boost_frac = 1.0 - bars_since / self._transition_boost_window
            transition_boost_additive = 0.20 * boost_frac

        # ── Layer 3c: Escape Momentum Veto ────────────────────────
        escape_penalty = 0.0
        z_velocity_val = float("nan")
        if len(spread_series) >= 6:
            try:
                z_velocity_val = float(
                    (spread_series.iloc[-1] - spread_series.iloc[-6])
                    / max(spread_series.std(), 1e-8)
                )
                escape_momentum = z_velocity_val * np.sign(z_score)
                if escape_momentum > 0.15:
                    escape_penalty = 0.30
                elif escape_momentum > 0.05:
                    escape_penalty = 0.40
            except Exception:
                pass

        base_size_mult = quality.size_multiplier
        net_size_mult = max(0.0, min(
            1.0,
            (base_size_mult * (1.0 + transition_boost_additive)) * (1.0 - escape_penalty),
        ))

        soft = SoftSignalModifiers(
            net_size_multiplier=net_size_mult,
            regime_transition_boost=transition_boost_additive,
            bars_since_transition=bars_since,
            escape_momentum_penalty=escape_penalty,
            active_modifiers=[
                *(["regime_transition_boost"] if transition_boost_additive > 0 else []),
                *(["escape_momentum_penalty"] if escape_penalty > 0 else []),
            ],
        )

        # ── Layer 4: Lifecycle Check ──────────────────────────────
        can_enter = self._lifecycle.can_enter()
        lifecycle_state = self._lifecycle.state

        if not can_enter and not self._lifecycle.is_position_active():
            # Cooldown is active — only block new entries, not exits
            hard_block_codes.append(HardBlockCode.LIFECYCLE_COOLDOWN.value)
            hard_block_reasons.append(
                f"Lifecycle state {lifecycle_state} — cooldown prevents new entry"
            )

        # ── Layer 5: Intent Generation ────────────────────────────
        hard_blocked = len(hard_block_codes) > 0
        intent: Optional[BaseIntent] = None

        if not hard_blocked and can_enter:
            abs_z = abs(z_score)
            if abs_z >= threshold_ctx.entry_z:
                direction = (
                    SignalDirection.LONG_SPREAD
                    if z_score < 0
                    else SignalDirection.SHORT_SPREAD
                )
                intent = EntryIntent(
                    pair_id=self.pair_id,
                    confidence=conviction,
                    direction=direction,
                    z_score=z_score,
                    entry_z_threshold=threshold_ctx.entry_z,
                    exit_z_target=threshold_ctx.exit_z,
                    stop_z=threshold_ctx.stop_z,
                    expected_half_life_days=half_life or 20.0,
                    quality_grade=quality.grade.value,
                    regime=regime_ctx.regime.value,
                    size_multiplier=net_size_mult,
                    half_life_days=half_life or 20.0,
                    rationale=[
                        f"z={z_score:.2f} crossed entry_z={threshold_ctx.entry_z:.2f}",
                        f"regime={regime_ctx.regime.value}, "
                        f"quality={quality.grade.value}",
                        f"size_mult={net_size_mult:.3f}",
                    ],
                )
        elif self._lifecycle.is_position_active():
            abs_z = abs(z_score)
            if abs_z <= threshold_ctx.exit_z:
                intent = ExitIntent(
                    pair_id=self.pair_id,
                    confidence=conviction,
                    exit_reasons=[ExitReason.MEAN_REVERSION_COMPLETE],
                    z_score=z_score,
                    rationale=[
                        f"z={z_score:.2f} reverted below exit_z={threshold_ctx.exit_z:.2f}",
                    ],
                )
            elif abs_z >= threshold_ctx.stop_z:
                intent = ExitIntent(
                    pair_id=self.pair_id,
                    confidence=conviction,
                    exit_reasons=[ExitReason.ADVERSE_EXCURSION_STOP],
                    z_score=z_score,
                    rationale=[
                        f"z={z_score:.2f} exceeded stop_z={threshold_ctx.stop_z:.2f}",
                    ],
                )

        # ── Layer 6: Advisory Overlays ────────────────────────────
        advisory = AdvisoryOverlays(
            z_velocity=z_velocity_val,
            notes=[
                *(["regime_transition_active"] if transition_boost_additive > 0 else []),
                *(["correlation"] + [str(round(correlation, 4))]
                  if correlation is not None else []),
                f"lifecycle_state={lifecycle_state.value if hasattr(lifecycle_state, 'value') else lifecycle_state}",
            ],
        )

        envelope = SignalEnvelope(
            pair_id=self.pair_id,
            as_of=as_of,
            run_id=run_id,
            z_score=z_score,
            intent=intent,
            hard_blocked=hard_blocked,
            hard_block_reasons=hard_block_reasons,
            hard_block_codes=hard_block_codes,
            regime_ctx=regime_ctx,
            threshold_ctx=threshold_ctx,
            quality=quality,
            soft=soft,
            advisory=advisory,
            warnings=warnings,
        )

        self._last_decision = envelope
        return envelope

    # ──────────────────────────────────────────────────────────────
    # BAR-BY-BAR EVALUATION (for backtester integration)
    # ──────────────────────────────────────────────────────────────

    def evaluate_bar(
        self,
        z_score: float,
        current_pos: float,
        holding_days: int = 0,
        max_holding: int = 999,
        *,
        spread_series: Optional[pd.Series] = None,
        prices_x: Optional[pd.Series] = None,
        prices_y: Optional[pd.Series] = None,
        conviction: float = 0.5,
        mr_score: float = 0.5,
        half_life: Optional[float] = None,
        regime_refresh_every: int = 20,
    ) -> BarDecision:
        """
        Lightweight bar-by-bar signal decision for backtester use.

        Regime classification is expensive, so it is cached and refreshed
        every ``regime_refresh_every`` bars (default: 20 = ~1 month).

        Parameters
        ----------
        z_score : float
            Current z-score for this bar.
        current_pos : float
            Current position: +X = long spread, -X = short spread, 0 = flat.
        holding_days : int
            Days currently held (for time-stop logic).
        max_holding : int
            Maximum holding period before forced exit.
        spread_series, prices_x, prices_y : pd.Series, optional
            If provided, regime is re-classified on the refresh cycle.
        conviction, mr_score : float
            Signal strength inputs for quality assessment.
        half_life : float, optional
            Half-life estimate for threshold adaptation.
        regime_refresh_every : int
            Re-classify regime every N bars (default 20).

        Returns
        -------
        BarDecision
            Lightweight decision with action (+1/-1/0), adaptive thresholds,
            regime, quality grade, and blocking status.
        """
        # ── Regime refresh (periodic, not every bar) ─────────────
        self._regime_bar_counter += 1
        if (
            self._regime_bar_counter >= regime_refresh_every
            and spread_series is not None
            and prices_x is not None
            and prices_y is not None
        ):
            self._refresh_regime_cache(
                spread_series, prices_x, prices_y,
                conviction, mr_score, half_life,
            )
            self._regime_bar_counter = 0

        regime = self._cached_regime
        ts = self._cached_threshold
        blocked = self._cached_blocked

        # ── NaN / invalid z → flatten ────────────────────────────
        if np.isnan(z_score):
            return BarDecision(
                action=0.0,
                entry_z=ts.entry_z, exit_z=ts.exit_z, stop_z=ts.stop_z,
                regime=regime.value,
                quality_grade=self._cached_quality_grade,
                blocked=True,
                size_multiplier=self._cached_size_mult,
            )

        abs_z = abs(z_score)

        # ── No position: check entry ─────────────────────────────
        if current_pos == 0.0:
            if blocked:
                action = 0.0
            elif abs_z >= ts.entry_z:
                action = +1.0 if z_score < 0 else -1.0
            else:
                action = 0.0
            return BarDecision(
                action=action,
                entry_z=ts.entry_z, exit_z=ts.exit_z, stop_z=ts.stop_z,
                regime=regime.value,
                quality_grade=self._cached_quality_grade,
                blocked=blocked,
                size_multiplier=self._cached_size_mult,
            )

        # ── Has position: check exit conditions ──────────────────
        if current_pos > 0:
            exit_revert = (z_score >= -ts.exit_z)
            exit_stop = (z_score <= -ts.stop_z)
        else:
            exit_revert = (z_score <= ts.exit_z)
            exit_stop = (z_score >= ts.stop_z)

        exit_time = (holding_days >= max_holding)

        if exit_revert or exit_stop or exit_time:
            action = 0.0
        else:
            action = current_pos  # hold

        return BarDecision(
            action=action,
            entry_z=ts.entry_z, exit_z=ts.exit_z, stop_z=ts.stop_z,
            regime=regime.value,
            quality_grade=self._cached_quality_grade,
            blocked=blocked,
            size_multiplier=self._cached_size_mult,
        )

    # ──────────────────────────────────────────────────────────────
    # INTERNAL HELPERS — typed contract versions
    # ──────────────────────────────────────────────────────────────

    def _classify_regime_typed(
        self,
        spread_series: pd.Series,
        prices_x: pd.Series,
        prices_y: pd.Series,
        as_of: datetime,
        warnings: list[str],
    ) -> RegimeContext:
        """Layer 1: classify regime and return typed RegimeContext."""
        try:
            regime_features = build_regime_features(
                spread=spread_series,
                prices_x=prices_x,
                prices_y=prices_y,
                as_of=as_of,
            )
            regime_result = self._regime_engine.classify(regime_features)
            return RegimeContext(
                regime=regime_result.regime,
                confidence=regime_result.confidence,
                spread_vol=getattr(regime_features, "spread_vol", float("nan")),
                spread_trend=getattr(regime_features, "spread_trend", float("nan")),
                hurst_exp=getattr(regime_features, "hurst_exp", float("nan")),
            )
        except Exception as e:
            logger.warning(
                "RegimeEngine failed for %s: %s — defaulting to UNKNOWN",
                self.pair_id, e,
            )
            warnings.append(f"Regime classification failed: {e}")
            return RegimeContext(
                regime=RegimeLabel.UNKNOWN,
                confidence=0.0,
                fallback_used=True,
                layer_error=str(e),
            )

    def _compute_thresholds_typed(
        self,
        regime: RegimeLabel,
        confidence: float,
        half_life: Optional[float],
        warnings: list[str],
    ) -> ThresholdContext:
        """Layer 2: compute thresholds and return typed ThresholdContext."""
        try:
            ts = self._threshold_engine.compute(
                regime=regime,
                signal_confidence=confidence,
                half_life_days=half_life if half_life is not None else float("nan"),
            )
            return ThresholdContext(
                entry_z=ts.entry_z,
                exit_z=ts.exit_z,
                stop_z=ts.stop_z,
            )
        except Exception as e:
            logger.warning(
                "ThresholdEngine failed for %s: %s — using static defaults",
                self.pair_id, e,
            )
            warnings.append(f"Threshold computation failed: {e}")
            return ThresholdContext(fallback_used=True, layer_error=str(e))

    def _assess_quality_typed(
        self,
        conviction: float,
        mr_score: float,
        regime: RegimeLabel,
        hard_block_codes: list[str],
        hard_block_reasons: list[str],
        warnings: list[str],
    ) -> QualityVerdict:
        """Layer 3: assess quality and return typed QualityVerdict."""
        try:
            qr = self._quality_engine.assess(
                conviction=conviction,
                mr_score=mr_score,
                regime=regime,
            )
            if qr.skip_recommended:
                hard_block_codes.append(HardBlockCode.GRADE_F_SKIP.value)
                hard_block_reasons.append(
                    f"Signal quality grade {qr.grade.value} — skip recommended"
                )
            return QualityVerdict(
                grade=qr.grade,
                score=getattr(qr, "score", conviction),
                size_multiplier=qr.size_multiplier,
                mr_score=mr_score,
                regime_score=getattr(qr, "regime_score", 0.5),
                conviction=conviction,
                skip_recommended=qr.skip_recommended,
            )
        except Exception as e:
            logger.warning(
                "SignalQualityEngine failed for %s: %s", self.pair_id, e,
            )
            warnings.append(f"Quality assessment failed: {e}")
            return QualityVerdict(
                grade=SignalQualityGrade.F,
                skip_recommended=False,
                fallback_used=True,
                layer_error=str(e),
            )

    # ── Legacy helpers kept for evaluate_bar internal use ─────────

    def _classify_regime(
        self,
        spread_series: pd.Series,
        prices_x: pd.Series,
        prices_y: pd.Series,
        as_of: datetime,
        warnings: list[str],
    ) -> tuple[RegimeLabel, float]:
        ctx = self._classify_regime_typed(spread_series, prices_x, prices_y, as_of, warnings)
        return ctx.regime, ctx.confidence

    def _compute_thresholds(
        self,
        regime: RegimeLabel,
        confidence: float,
        half_life: Optional[float],
        warnings: list[str],
    ) -> ThresholdSet:
        ctx = self._compute_thresholds_typed(regime, confidence, half_life, warnings)
        # Reconstruct a ThresholdSet-compatible object (duck-typed)
        ts = ThresholdSet()
        ts.entry_z = ctx.entry_z
        ts.exit_z = ctx.exit_z
        ts.stop_z = ctx.stop_z
        return ts

    def _assess_quality(
        self,
        conviction: float,
        mr_score: float,
        regime: RegimeLabel,
        block_reasons: list[str],
        warnings: list[str],
    ):
        hard_codes: list[str] = []
        hard_reasons: list[str] = []
        verdict = self._assess_quality_typed(
            conviction, mr_score, regime, hard_codes, hard_reasons, warnings,
        )
        block_reasons.extend(hard_reasons)
        # Return a duck-typed wrapper for backward-compat code
        class _QR:
            grade = verdict.grade
            size_multiplier = verdict.size_multiplier
            skip_recommended = verdict.skip_recommended
        return _QR()

    def _refresh_regime_cache(
        self,
        spread_series: pd.Series,
        prices_x: pd.Series,
        prices_y: pd.Series,
        conviction: float,
        mr_score: float,
        half_life: Optional[float],
    ) -> None:
        """Re-compute regime, thresholds, and quality for bar-by-bar cache."""
        warnings: list[str] = []
        hard_codes: list[str] = []
        hard_reasons: list[str] = []

        regime_ctx = self._classify_regime_typed(
            spread_series, prices_x, prices_y,
            datetime.utcnow(), warnings,
        )
        threshold_ctx = self._compute_thresholds_typed(
            regime_ctx.regime, conviction, half_life, warnings,
        )
        mr_score_computed = compute_mr_score(
            spread=spread_series,
            half_life=half_life if half_life is not None else float("nan"),
        )
        quality = self._assess_quality_typed(
            conviction, mr_score_computed, regime_ctx.regime,
            hard_codes, hard_reasons, warnings,
        )

        # Detect regime transition into MEAN_REVERTING and manage boost counter.
        new_regime = regime_ctx.regime
        if self._prev_regime is not None and self._prev_regime != new_regime:
            if new_regime == RegimeLabel.MEAN_REVERTING:
                self._bars_since_transition = 0
                logger.info(
                    "Regime transition → MEAN_REVERTING for %s (boost window active)",
                    self.pair_id,
                )
            else:
                self._bars_since_transition = 999
        self._prev_regime = new_regime
        self._bars_since_transition += 1

        # Cache typed contexts for evaluate_bar
        self._cached_regime = regime_ctx.regime
        self._cached_regime_ctx = regime_ctx
        self._cached_regime_confidence = regime_ctx.confidence
        self._cached_threshold = ThresholdSet()
        self._cached_threshold.entry_z = threshold_ctx.entry_z
        self._cached_threshold.exit_z = threshold_ctx.exit_z
        self._cached_threshold.stop_z = threshold_ctx.stop_z
        self._cached_threshold_ctx = threshold_ctx
        self._cached_quality = quality
        self._cached_blocked = bool(hard_codes)

        self._cached_quality_grade = quality.grade.value
        base_size_mult = quality.size_multiplier

        # Apply transition boost to cached size multiplier
        if self._bars_since_transition <= self._transition_boost_window:
            boost_frac = 1.0 - self._bars_since_transition / self._transition_boost_window
            transition_boost = 1.0 + 0.20 * boost_frac
            self._cached_size_mult = min(1.0, base_size_mult * transition_boost)
        else:
            self._cached_size_mult = base_size_mult

    @property
    def lifecycle_state(self):
        return self._lifecycle.state


# ═══════════════════════════════════════════════════════════════════
# REGISTRY
# ═══════════════════════════════════════════════════════════════════

class SignalPipelineRegistry:
    """
    Registry of SignalPipelines keyed by PairId.

    Manages one pipeline per active pair. Thread-safe for get/create.
    """

    def __init__(self):
        self._pipelines: dict[str, SignalPipeline] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self, pair_id: PairId, **pipeline_kwargs
    ) -> SignalPipeline:
        key = str(pair_id)
        with self._lock:
            if key not in self._pipelines:
                self._pipelines[key] = SignalPipeline(
                    pair_id=pair_id, **pipeline_kwargs,
                )
            return self._pipelines[key]

    def remove(self, pair_id: PairId) -> bool:
        key = str(pair_id)
        with self._lock:
            if key in self._pipelines:
                del self._pipelines[key]
                return True
            return False

    @property
    def active_pairs(self) -> list[PairId]:
        return [p.pair_id for p in self._pipelines.values()]


_registry: Optional[SignalPipelineRegistry] = None


def get_signal_pipeline_registry() -> SignalPipelineRegistry:
    global _registry
    if _registry is None:
        _registry = SignalPipelineRegistry()
    return _registry
