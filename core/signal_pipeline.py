# -*- coding: utf-8 -*-
"""
core/signal_pipeline.py
=======================
Canonical signal generation pipeline.

ADR-006: This module is the canonical integration point between the z-score
computation layer (common/signal_generator.py) and the portfolio intent layer
(portfolio/allocator.py via core/intents.py).

The pipeline:
    spread z-score + prices
        -> RegimeEngine.classify()
        -> ThresholdEngine.compute(regime)
        -> SignalQualityEngine.assess(conviction, regime)
        -> TradeLifecycleStateMachine.can_enter/can_add
        -> EntryIntent / ExitIntent / None

Integration Status (2026-04-01):
    This pipeline is the **default** backtester signal path (``use_signal_pipeline``
    defaults to ``True`` in optimization_backtester.py).  The backtester calls
    ``evaluate_bar()`` for each bar unless ``use_signal_pipeline=False`` is explicitly
    set to revert to legacy z-score threshold logic.

    See also: ``evaluate()`` for the full evaluation with regime features from
    price series, and ``evaluate_bar()`` for the lightweight bar-by-bar path
    used by the optimisation backtester.

Usage (full evaluation):
    pipeline = SignalPipeline(pair_id=PairId("AAPL", "MSFT"))
    decision = pipeline.evaluate(
        z_score=2.3,
        spread_series=spread,
        prices_x=px,
        prices_y=py,
        as_of=datetime.utcnow(),
    )
    if decision.intent is not None and not decision.blocked:
        # Route to PortfolioAllocator.allocate([decision.intent])
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

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# OUTPUT CONTRACTS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SignalDecision:
    """
    Output of the SignalPipeline for a single evaluation.

    Downstream consumer (PortfolioAllocator) should check:
        - decision.blocked -> if True, do not route to portfolio
        - decision.intent  -> EntryIntent or ExitIntent to route
        - decision.block_reasons -> human-readable rationale if blocked
    """
    pair_id: PairId
    as_of: datetime
    z_score: float
    regime: str                            # RegimeLabel.value
    quality_grade: str                     # SignalQualityGrade.value e.g. "A", "B", "F"
    intent: Optional[BaseIntent]           # EntryIntent, ExitIntent, or None
    blocked: bool                          # True = do not route to portfolio
    block_reasons: list[str] = field(default_factory=list)
    size_multiplier: float = 1.0           # From signal quality engine
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


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
        self._last_decision: Optional[SignalDecision] = None

        # Cached regime state for bar-by-bar use (refreshed periodically)
        self._cached_regime: RegimeLabel = RegimeLabel.UNKNOWN
        self._cached_regime_confidence: float = 0.0
        self._cached_threshold: ThresholdSet = ThresholdSet()
        self._cached_quality_grade: str = "UNKNOWN"
        self._cached_size_mult: float = 1.0
        self._cached_blocked: bool = False
        self._regime_bar_counter: int = 0

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
    ) -> SignalDecision:
        """
        Evaluate signal for this pair at the current moment.

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
            Mean-reversion quality score [0, 1].
        half_life : float, optional
            Current estimated half-life of spread.
        correlation : float, optional
            Current rolling correlation.
        """
        if as_of is None:
            as_of = datetime.utcnow()

        block_reasons: list[str] = []
        warnings: list[str] = []

        # ── Step 1: Regime Classification ────────────────────────
        regime_label, regime_confidence = self._classify_regime(
            spread_series, prices_x, prices_y, as_of, warnings,
        )

        # ── Step 2: Threshold Computation ────────────────────────
        threshold_set = self._compute_thresholds(
            regime_label, conviction, half_life, warnings,
        )

        # ── Step 3: Signal Quality Assessment ────────────────────
        quality_result = self._assess_quality(
            conviction, mr_score, regime_label, block_reasons, warnings,
        )

        # ── Step 4: Lifecycle Check ──────────────────────────────
        can_enter = self._lifecycle.can_enter()
        lifecycle_state = self._lifecycle.state

        # ── Step 5: Intent Generation ────────────────────────────
        intent: Optional[BaseIntent] = None
        blocked = len(block_reasons) > 0

        if not blocked and can_enter:
            abs_z = abs(z_score)
            if abs_z >= threshold_set.entry_z:
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
                    entry_z_threshold=threshold_set.entry_z,
                    exit_z_target=threshold_set.exit_z,
                    stop_z=threshold_set.stop_z,
                    expected_half_life_days=half_life or 20.0,
                    rationale=[
                        f"z={z_score:.2f} crossed entry_z={threshold_set.entry_z:.2f}",
                        f"regime={regime_label.value}, quality={quality_result.grade.value if quality_result else 'N/A'}",
                    ],
                )
        elif self._lifecycle.is_position_active():
            abs_z = abs(z_score)
            if abs_z <= threshold_set.exit_z:
                intent = ExitIntent(
                    pair_id=self.pair_id,
                    confidence=conviction,
                    exit_reasons=[ExitReason.MEAN_REVERSION_COMPLETE],
                    z_score=z_score,
                    rationale=[
                        f"z={z_score:.2f} reverted below exit_z={threshold_set.exit_z:.2f}",
                    ],
                )
            elif abs_z >= threshold_set.stop_z:
                intent = ExitIntent(
                    pair_id=self.pair_id,
                    confidence=conviction,
                    exit_reasons=[ExitReason.ADVERSE_EXCURSION_STOP],
                    z_score=z_score,
                    rationale=[
                        f"z={z_score:.2f} exceeded stop_z={threshold_set.stop_z:.2f}",
                    ],
                )

        size_mult = (
            quality_result.size_multiplier if quality_result is not None else 1.0
        )

        decision = SignalDecision(
            pair_id=self.pair_id,
            as_of=as_of,
            z_score=z_score,
            regime=regime_label.value,
            quality_grade=(
                quality_result.grade.value
                if quality_result is not None
                else "UNKNOWN"
            ),
            intent=intent,
            blocked=blocked,
            block_reasons=block_reasons,
            size_multiplier=size_mult,
            warnings=warnings,
            metadata={
                "regime_confidence": regime_confidence,
                "entry_z": threshold_set.entry_z,
                "exit_z": threshold_set.exit_z,
                "stop_z": threshold_set.stop_z,
                "lifecycle_state": (
                    lifecycle_state.value
                    if hasattr(lifecycle_state, "value")
                    else str(lifecycle_state)
                ),
                "half_life": half_life,
                "correlation": correlation,
            },
        )

        self._last_decision = decision
        return decision

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
    # INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────────

    def _classify_regime(
        self,
        spread_series: pd.Series,
        prices_x: pd.Series,
        prices_y: pd.Series,
        as_of: datetime,
        warnings: list[str],
    ) -> tuple[RegimeLabel, float]:
        try:
            regime_features = build_regime_features(
                spread=spread_series,
                prices_x=prices_x,
                prices_y=prices_y,
                as_of=as_of,
            )
            regime_result = self._regime_engine.classify(regime_features)
            return regime_result.regime, regime_result.confidence
        except Exception as e:
            logger.warning(
                "RegimeEngine failed for %s: %s — defaulting to UNKNOWN",
                self.pair_id, e,
            )
            warnings.append(f"Regime classification failed: {e}")
            return RegimeLabel.UNKNOWN, 0.0

    def _compute_thresholds(
        self,
        regime: RegimeLabel,
        confidence: float,
        half_life: Optional[float],
        warnings: list[str],
    ) -> ThresholdSet:
        try:
            return self._threshold_engine.compute(
                regime=regime,
                signal_confidence=confidence,
                half_life_days=half_life if half_life is not None else float("nan"),
            )
        except Exception as e:
            logger.warning(
                "ThresholdEngine failed for %s: %s — using static defaults",
                self.pair_id, e,
            )
            warnings.append(f"Threshold computation failed: {e}")
            return ThresholdSet()

    def _assess_quality(
        self,
        conviction: float,
        mr_score: float,
        regime: RegimeLabel,
        block_reasons: list[str],
        warnings: list[str],
    ):
        try:
            quality_result = self._quality_engine.assess(
                conviction=conviction,
                mr_score=mr_score,
                regime=regime,
            )
            if quality_result.skip_recommended:
                block_reasons.append(
                    f"Signal quality grade {quality_result.grade.value}"
                    " — skip recommended"
                )
            return quality_result
        except Exception as e:
            logger.warning(
                "SignalQualityEngine failed for %s: %s", self.pair_id, e,
            )
            warnings.append(f"Quality assessment failed: {e}")
            return None

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
        block_reasons: list[str] = []

        regime, confidence = self._classify_regime(
            spread_series, prices_x, prices_y,
            datetime.utcnow(), warnings,
        )
        threshold_set = self._compute_thresholds(
            regime, conviction, half_life, warnings,
        )
        quality_result = self._assess_quality(
            conviction, mr_score, regime, block_reasons, warnings,
        )

        self._cached_regime = regime
        self._cached_regime_confidence = confidence
        self._cached_threshold = threshold_set
        self._cached_blocked = len(block_reasons) > 0
        if quality_result is not None:
            self._cached_quality_grade = quality_result.grade.value
            self._cached_size_mult = quality_result.size_multiplier
        else:
            self._cached_quality_grade = "UNKNOWN"
            self._cached_size_mult = 1.0

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
