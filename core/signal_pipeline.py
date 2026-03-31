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
        → RegimeEngine.classify()
        → ThresholdEngine.compute(regime)
        → SignalQualityEngine.assess(conviction, regime)
        → TradeLifecycleStateMachine.can_enter/can_add
        → EntryIntent / ExitIntent / None

Integration Status (2026-03-31):
    This pipeline is implemented and tested. It is NOT yet called from the
    backtester or live trading path. See docs/remediation/remediation_ledger.md:P1-PIPE.

Usage:
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
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import pandas as pd

from core.contracts import PairId
from core.intents import EntryIntent, ExitIntent, BaseIntent
from core.threshold_engine import ThresholdEngine, ThresholdConfig
from core.regime_engine import RegimeEngine, RegimeFeatureSet, build_regime_features
from core.signal_quality import SignalQualityEngine, QualityConfig
from core.lifecycle import TradeLifecycleStateMachine, LifecycleRegistry, CooldownPolicy

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class SignalDecision:
    """
    Output of the SignalPipeline for a single evaluation.

    Downstream consumer (PortfolioAllocator) should check:
        - decision.blocked → if True, do not route to portfolio
        - decision.intent  → EntryIntent or ExitIntent to route
        - decision.block_reasons → human-readable rationale if blocked
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


class SignalPipeline:
    """
    Canonical signal generation pipeline (ADR-006).

    Wires: z-score → regime → threshold → quality → lifecycle → intent

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
        self._threshold_engine = ThresholdEngine.from_config(threshold_config or ThresholdConfig())
        self._regime_engine = RegimeEngine(ml_hook=ml_regime_hook)
        self._quality_engine = SignalQualityEngine(
            config=quality_config or QualityConfig(),
            ml_hook=ml_quality_hook,
        )
        self._lifecycle = TradeLifecycleStateMachine(pair_id=pair_id, cooldown_policy=cooldown_policy)
        self._last_decision: Optional[SignalDecision] = None

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
            Current estimated half-life of spread (informational; stored in metadata).
        correlation : float, optional
            Current rolling correlation (informational; stored in metadata).
        """
        if as_of is None:
            as_of = datetime.utcnow()

        block_reasons: list[str] = []
        warnings: list[str] = []

        # ── Step 1: Regime Classification ──────────────────────────────────
        try:
            regime_features = build_regime_features(
                spread=spread_series,
                prices_x=prices_x,
                prices_y=prices_y,
                as_of=as_of,
            )
            regime_result = self._regime_engine.classify(regime_features)
            regime_label = regime_result.regime
            regime_confidence = regime_result.confidence
        except Exception as e:
            logger.warning(
                "RegimeEngine failed for %s: %s — defaulting to UNKNOWN", self.pair_id, e
            )
            from core.contracts import RegimeLabel
            regime_label = RegimeLabel.UNKNOWN
            regime_confidence = 0.0
            warnings.append(f"Regime classification failed: {e}")

        # ── Step 2: Threshold Computation ──────────────────────────────────
        try:
            threshold_set = self._threshold_engine.compute(
                regime=regime_label,
                spread_series=spread_series,
                as_of=as_of,
            )
        except Exception as e:
            logger.warning(
                "ThresholdEngine failed for %s: %s — using static defaults", self.pair_id, e
            )
            from core.threshold_engine import ThresholdSet
            threshold_set = ThresholdSet(entry_z=2.0, exit_z=0.5, stop_z=3.5)
            warnings.append(f"Threshold computation failed: {e}")

        # ── Step 3: Signal Quality Assessment ──────────────────────────────
        try:
            quality_result = self._quality_engine.assess(
                conviction=conviction,
                mr_score=mr_score,
                regime=regime_label,
            )
            if quality_result.skip_recommended:
                block_reasons.append(
                    f"Signal quality grade {quality_result.grade.value} — skip recommended"
                )
        except Exception as e:
            logger.warning("SignalQualityEngine failed for %s: %s", self.pair_id, e)
            warnings.append(f"Quality assessment failed: {e}")
            quality_result = None

        # ── Step 4: Lifecycle Check ─────────────────────────────────────────
        can_enter = self._lifecycle.can_enter()
        lifecycle_state = self._lifecycle.state

        # ── Step 5: Intent Generation ───────────────────────────────────────
        intent: Optional[BaseIntent] = None
        blocked = len(block_reasons) > 0

        if not blocked and can_enter:
            abs_z = abs(z_score)
            if abs_z >= threshold_set.entry_z:
                direction = "long_x" if z_score < 0 else "short_x"
                intent = EntryIntent(
                    pair_id=self.pair_id,
                    direction=direction,
                    z_score=z_score,
                    threshold_set=threshold_set,
                    regime=regime_label,
                    confidence=conviction,
                    rationale=(
                        f"z={z_score:.2f} >= entry_z={threshold_set.entry_z:.2f}, "
                        f"regime={regime_label.value}"
                    ),
                )
        elif self._lifecycle.is_position_active():
            abs_z = abs(z_score)
            if abs_z <= threshold_set.exit_z or abs_z >= threshold_set.stop_z:
                from core.contracts import ExitReason
                exit_reason = (
                    ExitReason.MEAN_REVERSION
                    if abs_z <= threshold_set.exit_z
                    else ExitReason.STOP_LOSS
                )
                intent = ExitIntent(
                    pair_id=self.pair_id,
                    z_score=z_score,
                    exit_reason=exit_reason,
                    regime=regime_label,
                    confidence=conviction,
                    rationale=(
                        f"z={z_score:.2f}, "
                        f"exit_z={threshold_set.exit_z:.2f}, "
                        f"stop_z={threshold_set.stop_z:.2f}"
                    ),
                )

        size_mult = quality_result.size_multiplier if quality_result is not None else 1.0

        decision = SignalDecision(
            pair_id=self.pair_id,
            as_of=as_of,
            z_score=z_score,
            regime=regime_label.value,
            quality_grade=quality_result.grade.value if quality_result is not None else "UNKNOWN",
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

    @property
    def lifecycle_state(self):
        return self._lifecycle.state


class SignalPipelineRegistry:
    """
    Registry of SignalPipelines keyed by PairId.

    Manages one pipeline per active pair. Thread-safe for get/create.
    """

    def __init__(self):
        self._pipelines: dict[str, SignalPipeline] = {}
        self._lock = threading.Lock()

    def get_or_create(self, pair_id: PairId, **pipeline_kwargs) -> SignalPipeline:
        key = str(pair_id)
        with self._lock:
            if key not in self._pipelines:
                self._pipelines[key] = SignalPipeline(pair_id=pair_id, **pipeline_kwargs)
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
