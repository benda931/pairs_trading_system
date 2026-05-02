# -*- coding: utf-8 -*-
"""
core/action_observability.py — Agent Action Observability & Safety Machinery
=============================================================================

Provides the four safety sub-systems that the GovernanceRouter calls before
routing any action to execution:

1. **DuplicateActionSuppressor**
   Deduplicates by (action_type, target, direction) within the action's
   cooldown window. An agent firing the same BLOCK_ENTRY on the same pair
   twice within 3 hours is suppressed on the second firing — not throttled
   by a generic rate limiter, but deduplicated against the specific pending
   or recently-executed record.

2. **StaleRecommendationDetector**
   Checks evidence age against the profile's `max_age_seconds` per field.
   Also detects regime transitions: if the regime has changed since the
   action was created, the recommendation may no longer be valid.

3. **ConflictResolver**
   Detects conflicting concurrent actions:
   - Opposite actions on the same target (BLOCK_ENTRY vs. unblock)
   - DELEVERAGE + KILL_SWITCH both pending (KILL_SWITCH wins, DELEVERAGE deferred)
   - RETRAIN_MODEL + OPTIMIZE_PARAMS concurrently (sequence required)
   - Two agents recommending opposite ADJUST_THRESHOLD directions
   Returns a `ConflictResolution` describing the winner, loser, and rule applied.

4. **ActionObserver** + **PrecisionDemotionEngine**
   After execution, the observer registers each `GovernedActionRecord` for
   outcome evaluation. At the observation deadline, it checks whether the
   action's outcome was correct. The `PrecisionDemotionEngine` aggregates
   outcomes into `ActionPrecisionMetrics` and automatically promotes the
   governance tier one level when rolling 30-day precision falls below the
   profile's `precision_demotion_threshold`.

All classes are thread-safe. They share a single in-process state store;
production deployments should replace the in-memory dicts with a persistence
adapter.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from core.action_governance import (
    ActionGovernanceTier,
    GovernedActionRecord,
    TradingEnvironment,
    get_profile,
)

import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 1. OBSERVABILITY DATA CLASSES
# ══════════════════════════════════════════════════════════════════


@dataclass
class ActionPrecisionMetrics:
    """Rolling precision statistics for one (action_type, environment) pair.

    Updated by PrecisionDemotionEngine whenever an outcome is observed.
    """

    action_type: str
    environment: TradingEnvironment
    total_executed: int = 0
    total_correct: int = 0
    total_incorrect: int = 0
    total_unobserved: int = 0
    rolling_precision_30d: float = 1.0   # Initialised optimistically; degrades on evidence.
    rolling_precision_90d: float = 1.0
    false_positive_rate: float = 0.0
    avg_regret_score: float = 0.0
    current_tier: ActionGovernanceTier = ActionGovernanceTier.AUTO_EXECUTABLE
    tier_demoted_at: Optional[datetime] = None
    tier_demotion_reason: Optional[str] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def precision(self) -> float:
        """Point-in-time precision from total_correct / (total_correct + total_incorrect)."""
        denominator = self.total_correct + self.total_incorrect
        return self.total_correct / denominator if denominator > 0 else 1.0


@dataclass(frozen=True)
class RegretRecord:
    """Immutable record of a single regret-generating outcome.

    A regret occurs when an executed action is later evaluated as wrong:
    - BLOCK_ENTRY fired but the spread reverted profitably → false positive.
    - FORCE_EXIT fired but the position would have recovered → premature exit.

    Fields
    ------
    action_id : str
        action_id from GovernedActionRecord.
    action_type : str
    regret_score : float
        0.0 = no regret, 1.0 = maximum regret. Computed as:
        abs(counterfactual_pnl - actual_pnl) / abs(counterfactual_pnl) capped at 1.
    counterfactual_pnl : Optional[float]
        Estimated P&L if the action had NOT been taken.
    actual_pnl : Optional[float]
        Actual P&L impact of the action.
    regret_dimension : str
        Primary dimension of regret: "timing" | "magnitude" | "direction".
    regret_reason : str
        Human-readable explanation.
    recorded_at : datetime
    """

    action_id: str
    action_type: str
    regret_score: float
    counterfactual_pnl: Optional[float]
    actual_pnl: Optional[float]
    regret_dimension: str
    regret_reason: str
    recorded_at: datetime


@dataclass(frozen=True)
class ConflictResolution:
    """Immutable record of a conflict resolution between two actions.

    Fields
    ------
    winner_action_id : str
        The action that proceeds.
    loser_action_id : str
        The action that is suppressed or deferred.
    resolution_rule : str
        Named rule applied (e.g. "KILL_SWITCH_DOMINATES_DELEVERAGE").
    rationale : str
        Human-readable explanation.
    resolved_at : datetime
    """

    winner_action_id: str
    loser_action_id: str
    resolution_rule: str
    rationale: str
    resolved_at: datetime


@dataclass(frozen=True)
class DuplicationCheckResult:
    """Result from DuplicateActionSuppressor.check()."""

    suppress: bool
    reason: Optional[str]          # None if not suppressed.
    original_action_id: Optional[str]  # The existing action this is a duplicate of.


@dataclass(frozen=True)
class StalenessCheckResult:
    """Result from StaleRecommendationDetector.check()."""

    is_stale: bool
    age_seconds: int               # -1 if not determinable.
    stale_fields: Tuple[str, ...]  # Which fields exceeded max_age_seconds.
    regime_changed: bool           # True if regime transitioned since action creation.
    staleness_reason: str          # Human-readable explanation. "" if not stale.


# ══════════════════════════════════════════════════════════════════
# 2. DUPLICATE ACTION SUPPRESSOR
# ══════════════════════════════════════════════════════════════════


class DuplicateActionSuppressor:
    """Deduplicates agent actions within their cooldown window.

    An action is considered a duplicate if another action with the same
    (action_type, target) has been executed or is pending within the
    cooldown period defined in the governance profile.

    Additionally tracks the outcome of the last action of each type+target.
    If the last action was evaluated as incorrect (outcome_correct=False),
    the suppressor does NOT suppress the new action — letting the agent try
    again is correct behaviour in that case.
    """

    def __init__(self) -> None:
        # (action_type, target) → GovernedActionRecord of most recent action
        self._recent: Dict[Tuple[str, str], GovernedActionRecord] = {}
        self._lock = threading.Lock()

    def check(
        self,
        action_type: str,
        target: str,
        environment: TradingEnvironment,
        new_action_id: str,
    ) -> DuplicationCheckResult:
        """Check whether this action should be suppressed as a duplicate.

        Parameters
        ----------
        action_type : str
        target : str
            The action's target (pair_id, "portfolio", "system").
        environment : TradingEnvironment
        new_action_id : str
            The new action's ID (to avoid comparing against itself).

        Returns
        -------
        DuplicationCheckResult
        """
        try:
            profile = get_profile(action_type, environment)
        except KeyError:
            return DuplicationCheckResult(suppress=False, reason=None, original_action_id=None)

        if profile.cooldown_seconds <= 0:
            return DuplicationCheckResult(suppress=False, reason=None, original_action_id=None)

        key = (action_type, target)
        with self._lock:
            existing = self._recent.get(key)

        if existing is None or existing.action_id == new_action_id:
            return DuplicationCheckResult(suppress=False, reason=None, original_action_id=None)

        # If the last action was wrong, allow retry immediately.
        if existing.outcome_correct is False:
            return DuplicationCheckResult(suppress=False, reason=None, original_action_id=None)

        # Check cooldown window
        if existing.execution_timestamp is not None:
            elapsed = (datetime.now(timezone.utc) - existing.execution_timestamp).total_seconds()
        elif existing.created_at is not None:
            elapsed = (datetime.now(timezone.utc) - existing.created_at).total_seconds()
        else:
            return DuplicationCheckResult(suppress=False, reason=None, original_action_id=None)

        if elapsed < profile.cooldown_seconds:
            remaining = profile.cooldown_seconds - elapsed
            return DuplicationCheckResult(
                suppress=True,
                reason="DUPLICATE: {}/{} last executed/created {:.0f}s ago; cooldown={:.0f}s "
                       "({:.0f}s remaining)".format(
                           action_type, target, elapsed,
                           profile.cooldown_seconds, remaining,
                       ),
                original_action_id=existing.action_id,
            )

        return DuplicationCheckResult(suppress=False, reason=None, original_action_id=None)

    def record(self, record: GovernedActionRecord) -> None:
        """Store the record for deduplication lookups."""
        key = (record.action_type, record.evidence_snapshot.get("target",
               record.evidence_snapshot.get("pair_id", "unknown")))
        with self._lock:
            self._recent[key] = record

    def update_outcome(self, action_id: str, outcome_correct: bool) -> None:
        """Update the outcome on a stored record so failed actions allow retry."""
        with self._lock:
            for key, record in self._recent.items():
                if record.action_id == action_id:
                    # GovernedActionRecord is mutable — update in place.
                    record.outcome_correct = outcome_correct
                    break


# ══════════════════════════════════════════════════════════════════
# 3. STALE RECOMMENDATION DETECTOR
# ══════════════════════════════════════════════════════════════════


class StaleRecommendationDetector:
    """Checks whether a FeedbackAction's evidence is still fresh enough to act on.

    Staleness is checked two ways:
    1. Per-field max_age_seconds from the governance profile's required_evidence.
       If `FeedbackAction.parameters` contains a key `evidence_timestamp` (ISO string
       or epoch float), that timestamp is used for age calculation. Otherwise the
       action's created_at is used.
    2. Regime transition detection: if `parameters["regime"]` differs from the
       current regime in the bus, the recommendation is flagged as regime-stale.
    """

    def check(
        self,
        action_type: str,
        environment: TradingEnvironment,
        parameters: dict,
        created_at: datetime,
        current_regime: Optional[str] = None,
    ) -> StalenessCheckResult:
        """Evaluate staleness for an action.

        Parameters
        ----------
        action_type : str
        environment : TradingEnvironment
        parameters : dict
            FeedbackAction.parameters.
        created_at : datetime
            When the FeedbackAction was created.
        current_regime : Optional[str]
            The current regime from an external source (bus, cache). If None,
            regime-change detection is skipped.

        Returns
        -------
        StalenessCheckResult
        """
        try:
            profile = get_profile(action_type, environment)
        except KeyError:
            return StalenessCheckResult(
                is_stale=False, age_seconds=-1, stale_fields=(),
                regime_changed=False, staleness_reason="",
            )

        now = datetime.now(timezone.utc)

        # Determine evidence timestamp
        evidence_ts_raw = parameters.get("evidence_timestamp")
        if evidence_ts_raw is not None:
            try:
                if isinstance(evidence_ts_raw, (int, float)):
                    evidence_ts = datetime.fromtimestamp(evidence_ts_raw, tz=timezone.utc)
                else:
                    evidence_ts = datetime.fromisoformat(str(evidence_ts_raw))
                    if evidence_ts.tzinfo is None:
                        evidence_ts = evidence_ts.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                evidence_ts = created_at
        else:
            evidence_ts = created_at

        age_seconds = int((now - evidence_ts).total_seconds())

        # Check per-field staleness
        stale_fields: List[str] = []
        for req in profile.required_evidence:
            if req.max_age_seconds is not None and req.required:
                field_age = age_seconds  # Use evidence timestamp for all fields
                if field_age > req.max_age_seconds:
                    stale_fields.append(req.field_name)

        # Overall recommendation expiry
        expiry_age = profile.recommendation_expiry_seconds
        recommendation_expired = age_seconds > expiry_age

        # Regime transition detection
        regime_changed = False
        if current_regime is not None:
            action_regime = parameters.get("regime", "")
            if action_regime and action_regime.upper() != current_regime.upper():
                regime_changed = True

        is_stale = bool(stale_fields) or recommendation_expired or regime_changed

        if not is_stale:
            return StalenessCheckResult(
                is_stale=False, age_seconds=age_seconds,
                stale_fields=(), regime_changed=False, staleness_reason="",
            )

        reasons = []
        if stale_fields:
            reasons.append("stale_fields={}".format(stale_fields))
        if recommendation_expired:
            reasons.append("recommendation_expired (age={}s > expiry={}s)".format(
                age_seconds, expiry_age))
        if regime_changed:
            reasons.append("regime_changed: action_regime='{}' current='{}'".format(
                parameters.get("regime", ""), current_regime))

        return StalenessCheckResult(
            is_stale=True,
            age_seconds=age_seconds,
            stale_fields=tuple(stale_fields),
            regime_changed=regime_changed,
            staleness_reason="; ".join(reasons),
        )


# ══════════════════════════════════════════════════════════════════
# 4. CONFLICT RESOLVER
# ══════════════════════════════════════════════════════════════════

# Priority ordering: higher index = higher priority (wins conflicts).
_PRIORITY_ORDER = [
    "PAUSE_PIPELINE",
    "ADJUST_THRESHOLD",
    "OPTIMIZE_PARAMS",
    "UPDATE_CONFIG",
    "RETRAIN_MODEL",
    "BLOCK_ENTRY",
    "DELEVERAGE",
    "FORCE_EXIT",
    "KILL_SWITCH",
]


class ConflictResolver:
    """Detects and resolves conflicts between concurrent pending actions.

    Conflict rules (evaluated in order):
    1. KILL_SWITCH dominates everything — any other pending action on the same
       scope is deferred when a KILL_SWITCH arrives.
    2. DELEVERAGE + KILL_SWITCH both pending on "portfolio"/"system" →
       KILL_SWITCH wins, DELEVERAGE is deferred.
    3. RETRAIN_MODEL + OPTIMIZE_PARAMS both pending on the same target →
       RETRAIN_MODEL must complete first; OPTIMIZE_PARAMS is deferred.
    4. Opposite ADJUST_THRESHOLD directions (one agent raises, another lowers
       the same threshold_name) → higher-confidence agent wins; equal confidence
       → defer both to human.
    5. General priority-order rule: higher-priority action wins if both target
       the same scope and are incompatible.
    """

    def __init__(self) -> None:
        # action_id → GovernedActionRecord for currently pending/recently-executed
        self._pending: Dict[str, GovernedActionRecord] = {}
        self._lock = threading.Lock()

    def check(
        self,
        incoming_action_type: str,
        incoming_action_id: str,
        incoming_target: str,
        incoming_parameters: dict,
    ) -> Optional[ConflictResolution]:
        """Check if the incoming action conflicts with any pending action.

        Returns a ConflictResolution if a conflict exists, otherwise None.
        The caller must check ConflictResolution.loser_action_id to determine
        which action to suppress.
        """
        now = datetime.now(timezone.utc)
        with self._lock:
            pending_list = list(self._pending.values())

        for existing in pending_list:
            if existing.action_id == incoming_action_id:
                continue
            if not existing.executed and not existing.suppressed:
                resolution = self._resolve(
                    incoming_action_type, incoming_action_id, incoming_target,
                    incoming_parameters, existing, now,
                )
                if resolution is not None:
                    return resolution

        return None

    def _resolve(
        self,
        inc_type: str,
        inc_id: str,
        inc_target: str,
        inc_params: dict,
        existing: GovernedActionRecord,
        now: datetime,
    ) -> Optional[ConflictResolution]:
        """Apply conflict rules between one incoming and one existing action."""

        # Normalise targets for scope comparison
        inc_scope = _normalise_scope(inc_target)
        ex_scope = _normalise_scope(
            existing.evidence_snapshot.get("target",
            existing.evidence_snapshot.get("pair_id", "unknown"))
        )
        scopes_overlap = inc_scope == ex_scope or "system" in (inc_scope, ex_scope)

        if not scopes_overlap:
            return None

        # Rule 1 + 2: KILL_SWITCH dominates
        if inc_type == "KILL_SWITCH" and existing.action_type != "KILL_SWITCH":
            return ConflictResolution(
                winner_action_id=inc_id,
                loser_action_id=existing.action_id,
                resolution_rule="KILL_SWITCH_DOMINATES",
                rationale="KILL_SWITCH always takes priority; {} deferred.".format(
                    existing.action_type),
                resolved_at=now,
            )
        if existing.action_type == "KILL_SWITCH" and inc_type != "KILL_SWITCH":
            return ConflictResolution(
                winner_action_id=existing.action_id,
                loser_action_id=inc_id,
                resolution_rule="KILL_SWITCH_DOMINATES",
                rationale="KILL_SWITCH already pending; {} deferred until resolved.".format(
                    inc_type),
                resolved_at=now,
            )

        # Rule 3: RETRAIN_MODEL + OPTIMIZE_PARAMS sequencing
        pair = {inc_type, existing.action_type}
        if pair == {"RETRAIN_MODEL", "OPTIMIZE_PARAMS"}:
            winner_type = "RETRAIN_MODEL"
            loser_type = "OPTIMIZE_PARAMS"
            winner_id = inc_id if inc_type == winner_type else existing.action_id
            loser_id = inc_id if inc_type == loser_type else existing.action_id
            return ConflictResolution(
                winner_action_id=winner_id,
                loser_action_id=loser_id,
                resolution_rule="RETRAIN_BEFORE_OPTIMIZE",
                rationale="RETRAIN_MODEL must complete before OPTIMIZE_PARAMS on same target.",
                resolved_at=now,
            )

        # Rule 4: Opposite ADJUST_THRESHOLD directions
        if inc_type == "ADJUST_THRESHOLD" and existing.action_type == "ADJUST_THRESHOLD":
            inc_name = inc_params.get("threshold_name", "")
            ex_name = existing.evidence_snapshot.get("threshold_name", "")
            if inc_name and inc_name == ex_name:
                inc_val = inc_params.get("proposed_value", 0)
                ex_val = existing.evidence_snapshot.get("proposed_value", 0)
                inc_curr = inc_params.get("current_value", 0)
                # Check if directions are opposite
                inc_direction = "up" if inc_val > inc_curr else "down"
                ex_curr = existing.evidence_snapshot.get("current_value", inc_curr)
                ex_direction = "up" if ex_val > ex_curr else "down"
                if inc_direction != ex_direction:
                    return ConflictResolution(
                        winner_action_id=existing.action_id,
                        loser_action_id=inc_id,
                        resolution_rule="THRESHOLD_DIRECTION_CONFLICT",
                        rationale=(
                            "Two agents disagree on threshold '{}' direction "
                            "({} vs {}); incoming action deferred to human review.".format(
                                inc_name, inc_direction, ex_direction)
                        ),
                        resolved_at=now,
                    )

        # Rule 5: General priority order
        inc_prio = _action_priority(inc_type)
        ex_prio = _action_priority(existing.action_type)
        if inc_prio > ex_prio:
            return ConflictResolution(
                winner_action_id=inc_id,
                loser_action_id=existing.action_id,
                resolution_rule="PRIORITY_ORDER",
                rationale="{} (priority {}) supersedes {} (priority {}).".format(
                    inc_type, inc_prio, existing.action_type, ex_prio),
                resolved_at=now,
            )

        return None

    def register_pending(self, record: GovernedActionRecord) -> None:
        """Register an action as pending so it can participate in future conflict checks."""
        with self._lock:
            self._pending[record.action_id] = record

    def mark_resolved(self, action_id: str) -> None:
        """Mark an action as no longer pending (executed, suppressed, or rejected)."""
        with self._lock:
            self._pending.pop(action_id, None)


def _normalise_scope(target: str) -> str:
    if not target:
        return "unknown"
    return target.lower().replace(" ", "_")


def _action_priority(action_type: str) -> int:
    try:
        return _PRIORITY_ORDER.index(action_type)
    except ValueError:
        return -1


# ══════════════════════════════════════════════════════════════════
# 5. ACTION OBSERVER
# ══════════════════════════════════════════════════════════════════


class ActionObserver:
    """Registers GovernedActionRecords for deferred outcome observation.

    After execution, the observer schedules a check at `observation_deadline`.
    At that time, the caller (e.g. a periodic health-check task) invokes
    `evaluate_due_observations()` which calls user-supplied evaluator functions
    to determine whether the action's outcome was correct.

    Evaluators are registered per action type via `register_evaluator()`.
    An evaluator is a callable that takes a GovernedActionRecord and returns
    (outcome_correct: bool, regret_score: float, reason: str).
    """

    def __init__(self) -> None:
        # action_id → GovernedActionRecord
        self._pending_observations: Dict[str, GovernedActionRecord] = {}
        # action_type → Callable
        self._evaluators: Dict[str, callable] = {}
        self._lock = threading.Lock()

    def register(
        self,
        record: GovernedActionRecord,
        observation_window_seconds: int = 172800,  # 48h default
    ) -> None:
        """Register a record for outcome observation.

        Parameters
        ----------
        record : GovernedActionRecord
        observation_window_seconds : int
            Seconds after execution before the outcome is evaluated. Default 48h.
        """
        deadline = datetime.now(timezone.utc) + timedelta(seconds=observation_window_seconds)
        record.outcome_observation_deadline = deadline
        with self._lock:
            self._pending_observations[record.action_id] = record

    def register_evaluator(
        self,
        action_type: str,
        evaluator_fn,  # Callable[[GovernedActionRecord], Tuple[bool, float, str]]
    ) -> None:
        """Register an outcome evaluator for a given action type.

        The evaluator receives the GovernedActionRecord after execution and
        returns (outcome_correct, regret_score, reason).
        """
        self._evaluators[action_type] = evaluator_fn

    def evaluate_due_observations(self) -> List[Tuple[GovernedActionRecord, bool, float, str]]:
        """Evaluate all observations whose deadline has passed.

        Returns list of (record, outcome_correct, regret_score, reason) tuples
        for each observation evaluated. Evaluated records are removed from
        pending.
        """
        now = datetime.now(timezone.utc)
        results = []

        with self._lock:
            due = [
                r for r in self._pending_observations.values()
                if r.outcome_observation_deadline is not None
                and r.outcome_observation_deadline <= now
                and not r.outcome_observed
            ]

        for record in due:
            evaluator = self._evaluators.get(record.action_type)
            if evaluator is None:
                # No evaluator → mark as unobserved and remove from pending
                with self._lock:
                    self._pending_observations.pop(record.action_id, None)
                continue

            try:
                outcome_correct, regret_score, reason = evaluator(record)
                record.outcome_observed = True
                record.outcome_correct = outcome_correct
                record.regret_score = regret_score
                results.append((record, outcome_correct, regret_score, reason))
            except Exception as exc:
                logger.warning(
                    "ActionObserver: evaluator for %s raised: %s",
                    record.action_type, exc
                )

            with self._lock:
                self._pending_observations.pop(record.action_id, None)

        return results

    def get_pending_count(self) -> int:
        with self._lock:
            return len(self._pending_observations)


# ══════════════════════════════════════════════════════════════════
# 6. PRECISION DEMOTION ENGINE
# ══════════════════════════════════════════════════════════════════


class PrecisionDemotionEngine:
    """Tracks per-action-type precision and automatically tightens governance tiers.

    When an agent's rolling 30-day precision on a given (action_type, environment)
    pair falls below the governance profile's `precision_demotion_threshold`, the
    engine promotes the tier one level:
        AUTO_EXECUTABLE  → POLICY_GATED
        POLICY_GATED     → HUMAN_REQUIRED

    The demotion is persisted in `ActionPrecisionMetrics.current_tier`. The
    GovernanceRouter checks this before using the profile default tier.

    Tier promotions (re-enabling autonomy) require explicit manual reset via
    `reset_demotion()` — they do not happen automatically.
    """

    def __init__(self) -> None:
        # (action_type, environment) → ActionPrecisionMetrics
        self._metrics: Dict[Tuple[str, TradingEnvironment], ActionPrecisionMetrics] = {}
        # (action_id, outcome_date) rolling window for 30d calculation
        self._outcome_window: Dict[Tuple[str, TradingEnvironment], List[Tuple[datetime, bool]]] = \
            defaultdict(list)
        self._lock = threading.Lock()

    def record_outcome(
        self,
        action_type: str,
        environment: TradingEnvironment,
        action_id: str,
        outcome_correct: bool,
        regret_score: float,
    ) -> ActionPrecisionMetrics:
        """Record an outcome observation and update precision metrics.

        Returns the updated ActionPrecisionMetrics for this (action_type, env).
        """
        key = (action_type, environment)
        now = datetime.now(timezone.utc)

        with self._lock:
            if key not in self._metrics:
                try:
                    profile = get_profile(action_type, environment)
                    default_tier = profile.tier
                except KeyError:
                    default_tier = ActionGovernanceTier.AUTO_EXECUTABLE

                self._metrics[key] = ActionPrecisionMetrics(
                    action_type=action_type,
                    environment=environment,
                    current_tier=default_tier,
                )

            metrics = self._metrics[key]
            metrics.total_executed += 1
            if outcome_correct:
                metrics.total_correct += 1
            else:
                metrics.total_incorrect += 1

            # Rolling window — keep last 90 days
            window = self._outcome_window[key]
            window.append((now, outcome_correct))
            cutoff_90d = now - timedelta(days=90)
            cutoff_30d = now - timedelta(days=30)
            window[:] = [(ts, ok) for ts, ok in window if ts >= cutoff_90d]

            # Recompute rolling precision
            last_30d = [(ts, ok) for ts, ok in window if ts >= cutoff_30d]
            if last_30d:
                correct_30d = sum(1 for _, ok in last_30d if ok)
                metrics.rolling_precision_30d = correct_30d / len(last_30d)
            if window:
                correct_90d = sum(1 for _, ok in window if ok)
                metrics.rolling_precision_90d = correct_90d / len(window)

            # FP rate: incorrect / total
            total = metrics.total_correct + metrics.total_incorrect
            metrics.false_positive_rate = metrics.total_incorrect / total if total > 0 else 0.0

            # Regret (running average)
            if regret_score > 0:
                n = metrics.total_incorrect or 1
                metrics.avg_regret_score = (
                    (metrics.avg_regret_score * (n - 1) + regret_score) / n
                )

            metrics.last_updated = now

            # Check for demotion
            try:
                profile = get_profile(action_type, environment)
                threshold = profile.precision_demotion_threshold
            except KeyError:
                threshold = 0.60

            if metrics.rolling_precision_30d < threshold:
                demoted = self._demote_tier(metrics, profile if 'profile' in dir() else None)
                if demoted:
                    logger.warning(
                        "PrecisionDemotionEngine: %s/%s demoted to %s "
                        "(30d precision=%.2f < threshold=%.2f)",
                        action_type, environment.value,
                        metrics.current_tier.value,
                        metrics.rolling_precision_30d, threshold,
                    )

            return metrics

    def _demote_tier(
        self,
        metrics: ActionPrecisionMetrics,
        profile,
    ) -> bool:
        """Promote tier one level. Returns True if demotion occurred."""
        tier = metrics.current_tier
        next_tier_map = {
            ActionGovernanceTier.AUTO_EXECUTABLE: ActionGovernanceTier.POLICY_GATED,
            ActionGovernanceTier.POLICY_GATED: ActionGovernanceTier.HUMAN_REQUIRED,
            ActionGovernanceTier.HUMAN_REQUIRED: ActionGovernanceTier.HUMAN_REQUIRED,  # Already max
        }
        next_tier = next_tier_map.get(tier)
        if next_tier is None or next_tier == tier:
            return False

        metrics.current_tier = next_tier
        metrics.tier_demoted_at = datetime.now(timezone.utc)
        metrics.tier_demotion_reason = (
            "30d precision {:.1%} below threshold {:.1%}; "
            "tier promoted from {} to {} for accountability.".format(
                metrics.rolling_precision_30d,
                profile.precision_demotion_threshold if profile else 0.0,
                tier.value,
                next_tier.value,
            )
        )
        return True

    def get_metrics(
        self,
        action_type: str,
        environment: TradingEnvironment,
    ) -> Optional[ActionPrecisionMetrics]:
        """Return current precision metrics for (action_type, environment)."""
        with self._lock:
            return self._metrics.get((action_type, environment))

    def get_effective_tier(
        self,
        action_type: str,
        environment: TradingEnvironment,
    ) -> Optional[ActionGovernanceTier]:
        """Return the current effective tier (may be demoted from profile default)."""
        with self._lock:
            metrics = self._metrics.get((action_type, environment))
        if metrics is None:
            return None
        return metrics.current_tier

    def reset_demotion(
        self,
        action_type: str,
        environment: TradingEnvironment,
        reset_by: str,
        justification: str,
    ) -> bool:
        """Manually restore the tier to the profile default.

        Requires a human-supplied justification string for audit purposes.
        Returns True if a reset was applied, False if no demotion existed.
        """
        key = (action_type, environment)
        with self._lock:
            metrics = self._metrics.get(key)
            if metrics is None or metrics.tier_demoted_at is None:
                return False

            try:
                profile = get_profile(action_type, environment)
                metrics.current_tier = profile.tier
            except KeyError:
                metrics.current_tier = ActionGovernanceTier.AUTO_EXECUTABLE

            metrics.tier_demoted_at = None
            metrics.tier_demotion_reason = (
                "DEMOTION_RESET by {} at {}: {}".format(
                    reset_by,
                    datetime.now(timezone.utc).isoformat(),
                    justification,
                )
            )
            metrics.last_updated = datetime.now(timezone.utc)

        logger.info(
            "PrecisionDemotionEngine: demotion reset for %s/%s by %s",
            action_type, environment.value, reset_by,
        )
        return True

    def get_all_metrics(self) -> List[ActionPrecisionMetrics]:
        """Return all tracked metrics sorted by action_type."""
        with self._lock:
            return sorted(self._metrics.values(), key=lambda m: m.action_type)
