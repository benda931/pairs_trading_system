# -*- coding: utf-8 -*-
"""
incidents/detector.py — Governance Pattern Detector
====================================================

Analyses the GovernanceRouter's audit stream to detect systemic patterns
that warrant human attention beyond individual action outcomes.

Detection operates on the IncidentManager's audit ledger. The detector is
called periodically (e.g., in the health_check task) and may automatically
open new incidents when patterns exceed thresholds.

Patterns detected
-----------------
REPEATED_KILL_SWITCH
    More than 2 KILL_SWITCH activations in a 7-day window.
    Indicates instability in the risk model or regime engine.
    → Severity P1, opens incident, recommends investigation.

REPEATED_BLOCK_SAME_PAIR
    More than 3 BLOCK_ENTRY actions for the same pair in 30 days.
    Indicates the pair's signal quality is persistently degraded.
    → Severity P2, opens incident, recommends pair retirement review.

RETRAIN_CYCLE
    RETRAIN_MODEL executed within 7 days of a previous RETRAIN_MODEL
    on the same model_id.
    Indicates potential training instability or data distribution shift.
    → Severity P2, opens incident.

PRECISION_DEGRADATION
    An action type's 30-day precision has dropped more than 20 percentage
    points relative to its 90-day baseline, OR dropped below 50% overall.
    → Severity P2, opens incident.

APPROVAL_SLA_PATTERN
    More than 3 HUMAN_REVIEW_PENDING statuses that never resolved within
    their SLA window in the past 7 days.
    Indicates PM/risk committee review capacity issue.
    → Severity P3, opens incident.

CONFLICTING_AGENTS_PATTERN
    More than 5 CONFLICT_DEFERRED suppressions in 7 days for the same pair.
    Indicates two agents systematically disagree on that pair.
    → Severity P3, opens incident.

GOVERNANCE_BYPASS_PATTERN
    More than 1 EMERGENCY_BYPASS in 14 days.
    Indicates the governance framework is under stress.
    → Severity P1, opens incident.

Usage
-----
    from incidents.detector import GovernancePatternDetector, get_pattern_detector
    detector = get_pattern_detector()
    patterns = detector.scan()  # Returns list[DetectedPattern]

The detector does not store state between scans — it always re-reads the
audit ledger. This makes it idempotent and safe to call frequently.
"""

from __future__ import annotations

import enum
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from incidents.contracts import AuditRecordType, IncidentSeverity
from incidents.manager import get_incident_manager

import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 1. PATTERN TYPES
# ══════════════════════════════════════════════════════════════════


class PatternType(str, enum.Enum):
    """Categories of systemic governance patterns."""

    REPEATED_KILL_SWITCH = "repeated_kill_switch"
    REPEATED_BLOCK_SAME_PAIR = "repeated_block_same_pair"
    RETRAIN_CYCLE = "retrain_cycle"
    PRECISION_DEGRADATION = "precision_degradation"
    APPROVAL_SLA_PATTERN = "approval_sla_pattern"
    CONFLICTING_AGENTS_PATTERN = "conflicting_agents_pattern"
    GOVERNANCE_BYPASS_PATTERN = "governance_bypass_pattern"


# ══════════════════════════════════════════════════════════════════
# 2. DETECTED PATTERN
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DetectedPattern:
    """Immutable record of a detected systemic governance pattern.

    Fields
    ------
    pattern_id : str
        UUID for this detection event.
    pattern_type : PatternType
    severity : IncidentSeverity
        Recommended incident severity for auto-opened incident.
    evidence_action_ids : tuple[str, ...]
        action_ids from GovernedActionRecords that form the evidence.
    evidence_summary : str
        Human-readable summary of what was detected.
    recommended_investigation : str
        Suggested investigation steps for the PM/risk team.
    auto_open_incident : bool
        Whether this pattern should automatically open an incident.
    incident_id : Optional[str]
        ID of incident opened by this detection. Populated post-detection.
    detected_at : datetime
    """

    pattern_id: str
    pattern_type: PatternType
    severity: IncidentSeverity
    evidence_action_ids: Tuple[str, ...]
    evidence_summary: str
    recommended_investigation: str
    auto_open_incident: bool
    detected_at: datetime
    incident_id: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# 3. PATTERN DETECTOR
# ══════════════════════════════════════════════════════════════════


class GovernancePatternDetector:
    """Scans the audit ledger for systemic governance patterns.

    The detector reads from the IncidentManager's audit ledger and
    the GovernanceRouter's precision metrics (passed in at scan time).
    It does not maintain internal state — each `scan()` call re-reads
    the ledger to build its analysis.

    Parameters
    ----------
    kill_switch_window_days : int
        Look-back window for REPEATED_KILL_SWITCH detection. Default 7.
    kill_switch_threshold : int
        Number of KILL_SWITCH events in window that triggers pattern. Default 2.
    block_pair_window_days : int
        Look-back window for REPEATED_BLOCK_SAME_PAIR. Default 30.
    block_pair_threshold : int
        Number of BLOCK_ENTRY events for same pair that triggers pattern. Default 3.
    retrain_min_interval_days : int
        Minimum days between retrains. Default 7.
    sla_window_days : int
        Look-back window for APPROVAL_SLA_PATTERN. Default 7.
    sla_breach_threshold : int
        Number of SLA-breached tickets that triggers pattern. Default 3.
    conflict_window_days : int
        Look-back window for CONFLICTING_AGENTS_PATTERN. Default 7.
    conflict_threshold : int
        Number of CONFLICT_DEFERRED events for same pair that triggers pattern. Default 5.
    bypass_window_days : int
        Look-back window for GOVERNANCE_BYPASS_PATTERN. Default 14.
    bypass_threshold : int
        Number of emergency bypasses that triggers pattern. Default 1.
    """

    def __init__(
        self,
        kill_switch_window_days: int = 7,
        kill_switch_threshold: int = 2,
        block_pair_window_days: int = 30,
        block_pair_threshold: int = 3,
        retrain_min_interval_days: int = 7,
        sla_window_days: int = 7,
        sla_breach_threshold: int = 3,
        conflict_window_days: int = 7,
        conflict_threshold: int = 5,
        bypass_window_days: int = 14,
        bypass_threshold: int = 1,
    ) -> None:
        self._ks_window = kill_switch_window_days
        self._ks_threshold = kill_switch_threshold
        self._block_window = block_pair_window_days
        self._block_threshold = block_pair_threshold
        self._retrain_interval = retrain_min_interval_days
        self._sla_window = sla_window_days
        self._sla_threshold = sla_breach_threshold
        self._conflict_window = conflict_window_days
        self._conflict_threshold = conflict_threshold
        self._bypass_window = bypass_window_days
        self._bypass_threshold = bypass_threshold

        # Track which patterns have already opened incidents (avoid duplicates)
        self._opened_incident_keys: set = set()
        self._lock = threading.Lock()

    def scan(
        self,
        precision_metrics: Optional[list] = None,
        audit_limit: int = 5000,
    ) -> List[DetectedPattern]:
        """Scan the audit ledger and return all detected patterns.

        Parameters
        ----------
        precision_metrics : list[ActionPrecisionMetrics], optional
            Current precision metrics from GovernanceRouter. If provided,
            PRECISION_DEGRADATION detection is enabled.
        audit_limit : int
            Maximum number of audit records to retrieve for scanning.

        Returns
        -------
        list[DetectedPattern]
            All patterns detected in this scan. May be empty.
        """
        mgr = get_incident_manager()
        records = mgr.get_audit_records(limit=audit_limit)
        now = datetime.now(timezone.utc)

        patterns: List[DetectedPattern] = []

        patterns.extend(self._detect_repeated_kill_switch(records, now))
        patterns.extend(self._detect_repeated_block_same_pair(records, now))
        patterns.extend(self._detect_retrain_cycles(records, now))
        patterns.extend(self._detect_sla_breaches(records, now))
        patterns.extend(self._detect_conflicting_agents(records, now))
        patterns.extend(self._detect_bypass_pattern(records, now))

        if precision_metrics:
            patterns.extend(self._detect_precision_degradation(precision_metrics, now))

        # Auto-open incidents for patterns that qualify
        for pattern in patterns:
            if pattern.auto_open_incident:
                self._maybe_open_incident(pattern)

        return patterns

    # ──────────────────────────────────────────────────────────────
    # DETECTORS
    # ──────────────────────────────────────────────────────────────

    def _detect_repeated_kill_switch(
        self, records, now: datetime
    ) -> List[DetectedPattern]:
        cutoff = now - timedelta(days=self._ks_window)
        ks_records = [
            r for r in records
            if "KILL_SWITCH" in r.action
            and "EXECUTED" in r.action
            and r.outcome == "SUCCESS"
            and self._parse_ts(r.timestamp) >= cutoff
        ]
        if len(ks_records) > self._ks_threshold:
            return [DetectedPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type=PatternType.REPEATED_KILL_SWITCH,
                severity=IncidentSeverity.P1_HIGH,
                evidence_action_ids=tuple(r.task_id or r.subject for r in ks_records),
                evidence_summary=(
                    "{} KILL_SWITCH activations in {} days "
                    "(threshold={}).".format(
                        len(ks_records), self._ks_window, self._ks_threshold)
                ),
                recommended_investigation=(
                    "1. Review regime engine stability — check if CRISIS/BROKEN "
                    "regime is being triggered spuriously.\n"
                    "2. Review drawdown monitor thresholds — check if -20% threshold "
                    "is appropriate for current vol environment.\n"
                    "3. Audit agent output for each KILL_SWITCH — were they correct?\n"
                    "4. Check for data quality issues feeding the risk agents.\n"
                    "5. Consider whether KILL_SWITCH precision has dropped below 70%."
                ),
                auto_open_incident=True,
                detected_at=now,
            )]
        return []

    def _detect_repeated_block_same_pair(
        self, records, now: datetime
    ) -> List[DetectedPattern]:
        cutoff = now - timedelta(days=self._block_window)
        block_records = [
            r for r in records
            if "BLOCK_ENTRY" in r.action
            and "EXECUTED" in r.action
            and self._parse_ts(r.timestamp) >= cutoff
        ]

        # Group by subject (pair_id embedded in subject or task_id)
        pair_counts: Dict[str, List] = defaultdict(list)
        for r in block_records:
            pair_key = r.subject or "unknown"
            pair_counts[pair_key].append(r)

        patterns = []
        for pair, pair_records in pair_counts.items():
            if len(pair_records) > self._block_threshold:
                patterns.append(DetectedPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=PatternType.REPEATED_BLOCK_SAME_PAIR,
                    severity=IncidentSeverity.P2_MEDIUM,
                    evidence_action_ids=tuple(
                        r.task_id or r.subject for r in pair_records
                    ),
                    evidence_summary=(
                        "Pair '{}' blocked {} times in {} days "
                        "(threshold={}).".format(
                            pair, len(pair_records),
                            self._block_window, self._block_threshold)
                    ),
                    recommended_investigation=(
                        "1. Assess whether pair '{}' should be retired from the universe.\n"
                        "2. Review signal quality history for this pair — "
                        "has quality grade been persistently D/F?\n"
                        "3. Check if cointegration has structurally broken.\n"
                        "4. Consider increasing minimum quality threshold for "
                        "this pair cluster.".format(pair)
                    ),
                    auto_open_incident=True,
                    detected_at=now,
                ))
        return patterns

    def _detect_retrain_cycles(
        self, records, now: datetime
    ) -> List[DetectedPattern]:
        retrain_records = [
            r for r in records
            if "RETRAIN_MODEL" in r.action
            and "EXECUTED" in r.action
            and r.outcome == "SUCCESS"
        ]
        # Sort by timestamp
        retrain_records.sort(key=lambda r: r.timestamp)

        patterns = []
        min_interval = timedelta(days=self._retrain_interval)

        for i in range(1, len(retrain_records)):
            prev = retrain_records[i - 1]
            curr = retrain_records[i]
            prev_ts = self._parse_ts(prev.timestamp)
            curr_ts = self._parse_ts(curr.timestamp)
            if curr_ts - prev_ts < min_interval:
                key = "retrain_cycle_{}_{}".format(prev.task_id, curr.task_id)
                with self._lock:
                    if key in self._opened_incident_keys:
                        continue
                patterns.append(DetectedPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=PatternType.RETRAIN_CYCLE,
                    severity=IncidentSeverity.P2_MEDIUM,
                    evidence_action_ids=(
                        prev.task_id or prev.subject,
                        curr.task_id or curr.subject,
                    ),
                    evidence_summary=(
                        "RETRAIN_MODEL executed {:.1f} days after previous retrain "
                        "(minimum interval={} days).".format(
                            (curr_ts - prev_ts).total_seconds() / 86400,
                            self._retrain_interval)
                    ),
                    recommended_investigation=(
                        "1. Check if the rapid retraining was triggered by genuine "
                        "performance degradation or by a data quality issue.\n"
                        "2. Verify that PurgedKFold was used — rapid cycles can indicate "
                        "look-ahead leakage making the model appear to improve.\n"
                        "3. Review whether the cooldown enforcement in the governance "
                        "profile is being respected."
                    ),
                    auto_open_incident=True,
                    detected_at=now,
                ))
        return patterns

    def _detect_sla_breaches(
        self, records, now: datetime
    ) -> List[DetectedPattern]:
        cutoff = now - timedelta(days=self._sla_window)
        sla_breach_records = [
            r for r in records
            if "HUMAN_REVIEW_PENDING" in r.action
            and self._parse_ts(r.timestamp) >= cutoff
        ]
        if len(sla_breach_records) >= self._sla_threshold:
            return [DetectedPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type=PatternType.APPROVAL_SLA_PATTERN,
                severity=IncidentSeverity.P3_LOW,
                evidence_action_ids=tuple(
                    r.task_id or r.subject for r in sla_breach_records
                ),
                evidence_summary=(
                    "{} HUMAN_REQUIRED actions pending review in {} days "
                    "(threshold={}).".format(
                        len(sla_breach_records), self._sla_window, self._sla_threshold)
                ),
                recommended_investigation=(
                    "1. Check the ApprovalEngine open ticket queue — "
                    "are tickets being assigned and reviewed?\n"
                    "2. Review whether the HUMAN_REQUIRED tier is being applied "
                    "too broadly — some actions may be safely downgraded to POLICY_GATED.\n"
                    "3. Ensure the PM/risk committee review schedule is appropriate "
                    "for the current volume of governance actions."
                ),
                auto_open_incident=True,
                detected_at=now,
            )]
        return []

    def _detect_conflicting_agents(
        self, records, now: datetime
    ) -> List[DetectedPattern]:
        cutoff = now - timedelta(days=self._conflict_window)
        conflict_records = [
            r for r in records
            if "CONFLICT_DEFERRED" in (r.details or "")
            and self._parse_ts(r.timestamp) >= cutoff
        ]

        # Group by subject (pair_id)
        pair_counts: Dict[str, List] = defaultdict(list)
        for r in conflict_records:
            pair_counts[r.subject or "unknown"].append(r)

        patterns = []
        for pair, pair_records in pair_counts.items():
            if len(pair_records) >= self._conflict_threshold:
                patterns.append(DetectedPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=PatternType.CONFLICTING_AGENTS_PATTERN,
                    severity=IncidentSeverity.P3_LOW,
                    evidence_action_ids=tuple(
                        r.task_id or r.subject for r in pair_records
                    ),
                    evidence_summary=(
                        "{} agent conflicts on pair '{}' in {} days "
                        "(threshold={}).".format(
                            len(pair_records), pair,
                            self._conflict_window, self._conflict_threshold)
                    ),
                    recommended_investigation=(
                        "1. Identify which two agents are in persistent disagreement "
                        "on pair '{}'.\n"
                        "2. Review their signal sources — are they using stale or "
                        "inconsistent data?\n"
                        "3. Consider whether one agent should be deprecated for this "
                        "pair cluster.\n"
                        "4. If both agents have good precision, consider a voting "
                        "mechanism rather than conflict suppression.".format(pair)
                    ),
                    auto_open_incident=False,  # Advisory only
                    detected_at=now,
                ))
        return patterns

    def _detect_bypass_pattern(
        self, records, now: datetime
    ) -> List[DetectedPattern]:
        cutoff = now - timedelta(days=self._bypass_window)
        bypass_records = [
            r for r in records
            if "EMERGENCY_BYPASS" in (r.details or "")
            or ("EMERGENCY_ONLY" in (r.details or "") and "EXECUTED" in r.action)
            and self._parse_ts(r.timestamp) >= cutoff
        ]
        if len(bypass_records) > self._bypass_threshold:
            return [DetectedPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type=PatternType.GOVERNANCE_BYPASS_PATTERN,
                severity=IncidentSeverity.P1_HIGH,
                evidence_action_ids=tuple(
                    r.task_id or r.subject for r in bypass_records
                ),
                evidence_summary=(
                    "{} emergency governance bypasses in {} days "
                    "(threshold={}).".format(
                        len(bypass_records), self._bypass_window, self._bypass_threshold)
                ),
                recommended_investigation=(
                    "1. Review each emergency bypass: was the circuit-breaker "
                    "activation warranted?\n"
                    "2. Check whether the HUMAN_REQUIRED approval SLA is too slow "
                    "for live trading conditions — if PMs cannot respond within the "
                    "required window, the governance framework is under structural stress.\n"
                    "3. Review postmortem requirements: each bypass should have a "
                    "postmortem within 24h. Check compliance.\n"
                    "4. Consider whether the circuit-breaker thresholds are appropriately "
                    "calibrated for current market conditions."
                ),
                auto_open_incident=True,
                detected_at=now,
            )]
        return []

    def _detect_precision_degradation(
        self, precision_metrics: list, now: datetime
    ) -> List[DetectedPattern]:
        """Detect significant precision drops from ActionPrecisionMetrics list."""
        patterns = []
        for m in precision_metrics:
            # Drop > 20pp from 90d to 30d baseline
            degradation = m.rolling_precision_90d - m.rolling_precision_30d
            hard_floor = m.rolling_precision_30d < 0.50

            if degradation > 0.20 or hard_floor:
                key = "precision_deg_{}_{}".format(m.action_type, m.environment.value)
                with self._lock:
                    if key in self._opened_incident_keys:
                        continue

                patterns.append(DetectedPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type=PatternType.PRECISION_DEGRADATION,
                    severity=IncidentSeverity.P2_MEDIUM,
                    evidence_action_ids=(),
                    evidence_summary=(
                        "{}/{}: 30d precision={:.1%}, 90d precision={:.1%}, "
                        "drop={:.1%}{}".format(
                            m.action_type, m.environment.value,
                            m.rolling_precision_30d, m.rolling_precision_90d,
                            degradation,
                            " [BELOW 50% FLOOR]" if hard_floor else "",
                        )
                    ),
                    recommended_investigation=(
                        "1. Review the last 30 days of {} outcomes in {} environment — "
                        "are there systematic false positives?\n"
                        "2. Check whether a regime change has made the agent's "
                        "decision rules stale.\n"
                        "3. If the tier has been automatically demoted (see "
                        "PrecisionDemotionEngine), review whether the demotion is "
                        "appropriate before resetting it.\n"
                        "4. Consider retraining or reconfiguring the agent if "
                        "the degradation persists beyond 14 days.".format(
                            m.action_type, m.environment.value)
                    ),
                    auto_open_incident=True,
                    detected_at=now,
                ))
        return patterns

    # ──────────────────────────────────────────────────────────────
    # INCIDENT AUTO-OPENING
    # ──────────────────────────────────────────────────────────────

    def _maybe_open_incident(self, pattern: DetectedPattern) -> None:
        """Open an incident for a detected pattern if not already opened."""
        key = "{}_{}".format(pattern.pattern_type.value, "_".join(
            sorted(pattern.evidence_action_ids[:3])  # Use first 3 IDs as dedup key
        ))
        with self._lock:
            if key in self._opened_incident_keys:
                return
            self._opened_incident_keys.add(key)

        try:
            mgr = get_incident_manager()
            incident = mgr.create_incident(
                title="[PATTERN] {}".format(pattern.pattern_type.value.replace("_", " ").title()),
                description=(
                    "{}\n\nRecommended investigation:\n{}".format(
                        pattern.evidence_summary,
                        pattern.recommended_investigation,
                    )
                ),
                severity=pattern.severity,
                detected_by="governance_pattern_detector",
                affected_components=["agent_feedback", "governance_router"],
                evidence_bundle_ids=list(pattern.evidence_action_ids),
                tags=["governance_pattern", pattern.pattern_type.value],
            )
            # Mutate the frozen dataclass via object.__setattr__ — only for incident_id.
            # This is intentional: the incident_id is not known until after creation.
            object.__setattr__(pattern, "incident_id", incident.incident_id)

            logger.info(
                "PatternDetector: opened incident %s for pattern %s",
                incident.incident_id, pattern.pattern_type.value,
            )
        except Exception as exc:
            logger.warning(
                "PatternDetector: failed to open incident for pattern %s: %s",
                pattern.pattern_type.value, exc,
            )

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_ts(ts_str: str) -> datetime:
        """Parse ISO 8601 timestamp string to UTC datetime."""
        try:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return datetime.min.replace(tzinfo=timezone.utc)


# ══════════════════════════════════════════════════════════════════
# SINGLETON FACTORY
# ══════════════════════════════════════════════════════════════════

_detector_instance: Optional[GovernancePatternDetector] = None
_detector_lock = threading.Lock()


def get_pattern_detector(**kwargs) -> GovernancePatternDetector:
    """Return the process-level singleton GovernancePatternDetector.

    Keyword arguments are passed to the constructor on first call only.

    Returns
    -------
    GovernancePatternDetector
    """
    global _detector_instance
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = GovernancePatternDetector(**kwargs)
    return _detector_instance
