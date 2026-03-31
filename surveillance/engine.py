# -*- coding: utf-8 -*-
"""
surveillance/engine.py
======================
Automated market and operational surveillance with rule-based anomaly detection.

Rules are evaluated against submitted metrics. Anomalies become SurveillanceEvents.
Related events are grouped into SurveillanceCases for human review.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ── Enums ─────────────────────────────────────────────────────────────────────

class SurveillanceFamily(Enum):
    SPREAD_ANOMALY = "SPREAD_ANOMALY"
    REGIME_BREACH = "REGIME_BREACH"
    EXECUTION_ANOMALY = "EXECUTION_ANOMALY"
    MODEL_DEGRADATION = "MODEL_DEGRADATION"
    CONCENTRATION_BREACH = "CONCENTRATION_BREACH"
    DATA_INTEGRITY = "DATA_INTEGRITY"
    AGENT_BEHAVIOR = "AGENT_BEHAVIOR"
    RISK_LIMIT_APPROACH = "RISK_LIMIT_APPROACH"


class SurveillanceSeverity(Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SurveillanceEventStatus(Enum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    UNDER_INVESTIGATION = "UNDER_INVESTIGATION"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"


class SurveillanceCaseStatus(Enum):
    OPEN = "OPEN"
    IN_REVIEW = "IN_REVIEW"
    ESCALATED = "ESCALATED"
    CLOSED_CLEAN = "CLOSED_CLEAN"
    CLOSED_ACTION_TAKEN = "CLOSED_ACTION_TAKEN"


# ── Contracts ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SurveillanceRule:
    rule_id: str
    name: str
    family: SurveillanceFamily
    description: str
    severity: SurveillanceSeverity
    enabled: bool
    threshold: float
    lookback_periods: int
    auto_escalate: bool
    metadata: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class SurveillanceEvent:
    event_id: str
    rule_id: str
    family: SurveillanceFamily
    severity: SurveillanceSeverity
    detected_at: datetime
    entity_type: str
    entity_id: str
    description: str
    metric_value: float
    threshold_value: float
    evidence: tuple[str, ...]
    status: SurveillanceEventStatus
    acknowledged_by: Optional[str]
    resolved_at: Optional[datetime]
    case_id: Optional[str]


@dataclass(frozen=True)
class SurveillanceCase:
    case_id: str
    title: str
    family: SurveillanceFamily
    status: SurveillanceCaseStatus
    opened_at: datetime
    opened_by: str
    assigned_to: Optional[str]
    event_ids: tuple[str, ...]
    findings: str
    disposition: Optional[str]
    closed_at: Optional[datetime]
    closed_by: Optional[str]
    incident_id: Optional[str]


# ── Engine ─────────────────────────────────────────────────────────────────────

class SurveillanceEngine:
    """
    Automated market/operational surveillance with rule-based anomaly detection.

    Rules are evaluated against submitted metrics. Anomalies become SurveillanceEvents.
    Related events are grouped into SurveillanceCases for human review.
    """

    def __init__(self) -> None:
        self._rules: dict[str, SurveillanceRule] = {}
        self._events: dict[str, SurveillanceEvent] = {}
        self._cases: dict[str, SurveillanceCase] = {}
        self._event_log: list[SurveillanceEvent] = []
        self._lock = threading.Lock()
        self._populate_default_rules()

    def _populate_default_rules(self) -> None:
        """Register 12 default surveillance rules."""
        defaults = [
            SurveillanceRule(
                "SURV-SA-001", "Spread Z-Score Spike", SurveillanceFamily.SPREAD_ANOMALY,
                "Spread z-score exceeds 4.0 sigma (may indicate data error or regime break)",
                SurveillanceSeverity.HIGH, True, 4.0, 1, False,
            ),
            SurveillanceRule(
                "SURV-SA-002", "Spread Volatility Explosion", SurveillanceFamily.SPREAD_ANOMALY,
                "Rolling spread vol exceeds 3x historical average",
                SurveillanceSeverity.CRITICAL, True, 3.0, 20, True,
            ),
            SurveillanceRule(
                "SURV-RB-001", "Active Strategy in CRISIS Regime", SurveillanceFamily.REGIME_BREACH,
                "Strategy has open positions while regime is CRISIS",
                SurveillanceSeverity.CRITICAL, True, 0.0, 1, True,
            ),
            SurveillanceRule(
                "SURV-RB-002", "Entry During BROKEN Regime", SurveillanceFamily.REGIME_BREACH,
                "New entry attempted while regime is BROKEN",
                SurveillanceSeverity.CRITICAL, True, 0.0, 1, True,
            ),
            SurveillanceRule(
                "SURV-EA-001", "Excessive Slippage", SurveillanceFamily.EXECUTION_ANOMALY,
                "Average fill slippage exceeds 10bps",
                SurveillanceSeverity.HIGH, True, 0.001, 20, False,
            ),
            SurveillanceRule(
                "SURV-EA-002", "Order Rate Anomaly", SurveillanceFamily.EXECUTION_ANOMALY,
                "Order rate exceeds 10x rolling average (potential runaway loop)",
                SurveillanceSeverity.CRITICAL, True, 10.0, 10, True,
            ),
            SurveillanceRule(
                "SURV-MD-001", "Model IC Degradation", SurveillanceFamily.MODEL_DEGRADATION,
                "Rolling IC drops below 0.02 (near zero predictive power)",
                SurveillanceSeverity.HIGH, True, 0.02, 60, False,
            ),
            SurveillanceRule(
                "SURV-MD-002", "Feature Drift PSI Alert", SurveillanceFamily.MODEL_DEGRADATION,
                "PSI > 0.25 indicates significant feature distribution shift",
                SurveillanceSeverity.HIGH, True, 0.25, 1, False,
            ),
            SurveillanceRule(
                "SURV-CB-001", "Sector Concentration Breach", SurveillanceFamily.CONCENTRATION_BREACH,
                "Single sector exposure exceeds 40% of portfolio",
                SurveillanceSeverity.HIGH, True, 0.40, 1, False,
            ),
            SurveillanceRule(
                "SURV-DI-001", "Stale Price Data", SurveillanceFamily.DATA_INTEGRITY,
                "Price data older than 48 hours used in live computation",
                SurveillanceSeverity.HIGH, True, 48.0, 1, True,
            ),
            SurveillanceRule(
                "SURV-AB-001", "Agent Error Rate Spike", SurveillanceFamily.AGENT_BEHAVIOR,
                "Agent task failure rate exceeds 20% over last 100 tasks",
                SurveillanceSeverity.MEDIUM, True, 0.20, 100, False,
            ),
            SurveillanceRule(
                "SURV-RL-001", "Drawdown Approaching Limit", SurveillanceFamily.RISK_LIMIT_APPROACH,
                "Portfolio drawdown > 80% of hard limit (early warning)",
                SurveillanceSeverity.MEDIUM, True, 0.80, 1, False,
            ),
        ]
        for rule in defaults:
            self._rules[rule.rule_id] = rule

    def register_rule(self, rule: SurveillanceRule) -> None:
        """Register or replace a surveillance rule."""
        self._rules[rule.rule_id] = rule

    def detect(
        self,
        rule_id: str,
        entity_type: str,
        entity_id: str,
        metric_value: float,
        evidence: list[str] = None,
        description_override: str = None,
    ) -> Optional[SurveillanceEvent]:
        """
        Submit a metric value for evaluation against a rule.
        Returns a SurveillanceEvent if the threshold is breached, else None.
        """
        rule = self._rules.get(rule_id)
        if not rule or not rule.enabled:
            return None

        # Default: metric_value >= threshold triggers
        breached = metric_value >= rule.threshold

        # Special cases where lower is worse (IC degradation)
        lower_is_bad_rules = {"SURV-MD-001"}
        if rule_id in lower_is_bad_rules:
            breached = metric_value < rule.threshold

        if not breached:
            return None

        with self._lock:
            event = SurveillanceEvent(
                event_id=str(uuid.uuid4()),
                rule_id=rule_id,
                family=rule.family,
                severity=rule.severity,
                detected_at=datetime.utcnow(),
                entity_type=entity_type,
                entity_id=entity_id,
                description=(
                    description_override
                    or f"{rule.name}: {entity_id} triggered threshold {rule.threshold} with value {metric_value}"
                ),
                metric_value=metric_value,
                threshold_value=rule.threshold,
                evidence=tuple(evidence or []),
                status=SurveillanceEventStatus.OPEN,
                acknowledged_by=None,
                resolved_at=None,
                case_id=None,
            )
            self._events[event.event_id] = event
            self._event_log.append(event)

            if rule.auto_escalate:
                self._auto_create_case(event)

            return self._events[event.event_id]

    def _auto_create_case(self, event: SurveillanceEvent) -> None:
        """Create a SurveillanceCase automatically for high-severity events."""
        case = SurveillanceCase(
            case_id=str(uuid.uuid4()),
            title=f"[AUTO] {event.description[:80]}",
            family=event.family,
            status=SurveillanceCaseStatus.OPEN,
            opened_at=datetime.utcnow(),
            opened_by="surveillance_engine",
            assigned_to=None,
            event_ids=(event.event_id,),
            findings="",
            disposition=None,
            closed_at=None,
            closed_by=None,
            incident_id=None,
        )
        self._cases[case.case_id] = case

        # Link event to case by replacing the immutable event with an updated version
        linked_event = SurveillanceEvent(
            event_id=event.event_id,
            rule_id=event.rule_id,
            family=event.family,
            severity=event.severity,
            detected_at=event.detected_at,
            entity_type=event.entity_type,
            entity_id=event.entity_id,
            description=event.description,
            metric_value=event.metric_value,
            threshold_value=event.threshold_value,
            evidence=event.evidence,
            status=event.status,
            acknowledged_by=event.acknowledged_by,
            resolved_at=event.resolved_at,
            case_id=case.case_id,
        )
        self._events[event.event_id] = linked_event
        if self._event_log and self._event_log[-1].event_id == event.event_id:
            self._event_log[-1] = linked_event

    def acknowledge_event(self, event_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an open surveillance event."""
        if event_id not in self._events:
            return False
        e = self._events[event_id]
        self._events[event_id] = SurveillanceEvent(
            event_id=e.event_id,
            rule_id=e.rule_id,
            family=e.family,
            severity=e.severity,
            detected_at=e.detected_at,
            entity_type=e.entity_type,
            entity_id=e.entity_id,
            description=e.description,
            metric_value=e.metric_value,
            threshold_value=e.threshold_value,
            evidence=e.evidence,
            status=SurveillanceEventStatus.ACKNOWLEDGED,
            acknowledged_by=acknowledged_by,
            resolved_at=e.resolved_at,
            case_id=e.case_id,
        )
        return True

    def resolve_event(self, event_id: str, resolution: str = "") -> bool:
        """Resolve a surveillance event."""
        if event_id not in self._events:
            return False
        e = self._events[event_id]
        self._events[event_id] = SurveillanceEvent(
            event_id=e.event_id,
            rule_id=e.rule_id,
            family=e.family,
            severity=e.severity,
            detected_at=e.detected_at,
            entity_type=e.entity_type,
            entity_id=e.entity_id,
            description=e.description,
            metric_value=e.metric_value,
            threshold_value=e.threshold_value,
            evidence=e.evidence,
            status=SurveillanceEventStatus.RESOLVED,
            acknowledged_by=e.acknowledged_by,
            resolved_at=datetime.utcnow(),
            case_id=e.case_id,
        )
        return True

    def create_case(
        self,
        title: str,
        family: SurveillanceFamily,
        event_ids: list[str],
        opened_by: str,
    ) -> SurveillanceCase:
        """Manually create a surveillance case grouping one or more events."""
        case = SurveillanceCase(
            case_id=str(uuid.uuid4()),
            title=title,
            family=family,
            status=SurveillanceCaseStatus.OPEN,
            opened_at=datetime.utcnow(),
            opened_by=opened_by,
            assigned_to=None,
            event_ids=tuple(event_ids),
            findings="",
            disposition=None,
            closed_at=None,
            closed_by=None,
            incident_id=None,
        )
        self._cases[case.case_id] = case
        return case

    def close_case(
        self,
        case_id: str,
        closed_by: str,
        disposition: str,
        findings: str,
        incident_id: str = None,
    ) -> bool:
        """Close a surveillance case with a disposition."""
        if case_id not in self._cases:
            return False
        c = self._cases[case_id]
        new_status = (
            SurveillanceCaseStatus.CLOSED_ACTION_TAKEN
            if disposition != "no_action"
            else SurveillanceCaseStatus.CLOSED_CLEAN
        )
        self._cases[case_id] = SurveillanceCase(
            case_id=c.case_id,
            title=c.title,
            family=c.family,
            status=new_status,
            opened_at=c.opened_at,
            opened_by=c.opened_by,
            assigned_to=c.assigned_to,
            event_ids=c.event_ids,
            findings=findings,
            disposition=disposition,
            closed_at=datetime.utcnow(),
            closed_by=closed_by,
            incident_id=incident_id,
        )
        return True

    def get_open_events(
        self,
        family: SurveillanceFamily = None,
        severity: SurveillanceSeverity = None,
    ) -> list[SurveillanceEvent]:
        """Return all open surveillance events, optionally filtered."""
        events = [e for e in self._events.values() if e.status == SurveillanceEventStatus.OPEN]
        if family:
            events = [e for e in events if e.family == family]
        if severity:
            events = [e for e in events if e.severity == severity]
        return events

    def get_open_cases(self) -> list[SurveillanceCase]:
        """Return all open or in-review surveillance cases."""
        open_statuses = (
            SurveillanceCaseStatus.OPEN,
            SurveillanceCaseStatus.IN_REVIEW,
            SurveillanceCaseStatus.ESCALATED,
        )
        return [c for c in self._cases.values() if c.status in open_statuses]

    def get_metrics(self) -> dict:
        """Return aggregate metrics for monitoring dashboards."""
        return {
            "total_rules": len(self._rules),
            "total_events": len(self._events),
            "open_events": len(self.get_open_events()),
            "critical_open": len(self.get_open_events(severity=SurveillanceSeverity.CRITICAL)),
            "total_cases": len(self._cases),
            "open_cases": len(self.get_open_cases()),
        }

    def get_rule(self, rule_id: str) -> Optional[SurveillanceRule]:
        """Look up a rule by ID."""
        return self._rules.get(rule_id)

    def list_rules(
        self,
        family: SurveillanceFamily = None,
        enabled_only: bool = False,
    ) -> list[SurveillanceRule]:
        """List all rules, optionally filtered by family or enabled status."""
        rules = list(self._rules.values())
        if family:
            rules = [r for r in rules if r.family == family]
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        return rules


# ── Singleton accessor ─────────────────────────────────────────────────────────

_engine: Optional[SurveillanceEngine] = None


def get_surveillance_engine() -> SurveillanceEngine:
    """Return the module-level singleton SurveillanceEngine."""
    global _engine
    if _engine is None:
        _engine = SurveillanceEngine()
    return _engine
