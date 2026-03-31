# -*- coding: utf-8 -*-
"""
alerts/contracts.py — Alert Management Domain Contracts
========================================================

All typed domain objects for the alert subsystem.

Design principles:
  - AlertRule and AlertAcknowledgement/EscalationRecord are frozen
    (immutable value objects).
  - AlertEvent is a mutable dataclass so the engine can update status,
    acknowledgment, and resolution fields in-place.
  - Enums inherit from str for JSON round-trip compatibility.
  - stdlib only — no external dependencies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════
# 1. ENUMERATIONS
# ══════════════════════════════════════════════════════════════════


class AlertSeverity(str, enum.Enum):
    """Severity tiers for alert events.

    Higher severity implies faster required response and potentially
    automated escalation.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, enum.Enum):
    """Lifecycle status of an alert event."""

    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class AlertFamily(str, enum.Enum):
    """Functional domain that produced the alert."""

    DATA = "data"
    BROKER = "broker"
    EXECUTION = "execution"
    RISK = "risk"
    MODEL = "model"
    ORCHESTRATION = "orchestration"
    DEPLOYMENT = "deployment"
    SYSTEM = "system"
    RECONCILIATION = "reconciliation"
    POLICY = "policy"


# ══════════════════════════════════════════════════════════════════
# 2. RULE DEFINITION
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AlertRule:
    """Configuration for a single alert rule.

    Alert rules are registered with the AlertEngine at startup and
    referenced by rule_id when firing events.

    Fields
    ------
    rule_id : str
        Unique identifier for this rule (e.g. "STALE_DATA").
    name : str
        Short human-readable name.
    description : str
        Full description of what this rule detects.
    family : AlertFamily
        Functional domain category.
    severity : AlertSeverity
        Default severity for events fired by this rule.
    condition : str
        Human-readable description of the condition that fires the rule.
    dedup_key : str
        Deduplication key template: "{family}.{name}.{scope}".
    suppression_window_s : float
        Minimum seconds between successive fires of the same dedup_key.
    flap_threshold : int
        Number of fires within ``suppression_window_s`` that constitutes
        flapping behaviour.
    auto_resolve_s : Optional[float]
        If set, automatically resolve the alert after this many seconds
        of not being re-fired.
    requires_acknowledgment : bool
        Whether an operator must explicitly acknowledge this alert.
    runbook_id : Optional[str]
        Reference to the applicable operational runbook.
    incident_template : Optional[str]
        Title template for auto-created incidents.
    escalation_after_s : Optional[float]
        Seconds after firing without acknowledgment before escalation.
    routing_destination : str
        Target routing group: "ops" | "on_call" | "risk_team" |
        "data_team".
    enabled : bool
        Whether the rule is active.
    """

    rule_id: str
    name: str
    description: str
    family: AlertFamily
    severity: AlertSeverity
    condition: str
    dedup_key: str
    suppression_window_s: float
    flap_threshold: int
    auto_resolve_s: Optional[float]
    requires_acknowledgment: bool
    runbook_id: Optional[str]
    incident_template: Optional[str]
    escalation_after_s: Optional[float]
    routing_destination: str
    enabled: bool = True


# ══════════════════════════════════════════════════════════════════
# 3. ALERT EVENTS (MUTABLE)
# ══════════════════════════════════════════════════════════════════


@dataclass
class AlertEvent:
    """A single firing instance of an alert rule.

    Mutable so the engine can update status, acknowledgment, and
    resolution fields without creating new objects.

    Fields
    ------
    event_id : str
        UUID for this event instance.
    rule_id : str
        Identifier of the AlertRule that produced this event.
    rule_name : str
        Human-readable rule name (denormalised for display).
    family : AlertFamily
        Alert family (denormalised from the rule).
    severity : AlertSeverity
        Event severity at fire time.
    status : AlertStatus
        Current lifecycle status.
    source : str
        Component that fired this alert.
    scope : str
        Scope of impact: "global" | "strategy:{id}" | "feed:{name}" | etc.
    message : str
        Human-readable description of the alert condition.
    details : dict
        Structured diagnostic data.
    fired_at : str
        ISO-8601 timestamp when the alert was first fired.
    dedup_key : str
        Effective deduplication key for this event.
    acknowledged_by : Optional[str]
        Operator identifier who acknowledged this alert.
    acknowledged_at : Optional[str]
        ISO-8601 timestamp of acknowledgment.
    resolved_at : Optional[str]
        ISO-8601 timestamp of resolution.
    suppressed_until : Optional[str]
        ISO-8601 timestamp until which re-fires are suppressed.
    incident_id : Optional[str]
        Linked incident record identifier.
    escalated : bool
        True if this event has been escalated.
    """

    event_id: str
    rule_id: str
    rule_name: str
    family: AlertFamily
    severity: AlertSeverity
    status: AlertStatus
    source: str
    scope: str
    message: str
    details: dict
    fired_at: str
    dedup_key: str
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    suppressed_until: Optional[str] = None
    incident_id: Optional[str] = None
    escalated: bool = False


# ══════════════════════════════════════════════════════════════════
# 4. ACKNOWLEDGMENT AND ESCALATION RECORDS (IMMUTABLE)
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AlertAcknowledgement:
    """Immutable record of an operator acknowledgment.

    Fields
    ------
    ack_id : str
        UUID for this acknowledgment record.
    event_id : str
        Identifier of the AlertEvent being acknowledged.
    acknowledged_by : str
        Operator identifier.
    acknowledged_at : str
        ISO-8601 timestamp.
    notes : str
        Free-text operator notes.
    snooze_until : Optional[str]
        ISO-8601 timestamp until which re-fires are suppressed (snooze).
    """

    ack_id: str
    event_id: str
    acknowledged_by: str
    acknowledged_at: str
    notes: str
    snooze_until: Optional[str]


@dataclass(frozen=True)
class EscalationRecord:
    """Immutable record of an escalation action.

    Fields
    ------
    escalation_id : str
        UUID for this escalation record.
    event_id : str
        Identifier of the AlertEvent that was escalated.
    escalated_at : str
        ISO-8601 timestamp of escalation.
    escalated_to : str
        Target team or on-call identifier.
    reason : str
        Human-readable reason for escalation.
    acknowledged : bool
        True if the escalated party has acknowledged.
    """

    escalation_id: str
    event_id: str
    escalated_at: str
    escalated_to: str
    reason: str
    acknowledged: bool = False
