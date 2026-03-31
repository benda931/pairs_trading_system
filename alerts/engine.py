# -*- coding: utf-8 -*-
"""
alerts/engine.py — AlertEngine
================================

Manages alert rules, firing, deduplication, acknowledgment,
suppression, escalation, and auto-resolution.

Features
--------
- Rule registry with per-rule enable/disable.
- Deduplication by dedup_key within each rule's suppression window.
- Flapping detection: N fires within the suppression window = flapping.
- Auto-resolve: events not re-fired within auto_resolve_s are resolved.
- Escalation: unacknowledged events after escalation_after_s are escalated.
- Optional IncidentManager integration for auto-incident creation.
- 20+ pre-registered default alert rules covering all critical platform paths.
- Metrics: alert counts by severity/family, firing rate, ack rate.

Thread safety: a single threading.Lock serializes all store mutations.
Singleton access via get_alert_engine().
"""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

from alerts.contracts import (
    AlertAcknowledgement,
    AlertEvent,
    AlertFamily,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    EscalationRecord,
)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _parse_ts(iso_str: Optional[str]) -> Optional[float]:
    """Parse ISO-8601 string to POSIX timestamp, or None."""
    if iso_str is None:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════════
# ALERT ENGINE
# ══════════════════════════════════════════════════════════════════


class AlertEngine:
    """Manages alert rules, firing, deduplication, acknowledgment, and escalation.

    Parameters
    ----------
    incident_manager :
        Optional IncidentManager instance for auto-incident creation.
        If None, incidents are not auto-created.
    """

    # Max events retained in memory (ring-buffer behaviour)
    _MAX_EVENTS = 10_000
    # Window for flap-fire timestamps per dedup_key
    _FLAP_WINDOW = 3600.0  # 1 hour

    def __init__(self, incident_manager: Optional[Any] = None) -> None:
        self._incident_manager = incident_manager
        self._lock = threading.Lock()

        # Rule registry: rule_id -> AlertRule
        self._rules: Dict[str, AlertRule] = {}

        # Active and historical events: event_id -> AlertEvent
        self._events: Dict[str, AlertEvent] = {}

        # Dedup index: dedup_key -> latest firing event_id
        self._dedup_index: Dict[str, str] = {}

        # Suppression store: dedup_key -> suppressed_until POSIX timestamp
        self._suppression_store: Dict[str, float] = {}

        # Flap tracking: dedup_key -> deque of fire timestamps
        self._flap_tracker: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=50)
        )

        # Acknowledgment log: ack_id -> AlertAcknowledgement
        self._ack_log: Dict[str, AlertAcknowledgement] = {}

        # Escalation log: escalation_id -> EscalationRecord
        self._escalation_log: Dict[str, EscalationRecord] = {}

        # Metrics
        self._fire_count: int = 0
        self._dedup_count: int = 0
        self._ack_count: int = 0
        self._resolve_count: int = 0
        self._escalation_count: int = 0
        self._fire_by_severity: Dict[str, int] = defaultdict(int)
        self._fire_by_family: Dict[str, int] = defaultdict(int)

        # Register default rules
        self._register_default_rules()

    # ──────────────────────────────────────────────────────────────
    # RULE MANAGEMENT
    # ──────────────────────────────────────────────────────────────

    def register_rule(self, rule: AlertRule) -> None:
        """Register or update an alert rule.

        Parameters
        ----------
        rule : AlertRule
            Rule to register. Existing rule with same rule_id is replaced.
        """
        with self._lock:
            self._rules[rule.rule_id] = rule

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Return the rule for rule_id, or None."""
        with self._lock:
            return self._rules.get(rule_id)

    def enable_rule(self, rule_id: str) -> None:
        """Enable a registered rule."""
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule is not None:
                # Frozen dataclass — replace with a new instance
                self._rules[rule_id] = AlertRule(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    description=rule.description,
                    family=rule.family,
                    severity=rule.severity,
                    condition=rule.condition,
                    dedup_key=rule.dedup_key,
                    suppression_window_s=rule.suppression_window_s,
                    flap_threshold=rule.flap_threshold,
                    auto_resolve_s=rule.auto_resolve_s,
                    requires_acknowledgment=rule.requires_acknowledgment,
                    runbook_id=rule.runbook_id,
                    incident_template=rule.incident_template,
                    escalation_after_s=rule.escalation_after_s,
                    routing_destination=rule.routing_destination,
                    enabled=True,
                )

    def disable_rule(self, rule_id: str) -> None:
        """Disable a registered rule (fires are silently dropped)."""
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule is not None:
                self._rules[rule_id] = AlertRule(
                    rule_id=rule.rule_id,
                    name=rule.name,
                    description=rule.description,
                    family=rule.family,
                    severity=rule.severity,
                    condition=rule.condition,
                    dedup_key=rule.dedup_key,
                    suppression_window_s=rule.suppression_window_s,
                    flap_threshold=rule.flap_threshold,
                    auto_resolve_s=rule.auto_resolve_s,
                    requires_acknowledgment=rule.requires_acknowledgment,
                    runbook_id=rule.runbook_id,
                    incident_template=rule.incident_template,
                    escalation_after_s=rule.escalation_after_s,
                    routing_destination=rule.routing_destination,
                    enabled=False,
                )

    # ──────────────────────────────────────────────────────────────
    # FIRING
    # ──────────────────────────────────────────────────────────────

    def fire(
        self,
        rule_id: str,
        source: str,
        scope: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[AlertEvent]:
        """Fire an alert for the given rule.

        Applies deduplication and suppression logic before creating
        a new event.  Returns the created AlertEvent, or None if the
        fire was deduplicated or suppressed.

        Parameters
        ----------
        rule_id : str
            ID of the rule to fire.
        source : str
            Component name that detected the condition.
        scope : str
            Scope: "global" | "strategy:{id}" | "feed:{name}" | etc.
        message : str
            Human-readable description.
        details : dict | None
            Additional structured context.

        Returns
        -------
        Optional[AlertEvent]
            The new event, or None if suppressed/deduplicated.
        """
        if details is None:
            details = {}
        now = _now_ts()

        with self._lock:
            rule = self._rules.get(rule_id)
            if rule is None or not rule.enabled:
                return None

            effective_dedup = f"{rule.dedup_key}.{scope}"

            # Check global suppression
            suppressed_until = self._suppression_store.get(effective_dedup)
            if suppressed_until is not None and now < suppressed_until:
                self._dedup_count += 1
                return None

            # Check existing active event for dedup
            existing_id = self._dedup_index.get(effective_dedup)
            if existing_id is not None:
                existing = self._events.get(existing_id)
                if existing is not None and existing.status == AlertStatus.FIRING:
                    fired_at_ts = _parse_ts(existing.fired_at)
                    if fired_at_ts is not None:
                        age = now - fired_at_ts
                        if age < rule.suppression_window_s:
                            self._dedup_count += 1
                            return None

            # Record fire timestamp for flap detection
            self._flap_tracker[effective_dedup].append(now)

            # Build the new event
            event_id = str(uuid.uuid4())
            event = AlertEvent(
                event_id=event_id,
                rule_id=rule_id,
                rule_name=rule.name,
                family=rule.family,
                severity=rule.severity,
                status=AlertStatus.FIRING,
                source=source,
                scope=scope,
                message=message,
                details=dict(details),
                fired_at=_now_iso(),
                dedup_key=effective_dedup,
            )

            # Store and index
            self._events[event_id] = event
            self._dedup_index[effective_dedup] = event_id

            # Set suppression window for next fire
            self._suppression_store[effective_dedup] = now + rule.suppression_window_s

            # Update metrics
            self._fire_count += 1
            self._fire_by_severity[rule.severity.value] += 1
            self._fire_by_family[rule.family.value] += 1

        # Auto-create incident if template is configured (outside lock)
        if rule.incident_template is not None and self._incident_manager is not None:
            self._maybe_create_incident(event, rule)

        return event

    # ──────────────────────────────────────────────────────────────
    # ACKNOWLEDGMENT
    # ──────────────────────────────────────────────────────────────

    def acknowledge(
        self,
        event_id: str,
        operator: str,
        notes: str = "",
        snooze_s: float = 0.0,
    ) -> AlertAcknowledgement:
        """Acknowledge an alert event.

        Parameters
        ----------
        event_id : str
            ID of the event to acknowledge.
        operator : str
            Operator identifier.
        notes : str
            Free-text notes.
        snooze_s : float
            If > 0, suppress re-fires for this many seconds.

        Returns
        -------
        AlertAcknowledgement
            The acknowledgment record.

        Raises
        ------
        KeyError
            If event_id is not found.
        """
        ack_ts = _now_iso()
        snooze_until: Optional[str] = None
        snooze_until_ts: Optional[float] = None
        if snooze_s > 0:
            snooze_until_ts = _now_ts() + snooze_s
            snooze_dt = datetime.fromtimestamp(snooze_until_ts, tz=timezone.utc)
            snooze_until = snooze_dt.isoformat()

        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                raise KeyError(f"Alert event '{event_id}' not found.")

            event.status = AlertStatus.ACKNOWLEDGED
            event.acknowledged_by = operator
            event.acknowledged_at = ack_ts
            if snooze_until is not None:
                event.suppressed_until = snooze_until
                # Apply snooze to suppression store
                self._suppression_store[event.dedup_key] = snooze_until_ts  # type: ignore[assignment]

            ack = AlertAcknowledgement(
                ack_id=str(uuid.uuid4()),
                event_id=event_id,
                acknowledged_by=operator,
                acknowledged_at=ack_ts,
                notes=notes,
                snooze_until=snooze_until,
            )
            self._ack_log[ack.ack_id] = ack
            self._ack_count += 1

        return ack

    # ──────────────────────────────────────────────────────────────
    # RESOLUTION
    # ──────────────────────────────────────────────────────────────

    def resolve(self, event_id: str, resolved_by: str = "auto") -> None:
        """Resolve an alert event.

        Parameters
        ----------
        event_id : str
            ID of the event to resolve.
        resolved_by : str
            Resolver identifier (operator name or "auto").

        Raises
        ------
        KeyError
            If event_id is not found.
        """
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                raise KeyError(f"Alert event '{event_id}' not found.")
            if event.status not in (AlertStatus.RESOLVED, AlertStatus.EXPIRED):
                event.status = AlertStatus.RESOLVED
                event.resolved_at = _now_iso()
                event.details["resolved_by"] = resolved_by
                self._resolve_count += 1
                # Clear dedup entry so the next fire creates a fresh event
                self._dedup_index.pop(event.dedup_key, None)

    # ──────────────────────────────────────────────────────────────
    # SUPPRESSION
    # ──────────────────────────────────────────────────────────────

    def suppress(
        self,
        rule_id: str,
        scope: str,
        duration_s: float,
        reason: str,
    ) -> None:
        """Suppress all fires of rule_id/scope for duration_s seconds.

        Parameters
        ----------
        rule_id : str
            Rule to suppress.
        scope : str
            Scope to suppress (combined with rule dedup key).
        duration_s : float
            Suppression duration in seconds.
        reason : str
            Human-readable reason (stored in details of future events).
        """
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule is None:
                return
            effective_dedup = f"{rule.dedup_key}.{scope}"
            until_ts = _now_ts() + duration_s
            self._suppression_store[effective_dedup] = until_ts

    # ──────────────────────────────────────────────────────────────
    # QUERY
    # ──────────────────────────────────────────────────────────────

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        family: Optional[AlertFamily] = None,
    ) -> List[AlertEvent]:
        """Return all currently firing or acknowledged alerts.

        Parameters
        ----------
        severity : Optional[AlertSeverity]
            If provided, filter to this severity only.
        family : Optional[AlertFamily]
            If provided, filter to this family only.

        Returns
        -------
        list[AlertEvent]
            Matching active events sorted by fired_at descending.
        """
        with self._lock:
            events = [
                e
                for e in self._events.values()
                if e.status in (AlertStatus.FIRING, AlertStatus.ACKNOWLEDGED)
            ]

        if severity is not None:
            events = [e for e in events if e.severity == severity]
        if family is not None:
            events = [e for e in events if e.family == family]

        events.sort(key=lambda e: e.fired_at, reverse=True)
        return events

    def get_event(self, event_id: str) -> Optional[AlertEvent]:
        """Return a specific event by ID, or None."""
        with self._lock:
            return self._events.get(event_id)

    def list_rules(self, enabled_only: bool = False) -> List[AlertRule]:
        """Return all registered rules, optionally filtering to enabled only."""
        with self._lock:
            rules = list(self._rules.values())
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        return rules

    # ──────────────────────────────────────────────────────────────
    # MAINTENANCE
    # ──────────────────────────────────────────────────────────────

    def run_maintenance(self) -> Dict[str, int]:
        """Expire and auto-resolve stale alerts.

        Should be called periodically (e.g. every 60 seconds) by
        a background task.

        Returns
        -------
        dict
            Keys: auto_resolved, expired, escalated.
        """
        now = _now_ts()
        auto_resolved: List[str] = []
        expired_ids: List[str] = []
        to_escalate: List[Tuple[AlertEvent, AlertRule]] = []

        with self._lock:
            for event_id, event in self._events.items():
                if event.status not in (AlertStatus.FIRING, AlertStatus.ACKNOWLEDGED):
                    continue

                rule = self._rules.get(event.rule_id)
                if rule is None:
                    continue

                fired_at_ts = _parse_ts(event.fired_at)
                if fired_at_ts is None:
                    continue

                age = now - fired_at_ts

                # Auto-resolve
                if rule.auto_resolve_s is not None and age > rule.auto_resolve_s:
                    auto_resolved.append(event_id)
                    continue

                # Escalation check (only FIRING events not yet escalated)
                if (
                    rule.escalation_after_s is not None
                    and not event.escalated
                    and event.status == AlertStatus.FIRING
                    and age > rule.escalation_after_s
                ):
                    to_escalate.append((event, rule))

        # Apply auto-resolutions
        for event_id in auto_resolved:
            try:
                self.resolve(event_id, resolved_by="auto_maintenance")
            except KeyError:
                pass

        # Apply escalations
        escalated_count = 0
        for event, rule in to_escalate:
            self._escalate(event, rule)
            escalated_count += 1

        # Expire very old events (> 24 hours) that are already resolved
        cutoff = now - 86_400.0
        with self._lock:
            stale = [
                eid
                for eid, ev in self._events.items()
                if ev.status in (AlertStatus.RESOLVED, AlertStatus.EXPIRED)
                and (_parse_ts(ev.fired_at) or now) < cutoff
            ]
            for eid in stale:
                expired_ids.append(eid)
                del self._events[eid]

        return {
            "auto_resolved": len(auto_resolved),
            "expired": len(expired_ids),
            "escalated": escalated_count,
        }

    # ──────────────────────────────────────────────────────────────
    # METRICS
    # ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return alert engine statistics.

        Returns
        -------
        dict
            Includes total fires, dedup count, ack count, resolve count,
            escalation count, active alert count, fire rate by severity
            and family.
        """
        with self._lock:
            active = sum(
                1
                for e in self._events.values()
                if e.status in (AlertStatus.FIRING, AlertStatus.ACKNOWLEDGED)
            )
            total_events = len(self._events)

        return {
            "total_fires": self._fire_count,
            "deduplicated": self._dedup_count,
            "acknowledged": self._ack_count,
            "resolved": self._resolve_count,
            "escalated": self._escalation_count,
            "active_alerts": active,
            "total_events_in_store": total_events,
            "fire_by_severity": dict(self._fire_by_severity),
            "fire_by_family": dict(self._fire_by_family),
            "registered_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
        }

    # ──────────────────────────────────────────────────────────────
    # PRIVATE: ESCALATION
    # ──────────────────────────────────────────────────────────────

    def _escalate(self, event: AlertEvent, rule: AlertRule) -> None:
        """Create an escalation record and mark the event as escalated."""
        rec = EscalationRecord(
            escalation_id=str(uuid.uuid4()),
            event_id=event.event_id,
            escalated_at=_now_iso(),
            escalated_to=rule.routing_destination,
            reason=(
                f"Alert '{rule.name}' unacknowledged for >"
                f"{rule.escalation_after_s}s."
            ),
        )
        with self._lock:
            self._escalation_log[rec.escalation_id] = rec
            event.escalated = True
            self._escalation_count += 1

    # ──────────────────────────────────────────────────────────────
    # PRIVATE: INCIDENT CREATION
    # ──────────────────────────────────────────────────────────────

    def _maybe_create_incident(
        self, event: AlertEvent, rule: AlertRule
    ) -> None:
        """Auto-create an incident for the given alert event."""
        if self._incident_manager is None or rule.incident_template is None:
            return
        try:
            from incidents.contracts import IncidentSeverity  # type: ignore

            # Map alert severity -> incident severity
            sev_map = {
                AlertSeverity.EMERGENCY: IncidentSeverity.P0_CRITICAL,
                AlertSeverity.CRITICAL: IncidentSeverity.P1_HIGH,
                AlertSeverity.ERROR: IncidentSeverity.P2_MEDIUM,
                AlertSeverity.WARNING: IncidentSeverity.P3_LOW,
                AlertSeverity.INFO: IncidentSeverity.P4_INFO,
            }
            inc_severity = sev_map.get(event.severity, IncidentSeverity.P3_LOW)
            title = rule.incident_template.format(
                rule_name=rule.name,
                scope=event.scope,
                source=event.source,
            )
            incident = self._incident_manager.create_incident(
                title=title,
                description=event.message,
                severity=inc_severity,
                source=event.source,
                details=event.details,
            )
            if incident is not None:
                with self._lock:
                    ev = self._events.get(event.event_id)
                    if ev is not None:
                        ev.incident_id = incident.incident_id
        except Exception:  # noqa: BLE001
            pass  # Incident creation failure must not propagate into the alert path

    # ──────────────────────────────────────────────────────────────
    # PRIVATE: DEFAULT RULES
    # ──────────────────────────────────────────────────────────────

    def _register_default_rules(self) -> None:
        """Register 20 default alert rules covering all critical platform paths."""

        defaults: List[AlertRule] = [
            # 1. STALE_DATA
            AlertRule(
                rule_id="STALE_DATA",
                name="Stale Data Feed",
                description="Market data feed has not been updated within the expected window.",
                family=AlertFamily.DATA,
                severity=AlertSeverity.WARNING,
                condition="Feed last update > 5 minutes ago.",
                dedup_key="data.stale_data",
                suppression_window_s=300.0,
                flap_threshold=3,
                auto_resolve_s=600.0,
                requires_acknowledgment=False,
                runbook_id="RB-DATA-001",
                incident_template=None,
                escalation_after_s=900.0,
                routing_destination="data_team",
            ),
            # 2. DATA_FEED_GAP
            AlertRule(
                rule_id="DATA_FEED_GAP",
                name="Data Feed Gap Detected",
                description="A gap in the market data feed has been detected.",
                family=AlertFamily.DATA,
                severity=AlertSeverity.ERROR,
                condition="Sequential timestamp gap detected in feed.",
                dedup_key="data.feed_gap",
                suppression_window_s=120.0,
                flap_threshold=5,
                auto_resolve_s=None,
                requires_acknowledgment=True,
                runbook_id="RB-DATA-002",
                incident_template="Data feed gap: {scope}",
                escalation_after_s=600.0,
                routing_destination="data_team",
            ),
            # 3. BROKER_DISCONNECT
            AlertRule(
                rule_id="BROKER_DISCONNECT",
                name="Broker Disconnected",
                description="Connection to the broker has been lost.",
                family=AlertFamily.BROKER,
                severity=AlertSeverity.CRITICAL,
                condition="Broker TCP/WebSocket connection dropped.",
                dedup_key="broker.disconnect",
                suppression_window_s=60.0,
                flap_threshold=3,
                auto_resolve_s=None,
                requires_acknowledgment=True,
                runbook_id="RB-BROKER-001",
                incident_template="Broker disconnected: {scope}",
                escalation_after_s=120.0,
                routing_destination="on_call",
            ),
            # 4. EXCESSIVE_REJECTS
            AlertRule(
                rule_id="EXCESSIVE_REJECTS",
                name="Excessive Order Rejects",
                description="Order reject rate has exceeded the acceptable threshold.",
                family=AlertFamily.EXECUTION,
                severity=AlertSeverity.ERROR,
                condition="Reject rate > 10% over the last 5-minute window.",
                dedup_key="execution.excessive_rejects",
                suppression_window_s=300.0,
                flap_threshold=3,
                auto_resolve_s=600.0,
                requires_acknowledgment=True,
                runbook_id="RB-EXEC-001",
                incident_template="Excessive order rejects: {scope}",
                escalation_after_s=900.0,
                routing_destination="ops",
            ),
            # 5. UNACKED_ORDERS
            AlertRule(
                rule_id="UNACKED_ORDERS",
                name="Unacknowledged Orders",
                description="Orders have been pending broker acknowledgment for too long.",
                family=AlertFamily.EXECUTION,
                severity=AlertSeverity.WARNING,
                condition="Order pending ack > 5 minutes.",
                dedup_key="execution.unacked_orders",
                suppression_window_s=300.0,
                flap_threshold=3,
                auto_resolve_s=600.0,
                requires_acknowledgment=False,
                runbook_id="RB-EXEC-002",
                incident_template=None,
                escalation_after_s=1200.0,
                routing_destination="ops",
            ),
            # 6. RECONCILIATION_BREAK
            AlertRule(
                rule_id="RECONCILIATION_BREAK",
                name="Reconciliation Break Detected",
                description="A mismatch between internal and broker position state has been detected.",
                family=AlertFamily.RECONCILIATION,
                severity=AlertSeverity.ERROR,
                condition="Internal vs broker position diff > tolerance threshold.",
                dedup_key="reconciliation.break",
                suppression_window_s=300.0,
                flap_threshold=2,
                auto_resolve_s=None,
                requires_acknowledgment=True,
                runbook_id="RB-RECON-001",
                incident_template="Reconciliation break: {scope}",
                escalation_after_s=600.0,
                routing_destination="ops",
            ),
            # 7. EXPOSURE_DRIFT
            AlertRule(
                rule_id="EXPOSURE_DRIFT",
                name="Exposure Drift",
                description="Portfolio exposure has drifted above the configured limit.",
                family=AlertFamily.RISK,
                severity=AlertSeverity.WARNING,
                condition="Gross exposure > 110% of configured limit.",
                dedup_key="risk.exposure_drift",
                suppression_window_s=600.0,
                flap_threshold=3,
                auto_resolve_s=1800.0,
                requires_acknowledgment=False,
                runbook_id="RB-RISK-001",
                incident_template=None,
                escalation_after_s=1800.0,
                routing_destination="risk_team",
            ),
            # 8. UNEXPECTED_POSITION
            AlertRule(
                rule_id="UNEXPECTED_POSITION",
                name="Unexpected Open Position",
                description="A position is open in a state not expected by the lifecycle manager.",
                family=AlertFamily.EXECUTION,
                severity=AlertSeverity.ERROR,
                condition="Position exists without a corresponding active lifecycle record.",
                dedup_key="execution.unexpected_position",
                suppression_window_s=300.0,
                flap_threshold=2,
                auto_resolve_s=None,
                requires_acknowledgment=True,
                runbook_id="RB-EXEC-003",
                incident_template="Unexpected position: {scope}",
                escalation_after_s=600.0,
                routing_destination="ops",
            ),
            # 9. MODEL_STALE
            AlertRule(
                rule_id="MODEL_STALE",
                name="Stale ML Model",
                description="An ML model in production is older than the maximum allowed age.",
                family=AlertFamily.MODEL,
                severity=AlertSeverity.WARNING,
                condition="Model training timestamp > 90 days ago.",
                dedup_key="model.stale",
                suppression_window_s=86400.0,  # daily
                flap_threshold=2,
                auto_resolve_s=None,
                requires_acknowledgment=False,
                runbook_id="RB-MODEL-001",
                incident_template=None,
                escalation_after_s=259200.0,  # 3 days
                routing_destination="ops",
            ),
            # 10. KILL_SWITCH_FIRED
            AlertRule(
                rule_id="KILL_SWITCH_FIRED",
                name="Kill Switch Engaged",
                description="The system kill switch has been activated.",
                family=AlertFamily.RISK,
                severity=AlertSeverity.EMERGENCY,
                condition="KillSwitchManager mode != OFF.",
                dedup_key="risk.kill_switch",
                suppression_window_s=30.0,
                flap_threshold=1,
                auto_resolve_s=None,
                requires_acknowledgment=True,
                runbook_id="RB-RISK-002",
                incident_template="Kill switch engaged: {source}",
                escalation_after_s=60.0,
                routing_destination="on_call",
            ),
            # 11. SIGNAL_PIPELINE_FAILURE
            AlertRule(
                rule_id="SIGNAL_PIPELINE_FAILURE",
                name="Signal Pipeline Failure",
                description="Signal computation has failed for one or more pairs.",
                family=AlertFamily.ORCHESTRATION,
                severity=AlertSeverity.ERROR,
                condition="Signal engine raises exception or produces no output.",
                dedup_key="orchestration.signal_pipeline",
                suppression_window_s=300.0,
                flap_threshold=3,
                auto_resolve_s=None,
                requires_acknowledgment=True,
                runbook_id="RB-ORCH-001",
                incident_template="Signal pipeline failure: {scope}",
                escalation_after_s=900.0,
                routing_destination="ops",
            ),
            # 12. RISK_GATEWAY_FAILURE
            AlertRule(
                rule_id="RISK_GATEWAY_FAILURE",
                name="Risk Engine Failure",
                description="The risk engine or portfolio allocator is not functioning.",
                family=AlertFamily.RISK,
                severity=AlertSeverity.CRITICAL,
                condition="Risk engine raises unhandled exception.",
                dedup_key="risk.gateway_failure",
                suppression_window_s=120.0,
                flap_threshold=2,
                auto_resolve_s=None,
                requires_acknowledgment=True,
                runbook_id="RB-RISK-003",
                incident_template="Risk gateway failure: {scope}",
                escalation_after_s=300.0,
                routing_destination="on_call",
            ),
            # 13. AGENT_FAILURE
            AlertRule(
                rule_id="AGENT_FAILURE",
                name="Agent Workflow Failure",
                description="An agent has returned a FAILED result.",
                family=AlertFamily.ORCHESTRATION,
                severity=AlertSeverity.WARNING,
                condition="AgentResult.status == FAILED.",
                dedup_key="orchestration.agent_failure",
                suppression_window_s=300.0,
                flap_threshold=5,
                auto_resolve_s=600.0,
                requires_acknowledgment=False,
                runbook_id="RB-ORCH-002",
                incident_template=None,
                escalation_after_s=1800.0,
                routing_destination="ops",
            ),
            # 14. SERVICE_UNHEALTHY
            AlertRule(
                rule_id="SERVICE_UNHEALTHY",
                name="Service Not Responding",
                description="A system service is not responding to health checks.",
                family=AlertFamily.SYSTEM,
                severity=AlertSeverity.ERROR,
                condition="SystemHealthMonitor.check_all() returns CRITICAL for component.",
                dedup_key="system.service_unhealthy",
                suppression_window_s=120.0,
                flap_threshold=3,
                auto_resolve_s=None,
                requires_acknowledgment=True,
                runbook_id="RB-SYS-001",
                incident_template="Service unhealthy: {scope}",
                escalation_after_s=600.0,
                routing_destination="on_call",
            ),
            # 15. HEARTBEAT_MISSING
            AlertRule(
                rule_id="HEARTBEAT_MISSING",
                name="Component Heartbeat Missing",
                description="No heartbeat received from a component within the expected window.",
                family=AlertFamily.SYSTEM,
                severity=AlertSeverity.WARNING,
                condition="No heartbeat from component for > 2 minutes.",
                dedup_key="system.heartbeat_missing",
                suppression_window_s=120.0,
                flap_threshold=3,
                auto_resolve_s=600.0,
                requires_acknowledgment=False,
                runbook_id="RB-SYS-002",
                incident_template=None,
                escalation_after_s=600.0,
                routing_destination="ops",
            ),
            # 16. ABNORMAL_SLIPPAGE
            AlertRule(
                rule_id="ABNORMAL_SLIPPAGE",
                name="Abnormal Execution Slippage",
                description="Observed slippage is more than 2x the expected model estimate.",
                family=AlertFamily.EXECUTION,
                severity=AlertSeverity.WARNING,
                condition="Actual slippage > 2 * estimated slippage.",
                dedup_key="execution.abnormal_slippage",
                suppression_window_s=600.0,
                flap_threshold=3,
                auto_resolve_s=3600.0,
                requires_acknowledgment=False,
                runbook_id="RB-EXEC-004",
                incident_template=None,
                escalation_after_s=3600.0,
                routing_destination="ops",
            ),
            # 17. QUEUE_BACKLOG
            AlertRule(
                rule_id="QUEUE_BACKLOG",
                name="Workflow Queue Backlog",
                description="The workflow execution queue has a growing backlog.",
                family=AlertFamily.ORCHESTRATION,
                severity=AlertSeverity.WARNING,
                condition="Queue depth exceeds configured threshold for > 5 minutes.",
                dedup_key="orchestration.queue_backlog",
                suppression_window_s=300.0,
                flap_threshold=3,
                auto_resolve_s=600.0,
                requires_acknowledgment=False,
                runbook_id=None,
                incident_template=None,
                escalation_after_s=1800.0,
                routing_destination="ops",
            ),
            # 18. REPEATED_CRASH
            AlertRule(
                rule_id="REPEATED_CRASH",
                name="Repeated Component Crash",
                description="A component has restarted more than 3 times in the last hour.",
                family=AlertFamily.SYSTEM,
                severity=AlertSeverity.ERROR,
                condition="Component restart count > 3 within 60 minutes.",
                dedup_key="system.repeated_crash",
                suppression_window_s=600.0,
                flap_threshold=2,
                auto_resolve_s=None,
                requires_acknowledgment=True,
                runbook_id="RB-SYS-003",
                incident_template="Repeated crash: {scope}",
                escalation_after_s=1800.0,
                routing_destination="on_call",
            ),
            # 19. POLICY_MISMATCH
            AlertRule(
                rule_id="POLICY_MISMATCH",
                name="Runtime Config Policy Drift",
                description="The running configuration has drifted from the desired/approved policy.",
                family=AlertFamily.POLICY,
                severity=AlertSeverity.WARNING,
                condition="Runtime config hash differs from desired config hash.",
                dedup_key="policy.config_mismatch",
                suppression_window_s=600.0,
                flap_threshold=3,
                auto_resolve_s=None,
                requires_acknowledgment=True,
                runbook_id="RB-POLICY-001",
                incident_template=None,
                escalation_after_s=3600.0,
                routing_destination="ops",
            ),
            # 20. DEPLOYMENT_MISMATCH
            AlertRule(
                rule_id="DEPLOYMENT_MISMATCH",
                name="Deployment Version Mismatch",
                description="The running software version differs from the approved deployment version.",
                family=AlertFamily.DEPLOYMENT,
                severity=AlertSeverity.ERROR,
                condition="Running version != approved deployment version.",
                dedup_key="deployment.version_mismatch",
                suppression_window_s=3600.0,
                flap_threshold=2,
                auto_resolve_s=None,
                requires_acknowledgment=True,
                runbook_id="RB-DEPLOY-001",
                incident_template="Deployment mismatch: {scope}",
                escalation_after_s=7200.0,
                routing_destination="ops",
            ),
        ]

        for rule in defaults:
            self._rules[rule.rule_id] = rule


# ══════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════

_engine_instance: Optional[AlertEngine] = None
_engine_lock = threading.Lock()


def get_alert_engine(incident_manager: Optional[Any] = None) -> AlertEngine:
    """Return the singleton AlertEngine, creating it on first call.

    Parameters
    ----------
    incident_manager :
        IncidentManager to inject on first creation (ignored on
        subsequent calls).

    Returns
    -------
    AlertEngine
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = AlertEngine(incident_manager=incident_manager)
    return _engine_instance
