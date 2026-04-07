# -*- coding: utf-8 -*-
"""
core/operational_event_bus.py — Unified Operational Event Bus
=============================================================

Single integration point for:
- Drift detection → incidents
- Governance violations → incidents
- Agent decisions → approval queue
- Incidents → dashboard status bar
- Approvals → resolved decisions

Without this bus, each system domain (ML, agents, approvals, incidents)
is an island — no domain knows about events in another. This bus provides
the bridges between domains without creating hard circular dependencies.

Design:
- Publish/subscribe pattern with typed event objects
- No shared mutable state — events are immutable dicts
- Each subscriber receives events asynchronously (best-effort)
- Never raises — failed event delivery is logged and dropped
- Zero external dependencies (stdlib only)

Usage
-----
    from core.operational_event_bus import get_event_bus, OperationalEvent

    bus = get_event_bus()
    bus.publish(OperationalEvent(
        event_type="drift_critical",
        source="ml_monitoring",
        payload={"model_id": "abc123", "psi": 0.87},
    ))
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("core.operational_event_bus")


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

# Canonical event type strings
# All publishers MUST use these constants — no magic strings
EVENT_DRIFT_CRITICAL         = "drift_critical"          # ML drift → model auto-blocked
EVENT_DRIFT_SEVERE           = "drift_severe"            # ML drift → monitoring increased
EVENT_MODEL_BLOCKED          = "model_blocked"           # Model status set to BLOCKED
EVENT_MODEL_RETIRED          = "model_retired"           # Model status set to RETIRED
EVENT_MODEL_PROMOTED         = "model_promoted"          # Model promoted to CHAMPION
EVENT_GOVERNANCE_VIOLATION   = "governance_violation"    # Contract or policy violation
EVENT_AGENT_DECISION         = "agent_decision"          # Agent produced AgentDecision
EVENT_APPROVAL_RESOLVED      = "approval_resolved"       # Human approved/rejected a request
EVENT_INCIDENT_OPENED        = "incident_opened"         # Incident manager opened incident
EVENT_INCIDENT_CLOSED        = "incident_closed"         # Incident manager closed incident
EVENT_PIPELINE_COMPLETED     = "pipeline_completed"      # Daily pipeline completed
EVENT_PIPELINE_FAILED        = "pipeline_failed"         # Daily pipeline failed
EVENT_HEALTH_CHECK           = "health_check"            # System health check result
EVENT_POSITION_DISCREPANCY   = "position_discrepancy"    # Reconciliation found discrepancy
EVENT_NAKED_POSITION         = "naked_position"          # Naked position detected
EVENT_NAKED_POSITION_RESCUED = "naked_position_rescued"  # Naked position closed


# ---------------------------------------------------------------------------
# Event object
# ---------------------------------------------------------------------------

@dataclass
class OperationalEvent:
    """
    An immutable operational event published to the bus.

    Attributes
    ----------
    event_type : str
        One of the EVENT_* constants above.
    source : str
        Component that generated the event (e.g. "ml_monitoring", "agent_dispatcher").
    payload : dict
        Event-specific data. Must be JSON-serializable.
    severity : str
        "info" | "warning" | "error" | "critical"
    event_id : str
        Auto-generated unique ID.
    published_at : str
        UTC ISO-8601 timestamp.
    """
    event_type: str
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"
    event_id: str = field(
        default_factory=lambda: __import__("uuid").uuid4().hex[:12]
    )
    published_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

class OperationalEventBus:
    """
    Thread-safe publish/subscribe event bus for operational events.

    Subscribers register a handler function for one or more event types.
    When an event is published, all matching handlers are called synchronously
    (in registration order). Failed handlers are logged and skipped.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._handlers: Dict[str, List[Callable]] = {}
        self._wildcard_handlers: List[Callable] = []
        self._event_history: List[OperationalEvent] = []
        self._max_history = 200

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[OperationalEvent], None],
    ) -> None:
        """
        Register a handler for a specific event type.

        Parameters
        ----------
        event_type : str
            One of the EVENT_* constants, or "*" for all events.
        handler : callable
            Function called with the OperationalEvent when published.
            Must not raise — exceptions are caught and logged.
        """
        with self._lock:
            if event_type == "*":
                self._wildcard_handlers.append(handler)
            else:
                if event_type not in self._handlers:
                    self._handlers[event_type] = []
                self._handlers[event_type].append(handler)

    def publish(self, event: OperationalEvent) -> int:
        """
        Publish an event to all registered subscribers.

        Parameters
        ----------
        event : OperationalEvent

        Returns
        -------
        int : Number of handlers called.
        """
        with self._lock:
            handlers = list(self._handlers.get(event.event_type, []))
            wildcards = list(self._wildcard_handlers)
            # Record to history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

        called = 0
        for handler in handlers + wildcards:
            try:
                handler(event)
                called += 1
            except Exception as exc:
                logger.warning(
                    "OperationalEventBus: handler %s failed for event %s: %s",
                    getattr(handler, "__name__", repr(handler)),
                    event.event_type,
                    exc,
                )

        return called

    def get_recent(
        self,
        event_type: Optional[str] = None,
        n: int = 20,
    ) -> List[OperationalEvent]:
        """Return recent events, optionally filtered by type."""
        with self._lock:
            events = list(self._event_history)
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-n:]

    def get_critical_events(self, n: int = 10) -> List[OperationalEvent]:
        """Return recent critical-severity events."""
        with self._lock:
            events = list(self._event_history)
        critical = [e for e in events if e.severity == "critical"]
        return critical[-n:]


# ---------------------------------------------------------------------------
# Built-in handlers: connect domains
# ---------------------------------------------------------------------------

def _drift_to_incident_handler(event: OperationalEvent) -> None:
    """When critical drift is detected, open an incident."""
    if event.event_type != EVENT_DRIFT_CRITICAL:
        return
    try:
        from incidents.manager import IncidentManager
        im = IncidentManager()
        if hasattr(im, "open_incident"):
            im.open_incident(
                title=f"CRITICAL drift: model {event.payload.get('model_id', '?')} auto-blocked",
                severity="critical",
                source=event.source,
                details=event.payload,
            )
    except Exception as exc:
        logger.debug("_drift_to_incident_handler: %s", exc)


def _governance_violation_to_incident_handler(event: OperationalEvent) -> None:
    """When a governance violation occurs, open an incident."""
    if event.event_type != EVENT_GOVERNANCE_VIOLATION:
        return
    try:
        from incidents.manager import IncidentManager
        im = IncidentManager()
        if hasattr(im, "open_incident"):
            im.open_incident(
                title=f"Governance violation: {event.payload.get('violation_type', '?')}",
                severity="error",
                source=event.source,
                details=event.payload,
            )
    except Exception as exc:
        logger.debug("_governance_violation_to_incident_handler: %s", exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_bus_instance: Optional[OperationalEventBus] = None
_bus_lock = threading.Lock()


def get_event_bus() -> OperationalEventBus:
    """Return the process-wide singleton OperationalEventBus."""
    global _bus_instance
    with _bus_lock:
        if _bus_instance is None:
            _bus_instance = OperationalEventBus()
            # Register built-in domain bridges
            _bus_instance.subscribe(EVENT_DRIFT_CRITICAL, _drift_to_incident_handler)
            _bus_instance.subscribe(
                EVENT_GOVERNANCE_VIOLATION, _governance_violation_to_incident_handler
            )
            logger.debug("OperationalEventBus: initialized with built-in domain bridges")
    return _bus_instance
