# -*- coding: utf-8 -*-
"""
alerts — Alert management package.

Public re-exports for convenient top-level imports.
"""

from alerts.contracts import (
    AlertAcknowledgement,
    AlertEvent,
    AlertFamily,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    EscalationRecord,
)
from alerts.engine import AlertEngine, get_alert_engine

__all__ = [
    # Enums
    "AlertFamily",
    "AlertSeverity",
    "AlertStatus",
    # Value objects
    "AlertAcknowledgement",
    "AlertEvent",
    "AlertRule",
    "EscalationRecord",
    # Engine
    "AlertEngine",
    "get_alert_engine",
]
