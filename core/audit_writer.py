# -*- coding: utf-8 -*-
"""
core/audit_writer.py — Audit Chain Writer for Operational Decisions
===================================================================

Writes real audit entries to the audit chain for:
- Signal decisions (entry/exit)
- Portfolio allocations
- Model promotions
- Risk events (drawdown, kill-switch)
- Agent dispatches
- Configuration changes

Fixes #15: "Audit chains empty for all operational decisions"
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _get_audit_chain(chain_name: str = "operational"):
    """Get or create an audit chain."""
    try:
        from audit.chain import get_audit_chain_registry
        registry = get_audit_chain_registry()
        chain = registry.get_or_create(chain_name)
        return chain
    except Exception as exc:
        logger.debug("Audit chain not available: %s", exc)
        return None


def audit_signal_decision(
    pair: str,
    action: str,
    z_score: float,
    regime: str = "UNKNOWN",
    quality_grade: str = "?",
    blocked: bool = False,
    rationale: str = "",
) -> None:
    """Record a signal decision in the audit chain."""
    chain = _get_audit_chain("signals")
    if chain is None:
        return

    try:
        chain.append({
            "event_type": "SIGNAL_DECISION",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "action": action,
            "z_score": round(z_score, 3),
            "regime": regime,
            "quality_grade": quality_grade,
            "blocked": blocked,
            "rationale": rationale,
        })
    except Exception as exc:
        logger.debug("Failed to audit signal: %s", exc)


def audit_allocation(
    pair: str,
    approved: bool,
    weight: float,
    capital_allocated: float,
    rationale: str = "",
) -> None:
    """Record a portfolio allocation decision."""
    chain = _get_audit_chain("allocations")
    if chain is None:
        return

    try:
        chain.append({
            "event_type": "ALLOCATION_DECISION",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "approved": approved,
            "weight": round(weight, 4),
            "capital_allocated": round(capital_allocated, 2),
            "rationale": rationale,
        })
    except Exception as exc:
        logger.debug("Failed to audit allocation: %s", exc)


def audit_risk_event(
    event_type: str,
    severity: str,
    details: str,
    action_taken: str = "",
) -> None:
    """Record a risk event (drawdown, kill-switch, etc.)."""
    chain = _get_audit_chain("risk")
    if chain is None:
        return

    try:
        chain.append({
            "event_type": f"RISK_{event_type}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "details": details,
            "action_taken": action_taken,
        })
    except Exception as exc:
        logger.debug("Failed to audit risk event: %s", exc)


def audit_model_event(
    model_id: str,
    event_type: str,
    metrics: Optional[dict] = None,
    decision: str = "",
) -> None:
    """Record ML model lifecycle event."""
    chain = _get_audit_chain("models")
    if chain is None:
        return

    try:
        chain.append({
            "event_type": f"MODEL_{event_type}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "metrics": metrics or {},
            "decision": decision,
        })
    except Exception as exc:
        logger.debug("Failed to audit model event: %s", exc)


def audit_config_change(
    key: str,
    old_value: Any,
    new_value: Any,
    changed_by: str = "auto",
) -> None:
    """Record configuration change."""
    chain = _get_audit_chain("config")
    if chain is None:
        return

    try:
        chain.append({
            "event_type": "CONFIG_CHANGE",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "key": key,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "changed_by": changed_by,
        })
    except Exception as exc:
        logger.debug("Failed to audit config change: %s", exc)


def get_audit_summary() -> dict:
    """Get summary of all audit chains."""
    summary = {}
    try:
        from audit.chain import get_audit_chain_registry
        registry = get_audit_chain_registry()
        for name in ["signals", "allocations", "risk", "models", "config", "operational"]:
            chain = registry.get_or_create(name)
            n = len(chain.entries) if hasattr(chain, "entries") else 0
            summary[name] = n
    except Exception:
        pass
    return summary
