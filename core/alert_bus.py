# -*- coding: utf-8 -*-
"""
core/alert_bus.py - Dashboard Alert Bus (Streamlit-decoupled)
==============================================================

Event Bus for dashboard-level alerts and notifications.
Extracted from root/dashboard.py (Part 32/35).

ARCHITECTURE (Phase 2.2 refactor):
- Core logic uses StateProvider protocol (no direct streamlit dependency)
- Default provider resolution picks Streamlit session only when runtime exists
- Otherwise falls back to in-memory state for CLI, tests, and API contexts
- UI render functions moved to root/dashboard_alerts_bus.py

Public API (stable):
- DashboardAlert dataclass
- emit_dashboard_alert() / get_dashboard_alerts() / clear_dashboard_alerts()
- render_dashboard_alert_center() (shim re-exports from root/ if available)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

from core.state_provider import StateProvider, get_default_state_provider

try:  # pragma: no cover
    from common.json_safe import make_json_safe as _make_json_safe  # type: ignore[import]
except Exception:  # pragma: no cover
    def _make_json_safe(obj: Any) -> Any:
        return obj


logger = logging.getLogger(__name__)


SESSION_KEY_ALERTS: str = "dashboard_alerts"

_state_provider: Optional[StateProvider] = None
_alert_center_renderer: Optional[Callable[[int, Optional[Sequence[str]]], None]] = None


def _get_state_provider() -> StateProvider:
    """Return the active state provider without importing Streamlit here."""
    global _state_provider
    if _state_provider is None:
        _state_provider = get_default_state_provider()
    return _state_provider


def set_state_provider(provider: StateProvider) -> None:
    """Inject a specific state provider (for tests or custom deployments)."""
    global _state_provider
    _state_provider = provider


def set_dashboard_alert_renderer(
    renderer: Optional[Callable[[int, Optional[Sequence[str]]], None]],
) -> None:
    """Register a UI renderer from the root layer without importing root/ here."""
    global _alert_center_renderer
    _alert_center_renderer = renderer


@dataclass
class DashboardAlert:
    """
    Dashboard-level alert/notification - internal Event Bus.

    Use cases:
    - Risk Engine: "Exposure limit breached", "Kill-switch armed"
    - Macro Engine: regime shifts, macro events
    - Agents: action taken, rejected, or failed
    """

    id: str
    ts_utc: str
    level: Literal["info", "success", "warning", "error"]
    source: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


def _ensure_alerts_list() -> List[Dict[str, Any]]:
    """Ensure alerts list is present in the active state provider."""
    provider = _get_state_provider()
    obj = provider.get(SESSION_KEY_ALERTS, [])
    if not isinstance(obj, list):
        obj = []
        provider.set(SESSION_KEY_ALERTS, obj)
    return obj  # type: ignore[return-value]


def _alert_to_dict(alert: DashboardAlert) -> Dict[str, Any]:
    return {
        "id": alert.id,
        "ts_utc": alert.ts_utc,
        "level": alert.level,
        "source": alert.source,
        "message": alert.message,
        "details": alert.details,
    }


def _alert_from_mapping(data: Mapping[str, Any]) -> Optional[DashboardAlert]:
    """Deserialize DashboardAlert from mapping. Returns None on invalid input."""
    try:
        return DashboardAlert(
            id=str(data.get("id", uuid.uuid4().hex)),
            ts_utc=str(data.get("ts_utc", "")),
            level=data.get("level", "info"),  # type: ignore[arg-type]
            source=str(data.get("source", "unknown")),
            message=str(data.get("message", "")),
            details=dict(data.get("details", {})),
        )
    except Exception as exc:
        logger.debug("_alert_from_mapping failed: %s", exc)
        return None


def emit_dashboard_alert(
    level: Literal["info", "success", "warning", "error"],
    source: str,
    message: str,
    details: Optional[Mapping[str, Any]] = None,
) -> DashboardAlert:
    """Emit a new dashboard alert and persist it via the active state provider."""
    details_dict: Dict[str, Any] = dict(details) if isinstance(details, Mapping) else {}
    alert = DashboardAlert(
        id=uuid.uuid4().hex,
        ts_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        level=level,
        source=source,
        message=message,
        details=_make_json_safe(details_dict),
    )

    alerts_list = _ensure_alerts_list()
    alerts_list.append(_alert_to_dict(alert))
    _get_state_provider().set(SESSION_KEY_ALERTS, alerts_list)

    logger.info(
        "Dashboard alert emitted: level=%s, source=%s, message=%s",
        alert.level,
        alert.source,
        alert.message,
    )
    return alert


def get_dashboard_alerts(limit: Optional[int] = None) -> List[DashboardAlert]:
    """Return alerts (newest first). If limit provided, only return top N."""
    alerts_raw = _ensure_alerts_list()
    alerts: List[DashboardAlert] = []
    for item in alerts_raw:
        if isinstance(item, Mapping):
            alert = _alert_from_mapping(item)
            if alert is not None:
                alerts.append(alert)

    alerts.sort(key=lambda alert: alert.ts_utc, reverse=True)
    if limit is not None and limit > 0:
        alerts = alerts[:limit]
    return alerts


def clear_dashboard_alerts(
    level: Optional[str] = None,
    source: Optional[str] = None,
) -> None:
    """
    Clear alerts from the bus.

    - If level=None and source=None -> clear all
    - If level != None -> clear only that level
    - If source != None -> clear only from that source
    - Can be combined (level="info", source="system")
    """
    alerts_raw = _ensure_alerts_list()
    if not alerts_raw:
        return

    filtered: List[Dict[str, Any]] = []
    for item in alerts_raw:
        if not isinstance(item, Mapping):
            continue
        if level is not None and str(item.get("level") or "").lower() == level.lower():
            continue
        if source is not None and str(item.get("source") or "") == source:
            continue
        filtered.append(item)

    _get_state_provider().set(SESSION_KEY_ALERTS, filtered)


def render_dashboard_alert_center(
    max_items: int = 10,
    filter_levels: Optional[Sequence[str]] = None,
) -> None:
    """
    Render alert center (shim - delegates to root/dashboard_alerts_bus).

    Kept for backward compatibility. The actual Streamlit rendering lives
    in root/dashboard_alerts_bus.py (Tier 5).

    If called in a non-Streamlit context, this is a no-op.
    """
    renderer = _alert_center_renderer
    if renderer is None:
        logger.debug("render_dashboard_alert_center: UI renderer not registered")
        return

    try:
        renderer(max_items=max_items, filter_levels=filter_levels)
    except Exception as exc:
        logger.debug("render_dashboard_alert_center: UI unavailable: %s", exc)


__all__ = [
    "SESSION_KEY_ALERTS",
    "DashboardAlert",
    "emit_dashboard_alert",
    "get_dashboard_alerts",
    "clear_dashboard_alerts",
    "render_dashboard_alert_center",
    "set_dashboard_alert_renderer",
    "set_state_provider",
]
