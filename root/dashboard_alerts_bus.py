# -*- coding: utf-8 -*-
"""
root/dashboard_alerts_bus.py — Dashboard Alert Bus
====================================================

Extracted from dashboard.py Part 32/35.

Provides an in-session alert/notification bus for the dashboard.
Sources (risk engine, macro engine, agents, tabs) emit alerts;
consumers (Home, Risk, Agents, Logs tabs) render them.

Usage:
    from root.dashboard_alerts_bus import emit_dashboard_alert, get_dashboard_alerts

    emit_dashboard_alert("warning", "risk_engine", "Exposure limit exceeded")
    alerts = get_dashboard_alerts(limit=10)
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import streamlit as st

logger = logging.getLogger(__name__)

SESSION_KEY_ALERTS: str = "dashboard_alerts"


def _make_json_safe_simple(obj: Any) -> Any:
    """Minimal JSON-safe conversion for alert details."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe_simple(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe_simple(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


@dataclass
class DashboardAlert:
    """
    Alert/Notification for the dashboard event bus.

    Levels: info, success, warning, error
    Source examples: risk_engine, macro_engine, agent, broker, backtest_tab
    """
    id: str
    ts_utc: str
    level: Literal["info", "success", "warning", "error"]
    source: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


def _ensure_alerts_list() -> List[Dict[str, Any]]:
    """Ensure alerts list exists in session_state."""
    try:
        obj = st.session_state.get(SESSION_KEY_ALERTS, [])
    except Exception:
        obj = []
    if not isinstance(obj, list):
        obj = []
        try:
            st.session_state[SESSION_KEY_ALERTS] = obj
        except Exception:
            pass
    return obj  # type: ignore[return-value]


def _alert_to_dict(alert: DashboardAlert) -> Dict[str, Any]:
    """Convert DashboardAlert to JSON-friendly dict."""
    return {
        "id": alert.id,
        "ts_utc": alert.ts_utc,
        "level": alert.level,
        "source": alert.source,
        "message": alert.message,
        "details": alert.details,
    }


def _alert_from_mapping(data: Mapping[str, Any]) -> Optional[DashboardAlert]:
    """Convert dict to DashboardAlert (best-effort)."""
    try:
        alert_id = str(data.get("id") or uuid.uuid4().hex)
        ts = str(data.get("ts_utc") or datetime.now(timezone.utc).isoformat(timespec="seconds"))
        level_raw = str(data.get("level") or "info").lower()
        level: Literal["info", "success", "warning", "error"]
        if level_raw not in ("info", "success", "warning", "error"):
            level = "info"
        else:
            level = level_raw  # type: ignore[assignment]

        source = str(data.get("source") or "system")
        message = str(data.get("message") or "").strip()
        details = data.get("details") or {}
        if not isinstance(details, Mapping):
            details = {"value": details}

        if not message:
            return None

        return DashboardAlert(
            id=alert_id, ts_utc=ts, level=level, source=source,
            message=message, details=dict(details),
        )
    except Exception:
        return None


def emit_dashboard_alert(
    level: Literal["info", "success", "warning", "error"],
    source: str,
    message: str,
    details: Optional[Mapping[str, Any]] = None,
) -> DashboardAlert:
    """
    Emit a new alert to the dashboard alert bus.

    Examples:
        emit_dashboard_alert("warning", "risk_engine", "Exposure limit exceeded",
                             {"portfolio_id": "core_fund", "limit": "gross_exposure"})
        emit_dashboard_alert("error", "broker", "IBKR connection lost")
    """
    details_dict: Dict[str, Any] = dict(details) if isinstance(details, Mapping) else {}
    alert = DashboardAlert(
        id=uuid.uuid4().hex,
        ts_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        level=level,
        source=source,
        message=message,
        details=_make_json_safe_simple(details_dict),
    )

    alerts_list = _ensure_alerts_list()
    alerts_list.append(_alert_to_dict(alert))

    try:
        st.session_state[SESSION_KEY_ALERTS] = alerts_list
    except Exception:
        pass

    logger.info("Dashboard alert emitted: level=%s, source=%s, message=%s",
                alert.level, alert.source, alert.message)
    return alert


def get_dashboard_alerts(limit: Optional[int] = None) -> List[DashboardAlert]:
    """Return alerts list (newest first). If limit provided, return only N most recent."""
    alerts_raw = _ensure_alerts_list()
    alerts: List[DashboardAlert] = []
    for item in alerts_raw:
        if isinstance(item, Mapping):
            a = _alert_from_mapping(item)
            if a is not None:
                alerts.append(a)

    alerts.sort(key=lambda a: a.ts_utc, reverse=True)
    if limit is not None and limit > 0:
        alerts = alerts[:limit]
    return alerts


def clear_dashboard_alerts(
    level: Optional[str] = None, source: Optional[str] = None,
) -> None:
    """Clear alerts from the bus (optionally filtered by level/source)."""
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

    try:
        st.session_state[SESSION_KEY_ALERTS] = filtered
    except Exception:
        pass


def _render_alert_badge(alert: DashboardAlert) -> None:
    """Render a single alert badge in Streamlit UI."""
    icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "🚨"}.get(alert.level, "ℹ️")
    st.write(f"{icon} `{alert.ts_utc}` • **{alert.source}** – {alert.message}")


def render_dashboard_alert_center(
    max_items: int = 10,
    filter_levels: Optional[Sequence[str]] = None,
) -> None:
    """Render the alert center (used by Home, Risk, Agents, Logs tabs)."""
    alerts = get_dashboard_alerts(limit=max_items)
    if filter_levels:
        lvl_set = {lvl.lower() for lvl in filter_levels}
        alerts = [a for a in alerts if a.level.lower() in lvl_set]

    if not alerts:
        st.caption("No dashboard alerts at the moment.")
        return

    for alert in alerts:
        _render_alert_badge(alert)
