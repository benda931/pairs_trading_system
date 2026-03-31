# -*- coding: utf-8 -*-
"""
core/alert_bus.py — Dashboard Alert Bus
========================================

Event Bus for dashboard-level alerts and notifications.
Extracted from root/dashboard.py (Part 32/35).

Provides:
- DashboardAlert dataclass
- emit_dashboard_alert() / get_dashboard_alerts() / clear_dashboard_alerts()
- render_dashboard_alert_center() (Streamlit UI)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
)

from collections.abc import Mapping, Sequence

import streamlit as st

# Fallback for json_safe
try:  # pragma: no cover
    from common.json_safe import make_json_safe as _make_json_safe  # type: ignore[import]
except Exception:  # pragma: no cover
    def _make_json_safe(obj: Any) -> Any:
        return obj

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_KEY_ALERTS: str = "dashboard_alerts"

# ---------------------------------------------------------------------------
# DashboardAlert dataclass
# ---------------------------------------------------------------------------


@dataclass
class DashboardAlert:
    """
    Alert / Notification ברמת דשבורד – Event Bus פנימי לסיכון, מקרו, דאטה וסוכנים.

    שימושים:
    ---------
    - Risk Engine יכול לדווח:
        * "Exposure limit breached" / "Kill-switch armed" וכו'.
    - Macro Engine יכול לדווח:
        * שינוי Regime / אירוע מקרו חשוב.
    - Agents יכולים לדווח:
        * פעולה בוצעה / נדחתה / נכשלה.

    שדות:
    -----
    id:
        מזהה ייחודי (UUID4 hex).
    ts_utc:
        טיימסטמפ ב-UTC (isoformat, seconds).
    level:
        רמת חשיבות: "info" / "success" / "warning" / "error".
    source:
        מקור: למשל "risk_engine", "macro_engine", "agent", "backtest_tab".
    message:
        הודעה קצרה, קריאה לבני אדם.
    details:
        dict אופציונלי עם שדות נוספים (pair, portfolio_id, regime, limit_name וכו').
    """

    id: str
    ts_utc: str
    level: Literal["info", "success", "warning", "error"]
    source: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_alerts_list() -> List[Dict[str, Any]]:
    """
    מבטיח רשימת Alerts ב-session_state:

        [
            { ... DashboardAlert כ-dict JSON-friendly ... },
            ...
        ]
    """
    try:
        obj = st.session_state.get(SESSION_KEY_ALERTS, [])
    except Exception:  # pragma: no cover
        obj = []

    if not isinstance(obj, list):
        obj = []
        try:
            st.session_state[SESSION_KEY_ALERTS] = obj
        except Exception:
            pass
    return obj  # type: ignore[return-value]


def _alert_to_dict(alert: DashboardAlert) -> Dict[str, Any]:
    """
    המרה ל-dict JSON-friendly.
    """
    return {
        "id": alert.id,
        "ts_utc": alert.ts_utc,
        "level": alert.level,
        "source": alert.source,
        "message": alert.message,
        "details": alert.details,
    }


def _alert_from_mapping(data: Mapping[str, Any]) -> Optional[DashboardAlert]:
    """
    המרה ממבנה dict ל-DashboardAlert (Best-effort, לא נכשל על שדות חסרים קטנים).
    """
    try:
        alert_id = str(data.get("id") or uuid.uuid4().hex)
        ts = str(
            data.get("ts_utc")
            or datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
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
            id=alert_id,
            ts_utc=ts,
            level=level,
            source=source,
            message=message,
            details=dict(details),
        )
    except Exception:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def emit_dashboard_alert(
    level: Literal["info", "success", "warning", "error"],
    source: str,
    message: str,
    details: Optional[Mapping[str, Any]] = None,
) -> DashboardAlert:
    """
    מוסיף Alert חדש ל-Bus הדשבורד.

    דוגמאות:
    ---------
        emit_dashboard_alert(
            "warning",
            "risk_engine",
            "Exposure limit exceeded for portfolio 'core_fund'",
            {"portfolio_id": "core_fund", "limit": "gross_exposure", "value": 1.25},
        )

        emit_dashboard_alert(
            "error",
            "broker",
            "IBKR connection lost",
            {"host": "127.0.0.1", "port": 7497},
        )
    """
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

    try:
        st.session_state[SESSION_KEY_ALERTS] = alerts_list
    except Exception:  # pragma: no cover
        pass

    logger.info(
        "Dashboard alert emitted: level=%s, source=%s, message=%s",
        alert.level,
        alert.source,
        alert.message,
    )

    return alert


def get_dashboard_alerts(limit: Optional[int] = None) -> List[DashboardAlert]:
    """
    מחזיר רשימת Alerts (מהחדש לישן). אם limit סופק – מחזיר רק את ה-N האחרונים.
    """
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


def clear_dashboard_alerts(level: Optional[str] = None, source: Optional[str] = None) -> None:
    """
    מנקה Alerts מה-Bus:

    - אם level=None ו-source=None → מנקה הכל.
    - אם level לא None → מנקה Alerts רק ברמת level מסוימת.
    - אם source לא None → מנקה Alerts רק מ-Source מסוים.

    ניתן להשתמש גם בשילוב:
        clear_dashboard_alerts(level="info", source="system")
    """
    alerts_raw = _ensure_alerts_list()
    if not alerts_raw:
        return

    filtered: List[Dict[str, Any]] = []

    for item in alerts_raw:
        if not isinstance(item, Mapping):
            continue
        if level is not None:
            lvl = str(item.get("level") or "").lower()
            if lvl == level.lower():
                continue
        if source is not None:
            src = str(item.get("source") or "")
            if src == source:
                continue
        filtered.append(item)

    try:
        st.session_state[SESSION_KEY_ALERTS] = filtered
    except Exception:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Streamlit UI helpers
# ---------------------------------------------------------------------------


def _render_alert_badge(alert: DashboardAlert) -> None:
    """
    מציג Alert קטן (Badge-style) ב-UI, לשימוש על ידי Tabs/Toolbar בעתיד.
    """
    icon = {
        "info": "\u2139\ufe0f",
        "success": "\u2705",
        "warning": "\u26a0\ufe0f",
        "error": "\U0001f6a8",
    }.get(alert.level, "\u2139\ufe0f")

    st.write(
        f"{icon} `{alert.ts_utc}` \u2022 **{alert.source}** \u2013 {alert.message}"
    )


def render_dashboard_alert_center(
    max_items: int = 10,
    filter_levels: Optional[Sequence[str]] = None,
) -> None:
    """
    רנדר מרכזי של Alerts (Alert Center) – לא מחובר אוטומטית ל-Shell,
    אלא מיועד לשימוש מטאבים כמו:
        - Home
        - Risk
        - Agents
        - Logs

    פרמטרים:
    --------
    max_items:
        מספר ה-Alerts המקסימלי להצגה (ברירת מחדל: 10).
    filter_levels:
        אם לא None – מציג רק Alerts ש-level שלהם נמצא ברשימה (למשל ["warning", "error"]).
    """
    alerts = get_dashboard_alerts(limit=max_items)
    if filter_levels:
        lvl_set = {lvl.lower() for lvl in filter_levels}
        alerts = [a for a in alerts if a.level.lower() in lvl_set]

    if not alerts:
        st.caption("No dashboard alerts at the moment.")
        return

    for alert in alerts:
        _render_alert_badge(alert)


__all__ = [
    "SESSION_KEY_ALERTS",
    "DashboardAlert",
    "emit_dashboard_alert",
    "get_dashboard_alerts",
    "clear_dashboard_alerts",
    "render_dashboard_alert_center",
]
