# -*- coding: utf-8 -*-
"""
root/dashboard_agent_context.py — Agent/Consumer context
==========================================================

Extracted from dashboard.py Part 19/35.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# Import deps from dashboard at call time to avoid circular imports
DashboardRuntime = Any
FeatureFlags = Dict[str, Any]
AppContext = Any
TabKey = str
EnvName = str
ProfileName = str
NavPayload = Optional[Dict[str, Any]]

# Part 19/35 – Agent / External-Consumer context (HF-grade snapshot for AI agents & Desktop)
# =====================

SESSION_KEY_AGENT_CONTEXT: str = "dashboard_agent_context"


def _collect_session_nav_history_tail(limit: int = 50) -> List[Dict[str, Any]]:
    """
    מחלץ את זנב הניווט (Navigation History) מתוך session_state, עד `limit` רשומות.

    מבנה כל רשומה:
        {
            "ts":  ISO-UTC,
            "from": "<tab_key>",
            "to":   "<tab_key>",
            "payload_keys": ["pair", "preset", ...] or None,
        }

    מיועד לשימוש ע"י:
    - Agents (ללמוד את התנהגות המשתמש).
    - Desktop / Telemetry לניתוח שימושיות הדשבורד.
    """
    try:
        history = st.session_state.get(SESSION_KEY_NAV_HISTORY, [])
    except Exception:
        history = []

    if not isinstance(history, list) or not history:
        return []

    tail = history[-limit:]
    sanitized: List[Dict[str, Any]] = []
    for item in tail:
        if not isinstance(item, Mapping):
            continue
        sanitized.append(
            {
                "ts": str(item.get("ts")),
                "from": str(item.get("from")),
                "to": str(item.get("to")),
                "payload_keys": (
                    [str(k) for k in item.get("payload_keys", [])]
                    if isinstance(item.get("payload_keys"), Sequence)
                    else None
                ),
            }
        )
    return sanitized


def _collect_session_tab_errors() -> Dict[TabKey, Dict[str, Any]]:
    """
    מחלץ מידע על השגיאות האחרונות בכל טאב (אם קיימות) מתוך session_state.

    הפורמט מותאם ל-AI Agents / Desktop:
        {
            "backtest": {
                "ts_utc": "...",
                "exc_type": "...",
                "message": "...",
            },
            ...
        }
    """
    try:
        errors_map = st.session_state.get(SESSION_KEY_LAST_TAB_ERRORS, {})
    except Exception:
        errors_map = {}

    if not isinstance(errors_map, Mapping):
        return {}

    result: Dict[TabKey, Dict[str, Any]] = {}
    for key, info in errors_map.items():
        if not isinstance(info, TabErrorInfo):
            # נתמוך גם במצב שבו שמרו dict דומה
            if isinstance(info, Mapping):
                result[str(key)] = {
                    "ts_utc": str(info.get("ts_utc")),
                    "exc_type": str(info.get("exc_type")),
                    "message": str(info.get("message")),
                }
            continue

        result[str(info.tab_key)] = {
            "ts_utc": info.ts_utc,
            "exc_type": info.exc_type,
            "message": info.message,
        }

    return result


def _collect_session_tab_timings() -> Dict[TabKey, Dict[str, Any]]:
    """
    מחלץ מדדי ביצועים per-tab מתוך SESSION_KEY_TAB_TIMINGS:

        {
            "home": {"last": 0.123, "avg": 0.089, "count": 17},
            ...
        }

    מיועד ל:
    - Agents שמחליטים איפה "כדאי להשקיע" באופטימיזציה.
    - Desktop / Telemetry לצורך Performance Dashboard.
    """
    try:
        timings = st.session_state.get(SESSION_KEY_TAB_TIMINGS, {}) or {}
    except Exception:
        timings = {}

    if not isinstance(timings, Mapping):
        return {}

    out: Dict[TabKey, Dict[str, Any]] = {}
    for key, rec in timings.items():
        if not isinstance(rec, Mapping):
            continue
        try:
            last = float(rec.get("last", 0.0))
        except Exception:
            last = 0.0
        try:
            avg = float(rec.get("avg", 0.0))
        except Exception:
            avg = 0.0
        try:
            count = int(rec.get("count", 0))
        except Exception:
            count = 0

        out[str(key)] = {"last": last, "avg": avg, "count": count}

    return out


def build_agent_context_payload(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    בונה payload עשיר ו"Agent-ready" שמרכז את כל מה שסוכן AI/מערכת חיצונית צריכה לדעת
    על מצב הדשבורד והמערכת, כדי להציע שדרוגים/ניתוחים/פעולות:

    מבנה כללי:
    -----------
    {
        "meta": {
            "env": ...,
            "profile": ...,
            "run_id": ...,
            "app_name": ...,
            "version": ...,
            "host": ...,
            "user": ...,
        },
        "feature_flags": { ... subset ... },
        "capabilities": { ... },
        "domains": { ... },
        "services_status": { ... },
        "base_context": { ... },     # context לוגי (serialized)
        "nav_history_tail": [...],   # זנב ניווט (עד 50 רשומות)
        "tab_timings": { ... },      # ביצועים per-tab
        "tab_errors": { ... },       # שגיאות אחרונות per-tab
    }

    שימושים:
    --------
    - Agents Tab (🤖) שמנתח את ה-Context ומציע פעולות / refactors.
    - Desktop Bridge / "Supervisor" Agent שמנטר את ביצועי הדשבורד.
    - לוגים מתקדמים (למשל שמירה ל-SqlStore דרך Agents).
    """
    ff = runtime.feature_flags
    env = runtime.env
    profile = runtime.profile
    run_id = runtime.run_id

    app_name = ff.get("app_name", APP_NAME)
    version = ff.get("version", APP_VERSION)
    host = ff.get("host", RUNTIME_HOST)
    user = ff.get("user", RUNTIME_USER)

    base_ctx = st.session_state.get(SESSION_KEY_BASE_CTX)
    base_ctx_serialized = _make_json_safe(
        _serialize_base_ctx_for_snapshot(base_ctx)
    )

    # Nav history / errors / timings מתוך session_state
    nav_history_tail = _collect_session_nav_history_tail(limit=50)
    tab_errors = _collect_session_tab_errors()
    tab_timings = _collect_session_tab_timings()

    # נבחר subset "מעניין" של FeatureFlags – סוכנים לא חייבים הכל
    feature_flags_view = {
        "env": env,
        "profile": profile,
        "enable_live_trading_actions": ff.get("enable_live_trading_actions"),
        "use_sql_backed_state": ff.get("use_sql_backed_state"),
        "enable_experiment_mode": ff.get("enable_experiment_mode"),
        "desktop_integration": ff.get("desktop_integration"),
    }

    payload: Dict[str, Any] = {
        "meta": {
            "env": env,
            "profile": profile,
            "run_id": run_id,
            "app_name": app_name,
            "version": version,
            "host": host,
            "user": user,
        },
        "feature_flags": feature_flags_view,
        "capabilities": dict(runtime.capabilities),
        "domains": dict(runtime.domains),
        "services_status": runtime.services_status,
        "base_context": base_ctx_serialized,
        "nav_history_tail": nav_history_tail,
        "tab_timings": tab_timings,
        "tab_errors": tab_errors,
    }

    return payload


def update_agent_context_in_session(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    בונה ומעדכן את Agent Context ב-session_state, לשימוש ע"י:

    - טאב Agents (🤖) שיכול לקרוא st.session_state["dashboard_agent_context"]
      ולהזין אותו ישירות לסוכני ה-AI.
    - Desktop / תהליכי רקע שיקראו את המידע כדי להפעיל המלצות/שדרוגים.

    הפונקציה מחזירה את ה-payload שנשמר בפועל, לטובת שימוש מיידי.
    """
    payload = build_agent_context_payload(runtime)
    safe_payload = _make_json_safe(payload)

    try:
        st.session_state[SESSION_KEY_AGENT_CONTEXT] = safe_payload
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to store dashboard_agent_context in session_state: %s", exc
        )

    logger.debug(
        "Agent context updated in session_state (env=%s, profile=%s, keys=%s)",
        runtime.env,
        runtime.profile,
        list(safe_payload.keys()),
    )

    return safe_payload


def export_dashboard_state_for_agents(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    מעטפת ידידותית לייצוא מצב הדשבורד לסוכנים/מערכות חיצוניות:

    הבדל מול update_agent_context_in_session:
    -----------------------------------------
    - פונקציה זו אינה נוגעת ב-session_state.
    - מיועדת לשימוש ב:
        * Desktop Bridge שמקבל DashboardRuntime ומעביר את ה-payload
          לתהליכי AI או לשרת חיצוני.
        * בדיקות יחידה / סקריפטים חיצוניים.

    מחזירה:
        dict Agent-ready (כמו build_agent_context_payload) אך ללא כתיבה ל-session_state.
    """
    payload = build_agent_context_payload(runtime)
    return _make_json_safe(payload)


# עדכון __all__ עבור חלק 19
try:
    __all__ += [
        "SESSION_KEY_AGENT_CONTEXT",
        "build_agent_context_payload",
        "update_agent_context_in_session",
        "export_dashboard_state_for_agents",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_AGENT_CONTEXT",
        "build_agent_context_payload",
        "update_agent_context_in_session",
        "export_dashboard_state_for_agents",
    ]
# =====================
