# -*- coding: utf-8 -*-
"""
root/dashboard_api_bundle.py — Unified Dashboard API bundle
=============================================================

Extracted from dashboard.py Part 25/35.
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

# Part 25/35 – Unified Dashboard API bundle (HF-grade external interface)
# =====================

SESSION_KEY_API_BUNDLE_LAST: str = "dashboard_api_bundle_last"


@dataclass
class DashboardAPIBundle:
    """
    חבילה אחודה ("API Bundle") שמרכזת את כל מה שצרכן חיצוני צריך לדעת:

    שימושים טיפוסיים:
    ------------------
    - Desktop Bridge (Qt) שרוצה לקבל:
        * Health (ready / issues / warnings)
        * Summary (services, tabs, user/meta)
        * Agent context (nav history, timings, errors, base_ctx)
    - AI Supervisors / Orchestrators שצריכים תצוגה אחת של מצב הדשבורד.
    - REST API / gRPC שיחשפו "מצב דשבורד" החוצה.

    שדות:
    -----
    meta:
        מטא-דאטה בסיסי (env/profile/run_id/app_name/version/host/user/ts_utc).
    health:
        dict של DashboardHealth (ready/issues/warnings) – JSON-friendly.
    summary:
        dict של DashboardSummary – JSON-friendly.
    agent_context:
        dict של Agent Context (כמו build_agent_context_payload).
    """

    meta: Dict[str, Any]
    health: Dict[str, Any]
    summary: Dict[str, Any]
    agent_context: Dict[str, Any]


def _build_api_meta(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    בונה meta קטן עבור ה-API Bundle:

        {
            "env": ...,
            "profile": ...,
            "run_id": ...,
            "app_name": ...,
            "version": ...,
            "host": ...,
            "user": ...,
            "ts_utc": "...",
        }
    """
    ff = runtime.feature_flags
    return {
        "env": runtime.env,
        "profile": runtime.profile,
        "run_id": runtime.run_id,
        "app_name": ff.get("app_name", APP_NAME),
        "version": ff.get("version", APP_VERSION),
        "host": ff.get("host", RUNTIME_HOST),
        "user": ff.get("user", RUNTIME_USER),
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_dashboard_api_bundle(
    runtime: DashboardRuntime,
    include_summary: bool = True,
    include_health_summary: bool = True,
    include_agent_context: bool = True,
) -> DashboardAPIBundle:
    """
    בונה DashboardAPIBundle מתוך Runtime:

    Flow:
    -----
    1. meta  ← _build_api_meta(runtime)
    2. health  ← export_dashboard_health(runtime, include_summary=include_health_summary)
    3. summary ← export_dashboard_summary(runtime) if include_summary else {}
    4. agent_context ← export_dashboard_state_for_agents(runtime) if include_agent_context else {}

    זהו ה-"שכבת API" המרכזית – צרכן חיצוני יכול לעבוד רק עם האובייקט הזה
    ולא להצטרך לדעת על כל הפונקציות הפנימיות.
    """
    meta = _build_api_meta(runtime)
    health = export_dashboard_health(runtime, include_summary=include_health_summary)

    if include_summary:
        summary = export_dashboard_summary(runtime)
    else:
        summary = {}

    if include_agent_context:
        agent_ctx = export_dashboard_state_for_agents(runtime)
    else:
        agent_ctx = {}

    bundle = DashboardAPIBundle(
        meta=meta,
        health=health,
        summary=summary,
        agent_context=agent_ctx,
    )

    return bundle


def dashboard_api_bundle_to_dict(bundle: DashboardAPIBundle) -> Dict[str, Any]:
    """
    ממיר DashboardAPIBundle ל-dict JSON-friendly.

    מאפשר:
    - שמירה ב-SqlStore/Log.
    - שליחה כסטרוקטורה אחת לסוכני AI / Desktop / REST.
    """
    return {
        "meta": bundle.meta,
        "health": bundle.health,
        "summary": bundle.summary,
        "agent_context": bundle.agent_context,
    }


def update_dashboard_api_bundle_in_session(
    runtime: DashboardRuntime,
    include_summary: bool = True,
    include_health_summary: bool = True,
    include_agent_context: bool = True,
) -> DashboardAPIBundle:
    """
    בונה API Bundle ומעדכן אותו ב-session_state[SESSION_KEY_API_BUNDLE_LAST].

    - שומר dict JSON-friendly (לא את ה- dataclass עצמו).
    - מחזיר את ה-DashboardAPIBundle עבור שימוש מיידי.
    """
    bundle = build_dashboard_api_bundle(
        runtime,
        include_summary=include_summary,
        include_health_summary=include_health_summary,
        include_agent_context=include_agent_context,
    )
    obj = dashboard_api_bundle_to_dict(bundle)
    obj = _make_json_safe(obj)

    try:
        st.session_state[SESSION_KEY_API_BUNDLE_LAST] = obj
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to store dashboard_api_bundle_last in session_state: %s", exc
        )

    logger.debug(
        "Dashboard API bundle updated in session_state (env=%s, profile=%s, keys=%s)",
        bundle.meta.get("env"),
        bundle.meta.get("profile"),
        list(obj.keys()),
    )

    return bundle


def export_dashboard_api_bundle(
    app_ctx: Optional["AppContext"] = None,
    include_summary: bool = True,
    include_health_summary: bool = True,
    include_agent_context: bool = True,
) -> Dict[str, Any]:
    """
    Export "Headless" נוח:

    - בונה Runtime (ensure_dashboard_runtime).
    - בונה DashboardAPIBundle.
    - מחזיר dict JSON-friendly אחד שמכיל:
        * meta
        * health
        * summary
        * agent_context

    מיועד לשימוש:
    -------------
    - בסקריפטים חיצוניים.
    - ב-Desktop Bridge (שמקבל dict ומעבד אותו בצד ה-Qt).
    - ב-REST endpoint שיחשוף `/dashboard/state` או דומה.
    """
    if app_ctx is None:
        app_ctx = get_app_context()

    runtime = ensure_dashboard_runtime(app_ctx)
    bundle = build_dashboard_api_bundle(
        runtime,
        include_summary=include_summary,
        include_health_summary=include_health_summary,
        include_agent_context=include_agent_context,
    )
    payload = dashboard_api_bundle_to_dict(bundle)
    return _make_json_safe(payload)


# עדכון __all__ עבור חלק 25
try:
    __all__ += [
        "SESSION_KEY_API_BUNDLE_LAST",
        "DashboardAPIBundle",
        "build_dashboard_api_bundle",
        "dashboard_api_bundle_to_dict",
        "update_dashboard_api_bundle_in_session",
        "export_dashboard_api_bundle",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_API_BUNDLE_LAST",
        "DashboardAPIBundle",
        "build_dashboard_api_bundle",
        "dashboard_api_bundle_to_dict",
        "update_dashboard_api_bundle_in_session",
        "export_dashboard_api_bundle",
    ]
# =====================
