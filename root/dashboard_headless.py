# -*- coding: utf-8 -*-
"""
root/dashboard_headless.py — Headless/Testing helpers
=======================================================

Extracted from dashboard.py Part 31/35.
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

# Part 31/35 – Headless / Testing helpers (HF-grade programmatic interface)
# =====================

def get_minimal_dashboard_state(
    app_ctx: Optional["AppContext"] = None,
    include_health: bool = True,
    include_summary: bool = False,
    include_overview: bool = False,
) -> Dict[str, Any]:
    """
    מחזיר מצב מינימלי (אך אינפורמטיבי) של הדשבורד בצורה Headless – מתאים ל-Tests / CI / סקריפטים.

    Flow:
    -----
    1. מבטיח AppContext (get_app_context אם לא סופק).
    2. מקים/מרענן DashboardRuntime (ensure_dashboard_runtime).
    3. אוסף נתונים בסיסיים:
        - env / profile / run_id
        - app_name / version
        - active_tabs (keys בלבד)
        - capabilities (keys בלבד)
    4. אופציונלי:
        - health (אם include_health=True) – summary קל מתוך DashboardHealth.
        - summary "קל" (אם include_summary=True) – מתוך DashboardSummary.
        - overview "קל" (אם include_overview=True) – primary metrics + מספר כרטיסים.

    הפלט:
    -----
    dict JSON-friendly, לדוגמה:
        {
            "env": "dev",
            "profile": "trading",
            "run_id": "...",
            "app_name": "...",
            "version": "...",
            "active_tabs": ["home", "smart_scan", "pair", ...],
            "capabilities": ["sql_store", "risk_engine", ...],
            "health": { ... },
            "summary_light": { ... },
            "overview_light": { ... },
        }
    """
    if app_ctx is None:
        app_ctx = get_app_context()

    runtime = ensure_dashboard_runtime(app_ctx)

    ff = runtime.feature_flags
    env = runtime.env
    profile = runtime.profile
    run_id = runtime.run_id
    app_name = ff.get("app_name", APP_NAME)
    version = ff.get("version", APP_VERSION)

    active_tabs, active_keys, _ = build_active_tabs(runtime.tab_registry, ff)
    cap_keys = list(runtime.capabilities.keys())

    result: Dict[str, Any] = {
        "env": env,
        "profile": profile,
        "run_id": run_id,
        "app_name": app_name,
        "version": version,
        "active_tabs": active_keys,
        "capabilities": cap_keys,
    }

    # Health קל
    if include_health:
        health = compute_dashboard_health(runtime)
        result["health"] = {
            "ready": health.ready,
            "severity": health.severity,
            "score": health.score,
            "has_critical_issues": health.has_critical_issues,
            "has_warnings": health.has_warnings,
            "issues": list(health.issues),
            "warnings": list(health.warnings),
            "can_trade": health.can_trade,
            "can_backtest": health.can_backtest,
            "can_optimize": health.can_optimize,
            "can_monitor": health.can_monitor,
            "ts_utc": health.ts_utc,
        }

    # Summary "קל" – טיפה יותר עשיר, אבל לא כבד
    if include_summary:
        summary = build_dashboard_summary(runtime)
        result["summary_light"] = {
            "num_services": summary.num_services,
            "num_tabs": summary.num_tabs,
            "severity_counts": dict(summary.severity_counts),
            "service_coverage_ratio": summary.service_coverage_ratio,
            "tabs_coverage_ratio": summary.tabs_coverage_ratio,
            "degraded_services": list(summary.degraded_services),
            "slow_tabs": list(summary.slow_tabs),
            "heavy_tabs": list(summary.heavy_tabs),
        }

    # Overview "קל" – כמה metrics יש, מהם primary וכו'
    if include_overview:
        overview = get_dashboard_overview_from_session() or update_dashboard_overview_in_session(runtime)
        # primary metrics (ממש כמו ב-HomeContext)
        # נשתמש בפונקציה הפנימית מ-Part 34 אם זמינה, אחרת נבחר לפי priority
        try:
            primary_keys = _pick_primary_metrics(overview, max_items=4)  # type: ignore[name-defined]
        except Exception:
            # fallback – פשוט לקחת את 4 הראשונים לפי priority
            def _sort_key(m: Dict[str, Any]) -> Tuple[int, str]:
                return int(m.get("priority", 10)), str(m.get("key") or "")
            sorted_metrics = sorted(overview, key=_sort_key)
            primary_keys = [m.get("key") for m in sorted_metrics[:4]]

        result["overview_light"] = {
            "metrics_count": len(overview),
            "primary_keys": primary_keys,
        }

    return _make_json_safe(result)


def run_headless_agent_actions(
    actions: Sequence[Mapping[str, Any]],
    app_ctx: Optional["AppContext"] = None,
    refresh_overview: bool = True,
    refresh_home_ctx: bool = True,
) -> Dict[str, Any]:
    """
    מריץ סדרת פעולות Agent באופן Headless (ללא UI) – שימושי ל-Tests / CI / סקריפטים.

    דוגמה:
    -------
        res = run_headless_agent_actions([
            {"action": "open_tab", "tab_key": "backtest"},
            {
              "action": "run_backtest_for_pair",
              "payload": {"pair": "AAPL/MSFT", "preset": "smoke", "mode": "wf"}
            },
            {"action": "snapshot"},
        ])

    Flow:
    -----
    1. מבטיח AppContext + Runtime.
    2. מנקה nav_target כדי שנדע מה נוצר בעקבות ה-Actions.
    3. מריץ handle_agent_actions_batch(runtime, actions).
    4. אופציונלי: מרענן Overview/Home context (כדי להחזיר תמונה של המצב אחרי הפעולות).
    5. מחזיר:
        {
            "env": ...,
            "profile": ...,
            "run_id": ...,
            "results": [ {...}, {...}, ... ],
            "nav_target_after": {... or None},
            "overview_light": {...} (אם refresh_overview),
            "home_ctx_light": {...} (אם refresh_home_ctx),
        }
    """
    if app_ctx is None:
        app_ctx = get_app_context()

    runtime = ensure_dashboard_runtime(app_ctx)

    # לפני – ננקה nav_target כדי שנדע מה נוצר כתוצאה מהפעולות
    clear_nav_target()

    results = handle_agent_actions_batch(runtime, actions)

    nav = get_nav_target()

    out: Dict[str, Any] = {
        "env": runtime.env,
        "profile": runtime.profile,
        "run_id": runtime.run_id,
        "results": results,
        "nav_target_after": nav,
    }

    # רענון Overview (אם רוצים לראות השפעה – למשל snapshot, desktop push, וכו')
    if refresh_overview:
        overview = update_dashboard_overview_in_session(runtime)
        out["overview_light"] = {
            "metrics_count": len(overview),
            "keys": [m.get("key") for m in overview],
        }

    # רענון Home context (לבדיקה איך Home ייראה אחרי הפעולות)
    if refresh_home_ctx:
        home_ctx = update_dashboard_home_context_in_session(runtime)
        out["home_ctx_light"] = {
            "health_severity": home_ctx.get("health_severity"),
            "health_score": home_ctx.get("health_score"),
            "num_alerts": home_ctx.get("num_alerts"),
        }

    return _make_json_safe(out)


def export_dashboard_for_test_snapshot(
    app_ctx: Optional["AppContext"] = None,
    include_overview: bool = True,
) -> Dict[str, Any]:
    """
    ייצוא "Snapshot ל-Tests" – יותר עשיר מ-minimal אבל עדיין קומפקטי לצילום / השוואה.

    כולל:
    -----
    - minimal_state  (get_minimal_dashboard_state)
    - api_bundle.meta
    - api_bundle.health.ready / severity / score
    - רשימת active_tabs + services health (שמות+severity)
    - overview_light (מספר metrics + primary keys, אם include_overview=True)

    מאפשר בבדיקות:
    --------------
    - לוודא שהדשבורד "עולה" (ready=True, severity!=error).
    - לראות אם יש שירותים ב-severity=error.
    - להשוות תצורה בין ריצות (env/profile/tabs/services/overview).
    """
    if app_ctx is None:
        app_ctx = get_app_context()

    runtime = ensure_dashboard_runtime(app_ctx)

    minimal = get_minimal_dashboard_state(
        app_ctx,
        include_health=True,
        include_summary=True,
        include_overview=include_overview,
    )

    api_bundle = build_dashboard_api_bundle(
        runtime,
        include_summary=False,
        include_health_summary=True,
        include_agent_context=False,
    )
    api_dict = dashboard_api_bundle_to_dict(api_bundle)

    # שירותים – שם + severity + available
    health = compute_dashboard_health(runtime)
    services = [
        {"name": s.name, "severity": s.severity, "available": s.available}
        for s in (health.summary.services if health.summary is not None else [])
    ]

    snapshot: Dict[str, Any] = {
        "minimal": minimal,
        "api_meta": api_dict.get("meta", {}),
        "api_health": api_dict.get("health", {}),
        "services": services,
    }

    if include_overview:
        overview = get_dashboard_overview_from_session() or update_dashboard_overview_in_session(runtime)
        try:
            primary_keys = _pick_primary_metrics(overview, max_items=4)  # type: ignore[name-defined]
        except Exception:
            primary_keys = [m.get("key") for m in overview[:4]]
        snapshot["overview_light"] = {
            "metrics_count": len(overview),
            "primary_keys": primary_keys,
        }

    return _make_json_safe(snapshot)


# עדכון __all__ עבור חלק 31
try:
    __all__ += [
        "get_minimal_dashboard_state",
        "run_headless_agent_actions",
        "export_dashboard_for_test_snapshot",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "get_minimal_dashboard_state",
        "run_headless_agent_actions",
        "export_dashboard_for_test_snapshot",
    ]

# =====================
