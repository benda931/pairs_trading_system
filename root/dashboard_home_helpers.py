# -*- coding: utf-8 -*-
"""
root/dashboard_home_helpers.py — Dashboard Home Data Helpers
==============================================================

Extracted from dashboard.py Part 34/35.

Contains data-layer helpers for the Home/Monitoring tab context:
overview metrics, health, alerts summary, pinned views.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Import deps
try:
    from root.dashboard import _make_json_safe, ensure_dashboard_runtime
except (ImportError, AttributeError):
    _make_json_safe = lambda x: x
    ensure_dashboard_runtime = lambda ctx: None

try:
    from root.dashboard_health import compute_dashboard_health
except (ImportError, AttributeError):
    compute_dashboard_health = lambda runtime: None

try:
    from root.dashboard_metrics import build_dashboard_overview_metrics, dashboard_overview_metrics_to_dict
except (ImportError, AttributeError):
    build_dashboard_overview_metrics = lambda runtime: []
    dashboard_overview_metrics_to_dict = lambda m: []

try:
    from root.dashboard_alerts_bus import get_dashboard_alerts
except (ImportError, AttributeError):
    get_dashboard_alerts = lambda limit=None: []

try:
    from root.dashboard_integrations import list_saved_views
except (ImportError, AttributeError):
    list_saved_views = lambda: []

# Forward refs
DashboardRuntime = Any

# Part 34/35 – Dashboard Home data helpers (HF-grade Home/Monitoring context)
# =====================

SESSION_KEY_HOME_CONTEXT: str = "dashboard_home_context"


@dataclass
class DashboardHomeContext:
    """
    חבילת דאטה מרוכזת לטאב הבית (Dashboard Home v2 / Monitoring):

    הרעיון:
    --------
    לתת ל-dashboard_home_v2.py "מגש כסף" עם כל מה שהוא צריך:
    - Overview cards (metrics)
    - Alerts
    - מצב Health / Ready
    - Snapshot קל של API Bundle (meta בלבד)
    - מידע על Saved Views / Agent actions אחרונים
    - אינדיקציות יכולת (trade/backtest/optimize)
    - סיכומים לפי קטגוריות (system/risk/macro/data/agents/trading)

    שדות עיקריים:
    --------------
    env / profile / run_id:
        הקונטקסט הלוגי של ה-Runtime.
    overview_metrics:
        רשימת dict-ים עבור Cards (כמו dashboard_overview_metrics_to_dict).
    alerts:
        Alerts אחרונים (warning/error) בפורמט קל.
    health_light:
        סיכום קל של Health (ready / severity / score / issues/warnings קצר).
    api_meta:
        meta קל מתוך Dashboard API Bundle (app_name/version/host/user/ts_utc).
    saved_views:
        רשימה קלה של Saved Views (שם/טאב אחרון/prof/env/tags).
    agent_actions_tail:
        tail קצר של פעולות Agents אחרונות (כדי להציג "מה קרה עכשיו").

    שדות מתקדמים:
    --------------
    created_at_utc:
        זמן יצירת ה-Home context (UTC).
    health_severity:
        "ok" / "warning" / "error" (מ-DashboardHealth).
    health_score:
        ציון Health 0–100.
    num_alerts:
        מספר alerts (warning+error) שנכללו.
    num_critical_alerts:
        מספר alerts מסוג error.
    overview_by_category:
        מיפוי {category → [metric_keys]} (system/risk/macro/data/agents/trading/ops).
    primary_metrics:
        רשימת מפתחות metrics "עיקריים" להצגה ראשית (למשל top-4 לפי priority).
    pinned_view_name:
        שם view "מוקפא" (אם יש כזה – למשל view עם tag "monitoring"/"default").
    can_trade / can_backtest / can_optimize / can_monitor:
        flags שנלקחו מ-DashboardHealth (יכולת אמיתית ברמת קרן).
    """

    env: EnvName
    profile: ProfileName
    run_id: str

    overview_metrics: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    health_light: Dict[str, Any]
    api_meta: Dict[str, Any]
    saved_views: List[Dict[str, Any]]
    agent_actions_tail: List[Dict[str, Any]]

    created_at_utc: str
    health_severity: str
    health_score: float
    num_alerts: int
    num_critical_alerts: int
    overview_by_category: Dict[str, List[str]]
    primary_metrics: List[str]
    pinned_view_name: Optional[str] = None

    can_trade: bool = False
    can_backtest: bool = False
    can_optimize: bool = False
    can_monitor: bool = False


# -------------------------------------------------------------------
# 34.1 – Helpers: alert → light dict, primary metrics & pinned view
# -------------------------------------------------------------------

def _alert_to_light_dict(alert: "DashboardAlert") -> Dict[str, Any]:
    """
    ממיר DashboardAlert למבנה קל לטאב הבית (בלי details כבדים).

    פורמט:
        {
            "id": ...,
            "ts_utc": ...,
            "level": ...,
            "source": ...,
            "message": ...,
        }
    """
    return {
        "id": alert.id,
        "ts_utc": alert.ts_utc,
        "level": alert.level,
        "source": alert.source,
        "message": alert.message,
    }


def _compute_overview_by_category(
    overview_metrics: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    """
    בונה מיפוי category → רשימת metric keys.

    category נלקח מ-overview_metrics[i]["category"] (אם קיים).
    """
    by_cat: Dict[str, List[str]] = {}
    for m in overview_metrics:
        key = str(m.get("key") or "")
        if not key:
            continue
        cat = str(m.get("category") or "system")
        by_cat.setdefault(cat, []).append(key)
    return by_cat


def _pick_primary_metrics(
    overview_metrics: List[Dict[str, Any]],
    max_items: int = 4,
) -> List[str]:
    """
    בוחר את המטריקות "העיקריות" להצגה בראש ה-Home:

    לוגיקה:
    -------
    - קודם מסנן לפי קטגוריות חשובות (system/risk/trading/data).
    - מדרג לפי priority (1 → הכי חשוב).
    - לוקח עד max_items.
    """
    if not overview_metrics:
        return []

    # קטגוריות מועדפות: system, risk, trading, data
    preferred_order = ["system", "risk", "trading", "data", "macro", "agents", "ops"]

    def sort_key(m: Dict[str, Any]) -> Tuple[int, int, str]:
        cat = str(m.get("category") or "system")
        priority = int(m.get("priority", 10))
        key = str(m.get("key") or "")
        try:
            cat_rank = preferred_order.index(cat)
        except ValueError:
            cat_rank = len(preferred_order)
        return cat_rank, priority, key

    sorted_metrics = sorted(overview_metrics, key=sort_key)
    keys: List[str] = []
    for m in sorted_metrics:
        if len(keys) >= max_items:
            break
        key = str(m.get("key") or "")
        if key:
            keys.append(key)
    return keys


def _pick_pinned_view_name(views_export: Dict[str, Any]) -> Optional[str]:
    """
    בוחר pinned view מתוך export_saved_views_for_agents(runtime):

    אסטרטגיה:
    ---------
    1. אם יש view עם tag "monitoring" או "default" – ניקח אותו.
    2. אחרת – ניקח את הראשון ברשימה, אם יש.
    3. אם אין views – נחזיר None.
    """
    views = views_export.get("views") or []
    if not isinstance(views, list) or not views:
        return None

    # חפש view עם tag 'monitoring' או 'default'
    for v in views:
        if not isinstance(v, Mapping):
            continue
        tags = v.get("tags") or []
        if not isinstance(tags, list):
            continue
        tags_lower = [str(t).lower() for t in tags]
        if "monitoring" in tags_lower or "default" in tags_lower:
            return str(v.get("name") or "") or None

    # fallback – הראשון ברשימה
    name = views[0].get("name") if isinstance(views[0], Mapping) else None
    return str(name) if name else None


# -------------------------------------------------------------------
# 34.2 – Core builder: DashboardHomeContext (with caching)
# -------------------------------------------------------------------

def build_dashboard_home_context(runtime: DashboardRuntime) -> DashboardHomeContext:
    """
    בונה DashboardHomeContext מלא מתוך Runtime, עם Cache פנימי קצר (≈5 שניות):

    Flow:
    -----
    1. בודק Cache (namespace="home_ctx", key=env|profile|run_id).
    2. אם אין / פג תוקף:
        - Overview metrics (cards) – build_dashboard_overview_metrics/update_dashboard_overview_in_session.
        - Alerts recent (warning/error) – get_dashboard_alerts.
        - Health light – compute_dashboard_health.
        - API meta – מתוך Dashboard API Bundle.
        - Saved views (light) – export_saved_views_for_agents.
        - Agent actions tail – get_agent_actions_history_tail.
        - הפקה של:
            * overview_by_category
            * primary_metrics
            * pinned_view_name
            * יכולות מוצריות (can_trade/backtest/optimize/monitor)
    3. שומר ב-Cache ומחזיר.
    """
    cache_key = f"{runtime.env}|{runtime.profile}|{runtime.run_id}"
    cached = _cache_get("home_ctx", cache_key)
    if isinstance(cached, DashboardHomeContext):
        return cached

    # 1) Overview metrics (dict form) + עדכון ב-session
    overview_dicts = update_dashboard_overview_in_session(runtime)

    # 2) Alerts (רק warning/error)
    alerts_objs = get_dashboard_alerts(limit=20)
    alerts_filtered = [
        a for a in alerts_objs if a.level in ("warning", "error")
    ]
    alerts_light = [_alert_to_light_dict(a) for a in alerts_filtered]
    num_alerts = len(alerts_light)
    num_critical = sum(1 for a in alerts_light if a.get("level") == "error")

    # 3) Health (מלא) → Health light
    health = compute_dashboard_health(runtime)
    health_light = {
        "ready": health.ready,
        "severity": health.severity,
        "score": health.score,
        "has_critical_issues": health.has_critical_issues,
        "has_warnings": health.has_warnings,
        "issues": list(health.issues),
        "warnings": list(health.warnings),
        "ts_utc": health.ts_utc,
        "health_id": health.health_id,
    }

    # 4) API meta only (קל)
    api_bundle = build_dashboard_api_bundle(
        runtime,
        include_summary=False,
        include_health_summary=False,
        include_agent_context=False,
    )
    api_meta = api_bundle.meta

    # 5) Saved views (light)
    views_export = export_saved_views_for_agents(runtime)
    saved_views_light = views_export.get("views", [])
    pinned_view_name = _pick_pinned_view_name(views_export)

    # 6) Agent actions tail
    actions_tail = get_agent_actions_history_tail(limit=30)

    # 7) תוספות מתקדמות
    overview_by_category = _compute_overview_by_category(overview_dicts)
    primary_metrics = _pick_primary_metrics(overview_dicts, max_items=4)

    created_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    ctx = DashboardHomeContext(
        env=runtime.env,
        profile=runtime.profile,
        run_id=runtime.run_id,
        overview_metrics=overview_dicts,
        alerts=alerts_light,
        health_light=health_light,
        api_meta=api_meta,
        saved_views=saved_views_light,
        agent_actions_tail=actions_tail,
        created_at_utc=created_at_utc,
        health_severity=health.severity,
        health_score=health.score,
        num_alerts=num_alerts,
        num_critical_alerts=num_critical,
        overview_by_category=overview_by_category,
        primary_metrics=primary_metrics,
        pinned_view_name=pinned_view_name,
        can_trade=health.can_trade,
        can_backtest=health.can_backtest,
        can_optimize=health.can_optimize,
        can_monitor=health.can_monitor,
    )

    _cache_set("home_ctx", cache_key, ctx, ttl=5.0)
    return ctx


# -------------------------------------------------------------------
# 34.3 – Dict export / session update / external export
# -------------------------------------------------------------------

def dashboard_home_context_to_dict(ctx: DashboardHomeContext) -> Dict[str, Any]:
    """
    ממיר DashboardHomeContext ל-dict JSON-friendly – מיועד ל-Agents / Desktop / REST.
    """
    return {
        "env": ctx.env,
        "profile": ctx.profile,
        "run_id": ctx.run_id,
        "overview_metrics": _make_json_safe(ctx.overview_metrics),
        "alerts": _make_json_safe(ctx.alerts),
        "health_light": _make_json_safe(ctx.health_light),
        "api_meta": _make_json_safe(ctx.api_meta),
        "saved_views": _make_json_safe(ctx.saved_views),
        "agent_actions_tail": _make_json_safe(ctx.agent_actions_tail),
        "created_at_utc": ctx.created_at_utc,
        "health_severity": ctx.health_severity,
        "health_score": ctx.health_score,
        "num_alerts": ctx.num_alerts,
        "num_critical_alerts": ctx.num_critical_alerts,
        "overview_by_category": _make_json_safe(ctx.overview_by_category),
        "primary_metrics": list(ctx.primary_metrics),
        "pinned_view_name": ctx.pinned_view_name,
        "can_trade": ctx.can_trade,
        "can_backtest": ctx.can_backtest,
        "can_optimize": ctx.can_optimize,
        "can_monitor": ctx.can_monitor,
    }


def update_dashboard_home_context_in_session(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    בונה DashboardHomeContext, שומר אותו ב-session_state, ומחזיר dict JSON-friendly.

    מיועד ל:
    --------
    - Dashboard Home v2 (root/dashboard_home_v2.py) – יכול פשוט לקרוא
      get_dashboard_home_context_from_session() כדי לקבל את כל ההקשר.
    - Agents Tab / Desktop – לשימוש בתצוגות Monitoring.
    """
    ctx = build_dashboard_home_context(runtime)
    ctx_dict = dashboard_home_context_to_dict(ctx)

    try:
        st.session_state[SESSION_KEY_HOME_CONTEXT] = ctx_dict
    except Exception:  # pragma: no cover
        pass

    logger.debug(
        "Dashboard home context updated (env=%s, profile=%s, metrics=%d, alerts=%d, views=%d)",
        ctx.env,
        ctx.profile,
        len(ctx.overview_metrics),
        len(ctx.alerts),
        len(ctx.saved_views),
    )

    return ctx_dict


def get_dashboard_home_context_from_session() -> Optional[Dict[str, Any]]:
    """
    מחזיר את Dashboard Home context האחרון מה-Session, אם קיים.

    אם לא קיים – מחזיר None (הטאב home יכול לבחור לבנות אחד חדש דרך Runtime).
    """
    try:
        obj = st.session_state.get(SESSION_KEY_HOME_CONTEXT)
    except Exception:  # pragma: no cover
        obj = None

    if not isinstance(obj, Mapping):
        return None

    return dict(obj)


# עדכון __all__ עבור חלק 34
try:
    __all__ += [
        "SESSION_KEY_HOME_CONTEXT",
        "DashboardHomeContext",
        "build_dashboard_home_context",
        "dashboard_home_context_to_dict",
        "update_dashboard_home_context_in_session",
        "get_dashboard_home_context_from_session",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_HOME_CONTEXT",
        "DashboardHomeContext",
        "build_dashboard_home_context",
        "dashboard_home_context_to_dict",
        "update_dashboard_home_context_in_session",
        "get_dashboard_home_context_from_session",
    ]

# =====================
