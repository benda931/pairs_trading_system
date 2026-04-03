# -*- coding: utf-8 -*-
"""
root/dashboard_integrations.py — Desktop Bridge, Agent Router, Saved Views
==========================================================================

Extracted from dashboard.py Parts 27, 28, 29.

Contains:
- Desktop bridge hooks (DashboardDesktopBridgeConfig, push state to desktop)
- Agent action router (handle_agent_action, batch actions)
- Saved views/layout profiles (SavedDashboardView, list/add/apply/export)

All functions import their dependencies from root.dashboard at call time
to avoid circular imports.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# Import shared deps from dashboard at module level (these are stable exports)
try:
    from root.dashboard import (
        _make_json_safe,
        _normalize_env,
        _normalize_profile,
        _discover_services_mapping,
        _probe_service,
        set_nav_target,
        get_last_active_tab_key,
        _is_tab_enabled_for_profile,
        _collect_session_nav_history_tail,
        trigger_dashboard_snapshot,
        build_active_tabs,
        DEFAULT_ENV,
        DEFAULT_PROFILE,
        TAB_KEY_HOME,
        TAB_KEY_BACKTEST,
        TAB_KEY_SMART_SCAN,
        TAB_KEY_PAIR,
        TAB_KEY_RISK,
        TAB_KEY_MACRO,
        TAB_KEY_MATRIX,
        TAB_KEY_COMPARISON_MATRICES,
        TAB_KEY_PORTFOLIO,
        TAB_KEY_FAIR_VALUE,
        TAB_KEY_LOGS,
    )
except ImportError:
    # Fallbacks for standalone testing
    _make_json_safe = lambda x: x
    _normalize_env = lambda x: x or "dev"
    _normalize_profile = lambda x: x or "dev_default"
    _discover_services_mapping = lambda ctx: {}
    _probe_service = lambda ctx, m, candidates=(): (False, None)
    set_nav_target = lambda key, payload: None
    get_last_active_tab_key = lambda default="home": default
    _is_tab_enabled_for_profile = lambda meta, profile: True
    _collect_session_nav_history_tail = lambda limit=50: []
    trigger_dashboard_snapshot = lambda runtime: {}
    build_active_tabs = lambda registry, ff: []
    DEFAULT_ENV = "dev"
    DEFAULT_PROFILE = "dev_default"
    TAB_KEY_HOME = "home"
    TAB_KEY_BACKTEST = "backtest"
    TAB_KEY_SMART_SCAN = "smart_scan"
    TAB_KEY_PAIR = "pair"
    TAB_KEY_RISK = "risk"
    TAB_KEY_MACRO = "macro"
    TAB_KEY_MATRIX = "matrix"
    TAB_KEY_COMPARISON_MATRICES = "comparison_matrices"
    TAB_KEY_PORTFOLIO = "portfolio"
    TAB_KEY_FAIR_VALUE = "fair_value"
    TAB_KEY_LOGS = "logs"

# Type aliases (avoid importing heavy types)
EnvName = str
ProfileName = str
TabKey = str
DashboardRuntime = Any  # Forward reference
TabMeta = Any  # Forward reference

# Part 27/35 – Desktop bridge hooks & integration contracts (HF-grade)
# =====================

SESSION_KEY_DESKTOP_LAST_PUSH: str = "dashboard_desktop_last_push"


@dataclass
class DashboardDesktopBridgeConfig:
    """
    תצורת אינטגרציה לדסקטופ (Qt / root_desktop.app):

    מטרות:
    -------
    - להגדיר "חוזה" ברור בין Web Dashboard לבין שכבת ה-Desktop.
    - לאפשר ל-Desktop לדעת:
        * כל כמה זמן למשוך/לקבל סטטוס.
        * איזה פרופיל/טאבים מעניינים אותו (monitoring/trading/risk).
        * האם להסתמך על Health בלבד או גם על Summary/Agent Context.

    שדות:
    -----
    enabled:
        האם בכלל לנסות אינטגרציה לדסקטופ (ברירת מחדל: True אם capabilities.desktop_integration).
    preferred_profile_for_desktop:
        איזה profile הדסקטופ אמור לראות כברירת מחדל (למשל "monitoring").
    push_interval_sec:
        כל כמה שניות (בערך) מומלץ לדסקטופ למשוך/לקבל עדכון (health/api bundle).
        הערה: זה לא timer אמיתי – רק המלצה ל-Desktop Bridge.
    include_summary:
        האם לכלול Summary מלא ב-payload לדסקטופ (כבד יותר אבל עשיר).
    include_agent_context:
        האם לכלול Agent Context (nav history, timings, errors, base_ctx).
    """

    enabled: bool = True
    preferred_profile_for_desktop: ProfileName = "monitoring"
    push_interval_sec: int = 15
    include_summary: bool = True
    include_agent_context: bool = True


def _detect_desktop_bridge_service(app_ctx: "AppContext") -> Tuple[bool, Any]:
    """
    מנסה לגלות Desktop Bridge service מתוך AppContext:

    מחפש בשמות:
        - desktop_bridge
        - desktop
        - qt_bridge
        - qt_app
        - desktop_context

    ומנסה גם בתוך services_map (אם קיים).

    מחזיר:
        (found: bool, service_or_None)
    """
    services_map = _discover_services_mapping(app_ctx)
    found, svc = _probe_service(
        app_ctx,
        services_map,
        candidates=("desktop_bridge", "desktop", "qt_bridge", "qt_app", "desktop_context"),
    )
    return bool(found and svc is not None), svc


def build_desktop_integration_config(runtime: DashboardRuntime) -> DashboardDesktopBridgeConfig:
    """
    בונה DashboardDesktopBridgeConfig מתוך Runtime:

    לוגיקה:
    -------
    - enabled:
        * True אם capabilities.desktop_integration=True.
        * אחרת False (אלא אם בעתיד נחליט להכריח).
    - preferred_profile_for_desktop:
        * אם profile הנוכחי הוא monitoring/risk → נשאיר אותו.
        * אחרת – "monitoring" (דסקטופ נועד לרוב למוניטורינג).
    - push_interval_sec:
        * dev/research → 5–10 שניות (עדכונים תכופים).
        * live/paper → 15–30 שניות.
        * אחרת – 15 שניות.
    """
    caps = runtime.capabilities
    env = runtime.env
    profile = runtime.profile

    enabled = bool(caps.get("desktop_integration", False))

    if profile in ("monitoring", "risk"):
        preferred_profile = profile
    else:
        preferred_profile = "monitoring"

    if env in ("dev", "research", "test"):
        interval = 5
    elif env in ("live", "paper"):
        interval = 20
    else:
        interval = 15

    cfg = DashboardDesktopBridgeConfig(
        enabled=enabled,
        preferred_profile_for_desktop=preferred_profile,  # type: ignore[arg-type]
        push_interval_sec=interval,
        include_summary=True,
        include_agent_context=True,
    )

    return cfg


def build_desktop_payload_from_runtime(
    runtime: DashboardRuntime,
    config: Optional[DashboardDesktopBridgeConfig] = None,
) -> Dict[str, Any]:
    """
    בונה payload עשיר לדסקטופ מתוך Runtime + DesktopBridgeConfig:

    הפלט:
    -----
    dict עם מפתחות עיקריים:
        - "config"     – dict של DashboardDesktopBridgeConfig.
        - "api_bundle" – DashboardAPI bundle (meta+health+summary+agent_context).
        - "tabs"       – רשימת טאבים פעילים (key/label/group).
        - "env" / "profile" / "run_id"  – קונטקסט.

    מבוסס על:
        - export_dashboard_api_bundle(...) (מהחלקים הקודמים).
        - build_active_tabs(...) (לרשימת טאבים).
    """
    if config is None:
        config = build_desktop_integration_config(runtime)

    # api_bundle – מרכז health/summary/agent_context כבר עכשיו
    api_bundle = build_dashboard_api_bundle(
        runtime,
        include_summary=config.include_summary,
        include_health_summary=True,
        include_agent_context=config.include_agent_context,
    )
    api_bundle_dict = dashboard_api_bundle_to_dict(api_bundle)

    # tabs overview
    active_tabs, active_keys, _ = build_active_tabs(runtime.tab_registry, runtime.feature_flags)
    tabs_view = [
        {
            "key": t.key,
            "label": t.label,
            "group": t.group,
            "enabled": True,
        }
        for t in active_tabs
    ]

    payload: Dict[str, Any] = {
        "env": runtime.env,
        "profile": runtime.profile,
        "run_id": runtime.run_id,
        "config": {
            "enabled": config.enabled,
            "preferred_profile_for_desktop": config.preferred_profile_for_desktop,
            "push_interval_sec": config.push_interval_sec,
            "include_summary": config.include_summary,
            "include_agent_context": config.include_agent_context,
        },
        "api_bundle": api_bundle_dict,
        "tabs": tabs_view,
    }

    return _make_json_safe(payload)


def push_dashboard_state_to_desktop_if_available(
    runtime: DashboardRuntime,
    config: Optional[DashboardDesktopBridgeConfig] = None,
) -> Tuple[bool, Optional[str]]:
    """
    מנסה לדחוף את מצב הדשבורד ל-Desktop Bridge, אם קיים:

    Flow:
    -----
    1. בודק capabilities.desktop_integration.
    2. מחפש Desktop Bridge service ב-AppContext:
        - מצפה לאובייקט עם אחת מהפונקציות:
            * push_dashboard_state(payload)
            * push_state(payload)
            * send(payload)
    3. בונה payload דרך build_desktop_payload_from_runtime(...).
    4. קורא לפונקציה הראשונה שמצאנו, מחזיר (True, method_name) אם הצליח.

    הערות:
    -------
    - אם אין Desktop Bridge או capabilities.desktop_integration=False → מחזיר (False, None).
    - לא מפיל את הדשבורד במקרה של שגיאה – רק כותב לוג.
    - מתאים לטריגר:
        * מתוך Agents Tab.
        * מתוך Quick action עתידי "Push state to Desktop".
        * מתוך Desktop עצמו (pull/poll).
    """
    caps = runtime.capabilities
    if not caps.get("desktop_integration", False):
        return False, None

    found, bridge = _detect_desktop_bridge_service(runtime.app_ctx)
    if not found:
        logger.debug("Desktop integration capability enabled, but no desktop_bridge service found.")
        return False, None

    payload = build_desktop_payload_from_runtime(runtime, config)

    method_candidates: Sequence[str] = (
        "push_dashboard_state",
        "push_state",
        "send",
    )

    for name in method_candidates:
        fn = getattr(bridge, name, None)
        if not callable(fn):
            continue
        try:
            fn(payload)  # type: ignore[misc]
            logger.info(
                "Pushed dashboard state to Desktop via %s.%s",
                type(bridge).__name__,
                name,
            )
            # נשמור timestamp אחרון ל-Session (optional telemetry)
            try:
                st.session_state[SESSION_KEY_DESKTOP_LAST_PUSH] = {
                    "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "method": name,
                }
            except Exception:  # pragma: no cover
                pass
            return True, name
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Desktop bridge method %s.%s raised %s – skipping.",
                type(bridge).__name__,
                name,
                exc,
            )

    logger.warning(
        "Desktop bridge service found, but no compatible push method "
        "(push_dashboard_state/push_state/send) succeeded."
    )
    return False, None


def get_last_desktop_push_info() -> Optional[Dict[str, Any]]:
    """
    מחזיר מידע על ה-"Push" האחרון שבוצע ל-Desktop (אם קיים):

    מבנה:
        {
            "ts_utc": "...",
            "method": "push_dashboard_state" | "push_state" | "send",
        }

    שימוש:
    -------
    - להצגה ב-Logs Tab / Agents Tab / Toolbar בעת פיתוח.
    """
    try:
        info = st.session_state.get(SESSION_KEY_DESKTOP_LAST_PUSH)
    except Exception:  # pragma: no cover
        info = None

    if not isinstance(info, Mapping):
        return None

    return dict(info)


# עדכון __all__ עבור חלק 27
try:
    __all__ += [
        "SESSION_KEY_DESKTOP_LAST_PUSH",
        "DashboardDesktopBridgeConfig",
        "build_desktop_integration_config",
        "build_desktop_payload_from_runtime",
        "push_dashboard_state_to_desktop_if_available",
        "get_last_desktop_push_info",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_DESKTOP_LAST_PUSH",
        "DashboardDesktopBridgeConfig",
        "build_desktop_integration_config",
        "build_desktop_payload_from_runtime",
        "push_dashboard_state_to_desktop_if_available",
        "get_last_desktop_push_info",
    ]
# =====================
# Part 28/35 – Agent Action Router (HF-grade AI command interface)
# =====================

SESSION_KEY_AGENT_ACTIONS_HISTORY: str = "dashboard_agent_actions_history"


@dataclass
class AgentAction:
    """
    ייצוג מובנה לפעולה שמגיעה מסוכן AI / Agents Tab / Consumer חיצוני.

    דוגמאות:
    ---------
    - פתיחת טאב:
        action = "open_tab", tab_key = "backtest"
    - פוקוס על זוג והרצת Backtest:
        action = "run_backtest_for_pair",
        tab_key = "backtest",
        payload = {"pair": "AAPL/MSFT", "preset": "smoke", "mode": "wf"}
    - מעבר ל-Risk View:
        action = "open_risk_view",
        tab_key = "risk",
        payload = {"portfolio_id": "default", "view": "limits"}
    - Push של מצב לדסקטופ:
        action = "push_to_desktop"
    - שמירת View:
        action = "save_view",
        payload = {"name": "My favorite monitor setup"}

    שדות:
    -----
    source:
        מקור הפעולה – למשל "agent_tab", "external_api", "desktop", "auto_supervisor".
    action:
        שם הפעולה (open_tab / run_backtest_for_pair / open_risk_view / push_to_desktop / save_view / snapshot ...).
    tab_key:
        טאב יעד, אם רלוונטי (backtest / risk / macro / pair / matrix ...).
    payload:
        פרמטרים נוספים – pair/preset/mode/portfolio_id וכו'.
    ts_utc:
        זמן יצירת הפעולה ב-UTC.
    """

    source: str
    action: str
    tab_key: Optional[TabKey] = None
    payload: Optional[NavPayload] = None
    ts_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def _ensure_agent_actions_history() -> List[Dict[str, Any]]:
    """
    מבטיח שב-session_state תהיה רשימת היסטוריית Actions של סוכנים:

        [
            {
                "ts_utc": "...",
                "source": "...",
                "action": "...",
                "tab_key": "...",
                "payload_keys": [...],
                "result": "ok" | "ignored" | "error",
            },
            ...
        ]
    """
    try:
        obj = st.session_state.get(SESSION_KEY_AGENT_ACTIONS_HISTORY)
    except Exception:  # pragma: no cover
        obj = None

    if not isinstance(obj, list):
        obj = []
        try:
            st.session_state[SESSION_KEY_AGENT_ACTIONS_HISTORY] = obj
        except Exception:
            pass
    return obj  # type: ignore[return-value]


def _record_agent_action_entry(
    action: AgentAction,
    result: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    מוסיף רשומת Action אחת להיסטוריית סוכנים (Session-level).

    result:
        "ok"      – הפעולה מוּשׂמה (למשל nav_target נוצר, snapshot נשמר).
        "ignored" – הפעולה לא רלוונטית / לא נתמכת.
        "error"   – ניסיון ביצוע נכשל (למשל Desktop bridge זרק Exception).
    """
    history = _ensure_agent_actions_history()

    if isinstance(action.payload, Mapping):
        payload_keys = list(action.payload.keys())
    else:
        payload_keys = None

    entry: Dict[str, Any] = {
        "ts_utc": action.ts_utc,
        "source": action.source,
        "action": action.action,
        "tab_key": action.tab_key,
        "payload_keys": payload_keys,
        "result": result,
    }
    if extra:
        entry["extra"] = _make_json_safe(extra)

    history.append(entry)
    try:
        st.session_state[SESSION_KEY_AGENT_ACTIONS_HISTORY] = history
    except Exception:  # pragma: no cover
        pass


def _agent_action_open_tab(
    runtime: DashboardRuntime,
    action: AgentAction,
) -> Dict[str, Any]:
    """
    מטפל בפעולות מסוג "open_tab", "open_risk_view", "open_macro_view" וכו' – כלומר
    פעולות שהמשמעות העיקרית שלהן היא ניווט לטאב מסוים.

    לוגיקה:
    -------
    - אם tab_key לא הועבר בפעולה → נתעלם (ignored).
    - אם הטאב אינו מוגדר ב-registry → ignored.
    - אם הטאב לא enabled עבור הפרופיל הנוכחי → ignored.
    - אחרת → set_nav_target(tab_key, payload) (payload יכולה להיות None או dict).
    """
    tab_key = action.tab_key
    if not tab_key:
        _record_agent_action_entry(action, "ignored", {"reason": "missing_tab_key"})
        return {"status": "ignored", "reason": "missing_tab_key"}

    meta = runtime.tab_registry.get(tab_key)
    if meta is None:
        _record_agent_action_entry(action, "ignored", {"reason": "unknown_tab"})
        return {"status": "ignored", "reason": "unknown_tab"}

    if not _is_tab_enabled_for_profile(meta, runtime.profile):
        _record_agent_action_entry(
            action,
            "ignored",
            {"reason": "tab_not_enabled_for_profile", "profile": runtime.profile},
        )
        return {
            "status": "ignored",
            "reason": "tab_not_enabled_for_profile",
            "profile": runtime.profile,
        }

    set_nav_target(tab_key, action.payload)
    _record_agent_action_entry(action, "ok", {"nav_target_set": True})

    return {
        "status": "ok",
        "nav_target": {"tab_key": tab_key, "payload": action.payload or {}},
    }


def _agent_action_run_backtest_for_pair(
    runtime: DashboardRuntime,
    action: AgentAction,
) -> Dict[str, Any]:
    """
    פעולה יעודית: הרצת Backtest לזוג מסוים / preset / mode.

    היא בפועל *רק* מגדירה nav_target ל-"backtest" עם payload עשיר, כדי
    לאפשר לטאב ה-Backtest לטעון את ההקשר ולבצע את ההרצה.

    מצפה ל-payload בסגנון:
        {"pair": "AAPL/MSFT", "preset": "smoke", "mode": "wf"}
    """
    # אם לא הוגדר tab_key – נכפה "backtest"
    if not action.tab_key:
        action.tab_key = TAB_KEY_BACKTEST

    # נשתמש באותה לוגיקה של _agent_action_open_tab
    result = _agent_action_open_tab(runtime, action)
    result["hint"] = "backtest_for_pair"
    return result


def _agent_action_snapshot(
    runtime: DashboardRuntime,
    action: AgentAction,
) -> Dict[str, Any]:
    """
    פעולה: יצירת Snapshot של הדשבורד (Session + SqlStore אם זמין).

    קוראת ל-trigger_dashboard_snapshot(...) ובונה תוצאה.
    """
    success, method_name = trigger_dashboard_snapshot(
        runtime.app_ctx,
        runtime.feature_flags,
        runtime.services_status,
    )

    status = "ok" if success else "warning"
    info = {
        "status": status,
        "saved_to_sql_store": success,
        "sql_method": method_name,
    }

    _record_agent_action_entry(action, status, info)
    return info


def _agent_action_push_to_desktop(
    runtime: DashboardRuntime,
    action: AgentAction,
) -> Dict[str, Any]:
    """
    פעולה: Push של מצב הדשבורד ל-Desktop Bridge (אם קיים).

    קוראת ל-push_dashboard_state_to_desktop_if_available(...)
    ומחזירה מצב: הצליח/לא, באיזו שיטה.
    """
    config = build_desktop_integration_config(runtime)
    if not config.enabled:
        info = {
            "status": "ignored",
            "reason": "desktop_integration_disabled",
        }
        _record_agent_action_entry(action, "ignored", info)
        return info

    success, method_name = push_dashboard_state_to_desktop_if_available(runtime, config)
    status = "ok" if success else "warning"
    info = {
        "status": status,
        "method": method_name,
        "config": {
            "preferred_profile_for_desktop": config.preferred_profile_for_desktop,
            "push_interval_sec": config.push_interval_sec,
        },
    }
    _record_agent_action_entry(action, status, info)
    return info


def _agent_action_save_view(
    runtime: DashboardRuntime,
    action: AgentAction,
) -> Dict[str, Any]:
    """
    פעולה: שמירת "View" של הדשבורד (ברמת Session בלבד, בשלב זה).

    payload יכולה להכיל:
        - "name": שם ה-View (למשל "Morning monitor").
        - שדות נוספים (לשימוש עתידי).

    בשלב הזה:
        - נשמור רשומה ב-session_state["saved_views"] (אם אין – ניצור).
        - בהמשך ניתן לסנכרן ל-SqlStore.
    """
    try:
        views = st.session_state.get("dashboard_saved_views", [])
    except Exception:
        views = []

    if not isinstance(views, list):
        views = []

    payload = action.payload or {}
    name = str(payload.get("name") or "").strip() or f"view_{len(views) + 1}"

    view_entry = {
        "name": name,
        "ts_utc": action.ts_utc,
        "env": runtime.env,
        "profile": runtime.profile,
        "last_tab_key": get_last_active_tab_key(default=TAB_KEY_HOME),
        "nav_history_tail": _collect_session_nav_history_tail(limit=20),
        "feature_flags": {
            "env": runtime.env,
            "profile": runtime.profile,
            "enable_live_trading_actions": runtime.feature_flags.get(
                "enable_live_trading_actions"
            ),
            "enable_experiment_mode": runtime.feature_flags.get(
                "enable_experiment_mode"
            ),
        },
    }

    views.append(_make_json_safe(view_entry))
    try:
        st.session_state["dashboard_saved_views"] = views
    except Exception:  # pragma: no cover
        pass

    info = {"status": "ok", "name": name, "count": len(views)}
    _record_agent_action_entry(action, "ok", info)
    return info


def handle_agent_action(
    runtime: DashboardRuntime,
    action_data: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Router מרכזי לפעולות של סוכנים (Agent Action Router):

    קלט:
    ----
    action_data – Mapping, למשל:
        {
            "source": "agent_tab",
            "action": "run_backtest_for_pair",
            "tab_key": "backtest",
            "payload": {"pair": "AAPL/MSFT", "preset": "smoke", "mode": "wf"},
        }

    Flow:
    -----
    1. הופך את action_data ל-AgentAction (עם ts_utc).
    2. לפי action:
        - "open_tab" / "open_risk_view" / "open_macro_view" → _agent_action_open_tab
        - "run_backtest_for_pair" → _agent_action_run_backtest_for_pair
        - "snapshot" → _agent_action_snapshot
        - "push_to_desktop" → _agent_action_push_to_desktop
        - "save_view" → _agent_action_save_view
    3. עבור פעולה לא מוכרת → ignored.

    הפלט:
    -----
    dict עם שדות:
        - "status": "ok" / "ignored" / "warning"
        - שדות נוספים לפי הפעולה.
    """
    # ---- בניית AgentAction מובנה ----
    source = str(action_data.get("source") or "agent")
    act = str(action_data.get("action") or "").strip()
    tab_key_raw = action_data.get("tab_key")
    tab_key = str(tab_key_raw).strip() if tab_key_raw else None

    payload_raw = action_data.get("payload")
    if isinstance(payload_raw, Mapping):
        payload = dict(payload_raw)
    elif payload_raw is None:
        payload = None
    else:
        payload = {"value": payload_raw}

    action = AgentAction(
        source=source,
        action=act,
        tab_key=tab_key,
        payload=payload,
    )

    # ---- ניתוב לפי שם פעולה ----
    if act in ("open_tab", "open_risk_view", "open_macro_view", "open_insights"):
        return _agent_action_open_tab(runtime, action)

    if act == "run_backtest_for_pair":
        return _agent_action_run_backtest_for_pair(runtime, action)

    if act in ("snapshot", "create_snapshot"):
        return _agent_action_snapshot(runtime, action)

    if act in ("push_to_desktop", "sync_desktop"):
        return _agent_action_push_to_desktop(runtime, action)

    if act in ("save_view", "bookmark_view"):
        return _agent_action_save_view(runtime, action)

    # פעולה לא מוכרת → ignored
    info = {"status": "ignored", "reason": "unknown_action", "action": act}
    _record_agent_action_entry(action, "ignored", info)
    return info


def handle_agent_actions_batch(
    runtime: DashboardRuntime,
    actions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    מאפשר לסוכן/טאב לשלוח Batch של פעולות בפעם אחת:

        results = handle_agent_actions_batch(runtime, [
            {"action": "open_tab", "tab_key": "pair", ...},
            {"action": "run_backtest_for_pair", ...},
        ])

    הפלט:
        רשימת תוצאות (dict) עבור כל פעולה.
    """
    results: List[Dict[str, Any]] = []
    for data in actions:
        try:
            res = handle_agent_action(runtime, data)
        except Exception as exc:  # pragma: no cover
            res = {
                "status": "error",
                "error": str(exc),
            }
        results.append(res)
    return results


def get_agent_actions_history_tail(limit: int = 50) -> List[Dict[str, Any]]:
    """
    מחזיר את זנב היסטוריית פעולות הסוכנים (Agent Actions) לצורכי Debug/Agents Tab.
    """
    history = _ensure_agent_actions_history()
    if not history:
        return []
    return history[-limit:]


# עדכון __all__ עבור חלק 28
try:
    __all__ += [
        "SESSION_KEY_AGENT_ACTIONS_HISTORY",
        "AgentAction",
        "handle_agent_action",
        "handle_agent_actions_batch",
        "get_agent_actions_history_tail",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_AGENT_ACTIONS_HISTORY",
        "AgentAction",
        "handle_agent_action",
        "handle_agent_actions_batch",
        "get_agent_actions_history_tail",
    ]
# =====================
# Part 29/35 – Saved Views & Layout Profiles (HF-grade view management)
# =====================

SESSION_KEY_SAVED_VIEWS: str = "dashboard_saved_views"


@dataclass
class SavedDashboardView:
    """
    Saved View / Layout Profile ברמת קרן – צילום מצב "אנטומי" של הדשבורד.

    למה זה קיים?
    -------------
    - לאפשר:
        * "בוקר מסחר" → View מסוים (טאבים, פרופיל, סידור, קונטקסט).
        * "מחקר מאקרו" → View אחר (פרופיל macro, טאב macro/matrix וכו').
    - לתת לסוכנים / Desktop אופציה לטעון, לשמור ולשתף Views באופן מובנה.

    שדות:
    -----
    name:
        שם ה-View (ידידותי למשתמש), למשל "Morning monitor" / "Risk overview".
    ts_utc:
        זמן יצירת ה-View (UTC, isoformat).
    env / profile:
        הסביבה והפרופיל שבהם ה-View נוצר.
    last_tab_key:
        הטאב האחרון שהיה פעיל בעת יצירת ה-View.
    feature_flags:
        תת-סט רלוונטי של FeatureFlags – env/profile/live_actions/experiment_mode וכו'.
    nav_history_tail:
        זנב הניווט בזמן יצירת ה-View – מאפשר להבין "איך הגיעו לכאן".
    notes:
        הערות חופשיות (אופציונלי).
    tags:
        רשימת טאגים (למשל ["monitoring", "risk", "live"]).
    """

    name: str
    ts_utc: str
    env: EnvName
    profile: ProfileName
    last_tab_key: TabKey
    feature_flags: Dict[str, Any]
    nav_history_tail: List[Dict[str, Any]] = field(default_factory=list)
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)


def _ensure_saved_views_list() -> List[Dict[str, Any]]:
    """
    מבטיח שב-session_state יש רשימת Saved Views במפתח SESSION_KEY_SAVED_VIEWS.

    מבנה בסיסי:
        [
            { ... SavedDashboardView כ-dict JSON-friendly ... },
            ...
        ]
    """
    try:
        obj = st.session_state.get(SESSION_KEY_SAVED_VIEWS, [])
    except Exception:  # pragma: no cover
        obj = []

    if not isinstance(obj, list):
        obj = []
        try:
            st.session_state[SESSION_KEY_SAVED_VIEWS] = obj
        except Exception:
            pass
    return obj  # type: ignore[return-value]


def _saved_view_to_dict(view: SavedDashboardView) -> Dict[str, Any]:
    """
    ממיר SavedDashboardView ל-dict JSON-friendly.
    """
    return {
        "name": view.name,
        "ts_utc": view.ts_utc,
        "env": view.env,
        "profile": view.profile,
        "last_tab_key": view.last_tab_key,
        "feature_flags": view.feature_flags,
        "nav_history_tail": view.nav_history_tail,
        "notes": view.notes,
        "tags": list(view.tags),
    }


def _saved_view_from_mapping(data: Mapping[str, Any]) -> Optional[SavedDashboardView]:
    """
    מנסה לבנות SavedDashboardView מתוך Mapping.

    במקרה של שדות חסרים באופן קריטי – מחזיר None.
    """
    try:
        name = str(data.get("name") or "").strip()
        ts_utc = str(data.get("ts_utc") or "").strip()
        env = _normalize_env(str(data.get("env") or DEFAULT_ENV))
        profile = _normalize_profile(str(data.get("profile") or DEFAULT_PROFILE))
        last_tab = str(data.get("last_tab_key") or TAB_KEY_HOME).strip() or TAB_KEY_HOME
        ff = data.get("feature_flags") or {}
        if not isinstance(ff, Mapping):
            ff = {}

        nav_tail = data.get("nav_history_tail") or []
        if not isinstance(nav_tail, list):
            nav_tail = []

        notes_val = data.get("notes")
        notes = str(notes_val) if notes_val is not None else None

        tags_raw = data.get("tags") or []
        tags: List[str] = []
        if isinstance(tags_raw, Sequence) and not isinstance(tags_raw, (str, bytes)):
            for t in tags_raw:
                s = str(t).strip()
                if s:
                    tags.append(s)

        if not name or not ts_utc:
            return None

        return SavedDashboardView(
            name=name,
            ts_utc=ts_utc,
            env=env,
            profile=profile,
            last_tab_key=last_tab,
            feature_flags=dict(ff),
            nav_history_tail=list(nav_tail),
            notes=notes,
            tags=tags,
        )
    except Exception:  # pragma: no cover
        return None


def list_saved_views() -> List[SavedDashboardView]:
    """
    מחזיר רשימת SavedDashboardView מה-Session.

    מתעלם מרשומות לא תקינות (שלא מצליחים לפרש).
    """
    raw_list = _ensure_saved_views_list()
    views: List[SavedDashboardView] = []
    for item in raw_list:
        if isinstance(item, Mapping):
            v = _saved_view_from_mapping(item)
            if v is not None:
                views.append(v)
    return views


def add_saved_view_from_runtime(
    runtime: DashboardRuntime,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
) -> SavedDashboardView:
    """
    יוצר SavedDashboardView חדש מתוך Runtime + Session ומוסיף אותו ל-Session.

    אם name לא סופק → ניצור שם אוטומטי:
        view_{N+1} לפי מספר ה-Views הקיימים.

    שדות שנמשכים:
    --------------
    - env/profile/run_id מתוך runtime.
    - last_tab_key מתוך SESSION_KEY_LAST_TAB_KEY.
    - feature_flags subset (env/profile/live_actions/experiment_mode).
    - nav_history_tail (עד 20 רשומות אחרונות).
    """
    views_list = _ensure_saved_views_list()
    existing_views = list_saved_views()
    idx = len(existing_views) + 1

    if not name:
        name = f"view_{idx}"

    # last tab key
    last_tab = get_last_active_tab_key(default=TAB_KEY_HOME)

    # feature flags subset
    ff = runtime.feature_flags
    ff_subset = {
        "env": ff.get("env"),
        "profile": ff.get("profile"),
        "enable_live_trading_actions": ff.get("enable_live_trading_actions"),
        "enable_experiment_mode": ff.get("enable_experiment_mode"),
        "desktop_integration": ff.get("desktop_integration"),
    }

    # nav history tail
    nav_tail = _collect_session_nav_history_tail(limit=20)

    # tags
    tags_list: List[str] = []
    if tags:
        for t in tags:
            s = str(t).strip()
            if s:
                tags_list.append(s)

    view = SavedDashboardView(
        name=name,
        ts_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        env=runtime.env,
        profile=runtime.profile,
        last_tab_key=last_tab,
        feature_flags=ff_subset,
        nav_history_tail=nav_tail,
        notes=notes,
        tags=tags_list,
    )

    views_list.append(_make_json_safe(_saved_view_to_dict(view)))
    try:
        st.session_state[SESSION_KEY_SAVED_VIEWS] = views_list
    except Exception:  # pragma: no cover
        pass

    logger.info(
        "Saved view created: name=%s, env=%s, profile=%s, last_tab=%s",
        view.name,
        view.env,
        view.profile,
        view.last_tab_key,
    )

    return view


def find_saved_view_by_name(name: str) -> Optional[SavedDashboardView]:
    """
    מחפש Saved View לפי שם (case-sensitive).

    אם יש מספר Views עם אותו שם – מחזיר את הראשון שנמצא.
    """
    name = name.strip()
    if not name:
        return None
    for v in list_saved_views():
        if v.name == name:
            return v
    return None


def apply_saved_view(runtime: DashboardRuntime, view: SavedDashboardView) -> Dict[str, Any]:
    """
    מיישם Saved View על הדשבורד ברמת ניווט:

    מה אנחנו עושים עכשיו:
    -----------------------
    - לא נוגעים ב-env/profile (אלו נשלטים ע"י envvars/config/query).
    - כן נשתמש ב:
        * view.last_tab_key → ניווט לטאב המתאים.
        * feature_flags subset (למשל enable_experiment_mode) – אפשרי שימוש עתידי.
        * nav_history_tail – לצורכי Agents/Context בלבד.

    בפועל:
    -------
    - אם last_tab_key זמין ומופעל בפרופיל הנוכחי → set_nav_target(last_tab_key, {}).
    - אם לא זמין → לא נעשה nav_target אבל נחזיר reason.
    """
    tab_key = view.last_tab_key or TAB_KEY_HOME
    meta = runtime.tab_registry.get(tab_key)

    if meta is None or not _is_tab_enabled_for_profile(meta, runtime.profile):
        info = {
            "status": "warning",
            "reason": "tab_not_available_for_profile",
            "tab_key": tab_key,
            "profile": runtime.profile,
        }
        logger.warning(
            "Cannot apply saved view '%s': tab_key=%s not available for profile=%s",
            view.name,
            tab_key,
            runtime.profile,
        )
        return info

    set_nav_target(tab_key, {})  # payload ריק – הטאב יקרא מהקונטקסט הקיים
    logger.info(
        "Applying saved view '%s' (env=%s/profile=%s → tab=%s)",
        view.name,
        view.env,
        view.profile,
        tab_key,
    )

    return {
        "status": "ok",
        "applied_tab": tab_key,
        "view_name": view.name,
    }


def export_saved_views_for_agents(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    מייצא Saved Views בפורמט נוח לסוכנים / Agents Tab / Desktop:

    הפלט:
    -----
    {
        "views": [
            {
                "name": "...",
                "ts_utc": "...",
                "env": "...",
                "profile": "...",
                "last_tab_key": "...",
                "tags": [...],
            },
            ...
        ]
    }

    לא כולל את כל nav_history_tail/feature_flags במפורש – זה יכול להיות כבד.
    Agents יכולים תמיד למשוך את המידע המלא דרך list_saved_views אם הם רצים בפייתון.
    """
    views = list_saved_views()
    out = [
        {
            "name": v.name,
            "ts_utc": v.ts_utc,
            "env": v.env,
            "profile": v.profile,
            "last_tab_key": v.last_tab_key,
            "tags": list(v.tags),
        }
        for v in views
    ]
    return {"views": out}


# אופציונלי: Persist ל-SqlStore (Best-effort בלבד)
def persist_saved_views_to_sql_store(runtime: DashboardRuntime) -> Tuple[bool, Optional[str]]:
    """
    מנסה לשמור Saved Views ל-SqlStore (אם קיים):

    מחפש פונקציות ב-SqlStore:
        - save_dashboard_views(views_json)
        - save_json("dashboard_views", key, views_json)

    מחזיר:
        (success: bool, method_name_used: Optional[str])
    """
    caps = runtime.capabilities
    if not caps.get("sql_store", False):
        return False, None

    services_map = _discover_services_mapping(runtime.app_ctx)
    has_store, store_obj = _probe_service(
        runtime.app_ctx,
        services_map,
        candidates=("sql_store", "store", "db", "sql"),
    )
    if not (has_store and store_obj is not None):
        return False, None

    views = list_saved_views()
    views_dicts = [_make_json_safe(_saved_view_to_dict(v)) for v in views]

    method_candidates: Sequence[str] = (
        "save_dashboard_views",
        "save_json",
    )

    for name in method_candidates:
        fn = getattr(store_obj, name, None)
        if not callable(fn):
            continue
        try:
            if name == "save_json":
                fn("dashboard_views", "default", views_dicts)  # type: ignore[misc]
            else:
                fn(views_dicts)  # type: ignore[misc]
            logger.info(
                "Saved %d dashboard views to SqlStore via %s",
                len(views_dicts),
                name,
            )
            return True, name
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "SqlStore.%s failed to save dashboard views: %s", name, exc
            )

    return False, None


# עדכון __all__ עבור חלק 29
try:
    __all__ += [
        "SESSION_KEY_SAVED_VIEWS",
        "SavedDashboardView",
        "list_saved_views",
        "add_saved_view_from_runtime",
        "find_saved_view_by_name",
        "apply_saved_view",
        "export_saved_views_for_agents",
        "persist_saved_views_to_sql_store",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_SAVED_VIEWS",
        "SavedDashboardView",
        "list_saved_views",
        "add_saved_view_from_runtime",
        "find_saved_view_by_name",
        "apply_saved_view",
        "export_saved_views_for_agents",
        "persist_saved_views_to_sql_store",
    ]
# =====================
