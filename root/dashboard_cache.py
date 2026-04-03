# -*- coding: utf-8 -*-
"""
root/dashboard_cache.py — Quick Actions, Snapshot & Session Cache
===================================================================

Extracted from dashboard.py Part 11/35.

Contains:
- In-session cache helpers (_cache_get, _cache_set, _cache_clear_ns)
- Dashboard snapshot (trigger_dashboard_snapshot)
- Quick action helpers
- Session-level telemetry collectors (tab timings, errors, nav history)
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# Import deps
try:
    from root.dashboard import (
        _make_json_safe,
        _safe_getattr,
        _discover_services_mapping,
        _probe_service,
        ensure_dashboard_runtime,
        get_app_context,
        SESSION_KEY_NAV_HISTORY,
        SESSION_KEY_LAST_TAB_KEY,
        APP_NAME,
        APP_VERSION,
    )
except ImportError:
    _make_json_safe = lambda x: x
    _safe_getattr = getattr
    _discover_services_mapping = lambda ctx: {}
    _probe_service = lambda ctx, m, candidates=(): (False, None)
    ensure_dashboard_runtime = lambda ctx: None
    get_app_context = lambda: None
    SESSION_KEY_NAV_HISTORY = "dashboard_nav_history"
    SESSION_KEY_LAST_TAB_KEY = "dashboard_last_tab_key"
    APP_NAME = "PairsTrading"
    APP_VERSION = "?"

# Forward refs
DashboardRuntime = Any
FeatureFlags = Dict[str, Any]
AppContext = Any
TabKey = str

# Part 11/35 – Quick actions, snapshot & lightweight in-session cache
# =====================

SESSION_KEY_CACHE_ROOT: str = "dashboard_cache_v1"
SESSION_KEY_LAST_CACHE_CLEAR: str = "dashboard_last_cache_clear"
SESSION_KEY_SNAPSHOT_COUNTER: str = "dashboard_snapshot_counter"
SESSION_KEY_LAST_SNAPSHOT_META: str = "dashboard_last_snapshot_meta"


# -------------------------------------------------------------------
# 11.1 – Generic in-session cache layer (namespaced, TTL-based)
# -------------------------------------------------------------------

def _cache_now_ts() -> float:
    """
    מחזיר timestamp נוכחי ב-UTC (שניות) עבור שכבת ה-Cache הפנימית.

    שימושים:
    --------
    - Cache של services_status / health / overview / dev_diag / home_ctx.
    - כל ה-Cache נשמר תחת SESSION_KEY_CACHE_ROOT בתוך session_state.
    """
    return datetime.now(timezone.utc).timestamp()


def _cache_get(namespace: str, key: str) -> Any:
    """
    קריאת Cache כללית (per-session) לפי namespace + key.

    פרוטוקול:
    ---------
    * namespace – קבוצה לוגית:
        "services_status", "health", "overview_metrics",
        "dev_diag", "home_ctx" וכו'.
    * key – מפתח ייחודי, בדרך כלל:
        f"{env}|{profile}|{run_id}"

    מחזיר:
        value אם קיים ועדיין לא פג תוקף, אחרת None.
    """
    try:
        store = st.session_state.get(SESSION_KEY_CACHE_ROOT)
    except Exception:
        return None

    if not isinstance(store, dict):
        return None

    ns_store = store.get(namespace)
    if not isinstance(ns_store, dict):
        return None

    rec = ns_store.get(key)
    if not isinstance(rec, dict):
        return None

    ts = rec.get("ts")
    ttl = rec.get("ttl")

    if not isinstance(ts, (int, float)) or not isinstance(ttl, (int, float)):
        return None

    if _cache_now_ts() - float(ts) > float(ttl):
        # פג תוקף
        return None

    return rec.get("value")


def _cache_set(namespace: str, key: str, value: Any, ttl: float) -> None:
    """
    כתיבת Cache כללית (per-session):

    - namespace: קבוצה לוגית (services_status / health / overview / dev_diag / home_ctx).
    - key: מפתח לוגי (למשל "env|profile|run_id").
    - value: האובייקט לשמירה (dict / dataclass / כל אובייקט אחר).
    - ttl: זמן חיים בשניות.

    הערה:
    -----
    אם session_state לא זמין/לא תקין – הפונקציה בולעת את השגיאה ולא
    מפילה את הדשבורד (best-effort בלבד).
    """
    try:
        store = st.session_state.get(SESSION_KEY_CACHE_ROOT)
    except Exception:
        return

    if not isinstance(store, dict):
        store = {}

    ns_store = store.get(namespace)
    if not isinstance(ns_store, dict):
        ns_store = {}

    ns_store[key] = {
        "value": value,
        "ts": _cache_now_ts(),
        "ttl": float(ttl),
    }
    store[namespace] = ns_store

    try:
        st.session_state[SESSION_KEY_CACHE_ROOT] = store
    except Exception:
        # לא נכשיל את הדשבורד בגלל Cache פנימי
        pass


def _cache_clear(namespace: Optional[str] = None) -> None:
    """
    מנקה את Cache הדשבורד:

    - אם namespace=None → מנקה את כל ה-Cache.
    - אם namespace מוגדר → מנקה רק את ה-namespace הזה.
    """
    try:
        store = st.session_state.get(SESSION_KEY_CACHE_ROOT)
    except Exception:
        return

    if not isinstance(store, dict):
        return

    if namespace is None:
        try:
            st.session_state[SESSION_KEY_CACHE_ROOT] = {}
        except Exception:
            pass
    else:
        if namespace in store:
            del store[namespace]
        try:
            st.session_state[SESSION_KEY_CACHE_ROOT] = store
        except Exception:
            pass


# -------------------------------------------------------------------
# 11.2 – Snapshot bookkeeping (sequence, last meta, cache clear times)
# -------------------------------------------------------------------

def _bump_snapshot_counter() -> int:
    """
    מגדיל counter של Snapshots ברמת Session ומחזיר את הערך החדש.

    משמש ל:
    -------
    - תיוג snapshot עם מספר ריצה (sequence).
    - נוחות בדיבאג / SqlStore / Agents.
    """
    try:
        state = st.session_state
    except Exception:  # pragma: no cover
        return 1

    val = state.get(SESSION_KEY_SNAPSHOT_COUNTER, 0)
    try:
        current = int(val)
    except Exception:
        current = 0
    current += 1
    state[SESSION_KEY_SNAPSHOT_COUNTER] = current
    return current


def _get_snapshot_counter() -> int:
    """
    מחזיר את מספר ה-snapshot האחרון (אם קיים), אחרת 0.
    """
    try:
        val = st.session_state.get(SESSION_KEY_SNAPSHOT_COUNTER, 0)
    except Exception:  # pragma: no cover
        return 0

    try:
        return int(val)
    except Exception:
        return 0


def _record_cache_clear_ts() -> None:
    """
    רושם timestamp של ניקוי Cache אחרון (Quick action / פונקציה פנימית).

    נשמר תחת SESSION_KEY_LAST_CACHE_CLEAR ומוצג ב-Quick actions.
    """
    try:
        st.session_state[SESSION_KEY_LAST_CACHE_CLEAR] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
    except Exception:  # pragma: no cover
        pass


def _get_last_cache_clear_ts() -> Optional[str]:
    """
    מחזיר את זמן ניקוי ה-Cache האחרון (ISO-UTC), אם קיים.
    """
    try:
        val = st.session_state.get(SESSION_KEY_LAST_CACHE_CLEAR)
    except Exception:  # pragma: no cover
        return None

    if not isinstance(val, str):
        return None
    return val or None


def _store_last_snapshot_meta(snapshot: Dict[str, Any], saved: bool, method_name: Optional[str]) -> None:
    """
    שומר Meta קל של ה-Snapshot האחרון:

        {
            "ts_utc": ...,
            "env": ...,
            "profile": ...,
            "sequence": ...,
            "saved_to_sql_store": bool,
            "sql_method": "...",
        }

    זה מאפשר להציג מידע ידידותי ב-Quick actions בלי לקרוא את כל ה-snapshot.
    """
    meta = {
        "ts_utc": snapshot.get("ts_utc"),
        "env": snapshot.get("env"),
        "profile": snapshot.get("profile"),
        "sequence": snapshot.get("sequence"),
        "saved_to_sql_store": bool(saved),
        "sql_method": method_name,
    }
    try:
        st.session_state[SESSION_KEY_LAST_SNAPSHOT_META] = _make_json_safe(meta)
    except Exception:  # pragma: no cover
        pass


def _get_last_snapshot_meta() -> Optional[Dict[str, Any]]:
    """
    מחזיר Meta של ה-Snapshot האחרון אם קיים.
    """
    try:
        obj = st.session_state.get(SESSION_KEY_LAST_SNAPSHOT_META)
    except Exception:  # pragma: no cover
        return None

    if not isinstance(obj, Mapping):
        return None
    return dict(obj)


# -------------------------------------------------------------------
# 11.3 – Snapshot serialization & SqlStore integration
# -------------------------------------------------------------------

def _serialize_base_ctx_for_snapshot(base_ctx: Any) -> Any:
    """
    ממיר את base_dashboard_context למבנה snapshot ידידותי ל-SQL/JSON.

    הסדר:
    -----
    1. אם יש method בשם model_dump() (Pydantic v2) – להשתמש בו.
    2. אם יש method בשם dict() (Pydantic v1 / dataclass) – להשתמש בו.
    3. אם זה Mapping – להעתיק ל-dict רגיל.
    4. אחרת – להחזיר repr(...) כמחרוזת.
    """
    if base_ctx is None:
        return None

    # Pydantic v2 / dataclasses וכדו'
    try:
        model_dump = getattr(base_ctx, "model_dump", None)
        if callable(model_dump):
            return model_dump()
    except Exception:
        pass

    # Pydantic v1 / dataclasses
    try:
        as_dict = getattr(base_ctx, "dict", None)
        if callable(as_dict):
            return as_dict()
    except Exception:
        pass

    # Mapping
    try:
        if isinstance(base_ctx, Mapping):
            return dict(base_ctx)
    except Exception:
        pass

    # fallback – repr בלבד
    try:
        return {"__repr__": repr(base_ctx)}
    except Exception:
        return {"__repr__": "<unserializable base_ctx>"}


def _build_dashboard_snapshot(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    services_status: Dict[str, ServiceStatus],
    reason: str = "manual",
) -> Dict[str, Any]:
    """
    בונה snapshot מרוכז של מצב הדשבורד:

    כולל:
    -----
    * ts_utc
    * env/profile/run_id
    * sequence (מזהה ריצה של snapshot)
    * app/service statuses
    * base_dashboard_context סריאלי
    * user/host/app_name/version
    * reason – סיבת יצירת snapshot (manual / agent / auto / quick_action)
    * last_tab_key – הטאב האחרון שהיה פעיל (אם קיים)

    המטרה:
    -------
    - שמירה ל-SqlStore לצורכי Audit / ניטור / Backfill.
    - שמירה ב-session_state["dashboard_last_snapshot"] לשימוש ב-UI/Agents.
    """
    env: EnvName = feature_flags.get("env", DEFAULT_ENV)  # type: ignore[assignment]
    profile: ProfileName = feature_flags.get("profile", DEFAULT_PROFILE)  # type: ignore[assignment]
    run_id = get_session_run_id()

    ts_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sequence = _bump_snapshot_counter()

    base_ctx = st.session_state.get(SESSION_KEY_BASE_CTX)
    base_ctx_serialized = _make_json_safe(_serialize_base_ctx_for_snapshot(base_ctx))

    last_tab_key = get_last_active_tab_key(default="home")

    snapshot: Dict[str, Any] = {
        "ts_utc": ts_utc,
        "env": env,
        "profile": profile,
        "run_id": run_id,
        "sequence": sequence,
        "reason": str(reason),
        "app": services_status.get("app", {}),
        "services": {
            key: val
            for key, val in services_status.items()
            if key != "app"
        },
        "base_context": base_ctx_serialized,
        "user": feature_flags.get("user", RUNTIME_USER),
        "host": feature_flags.get("host", RUNTIME_HOST),
        "app_name": feature_flags.get("app_name", APP_NAME),
        "version": feature_flags.get("version", APP_VERSION),
        "last_tab_key": last_tab_key,
    }

    return snapshot


def _save_snapshot_to_sql_store(
    app_ctx: "AppContext",
    snapshot: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    מנסה לשמור snapshot ב-SqlStore, אם קיים.

    אסטרטגיה:
    ---------
    1. מגלה SqlStore מתוך AppContext (sql_store / store / db / sql).
    2. מנסה אחת מכמה פונקציות:
        - save_dashboard_snapshot(snapshot)
        - save_snapshot(snapshot)
        - save_context_snapshot("dashboard", snapshot)
        - save_context("dashboard", snapshot)
    3. במקרה של TypeError – מנסה וריאציה אחרת.

    מחזיר:
        (success: bool, method_name_used: Optional[str])
    """
    services_map = _discover_services_mapping(app_ctx)
    has_store, store_obj = _probe_service(
        app_ctx,
        services_map,
        candidates=("sql_store", "store", "db", "sql"),
    )

    if not (has_store and store_obj is not None):
        logger.warning("No SqlStore available for snapshot; skipping persist.")
        return False, None

    methods_order: Sequence[str] = (
        "save_dashboard_snapshot",
        "save_snapshot",
        "save_context_snapshot",
        "save_context",
    )

    for name in methods_order:
        func = getattr(store_obj, name, None)
        if not callable(func):
            continue

        # 1) ניסיון: func(snapshot)
        try:
            func(snapshot)
            logger.info("Dashboard snapshot saved via SqlStore.%s(snapshot)", name)
            return True, name
        except TypeError:
            # 2) ניסיון: func('dashboard', snapshot)
            try:
                func("dashboard", snapshot)
                logger.info(
                    "Dashboard snapshot saved via SqlStore.%s('dashboard', snapshot)",
                    name,
                )
                return True, name
            except TypeError:
                continue
            except Exception as exc:
                logger.warning(
                    "SqlStore.%s('dashboard', snapshot) raised %s", name, exc
                )
        except Exception as exc:
            logger.warning("SqlStore.%s(snapshot) raised %s", name, exc)

    logger.warning(
        "No compatible save_* method found on SqlStore for dashboard snapshot."
    )
    return False, None


def trigger_dashboard_snapshot(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    services_status: Dict[str, ServiceStatus],
    reason: str = "manual",
) -> Tuple[bool, Optional[str]]:
    """
    מפעיל Snapshot מלא של הדשבורד:

    Flow:
    -----
    1. בונה snapshot מרוכז (dict) כולל sequence+reason+last_tab_key.
    2. שומר אותו ב-session_state["dashboard_last_snapshot"].
    3. מנסה לשמור אותו ב-SqlStore (אם קיים).
    4. שומר Meta קל ב-SESSION_KEY_LAST_SNAPSHOT_META.
    5. מחזיר (success, method_name) עבור SqlStore.

    הערה:
    -----
    * גם אם SqlStore לא זמין – עדיין נשמור את snapshot ב-session_state,
      כך שניתן יהיה להשתמש בו ב-UI או בלוגים.
    * reason מאפשר להבדיל בין Snapshot שנוצר מ-Quick action / Agent / Auto.
    """
    snapshot = _build_dashboard_snapshot(app_ctx, feature_flags, services_status, reason=reason)

    try:
        st.session_state[SESSION_KEY_LAST_SNAPSHOT] = snapshot
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to store dashboard_last_snapshot in session_state: %s", exc
        )

    success, method_name = _save_snapshot_to_sql_store(app_ctx, snapshot)
    _store_last_snapshot_meta(snapshot, success, method_name)

    if success:
        logger.info("Dashboard snapshot persisted via %s", method_name)
    else:
        logger.info("Dashboard snapshot stored in session_state only (no SqlStore).")

    return success, method_name


# -------------------------------------------------------------------
# 11.4 – Cache & snapshot maintenance helpers
# -------------------------------------------------------------------

def _clear_streamlit_caches() -> None:
    """
    מנקה Cacheים של Streamlit (cache_data/cache_resource) + ה-Cache הפנימי של הדשבורד.

    לא נניח שכל אחד מהם קיים / עובד – במידת הצורך בולעים Exceptions.
    """
    # cache_data
    try:
        st.cache_data.clear()
        logger.info("Streamlit cache_data cleared.")
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to clear cache_data: %s", exc)

    # cache_resource
    try:
        st.cache_resource.clear()
        logger.info("Streamlit cache_resource cleared.")
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to clear cache_resource: %s", exc)

    # internal dashboard cache
    try:
        _cache_clear()
        _record_cache_clear_ts()
        logger.info("Dashboard internal cache cleared.")
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to clear internal dashboard cache: %s", exc)


# -------------------------------------------------------------------
# 11.5 – Quick actions (sidebar controls) – HF-grade UX
# -------------------------------------------------------------------

def _render_quick_actions(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    services_status: Dict[str, ServiceStatus],
) -> None:
    """
    מציג סט Quick Actions גלובליים בסיידבר:

    - 🔁 רענן דשבורד (st.rerun)
    - 🧹 נקה Cache (cache_data/cache_resource + Cache פנימי)
    - 💾 Snapshot now (שמירה ל-SqlStore אם קיים)
    - סטטוס אחרון של Snapshot ו-Caches

    כל פעולה מדווחת ל-Logger ומציגה feedback קצר ב-UI.
    """
    st.sidebar.markdown("#### ⚡ Quick actions")

    col_refresh, col_cache = st.sidebar.columns(2)
    with col_refresh:
        if st.button("🔁 רענן דשבורד", key="btn_dashboard_rerun"):
            logger.info("User requested dashboard rerun via Quick actions.")
            st.rerun()

    with col_cache:
        if st.button("🧹 נקה Cache", key="btn_dashboard_clear_cache"):
            logger.info("User requested cache clear via Quick actions.")
            _clear_streamlit_caches()
            st.success("Cache נוקה בהצלחה (cache_data + cache_resource + internal).")

    # Snapshot – נציג רק אם יש SqlStore או אם רוצים Debug mode
    caps: Mapping[str, Any] = feature_flags.get("capabilities", {}) or {}
    has_sql = bool(caps.get("sql_store", False))
    show_debug_snapshot = bool(feature_flags.get("show_debug_info", False))

    if has_sql or show_debug_snapshot:
        if st.button("💾 Snapshot now", key="btn_dashboard_snapshot_now"):
            logger.info("User requested dashboard snapshot via Quick actions.")
            success, method_name = trigger_dashboard_snapshot(
                app_ctx, feature_flags, services_status, reason="quick_action"
            )
            if success:
                st.success(
                    f"Snapshot נשמר בהצלחה (method={method_name or 'unknown'})."
                )
            else:
                st.warning(
                    "Snapshot נשמר רק ב-session_state "
                    "(SqlStore לא זמין או לא תומך בפורמט הזה)."
                )

    # מידע משלים: snapshot/cache אחרונים – מציגים בצורה קומפקטית
    last_meta = _get_last_snapshot_meta()
    last_cache_clear = _get_last_cache_clear_ts()

    with st.sidebar.expander("🧪 Snapshot & cache status", expanded=False):
        if last_meta:
            seq = last_meta.get("sequence")
            ts = last_meta.get("ts_utc")
            saved = last_meta.get("saved_to_sql_store")
            method_name = last_meta.get("sql_method")
            st.markdown(
                f"- Last snapshot: `{ts}`  •  seq=`{seq}`  \n"
                f"- Persisted: {'✅ SqlStore' if saved else '⭕ session only'}"
                + (f"  •  method=`{method_name}`" if method_name else "")
            )
        else:
            st.caption("No dashboard snapshot has been taken yet.")

        if last_cache_clear:
            st.markdown(f"- Last cache clear: `{last_cache_clear}`")
        else:
            st.caption("Cache was not explicitly cleared in this session yet.")

    # --- Sidebar: Live session telemetry mini-panel ---
    with st.sidebar.expander("📈 Session telemetry", expanded=False):
        try:
            _timings_sb = _collect_session_tab_timings()
            _errors_sb = _collect_session_tab_errors()
            _up_sb = (datetime.now(timezone.utc) - STARTED_AT_UTC).total_seconds()
            _up_min_sb = int(_up_sb // 60)
            st.caption(f"⏱ Session uptime: **{_up_min_sb}m**")
            st.caption(f"📂 Tabs rendered: **{sum(v.get('count', 0) for v in _timings_sb.values())}**")
            st.caption(f"🐛 Tabs with errors: **{len(_errors_sb)}**")
            if _timings_sb:
                _slowest_sb = max(_timings_sb.items(), key=lambda kv: kv[1].get("avg", 0.0))
                st.caption(
                    f"🐢 Slowest tab: **{_slowest_sb[0]}** "
                    f"({_slowest_sb[1].get('avg', 0.0):.2f}s avg)"
                )
        except Exception as _tsb_e:
            st.caption(f"Telemetry unavailable: {_tsb_e}")

    # --- Sidebar: Active alerts summary ---
    with st.sidebar.expander("🔔 Active alerts", expanded=False):
        try:
            _sidebar_alerts = get_dashboard_alerts(limit=5)
            if not _sidebar_alerts:
                st.caption("No active alerts.")
            else:
                for _sa in _sidebar_alerts:
                    _sa_icon = {"error": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(
                        _sa.level.lower(), "🔔"
                    )
                    st.caption(f"{_sa_icon} **{_sa.level.upper()}** — {_sa.message[:80]}")
                if len(_sidebar_alerts) == 5:
                    st.caption("… (showing first 5 — see Agents tab for full list)")
        except Exception as _alert_sb_e:
            st.caption(f"Alerts unavailable: {_alert_sb_e}")


# עדכון __all__ עבור חלק 11
try:
    __all__ += [
        "SESSION_KEY_CACHE_ROOT",
        "SESSION_KEY_LAST_CACHE_CLEAR",
        "SESSION_KEY_SNAPSHOT_COUNTER",
        "SESSION_KEY_LAST_SNAPSHOT_META",
        "_cache_get",
        "_cache_set",
        "_cache_clear",
        "_bump_snapshot_counter",
        "_get_snapshot_counter",
        "_record_cache_clear_ts",
        "_get_last_cache_clear_ts",
        "_store_last_snapshot_meta",
        "_get_last_snapshot_meta",
        "_serialize_base_ctx_for_snapshot",
        "_build_dashboard_snapshot",
        "_save_snapshot_to_sql_store",
        "trigger_dashboard_snapshot",
        "_clear_streamlit_caches",
        "_render_quick_actions",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_CACHE_ROOT",
        "SESSION_KEY_LAST_CACHE_CLEAR",
        "SESSION_KEY_SNAPSHOT_COUNTER",
        "SESSION_KEY_LAST_SNAPSHOT_META",
        "_cache_get",
        "_cache_set",
        "_cache_clear",
        "_bump_snapshot_counter",
        "_get_snapshot_counter",
        "_record_cache_clear_ts",
        "_get_last_cache_clear_ts",
        "_store_last_snapshot_meta",
        "_get_last_snapshot_meta",
        "_serialize_base_ctx_for_snapshot",
        "_build_dashboard_snapshot",
        "_save_snapshot_to_sql_store",
        "trigger_dashboard_snapshot",
        "_clear_streamlit_caches",
        "_render_quick_actions",
    ]

# =====================
