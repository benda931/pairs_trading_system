# -*- coding: utf-8 -*-
"""
root/dashboard_preferences.py — User Preferences & Shell Wrapping
===================================================================

Extracted from dashboard.py Part 21/35.

Contains:
- UserDashboardPrefs dataclass
- Preference loading/saving (session + SqlStore)
- CSS injection for theme/layout
- _render_dashboard_shell_core() — the main shell rendering function
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# Import deps
try:
    from root.dashboard import (
        _safe_getattr,
        _make_json_safe,
        _normalize_profile,
        _discover_services_mapping,
        _probe_service,
        _extract_benchmark_from_base_ctx,
        _render_global_header,
        _render_env_mode_banner,
        _render_global_sidebar,
        _render_critical_alerts_banner,
        _render_tabs_view,
        _render_runtime_debug_panel,
        ensure_dashboard_runtime,
        SESSION_KEY_BASE_CTX,
        SESSION_KEY_NAV_HISTORY,
    )
except ImportError:
    _safe_getattr = getattr
    _make_json_safe = lambda x: x
    _normalize_profile = lambda x: x or "dev_default"
    _discover_services_mapping = lambda ctx: {}
    _probe_service = lambda ctx, m, candidates=(): (False, None)
    _extract_benchmark_from_base_ctx = lambda ctx: "SPY"
    _render_global_header = lambda *a, **kw: None
    _render_env_mode_banner = lambda *a, **kw: None
    _render_global_sidebar = lambda *a, **kw: None
    _render_critical_alerts_banner = lambda *a, **kw: None
    _render_tabs_view = lambda *a, **kw: None
    _render_runtime_debug_panel = lambda *a, **kw: None
    ensure_dashboard_runtime = lambda ctx: None
    SESSION_KEY_BASE_CTX = "dashboard_base_ctx"
    SESSION_KEY_NAV_HISTORY = "dashboard_nav_history"

# Forward refs
DashboardRuntime = Any
FeatureFlags = Dict[str, Any]
AppContext = Any
EnvName = str
ProfileName = str

# Part 21/35 – User preferences, personalization & shell wrapping (HF-grade)
# =====================

from dataclasses import asdict as _dc_asdict

SESSION_KEY_USER_PREFS: str = "dashboard_user_prefs"

@dataclass
class UserDashboardPrefs:
    """
    העדפות משתמש לדשבורד – שכבת Personalization ברמת קרן.

    הרעיון:
    --------
    לאפשר לדשבורד "להתאים את עצמו" למשתמש:
    - איזה פרופיל להעדיף (trading/research/risk/macro/monitoring).
    - איזה טאב לפתוח כברירת מחדל.
    - האם להציג Debug panels כברירת מחדל.
    - כמה היסטוריית ניווט לשמור.
    - איזה benchmark להעדיף.

    שדות:
    -----
    user_key:
        מזהה ייחודי למשתמש (user@host או מזהה מה-Config).
    preferred_profile:
        פרופיל מועדף כשאין override חיצוני (query/env/config).
    default_tab_key:
        הטאב שייפתח כברירת מחדל אם אין nav_target מפורש.
    show_debug_by_default:
        האם להפעיל show_debug_info כברירת מחדל בפרופילים dev/research.
    max_nav_history:
        כמה רשומות לשמור ב-SESSION_KEY_NAV_HISTORY (מונע התנפחות).
    preferred_benchmark:
        Benchmark מועדף (למשל "SPY", "QQQ", "EWJ") – אם קיים ב-base_context.
    dense_layout:
        האם להעדיף Layout צפוף יותר (פחות מרווחים/כותרות).
    """

    user_key: str
    preferred_profile: Optional[ProfileName] = None
    default_tab_key: Optional[TabKey] = None
    show_debug_by_default: bool = True
    max_nav_history: int = 200
    preferred_benchmark: Optional[str] = None
    dense_layout: bool = False


def _derive_user_key(runtime: DashboardRuntime) -> str:
    """
    גוזר user_key מתוך Runtime:

    לוגיקה:
    -------
    1. אם ב-AppContext.settings יש user_id → נשתמש בו.
    2. אחרת → ניקח feature_flags["user"] + ["host"] ונרכיב user@host.
    3. fallback → "anonymous".
    """
    # 1) AppContext.settings.user_id אם קיים
    settings = _safe_getattr(runtime.app_ctx, "settings")
    if settings is not None:
        uid = _safe_getattr(settings, "user_id")
        if isinstance(uid, str) and uid.strip():
            return uid.strip()

        cfg = _safe_getattr(settings, "config")
        if isinstance(cfg, Mapping):
            v = cfg.get("user_id") or cfg.get("USER_ID")
            if isinstance(v, str) and v.strip():
                return v.strip()

    # 2) מתוך feature_flags
    ff = runtime.feature_flags
    user = str(ff.get("user", RUNTIME_USER) or "").strip() or "anonymous"
    host = str(ff.get("host", RUNTIME_HOST) or "").strip() or "unknown-host"
    return f"{user}@{host}"


def _user_prefs_to_dict(prefs: UserDashboardPrefs) -> Dict[str, Any]:
    """
    ממיר UserDashboardPrefs ל-dict JSON-friendly.
    """
    return {
        "user_key": prefs.user_key,
        "preferred_profile": prefs.preferred_profile,
        "default_tab_key": prefs.default_tab_key,
        "show_debug_by_default": prefs.show_debug_by_default,
        "max_nav_history": prefs.max_nav_history,
        "preferred_benchmark": prefs.preferred_benchmark,
        "dense_layout": prefs.dense_layout,
    }


def _user_prefs_from_mapping(data: Mapping[str, Any], fallback_key: str) -> UserDashboardPrefs:
    """
    בונה UserDashboardPrefs מתוך Mapping (למשל שהגיע מ-SqlStore/JSON).
    במקרה של שדות חסרים – משתמש בערכי ברירת מחדל.
    """
    return UserDashboardPrefs(
        user_key=str(data.get("user_key") or fallback_key),
        preferred_profile=(
            _normalize_profile(str(data["preferred_profile"]))
            if data.get("preferred_profile") is not None
            else None
        ),
        default_tab_key=str(data.get("default_tab_key") or "") or None,
        show_debug_by_default=bool(
            data.get("show_debug_by_default", True)
        ),
        max_nav_history=int(data.get("max_nav_history", 200)),
        preferred_benchmark=(
            str(data.get("preferred_benchmark")).strip().upper()
            if data.get("preferred_benchmark")
            else None
        ),
        dense_layout=bool(data.get("dense_layout", False)),
    )


def _load_user_prefs_from_sql_store(
    runtime: DashboardRuntime,
    user_key: str,
) -> Optional[UserDashboardPrefs]:
    """
    מנסה לטעון העדפות משתמש מ-SqlStore (אם קיים):

    אסטרטגיה:
    ---------
    1. מגלה SqlStore מתוך AppContext (sql_store / store / db / sql).
    2. מנסה אחת מכמה פונקציות:
        - load_dashboard_prefs(user_key)
        - load_user_prefs(user_key)
        - load_json("dashboard_prefs", user_key)
    3. אם מתקבל Mapping – בונה ממנו UserDashboardPrefs.
    4. במקרה של כשל – מחזיר None בלי להפיל את הדשבורד.
    """
    caps = runtime.capabilities
    if not caps.get("sql_store", False):
        return None

    services_map = _discover_services_mapping(runtime.app_ctx)
    has_store, store_obj = _probe_service(
        runtime.app_ctx,
        services_map,
        candidates=("sql_store", "store", "db", "sql"),
    )
    if not (has_store and store_obj is not None):
        return None

    # סדר פונקציות אפשרי
    method_candidates: Sequence[str] = (
        "load_dashboard_prefs",
        "load_user_prefs",
        "load_json",
    )

    for name in method_candidates:
        func = getattr(store_obj, name, None)
        if not callable(func):
            continue

        try:
            if name == "load_json":
                raw = func("dashboard_prefs", user_key)  # type: ignore[misc]
            else:
                raw = func(user_key)  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover
            logger.debug("SqlStore.%s(%r) raised %s", name, user_key, exc)
            continue

        if isinstance(raw, Mapping):
            try:
                prefs = _user_prefs_from_mapping(raw, user_key)
                logger.info(
                    "Loaded UserDashboardPrefs for user_key=%s via SqlStore.%s",
                    user_key,
                    name,
                )
                return prefs
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to parse user prefs for user_key=%s from SqlStore.%s: %s",
                    user_key,
                    name,
                    exc,
                )
                return None

    return None


def _save_user_prefs_to_sql_store(
    runtime: DashboardRuntime,
    prefs: UserDashboardPrefs,
) -> Tuple[bool, Optional[str]]:
    """
    מנסה לשמור העדפות משתמש ב-SqlStore (אם קיים).

    אסטרטגיה:
    ---------
    1. מגלה SqlStore.
    2. מנסה פונקציות:
        - save_dashboard_prefs(user_key, data)
        - save_user_prefs(user_key, data)
        - save_json("dashboard_prefs", user_key, data)
    3. במקרה של הצלחה – מחזיר (True, method_name).
    4. במקרה של כשל – מחזיר (False, None).

    לא מפיל את הדשבורד אם אין SqlStore/שיטות מתאימות.
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

    data = _user_prefs_to_dict(prefs)
    data = _make_json_safe(data)

    method_candidates: Sequence[str] = (
        "save_dashboard_prefs",
        "save_user_prefs",
        "save_json",
    )

    for name in method_candidates:
        func = getattr(store_obj, name, None)
        if not callable(func):
            continue

        try:
            if name == "save_json":
                func("dashboard_prefs", prefs.user_key, data)  # type: ignore[misc]
            else:
                func(prefs.user_key, data)  # type: ignore[misc]
            logger.info(
                "UserDashboardPrefs saved for user_key=%s via SqlStore.%s",
                prefs.user_key,
                name,
            )
            return True, name
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "SqlStore.%s failed to save user prefs for user_key=%s: %s",
                name,
                prefs.user_key,
                exc,
            )

    return False, None


def get_or_init_user_prefs(runtime: DashboardRuntime) -> UserDashboardPrefs:
    """
    מחזיר (או מאתחל) UserDashboardPrefs עבור ה-Runtime הנוכחי:

    Flow:
    -----
    1. אם יש כבר אובייקט ב-session_state → מחזיר אותו.
    2. אחרת:
        - גוזר user_key.
        - מנסה לטעון מ-SqlStore (אם זמין).
        - אם לא נמצא – בונה Prefs חדשים על בסיס Runtime (env/profile/base_ctx).
    3. שומר את ה-Prefs ב-session_state לצורך ריצות עתידיות.
    """
    try:
        existing = st.session_state.get(SESSION_KEY_USER_PREFS)
    except Exception:
        existing = None

    if isinstance(existing, UserDashboardPrefs):
        return existing
    if isinstance(existing, Mapping):
        # תמיכה במקרה שבו שמרנו dict בלבד בעבר
        user_key = _derive_user_key(runtime)
        prefs = _user_prefs_from_mapping(existing, user_key)
        st.session_state[SESSION_KEY_USER_PREFS] = prefs
        return prefs

    user_key = _derive_user_key(runtime)

    # ניסיון טעינה מ-SqlStore
    prefs = _load_user_prefs_from_sql_store(runtime, user_key)
    if prefs is None:
        # יצירת Prefs חדשים
        base_ctx = st.session_state.get(SESSION_KEY_BASE_CTX)
        preferred_benchmark = None
        if base_ctx is not None:
            preferred_benchmark = _extract_benchmark_from_base_ctx(base_ctx)

        prefs = UserDashboardPrefs(
            user_key=user_key,
            preferred_profile=runtime.profile,
            default_tab_key=TAB_KEY_HOME,
            show_debug_by_default=True if runtime.env in ("dev", "research") else False,
            max_nav_history=200,
            preferred_benchmark=preferred_benchmark,
            dense_layout=False,
        )
        logger.info("Initialized new UserDashboardPrefs for user_key=%s", user_key)

    try:
        st.session_state[SESSION_KEY_USER_PREFS] = prefs
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to store UserDashboardPrefs in session_state: %s", exc)

    return prefs


def apply_user_prefs_to_runtime(
    runtime: DashboardRuntime,
    prefs: UserDashboardPrefs,
) -> None:
    """
    מיישמת חלק מההעדפות על ה-Runtime/FeatureFlags:

    מה אנחנו כן משנים:
    -------------------
    - show_debug_info:
        * אם prefs.show_debug_by_default=True בפרופילי dev/research → מחייבים True.
    - benchmark:
        * אם prefs.preferred_benchmark קיים – ננסה לעדכן ב-base_context.
    - nav_history limit:
        * מגביל את SESSION_KEY_NAV_HISTORY לאורך prefs.max_nav_history.

    מה אנחנו *לא* משנים (חשוב!):
    -----------------------------
    - env/profile – אלו נשלטים ע"י config/envvars/query; Prefs יכולים לשמש רק כ-Default,
      אבל לא דורסים override אקטיבי.
    - סדר טאבים / יכולות – נשלט ע"י feature_flags + TabRegistry.
    """
    ff = runtime.feature_flags

    # show_debug_info – רק אם מדובר ב-env מסוג dev/research/test
    if prefs.show_debug_by_default and runtime.env in ("dev", "research", "test"):
        ff["show_debug_info"] = True

    # preferred_benchmark → base_context
    if prefs.preferred_benchmark:
        try:
            base_ctx = st.session_state.get(SESSION_KEY_BASE_CTX)
        except Exception:
            base_ctx = None

        if base_ctx is not None:
            try:
                # attrib
                if hasattr(base_ctx, "benchmark"):
                    setattr(base_ctx, "benchmark", prefs.preferred_benchmark)
                # mapping
                if isinstance(base_ctx, MutableMapping):
                    base_ctx["benchmark"] = prefs.preferred_benchmark
                st.session_state[SESSION_KEY_BASE_CTX] = base_ctx
            except Exception:  # pragma: no cover
                logger.debug(
                    "Failed to apply preferred_benchmark=%s on base_ctx",
                    prefs.preferred_benchmark,
                )

    # קיצוץ Nav history לאורך מקסימלי
    try:
        history = st.session_state.get(SESSION_KEY_NAV_HISTORY, [])
        if isinstance(history, list) and len(history) > prefs.max_nav_history:
            st.session_state[SESSION_KEY_NAV_HISTORY] = history[-prefs.max_nav_history :]
    except Exception:  # pragma: no cover
        pass


def persist_user_prefs_if_needed(
    runtime: DashboardRuntime,
    prefs: UserDashboardPrefs,
    auto_persist: bool = True,
) -> None:
    """
    שומר Prefs ב-SqlStore במקרה שיש שינוי משמעותי.

    כרגע הפונקציה היא Hook עתידי – ניתן להרחיב ולהשוות Prefs
    קודמים לחדשים ולשמור רק כשצריך. לעת עתה, אם auto_persist=True
    וקיים SqlStore – ננסה לשמור כל כמה ריצות.

    אפשרויות הרחבה:
    ----------------
    - להוסיף "dirty flag" ב-session_state.
    - לספור כמה פעמים מריצים Shell ולשמור אחת ל-N פעמים.
    """
    if not auto_persist:
        return

    saved, method_name = _save_user_prefs_to_sql_store(runtime, prefs)
    if saved:
        logger.debug(
            "UserDashboardPrefs persisted for user_key=%s via %s",
            prefs.user_key,
            method_name or "unknown",
        )


# -------------------------
# Global CSS Theme
# -------------------------

def _inject_global_css() -> None:
    """
    Injects a professional, institutional-grade CSS theme into the Streamlit app.
    Called once per render at the top of the shell core.
    """
    st.markdown(
        """
<style>
/* ============================================================
   PAIRS TRADING DASHBOARD — PROFESSIONAL THEME
   ============================================================ */

/* --- Root variables --- */
:root {
    --pt-blue:       #1565C0;
    --pt-blue-light: #1E88E5;
    --pt-blue-pale:  #E3F2FD;
    --pt-green:      #2E7D32;
    --pt-green-pale: #E8F5E9;
    --pt-red:        #C62828;
    --pt-red-pale:   #FFEBEE;
    --pt-amber:      #F57F17;
    --pt-amber-pale: #FFF8E1;
    --pt-gray-dark:  #1A1A2E;
    --pt-gray-mid:   #37474F;
    --pt-gray-light: #ECEFF1;
    --pt-text:       #212121;
    --pt-text-muted: #546E7A;
    --pt-border:     #CFD8DC;
    --pt-radius:     8px;
    --pt-shadow:     0 2px 8px rgba(0,0,0,0.10);
    --pt-shadow-md:  0 4px 16px rgba(0,0,0,0.14);
}

/* --- Typography --- */
body, .stApp {
    font-family: 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
    color: var(--pt-text) !important;
}

h1, h2, h3, h4 {
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}

h1 { font-size: 1.75rem !important; }
h2 { font-size: 1.35rem !important; border-bottom: 2px solid var(--pt-blue); padding-bottom: 6px; margin-bottom: 16px; }
h3 { font-size: 1.10rem !important; color: var(--pt-blue) !important; }
h4 { font-size: 0.95rem !important; color: var(--pt-gray-mid) !important; text-transform: uppercase; letter-spacing: 0.5px; }

/* --- Streamlit tab navigation --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--pt-gray-light);
    border-radius: var(--pt-radius);
    padding: 4px 6px;
    border-bottom: 2px solid var(--pt-border);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px !important;
    padding: 6px 14px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--pt-text-muted) !important;
    background: transparent !important;
    border: none !important;
    transition: background 0.15s, color 0.15s;
}

.stTabs [aria-selected="true"] {
    background: white !important;
    color: var(--pt-blue) !important;
    box-shadow: var(--pt-shadow);
}

.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: rgba(21, 101, 192, 0.08) !important;
    color: var(--pt-blue) !important;
}

/* --- Metric cards --- */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid var(--pt-border);
    border-radius: var(--pt-radius);
    padding: 12px 16px !important;
    box-shadow: var(--pt-shadow);
    transition: box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    box-shadow: var(--pt-shadow-md);
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--pt-text-muted) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    color: var(--pt-text) !important;
    line-height: 1.2;
}
[data-testid="stMetricDelta"] {
    font-size: 0.80rem !important;
    font-weight: 600 !important;
}

/* --- Dataframes / tables --- */
.stDataFrame, [data-testid="stDataFrame"] {
    border-radius: var(--pt-radius) !important;
    overflow: hidden !important;
    box-shadow: var(--pt-shadow) !important;
    border: 1px solid var(--pt-border) !important;
}
[data-testid="stDataFrame"] table {
    font-size: 0.82rem !important;
}
[data-testid="stDataFrame"] thead th {
    background: var(--pt-gray-light) !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: var(--pt-text-muted) !important;
    border-bottom: 2px solid var(--pt-border) !important;
}

/* --- Sidebar --- */
[data-testid="stSidebar"] {
    background: var(--pt-gray-dark) !important;
    border-right: 1px solid #283142;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #B0BEC5 !important;
}
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #E3F2FD !important;
}
[data-testid="stSidebar"] .stSelectbox select,
[data-testid="stSidebar"] .stTextInput input {
    background: #283142 !important;
    color: #E0E0E0 !important;
    border-color: #37474F !important;
}

/* --- Buttons --- */
.stButton > button {
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 6px 18px !important;
    transition: all 0.15s !important;
    border: 1.5px solid var(--pt-blue) !important;
    color: var(--pt-blue) !important;
    background: white !important;
}
.stButton > button:hover {
    background: var(--pt-blue) !important;
    color: white !important;
    box-shadow: var(--pt-shadow-md) !important;
}
.stButton > button[kind="primary"] {
    background: var(--pt-blue) !important;
    color: white !important;
}

/* --- Input widgets --- */
.stSelectbox > div > div,
.stMultiselect > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stDateInput > div > div > input {
    border-radius: 6px !important;
    border-color: var(--pt-border) !important;
    font-size: 0.85rem !important;
}

/* --- Section dividers --- */
hr {
    border: none;
    border-top: 1.5px solid var(--pt-border);
    margin: 20px 0;
}

/* --- Alerts & notices --- */
.stAlert {
    border-radius: var(--pt-radius) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="info"] {
    background: var(--pt-blue-pale) !important;
    border-left: 4px solid var(--pt-blue) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="success"] {
    background: var(--pt-green-pale) !important;
    border-left: 4px solid var(--pt-green) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="warning"] {
    background: var(--pt-amber-pale) !important;
    border-left: 4px solid var(--pt-amber) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="error"] {
    background: var(--pt-red-pale) !important;
    border-left: 4px solid var(--pt-red) !important;
}

/* --- Expander --- */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    background: var(--pt-gray-light) !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
}

/* --- Spinner --- */
.stSpinner {
    color: var(--pt-blue) !important;
}

/* --- Caption / helper text --- */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--pt-text-muted) !important;
    font-size: 0.78rem !important;
}

/* --- Code blocks --- */
.stCode, code {
    background: #F5F7FA !important;
    border: 1px solid var(--pt-border) !important;
    border-radius: 4px !important;
    font-size: 0.80rem !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace !important;
}

/* --- Plotly charts — remove Plotly toolbar clutter --- */
.js-plotly-plot .plotly .modebar {
    opacity: 0.3 !important;
    transition: opacity 0.2s;
}
.js-plotly-plot .plotly .modebar:hover {
    opacity: 1 !important;
}

/* --- Section header helper class --- */
.pt-section-header {
    background: linear-gradient(90deg, var(--pt-blue) 0%, var(--pt-blue-light) 100%);
    color: white !important;
    padding: 8px 16px;
    border-radius: var(--pt-radius);
    font-weight: 700;
    font-size: 0.90rem;
    margin-bottom: 12px;
    letter-spacing: 0.3px;
}

/* --- KPI card helper class --- */
.pt-kpi-card {
    background: white;
    border: 1px solid var(--pt-border);
    border-radius: var(--pt-radius);
    padding: 16px;
    box-shadow: var(--pt-shadow);
    text-align: center;
}
.pt-kpi-card .pt-kpi-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--pt-text-muted);
    margin-bottom: 4px;
}
.pt-kpi-card .pt-kpi-value {
    font-size: 1.60rem;
    font-weight: 800;
    color: var(--pt-text);
    line-height: 1.1;
}
.pt-kpi-card .pt-kpi-delta {
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 4px;
}
.pt-kpi-card .pt-kpi-delta.positive { color: var(--pt-green); }
.pt-kpi-card .pt-kpi-delta.negative { color: var(--pt-red); }

/* --- Badge helper --- */
.pt-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.pt-badge-blue   { background: var(--pt-blue-pale); color: var(--pt-blue); }
.pt-badge-green  { background: var(--pt-green-pale); color: var(--pt-green); }
.pt-badge-red    { background: var(--pt-red-pale); color: var(--pt-red); }
.pt-badge-amber  { background: var(--pt-amber-pale); color: var(--pt-amber); }
.pt-badge-gray   { background: var(--pt-gray-light); color: var(--pt-gray-mid); }

/* --- Scrollbar --- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--pt-border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--pt-text-muted); }

/* --- Main content padding --- */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}
</style>
""",
        unsafe_allow_html=True,
    )


# -------------------------
# Shell wrapping & wiring
# -------------------------

def _render_dashboard_shell_core(runtime: DashboardRuntime) -> None:
    """
    Shell ליבה של הדשבורד – בלי Personalization / Agents / Telemetry.

    זה ה-"core" שהעטיפות המאוחרות משתמשות בו:
    - Header עליון
    - Sidebar גלובלי
    - Banner של התראות קריטיות
    - Tabs (router)
    - Runtime debug (אופציונלי)
    """
    # Inject professional CSS theme (runs every render; Streamlit deduplicates)
    _inject_global_css()

    app_ctx = runtime.app_ctx
    feature_flags = runtime.feature_flags
    services_status = runtime.services_status
    tab_registry = runtime.tab_registry

    # Header עליון – אפליקציה, env/profile, benchmark וכו'
    _render_global_header(app_ctx, feature_flags, services_status)

    # Env-mode banner ברור (DEV / LIVE / PAPER / BACKTEST / RESEARCH)
    _render_env_mode_banner(runtime)

    # Sidebar גלובלי – App overview, Services, Quick actions, Debug
    _render_global_sidebar(app_ctx, feature_flags, services_status)

    # Banner של התראות קריטיות (LIVE בלי broker / risk, SqlStore חסר וכו')
    _render_critical_alerts_banner(runtime)

    # Tabs – הטאבים עצמם (Home / Pair / Backtest / Insights / ...)
    _render_tabs_view(app_ctx, feature_flags, tab_registry)

    # Debug panel קטן עם runtime/capabilities/perf (רק אם show_debug_info=True)
    _render_runtime_debug_panel(runtime)



# עדכון __all__ עבור חלק 21
try:
    __all__ += [
        "SESSION_KEY_USER_PREFS",
        "UserDashboardPrefs",
        "get_or_init_user_prefs",
        "apply_user_prefs_to_runtime",
        "persist_user_prefs_if_needed",
        "_render_dashboard_shell_core",
        # render_dashboard_shell is in dashboard_diagnostics.py (Part 26)
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_USER_PREFS",
        "UserDashboardPrefs",
        "get_or_init_user_prefs",
        "apply_user_prefs_to_runtime",
        "persist_user_prefs_if_needed",
        "_render_dashboard_shell_core",
        # render_dashboard_shell is in dashboard_diagnostics.py (Part 26)
    ]
# =====================
