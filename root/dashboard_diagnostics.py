# -*- coding: utf-8 -*-
"""
root/dashboard_diagnostics.py — Developer Diagnostics
========================================================

Extracted from dashboard.py Part 26/35.

Contains developer tools: enriched shell wrapper, runtime debug panel,
diagnostic exports, and internal inspection helpers.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# Import deps
try:
    from root.dashboard import (
        _make_json_safe,
        _safe_getattr,
        ensure_dashboard_runtime,
        get_app_context,
        _discover_services_mapping,
        _probe_service,
        _collect_session_tab_timings,
        _collect_session_tab_errors,
        _collect_session_nav_history_tail,
        build_active_tabs,
        APP_NAME,
        APP_VERSION,
        RUNTIME_HOST,
        RUNTIME_USER,
        STARTED_AT_UTC,
    )
except (ImportError, AttributeError):
    _make_json_safe = lambda x: x
    _safe_getattr = getattr
    ensure_dashboard_runtime = lambda ctx: None
    get_app_context = lambda: None
    _discover_services_mapping = lambda ctx: {}
    _probe_service = lambda ctx, m, candidates=(): (False, None)
    _collect_session_tab_timings = lambda: {}
    _collect_session_tab_errors = lambda: {}
    _collect_session_nav_history_tail = lambda limit=50: []
    build_active_tabs = lambda reg, ff: []
    APP_NAME = "PairsTrading"
    APP_VERSION = "?"
    RUNTIME_HOST = "localhost"
    RUNTIME_USER = "unknown"
    STARTED_AT_UTC = datetime.now(timezone.utc)

try:
    from root.dashboard_telemetry import build_dashboard_summary, dashboard_summary_to_dict
except (ImportError, AttributeError):
    build_dashboard_summary = lambda runtime: None
    dashboard_summary_to_dict = lambda s: {}

try:
    from root.dashboard_health import compute_dashboard_health, dashboard_health_to_dict
except (ImportError, AttributeError):
    compute_dashboard_health = lambda runtime: None
    dashboard_health_to_dict = lambda h, include_summary=False: {}

# Forward refs
DashboardRuntime = Any
FeatureFlags = Dict[str, Any]
AppContext = Any

# Part 26/35 – Developer diagnostics & enriched shell wrapper (HF-grade)
# =====================

import sys
import platform

SESSION_KEY_DEV_DIAG_LAST: str = "dashboard_dev_diag_last"


@dataclass
class DeveloperDiagnostics:
    """
    Developer-level diagnostics – תמונת מצב עשירה על סביבת הריצה, המודולים והקונפיג.

    מטרות:
    -------
    - לתת ל-Quant/Dev תמונה מיידית: מה נטען, מה זמין, איזו גרסת Python/Streamlit,
      איפה PROJECT_ROOT, האם מודולי core/root/root_desktop זמינים וכו'.
    - לשלב מידע פונקציונלי מה-Health/Summary/Home:
      * health_score / severity
      * overview metrics count
      * home context last update
    - לשלב אינדיקציות Cache:
      * אילו namespaces קיימים, כמה רשומות בכל אחד.
    """

    env: EnvName
    profile: ProfileName
    app_name: str
    version: str
    host: str
    user: str

    python_version: str
    python_executable: str
    platform: str
    machine: str

    streamlit_version: str
    pandas_version: str

    project_root: str
    cwd: str
    logs_path: str
    started_at_utc: str

    core_modules: Dict[str, bool]
    root_modules: Dict[str, bool]
    desktop_modules: Dict[str, bool]

    envvars: Dict[str, Optional[str]]
    session_keys_count: int
    session_keys_sample: List[str]

    # שדות חדשים – אינטגרציה עם Health/Overview/Home
    health_severity: str
    health_score: float
    overview_metrics_count: int
    home_ctx_last_updated: Optional[str]

    # מידע על שכבת Cache (namespaces + counters)
    cache_namespaces: Dict[str, int]
    cache_last_clear_ts: Optional[str]

    warnings: List[str]


def _probe_module_presence(mod_names: Sequence[str]) -> Dict[str, bool]:
    """
    בודק זמינות של מודולים לפי importlib.util.find_spec.
    """
    out: Dict[str, bool] = {}
    for name in mod_names:
        try:
            import importlib.util as _util  # local import to avoid global side-effects
            spec = _util.find_spec(name)
            out[name] = spec is not None
        except Exception:  # pragma: no cover
            out[name] = False
    return out


def _snapshot_cache_namespaces() -> Dict[str, int]:
    """
    מחלץ תמונת מצב קלה של שכבת ה-Cache הפנימית:

        {
            "services_status": 3,
            "health": 1,
            "summary": 1,
            "overview_metrics": 1,
            "home_ctx": 1,
            ...
        }

    הערכים הם מספר רשומות לכל namespace (לא תוכן מלא).
    """
    try:
        store = st.session_state.get(SESSION_KEY_CACHE_ROOT)
    except Exception:  # pragma: no cover
        return {}

    if not isinstance(store, dict):
        return {}

    ns_counts: Dict[str, int] = {}
    for ns, ns_store in store.items():
        if isinstance(ns_store, Mapping):
            ns_counts[str(ns)] = len(ns_store)
    return ns_counts


def build_developer_diagnostics(runtime: DashboardRuntime) -> DeveloperDiagnostics:
    """
    בונה DeveloperDiagnostics מתוך Runtime + סביבת ריצה, עם שכבת Cache:

    כולל:
    - גרסת Python / Streamlit / pandas.
    - PROJECT_ROOT / cwd / logs path.
    - זמינות מודולי core/root/root_desktop חשובים.
    - envvars רלוונטיים.
    - סטטוס session_state (מס' מפתחות ודוגמאות).
    - שילוב Health/Overview/Home:
        * health_score / severity
        * מספר metrics ב-Overview
        * זמן עדכון אחרון של Home context
    - Snapshot של Cache namespaces (services_status/health/summary/home_ctx/...)
    """
    ff = runtime.feature_flags

    env = runtime.env
    profile = runtime.profile
    app_name = ff.get("app_name", APP_NAME)
    version = ff.get("version", APP_VERSION)
    host = ff.get("host", RUNTIME_HOST)
    user = ff.get("user", RUNTIME_USER)

    # ננסה Cache פנימי – איננו קפדניים כאן (אם יש, נשתמש; אם לא, נבנה חדש)
    cache_key = f"{env}|{profile}|{runtime.run_id}"
    cached = _cache_get("dev_diag", cache_key)
    if isinstance(cached, DeveloperDiagnostics):
        return cached

    # Python / platform
    python_version = sys.version.replace("\n", " ")
    python_executable = sys.executable
    platform_str = platform.platform()
    machine = platform.machine()

    # Lib versions (best-effort)
    streamlit_version = getattr(st, "__version__", "unknown")
    try:
        pandas_version = pd.__version__
    except Exception:  # pragma: no cover
        pandas_version = "unknown"

    project_root_str = str(PROJECT_ROOT)
    cwd = os.getcwd()
    logs_path_str = str(DASHBOARD_LOG_PATH)
    started_at_str = STARTED_AT_UTC.isoformat(timespec="seconds")

    # Module availability – core/root/root_desktop
    core_modules = _probe_module_presence(
        (
            "core.optimization_backtester",
            "core.optimizer",
            "core.meta_optimizer",
            "core.risk_engine",
            "core.signals_engine",
            "core.macro_engine",
            "core.sql_store",
            "core.dashboard_service",
            "common.config_manager",
        )
    )
    root_modules = _probe_module_presence(
        (
            "root.dashboard",
            "root.dashboard_home_v2",
            "root.smart_scan_tab",
            "root.pair_tab",
            "root.matrix_research_tab",
            "root.tab_comparison_matrices",
            "root.backtest",
            "root.insights",
            "root.macro_tab",
            "root.portfolio_tab",
            "root.fair_value_api_tab",
            "root.agents_tab",
            "root.config_tab",
        )
    )
    desktop_modules = _probe_module_presence(
        (
            "root_desktop.app",
            "root_desktop.views.main_window",
        )
    )

    # Envvars רלוונטיים
    envvar_keys = (
        "PAIRS_ENV",
        "PAIRSTRADING_ENV",
        "APP_ENV",
        "ENV",
        "PAIRS_PROFILE",
        "PAIRSTRADING_PROFILE",
        "APP_PROFILE",
        "PROFILE",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    )
    envvars: Dict[str, Optional[str]] = {}
    for k in envvar_keys:
        try:
            envvars[k] = os.getenv(k)
        except Exception:  # pragma: no cover
            envvars[k] = None

    # session_state snapshot
    try:
        ss = st.session_state
        session_keys = list(ss.keys())
        session_keys_count = len(session_keys)
        session_keys_sample = sorted(session_keys)[:25]
    except Exception:  # pragma: no cover
        session_keys_count = 0
        session_keys_sample = []

    # Health/Overview/Home integration
    health = compute_dashboard_health(runtime)
    overview_dicts = get_dashboard_overview_from_session() or []
    home_ctx_dict = get_dashboard_home_context_from_session() or {}
    home_ctx_last_updated = home_ctx_dict.get("created_at_utc")

    # Cache namespaces snapshot
    cache_namespaces = _snapshot_cache_namespaces()
    cache_last_clear_ts = _get_last_cache_clear_ts()

    # warnings – ננסה להפיק כמה תובנות בסיסיות
    warnings: List[str] = []
    if not core_modules.get("core.sql_store", False):
        warnings.append("core.sql_store module not found – SqlStore integration may be limited.")
    if not root_modules.get("root.dashboard_home_v2", False):
        warnings.append("root.dashboard_home_v2 module not found – Home tab may be degraded.")
    if not desktop_modules.get("root_desktop.app", False):
        warnings.append("root_desktop.app module not found – Desktop app integration may be missing.")
    if health.has_critical_issues:
        warnings.append("DashboardHealth reports critical issues – see system_health overview & Logs tab.")
    if not cache_namespaces:
        warnings.append("Internal dashboard cache appears empty – either just cleared or not used yet.")

    diag = DeveloperDiagnostics(
        env=env,
        profile=profile,
        app_name=app_name,
        version=version,
        host=host,
        user=user,
        python_version=python_version,
        python_executable=python_executable,
        platform=platform_str,
        machine=machine,
        streamlit_version=streamlit_version,
        pandas_version=pandas_version,
        project_root=project_root_str,
        cwd=cwd,
        logs_path=logs_path_str,
        started_at_utc=started_at_str,
        core_modules=core_modules,
        root_modules=root_modules,
        desktop_modules=desktop_modules,
        envvars=envvars,
        session_keys_count=session_keys_count,
        session_keys_sample=session_keys_sample,
        health_severity=health.severity,
        health_score=health.score,
        overview_metrics_count=len(overview_dicts),
        home_ctx_last_updated=home_ctx_last_updated,
        cache_namespaces=cache_namespaces,
        cache_last_clear_ts=cache_last_clear_ts,
        warnings=warnings,
    )

    _cache_set("dev_diag", cache_key, diag, ttl=20.0)
    return diag


def developer_diagnostics_to_dict(diag: DeveloperDiagnostics) -> Dict[str, Any]:
    """
    ממיר DeveloperDiagnostics ל-dict JSON-friendly (ללא אובייקטים כבדים).
    """
    return {
        "env": diag.env,
        "profile": diag.profile,
        "app_name": diag.app_name,
        "version": diag.version,
        "host": diag.host,
        "user": diag.user,
        "python_version": diag.python_version,
        "python_executable": diag.python_executable,
        "platform": diag.platform,
        "machine": diag.machine,
        "streamlit_version": diag.streamlit_version,
        "pandas_version": diag.pandas_version,
        "project_root": diag.project_root,
        "cwd": diag.cwd,
        "logs_path": diag.logs_path,
        "started_at_utc": diag.started_at_utc,
        "core_modules": diag.core_modules,
        "root_modules": diag.root_modules,
        "desktop_modules": diag.desktop_modules,
        "envvars": diag.envvars,
        "session_keys_count": diag.session_keys_count,
        "session_keys_sample": diag.session_keys_sample,
        "health_severity": diag.health_severity,
        "health_score": diag.health_score,
        "overview_metrics_count": diag.overview_metrics_count,
        "home_ctx_last_updated": diag.home_ctx_last_updated,
        "cache_namespaces": diag.cache_namespaces,
        "cache_last_clear_ts": diag.cache_last_clear_ts,
        "warnings": diag.warnings,
    }


def update_developer_diagnostics_in_session(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    מחשב DeveloperDiagnostics, שומר אותו ב-session_state, ומחזיר dict JSON-friendly.

    זה מאפשר:
    - טאב Logs / Agents / Developer לראות Snapshot של סביבת הפיתוח.
    - Desktop Bridge / Agents להשתמש במידע ללא צורך בחישוב חוזר.
    """
    diag = build_developer_diagnostics(runtime)
    obj = _make_json_safe(developer_diagnostics_to_dict(diag))

    try:
        st.session_state[SESSION_KEY_DEV_DIAG_LAST] = obj
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to store developer diagnostics in session_state: %s", exc
        )

    logger.debug(
        "Developer diagnostics updated (env=%s, profile=%s, modules_core=%s)",
        diag.env,
        diag.profile,
        list(diag.core_modules.keys()),
    )

    return obj


def _render_developer_diagnostics_panel(
    runtime: DashboardRuntime,
    diag_dict: Optional[Dict[str, Any]] = None,
) -> None:
    """
    מציג פאנל Developer Diagnostics (רק אם show_debug_info=True):

    - Environment & versions
    - Paths (PROJECT_ROOT / cwd / logs)
    - Core/Root/Desktop modules availability
    - Envvars רלוונטיים
    - Session state snapshot
    - Health/Overview/Home summary
    - Cache namespaces snapshot
    """
    ff = runtime.feature_flags
    if not ff.get("show_debug_info"):
        return

    if diag_dict is None:
        try:
            diag_dict = st.session_state.get(SESSION_KEY_DEV_DIAG_LAST)
        except Exception:
            diag_dict = None

    if diag_dict is None or not isinstance(diag_dict, Mapping):
        diag_dict = update_developer_diagnostics_in_session(runtime)

    env = diag_dict.get("env")
    profile = diag_dict.get("profile")
    python_version = diag_dict.get("python_version")
    st_version = diag_dict.get("streamlit_version")
    pd_version = diag_dict.get("pandas_version")

    project_root_str = diag_dict.get("project_root")
    cwd = diag_dict.get("cwd")
    logs_path_str = diag_dict.get("logs_path")

    core_modules = diag_dict.get("core_modules") or {}
    root_modules = diag_dict.get("root_modules") or {}
    desktop_modules = diag_dict.get("desktop_modules") or {}
    envvars = diag_dict.get("envvars") or {}
    session_keys_count = diag_dict.get("session_keys_count", 0)
    session_keys_sample = diag_dict.get("session_keys_sample") or []
    warnings = diag_dict.get("warnings") or []

    health_severity = diag_dict.get("health_severity")
    health_score = diag_dict.get("health_score")
    overview_metrics_count = diag_dict.get("overview_metrics_count", 0)
    home_ctx_last_updated = diag_dict.get("home_ctx_last_updated")

    cache_namespaces = diag_dict.get("cache_namespaces") or {}
    cache_last_clear_ts = diag_dict.get("cache_last_clear_ts")

    with st.expander("🛠 Developer diagnostics (env, modules, paths, health, cache)", expanded=False):
        col1, col2, col3 = st.columns([1.4, 1.3, 1.3])

        # ---- Environment & versions + Health ----
        with col1:
            st.markdown("**Environment & versions**")
            st.write(f"Env / Profile: `{env}` / `{profile}`")
            st.write(f"Python: `{python_version}`")
            st.write(f"Streamlit: `{st_version}`  •  pandas: `{pd_version}`")
            st.write(f"Host/User: `{diag_dict.get('host')}` / `{diag_dict.get('user')}`")
            st.write(f"Started at (UTC): `{diag_dict.get('started_at_utc')}`")

            st.markdown("**Health / Overview / Home**")
            st.write(f"Health severity: `{health_severity}`  •  score≈`{health_score}`")
            st.write(f"Overview metrics: `{overview_metrics_count}`")
            st.write(f"Home context last updated: `{home_ctx_last_updated}`")

            st.markdown("**Paths**")
            st.code(
                f"PROJECT_ROOT = {project_root_str}\n"
                f"cwd          = {cwd}\n"
                f"logs_path    = {logs_path_str}",
                language="text",
            )

        # ---- Modules availability ----
        def _df_from_bool_map(mapping: Mapping[str, Any]) -> Optional[pd.DataFrame]:
            if not mapping:
                return None
            rows = []
            for k, v in mapping.items():
                rows.append({"module": k, "available": bool(v)})
            if not rows:
                return None
            df = pd.DataFrame(rows)
            df["status"] = df["available"].map(lambda x: "✅" if x else "⭕")
            df.set_index("module", inplace=True)
            return df

        with col2:
            st.markdown("**Core modules**")
            df_core = _df_from_bool_map(core_modules)
            if df_core is not None:
                st.dataframe(df_core, use_container_width=True)
            else:
                st.caption("No core module snapshot.")

            st.markdown("**Root / Desktop modules**")
            df_root = _df_from_bool_map(root_modules)
            df_desktop = _df_from_bool_map(desktop_modules)

            if df_root is not None:
                st.caption("Root modules:")
                st.dataframe(df_root, use_container_width=True)
            if df_desktop is not None:
                st.caption("Desktop modules:")
                st.dataframe(df_desktop, use_container_width=True)
            if df_root is None and df_desktop is None:
                st.caption("No root/desktop module snapshot.")

        # ---- Envvars, session state & cache ----
        with col3:
            st.markdown("**Relevant envvars**")
            if envvars:
                env_rows = [{"key": k, "value": v} for k, v in envvars.items()]
                df_env = pd.DataFrame(env_rows).set_index("key")
                st.dataframe(df_env, use_container_width=True, height=140)
            else:
                st.caption("No envvars snapshot.")

            st.markdown("**Session state**")
            st.write(f"Total keys: `{session_keys_count}`")
            if session_keys_sample:
                st.code(
                    "Sample keys:\n" + "\n".join(f"- {k}" for k in session_keys_sample),
                    language="text",
                )

            st.markdown("**Internal cache (namespaces)**")
            if cache_namespaces:
                rows = [
                    {"namespace": ns, "entries": cnt}
                    for ns, cnt in cache_namespaces.items()
                ]
                df_cache = pd.DataFrame(rows).set_index("namespace")
                st.dataframe(df_cache, use_container_width=True, height=140)
                st.caption(f"Last cache clear: `{cache_last_clear_ts}`")
            else:
                st.caption("Internal cache appears empty (or not yet used).")

        if warnings:
            st.markdown("**Warnings**")
            for w in warnings:
                st.warning(w)


# -------------------------
# Shell wrapper – enriched with diagnostics + health/API bundles
# -------------------------

def render_dashboard_shell(runtime: DashboardRuntime) -> None:  # type: ignore[override]
    """
    עטיפה מורחבת ל-Shell של הדשבורד (גרסת Dev Diagnostics + Health/API bundles):

    לפני הקריאה ל-_render_dashboard_shell_core(runtime), אנחנו:
    1. טוענים/מאתחלים UserDashboardPrefs.
    2. מיישמים את ההעדפות על ה-Runtime/FeatureFlags (show_debug_info, benchmark, nav_history).
    3. מעדכנים Agent Context (Session-level).
    4. מעדכנים Dashboard Summary (Telemetry).
    5. מעדכנים Dashboard Health (Ready / Issues / Warnings / score).
    6. מעדכנים Dashboard API Bundle (meta+health+summary+agent_context).
    7. מעדכנים Dashboard Home Context (Overview + Alerts + Health light).
    8. מציגים 🎛 Dashboard Toolbar (שליטה אישית).
    9. מציגים 🛠 Developer Diagnostics (במצב debug בלבד).
    10. שומרים Prefs ל-SqlStore (Best-effort).

    ואז:
    11. קוראים ל-_render_dashboard_shell_core(runtime) – Header + Sidebar + Tabs + Alerts.

    כך, בלי לשבור את ה-API, אנחנו מקבלים:
    - Personalization מתקדם.
    - Telemetry מלא (Summary+Health+API bundle+Overview+Home context).
    - Agent-ready context.
    - Developer Diagnostics ברמה של קרן גידור.
    """
    # 1) Prefs
    prefs = get_or_init_user_prefs(runtime)

    # 2) Apply prefs → runtime / session_state
    apply_user_prefs_to_runtime(runtime, prefs)

    # 3) Agent context – Session-level
    update_agent_context_in_session(runtime)

    # 4) Dashboard summary – Telemetry
    update_dashboard_summary_in_session(runtime, store_as_dict=True)

    # 5) Health – Ready / Issues / Warnings / score
    update_dashboard_health_in_session(runtime, include_summary=False)

    # 6) API bundle – meta+health+summary+agent_context (Session-level)
    update_dashboard_api_bundle_in_session(
        runtime,
        include_summary=True,
        include_health_summary=False,
        include_agent_context=True,
    )

    # 7) Home context – חבילת דאטה מרוכזת לטאב HOME / Monitoring
    update_dashboard_home_context_in_session(runtime)

    # 8) Toolbar – שליטה אישית (layout/debug וכו')
    _render_dashboard_toolbar(runtime, prefs)

    # 9) Developer diagnostics – רק בפרופילי debug
    diag_dict = update_developer_diagnostics_in_session(runtime)
    _render_developer_diagnostics_panel(runtime, diag_dict)

    # Persist prefs (Best-effort)
    persist_user_prefs_if_needed(runtime, prefs, auto_persist=True)

    # 11) Shell המקורי (Header + Sidebar + Tabs + Alerts + Debug timings)
    _render_dashboard_shell_core(runtime)


# עדכון __all__ עבור חלק 26
try:
    __all__ += [
        "SESSION_KEY_DEV_DIAG_LAST",
        "DeveloperDiagnostics",
        "build_developer_diagnostics",
        "developer_diagnostics_to_dict",
        "update_developer_diagnostics_in_session",
        "_render_developer_diagnostics_panel",
        "render_dashboard_shell",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_DEV_DIAG_LAST",
        "DeveloperDiagnostics",
        "build_developer_diagnostics",
        "developer_diagnostics_to_dict",
        "update_developer_diagnostics_in_session",
        "_render_developer_diagnostics_panel",
        "render_dashboard_shell",
    ]


# =====================
