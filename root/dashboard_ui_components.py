# -*- coding: utf-8 -*-
"""
root/dashboard_ui_components.py — Service Status + Global Header
==================================================================

Extracted from dashboard.py Parts 9-10/35.

Contains:
- Service status snapshot computation
- Global header rendering (app info, macro banner, risk snapshot)
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# Import deps
try:
    from root.dashboard import (
        _safe_getattr,
        _make_json_safe,
        _discover_services_mapping,
        _probe_service,
        ensure_dashboard_runtime,
        APP_NAME,
        APP_VERSION,
        RUNTIME_HOST,
        STARTED_AT_UTC,
    )
except (ImportError, AttributeError):
    _safe_getattr = getattr
    _make_json_safe = lambda x: x
    _discover_services_mapping = lambda ctx: {}
    _probe_service = lambda ctx, m, candidates=(): (False, None)
    ensure_dashboard_runtime = lambda ctx: None
    APP_NAME = "PairsTrading"
    APP_VERSION = "?"
    RUNTIME_HOST = "localhost"
    STARTED_AT_UTC = datetime.now(timezone.utc)

# Forward refs
DashboardRuntime = Any
FeatureFlags = Dict[str, Any]
AppContext = Any
ServiceStatus = Dict[str, Any]
EnvName = str
ProfileName = str

# Part 9/35 – Service status snapshot (for Sidebar & Header)
# =====================

def _guess_sql_backend_from_uri(uri: Optional[str]) -> Optional[str]:
    """
    ניחוש backend ל-SqlStore מתוך URI (duckdb/sqlite/postgres וכו').

    דוגמאות:
        duckdb:///file.duckdb  → "duckdb"
        sqlite:///file.db      → "sqlite"
        postgresql://...       → "postgresql"
    """
    if not uri:
        return None

    lower = uri.lower()
    for candidate in ("duckdb", "sqlite", "postgresql", "postgres", "mysql", "mssql"):
        if candidate in lower:
            return "postgres" if candidate == "postgresql" else candidate
    return None


def _extract_sql_store_status(
    app_ctx: "AppContext",
    caps: Mapping[str, Any],
    services_map: Mapping[str, Any],
) -> ServiceStatus:
    """
    בונה סטטוס שירות עבור SqlStore:

    מחפש:
    - זמינות (available)
    - URI / backend
    - מצב בסיסי (read_only, has_history)
    """
    has_cap = bool(caps.get("sql_store", False))
    found, sql_obj = _probe_service(
        app_ctx,
        services_map,
        candidates=("sql_store", "store", "db", "sql"),
    )

    status: ServiceStatus = {
        "available": has_cap and found and sql_obj is not None,
        "uri": None,
        "backend": None,
        "read_only": None,
        "has_history": None,
    }

    if not status["available"]:
        return status

    # URI
    uri = None
    for attr in ("uri", "url", "engine_url", "connection_url"):
        val = _safe_getattr(sql_obj, attr)
        if isinstance(val, str) and val.strip():
            uri = val.strip()
            break
    status["uri"] = uri
    status["backend"] = _guess_sql_backend_from_uri(uri)

    # read_only
    read_only = _safe_getattr(sql_obj, "read_only", None)
    if isinstance(read_only, bool):
        status["read_only"] = read_only

    # has_history – heuristic (האם יש טבלאות היסטוריות)
    try:
        tables = _safe_getattr(sql_obj, "list_tables", None)
        if callable(tables):
            tbls = tables()
            if isinstance(tbls, Sequence):
                status["has_history"] = any(
                    "history" in str(t).lower() or "pnl" in str(t).lower()
                    for t in tbls
                )
    except Exception:
        pass

    return status


def _extract_broker_status(
    app_ctx: "AppContext",
    caps: Mapping[str, Any],
    services_map: Mapping[str, Any],
) -> ServiceStatus:
    """
    בונה סטטוס שירות עבור Broker/IBKR:

    מחפש:
    - זמינות (available)
    - מצב חיבור (connected)
    - mode ("paper"/"live"/"sim")
    - account_id (אם קיים)
    """
    has_cap = bool(caps.get("broker", False))
    found, broker_obj = _probe_service(
        app_ctx,
        services_map,
        candidates=("broker", "ibkr", "ib", "broker_service", "execution"),
    )

    status: ServiceStatus = {
        "available": has_cap and found and broker_obj is not None,
        "connected": None,
        "mode": caps.get("broker_mode"),
        "account_id": None,
    }

    if not status["available"]:
        return status

    # connected
    connected = None
    for attr in ("connected", "is_connected", "isConnected", "is_alive", "isAlive"):
        val = _safe_getattr(broker_obj, attr)
        if isinstance(val, bool):
            connected = val
            break
    status["connected"] = connected

    # mode – אם ה-capabilities לא נתן, ננסה מהאובייקט
    if not status["mode"]:
        for attr in ("mode", "account_type", "profile", "env", "environment"):
            val = _safe_getattr(broker_obj, attr)
            if isinstance(val, str) and val.strip():
                status["mode"] = val.strip().lower()
                break

    # account_id
    for attr in ("account_id", "account", "accountNumber", "account_number"):
        val = _safe_getattr(broker_obj, attr)
        if isinstance(val, str) and val.strip():
            status["account_id"] = val.strip()
            break

    return status


def _extract_market_data_status(
    app_ctx: "AppContext",
    caps: Mapping[str, Any],
    services_map: Mapping[str, Any],
) -> ServiceStatus:
    """
    בונה סטטוס עבור Market Data Router:

    מחפש:
    - זמינות (available)
    - מקור / provider ("IBKR", "Yahoo", "Parquet"...)
    - latency_mode (live/delayed/offline)
    """
    has_cap = bool(caps.get("market_data_router", False))
    found, md_obj = _probe_service(
        app_ctx,
        services_map,
        candidates=("market_data_router", "market_data", "md_router", "data_router"),
    )

    status: ServiceStatus = {
        "available": has_cap and found and md_obj is not None,
        "source": None,
        "latency_mode": None,
    }

    if not status["available"]:
        return status

    # source/provider/backend
    for attr in ("source", "provider", "backend", "mode"):
        val = _safe_getattr(md_obj, attr)
        if isinstance(val, str) and val.strip():
            status["source"] = val.strip()
            break

    # latency_mode (live/delayed/offline)
    for attr in ("latency_mode", "latency", "data_mode"):
        val = _safe_getattr(md_obj, attr)
        if isinstance(val, str) and val.strip():
            status["latency_mode"] = val.strip().lower()
            break

    return status


def _extract_engine_status(
    engine_name: str,
    attr_candidates: Sequence[str],
    app_ctx: "AppContext",
    caps: Mapping[str, Any],
    services_map: Mapping[str, Any],
) -> ServiceStatus:
    """
    Extractor כללי ל-engines (risk / signals / macro / agents / fair_value):

    engine_name – לצרכי לוגים בלבד.
    attr_candidates – שמות אפשריים לשירות ב-AppContext/services_map.
    """
    has_cap = bool(caps.get(engine_name, False))
    found, engine_obj = _probe_service(app_ctx, services_map, candidates=attr_candidates)

    status: ServiceStatus = {
        "available": has_cap and found and engine_obj is not None,
        "status": None,
        "last_update": None,
        "extra": {},
    }

    if not status["available"]:
        return status

    # status / state / mode
    for attr in ("status", "state", "mode"):
        val = _safe_getattr(engine_obj, attr)
        if isinstance(val, str) and val.strip():
            status["status"] = val.strip()
            break

    # last_update / last_run
    last_ts = None
    for attr in ("last_update_ts", "last_run_ts", "last_snapshot_ts", "last_refresh"):
        val = _safe_getattr(engine_obj, attr)
        if isinstance(val, (datetime,)):
            last_ts = val
            break
        if isinstance(val, str) and val.strip():
            # נשאיר כמחרוזת; לא בהכרח נרצה לפרש כאן
            last_ts = val
            break
    if last_ts is not None:
        status["last_update"] = (
            last_ts.isoformat() if isinstance(last_ts, datetime) else str(last_ts)
        )

    # extra – שדות מעניינים ספציפיים
    extra: Dict[str, Any] = {}

    if engine_name == "macro_engine":
        # ננסה current_regime / active_regime
        for attr in ("current_regime", "active_regime", "regime"):
            val = _safe_getattr(engine_obj, attr)
            if isinstance(val, str) and val.strip():
                extra["regime"] = val.strip()
                break
    elif engine_name == "risk_engine":
        for attr in ("kill_switch_armed", "kill_switch_state"):
            val = _safe_getattr(engine_obj, attr)
            if isinstance(val, bool):
                extra[attr] = val
        # אולי יש summary של חריגות
        alerts = _safe_getattr(engine_obj, "active_alerts", None)
        if isinstance(alerts, Sequence) and alerts:
            extra["active_alerts_count"] = len(alerts)
    elif engine_name == "agents_manager":
        for attr in ("online", "is_online", "healthy"):
            val = _safe_getattr(engine_obj, attr)
            if isinstance(val, bool):
                extra["online"] = val
                break

    if extra:
        status["extra"] = extra

    return status


def _extract_backtest_opt_status(
    caps: Mapping[str, Any],
) -> Dict[str, ServiceStatus]:
    """
    סטטוס לוגי בלבד (לפי capabilities) עבור Backtester/Optimizer/Meta-Optimizer.

    אין כאן חיבור לאובייקטים כי הם בדרך כלל מודולים פונקציונליים ולא "שירות".
    """
    return {
        "backtester": {
            "available": bool(caps.get("backtester", False)),
        },
        "optimizer": {
            "available": bool(caps.get("optimizer", False)),
        },
        "meta_optimizer": {
            "available": bool(caps.get("meta_optimizer", False)),
        },
    }


def _get_app_runtime_status(feature_flags: FeatureFlags) -> ServiceStatus:
    """
    מחזיר סטטוס כללי של האפליקציה (App-level):

    - app_name, version
    - env, profile
    - host/user
    - uptime דקות/שעות
    """
    now = datetime.now(timezone.utc)
    uptime_seconds = (now - STARTED_AT_UTC).total_seconds()
    uptime_hours = round(uptime_seconds / 3600.0, 2)

    return {
        "app_name": feature_flags.get("app_name", APP_NAME),
        "version": feature_flags.get("version", APP_VERSION),
        "env": feature_flags.get("env", DEFAULT_ENV),
        "profile": feature_flags.get("profile", DEFAULT_PROFILE),
        "host": feature_flags.get("host", RUNTIME_HOST),
        "user": feature_flags.get("user", RUNTIME_USER),
        "uptime_hours": uptime_hours,
    }


def _get_services_status(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
) -> Dict[str, ServiceStatus]:
    """
    בונה תמונת מצב מלאה של שירותי המערכת עבור Sidebar/Header:

    מחזיר dict עם מפתחות:
        - "app"           – סטטוס כלל מערכת
        - "sql_store"     – סטטוס שכבת Persist
        - "broker"        – סטטוס ברוקר/IBKR
        - "market_data"   – סטטוס שכבת Market Data
        - "risk_engine"   – סטטוס מנוע סיכון
        - "signals_engine"– סטטוס מנוע סיגנלים
        - "macro_engine"  – סטטוס מנוע מקרו
        - "agents"        – סטטוס סוכני AI
        - "fair_value"    – סטטוס מנוע Fair Value
        - "backtester"    – זמינות Backtester
        - "optimizer"     – זמינות Optimizer
        - "meta_optimizer"– זמינות Meta-Optimizer

    הסטטוסים משתמשים גם ב-capabilities (feature_flags["capabilities"])
    וגם בגילוי שירותים מה-AppContext.
    """
    caps: Mapping[str, Any] = feature_flags.get("capabilities", {}) or {}
    services_map = _discover_services_mapping(app_ctx)

    statuses: Dict[str, ServiceStatus] = {}

    # App-level
    statuses["app"] = _get_app_runtime_status(feature_flags)

    # Core services
    statuses["sql_store"] = _extract_sql_store_status(app_ctx, caps, services_map)
    statuses["broker"] = _extract_broker_status(app_ctx, caps, services_map)
    statuses["market_data"] = _extract_market_data_status(app_ctx, caps, services_map)

    # Engines
    statuses["risk_engine"] = _extract_engine_status(
        "risk_engine",
        ("risk_engine", "risk", "risk_service"),
        app_ctx,
        caps,
        services_map,
    )
    statuses["signals_engine"] = _extract_engine_status(
        "signals_engine",
        ("signals_engine", "signal_engine", "signals", "signal_generator"),
        app_ctx,
        caps,
        services_map,
    )
    statuses["macro_engine"] = _extract_engine_status(
        "macro_engine",
        ("macro_engine", "macro_model", "macro", "macro_service"),
        app_ctx,
        caps,
        services_map,
    )
    statuses["agents"] = _extract_engine_status(
        "agents_manager",
        ("agents_manager", "agent_manager", "agents", "ai_agents"),
        app_ctx,
        caps,
        services_map,
    )
    statuses["fair_value"] = _extract_engine_status(
        "fair_value_engine",
        ("fair_value_engine", "fair_value", "fv_engine"),
        app_ctx,
        caps,
        services_map,
    )

    # Backtest/optimization – לפי capabilities בלבד
    bt_opt = _extract_backtest_opt_status(caps)
    statuses.update(bt_opt)

    if feature_flags.get("show_debug_info"):
        logger.debug("Services status snapshot: %s", statuses)

    return statuses


# עדכון __all__ עבור חלק 9
try:
    __all__ += [
        "_guess_sql_backend_from_uri",
        "_extract_sql_store_status",
        "_extract_broker_status",
        "_extract_market_data_status",
        "_extract_engine_status",
        "_extract_backtest_opt_status",
        "_get_app_runtime_status",
        "_get_services_status",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_guess_sql_backend_from_uri",
        "_extract_sql_store_status",
        "_extract_broker_status",
        "_extract_market_data_status",
        "_extract_engine_status",
        "_extract_backtest_opt_status",
        "_get_app_runtime_status",
        "_get_services_status",
    ]

# =====================
# Part 10/35 – Global header (App info, Macro banner, Risk snapshot)
# =====================

def _extract_dates_from_base_ctx(base_ctx: Any) -> Tuple[Optional[date], Optional[date]]:
    """
    מנסה להוציא start_date / end_date מתוך base_dashboard_context.

    תומך גם ב-objects (attribs) וגם ב-Mapping (keys).
    """
    start: Optional[date] = None
    end: Optional[date] = None

    # attribs
    try:
        if hasattr(base_ctx, "start_date"):
            val = getattr(base_ctx, "start_date")
            if isinstance(val, date):
                start = val
        if hasattr(base_ctx, "end_date"):
            val = getattr(base_ctx, "end_date")
            if isinstance(val, date):
                end = val
    except Exception:
        pass

    # mapping
    try:
        if isinstance(base_ctx, Mapping):
            s = base_ctx.get("start_date")
            e = base_ctx.get("end_date")
            if isinstance(s, date):
                start = s
            if isinstance(e, date):
                end = e
    except Exception:
        pass

    return start, end


def _extract_benchmark_from_base_ctx(base_ctx: Any) -> Optional[str]:
    """
    מנסה להוציא benchmark מתוך base_dashboard_context.
    """
    # attrib
    try:
        if hasattr(base_ctx, "benchmark"):
            val = getattr(base_ctx, "benchmark")
            if isinstance(val, str) and val.strip():
                return val.strip().upper()
    except Exception:
        pass

    # mapping
    try:
        if isinstance(base_ctx, Mapping):
            val = base_ctx.get("benchmark")
            if isinstance(val, str) and val.strip():
                return val.strip().upper()
    except Exception:
        pass

    return None


def _render_macro_banner(
    macro_status: ServiceStatus,
    feature_flags: FeatureFlags,
) -> None:
    """
    מציג "Macro banner" קומפקטי בחלק העליון של הדשבורד.

    לוגיקה:
    -------
    - אם Macro Engine לא זמין → הצגת הודעה ניטרלית.
    - אם קיים regime ב-extra["regime"] → הצגת מצב (Risk-On / Risk-Off / Mixed).
    - אם יש last_update → הצג זמן עדכון אחרון.
    """
    st.markdown("##### 🌐 Macro regime")

    if not macro_status.get("available"):
        st.caption("Macro engine not available – showing neutral regime.")
        st.info("Regime: **Neutral**  |  Signal: _No macro overlay_")
        return

    regime = None
    extra = macro_status.get("extra") or {}
    if isinstance(extra, Mapping):
        regime = extra.get("regime")

    regime_str = str(regime) if regime else "Neutral"

    last_update = macro_status.get("last_update")
    suffix = ""
    if last_update:
        suffix = f"  \n_Last update: {last_update}_"

    st.info(f"**Regime:** {regime_str}{suffix}")


def _render_risk_banner(
    risk_status: ServiceStatus,
    feature_flags: FeatureFlags,
) -> None:
    """
    מציג תקציר מצב סיכון:

    - האם Risk Engine זמין.
    - מצב Kill-switch (אם מוכר).
    - מספר Alerts פעילים (אם קיים).
    """
    st.markdown("##### ⚠️ Risk snapshot")

    if not risk_status.get("available"):
        st.caption("Risk engine not available – using basic limits only.")
        st.warning("Risk engine offline – **no automatic kill-switch**.")
        return

    extra = risk_status.get("extra") or {}
    kill_armed = None
    if isinstance(extra, Mapping):
        kill_armed = extra.get("kill_switch_armed") or extra.get("kill_switch_state")
    active_alerts_count = None
    if isinstance(extra, Mapping):
        active_alerts_count = extra.get("active_alerts_count")

    cols = st.columns(2)

    with cols[0]:
        if isinstance(kill_armed, bool):
            if kill_armed:
                st.success("Kill-switch: **ARMED**")
            else:
                st.warning("Kill-switch: **DISARMED**")
        else:
            st.info("Kill-switch: _Unknown_")

    with cols[1]:
        if isinstance(active_alerts_count, int):
            if active_alerts_count > 0:
                st.error(f"Active alerts: **{active_alerts_count}**")
            else:
                st.success("Active alerts: **0**")
        else:
            st.info("Active alerts: _N/A_")


def _render_trading_mode_badge(
    broker_status: ServiceStatus,
    feature_flags: FeatureFlags,
) -> None:
    """
    מציג Badge קטן של מצב המסחר (Live / Paper / Offline).

    מסתכל על:
    - feature_flags["enable_live_trading_actions"]
    - broker_status["available"] / ["connected"] / ["mode"]
    """
    enable_live = bool(feature_flags.get("enable_live_trading_actions", False))

    if not broker_status.get("available"):
        st.warning("Trading mode: **Offline**  \nBroker unavailable.")
        return

    mode = str(broker_status.get("mode") or "").lower()
    connected = broker_status.get("connected")

    if connected is False:
        st.warning("Trading mode: **Disconnected**  \nBroker not connected.")
        return

    if enable_live and mode in ("live", "prod", "production"):
        st.success("Trading mode: **LIVE** ✅")
    elif mode in ("paper", "sim", "demo"):
        st.info("Trading mode: **PAPER** 🧪")
    else:
        st.info(f"Trading mode: **{mode or 'Unknown'}**")


def _render_global_header(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    services_status: Dict[str, ServiceStatus],
) -> None:
    """
    מציג Header עליון ברמת קרן גידור:

    צד שמאל:
        - APP_NAME + version
        - env/profile + run_id
        - base_currency / timezone
        - טווח תאריכים + benchmark (אם קיימים ב-base_dashboard_context)

    צד ימין:
        - מצב מסחר (Trading mode badge)
        - Macro banner קצר
        - Risk snapshot קצר
    """
    env: EnvName = feature_flags.get("env", DEFAULT_ENV)  # type: ignore[assignment]
    profile: ProfileName = feature_flags.get("profile", DEFAULT_PROFILE)  # type: ignore[assignment]
    run_id = get_session_run_id()

    base_ctx = st.session_state.get(SESSION_KEY_BASE_CTX)
    base_currency = _extract_base_currency(app_ctx)
    tz_name = _extract_timezone(app_ctx)

    start_date, end_date = _extract_dates_from_base_ctx(base_ctx)
    benchmark = _extract_benchmark_from_base_ctx(base_ctx) or "SPY"

    app_status = services_status.get("app", {})
    broker_status = services_status.get("broker", {})
    macro_status = services_status.get("macro_engine", {})
    risk_status = services_status.get("risk_engine", {})

    top = st.container()
    with top:
        # Professional institutional header bar
        dates_line = ""
        if start_date and end_date:
            dates_line = f"{start_date.isoformat()} → {end_date.isoformat()}"
        elif start_date:
            dates_line = f"{start_date.isoformat()} → present"
        elif end_date:
            dates_line = f"↑ {end_date.isoformat()}"

        env_colors = {
            "live": "#C62828", "paper": "#1565C0", "dev": "#37474F",
            "research": "#2E7D32", "backtest": "#6A1B9A", "staging": "#E65100",
        }
        env_color = env_colors.get(env, "#37474F")

        st.markdown(
            f"""
<div style="
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(90deg, #0D1B2A 0%, #1A2744 100%);
    border-radius: 10px;
    padding: 12px 20px;
    margin-bottom: 6px;
    border-left: 5px solid #1E88E5;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
">
    <div style="display:flex;align-items:center;gap:16px;">
        <div>
            <div style="font-size:1.20rem;font-weight:800;color:#E3F2FD;letter-spacing:-0.3px;">
                {APP_ICON} {APP_NAME}
            </div>
            <div style="font-size:0.73rem;color:#78909C;margin-top:1px;">
                v{APP_VERSION} &nbsp;·&nbsp;
                <span style="color:#90A4AE;">Host:</span> {app_status.get('host', RUNTIME_HOST)} &nbsp;·&nbsp;
                <span style="color:#90A4AE;">User:</span> {app_status.get('user', RUNTIME_USER)}
                {f'&nbsp;·&nbsp;<span style="color:#90A4AE;">Dates:</span> {dates_line}' if dates_line else ''}
            </div>
        </div>
        <div style="display:flex;gap:6px;align-items:center;">
            <span style="background:{env_color};color:white;padding:3px 10px;border-radius:12px;
                         font-size:0.70rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">
                {env}
            </span>
            <span style="background:#283142;color:#B0BEC5;padding:3px 10px;border-radius:12px;
                         font-size:0.70rem;font-weight:600;text-transform:uppercase;">
                {profile}
            </span>
        </div>
    </div>
    <div style="font-size:0.68rem;color:#546E7A;text-align:right;">
        <div><b style="color:#90A4AE;">Currency:</b> {base_currency} &nbsp; <b style="color:#90A4AE;">TZ:</b> {tz_name}</div>
        <div><b style="color:#90A4AE;">Benchmark:</b> {benchmark} &nbsp; <b style="color:#90A4AE;">Run:</b> {run_id}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        col_left, col_mid, col_right = st.columns([1, 1, 1])

        with col_left:
            pass  # header info now in the banner above

        with col_mid:
            _render_macro_banner(macro_status, feature_flags)

        with col_right:
            _render_trading_mode_badge(broker_status, feature_flags)
            _render_risk_banner(risk_status, feature_flags)

def _render_env_mode_banner(runtime: DashboardRuntime) -> None:
    """
    מציג Banner קצר וברור של מצב הסביבה (DEV / RESEARCH / BACKTEST / PAPER / LIVE),
    בנוסף ל-Header ול-Critical Alerts.

    הרעיון:
    --------
    - לתת "פס" ויזואלי מיידי שמבהיר באיזו סביבה נמצאים.
    - לא מחליף את ה-Risk/Health, רק מחזק מודעות.
    """
    env = runtime.env
    profile = runtime.profile

    if env == "live":
        st.error(
            f"🚨 LIVE TRADING ENVIRONMENT — env=`{env}`, profile=`{profile}`.  \n"
            "כל פעולה עלולה להשפיע על חשבון אמיתי."
        )
    elif env == "paper":
        st.warning(
            f"🧪 PAPER environment — env=`{env}`, profile=`{profile}`.  \n"
            "מסחר מדומה בלבד (Paper / Demo)."
        )
    elif env in ("backtest", "research"):
        st.info(
            f"🔬 Research / Backtest environment — env=`{env}`, profile=`{profile}`.  \n"
            "ריצות סימולציה ומחקר בלבד."
        )
    else:
        st.info(
            f"💻 Development / Staging environment — env=`{env}`, profile=`{profile}`."
        )

# עדכון __all__ עבור חלק 10
try:
    __all__ += [
        "_extract_dates_from_base_ctx",
        "_extract_benchmark_from_base_ctx",
        "_render_macro_banner",
        "_render_risk_banner",
        "_render_trading_mode_badge",
        "_render_global_header",
        "_render_env_mode_banner",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_extract_dates_from_base_ctx",
        "_extract_benchmark_from_base_ctx",
        "_render_macro_banner",
        "_render_risk_banner",
        "_render_trading_mode_badge",
        "_render_global_header",
        "_render_env_mode_banner",
    ]

# =====================
