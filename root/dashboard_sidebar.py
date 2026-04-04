# -*- coding: utf-8 -*-
"""
root/dashboard_sidebar.py — Global Sidebar
=============================================

Extracted from dashboard.py Part 12/35.

Contains the Streamlit sidebar rendering: app overview, services status,
quick actions, and debug info.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
        RUNTIME_USER,
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
    RUNTIME_USER = "unknown"
    STARTED_AT_UTC = datetime.now(timezone.utc)

# Forward refs
DashboardRuntime = Any
FeatureFlags = Dict[str, Any]
AppContext = Any
ServiceStatus = Dict[str, Any]
EnvName = str
ProfileName = str

# Part 12/35 – Global sidebar (App overview, services, quick actions, debug)
# =====================

def _format_bool_tristate(
    value: Optional[bool],
    true_text: str = "Yes",
    false_text: str = "No",
    none_text: str = "N/A",
) -> str:
    """
    פורמט נוח לערכי True/False/None להצגה ב-Sidebar.
    """
    if value is True:
        return true_text
    if value is False:
        return false_text
    return none_text


def _render_app_overview_sidebar(
    feature_flags: FeatureFlags,
    services_status: Dict[str, ServiceStatus],
) -> None:
    """
    מציג סקירה מהירה של האפליקציה בסיידבר:

    - App name + version
    - env/profile
    - uptime
    """
    app_status = services_status.get("app", {})
    env: EnvName = feature_flags.get("env", DEFAULT_ENV)  # type: ignore[assignment]
    profile: ProfileName = feature_flags.get("profile", DEFAULT_PROFILE)  # type: ignore[assignment]

    # Professional branded sidebar header
    env_badge_color = {
        "live": "#C62828", "paper": "#1565C0", "dev": "#37474F",
        "research": "#2E7D32", "backtest": "#6A1B9A", "staging": "#E65100",
    }.get(env, "#37474F")

    st.sidebar.markdown(
        f"""
<div style="
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
    border-radius: 10px;
    padding: 16px 14px 12px 14px;
    margin-bottom: 8px;
    border-left: 4px solid #1E88E5;
">
    <div style="font-size: 1.15rem; font-weight: 800; color: #E3F2FD; letter-spacing: -0.3px;">
        {APP_ICON} {feature_flags.get('app_name', APP_NAME)}
    </div>
    <div style="font-size: 0.72rem; color: #90A4AE; margin-top: 2px;">
        v{feature_flags.get('version', APP_VERSION)}
    </div>
    <div style="margin-top: 8px; display: flex; gap: 6px; flex-wrap: wrap;">
        <span style="
            background: {env_badge_color};
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        ">{env}</span>
        <span style="
            background: #283142;
            color: #B0BEC5;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.68rem;
            font-weight: 600;
            text-transform: uppercase;
        ">{profile}</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    uptime_hours = app_status.get("uptime_hours")
    uptime_str = f"⏱ Uptime: ~{uptime_hours:.1f}h" if isinstance(uptime_hours, (int, float)) else ""
    host = app_status.get("host", RUNTIME_HOST)
    user = app_status.get("user", RUNTIME_USER)
    st.sidebar.caption(f"🖥 `{host}` · 👤 `{user}`" + (f"  {uptime_str}" if uptime_str else ""))


def _render_services_status_sidebar(
    services_status: Dict[str, ServiceStatus],
) -> None:
    """
    מציג סטטוס שירותים מרכזיים בסיידבר ברמת "קריאת מצב מהירה":

    - SqlStore (backend, history)
    - Broker (mode, connected)
    - Market Data (source, latency)
    - Risk / Macro / Agents / Fair Value / Backtest / Optimizer
    """
    sql = services_status.get("sql_store", {})
    broker = services_status.get("broker", {})
    market = services_status.get("market_data", {})
    risk = services_status.get("risk_engine", {})
    macro = services_status.get("macro_engine", {})
    agents = services_status.get("agents", {})
    fv = services_status.get("fair_value", {})
    bt = services_status.get("backtester", {})
    opt = services_status.get("optimizer", {})
    meta_opt = services_status.get("meta_optimizer", {})

    def _status_dot(available: Any) -> str:
        return "🟢" if available else "🔴"

    def _svc_row(icon: str, name: str, detail: str) -> str:
        return (
            f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;'
            f'border-bottom:1px solid #283142;">'
            f'<span style="font-size:0.70rem;">{icon}</span>'
            f'<span style="font-size:0.73rem;color:#CFD8DC;font-weight:600;flex:1;">{name}</span>'
            f'<span style="font-size:0.68rem;color:#78909C;text-align:right;">{detail}</span>'
            f'</div>'
        )

    sql_detail = f"{sql.get('backend','?')} · hist={'✓' if sql.get('has_history') else '✗'}"
    br_detail = f"{broker.get('mode','?')} · {'conn' if broker.get('connected') else 'off'}"
    md_detail = f"{market.get('source','?')} · {(market.get('latency_mode') or '?').lower()}"

    rows_html = "".join([
        _svc_row(_status_dot(sql.get("available")),   "SqlStore",    sql_detail),
        _svc_row(_status_dot(broker.get("available")), "Broker",      br_detail),
        _svc_row(_status_dot(market.get("available")), "Market Data", md_detail),
        _svc_row(_status_dot(risk.get("available")),   "Risk Engine", ""),
        _svc_row(_status_dot(macro.get("available")),  "Macro Engine",""),
        _svc_row(_status_dot(agents.get("available")), "Agents",      ""),
        _svc_row(_status_dot(fv.get("available")),     "Fair Value",  ""),
        _svc_row(_status_dot(bt.get("available")),     "Backtester",  ""),
        _svc_row(_status_dot(opt.get("available")),    "Optimizer",   ""),
        _svc_row(_status_dot(meta_opt.get("available")),"Meta-Opt",   ""),
    ])

    st.sidebar.markdown(
        f"""
<div style="
    background:#0F1923;
    border-radius:8px;
    padding:10px 12px;
    margin:8px 0;
    border:1px solid #1E2A38;
">
    <div style="font-size:0.72rem;font-weight:700;color:#607D8B;
                text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px;">
        🧩 Services
    </div>
    {rows_html}
</div>
""",
        unsafe_allow_html=True,
    )


def _render_sidebar_debug_section(
    feature_flags: FeatureFlags,
) -> None:
    """
    מציג סקשן Debug בסיידבר, אם show_debug_info=True:

    - Feature flags (תת-סט)
    - nav history קצר
    """
    if not feature_flags.get("show_debug_info"):
        return

    st.sidebar.markdown("#### 🐞 Debug info")

    # Feature flags – נציג תת-סט מרכזי כדי לא להציף
    ff_view = {
        "env": feature_flags.get("env"),
        "profile": feature_flags.get("profile"),
        "enable_live_trading_actions": feature_flags.get("enable_live_trading_actions"),
        "use_sql_backed_state": feature_flags.get("use_sql_backed_state"),
        "enable_experiment_mode": feature_flags.get("enable_experiment_mode"),
        "desktop_integration": feature_flags.get("desktop_integration"),
    }
    with st.sidebar.expander("Feature flags", expanded=False):
        st.json(ff_view)

    # Nav history
    try:
        history = st.session_state.get(SESSION_KEY_NAV_HISTORY, [])
    except Exception:
        history = []

    if isinstance(history, list) and history:
        with st.sidebar.expander("Navigation history", expanded=False):
            # נציג רק את 20 האחרונים כדי לשמור על ביצועים
            tail = history[-20:]
            st.json(tail)
    else:
        st.sidebar.caption("No navigation history yet.")


def _render_global_sidebar(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    services_status: Dict[str, ServiceStatus],
) -> None:
    """
    Sidebar ברמת קרן גידור:

    - App overview (env/profile/uptime).
    - Services status (SqlStore/Broker/MarketData/Engines).
    - Quick actions (rerun, clear cache, snapshot).
    - Debug section (אופציונלי, לפי show_debug_info).
    """
    _render_app_overview_sidebar(feature_flags, services_status)
    _render_services_status_sidebar(services_status)
    _render_quick_actions(app_ctx, feature_flags, services_status)
    _render_sidebar_debug_section(feature_flags)

# =====================
# Numeric input helpers (HF-grade, no sliders)
# =====================

from typing import Tuple

try:
    # אם יש לך כבר PARAM_SPECS ב-core/params.py
    from core.params import PARAM_SPECS  # type: ignore[import]
except Exception:
    PARAM_SPECS = {}  # fallback ריק


def _get_param_bounds_from_specs(
    name: str,
    fallback_lo: float,
    fallback_hi: float,
    fallback_step: float = 0.1,
) -> Tuple[float, float, float]:
    """
    מחלץ טווחים מ-PARAM_SPECS אם קיימים, אחרת משתמש ב-fallback.

    מניח שב-Spec יש שדות כמו lo/hi/step.
    """
    spec = PARAM_SPECS.get(name)
    if spec is None:
        return fallback_lo, fallback_hi, fallback_step

    lo = getattr(spec, "lo", fallback_lo)
    hi = getattr(spec, "hi", fallback_hi)
    step = getattr(spec, "step", fallback_step)
    try:
        lo_f = float(lo)
        hi_f = float(hi)
        step_f = float(step)
    except Exception:
        lo_f, hi_f, step_f = fallback_lo, fallback_hi, fallback_step

    return lo_f, hi_f, step_f


def number_param_input(
    name: str,
    label: str,
    default: float,
    lo: float,
    hi: float,
    step: float = 0.1,
    use_specs: bool = True,
    key_prefix: str = "",
) -> float:
    """
    תחליף לסליידר עבור פרמטר יחיד:

    - אם use_specs=True ויש PARAM_SPECS[name] → משתמש בטווחים משם (lo/hi/step).
    - אחרת → משתמש ב-lo/hi/step שהועברו.

    דוגמה שימוש:
        z_open = number_param_input(
            "z_open", "Z open",
            default=1.5, lo=-3.0, hi=3.0, step=0.1,
            key_prefix="opt_",
        )
    """
    if use_specs:
        lo, hi, step = _get_param_bounds_from_specs(name, lo, hi, step)

    widget_key = f"{key_prefix}{name}"
    return float(
        st.number_input(
            label,
            min_value=float(lo),
            max_value=float(hi),
            value=float(default),
            step=float(step),
            key=widget_key,
        )
    )


def number_range_input(
    name: str,
    label_min: str,
    label_max: str,
    default_min: float,
    default_max: float,
    lo: float,
    hi: float,
    step: float = 0.1,
    use_specs: bool = True,
    key_prefix: str = "",
) -> Tuple[float, float]:
    """
    קלט טווח (min, max) במספרים – תחליף ל-slider על טווח.

    - אם use_specs=True → lo/hi/step מגיעים מ-PARAM_SPECS[name] אם אפשר.
    - מבטיח ש-min <= max (אם לא – מתקנים בעדינות).

    דוגמה:
        z_min, z_max = number_range_input(
            "z_range", "Z min", "Z max",
            default_min=-2.0, default_max=2.0,
            lo=-5.0, hi=5.0, step=0.1,
            key_prefix="scan_",
        )
    """
    if use_specs:
        lo, hi, step = _get_param_bounds_from_specs(name, lo, hi, step)

    col_min, col_max = st.columns(2)
    widget_key_min = f"{key_prefix}{name}_min"
    widget_key_max = f"{key_prefix}{name}_max"

    with col_min:
        v_min = st.number_input(
            label_min,
            min_value=float(lo),
            max_value=float(hi),
            value=float(default_min),
            step=float(step),
            key=widget_key_min,
        )
    with col_max:
        v_max = st.number_input(
            label_max,
            min_value=float(lo),
            max_value=float(hi),
            value=float(default_max),
            step=float(step),
            key=widget_key_max,
        )

    # הבטחת סדר – לא זורקים שגיאה, רק מתקנים
    if v_min > v_max:
        v_min, v_max = v_max, v_min

    return float(v_min), float(v_max)

# =====================
# Widget key helper (centralized, reuses common.ui_helpers if available)
# =====================

try:  # pragma: no cover - optional external helper
    from common.ui_helpers import make_widget_key as _external_make_widget_key  # type: ignore[import]
except Exception:  # noqa: BLE001 - best-effort optional import
    _external_make_widget_key = None  # type: ignore[assignment]


def make_widget_key(*parts: str) -> str:
    """
    מחולל מפתח widget אחיד ל-Streamlit, עם תמיכה ב-helper חיצוני אם קיים.

    אם common.ui_helpers.make_widget_key קיים – נשתמש בו.
    אחרת – נייצר מפתח לוקלי בצורת "dash:part1:part2:...".
    """
    if _external_make_widget_key is not None:
        try:
            return _external_make_widget_key(*parts)
        except Exception:
            # אם helper חיצוני נפל – ניפול חזרה למימוש המקומי
            pass

    safe_parts: List[str] = []
    for p in parts:
        s = str(p).strip()
        if not s:
            continue
        safe_parts.append(s.replace(" ", "_"))

    base = ":".join(safe_parts) if safe_parts else "widget"
    return f"dash:{base}"

# עדכון __all__ עבור חלק 12
try:
    __all__ += [
        "_format_bool_tristate",
        "_render_app_overview_sidebar",
        "_render_services_status_sidebar",
        "_render_sidebar_debug_section",
        "_render_global_sidebar",
        "number_param_input",
        "number_range_input",
        "make_widget_key",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_format_bool_tristate",
        "_render_app_overview_sidebar",
        "_render_services_status_sidebar",
        "_render_sidebar_debug_section",
        "_render_global_sidebar",
        "number_param_input",
        "number_range_input",
        "make_widget_key",
    ]



    
# =====================
