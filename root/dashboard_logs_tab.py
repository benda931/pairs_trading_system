# -*- coding: utf-8 -*-
"""
root/dashboard_logs_tab.py — Logs / System Health Tab
=======================================================

Extracted from dashboard.py Part 23/35.

Contains the fallback logs/health tab implementation that renders
when no external logs_tab module is available.
"""
from __future__ import annotations

import importlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# Import deps
try:
    from root.dashboard import (
        ensure_dashboard_runtime,
        _log_tab_entry,
        _find_module,
        _find_tab_function_in_module,
        _invoke_tab_function,
        _collect_session_tab_timings,
        _collect_session_tab_errors,
        _collect_session_nav_history_tail,
        update_agent_context_in_session,
        DASHBOARD_LOG_PATH,
        TAB_KEY_LOGS,
    )
    from root.dashboard_telemetry import (
        build_dashboard_summary,
        DashboardSummary,
        SESSION_KEY_DASHBOARD_SUMMARY,
    )
except ImportError:
    ensure_dashboard_runtime = lambda ctx: None
    _log_tab_entry = lambda *a, **kw: None
    _find_module = lambda *a: None
    _find_tab_function_in_module = lambda *a: None
    _invoke_tab_function = lambda *a, **kw: None
    _collect_session_tab_timings = lambda: {}
    _collect_session_tab_errors = lambda: {}
    _collect_session_nav_history_tail = lambda limit=50: []
    update_agent_context_in_session = lambda *a, **kw: None
    DASHBOARD_LOG_PATH = Path("logs/dashboard.log")
    TAB_KEY_LOGS = "logs"
    build_dashboard_summary = lambda runtime: None
    DashboardSummary = Any
    SESSION_KEY_DASHBOARD_SUMMARY = "dashboard_summary"

# Forward refs
DashboardRuntime = Any
FeatureFlags = Dict[str, Any]
NavPayload = Optional[Dict[str, Any]]
AppContext = Any

# Part 23/35 – Advanced Logs / System Health tab (HF-grade fallback implementation)
# =====================

def _severity_to_emoji(severity: str) -> str:
    """
    ממפה רמת חומרה (ok/warning/error) לאימוג'י קצר וקריא.
    """
    s = severity.lower().strip()
    if s == "error":
        return "🚨"
    if s == "warning":
        return "⚠️"
    return "✅"


def _build_services_df(summary: Optional[DashboardSummary]) -> Optional[pd.DataFrame]:
    """
    בונה DataFrame של מצב שירותים מתוך DashboardSummary (אם קיים).
    """
    if summary is None:
        return None

    rows = []
    for svc in summary.services:
        rows.append(
            {
                "service": svc.name,
                "severity": svc.severity,
                "status": _severity_to_emoji(svc.severity),
                "available": svc.available,
                "summary": svc.summary,
            }
        )
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.set_index("service", inplace=True)
    return df


def _read_dashboard_log_tail(max_bytes: int = 40_000) -> str:
    """
    קורא את זנב קובץ הלוג של הדשבורד (dashboard_app.log) בצורה בטוחה.

    - אם הקובץ לא קיים → מחזיר הודעה ידידותית.
    - אם יש בעיית IO → מחזיר הודעת שגיאה ידידותית.
    """
    path = DASHBOARD_LOG_PATH
    try:
        if not path.exists():
            return f"[log] קובץ הלוג עדיין לא נוצר ({path})."

        size = path.stat().st_size
        offset = max(0, size - max_bytes)

        with path.open("rb") as f:
            if offset:
                f.seek(offset)
                # דילוג על שורת חיתוך חלקית
                _ = f.readline()
            data = f.read().decode("utf-8", errors="replace")

        return data or "[log] אין עדיין תוכן לוג להצגה."
    except Exception as exc:  # pragma: no cover
        return f"[log] כשל בקריאת קובץ הלוג: {exc}"


def _render_logs_internal_fallback(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
) -> None:
    """
    מימוש פנימי מתקדם לטאב Logs / System Health, כאשר אין מודול ייעודי.

    מציג:
    -----
    1. כותרת ראשית + סטטוס כללי (has_critical_issues / has_warnings).
    2. טבלת שירותים (ServiceHealthSnapshot) כולל severity.
    3. Telemetry:
        - Tab timings (זמן ריצה אחרון/ממוצע/מס' ריצות).
        - Tab errors (שגיאות per-tab).
        - Navigation history tail.
    4. Viewer לזנב קובץ הלוג dashboard_app.log.
    5. Agent context snapshot (אם רלוונטי).
    """
    # ננסה להשתמש ב-Runtime כדי לקבל Summary עשיר
    try:
        runtime = ensure_dashboard_runtime(app_ctx)
    except Exception:  # pragma: no cover
        runtime = None

    summary: Optional[DashboardSummary] = None

    if runtime is not None:
        # נעדיף summary שנשמר ב-session, אם קיים
        try:
            raw_summary = st.session_state.get(SESSION_KEY_DASHBOARD_SUMMARY)
        except Exception:
            raw_summary = None

        if isinstance(raw_summary, Mapping):
            # dict מוכן – ננסה לבנות ממנו DF של שירותים
            try:
                df_services = None
                if "services" in raw_summary and isinstance(raw_summary["services"], list):
                    rows = []
                    for svc in raw_summary["services"]:
                        if not isinstance(svc, Mapping):
                            continue
                        rows.append(
                            {
                                "service": svc.get("name"),
                                "severity": svc.get("severity"),
                                "status": _severity_to_emoji(str(svc.get("severity", ""))),
                                "available": bool(svc.get("available")),
                                "summary": svc.get("summary"),
                            }
                        )
                    if rows:
                        df_services = pd.DataFrame(rows).set_index("service")
                if df_services is not None:
                    services_df = df_services
                else:
                    summary = build_dashboard_summary(runtime)
                    services_df = _build_services_df(summary)
            except Exception:
                summary = build_dashboard_summary(runtime)
                services_df = _build_services_df(summary)
        else:
            # אין summary ב-session_state – נבנה Summary טרי
            summary = build_dashboard_summary(runtime)
            services_df = _build_services_df(summary)
    else:
        services_df = None

    st.markdown(
        """
<div style="
    background:linear-gradient(90deg,#212121 0%,#37474F 100%);
    border-radius:10px;padding:14px 20px;margin-bottom:14px;
    box-shadow:0 2px 6px rgba(33,33,33,0.22);
">
    <div style="font-size:1.15rem;font-weight:800;color:white;letter-spacing:-0.2px;">
        🧾 System Health &amp; Logs
    </div>
    <div style="font-size:0.76rem;color:rgba(255,255,255,0.74);margin-top:3px;">
        Service health · Tab telemetry · Log viewer · Agent context · Dashboard diagnostics
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # --- Summary headline ---
    if summary is not None:
        headline = []
        if summary.has_critical_issues:
            headline.append("🚨 **Critical issues detected**")
        if summary.has_warnings and not summary.has_critical_issues:
            headline.append("⚠️ **Warnings present**")
        if not headline:
            headline.append("✅ System health looks **OK**")

        st.markdown("  \n".join(headline))
        st.caption(
            f"env=`{summary.env}`, profile=`{summary.profile}`, "
            f"run_id=`{summary.run_id}`, app=`{summary.app_name} v{summary.version}`"
        )
    else:
        st.info("לא הצלחנו לבנות Summary מלא; מציג רק מידע בסיסי מהמערכת.")

    # --- Services table ---
    st.markdown("#### 🧩 Services health")

    if services_df is not None:
        st.dataframe(
            services_df,
            use_container_width=True,
        )
    else:
        st.write("אין נתוני שירותים מלאים; בדוק את SqlStore / Runtime.")

    # --- Telemetry: Tab timings / Tab errors / Nav history ---
    st.markdown("#### ⏱ Telemetry & diagnostics")

    col_left, col_mid, col_right = st.columns(3)

    # Tab timings
    with col_left:
        st.markdown("**Tab timings**")
        timings = _collect_session_tab_timings()
        if not timings:
            st.caption("אין עדיין מדידות זמן לטאבים.")
        else:
            rows = []
            for key, rec in timings.items():
                rows.append(
                    {
                        "tab": key,
                        "last_s": rec.get("last", 0.0),
                        "avg_s": rec.get("avg", 0.0),
                        "count": rec.get("count", 0),
                    }
                )
            df_timings = pd.DataFrame(rows).set_index("tab")
            st.dataframe(df_timings, use_container_width=True)

    # Tab errors
    with col_mid:
        st.markdown("**Tab errors**")
        errors = _collect_session_tab_errors()
        if not errors:
            st.caption("אין שגיאות שנשמרו ברמת הטאבים.")
        else:
            err_rows = []
            for key, info in errors.items():
                err_rows.append(
                    {
                        "tab": key,
                        "ts_utc": info.get("ts_utc"),
                        "exc_type": info.get("exc_type"),
                        "message": info.get("message"),
                    }
                )
            df_errors = pd.DataFrame(err_rows).set_index("tab")
            st.dataframe(df_errors, use_container_width=True)

    # Nav history
    with col_right:
        st.markdown("**Navigation history (tail)**")
        nav_tail = _collect_session_nav_history_tail(limit=50)
        if not nav_tail:
            st.caption("אין עדיין היסטוריית ניווט.")
        else:
            df_nav = pd.DataFrame(nav_tail)
            st.dataframe(df_nav, use_container_width=True)

    # --- External full-system health check (optional hook) ---
    st.markdown("#### 🧬 Full system health check (external module, optional)")

    try:
        import importlib.util as _hc_util  # local import כדי לא להפריע לייצוא
        spec = _hc_util.find_spec("health_check_full_system")
    except Exception:
        spec = None

    if spec is None:
        st.caption(
            "No `health_check_full_system.py` module found on PYTHONPATH/PROJECT_ROOT – "
            "skipping external health check."
        )
    else:
        try:
            hc_module = importlib.import_module("health_check_full_system")
            candidate_names = [
                "run_full_health_check",
                "run_health_check",
                "main",
            ]
            fn = None
            for name in candidate_names:
                cand = getattr(hc_module, name, None)
                if callable(cand):
                    fn = cand
                    break

            if fn is None:
                st.caption(
                    "Found `health_check_full_system.py` but no callable entrypoint "
                    "(expected one of: run_full_health_check / run_health_check / main)."
                )
            else:
                with st.spinner("Running external full-system health check..."):
                    result = fn()

                st.caption("External health check result:")
                if isinstance(result, Mapping):
                    st.json(result)
                elif isinstance(result, (list, tuple)):
                    st.write(result)
                else:
                    st.write(result)
        except Exception as exc:  # pragma: no cover
            st.warning(
                f"External health_check_full_system failed: {exc}"
            )
            logger.warning(
                "health_check_full_system external hook failed: %s", exc, exc_info=True
            )

    # --- Log file viewer ---
    st.markdown("#### 📁 Dashboard log file (tail)")

    log_text = _read_dashboard_log_tail()
    with st.expander(f"View log tail: {DASHBOARD_LOG_PATH.name}", expanded=False):
        st.code(log_text, language="text")

    # --- Agent context snapshot ---
    if feature_flags.get("show_debug_info"):
        st.markdown("#### 🤖 Agent context snapshot (debug)")
        try:
            agent_ctx = st.session_state.get(SESSION_KEY_AGENT_CONTEXT)
        except Exception:
            agent_ctx = None

        if agent_ctx is None and runtime is not None:
            agent_ctx = update_agent_context_in_session(runtime)

        st.json(agent_ctx or {})

    # --- Log level distribution analysis ---
    st.markdown("#### 📊 Log level & error distribution")
    try:
        _raw_log = _read_dashboard_log_tail(n_lines=500)
        if _raw_log:
            import re as _re_log
            import plotly.graph_objects as _go_log
            _level_counts = {"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}
            for _line in _raw_log.splitlines():
                for _lvl in _level_counts:
                    if f" {_lvl} " in _line or f"[{_lvl}]" in _line:
                        _level_counts[_lvl] += 1
                        break
            _total_log_lines = len(_raw_log.splitlines())
            _col_lvl, _col_dl = st.columns([3, 1])
            with _col_lvl:
                _colors_log = {"DEBUG": "#78909C", "INFO": "#42A5F5", "WARNING": "#FFA726", "ERROR": "#EF5350", "CRITICAL": "#B71C1C"}
                _fig_lvl = _go_log.Figure(
                    _go_log.Bar(
                        x=list(_level_counts.keys()),
                        y=list(_level_counts.values()),
                        marker_color=[_colors_log.get(k, "#90CAF9") for k in _level_counts],
                        text=list(_level_counts.values()),
                        textposition="outside",
                    )
                )
                _fig_lvl.update_layout(
                    title=f"Log levels in last {_total_log_lines} lines",
                    height=260,
                    margin=dict(l=10, r=10, t=40, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ECEFF1"),
                )
                st.plotly_chart(_fig_lvl, use_container_width=True)
            with _col_dl:
                st.download_button(
                    label="⬇ Download full log",
                    data=_raw_log,
                    file_name="dashboard_app.log",
                    mime="text/plain",
                    key="log_download_btn",
                )
                _err_pct = 100.0 * (_level_counts["ERROR"] + _level_counts["CRITICAL"]) / max(_total_log_lines, 1)
                st.metric("Error rate", f"{_err_pct:.1f}%", help="(ERROR+CRITICAL) / total lines")
                st.metric("Warnings", _level_counts["WARNING"])
        else:
            st.caption("Log file is empty or not yet written.")
    except Exception as _le:
        st.caption(f"Log level analysis failed: {_le}")

    # --- Session performance profiler ---
    st.markdown("#### 🏎 Session performance profiler")
    try:
        import plotly.graph_objects as _go_prof
        _timings_prof = _collect_session_tab_timings()
        _errors_prof = _collect_session_tab_errors()
        if _timings_prof:
            _t_rows_prof = sorted(
                [
                    {
                        "tab": k,
                        "last_s": v.get("last", 0.0),
                        "avg_s": v.get("avg", 0.0),
                        "runs": v.get("count", 0),
                    }
                    for k, v in _timings_prof.items()
                ],
                key=lambda x: x["avg_s"],
                reverse=True,
            )
            _col_p1, _col_p2 = st.columns([3, 1])
            with _col_p1:
                _tabs_s = [r["tab"] for r in _t_rows_prof]
                _avgs_s = [r["avg_s"] for r in _t_rows_prof]
                _bar_c = [
                    "#EF5350" if a > 3.0 else "#FFA726" if a > 1.0 else "#66BB6A"
                    for a in _avgs_s
                ]
                _fig_prof = _go_prof.Figure(
                    _go_prof.Bar(
                        y=_tabs_s,
                        x=_avgs_s,
                        orientation="h",
                        marker_color=_bar_c,
                        text=[f"{a:.2f}s" for a in _avgs_s],
                        textposition="outside",
                    )
                )
                _fig_prof.update_layout(
                    title="Avg render time per tab (🟢<1s  🟠<3s  🔴≥3s)",
                    height=max(250, 48 * len(_tabs_s)),
                    margin=dict(l=10, r=10, t=40, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ECEFF1"),
                    xaxis_title="seconds",
                )
                st.plotly_chart(_fig_prof, use_container_width=True)
            with _col_p2:
                _total_runs_p = sum(r["runs"] for r in _t_rows_prof)
                st.metric("Total tab renders", _total_runs_p)
                st.metric("Tabs with errors", len(_errors_prof))
                _slowest = _t_rows_prof[0] if _t_rows_prof else None
                if _slowest:
                    st.metric(
                        "Slowest (avg)",
                        f"{_slowest['avg_s']:.2f}s",
                        help=_slowest["tab"],
                    )
        else:
            st.caption("No tab timing data yet — open a few tabs first.")
    except Exception as _pe:
        st.caption(f"Session profiler failed: {_pe}")

    # --- Export diagnostics ---
    st.markdown("#### 📤 Export diagnostics")
    try:
        import json as _json_diag
        _session_keys_d = [k for k in st.session_state.keys() if not str(k).startswith("_")]
        _session_types = {k: type(st.session_state[k]).__name__ for k in _session_keys_d[:60]}
        _diag_data = _json_diag.dumps(
            {
                "session_key_types": _session_types,
                "tab_timings": _collect_session_tab_timings(),
                "tab_errors": {
                    k: {kk: str(vv) for kk, vv in v.items()}
                    for k, v in _collect_session_tab_errors().items()
                },
            },
            indent=2,
            default=str,
        )
        st.download_button(
            label="⬇ Download session diagnostics JSON",
            data=_diag_data,
            file_name="dashboard_session_diagnostics.json",
            mime="application/json",
            key="session_diag_download_btn",
        )
        st.caption(f"Session holds {len(_session_keys_d)} state keys (first 60 exported)")
    except Exception as _de:
        st.caption(f"Diagnostics export failed: {_de}")


def render_logs_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    📜 Logs / System Health – גרסה מתקדמת:

    סדר עדיפות:
    -----------
    1. אם קיים מודול ייעודי (logs_tab/system_health_tab) עם render_logs_tab / render_system_health_tab / render_tab –
       נריץ אותו קודם (תאימות מלאה לקוד קיים).
    2. אחרי זה, תמיד נוסיף מתחתיו את ה-fallback הפנימי (_render_logs_internal_fallback)
       שמציג:
       - Service health
       - Telemetry (timings/errors/nav history)
       - לוגים של dashboard_app.log
       - Agent context (במצב debug).
    """
    _log_tab_entry(TAB_KEY_LOGS, feature_flags, nav_payload)

    # 1) ניסיון להריץ מודול ייעודי, אם קיים
    module = _find_module(
        (
            "logs_tab",
            "root.logs_tab",
            "system_health_tab",
            "root.system_health_tab",
        )
    )
    if module is not None:
        fn = _find_tab_function_in_module(
            module,
            (
                "render_logs_tab",
                "render_system_health_tab",
                "render_tab",
            ),
        )
        if fn is not None:
            try:
                _invoke_tab_function("logs", fn, app_ctx, feature_flags, nav_payload)
            except Exception as exc:  # pragma: no cover
                logger.error(
                    "External logs_tab renderer raised %s – falling back to internal view.",
                    exc,
                    exc_info=True,
                )

    # 2) fallback פנימי – תמיד מוצג מתחת (גם אם היה מודול חיצוני)
    st.markdown("---")
    _render_logs_internal_fallback(app_ctx, feature_flags)


# עדכון __all__ עבור חלק 23
try:
    __all__ += [
        "_severity_to_emoji",
        "_build_services_df",
        "_read_dashboard_log_tail",
        "_render_logs_internal_fallback",
        "render_logs_tab",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_severity_to_emoji",
        "_build_services_df",
        "_read_dashboard_log_tail",
        "_render_logs_internal_fallback",
        "render_logs_tab",
    ]

# =====================
