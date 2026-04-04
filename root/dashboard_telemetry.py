# -*- coding: utf-8 -*-
"""
root/dashboard_telemetry.py — Dashboard Summary & Telemetry
=============================================================

Extracted from dashboard.py Part 20/35.

Contains:
- DashboardSummary dataclass (runtime snapshot for agents/export)
- TabUsageStats, ServiceHealthSnapshot dataclasses
- build_dashboard_summary() — main entry point
- Session persistence helpers
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Import shared deps from dashboard
try:
    from root.dashboard import (
        _make_json_safe,
        _cache_get,
        _cache_set,
        _collect_session_tab_timings,
        build_active_tabs,
        APP_NAME,
        APP_VERSION,
        RUNTIME_HOST,
        RUNTIME_USER,
    )
except (ImportError, AttributeError):
    _make_json_safe = lambda x: x
    _cache_get = lambda ns, key: None
    _cache_set = lambda ns, key, val, ttl=60: None
    _collect_session_tab_timings = lambda: {}
    build_active_tabs = lambda reg, ff: []
    APP_NAME = "PairsTrading"
    APP_VERSION = "?"
    RUNTIME_HOST = "localhost"
    RUNTIME_USER = "unknown"

# Forward reference types
DashboardRuntime = Any
EnvName = str
ProfileName = str
TabKey = str
ServiceStatus = Dict[str, Any]
FeatureFlags = Dict[str, Any]

# Part 20/35 – DashboardSummary & Telemetry Models (HF-grade analytics core)
# =====================

SESSION_KEY_DASHBOARD_SUMMARY: str = "dashboard_summary"


@dataclass
class TabUsageStats:
    """
    סטטיסטיקות שימוש/ביצועים לטאב אחד (ברמת קרן).

    שדות:
    -----
    key:
        מפתח הטאב (TabKey) – למשל "home", "backtest", "risk".
    label:
        הטקסט/אימוג'י שמוצג ב-UI – למשל "🏠 Dashboard".
    group:
        קבוצה לוגית ("core", "research", "risk", "macro", "system").
    enabled:
        האם הטאב פעיל בפרופיל הנוכחי (לא רק ב-Registry).
    render_count:
        כמה פעמים נמדדו זמני ריצה לטאבים (count מתוך SESSION_KEY_TAB_TIMINGS).
    last_render_time:
        זמן הריצה האחרון שנמדד (שניות).
    avg_render_time:
        זמן ריצה ממוצע (running average) מאז התחלת הסשן.
    total_render_time:
        זמן ריצה מצטבר (approx) ≈ avg * count.
    utilization_share:
        חלק יחסי מסך זמן הריצה של כל הטאבים (0–1).
    perf_bucket:
        דירוג ביצועים: "fast" / "medium" / "slow" – לפי avg_render_time.
    """

    key: TabKey
    label: TabLabel
    group: str
    enabled: bool
    render_count: int = 0
    last_render_time: float = 0.0
    avg_render_time: float = 0.0
    total_render_time: float = 0.0
    utilization_share: float = 0.0
    perf_bucket: Literal["fast", "medium", "slow"] = "fast"


@dataclass
class ServiceHealthSnapshot:
    """
    תמונת Health של שירות אחד (SqlStore/Broker/Macro/Risk וכו').

    שדות:
    -----
    name:
        שם השירות (sql_store, broker, macro_engine, risk_engine, ...).
    available:
        האם השירות זמין לוגית (based on capabilities + discovery).
    severity:
        רמת חומרה:
            - "ok"       → הכל תקין.
            - "warning"  → עניינים שדורשים תשומת לב אבל לא חוסמים.
            - "error"    → תקלה קריטית (למשל Broker לא מחובר ב-LIVE).
    summary:
        תיאור קצר (אנושי) למה המצב כרגע.
    details:
        dict חופשי עם שדות רלוונטיים (mode, backend, kill_switch, alerts וכו').
    group:
        קבוצה לוגית של השירות: "core" / "engine" / "data" / "aux".
    tags:
        רשימת טאגים קצרים (למשל ["critical", "live-only"]).
    """

    name: str
    available: bool
    severity: Literal["ok", "warning", "error"]
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    group: str = "core"
    tags: List[str] = field(default_factory=list)


@dataclass
class DashboardSummary:
    """
    סיכום מרוכז של מצב הדשבורד ברמת קרן גידור.

    שימושים:
    --------
    - Desktop Bridge (Qt) להצגת Health, services וטאבים.
    - Agents / Supervisors שמחליטים איפה להתמקד (ביצועים, תקלות).
    - Telemetry / לוגים מתקדמים (SqlStore או שירות חיצוני).

    שדות:
    -----
    env / profile / run_id:
        הקונטקסט הלוגי של הסשן.
    app_name / version / host / user:
        מטא-דאטה בסיסי.
    active_tabs:
        רשימת TabUsageStats לטאבים הפעילים בפרופיל הנוכחי.
    services:
        רשימת ServiceHealthSnapshot – health של שירותים מרכזיים.
    has_critical_issues:
        האם יש לפחות Service אחד עם severity="error".
    has_warnings:
        האם יש לפחות Service אחד עם severity="warning".
    num_services:
        מספר השירותים הכולל בסיכום.
    num_tabs:
        מספר הטאבים הפעילים בסשן.
    severity_counts:
        מיפוי חומרה → מספר שירותים, למשל {"ok": 5, "warning": 2, "error": 1}.
    tabs_by_group:
        מיפוי group → מספר טאבים (core/research/macro/risk/system).
    degraded_services:
        רשימת שמות השירותים ב-severity != "ok".
    slow_tabs:
        רשימת מפתחות טאבים שסווגו כ-"slow".
    heavy_tabs:
        רשימת מפתחות טאבים הכבדים ביותר (למשל top 3 לפי avg_render_time).
    service_coverage_ratio:
        יחס כיסוי שירותים: available / num_services (0–1).
    tabs_coverage_ratio:
        יחס כיסוי טאבים: active / total_defined_tabs (0–1).
    """

    env: EnvName
    profile: ProfileName
    run_id: str
    app_name: str
    version: str
    host: str
    user: str
    active_tabs: List[TabUsageStats]
    services: List[ServiceHealthSnapshot]
    has_critical_issues: bool
    has_warnings: bool
    num_services: int
    num_tabs: int
    severity_counts: Dict[str, int]
    tabs_by_group: Dict[str, int]
    degraded_services: List[str]
    slow_tabs: List[TabKey]
    heavy_tabs: List[TabKey]
    service_coverage_ratio: float
    tabs_coverage_ratio: float


# -------------------------------------------------------------------
# 20.1 – Service health derivation from runtime.services_status
# -------------------------------------------------------------------

def _classify_service_group(name: str) -> str:
    """
    מסווג שירות לקבוצה לוגית: core / engine / data / aux.
    """
    name = name.lower()
    if name in ("sql_store", "broker"):
        return "core"
    if name in ("market_data",):
        return "data"
    if name in ("risk_engine", "macro_engine", "agents", "fair_value"):
        return "engine"
    if name in ("backtester", "optimizer", "meta_optimizer"):
        return "engine"
    return "aux"


def _build_service_health_from_status(
    runtime: DashboardRuntime,
) -> List[ServiceHealthSnapshot]:
    """
    ממפה את services_status של ה-Runtime ל-List[ServiceHealthSnapshot]
    עם לוגיקת חומרה (severity) ברמת קרן גידור.

    לוגיקה לדוגמה:
    --------------
    - Broker:
        * env="live" & (not available or not connected) → error.
        * env in {paper, backtest} & not available → warning.
    - Risk Engine:
        * env="live" & not available → error.
        * env in {paper, backtest} & not available → warning.
    - SqlStore:
        * not available → warning (אין Persist).
    - Macro Engine:
        * profile="macro" & not available → warning.
    - Agents / Fair Value / Backtester / Optimizer → warnings רכים יותר.
    """
    svc = runtime.services_status
    caps = runtime.capabilities
    env = runtime.env
    profile = runtime.profile

    snapshots: List[ServiceHealthSnapshot] = []

    # Helper פנימי לקיצור
    def add_snapshot(
        name: str,
        status: ServiceStatus,
        default_summary: str,
        default_severity: Literal["ok", "warning", "error"] = "ok",
        extra_tags: Optional[Sequence[str]] = None,
    ) -> None:
        available = bool(status.get("available"))
        severity = default_severity
        summary = default_summary
        details: Dict[str, Any] = dict(status)
        group = _classify_service_group(name)
        tags: List[str] = list(extra_tags or [])

        # בסיס: אם לא available אבל default_severity=="ok" → לפחות warning
        if not available and severity == "ok":
            severity = "warning"
            summary = "Service not available."

        snapshots.append(
            ServiceHealthSnapshot(
                name=name,
                available=available,
                severity=severity,
                summary=summary,
                details=details,
                group=group,
                tags=tags,
            )
        )

    # SqlStore
    sql = svc.get("sql_store", {}) or {}
    sql_available = bool(sql.get("available"))
    sql_summary = "SqlStore available." if sql_available else "SqlStore not available."
    add_snapshot(
        "sql_store",
        sql,
        sql_summary,
        "ok",
        extra_tags=["persist", "storage"],
    )

    # Broker
    broker = svc.get("broker", {}) or {}
    br_available = bool(broker.get("available"))
    br_connected = broker.get("connected")
    br_mode = broker.get("mode") or "N/A"

    br_severity: Literal["ok", "warning", "error"] = "ok"
    br_summary = f"Broker mode={br_mode}, connected={br_connected}."

    if env == "live":
        if (not br_available) or (br_connected is False):
            br_severity = "error"
            br_summary = "Broker unavailable or disconnected in LIVE environment."
    elif env in ("paper", "backtest"):
        if not br_available:
            br_severity = "warning"
            br_summary = "Broker not available in non-live environment."

    snapshots.append(
        ServiceHealthSnapshot(
            name="broker",
            available=br_available,
            severity=br_severity,
            summary=br_summary,
            details=dict(broker),
            group="core",
            tags=["critical", "trading"],
        )
    )

    # Market Data
    market = svc.get("market_data", {}) or {}
    md_available = bool(market.get("available"))
    md_source = market.get("source") or "N/A"
    md_latency = market.get("latency_mode") or "N/A"
    md_summary = f"Source={md_source}, latency={md_latency}."
    add_snapshot(
        "market_data",
        market,
        md_summary,
        "ok",
        extra_tags=["data"],
    )

    # Risk Engine
    risk = svc.get("risk_engine", {}) or {}
    risk_available = bool(risk.get("available"))
    risk_severity: Literal["ok", "warning", "error"] = "ok"
    risk_summary = "Risk engine available." if risk_available else "Risk engine not available."

    if env == "live" and not risk_available:
        risk_severity = "error"
        risk_summary = "Risk engine missing in LIVE environment."
    elif env in ("paper", "backtest") and not risk_available:
        risk_severity = "warning"
        risk_summary = "Risk engine missing in non-live environment."

    snapshots.append(
        ServiceHealthSnapshot(
            name="risk_engine",
            available=risk_available,
            severity=risk_severity,
            summary=risk_summary,
            details=dict(risk),
            group="engine",
            tags=["risk", "critical"],
        )
    )

    # Macro Engine
    macro = svc.get("macro_engine", {}) or {}
    macro_available = bool(macro.get("available"))
    macro_severity: Literal["ok", "warning", "error"] = "ok"
    macro_summary = "Macro engine available." if macro_available else "Macro engine not available."

    if profile == "macro" and not macro_available:
        macro_severity = "warning"
        macro_summary = "Profile='macro' but Macro engine is not available."

    snapshots.append(
        ServiceHealthSnapshot(
            name="macro_engine",
            available=macro_available,
            severity=macro_severity,
            summary=macro_summary,
            details=dict(macro),
            group="engine",
            tags=["macro"],
        )
    )

    # Agents
    agents = svc.get("agents", {}) or {}
    agents_available = bool(agents.get("available"))
    agents_severity: Literal["ok", "warning", "error"] = "ok"
    agents_summary = "Agents manager available." if agents_available else "Agents manager not available."

    snapshots.append(
        ServiceHealthSnapshot(
            name="agents",
            available=agents_available,
            severity=agents_severity,
            summary=agents_summary,
            details=dict(agents),
            group="engine",
            tags=["agents", "ai"],
        )
    )

    # Fair Value Engine
    fv = svc.get("fair_value", {}) or {}
    fv_available = bool(fv.get("available"))
    fv_summary = "Fair Value engine available." if fv_available else "Fair Value engine not available."
    add_snapshot(
        "fair_value",
        fv,
        fv_summary,
        "ok",
        extra_tags=["research", "relative_value"],
    )

    # Backtester / Optimizer / Meta-Optimizer – Health לוגי בלבד
    bt = svc.get("backtester", {}) or {}
    opt = svc.get("optimizer", {}) or {}
    mo = svc.get("meta_optimizer", {}) or {}

    add_snapshot(
        "backtester",
        bt,
        "Backtester module available." if bt.get("available") else "Backtester not available.",
        "ok",
        extra_tags=["backtest"],
    )
    add_snapshot(
        "optimizer",
        opt,
        "Optimizer module available." if opt.get("available") else "Optimizer not available.",
        "ok",
        extra_tags=["optimizer"],
    )
    add_snapshot(
        "meta_optimizer",
        mo,
        "Meta-Optimizer module available." if mo.get("available") else "Meta-Optimizer not available.",
        "ok",
        extra_tags=["optimizer", "meta"],
    )

    return snapshots


# -------------------------------------------------------------------
# 20.2 – Tab usage stats from timings + registry
# -------------------------------------------------------------------

def _build_tab_usage_stats(runtime: DashboardRuntime) -> List[TabUsageStats]:
    """
    בונה רשימת TabUsageStats עבור כל הטאבים הפעילים בפרופיל הנוכחי:

    משתמש ב:
    - runtime.tab_registry  (מטא-דאטה).
    - runtime.feature_flags (profile/env).
    - SESSION_KEY_TAB_TIMINGS (מדדי ביצועים שנאספו בפועל).

    כמו כן:
    - מחשב total_render_time ≈ avg * count.
    - מחשב utilization_share ביחס לכל הטאבים.
    - מסווג perf_bucket: fast / medium / slow לפי avg_render_time.
    """
    registry = runtime.tab_registry
    ff = runtime.feature_flags

    active_tabs, active_keys, _ = build_active_tabs(registry, ff)
    timings = _collect_session_tab_timings()

    stats: List[TabUsageStats] = []

    # שלב 1 – בונים את הרשימה עם total_render_time בסיסי
    for meta in active_tabs:
        key = meta.key
        timing_rec = timings.get(key, {}) or {}

        try:
            last = float(timing_rec.get("last", 0.0))
        except Exception:
            last = 0.0
        try:
            avg = float(timing_rec.get("avg", 0.0))
        except Exception:
            avg = 0.0
        try:
            count = int(timing_rec.get("count", 0))
        except Exception:
            count = 0

        total = avg * float(count)

        # סיווג ביצועים גס
        if avg <= 0.2:
            bucket: Literal["fast", "medium", "slow"] = "fast"
        elif avg <= 0.7:
            bucket = "medium"
        else:
            bucket = "slow"

        stats.append(
            TabUsageStats(
                key=key,
                label=meta.label,
                group=meta.group,
                enabled=True,  # build_active_tabs כבר פילטר את הלא-פעילים
                render_count=count,
                last_render_time=last,
                avg_render_time=avg,
                total_render_time=total,
                perf_bucket=bucket,
            )
        )

    # שלב 2 – חישוב utilization_share
    total_all = sum(t.total_render_time for t in stats)
    if total_all > 0:
        for t in stats:
            t.utilization_share = t.total_render_time / total_all

    return stats


# -------------------------------------------------------------------
# 20.3 – DashboardSummary core builder (with caching)
# -------------------------------------------------------------------

def build_dashboard_summary(runtime: DashboardRuntime) -> DashboardSummary:
    """
    בונה DashboardSummary מלא מתוך DashboardRuntime:

    Flow:
    -----
    1. בודק Cache פנימי (namespace="summary", key=env|profile|run_id).
    2. אם אין Cache / פג תוקף:
        - בונה ServiceHealthSnapshot מרמת runtime.services_status.
        - בונה TabUsageStats מרמת timings + registry.
        - מחשב counts/ratios/aggregations.
    3. שומר ל-Cache לטווח קצר (5 שניות).

    הערה:
    -----
    • ל-Health המתקדם (DashboardHealth) יש לוגיקה נוספת בחלק 24,
      אבל הוא משתמש ב-Summary הזה כבסיס לשיקוף services/tabs.
    """
    env = runtime.env
    profile = runtime.profile
    run_id = runtime.run_id
    ff = runtime.feature_flags

    cache_key = f"{env}|{profile}|{run_id}"
    cached = _cache_get("summary", cache_key)
    if isinstance(cached, DashboardSummary):
        return cached

    app_name = ff.get("app_name", APP_NAME)
    version = ff.get("version", APP_VERSION)
    host = ff.get("host", RUNTIME_HOST)
    user = ff.get("user", RUNTIME_USER)

    # שירותים
    services = _build_service_health_from_status(runtime)
    num_services = len(services)

    severity_counts: Dict[str, int] = {"ok": 0, "warning": 0, "error": 0}
    degraded_services: List[str] = []

    for s in services:
        severity_counts[s.severity] = severity_counts.get(s.severity, 0) + 1
        if s.severity != "ok":
            degraded_services.append(s.name)

    has_critical = severity_counts.get("error", 0) > 0
    has_warn = severity_counts.get("warning", 0) > 0

    # כיסוי שירותים (0–1)
    available_count = sum(1 for s in services if s.available)
    service_coverage_ratio = float(available_count) / float(num_services) if num_services > 0 else 0.0

    # טאבים
    tabs_stats = _build_tab_usage_stats(runtime)
    num_tabs = len(tabs_stats)

    tabs_by_group: Dict[str, int] = {}
    slow_tabs: List[TabKey] = []
    heavy_tabs: List[TabKey] = []

    for t in tabs_stats:
        tabs_by_group[t.group] = tabs_by_group.get(t.group, 0) + 1
        if t.perf_bucket == "slow":
            slow_tabs.append(t.key)

    # heavy_tabs – שלושת הטאבים הכי כבדים לפי avg_render_time
    sorted_by_avg = sorted(tabs_stats, key=lambda x: x.avg_render_time, reverse=True)
    heavy_tabs = [t.key for t in sorted_by_avg[:3]]

    # כיסוי טאבים: active / defined
    total_defined_tabs = len(runtime.tab_registry)
    tabs_coverage_ratio = (
        float(num_tabs) / float(total_defined_tabs) if total_defined_tabs > 0 else 0.0
    )

    summary = DashboardSummary(
        env=env,
        profile=profile,
        run_id=run_id,
        app_name=app_name,
        version=version,
        host=host,
        user=user,
        active_tabs=tabs_stats,
        services=services,
        has_critical_issues=has_critical,
        has_warnings=has_warn,
        num_services=num_services,
        num_tabs=num_tabs,
        severity_counts=severity_counts,
        tabs_by_group=tabs_by_group,
        degraded_services=degraded_services,
        slow_tabs=slow_tabs,
        heavy_tabs=heavy_tabs,
        service_coverage_ratio=service_coverage_ratio,
        tabs_coverage_ratio=tabs_coverage_ratio,
    )

    # Cache ל-5 שניות
    _cache_set("summary", cache_key, summary, ttl=5.0)

    return summary


# -------------------------------------------------------------------
# 20.4 – Dict export / session update / external export
# -------------------------------------------------------------------

def dashboard_summary_to_dict(summary: DashboardSummary) -> Dict[str, Any]:
    """
    ממיר DashboardSummary ל-dict JSON-friendly.

    מאפשר:
    - שמירה ב-SqlStore/Log.
    - שליחה כסטרוקטורה אחת לסוכני AI / Desktop / REST.
    """
    return {
        "env": summary.env,
        "profile": summary.profile,
        "run_id": summary.run_id,
        "app_name": summary.app_name,
        "version": summary.version,
        "host": summary.host,
        "user": summary.user,
        "active_tabs": [
            {
                "key": t.key,
                "label": t.label,
                "group": t.group,
                "enabled": t.enabled,
                "render_count": t.render_count,
                "last_render_time": t.last_render_time,
                "avg_render_time": t.avg_render_time,
                "total_render_time": t.total_render_time,
                "utilization_share": t.utilization_share,
                "perf_bucket": t.perf_bucket,
            }
            for t in summary.active_tabs
        ],
        "services": [
            {
                "name": s.name,
                "available": s.available,
                "severity": s.severity,
                "summary": s.summary,
                "details": s.details,
                "group": s.group,
                "tags": list(s.tags),
            }
            for s in summary.services
        ],
        "has_critical_issues": summary.has_critical_issues,
        "has_warnings": summary.has_warnings,
        "num_services": summary.num_services,
        "num_tabs": summary.num_tabs,
        "severity_counts": dict(summary.severity_counts),
        "tabs_by_group": dict(summary.tabs_by_group),
        "degraded_services": list(summary.degraded_services),
        "slow_tabs": list(summary.slow_tabs),
        "heavy_tabs": list(summary.heavy_tabs),
        "service_coverage_ratio": summary.service_coverage_ratio,
        "tabs_coverage_ratio": summary.tabs_coverage_ratio,
    }


def update_dashboard_summary_in_session(
    runtime: DashboardRuntime,
    store_as_dict: bool = True,
) -> DashboardSummary:
    """
    בונה DashboardSummary ומעדכן אותו ב-session_state:

    שדרוג:
    -------
    במקום לחשב תמיד build_dashboard_summary מהריק, אנחנו קודם כל
    משתמשים ב-compute_dashboard_health(runtime) (שכבר ממילא בונה Summary),
    וכך נהנים גם מה-Cache של Health.

    אם מסיבה כלשהי health.summary=None, נ fallback ל-build_dashboard_summary(runtime).
    """
    health = compute_dashboard_health(runtime)
    summary = health.summary or build_dashboard_summary(runtime)

    if store_as_dict:
        obj: Any = dashboard_summary_to_dict(summary)
        obj = _make_json_safe(obj)
    else:
        obj = summary  # dataclass – שימושי בצד Python

    try:
        st.session_state[SESSION_KEY_DASHBOARD_SUMMARY] = obj
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to store dashboard summary in session_state: %s", exc
        )

    logger.debug(
        "Dashboard summary updated in session_state (env=%s, profile=%s, num_services=%d, num_tabs=%d)",
        summary.env,
        summary.profile,
        summary.num_services,
        summary.num_tabs,
    )

    return summary


def export_dashboard_summary(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    Export ידידותי ל-Desktop/Agents:

    נשתמש ב-compute_dashboard_health(runtime) כדי להימנע מחישוב כפול של Summary.
    """
    health = compute_dashboard_health(runtime)
    summary = health.summary or build_dashboard_summary(runtime)
    payload = dashboard_summary_to_dict(summary)
    return _make_json_safe(payload)


# עדכון __all__ עבור חלק 20
try:
    __all__ += [
        "SESSION_KEY_DASHBOARD_SUMMARY",
        "TabUsageStats",
        "ServiceHealthSnapshot",
        "DashboardSummary",
        "build_dashboard_summary",
        "dashboard_summary_to_dict",
        "update_dashboard_summary_in_session",
        "export_dashboard_summary",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_DASHBOARD_SUMMARY",
        "TabUsageStats",
        "ServiceHealthSnapshot",
        "DashboardSummary",
        "build_dashboard_summary",
        "dashboard_summary_to_dict",
        "update_dashboard_summary_in_session",
        "export_dashboard_summary",
    ]


# =====================
