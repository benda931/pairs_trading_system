# -*- coding: utf-8 -*-
"""
root/dashboard_metrics.py — Dashboard Overview Metrics
========================================================

Extracted from dashboard.py Part 33/35.

Contains:
- OverviewMetric dataclass
- build_dashboard_overview_metrics() — builds all dashboard cards
- Session persistence helpers
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Import deps from sibling modules
try:
    from root.dashboard import (
        _make_json_safe,
        _cache_get,
        _cache_set,
        _discover_services_mapping,
        _probe_service,
    )
except (ImportError, AttributeError):
    _make_json_safe = lambda x: x
    _cache_get = lambda ns, key: None
    _cache_set = lambda ns, key, val, ttl=60: None
    _discover_services_mapping = lambda ctx: {}
    _probe_service = lambda ctx, m, candidates=(): (False, None)

try:
    from root.dashboard_health import compute_dashboard_health, DashboardHealth
except (ImportError, AttributeError):
    compute_dashboard_health = lambda runtime: None
    DashboardHealth = Any

try:
    from root.dashboard_telemetry import build_dashboard_summary, DashboardSummary, TabUsageStats
except (ImportError, AttributeError):
    build_dashboard_summary = lambda runtime: None
    DashboardSummary = Any
    TabUsageStats = Any

try:
    from root.dashboard_alerts_bus import get_dashboard_alerts
except (ImportError, AttributeError):
    get_dashboard_alerts = lambda limit=None: []

try:
    from root.dashboard_integrations import get_last_desktop_push_info, get_agent_actions_history_tail
except (ImportError, AttributeError):
    get_last_desktop_push_info = lambda: None
    get_agent_actions_history_tail = lambda limit=10: []

# Forward reference types
DashboardRuntime = Any

# Part 33/35 – Dashboard overview metrics (HF-grade “cards” data layer)
# =====================

SESSION_KEY_OVERVIEW_LAST: str = "dashboard_overview_last"


@dataclass
class OverviewMetric:
    """
    אבני בניין ל-“Dashboard Cards” – מטריקות מסוכמות שמיועדות לטאב הבית / Monitoring.

    דוגמאות:
    ---------
    - "system_health"    → OK / Degraded / Critical
    - "risk_status"      → OK / Warnings / Kill-switch armed
    - "macro_regime"     → Risk-On / Risk-Off / Neutral
    - "agents_status"    → Online / Disabled
    - "desktop_link"     → Connected / Not configured
    - "experiments"      → On / Off
    - "data_availability"→ Live / Delayed / Offline
    - "alerts"           → #Warnings / #Errors
    - "snapshot"         → Last snapshot time / persisted?

    שדות:
    -----
    key:
        מפתח לוגי ייחודי (system_health / risk_status / macro_regime וכו').
    label:
        תיאור ידידותי למשתמש (למשל "System health").
    value:
        ערך קצר – מוצג גדול על ה-“Card” (למשל "OK", "Critical", "Risk-On").
    level:
        רמת צבע/חומרה: "info" / "success" / "warning" / "error".
    description:
        טקסט קצר שמסביר את המצב (שורה–שתיים).
    extra:
        dict אופציונלי עם מידע נוסף למי שרוצה לצלול לעומק (Agents/Home Tab).
    icon:
        אימוג'י/אייקון לכרטיס (למשל "🩺", "⚠️", "🌐").
    category:
        קטגוריית על: "system" / "risk" / "macro" / "agents" / "data" / "trading" / "ops".
    priority:
        עדיפות לתצוגה (1 = הכי חשוב).
    value_numeric:
        ערך מספרי (אם יש) – למשל score/ratio (או None).
    unit:
        יחידת מידה לערך המספרי (%, score, alerts, וכו').
    trend:
        מגמה משוערת: "up" / "down" / "flat" / None – כרגע best-effort.
    """

    key: str
    label: str
    value: str
    level: Literal["info", "success", "warning", "error"]
    description: str
    extra: Dict[str, Any] = field(default_factory=dict)
    icon: str = ""
    category: str = "system"
    priority: int = 10
    value_numeric: Optional[float] = None
    unit: Optional[str] = None
    trend: Optional[Literal["up", "down", "flat"]] = None


# -------------------------------------------------------------------
# 33.1 – Helpers for interpreting health/summary into overview cards
# -------------------------------------------------------------------

def _overview_level_from_health(
    ready: bool,
    has_critical: bool,
    has_warnings: bool,
) -> Literal["info", "success", "warning", "error"]:
    """
    ממיר מצב Health כללי לרמת חומרה עבור כרטיס System Health.
    """
    if has_critical:
        return "error"
    if has_warnings or not ready:
        return "warning"
    return "success"


def _build_system_health_metric(
    runtime: DashboardRuntime,
    health: DashboardHealth,
) -> OverviewMetric:
    """
    כרטיס System Health – תמצית מצב המערכת:

    - מתבסס על:
        * ready
        * severity
        * score
        * issues/warnings
    """
    level = health.severity
    if level not in ("ok", "warning", "error"):
        level = "info"

    if health.ready and not health.has_critical_issues and not health.has_warnings:
        value = "OK"
        desc = "כל המערכות העיקריות במצב תקין."
    elif health.has_critical_issues:
        value = "CRITICAL"
        desc = "קיימות תקלות קריטיות בשירותי הליבה – מומלץ לבדוק מיד."
    elif health.has_warnings:
        value = "Degraded"
        desc = "חלק מהשירותים במצב אזהרה – המערכת עובדת אך לא מושלמת."
    else:
        value = "Unknown"
        desc = "לא ניתן לקבוע מצב Health מלא."

    # מיפוי severity→level Overview
    overview_level = {
        "ok": "success",
        "warning": "warning",
        "error": "error",
    }.get(health.severity, "info")

    extra = {
        "ready": health.ready,
        "has_critical_issues": health.has_critical_issues,
        "has_warnings": health.has_warnings,
        "issues": health.issues,
        "warnings": health.warnings,
        "score": health.score,
        "health_id": health.health_id,
    }

    return OverviewMetric(
        key="system_health",
        label="System health",
        value=value,
        level=overview_level,
        description=desc,
        extra=extra,
        icon="🩺",
        category="system",
        priority=1,
        value_numeric=health.score,
        unit="score",
        trend=None,  # ניתן להרחיב בעתיד ע"י השוואת score להיסטוריה
    )


def _build_risk_metric(
    runtime: DashboardRuntime,
    summary: DashboardSummary,
    health: DashboardHealth,
) -> OverviewMetric:
    """
    כרטיס Risk Status – מצב מנוע הסיכון / מגבלות.

    לוגיקה:
    -------
    - אם Risk Engine חסר ב-LIVE → error.
    - אם Risk Engine חסר ב-dev/backtest → warning.
    - אחרת – מתבסס על ServiceHealthSnapshot של risk_engine.
    """
    caps = runtime.capabilities
    svc_map = {svc.name: svc for svc in summary.services}
    risk = svc_map.get("risk_engine")

    if not caps.get("risk_engine", False) or risk is None:
        return OverviewMetric(
            key="risk_status",
            label="Risk status",
            value="Offline",
            level="warning",
            description="Risk Engine אינו זמין – מגבלות סיכון לא נאכפות אוטומטית.",
            extra={"risk_available": False},
            icon="⚠️",
            category="risk",
            priority=2,
        )

    if risk.severity == "error":
        value = "Error"
        level: Literal["info", "success", "warning", "error"] = "error"
        desc = "Risk Engine דיווח על תקלה – בדוק מגבלות, חשיפות ו-Kill-switch."
    elif risk.severity == "warning":
        value = "Warning"
        level = "warning"
        desc = "Risk Engine פעיל אך במצב אזהרה (למשל חריגות מגבלה)."
    else:
        value = "OK"
        level = "success"
        desc = "Risk Engine זמין ומדווח מצב תקין."

    extra = {
        "risk_available": risk.available,
        "severity": risk.severity,
        "summary": risk.summary,
        "details": risk.details,
    }

    return OverviewMetric(
        key="risk_status",
        label="Risk status",
        value=value,
        level=level,
        description=desc,
        extra=extra,
        icon="⚠️",
        category="risk",
        priority=2,
    )


def _build_macro_metric(
    runtime: DashboardRuntime,
    summary: DashboardSummary,
    health: DashboardHealth,
) -> OverviewMetric:
    """
    כרטיס Macro Regime – אם Macro Engine זמין.

    - שואב regime מתוך details["regime"] אם קיים.
    - ממפה ל-level: Risk-On → success, Risk-Off/Crisis → warning/error.
    """
    caps = runtime.capabilities
    svc_map = {svc.name: svc for svc in summary.services}
    macro = svc_map.get("macro_engine")

    if not caps.get("macro_engine", False) or macro is None:
        return OverviewMetric(
            key="macro_regime",
            label="Macro regime",
            value="N/A",
            level="info",
            description="Macro Engine לא זמין – אין overlay מקרו על החשיפות.",
            extra={"macro_available": False},
            icon="🌐",
            category="macro",
            priority=4,
        )

    details = macro.details or {}
    regime = details.get("regime") or "Neutral"

    regime_lower = str(regime).lower()
    if "risk-off" in regime_lower:
        level: Literal["info", "success", "warning", "error"] = "warning"
    elif "crisis" in regime_lower or "stress" in regime_lower:
        level = "error"
    elif "risk-on" in regime_lower:
        level = "success"
    else:
        level = "info"

    desc = f"Regime נוכחי לפי Macro Engine: {regime}."

    return OverviewMetric(
        key="macro_regime",
        label="Macro regime",
        value=str(regime),
        level=level,
        description=desc,
        extra={"macro_available": True, "regime": regime, "details": details},
        icon="🌐",
        category="macro",
        priority=5,
    )


def _build_agents_metric(
    runtime: DashboardRuntime,
    summary: DashboardSummary,
) -> OverviewMetric:
    """
    כרטיס: מצב סוכנים (Agents).
    """
    caps = runtime.capabilities
    svc_map = {svc.name: svc for svc in summary.services}
    agents = svc_map.get("agents")

    if not caps.get("agents_manager", False) or agents is None:
        return OverviewMetric(
            key="agents_status",
            label="Agents",
            value="Disabled",
            level="info",
            description="Agents Manager לא זמין – טאב Agents ופונקציות AI מוגבלות.",
            extra={"agents_available": False},
            icon="🤖",
            category="agents",
            priority=7,
        )

    if agents.severity == "error":
        value = "Error"
        level: Literal["info", "success", "warning", "error"] = "error"
        desc = "Agents Manager מדווח על תקלה – בדוק את הגדרות הסוכנים."
    elif agents.severity == "warning":
        value = "Degraded"
        level = "warning"
        desc = "Agents Manager פעיל אך במצב אזהרה."
    else:
        value = "Online"
        level = "success"
        desc = "Agents Manager פעיל וזמין."

    return OverviewMetric(
        key="agents_status",
        label="Agents",
        value=value,
        level=level,
        description=desc,
        extra={"agents_available": True, "severity": agents.severity},
        icon="🤖",
        category="agents",
        priority=7,
    )


def _build_desktop_metric(runtime: DashboardRuntime) -> OverviewMetric:
    """
    כרטיס: מצב אינטגרציית Desktop.
    """
    caps = runtime.capabilities
    if not caps.get("desktop_integration", False):
        return OverviewMetric(
            key="desktop_link",
            label="Desktop link",
            value="Not configured",
            level="info",
            description="Desktop Bridge לא מוגדר – אין אינטגרציה לדסקטופ כרגע.",
            extra={"desktop_integration": False},
            icon="🖥",
            category="system",
            priority=8,
        )

    last_push = get_last_desktop_push_info()
    if last_push is None:
        return OverviewMetric(
            key="desktop_link",
            label="Desktop link",
            value="Enabled",
            level="success",
            description="Desktop integration מוגדרת, אך עדיין לא בוצעה העברת State.",
            extra={"desktop_integration": True, "last_push": None},
            icon="🖥",
            category="system",
            priority=8,
        )

    return OverviewMetric(
        key="desktop_link",
        label="Desktop link",
        value="Synced",
        level="success",
        description=(
            f"Desktop Bridge קיבל State לאחרונה ב-{last_push.get('ts_utc')} "
            f"(method={last_push.get('method')})."
        ),
        extra={"desktop_integration": True, "last_push": last_push},
        icon="🖥",
        category="system",
        priority=8,
    )


def _build_experiments_metric(runtime: DashboardRuntime) -> OverviewMetric:
    """
    כרטיס: מצב “Experiment mode” (Backtests/Optimization פעילים).

    מתבסס על:
    - feature_flags["enable_experiment_mode"]
    - capabilities.backtester / optimizer / meta_optimizer
    """
    ff = runtime.feature_flags
    caps = runtime.capabilities

    experiments_enabled = bool(ff.get("enable_experiment_mode", False))
    has_backtest = bool(caps.get("backtester", False))
    has_opt = bool(caps.get("optimizer", False) or caps.get("meta_optimizer", False))

    if not experiments_enabled:
        value = "Off"
        level: Literal["info", "success", "warning", "error"] = "info"
        desc = (
            "Experiment mode כבוי – "
            "Backtests/Optimization עדיין זמינים ידנית, אך לא במצב אינטנסיבי."
        )
    else:
        value = "On"
        level = "success"
        desc = "Experiment mode פעיל – ניתן להריץ Backtests/Optimization בצורה אינטנסיבית."

    extra = {
        "enable_experiment_mode": experiments_enabled,
        "has_backtester": has_backtest,
        "has_optimizer": has_opt,
    }

    return OverviewMetric(
        key="experiments",
        label="Experiments",
        value=value,
        level=level,
        description=desc,
        extra=extra,
        icon="🧪",
        category="system",
        priority=9,
    )


# -------------------------------------------------------------------
# 33.2 – 8+ כרטיסים מתקדמים נוספים (Data, Trading, Alerts, Snapshot, Utilization)
# -------------------------------------------------------------------

def _build_data_availability_metric(runtime: DashboardRuntime, summary: DashboardSummary) -> OverviewMetric:
    """
    כרטיס: Data availability – מצב Market Data.

    לוגיקה:
    -------
    - אם אין market_data.available → Offline.
    - אחרת – לפי source + latency_mode.
    """
    svc_map = {svc.name: svc for svc in summary.services}
    md = svc_map.get("market_data")

    if md is None or not md.available:
        return OverviewMetric(
            key="data_availability",
            label="Data availability",
            value="Offline",
            level="warning",
            description="שכבת Market Data לא זמינה – אין דאטה חי/דלייד.",
            extra={"available": False},
            icon="📡",
            category="data",
            priority=3,
        )

    details = md.details or {}
    source = details.get("source") or "N/A"
    latency = (details.get("latency_mode") or "unknown").lower()

    if latency in ("live", "real-time"):
        level: Literal["info", "success", "warning", "error"] = "success"
        value = "Live"
        desc = f"נתוני שוק חיים מ-{source} ({latency})."
    elif latency in ("delayed", "15min"):
        level = "info"
        value = "Delayed"
        desc = f"נתוני שוק מושהים מ-{source} ({latency})."
    else:
        level = "warning"
        value = "Unknown"
        desc = f"מצב latency לא ידוע ({latency}) עבור מקור {source}."

    return OverviewMetric(
        key="data_availability",
        label="Data availability",
        value=value,
        level=level,
        description=desc,
        extra={"available": True, "source": source, "latency_mode": latency},
        icon="📡",
        category="data",
        priority=3,
    )


def _build_sql_persistence_metric(runtime: DashboardRuntime, summary: DashboardSummary) -> OverviewMetric:
    """
    כרטיס: Persistence / SqlStore – עד כמה המערכת באמת שומרת היסטוריה.

    מתבסס על:
    - services["sql_store"].available
    - details["backend"], details["has_history"], details["read_only"]
    """
    svc_map = {svc.name: svc for svc in summary.services}
    sql = svc_map.get("sql_store")

    if sql is None or not sql.available:
        return OverviewMetric(
            key="persistence",
            label="Persistence",
            value="No SqlStore",
            level="warning",
            description="SqlStore אינו זמין – תוצאות, לוגים ו-Snapshots לא יישמרו ב-DB.",
            extra={"available": False},
            icon="💾",
            category="system",
            priority=6,
        )

    details = sql.details or {}
    backend = details.get("backend") or "unknown"
    has_history = details.get("has_history")
    read_only = details.get("read_only")

    if read_only:
        value = "Read-only"
        level: Literal["info", "success", "warning", "error"] = "info"
        desc = f"SqlStore ({backend}) במצב read-only – לא נשמרים נתונים חדשים."
    else:
        if has_history is True:
            value = "OK"
            level = "success"
            desc = f"SqlStore ({backend}) זמין עם טבלאות history/PNL."
        elif has_history is False:
            value = "Empty"
            level = "warning"
            desc = f"SqlStore ({backend}) זמין אך ללא טבלאות history/PNL."
        else:
            value = "Unknown"
            level = "info"
            desc = f"SqlStore ({backend}) זמין – מצב היסטוריה לא ידוע."

    return OverviewMetric(
        key="persistence",
        label="Persistence",
        value=value,
        level=level,
        description=desc,
        extra={
            "available": True,
            "backend": backend,
            "has_history": has_history,
            "read_only": read_only,
        },
        icon="💾",
        category="system",
        priority=6,
    )


def _build_trading_mode_metric(runtime: DashboardRuntime, health: DashboardHealth) -> OverviewMetric:
    """
    כרטיס: Trading mode – Live / Paper / Offline, האם אפשר לסחור.

    מתבסס על:
    - runtime.env
    - health.can_trade
    - capabilities.broker / market_data_router
    """
    env = runtime.env
    caps = runtime.capabilities

    if not caps.get("broker", False):
        return OverviewMetric(
            key="trading_mode",
            label="Trading mode",
            value="No broker",
            level="warning",
            description="Broker לא מוגדר – אין אפשרות לבצע מסחר.",
            extra={"can_trade": False, "env": env},
            icon="📉",
            category="trading",
            priority=3,
        )

    if env == "live":
        value = "LIVE" if health.can_trade else "LIVE (blocked)"
        level: Literal["info", "success", "warning", "error"] = "success" if health.can_trade else "error"
        desc = (
            "מסחר LIVE אפשרי (Broker+Data זמינים)."
            if health.can_trade
            else "סביבת LIVE אך לא ניתן לסחור בפועל – בדוק Broker/Market Data/Health."
        )
    elif env == "paper":
        value = "PAPER"
        level = "info"
        desc = "סביבת Paper – מסחר מדומה/חשבון דמו."
    else:
        value = env.upper()
        level = "info"
        desc = f"סביבה הגדרתית ({env}) – ללא מסחר אמיתי."

    return OverviewMetric(
        key="trading_mode",
        label="Trading mode",
        value=value,
        level=level,
        description=desc,
        extra={"can_trade": health.can_trade, "env": env},
        icon="📉",
        category="trading",
        priority=3,
    )


def _build_alerts_metric(runtime: DashboardRuntime) -> OverviewMetric:
    """
    כרטיס: Alerts – מספר התראות ברמת דשבורד (warning/error).

    מבוסס על Dashboard Alert Bus (SESSION_KEY_ALERTS).
    """
    alerts = get_dashboard_alerts(limit=50)
    warnings_count = sum(1 for a in alerts if a.level == "warning")
    errors_count = sum(1 for a in alerts if a.level == "error")
    total = len(alerts)

    if errors_count > 0:
        level: Literal["info", "success", "warning", "error"] = "error"
        value = f"{errors_count} errors"
    elif warnings_count > 0:
        level = "warning"
        value = f"{warnings_count} warnings"
    else:
        level = "success"
        value = "0 alerts"

    desc = f"{total} alerts in bus (warnings={warnings_count}, errors={errors_count})."

    extra = {
        "total": total,
        "warnings": warnings_count,
        "errors": errors_count,
    }

    return OverviewMetric(
        key="alerts",
        label="Alerts",
        value=value,
        level=level,
        description=desc,
        extra=extra,
        icon="🚨",
        category="system",
        priority=2,
        value_numeric=float(total),
        unit="alerts",
    )


def _build_snapshot_metric(runtime: DashboardRuntime) -> OverviewMetric:
    """
    כרטיס: Snapshot status – מתי היה snapshot אחרון, האם נשמר ל-SqlStore.

    מתבסס על:
    - SESSION_KEY_LAST_SNAPSHOT_META (Part 11).
    """
    meta = _get_last_snapshot_meta()
    if not meta:
        return OverviewMetric(
            key="snapshot",
            label="Snapshots",
            value="None",
            level="info",
            description="עדיין לא נוצר snapshot של הדשבורד בסשן זה.",
            extra={"exists": False},
            icon="📸",
            category="ops",
            priority=9,
        )

    ts = meta.get("ts_utc")
    seq = meta.get("sequence")
    saved = bool(meta.get("saved_to_sql_store"))
    method_name = meta.get("sql_method")

    if saved:
        value = f"#{seq} (SQL)"
        level: Literal["info", "success", "warning", "error"] = "success"
        desc = f"Snapshot #{seq} נשמר ל-SqlStore ({method_name or 'unknown'}) בזמן {ts}."
    else:
        value = f"#{seq} (session)"
        level = "info"
        desc = f"Snapshot #{seq} נשמר רק ב-session_state בזמן {ts}."

    extra = {
        "sequence": seq,
        "ts_utc": ts,
        "saved_to_sql_store": saved,
        "sql_method": method_name,
    }

    return OverviewMetric(
        key="snapshot",
        label="Snapshots",
        value=value,
        level=level,
        description=desc,
        extra=extra,
        icon="📸",
        category="ops",
        priority=9,
    )


def _build_utilization_metric(runtime: DashboardRuntime, summary: DashboardSummary) -> OverviewMetric:
    """
    כרטיס: UI Utilization – “מי אוכל את רוב זמן הריצה”.

    מתבסס על:
    - TabUsageStats.total_render_time + utilization_share
    - slow_tabs / heavy_tabs מה-Summary
    """
    tabs = summary.active_tabs
    if not tabs:
        return OverviewMetric(
            key="ui_utilization",
            label="UI utilization",
            value="No data",
            level="info",
            description="אין עדיין מדידות זמן לטאבים.",
            extra={},
            icon="⏱",
            category="system",
            priority=10,
        )

    # כרטיס יציג את הטאב הכי כבד + חלק יחסי
    heaviest = max(tabs, key=lambda t: t.total_render_time)
    share = heaviest.utilization_share

    # רמת חומרה לפי share וזמן ממוצע
    if heaviest.perf_bucket == "slow" and share > 0.4:
        level: Literal["info", "success", "warning", "error"] = "warning"
    else:
        level = "info"

    value = heaviest.key
    desc = (
        f"טאב `{heaviest.key}` צורך ~{share * 100:.1f}% מזמן הריצה המצטבר "
        f"(avg≈{heaviest.avg_render_time:.3f}s)."
    )

    extra = {
        "heaviest_tab": heaviest.key,
        "heaviest_share": share,
        "avg_render_time": heaviest.avg_render_time,
        "perf_bucket": heaviest.perf_bucket,
        "slow_tabs": summary.slow_tabs,
        "heavy_tabs": summary.heavy_tabs,
    }

    return OverviewMetric(
        key="ui_utilization",
        label="UI utilization",
        value=value,
        level=level,
        description=desc,
        extra=extra,
        icon="⏱",
        category="system",
        priority=10,
        value_numeric=share,
        unit="share",
    )


def _build_agent_activity_metric(runtime: DashboardRuntime) -> OverviewMetric:
    """
    כרטיס: Agent activity – כמה פעולות Agents היו לאחרונה.

    מתבסס על:
    - get_agent_actions_history_tail(limit=100)
    """
    tail = get_agent_actions_history_tail(limit=100)
    total = len(tail)
    if total == 0:
        return OverviewMetric(
            key="agent_activity",
            label="Agent activity",
            value="Idle",
            level="info",
            description="אין פעולות Agents שנרשמו בסשן זה.",
            extra={"count": 0},
            icon="🤖",
            category="agents",
            priority=9,
        )

    # ננסה להוציא source/action פופולריים
    sources: Dict[str, int] = {}
    actions: Dict[str, int] = {}
    last_ts = None
    for item in tail:
        src = str(item.get("source") or "agent")
        act = str(item.get("action") or "unknown")
        sources[src] = sources.get(src, 0) + 1
        actions[act] = actions.get(act, 0) + 1
        last_ts = item.get("ts_utc") or last_ts

    top_source = max(sources.items(), key=lambda x: x[1])[0] if sources else "agent"
    top_action = max(actions.items(), key=lambda x: x[1])[0] if actions else "unknown"

    value = f"{total} actions"
    level: Literal["info", "success", "warning", "error"] = "info"
    desc = f"{total} פעולות Agents בסשן (source מוביל: {top_source}, action מוביל: {top_action})."

    extra = {
        "count": total,
        "sources": sources,
        "actions": actions,
        "last_ts_utc": last_ts,
    }

    return OverviewMetric(
        key="agent_activity",
        label="Agent activity",
        value=value,
        level=level,
        description=desc,
        extra=extra,
        icon="🤖",
        category="agents",
        priority=9,
        value_numeric=float(total),
        unit="actions",
    )


# -------------------------------------------------------------------
# 33.3 – Build overview metrics list (with cache)
# -------------------------------------------------------------------

def build_dashboard_overview_metrics(runtime: DashboardRuntime) -> List[OverviewMetric]:
    """
    בונה רשימת OverviewMetric – "תוכן" לכרטיסי Overview בדשבורד (Home / Monitoring).

    משתמש ב:
    ---------
    - DashboardHealth (compute_dashboard_health)
    - DashboardSummary (build_dashboard_summary)
    - capabilities (Risk/Macro/Agents/Desktop/Experiments/Data)
    - Alert Bus / Snapshot Meta / Agent Actions

    Cache:
    ------
    - namespace="overview_metrics"
    - key=env|profile|run_id
    - TTL≈5 שניות
    """
    cache_key = f"{runtime.env}|{runtime.profile}|{runtime.run_id}"
    cached = _cache_get("overview_metrics", cache_key)
    if isinstance(cached, list) and all(isinstance(x, OverviewMetric) for x in cached):
        return cached  # type: ignore[return-value]

    health = compute_dashboard_health(runtime)
    summary = health.summary or build_dashboard_summary(runtime)

    metrics: List[OverviewMetric] = []

    # כרטיסי ליבה
    metrics.append(_build_system_health_metric(runtime, health))
    metrics.append(_build_risk_metric(runtime, summary, health))
    metrics.append(_build_trading_mode_metric(runtime, health))
    metrics.append(_build_data_availability_metric(runtime, summary))

    # מקרו/Agents/Experiments
    metrics.append(_build_macro_metric(runtime, summary, health))
    metrics.append(_build_agents_metric(runtime, summary))
    metrics.append(_build_experiments_metric(runtime))
    metrics.append(_build_desktop_metric(runtime))

    # Persistence / Alerts / Snapshot / Utilization / Agent activity
    metrics.append(_build_sql_persistence_metric(runtime, summary))
    metrics.append(_build_alerts_metric(runtime))
    metrics.append(_build_snapshot_metric(runtime))
    metrics.append(_build_utilization_metric(runtime, summary))
    metrics.append(_build_agent_activity_metric(runtime))

    # נרצה למיין לפי priority (ואולי key כשובר שוויון)
    metrics.sort(key=lambda m: (m.priority, m.key))

    _cache_set("overview_metrics", cache_key, metrics, ttl=5.0)
    return metrics


# -------------------------------------------------------------------
# 33.4 – Dict export / session update / external export
# -------------------------------------------------------------------

def dashboard_overview_metrics_to_dict(
    metrics: Sequence[OverviewMetric],
) -> List[Dict[str, Any]]:
    """
    ממיר רשימת OverviewMetric ל-List[dict] JSON-friendly – מתאים ל-Agents / Desktop / REST.
    """
    out: List[Dict[str, Any]] = []
    for m in metrics:
        out.append(
            {
                "key": m.key,
                "label": m.label,
                "value": m.value,
                "level": m.level,
                "description": m.description,
                "extra": _make_json_safe(m.extra),
                "icon": m.icon,
                "category": m.category,
                "priority": m.priority,
                "value_numeric": m.value_numeric,
                "unit": m.unit,
                "trend": m.trend,
            }
        )
    return out


def update_dashboard_overview_in_session(runtime: DashboardRuntime) -> List[Dict[str, Any]]:
    """
    מחשב את ה-OverviewMetrics, שומר אותם ב-session_state, ומחזיר אותם כ-List[dict].

    מיועד לשימוש:
    -------------
    - Home Tab (Dashboard Home v2) – כדי לבנות Cards.
    - Monitoring Tab עתידי.
    - Agents Tab – לתת לסוכנים “תמונה מנטלית” של המערכת.
    """
    metrics = build_dashboard_overview_metrics(runtime)
    metrics_dicts = dashboard_overview_metrics_to_dict(metrics)

    try:
        st.session_state[SESSION_KEY_OVERVIEW_LAST] = metrics_dicts
    except Exception:  # pragma: no cover
        pass

    logger.debug(
        "Dashboard overview metrics updated (env=%s, profile=%s, count=%d, keys=%s)",
        runtime.env,
        runtime.profile,
        len(metrics_dicts),
        [m["key"] for m in metrics_dicts],
    )

    return metrics_dicts


def get_dashboard_overview_from_session() -> Optional[List[Dict[str, Any]]]:
    """
    מחזיר את ה-Overview Metrics האחרונים מה-Session, אם קיימים.
    """
    try:
        obj = st.session_state.get(SESSION_KEY_OVERVIEW_LAST)
    except Exception:  # pragma: no cover
        obj = None

    if not isinstance(obj, list):
        return None
    return obj


# עדכון __all__ עבור חלק 33
try:
    __all__ += [
        "SESSION_KEY_OVERVIEW_LAST",
        "OverviewMetric",
        "build_dashboard_overview_metrics",
        "dashboard_overview_metrics_to_dict",
        "update_dashboard_overview_in_session",
        "get_dashboard_overview_from_session",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_OVERVIEW_LAST",
        "OverviewMetric",
        "build_dashboard_overview_metrics",
        "dashboard_overview_metrics_to_dict",
        "update_dashboard_overview_in_session",
        "get_dashboard_overview_from_session",
    ]

# =====================
