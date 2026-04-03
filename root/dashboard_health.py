# -*- coding: utf-8 -*-
"""
root/dashboard_health.py — Dashboard Health & Readiness API
=============================================================

Extracted from dashboard.py Part 24/35.

Contains:
- DashboardHealth dataclass (health/readiness model)
- compute_dashboard_health() — main health computation
- Session persistence and export helpers
- check_dashboard_ready() — quick readiness probe
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Import shared deps
try:
    from root.dashboard import (
        _make_json_safe,
        get_app_context,
        ensure_dashboard_runtime,
    )
    from root.dashboard_telemetry import (
        build_dashboard_summary,
        DashboardSummary,
        ServiceHealthSnapshot,
    )
except ImportError:
    _make_json_safe = lambda x: x
    get_app_context = lambda: None
    ensure_dashboard_runtime = lambda ctx: None
    build_dashboard_summary = lambda runtime: None
    DashboardSummary = Any
    ServiceHealthSnapshot = Any

# Forward reference types
DashboardRuntime = Any
EnvName = str
ProfileName = str

# Part 24/35 – Readiness & Health API (HF-grade health-check + probes)
# =====================

SESSION_KEY_HEALTH_LAST: str = "dashboard_health_last"


@dataclass
class DashboardHealth:
    """
    מודל Health / Readiness ברמת קרן – מיועד לסקריפטים, Desktop ו-AI Supervisors.

    שדות:
    -----
    env / profile:
        הקונטקסט הלוגי (כמו ב-DashboardRuntime/DashboardSummary).
    ready:
        האם הדשבורד "מוכן" לשימוש:
        - אין תקלות קריטיות (critical services OK).
        - לפחות יכולת מרכזית אחת זמינה (SqlStore / Broker / Backtester / Optimizer / Macro).
    has_critical_issues:
        True אם יש לפחות Service אחד עם severity="error".
    has_warnings:
        True אם יש לפחות Service אחד עם severity="warning".
    issues:
        רשימת תיאורים אנושיים של בעיות קריטיות (כשלי Broker ב-LIVE, חסר Risk Engine וכו').
    warnings:
        רשימת תיאורים אנושיים של אזהרות (אין SqlStore, Macro חסר בפרופיל macro וכו').
    ts_utc:
        זמן יצירת ה-Health ב-UTC (isoformat).
    summary:
        DashboardSummary (אופציונלי) – תמונה מלאה של services+tabs (יכול להיות כבד).
    severity:
        רמת חומרה כוללת: "ok" / "warning" / "error".
    score:
        ציון בריאות 0–100 (100 = מושלם, 0 = שבור) – מבוסס על critical+warnings.
    missing_critical_caps:
        רשימת capabilities קריטיים שחסרים (למשל broker/risk_engine ב-LIVE).
    missing_noncritical_caps:
        רשימת capabilities משניים שחסרים (optimizer/macro_engine ב-dev וכו').
    can_trade:
        האם המערכת יכולה לבצע מסחר (env in {live,paper} + Broker + MarketData + אין critical).
    can_backtest:
        האם Backtest זמין (capabilities.backtester==True).
    can_optimize:
        האם Optimizer/Meta-Optimizer זמינים (לפחות אחד).
    can_monitor:
        האם יש יכולת Monitoring בסיסית (SqlStore או Risk Engine או Macro Engine).
    recommended_actions:
        רשימת המלצות קצרות לפעולה (למשל "Connect IBKR in LIVE", "Enable SqlStore").
    health_id:
        מזהה ייחודי של Health snapshot (env|profile|run_id|ts_utc).
    seconds_since_prev:
        כמה שניות עברו מאז Health קודם (אם יש נתון כזה), אחרת None.
    """

    env: EnvName
    profile: ProfileName
    ready: bool
    has_critical_issues: bool
    has_warnings: bool
    issues: List[str]
    warnings: List[str]
    ts_utc: str
    summary: Optional["DashboardSummary"] = None  # מוגדר בחלק 20
    severity: Literal["ok", "warning", "error"] = "ok"
    score: float = 100.0
    missing_critical_caps: List[str] = field(default_factory=list)
    missing_noncritical_caps: List[str] = field(default_factory=list)
    can_trade: bool = False
    can_backtest: bool = False
    can_optimize: bool = False
    can_monitor: bool = False
    recommended_actions: List[str] = field(default_factory=list)
    health_id: str = ""
    seconds_since_prev: Optional[float] = None


# -------------------------------------------------------------------
# 24.1 – Issues / warnings derivation from DashboardSummary
# -------------------------------------------------------------------

def _compute_health_issues_from_summary(
    runtime: DashboardRuntime,
    summary: "DashboardSummary",
) -> Tuple[List[str], List[str]]:
    """
    גוזר issues/warnings "אנושיים" מתוך DashboardSummary + Runtime:

    לוגיקה:
    -------
    - Broker ב-LIVE לא מחובר → Critical issue.
    - Risk Engine חסר ב-LIVE → Critical issue.
    - SqlStore חסר → Warning.
    - Macro Engine חסר כש-profile="macro" → Warning.
    - Agents חסרים → Warning רך (רלוונטי בעיקר לשכבת Agents).
    - מוסיף גם summaries של services עם severity=error/warning.
    """
    issues: List[str] = []
    warnings: List[str] = []

    env = runtime.env
    profile = runtime.profile

    svc_by_name: Dict[str, ServiceHealthSnapshot] = {
        svc.name: svc for svc in summary.services
    }

    broker = svc_by_name.get("broker")
    sql = svc_by_name.get("sql_store")
    risk = svc_by_name.get("risk_engine")
    macro = svc_by_name.get("macro_engine")
    agents = svc_by_name.get("agents")

    # Broker + LIVE
    if env == "live":
        if broker is None or not broker.available or broker.severity == "error":
            issues.append(
                "Broker לא זמין / לא מחובר בסביבת LIVE – המערכת לא מוכנה למסחר אמיתי."
            )

    # Risk Engine + LIVE
    if env == "live":
        if risk is None or not risk.available or risk.severity == "error":
            issues.append(
                "Risk Engine חסר או לא תקין בסביבת LIVE – אין ניהול סיכון אוטומטי."
            )

    # SqlStore – חסר Persist
    if sql is None or not sql.available:
        warnings.append(
            "SqlStore לא זמין – תוצאות, קונפיגים ו-Snapshots לא יישמרו בצורה פרסיסטנטית."
        )

    # Macro Engine – mismatch לפרופיל
    if profile == "macro" and (macro is None or not macro.available):
        warnings.append(
            "Profile='macro' אך Macro Engine לא זמין – טאב Macro יעבוד בצורה מוגבלת."
        )

    # Agents – warning רך
    if agents is None or not agents.available:
        warnings.append(
            "Agents Manager לא זמין – טאב Agents וסוכני AI לא יוכלו לפעול במלואם."
        )

    # נשלב summaries של services עם severity=error/warning
    for svc in summary.services:
        if svc.severity == "error":
            msg = f"[{svc.name}] {svc.summary}"
            if msg not in issues:
                issues.append(msg)
        elif svc.severity == "warning":
            msg = f"[{svc.name}] {svc.summary}"
            if msg not in warnings:
                warnings.append(msg)

    return issues, warnings


# -------------------------------------------------------------------
# 24.2 – Advanced health metrics: severity, score, capabilities flags
# -------------------------------------------------------------------

def _compute_health_severity(
    ready: bool,
    has_critical: bool,
    has_warnings: bool,
) -> Literal["ok", "warning", "error"]:
    """
    ממיר מצב Health כללי לרמת חומרה:

    - error   → יש לפחות critical issue אחד.
    - warning → אין critical, אבל יש warnings או המערכת לא מוכנה.
    - ok      → מוכנה, אין critical ואין warnings.
    """
    if has_critical:
        return "error"
    if has_warnings or not ready:
        return "warning"
    return "ok"


def _compute_health_score(
    has_critical: bool,
    num_issues: int,
    num_warnings: int,
) -> float:
    """
    מחשב ציון Health 0–100:

    בסיס:
    -----
    - מתחילים מ-100.
    - כל critical issue מוריד 30 נקודות (לפחות אחד).
    - כל warning מוריד 5 נקודות.
    - כל issue מעבר לראשון מוריד 10 נקודות נוספים.
    - הציון תמיד בתחום [0, 100].
    """
    score = 100.0

    if has_critical:
        score -= 30.0

    if num_issues > 0:
        # issue ראשון כבר נכלל ב-"has_critical" (בדרך כלל),
        # אבל אם אין critical וזו רק בעיה "חמורה" – נוריד 10.
        score -= 10.0 * max(0, num_issues - (1 if has_critical else 0))

    score -= 5.0 * num_warnings

    if score < 0.0:
        score = 0.0
    if score > 100.0:
        score = 100.0

    return score


def _compute_capabilities_health_flags(
    runtime: DashboardRuntime,
    summary: "DashboardSummary",
) -> Tuple[
    List[str],  # missing_critical
    List[str],  # missing_noncritical
    bool,       # can_trade
    bool,       # can_backtest
    bool,       # can_optimize
    bool,       # can_monitor
]:
    """
    נגזרות Health ברמת capabilities:

    - missing_critical_caps:
        LIVE:  broker, risk_engine
        ALL :  (אופציונלי) sql_store
    - missing_noncritical_caps:
        backtester / optimizer / macro_engine / agents
    - can_trade:
        env in {live, paper} + broker + market_data + אין critical.
    - can_backtest:
        backtester==True
    - can_optimize:
        optimizer==True or meta_optimizer==True
    - can_monitor:
        sql_store or risk_engine or macro_engine
    """
    caps = runtime.capabilities
    env = runtime.env

    missing_critical: List[str] = []
    missing_noncritical: List[str] = []

    # קריטיים בסביבת LIVE
    if env == "live":
        if not caps.get("broker", False):
            missing_critical.append("broker")
        if not caps.get("risk_engine", False):
            missing_critical.append("risk_engine")

    # SqlStore – לא קריטי אבל מאוד מומלץ
    if not caps.get("sql_store", False):
        missing_noncritical.append("sql_store")

    # כישורים משניים
    if not caps.get("backtester", False):
        missing_noncritical.append("backtester")
    if not caps.get("optimizer", False) and not caps.get("meta_optimizer", False):
        missing_noncritical.append("optimizer/meta_optimizer")
    if not caps.get("macro_engine", False):
        missing_noncritical.append("macro_engine")
    if not caps.get("agents_manager", False):
        missing_noncritical.append("agents_manager")

    # can_* flags
    can_trade = (
        env in ("live", "paper")
        and bool(caps.get("broker", False))
        and bool(caps.get("market_data_router", False))
    )
    can_backtest = bool(caps.get("backtester", False))
    can_optimize = bool(caps.get("optimizer", False) or caps.get("meta_optimizer", False))
    can_monitor = bool(
        caps.get("sql_store", False)
        or caps.get("risk_engine", False)
        or caps.get("macro_engine", False)
    )

    return (
        missing_critical,
        missing_noncritical,
        can_trade,
        can_backtest,
        can_optimize,
        can_monitor,
    )


def _compute_recommended_actions(
    runtime: DashboardRuntime,
    health: "DashboardHealth",
) -> List[str]:
    """
    מחלץ רשימת המלצות קצרות לפעולה מתוך issues/warnings/capabilities.

    דוגמאות:
    ---------
    - "Connect IBKR broker for LIVE trading."
    - "Enable Risk Engine before trading live."
    - "Configure SqlStore for persistent logging."
    - "Enable Macro Engine for profile='macro'."
    """
    suggestions: List[str] = []
    env = runtime.env
    profile = runtime.profile
    caps = runtime.capabilities

    # Broker / LIVE
    if env == "live" and not caps.get("broker", False):
        suggestions.append("חבר Broker (IBKR) לפני מסחר בסביבת LIVE.")

    # Risk Engine / LIVE
    if env == "live" and not caps.get("risk_engine", False):
        suggestions.append("הפעל Risk Engine כדי להגן על חשיפות בסביבת LIVE.")

    # SqlStore
    if not caps.get("sql_store", False):
        suggestions.append("הגדר SqlStore (DuckDB/SQLite/Postgres) כדי לשמור תוצאות ולוגים.")

    # Macro
    if profile == "macro" and not caps.get("macro_engine", False):
        suggestions.append("הפעל Macro Engine כדי להשתמש בפרופיל macro בצורה מלאה.")

    # Agents
    if not caps.get("agents_manager", False):
        suggestions.append("הפעל Agents Manager כדי לאפשר סוכני AI (Agents Tab).")

    # Backtester / Optimizer
    if not caps.get("backtester", False):
        suggestions.append("התקן/הפעל מודול Backtester לצורך סימולציות היסטוריות.")
    if not (caps.get("optimizer", False) or caps.get("meta_optimizer", False)):
        suggestions.append("הפעל Optimizer/Meta-Optimizer לשיפור פרמטרים.")

    # אם אין הצעות מפורשות – נוסיף הצעה כללית
    if not suggestions and (health.has_warnings or health.has_critical_issues):
        suggestions.append("בדוק את לשונית Logs / System Health לפרטים על אזהרות ותקלות.")

    return suggestions


# -------------------------------------------------------------------
# 24.3 – Core health computation (with cache & deltas)
# -------------------------------------------------------------------

def compute_dashboard_health(runtime: DashboardRuntime) -> DashboardHealth:
    """
    בונה DashboardHealth מלא מתוך DashboardRuntime, עם Cache חכם per-session.

    Ready criteria (ברירת מחדל):
    ----------------------------
    ready = (
        not summary.has_critical_issues
        and at_least_one_core_capability
    )

    at_least_one_core_capability:
        - SqlStore
        או Broker
        או Backtester
        או Optimizer/Meta-Optimizer
        או Macro Engine

    Cache:
    ------
    - משתמש ב-_cache_get/_cache_set (namespace="health", key=env|profile|run_id).
    - TTL קצר (ברירת מחדל 5 שניות) כדי לשקף שינויים בזמן סביר
      בלי לעומס את המערכת בחישובים חוזרים.
    """
    cache_key = f"{runtime.env}|{runtime.profile}|{runtime.run_id}"
    cached = _cache_get("health", cache_key)
    if isinstance(cached, DashboardHealth):
        return cached

    # 1) Summary מלא (כולל services) – יכול לבוא מקודם או להיבנות מחדש
    summary = build_dashboard_summary(runtime)

    # 2) Issues / warnings אנושיים
    issues, warnings = _compute_health_issues_from_summary(runtime, summary)

    caps = runtime.capabilities
    core_caps = [
        caps.get("sql_store", False),
        caps.get("broker", False),
        caps.get("backtester", False),
        caps.get("optimizer", False),
        caps.get("meta_optimizer", False),
        caps.get("macro_engine", False),
    ]
    at_least_one_core = any(bool(x) for x in core_caps)

    has_critical = summary.has_critical_issues or any("error" in i.lower() for i in issues)
    has_warnings = summary.has_warnings or bool(warnings)

    # 3) ready + severity + score
    ready = (not has_critical) and at_least_one_core
    severity = _compute_health_severity(ready, has_critical, has_warnings)
    score = _compute_health_score(has_critical, len(issues), len(warnings))

    # 4) capabilities flags
    (
        missing_critical_caps,
        missing_noncritical_caps,
        can_trade,
        can_backtest,
        can_optimize,
        can_monitor,
    ) = _compute_capabilities_health_flags(runtime, summary)

    # 5) recommended actions
    dummy_health = DashboardHealth(
        env=runtime.env,
        profile=runtime.profile,
        ready=ready,
        has_critical_issues=has_critical,
        has_warnings=has_warnings,
        issues=list(issues),
        warnings=list(warnings),
        ts_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    recommended_actions = _compute_recommended_actions(runtime, dummy_health)

    # 6) delta לעומת Health קודם (אם קיים ב-session_state)
    seconds_since_prev: Optional[float] = None
    try:
        prev_obj = st.session_state.get(SESSION_KEY_HEALTH_LAST)
        if isinstance(prev_obj, Mapping):
            prev_ts = prev_obj.get("ts_utc")
            if isinstance(prev_ts, str) and prev_ts:
                try:
                    prev_dt = datetime.fromisoformat(prev_ts)
                    now_dt = datetime.fromisoformat(dummy_health.ts_utc)
                    delta = (now_dt - prev_dt).total_seconds()
                    if delta >= 0:
                        seconds_since_prev = float(delta)
                except Exception:
                    seconds_since_prev = None
    except Exception:  # pragma: no cover
        seconds_since_prev = None

    # 7) health_id
    health_id = f"{runtime.env}|{runtime.profile}|{runtime.run_id}|{dummy_health.ts_utc}"

    health = DashboardHealth(
        env=runtime.env,
        profile=runtime.profile,
        ready=ready,
        has_critical_issues=has_critical,
        has_warnings=has_warnings,
        issues=list(issues),
        warnings=list(warnings),
        ts_utc=dummy_health.ts_utc,
        summary=summary,
        severity=severity,
        score=score,
        missing_critical_caps=missing_critical_caps,
        missing_noncritical_caps=missing_noncritical_caps,
        can_trade=can_trade,
        can_backtest=can_backtest,
        can_optimize=can_optimize,
        can_monitor=can_monitor,
        recommended_actions=recommended_actions,
        health_id=health_id,
        seconds_since_prev=seconds_since_prev,
    )

    _cache_set("health", cache_key, health, ttl=5.0)
    return health


# -------------------------------------------------------------------
# 24.4 – Dict export / session update / external export
# -------------------------------------------------------------------

def dashboard_health_to_dict(health: DashboardHealth, include_summary: bool = False) -> Dict[str, Any]:
    """
    ממיר DashboardHealth ל-dict JSON-friendly.

    אם include_summary=True → מכניס גם Summary מלא (dict) תחת key="summary".
    """
    base: Dict[str, Any] = {
        "env": health.env,
        "profile": health.profile,
        "ready": health.ready,
        "has_critical_issues": health.has_critical_issues,
        "has_warnings": health.has_warnings,
        "issues": list(health.issues),
        "warnings": list(health.warnings),
        "ts_utc": health.ts_utc,
        "severity": health.severity,
        "score": health.score,
        "missing_critical_caps": list(health.missing_critical_caps),
        "missing_noncritical_caps": list(health.missing_noncritical_caps),
        "can_trade": health.can_trade,
        "can_backtest": health.can_backtest,
        "can_optimize": health.can_optimize,
        "can_monitor": health.can_monitor,
        "recommended_actions": list(health.recommended_actions),
        "health_id": health.health_id,
        "seconds_since_prev": health.seconds_since_prev,
    }

    if include_summary and health.summary is not None:
        base["summary"] = dashboard_summary_to_dict(health.summary)

    return base


def update_dashboard_health_in_session(
    runtime: DashboardRuntime,
    include_summary: bool = False,
) -> DashboardHealth:
    """
    מחשב DashboardHealth ומעדכן אותו ב-session_state[SESSION_KEY_HEALTH_LAST].

    - שומר dict JSON-friendly (לא את ה-DashboardHealth עצמו).
    - מחזיר את ה-DashboardHealth עבור שימוש מיידי.
    """
    health = compute_dashboard_health(runtime)
    obj = dashboard_health_to_dict(health, include_summary=include_summary)
    obj = _make_json_safe(obj)

    try:
        st.session_state[SESSION_KEY_HEALTH_LAST] = obj
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to store dashboard health in session_state: %s", exc)

    logger.debug(
        "Dashboard health updated in session_state (env=%s, profile=%s, ready=%s, severity=%s, score=%.1f)",
        health.env,
        health.profile,
        health.ready,
        health.severity,
        health.score,
    )

    return health


def export_dashboard_health(runtime: DashboardRuntime, include_summary: bool = False) -> Dict[str, Any]:
    """
    Export ידידותי ל-Desktop/Agents:

    - לא נוגע ב-session_state.
    - מחזיר dict JSON-friendly (ready/issues/warnings/score/flags/...).
    - אופציונלית כולל Summary מלא (heavy יותר, אבל נותן תמונה מלאה).
    """
    health = compute_dashboard_health(runtime)
    payload = dashboard_health_to_dict(health, include_summary=include_summary)
    return _make_json_safe(payload)


# -------------------------------------------------------------------
# 24.5 – Headless helper for CI / Desktop readiness checks
# -------------------------------------------------------------------

def check_dashboard_ready(app_ctx: Optional["AppContext"] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    פונקציית נוחות "Headless" לבדיקה האם הדשבורד מוכן:

    שימוש טיפוסי:
    --------------
    - בסקריפט CI/CD:
        ready, info = check_dashboard_ready()
        if not ready: exit(1)
    - ב-Desktop Bridge:
        לפני פתיחת המסכים, בודקים אם המערכת במצב תקין.

    הפלט:
    -----
    (ready: bool, info: dict)
        ready – האם המערכת במצב "Ready" לפי הקריטריונים של compute_dashboard_health.
        info  – dict קטן עם env/profile/issues/warnings/score/severity
                (לא כולל Summary מלא).
    """
    if app_ctx is None:
        app_ctx = get_app_context()

    runtime = ensure_dashboard_runtime(app_ctx)
    health = compute_dashboard_health(runtime)

    info = {
        "env": health.env,
        "profile": health.profile,
        "ready": health.ready,
        "severity": health.severity,
        "score": health.score,
        "has_critical_issues": health.has_critical_issues,
        "has_warnings": health.has_warnings,
        "issues": list(health.issues),
        "warnings": list(health.warnings),
        "seconds_since_prev": health.seconds_since_prev,
        "ts_utc": health.ts_utc,
    }

    return health.ready, _make_json_safe(info)


# עדכון __all__ עבור חלק 24
try:
    __all__ += [
        "SESSION_KEY_HEALTH_LAST",
        "DashboardHealth",
        "compute_dashboard_health",
        "dashboard_health_to_dict",
        "update_dashboard_health_in_session",
        "export_dashboard_health",
        "check_dashboard_ready",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_HEALTH_LAST",
        "DashboardHealth",
        "compute_dashboard_health",
        "dashboard_health_to_dict",
        "update_dashboard_health_in_session",
        "export_dashboard_health",
        "check_dashboard_ready",
    ]

# =====================
