# -*- coding: utf-8 -*-
"""
root/dashboard_home_v2.py — Hedge-Fund-Grade Dashboard Home (Streamlit)
=======================================================================

טאב דשבורד ראשי ברמת קרן גידור, מחובר לעומק למערכת:

מה הטאב הזה יודע לעשות ברמת תכנון (50 רעיונות):
------------------------------------------------
A. ביצועי פורטפוליו & חשיפות
    1.  Performance Tiles (Today / MTD / YTD / Since Inception)
    2.  Equity Curve קצרה + Overlay מול Benchmark
    3.  PnL Attribution לפי אסטרטגיה (pairs / macro / hedge / stat-arb)
    4.  PnL לפי Asset Class (Equities / Rates / FX / Crypto / ...)
    5.  Top Winners / Losers (פוזיציות)
    6.  Heatmap של משקל + PnL
    7.  Exposure Drift vs Target (Over/Under-weight)
    8.  Turnover Monitor (1D / 5D / 20D)
    9.  Liquidity Snapshot (Illiquid vs Liquid)
    10. Concentration Monitor (top 1/5/10 positions, VaR contribution)

B. סיכון & Risk Engine
    11. VaR/ES Panel עם היסטוריה קצרה
    12. Risk Budget vs Usage (Vol / VaR)
    13. Factor Risk Decomposition (Market / Size / Value / ...)
    14. Stress Test Scenarios Box (2008 / Covid / Rate shock)
    15. Drawdown Monitor (Current / 1Y / ITD)
    16. Regime-Aware Risk Mode (Risk-on/off, Crisis mode)
    17. Kill-Switch Status Tile
    18. Risk Alerts Feed (breaches, concentration, VaR spikes)
    19. Beta & Tracking Error Tracker
    20. Tail Risk Index

C. סיגנלים & הזדמנויות
    21. Signal Funnel Overview (Universe → Filtered → Signals → Deployed)
    22. Signal Quality Ladder (Top High-Conviction)
    23. Conflict Detector (סיגנלים סותרים)
    24. Regime-Aware Signal Tagging (aligned vs anti-regime)
    25. Expected Sharpe / Edge Estimator
    26. Signal Aging (Fresh / Stale / Expired)
    27. Pairs Heatmap (β vs Half-Life vs |Z|)
    28. Quick Actions (Analyze / Backtest / Send to Live)
    29. Signal History Sparkline (Z/Spread)
    30. Watchlist Integration (⭐ + Panel ייעודי)

D. מאקרו & שוק
    31. Macro Regime Banner (Growth / Inflation / Stagflation / Recession)
    32. Cross-Asset Snapshot (Equities / Rates / Credit / FX / Commodities)
    33. Yield Curve Mini-Chart (slope, steepening/flattening)
    34. Volatility Grid (Implied/Realized across markets)
    35. Macro Events Box (FOMC / NFP / CPI וכו')
    36. Risk-On / Risk-Off Indicator
    37. FX & Commodities Snapshot
    38. Overlay של Positioning מול Regime (Pro-risk vs Defensive)

E. מערכת, Agents & UX
    39. Deployment Banner (גרסה, env, host)
    40. Latency Breakdown (Market data / Broker / SQL / Engine)
    41. Error Heatmap (Data / Broker / SQL / Agents)
    42. Agents Activity Feed (SystemUpgrader, Insights, Macro, וכו')
    43. Scheduled Tasks Overview (Nightly jobs / cron)
    44. Logging Level & Debug Toggle
    45. Resource Utilization Panel (CPU / RAM / Disk)

F. UX / Flow / Notes
    46. “Story of the Day” — סיכום טקסטואלי אוטומטי
    47. Saved Views / Layout Profiles (Trader / CIO / Risk)
    48. Drill-Down Links מכל Panel לטאבים הרלוונטיים
    49. Daily Checklist / Runbook (checklist יומי)
    50. Comment / Notes Integration (Desk Notes ל-SQL)

תפקיד חלק 1/5:
----------------
- חיבור עמוק ל-AppContext ול-settings (base_currency, timezone, app_version).
- הגדרת קונפיג מרכזי לדשבורד (env/profile/ui/ranges).
- Helpers ל:
    * פורמט מספרים/אחוזים/דלתא.
    * זיהוי language ו-theme.
    * שליפת feature_flags ו־meta מתוך DashboardContext.extra.
    * בניית כותרת / תיאור עליון “Desk-aware” (host, user וכו').
- מבלי לערב עדיין Panels ספציפיים (זה יבוא בחלקים הבאים).
"""

from __future__ import annotations

import logging
import os
import socket
import getpass
from datetime import date, datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.dashboard_models import (
    DashboardContext,
    DashboardSnapshot,
    PortfolioSnapshot,
    PortfolioExposureBreakdown,
    MarketSnapshot,
    RiskSnapshot,
    SignalsSnapshot,
    SystemHealthSnapshot,
    PositionSnapshot,
)

from core.app_context import AppContext

from root.dashboard import (
    DashboardRuntime,
    ensure_dashboard_runtime,
    update_dashboard_home_context_in_session,
    get_dashboard_home_context_from_session,
    FeatureFlags,
    NavPayload,
)

from root.dashboard_service_factory import bootstrap_dashboard
from core.ib_order_router import PairOrderLeg, PairOrderRequest
from root.ibkr_connection import ib_connection_status, get_ib_instance

logger = logging.getLogger(__name__)



# ============================================================
# קונפיג מרכזי לדשבורד (ניתן להרחיב/לדרוס)
# ============================================================

DEFAULT_DATE_MODE: str = "1m"

DEFAULT_ENV_OPTIONS: List[str] = [
    "dev",
    "paper",
    "live",
    "research",
    "backtest",
]

DEFAULT_PROFILE_OPTIONS: List[str] = [
    "monitoring",
    "trading",
    "research",
    "risk",
    "macro",
]

DEFAULT_UI_MODES: List[str] = [
    "simple",
    "advanced",
    "research",
]

# כמה פריטים לכל panel (ברירת מחדל, ניתן לשנות ב-Controls וגם לפי ctx)
DEFAULT_TOP_POSITIONS: int = 20
DEFAULT_TOP_SIGNALS: int = 30

# מגבלות סף ויזואליים (Thresholds) — אפשר להשתמש בהם ב-panels מאוחרים
RISK_VOL_WARNING: float = 0.25     # 25%
RISK_VOL_CRITICAL: float = 0.40    # 40%
DD_WARNING: float = 0.10           # 10%
DD_CRITICAL: float = 0.20          # 20%


# ============================================================
# Helpers — App / Context / Meta
# ============================================================


def _get_global_app_context() -> Optional[AppContext]:
    """
    מחזיר AppContext גלובלי אם קיים, אחרת None.
    """
    try:
        return AppContext.get_global()
    except Exception:
        return None


def _resolve_host_user() -> Tuple[str, str]:
    """
    מחזיר (host, user) — משמש לכותרת, meta בדשבורד, ושמירה ב-extra.
    """
    host = socket.gethostname()
    try:
        user = getpass.getuser()
    except Exception:
        user = os.getenv("USERNAME") or os.getenv("USER") or "unknown"
    return host, user


def _extract_feature_flags(ctx: DashboardContext) -> Dict[str, Any]:
    """
    מחלץ feature_flags מתוך ctx.extra אם קיימים, אחרת מחזיר dict ריק.

    הציפייה:
        ctx.extra.get("feature_flags") → dict עם דגלים לוגיים,
        למשל:
            {
                "enable_experimental_tabs": True,
                "enable_risk_tab": True,
                "enable_live_trading_actions": False,
                ...
            }
    """
    try:
        extra = getattr(ctx, "extra", {}) or {}
        flags = extra.get("feature_flags") or {}
        if isinstance(flags, dict):
            return dict(flags)
        return {}
    except Exception:
        return {}


def _get_app_version_and_schema(app_ctx: Optional[AppContext]) -> Tuple[Optional[str], Optional[str]]:
    """
    מנסה לשלוף app_version ו-sql_schema מתוך settings (אם קיימים).
    """
    version = None
    schema = None
    if app_ctx is None:
        return version, schema

    settings = getattr(app_ctx, "settings", None)
    if settings is None:
        return version, schema

    try:
        if getattr(settings, "app_version", None):
            version = str(settings.app_version)
    except Exception:
        pass

    try:
        if getattr(settings, "sql_schema", None):
            schema = str(settings.sql_schema)
    except Exception:
        pass

    return version, schema


def _now_utc_str() -> str:
    """
    מחזיר מחרוזת זמן נוכחי ב-UTC, לצורך 'Last updated' וכו'.
    """
    try:
        return datetime.now(timezone.utc)().strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "N/A"


def _detect_language(ctx: Optional[DashboardContext]) -> str:
    """
    מנסה לזהות language לעבודה (he / en) לפי ctx, אחרת 'he'.
    """
    if ctx is None:
        return "he"
    try:
        lang = getattr(ctx, "language", None)
        if isinstance(lang, str) and lang.lower() in ("he", "en"):
            return lang.lower()
    except Exception:
        pass
    return "he"


# ============================================================
# Helpers — פורמטים בסיסיים (מספרים, אחוזים, דלתא)
# ============================================================


def _format_pct(x: Optional[float], digits: int = 2) -> str:
    """
    פורמט לאחוזים (0.123 → '12.30%').
    אם x=None או NaN → '—'.
    """
    if x is None:
        return "—"
    try:
        if pd.isna(x):
            return "—"
        return f"{x:.{digits}%}"
    except Exception:
        return "—"


def _format_num(x: Optional[float], digits: int = 0) -> str:
    """
    פורמט למספרים עם K/M/B:

    123              → 123
    1_500            → 1.5K
    2_000_000        → 2.0M
    3_000_000_000    → 3.0B
    """
    if x is None:
        return "—"
    try:
        if pd.isna(x):
            return "—"
        v = float(x)
        if abs(v) >= 1_000_000_000:
            return f"{v / 1_000_000_000:.{digits}f}B"
        if abs(v) >= 1_000_000:
            return f"{v / 1_000_000:.{digits}f}M"
        if abs(v) >= 1_000:
            return f"{v / 1_000:.{digits}f}K"
        return f"{v:.{digits}f}"
    except Exception:
        return "—"


def _format_delta(x: Optional[float], digits: int = 0, is_pct: bool = False) -> Optional[str]:
    """
    פורמט לשינוי (delta) עם סימן +/-

    דוגמאות:
        _format_delta(123)                  → '+123'
        _format_delta(-50)                  → '-50'
        _format_delta(0.03, is_pct=True)    → '+3.00%'
    """
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
        if is_pct:
            return f"{x:+.{digits}%}"
        return f"{x:+.{digits}f}"
    except Exception:
        return None


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """
    getattr בטוח, למניעת נפילות בפאנלים (במקרה שחסר ערך/שדה).
    """
    try:
        return getattr(obj, name)
    except Exception:
        return default


# ============================================================
# Helpers — צבעים / רמות / Risk-levels (לשימוש מאוחר יותר בפאנלים)
# ============================================================


def _traffic_color_from_level(
    value: Optional[float],
    warn: float,
    critical: float,
    *,
    reverse: bool = False,
) -> str:
    """
    מחזיר 'green' / 'yellow' / 'red' לפי value ו-thresholds.

    reverse=True → ערך נמוך יותר גרוע (למשל ל-liquidity).
    """
    if value is None:
        return "gray"

    try:
        v = float(value)
    except Exception:
        return "gray"

    if reverse:
        # נמוך = רע
        if v <= critical:
            return "red"
        if v <= warn:
            return "yellow"
        return "green"

    # גבוה = רע
    if v >= critical:
        return "red"
    if v >= warn:
        return "yellow"
    return "green"


def _emoji_from_color(color: str) -> str:
    """
    ממיר צבע טכני ל-emoji נוח.
    """
    c = color.lower()
    if c == "red":
        return "🔴"
    if c == "yellow":
        return "🟡"
    if c == "green":
        return "🟢"
    if c == "gray":
        return "⚪"
    return "⚪"

# ============================================================
# Data Builders — Performance, PnL, Exposure, Concentration
# ============================================================

from typing import Any, Dict, List, Optional, Mapping


def _compute_investment_readiness(
    snapshot: Optional[DashboardSnapshot],
    home_ctx: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    מחשב "Investment Readiness" ברמה גבוהה מתוך snapshot + home_ctx.

    לוגיקה (HF-grade, גרסה מורחבת):
    --------------------------------
    1. אם home_ctx מכיל investment_readiness מוכן מראש → נשתמש בו (למשל
       חישוב של Agent / Optimizer / Risk Engine).
    2. אחרת נבנה סקור מרובה-מימדים:
       - Performance (Sharpe / total_return / recent vs full)
       - Risk (max drawdown, tail risk / ES / worst loss)
       - Stability (תנודתיות, יציבות equity)
       - Signals (count / quality אם קיימים)
       - Diversification (n_positions, concentration)
       - Macro (regime fit)
       - System / Infra (health, errors/warnings)
       - Data quality (feeds / missing data אם קיימים)
       - Robustness (WF / N scenarios / is_robust flags)
       - Live drift (פער בין ביצוע חי לבק-טסט)

    פלט:
    ----
    {
      "overall_score": int | None,    # 0–100
      "dimensions": [
        {
          "key": "performance" | "risk" | ...,
          "label": str,
          "score": int,
          "status": "good" | "ok" | "watch" | "bad",
          "reason": str,
        },
        ...
      ]
    }

    משקולות:
    ---------
    - ברירת מחדל: balanced (performance, risk, system, tail-risk מקבלים משקל גבוה).
    - במקרה של risk_profile="defense" → משקל גבוה יותר ל-risk, tail, data_quality, system.
    - במקרה של risk_profile="offense" → משקל גבוה יותר ל-performance, recent_vs_full, signals.
    """

    # אם יש כבר חישוב מוכן ב-HomeContext – נשתמש בו
    if home_ctx:
        ir = home_ctx.get("investment_readiness")
        if isinstance(ir, Mapping) and ir.get("dimensions"):
            # אם אין overall_score שם, נחשב מחדש מסכום המימדים
            dims_existing = list(ir.get("dimensions", []))
            scores_existing = [
                d.get("score")
                for d in dims_existing
                if isinstance(d.get("score"), (int, float))
            ]
            if scores_existing and not ir.get("overall_score"):
                overall_existing = int(
                    round(sum(scores_existing) / len(scores_existing))
                )
                new_ir = dict(ir)
                new_ir["overall_score"] = overall_existing
                return new_ir
            return dict(ir)

    if snapshot is None:
        return {"overall_score": None, "dimensions": []}

    def find_attr(obj: Any, *names: str) -> Any:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    def _safe_num(x: Any) -> Optional[float]:
        try:
            if isinstance(x, (int, float)):
                return float(x)
            return None
        except Exception:
            return None

    dims: List[Dict[str, Any]] = []

    # ---- Risk profile / focus מה-HomeContext (למשקולות) ----
    risk_profile = "balanced"
    if home_ctx:
        risk_profile = str(
            home_ctx.get("risk_profile", home_ctx.get("risk_focus", "balanced"))
        ).lower()
        if risk_profile not in ("balanced", "defense", "offense"):
            risk_profile = "balanced"

    # ======================================================================
    # 1) Performance dimension
    # ======================================================================
    portfolio = getattr(snapshot, "portfolio", None)
    perf_score: Optional[int] = None
    perf_reason = ""

    sharpe = None
    total_ret = None
    recent_sharpe = None
    recent_ret = None

    if portfolio is not None:
        sharpe = _safe_num(
            find_attr(portfolio, "sharpe", "sharpe_ratio", "sr", "sharpe_full")
        )
        total_ret = _safe_num(
            find_attr(portfolio, "total_return", "return_pct", "cagr", "ann_return")
        )
        recent_sharpe = _safe_num(
            find_attr(portfolio, "recent_sharpe", "sharpe_recent", "sr_recent")
        )
        recent_ret = _safe_num(
            find_attr(portfolio, "recent_return", "return_recent_pct", "ytd_return")
        )

    # Performance בסיסי
    if sharpe is not None:
        if sharpe >= 1.8:
            perf_score = 93
            perf_reason = f"Sharpe≈{sharpe:.2f} (excellent)."
        elif sharpe >= 1.3:
            perf_score = 88
            perf_reason = f"Sharpe≈{sharpe:.2f} (very good)."
        elif sharpe >= 1.0:
            perf_score = 80
            perf_reason = f"Sharpe≈{sharpe:.2f} (good)."
        elif sharpe >= 0.6:
            perf_score = 65
            perf_reason = f"Sharpe≈{sharpe:.2f} (ok)."
        else:
            perf_score = 45
            perf_reason = f"Sharpe≈{sharpe:.2f} (weak)."
    elif total_ret is not None:
        if total_ret >= 0.5:
            perf_score = 88
            perf_reason = f"Total return≈{total_ret:.0%} (strong)."
        elif total_ret >= 0.3:
            perf_score = 80
            perf_reason = f"Total return≈{total_ret:.0%} (good)."
        elif total_ret >= 0.1:
            perf_score = 70
            perf_reason = f"Total return≈{total_ret:.0%} (ok)."
        elif total_ret >= 0.0:
            perf_score = 60
            perf_reason = f"Total return≈{total_ret:.0%} (flat/ok)."
        else:
            perf_score = 42
            perf_reason = f"Total return≈{total_ret:.0%} (negative)."

    # Recent vs full (רעיון חדש #1)
    recent_vs_full_score: Optional[int] = None
    recent_vs_full_reason = ""
    if recent_sharpe is not None and sharpe is not None:
        # האם הביצוע האחרון תומך בתמונה ההיסטורית?
        delta = recent_sharpe - sharpe
        if delta >= 0.3:
            recent_vs_full_score = 90
            recent_vs_full_reason = (
                f"Recent Sharpe≈{recent_sharpe:.2f} > full≈{sharpe:.2f} (improving)."
            )
        elif -0.2 <= delta < 0.3:
            recent_vs_full_score = 75
            recent_vs_full_reason = (
                f"Recent Sharpe≈{recent_sharpe:.2f} ~ full≈{sharpe:.2f} (stable)."
            )
        elif -0.5 <= delta < -0.2:
            recent_vs_full_score = 60
            recent_vs_full_reason = (
                f"Recent Sharpe≈{recent_sharpe:.2f} < full≈{sharpe:.2f} (mild decay)."
            )
        else:
            recent_vs_full_score = 45
            recent_vs_full_reason = (
                f"Recent Sharpe≈{recent_sharpe:.2f} << full≈{sharpe:.2f} (sharp decay)."
            )

    if perf_score is not None:
        status = (
            "good"
            if perf_score >= 85
            else "ok" if perf_score >= 65 else "watch"
        )
        dims.append(
            {
                "key": "performance",
                "label": "Backtest / Performance",
                "score": perf_score,
                "status": status,
                "reason": perf_reason or "Performance metrics available.",
            }
        )

    if recent_vs_full_score is not None:
        status = (
            "good"
            if recent_vs_full_score >= 85
            else "ok" if recent_vs_full_score >= 65 else "watch"
        )
        dims.append(
            {
                "key": "recent_vs_full",
                "label": "Recent vs full performance",
                "score": recent_vs_full_score,
                "status": status,
                "reason": recent_vs_full_reason,
            }
        )

    # ======================================================================
    # 2) Risk & tail risk dimensions
    # ======================================================================
    risk = getattr(snapshot, "risk", None)
    if risk is None:
        risk = getattr(snapshot, "risk_snapshot", None)

    risk_score: Optional[int] = None
    risk_reason = ""
    tail_score: Optional[int] = None
    tail_reason = ""

    if risk is not None:
        dd = _safe_num(
            find_attr(risk, "max_drawdown", "max_drawdown_pct", "max_dd")
        )

        if dd is not None:
            dd_abs = abs(dd)
            # DD עד 5% מעולה, עד 10% טוב, עד 20% סביר, מעבר = חלש
            if dd_abs <= 0.05:
                risk_score = 92
                risk_reason = f"Max DD≈{dd_abs:.0%} (very low)."
            elif dd_abs <= 0.10:
                risk_score = 85
                risk_reason = f"Max DD≈{dd_abs:.0%} (controlled)."
            elif dd_abs <= 0.20:
                risk_score = 70
                risk_reason = f"Max DD≈{dd_abs:.0%} (moderate)."
            elif dd_abs <= 0.40:
                risk_score = 50
                risk_reason = f"Max DD≈{dd_abs:.0%} (high)."
            else:
                risk_score = 35
                risk_reason = f"Max DD≈{dd_abs:.0%} (very high)."

        es = _safe_num(
            find_attr(
                risk,
                "cvar_95",
                "es_95",
                "expected_shortfall",
                "tail_risk",
                "es",
            )
        )
        worst_loss = _safe_num(
            find_attr(
                risk,
                "worst_trade",
                "worst_day",
                "min_daily_return",
            )
        )

        # Tail risk (רעיון חדש #2)
        if es is not None:
            es_abs = abs(es)
            if es_abs <= 0.03:
                tail_score = 90
                tail_reason = f"ES95≈{es_abs:.0%} (low tail risk)."
            elif es_abs <= 0.06:
                tail_score = 78
                tail_reason = f"ES95≈{es_abs:.0%} (controlled tails)."
            elif es_abs <= 0.12:
                tail_score = 60
                tail_reason = f"ES95≈{es_abs:.0%} (meaningful tails)."
            else:
                tail_score = 40
                tail_reason = f"ES95≈{es_abs:.0%} (heavy tails)."
        elif worst_loss is not None:
            wl_abs = abs(worst_loss)
            if wl_abs <= 0.05:
                tail_score = 80
                tail_reason = f"Worst loss≈{wl_abs:.0%} (contained)."
            elif wl_abs <= 0.10:
                tail_score = 65
                tail_reason = f"Worst loss≈{wl_abs:.0%} (moderate)."
            else:
                tail_score = 45
                tail_reason = f"Worst loss≈{wl_abs:.0%} (large)."

    if risk_score is None:
        risk_score = 65
        risk_reason = "No explicit max-drawdown, using neutral risk score."

    if risk_score is not None:
        status = (
            "good"
            if risk_score >= 85
            else "ok" if risk_score >= 65 else "watch"
        )
        dims.append(
            {
                "key": "risk",
                "label": "Risk profile & drawdown",
                "score": risk_score,
                "status": status,
                "reason": risk_reason,
            }
        )

    if tail_score is not None:
        status = (
            "good"
            if tail_score >= 85
            else "ok" if tail_score >= 65 else "watch"
        )
        dims.append(
            {
                "key": "tail_risk",
                "label": "Tail risk (ES / worst loss)",
                "score": tail_score,
                "status": status,
                "reason": tail_reason,
            }
        )

    # ======================================================================
    # 3) Stability dimension (רעיון חדש #3)
    # ======================================================================
    stability_score: Optional[int] = None
    stability_reason = ""

    if portfolio is not None:
        eq_vol = _safe_num(
            find_attr(
                portfolio,
                "equity_vol",
                "pnl_vol",
                "volatility",
                "equity_stdev",
            )
        )
        stability_metric = _safe_num(
            find_attr(
                portfolio,
                "return_stability",
                "equity_stability",
                "stability_score",
            )
        )

        if stability_metric is not None:
            # מניחים 0–1 → נמפה ל־50–95
            s = max(0.0, min(1.0, stability_metric))
            stability_score = int(50 + s * 45)
            stability_reason = (
                f"Stability metric≈{stability_metric:.2f} (higher is better)."
            )
        elif eq_vol is not None:
            # ככל שהתנודתיות נמוכה יותר → יציבות גבוהה יותר
            v = eq_vol
            if v <= 0.10:
                stability_score = 90
                stability_reason = f"Equity vol≈{v:.0%} (very stable)."
            elif v <= 0.20:
                stability_score = 78
                stability_reason = f"Equity vol≈{v:.0%} (controlled)."
            elif v <= 0.35:
                stability_score = 62
                stability_reason = f"Equity vol≈{v:.0%} (moderate)."
            else:
                stability_score = 48
                stability_reason = f"Equity vol≈{v:.0%} (high)."

    if stability_score is not None:
        status = (
            "good"
            if stability_score >= 85
            else "ok" if stability_score >= 65 else "watch"
        )
        dims.append(
            {
                "key": "stability",
                "label": "Equity / PnL stability",
                "score": stability_score,
                "status": status,
                "reason": stability_reason,
            }
        )

    # ======================================================================
    # 4) Signals dimension
    # ======================================================================
    signals = getattr(snapshot, "signals", None)
    if signals is None:
        signals = getattr(snapshot, "signals_snapshot", None)

    if signals is not None:
        n_signals = _safe_num(
            find_attr(signals, "n_signals", "num_signals", "count")
        )
        win_rate = _safe_num(
            find_attr(signals, "win_rate", "hit_ratio", "success_ratio")
        )

        sig_score: Optional[int] = None
        sig_reason = ""

        if n_signals is not None:
            if n_signals == 0:
                sig_score = 42
                sig_reason = "No active signals."
            elif n_signals <= 10:
                sig_score = 78
                sig_reason = f"{int(n_signals)} signals (focused)."
            elif n_signals <= 50:
                sig_score = 70
                sig_reason = f"{int(n_signals)} signals (diversified)."
            else:
                sig_score = 55
                sig_reason = f"{int(n_signals)} signals (potentially noisy)."

        if win_rate is not None:
            # משקלל את win_rate לתוך הסיגנלים (רעיון חדש #4)
            if win_rate >= 0.65:
                adj = 10
                sig_reason += f" Win-rate≈{win_rate:.0%} (strong)."
            elif win_rate >= 0.55:
                adj = 5
                sig_reason += f" Win-rate≈{win_rate:.0%} (ok+)."
            elif win_rate >= 0.50:
                adj = 0
                sig_reason += f" Win-rate≈{win_rate:.0%} (neutral)."
            else:
                adj = -7
                sig_reason += f" Win-rate≈{win_rate:.0%} (weak)."
            sig_score = int((sig_score or 60) + adj)

        if sig_score is not None:
            sig_score = max(0, min(100, sig_score))
            status = (
                "good"
                if sig_score >= 85
                else "ok" if sig_score >= 65 else "watch"
            )
            dims.append(
                {
                    "key": "signals",
                    "label": "Signal universe",
                    "score": sig_score,
                    "status": status,
                    "reason": sig_reason,
                }
            )

    # ======================================================================
    # 5) Diversification dimension (רעיון חדש #5)
    # ======================================================================
    div_score: Optional[int] = None
    div_reason = ""
    if portfolio is not None:
        n_pos = _safe_num(
            find_attr(portfolio, "n_positions", "num_positions", "positions_count")
        )
        n_pairs = _safe_num(
            find_attr(portfolio, "n_pairs", "num_pairs", "pairs_count")
        )
        hhi = _safe_num(
            find_attr(
                portfolio,
                "herfindahl_index",
                "concentration_index",
                "concentration_hhi",
            )
        )
        top_weight = _safe_num(
            find_attr(portfolio, "top_weight", "max_weight", "largest_pos_weight")
        )

        effective_n = None
        if n_pos is not None:
            effective_n = n_pos
        elif n_pairs is not None:
            effective_n = n_pairs

        if effective_n is not None:
            if effective_n <= 3:
                div_score = 50
                div_reason = f"{int(effective_n)} positions/pairs (concentrated)."
            elif effective_n <= 10:
                div_score = 75
                div_reason = f"{int(effective_n)} positions/pairs (good focus)."
            elif effective_n <= 25:
                div_score = 70
                div_reason = f"{int(effective_n)} positions/pairs (diversified)."
            else:
                div_score = 60
                div_reason = f"{int(effective_n)} positions/pairs (very wide)."

        if hhi is not None:
            # HHI נמוך → פיזור טוב
            if hhi <= 0.10:
                div_score = max(div_score or 70, 85)
                div_reason += " HHI low (well diversified)."
            elif hhi <= 0.20:
                div_score = max(div_score or 65, 75)
                div_reason += " HHI moderate."
            else:
                div_score = min(div_score or 65, 55)
                div_reason += " HHI high (concentrated)."

        if top_weight is not None:
            if top_weight >= 0.3:
                div_score = min(div_score or 70, 55)
                div_reason += f" Top weight≈{top_weight:.0%} (concentrated)."
            elif top_weight <= 0.15:
                div_score = max(div_score or 70, 80)
                div_reason += f" Top weight≈{top_weight:.0%} (balanced)."

    if div_score is not None:
        status = (
            "good"
            if div_score >= 85
            else "ok" if div_score >= 65 else "watch"
        )
        dims.append(
            {
                "key": "diversification",
                "label": "Diversification / concentration",
                "score": div_score,
                "status": status,
                "reason": div_reason,
            }
        )

    # ======================================================================
    # 6) Macro dimension
    # ======================================================================
    market = getattr(snapshot, "market", None)
    if market is None:
        market = getattr(snapshot, "market_snapshot", None)

    if market is not None:
        regime = find_attr(market, "regime_label", "macro_regime", "regime")
        macro_score = 72
        macro_reason = "Macro-neutral regime."

        if isinstance(regime, str):
            r = regime.lower()
            if "crisis" in r or "stress" in r or "recession" in r:
                macro_score = 48
                macro_reason = f"Macro regime: {regime} (stress)."
            elif "late" in r or "tight" in r:
                macro_score = 62
                macro_reason = f"Macro regime: {regime} (late-cycle)."
            elif "early" in r or "expansion" in r or "reflation" in r:
                macro_score = 82
                macro_reason = f"Macro regime: {regime} (favorable)."

        status = (
            "good"
            if macro_score >= 85
            else "ok" if macro_score >= 65 else "watch"
        )
        dims.append(
            {
                "key": "macro",
                "label": "Macro regime fit",
                "score": macro_score,
                "status": status,
                "reason": macro_reason,
            }
        )

    # ======================================================================
    # 7) System / infra dimension
    # ======================================================================
    system = getattr(snapshot, "system", None)
    if system is None:
        system = getattr(snapshot, "system_health", None)

    sys_score: Optional[int] = None
    sys_reason = ""

    if system is not None:
        level = find_attr(system, "level", "status", "health_level")
        errors = _safe_num(find_attr(system, "error_count", "errors"))
        warns = _safe_num(find_attr(system, "warning_count", "warnings"))

        if isinstance(level, str):
            lv = level.lower()
            if "critical" in lv or "error" in lv:
                sys_score = 32
                sys_reason = f"System status: {level}."
            elif "warning" in lv or "degraded" in lv:
                sys_score = 58
                sys_reason = f"System status: {level}."
            elif any(k in lv for k in ("ok", "healthy", "ready", "green")):
                sys_score = 86
                sys_reason = f"System status: {level}."

        if errors is not None and errors > 0:
            sys_score = min(sys_score or 70, 45)
            sys_reason += f" {int(errors)} errors detected."
        if warns is not None and warns > 0:
            sys_score = min(sys_score or 80, 60)
            sys_reason += f" {int(warns)} warnings detected."

    if sys_score is None and home_ctx:
        # fallback ל-health_light
        health_light = home_ctx.get("health_light", {}) or {}
        if health_light.get("has_critical_issues"):
            sys_score = 38
            sys_reason = "Critical issues in health_light."
        elif health_light.get("has_warnings"):
            sys_score = 62
            sys_reason = "Warnings in health_light."
        else:
            sys_score = 82
            sys_reason = "No critical health issues."

    if sys_score is not None:
        status = (
            "good"
            if sys_score >= 85
            else "ok" if sys_score >= 65 else "watch"
        )
        dims.append(
            {
                "key": "system",
                "label": "System & infrastructure",
                "score": sys_score,
                "status": status,
                "reason": sys_reason,
            }
        )

    # ======================================================================
    # 8) Data quality dimension (רעיון חדש #6)
    # ======================================================================
    dq_score: Optional[int] = None
    dq_reason = ""
    data_quality_ctx = None
    if home_ctx:
        data_quality_ctx = home_ctx.get("data_quality")

    if isinstance(data_quality_ctx, list) and data_quality_ctx:
        # אם יש רשימה של אייטמים, נדרג לפי האם יש "down"/"degraded"
        has_down = any(
            str(it.get("status", "")).lower() in ("down", "error", "critical")
            for it in data_quality_ctx
        )
        has_degraded = any(
            str(it.get("status", "")).lower()
            in ("degraded", "warning", "stale")
            for it in data_quality_ctx
        )
        if has_down:
            dq_score = 45
            dq_reason = "Some data feeds are down / error."
        elif has_degraded:
            dq_score = 65
            dq_reason = "Some data feeds are degraded / stale."
        else:
            dq_score = 85
            dq_reason = "All data feeds reported healthy."
    # אפשר גם לנסות למשוך מה-snapshot עצמו אם יש field כזה
    elif snapshot is not None:
        dq_attr = getattr(snapshot, "data_quality", None)
        if dq_attr is not None:
            quality = _safe_num(find_attr(dq_attr, "score", "quality_score"))
            missing = _safe_num(
                find_attr(dq_attr, "missing_ratio", "missing_pct")
            )
            if quality is not None:
                dq_score = int(max(0, min(100, quality)))
                dq_reason = f"Data quality score≈{dq_score}/100."
            elif missing is not None:
                m = missing
                if m <= 0.01:
                    dq_score = 88
                    dq_reason = "Missing data <1%."
                elif m <= 0.05:
                    dq_score = 75
                    dq_reason = "Missing data <5%."
                else:
                    dq_score = 55
                    dq_reason = "Significant missing data."
    if dq_score is not None:
        status = (
            "good"
            if dq_score >= 85
            else "ok" if dq_score >= 65 else "watch"
        )
        dims.append(
            {
                "key": "data_quality",
                "label": "Data quality",
                "score": dq_score,
                "status": status,
                "reason": dq_reason,
            }
        )

    # ======================================================================
    # 9) Robustness dimension (רעיון חדש #7)
    # ======================================================================
    rob_score: Optional[int] = None
    rob_reason = ""
    if portfolio is not None:
        is_robust = find_attr(
            portfolio,
            "is_robust",
            "robust",
            "passed_robustness",
        )
        n_scenarios = _safe_num(
            find_attr(
                portfolio,
                "n_scenarios",
                "wf_scenarios",
                "robustness_scenarios",
            )
        )
        wf_stability = _safe_num(
            find_attr(
                portfolio,
                "wf_stability_score",
                "walk_forward_stability",
            )
        )

        if isinstance(is_robust, bool) and is_robust:
            rob_score = 85
            rob_reason = "Marked as robust by optimizer."
        if n_scenarios is not None and n_scenarios >= 10:
            rob_score = max(rob_score or 75, 88)
            rob_reason += f" Tested on {int(n_scenarios)} scenarios."
        if wf_stability is not None:
            s = max(0.0, min(1.0, wf_stability))
            score_from_wf = int(60 + s * 35)
            rob_score = max(rob_score or score_from_wf, score_from_wf)
            rob_reason += f" Walk-forward stability≈{wf_stability:.2f}."

    if rob_score is not None:
        status = (
            "good"
            if rob_score >= 85
            else "ok" if rob_score >= 65 else "watch"
        )
        dims.append(
            {
                "key": "robustness",
                "label": "Robustness / WF / scenarios",
                "score": rob_score,
                "status": status,
                "reason": rob_reason,
            }
        )

    # ======================================================================
    # 10) Live drift dimension (רעיון חדש #8)
    # ======================================================================
    live_score: Optional[int] = None
    live_reason = ""
    if portfolio is not None:
        live_ret = _safe_num(
            find_attr(
                portfolio,
                "live_return",
                "live_pnl_pct",
                "live_ytd",
            )
        )
        model_ret = total_ret or sharpe

        if live_ret is not None and model_ret is not None:
            # בודקים כיוון גס: האם live מתיישב עם ה-backtest?
            if isinstance(model_ret, float) and abs(model_ret) < 1.0:
                # אם זה Sharpe ולא return, לא נשווה מספרית יותר מדי
                pass
            else:
                # מניחים ששניהם returns
                ratio = None
                try:
                    if model_ret != 0:
                        ratio = live_ret / model_ret
                except Exception:
                    ratio = None

                if ratio is not None:
                    if 0.8 <= ratio <= 1.2:
                        live_score = 80
                        live_reason = "Live vs backtest in line (ratio ~1)."
                    elif 0.6 <= ratio < 0.8 or 1.2 < ratio <= 1.5:
                        live_score = 65
                        live_reason = "Moderate drift between live and backtest."
                    else:
                        live_score = 50
                        live_reason = "Significant drift between live and backtest."

    if live_score is not None:
        status = (
            "good"
            if live_score >= 85
            else "ok" if live_score >= 65 else "watch"
        )
        dims.append(
            {
                "key": "live_drift",
                "label": "Live vs backtest drift",
                "score": live_score,
                "status": status,
                "reason": live_reason,
            }
        )

    # ======================================================================
    # 11) חישוב overall עם משקולות דינמיים (רעיון חדש #9 + #10)
    # ======================================================================
    if not dims:
        return {"overall_score": None, "dimensions": []}

    # בסיס משקולות
    base_weights: Dict[str, float] = {
        "performance": 1.6,
        "recent_vs_full": 1.2,
        "risk": 2.0,
        "tail_risk": 1.6,
        "stability": 1.2,
        "signals": 1.0,
        "diversification": 0.9,
        "macro": 0.8,
        "system": 1.4,
        "data_quality": 1.3,
        "robustness": 1.2,
        "live_drift": 1.0,
    }

    # התאמת משקולות לפי risk_profile
    if risk_profile == "defense":
        # מדגישים risk / tail / data / system, מקטינים חשיבות performance/ signals
        for k in ("risk", "tail_risk", "data_quality", "system", "stability"):
            base_weights[k] = base_weights.get(k, 1.0) * 1.3
        for k in ("performance", "signals", "recent_vs_full"):
            base_weights[k] = base_weights.get(k, 1.0) * 0.8
    elif risk_profile == "offense":
        # מדגישים performance / recent / signals, מרככים קצת risk
        for k in ("performance", "recent_vs_full", "signals"):
            base_weights[k] = base_weights.get(k, 1.0) * 1.3
        for k in ("risk", "tail_risk"):
            base_weights[k] = base_weights.get(k, 1.0) * 0.9

    # חישוב ממוצע משוקלל
    num = 0.0
    den = 0.0
    for d in dims:
        s = _safe_num(d.get("score"))
        if s is None:
            continue
        key = str(d.get("key"))
        w = base_weights.get(key, 1.0)
        num += float(s) * w
        den += w

    overall = None
    if den > 0:
        overall = int(round(num / den))
        overall = max(0, min(100, overall))

    # מסווגים סטטוס כללי (לא חובה לשימוש, אבל נחמד ל-UI)
    # (את הסטטוס הזה אפשר לחשב בטאב HOME מ-Overall)
    return {
        "overall_score": overall,
        "dimensions": dims,
    }

def _render_top_issues_panel(
    readiness: Dict[str, Any],
    alerts: List[Dict[str, Any]],
    health_light: Dict[str, Any],
) -> None:
    """
    מציג 'Top 3 things to look at today' על בסיס:
    - Investment Readiness dimensions (score נמוך)
    - health_light (critical/warnings)
    - alerts מה-HomeContext
    """
    st.markdown("### 🎯 Top 3 things to look at")

    issues: List[Dict[str, Any]] = []

    dims = readiness.get("dimensions", []) or []
    for d in dims:
        score = d.get("score")
        if not isinstance(score, (int, float)):
            continue
        label = d.get("label", d.get("key", ""))
        reason = d.get("reason", "")
        # ציון נמוך = בעיה – נסמן severity גבוה
        if score < 70:
            if score < 55:
                sev = 3  # חמור
            elif score < 65:
                sev = 2  # בינוני
            else:
                sev = 1  # קל
            issues.append(
                {
                    "source": "readiness",
                    "area": label,
                    "severity": sev,
                    "score": score,
                    "text": reason or f"Low score in {label}",
                }
            )

    # health_light – קריטי/אזהרות
    if health_light:
        if health_light.get("has_critical_issues"):
            for msg in health_light.get("issues", []) or ["Critical issues in system health."]:
                issues.append(
                    {
                        "source": "health",
                        "area": "System Health",
                        "severity": 3,
                        "score": 40,
                        "text": msg,
                    }
                )
        if health_light.get("has_warnings"):
            for msg in health_light.get("warnings", []) or ["Warnings in system health."]:
                issues.append(
                    {
                        "source": "health",
                        "area": "System Health",
                        "severity": 2,
                        "score": 60,
                        "text": msg,
                    }
                )

    # Alerts – warning/error מה-HomeContext
    for a in alerts or []:
        lvl = str(a.get("level", "")).lower()
        if lvl not in ("warning", "error"):
            continue
        sev = 3 if lvl == "error" else 2
        msg = a.get("message", "")
        src = a.get("source", "system")
        issues.append(
            {
                "source": "alert",
                "area": f"Alert: {src}",
                "severity": sev,
                "score": 50 if lvl == "error" else 65,
                "text": msg,
            }
        )

    if not issues:
        st.caption("No major issues detected – focus on research / optimization.")
        return

    # ניקוי וכיבוד top 3 לפי severity ואז score
    issues_sorted = sorted(
        issues,
        key=lambda x: (x["severity"], -x.get("score", 0)),
        reverse=True,
    )
    top3 = issues_sorted[:3]

    for i, it in enumerate(top3, start=1):
        sev = it["severity"]
        if sev == 3:
            icon = "🚨"
        elif sev == 2:
            icon = "⚠️"
        else:
            icon = "ℹ️"
        area = it["area"]
        text = it["text"]
        score = it.get("score")
        score_txt = f"(score={score})" if score is not None else ""
        st.markdown(f"{i}. {icon} **{area}** {score_txt} – {text}")

def _render_daily_runbook_panel(
    snapshot: DashboardSnapshot,
    home_ctx: Dict[str, Any],
) -> None:
    """
    Daily Runbook / Checklist אוטומטי:

    בודק:
    - Macro tab רץ היום? (macro_metrics קיימים?)
    - Risk limits נבדקו? (אין limits_breached)
    - Data feeds תקינים? (data_quality / health_light)
    - IB מחובר? (ib_connection_status)
    - Experiment/optimization רצו ב-24 שעות האחרונות? (experiments_tail)
    """
    st.markdown("### 📋 Daily runbook")

    today = date.today()

    macro_metrics = st.session_state.get("macro_metrics", {})
    risk_metrics = st.session_state.get("risk_metrics", {})
    data_quality = home_ctx.get("data_quality", []) or []
    experiments_tail = home_ctx.get("experiments_tail", []) or []
    health_light = home_ctx.get("health_light", {}) or {}

    # 1) Macro snapshot up-to-date
    macro_ok = bool(macro_metrics)
    st.checkbox("Macro snapshot updated (run Macro tab)", value=macro_ok, disabled=True)

    # 2) Risk limits OK (אין breached)
    risk_ok = not getattr(snapshot.risk, "limits_breached", False)
    st.checkbox("Risk limits OK (no breached limits)", value=risk_ok, disabled=True)

    # 3) IB connected
    ib = get_ib_instance(readonly=True, use_singleton=True)
    ib_status = ib_connection_status(ib)
    ib_ok = bool(ib_status.get("connected", False))
    st.checkbox("IBKR connected (Gateway/TWS up)", value=ib_ok, disabled=True)

    # 4) Data feeds healthy
    dq_ok = True
    if data_quality:
        # אם יש item אחד עם status down/error → לא ok
        for dq in data_quality:
            status = str(dq.get("status", "")).lower()
            if status in ("down", "error", "critical"):
                dq_ok = False
                break
    else:
        # fallback ל-health_light
        dq_ok = not health_light.get("has_critical_issues", False)
    st.checkbox("Data feeds healthy (no critical data-quality issues)", value=dq_ok, disabled=True)

    # 5) Optimization / Experiments ran recently
    exp_ok = False
    cutoff_ts = datetime.now(timezone.utc).timestamp() - 24 * 3600
    for e in experiments_tail:
        ts = e.get("ts_utc")
        if not ts:
            continue
        try:
            # פורמט ISO
            dt = datetime.fromisoformat(ts.replace("Z", ""))
            if dt.timestamp() >= cutoff_ts:
                exp_ok = True
                break
        except Exception:
            continue
    st.checkbox("Recent optimization / experiment run (last 24h)", value=exp_ok, disabled=True)

    st.caption(
        "הצ'ק־ליסט הזה אוטומטי בלבד – הוא לא מחליף Runbook ידני / ביקורת אנושית, "
        "אבל הוא עוזר לזהות חורים שדורשים תשומת לב לפני שמפעילים מסחר חי."
    )


def build_performance_tiles_data(snapshot: DashboardSnapshot) -> Dict[str, Any]:
    """
    בונה נתונים ל-Tiles של ביצועים:

    מחפש:
        - snapshot.portfolio.pnl (today / mtd / ytd)
        - snapshot.portfolio.nav / total_equity
        - snapshot.ctx.extra:
            * 'pnl_mtd', 'pnl_ytd'
            * 'nav_since_inception_start'
            * 'pnl_since_inception' (אם כבר חושב ב-Service)

    הפלט:
        {
            "today":  {"pnl": float, "nav": float},
            "mtd":    {"pnl": float},
            "ytd":    {"pnl": float},
            "since":  {"pnl": float|None, "nav_start": float|None, "nav": float},
        }
    """
    p = snapshot.portfolio
    ctx = snapshot.ctx
    extra = getattr(ctx, "extra", {}) or {}

    pnl = p.pnl

    today_pnl = float(pnl.total_today() or 0.0)
    nav_now = float(p.total_equity or p.nav or 0.0)

    mtd_pnl = float(extra.get("pnl_mtd", pnl.realized_mtd or 0.0) or 0.0)
    ytd_pnl = float(extra.get("pnl_ytd", pnl.realized_ytd or 0.0) or 0.0)

    nav_start = extra.get("nav_since_inception_start")
    since_pnl = extra.get("pnl_since_inception")
    if since_pnl is None and nav_start is not None:
        try:
            since_pnl = nav_now - float(nav_start)
        except Exception:
            since_pnl = None

    data = {
        "today": {
            "pnl": today_pnl,
            "nav": nav_now,
        },
        "mtd": {
            "pnl": mtd_pnl,
        },
        "ytd": {
            "pnl": ytd_pnl,
        },
        "since": {
            "pnl": float(since_pnl) if since_pnl is not None else None,
            "nav_start": float(nav_start) if nav_start is not None else None,
            "nav": nav_now,
        },
    }
    return data


def build_equity_curve_df(snapshot: DashboardSnapshot) -> pd.DataFrame:
    """
    בונה Equity curve קצרה לדשבורד.

    ציפייה:
        snapshot.ctx.extra יכול להכיל:
            - 'nav_history': list of dicts עם 'ts', 'nav', 'bench'
        או:
            - 'nav_history_df': DataFrame עם columns=['ts','nav','bench']

    אם אין, מחזירים DataFrame ריק.
    """
    extra = getattr(snapshot.ctx, "extra", {}) or {}
    history_df = None

    if "nav_history_df" in extra:
        try:
            history_df = extra["nav_history_df"]
            if isinstance(history_df, pd.DataFrame):
                return history_df
        except Exception:
            history_df = None

    if "nav_history" in extra:
        try:
            hist_list = extra["nav_history"] or []
            history_df = pd.DataFrame(hist_list)
        except Exception:
            history_df = None

    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=["ts", "nav", "bench"])

    # מוודאים שיש עמודות בסיסיות
    if "ts" not in history_df.columns:
        history_df["ts"] = pd.date_range(
            end=snapshot.as_of, periods=len(history_df), freq="D"
        )
    if "nav" not in history_df.columns:
        history_df["nav"] = None
    if "bench" not in history_df.columns:
        history_df["bench"] = None

    return history_df[["ts", "nav", "bench"]]


def build_pnl_attribution_df(snapshot: DashboardSnapshot) -> pd.DataFrame:
    """
    PnL Attribution לפי Asset Class / Strategy / Desk.

    מתבסס על:
        snapshot.portfolio.pnl.pnl_by_asset_class
        snapshot.portfolio.pnl.pnl_by_strategy
        snapshot.portfolio.pnl.pnl_by_desk
    """
    pnl = snapshot.portfolio.pnl
    rows: List[Dict[str, Any]] = []

    def add_rows(source: Dict[str, float], dim: str) -> None:
        if not source:
            return
        n = max(1, len(source))
        for k, v in source.items():
            rows.append(
                {
                    "Dimension": dim,
                    "Bucket": k,
                    "PnL_Today": float(v or 0.0),
                    "PnL_MTD": float(pnl.realized_mtd or 0.0) / n,
                    "PnL_YTD": float(pnl.realized_ytd or 0.0) / n,
                }
            )

    add_rows(pnl.pnl_by_asset_class, "AssetClass")
    add_rows(pnl.pnl_by_strategy, "Strategy")
    add_rows(pnl.pnl_by_desk, "Desk")

    if not rows:
        return pd.DataFrame(
            columns=["Dimension", "Bucket", "PnL_Today", "PnL_MTD", "PnL_YTD"]
        )

    df = pd.DataFrame(rows)
    df["abs_today"] = df["PnL_Today"].abs()
    df = df.sort_values("abs_today", ascending=False).drop(columns=["abs_today"])
    return df


def build_exposure_df(exposure: PortfolioExposureBreakdown) -> pd.DataFrame:
    """
    סיכום חשיפות לפי Asset Class / Sector / Currency / Country.
    """
    rows: List[Dict[str, Any]] = []

    def add_rows_from_dict(d: Dict[str, float], label: str) -> None:
        for k, v in (d or {}).items():
            rows.append(
                {
                    "Dimension": label,
                    "Bucket": k,
                    "Exposure": float(v or 0.0),
                }
            )

    add_rows_from_dict(exposure.by_asset_class, "Asset Class")
    add_rows_from_dict(exposure.by_sector, "Sector")
    add_rows_from_dict(exposure.by_currency, "Currency")
    add_rows_from_dict(exposure.by_country, "Country")

    if not rows:
        return pd.DataFrame(columns=["Dimension", "Bucket", "Exposure"])

    df = pd.DataFrame(rows)
    df["abs_exposure"] = df["Exposure"].abs()
    return df.sort_values("abs_exposure", ascending=False)


def build_exposure_drift_df(snapshot: DashboardSnapshot) -> pd.DataFrame:
    """
    Exposure Drift vs Target:

    ציפייה:
        snapshot.ctx.extra.get("target_exposures") → dict:
            {
                "by_asset_class": {"Equities": 0.7, "Rates": 0.2, ...},
                "by_sector": {"Tech": 0.2, "Financials": 0.15, ...},
                ...
            }

    הפלט:
        Dimension, Bucket, Actual, Target, Drift
    """
    ctx = snapshot.ctx
    exposure = snapshot.portfolio.exposure
    extra = getattr(ctx, "extra", {}) or {}
    targets = extra.get("target_exposures", {}) or {}

    rows: List[Dict[str, Any]] = []

    def add_dim(actual: Dict[str, float], target_key: str, dim_label: str) -> None:
        # actual ו-target אמורים להיות ב-% מה-NAV או ב-notional יחסי
        target_map = targets.get(target_key, {}) or {}
        all_buckets = set(actual.keys()) | set(target_map.keys())
        for bucket in all_buckets:
            a = float(actual.get(bucket, 0.0) or 0.0)
            t = float(target_map.get(bucket, 0.0) or 0.0)
            rows.append(
                {
                    "Dimension": dim_label,
                    "Bucket": bucket,
                    "Actual": a,
                    "Target": t,
                    "Drift": a - t,
                }
            )

    add_dim(exposure.by_asset_class, "by_asset_class", "Asset Class")
    add_dim(exposure.by_sector, "by_sector", "Sector")
    add_dim(exposure.by_country, "by_country", "Country")

    if not rows:
        return pd.DataFrame(columns=["Dimension", "Bucket", "Actual", "Target", "Drift"])

    df = pd.DataFrame(rows)
    df["abs_drift"] = df["Drift"].abs()
    df = df.sort_values("abs_drift", ascending=False).drop(columns=["abs_drift"])
    return df


def build_liquidity_snapshot_df(snapshot: PortfolioSnapshot) -> pd.DataFrame:
    """
    Liquidity Snapshot:

    מחפש במטא-דאטה של PositionSnapshot:
        - 'avg_daily_volume' / 'ADV'
        - 'bid_ask_spread_bps' / 'spread_bps'
        - 'liq_score' / 'liquidity_score'

    הפלט:
        Symbol, Side, Weight, MV, ADV, Spread_bps, LiquidityScore
    """
    rows: List[Dict[str, Any]] = []
    for p in snapshot.positions:
        md = p.metadata or {}
        adv = md.get("avg_daily_volume") or md.get("ADV")
        spread = md.get("bid_ask_spread_bps") or md.get("spread_bps")
        liq_score = md.get("liq_score") or md.get("liquidity_score")

        rows.append(
            {
                "Symbol": p.symbol,
                "Side": p.side,
                "Weight": float(p.weight or 0.0),
                "MV": float(p.market_value or 0.0),
                "ADV": adv,
                "Spread_bps": spread,
                "LiquidityScore": liq_score,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Side",
                "Weight",
                "MV",
                "ADV",
                "Spread_bps",
                "LiquidityScore",
            ]
        )

    df = pd.DataFrame(rows)
    df["abs_weight"] = df["Weight"].abs()
    df = df.sort_values("abs_weight", ascending=False).drop(columns=["abs_weight"])
    return df


def build_concentration_df(snapshot: PortfolioSnapshot) -> pd.DataFrame:
    """
    ריכוזיות הפורטפוליו: top 10 positions לפי weight.

    הפלט:
        Rank, Symbol, Side, Weight, MV
    """
    rows: List[Dict[str, Any]] = []
    for p in snapshot.positions:
        rows.append(
            {
                "Symbol": p.symbol,
                "Side": p.side,
                "Weight": float(p.weight or 0.0),
                "MV": float(p.market_value or 0.0),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Rank", "Symbol", "Side", "Weight", "MV"])

    df = pd.DataFrame(rows)
    df["abs_weight"] = df["Weight"].abs()
    df = df.sort_values("abs_weight", ascending=False).head(10)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.drop(columns=["abs_weight"])


def build_turnover_summary(snapshot: PortfolioSnapshot) -> Dict[str, Optional[float]]:
    """
    Turnover summary:

        snapshot.turnover_1d, turnover_5d, turnover_20d
    """
    return {
        "turnover_1d": snapshot.turnover_1d,
        "turnover_5d": snapshot.turnover_5d,
        "turnover_20d": snapshot.turnover_20d,
    }
def _render_home_execution_panel(app_ctx: AppContext, feature_flags: Optional[FeatureFlags] = None) -> None:

    """
    פאנל ביצוע ידני קטן מתוך ה-Home של הדשבורד.

    זה מיועד ל:
    - בדיקת חיבור ל-IBKR.
    - שליחת עסקת זוג פשוטה (Pair Trade) בצורה מבוקרת.
    - Kill-switch בסיסי (Cancel ALL).

    שמים פה הרבה "חגורות בטיחות" כדי לא לעשות טעות בלייב.
    """
    import streamlit as st

    st.markdown("### 🔌 IBKR Execution (Mini Panel)")

    enable_live = False
    if feature_flags is not None:
        try:
            enable_live = bool(feature_flags.get("enable_live_trading_actions", False))
        except Exception:
            enable_live = False

    # ודא שיש Router
    ib_router = getattr(app_ctx, "ib_router", None)
    if ib_router is None:
        # אם יש לך מתודה init_ib_router ב-AppContext – נשתמש בה
        if hasattr(app_ctx, "init_ib_router"):
            app_ctx.init_ib_router()
            ib_router = getattr(app_ctx, "ib_router", None)

    # סטטוס חיבור
    # אם כבר יש singleton מחובר – get_ib_instance יחזיר אותו; אחרת פשוט ננסה לבדוק סטטוס
    ib = get_ib_instance(readonly=True, use_singleton=True)  # לא שולח פקודות, רק חיבור
    status = ib_connection_status(ib)

    cols_status = st.columns(3)
    with cols_status[0]:
        st.metric("Connected", "✅" if status.get("connected") else "❌")
    with cols_status[1]:
        st.metric("Host", str(status.get("host", "N/A")))
    with cols_status[2]:
        st.metric("Port", str(status.get("port", "N/A")))

    if not status.get("connected"):
        st.info("לא מחובר ל-IBKR כרגע. פתח TWS / Gateway ואז רענן את הדשבורד.")
        return

    if ib_router is None:
        st.error("IBOrderRouter לא מאותחל ב-AppContext (ib_router=None).")
        return

    st.markdown(
        "<div style='padding:6px 10px;border-radius:8px;background:#1f2933;color:#f9fafb;font-size:13px;'>"
        "⚠️ זה פאנל ביצוע אמיתי. מומלץ להתחיל ב-PAPER בלבד, ולהגביל כמויות קטנות."
        "</div>",
        unsafe_allow_html=True,
    )

    # --- טופס לעסקת זוג פשוטה ---
    with st.form("home_exec_form"):
        c1, c2 = st.columns(2)
        with c1:
            sym_a = st.text_input("Symbol A", value="SPY")
            side_a = st.selectbox("Side A", ["BUY", "SELL"], key="home_exec_side_a")
            qty_a = st.number_input("Qty A", min_value=1.0, value=10.0, step=1.0, key="home_exec_qty_a")
        with c2:
            sym_b = st.text_input("Symbol B", value="QQQ")
            side_b = st.selectbox("Side B", ["BUY", "SELL"], key="home_exec_side_b")
            qty_b = st.number_input("Qty B", min_value=1.0, value=10.0, step=1.0, key="home_exec_qty_b")

        pair_id = st.text_input("Pair ID", value=f"{sym_a}-{sym_b}")
        account = st.text_input("Account (אופציונלי)", value="")

        st.markdown("---")
        confirm = st.checkbox("אני מבין שזו עסקה אמיתית (מומלץ PAPER בלבד בהתחלה)")

        submitted = st.form_submit_button(
            "🚀 שלח עסקת זוג (Market)",
            type="primary",
            disabled=not enable_live,
        )
        if not enable_live:
            st.caption("Live trading actions are currently **blocked** (enable_live_trading_actions=False).")


    if submitted:
        if not confirm:
            st.error("סמן את תיבת האישור לפני ביצוע עסקה.")
            return

        req = PairOrderRequest(
            pair_id=pair_id,
            legs=[
                PairOrderLeg(symbol=sym_a, action=side_a, quantity=qty_a),
                PairOrderLeg(symbol=sym_b, action=side_b, quantity=qty_b),
            ],
            order_type="MKT",
            time_in_force="DAY",
            account=account or None,
            allow_partial=False,
            tags={"source": "home_tab_manual"},
        )

        # risk_params בסיסיים – אתה יכול לשפר לפי ה-Risk Engine שלך
        risk_params = {
            "max_quantity_per_leg": app_ctx.settings.get("max_qty_per_leg", 10_000),
            "allowed_symbols": app_ctx.settings.get("allowed_symbols", None),
        }

        with st.spinner("שולח הוראות ל-IBKR..."):
            result = ib_router.submit_pair_order(req, risk_params=risk_params)

        if result.success:
            st.success(f"Orders נשלחו בהצלחה. IDs: {result.order_ids}")
            st.json(result.details, expanded=False)
        else:
            st.error(f"נכשל: {result.error}")
            if result.details:
                st.json(result.details, expanded=False)

    # Kill-switch בסיסי
    st.markdown("---")
    col_kill, col_note = st.columns([1, 2])
    with col_kill:
        kill = st.button(
            "🧨 Cancel ALL open orders (GLOBAL)",
            help="זה מבטל *כל* ההוראות הפתוחות בחשבון. להשתמש בזהירוּת.",
        )
    with col_note:
        st.caption("Kill-switch בסיסי. לבקרת סיכון אמיתית כדאי לבנות Risk Engine סביב זה.")

    if kill:
        ok = ib_router.cancel_all_open_orders()
        if ok:
            st.warning("נשלחה בקשת Cancel Global לכל ההוראות הפתוחות בחשבון.")
        else:
            st.error("כשל בשליחת Cancel Global ל-IB.")


# ============================================================
# Data Builders — Risk: Budget, Drawdown, Tail, Beta/TE
# ============================================================


def build_risk_budget_summary(snapshot: DashboardSnapshot) -> Dict[str, Any]:
    """
    Risk Budget vs Usage:

    מתבסס על:
        - snapshot.ctx.target_vol_annual / max_vol_annual
        - snapshot.risk.portfolio_risk.vol_annual
        - snapshot.risk.portfolio_risk.var_95 / es_95
    """
    ctx = snapshot.ctx
    pr = snapshot.risk.portfolio_risk

    target_vol = getattr(ctx, "target_vol_annual", None)
    max_vol = getattr(ctx, "max_vol_annual", None)
    vol = pr.vol_annual

    usage_vol = None
    if target_vol:
        try:
            usage_vol = vol / target_vol
        except Exception:
            usage_vol = None

    return {
        "target_vol": target_vol,
        "max_vol": max_vol,
        "vol": vol,
        "usage_vs_target": usage_vol,
        "var_95": pr.var_95,
        "es_95": pr.es_95,
    }


def build_drawdown_state(snapshot: DashboardSnapshot) -> Dict[str, Any]:
    """
    Drawdown state:

    מתבסס על:
        - snapshot.risk.portfolio_risk.max_drawdown_1y / max_drawdown_itd
        - ctx.drawdown_soft_limit / drawdown_hard_limit
    """
    ctx = snapshot.ctx
    pr = snapshot.risk.portfolio_risk

    dd_1y = pr.max_drawdown_1y
    dd_itd = pr.max_drawdown_itd

    soft = getattr(ctx, "drawdown_soft_limit", DD_WARNING)
    hard = getattr(ctx, "drawdown_hard_limit", DD_CRITICAL)

    return {
        "dd_1y": dd_1y,
        "dd_itd": dd_itd,
        "soft_limit": soft,
        "hard_limit": hard,
    }


def build_tail_risk_summary(snapshot: DashboardSnapshot) -> Dict[str, Any]:
    """
    Tail risk summary:

    מתבסס על:
        - snapshot.risk.portfolio_risk.tail_risk_index
        - snapshot.risk.portfolio_risk.stress_test_scenarios

    הפלט:
        { "tail_index": float|None,
          "scenarios": {name: pnl_impact, ...}
        }
    """
    pr = snapshot.risk.portfolio_risk
    tail = pr.tail_risk_index
    scenarios = pr.stress_test_scenarios or {}
    return {
        "tail_index": tail,
        "scenarios": scenarios,
    }


def build_beta_te_summary(snapshot: DashboardSnapshot) -> Dict[str, Any]:
    """
    Beta & Tracking Error summary:

    מתבסס על:
        - snapshot.risk.portfolio_risk.beta_vs_bench
        - snapshot.risk.portfolio_risk.corr_vs_bench
        - snapshot.risk.portfolio_risk.tracking_error_vs_bench
    """
    pr = snapshot.risk.portfolio_risk
    return {
        "beta": pr.beta_vs_bench,
        "corr": pr.corr_vs_bench,
        "tracking_error": pr.tracking_error_vs_bench,
    }


# ============================================================
# Data Builders — Signals: Core, Funnel, Ladder, Conflicts, Aging
# ============================================================


def build_signals_core_df(snapshot: DashboardSnapshot) -> pd.DataFrame:
    """
    DataFrame בסיסי לכל הסיגנלים, כולל גיל יחסית ל-snapshot.as_of.
    """
    signals = snapshot.signals
    rows: List[Dict[str, Any]] = []

    for s in signals.items:
        try:
            pair = f"{s.symbol_1}/{s.symbol_2}" if s.symbol_2 else s.symbol_1
        except Exception:
            pair = s.symbol_1

        age_days = None
        if s.created_at is not None:
            try:
                age_days = (snapshot.as_of - s.created_at).total_seconds() / 86400.0
            except Exception:
                age_days = None

        rows.append(
            {
                "Pair": pair,
                "Symbol_1": s.symbol_1,
                "Symbol_2": s.symbol_2,
                "Direction": s.direction,
                "Confidence": s.confidence,
                "Edge": s.edge,
                "Z": s.zscore,
                "AbsZ": abs(s.zscore) if s.zscore is not None else None,
                "HalfLife": s.half_life,
                "Corr": s.corr,
                "Quality": s.quality_score,
                "Regime": s.regime,
                "TF": s.time_frame,
                "Strategy": s.strategy_family,
                "SubStrategy": s.sub_strategy,
                "Model": s.model_name,
                "CreatedAt": s.created_at,
                "ExpiresAt": s.expires_at,
                "AgeDays": age_days,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Pair",
                "Symbol_1",
                "Symbol_2",
                "Direction",
                "Confidence",
                "Edge",
                "Z",
                "AbsZ",
                "HalfLife",
                "Corr",
                "Quality",
                "Regime",
                "TF",
                "Strategy",
                "SubStrategy",
                "Model",
                "CreatedAt",
                "ExpiresAt",
                "AgeDays",
            ]
        )

    return pd.DataFrame(rows)


def build_signal_funnel_metrics(snapshot: DashboardSnapshot) -> Dict[str, Any]:
    """
    Signal Funnel Overview:

    ctx.extra:
        - 'universe_size'
        - 'filtered_candidates'
        - 'backtested_candidates'
        - 'deployed_signals'
    """
    signals = snapshot.signals
    ctx = snapshot.ctx
    extra = getattr(ctx, "extra", {}) or {}

    universe_size = extra.get("universe_size")
    if universe_size is None:
        universe_size = max(len(signals.items), 0)

    filtered_candidates = extra.get("filtered_candidates", None)
    backtested_candidates = extra.get("backtested_candidates", None)
    deployed_signals = extra.get("deployed_signals", None)

    return {
        "universe_size": int(universe_size or 0),
        "filtered_candidates": int(filtered_candidates or 0),
        "signal_count": int(signals.n_total or len(signals.items)),
        "deployed_signals": int(deployed_signals or 0),
        "backtested_candidates": int(backtested_candidates or 0),
    }


def build_signal_ladder_df(snapshot: DashboardSnapshot, limit: int = 20) -> pd.DataFrame:
    """
    Signal Quality Ladder — מיון לפי Quality ואח"כ |Z|.
    """
    df = build_signals_core_df(snapshot)
    if df.empty:
        return df

    sort_cols = []
    if "Quality" in df.columns:
        sort_cols.append(("Quality", False))
    if "AbsZ" in df.columns:
        sort_cols.append(("AbsZ", False))

    if not sort_cols:
        return df.head(limit)

    by = [c for c, _ in sort_cols]
    ascending = [asc for _, asc in sort_cols]

    df = df.sort_values(by=by, ascending=ascending)
    return df.head(limit)


def build_signal_conflicts_df(snapshot: DashboardSnapshot) -> pd.DataFrame:
    """
    Conflict Detector בסיסי:

    Pair אחד עם גם LONG וגם SHORT → קונפליקט.
    """
    df = build_signals_core_df(snapshot)
    if df.empty:
        return df

    groups = df.groupby("Pair")
    rows: List[Dict[str, Any]] = []

    for pair, g in groups:
        dirs = set(g["Direction"].dropna().str.upper())
        if len(dirs) > 1 and {"LONG", "SHORT"} <= dirs:
            rows.append(
                {
                    "Pair": pair,
                    "Directions": ", ".join(sorted(dirs)),
                    "Count": len(g),
                    "MaxAbsZ": g["AbsZ"].max(),
                    "MaxQuality": g["Quality"].max(),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Pair", "Directions", "Count", "MaxAbsZ", "MaxQuality"])

    df_conf = pd.DataFrame(rows)
    return df_conf.sort_values("MaxQuality", ascending=False)


def build_signal_aging_distribution(snapshot: DashboardSnapshot) -> Dict[str, int]:
    """
    גיל סיגנלים:
        Fresh  : AgeDays <= 2h (~0.083d)
        Today  : AgeDays <= 1d
        Recent : AgeDays <= 3d
        Stale  : AgeDays > 3d
    """
    df = build_signals_core_df(snapshot)
    if df.empty or "AgeDays" not in df.columns:
        return {"Fresh": 0, "Today": 0, "Recent": 0, "Stale": 0}

    age = df["AgeDays"].dropna()

    fresh = (age <= (2.0 / 24.0)).sum()
    today = ((age > (2.0 / 24.0)) & (age <= 1.0)).sum()
    recent = ((age > 1.0) & (age <= 3.0)).sum()
    stale = (age > 3.0).sum()

    return {
        "Fresh": int(fresh),
        "Today": int(today),
        "Recent": int(recent),
        "Stale": int(stale),
    }


def build_pairs_heatmap_df(snapshot: DashboardSnapshot) -> pd.DataFrame:
    """
    Pairs Heatmap:

    צור DataFrame שמכיל:
        Pair, AbsZ, HalfLife, Corr, Strategy

    מיועד ל-plot (β vs HL vs |Z| וכו', בהמשך).
    """
    df = build_signals_core_df(snapshot)
    if df.empty:
        return df

    cols = ["Pair", "AbsZ", "HalfLife", "Corr", "Strategy"]
    for col in cols:
        if col not in df.columns:
            df[col] = None

    return df[cols]


def build_watchlist_df(snapshot: DashboardSnapshot) -> pd.DataFrame:
    """
    Watchlist Integration:

    ציפייה:
        snapshot.ctx.extra.get("watchlist") → רשימת pairs/tickers
        או:
        signals עם tag 'watch' (אם מימשת tags ב-SignalItem.metadata).

    הפלט:
        Pair, Direction, Quality, Strategy, Source
    """
    ctx = snapshot.ctx
    extra = getattr(ctx, "extra", {}) or {}
    watchlist = extra.get("watchlist", []) or []

    rows: List[Dict[str, Any]] = []

    # מקור 1: watchlist מ-extra
    for w in watchlist:
        if isinstance(w, dict):
            pair = w.get("pair") or w.get("symbol")
            rows.append(
                {
                    "Pair": pair,
                    "Direction": w.get("direction"),
                    "Quality": w.get("quality"),
                    "Strategy": w.get("strategy"),
                    "Source": "manual",
                }
            )
        else:
            rows.append(
                {
                    "Pair": str(w),
                    "Direction": None,
                    "Quality": None,
                    "Strategy": None,
                    "Source": "manual",
                }
            )

    # מקור 2: signals עם tag 'watch'
    for s in snapshot.signals.items:
        tags = getattr(s, "tags", []) or []
        if "watch" in [t.lower() for t in tags]:
            pair = f"{s.symbol_1}/{s.symbol_2}" if s.symbol_2 else s.symbol_1
            rows.append(
                {
                    "Pair": pair,
                    "Direction": s.direction,
                    "Quality": s.quality_score,
                    "Strategy": s.strategy_family,
                    "Source": "signal_tag",
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Pair", "Direction", "Quality", "Strategy", "Source"])

    df = pd.DataFrame(rows)
    # dedup
    df = df.drop_duplicates(subset=["Pair", "Source"], keep="first")
    return df


# ============================================================
# Data Builders — Macro / Cross-Asset / Vol Grid / Regime
# ============================================================


def build_cross_asset_snapshot_df(snapshot: DashboardSnapshot) -> pd.DataFrame:
    """
    Cross-Asset Snapshot:

    משתמש ב:
        - snapshot.market.bench_ret_* (benchmark)
        - snapshot.market.factor_returns (אם יש)
        - ctx.extra.get("cross_asset") (אם יש Svc שמזריק)
    """
    market = snapshot.market
    ctx = snapshot.ctx
    extra = getattr(ctx, "extra", {}) or {}

    rows: List[Dict[str, Any]] = []

    # benchmark
    rows.append(
        {
            "Asset": ctx.benchmark,
            "Group": "Equity",
            "Ret_1D": market.bench_ret_1d,
            "Ret_5D": market.bench_ret_5d,
            "Ret_30D": market.bench_ret_30d,
        }
    )

    # factors מתוך MarketSnapshot.factor_returns
    for name, ret in (market.factor_returns or {}).items():
        g = "Factor"
        upper = str(name).upper()
        if any(k in upper for k in ("SPX", "NDX", "RUT", "DAX", "EURO")):
            g = "Equity"
        elif any(k in upper for k in ("UST", "BOND", "RATE", "YIELD", "TNX")):
            g = "Rates"
        elif any(k in upper for k in ("HY", "IG", "CREDIT", "CDX")):
            g = "Credit"
        elif any(k in upper for k in ("FX", "USD", "EUR", "JPY", "AUD", "GBP")):
            g = "FX"
        elif any(k in upper for k in ("OIL", "BRENT", "WTI", "GOLD", "SILVER", "COPPER")):
            g = "Commodities"

        rows.append(
            {
                "Asset": name,
                "Group": g,
                "Ret_1D": ret,
                "Ret_5D": None,
                "Ret_30D": None,
            }
        )

    # cross_asset מפורש מה-extra (אם יש)
    ca = extra.get("cross_asset", [])
    for item in ca or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Asset": item.get("asset"),
                "Group": item.get("group", "Other"),
                "Ret_1D": item.get("ret_1d"),
                "Ret_5D": item.get("ret_5d"),
                "Ret_30D": item.get("ret_30d"),
            }
        )

    df = pd.DataFrame(rows)
    return df


def build_volatility_grid_df(snapshot: DashboardSnapshot) -> pd.DataFrame:
    """
    Volatility Grid:

    משתמש ב:
        - snapshot.market.factor_zscores: למשל 'EQUITY_VOL', 'RATES_VOL' וכו'.
    """
    market = snapshot.market
    rows: List[Dict[str, Any]] = []

    for name, z in (market.factor_zscores or {}).items():
        rows.append(
            {
                "Factor": name,
                "Z_Vol": z,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Factor", "Z_Vol"])

    df = pd.DataFrame(rows)
    return df.sort_values("Z_Vol", ascending=False)


def build_macro_regime_summary(snapshot: DashboardSnapshot) -> Dict[str, Any]:
    """
    Macro regime summary:

    משתמש ב:
        - ctx.macro_regime_hint
        - ctx.vol_regime_hint
        - ctx.regime_confidence
    """
    ctx = snapshot.ctx
    return {
        "macro_regime": getattr(ctx, "macro_regime_hint", None),
        "vol_regime": getattr(ctx, "vol_regime_hint", None),
        "confidence": getattr(ctx, "regime_confidence", 0.0),
    }


# ============================================================
# Data Builders — System / Errors / Agents / Latency / Resources
# ============================================================


def build_error_summary_df(system: SystemHealthSnapshot) -> pd.DataFrame:
    """
    טבלת שגיאות אחרונות:

        Index, Message
    """
    errors = system.recent_errors or []
    rows = [{"Index": i + 1, "Message": str(msg)} for i, msg in enumerate(errors)]
    return pd.DataFrame(rows)


def build_agents_status_df(system: SystemHealthSnapshot) -> pd.DataFrame:
    """
    טבלת Agents:

        Agent, Status
    """
    rows: List[Dict[str, Any]] = []
    for name, status in (system.agents_status or {}).items():
        rows.append(
            {
                "Agent": name,
                "Status": status,
            }
        )
    return pd.DataFrame(rows)


def build_latency_breakdown(snapshot: DashboardSnapshot) -> Dict[str, Any]:
    """
    Latency breakdown:

    ציפייה:
        snapshot.ctx.extra.get("latency_breakdown") → dict:
            {
                "market_data_ms": ...,
                "broker_ms": ...,
                "sql_ms": ...,
                "risk_engine_ms": ...,
                ...
            }
    """
    extra = getattr(snapshot.ctx, "extra", {}) or {}
    lb = extra.get("latency_breakdown", {}) or {}
    return {
        "market_data_ms": lb.get("market_data_ms"),
        "broker_ms": lb.get("broker_ms"),
        "sql_ms": lb.get("sql_ms"),
        "risk_engine_ms": lb.get("risk_engine_ms"),
        "other_ms": lb.get("other_ms"),
    }


def build_resource_utilization_dict(system: SystemHealthSnapshot) -> Dict[str, Any]:
    """
    ניצול משאבים (CPU/RAM):

    משתמש ב:
        - system.cpu_load_pct
        - system.memory_used_pct
    """
    return {
        "cpu_pct": system.cpu_load_pct,
        "ram_pct": system.memory_used_pct,
    }

# ============================================================
# UI Panels — Header, Meta & Story of the Day
# ============================================================


def _build_story_of_the_day(snapshot: DashboardSnapshot, diff: Optional[Dict[str, Any]]) -> str:
    """
    מייצר טקסט קצר בסגנון "Story of the Day" מבוסס:
    - PnL Today
    - שינוי NAV
    - מצב Macro Regime
    - מצב Vol / Risk
    """
    ctx = snapshot.ctx
    p = snapshot.portfolio
    r = snapshot.risk
    pr = r.portfolio_risk

    perf = build_performance_tiles_data(snapshot)
    today_pnl = perf["today"]["pnl"]
    nav = perf["today"]["nav"]

    macro = build_macro_regime_summary(snapshot)
    macro_regime = macro.get("macro_regime") or "Unknown"
    vol_regime = macro.get("vol_regime") or "Unknown"

    eq_delta = diff.get("equity_change") if diff else None

    # כיוון היום
    if today_pnl > 0:
        pnl_phrase = "היום בינתיים חיובי"
    elif today_pnl < 0:
        pnl_phrase = "היום בינתיים שלילי"
    else:
        pnl_phrase = "היום בערך שטוח"

    # גודל היום ביחס ל־NAV
    pnl_ratio = today_pnl / nav if nav else 0.0
    if abs(pnl_ratio) > 0.02:
        pnl_intensity = "מהותי"
    elif abs(pnl_ratio) > 0.005:
        pnl_intensity = "מתון"
    else:
        pnl_intensity = "קטן"

    # סיכון
    vol_text = _format_pct(pr.vol_annual)
    beta_text = f"{pr.beta_vs_bench:.2f}" if pr.beta_vs_bench is not None else "—"

    # שינוי NAV
    if eq_delta is not None and eq_delta != 0:
        nav_change_txt = f" שינוי NAV של {_format_num(eq_delta, digits=0)} היום"
    else:
        nav_change_txt = ""

    # Regime
    story = (
        f"{pnl_phrase} ({pnl_intensity}, PnL היום {_format_num(today_pnl, digits=0)}"
        f"{nav_change_txt}). "
        f"משטר מאקרו: **{macro_regime}**, משטר תנודתיות: **{vol_regime}**. "
        f"Vol שנתי משוער: {vol_text}, Beta לבנצ'מרק: {beta_text}."
    )

    if r.limits_breached:
        story += " ⚠️ קיימת חריגה בלימיטי סיכון — שווה לבדוק את פאנל ה-Risk."
    elif ctx.stress_mode:
        story += " מצב המערכת: **Stress mode** פעיל — דגש על הקטנת סיכון."

    return story


def render_header_panel(snapshot: DashboardSnapshot, diff: Optional[Dict[str, Any]]) -> None:
    """
    כותרת ראשית + Meta על המערכת + Story of the Day + Macro/Risk pills.
    """
    ctx = snapshot.ctx
    app_ctx = _get_global_app_context()
    host, user = _resolve_host_user()
    app_version, sql_schema = _get_app_version_and_schema(app_ctx)
    flags = _extract_feature_flags(ctx)
    macro = build_macro_regime_summary(snapshot)

    macro_regime = macro.get("macro_regime") or "Unknown"
    vol_regime = macro.get("vol_regime") or "Unknown"
    regime_conf = macro.get("confidence") or 0.0

    # שורת כותרת
    st.markdown(
        f"### 🏠 Dashboard v2 — {ctx.profile.upper()} / {ctx.env.upper()}  "
        f"({ctx.start_date} → {ctx.end_date})"
    )

    # שורת Meta ראשונה
    st.caption(
        f"Base currency: **{ctx.base_currency}** · "
        f"Benchmark: **{ctx.benchmark}** · "
        f"Portfolio: **{ctx.portfolio_id}** · "
        f"Timezone: **{ctx.timezone}**"
    )

    # שורת Meta שנייה (מערכת)
    meta_parts = [f"Host: {host}", f"User: {user}"]
    if app_version:
        meta_parts.append(f"App v{app_version}")
    if sql_schema:
        meta_parts.append(f"SQL schema: {sql_schema}")
    meta_parts.append(f"Last updated: {_now_utc_str()}")

    st.caption(" · ".join(meta_parts))

    # שורת "Pills" קצרה: Risk mode / Macro regime / Kill switch / Live actions
    pill_col1, pill_col2, pill_col3, pill_col4 = st.columns(4)

    risk_mode = (ctx.risk_profile or "balanced").capitalize()
    data_latency = getattr(ctx, "extra", {}).get("data_latency_mode", "end_of_day")
    enable_live = flags.get("enable_live_trading_actions", False)

    with pill_col1:
        st.markdown(f"**Risk mode:** `{risk_mode}`")
        st.markdown(f"**Latency:** `{data_latency}`")

    with pill_col2:
        st.markdown(f"**Macro regime:** `{macro_regime}`")
        st.markdown(f"**Vol regime:** `{vol_regime}`")

    kill_status = "Enabled" if ctx.kill_switch_enabled else "Disabled"
    with pill_col3:
        st.markdown(f"**Kill switch:** `{kill_status}`")
        st.markdown(f"**Stress mode:** `{'ON' if ctx.stress_mode else 'off'}`")

    with pill_col4:
        st.markdown(f"**Live actions:** `{ 'allowed' if enable_live else 'blocked' }`")
        st.markdown(f"**Env:** `{ctx.env}` / **Profile:** `{ctx.profile}`")

    # Story of the Day
    st.markdown("#### 🧾 Story of the Day")
    story = _build_story_of_the_day(snapshot, diff)
    st.write(story)

    # Feature flags (debug) כ-expander
    if flags.get("show_debug_info", False):
        with st.expander("Feature flags / Debug info", expanded=False):
            st.json(flags)


# ============================================================
# UI Panels — Performance & Equity Curve
# ============================================================


def render_performance_and_equity_panel(
    snapshot: DashboardSnapshot,
    diff: Optional[Dict[str, Any]],
) -> None:
    """
    Performance Tiles + Equity Curve + summary table:
    - Today / MTD / YTD / Since inception
    - שינוי NAV/VIX לעומת snapshot קודם
    - Equity curve קצרה לבנצ'מרק ולקרן
    - טבלת סיכום קטנה: return Today/MTD/YTD/Since vs benchmark (אם קיים ב-extra)
    """
    perf = build_performance_tiles_data(snapshot)
    eq_df = build_equity_curve_df(snapshot)

    p = snapshot.portfolio
    m = snapshot.market
    ctx = snapshot.ctx
    extra = getattr(ctx, "extra", {}) or {}

    equity_delta = diff.get("equity_change") if diff else None
    vix_delta = diff.get("vix_change") if diff else None

    today_nav = perf["today"]["nav"]
    today_pnl = perf["today"]["pnl"]
    pnl_ratio = today_pnl / today_nav if today_nav else 0.0

    # Return estimates אם ה-Service הכניס אותם ל-extra
    ret_today = extra.get("ret_today")
    ret_mtd = extra.get("ret_mtd")
    ret_ytd = extra.get("ret_ytd")
    ret_since = extra.get("ret_since_inception")

    # ---- KPI tiles (2 שורות) ----
    row1 = st.columns(5)
    row2 = st.columns(4)

    # שורה 1
    with row1[0]:
        st.metric(
            "PnL Today",
            _format_num(perf["today"]["pnl"], digits=0),
            _format_delta(equity_delta, digits=0),
        )
    with row1[1]:
        st.metric(
            "NAV / Total Equity",
            _format_num(perf["today"]["nav"], digits=0),
        )
    with row1[2]:
        st.metric(
            "Return Today",
            _format_pct(ret_today) if ret_today is not None else _format_pct(pnl_ratio),
        )
    with row1[3]:
        st.metric(
            "VIX",
            f"{m.vix_level:.2f}",
            _format_delta(vix_delta, digits=2),
        )
    with row1[4]:
        st.metric(
            "Exposure (Gross)",
            _format_num(p.gross_exposure, digits=0),
        )

    # שורה 2
    with row2[0]:
        st.metric(
            "PnL MTD",
            _format_num(perf["mtd"]["pnl"], digits=0),
        )
    with row2[1]:
        st.metric(
            "PnL YTD",
            _format_num(perf["ytd"]["pnl"], digits=0),
        )
    with row2[2]:
        st.metric(
            "Return MTD",
            _format_pct(ret_mtd) if ret_mtd is not None else "—",
        )
    with row2[3]:
        st.metric(
            "Return YTD",
            _format_pct(ret_ytd) if ret_ytd is not None else "—",
        )

    # ---- Equity curve ----
    if not eq_df.empty:
        fig = go.Figure()
        if "nav" in eq_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=eq_df["ts"],
                    y=eq_df["nav"],
                    mode="lines",
                    name="NAV",
                )
            )
        if "bench" in eq_df.columns and eq_df["bench"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=eq_df["ts"],
                    y=eq_df["bench"],
                    mode="lines",
                    name=f"Benchmark ({ctx.benchmark})",
                )
            )
        fig.update_layout(
            title="Equity curve (short horizon)",
            margin=dict(l=10, r=10, t=30, b=10),
            height=260,
            xaxis_title="Date",
            yaxis_title="Value",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("אין עדיין היסטוריית NAV זמינה (nav_history). אפשר לחבר דרך DashboardService.")

    # ---- טבלת סיכום ביצועים מול הבנצ'מרק ----
    ret_bench_today = m.bench_ret_1d
    ret_bench_5d = m.bench_ret_5d
    ret_bench_30d = m.bench_ret_30d

    rows = [
        {
            "Horizon": "Today",
            "Fund": ret_today,
            "Benchmark": ret_bench_today,
            "Excess": (ret_today - ret_bench_today) if (ret_today is not None and ret_bench_today is not None) else None,
        },
        {
            "Horizon": "5D",
            "Fund": extra.get("ret_5d"),
            "Benchmark": ret_bench_5d,
            "Excess": (extra.get("ret_5d") - ret_bench_5d) if (extra.get("ret_5d") is not None and ret_bench_5d is not None) else None,
        },
        {
            "Horizon": "30D",
            "Fund": extra.get("ret_30d"),
            "Benchmark": ret_bench_30d,
            "Excess": (extra.get("ret_30d") - ret_bench_30d) if (extra.get("ret_30d") is not None and ret_bench_30d is not None) else None,
        },
        {
            "Horizon": "YTD",
            "Fund": ret_ytd,
            "Benchmark": extra.get("bench_ret_ytd"),
            "Excess": (ret_ytd - extra.get("bench_ret_ytd")) if (ret_ytd is not None and extra.get("bench_ret_ytd") is not None) else None,
        },
    ]
    df_perf = pd.DataFrame(rows)
    if not df_perf.empty:
        st.markdown("##### 📈 Fund vs Benchmark (returns snapshot)")
        df_show = df_perf.copy()
        for col in ["Fund", "Benchmark", "Excess"]:
            df_show[col] = df_show[col].apply(lambda v: _format_pct(v) if v is not None else "—")
        st.dataframe(df_show, width="stretch", height=200)


# ============================================================
# UI Panels — PnL Attribution & Exposure / Drift / Liquidity / Concentration
# ============================================================



def render_pnl_attribution_panel(snapshot: DashboardSnapshot) -> None:
    """
    PnL Attribution panel:
    - בחירת Dimension (AssetClass / Strategy / Desk)
    - טבלת Attribution
    - bar chart ל-top Buckets
    """
    st.markdown("#### 💰 PnL Attribution")

    df = build_pnl_attribution_df(snapshot)
    if df.empty:
        st.info("אין כרגע נתוני Attribution מפורטים. אפשר להרחיב את PortfolioPnLBreakdown.")
        return

    dims = df["Dimension"].unique().tolist()
    dim = st.selectbox("Dimension", dims, index=0)

    df_dim = df[df["Dimension"] == dim].copy()

    col_table, col_chart = st.columns([2, 1])

    with col_table:
        st.dataframe(df_dim, width="stretch", height=260)

    with col_chart:
        df_plot = df_dim.sort_values("PnL_Today", ascending=False).head(10)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df_plot["Bucket"],
                    y=df_plot["PnL_Today"],
                )
            ]
        )
        fig.update_layout(
            title=f"PnL Today by {dim}",
            margin=dict(l=10, r=10, t=30, b=10),
            height=260,
            xaxis_title=dim,
            yaxis_title="PnL Today",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_exposure_and_concentration_panels(snapshot: DashboardSnapshot) -> None:
    """
    Panels לחשיפות, Drift מול Target, Liquidity ו-Concentration.
    """
    p = snapshot.portfolio
    ctx = snapshot.ctx

    col_left, col_right = st.columns([2, 1])

    # ---- צד שמאל: Exposure + Drift ----
    with col_left:
        st.markdown("#### 📊 Exposures & Drift vs Targets")

        exp_df = build_exposure_df(p.exposure)
        drift_df = build_exposure_drift_df(snapshot)

        if exp_df.empty:
            st.info("אין נתוני חשיפה מפורטים (by_asset_class/by_sector/by_country).")
        else:
            st.dataframe(exp_df, width="stretch", height=220)

        if not drift_df.empty:
            st.markdown("**Exposure drift vs targets**")
            st.dataframe(drift_df, width="stretch", height=220)
        else:
            st.caption("אין יעד חשיפות מוגדרים ב-ctx.extra['target_exposures'].")

    # ---- צד ימין: Liquidity / Concentration / Turnover ----
    with col_right:
        st.markdown("#### 💧 Liquidity & Concentration")

        # Liquidity
        liq_df = build_liquidity_snapshot_df(p)
        if not liq_df.empty:
            st.caption("Top illiquid / large positions (Liquidity snapshot)")
            st.dataframe(liq_df.head(10), width="stretch", height=150)
        else:
            st.caption("אין Metadata נזילות בפוזיציות (avg_daily_volume/spread).")

        # Concentration
        conc_df = build_concentration_df(p)
        if not conc_df.empty:
            st.caption("Top 10 by weight (Concentration)")
            st.dataframe(conc_df, width="stretch", height=160)

        # Turnover
        to = build_turnover_summary(p)
        st.markdown("**Turnover**")
        st.write(
            f"1D: {_format_pct(to.get('turnover_1d'))} · "
            f"5D: {_format_pct(to.get('turnover_5d'))} · "
            f"20D: {_format_pct(to.get('turnover_20d'))}"
        )

        # Highlight אם ריכוזיות חריגה
        try:
            max_single = getattr(ctx, "max_single_position_weight", 0.10) or 0.10
            top_weight = conc_df["Weight"].abs().max() if not conc_df.empty else 0.0
            if top_weight and top_weight > max_single:
                st.error(
                    f"⚠️ ריכוזיות חריגה: פוזיציה אחת עם משקל "
                    f"{_format_pct(top_weight)} מעל limit {_format_pct(max_single)}."
                )
        except Exception:
            pass

def _render_runtime_overview_strip_from_session() -> None:
    """
    מציג בפס עליון את ה-Overview / Alerts / Agent activity שמגיעים
    מה-runtime החדש (dashboard.py), אם הם זמינים ב-session_state.

    מצפה ל- dict בסגנון:
        st.session_state["dashboard_home_context"] = {
            "env": ...,
            "profile": ...,
            "run_id": ...,
            "overview_metrics": [...],
            "alerts": [...],
            "health_light": {...},
            "api_meta": {...},
            "saved_views": [...],
            "agent_actions_tail": [...],
        }
    """
    try:
        ctx = st.session_state.get("dashboard_home_context")
    except Exception:
        ctx = None

    if not isinstance(ctx, dict):
        return

    metrics = ctx.get("overview_metrics") or []
    alerts = ctx.get("alerts") or []
    health_light = ctx.get("health_light") or {}
    actions_tail = ctx.get("agent_actions_tail") or []

    # ---- Row 1: Overview cards ----
    if metrics:
        cols = st.columns(min(6, len(metrics)))
        for col, m in zip(cols, metrics):
            with col:
                label = m.get("label", m.get("key", "Metric"))
                value = m.get("value", "")
                level = (m.get("level") or "info").lower()
                desc = m.get("description") or ""

                icon = "ℹ️"
                if level == "success":
                    icon = "✅"
                elif level == "warning":
                    icon = "⚠️"
                elif level == "error":
                    icon = "🚨"

                st.markdown(f"{icon} **{label}**")
                st.markdown(f"### {value}")
                if desc:
                    st.caption(desc)

    # ---- Row 2: Alerts ----
    with st.expander("🚨 Alerts & notifications (from dashboard runtime)", expanded=bool(alerts)):
        if not alerts:
            st.caption("No dashboard-level alerts at the moment.")
        else:
            for a in alerts:
                lvl = str(a.get("level") or "info").lower()
                icon = "ℹ️"
                if lvl == "success":
                    icon = "✅"
                elif lvl == "warning":
                    icon = "⚠️"
                elif lvl == "error":
                    icon = "🚨"

                ts = a.get("ts_utc", "")
                src = a.get("source", "system")
                msg = a.get("message", "")
                st.write(f"{icon} `{ts}` • **{src}** – {msg}")

    # ---- Row 3: Agent actions tail ----
    if actions_tail:
        with st.expander("🤖 Recent agent actions (tail)", expanded=False):
            rows = []
            for item in actions_tail:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    {
                        "ts": item.get("ts_utc"),
                        "source": item.get("source"),
                        "action": item.get("action"),
                        "tab_key": item.get("tab_key"),
                        "result": item.get("result"),
                    }
                )
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, width="stretch", height=220)
            else:
                st.caption("No structured agent actions to display.")

def _render_home_top_tiles() -> None:
    """
    שורת Tiles עליונה להום:
    - Macro Score / Sensitivity (מהמאקרו)
    - Risk Exposure / Risk Score (מה-risk_metrics)
    - IB Status (ib_insync / Gateway)
    - Portfolio PnL / NAV / #Positions (מה-snapshot דרך session_state)
    """
    # מאקרו (ממולא על ידי macro_tab.push_macro_metrics_to_ctx)
    macro_metrics = st.session_state.get("macro_metrics", {}) or {}
    macro_score = macro_metrics.get("macro_score")
    macro_sens = macro_metrics.get("macro_sensitivity")

    # ריסק (ממולא על ידי macro_tab.push_risk_metrics_to_ctx)
    risk_metrics = st.session_state.get("risk_metrics", {}) or {}
    risk_label = st.session_state.get("risk_metrics_label", None)
    risk_exposure = risk_metrics.get("risk_exposure")
    risk_score = risk_metrics.get("risk_score")

    # IB Status (חיבור חי)
    ib = get_ib_instance(readonly=True, use_singleton=True)
    ib_status = ib_connection_status(ib)
    ib_connected = bool(ib_status.get("connected", False))

    # תיק – ממולא מה-snapshot (נעדכן עוד רגע ב-render_dashboard_home_v2)
    gross_pnl = st.session_state.get("portfolio_gross_pnl")
    nav = st.session_state.get("portfolio_nav")
    pos_count = st.session_state.get("portfolio_positions_count")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Macro Score",
            f"{macro_score:.1f}" if isinstance(macro_score, (int, float)) else "N/A",
        )
        if isinstance(macro_sens, (int, float, float)):
            st.caption(f"Sensitivity: {macro_sens:.2f}")
        else:
            st.caption("Sensitivity: N/A")

    with col2:
        st.metric(
            "Risk Exposure (×)",
            f"{risk_exposure:.2f}" if isinstance(risk_exposure, (int, float)) else "N/A",
        )
        if isinstance(risk_score, (int, float)):
            st.caption(f"Risk Score: {risk_score:.1f}")
        if risk_label:
            st.caption(f"Profile: {risk_label}")

    with col3:
        st.metric(
            "IBKR",
            "✅ Connected" if ib_connected else "❌ Disconnected",
        )
        host = ib_status.get("host") or "N/A"
        port = ib_status.get("port") or "N/A"
        st.caption(f"{host}:{port}")

    with col4:
        st.metric(
            "PnL Today",
            _format_num(gross_pnl, digits=0) if gross_pnl is not None else "N/A",
        )
        st.caption(
            f"NAV: {_format_num(nav, digits=0)} | Positions: {int(pos_count)}"
            if nav is not None and pos_count is not None
            else "NAV / Positions: N/A"
        )

# ============================================================
# UI Panels — Risk: Budget, Drawdown, Tail, Beta/TE
# ============================================================


def render_risk_panels(snapshot: DashboardSnapshot) -> None:
    """
    Panels לסיכון:
    - Risk Budget vs Usage (Vol / VaR / ES)
    - Drawdown state (1Y / ITD מול soft/hard limits)
    - Tail risk summary & stress scenarios
    - Beta / Tracking Error panel
    """
    risk_budget = build_risk_budget_summary(snapshot)
    dd_state = build_drawdown_state(snapshot)
    tail = build_tail_risk_summary(snapshot)
    beta_te = build_beta_te_summary(snapshot)

    ctx = snapshot.ctx

    col1, col2 = st.columns(2)

    # ---- צד שמאל: Risk Budget & Drawdown ----
    with col1:
        st.markdown("#### ⚖️ Risk Budget")

        vol = risk_budget["vol"]
        target_vol = risk_budget["target_vol"]
        using = risk_budget["usage_vs_target"]

        vol_color = _traffic_color_from_level(
            vol,
            warn=RISK_VOL_WARNING,
            critical=RISK_VOL_CRITICAL,
        )
        vol_emoji = _emoji_from_color(vol_color)

        st.write(
            f"{vol_emoji} **Vol (annual)**: {_format_pct(vol)}  "
            f"(target: {_format_pct(target_vol)}, "
            f"usage: {_format_pct(using) if using is not None else '—'})"
        )

        st.write(
            f"**VaR 95%:** {_format_pct(risk_budget['var_95'])} · "
            f"**ES 95%:** {_format_pct(risk_budget['es_95'])}"
        )

        st.markdown("#### 📉 Drawdown")
        dd_1y = dd_state["dd_1y"]
        dd_itd = dd_state["dd_itd"]
        soft = dd_state["soft_limit"]
        hard = dd_state["hard_limit"]

        dd_color = _traffic_color_from_level(
            abs(dd_1y) if dd_1y is not None else None,
            warn=soft,
            critical=hard,
        )
        dd_emoji = _emoji_from_color(dd_color)

        st.write(
            f"{dd_emoji} **DD 1Y**: {_format_pct(dd_1y)}  "
            f"(soft limit: {_format_pct(soft)}, hard limit: {_format_pct(hard)})"
        )
        if dd_itd is not None:
            st.write(f"**DD ITD**: {_format_pct(dd_itd)}")

        if snapshot.risk.limits_breached:
            st.error("⛔️ One or more risk limits breached:")
            for txt in snapshot.risk.breached_limits:
                st.write(f"- {txt}")
        else:
            st.success("✅ אין כרגע חריגה מגבלות סיכון.")

    # ---- צד ימין: Tail risk + Beta/TE & narrative ----
    with col2:
        st.markdown("#### 🐾 Tail Risk & Scenarios")

        tail_idx = tail["tail_index"]
        scenarios = tail["scenarios"] or {}

        if tail_idx is not None:
            tail_color = _traffic_color_from_level(
                tail_idx,
                warn=1.0,
                critical=2.0,
            )
            tail_emoji = _emoji_from_color(tail_color)
            st.write(f"{tail_emoji} **Tail risk index**: {tail_idx:.2f}")
        else:
            st.caption("Tail index not computed (tail_risk_index is None).")

        if scenarios:
            rows = [{"Scenario": k, "P&L Impact": v} for k, v in scenarios.items()]
            df_sc = pd.DataFrame(rows)
            df_sc["abs_impact"] = df_sc["P&L Impact"].abs()
            df_sc = df_sc.sort_values("abs_impact", ascending=False).drop(
                columns=["abs_impact"]
            )
            st.dataframe(df_sc, width="stretch", height=180)
        else:
            st.caption(
                "No stress scenarios provided in PortfolioRiskMetrics.stress_test_scenarios."
            )

        st.markdown("#### 📐 Beta & Tracking Error")
        st.write(
            f"**Beta vs benchmark**: "
            f"{beta_te['beta']:.2f if beta_te['beta'] is not None else '—'} · "
            f"**Corr vs benchmark**: "
            f"{_format_pct(beta_te['corr']) if beta_te['corr'] is not None else '—'}"
        )
        st.write(
            f"**Tracking error**: "
            f"{_format_pct(beta_te['tracking_error']) if beta_te['tracking_error'] is not None else '—'}"
        )

        # Narrative קצר על מצב הסיכון
        risk_mode = (ctx.risk_profile or "balanced").capitalize()
        vol = risk_budget["vol"]
        dd_1y = dd_state["dd_1y"]
        narrative = (
            f"Risk mode: **{risk_mode}**. "
            f"Volatility: {_format_pct(vol)}, "
            f"1Y drawdown: {_format_pct(dd_1y)}."
        )
        if snapshot.risk.limits_breached:
            narrative += " ⚠️ Limits breached — consider reducing gross exposure / leverage."
        elif ctx.stress_mode:
            narrative += " ⚠️ Stress mode is ON — portfolio should be in defensive posture."
        st.caption(narrative)

# ============================================================
# UI Panels — Signals: Funnel, Filters, Ladder, Conflicts, Aging, Watchlist, Heatmap
# ============================================================


def _filter_signals_core_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    פילטרים לסיגנלים לפי UI:
    - Strategy
    - Regime
    - Timeframe (TF)
    - מינימום |Z|
    - מינימום Quality
    """
    if df.empty:
        return df

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        strategies = sorted(df["Strategy"].dropna().unique().tolist())
        strat_sel = st.multiselect(
            "Strategies",
            options=strategies,
            default=strategies[: min(3, len(strategies))] if strategies else [],
        )
    with col2:
        regimes = sorted(df["Regime"].dropna().unique().tolist())
        regime_sel = st.multiselect("Regimes", options=regimes, default=regimes)
    with col3:
        tfs = sorted(df["TF"].dropna().unique().tolist())
        tf_sel = st.multiselect("TFs", options=tfs, default=tfs)
    with col4:
        min_absz = st.number_input("Min |Z|", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    with col5:
        min_quality = st.number_input(
            "Min Quality", min_value=0.0, max_value=10.0, value=0.0, step=0.1
        )

    df_f = df.copy()

    if strat_sel:
        df_f = df_f[df_f["Strategy"].isin(strat_sel)]
    if regime_sel:
        df_f = df_f[df_f["Regime"].isin(regime_sel)]
    if tf_sel:
        df_f = df_f[df_f["TF"].isin(tf_sel)]

    if "AbsZ" in df_f.columns:
        df_f = df_f[df_f["AbsZ"].fillna(0.0) >= min_absz]

    if "Quality" in df_f.columns:
        df_f = df_f[df_f["Quality"].fillna(0.0) >= min_quality]

    return df_f


def render_signals_panels(snapshot: DashboardSnapshot) -> None:
    """
    אזור סיגנלים/הזדמנויות מלא:

    • Signal Funnel Overview (Universe → Filtered → Signals → Deployed)
    • Filters: Strategy / Regime / TF / |Z| / Quality
    • Signal Quality Ladder (Top High-Conviction)
    • Conflict Detector (סיגנלים סותרים)
    • Signal Aging Distribution (Fresh / Today / Recent / Stale) + bar chart
    • Watchlist (ידני + tags ב-Signals)
    • Pairs Heatmap (AbsZ vs HalfLife vs Corr/Strategy)
    • Drill-down לטאב comparison_matrices דרך session_state (אם תרצה)
    """
    st.markdown("### 🔍 Signals & Opportunities")

    # ---- סיכום Funnel + Aging ----
    funnel = build_signal_funnel_metrics(snapshot)
    aging = build_signal_aging_distribution(snapshot)
    sigs = snapshot.signals

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Universe size", funnel["universe_size"])
    with col2:
        st.metric("Filtered candidates", funnel["filtered_candidates"])
    with col3:
        st.metric("Signals (current)", funnel["signal_count"])
    with col4:
        st.metric("Backtested", funnel["backtested_candidates"])
    with col5:
        st.metric("Deployed to live", funnel["deployed_signals"])

    col_a1, col_a2, col_a3, col_a4, col_a5 = st.columns(5)
    with col_a1:
        st.metric("Fresh (≤2h)", aging["Fresh"])
    with col_a2:
        st.metric("Today", aging["Today"])
    with col_a3:
        st.metric("Recent (≤3d)", aging["Recent"])
    with col_a4:
        st.metric("Stale (>3d)", aging["Stale"])
    with col_a5:
        st.metric("Total signals", sigs.n_total or len(sigs.items))

    # ---- פילטרים כלליים (Ladder/Conflicts/Heatmap ישתמשו בהם) ----
    df_core_raw = build_signals_core_df(snapshot)
    df_core = _filter_signals_core_df(df_core_raw)

    # ---- גרף aging קטן ----
    aging_df = pd.DataFrame(
        {
            "Bucket": ["Fresh", "Today", "Recent", "Stale"],
            "Count": [aging["Fresh"], aging["Today"], aging["Recent"], aging["Stale"]],
        }
    )
    if not aging_df.empty:
        fig_age = go.Figure(
            data=[go.Bar(x=aging_df["Bucket"], y=aging_df["Count"])]
        )
        fig_age.update_layout(
            title="Signal aging distribution",
            margin=dict(l=10, r=10, t=30, b=10),
            height=220,
            xaxis_title="Age bucket",
            yaxis_title="# signals",
        )
        st.plotly_chart(fig_age, width="stretch")

    # ---- טאבונים פנימיים ----
    tab_ladder, tab_conflicts, tab_watchlist, tab_heatmap = st.tabs(
        ["Ladder", "Conflicts", "Watchlist", "Pairs Heatmap"]
    )

    with tab_ladder:
        render_signal_ladder_panel_filtered(snapshot, df_core)

    with tab_conflicts:
        render_signal_conflicts_panel_filtered(snapshot, df_core)

    with tab_watchlist:
        render_watchlist_panel(snapshot)

    with tab_heatmap:
        render_pairs_heatmap_panel_filtered(snapshot, df_core)

    # ---- Drill-down לטאב comparison_matrices (אופציונלי) ----
    st.markdown("---")
    col_drill1, col_drill2 = st.columns([3, 1])
    with col_drill1:
        st.caption(
            "לניתוח עומק של מטריצות קורלציה / קו-אינטגרציה / Tail dependencies "
            "אפשר לעבור לטאב **Comparison Matrices**."
        )
    with col_drill2:
        if st.button("🔬 פתח Tab Comparison Matrices", width="stretch"):
            # כאן אנחנו רק מסמנים intent ב-session_state.
            # בקובץ dashboard.py אפשר לקרוא לזה ולנווט לטאב הרלוונטי.
            st.session_state["nav_target"] = "comparison_matrices"


def render_signal_ladder_panel_filtered(
    snapshot: DashboardSnapshot,
    df_core_filtered: pd.DataFrame,
) -> None:
    """
    Signal Quality Ladder — לגמרי אחרי פילטרים.
    """
    ctx = snapshot.ctx
    top_n = int(
        getattr(ctx, "top_signals_limit", DEFAULT_TOP_SIGNALS) or DEFAULT_TOP_SIGNALS
    )

    st.markdown("#### 🧱 Signal Quality Ladder (Filtered, High-Conviction)")

    if df_core_filtered.empty:
        st.info("לא נשארו סיגנלים אחרי הפילטרים. נסה לרכך את הסינון.")
        return

    df = df_core_filtered.copy()
    sort_cols = []
    if "Quality" in df.columns:
        sort_cols.append(("Quality", False))
    if "AbsZ" in df.columns:
        sort_cols.append(("AbsZ", False))

    if sort_cols:
        by = [c for c, _ in sort_cols]
        ascending = [asc for _, asc in sort_cols]
        df = df.sort_values(by=by, ascending=ascending)

    st.dataframe(df.head(top_n), width="stretch", height=340)


def render_signal_conflicts_panel_filtered(
    snapshot: DashboardSnapshot,
    df_core_filtered: pd.DataFrame,
) -> None:
    """
    Conflict Detector — מבוסס על df_core_filtered (כולל פילטרים).
    """
    st.markdown("#### ⚔️ Conflicting signals (LONG vs SHORT on same pair)")

    if df_core_filtered.empty:
        st.info("אין סיגנלים לאחר פילטרים, ולכן אין קונפליקטים להציג.")
        return

    # נשתמש בלוגיקה של build_signal_conflicts_df אך באותו DF
    groups = df_core_filtered.groupby("Pair")
    rows: List[Dict[str, Any]] = []

    for pair, g in groups:
        dirs = set(g["Direction"].dropna().str.upper())
        if len(dirs) > 1 and {"LONG", "SHORT"} <= dirs:
            rows.append(
                {
                    "Pair": pair,
                    "Directions": ", ".join(sorted(dirs)),
                    "Count": len(g),
                    "MaxAbsZ": g["AbsZ"].max(),
                    "MaxQuality": g["Quality"].max(),
                }
            )

    if not rows:
        st.success("אין כרגע קונפליקטים ברמת כיוון על אותו Pair (לאחר הפילטרים).")
        return

    df_conf = pd.DataFrame(rows)
    df_conf = df_conf.sort_values("MaxQuality", ascending=False)
    st.dataframe(df_conf, width="stretch", height=260)
    st.caption(
        "קונפליקט מוגדר כאן כמצב שבו יש על אותו Pair גם סיגנל LONG וגם SHORT "
        "ממקורות/מודלים שונים."
    )


def render_watchlist_panel(snapshot: DashboardSnapshot) -> None:
    """
    Watchlist — שילוב בין:
      • ctx.extra['watchlist']
      • signals עם tag 'watch'
    """
    st.markdown("#### ⭐ Watchlist")

    df_watch = build_watchlist_df(snapshot)
    if df_watch.empty:
        st.info("אין עדיין פריטים ב-Watchlist (לא extra['watchlist'] ולא tags על signals).")
        return

    st.dataframe(df_watch, width="stretch", height=260)


def render_pairs_heatmap_panel_filtered(
    snapshot: DashboardSnapshot,
    df_core_filtered: pd.DataFrame,
) -> None:
    """
    Pairs Heatmap מסונן:

    • scatter של HalfLife vs AbsZ
    • צביעה לפי Corr או Strategy
    • tooltip עם Pair + Direction + Quality
    """
    st.markdown("#### 🗺 Pairs Heatmap (filtered) — |Z| vs Half-Life vs Corr/Strategy")

    if df_core_filtered.empty:
        st.info("אין מספיק נתוני signals בשביל לבנות Heatmap אחרי הפילטרים.")
        return

    df = df_core_filtered.copy()
    df["AbsZ"] = pd.to_numeric(df["AbsZ"], errors="coerce")
    df["HalfLife"] = pd.to_numeric(df["HalfLife"], errors="coerce")

    df = df.dropna(subset=["AbsZ", "HalfLife"])
    if df.empty:
        st.info("אין ערכים תקינים ל-AbsZ/HalfLife אחרי ניקוי ערכים חסרים.")
        return

    color_mode = st.selectbox("Color by", ["Corr", "Strategy"], index=0)

    if color_mode == "Corr":
        df["Color"] = df["Corr"]
        color_label = "Corr"
    else:
        df["Color"] = df["Strategy"]
        color_label = "Strategy"

    hover_text = (
        "Pair: " + df["Pair"].astype(str)
        + "<br>Dir: " + df["Direction"].astype(str)
        + "<br>Quality: " + df["Quality"].astype(str)
        + "<br>AbsZ: " + df["AbsZ"].round(2).astype(str)
        + "<br>HL: " + df["HalfLife"].round(1).astype(str)
    )

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["HalfLife"],
                y=df["AbsZ"],
                mode="markers",
                text=hover_text,
                hoverinfo="text",
                marker=dict(
                    size=8,
                    color=df["Color"],
                    showscale=(color_mode == "Corr"),
                    colorbar=dict(title=color_label),
                ),
            )
        ]
    )
    fig.update_layout(
        title="Pairs Heatmap — |Z| vs Half-Life",
        xaxis_title="Half-Life (days)",
        yaxis_title="|Z|",
        margin=dict(l=10, r=10, t=30, b=10),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# UI Panels — Macro: Regime, Cross-Asset, Vol Grid
# ============================================================


def render_macro_panels(snapshot: DashboardSnapshot) -> None:
    """
    אזור מאקרו / שוק:

    • Macro Regime Banner (growth/inflation וכו')
    • Cross-Asset Snapshot (Equities / Rates / Credit / FX / Commodities)
    • Volatility Grid (Implied/Realized Z-scores)
    """
    st.markdown("### 🌐 Macro & Cross-Asset Snapshot")

    macro = build_macro_regime_summary(snapshot)
    cross_df = build_cross_asset_snapshot_df(snapshot)
    vol_df = build_volatility_grid_df(snapshot)

    # ---- Macro Regime Banner ----
    col1, col2, col3 = st.columns(3)

    macro_regime = macro.get("macro_regime") or "Unknown"
    vol_regime = macro.get("vol_regime") or "Unknown"
    conf = macro.get("confidence") or 0.0

    with col1:
        st.markdown(f"**Macro regime:** `{macro_regime}`")
        st.markdown(f"**Vol regime:** `{vol_regime}`")
    with col2:
        st.markdown(f"**Regime confidence:** {_format_pct(conf)}")
        st.markdown(
            f"**Risk mode:** `{(snapshot.ctx.risk_profile or 'balanced').capitalize()}`"
        )
    with col3:
        st.markdown(f"**Env:** `{snapshot.ctx.env}` · **Profile:** `{snapshot.ctx.profile}`")
        st.markdown(f"**Benchmark:** `{snapshot.ctx.benchmark}`")

    # ---- Cross-Asset table + bar ----
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### 🧱 Cross-Asset Returns")
        if cross_df.empty:
            st.info("אין כרגע נתוני Cross-Asset מפורטים.")
        else:
            # אפשר לאפשר פילטר לפי Group
            groups = sorted(cross_df["Group"].dropna().unique().tolist())
            group_sel = st.multiselect(
                "Asset groups", options=groups, default=groups if groups else []
            )
            df_show = cross_df.copy()
            if group_sel:
                df_show = df_show[df_show["Group"].isin(group_sel)]
            st.dataframe(df_show, width="stretch", height=260)

    with col_right:
        st.markdown("#### 📊 Top moves (1D)")
        if not cross_df.empty and "Ret_1D" in cross_df.columns:
            df_plot = cross_df.copy()
            df_plot["abs_ret"] = df_plot["Ret_1D"].abs()
            df_plot = df_plot.sort_values("abs_ret", ascending=False).head(10)
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=df_plot["Asset"],
                        y=df_plot["Ret_1D"],
                        text=df_plot["Group"],
                    )
                ]
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                height=260,
                xaxis_title="Asset",
                yaxis_title="Return 1D",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("אין נתוני Ret_1D לקרוס-אססט כרגע.")

    # ---- Volatility Grid ----
    st.markdown("#### 🌪 Volatility Grid (factor z-scores)")
    if vol_df.empty:
        st.info("אין factor_zscores במבנה של MarketSnapshot (vol grid).")
    else:
        # הדגשת פקטורים 'חמים' (|Z|>2)
        vol_df_show = vol_df.copy()
        vol_df_show["Hot"] = vol_df_show["Z_Vol"].apply(
            lambda z: "🔥" if z is not None and abs(z) >= 2.0 else ""
        )
        st.dataframe(vol_df_show, width="stretch", height=220)

# ============================================================
# UI Panels — System Health (Broker / Data / SQL / Agents / Resources)
# ============================================================


def render_system_panels(snapshot: DashboardSnapshot) -> None:
    """
    System Health אזור:
    - Broker / Data / SQL status
    - Latency breakdown (Market data / Broker / SQL / Risk Engine)
    - Agents status
    - Resource utilization (CPU / RAM)
    - Errors feed
    """
    st.markdown("### 🛠 System Health & Infrastructure")

    system = snapshot.system
    latency = build_latency_breakdown(snapshot)
    resources = build_resource_utilization_dict(system)

    # ---- Row 1: High-level metrics ----
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Broker connected", "Yes" if system.broker_connected else "No")
        st.metric("Data fresh", "Yes" if system.data_fresh else "No")

    with col2:
        st.metric("Data latency (ms)", f"{system.data_latency_ms:.0f}")
        if system.last_price_update:
            st.caption(f"Last price update: {system.last_price_update}")
        else:
            st.caption("Last price update: N/A")

    with col3:
        st.metric("SQL OK", "Yes" if system.sql_ok else "No")
        if not system.sql_ok and system.sql_last_error:
            st.caption(f"Last SQL error: {system.sql_last_error[:80]}…")

    with col4:
        cpu = resources.get("cpu_pct")
        ram = resources.get("ram_pct")
        st.metric("CPU load %", f"{cpu:.0f}%" if cpu is not None else "N/A")
        st.metric("RAM used %", f"{ram:.0f}%" if ram is not None else "N/A")

    # ---- Row 2: Latency breakdown + Agents ----
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("#### ⏱ Latency breakdown (ms)")
        # Market / Broker / SQL / Risk Engine
        lb_rows = []
        for label in ["market_data_ms", "broker_ms", "sql_ms", "risk_engine_ms", "other_ms"]:
            val = latency.get(label)
            if val is not None:
                lb_rows.append({"Component": label.replace("_ms", ""), "Latency_ms": float(val)})

        if not lb_rows:
            st.caption("No detailed latency metrics in ctx.extra['latency_breakdown'].")
        else:
            df_lb = pd.DataFrame(lb_rows)
            st.dataframe(df_lb, width="stretch", height=220)

            fig = go.Figure(
                data=[go.Bar(x=df_lb["Component"], y=df_lb["Latency_ms"])]
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                height=220,
                xaxis_title="Component",
                yaxis_title="Latency (ms)",
                title="Latency by component",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### 🤖 Agents")
        df_agents = build_agents_status_df(system)
        if df_agents.empty:
            st.caption("No agents status recorded (system.agents_status is empty).")
        else:
            st.dataframe(df_agents, width="stretch", height=220)

        if system.running_strategies:
            st.markdown("**Running strategies:**")
            st.write(", ".join(map(str, system.running_strategies)))

    # ---- Row 3: Errors feed ----
    st.markdown("#### ❗ Recent errors")
    df_err = build_error_summary_df(system)
    if df_err.empty:
        st.success("No recent errors reported by the system.")
    else:
        st.dataframe(df_err, width="stretch", height=220)


# ============================================================
# Context Controls (DashboardContext) — משודרג
# ============================================================


def _apply_date_mode(mode: str, today: Optional[date] = None) -> Tuple[date, date]:
    """
    לוקח mode (today / 1w / 1m / 3m / 6m / 1y / ytd / mtd) ומחזיר start/end.
    """
    if today is None:
        today = date.today()

    mode = (mode or "today").lower()

    if mode == "ytd":
        start = date(today.year, 1, 1)
        return start, today

    if mode == "mtd":
        start = date(today.year, today.month, 1)
        return start, today

    if mode in ("1w", "5d"):
        start = today - pd.Timedelta(days=5)
        return start, today  # type: ignore[return-value]

    if mode == "1m":
        start = today - pd.Timedelta(days=30)
        return start, today  # type: ignore[return-value]

    if mode == "3m":
        start = today - pd.Timedelta(days=90)
        return start, today  # type: ignore[return-value]

    if mode == "6m":
        start = today - pd.Timedelta(days=180)
        return start, today  # type: ignore[return-value]

    if mode == "1y":
        start = today - pd.Timedelta(days=365)
        return start, today  # type: ignore[return-value]

    # ברירת מחדל: היום
    return today, today


def _clone_ctx(ctx: DashboardContext) -> DashboardContext:
    """
    יוצר עותק שטחי של DashboardContext (dataclass) כדי לא להרוס את המקור.
    """
    data = {k: getattr(ctx, k) for k in ctx.__dataclass_fields__.keys()}
    return DashboardContext(**data)


def _render_context_controls(base_ctx: DashboardContext) -> DashboardContext:
    """
    מעלה Controls בראש הטאב ומחזיר DashboardContext מעודכן לפי בחירת המשתמש.

    שולט על:
    - env/profile (dev/live/research וכו')
    - date range mode
    - ui_mode
    - benchmark / portfolio_id
    - top_signals_limit / top_positions_limit
    """
    ctx = _clone_ctx(base_ctx)

    st.markdown("#### 🎚 הגדרות דשבורד (Context)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        env = st.selectbox(
            "Environment",
            DEFAULT_ENV_OPTIONS,
            index=DEFAULT_ENV_OPTIONS.index(ctx.env)
            if ctx.env in DEFAULT_ENV_OPTIONS
            else 0,
        )
    with col2:
        profile = st.selectbox(
            "Profile",
            DEFAULT_PROFILE_OPTIONS,
            index=DEFAULT_PROFILE_OPTIONS.index(ctx.profile)
            if ctx.profile in DEFAULT_PROFILE_OPTIONS
            else 0,
        )
    with col3:
        date_mode = st.selectbox(
            "טווח זמן",
            ["today", "1w", "1m", "3m", "6m", "1y", "ytd", "mtd"],
            index=["today", "1w", "1m", "3m", "6m", "1y", "ytd", "mtd"].index(
                DEFAULT_DATE_MODE
            ),
        )
    with col4:
        ui_mode = st.selectbox(
            "מצב תצוגה",
            DEFAULT_UI_MODES,
            index=DEFAULT_UI_MODES.index(ctx.ui_mode)
            if ctx.ui_mode in DEFAULT_UI_MODES
            else 0,
        )

    ctx.env = env
    ctx.profile = profile
    ctx.ui_mode = ui_mode

    start, end = _apply_date_mode(date_mode)
    ctx.start_date = start
    ctx.end_date = end

    # שורה שנייה: Benchmark / Portfolio / Top N
    col1b, col2b, col3b, col4b = st.columns(4)
    with col1b:
        ctx.benchmark = st.text_input("Benchmark", value=ctx.benchmark or "SPY")
    with col2b:
        ctx.portfolio_id = st.text_input(
            "Portfolio ID", value=ctx.portfolio_id or "default"
        )
    with col3b:
        ctx.top_positions_limit = int(
            st.number_input(
                "כמה פוזיציות להציג?",
                min_value=5,
                max_value=100,
                value=int(ctx.top_positions_limit or DEFAULT_TOP_POSITIONS),
                step=5,
            )
        )
    with col4b:
        ctx.top_signals_limit = int(
            st.number_input(
                "כמה סיגנלים להציג?",
                min_value=5,
                max_value=200,
                value=int(ctx.top_signals_limit or DEFAULT_TOP_SIGNALS),
                step=5,
            )
        )

    return ctx


# ============================================================
# Public entry point — render_dashboard_home_v2
# ============================================================


def render_dashboard_home_v2(
    app_ctx: AppContext,
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    🏠 Dashboard Home v2 – Tab 0 ברמת קרן גידור.

    מבנה (HF-grade, גרסה מורחבת):
    -----------------------------
    1. Runtime + HomeContext (Health, Overview, Alerts):
       - health_light (ready / issues / warnings)
       - overview_metrics (system / risk / macro / agents / desktop / experiments)
       - alerts (warning/error)
       - saved_views (light)
       - agent_actions_tail (what agents עשו לאחרונה)
       - timeseries (equity / vol / risk) — אופציונלי
       - action_suggestions / data_quality / experiments_tail — אופציונלי

    2. פס עליון:
       - System health + meta (env/profile/run_id/app/version)
       - Context strip: env/profile/date-range/tab-count/pinned-view
       - כרטיסי Overview עיקריים (System / Risk / Macro / Agents / Desktop / Experiments)

    3. פאנלים צדדיים:
       - Alerts אחרונים
       - Saved views / layouts
       - Agent actions history (tail)
       - Action center (מה כדאי לעשות עכשיו)

    4. Quick filters & mini charts:
       - Risk focus / Macro view filters (משתלבים עם Tabs אחרים דרך session_state)
       - Mini trend charts: Equity / Volatility אם קיימים ב-timeseries

    5. החלק הישן (DashboardService + Snapshots):
       - Runtime strip
       - Context controls (env/profile/dates/ui_mode/benchmark/portfolio/topN)
       - Snapshot diff
       - Performance & equity
       - Portfolio & Risk & Macro & System panels
    """

    # ========= חלק 1 – Runtime + Home Context (החלק החדש והחשוב) =========

    # 1) Runtime ברמת קרן (env/profile/capabilities/tabs/health...)
    runtime: DashboardRuntime = ensure_dashboard_runtime(app_ctx)

    # 2) Home context (עם Cache): overview_metrics + alerts + health + views + agent actions
    home_ctx: Dict[str, Any] = (
        get_dashboard_home_context_from_session()
        or update_dashboard_home_context_in_session(runtime)
    )

    st.markdown("### 🏠 Dashboard Home – Monitoring view")

    # -------- 1.A – Health light + App meta --------
    hl: Dict[str, Any] = home_ctx.get("health_light", {}) or {}
    ready = bool(hl.get("ready", True))
    has_critical = bool(hl.get("has_critical_issues", False))
    has_warnings = bool(hl.get("has_warnings", False))

    # נגזור “severity” לוגי (לשימוש מאוחד ב-UI)
    if has_critical:
        health_severity = "critical"
    elif has_warnings:
        health_severity = "warning"
    else:
        health_severity = "ok"

    col_hl, col_meta = st.columns([1.5, 1.5])

    with col_hl:
        icon = "🚨" if has_critical else ("⚠️" if has_warnings else "✅")
        st.markdown(
            f"**System health:** {icon} `{health_severity}`  •  "
            f"ready=`{ready}`"
        )
        if hl.get("issues") or hl.get("warnings"):
            with st.expander("Health details (issues & warnings)", expanded=False):
                if hl.get("issues"):
                    st.markdown("**Issues (critical):**")
                    for msg in hl.get("issues", []):
                        st.write(f"- {msg}")
                if hl.get("warnings"):
                    st.markdown("**Warnings:**")
                    for msg in hl.get("warnings", []):
                        st.write(f"- {msg}")
        else:
            st.caption("No critical issues reported by the Health engine.")

    with col_meta:
        api_meta = home_ctx.get("api_meta", {}) or {}
        st.caption(
            f"App: `{api_meta.get('app_name', 'Pairs Trading Dashboard')}` "
            f"(v{api_meta.get('version', '')})  •  "
            f"Host: `{api_meta.get('host', '')}`  •  "
            f"User: `{api_meta.get('user', '')}`"
        )
        st.caption(
            f"Run ID: `{runtime.run_id}`  •  "
            f"Env/Profile: `{runtime.env}` / `{runtime.profile}`"
        )
        if api_meta.get("ts_utc"):
            st.caption(f"Snapshot ts (UTC): `{api_meta.get('ts_utc')}`")

    # ===== רעיון חדש #1 – Context strip עליון (env/profile/dates/tabs/pinned-view) =====
    st.markdown("---")
    ctx_from_session = st.session_state.get("ctx", {})
    ctx_env = ctx_from_session.get("env", runtime.env)
    ctx_profile = ctx_from_session.get("profile", runtime.profile)
    ctx_start = ctx_from_session.get("start_date", "")
    ctx_end = ctx_from_session.get("end_date", "")
    pinned_view_name = home_ctx.get("pinned_view_name")
    tabs_count = len(getattr(runtime, "tabs", []) or [])

    col_ctx1, col_ctx2, col_ctx3 = st.columns([2, 2, 2])
    with col_ctx1:
        st.caption(
            f"📌 Context: env=`{ctx_env}` • profile=`{ctx_profile}`"
        )
    with col_ctx2:
        if ctx_start and ctx_end:
            st.caption(f"📆 Date range: `{ctx_start}` → `{ctx_end}`")
        else:
            st.caption("📆 Date range: not set (using defaults)")
    with col_ctx3:
        extra = f" • pinned=`{pinned_view_name}`" if pinned_view_name else ""
        st.caption(f"🧭 Tabs active: `{tabs_count}`{extra}")

    # -------- 1.B – כרטיסי Overview עיקריים (primary metrics) --------
    overview_metrics: List[Dict[str, Any]] = home_ctx.get("overview_metrics", []) or []

    primary_keys: List[str] = home_ctx.get("primary_metrics", []) or [
        "system_health",
        "risk_status",
        "macro_regime",
        "agents_status",
        "desktop_link",
        "experiments",
    ]

    icon_map: Dict[str, str] = {
        "system_health": "🩺",
        "risk_status": "⚠️",
        "macro_regime": "🌐",
        "agents_status": "🤖",
        "desktop_link": "🖥",
        "experiments": "🧪",
    }

    metrics_by_key: Dict[str, Dict[str, Any]] = {
        m.get("key"): m for m in overview_metrics if m.get("key")
    }

    primary_cards: List[Dict[str, Any]] = []
    for k in primary_keys:
        m = metrics_by_key.get(k)
        if not m:
            continue
        m = dict(m)  # לא נוגעים במקור
        m.setdefault("icon", icon_map.get(k, ""))
        primary_cards.append(m)

    if primary_cards:
        st.markdown("#### 🔎 Overview – מצב קרן במבט מהיר")
        # נפרוס אותם על עד 2 שורות אם יש המון
        row_size = min(4, len(primary_cards))
        for i in range(0, len(primary_cards), row_size):
            row = primary_cards[i : i + row_size]
            cols = st.columns(len(row))
            for col, metric in zip(cols, row):
                with col:
                    icon = metric.get("icon") or ""
                    label = metric.get("label") or metric.get("key")
                    value = metric.get("value")
                    desc = metric.get("description", "")
                    level = str(metric.get("level", "info")).lower()

                    if level == "error":
                        prefix = "🚨"
                    elif level == "warning":
                        prefix = "⚠️"
                    elif level == "success":
                        prefix = "✅"
                    else:
                        prefix = "ℹ️"

                    st.metric(
                        label=f"{icon} {label}",
                        value=value,
                        help=desc,
                    )
                    if desc:
                        st.caption(f"{prefix} {desc}")

        with st.expander("All overview metrics (advanced)", expanded=False):
            st.json(overview_metrics)
    else:
        st.caption("No overview metrics available yet – check SqlStore / Health engine.")

    # ===== רעיון חדש #2 – Quick filters ל-Home (Risk & Macro focus) =====
    st.markdown("---")
    col_qf1, col_qf2, col_qf3 = st.columns([1.5, 1.5, 2])

    with col_qf1:
        risk_focus_default = st.session_state.get("home_risk_focus", "balanced")
        risk_focus = st.radio(
            "Risk focus",
            options=["balanced", "defense", "offense"],
            index=["balanced", "defense", "offense"].index(risk_focus_default),
            horizontal=True,
            key="home_risk_focus_radio",
        )
        st.session_state["home_risk_focus"] = risk_focus
        st.caption("משפיע על הדגשה של פאנלים / Alerts בטאבים אחרים.")

    with col_qf2:
        macro_focus_default = st.session_state.get("home_macro_focus", "index")
        macro_focus = st.radio(
            "Macro view",
            options=["index", "rates", "vol"],
            index=["index", "rates", "vol"].index(macro_focus_default),
            horizontal=True,
            key="home_macro_focus_radio",
        )
        st.session_state["home_macro_focus"] = macro_focus
        st.caption("בחירת זווית מקרו מועדפת למעקב.")

    with col_qf3:
        st.caption("🎯 Current filters (home-level)")
        st.json(
            {
                "risk_focus": st.session_state.get("home_risk_focus", "balanced"),
                "macro_focus": st.session_state.get("home_macro_focus", "index"),
            }
        )

    # ===== רעיון חדש #3 – Mini trend charts (Equity / Vol) =====
    st.markdown("---")
    timeseries = home_ctx.get("timeseries", {}) or {}
    eq = timeseries.get("equity_curve")
    vol = timeseries.get("volatility") or timeseries.get("vix")

    col_ts1, col_ts2 = st.columns(2)
    with col_ts1:
        st.markdown("**📈 Equity trend (mini)**")
        if isinstance(eq, dict) and eq.get("values"):
            st.line_chart(eq.get("values"))
        else:
            st.caption("No equity timeseries in HomeContext yet.")

    with col_ts2:
        st.markdown("**🌪 Volatility / VIX trend (mini)**")
        if isinstance(vol, dict) and vol.get("values"):
            st.line_chart(vol.get("values"))
        else:
            st.caption("No volatility timeseries in HomeContext yet.")

    # -------- 1.C – Alerts + Saved Views + Agent Actions + Action Center --------
    st.markdown("---")
    alerts = home_ctx.get("alerts", []) or []
    saved_views_light = home_ctx.get("saved_views", []) or []
    agent_actions_tail = home_ctx.get("agent_actions_tail", []) or []
    action_suggestions = home_ctx.get("action_suggestions", []) or []

    col_alerts, col_views, col_actions, col_center = st.columns([1.4, 1.2, 1.4, 1.4])

    # Alerts
    with col_alerts:
        st.markdown("**🚨 Recent alerts (warning/error)**")
        if alerts:
            for a in alerts[:5]:
                lvl = str(a.get("level", "")).lower()
                icon = "⚠️" if lvl == "warning" else "🚨"
                st.write(
                    f"{icon} `{a.get('ts_utc')}` • "
                    f"**{a.get('source', 'unknown')}** – {a.get('message', '')}"
                )
        else:
            st.caption("No recent dashboard alerts (warning/error).")

    # Saved views
    with col_views:
        st.markdown("**📌 Saved views**")
        if saved_views_light:
            for v in saved_views_light[:6]:
                name = v.get("name", "view")
                env = v.get("env", "")
                prof = v.get("profile", "")
                last_tab = v.get("last_tab_key", "")
                st.write(
                    f"- `{name}`  •  env=`{env}`  •  profile=`{prof}`  •  last_tab=`{last_tab}`"
                )
        else:
            st.caption("No saved views yet – create one from the Agents tab.")

    # Agent actions
    with col_actions:
        st.markdown("**🤖 Agent actions (tail)**")
        if agent_actions_tail:
            for act in agent_actions_tail[-6:]:
                st.write(
                    f"- `{act.get('ts_utc')}` • source=`{act.get('source')}` • "
                    f"action=`{act.get('action')}` • result=`{act.get('result')}`"
                )
        else:
            st.caption("No agent actions recorded in this session.")

    # ===== רעיון חדש #4 – Action center (מה כדאי לעשות עכשיו) =====
    with col_center:
        st.markdown("**🎯 Action center**")
        if action_suggestions:
            for s in action_suggestions[:6]:
                src = s.get("source", "engine")
                kind = s.get("kind", "generic")
                msg = s.get("message", "")
                priority = str(s.get("priority", "normal")).lower()
                icon = "🔥" if priority in ("high", "urgent") else "➡️"
                st.write(
                    f"{icon} `{src}` [{kind}] – {msg}"
                )
        else:
            st.caption("No concrete action suggestions yet.")

    # ===== רעיון חדש #5 – Data quality & Experiments panels =====
    st.markdown("---")
    data_quality = home_ctx.get("data_quality", []) or []
    experiments_tail = home_ctx.get("experiments_tail", []) or []

    col_dq, col_exp = st.columns(2)

    with col_dq:
        st.markdown("**📡 Data quality snapshot**")
        if data_quality:
            for dq in data_quality[:10]:
                src = dq.get("source", "feed")
                status = dq.get("status", "ok")
                msg = dq.get("message", "")
                icon = "✅"
                if status.lower() in ("degraded", "warning"):
                    icon = "⚠️"
                elif status.lower() in ("down", "error", "critical"):
                    icon = "🚨"
                st.write(f"{icon} `{src}` – {status} – {msg}")
        else:
            st.caption("No explicit data-quality issues reported.")

    with col_exp:
        st.markdown("**🧪 Experiments / optimizations (tail)**")
        if experiments_tail:
            for e in experiments_tail[:8]:
                name = e.get("name", "exp")
                status = e.get("status", "done")
                score = e.get("score", "")
                ts = e.get("ts_utc", "")
                st.write(
                    f"- `{name}` • status=`{status}` • score=`{score}` • ts=`{ts}`"
                )
        else:
            st.caption("No experiment/optimization records in HomeContext yet.")

    # לדוגמה: שני טורים – משמאל ביצועים, מימין Execution
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # כל הדברים שכבר יש לך בהום (KPIs, גרף, וכו')
        st.subheader("📈 System Overview")
        # ...

    with col_right:
        _render_home_execution_panel(app_ctx, feature_flags)

    # ===== רעיון חדש #6 – System debug / runtime meta (expander) =====
    with st.expander("🔧 Runtime & system debug (for quant engineer)", expanded=False):
        st.json(
            {
                "env": runtime.env,
                "profile": runtime.profile,
                "run_id": runtime.run_id,
                "tabs": getattr(runtime, "tabs", []),
                "capabilities": getattr(runtime, "capabilities", {}),
            }
        )

    st.markdown("---")

    # ========= חלק 2 – הפאנלים הקיימים שלך (DashboardService / Snapshots) =========

    # ===== רעיון חדש #7 – Runtime overview strip (אם קיים helper) =====
    try:
        _render_runtime_overview_strip_from_session()
        st.markdown("---")
    except Exception:
        # אם היא לא קיימת / קרסה – לא נעיף את כל הטאב בגלל זה.
        pass

    # 2.B – Bootstrap בסיסי: Service + Context דיפולט
    try:
        service, base_ctx = bootstrap_dashboard()
    except Exception as exc:
        st.error("Failed to bootstrap dashboard service.")
        st.caption(str(exc))
        return

    # 2.C – Controls (env/profile/date range וכו')
    try:
        ctx = _render_context_controls(base_ctx)
    except Exception as exc:
        st.error("Failed to render context controls.")
        st.caption(str(exc))
        return

    st.markdown("---")

    # 2.D – Snapshot קודם (ל-diff); חשוב לקרוא לפני build_dashboard_snapshot
    try:
        old_snapshot = service.get_last_snapshot()
    except Exception:
        old_snapshot = None

    # 2.E – Snapshot חדש (נשמר גם ב-Service וגם ב-SQL אם persistence פעיל)
    try:
        snapshot = service.build_dashboard_snapshot(ctx)
    except Exception as exc:
        st.error("Failed to build dashboard snapshot.")
        st.caption(str(exc))
        return

    # 2.F – Diff (Equity / VIX / Risk) בין הישן לחדש
    try:
        diff = service.diff_snapshots(old_snapshot, snapshot)
    except Exception:
        diff = None

    # להזין נתוני תיק בסיסיים ל-session_state עבור ה-Top Tiles
    try:
        perf_tiles = build_performance_tiles_data(snapshot)
        st.session_state["portfolio_gross_pnl"] = perf_tiles["today"]["pnl"]
        st.session_state["portfolio_nav"] = perf_tiles["today"]["nav"]
        st.session_state["portfolio_positions_count"] = len(snapshot.portfolio.positions)
    except Exception:
        # לא מפילים את הטאב אם משהו פה נשבר
        logger.debug("Failed to push portfolio metrics to session_state", exc_info=True)

    # שורת Tiles עליונה — Macro / Risk / IB / Portfolio
    _render_home_top_tiles()
    st.markdown("---")

    # ===== Investment Readiness Snapshot (חדשה) =====
    st.markdown("### ✅ Investment readiness snapshot")

    # מעדכנים את home_ctx ב-risk_focus מה-Quick Filter כדי שישפיע על משקולות הציון
    try:
        home_ctx["risk_focus"] = st.session_state.get("home_risk_focus", "balanced")
    except Exception:
        pass

    try:
        readiness = _compute_investment_readiness(snapshot, home_ctx)
    except Exception as exc:
        st.error("Failed to compute investment readiness.")
        st.caption(str(exc))
        readiness = {"overall_score": None, "dimensions": []}

    overall_score = readiness.get("overall_score")
    dims = readiness.get("dimensions", []) or []

    col_r1, col_r2 = st.columns([1.2, 2.0])

    with col_r1:
        if overall_score is None:
            st.metric(
                label="Overall readiness",
                value="N/A",
                help="No sufficient data yet to compute investment readiness.",
            )
        else:
            # דירוג מילולי קטן
            if overall_score >= 85:
                label_txt = "Ready – HF-grade"
                icon = "🟢"
            elif overall_score >= 70:
                label_txt = "Conditionally ready"
                icon = "🟡"
            elif overall_score >= 55:
                label_txt = "Watch / refine"
                icon = "🟠"
            else:
                label_txt = "Not ready"
                icon = "🔴"

            st.metric(
                label=f"{icon} Overall readiness",
                value=f"{overall_score} / 100",
                help=label_txt,
            )
            st.caption(f"Interpretation: **{label_txt}**")

    with col_r2:
        if dims:
            st.caption("Breakdown by dimension:")
            # טבלה קטנה עם score + status + reason
            dim_rows = []
            for d in dims:
                dim_rows.append(
                    {
                        "Area": d.get("label", d.get("key", "")),
                        "Score": d.get("score"),
                        "Status": d.get("status"),
                        "Reason": d.get("reason", ""),
                    }
                )
            import pandas as pd  # לוודא שקיים למעלה, ואם לא – להוסיף ב-imports
            df_dims = pd.DataFrame(dim_rows)
            st.dataframe(df_dims, width="stretch", hide_index=True)
        else:
            st.caption("No dimension-level readiness data yet.")

    # Top issues + Daily runbook על בסיס readiness + alerts + health_light
    st.markdown("---")
    _render_top_issues_panel(readiness, alerts, hl)
    st.markdown("---")
    _render_daily_runbook_panel(snapshot, home_ctx)
    st.markdown("---")

    # 2.G – Header + Story + Performance
    try:
        render_header_panel(snapshot, diff)
        st.markdown("---")
        render_performance_and_equity_panel(snapshot, diff)
    except Exception as exc:
        st.error("Failed to render header/performance panels.")
        st.caption(str(exc))

    # 2.H – Portfolio: PnL attribution + Exposure/Liquidity/Concentration
    try:
        st.markdown("---")
        render_pnl_attribution_panel(snapshot)
        render_exposure_and_concentration_panels(snapshot)
    except Exception as exc:
        st.error("Failed to render portfolio panels.")
        st.caption(str(exc))

    # 2.I – Risk panels
    try:
        st.markdown("---")
        render_risk_panels(snapshot)
    except Exception as exc:
        st.error("Failed to render risk panels.")
        st.caption(str(exc))

    # 2.J – Signals & Macro
    try:
        st.markdown("---")
        render_signals_panels(snapshot)
    except Exception as exc:
        st.error("Failed to render signals panels.")
        st.caption(str(exc))

    try:
        st.markdown("---")
        render_macro_panels(snapshot)
    except Exception as exc:
        st.error("Failed to render macro panels.")
        st.caption(str(exc))

    # 2.K – System / Infrastructure
    try:
        st.markdown("---")
        render_system_panels(snapshot)
    except Exception as exc:
        st.error("Failed to render system panels.")
        st.caption(str(exc))
