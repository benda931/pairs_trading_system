# -*- coding: utf-8 -*-
"""
portfolio_tab.py — טאב פרוטפוליו לייב ברמת קרן גידור (v2, Part 1/10)
====================================================================

טאב "פרוטפוליו לייב" אחראי לתת מבט-על ברמת קרן גידור על *מערכת המסחר החיה*:

- כל הזוגות בפוזיציות חיות כרגע (ברמת זוג וברמת לג).
- KPI-ים מרכזיים ברמת חשבון:
    * Equity, Cash, Margin, Leverage, VaR, Drawdown.
    * PnL יומי / פתוח / מצטבר.
    * חשיפה נטו/ברוטו, Long/Short, ריכוזי סיכון.
- חיבור ל-Live Universe:
    * LivePairProfile (פרופיל לייב לכל זוג).
    * LivePairStore (DuckDB) כ-Source-of-Truth לזוגות המאושרים.
    * אינדיקציה מי מהזוגות "Active", מי "Suspended", מה Score/Edge/Regime.
- הכנה מלאה לחיבור למנוע המסחר החי:
    * ctx["engine_status"] – מצב מנוע (RUNNING/PAUSED/HALTED).
    * ctx["live_positions"] – DataFrame של פוזיציות לייב (אפשר מ־IBKR / DB).
    * בעתיד – פקודות Pause/Resume/Close-all וכו'.

המבנה ב-10 חלקים:
-------------------
1. (כאן) Imports, logger, type aliases, Enums, קונפיגים, מודלי KPI/Analytics לייב.
2. שכבת טעינת נתונים לייב: live_positions, equity_curve, trades, live_universe.
3. חישובי KPI, חשיפות, VaR, Drawdown, בריאות (Health Checks) מותאמים ללייב.
4. לוגיקת פילטרים לייב (Strategies, Sectors, Asset Classes, Text Search...).
5. תצוגת KPI Header + Engine Status + Equity & PnL Charts.
6. תצוגת Exposure לפי זוג/סקטור/סימבול/אסטרטגיה + Drilldown מנהלי.
7. שילוב LivePairProfile ב-UI (Score, Edge, Regime, Active/Suspended) + תצוגת Universe.
8. Action Hints & Risk Alerts – הצעות פעולה + דגלים אדום/צהוב ברמת קרן גידור.
9. חיבור לפעולות מנוע (בהמשך): Hooks ל-API פנימי / פקודות Control.
10. Orchestrator ראשי: render_portfolio_tab_live(...).

חלק 1/10 מתמקד בהגדרת:
    - Logger ו-Type Aliases.
    - Enums (מודלי VaR, Risk, Aggregation, ViewMode, HealthSeverity).
    - קונפיגים (RiskConfig, DisplayConfig, LiveIntegrationConfig, PortfolioConfig).
    - מודלי KPI (PortfolioKPI) ו-Analytics (PortfolioAnalytics) מותאמים לייב.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px  # לשימוש בחלקים הבאים
import plotly.graph_objects as go  # לשימוש בחלקים הבאים
import streamlit as st
import os
import socket
import getpass

# טיפוסים זמינים רק בזמן type-checking (Pylance/mypy),
# כדי לא לשבור ריצה אם הקבצים עוד לא קיימים בפועל.
if TYPE_CHECKING:
    from common.live_profiles import LivePairProfile
    from common.live_pair_store import LivePairStore

from core.risk_parity import (
    risk_parity_from_returns,
    compute_risk_contributions,
)

# ============================================================
# Logger
# ============================================================

logger = logging.getLogger("LivePortfolioTab")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
        )
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

__all__ = [
    # פונקציית orchestrator ראשית תוגדר בחלק 10
    "render_portfolio_tab_live",
    "render_portfolio_tab",
    # קונפיגים וקבועים
    "VarMethod",
    "RiskModel",
    "AggregationLevel",
    "PortfolioViewMode",
    "HealthSeverity",
    "PortfolioKPI",
    "HealthCheckResult",
    "LiveIntegrationConfig",
    "RiskConfig",
    "DisplayConfig",
    "PortfolioConfig",
    "PortfolioAnalytics",
    # Utilities ל-KPI/ctx (יוגדרו בהמשך החלקים)
]


# ============================================================
# Type aliases
# ============================================================

Number = Union[float, int]
Frame = pd.DataFrame
Series = pd.Series
MaybeFrame = Optional[Frame]
JSONLike = Mapping[str, Any]


# ============================================================
# Enums — מודלי VaR / סיכון / תצוגה / בריאות
# ============================================================

class VarMethod(str, Enum):
    """שיטת חישוב VaR (Value-at-Risk)."""

    HISTORICAL = "historical"   # סימולציה על התפלגות תשואות היסטורית
    PARAMETRIC = "parametric"   # גאוסי (μ, σ) – קירוב מהיר
    CORRIDOR = "corridor"       # Historical trimmed – חיתוך זנבות קיצוניים


class RiskModel(str, Enum):
    """מודל סיכון בסיסי לפרוטפוליו."""

    DIAGONAL = "diagonal"   # סטיות תקן עצמאיות (cov ≈ 0)
    FULL = "full"           # Covariance מלאה (Correlation * Vol)
    FACTOR = "factor"       # מודל פקטורים (סקטורים / מאקרו) – לעתיד


class AggregationLevel(str, Enum):
    """רמת אגרגציה לניתוח חשיפות."""

    PAIR = "pair"
    SYMBOL = "symbol"
    SECTOR = "sector"
    STRATEGY = "strategy"
    ASSET_CLASS = "asset_class"


class PortfolioViewMode(str, Enum):
    """
    מצב תצוגת הטאב:

    - OVERVIEW: מבט כללי (KPI, Equity, Exposure).
    - RISK: זווית סיכון (VaR, Drawdown, Concentration).
    - LIVE_UNIVERSE: מבט על universe הלייב (LivePairProfile/Store).
    - DIAGNOSTICS: בדיקות בריאות / בעיות דאטה / missing config.
    """

    OVERVIEW = "overview"
    RISK = "risk"
    LIVE_UNIVERSE = "live_universe"
    DIAGNOSTICS = "diagnostics"


class HealthSeverity(str, Enum):
    """רמת חומרה של Health Check."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ============================================================
# Dataclasses — קונפיגים: Live Integration / Risk / Display / Portfolio
# ============================================================

@dataclass
class LiveIntegrationConfig:
    """
    הגדרות אינטגרציה למסחר חי.

    מגדיר:
      - איפה יושב LivePairStore (DuckDB).
      - ספי כניסה לזוג לייב (score, ml_edge, priority).
      - אינדיקציה למנוע הלייב (API פנימי, ctx וכו').

    בשלב ראשון:
      - רק קובץ DuckDB + table name + ספים.
    אחר כך:
      - endpoint ל-API פנימי (FastAPI / internal service).
      - timeouts, heartbeat וכו'.
    """

    live_pairs_db_path: str = "data/live_pairs.duckdb"
    live_pairs_table: str = "live_pairs_profile"

    # סף מינימלי לסינון universe לייב (score_total, ml_edge, priority)
    min_score_total: float = 0.0
    min_ml_edge: Optional[float] = None
    max_priority_rank: Optional[int] = None

    # אינדיקציה למנוע – רק מידע, לא חובה לחישובים כאן
    engine_name: str = "live_engine"
    engine_status_key: str = "engine_status"   # ctx["engine_status"]
    engine_positions_key: str = "live_positions"  # ctx["live_positions"], אם קיים
    engine_health_key: str = "engine_health"   # ctx["engine_health"], אם קיים

    # דגל דמו – אם אין live universe אמיתי, האם להראות דמו/להסתיר הכל?
    allow_demo_live_universe: bool = True


@dataclass
class RiskConfig:
    """
    הגדרות סיכון גלובליות לטאב הפרוטפוליו (לייב).

    ההגדרות כאן לא פותחות/סוגרות טריידים בפועל (זה תפקיד מנוע המסחר),
    אלא משמשות ל:
      - הת预ראה (Alerts) בדשבורד.
      - ניתוח סיכונים (VaR, Leverage, Concentration וכו').
    """

    # VaR
    var_horizon_days: int = 1
    var_confidence: float = 0.95
    var_method: VarMethod = VarMethod.HISTORICAL
    risk_model: RiskModel = RiskModel.FULL

    # חלונות תנודתיות וקורלציה
    volatility_lookback_days: int = 60
    correlation_lookback_days: int = 60

    # מינוף וריכוזי סיכון
    max_leverage: float = 3.0
    max_single_pair_pct: float = 0.15      # אחוז מקס' מה-Eq בזוג בודד
    max_sector_pct: float = 0.35           # אחוז מקס' לסקטור
    max_asset_class_pct: float = 0.50      # אחוז מקס' ל-Asset Class

    # ספי Alerts
    alert_var_ratio: float = 0.03          # VaR/Eq מעל 3% = אזהרה
    alert_drawdown_pct: float = 0.10       # DD מעל 10% = אזהרה
    alert_margin_utilization: float = 0.60 # שימוש במרווח מעל 60% = אזהרה


@dataclass
class DisplayConfig:
    """
    הגדרות תצוגה לייב — מה וכמה להראות בטאב.

    זה קונפיג "חזותי" בלבד, לא משפיע על המסחר.
    """

    base_currency: str = "USD"

    # אילו חלקים להציג
    show_equity_curve: bool = True
    show_risk_section: bool = True
    show_live_universe_section: bool = True
    show_diagnostics_section: bool = True
    show_actions_section: bool = True

    # פילטר בסיסי לגודל פוזיציות
    min_notional_to_show: float = 1_000.0
    hide_tiny_pnl_abs: float = 10.0

    # דיוק בפורמט
    precision_money: int = 2
    precision_pct: int = 2
    precision_qty: int = 0

    # ברירות מחדל לתצוגה
    default_aggregation: AggregationLevel = AggregationLevel.PAIR
    default_view_mode: PortfolioViewMode = PortfolioViewMode.OVERVIEW


@dataclass
class PortfolioConfig:
    """
    תצורת-על של טאב הפרוטפוליו לייב.

    משלבת:
      - RiskConfig   — פרמטרים לניתוח סיכון ברמת קרן גידור.
      - DisplayConfig — איך להציג.
      - LiveIntegrationConfig — איך להתחבר ל-Live Universe & Engine.
      - מזהי חשבון/קבוצה ופרטי דאטה (DB names).
    """

    risk: RiskConfig = field(default_factory=RiskConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    live: LiveIntegrationConfig = field(default_factory=LiveIntegrationConfig)

    # חשבון/קבוצה (אם יש בקרן מספר חשבונות/קבוצות)
    account_id: Optional[str] = None
    portfolio_group: Optional[str] = None

    # טבלאות בדאטהבייס (אם תרצה להשתמש ב-DB להיסטוריה)
    trades_table: str = "trades"
    positions_table: str = "positions"
    equity_curve_table: str = "equity_curve"

    # דמו/פיתוח
    allow_demo_mode: bool = True


# ============================================================
# Dataclasses — KPI & Health & Analytics
# ============================================================

@dataclass
class PortfolioKPI:
    """
    KPI מרכזיים של החשבון/פרוטפוליו לייב.

    כל השדות מייצגים צילום מצב (Snapshot) בנקודת זמן אחת:
        - Equity / Cash / Margin / Exposure / Leverage.
        - PnL פתוח, סגור, יומי.
        - VaR יומי ו-Drawdown (אם ניתן לחשב).
    """

    as_of: datetime

    equity: float                      # הון כולל (Equity / NLV)
    cash: float                        # מזומן פנוי
    margin_available: float            # מרווח זמין
    margin_used: float                 # מרווח בשימוש

    gross_exposure: float              # Σ |MV_i|
    net_exposure: float                # Σ MV_i
    long_exposure: float               # Σ MV_i חיובי
    short_exposure: float              # Σ |MV_i שלילי|

    leverage: float                    # gross_exposure / equity
    margin_utilization: float          # margin_used / (margin_used + margin_available)

    pnl_unrealized: float              # PnL פתוח
    pnl_realized: float                # PnL סגור מצטבר
    pnl_today: float                   # שינוי יומי (PnL של היום)

    # Risk diagnostics
    var_1d: Optional[float] = None     # VaR יומי (Value-at-Risk)
    var_1d_ratio: Optional[float] = None  # VaR / Equity

    max_drawdown_pct: Optional[float] = None
    current_drawdown_pct: Optional[float] = None


@dataclass
class HealthCheckResult:
    """
    תוצאה בודדת של בדיקת בריאות פרוטפוליו:

    code:
        מזהה קצר (למשל "LEVERAGE_HIGH", "MARGIN_UTILIZATION_HIGH").
    severity:
        INFO / WARNING / CRITICAL.
    message:
        ניסוח קצר וברור ל-UI.
    details:
        טקסט חופשי להרחבה (אפשר ב-expander או tooltip).
    """

    code: str
    severity: HealthSeverity
    message: str
    details: Optional[str] = None


@dataclass
class PortfolioAnalytics:
    """
    אובייקט אנליטיקה מרכזי של טאב הפרוטפוליו לייב:

    - kpi: מדדי על ברמת החשבון.
    - exposure_*: טבלאות חשיפה לפי זוג/סקטור/סימבול/אסטרטגיה.
    - health_checks: בדיקות בריאות שהופעלו.
    - live_profiles_by_pair: מפה מ-pair_id ל-LivePairProfile (אם קיים).
    - engine_status: מידע תפעולי על מצב המנוע החי (ctx["engine_status"]).
    """

    kpi: PortfolioKPI
    exposure_by_pair: Frame
    exposure_by_sector: Frame
    exposure_by_symbol: Frame
    exposure_by_strategy: Frame
    health_checks: List[HealthCheckResult] = field(default_factory=list)

    # מפה מ-pair_id -> פרופיל לייב מלא (מה-LivePairStore)
    live_profiles_by_pair: Dict[str, "LivePairProfile"] = field(default_factory=dict)

    # engine_status יכול להיות dict עם:
    #   mode: RUNNING/PAUSED/HALTED
    #   last_heartbeat: datetime
    #   open_pairs_count, errors_last_1h וכו'
    engine_status: Dict[str, Any] = field(default_factory=dict)

    # ============================================================
# ctx helpers — חילוץ מידע בסיסי מחשבון מתוך ctx
# ============================================================

def extract_account_snapshot_from_ctx(ctx: Optional[JSONLike]) -> Dict[str, Optional[float]]:
    """
    מנסה לחלץ נתוני חשבון (Equity, Cash, Margin) מתוך ctx של הדשבורד.

    ציפייה:
        ctx = {
            "account": {
                "equity": 250000.0,
                "cash": 75000.0,
                "margin_available": 120000.0,
                "margin_used": 55000.0,
                ...
            },
            ...
        }

    הפונקציה מחזירה מילון עם המפתחות:
        equity, cash, margin_available, margin_used
    גם אם חלק מהם חסרים (יחזרו כ-None).
    """
    snapshot: Dict[str, Optional[float]] = {
        "equity": None,
        "cash": None,
        "margin_available": None,
        "margin_used": None,
    }

    if not ctx:
        return snapshot

    try:
        account = ctx.get("account", {})
        if isinstance(account, Mapping):
            for key in snapshot:
                value = account.get(key)
                if isinstance(value, (int, float)):
                    snapshot[key] = float(value)
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to extract account snapshot from ctx: %s", exc)

    return snapshot

__all__ += ["extract_account_snapshot_from_ctx"]

# ============================================================
# Part 2 — Live data loading: positions, equity, trades, live universe, engine status
# ============================================================

# נוסיף את הפונקציות החשובות ל-__all__ שהוגדר בחלק 1
__all__ += [
    "load_live_positions_from_ctx_or_system",
    "load_equity_curve_live",
    "load_trades_history_live",
    "load_live_universe_for_portfolio",
    "extract_engine_status_from_ctx",
]

# ==========================
# Optional imports מהמערכת
# ==========================

# אם יש לך כבר loaders קיימים במערכת – נתחבר אליהם.
try:
    # לדוגמה: טעינת צילום מצב חשבון / פוזיציות / equity curve
    from common.portfolio_loader import (  # type: ignore
        load_portfolio_snapshot,
        load_equity_curve,
    )
except Exception:  # pragma: no cover
    load_portfolio_snapshot = None  # type: ignore[misc]
    load_equity_curve = None        # type: ignore[misc]

try:
    # אם קיים loader להיסטוריית טריידים
    from common.data_loader import load_trades_history  # type: ignore
except Exception:  # pragma: no cover
    load_trades_history = None  # type: ignore[misc]

# ------------------------------------------------------------
# Canonical column sets — positions / equity / trades (LIVE)
# ------------------------------------------------------------

# עמודות קאנוניות לפוזיציות לייב (זוגי פוזיציות)
_LIVE_POSITION_COLUMNS: List[str] = [
    # זיהוי פוזיציה / זוג / חשבון
    "position_id",
    "pair_id",
    "strategy",
    "strategy_bucket",
    "asset_class",
    "account_id",
    "portfolio_group",
    "source",          # "live", "paper", "backtest", "demo"...

    # פרטי לג X/Y
    "symbol_x",
    "symbol_y",
    "currency_x",
    "currency_y",
    "exchange_x",
    "exchange_y",
    "multiplier_x",
    "multiplier_y",
    "qty_x",
    "qty_y",
    "side_x",          # LONG/SHORT
    "side_y",

    "entry_price_x",
    "entry_price_y",
    "last_price_x",
    "last_price_y",

    # Greeks/בטא (אופציונלי)
    "beta_x",
    "beta_y",

    # ערכים כספיים
    "mv_x",                 # market value leg X (signed)
    "mv_y",                 # market value leg Y (signed)
    "notional_pair",        # |mv_x| + |mv_y|
    "unrealized_pnl",
    "realized_pnl",
    "pnl_today",

    # מאפייני סקטור/asset נוספים
    "sector_x",
    "sector_y",
    "industry_x",
    "industry_y",

    # סטטוס וזמנים
    "status",               # "OPEN", "PARTIAL", "CLOSING"
    "open_time",
    "update_time",

    # מדדי ספרד (היום/נוכחיים)
    "spread",
    "spread_z",
]

# עמודות קאנוניות לעקומת Equity Live
_LIVE_EQUITY_COLUMNS: List[str] = [
    "timestamp",
    "equity",
    "cash",
    "pnl_daily",
    "drawdown_pct",
]

# עמודות קאנוניות להיסטוריית טריידים לייב
_LIVE_TRADES_COLUMNS: List[str] = [
    "trade_id",
    "position_id",
    "pair_id",
    "strategy",
    "asset_class",
    "account_id",
    "portfolio_group",
    "symbol",
    "side",             # BUY/SELL
    "qty",
    "price",
    "commission",
    "fees",
    "notional",
    "realized_pnl",
    "open_time",
    "close_time",
    "source",           # "live" / "paper" / "backtest" / "import"
]

# ============================================================
# Engine / ctx helpers
# ============================================================

def extract_engine_status_from_ctx(ctx: Optional[JSONLike]) -> Dict[str, Any]:
    """
    מחלץ סטטוס מנוע לייב מתוך ctx, אם קיים.

    צפוי מבנה (לא חובה הכל):
        ctx["engine_status"] = {
            "mode": "RUNNING" | "PAUSED" | "HALTED",
            "last_heartbeat": datetime ISO string,
            "open_pairs": int,
            "open_positions": int,
            "error_count_1h": int,
            "last_error": "טקסט...",
        }

    אם אין ctx או אין מפתח – מחזיר dict ריק.
    """
    if not ctx:
        return {}

    # שם המפתח מוגדר ב-LiveIntegrationConfig
    engine_key = "engine_status"
    try:
        status = ctx.get(engine_key, {})
        if not isinstance(status, Mapping):
            return {}
        # אפשר להמיר תאריך ל-datetime אם הוא string
        if "last_heartbeat" in status:
            try:
                status = dict(status)
                status["last_heartbeat"] = pd.to_datetime(
                    status["last_heartbeat"],
                    errors="coerce",
                    utc=True,
                )
            except Exception:  # pragma: no cover
                pass
        return dict(status)
    except Exception:  # pragma: no cover
        logger.debug("Failed to extract engine_status from ctx", exc_info=True)
        return {}


# ============================================================
# LIVE POSITIONS — Loading & Normalization
# ============================================================

def load_live_positions_from_ctx_or_system(
    config: PortfolioConfig,
    ctx: Optional[JSONLike] = None,
) -> Frame:
    """
    טוען פוזיציות לייב (ברמת זוג) ממערכת המסחר / DB / ctx, ומחזיר DataFrame קאנוני.

    סדר עדיפויות:
    1. ctx[config.live.engine_positions_key] אם קיים כ-DataFrame (הזרקה ישירה מה-Engine).
    2. common.portfolio_loader.load_portfolio_snapshot(...) אם קיים.
    3. DataFrame דמו (אם allow_demo_mode=True).
    4. אחרת: DataFrame ריק בסכמה `_LIVE_POSITION_COLUMNS`.

    התוצאה מנורמלת דרך `_normalize_live_positions`.
    """
    # 1) ctx override – engine כבר נותן DataFrame מוכן
    if ctx is not None:
        key = config.live.engine_positions_key
        snap = ctx.get(key)
        if isinstance(snap, pd.DataFrame):
            logger.info("Using live_positions from ctx[%s]", key)
            return _normalize_live_positions(snap.copy(), config)

    # 2) Loader מערכת (אם קיים)
    if callable(load_portfolio_snapshot):
        try:
            logger.info("Loading live positions via common.portfolio_loader.load_portfolio_snapshot")
            raw = load_portfolio_snapshot(  # type: ignore[call-arg]
                account_id=config.account_id,
                portfolio_group=config.portfolio_group,
            )
            if not isinstance(raw, pd.DataFrame):
                raise TypeError(
                    f"load_portfolio_snapshot returned {type(raw)!r}, expected DataFrame"
                )
            return _normalize_live_positions(raw.copy(), config)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to load live positions from portfolio_loader, "
                "falling back to demo if allowed: %s",
                exc,
            )

    # 3) Demo mode (fallback)
    if config.allow_demo_mode:
        logger.info("Using demo live positions (no real loader configured)")
        demo = _build_demo_live_positions(config, ctx)
        return _normalize_live_positions(demo, config)

    # 4) ללא דמו וללא Loader
    logger.warning("No live positions available (loader missing & demo disabled)")
    empty = pd.DataFrame(columns=_LIVE_POSITION_COLUMNS)
    return _normalize_live_positions(empty, config)


def _normalize_live_positions(
    df_raw: Frame,
    config: PortfolioConfig,
) -> Frame:
    """
    מנרמל DataFrame של פוזיציות לייב לסכמה `_LIVE_POSITION_COLUMNS`.

    כולל:
      - rename של שמות עמודות נפוצים.
      - הוספת עמודות חסרות בערכי ברירת מחדל.
      - המרת טיפוסים (מספרים, תאריכים).
      - חישוב mv_x/mv_y/notional_pair אם אפשר.
      - מילוי account_id/portfolio_group/source מ-PortfolioConfig.
    """
    df = df_raw.copy()

    if df.empty:
        logger.info("normalize_live_positions: received empty frame")
        df = pd.DataFrame(columns=_LIVE_POSITION_COLUMNS)

    # ---- 1) rename mapping ----
    rename_map: Dict[str, str] = {
        # pair id / name
        "pair": "pair_id",
        "pair_name": "pair_id",
        "pair_symbol": "pair_id",
        # position id
        "pos_id": "position_id",
        "position": "position_id",
        # strategy
        "strategy_name": "strategy",
        "strat": "strategy",
        "bucket": "strategy_bucket",
        # asset class
        "assetClass": "asset_class",
        "assetclass": "asset_class",
        # account / group
        "account": "account_id",
        "acct": "account_id",
        "group": "portfolio_group",
        "portfolio": "portfolio_group",
        # source
        "env": "source",
        "environment": "source",
        # symbols
        "symbol1": "symbol_x",
        "symbol2": "symbol_y",
        "leg1": "symbol_x",
        "leg2": "symbol_y",
        # currencies
        "ccy_x": "currency_x",
        "ccy_y": "currency_y",
        "currency1": "currency_x",
        "currency2": "currency_y",
        # exchange
        "exch1": "exchange_x",
        "exch2": "exchange_y",
        # multiplier
        "mult1": "multiplier_x",
        "mult2": "multiplier_y",
        # qty
        "qty1": "qty_x",
        "qty2": "qty_y",
        "quantity_x": "qty_x",
        "quantity_y": "qty_y",
        # side (LONG/SHORT)
        "side1": "side_x",
        "side2": "side_y",
        # prices
        "price1": "last_price_x",
        "price2": "last_price_y",
        "entry1": "entry_price_x",
        "entry2": "entry_price_y",
        # PnL
        "unrealized": "unrealized_pnl",
        "realized": "realized_pnl",
        "pnl": "unrealized_pnl",
        "pnl_today": "pnl_today",
        # sectors / industries
        "sector1": "sector_x",
        "sector2": "sector_y",
        "industry1": "industry_x",
        "industry2": "industry_y",
        # status
        "state": "status",
        # times
        "open_datetime": "open_time",
        "open_dt": "open_time",
        "timestamp": "update_time",
        "as_of": "update_time",
        "last_update": "update_time",
        # spread fields
        "zscore": "spread_z",
        "spread_zscore": "spread_z",
    }

    intersecting = {c: rename_map[c] for c in df.columns if c in rename_map}
    if intersecting:
        df = df.rename(columns=intersecting)

    # ---- 2) ensure all LIVE_POSITION_COLUMNS exist ----
    for col in _LIVE_POSITION_COLUMNS:
        if col not in df.columns:
            if col in (
                "qty_x",
                "qty_y",
                "entry_price_x",
                "entry_price_y",
                "last_price_x",
                "last_price_y",
                "mv_x",
                "mv_y",
                "multiplier_x",
                "multiplier_y",
                "notional_pair",
                "unrealized_pnl",
                "realized_pnl",
                "pnl_today",
                "beta_x",
                "beta_y",
                "spread",
                "spread_z",
            ):
                df[col] = np.nan
            elif col in (
                "strategy",
                "strategy_bucket",
                "asset_class",
                "sector_x",
                "sector_y",
                "industry_x",
                "industry_y",
                "currency_x",
                "currency_y",
                "exchange_x",
                "exchange_y",
                "status",
                "source",
            ):
                df[col] = "Unknown"
            elif col in ("position_id", "pair_id", "account_id", "portfolio_group"):
                df[col] = ""
            elif col in ("open_time", "update_time"):
                df[col] = pd.NaT
            else:
                df[col] = np.nan

    # ---- 3) type coercion ----
    numeric_cols = [
        "qty_x",
        "qty_y",
        "entry_price_x",
        "entry_price_y",
        "last_price_x",
        "last_price_y",
        "mv_x",
        "mv_y",
        "multiplier_x",
        "multiplier_y",
        "notional_pair",
        "unrealized_pnl",
        "realized_pnl",
        "pnl_today",
        "beta_x",
        "beta_y",
        "spread",
        "spread_z",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    datetime_cols = ["open_time", "update_time"]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    str_cols = [
        "position_id",
        "pair_id",
        "strategy",
        "strategy_bucket",
        "asset_class",
        "account_id",
        "portfolio_group",
        "symbol_x",
        "symbol_y",
        "currency_x",
        "currency_y",
        "exchange_x",
        "exchange_y",
        "sector_x",
        "sector_y",
        "industry_x",
        "industry_y",
        "status",
        "side_x",
        "side_y",
        "source",
    ]
    for col in str_cols:
        df[col] = df[col].astype(str).fillna("")

    # ---- 4) pair_id אם חסר ----
    if df["pair_id"].eq("").all():
        df["pair_id"] = (
            df["symbol_x"].astype(str).str.strip()
            + "_"
            + df["symbol_y"].astype(str).str.strip()
        )

    # ---- 5) account_id / portfolio_group / source מ-PortfolioConfig אם חסר ----
    if config.account_id:
        df.loc[df["account_id"] == "", "account_id"] = str(config.account_id)
    if config.portfolio_group:
        df.loc[df["portfolio_group"] == "", "portfolio_group"] = str(config.portfolio_group)
    df.loc[df["source"] == "Unknown", "source"] = "live"

    # ---- 6) חישוב mv_x, mv_y אם חסרים ----
    def _signed_mv(qty: float, price: float, side: str) -> float:
        if pd.isna(qty) or pd.isna(price):
            return np.nan
        s = str(side).upper()
        sign = 1.0
        if "SHORT" in s or s.startswith("S"):
            sign = -1.0
        return float(qty) * float(price) * sign

    need_mv_x = df["mv_x"].isna()
    if need_mv_x.any():
        df.loc[need_mv_x, "mv_x"] = df.loc[
            need_mv_x, ["qty_x", "last_price_x", "side_x"]
        ].apply(
            lambda row: _signed_mv(row["qty_x"], row["last_price_x"], row["side_x"]),
            axis=1,
        )

    need_mv_y = df["mv_y"].isna()
    if need_mv_y.any():
        df.loc[need_mv_y, "mv_y"] = df.loc[
            need_mv_y, ["qty_y", "last_price_y", "side_y"]
        ].apply(
            lambda row: _signed_mv(row["qty_y"], row["last_price_y"], row["side_y"]),
            axis=1,
        )

    # ---- 7) notional_pair ----
    need_notional = df["notional_pair"].isna()
    if need_notional.any():
        df.loc[need_notional, "notional_pair"] = (
            df["mv_x"].abs() + df["mv_y"].abs()
        )

    df["notional_pair"] = df["notional_pair"].fillna(0.0)

    # סינון פוזיציות זעירות לפי config.display.min_notional_to_show
    min_notional = config.display.min_notional_to_show
    if min_notional > 0:
        before = len(df)
        df = df[df["notional_pair"].abs() >= min_notional].copy()
        after = len(df)
        if before != after:
            logger.info(
                "Filtered live positions below notional %.2f: %d -> %d rows",
                min_notional,
                before,
                after,
            )

    df = df[_LIVE_POSITION_COLUMNS].copy()
    df = df.reset_index(drop=True)
    return df


def _build_demo_live_positions(
    config: PortfolioConfig,
    ctx: Optional[JSONLike] = None,
) -> Frame:
    """
    בונה צילום מצב דמו של פוזיציות לייב (לשימוש בפיתוח/ללא חיבור ל-IBKR).

    לא מיועד לייצור – רק כדי שהטאב יוכל לעבוד גם בסביבה "רותחת" בלי דאטה.
    """
    rng = np.random.default_rng(42)

    demo_pairs: List[Tuple[str, str, str, str]] = [
        ("XLY_XLC", "EQ_Pairs", "Equity", "XLY"),
        ("QQQ_SOXX", "Tech_Pairs", "Equity", "QQQ"),
        ("TLT_SHY", "Rates_Spread", "Rates", "TLT"),
    ]
    rows: List[Dict[str, Any]] = []

    account_id = config.account_id or "DEMO-ACCT"
    portfolio_group = config.portfolio_group or "DEMO-GROUP"

    now = pd.Timestamp.utcnow()

    for i, (pair_id, strategy, asset_class, sym_x) in enumerate(demo_pairs, start=1):
        if pair_id == "XLY_XLC":
            sym_y = "XLC"
            sector_x = "Consumer"
            sector_y = "Communication"
        elif pair_id == "QQQ_SOXX":
            sym_y = "SOXX"
            sector_x = "Tech"
            sector_y = "Semiconductors"
        else:
            sym_y = "SHY"
            sector_x = "Rates"
            sector_y = "Rates"

        position_id = f"DEMO-{i:03d}"

        qty_x = int(rng.integers(50, 300))
        qty_y = int(rng.integers(50, 300))

        side_x = "LONG"
        side_y = "SHORT"

        entry_x = float(rng.normal(100, 10))
        entry_y = float(rng.normal(90, 8))

        last_x = entry_x * float(rng.normal(1.02, 0.02))
        last_y = entry_y * float(rng.normal(0.98, 0.02))

        mv_x = qty_x * last_x
        mv_y = -qty_y * last_y  # short

        notional_pair = abs(mv_x) + abs(mv_y)

        unrealized = (mv_x - qty_x * entry_x) + (
            mv_y - (-qty_y * entry_y)
        )

        realized = float(rng.normal(0, 0.002 * notional_pair))
        pnl_today = float(rng.normal(0, 0.0005 * notional_pair))

        beta_x = float(rng.normal(1.0, 0.2))
        beta_y = float(rng.normal(-1.0, 0.2))

        spread = last_x - last_y
        spread_z = float(rng.normal(0, 1))

        rows.append(
            {
                "position_id": position_id,
                "pair_id": pair_id,
                "strategy": strategy,
                "strategy_bucket": "Core",
                "asset_class": asset_class,
                "account_id": account_id,
                "portfolio_group": portfolio_group,
                "source": "demo_live",
                "symbol_x": sym_x,
                "symbol_y": sym_y,
                "currency_x": "USD",
                "currency_y": "USD",
                "exchange_x": "NYSE",
                "exchange_y": "NYSE",
                "multiplier_x": 1.0,
                "multiplier_y": 1.0,
                "qty_x": qty_x,
                "qty_y": qty_y,
                "side_x": side_x,
                "side_y": side_y,
                "entry_price_x": entry_x,
                "entry_price_y": entry_y,
                "last_price_x": last_x,
                "last_price_y": last_y,
                "mv_x": mv_x,
                "mv_y": mv_y,
                "notional_pair": notional_pair,
                "unrealized_pnl": unrealized,
                "realized_pnl": realized,
                "pnl_today": pnl_today,
                "sector_x": sector_x,
                "sector_y": sector_y,
                "industry_x": sector_x,
                "industry_y": sector_y,
                "beta_x": beta_x,
                "beta_y": beta_y,
                "spread": spread,
                "spread_z": spread_z,
                "status": "OPEN",
                "open_time": now - pd.Timedelta(days=int(rng.integers(5, 60))),
                "update_time": now,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# LIVE EQUITY CURVE — Loading & Normalization
# ============================================================

def load_equity_curve_live(
    config: PortfolioConfig,
    ctx: Optional[JSONLike] = None,
) -> MaybeFrame:
    """
    טוען עקומת Equity לייב (אם קיימת) ומחזיר DataFrame קאנוני.

    סדר עדיפויות:
    1. ctx["equity_curve_live"] / ctx["equity_curve"] אם קיימים כ-DataFrame.
    2. common.portfolio_loader.load_equity_curve(...) אם קיים.
    3. Equity Curve דמו (על בסיס פוזיציות) אם allow_demo_mode=True.
    4. אחרת: None.
    """
    # 1) ctx override
    if ctx is not None:
        for key in ("equity_curve_live", "equity_curve"):
            curve = ctx.get(key)
            if isinstance(curve, pd.DataFrame):
                logger.info("Using equity curve live from ctx[%s]", key)
                return _normalize_equity_curve_live(curve.copy(), config)

    # 2) Loader מערכת (אם קיים)
    if callable(load_equity_curve):
        try:
            logger.info("Loading live equity curve via common.portfolio_loader.load_equity_curve")
            raw = load_equity_curve(  # type: ignore[call-arg]
                account_id=config.account_id,
                portfolio_group=config.portfolio_group,
            )
            if not isinstance(raw, pd.DataFrame):
                raise TypeError(
                    f"load_equity_curve returned {type(raw)!r}, expected DataFrame"
                )
            return _normalize_equity_curve_live(raw.copy(), config)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to load equity curve from loader, "
                "falling back to demo if allowed: %s",
                exc,
            )

    # 3) Demo curve
    if config.allow_demo_mode:
        logger.info("Using demo equity curve (no live equity curve provided)")
        demo_positions = _build_demo_live_positions(config, ctx)
        curve_demo = _build_demo_equity_curve_from_positions(demo_positions, config)
        return _normalize_equity_curve_live(curve_demo, config)

    # 4) אין
    logger.info("No equity curve live available (loader missing & demo disabled)")
    return None


def _normalize_equity_curve_live(
    df_raw: Frame,
    config: PortfolioConfig,
) -> Frame:
    """
    מנרמל עקומת Equity לסכמה `_LIVE_EQUITY_COLUMNS`.
    """
    df = df_raw.copy()

    if df.empty:
        logger.info("normalize_equity_curve_live: received empty frame")
        return pd.DataFrame(columns=_LIVE_EQUITY_COLUMNS)

    rename_map: Dict[str, str] = {
        "date": "timestamp",
        "dt": "timestamp",
        "time": "timestamp",
        "equity_value": "equity",
        "portfolio_value": "equity",
        "nav": "equity",
        "nlv": "equity",
        "cash_balance": "cash",
        "cash_equiv": "cash",
        "pnl": "pnl_daily",
        "daily_pnl": "pnl_daily",
        "dd_pct": "drawdown_pct",
        "drawdown": "drawdown_pct",
    }
    intersecting = {c: rename_map[c] for c in df.columns if c in rename_map}
    if intersecting:
        df = df.rename(columns=intersecting)

    for col in _LIVE_EQUITY_COLUMNS:
        if col not in df.columns:
            if col == "timestamp":
                df[col] = pd.NaT
            else:
                df[col] = np.nan

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df["cash"] = pd.to_numeric(df["cash"], errors="coerce")
    df["pnl_daily"] = pd.to_numeric(df["pnl_daily"], errors="coerce")
    df["drawdown_pct"] = pd.to_numeric(df["drawdown_pct"], errors="coerce")

    df = df.sort_values("timestamp").reset_index(drop=True)

    # pnl_daily אם חסר
    if df["pnl_daily"].isna().mean() > 0.5 and df["equity"].notna().sum() > 1:
        df["pnl_daily"] = df["equity"].diff().fillna(0.0)

    # drawdown אם חסר
    if df["drawdown_pct"].isna().mean() > 0.5 and df["equity"].notna().sum() > 1:
        rolling_max = df["equity"].cummax()
        df["drawdown_pct"] = (df["equity"] / rolling_max - 1.0).fillna(0.0)

    df = df[_LIVE_EQUITY_COLUMNS].copy()
    return df


def _build_demo_equity_curve_from_positions(
    positions: Frame,
    config: PortfolioConfig,
) -> Frame:
    """
    בונה Equity Curve דמו על בסיס notional / PnL מהפוזיציות.

    - base_equity נגזר מגודל הפוזיציות.
    - תשואות יומיות סינתטיות עם תנודתיות סבירה.
    """
    rng = np.random.default_rng(123)

    approx_notional = float(
        pd.to_numeric(positions.get("notional_pair", 0.0), errors="coerce").fillna(0.0).sum()
    )
    base_equity = max(approx_notional * 0.3, 250_000.0)

    n_days = max(config.risk.volatility_lookback_days, 60)
    dates = pd.date_range(
        end=pd.Timestamp.utcnow().normalize(),
        periods=n_days,
        freq="B",
        tz="UTC",
    )

    daily_vol = 0.01
    daily_rets = rng.normal(loc=0.0002, scale=daily_vol, size=n_days)
    equity = base_equity * (1.0 + pd.Series(daily_rets)).cumprod()

    cash = equity * 0.2 + rng.normal(0, 1000, size=n_days)
    pnl_daily = np.concatenate([[0.0], np.diff(equity)])

    eq_series = pd.Series(equity)
    rolling_max = eq_series.cummax()
    drawdown_pct = (eq_series / rolling_max - 1.0).values

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "equity": equity,
            "cash": cash,
            "pnl_daily": pnl_daily,
            "drawdown_pct": drawdown_pct,
        }
    )
    return df


# ============================================================
# LIVE TRADES HISTORY — Loading & Normalization
# ============================================================

def load_trades_history_live(
    config: PortfolioConfig,
    ctx: Optional[JSONLike] = None,
    days_back: Optional[int] = 90,
) -> MaybeFrame:
    """
    טוען היסטוריית טריידים לייב (אם קיימת) ומחזיר DataFrame קאנוני.

    סדר עדיפויות:
    1. ctx["trades_history_live"] / ctx["trades_history"] אם קיים כ-DataFrame.
    2. common.data_loader.load_trades_history(...) אם קיים.
    3. טבלת דמו (על בסיס פוזיציות) אם allow_demo_mode=True.
    4. אחרת: None.
    """
    # 1) ctx override
    if ctx is not None:
        for key in ("trades_history_live", "trades_history"):
            th = ctx.get(key)
            if isinstance(th, pd.DataFrame):
                logger.info("Using trades_history from ctx[%s]", key)
                return _normalize_trades_history_live(th.copy(), config)

    # 2) data_loader
    if callable(load_trades_history):
        try:
            logger.info("Loading live trades history via common.data_loader.load_trades_history")
            raw = load_trades_history(  # type: ignore[call-arg]
                account_id=config.account_id,
                portfolio_group=config.portfolio_group,
                days_back=days_back,
            )
            if not isinstance(raw, pd.DataFrame):
                raise TypeError(
                    f"load_trades_history returned {type(raw)!r}, expected DataFrame"
                )
            return _normalize_trades_history_live(raw.copy(), config)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to load trades history from loader, "
                "falling back to demo if allowed: %s",
                exc,
            )

    # 3) Demo history
    if config.allow_demo_mode:
        logger.info("Using demo trades history (no live trades history provided)")
        demo_positions = _build_demo_live_positions(config, ctx)
        demo_trades = _build_demo_trades_history_from_positions(demo_positions, config)
        return _normalize_trades_history_live(demo_trades, config)

    # 4) אין
    logger.info("No trades history live available (loader missing & demo disabled)")
    return None


def _normalize_trades_history_live(
    df_raw: Frame,
    config: PortfolioConfig,
) -> Frame:
    """
    מנרמל היסטוריית טריידים לייב לסכמה `_LIVE_TRADES_COLUMNS`.
    """
    df = df_raw.copy()

    if df.empty:
        logger.info("normalize_trades_history_live: received empty frame")
        return pd.DataFrame(columns=_LIVE_TRADES_COLUMNS)

    rename_map: Dict[str, str] = {
        "id": "trade_id",
        "tradeID": "trade_id",
        "pos_id": "position_id",
        "position": "position_id",
        "pair": "pair_id",
        "pair_name": "pair_id",
        "strategy_name": "strategy",
        "assetClass": "asset_class",
        "account": "account_id",
        "acct": "account_id",
        "group": "portfolio_group",
        "portfolio": "portfolio_group",
        "ticker": "symbol",
        "qty": "qty",
        "quantity": "qty",
        "side1": "side",
        "price_fill": "price",
        "avg_price": "price",
        "commission_amount": "commission",
        "fee": "fees",
        "pnl": "realized_pnl",
        "realized": "realized_pnl",
        "open_datetime": "open_time",
        "close_datetime": "close_time",
        "env": "source",
    }

    intersecting = {c: rename_map[c] for c in df.columns if c in rename_map}
    if intersecting:
        df = df.rename(columns=intersecting)

    for col in _LIVE_TRADES_COLUMNS:
        if col not in df.columns:
            if col in ("qty", "price", "commission", "fees", "notional", "realized_pnl"):
                df[col] = np.nan
            elif col in ("open_time", "close_time"):
                df[col] = pd.NaT
            elif col in (
                "trade_id",
                "position_id",
                "pair_id",
                "strategy",
                "asset_class",
                "account_id",
                "portfolio_group",
                "symbol",
                "side",
                "source",
            ):
                df[col] = ""
            else:
                df[col] = np.nan

    numeric_cols = ["qty", "price", "commission", "fees", "notional", "realized_pnl"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    datetime_cols = ["open_time", "close_time"]
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    str_cols = [
        "trade_id",
        "position_id",
        "pair_id",
        "strategy",
        "asset_class",
        "account_id",
        "portfolio_group",
        "symbol",
        "side",
        "source",
    ]
    for col in str_cols:
        df[col] = df[col].astype(str).fillna("")

    # notional אם חסר
    need_notional = df["notional"].isna()
    if need_notional.any():
        df.loc[need_notional, "notional"] = (
            df["qty"].abs() * df["price"].abs()
        )

    df = df[_LIVE_TRADES_COLUMNS].copy()
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def _build_demo_trades_history_from_positions(
    positions: Frame,
    config: PortfolioConfig,
) -> Frame:
    """
    בונה היסטוריית טריידים דמו מתוך פוזיציות הדמו.
    """
    rng = np.random.default_rng(999)

    rows: List[Dict[str, Any]] = []

    account_id = (
        positions["account_id"].iloc[0] if not positions.empty else "DEMO-ACCT"
    )
    portfolio_group = (
        positions["portfolio_group"].iloc[0] if not positions.empty else "DEMO-GROUP"
    )

    for _, row in positions.iterrows():
        position_id = row.get("position_id", "")
        pair_id = row.get("pair_id", "")
        strategy = row.get("strategy", "Unknown")
        asset_class = row.get("asset_class", "Equity")
        qty_x = float(row.get("qty_x", 0.0))
        qty_y = float(row.get("qty_y", 0.0))
        sym_x = str(row.get("symbol_x", ""))
        sym_y = str(row.get("symbol_y", ""))
        entry_x = float(row.get("entry_price_x", np.nan))
        entry_y = float(row.get("entry_price_y", np.nan))
        open_time = row.get("open_time", pd.Timestamp.utcnow() - pd.Timedelta(days=5))

        for leg_symbol, side, qty, entry_price, leg_idx in [
            (sym_x, "BUY", qty_x, entry_x, 1),
            (sym_y, "SELL", qty_y, entry_y, 2),
        ]:
            if qty <= 0 or np.isnan(entry_price):
                continue

            n_trades = int(rng.integers(1, 3))
            split = np.cumsum(np.abs(rng.normal(1.0, 0.2, size=n_trades)))
            split = split / split.sum()
            partial_qtys = (qty * split).round().astype(int)
            partial_qtys[partial_qtys == 0] = 1

            for j in range(n_trades):
                trade_id = f"DEMO-{position_id}-{leg_idx}-{j+1}"
                t_open = open_time + pd.Timedelta(minutes=int(rng.integers(0, 60 * 3)))
                commission = float(rng.normal(0.2, 0.1))
                fees = float(rng.normal(0.05, 0.02))
                notional = partial_qtys[j] * entry_price
                realized = float(rng.normal(0, notional * 0.002))

                rows.append(
                    {
                        "trade_id": trade_id,
                        "position_id": position_id,
                        "pair_id": pair_id,
                        "strategy": strategy,
                        "asset_class": asset_class,
                        "account_id": account_id,
                        "portfolio_group": portfolio_group,
                        "symbol": leg_symbol,
                        "side": side,
                        "qty": partial_qtys[j],
                        "price": entry_price,
                        "commission": commission,
                        "fees": fees,
                        "notional": notional,
                        "realized_pnl": realized,
                        "open_time": t_open,
                        "close_time": t_open + pd.Timedelta(minutes=5),
                        "source": "demo_live",
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# LIVE UNIVERSE — טעינת LivePairProfile מ-LivePairStore
# ============================================================

def load_live_universe_for_portfolio(
    config: PortfolioConfig,
) -> Dict[str, "LivePairProfile"]:
    """
    טוען את ה-Live Universe מתוך LivePairStore (אם קיים),
    ומחזיר dict מ-pair_id -> LivePairProfile.

    אם:
      - אין common.live_pair_store/live_profiles,
      - או אין קובץ DB,
      - או שיש שגיאה בטעינה,
    מחזיר dict ריק (הטאב עדיין יעבוד, רק בלי שכבת live profiles).
    """
    try:
        from common.live_pair_store import LivePairStore  # type: ignore
        from common.live_profiles import LivePairProfile  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.info(
            "LivePairStore/LivePairProfile not available — skipping live universe overlay (%s).",
            exc,
        )
        return {}

    db_path = Path(config.live.live_pairs_db_path)
    if not db_path.exists():
        if config.live.allow_demo_live_universe:
            logger.info(
                "Live pairs DB not found at %s; using empty live universe (demo).",
                db_path,
            )
        else:
            logger.warning(
                "Live pairs DB not found at %s and allow_demo_live_universe=False; live universe disabled.",
                db_path,
            )
        return {}

    try:
        store = LivePairStore(
            db_path=config.live.live_pairs_db_path,
            table_name=config.live.live_pairs_table,
        )
    except Exception as exc:
        logger.warning("Failed to open LivePairStore: %s", exc)
        return {}

    try:
        profiles = store.load_for_engine(
            min_score=config.live.min_score_total,
            min_ml_edge=config.live.min_ml_edge,
            max_priority_rank=config.live.max_priority_rank,
            limit=None,
        )
        # נוודא טיפוס – במקרה ש-storeחזיר משהו לא צפוי
        result: Dict[str, LivePairProfile] = {}
        for p in profiles:
            try:
                result[p.pair_id] = p
            except Exception:  # pragma: no cover
                logger.debug("Skipping malformed LivePairProfile from store")
        return result
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load live universe from store: %s", exc)
        return {}
    finally:
        try:
            store.close()
        except Exception:
            pass
# ============================================================
# Part 3 — KPI, Exposure & Health Analytics (Live)
# ============================================================

__all__ += [
    "compute_portfolio_kpi",
    "compute_exposure_breakdown",
    "evaluate_portfolio_health",
    "compute_portfolio_analytics",
]

# ------------------------------------------------------------
# Portfolio Analytics — פונקציה ראשית
# ------------------------------------------------------------

def compute_portfolio_analytics(
    positions: Frame,
    equity_curve: MaybeFrame,
    config: PortfolioConfig,
    ctx: Optional[JSONLike] = None,
    live_profiles_by_pair: Optional[Dict[str, "LivePairProfile"]] = None,
) -> PortfolioAnalytics:
    """
    פונקציה מרכזית שמייצרת אובייקט PortfolioAnalytics:

      - kpi: מדדי על של החשבון/פרוטפוליו לייב.
      - exposure_*: טבלאות חשיפה שונות (זוג/סקטור/סימבול/אסטרטגיה).
      - health_checks: רשימת בדיקות בריאות.
      - live_profiles_by_pair: מפה מ-pair_id -> LivePairProfile (אם סופק).
      - engine_status: סטטוס מנוע המסחר החי (אם ctx מכיל engine_status).

    מיועדת לרוץ פעם אחת בתחילת render_tab, ואז להעביר את האובייקט
    לתת-פונקציות ה-UI (KPI Header, Exposure, Health וכו').
    """
    positions_norm = positions.copy()

    # KPI כולל
    kpi = compute_portfolio_kpi(
        positions=positions_norm,
        equity_curve=equity_curve,
        config=config,
        ctx=ctx,
    )

    # חשיפות לפי רמות שונות
    exposure_pair = compute_exposure_breakdown(
        positions_norm,
        equity=kpi.equity,
        level=AggregationLevel.PAIR,
    )
    exposure_sector = compute_exposure_breakdown(
        positions_norm,
        equity=kpi.equity,
        level=AggregationLevel.SECTOR,
    )
    exposure_symbol = compute_exposure_breakdown(
        positions_norm,
        equity=kpi.equity,
        level=AggregationLevel.SYMBOL,
    )
    exposure_strategy = compute_exposure_breakdown(
        positions_norm,
        equity=kpi.equity,
        level=AggregationLevel.STRATEGY,
    )

    # סטטוס מנוע מה-ctx (אם קיים)
    engine_status = extract_engine_status_from_ctx(ctx)

    # Health checks (דגלים אדומים/צהובים)
    health_checks = evaluate_portfolio_health(
        kpi=kpi,
        positions=positions_norm,
        exposure_by_pair=exposure_pair,
        exposure_by_sector=exposure_sector,
        config=config,
    )

    return PortfolioAnalytics(
        kpi=kpi,
        exposure_by_pair=exposure_pair,
        exposure_by_sector=exposure_sector,
        exposure_by_symbol=exposure_symbol,
        exposure_by_strategy=exposure_strategy,
        health_checks=health_checks,
        live_profiles_by_pair=live_profiles_by_pair or {},
        engine_status=engine_status,
    )


# ------------------------------------------------------------
# KPI Engine — חישוב KPI מרכזיים
# ------------------------------------------------------------

def compute_portfolio_kpi(
    positions: Frame,
    equity_curve: MaybeFrame,
    config: PortfolioConfig,
    ctx: Optional[JSONLike] = None,
) -> PortfolioKPI:
    """
    מחשב KPI מרכזיים ברמת החשבון/פרוטפוליו לייב:

      - Equity / Cash / Margin (אם זמין מ-ctx / EquityCurve).
      - Gross / Net / Long / Short Exposure.
      - Leverage & Margin Utilization.
      - PnL פתוח / סגור / יומי.
      - VaR יומי בקירוב (1D VaR).
      - Drawdown נוכחי ומקסימלי (אם יש Equity Curve).

    הלוגיקה היא "Robust":
      1. מנסה להשתמש בנתוני account מתוך ctx["account"] אם קיימים.
      2. אם אין, מנסה להשתמש בעקומת Equity.
      3. אם אין כלום – קירוב גס מתוך הפוזיציות.
    """
    now = pd.Timestamp.utcnow()
    risk_cfg = config.risk

    # ----- חשיפת פוזיציות -----
    df = positions.copy()

    for col in ("mv_x", "mv_y", "notional_pair"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    mv_x = df["mv_x"]
    mv_y = df["mv_y"]

    long_exposure = mv_x.clip(lower=0).sum() + mv_y.clip(lower=0).sum()
    short_exposure_signed = mv_x.clip(upper=0).sum() + mv_y.clip(upper=0).sum()
    short_exposure = abs(short_exposure_signed)
    gross_exposure = float(long_exposure + short_exposure)
    net_exposure = float(long_exposure + short_exposure_signed)

    unrealized = float(pd.to_numeric(df.get("unrealized_pnl", 0.0), errors="coerce").sum())
    realized = float(pd.to_numeric(df.get("realized_pnl", 0.0), errors="coerce").sum())
    pnl_today_positions = float(pd.to_numeric(df.get("pnl_today", 0.0), errors="coerce").sum())

    # ----- Equity / Cash / Margin -----
    acct_snapshot = extract_account_snapshot_from_ctx(ctx or {})

    equity_ctx = acct_snapshot.get("equity")
    cash_ctx = acct_snapshot.get("cash")
    margin_av_ctx = acct_snapshot.get("margin_available")
    margin_used_ctx = acct_snapshot.get("margin_used")

    # Attempt 2: from equity curve
    equity_from_curve = None
    cash_from_curve = None
    pnl_today_curve = None
    if equity_curve is not None and not equity_curve.empty:
        last_row = equity_curve.sort_values("timestamp").iloc[-1]
        equity_from_curve = float(pd.to_numeric(last_row["equity"], errors="coerce"))
        cash_from_curve = float(pd.to_numeric(last_row["cash"], errors="coerce"))
        pnl_today_curve = float(pd.to_numeric(last_row["pnl_daily"], errors="coerce"))

    # Equity
    if isinstance(equity_ctx, (int, float)) and equity_ctx > 0:
        equity = float(equity_ctx)
    elif equity_from_curve is not None and equity_from_curve > 0:
        equity = float(equity_from_curve)
    else:
        approx_notional = float(df["notional_pair"].abs().sum())
        equity = max(approx_notional * 0.3, 100_000.0)

    # Cash
    if isinstance(cash_ctx, (int, float)):
        cash = float(cash_ctx)
    elif cash_from_curve is not None:
        cash = float(cash_from_curve)
    else:
        cash = equity * 0.2

    # Margin (בהיעדר מידע – הערכה מה-Equity והחשיפה)
    if isinstance(margin_av_ctx, (int, float)) and isinstance(margin_used_ctx, (int, float)):
        margin_available = float(margin_av_ctx)
        margin_used = float(margin_used_ctx)
    else:
        capacity = equity * risk_cfg.max_leverage
        margin_used = max(gross_exposure - equity, 0.0)
        margin_available = max(capacity - gross_exposure, 0.0)

    # ----- Leverage & Margin Utilization -----
    if equity > 0:
        leverage = min(gross_exposure / equity, risk_cfg.max_leverage * 2.0)
    else:
        leverage = 0.0

    denom = margin_used + margin_available
    margin_utilization = float(margin_used / denom) if denom > 0 else 0.0

    # ----- PnL יומי -----
    if pnl_today_curve is not None and not np.isnan(pnl_today_curve):
        pnl_today = float(pnl_today_curve)
    else:
        pnl_today = pnl_today_positions

    # ----- VaR + Drawdown -----
    var_1d, var_ratio = _estimate_portfolio_var1d(
        equity_curve=equity_curve,
        equity_current=equity,
        risk_cfg=risk_cfg,
    )
    dd_current, dd_max = _compute_drawdown_from_curve(equity_curve)

    return PortfolioKPI(
        as_of=now,
        equity=equity,
        cash=cash,
        margin_available=margin_available,
        margin_used=margin_used,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        long_exposure=float(long_exposure),
        short_exposure=float(short_exposure),
        leverage=leverage,
        margin_utilization=margin_utilization,
        pnl_unrealized=unrealized,
        pnl_realized=realized,
        pnl_today=pnl_today,
        var_1d=var_1d,
        var_1d_ratio=var_ratio,
        max_drawdown_pct=dd_max,
        current_drawdown_pct=dd_current,
    )


def _estimate_portfolio_var1d(
    equity_curve: MaybeFrame,
    equity_current: float,
    risk_cfg: RiskConfig,
) -> Tuple[Optional[float], Optional[float]]:
    """
    הערכת VaR יומי (1D) בקירוב:

    סדר עדיפויות:
      1. אם יש Equity Curve – VaR היסטורי/קורידור/פרמטרי על תשואות יומיות.
      2. אם אין – fallback גס עם סטיית תקן יומית טיפוסית (~1%).
    """
    if equity_current <= 0:
        return None, None

    # אין curve / מעט מדי נתונים
    if equity_curve is None or equity_curve.empty or equity_curve["equity"].notna().sum() < 10:
        z = _approx_z_score(risk_cfg.var_confidence)
        daily_vol = 0.01
        var_est = max(0.0, float(z * daily_vol * equity_current))
        return var_est, var_est / equity_current

    df = equity_curve.sort_values("timestamp").copy()
    eq = pd.to_numeric(df["equity"], errors="coerce")
    pnl_daily = pd.to_numeric(df["pnl_daily"], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        lag_eq = eq.shift(1)
        returns = pnl_daily / lag_eq
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        z = _approx_z_score(risk_cfg.var_confidence)
        daily_vol = 0.01
        var_est = max(0.0, float(z * daily_vol * equity_current))
        return var_est, var_est / equity_current

    window = risk_cfg.volatility_lookback_days
    if len(returns) > window:
        returns = returns.iloc[-window:]

    conf = risk_cfg.var_confidence
    method = risk_cfg.var_method

    if method == VarMethod.HISTORICAL:
        alpha = 1.0 - conf
        q = np.quantile(returns.values, alpha)
        var_est = max(0.0, float(-q * equity_current))
    elif method == VarMethod.CORRIDOR:
        r = np.sort(returns.values)
        lo = int(len(r) * 0.05)
        hi = int(len(r) * 0.95)
        r_trim = r[lo:hi] if hi > lo else r
        alpha = 1.0 - conf
        q = np.quantile(r_trim, alpha)
        var_est = max(0.0, float(-q * equity_current))
    else:  # PARAMETRIC
        mu = float(returns.mean())
        sigma = float(returns.std(ddof=1))
        z = _approx_z_score(conf)
        var_est = max(0.0, float(z * sigma * equity_current))

    if equity_current <= 0:
        return var_est, None
    return var_est, var_est / equity_current


def _approx_z_score(confidence: float) -> float:
    """
    קירוב מהיר ל-z-score עבור רמות ביטחון טיפוסיות (90%–99%).

    אם confidence לא סטנדרטי – מחזיר ערך סביר לפי טווח.
    """
    mapping = {
        0.90: 1.2816,
        0.95: 1.6449,
        0.975: 1.96,
        0.99: 2.3263,
    }
    if confidence in mapping:
        return mapping[confidence]

    if confidence >= 0.99:
        return 2.33
    if confidence >= 0.975:
        return 1.96
    if confidence >= 0.95:
        return 1.65
    if confidence >= 0.90:
        return 1.28
    return 2.0


def _compute_drawdown_from_curve(
    equity_curve: MaybeFrame,
) -> Tuple[Optional[float], Optional[float]]:
    """
    מחשב Drawdown נוכחי ומקסימלי (current_dd, max_dd) מתוך עקומת Equity.

    אם אין curve או אין מספיק נקודות – מחזיר (None, None).
    """
    if equity_curve is None or equity_curve.empty:
        return None, None

    df = equity_curve.sort_values("timestamp").copy()
    eq = pd.to_numeric(df["equity"], errors="coerce").dropna()
    if eq.empty:
        return None, None

    rolling_max = eq.cummax()
    dd = eq / rolling_max - 1.0

    current_dd = float(dd.iloc[-1])
    max_dd = float(dd.min())
    return current_dd, max_dd


# ------------------------------------------------------------
# Exposure Breakdown — אגרגציית חשיפות לפי רמות שונות
# ------------------------------------------------------------

def compute_exposure_breakdown(
    positions: Frame,
    equity: float,
    level: AggregationLevel,
) -> Frame:
    """
    מחזיר טבלת חשיפה לפי רמת אגרגציה:

      level == PAIR:
          group by pair_id (רמת זוג).
      level == SYMBOL:
          מתפוצץ ללגים ומחשב חשיפה לפי סימבול.
      level == SECTOR:
          מתפוצץ ללגים ומחשב חשיפה לפי סקטור.
      level == STRATEGY:
          group by strategy (רמת זוג).
      level == ASSET_CLASS:
          group by asset_class (רמת זוג).

    הפלט כולל עמודות:
      - bucket: שם הקבוצה (pair/sector/symbol/strategy/asset_class).
      - notional: Σ |exposure|.
      - notional_pct: notional / equity (אם equity>0).
      - long_notional: Σ חשיפות חיוביות.
      - short_notional: Σ חשיפות שליליות (בערך מוחלט).
      - net_notional: long_notional - short_notional.
      - n_positions: מספר פוזיציות פתוחות בקבוצה.
      - n_pairs / n_legs בהתאם לרמת האגרגציה.
    """
    df = positions.copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "bucket",
                "notional",
                "notional_pct",
                "long_notional",
                "short_notional",
                "net_notional",
                "n_positions",
                "n_pairs",
                "n_legs",
            ]
        )

    for col in ("mv_x", "mv_y", "notional_pair"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["mv_long"] = df["mv_x"].clip(lower=0) + df["mv_y"].clip(lower=0)
    df["mv_short"] = (
        df["mv_x"].clip(upper=0).abs() + df["mv_y"].clip(upper=0).abs()
    )
    df["notional_pair"] = df["notional_pair"].fillna(
        df["mv_long"] + df["mv_short"]
    )

    if level == AggregationLevel.PAIR:
        grp = df.groupby("pair_id", dropna=False).agg(
            notional=("notional_pair", "sum"),
            long_notional=("mv_long", "sum"),
            short_notional=("mv_short", "sum"),
            n_positions=("position_id", "nunique"),
        )
        grp["n_pairs"] = grp.index.to_series().nunique()
        grp["n_legs"] = np.nan
        grp.index.name = "bucket"

    elif level in (AggregationLevel.SYMBOL, AggregationLevel.SECTOR):
        legs = _build_leg_level_exposure(df)
        group_col = "symbol" if level == AggregationLevel.SYMBOL else "sector"

        grp = legs.groupby(group_col, dropna=False).agg(
            notional=("notional_leg", "sum"),
            long_notional=("mv_long_leg", "sum"),
            short_notional=("mv_short_leg", "sum"),
            n_positions=("position_id", "nunique"),
            n_pairs=("pair_id", "nunique"),
            n_legs=("symbol", "count"),
        )
        grp.index.name = "bucket"

    elif level == AggregationLevel.STRATEGY:
        grp = df.groupby("strategy", dropna=False).agg(
            notional=("notional_pair", "sum"),
            long_notional=("mv_long", "sum"),
            short_notional=("mv_short", "sum"),
            n_positions=("position_id", "nunique"),
            n_pairs=("pair_id", "nunique"),
        )
        grp["n_legs"] = np.nan
        grp.index.name = "bucket"

    elif level == AggregationLevel.ASSET_CLASS:
        grp = df.groupby("asset_class", dropna=False).agg(
            notional=("notional_pair", "sum"),
            long_notional=("mv_long", "sum"),
            short_notional=("mv_short", "sum"),
            n_positions=("position_id", "nunique"),
            n_pairs=("pair_id", "nunique"),
        )
        grp["n_legs"] = np.nan
        grp.index.name = "bucket"

    else:
        return pd.DataFrame(
            columns=[
                "bucket",
                "notional",
                "notional_pct",
                "long_notional",
                "short_notional",
                "net_notional",
                "n_positions",
                "n_pairs",
                "n_legs",
            ]
        )

    grp["long_notional"] = grp["long_notional"].astype(float)
    grp["short_notional"] = grp["short_notional"].astype(float)
    grp["notional"] = grp["notional"].astype(float)
    grp["net_notional"] = grp["long_notional"] - grp["short_notional"]

    if equity > 0:
        grp["notional_pct"] = grp["notional"] / equity
    else:
        grp["notional_pct"] = np.nan

    grp = grp.reset_index()
    return grp[
        [
            "bucket",
            "notional",
            "notional_pct",
            "long_notional",
            "short_notional",
            "net_notional",
            "n_positions",
            "n_pairs",
            "n_legs",
        ]
    ].sort_values("notional", ascending=False)


def _build_leg_level_exposure(df: Frame) -> Frame:
    """
    מחזיר DataFrame ברמת לג (Leg-Level) לצורך אגרגציה לפי סימבול/סקטור.

    עמודות:
      - position_id, pair_id, strategy, asset_class
      - symbol, sector, industry
      - mv (signed), notional_leg, mv_long_leg, mv_short_leg
    """
    base_cols = ["position_id", "pair_id", "strategy", "asset_class"]

    # leg X
    cols_x = base_cols + ["symbol_x", "sector_x", "industry_x", "mv_x"]
    legs_x = df[cols_x].copy()
    legs_x.rename(
        columns={
            "symbol_x": "symbol",
            "sector_x": "sector",
            "industry_x": "industry",
            "mv_x": "mv",
        },
        inplace=True,
    )

    # leg Y
    cols_y = base_cols + ["symbol_y", "sector_y", "industry_y", "mv_y"]
    legs_y = df[cols_y].copy()
    legs_y.rename(
        columns={
            "symbol_y": "symbol",
            "sector_y": "sector",
            "industry_y": "industry",
            "mv_y": "mv",
        },
        inplace=True,
    )

    legs = pd.concat([legs_x, legs_y], ignore_index=True)

    legs["mv"] = pd.to_numeric(legs["mv"], errors="coerce").fillna(0.0)
    legs["notional_leg"] = legs["mv"].abs()
    legs["mv_long_leg"] = legs["mv"].clip(lower=0)
    legs["mv_short_leg"] = legs["mv"].clip(upper=0).abs()

    return legs


# ------------------------------------------------------------
# Health Checks — דגלים אדומים/צהובים ברמה מקצועית
# ------------------------------------------------------------

def evaluate_portfolio_health(
    kpi: PortfolioKPI,
    positions: Frame,
    exposure_by_pair: Frame,
    exposure_by_sector: Frame,
    config: PortfolioConfig,
) -> List[HealthCheckResult]:
    """
    מפעיל סדרת בדיקות בריאות על הפרוטפוליו:

      - Equity ומינוף.
      - שימוש במרווח (Margin Utilization).
      - VaR ביחס ל-Eq.
      - Drawdown נוכחי ומקסימלי.
      - ריכוזי סיכון בזוג / סקטור.
      - איכות מטא-דאטה (sector/industry).
      - יחס Net/Gross חריג (פוזיציה לא מאוזנת).
    """
    checks: List[HealthCheckResult] = []
    risk_cfg = config.risk

    # 1) Equity חיובי
    if kpi.equity <= 0:
        checks.append(
            HealthCheckResult(
                code="EQUITY_NON_POSITIVE",
                severity=HealthSeverity.CRITICAL,
                message="Equity אינו חיובי – ייתכן חשבון במינוס או בעיית דאטה.",
                details=(
                    f"Equity מחושב: {kpi.equity:.2f}. "
                    "בדוק את מקור נתוני החשבון / Equity Curve."
                ),
            )
        )

    # 2) מינוף
    if kpi.leverage > risk_cfg.max_leverage * 1.5:
        checks.append(
            HealthCheckResult(
                code="LEVERAGE_TOO_HIGH",
                severity=HealthSeverity.CRITICAL,
                message="המינוף גבוה משמעותית מהמותר בהגדרות.",
                details=(
                    f"Leverage={kpi.leverage:.2f}x, "
                    f"Max Allowed={risk_cfg.max_leverage:.2f}x."
                ),
            )
        )
    elif kpi.leverage > risk_cfg.max_leverage:
        checks.append(
            HealthCheckResult(
                code="LEVERAGE_ABOVE_TARGET",
                severity=HealthSeverity.WARNING,
                message="המינוף חוצה את היעד (אך עדיין לא חריג מאוד).",
                details=(
                    f"Leverage={kpi.leverage:.2f}x, "
                    f"Target={risk_cfg.max_leverage:.2f}x."
                ),
            )
        )

    # 3) Margin Utilization
    if kpi.margin_utilization is not None:
        if kpi.margin_utilization > 0.9:
            checks.append(
                HealthCheckResult(
                    code="MARGIN_UTILIZATION_CRITICAL",
                    severity=HealthSeverity.CRITICAL,
                    message="שימוש במרווח מעל 90% – סיכון גבוה ל-Margin Call.",
                    details=f"Margin utilization: {kpi.margin_utilization:.2%}.",
                )
            )
        elif kpi.margin_utilization > risk_cfg.alert_margin_utilization:
            checks.append(
                HealthCheckResult(
                    code="MARGIN_UTILIZATION_HIGH",
                    severity=HealthSeverity.WARNING,
                    message="שימוש במרווח גבוה מסף האזהרה.",
                    details=(
                        f"Margin utilization: {kpi.margin_utilization:.2%}, "
                        f"Alert threshold: {risk_cfg.alert_margin_utilization:.2%}."
                    ),
                )
            )

    # 4) VaR / Equity
    if kpi.var_1d is not None and kpi.var_1d_ratio is not None:
        if kpi.var_1d_ratio > risk_cfg.alert_var_ratio * 2:
            checks.append(
                HealthCheckResult(
                    code="VAR_RATIO_CRITICAL",
                    severity=HealthSeverity.CRITICAL,
                    message="VaR יומי גבוה מאוד ביחס ל-Eq.",
                    details=(
                        f"1D VaR ≈ {kpi.var_1d:,.0f} "
                        f"({kpi.var_1d_ratio:.2%} מה-Eq). "
                        f"סף אזהרה: {risk_cfg.alert_var_ratio:.2%}."
                    ),
                )
            )
        elif kpi.var_1d_ratio > risk_cfg.alert_var_ratio:
            checks.append(
                HealthCheckResult(
                    code="VAR_RATIO_HIGH",
                    severity=HealthSeverity.WARNING,
                    message="VaR יומי מעל סף האזהרה.",
                    details=(
                        f"1D VaR ≈ {kpi.var_1d:,.0f} "
                        f"({kpi.var_1d_ratio:.2%} מה-Eq). "
                        f"סף אזהרה: {risk_cfg.alert_var_ratio:.2%}."
                    ),
                )
            )

    # 5) Drawdown
    if kpi.current_drawdown_pct is not None and kpi.max_drawdown_pct is not None:
        if kpi.current_drawdown_pct < -risk_cfg.alert_drawdown_pct:
            checks.append(
                HealthCheckResult(
                    code="DRAWDOWN_CURRENT_HIGH",
                    severity=HealthSeverity.WARNING,
                    message="הפרוטפוליו נמצא ב-Drawdown גבוה מהסף.",
                    details=(
                        f"Current DD: {kpi.current_drawdown_pct:.2%}, "
                        f"Alert threshold: -{risk_cfg.alert_drawdown_pct:.2%}."
                    ),
                )
            )
        if kpi.max_drawdown_pct < -risk_cfg.alert_drawdown_pct * 2:
            checks.append(
                HealthCheckResult(
                    code="DRAWDOWN_MAX_EXTREME",
                    severity=HealthSeverity.CRITICAL,
                    message="Drawdown מקסימלי היסטורי חריג מאוד.",
                    details=(
                        f"Max historical DD: {kpi.max_drawdown_pct:.2%}, "
                        f"Double alert threshold: -{(risk_cfg.alert_drawdown_pct * 2):.2%}."
                    ),
                )
            )

    # 6) ריכוזי סיכון בזוג בודד
    if not exposure_by_pair.empty and kpi.equity > 0:
        largest_pair = exposure_by_pair.iloc[0]
        pct = float(largest_pair["notional_pct"])
        if pct > risk_cfg.max_single_pair_pct * 1.5:
            checks.append(
                HealthCheckResult(
                    code="PAIR_CONCENTRATION_CRITICAL",
                    severity=HealthSeverity.CRITICAL,
                    message="חשיפה גבוהה מאוד בזוג בודד.",
                    details=(
                        f"Pair '{largest_pair['bucket']}' מהווה "
                        f"{pct:.2%} מה-Eq. "
                        f"סף מוגדר: {risk_cfg.max_single_pair_pct:.2%}."
                    ),
                )
            )
        elif pct > risk_cfg.max_single_pair_pct:
            checks.append(
                HealthCheckResult(
                    code="PAIR_CONCENTRATION_HIGH",
                    severity=HealthSeverity.WARNING,
                    message="ריכוז סיכון גבוה בזוג אחד.",
                    details=(
                        f"Pair '{largest_pair['bucket']}' מהווה "
                        f"{pct:.2%} מה-Eq. "
                        f"סף מוגדר: {risk_cfg.max_single_pair_pct:.2%}."
                    ),
                )
            )

    # 7) ריכוזי סיכון בסקטור בודד
    if not exposure_by_sector.empty and kpi.equity > 0:
        largest_sector = exposure_by_sector.iloc[0]
        pct = float(largest_sector["notional_pct"])
        name = str(largest_sector["bucket"])
        if name.lower() not in ("unknown", ""):
            if pct > risk_cfg.max_sector_pct * 1.5:
                checks.append(
                    HealthCheckResult(
                        code="SECTOR_CONCENTRATION_CRITICAL",
                        severity=HealthSeverity.CRITICAL,
                        message="חשיפה גבוהה מאוד בסקטור בודד.",
                        details=(
                            f"Sector '{name}' מהווה {pct:.2%} מה-Eq. "
                            f"סף מוגדר: {risk_cfg.max_sector_pct:.2%}."
                        ),
                    )
                )
            elif pct > risk_cfg.max_sector_pct:
                checks.append(
                    HealthCheckResult(
                        code="SECTOR_CONCENTRATION_HIGH",
                        severity=HealthSeverity.WARNING,
                        message="ריכוז סיכון גבוה בסקטור אחד.",
                        details=(
                            f"Sector '{name}' מהווה {pct:.2%} מה-Eq. "
                            f"סף מוגדר: {risk_cfg.max_sector_pct:.2%}."
                        ),
                    )
                )

    # 8) איכות מטא-דאטה (sector/industry)
    if not positions.empty:
        sector_x = positions.get("sector_x", pd.Series(dtype=str)).astype(str).str.lower()
        sector_y = positions.get("sector_y", pd.Series(dtype=str)).astype(str).str.lower()
        unknown_x = sector_x.isin(["unknown", "nan", ""])
        unknown_y = sector_y.isin(["unknown", "nan", ""])
        total_legs = len(positions) * 2
        unknown_legs = int(unknown_x.sum() + unknown_y.sum())
        if total_legs > 0:
            frac_unknown = unknown_legs / total_legs
            if frac_unknown > 0.5:
                checks.append(
                    HealthCheckResult(
                        code="METADATA_SECTOR_MISSING",
                        severity=HealthSeverity.INFO,
                        message="חלק גדול מהלגים ללא תיוג sector – מומלץ להשלים מטא דאטה.",
                        details=(
                            f"{frac_unknown:.1%} מהלגים ללא תיוג sector. "
                            "זה לא בהכרח סיכון מיידי, אבל מגביל אנליטיקה וריכוזי סקטור."
                        ),
                    )
                )

    # 9) Net / Gross Ratio – מסחר לא מאוזן
    if kpi.gross_exposure > 0:
        net_gross_ratio = abs(kpi.net_exposure) / kpi.gross_exposure
        if net_gross_ratio > 0.7:
            checks.append(
                HealthCheckResult(
                    code="NET_GROSS_RATIO_HIGH",
                    severity=HealthSeverity.WARNING,
                    message="Net/Gross ratio גבוה – פוזיציה לא מאוזנת יחסית לפרוטפוליו זוגי.",
                    details=(
                        f"Net exposure={kpi.net_exposure:,.0f}, "
                        f"Gross exposure={kpi.gross_exposure:,.0f}, "
                        f"Net/Gross ratio ≈ {net_gross_ratio:.2f}."
                    ),
                )
            )

    return checks
# ============================================================
# Part 4 — Formatting utilities + KPI header + Equity & Exposure UI
# ============================================================

__all__ += [
    "fmt_money",
    "fmt_signed_money",
    "fmt_pct",
    "fmt_signed_pct",
    "fmt_qty",
    "render_kpi_header",
    "render_equity_section",
    "render_exposure_overview",
]

# ------------------------------------------------------------
# Formatting Utilities — ל-UI
# ------------------------------------------------------------

def fmt_money(value: Optional[Number], currency: str = "USD", precision: int = 2) -> str:
    """פורמט כספי נחמד ל-UI."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    try:
        return f"{float(value):,.{precision}f} {currency}"
    except Exception:
        return f"{value} {currency}"


def fmt_signed_money(value: Optional[Number], currency: str = "USD", precision: int = 2) -> str:
    """כמו fmt_money אבל עם סימן +/− – מתאים במיוחד ל-PnL."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    try:
        v = float(value)
        sign = "+" if v > 0 else ""
        return f"{sign}{v:,.{precision}f} {currency}"
    except Exception:
        return f"{value} {currency}"


def fmt_pct(value: Optional[Number], precision: int = 2) -> str:
    """פורמט אחוזים (value בין 0 ל-1)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    try:
        return f"{float(value) * 100:.{precision}f}%"
    except Exception:
        return f"{value}"


def fmt_signed_pct(value: Optional[Number], precision: int = 2) -> str:
    """פורמט אחוזים עם סימן +/− – נוח לתשואות יומיות/Drawdown."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    try:
        v = float(value) * 100
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.{precision}f}%"
    except Exception:
        return f"{value}"


def fmt_qty(value: Optional[Number], precision: int = 0) -> str:
    """פורמט כמות יחידות (Quantity)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    try:
        return f"{float(value):,.{precision}f}"
    except Exception:
        return f"{value}"


# ------------------------------------------------------------
# Engine Status UI helper
# ------------------------------------------------------------

def _render_engine_status_badge(engine_status: Dict[str, Any]) -> None:
    """
    מציג Badge קטן עם סטטוס המנוע:

      - מצב: RUNNING / PAUSED / HALTED / UNKNOWN
      - זמן heartbeat אחרון (אם קיים)
      - מספר זוגות פתוחים / שגיאות בשעה אחרונה (אם קיים)

    הצגה בראש הטאב, מעל ה-KPI.
    """
    mode = str(engine_status.get("mode", "UNKNOWN")).upper()
    last_heartbeat = engine_status.get("last_heartbeat")
    open_pairs = engine_status.get("open_pairs")
    open_positions = engine_status.get("open_positions")
    error_count = engine_status.get("error_count_1h")

    if isinstance(last_heartbeat, (str, datetime)):
        try:
            last_heartbeat_dt = pd.to_datetime(last_heartbeat, errors="coerce", utc=True)
            if pd.isna(last_heartbeat_dt):
                last_heartbeat_str = "לא ידוע"
            else:
                last_heartbeat_str = last_heartbeat_dt.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:  # pragma: no cover
            last_heartbeat_str = str(last_heartbeat)
    else:
        last_heartbeat_str = "לא ידוע"

    if mode == "RUNNING":
        color = "🌱"
        text = "המנוע פועל (RUNNING)"
    elif mode == "PAUSED":
        color = "⏸️"
        text = "המנוע מושהה (PAUSED)"
    elif mode == "HALTED":
        color = "🛑"
        text = "המנוע נעצר (HALTED)"
    else:
        color = "⚪"
        text = "מצב מנוע לא ידוע (UNKNOWN)"

    cols = st.columns([2, 3])
    with cols[0]:
        st.markdown(f"### {color} סטטוס מנוע מסחר חי")
        st.markdown(f"**{text}**")
        st.caption(f"Heartbeat אחרון: {last_heartbeat_str}")
    with cols[1]:
        lines: List[str] = []
        if open_pairs is not None:
            lines.append(f"- זוגות פתוחים: **{open_pairs}**")
        if open_positions is not None:
            lines.append(f"- פוזיציות פתוחות: **{open_positions}**")
        if error_count is not None:
            lines.append(f"- שגיאות בשעה האחרונה: **{error_count}**")
        if lines:
            st.markdown("\n".join(lines))
        else:
            st.caption("עדיין לא הוזנו נתוני מנוע מפורטים (ctx['engine_status']).")

    st.divider()


# ------------------------------------------------------------
# KPI Header UI
# ------------------------------------------------------------

def render_kpi_header(
    analytics: PortfolioAnalytics,
    config: PortfolioConfig,
) -> None:
    """
    מציג שורת KPI ראשית ברמת קרן גידור + סטטוס מנוע:

    - Equity, Cash, PnL יומי.
    - Gross/Net Exposure.
    - Leverage & Margin Utilization.
    - VaR יומי ו-Drawdown.
    """
    kpi = analytics.kpi
    disp = config.display
    ccy = disp.base_currency

    # סטטוס מנוע
    _render_engine_status_badge(analytics.engine_status)

    # שורת KPI ראשונה
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Equity (הון כולל)",
            value=fmt_money(kpi.equity, ccy, disp.precision_money),
            delta=fmt_signed_money(kpi.pnl_today, ccy, disp.precision_money),
            delta_color="normal",
        )
        st.caption(
            "הון כולל נכון לעכשיו (Equity/NLV), ודלתא יומית (PnL היום)."
        )

    with col2:
        st.metric(
            label="Cash (מזומן פנוי)",
            value=fmt_money(kpi.cash, ccy, disp.precision_money),
        )
        st.caption("מזומן זמין מתוך החשבון לשימוש/מרווח.")

    with col3:
        st.metric(
            label="Gross Exposure (חשיפה ברוטו)",
            value=fmt_money(kpi.gross_exposure, ccy, disp.precision_money),
        )
        st.metric(
            label="Net Exposure (נטו)",
            value=fmt_money(kpi.net_exposure, ccy, disp.precision_money),
        )

    with col4:
        lev_str = f"{kpi.leverage:.2f}x" if kpi.leverage is not None else "–"
        mu_str = fmt_pct(kpi.margin_utilization, disp.precision_pct)
        st.metric(
            label="Leverage (מינוף)",
            value=lev_str,
        )
        st.metric(
            label="Margin Utilization (שימוש במרווח)",
            value=mu_str,
        )

    st.divider()

    # שורת KPI שניה – PnL, VaR, Drawdown
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric(
            label="PnL פתוח (Unrealized)",
            value=fmt_signed_money(kpi.pnl_unrealized, ccy, disp.precision_money),
        )
        st.metric(
            label="PnL סגור (Realized מצטבר)",
            value=fmt_signed_money(kpi.pnl_realized, ccy, disp.precision_money),
        )

    with col_b:
        if kpi.var_1d is not None:
            st.metric(
                label="1D VaR (Value-at-Risk)",
                value=fmt_money(kpi.var_1d, ccy, disp.precision_money),
            )
            st.caption(
                f"הערכת VaR יומי ברמת ביטחון {config.risk.var_confidence:.1%}."
            )
        else:
            st.metric("1D VaR (Value-at-Risk)", "–")
            st.caption("לא ניתן לחשב VaR — חסרה עקומת Equity משמעותית.")

    with col_c:
        if kpi.current_drawdown_pct is not None:
            st.metric(
                label="Drawdown נוכחי",
                value=fmt_signed_pct(kpi.current_drawdown_pct, disp.precision_pct),
            )
        else:
            st.metric("Drawdown נוכחי", "–")

        if kpi.max_drawdown_pct is not None:
            st.metric(
                label="Max Drawdown היסטורי",
                value=fmt_signed_pct(kpi.max_drawdown_pct, disp.precision_pct),
            )
        else:
            st.metric("Max Drawdown היסטורי", "–")

    with col_d:
        st.markdown("#### Breakdown מהיר")
        st.write(
            f"- **Long Exposure**: {fmt_money(kpi.long_exposure, ccy, disp.precision_money)}"
        )
        st.write(
            f"- **Short Exposure**: {fmt_money(kpi.short_exposure, ccy, disp.precision_money)}"
        )
        if kpi.equity > 0:
            st.write(
                f"- **Gross/Eq**: {fmt_pct(kpi.gross_exposure / kpi.equity, disp.precision_pct)}"
            )

    st.divider()


# ------------------------------------------------------------
# Equity & PnL Charts
# ------------------------------------------------------------

def render_equity_section(
    equity_curve: MaybeFrame,
    config: PortfolioConfig,
) -> None:
    """
    מציג גרף Equity Curve + Drawdown + PnL יומי.

    אם אין עקומה – מציג הודעה ידידותית.
    """
    if not config.display.show_equity_curve:
        return

    st.subheader("📉 עקומת Equity, Drawdown ו-PnL יומי")

    if equity_curve is None or equity_curve.empty:
        st.info("לא נמצאה עקומת Equity להצגה (אין חיבור ל-DB/Loader).")
        return

    df = equity_curve.copy().sort_values("timestamp")

    # גרף Equity
    fig_eq = go.Figure()
    fig_eq.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["equity"],
            mode="lines",
            name="Equity",
        )
    )
    fig_eq.update_layout(
        margin=dict(l=10, r=10, t=40, b=40),
        height=350,
        xaxis_title="תאריך",
        yaxis_title=f"Equity ({config.display.base_currency})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_eq, width = "stretch")

    col1, col2 = st.columns(2)

    # Drawdown
    with col1:
        dd = df.get("drawdown_pct")
        if dd is not None and dd.notna().any():
            fig_dd = go.Figure()
            fig_dd.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=dd,
                    mode="lines",
                    name="Drawdown %",
                )
            )
            fig_dd.update_layout(
                margin=dict(l=10, r=10, t=30, b=40),
                height=300,
                xaxis_title="תאריך",
                yaxis_title="Drawdown (%)",
            )
            st.plotly_chart(fig_dd, width = "stretch")
        else:
            st.info("אין נתוני Drawdown מחושבים בעקומת ה-Equity.")

    # PnL יומי
    with col2:
        pnl_daily = df.get("pnl_daily")
        if pnl_daily is not None and pnl_daily.notna().any():
            fig_pnl = go.Figure()
            fig_pnl.add_trace(
                go.Bar(
                    x=df["timestamp"],
                    y=pnl_daily,
                    name="PnL יומי",
                )
            )
            fig_pnl.update_layout(
                margin=dict(l=10, r=10, t=30, b=40),
                height=300,
                xaxis_title="תאריך",
                yaxis_title=f"PnL ({config.display.base_currency})",
            )
            st.plotly_chart(fig_pnl, width = "stretch")
        else:
            st.info("אין נתוני PnL יומי זמינים בעקומת ה-Equity.")

    st.divider()


# ------------------------------------------------------------
# Exposure Overview UI
# ------------------------------------------------------------

def render_exposure_overview(
    analytics: PortfolioAnalytics,
    config: PortfolioConfig,
) -> None:
    """
    מציג סקירת חשיפות:

      - טבלאות Exposure לפי זוג/סקטור/סימבול/אסטרטגיה.
      - גרפי Bar לאיתור ריכוזי סיכון.
    """
    st.subheader("📊 חלוקת סיכונים וחשיפות (Live)")

    tabs = st.tabs(
        [
            "זוגות (Pairs)",
            "סקטורים (Sectors)",
            "סימבולים (Symbols)",
            "אסטרטגיות (Strategies)",
        ]
    )

    with tabs[0]:
        _render_single_exposure_view(
            df=analytics.exposure_by_pair,
            title="חשיפה לפי זוג",
            bucket_label="Pair",
            config=config,
        )

    with tabs[1]:
        _render_single_exposure_view(
            df=analytics.exposure_by_sector,
            title="חשיפה לפי סקטור",
            bucket_label="Sector",
            config=config,
        )

    with tabs[2]:
        _render_single_exposure_view(
            df=analytics.exposure_by_symbol,
            title="חשיפה לפי סימבול",
            bucket_label="Symbol",
            config=config,
        )

    with tabs[3]:
        _render_single_exposure_view(
            df=analytics.exposure_by_strategy,
            title="חשיפה לפי אסטרטגיה",
            bucket_label="Strategy",
            config=config,
        )

    st.divider()

def render_risk_parity_engine(
    analytics: PortfolioAnalytics,
    config: PortfolioConfig,
) -> None:
    """
    מנוע Risk-Parity ברמת תיק חי.

    לוגיקה:
    -------
    1. מנסה למצוא מטריצת תשואות ב-st.session_state["returns_wide"].
    2. מאפשר למשתמש להעלות קובץ CSV של תשואות (T x N) במידת הצורך.
    3. מחשב משקולות Risk-Parity באמצעות core.risk_parity.risk_parity_from_returns.
    4. מחשב תרומות סיכון לכל נכס ומציג טבלה מסודרת.
    """
    with st.expander("🛡️ Risk Parity – משקולות לפי סיכון", expanded=False):
        st.caption(
            "מנוע Risk-Parity משתמש בתשואות היסטוריות כדי לחלק משקל כך שכל נכס "
            "יתורם חלק דומה לסיכון הכולל."
        )

        # 1) ניסיון ראשון: לקחת תשואות ממערכת חיה (אם קיימות)
        returns = st.session_state.get("returns_wide")
        if isinstance(returns, pd.DataFrame) and not returns.empty:
            st.caption(
                f"נמצאה מטריצת תשואות מתוך המערכת (returns_wide) "
                f"— צורה: {returns.shape[0]}×{returns.shape[1]}"
            )
        else:
            returns = None

        # 2) אפשרות לטעון קובץ ידני / להחליף דאטה
        up = st.file_uploader(
            "טען מטריצת תשואות (CSV, שורות=ימים, עמודות=נכסים)",
            type=["csv"],
            key="rp_returns_upl",
        )
        if up is not None:
            try:
                df_up = pd.read_csv(up)
                # ננסה לזהות עמודת תאריך ולהפוך ל-index אם קיימת
                for cand in ["date", "Date", "timestamp", "ts"]:
                    if cand in df_up.columns:
                        df_up[cand] = pd.to_datetime(df_up[cand])
                        df_up = df_up.set_index(cand)
                        break
                returns = df_up.select_dtypes(include=["float", "int"])
                st.success(
                    f"נטענו תשואות מקובץ — צורה: {returns.shape[0]}×{returns.shape[1]}"
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"שגיאה בקריאת קובץ התשואות: {e}")
                returns = None

        if returns is None or not isinstance(returns, pd.DataFrame) or returns.empty:
            st.info(
                "אין מטריצת תשואות זמינה לחישוב Risk-Parity.\n\n"
                "- ודא ש-returns_wide נשמר ב-session_state, או\n"
                "- טען קובץ CSV של תשואות בעזרת המעלה למעלה."
            )
            return

        # 3) חישוב משקולות Risk-Parity
        try:
            w = risk_parity_from_returns(returns)
        except Exception as e:  # noqa: BLE001
            st.error(f"Risk-Parity נכשל על מטריצת התשואות: {e}")
            return

        # 4) תרומות סיכון ודוח טבלאי
        try:
            cov = returns.cov()
            rc = compute_risk_contributions(w, cov)
        except Exception as e:  # noqa: BLE001
            rc = None
            st.warning(f"לא הצלחתי לחשב תרומות סיכון מפורטות: {e}")

        df_out = pd.DataFrame(
            {
                "symbol": w.index,
                "w_risk_parity": w.values,
            }
        ).sort_values("w_risk_parity", ascending=False)

        if rc is not None and isinstance(rc, pd.DataFrame):
            # נוודא סדר עמודות לפי w.index
            common_idx = w.index.intersection(rc.index)
            rc_view = rc.loc[common_idx]
            df_out = df_out.set_index("symbol").reindex(common_idx)
            df_out.reset_index(inplace=True)
            if "rc_frac" in rc_view.columns:
                df_out["risk_contribution_pct"] = rc_view["rc_frac"].values
            elif "rc_pct" in rc_view.columns:
                df_out["risk_contribution_pct"] = rc_view["rc_pct"].values

        st.markdown("**משקולות Risk-Parity (ממויין מהגדול לקטן):**")
        st.dataframe(df_out, width = "stretch")

        # הורדה כ-CSV
        st.download_button(
            "⬇️ הורד משקולות Risk-Parity (CSV)",
            df_out.to_csv(index=False).encode("utf-8"),
            file_name="risk_parity_weights.csv",
            mime="text/csv",
            key="rp_dl_csv",
        )

def _render_single_exposure_view(
    df: Frame,
    title: str,
    bucket_label: str,
    config: PortfolioConfig,
) -> None:
    """
    תצוגה אחידה לכל סוגי ה-Exposure:

      - טבלת Exposure ממוינת.
      - גרף בר של % מה-Eq.
    """
    disp = config.display
    ccy = disp.base_currency

    st.markdown(f"#### {title}")

    if df is None or df.empty:
        st.info("אין נתוני חשיפה להצגה.")
        return

    df_view = df.copy()

    # עמודות תצוגה מעוצבות
    df_view["notional_fmt"] = df_view["notional"].apply(
        lambda x: fmt_money(x, ccy, disp.precision_money)
    )
    df_view["notional_pct_fmt"] = df_view["notional_pct"].apply(
        lambda x: fmt_pct(x, disp.precision_pct)
    )
    df_view["long_fmt"] = df_view["long_notional"].apply(
        lambda x: fmt_money(x, ccy, disp.precision_money)
    )
    df_view["short_fmt"] = df_view["short_notional"].apply(
        lambda x: fmt_money(x, ccy, disp.precision_money)
    )

    table_cols = [
        "bucket",
        "notional_fmt",
        "notional_pct_fmt",
        "long_fmt",
        "short_fmt",
        "net_notional",
        "n_positions",
        "n_pairs",
        "n_legs",
    ]
    table_cols = [c for c in table_cols if c in df_view.columns]

    st.dataframe(
        df_view[table_cols].rename(
            columns={
                "bucket": bucket_label,
                "notional_fmt": "חשיפה נומינלית",
                "notional_pct_fmt": "% מה-Eq",
                "long_fmt": "Long Notional",
                "short_fmt": "Short Notional",
                "net_notional": "Net Notional",
                "n_positions": "# Positions",
                "n_pairs": "# Pairs",
                "n_legs": "# Legs",
            }
        ),
        width = "stretch", hide_index=True,
    )

    # גרף בר של % מה-Eq
    top_df = df_view.sort_values("notional", ascending=False).head(20)
    fig = px.bar(
        top_df,
        x="bucket",
        y="notional_pct",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=80),
        height=350,
        xaxis_title=bucket_label,
        yaxis_title="% מה-Eq",
    )
    st.plotly_chart(fig, width = "stretch")
# ============================================================
# Part 5 — Live filters, positions tables & pair drilldown (HF-grade)
# ============================================================

__all__ += [
    "PortfolioFilterState",
    "build_portfolio_filter_state_live",
    "apply_portfolio_filters_live",
    "render_positions_section_live",
]



# ------------------------------------------------------------
# Filter state — מודל לפילטרים של הטאב (לייב)
# ------------------------------------------------------------

@dataclass
class PortfolioFilterState:
    """
    מצב הפילטרים בטאב הפרוטפוליו לייב.

    משפיע על:
      - טבלת הפוזיציות (זוג/לג).
      - Drilldown לזוג.
      - Breakdown חשיפות (באופן עקיף אם תשתמש בפילטרים על positions_filtered).

    ברירת מחדל:
      - הכל פתוח (כל האסטרטגיות/סקטורים/AssetClass).
      - notional מינימלי לפי config.display.min_notional_to_show.
    """

    selected_strategies: List[str]
    selected_sectors: List[str]
    selected_asset_classes: List[str]
    selected_sources: List[str]
    selected_statuses: List[str]

    min_notional: float
    max_holding_days: Optional[float]

    text_query: str
    include_small_positions: bool
    only_live_active_pairs: bool
    only_non_suspended_pairs: bool


def build_portfolio_filter_state_live(
    positions: Frame,
    analytics: PortfolioAnalytics,
    config: PortfolioConfig,
) -> PortfolioFilterState:
    """
    בונה מצב פילטרים באמצעות Streamlit Sidebar לטאב הלייב.

    אם אין פוזיציות – מחזיר סט ברירת מחדל.
    """
    disp = config.display
    min_default = float(disp.min_notional_to_show)

    if positions is None or positions.empty:
        return PortfolioFilterState(
            selected_strategies=[],
            selected_sectors=[],
            selected_asset_classes=[],
            selected_sources=[],
            selected_statuses=[],
            min_notional=min_default,
            max_holding_days=None,
            text_query="",
            include_small_positions=True,
            only_live_active_pairs=False,
            only_non_suspended_pairs=False,
        )

    df = positions.copy()

    strategies = sorted(
        {s for s in df.get("strategy", pd.Series(dtype=str)).astype(str) if s}
    )
    sectors = sorted(
        {
            s
            for s in pd.concat(
                [
                    df.get("sector_x", pd.Series(dtype=str)),
                    df.get("sector_y", pd.Series(dtype=str)),
                ]
            )
            .astype(str)
            .unique()
            if s and s.lower() not in {"unknown", "nan"}
        }
    )
    asset_classes = sorted(
        {s for s in df.get("asset_class", pd.Series(dtype=str)).astype(str) if s}
    )
    sources = sorted(
        {s for s in df.get("source", pd.Series(dtype=str)).astype(str) if s}
    )
    statuses = sorted(
        {s for s in df.get("status", pd.Series(dtype=str)).astype(str) if s}
    )

    with st.sidebar:
        st.markdown("### ⚙️ פילטרים למסחר לייב")

        text_query = st.text_input(
            "חיפוש לפי זוג / סימבול / אסטרטגיה",
            value="",
            placeholder="לדוגמה: QQQ, Tech, XLY_XLC ...",
        )

        col_top1, col_top2 = st.columns(2)
        with col_top1:
            include_small_positions = st.checkbox(
                "להציג גם פוזיציות קטנות מאוד",
                value=False,
                help="אם מסומן, תראה גם פוזיציות עם נוטיונל קטן מהמינימום.",
            )
        with col_top2:
            max_holding_days = st.number_input(
                "סינון לפי מקס' ימי החזקה (אופציונלי)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="0 = ללא סינון לפי ימי החזקה.",
            )
            if max_holding_days <= 0:
                max_holding_days = None

        min_notional = st.slider(
            "מינימום נוטיונל להצגה (USD)",
            min_value=0.0,
            max_value=float(max(min_default * 10, df["notional_pair"].abs().max() or 1_000.0)),
            value=min_default,
            step=max(min_default / 10, 100.0),
        )

        st.markdown("---")

        selected_strategies = st.multiselect(
            "פילטר אסטרטגיות",
            options=strategies,
            default=strategies,
        )
        selected_sectors = st.multiselect(
            "פילטר סקטורים",
            options=sectors,
            default=sectors,
        )
        selected_asset_classes = st.multiselect(
            "פילטר Asset Classes",
            options=asset_classes,
            default=asset_classes,
        )
        selected_sources = st.multiselect(
            "פילטר מקור (live/paper/demo/backtest)",
            options=sources,
            default=sources,
        )
        selected_statuses = st.multiselect(
            "פילטר סטטוס פוזיציה",
            options=statuses,
            default=statuses,
        )

        st.markdown("---")

        only_live_active_pairs = st.checkbox(
            "להציג רק זוגות שמסומנים כ-Active ב-Live Universe",
            value=False,
            help="מסתכל על LivePairProfile דרך LivePairStore (אם קיים).",
        )
        only_non_suspended_pairs = st.checkbox(
            "להסתיר זוגות מסומנים כ-Suspended ב-Live Universe",
            value=False,
        )

    return PortfolioFilterState(
        selected_strategies=selected_strategies,
        selected_sectors=selected_sectors,
        selected_asset_classes=selected_asset_classes,
        selected_sources=selected_sources,
        selected_statuses=selected_statuses,
        min_notional=min_notional,
        max_holding_days=max_holding_days,
        text_query=text_query.strip(),
        include_small_positions=include_small_positions,
        only_live_active_pairs=only_live_active_pairs,
        only_non_suspended_pairs=only_non_suspended_pairs,
    )


def apply_portfolio_filters_live(
    positions: Frame,
    live_profiles_by_pair: Dict[str, "LivePairProfile"],
    filters: PortfolioFilterState,
) -> Frame:
    """
    מיישם את הפילטרים על טבלת הפוזיציות לייב.

    פילטרים:
      - strategy/sector/asset_class/source/status.
      - notional_pair מינימלי.
      - max_holding_days (אם הוגדר).
      - text_query (חיפוש בזוג/סימבולים/אסטרטגיה).
      - only_live_active_pairs / only_non_suspended_pairs לפי LivePairProfile.
    """
    if positions is None or positions.empty:
        return positions

    df = positions.copy()

    # notional_min
    df["notional_pair"] = pd.to_numeric(
        df.get("notional_pair", 0.0), errors="coerce"
    ).fillna(0.0)

    if not filters.include_small_positions and filters.min_notional > 0:
        df = df[df["notional_pair"].abs() >= filters.min_notional]

    # holding_days (אם נרצה לסנן לפי אורך חיים)
    now = pd.Timestamp.utcnow()
    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce", utc=True)
    df["holding_days"] = (now - df["open_time"]).dt.days.astype("float64")
    df["holding_days"] = df["holding_days"].fillna(0.0)

    if filters.max_holding_days is not None and filters.max_holding_days > 0:
        df = df[df["holding_days"] <= filters.max_holding_days]

    # strategy
    if filters.selected_strategies:
        df = df[df["strategy"].isin(filters.selected_strategies)]

    # asset_class
    if filters.selected_asset_classes:
        df = df[df["asset_class"].isin(filters.selected_asset_classes)]

    # source
    if filters.selected_sources:
        df = df[df["source"].isin(filters.selected_sources)]

    # status
    if filters.selected_statuses:
        df = df[df["status"].isin(filters.selected_statuses)]

    # sector (X/Y)
    if filters.selected_sectors:
        sector_x = df.get("sector_x", pd.Series(dtype=str)).astype(str)
        sector_y = df.get("sector_y", pd.Series(dtype=str)).astype(str)
        mask_sector = sector_x.isin(filters.selected_sectors) | sector_y.isin(filters.selected_sectors)
        df = df[mask_sector]

    # פילטרים לפי LivePairProfile
    if live_profiles_by_pair:
        # נוסיף עמודות בוליאניות לניתוח
        df["live_is_active"] = df["pair_id"].map(
            lambda pid: bool(live_profiles_by_pair.get(pid).is_active)
            if pid in live_profiles_by_pair else False
        )
        df["live_is_suspended"] = df["pair_id"].map(
            lambda pid: bool(live_profiles_by_pair.get(pid).is_suspended)
            if pid in live_profiles_by_pair else False
        )
        if filters.only_live_active_pairs:
            df = df[df["live_is_active"]]
        if filters.only_non_suspended_pairs:
            df = df[~df["live_is_suspended"]]
    else:
        df["live_is_active"] = False
        df["live_is_suspended"] = False

    # חיפוש טקסט חופשי
    q = filters.text_query.lower()
    if q:
        def _row_match(row: pd.Series) -> bool:
            s = (
                f"{row.get('pair_id','')}"
                f"{row.get('symbol_x','')}"
                f"{row.get('symbol_y','')}"
                f"{row.get('strategy','')}"
            ).lower()
            return q in s

        df = df[df.apply(_row_match, axis=1)]

    df = df.reset_index(drop=True)
    return df


# ------------------------------------------------------------
# Positions tables & Drilldown UI
# ------------------------------------------------------------

def render_positions_section_live(
    positions_filtered: Frame,
    analytics: PortfolioAnalytics,
    config: PortfolioConfig,
    filters: PortfolioFilterState,
) -> None:
    """
    מציג:
      - טבלת פוזיציות ברמת זוג (עם אינדיקציה LiveProfile).
      - טבלת פוזיציות ברמת לג.
      - Drilldown לזוג נבחר כולל פרופיל לייב (z_entry, sizing, score/edge).
    """
    st.subheader("📋 פוזיציות לייב ו-Drilldown")

    if positions_filtered is None or positions_filtered.empty:
        st.info("אין פוזיציות שעומדות בתנאי הפילטרים הנוכחיים.")
        return

    tab_pairs, tab_legs, tab_drill = st.tabs(
        ["פוזיציות לפי זוג", "פוזיציות לפי לג (Leg)", "Drilldown + Live Profile"]
    )

    with tab_pairs:
        _render_pairs_positions_table_live(
            positions_filtered,
            config,
            analytics.live_profiles_by_pair,
        )

    with tab_legs:
        _render_legs_positions_table_live(positions_filtered, config)

    with tab_drill:
        _render_pair_drilldown_live(
            positions_filtered,
            analytics,
            config,
        )


def _render_pairs_positions_table_live(
    positions: Frame,
    config: PortfolioConfig,
    live_profiles_by_pair: Dict[str, "LivePairProfile"],
) -> None:
    """
    טבלת פוזיציות ברמת זוג (שורה לכל position_id / pair_id):

    מציגה:
      - pair_id, strategy, asset_class, status.
      - notional, unrealized, realized, pnl_today.
      - spread, spread_z, holding_days.
      - live_is_active, live_is_suspended, score_total, ml_edge, regime.
    """
    disp = config.display
    ccy = disp.base_currency

    df = positions.copy()
    now = pd.Timestamp.utcnow()

    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce", utc=True)
    df["holding_days"] = (now - df["open_time"]).dt.days.astype("float64")
    df["holding_days"] = df["holding_days"].fillna(0.0)

    # enrich from LivePairProfile
    df["live_is_active"] = False
    df["live_is_suspended"] = False
    df["live_score"] = np.nan
    df["live_ml_edge"] = np.nan
    df["live_regime"] = ""
    df["live_model_version"] = ""

    for idx, row in df.iterrows():
        pid = row.get("pair_id", "")
        lp = live_profiles_by_pair.get(pid)
        if lp is None:
            continue
        try:
            df.at[idx, "live_is_active"] = bool(lp.is_active)
            df.at[idx, "live_is_suspended"] = bool(lp.is_suspended)
            df.at[idx, "live_score"] = float(lp.score_total)
            df.at[idx, "live_ml_edge"] = float(lp.ml_edge_score) if lp.ml_edge_score is not None else np.nan
            df.at[idx, "live_regime"] = str(lp.regime_id or "")
            df.at[idx, "live_model_version"] = str(lp.model_version or "")
        except Exception:  # pragma: no cover
            continue

    view = pd.DataFrame(
        {
            "Pair": df["pair_id"],
            "Live Active": df["live_is_active"],
            "Suspended": df["live_is_suspended"],
            "Score": df["live_score"],
            "ML Edge": df["live_ml_edge"],
            "Regime": df["live_regime"],
            "Strategy": df["strategy"],
            "Asset Class": df["asset_class"],
            "Status": df["status"],
            "Notional": df["notional_pair"],
            "Unrealized PnL": df["unrealized_pnl"],
            "Realized PnL": df["realized_pnl"],
            "PnL Today": df["pnl_today"],
            "Spread": df["spread"],
            "Z-Score": df["spread_z"],
            "Holding (days)": df["holding_days"],
        }
    )

    # פורמט כספי
    for col in ("Notional", "Unrealized PnL", "Realized PnL", "PnL Today"):
        view[col] = view[col].apply(
            lambda x: fmt_signed_money(x, ccy, disp.precision_money)
        )

    # פורמט ספרד / Z
    view["Spread"] = view["Spread"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )
    view["Z-Score"] = view["Z-Score"].apply(
        lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )
    view["Holding (days)"] = view["Holding (days)"].apply(
        lambda x: f"{x:.0f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )

    # פורמט Score/ML edge
    view["Score"] = view["Score"].apply(
        lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )
    view["ML Edge"] = view["ML Edge"].apply(
        lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )

    st.markdown("#### טבלת פוזיציות לפי זוג (עם Live Profile Overlay)")
    st.dataframe(
        view.sort_values(["Live Active", "Suspended", "Pair"], ascending=[False, True, True]),
        width = "stretch", hide_index=True,
    )


def _render_legs_positions_table_live(
    positions: Frame,
    config: PortfolioConfig,
) -> None:
    """
    טבלת פוזיציות ברמת לג (Leg-Level):

    - כל שורה היא לג (symbol_x / symbol_y).
    - מציג: Pair, Symbol, Sector, Strategy, Notional Leg, Long/Short Leg.
    """
    disp = config.display
    ccy = disp.base_currency

    df = positions.copy()
    legs = _build_leg_level_exposure(df)

    view = pd.DataFrame(
        {
            "Pair": legs["pair_id"],
            "Symbol": legs["symbol"],
            "Sector": legs["sector"],
            "Industry": legs["industry"],
            "Asset Class": legs["asset_class"],
            "Strategy": legs["strategy"],
            "Notional Leg": legs["notional_leg"],
            "Long Leg": legs["mv_long_leg"],
            "Short Leg": legs["mv_short_leg"],
        }
    )

    view["Notional Leg"] = view["Notional Leg"].apply(
        lambda x: fmt_money(x, ccy, disp.precision_money)
    )
    view["Long Leg"] = view["Long Leg"].apply(
        lambda x: fmt_money(x, ccy, disp.precision_money)
    )
    view["Short Leg"] = view["Short Leg"].apply(
        lambda x: fmt_money(x, ccy, disp.precision_money)
    )

    st.markdown("#### טבלת פוזיציות ברמת לג (Leg-Level)")
    st.dataframe(
        view.sort_values(["Pair", "Symbol"]),
        width = "stretch", hide_index=True,
    )


def _render_pair_drilldown_live(
    positions_filtered: Frame,
    analytics: PortfolioAnalytics,
    config: PortfolioConfig,
) -> None:
    """
    Drilldown לזוג בודד ברמת לייב:

      - בחירת pair_id.
      - חשיפה יחסית מה-Eq.
      - Spread/Z ממוצע, Holding Period.
      - הצגת פרופיל לייב מתוך LivePairProfile (z_entry/z_exit, sizing, score/edge/regime).
      - טבלת פוזיציות מפורטת לזוג.
    """
    disp = config.display
    ccy = disp.base_currency

    unique_pairs = sorted(
        {p for p in positions_filtered.get("pair_id", pd.Series(dtype=str)).astype(str) if p}
    )
    if not unique_pairs:
        st.info("אין זוגות לביצוע Drilldown.")
        return

    st.markdown("#### 🔍 Drilldown לזוג + פרופיל לייב")

    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        selected_pair = st.selectbox(
            "בחר זוג לניתוח:",
            options=unique_pairs,
            index=0,
        )

    df_pair = positions_filtered[positions_filtered["pair_id"] == selected_pair].copy()
    if df_pair.empty:
        with col_info:
            st.info("לא נמצאו פוזיציות פעילות לזוג שנבחר.")
        return

    notional = float(
        pd.to_numeric(df_pair.get("notional_pair", 0.0), errors="coerce").sum()
    )
    eq = analytics.kpi.equity
    pct_eq = notional / eq if eq > 0 else np.nan

    spread = float(
        pd.to_numeric(df_pair.get("spread", np.nan), errors="coerce").mean()
    )
    zscore = float(
        pd.to_numeric(df_pair.get("spread_z", np.nan), errors="coerce").mean()
    )

    now = pd.Timestamp.utcnow()
    df_pair["open_time"] = pd.to_datetime(df_pair["open_time"], errors="coerce", utc=True)
    df_pair["holding_days"] = (now - df_pair["open_time"]).dt.days.astype("float64")
    holding_mean = float(df_pair["holding_days"].mean())

    # שליפת LivePairProfile אם קיים
    lp = analytics.live_profiles_by_pair.get(selected_pair)

    with col_info:
        st.metric(
            label="חשיפה לזוג (Notional)",
            value=fmt_money(notional, ccy, disp.precision_money),
            delta=fmt_pct(pct_eq, disp.precision_pct) if eq > 0 else "–",
            delta_color="off",
        )
        st.write(
            f"- ממוצע Z-Score: **{zscore:.2f}**" if not np.isnan(zscore) else "- אין נתון Z-Score."
        )
        st.write(
            f"- ממוצע Spread: **{spread:.4f}**" if not np.isnan(spread) else "- אין נתון Spread."
        )
        st.write(
            f"- ממוצע תקופת החזקה: **{holding_mean:.1f} ימים**"
            if not np.isnan(holding_mean)
            else "- אין נתוני תקופת החזקה."
        )

        if lp is not None:
            st.markdown("##### פרופיל לייב לזוג (LivePairProfile)")
            cols_lp = st.columns(3)
            with cols_lp[0]:
                st.write(f"- **Active:** {lp.is_active}")
                st.write(f"- **Suspended:** {lp.is_suspended}")
                st.write(f"- **Score Total:** {lp.score_total:.3f}")
                if lp.ml_edge_score is not None:
                    st.write(f"- **ML Edge:** {lp.ml_edge_score:.3f}")
                if lp.ml_confidence is not None:
                    st.write(f"- **ML Conf.:** {lp.ml_confidence:.2f}")

            with cols_lp[1]:
                st.write(f"- **z_entry:** {lp.z_entry}")
                st.write(f"- **z_exit:** {lp.z_exit}")
                if lp.z_take_profit is not None:
                    st.write(f"- **z_take_profit:** {lp.z_take_profit}")
                if lp.z_hard_stop is not None:
                    st.write(f"- **z_hard_stop:** {lp.z_hard_stop}")
                st.write(f"- **sizing_mode:** {lp.sizing_mode}")
                st.write(f"- **base_notional_usd:** {lp.base_notional_usd:,.0f}")

            with cols_lp[2]:
                if lp.regime_id:
                    st.write(f"- **Regime:** {lp.regime_id}")
                if lp.macro_regime_id:
                    st.write(f"- **Macro Regime:** {lp.macro_regime_id}")
                if lp.model_version:
                    st.write(f"- **Model Version:** {lp.model_version}")
                if lp.last_optimized_at:
                    st.write(f"- **Last Opt:** {lp.last_optimized_at}")
                if lp.last_ml_update_at:
                    st.write(f"- **Last ML Update:** {lp.last_ml_update_at}")
        else:
            st.info("לזוג זה אין עדיין פרופיל לייב ב-Live Universe (LivePairStore).")

    # טבלת פירוט לפוזיציות בזוג
    st.markdown("##### פירוט פוזיציות לזוג שנבחר")

    view = pd.DataFrame(
        {
            "Position ID": df_pair["position_id"],
            "Symbol X": df_pair["symbol_x"],
            "Symbol Y": df_pair["symbol_y"],
            "Qty X": df_pair["qty_x"],
            "Qty Y": df_pair["qty_y"],
            "Side X": df_pair["side_x"],
            "Side Y": df_pair["side_y"],
            "Entry X": df_pair["entry_price_x"],
            "Entry Y": df_pair["entry_price_y"],
            "Last X": df_pair["last_price_x"],
            "Last Y": df_pair["last_price_y"],
            "Notional": df_pair["notional_pair"],
            "Unrealized PnL": df_pair["unrealized_pnl"],
            "Realized PnL": df_pair["realized_pnl"],
            "PnL Today": df_pair["pnl_today"],
            "Holding (days)": df_pair["holding_days"],
        }
    )

    for col in ("Notional", "Unrealized PnL", "Realized PnL", "PnL Today"):
        view[col] = view[col].apply(
            lambda x: fmt_signed_money(x, ccy, disp.precision_money)
        )
    for col in ("Entry X", "Entry Y", "Last X", "Last Y"):
        view[col] = view[col].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
        )
    view["Holding (days)"] = view["Holding (days)"].apply(
        lambda x: f"{x:.0f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )

    st.dataframe(
        view,
        width = "stretch", hide_index=True,
    )
# ============================================================
# Part 6 — Health checks UI & Action Hints (Live, extended HF-grade)
# ============================================================

__all__ += [
    "render_health_section_live",
    "render_action_hints_live",
]

# ------------------------------------------------------------
# Health Section UI (מורחב)
# ------------------------------------------------------------

def render_health_section_live(
    analytics: PortfolioAnalytics,
    config: PortfolioConfig,
) -> None:
    """
    מציג פאנל Health Checks ברמת קרן גידור, עם חלוקה לקטגוריות:

      - Account & Leverage (Equity, מינוף, מרווח, Drawdown).
      - Risk Metrics (VaR, Net/Gross).
      - Concentration (זוג/סקטור).
      - Data Quality (מטא-דאטה חסר וכו').

    כל בדיקה מתקבלת מ-evaluate_portfolio_health, עם:
      - code (זיהוי קצר),
      - severity (INFO/WARNING/CRITICAL),
      - message קצר,
      - details להסבר מעמיק.

    בנוסף:
      - Summary בראש (counts לפי חומרה).
      - הפרדה ויזואלית בין קטגוריות.
    """
    if not config.display.show_diagnostics_section:
        return

    st.subheader("🩺 בריאות הפרוטפוליו (Live Health Checks)")

    checks = analytics.health_checks or []
    if not checks:
        st.success("לא זוהו בעיות ברמת סיכון כללית לפי ההגדרות הנוכחיות.")
        st.caption(
            "המשמעות: לפי ה-KPI, ה-VaR, ה-Drawdown וריכוזי הסיכון שהוגדרו ב-RiskConfig "
            "הפרוטפוליו בתוך טווחי היעד. "
            "עדיין מומלץ לעקוב אחרי הזוגות והסקטורים הדומיננטיים."
        )
        return

    # ---- 1) סטטיסטיקה כללית ----
    severity_order = {
        HealthSeverity.CRITICAL: 0,
        HealthSeverity.WARNING: 1,
        HealthSeverity.INFO: 2,
    }
    checks_sorted = sorted(
        checks,
        key=lambda hc: severity_order.get(hc.severity, 99),
    )

    n_crit = sum(1 for c in checks_sorted if c.severity == HealthSeverity.CRITICAL)
    n_warn = sum(1 for c in checks_sorted if c.severity == HealthSeverity.WARNING)
    n_info = sum(1 for c in checks_sorted if c.severity == HealthSeverity.INFO)

    summary_msg = f"CRITICAL: {n_crit} | WARNING: {n_warn} | INFO: {n_info}"

    if n_crit > 0:
        st.error(f"סיכום מצב בריאות: {summary_msg}")
    elif n_warn > 0:
        st.warning(f"סיכום מצב בריאות: {summary_msg}")
    else:
        st.info(f"סיכום מצב בריאות: {summary_msg}")

    st.caption(
        "הערה: הבדיקות אינן הוראה למסחר, אלא אינדיקציות ניהוליות "
        "המבוססות על RiskConfig ועל נתוני הפרוטפוליו."
    )

    # ---- 2) חלוקה לקטגוריות לוגיות לפי code ----
    def _bucket(hc: HealthCheckResult) -> str:
        code = hc.code.upper()
        if any(key in code for key in ("EQUITY", "LEVERAGE", "MARGIN", "DRAWDOWN")):
            return "Account & Leverage"
        if any(key in code for key in ("VAR_", "NET_GROSS")):
            return "Risk Metrics (VaR & Net/Gross)"
        if any(key in code for key in ("PAIR_CONCENTRATION", "SECTOR_CONCENTRATION")):
            return "Concentration (Pair / Sector)"
        if "METADATA" in code:
            return "Data Quality"
        return "Other"

    buckets: Dict[str, List[HealthCheckResult]] = {}
    for hc in checks_sorted:
        b = _bucket(hc)
        buckets.setdefault(b, []).append(hc)

    # ---- 3) הצגה לפי קטגוריות ----
    for bucket_name, hc_list in buckets.items():
        st.markdown(f"### {bucket_name}")

        for hc in hc_list:
            label = f"[{hc.code}] {hc.message}"

            if hc.severity == HealthSeverity.CRITICAL:
                box = st.error
                icon = "🛑"
            elif hc.severity == HealthSeverity.WARNING:
                box = st.warning
                icon = "⚠️"
            else:
                box = st.info
                icon = "ℹ️"

            box(f"{icon} {label}")
            if hc.details:
                with st.expander("לפרטים נוספים", expanded=False):
                    st.write(hc.details)

        st.markdown("---")

    st.divider()


# ------------------------------------------------------------
# Action Hints — הצעות פעולה ניהוליות (מורחבות)
# ------------------------------------------------------------

def render_action_hints_live(
    analytics: PortfolioAnalytics,
    config: PortfolioConfig,
) -> None:
    """
    מייצר ומציג "Hints" ברמת מאקרו על בסיס:

      - KPI (Equity, Leverage, Margin, VaR, Drawdown).
      - חשיפות (Pairs/Sectors/Strategies).
      - סטטוס מנוע (RUNNING/PAUSED/HALTED).
      - Live Universe (כמה זוגות Active/Suspended, האם ה-ML/Opt "מאכילים" את הלייב).

    המטרה:
      - לתת למנהל הקרן / הסוחר "סיפור" תמציתי:
          * מה המצב?
          * איפה הכי כדאי להסתכל?
          * אילו צעדים פוטנציאליים אפשר לשקול?

    ההמלצות כאן נשארות ברמת טקסט – אין ביצוע בפועל של פעולות.
    """
    if not config.display.show_actions_section:
        return

    kpi = analytics.kpi
    risk_cfg = config.risk
    ccy = config.display.base_currency

    st.subheader("💡 הצעות פעולה ניהוליות (Action Hints)")

    hints: List[str] = []

    # ===== 1. סטטוס מנוע =====
    mode = str(analytics.engine_status.get("mode", "UNKNOWN")).upper()
    last_heartbeat = analytics.engine_status.get("last_heartbeat")

    if mode == "HALTED":
        hints.append(
            "- המנוע נמצא במצב **HALTED**. לפני חידוש המסחר מומלץ:\n"
            "  • לבדוק מה גרם לעצירה (Kill-Switch, בעיית חיבור, חריגת סיכון).\n"
            "  • לוודא שהגורם טופל (למשל Drawdown ירד מתחת סף, החיבור ל-IBKR יציב).\n"
            "  • לעבור על הדגלים הקריטיים בלשונית Health ולהחליט אם לפתוח מחדש."
        )
    elif mode == "PAUSED":
        hints.append(
            "- המנוע ב-**PAUSED**. זה זמן טוב לעשות \"review\":\n"
            "  • לסרוק את הזוגות הגדולים ביותר מבחינת Notional.\n"
            "  • לבדוק אם כולם עדיין עומדים בתנאי ה-Regime/Score/Edge.\n"
            "  • אולי להקפיא זמנית זוגות רועשים לפני חזרה ל-RUNNING."
        )
    elif mode == "RUNNING":
        hints.append(
            "- המנוע **פועל (RUNNING)**. מומלץ לוודא:\n"
            "  • שיש ניטור אקטיבי (דשבורד פתוח, לוגים/התראות).\n"
            "  • שהסף ב-Kill-Switch (Max Daily Loss / Drawdown) עדיין רלוונטי.\n"
            "  • שבאירועי מאקרו גדולים (FOMC, CPI וכו') יש מדיניות ברורה (להאט/להפסיק פתיחות חדשות)."
        )
    else:
        hints.append(
            "- סטטוס המנוע אינו מוגדר (mode=UNKNOWN). מומלץ:\n"
            "  • להעביר מ-engine את ctx['engine_status'] לטאב.\n"
            "  • לוודא שיש Heartbeat תקין וניטור לשגיאות/ניתוקים."
        )

    # ===== 2. מינוף ומרווח =====
    if kpi.leverage > risk_cfg.max_leverage * 1.5:
        hints.append(
            f"- המינוף ({kpi.leverage:.2f}x) גבוה משמעותית מהמקסימום המוגדר ({risk_cfg.max_leverage:.2f}x).\n"
            "  • שקול לסגור חלקית פוזיציות בזוגות עם notional גבוה.\n"
            "  • בדוק אם חלק מהזוגות הללו כבר קיבלו Score/Edge נמוך ב-Live Universe.\n"
            "  • ייתכן שכדאי להחזיר את המינוף לטווח היעד לפני פתיחת עסקאות חדשות."
        )
    elif kpi.leverage > risk_cfg.max_leverage:
        hints.append(
            f"- המינוף ({kpi.leverage:.2f}x) מעט מעל היעד ({risk_cfg.max_leverage:.2f}x).\n"
            "  • היום זה יכול להיות בסדר, אבל לטווח ארוך מומלץ להחזיר לאזור היעד ע\"י התאמות עדינות.\n"
        )

    if kpi.margin_utilization is not None and kpi.margin_utilization > risk_cfg.alert_margin_utilization:
        hints.append(
            f"- שימוש במרווח ({fmt_pct(kpi.margin_utilization, 2)}) גבוה מסף האזהרה "
            f"({fmt_pct(risk_cfg.alert_margin_utilization, 2)}).\n"
            "  • זוהי סביבה שבה Margin Call אפשרי בענפי זעזוע.\n"
            "  • שקול להקטין פוזיציות עתירות מרווח (למשל, מניות יקרות/חוזים עתידיים כבדים)."
        )

    # ===== 3. VaR & Drawdown =====
    if kpi.var_1d_ratio is not None and kpi.var_1d_ratio > risk_cfg.alert_var_ratio:
        hints.append(
            f"- 1D VaR ≈ {fmt_money(kpi.var_1d, ccy, config.display.precision_money)} "
            f"({fmt_pct(kpi.var_1d_ratio, 2)} מה-Eq) מעל סף האזהרה "
            f"({fmt_pct(risk_cfg.alert_var_ratio, 2)}).\n"
            "  • בדוק אילו זוגות תורמים הכי הרבה ל-VaR (זוגות עם נוטיונל גבוה ותנודתיות גבוהה).\n"
            "  • אפשר לצמצם סיכון ע\"י הורדה מדורגת של חלק מה exposure בזוגות אלו."
        )

    if kpi.current_drawdown_pct is not None and kpi.current_drawdown_pct < -risk_cfg.alert_drawdown_pct:
        hints.append(
            f"- הפרוטפוליו נמצא ב-Drawdown של {fmt_signed_pct(kpi.current_drawdown_pct, 2)}, "
            "שגבוה מסף האזהרה.\n"
            "  • בתקופות כאלה נהוג להאט פתיחת פוזיציות חדשות.\n"
            "  • להתמקד בזוגות עם half-life קצר יותר ו-Score/Edge גבוה.\n"
            "  • אפשר לשקול להפעיל 'soft Kill-Switch' לפתיחות חדשות עד שיפור מצב."
        )

    # ===== 4. Live Universe Integration =====
    live_pairs = analytics.live_profiles_by_pair
    if live_pairs:
        n_total = len(live_pairs)
        n_active = sum(1 for p in live_pairs.values() if p.is_active and not p.is_suspended)
        n_suspended = sum(1 for p in live_pairs.values() if p.is_suspended)
        avg_score = np.mean([p.score_total for p in live_pairs.values()]) if live_pairs else None

        hints.append(
            f"- Live Universe כולל {n_total} זוגות, "
            f"{n_active} Active, {n_suspended} Suspended."
        )
        if avg_score is not None and not np.isnan(avg_score):
            hints.append(
                f"- ממוצע Score ב-Live Universe הוא בערך {avg_score:.3f}. "
                "אם זה נמוך, אולי שווה להריץ מחדש אופטימיזציה/ML על universe עדכני."
            )

        if n_active == 0:
            hints.append(
                "- כרגע אין זוגות Active ב-Live Universe.\n"
                "  • אם המנוע אמור לפתוח עסקאות, בדוק האם הספים ב-LivePairStore (score/edge) גבוהים מדי.\n"
                "  • ייתכן שצריך להריץ מחדש 'export to live' מטאב האופטימיזציה."
            )
        elif n_suspended > 0:
            hints.append(
                f"- ישנם {n_suspended} זוגות מסומנים כ-Suspended.\n"
                "  • חשוב לוודא שכל Suspension מתועד (earnings, macro event, בעיית דאטה).\n"
                "  • אחרי שהאירוע עבר, בדוק האם אפשר להחזיר חלק מהזוגות ל-Active."
            )
    else:
        hints.append(
            "- כרגע Live Universe לא נטען (אין LivePairStore/Profiles). "
            "מומלץ:\n"
            "  • לחבר את טאב האופטימיזציה כך שיכתוב LivePairProfile ל-DB.\n"
            "  • לתת לטאב הזה לצרוך את הפרופילים ולהראות Score/Edge/Regime לכל זוג."
        )

    # ===== 5. Net/Gross Ratio =====
    if kpi.gross_exposure > 0:
        net_gross_ratio = abs(kpi.net_exposure) / kpi.gross_exposure
        if net_gross_ratio > 0.7:
            hints.append(
                f"- Net/Gross ratio ≈ {net_gross_ratio:.2f} – הפרוטפוליו לא מאוזן.\n"
                "  • למסחר זוגי, נהוג לשמור על Net נמוך יחסית (סביב 0) כדי להיות נייטרלי לשוק.\n"
                "  • שקול לאזן מחדש את החשיפות (למשל, להקטין חלק מהזוגות שנותנים נטו חזק לשוק)."
            )

    # ===== 6. סיכום חכם =====
    if not hints:
        st.success("כרגע לא זוהו נקודות המחייבות פעולה מיוחדת ברמת מאקרו.")
        st.caption(
            "כמובן שתמיד כדאי לעקוב אחרי הזוגות והסקטורים הדומיננטיים, "
            "אבל לפי RiskConfig והמצב הנוכחי – אין דגלים חריגים."
        )
        return

    for h in hints:
        st.write(h)
# ============================================================
# Part 7 — Live Universe (LivePairProfile) — Full View & Analytics
# ============================================================

__all__ += [
    "render_live_universe_section",
]

from dataclasses import dataclass

# ------------------------------------------------------------
# Live Universe Filters — מודל מצב פילטרים ל-LivePairProfile
# ------------------------------------------------------------

@dataclass
class LiveUniverseFilterState:
    """
    מצב הפילטרים ל-Live Universe:

    משפיע על הטבלה/גרפים של ה-LivePairProfile:
      - Active/Suspended.
      - AssetClass / Timeframe / Cluster.
      - Regime / Macro Regime / Model Version.
      - Score / ML Edge / Vol Target.
      - חיפוש טקסט חופשי (pair_id/symbols/strategy/tag).
    """

    show_only_active: bool
    hide_suspended: bool

    selected_asset_classes: List[str]
    selected_timeframes: List[str]
    selected_regimes: List[str]
    selected_macro_regimes: List[str]
    selected_model_versions: List[str]

    min_score_total: float
    min_ml_edge: Optional[float]

    text_query: str


def _load_live_universe_full(
    config: PortfolioConfig,
) -> Dict[str, "LivePairProfile"]:
    """
    טוען את כל ה-Live Universe (כל הזוגות) מתוך LivePairStore.

    שונה מ-load_live_universe_for_portfolio (שמסנן ל-Engine):
      - כאן **לא מסננים** לפי min_score/min_edge/priority.
      - הכוונה: טאב Universe צריך לראות *הכל*, גם זוגות חלשים/כבויים.

    אם:
      - אין קובץ DB,
      - או אין LivePairStore,
      מחזיר dict ריק.
    """
    try:
        from common.live_pair_store import LivePairStore  # type: ignore
        from common.live_profiles import LivePairProfile  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.info(
            "LivePairStore/LivePairProfile not available — cannot load full live universe (%s).",
            exc,
        )
        return {}

    db_path = Path(config.live.live_pairs_db_path)
    if not db_path.exists():
        logger.info("Live pairs DB not found at %s — Live Universe tab will be empty.", db_path)
        return {}

    try:
        store = LivePairStore(
            db_path=config.live.live_pairs_db_path,
            table_name=config.live.live_pairs_table,
        )
    except Exception as exc:
        logger.warning("Failed to open LivePairStore (full universe): %s", exc)
        return {}

    try:
        profiles = store.load_all(order_by="pair_id")
        result: Dict[str, LivePairProfile] = {}
        for p in profiles:
            try:
                result[p.pair_id] = p
            except Exception:  # pragma: no cover
                logger.debug("Skipping malformed LivePairProfile from store")
        return result
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load full live universe from store: %s", exc)
        return {}
    finally:
        try:
            store.close()
        except Exception:
            pass


def _build_live_universe_df(
    live_profiles_by_pair: Dict[str, "LivePairProfile"],
) -> Frame:
    """
    ממיר dict של LivePairProfile → DataFrame נוח לתצוגה/פילטר/גרפים.

    כל שורה = זוג אחד ב-Live Universe.
    """
    if not live_profiles_by_pair:
        return pd.DataFrame(
            columns=[
                "pair_id",
                "sym_x",
                "sym_y",
                "asset_class",
                "base_currency",
                "timeframe",
                "cluster_id",
                "is_active",
                "is_suspended",
                "score_total",
                "ml_edge_score",
                "ml_confidence",
                "regime_id",
                "macro_regime_id",
                "vol_target_annual",
                "sizing_mode",
                "base_notional_usd",
                "risk_budget_fraction",
                "half_life_bars",
                "hurst_exponent",
                "cointegration_pvalue",
                "adf_pvalue_spread",
                "backtest_sharpe",
                "backtest_max_drawdown",
                "backtest_trades_count",
                "model_version",
                "last_optimized_at",
                "last_backtest_at",
                "last_ml_update_at",
                "tags_str",
            ]
        )

    records: List[Dict[str, Any]] = []
    for p in live_profiles_by_pair.values():
        try:
            rec: Dict[str, Any] = {}
            rec["pair_id"] = p.pair_id
            rec["sym_x"] = p.sym_x
            rec["sym_y"] = p.sym_y
            rec["asset_class"] = p.asset_class
            rec["base_currency"] = p.base_currency
            rec["timeframe"] = p.timeframe
            rec["cluster_id"] = getattr(p, "cluster_id", None)

            rec["is_active"] = p.is_active
            rec["is_suspended"] = p.is_suspended
            rec["score_total"] = p.score_total
            rec["ml_edge_score"] = p.ml_edge_score
            rec["ml_confidence"] = p.ml_confidence

            rec["regime_id"] = p.regime_id
            rec["macro_regime_id"] = p.macro_regime_id
            rec["vol_target_annual"] = p.vol_target_annual
            rec["sizing_mode"] = p.sizing_mode
            rec["base_notional_usd"] = p.base_notional_usd
            rec["risk_budget_fraction"] = getattr(p, "risk_budget_fraction", None)

            rec["half_life_bars"] = p.half_life_bars
            rec["hurst_exponent"] = p.hurst_exponent
            rec["cointegration_pvalue"] = p.cointegration_pvalue
            rec["adf_pvalue_spread"] = p.adf_pvalue_spread

            rec["backtest_sharpe"] = p.backtest_sharpe
            rec["backtest_max_drawdown"] = p.backtest_max_drawdown
            rec["backtest_trades_count"] = p.backtest_trades_count

            rec["model_version"] = p.model_version
            rec["last_optimized_at"] = p.last_optimized_at
            rec["last_backtest_at"] = p.last_backtest_at
            rec["last_ml_update_at"] = p.last_ml_update_at

            tags = getattr(p, "tags", [])
            if isinstance(tags, (list, tuple)):
                rec["tags_str"] = ", ".join(str(t) for t in tags)
            else:
                rec["tags_str"] = ""

            records.append(rec)
        except Exception:  # pragma: no cover
            logger.debug("Failed to convert LivePairProfile '%s' to row", p.pair_id)

    df = pd.DataFrame.from_records(records)
    # טיפוסים נוחים
    for col in ("score_total", "ml_edge_score", "ml_confidence",
                "vol_target_annual", "base_notional_usd", "risk_budget_fraction",
                "half_life_bars", "hurst_exponent",
                "cointegration_pvalue", "adf_pvalue_spread",
                "backtest_sharpe", "backtest_max_drawdown"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("last_optimized_at", "last_backtest_at", "last_ml_update_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    return df


def _build_live_universe_filter_state(
    df_universe: Frame,
) -> LiveUniverseFilterState:
    """
    בונה מצב פילטרים ל-Live Universe מתוך DataFrame.

    הפילטרים מוצגים בתוך הסקשן של Live Universe (לא בסיידבר).
    """
    if df_universe.empty:
        # ברירת מחדל "ריקה"
        return LiveUniverseFilterState(
            show_only_active=False,
            hide_suspended=False,
            selected_asset_classes=[],
            selected_timeframes=[],
            selected_regimes=[],
            selected_macro_regimes=[],
            selected_model_versions=[],
            min_score_total=0.0,
            min_ml_edge=None,
            text_query="",
        )

    asset_classes = sorted(
        {str(x) for x in df_universe.get("asset_class", pd.Series(dtype=str)).dropna().unique()}
    )
    timeframes = sorted(
        {str(x) for x in df_universe.get("timeframe", pd.Series(dtype=str)).dropna().unique()}
    )
    regimes = sorted(
        {str(x) for x in df_universe.get("regime_id", pd.Series(dtype=str)).dropna().unique() if x}
    )
    macro_regimes = sorted(
        {str(x) for x in df_universe.get("macro_regime_id", pd.Series(dtype=str)).dropna().unique() if x}
    )
    model_versions = sorted(
        {str(x) for x in df_universe.get("model_version", pd.Series(dtype=str)).dropna().unique() if x}
    )

    min_score = float(pd.to_numeric(df_universe["score_total"], errors="coerce").min() or 0.0)
    max_score = float(pd.to_numeric(df_universe["score_total"], errors="coerce").max() or 1.0)
    if np.isnan(min_score): min_score = 0.0
    if np.isnan(max_score): max_score = 1.0

    ml_edge_series = pd.to_numeric(df_universe.get("ml_edge_score", pd.Series(dtype=float)), errors="coerce")
    ml_edge_series = ml_edge_series.dropna()
    has_ml_edge = not ml_edge_series.empty

    st.markdown("##### פילטרים ל-Live Universe (Profiles)")

    col_top1, col_top2, col_top3 = st.columns([1, 1, 2])

    with col_top1:
        show_only_active = st.checkbox(
            "רק Active",
            value=False,
            help="הצגת רק זוגות שמסומנים כ-is_active=True.",
        )
        hide_suspended = st.checkbox(
            "להסתיר Suspended",
            value=False,
            help="הסתרת זוגות שנמצאים במצב is_suspended=True.",
        )

    with col_top2:
        min_score_total = st.slider(
            "מינימום Score Total",
            min_value=float(min_score),
            max_value=float(max_score),
            value=float(min_score),
            step=(max_score - min_score) / 100 if max_score > min_score else 0.01,
        )
        if has_ml_edge:
            ml_min_val = float(ml_edge_series.min())
            ml_max_val = float(ml_edge_series.max())
            min_ml_edge = st.slider(
                "מינימום ML Edge (אם קיים)",
                min_value=ml_min_val,
                max_value=ml_max_val,
                value=ml_min_val,
                step=(ml_max_val - ml_min_val) / 100 if ml_max_val > ml_min_val else 0.01,
            )
        else:
            min_ml_edge = None
            st.caption("אין עדיין ערכי ML Edge (ml_edge_score) בפרופילים — אפשר להוסיף בהמשך.")
    with col_top3:
        text_query = st.text_input(
            "חיפוש לפי pair_id / סימבול / Regime / Tag / Model Version",
            value="",
            placeholder="לדוגמה: QQQ, semiconductors, v3.1, risk_off ...",
        )

    col_mid1, col_mid2, col_mid3 = st.columns(3)
    with col_mid1:
        selected_asset_classes = st.multiselect(
            "Asset Classes",
            options=asset_classes,
            default=asset_classes,
        )
        selected_timeframes = st.multiselect(
            "Timeframes",
            options=timeframes,
            default=timeframes,
        )
    with col_mid2:
        selected_regimes = st.multiselect(
            "Regime IDs",
            options=regimes,
            default=regimes,
        )
        selected_macro_regimes = st.multiselect(
            "Macro Regimes",
            options=macro_regimes,
            default=macro_regimes,
        )
    with col_mid3:
        selected_model_versions = st.multiselect(
            "Model Versions",
            options=model_versions,
            default=model_versions,
        )

    return LiveUniverseFilterState(
        show_only_active=show_only_active,
        hide_suspended=hide_suspended,
        selected_asset_classes=selected_asset_classes,
        selected_timeframes=selected_timeframes,
        selected_regimes=selected_regimes,
        selected_macro_regimes=selected_macro_regimes,
        selected_model_versions=selected_model_versions,
        min_score_total=float(min_score_total),
        min_ml_edge=float(min_ml_edge) if min_ml_edge is not None else None,
        text_query=text_query.strip(),
    )


def _apply_live_universe_filters(
    df_universe: Frame,
    filters: LiveUniverseFilterState,
) -> Frame:
    """
    מיישם פילטרים על DataFrame של Live Universe.
    """
    if df_universe.empty:
        return df_universe

    df = df_universe.copy()

    # Active / Suspended
    if filters.show_only_active:
        df = df[df["is_active"] == True]  # noqa: E712
    if filters.hide_suspended:
        df = df[df["is_suspended"] != True]  # noqa: E712

    # AssetClass / Timeframe / Regime / MacroRegime / Model
    if filters.selected_asset_classes:
        df = df[df["asset_class"].isin(filters.selected_asset_classes)]
    if filters.selected_timeframes:
        df = df[df["timeframe"].isin(filters.selected_timeframes)]
    if filters.selected_regimes:
        df = df[df["regime_id"].isin(filters.selected_regimes)]
    if filters.selected_macro_regimes:
        df = df[df["macro_regime_id"].isin(filters.selected_macro_regimes)]
    if filters.selected_model_versions:
        df = df[df["model_version"].isin(filters.selected_model_versions)]

    # Score / ML Edge
    df["score_total"] = pd.to_numeric(df.get("score_total", 0.0), errors="coerce")
    df["ml_edge_score"] = pd.to_numeric(df.get("ml_edge_score", np.nan), errors="coerce")

    df = df[df["score_total"] >= filters.min_score_total]
    if filters.min_ml_edge is not None:
        df = df[df["ml_edge_score"].notna() & (df["ml_edge_score"] >= filters.min_ml_edge)]

    # Text search
    q = filters.text_query.lower()
    if q:
        def _match(row: pd.Series) -> bool:
            text = (
                f"{row.get('pair_id','')}"
                f"{row.get('sym_x','')}"
                f"{row.get('sym_y','')}"
                f"{row.get('regime_id','')}"
                f"{row.get('macro_regime_id','')}"
                f"{row.get('model_version','')}"
                f"{row.get('tags_str','')}"
            ).lower()
            return q in text

        df = df[df.apply(_match, axis=1)]

    df = df.reset_index(drop=True)
    return df


# ------------------------------------------------------------
# Live Universe Section UI
# ------------------------------------------------------------

def render_live_universe_section(
    config: PortfolioConfig,
    ctx: Optional[JSONLike] = None,
) -> None:
    """
    מציג את Live Universe (LivePairProfile) במבט כולל:

      - סיכום Universe: כמה זוגות, כמה Active/Suspended, ממוצע Score/Edge.
      - טבלת פרופילים מלאה.
      - גרף התפלגות Score ו-ML Edge.
      - גרף התפלגות Regime / MacroRegime.

    אם אין DB או אין פרופילים – מציג הודעה ידידותית.
    """
    if not config.display.show_live_universe_section:
        return

    st.subheader("🌐 Live Universe — פרופיל לייב לכל זוג")

    # טען את כל ה-Universe
    live_profiles_all = _load_live_universe_full(config)

    if not live_profiles_all:
        st.info(
            "לא נמצאו פרופילים ב-Live Universe (LivePairStore).\n\n"
            "- אם כבר בנית LivePairProfile/Store, ודא שהמסלול והטבלה נכונים.\n"
            "- אם לא, אפשר לחבר את טאב האופטימיזציה כך שיכתוב פרופילים לייב ל-DB."
        )
        return

    df_universe = _build_live_universe_df(live_profiles_all)

    # Summary מהיר
    n_total = len(df_universe)
    n_active = int(df_universe["is_active"].sum())
    n_suspended = int(df_universe["is_suspended"].sum())
    avg_score = float(pd.to_numeric(df_universe["score_total"], errors="coerce").mean())
    avg_ml_edge = float(pd.to_numeric(df_universe["ml_edge_score"], errors="coerce").mean())

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("זוגות בסך הכל (Universe)", n_total)
    with col_s2:
        st.metric("Active", n_active)
        st.metric("Suspended", n_suspended)
    with col_s3:
        st.metric(
            "Score ממוצע",
            f"{avg_score:.3f}" if not np.isnan(avg_score) else "–",
        )
    with col_s4:
        st.metric(
            "ML Edge ממוצע",
            f"{avg_ml_edge:.3f}" if not np.isnan(avg_ml_edge) else "–",
        )

    st.markdown("---")

    # פילטרים על Universe (בתוך הסקשן)
    filters = _build_live_universe_filter_state(df_universe)
    df_filtered = _apply_live_universe_filters(df_universe, filters)

    # טבלה ראשית
    st.markdown("#### טבלת Live Universe (מסוננת)")

    if df_filtered.empty:
        st.info("אחרי הפילטרים הנוכחיים לא נשארו זוגות להציג.")
        return

    # נבנה DataFrame ידידותי ל-UI
    view = df_filtered.copy()

    # עמודות לתצוגה
    view["Pair"] = view["pair_id"]
    view["Symbols"] = view["sym_x"] + " / " + view["sym_y"]
    view["Score"] = view["score_total"].apply(
        lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )
    view["ML Edge"] = view["ml_edge_score"].apply(
        lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )
    view["Vol Target"] = view["vol_target_annual"].apply(
        lambda x: fmt_pct(x, 2) if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )
    view["Base Notional"] = view["base_notional_usd"].apply(
        lambda x: fmt_money(x, config.display.base_currency, 0)
        if isinstance(x, (int, float)) and not np.isnan(x)
        else "–"
    )
    view["Half-Life"] = view["half_life_bars"].apply(
        lambda x: f"{x:.1f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )
    view["Hurst"] = view["hurst_exponent"].apply(
        lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )
    view["Sharpe"] = view["backtest_sharpe"].apply(
        lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )
    view["Max DD"] = view["backtest_max_drawdown"].apply(
        lambda x: fmt_signed_pct(x, 2) if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )
    view["Trades Count"] = view["backtest_trades_count"].apply(
        lambda x: fmt_qty(x, 0) if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )
    view["Regime"] = view["regime_id"].fillna("")
    view["Macro Regime"] = view["macro_regime_id"].fillna("")
    view["Model Version"] = view["model_version"].fillna("")
    view["Active"] = view["is_active"]
    view["Suspended"] = view["is_suspended"]
    view["Tags"] = view["tags_str"].fillna("")

    table_cols = [
        "Pair",
        "Symbols",
        "asset_class",
        "timeframe",
        "Active",
        "Suspended",
        "Score",
        "ML Edge",
        "Vol Target",
        "Base Notional",
        "Half-Life",
        "Hurst",
        "Sharpe",
        "Max DD",
        "Trades Count",
        "Regime",
        "Macro Regime",
        "Model Version",
        "Tags",
    ]
    table_cols = [c for c in table_cols if c in view.columns]

    st.dataframe(
        view[table_cols].rename(
            columns={
                "asset_class": "Asset Class",
                "timeframe": "TF",
            }
        ),
        width = "stretch", hide_index=True,
    )

    st.markdown("---")

    # גרפים: Score / ML Edge / Regime distribution
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("##### התפלגות Score Total")
        df_score = df_filtered.copy()
        df_score = df_score.dropna(subset=["score_total"])
        if df_score.empty:
            st.caption("אין מספיק נתוני Score להצגת גרף.")
        else:
            fig_score = px.histogram(
                df_score,
                x="score_total",
                nbins=30,
                title="Histogram of Score Total",
            )
            fig_score.update_layout(
                margin=dict(l=10, r=10, t=40, b=40),
                height=300,
                xaxis_title="Score Total",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_score, width = "stretch")

    with col_g2:
        st.markdown("##### התפלגות ML Edge")
        df_edge = df_filtered.copy()
        df_edge = df_edge.dropna(subset=["ml_edge_score"])
        if df_edge.empty:
            st.caption("אין מספיק נתוני ML Edge להצגת גרף.")
        else:
            fig_edge = px.histogram(
                df_edge,
                x="ml_edge_score",
                nbins=30,
                title="Histogram of ML Edge",
            )
            fig_edge.update_layout(
                margin=dict(l=10, r=10, t=40, b=40),
                height=300,
                xaxis_title="ML Edge",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_edge, width = "stretch")

    st.markdown("---")

    st.markdown("##### התפלגות Regime / Macro Regime")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        df_regime = (
            df_filtered.groupby("regime_id", dropna=True)
            .size()
            .reset_index(name="count")
        )
        df_regime = df_regime[df_regime["regime_id"].notna()]
        if df_regime.empty:
            st.caption("אין Regime IDs בפרופילים.")
        else:
            fig_regime = px.bar(
                df_regime,
                x="regime_id",
                y="count",
                title="Regime Distribution",
            )
            fig_regime.update_layout(
                margin=dict(l=10, r=10, t=40, b=40),
                height=300,
                xaxis_title="Regime",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_regime, width = "stretch")

    with col_r2:
        df_macro = (
            df_filtered.groupby("macro_regime_id", dropna=True)
            .size()
            .reset_index(name="count")
        )
        df_macro = df_macro[df_macro["macro_regime_id"].notna()]
        if df_macro.empty:
            st.caption("אין Macro Regimes בפרופילים.")
        else:
            fig_macro = px.bar(
                df_macro,
                x="macro_regime_id",
                y="count",
                title="Macro Regime Distribution",
            )
            fig_macro.update_layout(
                margin=dict(l=10, r=10, t=40, b=40),
                height=300,
                xaxis_title="Macro Regime",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_macro, width = "stretch")

    st.divider()
# ============================================================
# Part 8 — Live Trades History: Extended Summary, Tables & Charts
# ============================================================

__all__ += [
    "render_trades_section_live",
]

def render_trades_section_live(
    trades_history: MaybeFrame,
    config: PortfolioConfig,
) -> None:
    """
    סקשן מלא להיסטוריית טריידים לייב ברמת קרן גידור:

    Features:
      - בחירת חלון זמן לניתוח: 7 / 30 / 90 ימים / כל התקופה.
      - Summary מפורט לחלון שנבחר:
          * Realized PnL
          * # טריידים
          * Win-Rate
          * Avg Notional per trade
          * Avg Holding Period
      - טבלת 100 הטריידים האחרונים (פירוט מלא לזוג/סימבול/אסטרטגיה).
      - גרף PnL יומי + PnL מצטבר (בחלון הנבחר).
      - Histogram של Realized PnL לטריידים.
      - Breakdown לפי Strategy / Asset Class / Pair.

    trades_history צפוי להגיע מ-load_trades_history_live (חלק 2).
    """
    st.subheader("🧾 היסטוריית טריידים לייב (Realized PnL Analysis)")

    if trades_history is None or trades_history.empty:
        st.info("אין היסטוריית טריידים לייב להצגה (או שלא סופקה).")
        return

    df = trades_history.copy()
    ccy = config.display.base_currency

    # טיפוסים בסיסיים
    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], errors="coerce", utc=True)
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
    df["notional"] = pd.to_numeric(df["notional"], errors="coerce").fillna(0.0)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)

    # נוודא שיש עמודות strategy/asset_class/pair_id אפילו אם ריקות
    for col in ("strategy", "asset_class", "pair_id", "source"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")

    df = df.sort_values("open_time").reset_index(drop=True)

    # ===== 1. בחירת חלון זמן לניתוח =====
    st.markdown("#### בחירת חלון זמן לניתוח")

    now = pd.Timestamp.utcnow()
    window_label = st.radio(
        "בחר חלון זמן:",
        options=["7 ימים", "30 ימים", "90 ימים", "כל התקופה"],
        index=1,
        horizontal=True,
    )

    days_back: Optional[int]
    if window_label == "7 ימים":
        days_back = 7
    elif window_label == "30 ימים":
        days_back = 30
    elif window_label == "90 ימים":
        days_back = 90
    else:
        days_back = None  # כל התקופה

    if days_back is not None:
        cutoff = now - pd.Timedelta(days=days_back)
        df_win = df[df["open_time"] >= cutoff].copy()
    else:
        df_win = df.copy()

    if df_win.empty:
        st.info("אין טריידים בחלון הזמן שנבחר.")
        return

    # ===== 2. חישוב Summary לחלון הנבחר =====
    realized_win = float(df_win["realized_pnl"].sum())
    n_trades_win = int(len(df_win))
    win_rate_win = float((df_win["realized_pnl"] > 0).mean()) if n_trades_win > 0 else None
    avg_notional_win = float(df_win["notional"].mean()) if n_trades_win > 0 else 0.0

    # זמן החזקה (ימים)
    holding_secs = (df_win["close_time"] - df_win["open_time"]).dt.total_seconds()
    holding_days = holding_secs / (60 * 60 * 24)
    avg_holding_days = float(holding_days.mean()) if holding_days.notna().any() else None

    # חלון "מובהק" להשוואה (30 ימים) בסיכום קטן
    cutoff_30 = now - pd.Timedelta(days=30)
    df_30 = df[df["open_time"] >= cutoff_30].copy()
    realized_30 = float(df_30["realized_pnl"].sum()) if not df_30.empty else 0.0
    n_trades_30 = int(len(df_30))
    win_rate_30 = float((df_30["realized_pnl"] > 0).mean()) if n_trades_30 > 0 else None

    # ===== 3. KPI שורת Summary =====
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.metric(
            label=f"Realized PnL ({window_label})",
            value=fmt_signed_money(realized_win, ccy, config.display.precision_money),
        )
        st.caption("סך Realized PnL בחלון הזמן שנבחר.")

    with col_s2:
        st.metric(
            label=f"# טריידים ({window_label})",
            value=n_trades_win,
        )
        if win_rate_win is not None:
            st.caption(f"Win-Rate {fmt_pct(win_rate_win, 1)} בחלון הנבחר.")
        else:
            st.caption("Win-Rate לא זמין (אין מספיק טריידים).")

    with col_s3:
        st.metric(
            label=f"גודל עסקה ממוצע ({window_label})",
            value=fmt_money(avg_notional_win, ccy, 0),
        )
        if avg_holding_days is not None:
            st.caption(f"תקופת החזקה ממוצעת: ~{avg_holding_days:.1f} ימים.")
        else:
            st.caption("אין מספיק מידע על זמן פתיחה/סגירה.")

    with col_s4:
        # שורת השוואה קטנה ל-30 יום
        if days_back != 30 and n_trades_30 > 0:
            delta_vs_30 = realized_win - realized_30
            st.metric(
                label="השוואה ל-Realized PnL (30 ימים)",
                value=fmt_signed_money(realized_30, ccy, config.display.precision_money),
                delta=fmt_signed_money(delta_vs_30, ccy, config.display.precision_money),
            )
        else:
            st.metric(
                label="Realized PnL (30 ימים)",
                value=fmt_signed_money(realized_30, ccy, config.display.precision_money),
            )

    st.markdown("---")

    # ===== 4. טבלת 100 הטריידים האחרונים (בחלון הנבחר) =====
    st.markdown("#### 100 הטריידים האחרונים בחלון הזמן הנבחר")

    df_last = df_win.sort_values("open_time", ascending=False).head(100).copy()

    view = pd.DataFrame(
        {
            "Open Time": df_last["open_time"].dt.tz_convert("UTC"),
            "Close Time": df_last["close_time"].dt.tz_convert("UTC"),
            "Pair": df_last["pair_id"],
            "Symbol": df_last["symbol"],
            "Side": df_last["side"],
            "Qty": df_last["qty"],
            "Price": df_last["price"],
            "Notional": df_last["notional"],
            "Realized PnL": df_last["realized_pnl"],
            "Strategy": df_last.get("strategy", ""),
            "Asset Class": df_last.get("asset_class", ""),
            "Source": df_last.get("source", ""),
        }
    )

    view["Qty"] = view["Qty"].apply(
        lambda x: fmt_qty(x, config.display.precision_qty)
    )
    view["Price"] = view["Price"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and not np.isnan(x) else "–"
    )
    view["Notional"] = view["Notional"].apply(
        lambda x: fmt_money(x, ccy, config.display.precision_money)
    )
    view["Realized PnL"] = view["Realized PnL"].apply(
        lambda x: fmt_signed_money(x, ccy, config.display.precision_money)
    )

    st.dataframe(
        view.sort_values("Open Time", ascending=False),
        width = "stretch", hide_index=True,
    )

    st.markdown("---")

    # ===== 5. גרף PnL יומי + מצטבר (בחלון הנבחר) =====
    st.markdown("#### PnL יומי ומצטבר (לפי טריידים סגורים בחלון הנבחר)")

    df_pnl = df_win.copy()
    df_pnl["pnl_date"] = df_pnl["close_time"].dt.floor("D")
    pnl_by_day = (
        df_pnl.dropna(subset=["pnl_date"])
        .groupby("pnl_date")["realized_pnl"]
        .sum()
        .reset_index()
        .sort_values("pnl_date")
    )

    if pnl_by_day.empty:
        st.info("אין מספיק נתוני close_time כדי ליצור גרף PnL יומי.")
    else:
        pnl_by_day["cum_pnl"] = pnl_by_day["realized_pnl"].cumsum()

        fig_pnl = go.Figure()
        fig_pnl.add_trace(
            go.Bar(
                x=pnl_by_day["pnl_date"],
                y=pnl_by_day["realized_pnl"],
                name="PnL יומי",
            )
        )
        fig_pnl.add_trace(
            go.Scatter(
                x=pnl_by_day["pnl_date"],
                y=pnl_by_day["cum_pnl"],
                mode="lines",
                name="PnL מצטבר",
                yaxis="y2",
            )
        )

        fig_pnl.update_layout(
            margin=dict(l=10, r=10, t=40, b=40),
            height=350,
            xaxis_title="תאריך סגירה",
            yaxis=dict(
                title=f"Realized PnL ({ccy})",
                side="left",
            ),
            yaxis2=dict(
                title=f"Cumulative PnL ({ccy})",
                overlaying="y",
                side="right",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_pnl, width = "stretch")

    st.markdown("---")

    # ===== 6. Histogram של Realized PnL לטריידים =====
    st.markdown("#### התפלגות Realized PnL לטריידים (Trade-Level)")

    df_hist = df_win.copy()
    df_hist = df_hist[df_hist["realized_pnl"].notna()]

    if df_hist.empty:
        st.info("אין ערכי Realized PnL להצגת Histogram.")
    else:
        fig_hist = px.histogram(
            df_hist,
            x="realized_pnl",
            nbins=40,
            title="Histogram of Trade Realized PnL",
        )
        fig_hist.update_layout(
            margin=dict(l=10, r=10, t=40, b=40),
            height=300,
            xaxis_title=f"Realized PnL per trade ({ccy})",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_hist, width = "stretch")

    st.markdown("---")

    # ===== 7. Histogram של Holding Period =====
    st.markdown("#### התפלגות תקופת החזקה (Holding Period)")

    df_hp = df_win.copy()
    hp_days = (df_hp["close_time"] - df_hp["open_time"]).dt.total_seconds() / (60 * 60 * 24)
    df_hp["holding_days"] = hp_days

    hp_valid = df_hp["holding_days"].dropna()
    if hp_valid.empty:
        st.caption("אין מספיק מידע על open_time/close_time כדי לנתח תקופת החזקה.")
    else:
        fig_hp = px.histogram(
            df_hp,
            x="holding_days",
            nbins=30,
            title="Histogram of Holding Period (Days)",
        )
        fig_hp.update_layout(
            margin=dict(l=10, r=10, t=40, b=40),
            height=300,
            xaxis_title="Holding Period (days)",
            yaxis_title="Count",
        )
        st.plotly_chart(fig_hp, width = "stretch")

    st.markdown("---")

    # ===== 8. Breakdown לפי Strategy / Asset Class / Pair =====
    st.markdown("#### Breakdown לפי אסטרטגיה / Asset Class / Pair")

    df_sum = df_win.copy()

    col_b1, col_b2, col_b3 = st.columns(3)

    # Strategy breakdown
    with col_b1:
        st.markdown("##### Strategy (Realized PnL)")

        strat_group = (
            df_sum.groupby("strategy", dropna=False)["realized_pnl"]
            .sum()
            .reset_index()
            .sort_values("realized_pnl", ascending=False)
        )
        strat_group["strategy"] = strat_group["strategy"].astype(str).replace({"": "Unknown"})

        if strat_group.empty:
            st.caption("אין נתוני Strategy בטריידים בחלון הנבחר.")
        else:
            fig_strat = px.bar(
                strat_group,
                x="strategy",
                y="realized_pnl",
                title="Realized PnL by Strategy",
            )
            fig_strat.update_layout(
                margin=dict(l=10, r=10, t=40, b=40),
                height=300,
                xaxis_title="Strategy",
                yaxis_title=f"Realized PnL ({ccy})",
            )
            st.plotly_chart(fig_strat, width = "stretch")

    # Asset class breakdown
    with col_b2:
        st.markdown("##### Asset Class (Realized PnL)")

        ac_group = (
            df_sum.groupby("asset_class", dropna=False)["realized_pnl"]
            .sum()
            .reset_index()
            .sort_values("realized_pnl", ascending=False)
        )
        ac_group["asset_class"] = ac_group["asset_class"].astype(str).replace({"": "Unknown"})

        if ac_group.empty:
            st.caption("אין נתוני Asset Class בטריידים בחלון הנבחר.")
        else:
            fig_ac = px.bar(
                ac_group,
                x="asset_class",
                y="realized_pnl",
                title="Realized PnL by Asset Class",
            )
            fig_ac.update_layout(
                margin=dict(l=10, r=10, t=40, b=40),
                height=300,
                xaxis_title="Asset Class",
                yaxis_title=f"Realized PnL ({ccy})",
            )
            st.plotly_chart(fig_ac, width = "stretch")

    # Pair breakdown
    with col_b3:
        st.markdown("##### Pair (Realized PnL)")

        pair_group = (
            df_sum.groupby("pair_id", dropna=False)["realized_pnl"]
            .sum()
            .reset_index()
            .sort_values("realized_pnl", ascending=False)
            .head(15)
        )
        pair_group["pair_id"] = pair_group["pair_id"].astype(str).replace({"": "Unknown"})

        if pair_group.empty:
            st.caption("אין נתוני pair_id בטריידים בחלון הנבחר.")
        else:
            fig_pair = px.bar(
                pair_group,
                x="pair_id",
                y="realized_pnl",
                title="Top 15 Pairs by Realized PnL",
            )
            fig_pair.update_layout(
                margin=dict(l=10, r=10, t=40, b=40),
                height=300,
                xaxis_title="Pair",
                yaxis_title=f"Realized PnL ({ccy})",
            )
            st.plotly_chart(fig_pair, width = "stretch")

    st.divider()
# ============================================================
# Part 9 — Live Engine Control: Commands, History & Control Panel (Extended)
# ============================================================

__all__ += [
    "EngineCommand",
    "EngineCommandPayload",
    "EngineCommandResult",
    "render_engine_control_panel",
]

# ------------------------------------------------------------
# Engine command types & payload
# ------------------------------------------------------------

class EngineCommand(str, Enum):
    """
    פקודות לוגיות למנוע המסחר החי.

    שים לב:
        - זהו חוזה לוגי בלבד בטאב.
        - בפועל, הפונקציה _send_engine_command_stub אינה מחוברת עדיין ל-API אמיתי.
        - בעתיד אפשר להחליף ב-HTTP / IPC / gRPC / כל מנגנון אחר.
    """

    PAUSE = "pause"                    # לעצור פתיחה של עסקאות חדשות
    RESUME = "resume"                  # להמשיך פתיחה של עסקאות
    HALT = "halt"                      # לעצור הכל (Kill-Switch מלא)
    CLOSE_ALL = "close_all"            # לסגור כל הפוזיציות הפתוחות (לוגי)
    RELOAD_UNIVERSE = "reload_universe"  # לטעון מחדש את Live Universe מה-DB
    SOFT_RESET = "soft_reset"          # לאתחל state פנימי (טיימרים/ספירות) בלי לסגור פוזיציות
    # אפשר להוסיף בעתיד:
    # CLOSE_PAIR = "close_pair"
    # RELOAD_CONFIG = "reload_config"


@dataclass
class EngineCommandPayload:
    """
    Payload לוגי לפקודות מנוע.

    אפשר להרחיב אותו בעתיד לפי צורכי ה-API שלך:
      - pair_id לסגירת זוג ספציפי,
      - dry_run להפעלה ללא ביצוע אמיתי,
      - user_id / source לצורך לוג audit.
    """

    pair_id: Optional[str] = None      # לעתיד: Close Pair ספציפי
    dry_run: bool = False              # כשנרצה לבדוק זרימה בלי לבצע בפועל
    source: str = "dashboard"          # מי יזם (dashboard/web/cli/...)


@dataclass
class EngineCommandResult:
    """
    תוצאה לוגית של פקודה למנוע:

    command:
        איזה Command נשלח (PAUSE/RESUME/...).
    payload:
        מה היה Payload (אם היה).
    success:
        האם הפקודה התקבלה/בוצעה (לוגית).
    message:
        מסר קצר ברור למשתמש.
    details:
        מידע נוסף טכני (לא חובה).
    timestamp:
        זמן שליחת הפקודה מהטאב (UTC).
    """

    command: EngineCommand
    payload: EngineCommandPayload
    success: bool
    message: str
    details: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc)())


# ------------------------------------------------------------
# Internal helpers: session_state history & stub sender
# ------------------------------------------------------------

_ENGINE_HISTORY_KEY = "engine_control_history"


def _get_engine_control_history() -> List[EngineCommandResult]:
    """
    מחזיר רשימת פקודות היסטוריות מ-session_state.

    היסטוריה זו:
      - חיה רק בריצה הנוכחית של Streamlit.
      - טובה ליישור קו ויזואלי בדשבורד (מה לחצו ומה קרה).
    """
    if _ENGINE_HISTORY_KEY not in st.session_state:
        st.session_state[_ENGINE_HISTORY_KEY] = []  # type: ignore[assignment]
    # type: ignore כי Streamlit לא מקל על type checkers
    return st.session_state[_ENGINE_HISTORY_KEY]  # type: ignore[return-value]


def _append_engine_control_history(result: EngineCommandResult) -> None:
    """
    מוסיף תוצאת פקודה להיסטוריה ב-session_state.
    """
    history = _get_engine_control_history()
    history.append(result)
    # נשמור את הרשימה בחזרה כדי לוודא ש-Streamlit מזהה שינוי
    st.session_state[_ENGINE_HISTORY_KEY] = history  # type: ignore[assignment]


def _send_engine_command_stub(
    command: EngineCommand,
    payload: EngineCommandPayload,
    config: PortfolioConfig,
    ctx: Optional[JSONLike] = None,
) -> EngineCommandResult:
    """
    Stub לשליחת פקודה למנוע הלייב.

    כרגע:
      - לא מחובר ל-API אמיתי (לוג בלבד).
      - מחזיר EngineCommandResult עם success=True.
      - שומר היסטוריה בתוך session_state.

    איך לחבר בעתיד:
      - להחליף את גוף הפונקציה ב:
          resp = requests.post("http://localhost:8000/engine/command", json={...})
          ואז להמיר את resp ל-EngineCommandResult.
    """
    logger.info(
        "[EngineControl] command=%s payload=%s engine_name=%s",
        command.value,
        payload,
        config.live.engine_name,
    )

    # כאן תוכל להכניס התקשרות אמיתית ל-API בעתיד
    msg = f"Command '{command.value}' נשלחה (stub; טרם מחובר ל-API אמיתי)."

    res = EngineCommandResult(
        command=command,
        payload=payload,
        success=True,
        message=msg,
        details=None,
    )
    _append_engine_control_history(res)
    return res


def _render_engine_command_result(res: EngineCommandResult) -> None:
    """
    מציג תוצאת פקודה ב-UI (מיד לאחר לחיצה על כפתור).

    - אם success=True → st.success(...) + timestamp.
    - אם success=False → st.error(...).
    """
    ts_str = res.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    base_msg = f"{res.message}\n\nזמן שליחה: {ts_str}"

    if res.success:
        st.success(base_msg)
    else:
        st.error(base_msg)

    if res.details:
        with st.expander("פרטים נוספים על הפקודה", expanded=False):
            st.write(res.details)


# ------------------------------------------------------------
# Engine Control Panel UI
# ------------------------------------------------------------

def render_engine_control_panel(
    analytics: PortfolioAnalytics,
    config: PortfolioConfig,
    ctx: Optional[JSONLike] = None,
) -> None:
    """
    פאנל שליטה במנוע הלייב (UI בלבד, עדיין Stub):

      - שורת פקודות "שגרתיות":
          * Pause / Resume / Soft Reset / Reload Universe.
      - אזור "Danger Zone":
          * Halt (Kill-Switch מלא).
          * Close All Positions (לוגי).
        עם אישור כפול (checkbox) לפני ביצוע.

      - בתחתית: היסטוריית פקודות שנשלחו מהטאב (session_state).

    אינטגרציה עתידית:
      - אפשר להחליף את _send_engine_command_stub ב-API אמיתי,
        בלי לשנות את ה-UI.
    """
    st.subheader("🎛️ שליטה במנוע המסחר החי (Engine Control Panel)")

    # מידע קצר על מצב המנוע
    mode = str(analytics.engine_status.get("mode", "UNKNOWN")).upper()
    last_heartbeat = analytics.engine_status.get("last_heartbeat")
    if isinstance(last_heartbeat, (str, datetime)):
        try:
            last_heartbeat_dt = pd.to_datetime(last_heartbeat, errors="coerce", utc=True)
            if pd.isna(last_heartbeat_dt):
                last_heartbeat_str = "לא ידוע"
            else:
                last_heartbeat_str = last_heartbeat_dt.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            last_heartbeat_str = str(last_heartbeat)
    else:
        last_heartbeat_str = "לא ידוע"

    engine_mode_text = {
        "RUNNING": "המנוע פועל (RUNNING)",
        "PAUSED": "המנוע מושהה (PAUSED)",
        "HALTED": "המנוע נעצר (HALTED)",
        "UNKNOWN": "סטטוס מנוע לא ידוע (UNKNOWN)",
    }.get(mode, "סטטוס מנוע לא ידוע (UNKNOWN)")

    st.caption(
        f"סטטוס מנוע: **{engine_mode_text}** | "
        f"Heartbeat אחרון: {last_heartbeat_str}"
    )

    st.info(
        "הערה: כרגע הפאנל מחובר ל-stub בלבד (אין API אמיתי). "
        "ניתן להחליף את _send_engine_command_stub כדי לחבר למנוע."
    )

    # --- שורת כפתורי שליטה שוטפים ---
    st.markdown("#### פקודות שוטפות")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("⏸️ Pause", help="עצירת פתיחת עסקאות חדשות (המנוע נשאר דולק)."):
            payload = EngineCommandPayload(dry_run=False, source="dashboard")
            res = _send_engine_command_stub(EngineCommand.PAUSE, payload, config, ctx)
            _render_engine_command_result(res)

    with col2:
        if st.button("▶️ Resume", help="חידוש פתיחת עסקאות חדשות."):
            payload = EngineCommandPayload(dry_run=False, source="dashboard")
            res = _send_engine_command_stub(EngineCommand.RESUME, payload, config, ctx)
            _render_engine_command_result(res)

    with col3:
        if st.button("🔄 Reload Universe", help="טעינה מחדש של Live Universe מה-DB (שינוי פרופילים):"):
            payload = EngineCommandPayload(dry_run=False, source="dashboard")
            res = _send_engine_command_stub(EngineCommand.RELOAD_UNIVERSE, payload, config, ctx)
            _render_engine_command_result(res)

    with col4:
        if st.button("♻️ Soft Reset", help="איפוס state פנימיים של המנוע (ללא סגירת פוזיציות)."):
            payload = EngineCommandPayload(dry_run=False, source="dashboard")
            res = _send_engine_command_stub(EngineCommand.SOFT_RESET, payload, config, ctx)
            _render_engine_command_result(res)

    st.markdown("---")

    # --- Danger Zone ---
    st.markdown("#### ⚠️ Danger Zone (פקודות מסוכנות)")

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("##### 🛑 Halt (Kill-Switch)")
        confirm_halt = st.checkbox(
            "אני מבין ש-Halt עוצר את המנוע לגמרי.",
            key="confirm_halt",
            help="במצב HALTED המנוע לא פותח ולא סוגר עסקאות אוטומטית.",
        )
        if st.button("🛑 Halt Engine", help="עצירת המנוע לגמרי. לא פותח/סוגר עסקאות אוטומטיות."):
            if not confirm_halt:
                st.error("יש לסמן את תיבת האישור לפני הרצת Halt.")
            else:
                payload = EngineCommandPayload(dry_run=False, source="dashboard")
                res = _send_engine_command_stub(EngineCommand.HALT, payload, config, ctx)
                _render_engine_command_result(res)

    with col_d2:
        st.markdown("##### ❌ Close All Positions")
        confirm_close_all = st.checkbox(
            "אני מבין שסגירת כל הפוזיציות היא פעולה מסוכנת.",
            key="confirm_close_all",
            help="מיועד לשימוש במקרי חירום בלבד. יש לוודא שיש נזילות מספקת וכו'.",
        )
        if st.button("❌ Close All (לוגי)", help="שילוח פקודת Close All למנוע (stub כרגע)."):
            if not confirm_close_all:
                st.error("יש לסמן את תיבת האישור לפני Close All.")
            else:
                payload = EngineCommandPayload(dry_run=False, source="dashboard")
                res = _send_engine_command_stub(EngineCommand.CLOSE_ALL, payload, config, ctx)
                _render_engine_command_result(res)

    st.divider()

    # --- היסטוריית פקודות ---
    st.markdown("#### 🧬 היסטוריית פקודות שנשלחו למנוע (Session)")

    history = _get_engine_control_history()
    if not history:
        st.caption("לא נשלחו עדיין פקודות מהטאב הנוכחי.")
        return

    # נבנה DataFrame מסודר
    hist_rows = []
    for r in history:
        hist_rows.append(
            {
                "Timestamp (UTC)": r.timestamp,
                "Command": r.command.value,
                "Success": r.success,
                "Message": r.message,
                "Payload.dry_run": r.payload.dry_run,
                "Payload.pair_id": r.payload.pair_id or "",
                "Source": r.payload.source,
            }
        )

    df_hist = pd.DataFrame(hist_rows).sort_values("Timestamp (UTC)", ascending=False)

    st.dataframe(
        df_hist,
        width = "stretch", hide_index=True,
    )
# ============================================================
# Part 10 — Orchestrator: render_portfolio_tab_live
# ============================================================

__all__ += [
    "render_portfolio_tab_live",
]

# ------------------------------------------------------------
# Helper: טעינת קונפיג מה-ctx (אם קיים) או ברירת מחדל
# ------------------------------------------------------------

def _load_portfolio_config_from_ctx(
    ctx: Optional[JSONLike],
) -> PortfolioConfig:
    """
    טוען PortfolioConfig:

      1. מתחיל מ-PortfolioConfig() עם ברירות המחדל (Risk/Display/Live).
      2. אם ctx מכיל:
           - ctx["portfolio_tab_config"] או
           - ctx["portfolio_config"] או
           - ctx["live_portfolio_config"]
         ומדובר במילון, הוא ידרוס שדות רלוונטיים.

    הערה:
      - איננו תלויים כאן ב-TradingConfigManager; אם תרצה, אפשר להרחיב בעתיד.
    """
    cfg = PortfolioConfig()

    if ctx is None or not isinstance(ctx, Mapping):
        return cfg

    overrides = (
        ctx.get("portfolio_tab_config")
        or ctx.get("portfolio_config")
        or ctx.get("live_portfolio_config")
    )
    if not isinstance(overrides, Mapping):
        return cfg

    # risk/display/live כבלוקים
    risk_ovr = overrides.get("risk")
    if isinstance(risk_ovr, Mapping):
        for key, value in risk_ovr.items():
            if hasattr(cfg.risk, key):
                setattr(cfg.risk, key, value)

    display_ovr = overrides.get("display")
    if isinstance(display_ovr, Mapping):
        for key, value in display_ovr.items():
            if hasattr(cfg.display, key):
                setattr(cfg.display, key, value)

    live_ovr = overrides.get("live")
    if isinstance(live_ovr, Mapping):
        for key, value in live_ovr.items():
            if hasattr(cfg.live, key):
                setattr(cfg.live, key, value)

    # שדות ישירים ברמת PortfolioConfig
    for key, value in overrides.items():
        if key in {"risk", "display", "live"}:
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    return cfg


# ------------------------------------------------------------
# Orchestrator Main — פונקציה ראשית לטאב
# ------------------------------------------------------------

def render_portfolio_tab_live(
    start_date: date,
    end_date: date,
    *,
    ctx: Optional[JSONLike] = None,
    **kwargs: Any,
) -> None:
    """
    פונקציה ראשית להרצת טאב הפרוטפוליו לייב מתוך ה-dashboard.

    שימוש מתוך root/dashboard.py:
        from root.portfolio_tab import render_portfolio_tab_live

        ...
        elif active_tab == "Portfolio":
            render_portfolio_tab_live(
                start_date,
                end_date,
                ctx=asdict(ctx),   # או dict אחר שמכיל account/engine_status וכו'
                **ctrl_dict,       # אם תרצה להעביר פילטרים נוספים בעתיד
            )

    השלבים:
      1. טעינת קונפיג (Risk/Display/Live).
      2. טעינת דאטה לייב:
           - live_positions (זוגות + פוזיציות מהמערכת/IBKR/DB/ctx).
           - equity_curve_live.
           - trades_history_live.
           - live_universe_for_engine (רק זוגות שהמנוע אמור לסחור בהם).
      3. חישוב אנליטיקה (KPI, Exposure, Health Checks).
      4. בניית פילטרים + פוזיציות מסוננות.
      5. רנדר טאב מלא עם tabs פנימיים:
           - Overview
           - Risk & Health
           - Positions
           - Live Universe
           - Trades
           - Engine Control
    """
    try:
        # --------------------------------------------------------
        # 1) קונפיג
        # --------------------------------------------------------
        config = _load_portfolio_config_from_ctx(ctx)

        # --------------------------------------------------------
        # 2) טעינת דאטה לייב (אפשר עם spinner)
        # --------------------------------------------------------
        with st.spinner("טוען נתוני פרוטפוליו לייב..."):
            positions = load_live_positions_from_ctx_or_system(config, ctx)
            equity_curve = load_equity_curve_live(config, ctx)
            trades_history = load_trades_history_live(config, ctx, days_back=180)
            live_profiles_for_engine = load_live_universe_for_portfolio(config)

        # אם אין בכלל דאטה – נאמר זאת בצורה ברורה
        if (positions is None or positions.empty) and (
            equity_curve is None or equity_curve.empty
        ) and not live_profiles_for_engine:
            st.warning(
                "לא נמצאו נתוני פרוטפוליו לייב להצגה.\n\n"
                "- בדוק חיבור ל-IBKR / DB / Engine.\n"
                "- בדוק שהמודולים common.portfolio_loader / data_loader זמינים.\n"
                "- אם יש LivePairStore, ודא שהנתיב בקונפיג נכון.\n"
                "- לחלופין, אפשר להפעיל demo_mode ב-PortfolioConfig."
            )
            return

        # --------------------------------------------------------
        # 3) חישוב אנליטיקה (KPI, Exposure, Health), כולל Live Profiles
        # --------------------------------------------------------
        analytics = compute_portfolio_analytics(
            positions=positions,
            equity_curve=equity_curve,
            config=config,
            ctx=ctx,
            live_profiles_by_pair=live_profiles_for_engine,
        )

        # --------------------------------------------------------
        # 4) פילטרים על פוזיציות מסוננות
        # --------------------------------------------------------
        filters = build_portfolio_filter_state_live(positions, analytics, config)
        positions_filtered = apply_portfolio_filters_live(
            positions, analytics.live_profiles_by_pair, filters
        )

        # --------------------------------------------------------
        # 5) Tabs פנימיים – מבט כולל, סיכון, פוזיציות, Universe, Trades, Engine
        # --------------------------------------------------------
        main_tabs = st.tabs(
            [
                "Overview",
                "Risk & Health",
                "Positions",
                "Live Universe",
                "Trades",
                "Engine Control",
            ]
        )

        # === Overview ===
        with main_tabs[0]:
            render_kpi_header(analytics, config)
            render_equity_section(equity_curve, config)
            render_exposure_overview(analytics, config)

            # 🔹 שכבת Risk-Parity חדשה
            st.markdown("### 🛡️ Risk-Parity – ניהול סיכון לפי תנודתיות התיק")
            render_risk_parity_engine(analytics, config)

            # הכותרת והחלק הקיים של ה-Overview נשארים כמו שהם
            # (שורה זו היא מהקוד המקורי שלך)
            st.markdown("### 🧭 סיכום הצעות פעולה (Overview)")
            render_action_hints_live(analytics, config)

        # === Risk & Health ===
        with main_tabs[1]:
            render_kpi_header(analytics, config)
            render_health_section_live(analytics, config)
            st.markdown("### 💡 הצעות פעולה מפורטות (Risk & Health)")
            render_action_hints_live(analytics, config)

        # === Positions ===
        with main_tabs[2]:
            render_positions_section_live(
                positions_filtered=positions_filtered,
                analytics=analytics,
                config=config,
                filters=filters,
            )

        # === Live Universe ===
        with main_tabs[3]:
            render_live_universe_section(config=config, ctx=ctx)

        # === Trades ===
        with main_tabs[4]:
            render_trades_section_live(trades_history, config)

        # === Engine Control ===
        with main_tabs[5]:
            render_engine_control_panel(
                analytics=analytics,
                config=config,
                ctx=ctx,
            )


    except Exception as exc:  # pragma: no cover
        logger.exception("Error while rendering live portfolio tab: %s", exc)
        st.error("אירעה שגיאה בעת טעינת טאב הפרוטפוליו לייב.")
        st.exception(exc)

def render_portfolio_tab(
    start_date: date,
    end_date: date,
    *,
    ctx: Optional[JSONLike] = None,
    **kwargs: Any,
) -> None:
    """
    Wrapper רשמי עבור ה-router של הדשבורד לטאב 'portfolio'.

    מעבר ל-pass-through פשוט ל-render_portfolio_tab_live, ה-wrapper הזה משמש
    כ"שכבת Runtime" ברמת קרן גידור:

    מה הוא עושה בפועל (בתרגום חופשי ל-25 רעיונות שדרוג):
    --------------------------------------------------------
    1. Normalized ctx:
       • תומך בכל הצורות:
         - dict רגיל
         - DashboardContext / AppContext / dataclass כלשהו (asdict)
         - אובייקט "עשיר" כמו SimpleNamespace עם שדות.
       • ממזג גם ctx מה-params, גם dashboard_ctx מתוך session_state,
         וגם nav_payload (אם קיים) בצורה רכה.

    2. זיהוי env/profile חכם:
       • משתמש ב:
         - kwargs["env"] / kwargs["profile"]
         - ctx["env"] / ctx["profile"]
         - משתני סביבה (DASH_ENV / DASH_PROFILE)
         - ברירת מחדל dev/default.
       • שומר ב-effective_ctx גם env/profile וגם "__tab_profile__" בסגנון "dev:monitoring".

    3. run metadata ברמת טאב:
       • מייצר tab_run_id ייחודי (uuid קצר).
       • שומר:
         - start_date / end_date
         - env/profile
         - capabilities (אם זמינים)
         - שם המשתמש / hostname
         - timestamp של הריצה
         - snapshot_id ייחודי.
       • נגיש דרך ctx["portfolio_tab_meta"].

    4. Telemetry & timing:
       • מודד זמן ריצה (ms) ומדווח ללוג.
       • שומר היסטוריה ב-st.session_state["portfolio_tab_telemetry"]:
         - רשימת ריצות עם זמן, env/profile, מספר ריצה (#1, #2, ...),
           error_message ו־ok=True/False.

    5. User prefs לטאב:
       • st.session_state["portfolio_tab_prefs"] כולל:
         - last_view_mode (Overview / Risk / Positions / Universe / Trades / Engine)
         - last_date_range_mode (לסינכרון עתידי עם date_range_mode של הדשבורד)
         - show_demo_mode (האם לאפשר דמו אם אין דאטה)
         - preferred_risk_view ("Risk & Health" כברירת מחדל)
         - preferred_language ("he" / "en" לעתיד)
         - show_runtime_banner (האם להציג סטריפ מידע למעלה).
       • prefs מוזרק לתוך ctx["portfolio_prefs"] כך שהלייב יכול להשתמש בו.

    6. Demo mode חכם:
       • אם env הוא dev/research והמשתמש לא כיבה demo:
         - מוסיף ctx["portfolio_tab_config"]["allow_demo_mode"] = True
       • אם env הוא prod/live:
         - לא נוגע ב-demo אלא אם הוגדר מפורשות.

    7. Focus universe integration:
       • בונה ctx["portfolio_focus_universe"] מרשימות קיימות:
         - smart_scan_shortlist (אם קיימת) → העדפה ראשונה.
         - אחרת opt_batch_pairs (אם קיימת).
       • מאפשר לטאב הפרוטפוליו לדעת מי "היחידות המועדפות" לחפירה.

    8. Risk budget overlay:
       • מנסה להרכיב ctx["portfolio_risk_budget"] מתוך:
         - ctx["risk_capital"] (אם קיים)
         - או st.session_state["risk_capital"] (אם קיים).
       • כולל:
         - capital
         - max_exposure_per_trade
         - max_leverage
         - sleeve_limits אם קיימים
         - risk_profile (למשל "conservative"/"balanced"/"aggressive").

    9. Engine status / capabilities:
       • מכניס ctx["engine_status"] אם קיים ב-session/ctx.
       • מכניס ctx["capabilities"] אם st.session_state["dashboard_capabilities"] קיים.
       • metadata["engine_status"] / ["capabilities"] זמינים גם לסוכנים/לוגים.

    10. SQL/Log section hint:
        • מגדיר ctx["sql_section"] = "portfolio_live" אם env in {"prod","live"},
          אחרת "portfolio_dev".
        • מאפשר ל-SqlStore/RiskEngine לדעת לאן לשייך את הריצה.

    11. Health flags:
        • בונה רשימת meta["health_flags"]:
          - "no_account" אם אין account_id.
          - "low_equity" אם equity_ctx קטן מסף מסוים (למשל < 10k).
          - "demo_mode" אם show_demo_mode=True וה-env לא prod.
        • אפשר להרחיב בעתיד ולתחבר לטאב ה-Risk.

    12. Banner עליון קטן (אופציונלי, לא פולשני):
        • בראש הטאב (מעל הכל) מציג סטריפ מידע:
          - env/profile
          - range תאריכים
          - tab_run_id ו-run_index
        • ניתן לכבות ע"י prefs["show_runtime_banner"]=False.

    13. nav_target Integration:
        • אם ב-session_state["nav_target"] == "portfolio":
          - מסמן ctx["nav_target_consumed"] = True (טאב כבר נטען).

    14. אינפורמציית חשבון לבריאות:
        • מנסה לחלץ ctx["account"]["equity"] / ["account_id"]:
          - מדווח ללוג: equity_preview/account_id.

    15. שמירת ctx האפקטיבי:
        • שומר את המפה הסופית ב-st.session_state["portfolio_ctx_effective"].
        • מאפשר future-debug לראות בדיוק מה הועבר לטאב.

    16. Snapshot אחרון:
        • שומר st.session_state["portfolio_last_snapshot"] עם:
          - meta, prefs, risk_budget, focus_universe, env/profile וכו'.

    17. Error handling מקצועי:
        • עוטף את הקריאה ל-render_portfolio_tab_live ב-try/except.
        • במצב רגיל:
          - מציג st.error מרוכך, בלי לקרוס.
        • במצב DEBUG (APP_DEBUG=1 או ctx["debug_dashboard"]=True):
          - מציג גם traceback מלא ב-expander.
        • טלמטריה: tele_entry["ok"]=False + error_message.

    18. Auto-refresh hint:
        • meta["refresh_hint_sec"]:
          - 15 ב-live/prod
          - 45 ב-dev/research
          - 60+ בפרופילים איטיים.
        • אפשר להשתמש בעתיד ל-auto-refresh של טאב הפרוטפוליו.

    19–25. Hooks לעתיד:
        • meta["hooks"] שמשמש כנקודת חיבור:
          - "sql_log_ready"
          - "agent_portfolio_view"
          - "dashboard_export_ready"
        • בלי לגעת בלוגיקה של הטאב עצמו.
    """
    from dataclasses import asdict, is_dataclass  # import מקומי (שומר על ראש נקי)
    from collections.abc import Mapping
    from datetime import datetime, timezone
    import time
    import uuid

    t0 = time.perf_counter()
    tab_run_id = uuid.uuid4().hex[:12]
    snapshot_id = uuid.uuid4().hex[:8]

    # --------------------------------------------------------
    # 1) בניית effective_ctx – מיזוג ctx + dashboard_ctx + nav_payload
    # --------------------------------------------------------
    effective_ctx: Dict[str, Any] = {}

    # מקור 1: ctx שהועבר מה-router
    if ctx is not None:
        try:
            if isinstance(ctx, Mapping):
                effective_ctx.update(ctx)
            elif is_dataclass(ctx):
                effective_ctx.update(asdict(ctx))
            else:
                attrs = {
                    k: getattr(ctx, k)
                    for k in dir(ctx)
                    if not k.startswith("_") and not callable(getattr(ctx, k, None))
                }
                effective_ctx.update(attrs)
        except Exception as exc:  # pragma: no cover
            logger.debug("render_portfolio_tab: failed to normalize ctx param (%s)", exc)

    # מקור 2: dashboard_ctx מתוך session_state (אם קיים)
    dash_ctx = st.session_state.get("dashboard_ctx")
    if isinstance(dash_ctx, Mapping):
        for k, v in dash_ctx.items():
            effective_ctx.setdefault(k, v)

    # מקור 3: nav_payload מה-router (אם נשלח)
    nav_payload = kwargs.get("nav_payload") or st.session_state.get("nav_payload")
    if isinstance(nav_payload, Mapping):
        # נשמור גם raw וגם override רך לשדות בסיסיים
        effective_ctx.setdefault("nav_payload", dict(nav_payload))
        for k in ("env", "profile", "start_date", "end_date"):
            if k in nav_payload and k not in effective_ctx:
                effective_ctx[k] = nav_payload[k]

    # --------------------------------------------------------
    # 2) env/profile חכם + מטא-דאטה בסיסית
    # --------------------------------------------------------
    env = (
        kwargs.get("env")
        or effective_ctx.get("env")
        or os.getenv("DASH_ENV")
        or "dev"
    )
    profile = (
        kwargs.get("profile")
        or effective_ctx.get("profile")
        or os.getenv("DASH_PROFILE")
        or "default"
    )

    env = str(env).lower()
    profile = str(profile).lower()

    effective_ctx["env"] = env
    effective_ctx["profile"] = profile
    effective_ctx["__tab__"] = "portfolio"
    effective_ctx["__tab_run_id__"] = tab_run_id
    effective_ctx["__tab_profile__"] = f"{env}:{profile}"

    # capabilities (אם הדשבורד שומר אותם)
    capabilities = st.session_state.get("dashboard_capabilities")
    if isinstance(capabilities, Mapping):
        effective_ctx.setdefault("capabilities", dict(capabilities))

    # --------------------------------------------------------
    # 3) Telemetry – היסטוריית ריצות לטאב
    # --------------------------------------------------------
    tele_key = "portfolio_tab_telemetry"
    tele_history = st.session_state.get(tele_key)
    if not isinstance(tele_history, list):
        tele_history = []
    run_index = len(tele_history) + 1

    tele_entry: Dict[str, Any] = {
        "run_index": run_index,
        "tab_run_id": tab_run_id,
        "snapshot_id": snapshot_id,
        "env": env,
        "profile": profile,
        "start_ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "duration_ms": None,   # יתעדכן ב-finally
        "ok": True,
        "error_message": None,
    }

    # --------------------------------------------------------
    # 4) User prefs לטאב: טעינה/עדכון
    # --------------------------------------------------------
    prefs_key = "portfolio_tab_prefs"
    default_prefs: Dict[str, Any] = {
        "last_view_mode": "Overview",
        "last_date_range_mode": "today",
        "show_demo_mode": True,
        "preferred_risk_view": "Risk & Health",
        "preferred_language": "he",
        "show_runtime_banner": True,
    }
    prefs = st.session_state.get(prefs_key)
    if not isinstance(prefs, Mapping):
        prefs = default_prefs.copy()
    else:
        merged = default_prefs.copy()
        merged.update(dict(prefs))
        prefs = merged

    # demo mode חכם לפי env
    if env in ("dev", "research") and "show_demo_mode" in prefs:
        # משאירים כפי שהמשתמש הגדיר (ברירת מחדל True)
        pass
    elif env in ("prod", "live"):
        # לא נוגעים ב-show_demo_mode, רק לא מכריחים דמו
        pass

    effective_ctx["portfolio_prefs"] = prefs
    st.session_state[prefs_key] = prefs

    # --------------------------------------------------------
    # 5) Demo Mode Override → PortfolioConfig דרך ctx
    # --------------------------------------------------------
    if prefs.get("show_demo_mode", True) and env in ("dev", "research"):
        tab_cfg = effective_ctx.get("portfolio_tab_config") or {}
        if not isinstance(tab_cfg, Mapping):
            tab_cfg = {}
        tab_cfg = dict(tab_cfg)
        tab_cfg.setdefault("allow_demo_mode", True)
        effective_ctx["portfolio_tab_config"] = tab_cfg

    # --------------------------------------------------------
    # 6) Focus universe integration (Smart Scan / Optimisation)
    # --------------------------------------------------------
    focus_universe: List[str] = []
    try:
        # עדיפות ראשונה: Shortlist מה-Smart Scan
        shortlist_df = st.session_state.get("smart_scan_shortlist")
        if isinstance(shortlist_df, pd.DataFrame) and "pair" in shortlist_df.columns:
            focus_universe = (
                shortlist_df["pair"].astype(str).dropna().unique().tolist()
            )

        # אם עדיין ריק → opt_batch_pairs
        if not focus_universe:
            batch_pairs = st.session_state.get("opt_batch_pairs")
            if isinstance(batch_pairs, list):
                focus_universe = [str(x) for x in batch_pairs if x]
    except Exception:
        focus_universe = []

    if focus_universe:
        effective_ctx["portfolio_focus_universe"] = focus_universe

    # --------------------------------------------------------
    # 7) Risk budget overlay
    # --------------------------------------------------------
    risk_cap = None
    try:
        if isinstance(effective_ctx.get("risk_capital"), Mapping):
            risk_cap = dict(effective_ctx["risk_capital"])
        elif isinstance(st.session_state.get("risk_capital"), Mapping):
            risk_cap = dict(st.session_state["risk_capital"])
    except Exception:
        risk_cap = None

    if risk_cap is not None:
        portfolio_risk_budget = {
            "capital": risk_cap.get("capital"),
            "max_exposure_per_trade": risk_cap.get("max_exposure_per_trade"),
            "max_leverage": risk_cap.get("max_leverage"),
            "sleeve_limits": risk_cap.get("sleeve_limits"),
            "risk_profile": risk_cap.get("risk_profile"),
        }
        effective_ctx["portfolio_risk_budget"] = portfolio_risk_budget
    else:
        portfolio_risk_budget = None

    # --------------------------------------------------------
    # 8) nav_target Integration (read-only + flag)
    # --------------------------------------------------------
    nav_target = st.session_state.get("nav_target")
    if isinstance(nav_target, str) and nav_target.lower() == "portfolio":
        effective_ctx["nav_target_consumed"] = True

    # --------------------------------------------------------
    # 9) Engine status / capabilities
    # --------------------------------------------------------
    engine_status = effective_ctx.get("engine_status") or st.session_state.get("engine_status")
    if isinstance(engine_status, Mapping):
        effective_ctx["engine_status"] = dict(engine_status)

    # --------------------------------------------------------
    # 10) Meta object לטאב – שימוש עתידי ל-SQL/Agents
    # --------------------------------------------------------
    account_info = effective_ctx.get("account", {})
    account_id = None
    equity_preview = None
    if isinstance(account_info, Mapping):
        account_id = account_info.get("account_id") or account_info.get("id")
        equity_preview = account_info.get("equity")

    try:
        user_name = getpass.getuser()
    except Exception:
        user_name = os.getenv("USERNAME") or os.getenv("USER") or "unknown"

    host_name = socket.gethostname()

    # health flags בסיסיים
    health_flags: List[str] = []
    if account_id is None:
        health_flags.append("no_account")
    try:
        if equity_preview is not None and float(equity_preview) < 10_000:
            health_flags.append("low_equity")
    except Exception:
        pass
    if prefs.get("show_demo_mode", False) and env not in ("prod", "live"):
        health_flags.append("demo_mode")

    # refresh hint
    if env in ("prod", "live"):
        refresh_hint_sec = 15
    elif env in ("dev", "research"):
        refresh_hint_sec = 45
    else:
        refresh_hint_sec = 60

    meta: Dict[str, Any] = {
        "tab_run_id": tab_run_id,
        "snapshot_id": snapshot_id,
        "run_index": run_index,
        "env": env,
        "profile": profile,
        "host": host_name,
        "user": user_name,
        "account_id": account_id,
        "equity_ctx": equity_preview,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "capabilities": dict(capabilities) if isinstance(capabilities, Mapping) else None,
        "engine_status": dict(engine_status) if isinstance(engine_status, Mapping) else None,
        "health_flags": health_flags,
        "refresh_hint_sec": refresh_hint_sec,
        "hooks": {
            "sql_log_ready": True,
            "agent_portfolio_view": True,
            "dashboard_export_ready": True,
        },
    }
    effective_ctx["portfolio_tab_meta"] = meta

    # sql_section hint
    effective_ctx["sql_section"] = "portfolio_live" if env in ("prod", "live") else "portfolio_dev"

    # נשמור את ctx האפקטיבי ל-debug עתידי
    st.session_state["portfolio_ctx_effective"] = effective_ctx

    # Snapshot אחרון (ל־Agents/Exports)
    st.session_state["portfolio_last_snapshot"] = {
        "meta": meta,
        "prefs": prefs,
        "risk_budget": portfolio_risk_budget,
        "focus_universe": focus_universe,
        "env": env,
        "profile": profile,
    }

    # --------------------------------------------------------
    # 11) Banner עליון קטן – מידע Runtime
    # --------------------------------------------------------
    if prefs.get("show_runtime_banner", True):
        env_label = env.upper()
        profile_label = profile.upper()
        st.caption(
            f"**Portfolio Tab Runtime** | env=`{env_label}` | profile=`{profile_label}` | "
            f"range=`{start_date} → {end_date}` | run_id=`{tab_run_id}` (#{run_index}) | snapshot=`{snapshot_id}`"
        )

    # לוג קצר על התחלה
    try:
        logger.info(
            "render_portfolio_tab: start (run_id=%s, idx=%d, env=%s, profile=%s, equity_ctx=%s)",
            tab_run_id,
            run_index,
            env,
            profile,
            equity_preview,
        )
    except Exception:
        pass

    # --------------------------------------------------------
    # 12) הקריאה בפועל ל-tab live + טיפול חריגים
    # --------------------------------------------------------
    error_obj: Optional[BaseException] = None
    try:
        render_portfolio_tab_live(
            start_date=start_date,
            end_date=end_date,
            ctx=effective_ctx,
            **kwargs,
        )
    except Exception as exc:  # pragma: no cover
        error_obj = exc
        tele_entry["ok"] = False
        tele_entry["error_message"] = str(exc)

        logger.exception(
            "render_portfolio_tab: failed (run_id=%s, env=%s, profile=%s)",
            tab_run_id,
            env,
            profile,
        )
        st.error(
            "אירעה שגיאה בעת טעינת טאב הפרוטפוליו.\n\n"
            "הטאב ממשיך להיות זמין, אבל ייתכן שחלק מהפאנלים לא הוצגו."
        )

        debug_flag = (
            os.getenv("APP_DEBUG", "0") == "1"
            or bool(effective_ctx.get("debug_dashboard", False))
        )
        if debug_flag:
            with st.expander("פרטי שגיאה (Debug)", expanded=False):
                st.exception(exc)
    finally:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        tele_entry["duration_ms"] = duration_ms
        tele_history.append(tele_entry)
        st.session_state[tele_key] = tele_history

        try:
            logger.info(
                "render_portfolio_tab: completed in %.1f ms (run_id=%s, idx=%d, env=%s, profile=%s, ok=%s)",
                duration_ms,
                tab_run_id,
                run_index,
                env,
                profile,
                tele_entry["ok"],
            )
        except Exception:
            pass
