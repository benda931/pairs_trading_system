# -*- coding: utf-8 -*-
"""
root/dashboard_service_factory.py — HF-grade Dashboard Service Factory
======================================================================

מטרה:
------
Factory חכם ומורחב שמספק:

1. DashboardService
   • מחובר ל-AppContext, SqlStore ולכל המודולים הקריטיים (Risk, Signals, Data).

2. DashboardContext
   • מלא בפרופיל סיכון, Universe, Benchmark, Time Horizon, Data Sources.
   • מותאם לסוג הסביבה (dev/paper/live/research/backtest).
   • מותאם ל-Profile (monitoring/trading/research/risk/macro).

3. bootstrap_dashboard
   • פונקציה נוחה שמחזירה גם Service וגם Context.
   • מתאימה גם ל-Web (Streamlit), גם ל-Desktop וגם ל-API.

4. describe_dashboard_bootstrap  (חדש)
   • מחזירה snapshot קל של ה-bootstrap (לבדיקות/Agents) בלי להחזיק אובייקטים כבדים.

שכבה זו:
---------
- אינה תלויה ב-Streamlit.
- מדברת רק עם:
    core.app_context.AppContext
    core.sql_store.SqlStore
    core.dashboard_models.DashboardContext
    core.dashboard_service.DashboardService
"""

from __future__ import annotations

import logging
import os
import socket
import getpass
from datetime import date, timedelta
from typing import Optional, Tuple, Dict, Any, Mapping, MutableMapping, TYPE_CHECKING

from core.app_context import AppContext
from core.sql_store import SqlStore
from core.dashboard_models import DashboardContext

if TYPE_CHECKING:
    # רק לטייפ-הינטס / IDE, לא רץ בזמן אמת → לא יוצר לולאה
    from core.dashboard_service import DashboardService  # pragma: no cover

# ---------------------------------------------------------------------------
# Risk Engine integration (type-safe)
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    # טייפ לצורך type hints בלבד
    from core.risk_engine import RiskLimits as RiskLimitsType  # pragma: no cover
else:
    # בזמן ריצה זה רק placeholder; לא מעניין את פייתון, רק את הטייפצ'קר
    class RiskLimitsType:  # pragma: no cover
        ...
try:
    # מחלקות / פונקציות אמיתיות בזמן ריצה
    from core.risk_engine import (  # type: ignore[import]
        RiskLimits as RiskLimitsRuntime,
        risk_assessment_to_dashboard_dict,
        update_app_context_with_risk,
        get_risk_events_dataframe,
    )
    _HAS_RISK_ENGINE = True
except Exception:
    RiskLimitsRuntime = None  # type: ignore[assignment]
    risk_assessment_to_dashboard_dict = None  # type: ignore[assignment]
    update_app_context_with_risk = None  # type: ignore[assignment]
    get_risk_events_dataframe = None  # type: ignore[assignment]
    _HAS_RISK_ENGINE = False

logger = logging.getLogger(__name__)


# ============================================================================
# 0) Internal helpers — service wiring & meta
# ============================================================================


def _ensure_services_mapping(app_ctx: AppContext) -> MutableMapping[str, Any]:
    """
    דואג שתהיה על ה-AppContext מפה לוגית של services.

    זוהי נקודת החיבור בין:
    - dashboard_service_factory  (הקובץ הנוכחי)
    - root/dashboard.py          (discover_capabilities מפעיל _discover_services_mapping)

    אם אין attr services → ניצור dict חדש ונחבר אותו.
    """
    services = getattr(app_ctx, "services", None)
    if not isinstance(services, MutableMapping):
        services = {}
        try:
            setattr(app_ctx, "services", services)
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed to attach services mapping to AppContext: %s", exc)
    return services


def _attach_service(
    app_ctx: AppContext,
    key: str,
    obj: Any,
    *,
    also_attr_names: Tuple[str, ...] = (),
) -> None:
    """
    מחבר שירות ל-AppContext בצורה אחידה:

    - app_ctx.services[key] = obj
    - לכל שם ב-also_attr_names שאין עליו attr → setattr(app_ctx, name, obj)

    שימוש:
        _attach_service(app_ctx, "sql_store", store, also_attr_names=("store", "db", "sql_store"))
    """
    if obj is None:
        return

    try:
        services = _ensure_services_mapping(app_ctx)
        if key not in services:
            services[key] = obj
        # שמות נוספים (קונבנציה ל-discover_capabilities)
        for name in also_attr_names:
            if not hasattr(app_ctx, name):
                setattr(app_ctx, name, obj)
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to attach service '%s' to AppContext: %s", key, exc)


def _resolve_user_and_host() -> Tuple[str, str]:
    """Helper קטן להוצאת user/host בצורה בטוחה."""
    host = socket.gethostname()
    try:
        user = getpass.getuser()
    except Exception:
        user = os.getenv("USERNAME") or os.getenv("USER") or "unknown"
    return user, host


# ============================================================================
# 1) Env / Profile detection
# ============================================================================


def _detect_env(explicit_env: Optional[str] = None, app_ctx: Optional[AppContext] = None) -> str:
    """
    זיהוי env ("dev" / "stage" / "prod" / "live" / "paper" / "research" / "backtest").

    סדר עדיפויות:
    1. explicit_env
    2. משתנה סביבה DASH_ENV
    3. settings.env מתוך AppContext אם קיים
    4. dev (ברירת מחדל)
    """
    if explicit_env:
        return explicit_env.lower()

    env = os.getenv("DASH_ENV")
    if env:
        return env.lower()

    if app_ctx is not None:
        try:
            settings = getattr(app_ctx, "settings", None)
            if settings is not None and getattr(settings, "env", None):
                return str(settings.env).lower()
        except Exception:
            pass

    return "dev"


def _detect_profile(explicit_profile: Optional[str] = None) -> str:
    """
    זיהוי profile ("monitoring" / "trading" / "research" / "risk" / "macro" / ...).

    סדר עדיפויות:
    1. explicit_profile
    2. DASH_PROFILE
    3. "monitoring"
    """
    if explicit_profile:
        return explicit_profile.lower()

    prof = os.getenv("DASH_PROFILE")
    if prof:
        return prof.lower()

    return "monitoring"


def _default_benchmark_for_profile(profile: str) -> str:
    """Benchmark ברירת מחדל לפי profile."""
    profile = profile.lower()
    if profile in ("trading", "live", "prop", "intraday"):
        return "SPY"
    if profile in ("tech", "growth"):
        return "QQQ"
    if profile in ("rates", "macro", "fixed_income"):
        return "IEF"
    if profile in ("global", "multi_asset"):
        return "VT"
    if profile in ("em", "emerging"):
        return "EEM"
    return "SPY"


def _default_portfolio_id_for_profile(profile: str) -> str:
    """portfolio_id ברירת מחדל לפי profile."""
    profile = profile.lower()
    if profile in ("trading", "live"):
        return "live_trading"
    if profile in ("paper", "papertrading", "demo"):
        return "paper_trading"
    if profile in ("research", "backtest"):
        return "research"
    if profile in ("risk", "macro"):
        return "risk_monitor"
    return "default"


def _default_ui_mode_for_profile(profile: str) -> str:
    """מצב UI ברירת מחדל לפי profile."""
    profile = profile.lower()
    if profile in ("trading", "risk"):
        return "advanced"
    if profile in ("research", "macro"):
        return "research"
    return "simple"


def _default_risk_mode_for_profile(profile: str) -> str:
    """
    Risk mode לוגי לפי profile.
    מתורגם אחר כך ל-risk_profile ולפרמטרי סיכון אמיתיים.
    """
    profile = profile.lower()
    if profile in ("trading", "prop", "intraday"):
        return "aggressive"
    if profile in ("risk", "macro"):
        return "conservative"
    return "balanced"


def _default_data_latency_for_env(env: str) -> str:
    """מצב latency של הדאטה לפי env (realtime / delayed / end_of_day)."""
    env = env.lower()
    if env in ("prod", "trading", "live"):
        return "realtime"
    if env in ("stage", "uat", "paper"):
        return "delayed"
    return "end_of_day"


# ============================================================================
# 2) Date range helpers
# ============================================================================


def _infer_date_range(
    mode: str,
    *,
    today: Optional[date] = None,
    custom_start: Optional[date] = None,
    custom_end: Optional[date] = None,
) -> Tuple[date, date]:
    """
    מחזיר (start_date, end_date) לפי mode:

    modes נתמכים:
        "today"       – יום נוכחי
        "ytd"         – מתחילת השנה
        "mtd"         – מתחילת החודש
        "1w" / "5d"   – שבוע/5 ימים
        "1m"          – חודש אחורה
        "3m"          – 3 חודשים אחורה
        "6m"          – 6 חודשים אחורה
        "1y"          – שנה אחורה
        "custom"      – משתמש ב-custom_start/custom_end (עם fallback ל-1y)
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
        start = today - timedelta(days=5)
        return start, today

    if mode == "1m":
        start = today - timedelta(days=30)
        return start, today

    if mode == "3m":
        start = today - timedelta(days=90)
        return start, today

    if mode == "6m":
        start = today - timedelta(days=180)
        return start, today

    if mode == "1y":
        start = today - timedelta(days=365)
        return start, today

    if mode == "custom":
        if custom_start is not None and custom_end is not None:
            return custom_start, custom_end
        # fallback: 1y
        start = today - timedelta(days=365)
        return start, today

    # ברירת מחדל: היום עצמו
    return today, today


# ============================================================================
# 3) SqlStore creation & feature flags
# ============================================================================


def _create_sql_store_for_env(app_ctx: AppContext, env: str) -> SqlStore:
    """
    יוצר SqlStore בהתאם לסביבה.

    כרגע:
    -------
    - משתמש ב-SqlStore.from_settings(app_ctx.settings).
    - אם יש פונקציה set_env / set_schema ב-SqlStore, נשתמש בה.
    """
    store = SqlStore.from_settings(app_ctx.settings)

    # Hook עדין לשינוי סכמה/סביבה בתוך SqlStore
    try:
        if hasattr(store, "set_env"):
            store.set_env(env)
    except Exception:
        pass

    try:
        if getattr(app_ctx.settings, "sql_schema", None) and hasattr(store, "set_schema"):
            store.set_schema(app_ctx.settings.sql_schema)  # type: ignore[attr-defined]
    except Exception:
        pass

    return store


def _compute_feature_flags(env: str, profile: str) -> Dict[str, bool]:
    """
    מחזיר dict עם feature flags ברמת דשבורד.

    התפקיד:
    --------
    • לסמן ל-UI ולטאבים מה צפוי להיות זמין.
    • זה לא במקום feature_flags של root/dashboard.py, אלא metadata נוסף.
    """
    env = env.lower()
    profile = profile.lower()

    flags: Dict[str, bool] = {
        "enable_experimental_tabs": False,
        "enable_risk_tab": False,
        "enable_live_trading_actions": False,
        "show_debug_info": False,
        "enable_macro_tab": False,
        "enable_insights_tab": True,
        "enable_sql_persistence": True,
        "enable_desktop_integration": False,
        "enable_agents": False,
        "enable_matrix_research": True,
    }

    if env in ("dev", "research"):
        flags["enable_experimental_tabs"] = True
        flags["show_debug_info"] = True

    if profile in ("risk", "macro"):
        flags["enable_risk_tab"] = True
        flags["enable_macro_tab"] = True

    if env in ("prod", "live") and profile in ("trading", "live"):
        flags["enable_live_trading_actions"] = True

    if env in ("backtest",):
        # ב-backtest לא תמיד רוצים SQL persistence אמיתי
        flags["enable_sql_persistence"] = False

    # agents/desktop — פרופילים עתידיים (אפשר לעדכן לפי settings)
    if env in ("dev", "research") and profile in ("monitoring", "trading", "research"):
        flags["enable_agents"] = True
    if os.getenv("DESKTOP_INTEGRATION", "").lower() in ("1", "true", "yes"):
        flags["enable_desktop_integration"] = True

    return flags


# ============================================================================
# 4) Risk presets & profile-specific overrides
# ============================================================================


def _apply_risk_preset(ctx: DashboardContext, risk_profile: str, *, env: str) -> None:
    """
    ממלא target_vol, max_vol, drawdown limits, leverage וכו' לפי risk_profile.

    risk_profile:
        "conservative" / "balanced" / "aggressive"
    """
    risk_profile = (risk_profile or "balanced").lower()
    env = env.lower()

    # ברירות מחדל עדינות
    target_vol = 0.10
    max_vol = 0.20
    soft_dd = 0.08
    hard_dd = 0.15
    max_lev = 2.0

    if risk_profile == "conservative":
        target_vol = 0.06
        max_vol = 0.12
        soft_dd = 0.05
        hard_dd = 0.10
        max_lev = 1.5
    elif risk_profile == "aggressive":
        target_vol = 0.18
        max_vol = 0.35
        soft_dd = 0.15
        hard_dd = 0.25
        max_lev = 4.0
    else:  # balanced
        target_vol = 0.12
        max_vol = 0.25
        soft_dd = 0.10
        hard_dd = 0.20
        max_lev = 3.0

    # הקלה קלה ב-dev/research
    if env in ("dev", "research"):
        hard_dd = min(0.30, hard_dd + 0.05)

    ctx.risk_profile = risk_profile
    ctx.target_vol_annual = target_vol
    ctx.max_vol_annual = max_vol
    ctx.drawdown_soft_limit = soft_dd
    ctx.drawdown_hard_limit = hard_dd
    ctx.max_leverage = max_lev

    # max_single_position_weight / max_sector_exposure דיפולטים הגיוניים
    ctx.max_single_position_weight = 0.10
    ctx.max_sector_exposure = 0.30

    # ⚙️ רעיון 1 — לשמור גם ב-extra בצורה מובנית
    ctx.extra.setdefault("risk_limits", {})
    ctx.extra["risk_limits"].update(
        {
            "risk_profile": risk_profile,
            "target_vol_annual": target_vol,
            "max_vol_annual": max_vol,
            "drawdown_soft_limit": soft_dd,
            "drawdown_hard_limit": hard_dd,
            "max_leverage": max_lev,
            "max_single_position_weight": ctx.max_single_position_weight,
            "max_sector_exposure": ctx.max_sector_exposure,
        }
    )


def _apply_profile_overrides(ctx: DashboardContext, profile: str) -> None:
    """
    Overrides עדינים לפי profile:
    - trading: יותר live, טיימפריים קטן.
    - research: יותר SQL/cache, תקופות ארוכות.
    - risk/macro: דגש על daily, לא intraday.
    """
    profile = profile.lower()

    if profile in ("trading", "live", "intraday"):
        ctx.intraday = True
        ctx.bar_size = "5min"
        ctx.use_live_data = True
        ctx.use_sql_cache = True
        ctx.rebalance_frequency = "intraday"
        ctx.ui_mode = "advanced"
        ctx.top_signals_limit = 50

    elif profile in ("research", "backtest"):
        ctx.intraday = False
        ctx.bar_size = "1D"
        ctx.use_live_data = False
        ctx.use_sql_cache = True
        ctx.rebalance_frequency = "daily"
        ctx.ui_mode = "research"
        ctx.top_signals_limit = 100

    elif profile in ("risk", "macro"):
        ctx.intraday = False
        ctx.bar_size = "1D"
        ctx.use_live_data = False
        ctx.use_sql_cache = True
        ctx.rebalance_frequency = "daily"
        ctx.ui_mode = "advanced"
        ctx.top_signals_limit = 30

    # ⚙️ רעיון 2 — לשמור פרופיל UI ל-extra
    ctx.extra.setdefault("ui_profile", {})
    ctx.extra["ui_profile"].update(
        {
            "profile": profile,
            "ui_mode": ctx.ui_mode,
            "intraday": getattr(ctx, "intraday", False),
            "rebalance_frequency": getattr(ctx, "rebalance_frequency", None),
            "top_signals_limit": getattr(ctx, "top_signals_limit", None),
        }
    )


def _apply_env_overrides(ctx: DashboardContext, env: str) -> None:
    """
    התאמות לפי env:
    - dev: debug, יותר ניסיוני.
    - prod/live: יותר strict, data_quality_checks פעיל.
    - backtest: ללא live data בכלל.
    """
    env = env.lower()

    if env in ("dev", "research"):
        ctx.enable_data_quality_checks = True
        ctx.show_experimental_panels = True
        ctx.use_sql_cache = True

    if env in ("prod", "live"):
        ctx.enable_data_quality_checks = True
        ctx.show_experimental_panels = False
        ctx.use_live_data = True

    if env in ("backtest",):
        ctx.use_live_data = False
        ctx.use_sql_cache = True
        # אולי לא צריך Kill switch ב-backtest:
        ctx.kill_switch_enabled = False

    # ⚙️ רעיון 3 — לשמור “data_profile” ב-extra
    ctx.extra.setdefault("data_profile", {})
    ctx.extra["data_profile"].update(
        {
            "env": env,
            "use_live_data": getattr(ctx, "use_live_data", False),
            "use_sql_cache": getattr(ctx, "use_sql_cache", False),
            "enable_data_quality_checks": getattr(ctx, "enable_data_quality_checks", False),
            "show_experimental_panels": getattr(ctx, "show_experimental_panels", False),
        }
    )


# ============================================================================
# 5) Settings / AppContext integration
# ============================================================================


def _resolve_base_currency_and_timezone(
    app_ctx: Optional[AppContext],
    base_currency: Optional[str],
    timezone: Optional[str],
) -> Tuple[str, str]:
    """מנסה למשוך base_currency ו-timezone מ-AppContext.settings, אחרת ברירת מחדל."""
    if base_currency is None:
        base_currency = "USD"
    if timezone is None:
        timezone = "Asia/Jerusalem"

    if app_ctx is None:
        return base_currency, timezone

    settings = getattr(app_ctx, "settings", None)
    if settings is None:
        return base_currency, timezone

    # base_currency
    try:
        if base_currency == "USD" and getattr(settings, "base_currency", None):
            base_currency = str(settings.base_currency)
    except Exception:
        pass

    # timezone
    try:
        if timezone == "Asia/Jerusalem" and getattr(settings, "timezone", None):
            timezone = str(settings.timezone)
    except Exception:
        pass

    return base_currency, timezone


def _resolve_universe_and_strategy(
    app_ctx: Optional[AppContext], profile: str
) -> Tuple[str, int, str, str]:
    """
    מנסה להחליט universe_name, universe_size_limit, strategy_family, sub_strategy
    מתוך settings, ואם אין – לפי profile.
    """
    universe_name = "default"
    universe_size_limit = 500
    strategy_family = "pairs_trading"
    sub_strategy = "mean_reversion"

    if app_ctx is not None:
        settings = getattr(app_ctx, "settings", None)
        try:
            if settings is not None:
                if getattr(settings, "universe_name", None):
                    universe_name = str(settings.universe_name)
                if getattr(settings, "universe_size_limit", None):
                    universe_size_limit = int(settings.universe_size_limit)
                if getattr(settings, "strategy_family", None):
                    strategy_family = str(settings.strategy_family)
                if getattr(settings, "sub_strategy", None):
                    sub_strategy = str(settings.sub_strategy)
        except Exception:
            pass

    # Override עדינים לפי profile
    p = profile.lower()
    if p in ("macro", "risk"):
        strategy_family = "macro_overlay"
    elif p in ("research",):
        # נניח שמשתמשים יותר ב-stat arb / research-mode
        strategy_family = strategy_family or "stat_arb"

    return universe_name, universe_size_limit, strategy_family, sub_strategy

def build_base_risk_limits_for_ctx(ctx: DashboardContext) -> Optional[RiskLimitsType]:

    """
    גוזר RiskLimits בסיסי מתוך DashboardContext:

    - משתמש ב:
        ctx.risk_profile
        ctx.target_vol_annual
        ctx.drawdown_hard_limit
        ctx.max_leverage
    במידת האפשר.
    """
    if not _HAS_RISK_ENGINE or RiskLimitsRuntime is None:
        return None

    limits = RiskLimitsRuntime()

    # 1) פרופיל סיכון הגיוני לפי ctx.risk_profile
    rp = (getattr(ctx, "risk_profile", None) or "balanced").lower()
    if rp in ("conservative", "low"):
        limits.max_daily_loss_pct = 0.02
        limits.max_weekly_loss_pct = 0.04
        limits.max_monthly_loss_pct = 0.08
        limits.max_drawdown_pct = 0.15
        limits.max_gross_leverage = 1.5
        limits.max_net_leverage = 1.5
        limits.profile_name = "conservative"
    elif rp in ("aggressive", "high"):
        limits.max_daily_loss_pct = 0.04
        limits.max_weekly_loss_pct = 0.08
        limits.max_monthly_loss_pct = 0.16
        limits.max_drawdown_pct = 0.25
        limits.max_gross_leverage = 4.0
        limits.max_net_leverage = 3.0
        limits.profile_name = "aggressive"
    else:
        # balanced (ברירת מחדל)
        limits.max_daily_loss_pct = 0.03
        limits.max_weekly_loss_pct = 0.06
        limits.max_monthly_loss_pct = 0.12
        limits.max_drawdown_pct = 0.20
        limits.max_gross_leverage = 3.0
        limits.max_net_leverage = 2.0
        limits.profile_name = "balanced"

    # 2) התאמה ליעדי Vol / DD מתוך DashboardContext אם קיימים
    try:
        if getattr(ctx, "target_vol_annual", None) is not None:
            limits.target_vol_pct = float(ctx.target_vol_annual)

        if getattr(ctx, "drawdown_hard_limit", None) is not None:
            limits.max_drawdown_pct = float(abs(ctx.drawdown_hard_limit))

        if getattr(ctx, "max_leverage", None) is not None:
            lev = float(ctx.max_leverage)
            limits.max_gross_leverage = max(limits.min_gross_leverage, lev)
            limits.max_net_leverage = max(limits.min_gross_leverage, lev * 0.8)
    except Exception:
        # לא מפילים את הפונקציה על שגיאה קטנה
        logger.exception("build_base_risk_limits_for_ctx: failed to align limits from ctx")

    return limits

# ============================================================================
# 6) Public factory: DashboardService
# ============================================================================


def create_dashboard_service(
    app_ctx: Optional[AppContext] = None,
    *,
    sql_store: Optional[SqlStore] = None,
    env: Optional[str] = None,
    profile: Optional[str] = None,
    enable_persistence: Optional[bool] = None,
) -> "DashboardService":
    """
    יוצר DashboardService ברמת קרן גידור.

    תאימות אחורה:
    --------------
    create_dashboard_service() בלי ארגומנטים עדיין עובד.

    שדרוגים:
    --------
    1. אם app_ctx קיים – מחבר את SqlStore ו-DashboardService ל-AppContext.services
       וגם כ-attribs ישירים (sql_store, dashboard_service).
    2. זה מאפשר ל-root/dashboard.py לגלות capabilities בצורה אוטומטית:
        - sql_store=True
        - dashboard_service=True
    """
    # import לוקלי כדי לשבור circular imports
    from core.dashboard_service import DashboardService as _DashboardService

    # 1. AppContext
    if app_ctx is None:
        app_ctx = AppContext.get_global()

    # 2. env/profile
    env_norm = _detect_env(env, app_ctx=app_ctx)
    profile_norm = _detect_profile(profile)

    # 3. SqlStore
    if sql_store is None:
        sql_store = _create_sql_store_for_env(app_ctx, env_norm)

    # 4. enable_persistence
    if enable_persistence is None:
        enable_persistence = env_norm not in ("backtest",)

    # 5. יצירת DashboardService
    service = _DashboardService(
        app_ctx,
        sql_store=sql_store,
        enable_persistence=enable_persistence,
    )

    # 6. Wiring ל-AppContext (כדי שה-capabilities יתעדכנו יפה)
    try:
        _attach_service(
            app_ctx,
            "sql_store",
            sql_store,
            also_attr_names=("sql_store", "store", "db", "sql"),
        )
        _attach_service(
            app_ctx,
            "dashboard_service",
            service,
            also_attr_names=("dashboard_service", "dashboard", "dashboard_facade"),
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("create_dashboard_service: failed to wire services into AppContext: %s", exc)

    # לוגינג למעקב
    try:
        app_ctx.logger.info(  # type: ignore[attr-defined]
            "DashboardService created (env=%s, profile=%s, persistence=%s)",
            env_norm,
            profile_norm,
            enable_persistence,
        )
    except Exception:
        logger.info(
            "DashboardService created (env=%s, profile=%s, persistence=%s)",
            env_norm,
            profile_norm,
            enable_persistence,
        )

    return service


# ============================================================================
# 7) Public factory: DashboardContext
# ============================================================================


def build_default_dashboard_context(
    *,
    env: Optional[str] = None,
    profile: Optional[str] = None,
    date_range_mode: str = "today",
    benchmark: Optional[str] = None,
    portfolio_id: Optional[str] = None,
    ui_mode: Optional[str] = None,
    risk_mode: Optional[str] = None,              # "conservative" / "balanced" / "aggressive"
    data_latency_mode: Optional[str] = None,      # "realtime" / "delayed" / "end_of_day"
    base_currency: Optional[str] = None,
    timezone: Optional[str] = None,
    custom_start: Optional[date] = None,
    custom_end: Optional[date] = None,
    app_ctx: Optional[AppContext] = None,
) -> DashboardContext:
    """
    בונה DashboardContext "חכם" עם ברירות מחדל בהתאם ל-env/profile.

    שדות שנקבעים:
    -------------
    • env, profile, ui_mode, benchmark, portfolio_id
    • start_date, end_date, bar_size, intraday
    • base_currency, timezone
    • risk_profile + מגבלות סיכון (ע"י _apply_risk_preset)
    • universe_name, universe_size_limit, strategy_family, sub_strategy
    • price_source, fundamental_source, macro_source
    • use_live_data / use_sql_cache בהתאם ל-data_latency_mode/env/profile
    • extra: feature_flags, host, user, app_version, sql_schema, risk_limits,
             ui_profile, data_profile, universe_profile, runtime_meta, וכו'.
    """
    # AppContext (אם לא נשלח מבחוץ)
    if app_ctx is None:
        try:
            app_ctx = AppContext.get_global()
        except Exception:
            app_ctx = None

    # env/profile
    env_norm = _detect_env(env, app_ctx=app_ctx)
    profile_norm = _detect_profile(profile)

    # טווח תאריכים
    start_date, end_date = _infer_date_range(
        date_range_mode, custom_start=custom_start, custom_end=custom_end
    )

    # benchmark / portfolio
    if benchmark is None:
        benchmark = _default_benchmark_for_profile(profile_norm)
    if portfolio_id is None:
        portfolio_id = _default_portfolio_id_for_profile(profile_norm)

    # ui_mode
    if ui_mode is None:
        ui_mode = _default_ui_mode_for_profile(profile_norm)

    # risk_mode → risk_profile
    if risk_mode is None:
        risk_mode = _default_risk_mode_for_profile(profile_norm)
    risk_profile = risk_mode

    # data_latency_mode
    if data_latency_mode is None:
        data_latency_mode = _default_data_latency_for_env(env_norm)

    # base_currency & timezone
    base_currency, timezone = _resolve_base_currency_and_timezone(
        app_ctx, base_currency, timezone
    )

    # universe & strategy
    universe_name, universe_size_limit, strategy_family, sub_strategy = _resolve_universe_and_strategy(
        app_ctx, profile_norm
    )

    # Feature Flags
    feature_flags = _compute_feature_flags(env_norm, profile_norm)

    # use_live_data לפי data_latency_mode
    use_live_data = data_latency_mode == "realtime"

    # price_source/fundamental_source/macro_source – ניתן לשפר לפי settings
    price_source = "ibkr" if env_norm in ("prod", "live") else "sql"
    fundamental_source = "sql"
    macro_source = "sql"

    # בניית DashboardContext בסיסי
    ctx = DashboardContext(
        start_date=start_date,
        end_date=end_date,
        env=env_norm,
        profile=profile_norm,
        ui_mode=ui_mode,
        benchmark=benchmark,
        portfolio_id=portfolio_id,
        base_currency=base_currency,
        timezone=timezone,
        universe_name=universe_name,
        universe_size_limit=universe_size_limit,
        strategy_family=strategy_family,
        sub_strategy=sub_strategy,
        price_source=price_source,
        fundamental_source=fundamental_source,
        macro_source=macro_source,
        use_live_data=use_live_data,
        use_sql_cache=True,
    )

    # מילוי פרופיל סיכון (יעדים ומגבלות)
    _apply_risk_preset(ctx, risk_profile, env=env_norm)

    # Overrides לפי profile/env
    _apply_profile_overrides(ctx, profile_norm)
    _apply_env_overrides(ctx, env_norm)

    # סימון tags
    ctx.tags.append(f"env:{env_norm}")
    ctx.tags.append(f"profile:{profile_norm}")
    ctx.tags.append(f"risk:{ctx.risk_profile}")

    # extra עשיר למטא דאטה (שימושי ל-SQL / דוחות / Agents)
    user, host = _resolve_user_and_host()

    app_version = None
    sql_schema = None
    if app_ctx is not None:
        settings = getattr(app_ctx, "settings", None)
        if settings is not None:
            try:
                if getattr(settings, "app_version", None):
                    app_version = str(settings.app_version)
            except Exception:
                pass
            try:
                if getattr(settings, "sql_schema", None):
                    sql_schema = str(settings.sql_schema)
            except Exception:
                pass

    # ⚙️ רעיון 4 — universe_profile ל-extra
    ctx.extra.setdefault("universe_profile", {})
    ctx.extra["universe_profile"].update(
        {
            "universe_name": universe_name,
            "universe_size_limit": universe_size_limit,
            "strategy_family": strategy_family,
            "sub_strategy": sub_strategy,
        }
    )

    # ⚙️ רעיון 5 — runtime_meta ל-extra (קל לעיבוד ע"י Agents/דוחות)
    runtime_meta = {
        "env": env_norm,
        "profile": profile_norm,
        "host": host,
        "user": user,
        "app_version": app_version,
        "sql_schema": sql_schema,
        "date_range_mode": date_range_mode,
    }

    # ⚙️ רעיון 6 — data_latency_meta
    data_latency_meta = {
        "data_latency_mode": data_latency_mode,
        "price_source": price_source,
        "fundamental_source": fundamental_source,
        "macro_source": macro_source,
    }

    ctx.extra.update(
        {
            "risk_mode": risk_mode,
            "data_latency_mode": data_latency_mode,
            "feature_flags": feature_flags,
            "runtime_meta": runtime_meta,
            "data_latency_meta": data_latency_meta,
        }
    )

    # ⚙️ רעיון 7 — agents_profile: איזה Agents “מותר” להפעיל בפרופיל הזה
    ctx.extra.setdefault("agents_profile", {})
    ctx.extra["agents_profile"].update(
        {
            "allow_param_tuning": profile_norm in ("dev", "research", "trading"),
            "allow_live_actions": feature_flags.get("enable_live_trading_actions", False),
            "allow_snapshot": True,
            "allow_matrix_research": feature_flags.get("enable_matrix_research", True),
        }
    )

    # ⚙️ רעיון 8 — desktop_profile (לשימוש ע"י Desktop bridge)
    ctx.extra.setdefault("desktop_profile", {})
    ctx.extra["desktop_profile"].update(
        {
            "suggest_desktop_mode": feature_flags.get("enable_desktop_integration", False),
            "preferred_profile_for_desktop": "monitoring",
        }
    )

    # ⚙️ רעיון 9 — persistence_profile: האם מצפים לשמור context ל-SQL
    ctx.extra["persistence_profile"] = {
        "env": env_norm,
        "enable_sql_persistence": feature_flags.get("enable_sql_persistence", True),
    }

    # ⚙️ רעיון 10 — dashboard_flags “שטוחים” (נוח לגרפים/דשבורדים)
    ctx.extra["dashboard_flags"] = {
        "env": env_norm,
        "profile": profile_norm,
        "risk_profile": ctx.risk_profile,
        "ui_mode": ctx.ui_mode,
        "has_sql_store": feature_flags.get("enable_sql_persistence", True),
        "has_risk_tab": feature_flags.get("enable_risk_tab", False),
        "has_macro_tab": feature_flags.get("enable_macro_tab", False),
        "has_agents": feature_flags.get("enable_agents", False),
    }

    return ctx


# ============================================================================
# 8) Convenience bootstrap function
# ============================================================================


def bootstrap_dashboard(
    app_ctx: Optional[AppContext] = None,
    *,
    env: Optional[str] = None,
    profile: Optional[str] = None,
    date_range_mode: str = "today",
    benchmark: Optional[str] = None,
    portfolio_id: Optional[str] = None,
    custom_start: Optional[date] = None,
    custom_end: Optional[date] = None,
) -> tuple["DashboardService", DashboardContext]:
    """
    Helper נוח שמחזיר גם DashboardService וגם DashboardContext בריצה אחת.

    שימוש טיפוסי (Streamlit / Desktop):
    -----------------------------------
        from root.dashboard_service_factory import bootstrap_dashboard

        service, dctx = bootstrap_dashboard(
            env="dev",
            profile="trading",
            date_range_mode="1m",
        )
        snapshot = service.build_dashboard_snapshot(dctx)
    """
    # AppContext
    if app_ctx is None:
        try:
            app_ctx = AppContext.get_global()
        except Exception:
            app_ctx = None

    # Service
    service = create_dashboard_service(
        app_ctx=app_ctx,
        env=env,
        profile=profile,
    )

    # Context
    dash_ctx = build_default_dashboard_context(
        env=env,
        profile=profile,
        date_range_mode=date_range_mode,
        benchmark=benchmark,
        portfolio_id=portfolio_id,
        custom_start=custom_start,
        custom_end=custom_end,
        app_ctx=app_ctx,
    )

    return service, dash_ctx

def run_risk_for_history(
    hist: "Any",  # בד"כ DataFrame מה-Backtester / Portfolio
    dash_ctx: DashboardContext,
    *,
    equity_col: str = "Equity",
    pnl_col: str = "PnL",
    date_col: str | None = None,
    bucket_col: str | None = None,
    realized_sharpe: float | None = None,
) -> Dict[str, Any]:
    """
    Utility ברמת root שמחבר בין Backtest/Portfolio History לבין RiskEngine.

    שימוש טיפוסי:
        hist = backtest_result.history_df
        risk_dash = run_risk_for_history(hist, dash_ctx, pnl_col="pnl", date_col="date")

    מחזיר:
        risk_dashboard_dict מוכן ל-UI (Home / Risk Tab / Reports).
    """
    import pandas as pd  # import לוקלי כדי לא לשבור תלות במודולים אחרים

    if not _HAS_RISK_ENGINE or RiskLimitsType is None or risk_assessment_to_dashboard_dict is None:
        logger.info("run_risk_for_history: RiskEngine not available; returning empty dict.")
        return {}

    if hist is None:
        return {}
    if not isinstance(hist, pd.DataFrame):
        try:
            hist = pd.DataFrame(hist)
        except Exception:
            logger.warning("run_risk_for_history: history is not a DataFrame and cannot be coerced.")
            return {}

    if hist.empty:
        return {}

    # בונים מגבלות בסיס לפי ה-Context
    base_limits = build_base_risk_limits_for_ctx(dash_ctx)
    if base_limits is None:
        return {}

    # מנסים לבחור date_col אם לא נשלח
    if date_col is None:
        for cand in ("date", "Date", "ts", "timestamp"):
            if cand in hist.columns:
                date_col = cand
                break

    # מנסים לבחור bucket_col (למשל strategy / cluster) אם יש
    if bucket_col is not None and bucket_col not in hist.columns:
        bucket_col = None

    # נרוץ דרך RiskEngine ונקבל dict ל-UI
    risk_dash = risk_assessment_to_dashboard_dict(
        hist,
        base_limits,
        equity_col=equity_col if equity_col in hist.columns else "Equity",
        pnl_col=pnl_col if pnl_col in hist.columns else pnl_col,
        date_col=date_col,
        bucket_col=bucket_col,
        realized_sharpe=realized_sharpe,
    )

    return risk_dash

def run_risk_and_update_app_ctx(
    app_ctx: AppContext,
    hist: "Any",
    dash_ctx: DashboardContext,
    *,
    equity_col: str = "Equity",
    pnl_col: str = "PnL",
    date_col: str | None = None,
    bucket_col: str | None = None,
    realized_sharpe: float | None = None,
) -> tuple[AppContext, Dict[str, Any]]:
    """
    גרסה מתקדמת: מריצה RiskEngine, מעדכנת AppContext דרך apply_policies_to_ctx
    (אם זמין), ומחזירה גם risk_dashboard_dict.

    זה מתאים לזרימה:
        service, dash_ctx = bootstrap_dashboard(...)
        hist = backtester.history_df
        app_ctx, risk_dash = run_risk_and_update_app_ctx(service.app_ctx, hist, dash_ctx)
    """
    import pandas as pd

    if not _HAS_RISK_ENGINE or RiskLimitsType is None or update_app_context_with_risk is None:
        # fallback: נשתמש רק ב-run_risk_for_history ונחזיר את app_ctx כמו שהוא
        return app_ctx, run_risk_for_history(
            hist,
            dash_ctx,
            equity_col=equity_col,
            pnl_col=pnl_col,
            date_col=date_col,
            bucket_col=bucket_col,
            realized_sharpe=realized_sharpe,
        )

    if hist is None:
        return app_ctx, {}
    if not isinstance(hist, pd.DataFrame):
        try:
            hist = pd.DataFrame(hist)
        except Exception:
            logger.warning("run_risk_and_update_app_ctx: history is not a DataFrame and cannot be coerced.")
            return app_ctx, {}

    if hist.empty:
        return app_ctx, {}

    base_limits = build_base_risk_limits_for_ctx(dash_ctx)
    if base_limits is None:
        return app_ctx, {}

    # בוחרים date_col אם לא נשלח
    if date_col is None:
        for cand in ("date", "Date", "ts", "timestamp"):
            if cand in hist.columns:
                date_col = cand
                break

    # breach log ל-window אחרון (למשל 60 דקות) – אם אין, פשוט None
    recent_breach_df = None
    if get_risk_events_dataframe is not None:
        try:
            recent_breach_df = get_risk_events_dataframe(limit=500)
        except Exception:
            recent_breach_df = None

    # נריץ את אינטגרציית ה-RiskEngine עם הקונטקסט
    app_ctx_updated, risk_dash = update_app_context_with_risk(
        ctx=app_ctx,
        hist=hist,
        base_limits=base_limits,
        equity_col=equity_col if equity_col in hist.columns else "Equity",
        pnl_col=pnl_col if pnl_col in hist.columns else pnl_col,
        date_col=date_col,
        bucket_col=bucket_col if bucket_col and bucket_col in hist.columns else None,
        realized_sharpe=realized_sharpe,
        recent_breach_df=recent_breach_df,
        prev_live_readiness=None,
    )

    return app_ctx_updated, risk_dash

# ============================================================================
# 9) Lightweight descriptor for tests / Agents / Desktop
# ============================================================================


def describe_dashboard_bootstrap(
    *,
    env: Optional[str] = None,
    profile: Optional[str] = None,
    date_range_mode: str = "today",
    custom_start: Optional[date] = None,
    custom_end: Optional[date] = None,
    app_ctx: Optional[AppContext] = None,
) -> Dict[str, Any]:
    """
    מחזיר תיאור קל (dict) של ה-bootstrap בלי להחזיק אובייקטים כבדים.

    מיועד ל:
    --------
    - בדיקות יחידה / CI שרוצות לוודא שהדשבורד מוגדר כמו שצריך.
    - Agents/Desktop שרוצים להציץ ב-"meta" של ההגדרות לפני שהם מרימים service אמיתי.

    הפלט:
    -----
        {
            "env": ...,
            "profile": ...,
            "start_date": ...,
            "end_date": ...,
            "benchmark": ...,
            "portfolio_id": ...,
            "risk_profile": ...,
            "feature_flags": {...},
            "extra": {
                "runtime_meta": {...},
                "risk_limits": {...},
                "universe_profile": {...},
                ...
            }
        }
    """
    if app_ctx is None:
        try:
            app_ctx = AppContext.get_global()
        except Exception:
            app_ctx = None

    ctx = build_default_dashboard_context(
        env=env,
        profile=profile,
        date_range_mode=date_range_mode,
        custom_start=custom_start,
        custom_end=custom_end,
        app_ctx=app_ctx,
    )

    desc: Dict[str, Any] = {
        "env": ctx.env,
        "profile": ctx.profile,
        "start_date": ctx.start_date,
        "end_date": ctx.end_date,
        "benchmark": ctx.benchmark,
        "portfolio_id": ctx.portfolio_id,
        "risk_profile": ctx.risk_profile,
        "ui_mode": ctx.ui_mode,
        "base_currency": ctx.base_currency,
        "timezone": ctx.timezone,
        "feature_flags": ctx.extra.get("feature_flags", {}),
        "extra": {
            "risk_limits": ctx.extra.get("risk_limits", {}),
            "runtime_meta": ctx.extra.get("runtime_meta", {}),
            "universe_profile": ctx.extra.get("universe_profile", {}),
            "data_profile": ctx.extra.get("data_profile", {}),
            "data_latency_meta": ctx.extra.get("data_latency_meta", {}),
            "agents_profile": ctx.extra.get("agents_profile", {}),
            "dashboard_flags": ctx.extra.get("dashboard_flags", {}),
        },
    }
    return desc
