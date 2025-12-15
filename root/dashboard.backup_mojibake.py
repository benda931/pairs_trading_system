# -*- coding: utf-8 -*-
"""
root/dashboard.py — Full Pairs Trading Dashboard (HF-grade, v4)
===============================================================

Shell / Router / Orchestrator ראשי לדשבורד ה-Web של מערכת ה-Pairs Trading.

תפקידים מרכזיים:
----------------
- אתחול Streamlit (page_config, layout, theme בסיסי).
- הגדרת App-level metadata (APP_NAME, גרסה, env, profile).
- יצירת לוגרים ברמת אפליקציה (RotatingFileHandler + console).
- הגדרת type aliases ו־Tab/Navigation primitives לשימוש בחלקים הבאים.
"""

from __future__ import annotations

# =====================
# Part 1/35 – Header, imports, logger, page config, metadata, type aliases
# =====================

import getpass
import logging
from logging.handlers import RotatingFileHandler
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Optional,
    List,
    Tuple,
    TYPE_CHECKING,
    TypedDict,
    Literal,
    Protocol,
)

from collections.abc import Mapping, Sequence, Callable

import pandas as pd
import streamlit as st

# TYPE_CHECKING imports בלבד כדי להימנע מ-Circular imports בזמן ריצה
if TYPE_CHECKING:
    from core.app_context import AppContext  # pragma: no cover

# -----------------------------------------------------------------------------
# פרמטרים גלובליים, נתיבי פרויקט ומידע סביבת ריצה
# -----------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
LOGS_DIR: Path = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DASHBOARD_LOGGER_NAME: str = "Dashboard"
DASHBOARD_LOG_PATH: Path = LOGS_DIR / "dashboard_app.log"

APP_NAME: str = "Pairs Trading Dashboard"
APP_ICON: str = "📊"
APP_VERSION: str = "0.4.0-hf"

EnvName = Literal["dev", "live", "paper", "research", "backtest", "staging", "test"]
ProfileName = Literal["trading", "research", "risk", "macro", "monitoring", "default"]

DEFAULT_ENV: EnvName = "dev"
DEFAULT_PROFILE: ProfileName = "trading"

RUNTIME_HOST: str = socket.gethostname()
RUNTIME_USER: str = getpass.getuser()
STARTED_AT_UTC: datetime = datetime.now(timezone.utc)

# -----------------------------------------------------------------------------
# Type aliases כלליים לדשבורד
# -----------------------------------------------------------------------------

FeatureFlags = Dict[str, Any]
TabKey = str
TabLabel = str

NavPayload = Dict[str, Any]


class NavTarget(TypedDict, total=False):
    """
    מבנה הניווט הגלובלי בין טאבים, שנשמר בתוך st.session_state["nav_target"].
    """
    tab_key: TabKey
    payload: NavPayload


ServiceStatus = Dict[str, Any]


class TabRenderer(Protocol):
    """
    חתימה אחידה לכל Renderer של טאב בדשבורד.

    כל טאב ייושם עם פונקציה המתאימה לחתימה:
        def render_some_tab(
            app_ctx: AppContext,
            feature_flags: FeatureFlags,
            nav_payload: Optional[NavPayload] = None,
        ) -> None: ...
    """

    def __call__(
        self,
        app_ctx: "AppContext",
        feature_flags: FeatureFlags,
        nav_payload: Optional[NavPayload] = None,
    ) -> None:
        ...


@dataclass
class TabMeta:
    """
    מטא-דאטה מלא לטאב בדשבורד.

    key:
        מפתח פנימי לטאב (למשל "home", "smart_scan", "backtest").
    label:
        טקסט/אימוג'י שמוצג ב-UI (למשל "🏠 Dashboard").
    renderer:
        פונקציית Renderer המתאימה ל-TabRenderer.
    group:
        קבוצה לוגית ("core", "research", "risk", "macro", "system").
    order:
        סדר מיון ברירת מחדל (מספר קטן → טאב מוקדם יותר).
    enabled:
        האם הטאב פעיל גלובלית (לפני פילטרים של profile/feature_flags).
    requires:
        רשימת capabilities / services נדרשים (למשל {"sql_store", "risk_engine"}).
    profile_overrides:
        מיפוי של פרופיל→enabled/disabled, כדי לאפשר התאמה עדינה יותר.
    """
    key: TabKey
    label: TabLabel
    renderer: TabRenderer
    group: str = "core"
    order: int = 100
    enabled: bool = True
    requires: Sequence[str] = field(default_factory=tuple)
    profile_overrides: Mapping[ProfileName, bool] = field(default_factory=dict)


TabRegistry = Dict[TabKey, TabMeta]


# -----------------------------------------------------------------------------
# הגדרת לוגרים ברמת דשבורד
# -----------------------------------------------------------------------------

def _setup_dashboard_logger() -> logging.Logger:
    """
    מגדיר לוגר מרכזי לדשבורד עם RotatingFileHandler + Console.

    - פורמט אחיד עם זמן, רמת לוג, מודול והודעה.
    - מניעת ריבוי handlers במקרה של ריצות חוזרות (Streamlit reruns).
    """
    logger = logging.getLogger(DASHBOARD_LOGGER_NAME)

    if logger.handlers:
        # כבר הוגדר בעבר (למשל בריצה קודמת של Streamlit)
        return logger

    logger.setLevel(logging.INFO)

    log_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # File handler עם Rotation
    try:
        file_handler = RotatingFileHandler(
            DASHBOARD_LOG_PATH,
            maxBytes=2 * 1024 * 1024,  # 2MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    except Exception as exc:  # pragma: no cover - קורה רק בתקלות IO
        console_handler.stream.write(
            f"[Dashboard] WARNING: Failed to attach file handler: {exc}\n"
        )

    logger.propagate = False
    logger.info("===== Dashboard startup =====")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Logs path: %s", DASHBOARD_LOG_PATH)
    logger.info("User: %s | Host: %s", RUNTIME_USER, RUNTIME_HOST)

    return logger


logger: logging.Logger = _setup_dashboard_logger()

# -----------------------------------------------------------------------------
# Streamlit page_config – הגדרה חד-פעמית ברמת דף
# -----------------------------------------------------------------------------

def _configure_streamlit_page() -> None:
    """
    מוודאת שהגדרת ה-page_config של Streamlit מתבצעת פעם אחת בלבד.

    משתמשים במפתח פנימי ב-session_state כדי להימנע משגיאות:
    - Streamlit מאפשר set_page_config פעם אחת בכל ריצה.
    """
    try:
        if st.session_state.get("_dashboard_page_configured", False):
            return
    except Exception:
        # session_state אולי עדיין לא מאותחל — נתעלם בשקט
        pass

    try:
        st.set_page_config(
            page_title=APP_NAME,
            page_icon=APP_ICON,
            layout="wide",
            initial_sidebar_state="expanded",
        )
        try:
            st.session_state["_dashboard_page_configured"] = True
        except Exception:
            # אם session_state לא זמין (בדיקות יחידה), נבלע את השגיאה
            logger.debug("session_state not available while tagging page_configured")
    except Exception as exc:  # pragma: no cover - מקרים קיצוניים בלבד
        logger.debug("set_page_config failed: %s", exc)


_configure_streamlit_page()

__all__ = [
    "APP_NAME",
    "APP_ICON",
    "APP_VERSION",
    "EnvName",
    "ProfileName",
    "DEFAULT_ENV",
    "DEFAULT_PROFILE",
    "PROJECT_ROOT",
    "LOGS_DIR",
    "DASHBOARD_LOGGER_NAME",
    "DASHBOARD_LOG_PATH",
    "RUNTIME_HOST",
    "RUNTIME_USER",
    "STARTED_AT_UTC",
    "FeatureFlags",
    "TabKey",
    "TabLabel",
    "NavPayload",
    "NavTarget",
    "ServiceStatus",
    "TabRenderer",
    "TabMeta",
    "TabRegistry",
    "logger",
]

# =====================
# Part 2/35 – AppContext access & env/profile detection helpers
# =====================

EnvProfile = Tuple[EnvName, ProfileName]

# מפתחות אפשריים לסביבת הריצה (env) ופרופיל (profile) במשתני סביבה
_ENV_ENVVAR_KEYS: Sequence[str] = (
    "PAIRS_ENV",
    "PAIRSTRADING_ENV",
    "APP_ENV",
    "ENV",
)

_PROFILE_ENVVAR_KEYS: Sequence[str] = (
    "PAIRS_PROFILE",
    "PAIRSTRADING_PROFILE",
    "APP_PROFILE",
    "PROFILE",
)

# סט ערכים מוכרים + מילון סינונימים
_KNOWN_ENVS: Sequence[EnvName] = (
    "dev",
    "live",
    "paper",
    "research",
    "backtest",
    "staging",
    "test",
)

_ENV_SYNONYMS: Dict[str, EnvName] = {
    "prod": "live",
    "production": "live",
    "rt": "live",
    "debug": "dev",
    "local": "dev",
}

_KNOWN_PROFILES: Sequence[ProfileName] = (
    "trading",
    "research",
    "risk",
    "macro",
    "monitoring",
    "default",
)

_PROFILE_SYNONYMS: Dict[str, ProfileName] = {
    "trade": "trading",
    "trader": "trading",
    "researcher": "research",
    "rm": "risk",
    "risk_mgmt": "risk",
    "ops": "monitoring",
    "mon": "monitoring",
}


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    """
    getattr בטוח שמנסה לקרוא גם ממילון (אם obj הוא Mapping).

    שימושי כדי לחלץ env/profile מתוך app_ctx.settings, גם אם זה אובייקט
    וגם אם זה dict/Pydantic.
    """
    if obj is None:
        return default

    # קודם מנסה כ-attrib רגיל
    try:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    except Exception:
        pass

    # אם זה Mapping – ננסה כ-key
    try:
        if isinstance(obj, Mapping) and name in obj:
            value = obj.get(name, default)
            if value is not None:
                return value
    except Exception:
        pass

    return default


def _session_raw_env_profile() -> Tuple[Optional[str], Optional[str]]:
    """
    מחזיר env/profile גולמיים מ-session_state, אם קיימים (לפני נרמול).

    לא נכשל אם session_state עדיין לא מאותחל.
    """
    try:
        raw_env = st.session_state.get("env")  # type: ignore[arg-type]
        raw_profile = st.session_state.get("profile")  # type: ignore[arg-type]
        return (
            str(raw_env) if raw_env is not None else None,
            str(raw_profile) if raw_profile is not None else None,
        )
    except Exception:
        return None, None


def _env_from_envvars() -> Optional[str]:
    """
    מחזיר env מתוך משתני סביבה, אם נמצא אחד רלוונטי.
    """
    for key in _ENV_ENVVAR_KEYS:
        value = os.getenv(key)
        if value:
            return value
    return None


def _profile_from_envvars() -> Optional[str]:
    """
    מחזיר profile מתוך משתני סביבה, אם נמצא אחד רלוונטי.
    """
    for key in _PROFILE_ENVVAR_KEYS:
        value = os.getenv(key)
        if value:
            return value
    return None


def _env_from_app_ctx(app_ctx: "AppContext") -> Optional[str]:
    """
    נסיון לחלץ env מתוך app_ctx:
    - app_ctx.env / app_ctx.environment
    - app_ctx.settings.env / .ENV / .environment
    - app_ctx.settings.config["env"] / ["ENV"] / ["environment"]
    """
    # app_ctx.* ישירות
    for attr in ("env", "environment"):
        val = _safe_getattr(app_ctx, attr)
        if val:
            return str(val)

    settings = _safe_getattr(app_ctx, "settings")
    if settings is None:
        return None

    # settings.* כ-attrib
    for attr in ("env", "ENV", "environment"):
        val = _safe_getattr(settings, attr)
        if val:
            return str(val)

    # settings.config כ-Mapping
    config_obj = _safe_getattr(settings, "config")
    if config_obj is not None:
        for key in ("env", "ENV", "environment"):
            try:
                if isinstance(config_obj, Mapping) and key in config_obj:
                    val = config_obj.get(key)
                    if val:
                        return str(val)
            except Exception:
                continue

    return None


def _profile_from_app_ctx(app_ctx: "AppContext") -> Optional[str]:
    """
    נסיון לחלץ profile מתוך app_ctx:
    - app_ctx.profile
    - app_ctx.settings.profile / .PROFILE
    - app_ctx.settings.config["profile"] / ["PROFILE"]
    """
    for attr in ("profile",):
        val = _safe_getattr(app_ctx, attr)
        if val:
            return str(val)

    settings = _safe_getattr(app_ctx, "settings")
    if settings is None:
        return None

    for attr in ("profile", "PROFILE"):
        val = _safe_getattr(settings, attr)
        if val:
            return str(val)

    config_obj = _safe_getattr(settings, "config")
    if config_obj is not None:
        for key in ("profile", "PROFILE"):
            try:
                if isinstance(config_obj, Mapping) and key in config_obj:
                    val = config_obj.get(key)
                    if val:
                        return str(val)
            except Exception:
                continue

    return None


def _normalize_env(raw: Optional[str]) -> EnvName:
    """
    מנרמל מחרוזת env לערך EnvName תקין.

    תומך בסינונימים ("prod" → "live", "debug" → "dev").
    אם לא מצליח – חוזר ל-DEFAULT_ENV.
    """
    if not raw:
        return DEFAULT_ENV

    value = raw.strip().lower()
    if value in _KNOWN_ENVS:
        return value  # type: ignore[return-value]

    if value in _ENV_SYNONYMS:
        return _ENV_SYNONYMS[value]

    logger.debug("Unknown env '%s', falling back to DEFAULT_ENV=%s", value, DEFAULT_ENV)
    return DEFAULT_ENV


def _normalize_profile(raw: Optional[str]) -> ProfileName:
    """
    מנרמל מחרוזת profile לערך ProfileName תקין.

    תומך בסינונימים ("trade" → "trading", "rm" → "risk").
    אם לא מצליח – חוזר ל-DEFAULT_PROFILE.
    """
    if not raw:
        return DEFAULT_PROFILE

    value = raw.strip().lower()
    if value in _KNOWN_PROFILES:
        return value  # type: ignore[return-value]

    if value in _PROFILE_SYNONYMS:
        return _PROFILE_SYNONYMS[value]

    logger.debug(
        "Unknown profile '%s', falling back to DEFAULT_PROFILE=%s",
        value,
        DEFAULT_PROFILE,
    )
    return DEFAULT_PROFILE


def detect_env_profile(app_ctx: "AppContext") -> EnvProfile:
    """
    מגלה env/profile לפי סדר עדיפות:

    1. session_state["env"], session_state["profile"] (אם כבר מאותחל).
    2. app_ctx / app_ctx.settings / config.
    3. משתני סביבה (APP_ENV, PAIRS_ENV וכו').
    4. ברירת מחדל (DEFAULT_ENV, DEFAULT_PROFILE).

    מחזיר זוג (EnvName, ProfileName) מנורמל.
    """
    # 1) session_state (אם כבר נכתב שם משהו)
    sess_env, sess_profile = _session_raw_env_profile()

    # 2) app_ctx
    ctx_env = _env_from_app_ctx(app_ctx)
    ctx_profile = _profile_from_app_ctx(app_ctx)

    # 3) env vars
    envvar_env = _env_from_envvars()
    envvar_profile = _profile_from_envvars()

    # קביעת env לפי סדר עדיפות
    raw_env = sess_env or ctx_env or envvar_env or DEFAULT_ENV
    # קביעת profile לפי סדר עדיפות
    raw_profile = sess_profile or ctx_profile or envvar_profile or DEFAULT_PROFILE

    env = _normalize_env(raw_env)
    profile = _normalize_profile(raw_profile)

    logger.info("Resolved dashboard env/profile: env=%s | profile=%s", env, profile)
    return env, profile


# -------------------------
# AppContext – גישה בטוחה
# -------------------------

_SESSION_APP_CTX_KEY = "_dashboard_app_ctx"


def _create_app_context() -> "AppContext":
    """
    מנסה לייצר / לאחזר AppContext מתוך core.app_context.

    סדר נסיון:
    1. AppContext.get_global()
    2. AppContext.create_default() / create() / from_env()
    3. AppContext() (קונסטרקטור ברירת מחדל)

    במקרה של כשל – זורק RuntimeError עם הודעה ברורה ללוג.
    """
    try:
        from core.app_context import AppContext  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - כשלי import
        logger.error("Failed to import core.app_context.AppContext: %s", exc)
        raise RuntimeError("Cannot import AppContext from core.app_context") from exc

    ctx: Optional["AppContext"] = None

    # 1) get_global()
    get_global = getattr(AppContext, "get_global", None)
    if callable(get_global):
        try:
            ctx = get_global()  # type: ignore[misc]
        except Exception as exc:
            logger.warning("AppContext.get_global() raised: %s", exc)

    if ctx is not None:
        logger.info("Using AppContext from AppContext.get_global()")
        return ctx

    # 2) create_default / create / from_env
    for factory_name in ("create_default", "create", "from_env"):
        factory = getattr(AppContext, factory_name, None)
        if callable(factory):
            try:
                ctx = factory()  # type: ignore[misc]
                if ctx is not None:
                    logger.info("Using AppContext from AppContext.%s()", factory_name)
                    return ctx
            except Exception as exc:
                logger.warning("AppContext.%s() raised: %s", factory_name, exc)

    # 3) קונסטרקטור ישיר
    try:
        ctx = AppContext()  # type: ignore[call-arg]
        logger.info("Using AppContext from direct constructor AppContext()")
        return ctx
    except Exception as exc:
        logger.error("Failed to construct AppContext(): %s", exc)
        raise RuntimeError("Could not initialize AppContext for dashboard") from exc


def get_app_context() -> "AppContext":
    """
    מחזיר מופע יחיד (per-session) של AppContext עבור הדשבורד.

    - אם כבר קיים ב-session_state – משתמש בו.
    - אחרת – יוצר חדש דרך _create_app_context ומאחסן אותו.
    """
    try:
        existing = st.session_state.get(_SESSION_APP_CTX_KEY)  # type: ignore[arg-type]
    except Exception:
        existing = None

    if existing is not None:
        return existing  # type: ignore[return-value]

    ctx = _create_app_context()
    try:
        st.session_state[_SESSION_APP_CTX_KEY] = ctx
    except Exception:
        # אם session_state לא זמין (בדיקות/CLI) – נרשום ללוג ונמשיך.
        logger.debug("session_state not available to store AppContext; using local instance only")
    return ctx

# =====================
# Part 3/35 – Service discovery & feature flag computation
# =====================

@dataclass
class ServiceCapabilities:
    """
    תמונת מצב מרוכזת של כל השירותים והיכולות שה-AppContext חושף.

    המטרה:
    - שכבה אחת שמסכמת "מה באמת קיים" במערכת (SqlStore, Broker, Macro, Risk...).
    - משמשת בסיס לחישוב feature_flags וטאבים פעילים.
    - מאפשרת לדשבורד Web ו-Desktop לשתף את אותה לוגיקה.

    לא כל שדה חייב להיות בשימוש מיידי; חשוב שתהיה תשתית להתרחבות עתידית.
    """

    # Data & Persistence
    sql_store: bool = False

    # Broker / Execution
    broker: bool = False
    broker_mode: Optional[str] = None  # למשל "paper" / "live" / "sim"

    # Market Data
    market_data_router: bool = False  # שכבת ניתוב למקורות כמו IBKR/Yahoo/Parquet

    # Core Engines
    risk_engine: bool = False
    signals_engine: bool = False
    macro_engine: bool = False
    agents_manager: bool = False
    fair_value_engine: bool = False

    # Backtesting & Optimization
    backtester: bool = False
    optimizer: bool = False
    meta_optimizer: bool = False

    # UX / Integration
    desktop_integration: bool = False  # חיבור ל-Desktop/Qt
    dashboard_service: bool = False    # DashboardService/Facade קיים

    def as_dict(self) -> Dict[str, Any]:
        """
        מייצר ייצוג dict של כל היכולות, לשימוש בתוך feature_flags / לוגים.
        """
        return {
            "sql_store": self.sql_store,
            "broker": self.broker,
            "broker_mode": self.broker_mode,
            "market_data_router": self.market_data_router,
            "risk_engine": self.risk_engine,
            "signals_engine": self.signals_engine,
            "macro_engine": self.macro_engine,
            "agents_manager": self.agents_manager,
            "fair_value_engine": self.fair_value_engine,
            "backtester": self.backtester,
            "optimizer": self.optimizer,
            "meta_optimizer": self.meta_optimizer,
            "desktop_integration": self.desktop_integration,
            "dashboard_service": self.dashboard_service,
        }


def _discover_services_mapping(app_ctx: "AppContext") -> Mapping[str, Any]:
    """
    מנסה לחלץ Mapping של services מתוך ה-AppContext, אם קיים.

    תומך במספר סגנונות אפשריים:
    - app_ctx.services
    - app_ctx.service_registry
    - app_ctx.service_container

    מחזיר always Mapping (יכול להיות dict ריק).
    """
    candidates = ("services", "service_registry", "service_container")

    for name in candidates:
        maybe = _safe_getattr(app_ctx, name)
        try:
            if isinstance(maybe, Mapping):
                # מנרמל את ה-keys ל-lowercase כדי להקל על חיפושים
                return {str(k).lower(): v for k, v in maybe.items()}
        except Exception:
            continue

    return {}


def _probe_service(
    app_ctx: "AppContext",
    services_map: Mapping[str, Any],
    candidates: Sequence[str],
) -> Tuple[bool, Optional[Any]]:
    """
    מנסה למצוא אובייקט שירות לפי רשימת שמות אפשריים (candidates):

    1. app_ctx.<candidate_name>
    2. services_map[<candidate_name>] לפי key נמוך (lowercase).

    מחזיר:
        (found: bool, obj_or_none: Any)
    """
    # ניסיון כ-attrib ישיר על app_ctx
    for name in candidates:
        obj = _safe_getattr(app_ctx, name)
        if obj is not None:
            return True, obj

    # ניסיון דרך services_map
    for name in candidates:
        key = name.lower()
        try:
            if key in services_map:
                obj = services_map[key]
                if obj is not None:
                    return True, obj
        except Exception:
            continue

    return False, None


def discover_capabilities(app_ctx: "AppContext") -> ServiceCapabilities:
    """
    מנוע גילוי יכולות (Capabilities) מתוך ה-AppContext.

    זה לא מניח מבנה קשיח של AppContext, אלא מנסה:
    - לגלות services mapping אם קיים.
    - לחפש שירותים לפי מגוון שמות אפשריים.
    - לזהות זמינות מודולי backtest/optimizer מתוך core.*.

    תוצאה:
        ServiceCapabilities — אובייקט מרוכז שמייצג אילו רכיבים קיימים בפועל.
    """
    services_map = _discover_services_mapping(app_ctx)

    # --- SqlStore / Persistence ---
    has_sql_store, sql_store_obj = _probe_service(
        app_ctx,
        services_map,
        candidates=("sql_store", "store", "db", "sql"),
    )

    # --- Broker / Execution ---
    has_broker, broker_obj = _probe_service(
        app_ctx,
        services_map,
        candidates=("broker", "ibkr", "ib", "broker_service", "execution"),
    )
    broker_mode: Optional[str] = None
    if has_broker and broker_obj is not None:
        for attr in ("mode", "account_type", "profile", "env", "environment"):
            val = _safe_getattr(broker_obj, attr)
            if isinstance(val, str) and val.strip():
                broker_mode = val.strip().lower()
                break

    # --- Market Data Router ---
    has_market_data, market_data_obj = _probe_service(
        app_ctx,
        services_map,
        candidates=("market_data_router", "market_data", "md_router", "data_router"),
    )
    _ = market_data_obj  # לא בשימוש כרגע, שמור להרחבות עתידיות

    # --- Core Engines ---
    has_risk_engine, _ = _probe_service(
        app_ctx,
        services_map,
        candidates=("risk_engine", "risk", "risk_service"),
    )
    has_signals_engine, _ = _probe_service(
        app_ctx,
        services_map,
        candidates=("signals_engine", "signal_engine", "signals", "signal_generator"),
    )
    has_macro_engine, _ = _probe_service(
        app_ctx,
        services_map,
        candidates=("macro_engine", "macro_model", "macro", "macro_service"),
    )
    has_agents_manager, _ = _probe_service(
        app_ctx,
        services_map,
        candidates=("agents_manager", "agent_manager", "agents", "ai_agents"),
    )
    has_fair_value_engine, _ = _probe_service(
        app_ctx,
        services_map,
        candidates=("fair_value_engine", "fair_value", "fv_engine"),
    )

    # --- Dashboard / Desktop Integration ---
    has_dashboard_service, _ = _probe_service(
        app_ctx,
        services_map,
        candidates=("dashboard_service", "dashboard_facade", "dashboard"),
    )
    has_desktop_bridge, _ = _probe_service(
        app_ctx,
        services_map,
        candidates=("desktop_bridge", "desktop", "qt_app", "desktop_context"),
    )

    # --- Backtesting & Optimization (בזיהוי מודולים, לא שירותים) ---
    backtester_available = False
    optimizer_available = False
    meta_optimizer_available = False

    try:
        import importlib.util as _importlib_util  # type: ignore[import]

        if _importlib_util.find_spec("core.optimization_backtester") is not None:
            backtester_available = True
    except Exception:
        backtester_available = False

    try:
        import importlib.util as _importlib_util  # type: ignore[import]

        if _importlib_util.find_spec("core.optimizer") is not None:
            optimizer_available = True
    except Exception:
        optimizer_available = False

    try:
        import importlib.util as _importlib_util  # type: ignore[import]

        if _importlib_util.find_spec("core.meta_optimizer") is not None:
            meta_optimizer_available = True
    except Exception:
        meta_optimizer_available = False

    caps = ServiceCapabilities(
        sql_store=has_sql_store,
        broker=has_broker,
        broker_mode=broker_mode,
        market_data_router=has_market_data,
        risk_engine=has_risk_engine,
        signals_engine=has_signals_engine,
        macro_engine=has_macro_engine,
        agents_manager=has_agents_manager,
        fair_value_engine=has_fair_value_engine,
        backtester=backtester_available,
        optimizer=optimizer_available,
        meta_optimizer=meta_optimizer_available,
        desktop_integration=has_desktop_bridge,
        dashboard_service=has_dashboard_service,
    )

    logger.info(
        "Service capabilities resolved: "
        "sql_store=%s, broker=%s(mode=%s), market_data=%s, "
        "risk=%s, signals=%s, macro=%s, agents=%s, fair_value=%s, "
        "backtester=%s, optimizer=%s, meta_opt=%s, desktop=%s, dashboard_service=%s",
        caps.sql_store,
        caps.broker,
        caps.broker_mode,
        caps.market_data_router,
        caps.risk_engine,
        caps.signals_engine,
        caps.macro_engine,
        caps.agents_manager,
        caps.fair_value_engine,
        caps.backtester,
        caps.optimizer,
        caps.meta_optimizer,
        caps.desktop_integration,
        caps.dashboard_service,
    )

    return caps


def compute_feature_flags(app_ctx: "AppContext") -> FeatureFlags:
    """
    חישוב Feature Flags חכמים מתוך AppContext:

    אחריות:
    --------
    - זיהוי env/profile (דרך detect_env_profile).
    - גילוי capabilities (discover_capabilities).
    - קבלת החלטה אילו טאבים / דומיינים / מצבים מופעלים.
    - קביעת פלגים כמו: enable_live_trading_actions, show_debug_info, use_sql_backed_state.

    הפלט:
    -----
    FeatureFlags = Dict[str, Any], עם מפתחות מרכזיים:
        - env, profile, env_profile
        - app_name, version, host, user
        - capabilities: dict של ServiceCapabilities
        - domains: dict דומיינים (risk/macro/signals/portfolio/...)
        - tabs: dict של טאבים ופלג enabled
        - enable_live_trading_actions
        - show_debug_info
        - use_sql_backed_state
        - enable_experiment_mode
        - desktop_integration
    """
    env, profile = detect_env_profile(app_ctx)
    caps = discover_capabilities(app_ctx)

    enable_live_trading_actions: bool = (
        env == "live"
        and caps.broker
        and caps.market_data_router
        and profile == "trading"
    )

    show_debug_info: bool = env in ("dev", "research", "test")
    use_sql_backed_state: bool = caps.sql_store

    enable_experiment_mode: bool = (
        env in ("dev", "research", "backtest")
        and (caps.backtester or caps.optimizer or caps.meta_optimizer)
    )

    # דומיינים לוגיים — מאפשרים ל-Tab Registry לדעת מה "עולם התוכן" הזמין
    domains: Dict[str, bool] = {
        "risk": caps.risk_engine,
        "macro": caps.macro_engine,
        "signals": caps.signals_engine,
        "portfolio": caps.sql_store or caps.broker,
        "fair_value": caps.fair_value_engine,
        "agents": caps.agents_manager,
        "backtest": caps.backtester,
        "optimization": caps.optimizer or caps.meta_optimizer,
    }

    # המלצה ראשונית אילו טאבים אמורים להיות פעילים.
    # ה-Tab Registry בהמשך ישלב את זה עם קיום מודולים/קבצים בפועל.
    tabs_flags: Dict[str, bool] = {
        "home": True,  # תמיד קיים
        "smart_scan": caps.signals_engine,
        "pair": True,  # Pair Tab יכול לעבוד גם רק עם דאטה בסיסית
        "matrix": caps.market_data_router or caps.sql_store,
        "comparison_matrices": caps.market_data_router or caps.sql_store,
        "backtest": caps.backtester,
        "insights": caps.backtester or caps.optimizer,
        "macro": caps.macro_engine,
        "portfolio": caps.sql_store or caps.broker,
        "risk": caps.risk_engine or caps.sql_store,
        "fair_value": caps.fair_value_engine,
        "config": True,
        "agents": caps.agents_manager,
        "logs": True,  # לוגים/בריאות מערכת — נרצה זמינות תמידית
    }

    flags: FeatureFlags = {
        "env": env,
        "profile": profile,
        "env_profile": (env, profile),
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "host": RUNTIME_HOST,
        "user": RUNTIME_USER,
        "capabilities": caps.as_dict(),
        "domains": domains,
        "tabs": tabs_flags,
        "enable_live_trading_actions": enable_live_trading_actions,
        "show_debug_info": show_debug_info,
        "use_sql_backed_state": use_sql_backed_state,
        "enable_experiment_mode": enable_experiment_mode,
        "desktop_integration": caps.desktop_integration,
    }

    logger.info(
        "Feature flags resolved: env=%s, profile=%s, live_actions=%s, "
        "sql_state=%s, experiment_mode=%s, tabs={risk=%s, macro=%s, agents=%s}",
        env,
        profile,
        enable_live_trading_actions,
        use_sql_backed_state,
        enable_experiment_mode,
        tabs_flags.get("risk"),
        tabs_flags.get("macro"),
        tabs_flags.get("agents"),
    )

    return flags


# עדכון __all__ לכל החלקים החדשים (כולל חלק 2 שלא הוספנו קודם)
try:
    __all__ += [
        "EnvProfile",
        "detect_env_profile",
        "get_app_context",
        "ServiceCapabilities",
        "discover_capabilities",
        "compute_feature_flags",
    ]
except NameError:  # pragma: no cover – למקרה שהקובץ ירוץ מחוץ להקשר הרגיל
    __all__ = [
        "EnvProfile",
        "detect_env_profile",
        "get_app_context",
        "ServiceCapabilities",
        "discover_capabilities",
        "compute_feature_flags",
    ]

# =====================
# Part 4/35 – Session bootstrap & base dashboard context
# =====================

import uuid
from datetime import date

# מפתחות session_state סטנדרטיים לדשבורד
SESSION_KEY_ENV: str = "env"
SESSION_KEY_PROFILE: str = "profile"
SESSION_KEY_FEATURE_FLAGS: str = "feature_flags"
SESSION_KEY_NAV_TARGET: str = "nav_target"
SESSION_KEY_LAST_TAB_KEY: str = "last_active_tab_key"
SESSION_KEY_BASE_CTX: str = "base_dashboard_context"
SESSION_KEY_LAST_SNAPSHOT: str = "dashboard_last_snapshot"
SESSION_KEY_RUN_ID: str = "dashboard_run_id"
SESSION_KEY_MACRO_CTX: str = "macro_context"
SESSION_KEY_RISK_CTX: str = "risk_context"
SESSION_KEY_EXPERIMENT_CTX: str = "experiment_context"


def _ensure_session_default(key: str, default: Any) -> Any:
    """
    מבטיח שקיים ערך עבור key ב-session_state.

    אם key לא קיים – מגדיר את default ומחזיר אותו.
    אם קיים – מחזיר את הערך הקיים בלי לשנות.
    """
    try:
        state = st.session_state
    except Exception as exc:  # pragma: no cover
        logger.error("Streamlit session_state not available: %s", exc)
        return default

    if key not in state:
        state[key] = default
    return state[key]


def _ensure_session_default_factory(key: str, factory: Callable[[], Any]) -> Any:
    """
    כמו _ensure_session_default, אבל מקבל factory שמייצר default רק אם צריך.

    זה מאפשר להימנע מחישובים כבדים כשכבר יש ערך ב-session_state.
    """
    try:
        state = st.session_state
    except Exception as exc:  # pragma: no cover
        logger.error("Streamlit session_state not available: %s", exc)
        return factory()

    if key not in state:
        state[key] = factory()
    return state[key]


def _get_or_create_run_id() -> str:
    """
    מחזיר מזהה ריצה (run_id) ייחודי לדשבורד הנוכחי.

    - פעם ראשונה: יוצר run_id חדש (UUID4 hex) ושומר ב-session_state.
    - ריצות חוזרות (rerun): משתמש באותו run_id.
    """
    def _factory() -> str:
        rid = uuid.uuid4().hex
        logger.info("Created new dashboard run_id=%s", rid)
        return rid

    run_id = _ensure_session_default_factory(SESSION_KEY_RUN_ID, _factory)
    return str(run_id)


def _extract_base_currency(app_ctx: "AppContext", default: str = "USD") -> str:
    """
    מנסה להוציא base_currency מתוך AppContext/settings/config.
    """
    settings = _safe_getattr(app_ctx, "settings")
    if settings is None:
        return default

    for name in ("base_currency", "BASE_CURRENCY", "currency"):
        val = _safe_getattr(settings, name)
        if isinstance(val, str) and val.strip():
            return val.strip().upper()

    cfg = _safe_getattr(settings, "config")
    if isinstance(cfg, Mapping):
        for key in ("base_currency", "BASE_CURRENCY", "currency"):
            try:
                val = cfg.get(key)
            except Exception:
                continue
            if isinstance(val, str) and val.strip():
                return val.strip().upper()

    return default


def _extract_timezone(app_ctx: "AppContext", default: str = "UTC") -> str:
    """
    מנסה להוציא timezone מתוך AppContext/settings/config.
    מחזיר מחרוזת IANA timezone (למשל 'UTC', 'Asia/Jerusalem').
    """
    settings = _safe_getattr(app_ctx, "settings")
    if settings is None:
        return default

    for name in ("timezone", "TIMEZONE", "tz", "tz_name"):
        val = _safe_getattr(settings, name)
        if isinstance(val, str) and val.strip():
            return val.strip()

    cfg = _safe_getattr(settings, "config")
    if isinstance(cfg, Mapping):
        for key in ("timezone", "TIMEZONE", "tz", "tz_name"):
            try:
                val = cfg.get(key)
            except Exception:
                continue
            if isinstance(val, str) and val.strip():
                return val.strip()

    return default


def _patch_dashboard_context_env_profile(
    ctx_obj: Any,
    env: EnvName,
    profile: ProfileName,
) -> Any:
    """
    מוודא שה-Base Dashboard Context מכיל env/profile עדכניים:

    - מנסה כ-attrib (ctx.env / ctx.profile).
    - אם זה Mapping – מנסה ctx["env"], ctx["profile"].
    - אם אין ל-context את השדות האלה – לא נכשל; פשוט מחזיר את האובייקט.
    """
    # attribs
    try:
        if hasattr(ctx_obj, "env"):
            setattr(ctx_obj, "env", env)
        if hasattr(ctx_obj, "profile"):
            setattr(ctx_obj, "profile", profile)
    except Exception:
        # לא עוצר את הזרימה בגלל כשל קוסמטי
        pass

    # mapping
    try:
        if isinstance(ctx_obj, MutableMapping):
            if "env" not in ctx_obj:
                ctx_obj["env"] = env
            else:
                ctx_obj["env"] = env
            if "profile" not in ctx_obj:
                ctx_obj["profile"] = profile
            else:
                ctx_obj["profile"] = profile
    except Exception:
        pass

    return ctx_obj


def _build_base_dashboard_context(
    app_ctx: "AppContext",
    env: EnvName,
    profile: ProfileName,
) -> Any:
    """
    בונה Base Dashboard Context עבור הדשבורד:

    סדר עדיפות:
    1. שימוש בפונקציה build_default_dashboard_context מתוך dashboard_service_factory,
       אם קיימת.
    2. אם יש DashboardService שמציע default context – להשתמש בו.
    3. נפילה חכמה ל-dict פשוט עם שדות בסיסיים.

    המטרה: תמיד להחזיר אובייקט context נוח לעבודה בטאבים, בלי להיכשל
    אם מודולים מתקדמים חסרים.
    """
    today = date.today()
    base_currency = _extract_base_currency(app_ctx)
    tz_name = _extract_timezone(app_ctx)

    # ננסה קודם dashboard_service_factory.build_default_dashboard_context
    ctx_obj: Any = None

    try:
        import dashboard_service_factory as _dsf  # type: ignore[import]
    except Exception:
        _dsf = None  # type: ignore[assignment]

    if _dsf is not None:
        try:
            build_ctx = getattr(_dsf, "build_default_dashboard_context", None)
            if callable(build_ctx):
                ctx_obj = build_ctx()
                logger.info(
                    "Base dashboard context obtained from "
                    "dashboard_service_factory.build_default_dashboard_context()"
                )
        except Exception as exc:
            logger.warning(
                "build_default_dashboard_context() raised %s; "
                "falling back to generic context",
                exc,
            )

    # אופציה שניה: אם יש DashboardService עם API שמספק default context
    if ctx_obj is None:
        try:
            from core.dashboard_service import DashboardService  # type: ignore[import]
        except Exception:
            DashboardService = None  # type: ignore[assignment]

        if DashboardService is not None:
            try:
                # אם ל-AppContext יש helper ל-DashboardService – נשתמש בו
                svc = _safe_getattr(app_ctx, "dashboard_service")
                if svc is not None and isinstance(svc, DashboardService):
                    get_default_ctx = getattr(svc, "build_default_context", None)
                    if callable(get_default_ctx):
                        ctx_obj = get_default_ctx()
                        logger.info(
                            "Base dashboard context obtained from DashboardService.build_default_context()"
                        )
            except Exception as exc:
                logger.warning(
                    "DashboardService default context discovery failed: %s", exc
                )

    # fallback סופי: dict פשוט
    if ctx_obj is None:
        ctx_obj = {
            "start_date": today,
            "end_date": today,
            "env": env,
            "profile": profile,
            "ui_mode": "full",
            "benchmark": "SPY",
            "portfolio_id": "default",
            "base_currency": base_currency,
            "timezone": tz_name,
        }
        logger.info(
            "Base dashboard context constructed as plain dict: "
            "start_date=%s, env=%s, profile=%s, base_currency=%s, tz=%s",
            today,
            env,
            profile,
            base_currency,
            tz_name,
        )
    else:
        ctx_obj = _patch_dashboard_context_env_profile(ctx_obj, env, profile)

    return ctx_obj


def _get_current_env_profile_from_session() -> EnvProfile:
    """
    מחזיר EnvProfile מתוך session_state בלבד (ללא גילוי AppContext/envvars).

    אם אין ערכים – נופל לברירות המחדל המנורמלות (DEFAULT_ENV/DEFAULT_PROFILE).
    """
    try:
        state = st.session_state
    except Exception as exc:  # pragma: no cover
        logger.error("Streamlit session_state not available: %s", exc)
        return DEFAULT_ENV, DEFAULT_PROFILE

    raw_env = state.get(SESSION_KEY_ENV, DEFAULT_ENV)
    raw_profile = state.get(SESSION_KEY_PROFILE, DEFAULT_PROFILE)

    env = _normalize_env(str(raw_env))
    profile = _normalize_profile(str(raw_profile))

    return env, profile


def get_feature_flags_from_session() -> FeatureFlags:
    """
    מחזיר FeatureFlags מתוך session_state.

    אם feature_flags לא קיים – זורק RuntimeError; האחריות על הקורא
    לוודא קריאה ל-bootstrap_session קודם.
    """
    try:
        state = st.session_state
    except Exception as exc:  # pragma: no cover
        logger.error("Streamlit session_state not available: %s", exc)
        raise RuntimeError("Streamlit session_state not available") from exc

    flags = state.get(SESSION_KEY_FEATURE_FLAGS)
    if not isinstance(flags, dict):
        raise RuntimeError(
            "feature_flags not found in session_state; "
            "did you call bootstrap_session(app_ctx, feature_flags)?"
        )
    return flags  # type: ignore[return-value]


def get_session_run_id() -> str:
    """
    מחזיר את run_id השמור ב-session_state (ויוצר חדש אם אין).
    """
    rid = _get_or_create_run_id()
    return str(rid)


def bootstrap_session(app_ctx: "AppContext", feature_flags: FeatureFlags) -> None:
    """
    מאתחל את מצב ה-Session של הדשבורד בצורה מקצועית:

    מבטיח שיהיו:
    - env, profile (מסונכרנים עם feature_flags).
    - feature_flags (התוצאה המלאה של compute_feature_flags).
    - nav_target (ברירת מחדל None).
    - last_active_tab_key (ברירת מחדל "home").
    - base_dashboard_context (אובייקט context מלא).
    - dashboard_last_snapshot (placeholder).
    - dashboard_run_id (מזהה ריצה ייחודי).
    - macro_context / risk_context / experiment_context (placeholders להרחבות עתידיות).
    """
    env: EnvName = feature_flags.get("env", DEFAULT_ENV)  # type: ignore[assignment]
    profile: ProfileName = feature_flags.get("profile", DEFAULT_PROFILE)  # type: ignore[assignment]

    # env/profile תמיד מסונכרנים עם feature_flags
    try:
        st.session_state[SESSION_KEY_ENV] = env
        st.session_state[SESSION_KEY_PROFILE] = profile
        st.session_state[SESSION_KEY_FEATURE_FLAGS] = feature_flags
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to write env/profile/feature_flags to session_state: %s", exc)

    # run_id – per-session
    run_id = _get_or_create_run_id()

    # nav_target – מיועד לניווט בין טאבים (flow)
    _ensure_session_default(SESSION_KEY_NAV_TARGET, None)

    # last_active_tab_key – ברירת מחדל home
    _ensure_session_default(SESSION_KEY_LAST_TAB_KEY, "home")

    # base_dashboard_context – נבנה פעם אחת ונשמר
    def _base_ctx_factory() -> Any:
        return _build_base_dashboard_context(app_ctx, env, profile)

    _ensure_session_default_factory(SESSION_KEY_BASE_CTX, _base_ctx_factory)

    # snapshot אחרון – מוכן לשימוש ע"י DashboardService/SqlStore
    _ensure_session_default(SESSION_KEY_LAST_SNAPSHOT, None)

    # הקשרים נוספים – לרמת קרן גידור (Risk/Macro/Experiments)
    _ensure_session_default(SESSION_KEY_MACRO_CTX, {})
    _ensure_session_default(SESSION_KEY_RISK_CTX, {})
    _ensure_session_default(SESSION_KEY_EXPERIMENT_CTX, {})

    logger.info(
        "Session bootstrap complete: env=%s, profile=%s, run_id=%s, "
        "base_ctx_type=%s",
        env,
        profile,
        run_id,
        type(st.session_state.get(SESSION_KEY_BASE_CTX)).__name__,
    )


# עדכון __all__ עבור חלק 4
try:
    __all__ += [
        "SESSION_KEY_ENV",
        "SESSION_KEY_PROFILE",
        "SESSION_KEY_FEATURE_FLAGS",
        "SESSION_KEY_NAV_TARGET",
        "SESSION_KEY_LAST_TAB_KEY",
        "SESSION_KEY_BASE_CTX",
        "SESSION_KEY_LAST_SNAPSHOT",
        "SESSION_KEY_RUN_ID",
        "SESSION_KEY_MACRO_CTX",
        "SESSION_KEY_RISK_CTX",
        "SESSION_KEY_EXPERIMENT_CTX",
        "_ensure_session_default",
        "_ensure_session_default_factory",
        "_get_or_create_run_id",
        "_extract_base_currency",
        "_extract_timezone",
        "_patch_dashboard_context_env_profile",
        "_build_base_dashboard_context",
        "_get_current_env_profile_from_session",
        "get_feature_flags_from_session",
        "get_session_run_id",
        "bootstrap_session",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_ENV",
        "SESSION_KEY_PROFILE",
        "SESSION_KEY_FEATURE_FLAGS",
        "SESSION_KEY_NAV_TARGET",
        "SESSION_KEY_LAST_TAB_KEY",
        "SESSION_KEY_BASE_CTX",
        "SESSION_KEY_LAST_SNAPSHOT",
        "SESSION_KEY_RUN_ID",
        "SESSION_KEY_MACRO_CTX",
        "SESSION_KEY_RISK_CTX",
        "SESSION_KEY_EXPERIMENT_CTX",
        "_ensure_session_default",
        "_ensure_session_default_factory",
        "_get_or_create_run_id",
        "_extract_base_currency",
        "_extract_timezone",
        "_patch_dashboard_context_env_profile",
        "_build_base_dashboard_context",
        "_get_current_env_profile_from_session",
        "get_feature_flags_from_session",
        "get_session_run_id",
        "bootstrap_session",
    ]

# =====================
# Part 5/35 – Global navigation helpers (nav_target, last tab, history)
# =====================

from collections.abc import Collection

# Fallback עדין ל-json_safe – לא חובה, אבל עוזר לנקות payloadים ל-session_state
try:  # pragma: no cover - תלוי במבנה הפרויקט
    from common.json_safe import make_json_safe as _make_json_safe  # type: ignore[import]
except Exception:  # pragma: no cover
    def _make_json_safe(obj: Any) -> Any:
        return obj


# היסטוריית ניווט (אופציונלית, אבל שימושית לניתוח/Debug)
SESSION_KEY_NAV_HISTORY: str = "dashboard_nav_history"


def _ensure_nav_history_list() -> List[Dict[str, Any]]:
    """
    מבטיח שב-session_state יישב list שיכיל את היסטוריית הניווט בין טאבים.

    כל רשומה: {"ts": ISO-UTC, "from": last_tab_key, "to": tab_key, "payload_keys": [...]}
    """
    history = _ensure_session_default(SESSION_KEY_NAV_HISTORY, [])
    if not isinstance(history, list):
        history = []
        st.session_state[SESSION_KEY_NAV_HISTORY] = history
    return history  # type: ignore[return-value]


def set_last_active_tab_key(tab_key: TabKey) -> None:
    """
    מעדכן את הטאב האחרון שהיה פעיל.

    Router יעבוד עם הפונקציה הזו אחרי שהוא יודע איזה טאב באמת מוצג.
    """
    if not tab_key:
        return

    try:
        st.session_state[SESSION_KEY_LAST_TAB_KEY] = str(tab_key)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to set last_active_tab_key in session_state: %s", exc)


def get_last_active_tab_key(default: TabKey = "home") -> TabKey:
    """
    מחזיר את מפתח הטאב האחרון שהוצג (אם אין – מחזיר default).
    """
    try:
        value = st.session_state.get(SESSION_KEY_LAST_TAB_KEY, default)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to read last_active_tab_key from session_state: %s", exc)
        return default

    if not value:
        return default
    return str(value)


def set_nav_target(
    tab_key: TabKey,
    payload: Optional[NavPayload] = None,
) -> None:
    """
    שומר nav_target גלובלי ב-session_state כדי לאפשר flow בין טאבים.

    שימוש טיפוסי מתוך טאב אחר (למשל home / smart_scan / matrix):

        set_nav_target("backtest", {"pair": "AAPL/MSFT", "preset": "smoke"})

    Router ישתמש ב-nav_target בקריאה הבאה וינסה:
    - למפות את tab_key לטאב פעיל.
    - להעביר את payload ל-renderer של אותו טאב.
    - לנקות את nav_target אחרי צריכה.

    הערה:
    -----
    * payload מנוקה בעדינות דרך _make_json_safe כדי להקטין סיכוי
      לאובייקטים לא-serializable ב-session_state.
    * אין כאן st.experimental_rerun – הזרימה מתבססת על rerun טבעי
      של Streamlit אחרי אינטראקציה (כפתור/בחירה).
    """
    if not tab_key:
        logger.warning("set_nav_target called with empty tab_key; ignoring")
        return

    cleaned_payload: Optional[NavPayload]
    if payload is None:
        cleaned_payload = None
    elif isinstance(payload, Mapping):
        # נשתמש ב-dict רגיל + json_safe
        cleaned_payload = dict(_make_json_safe(dict(payload)))  # type: ignore[arg-type]
    else:
        logger.warning(
            "set_nav_target payload is not a Mapping (type=%s); "
            "forcing to dict under 'value' key",
            type(payload).__name__,
        )
        cleaned_payload = {"value": _make_json_safe(payload)}  # type: ignore[assignment]

    nav: NavTarget = {"tab_key": str(tab_key)}
    if cleaned_payload is not None:
        nav["payload"] = cleaned_payload

    try:
        st.session_state[SESSION_KEY_NAV_TARGET] = nav
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to set nav_target in session_state: %s", exc)
        return

    # עדכון היסטוריית ניווט (לא חובה, אבל שימושי לניתוח התנהגות/Debug)
    try:
        history = _ensure_nav_history_list()
        last_tab = get_last_active_tab_key(default="home")
        payload_keys: Optional[List[str]] = None
        if isinstance(cleaned_payload, Mapping):
            payload_keys = [str(k) for k in cleaned_payload.keys()]

        history.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "from": last_tab,
                "to": str(tab_key),
                "payload_keys": payload_keys,
            }
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to append nav history: %s", exc)

    logger.info(
        "nav_target set: tab_key=%s, payload_keys=%s",
        tab_key,
        list(cleaned_payload.keys()) if isinstance(cleaned_payload, Mapping) else None,
    )


def get_nav_target() -> Optional[NavTarget]:
    """
    מחזיר את nav_target הגלובלי, אם קיים, בלי למחוק אותו.

    מבטיח:
    - תמיד להחזיר dict עם 'tab_key' (str) ו-'payload' (dict) אם קיימים.
    - במקרה חריג (טיפוס שגוי ב-session_state) – מחזיר None.
    """
    try:
        raw = st.session_state.get(SESSION_KEY_NAV_TARGET)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to read nav_target from session_state: %s", exc)
        return None

    if raw is None:
        return None

    if not isinstance(raw, Mapping):
        logger.warning(
            "nav_target in session_state is not a Mapping (type=%s); clearing",
            type(raw).__name__,
        )
        try:
            st.session_state[SESSION_KEY_NAV_TARGET] = None
        except Exception:
            pass
        return None

    tab_key_val = raw.get("tab_key")
    if not tab_key_val:
        return None

    tab_key_str = str(tab_key_val)
    payload = raw.get("payload")
    nav: NavTarget = {"tab_key": tab_key_str}

    if isinstance(payload, Mapping):
        nav["payload"] = dict(payload)  # shallow copy
    elif payload is not None:
        # אם מישהו שם payload לא-מילוני – נדחוס אותו תחת "value"
        nav["payload"] = {"value": payload}  # type: ignore[assignment]

    return nav


def clear_nav_target() -> None:
    """
    מנקה את nav_target מה-session_state.
    """
    try:
        st.session_state[SESSION_KEY_NAV_TARGET] = None
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to clear nav_target from session_state: %s", exc)


def _consume_nav_target(
    active_tab_keys: Collection[TabKey],
) -> Tuple[Optional[TabKey], Optional[NavPayload]]:
    """
    נצרכת nav_target בצורה בטוחה ביחס לטאבים הפעילים כרגע.

    פרוטוקול:
    ---------
    1. קורא את nav_target הנוכחי (אם אין – מחזיר (None, None)).
    2. בודק אם tab_key יעד נמצא ב-active_tab_keys:
        - אם לא נמצא → כותב אזהרה ללוג, מנקה nav_target, מחזיר (None, None).
    3. אם כן נמצא:
        - מחלץ payload (dict או {}).
        - מנקה nav_target ב-session_state.
        - מחזיר (tab_key, payload).

    Router ישתמש בפונקציה הזו כדי להחליט:
    - איזה טאב הופך להיות "יעד" (nav_tab_key).
    - איזה payload יש להעביר ל-renderer של הטאב.
    """
    nav = get_nav_target()
    if nav is None:
        return None, None

    tab_key = str(nav.get("tab_key", "")).strip()
    if not tab_key:
        clear_nav_target()
        return None, None

    # בדיקה מול הטאבים הפעילים
    if tab_key not in active_tab_keys:
        logger.warning(
            "nav_target requested tab_key='%s' which is not in active_tab_keys=%s; "
            "clearing nav_target",
            tab_key,
            list(active_tab_keys),
        )
        clear_nav_target()
        return None, None

    raw_payload = nav.get("payload")  # Optional[Any]
    payload: NavPayload
    if isinstance(raw_payload, Mapping):
        payload = dict(raw_payload)
    elif raw_payload is None:
        payload = {}
    else:
        payload = {"value": raw_payload}  # type: ignore[assignment]

    clear_nav_target()

    logger.info(
        "nav_target consumed for tab_key=%s, payload_keys=%s",
        tab_key,
        list(payload.keys()),
    )

    return tab_key, payload


# עדכון __all__ עבור חלק 5
try:
    __all__ += [
        "SESSION_KEY_NAV_HISTORY",
        "set_last_active_tab_key",
        "get_last_active_tab_key",
        "set_nav_target",
        "get_nav_target",
        "clear_nav_target",
        "_consume_nav_target",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_NAV_HISTORY",
        "set_last_active_tab_key",
        "get_last_active_tab_key",
        "set_nav_target",
        "get_nav_target",
        "clear_nav_target",
        "_consume_nav_target",
    ]

# =====================
# Part 6/35 – Tab renderer discovery, lazy import & adaptive call
# =====================

import importlib
import importlib.util
import inspect

from collections.abc import MutableMapping  # משלים שימושים מחלקים קודמים

# Cache פנימי לפונקציות טאבים שכבר נמצאו
_TabFn = Callable[..., Any]
_TAB_FN_CACHE: Dict[str, _TabFn] = {}


def _find_module(module_candidates: Sequence[str]) -> Optional[Any]:
    """
    מנסה לאתר ולייבא מודול מתוך רשימת שמות אפשריים.

    לדוגמה:
        ("dashboard_home_v2", "root.dashboard_home_v2")

    אסטרטגיה:
    ---------
    1. שימוש ב-importlib.util.find_spec כדי לבדוק אם המודול קיים.
    2. אם קיים – importlib.import_module ומחזיר את המודול.
    3. אם כל המועמדים נכשלים – מחזיר None.
    """
    for mod_name in module_candidates:
        try:
            spec = importlib.util.find_spec(mod_name)
        except Exception as exc:  # pragma: no cover
            logger.debug("find_spec(%s) failed: %s", mod_name, exc)
            continue

        if spec is None:
            continue

        try:
            module = importlib.import_module(mod_name)
            logger.info("Imported tab module '%s'", mod_name)
            return module
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to import module '%s': %s", mod_name, exc)
            continue

    return None


def _find_tab_function_in_module(
    module: Any,
    func_candidates: Sequence[str],
) -> Optional[_TabFn]:
    """
    מחפש פונקציית רנדרר במודול אחד לפי רשימת שמות אפשריים.

    שימושי לעבודה עם קוד קיים:
    - render_dashboard_home_v2
    - render_dashboard_home
    - render_home_tab
    - render_tab
    """
    for func_name in func_candidates:
        try:
            fn = getattr(module, func_name, None)
        except Exception:
            fn = None

        if callable(fn):
            logger.info(
                "Resolved tab renderer function '%s.%s'",
                getattr(module, "__name__", "<unknown>"),
                func_name,
            )
            return fn  # type: ignore[return-value]

    return None


def _resolve_tab_function(
    logical_name: str,
    module_candidates: Sequence[str],
    func_candidates: Sequence[str],
) -> Optional[_TabFn]:
    """
    מאתר פונקציית Renderer לטאב לפי:
    - logical_name (לצורך cache / לוגים).
    - רשימת שמות מודולים אפשריים.
    - רשימת שמות פונקציות אפשריים.

    משתמש ב-_TAB_FN_CACHE כדי להימנע מחיפושים חוזרים.
    """
    if logical_name in _TAB_FN_CACHE:
        return _TAB_FN_CACHE[logical_name]

    module = _find_module(module_candidates)
    if module is None:
        logger.warning(
            "No module found for tab '%s' (candidates=%s)",
            logical_name,
            list(module_candidates),
        )
        return None

    fn = _find_tab_function_in_module(module, func_candidates)
    if fn is None:
        logger.warning(
            "No renderer function found for tab '%s' in module '%s' (candidates=%s)",
            logical_name,
            getattr(module, "__name__", "<unknown>"),
            list(func_candidates),
        )
        return None

    _TAB_FN_CACHE[logical_name] = fn
    return fn


def _build_call_kwargs_for_tab_fn(
    fn: _TabFn,
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload],
) -> Dict[str, Any]:
    """
    בונה מילון kwargs חכם לפונקציית טאב קיימת, לפי החתימה שלה.

    המטרה:
    -------
    לאפשר שימוש בפונקציות קיימות עם חתימות שונות, למשל:
        def render_tab() -> None: ...
        def render_tab(app_ctx: AppContext) -> None: ...
        def render_tab(app_ctx: AppContext, feature_flags: Dict[str, Any]) -> None: ...
        def render_tab(app_ctx, feature_flags, nav_payload: Optional[Dict[str, Any]]) -> None: ...

    אסטרטגיה:
    ---------
    - קורא את inspect.signature(fn).
    - לכל פרמטר, מחליט מה "הכי הגיוני" להעביר:
        * "app_ctx", "ctx", "app_context", "context" → app_ctx
        * "feature_flags", "flags", "ff" → feature_flags
        * "nav_payload", "payload", "navigation" → nav_payload (או {} אם None)
        * "env", "profile" → נשלף מתוך feature_flags
        * פרמטרים עם default בלבד – ניתן לדלג עליהם.
    """
    try:
        sig = inspect.signature(fn)
    except Exception as exc:  # pragma: no cover
        logger.debug("inspect.signature failed for %s: %s", fn, exc)
        return {}

    kwargs: Dict[str, Any] = {}
    env = feature_flags.get("env", DEFAULT_ENV)
    profile = feature_flags.get("profile", DEFAULT_PROFILE)
    safe_payload: NavPayload = nav_payload or {}

    for name, param in sig.parameters.items():
        # נתעלם מפרמטרים מסוג VAR_POSITIONAL / VAR_KEYWORD – הם יטופלו ע"י **kwargs
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        lname = name.lower()

        if lname in ("app_ctx", "ctx", "app_context", "context"):
            kwargs[name] = app_ctx
        elif lname in ("feature_flags", "flags", "ff"):
            kwargs[name] = feature_flags
        elif lname in ("nav_payload", "payload", "navigation"):
            kwargs[name] = safe_payload
        elif lname == "env":
            kwargs[name] = env
        elif lname == "profile":
            kwargs[name] = profile
        elif param.default is not inspect._empty:
            # יש default – אפשר לא להעביר ערך ונשענים על ברירת המחדל
            continue
        else:
            # פרמטר ללא default שלא זיהינו – עדיף לא להעביר מאשר לבנות ערך אקראי,
            # כדי לכבד חתימות מסוימות. אם הפונקציה באמת דורשת אותו,
            # ייזרק TypeError ויתפס ע"י safe_render_tab.
            continue

    return kwargs


def _invoke_tab_function(
    logical_name: str,
    fn: _TabFn,
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload],
) -> None:
    """
    מפעיל פונקציית טאב תוך הסתגלות לחתימה שלה.

    Flow:
    -----
    1. בונה kwargs חכם לפי החתימה.
    2. מנסה fn(**kwargs).
    3. אם מתקבל TypeError (בגלל התאמת חתימה שגויה):
       - מנסה פולי-בק במספר וריאציות:
            * fn(app_ctx, feature_flags, nav_payload)
            * fn(app_ctx, feature_flags)
            * fn(app_ctx)
            * fn()
       - אם כולן נכשלות → מעלה את ה-Exception כדי שיתפס ב-safe_render_tab.
    """
    kwargs = _build_call_kwargs_for_tab_fn(fn, app_ctx, feature_flags, nav_payload)

    try:
        fn(**kwargs)
        return
    except TypeError as exc:
        logger.debug(
            "Primary call of tab '%s' with kwargs failed (%s); trying fallbacks",
            logical_name,
            exc,
        )
        # פוליבקים מתונים ומסודרים
        fallbacks = [
            (fn, (app_ctx, feature_flags, nav_payload), {}),
            (fn, (app_ctx, feature_flags), {}),
            (fn, (app_ctx,), {}),
            (fn, tuple(), {}),
        ]
        for f, args, kw in fallbacks:
            try:
                f(*args, **kw)
                return
            except TypeError:
                continue
        # אם הגענו עד כאן – נרים את החריגה המקורית
        raise
    # חריגות אחרות (ValueError, RuntimeError וכו') – יטופלו ב-safe_render_tab


def _lazy_render_tab(
    logical_name: str,
    module_candidates: Sequence[str],
    func_candidates: Sequence[str],
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload],
) -> None:
    """
    Helper כללי שמשמש את כל הטאבים:

    - מאתר/מייבא מודול וטאב-פונקציה פעם אחת בלבד (עם cache).
    - אם לא נמצא כלום – מציג הודעה ידידותית בטאב, אבל לא מפיל את הדשבורד.
    - אם נמצא – מפעיל את fn בצורה אדפטיבית (עם תמיכה בחתימות שונות).

    דוגמה לשימוש מתוך render_home_tab:

        def render_home_tab(app_ctx, feature_flags, nav_payload=None):
            _lazy_render_tab(
                logical_name="home",
                module_candidates=("dashboard_home_v2", "root.dashboard_home_v2"),
                func_candidates=(
                    "render_dashboard_home_v2",
                    "render_dashboard_home",
                    "render_home_tab",
                    "render_home",
                ),
                app_ctx=app_ctx,
                feature_flags=feature_flags,
                nav_payload=nav_payload,
            )
    """
    fn = _resolve_tab_function(logical_name, module_candidates, func_candidates)
    if fn is None:
        st.warning(
            f"⚠️ טאב '{logical_name}' אינו זמין כרגע "
            f"(מודול/פונקציית רנדרר לא נמצאו)."
        )
        return

    _invoke_tab_function(logical_name, fn, app_ctx, feature_flags, nav_payload)


# עדכון __all__ עבור חלק 6
try:
    __all__ += [
        "_TAB_FN_CACHE",
        "_find_module",
        "_find_tab_function_in_module",
        "_resolve_tab_function",
        "_build_call_kwargs_for_tab_fn",
        "_invoke_tab_function",
        "_lazy_render_tab",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_TAB_FN_CACHE",
        "_find_module",
        "_find_tab_function_in_module",
        "_resolve_tab_function",
        "_build_call_kwargs_for_tab_fn",
        "_invoke_tab_function",
        "_lazy_render_tab",
    ]

# =====================
# Part 7/35 – Tabs registry: base definitions, profile order & registry builder
# =====================

# -----------------------------
# Tab keys (centralised naming)
# -----------------------------

TAB_KEY_HOME: TabKey = "home"
TAB_KEY_SMART_SCAN: TabKey = "smart_scan"
TAB_KEY_PAIR: TabKey = "pair"
TAB_KEY_MATRIX: TabKey = "matrix"
TAB_KEY_COMPARISON_MATRICES: TabKey = "comparison_matrices"
TAB_KEY_BACKTEST: TabKey = "backtest"
TAB_KEY_INSIGHTS: TabKey = "insights"
TAB_KEY_MACRO: TabKey = "macro"
TAB_KEY_PORTFOLIO: TabKey = "portfolio"
TAB_KEY_RISK: TabKey = "risk"
TAB_KEY_FAIR_VALUE: TabKey = "fair_value"
TAB_KEY_CONFIG: TabKey = "config"
TAB_KEY_AGENTS: TabKey = "agents"
TAB_KEY_LOGS: TabKey = "logs"

ALL_TAB_KEYS: Tuple[TabKey, ...] = (
    TAB_KEY_HOME,
    TAB_KEY_SMART_SCAN,
    TAB_KEY_PAIR,
    TAB_KEY_MATRIX,
    TAB_KEY_COMPARISON_MATRICES,
    TAB_KEY_BACKTEST,
    TAB_KEY_INSIGHTS,
    TAB_KEY_MACRO,
    TAB_KEY_PORTFOLIO,
    TAB_KEY_RISK,
    TAB_KEY_FAIR_VALUE,
    TAB_KEY_CONFIG,
    TAB_KEY_AGENTS,
    TAB_KEY_LOGS,
)

# -----------------------------------
# Profile-based ordering of tab keys
# -----------------------------------

_PROFILE_TAB_ORDER: Mapping[ProfileName, Tuple[TabKey, ...]] = {
    # ברירת מחדל / trading-classic – כל הטאבים העיקריים
    "default": (
        TAB_KEY_HOME,
        TAB_KEY_SMART_SCAN,
        TAB_KEY_PAIR,
        TAB_KEY_MATRIX,
        TAB_KEY_COMPARISON_MATRICES,
        TAB_KEY_BACKTEST,
        TAB_KEY_INSIGHTS,
        TAB_KEY_MACRO,
        TAB_KEY_PORTFOLIO,
        TAB_KEY_RISK,
        TAB_KEY_FAIR_VALUE,
        TAB_KEY_AGENTS,
        TAB_KEY_CONFIG,
        TAB_KEY_LOGS,
    ),
    "trading": (
        TAB_KEY_HOME,
        TAB_KEY_SMART_SCAN,
        TAB_KEY_PAIR,
        TAB_KEY_BACKTEST,
        TAB_KEY_RISK,
        TAB_KEY_MACRO,
        TAB_KEY_PORTFOLIO,
        TAB_KEY_FAIR_VALUE,
        TAB_KEY_INSIGHTS,
        TAB_KEY_AGENTS,
        TAB_KEY_CONFIG,
        TAB_KEY_LOGS,
    ),
    "research": (
        TAB_KEY_HOME,
        TAB_KEY_MATRIX,
        TAB_KEY_COMPARISON_MATRICES,
        TAB_KEY_PAIR,
        TAB_KEY_BACKTEST,
        TAB_KEY_INSIGHTS,
        TAB_KEY_MACRO,
        TAB_KEY_FAIR_VALUE,
        TAB_KEY_CONFIG,
        TAB_KEY_LOGS,
    ),
    "risk": (
        TAB_KEY_HOME,
        TAB_KEY_RISK,
        TAB_KEY_MACRO,
        TAB_KEY_PORTFOLIO,
        TAB_KEY_LOGS,
        TAB_KEY_CONFIG,
    ),
    "macro": (
        TAB_KEY_HOME,
        TAB_KEY_MACRO,
        TAB_KEY_PORTFOLIO,
        TAB_KEY_RISK,
        TAB_KEY_LOGS,
    ),
    "monitoring": (
        TAB_KEY_HOME,
        TAB_KEY_LOGS,
        TAB_KEY_RISK,
        TAB_KEY_MACRO,
        TAB_KEY_AGENTS,
        TAB_KEY_CONFIG,
    ),
}


def get_profile_tab_order(profile: ProfileName) -> Tuple[TabKey, ...]:
    """
    מחזיר את סדר הטאבים המועדף עבור profile נתון.

    אם הפרופיל לא מוגדר ב-_PROFILE_TAB_ORDER:
    - חוזר לברירת המחדל ("default").
    """
    try:
        return _PROFILE_TAB_ORDER.get(profile, _PROFILE_TAB_ORDER["default"])
    except Exception:  # pragma: no cover
        return _PROFILE_TAB_ORDER["default"]


# ---------------------------------------------------
# Base Tab definitions – label, group, requires, etc
# ---------------------------------------------------

# כל key כאן הוא TabKey, וכל value הוא dict עם תיאור הטאב:
# - label:     הטקסט/אימוג'י שמוצג ב-UI
# - group:     קבוצה לוגית (core / research / risk / macro / system)
# - order:     סדר סדר-גלובלי (ישולב עם סדר לפי profile)
# - requires:  capabilities שנדרשים מהמערכת (keys מתוך ServiceCapabilities.as_dict)
# - enabled_default: האם הטאב פעיל ברמת "תשתית" לפני feature_flags
# - profile_overrides: אילו פרופילים משנים enabled (True/False) ביחס לברירת המחדל
_BASE_TAB_DEFS: Dict[TabKey, Dict[str, Any]] = {
    TAB_KEY_HOME: {
        "label": "🏠 Dashboard",
        "group": "core",
        "order": 10,
        "requires": (),
        "enabled_default": True,
        "profile_overrides": {},  # תמיד נרצה את Home
    },
    TAB_KEY_SMART_SCAN: {
        "label": "🔍 Smart Scan",
        "group": "core",
        "order": 20,
        "requires": ("signals_engine",),
        "enabled_default": True,
        "profile_overrides": {
            "risk": False,
            "macro": False,
            "monitoring": False,
        },
    },
    TAB_KEY_PAIR: {
        "label": "🧪 Pair Analysis",
        "group": "core",
        "order": 30,
        "requires": (),  # עובד גם עם דאטה בסיסית
        "enabled_default": True,
        "profile_overrides": {},
    },
    TAB_KEY_MATRIX: {
        "label": "🧮 Matrix / Correlation",
        "group": "research",
        "order": 40,
        "requires": ("market_data_router",),
        "enabled_default": True,
        "profile_overrides": {
            "trading": False,
            "risk": False,
            "macro": False,
            "monitoring": False,
        },
    },
    TAB_KEY_COMPARISON_MATRICES: {
        "label": "🔬 Comparison Matrices",
        "group": "research",
        "order": 50,
        "requires": ("market_data_router",),
        "enabled_default": True,
        "profile_overrides": {
            "trading": True,
            "research": True,
            "risk": False,
            "macro": False,
            "monitoring": False,
        },
    },
    TAB_KEY_BACKTEST: {
        "label": "📈 Backtest",
        "group": "core",
        "order": 60,
        "requires": ("backtester",),
        "enabled_default": True,
        "profile_overrides": {},
    },
    TAB_KEY_INSIGHTS: {
        "label": "🧠 Insights",
        "group": "research",
        "order": 70,
        "requires": ("backtester",),
        "enabled_default": True,
        "profile_overrides": {
            "risk": False,
            "macro": False,
            "monitoring": False,
        },
    },
    TAB_KEY_MACRO: {
        "label": "🌐 Macro",
        "group": "macro",
        "order": 80,
        "requires": ("macro_engine",),
        "enabled_default": True,
        "profile_overrides": {},
    },
    TAB_KEY_PORTFOLIO: {
        "label": "💼 Portfolio / Fund View",
        "group": "core",
        "order": 90,
        "requires": ("sql_store",),
        "enabled_default": True,
        "profile_overrides": {},
    },
    TAB_KEY_RISK: {
        "label": "⚠️ Risk",
        "group": "risk",
        "order": 100,
        "requires": ("risk_engine",),
        "enabled_default": True,
        "profile_overrides": {},
    },
    TAB_KEY_FAIR_VALUE: {
        "label": "💲 Fair Value / Relative Value",
        "group": "research",
        "order": 110,
        "requires": ("fair_value_engine",),
        "enabled_default": True,
        "profile_overrides": {
            "monitoring": False,
        },
    },
    TAB_KEY_CONFIG: {
        "label": "⚙️ Config",
        "group": "system",
        "order": 120,
        "requires": (),
        "enabled_default": True,
        "profile_overrides": {},
    },
    TAB_KEY_AGENTS: {
        "label": "🤖 Agents",
        "group": "system",
        "order": 130,
        "requires": ("agents_manager",),
        "enabled_default": True,
        "profile_overrides": {},
    },
    TAB_KEY_LOGS: {
        "label": "📜 Logs / System Health",
        "group": "system",
        "order": 140,
        "requires": (),
        "enabled_default": True,
        "profile_overrides": {},
    },
}


def _capabilities_satisfy_requires(
    capabilities: Mapping[str, Any],
    requires: Sequence[str],
) -> bool:
    """
    בודק אם capabilities מספקים את כל הדרישות בטאב אחד.

    * כל דרישה נבחנת כ-capabilities[req] == truthy (למעט broker_mode שהוא str).
    * אם אין דרישות – תמיד True.
    """
    if not requires:
        return True

    for req in requires:
        # מחזיר False אם המפתח לא קיים או הערך לא truthy
        if not bool(capabilities.get(req, False)):
            return False
    return True


def _get_tab_renderer(tab_key: TabKey) -> TabRenderer:
    """
    מחזיר Renderer עבור tab_key נתון.

    מיפוי זה מחבר בין TabMeta לבין פונקציות הרנדרינג בפועל.
    הפונקציות עצמן יוגדרו בחלקים הבאים (render_*_tab) ויקראו ל-_lazy_render_tab.
    """
    # פונקציית fallback – במקרה שמפתח לא מוכר (לא אמור לקרות בשימוש רגיל)
    def _render_missing_tab(
        app_ctx: "AppContext",
        feature_flags: FeatureFlags,
        nav_payload: Optional[NavPayload] = None,
    ) -> None:
        st.warning(
            f"⚠️ Tab '{tab_key}' מוגדר ברג'יסטרי אבל אין לו Renderer תואם."
        )

    # מיפוי לוגי → wrapper functions (שיוגדרו בהמשך הקובץ)
    mapping: Dict[TabKey, TabRenderer] = {
        TAB_KEY_HOME: render_home_tab,  # type: ignore[name-defined]
        TAB_KEY_SMART_SCAN: render_smart_scan_tab,  # type: ignore[name-defined]
        TAB_KEY_PAIR: render_pair_tab,  # type: ignore[name-defined]
        TAB_KEY_MATRIX: render_matrix_tab,  # type: ignore[name-defined]
        TAB_KEY_COMPARISON_MATRICES: render_comparison_matrices_tab,  # type: ignore[name-defined]
        TAB_KEY_BACKTEST: render_backtest_tab,  # type: ignore[name-defined]
        TAB_KEY_INSIGHTS: render_insights_tab,  # type: ignore[name-defined]
        TAB_KEY_MACRO: render_macro_tab,  # type: ignore[name-defined]
        TAB_KEY_PORTFOLIO: render_portfolio_tab,  # type: ignore[name-defined]
        TAB_KEY_RISK: render_risk_tab,  # type: ignore[name-defined]
        TAB_KEY_FAIR_VALUE: render_fair_value_tab,  # type: ignore[name-defined]
        TAB_KEY_CONFIG: render_config_tab_wrapper,  # type: ignore[name-defined]
        TAB_KEY_AGENTS: render_agents_tab,  # type: ignore[name-defined]
        TAB_KEY_LOGS: render_logs_tab,  # type: ignore[name-defined]
    }

    renderer = mapping.get(tab_key)
    if renderer is None:
        logger.warning("Requested renderer for unknown tab_key='%s'", tab_key)
        return _render_missing_tab
    return renderer


def build_tab_registry(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
) -> TabRegistry:
    """
    בונה TabRegistry מלא מתוך:

    - _BASE_TAB_DEFS (metadata ליבה).
    - feature_flags["tabs"] (המלצות enable/disable).
    - feature_flags["capabilities"] (ServiceCapabilities / דרישות טכניות).

    תוצאה:
    -------
    * כל TabMeta כולל:
        - key, label, group, order.
        - renderer (TabRenderer מלא; עוטף את הטאב בפועל).
        - enabled (enabled גלובלי, לפני פילטר לפי profile).
        - requires (capabilities נדרשים).
        - profile_overrides (מיפוי ProfileName→bool).
    """
    caps: Mapping[str, Any] = feature_flags.get("capabilities", {}) or {}
    tabs_flags: Mapping[str, Any] = feature_flags.get("tabs", {}) or {}

    registry: TabRegistry = {}

    for tab_key, base_def in _BASE_TAB_DEFS.items():
        label = str(base_def.get("label", tab_key))
        group = str(base_def.get("group", "core"))
        order_val = int(base_def.get("order", 9999))
        requires: Tuple[str, ...] = tuple(base_def.get("requires", ()))  # type: ignore[assignment]
        enabled_default: bool = bool(base_def.get("enabled_default", True))
        profile_overrides: Mapping[ProfileName, bool] = base_def.get("profile_overrides", {})  # type: ignore[assignment]

        # המלצת feature_flags["tabs"] – אם לא מוגדר, נניח enabled_default
        ff_tab_enabled = bool(tabs_flags.get(tab_key, enabled_default))

        # דרישות capabilities
        req_ok = _capabilities_satisfy_requires(caps, requires)

        # enabled גלובלי: בסיס + feature_flags + capabilities
        enabled_global = enabled_default and ff_tab_enabled and req_ok

        renderer = _get_tab_renderer(tab_key)

        meta = TabMeta(
            key=tab_key,
            label=label,
            renderer=renderer,
            group=group,
            order=order_val,
            enabled=enabled_global,
            requires=requires,
            profile_overrides=profile_overrides,
        )
        registry[tab_key] = meta

    logger.info(
        "Tab registry built with %d entries (enabled=%d, disabled=%d)",
        len(registry),
        sum(1 for m in registry.values() if m.enabled),
        sum(1 for m in registry.values() if not m.enabled),
    )

    if feature_flags.get("show_debug_info"):
        # הצגה קצרה בלוג – לא ב-UI
        debug_info = {
            key: {
                "label": meta.label,
                "enabled": meta.enabled,
                "group": meta.group,
                "order": meta.order,
                "requires": list(meta.requires),
            }
            for key, meta in registry.items()
        }
        logger.debug("Tab registry debug snapshot: %s", debug_info)

    return registry


# עדכון __all__ עבור חלק 7
try:
    __all__ += [
        "TAB_KEY_HOME",
        "TAB_KEY_SMART_SCAN",
        "TAB_KEY_PAIR",
        "TAB_KEY_MATRIX",
        "TAB_KEY_COMPARISON_MATRICES",
        "TAB_KEY_BACKTEST",
        "TAB_KEY_INSIGHTS",
        "TAB_KEY_MACRO",
        "TAB_KEY_PORTFOLIO",
        "TAB_KEY_RISK",
        "TAB_KEY_FAIR_VALUE",
        "TAB_KEY_CONFIG",
        "TAB_KEY_AGENTS",
        "TAB_KEY_LOGS",
        "ALL_TAB_KEYS",
        "get_profile_tab_order",
        "_PROFILE_TAB_ORDER",
        "_BASE_TAB_DEFS",
        "_capabilities_satisfy_requires",
        "_get_tab_renderer",
        "build_tab_registry",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "TAB_KEY_HOME",
        "TAB_KEY_SMART_SCAN",
        "TAB_KEY_PAIR",
        "TAB_KEY_MATRIX",
        "TAB_KEY_COMPARISON_MATRICES",
        "TAB_KEY_BACKTEST",
        "TAB_KEY_INSIGHTS",
        "TAB_KEY_MACRO",
        "TAB_KEY_PORTFOLIO",
        "TAB_KEY_RISK",
        "TAB_KEY_FAIR_VALUE",
        "TAB_KEY_CONFIG",
        "TAB_KEY_AGENTS",
        "TAB_KEY_LOGS",
        "ALL_TAB_KEYS",
        "get_profile_tab_order",
        "_PROFILE_TAB_ORDER",
        "_BASE_TAB_DEFS",
        "_capabilities_satisfy_requires",
        "_get_tab_renderer",
        "build_tab_registry",
    ]

# =====================
# Part 8/35 – Tab render wrappers (uniform signature, lazy wiring)
# =====================

def _log_tab_entry(
    tab_key: TabKey,
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload],
) -> None:
    """
    לוג מרוכז לכניסה לטאב מסוים.

    עוזר לנתח זרימת ניווט (flow) ולראות אילו payloadים מגיעים לכל טאב.
    """
    try:
        env = feature_flags.get("env", DEFAULT_ENV)
        profile = feature_flags.get("profile", DEFAULT_PROFILE)
        payload_keys: Optional[Sequence[str]] = None
        if isinstance(nav_payload, Mapping):
            payload_keys = list(nav_payload.keys())
        logger.info(
            "Rendering tab '%s' (env=%s, profile=%s, payload_keys=%s)",
            tab_key,
            env,
            profile,
            payload_keys,
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to log tab entry for '%s': %s", tab_key, exc)


# ----------------------
# Home / Dashboard tab
# ----------------------

def render_home_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    🏠 טאב הבית – Dashboard Home v2

    עטיפה סביב dashboard_home_v2 (או וריאציות שלו), עם חתימה אחידה.
    """
    _log_tab_entry(TAB_KEY_HOME, feature_flags, nav_payload)

    _lazy_render_tab(
        logical_name="home",
        module_candidates=(
            "dashboard_home_v2",
            "root.dashboard_home_v2",
            "root.dashboard_home",
            "dashboard_home",
        ),
        func_candidates=(
            "render_dashboard_home_v2",
            "render_dashboard_home",
            "render_home_tab",
            "render_home",
        ),
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        nav_payload=nav_payload,
    )


# ----------------------
# Smart Scan tab
# ----------------------

def render_smart_scan_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    🔍 Smart Scan – סריקה חכמה של universe לפי פרופילים.

    עטיפה סביב smart_scan_tab.* עם חתימה אחידה.
    """
    _log_tab_entry(TAB_KEY_SMART_SCAN, feature_flags, nav_payload)

    _lazy_render_tab(
        logical_name="smart_scan",
        module_candidates=(
            "smart_scan_tab",
            "root.smart_scan_tab",
        ),
        func_candidates=(
            "render_smart_scan_tab",
            "render_tab",
            "render_smart_scan",
        ),
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        nav_payload=nav_payload,
    )


# ----------------------
# Pair Analysis tab
# ----------------------

def render_pair_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    🧪 Pair Analysis – ניתוח זוג ברמת קרן.

    nav_payload טיפוסי:
        {"pair": "AAPL/MSFT", "view": "spread"} וכדומה.
    """
    _log_tab_entry(TAB_KEY_PAIR, feature_flags, nav_payload)

    _lazy_render_tab(
        logical_name="pair",
        module_candidates=(
            "pair_tab",
            "root.pair_tab",
        ),
        func_candidates=(
            "render_pair_tab",
            "render_tab",
            "render_pair_analysis_tab",
        ),
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        nav_payload=nav_payload,
    )


# ----------------------
# Matrix / Correlation tab
# ----------------------

def render_matrix_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    🧮 Matrix / Correlation – מחקר מטריצות קורלציה/קו-אינטגרציה וכו'.

    עטיפה סביב matrix_research_tab או מודולים דומים.
    """
    _log_tab_entry(TAB_KEY_MATRIX, feature_flags, nav_payload)

    _lazy_render_tab(
        logical_name="matrix",
        module_candidates=(
            "matrix_research_tab",
            "root.matrix_research_tab",
            "matrix_tab",
            "root.matrix_tab",
        ),
        func_candidates=(
            "render_matrix_research_tab",
            "render_matrix_tab",
            "render_tab",
        ),
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        nav_payload=nav_payload,
    )


# ----------------------
# Comparison Matrices tab
# ----------------------

def render_comparison_matrices_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    🔬 Comparison Matrices – טאב השוואות מטריצות/זוגות/סביבות.

    nav_payload יכול להכיל pair / universe / preset וכו'.
    """
    _log_tab_entry(TAB_KEY_COMPARISON_MATRICES, feature_flags, nav_payload)

    _lazy_render_tab(
        logical_name="comparison_matrices",
        module_candidates=(
            "tab_comparison_matrices",
            "root.tab_comparison_matrices",
        ),
        func_candidates=(
            "render_comparison_matrices_tab",
            "render_tab",
        ),
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        nav_payload=nav_payload,
    )


# ----------------------
# Backtest tab
# ----------------------

def render_backtest_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    📈 Backtest – סימולציות היסטוריות ברמת קרן.

    nav_payload טיפוסי:
        {"pair": "AAPL/MSFT", "preset": "smoke", "mode": "wf"}.

    עטיפה סביב backtest_tab / backtest עם וריאציות v2/v3.
    """
    _log_tab_entry(TAB_KEY_BACKTEST, feature_flags, nav_payload)

    _lazy_render_tab(
        logical_name="backtest",
        module_candidates=(
            "backtest_tab",
            "root.backtest_tab",
            "backtest",
            "root.backtest",
        ),
        func_candidates=(
            "render_backtest_tab_v3",
            "render_backtest_tab_v2",
            "render_backtest_tab",
            "render_backtest",
            "render_tab",
        ),
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        nav_payload=nav_payload,
    )


# ----------------------
# Insights tab
# ----------------------

def render_insights_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    🧠 Insights – ML/SHAP/Drivers/Explainability.

    עטיפה סביב insights / insights_tab.
    """
    _log_tab_entry(TAB_KEY_INSIGHTS, feature_flags, nav_payload)

    _lazy_render_tab(
        logical_name="insights",
        module_candidates=(
            "insights",
            "root.insights",
            "insights_tab",
            "root.insights_tab",
        ),
        func_candidates=(
            "render_insights_tab",
            "render_tab",
        ),
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        nav_payload=nav_payload,
    )


# ----------------------
# Macro tab
# ----------------------

def render_macro_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    🌐 Macro – מקרו כלכלי, Regimes, אינדיקטורים ו-Overlays.

    עטיפה סביב macro_tab / macro.
    """
    _log_tab_entry(TAB_KEY_MACRO, feature_flags, nav_payload)

    _lazy_render_tab(
        logical_name="macro",
        module_candidates=(
            "macro_tab",
            "root.macro_tab",
            "macro",
            "root.macro",
        ),
        func_candidates=(
            "render_macro_tab",
            "render_tab",
        ),
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        nav_payload=nav_payload,
    )


# ----------------------
# Portfolio / Fund View tab
# ----------------------

def render_portfolio_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    💼 Portfolio / Fund View – ניתוח פורטפוליו/קרן, חשיפות ותרומות סיכון.

    עטיפה סביב portfolio_tab / fund_view_tab.
    """
    _log_tab_entry(TAB_KEY_PORTFOLIO, feature_flags, nav_payload)

    _lazy_render_tab(
        logical_name="portfolio",
        module_candidates=(
            "portfolio_tab",
            "root.portfolio_tab",
            "fund_view_tab",
            "root.fund_view_tab",
        ),
        func_candidates=(
            "render_portfolio_tab",
            "render_fund_view_tab",
            "render_tab",
        ),
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        nav_payload=nav_payload,
    )


# ----------------------
# Risk tab
# ----------------------

def render_risk_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    ⚠️ Risk – מגבלות, חשיפות, Kill-switch ותצוגת סיכון.

    nav_payload טיפוסי:
        {"portfolio_id": "default", "view": "limits"}.
    """
    _log_tab_entry(TAB_KEY_RISK, feature_flags, nav_payload)

    _lazy_render_tab(
        logical_name="risk",
        module_candidates=(
            "risk_tab",
            "root.risk_tab",
        ),
        func_candidates=(
            "render_risk_tab",
            "render_tab",
        ),
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        nav_payload=nav_payload,
    )


# ----------------------
# Fair Value tab
# ----------------------

def render_fair_value_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    💲 Fair Value / Relative Value – ניתוח Fair Value לזוגות/סקטורים.

    עטיפה סביב fair_value_api_tab או מודול דומה.
    """
    _log_tab_entry(TAB_KEY_FAIR_VALUE, feature_flags, nav_payload)

    _lazy_render_tab(
        logical_name="fair_value",
        module_candidates=(
            "fair_value_api_tab",
            "root.fair_value_api_tab",
            "fair_value_tab",
            "root.fair_value_tab",
        ),
        func_candidates=(
            "render_fair_value_tab",
            "render_fair_value_api_tab",
            "render_tab",
        ),
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        nav_payload=nav_payload,
    )


# ----------------------
# Config tab (special wrapper name)
# ----------------------

def render_config_tab_wrapper(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    ⚙️ Config – ניהול קונפיגים.

    עטיפה סביב config_tab.render_config_tab, עם fallback ידידותי אם המודול חסר.
    """
    _log_tab_entry(TAB_KEY_CONFIG, feature_flags, nav_payload)

    _lazy_render_tab(
        logical_name="config",
        module_candidates=(
            "config_tab",
            "root.config_tab",
        ),
        func_candidates=(
            "render_config_tab",
            "render_tab",
        ),
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        nav_payload=nav_payload,
    )



# ----------------------
# Logs / System Health tab
# ----------------------


# עדכון __all__ עבור חלק 8
try:
    __all__ += [
        "_log_tab_entry",
        "render_home_tab",
        "render_smart_scan_tab",
        "render_pair_tab",
        "render_matrix_tab",
        "render_comparison_matrices_tab",
        "render_backtest_tab",
        "render_insights_tab",
        "render_macro_tab",
        "render_portfolio_tab",
        "render_risk_tab",
        "render_fair_value_tab",
        "render_config_tab_wrapper",
        "render_agents_tab",
        "render_logs_tab",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_log_tab_entry",
        "render_home_tab",
        "render_smart_scan_tab",
        "render_pair_tab",
        "render_matrix_tab",
        "render_comparison_matrices_tab",
        "render_backtest_tab",
        "render_insights_tab",
        "render_macro_tab",
        "render_portfolio_tab",
        "render_risk_tab",
        "render_fair_value_tab",
        "render_config_tab_wrapper",
        "render_agents_tab",
        "render_logs_tab",
    ]

# =====================
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
        col_left, col_mid, col_right = st.columns([2, 1, 1.4])

        with col_left:
            st.markdown(
                f"### {APP_ICON} {APP_NAME}  \n"
                f"Version: `{APP_VERSION}`  •  Env: `{env}`  •  Profile: `{profile}`"
            )
            st.caption(
                f"Host: `{app_status.get('host', RUNTIME_HOST)}`  •  "
                f"User: `{app_status.get('user', RUNTIME_USER)}`  •  "
                f"Run ID: `{run_id}`"
            )

            dates_line = ""
            if start_date and end_date:
                dates_line = f"{start_date.isoformat()} → {end_date.isoformat()}"
            elif start_date:
                dates_line = f"{start_date.isoformat()} → ?"
            elif end_date:
                dates_line = f"? → {end_date.isoformat()}"

            st.markdown(
                f"- **Base currency:** `{base_currency}`  •  "
                f"**Timezone:** `{tz_name}`  \n"
                f"- **Benchmark:** `{benchmark}`"
                + (f"  •  **Dates:** {dates_line}" if dates_line else "")
            )

        with col_mid:
            _render_macro_banner(macro_status, feature_flags)

        with col_right:
            _render_trading_mode_badge(broker_status, feature_flags)
            _render_risk_banner(risk_status, feature_flags)


# עדכון __all__ עבור חלק 10
try:
    __all__ += [
        "_extract_dates_from_base_ctx",
        "_extract_benchmark_from_base_ctx",
        "_render_macro_banner",
        "_render_risk_banner",
        "_render_trading_mode_badge",
        "_render_global_header",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_extract_dates_from_base_ctx",
        "_extract_benchmark_from_base_ctx",
        "_render_macro_banner",
        "_render_risk_banner",
        "_render_trading_mode_badge",
        "_render_global_header",
    ]
# =====================
# Part 11/35 – Quick actions & dashboard snapshot utilities
# =====================

def _serialize_base_ctx_for_snapshot(base_ctx: Any) -> Any:
    """
    ממיר את base_dashboard_context למבנה snapshot ידידותי ל-SQL/JSON.

    הסדר:
    -----
    1. אם יש method בשם model_dump() (Pydantic v2) – להשתמש בו.
    2. אם יש method בשם dict() (Pydantic v1 / dataclass) – להשתמש בו.
    3. אם זה Mapping – להעתיק ל-dict רגיל.
    4. אחרת – להחזיר repr(...) כמחרוזת.
    """
    if base_ctx is None:
        return None

    # Pydantic v2 / dataclasses וכדו'
    try:
        model_dump = getattr(base_ctx, "model_dump", None)
        if callable(model_dump):
            return model_dump()
    except Exception:
        pass

    # Pydantic v1 / dataclasses
    try:
        as_dict = getattr(base_ctx, "dict", None)
        if callable(as_dict):
            return as_dict()
    except Exception:
        pass

    # Mapping
    try:
        if isinstance(base_ctx, Mapping):
            return dict(base_ctx)
    except Exception:
        pass

    # fallback – repr בלבד
    try:
        return {"__repr__": repr(base_ctx)}
    except Exception:
        return {"__repr__": "<unserializable base_ctx>"}


def _build_dashboard_snapshot(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    services_status: Dict[str, ServiceStatus],
) -> Dict[str, Any]:
    """
    בונה snapshot מרוכז של מצב הדשבורד:

    * טיימסטמפ UTC
    * env/profile/run_id
    * app/service statuses
    * base_dashboard_context סריאלי
    * מידע מינימלי על user/host

    המטרה:
    -------
    - לשמור אותו ב-SqlStore לצורכי Audit / ניטור / Backfill.
    - לשמור עותק ב-session_state["dashboard_last_snapshot"].
    """
    env: EnvName = feature_flags.get("env", DEFAULT_ENV)  # type: ignore[assignment]
    profile: ProfileName = feature_flags.get("profile", DEFAULT_PROFILE)  # type: ignore[assignment]
    run_id = get_session_run_id()

    ts_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    base_ctx = st.session_state.get(SESSION_KEY_BASE_CTX)
    base_ctx_serialized = _make_json_safe(_serialize_base_ctx_for_snapshot(base_ctx))

    snapshot: Dict[str, Any] = {
        "ts_utc": ts_utc,
        "env": env,
        "profile": profile,
        "run_id": run_id,
        "app": services_status.get("app", {}),
        "services": {
            key: val
            for key, val in services_status.items()
            if key != "app"
        },
        "base_context": base_ctx_serialized,
        "user": feature_flags.get("user", RUNTIME_USER),
        "host": feature_flags.get("host", RUNTIME_HOST),
        "app_name": feature_flags.get("app_name", APP_NAME),
        "version": feature_flags.get("version", APP_VERSION),
    }

    return snapshot


def _save_snapshot_to_sql_store(
    app_ctx: "AppContext",
    snapshot: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """
    מנסה לשמור snapshot ב-SqlStore, אם קיים.

    אסטרטגיה:
    ---------
    1. מגלה SqlStore מתוך AppContext (sql_store / store / db / sql).
    2. מנסה אחת מכמה פונקציות:
        - save_dashboard_snapshot(snapshot)
        - save_snapshot(snapshot)
        - save_context_snapshot("dashboard", snapshot)
        - save_context("dashboard", snapshot)
    3. במקרה של TypeError – מנסה וריאציה אחרת.

    מחזיר:
        (success: bool, method_name_used: Optional[str])
    """
    services_map = _discover_services_mapping(app_ctx)
    has_store, store_obj = _probe_service(
        app_ctx,
        services_map,
        candidates=("sql_store", "store", "db", "sql"),
    )

    if not (has_store and store_obj is not None):
        logger.warning("No SqlStore available for snapshot; skipping persist.")
        return False, None

    methods_order: Sequence[str] = (
        "save_dashboard_snapshot",
        "save_snapshot",
        "save_context_snapshot",
        "save_context",
    )

    for name in methods_order:
        func = getattr(store_obj, name, None)
        if not callable(func):
            continue

        # 1) ניסיון: func(snapshot)
        try:
            func(snapshot)
            logger.info("Dashboard snapshot saved via SqlStore.%s(snapshot)", name)
            return True, name
        except TypeError:
            # 2) ניסיון: func('dashboard', snapshot)
            try:
                func("dashboard", snapshot)
                logger.info(
                    "Dashboard snapshot saved via SqlStore.%s('dashboard', snapshot)",
                    name,
                )
                return True, name
            except TypeError:
                continue
            except Exception as exc:
                logger.warning(
                    "SqlStore.%s('dashboard', snapshot) raised %s", name, exc
                )
        except Exception as exc:
            logger.warning("SqlStore.%s(snapshot) raised %s", name, exc)

    logger.warning(
        "No compatible save_* method found on SqlStore for dashboard snapshot."
    )
    return False, None


def trigger_dashboard_snapshot(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    services_status: Dict[str, ServiceStatus],
) -> Tuple[bool, Optional[str]]:
    """
    מפעיל Snapshot מלא של הדשבורד:

    Flow:
    -----
    1. בונה snapshot מרוכז (dict).
    2. שומר אותו ב-session_state["dashboard_last_snapshot"].
    3. מנסה לשמור אותו ב-SqlStore (אם קיים).
    4. מחזיר (success, method_name) עבור SqlStore.

    הערה:
    -----
    * גם אם SqlStore לא זמין – עדיין נשמור את snapshot ב-session_state,
      כך שניתן יהיה להשתמש בו ב-UI או בלוגים.
    """
    snapshot = _build_dashboard_snapshot(app_ctx, feature_flags, services_status)

    try:
        st.session_state[SESSION_KEY_LAST_SNAPSHOT] = snapshot
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to store dashboard_last_snapshot in session_state: %s", exc
        )

    success, method_name = _save_snapshot_to_sql_store(app_ctx, snapshot)
    if success:
        logger.info("Dashboard snapshot persisted via %s", method_name)
    else:
        logger.info("Dashboard snapshot stored in session_state only (no SqlStore).")

    return success, method_name


def _clear_streamlit_caches() -> None:
    """
    מנקה Cacheים של Streamlit (cache_data/cache_resource) בצורה בטוחה.

    לא נניח שהם בשימוש, לכן נעטוף ב-try/except.
    """
    # cache_data
    try:
        st.cache_data.clear()
        logger.info("Streamlit cache_data cleared.")
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to clear cache_data: %s", exc)

    # cache_resource
    try:
        st.cache_resource.clear()
        logger.info("Streamlit cache_resource cleared.")
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to clear cache_resource: %s", exc)


def _render_quick_actions(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    services_status: Dict[str, ServiceStatus],
) -> None:
    """
    מציג סט Quick Actions גלובליים בסיידבר:

    - 🔁 רענן דשבורד (st.rerun)
    - 🧹 נקה Cache (cache_data/cache_resource)
    - 💾 Snapshot now (שמירה ל-SqlStore אם קיים)

    כל פעולה מדווחת ל-Logger ומציגה feedback קצר ב-UI.
    """
    st.markdown("#### ⚡ Quick actions")

    col_refresh, col_cache = st.columns(2)
    with col_refresh:
        if st.button("🔁 רענן דשבורד", key="btn_dashboard_rerun"):
            logger.info("User requested dashboard rerun via Quick actions.")
            st.rerun()

    with col_cache:
        if st.button("🧹 נקה Cache", key="btn_dashboard_clear_cache"):
            logger.info("User requested cache clear via Quick actions.")
            _clear_streamlit_caches()
            st.success("Cache נוקה בהצלחה (cache_data + cache_resource).")

    # Snapshot – נציג רק אם יש SqlStore או אם רוצים Debug mode
    caps: Mapping[str, Any] = feature_flags.get("capabilities", {}) or {}
    has_sql = bool(caps.get("sql_store", False))
    show_debug_snapshot = bool(feature_flags.get("show_debug_info", False))

    if has_sql or show_debug_snapshot:
        if st.button("💾 Snapshot now", key="btn_dashboard_snapshot_now"):
            logger.info("User requested dashboard snapshot via Quick actions.")
            success, method_name = trigger_dashboard_snapshot(
                app_ctx, feature_flags, services_status
            )
            if success:
                st.success(
                    f"Snapshot נשמר בהצלחה (method={method_name or 'unknown'})."
                )
            else:
                st.warning(
                    "Snapshot נשמר רק ב-session_state "
                    "(SqlStore לא זמין או לא תומך בפורמט הזה)."
                )

    # ניתן להוסיף בעתיד כפתורים כמו "Export current view" / "Send to agent" וכו'.


# עדכון __all__ עבור חלק 11
try:
    __all__ += [
        "_serialize_base_ctx_for_snapshot",
        "_build_dashboard_snapshot",
        "_save_snapshot_to_sql_store",
        "trigger_dashboard_snapshot",
        "_clear_streamlit_caches",
        "_render_quick_actions",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_serialize_base_ctx_for_snapshot",
        "_build_dashboard_snapshot",
        "_save_snapshot_to_sql_store",
        "trigger_dashboard_snapshot",
        "_clear_streamlit_caches",
        "_render_quick_actions",
    ]
# =====================
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

    st.sidebar.markdown("### 🧭 App overview")

    st.sidebar.markdown(
        f"**{APP_ICON} {feature_flags.get('app_name', APP_NAME)}**  "
        f"`v{feature_flags.get('version', APP_VERSION)}`"
    )
    st.sidebar.caption(f"Env: `{env}` • Profile: `{profile}`")

    uptime_hours = app_status.get("uptime_hours")
    if isinstance(uptime_hours, (int, float)):
        st.sidebar.caption(f"Uptime: ~{uptime_hours:.2f}h")

    st.sidebar.caption(
        f"Host: `{app_status.get('host', RUNTIME_HOST)}` • "
        f"User: `{app_status.get('user', RUNTIME_USER)}`"
    )


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
    st.sidebar.markdown("#### 🧩 Services status")

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

    # SqlStore
    sql_icon = "✅" if sql.get("available") else "⭕"
    sql_backend = sql.get("backend") or "N/A"
    sql_hist = _format_bool_tristate(sql.get("has_history"), "Yes", "No", "Unknown")
    st.sidebar.caption(f"{sql_icon} SqlStore – backend: `{sql_backend}`, history: {sql_hist}")

    # Broker
    br_icon = "✅" if broker.get("available") else "⭕"
    br_mode = broker.get("mode") or "N/A"
    br_conn = _format_bool_tristate(broker.get("connected"), "Connected", "Disconnected", "Unknown")
    st.sidebar.caption(f"{br_icon} Broker – mode: `{br_mode}`, {br_conn}")

    # Market Data
    md_icon = "✅" if market.get("available") else "⭕"
    md_src = market.get("source") or "N/A"
    md_lat = (market.get("latency_mode") or "N/A").lower()
    st.sidebar.caption(f"{md_icon} Market Data – source: `{md_src}`, latency: `{md_lat}`")

    # Engines – condensed line
    risk_icon = "✅" if risk.get("available") else "⭕"
    macro_icon = "✅" if macro.get("available") else "⭕"
    agents_icon = "✅" if agents.get("available") else "⭕"
    fv_icon = "✅" if fv.get("available") else "⭕"

    st.sidebar.caption(
        f"{risk_icon} Risk • {macro_icon} Macro • {agents_icon} Agents • {fv_icon} Fair Value"
    )

    # Backtest / Optimizer
    bt_icon = "✅" if bt.get("available") else "⭕"
    opt_icon = "✅" if opt.get("available") else "⭕"
    mo_icon = "✅" if meta_opt.get("available") else "⭕"
    st.sidebar.caption(
        f"{bt_icon} Backtester • {opt_icon} Optimizer • {mo_icon} Meta-Opt"
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


# עדכון __all__ עבור חלק 12
try:
    __all__ += [
        "_format_bool_tristate",
        "_render_app_overview_sidebar",
        "_render_services_status_sidebar",
        "_render_sidebar_debug_section",
        "_render_global_sidebar",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_format_bool_tristate",
        "_render_app_overview_sidebar",
        "_render_services_status_sidebar",
        "_render_sidebar_debug_section",
        "_render_global_sidebar",
    ]
# =====================
# Part 13/35 – Active tabs builder (profile-aware ordering & enable logic)
# =====================

from collections import OrderedDict


def _is_tab_enabled_for_profile(meta: TabMeta, profile: ProfileName) -> bool:
    """
    מחליט האם טאב ספציפי פעיל עבור profile נתון.

    לוגיקה:
    -------
    1. קודם מסתכלים על meta.profile_overrides:
        - אם profile נמצא במיפוי:
            * True  → הטאב **חייב** להיות פעיל (גם אם meta.enabled=False).
            * False → הטאב מושבת עבור הפרופיל הזה, גם אם meta.enabled=True.
    2. אחרת – משתמשים ב-meta.enabled הגלובלי.
    """
    overrides = meta.profile_overrides or {}
    if profile in overrides:
        return bool(overrides[profile])
    return bool(meta.enabled)


def _compute_tab_sort_key(
    meta: TabMeta,
    profile: ProfileName,
    profile_order: Sequence[TabKey],
) -> Tuple[int, int, str]:
    """
    מחזיר מפתח מיון (tuple) עבור טאב, כדי לקבוע את סדרו הסופי.

    הקריטריונים:
    -------------
    1. מיקום ב-profile_order:
        - אם key נמצא ב-profile_order → index שלו.
        - אם לא → index גדול (len(profile_order) + משהו קטן לפי order).
    2. meta.order – מספר סדר גלובלי לוגי (לפי _BASE_TAB_DEFS).
    3. meta.key – שובר שוויון אלפביתי יציב.
    """
    profile_index_map = {k: i for i, k in enumerate(profile_order)}
    base_index = profile_index_map.get(meta.key, len(profile_order) + meta.order)
    return base_index, int(meta.order), str(meta.key)


def build_active_tabs(
    registry: TabRegistry,
    feature_flags: FeatureFlags,
) -> Tuple[List[TabMeta], List[TabKey], List[TabLabel]]:
    """
    בונה רשימת טאבים פעילים לפי:

    - TabRegistry (המצב הגלובלי – capabilities, requires, enabled).
    - profile (trading/research/risk/macro/monitoring/default).
    - סדר פרופיל (get_profile_tab_order).

    מחזיר:
    -------
    active_tabs: List[TabMeta]
        רשימת TabMeta לפי סדר סופי.

    active_keys: List[TabKey]
        רשימת המפתחות (keys) לפי אותו סדר – שימושי ל-router/ניווט.

    active_labels: List[TabLabel]
        רשימת ה-labels (שמות הטאבים) לפי אותו סדר – לשימוש ב-st.tabs.
    """
    env: EnvName = feature_flags.get("env", DEFAULT_ENV)  # type: ignore[assignment]
    profile: ProfileName = feature_flags.get("profile", DEFAULT_PROFILE)  # type: ignore[assignment]

    profile_order = get_profile_tab_order(profile)

    candidates: List[TabMeta] = list(registry.values())
    enabled_for_profile: List[TabMeta] = [
        meta for meta in candidates if _is_tab_enabled_for_profile(meta, profile)
    ]

    # מיון לפי Profile order + meta.order + key
    enabled_for_profile.sort(
        key=lambda m: _compute_tab_sort_key(m, profile, profile_order)
    )

    active_tabs: List[TabMeta] = enabled_for_profile
    active_keys: List[TabKey] = [m.key for m in active_tabs]
    active_labels: List[TabLabel] = [m.label for m in active_tabs]

    logger.info(
        "Active tabs for env=%s, profile=%s: %s",
        env,
        profile,
        active_keys,
    )

    if feature_flags.get("show_debug_info"):
        logger.debug(
            "Active tabs debug snapshot: %s",
            [
                {
                    "key": m.key,
                    "label": m.label,
                    "group": m.group,
                    "order": m.order,
                    "requires": list(m.requires),
                }
                for m in active_tabs
            ],
        )

    return active_tabs, active_keys, active_labels


def build_tab_index_map(
    active_keys: Sequence[TabKey],
) -> Dict[TabKey, int]:
    """
    בונה מיפוי TabKey → index בתוך רשימת הטאבים הפעילים.

    שימושי ל:
    - איתור אינדקס עבור nav_target.tab_key.
    - לוגים / Debug.
    """
    return {key: idx for idx, key in enumerate(active_keys)}


def _get_nav_payload_for_tab(
    tab_key: TabKey,
    nav_tab_key: Optional[TabKey],
    nav_payload: Optional[NavPayload],
) -> Optional[NavPayload]:
    """
    מחליט איזה payload (אם בכלל) להעביר לטאב ספציפי.

    - אם nav_tab_key == tab_key → מחזיר nav_payload (או {}).
    - אחרת → מחזיר None (לא מעבירים payload לטאבים אחרים).
    """
    if nav_tab_key is None:
        return None
    if tab_key != nav_tab_key:
        return None
    return nav_payload or {}


# עדכון __all__ עבור חלק 13
try:
    __all__ += [
        "build_active_tabs",
        "build_tab_index_map",
        "_is_tab_enabled_for_profile",
        "_compute_tab_sort_key",
        "_get_nav_payload_for_tab",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "build_active_tabs",
        "build_tab_index_map",
        "_is_tab_enabled_for_profile",
        "_compute_tab_sort_key",
        "_get_nav_payload_for_tab",
    ]

# =====================
# Part 14/35 – Safe tab rendering (per-tab error isolation & diagnostics)
# =====================

import traceback


@dataclass
class TabErrorInfo:
    """
    מידע על שגיאה שקרתה בתוך טאב ספציפי.

    נשמר (אופציונלית) בתוך session_state כדי:
    - להציג למשתמש פאנל שגיאה עשיר בטאב עצמו.
    - לאפשר Debug מהיר (stacktrace) בפרופיל dev/research.
    - למנוע קריסה של כל הדשבורד בגלל טאב אחד.

    שדות:
    -----
    tab_key:
        מפתח הטאב שבו קרתה השגיאה.
    ts_utc:
        טיימסטמפ UTC של השגיאה (isoformat).
    exc_type:
        שם סוג השגיאה (ValueError, RuntimeError וכו').
    message:
        str(exc) מקוצר.
    traceback_str:
        stacktrace מלא כטקסט – מוצג רק אם show_debug_info=True.
    payload_keys:
        רשימת מפתחות מה-nav_payload (אם היה כזה).
    env / profile:
        ה-context שבו השגיאה התרחשה.
    run_id:
        מזהה ריצה של הדשבורד (לסנכרון עם לוגים/SqlStore).
    """

    tab_key: TabKey
    ts_utc: str
    exc_type: str
    message: str
    traceback_str: str
    payload_keys: Optional[List[str]] = None
    env: Optional[str] = None
    profile: Optional[str] = None
    run_id: Optional[str] = None


SESSION_KEY_LAST_TAB_ERRORS: str = "dashboard_last_tab_errors"


def _ensure_tab_errors_state() -> Dict[TabKey, TabErrorInfo]:
    """
    מבטיח שקיימת ב-session_state מפה של שגיאות per-tab:

        { tab_key: TabErrorInfo }

    אם לא קיימת – יוצר dict ריק.
    """
    try:
        obj = st.session_state.get(SESSION_KEY_LAST_TAB_ERRORS)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to access tab errors state: %s", exc)
        return {}

    if not isinstance(obj, dict):
        obj = {}
        try:
            st.session_state[SESSION_KEY_LAST_TAB_ERRORS] = obj
        except Exception:
            pass
    return obj  # type: ignore[return-value]


def _record_tab_error(
    tab_key: TabKey,
    exc: BaseException,
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload],
) -> TabErrorInfo:
    """
    יוצר TabErrorInfo משגיאה שקרתה בטאב, ושומר אותו ב-session_state.

    היסטוריית שגיאות per-tab עוזרת:
    - לזהות טאב "רעיל" שחוזר על שגיאה.
    - להציג למשתמש מידע מסודר (כולל stacktrace בפרופיל dev/research).
    - לנטר שגיאות לאורך זמן (אם נשמור בסופו של דבר ב-SqlStore).
    """
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    env: EnvName = feature_flags.get("env", DEFAULT_ENV)  # type: ignore[assignment]
    profile: ProfileName = feature_flags.get("profile", DEFAULT_PROFILE)  # type: ignore[assignment]
    run_id = get_session_run_id()

    payload_keys: Optional[List[str]] = None
    if isinstance(nav_payload, Mapping):
        payload_keys = [str(k) for k in nav_payload.keys()]

    info = TabErrorInfo(
        tab_key=tab_key,
        ts_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        exc_type=type(exc).__name__,
        message=str(exc),
        traceback_str=tb_str,
        payload_keys=payload_keys,
        env=str(env),
        profile=str(profile),
        run_id=run_id,
    )

    try:
        errors_map = _ensure_tab_errors_state()
        errors_map[tab_key] = info
        st.session_state[SESSION_KEY_LAST_TAB_ERRORS] = errors_map
    except Exception as e:  # pragma: no cover
        logger.debug("Failed to store TabErrorInfo in session_state: %s", e)

    return info


def _render_tab_error_panel(
    tab_key: TabKey,
    error_info: TabErrorInfo,
    feature_flags: FeatureFlags,
) -> None:
    """
    מציג Panel אדום בטאב שנכשל, במקום להפיל את כל הדשבורד.

    התנהגות:
    --------
    - תמיד: הודעת שגיאה קצרה, זמני, env/profile, run_id.
    - אם show_debug_info=True:
        * expander עם stacktrace מלא.
        * הצגת payload_keys (אם יש).
    """
    st.markdown(f"### ❌ Tab error – `{tab_key}`")

    with st.container():
        st.error(
            f"קרתה שגיאה בעת רנדר של הטאב.\n\n"
            f"- Type: `{error_info.exc_type}`  \n"
            f"- Message: `{error_info.message}`  \n"
            f"- Time (UTC): `{error_info.ts_utc}`  \n"
            f"- Env/Profile: `{error_info.env}` / `{error_info.profile}`  \n"
            f"- Run ID: `{error_info.run_id}`"
        )

        if error_info.payload_keys:
            st.caption(
                "Payload keys: " + ", ".join(f"`{k}`" for k in error_info.payload_keys)
            )

        if feature_flags.get("show_debug_info"):
            with st.expander("Stacktrace (debug)", expanded=False):
                st.code(error_info.traceback_str, language="python")


def safe_render_tab(
    meta: TabMeta,
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload],
) -> None:
    """
    עטיפת רנדרינג בטוחה לכל טאב:

    מטרות:
    -------
    - לא לאפשר לטאב אחד להפיל את כל האפליקציה.
    - לרשום לוגים ברמת קרן (כולל stacktrace) לכל שגיאה.
    - להציג למשתמש הודעת שגיאה מסודרת + אפשרות ל-Debug.

    Flow:
    -----
    1. לוג INFO לכניסה לטאב (נעזר ב-_log_tab_entry שכבר קיים).
    2. ניסיון לקרוא meta.renderer(app_ctx, feature_flags, nav_payload_for_tab).
    3. במקרה של Exception:
        - לוג ERROR + logger.exception(stacktrace).
        - יצירת TabErrorInfo + שמירה ב-session_state.
        - הצגת Panel בטאב עצמו עם פרטים.
    """
    tab_key = meta.key
    _log_tab_entry(tab_key, feature_flags, nav_payload)

    try:
        meta.renderer(app_ctx, feature_flags, nav_payload)
        # אם הכל עבר בסדר – אפשר (אופציונלית) לנקות שגיאה היסטורית לטאב הזה
        try:
            errors_map = st.session_state.get(SESSION_KEY_LAST_TAB_ERRORS, {})
            if isinstance(errors_map, dict) and tab_key in errors_map:
                del errors_map[tab_key]
                st.session_state[SESSION_KEY_LAST_TAB_ERRORS] = errors_map
        except Exception:  # pragma: no cover
            pass

    except Exception as exc:
        # לוג מפורט
        logger.error(
            "Exception while rendering tab '%s': %s", tab_key, exc, exc_info=True
        )

        # שמירת מידע מסודר ב-session_state
        info = _record_tab_error(tab_key, exc, feature_flags, nav_payload)

        # הצגת Panel אדום בתוך הטאב (UI)
        _render_tab_error_panel(tab_key, info, feature_flags)


# עדכון __all__ עבור חלק 14
try:
    __all__ += [
        "TabErrorInfo",
        "SESSION_KEY_LAST_TAB_ERRORS",
        "_ensure_tab_errors_state",
        "_record_tab_error",
        "_render_tab_error_panel",
        "safe_render_tab",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "TabErrorInfo",
        "SESSION_KEY_LAST_TAB_ERRORS",
        "_ensure_tab_errors_state",
        "_record_tab_error",
        "_render_tab_error_panel",
        "safe_render_tab",
    ]
# =====================
# Part 15/35 – Tabs view router (st.tabs orchestration & perf tracking)
# =====================

import time  # מדידת זמנים לרנדר של כל טאב (HF-grade instrumentation)

SESSION_KEY_TAB_TIMINGS: str = "dashboard_tab_timings"


def _ensure_tab_timings_state() -> Dict[TabKey, Dict[str, Any]]:
    """
    מבטיח שב-session_state קיים מבנה למדידת ביצועים per-tab:

        {
            "<tab_key>": {
                "last": float,   # זמן הריצה האחרון (שניות)
                "count": int,    # כמה פעמים נמדד
                "avg": float,    # ממוצע זמנים (running average)
            },
            ...
        }

    מטרות:
    -------
    - ניטור ביצועים ברמת טאב (מי כבד, מי קל).
    - אפשרות להציג בעתיד Dashboard לביצועים (Performance Tab).
    - להחזיק מידע גם בצד ה-Desktop (reuse בעתיד).
    """
    try:
        obj = st.session_state.get(SESSION_KEY_TAB_TIMINGS)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to access tab timings state: %s", exc)
        return {}

    if not isinstance(obj, dict):
        obj = {}
        try:
            st.session_state[SESSION_KEY_TAB_TIMINGS] = obj
        except Exception:
            pass
    return obj  # type: ignore[return-value]


def _record_tab_timing(tab_key: TabKey, elapsed_seconds: float) -> None:
    """
    רושם מדידת זמן ריצה אחת לטאב מסוים.

    לוגיקה:
    -------
    - שומר per-tab:
        * last – הזמן האחרון שנמדד.
        * count – כמה מדידות היו.
        * avg – ממוצע רץ (ללא שמירת כל ההיסטוריה).
    - לא זורק שגיאות – מיועד להיות "fire and forget".
    """
    try:
        timings = _ensure_tab_timings_state()
        rec = timings.get(tab_key) or {"last": 0.0, "count": 0, "avg": 0.0}

        try:
            prev_count = int(rec.get("count", 0))
        except Exception:
            prev_count = 0

        try:
            prev_avg = float(rec.get("avg", 0.0))
        except Exception:
            prev_avg = 0.0

        new_count = prev_count + 1
        last = float(elapsed_seconds)
        # ממוצע חדש: (avg * (n-1) + last) / n
        new_avg = (prev_avg * prev_count + last) / float(new_count)

        rec.update(
            {
                "last": last,
                "count": new_count,
                "avg": new_avg,
            }
        )
        timings[tab_key] = rec
        st.session_state[SESSION_KEY_TAB_TIMINGS] = timings

        logger.debug(
            "Tab timing updated: key=%s, last=%.4fs, avg=%.4fs (count=%d)",
            tab_key,
            last,
            new_avg,
            new_count,
        )
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to record tab timing for '%s': %s", tab_key, exc)


def _render_tabs_view(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    tab_registry: TabRegistry,
) -> None:
    """
    Router מרכזי של הדשבורד – אחראי על:

    1. בניית רשימת טאבים פעילים לפי:
       - capabilities (TabRegistry / FeatureFlags).
       - profile (trading/research/risk/macro/monitoring).
    2. צריכת nav_target (אם מוגדר) והפצת nav_payload לטאב היעד בלבד.
    3. יצירת st.tabs(...) לפי labels.
    4. רנדר בטוח של כל טאב (safe_render_tab) + מדידת ביצועים per-tab.
    5. עדכון last_active_tab_key וניטור ניווט (ברמת לוג ו-state).

    שים לב:
    -------
    * כל הטאבים נרנדרים בכל ריצה של Streamlit (כפי שמקובל ב-st.tabs),
      אבל nav_payload מועבר רק לטאב היעד בפועל.
    * last_active_tab_key מתבסס בעיקר על nav_target (Flow מונע-קונטרול),
      ולא על "הטאב שהמשתמש לחץ עליו" (שלא נחשף ישירות ע"י Streamlit).
    """
    if not tab_registry:
        st.error(
            "No tabs registered in TabRegistry.  \n"
            "בדוק את compute_feature_flags / build_tab_registry."
        )
        logger.error("Tab registry is empty – nothing to render.")
        return

    env_sess, profile_sess = _get_current_env_profile_from_session()
    env_ff: EnvName = feature_flags.get("env", env_sess)  # type: ignore[assignment]
    profile_ff: ProfileName = feature_flags.get("profile", profile_sess)  # type: ignore[assignment]

    # 1) רשימת טאבים פעילים (כולל מיון לפי profile)
    active_tabs, active_keys, active_labels = build_active_tabs(
        tab_registry, feature_flags
    )
    if not active_tabs:
        st.warning(
            "אין אף טאב פעיל עבור הפרופיל/סביבה הנוכחיים.  \n"
            "נסה לשנות env/profile או להפעיל יכולות נוספות."
        )
        logger.warning(
            "No active tabs for env=%s, profile=%s (registry size=%d).",
            env_ff,
            profile_ff,
            len(tab_registry),
        )
        return

    active_key_set = set(active_keys)
    tab_index_map = build_tab_index_map(active_keys)

    # 2) nav_target – צריכה בטוחה בהתאם לטאבים הפעילים
    nav_tab_key, nav_payload = _consume_nav_target(active_key_set)
    if nav_tab_key is not None:
        set_last_active_tab_key(nav_tab_key)
        nav_index = tab_index_map.get(nav_tab_key)
    else:
        nav_index = None

    logger.info(
        "Tabs view: env=%s, profile=%s, active=%s, nav_target=%s (index=%s)",
        env_ff,
        profile_ff,
        active_keys,
        nav_tab_key,
        nav_index,
    )

    # 3) יצירת אובייקטי הטאבים של Streamlit
    tab_objects = st.tabs(active_labels)

    # 4) רנדר בטוח של כל טאב + מדידת ביצועים
    for idx, (meta, tab_obj) in enumerate(zip(active_tabs, tab_objects)):
        tab_key = meta.key

        # nav_payload לטאב הנוכחי (אם זה יעד הניווט)
        tab_payload = _get_nav_payload_for_tab(tab_key, nav_tab_key, nav_payload)

        with tab_obj:
            # מדידת זמן ריצה פר טאב (HF-grade instrumentation)
            t0 = time.perf_counter()
            safe_render_tab(meta, app_ctx, feature_flags, tab_payload)
            t1 = time.perf_counter()
            elapsed = t1 - t0

            _record_tab_timing(tab_key, elapsed)

            # מידע Debug עדין – רק בפרופילים מתאימים
            if feature_flags.get("show_debug_info"):
                st.caption(f"⏱ Render time for `{tab_key}`: ~{elapsed:.3f}s")

    # אם לא היה nav_target בריצה הזו – נשמור לפחות את "הטאב הראשון" כ-last_active,
    # כדי לתת ערך הגיוני כברירת מחדל (אבל לא נדרוס בחירה קיימת).
    if nav_tab_key is None:
        current_last = get_last_active_tab_key(default=active_keys[0])
        if current_last not in active_key_set:
            # במצב לא עקבי – נכפה ערך סביר
            set_last_active_tab_key(active_keys[0])


# עדכון __all__ עבור חלק 15
try:
    __all__ += [
        "SESSION_KEY_TAB_TIMINGS",
        "_ensure_tab_timings_state",
        "_record_tab_timing",
        "_render_tabs_view",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_TAB_TIMINGS",
        "_ensure_tab_timings_state",
        "_record_tab_timing",
        "_render_tabs_view",
    ]

# =====================
# Part 16/35 – DashboardRuntime orchestrator (single source of truth)
# =====================

SESSION_KEY_RUNTIME: str = "_dashboard_runtime"


@dataclass
class DashboardRuntime:
    """
    אובייקט Orchestrator יחיד לדשבורד – "מוח" מרוכז ברמת קרן גידור.

    תפקידים:
    ---------
    - להחזיק את כל ה-State המחושב ברמת האפליקציה:
        * AppContext
        * FeatureFlags (env/profile/capabilities/tabs/flags)
        * Services status (SqlStore/Broker/Macro/Risk/Agents/...)
        * TabRegistry (כל המטא-דאטה של הטאבים)
        * env/profile/run_id הנוכחיים

    - לשמש כנקודת כניסה אחת ל-Web Dashboard:
        * מחושב פעם אחת (per-session) ונשמר ב-session_state.
        * מאפשר רענון מבוקר של services_status / tab_registry בעתיד.

    - להוות שכבה משותפת ל-Web + Desktop:
        * קל לחבר את אותה לוגיקה גם ל-Qt/desktop דרך API דומה.
    """

    app_ctx: "AppContext"
    feature_flags: FeatureFlags
    services_status: Dict[str, ServiceStatus]
    tab_registry: TabRegistry
    env: EnvName
    profile: ProfileName
    run_id: str

    def refresh_services_status(self) -> None:
        """
        מרענן את תמונת מצב השירותים (SqlStore/Broker/Macro/Risk/Agents/...).

        שימושי כאשר:
        - בוצעו שינויים ברמת חיבור (למשל Broker התחבר/התנתק).
        - רוצים לייצר Snapshot חדש אחרי שינוי מצב.
        """
        self.services_status = _get_services_status(self.app_ctx, self.feature_flags)
        logger.info(
            "DashboardRuntime: services_status refreshed for env=%s, profile=%s",
            self.env,
            self.profile,
        )

    def refresh_tab_registry(self) -> None:
        """
        מרענן את TabRegistry על בסיס feature_flags + capabilities עדכניים.

        שימושי כאשר:
        - capabilities השתנו (למשל הופעלו Agents / Macro / SqlStore בזמן ריצה).
        - רוצים לאפשר/לכבות טאבים בצורה דינמית אחרי שינוי קונפיג.
        """
        self.tab_registry = build_tab_registry(self.app_ctx, self.feature_flags)
        logger.info(
            "DashboardRuntime: tab_registry refreshed (entries=%d)",
            len(self.tab_registry),
        )

    @property
    def capabilities(self) -> Mapping[str, Any]:
        """
        מחזיר capabilities כפי שנגזרו מ-AppContext בתוך feature_flags.
        """
        return self.feature_flags.get("capabilities", {}) or {}

    @property
    def domains(self) -> Mapping[str, bool]:
        """
        מחזיר Domains לוגיים (risk/macro/signals/portfolio/...) מתוך feature_flags.
        """
        return self.feature_flags.get("domains", {}) or {}

    @property
    def tabs_flags(self) -> Mapping[str, Any]:
        """
        מחזיר מפת flags בסיסית עבור טאבים מתוך feature_flags["tabs"].
        """
        return self.feature_flags.get("tabs", {}) or {}

    @property
    def desktop_integration_enabled(self) -> bool:
        """
        האם יש אינדיקציה ליכולת אינטגרציה עם Desktop/Qt.
        """
        return bool(self.feature_flags.get("desktop_integration", False))

    def to_debug_dict(self) -> Dict[str, Any]:
        """
        מייצר ייצוג Debug קומפקטי של ה-Runtime – שימושי ללוגים או לטאב Debug.

        לא כולל אובייקטים כבדים (app_ctx עצמו וכו'), אלא רק Metadata.
        """
        return {
            "env": self.env,
            "profile": self.profile,
            "run_id": self.run_id,
            "app_name": self.feature_flags.get("app_name", APP_NAME),
            "version": self.feature_flags.get("version", APP_VERSION),
            "capabilities": dict(self.capabilities),
            "tabs_enabled": {
                key: meta.enabled for key, meta in self.tab_registry.items()
            },
        }


def create_dashboard_runtime(app_ctx: "AppContext") -> DashboardRuntime:
    """
    יוצר DashboardRuntime חדש מ-AppContext "נא":

    Flow:
    -----
    1. compute_feature_flags(app_ctx) – זיהוי env/profile/capabilities/flags.
    2. bootstrap_session(app_ctx, feature_flags) – אתחול session_state.
    3. feature_flags = get_feature_flags_from_session() – מקור אמת סופי.
    4. services_status = _get_services_status(app_ctx, feature_flags).
    5. tab_registry = build_tab_registry(app_ctx, feature_flags).
    6. run_id = get_session_run_id().
    7. env/profile מתוך feature_flags.

    במידה וקורה כשל ברמה כלשהי – השגיאה נזרקת למעלה, כי זה
    שלב קריטי לאתחול האפליקציה.
    """
    # 1) Feature flags
    raw_flags = compute_feature_flags(app_ctx)

    # 2) Bootstrap session (env/profile/flags/base_ctx/...)
    bootstrap_session(app_ctx, raw_flags)

    # 3) Feature flags אחרי bootstrap (מקור אמת – יכול לכלול התאמות)
    feature_flags = get_feature_flags_from_session()

    # 4) Services status
    services_status = _get_services_status(app_ctx, feature_flags)

    # 5) Tab registry
    tab_registry = build_tab_registry(app_ctx, feature_flags)

    # 6) Run ID (per-session)
    run_id = get_session_run_id()

    # 7) Env/Profile סופיים
    env: EnvName = feature_flags.get("env", DEFAULT_ENV)  # type: ignore[assignment]
    profile: ProfileName = feature_flags.get("profile", DEFAULT_PROFILE)  # type: ignore[assignment]

    rt = DashboardRuntime(
        app_ctx=app_ctx,
        feature_flags=feature_flags,
        services_status=services_status,
        tab_registry=tab_registry,
        env=env,
        profile=profile,
        run_id=run_id,
    )

    logger.info(
        "DashboardRuntime created: env=%s, profile=%s, run_id=%s, "
        "tabs=%d, capabilities=%s",
        env,
        profile,
        run_id,
        len(tab_registry),
        list(rt.capabilities.keys()),
    )

    return rt


def get_dashboard_runtime(
    app_ctx: "AppContext",
    force_rebuild: bool = False,
) -> DashboardRuntime:
    """
    מחזיר DashboardRuntime קיים מ-session_state (אם אפשר), או בונה חדש.

    התנהגות:
    --------
    - אם force_rebuild=True → תמיד בונה חדש דרך create_dashboard_runtime.
    - אחרת:
        * מנסה לקרוא SESSION_KEY_RUNTIME.
        * אם קיים והוא DashboardRuntime → מחזיר אותו.
        * אם חסר/טיפוס שגוי → בונה חדש ושומר.

    הערה:
    -----
    * ההנחה: env/profile לא השתנו מאז האתחול. אם תרצה לכבד שינוי env/profile
      בזמן ריצה (למשל דרך query params) – השתמש ב-ensure_dashboard_runtime(...)
      שמתייחס לכך.
    """
    if not force_rebuild:
        try:
            existing = st.session_state.get(SESSION_KEY_RUNTIME)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to access SESSION_KEY_RUNTIME: %s", exc)
            existing = None

        if isinstance(existing, DashboardRuntime):
            return existing

    runtime = create_dashboard_runtime(app_ctx)

    try:
        st.session_state[SESSION_KEY_RUNTIME] = runtime
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to store DashboardRuntime in session_state: %s", exc
        )

    return runtime


def ensure_dashboard_runtime(app_ctx: "AppContext") -> DashboardRuntime:
    """
    מוודא שקיים DashboardRuntime שמתאים ל-env/profile הנוכחיים:

    Flow:
    -----
    1. מנסה לשלוף runtime קיים מה-session_state.
    2. מזהה env/profile "רצויים" דרך detect_env_profile(app_ctx).
    3. אם יש runtime קיים וה-env/profile שלו תואמים ל"רצויים":
        -> מחזיר אותו.
    4. אחרת:
        -> בונה DashboardRuntime חדש דרך create_dashboard_runtime
           ומחליף את הישן.

    המטרה:
    -------
    - לאפשר שינוי env/profile בזמן ריצה (למשל דרך config/query),
      בלי להיתקע על Runtime ישן שלא מתאים.
    """
    desired_env, desired_profile = detect_env_profile(app_ctx)

    try:
        existing = st.session_state.get(SESSION_KEY_RUNTIME)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to access SESSION_KEY_RUNTIME in ensure: %s", exc)
        existing = None

    if isinstance(existing, DashboardRuntime):
        if existing.env == desired_env and existing.profile == desired_profile:
            return existing

        logger.info(
            "DashboardRuntime env/profile changed "
            "(was env=%s, profile=%s; now env=%s, profile=%s) – rebuilding.",
            existing.env,
            existing.profile,
            desired_env,
            desired_profile,
        )

    runtime = create_dashboard_runtime(app_ctx)
    try:
        st.session_state[SESSION_KEY_RUNTIME] = runtime
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to store rebuilt DashboardRuntime: %s", exc)

    return runtime


# עדכון __all__ עבור חלק 16
try:
    __all__ += [
        "SESSION_KEY_RUNTIME",
        "DashboardRuntime",
        "create_dashboard_runtime",
        "get_dashboard_runtime",
        "ensure_dashboard_runtime",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_RUNTIME",
        "DashboardRuntime",
        "create_dashboard_runtime",
        "get_dashboard_runtime",
        "ensure_dashboard_runtime",
    ]
# =====================
# Part 17/35 – URL query params bridge (env/profile + deep-link navigation)
# =====================

# הרעיון:
# --------
# מאפשר:
#   1. שליטה על env/profile דרך ה-URL:
#        ?env=live&profile=trading
#   2. Deep-link לטאבים עם payload:
#        ?tab=backtest&pair=AAPL/MSFT&preset=smoke&mode=wf
#   3. ניווט ראשוני "חכם" – query params → nav_target → Router.
#
# זה קריטי כדי:
#   - לשתף לינקים מדויקים (למשל למקרו/ריסק/Backtest מסוים).
#   - לאפשר אינטגרציה עם Desktop/Agents שמייצרים URLs לטאבים ספציפיים.

SESSION_KEY_QUERY_INTENT: str = "_dashboard_query_intent"


def _get_query_params() -> Mapping[str, List[str]]:
    """
    עטיפה בטוחה סביב st.experimental_get_query_params.

    מחזירה:
        Mapping[str, List[str]]
    """
    try:
        return st.query_params
    except Exception as exc:  # pragma: no cover
        logger.debug("experimental_get_query_params failed: %s", exc)
        return {}


def _first_query_value(
    params: Mapping[str, Sequence[str]],
    key: str,
) -> Optional[str]:
    """
    מחזיר את הערך הראשון של פרמטר query נתון, אם קיים ולא ריק.

    לדוגמה:
        params = {"env": ["live"], "pair": ["AAPL/MSFT"]}
        _first_query_value(params, "env") → "live"
    """
    try:
        values = params.get(key)
    except Exception:
        return None

    if not values:
        return None

    try:
        first = values[0]
    except Exception:
        return None

    if first is None:
        return None

    s = str(first).strip()
    return s or None


def parse_dashboard_query_params() -> Dict[str, Any]:
    """
    מפרש את query params של הדשבורד למבנה נוח:

    תומך בפרמטרים:
    ---------------
    - env:        env נדרש (dev/live/paper/research/backtest/...)
    - profile:    profile נדרש (trading/research/risk/macro/monitoring)
    - tab:        טאב יעד (home/backtest/risk/macro/...)
    - pair:       זוג (למשל "AAPL/MSFT")
    - preset:     preset עבור Backtest/Scan (למשל "smoke")
    - mode:       מצב Backtest (למשל "wf", "single")
    - portfolio:  מזהה פורטפוליו ("default", "fund_core" וכו')
    - view:       תת-תצוגה (למשל "limits", "overview")
    - macro_view: תצוגת מקרו (למשל "regimes", "indicators")

    הפלט:
    -----
    dict עם מפתחות:
        "env", "profile", "tab", "pair", "preset", "mode",
        "portfolio_id", "view", "macro_view"
    """
    params = _get_query_params()

    env_raw = _first_query_value(params, "env")
    profile_raw = _first_query_value(params, "profile")
    tab_raw = _first_query_value(params, "tab")

    pair_raw = _first_query_value(params, "pair")
    preset_raw = _first_query_value(params, "preset")
    mode_raw = _first_query_value(params, "mode")
    portfolio_raw = _first_query_value(params, "portfolio")
    view_raw = _first_query_value(params, "view")
    macro_view_raw = _first_query_value(params, "macro_view")

    parsed: Dict[str, Any] = {
        "env": env_raw,
        "profile": profile_raw,
        "tab": tab_raw,
        "pair": pair_raw,
        "preset": preset_raw,
        "mode": mode_raw,
        "portfolio_id": portfolio_raw,
        "view": view_raw,
        "macro_view": macro_view_raw,
    }

    # אם אין אף פרמטר משמעותי – נחזיר dict ריק כדי שלא נדרוס כלום.
    if not any(parsed.values()):
        return {}

    logger.info("Dashboard query params parsed: %s", parsed)
    return parsed


def _apply_query_env_profile_overrides(parsed: Mapping[str, Any]) -> None:
    """
    מיישם env/profile מה-URL אל session_state לפני יצירת ה-Runtime.

    אסטרטגיה:
    ----------
    - אם parsed["env"] קיים → מנרמל לערך EnvName ומשתמש בו כ-session_state["env"].
    - אם parsed["profile"] קיים → מנרמל לערך ProfileName ומשתמש בו כ-session_state["profile"].

    בכך, detect_env_profile(app_ctx) יראה את הערכים האלה בעדיפות ראשונה.
    """
    raw_env = parsed.get("env")
    raw_profile = parsed.get("profile")

    if raw_env:
        env_norm = _normalize_env(str(raw_env))
        try:
            st.session_state[SESSION_KEY_ENV] = env_norm
            st.session_state["env"] = env_norm
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to apply query env override to session_state: %s", exc)
        else:
            logger.info("Applied env override from query params: env=%s", env_norm)

    if raw_profile:
        profile_norm = _normalize_profile(str(raw_profile))
        try:
            st.session_state[SESSION_KEY_PROFILE] = profile_norm
            st.session_state["profile"] = profile_norm
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Failed to apply query profile override to session_state: %s", exc
            )
        else:
            logger.info(
                "Applied profile override from query params: profile=%s", profile_norm
            )


def _build_query_nav_payload(parsed: Mapping[str, Any]) -> Optional[NavPayload]:
    """
    בונה NavPayload מתוך הפרמטרים שקיבלנו מה-URL.

    לוגיקה:
    -------
    - אם אין אף שדה "עסקי" (pair/preset/mode/portfolio_id/view/macro_view) → מחזיר None.
    - אחרת → מחזיר dict עם המפתחות הלא-ריקים.
    """
    keys = ("pair", "preset", "mode", "portfolio_id", "view", "macro_view")
    payload: NavPayload = {}

    for key in keys:
        val = parsed.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if not s:
            continue
        payload[key] = s

    if not payload:
        return None

    return payload


def _store_query_nav_intent(parsed: Mapping[str, Any]) -> None:
    """
    שומר Intent לניווט ראשוני מתוך ה-URL ב-session_state:

    מבנה:
        SESSION_KEY_QUERY_INTENT = {
            "tab_key": "<tab_key>",
            "payload": {...} or None,
            "applied": bool  # האם כבר יוצר nav_target בפועל
        }

    חשוב:
    -----
    * איננו מתקשרים כאן ישירות עם set_nav_target, משום שעדיין אין לנו
      Runtime/TabRegistry – יתכן שהטאב לא זמין בפרופיל הנוכחי.
    * היישום בפועל ל-nav_target יתרחש אחרי יצירת DashboardRuntime.
    """
    tab_raw = parsed.get("tab")
    if not tab_raw:
        return

    tab_key = str(tab_raw).strip()
    if not tab_key:
        return

    payload = _build_query_nav_payload(parsed)
    intent = {
        "tab_key": tab_key,
        "payload": payload,
        "applied": False,
    }

    try:
        st.session_state[SESSION_KEY_QUERY_INTENT] = intent
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to store query nav intent in session_state: %s", exc)
        return

    logger.info(
        "Stored query nav intent: tab_key=%s, payload_keys=%s",
        tab_key,
        list(payload.keys()) if isinstance(payload, Mapping) else None,
    )


def apply_query_params_pre_runtime(app_ctx: "AppContext") -> Dict[str, Any]:
    """
    נקודת הכניסה הראשית לעיבוד Query Params לפני יצירת ה-Runtime:

    Flow:
    -----
    1. parse_dashboard_query_params() – פענוח מלא.
    2. אם אין פרמטרים רלוונטיים → החזרה {} (ולא נעשה כלום).
    3. יישום env/profile (אם קיימים) אל session_state.
    4. שמירת Intent ל-nav_target (אם tab הוגדר ב-URL).

    החזרה:
    ------
    dict parsed – כדי שאם תרצה בעתיד להשתמש במידע הזה ישירות
    (למשל להצגה ב-UI), הוא יהיה זמין.
    """
    parsed = parse_dashboard_query_params()
    if not parsed:
        return {}

    _apply_query_env_profile_overrides(parsed)
    _store_query_nav_intent(parsed)

    return parsed


def apply_query_nav_target_if_needed(runtime: DashboardRuntime) -> None:
    """
    מממש את Intent הניווט שנשמר מ-URL (אם קיים) ל-nav_target בפועל.

    Flow:
    -----
    1. קורא SESSION_KEY_QUERY_INTENT.
    2. אם intent["applied"] == True → לא עושה כלום.
    3. אם tab_key לא קיים ב-runtime.tab_registry → מתעלם (לוג אזהרה).
    4. אם הטאב לא enabled עבור profile נוכחי → מתעלם (לוג אזהרה).
    5. אחרת → קורא set_nav_target(tab_key, payload), מסמן applied=True.

    היתרון:
    --------
    * Separation of concerns:
        - Part 17 מטפל רק ב-query → intent → nav_target.
        - Part 15 (router) ממשיך לעבוד עם nav_target הרגיל (__consume_nav_target).
    """
    try:
        intent_raw = st.session_state.get(SESSION_KEY_QUERY_INTENT)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to access SESSION_KEY_QUERY_INTENT: %s", exc)
        return

    if not isinstance(intent_raw, Mapping):
        return

    if intent_raw.get("applied"):
        return

    tab_key = str(intent_raw.get("tab_key") or "").strip()
    if not tab_key:
        return

    payload = intent_raw.get("payload")
    if payload is not None and not isinstance(payload, Mapping):
        # נשמור רק Mapping; אם זה משהו אחר – נדחוס תחת "value"
        payload = {"value": payload}

    # בדיקה מול TabRegistry
    meta = runtime.tab_registry.get(tab_key)
    if meta is None:
        logger.warning(
            "Query nav intent refers to unknown tab_key='%s'; ignoring.", tab_key
        )
        # נסמן כ-applied כדי למנוע לופים אינסופיים
        intent_raw = dict(intent_raw)
        intent_raw["applied"] = True
        try:
            st.session_state[SESSION_KEY_QUERY_INTENT] = intent_raw
        except Exception:
            pass
        return

    # בדיקה אם הטאב enabled עבור הפרופיל הנוכחי
    if not _is_tab_enabled_for_profile(meta, runtime.profile):
        logger.warning(
            "Query nav intent tab_key='%s' is not enabled for profile='%s'; ignoring.",
            tab_key,
            runtime.profile,
        )
        intent_raw = dict(intent_raw)
        intent_raw["applied"] = True
        try:
            st.session_state[SESSION_KEY_QUERY_INTENT] = intent_raw
        except Exception:
            pass
        return

    # אם עברנו את הבדיקות – ניצור nav_target בפועל
    set_nav_target(tab_key, payload if isinstance(payload, Mapping) else None)

    intent_raw = dict(intent_raw)
    intent_raw["applied"] = True
    try:
        st.session_state[SESSION_KEY_QUERY_INTENT] = intent_raw
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to mark query nav intent as applied: %s", exc)

    logger.info(
        "Query nav intent applied as nav_target: tab_key=%s, payload_keys=%s",
        tab_key,
        list(payload.keys()) if isinstance(payload, Mapping) else None,
    )


# עדכון __all__ עבור חלק 17
try:
    __all__ += [
        "SESSION_KEY_QUERY_INTENT",
        "parse_dashboard_query_params",
        "apply_query_params_pre_runtime",
        "apply_query_nav_target_if_needed",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_QUERY_INTENT",
        "parse_dashboard_query_params",
        "apply_query_params_pre_runtime",
        "apply_query_nav_target_if_needed",
    ]

# =====================
# Part 18/35 – HF-grade Dashboard shell & main entrypoint (orchestration + guards)
# =====================

def _render_critical_alerts_banner(runtime: DashboardRuntime) -> None:
    """
    מציג Banner עליון של "Critical Alerts" ברמת קרן:

    מטרות:
    -------
    - אם env="live" → לוודא שיש Broker מחובר + Risk Engine זמין.
    - להתריע כאשר SqlStore חסר (אין Persist), במיוחד ב-research/backtest.
    - להתריע כאשר Kill-Switch כבוי בסביבת live/paper.
    - לסמן חוסר התאמה בין env/profile לבין יכולות (למשל env=macro בלי Macro Engine).

    לא נועד להחליף טאב Risk/Logs – אלא לתת "מבזק" מיידי של תקלות קריטיות.
    """
    ff = runtime.feature_flags
    caps = runtime.capabilities
    env = runtime.env
    profile = runtime.profile
    svc = runtime.services_status

    broker = svc.get("broker", {}) or {}
    risk = svc.get("risk_engine", {}) or {}
    sql = svc.get("sql_store", {}) or {}
    macro = svc.get("macro_engine", {}) or {}

    alerts: List[str] = []
    errors: List[str] = []

    # 1) LIVE env – דרישות מחמירות
    if env == "live":
        if not broker.get("available") or broker.get("connected") is False:
            errors.append(
                "סביבה **LIVE** ללא Broker מחובר – אין אפשרות לבצע מסחר אמיתי."
            )
        if not caps.get("risk_engine", False):
            errors.append(
                "סביבה **LIVE** ללא Risk Engine פעיל – אין ניהול סיכון אוטומטי."
            )

    # 2) PAPER env – אזהרות רכות יותר
    if env in ("paper", "backtest") and not caps.get("risk_engine", False):
        alerts.append("אין Risk Engine פעיל – ניתוחי Backtest יהיו ללא מגבלות סיכון.")

    # 3) SqlStore – היעדר Persist
    if not caps.get("sql_store", False):
        alerts.append(
            "SqlStore אינו זמין – תוצאות ו-Context לא יישמרו בצורה פרסיסטנטית."
        )

    # 4) Macro mismatch
    if profile == "macro" and not caps.get("macro_engine", False):
        alerts.append("Profile='macro' ללא Macro Engine – טאב Macro יעבוד בצורה מוגבלת.")

    # 5) Kill-switch מצב
    extra_risk = risk.get("extra") or {}
    kill_armed = None
    if isinstance(extra_risk, Mapping):
        kill_armed = extra_risk.get("kill_switch_armed") or extra_risk.get(
            "kill_switch_state"
        )

    if env in ("live", "paper") and isinstance(kill_armed, bool) and not kill_armed:
        alerts.append("Kill-switch **כבוי** בסביבת מסחר – מומלץ לבדוק את הגדרות הסיכון.")

    # אם אין שום דבר – אין צורך ב-Banner
    if not alerts and not errors:
        return

    # קודם כל שגיאות קריטיות (אדום), ואז אזהרות (צהוב)
    if errors:
        st.error(
            "🚨 **Critical environment issues detected:**  \n- "
            + "\n- ".join(errors)
        )

    if alerts:
        st.warning(
            "⚠️ **Environment / risk alerts:**  \n- "
            + "\n- ".join(alerts)
        )


def _render_runtime_debug_panel(runtime: DashboardRuntime) -> None:
    """
    מציג פאנל Debug קומפקטי (רק אם show_debug_info=True):

    - env/profile/run_id
    - capabilities (keys בלבד)
    - מספר טאבים enabled
    - מידע על ביצועי טאבים (אם קיים SESSION_KEY_TAB_TIMINGS)
    """
    ff = runtime.feature_flags
    if not ff.get("show_debug_info"):
        return

    debug_dict = runtime.to_debug_dict()

    # ננסה להוסיף גם מדדי זמן per-tab אם קיימים
    try:
        timings = st.session_state.get(SESSION_KEY_TAB_TIMINGS, {}) or {}
        if isinstance(timings, Mapping):
            debug_dict["tab_timings"] = {
                k: {
                    "last": float(v.get("last", 0.0)),
                    "avg": float(v.get("avg", 0.0)),
                    "count": int(v.get("count", 0)),
                }
                for k, v in timings.items()
                if isinstance(v, Mapping)
            }
    except Exception:
        pass

    with st.expander("🧪 Runtime debug (env/profile/capabilities/perf)", expanded=False):
        st.json(debug_dict)


def run_dashboard_entry(app_ctx: Optional["AppContext"] = None) -> None:
    """
    נקודת כניסה ראשית לדשבורד ה-Web (לשימוש Streamlit / Desktop Bridge):

    Flow:
    -----
    1. AppContext:
        - אם app_ctx לא סופק → get_app_context() (כולל reuse של singleton).
    2. Query params:
        - apply_query_params_pre_runtime(app_ctx) – env/profile + nav intent
          לפי ה-URL (?env=..., ?profile=..., ?tab=..., pair=...).
    3. Runtime:
        - ensure_dashboard_runtime(app_ctx) – יצירה/רענון של DashboardRuntime
          בהתאם ל-env/profile (כולל feature_flags, tab_registry וכו').
    4. Query → nav_target:
        - apply_query_nav_target_if_needed(runtime) – מממש Intent ל-flow
          רק אם הטאב קיים ומופעל בפרופיל הנוכחי.
    5. Shell:
        - render_dashboard_shell(runtime) – Header + Sidebar + Tabs + Alerts.
    6. Error guard:
        - אם משהו משתבש בשלב האתחול – מציג פאנל שגיאה גלובלי במקום להפיל את
          כל Streamlit, ו-logger.error עם stacktrace.

    שימוש:
    -------
    בקובץ Streamlit הראשי:

        if __name__ == "__main__":
            run_dashboard_entry()

    או בתוך Desktop-Bridge שמטעין את אותו Dashboard בתוך WebView.
    """
    try:
        # 1) AppContext
        if app_ctx is None:
            app_ctx = get_app_context()

        # 2) Query params → env/profile overrides + nav intent
        parsed_query = apply_query_params_pre_runtime(app_ctx)

        # 3) Runtime (env/profile/capabilities/tab_registry)
        runtime = ensure_dashboard_runtime(app_ctx)

        # 4) Query → nav_target (אם יש Intent שעדיין לא מומש)
        apply_query_nav_target_if_needed(runtime)

        if parsed_query and runtime.feature_flags.get("show_debug_info"):
            logger.debug("Dashboard entry with parsed_query=%s", parsed_query)

        # 5) Shell מלא – Header + Sidebar + Tabs + Alerts + Debug
        render_dashboard_shell(runtime)

    except Exception as exc:
        # Guard גלובלי – אם משהו קרה לפני/מחוץ לטאבים (למשל AppContext/SqlStore),
        # נרצה להציג UI סביר במקום הלבנה.
        logger.error("Fatal error in run_dashboard_entry: %s", exc, exc_info=True)

        st.markdown("## ❌ Dashboard initialization error")
        st.error(
            "קרתה שגיאה ברמת אתחול הדשבורד (AppContext/Runtime/Config).\n\n"
            "בדוק את קובצי הלוג (dashboard_app.log) לפרטים נוספים."
        )

        # בפרופילי dev/research – נציג גם Stacktrace
        try:
            # אם יש לנו feature_flags בסיסיים ב-session_state – נבדוק אם debug
            ff = st.session_state.get(SESSION_KEY_FEATURE_FLAGS, {}) or {}
            show_debug = bool(ff.get("show_debug_info"))
        except Exception:
            show_debug = True  # נוטה לשקיפות במצב שגיאה

        if show_debug:
            import traceback as _tb

            with st.expander("Stacktrace (debug)", expanded=True):
                st.code(
                    "".join(_tb.format_exception(type(exc), exc, exc.__traceback__)),
                    language="python",
                )


# עדכון __all__ עבור חלק 18
try:
    __all__ += [
        "_render_critical_alerts_banner",
        "_render_runtime_debug_panel",
        "render_dashboard_shell",
        "run_dashboard_entry",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_render_critical_alerts_banner",
        "_render_runtime_debug_panel",
        "render_dashboard_shell",
        "run_dashboard_entry",
    ]

# =====================
# Part 19/35 – Agent / External-Consumer context (HF-grade snapshot for AI agents & Desktop)
# =====================

SESSION_KEY_AGENT_CONTEXT: str = "dashboard_agent_context"


def _collect_session_nav_history_tail(limit: int = 50) -> List[Dict[str, Any]]:
    """
    מחלץ את זנב הניווט (Navigation History) מתוך session_state, עד `limit` רשומות.

    מבנה כל רשומה:
        {
            "ts":  ISO-UTC,
            "from": "<tab_key>",
            "to":   "<tab_key>",
            "payload_keys": ["pair", "preset", ...] or None,
        }

    מיועד לשימוש ע"י:
    - Agents (ללמוד את התנהגות המשתמש).
    - Desktop / Telemetry לניתוח שימושיות הדשבורד.
    """
    try:
        history = st.session_state.get(SESSION_KEY_NAV_HISTORY, [])
    except Exception:
        history = []

    if not isinstance(history, list) or not history:
        return []

    tail = history[-limit:]
    sanitized: List[Dict[str, Any]] = []
    for item in tail:
        if not isinstance(item, Mapping):
            continue
        sanitized.append(
            {
                "ts": str(item.get("ts")),
                "from": str(item.get("from")),
                "to": str(item.get("to")),
                "payload_keys": (
                    [str(k) for k in item.get("payload_keys", [])]
                    if isinstance(item.get("payload_keys"), Sequence)
                    else None
                ),
            }
        )
    return sanitized


def _collect_session_tab_errors() -> Dict[TabKey, Dict[str, Any]]:
    """
    מחלץ מידע על השגיאות האחרונות בכל טאב (אם קיימות) מתוך session_state.

    הפורמט מותאם ל-AI Agents / Desktop:
        {
            "backtest": {
                "ts_utc": "...",
                "exc_type": "...",
                "message": "...",
            },
            ...
        }
    """
    try:
        errors_map = st.session_state.get(SESSION_KEY_LAST_TAB_ERRORS, {})
    except Exception:
        errors_map = {}

    if not isinstance(errors_map, Mapping):
        return {}

    result: Dict[TabKey, Dict[str, Any]] = {}
    for key, info in errors_map.items():
        if not isinstance(info, TabErrorInfo):
            # נתמוך גם במצב שבו שמרו dict דומה
            if isinstance(info, Mapping):
                result[str(key)] = {
                    "ts_utc": str(info.get("ts_utc")),
                    "exc_type": str(info.get("exc_type")),
                    "message": str(info.get("message")),
                }
            continue

        result[str(info.tab_key)] = {
            "ts_utc": info.ts_utc,
            "exc_type": info.exc_type,
            "message": info.message,
        }

    return result


def _collect_session_tab_timings() -> Dict[TabKey, Dict[str, Any]]:
    """
    מחלץ מדדי ביצועים per-tab מתוך SESSION_KEY_TAB_TIMINGS:

        {
            "home": {"last": 0.123, "avg": 0.089, "count": 17},
            ...
        }

    מיועד ל:
    - Agents שמחליטים איפה "כדאי להשקיע" באופטימיזציה.
    - Desktop / Telemetry לצורך Performance Dashboard.
    """
    try:
        timings = st.session_state.get(SESSION_KEY_TAB_TIMINGS, {}) or {}
    except Exception:
        timings = {}

    if not isinstance(timings, Mapping):
        return {}

    out: Dict[TabKey, Dict[str, Any]] = {}
    for key, rec in timings.items():
        if not isinstance(rec, Mapping):
            continue
        try:
            last = float(rec.get("last", 0.0))
        except Exception:
            last = 0.0
        try:
            avg = float(rec.get("avg", 0.0))
        except Exception:
            avg = 0.0
        try:
            count = int(rec.get("count", 0))
        except Exception:
            count = 0

        out[str(key)] = {"last": last, "avg": avg, "count": count}

    return out


def build_agent_context_payload(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    בונה payload עשיר ו"Agent-ready" שמרכז את כל מה שסוכן AI/מערכת חיצונית צריכה לדעת
    על מצב הדשבורד והמערכת, כדי להציע שדרוגים/ניתוחים/פעולות:

    מבנה כללי:
    -----------
    {
        "meta": {
            "env": ...,
            "profile": ...,
            "run_id": ...,
            "app_name": ...,
            "version": ...,
            "host": ...,
            "user": ...,
        },
        "feature_flags": { ... subset ... },
        "capabilities": { ... },
        "domains": { ... },
        "services_status": { ... },
        "base_context": { ... },     # context לוגי (serialized)
        "nav_history_tail": [...],   # זנב ניווט (עד 50 רשומות)
        "tab_timings": { ... },      # ביצועים per-tab
        "tab_errors": { ... },       # שגיאות אחרונות per-tab
    }

    שימושים:
    --------
    - Agents Tab (🤖) שמנתח את ה-Context ומציע פעולות / refactors.
    - Desktop Bridge / "Supervisor" Agent שמנטר את ביצועי הדשבורד.
    - לוגים מתקדמים (למשל שמירה ל-SqlStore דרך Agents).
    """
    ff = runtime.feature_flags
    env = runtime.env
    profile = runtime.profile
    run_id = runtime.run_id

    app_name = ff.get("app_name", APP_NAME)
    version = ff.get("version", APP_VERSION)
    host = ff.get("host", RUNTIME_HOST)
    user = ff.get("user", RUNTIME_USER)

    base_ctx = st.session_state.get(SESSION_KEY_BASE_CTX)
    base_ctx_serialized = _make_json_safe(
        _serialize_base_ctx_for_snapshot(base_ctx)
    )

    # Nav history / errors / timings מתוך session_state
    nav_history_tail = _collect_session_nav_history_tail(limit=50)
    tab_errors = _collect_session_tab_errors()
    tab_timings = _collect_session_tab_timings()

    # נבחר subset "מעניין" של FeatureFlags – סוכנים לא חייבים הכל
    feature_flags_view = {
        "env": env,
        "profile": profile,
        "enable_live_trading_actions": ff.get("enable_live_trading_actions"),
        "use_sql_backed_state": ff.get("use_sql_backed_state"),
        "enable_experiment_mode": ff.get("enable_experiment_mode"),
        "desktop_integration": ff.get("desktop_integration"),
    }

    payload: Dict[str, Any] = {
        "meta": {
            "env": env,
            "profile": profile,
            "run_id": run_id,
            "app_name": app_name,
            "version": version,
            "host": host,
            "user": user,
        },
        "feature_flags": feature_flags_view,
        "capabilities": dict(runtime.capabilities),
        "domains": dict(runtime.domains),
        "services_status": runtime.services_status,
        "base_context": base_ctx_serialized,
        "nav_history_tail": nav_history_tail,
        "tab_timings": tab_timings,
        "tab_errors": tab_errors,
    }

    return payload


def update_agent_context_in_session(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    בונה ומעדכן את Agent Context ב-session_state, לשימוש ע"י:

    - טאב Agents (🤖) שיכול לקרוא st.session_state["dashboard_agent_context"]
      ולהזין אותו ישירות לסוכני ה-AI.
    - Desktop / תהליכי רקע שיקראו את המידע כדי להפעיל המלצות/שדרוגים.

    הפונקציה מחזירה את ה-payload שנשמר בפועל, לטובת שימוש מיידי.
    """
    payload = build_agent_context_payload(runtime)
    safe_payload = _make_json_safe(payload)

    try:
        st.session_state[SESSION_KEY_AGENT_CONTEXT] = safe_payload
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to store dashboard_agent_context in session_state: %s", exc
        )

    logger.debug(
        "Agent context updated in session_state (env=%s, profile=%s, keys=%s)",
        runtime.env,
        runtime.profile,
        list(safe_payload.keys()),
    )

    return safe_payload


def export_dashboard_state_for_agents(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    מעטפת ידידותית לייצוא מצב הדשבורד לסוכנים/מערכות חיצוניות:

    הבדל מול update_agent_context_in_session:
    -----------------------------------------
    - פונקציה זו אינה נוגעת ב-session_state.
    - מיועדת לשימוש ב:
        * Desktop Bridge שמקבל DashboardRuntime ומעביר את ה-payload
          לתהליכי AI או לשרת חיצוני.
        * בדיקות יחידה / סקריפטים חיצוניים.

    מחזירה:
        dict Agent-ready (כמו build_agent_context_payload) אך ללא כתיבה ל-session_state.
    """
    payload = build_agent_context_payload(runtime)
    return _make_json_safe(payload)


# עדכון __all__ עבור חלק 19
try:
    __all__ += [
        "SESSION_KEY_AGENT_CONTEXT",
        "build_agent_context_payload",
        "update_agent_context_in_session",
        "export_dashboard_state_for_agents",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_AGENT_CONTEXT",
        "build_agent_context_payload",
        "update_agent_context_in_session",
        "export_dashboard_state_for_agents",
    ]
# =====================
# Part 20/35 – DashboardSummary & Telemetry Models (HF-grade analytics core)
# =====================

SESSION_KEY_DASHBOARD_SUMMARY: str = "dashboard_summary"


@dataclass
class TabUsageStats:
    """
    סטטיסטיקות שימוש/ביצועים לטאב אחד.

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
        כמה פעמים נמדדו זמני ריצה לטאב (count מתוך SESSION_KEY_TAB_TIMINGS).
    last_render_time:
        זמן הריצה האחרון שנמדד (שניות).
    avg_render_time:
        זמן ריצה ממוצע (running average) מאז התחלת הסשן.
    """

    key: TabKey
    label: TabLabel
    group: str
    enabled: bool
    render_count: int = 0
    last_render_time: float = 0.0
    avg_render_time: float = 0.0


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
    """

    name: str
    available: bool
    severity: Literal["ok", "warning", "error"]
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardSummary:
    """
    סיכום מרוכז של מצב הדשבורד ברמת קרן גידור.

    נועד ל:
    --------
    - Desktop Bridge (Qt) להצגת Health ו-Tabs.
    - Agents / Supervisors שמחליטים איפה להתמקד (ביצועים, תקלות).
    - לוגים/Telemetry (שמירה ל-SqlStore או ל-service חיצוני).

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
    - Agents / FairValue / MarketData → warnings רכים יותר.
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
    ) -> None:
        available = bool(status.get("available"))
        severity = default_severity
        summary = default_summary
        details: Dict[str, Any] = dict(status)

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
            )
        )

    # SqlStore
    sql = svc.get("sql_store", {}) or {}
    sql_available = bool(sql.get("available"))
    sql_summary = "SqlStore available." if sql_available else "SqlStore not available."
    add_snapshot("sql_store", sql, sql_summary, "ok")

    # Broker
    broker = svc.get("broker", {}) or {}
    br_available = bool(broker.get("available"))
    br_connected = broker.get("connected")
    br_mode = str(broker.get("mode") or "").lower() or "unknown"

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
        )
    )

    # Market Data
    market = svc.get("market_data", {}) or {}
    md_available = bool(market.get("available"))
    md_source = market.get("source") or "N/A"
    md_latency = market.get("latency_mode") or "N/A"
    md_summary = f"Source={md_source}, latency={md_latency}."
    add_snapshot("market_data", market, md_summary, "ok")

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
        )
    )

    # Fair Value Engine
    fv = svc.get("fair_value", {}) or {}
    fv_available = bool(fv.get("available"))
    fv_summary = "Fair Value engine available." if fv_available else "Fair Value engine not available."
    add_snapshot("fair_value", fv, fv_summary, "ok")

    # Backtester / Optimizer / Meta-Optimizer – Health לוגי בלבד
    bt = svc.get("backtester", {}) or {}
    opt = svc.get("optimizer", {}) or {}
    mo = svc.get("meta_optimizer", {}) or {}

    add_snapshot(
        "backtester",
        bt,
        "Backtester module available." if bt.get("available") else "Backtester not available.",
        "ok",
    )
    add_snapshot(
        "optimizer",
        opt,
        "Optimizer module available." if opt.get("available") else "Optimizer not available.",
        "ok",
    )
    add_snapshot(
        "meta_optimizer",
        mo,
        "Meta-Optimizer module available." if mo.get("available") else "Meta-Optimizer not available.",
        "ok",
    )

    return snapshots


def _build_tab_usage_stats(runtime: DashboardRuntime) -> List[TabUsageStats]:
    """
    בונה רשימת TabUsageStats עבור כל הטאבים הפעילים בפרופיל הנוכחי:

    משתמש ב:
    - runtime.tab_registry  (מטא-דאטה).
    - runtime.feature_flags (profile/env).
    - SESSION_KEY_TAB_TIMINGS (מדדי ביצועים שנאספו בפועל).
    """
    registry = runtime.tab_registry
    ff = runtime.feature_flags

    active_tabs, active_keys, _ = build_active_tabs(registry, ff)
    timings = _collect_session_tab_timings()

    stats: List[TabUsageStats] = []

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

        stats.append(
            TabUsageStats(
                key=key,
                label=meta.label,
                group=meta.group,
                enabled=True,  # לפי build_active_tabs – כל meta כאן כבר Enabled לפרופיל
                render_count=count,
                last_render_time=last,
                avg_render_time=avg,
            )
        )

    return stats


def build_dashboard_summary(runtime: DashboardRuntime) -> DashboardSummary:
    """
    בונה DashboardSummary מלא מתוך DashboardRuntime:

    Flow:
    -----
    1. אוסף App-level meta (env/profile/run_id/app_name/version/host/user).
    2. בונה ServiceHealthSnapshot לכל שירות מרכזי.
    3. בונה TabUsageStats לכל טאב פעיל.
    4. קובע דגלים גלובליים:
        * has_critical_issues – האם יש Service severity="error".
        * has_warnings       – האם יש Service severity="warning".
    """
    ff = runtime.feature_flags
    env = runtime.env
    profile = runtime.profile
    run_id = runtime.run_id

    app_name = ff.get("app_name", APP_NAME)
    version = ff.get("version", APP_VERSION)
    host = ff.get("host", RUNTIME_HOST)
    user = ff.get("user", RUNTIME_USER)

    # שירותים
    services = _build_service_health_from_status(runtime)
    has_critical = any(s.severity == "error" for s in services)
    has_warn = any(s.severity == "warning" for s in services)

    # טאבים
    tabs_stats = _build_tab_usage_stats(runtime)

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
    )

    return summary


def dashboard_summary_to_dict(summary: DashboardSummary) -> Dict[str, Any]:
    """
    ממיר DashboardSummary ל-dict JSON-friendly.

    זאת פונקציה נפרדת (ולא רק dataclasses.asdict) כדי:
    - לשלוט במבנה (למשל הרשאות/שדות).
    - לשמור על תאימות קדימה/אחורה אם נוסיף שדות בעתיד.
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
            }
            for s in summary.services
        ],
        "has_critical_issues": summary.has_critical_issues,
        "has_warnings": summary.has_warnings,
    }


def update_dashboard_summary_in_session(
    runtime: DashboardRuntime,
    store_as_dict: bool = True,
) -> DashboardSummary:
    """
    בונה DashboardSummary ומעדכן אותו ב-session_state:

    - אם store_as_dict=True → שומר dict JSON-friendly תחת SESSION_KEY_DASHBOARD_SUMMARY.
    - אחרת → שומר את אובייקט DashboardSummary עצמו.

    הפונקציה מחזירה תמיד את ה-DashboardSummary עבור שימוש מיידי.
    """
    summary = build_dashboard_summary(runtime)
    if store_as_dict:
        obj: Any = dashboard_summary_to_dict(summary)
        obj = _make_json_safe(obj)
    else:
        obj = summary  # dataclass – יכול להיות לא JSON-safe אבל שימושי ב-Python צד שני

    try:
        st.session_state[SESSION_KEY_DASHBOARD_SUMMARY] = obj
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to store dashboard summary in session_state: %s", exc
        )

    logger.debug(
        "Dashboard summary updated in session_state (env=%s, profile=%s)",
        summary.env,
        summary.profile,
    )

    return summary


def export_dashboard_summary(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    פונקציית Export ידידותית ל-Desktop/Agents:

    - לא נוגעת ב-session_state.
    - מחזירה dict JSON-friendly (כולל active_tabs + services + flags גלובליים).
    - יכולה להישלח כפי שהיא לסוכן AI, ל-log server, או ל-Desktop Bridge.
    """
    summary = build_dashboard_summary(runtime)
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
# Part 21/35 – User preferences, personalization & shell wrapping (HF-grade)
# =====================

from dataclasses import asdict as _dc_asdict

SESSION_KEY_USER_PREFS: str = "dashboard_user_prefs"

@dataclass
class UserDashboardPrefs:
    """
    העדפות משתמש לדשבורד – שכבת Personalization ברמת קרן.

    הרעיון:
    --------
    לאפשר לדשבורד "להתאים את עצמו" למשתמש:
    - איזה פרופיל להעדיף (trading/research/risk/macro/monitoring).
    - איזה טאב לפתוח כברירת מחדל.
    - האם להציג Debug panels כברירת מחדל.
    - כמה היסטוריית ניווט לשמור.
    - איזה benchmark להעדיף.

    שדות:
    -----
    user_key:
        מזהה ייחודי למשתמש (user@host או מזהה מה-Config).
    preferred_profile:
        פרופיל מועדף כשאין override חיצוני (query/env/config).
    default_tab_key:
        הטאב שייפתח כברירת מחדל אם אין nav_target מפורש.
    show_debug_by_default:
        האם להפעיל show_debug_info כברירת מחדל בפרופילים dev/research.
    max_nav_history:
        כמה רשומות לשמור ב-SESSION_KEY_NAV_HISTORY (מונע התנפחות).
    preferred_benchmark:
        Benchmark מועדף (למשל "SPY", "QQQ", "EWJ") – אם קיים ב-base_context.
    dense_layout:
        האם להעדיף Layout צפוף יותר (פחות מרווחים/כותרות).
    """

    user_key: str
    preferred_profile: Optional[ProfileName] = None
    default_tab_key: Optional[TabKey] = None
    show_debug_by_default: bool = True
    max_nav_history: int = 200
    preferred_benchmark: Optional[str] = None
    dense_layout: bool = False


def _derive_user_key(runtime: DashboardRuntime) -> str:
    """
    גוזר user_key מתוך Runtime:

    לוגיקה:
    -------
    1. אם ב-AppContext.settings יש user_id → נשתמש בו.
    2. אחרת → ניקח feature_flags["user"] + ["host"] ונרכיב user@host.
    3. fallback → "anonymous".
    """
    # 1) AppContext.settings.user_id אם קיים
    settings = _safe_getattr(runtime.app_ctx, "settings")
    if settings is not None:
        uid = _safe_getattr(settings, "user_id")
        if isinstance(uid, str) and uid.strip():
            return uid.strip()

        cfg = _safe_getattr(settings, "config")
        if isinstance(cfg, Mapping):
            v = cfg.get("user_id") or cfg.get("USER_ID")
            if isinstance(v, str) and v.strip():
                return v.strip()

    # 2) מתוך feature_flags
    ff = runtime.feature_flags
    user = str(ff.get("user", RUNTIME_USER) or "").strip() or "anonymous"
    host = str(ff.get("host", RUNTIME_HOST) or "").strip() or "unknown-host"
    return f"{user}@{host}"


def _user_prefs_to_dict(prefs: UserDashboardPrefs) -> Dict[str, Any]:
    """
    ממיר UserDashboardPrefs ל-dict JSON-friendly.
    """
    return {
        "user_key": prefs.user_key,
        "preferred_profile": prefs.preferred_profile,
        "default_tab_key": prefs.default_tab_key,
        "show_debug_by_default": prefs.show_debug_by_default,
        "max_nav_history": prefs.max_nav_history,
        "preferred_benchmark": prefs.preferred_benchmark,
        "dense_layout": prefs.dense_layout,
    }


def _user_prefs_from_mapping(data: Mapping[str, Any], fallback_key: str) -> UserDashboardPrefs:
    """
    בונה UserDashboardPrefs מתוך Mapping (למשל שהגיע מ-SqlStore/JSON).
    במקרה של שדות חסרים – משתמש בערכי ברירת מחדל.
    """
    return UserDashboardPrefs(
        user_key=str(data.get("user_key") or fallback_key),
        preferred_profile=(
            _normalize_profile(str(data["preferred_profile"]))
            if data.get("preferred_profile") is not None
            else None
        ),
        default_tab_key=str(data.get("default_tab_key") or "") or None,
        show_debug_by_default=bool(
            data.get("show_debug_by_default", True)
        ),
        max_nav_history=int(data.get("max_nav_history", 200)),
        preferred_benchmark=(
            str(data.get("preferred_benchmark")).strip().upper()
            if data.get("preferred_benchmark")
            else None
        ),
        dense_layout=bool(data.get("dense_layout", False)),
    )


def _load_user_prefs_from_sql_store(
    runtime: DashboardRuntime,
    user_key: str,
) -> Optional[UserDashboardPrefs]:
    """
    מנסה לטעון העדפות משתמש מ-SqlStore (אם קיים):

    אסטרטגיה:
    ---------
    1. מגלה SqlStore מתוך AppContext (sql_store / store / db / sql).
    2. מנסה אחת מכמה פונקציות:
        - load_dashboard_prefs(user_key)
        - load_user_prefs(user_key)
        - load_json("dashboard_prefs", user_key)
    3. אם מתקבל Mapping – בונה ממנו UserDashboardPrefs.
    4. במקרה של כשל – מחזיר None בלי להפיל את הדשבורד.
    """
    caps = runtime.capabilities
    if not caps.get("sql_store", False):
        return None

    services_map = _discover_services_mapping(runtime.app_ctx)
    has_store, store_obj = _probe_service(
        runtime.app_ctx,
        services_map,
        candidates=("sql_store", "store", "db", "sql"),
    )
    if not (has_store and store_obj is not None):
        return None

    # סדר פונקציות אפשרי
    method_candidates: Sequence[str] = (
        "load_dashboard_prefs",
        "load_user_prefs",
        "load_json",
    )

    for name in method_candidates:
        func = getattr(store_obj, name, None)
        if not callable(func):
            continue

        try:
            if name == "load_json":
                raw = func("dashboard_prefs", user_key)  # type: ignore[misc]
            else:
                raw = func(user_key)  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover
            logger.debug("SqlStore.%s(%r) raised %s", name, user_key, exc)
            continue

        if isinstance(raw, Mapping):
            try:
                prefs = _user_prefs_from_mapping(raw, user_key)
                logger.info(
                    "Loaded UserDashboardPrefs for user_key=%s via SqlStore.%s",
                    user_key,
                    name,
                )
                return prefs
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to parse user prefs for user_key=%s from SqlStore.%s: %s",
                    user_key,
                    name,
                    exc,
                )
                return None

    return None


def _save_user_prefs_to_sql_store(
    runtime: DashboardRuntime,
    prefs: UserDashboardPrefs,
) -> Tuple[bool, Optional[str]]:
    """
    מנסה לשמור העדפות משתמש ב-SqlStore (אם קיים).

    אסטרטגיה:
    ---------
    1. מגלה SqlStore.
    2. מנסה פונקציות:
        - save_dashboard_prefs(user_key, data)
        - save_user_prefs(user_key, data)
        - save_json("dashboard_prefs", user_key, data)
    3. במקרה של הצלחה – מחזיר (True, method_name).
    4. במקרה של כשל – מחזיר (False, None).

    לא מפיל את הדשבורד אם אין SqlStore/שיטות מתאימות.
    """
    caps = runtime.capabilities
    if not caps.get("sql_store", False):
        return False, None

    services_map = _discover_services_mapping(runtime.app_ctx)
    has_store, store_obj = _probe_service(
        runtime.app_ctx,
        services_map,
        candidates=("sql_store", "store", "db", "sql"),
    )
    if not (has_store and store_obj is not None):
        return False, None

    data = _user_prefs_to_dict(prefs)
    data = _make_json_safe(data)

    method_candidates: Sequence[str] = (
        "save_dashboard_prefs",
        "save_user_prefs",
        "save_json",
    )

    for name in method_candidates:
        func = getattr(store_obj, name, None)
        if not callable(func):
            continue

        try:
            if name == "save_json":
                func("dashboard_prefs", prefs.user_key, data)  # type: ignore[misc]
            else:
                func(prefs.user_key, data)  # type: ignore[misc]
            logger.info(
                "UserDashboardPrefs saved for user_key=%s via SqlStore.%s",
                prefs.user_key,
                name,
            )
            return True, name
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "SqlStore.%s failed to save user prefs for user_key=%s: %s",
                name,
                prefs.user_key,
                exc,
            )

    return False, None


def get_or_init_user_prefs(runtime: DashboardRuntime) -> UserDashboardPrefs:
    """
    מחזיר (או מאתחל) UserDashboardPrefs עבור ה-Runtime הנוכחי:

    Flow:
    -----
    1. אם יש כבר אובייקט ב-session_state → מחזיר אותו.
    2. אחרת:
        - גוזר user_key.
        - מנסה לטעון מ-SqlStore (אם זמין).
        - אם לא נמצא – בונה Prefs חדשים על בסיס Runtime (env/profile/base_ctx).
    3. שומר את ה-Prefs ב-session_state לצורך ריצות עתידיות.
    """
    try:
        existing = st.session_state.get(SESSION_KEY_USER_PREFS)
    except Exception:
        existing = None

    if isinstance(existing, UserDashboardPrefs):
        return existing
    if isinstance(existing, Mapping):
        # תמיכה במקרה שבו שמרנו dict בלבד בעבר
        user_key = _derive_user_key(runtime)
        prefs = _user_prefs_from_mapping(existing, user_key)
        st.session_state[SESSION_KEY_USER_PREFS] = prefs
        return prefs

    user_key = _derive_user_key(runtime)

    # ניסיון טעינה מ-SqlStore
    prefs = _load_user_prefs_from_sql_store(runtime, user_key)
    if prefs is None:
        # יצירת Prefs חדשים
        base_ctx = st.session_state.get(SESSION_KEY_BASE_CTX)
        preferred_benchmark = None
        if base_ctx is not None:
            preferred_benchmark = _extract_benchmark_from_base_ctx(base_ctx)

        prefs = UserDashboardPrefs(
            user_key=user_key,
            preferred_profile=runtime.profile,
            default_tab_key=TAB_KEY_HOME,
            show_debug_by_default=True if runtime.env in ("dev", "research") else False,
            max_nav_history=200,
            preferred_benchmark=preferred_benchmark,
            dense_layout=False,
        )
        logger.info("Initialized new UserDashboardPrefs for user_key=%s", user_key)

    try:
        st.session_state[SESSION_KEY_USER_PREFS] = prefs
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to store UserDashboardPrefs in session_state: %s", exc)

    return prefs


def apply_user_prefs_to_runtime(
    runtime: DashboardRuntime,
    prefs: UserDashboardPrefs,
) -> None:
    """
    מיישמת חלק מההעדפות על ה-Runtime/FeatureFlags:

    מה אנחנו כן משנים:
    -------------------
    - show_debug_info:
        * אם prefs.show_debug_by_default=True בפרופילי dev/research → מחייבים True.
    - benchmark:
        * אם prefs.preferred_benchmark קיים – ננסה לעדכן ב-base_context.
    - nav_history limit:
        * מגביל את SESSION_KEY_NAV_HISTORY לאורך prefs.max_nav_history.

    מה אנחנו *לא* משנים (חשוב!):
    -----------------------------
    - env/profile – אלו נשלטים ע"י config/envvars/query; Prefs יכולים לשמש רק כ-Default,
      אבל לא דורסים override אקטיבי.
    - סדר טאבים / יכולות – נשלט ע"י feature_flags + TabRegistry.
    """
    ff = runtime.feature_flags

    # show_debug_info – רק אם מדובר ב-env מסוג dev/research/test
    if prefs.show_debug_by_default and runtime.env in ("dev", "research", "test"):
        ff["show_debug_info"] = True

    # preferred_benchmark → base_context
    if prefs.preferred_benchmark:
        try:
            base_ctx = st.session_state.get(SESSION_KEY_BASE_CTX)
        except Exception:
            base_ctx = None

        if base_ctx is not None:
            try:
                # attrib
                if hasattr(base_ctx, "benchmark"):
                    setattr(base_ctx, "benchmark", prefs.preferred_benchmark)
                # mapping
                if isinstance(base_ctx, MutableMapping):
                    base_ctx["benchmark"] = prefs.preferred_benchmark
                st.session_state[SESSION_KEY_BASE_CTX] = base_ctx
            except Exception:  # pragma: no cover
                logger.debug(
                    "Failed to apply preferred_benchmark=%s on base_ctx",
                    prefs.preferred_benchmark,
                )

    # קיצוץ Nav history לאורך מקסימלי
    try:
        history = st.session_state.get(SESSION_KEY_NAV_HISTORY, [])
        if isinstance(history, list) and len(history) > prefs.max_nav_history:
            st.session_state[SESSION_KEY_NAV_HISTORY] = history[-prefs.max_nav_history :]
    except Exception:  # pragma: no cover
        pass


def persist_user_prefs_if_needed(
    runtime: DashboardRuntime,
    prefs: UserDashboardPrefs,
    auto_persist: bool = True,
) -> None:
    """
    שומר Prefs ב-SqlStore במקרה שיש שינוי משמעותי.

    כרגע הפונקציה היא Hook עתידי – ניתן להרחיב ולהשוות Prefs
    קודמים לחדשים ולשמור רק כשצריך. לעת עתה, אם auto_persist=True
    וקיים SqlStore – ננסה לשמור כל כמה ריצות.

    אפשרויות הרחבה:
    ----------------
    - להוסיף "dirty flag" ב-session_state.
    - לספור כמה פעמים מריצים Shell ולשמור אחת ל-N פעמים.
    """
    if not auto_persist:
        return

    saved, method_name = _save_user_prefs_to_sql_store(runtime, prefs)
    if saved:
        logger.debug(
            "UserDashboardPrefs persisted for user_key=%s via %s",
            prefs.user_key,
            method_name or "unknown",
        )


# -------------------------
# Shell wrapping & wiring
# -------------------------

# שומרים עותק של ה-implementation המקורי של render_dashboard_shell
# כדי שנוכל לעטוף אותו ביכולות חדשות (Prefs + Agent context + Summary)
try:
    _render_dashboard_shell_core = render_dashboard_shell  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    def _render_dashboard_shell_core(runtime: DashboardRuntime) -> None:
        raise RuntimeError(
            "render_dashboard_shell must be defined before Part 21 "
            "(shell wrapper). Please check the ordering of parts in dashboard.py."
        )


# עדכון __all__ עבור חלק 21
try:
    __all__ += [
        "SESSION_KEY_USER_PREFS",
        "UserDashboardPrefs",
        "get_or_init_user_prefs",
        "apply_user_prefs_to_runtime",
        "persist_user_prefs_if_needed",
        "_render_dashboard_shell_core",
        "render_dashboard_shell",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_USER_PREFS",
        "UserDashboardPrefs",
        "get_or_init_user_prefs",
        "apply_user_prefs_to_runtime",
        "persist_user_prefs_if_needed",
        "_render_dashboard_shell_core",
        "render_dashboard_shell",
    ]
# =====================
# Part 22/35 – Dashboard toolbar (HF-grade controls & personalization UI)
# =====================

def _render_dashboard_toolbar(
    runtime: DashboardRuntime,
    prefs: UserDashboardPrefs,
) -> None:
    """
    🎛 Dashboard Toolbar – שכבת שליטה אישית ברמת קרן, בראש הדשבורד.

    מטרות:
    -------
    - להציג למשתמש:
        * env / profile / run_id
        * מצב Live-trading actions / Experiment mode
    - לאפשר:
        * Toggle ל-"Dense layout" (layout מתקדם וצפוף).
        * Toggle ל-Debug mode (פאנלים ו-Telemetry נוספים) בסביבות dev/research/test.

    העקרונות:
    ----------
    - אין שינוי env/profile מה-UI (נשלטים ע"י config/envvars/query).
    - כל שינוי נעשה:
        * הן ב-FeatureFlags (לשאר המערכת).
        * הן ב-UserDashboardPrefs (Persist ב-SqlStore).
    """
    ff = runtime.feature_flags
    env = runtime.env
    profile = runtime.profile
    run_id = runtime.run_id

    live_actions = bool(ff.get("enable_live_trading_actions", False))
    experiment_mode = bool(ff.get("enable_experiment_mode", False))

    dense_layout = bool(prefs.dense_layout)
    debug_allowed = env in ("dev", "research", "test")
    current_debug = bool(ff.get("show_debug_info", False))

    toolbar = st.container()
    with toolbar:
        st.markdown("#### 🎛 Dashboard controls")

        col_main, col_layout, col_debug = st.columns([2.2, 1.3, 1.3])

        # ---------- צד שמאל: Context & modes ----------
        with col_main:
            st.markdown(
                f"- **Env:** `{env}`  •  **Profile:** `{profile}`  •  "
                f"**Run ID:** `{run_id}`"
            )
            st.caption(
                f"Live actions: "
                f"{'✅ Enabled' if live_actions else '⭕ Disabled'}  •  "
                f"Experiment mode: "
                f"{'🧪 On' if experiment_mode else 'Off'}"
            )

            if prefs.preferred_benchmark:
                st.caption(
                    f"Preferred benchmark: `{prefs.preferred_benchmark}` "
                    f"(משפיע על ניתוחים/דוחות ברמת ברירת מחדל)."
                )

        # ---------- אמצע: Layout / UX ----------
        with col_layout:
            st.markdown("**Layout & UX**")
            dense_new = st.checkbox(
                "Dense layout",
                key="toolbar_dense_layout",
                help=(
                    "מצמצם מרווחים ומסתיר חלק מהכותרות – "
                    "מותאם למשתמשים מתקדמים שרוצים לראות יותר דאטה על המסך."
                ),
                value=dense_layout,
            )

            # עדכון Prefs אם יש שינוי
            if dense_new != dense_layout:
                prefs.dense_layout = dense_new
                try:
                    st.session_state[SESSION_KEY_USER_PREFS] = prefs
                except Exception:  # pragma: no cover
                    pass
                logger.info(
                    "UserDashboardPrefs: dense_layout changed to %s for user_key=%s",
                    dense_new,
                    prefs.user_key,
                )

            st.caption(
                f"Nav history limit: {prefs.max_nav_history} events "
                "(ניתן לשנות בקונפיג/Prefs בעתיד)."
            )

        # ---------- ימין: Debug mode ----------
        with col_debug:
            st.markdown("**Debug & Telemetry**")

            if debug_allowed:
                debug_new = st.checkbox(
                    "Debug mode",
                    key="toolbar_debug_mode",
                    help=(
                        "מציג פאנלי Debug, stacktraces, Telemetry וזמני ריצה. "
                        "מומלץ רק למצב פיתוח/מחקר."
                    ),
                    value=current_debug,
                )

                if debug_new != current_debug:
                    # עדכון FeatureFlags
                    ff["show_debug_info"] = debug_new
                    # עדכון Prefs
                    prefs.show_debug_by_default = debug_new
                    try:
                        st.session_state[SESSION_KEY_USER_PREFS] = prefs
                    except Exception:  # pragma: no cover
                        pass

                    logger.info(
                        "Debug mode toggled to %s (env=%s, profile=%s, user_key=%s)",
                        debug_new,
                        env,
                        profile,
                        prefs.user_key,
                    )

                st.caption(
                    "זמין בסביבות dev/research/test בלבד. "
                    "ב-LIVE לא ניתן להפעיל debug מה-UI."
                )
            else:
                st.caption(
                    "Debug mode נעול בסביבה זו.  "
                    "הפעל דרך config/envvars בלבד."
                )


# -------------------------
# Shell wrapper – גרסה מעודכנת עם Toolbar
# -------------------------


# עדכון __all__ עבור חלק 22
try:
    __all__ += [
        "_render_dashboard_toolbar",
        "render_dashboard_shell",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_render_dashboard_toolbar",
        "render_dashboard_shell",
    ]
# =====================
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

    st.markdown("### 📜 Logs / System Health – Dashboard-level view")

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
            width = "stretch")
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
            st.dataframe(df_timings, width = "stretch")

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
            st.dataframe(df_errors, width = "stretch")

    # Nav history
    with col_right:
        st.markdown("**Navigation history (tail)**")
        nav_tail = _collect_session_nav_history_tail(limit=50)
        if not nav_tail:
            st.caption("אין עדיין היסטוריית ניווט.")
        else:
            df_nav = pd.DataFrame(nav_tail)
            st.dataframe(df_nav, width = "stretch")

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
        - לפחות יכולת מרכזית אחת זמינה (למשל SqlStore/Broker/Backtester).
    has_critical_issues:
        True אם יש לפחות Service אחד עם severity="error".
    has_warnings:
        True אם יש לפחות Service אחד עם severity="warning".
    issues:
        רשימת תיאורים אנושיים של בעיות קריטיות (לוגיקה מורחבת מעבר ל-ServiceHealthSnapshot).
    warnings:
        רשימת תיאורים אנושיים של אזהרות (למשל חסר Risk Engine ב-backtest).
    ts_utc:
        זמן יצירת ה-Health ב-UTC (isoformat).
    summary:
        DashboardSummary (אופציונלי) – טיפה "כבד" יותר, אבל נותן תמונה מלאה.
    """

    env: EnvName
    profile: ProfileName
    ready: bool
    has_critical_issues: bool
    has_warnings: bool
    issues: List[str]
    warnings: List[str]
    ts_utc: str
    summary: Optional[DashboardSummary] = None


def _compute_health_issues_from_summary(
    runtime: DashboardRuntime,
    summary: DashboardSummary,
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

    # נשלב גם summaries של services עם severity=error/warning
    for svc in summary.services:
        if svc.severity == "error":
            # נימנע מכפילות – אם כבר ניסחנו את אותה בעיה, לא נוסיף שוב
            msg = f"[{svc.name}] {svc.summary}"
            if msg not in issues:
                issues.append(msg)
        elif svc.severity == "warning":
            msg = f"[{svc.name}] {svc.summary}"
            if msg not in warnings:
                warnings.append(msg)

    return issues, warnings


def compute_dashboard_health(runtime: DashboardRuntime) -> DashboardHealth:
    """
    בונה DashboardHealth מלא מתוך DashboardRuntime:

    Ready criteria (ברירת מחדל):
    ----------------------------
    ready = (
        not summary.has_critical_issues
        and at_least_one_core_capability
    )

    where at_least_one_core_capability מוגדר כ:
        - SqlStore
        או Broker
        או Backtester
        או Optimizer
        או Macro Engine
        (כלומר – יש לפחות "עמוד שדרה" אחד שהמערכת יכולה לעבוד עליו).

    הפונקציה משמשת:
    ----------------
    - Health-check endpoints (אם תעטוף אותה ב-API).
    - Desktop Bridge (decision האם להציג אזהרות/לחסום פעולות).
    - Supervisor Agents שמחליטים אם להפעיל שדרוגים/Backtests עכשיו.
    """
    summary = build_dashboard_summary(runtime)
    issues, warnings = _compute_health_issues_from_summary(runtime, summary)

    caps = runtime.capabilities
    core_caps = [
        caps.get("sql_store", False),
        caps.get("broker", False),
        caps.get("backtester", False),
        caps.get("optimizer", False),
        caps.get("macro_engine", False),
    ]
    at_least_one_core = any(bool(x) for x in core_caps)

    ready = (not summary.has_critical_issues) and at_least_one_core

    health = DashboardHealth(
        env=runtime.env,
        profile=runtime.profile,
        ready=ready,
        has_critical_issues=summary.has_critical_issues,
        has_warnings=summary.has_warnings,
        issues=issues,
        warnings=warnings,
        ts_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        summary=summary,
    )

    return health


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
        "Dashboard health updated in session_state (env=%s, profile=%s, ready=%s)",
        health.env,
        health.profile,
        health.ready,
    )

    return health


def export_dashboard_health(runtime: DashboardRuntime, include_summary: bool = False) -> Dict[str, Any]:
    """
    Export ידידותי ל-Desktop/Agents:

    - לא נוגע ב-session_state.
    - מחזיר dict JSON-friendly עם ready/issues/warnings ועוד.
    - אופציונלית כולל Summary מלא (heavy יותר, אבל נותן תמונה מלאה).
    """
    health = compute_dashboard_health(runtime)
    payload = dashboard_health_to_dict(health, include_summary=include_summary)
    return _make_json_safe(payload)


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
        info  – dict קטן עם env/profile/issues/warnings (לא כולל Summary מלא).
    """
    if app_ctx is None:
        app_ctx = get_app_context()

    runtime = ensure_dashboard_runtime(app_ctx)
    health = compute_dashboard_health(runtime)

    info = {
        "env": health.env,
        "profile": health.profile,
        "ready": health.ready,
        "has_critical_issues": health.has_critical_issues,
        "has_warnings": health.has_warnings,
        "issues": list(health.issues),
        "warnings": list(health.warnings),
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
# Part 25/35 – Unified Dashboard API bundle (HF-grade external interface)
# =====================

SESSION_KEY_API_BUNDLE_LAST: str = "dashboard_api_bundle_last"


@dataclass
class DashboardAPIBundle:
    """
    חבילה אחודה ("API Bundle") שמרכזת את כל מה שצרכן חיצוני צריך לדעת:

    שימושים טיפוסיים:
    ------------------
    - Desktop Bridge (Qt) שרוצה לקבל:
        * Health (ready / issues / warnings)
        * Summary (services, tabs, user/meta)
        * Agent context (nav history, timings, errors, base_ctx)
    - AI Supervisors / Orchestrators שצריכים תצוגה אחת של מצב הדשבורד.
    - REST API / gRPC שיחשפו "מצב דשבורד" החוצה.

    שדות:
    -----
    meta:
        מטא-דאטה בסיסי (env/profile/run_id/app_name/version/host/user/ts_utc).
    health:
        dict של DashboardHealth (ready/issues/warnings) – JSON-friendly.
    summary:
        dict של DashboardSummary – JSON-friendly.
    agent_context:
        dict של Agent Context (כמו build_agent_context_payload).
    """

    meta: Dict[str, Any]
    health: Dict[str, Any]
    summary: Dict[str, Any]
    agent_context: Dict[str, Any]


def _build_api_meta(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    בונה meta קטן עבור ה-API Bundle:

        {
            "env": ...,
            "profile": ...,
            "run_id": ...,
            "app_name": ...,
            "version": ...,
            "host": ...,
            "user": ...,
            "ts_utc": "...",
        }
    """
    ff = runtime.feature_flags
    return {
        "env": runtime.env,
        "profile": runtime.profile,
        "run_id": runtime.run_id,
        "app_name": ff.get("app_name", APP_NAME),
        "version": ff.get("version", APP_VERSION),
        "host": ff.get("host", RUNTIME_HOST),
        "user": ff.get("user", RUNTIME_USER),
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_dashboard_api_bundle(
    runtime: DashboardRuntime,
    include_summary: bool = True,
    include_health_summary: bool = True,
    include_agent_context: bool = True,
) -> DashboardAPIBundle:
    """
    בונה DashboardAPIBundle מתוך Runtime:

    Flow:
    -----
    1. meta  ← _build_api_meta(runtime)
    2. health  ← export_dashboard_health(runtime, include_summary=include_health_summary)
    3. summary ← export_dashboard_summary(runtime) if include_summary else {}
    4. agent_context ← export_dashboard_state_for_agents(runtime) if include_agent_context else {}

    זהו ה-"שכבת API" המרכזית – צרכן חיצוני יכול לעבוד רק עם האובייקט הזה
    ולא להצטרך לדעת על כל הפונקציות הפנימיות.
    """
    meta = _build_api_meta(runtime)
    health = export_dashboard_health(runtime, include_summary=include_health_summary)

    if include_summary:
        summary = export_dashboard_summary(runtime)
    else:
        summary = {}

    if include_agent_context:
        agent_ctx = export_dashboard_state_for_agents(runtime)
    else:
        agent_ctx = {}

    bundle = DashboardAPIBundle(
        meta=meta,
        health=health,
        summary=summary,
        agent_context=agent_ctx,
    )

    return bundle


def dashboard_api_bundle_to_dict(bundle: DashboardAPIBundle) -> Dict[str, Any]:
    """
    ממיר DashboardAPIBundle ל-dict JSON-friendly.

    מאפשר:
    - שמירה ב-SqlStore/Log.
    - שליחה כסטרוקטורה אחת לסוכני AI / Desktop / REST.
    """
    return {
        "meta": bundle.meta,
        "health": bundle.health,
        "summary": bundle.summary,
        "agent_context": bundle.agent_context,
    }


def update_dashboard_api_bundle_in_session(
    runtime: DashboardRuntime,
    include_summary: bool = True,
    include_health_summary: bool = True,
    include_agent_context: bool = True,
) -> DashboardAPIBundle:
    """
    בונה API Bundle ומעדכן אותו ב-session_state[SESSION_KEY_API_BUNDLE_LAST].

    - שומר dict JSON-friendly (לא את ה- dataclass עצמו).
    - מחזיר את ה-DashboardAPIBundle עבור שימוש מיידי.
    """
    bundle = build_dashboard_api_bundle(
        runtime,
        include_summary=include_summary,
        include_health_summary=include_health_summary,
        include_agent_context=include_agent_context,
    )
    obj = dashboard_api_bundle_to_dict(bundle)
    obj = _make_json_safe(obj)

    try:
        st.session_state[SESSION_KEY_API_BUNDLE_LAST] = obj
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Failed to store dashboard_api_bundle_last in session_state: %s", exc
        )

    logger.debug(
        "Dashboard API bundle updated in session_state (env=%s, profile=%s, keys=%s)",
        bundle.meta.get("env"),
        bundle.meta.get("profile"),
        list(obj.keys()),
    )

    return bundle


def export_dashboard_api_bundle(
    app_ctx: Optional["AppContext"] = None,
    include_summary: bool = True,
    include_health_summary: bool = True,
    include_agent_context: bool = True,
) -> Dict[str, Any]:
    """
    Export "Headless" נוח:

    - בונה Runtime (ensure_dashboard_runtime).
    - בונה DashboardAPIBundle.
    - מחזיר dict JSON-friendly אחד שמכיל:
        * meta
        * health
        * summary
        * agent_context

    מיועד לשימוש:
    -------------
    - בסקריפטים חיצוניים.
    - ב-Desktop Bridge (שמקבל dict ומעבד אותו בצד ה-Qt).
    - ב-REST endpoint שיחשוף `/dashboard/state` או דומה.
    """
    if app_ctx is None:
        app_ctx = get_app_context()

    runtime = ensure_dashboard_runtime(app_ctx)
    bundle = build_dashboard_api_bundle(
        runtime,
        include_summary=include_summary,
        include_health_summary=include_health_summary,
        include_agent_context=include_agent_context,
    )
    payload = dashboard_api_bundle_to_dict(bundle)
    return _make_json_safe(payload)


# עדכון __all__ עבור חלק 25
try:
    __all__ += [
        "SESSION_KEY_API_BUNDLE_LAST",
        "DashboardAPIBundle",
        "build_dashboard_api_bundle",
        "dashboard_api_bundle_to_dict",
        "update_dashboard_api_bundle_in_session",
        "export_dashboard_api_bundle",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_API_BUNDLE_LAST",
        "DashboardAPIBundle",
        "build_dashboard_api_bundle",
        "dashboard_api_bundle_to_dict",
        "update_dashboard_api_bundle_in_session",
        "export_dashboard_api_bundle",
    ]
# =====================
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
    - לשמש בסיס ל-AI Agents / Desktop Bridge כשצריך לתחקר תקלות או בעיות ביצועים.
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


def build_developer_diagnostics(runtime: DashboardRuntime) -> DeveloperDiagnostics:
    """
    בונה DeveloperDiagnostics מתוך Runtime + סביבת ריצה.

    כולל:
    - גרסת Python / Streamlit / pandas.
    - PROJECT_ROOT / cwd / logs path.
    - זמינות מודולי core/root/root_desktop חשובים.
    - envvars רלוונטיים.
    - סטטוס session_state (מס' מפתחות ודוגמאות).
    """
    ff = runtime.feature_flags

    env = runtime.env
    profile = runtime.profile
    app_name = ff.get("app_name", APP_NAME)
    version = ff.get("version", APP_VERSION)
    host = ff.get("host", RUNTIME_HOST)
    user = ff.get("user", RUNTIME_USER)

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

    # warnings – ננסה להפיק כמה תובנות בסיסיות
    warnings: List[str] = []
    if not core_modules.get("core.sql_store", False):
        warnings.append("core.sql_store module not found – SqlStore integration may be limited.")
    if not root_modules.get("root.dashboard_home_v2", False):
        warnings.append("root.dashboard_home_v2 module not found – Home tab may be degraded.")
    if not desktop_modules.get("root_desktop.app", False):
        warnings.append("root_desktop.app module not found – Desktop app integration may be missing.")

    return DeveloperDiagnostics(
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
        warnings=warnings,
    )


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
    - Session keys snapshot
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

    with st.expander("🛠 Developer diagnostics (env, modules, paths)", expanded=False):
        col1, col2, col3 = st.columns([1.4, 1.3, 1.3])

        # ---- Environment & versions ----
        with col1:
            st.markdown("**Environment & versions**")
            st.write(f"Env / Profile: `{env}` / `{profile}`")
            st.write(f"Python: `{python_version}`")
            st.write(f"Streamlit: `{st_version}`  •  pandas: `{pd_version}`")
            st.write(f"Host/User: `{diag_dict.get('host')}` / `{diag_dict.get('user')}`")
            st.write(f"Started at (UTC): `{diag_dict.get('started_at_utc')}`")

            st.markdown("**Paths**")
            st.code(
                f"PROJECT_ROOT = {project_root_str}\n"
                f"cwd          = {cwd}\n"
                f"logs_path    = {logs_path_str}",
                language="text",
            )

        # ---- Modules availability ----
        def _df_from_bool_map(name: str, mapping: Mapping[str, Any]) -> Optional[pd.DataFrame]:
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
            df_core = _df_from_bool_map("core", core_modules)
            if df_core is not None:
                st.dataframe(df_core, width = "stretch")
            else:
                st.caption("No core module snapshot.")

            st.markdown("**Root / Desktop modules**")
            df_root = _df_from_bool_map("root", root_modules)
            df_desktop = _df_from_bool_map("desktop", desktop_modules)

            if df_root is not None:
                st.caption("Root modules:")
                st.dataframe(df_root, width = "stretch")
            if df_desktop is not None:
                st.caption("Desktop modules:")
                st.dataframe(df_desktop, width = "stretch")
            if df_root is None and df_desktop is None:
                st.caption("No root/desktop module snapshot.")

        # ---- Envvars & session state ----
        with col3:
            st.markdown("**Relevant envvars**")
            if envvars:
                env_rows = [{"key": k, "value": v} for k, v in envvars.items()]
                df_env = pd.DataFrame(env_rows).set_index("key")
                st.dataframe(df_env, width = "stretch")
            else:
                st.caption("No envvars snapshot.")

            st.markdown("**Session state**")
            st.write(f"Total keys: `{session_keys_count}`")
            if session_keys_sample:
                st.code(
                    "Sample keys:\n" + "\n".join(f"- {k}" for k in session_keys_sample),
                    language="text",
                )

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
    2. מיישמים את ההעדפות על ה-Runtime (show_debug_info, benchmark, nav_history).
    3. מעדכנים Agent Context (Session-level).
    4. מעדכנים Dashboard Summary (Telemetry).
    5. מעדכנים Dashboard Health (Ready / Issues / Warnings).
    6. מעדכנים Dashboard API Bundle (meta+health+summary+agent_context).
    7. מציגים 🎛 Dashboard Toolbar (שליטה אישית).
    8. מציגים 🛠 Developer Diagnostics (במצב debug בלבד).
    9. שומרים Prefs ל-SqlStore (Best-effort).

    ואז:
    10. קוראים ל-_render_dashboard_shell_core(runtime) – Header + Sidebar + Tabs + Alerts.

    כך, בלי לשבור את ה-API, אנחנו מקבלים:
    - Personalization מתקדם.
    - Telemetry מלא (Summary+Health+API bundle).
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

    # 5) Health – Ready / Issues / Warnings
    update_dashboard_health_in_session(runtime, include_summary=False)

    # 6) API bundle – meta+health+summary+agent_context (Session-level)
    update_dashboard_api_bundle_in_session(
        runtime,
        include_summary=True,
        include_health_summary=False,
        include_agent_context=True,
    )

    # 6.5) Home context – חבילת דאטה מרוכזת לטאב HOME / Monitoring
    update_dashboard_home_context_in_session(runtime)

    # 7) Toolbar – שליטה אישית (layout/debug וכו')
    _render_dashboard_toolbar(runtime, prefs)

    # 8) Developer diagnostics – רק בפרופילי debug
    diag_dict = update_developer_diagnostics_in_session(runtime)
    _render_developer_diagnostics_panel(runtime, diag_dict)

    # Persist prefs (Best-effort)
    persist_user_prefs_if_needed(runtime, prefs, auto_persist=True)

    # 10) Shell המקורי (Header + Sidebar + Tabs + Alerts + Debug timings)
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
# Part 27/35 – Desktop bridge hooks & integration contracts (HF-grade)
# =====================

SESSION_KEY_DESKTOP_LAST_PUSH: str = "dashboard_desktop_last_push"


@dataclass
class DashboardDesktopBridgeConfig:
    """
    תצורת אינטגרציה לדסקטופ (Qt / root_desktop.app):

    מטרות:
    -------
    - להגדיר "חוזה" ברור בין Web Dashboard לבין שכבת ה-Desktop.
    - לאפשר ל-Desktop לדעת:
        * כל כמה זמן למשוך/לקבל סטטוס.
        * איזה פרופיל/טאבים מעניינים אותו (monitoring/trading/risk).
        * האם להסתמך על Health בלבד או גם על Summary/Agent Context.

    שדות:
    -----
    enabled:
        האם בכלל לנסות אינטגרציה לדסקטופ (ברירת מחדל: True אם capabilities.desktop_integration).
    preferred_profile_for_desktop:
        איזה profile הדסקטופ אמור לראות כברירת מחדל (למשל "monitoring").
    push_interval_sec:
        כל כמה שניות (בערך) מומלץ לדסקטופ למשוך/לקבל עדכון (health/api bundle).
        הערה: זה לא timer אמיתי – רק המלצה ל-Desktop Bridge.
    include_summary:
        האם לכלול Summary מלא ב-payload לדסקטופ (כבד יותר אבל עשיר).
    include_agent_context:
        האם לכלול Agent Context (nav history, timings, errors, base_ctx).
    """

    enabled: bool = True
    preferred_profile_for_desktop: ProfileName = "monitoring"
    push_interval_sec: int = 15
    include_summary: bool = True
    include_agent_context: bool = True


def _detect_desktop_bridge_service(app_ctx: "AppContext") -> Tuple[bool, Any]:
    """
    מנסה לגלות Desktop Bridge service מתוך AppContext:

    מחפש בשמות:
        - desktop_bridge
        - desktop
        - qt_bridge
        - qt_app
        - desktop_context

    ומנסה גם בתוך services_map (אם קיים).

    מחזיר:
        (found: bool, service_or_None)
    """
    services_map = _discover_services_mapping(app_ctx)
    found, svc = _probe_service(
        app_ctx,
        services_map,
        candidates=("desktop_bridge", "desktop", "qt_bridge", "qt_app", "desktop_context"),
    )
    return bool(found and svc is not None), svc


def build_desktop_integration_config(runtime: DashboardRuntime) -> DashboardDesktopBridgeConfig:
    """
    בונה DashboardDesktopBridgeConfig מתוך Runtime:

    לוגיקה:
    -------
    - enabled:
        * True אם capabilities.desktop_integration=True.
        * אחרת False (אלא אם בעתיד נחליט להכריח).
    - preferred_profile_for_desktop:
        * אם profile הנוכחי הוא monitoring/risk → נשאיר אותו.
        * אחרת – "monitoring" (דסקטופ נועד לרוב למוניטורינג).
    - push_interval_sec:
        * dev/research → 5–10 שניות (עדכונים תכופים).
        * live/paper → 15–30 שניות.
        * אחרת – 15 שניות.
    """
    caps = runtime.capabilities
    env = runtime.env
    profile = runtime.profile

    enabled = bool(caps.get("desktop_integration", False))

    if profile in ("monitoring", "risk"):
        preferred_profile = profile
    else:
        preferred_profile = "monitoring"

    if env in ("dev", "research", "test"):
        interval = 5
    elif env in ("live", "paper"):
        interval = 20
    else:
        interval = 15

    cfg = DashboardDesktopBridgeConfig(
        enabled=enabled,
        preferred_profile_for_desktop=preferred_profile,  # type: ignore[arg-type]
        push_interval_sec=interval,
        include_summary=True,
        include_agent_context=True,
    )

    return cfg


def build_desktop_payload_from_runtime(
    runtime: DashboardRuntime,
    config: Optional[DashboardDesktopBridgeConfig] = None,
) -> Dict[str, Any]:
    """
    בונה payload עשיר לדסקטופ מתוך Runtime + DesktopBridgeConfig:

    הפלט:
    -----
    dict עם מפתחות עיקריים:
        - "config"     – dict של DashboardDesktopBridgeConfig.
        - "api_bundle" – DashboardAPI bundle (meta+health+summary+agent_context).
        - "tabs"       – רשימת טאבים פעילים (key/label/group).
        - "env" / "profile" / "run_id"  – קונטקסט.

    מבוסס על:
        - export_dashboard_api_bundle(...) (מהחלקים הקודמים).
        - build_active_tabs(...) (לרשימת טאבים).
    """
    if config is None:
        config = build_desktop_integration_config(runtime)

    # api_bundle – מרכז health/summary/agent_context כבר עכשיו
    api_bundle = build_dashboard_api_bundle(
        runtime,
        include_summary=config.include_summary,
        include_health_summary=True,
        include_agent_context=config.include_agent_context,
    )
    api_bundle_dict = dashboard_api_bundle_to_dict(api_bundle)

    # tabs overview
    active_tabs, active_keys, _ = build_active_tabs(runtime.tab_registry, runtime.feature_flags)
    tabs_view = [
        {
            "key": t.key,
            "label": t.label,
            "group": t.group,
            "enabled": True,
        }
        for t in active_tabs
    ]

    payload: Dict[str, Any] = {
        "env": runtime.env,
        "profile": runtime.profile,
        "run_id": runtime.run_id,
        "config": {
            "enabled": config.enabled,
            "preferred_profile_for_desktop": config.preferred_profile_for_desktop,
            "push_interval_sec": config.push_interval_sec,
            "include_summary": config.include_summary,
            "include_agent_context": config.include_agent_context,
        },
        "api_bundle": api_bundle_dict,
        "tabs": tabs_view,
    }

    return _make_json_safe(payload)


def push_dashboard_state_to_desktop_if_available(
    runtime: DashboardRuntime,
    config: Optional[DashboardDesktopBridgeConfig] = None,
) -> Tuple[bool, Optional[str]]:
    """
    מנסה לדחוף את מצב הדשבורד ל-Desktop Bridge, אם קיים:

    Flow:
    -----
    1. בודק capabilities.desktop_integration.
    2. מחפש Desktop Bridge service ב-AppContext:
        - מצפה לאובייקט עם אחת מהפונקציות:
            * push_dashboard_state(payload)
            * push_state(payload)
            * send(payload)
    3. בונה payload דרך build_desktop_payload_from_runtime(...).
    4. קורא לפונקציה הראשונה שמצאנו, מחזיר (True, method_name) אם הצליח.

    הערות:
    -------
    - אם אין Desktop Bridge או capabilities.desktop_integration=False → מחזיר (False, None).
    - לא מפיל את הדשבורד במקרה של שגיאה – רק כותב לוג.
    - מתאים לטריגר:
        * מתוך Agents Tab.
        * מתוך Quick action עתידי "Push state to Desktop".
        * מתוך Desktop עצמו (pull/poll).
    """
    caps = runtime.capabilities
    if not caps.get("desktop_integration", False):
        return False, None

    found, bridge = _detect_desktop_bridge_service(runtime.app_ctx)
    if not found:
        logger.debug("Desktop integration capability enabled, but no desktop_bridge service found.")
        return False, None

    payload = build_desktop_payload_from_runtime(runtime, config)

    method_candidates: Sequence[str] = (
        "push_dashboard_state",
        "push_state",
        "send",
    )

    for name in method_candidates:
        fn = getattr(bridge, name, None)
        if not callable(fn):
            continue
        try:
            fn(payload)  # type: ignore[misc]
            logger.info(
                "Pushed dashboard state to Desktop via %s.%s",
                type(bridge).__name__,
                name,
            )
            # נשמור timestamp אחרון ל-Session (optional telemetry)
            try:
                st.session_state[SESSION_KEY_DESKTOP_LAST_PUSH] = {
                    "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "method": name,
                }
            except Exception:  # pragma: no cover
                pass
            return True, name
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Desktop bridge method %s.%s raised %s – skipping.",
                type(bridge).__name__,
                name,
                exc,
            )

    logger.warning(
        "Desktop bridge service found, but no compatible push method "
        "(push_dashboard_state/push_state/send) succeeded."
    )
    return False, None


def get_last_desktop_push_info() -> Optional[Dict[str, Any]]:
    """
    מחזיר מידע על ה-"Push" האחרון שבוצע ל-Desktop (אם קיים):

    מבנה:
        {
            "ts_utc": "...",
            "method": "push_dashboard_state" | "push_state" | "send",
        }

    שימוש:
    -------
    - להצגה ב-Logs Tab / Agents Tab / Toolbar בעת פיתוח.
    """
    try:
        info = st.session_state.get(SESSION_KEY_DESKTOP_LAST_PUSH)
    except Exception:  # pragma: no cover
        info = None

    if not isinstance(info, Mapping):
        return None

    return dict(info)


# עדכון __all__ עבור חלק 27
try:
    __all__ += [
        "SESSION_KEY_DESKTOP_LAST_PUSH",
        "DashboardDesktopBridgeConfig",
        "build_desktop_integration_config",
        "build_desktop_payload_from_runtime",
        "push_dashboard_state_to_desktop_if_available",
        "get_last_desktop_push_info",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_DESKTOP_LAST_PUSH",
        "DashboardDesktopBridgeConfig",
        "build_desktop_integration_config",
        "build_desktop_payload_from_runtime",
        "push_dashboard_state_to_desktop_if_available",
        "get_last_desktop_push_info",
    ]
# =====================
# Part 28/35 – Agent Action Router (HF-grade AI command interface)
# =====================

SESSION_KEY_AGENT_ACTIONS_HISTORY: str = "dashboard_agent_actions_history"


@dataclass
class AgentAction:
    """
    ייצוג מובנה לפעולה שמגיעה מסוכן AI / Agents Tab / Consumer חיצוני.

    דוגמאות:
    ---------
    - פתיחת טאב:
        action = "open_tab", tab_key = "backtest"
    - פוקוס על זוג והרצת Backtest:
        action = "run_backtest_for_pair",
        tab_key = "backtest",
        payload = {"pair": "AAPL/MSFT", "preset": "smoke", "mode": "wf"}
    - מעבר ל-Risk View:
        action = "open_risk_view",
        tab_key = "risk",
        payload = {"portfolio_id": "default", "view": "limits"}
    - Push של מצב לדסקטופ:
        action = "push_to_desktop"
    - שמירת View:
        action = "save_view",
        payload = {"name": "My favorite monitor setup"}

    שדות:
    -----
    source:
        מקור הפעולה – למשל "agent_tab", "external_api", "desktop", "auto_supervisor".
    action:
        שם הפעולה (open_tab / run_backtest_for_pair / open_risk_view / push_to_desktop / save_view / snapshot ...).
    tab_key:
        טאב יעד, אם רלוונטי (backtest / risk / macro / pair / matrix ...).
    payload:
        פרמטרים נוספים – pair/preset/mode/portfolio_id וכו'.
    ts_utc:
        זמן יצירת הפעולה ב-UTC.
    """

    source: str
    action: str
    tab_key: Optional[TabKey] = None
    payload: Optional[NavPayload] = None
    ts_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def _ensure_agent_actions_history() -> List[Dict[str, Any]]:
    """
    מבטיח שב-session_state תהיה רשימת היסטוריית Actions של סוכנים:

        [
            {
                "ts_utc": "...",
                "source": "...",
                "action": "...",
                "tab_key": "...",
                "payload_keys": [...],
                "result": "ok" | "ignored" | "error",
            },
            ...
        ]
    """
    try:
        obj = st.session_state.get(SESSION_KEY_AGENT_ACTIONS_HISTORY)
    except Exception:  # pragma: no cover
        obj = None

    if not isinstance(obj, list):
        obj = []
        try:
            st.session_state[SESSION_KEY_AGENT_ACTIONS_HISTORY] = obj
        except Exception:
            pass
    return obj  # type: ignore[return-value]


def _record_agent_action_entry(
    action: AgentAction,
    result: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    מוסיף רשומת Action אחת להיסטוריית סוכנים (Session-level).

    result:
        "ok"      – הפעולה מוּשׂמה (למשל nav_target נוצר, snapshot נשמר).
        "ignored" – הפעולה לא רלוונטית / לא נתמכת.
        "error"   – ניסיון ביצוע נכשל (למשל Desktop bridge זרק Exception).
    """
    history = _ensure_agent_actions_history()

    if isinstance(action.payload, Mapping):
        payload_keys = list(action.payload.keys())
    else:
        payload_keys = None

    entry: Dict[str, Any] = {
        "ts_utc": action.ts_utc,
        "source": action.source,
        "action": action.action,
        "tab_key": action.tab_key,
        "payload_keys": payload_keys,
        "result": result,
    }
    if extra:
        entry["extra"] = _make_json_safe(extra)

    history.append(entry)
    try:
        st.session_state[SESSION_KEY_AGENT_ACTIONS_HISTORY] = history
    except Exception:  # pragma: no cover
        pass


def _agent_action_open_tab(
    runtime: DashboardRuntime,
    action: AgentAction,
) -> Dict[str, Any]:
    """
    מטפל בפעולות מסוג "open_tab", "open_risk_view", "open_macro_view" וכו' – כלומר
    פעולות שהמשמעות העיקרית שלהן היא ניווט לטאב מסוים.

    לוגיקה:
    -------
    - אם tab_key לא הועבר בפעולה → נתעלם (ignored).
    - אם הטאב אינו מוגדר ב-registry → ignored.
    - אם הטאב לא enabled עבור הפרופיל הנוכחי → ignored.
    - אחרת → set_nav_target(tab_key, payload) (payload יכולה להיות None או dict).
    """
    tab_key = action.tab_key
    if not tab_key:
        _record_agent_action_entry(action, "ignored", {"reason": "missing_tab_key"})
        return {"status": "ignored", "reason": "missing_tab_key"}

    meta = runtime.tab_registry.get(tab_key)
    if meta is None:
        _record_agent_action_entry(action, "ignored", {"reason": "unknown_tab"})
        return {"status": "ignored", "reason": "unknown_tab"}

    if not _is_tab_enabled_for_profile(meta, runtime.profile):
        _record_agent_action_entry(
            action,
            "ignored",
            {"reason": "tab_not_enabled_for_profile", "profile": runtime.profile},
        )
        return {
            "status": "ignored",
            "reason": "tab_not_enabled_for_profile",
            "profile": runtime.profile,
        }

    set_nav_target(tab_key, action.payload)
    _record_agent_action_entry(action, "ok", {"nav_target_set": True})

    return {
        "status": "ok",
        "nav_target": {"tab_key": tab_key, "payload": action.payload or {}},
    }


def _agent_action_run_backtest_for_pair(
    runtime: DashboardRuntime,
    action: AgentAction,
) -> Dict[str, Any]:
    """
    פעולה יעודית: הרצת Backtest לזוג מסוים / preset / mode.

    היא בפועל *רק* מגדירה nav_target ל-"backtest" עם payload עשיר, כדי
    לאפשר לטאב ה-Backtest לטעון את ההקשר ולבצע את ההרצה.

    מצפה ל-payload בסגנון:
        {"pair": "AAPL/MSFT", "preset": "smoke", "mode": "wf"}
    """
    # אם לא הוגדר tab_key – נכפה "backtest"
    if not action.tab_key:
        action.tab_key = TAB_KEY_BACKTEST

    # נשתמש באותה לוגיקה של _agent_action_open_tab
    result = _agent_action_open_tab(runtime, action)
    result["hint"] = "backtest_for_pair"
    return result


def _agent_action_snapshot(
    runtime: DashboardRuntime,
    action: AgentAction,
) -> Dict[str, Any]:
    """
    פעולה: יצירת Snapshot של הדשבורד (Session + SqlStore אם זמין).

    קוראת ל-trigger_dashboard_snapshot(...) ובונה תוצאה.
    """
    success, method_name = trigger_dashboard_snapshot(
        runtime.app_ctx,
        runtime.feature_flags,
        runtime.services_status,
    )

    status = "ok" if success else "warning"
    info = {
        "status": status,
        "saved_to_sql_store": success,
        "sql_method": method_name,
    }

    _record_agent_action_entry(action, status, info)
    return info


def _agent_action_push_to_desktop(
    runtime: DashboardRuntime,
    action: AgentAction,
) -> Dict[str, Any]:
    """
    פעולה: Push של מצב הדשבורד ל-Desktop Bridge (אם קיים).

    קוראת ל-push_dashboard_state_to_desktop_if_available(...)
    ומחזירה מצב: הצליח/לא, באיזו שיטה.
    """
    config = build_desktop_integration_config(runtime)
    if not config.enabled:
        info = {
            "status": "ignored",
            "reason": "desktop_integration_disabled",
        }
        _record_agent_action_entry(action, "ignored", info)
        return info

    success, method_name = push_dashboard_state_to_desktop_if_available(runtime, config)
    status = "ok" if success else "warning"
    info = {
        "status": status,
        "method": method_name,
        "config": {
            "preferred_profile_for_desktop": config.preferred_profile_for_desktop,
            "push_interval_sec": config.push_interval_sec,
        },
    }
    _record_agent_action_entry(action, status, info)
    return info


def _agent_action_save_view(
    runtime: DashboardRuntime,
    action: AgentAction,
) -> Dict[str, Any]:
    """
    פעולה: שמירת "View" של הדשבורד (ברמת Session בלבד, בשלב זה).

    payload יכולה להכיל:
        - "name": שם ה-View (למשל "Morning monitor").
        - שדות נוספים (לשימוש עתידי).

    בשלב הזה:
        - נשמור רשומה ב-session_state["saved_views"] (אם אין – ניצור).
        - בהמשך ניתן לסנכרן ל-SqlStore.
    """
    try:
        views = st.session_state.get("dashboard_saved_views", [])
    except Exception:
        views = []

    if not isinstance(views, list):
        views = []

    payload = action.payload or {}
    name = str(payload.get("name") or "").strip() or f"view_{len(views) + 1}"

    view_entry = {
        "name": name,
        "ts_utc": action.ts_utc,
        "env": runtime.env,
        "profile": runtime.profile,
        "last_tab_key": get_last_active_tab_key(default=TAB_KEY_HOME),
        "nav_history_tail": _collect_session_nav_history_tail(limit=20),
        "feature_flags": {
            "env": runtime.env,
            "profile": runtime.profile,
            "enable_live_trading_actions": runtime.feature_flags.get(
                "enable_live_trading_actions"
            ),
            "enable_experiment_mode": runtime.feature_flags.get(
                "enable_experiment_mode"
            ),
        },
    }

    views.append(_make_json_safe(view_entry))
    try:
        st.session_state["dashboard_saved_views"] = views
    except Exception:  # pragma: no cover
        pass

    info = {"status": "ok", "name": name, "count": len(views)}
    _record_agent_action_entry(action, "ok", info)
    return info


def handle_agent_action(
    runtime: DashboardRuntime,
    action_data: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Router מרכזי לפעולות של סוכנים (Agent Action Router):

    קלט:
    ----
    action_data – Mapping, למשל:
        {
            "source": "agent_tab",
            "action": "run_backtest_for_pair",
            "tab_key": "backtest",
            "payload": {"pair": "AAPL/MSFT", "preset": "smoke", "mode": "wf"},
        }

    Flow:
    -----
    1. הופך את action_data ל-AgentAction (עם ts_utc).
    2. לפי action:
        - "open_tab" / "open_risk_view" / "open_macro_view" → _agent_action_open_tab
        - "run_backtest_for_pair" → _agent_action_run_backtest_for_pair
        - "snapshot" → _agent_action_snapshot
        - "push_to_desktop" → _agent_action_push_to_desktop
        - "save_view" → _agent_action_save_view
    3. עבור פעולה לא מוכרת → ignored.

    הפלט:
    -----
    dict עם שדות:
        - "status": "ok" / "ignored" / "warning"
        - שדות נוספים לפי הפעולה.
    """
    # ---- בניית AgentAction מובנה ----
    source = str(action_data.get("source") or "agent")
    act = str(action_data.get("action") or "").strip()
    tab_key_raw = action_data.get("tab_key")
    tab_key = str(tab_key_raw).strip() if tab_key_raw else None

    payload_raw = action_data.get("payload")
    if isinstance(payload_raw, Mapping):
        payload = dict(payload_raw)
    elif payload_raw is None:
        payload = None
    else:
        payload = {"value": payload_raw}

    action = AgentAction(
        source=source,
        action=act,
        tab_key=tab_key,
        payload=payload,
    )

    # ---- ניתוב לפי שם פעולה ----
    if act in ("open_tab", "open_risk_view", "open_macro_view", "open_insights"):
        return _agent_action_open_tab(runtime, action)

    if act == "run_backtest_for_pair":
        return _agent_action_run_backtest_for_pair(runtime, action)

    if act in ("snapshot", "create_snapshot"):
        return _agent_action_snapshot(runtime, action)

    if act in ("push_to_desktop", "sync_desktop"):
        return _agent_action_push_to_desktop(runtime, action)

    if act in ("save_view", "bookmark_view"):
        return _agent_action_save_view(runtime, action)

    # פעולה לא מוכרת → ignored
    info = {"status": "ignored", "reason": "unknown_action", "action": act}
    _record_agent_action_entry(action, "ignored", info)
    return info


def handle_agent_actions_batch(
    runtime: DashboardRuntime,
    actions: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """
    מאפשר לסוכן/טאב לשלוח Batch של פעולות בפעם אחת:

        results = handle_agent_actions_batch(runtime, [
            {"action": "open_tab", "tab_key": "pair", ...},
            {"action": "run_backtest_for_pair", ...},
        ])

    הפלט:
        רשימת תוצאות (dict) עבור כל פעולה.
    """
    results: List[Dict[str, Any]] = []
    for data in actions:
        try:
            res = handle_agent_action(runtime, data)
        except Exception as exc:  # pragma: no cover
            res = {
                "status": "error",
                "error": str(exc),
            }
        results.append(res)
    return results


def get_agent_actions_history_tail(limit: int = 50) -> List[Dict[str, Any]]:
    """
    מחזיר את זנב היסטוריית פעולות הסוכנים (Agent Actions) לצורכי Debug/Agents Tab.
    """
    history = _ensure_agent_actions_history()
    if not history:
        return []
    return history[-limit:]


# עדכון __all__ עבור חלק 28
try:
    __all__ += [
        "SESSION_KEY_AGENT_ACTIONS_HISTORY",
        "AgentAction",
        "handle_agent_action",
        "handle_agent_actions_batch",
        "get_agent_actions_history_tail",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_AGENT_ACTIONS_HISTORY",
        "AgentAction",
        "handle_agent_action",
        "handle_agent_actions_batch",
        "get_agent_actions_history_tail",
    ]
# =====================
# Part 29/35 – Saved Views & Layout Profiles (HF-grade view management)
# =====================

SESSION_KEY_SAVED_VIEWS: str = "dashboard_saved_views"


@dataclass
class SavedDashboardView:
    """
    Saved View / Layout Profile ברמת קרן – צילום מצב "אנטומי" של הדשבורד.

    למה זה קיים?
    -------------
    - לאפשר:
        * "בוקר מסחר" → View מסוים (טאבים, פרופיל, סידור, קונטקסט).
        * "מחקר מאקרו" → View אחר (פרופיל macro, טאב macro/matrix וכו').
    - לתת לסוכנים / Desktop אופציה לטעון, לשמור ולשתף Views באופן מובנה.

    שדות:
    -----
    name:
        שם ה-View (ידידותי למשתמש), למשל "Morning monitor" / "Risk overview".
    ts_utc:
        זמן יצירת ה-View (UTC, isoformat).
    env / profile:
        הסביבה והפרופיל שבהם ה-View נוצר.
    last_tab_key:
        הטאב האחרון שהיה פעיל בעת יצירת ה-View.
    feature_flags:
        תת-סט רלוונטי של FeatureFlags – env/profile/live_actions/experiment_mode וכו'.
    nav_history_tail:
        זנב הניווט בזמן יצירת ה-View – מאפשר להבין "איך הגיעו לכאן".
    notes:
        הערות חופשיות (אופציונלי).
    tags:
        רשימת טאגים (למשל ["monitoring", "risk", "live"]).
    """

    name: str
    ts_utc: str
    env: EnvName
    profile: ProfileName
    last_tab_key: TabKey
    feature_flags: Dict[str, Any]
    nav_history_tail: List[Dict[str, Any]] = field(default_factory=list)
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)


def _ensure_saved_views_list() -> List[Dict[str, Any]]:
    """
    מבטיח שב-session_state יש רשימת Saved Views במפתח SESSION_KEY_SAVED_VIEWS.

    מבנה בסיסי:
        [
            { ... SavedDashboardView כ-dict JSON-friendly ... },
            ...
        ]
    """
    try:
        obj = st.session_state.get(SESSION_KEY_SAVED_VIEWS, [])
    except Exception:  # pragma: no cover
        obj = []

    if not isinstance(obj, list):
        obj = []
        try:
            st.session_state[SESSION_KEY_SAVED_VIEWS] = obj
        except Exception:
            pass
    return obj  # type: ignore[return-value]


def _saved_view_to_dict(view: SavedDashboardView) -> Dict[str, Any]:
    """
    ממיר SavedDashboardView ל-dict JSON-friendly.
    """
    return {
        "name": view.name,
        "ts_utc": view.ts_utc,
        "env": view.env,
        "profile": view.profile,
        "last_tab_key": view.last_tab_key,
        "feature_flags": view.feature_flags,
        "nav_history_tail": view.nav_history_tail,
        "notes": view.notes,
        "tags": list(view.tags),
    }


def _saved_view_from_mapping(data: Mapping[str, Any]) -> Optional[SavedDashboardView]:
    """
    מנסה לבנות SavedDashboardView מתוך Mapping.

    במקרה של שדות חסרים באופן קריטי – מחזיר None.
    """
    try:
        name = str(data.get("name") or "").strip()
        ts_utc = str(data.get("ts_utc") or "").strip()
        env = _normalize_env(str(data.get("env") or DEFAULT_ENV))
        profile = _normalize_profile(str(data.get("profile") or DEFAULT_PROFILE))
        last_tab = str(data.get("last_tab_key") or TAB_KEY_HOME).strip() or TAB_KEY_HOME
        ff = data.get("feature_flags") or {}
        if not isinstance(ff, Mapping):
            ff = {}

        nav_tail = data.get("nav_history_tail") or []
        if not isinstance(nav_tail, list):
            nav_tail = []

        notes_val = data.get("notes")
        notes = str(notes_val) if notes_val is not None else None

        tags_raw = data.get("tags") or []
        tags: List[str] = []
        if isinstance(tags_raw, Sequence) and not isinstance(tags_raw, (str, bytes)):
            for t in tags_raw:
                s = str(t).strip()
                if s:
                    tags.append(s)

        if not name or not ts_utc:
            return None

        return SavedDashboardView(
            name=name,
            ts_utc=ts_utc,
            env=env,
            profile=profile,
            last_tab_key=last_tab,
            feature_flags=dict(ff),
            nav_history_tail=list(nav_tail),
            notes=notes,
            tags=tags,
        )
    except Exception:  # pragma: no cover
        return None


def list_saved_views() -> List[SavedDashboardView]:
    """
    מחזיר רשימת SavedDashboardView מה-Session.

    מתעלם מרשומות לא תקינות (שלא מצליחים לפרש).
    """
    raw_list = _ensure_saved_views_list()
    views: List[SavedDashboardView] = []
    for item in raw_list:
        if isinstance(item, Mapping):
            v = _saved_view_from_mapping(item)
            if v is not None:
                views.append(v)
    return views


def add_saved_view_from_runtime(
    runtime: DashboardRuntime,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
) -> SavedDashboardView:
    """
    יוצר SavedDashboardView חדש מתוך Runtime + Session ומוסיף אותו ל-Session.

    אם name לא סופק → ניצור שם אוטומטי:
        view_{N+1} לפי מספר ה-Views הקיימים.

    שדות שנמשכים:
    --------------
    - env/profile/run_id מתוך runtime.
    - last_tab_key מתוך SESSION_KEY_LAST_TAB_KEY.
    - feature_flags subset (env/profile/live_actions/experiment_mode).
    - nav_history_tail (עד 20 רשומות אחרונות).
    """
    views_list = _ensure_saved_views_list()
    existing_views = list_saved_views()
    idx = len(existing_views) + 1

    if not name:
        name = f"view_{idx}"

    # last tab key
    last_tab = get_last_active_tab_key(default=TAB_KEY_HOME)

    # feature flags subset
    ff = runtime.feature_flags
    ff_subset = {
        "env": ff.get("env"),
        "profile": ff.get("profile"),
        "enable_live_trading_actions": ff.get("enable_live_trading_actions"),
        "enable_experiment_mode": ff.get("enable_experiment_mode"),
        "desktop_integration": ff.get("desktop_integration"),
    }

    # nav history tail
    nav_tail = _collect_session_nav_history_tail(limit=20)

    # tags
    tags_list: List[str] = []
    if tags:
        for t in tags:
            s = str(t).strip()
            if s:
                tags_list.append(s)

    view = SavedDashboardView(
        name=name,
        ts_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        env=runtime.env,
        profile=runtime.profile,
        last_tab_key=last_tab,
        feature_flags=ff_subset,
        nav_history_tail=nav_tail,
        notes=notes,
        tags=tags_list,
    )

    views_list.append(_make_json_safe(_saved_view_to_dict(view)))
    try:
        st.session_state[SESSION_KEY_SAVED_VIEWS] = views_list
    except Exception:  # pragma: no cover
        pass

    logger.info(
        "Saved view created: name=%s, env=%s, profile=%s, last_tab=%s",
        view.name,
        view.env,
        view.profile,
        view.last_tab_key,
    )

    return view


def find_saved_view_by_name(name: str) -> Optional[SavedDashboardView]:
    """
    מחפש Saved View לפי שם (case-sensitive).

    אם יש מספר Views עם אותו שם – מחזיר את הראשון שנמצא.
    """
    name = name.strip()
    if not name:
        return None
    for v in list_saved_views():
        if v.name == name:
            return v
    return None


def apply_saved_view(runtime: DashboardRuntime, view: SavedDashboardView) -> Dict[str, Any]:
    """
    מיישם Saved View על הדשבורד ברמת ניווט:

    מה אנחנו עושים עכשיו:
    -----------------------
    - לא נוגעים ב-env/profile (אלו נשלטים ע"י envvars/config/query).
    - כן נשתמש ב:
        * view.last_tab_key → ניווט לטאב המתאים.
        * feature_flags subset (למשל enable_experiment_mode) – אפשרי שימוש עתידי.
        * nav_history_tail – לצורכי Agents/Context בלבד.

    בפועל:
    -------
    - אם last_tab_key זמין ומופעל בפרופיל הנוכחי → set_nav_target(last_tab_key, {}).
    - אם לא זמין → לא נעשה nav_target אבל נחזיר reason.
    """
    tab_key = view.last_tab_key or TAB_KEY_HOME
    meta = runtime.tab_registry.get(tab_key)

    if meta is None or not _is_tab_enabled_for_profile(meta, runtime.profile):
        info = {
            "status": "warning",
            "reason": "tab_not_available_for_profile",
            "tab_key": tab_key,
            "profile": runtime.profile,
        }
        logger.warning(
            "Cannot apply saved view '%s': tab_key=%s not available for profile=%s",
            view.name,
            tab_key,
            runtime.profile,
        )
        return info

    set_nav_target(tab_key, {})  # payload ריק – הטאב יקרא מהקונטקסט הקיים
    logger.info(
        "Applying saved view '%s' (env=%s/profile=%s → tab=%s)",
        view.name,
        view.env,
        view.profile,
        tab_key,
    )

    return {
        "status": "ok",
        "applied_tab": tab_key,
        "view_name": view.name,
    }


def export_saved_views_for_agents(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    מייצא Saved Views בפורמט נוח לסוכנים / Agents Tab / Desktop:

    הפלט:
    -----
    {
        "views": [
            {
                "name": "...",
                "ts_utc": "...",
                "env": "...",
                "profile": "...",
                "last_tab_key": "...",
                "tags": [...],
            },
            ...
        ]
    }

    לא כולל את כל nav_history_tail/feature_flags במפורש – זה יכול להיות כבד.
    Agents יכולים תמיד למשוך את המידע המלא דרך list_saved_views אם הם רצים בפייתון.
    """
    views = list_saved_views()
    out = [
        {
            "name": v.name,
            "ts_utc": v.ts_utc,
            "env": v.env,
            "profile": v.profile,
            "last_tab_key": v.last_tab_key,
            "tags": list(v.tags),
        }
        for v in views
    ]
    return {"views": out}


# אופציונלי: Persist ל-SqlStore (Best-effort בלבד)
def persist_saved_views_to_sql_store(runtime: DashboardRuntime) -> Tuple[bool, Optional[str]]:
    """
    מנסה לשמור Saved Views ל-SqlStore (אם קיים):

    מחפש פונקציות ב-SqlStore:
        - save_dashboard_views(views_json)
        - save_json("dashboard_views", key, views_json)

    מחזיר:
        (success: bool, method_name_used: Optional[str])
    """
    caps = runtime.capabilities
    if not caps.get("sql_store", False):
        return False, None

    services_map = _discover_services_mapping(runtime.app_ctx)
    has_store, store_obj = _probe_service(
        runtime.app_ctx,
        services_map,
        candidates=("sql_store", "store", "db", "sql"),
    )
    if not (has_store and store_obj is not None):
        return False, None

    views = list_saved_views()
    views_dicts = [_make_json_safe(_saved_view_to_dict(v)) for v in views]

    method_candidates: Sequence[str] = (
        "save_dashboard_views",
        "save_json",
    )

    for name in method_candidates:
        fn = getattr(store_obj, name, None)
        if not callable(fn):
            continue
        try:
            if name == "save_json":
                fn("dashboard_views", "default", views_dicts)  # type: ignore[misc]
            else:
                fn(views_dicts)  # type: ignore[misc]
            logger.info(
                "Saved %d dashboard views to SqlStore via %s",
                len(views_dicts),
                name,
            )
            return True, name
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "SqlStore.%s failed to save dashboard views: %s", name, exc
            )

    return False, None


# עדכון __all__ עבור חלק 29
try:
    __all__ += [
        "SESSION_KEY_SAVED_VIEWS",
        "SavedDashboardView",
        "list_saved_views",
        "add_saved_view_from_runtime",
        "find_saved_view_by_name",
        "apply_saved_view",
        "export_saved_views_for_agents",
        "persist_saved_views_to_sql_store",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_SAVED_VIEWS",
        "SavedDashboardView",
        "list_saved_views",
        "add_saved_view_from_runtime",
        "find_saved_view_by_name",
        "apply_saved_view",
        "export_saved_views_for_agents",
        "persist_saved_views_to_sql_store",
    ]
# =====================
# Part 30/35 – Advanced Agents Tab fallback (HF-grade AI control center)
# =====================

def _render_agents_internal_fallback(
    runtime: DashboardRuntime,
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    מימוש פנימי מתקדם לטאב 🤖 Agents, במקרה שאין מודול root.agents_tab ייעודי.

    מה הטאב הזה נותן:
    ------------------
    1. **Context Panel** – מצב המערכת בעיניים של סוכן:
       - env/profile/run_id
       - capabilities/domains
       - health snapshot בסיסי

    2. **Quick Actions** – כפתורים שמפעילים handle_agent_action:
       - Open Backtest / Risk / Macro / Pair tab.
       - Run backtest for selected pair.
       - Snapshot (ל-SqlStore אם זמין).
       - Push state to Desktop (אם יש desktop_integration).

    3. **Saved Views** – יצירה ויישום Views:
       - Create view from current state.
       - Apply existing view.
       - Export views (לסוכנים/desktop).

    4. **Agent Actions History** – tail של פעולות שבוצעו דרך סוכנים.

    זה נותן “Agent Control Center” מינימלי גם בלי לכתוב agents_tab.py נפרד.
    """
    ff = feature_flags
    env = runtime.env
    profile = runtime.profile

    st.markdown("### 🤖 Agents – AI Control & Automation Center")

    # ---- חלק 1: Context Panel ----
    st.markdown("#### 1️⃣ Context overview (Agent view)")

    cols_ctx = st.columns(3)
    with cols_ctx[0]:
        st.markdown("**Runtime**")
        st.write(f"Env / Profile: `{env}` / `{profile}`")
        st.write(f"Run ID: `{runtime.run_id}`")
        st.write(f"User / Host: `{ff.get('user', RUNTIME_USER)}` / `{ff.get('host', RUNTIME_HOST)}`")

    with cols_ctx[1]:
        st.markdown("**Capabilities**")
        caps = runtime.capabilities
        caps_rows = [{"capability": k, "enabled": bool(v)} for k, v in caps.items()]
        if caps_rows:
            df_caps = pd.DataFrame(caps_rows)
            df_caps["status"] = df_caps["enabled"].map(lambda x: "✅" if x else "⭕")
            df_caps.set_index("capability", inplace=True)
            st.dataframe(df_caps, width = "stretch", height=210)
        else:
            st.caption("No capabilities detected.")

    with cols_ctx[2]:
        st.markdown("**Domains**")
        domains = runtime.domains
        dom_rows = [{"domain": k, "active": bool(v)} for k, v in domains.items()]
        if dom_rows:
            df_dom = pd.DataFrame(dom_rows)
            df_dom["status"] = df_dom["active"].map(lambda x: "✅" if x else "⭕")
            df_dom.set_index("domain", inplace=True)
            st.dataframe(df_dom, width = "stretch", height=210)
        else:
            st.caption("No domain flags defined.")

    # ---- חלק 2: Quick Actions (Agent-style) ----
    st.markdown("#### 2️⃣ Quick AI-style actions")

    col_open, col_bt, col_misc = st.columns([1.4, 1.6, 1.3])

    # 2.1 Open tabs
    with col_open:
        st.markdown("**Open / focus tabs**")

        if st.button("🔍 Open Smart Scan", key="agents_open_smart_scan"):
            handle_agent_action(
                runtime,
                {"source": "agents_tab", "action": "open_tab", "tab_key": TAB_KEY_SMART_SCAN},
            )
            st.success("Smart Scan tab will be focused on next refresh.")

        if st.button("🧪 Open Pair Analysis", key="agents_open_pair"):
            handle_agent_action(
                runtime,
                {"source": "agents_tab", "action": "open_tab", "tab_key": TAB_KEY_PAIR},
            )
            st.success("Pair Analysis tab will be focused on next refresh.")

        if st.button("📈 Open Backtest", key="agents_open_backtest"):
            handle_agent_action(
                runtime,
                {"source": "agents_tab", "action": "open_tab", "tab_key": TAB_KEY_BACKTEST},
            )
            st.success("Backtest tab will be focused on next refresh.")

        if st.button("⚠️ Open Risk", key="agents_open_risk"):
            handle_agent_action(
                runtime,
                {"source": "agents_tab", "action": "open_tab", "tab_key": TAB_KEY_RISK},
            )
            st.success("Risk tab will be focused on next refresh.")

    # 2.2 Backtest for pair
    with col_bt:
        st.markdown("**Backtest for pair**")

        pair_str = st.text_input(
            "Pair (symbol1/symbol2)",
            key="agents_bt_pair",
            placeholder="AAPL/MSFT",
        )
        preset = st.selectbox(
            "Preset",
            options=["smoke", "default", "deep", "wf"],
            index=0,
            key="agents_bt_preset",
        )
        mode = st.selectbox(
            "Mode",
            options=["single", "wf"],
            index=1,
            key="agents_bt_mode",
        )

        if st.button("🚀 Run backtest for pair", key="agents_bt_run"):
            pair_clean = pair_str.strip()
            if not pair_clean or "/" not in pair_clean:
                st.error("Please provide pair in format `SYMBOL1/SYMBOL2`.")
            else:
                res = handle_agent_action(
                    runtime,
                    {
                        "source": "agents_tab",
                        "action": "run_backtest_for_pair",
                        "tab_key": TAB_KEY_BACKTEST,
                        "payload": {
                            "pair": pair_clean,
                            "preset": preset,
                            "mode": mode,
                        },
                    },
                )
                if res.get("status") == "ok":
                    st.success(
                        f"Backtest nav_target set for pair `{pair_clean}` "
                        f"(preset={preset}, mode={mode})."
                    )
                else:
                    st.warning(f"Backtest action result: {res}")

    # 2.3 Snapshot / Desktop push
    with col_misc:
        st.markdown("**Snapshot & Desktop**")

        if st.button("💾 Snapshot dashboard", key="agents_snapshot"):
            res = handle_agent_action(
                runtime,
                {"source": "agents_tab", "action": "snapshot"},
            )
            if res.get("status") == "ok":
                st.success("Snapshot saved (SqlStore + session).")
            else:
                st.warning(res)

        if runtime.capabilities.get("desktop_integration", False):
            if st.button("🖥 Push state to Desktop", key="agents_push_desktop"):
                res = handle_agent_action(
                    runtime,
                    {"source": "agents_tab", "action": "push_to_desktop"},
                )
                if res.get("status") == "ok":
                    st.success("Dashboard state pushed to Desktop.")
                else:
                    st.warning(res)
        else:
            st.caption("Desktop integration is not enabled for this runtime.")

    # ---- חלק 3: Saved Views management ----
    st.markdown("#### 3️⃣ Saved views (layouts)")

    views = list_saved_views()
    cols_views = st.columns([1.4, 1.3, 1.3])

    with cols_views[0]:
        st.markdown("**Create new view**")
        new_view_name = st.text_input(
            "View name",
            key="agents_new_view_name",
            placeholder="Morning monitor / Risk overview / ...",
        )
        new_view_tags = st.text_input(
            "Tags (comma-separated)",
            key="agents_new_view_tags",
            placeholder="monitoring, risk, live",
        )
        new_view_notes = st.text_area(
            "Notes (optional)",
            key="agents_new_view_notes",
            height=80,
        )

        if st.button("📌 Save current layout as view", key="agents_save_view"):
            tags_list = [
                t.strip()
                for t in new_view_tags.split(",")
                if t.strip()
            ] if new_view_tags else []
            view = add_saved_view_from_runtime(
                runtime,
                name=new_view_name or None,
                notes=new_view_notes or None,
                tags=tags_list,
            )
            st.success(f"Saved view `{view.name}`.")

    with cols_views[1]:
        st.markdown("**Apply existing view**")
        if views:
            view_names = [v.name for v in views]
            selected_view_name = st.selectbox(
                "Choose a view",
                options=view_names,
                key="agents_apply_view_select",
            )
            if st.button("🎯 Apply view", key="agents_apply_view_btn"):
                v = find_saved_view_by_name(selected_view_name)
                if v is None:
                    st.error("Selected view not found.")
                else:
                    res = apply_saved_view(runtime, v)
                    if res.get("status") == "ok":
                        st.success(
                            f"View `{v.name}` applied – target tab `{res.get('applied_tab')}` will focus next run."
                        )
                    else:
                        st.warning(res)
        else:
            st.caption("No saved views yet – create one on the left.")

    with cols_views[2]:
        st.markdown("**Views snapshot for agents**")
        views_export = export_saved_views_for_agents(runtime)
        st.json(views_export)

    # ---- חלק 4: Agent actions history ----
    st.markdown("#### 4️⃣ Agent actions history (tail)")

    history_tail = get_agent_actions_history_tail(limit=50)
    if not history_tail:
        st.caption("No agent actions recorded yet.")
    else:
        df_hist = pd.DataFrame(history_tail)
        st.dataframe(df_hist, width = "stretch")


# -------------------------
# override: render_agents_tab – keep external, then fallback
# -------------------------

def render_agents_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:  # type: ignore[override]
    """
    🤖 Agents Tab – גרסה משודרגת:

    סדר עדיפות:
    -----------
    1. ניסיון להריץ מודול ייעודי (agents_tab) – כדי לשמור על תאימות מלא.
    2. אם אין מודול / פונקציה מתאימה:
        → fallback פנימי עשיר (_render_agents_internal_fallback) על בסיס DashboardRuntime.

    יתרונות ה-fallback:
    --------------------
    - נותן Agent Control Center מלא גם בלי לבנות agents_tab.py.
    - משתמש בכל התשתיות שכבר בנינו:
        * DashboardRuntime
        * handle_agent_action / batch
        * Saved Views
        * Desktop push
        * nav_target
    """
    _log_tab_entry(TAB_KEY_AGENTS, feature_flags, nav_payload)

    # 1) ניסיון להריץ מודול ייעודי (אם קיים)
    module = _find_module(
        (
            "agents_tab",
            "root.agents_tab",
        )
    )
    if module is not None:
        fn = _find_tab_function_in_module(
            module,
            (
                "render_agents_tab",
                "render_tab",
            ),
        )
        if fn is not None:
            try:
                _invoke_tab_function("agents", fn, app_ctx, feature_flags, nav_payload)
                return
            except Exception as exc:  # pragma: no cover
                logger.error(
                    "External agents_tab renderer raised %s – falling back to internal view.",
                    exc,
                    exc_info=True,
                )

    # 2) fallback פנימי – DashboardRuntime-based
    runtime = ensure_dashboard_runtime(app_ctx)
    _render_agents_internal_fallback(runtime, feature_flags, nav_payload)


# עדכון __all__ עבור חלק 30
try:
    __all__ += [
        "_render_agents_internal_fallback",
        "render_agents_tab",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_render_agents_internal_fallback",
        "render_agents_tab",
    ]
# =====================
# Part 31/35 – Headless / Testing helpers (HF-grade programmatic interface)
# =====================

def get_minimal_dashboard_state(
    app_ctx: Optional["AppContext"] = None,
    include_health: bool = True,
) -> Dict[str, Any]:
    """
    מחזיר מצב מינימלי של הדשבורד בצורה Headless – מתאים ל-Tests / CI / סקריפטים.

    Flow:
    -----
    1. מבטיח AppContext (get_app_context אם לא סופק).
    2. מקים/מרענן DashboardRuntime (ensure_dashboard_runtime).
    3. אוסף נתונים מינימליים:
        - env / profile / run_id
        - app_name / version
        - active_tabs (keys בלבד)
        - capabilities (keys בלבד)
        - health summary (אופציונלי, include_health=True)

    הפלט:
    -----
    dict JSON-friendly, לדוגמה:
        {
            "env": "dev",
            "profile": "trading",
            "run_id": "...",
            "app_name": "...",
            "version": "...",
            "active_tabs": ["home", "smart_scan", "pair", ...],
            "capabilities": ["sql_store", "risk_engine", ...],
            "health": {
                "ready": true,
                "has_critical_issues": false,
                "has_warnings": true,
                ...
            }
        }
    """
    if app_ctx is None:
        app_ctx = get_app_context()

    runtime = ensure_dashboard_runtime(app_ctx)

    ff = runtime.feature_flags
    env = runtime.env
    profile = runtime.profile
    run_id = runtime.run_id
    app_name = ff.get("app_name", APP_NAME)
    version = ff.get("version", APP_VERSION)

    active_tabs, active_keys, _ = build_active_tabs(runtime.tab_registry, ff)
    cap_keys = list(runtime.capabilities.keys())

    result: Dict[str, Any] = {
        "env": env,
        "profile": profile,
        "run_id": run_id,
        "app_name": app_name,
        "version": version,
        "active_tabs": active_keys,
        "capabilities": cap_keys,
    }

    if include_health:
        health = compute_dashboard_health(runtime)
        result["health"] = {
            "ready": health.ready,
            "has_critical_issues": health.has_critical_issues,
            "has_warnings": health.has_warnings,
            "issues": list(health.issues),
            "warnings": list(health.warnings),
            "ts_utc": health.ts_utc,
        }

    return _make_json_safe(result)


def run_headless_agent_actions(
    actions: Sequence[Mapping[str, Any]],
    app_ctx: Optional["AppContext"] = None,
) -> Dict[str, Any]:
    """
    מריץ סדרת פעולות Agent באופן Headless (ללא UI) – שימושי ל-Tests / CI / סקריפטים.

    דוגמה:
    -------
        res = run_headless_agent_actions([
            {"action": "open_tab", "tab_key": "backtest"},
            {"action": "run_backtest_for_pair",
             "payload": {"pair": "AAPL/MSFT", "preset": "smoke", "mode": "wf"}},
            {"action": "snapshot"},
        ])

    הפלט:
    -----
    dict:
        {
            "env": "...",
            "profile": "...",
            "run_id": "...",
            "results": [ {...}, {...}, ... ],
            "nav_target_after": {... or None},
        }

    הערות:
    -------
    - הפונקציה לא מציירת כלום; היא עובדת רק עם Runtime + nav_target.
    - מתאימה גם ל-Supervisor Agents שרוצים להריץ סיקוונסי פעולות
      בלי להריץ `streamlit run`.
    """
    if app_ctx is None:
        app_ctx = get_app_context()

    runtime = ensure_dashboard_runtime(app_ctx)

    # לפני – ננקה nav_target כדי שנדע מה נוצר כתוצאה מהפעולות
    clear_nav_target()

    results = handle_agent_actions_batch(runtime, actions)

    nav = get_nav_target()

    out: Dict[str, Any] = {
        "env": runtime.env,
        "profile": runtime.profile,
        "run_id": runtime.run_id,
        "results": results,
        "nav_target_after": nav,
    }

    return _make_json_safe(out)


def export_dashboard_for_test_snapshot(
    app_ctx: Optional["AppContext"] = None,
) -> Dict[str, Any]:
    """
    ייצוא "Snapshot ל-Tests" – יותר עשיר מ-minimal אבל קל לצילום / השוואה.

    כולל:
    -----
    - minimal_state  (get_minimal_dashboard_state)
    - api_bundle.meta
    - api_bundle.health.ready / has_critical_issues
    - רשימת active_tabs + services health (שמות+severity)

    זה מאפשר בבדיקות:
    - לוודא שהדשבורד "עולה" (ready=True).
    - לראות אם יש שירותים ב-severity=error.
    - להשוות תצורה בין ריצות (בצורה מרוכזת).
    """
    if app_ctx is None:
        app_ctx = get_app_context()

    runtime = ensure_dashboard_runtime(app_ctx)

    minimal = get_minimal_dashboard_state(app_ctx, include_health=True)
    api_bundle = build_dashboard_api_bundle(
        runtime,
        include_summary=False,
        include_health_summary=True,
        include_agent_context=False,
    )
    api_dict = dashboard_api_bundle_to_dict(api_bundle)

    # שירותים – שם + severity
    health = compute_dashboard_health(runtime)
    services = [
        {"name": s.name, "severity": s.severity, "available": s.available}
        for s in health.summary.services
    ] if health.summary is not None else []

    snapshot: Dict[str, Any] = {
        "minimal": minimal,
        "api_meta": api_dict.get("meta", {}),
        "api_health": api_dict.get("health", {}),
        "services": services,
    }

    return _make_json_safe(snapshot)


# עדכון __all__ עבור חלק 31
try:
    __all__ += [
        "get_minimal_dashboard_state",
        "run_headless_agent_actions",
        "export_dashboard_for_test_snapshot",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "get_minimal_dashboard_state",
        "run_headless_agent_actions",
        "export_dashboard_for_test_snapshot",
    ]

# =====================
# Part 32/35 – Dashboard Alert Bus (HF-grade alerts & notifications)
# =====================

SESSION_KEY_ALERTS: str = "dashboard_alerts"


@dataclass
class DashboardAlert:
    """
    Alert / Notification ברמת דשבורד – Event Bus פנימי לסיכון, מקרו, דאטה וסוכנים.

    שימושים:
    ---------
    - Risk Engine יכול לדווח:
        * "Exposure limit breached" / "Kill-switch armed" וכו'.
    - Macro Engine יכול לדווח:
        * שינוי Regime / אירוע מקרו חשוב.
    - Agents יכולים לדווח:
        * פעולה בוצעה / נדחתה / נכשלה.

    שדות:
    -----
    id:
        מזהה ייחודי (UUID4 hex).
    ts_utc:
        טיימסטמפ ב-UTC (isoformat, seconds).
    level:
        רמת חשיבות: "info" / "success" / "warning" / "error".
    source:
        מקור: למשל "risk_engine", "macro_engine", "agent", "backtest_tab".
    message:
        הודעה קצרה, קריאה לבני אדם.
    details:
        dict אופציונלי עם שדות נוספים (pair, portfolio_id, regime, limit_name וכו').
    """

    id: str
    ts_utc: str
    level: Literal["info", "success", "warning", "error"]
    source: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


def _ensure_alerts_list() -> List[Dict[str, Any]]:
    """
    מבטיח רשימת Alerts ב-session_state:

        [
            { ... DashboardAlert כ-dict JSON-friendly ... },
            ...
        ]
    """
    try:
        obj = st.session_state.get(SESSION_KEY_ALERTS, [])
    except Exception:  # pragma: no cover
        obj = []

    if not isinstance(obj, list):
        obj = []
        try:
            st.session_state[SESSION_KEY_ALERTS] = obj
        except Exception:
            pass
    return obj  # type: ignore[return-value]


def _alert_to_dict(alert: DashboardAlert) -> Dict[str, Any]:
    """
    המרה ל-dict JSON-friendly.
    """
    return {
        "id": alert.id,
        "ts_utc": alert.ts_utc,
        "level": alert.level,
        "source": alert.source,
        "message": alert.message,
        "details": alert.details,
    }


def _alert_from_mapping(data: Mapping[str, Any]) -> Optional[DashboardAlert]:
    """
    המרה ממבנה dict ל-DashboardAlert (Best-effort, לא נכשל על שדות חסרים קטנים).
    """
    try:
        alert_id = str(data.get("id") or uuid.uuid4().hex)
        ts = str(
            data.get("ts_utc")
            or datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        level_raw = str(data.get("level") or "info").lower()
        level: Literal["info", "success", "warning", "error"]
        if level_raw not in ("info", "success", "warning", "error"):
            level = "info"
        else:
            level = level_raw  # type: ignore[assignment]

        source = str(data.get("source") or "system")
        message = str(data.get("message") or "").strip()
        details = data.get("details") or {}
        if not isinstance(details, Mapping):
            details = {"value": details}

        if not message:
            return None

        return DashboardAlert(
            id=alert_id,
            ts_utc=ts,
            level=level,
            source=source,
            message=message,
            details=dict(details),
        )
    except Exception:  # pragma: no cover
        return None


def emit_dashboard_alert(
    level: Literal["info", "success", "warning", "error"],
    source: str,
    message: str,
    details: Optional[Mapping[str, Any]] = None,
) -> DashboardAlert:
    """
    מוסיף Alert חדש ל-Bus הדשבורד.

    דוגמאות:
    ---------
        emit_dashboard_alert(
            "warning",
            "risk_engine",
            "Exposure limit exceeded for portfolio 'core_fund'",
            {"portfolio_id": "core_fund", "limit": "gross_exposure", "value": 1.25},
        )

        emit_dashboard_alert(
            "error",
            "broker",
            "IBKR connection lost",
            {"host": "127.0.0.1", "port": 7497},
        )
    """
    details_dict: Dict[str, Any] = dict(details) if isinstance(details, Mapping) else {}
    alert = DashboardAlert(
        id=uuid.uuid4().hex,
        ts_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        level=level,
        source=source,
        message=message,
        details=_make_json_safe(details_dict),
    )

    alerts_list = _ensure_alerts_list()
    alerts_list.append(_alert_to_dict(alert))

    try:
        st.session_state[SESSION_KEY_ALERTS] = alerts_list
    except Exception:  # pragma: no cover
        pass

    logger.info(
        "Dashboard alert emitted: level=%s, source=%s, message=%s",
        alert.level,
        alert.source,
        alert.message,
    )

    return alert


def get_dashboard_alerts(limit: Optional[int] = None) -> List[DashboardAlert]:
    """
    מחזיר רשימת Alerts (מהחדש לישן). אם limit סופק – מחזיר רק את ה-N האחרונים.
    """
    alerts_raw = _ensure_alerts_list()
    alerts: List[DashboardAlert] = []
    for item in alerts_raw:
        if isinstance(item, Mapping):
            a = _alert_from_mapping(item)
            if a is not None:
                alerts.append(a)

    alerts.sort(key=lambda a: a.ts_utc, reverse=True)

    if limit is not None and limit > 0:
        alerts = alerts[:limit]

    return alerts


def clear_dashboard_alerts(level: Optional[str] = None, source: Optional[str] = None) -> None:
    """
    מנקה Alerts מה-Bus:

    - אם level=None ו-source=None → מנקה הכל.
    - אם level לא None → מנקה Alerts רק ברמת level מסוימת.
    - אם source לא None → מנקה Alerts רק מ-Source מסוים.

    ניתן להשתמש גם בשילוב:
        clear_dashboard_alerts(level="info", source="system")
    """
    alerts_raw = _ensure_alerts_list()
    if not alerts_raw:
        return

    filtered: List[Dict[str, Any]] = []

    for item in alerts_raw:
        if not isinstance(item, Mapping):
            continue
        if level is not None:
            lvl = str(item.get("level") or "").lower()
            if lvl == level.lower():
                continue
        if source is not None:
            src = str(item.get("source") or "")
            if src == source:
                continue
        filtered.append(item)

    try:
        st.session_state[SESSION_KEY_ALERTS] = filtered
    except Exception:  # pragma: no cover
        pass


def _render_alert_badge(alert: DashboardAlert) -> None:
    """
    מציג Alert קטן (Badge-style) ב-UI, לשימוש על ידי Tabs/Toolbar בעתיד.
    """
    icon = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "🚨",
    }.get(alert.level, "ℹ️")

    st.write(
        f"{icon} `{alert.ts_utc}` • **{alert.source}** – {alert.message}"
    )


def render_dashboard_alert_center(
    max_items: int = 10,
    filter_levels: Optional[Sequence[str]] = None,
) -> None:
    """
    רנדר מרכזי של Alerts (Alert Center) – לא מחובר אוטומטית ל-Shell,
    אלא מיועד לשימוש מטאבים כמו:
        - Home
        - Risk
        - Agents
        - Logs

    פרמטרים:
    --------
    max_items:
        מספר ה-Alerts המקסימלי להצגה (ברירת מחדל: 10).
    filter_levels:
        אם לא None – מציג רק Alerts ש-level שלהם נמצא ברשימה (למשל ["warning", "error"]).
    """
    alerts = get_dashboard_alerts(limit=max_items)
    if filter_levels:
        lvl_set = {lvl.lower() for lvl in filter_levels}
        alerts = [a for a in alerts if a.level.lower() in lvl_set]

    if not alerts:
        st.caption("No dashboard alerts at the moment.")
        return

    for alert in alerts:
        _render_alert_badge(alert)


# עדכון __all__ עבור חלק 32
try:
    __all__ += [
        "SESSION_KEY_ALERTS",
        "DashboardAlert",
        "emit_dashboard_alert",
        "get_dashboard_alerts",
        "clear_dashboard_alerts",
        "render_dashboard_alert_center",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_ALERTS",
        "DashboardAlert",
        "emit_dashboard_alert",
        "get_dashboard_alerts",
        "clear_dashboard_alerts",
        "render_dashboard_alert_center",
    ]
# =====================
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
    - "experiments_mode" → On / Off

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
        טקסט קצר שמסביר את המצב, שורה–שתיים.
    extra:
        dict אופציונלי עם מידע נוסף למי שרוצה לצלול לעומק (Agents/Home Tab).
    """

    key: str
    label: str
    value: str
    level: Literal["info", "success", "warning", "error"]
    description: str
    extra: Dict[str, Any] = field(default_factory=dict)


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
    if has_warnings:
        return "warning"
    if ready:
        return "success"
    return "info"


def _build_system_health_metric(
    runtime: DashboardRuntime,
    health: DashboardHealth,
) -> OverviewMetric:
    """
    בונה מטריקת System Health מסיכום ברמת DashboardHealth.
    """
    level = _overview_level_from_health(
        ready=health.ready,
        has_critical=health.has_critical_issues,
        has_warnings=health.has_warnings,
    )

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

    return OverviewMetric(
        key="system_health",
        label="System health",
        value=value,
        level=level,
        description=desc,
        extra={
            "ready": health.ready,
            "has_critical_issues": health.has_critical_issues,
            "has_warnings": health.has_warnings,
            "issues": health.issues,
            "warnings": health.warnings,
        },
    )


def _build_risk_metric(
    runtime: DashboardRuntime,
    summary: DashboardSummary,
) -> OverviewMetric:
    """
    בונה מטריקת Risk Status מתוך DashboardSummary + capabilities.
    """
    caps = runtime.capabilities
    svc_map = {svc.name: svc for svc in summary.services}
    risk = svc_map.get("risk_engine")
    vec: Dict[str, Any] = {
        "risk_available": bool(risk.available) if risk else False,
        "severity": risk.severity if risk else "warning",
    }

    if not caps.get("risk_engine", False) or risk is None:
        value = "Offline"
        level: Literal["info", "success", "warning", "error"] = "warning"
        desc = "Risk Engine אינו זמין – מגבלות סיכון לא נאכפות אוטומטית."
    else:
        if risk.severity == "error":
            value = "Error"
            level = "error"
            desc = "Risk Engine דיווח על תקלה – בדוק מגבלות וסיגנאלים."
        elif risk.severity == "warning":
            value = "Warning"
            level = "warning"
            desc = "Risk Engine פעיל אך קיים מצב אזהרה (למשל חריגות מגבלה)."
        else:
            value = "OK"
            level = "success"
            desc = "Risk Engine זמין ומדווח מצב תקין."

    return OverviewMetric(
        key="risk_status",
        label="Risk status",
        value=value,
        level=level,
        description=desc,
        extra=vec,
    )


def _build_macro_metric(
    runtime: DashboardRuntime,
    summary: DashboardSummary,
) -> OverviewMetric:
    """
    בונה מטריקת Macro Regime (אם Macro Engine זמין).
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
        )

    details = macro.details or {}
    regime = details.get("regime") or "Neutral"

    # פירוש Regime → Level
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
        extra={"macro_available": True, "regime": regime},
    )


def _build_agents_metric(runtime: DashboardRuntime, summary: DashboardSummary) -> OverviewMetric:
    """
    מטריקה: מצב סוכנים (Agents).
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
    )


def _build_desktop_metric(runtime: DashboardRuntime) -> OverviewMetric:
    """
    מטריקה: מצב אינטגרציית Desktop.
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
    )


def _build_experiments_metric(runtime: DashboardRuntime) -> OverviewMetric:
    """
    מטריקה: מצב מצב “Experiment” (Backtests/Optimization פעילים).
    """
    ff = runtime.feature_flags
    experiments_enabled = bool(ff.get("enable_experiment_mode", False))

    if not experiments_enabled:
        return OverviewMetric(
            key="experiments",
            label="Experiments",
            value="Off",
            level="info",
            description=(
                "מצב ניסוי (Experiments) כבוי – "
                "Backtests/Optimization עדיין זמינים ידנית."
            ),
            extra={"enable_experiment_mode": False},
        )

    return OverviewMetric(
        key="experiments",
        label="Experiments",
        value="On",
        level="success",
        description=(
            "Experiment mode פעיל – "
            "ניתן להריץ Backtests/Optimization בצורה אינטנסיבית."
        ),
        extra={"enable_experiment_mode": True},
    )



def build_dashboard_overview_metrics(runtime: DashboardRuntime) -> List[OverviewMetric]:
    """
    בונה רשימת OverviewMetric – "תוכן" לכרטיסי Overview בדשבורד (Home / Monitoring).

    משתמש ב:
    ---------
    - DashboardHealth (compute_dashboard_health)
    - DashboardSummary (build_dashboard_summary)
    - capabilities (Risk/Macro/Agents/Desktop/Experiments)

    התוצאה:
    --------
    רשימה מסודרת של כרטיסים לוגיים:
        [system_health, risk_status, macro_regime, agents_status, desktop_link, experiments]
    """
    health = compute_dashboard_health(runtime)
    summary = health.summary or build_dashboard_summary(runtime)

    metrics: List[OverviewMetric] = []

    metrics.append(_build_system_health_metric(runtime, health))
    metrics.append(_build_risk_metric(runtime, summary))
    metrics.append(_build_macro_metric(runtime, summary))
    metrics.append(_build_agents_metric(runtime, summary))
    metrics.append(_build_desktop_metric(runtime))
    metrics.append(_build_experiments_metric(runtime))

    return metrics


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
        "Dashboard overview metrics updated (count=%d, keys=%s)",
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
# Part 34/35 – Dashboard Home data helpers (HF-grade Home/Monitoring context)
# =====================

SESSION_KEY_HOME_CONTEXT: str = "dashboard_home_context"


@dataclass
class DashboardHomeContext:
    """
    חבילת דאטה מרוכזת לטאב הבית (Dashboard Home v2 / Monitoring):

    הרעיון:
    --------
    לתת ל-dashboard_home_v2.py "מגש כסף" עם כל מה שהוא צריך:
    - Overview cards (metrics)
    - Alerts
    - מצב Health / Ready
    - Snapshot קצר של API Bundle (meta בלבד)
    - מידע על Saved Views / Agent actions אחרונים

    שדות:
    -----
    env / profile / run_id:
        הקונטקסט הלוגי של ה-Runtime.
    overview_metrics:
        רשימת dict-ים עבור Cards (build_dashboard_overview_metrics → dict).
    alerts:
        Alerts אחרונים (warning/error בעיקר).
    health_light:
        סיכום קל של Health (ready / has_critical_issues / has_warnings).
    api_meta:
        meta קל מתוך Dashboard API Bundle (app_name/version/host/user/ts_utc).
    saved_views:
        רשימה קלה של Saved Views (שם/טאב אחרון/prof/env/tags).
    agent_actions_tail:
        tail קצר של פעולות Agents אחרונות (כדי להציג "מה קרה עכשיו").
    """

    env: EnvName
    profile: ProfileName
    run_id: str
    overview_metrics: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    health_light: Dict[str, Any]
    api_meta: Dict[str, Any]
    saved_views: List[Dict[str, Any]]
    agent_actions_tail: List[Dict[str, Any]]


def _alert_to_light_dict(alert: DashboardAlert) -> Dict[str, Any]:
    """
    ממיר DashboardAlert למבנה קל לטאב הבית (בלי details כבדים).
    """
    return {
        "id": alert.id,
        "ts_utc": alert.ts_utc,
        "level": alert.level,
        "source": alert.source,
        "message": alert.message,
    }


def build_dashboard_home_context(runtime: DashboardRuntime) -> DashboardHomeContext:
    """
    בונה DashboardHomeContext מלא מתוך Runtime:

    Flow:
    -----
    1. Overview metrics (update_dashboard_overview_in_session).
    2. Alerts אחרונים (warning/error).
    3. Health קל (compute_dashboard_health).
    4. API meta מתוך Dashboard API Bundle.
    5. Saved views (export_saved_views_for_agents).
    6. Agent actions tail (get_agent_actions_history_tail).

    הטאב home יכול פשוט לקרוא:
        ctx = build_dashboard_home_context(runtime)
    או:
        ctx_dict = dashboard_home_context_to_dict(ctx)
    ולהשתמש בו ל-Cards / Alerts / Panels.
    """
    # 1) Overview metrics
    overview = update_dashboard_overview_in_session(runtime)

    # 2) Alerts (רק warning/error)
    alerts_objs = get_dashboard_alerts(limit=20)
    alerts_filtered = [
        a for a in alerts_objs if a.level in ("warning", "error")
    ]
    alerts_light = [_alert_to_light_dict(a) for a in alerts_filtered]

    # 3) Health light
    health = compute_dashboard_health(runtime)
    health_light = {
        "ready": health.ready,
        "has_critical_issues": health.has_critical_issues,
        "has_warnings": health.has_warnings,
        "issues": list(health.issues),
        "warnings": list(health.warnings),
        "ts_utc": health.ts_utc,
    }

    # 4) API meta only (קל)
    api_bundle = build_dashboard_api_bundle(
        runtime,
        include_summary=False,
        include_health_summary=False,
        include_agent_context=False,
    )
    api_meta = api_bundle.meta

    # 5) Saved views (light)
    views_light = export_saved_views_for_agents(runtime).get("views", [])

    # 6) Agent actions tail
    actions_tail = get_agent_actions_history_tail(limit=30)

    ctx = DashboardHomeContext(
        env=runtime.env,
        profile=runtime.profile,
        run_id=runtime.run_id,
        overview_metrics=overview,
        alerts=alerts_light,
        health_light=health_light,
        api_meta=api_meta,
        saved_views=views_light,
        agent_actions_tail=actions_tail,
    )

    return ctx


def dashboard_home_context_to_dict(ctx: DashboardHomeContext) -> Dict[str, Any]:
    """
    ממיר DashboardHomeContext ל-dict JSON-friendly – מיועד ל-Agents / Desktop / REST.
    """
    return {
        "env": ctx.env,
        "profile": ctx.profile,
        "run_id": ctx.run_id,
        "overview_metrics": _make_json_safe(ctx.overview_metrics),
        "alerts": _make_json_safe(ctx.alerts),
        "health_light": _make_json_safe(ctx.health_light),
        "api_meta": _make_json_safe(ctx.api_meta),
        "saved_views": _make_json_safe(ctx.saved_views),
        "agent_actions_tail": _make_json_safe(ctx.agent_actions_tail),
    }


def update_dashboard_home_context_in_session(runtime: DashboardRuntime) -> Dict[str, Any]:
    """
    בונה DashboardHomeContext, שומר אותו ב-session_state, ומחזיר dict JSON-friendly.

    מיועד ל:
    --------
    - Dashboard Home v2 (root/dashboard_home_v2.py) – יכול פשוט לקרוא
      get_dashboard_home_context_from_session() כדי לקבל את ההקשר.
    - Agents Tab / Desktop – לשימוש בתצוגות Monitoring.
    """
    ctx = build_dashboard_home_context(runtime)
    ctx_dict = dashboard_home_context_to_dict(ctx)

    try:
        st.session_state[SESSION_KEY_HOME_CONTEXT] = ctx_dict
    except Exception:  # pragma: no cover
        pass

    logger.debug(
        "Dashboard home context updated (env=%s, profile=%s, metrics=%d, alerts=%d)",
        ctx.env,
        ctx.profile,
        len(ctx.overview_metrics),
        len(ctx.alerts),
    )

    return ctx_dict


def get_dashboard_home_context_from_session() -> Optional[Dict[str, Any]]:
    """
    מחזיר את Dashboard Home context האחרון מה-Session, אם קיים.

    אם לא קיים – מחזיר None (הטאב home יכול לבחור לבנות אחד חדש דרך Runtime).
    """
    try:
        obj = st.session_state.get(SESSION_KEY_HOME_CONTEXT)
    except Exception:  # pragma: no cover
        obj = None

    if not isinstance(obj, Mapping):
        return None

    return dict(obj)


# עדכון __all__ עבור חלק 34
try:
    __all__ += [
        "SESSION_KEY_HOME_CONTEXT",
        "DashboardHomeContext",
        "build_dashboard_home_context",
        "dashboard_home_context_to_dict",
        "update_dashboard_home_context_in_session",
        "get_dashboard_home_context_from_session",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_HOME_CONTEXT",
        "DashboardHomeContext",
        "build_dashboard_home_context",
        "dashboard_home_context_to_dict",
        "update_dashboard_home_context_in_session",
        "get_dashboard_home_context_from_session",
    ]
# =====================
# Part 35/35 – Public API manifest & __main__ entrypoint (HF-grade)
# =====================

def get_dashboard_public_api() -> Dict[str, Any]:
    """
    מחזיר Manifest של ה-API הציבורי של dashboard.py לשימוש ע"י מודולים אחרים:

    מטרות:
    -------
    - לתת נקודת גישה אחת ברורה למה שמומלץ לייבא ממודול הדשבורד.
    - להקל על Desktop / Agents / Tabs אחרים לדעת מה זמין בלי לנבור בקוד.

    הפלט:
    -----
    dict עם קבוצות מפתחות:

        {
            "entrypoints": {
                "run_dashboard_entry": run_dashboard_entry,
            },
            "runtime": {
                "ensure_dashboard_runtime": ensure_dashboard_runtime,
                "get_dashboard_runtime": get_dashboard_runtime,
            },
            "headless": {
                "get_minimal_dashboard_state": get_minimal_dashboard_state,
                "run_headless_agent_actions": run_headless_agent_actions,
                "export_dashboard_for_test_snapshot": export_dashboard_for_test_snapshot,
                "check_dashboard_ready": check_dashboard_ready,
                "export_dashboard_api_bundle": export_dashboard_api_bundle,
            },
            "agents": {
                "handle_agent_action": handle_agent_action,
                "handle_agent_actions_batch": handle_agent_actions_batch,
            },
            "views": {
                "list_saved_views": list_saved_views,
                "add_saved_view_from_runtime": add_saved_view_from_runtime,
                "apply_saved_view": apply_saved_view,
                "export_saved_views_for_agents": export_saved_views_for_agents,
            },
            "alerts": {
                "emit_dashboard_alert": emit_dashboard_alert,
                "get_dashboard_alerts": get_dashboard_alerts,
                "clear_dashboard_alerts": clear_dashboard_alerts,
            },
            "overview": {
                "build_dashboard_overview_metrics": build_dashboard_overview_metrics,
                "update_dashboard_overview_in_session": update_dashboard_overview_in_session,
            },
            "home": {
                "update_dashboard_home_context_in_session": update_dashboard_home_context_in_session,
                "get_dashboard_home_context_from_session": get_dashboard_home_context_from_session,
            },
        }

    הערה:
    -----
    • ניתן להשתמש ב-manifest הזה גם לצרכי introspection ע"י Agents/Desktop.
    """
    return {
        "entrypoints": {
            "run_dashboard_entry": run_dashboard_entry,
        },
        "runtime": {
            "ensure_dashboard_runtime": ensure_dashboard_runtime,
            "get_dashboard_runtime": get_dashboard_runtime,
            "create_dashboard_runtime": create_dashboard_runtime,
        },
        "headless": {
            "get_minimal_dashboard_state": get_minimal_dashboard_state,
            "run_headless_agent_actions": run_headless_agent_actions,
            "export_dashboard_for_test_snapshot": export_dashboard_for_test_snapshot,
            "check_dashboard_ready": check_dashboard_ready,
            "export_dashboard_api_bundle": export_dashboard_api_bundle,
        },
        "agents": {
            "handle_agent_action": handle_agent_action,
            "handle_agent_actions_batch": handle_agent_actions_batch,
            "get_agent_actions_history_tail": get_agent_actions_history_tail,
        },
        "views": {
            "list_saved_views": list_saved_views,
            "add_saved_view_from_runtime": add_saved_view_from_runtime,
            "find_saved_view_by_name": find_saved_view_by_name,
            "apply_saved_view": apply_saved_view,
            "export_saved_views_for_agents": export_saved_views_for_agents,
        },
        "alerts": {
            "emit_dashboard_alert": emit_dashboard_alert,
            "get_dashboard_alerts": get_dashboard_alerts,
            "clear_dashboard_alerts": clear_dashboard_alerts,
        },
        "overview": {
            "build_dashboard_overview_metrics": build_dashboard_overview_metrics,
            "update_dashboard_overview_in_session": update_dashboard_overview_in_session,
            "get_dashboard_overview_from_session": get_dashboard_overview_from_session,
        },
        "home": {
            "update_dashboard_home_context_in_session": update_dashboard_home_context_in_session,
            "get_dashboard_home_context_from_session": get_dashboard_home_context_from_session,
        },
        "desktop": {
            "build_desktop_integration_config": build_desktop_integration_config,
            "build_desktop_payload_from_runtime": build_desktop_payload_from_runtime,
            "push_dashboard_state_to_desktop_if_available": push_dashboard_state_to_desktop_if_available,
            "get_last_desktop_push_info": get_last_desktop_push_info,
        },
        "diagnostics": {
            "build_developer_diagnostics": build_developer_diagnostics,
            "update_developer_diagnostics_in_session": update_developer_diagnostics_in_session,
            "export_dashboard_health": export_dashboard_health,
            "export_dashboard_summary": export_dashboard_summary,
        },
    }


def _debug_print_structure() -> None:
    """
    כלי עזר קטן למפתחים: מדפיס לקונסול מבט-על על מבנה הדשבורד.

    • לא מיועד לשימוש Production, אלא לעבודה מקומית/CI.
    • ניתן להריץ:
        python -m root.dashboard --inspect
    """
    print("=== Pairs Trading Dashboard – Structure Overview ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"LOGS_DIR    : {LOGS_DIR}")
    print(f"APP_NAME    : {APP_NAME} (v{APP_VERSION})")
    print()
    print("Public API groups:")
    api = get_dashboard_public_api()
    for group, mapping in api.items():
        print(f"- {group}:")
        for name in mapping.keys():
            print(f"  • {name}")
    print("====================================================")


if __name__ == "__main__":
    """
    Entry point להרצה ישירה של הקובץ:

    שימושים:
    --------
    1. הרצת דשבורד כ-Web (דרך Streamlit):
         streamlit run root/dashboard.py

    2. הרצה דיאגנוסטית מהטרמינל:
         python -m root.dashboard --inspect

       במצב זה נדפיס את מבנה ה-API ולא ננסה להריץ Streamlit.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Pairs Trading Dashboard entrypoint")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print dashboard structure / public API and exit (no Streamlit).",
    )
    args, _ = parser.parse_known_args()

    if args.inspect:
        _debug_print_structure()
    else:
        # במצב רגיל – נסמוך על Streamlit שיקרא את run_dashboard_entry
        # כאשר מריצים: streamlit run root/dashboard.py
        # אם בכל זאת קראו את הקובץ כ-__main__ ללא Streamlit – ננסה להריץ את הדשבורד
        # (זה יעבוד רק אם הקוד רץ בסביבה התואמת Streamlit).
        run_dashboard_entry()


# עדכון __all__ עבור חלק 35
try:
    __all__ += [
        "get_dashboard_public_api",
        "_debug_print_structure",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "get_dashboard_public_api",
        "_debug_print_structure",
    ]
