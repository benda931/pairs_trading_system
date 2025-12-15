### Summary of changes

- Fixed minor logic/cleanliness issues (`_decorate_label` unreachable return, stray `...` statement) and kept the module structurally identical.
- Added a small, centralized cancellation API (`SESSION_KEY_CANCEL_REQUESTED` + helpers) and wired it into sidebar quick actions and heavy-tab nav payloads (Backtest, Optimisation, Smart Scan) to act as a cooperative cancel hook.
- Extended session bootstrap to initialize the new cancellation state for deterministic behaviour across runs.
- Updated quick actions sidebar to expose a clear “Cancel jobs” control and surfaced current cancel state/metadata alongside snapshot and cache status.
- Propagated the cancel state into the backtest/optimisation/smart-scan nav payloads so underlying tab modules can read it without any inline trading logic in the dashboard.
- Kept environment banners, toolbar env/profile switches, health summaries, logs/latency/health-check integration, and the single-coherent DashboardRuntime orchestration intact.
- Ensured new helpers are exported via `__all__` and safe-guarded against session_state/Streamlit errors for robustness in headless/tests.

```python
﻿# -*- coding: utf-8 -*-
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
    cast,
)

from collections.abc import Mapping, Sequence, Callable, MutableMapping
import json

import pandas as pd
import streamlit as st
import sys
import platform

# -----------------------------------------------------------------------------
# פרמטרים גלובליים, נתיבי פרויקט ומידע סביבת ריצה
# -----------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# חשוב: לוודא שה-ROOT של הפרויקט נמצא על sys.path,
# כדי ש-importים כמו core.*, common.*, dashboard_service_factory יעבדו.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.app_context import get_app_context  # noqa: E402

# TYPE_CHECKING imports בלבד כדי להימנע מ-Circular imports בזמן ריצה
if TYPE_CHECKING:  # pragma: no cover
    from core.app_context import AppContext  # pragma: no cover

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


def _get_dashboard_ctx(section: str = "dashboard") -> "AppContext":
    """
    Wrapper נוח לדשבורד:
    - משתמש ב-get_app_context (ששומר ctx ברמת session).
    - מעדכן section (למשל 'dashboard' / 'backtest' / 'macro' וכו').
    """
    return get_app_context(section=section, refresh=False, ensure_services=True)

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

    בנוסף:
    --------
    - מצב מקצועי "force_all_tabs" שמיועד ל-dev/research:
        * מציג את כל הטאבים ב-UI, גם אם capabilities חסרים.
        * עדיין משאיר את הלוגיקה הפנימית של כל טאב (אם אין engine → הטאב יציג אזהרה).
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
        "pair": True,
        "matrix": caps.market_data_router or caps.sql_store,
        "comparison_matrices": caps.market_data_router or caps.sql_store,
        "backtest": caps.backtester,
        "optimization": caps.optimizer or caps.meta_optimizer,
        "insights": caps.backtester or caps.optimizer,
        "macro": caps.macro_engine,
        "portfolio": caps.sql_store or caps.broker,
        "risk": caps.risk_engine or caps.sql_store,
        "fair_value": caps.fair_value_engine,
        "config": True,
        "agents": caps.agents_manager,
        "logs": True,
    }

    # -------- מצב "כל הטאבים" (HF-grade, לא פלסטר) --------
    #
    # לוגיקה:
    # - כברירת מחדל: ב-env=dev או profile=research → force_all_tabs=True
    # - אפשר לבטל/להדליק ידנית דרך env var: PAIRS_FORCE_ALL_TABS
    raw_force = os.getenv("PAIRS_FORCE_ALL_TABS")
    if raw_force is None:
        # ברירת מחדל: להראות הכל ב-dev/research
        force_all_tabs = env in ("dev", "research")
    else:
        force_all_tabs = raw_force.strip().lower() not in ("0", "false", "no", "off")

    # אם force_all_tabs=True → לא נסתמך על capabilities כדי להדליק/לכבות טאבים
    if force_all_tabs:
        for k in tabs_flags.keys():
            tabs_flags[k] = True

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
        # הדגל החדש – שאר הקוד יכבד אותו
        "force_all_tabs": force_all_tabs,
    }

    logger.info(
        "Feature flags resolved: env=%s, profile=%s, live_actions=%s, "
        "sql_state=%s, experiment_mode=%s, tabs={risk=%s, macro=%s, agents=%s}, force_all_tabs=%s",
        env,
        profile,
        enable_live_trading_actions,
        use_sql_backed_state,
        enable_experiment_mode,
        tabs_flags.get("risk"),
        tabs_flags.get("macro"),
        tabs_flags.get("agents"),
        force_all_tabs,
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
SESSION_KEY_CANCEL_REQUESTED: str = "dashboard_cancel_requested"
SESSION_KEY_CANCEL_META: str = "dashboard_cancel_meta"


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
    - cancel flags (ברירת מחדל False/None).
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

    # cancel flags – ברירת מחדל
    _ensure_session_default(SESSION_KEY_CANCEL_REQUESTED, False)
    _ensure_session_default(SESSION_KEY_CANCEL_META, None)

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
        "SESSION_KEY_CANCEL_REQUESTED",
        "SESSION_KEY_CANCEL_META",
        "_ensure_session_default",
        "_ensure_session_default_factory",
        "_get_or_create_run_id",
        "_extract_base_currency",
        "_extract_timezone",
       