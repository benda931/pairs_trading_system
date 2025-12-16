# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict, field, replace
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)
from types import SimpleNamespace
import json
import logging
import os

import numpy as np
import pandas as pd
import streamlit as st

from common.json_safe import make_json_safe
from core.ib_order_router import IBOrderRouter
from common.config_manager import load_config
from core.fair_value_config import FairValueAPIConfig

logger = logging.getLogger(__name__)

# ========= Type aliases =========

JSONValue = Any
JSONDict = Dict[str, JSONValue]
TagDict = Dict[str, str]
HealthFlagList = List[str]


# ========= Core constants =========

CURRENT_CTX_SCHEMA_VERSION: str = "1.0.0"
_DEF_SEED: int = 1337

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
LOGS_DIR: Path = PROJECT_ROOT / "logs"
SNAPSHOTS_DIR: Path = LOGS_DIR / "snapshots"

APP_PROFILE: str = (os.getenv("APP_PROFILE", "default") or "default").strip() or "default"
_APP_ENV_RAW: str = (os.getenv("APP_ENV", "dev") or "dev").strip() or "dev"


def _normalize_env_value(env: str) -> str:
    """
    Normalize environment name into one of:
    - 'dev', 'research', 'paper', 'live'

    Unknown values are mapped to 'dev' for safety.
    """
    env_norm = (env or "").strip().lower()
    if not env_norm:
        return "dev"
    if env_norm in {"dev", "research", "paper", "live"}:
        return env_norm
    # Allow some common aliases
    if env_norm in {"backtest", "research-dev", "sandbox"}:
        return "research"
    if env_norm in {"paper-trading", "papertrade"}:
        return "paper"
    if env_norm in {"production", "prod"}:
        return "live"
    return "dev"


APP_ENVIRONMENT: str = _normalize_env_value(_APP_ENV_RAW)


# ========= Service Protocols (typed interfaces, non-invasive) =========

@runtime_checkable
class SqlStoreLike(Protocol):
    """Minimal interface we expect from SqlStore for lifecycle management."""

    def close(self) -> None:  # pragma: no cover - depends on concrete impl
        ...

    def dispose(self) -> None:  # pragma: no cover - optional on some impls
        ...


@runtime_checkable
class MarketDataRouterLike(Protocol):
    """Minimal interface we expect from the market data router."""

    def close(self) -> None:  # pragma: no cover
        ...

    def shutdown(self) -> None:  # pragma: no cover
        ...


@runtime_checkable
class BrokerLike(Protocol):
    """Minimal interface we expect from a broker / order router."""

    def close(self) -> None:  # pragma: no cover
        ...

    def disconnect(self) -> None:  # pragma: no cover
        ...


# ========= Provenance & scenario overlay =========


@dataclass
class ContextProvenance:
    parent_run_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    created_by: str = "dashboard"        # "dashboard" / "agent:optimizer" / "cli" וכו'
    source_section: Optional[str] = None  # "scan" / "backtest" / "optimize" / ...
    source_file: Optional[str] = None
    git_rev: Optional[str] = field(default_factory=lambda: os.getenv("GIT_REV", None))
    notes: Optional[str] = None          # הערות קצרות על למה הקונטקסט קיים

    # Lineage (עץ ניסויים / DAG)
    children_run_ids: List[str] = field(default_factory=list)
    lineage_depth: int = 0               # 0 = root, 1 = child, 2 = grand-child...


@dataclass
class ScenarioOverlay:
    """
    תוצאה של סימולציית תרחיש על הקונטקסט.
    """
    name: str
    spy_shock_pct: float
    vix_target: Optional[float] = None
    equity_impact: Optional[float] = None
    equity_impact_pct: Optional[float] = None
    dd_impact_pct: Optional[float] = None
    meta: JSONDict = field(default_factory=dict)


@dataclass
class ScenarioGridResult:
    """
    גריד של תרחישים על אותו קונטקסט:
    - רשימת תרחישים (ScenarioOverlay)
    - טבלת סיכום (summary) לפילוח / הצגה בטאב Insights / Risk.
    """
    scenarios: List[ScenarioOverlay]
    summary: JSONDict = field(default_factory=dict)


@dataclass
class AppContext:
    """
    AppContext — קונטקסט אחד לכל ריצה / טאבים — ברמת קרן גידור.

    מה הוא מחזיק?
    --------------
    1. ליבה של הריצה:
       - start_date, end_date, capital, max_exposure_per_trade, max_leverage, pairs
       - config, controls, seed, run_id

    2. זיהוי סביבת ריצה:
       - section     — "dashboard" / "scan" / "backtest" / ...
       - profile     — APP_PROFILE (קונפיג לוגי: dev_default, live_fund, וכו')
       - environment — "dev" / "research" / "paper" / "live"
       - settings    — dict כללי להגדרות גלובליות (שיתוף עם קונפיגים אחרים)

    3. Services / Engines:
       - sql_store, market_data_router, risk_engine, signals_engine, macro_engine,
         fair_value_engine, agents_manager, broker, dashboard_service, desktop_bridge,
         ib_router
       - services: mapping חופשי לשירותים נוספים (key → object)
       - Extension point: טאב/סרוויס חדש יכול להירשם דרך set_service("name", obj)
         או לדרוס דיפולטים (למשל market_data_router) לפני/אחרי init_services.

    4. Fair Value API:
       - fair_value_api: FairValueAPIConfig — קונפיג מלא ל-HTTP API (local/remote/mock)

    5. Governance / Meta:
       - ctx_version, config_hash, parent_run_id, source_section
       - tags, health_score, health_flags, policy_status, risk_budget,
         scenario_overlay, provenance

    6. ניסויים / RL / הערות:
       - experiment_name, experiment_group, run_label
       - reward, state_features, context_notes

    בנוסף:
    -------
    - עוזרי נוחות: get_service/set_service, clone_for_section, add_tag, add_health_flag,
      to_dict, log_event, save_snapshot.
    - Lifecycle: close()/__enter__/__exit__ לסגירה מסודרת של Broker/DB/Router.
    """

    # --- ליבה: חובה (arguments ללא default) ---
    start_date: date
    end_date: date
    capital: float
    max_exposure_per_trade: float
    max_leverage: float
    pairs: List[Any]
    config: JSONDict
    controls: JSONDict
    seed: int
    run_id: str

    # --- זיהוי סביבת הריצה / פרופיל ---
    section: str = "dashboard"              # "dashboard" / "scan" / "backtest" / ...
    profile: str = APP_PROFILE              # פרופיל קונפיג (לוגי)
    environment: str = APP_ENVIRONMENT      # "dev" / "research" / "paper" / "live"

    # router ישן (נשאר ל-backward-compat)
    md_router: Any = None                   # MarketDataRouter / None

    # --- Services / Engines (נחשפים ל-dashboard.py דרך discover_capabilities) ---
    services: Dict[str, Any] = field(default_factory=dict, repr=False)

    sql_store: Optional[SqlStoreLike] = None
    market_data_router: Optional[MarketDataRouterLike] = None
    risk_engine: Optional[Any] = None
    signals_engine: Optional[Any] = None
    macro_engine: Optional[Any] = None
    fair_value_engine: Optional[Any] = None
    agents_manager: Optional[Any] = None
    broker: Optional[BrokerLike] = None
    dashboard_service: Optional[Any] = None
    desktop_bridge: Optional[Any] = None
    ib_router: Optional[IBOrderRouter] = field(default=None, repr=False)

    # --- Fair Value API Config (חדש) ---
    fair_value_api: FairValueAPIConfig = field(
        default_factory=FairValueAPIConfig,
        repr=False,
    )

    # --- Governance / Meta ---
    ctx_version: str = CURRENT_CTX_SCHEMA_VERSION
    config_hash: Optional[str] = None
    parent_run_id: Optional[str] = None
    source_section: Optional[str] = None

    tags: TagDict = field(default_factory=dict)               # macro_regime / experiment_group / ...
    health_score: Optional[float] = None                      # 0–100
    health_flags: HealthFlagList = field(default_factory=list)
    policy_status: JSONDict = field(default_factory=dict)     # {"risk_policy_ok": True, ...}
    risk_budget: Optional[JSONDict] = None                    # תקציב ריסק נוכחי (var/dd/leverage/vix)
    scenario_overlay: Optional[ScenarioOverlay] = None
    provenance: Optional[ContextProvenance] = None

    # --- ניסויים / RL / הערות ---
    experiment_name: Optional[str] = None
    experiment_group: Optional[str] = None
    run_label: Optional[str] = None        # label חופשי ("MR_30d_v3", "WF_tail_risk"...)
    reward: Optional[float] = None         # reward ל-RL (פונקציה על KPIs)
    state_features: Optional[JSONDict] = None  # פיצ'רים מרוכזים לשימוש ב-Insights/RL
    context_notes: Optional[str] = None    # הערות על הריצה / החלטות
    _settings_cache: Any = field(default=None, repr=False, init=False)

    # ==================================================================
    # Helpers / Convenience methods
    # ==================================================================

    # ---- Services management ----

    def get_service(self, name: str, default: Any = None) -> Any:
        """
        החזרת service לפי שם:
        - אם יש attribute בשם הזה (למשל risk_engine) → מחזירים אותו.
        - אחרת מחפשים ב-services[name].
        - אחרת מחזירים default.
        """
        if hasattr(self, name):
            val = getattr(self, name)
            if val is not None:
                return val
        return self.services.get(name, default)

    def set_service(self, name: str, service: Any) -> None:
        """
        רישום/Override של service חדש:
        - אם יש attribute עם השם הזה בקלאס → נשבץ אליו.
        - אחרת → נכניס ל-services[name].

        דוגמה להרחבת פלטפורמה:
        ----------------------
        - רישום Data Provider חדש:
            ctx.set_service("market_data_router", my_router)
        - רישום אסטרטגיה חדשה:
            ctx.set_service("strategy:my_pair_strategy", my_strategy_obj)
        """
        if hasattr(self, name):
            setattr(self, name, service)
        else:
            self.services[name] = service

    # ---- Tags / health ----

    def add_tag(self, key: str, value: str) -> None:
        """
        הוספת tag יחיד (למשל macro_regime="risk_on").
        """
        self.tags[key] = value

    def add_health_flag(self, flag: str) -> None:
        """
        הוספת דגל health – למשל:
        - "risk_policy_violation"
        - "missing_data"
        - "ib_disconnected"
        """
        if flag not in self.health_flags:
            self.health_flags.append(flag)

    def set_health_score(self, score: float) -> None:
        """
        עדכון health_score (0–100) עם גזירה בטוחה.
        """
        self.health_score = float(max(0.0, min(100.0, score)))

    # ---- Cloning / section-specific variations ----

    def clone_for_section(self, section: str, **overrides: Any) -> AppContext:
        """
        יצירת AppContext חדש עבור section אחר, עם אותן הגדרות בסיסיות.

        לדוגמה:
        >>> ctx_scan = ctx.clone_for_section("scan", run_id="scan_123")
        """
        return replace(self, section=section, **overrides)

    # ---- Structured logging helpers ----

    def log_event(self, level: int, message: str, **extra: Any) -> None:
        """
        Structured log helper:
        - מוסיף run_id / section / environment לכל לוג.
        - מיועד לשימוש מתוך טאב/שירותים שמקבלים ctx.
        """
        if not logger.isEnabledFor(level):
            return
        base_extra: Dict[str, Any] = {
            "run_id": self.run_id,
            "section": self.section,
            "environment": self.environment,
            "profile": self.profile,
        }
        base_extra.update(extra)
        logger.log(level, message, extra={"ctx": base_extra})

    # ---- Lifecycle management (Broker / DB / Routers) ----

    def close(self) -> None:
        """
        Gracefully shut down heavy resources attached to this context.

        Safe to call multiple times, and safe in Streamlit rerun scenarios.
        """
        # Broker / IB router
        for broker_obj in (self.broker, self.ib_router):
            if broker_obj is None:
                continue
            for method_name in ("close", "disconnect", "stop"):
                method = getattr(broker_obj, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as exc:  # pragma: no cover - best-effort
                        logger.debug(
                            "AppContext.close: broker %s() failed: %s",
                            method_name,
                            exc,
                        )

        # Market data router
        if self.market_data_router is not None:
            for method_name in ("close", "shutdown"):
                method = getattr(self.market_data_router, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as exc:  # pragma: no cover
                        logger.debug(
                            "AppContext.close: market_data_router %s() failed: %s",
                            method_name,
                            exc,
                        )

        # SqlStore
        if self.sql_store is not None:
            for method_name in ("close", "dispose"):
                method = getattr(self.sql_store, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as exc:  # pragma: no cover
                        logger.debug(
                            "AppContext.close: sql_store %s() failed: %s",
                            method_name,
                            exc,
                        )

        # Agents manager (if has shutdown semantics)
        if self.agents_manager is not None:
            for method_name in ("close", "shutdown", "stop"):
                method = getattr(self.agents_manager, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception as exc:  # pragma: no cover
                        logger.debug(
                            "AppContext.close: agents_manager %s() failed: %s",
                            method_name,
                            exc,
                        )

        # After closing, mark services as uninitialised; safe to re-init later
        self._services_initialized = False  # type: ignore[attr-defined]

    def save_snapshot(
        self,
        *,
        section: Optional[str] = None,
        extra: Optional[JSONDict] = None,
    ) -> Path:
        """
        Convenience wrapper ל-save_run_snapshot.
        מאפשר לכל טאב/שירות לשמור snapshot לוגי (כולל config_hash/run_id).
        """
        return save_run_snapshot(self, section=section, extra=extra)

    def __enter__(self) -> AppContext:
        """
        Allow 'with AppContext(...) as ctx:' usage in CLI/background jobs.
        """
        try:
            self.init_services()
        except Exception as exc:  # pragma: no cover - protective
            logger.warning("AppContext.__enter__: init_services failed: %s", exc)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ========= Class-level global ctx =========
    _GLOBAL_CTX: ClassVar[Optional[AppContext]] = None

    @classmethod
    def get_global(cls) -> AppContext:
        """
        מחזיר קונטקסט ברירת מחדל יחיד (singleton) שהדשבורד יכול לעבוד איתו.

        עדכון HF-grade:
        ----------------
        - קודם ננסה לאחזר ctx מ-st.session_state["app_ctx"] (per-user ב-Streamlit).
        - אם אין, נשתמש ב-_GLOBAL_CTX (תמיכה אחורה ל-CLI/tests).
        - רק אם לא קיים כלל — נבנה ctx חדש מה-config.

        זהו *החוזה הרשמי* עם dashboard.py, אך לקוד חדש מומלץ להשתמש ב־get_app_context().
        """
        # 1) Streamlit session-scoped AppContext (מועדף ל-Web).
        sess_ctx: Optional[AppContext]
        try:
            sess_ctx = st.session_state.get("app_ctx")  # type: ignore[assignment]
        except Exception:
            sess_ctx = None

        if isinstance(sess_ctx, cls):
            ctx = sess_ctx
            try:
                ctx.init_services()
            except Exception as exc:  # pragma: no cover
                logger.debug("init_services on session app_ctx failed: %s", exc)
            cls._GLOBAL_CTX = ctx
            return ctx

        # 2) תהליך-גלובלי (שימוש קיים ב-CLI/בדיקות)
        if cls._GLOBAL_CTX is not None:
            try:
                cls._GLOBAL_CTX.init_services()
            except Exception as exc:  # pragma: no cover
                logger.debug("init_services on existing GLOBAL_CTX failed: %s", exc)
            return cls._GLOBAL_CTX

        # 3) בנייה חדשה מקובץ config.json
        try:
            cfg = load_config()
            logger.info("AppContext.get_global: loaded config via load_config()")
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "AppContext.get_global: failed to load config.json via load_config(): %s",
                exc,
            )
            cfg = {}

        # --- 2) Umwelt: env/profile ברירת מחדל מתוך הקונפיג (אם קיימים) ---
        env_cfg = cfg.get("environment") or {}
        default_env_raw = str(
            env_cfg.get("default_env") or APP_ENVIRONMENT or "dev"
        ).strip() or "dev"
        default_env = _normalize_env_value(default_env_raw)
        default_profile = str(
            env_cfg.get("default_profile") or APP_PROFILE or "default"
        ).strip() or "default"

        today = date.today()
        seed_val = _DEF_SEED
        run_id = f"bootstrap-{seed_val}-{today.strftime('%Y%m%d')}"

        # ננסה לקחת md_router קיים מה-Session אם כבר יש אחד
        try:
            md_router = st.session_state.get("md_router")
        except Exception:
            md_router = None

        ctx = AppContext(
            start_date=today,
            end_date=today,
            capital=float(cfg.get("risk", {}).get("max_gross_exposure", 100_000.0)),
            max_exposure_per_trade=float(
                cfg.get("strategy", {}).get("max_exposure_per_trade", 0.05)
            ),
            max_leverage=float(cfg.get("risk", {}).get("max_leverage", 2.0)),
            pairs=[],
            config=cfg,
            controls={},
            seed=seed_val,
            run_id=run_id,
            section="dashboard",
            profile=default_profile,
            environment=default_env,
            md_router=md_router,
        )

        try:
            ctx.init_services()
        except Exception as exc:  # pragma: no cover
            logger.warning("AppContext.init_services() failed in get_global: %s", exc)

        cls._GLOBAL_CTX = ctx
        try:
            # לא חובה ב-CLI, אבל מוסיף עקביות ב-Web
            st.session_state["app_ctx"] = ctx
        except Exception:
            pass
        return ctx

    # ---- Compatibility / integration with dashboard.py ----

    @property
    def env(self) -> str:
        """Alias כדי ש-dashboard יוכל לקרוא app_ctx.env אם ירצה."""
        return self.environment

    @property
    def settings(self) -> Any:
        """
        אובייקט settings מינימלי שהמערכת כולה יכולה לעבוד איתו.

        חשוב:
        - settings.config  → ה-dict המלא של config.json
        - settings.engine_url / settings.sql_store_url → נגזר מ-config["sql_store"] /
          config["data"]["sql_store"] / config["paths"]["sql_store_url"] / config["engine_url"]
        """
        # Cache כדי לא לחשב בכל קריאה מחדש
        if self._settings_cache is not None:
            return self._settings_cache

        cfg = dict(self.config or {})

        # -------- SQL: חיפוש engine_url לפי הסכמה החדשה --------
        sql_cfg = cfg.get("sql_store") or {}
        data_sql = (cfg.get("data") or {}).get("sql_store") or {}
        paths = cfg.get("paths") or {}

        engine_url: Optional[str] = None
        candidates = [
            sql_cfg.get("engine_url"),
            data_sql.get("engine_url"),
            paths.get("sql_store_url"),
            cfg.get("engine_url"),
            data_sql.get("url"),
            sql_cfg.get("url"),
            cfg.get("sql_store_url"),
            cfg.get("SQL_STORE_URL"),
        ]
        for cand in candidates:
            if isinstance(cand, str) and cand.strip():
                engine_url = cand.strip()
                break

        # fallback ל־env vars אם אין כלום בקונפיג
        if engine_url is None:
            env_candidates = [
                os.getenv("SQL_STORE_URL"),
                os.getenv("PAIRS_SQL_STORE_URL"),
                os.getenv("PAIRS_SQL_URL"),
            ]
            for cand in env_candidates:
                if isinstance(cand, str) and cand.strip():
                    engine_url = cand.strip()
                    break

        # אם גם כאן אין – נשאיר None; SqlStore.from_settings כבר ידאג ל־fallback
        sql_store_url = engine_url

        # -------- IB: לוקחים מתוך בלוק ibkr בקונפיג החדש אם קיים --------
        ib_cfg = cfg.get("ibkr") or {}
        ib_host = ib_cfg.get("host", cfg.get("ib_host", "127.0.0.1"))
        ib_port = int(ib_cfg.get("port", cfg.get("ib_port", 7497)))
        ib_client_id = int(ib_cfg.get("client_id", cfg.get("ib_client_id", 1)))
        ib_mode = ib_cfg.get(
            "mode",
            cfg.get("ib_profile", cfg.get("IB_MODE", self.environment)),
        )
        ib_readonly = bool(ib_cfg.get("readonly", cfg.get("ib_readonly", False)))

        base_currency = (
            (cfg.get("portfolio") or {}).get("currency")
            or cfg.get("base_currency", "USD")
        )
        tz = cfg.get("timezone", "Asia/Jerusalem")

        ns = SimpleNamespace(
            # ליבה
            env=self.environment,
            profile=self.profile,
            base_currency=base_currency,
            timezone=tz,
            # IB-related
            ib_host=ib_host,
            ib_port=ib_port,
            ib_client_id=ib_client_id,
            ib_readonly=ib_readonly,
            ib_profile=str(ib_mode),
            # SQL
            engine_url=engine_url,
            sql_store_url=sql_store_url,
            sql_read_only=bool(cfg.get("sql_read_only", False)),
            sql_echo=bool(cfg.get("sql_echo", False)),
            # כל הקונפיג הגולמי (למי שרוצה להיכנס פנימה)
            config=cfg,
        )

        self._settings_cache = ns
        return ns


    @settings.setter
    def settings(self, value: Any) -> None:  # type: ignore[override]
        """
        Setter 'דמי' כדי להרגיע את __init__ של dataclass אם הוא מנסה להציב settings.

        כאן אנחנו מאפשרים גם override יזום:
        - אם מישהו מציב settings חיצוני (למשל ב-tests/CLI) → נשמור אותו ב-cache.
        """
        self._settings_cache = value


    # ========= Methods =========

    def to_dict(self) -> JSONDict:
        """
        JSON-safe dict לכל הקונטקסט (כולל תתי-אובייקטים),
        *ללא* אובייקטי services/engines שלא ניתנים לסיריאליזציה.
        """
        raw = asdict(self)

        # שדות runtime-only שלא נכנסים ל-JSON / snapshots
        runtime_keys = [
            "services",
            "sql_store",
            "market_data_router",
            "risk_engine",
            "signals_engine",
            "macro_engine",
            "fair_value_engine",
            "agents_manager",
            "broker",
            "dashboard_service",
            "desktop_bridge",
        ]
        for k in runtime_keys:
            raw.pop(k, None)

        return make_json_safe(raw)

    def short_summary(self) -> JSONDict:
        """
        תקציר קומפקטי ל-UI / לוגים.
        """
        return {
            "run_id": self.run_id,
            "section": self.section,
            "profile": self.profile,
            "environment": self.environment,
            "dates": f"{self.start_date} → {self.end_date}",
            "capital": self.capital,
            "max_exposure_per_trade": self.max_exposure_per_trade,
            "max_leverage": self.max_leverage,
            "pairs_count": len(self.pairs),
            "ctx_version": self.ctx_version,
            "config_hash": self.config_hash,
            "health_score": self.health_score,
            "tags": dict(self.tags),
        }

    def init_services(self) -> None:
        """
        מאתחל שכבת שירותים (SqlStore, MarketDataRouter, Engines) וממלא self.services.

        מטרות:
        -------
        - לאפשר ל-root/dashboard.py לגלות capabilities בצורה מקצועית דרך discover_capabilities().
        - לרכז את כל ה-"מנועים" ברמת קרן במקום לפזר לוגיקה בטאבים.
        - Extension points מקצועיים:
            • ניתן לדרוס SqlStore/MarketDataRouter ע"י set_service(...) לפני הקריאה.
            • ניתן להחליף Broker/IBRouter ע"י הצבה ב-self.broker/self.ib_router.
        """
        # מנגנון הגנה מריבוי קריאות
        if getattr(self, "_services_initialized", False):
            return

        # תמיד עובדים עם dict רגיל
        if not isinstance(self.services, dict):
            self.services = {}

        # ---------- 1) SqlStore ----------
        if self.sql_store is None:
            try:
                from core.sql_store import SqlStore  # type: ignore

                cfg = getattr(self.settings, "config", {}) or {}
                cfg_ro = bool(cfg.get("sql_read_only", False))
                env_ro = os.getenv("SQL_STORE_READ_ONLY")
                if env_ro is not None:
                    env_ro_flag = env_ro.strip().lower() in ("1", "true", "yes", "on")
                else:
                    env_ro_flag = False

                if cfg_ro or env_ro_flag:
                    read_only = True
                else:
                    # policy דיפולטית: dev → writable, paper/live → read_only
                    read_only = self.environment in ("paper", "live")

                self.sql_store = SqlStore.from_settings(
                    self.settings,
                    env=self.environment,
                    table_prefix="",
                    read_only=read_only,
                )

                self.services.setdefault("sql_store", self.sql_store)
                logger.info(
                    "SqlStore initialised via from_settings (url=%s, env=%s)",
                    getattr(self.sql_store, "engine_url", None),
                    getattr(self.sql_store, "default_env", None),
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("SqlStore init failed: %s", exc)

        # ---------- 2) MarketDataRouter ----------
        # קודם כל – אם כבר יש md_router (למשל מ-root.smart_scan/backtest) נשתמש בו
        if self.market_data_router is None:
            if self.md_router is not None:
                self.market_data_router = self.md_router
            else:
                # ננסה להביא מה-session_state אם הוגדר
                try:
                    sess_md = st.session_state.get("md_router")
                except Exception:
                    sess_md = None
                if sess_md is not None:
                    self.market_data_router = sess_md
                    self.md_router = sess_md

        # אם עדיין אין – ננסה לבנות Router דיפולטי מ-common.market_data_router
        if self.market_data_router is None:
            try:
                from common.market_data_router import build_default_router  # type: ignore

                router = build_default_router(ib=None, use_yahoo=True)
                self.market_data_router = router
                self.md_router = router
                logger.info(
                    "MarketDataRouter initialised via build_default_router(use_yahoo=True)"
                )
            except Exception as exc:  # pragma: no cover
                logger.debug("MarketDataRouter init skipped/failed: %s", exc)

        if self.market_data_router is not None:
            self.services.setdefault("market_data_router", self.market_data_router)

        # ---------- 3) Macro / Risk / Signals / Fair Value Engines ----------
        # כאן אנחנו חושפים את המודולים עצמם כ-"engine objects" – מספיק לצורך capabilities.

        try:
            import core.macro_engine as macro_engine_mod  # type: ignore

            self.macro_engine = macro_engine_mod
            self.services.setdefault("macro_engine", macro_engine_mod)
        except Exception:
            pass

        try:
            import core.risk_engine as risk_engine_mod  # type: ignore

            self.risk_engine = risk_engine_mod
            self.services.setdefault("risk_engine", risk_engine_mod)
        except Exception:
            pass

        try:
            import core.signals_engine as signals_engine_mod  # type: ignore

            self.signals_engine = signals_engine_mod
            # discover_capabilities מחפש גם שמות אחרים:
            self.services.setdefault("signals_engine", signals_engine_mod)
            self.services.setdefault("signal_engine", signals_engine_mod)
            self.services.setdefault("signals", signals_engine_mod)
            self.services.setdefault("signal_generator", signals_engine_mod)
        except Exception:
            pass

        try:
            import core.fair_value_engine as fair_value_engine_mod  # type: ignore

            self.fair_value_engine = fair_value_engine_mod
            self.services.setdefault("fair_value_engine", fair_value_engine_mod)
        except Exception:
            pass

        # ---------- 3b) Fair Value API config (from config.json + ENV) ----------
        try:
            from core.fair_value_config import FairValueAPIConfig as FVCfg  # type: ignore

            # self.config הוא dict של כל config.json
            fv_cfg = FVCfg.from_settings(self.config or {})
            self.fair_value_api = fv_cfg
            # נרשום גם ב-services למקרה שסוכנים ירצו לגשת אליו
            self.services.setdefault("fair_value_api", fv_cfg)

            logger.info(
                "FairValueAPIConfig initialised: enabled=%s, profile=%s, base_url=%s",
                fv_cfg.enabled,
                fv_cfg.profile,
                fv_cfg.base_url,
            )
        except Exception as exc:
            logger.debug("FairValueAPIConfig init failed: %s", exc)

        # ---------- 4) Agents manager (Placeholder מקצועי) ----------
        # כרגע אין core/agents_manager.py, לכן נגדיר מנהל מינימלי כדי לאפשר את טאב Agents.
        if self.agents_manager is None:
            try:
                self.agents_manager = SimpleNamespace(
                    name="agents_manager_placeholder",
                    online=True,
                    status="online",
                )
                self.services.setdefault("agents_manager", self.agents_manager)
                self.services.setdefault("agents", self.agents_manager)
                self.services.setdefault("ai_agents", self.agents_manager)
            except Exception:
                pass

        # ---------- 5) Broker / IBKR Order Router ----------
        try:
            self.init_ib_router()
        except Exception as exc:  # pragma: no cover
            logger.warning("AppContext.init_ib_router() failed: %s", exc)

        self._services_initialized = True  # type: ignore[attr-defined]

    def init_ib_router(self) -> None:
        """
        אתחול Router למסחר מול IBKR ברמת הקונטקסט.

        - משתמש ב-self.settings (SimpleNamespace) כדי לקרוא env/profile/config.
        - קורא פרופיל (paper/live) מתוך config/ENV.
        - מאתחל IBOrderRouter בצורה בטוחה ולא מפיל את הדשבורד.
        - נשלט ע"י feature flags / config:
            * config["ibkr"]["enabled"]=False → לא נאתחל בכלל.
            * controls["feature_flags_snapshot"]["enable_ib_router"]=False → override.
        """
        if self.ib_router is not None:
            return

        cfg = self.config or {}
        ib_cfg = cfg.get("ibkr") or {}
        enabled: bool = bool(ib_cfg.get("enabled", True))

        # Feature flag override מה-UI / session_state (אם קיים)
        ff = self.controls.get("feature_flags_snapshot", {}) if isinstance(
            self.controls, dict
        ) else {}
        if isinstance(ff, dict) and "enable_ib_router" in ff:
            try:
                enabled = bool(ff.get("enable_ib_router"))
            except Exception:
                pass

        if not enabled:
            logger.info("IBOrderRouter init skipped: disabled via config/feature flag")
            return

        # settings כפי שהגדרת ב-property למטה
        settings_obj = self.settings

        # פרופיל (paper / live) + readonly
        try:
            profile = str(
                getattr(settings_obj, "ib_profile", None)
                or getattr(settings_obj, "profile", None)
                or os.getenv("IB_MODE", "paper")
            ).lower()
        except Exception:
            profile = "paper"

        try:
            readonly = bool(
                getattr(settings_obj, "ib_readonly", None)
                or cfg.get("ib_readonly", False)
            )
        except Exception:
            readonly = False

        try:
            self.ib_router = IBOrderRouter(
                settings=settings_obj,
                use_singleton=True,
                profile=profile,
                readonly=readonly,
            )
            # אם יש connect() על ה-router – ננסה לקרוא אותו ברכות
            if hasattr(self.ib_router, "connect"):
                try:
                    self.ib_router.connect()
                except Exception as exc:  # pragma: no cover
                    logger.warning("IBOrderRouter.connect() failed: %s", exc)

            # נחשוף גם כ-broker לשכבות אחרות (Dashboard / Backtest / Agents)
            self.broker = self.ib_router
            self.services.setdefault("ib_router", self.ib_router)
            self.services.setdefault("broker", self.broker)

            logger.info(
                "IBOrderRouter initialised (profile=%s, readonly=%s)",
                profile,
                readonly,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("IBOrderRouter init failed: %s", exc)
            self.ib_router = None


# ========= High-level AppContext entry point =========


def get_app_context(
    section: str = "dashboard",
    *,
    refresh: bool = False,
    ensure_services: bool = True,
) -> AppContext:
    """
    Preferred, hedge-fund-grade entry point לקבלת AppContext פעיל.

    לוגיקה:
    --------
    1) אם יש st.session_state["app_ctx"] (אובייקט AppContext) ולא refresh:
         → נעדכן section (אם שונה), נאתחל שירותים ונחזיר.
    2) אחרת נשתמש ב-AppContext.get_global() (תמיכה במודולים קיימים),
       נעדכן section, נשמור ב-session_state["app_ctx"], ונחזיר.

    פרמטרים:
    ---------
    section:
        שם הלשונית/תסריט ("dashboard" / "research" / "backtest" / "live_monitor" / ...).
    refresh:
        אם True – מתעלם מקונטקסט קיים ובונה/מביא חדש מ-get_global().
    ensure_services:
        אם True – קורא ctx.init_services() לפני ההחזרה.
    """
    ctx: Optional[AppContext] = None

    if not refresh:
        try:
            sess_ctx = st.session_state.get("app_ctx")  # type: ignore[assignment]
        except Exception:
            sess_ctx = None
        if isinstance(sess_ctx, AppContext):
            ctx = sess_ctx

    if ctx is None:
        ctx = AppContext.get_global()

    if section and ctx.section != section:
        ctx.section = section

    if ensure_services:
        try:
            ctx.init_services()
        except Exception as exc:  # pragma: no cover
            logger.warning("get_app_context: init_services failed: %s", exc)

    try:
        st.session_state["app_ctx"] = ctx
    except Exception:
        # לא נכשלים בסביבה ללא Streamlit
        pass

    return ctx


# ========= Helpers: dirs, hashing, feature flags =========


def ensure_logs_dirs() -> None:
    """דואג שתיקיות logs/snapshots קיימות."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def hash_config(config: JSONDict) -> str:
    """
    hash דטרמיניסטי ל-config + profile + env (לזהות "תצורה").
    """
    import hashlib

    try:
        payload = {
            "config": make_json_safe(config),
            "profile": APP_PROFILE,
            "environment": APP_ENVIRONMENT,
        }
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return ""


def get_feature_flags_snapshot() -> JSONDict:
    """העתק נקי של feature_flags מה-session."""
    try:
        raw = st.session_state.get("feature_flags", {})
    except Exception:
        raw = {}
    return dict(raw) if isinstance(raw, dict) else {}


# ========= Seed & run_id management =========


def set_global_seed(seed: int) -> int:
    """קובע seed גלובלי (random + numpy) ושומר אותו ב-session_state."""
    import random

    seed = int(seed)
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass

    try:
        st.session_state["global_seed"] = seed
    except Exception:
        # אם אין Streamlit context – עדיין הגדרנו random/np
        pass
    return seed


def get_global_seed(default: Optional[int] = None) -> int:
    """
    מחזיר seed גלובלי:
    1) מה-session_state["global_seed"]
    2) מה ENV GLOBAL_SEED אם קיים
    3) מה-arg default
    4) מה _DEF_SEED
    """
    try:
        if "global_seed" in st.session_state:
            try:
                return int(st.session_state["global_seed"])
            except Exception:
                pass
    except Exception:
        # אין Streamlit context
        pass

    env_seed = os.getenv("GLOBAL_SEED")
    if env_seed is not None:
        try:
            return int(env_seed)
        except Exception:
            pass

    if default is not None:
        return int(default)
    return _DEF_SEED


def _generate_run_id(seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"run-{seed}-{ts}"


def ensure_run_id(seed: int) -> str:
    """
    מוודא שיש run_id בסשן. אם אין – יוצר חדש.
    """
    try:
        rid = st.session_state.get("run_id")
    except Exception:
        rid = None
    if not rid:
        rid = _generate_run_id(int(seed))
        try:
            st.session_state["run_id"] = rid
        except Exception:
            pass
    return str(rid)


def get_current_run_id() -> Optional[str]:
    """מחזיר run_id נוכחי מה-session_state (אם קיים)."""
    try:
        rid = st.session_state.get("run_id")
    except Exception:
        rid = None
    return str(rid) if rid else None


# ========= Session helpers =========


def save_ctx_to_session(ctx: AppContext) -> None:
    """שומר ctx כ-dict JSON-safe ב-session_state['ctx']."""
    try:
        st.session_state["ctx"] = ctx.to_dict()
    except Exception:
        pass


def get_current_ctx() -> Optional[AppContext]:
    """
    מנסה לבנות AppContext מתוך session_state['ctx'] (אם קיים).
    הערה:
    - זה מחזיר עותק חדש מה-dict; לקונטקסט ה"חי" בטאב עדיף get_app_context().
    """
    try:
        raw = st.session_state.get("ctx")
    except Exception:
        raw = None
    if not isinstance(raw, dict):
        return None

    try:
        prov_raw = raw.get("provenance")
        scen_raw = raw.get("scenario_overlay")

        provenance = ContextProvenance(**prov_raw) if isinstance(
            prov_raw, dict
        ) else None
        scenario_overlay = ScenarioOverlay(**scen_raw) if isinstance(
            scen_raw, dict
        ) else None

        ctx = AppContext(
            start_date=date.fromisoformat(str(raw["start_date"])),
            end_date=date.fromisoformat(str(raw["end_date"])),
            capital=float(raw["capital"]),
            max_exposure_per_trade=float(raw["max_exposure_per_trade"]),
            max_leverage=float(raw["max_leverage"]),
            pairs=list(raw.get("pairs", [])),
            config=dict(raw.get("config", {})),
            controls=dict(raw.get("controls", {})),
            seed=int(raw["seed"]),
            run_id=str(raw["run_id"]),
            section=str(raw.get("section", "dashboard")),
            profile=str(raw.get("profile", APP_PROFILE)),
            environment=_normalize_env_value(
                str(raw.get("environment", APP_ENVIRONMENT))
            ),
            md_router=None,  # לא משחזרים router דרך dict
            ctx_version=str(raw.get("ctx_version", CURRENT_CTX_SCHEMA_VERSION)),
            config_hash=raw.get("config_hash"),
            parent_run_id=raw.get("parent_run_id"),
            source_section=raw.get("source_section"),
            tags=dict(raw.get("tags", {})),
            health_score=raw.get("health_score"),
            health_flags=list(raw.get("health_flags", [])),
            policy_status=dict(raw.get("policy_status", {})),
            risk_budget=raw.get("risk_budget"),
            scenario_overlay=scenario_overlay,
            provenance=provenance,
            experiment_name=raw.get("experiment_name"),
            experiment_group=raw.get("experiment_group"),
            run_label=raw.get("run_label"),
            reward=raw.get("reward"),
            state_features=raw.get("state_features"),
            context_notes=raw.get("context_notes"),
        )
        return ctx
    except Exception:
        return None


# ========= Auto-tags & basic health scoring =========


def _auto_infer_tags(config: JSONDict) -> TagDict:
    """
    מנסה להסיק tags אוטומטיים מה-config ומה-session:
    - macro_regime מתוך macro_meta / macro_profile
    - matrix_regime מתוך global_insights['matrix']
    - profile/environment
    יוצר גם מפתחות "שטוחים" וגם היררכיים (למשל macro.regime, matrix.crowded_regime).
    """
    tags: TagDict = {}

    # profile / env
    tags["profile"] = APP_PROFILE
    tags["env"] = APP_ENVIRONMENT
    tags["env.type"] = APP_ENVIRONMENT  # namespace לניתוח קל

    # macro_profile מטאב Macro / config
    try:
        macro_prof = st.session_state.get("macro_profile")
    except Exception:
        macro_prof = None
    macro_prof = macro_prof or config.get("macro_profile")
    if macro_prof:
        mp = str(macro_prof)
        tags["macro_profile"] = mp
        tags["macro.profile"] = mp

    # macro_meta (אם טאב Macro הריץ משהו)
    try:
        macro_meta = st.session_state.get("macro_meta") or {}
    except Exception:
        macro_meta = {}
    if isinstance(macro_meta, dict):
        reg = macro_meta.get("regime_label")
        if reg:
            mr = str(reg)
            tags["macro_regime"] = mr
            tags["macro.regime"] = mr

    # matrix insights
    try:
        gi = st.session_state.get("global_insights") or {}
    except Exception:
        gi = {}
    if isinstance(gi, dict):
        mx = gi.get("matrix") or {}
        if isinstance(mx, dict):
            crowd = mx.get("crowded_regime")
            drift = mx.get("drift_label")
            if crowd:
                cr = str(crowd)
                tags["matrix_crowded_regime"] = cr
                tags["matrix.crowded_regime"] = cr
            if drift:
                dl = str(drift)
                tags["matrix_drift_label"] = dl
                tags["matrix.drift_label"] = dl

    return tags


def get_tag_value(tags: TagDict, key: str, default: Any = None) -> Any:
    """
    מחזיר ערך tag מתוך mapping (dict) לפי:
    1. התאמה ישירה (key)
    2. וריאציות '.' <-> '_'
    3. התאמת סיומת (למשל key='regime' יתפוס 'macro.regime')

    מיועד לשימוש פנימי; עבור AppContext עדיף get_ctx_tag.
    """
    if not isinstance(tags, dict):
        return default

    # התאמה ישירה
    if key in tags:
        return tags[key]

    # וריאציות . / _
    alt_keys = {
        key.replace(".", "_"),
        key.replace("_", "."),
    }
    for k in alt_keys:
        if k in tags:
            return tags[k]

    # התאמת סיומת: '...<sep>key'
    for k, v in tags.items():
        if k.endswith("." + key) or k.endswith("_" + key):
            return v

    return default


def get_ctx_tag(ctx: AppContext, key: str, default: Any = None) -> Any:
    """
    Helper נוח לשליפת tag מהקונטקסט (תומך בהיררכיה).
    """
    tags = ctx.tags or {}
    return get_tag_value(tags, key, default)


def set_ctx_tag(ctx: AppContext, key: str, value: Any) -> None:
    """
    Helper נוח להגדרת tag בקונטקסט.
    שומר value כמחרוזת (בהתאם ל-Type alias TagDict = Dict[str, str]).
    """
    if ctx.tags is None:
        ctx.tags = {}
    ctx.tags[str(key)] = str(value)

def _estimate_basic_health_score(
    pairs: List[Any],
    start_date: date,
    end_date: date,
    risk_cfg: JSONDict,
) -> tuple[Optional[float], HealthFlagList]:
    """
    Health בסיסי על הקונטקסט:
    - זוגות ב-universe
    - אורך תקופה
    - האם הון/מינוף נראים סבירים
    """
    flags: HealthFlagList = []
    if not pairs:
        return None, ["no_pairs"]

    days = (end_date - start_date).days
    if days < 30:
        flags.append("short_period")
    elif days > 365 * 5:
        flags.append("very_long_period")

    cap = float(risk_cfg.get("capital", 0.0))
    if cap <= 0:
        flags.append("non_positive_capital")

    max_lev = float(risk_cfg.get("max_leverage", 0.0))
    if max_lev > 5.0:
        flags.append("high_leverage")
    if max_lev < 1.0:
        flags.append("weird_leverage")

    # ציון גס 0–100
    score = 80.0
    if "no_pairs" in flags:
        score = 10.0
    if "short_period" in flags:
        score -= 15.0
    if "very_long_period" in flags:
        score -= 10.0
    if "non_positive_capital" in flags:
        score -= 40.0
    if "high_leverage" in flags:
        score -= 15.0

    score = max(0.0, min(100.0, score))
    return score, flags


# ========= Context history in session =========

def _register_ctx_in_session_history(ctx: AppContext) -> None:
    """
    שומר היסטוריה קצרה של קונטקסטים ב-session_state['_ctx_history'].
    שימושי לניתוח/השוואה מהירה בלי DB חיצוני.
    """
    try:
        hist = st.session_state.get("_ctx_history", [])
        if not isinstance(hist, list):
            hist = []
        hist.append(
            {
                "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "run_id": ctx.run_id,
                "section": ctx.section,
                "profile": ctx.profile,
                "env": ctx.environment,
                "start": str(ctx.start_date),
                "end": str(ctx.end_date),
                "health_score": ctx.health_score,
                "tags": make_json_safe(ctx.tags),
            }
        )
        # נגביל לגודל סביר
        st.session_state["_ctx_history"] = hist[-200:]
    except Exception:
        pass


# ========= build_ctx (base) =========

def build_ctx(
    start_date: date,
    end_date: date,
    controls: JSONDict,
    *,
    section: str = "dashboard",
    config: Optional[JSONDict] = None,
    pairs: Optional[List[Any]] = None,
    tags: Optional[TagDict] = None,
    parent_run_id: Optional[str] = None,
    source_section: Optional[str] = None,
    provenance_notes: Optional[str] = None,
) -> AppContext:
    """
    בונה AppContext מלא מרמת הדשבורד/טאב.
    """
    ensure_logs_dirs()

    cfg = dict(config or {})
    pairs_list = list(pairs or [])

    # seed & run_id
    seed_default = int(cfg.get("seed", _DEF_SEED))
    seed_val = get_global_seed(seed_default)
    set_global_seed(seed_val)
    run_id = ensure_run_id(seed_val)

    # risk params
    risk_cfg = st.session_state.get("risk_capital", {}) or {}
    capital = float(
        controls.get(
            "capital",
            risk_cfg.get("capital", cfg.get("capital", 100_000.0)),
        )
    )
    max_expo = float(
        controls.get(
            "max_exposure_per_trade",
            risk_cfg.get(
                "max_exposure_per_trade",
                cfg.get("max_exposure_per_trade", 0.05),
            ),
        )
    )
    max_lev = float(
        controls.get(
            "max_leverage",
            risk_cfg.get("max_leverage", cfg.get("max_leverage", 2.0)),
        )
    )

    # feature_flags snapshot לתוך controls
    ff_snapshot = get_feature_flags_snapshot()
    controls_full = dict(controls)
    controls_full.setdefault("feature_flags_snapshot", ff_snapshot)

    # hash לקונפיג
    cfg_hash = hash_config(cfg)

    # tags: אוטומטי + custom
    base_tags = _auto_infer_tags(cfg)
    if tags:
        base_tags.update(tags)

    # provenance בסיסי
    provenance = ContextProvenance(
        parent_run_id=parent_run_id,
        created_by="dashboard",
        source_section=source_section or section,
        source_file="root/dashboard.py",
        notes=provenance_notes,
    )

    # health בסיסי
    health_score, health_flags = _estimate_basic_health_score(
        pairs=pairs_list,
        start_date=start_date,
        end_date=end_date,
        risk_cfg={"capital": capital, "max_leverage": max_lev},
    )

    ctx = AppContext(
        start_date=start_date,
        end_date=end_date,
        capital=capital,
        max_exposure_per_trade=max_expo,
        max_leverage=max_lev,
        pairs=pairs_list,
        config=cfg,
        controls=controls_full,
        seed=seed_val,
        run_id=run_id,
        section=section,
        profile=APP_PROFILE,
        environment=APP_ENVIRONMENT,
        md_router=st.session_state.get("md_router"),
        ctx_version=CURRENT_CTX_SCHEMA_VERSION,
        config_hash=cfg_hash,
        parent_run_id=parent_run_id,
        source_section=source_section or section,
        tags=base_tags,
        health_score=health_score,
        health_flags=health_flags,
        policy_status={},          # יתמלא בחלק Policy Engine
        scenario_overlay=None,
        provenance=provenance,
        experiment_name=st.session_state.get("opt_experiment_name"),
        experiment_group=st.session_state.get("experiment_group"),
        run_label=st.session_state.get("bt_run_label") or controls.get("run_label"),
        reward=None,
        state_features=None,
        context_notes=None,
    )

    save_ctx_to_session(ctx)
    _register_ctx_in_session_history(ctx)
    return ctx

# ========= Markdown report =========

def ctx_to_markdown(
    ctx: AppContext,
    kpis: Optional[JSONDict] = None,
    scenario_grid: Optional[JSONDict] = None,
    playbook: Optional[JSONDict] = None,
) -> str:

    """
    דוח Markdown לקונטקסט אחד, ברמת "דוח קרן".
    """
    lines: List[str] = []

    # Header
    lines.append(f"# {ctx.section.title()} Report")
    lines.append("")
    lines.append(f"**Run ID:** `{ctx.run_id}`  |  **Seed:** `{ctx.seed}`")
    lines.append(
        f"**Profile:** `{ctx.profile}`  |  **Env:** `{ctx.environment}`  |  "
        f"**Ctx Version:** `{ctx.ctx_version}`"
    )
    if ctx.config_hash:
        lines.append(f"**Config Hash:** `{ctx.config_hash}`")
    lines.append(
        f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local time)"
    )
    lines.append("")

    # Context core
    lines.append("## Context")
    lines.append(f"- Dates: `{ctx.start_date}` → `{ctx.end_date}`")
    lines.append(f"- Capital: `{ctx.capital:,.0f}`")
    lines.append(f"- Max Exposure / Trade: `{ctx.max_exposure_per_trade}`")
    lines.append(f"- Max Leverage: `{ctx.max_leverage}`")
    lines.append(f"- Pairs Count: `{len(ctx.pairs)}`")
    if ctx.health_score is not None:
        lines.append(f"- Health Score: `{ctx.health_score:.1f}`")
    if ctx.health_flags:
        flags_str = ", ".join(ctx.health_flags)
        lines.append(f"- Health Flags: `{flags_str}`")
    if ctx.tags:
        tags_str = ", ".join(f"{k}={v}" for k, v in ctx.tags.items())
        lines.append(f"- Tags: {tags_str}")
    lines.append("")

    # Policies
    if ctx.policy_status:
        lines.append("## Policy Status")
        for k, v in ctx.policy_status.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    # Risk Budget
    if getattr(ctx, "risk_budget", None):
        lines.append("## Risk Budget")
        rb = make_json_safe(ctx.risk_budget)
        lines.append("```json")
        lines.append(json.dumps(rb, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    # KPIs
    if kpis:
        lines.append("## KPIs")
        clean_kpis = make_json_safe(kpis)
        lines.append("```json")
        lines.append(json.dumps(clean_kpis, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    # Pairs preview (לא הכל, רק טעימה)
    if ctx.pairs:
        max_preview = 10
        lines.append("## Pairs Preview")
        sample = ctx.pairs[:max_preview]
        for p in sample:
            lines.append(f"- `{str(p)}`")
        if len(ctx.pairs) > max_preview:
            lines.append(f"... (+ עוד {len(ctx.pairs) - max_preview} זוגות)")
        lines.append("")

    # Provenance
    if ctx.provenance is not None:
        p = ctx.provenance
        lines.append("## Provenance")
        lines.append(f"- Parent Run ID: `{p.parent_run_id}`")
        lines.append(f"- Created At (UTC): `{p.created_at}`")
        lines.append(f"- Created By: `{p.created_by}`")
        lines.append(f"- Source Section: `{p.source_section}`")
        lines.append(f"- Source File: `{p.source_file}`")
        if p.git_rev:
            lines.append(f"- Git Rev: `{p.git_rev}`")
        if p.notes:
            lines.append(f"- Notes: {p.notes}")
        # Lineage info
        lines.append(f"- Lineage Depth: `{p.lineage_depth}`")
        if p.children_run_ids:
            lines.append(f"- Children Run IDs: {', '.join(p.children_run_ids)}")
        lines.append("")

    # Scenario overlay (אם יש)
    if ctx.scenario_overlay is not None:
        s = ctx.scenario_overlay
        lines.append("## Scenario Overlay")
        lines.append(f"- Name: `{s.name}`")
        lines.append(f"- SPY Shock: `{s.spy_shock_pct:.2%}`")
        if s.vix_target is not None:
            lines.append(f"- VIX Target: `{s.vix_target}`")
        if s.equity_impact is not None:
            lines.append(f"- Equity Impact: `{s.equity_impact:,.0f}`")
        if s.equity_impact_pct is not None:
            lines.append(f"- Equity Impact (%): `{s.equity_impact_pct:.2%}`")
        if s.dd_impact_pct is not None:
            lines.append(f"- DD Impact (%): `{s.dd_impact_pct:.2%}`")
        if s.meta:
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(make_json_safe(s.meta), ensure_ascii=False, indent=2))
            lines.append("```")
        lines.append("")

    # Scenario grid (אם הועבר לגרסה זו של הדוח)
    if scenario_grid:
        lines.append("## Scenario Grid")
        grid_safe = make_json_safe(scenario_grid)
        # אם יש summary טבלאי – נדפיס אותו יפה
        summary = grid_safe.get("summary")
        if isinstance(summary, dict):
            lines.append("### Summary")
            lines.append("```json")
            lines.append(json.dumps(summary, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
        # וגם רשימת תרחישים בסיסית אם קיימת
        scenarios = grid_safe.get("scenarios") or grid_safe.get("rows")
        if isinstance(scenarios, list) and scenarios:
            lines.append("### Scenarios")
            for row in scenarios[:20]:
                # מצפה ל- dict עם name / spy_shock_pct / vix_target / וכו'
                if not isinstance(row, dict):
                    continue
                name = row.get("name", "scenario")
                shock = row.get("spy_shock_pct")
                vix_t = row.get("vix_target")
                impacts = []
                if shock is not None:
                    try:
                        impacts.append(f"SPY={float(shock)*100:.1f}%")
                    except Exception:
                        pass
                if vix_t is not None:
                    impacts.append(f"VIX≈{vix_t}")
                if impacts:
                    suffix = " (" + ", ".join(impacts) + ")"
                else:
                    suffix = ""
                lines.append(f"- `{name}`{suffix}")
            if len(scenarios) > 20:
                lines.append(f"... (+ עוד {len(scenarios) - 20} תרחישים)")
            lines.append("")

    # Experiments / Labels
    if ctx.experiment_name or ctx.experiment_group or ctx.run_label:
        lines.append("## Experiment / Labels")
        if ctx.experiment_name:
            lines.append(f"- Experiment Name: `{ctx.experiment_name}`")
        if ctx.experiment_group:
            lines.append(f"- Experiment Group: `{ctx.experiment_group}`")
        if ctx.run_label:
            lines.append(f"- Run Label: `{ctx.run_label}`")
        lines.append("")

    # Action Playbook (אם יש או ניתן לגזור)
    if playbook is None:
        try:
            playbook = ctx_to_action_playbook(ctx, kpis=kpis)
        except Exception:
            playbook = None

    if playbook and isinstance(playbook, dict) and playbook.get("actions"):
        lines.append("## Action Playbook")
        actions = playbook.get("actions") or []
        for act in actions:
            if not isinstance(act, dict):
                continue
            code = act.get("code", "action")
            text = act.get("text", "")
            prio = act.get("priority", "medium")
            prefix = f"[{prio.upper()}] " if prio else ""
            if text:
                lines.append(f"- {prefix}{text} (`{code}`)")
            else:
                lines.append(f"- {prefix}`{code}`")
        lines.append("")

        summary = playbook.get("summary")
        if summary:
            lines.append("```json")
            lines.append(json.dumps(make_json_safe(summary), ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

    # Notes
    if ctx.context_notes:
        lines.append("## Notes")
        lines.append(ctx.context_notes)
        lines.append("")

    # Machine-readable footer
    lines.append("---")
    lines.append("```json")
    footer = {
        "run_id": ctx.run_id,
        "seed": ctx.seed,
        "section": ctx.section,
        "profile": ctx.profile,
        "environment": ctx.environment,
        "ctx_version": ctx.ctx_version,
        "config_hash": ctx.config_hash,
    }
    lines.append(json.dumps(make_json_safe(footer), ensure_ascii=False, indent=2))
    lines.append("```")

    return "\n".join(lines)


# ========= Snapshots =========

def save_run_snapshot(
    ctx: AppContext,
    *,
    section: Optional[str] = None,
    extra: Optional[JSONDict] = None,
) -> Path:
    """
    שומר Snapshot JSON ל-ctx:
    - ctx.to_dict()
    - feature_flags snapshot
    - policy_status / health
    """
    ensure_logs_dirs()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    sec = section or ctx.section
    fname = f"{sec}-{ctx.run_id}-{ts}.json"
    path = SNAPSHOTS_DIR / fname

    payload: JSONDict = {
        "schema": "AppContextSnapshot",
        "ctx_version": ctx.ctx_version,
        "timestamp": ts,
        "run_id": ctx.run_id,
        "section": sec,
        "profile": ctx.profile,
        "environment": ctx.environment,
        "config_hash": ctx.config_hash,
        "health_score": ctx.health_score,
        "health_flags": ctx.health_flags,
        "policy_status": make_json_safe(ctx.policy_status),
        "ctx": ctx.to_dict(),
        "feature_flags": get_feature_flags_snapshot(),
    }
    if extra:
        payload["extra"] = make_json_safe(extra)

    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # לא מפילים את הדשבורד אם כשל בכתיבה
        pass

    return path


def list_snapshots(limit: int = 100) -> List[JSONDict]:
    """
    מחזיר רשימת snapshots (שם קובץ + מטא), ממויין מהחדש לישן.
    """
    ensure_logs_dirs()
    entries: List[JSONDict] = []
    try:
        files = sorted(
            SNAPSHOTS_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in files[:limit]:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                doc = json.loads(txt)
            except Exception:
                continue
            entries.append(
                {
                    "path": str(p),
                    "name": p.name,
                    "run_id": doc.get("run_id"),
                    "section": doc.get("section"),
                    "timestamp": doc.get("timestamp"),
                    "profile": doc.get("profile"),
                    "environment": doc.get("environment"),
                    "config_hash": doc.get("config_hash"),
                    "health_score": doc.get("health_score"),
                }
            )
    except Exception:
        return []
    return entries


def load_snapshot_as_dict(path_or_name: str) -> Optional[JSONDict]:
    """
    טוען snapshot גולמי (dict) לפי path מלא או שם קובץ.
    """
    ensure_logs_dirs()

    path = Path(path_or_name)
    if not path.is_absolute():
        path = SNAPSHOTS_DIR / path

    if not path.exists():
        return None

    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return json.loads(txt)
    except Exception:
        return None


def load_snapshot_as_ctx(path_or_name: str) -> Optional[AppContext]:
    """
    טוען snapshot והופך אותו ל-AppContext (אם אפשר).
    """
    doc = load_snapshot_as_dict(path_or_name)
    if not isinstance(doc, dict):
        return None

    ctx_raw = doc.get("ctx")
    if not isinstance(ctx_raw, dict):
        return None

    # משתמשים ב-get_current_ctx style אבל על dict חיצוני
    try:
        prov_raw = ctx_raw.get("provenance")
        scen_raw = ctx_raw.get("scenario_overlay")

        provenance = ContextProvenance(**prov_raw) if isinstance(prov_raw, dict) else None
        scenario_overlay = ScenarioOverlay(**scen_raw) if isinstance(scen_raw, dict) else None

        ctx = AppContext(
            start_date=date.fromisoformat(str(ctx_raw["start_date"])),
            end_date=date.fromisoformat(str(ctx_raw["end_date"])),
            capital=float(ctx_raw["capital"]),
            max_exposure_per_trade=float(ctx_raw["max_exposure_per_trade"]),
            max_leverage=float(ctx_raw["max_leverage"]),
            pairs=list(ctx_raw.get("pairs", [])),
            config=dict(ctx_raw.get("config", {})),
            controls=dict(ctx_raw.get("controls", {})),
            seed=int(ctx_raw["seed"]),
            run_id=str(ctx_raw["run_id"]),
            section=str(ctx_raw.get("section", "dashboard")),
            profile=str(ctx_raw.get("profile", APP_PROFILE)),
            environment=_normalize_env_value(
                str(ctx_raw.get("environment", APP_ENVIRONMENT))
            ),
            md_router=None,
            ctx_version=str(ctx_raw.get("ctx_version", CURRENT_CTX_SCHEMA_VERSION)),
            config_hash=ctx_raw.get("config_hash"),
            parent_run_id=ctx_raw.get("parent_run_id"),
            source_section=ctx_raw.get("source_section"),
            tags=dict(ctx_raw.get("tags", {})),
            health_score=ctx_raw.get("health_score"),
            health_flags=list(ctx_raw.get("health_flags", [])),
            policy_status=dict(ctx_raw.get("policy_status", {})),
            risk_budget=ctx_raw.get("risk_budget"),
            scenario_overlay=scenario_overlay,
            provenance=provenance,
            experiment_name=ctx_raw.get("experiment_name"),
            experiment_group=ctx_raw.get("experiment_group"),
            run_label=ctx_raw.get("run_label"),
            reward=ctx_raw.get("reward"),
            state_features=ctx_raw.get("state_features"),
            context_notes=ctx_raw.get("context_notes"),
        )
        return ctx
    except Exception:
        return None


def latest_snapshot_for_run(run_id: str) -> Optional[JSONDict]:
    """
    מחזיר את ה-snapshot האחרון עבור run_id מסוים (אם קיים).
    """
    snaps = list_snapshots(limit=500)
    for s in snaps:
        if s.get("run_id") == run_id:
            return s
    return None


# ========= Context diff =========

def diff_ctx(
    ctx_a: AppContext,
    ctx_b: AppContext,
    *,
    ignore_fields: Optional[List[str]] = None,
) -> JSONDict:
    """
    השוואה בין שני קונטקסטים:
    - changed: mapping field -> (a_value, b_value)
    - only_a: שדות שקיימים רק ב-A
    - only_b: שדות שקיימים רק ב-B
    """
    ignore_fields = set(ignore_fields or [])
    default_ignore = {
        "provenance",
        "md_router",
        "state_features",
        "context_notes",
    }
    ignore_fields |= default_ignore

    a = ctx_a.to_dict()
    b = ctx_b.to_dict()

    keys_a = set(a.keys())
    keys_b = set(b.keys())

    changed: JSONDict = {}
    only_a = sorted(list(keys_a - keys_b))
    only_b = sorted(list(keys_b - keys_a))

    for k in sorted(keys_a & keys_b):
        if k in ignore_fields:
            continue
        va = a.get(k)
        vb = b.get(k)
        if va != vb:
            changed[k] = {"a": va, "b": vb}

    return {
        "changed": changed,
        "only_a": only_a,
        "only_b": only_b,
    }


# ========= History analytics =========

def summarize_ctx_history() -> Optional[pd.DataFrame]:
    """
    מסכם את _ctx_history (אם קיים) ל-DataFrame + מוסיף כמה מדדים:
    - period_days
    - המרה של ts ל-datetime אם קיים
    - health_score rolling mean/delta (לניתוח drift)
    """
    hist = st.session_state.get("_ctx_history")
    if not isinstance(hist, list) or not hist:
        return None

    try:
        df = pd.DataFrame(hist)
        if "health_score" in df.columns:
            df["health_score"] = pd.to_numeric(df["health_score"], errors="coerce")
        if "start" in df.columns:
            df["start"] = pd.to_datetime(df["start"], errors="coerce")
        if "end" in df.columns:
            df["end"] = pd.to_datetime(df["end"], errors="coerce")
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")

        # מדדים בסיסיים לכל section/profile/env
        df["period_days"] = (df["end"] - df["start"]).dt.days

        # מיון כרונולוגי לפי ts אם קיים, אחרת לפי start
        if "ts" in df.columns and df["ts"].notna().any():
            df = df.sort_values("ts")
        elif "start" in df.columns and df["start"].notna().any():
            df = df.sort_values("start")

        # Rolling drift על health_score (חלון 10 כברירת מחדל)
        if "health_score" in df.columns:
            df["health_score_rolling_mean_10"] = (
                df["health_score"]
                .rolling(window=10, min_periods=3)
                .mean()
            )
            df["health_score_delta"] = df["health_score"].diff()

        return df
    except Exception:
        return None

def detect_ctx_drift(
    df_hist: pd.DataFrame,
    *,
    min_points: int = 5,
    lookback: int = 20,
    dropna: bool = True,
) -> JSONDict:
    """
    מזהה drift בסיסי ב-health_score מתוך היסטוריית קונטקסטים.

    החזרה:
    -------
    {
        "has_drift": bool,
        "direction": "up" | "down" | "stable",
        "health_score_last": float | None,
        "health_score_prev_mean": float | None,
        "health_score_delta": float | None,
        "n_points": int,
    }
    """
    if df_hist is None or df_hist.empty:
        return {
            "has_drift": False,
            "direction": "stable",
            "health_score_last": None,
            "health_score_prev_mean": None,
            "health_score_delta": None,
            "n_points": 0,
        }

    df = df_hist.copy()
    if "health_score" not in df.columns:
        return {
            "has_drift": False,
            "direction": "stable",
            "health_score_last": None,
            "health_score_prev_mean": None,
            "health_score_delta": None,
            "n_points": int(len(df)),
        }

    hs = pd.to_numeric(df["health_score"], errors="coerce")
    if dropna:
        hs = hs.dropna()

    n = int(hs.shape[0])
    if n < min_points:
        return {
            "has_drift": False,
            "direction": "stable",
            "health_score_last": float(hs.iloc[-1]) if n > 0 else None,
            "health_score_prev_mean": None,
            "health_score_delta": None,
            "n_points": n,
        }

    # נסתכל על lookback אחרון
    tail = hs.tail(lookback)
    last = float(tail.iloc[-1])
    prev = tail.iloc[:-1]
    prev_mean = float(prev.mean()) if not prev.empty else last
    delta = last - prev_mean

    # סף פשוט לדריפט "משמעותי"
    threshold = 5.0  # נקודות health
    if abs(delta) < threshold:
        direction = "stable"
        has_drift = False
    else:
        direction = "up" if delta > 0 else "down"
        has_drift = True

    return {
        "has_drift": has_drift,
        "direction": direction,
        "health_score_last": last,
        "health_score_prev_mean": prev_mean,
        "health_score_delta": delta,
        "n_points": n,
    }

# ========= Policy & Health Engine (Risk / Data / Research, HF-grade) =========

def _merge_health_flags(existing: HealthFlagList, new_flags: HealthFlagList) -> HealthFlagList:
    """מאחד רשימות flags בלי כפילויות, שומר סדר."""
    seen: set[str] = set()
    out: HealthFlagList = []
    for f in list(existing) + list(new_flags):
        if not f:
            continue
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _score_from_flags(base: float, flags: List[str], penalties: Dict[str, float]) -> float:
    """
    מקבל ציון בסיס ודגלים → מחזיר ציון לאחר קנסות לפי penalties.
    penalties: dict[flag_prefix → penalty_value]
    """
    score = float(base)
    for fl in flags:
        for prefix, pen in penalties.items():
            if fl.startswith(prefix):
                score -= pen
    return max(0.0, min(100.0, score))


# ======== Risk policy ========

def evaluate_risk_policy(
    *,
    daily_pnl_pct: Optional[float] = None,
    max_dd_pct: Optional[float] = None,
    gross_leverage: Optional[float] = None,
    vix: Optional[float] = None,
    limits: Optional[JSONDict] = None,
) -> JSONDict:
    """
    הערכת Policy סיכון:
    מחזיר:
    - risk_policy_ok: bool
    - risk_policy_score: 0–100 (100 = מצוין)
    - risk_policy_flags: רשימת דגלים
    - risk_policy_triggered_rules: רשימת טקסטים קריאים
    """
    limits = dict(limits or {})
    flags: List[str] = []
    rules: List[str] = []

    max_daily_loss = float(limits.get("max_daily_loss_pct", 0.03))
    max_drawdown = float(limits.get("max_drawdown_pct", 0.20))
    max_lev = float(limits.get("max_gross_leverage", 3.0))
    vix_thr = float(limits.get("vix_kill_threshold", 35.0))

    # Daily loss
    if daily_pnl_pct is not None:
        try:
            x = float(daily_pnl_pct)
            if x <= -max_daily_loss:
                flags.append("risk_daily_loss_breach")
                rules.append(
                    f"daily_loss {x*100:.2f}% ≤ -{max_daily_loss*100:.2f}%"
                )
            elif x < 0:
                flags.append("risk_daily_loss_negative")
        except Exception:
            pass

    # Drawdown
    if max_dd_pct is not None:
        try:
            d = float(max_dd_pct)
            if d <= -max_drawdown:
                flags.append("risk_drawdown_breach")
                rules.append(
                    f"max_dd {d*100:.2f}% ≤ -{max_drawdown*100:.2f}%"
                )
            elif d < 0:
                flags.append("risk_drawdown_present")
        except Exception:
            pass

    # Leverage
    if gross_leverage is not None:
        try:
            lev = float(gross_leverage)
            if lev > max_lev:
                flags.append("risk_leverage_breach")
                rules.append(
                    f"leverage {lev:.2f}x > {max_lev:.2f}x"
                )
            elif lev > max_lev * 0.8:
                flags.append("risk_leverage_high")
        except Exception:
            pass

    # VIX
    if vix is not None:
        try:
            vx = float(vix)
            if vx >= vix_thr:
                flags.append("risk_vix_breach")
                rules.append(
                    f"vix {vx:.2f} ≥ {vix_thr:.2f}"
                )
            elif vx >= 25:
                flags.append("risk_vix_elevated")
        except Exception:
            pass

    # ציון בסיס + קנסות
    base_score = 90.0
    penalties = {
        "risk_daily_loss_breach": 25.0,
        "risk_drawdown_breach": 25.0,
        "risk_leverage_breach": 20.0,
        "risk_vix_breach": 20.0,
        "risk_leverage_high": 10.0,
        "risk_vix_elevated": 8.0,
        "risk_daily_loss_negative": 3.0,
        "risk_drawdown_present": 3.0,
    }
    score = _score_from_flags(base_score, flags, penalties)

    ok = score >= 60.0 and not any(fl.endswith("breach") for fl in flags)
    return {
        "risk_policy_ok": ok,
        "risk_policy_score": score,
        "risk_policy_flags": flags,
        "risk_policy_triggered_rules": rules,
    }

def compute_risk_budget(
    *,
    daily_pnl_pct: Optional[float] = None,
    max_dd_pct: Optional[float] = None,
    gross_leverage: Optional[float] = None,
    vix: Optional[float] = None,
    limits: Optional[JSONDict] = None,
) -> JSONDict:
    """
    מחשב תקציב ריסק נוכחי יחסית לגבולות:
    מחזיר:
    - utilization_daily_loss: שימוש בתקציב הפסד יומי (0–1)
    - utilization_drawdown: שימוש בתקציב DD (0–1)
    - utilization_leverage: שימוש בתקציב מינוף (0–1)
    - utilization_vix: שימוש בתקציב וולא/מצב שוק (0–1)
    - overall_utilization: max מכל אלה (0–1)
    """
    limits = dict(limits or {})

    max_daily_loss = float(limits.get("max_daily_loss_pct", 0.03))
    max_drawdown = float(limits.get("max_drawdown_pct", 0.20))
    max_lev = float(limits.get("max_gross_leverage", 3.0))
    vix_thr = float(limits.get("vix_kill_threshold", 35.0))

    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    # Daily loss utilization (רק אם שלילי)
    util_daily = 0.0
    if daily_pnl_pct is not None and max_daily_loss > 0:
        try:
            x = float(daily_pnl_pct)
            if x < 0:
                util_daily = _clip01(abs(x) / max_daily_loss)
        except Exception:
            pass

    # Drawdown utilization (יחסי למקסימום DD)
    util_dd = 0.0
    if max_dd_pct is not None and max_drawdown > 0:
        try:
            d = float(max_dd_pct)
            if d < 0:
                util_dd = _clip01(abs(d) / max_drawdown)
        except Exception:
            pass

    # Leverage utilization
    util_lev = 0.0
    if gross_leverage is not None and max_lev > 0:
        try:
            lev = float(gross_leverage)
            util_lev = _clip01(lev / max_lev)
        except Exception:
            pass

    # VIX utilization
    util_vix = 0.0
    if vix is not None and vix_thr > 0:
        try:
            vx = float(vix)
            util_vix = _clip01(vx / vix_thr)
        except Exception:
            pass

    overall = max(util_daily, util_dd, util_lev, util_vix)

    return {
        "utilization_daily_loss": util_daily,
        "utilization_drawdown": util_dd,
        "utilization_leverage": util_lev,
        "utilization_vix": util_vix,
        "overall_utilization": overall,
        "limits": {
            "max_daily_loss_pct": max_daily_loss,
            "max_drawdown_pct": max_drawdown,
            "max_gross_leverage": max_lev,
            "vix_kill_threshold": vix_thr,
        },
    }

# ======== Data policy ========

def evaluate_data_policy(
    *,
    data_quality: Optional[JSONDict] = None,
) -> JSONDict:
    """
    הערכת Policy דאטה:
    data_quality יכול לכלול:
    - coverage_ratio
    - nan_fraction
    - missing_days_est
    - overlap_ratio
    """
    dq = dict(data_quality or {})
    flags: List[str] = []

    # coverage
    cov = dq.get("coverage_ratio")
    if cov is not None:
        try:
            c = float(cov)
            if c < 0.5:
                flags.append("data_coverage_very_low")
            elif c < 0.7:
                flags.append("data_coverage_low")
        except Exception:
            pass

    # NaNs
    nan_frac = dq.get("nan_fraction")
    if nan_frac is not None:
        try:
            n = float(nan_frac)
            if n > 0.10:
                flags.append("data_many_nans")
            elif n > 0.02:
                flags.append("data_some_nans")
        except Exception:
            pass

    # overlap (לזוגות)
    overlap = dq.get("overlap_ratio")
    if overlap is not None:
        try:
            o = float(overlap)
            if o < 0.5:
                flags.append("data_overlap_very_low")
            elif o < 0.7:
                flags.append("data_overlap_low")
        except Exception:
            pass

    # missing days
    missing = dq.get("missing_days_est")
    if missing is not None:
        try:
            m = int(missing)
            if m > 200:
                flags.append("data_many_missing_days")
            elif m > 50:
                flags.append("data_missing_days")
        except Exception:
            pass

    base_score = 95.0
    penalties = {
        "data_coverage_very_low": 35.0,
        "data_coverage_low": 20.0,
        "data_many_nans": 30.0,
        "data_some_nans": 10.0,
        "data_overlap_very_low": 25.0,
        "data_overlap_low": 15.0,
        "data_many_missing_days": 25.0,
        "data_missing_days": 10.0,
    }
    score = _score_from_flags(base_score, flags, penalties)
    ok = score >= 70.0 and not any(fl.startswith("data_coverage_very_low") for fl in flags)

    return {
        "data_policy_ok": ok,
        "data_policy_score": score,
        "data_policy_flags": flags,
    }


# ======== Research policy ========

def evaluate_research_policy(
    *,
    num_trades: Optional[int] = None,
    period_days: Optional[int] = None,
    num_pairs: Optional[int] = None,
) -> JSONDict:
    """
    הערכת Policy מחקר:
    - num_trades: כמות טריידים
    - period_days: אורך התקופה
    - num_pairs: מספר זוגות/נכסים
    """
    flags: List[str] = []

    if num_trades is not None:
        try:
            n = int(num_trades)
            if n < 30:
                flags.append("research_few_trades")
            elif n < 100:
                flags.append("research_low_sample")
        except Exception:
            pass

    if period_days is not None:
        try:
            d = int(period_days)
            if d < 60:
                flags.append("research_short_period")
            elif d < 180:
                flags.append("research_mid_period")
        except Exception:
            pass

    if num_pairs is not None:
        try:
            k = int(num_pairs)
            if k < 3:
                flags.append("research_very_few_pairs")
            elif k < 8:
                flags.append("research_few_pairs")
        except Exception:
            pass

    base_score = 85.0
    penalties = {
        "research_few_trades": 30.0,
        "research_low_sample": 15.0,
        "research_short_period": 20.0,
        "research_mid_period": 5.0,
        "research_very_few_pairs": 25.0,
        "research_few_pairs": 10.0,
    }
    score = _score_from_flags(base_score, flags, penalties)
    ok = score >= 65.0 and not any(fl.startswith("research_very_few_pairs") for fl in flags)

    return {
        "research_policy_ok": ok,
        "research_policy_score": score,
        "research_policy_flags": flags,
    }

def _should_auto_snapshot(ctx: AppContext) -> bool:
    """
    מחליט האם לבצע snapshot אוטומטי אחרי apply_policies_to_ctx.

    לוגיקה שמרנית:
    --------------
    - רק ב-env 'paper' או 'live'.
    - חייב להיות health_score.
    - נבצע snapshot אם:
        • health_score < 60, או
        • risk_policy_ok / data_policy_ok / research_policy_ok = False, או
        • utilization_vix / utilization_leverage / overall_utilization גבוהים מאוד.
    """
    if ctx.environment not in ("paper", "live"):
        return False
    if ctx.health_score is None:
        return False

    ps = ctx.policy_status or {}
    if not ps:
        return False

    # פוליסי בסיסי
    risk_ok = bool(ps.get("risk_policy_ok", True))
    data_ok = bool(ps.get("data_policy_ok", True))
    research_ok = bool(ps.get("research_policy_ok", True))

    if not (risk_ok and data_ok and research_ok):
        return True

    # Health נמוך
    if float(ctx.health_score) < 60.0:
        return True

    # תקציב ריסק כמעט מלא
    rb = getattr(ctx, "risk_budget", None) or ps.get("risk_budget")
    if isinstance(rb, dict):
        try:
            overall = float(rb.get("overall_utilization", 0.0))
            util_vix = float(rb.get("utilization_vix", 0.0))
            util_lev = float(rb.get("utilization_leverage", 0.0))
            if overall > 0.90 or util_vix > 0.90 or util_lev > 0.90:
                return True
        except Exception:
            pass

    return False

# ========= Apply policies & recompute health / history =========

def apply_policies_to_ctx(
    ctx: AppContext,
    *,
    risk_state: Optional[JSONDict] = None,
    risk_limits: Optional[JSONDict] = None,
    data_quality: Optional[JSONDict] = None,
    research_stats: Optional[JSONDict] = None,
) -> AppContext:
    """
    מעדכן:
    - ctx.policy_status
    - ctx.health_score
    - ctx.health_flags
    לפי:
    - risk_state (PnL/DD/leverage/VIX)
    - data_quality (כמו Data Quality panel)
    - research_stats (כמו Backtest history)

    שדרוג HF-grade:
    ----------------
    - משקלול health/risk/data/research עקבי.
    - קריאת _should_auto_snapshot(ctx) לאחר עדכון health_score.
      אם True וב-env 'paper'/'live' → ירשום snapshot אוטומטי.
    """
    policy_status = dict(ctx.policy_status or {})
    base_health = ctx.health_score if ctx.health_score is not None else 80.0
    flags_acc: HealthFlagList = list(ctx.health_flags or [])

    # --- Risk policy ---
    if risk_state is not None:
        rp = evaluate_risk_policy(
            daily_pnl_pct=risk_state.get("daily_pnl_pct"),
            max_dd_pct=risk_state.get("max_dd_pct"),
            gross_leverage=risk_state.get("gross_leverage"),
            vix=risk_state.get("vix"),
            limits=risk_limits,
        )
        policy_status.update(rp)
        flags_acc = _merge_health_flags(flags_acc, rp.get("risk_policy_flags", []))

    # --- Data policy ---
    if data_quality is not None:
        dp = evaluate_data_policy(data_quality=data_quality)
        policy_status.update(dp)
        flags_acc = _merge_health_flags(flags_acc, dp.get("data_policy_flags", []))

    # --- Research policy ---
    if research_stats is not None:
        rp2 = evaluate_research_policy(
            num_trades=research_stats.get("num_trades"),
            period_days=research_stats.get("period_days"),
            num_pairs=research_stats.get("num_pairs"),
        )
        policy_status.update(rp2)
        flags_acc = _merge_health_flags(flags_acc, rp2.get("research_policy_flags", []))

    # --- Risk budget (לפי risk_state/limits) ---
    if risk_state is not None or risk_limits is not None:
        try:
            rb = compute_risk_budget(
                daily_pnl_pct=(risk_state or {}).get("daily_pnl_pct") if risk_state else None,
                max_dd_pct=(risk_state or {}).get("max_dd_pct") if risk_state else None,
                gross_leverage=(risk_state or {}).get("gross_leverage") if risk_state else None,
                vix=(risk_state or {}).get("vix") if risk_state else None,
                limits=risk_limits,
            )
            policy_status["risk_budget"] = rb
            ctx.risk_budget = rb
        except Exception:
            # אל ניפול בגלל חישוב תקציב
            pass

    # --- Context drift (ברמת health_score לאורך זמן) ---
    try:
        df_hist = summarize_ctx_history()
        if df_hist is not None and not df_hist.empty:
            drift_info = detect_ctx_drift(df_hist)
            policy_status["ctx_drift"] = drift_info
            if drift_info.get("has_drift"):
                direction = drift_info.get("direction")
                if direction == "down":
                    flags_acc = _merge_health_flags(flags_acc, ["ctx_health_drift_down"])
                    # קנס קטן לבסיס הבריאות
                    base_health = max(0.0, float(base_health) - 5.0)
                elif direction == "up":
                    flags_acc = _merge_health_flags(flags_acc, ["ctx_health_drift_up"])
                    # בונוס קטן לעלייה יציבה
                    base_health = min(100.0, float(base_health) + 2.0)
    except Exception:
        # לא נכשלים בגלל drift monitor
        pass

    # ציון health חדש – משקלול של base + שלושת ה-scores אם קיימים
    risk_score = float(policy_status.get("risk_policy_score", 80.0))
    data_score = float(policy_status.get("data_policy_score", 90.0))
    research_score = float(policy_status.get("research_policy_score", 80.0))

    # אפשר לשנות משקלים – כרגע:
    # risk 40%, data 30%, research 30%, משולב עם base_health
    combined = (
        0.25 * float(base_health)
        + 0.40 * risk_score
        + 0.30 * data_score
        + 0.30 * research_score
    ) / 1.25
    combined = max(0.0, min(100.0, combined))

    ctx.policy_status = policy_status
    ctx.health_score = combined
    ctx.health_flags = flags_acc

    save_ctx_to_session(ctx)
    _register_ctx_in_session_history(ctx)

    # Snapshot אוטומטי ב-paper/live כאשר יש אינדיקציות משמעותיות
    try:
        if _should_auto_snapshot(ctx):
            ctx.save_snapshot(
                extra={
                    "policy_status": make_json_safe(ctx.policy_status),
                    "auto_snapshot_reason": "apply_policies_to_ctx",
                }
            )
    except Exception:
        # לא מפילים את המערכת בגלל snapshot אוטומטי
        pass

    return ctx


# ========= Live readiness =========

def compute_live_readiness(ctx: AppContext) -> JSONDict:
    """
    חיווי live_ready מהקונטקסט:
    - health_score
    - מדיניות (risk/data/research)
    - גודל universe
    """
    ps = ctx.policy_status or {}
    flags = ctx.health_flags or []

    risk_ok = bool(ps.get("risk_policy_ok", True))
    data_ok = bool(ps.get("data_policy_ok", True))
    research_ok = bool(ps.get("research_policy_ok", True))

    score = float(ctx.health_score or 0.0)
    n_pairs = len(ctx.pairs)

    # קריטריון בסיסי ללייב
    ready = (
        risk_ok
        and data_ok
        and research_ok
        and score >= 70.0
        and n_pairs >= 5
    )

    reasons: List[str] = []
    if not risk_ok:
        reasons.append("risk_policy_not_ok")
    if not data_ok:
        reasons.append("data_policy_not_ok")
    if not research_ok:
        reasons.append("research_policy_not_ok")
    if score < 70.0:
        reasons.append(f"health_score_too_low({score:.1f})")
    if n_pairs < 5:
        reasons.append("too_few_pairs")

    # תקציב ריסק – אם כמעט אזל, נסמן
    rb = getattr(ctx, "risk_budget", None) or (ctx.policy_status or {}).get("risk_budget")
    if isinstance(rb, dict):
        try:
            util = float(rb.get("overall_utilization", 0.0))
            if util > 0.95:
                ready = False
                reasons.append(f"risk_budget_exhausted({util:.2f})")
            elif util > 0.85:
                reasons.append(f"risk_budget_high({util:.2f})")
        except Exception:
            pass

    return {
        "live_ready": ready,
        "health_score": score,
        "risk_policy_ok": risk_ok,
        "data_policy_ok": data_ok,
        "research_policy_ok": research_ok,
        "health_flags": list(flags),
        "reasons": reasons,
    }

# ========= Scenario grid (stress / what-if) =========

def run_scenario_grid(
    ctx: AppContext,
    *,
    spy_shocks: Sequence[float],
    vix_targets: Sequence[float],
    base_kpis: Optional[JSONDict] = None,
    scenario_evaluator: Optional[
        Callable[[float, float, AppContext, Optional[JSONDict]], JSONDict]
    ] = None,
) -> ScenarioGridResult:
    """
    מריץ גריד תרחישים על אותו קונטקסט.

    פרמטרים:
    ----------
    spy_shocks:
        רשימת זעזועים על SPY (למשל [-0.05, -0.10, -0.20]).
    vix_targets:
        רמות VIX יעד (למשל [25, 35, 45]).
    base_kpis:
        KPIs בסיסיים של הריצה הנוכחית (PnL/Sharpe/DD וכו').
    scenario_evaluator:
        פונקציה אופציונלית שמקבלת:
            (spy_shock_pct, vix_target, ctx, base_kpis)
        ומחזירה dict עם:
            {
              "equity_impact": float | None,
              "equity_impact_pct": float | None,
              "dd_impact_pct": float | None,
              "extra_meta": dict | None,
            }
        אם לא סופק – נבצע חישוב “טכני” פשוט/נייטרלי.

    החזרה:
    -------
    ScenarioGridResult
        כולל:
        - רשימת ScenarioOverlay
        - summary: טבלת ציר בסיסית לפי shock / vix
    """
    scenarios: List[ScenarioOverlay] = []
    base_kpis = dict(base_kpis or {})

    # ערכי ברירת מחדל אם אין evaluator – לא נשבור, אלא ניתן מבנה בלבד
    total_pnl = base_kpis.get("total_pnl")
    max_dd = base_kpis.get("max_dd")

    for shock in spy_shocks:
        for vix_t in vix_targets:
            name = f"SPY {shock*100:.1f}% / VIX≈{vix_t}"
            eq_impact = None
            eq_impact_pct = None
            dd_impact_pct = None
            extra_meta: JSONDict = {}

            if scenario_evaluator is not None:
                try:
                    res = scenario_evaluator(float(shock), float(vix_t), ctx, base_kpis)
                    if isinstance(res, dict):
                        eq_impact = res.get("equity_impact")
                        eq_impact_pct = res.get("equity_impact_pct")
                        dd_impact_pct = res.get("dd_impact_pct")
                        extra_meta = dict(res.get("extra_meta", {}))
                except Exception:
                    # לא מפילים בגלל תרחיש אחד
                    pass
            else:
                # fallback היפותטי מאוד בסיסי:
                # - אם יש total_pnl/dd – נעשה התאמה לינארית גסה.
                try:
                    if total_pnl is not None:
                        eq_impact = float(total_pnl) * (1.0 + float(shock))
                        eq_impact_pct = float(shock)
                except Exception:
                    pass
                try:
                    if max_dd is not None:
                        dd_impact_pct = float(max_dd) * (1.0 - 0.5 * abs(float(shock)))
                except Exception:
                    pass

            overlay = ScenarioOverlay(
                name=name,
                spy_shock_pct=float(shock),
                vix_target=float(vix_t),
                equity_impact=eq_impact,
                equity_impact_pct=eq_impact_pct,
                dd_impact_pct=dd_impact_pct,
                meta={
                    "base_kpis": make_json_safe(base_kpis),
                    "extra": make_json_safe(extra_meta),
                },
            )
            scenarios.append(overlay)

    # סיכום פשוט: ממוצע השפעה לכל shock / vix
    rows: List[JSONDict] = []
    for s in scenarios:
        rows.append(
            {
                "name": s.name,
                "spy_shock_pct": s.spy_shock_pct,
                "vix_target": s.vix_target,
                "equity_impact": s.equity_impact,
                "equity_impact_pct": s.equity_impact_pct,
                "dd_impact_pct": s.dd_impact_pct,
            }
        )

    try:
        df = pd.DataFrame(rows)
        summary: JSONDict = {}
        if not df.empty:
            # pivot לפי shock ו-vix על equity_impact_pct
            try:
                piv = (
                    df.pivot_table(
                        index="spy_shock_pct",
                        columns="vix_target",
                        values="equity_impact_pct",
                        aggfunc="mean",
                    )
                    .sort_index()
                    .sort_index(axis=1)
                )
                summary["equity_impact_pct_pivot"] = make_json_safe(
                    piv.to_dict()
                )
            except Exception:
                pass

            # סטטיסטיקות כלליות
            summary["stats"] = {
                "n_scenarios": int(len(df)),
                "equity_impact_pct_min": float(df["equity_impact_pct"].min(skipna=True))
                if "equity_impact_pct" in df.columns
                else None,
                "equity_impact_pct_max": float(df["equity_impact_pct"].max(skipna=True))
                if "equity_impact_pct" in df.columns
                else None,
            }
        else:
            summary["stats"] = {"n_scenarios": 0}
    except Exception:
        summary = {"stats": {"n_scenarios": len(scenarios)}}

    return ScenarioGridResult(
        scenarios=scenarios,
        summary=summary,
    )

# ========= Context analytics & feature vectors =========

def ctx_to_feature_vector(ctx: AppContext, *, kpis: Optional[JSONDict] = None) -> JSONDict:
    """
    מייצר וקטור פיצ'רים מ-AppContext + KPIs (אם יש).
    מיועד ל:
    - דירוג ניסויים / קונפיגים.
    - מודלי ML/RL.
    """
    days = (ctx.end_date - ctx.start_date).days
    feat: JSONDict = {
        # בסיס
        "period_days": float(days),
        "capital": float(ctx.capital),
        "max_exposure_per_trade": float(ctx.max_exposure_per_trade),
        "max_leverage": float(ctx.max_leverage),
        "pairs_count": float(len(ctx.pairs)),
        "health_score": float(ctx.health_score or 0.0),
        # env (מורחב ל-research, תוך שמירה על dev/paper/live)
        "env_dev": 1.0 if ctx.environment == "dev" else 0.0,
        "env_research": 1.0 if ctx.environment == "research" else 0.0,
        "env_paper": 1.0 if ctx.environment == "paper" else 0.0,
        "env_live": 1.0 if ctx.environment == "live" else 0.0,
        # section
        f"section_{ctx.section}": 1.0,
        # profile
        f"profile_{ctx.profile}": 1.0,
    }


    # tags נפוצים כאינדיקטורים (כולל היררכיים)
    macro_regime = (
        get_ctx_tag(ctx, "macro.regime")
        or get_ctx_tag(ctx, "macro_regime")
        or get_ctx_tag(ctx, "macro_profile")
    )
    if macro_regime:
        mr = str(macro_regime).lower()
        for r in ("risk_on", "risk-off", "risk_off", "crisis", "neutral"):
            key = f"tag_macro_{r.replace('-', '_')}"
            feat[key] = 1.0 if r in mr else 0.0

    matrix_regime = (
        get_ctx_tag(ctx, "matrix.crowded_regime")
        or get_ctx_tag(ctx, "matrix_crowded_regime")
    )
    if matrix_regime:
        mr2 = str(matrix_regime).lower()
        for r in ("highly_crowded", "crowded", "fragmented", "neutral"):
            key = f"tag_matrix_{r}"
            feat[key] = 1.0 if r.replace("_", " ") in mr2 or r in mr2 else 0.0

    drift_label = (
        get_ctx_tag(ctx, "matrix.drift_label")
        or get_ctx_tag(ctx, "matrix_drift_label")
    )
    if drift_label:
        dl = str(drift_label).lower()
        for r in ("stable", "transition", "drifting"):
            key = f"tag_drift_{r}"
            feat[key] = 1.0 if r in dl else 0.0


    # policy scores
    ps = ctx.policy_status or {}
    for name in ("risk_policy_score", "data_policy_score", "research_policy_score"):
        try:
            if name in ps:
                feat[name] = float(ps[name])
        except Exception:
            feat[name] = None

    # KPIs (אם נשלחו)
    if kpis:
        for key, val in kpis.items():
            k = f"kpi_{key}"
            try:
                feat[k] = float(val)
            except Exception:
                # לא מספרי – נשאיר מחוץ לפיצ'רים
                continue

    return make_json_safe(feat)


def compute_context_reward(
    *,
    total_pnl: Optional[float] = None,
    sharpe: Optional[float] = None,
    max_dd: Optional[float] = None,
    tail_risk_score: Optional[float] = None,
    stability_score: Optional[float] = None,
) -> float:
    """
    פונקציית reward לדרוג קונטקסט:
    - Sharpe: driver ראשי
    - PnL: בונוס קטן
    - MaxDD / tail_risk_score: קנס
    - stability_score: בונוס קטן ליציבות (למשל WF stability)
    """
    r = 0.0

    if sharpe is not None:
        try:
            s = float(sharpe)
            r += 3.0 * s
        except Exception:
            pass

    if total_pnl is not None:
        try:
            pnl = float(total_pnl)
            r += 0.0001 * pnl
        except Exception:
            pass

    if max_dd is not None:
        try:
            dd = float(max_dd)
            if dd < 0:
                r += dd  # קנס שלילי – מחמיר ככול שה-DD עמוק יותר
        except Exception:
            pass

    if tail_risk_score is not None:
        try:
            tr = float(tail_risk_score)
            r -= 0.5 * max(tr, 0.0)
        except Exception:
            pass

    if stability_score is not None:
        try:
            stbl = float(stability_score)
            r += 0.5 * stbl
        except Exception:
            pass

    return float(r)

def compute_meta_score(
    ctx: AppContext,
    *,
    kpis: Optional[JSONDict] = None,
    weights: Optional[JSONDict] = None,
) -> float:
    """
    ציון Meta כללי לקונטקסט:
    משלב:
    - health_score
    - reward (לפי compute_context_reward)
    - risk/data/research policy scores
    - KPIs (Sharpe, PnL, stability)
    ומחזיר ציון 0–100.

    weights יכול לכלול:
    - health
    - reward
    - risk
    - data
    - research
    """
    kpis = dict(kpis or {})
    ps = ctx.policy_status or {}

    default_weights = {
        "health": 0.4,
        "reward": 0.3,
        "risk": 0.1,
        "data": 0.1,
        "research": 0.1,
    }
    w = dict(default_weights)
    if weights:
        w.update(weights)

    # health
    health = float(ctx.health_score or 0.0)

    # policy scores (אם אין – נשתמש ב-health)
    risk_score = float(ps.get("risk_policy_score", health))
    data_score = float(ps.get("data_policy_score", health))
    research_score = float(ps.get("research_policy_score", health))

    # KPIs בסיסיים
    sharpe = kpis.get("sharpe")
    total_pnl = kpis.get("total_pnl")
    max_dd = kpis.get("max_dd")
    tail_risk_score = kpis.get("tail_risk_score")
    stability_score = kpis.get("wf_stability_score")

    # reward לפי אותה פונקציה קיימת
    base_reward = compute_context_reward(
        total_pnl=total_pnl,
        sharpe=sharpe,
        max_dd=max_dd,
        tail_risk_score=tail_risk_score,
        stability_score=stability_score,
    )

    # ננרמל הכל ל-0–1 פחות או יותר:
    h_norm = health / 100.0
    # reward יכול להיות חיובי/שלילי – נעשה squash גס
    r_norm = max(-1.0, min(1.0, base_reward / 10.0))
    risk_norm = risk_score / 100.0
    data_norm = data_score / 100.0
    research_norm = research_score / 100.0

    meta = (
        w["health"] * h_norm
        + w["reward"] * r_norm
        + w["risk"] * risk_norm
        + w["data"] * data_norm
        + w["research"] * research_norm
    )

    # map חזרה ל-0–100, תוך clipping
    meta_score = max(0.0, min(100.0, 50.0 + 50.0 * meta))
    return float(meta_score)
def ctx_to_action_playbook(
    ctx: AppContext,
    *,
    kpis: Optional[JSONDict] = None,
) -> JSONDict:
    """
    מייצר "Playbook" טקסטואלי של פעולות מומלצות לפי:
    - health_score
    - live_readiness (compute_live_readiness)
    - policy_status (risk/data/research flags)
    - KPIs (Sharpe, PnL, MaxDD)

    החזרה:
    {
      "actions": [
        {"code": str, "text": str, "priority": "high"/"medium"/"low"},
        ...
      ],
      "summary": {...}
    }
    """
    kpis = dict(kpis or {})
    actions: List[JSONDict] = []

    # live readiness / health / reasons
    try:
        readiness = compute_live_readiness(ctx)
    except Exception:
        readiness = {}

    live_ready = bool(readiness.get("live_ready", False))
    reasons = readiness.get("reasons", []) or []
    health = float(ctx.health_score or 0.0)

    ps = ctx.policy_status or {}
    risk_flags = ps.get("risk_policy_flags", []) or []
    data_flags = ps.get("data_policy_flags", []) or []
    research_flags = ps.get("research_policy_flags", []) or []

    sharpe = kpis.get("sharpe")
    total_pnl = kpis.get("total_pnl")
    max_dd = kpis.get("max_dd")

    def add_action(code: str, text: str, priority: str = "medium") -> None:
        actions.append(
            {
                "code": code,
                "text": text,
                "priority": priority,
            }
        )

    # 1) אם לא לייב-רדי – פעולות בלוק ראשי
    if not live_ready:
        add_action(
            "do_not_go_live_yet",
            "אל תעביר את הקונפיג ללייב לפני טיפול בדגלים ובמדיניות.",
            priority="high",
        )
        if any("risk_policy_not_ok" in r or "risk_" in r for r in reasons) or risk_flags:
            add_action(
                "review_risk_limits",
                "בדוק מחדש את גבולות הסיכון (daily loss, max DD, leverage, VIX) והקשח אותם במידת הצורך.",
                priority="high",
            )
        if any("data_policy_not_ok" in r or "data_" in r for r in reasons) or data_flags:
            add_action(
                "improve_data_quality",
                "שפר את איכות הדאטה: כיסוי, NaNs, overlap בין זוגות ומספר ימי מסחר חסרים.",
                priority="high",
            )
        if any("research_policy_not_ok" in r or "research_" in r for r in reasons) or research_flags:
            add_action(
                "extend_research_sample",
                "הרחב את תקופת המחקר ומספר הטריידים כדי לשפר את מובהקות התוצאות.",
                priority="medium",
            )
        if any("too_few_pairs" in r for r in reasons):
            add_action(
                "expand_universe",
                "הרחב את מספר הזוגות/נכסים ב-universe כדי לקבל פיזור טוב יותר ומדגם אמין יותר.",
                priority="medium",
            )

    # 2) לפי health_score
    if health < 60.0:
        add_action(
            "raise_health_score",
            "ציון הבריאות נמוך (<60). מומלץ לטפל בדגלי סיכון/דאטה ולהריץ מחדש Backtest/Optimization.",
            priority="high",
        )
    elif health < 75.0:
        add_action(
            "stabilize_config",
            "ציון הבריאות בינוני – חזק את איכות הדאטה ויציבות ה-Backtests לפני העלאת scale.",
            priority="medium",
        )
    else:
        add_action(
            "monitor_health",
            "ה-health_score גבוה יחסית – המשך לנטר שינויים חדים דרמטיים בין ריצות.",
            priority="low",
        )

    # 3) לפי KPIs
    try:
        if sharpe is not None:
            s = float(sharpe)
            if s < 0:
                add_action(
                    "fix_negative_sharpe",
                    "Sharpe שלילי – בדוק מחדש את הלוגיקה / פרמטרים של האסטרטגיה לפני המשך שימוש.",
                    priority="high",
                )
            elif s < 0.5:
                add_action(
                    "improve_sharpe",
                    "Sharpe נמוך – נסה לשפר פילטרים, תקופות, ופרמטרים אופטימליים.",
                    priority="medium",
                )
            elif s > 1.5 and health >= 75.0 and live_ready:
                add_action(
                    "consider_scaling_up",
                    "Sharpe גבוה ו-health טוב – ניתן לשקול הגדלת scale בצורה הדרגתית תחת מגבלות הסיכון.",
                    priority="medium",
                )
    except Exception:
        pass

    try:
        if max_dd is not None:
            dd = float(max_dd)
            if dd < -0.2:  # DD עמוק יותר מ-20%
                add_action(
                    "reduce_drawdown_risk",
                    "Drawdown עמוק – שקול הורדת מינוף, קיצור תקופות החזקה או חיזוק מנגנוני Cut.",
                    priority="high",
                )
    except Exception:
        pass

    # 4) אם כן לייב-רדי
    if live_ready and health >= 70.0:
        add_action(
            "go_live_with_limits",
            "התצורה מוגדרת כ-Live Ready – ניתן לשקול מעבר ללייב תחת Kill-Switch ברור ומוניטורינג צמוד.",
            priority="high",
        )

    summary = {
        "live_ready": live_ready,
        "health_score": health,
        "reasons": reasons,
        "sharpe": sharpe,
        "total_pnl": total_pnl,
        "max_dd": max_dd,
    }

    return {
        "actions": actions,
        "summary": make_json_safe(summary),
    }

# ========= Experiment / run registry (session-level) =========

def register_experiment_run(
    ctx: AppContext,
    *,
    kpis: Optional[JSONDict] = None,
    extra_meta: Optional[JSONDict] = None,
    reward: Optional[float] = None,
    auto_reward: bool = True,
) -> None:
    """
    רושם ריצה (קונטקסט + KPIs) ב-_experiment_runs בסשן.

    kpis יכול לכלול:
    - total_pnl
    - sharpe
    - max_dd
    - tail_risk_score
    - wf_stability_score
    """
    try:
        runs = st.session_state.get("_experiment_runs", [])
        if not isinstance(runs, list):
            runs = []

        kpis = dict(kpis or {})
        feat = ctx_to_feature_vector(ctx, kpis=kpis)

        # חישוב reward בסיסי (אם לא הועבר מבחוץ)
        if reward is None and auto_reward:
            reward = compute_context_reward(
                total_pnl=kpis.get("total_pnl"),
                sharpe=kpis.get("sharpe"),
                max_dd=kpis.get("max_dd"),
                tail_risk_score=kpis.get("tail_risk_score"),
                stability_score=kpis.get("wf_stability_score"),
            )

        # Meta-score משולב (0–100)
        try:
            meta_score = compute_meta_score(ctx, kpis=kpis)
        except Exception:
            meta_score = None

        record: JSONDict = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_id": ctx.run_id,
            "section": ctx.section,
            "profile": ctx.profile,
            "environment": ctx.environment,
            "experiment_name": ctx.experiment_name,
            "experiment_group": ctx.experiment_group,
            "run_label": ctx.run_label,
            "config_hash": ctx.config_hash,
            "health_score": ctx.health_score,
            "tags": make_json_safe(ctx.tags),
            "features": feat,
            "kpis": make_json_safe(kpis),
            "reward": float(reward) if reward is not None else None,
            "meta_score": float(meta_score) if meta_score is not None else None,
            "extra_meta": make_json_safe(extra_meta or {}),
        }
        runs.append(record)
        st.session_state["_experiment_runs"] = runs[-500:]
    except Exception:
        pass


def get_experiment_runs_dataframe() -> Optional[pd.DataFrame]:
    """
    מחזיר DataFrame של _experiment_runs (אם קיים).
    """
    runs = st.session_state.get("_experiment_runs")
    if not isinstance(runs, list) or not runs:
        return None
    try:
        df = pd.DataFrame(runs)
        if "health_score" in df.columns:
            df["health_score"] = pd.to_numeric(df["health_score"], errors="coerce")
        if "reward" in df.columns:
            df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
        if "meta_score" in df.columns:
            df["meta_score"] = pd.to_numeric(df["meta_score"], errors="coerce")
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            
        return df
    except Exception:
        return None


def filter_experiment_runs(
    *,
    section: Optional[str] = None,
    profile: Optional[str] = None,
    environment: Optional[str] = None,
    experiment_group: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    פילטר ריצות ניסוי לפי section/profile/env/experiment_group.
    """
    df = get_experiment_runs_dataframe()
    if df is None or df.empty:
        return None

    mask = pd.Series(True, index=df.index)
    if section is not None:
        mask &= df["section"].astype(str) == str(section)
    if profile is not None:
        mask &= df["profile"].astype(str) == str(profile)
    if environment is not None:
        mask &= df["environment"].astype(str) == str(environment)
    if experiment_group is not None:
        mask &= df["experiment_group"].astype(str) == str(experiment_group)

    df_f = df[mask].copy()
    return df_f if not df_f.empty else None


def suggest_top_experiments(
    *,
    section: Optional[str] = None,
    profile: Optional[str] = None,
    environment: Optional[str] = None,
    experiment_group: Optional[str] = None,
    top_n: int = 5,
    metric: str = "meta_score",  # אפשר גם "reward", "health_score", "kpis_total_pnl", ...
) -> Optional[pd.DataFrame]:

    """
    מחזיר את ה-Top-N ריצות הניסוי לפי metric מסוים.
    """
    df = filter_experiment_runs(
        section=section,
        profile=profile,
        environment=environment,
        experiment_group=experiment_group,
    )
    if df is None or df.empty:
        return None

    # במקרה שרוצים metric מתוך kpis (למשל kpis.total_pnl)
    if metric.startswith("kpis_"):
        col = metric
        if col not in df.columns:
            # ננסה לפצל kpis לדאטה אמתית
            try:
                kpis_expanded = df["kpis"].apply(lambda x: x or {})
                kdf = pd.json_normalize(kpis_expanded)
                kdf.columns = [f"kpis_{c}" for c in kdf.columns]
                df = pd.concat([df, kdf], axis=1)
            except Exception:
                return None
        if col not in df.columns:
            return None

    if metric not in df.columns:
        # fallback ל-reward
        metric = "reward"
        if metric not in df.columns:
            return None

    try:
        df_rank = df.copy()
        df_rank[metric] = pd.to_numeric(df_rank[metric], errors="coerce")
        df_rank = df_rank.sort_values(metric, ascending=False)
        return df_rank.head(int(top_n))
    except Exception:
        return None


# ========= Export ctx history & experiments to Parquet =========

def export_ctx_history_to_parquet(path: Optional[Path] = None) -> Optional[Path]:
    """
    מייצא את _ctx_history ל-Parquet (לניתוח offline).
    """
    df = summarize_ctx_history()
    if df is None or df.empty:
        return None
    try:
        ensure_logs_dirs()
        if path is None:
            fname = f"ctx_history_{datetime.now().strftime('%Y%m%d-%H%M%S')}.parquet"
            path = LOGS_DIR / fname
        df.to_parquet(path)
        return path
    except Exception:
        return None


def export_experiment_runs_to_parquet(path: Optional[Path] = None) -> Optional[Path]:
    """
    מייצא את _experiment_runs ל-Parquet (לניתוח חיצוני).
    """
    df = get_experiment_runs_dataframe()
    if df is None or df.empty:
        return None
    try:
        ensure_logs_dirs()
        if path is None:
            fname = f"experiment_runs_{datetime.now().strftime('%Y%m%d-%H%M%S')}.parquet"
            path = LOGS_DIR / fname
        df.to_parquet(path)
        return path
    except Exception:
        return None
    
def _get_default_db_path() -> Path:
    """
    מחזיר נתיב ברירת מחדל ל-DB (DuckDB) תחת logs/.
    """
    ensure_logs_dirs()
    return LOGS_DIR / "ctx_experiments.duckdb"

def persist_ctx_history_to_db(
    path: Optional[Path] = None,
    *,
    table_name: str = "ctx_history",
) -> Optional[Path]:
    """
    שומר את היסטוריית הקונטקסטים (ctx_history) ל-DB (DuckDB) בטבלה אחת:
    - אם הטבלה לא קיימת: נוצרת.
    - אם קיימת: נוסיף שורות (append).

    החזרה:
    - Path ל-DB אם הצליח
    - None אם נכשל (לא מפיל את הדשבורד)
    """
    df = summarize_ctx_history()
    if df is None or df.empty:
        return None

    try:
        import duckdb  # type: ignore
    except Exception:
        # DuckDB לא זמין בסביבה
        return None

    try:
        if path is None:
            path = _get_default_db_path()
        else:
            path = Path(path)

        ensure_logs_dirs()
        con = duckdb.connect(str(path))

        # נרשום את df כטבלה זמנית ב-DuckDB
        con.register("df_ctx_hist", df)

        # ננסה ליצור טבלה אם לא קיימת (עם אותו schema, בלי שורות)
        try:
            con.execute(
                f"CREATE TABLE {table_name} AS SELECT * FROM df_ctx_hist LIMIT 0"
            )
        except Exception:
            # כנראה הטבלה כבר קיימת – נתעלם
            pass

        # נכניס את כל ההיסטוריה (append)
        con.execute(
            f"INSERT INTO {table_name} SELECT * FROM df_ctx_hist"
        )

        con.close()
        return path
    except Exception:
        return None

def persist_experiment_runs_to_db(
    path: Optional[Path] = None,
    *,
    table_name: str = "experiment_runs",
) -> Optional[Path]:
    """
    שומר את ריצות הניסוי (_experiment_runs) ל-DB (DuckDB) בטבלה אחת:
    - אם הטבלה לא קיימת: נוצרת.
    - אם קיימת: נוסיף שורות (append).
    """
    df = get_experiment_runs_dataframe()
    if df is None or df.empty:
        return None

    try:
        import duckdb  # type: ignore
    except Exception:
        return None

    try:
        if path is None:
            path = _get_default_db_path()
        else:
            path = Path(path)

        ensure_logs_dirs()
        con = duckdb.connect(str(path))

        con.register("df_experiments", df)

        try:
            con.execute(
                f"CREATE TABLE {table_name} AS SELECT * FROM df_experiments LIMIT 0"
            )
        except Exception:
            pass  # טבלה קיימת כבר

        con.execute(
            f"INSERT INTO {table_name} SELECT * FROM df_experiments"
        )

        con.close()
        return path
    except Exception:
        return None

# ========= Multiverse: compare multiple contexts (high-level) =========

def multiverse_compare_contexts(
    ctx_list: List[AppContext],
) -> JSONDict:
    """
    השוואת כמה קונטקסטים:
    מחזיר סיכום:
    - רשימת run_ids
    - טווחי תאריכים
    - ממוצע/מינימום/מקסימום health_score
    - ספירת sections / profiles / envs
    - אינדיקציה בסיסית אם יש ctx ב-dev/paper/live
    """
    if not ctx_list:
        return {}

    run_ids = [ctx.run_id for ctx in ctx_list]
    sections = [ctx.section for ctx in ctx_list]
    profiles = [ctx.profile for ctx in ctx_list]
    envs = [ctx.environment for ctx in ctx_list]
    hs = [float(ctx.health_score or 0.0) for ctx in ctx_list]

    start_dates = [ctx.start_date for ctx in ctx_list]
    end_dates = [ctx.end_date for ctx in ctx_list]

    envs_series = pd.Series(envs) if envs else pd.Series(dtype=object)
    envs_count = dict(envs_series.value_counts().to_dict()) if not envs_series.empty else {}
    envs_set = set(envs)

    summary: JSONDict = {
        "run_ids": run_ids,
        "sections": sections,
        "profiles": profiles,
        "environments": envs,
        "start_dates": [str(d) for d in start_dates],
        "end_dates": [str(d) for d in end_dates],
        "health_score_min": float(min(hs)) if hs else None,
        "health_score_max": float(max(hs)) if hs else None,
        "health_score_avg": float(sum(hs) / len(hs)) if hs else None,
        "sections_count": dict(pd.Series(sections).value_counts().to_dict()),
        "profiles_count": dict(pd.Series(profiles).value_counts().to_dict()),
        "envs_count": envs_count,
        "n_contexts": len(ctx_list),
        "has_dev": "dev" in envs_set,
        "has_paper": "paper" in envs_set,
        "has_live": "live" in envs_set,
    }
    return make_json_safe(summary)

# ========= Context lineage & DAG (עץ ניסויים) =========

def _build_run_metadata_index() -> Dict[str, JSONDict]:
    """
    בונה אינדקס run_id -> מטא (section/profile/env/ts/health/reward)
    מתוך _ctx_history ו-_experiment_runs בסשן.
    """
    idx: Dict[str, JSONDict] = {}

    # מהיסטוריית קונטקסטים
    hist = st.session_state.get("_ctx_history") or []
    if isinstance(hist, list):
        for row in hist:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("run_id") or "").strip()
            if not rid:
                continue
            meta = idx.get(rid, {"run_id": rid})
            meta.setdefault("section", row.get("section"))
            meta.setdefault("profile", row.get("profile"))
            # env בהיסטוריה נקרא "env"
            env = row.get("env")
            if env is not None:
                meta.setdefault("environment", env)
            meta.setdefault("health_score", row.get("health_score"))
            meta.setdefault("start", row.get("start"))
            meta.setdefault("end", row.get("end"))
            idx[rid] = meta

    # מריצות ניסוי (_experiment_runs)
    runs = st.session_state.get("_experiment_runs") or []
    if isinstance(runs, list):
        for rec in runs:
            if not isinstance(rec, dict):
                continue
            rid = str(rec.get("run_id") or "").strip()
            if not rid:
                continue
            meta = idx.get(rid, {"run_id": rid})
            meta.setdefault("section", rec.get("section"))
            meta.setdefault("profile", rec.get("profile"))
            meta.setdefault("environment", rec.get("environment"))
            if "ts" in rec and "ts" not in meta:
                meta["ts"] = rec["ts"]
            if "reward" in rec and "reward" not in meta:
                meta["reward"] = rec["reward"]
            idx[rid] = meta

    return idx


def register_child_ctx(parent: AppContext, child: AppContext) -> None:
    """
    רושם קשר אב→בן בין שני קונטקסטים:
    - מעדכן mapping ב-session_state['_ctx_lineage']
    - מעדכן provenance של parent/child (children_run_ids + lineage_depth)
    """
    try:
        parent_id = str(parent.run_id)
        child_id = str(child.run_id)

        # lineage map ב-session: parent_run_id -> [child_run_ids...]
        lineage = st.session_state.get("_ctx_lineage", {})
        if not isinstance(lineage, dict):
            lineage = {}

        children = list(lineage.get(parent_id, []))
        if child_id not in children:
            children.append(child_id)
        lineage[parent_id] = children
        st.session_state["_ctx_lineage"] = lineage

        # עדכון provenance של child
        if child.provenance is None:
            child.provenance = ContextProvenance(
                parent_run_id=parent_id,
                source_section=child.section,
                source_file="root/dashboard.py",
                notes="registered via register_child_ctx",
            )
        else:
            child.provenance.parent_run_id = parent_id

        # עומק (depth)
        depth = 0
        if parent.provenance is not None:
            try:
                depth = max(0, int(parent.provenance.lineage_depth))
            except Exception:
                depth = 0
        child.provenance.lineage_depth = depth + 1

        # רשימת ילדים ב-provenance של parent (אם אפשר)
        if parent.provenance is not None:
            if child_id not in parent.provenance.children_run_ids:
                parent.provenance.children_run_ids.append(child_id)

        # שומרים את child כקונטקסט נוכחי אם רוצים
        save_ctx_to_session(child)
    except Exception:
        # לא מפילים את הדשבורד בגלל lineage
        pass


def build_ctx_lineage_graph(root_run_id: str) -> JSONDict:
    """
    בונה DAG של ריצות סביב run_id מסוים על בסיס:
    - session_state['_ctx_lineage'] (parent -> children)
    - _ctx_history / _experiment_runs למטא
    מחזיר:
    {
        "root": run_id בפסגה,
        "nodes": [ {run_id, section, profile, environment, ...}, ... ],
        "edges": [ {"parent": run_id, "child": run_id}, ... ]
    }
    """
    lineage = st.session_state.get("_ctx_lineage", {})
    if not isinstance(lineage, dict) or not lineage:
        return {"root": str(root_run_id), "nodes": [], "edges": []}

    # child -> parent (כדי לעלות אחורה עד root אמיתי)
    parent_by_child: Dict[str, str] = {}
    for p_id, children in lineage.items():
        p_id = str(p_id)
        for c_id in children or []:
            parent_by_child[str(c_id)] = p_id

    # מוצאים root אמיתי (עולים למעלה כל עוד יש אב)
    root = str(root_run_id)
    while parent_by_child.get(root):
        root = parent_by_child[root]

    from collections import deque

    idx = _build_run_metadata_index()
    visited: set[str] = set()
    nodes: Dict[str, JSONDict] = {}
    edges: List[JSONDict] = []

    q: deque[str] = deque([root])
    while q:
        rid = q.popleft()
        if rid in visited:
            continue
        visited.add(rid)

        meta = idx.get(rid, {"run_id": rid})
        # נוודא מפתחות בסיסיים
        meta.setdefault("section", None)
        meta.setdefault("profile", None)
        meta.setdefault("environment", None)
        nodes[rid] = meta

        for child_id in map(str, lineage.get(rid, []) or []):
            edges.append({"parent": rid, "child": child_id})
            if child_id not in visited:
                q.append(child_id)

    return {
        "root": root,
        "nodes": list(nodes.values()),
        "edges": edges,
    }
def check_env_consistency(
    dev_ctx: Optional[AppContext] = None,
    paper_ctx: Optional[AppContext] = None,
    live_ctx: Optional[AppContext] = None,
) -> JSONDict:
    """
    בודק עקביות בין Dev / Paper / Live:

    - האם config_hash זהה?
    - האם פרמטרי סיכון בסיסיים (capital / max_exposure / max_leverage) תואמים?
    - החזר:
      {
        "envs_present": [...],
        "config_hashes": {...},
        "config_hash_consistent": bool | None,
        "risk_params": {
            "capital": {...},
            "max_exposure_per_trade": {...},
            "max_leverage": {...},
        },
        "risk_params_consistent": bool | None,
        "warnings": [str, ...],
      }
    """
    env_map: Dict[str, AppContext] = {}
    if dev_ctx is not None:
        env_map["dev"] = dev_ctx
    if paper_ctx is not None:
        env_map["paper"] = paper_ctx
    if live_ctx is not None:
        env_map["live"] = live_ctx

    envs_present = list(env_map.keys())
    if not envs_present:
        return {
            "envs_present": [],
            "config_hashes": {},
            "config_hash_consistent": None,
            "risk_params": {},
            "risk_params_consistent": None,
            "warnings": ["no_environments_provided"],
        }

    # config_hash לכל env
    config_hashes: Dict[str, Optional[str]] = {
        env: ctx.config_hash for env, ctx in env_map.items()
    }
    unique_hashes = {h for h in config_hashes.values() if h is not None}
    config_consistent: Optional[bool]
    if not unique_hashes:
        config_consistent = None
    else:
        config_consistent = len(unique_hashes) == 1

    # פרמטרי סיכון בסיסיים
    risk_params: Dict[str, Dict[str, float]] = {
        "capital": {},
        "max_exposure_per_trade": {},
        "max_leverage": {},
    }
    for env, ctx in env_map.items():
        try:
            risk_params["capital"][env] = float(ctx.capital)
        except Exception:
            pass
        try:
            risk_params["max_exposure_per_trade"][env] = float(ctx.max_exposure_per_trade)
        except Exception:
            pass
        try:
            risk_params["max_leverage"][env] = float(ctx.max_leverage)
        except Exception:
            pass

    # עקביות פרמטרי סיכון: נבדוק אם יש variance בין envs
    def _is_consistent(d: Dict[str, float]) -> bool:
        if not d:
            return True
        vals = list(d.values())
        return max(vals) - min(vals) < 1e-6

    risk_consistent = (
        _is_consistent(risk_params["capital"])
        and _is_consistent(risk_params["max_exposure_per_trade"])
        and _is_consistent(risk_params["max_leverage"])
    )

    warnings: List[str] = []
    if config_consistent is False:
        warnings.append("config_hash_mismatch_between_envs")
    if not risk_consistent:
        warnings.append("risk_params_mismatch_between_envs")

    # ספציפית לשילוב Dev → Paper → Live
    if "live" in envs_present and ("paper" in envs_present or "dev" in envs_present):
        if config_consistent is False:
            warnings.append("live_not_aligned_with_non_live_config")
        if not risk_consistent:
            warnings.append("live_not_aligned_with_non_live_risk_params")

    return {
        "envs_present": envs_present,
        "config_hashes": config_hashes,
        "config_hash_consistent": config_consistent,
        "risk_params": risk_params,
        "risk_params_consistent": risk_consistent,
        "warnings": warnings,
    }

# ========= Context Templates (named setups) =========

@dataclass
class ContextTemplate:
    """
    תבנית קונטקסט ליצירת ctx חדש / build_ctx מהיר.
    """
    name: str
    description: str = ""
    section: str = "backtest"
    profile: str = APP_PROFILE
    environment: str = APP_ENVIRONMENT
    # טווח ברירת מחדל (בימים אחורה מ"היום") אם לא ניתנים start/end ישירים
    default_lookback_days: int = 365

    # פרמטרי סיכון דיפולטיים
    capital: float = 100_000.0
    max_exposure_per_trade: float = 0.05
    max_leverage: float = 2.0

    # אפשר להכניס עוד defaults ל-controls
    default_controls: JSONDict = field(default_factory=dict)

    # תגים לוגיים
    tags: TagDict = field(default_factory=dict)


_TEMPLATE_REGISTRY: Dict[str, ContextTemplate] = {}


def _get_templates_path() -> Path:
    cfg_dir = PROJECT_ROOT / "configs"
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return cfg_dir / "context_templates.json"


def register_context_template(tmpl: ContextTemplate) -> None:
    """
    רושם תבנית לזיכרון (רמת סשן/תהליך).
    """
    _TEMPLATE_REGISTRY[tmpl.name] = tmpl


def get_context_template(name: str) -> Optional[ContextTemplate]:
    return _TEMPLATE_REGISTRY.get(name)


def list_context_templates() -> List[ContextTemplate]:
    return list(_TEMPLATE_REGISTRY.values())


def load_context_templates_from_disk() -> None:
    """
    טוען תבניות מ-json ל-_TEMPLATE_REGISTRY.
    """
    p = _get_templates_path()
    if not p.exists():
        return
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        doc = json.loads(txt)
        if isinstance(doc, list):
            for item in doc:
                if not isinstance(item, dict):
                    continue
                try:
                    tmpl = ContextTemplate(
                        name=str(item["name"]),
                        description=str(item.get("description", "")),
                        section=str(item.get("section", "backtest")),
                        profile=str(item.get("profile", APP_PROFILE)),
                        environment=str(item.get("environment", APP_ENVIRONMENT)),
                        default_lookback_days=int(item.get("default_lookback_days", 365)),
                        capital=float(item.get("capital", 100_000.0)),
                        max_exposure_per_trade=float(item.get("max_exposure_per_trade", 0.05)),
                        max_leverage=float(item.get("max_leverage", 2.0)),
                        default_controls=dict(item.get("default_controls", {})),
                        tags=dict(item.get("tags", {})),
                    )
                    register_context_template(tmpl)
                except Exception:
                    continue
    except Exception:
        return


def save_context_templates_to_disk() -> Optional[Path]:
    """
    שומר את כל התבניות ל-json תחת configs/context_templates.json.
    """
    p = _get_templates_path()
    try:
        payload = []
        for tmpl in list_context_templates():
            payload.append(
                {
                    "name": tmpl.name,
                    "description": tmpl.description,
                    "section": tmpl.section,
                    "profile": tmpl.profile,
                    "environment": tmpl.environment,
                    "default_lookback_days": tmpl.default_lookback_days,
                    "capital": tmpl.capital,
                    "max_exposure_per_trade": tmpl.max_exposure_per_trade,
                    "max_leverage": tmpl.max_leverage,
                    "default_controls": make_json_safe(tmpl.default_controls),
                    "tags": make_json_safe(tmpl.tags),
                }
            )
        p.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return p
    except Exception:
        return None


def create_template_from_ctx(
    ctx: AppContext,
    *,
    name: str,
    description: str = "",
) -> ContextTemplate:
    """
    יוצר תבנית חדשה מתוך קונטקסט קיים.
    (לא שומר לדיסק – רק רושם ב-REGISTRY; אפשר אח"כ save_context_templates_to_disk).
    """
    days = (ctx.end_date - ctx.start_date).days
    tmpl = ContextTemplate(
        name=name,
        description=description,
        section=ctx.section,
        profile=ctx.profile,
        environment=ctx.environment,
        default_lookback_days=max(1, days),
        capital=ctx.capital,
        max_exposure_per_trade=ctx.max_exposure_per_trade,
        max_leverage=ctx.max_leverage,
        default_controls={
            k: v
            for k, v in ctx.controls.items()
            if k not in ("seed", "feature_flags_snapshot")
        },
        tags=dict(ctx.tags),
    )
    register_context_template(tmpl)
    return tmpl


def build_ctx_from_template(
    template_name: str,
    *,
    config: Optional[JSONDict] = None,
    pairs: Optional[List[Any]] = None,
    override_controls: Optional[JSONDict] = None,
) -> Optional[AppContext]:
    """
    יוצר AppContext חדש מתבנית + config + pairs.
    מגדיר start/end לפי lookback מהיום, אם ה-UI לא מספק start/end ידניים.

    שדרוג קטן:
    ----------
    משתמש ב-datetime.timedelta במקום pd.Timedelta כדי לא לתלות את app_context בפנדות.
    """
    tmpl = get_context_template(template_name)
    if tmpl is None:
        return None

    # נקבע טווח לפי היום
    today = date.today()
    lookback_days = int(tmpl.default_lookback_days)
    if lookback_days < 1:
        lookback_days = 1
    start_date = today - timedelta(days=lookback_days)
    end_date = today

    base_controls = dict(tmpl.default_controls)
    if override_controls:
        base_controls.update(override_controls)

    # פרמטרי סיכון מהתבנית
    base_controls.setdefault("capital", tmpl.capital)
    base_controls.setdefault("max_exposure_per_trade", tmpl.max_exposure_per_trade)
    base_controls.setdefault("max_leverage", tmpl.max_leverage)

    ctx = build_ctx(
        start_date=start_date,
        end_date=end_date,
        controls=base_controls,
        section=tmpl.section,
        config=config,
        pairs=pairs,
        tags=tmpl.tags,
        parent_run_id=None,
        source_section=tmpl.section,
        provenance_notes=f"created-from-template:{tmpl.name}",
    )

    return ctx

def discover_best_templates_from_experiments(
    *,
    section: str = "backtest",
    min_meta_score: float = 60.0,
    min_health_score: float = 60.0,
    top_n: int = 10,
) -> List[ContextTemplate]:
    """
    מחפש ריצות ניסוי חזקות ב-_experiment_runs והופך אותן לתבניות ContextTemplate.

    לוגיקה:
    -------
    1. מסנן ריצות לפי section (ברירת מחדל: "backtest").
    2. מסנן לפי meta_score ו-health_score מינימלי.
    3. ממיין לפי meta_score (ואז לפי ts אם קיים).
    4. לכל run_id מנסה:
        - למצוא snapshot אחרון (latest_snapshot_for_run).
        - לטעון AppContext מה-snapshot.
        - ליצור ContextTemplate דרך create_template_from_ctx.
    5. רושם את התבניות ב-_TEMPLATE_REGISTRY ומחזיר רשימה.

    שים לב:
    - לא שומר לדיסק – אפשר לקרוא אחרי זה save_context_templates_to_disk().
    """
    df = filter_experiment_runs(section=section)
    if df is None or df.empty:
        return []

    df = df.copy()

    # נוודא שעמודות רלוונטיות הן מספריות
    if "meta_score" in df.columns:
        df["meta_score"] = pd.to_numeric(df["meta_score"], errors="coerce")
    else:
        df["meta_score"] = None

    if "health_score" in df.columns:
        df["health_score"] = pd.to_numeric(df["health_score"], errors="coerce")
    else:
        df["health_score"] = None

    # סינון לפי ספים
    mask = pd.Series(True, index=df.index)
    if min_meta_score is not None:
        mask &= (df["meta_score"].isna()) | (df["meta_score"] >= float(min_meta_score))
    if min_health_score is not None:
        mask &= (df["health_score"].isna()) | (df["health_score"] >= float(min_health_score))

    df_f = df[mask].copy()
    if df_f.empty:
        # fallback – ננסה לפי reward בלבד
        if "reward" in df.columns:
            df_f = df.copy()
        else:
            return []

    # מיון: קודם לפי meta_score (אם קיים), אחרת לפי reward; ואז לפי ts יורד
    sort_cols: List[str] = []
    ascending: List[bool] = []

    if "meta_score" in df_f.columns:
        sort_cols.append("meta_score")
        ascending.append(False)
    if "reward" in df_f.columns:
        sort_cols.append("reward")
        ascending.append(False)
    if "ts" in df_f.columns:
        sort_cols.append("ts")
        ascending.append(False)

    if sort_cols:
        df_f = df_f.sort_values(sort_cols, ascending=ascending)

    # ניקח מועמדים ראשונים (יותר מ-top_n כדי לגבות כישלונות טעינת snapshot)
    max_candidates = max(top_n * 3, top_n)
    df_f = df_f.head(max_candidates)

    templates: List[ContextTemplate] = []
    used_names: set[str] = set()

    for _, row in df_f.iterrows():
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue

        snap_meta = latest_snapshot_for_run(run_id)
        if not isinstance(snap_meta, dict):
            continue

        snap_path = snap_meta.get("path") or snap_meta.get("name")
        if not snap_path:
            continue

        # נטען קונטקסט מה-snapshot
        ctx = load_snapshot_as_ctx(str(snap_path))
        if ctx is None:
            continue

        # נייצר שם תבנית יציב (section + env + קצה run_id)
        env = ctx.environment or "env"
        prof = ctx.profile or "profile"
        suffix = run_id[-6:] if len(run_id) >= 6 else run_id
        base_name = f"{section}_{env}_{prof}_auto_{suffix}"

        # למנוע כפילויות בשם
        name = base_name
        i = 1
        while name in _TEMPLATE_REGISTRY or name in used_names:
            i += 1
            name = f"{base_name}_{i}"

        desc = (
            f"Auto template discovered from run {run_id} "
            f"(section={ctx.section}, env={ctx.environment}, profile={ctx.profile})."
        )

        try:
            tmpl = create_template_from_ctx(
                ctx,
                name=name,
                description=desc,
            )
        except Exception:
            continue

        templates.append(tmpl)
        used_names.add(name)

        if len(templates) >= top_n:
            break

    return templates

# ========= Context bundles (ZIP: ctx + KPIs + config) =========

def export_context_bundle(
    ctx: AppContext,
    *,
    kpis: Optional[JSONDict] = None,
    include_config: bool = True,
    include_history: bool = False,
) -> Optional[Path]:
    """
    יוצר ZIP bundle:
    - ctx.json        (AppContext מלא)
    - kpis.json       (אם קיים)
    - config.json     (אם include_config=True)
    - ctx_history.parquet (אם include_history=True ויש history)
    """
    import io
    import zipfile

    ensure_logs_dirs()
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # ctx.json
            zf.writestr(
                "ctx.json",
                json.dumps(ctx.to_dict(), ensure_ascii=False, indent=2),
            )

            # kpis.json
            if kpis:
                zf.writestr(
                    "kpis.json",
                    json.dumps(make_json_safe(kpis), ensure_ascii=False, indent=2),
                )

            # config.json (אפקטיבי)
            if include_config:
                zf.writestr(
                    "config.json",
                    json.dumps(make_json_safe(ctx.config), ensure_ascii=False, indent=2),
                )

            # history parquet
            if include_history:
                df_hist = summarize_ctx_history()
                if df_hist is not None and not df_hist.empty:
                    import tempfile
                    tmp_path = Path(tempfile.gettempdir()) / "tmp_ctx_history.parquet"
                    df_hist.to_parquet(tmp_path)
                    with tmp_path.open("rb") as fh:
                        zf.writestr("ctx_history.parquet", fh.read())

        fname = f"ctx_bundle_{ctx.run_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
        out_path = LOGS_DIR / fname
        out_path.write_bytes(buf.getvalue())
        return out_path
    except Exception:
        return None


def import_context_bundle(path: Path) -> Optional[Dict[str, Any]]:
    """
    טוען bundle (zip) ומחזיר dict עם:
    - ctx: dict של AppContext
    - kpis: dict אם קיים
    - config: dict אם קיים
    - ctx_history: DataFrame אם קיים
    """
    import zipfile
    import io

    if not path.exists():
        return None

    result: Dict[str, Any] = {}
    try:
        with zipfile.ZipFile(path, mode="r") as zf:
            # ctx.json
            try:
                with zf.open("ctx.json") as fh:
                    ctx_doc = json.loads(fh.read().decode("utf-8", errors="ignore"))
                    result["ctx"] = ctx_doc
            except Exception:
                result["ctx"] = None

            # kpis.json
            try:
                with zf.open("kpis.json") as fh:
                    kpis_doc = json.loads(fh.read().decode("utf-8", errors="ignore"))
                    result["kpis"] = kpis_doc
            except Exception:
                result["kpis"] = None

            # config.json
            try:
                with zf.open("config.json") as fh:
                    cfg_doc = json.loads(fh.read().decode("utf-8", errors="ignore"))
                    result["config"] = cfg_doc
            except Exception:
                result["config"] = None

            # ctx_history.parquet (אם יש)
            try:
                names = [info.filename for info in zf.infolist()]
                if "ctx_history.parquet" in names:
                    with zf.open("ctx_history.parquet") as fh:
                        data = fh.read()
                        bio = io.BytesIO(data)
                        df_hist = pd.read_parquet(bio)
                        result["ctx_history"] = df_hist
                else:
                    result["ctx_history"] = None
            except Exception:
                result["ctx_history"] = None

        return result
    except Exception:
        return None
