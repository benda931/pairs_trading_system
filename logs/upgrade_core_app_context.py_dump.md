### Summary of changes

- Normalized and documented environment modes (`dev`, `research`, `paper`, `live`) and wired them through a new, safer `get_app_context` entry point (session-aware, Streamlit-friendly).
- Introduced lightweight typed service protocols (`SqlStoreLike`, `MarketDataRouterLike`, `BrokerLike`) and updated `AppContext` service attributes accordingly (no behavioral changes, typing/IDE help only).
- Enhanced `AppContext.get_global` to prefer a per-Streamlit-session context (`st.session_state["app_ctx"]`) while keeping the existing class-level singleton for backward compatibility.
- Added lifecycle management to `AppContext` via `close`, `__enter__`, and `__exit__` to gracefully shut down broker/IB, market-data router, and SQL store resources.
- Added `AppContext.log_event` helper for structured logging with `run_id`, `section`, and `environment` baked into log records.
- Made IBKR router initialization feature-flag/config aware (`ibkr.enabled` / `feature_flags_snapshot.enable_ib_router`) to avoid unexpected live connections in restricted modes.
- Slightly extended feature-vector encoding to distinguish a `research` environment while keeping existing `dev/paper/live` behavior.
- Added an `AppContext.save_snapshot` convenience method and hooked automatic snapshots for `paper/live` contexts into `apply_policies_to_ctx` under a conservative helper `_should_auto_snapshot`.
- Improved docstrings and comments around extension points (how to register/override services and plug new data providers/strategies under feature flags).

---

```python
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict, field, replace
from datetime import date, datetime, timezone
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

        return SimpleNamespace(
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

    @settings.setter
    def settings(self, value: Any) -> None:  # type: ignore[override]
        """
        Setter 'דמי' כדי להרגיע את __init__ של dataclass אם הוא מנסה להציב settings.
        בפועל אנחנו מתעלמים מהערך, כי settings נגזר מ-self.config ו-ENV.
        """
        # אם תרצה בעתיד לשמור raw settings:
        # self._settings_raw = value
        # כרגע: מתעלמים.
        pass

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

                self.sql_store = SqlStore.from_settings(
                    self.settings,          # SimpleNamespace עם .config = config.json
                    env=self.environment,   # "dev" / "research" / "paper" / "live"
                    table_prefix="",        # אפשר להחליף אם תרצה prefix
                    read_only=False,        # או למשוך מ-config/env אם תרצה
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
