"""
core/context_services.py — Service initialization for AppContext
================================================================

Extracted from app_context.py to reduce the god-object's size.

This module contains the service bootstrapping logic that was previously
embedded in AppContext.init_services() and AppContext.init_ib_router().
Each service is initialized independently with graceful degradation.

Usage (from app_context.py):
    from core.context_services import init_all_services
    init_all_services(ctx)
"""
from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass  # AppContext forward reference

logger = logging.getLogger(__name__)


# =====================================================================
# Individual service initializers
# =====================================================================

def _init_sql_store(ctx: Any) -> None:
    """Initialize SqlStore with environment-aware read-only policy."""
    if ctx.sql_store is not None:
        return
    try:
        from core.sql_store import SqlStore  # type: ignore

        cfg = getattr(ctx.settings, "config", {}) or {}
        cfg_ro = bool(cfg.get("sql_read_only", False))
        env_ro = os.getenv("SQL_STORE_READ_ONLY")
        env_ro_flag = env_ro is not None and env_ro.strip().lower() in ("1", "true", "yes", "on")

        if cfg_ro or env_ro_flag:
            read_only = True
        else:
            read_only = ctx.environment in ("paper", "live")

        ctx.sql_store = SqlStore.from_settings(
            ctx.settings, env=ctx.environment, table_prefix="", read_only=read_only,
        )
        ctx.services.setdefault("sql_store", ctx.sql_store)
        logger.info(
            "SqlStore initialised (url=%s, env=%s)",
            getattr(ctx.sql_store, "engine_url", None),
            getattr(ctx.sql_store, "default_env", None),
        )
    except Exception as exc:
        logger.warning("SqlStore init failed: %s", exc)


def _init_market_data_router(ctx: Any) -> None:
    """Initialize MarketDataRouter with session/legacy fallback."""
    if ctx.market_data_router is None:
        if ctx.md_router is not None:
            ctx.market_data_router = ctx.md_router
        else:
            try:
                import streamlit as st
                sess_md = st.session_state.get("md_router")
            except Exception:
                sess_md = None
            if sess_md is not None:
                ctx.market_data_router = sess_md
                ctx.md_router = sess_md

    if ctx.market_data_router is None:
        try:
            from common.market_data_router import build_default_router  # type: ignore
            router = build_default_router(ib=None, use_yahoo=True)
            ctx.market_data_router = router
            ctx.md_router = router
            logger.info("MarketDataRouter initialised via build_default_router(use_yahoo=True)")
        except Exception as exc:
            logger.debug("MarketDataRouter init skipped/failed: %s", exc)

    if ctx.market_data_router is not None:
        ctx.services.setdefault("market_data_router", ctx.market_data_router)


def _init_engines(ctx: Any) -> None:
    """Initialize domain engines (macro, risk, signals, fair value)."""
    engine_specs = [
        ("core.macro_engine", "macro_engine", ["macro_engine"]),
        ("core.risk_engine", "risk_engine", ["risk_engine"]),
        ("core.signals_engine", "signals_engine", ["signals_engine", "signal_engine", "signals", "signal_generator"]),
        ("core.fair_value_engine", "fair_value_engine", ["fair_value_engine"]),
    ]

    for module_path, attr_name, service_names in engine_specs:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            setattr(ctx, attr_name, mod)
            for sn in service_names:
                ctx.services.setdefault(sn, mod)
        except Exception:
            pass


def _init_fair_value_api(ctx: Any) -> None:
    """Initialize FairValueAPIConfig from config.json + ENV."""
    try:
        from core.fair_value_config import FairValueAPIConfig as FVCfg  # type: ignore
        fv_cfg = FVCfg.from_settings(ctx.config or {})
        ctx.fair_value_api = fv_cfg
        ctx.services.setdefault("fair_value_api", fv_cfg)
        logger.info(
            "FairValueAPIConfig: enabled=%s, profile=%s, base_url=%s",
            fv_cfg.enabled, fv_cfg.profile, fv_cfg.base_url,
        )
    except Exception as exc:
        logger.debug("FairValueAPIConfig init failed: %s", exc)


def _init_agents_manager(ctx: Any) -> None:
    """Initialize placeholder agents manager."""
    if ctx.agents_manager is not None:
        return
    try:
        ctx.agents_manager = SimpleNamespace(
            name="agents_manager_placeholder", online=True, status="online",
        )
        for name in ("agents_manager", "agents", "ai_agents"):
            ctx.services.setdefault(name, ctx.agents_manager)
    except Exception:
        pass


def _init_ib_router(ctx: Any) -> None:
    """Initialize IBKR order router with safety checks."""
    if ctx.ib_router is not None:
        return

    cfg = ctx.config or {}
    ib_cfg = cfg.get("ibkr") or {}
    enabled: bool = bool(ib_cfg.get("enabled", True))

    # Feature flag override
    ff = ctx.controls.get("feature_flags_snapshot", {}) if isinstance(ctx.controls, dict) else {}
    if isinstance(ff, dict) and "enable_ib_router" in ff:
        try:
            enabled = bool(ff.get("enable_ib_router"))
        except Exception:
            pass

    if not enabled:
        logger.info("IBOrderRouter init skipped: disabled via config/feature flag")
        return

    settings_obj = ctx.settings

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
        from core.ib_order_router import IBOrderRouter
        ctx.ib_router = IBOrderRouter(
            settings=settings_obj, use_singleton=True, profile=profile, readonly=readonly,
        )
        if hasattr(ctx.ib_router, "connect"):
            try:
                ctx.ib_router.connect()
            except Exception as exc:
                logger.warning("IBOrderRouter.connect() failed: %s", exc)

        ctx.broker = ctx.ib_router
        ctx.services.setdefault("ib_router", ctx.ib_router)
        ctx.services.setdefault("broker", ctx.broker)
        logger.info("IBOrderRouter initialised (profile=%s, readonly=%s)", profile, readonly)
    except Exception as exc:
        logger.warning("IBOrderRouter init failed: %s", exc)
        ctx.ib_router = None


# =====================================================================
# Main entry point
# =====================================================================

def init_all_services(ctx: Any) -> None:
    """
    Initialize all services on an AppContext instance.

    This is the canonical service bootstrapping function, extracted from
    AppContext.init_services() to reduce the god-object's size.

    Parameters
    ----------
    ctx : AppContext
        The context instance to initialize services on.
    """
    if getattr(ctx, "_services_initialized", False):
        return

    if not isinstance(ctx.services, dict):
        ctx.services = {}

    _init_sql_store(ctx)
    _init_market_data_router(ctx)
    _init_engines(ctx)
    _init_fair_value_api(ctx)
    _init_agents_manager(ctx)

    try:
        _init_ib_router(ctx)
    except Exception as exc:
        logger.warning("init_ib_router failed: %s", exc)

    ctx._services_initialized = True  # type: ignore[attr-defined]
