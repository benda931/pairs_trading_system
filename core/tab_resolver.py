# -*- coding: utf-8 -*-
"""
core/tab_resolver.py — Tab Renderer Discovery, Lazy Import & Adaptive Call
==========================================================================

Extracted from root/dashboard.py (Part 6/35).

Provides the tab resolution infrastructure:
- TAB_RESOLUTION_SPEC manifest mapping logical tab names to candidate modules/functions.
- Lazy import and resolution of tab renderer functions.
- Adaptive invocation with signature introspection.
- Caching and diagnostics (stats, warm-up, cache clearing).
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from core.app_context import AppContext  # pragma: no cover
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Type aliases (mirrored from dashboard.py to keep this module self-contained)
# ---------------------------------------------------------------------------

FeatureFlags = Dict[str, Any]
NavPayload = Dict[str, Any]

from typing import Callable as _CallableT

_TabFn = _CallableT[..., Any]

# Defaults (same values as dashboard.py)
DEFAULT_ENV = "dev"
DEFAULT_PROFILE = "trading"

logger = logging.getLogger("Dashboard")

# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

_TAB_FN_CACHE: Dict[str, _TabFn] = {}
_TAB_RESOLUTION_STATS: Dict[str, "TabResolutionResult"] = {}
_FAILED_TAB_CACHE: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# TabResolutionResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class TabResolutionResult:
    """
    Result of resolving a single tab — used for diagnostics and dev tools.
    """
    logical_name: str
    module_name: str | None
    func_name: str | None
    success: bool
    import_time_ms: float
    notes: str = ""


# ---------------------------------------------------------------------------
# TAB_RESOLUTION_SPEC — central manifest
# ---------------------------------------------------------------------------

TAB_RESOLUTION_SPEC: Dict[str, Dict[str, Sequence[str]]] = {
    "home": {
        "modules": (
            "dashboard_home_v2",
            "root.dashboard_home_v2",
            "root.dashboard_home",
            "dashboard_home",
        ),
        "funcs": (
            "render_dashboard_home_v2",
            "render_dashboard_home",
            "render_home_tab",
            "render_home",
        ),
    },
    "smart_scan": {
        "modules": (
            "smart_scan_tab",
            "root.smart_scan_tab",
        ),
        "funcs": (
            "render_smart_scan_tab",
            "render_tab",
            "render_smart_scan",
        ),
    },
    "pair": {
        "modules": (
            "pair_tab",
            "root.pair_tab",
        ),
        "funcs": (
            "render_pair_tab",
            "render_tab",
            "render_pair_analysis_tab",
        ),
    },
    "matrix": {
        "modules": (
            "matrix_research_tab",
            "root.matrix_research_tab",
            "matrix_tab",
            "root.matrix_tab",
        ),
        "funcs": (
            "render_matrix_research_tab",
            "render_matrix_tab",
            "render_tab",
        ),
    },
    "comparison_matrices": {
        "modules": (
            "tab_comparison_matrices",
            "root.tab_comparison_matrices",
        ),
        "funcs": (
            "render_comparison_matrices_tab",
            "render_tab",
        ),
    },
    "backtest": {
        "modules": (
            "backtest_tab",
            "root.backtest_tab",
            "backtest",
            "root.backtest",
        ),
        "funcs": (
            "render_backtest_tab_v3",
            "render_backtest_tab_v2",
            "render_backtest_tab",
            "render_backtest",
            "render_tab",
        ),
    },
    "optimization": {
        "modules": (
            "optimization_tab",
            "root.optimization_tab",
        ),
        "funcs": (
            "render_optimization_tab",
            "render_tab",
        ),
    },
    "insights": {
        "modules": (
            "insights",
            "root.insights",
            "insights_tab",
            "root.insights_tab",
        ),
        "funcs": (
            "render_insights_tab",
            "render_tab",
        ),
    },
    "macro": {
        "modules": (
            "macro_tab",
            "root.macro_tab",
            "macro",
            "root.macro",
        ),
        "funcs": (
            "render_macro_tab",
            "render_tab",
        ),
    },
    "portfolio": {
        "modules": (
            "portfolio_tab",
            "root.portfolio_tab",
            "fund_view_tab",
            "root.fund_view_tab",
        ),
        "funcs": (
            "render_portfolio_tab",
            "render_fund_view_tab",
            "render_tab",
        ),
    },
    "risk": {
        "modules": (
            "risk_tab",
            "root.risk_tab",
        ),
        "funcs": (
            "render_risk_tab",
            "render_tab",
        ),
    },
    "fair_value": {
        "modules": (
            "fair_value_api_tab",
            "root.fair_value_api_tab",
            "fair_value_tab",
            "root.fair_value_tab",
        ),
        "funcs": (
            "render_fair_value_tab",
            "render_fair_value_api_tab",
            "render_tab",
        ),
    },
    "config": {
        "modules": (
            "config_tab",
            "root.config_tab",
        ),
        "funcs": (
            "render_config_tab",
            "render_tab",
        ),
    },
    "agents": {
        "modules": (
            "agents_tab",
            "root.agents_tab",
        ),
        "funcs": (
            "render_agents_tab",
            "render_tab",
        ),
    },
    "logs": {
        "modules": (
            "logs_tab",
            "root.logs_tab",
            "system_health_tab",
            "root.system_health_tab",
        ),
        "funcs": (
            "render_logs_tab",
            "render_system_health_tab",
            "render_tab",
        ),
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _is_strict_tab_mode() -> bool:
    """
    Whether to run in "strict" mode for tabs.
    In strict mode, missing module/renderer => st.error instead of st.warning.
    Based on env vars PAIRS_STRICT_TABS / DASHBOARD_STRICT_TABS.
    """
    val = os.getenv("PAIRS_STRICT_TABS") or os.getenv("DASHBOARD_STRICT_TABS")
    if not val:
        return False
    return val.strip().lower() not in ("0", "false", "no", "off")


def _find_module(
    module_candidates: Sequence[str],
    logical_name: str | None = None,
) -> Optional[Any]:
    """
    Try to locate and import a module from a list of candidate names.
    Uses importlib.util.find_spec to check existence, then importlib.import_module.
    Updates _TAB_RESOLUTION_STATS and _FAILED_TAB_CACHE.
    """
    if logical_name and logical_name in _FAILED_TAB_CACHE:
        return None

    candidates = list(module_candidates)
    if not candidates and logical_name:
        spec_entry = TAB_RESOLUTION_SPEC.get(logical_name, {})
        candidates = list(spec_entry.get("modules", []))

    if not candidates:
        return None

    selected_module = None
    selected_name: str | None = None
    import_time_ms: float = 0.0

    for mod_name in candidates:
        try:
            spec = importlib.util.find_spec(mod_name)
        except Exception as exc:  # pragma: no cover
            logger.debug("find_spec(%s) failed: %s", mod_name, exc)
            continue

        if spec is None:
            continue

        t0 = time.perf_counter()
        try:
            module = importlib.import_module(mod_name)
            t1 = time.perf_counter()
            import_time_ms = (t1 - t0) * 1000.0
            logger.info("Imported tab module '%s' (%.2f ms)", mod_name, import_time_ms)
            selected_module = module
            selected_name = mod_name
            break
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to import module '%s': %s", mod_name, exc)
            continue

    if logical_name:
        _TAB_RESOLUTION_STATS[logical_name] = TabResolutionResult(
            logical_name=logical_name,
            module_name=selected_name,
            func_name=None,
            success=selected_module is not None,
            import_time_ms=import_time_ms,
            notes="" if selected_module is not None else "no module matched candidates",
        )

        if selected_module is None:
            _FAILED_TAB_CACHE[logical_name] = "module_import_failed"

    return selected_module


def _find_tab_function_in_module(
    module: Any,
    func_candidates: Sequence[str],
    logical_name: str | None = None,
) -> Optional[_TabFn]:
    """
    Search for a renderer function in a module by candidate names.
    Falls back to heuristic: single render_* function.
    Updates _TAB_RESOLUTION_STATS.
    """
    resolved_fn: Optional[_TabFn] = None
    resolved_name: str | None = None

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
            resolved_fn = fn  # type: ignore[assignment]
            resolved_name = func_name
            break

    # Heuristic fallback: single render_* function
    if resolved_fn is None:
        try:
            render_like = [
                name
                for name, obj in vars(module).items()
                if name.startswith("render_") and callable(obj)
            ]
            if len(render_like) == 1:
                resolved_name = render_like[0]
                resolved_fn = getattr(module, resolved_name)
                logger.info(
                    "Heuristic: using '%s.%s' as tab renderer (single render_* function found).",
                    getattr(module, "__name__", "<unknown>"),
                    resolved_name,
                )
        except Exception:
            resolved_fn = None  # pragma: no cover

    if logical_name:
        stats = _TAB_RESOLUTION_STATS.get(logical_name)
        if stats is None:
            _TAB_RESOLUTION_STATS[logical_name] = TabResolutionResult(
                logical_name=logical_name,
                module_name=getattr(module, "__name__", None),
                func_name=resolved_name,
                success=resolved_fn is not None,
                import_time_ms=0.0,
                notes="function_resolved" if resolved_fn is not None else "no render function found",
            )
        else:
            stats.func_name = resolved_name
            stats.success = stats.success and resolved_fn is not None
            if resolved_fn is None:
                stats.notes = (stats.notes or "") + " | no render function matched candidates"

    return resolved_fn


def _resolve_tab_function(
    logical_name: str,
    module_candidates: Sequence[str],
    func_candidates: Sequence[str],
) -> Optional[_TabFn]:
    """
    Locate a renderer function for a tab by logical name.
    Uses caching (positive and negative) and the TAB_RESOLUTION_SPEC manifest.
    """
    if logical_name in _TAB_FN_CACHE:
        return _TAB_FN_CACHE[logical_name]

    if logical_name in _FAILED_TAB_CACHE:
        return None

    spec_entry = TAB_RESOLUTION_SPEC.get(logical_name, {})
    if not module_candidates:
        module_candidates = tuple(spec_entry.get("modules", ()))
    if not func_candidates:
        func_candidates = tuple(spec_entry.get("funcs", ()))

    module = _find_module(module_candidates, logical_name=logical_name)
    if module is None:
        logger.warning(
            "No module found for tab '%s' (candidates=%s)",
            logical_name,
            list(module_candidates),
        )
        _FAILED_TAB_CACHE[logical_name] = "module_not_found"
        return None

    fn = _find_tab_function_in_module(module, func_candidates, logical_name=logical_name)
    if fn is None:
        logger.warning(
            "No renderer function found for tab '%s' in module '%s' (candidates=%s)",
            logical_name,
            getattr(module, "__name__", "<unknown>"),
            list(func_candidates),
        )
        _FAILED_TAB_CACHE[logical_name] = "renderer_not_found"
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
    Build a smart kwargs dict for a tab function based on its signature.
    Supports various parameter naming conventions (app_ctx, ctx, context, etc.).
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
            continue
        else:
            logger.debug(
                "Tab renderer %s has unhandled required parameter '%s' – "
                "not passed by dashboard router.",
                getattr(fn, "__name__", "<anonymous>"),
                name,
            )
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
    Invoke a tab function with adaptive signature matching.
    Tries kwargs first, then falls back to positional argument variants.
    """
    kwargs = _build_call_kwargs_for_tab_fn(fn, app_ctx, feature_flags, nav_payload)

    t0 = time.perf_counter()
    try:
        fn(**kwargs)
        return
    except TypeError as exc:
        logger.debug(
            "Primary call of tab '%s' with kwargs failed (%s); trying fallbacks",
            logical_name,
            exc,
        )
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
        raise
    finally:
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        logger.debug(
            "Tab '%s' renderer %s executed in %.2f ms",
            logical_name,
            getattr(fn, "__name__", "<anonymous>"),
            elapsed_ms,
        )


def _lazy_render_tab(
    logical_name: str,
    module_candidates: Sequence[str],
    func_candidates: Sequence[str],
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload],
) -> None:
    """
    General helper used by all tabs:
    - Locates/imports module and tab function once (with cache).
    - Shows friendly message if not found (respects strict mode).
    - Invokes adaptively with signature support.
    """
    fn = _resolve_tab_function(logical_name, module_candidates, func_candidates)
    if fn is None:
        msg = (
            f"\u26a0\ufe0f Tab '{logical_name}' is not available "
            f"(module/renderer function not found)."
        )
        if _is_strict_tab_mode():
            st.error(msg)
        else:
            st.warning(msg)
        return

    _invoke_tab_function(logical_name, fn, app_ctx, feature_flags, nav_payload)


# ---------------------------------------------------------------------------
# Diagnostics / Agents / Desktop APIs
# ---------------------------------------------------------------------------

def get_tab_resolution_stats() -> Dict[str, TabResolutionResult]:
    """Return a copy of tab resolution stats for diagnostics."""
    return _TAB_RESOLUTION_STATS.copy()


def clear_tab_resolution_caches() -> None:
    """Clear all resolution caches (positive, stats, and negative)."""
    _TAB_FN_CACHE.clear()
    _TAB_RESOLUTION_STATS.clear()
    _FAILED_TAB_CACHE.clear()
    logger.info("Tab resolution caches cleared (fn_cache + stats + failed_cache).")


def warmup_tab_resolvers(
    logical_names: Optional[Sequence[str]] = None,
) -> Dict[str, TabResolutionResult]:
    """
    Warm up tab resolvers by resolving modules and functions without invoking them.
    If logical_names is None, resolves all tabs in TAB_RESOLUTION_SPEC.
    """
    if logical_names is None:
        logical_names = list(TAB_RESOLUTION_SPEC.keys())

    results: Dict[str, TabResolutionResult] = {}

    for name in logical_names:
        spec_entry = TAB_RESOLUTION_SPEC.get(name, {})
        modules = tuple(spec_entry.get("modules", ()))
        funcs = tuple(spec_entry.get("funcs", ()))
        if not modules and not funcs:
            continue
        module = _find_module(modules, logical_name=name)
        if module is not None:
            _ = _find_tab_function_in_module(module, funcs, logical_name=name)
        res = _TAB_RESOLUTION_STATS.get(name)
        if res is not None:
            results[name] = res

    return results


__all__ = [
    "_TAB_FN_CACHE",
    "_TAB_RESOLUTION_STATS",
    "_FAILED_TAB_CACHE",
    "TAB_RESOLUTION_SPEC",
    "TabResolutionResult",
    "_is_strict_tab_mode",
    "_find_module",
    "_find_tab_function_in_module",
    "_resolve_tab_function",
    "_build_call_kwargs_for_tab_fn",
    "_invoke_tab_function",
    "_lazy_render_tab",
    "get_tab_resolution_stats",
    "clear_tab_resolution_caches",
    "warmup_tab_resolvers",
]
