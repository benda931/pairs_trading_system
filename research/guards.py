# -*- coding: utf-8 -*-
"""
research/guards.py — Research-Only Guards
==========================================

Prevents research/experimental code from executing in production environments.

Usage:
    from research.guards import research_only, research_context, assert_not_production

    @research_only
    def experimental_strategy(prices):
        ...

    with research_context("Running experimental backtest"):
        ...

    assert_not_production("walk_forward_experimental")
"""
from __future__ import annotations

import functools
import logging
import os
from contextlib import contextmanager
from typing import Callable, Generator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class ResearchOnlyError(RuntimeError):
    """Raised when research-only code is executed in a production context."""


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

_PRODUCTION_ENV_VALUES = frozenset({"production", "prod", "live"})


def is_production_context() -> bool:
    """
    Returns True if any production environment variable is set.

    Checks (in order):
      - ENV
      - PAIRS_TRADING_ENV
      - ENVIRONMENT
    """
    for var in ("ENV", "PAIRS_TRADING_ENV", "ENVIRONMENT"):
        val = os.environ.get(var, "").strip().lower()
        if val in _PRODUCTION_ENV_VALUES:
            return True
    return False


def assert_not_production(context_description: str = "") -> None:
    """
    Raises ResearchOnlyError if current environment is production.

    Args:
        context_description: Human-readable description of what is being protected.
                             Included in the error message.

    Raises:
        ResearchOnlyError: If ENV/PAIRS_TRADING_ENV/ENVIRONMENT indicates production.
    """
    if is_production_context():
        env_val = (
            os.environ.get("ENV")
            or os.environ.get("PAIRS_TRADING_ENV")
            or os.environ.get("ENVIRONMENT")
            or "unknown"
        )
        msg = (
            f"[ResearchGuard] Research-only code invoked in production context "
            f"(env={env_val!r})"
        )
        if context_description:
            msg += f" — context: {context_description}"
        raise ResearchOnlyError(msg)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def research_only(func: Callable) -> Callable:
    """
    Decorator that raises ResearchOnlyError if the decorated function is called
    while ENV / PAIRS_TRADING_ENV / ENVIRONMENT is set to a production value.

    Example:
        @research_only
        def run_experimental_walk_forward(prices):
            ...
    """
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        assert_not_production(context_description=f"{func.__module__}.{func.__qualname__}")
        return func(*args, **kwargs)
    return _wrapper


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def research_context(description: str = "") -> Generator[None, None, None]:
    """
    Context manager that marks a code block as research-safe.
    Raises ResearchOnlyError if entered from a production context.

    Example:
        with research_context("Experimental cointegration scan"):
            run_johansen_scan(prices)
    """
    assert_not_production(context_description=description)
    logger.debug("[ResearchGuard] Entering research context: %s", description or "(unnamed)")
    try:
        yield
    finally:
        logger.debug("[ResearchGuard] Exiting research context: %s", description or "(unnamed)")


# ---------------------------------------------------------------------------
# Convenience: soft guard (warns instead of raising)
# ---------------------------------------------------------------------------

def warn_if_production(context_description: str = "") -> bool:
    """
    Logs a warning (but does NOT raise) if in production context.

    Returns:
        True if production context detected (caller may choose to skip).
        False if safe to proceed.
    """
    if is_production_context():
        env_val = (
            os.environ.get("ENV")
            or os.environ.get("PAIRS_TRADING_ENV")
            or os.environ.get("ENVIRONMENT")
            or "unknown"
        )
        logger.warning(
            "[ResearchGuard] Research code in production env=%r — context: %s",
            env_val,
            context_description or "(unspecified)",
        )
        return True
    return False
