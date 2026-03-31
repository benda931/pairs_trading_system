# -*- coding: utf-8 -*-
"""
runtime/environment.py — Pre-defined Environment Specifications
================================================================

Canonical EnvironmentSpec definitions for each named runtime environment.

Each spec encodes:
  - Which RuntimeMode the environment maps to.
  - Whether live capital and broker orders are permitted.
  - Allowed data sources and agent classes.
  - Approval and human-review thresholds.
  - Position-size caps and kill-switch auto-engage behaviour.

Usage::

    from runtime.environment import get_environment_spec, validate_environment_action

    spec = get_environment_spec("paper")
    allowed, reason = validate_environment_action("paper", "activate_strategy", "HIGH_RISK")

Design:
  - Specs are frozen dataclasses — never mutated at runtime.
  - Unknown environment names fall back to the "research" spec (most restrictive
    w.r.t. capital) so that missing configuration never opens an unsafe door.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from runtime.contracts import EnvironmentSpec, RuntimeMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk-class ordering (higher index = higher risk)
# ---------------------------------------------------------------------------

_RISK_CLASS_ORDER: list[str] = [
    "INFORMATIONAL",
    "BOUNDED_SAFE",
    "MEDIUM_RISK",
    "HIGH_RISK",
    "SENSITIVE",
]


def _risk_level(risk_class: str) -> int:
    """Return integer ordinal for a risk class string.  Unknown classes are treated
    as maximally risky so unknown values never sneak through a permissive check."""
    try:
        return _RISK_CLASS_ORDER.index(risk_class)
    except ValueError:
        return len(_RISK_CLASS_ORDER)  # unknown = maximum risk


# ---------------------------------------------------------------------------
# ENVIRONMENT_SPECS — canonical definitions
# ---------------------------------------------------------------------------

ENVIRONMENT_SPECS: dict[str, EnvironmentSpec] = {
    # ── Research ────────────────────────────────────────────────────────────
    "research": EnvironmentSpec(
        env_name="research",
        runtime_mode=RuntimeMode.RESEARCH,
        allow_live_capital=False,
        allow_broker_orders=False,
        allowed_data_sources=("sql", "fmp", "yahoo"),
        allowed_agent_classes=("research", "ml", "monitoring"),
        max_risk_class="MEDIUM_RISK",
        requires_approval_for=(),
        requires_human_review_above_risk="HIGH_RISK",
        audit_level="minimal",
        log_level="DEBUG",
        max_position_size_usd=0.0,
        kill_switch_auto_engage=False,
        notes=(
            "Offline research environment. No capital exposure. "
            "No broker connectivity. Used for hypothesis generation and "
            "model development."
        ),
    ),
    # ── Backtest ─────────────────────────────────────────────────────────────
    "backtest": EnvironmentSpec(
        env_name="backtest",
        runtime_mode=RuntimeMode.BACKTEST,
        allow_live_capital=False,
        allow_broker_orders=False,
        allowed_data_sources=("sql", "fmp", "yahoo"),
        allowed_agent_classes=("research", "ml", "monitoring"),
        max_risk_class="MEDIUM_RISK",
        requires_approval_for=(),
        requires_human_review_above_risk="HIGH_RISK",
        audit_level="minimal",
        log_level="DEBUG",
        max_position_size_usd=0.0,
        kill_switch_auto_engage=False,
        notes=(
            "Historical simulation environment. No capital exposure. "
            "Walk-forward and in-sample / out-of-sample testing runs here."
        ),
    ),
    # ── Paper ────────────────────────────────────────────────────────────────
    "paper": EnvironmentSpec(
        env_name="paper",
        runtime_mode=RuntimeMode.PAPER,
        allow_live_capital=False,
        allow_broker_orders=True,       # paper broker (IBKR paper account) only
        allowed_data_sources=("sql", "fmp", "ibkr_paper"),
        allowed_agent_classes=("research", "ml", "monitoring", "portfolio", "signal"),
        max_risk_class="HIGH_RISK",
        requires_approval_for=("activate_strategy", "promote_model"),
        requires_human_review_above_risk="SENSITIVE",
        audit_level="standard",
        log_level="INFO",
        max_position_size_usd=100_000.0,
        kill_switch_auto_engage=True,
        notes=(
            "Paper trading environment. No real capital. "
            "Full order routing to a paper broker account. "
            "Activation of strategies and model promotions require approval."
        ),
    ),
    # ── Shadow ───────────────────────────────────────────────────────────────
    "shadow": EnvironmentSpec(
        env_name="shadow",
        runtime_mode=RuntimeMode.SHADOW,
        allow_live_capital=False,
        allow_broker_orders=False,      # signals computed, but orders are NOT sent
        allowed_data_sources=("sql", "fmp", "ibkr_paper"),
        allowed_agent_classes=(
            "research",
            "ml",
            "monitoring",
            "portfolio",
            "signal",
            "governance",
        ),
        max_risk_class="HIGH_RISK",
        requires_approval_for=("activate_strategy", "promote_model"),
        requires_human_review_above_risk="SENSITIVE",
        audit_level="full",
        log_level="INFO",
        max_position_size_usd=0.0,
        kill_switch_auto_engage=True,
        notes=(
            "Shadow mode. All signals computed against live market data "
            "but orders are suppressed. Used for live dry-run before "
            "graduating to staging or live."
        ),
    ),
    # ── Staging ──────────────────────────────────────────────────────────────
    "staging": EnvironmentSpec(
        env_name="staging",
        runtime_mode=RuntimeMode.STAGING,
        allow_live_capital=False,
        allow_broker_orders=True,       # limited paper orders for integration testing
        allowed_data_sources=("sql", "fmp", "ibkr_paper"),
        allowed_agent_classes=(
            "research",
            "ml",
            "monitoring",
            "portfolio",
            "signal",
            "governance",
        ),
        max_risk_class="HIGH_RISK",
        requires_approval_for=(
            "activate_strategy",
            "promote_model",
            "alter_risk_limit",
        ),
        requires_human_review_above_risk="SENSITIVE",
        audit_level="full",
        log_level="INFO",
        max_position_size_usd=50_000.0,
        kill_switch_auto_engage=True,
        notes=(
            "Pre-production staging environment. Validates production readiness "
            "with constrained position sizes before full live deployment."
        ),
    ),
    # ── Live ─────────────────────────────────────────────────────────────────
    "live": EnvironmentSpec(
        env_name="live",
        runtime_mode=RuntimeMode.LIVE,
        allow_live_capital=True,
        allow_broker_orders=True,
        allowed_data_sources=("ibkr_live", "fmp"),
        allowed_agent_classes=("monitoring", "portfolio", "signal", "governance"),
        max_risk_class="SENSITIVE",
        requires_approval_for=(
            "activate_strategy",
            "promote_model",
            "alter_risk_limit",
            "enable_agent",
            "apply_override",
        ),
        requires_human_review_above_risk="HIGH_RISK",
        audit_level="full",
        log_level="WARNING",
        max_position_size_usd=500_000.0,
        kill_switch_auto_engage=True,
        notes=(
            "Live capital environment. All material actions require explicit "
            "approval. Kill switch auto-engages on any critical breach. "
            "Research and backtest agent classes are blocked."
        ),
    ),
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_environment_spec(env_name: str) -> EnvironmentSpec:
    """Return the EnvironmentSpec for the given environment name.

    Falls back to the "research" spec if the name is not recognised.
    The research spec is the safest fallback: no capital exposure,
    no broker orders, minimal permissions.

    Args:
        env_name: One of "research", "backtest", "paper", "shadow",
                  "staging", "live".

    Returns:
        The matching EnvironmentSpec, or the research spec as fallback.
    """
    spec = ENVIRONMENT_SPECS.get(env_name.lower().strip())
    if spec is None:
        logger.warning(
            "Unknown environment %r — falling back to 'research' spec", env_name
        )
        return ENVIRONMENT_SPECS["research"]
    return spec


def validate_environment_action(
    env_name: str, action_type: str, risk_class: str
) -> Tuple[bool, str]:
    """Check whether an action is permitted in the given environment.

    Rules applied (in order):
      1. If the action_type is explicitly in requires_approval_for, it is
         BLOCKED at this level — the caller must obtain approval first.
      2. If risk_class ordinal exceeds max_risk_class ordinal, BLOCKED.
      3. If risk_class ordinal >= requires_human_review_above_risk, BLOCKED
         (caller must obtain human review before proceeding).
      4. Otherwise ALLOWED.

    Args:
        env_name:    Target environment name.
        action_type: String identifier for the action, e.g. "activate_strategy".
        risk_class:  RiskClass string of the proposed action.

    Returns:
        Tuple of (allowed: bool, reason: str).
        reason is empty when allowed=True.
    """
    spec = get_environment_spec(env_name)

    # Rule 1 — action requires explicit approval
    if action_type in spec.requires_approval_for:
        return (
            False,
            (
                f"Action '{action_type}' requires explicit approval in "
                f"environment '{env_name}'."
            ),
        )

    # Rule 2 — risk class exceeds environment ceiling
    if _risk_level(risk_class) > _risk_level(spec.max_risk_class):
        return (
            False,
            (
                f"Action risk class '{risk_class}' exceeds the maximum permitted "
                f"risk class '{spec.max_risk_class}' for environment '{env_name}'."
            ),
        )

    # Rule 3 — risk class requires human review
    if _risk_level(risk_class) >= _risk_level(spec.requires_human_review_above_risk):
        return (
            False,
            (
                f"Action risk class '{risk_class}' requires human review in "
                f"environment '{env_name}' "
                f"(threshold: '{spec.requires_human_review_above_risk}')."
            ),
        )

    return True, ""


def list_environments() -> list[str]:
    """Return the list of known environment names."""
    return list(ENVIRONMENT_SPECS.keys())


def requires_live_capital(env_name: str) -> bool:
    """Return True if the environment allows (and uses) live capital."""
    return get_environment_spec(env_name).allow_live_capital


def allows_broker_orders(env_name: str) -> bool:
    """Return True if the environment allows broker order submission."""
    return get_environment_spec(env_name).allow_broker_orders
