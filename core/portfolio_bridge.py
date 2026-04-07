# -*- coding: utf-8 -*-
"""
core/portfolio_bridge.py — Signal-to-Portfolio Bridge
=====================================================

This module is the canonical integration point between the signal pipeline
(core/signal_pipeline.py) and the portfolio allocator (portfolio/allocator.py).

It resolves P1-PORTINT ("PortfolioAllocator never receives real signals") and
P1-SAFE ("is_safe_to_trade() never called in execution paths").

Contract:
    SignalPipeline.evaluate() -> SignalDecision (with EntryIntent inside)
        |
    bridge_signals_to_allocator(safety_check=...)
        | 1. Check runtime safety (if callback provided)
        | 2. Filter non-blocked EntryIntents
        | 3. Enrich with quality/regime metadata
        | 4. Feed to PortfolioAllocator.run_cycle()
        |
    list[AllocationDecision], PortfolioDiagnostics

Safety gating (P1-SAFE):
    The ``safety_check`` parameter accepts an optional callback with signature
    ``() -> tuple[bool, list[str]]``.  The infrastructure layer (orchestrator,
    dashboard, control plane) injects this — for example:

        from runtime.state import get_runtime_state_manager
        mgr = get_runtime_state_manager()
        bridge_signals_to_allocator(decisions, safety_check=mgr.is_safe_to_trade)

    This preserves the architecture boundary: core/ does NOT import runtime/.
    The safety check is injected from outside, not hard-wired.

    When safety_check returns (False, reasons):
        - All new entries are BLOCKED
        - Diagnostics record the safety block with rationale
        - No allocation cycle runs
        - This is explicit, not silent

The bridge does NOT:
    - Size positions (that is the portfolio layer's job)
    - Execute trades (no execution layer exists yet)
    - Override portfolio risk constraints
    - Bypass the ranking/competition logic
    - Import from runtime/ or control_plane/ (architecture boundary)

Usage:
    from core.portfolio_bridge import bridge_signals_to_allocator

    # Without safety check (research/backtest mode)
    allocations, diag = bridge_signals_to_allocator(decisions, capital=1_000_000.0)

    # With safety check (operational mode)
    from runtime.state import get_runtime_state_manager
    allocations, diag = bridge_signals_to_allocator(
        decisions,
        capital=1_000_000.0,
        safety_check=get_runtime_state_manager().is_safe_to_trade,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

from core.contracts import PairId
from core.intents import EntryIntent
from core.signal_pipeline import SignalDecision

logger = logging.getLogger(__name__)

# Type alias for safety check callback.
# Signature: () -> (is_safe: bool, blocking_reasons: list[str])
# This is injected by the infrastructure layer — core/ never imports runtime/.
SafetyCheckFn = Callable[[], tuple[bool, list[str]]]


def extract_entry_intents(
    decisions: list[SignalDecision],
) -> list[EntryIntent]:
    """
    Extract non-blocked EntryIntents from a list of SignalDecisions.

    Each EntryIntent is enriched with ``quality_grade`` and ``regime``
    attributes so the portfolio ranker can score them without needing
    to look up the original SignalDecision.

    Parameters
    ----------
    decisions : list[SignalDecision]
        Output from SignalPipeline.evaluate() for multiple pairs.

    Returns
    -------
    list[EntryIntent]
        Intents eligible for portfolio competition. May be empty.
    """
    intents: list[EntryIntent] = []
    n_blocked = 0
    n_no_intent = 0
    n_not_entry = 0

    for d in decisions:
        if d.blocked:
            n_blocked += 1
            continue
        if d.intent is None:
            n_no_intent += 1
            continue
        if not isinstance(d.intent, EntryIntent):
            n_not_entry += 1
            continue

        intent = d.intent

        # Enrich the intent with metadata the ranker uses via getattr().
        # These are NOT part of the EntryIntent dataclass — they are set
        # as ad-hoc attributes.  The ranker reads them with getattr() and
        # falls back to defaults if missing.  This is the adapter layer.
        object.__setattr__(intent, "quality_grade", d.quality_grade)
        object.__setattr__(intent, "regime", d.regime)
        object.__setattr__(intent, "size_multiplier", d.size_multiplier)
        if "half_life" in d.metadata:
            hl = d.metadata["half_life"]
            if hl is not None:
                object.__setattr__(intent, "half_life_days", hl)

        intents.append(intent)

    if n_blocked or n_no_intent or n_not_entry:
        logger.info(
            "extract_entry_intents: %d eligible, %d blocked, %d no intent, %d non-entry",
            len(intents), n_blocked, n_no_intent, n_not_entry,
        )
    return intents


def bridge_signals_to_allocator(
    signal_decisions: list[SignalDecision],
    capital: float = 1_000_000.0,
    *,
    safety_check: Optional[SafetyCheckFn] = None,
    active_allocations=None,
    drawdown_state=None,
    throttle_state=None,
    kill_switch=None,
    heat_state=None,
    portfolio_vol: Optional[float] = None,
    current_drawdown: float = 0.0,
    vix_level: float = 20.0,
    sector_map: Optional[dict[str, str]] = None,
    cluster_map: Optional[dict[str, str]] = None,
    spread_vols: Optional[dict[str, float]] = None,
    hedge_ratios: Optional[dict[str, float]] = None,
):
    """
    End-to-end bridge: SignalDecision list -> PortfolioAllocator.run_cycle().

    Parameters
    ----------
    signal_decisions : list[SignalDecision]
        Raw output from SignalPipeline.evaluate() for each pair.
    capital : float
        Total capital for allocation (default $1M).
    safety_check : callable, optional
        Runtime safety gate.  Signature: ``() -> (bool, list[str])``.
        When provided and returning ``(False, reasons)``, all new entries
        are blocked and diagnostics record the safety block with rationale.
        Inject from infrastructure layer:
        ``safety_check=get_runtime_state_manager().is_safe_to_trade``
        This preserves the core/ architecture boundary — no runtime/ imports.
    sector_map, cluster_map : dict, optional
        Metadata for concentration enforcement.  If missing, sector/cluster
        constraints are **not enforced** — this is documented, not silent.
    Other parameters are forwarded directly to PortfolioAllocator.run_cycle().

    Returns
    -------
    tuple[list[AllocationDecision], PortfolioDiagnostics]
        Allocation decisions (funded and unfunded with rationale) and
        diagnostic summary of the allocation cycle.
    """
    from portfolio.contracts import PortfolioDiagnostics

    # ── Step 0: Runtime safety gate (P1-SAFE) ────────────────────
    if safety_check is not None:
        try:
            is_safe, blocking_reasons = safety_check()
        except Exception as e:
            logger.error("Safety check raised: %s — treating as UNSAFE", e)
            is_safe = False
            blocking_reasons = [f"safety_check_exception: {e}"]

        if not is_safe:
            logger.warning(
                "SAFETY BLOCK: %d signal decisions rejected — %s",
                len(signal_decisions),
                "; ".join(blocking_reasons),
            )
            return [], PortfolioDiagnostics(
                n_intents_received=len(signal_decisions),
                n_blocked_risk=len(signal_decisions),
                binding_constraints=[
                    f"SAFETY_BLOCK: {r}" for r in blocking_reasons
                ],
                kill_switch_mode="ACTIVE" if any(
                    "kill_switch" in r.lower() for r in blocking_reasons
                ) else "OFF",
            )

    from portfolio.allocator import PortfolioAllocator
    from portfolio.capital import CapitalManager

    # 1. Extract non-blocked EntryIntents
    intents = extract_entry_intents(signal_decisions)

    if not intents:
        logger.info(
            "bridge_signals_to_allocator: no eligible intents — returning empty cycle"
        )
        return [], PortfolioDiagnostics(
            n_intents_received=len(signal_decisions),
            n_blocked_signal=len(signal_decisions) - len(intents),
        )

    # 2. Create allocator with capital
    capital_mgr = CapitalManager(total_capital=capital)
    allocator = PortfolioAllocator(capital_mgr)

    # 3. Run allocation cycle
    allocations, diagnostics = allocator.run_cycle(
        intents,
        active_allocations=active_allocations,
        drawdown_state=drawdown_state,
        throttle_state=throttle_state,
        kill_switch=kill_switch,
        heat_state=heat_state,
        portfolio_vol=portfolio_vol,
        current_drawdown=current_drawdown,
        vix_level=vix_level,
        sector_map=sector_map,
        cluster_map=cluster_map,
        spread_vols=spread_vols,
        hedge_ratios=hedge_ratios,
    )

    # 4. Log summary
    n_funded = sum(1 for a in allocations if a.approved)
    n_unfunded = len(allocations) - n_funded
    logger.info(
        "bridge_signals_to_allocator: %d decisions (%d funded, %d unfunded) "
        "from %d intents out of %d signal decisions",
        len(allocations), n_funded, n_unfunded,
        len(intents), len(signal_decisions),
    )

    return allocations, diagnostics


class EnvironmentMismatchError(Exception):
    """Raised when portfolio environment and gateway environment are incompatible."""
    pass


def environment_check(portfolio_env: str, gateway_env: str) -> None:
    """
    Verify portfolio environment matches gateway environment before any order submission.

    A paper portfolio allocation must NEVER flow to a live gateway.
    A live portfolio allocation must NEVER flow to a paper gateway.

    This is a hard stop — raises EnvironmentMismatchError on any mismatch.
    Call this before forwarding any ExecutionIntent to the order router.

    Parameters
    ----------
    portfolio_env : str
        Environment of the portfolio allocation ("paper", "live", "research").
    gateway_env : str
        Environment of the target gateway ("paper", "live", "dry_run").

    Raises
    ------
    EnvironmentMismatchError
        If portfolio_env and gateway_env are incompatible.
    """
    LIVE_ENVS = {"live", "production", "prod"}
    PAPER_ENVS = {"paper", "paper_trading", "sim", "simulation"}

    portfolio_is_live = portfolio_env.lower() in LIVE_ENVS
    gateway_is_live = gateway_env.lower() in LIVE_ENVS
    portfolio_is_paper = portfolio_env.lower() in PAPER_ENVS
    gateway_is_paper = gateway_env.lower() in PAPER_ENVS

    if portfolio_is_live and not gateway_is_live:
        raise EnvironmentMismatchError(
            f"LIVE portfolio allocation targeted at non-live gateway '{gateway_env}'. "
            f"Live capital must only flow through a live gateway. "
            f"Check your gateway configuration."
        )
    if portfolio_is_paper and gateway_is_live:
        raise EnvironmentMismatchError(
            f"PAPER portfolio allocation targeted at LIVE gateway '{gateway_env}'. "
            f"Paper allocations must never submit to live broker. "
            f"This would execute paper-simulated orders with real money."
        )
