# -*- coding: utf-8 -*-
"""
core/portfolio_bridge.py — Signal-to-Portfolio Bridge
=====================================================

This module is the canonical integration point between the signal pipeline
(core/signal_pipeline.py) and the portfolio allocator (portfolio/allocator.py).

It resolves the P1-PORTINT finding: "PortfolioAllocator never receives real signals."

Contract:
    SignalPipeline.evaluate() → SignalDecision (with EntryIntent inside)
        ↓
    bridge_signals_to_allocator()
        ↓ filters non-blocked EntryIntents, enriches with quality/regime metadata
    PortfolioAllocator.run_cycle(intents=[...])
        ↓
    list[AllocationDecision], PortfolioDiagnostics

The bridge does NOT:
    - Size positions (that is the portfolio layer's job)
    - Execute trades (no execution layer exists yet)
    - Override portfolio risk constraints
    - Bypass the ranking/competition logic

Usage:
    from core.portfolio_bridge import bridge_signals_to_allocator

    decisions = [pipeline.evaluate(z, spread, px, py) for ...]
    allocations, diagnostics = bridge_signals_to_allocator(
        signal_decisions=decisions,
        capital=1_000_000.0,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from core.contracts import PairId
from core.intents import EntryIntent
from core.signal_pipeline import SignalDecision

logger = logging.getLogger(__name__)


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
    End-to-end bridge: SignalDecision list → PortfolioAllocator.run_cycle().

    Parameters
    ----------
    signal_decisions : list[SignalDecision]
        Raw output from SignalPipeline.evaluate() for each pair.
    capital : float
        Total capital for allocation (default $1M).
    sector_map, cluster_map : dict, optional
        Metadata for concentration enforcement.  If missing, sector/cluster
        constraints are **not enforced** — this is documented, not silent.
    Other parameters are forwarded directly to PortfolioAllocator.run_cycle().

    Returns
    -------
    tuple[list[AllocationDecision], PortfolioDiagnostics]
        Allocation decisions (funded and unfunded with rationale) and
        diagnostic summary of the allocation cycle.

    Notes
    -----
    If sector_map or cluster_map is None, concentration constraints are
    skipped by the allocator.  This is honest — the CLAUDE.md doctrine says:
    "UNKNOWN cluster/sector skips concentration enforcement."
    """
    from portfolio.allocator import PortfolioAllocator
    from portfolio.capital import CapitalManager

    # 1. Extract non-blocked EntryIntents
    intents = extract_entry_intents(signal_decisions)

    if not intents:
        logger.info("bridge_signals_to_allocator: no eligible intents — returning empty cycle")
        from portfolio.contracts import PortfolioDiagnostics
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
