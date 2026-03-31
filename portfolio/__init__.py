# -*- coding: utf-8 -*-
"""
portfolio/ — Institutional Portfolio Construction & Risk Operating Model
=========================================================================

This package implements the allocation layer that sits between the signal
engine and the execution layer.

Layer stack (top to bottom):
  1. Signal layer  →  approved EntryIntent objects with quality/regime context
  2. Portfolio layer (THIS PACKAGE) →
       opportunity_ranking  — rank and score competing intents
       capital             — capital pools, sleeves, budget tracking
       sizing              — spread-level size / risk-budget calculation
       exposures           — shared-leg, sector, factor, cluster overlap
       allocator           — combines ranking + sizing + constraints
       risk_ops            — drawdown states, throttling, kill-switch
       analytics           — diagnostics, reports, attribution
  3. Execution layer  →  order translation, fills, position updates

Key doctrine (see docs/portfolio_architecture.md for full discussion):
  - A good pair is not automatically a good portfolio position
  - Capital is scarce; signals compete for it
  - Diversification ≠ number of positions
  - Pair-level neutrality ≠ portfolio-level neutrality
  - Every allocation decision must be auditable
"""

from portfolio.contracts import (
    AllocationDecision,
    AllocationProposal,
    AllocationRationale,
    CapitalBudget,
    CapitalPool,
    CapitalRecycleDecision,
    ClusterExposureSummary,
    ConstraintViolation,
    DeRiskingDecision,
    DrawdownState,
    ExposureContribution,
    ExposureSummary,
    HeatState,
    KillSwitchState,
    OpportunitySet,
    PortfolioDiagnostics,
    PortfolioHeatLevel,
    PortfolioSnapshot,
    RankedOpportunity,
    RebalanceDecision,
    RiskConstraintResult,
    SharedLegSummary,
    SizingDecision,
    SleeveDef,
    ThrottleState,
)

__all__ = [
    "AllocationDecision",
    "AllocationProposal",
    "AllocationRationale",
    "CapitalBudget",
    "CapitalPool",
    "CapitalRecycleDecision",
    "ClusterExposureSummary",
    "ConstraintViolation",
    "DeRiskingDecision",
    "DrawdownState",
    "ExposureContribution",
    "ExposureSummary",
    "HeatState",
    "KillSwitchState",
    "OpportunitySet",
    "PortfolioDiagnostics",
    "PortfolioHeatLevel",
    "PortfolioSnapshot",
    "RankedOpportunity",
    "RebalanceDecision",
    "RiskConstraintResult",
    "SharedLegSummary",
    "SizingDecision",
    "SleeveDef",
    "ThrottleState",
]
