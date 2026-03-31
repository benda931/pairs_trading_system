# -*- coding: utf-8 -*-
"""
portfolio/analytics.py — Portfolio Analytics & Diagnostics
===========================================================

Aggregates portfolio construction cycle outputs into:
  1. PortfolioSnapshot  — point-in-time state for downstream consumption
  2. PortfolioDiagnostics — cycle-level performance metrics
  3. Attribution        — which decisions drove capital allocation
  4. Opportunity funnel — conversion rates at each stage

Usage:
    analytics = PortfolioAnalytics()
    snapshot = analytics.build_snapshot(capital_mgr, dd_state, ks_state, decisions)
    report = analytics.summarise_cycle(opportunity_set, decisions, diagnostics)
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from portfolio.capital import CapitalManager
from portfolio.contracts import (
    AllocationDecision,
    AllocationOutcome,
    DrawdownState,
    ExposureSummary,
    HeatState,
    KillSwitchState,
    OpportunitySet,
    PortfolioAuditRecord,
    PortfolioDiagnostics,
    PortfolioSnapshot,
    ThrottleState,
)
from portfolio.risk_ops import DrawdownManager, KillSwitchManager

logger = logging.getLogger("portfolio.analytics")


# ── Funnel report ─────────────────────────────────────────────────

@dataclass
class OpportunityFunnelReport:
    """Conversion funnel from intents received → capital allocated."""
    cycle_id: str = ""
    n_intents_in: int = 0
    n_blocked_signal: int = 0   # quality=F, regime CRISIS/BROKEN
    n_blocked_overlap: int = 0  # shared-leg / cluster
    n_blocked_risk: int = 0     # sector, leverage, cluster concentration
    n_blocked_capital: int = 0  # insufficient capital
    n_unfunded: int = 0         # passed all checks but scored too low
    n_queued: int = 0           # cycle limit
    n_partial: int = 0          # funded but scaled down
    n_funded: int = 0           # fully funded
    capital_deployed: float = 0.0
    capital_available: float = 0.0
    top_blockers: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.n_intents_in == 0:
            return 0.0
        return (self.n_funded + self.n_partial) / self.n_intents_in

    @property
    def deployment_rate(self) -> float:
        if self.capital_available <= 0:
            return 0.0
        return self.capital_deployed / self.capital_available

    def to_dict(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "n_intents_in": self.n_intents_in,
            "n_funded": self.n_funded,
            "n_partial": self.n_partial,
            "n_blocked_signal": self.n_blocked_signal,
            "n_blocked_overlap": self.n_blocked_overlap,
            "n_blocked_risk": self.n_blocked_risk,
            "n_blocked_capital": self.n_blocked_capital,
            "n_queued": self.n_queued,
            "pass_rate": round(self.pass_rate, 4),
            "deployment_rate": round(self.deployment_rate, 4),
            "top_blockers": self.top_blockers[:5],
        }


# ── Position attribution ──────────────────────────────────────────

@dataclass
class PositionAttribution:
    """Attribution of a single funded position."""
    pair_label: str
    capital_allocated: float
    weight: float
    rank: int
    composite_score: float
    key_driver: str           # Highest scoring dimension
    limiting_factor: str      # What capped the size
    sleeve: str


# ── Portfolio Analytics ───────────────────────────────────────────

class PortfolioAnalytics:
    """
    Builds snapshots and reports from portfolio construction outputs.

    Stateless: pass current state to each method.
    """

    def build_snapshot(
        self,
        capital_manager: CapitalManager,
        drawdown_state: DrawdownState,
        kill_switch_state: KillSwitchState,
        active_allocations: list[AllocationDecision],
        pending_allocations: Optional[list[AllocationDecision]] = None,
        exposure_summary: Optional[ExposureSummary] = None,
    ) -> PortfolioSnapshot:
        """
        Build a complete PortfolioSnapshot from current state.

        This is the canonical output object passed to downstream layers
        (execution, monitoring, risk reporting).
        """
        pending_allocations = pending_allocations or []
        pool = capital_manager.pool_snapshot()

        # Build HeatState
        gross_exp = exposure_summary.gross_exposure if exposure_summary else 0.0
        net_exp = exposure_summary.net_exposure if exposure_summary else 0.0
        total_cap = pool.total_capital
        consumed_risk = sum(
            d.sizing.risk_contribution for d in active_allocations if d.approved
        )

        heat_state = HeatState(
            total_risk_budget=1.0,
            consumed_risk=consumed_risk,
            reserved_risk=0.0,
            free_risk=max(0.0, 1.0 - consumed_risk),
            heat_pct=min(1.0, consumed_risk),
            n_active_positions=len([d for d in active_allocations if d.approved]),
            n_pending_orders=len(pending_allocations),
            gross_exposure=gross_exp,
            net_exposure=net_exp,
            gross_leverage=gross_exp / max(1.0, total_cap),
            net_leverage=abs(net_exp) / max(1.0, total_cap),
        )

        throttle_state = ThrottleState(
            heat_level=drawdown_state.heat_level,
            size_multiplier=drawdown_state.throttle_factor,
            max_positions=drawdown_state.max_new_positions,
        )

        return PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            capital_pool=pool,
            heat_state=heat_state,
            drawdown_state=drawdown_state,
            throttle_state=throttle_state,
            kill_switch=kill_switch_state,
            exposure=exposure_summary or ExposureSummary(total_capital=total_cap),
            active_allocations=active_allocations,
            pending_allocations=pending_allocations,
        )

    def build_diagnostics(
        self,
        cycle_id: str,
        opportunity_set: OpportunitySet,
        decisions: list[AllocationDecision],
        snapshot: PortfolioSnapshot,
    ) -> PortfolioDiagnostics:
        """
        Build PortfolioDiagnostics from cycle outputs.
        """
        funded = [d for d in decisions if d.approved]
        unfunded = [d for d in decisions if not d.approved]

        # Count by outcome
        outcome_counts: dict[str, int] = Counter(
            d.rationale.outcome.value for d in unfunded
        )

        # Binding constraints
        binding: list[str] = []
        for d in unfunded:
            for v in d.constraint_result.hard_violations:
                binding.append(v.constraint_type.value)
        binding_unique = list(dict.fromkeys(binding))[:5]

        # Top unfunded reasons
        top_unfunded: dict[str, str] = {}
        for d in unfunded[:10]:
            notes = d.rationale.decision_notes
            top_unfunded[d.pair_id.label] = notes[0] if notes else d.rationale.outcome.value

        diag = PortfolioDiagnostics(
            cycle_id=cycle_id,
            n_intents_received=opportunity_set.n_input_intents,
            n_qualified=len(opportunity_set.opportunities),
            n_ranked=len(opportunity_set.opportunities),
            n_funded=len(funded),
            n_blocked_signal=sum(
                1 for d in unfunded
                if d.rationale.outcome == AllocationOutcome.BLOCKED_SIGNAL
            ),
            n_blocked_risk=sum(
                1 for d in unfunded
                if d.rationale.outcome in (
                    AllocationOutcome.BLOCKED_RISK,
                    AllocationOutcome.BLOCKED_OVERLAP,
                    AllocationOutcome.BLOCKED_REGIME,
                )
            ),
            n_blocked_capital=sum(
                1 for d in unfunded
                if d.rationale.outcome == AllocationOutcome.BLOCKED_CAPITAL
            ),
            n_blocked_overlap=sum(
                1 for d in unfunded
                if d.rationale.outcome == AllocationOutcome.BLOCKED_OVERLAP
            ),
            capital_allocated=snapshot.capital_pool.allocated_capital,
            capital_reserved=snapshot.capital_pool.reserved_capital,
            capital_free=snapshot.capital_pool.free_capital,
            utilisation_pct=snapshot.capital_pool.utilisation_pct,
            portfolio_heat=snapshot.heat_state.heat_pct,
            gross_leverage=snapshot.heat_state.gross_leverage,
            net_leverage=snapshot.heat_state.net_leverage,
            max_sector_concentration=snapshot.exposure.max_sector_concentration,
            max_cluster_concentration=snapshot.exposure.max_cluster_concentration,
            n_shared_leg_alerts=sum(1 for s in snapshot.exposure.shared_legs if s.is_dominant),
            heat_level=snapshot.drawdown_state.heat_level.value,
            kill_switch_mode=snapshot.kill_switch.mode.value,
            n_hard_violations=sum(len(d.constraint_result.hard_violations) for d in unfunded),
            n_soft_violations=sum(len(d.constraint_result.soft_violations) for d in decisions),
            binding_constraints=binding_unique,
            top_funded=[d.pair_id.label for d in funded[:10]],
            top_unfunded_reasons=top_unfunded,
        )
        return diag

    def build_funnel_report(
        self,
        cycle_id: str,
        opportunity_set: OpportunitySet,
        decisions: list[AllocationDecision],
        capital_available: float,
    ) -> OpportunityFunnelReport:
        """Build the opportunity conversion funnel report."""
        funded = [d for d in decisions if d.approved and d.rationale.outcome == AllocationOutcome.FUNDED]
        partial = [d for d in decisions if d.approved and d.rationale.outcome == AllocationOutcome.PARTIAL_FUNDED]

        # Classify unfunded
        n_signal, n_overlap, n_risk, n_capital, n_unfunded, n_queued = 0, 0, 0, 0, 0, 0
        blocker_counts: Counter = Counter()

        for d in decisions:
            if d.approved:
                continue
            outcome = d.rationale.outcome
            if outcome == AllocationOutcome.BLOCKED_SIGNAL:
                n_signal += 1
            elif outcome == AllocationOutcome.BLOCKED_OVERLAP:
                n_overlap += 1
            elif outcome in (AllocationOutcome.BLOCKED_RISK, AllocationOutcome.BLOCKED_REGIME):
                n_risk += 1
            elif outcome == AllocationOutcome.BLOCKED_CAPITAL:
                n_capital += 1
            elif outcome == AllocationOutcome.QUEUED:
                n_queued += 1
            else:
                n_unfunded += 1

            for note in d.rationale.decision_notes:
                blocker_counts[note.split(":")[0]] += 1

        capital_deployed = sum(d.approved_capital for d in decisions if d.approved)

        return OpportunityFunnelReport(
            cycle_id=cycle_id,
            n_intents_in=opportunity_set.n_input_intents,
            n_blocked_signal=n_signal,
            n_blocked_overlap=n_overlap,
            n_blocked_risk=n_risk,
            n_blocked_capital=n_capital,
            n_unfunded=n_unfunded,
            n_queued=n_queued,
            n_partial=len(partial),
            n_funded=len(funded),
            capital_deployed=capital_deployed,
            capital_available=capital_available,
            top_blockers=[k for k, _ in blocker_counts.most_common(5)],
        )

    def build_position_attributions(
        self,
        decisions: list[AllocationDecision],
    ) -> list[PositionAttribution]:
        """Attribute funded positions to key drivers."""
        attributions = []
        for d in decisions:
            if not d.approved:
                continue
            opp = d.proposal.ranked_opportunity
            sizing = d.sizing

            # Identify key driver (highest score dimension)
            scores = {
                "signal_strength": opp.signal_strength_score,
                "signal_quality": opp.signal_quality_score,
                "regime_suitability": opp.regime_suitability_score,
                "reversion_probability": opp.reversion_probability,
                "diversification": opp.diversification_value,
                "stability": opp.stability_score,
            }
            key_driver = max(scores, key=lambda k: scores[k])

            # Limiting factor
            limiting = "none"
            if sizing.was_capped:
                limiting = sizing.cap_reason
            elif opp.overlap_penalty > 0.1:
                limiting = f"overlap:{opp.overlap_penalty:.2f}"
            elif opp.regime_suitability_score < 0.5:
                limiting = f"regime:{opp.regime}"

            attributions.append(PositionAttribution(
                pair_label=d.pair_id.label,
                capital_allocated=d.approved_capital,
                weight=d.approved_weight,
                rank=opp.rank,
                composite_score=opp.composite_score,
                key_driver=key_driver,
                limiting_factor=limiting,
                sleeve=d.proposal.sleeve,
            ))

        return sorted(attributions, key=lambda a: a.rank)

    def build_audit_record(
        self,
        cycle_id: str,
        started_at: datetime,
        snapshot_before: PortfolioSnapshot,
        snapshot_after: PortfolioSnapshot,
        decisions: list[AllocationDecision],
        diagnostics: PortfolioDiagnostics,
    ) -> PortfolioAuditRecord:
        """Build the master audit record for one cycle."""
        return PortfolioAuditRecord(
            cycle_id=cycle_id,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
            diagnostics=diagnostics,
            allocation_decisions=decisions,
        )
