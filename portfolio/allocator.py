# -*- coding: utf-8 -*-
"""
portfolio/allocator.py — Portfolio Allocator (Main Engine)
==========================================================

The central engine of the portfolio construction layer.

Allocation pipeline (per cycle):
  1. receive(intents)              — EntryIntent list from signal layer
  2. rank(intents)                 → OpportunitySet (via OpportunityRanker)
  3. for each fundable opportunity:
       a. size(opportunity)        → SizingDecision (via SizingEngine)
       b. check_constraints(sizing) → RiskConstraintResult
       c. if approved: allocate capital (via CapitalManager)
       d. record AllocationDecision
  4. return PortfolioSnapshot (via PortfolioAnalytics)

Constraint enforcement order (fast-path):
  1. Kill-switch HARD → block all new entries
  2. Heat level HALTED / RECOVERY_ONLY → block new entries
  3. Capital budget → BLOCKED_CAPITAL
  4. Exposure limits (sector, cluster, leverage, shared-leg) → BLOCKED_RISK
  5. Sleeve constraints (regime-blocked, min-quality) → BLOCKED_SIGNAL
  6. Per-position limits (max_single_pair) → partial approval

All decisions — approved AND rejected — are recorded with full rationale.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.contracts import PairId
from core.intents import EntryIntent
from portfolio.capital import CapitalManager
from portfolio.contracts import (
    AllocationDecision,
    AllocationOutcome,
    AllocationProposal,
    AllocationRationale,
    CapitalPool,
    ConstraintType,
    ConstraintViolation,
    DeRiskingDecision,
    DrawdownState,
    ExposureSummary,
    HeatState,
    KillSwitchState,
    OpportunitySet,
    PortfolioDiagnostics,
    PortfolioHeatLevel,
    PortfolioSnapshot,
    RankedOpportunity,
    RiskConstraintResult,
    SizingDecision,
    ThrottleState,
)
from portfolio.exposures import ExposureAnalyzer, ExposureConfig
from portfolio.ranking import OpportunityRanker, RankingConfig
from portfolio.sizing import SizingConfig, SizingEngine

logger = logging.getLogger("portfolio.allocator")


# ── Spread covariance manager ─────────────────────────────────────

class SpreadCovarianceManager:
    """
    Estimates pairwise correlation of spread returns for active positions.

    Used to detect hidden concentration: pairs that appear independent by
    leg-overlap but are correlated through shared factor exposure (e.g., all
    long-SPY-correlated).  Complements the instrument-level shared-leg check
    in ExposureAnalyzer with a return-based view.
    """

    def __init__(self, window: int = 63, min_obs: int = 30) -> None:
        self.window = window
        self.min_obs = min_obs
        self._spread_returns: dict[str, pd.Series] = {}

    def update(self, pair_id: str, spread_return: float, ts: pd.Timestamp) -> None:
        """Record the latest spread return for a pair."""
        if pair_id not in self._spread_returns:
            self._spread_returns[pair_id] = pd.Series(dtype=float)
        new_pt = pd.Series([spread_return], index=[ts])
        self._spread_returns[pair_id] = pd.concat(
            [self._spread_returns[pair_id], new_pt]
        ).iloc[-self.window * 2:]

    def correlation_matrix(self, pair_ids: list[str]) -> pd.DataFrame:
        """
        Returns pairwise Pearson correlation matrix of spread returns.

        Pairs with fewer than ``min_obs`` observations are treated as
        uncorrelated (off-diagonal = 0.0).
        """
        data: dict[str, pd.Series] = {}
        for pid in pair_ids:
            series = self._spread_returns.get(pid, pd.Series(dtype=float))
            if len(series) >= self.min_obs:
                data[pid] = series.iloc[-self.window:]

        if len(data) < 2:
            return pd.DataFrame(
                np.eye(len(pair_ids)),
                index=pair_ids,
                columns=pair_ids,
            )

        df = pd.DataFrame(data).dropna(how="all")
        corr = df.corr().reindex(index=pair_ids, columns=pair_ids).fillna(0.0)
        np.fill_diagonal(corr.values, 1.0)
        return corr

    def max_pairwise_correlation(self, pair_ids: list[str]) -> float:
        """Returns the maximum absolute off-diagonal correlation between any two active pairs."""
        if len(pair_ids) < 2:
            return 0.0
        corr = self.correlation_matrix(pair_ids)
        corr_vals = corr.values.copy()
        np.fill_diagonal(corr_vals, 0.0)
        return float(np.max(np.abs(corr_vals)))

    def effective_n_independent(self, pair_ids: list[str]) -> float:
        """
        Effective number of independent positions (eigenvalue-based diversification ratio).

        Equals ``len(pair_ids)`` when all pairs are fully independent, and
        approaches 1 when all pairs are perfectly correlated.
        """
        if len(pair_ids) < 2:
            return float(len(pair_ids))
        corr = self.correlation_matrix(pair_ids)
        eigenvalues = np.linalg.eigvalsh(corr.values)
        eigenvalues = eigenvalues[eigenvalues > 0]
        if len(eigenvalues) == 0:
            return 1.0
        # DR = (sum λ)² / sum(λ²) — equivalent to (trace)² / ||λ||²
        total  = float(np.sum(eigenvalues))
        sum_sq = float(np.sum(eigenvalues ** 2))
        return float(total ** 2 / sum_sq) if sum_sq > 0 else 1.0


# ── Allocator configuration ───────────────────────────────────────

@dataclass
class AllocatorConfig:
    """Master configuration for the PortfolioAllocator."""

    # Sub-engine configs
    ranking: RankingConfig = field(default_factory=RankingConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)

    # Portfolio-level limits
    max_total_positions: int = 30
    max_new_entries_per_cycle: int = 5

    # Risk budget
    risk_budget: float = 1.0         # Normalised total risk budget

    # Minimum composite score to receive any capital
    min_score_to_fund: float = 0.15

    # Allow partial funding (scale down when capital is tight)
    allow_partial_funding: bool = True
    partial_funding_floor: float = 0.50  # Never fund less than 50% of target

    # Rebalance tolerance: don't rebalance if drift < this
    rebalance_drift_tolerance: float = 0.02  # 2% weight drift


# ── Portfolio Allocator ───────────────────────────────────────────

class PortfolioAllocator:
    """
    Main portfolio construction engine.

    Orchestrates ranking → sizing → constraint-checking → capital allocation.

    Parameters
    ----------
    capital_manager : CapitalManager
    config : AllocatorConfig
    """

    def __init__(
        self,
        capital_manager: CapitalManager,
        config: Optional[AllocatorConfig] = None,
    ):
        self._capital = capital_manager
        self._cfg = config or AllocatorConfig()

        self._ranker = OpportunityRanker(config=self._cfg.ranking)
        self._sizer = SizingEngine(config=self._cfg.sizing)
        self._exposure_analyzer = ExposureAnalyzer(config=self._cfg.exposure)
        self._covariance_manager = SpreadCovarianceManager()

        # Cycle-level state (reset each cycle)
        self._current_cycle_id: str = ""
        self._decisions_this_cycle: list[AllocationDecision] = []

    @property
    def capital(self) -> CapitalManager:
        return self._capital

    # ── Main entry point ──────────────────────────────────────────

    def run_cycle(
        self,
        intents: list[EntryIntent],
        *,
        # Portfolio context
        active_allocations: Optional[list[AllocationDecision]] = None,
        drawdown_state: Optional[DrawdownState] = None,
        throttle_state: Optional[ThrottleState] = None,
        kill_switch: Optional[KillSwitchState] = None,
        heat_state: Optional[HeatState] = None,
        # Market context
        portfolio_vol: Optional[float] = None,
        current_drawdown: float = 0.0,
        vix_level: float = 20.0,
        # Metadata
        sector_map: Optional[dict[str, str]] = None,
        cluster_map: Optional[dict[str, str]] = None,
        spread_vols: Optional[dict[str, float]] = None,
        hedge_ratios: Optional[dict[str, float]] = None,
    ) -> tuple[list[AllocationDecision], PortfolioDiagnostics]:
        """
        Run one portfolio construction cycle.

        Parameters
        ----------
        intents : list[EntryIntent] — approved signals from the signal layer
        active_allocations : current open positions
        ... (see full docstring)

        Returns
        -------
        (decisions, diagnostics) — list of AllocationDecisions + cycle diagnostics
        """
        cycle_id = str(uuid.uuid4())[:8]
        self._current_cycle_id = cycle_id
        self._decisions_this_cycle = []

        active_allocations = active_allocations or []
        sector_map = sector_map or {}
        cluster_map = cluster_map or {}
        spread_vols = spread_vols or {}
        hedge_ratios = hedge_ratios or {}

        drawdown_state = drawdown_state or DrawdownState()
        throttle_state = throttle_state or ThrottleState()
        kill_switch = kill_switch or KillSwitchState()
        heat_state = heat_state or HeatState()

        diag = PortfolioDiagnostics(
            cycle_id=cycle_id,
            n_intents_received=len(intents),
        )

        logger.info("Allocator cycle %s: %d intents received", cycle_id, len(intents))

        # ── Phase 1: Kill-switch check ────────────────────────────
        if kill_switch.is_blocking_new_entries():
            logger.warning("Kill-switch ACTIVE (%s) — blocking all new entries", kill_switch.mode.value)
            # Return empty — all intents rejected, record in diagnostics
            diag.n_blocked_risk = len(intents)
            diag.heat_level = drawdown_state.heat_level.value
            diag.kill_switch_mode = kill_switch.mode.value
            return [], diag

        # ── Phase 2: Heat level check ─────────────────────────────
        if drawdown_state.heat_level in (PortfolioHeatLevel.HALTED, PortfolioHeatLevel.RECOVERY_ONLY):
            logger.warning("Heat level %s — blocking new entries", drawdown_state.heat_level.value)
            diag.n_blocked_risk = len(intents)
            diag.heat_level = drawdown_state.heat_level.value
            return [], diag

        # ── Phase 3: Rank opportunities ───────────────────────────
        active_pairs = [d.pair_id for d in active_allocations if d.approved]
        active_instruments = self._exposure_analyzer.instrument_pair_count(active_allocations)

        opportunity_set: OpportunitySet = self._ranker.rank(
            intents,
            active_pairs=active_pairs,
            active_cluster_ids=cluster_map,
            active_instruments=active_instruments,
            entry_z_threshold=2.0,
        )

        diag.n_ranked = len(opportunity_set.opportunities)
        diag.n_blocked_signal = opportunity_set.n_blocked

        # ── Phase 4: Compute current exposure ────────────────────
        current_exposure = self._exposure_analyzer.compute(
            active_allocations,
            self._capital.pool_snapshot().total_capital,
            sector_map=sector_map,
            cluster_map=cluster_map,
        )

        # ── Phase 5: Throttle limits ──────────────────────────────
        n_active = self._capital.n_active()
        max_positions = min(
            self._cfg.max_total_positions,
            throttle_state.max_positions,
        )
        max_new = min(
            self._cfg.max_new_entries_per_cycle,
            throttle_state.max_new_entries_per_cycle,
        )
        min_score = max(
            self._cfg.min_score_to_fund,
            throttle_state.min_ranking_score_to_fund,
        )

        n_new_this_cycle = 0
        total_capital = self._capital.pool_snapshot().total_capital

        # ── Phase 6: Allocate to fundable opportunities ───────────
        decisions: list[AllocationDecision] = []

        for opp in opportunity_set.fundable_only():
            # Score filter
            if opp.composite_score < min_score:
                self._record_unfunded(
                    opp, AllocationOutcome.UNFUNDED,
                    f"score_below_min:{opp.composite_score:.3f}<{min_score:.3f}",
                    decisions,
                )
                continue

            # Position count limit
            if n_active + n_new_this_cycle >= max_positions:
                self._record_unfunded(
                    opp, AllocationOutcome.BLOCKED_CAPITAL,
                    f"max_positions:{max_positions}",
                    decisions,
                )
                diag.n_blocked_capital += 1
                continue

            # Per-cycle new-entry limit
            if n_new_this_cycle >= max_new:
                self._record_unfunded(
                    opp, AllocationOutcome.QUEUED,
                    f"cycle_limit:{max_new}_per_cycle",
                    decisions,
                )
                continue

            # Sleeve availability
            sleeve_name = opp.recommended_sleeve
            if not self._capital.get_sleeve(sleeve_name):
                sleeve_name = "default"
            opp.recommended_sleeve = sleeve_name

            # Size the position
            sv = spread_vols.get(opp.pair_id.label)
            hr = hedge_ratios.get(opp.pair_id.label, 1.0)
            sizing = self._sizer.size(
                opp,
                total_capital=total_capital,
                n_active_positions=n_active + n_new_this_cycle,
                portfolio_vol=portfolio_vol,
                spread_vol=sv,
                hedge_ratio=hr,
                drawdown_state=drawdown_state,
                throttle_state=throttle_state,
                current_drawdown=current_drawdown,
                vix_level=vix_level,
            )

            if not sizing.is_executable:
                self._record_unfunded(
                    opp, AllocationOutcome.UNFUNDED,
                    f"below_min_executable:{sizing.gross_notional:.0f}",
                    decisions,
                )
                continue

            # Risk constraint check
            constraint_result = self._check_constraints(
                opp, sizing, current_exposure, total_capital,
                sector_map=sector_map,
                cluster_map=cluster_map,
            )

            if not constraint_result.approved:
                outcome = self._constraint_outcome(constraint_result)
                self._record_unfunded(
                    opp, outcome,
                    constraint_result.rejection_reason,
                    decisions,
                    constraint_result=constraint_result,
                    sizing=sizing,
                )
                diag.n_blocked_risk += 1
                continue

            # Capital availability
            proposed_capital = sizing.capital_usage
            ok, reason = self._capital.can_allocate(sleeve_name, proposed_capital)

            if not ok:
                # Partial funding?
                if self._cfg.allow_partial_funding:
                    free = self._capital.free_capital_in_sleeve(sleeve_name)
                    floor = self._cfg.partial_funding_floor * proposed_capital
                    if free >= floor:
                        proposed_capital = free
                        sizing.gross_notional *= (free / sizing.capital_usage)
                        sizing.capital_usage = free
                        sizing.was_capped = True
                        sizing.cap_reason = f"partial_funding:{free:.0f}"
                    else:
                        self._record_unfunded(
                            opp, AllocationOutcome.BLOCKED_CAPITAL,
                            reason, decisions,
                        )
                        diag.n_blocked_capital += 1
                        continue
                else:
                    self._record_unfunded(
                        opp, AllocationOutcome.BLOCKED_CAPITAL,
                        reason, decisions,
                    )
                    diag.n_blocked_capital += 1
                    continue

            # ✅ Approved — commit capital
            self._capital.allocate(sleeve_name, opp.pair_id, proposed_capital)
            n_new_this_cycle += 1

            outcome = AllocationOutcome.FUNDED if not sizing.was_capped else AllocationOutcome.PARTIAL_FUNDED
            rationale = AllocationRationale(
                pair_id=opp.pair_id,
                outcome=outcome,
                rank=opp.rank,
                composite_score=opp.composite_score,
                capital_requested=sizing.capital_usage,
                capital_granted=proposed_capital,
                sleeve=sleeve_name,
                sizing_factors={
                    "conviction": sizing.conviction_scalar,
                    "quality": sizing.quality_scalar,
                    "regime": sizing.regime_scalar,
                    "drawdown": sizing.drawdown_scalar,
                    "vol_target": sizing.vol_target_scalar,
                },
                decision_notes=opp.strengths + opp.penalties,
            )

            proposal = AllocationProposal(
                pair_id=opp.pair_id,
                ranked_opportunity=opp,
                sizing=sizing,
                sleeve=sleeve_name,
                proposed_capital=proposed_capital,
                risk_budget_consumed=sizing.risk_contribution,
            )

            decision = AllocationDecision(
                pair_id=opp.pair_id,
                proposal=proposal,
                rationale=rationale,
                constraint_result=constraint_result,
                sizing=sizing,
                approved_capital=proposed_capital,
                approved_weight=sizing.weight_of_portfolio,
                approved=True,
                scaling_applied=proposed_capital / max(1.0, sizing.capital_usage),
            )
            decisions.append(decision)
            diag.n_funded += 1
            diag.top_funded.append(opp.pair_id.label)

        # ── Phase 7: Build diagnostics ────────────────────────────
        pool = self._capital.pool_snapshot()
        diag.capital_allocated = pool.allocated_capital
        diag.capital_reserved = pool.reserved_capital
        diag.capital_free = pool.free_capital
        diag.utilisation_pct = pool.utilisation_pct
        diag.heat_level = drawdown_state.heat_level.value
        diag.kill_switch_mode = kill_switch.mode.value
        diag.gross_leverage = current_exposure.gross_leverage
        diag.net_leverage = current_exposure.net_leverage
        diag.max_sector_concentration = current_exposure.max_sector_concentration
        diag.max_cluster_concentration = current_exposure.max_cluster_concentration
        diag.n_shared_leg_alerts = sum(1 for s in current_exposure.shared_legs if s.is_dominant)

        n_hard = sum(1 for d in decisions if not d.approved and d.constraint_result.hard_violations)
        n_soft = sum(1 for d in decisions if not d.approved and d.constraint_result.soft_violations)
        diag.n_hard_violations = n_hard
        diag.n_soft_violations = n_soft

        # ── Spread-return correlation check (hidden concentration) ─
        # Active pair IDs across all currently open positions.
        active_pair_ids = [d.pair_id.label for d in active_allocations if d.approved]
        if len(active_pair_ids) >= 2:
            max_corr = self._covariance_manager.max_pairwise_correlation(active_pair_ids)
            eff_n    = self._covariance_manager.effective_n_independent(active_pair_ids)

            if max_corr > 0.60:
                warn_msg = (
                    f"High spread correlation: max pairwise = {max_corr:.2f} "
                    "(hidden concentration risk)"
                )
                diag.binding_constraints.append(warn_msg)
                logger.warning("Cycle %s — %s", cycle_id, warn_msg)

            if eff_n < len(active_pair_ids) * 0.5:
                warn_msg = (
                    f"Low effective diversification: {eff_n:.1f} independent positions "
                    f"out of {len(active_pair_ids)} active"
                )
                diag.binding_constraints.append(warn_msg)
                logger.warning("Cycle %s — %s", cycle_id, warn_msg)

        logger.info(
            "Cycle %s complete: %d funded, %d signal-blocked, %d risk-blocked, %d capital-blocked",
            cycle_id, diag.n_funded, diag.n_blocked_signal,
            diag.n_blocked_risk, diag.n_blocked_capital,
        )

        self._decisions_this_cycle = decisions
        return decisions, diag

    # ── Constraint checking ───────────────────────────────────────

    def _check_constraints(
        self,
        opp: RankedOpportunity,
        sizing: SizingDecision,
        current_exposure: ExposureSummary,
        total_capital: float,
        *,
        sector_map: dict[str, str],
        cluster_map: dict[str, str],
    ) -> RiskConstraintResult:
        """Run all risk constraints against a proposed position."""
        result = RiskConstraintResult()
        cfg = self._cfg
        exp_cfg = self._cfg.exposure

        # Gross leverage check
        new_gross = current_exposure.gross_exposure + sizing.gross_notional
        new_leverage = new_gross / max(1.0, total_capital)
        if new_leverage > exp_cfg.max_gross_leverage:
            result.add(ConstraintViolation(
                constraint_type=ConstraintType.MAX_LEVERAGE,
                pair_id=opp.pair_id,
                description=f"Gross leverage {new_leverage:.2f} > {exp_cfg.max_gross_leverage}",
                current_value=new_leverage,
                limit_value=exp_cfg.max_gross_leverage,
                severity="HARD",
            ))

        # Sector exposure — only checked when sector is known (not the default "UNKNOWN")
        sector_x = sector_map.get(opp.pair_id.sym_x, "UNKNOWN")
        sector_y = sector_map.get(opp.pair_id.sym_y, "UNKNOWN")
        sector = sector_x if sector_x == sector_y else f"{sector_x}/{sector_y}"
        if sector and sector != "UNKNOWN":
            current_sector_frac = current_exposure.by_sector.get(sector, 0.0)
            current_sector_notional = current_sector_frac * current_exposure.gross_exposure
            new_sector_frac = (current_sector_notional + sizing.gross_notional) / max(1.0, new_gross)
            if new_sector_frac > exp_cfg.max_sector_fraction:
                result.add(ConstraintViolation(
                    constraint_type=ConstraintType.MAX_SECTOR_EXPOSURE,
                    pair_id=opp.pair_id,
                    description=f"Sector {sector} concentration {new_sector_frac:.1%} > {exp_cfg.max_sector_fraction:.1%}",
                    current_value=new_sector_frac,
                    limit_value=exp_cfg.max_sector_fraction,
                    severity="HARD",
                ))

        # Cluster exposure — only checked when cluster is known (not the default "UNKNOWN")
        cluster = cluster_map.get(opp.pair_id.label, "UNKNOWN")
        if cluster and cluster != "UNKNOWN":
            current_cluster_frac = current_exposure.by_cluster.get(cluster, 0.0)
            current_cluster_notional = current_cluster_frac * current_exposure.gross_exposure
            new_cluster_frac = (current_cluster_notional + sizing.gross_notional) / max(1.0, new_gross)
            if new_cluster_frac > exp_cfg.max_cluster_fraction:
                result.add(ConstraintViolation(
                    constraint_type=ConstraintType.MAX_CLUSTER_EXPOSURE,
                    pair_id=opp.pair_id,
                    description=f"Cluster {cluster} at {new_cluster_frac:.1%} > {exp_cfg.max_cluster_fraction:.1%}",
                    current_value=new_cluster_frac,
                    limit_value=exp_cfg.max_cluster_fraction,
                    severity="HARD",
                ))

        # Shared-leg check
        for instrument in (opp.pair_id.sym_x, opp.pair_id.sym_y):
            existing = next(
                (s for s in current_exposure.shared_legs if s.instrument == instrument),
                None,
            )
            n_using = existing.n_pairs_using if existing else 0
            if n_using >= exp_cfg.shared_leg_threshold:
                result.add(ConstraintViolation(
                    constraint_type=ConstraintType.MAX_SHARED_LEG,
                    pair_id=opp.pair_id,
                    description=f"Instrument {instrument} used in {n_using} pairs >= threshold {exp_cfg.shared_leg_threshold}",
                    current_value=float(n_using),
                    limit_value=float(exp_cfg.shared_leg_threshold),
                    severity="SOFT",  # warn, don't block by default
                ))

        # Single-pair weight
        if sizing.weight_of_portfolio > cfg.sizing.max_single_pair_weight * 1.1:
            result.add(ConstraintViolation(
                constraint_type=ConstraintType.MAX_SINGLE_PAIR,
                pair_id=opp.pair_id,
                description=f"Single pair weight {sizing.weight_of_portfolio:.2%} > {cfg.sizing.max_single_pair_weight:.2%}",
                current_value=sizing.weight_of_portfolio,
                limit_value=cfg.sizing.max_single_pair_weight,
                severity="SOFT",
            ))

        return result

    # ── De-risking ────────────────────────────────────────────────

    def compute_derisking(
        self,
        active_allocations: list[AllocationDecision],
        drawdown_state: DrawdownState,
    ) -> DeRiskingDecision:
        """
        Compute de-risking actions based on drawdown state.

        Returns DeRiskingDecision with pairs to exit / reduce.
        """
        heat = drawdown_state.heat_level
        decision = DeRiskingDecision(
            heat_level=heat,
            reason=f"heat_level:{heat.value}",
        )

        if heat == PortfolioHeatLevel.NORMAL:
            return decision  # Nothing to do

        # Sort by lowest composite_score first (weakest positions exit first)
        active = sorted(
            active_allocations,
            key=lambda d: d.proposal.ranked_opportunity.composite_score,
        )

        if heat == PortfolioHeatLevel.HALTED:
            # Exit everything
            decision.pairs_to_exit = [d.pair_id for d in active]
            decision.reason = "HALTED: full de-risk"
            decision.urgency = "URGENT"

        elif heat == PortfolioHeatLevel.RECOVERY_ONLY:
            # Exit bottom 50% by score
            n_exit = max(1, len(active) // 2)
            decision.pairs_to_exit = [d.pair_id for d in active[:n_exit]]
            decision.reason = "RECOVERY_ONLY: exit weakest 50%"
            decision.urgency = "URGENT"

        elif heat == PortfolioHeatLevel.DEFENSIVE:
            # Reduce all positions by throttle factor
            tf = drawdown_state.throttle_factor
            for d in active:
                reduction = 1.0 - tf
                decision.pairs_to_reduce.append(d.pair_id)
                decision.reduction_fractions[d.pair_id.label] = reduction
            decision.reason = f"DEFENSIVE: reduce all by {1-tf:.0%}"
            decision.urgency = "NORMAL"

        elif heat == PortfolioHeatLevel.THROTTLED:
            # Exit bottom 20% by score
            n_exit = max(0, int(len(active) * 0.20))
            if n_exit > 0:
                decision.pairs_to_exit = [d.pair_id for d in active[:n_exit]]
                # Reduce remainder by 20%
                for d in active[n_exit:]:
                    decision.pairs_to_reduce.append(d.pair_id)
                    decision.reduction_fractions[d.pair_id.label] = 0.20
            decision.reason = "THROTTLED: trim weakest positions"

        return decision

    # ── Helpers ───────────────────────────────────────────────────

    def _record_unfunded(
        self,
        opp: RankedOpportunity,
        outcome: AllocationOutcome,
        reason: str,
        decisions: list[AllocationDecision],
        constraint_result: Optional[RiskConstraintResult] = None,
        sizing: Optional[SizingDecision] = None,
    ) -> None:
        """Record a rejected allocation decision for audit trail."""
        if sizing is None:
            from portfolio.contracts import SizingDecision as _SD
            sizing = _SD(pair_id=opp.pair_id)
        if constraint_result is None:
            constraint_result = RiskConstraintResult()

        rationale = AllocationRationale(
            pair_id=opp.pair_id,
            outcome=outcome,
            rank=opp.rank,
            composite_score=opp.composite_score,
            sleeve=opp.recommended_sleeve,
            decision_notes=[reason],
        )
        proposal = AllocationProposal(
            pair_id=opp.pair_id,
            ranked_opportunity=opp,
            sizing=sizing,
            sleeve=opp.recommended_sleeve,
            proposed_capital=sizing.capital_usage,
        )
        decision = AllocationDecision(
            pair_id=opp.pair_id,
            proposal=proposal,
            rationale=rationale,
            constraint_result=constraint_result,
            sizing=sizing,
            approved_capital=0.0,
            approved_weight=0.0,
            approved=False,
            scaling_applied=0.0,
        )
        decisions.append(decision)

    @staticmethod
    def _constraint_outcome(result: RiskConstraintResult) -> AllocationOutcome:
        """Map constraint result → AllocationOutcome."""
        if not result.hard_violations:
            return AllocationOutcome.UNFUNDED
        vt = result.hard_violations[0].constraint_type
        if vt in (ConstraintType.MAX_SECTOR_EXPOSURE, ConstraintType.MAX_CLUSTER_EXPOSURE,
                  ConstraintType.MAX_SHARED_LEG, ConstraintType.MAX_LEVERAGE):
            return AllocationOutcome.BLOCKED_RISK
        if vt == ConstraintType.REGIME_BLOCKED:
            return AllocationOutcome.BLOCKED_REGIME
        if vt == ConstraintType.CAPITAL_BUDGET:
            return AllocationOutcome.BLOCKED_CAPITAL
        return AllocationOutcome.BLOCKED_SIGNAL
