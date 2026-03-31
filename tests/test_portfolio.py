# -*- coding: utf-8 -*-
"""
tests/test_portfolio.py — Portfolio Construction & Risk Model Tests
====================================================================

Test plan:
  TestCapitalManager          — pool, sleeve budgets, allocate/release
  TestOpportunityRanker       — scoring, diversification, overlap, blockers
  TestSizingEngine            — scalar stack, leg split, executability
  TestExposureAnalyzer        — leverage, sector, cluster, shared-leg
  TestPortfolioAllocator      — full cycle, constraint enforcement, de-risking
  TestDrawdownManager         — heat state machine, transitions, recovery
  TestKillSwitchManager       — trigger modes, escalation, reset
  TestRiskOperationsManager   — integrated facade
  TestPortfolioAgents         — each of the 6 agents
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pytest

from core.contracts import PairId, RegimeLabel, SignalQualityGrade
from core.intents import EntryIntent, IntentAction

# Portfolio modules
from portfolio.capital import CapitalManager
from portfolio.contracts import (
    AllocationDecision,
    AllocationOutcome,
    AllocationProposal,
    AllocationRationale,
    CapitalPool,
    ConstraintType,
    ConstraintViolation,
    DrawdownState,
    ExposureSummary,
    HeatState,
    KillSwitchMode,
    KillSwitchState,
    PortfolioHeatLevel,
    RankedOpportunity,
    RiskConstraintResult,
    SizingDecision,
    SleeveDef,
    ThrottleState,
)
from portfolio.exposures import ExposureAnalyzer, ExposureConfig
from portfolio.ranking import (
    OpportunityRanker,
    RankingConfig,
    RankingWeights,
    _signal_strength_score,
    _half_life_stability_score,
)
from portfolio.risk_ops import (
    DrawdownConfig,
    DrawdownManager,
    KillSwitchConfig,
    KillSwitchManager,
    RiskOperationsManager,
)
from portfolio.sizing import SizingConfig, SizingEngine
from portfolio.allocator import AllocatorConfig, PortfolioAllocator
from portfolio.analytics import PortfolioAnalytics


# ── Test helpers ─────────────────────────────────────────────────

def make_pair(x: str, y: str) -> PairId:
    return PairId(x, y)


def make_entry_intent(
    sym_x: str = "AAPL",
    sym_y: str = "MSFT",
    z_score: float = 2.5,
    confidence: float = 0.7,
    quality_grade: str = "B",
    regime: str = "MEAN_REVERTING",
    expected_half_life_days: float = 20.0,
    skip: bool = False,
) -> EntryIntent:
    """Build a minimal EntryIntent for testing."""
    pair_id = make_pair(sym_x, sym_y)
    intent = EntryIntent(
        pair_id=pair_id,
        action=IntentAction.ENTER,
        confidence=confidence,
        z_score=z_score,
        expected_half_life_days=expected_half_life_days,
    )
    # Attach extra context that signal layer would normally populate
    object.__setattr__(intent, "quality_grade", quality_grade)
    object.__setattr__(intent, "regime", type("R", (), {"value": regime})())
    object.__setattr__(intent, "skip_recommended", skip)
    return intent


def make_sizing_decision(
    pair_id: PairId,
    gross_notional: float = 50_000.0,
    capital_usage: float = 25_000.0,
    weight: float = 0.05,
    is_executable: bool = True,
    sleeve: str = "default",
) -> SizingDecision:
    return SizingDecision(
        pair_id=pair_id,
        sleeve=sleeve,
        gross_notional=gross_notional,
        risk_notional=gross_notional * 0.5,
        capital_usage=capital_usage,
        weight_of_portfolio=weight,
        risk_contribution=0.05,
        leg_x_notional=gross_notional * 0.5,
        leg_y_notional=gross_notional * 0.5,
        is_executable=is_executable,
    )


def make_ranked_opportunity(
    pair_id: PairId,
    composite_score: float = 0.60,
    quality_grade: str = "B",
    regime: str = "MEAN_REVERTING",
    z_score: float = 2.5,
    conviction: float = 0.7,
    blockers: Optional[list] = None,
    sleeve: str = "default",
) -> RankedOpportunity:
    return RankedOpportunity(
        pair_id=pair_id,
        composite_score=composite_score,
        quality_grade=quality_grade,
        regime=regime,
        z_score=z_score,
        conviction=conviction,
        signal_strength_score=0.6,
        signal_quality_score=0.7,
        regime_suitability_score=1.0,
        reversion_probability=conviction,
        stability_score=0.8,
        freshness_score=0.9,
        diversification_value=1.0,
        recommended_sleeve=sleeve,
        blockers=blockers or [],
    )


def make_allocation_decision(
    pair_id: PairId,
    approved: bool = True,
    capital: float = 25_000.0,
    weight: float = 0.025,
    sleeve: str = "default",
    composite_score: float = 0.60,
) -> AllocationDecision:
    opp = make_ranked_opportunity(pair_id, composite_score=composite_score, sleeve=sleeve)
    sizing = make_sizing_decision(pair_id, capital_usage=capital, weight=weight, sleeve=sleeve)
    proposal = AllocationProposal(
        pair_id=pair_id,
        ranked_opportunity=opp,
        sizing=sizing,
        sleeve=sleeve,
        proposed_capital=capital,
    )
    rationale = AllocationRationale(
        pair_id=pair_id,
        outcome=AllocationOutcome.FUNDED if approved else AllocationOutcome.UNFUNDED,
        sleeve=sleeve,
        capital_granted=capital if approved else 0.0,
        capital_requested=capital,
    )
    return AllocationDecision(
        pair_id=pair_id,
        proposal=proposal,
        rationale=rationale,
        constraint_result=RiskConstraintResult(),
        sizing=sizing,
        approved_capital=capital if approved else 0.0,
        approved_weight=weight if approved else 0.0,
        approved=approved,
    )


# ════════════════════════════════════════════════════════════════
# TestCapitalManager
# ════════════════════════════════════════════════════════════════

class TestCapitalManager:

    def test_initial_pool(self):
        mgr = CapitalManager(total_capital=1_000_000.0)
        pool = mgr.pool_snapshot()
        assert pool.total_capital == 1_000_000.0
        assert pool.free_capital == 1_000_000.0
        assert pool.utilisation_pct == 0.0

    def test_add_sleeve(self):
        mgr = CapitalManager(total_capital=1_000_000.0)
        mgr.add_sleeve(SleeveDef("alpha", max_capital_fraction=0.40))
        s = mgr.get_sleeve("alpha")
        assert s is not None
        assert s.max_capital_fraction == 0.40

    def test_allocate_and_pool_update(self):
        mgr = CapitalManager(total_capital=1_000_000.0)
        mgr.add_sleeve(SleeveDef("core", max_capital_fraction=0.50))
        pair = make_pair("AAPL", "MSFT")
        mgr.allocate("core", pair, 100_000.0)
        pool = mgr.pool_snapshot()
        assert pool.allocated_capital == 100_000.0
        assert pool.free_capital == 900_000.0

    def test_release_restores_capital(self):
        mgr = CapitalManager(total_capital=1_000_000.0)
        mgr.add_sleeve(SleeveDef("core", max_capital_fraction=0.50))
        pair = make_pair("AAPL", "MSFT")
        mgr.allocate("core", pair, 100_000.0)
        released = mgr.release(pair)
        assert released == 100_000.0
        pool = mgr.pool_snapshot()
        assert pool.free_capital == 1_000_000.0

    def test_sleeve_budget_tracks_separately(self):
        mgr = CapitalManager(total_capital=1_000_000.0)
        mgr.add_sleeve(SleeveDef("sleeve_a", max_capital_fraction=0.20))
        mgr.add_sleeve(SleeveDef("sleeve_b", max_capital_fraction=0.30))
        pair_a = make_pair("AAPL", "MSFT")
        pair_b = make_pair("GOOG", "META")
        mgr.allocate("sleeve_a", pair_a, 50_000.0)
        mgr.allocate("sleeve_b", pair_b, 80_000.0)
        budget_a = mgr.sleeve_budget("sleeve_a")
        budget_b = mgr.sleeve_budget("sleeve_b")
        assert budget_a.allocated == 50_000.0
        assert budget_b.allocated == 80_000.0
        assert budget_a.n_positions == 1
        assert budget_b.n_positions == 1

    def test_can_allocate_respects_sleeve_max(self):
        mgr = CapitalManager(total_capital=1_000_000.0)
        mgr.add_sleeve(SleeveDef("small", max_capital_fraction=0.10))
        ok, reason = mgr.can_allocate("small", 150_000.0)
        assert not ok
        assert "budget exceeded" in reason.lower() or "sleeve" in reason.lower()

    def test_can_allocate_respects_position_limit(self):
        mgr = CapitalManager(total_capital=1_000_000.0)
        mgr.add_sleeve(SleeveDef("limited", max_positions=2))
        p1 = make_pair("A", "B")
        p2 = make_pair("C", "D")
        mgr.allocate("limited", p1, 10_000.0)
        mgr.allocate("limited", p2, 10_000.0)
        ok, reason = mgr.can_allocate("limited", 10_000.0)
        assert not ok
        assert "position limit" in reason.lower()

    def test_n_active_counts_only_active(self):
        mgr = CapitalManager(total_capital=1_000_000.0)
        p1 = make_pair("A", "B")
        p2 = make_pair("C", "D")
        mgr.allocate("default", p1, 10_000.0)
        mgr.allocate("default", p2, 10_000.0)
        assert mgr.n_active() == 2
        mgr.release(p1)
        assert mgr.n_active() == 1

    def test_adjust_increases_and_decreases(self):
        mgr = CapitalManager(total_capital=1_000_000.0)
        pair = make_pair("AAPL", "GOOG")
        mgr.allocate("default", pair, 50_000.0)
        mgr.adjust(pair, 75_000.0)
        assert mgr.allocated_for(pair) == 75_000.0

    def test_total_capital_must_be_positive(self):
        with pytest.raises(ValueError):
            CapitalManager(total_capital=0.0)


# ════════════════════════════════════════════════════════════════
# TestOpportunityRanker
# ════════════════════════════════════════════════════════════════

class TestOpportunityRanker:

    def test_signal_strength_score_basic(self):
        # At entry_z=2.0, z=2.5 → small positive score
        score = _signal_strength_score(2.5, entry_z=2.0, saturation_z=3.0)
        assert 0.0 < score < 1.0

    def test_signal_strength_score_below_entry(self):
        score = _signal_strength_score(1.5, entry_z=2.0, saturation_z=3.0)
        assert score == 0.0

    def test_signal_strength_score_saturation(self):
        score = _signal_strength_score(5.0, entry_z=2.0, saturation_z=3.0)
        assert score == 1.0

    def test_half_life_stability_scores(self):
        assert _half_life_stability_score(15.0) == 1.0    # Sweet spot
        assert _half_life_stability_score(1.0) < 0.5      # Too fast
        assert _half_life_stability_score(100.0) < 0.5    # Too slow
        assert _half_life_stability_score(float("nan")) == 0.3

    def test_ranker_produces_opportunity_set(self):
        ranker = OpportunityRanker()
        intents = [make_entry_intent("AAPL", "MSFT"), make_entry_intent("GOOG", "META")]
        opp_set = ranker.rank(intents)
        assert len(opp_set.opportunities) == 2
        assert opp_set.n_input_intents == 2

    def test_rank_1_is_best(self):
        ranker = OpportunityRanker()
        intents = [
            make_entry_intent("AAPL", "MSFT", z_score=3.5, confidence=0.9),  # Strong
            make_entry_intent("GOOG", "META", z_score=2.1, confidence=0.4),  # Weak
        ]
        opp_set = ranker.rank(intents)
        ranked = opp_set.sorted_by_rank()
        assert ranked[0].composite_score >= ranked[1].composite_score

    def test_crisis_regime_blocks_entry(self):
        ranker = OpportunityRanker()
        intents = [make_entry_intent("AAPL", "MSFT", regime="CRISIS")]
        opp_set = ranker.rank(intents)
        assert len(opp_set.fundable_only()) == 0
        assert opp_set.opportunities[0].blockers != []
        assert opp_set.n_blocked == 1

    def test_grade_f_blocks_entry(self):
        ranker = OpportunityRanker()
        intents = [make_entry_intent("AAPL", "MSFT", quality_grade="F")]
        opp_set = ranker.rank(intents)
        assert len(opp_set.fundable_only()) == 0
        assert opp_set.n_blocked == 1

    def test_skip_recommended_blocks_entry(self):
        ranker = OpportunityRanker()
        intents = [make_entry_intent("AAPL", "MSFT", skip=True)]
        opp_set = ranker.rank(intents)
        assert len(opp_set.fundable_only()) == 0

    def test_diversification_penalty_for_shared_legs(self):
        """Second pair sharing a leg should get lower diversification_value."""
        ranker = OpportunityRanker()
        active_pairs = [make_pair("AAPL", "MSFT")]
        intents = [make_entry_intent("AAPL", "GOOG")]  # AAPL is already in active
        opp_set = ranker.rank(intents, active_pairs=active_pairs)
        opp = opp_set.opportunities[0]
        # Should have diversification_value < 1.0 since AAPL is already active
        assert opp.diversification_value < 1.0

    def test_overlap_penalty_applied_to_composite(self):
        ranker = OpportunityRanker()
        # Active positions using both instruments of a new pair
        active_pairs = [
            make_pair("AAPL", "GOOG"),
            make_pair("AAPL", "META"),
        ]
        active_instruments = {"AAPL": 2, "MSFT": 1}
        intents = [make_entry_intent("AAPL", "MSFT")]
        opp_set = ranker.rank(intents, active_pairs=active_pairs, active_instruments=active_instruments)
        opp = opp_set.opportunities[0]
        assert opp.overlap_penalty > 0.0

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            RankingWeights(signal_strength=0.9, signal_quality=0.9)

    def test_fundable_only_excludes_blocked(self):
        ranker = OpportunityRanker()
        intents = [
            make_entry_intent("AAPL", "MSFT", z_score=2.5),          # OK
            make_entry_intent("GOOG", "META", quality_grade="F"),      # Blocked
            make_entry_intent("AMZN", "NFLX", regime="CRISIS"),       # Blocked
        ]
        opp_set = ranker.rank(intents)
        fundable = opp_set.fundable_only()
        assert len(fundable) == 1
        assert fundable[0].pair_id.label == "AAPL/MSFT"


# ════════════════════════════════════════════════════════════════
# TestSizingEngine
# ════════════════════════════════════════════════════════════════

class TestSizingEngine:

    def test_returns_sizing_decision(self):
        sizer = SizingEngine()
        opp = make_ranked_opportunity(make_pair("AAPL", "MSFT"))
        decision = sizer.size(opp, total_capital=1_000_000.0)
        assert isinstance(decision, SizingDecision)
        assert decision.gross_notional >= 0

    def test_crisis_regime_zeroes_size(self):
        cfg = SizingConfig()
        sizer = SizingEngine(config=cfg)
        opp = make_ranked_opportunity(make_pair("AAPL", "MSFT"), regime="CRISIS")
        decision = sizer.size(opp, total_capital=1_000_000.0)
        # Regime scalar for CRISIS = 0.0 → gross_notional should be 0 or near 0
        assert decision.gross_notional < 1.0 or not decision.is_executable

    def test_leg_split_sums_to_gross(self):
        sizer = SizingEngine()
        opp = make_ranked_opportunity(make_pair("AAPL", "MSFT"))
        decision = sizer.size(opp, total_capital=1_000_000.0, hedge_ratio=1.5)
        total_legs = decision.leg_x_notional + decision.leg_y_notional
        assert abs(total_legs - decision.gross_notional) < 0.01

    def test_drawdown_throttle_reduces_size(self):
        sizer = SizingEngine()
        opp = make_ranked_opportunity(make_pair("AAPL", "MSFT"))
        # High drawdown throttle
        dd_stressed = DrawdownState(throttle_factor=0.25)
        decision_stressed = sizer.size(opp, total_capital=1_000_000.0, drawdown_state=dd_stressed)
        decision_normal = sizer.size(opp, total_capital=1_000_000.0)
        assert decision_stressed.gross_notional < decision_normal.gross_notional

    def test_conviction_scaling(self):
        cfg = SizingConfig(conviction_scaling_enabled=True)
        sizer = SizingEngine(config=cfg)
        opp_high = make_ranked_opportunity(make_pair("AAPL", "MSFT"), conviction=0.95)
        opp_low = make_ranked_opportunity(make_pair("AAPL", "MSFT"), conviction=0.20)
        d_high = sizer.size(opp_high, total_capital=1_000_000.0)
        d_low = sizer.size(opp_low, total_capital=1_000_000.0)
        assert d_high.gross_notional > d_low.gross_notional

    def test_max_single_pair_cap(self):
        cfg = SizingConfig(max_single_pair_weight=0.05)
        sizer = SizingEngine(config=cfg)
        opp = make_ranked_opportunity(make_pair("AAPL", "MSFT"), conviction=0.99)
        decision = sizer.size(opp, total_capital=1_000_000.0)
        assert decision.weight_of_portfolio <= cfg.max_single_pair_weight * 1.01  # Allow tiny float error

    def test_below_min_executable_is_flagged(self):
        cfg = SizingConfig(min_executable_notional=100_000.0)
        sizer = SizingEngine(config=cfg)
        opp = make_ranked_opportunity(make_pair("AAPL", "MSFT"), conviction=0.01)
        decision = sizer.size(opp, total_capital=10_000.0)  # Very small capital
        assert not decision.is_executable

    def test_direction_inferred_from_zscore(self):
        sizer = SizingEngine()
        opp_pos = make_ranked_opportunity(make_pair("AAPL", "MSFT"), z_score=2.5)
        opp_neg = make_ranked_opportunity(make_pair("AAPL", "MSFT"), z_score=-2.5)
        d_pos = sizer.size(opp_pos, total_capital=1_000_000.0)
        d_neg = sizer.size(opp_neg, total_capital=1_000_000.0)
        assert d_pos.direction == "SHORT_SPREAD"
        assert d_neg.direction == "LONG_SPREAD"


# ════════════════════════════════════════════════════════════════
# TestExposureAnalyzer
# ════════════════════════════════════════════════════════════════

class TestExposureAnalyzer:

    def _make_decisions(self, pairs_and_notionals):
        decisions = []
        for (sx, sy), notional, sleeve in pairs_and_notionals:
            pair = make_pair(sx, sy)
            d = make_allocation_decision(pair, capital=notional / 2, weight=notional / 1_000_000, sleeve=sleeve)
            # Override sizing gross_notional for exposure test
            d.sizing.gross_notional = notional
            d.sizing.leg_x_notional = notional / 2
            d.sizing.leg_y_notional = notional / 2
            decisions.append(d)
        return decisions

    def test_basic_exposure_computes(self):
        analyzer = ExposureAnalyzer()
        decisions = self._make_decisions([
            (("AAPL", "MSFT"), 100_000.0, "default"),
            (("GOOG", "META"), 80_000.0, "default"),
        ])
        summary = analyzer.compute(decisions, 1_000_000.0)
        assert summary.gross_exposure == pytest.approx(180_000.0)
        assert summary.gross_leverage == pytest.approx(0.18)

    def test_sector_concentration(self):
        analyzer = ExposureAnalyzer()
        decisions = self._make_decisions([
            (("AAPL", "MSFT"), 400_000.0, "default"),  # Tech/Tech
            (("GOOG", "META"), 100_000.0, "default"),  # Tech/Tech
        ])
        sector_map = {"AAPL": "TECH", "MSFT": "TECH", "GOOG": "TECH", "META": "TECH"}
        summary = analyzer.compute(decisions, 1_000_000.0, sector_map=sector_map)
        assert summary.by_sector.get("TECH", 0.0) == pytest.approx(1.0, abs=0.01)

    def test_shared_leg_detection(self):
        analyzer = ExposureAnalyzer(ExposureConfig(shared_leg_threshold=2))
        decisions = self._make_decisions([
            (("AAPL", "MSFT"), 50_000.0, "default"),
            (("AAPL", "GOOG"), 50_000.0, "default"),
            (("AAPL", "META"), 50_000.0, "default"),
        ])
        summary = analyzer.compute(decisions, 1_000_000.0)
        dominant = [s for s in summary.shared_legs if s.instrument == "AAPL"]
        assert len(dominant) == 1
        assert dominant[0].is_dominant
        assert "AAPL" in summary.dominant_legs

    def test_cluster_overcrowding(self):
        analyzer = ExposureAnalyzer(ExposureConfig(max_cluster_fraction=0.20))
        decisions = self._make_decisions([
            (("AAPL", "MSFT"), 500_000.0, "default"),
        ])
        cluster_map = {"AAPL/MSFT": "cluster_1"}
        summary = analyzer.compute(decisions, 1_000_000.0, cluster_map=cluster_map)
        # 500k/500k = 100% in cluster_1 → overcrowded
        overcrowded = [c for c in summary.cluster_exposures if c.is_overcrowded]
        assert len(overcrowded) >= 1

    def test_empty_portfolio_returns_zero_exposure(self):
        analyzer = ExposureAnalyzer()
        summary = analyzer.compute([], 1_000_000.0)
        assert summary.gross_exposure == 0.0
        assert summary.gross_leverage == 0.0

    def test_check_new_position_leverage_violation(self):
        analyzer = ExposureAnalyzer(ExposureConfig(max_gross_leverage=2.0))
        decisions = self._make_decisions([
            (("AAPL", "MSFT"), 1_800_000.0, "default"),
        ])
        current = analyzer.compute(decisions, 1_000_000.0)
        violations = analyzer.check_new_position(
            make_pair("GOOG", "META"), 300_000.0,
            current, 1_000_000.0,
        )
        assert any("gross_leverage" in v for v in violations)


# ════════════════════════════════════════════════════════════════
# TestPortfolioAllocator
# ════════════════════════════════════════════════════════════════

class TestPortfolioAllocator:

    def _make_allocator(self, total_capital: float = 1_000_000.0) -> tuple:
        capital_mgr = CapitalManager(total_capital=total_capital)
        capital_mgr.add_sleeve(SleeveDef("default", max_capital_fraction=1.0))
        capital_mgr.add_sleeve(SleeveDef("high_conviction", max_capital_fraction=0.40))
        allocator = PortfolioAllocator(capital_mgr, config=AllocatorConfig())
        return allocator, capital_mgr

    def test_basic_cycle_funds_eligible_intents(self):
        allocator, _ = self._make_allocator()
        intents = [
            make_entry_intent("AAPL", "MSFT", z_score=2.8, confidence=0.75),
            make_entry_intent("GOOG", "META", z_score=2.6, confidence=0.65),
        ]
        decisions, diag = allocator.run_cycle(intents)
        funded = [d for d in decisions if d.approved]
        assert len(funded) >= 1
        assert diag.n_funded >= 1

    def test_kill_switch_hard_blocks_all(self):
        allocator, _ = self._make_allocator()
        intents = [make_entry_intent("AAPL", "MSFT")]
        ks = KillSwitchState(mode=KillSwitchMode.HARD, triggered=True)
        decisions, diag = allocator.run_cycle(
            intents,
            kill_switch=ks,
        )
        assert len([d for d in decisions if d.approved]) == 0

    def test_halted_heat_blocks_all(self):
        allocator, _ = self._make_allocator()
        intents = [make_entry_intent("AAPL", "MSFT")]
        dd = DrawdownState(heat_level=PortfolioHeatLevel.HALTED, throttle_factor=0.0)
        decisions, diag = allocator.run_cycle(intents, drawdown_state=dd)
        funded = [d for d in decisions if d.approved]
        assert len(funded) == 0

    def test_blocked_signal_grade_f(self):
        allocator, _ = self._make_allocator()
        intents = [make_entry_intent("AAPL", "MSFT", quality_grade="F")]
        decisions, diag = allocator.run_cycle(intents)
        funded = [d for d in decisions if d.approved]
        assert len(funded) == 0
        assert diag.n_blocked_signal >= 1

    def test_max_new_entries_per_cycle_limit(self):
        cfg = AllocatorConfig(max_new_entries_per_cycle=2)
        capital_mgr = CapitalManager(total_capital=5_000_000.0)
        capital_mgr.add_sleeve(SleeveDef("default", max_capital_fraction=1.0))
        allocator = PortfolioAllocator(capital_mgr, config=cfg)
        # 5 good intents, limit is 2
        intents = [
            make_entry_intent(f"S{i}", f"T{i}", z_score=3.0, confidence=0.8)
            for i in range(5)
        ]
        decisions, diag = allocator.run_cycle(intents)
        funded = [d for d in decisions if d.approved]
        assert len(funded) <= 2

    def test_derisking_normal_heat_empty(self):
        allocator, _ = self._make_allocator()
        dd = DrawdownState(heat_level=PortfolioHeatLevel.NORMAL)
        decision = allocator.compute_derisking([], dd)
        assert len(decision.pairs_to_exit) == 0

    def test_derisking_halted_exits_all(self):
        allocator, _ = self._make_allocator()
        dd = DrawdownState(heat_level=PortfolioHeatLevel.HALTED, throttle_factor=0.0)
        pairs = [make_pair("A", "B"), make_pair("C", "D")]
        decisions = [make_allocation_decision(p) for p in pairs]
        dr = allocator.compute_derisking(decisions, dd)
        assert len(dr.pairs_to_exit) == len(pairs)
        assert dr.urgency == "URGENT"

    def test_derisking_defensive_reduces_all(self):
        allocator, _ = self._make_allocator()
        dd = DrawdownState(heat_level=PortfolioHeatLevel.DEFENSIVE, throttle_factor=0.25)
        pairs = [make_pair("A", "B"), make_pair("C", "D")]
        decisions = [make_allocation_decision(p) for p in pairs]
        dr = allocator.compute_derisking(decisions, dd)
        assert len(dr.pairs_to_reduce) == len(pairs)


# ════════════════════════════════════════════════════════════════
# TestDrawdownManager
# ════════════════════════════════════════════════════════════════

class TestDrawdownManager:

    def test_initial_state_is_normal(self):
        mgr = DrawdownManager()
        assert mgr.state.heat_level == PortfolioHeatLevel.NORMAL

    def test_small_dd_triggers_cautious(self):
        cfg = DrawdownConfig(cautious_dd_threshold=0.03)
        mgr = DrawdownManager(cfg)
        state = mgr.update(0.97, peak_value=1.0)  # 3% DD
        assert state.heat_level == PortfolioHeatLevel.CAUTIOUS

    def test_large_dd_triggers_halted(self):
        cfg = DrawdownConfig(halted_dd_threshold=0.20)
        mgr = DrawdownManager(cfg)
        state = mgr.update(0.79, peak_value=1.0)  # 21% DD
        assert state.heat_level == PortfolioHeatLevel.HALTED

    def test_heat_cannot_recover_automatically(self):
        """Once throttled, a small improvement doesn't auto-recover."""
        cfg = DrawdownConfig(throttled_dd_threshold=0.06, cautious_dd_threshold=0.03)
        mgr = DrawdownManager(cfg)
        mgr.update(0.93, peak_value=1.0)   # → THROTTLED
        # Now recover to 5% DD — still above cautious threshold
        state = mgr.update(0.95, peak_value=1.0)
        # One-way rule: should stay at THROTTLED or worse, not drop back to CAUTIOUS
        assert state.heat_level.value in (
            "THROTTLED", "DEFENSIVE", "RECOVERY_ONLY", "HALTED"
        )

    def test_throttle_factor_matches_level(self):
        cfg = DrawdownConfig(throttled_dd_threshold=0.06)
        cfg.throttle_multipliers["THROTTLED"] = 0.55
        mgr = DrawdownManager(cfg)
        state = mgr.update(0.93, peak_value=1.0)  # → THROTTLED
        assert state.throttle_factor == 0.55

    def test_crisis_pairs_trigger_cautious(self):
        mgr = DrawdownManager()
        state = mgr.update(1.0, peak_value=1.0, n_crisis_pairs=1)
        assert state.heat_level in (PortfolioHeatLevel.CAUTIOUS, PortfolioHeatLevel.THROTTLED)

    def test_build_throttle_state_from_level(self):
        mgr = DrawdownManager()
        mgr.update(0.93, peak_value=1.0)  # → some stressed level
        ts = mgr.build_throttle_state()
        assert isinstance(ts, ThrottleState)
        assert ts.size_multiplier >= 0.0

    def test_force_level_override(self):
        mgr = DrawdownManager()
        state = mgr.update(1.0, peak_value=1.0, force_level=PortfolioHeatLevel.DEFENSIVE)
        assert state.heat_level == PortfolioHeatLevel.DEFENSIVE


# ════════════════════════════════════════════════════════════════
# TestKillSwitchManager
# ════════════════════════════════════════════════════════════════

class TestKillSwitchManager:

    def test_initial_state_is_off(self):
        mgr = KillSwitchManager()
        assert mgr.state.mode == KillSwitchMode.OFF
        assert not mgr.state.triggered

    def test_large_dd_triggers_hard(self):
        cfg = KillSwitchConfig(hard_dd_threshold=0.20)
        mgr = KillSwitchManager(cfg)
        state = mgr.check(0.79, 1.0)
        assert state.mode == KillSwitchMode.HARD
        assert state.triggered

    def test_soft_dd_triggers_soft(self):
        cfg = KillSwitchConfig(soft_dd_threshold=0.12, reduce_dd_threshold=0.16, hard_dd_threshold=0.20)
        mgr = KillSwitchManager(cfg)
        state = mgr.check(0.87, 1.0)   # 13% DD → SOFT
        assert state.mode == KillSwitchMode.SOFT

    def test_kill_switch_only_escalates(self):
        mgr = KillSwitchManager()
        # First trigger REDUCE
        mgr.check(0.83, 1.0)  # 17% → REDUCE
        # Then check with lower DD
        state = mgr.check(0.90, 1.0)   # 10% → would be SOFT; but stays at REDUCE
        assert state.mode in (KillSwitchMode.REDUCE, KillSwitchMode.HARD)

    def test_manual_trigger(self):
        mgr = KillSwitchManager()
        state = mgr.trigger_manual(KillSwitchMode.HARD, "operator test")
        assert state.mode == KillSwitchMode.HARD
        assert state.triggered
        assert "operator test" in state.reason

    def test_reset_requires_acknowledgment(self):
        cfg = KillSwitchConfig(require_acknowledgment=True)
        mgr = KillSwitchManager(cfg)
        mgr.check(0.79, 1.0)  # Trigger HARD
        success = mgr.reset()
        assert not success  # Blocked without ack

    def test_reset_after_acknowledgment(self):
        mgr = KillSwitchManager()
        mgr.check(0.79, 1.0)
        mgr.acknowledge()
        success = mgr.reset()
        assert success
        assert mgr.state.mode == KillSwitchMode.OFF

    def test_is_blocking_for_hard_and_reduce(self):
        mgr = KillSwitchManager()
        mgr.trigger_manual(KillSwitchMode.HARD, "test")
        assert mgr.state.is_blocking_new_entries()
        mgr.reset(force=True)
        mgr.trigger_manual(KillSwitchMode.REDUCE, "test")
        assert mgr.state.is_blocking_new_entries()

    def test_soft_mode_does_not_block(self):
        mgr = KillSwitchManager()
        mgr.trigger_manual(KillSwitchMode.SOFT, "test")
        assert not mgr.state.is_blocking_new_entries()


# ════════════════════════════════════════════════════════════════
# TestRiskOperationsManager
# ════════════════════════════════════════════════════════════════

class TestRiskOperationsManager:

    def test_update_returns_both_states(self):
        rom = RiskOperationsManager()
        dd, ks = rom.update(0.95, 1.0, rolling_dd_7d=0.02, rolling_dd_30d=0.03)
        assert isinstance(dd, DrawdownState)
        assert isinstance(ks, KillSwitchState)

    def test_no_restriction_initially(self):
        rom = RiskOperationsManager()
        assert not rom.is_any_restriction_active()

    def test_restriction_detected_on_stress(self):
        cfg_dd = DrawdownConfig(cautious_dd_threshold=0.03)
        rom = RiskOperationsManager(drawdown_config=cfg_dd)
        rom.update(0.96, 1.0)  # 4% DD → CAUTIOUS
        assert rom.is_any_restriction_active()

    def test_summary_dict_has_required_keys(self):
        rom = RiskOperationsManager()
        s = rom.summary()
        assert "heat_level" in s
        assert "kill_switch_mode" in s
        assert "any_restriction_active" in s


# ════════════════════════════════════════════════════════════════
# TestPortfolioAnalytics
# ════════════════════════════════════════════════════════════════

class TestPortfolioAnalytics:

    def test_build_snapshot_returns_snapshot(self):
        analytics = PortfolioAnalytics()
        capital_mgr = CapitalManager(total_capital=1_000_000.0)
        dd = DrawdownState()
        ks = KillSwitchState()
        snapshot = analytics.build_snapshot(capital_mgr, dd, ks, [])
        assert snapshot.capital_pool.total_capital == 1_000_000.0
        assert snapshot.kill_switch.mode == KillSwitchMode.OFF

    def test_build_diagnostics_counts_correctly(self):
        analytics = PortfolioAnalytics()
        capital_mgr = CapitalManager(total_capital=1_000_000.0)
        pair_a = make_pair("AAPL", "MSFT")
        pair_b = make_pair("GOOG", "META")
        decisions = [
            make_allocation_decision(pair_a, approved=True),
            make_allocation_decision(pair_b, approved=False),
        ]
        from portfolio.contracts import OpportunitySet
        opp_set = OpportunitySet(n_input_intents=2)

        snapshot = analytics.build_snapshot(capital_mgr, DrawdownState(), KillSwitchState(), decisions)
        diag = analytics.build_diagnostics("test_cycle", opp_set, decisions, snapshot)
        assert diag.n_funded == 1
        assert diag.cycle_id == "test_cycle"

    def test_funnel_report_correct_counts(self):
        analytics = PortfolioAnalytics()
        pair_a = make_pair("A", "B")
        pair_b = make_pair("C", "D")
        funded_d = make_allocation_decision(pair_a, approved=True)
        unfunded_d = make_allocation_decision(pair_b, approved=False)
        from portfolio.contracts import OpportunitySet
        opp_set = OpportunitySet(n_input_intents=2)
        report = analytics.build_funnel_report("c1", opp_set, [funded_d, unfunded_d], 1_000_000.0)
        assert report.n_funded == 1
        assert report.pass_rate == 0.5


# ════════════════════════════════════════════════════════════════
# TestPortfolioAgents
# ════════════════════════════════════════════════════════════════

class TestPortfolioAgents:

    def _task(self, task_type: str, payload: dict) -> "AgentTask":
        from core.contracts import AgentTask
        import uuid
        return AgentTask(
            task_id=str(uuid.uuid4()),
            agent_name="test",
            task_type=task_type,
            payload=payload,
        )

    def test_capital_budget_agent_fundable(self):
        from agents.portfolio_agents import CapitalBudgetAgent
        agent = CapitalBudgetAgent()
        task = self._task("check_capital_budget", {
            "total_capital": 1_000_000.0,
            "pairs": [
                {"pair_label": "AAPL/MSFT", "proposed_notional": 50_000.0, "sleeve": "default"},
                {"pair_label": "GOOG/META", "proposed_notional": 200_000.0, "sleeve": "default"},
            ],
        })
        result = agent.execute(task)
        from core.contracts import AgentStatus
        assert result.status == AgentStatus.COMPLETED
        assert result.output["n_fundable"] >= 1
        assert result.output["free_capital"] > 0

    def test_drawdown_monitor_agent(self):
        from agents.portfolio_agents import DrawdownMonitorAgent
        agent = DrawdownMonitorAgent()
        task = self._task("update_drawdown", {
            "current_value": 0.94,
            "peak_value": 1.0,
            "rolling_dd_30d": 0.04,
        })
        result = agent.execute(task)
        from core.contracts import AgentStatus
        assert result.status == AgentStatus.COMPLETED
        dd = result.output["drawdown_state"]
        assert dd["current_dd_pct"] == pytest.approx(0.06, abs=0.001)
        assert "heat_level" in dd

    def test_kill_switch_agent_evaluate(self):
        from agents.portfolio_agents import KillSwitchAgent
        agent = KillSwitchAgent()
        task = self._task("evaluate_kill_switch", {
            "current_value": 0.75,
            "peak_value": 1.0,
        })
        result = agent.execute(task)
        from core.contracts import AgentStatus
        assert result.status == AgentStatus.COMPLETED
        assert result.output["triggered"] is True
        assert result.output["mode"] == "HARD"

    def test_kill_switch_agent_manual_trigger(self):
        from agents.portfolio_agents import KillSwitchAgent
        agent = KillSwitchAgent()
        task = self._task("manual_trigger", {
            "mode": "SOFT",
            "reason": "test_trigger",
        })
        result = agent.execute(task)
        from core.contracts import AgentStatus
        assert result.status == AgentStatus.COMPLETED
        assert result.output["mode"] == "SOFT"

    def test_exposure_monitor_agent(self):
        from agents.portfolio_agents import ExposureMonitorAgent
        pair = make_pair("AAPL", "MSFT")
        decisions = [make_allocation_decision(pair)]
        agent = ExposureMonitorAgent()
        task = self._task("compute_exposure", {
            "active_allocations": decisions,
            "total_capital": 1_000_000.0,
        })
        result = agent.execute(task)
        from core.contracts import AgentStatus
        assert result.status == AgentStatus.COMPLETED
        assert "gross_leverage" in result.output

    def test_derisking_agent_normal_heat(self):
        from agents.portfolio_agents import DeRiskingAgent
        agent = DeRiskingAgent()
        dd = DrawdownState(heat_level=PortfolioHeatLevel.NORMAL)
        task = self._task("compute_derisking", {
            "active_allocations": [],
            "drawdown_state": dd,
        })
        result = agent.execute(task)
        from core.contracts import AgentStatus
        assert result.status == AgentStatus.COMPLETED
        assert result.output["n_exits"] == 0

    def test_derisking_agent_halted(self):
        from agents.portfolio_agents import DeRiskingAgent
        agent = DeRiskingAgent()
        pairs = [make_pair("A", "B"), make_pair("C", "D")]
        decisions = [make_allocation_decision(p) for p in pairs]
        dd = DrawdownState(heat_level=PortfolioHeatLevel.HALTED, throttle_factor=0.0)
        task = self._task("compute_derisking", {
            "active_allocations": decisions,
            "drawdown_state": dd,
        })
        result = agent.execute(task)
        from core.contracts import AgentStatus
        assert result.status == AgentStatus.COMPLETED
        assert result.output["n_exits"] == 2

    def test_portfolio_construction_agent_basic(self):
        from agents.portfolio_agents import PortfolioConstructionAgent
        agent = PortfolioConstructionAgent()
        intents = [
            make_entry_intent("AAPL", "MSFT", z_score=2.8, confidence=0.75),
        ]
        task = self._task("run_allocation_cycle", {
            "intents": intents,
            "total_capital": 1_000_000.0,
        })
        result = agent.execute(task)
        from core.contracts import AgentStatus
        assert result.status == AgentStatus.COMPLETED
        assert "n_funded" in result.output
        assert "diagnostics" in result.output

    def test_agent_invalid_task_type_fails(self):
        from agents.portfolio_agents import DrawdownMonitorAgent
        from core.contracts import AgentStatus
        agent = DrawdownMonitorAgent()
        task = self._task("bad_task_type", {"current_value": 1.0, "peak_value": 1.0})
        result = agent.execute(task)
        assert result.status == AgentStatus.FAILED

    def test_agent_missing_required_payload_fails(self):
        from agents.portfolio_agents import CapitalBudgetAgent
        from core.contracts import AgentStatus
        agent = CapitalBudgetAgent()
        task = self._task("check_capital_budget", {"total_capital": 1_000_000.0})  # missing 'pairs'
        result = agent.execute(task)
        assert result.status == AgentStatus.FAILED


# ════════════════════════════════════════════════════════════════
# Integration Tests
# ════════════════════════════════════════════════════════════════

class TestPortfolioIntegration:

    def test_full_cycle_flow(self):
        """
        End-to-end: intents → rank → size → allocate → snapshot.
        Validates capital is tracked correctly across the cycle.
        """
        capital_mgr = CapitalManager(total_capital=2_000_000.0)
        capital_mgr.add_sleeve(SleeveDef("default", max_capital_fraction=1.0))

        allocator = PortfolioAllocator(capital_mgr)
        intents = [
            make_entry_intent("AAPL", "MSFT", z_score=3.0, confidence=0.85),
            make_entry_intent("GOOG", "META", z_score=2.7, confidence=0.70),
            make_entry_intent("JPM", "BAC", z_score=2.5, confidence=0.65),
            make_entry_intent("XOM", "CVX", quality_grade="F"),  # Should be blocked
        ]

        decisions, diag = allocator.run_cycle(intents)

        funded = [d for d in decisions if d.approved]
        assert len(funded) >= 2  # At least the good ones get funded
        assert diag.n_blocked_signal >= 1  # grade F blocked

        # Capital should be allocated
        pool = capital_mgr.pool_snapshot()
        assert pool.allocated_capital > 0

        # Build analytics
        analytics = PortfolioAnalytics()
        snapshot = analytics.build_snapshot(
            capital_mgr, DrawdownState(), KillSwitchState(), funded
        )
        assert snapshot.capital_pool.allocated_capital > 0
        assert snapshot.heat_state.n_active_positions == len(funded)

    def test_capital_exhaustion_blocks_later_entries(self):
        """
        When capital is nearly exhausted, later entries should be blocked.
        """
        capital_mgr = CapitalManager(total_capital=100_000.0)  # Very small
        capital_mgr.add_sleeve(SleeveDef("default", max_capital_fraction=1.0))

        # Use very small min_executable so positions don't get rejected for being too small
        cfg = AllocatorConfig()
        cfg.sizing.min_executable_notional = 1_000.0

        allocator = PortfolioAllocator(capital_mgr, config=cfg)
        intents = [
            make_entry_intent(f"S{i}", f"T{i}", z_score=2.5, confidence=0.7)
            for i in range(20)
        ]
        decisions, diag = allocator.run_cycle(intents)

        funded = [d for d in decisions if d.approved]
        # Can't fund all 20 with only 100k
        # Some should be blocked/unfunded
        assert len(funded) < 20

    def test_risk_ops_and_allocator_integration(self):
        """
        Stress scenario: drawdown hits THROTTLED → allocator respects throttle.
        """
        dd_cfg = DrawdownConfig(throttled_dd_threshold=0.05)
        rom = RiskOperationsManager(drawdown_config=dd_cfg)
        dd_state, ks_state = rom.update(0.94, 1.0)  # 6% DD → THROTTLED

        assert dd_state.heat_level == PortfolioHeatLevel.THROTTLED
        assert dd_state.throttle_factor < 1.0

        capital_mgr = CapitalManager(total_capital=1_000_000.0)
        capital_mgr.add_sleeve(SleeveDef("default", max_capital_fraction=1.0))
        allocator = PortfolioAllocator(capital_mgr)
        intents = [make_entry_intent("AAPL", "MSFT", z_score=3.0, confidence=0.85)]

        decisions, diag = allocator.run_cycle(
            intents,
            drawdown_state=dd_state,
            kill_switch=ks_state,
        )

        funded = [d for d in decisions if d.approved]
        if funded:
            # Sizing should reflect drawdown throttle
            assert funded[0].sizing.drawdown_scalar <= 1.0
