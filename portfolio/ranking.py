# -*- coding: utf-8 -*-
"""
portfolio/ranking.py — Opportunity Ranking Engine
==================================================

Converts approved EntryIntent objects into a ranked OpportunitySet
where every pair competes for capital on a level, auditable playing field.

Ranking doctrine:
  - A good pair is not automatically a good portfolio position
  - Diversification value (what does this pair add?) is explicit in the score
  - Overlap penalties are applied before ranking, not after
  - Every score component is recorded for audit

Score decomposition (all [0,1]):
  1. signal_strength_score   — z-score attractiveness vs threshold
  2. signal_quality_score    — quality grade A+→F → numeric
  3. regime_suitability_score — how favourable is the current regime
  4. reversion_probability   — estimated success probability (rule/ML)
  5. diversification_value   — marginal diversification contribution
  6. stability_score         — rolling spread stability
  7. freshness_score         — model/signal recency penalty

composite = weighted sum, with diversification_value and overlap_penalty applied last.

ML hook protocol:
  Implement RankingMLHookProtocol (classify a RankedOpportunity, return [0,1]).
  The hook output *blends* with the rule-based composite; it never replaces it.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from core.contracts import (
    PairId,
    RegimeLabel,
    SignalQualityGrade,
)
from core.intents import EntryIntent
from portfolio.contracts import (
    OpportunitySet,
    RankedOpportunity,
)

logger = logging.getLogger("portfolio.ranking")


# ── ML hook protocol ──────────────────────────────────────────────

@runtime_checkable
class RankingMLHookProtocol(Protocol):
    """Optional ML hook for additional ranking signal."""
    model_id: str

    def score(self, opportunity: RankedOpportunity) -> float:
        """Return reversion probability [0, 1]."""
        ...


# ── Ranking configuration ─────────────────────────────────────────

@dataclass
class RankingWeights:
    """Configurable weights for the 7 ranking dimensions."""
    signal_strength: float = 0.25
    signal_quality: float = 0.20
    regime_suitability: float = 0.20
    reversion_probability: float = 0.15
    diversification_value: float = 0.10
    stability: float = 0.05
    freshness: float = 0.05

    def __post_init__(self) -> None:
        total = sum([
            self.signal_strength, self.signal_quality, self.regime_suitability,
            self.reversion_probability, self.diversification_value,
            self.stability, self.freshness,
        ])
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"RankingWeights must sum to 1.0, got {total:.3f}")


@dataclass
class RankingConfig:
    """Full ranking engine configuration."""
    weights: RankingWeights = field(default_factory=RankingWeights)

    # Regime suitability scores (regime label → [0,1])
    regime_scores: dict[str, float] = field(default_factory=lambda: {
        "MEAN_REVERTING": 1.0,
        "TRENDING": 0.1,
        "VOLATILE": 0.4,
        "CALM": 0.85,
        "NORMAL": 0.7,
        "TENSION": 0.35,
        "CRISIS": 0.0,
        "BROKEN": 0.0,
        "UNKNOWN": 0.4,
    })

    # Quality grade → numeric score
    grade_scores: dict[str, float] = field(default_factory=lambda: {
        "A+": 1.00, "A": 0.90, "B+": 0.80, "B": 0.70,
        "C+": 0.55, "C": 0.45, "D": 0.25, "F": 0.0,
    })

    # Blocking grades — hard block, not just penalised
    blocking_grades: set[str] = field(default_factory=lambda: {"F"})

    # Minimum composite score to enter OpportunitySet
    min_composite_for_funnel: float = 0.10

    # Maximum overlap penalty (applied to composite_score)
    max_overlap_penalty: float = 0.40

    # Shared-leg penalty per pair
    shared_leg_penalty_per_pair: float = 0.10

    # Cluster concentration penalty (applied when cluster > threshold)
    cluster_concentration_threshold: float = 0.25
    cluster_concentration_penalty: float = 0.15

    # ML blend fraction (0 = rule only, 1 = ML only)
    ml_blend_fraction: float = 0.20
    ml_enabled: bool = False

    # Signal strength: entry z threshold (above this → score=1.0)
    z_score_saturation: float = 3.0


# ── Helpers ───────────────────────────────────────────────────────

def _grade_to_score(grade: str, grade_scores: dict[str, float]) -> float:
    return grade_scores.get(grade.upper(), 0.3)


def _signal_strength_score(z_score: float, entry_z: float, saturation_z: float) -> float:
    """Map |z_score| above entry_z to [0,1], saturating at saturation_z."""
    excess = abs(z_score) - abs(entry_z)
    if excess <= 0:
        return 0.0
    span = max(0.01, abs(saturation_z) - abs(entry_z))
    return min(1.0, excess / span)


def _regime_score(regime_label: str, regime_scores: dict[str, float]) -> float:
    return regime_scores.get(regime_label.upper(), 0.4)


def _freshness_score(generated_at: Optional[datetime], max_age_hours: float = 4.0) -> float:
    """Decay signal freshness linearly over max_age_hours."""
    if generated_at is None:
        return 0.5
    age_hours = (datetime.utcnow() - generated_at).total_seconds() / 3600.0
    if age_hours <= 0:
        return 1.0
    return max(0.0, 1.0 - age_hours / max_age_hours)


def _half_life_stability_score(half_life_days: float) -> float:
    """
    Score based on mean-reversion speed.
    Ideal: 5-30 days. Penalty for very fast (<3) or very slow (>90).
    """
    if math.isnan(half_life_days) or half_life_days <= 0:
        return 0.3
    if half_life_days < 2:
        return 0.2     # Too fast — unstable
    if half_life_days <= 5:
        return 0.6
    if half_life_days <= 30:
        return 1.0     # Sweet spot
    if half_life_days <= 60:
        return 0.8
    if half_life_days <= 90:
        return 0.55
    return 0.3         # Too slow — weak mean reversion


# ── Overlap penalty ───────────────────────────────────────────────

def _compute_overlap_penalty(
    pair_id: PairId,
    existing_pairs: list[PairId],
    active_cluster_ids: dict[str, str],   # pair_label → cluster_id
    active_instruments: dict[str, int],   # instrument → n_pairs_using
    config: RankingConfig,
) -> tuple[float, list[str]]:
    """
    Compute total overlap penalty and explanation strings.

    Returns (penalty [0, max_overlap_penalty], penalty_descriptions).
    """
    penalty = 0.0
    notes: list[str] = []

    # Shared-leg penalty
    legs = {pair_id.sym_x, pair_id.sym_y}
    for leg in legs:
        n_using = active_instruments.get(leg, 0)
        if n_using > 0:
            leg_pen = config.shared_leg_penalty_per_pair * n_using
            penalty += leg_pen
            notes.append(f"shared_leg:{leg}×{n_using}")

    # Cluster concentration penalty
    pair_cluster = active_cluster_ids.get(pair_id.label)
    if pair_cluster:
        cluster_members = [
            p for p, c in active_cluster_ids.items()
            if c == pair_cluster and p != pair_id.label
        ]
        cluster_fraction = len(cluster_members) / max(1, len(existing_pairs))
        if cluster_fraction >= config.cluster_concentration_threshold:
            penalty += config.cluster_concentration_penalty
            notes.append(f"cluster_crowded:{pair_cluster}({cluster_fraction:.0%})")

    return min(penalty, config.max_overlap_penalty), notes


# ── Core ranker ───────────────────────────────────────────────────

class OpportunityRanker:
    """
    Converts EntryIntent objects into a ranked OpportunitySet.

    This class is stateless per-call: it does not maintain position history.
    Pass active pair info via the `rank()` call.

    Parameters
    ----------
    config : RankingConfig
    ml_hook : optional RankingMLHookProtocol
    """

    def __init__(
        self,
        config: Optional[RankingConfig] = None,
        ml_hook: Optional[RankingMLHookProtocol] = None,
    ):
        self._config = config or RankingConfig()
        self._ml_hook = ml_hook

    @property
    def config(self) -> RankingConfig:
        return self._config

    def rank(
        self,
        intents: list[EntryIntent],
        *,
        active_pairs: Optional[list[PairId]] = None,
        active_cluster_ids: Optional[dict[str, str]] = None,
        active_instruments: Optional[dict[str, int]] = None,
        entry_z_threshold: float = 2.0,
    ) -> OpportunitySet:
        """
        Score and rank a list of EntryIntent objects.

        Parameters
        ----------
        intents : list[EntryIntent] — approved signals waiting for capital
        active_pairs : list[PairId] — currently open positions
        active_cluster_ids : dict[pair_label → cluster_id] — cluster memberships
        active_instruments : dict[instrument → n_pairs_using] — shared-leg count
        entry_z_threshold : float — minimum |z| for non-zero signal strength score

        Returns
        -------
        OpportunitySet — sorted by rank (lower = better)
        """
        active_pairs = active_pairs or []
        active_cluster_ids = active_cluster_ids or {}
        active_instruments = active_instruments or {}

        cfg = self._config
        w = cfg.weights

        opportunities: list[RankedOpportunity] = []
        n_blocked = 0

        for intent in intents:
            pair_id = intent.pair_id
            blockers: list[str] = []
            strengths: list[str] = []
            penalties: list[str] = []

            # ── Extract context from intent ───────────────────────
            quality_grade = getattr(intent, "quality_grade", "B") or "B"
            if hasattr(quality_grade, "value"):
                quality_grade = quality_grade.value  # SignalQualityGrade enum

            regime = getattr(intent, "regime", None)
            regime_str = regime.value if regime is not None and hasattr(regime, "value") else str(regime or "UNKNOWN")

            z_score = getattr(intent, "z_score", 0.0) or 0.0
            # EntryIntent uses 'confidence'; some callers may pass 'conviction'
            conviction = (
                getattr(intent, "conviction", None)
                or getattr(intent, "confidence", 0.5)
                or 0.5
            )
            # EntryIntent uses 'expected_half_life_days'; accept 'half_life_days' too
            half_life = (
                getattr(intent, "half_life_days", None)
                or getattr(intent, "expected_half_life_days", float("nan"))
            )
            if half_life is None:
                half_life = float("nan")
            generated_at = getattr(intent, "generated_at", None)
            skip = getattr(intent, "skip_recommended", False)

            # ── Hard vetoes ───────────────────────────────────────
            if quality_grade in cfg.blocking_grades:
                blockers.append(f"quality_grade:{quality_grade}")
            if skip:
                blockers.append("signal_analyst:skip_recommended")
            if regime_str in ("CRISIS", "BROKEN"):
                blockers.append(f"regime:{regime_str}")

            # ── Score components ─────────────────────────────────
            ss = _signal_strength_score(z_score, entry_z_threshold, cfg.z_score_saturation)
            sq = _grade_to_score(quality_grade, cfg.grade_scores)
            rs = _regime_score(regime_str, cfg.regime_scores)
            rp = min(1.0, conviction)          # proxy for reversion probability
            fs = _freshness_score(generated_at)
            stab = _half_life_stability_score(half_life)
            dv = 1.0  # diversification_value computed after all pairs scored

            if ss > 0.7:
                strengths.append(f"strong_z:{z_score:.2f}")
            if sq >= 0.80:
                strengths.append(f"high_quality:{quality_grade}")
            if rs >= 0.85:
                strengths.append(f"ideal_regime:{regime_str}")

            if ss < 0.3:
                penalties.append(f"weak_signal:{z_score:.2f}")
            if stab < 0.5:
                penalties.append(f"poor_stability:hl={half_life:.1f}d")

            # ── Overlap penalty ───────────────────────────────────
            overlap_penalty, overlap_notes = _compute_overlap_penalty(
                pair_id, active_pairs, active_cluster_ids, active_instruments, cfg,
            )
            penalties.extend(overlap_notes)

            # ── Composite score ───────────────────────────────────
            composite = (
                w.signal_strength    * ss
                + w.signal_quality   * sq
                + w.regime_suitability * rs
                + w.reversion_probability * rp
                + w.diversification_value * dv
                + w.stability        * stab
                + w.freshness        * fs
            )
            # Apply overlap penalty to composite
            composite = max(0.0, composite - overlap_penalty)

            opp = RankedOpportunity(
                pair_id=pair_id,
                raw_pair_label=pair_id.label,
                signal_strength_score=ss,
                signal_quality_score=sq,
                regime_suitability_score=rs,
                reversion_probability=rp,
                diversification_value=dv,
                stability_score=stab,
                freshness_score=fs,
                composite_score=composite,
                rank=0,  # assigned below
                quality_grade=quality_grade,
                regime=regime_str,
                conviction=conviction,
                z_score=z_score,
                half_life_days=half_life,
                recommended_sleeve=self._suggest_sleeve(regime_str, quality_grade),
                overlap_penalty=overlap_penalty,
                blockers=blockers,
                strengths=strengths,
                penalties=penalties,
            )
            opportunities.append(opp)

            if blockers:
                n_blocked += 1

        # ── Diversification adjustment ────────────────────────────
        self._apply_diversification_scores(opportunities, active_pairs)

        # ── ML overlay ────────────────────────────────────────────
        if cfg.ml_enabled and self._ml_hook is not None:
            for opp in opportunities:
                try:
                    ml_score = float(self._ml_hook.score(opp))
                    ml_score = max(0.0, min(1.0, ml_score))
                    opp.ml_ranking_score = ml_score
                    opp.ml_model_id = self._ml_hook.model_id
                    # Blend ML score into composite
                    blend = cfg.ml_blend_fraction
                    opp.composite_score = (1 - blend) * opp.composite_score + blend * ml_score
                except Exception as exc:
                    logger.warning("ML hook failed for %s: %s", opp.pair_id.label, exc)

        # ── Sort and assign ranks ─────────────────────────────────
        # Blocked opportunities go to bottom; within each group, sort by composite desc
        def sort_key(o: RankedOpportunity) -> tuple:
            return (1 if o.blockers else 0, -o.composite_score)

        opportunities.sort(key=sort_key)
        for i, opp in enumerate(opportunities):
            opp.rank = i + 1

        n_fundable = sum(1 for o in opportunities if o.is_fundable())

        logger.info(
            "Ranked %d intents: %d fundable, %d blocked",
            len(intents), n_fundable, n_blocked,
        )

        return OpportunitySet(
            opportunities=opportunities,
            generated_at=datetime.utcnow(),
            n_input_intents=len(intents),
            n_blocked=n_blocked,
            n_fundable=n_fundable,
        )

    def _apply_diversification_scores(
        self,
        opportunities: list[RankedOpportunity],
        active_pairs: list[PairId],
    ) -> None:
        """
        Adjust diversification_value for each opportunity.

        Pairs whose legs are not yet represented in active portfolio get
        full diversification credit (1.0). Pairs that duplicate active
        legs get penalised.
        """
        cfg = self._config
        active_legs: set[str] = set()
        for p in active_pairs:
            active_legs.add(p.sym_x)
            active_legs.add(p.sym_y)

        # Track legs being added in this ranking cycle (order matters)
        cycle_legs: set[str] = set()

        for opp in opportunities:
            if opp.blockers:
                # Blocked pairs still get their diversification value, but it won't matter
                continue
            legs = {opp.pair_id.sym_x, opp.pair_id.sym_y}
            overlap_with_active = len(legs & active_legs)
            overlap_with_cycle = len(legs & cycle_legs)

            if overlap_with_active == 0 and overlap_with_cycle == 0:
                dv = 1.0
            elif overlap_with_active + overlap_with_cycle >= 2:
                dv = 0.4   # Both legs already represented
            else:
                dv = 0.7   # One new leg

            opp.diversification_value = dv

            # Recompute composite with updated diversification_value
            w = cfg.weights
            opp.composite_score = (
                w.signal_strength    * opp.signal_strength_score
                + w.signal_quality   * opp.signal_quality_score
                + w.regime_suitability * opp.regime_suitability_score
                + w.reversion_probability * opp.reversion_probability
                + w.diversification_value * dv
                + w.stability        * opp.stability_score
                + w.freshness        * opp.freshness_score
            )
            opp.composite_score = max(0.0, opp.composite_score - opp.overlap_penalty)

            cycle_legs.update(legs)

    def _suggest_sleeve(self, regime: str, quality_grade: str) -> str:
        """Heuristic sleeve assignment based on regime and quality."""
        if regime in ("CRISIS", "BROKEN"):
            return "defensive"
        if quality_grade in ("A+", "A", "B+") and regime == "MEAN_REVERTING":
            return "high_conviction"
        return "default"
