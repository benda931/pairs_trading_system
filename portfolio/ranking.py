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
  1. signal_strength_score    — z-score attractiveness vs threshold       (w=0.20)
  2. signal_quality_score     — quality grade A+→F → numeric              (w=0.16)
  3. regime_suitability_score — how favourable is the current regime      (w=0.15)
  4. reversion_probability    — estimated success probability (rule/ML)   (w=0.12)
  5. diversification_value    — marginal diversification contribution      (w=0.07)
  6. stability_score          — rolling spread stability                   (w=0.05)
  7. freshness_score          — model/signal recency penalty               (w=0.00)
  8. capital_efficiency_score — expected P&L per unit capital × time      (w=0.08)
  9. liquidity_score          — ADV participation rate executability       (w=0.07)
 10. edge_quality_score       — OOS walk-forward edge quality              (w=0.10)

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

import numpy as np

from core.contracts import (
    PairId,
    RegimeLabel,
    SignalQualityGrade,
)
from core.intents import EntryIntent
from core.transaction_costs import TransactionCostModel, estimate_trade_cost
from portfolio.contracts import (
    OpportunitySet,
    RankedOpportunity,
)

logger = logging.getLogger("portfolio.ranking")

# Module-level singleton — avoids re-instantiating per opportunity
_tc_model = TransactionCostModel()


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
    """
    Configurable weights for the 10 ranking dimensions.

    Weight allocation (must sum to 1.0):
      signal_strength    0.20  — z-score attractiveness
      signal_quality     0.16  — grade A+→F
      regime_suitability 0.15  — regime fit
      reversion_prob     0.12  — conviction / ML probability
      diversification    0.07  — marginal portfolio diversification
      stability          0.05  — half-life stability
      freshness          0.00  — signal age (operational, no alpha weight)
      capital_efficiency 0.08  — expected P&L per unit capital × time (NEW)
      liquidity          0.07  — ADV participation rate (NEW)
      edge_quality       0.10  — OOS walk-forward edge (NEW)
                         ────
                         1.00
    """
    signal_strength: float = 0.20
    signal_quality: float = 0.16
    regime_suitability: float = 0.15
    reversion_probability: float = 0.12
    diversification_value: float = 0.07
    stability: float = 0.05
    freshness: float = 0.00
    capital_efficiency: float = 0.08
    liquidity: float = 0.07
    edge_quality: float = 0.10

    def __post_init__(self) -> None:
        total = sum([
            self.signal_strength, self.signal_quality, self.regime_suitability,
            self.reversion_probability, self.diversification_value,
            self.stability, self.freshness,
            self.capital_efficiency, self.liquidity, self.edge_quality,
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

    def __post_init__(self) -> None:
        # Hard governance cap: ML can influence at most 30% of composite score.
        # Prevents misconfiguration from giving ML outsized control.
        MAX_ML_BLEND = 0.30
        if self.ml_blend_fraction > MAX_ML_BLEND:
            import logging as _logging
            _logging.getLogger("portfolio.ranking").warning(
                "ml_blend_fraction=%.2f exceeds cap %.2f — clamped",
                self.ml_blend_fraction, MAX_ML_BLEND,
            )
            self.ml_blend_fraction = MAX_ML_BLEND


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


def _capital_efficiency_score(
    half_life: Optional[float],
    spread_vol: float,
    abs_z: float,
    entry_z_threshold: float = 2.0,
) -> float:
    """
    Capital efficiency: expected convergence P&L per unit of capital × time.

    Logic: A pair with HL=10d and |z|=2.5 frees up capital 3x faster than
    HL=30d. Higher turnover at same risk = better use of capital.

    Score = (expected_z_reversion / half_life) normalised to [0, 1]
    expected_z_reversion ≈ |z| - exit_z (≈ |z| - 0.5)
    """
    if half_life is None or half_life <= 0 or np.isnan(half_life) or np.isinf(half_life):
        return 0.5  # neutral
    if abs_z <= 0 or spread_vol <= 0:
        return 0.5

    expected_reversion = max(0.0, abs_z - 0.5)   # target: exit at z=0.5
    # Capital efficiency: reversion magnitude / holding period
    raw_efficiency = expected_reversion / half_life
    # Normalise: 0.05 reversion/day → score=0.5; 0.15+ → score=1.0; <0.01 → score=0.0
    score = float(np.clip((raw_efficiency - 0.01) / 0.14, 0.0, 1.0))
    return score


def _liquidity_score(
    avg_dollar_vol_x: Optional[float] = None,
    avg_dollar_vol_y: Optional[float] = None,
    gross_notional: float = 100_000.0,
    adv_participation_cap: float = 0.05,  # max 5% of ADV
) -> float:
    """
    Liquidity quality score [0, 1].

    Checks whether the intended trade size is executable without significant
    market impact. Uses participation rate = trade_size / ADV.

    adv_participation_cap: maximum acceptable fraction of ADV (5% is institutional standard).
    """
    if avg_dollar_vol_x is None or avg_dollar_vol_y is None:
        return 0.7  # unknown → assume reasonable liquidity (don't penalise missing data)

    min_adv = min(avg_dollar_vol_x, avg_dollar_vol_y)
    if min_adv <= 0:
        return 0.1  # effectively illiquid

    # Each leg is roughly half the gross notional
    leg_size = gross_notional / 2.0
    participation_rate = leg_size / min_adv

    if participation_rate <= 0.01:      # <1% ADV: highly liquid
        return 1.0
    elif participation_rate <= 0.05:    # 1-5%: good
        return 1.0 - (participation_rate - 0.01) / 0.04 * 0.3   # 1.0 → 0.7
    elif participation_rate <= 0.20:    # 5-20%: acceptable but degraded
        return 0.7 - (participation_rate - 0.05) / 0.15 * 0.5   # 0.7 → 0.2
    else:                               # >20%: illiquid
        return max(0.0, 0.2 - (participation_rate - 0.20) * 0.5)


def _edge_quality_score(
    oos_sharpe: Optional[float] = None,
    oos_ic: Optional[float] = None,
    n_oos_trades: int = 0,
    stability_score: Optional[float] = None,
) -> float:
    """
    Historical OOS edge quality [0, 1].

    Uses walk-forward OOS results if available. Rewards:
    - Consistent OOS Sharpe (0.5–2.0 is good)
    - Positive OOS IC with statistical significance
    - Sufficient trade count (n >= 10 for meaningful stats)
    """
    components = []

    # OOS Sharpe component
    if oos_sharpe is not None and not np.isnan(oos_sharpe):
        if oos_sharpe <= 0:
            sharpe_score = 0.0
        elif oos_sharpe >= 2.0:
            sharpe_score = 1.0
        else:
            sharpe_score = oos_sharpe / 2.0
        # Discount for insufficient trades
        if n_oos_trades < 5:
            sharpe_score *= 0.3
        elif n_oos_trades < 10:
            sharpe_score *= 0.6
        components.append((sharpe_score, 0.6))

    # OOS IC component
    if oos_ic is not None and not np.isnan(oos_ic):
        ic_score = float(np.clip((oos_ic + 0.05) / 0.15, 0.0, 1.0))  # IC=0.05→0.33, IC=0.20→1.0
        components.append((ic_score, 0.4))

    # Stability score component (from pair_validator)
    if stability_score is not None and not np.isnan(stability_score):
        components.append((float(np.clip(stability_score, 0.0, 1.0)), 0.3))

    if not components:
        return 0.5  # no historical data → neutral

    total_weight = sum(w for _, w in components)
    weighted_score = sum(s * w for s, w in components) / total_weight
    return float(np.clip(weighted_score, 0.0, 1.0))


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
            # Spread volatility — used by capital efficiency score
            spread_vol = float(intent.metadata.get("spread_vol", 1.0) or 1.0)

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

            # ── New extended scores ───────────────────────────────
            abs_z = abs(z_score)
            cap_eff = _capital_efficiency_score(
                half_life=half_life,
                spread_vol=spread_vol,
                abs_z=abs_z,
            )
            liq = _liquidity_score(
                avg_dollar_vol_x=intent.metadata.get("avg_dollar_vol_x"),
                avg_dollar_vol_y=intent.metadata.get("avg_dollar_vol_y"),
                gross_notional=cfg.default_notional if hasattr(cfg, "default_notional") else 100_000.0,
            )
            edge_q = _edge_quality_score(
                oos_sharpe=intent.metadata.get("oos_sharpe"),
                oos_ic=intent.metadata.get("oos_ic"),
                n_oos_trades=int(intent.metadata.get("n_oos_trades", 0)),
                stability_score=intent.metadata.get("stability_score"),
            )

            if ss > 0.7:
                strengths.append(f"strong_z:{z_score:.2f}")
            if sq >= 0.80:
                strengths.append(f"high_quality:{quality_grade}")
            if rs >= 0.85:
                strengths.append(f"ideal_regime:{regime_str}")
            if cap_eff >= 0.70:
                strengths.append(f"high_cap_efficiency:{cap_eff:.2f}")
            if edge_q >= 0.70:
                strengths.append(f"strong_oos_edge:{edge_q:.2f}")

            if ss < 0.3:
                penalties.append(f"weak_signal:{z_score:.2f}")
            if stab < 0.5:
                penalties.append(f"poor_stability:hl={half_life:.1f}d")
            if liq < 0.3:
                penalties.append(f"low_liquidity:{liq:.2f}")

            # ── Overlap penalty ───────────────────────────────────
            overlap_penalty, overlap_notes = _compute_overlap_penalty(
                pair_id, active_pairs, active_cluster_ids, active_instruments, cfg,
            )
            penalties.extend(overlap_notes)

            # ── Composite score ───────────────────────────────────
            signals: list[str] = []
            composite = (
                w.signal_strength      * ss
                + w.signal_quality     * sq
                + w.regime_suitability * rs
                + w.reversion_probability * rp
                + w.diversification_value * dv
                + w.stability          * stab
                + w.freshness          * fs
                + w.capital_efficiency * cap_eff
                + w.liquidity          * liq
                + w.edge_quality       * edge_q
            )
            # Apply overlap penalty to composite
            composite = max(0.0, composite - overlap_penalty)

            # ── Transaction cost gate ─────────────────────────────
            # Estimate round-trip costs and compute breakeven_z. If the entry
            # z-score barely covers costs, penalise or zero out the composite.
            try:
                tc_estimate = _tc_model.estimate(
                    notional_x=50_000,   # half of standard notional
                    notional_y=50_000,
                    price_x=100.0, price_y=100.0,   # normalised
                    adv_x=intent.metadata.get("avg_dollar_vol_x"),
                    adv_y=intent.metadata.get("avg_dollar_vol_y"),
                    spread_bps_x=intent.metadata.get("spread_bps_x"),
                    spread_bps_y=intent.metadata.get("spread_bps_y"),
                    holding_days=max(5, int(half_life)) if (half_life and not math.isnan(half_life)) else 20,
                    spread_vol_pct=float(intent.metadata.get("spread_vol_pct", 0.02)),
                )

                # Store cost audit fields in intent metadata snapshot
                tc_total_bps = tc_estimate.total_bps
                tc_breakeven_z = tc_estimate.breakeven_z

                if tc_breakeven_z > 0:
                    z_coverage = abs_z / max(tc_breakeven_z, 1e-6)
                    if z_coverage < 1.0:
                        # Can't cover costs at current z — zero composite
                        composite = 0.0
                        signals.append("cost_exceeds_signal")
                    elif z_coverage < 2.0:
                        # z covers costs but thin margin — scale down
                        cost_factor = (z_coverage - 1.0)   # 0 at breakeven, 1 at 2x breakeven
                        composite *= max(0.3, cost_factor)
                        signals.append(f"thin_cost_margin_{z_coverage:.1f}x")
                    # else: z >> breakeven → no penalty

            except Exception as _tc_exc:
                tc_total_bps = float("nan")
                tc_breakeven_z = float("nan")
                logger.debug("TC estimate failed for %s: %s", pair_id.label, _tc_exc)

            # Merge TC signals into penalties list for audit
            penalties.extend(signals)
            # Emit cost audit tags regardless of penalty outcome
            if not math.isnan(tc_total_bps):
                penalties.append(f"tc_cost_bps:{tc_total_bps:.1f}")
            if not math.isnan(tc_breakeven_z):
                penalties.append(f"tc_breakeven_z:{tc_breakeven_z:.3f}")

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
                capital_efficiency_score=cap_eff,
                liquidity_score=liq,
                edge_quality_score=edge_q,
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
            # Governance: only blend if model has champion status
            # (hooks that don't expose model_status are treated as non-champion)
            try:
                from ml.contracts import ModelStatus
                hook_status = getattr(self._ml_hook, "model_status", None)
                ml_governed = hook_status == ModelStatus.CHAMPION
            except Exception:
                ml_governed = False

            if not ml_governed:
                logger.warning(
                    "ML hook for ranking skipped: model_status is not CHAMPION "
                    "(model_id=%s). Set model to CHAMPION in the ML registry to enable blending.",
                    getattr(self._ml_hook, "model_id", "unknown"),
                )
            else:
                # Hard cap: blend fraction must not exceed MAX_ML_BLEND
                MAX_ML_BLEND = 0.30
                blend = min(float(cfg.ml_blend_fraction), MAX_ML_BLEND)
                if blend != cfg.ml_blend_fraction:
                    logger.warning(
                        "ml_blend_fraction=%.2f capped to %.2f (MAX_ML_BLEND)",
                        cfg.ml_blend_fraction,
                        MAX_ML_BLEND,
                    )

                for opp in opportunities:
                    try:
                        ml_score = float(self._ml_hook.score(opp))
                        ml_score = max(0.0, min(1.0, ml_score))
                        opp.ml_ranking_score = ml_score
                        opp.ml_model_id = getattr(self._ml_hook, "model_id", "")

                        # Skip blending if ML hook returned its neutral fallback (0.5).
                        # Blending 0.5 into all composites compresses signal dispersion
                        # without adding information — the correct behavior is no blend.
                        if abs(ml_score - 0.5) < 1e-9:
                            logger.debug(
                                "ML score is neutral (0.5) for %s — skipping blend to preserve signal dispersion",
                                opp.pair_id.label,
                            )
                            continue

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
                w.signal_strength      * opp.signal_strength_score
                + w.signal_quality     * opp.signal_quality_score
                + w.regime_suitability * opp.regime_suitability_score
                + w.reversion_probability * opp.reversion_probability
                + w.diversification_value * dv
                + w.stability          * opp.stability_score
                + w.freshness          * opp.freshness_score
                + w.capital_efficiency * opp.capital_efficiency_score
                + w.liquidity          * opp.liquidity_score
                + w.edge_quality       * opp.edge_quality_score
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
