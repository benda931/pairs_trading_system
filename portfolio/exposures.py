# -*- coding: utf-8 -*-
"""
portfolio/exposures.py — Portfolio Exposure Analysis
=====================================================

Computes and monitors all exposure dimensions for the portfolio:

  1. Gross/net leverage
  2. Sector exposure (by instrument sector membership)
  3. Cluster exposure (by correlation cluster)
  4. Shared-leg concentration (how many pairs use the same instrument)
  5. Factor exposures (optional; pass factor_betas dict)

ExposureAnalyzer is stateless — pass it the current allocations and it
returns an ExposureSummary. No side effects.

Shared-leg doctrine:
  A pair with (AAPL, MSFT) creates directional exposure to both instruments.
  If 5 pairs all use AAPL as a leg, the portfolio has significant single-name
  concentration that pair-level neutrality analysis would miss completely.
  ExposureAnalyzer flags instruments used in > threshold × total pairs.

Usage:
    analyzer = ExposureAnalyzer(config)
    summary = analyzer.compute(active_allocations, total_capital, sector_map, cluster_map)
    if summary.max_sector_concentration > 0.40:
        # alert / block
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from core.contracts import PairId
from portfolio.contracts import (
    AllocationDecision,
    ClusterExposureSummary,
    ExposureContribution,
    ExposureSummary,
    SharedLegSummary,
    SizingDecision,
)

logger = logging.getLogger("portfolio.exposures")


# ── Configuration ─────────────────────────────────────────────────

@dataclass
class ExposureConfig:
    """Concentration limits and thresholds for exposure analysis."""

    # Sector concentration: max fraction of gross exposure in one sector
    max_sector_fraction: float = 0.40

    # Cluster concentration: max fraction of gross exposure in one cluster
    max_cluster_fraction: float = 0.25

    # Shared-leg: instrument used in > this many pairs → dominant
    shared_leg_threshold: int = 3

    # Shared-leg: instrument notional > this fraction of total → dominant
    shared_leg_notional_threshold: float = 0.20

    # Gross leverage limit
    max_gross_leverage: float = 4.0

    # Net leverage limit
    max_net_leverage: float = 1.5

    # Default sector when not in map
    default_sector: str = "UNKNOWN"

    # Default cluster when not in map
    default_cluster: str = "UNKNOWN"


# ── Exposure Analyzer ─────────────────────────────────────────────

class ExposureAnalyzer:
    """
    Computes ExposureSummary from active AllocationDecisions.

    Parameters
    ----------
    config : ExposureConfig
    """

    def __init__(self, config: Optional[ExposureConfig] = None):
        self._cfg = config or ExposureConfig()

    @property
    def config(self) -> ExposureConfig:
        return self._cfg

    def compute(
        self,
        active_allocations: list[AllocationDecision],
        total_capital: float,
        *,
        sector_map: Optional[dict[str, str]] = None,  # instrument → sector
        cluster_map: Optional[dict[str, str]] = None,  # pair_label → cluster_id
        factor_betas: Optional[dict[str, dict[str, float]]] = None,  # pair_label → {factor: beta}
    ) -> ExposureSummary:
        """
        Compute full exposure summary from active allocations.

        Parameters
        ----------
        active_allocations : list[AllocationDecision]
        total_capital : float
        sector_map : dict[instrument → sector]
        cluster_map : dict[pair_label → cluster_id]
        factor_betas : dict[pair_label → {factor: beta}]
        """
        sector_map = sector_map or {}
        cluster_map = cluster_map or {}
        factor_betas = factor_betas or {}
        cfg = self._cfg

        if total_capital <= 0:
            return ExposureSummary(total_capital=total_capital)

        # ── Collect contributions ─────────────────────────────────
        contributions: list[ExposureContribution] = []
        by_sector: dict[str, float] = defaultdict(float)
        by_cluster: dict[str, float] = defaultdict(float)
        by_regime: dict[str, float] = defaultdict(float)
        by_sleeve: dict[str, float] = defaultdict(float)

        # Shared-leg tracking: instrument → (total_notional, net_notional, [pair_labels])
        leg_notional: dict[str, float] = defaultdict(float)
        leg_net: dict[str, float] = defaultdict(float)
        leg_pairs: dict[str, list[str]] = defaultdict(list)

        gross_exposure = 0.0
        net_exposure = 0.0

        for decision in active_allocations:
            if not decision.approved:
                continue

            sizing = decision.sizing
            pair_id = decision.pair_id
            label = pair_id.label

            notional = sizing.gross_notional
            direction = sizing.direction  # "LONG_SPREAD" or "SHORT_SPREAD"

            sector_x = sector_map.get(pair_id.sym_x, cfg.default_sector)
            sector_y = sector_map.get(pair_id.sym_y, cfg.default_sector)
            # Use the "dominant" sector (both stocks usually same sector for pairs)
            sector = sector_x if sector_x == sector_y else f"{sector_x}/{sector_y}"
            cluster = cluster_map.get(label, cfg.default_cluster)
            regime = decision.rationale.sleeve or "UNKNOWN"

            contributions.append(ExposureContribution(
                pair_id=pair_id,
                notional=notional,
                weight=notional / total_capital,
                direction=direction,
                sector=sector,
                cluster_id=cluster,
                leg_x=pair_id.sym_x,
                leg_y=pair_id.sym_y,
            ))

            # Aggregate by dimension
            by_sector[sector] += notional
            by_cluster[cluster] += notional
            by_regime[regime] += notional
            by_sleeve[decision.rationale.sleeve or "default"] += notional

            gross_exposure += notional

            # Net exposure: long spread = long X short Y → contribution is delta
            # Use leg-level notionals
            sign = 1.0 if direction == "LONG_SPREAD" else -1.0
            net_exposure += sizing.leg_x_notional * sign - sizing.leg_y_notional * sign

            # Shared-leg tracking
            # leg_x direction: LONG_SPREAD → long X, SHORT_SPREAD → short X
            leg_x_sign = 1.0 if direction == "LONG_SPREAD" else -1.0
            leg_notional[pair_id.sym_x] += sizing.leg_x_notional
            leg_net[pair_id.sym_x] += leg_x_sign * sizing.leg_x_notional
            leg_pairs[pair_id.sym_x].append(label)

            # leg_y direction: opposite of leg_x
            leg_notional[pair_id.sym_y] += sizing.leg_y_notional
            leg_net[pair_id.sym_y] += -leg_x_sign * sizing.leg_y_notional
            leg_pairs[pair_id.sym_y].append(label)

            # Factor exposures
            if label in factor_betas:
                pass  # aggregated separately below

        # ── Leverage ──────────────────────────────────────────────
        gross_leverage = gross_exposure / max(1.0, total_capital)
        net_leverage = abs(net_exposure) / max(1.0, total_capital)

        # ── Normalise by-dimension ────────────────────────────────
        sector_fractions = {s: v / max(1.0, gross_exposure) for s, v in by_sector.items()}
        cluster_fractions = {c: v / max(1.0, gross_exposure) for c, v in by_cluster.items()}

        # ── Shared-leg summaries ──────────────────────────────────
        shared_legs: list[SharedLegSummary] = []
        for instrument, total_n in leg_notional.items():
            n_pairs = len(leg_pairs[instrument])
            concentration = total_n / max(1.0, gross_exposure)
            is_dominant = (
                n_pairs >= cfg.shared_leg_threshold
                or concentration >= cfg.shared_leg_notional_threshold
            )
            shared_legs.append(SharedLegSummary(
                instrument=instrument,
                n_pairs_using=n_pairs,
                total_notional=total_n,
                net_notional=leg_net[instrument],
                pairs=leg_pairs[instrument],
                is_dominant=is_dominant,
                concentration_score=concentration,
            ))

        dominant_legs = [s.instrument for s in shared_legs if s.is_dominant]

        # ── Cluster summaries ─────────────────────────────────────
        cluster_exposures: list[ClusterExposureSummary] = []
        for cluster_id, notional in by_cluster.items():
            fraction = notional / max(1.0, gross_exposure)
            pairs_in_cluster = [
                c.pair_id.label for c in contributions
                if c.cluster_id == cluster_id
            ]
            cluster_exposures.append(ClusterExposureSummary(
                cluster_id=cluster_id,
                n_pairs=len(pairs_in_cluster),
                total_notional=notional,
                fraction_of_portfolio=fraction,
                pairs=pairs_in_cluster,
                is_overcrowded=fraction > cfg.max_cluster_fraction,
                max_allowed_fraction=cfg.max_cluster_fraction,
            ))

        # ── Factor exposures ──────────────────────────────────────
        aggregated_factors: dict[str, float] = defaultdict(float)
        for decision in active_allocations:
            label = decision.pair_id.label
            if label in factor_betas:
                weight = decision.sizing.weight_of_portfolio
                for factor, beta in factor_betas[label].items():
                    aggregated_factors[factor] += beta * weight

        max_sector = max(sector_fractions.values(), default=0.0)
        max_cluster = max(cluster_fractions.values(), default=0.0)

        return ExposureSummary(
            total_capital=total_capital,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            gross_leverage=gross_leverage,
            net_leverage=net_leverage,
            by_sector=dict(sector_fractions),
            by_cluster=dict(cluster_fractions),
            by_regime=dict(by_regime),
            by_sleeve=dict(by_sleeve),
            shared_legs=shared_legs,
            cluster_exposures=cluster_exposures,
            factor_exposures=dict(aggregated_factors),
            max_sector_concentration=max_sector,
            max_cluster_concentration=max_cluster,
            dominant_legs=dominant_legs,
        )

    def check_new_position(
        self,
        proposal_pair: PairId,
        proposal_notional: float,
        current_summary: ExposureSummary,
        total_capital: float,
        *,
        sector_map: Optional[dict[str, str]] = None,
        cluster_map: Optional[dict[str, str]] = None,
    ) -> list[str]:
        """
        Check if adding a new position would breach any exposure constraint.

        Returns list of violation descriptions (empty = OK).
        """
        sector_map = sector_map or {}
        cluster_map = cluster_map or {}
        cfg = self._cfg
        violations: list[str] = []

        new_gross = current_summary.gross_exposure + proposal_notional
        new_leverage = new_gross / max(1.0, total_capital)
        if new_leverage > cfg.max_gross_leverage:
            violations.append(
                f"gross_leverage:{new_leverage:.2f} > limit:{cfg.max_gross_leverage}"
            )

        # Sector check
        sector_x = sector_map.get(proposal_pair.sym_x, cfg.default_sector)
        sector_y = sector_map.get(proposal_pair.sym_y, cfg.default_sector)
        sector = sector_x if sector_x == sector_y else f"{sector_x}/{sector_y}"
        current_sector_notional = current_summary.by_sector.get(sector, 0.0) * current_summary.gross_exposure
        new_sector_fraction = (current_sector_notional + proposal_notional) / max(1.0, new_gross)
        if new_sector_fraction > cfg.max_sector_fraction:
            violations.append(
                f"sector:{sector} at {new_sector_fraction:.1%} > limit:{cfg.max_sector_fraction:.1%}"
            )

        # Cluster check
        cluster = cluster_map.get(proposal_pair.label, cfg.default_cluster)
        current_cluster_notional = current_summary.by_cluster.get(cluster, 0.0) * current_summary.gross_exposure
        new_cluster_fraction = (current_cluster_notional + proposal_notional) / max(1.0, new_gross)
        if new_cluster_fraction > cfg.max_cluster_fraction:
            violations.append(
                f"cluster:{cluster} at {new_cluster_fraction:.1%} > limit:{cfg.max_cluster_fraction:.1%}"
            )

        # Shared-leg check
        for instrument in (proposal_pair.sym_x, proposal_pair.sym_y):
            existing_leg = next(
                (s for s in current_summary.shared_legs if s.instrument == instrument),
                None,
            )
            if existing_leg and existing_leg.n_pairs_using >= cfg.shared_leg_threshold:
                violations.append(
                    f"shared_leg:{instrument} used_in:{existing_leg.n_pairs_using}_pairs"
                )

        return violations

    def instrument_pair_count(self, active_allocations: list[AllocationDecision]) -> dict[str, int]:
        """Return {instrument → n_active_pairs_using_it}."""
        counts: dict[str, int] = defaultdict(int)
        for d in active_allocations:
            if d.approved:
                counts[d.pair_id.sym_x] += 1
                counts[d.pair_id.sym_y] += 1
        return dict(counts)
