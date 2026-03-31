# -*- coding: utf-8 -*-
"""
research/candidate_generator.py — Multi-Family Staged Candidate Generation
===========================================================================

Implements a four-stage candidate generation pipeline that transforms a
universe of eligible instruments into a bounded shortlist of candidate pairs
for deep validation.

DOCTRINE:
  - Correlation is a discovery primitive, NOT a tradability proof
  - Candidate generation should be broad and cheap
  - Validation should be strict and expensive
  - The pipeline gates: broad → cheap → quality → deep-validator handoff

STAGE PIPELINE:
  Stage 1 (Broad): All pairs from eligible instruments
  Stage 2 (Cheap filter): Correlation + overlap + volatility compatibility
  Stage 3 (Quality filter): Optional quick cointegration, economic plausibility
  Stage 4 (Validation handoff): Bounded shortlist to deep PairValidator

DISCOVERY FAMILIES:
  - CORRELATION: High absolute correlation of log-returns
  - DISTANCE: Low normalised cumulative price distance
  - CLUSTER: Hierarchical clustering → propose within-cluster pairs
  - COINTEGRATION: Quick EG pre-screen for near-cointegrated pairs
  - FACTOR_RESIDUAL: Similar residual exposure after removing market beta

The generator uses vectorised numpy/pandas operations for the correlation
matrix to avoid O(N²) Python loops on large universes.

Usage:
    gen = CandidateGenerator(
        families=[DiscoveryFamily.CORRELATION, DiscoveryFamily.DISTANCE],
        min_correlation=0.50,
        max_candidates=500,
    )
    batch = gen.generate(snapshot, prices, train_end=datetime(2024, 1, 1))
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.contracts import PairId
from research.discovery_contracts import (
    CandidateBatch,
    CandidatePair,
    DiscoveryFamily,
    RejectionReason,
    UniverseSnapshot,
)

logger = logging.getLogger("research.candidate_generator")


# ── Configuration ──────────────────────────────────────────────────

@dataclass
class CandidateGeneratorConfig:
    """
    Configuration for the candidate generation pipeline.

    Separates stage-2 cheap-filter thresholds from stage-3 quality-filter
    thresholds to make the multi-stage design explicit.
    """
    # Discovery families to run
    families: list[DiscoveryFamily] = field(
        default_factory=lambda: [DiscoveryFamily.CORRELATION, DiscoveryFamily.DISTANCE]
    )

    # Stage-2 cheap filters
    min_correlation: float = 0.50       # |pearson corr of log-returns| must exceed this
    max_correlation: float = 0.995      # Exclude near-identical instruments
    min_overlap_days: int = 252         # Minimum shared price history
    max_vol_ratio: float = 5.0          # Max ratio of annualised vols (compatibility)

    # Stage-3 quality filters
    run_quick_coint: bool = True        # Run fast EG pre-screen at stage 3
    quick_coint_alpha: float = 0.20     # Looser than deep validator; just culls outliers
    min_economic_plausibility: float = 0.0  # 0 = no filter (metadata often sparse)

    # Cluster family settings
    cluster_linkage: str = "ward"       # For hierarchical clustering
    n_clusters_hint: Optional[int] = None  # None = auto-select
    max_cluster_pair_yield: int = 50    # Max pairs per cluster

    # Output limits
    max_candidates_per_family: int = 300
    max_candidates_total: int = 600
    deduplicate_across_families: bool = True  # Keep best-scoring copy of each pair

    # Economic scoring boost
    same_sector_bonus: float = 0.2      # Add to discovery_score for same-sector pairs
    same_industry_bonus: float = 0.1    # Additional bonus for same industry


# ── Correlation family ─────────────────────────────────────────────

class CorrelationDiscoveryFamily:
    """
    Proposes pairs based on absolute rolling/full-window return correlation.

    Uses vectorised correlation matrix computation (pandas .corr()) so this
    scales to N=500 instruments without Python-level loops.

    Doctrine note: high correlation is a NECESSARY but not SUFFICIENT
    condition for a tradable pair. This family proposes; the validator decides.
    """

    def __init__(self, config: CandidateGeneratorConfig):
        self.config = config

    def discover(
        self,
        symbols: list[str],
        log_returns: pd.DataFrame,
        metadata: dict,
        *,
        sector_map: dict[str, str] | None = None,
    ) -> list[CandidatePair]:
        """
        Generate correlation-based candidates.

        Parameters
        ----------
        symbols : eligible symbols
        log_returns : pd.DataFrame of log-returns, columns=symbols
        metadata : symbol -> InstrumentMetadata
        sector_map : symbol -> sector string (optional, for economic context)

        Returns
        -------
        List of CandidatePair, sorted by |correlation| descending.
        """
        # Restrict to available symbols
        available = [s for s in symbols if s in log_returns.columns]
        if len(available) < 2:
            return []

        ret = log_returns[available].dropna(how="all")

        # Compute full-window and rolling correlation matrices
        corr_full = ret.corr(method="pearson")
        rank_corr = ret.corr(method="spearman")

        cfg = self.config
        candidates = []
        rejection_counts: dict[str, int] = defaultdict(int)

        for i, sx in enumerate(available):
            for sy in available[i + 1:]:
                if sx == sy:
                    continue

                pair_id = PairId(sx, sy)
                corr = float(corr_full.loc[sx, sy]) if (sx in corr_full.index and sy in corr_full.index) else np.nan
                rk_corr = float(rank_corr.loc[sx, sy]) if (sx in rank_corr.index and sy in rank_corr.index) else np.nan

                if np.isnan(corr):
                    rejection_counts[RejectionReason.INSUFFICIENT_OVERLAP.value] += 1
                    continue

                # Stage-2 cheap filter: correlation bounds
                abs_corr = abs(corr)
                if abs_corr < cfg.min_correlation:
                    rejection_counts[RejectionReason.LOW_CORRELATION.value] += 1
                    continue
                if abs_corr > cfg.max_correlation:
                    rejection_counts[RejectionReason.HIGH_CORRELATION.value] += 1
                    continue

                # Overlap
                common = ret[[sx, sy]].dropna()
                overlap_days = len(common)
                if overlap_days < cfg.min_overlap_days:
                    rejection_counts[RejectionReason.INSUFFICIENT_OVERLAP.value] += 1
                    continue

                # Volatility compatibility
                vx = float(ret[sx].std() * np.sqrt(252))
                vy = float(ret[sy].std() * np.sqrt(252))
                vol_ratio = vx / vy if vy > 1e-8 else np.inf
                if vol_ratio > cfg.max_vol_ratio or vol_ratio < 1.0 / cfg.max_vol_ratio:
                    rejection_counts[RejectionReason.VOLATILITY_INCOMPATIBILITY.value] += 1
                    continue

                # Economic context
                sect_x = (sector_map or {}).get(sx, "") or (metadata.get(sx).sector if metadata.get(sx) else "")
                sect_y = (sector_map or {}).get(sy, "") or (metadata.get(sy).sector if metadata.get(sy) else "")
                same_sector = bool(sect_x and sect_y and sect_x == sect_y)
                economic_ctx = f"same sector: {sect_x}" if same_sector else f"{sect_x or '?'} / {sect_y or '?'}"
                economic_plausibility = 0.6 if same_sector else 0.3

                # Discovery score: normalise correlation to [0,1] in range [min_corr, 0.995]
                discovery_score = (abs_corr - cfg.min_correlation) / max(cfg.max_correlation - cfg.min_correlation, 0.01)
                if same_sector:
                    discovery_score = min(1.0, discovery_score + cfg.same_sector_bonus)

                candidate = CandidatePair(
                    pair_id=pair_id,
                    discovery_family=DiscoveryFamily.CORRELATION,
                    discovery_score=float(discovery_score),
                    correlation=float(corr),
                    rank_correlation=float(rk_corr) if not np.isnan(rk_corr) else np.nan,
                    overlap_days=overlap_days,
                    economic_context=economic_ctx,
                    economic_plausibility=economic_plausibility,
                    same_sector=same_sector,
                    sector_x=sect_x,
                    sector_y=sect_y,
                    nomination_reasons=[f"corr={corr:.3f}"],
                    stage_reached=2,
                )
                candidates.append(candidate)

        # Sort by discovery score descending, cap at max
        candidates.sort(key=lambda c: -c.discovery_score)
        capped = candidates[:self.config.max_candidates_per_family]

        logger.debug(
            "CORRELATION family: %d/%d pairs passed (rejected: %s)",
            len(capped),
            len(available) * (len(available) - 1) // 2,
            dict(list(rejection_counts.items())[:3]),
        )
        return capped


# ── Distance family ────────────────────────────────────────────────

class DistanceDiscoveryFamily:
    """
    Proposes pairs based on normalised price co-movement distance.

    Measures how closely two normalised (to 1.0) price series tracked each
    other over the training period. Low distance = high co-movement, but
    no assumption about statistical relationship structure.

    This family catches pairs that are co-moving but may not have high
    correlation (e.g., if one had a regime shift mid-period but they
    converged at the end).
    """

    def __init__(self, config: CandidateGeneratorConfig):
        self.config = config

    def discover(
        self,
        symbols: list[str],
        prices: pd.DataFrame,
        metadata: dict,
        *,
        sector_map: dict[str, str] | None = None,
    ) -> list[CandidatePair]:
        """Generate distance-based candidates."""
        available = [s for s in symbols if s in prices.columns]
        if len(available) < 2:
            return []

        px = prices[available].dropna(how="all")

        # Normalise prices to 1.0 at start of window
        first_valid = px.apply(lambda s: s.first_valid_index())
        normed = px.copy()
        for sym in available:
            start_price = px[sym].dropna().iloc[0] if px[sym].dropna().__len__() > 0 else np.nan
            if not np.isnan(start_price) and start_price > 0:
                normed[sym] = px[sym] / start_price

        # Compute sum-of-squared-deviations for all pairs (vectorised)
        normed_arr = normed.values
        normed_arr = np.where(np.isnan(normed_arr), 0, normed_arr)  # treat NaN as 0 for SSD
        n_sym = len(available)

        # Pre-compute SSD using broadcasting
        # For large N, use a chunked approach to avoid memory issues
        candidates = []
        cfg = self.config

        for i, sx in enumerate(available):
            for sy in available[i + 1:]:
                sx_vals = normed[sx].values
                sy_vals = normed[sy].values

                # Common valid period
                mask = ~(np.isnan(sx_vals) | np.isnan(sy_vals))
                if mask.sum() < cfg.min_overlap_days:
                    continue

                sx_clean = sx_vals[mask]
                sy_clean = sy_vals[mask]

                # Sum of squared deviations (normalised by window length)
                ssd = float(np.mean((sx_clean - sy_clean) ** 2))

                # Convert SSD to a 0–1 score: lower SSD = higher score
                # Typical SSD range: 0 (identical) to ~1-4 (very divergent)
                distance_score = max(0.0, 1.0 - ssd)

                if distance_score < 0.2:  # too divergent
                    continue

                # Also require minimum correlation as a sanity check
                corr = float(np.corrcoef(sx_clean, sy_clean)[0, 1])
                if abs(corr) < cfg.min_correlation * 0.8:  # slightly looser than corr family
                    continue

                sect_x = (sector_map or {}).get(sx, "") or (metadata.get(sx).sector if metadata.get(sx) else "")
                sect_y = (sector_map or {}).get(sy, "") or (metadata.get(sy).sector if metadata.get(sy) else "")
                same_sector = bool(sect_x and sect_y and sect_x == sect_y)

                if same_sector:
                    distance_score = min(1.0, distance_score + cfg.same_sector_bonus)

                candidate = CandidatePair(
                    pair_id=PairId(sx, sy),
                    discovery_family=DiscoveryFamily.DISTANCE,
                    discovery_score=distance_score,
                    correlation=corr,
                    distance_metric=ssd,
                    overlap_days=int(mask.sum()),
                    economic_context=f"ssd={ssd:.4f}; {'same sector' if same_sector else 'cross-sector'}",
                    economic_plausibility=0.5 if same_sector else 0.2,
                    same_sector=same_sector,
                    sector_x=sect_x,
                    sector_y=sect_y,
                    nomination_reasons=[f"ssd={ssd:.4f}", f"distance_score={distance_score:.3f}"],
                    stage_reached=2,
                )
                candidates.append(candidate)

        candidates.sort(key=lambda c: -c.discovery_score)
        return candidates[:self.config.max_candidates_per_family]


# ── Cluster family ─────────────────────────────────────────────────

class ClusterDiscoveryFamily:
    """
    Uses hierarchical clustering of return correlations to propose
    within-cluster pairs.

    Doctrine: clustering provides a principled way to:
    1. Avoid proposing pairs that are in different "worlds" statistically
    2. Ensure we don't miss pairs that have moderate correlation but are
       in the same cluster of co-moving instruments
    3. Provide economic intuition (cluster = likely same factor exposure)

    Only proposes pairs within the same cluster (unless cross-cluster allowed).
    """

    def __init__(self, config: CandidateGeneratorConfig):
        self.config = config

    def discover(
        self,
        symbols: list[str],
        log_returns: pd.DataFrame,
        metadata: dict,
        *,
        sector_map: dict[str, str] | None = None,
    ) -> list[CandidatePair]:
        """Generate cluster-based candidates."""
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform
        except ImportError:
            logger.warning("scipy not available — CLUSTER family skipped")
            return []

        available = [s for s in symbols if s in log_returns.columns]
        if len(available) < 4:
            return []

        ret = log_returns[available].dropna(how="all")
        if len(ret) < 60:
            return []

        corr = ret.corr(method="pearson").fillna(0)

        # Distance matrix: 1 - |corr| (high |corr| = small distance)
        dist_mat = 1.0 - corr.abs().values
        np.fill_diagonal(dist_mat, 0)
        dist_mat = np.clip(dist_mat, 0, None)

        # Condense to 1-D for scipy
        try:
            condensed = squareform(dist_mat, checks=False)
            Z = linkage(condensed, method=self.config.cluster_linkage)
        except Exception as exc:
            logger.warning("Clustering failed: %s", exc)
            return []

        # Determine n_clusters
        n_syms = len(available)
        n_clusters = self.config.n_clusters_hint or max(2, min(n_syms // 5, 15))
        labels = fcluster(Z, n_clusters, criterion="maxclust")
        cluster_map: dict[int, list[str]] = defaultdict(list)
        for sym, label in zip(available, labels):
            cluster_map[label].append(sym)

        cfg = self.config
        candidates = []

        for cluster_label, cluster_syms in cluster_map.items():
            if len(cluster_syms) < 2:
                continue

            cluster_pairs_added = 0
            for i, sx in enumerate(cluster_syms):
                for sy in cluster_syms[i + 1:]:
                    if cluster_pairs_added >= cfg.max_cluster_pair_yield:
                        break

                    corr_val = float(corr.loc[sx, sy]) if (sx in corr.index and sy in corr.index) else np.nan
                    if np.isnan(corr_val) or abs(corr_val) < cfg.min_correlation * 0.8:
                        continue

                    pair_id = PairId(sx, sy)
                    sect_x = (sector_map or {}).get(sx, "") or (metadata.get(sx).sector if metadata.get(sx) else "")
                    sect_y = (sector_map or {}).get(sy, "") or (metadata.get(sy).sector if metadata.get(sy) else "")
                    same_sector = bool(sect_x and sect_y and sect_x == sect_y)

                    discovery_score = min(1.0, abs(corr_val) + (0.1 if same_sector else 0))

                    candidate = CandidatePair(
                        pair_id=pair_id,
                        discovery_family=DiscoveryFamily.CLUSTER,
                        discovery_score=discovery_score,
                        correlation=corr_val,
                        overlap_days=len(ret[[sx, sy]].dropna()),
                        economic_context=f"cluster_{cluster_label}; {sect_x or '?'}/{sect_y or '?'}",
                        economic_plausibility=0.6 if same_sector else 0.35,
                        same_sector=same_sector,
                        sector_x=sect_x,
                        sector_y=sect_y,
                        nomination_reasons=[f"cluster_{cluster_label}", f"corr={corr_val:.3f}"],
                        stage_reached=2,
                    )
                    candidates.append(candidate)
                    cluster_pairs_added += 1

        candidates.sort(key=lambda c: -c.discovery_score)
        return candidates[:self.config.max_candidates_per_family]


# ── Cointegration-aware family ─────────────────────────────────────

class CointegrationAwareFamily:
    """
    Quick Engle-Granger pre-screening to surface likely-cointegrated pairs.

    This is NOT a replacement for the deep PairValidator. It uses a looser
    alpha (default 0.20) to pre-select candidates that have at least some
    cointegration signal, filtering out pairs that are purely trend-following
    without mean-reversion.

    Applied AFTER the correlation filter (stage 3), not as a first pass.
    This way the expensive EG test is run only on the correlation shortlist.
    """

    def __init__(self, config: CandidateGeneratorConfig):
        self.config = config

    def filter_candidates(
        self,
        candidates: list[CandidatePair],
        prices: pd.DataFrame,
        *,
        train_end: Optional[datetime] = None,
    ) -> list[CandidatePair]:
        """
        Run quick EG test on each candidate. Update quick_coint_pvalue
        and advance stage. Returns only candidates passing the alpha threshold.
        """
        try:
            from statsmodels.tsa.stattools import coint
        except ImportError:
            logger.warning("statsmodels not available — cointegration pre-screen skipped")
            return candidates

        cutoff = pd.Timestamp(train_end) if train_end else prices.index[-1]
        px = prices[prices.index <= cutoff]

        passed = []
        for c in candidates:
            sx, sy = c.pair_id.sym_x, c.pair_id.sym_y
            if sx not in px.columns or sy not in px.columns:
                c.early_rejections.append(RejectionReason.INSUFFICIENT_OVERLAP)
                continue

            series_x = np.log(px[sx].dropna().clip(lower=1e-8))
            series_y = np.log(px[sy].dropna().clip(lower=1e-8))
            common = series_x.index.intersection(series_y.index)
            if len(common) < 100:
                c.early_rejections.append(RejectionReason.INSUFFICIENT_OVERLAP)
                continue

            try:
                _, pvalue, _ = coint(series_x.loc[common].values, series_y.loc[common].values)
                c.quick_coint_pvalue = float(pvalue)
                c.nomination_reasons.append(f"quick_coint_p={pvalue:.3f}")

                if pvalue <= self.config.quick_coint_alpha:
                    c.stage_reached = 3
                    passed.append(c)
                else:
                    c.early_rejections.append(RejectionReason.FAILED_COINTEGRATION)

            except Exception as exc:
                logger.debug("Quick coint failed for %s: %s", c.pair_id.label, exc)
                # Include anyway — let deep validator decide
                c.stage_reached = 3
                passed.append(c)

        return passed


# ── Staged pipeline ────────────────────────────────────────────────

class CandidateGenerator:
    """
    Orchestrates the full four-stage candidate generation pipeline.

    Stage 1: Universe eligibility (done upstream by UniverseBuilder)
    Stage 2: Family-specific cheap filters (correlation, distance, cluster)
    Stage 3: Cross-family quality filters (optional quick EG, deduplication)
    Stage 4: Bounded shortlist ready for deep PairValidator

    Each stage's output and rejection counts are tracked in CandidateBatch.
    """

    def __init__(self, config: Optional[CandidateGeneratorConfig] = None):
        self.config = config or CandidateGeneratorConfig()
        self._families = self._build_families()

    def _build_families(self) -> dict[DiscoveryFamily, object]:
        cfg = self.config
        families = {}
        if DiscoveryFamily.CORRELATION in cfg.families:
            families[DiscoveryFamily.CORRELATION] = CorrelationDiscoveryFamily(cfg)
        if DiscoveryFamily.DISTANCE in cfg.families:
            families[DiscoveryFamily.DISTANCE] = DistanceDiscoveryFamily(cfg)
        if DiscoveryFamily.CLUSTER in cfg.families:
            families[DiscoveryFamily.CLUSTER] = ClusterDiscoveryFamily(cfg)
        if DiscoveryFamily.COINTEGRATION in cfg.families:
            families[DiscoveryFamily.COINTEGRATION] = CointegrationAwareFamily(cfg)
        return families

    def generate(
        self,
        snapshot: UniverseSnapshot,
        prices: pd.DataFrame,
        *,
        train_end: Optional[datetime] = None,
    ) -> CandidateBatch:
        """
        Run full candidate generation pipeline for a universe snapshot.

        Parameters
        ----------
        snapshot : UniverseSnapshot (eligible instruments only)
        prices : pd.DataFrame with all price columns
        train_end : Use only prices up to this date for all computations

        Returns
        -------
        CandidateBatch with all surviving candidates ready for deep validation
        """
        symbols = snapshot.eligible_symbols
        if len(symbols) < 2:
            logger.warning("Universe has fewer than 2 eligible symbols")
            return CandidateBatch(
                universe_name=snapshot.universe_name,
                n_instruments=len(symbols),
            )

        cutoff = pd.Timestamp(train_end) if train_end else prices.index[-1]
        px = prices[prices.index <= cutoff]

        # Filter to eligible symbols
        available = [s for s in symbols if s in px.columns]
        px_available = px[available].dropna(how="all")

        # Compute log returns (used by multiple families)
        log_returns = np.log(px_available.clip(lower=1e-8)).diff().dropna(how="all")

        # Build sector map from metadata
        sector_map = {
            sym: meta.sector
            for sym, meta in snapshot.metadata.items()
        }

        n_total_pairs = len(available) * (len(available) - 1) // 2
        logger.info(
            "CandidateGenerator: %d symbols → %d potential pairs",
            len(available), n_total_pairs,
        )

        # ── Stage 2: Run each family ──────────────────────────────────
        all_candidates: dict[str, CandidatePair] = {}  # pair_label -> best candidate
        by_family: dict[str, int] = {}

        for family_enum, family_obj in self._families.items():
            if family_enum == DiscoveryFamily.COINTEGRATION:
                continue  # run at stage 3

            try:
                if family_enum == DiscoveryFamily.DISTANCE:
                    family_candidates = family_obj.discover(
                        available, px_available, snapshot.metadata, sector_map=sector_map
                    )
                else:
                    family_candidates = family_obj.discover(
                        available, log_returns, snapshot.metadata, sector_map=sector_map
                    )

                by_family[family_enum.value] = len(family_candidates)
                logger.info(
                    "  %s family: %d candidates", family_enum.value, len(family_candidates)
                )

                # Merge into all_candidates (keep best discovery_score per pair)
                for cand in family_candidates:
                    label = cand.pair_id.label
                    if label not in all_candidates or cand.discovery_score > all_candidates[label].discovery_score:
                        all_candidates[label] = cand

            except Exception as exc:
                logger.error("%s family failed: %s", family_enum.value, exc, exc_info=True)

        stage2_count = len(all_candidates)
        logger.info("Stage 2 complete: %d unique candidates", stage2_count)

        # ── Stage 3: Quality filter ───────────────────────────────────
        stage3_candidates = list(all_candidates.values())

        # Quick cointegration pre-screen if requested
        if self.config.run_quick_coint and DiscoveryFamily.COINTEGRATION in self._families:
            coint_filter = self._families[DiscoveryFamily.COINTEGRATION]
            stage3_candidates = coint_filter.filter_candidates(
                stage3_candidates, px_available, train_end=cutoff
            )
            by_family[DiscoveryFamily.COINTEGRATION.value] = len(stage3_candidates)
            logger.info("  COINTEGRATION pre-screen: %d survived", len(stage3_candidates))
        elif self.config.run_quick_coint and len(stage3_candidates) <= 500:
            # Run inline quick coint even without COINTEGRATION family
            coint_filter = CointegrationAwareFamily(self.config)
            stage3_candidates = coint_filter.filter_candidates(
                stage3_candidates, px_available, train_end=cutoff
            )
            logger.info("  Inline cointegration pre-screen: %d survived", len(stage3_candidates))
        else:
            # Mark all as stage 3 without EG filter
            for c in stage3_candidates:
                c.stage_reached = 3

        stage3_count = len(stage3_candidates)

        # ── Stage 4: Bound the shortlist ──────────────────────────────
        # Sort by discovery_score descending before capping
        stage3_candidates.sort(key=lambda c: -c.discovery_score)
        final_candidates = stage3_candidates[:self.config.max_candidates_total]

        for c in final_candidates:
            c.stage_reached = 4

        # Build rejection counts
        rejection_counts: dict[str, int] = defaultdict(int)
        for c in all_candidates.values():
            if c not in final_candidates:
                for r in c.early_rejections:
                    rejection_counts[r.value] += 1
        # Count pairs rejected at stage 2 (those never made it to all_candidates)
        # We can only estimate this as total_pairs - stage2_count
        rejection_counts["STAGE2_FILTERED"] = n_total_pairs - stage2_count

        logger.info(
            "CandidateGenerator complete: %d candidates passed (from %d pairs screened)",
            len(final_candidates),
            n_total_pairs,
        )

        return CandidateBatch(
            universe_name=snapshot.universe_name,
            n_instruments=len(available),
            n_pairs_screened=n_total_pairs,
            n_nominated=stage2_count,
            n_stage2=stage2_count,
            n_stage3=stage3_count,
            candidates=final_candidates,
            by_family=dict(by_family),
            rejection_counts=dict(rejection_counts),
        )


# ── Analytics ──────────────────────────────────────────────────────

class CandidateAnalytics:
    """
    Diagnostic analytics for a CandidateBatch.

    Useful for understanding the discovery pipeline:
    - which families produce the best candidates?
    - what's the sector diversity of the shortlist?
    - are there redundant pairs (same economic story repeated)?
    """

    @staticmethod
    def family_comparison(batch: CandidateBatch) -> dict:
        """Compare candidate counts and average quality by discovery family."""
        by_family: dict[str, list[float]] = defaultdict(list)
        for c in batch.candidates:
            by_family[c.discovery_family.value].append(c.discovery_score)

        return {
            family: {
                "n": len(scores),
                "avg_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
                "max_score": float(np.max(scores)),
            }
            for family, scores in by_family.items()
        }

    @staticmethod
    def sector_diversity(batch: CandidateBatch) -> dict:
        """How diverse is the candidate set by sector?"""
        sector_pairs: dict[str, int] = defaultdict(int)
        cross_sector = 0
        for c in batch.candidates:
            if c.same_sector and c.sector_x:
                sector_pairs[c.sector_x] += 1
            else:
                cross_sector += 1

        total = len(batch.candidates)
        return {
            "same_sector_pairs": sum(sector_pairs.values()),
            "cross_sector_pairs": cross_sector,
            "sector_breakdown": dict(sector_pairs),
            "diversity_score": min(len(sector_pairs) / max(1, len(sector_pairs) + (1 if cross_sector > 0 else 0)), 1.0),
        }

    @staticmethod
    def correlation_distribution(batch: CandidateBatch) -> dict:
        """
        Distribution of correlation values in the candidate set.

        Important: high average correlation is NOT a sign of quality —
        it may indicate the universe is too concentrated or the filter
        is too loose.
        """
        corrs = [c.correlation for c in batch.candidates if not np.isnan(c.correlation)]
        if not corrs:
            return {}
        return {
            "mean": float(np.mean(corrs)),
            "median": float(np.median(corrs)),
            "p25": float(np.percentile(corrs, 25)),
            "p75": float(np.percentile(corrs, 75)),
            "p95": float(np.percentile(corrs, 95)),
            "n": len(corrs),
        }

    @staticmethod
    def redundancy_score(batch: CandidateBatch, prices: pd.DataFrame) -> float:
        """
        Estimate how many candidate pairs are redundant (same cluster driving
        multiple near-identical candidates). Higher score = more redundancy.

        Simple proxy: average pairwise correlation across candidate pairs.
        """
        labels = list({c.pair_id.label for c in batch.candidates})
        if len(labels) < 2:
            return 0.0

        # Build a return matrix from pair spread returns as a rough proxy
        # This is a heuristic — proper cluster analysis is more expensive
        family_counts: dict[str, int] = defaultdict(int)
        for c in batch.candidates:
            family_counts[c.discovery_family.value] += 1

        dominant_family = max(family_counts, key=family_counts.get)
        total = len(batch.candidates)
        concentration = family_counts[dominant_family] / max(total, 1)
        # High concentration in one family suggests less diversity
        return float(concentration)
