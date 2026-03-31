# -*- coding: utf-8 -*-
"""
research/discovery_pipeline.py — End-to-End Discovery Pipeline
===============================================================

Orchestrates the full lifecycle from universe snapshot to ranked
tradable shortlist:

  UniverseSnapshot
       ↓ CandidateGenerator
  CandidateBatch
       ↓ PairValidator (research/pair_validator.py)
  PairValidationReport × N
       ↓ StabilityAnalyzer
  StabilityReport × N
       ↓ SpreadConstructor + DiscoveryScorer
  DiscoveryScore × N + SpreadSpecification × N
       ↓ Ranking + Promotion
  ResearchRunArtifact (ranked shortlist, full audit trail)

Design principles:
  1. The pipeline is fully reproducible: ResearchRunConfig + prices → same output
  2. Every rejection is recorded with a typed RejectionReason
  3. Correlation is NOT tradability: a candidate can fail at any stage
  4. The final ranking combines statistical quality + stability + economic plausibility
  5. False discovery control: we report validation_pass_rate and reject too-small samples

Usage:
    pipeline = DiscoveryPipeline()
    artifact = pipeline.run(
        config=ResearchRunConfig(
            name="SP500_sectors_2024",
            universe_name="us_large_cap",
            ...
        ),
        prices=prices_df,
    )
    print(artifact.to_summary_dict())
    df = pd.DataFrame([r.to_dict() for r in artifact.top_pairs(20)])
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.contracts import PairId, SpreadModel, ValidationResult
from research.candidate_generator import CandidateGenerator, CandidateGeneratorConfig
from research.discovery_contracts import (
    CandidatePair,
    DiscoveryFamily,
    DiscoveryScore,
    RankingResult,
    RejectionReason,
    ResearchRunArtifact,
    ResearchRunConfig,
    SpreadSpecification,
    StabilityReport,
    ValidationDecision,
)
from research.pair_validator import PairValidator, ValidationConfig
from research.spread_constructor import build_spread
from research.stability_analysis import StabilityAnalyzer
from research.universe import UniverseBuilder

logger = logging.getLogger("research.discovery_pipeline")


# ── Discovery scorer ───────────────────────────────────────────────

class DiscoveryScorer:
    """
    Computes multi-dimensional DiscoveryScore from validation + stability reports.

    Scoring dimensions:
      correlation_score  (0–1): quality of the correlation signal
      cointegration_score (0–1): strength + stability of cointegration
      stability_score (0–1):  rolling stability from StabilityReport
      liquidity_score (0–1):  compatibility of ADVs (from metadata)
      economic_score (0–1):   same sector/industry, explainability
      regime_score (0–1):     is current regime suitable?

    Composite score = weighted combination (configurable).
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
    ):
        self.weights = weights or {
            "cointegration": 0.30,
            "stability": 0.25,
            "correlation": 0.15,
            "regime": 0.15,
            "economic": 0.10,
            "liquidity": 0.05,
        }

    def score(
        self,
        pair_id: PairId,
        validation_report,          # PairValidationReport
        stability_report: StabilityReport,
        candidate: Optional[CandidatePair] = None,
    ) -> DiscoveryScore:
        """Compute DiscoveryScore for one pair."""
        ds = DiscoveryScore(pair_id=pair_id)
        caveats = []

        # ── Correlation score ────────────────────────────────────────
        corr = getattr(validation_report, "correlation", np.nan)
        if not np.isnan(corr):
            ds.correlation = corr
            # Score: maps 0.5→0, 0.95→0.7, 0.999→1.0 (avoid too-perfect clones)
            abs_corr = abs(corr)
            if abs_corr >= 0.999:
                ds.correlation_score = 0.3  # Penalise near-identical instruments
                caveats.append("Near-identical instruments (corr≥0.999)")
            else:
                ds.correlation_score = min(1.0, max(0.0, (abs_corr - 0.4) / 0.55))

        # ── Cointegration score ──────────────────────────────────────
        adf_p = getattr(validation_report, "adf_pvalue", np.nan)
        coint_p = getattr(validation_report, "cointegration_pvalue", np.nan)
        half_life = getattr(validation_report, "half_life_days", np.nan)
        hurst = getattr(validation_report, "hurst_exponent", np.nan)

        ds.coint_pvalue = coint_p if not np.isnan(coint_p) else np.nan
        ds.half_life_days = half_life if not np.isnan(half_life) else np.nan
        ds.hurst_exponent = hurst if not np.isnan(hurst) else np.nan

        coint_components = []
        if not np.isnan(adf_p):
            adf_score = max(0.0, 1.0 - adf_p / 0.10)
            coint_components.append(adf_score * 0.4)
        if not np.isnan(coint_p):
            coint_score = max(0.0, 1.0 - coint_p / 0.10)
            coint_components.append(coint_score * 0.4)
        if not np.isnan(half_life):
            # Sweet spot: 5–60 days
            if 5 <= half_life <= 60:
                hl_score = 1.0 - abs(half_life - 20) / 40
            elif half_life < 5:
                hl_score = half_life / 5 * 0.7  # too short
            else:
                hl_score = max(0.0, 1.0 - (half_life - 60) / 60)  # too long
            coint_components.append(max(0.0, hl_score) * 0.2)

        ds.cointegration_score = float(sum(coint_components)) if coint_components else np.nan

        # ── Stability score ──────────────────────────────────────────
        stab_score = stability_report.stability_score if stability_report else np.nan
        ds.stability_score = stab_score
        ds.hedge_ratio_cv = getattr(stability_report, "beta_cv", np.nan)

        if stability_report:
            if stability_report.has_structural_break:
                caveats.append(f"Structural break detected (confidence={stability_report.break_confidence:.2f})")
            if not stability_report.regime_suitable:
                caveats.append(f"Regime: {stability_report.regime_label}")

        # ── Regime score ─────────────────────────────────────────────
        if stability_report and stability_report.regime_suitable:
            ds.regime_score = 1.0
        elif stability_report:
            # Partial credit: structural instability lowers but doesn't zero the score
            ds.regime_score = max(0.0, 0.4 - stability_report.break_confidence * 0.4)
        else:
            ds.regime_score = np.nan

        # ── Economic score ───────────────────────────────────────────
        if candidate is not None:
            ds.economic_score = candidate.economic_plausibility
        else:
            ds.economic_score = 0.3  # default neutral

        # ── Liquidity score ──────────────────────────────────────────
        if candidate is not None and not np.isnan(candidate.liquidity_score):
            ds.liquidity_score = candidate.liquidity_score
        else:
            ds.liquidity_score = 0.5  # neutral default when unknown

        # ── Composite ────────────────────────────────────────────────
        w = self.weights
        components = {}
        available_components = []

        for dim, wt in w.items():
            val = {
                "correlation": ds.correlation_score,
                "cointegration": ds.cointegration_score,
                "stability": ds.stability_score,
                "regime": ds.regime_score,
                "economic": ds.economic_score,
                "liquidity": ds.liquidity_score,
            }.get(dim, np.nan)

            if not np.isnan(val):
                components[dim] = val
                available_components.append((val, wt))

        if available_components:
            total_weight = sum(wt for _, wt in available_components)
            composite = sum(v * wt for v, wt in available_components) / max(total_weight, 1e-8)
            ds.composite_score = float(min(1.0, max(0.0, composite)))
        else:
            ds.composite_score = np.nan

        ds.grade = DiscoveryScore.grade_from_score(ds.composite_score)
        ds.caveats = caveats
        ds.score_breakdown = components

        return ds


# ── Spread specification builder ───────────────────────────────────

class SpreadSpecBuilder:
    """
    Builds a SpreadSpecification by comparing multiple spread models
    and recommending the best fit for each pair.

    Model selection logic:
    - Compute half-life for OLS and Rolling OLS (Kalman is expensive)
    - If hedge ratio CV > 0.3 → prefer Rolling OLS or Kalman
    - If ADF passes with OLS spread → use Static OLS (simpler, more robust)
    - Default to Kalman when relationship drifts significantly
    """

    def build(
        self,
        pair_id: PairId,
        prices: pd.DataFrame,
        validation_report,          # PairValidationReport
        stability_report: StabilityReport,
        *,
        train_end: Optional[datetime] = None,
        window: int = 60,
    ) -> Optional[SpreadSpecification]:
        """Build a SpreadSpecification. Returns None if spread fitting fails."""
        sx, sy = pair_id.sym_x, pair_id.sym_y
        if sx not in prices.columns or sy not in prices.columns:
            return None

        cutoff = pd.Timestamp(train_end) if train_end else prices.index[-1]
        px = prices[prices.index <= cutoff]

        # Determine recommended model
        beta_cv = getattr(stability_report, "beta_cv", 0.0) or 0.0
        half_life = getattr(validation_report, "half_life_days", np.nan)

        if beta_cv > 0.40 or (not np.isnan(half_life) and half_life > 80):
            recommended_model = SpreadModel.KALMAN
            model_reason = f"Kalman preferred: beta_cv={beta_cv:.2f} or long half-life={half_life:.1f}"
        elif beta_cv > 0.20:
            recommended_model = SpreadModel.ROLLING_OLS
            model_reason = f"Rolling OLS: moderate beta_cv={beta_cv:.2f}"
        else:
            recommended_model = SpreadModel.STATIC_OLS
            model_reason = f"Static OLS: stable hedge ratio (beta_cv={beta_cv:.2f})"

        try:
            defn, z_scores = build_spread(
                pair_id, px,
                model=recommended_model,
                train_end=cutoff,
                window=window,
            )
        except Exception as exc:
            logger.warning("%s: spread build failed — %s", pair_id.label, exc)
            return None

        # Spread diagnostics
        z_clean = z_scores.dropna()
        spread_autocorr = float(z_clean.autocorr(lag=1)) if len(z_clean) > 10 else np.nan

        # Entry/exit suggestions based on half-life and Hurst
        hurst = getattr(validation_report, "hurst_exponent", np.nan)
        entry_z = 2.0
        if not np.isnan(hurst):
            # Stronger mean reversion → can use tighter entry
            if hurst < 0.3:
                entry_z = 1.8
            elif hurst > 0.45:
                entry_z = 2.2  # weaker mean reversion → need more z before entry

        max_hold = min(120, max(10, int(3.0 * half_life))) if not np.isnan(half_life) else 60

        warnings = []
        if stability_report.has_structural_break:
            warnings.append(f"Structural break detected (break_confidence={stability_report.break_confidence:.2f})")
        if not stability_report.regime_suitable:
            warnings.append(f"Current regime: {stability_report.regime_label}")
        if beta_cv > 0.30:
            warnings.append(f"Unstable hedge ratio (CV={beta_cv:.2f})")

        return SpreadSpecification(
            pair_id=pair_id,
            recommended_model=recommended_model,
            hedge_ratio=defn.hedge_ratio,
            hedge_ratio_method=defn.hedge_ratio_method,
            intercept=defn.intercept,
            window=window,
            expected_half_life=half_life if not np.isnan(half_life) else np.nan,
            spread_std=defn.std,
            spread_autocorr_1d=spread_autocorr if not np.isnan(spread_autocorr) else np.nan,
            model_selection_reason=model_reason,
            suggested_entry_z=entry_z,
            suggested_exit_z=entry_z / 4.0,
            suggested_stop_z=entry_z * 2.0,
            suggested_max_hold_days=max_hold,
            warnings=warnings,
            diagnostics={
                "beta_cv": beta_cv,
                "hurst": hurst,
                "z_autocorr": spread_autocorr,
            },
        )


# ── Promotion logic ────────────────────────────────────────────────

def _determine_validation_decision(
    validation_report,
    stability_report: StabilityReport,
    discovery_score: DiscoveryScore,
) -> ValidationDecision:
    """
    Map validation + stability + score to a ValidationDecision.

    REJECTED: hard statistical failure
    RESEARCH_ONLY: passes stats but has significant caveats
    WATCHLIST: passes all checks, suitable for live monitoring
    PORTFOLIO_READY: high score + stable + suitable regime
    """
    result = getattr(validation_report, "result", None)

    if result is not None and result == ValidationResult.FAIL:
        return ValidationDecision.REJECTED

    # Check stability
    has_break = stability_report.has_structural_break if stability_report else False
    break_conf = stability_report.break_confidence if stability_report else 0.0
    regime_ok = stability_report.regime_suitable if stability_report else True
    stab_score = stability_report.stability_score if stability_report else 1.0
    is_stable = stability_report.is_stable if stability_report else True

    score = discovery_score.composite_score

    # Demote to RESEARCH_ONLY conditions
    if has_break and break_conf > 0.7:
        return ValidationDecision.RESEARCH_ONLY
    if not is_stable:
        return ValidationDecision.RESEARCH_ONLY
    if np.isnan(score) or score < 0.30:
        return ValidationDecision.RESEARCH_ONLY

    # PORTFOLIO_READY conditions
    if score >= 0.70 and regime_ok and is_stable and not has_break:
        return ValidationDecision.PORTFOLIO_READY

    # WATCHLIST
    if score >= 0.45:
        return ValidationDecision.WATCHLIST

    return ValidationDecision.RESEARCH_ONLY


# ── Main pipeline ──────────────────────────────────────────────────

class DiscoveryPipeline:
    """
    Full discovery pipeline: universe → candidates → validation → ranking.

    Designed to be stateless and reproducible:
    - Same config + prices + train_end → same artifact
    - All state is captured in ResearchRunArtifact

    Architecture:
      1. (Optional) Build UniverseSnapshot from definition
      2. Generate candidates from snapshot
      3. Deep-validate each candidate (PairValidator)
      4. Run stability analysis on passing candidates
      5. Build SpreadSpecification for each valid pair
      6. Score (DiscoveryScorer) and rank
      7. Return ResearchRunArtifact
    """

    def __init__(
        self,
        validator: Optional[PairValidator] = None,
        stability_analyzer: Optional[StabilityAnalyzer] = None,
        scorer: Optional[DiscoveryScorer] = None,
        spread_spec_builder: Optional[SpreadSpecBuilder] = None,
    ):
        self.validator = validator or PairValidator()
        self.stability_analyzer = stability_analyzer or StabilityAnalyzer()
        self.scorer = scorer or DiscoveryScorer()
        self.spread_spec_builder = spread_spec_builder or SpreadSpecBuilder()

    def run(
        self,
        config: ResearchRunConfig,
        prices: pd.DataFrame,
        *,
        universe_snapshot=None,  # Optional[UniverseSnapshot] — if pre-built
        candidate_batch=None,    # Optional[CandidateBatch] — if pre-generated
    ) -> ResearchRunArtifact:
        """
        Run the full discovery pipeline.

        Parameters
        ----------
        config : ResearchRunConfig
        prices : pd.DataFrame with all price columns
        universe_snapshot : Pre-built snapshot (skips universe construction)
        candidate_batch : Pre-built candidates (skips candidate generation)

        Returns
        -------
        ResearchRunArtifact with full audit trail
        """
        artifact = ResearchRunArtifact(config=config)
        artifact.started_at = datetime.utcnow()
        start_time = time.monotonic()

        train_end = config.train_end
        logger.info(
            "DiscoveryPipeline: '%s' | train_end=%s | families=%s",
            config.name, train_end,
            [f.value for f in config.discovery_families],
        )

        # ── Step 1: Universe snapshot ─────────────────────────────────
        if universe_snapshot is None and config.universe_symbols:
            from research.universe import BuiltinUniverses
            definition = BuiltinUniverses.custom(
                name=config.universe_name,
                symbols=config.universe_symbols,
            )
            builder = UniverseBuilder()
            universe_snapshot = builder.build(definition, prices, train_end=train_end)
            logger.info(
                "Universe built: %d eligible, %d excluded",
                universe_snapshot.n_eligible, universe_snapshot.n_excluded,
            )

        artifact.universe_snapshot = universe_snapshot
        artifact.stage_counts["n_eligible"] = (
            universe_snapshot.n_eligible if universe_snapshot else 0
        )

        # ── Step 2: Candidate generation ─────────────────────────────
        if candidate_batch is None:
            if universe_snapshot is None:
                logger.warning("No universe snapshot and no candidate batch — cannot generate candidates")
                artifact.errors.append("No universe snapshot provided")
                artifact.run_duration_seconds = time.monotonic() - start_time
                artifact.completed_at = datetime.utcnow()
                return artifact

            gen_config = CandidateGeneratorConfig(
                families=config.discovery_families,
                min_correlation=config.min_correlation,
                quick_coint_alpha=config.quick_coint_alpha,
                run_quick_coint=True,
                max_candidates_total=config.max_candidates_total,
            )
            generator = CandidateGenerator(gen_config)
            candidate_batch = generator.generate(
                universe_snapshot, prices, train_end=train_end
            )
            logger.info(
                "Candidate generation: %d candidates from %d pairs screened",
                candidate_batch.n_candidates, candidate_batch.n_pairs_screened,
            )

        artifact.candidate_batch = candidate_batch
        artifact.stage_counts["n_candidates"] = candidate_batch.n_candidates

        if candidate_batch.n_candidates == 0:
            logger.warning("No candidates generated — stopping pipeline")
            artifact.run_duration_seconds = time.monotonic() - start_time
            artifact.completed_at = datetime.utcnow()
            return artifact

        # ── Step 3: Deep validation ───────────────────────────────────
        pair_ids = [c.pair_id for c in candidate_batch.candidates]
        candidate_map = {c.pair_id.label: c for c in candidate_batch.candidates}

        logger.info("Deep validation: %d pairs", len(pair_ids))
        validation_reports = self.validator.validate_batch(
            pair_ids, prices, train_end=train_end
        )

        passed_reports = []
        rejected_pairs: dict[str, list[str]] = {}

        for report in validation_reports:
            label = report.pair_id.label
            if report.result == ValidationResult.FAIL:
                rejected_pairs[label] = report.rejection_reasons
                artifact.rejected_pairs[label] = report.rejection_reasons
            else:
                passed_reports.append(report)

        logger.info(
            "Validation: %d passed, %d failed",
            len(passed_reports), len(rejected_pairs),
        )
        artifact.validation_reports = validation_reports
        artifact.stage_counts["n_validated"] = len(validation_reports)
        artifact.stage_counts["n_passed_validation"] = len(passed_reports)

        if not passed_reports:
            artifact.run_duration_seconds = time.monotonic() - start_time
            artifact.completed_at = datetime.utcnow()
            return artifact

        # ── Step 4: Stability analysis ────────────────────────────────
        passed_pair_ids = [r.pair_id for r in passed_reports]
        logger.info("Stability analysis: %d pairs", len(passed_pair_ids))
        stability_map = self.stability_analyzer.analyze_batch(
            passed_pair_ids, prices, train_end=train_end, run_rolling_coint=False
        )
        artifact.stability_reports = list(stability_map.values())

        # ── Step 5 & 6: Scoring + SpreadSpec ─────────────────────────
        validation_map = {r.pair_id.label: r for r in passed_reports}
        scores = []
        spread_specs = []

        for report in passed_reports:
            label = report.pair_id.label
            stab = stability_map.get(label, StabilityReport(pair_id=report.pair_id))
            candidate = candidate_map.get(label)

            # Score
            ds = self.scorer.score(report.pair_id, report, stab, candidate)
            scores.append(ds)

            # Spread spec
            try:
                spec = self.spread_spec_builder.build(
                    report.pair_id, prices, report, stab,
                    train_end=train_end,
                )
                if spec:
                    spread_specs.append(spec)
            except Exception as exc:
                logger.warning("SpreadSpec failed for %s: %s", label, exc)

        artifact.discovery_scores = scores
        artifact.spread_specs = spread_specs

        # ── Step 7: Ranking ───────────────────────────────────────────
        score_map = {ds.pair_id.label: ds for ds in scores}
        spec_map = {ss.pair_id.label: ss for ss in spread_specs}

        ranking = []
        for rank_idx, ds in enumerate(
            sorted(scores, key=lambda s: s.composite_score if not np.isnan(s.composite_score) else -1,
                   reverse=True),
            start=1,
        ):
            label = ds.pair_id.label
            if rank_idx > config.max_final_pairs:
                break

            report = validation_map.get(label)
            stab = stability_map.get(label, StabilityReport(pair_id=ds.pair_id))

            decision = _determine_validation_decision(report, stab, ds)

            if decision == ValidationDecision.REJECTED:
                artifact.rejected_pairs[label] = [RejectionReason.OTHER.value]
                continue

            # Filter by min_validation_decision
            decision_order = [
                ValidationDecision.REJECTED,
                ValidationDecision.RESEARCH_ONLY,
                ValidationDecision.WATCHLIST,
                ValidationDecision.PORTFOLIO_READY,
            ]
            min_order = decision_order.index(config.min_validation_decision)
            curr_order = decision_order.index(decision)
            if curr_order < min_order:
                continue

            rr = RankingResult(
                rank=rank_idx,
                pair_id=ds.pair_id,
                discovery_score=ds,
                validation_report=report,
                validation_decision=decision,
                spread_spec=spec_map.get(label),
                stability_report=stab,
                promotion_notes=decision.value,
            )
            ranking.append(rr)

        artifact.ranking = ranking
        artifact.stage_counts["n_final"] = len(ranking)

        artifact.run_duration_seconds = time.monotonic() - start_time
        artifact.completed_at = datetime.utcnow()

        logger.info(
            "DiscoveryPipeline complete: %d final pairs in %.1fs | top grade: %s",
            len(ranking),
            artifact.run_duration_seconds,
            ranking[0].discovery_score.grade if ranking else "N/A",
        )

        return artifact

    def run_from_pair_list(
        self,
        pair_ids: list[PairId],
        prices: pd.DataFrame,
        config: Optional[ResearchRunConfig] = None,
        *,
        train_end: Optional[datetime] = None,
    ) -> ResearchRunArtifact:
        """
        Convenience method: run the pipeline for an explicit list of pairs,
        skipping universe construction and candidate generation.

        Useful for: re-validating a known watchlist, running research on a
        hand-curated pair set.
        """
        if config is None:
            config = ResearchRunConfig(
                name="manual_pair_list",
                train_end=train_end,
                min_validation_decision=ValidationDecision.RESEARCH_ONLY,
            )
        elif train_end is not None:
            config.train_end = train_end

        artifact = ResearchRunArtifact(config=config)
        artifact.started_at = datetime.utcnow()
        start_time = time.monotonic()

        # Build a minimal candidate batch
        from research.discovery_contracts import CandidateBatch, CandidatePair
        candidates = [
            CandidatePair(
                pair_id=pid,
                discovery_family=DiscoveryFamily.MANUAL,
                discovery_score=0.5,
                stage_reached=4,
            )
            for pid in pair_ids
        ]
        batch = CandidateBatch(
            universe_name="manual",
            n_instruments=len({sym for pid in pair_ids for sym in [pid.sym_x, pid.sym_y]}),
            n_pairs_screened=len(pair_ids),
            n_nominated=len(pair_ids),
            n_stage2=len(pair_ids),
            n_stage3=len(pair_ids),
            candidates=candidates,
        )

        return self.run(config, prices, candidate_batch=batch)


# ── Analytics helpers ──────────────────────────────────────────────

def print_discovery_report(artifact: ResearchRunArtifact) -> None:
    """Print a concise discovery report to stdout."""
    s = artifact.to_summary_dict()
    print("\n" + "=" * 60)
    print(f"Discovery Report: {s.get('run_name', '')}")
    print("=" * 60)
    print(f"  Universe:   {s.get('n_eligible_instruments', 0)} eligible instruments")
    print(f"  Candidates: {s.get('n_candidates', 0)}")
    print(f"  Validated:  {s.get('n_validated', 0)} ({s.get('n_passed', 0)} passed)")
    print(f"  Final:      {s.get('n_final', 0)} pairs")
    print(f"  Duration:   {s.get('run_duration_seconds', 0):.1f}s")

    if artifact.ranking:
        print("\nTop 10 pairs:")
        for rr in artifact.top_pairs(10):
            ds = rr.discovery_score
            print(
                f"  #{rr.rank:2d} {rr.pair_id.label:20s} "
                f"score={ds.composite_score:.2f} grade={ds.grade} "
                f"hl={ds.half_life_days:.0f}d "
                f"decision={rr.validation_decision.value}"
            )

    top_rejections = dict(list(artifact.rejection_summary().items())[:5])
    if top_rejections:
        print(f"\nTop rejection reasons: {top_rejections}")

    print("=" * 60 + "\n")
