# -*- coding: utf-8 -*-
"""
research/discovery_contracts.py — Discovery Layer Domain Objects
================================================================

All typed domain objects for the discovery, universe construction,
candidate generation, validation, and research workflow layers.

Design rules:
- Every rejection is typed via RejectionReason enum (no free-form strings for
  machine-consumable fields, though human_reason is kept alongside)
- All outputs are dataclasses — inspectable, serialisable, diffable
- No discovery metric implies tradability — CandidatePair has a discovery_score
  that is deliberately separate from ValidationDecision
- Universe snapshots are versioned and reproducible
- ResearchRunArtifact captures the full audit trail of one research run

Imports from core.contracts for shared types (PairId, SpreadModel, etc.).
"""

from __future__ import annotations

import enum
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from core.contracts import (
    HedgeRatioMethod,
    PairId,
    PairValidationReport,
    SpreadModel,
    ValidationResult,
    ValidationThresholds,
)


# ── Rejection reasons ──────────────────────────────────────────────

class RejectionReason(str, enum.Enum):
    """
    Typed rejection reasons for candidate, validation, and eligibility decisions.

    Using an enum ensures rejection reasons are machine-comparable across
    experiments and reports, not just human-readable strings.
    """
    # Universe / eligibility
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    LOW_DOLLAR_VOLUME = "LOW_DOLLAR_VOLUME"
    LOW_PRICE = "LOW_PRICE"
    EXCESSIVE_MISSING_DATA = "EXCESSIVE_MISSING_DATA"
    STALE_DATA = "STALE_DATA"
    PRICE_ANOMALY = "PRICE_ANOMALY"
    UNIVERSE_EXCLUDED = "UNIVERSE_EXCLUDED"

    # Candidate generation
    LOW_CORRELATION = "LOW_CORRELATION"
    HIGH_CORRELATION = "HIGH_CORRELATION"           # Too similar / clone-like
    REDUNDANT_PAIR = "REDUNDANT_PAIR"              # Dominated by a better pair
    CALENDAR_MISMATCH = "CALENDAR_MISMATCH"
    VOLATILITY_INCOMPATIBILITY = "VOLATILITY_INCOMPATIBILITY"
    INSUFFICIENT_OVERLAP = "INSUFFICIENT_OVERLAP"
    SAME_INSTRUMENT = "SAME_INSTRUMENT"

    # Statistical validation
    FAILED_COINTEGRATION = "FAILED_COINTEGRATION"
    FAILED_ADF_STATIONARITY = "FAILED_ADF_STATIONARITY"
    EXCESSIVE_HALF_LIFE = "EXCESSIVE_HALF_LIFE"
    TOO_SHORT_HALF_LIFE = "TOO_SHORT_HALF_LIFE"
    UNSTABLE_HEDGE_RATIO = "UNSTABLE_HEDGE_RATIO"
    POOR_HURST_EXPONENT = "POOR_HURST_EXPONENT"
    DEGENERATE_SPREAD = "DEGENERATE_SPREAD"

    # Economic / structural
    ECONOMIC_IMPLAUSIBILITY = "ECONOMIC_IMPLAUSIBILITY"
    STRUCTURAL_BREAK = "STRUCTURAL_BREAK"
    REGIME_UNSUITABLE = "REGIME_UNSUITABLE"

    # Research / quality
    UNSTABLE_CORRELATION = "UNSTABLE_CORRELATION"
    POOR_OUT_OF_SAMPLE = "POOR_OUT_OF_SAMPLE"
    INSUFFICIENT_REVERSION_QUALITY = "INSUFFICIENT_REVERSION_QUALITY"

    # Other
    OTHER = "OTHER"


# ── Discovery families ─────────────────────────────────────────────

class DiscoveryFamily(str, enum.Enum):
    """Which family of methods produced this candidate."""
    CORRELATION = "CORRELATION"           # Rolling/rank correlation
    DISTANCE = "DISTANCE"                 # Normalised price distance / SSD
    COINTEGRATION = "COINTEGRATION"       # EG/Johansen-guided search
    CLUSTER = "CLUSTER"                   # Hierarchical / k-means cluster
    FACTOR_RESIDUAL = "FACTOR_RESIDUAL"   # Factor-neutral residual similarity
    BASKET = "BASKET"                     # Multi-leg basket / relative value
    MANUAL = "MANUAL"                     # User-defined / pre-screened


# ── Validation decision (richer than PASS/FAIL) ───────────────────

class ValidationDecision(str, enum.Enum):
    """
    Three-way tradability verdict beyond the binary PASS/FAIL.

    REJECTED: hard statistical or quality failure
    RESEARCH_ONLY: passes statistical tests but has caveats that prevent live trading
    WATCHLIST: strong statistical case + acceptable liquidity/stability
    PORTFOLIO_READY: passes all checks including regime, capacity, cost
    """
    REJECTED = "REJECTED"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    WATCHLIST = "WATCHLIST"
    PORTFOLIO_READY = "PORTFOLIO_READY"


# ── Instrument metadata ────────────────────────────────────────────

@dataclass
class InstrumentMetadata:
    """
    Enriched metadata for one instrument in the research universe.

    Used for: eligibility filtering, search-space pruning (sector/geography),
    liquidity checks, universe composition analytics.
    """
    symbol: str

    # Identity
    name: str = ""
    asset_class: str = "equity"         # equity, etf, fx, commodity, future, crypto
    exchange: str = ""
    currency: str = "USD"
    isin: str = ""

    # Classification
    sector: str = ""                    # GICS sectors or "ETF", "Macro"
    industry: str = ""
    sub_industry: str = ""
    country: str = "US"
    region: str = "us"                  # us, eu, apac, em, global
    is_etf: bool = False
    etf_category: str = ""             # broad_market, sector, factor, commodity, ...
    factor_exposures: dict[str, float] = field(default_factory=dict)  # {"value": 0.3, "momentum": -0.1}

    # Liquidity / size
    avg_volume: float = np.nan          # shares
    avg_dollar_volume: float = np.nan   # USD notional
    avg_price: float = np.nan
    market_cap: float = np.nan
    volatility_ann: float = np.nan      # annualised daily return vol
    bid_ask_spread_bps: float = np.nan  # basis points

    # Data quality
    history_start: Optional[datetime] = None
    history_end: Optional[datetime] = None
    history_days: int = 0
    missing_data_pct: float = 0.0       # fraction of expected trading days missing
    has_stale_periods: bool = False
    has_price_anomalies: bool = False

    # Eligibility result (set by EligibilityFilter)
    is_eligible: bool = True
    ineligibility_reasons: list[RejectionReason] = field(default_factory=list)
    ineligibility_notes: str = ""

    # Tags
    tags: list[str] = field(default_factory=list)
    custom_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
            elif isinstance(v, list) and v and isinstance(v[0], RejectionReason):
                d[k] = [r.value for r in v]
            else:
                d[k] = v
        return d

    @property
    def liquidity_tier(self) -> str:
        """Bucket instrument by dollar volume."""
        adv = self.avg_dollar_volume
        if np.isnan(adv):
            return "unknown"
        if adv >= 1e9:
            return "large"
        if adv >= 1e8:
            return "medium"
        if adv >= 1e7:
            return "small"
        return "micro"

    @property
    def sector_label(self) -> str:
        """Normalised sector or ETF category."""
        if self.is_etf and self.etf_category:
            return f"etf:{self.etf_category}"
        return self.sector or "unknown"


# ── Eligibility decision ───────────────────────────────────────────

@dataclass
class EligibilityDecision:
    """Result of running an EligibilityFilter over one instrument."""
    symbol: str
    eligible: bool
    rejection_reasons: list[RejectionReason] = field(default_factory=list)
    human_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)  # history_days, dollar_vol, ...


# ── Universe definition ────────────────────────────────────────────

@dataclass
class UniverseDefinition:
    """
    Defines a research universe: which instruments are in scope and
    what eligibility criteria they must satisfy.

    Deliberately separate from the snapshot (what was actually eligible
    at a given point in time) so the definition can be re-applied as
    data changes.
    """
    name: str
    description: str = ""
    symbols: list[str] = field(default_factory=list)  # seed list (may be filtered)

    # Eligibility thresholds — these also serve as survivorship bias mitigations (R-004).
    # min_history_days: exclude recent IPOs/spinoffs; must be >= 252 (1 trading year).
    #   Default 504 (~2 trading years) provides enough data for cointegration tests.
    min_history_days: int = 504         # ~2 years of daily data (R-004: floor 252)
    # min_dollar_volume: exclude illiquid / near-delisting names.
    min_dollar_volume: float = 1_000_000  # $1M/day ADV minimum
    # min_price: exclude penny stocks / distressed names (survivorship bias mitigation).
    min_price: float = 1.0             # exclude sub-$1 instruments
    max_missing_data_pct: float = 0.05  # ≤5% missing days

    # Scoping filters (empty = no restriction)
    allowed_sectors: list[str] = field(default_factory=list)
    excluded_sectors: list[str] = field(default_factory=list)
    allowed_asset_classes: list[str] = field(default_factory=list)
    allowed_countries: list[str] = field(default_factory=list)
    required_tags: list[str] = field(default_factory=list)

    # Search-space constraints
    max_search_pairs: int = 500_000     # cap O(N^2) growth
    sector_pairs_only: bool = False     # only pair within same sector
    cross_sector_allowed: bool = True   # allow cross-sector pairs

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    author: str = ""
    version: str = "1.0"
    tags: list[str] = field(default_factory=list)

    @property
    def universe_id(self) -> str:
        content = json.dumps({
            "name": self.name,
            "symbols": sorted(self.symbols),
            "min_history_days": self.min_history_days,
            "min_dollar_volume": self.min_dollar_volume,
        })
        return hashlib.sha256(content.encode()).hexdigest()[:12]


# ── Universe snapshot ──────────────────────────────────────────────

@dataclass
class UniverseSnapshot:
    """
    Point-in-time snapshot of a research universe.

    Captures exactly which instruments were eligible at snapshot_date,
    which were excluded (and why), and a hash of the definition used.
    Makes research fully reproducible: re-run with the same snapshot
    to get identical candidates.
    """
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    universe_name: str = ""
    snapshot_date: datetime = field(default_factory=datetime.utcnow)

    # Eligible instruments
    eligible_symbols: list[str] = field(default_factory=list)
    metadata: dict[str, InstrumentMetadata] = field(default_factory=dict)

    # Exclusion audit
    excluded_symbols: dict[str, list[str]] = field(default_factory=dict)  # symbol -> [reasons]

    # Universe composition
    sector_counts: dict[str, int] = field(default_factory=dict)
    asset_class_counts: dict[str, int] = field(default_factory=dict)
    region_counts: dict[str, int] = field(default_factory=dict)
    liquidity_tier_counts: dict[str, int] = field(default_factory=dict)

    # Provenance
    definition_id: str = ""
    filter_config_hash: str = ""

    @property
    def n_eligible(self) -> int:
        return len(self.eligible_symbols)

    @property
    def n_excluded(self) -> int:
        return len(self.excluded_symbols)

    def composition_summary(self) -> dict:
        return {
            "n_eligible": self.n_eligible,
            "n_excluded": self.n_excluded,
            "snapshot_date": self.snapshot_date.isoformat(),
            "sectors": self.sector_counts,
            "asset_classes": self.asset_class_counts,
            "regions": self.region_counts,
            "liquidity_tiers": self.liquidity_tier_counts,
        }

    def exclusion_breakdown(self) -> dict[str, int]:
        """Count exclusions by reason string."""
        counts: dict[str, int] = {}
        for reasons in self.excluded_symbols.values():
            for r in reasons:
                counts[r] = counts.get(r, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ── Candidate pair ─────────────────────────────────────────────────

@dataclass
class CandidatePair:
    """
    A pair that has been proposed by at least one discovery family.

    Captures: who proposed it, the raw discovery metrics, and which
    pipeline stage it has survived. Does NOT imply tradability.

    The distinction matters: a candidate may have correlation=0.95 but
    fail cointegration. The discovery_score reflects the nomination
    reason; it is NOT a tradability score.
    """
    pair_id: PairId
    discovery_family: DiscoveryFamily

    # Raw discovery metrics (vary by family)
    discovery_score: float = np.nan     # 0–1, family-specific quality
    correlation: float = np.nan         # Pearson correlation of log-returns
    rank_correlation: float = np.nan    # Spearman, more robust to outliers
    distance_metric: float = np.nan     # Normalised co-movement distance
    quick_coint_pvalue: float = np.nan  # Fast EG p-value for pre-screening
    overlap_days: int = 0               # Days of common price history

    # Economic context
    economic_context: str = ""          # e.g., "same sector: Technology"
    economic_plausibility: float = 0.5  # 0–1 (1 = highly plausible)
    same_sector: bool = False
    same_industry: bool = False
    sector_x: str = ""
    sector_y: str = ""

    # Liquidity compatibility
    liquidity_score: float = np.nan     # 0–1, how compatible are ADVs

    # Pipeline staging
    stage_reached: int = 1              # 1=nominated, 2=cheap_filtered, 3=quality_filtered, 4=deep_validated
    nomination_reasons: list[str] = field(default_factory=list)
    early_rejections: list[RejectionReason] = field(default_factory=list)

    # Deduplication
    is_dominant: bool = True            # False if dominated by a better version

    def to_dict(self) -> dict:
        return {
            "pair_label": self.pair_id.label,
            "sym_x": self.pair_id.sym_x,
            "sym_y": self.pair_id.sym_y,
            "discovery_family": self.discovery_family.value,
            "discovery_score": self.discovery_score,
            "correlation": self.correlation,
            "rank_correlation": self.rank_correlation,
            "distance_metric": self.distance_metric,
            "quick_coint_pvalue": self.quick_coint_pvalue,
            "overlap_days": self.overlap_days,
            "economic_context": self.economic_context,
            "economic_plausibility": self.economic_plausibility,
            "same_sector": self.same_sector,
            "sector_x": self.sector_x,
            "sector_y": self.sector_y,
            "liquidity_score": self.liquidity_score,
            "stage_reached": self.stage_reached,
            "nomination_reasons": self.nomination_reasons,
            "early_rejections": [r.value for r in self.early_rejections],
        }


# ── Candidate batch ────────────────────────────────────────────────

@dataclass
class CandidateBatch:
    """Output of one full candidate-generation run."""
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    universe_name: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Counts
    n_instruments: int = 0
    n_pairs_screened: int = 0           # Total pairs considered
    n_nominated: int = 0               # Passed stage 1
    n_stage2: int = 0                  # Passed stage 2
    n_stage3: int = 0                  # Passed stage 3 (handed to deep validator)

    # Results
    candidates: list[CandidatePair] = field(default_factory=list)

    # Breakdown by family
    by_family: dict[str, int] = field(default_factory=dict)

    # Rejection audit
    rejection_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    def family_summary(self) -> dict:
        return dict(self.by_family)

    def rejection_summary(self) -> dict:
        return dict(sorted(self.rejection_counts.items(), key=lambda x: -x[1]))

    def to_dataframe(self):
        import pandas as pd
        if not self.candidates:
            return pd.DataFrame()
        return pd.DataFrame([c.to_dict() for c in self.candidates])


# ── Discovery score ────────────────────────────────────────────────

@dataclass
class DiscoveryScore:
    """
    Multi-dimensional scoring of a validated candidate.

    Separates five orthogonal dimensions of quality. The composite_score
    is a weighted combination, but callers can inspect each dimension
    independently (e.g., to prefer statistically stable pairs over just
    high-correlation ones).

    Discovery score ≠ validation verdict. A pair can score 0.85 here
    but still have ValidationDecision=WATCHLIST if liquidity is weak.
    """
    pair_id: PairId
    composite_score: float = np.nan     # 0–1 weighted combination

    # Dimension scores (0–1 each)
    correlation_score: float = np.nan
    cointegration_score: float = np.nan
    stability_score: float = np.nan     # rolling hedge ratio + correlation stability
    liquidity_score: float = np.nan
    economic_score: float = np.nan
    regime_score: float = np.nan        # is current regime suitable?

    # Underlying metrics
    correlation: float = np.nan
    coint_pvalue: float = np.nan
    half_life_days: float = np.nan
    hurst_exponent: float = np.nan
    hedge_ratio_cv: float = np.nan      # coefficient of variation of rolling betas
    test_sharpe: float = np.nan         # from walk-forward test window

    # Grade
    grade: str = ""                     # A+ / A / B / C / D
    score_breakdown: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pair_label": self.pair_id.label,
            "composite_score": self.composite_score,
            "correlation_score": self.correlation_score,
            "cointegration_score": self.cointegration_score,
            "stability_score": self.stability_score,
            "liquidity_score": self.liquidity_score,
            "economic_score": self.economic_score,
            "regime_score": self.regime_score,
            "correlation": self.correlation,
            "coint_pvalue": self.coint_pvalue,
            "half_life_days": self.half_life_days,
            "hurst_exponent": self.hurst_exponent,
            "hedge_ratio_cv": self.hedge_ratio_cv,
            "test_sharpe": self.test_sharpe,
            "grade": self.grade,
            "caveats": self.caveats,
        }

    @classmethod
    def grade_from_score(cls, score: float) -> str:
        if np.isnan(score):
            return "?"
        if score >= 0.80:
            return "A+"
        if score >= 0.65:
            return "A"
        if score >= 0.50:
            return "B"
        if score >= 0.35:
            return "C"
        return "D"


# ── Spread specification ───────────────────────────────────────────

@dataclass
class SpreadSpecification:
    """
    Full tradable spread proposal for a validated pair.

    Richer than SpreadDefinition: includes model selection reasoning,
    caveats, and expected forward performance indicators.
    """
    pair_id: PairId
    recommended_model: SpreadModel = SpreadModel.STATIC_OLS
    hedge_ratio: float = 1.0
    hedge_ratio_method: HedgeRatioMethod = HedgeRatioMethod.OLS
    intercept: float = 0.0

    # Normalisation
    normalization: str = "zscore"       # zscore | rolling_zscore | percentile
    window: int = 60                    # rolling window for normalisation

    # Expected performance
    expected_half_life: float = np.nan
    half_life_lower: float = np.nan     # bootstrap / rolling 5th percentile
    half_life_upper: float = np.nan
    validation_sharpe: float = np.nan
    spread_std: float = np.nan
    spread_autocorr_1d: float = np.nan  # AR(1) — lower = faster mean reversion

    # Model selection
    model_selection_reason: str = ""
    ols_half_life: float = np.nan
    kalman_half_life: float = np.nan
    rolling_half_life: float = np.nan

    # Entry/exit suggestions
    suggested_entry_z: float = 2.0
    suggested_exit_z: float = 0.5
    suggested_stop_z: float = 4.0
    suggested_max_hold_days: int = 60

    # Warnings and caveats
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pair_label": self.pair_id.label,
            "recommended_model": self.recommended_model.value,
            "hedge_ratio": self.hedge_ratio,
            "normalization": self.normalization,
            "window": self.window,
            "expected_half_life": self.expected_half_life,
            "validation_sharpe": self.validation_sharpe,
            "model_selection_reason": self.model_selection_reason,
            "suggested_entry_z": self.suggested_entry_z,
            "suggested_exit_z": self.suggested_exit_z,
            "suggested_stop_z": self.suggested_stop_z,
            "suggested_max_hold_days": self.suggested_max_hold_days,
            "warnings": self.warnings,
        }


# ── Stability report ───────────────────────────────────────────────

@dataclass
class StabilityReport:
    """
    Rolling relationship stability analysis for a pair.

    Captures how the statistical relationship has evolved over time.
    Crucial for detecting pairs that only look good in a specific regime.
    """
    pair_id: PairId
    analysis_date: datetime = field(default_factory=datetime.utcnow)

    # Rolling correlation stability
    corr_mean: float = np.nan
    corr_std: float = np.nan
    corr_min: float = np.nan
    corr_max: float = np.nan
    corr_trend: float = np.nan          # slope of rolling correlation

    # Rolling hedge ratio stability
    beta_mean: float = np.nan
    beta_std: float = np.nan
    beta_cv: float = np.nan             # coeff of variation
    beta_trend: float = np.nan

    # Spread stationarity over sub-periods
    adf_rolling_pass_rate: float = np.nan  # fraction of rolling windows where ADF passes
    coint_rolling_pass_rate: float = np.nan

    # Structural break signals
    has_structural_break: bool = False
    break_date: Optional[datetime] = None
    break_confidence: float = np.nan    # 0–1
    break_detection_method: str = ""

    # Regime suitability
    regime_suitable: bool = True
    regime_label: str = "UNKNOWN"
    regime_caveats: list[str] = field(default_factory=list)

    # Overall verdict
    is_stable: bool = True
    stability_score: float = np.nan     # 0–1 composite stability
    instability_reasons: list[RejectionReason] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "pair_label": self.pair_id.label,
            "corr_mean": self.corr_mean,
            "corr_std": self.corr_std,
            "beta_cv": self.beta_cv,
            "adf_rolling_pass_rate": self.adf_rolling_pass_rate,
            "coint_rolling_pass_rate": self.coint_rolling_pass_rate,
            "has_structural_break": self.has_structural_break,
            "break_date": self.break_date.isoformat() if self.break_date else None,
            "break_confidence": self.break_confidence,
            "regime_suitable": self.regime_suitable,
            "is_stable": self.is_stable,
            "stability_score": self.stability_score,
            "instability_reasons": [r.value for r in self.instability_reasons],
        }


# ── Research run config ────────────────────────────────────────────

@dataclass
class ResearchRunConfig:
    """
    Full specification of one research experiment.

    Stored with every ResearchRunArtifact so any experiment can be
    exactly reproduced from this config + the raw prices.
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # Universe
    universe_name: str = ""
    universe_symbols: list[str] = field(default_factory=list)

    # Discovery settings
    discovery_families: list[DiscoveryFamily] = field(
        default_factory=lambda: [DiscoveryFamily.CORRELATION, DiscoveryFamily.DISTANCE]
    )
    min_correlation: float = 0.50       # Stage-2 correlation pre-filter
    quick_coint_alpha: float = 0.20     # Stage-3 quick cointegration pre-filter

    # Validation settings
    validation_thresholds: ValidationThresholds = field(default_factory=ValidationThresholds)
    max_candidates_per_family: int = 500
    max_candidates_total: int = 1000

    # Time range
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None

    # Walk-forward
    n_wf_splits: int = 3
    wf_test_days: int = 252
    wf_embargo_days: int = 20
    wf_min_train_days: int = 504

    # Output settings
    max_final_pairs: int = 50
    min_validation_decision: ValidationDecision = ValidationDecision.RESEARCH_ONLY

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    author: str = ""
    notes: str = ""
    version: str = "1.0"

    def config_hash(self) -> str:
        """Stable hash of this config for de-duplication."""
        content = json.dumps({
            "universe_symbols": sorted(self.universe_symbols),
            "discovery_families": [f.value for f in self.discovery_families],
            "min_correlation": self.min_correlation,
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
        })
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Ranking result ─────────────────────────────────────────────────

@dataclass
class RankingResult:
    """
    One entry in the final ranked shortlist of viable pairs.
    Combines discovery score, validation verdict, and spread specification.
    """
    rank: int
    pair_id: PairId
    discovery_score: DiscoveryScore
    validation_report: PairValidationReport
    validation_decision: ValidationDecision
    spread_spec: Optional[SpreadSpecification] = None
    stability_report: Optional[StabilityReport] = None
    promotion_notes: str = ""

    def to_dict(self) -> dict:
        d = {
            "rank": self.rank,
            "pair_label": self.pair_id.label,
            "validation_decision": self.validation_decision.value,
            "validation_result": self.validation_report.result.value,
            "promotion_notes": self.promotion_notes,
        }
        d.update(self.discovery_score.to_dict())
        if self.spread_spec:
            d.update({f"spread_{k}": v for k, v in self.spread_spec.to_dict().items()
                      if k != "pair_label"})
        return d


# ── Research run artifact ──────────────────────────────────────────

@dataclass
class ResearchRunArtifact:
    """
    Full output of one research experiment.

    Stores every layer of output so that:
    - the experiment can be reproduced
    - the rejection audit is inspectable
    - the final ranking is ready for downstream use
    """
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    config: Optional[ResearchRunConfig] = None

    # Inputs
    universe_snapshot: Optional[UniverseSnapshot] = None
    candidate_batch: Optional[CandidateBatch] = None

    # Outputs
    validation_reports: list[PairValidationReport] = field(default_factory=list)
    discovery_scores: list[DiscoveryScore] = field(default_factory=list)
    stability_reports: list[StabilityReport] = field(default_factory=list)
    spread_specs: list[SpreadSpecification] = field(default_factory=list)
    ranking: list[RankingResult] = field(default_factory=list)

    # Rejection audit
    rejected_pairs: dict[str, list[str]] = field(default_factory=dict)  # label -> reasons

    # Run metadata
    run_duration_seconds: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: list[str] = field(default_factory=list)

    # Pipeline stage counts
    stage_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_candidates(self) -> int:
        return self.candidate_batch.n_candidates if self.candidate_batch else 0

    @property
    def n_validated(self) -> int:
        return len(self.validation_reports)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.validation_reports if r.result != ValidationResult.FAIL)

    @property
    def n_final(self) -> int:
        return len(self.ranking)

    def rejection_summary(self) -> dict[str, int]:
        """Count rejection reasons across all rejected pairs."""
        counts: dict[str, int] = {}
        for reasons in self.rejected_pairs.values():
            for r in reasons:
                counts[r] = counts.get(r, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def to_summary_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "run_id": self.config.run_id if self.config else None,
            "run_name": self.config.name if self.config else "",
            "universe_name": self.config.universe_name if self.config else "",
            "n_eligible_instruments": self.universe_snapshot.n_eligible if self.universe_snapshot else 0,
            "n_candidates": self.n_candidates,
            "n_validated": self.n_validated,
            "n_passed": self.n_passed,
            "n_final": self.n_final,
            "stage_counts": self.stage_counts,
            "top_rejection_reasons": dict(list(self.rejection_summary().items())[:5]),
            "run_duration_seconds": self.run_duration_seconds,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def top_pairs(self, n: int = 20) -> list[RankingResult]:
        return sorted(self.ranking, key=lambda r: r.rank)[:n]


# ── Discovery experiment summary ───────────────────────────────────

@dataclass
class DiscoveryExperimentSummary:
    """
    High-level comparison across multiple research runs.

    Useful for comparing: which discovery family works best? which
    universe produces the most viable pairs? how does the yield
    (n_final / n_candidates) vary across configurations?
    """
    summaries: list[dict] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def add(self, artifact: ResearchRunArtifact) -> None:
        self.summaries.append(artifact.to_summary_dict())

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.summaries) if self.summaries else pd.DataFrame()
