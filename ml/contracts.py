# -*- coding: utf-8 -*-
"""
ml/contracts.py — ML Platform Domain Objects
============================================

Single source of truth for all ML-platform typed objects.
Every feature, label, dataset, model, prediction, and governance
artifact is represented here with explicit types.

Design principles:
- All objects are frozen dataclasses (immutable after creation)
- All objects are JSON-serializable via asdict()
- No loose dicts for critical ML artifacts
- Timestamps are ISO-8601 strings (UTC) for serialization safety
- Every object has a unique identifier
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EntityScope(str, Enum):
    """Entity scope for feature / label definitions."""
    INSTRUMENT = "instrument"
    PAIR = "pair"
    REGIME = "regime"
    SIGNAL = "signal"
    PORTFOLIO = "portfolio"
    EXECUTION = "execution"


class FeatureCategory(str, Enum):
    """High-level category for feature grouping and filtering."""
    Z_SCORE = "z_score"
    RETURN = "return"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    SPREAD = "spread"
    LIQUIDITY = "liquidity"
    MOMENTUM = "momentum"
    REGIME = "regime"
    SIGNAL_CONTEXT = "signal_context"
    PORTFOLIO_CONTEXT = "portfolio_context"
    EXECUTION_CONTEXT = "execution_context"
    STABILITY = "stability"
    MACRO = "macro"


class LabelFamily(str, Enum):
    """Label family / use-case category."""
    REVERSION_SUCCESS = "reversion_success"
    META_LABEL = "meta_label"
    RELATIONSHIP_PERSISTENCE = "relationship_persistence"
    BREAK_INSTABILITY = "break_instability"
    HOLDING_TIME = "holding_time"
    THRESHOLD_BAND = "threshold_band"
    PORTFOLIO_UTILITY = "portfolio_utility"
    EXECUTION_VIABILITY = "execution_viability"
    REGIME = "regime"
    ANOMALY = "anomaly"


class MLTaskFamily(str, Enum):
    """Supported ML task families."""
    CANDIDATE_RANKING = "candidate_ranking"
    META_LABELING = "meta_labeling"
    REGIME_CLASSIFICATION = "regime_classification"
    BREAK_DETECTION = "break_detection"
    HOLDING_TIME = "holding_time"
    THRESHOLD_RECOMMENDATION = "threshold_recommendation"
    SIZING_ASSISTANCE = "sizing_assistance"
    PORTFOLIO_RANKING = "portfolio_ranking"
    EXECUTION_VIABILITY = "execution_viability"
    ANOMALY_DETECTION = "anomaly_detection"


class MetaLabelAction(str, Enum):
    """Structured meta-label actions — what the model recommends doing with a signal."""
    TAKE = "TAKE"
    SKIP = "SKIP"
    TAKE_SMALLER = "TAKE_SMALLER"
    WAIT_FOR_CONFIRMATION = "WAIT_FOR_CONFIRMATION"
    EXIT_EARLIER_BIAS = "EXIT_EARLIER_BIAS"
    REQUIRE_HIGHER_THRESHOLD = "REQUIRE_HIGHER_THRESHOLD"
    REQUIRE_BETTER_REGIME = "REQUIRE_BETTER_REGIME"


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    RESEARCH = "research"
    CANDIDATE = "candidate"
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    RETIRED = "retired"
    BLOCKED = "blocked"


class SplitPolicy(str, Enum):
    """Temporal split strategy."""
    CHRONOLOGICAL = "chronological"
    ROLLING_ORIGIN = "rolling_origin"
    WALK_FORWARD = "walk_forward"
    PURGED_KFOLD = "purged_kfold"
    REGIME_SLICED = "regime_sliced"


class DriftSeverity(str, Enum):
    """Severity classification for drift reports."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class ModelHealthState(str, Enum):
    """Top-level model health state."""
    HEALTHY = "healthy"
    STALE = "stale"
    DRIFTING = "drifting"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"
    FALLBACK_ACTIVE = "fallback_active"


class PromotionOutcome(str, Enum):
    """Champion/challenger promotion decision."""
    PROMOTE = "promote"
    REJECT = "reject"
    DEFER = "defer"
    REQUIRE_MORE_DATA = "require_more_data"


class GovernanceStatus(str, Enum):
    """Governance approval status."""
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    CONDITIONALLY_APPROVED = "conditionally_approved"
    REJECTED = "rejected"
    SUSPENDED = "suspended"


# ---------------------------------------------------------------------------
# Feature Platform Objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MissingValuePolicy:
    """How to handle missing feature values."""
    strategy: str = "ffill"          # "ffill" | "bfill" | "zero" | "mean" | "drop"
    max_fill_periods: int = 5        # Maximum consecutive periods to fill
    warn_threshold: float = 0.10     # Warn if >10% of values are missing


@dataclass(frozen=True)
class FeatureDefinition:
    """
    Canonical definition for a single feature.

    Every feature in the ML platform must be registered here.
    No ad-hoc computation without a definition.
    """
    name: str
    description: str
    entity_scope: EntityScope
    category: FeatureCategory
    required_inputs: Tuple[str, ...]          # Raw data series required
    lookback_days: int                         # Minimum history needed
    version: str = "1.0"
    normalise: bool = True                     # Whether downstream pipeline should z-score
    missing_value_policy: MissingValuePolicy = field(default_factory=MissingValuePolicy)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_scope": self.entity_scope.value,
            "category": self.category.value,
            "required_inputs": list(self.required_inputs),
            "lookback_days": self.lookback_days,
            "version": self.version,
            "normalise": self.normalise,
        }


@dataclass(frozen=True)
class FeatureGroup:
    """A named collection of related features."""
    name: str
    description: str
    feature_names: Tuple[str, ...]
    entity_scope: EntityScope
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_scope": self.entity_scope.value,
            "feature_names": list(self.feature_names),
            "version": self.version,
        }


@dataclass
class FeatureSetVersion:
    """A versioned, reproducible set of features used in a specific training run."""
    feature_set_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    feature_names: List[str] = field(default_factory=list)
    group_names: List[str] = field(default_factory=list)
    entity_scope: EntityScope = EntityScope.PAIR
    version: str = "1.0"
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    notes: str = ""

    def feature_hash(self) -> str:
        """Deterministic hash of the feature set for lineage tracking."""
        import hashlib
        key = "|".join(sorted(self.feature_names))
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class FeatureSnapshot:
    """A point-in-time materialized feature matrix."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    feature_set_id: str = ""
    entity_id: str = ""                        # pair_id or instrument id
    as_of: str = ""                            # ISO timestamp — features valid as of this time
    feature_values: Dict[str, float] = field(default_factory=dict)
    null_count: int = 0
    computed_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    warnings: List[str] = field(default_factory=list)


@dataclass
class FeatureLineageRecord:
    """Tracks the lineage of a feature — inputs, computation, version."""
    feature_name: str
    feature_version: str
    entity_id: str
    as_of: str
    inputs_used: List[str] = field(default_factory=list)       # Raw data series used
    lookback_start: str = ""
    lookback_end: str = ""
    transform_steps: List[str] = field(default_factory=list)   # Description of transforms
    output_shape: str = ""
    null_fraction: float = 0.0
    recorded_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Label Platform Objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelDefinition:
    """
    Formal definition of a label / learning target.

    Enforces explicit horizon, event window, censoring, and leakage documentation.
    """
    name: str
    family: LabelFamily
    description: str
    horizon_days: int                          # Forward horizon used to compute label
    entity_scope: EntityScope
    output_type: str                           # "binary" | "multiclass" | "continuous" | "ordinal"
    event_window_days: int = 0                 # If event-based, window around event
    censoring_logic: str = "none"              # How incomplete observations are handled
    path_dependent: bool = False               # Does label depend on intra-horizon path?
    cost_treatment: str = "none"              # "none" | "fixed_bps" | "model"
    class_balance_note: str = ""
    leakage_risks: str = ""
    valid_use_cases: Tuple[str, ...] = ()
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family.value,
            "horizon_days": self.horizon_days,
            "entity_scope": self.entity_scope.value,
            "output_type": self.output_type,
            "version": self.version,
        }


@dataclass
class LabelArtifact:
    """A computed label series associated with a dataset snapshot."""
    label_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    label_name: str = ""
    label_version: str = "1.0"
    entity_id: str = ""
    horizon_days: int = 0
    n_samples: int = 0
    class_balance: Dict[str, float] = field(default_factory=dict)
    null_fraction: float = 0.0
    label_start: str = ""
    label_end: str = ""
    computed_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    notes: str = ""


# ---------------------------------------------------------------------------
# Dataset Objects
# ---------------------------------------------------------------------------

@dataclass
class DatasetDefinition:
    """
    Specification for building a dataset — what goes in, how it's split.
    """
    dataset_def_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    task_family: MLTaskFamily = MLTaskFamily.META_LABELING
    entity_scope: EntityScope = EntityScope.PAIR
    feature_set_id: str = ""
    label_name: str = ""
    universe_id: str = ""
    time_start: str = ""
    time_end: str = ""
    split_policy: SplitPolicy = SplitPolicy.CHRONOLOGICAL
    purge_days: int = 5
    embargo_days: int = 10
    min_samples_per_entity: int = 50
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


@dataclass
class DatasetSplitPlan:
    """Concrete train/validation/test split boundaries."""
    split_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    dataset_def_id: str = ""
    split_policy: SplitPolicy = SplitPolicy.CHRONOLOGICAL
    train_start: str = ""
    train_end: str = ""
    validation_start: str = ""
    validation_end: str = ""
    test_start: str = ""
    test_end: str = ""
    purge_days: int = 5
    embargo_days: int = 10
    n_folds: int = 1                           # For walk-forward / k-fold
    fold_index: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


@dataclass
class DatasetSnapshot:
    """
    An immutable, reproducible dataset artifact — the record of exactly
    what was used to train a model.
    """
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    dataset_def_id: str = ""
    feature_set_id: str = ""
    feature_hash: str = ""
    label_name: str = ""
    label_version: str = "1.0"
    universe_snapshot_id: str = ""
    time_start: str = ""
    time_end: str = ""
    train_rows: int = 0
    validation_rows: int = 0
    test_rows: int = 0
    n_features: int = 0
    n_entities: int = 0
    class_balance_train: Dict[str, float] = field(default_factory=dict)
    null_profile: Dict[str, float] = field(default_factory=dict)
    leakage_audit_passed: bool = False
    leakage_audit_warnings: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    notes: str = ""


@dataclass
class LeakageAuditReport:
    """
    Audit result for a dataset — documents any detected or potential leakage.
    """
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    dataset_snapshot_id: str = ""
    passed: bool = True
    checks_run: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    future_feature_risk: bool = False
    future_label_risk: bool = False
    normalization_leak_risk: bool = False
    overlap_label_risk: bool = False
    embargo_adequate: bool = True
    purge_adequate: bool = True
    audited_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    notes: str = ""


# ---------------------------------------------------------------------------
# Model Objects
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """High-level specification for a model."""
    task_family: MLTaskFamily
    model_class: str                           # e.g. "LogisticRegression", "XGBClassifier"
    feature_set_id: str
    label_name: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    calibration: bool = False
    calibration_method: str = "isotonic"       # "isotonic" | "platt"
    random_seed: int = 42
    notes: str = ""


@dataclass
class ModelConfig:
    """
    Full model configuration — spec + training policy + evaluation policy.
    """
    config_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    spec: ModelSpec = field(default_factory=lambda: ModelSpec(
        task_family=MLTaskFamily.META_LABELING,
        model_class="LogisticRegression",
        feature_set_id="",
        label_name="",
    ))
    split_plan: Optional[DatasetSplitPlan] = None
    n_cv_folds: int = 5
    purge_days: int = 5
    embargo_days: int = 10
    evaluation_metrics: List[str] = field(default_factory=lambda: ["auc", "brier", "ic"])
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


@dataclass
class TrainingRunConfig:
    """Configuration for a single training run — fully reproducible."""
    run_config_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_config: ModelConfig = field(default_factory=ModelConfig)
    dataset_snapshot_id: str = ""
    random_seed: int = 42
    n_jobs: int = 1
    early_stopping: bool = False
    max_iterations: int = 1000
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


@dataclass
class TrainingRunArtifact:
    """
    Artifacts produced by a completed training run.
    The canonical provenance record for a trained model.
    """
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_config_id: str = ""
    dataset_snapshot_id: str = ""
    model_id: str = ""
    task_family: str = ""
    feature_set_id: str = ""
    feature_hash: str = ""
    label_name: str = ""
    train_start: str = ""
    train_end: str = ""
    n_train_samples: int = 0
    n_validation_samples: int = 0
    n_features: int = 0
    # Evaluation metrics
    train_auc: float = float("nan")
    val_auc: float = float("nan")
    test_auc: float = float("nan")
    train_brier: float = float("nan")
    val_brier: float = float("nan")
    test_brier: float = float("nan")
    cv_ic_mean: float = float("nan")
    cv_ic_std: float = float("nan")
    # Feature importance
    top_features: List[str] = field(default_factory=list)
    feature_importances: Dict[str, float] = field(default_factory=dict)
    # Calibration
    calibration_applied: bool = False
    calibration_brier_improvement: float = float("nan")
    # Artifact paths
    model_artifact_path: str = ""
    evaluation_report_path: str = ""
    # Status
    status: str = "completed"
    error_message: str = ""
    completed_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    notes: str = ""


# ---------------------------------------------------------------------------
# Evaluation Objects
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    """Structured evaluation report for a trained model."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    run_id: str = ""
    task_family: str = ""
    split: str = "test"                        # "train" | "validation" | "test"
    # Classification metrics
    auc: float = float("nan")
    pr_auc: float = float("nan")
    brier_score: float = float("nan")
    log_loss: float = float("nan")
    accuracy: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    # Ranking metrics
    information_coefficient: float = float("nan")
    ic_t_stat: float = float("nan")
    # Regression metrics
    mae: float = float("nan")
    rmse: float = float("nan")
    # Trading metrics
    meta_label_precision: float = float("nan")  # P(reversion | model says TAKE)
    baseline_precision: float = float("nan")    # P(reversion | no filter)
    filter_improvement: float = float("nan")    # meta_label_precision - baseline
    # Regime-sliced
    regime_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Diagnostics
    n_samples: int = 0
    class_balance: Dict[str, float] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    notes: str = ""


@dataclass
class CalibrationReport:
    """Calibration diagnostics for a probabilistic model."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    split: str = "test"
    # Calibration metrics
    brier_score_uncalibrated: float = float("nan")
    brier_score_calibrated: float = float("nan")
    brier_improvement: float = float("nan")
    # Reliability (probability bucketing)
    bucket_edges: List[float] = field(default_factory=list)
    bucket_mean_predicted: List[float] = field(default_factory=list)
    bucket_mean_actual: List[float] = field(default_factory=list)
    bucket_counts: List[int] = field(default_factory=list)
    # Regime-conditional calibration
    regime_calibration: Dict[str, Dict[str, float]] = field(default_factory=dict)
    calibration_method: str = "isotonic"
    n_samples: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Prediction / Inference Objects
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """
    A single structured prediction from an ML model.
    Downstream consumers must not guess what a model output means.
    """
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    model_version: str = ""
    task_family: str = ""
    entity_id: str = ""                        # pair_id or instrument id
    prediction_timestamp: str = ""
    # Output
    score: float = float("nan")                # Raw score / probability
    label: Optional[str] = None               # Predicted class label if applicable
    meta_action: Optional[MetaLabelAction] = None  # For meta-labeling tasks
    # Uncertainty
    confidence: float = float("nan")          # Model confidence / calibration quality
    uncertainty: float = float("nan")         # Prediction uncertainty estimate
    # Context
    feature_snapshot_id: str = ""
    feature_availability: str = "full"        # "full" | "partial" | "degraded"
    missing_features: List[str] = field(default_factory=list)
    # Governance
    calibration_applied: bool = False
    warnings: List[str] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str = ""
    # Usage constraints
    recommended_use: str = ""
    forbidden_use: str = ""


@dataclass
class PredictionBatch:
    """A batch of predictions produced in a single scoring run."""
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    task_family: str = ""
    scoring_timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    n_predictions: int = 0
    predictions: List[PredictionRecord] = field(default_factory=list)
    feature_set_id: str = ""
    feature_snapshot_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fallback_count: int = 0


@dataclass
class InferenceRequest:
    """Structured request for an ML prediction."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    entity_id: str = ""
    task_family: MLTaskFamily = MLTaskFamily.META_LABELING
    as_of: str = ""                            # Point-in-time timestamp
    features: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    allow_fallback: bool = True
    strict_feature_check: bool = False


@dataclass
class InferenceResult:
    """Structured result from the inference layer."""
    request_id: str = ""
    model_id: str = ""
    task_family: str = ""
    entity_id: str = ""
    score: float = float("nan")
    label: Optional[str] = None
    meta_action: Optional[MetaLabelAction] = None
    confidence: float = float("nan")
    feature_availability: str = "full"
    missing_features: List[str] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str = ""
    warnings: List[str] = field(default_factory=list)
    served_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    @property
    def is_valid(self) -> bool:
        return not (self.score != self.score)  # NaN check

    @property
    def should_act(self) -> bool:
        """True if a downstream consumer should act on this prediction."""
        return self.is_valid and not self.fallback_used and self.confidence >= 0.0


# ---------------------------------------------------------------------------
# Registry and Governance Objects
# ---------------------------------------------------------------------------

@dataclass
class ModelMetadata:
    """
    Complete metadata for a model version — provenance, status, governance.
    """
    model_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_family: MLTaskFamily = MLTaskFamily.META_LABELING
    model_class: str = ""
    version: str = "1.0"
    # Dataset + feature provenance
    feature_set_id: str = ""
    feature_hash: str = ""
    label_name: str = ""
    dataset_snapshot_id: str = ""
    # Training provenance
    train_start: str = ""
    train_end: str = ""
    trained_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    n_train_samples: int = 0
    n_features: int = 0
    # Evaluation summary
    val_auc: float = float("nan")
    val_brier: float = float("nan")
    cv_ic_mean: float = float("nan")
    cv_ic_std: float = float("nan")
    # Calibration
    calibrated: bool = False
    calibration_brier: float = float("nan")
    # Lifecycle
    status: ModelStatus = ModelStatus.RESEARCH
    governance_status: GovernanceStatus = GovernanceStatus.PENDING_REVIEW
    promoted_at: str = ""
    retired_at: str = ""
    # Allowed usage
    allowed_consumers: List[str] = field(default_factory=list)
    forbidden_consumers: List[str] = field(default_factory=list)
    # Monitoring
    monitoring_rules: List[str] = field(default_factory=list)
    retirement_criteria: List[str] = field(default_factory=list)
    fallback_policy: str = "use_rule_based"
    # Artifact paths
    artifact_path: str = ""
    run_id: str = ""
    notes: str = ""


@dataclass
class ModelCard:
    """
    Human-readable and machine-readable model card.
    Required for any model promoted to CHAMPION.
    """
    model_id: str = ""
    model_class: str = ""
    task_family: str = ""
    version: str = "1.0"
    # Intent
    intended_use: str = ""
    out_of_scope_use: str = ""
    # Performance summary
    primary_metric: str = ""
    primary_metric_value: float = float("nan")
    baseline_comparison: str = ""
    regime_performance_summary: str = ""
    # Limitations
    known_limitations: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    # Data
    training_data_summary: str = ""
    training_period: str = ""
    feature_groups_used: List[str] = field(default_factory=list)
    label_description: str = ""
    # Governance
    owner: str = ""
    review_date: str = ""
    next_review_date: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


@dataclass
class ChampionChallengerRecord:
    """
    Formal record of a champion/challenger comparison.
    Must be created whenever a challenger is evaluated against a champion.
    """
    comparison_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    champion_model_id: str = ""
    challenger_model_id: str = ""
    task_family: str = ""
    # Evaluation window
    eval_start: str = ""
    eval_end: str = ""
    n_regimes_tested: int = 0
    # Comparison results
    champion_auc: float = float("nan")
    challenger_auc: float = float("nan")
    champion_brier: float = float("nan")
    challenger_brier: float = float("nan")
    champion_ic: float = float("nan")
    challenger_ic: float = float("nan")
    # Regime-sliced comparison
    regime_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Decision
    outcome: PromotionOutcome = PromotionOutcome.DEFER
    decision_rationale: str = ""
    decided_by: str = "system"
    decided_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


@dataclass
class FallbackPolicy:
    """Defines fallback behavior when a model is unavailable or unhealthy."""
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    task_family: str = ""
    # Fallback triggers
    trigger_on_stale: bool = True
    stale_threshold_hours: int = 48
    trigger_on_drift: bool = True
    drift_severity_threshold: DriftSeverity = DriftSeverity.MODERATE
    trigger_on_missing_features: bool = True
    missing_feature_threshold: float = 0.20   # >20% missing → fallback
    # Fallback behavior
    fallback_type: str = "rule_based"          # "rule_based" | "prior_champion" | "neutral"
    neutral_score: float = 0.5
    fallback_meta_action: MetaLabelAction = MetaLabelAction.TAKE  # Default action on fallback
    # Logging
    log_fallback: bool = True
    alert_on_fallback: bool = False


@dataclass
class MLUsageContract:
    """
    Formal contract governing how a model may be used by a consuming layer.
    Enforces the principle that ML emits recommendations, not decisions.
    """
    contract_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    consumer: str = ""                         # "signal_layer" | "portfolio_layer" | etc.
    task_family: str = ""
    # Permissions
    may_override_hard_rules: bool = False      # ML must NEVER override hard risk rules
    may_block_entries: bool = False            # ML may suggest blocking, not enforce
    may_resize_positions: bool = True          # ML may suggest sizing adjustments
    may_adjust_thresholds: bool = True         # ML may suggest threshold changes
    max_grade_upgrade: int = 1                 # How many grade levels ML can upgrade
    # Constraints
    min_model_confidence: float = 0.55
    require_calibration: bool = True
    require_champion_status: bool = False
    # Fallback
    fallback_policy_id: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Monitoring and Drift Objects
# ---------------------------------------------------------------------------

@dataclass
class DriftReport:
    """Drift monitoring report for a model."""
    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    report_type: str = "feature"               # "feature" | "label" | "prediction" | "calibration"
    severity: DriftSeverity = DriftSeverity.NONE
    # Feature drift (PSI scores)
    feature_psi_scores: Dict[str, float] = field(default_factory=dict)
    features_drifted: List[str] = field(default_factory=list)
    mean_psi: float = float("nan")
    # Prediction drift
    score_distribution_shift: float = float("nan")
    class_balance_shift: float = float("nan")
    # Performance drift
    ic_current: float = float("nan")
    ic_baseline: float = float("nan")
    ic_degraded: bool = False
    # Calibration drift
    brier_current: float = float("nan")
    brier_baseline: float = float("nan")
    # Recommended action
    recommended_action: str = "none"           # "none" | "monitor" | "retrain" | "suspend"
    generated_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    notes: str = ""


@dataclass
class ModelHealthStatus:
    """Current health state of a deployed model."""
    status_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    task_family: str = ""
    state: ModelHealthState = ModelHealthState.HEALTHY
    # Freshness
    last_trained_at: str = ""
    hours_since_training: float = float("nan")
    stale: bool = False
    # Drift
    latest_drift_report_id: str = ""
    drift_severity: DriftSeverity = DriftSeverity.NONE
    # Performance
    recent_ic: float = float("nan")
    recent_brier: float = float("nan")
    performance_degraded: bool = False
    # Inference health
    inference_failure_rate: float = float("nan")
    fallback_rate: float = float("nan")
    # Action
    recommended_action: str = "none"
    action_reason: str = ""
    checked_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    @property
    def is_usable(self) -> bool:
        return self.state in (
            ModelHealthState.HEALTHY,
            ModelHealthState.DRIFTING,  # Drifting is usable with caution
        )


# ---------------------------------------------------------------------------
# Explainability Objects
# ---------------------------------------------------------------------------

@dataclass
class ExplainabilityArtifact:
    """
    Explainability artifact for a model — feature importance + lineage.
    """
    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    # Global importance
    global_feature_importances: Dict[str, float] = field(default_factory=dict)
    top_features_ranked: List[str] = field(default_factory=list)
    importance_method: str = "gain"            # "gain" | "permutation" | "shap_values"
    # Feature lineage
    feature_lineage_records: List[str] = field(default_factory=list)  # lineage record IDs
    # Score decomposition (for composite models)
    score_decomposition: Dict[str, float] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Research Objects
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSummary:
    """Summary of an ML experiment (research run)."""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    task_family: str = ""
    # Config
    feature_sets_tested: List[str] = field(default_factory=list)
    models_tested: List[str] = field(default_factory=list)
    label_name: str = ""
    # Results
    best_model_id: str = ""
    best_val_auc: float = float("nan")
    best_ic: float = float("nan")
    baseline_auc: float = float("nan")
    improvement_over_baseline: float = float("nan")
    # Robustness
    walk_forward_ic_mean: float = float("nan")
    walk_forward_ic_std: float = float("nan")
    regime_stability: float = float("nan")
    # Recommendation
    promotion_recommended: bool = False
    recommendation_rationale: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


@dataclass
class HyperparameterStudyArtifact:
    """Record of a hyperparameter search study."""
    study_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_class: str = ""
    n_trials: int = 0
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_val_score: float = float("nan")
    search_space: Dict[str, Any] = field(default_factory=dict)
    framework: str = "optuna"
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )


@dataclass
class PromotionDecision:
    """
    Explicit promotion decision for a model — never implicit.
    Required to move a model from CANDIDATE → CHAMPION.
    """
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_id: str = ""
    from_status: ModelStatus = ModelStatus.CANDIDATE
    to_status: ModelStatus = ModelStatus.CHAMPION
    comparison_id: str = ""                    # ChampionChallengerRecord ID
    evidence_summary: str = ""
    criteria_met: List[str] = field(default_factory=list)
    criteria_failed: List[str] = field(default_factory=list)
    outcome: PromotionOutcome = PromotionOutcome.DEFER
    decided_by: str = "system"
    requires_manual_approval: bool = True
    manually_approved: bool = False
    approved_by: str = ""
    decided_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    notes: str = ""


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def contracts_version() -> str:
    """Return the version of the contracts module."""
    return "1.0.0"
