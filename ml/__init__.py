# -*- coding: utf-8 -*-
"""
ml/ — ML Platform for the Pairs Trading System
===============================================

Sub-packages:
  evaluation    : Metrics (IC, AUC, Brier, calibration) and ModelEvaluator
  registry      : MLModelRegistry with champion/challenger tracking
  inference     : ModelScorer — inference with fallback and feature validation
  monitoring    : FeatureDriftMonitor, ModelHealthMonitor, PSI/KS primitives
  governance    : GovernanceEngine — promotion, usage contracts, retirement
  explainability: ExplainabilityArtifact (SHAP engine forthcoming)
  research      : ExperimentSummary, HyperparameterStudyArtifact

All domain objects live in ml.contracts.
"""

# Expose most-used public classes at package level for convenience
from ml.contracts import (
    # Enums
    MLTaskFamily,
    ModelStatus,
    DriftSeverity,
    ModelHealthState,
    PromotionOutcome,
    GovernanceStatus,
    MetaLabelAction,
    # Core domain objects
    ModelMetadata,
    ModelCard,
    TrainingRunArtifact,
    EvaluationReport,
    CalibrationReport,
    InferenceRequest,
    InferenceResult,
    DriftReport,
    ModelHealthStatus,
    PromotionDecision,
    ChampionChallengerRecord,
    FallbackPolicy,
    MLUsageContract,
)
from ml.evaluation import ModelEvaluator
from ml.registry import MLModelRegistry, get_ml_registry
from ml.inference import ModelScorer
from ml.monitoring import FeatureDriftMonitor, ModelHealthMonitor
from ml.governance import GovernanceEngine

__all__ = [
    # Enums
    "MLTaskFamily",
    "ModelStatus",
    "DriftSeverity",
    "ModelHealthState",
    "PromotionOutcome",
    "GovernanceStatus",
    "MetaLabelAction",
    # Domain objects
    "ModelMetadata",
    "ModelCard",
    "TrainingRunArtifact",
    "EvaluationReport",
    "CalibrationReport",
    "InferenceRequest",
    "InferenceResult",
    "DriftReport",
    "ModelHealthStatus",
    "PromotionDecision",
    "ChampionChallengerRecord",
    "FallbackPolicy",
    "MLUsageContract",
    # Engines
    "ModelEvaluator",
    "MLModelRegistry",
    "get_ml_registry",
    "ModelScorer",
    "FeatureDriftMonitor",
    "ModelHealthMonitor",
    "GovernanceEngine",
]
