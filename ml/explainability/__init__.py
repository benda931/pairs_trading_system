# -*- coding: utf-8 -*-
"""
ml/explainability — Feature importance and explainability artifacts.

Exposes:
  - ExplainabilityArtifact (from ml.contracts)
  - FeatureLineageTracker  (lineage.py)
  - compute_feature_importance, generate_importance_report,
    rank_features_by_importance (importance.py)
"""

from ml.contracts import ExplainabilityArtifact
from ml.explainability.lineage import FeatureLineageTracker
from ml.explainability.importance import (
    compute_feature_importance,
    generate_importance_report,
    rank_features_by_importance,
)

__all__ = [
    "ExplainabilityArtifact",
    "FeatureLineageTracker",
    "compute_feature_importance",
    "generate_importance_report",
    "rank_features_by_importance",
]
