# -*- coding: utf-8 -*-
"""
ml/research — Offline research and experiment management.

Exposes ExperimentSummary and HyperparameterStudyArtifact from ml.contracts.
Full research pipeline (hyperparameter search, experiment tracking) is in
ml/research/experiments.py (future addition).
"""

from ml.contracts import ExperimentSummary, HyperparameterStudyArtifact

__all__ = [
    "ExperimentSummary",
    "HyperparameterStudyArtifact",
]
