# -*- coding: utf-8 -*-
"""
ml/models/__init__.py — ML Model Implementations
=================================================

Exports all concrete ML model classes for use across the ML platform.
"""

from __future__ import annotations

from ml.models.base import MLModel
from ml.models.calibration import (
    CalibratedModelWrapper,
    compute_calibration_metrics,
    brier_score,
)
from ml.models.classifiers import (
    LogisticRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    XGBoostModel,
    build_model,
)
from ml.models.meta_labeler import MetaLabelModel
from ml.models.regime_classifier import RegimeClassificationModel
from ml.models.break_detector import BreakDetectionModel

__all__ = [
    # Base
    "MLModel",
    # Calibration utilities
    "CalibratedModelWrapper",
    "compute_calibration_metrics",
    "brier_score",
    # Concrete classifiers
    "LogisticRegressionModel",
    "RandomForestModel",
    "GradientBoostingModel",
    "XGBoostModel",
    "build_model",
    # Specialized models
    "MetaLabelModel",
    "RegimeClassificationModel",
    "BreakDetectionModel",
]
