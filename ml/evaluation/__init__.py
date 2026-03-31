# -*- coding: utf-8 -*-
"""
ml/evaluation — Evaluation metrics and reporting for ML models.
"""

from ml.evaluation.metrics import (
    information_coefficient,
    ic_t_stat,
    auc_roc,
    pr_auc,
    brier_score,
    calibration_error,
    meta_label_utility,
    walk_forward_ic,
    regime_sliced_metrics,
    robustness_score,
)
from ml.evaluation.reports import ModelEvaluator

__all__ = [
    "information_coefficient",
    "ic_t_stat",
    "auc_roc",
    "pr_auc",
    "brier_score",
    "calibration_error",
    "meta_label_utility",
    "walk_forward_ic",
    "regime_sliced_metrics",
    "robustness_score",
    "ModelEvaluator",
]
