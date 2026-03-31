# -*- coding: utf-8 -*-
"""
ml/models/classifiers.py — Concrete Classifier Implementations
==============================================================

Provides ready-to-use classifier wrappers built on top of MLModel:

  LogisticRegressionModel  — interpretable calibrated baseline
  RandomForestModel        — robust ensemble for regime/break detection
  GradientBoostingModel    — sklearn GBM ranking model
  XGBoostModel             — XGBoost with sklearn GBM fallback

All classes share the same MLModel interface (fit, predict_proba,
predict_structured, score, save, load) and are fallback-safe.

Factory function:
  build_model(model_class, task_family, **kwargs) -> MLModel
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ml.contracts import MLTaskFamily
from ml.models.base import MLModel, META_LABEL_FEATURES

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

class LogisticRegressionModel(MLModel):
    """
    Calibrated logistic regression — best interpretable baseline.

    Suitable for:
    - Meta-labeling with small feature sets
    - Quick baseline before tree-based models
    - Interpretability-first deployments

    Default hyperparameters:
    - C=0.1      — moderate L2 regularisation (prevents overfit on small samples)
    - max_iter=1000
    - class_weight="balanced" — handles class imbalance automatically
    - solver="lbfgs"          — efficient for small-medium problems
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "C": 0.1,
        "max_iter": 1000,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "random_state": 42,
    }

    def __init__(
        self,
        task_family: MLTaskFamily = MLTaskFamily.META_LABELING,
        feature_names: Optional[list] = None,
        calibrate: bool = True,
        calibration_method: str = "isotonic",
        **params: Any,
    ) -> None:
        from sklearn.linear_model import LogisticRegression

        merged = {**self.DEFAULT_PARAMS, **params}
        estimator = LogisticRegression(**merged)

        super().__init__(
            estimator=estimator,
            task_family=task_family,
            feature_names=feature_names or META_LABEL_FEATURES,
            calibrate=calibrate,
            calibration_method=calibration_method,
        )
        logger.debug("LogisticRegressionModel created with params: %s", merged)


# ══════════════════════════════════════════════════════════════════════════════
# RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════

class RandomForestModel(MLModel):
    """
    Random forest — robust ensemble for regime classification and break detection.

    Suitable for:
    - Regime classification (non-linear decision boundaries)
    - Break / instability detection
    - Cases where feature interactions matter

    Default hyperparameters are conservative (shallow trees, large leaf minimum)
    to reduce overfitting on financial time series.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 4,
        "min_samples_leaf": 20,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(
        self,
        task_family: MLTaskFamily = MLTaskFamily.META_LABELING,
        feature_names: Optional[list] = None,
        calibrate: bool = True,
        calibration_method: str = "isotonic",
        **params: Any,
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier

        merged = {**self.DEFAULT_PARAMS, **params}
        estimator = RandomForestClassifier(**merged)

        super().__init__(
            estimator=estimator,
            task_family=task_family,
            feature_names=feature_names or META_LABEL_FEATURES,
            calibrate=calibrate,
            calibration_method=calibration_method,
        )
        logger.debug("RandomForestModel created with params: %s", merged)


# ══════════════════════════════════════════════════════════════════════════════
# GRADIENT BOOSTING
# ══════════════════════════════════════════════════════════════════════════════

class GradientBoostingModel(MLModel):
    """
    Gradient boosting (sklearn GBM) — calibrated ranking model.

    Suitable for:
    - Candidate ranking tasks
    - Meta-labeling with larger training sets
    - Portfolio utility scoring

    Uses shallow trees (max_depth=3) and subsample to reduce overfitting.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_samples_leaf": 20,
        "random_state": 42,
    }

    def __init__(
        self,
        task_family: MLTaskFamily = MLTaskFamily.META_LABELING,
        feature_names: Optional[list] = None,
        calibrate: bool = True,
        calibration_method: str = "isotonic",
        **params: Any,
    ) -> None:
        from sklearn.ensemble import GradientBoostingClassifier

        merged = {**self.DEFAULT_PARAMS, **params}
        estimator = GradientBoostingClassifier(**merged)

        super().__init__(
            estimator=estimator,
            task_family=task_family,
            feature_names=feature_names or META_LABEL_FEATURES,
            calibrate=calibrate,
            calibration_method=calibration_method,
        )
        logger.debug("GradientBoostingModel created with params: %s", merged)


# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

class XGBoostModel(MLModel):
    """
    XGBoost classifier wrapper with lazy import and sklearn GBM fallback.

    If xgboost is not installed, silently falls back to GradientBoostingClassifier.
    The caller cannot distinguish the difference at the MLModel interface level —
    both support predict_proba.

    Suitable for:
    - Large feature sets (> 50 features)
    - Best absolute performance on well-sized datasets
    - When interpretability is a secondary concern
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
    }

    def __init__(
        self,
        task_family: MLTaskFamily = MLTaskFamily.META_LABELING,
        feature_names: Optional[list] = None,
        calibrate: bool = True,
        calibration_method: str = "isotonic",
        **params: Any,
    ) -> None:
        merged = {**self.DEFAULT_PARAMS, **params}

        try:
            from xgboost import XGBClassifier
            # XGBoost >= 1.6 removed use_label_encoder
            xgb_params = {k: v for k, v in merged.items() if k != "use_label_encoder"}
            try:
                estimator = XGBClassifier(**xgb_params)
            except TypeError:
                # Older XGBoost — pass all params
                estimator = XGBClassifier(**merged)
            logger.debug("XGBoostModel using xgboost backend")
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            gbm_params = {
                k: v
                for k, v in GradientBoostingModel.DEFAULT_PARAMS.items()
            }
            estimator = GradientBoostingClassifier(**gbm_params)
            logger.warning(
                "xgboost not installed — XGBoostModel falling back to GradientBoostingClassifier"
            )

        super().__init__(
            estimator=estimator,
            task_family=task_family,
            feature_names=feature_names or META_LABEL_FEATURES,
            calibrate=calibrate,
            calibration_method=calibration_method,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_REGISTRY: Dict[str, type] = {
    "logistic": LogisticRegressionModel,
    "logisticregression": LogisticRegressionModel,
    "logisticregressionmodel": LogisticRegressionModel,
    "randomforest": RandomForestModel,
    "randomforestmodel": RandomForestModel,
    "gradientboosting": GradientBoostingModel,
    "gradientboostingmodel": GradientBoostingModel,
    "xgboost": XGBoostModel,
    "xgboostmodel": XGBoostModel,
    "gbm": GradientBoostingModel,
    "rf": RandomForestModel,
    "lr": LogisticRegressionModel,
    "xgb": XGBoostModel,
}


def build_model(
    model_class: str,
    task_family: MLTaskFamily,
    **kwargs: Any,
) -> MLModel:
    """
    Factory function — build a model by class name.

    Parameters
    ----------
    model_class : str
        Case-insensitive model class identifier. Accepted values:
        "logistic", "randomforest", "gradientboosting", "xgboost",
        and common abbreviations ("lr", "rf", "gbm", "xgb").
    task_family : MLTaskFamily
        Task family for the model.
    **kwargs
        Additional hyperparameter overrides passed to the model constructor.

    Returns
    -------
    MLModel

    Raises
    ------
    ValueError
        If model_class is not recognised.
    """
    key = model_class.lower().replace("_", "").replace(" ", "")
    cls = _MODEL_REGISTRY.get(key)
    if cls is None:
        available = sorted(set(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model class {model_class!r}. Available: {available}"
        )
    return cls(task_family=task_family, **kwargs)
