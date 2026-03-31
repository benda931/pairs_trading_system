# -*- coding: utf-8 -*-
"""
ml/explainability/importance.py — Feature Importance Utilities
===============================================================

Compute, rank, and report feature importances for trained ML models.

Supports three methods:
  - "gain"        : use estimator.feature_importances_ (tree models)
  - "coef"        : use estimator.coef_ (linear models)
  - "permutation" : sklearn permutation_importance (model-agnostic, slower)

All public functions are fallback-safe: they never raise. On any failure
they return an empty dict or a neutral ExplainabilityArtifact.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.contracts import ExplainabilityArtifact

if TYPE_CHECKING:
    pass  # avoid circular imports for MLModel type hint

logger = logging.getLogger("ml.explainability.importance")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_feature_importance(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "gain",
) -> Dict[str, float]:
    """
    Compute feature importances using the requested method.

    Methods
    -------
    "gain"
        Use ``estimator.feature_importances_`` from tree-based models
        (RandomForest, GradientBoosting, XGBoost, LightGBM, etc.).
        Fast — no re-fitting required.

    "coef"
        Use ``estimator.coef_`` from linear models (LogisticRegression,
        Ridge, Lasso, etc.). Absolute values are used so positive and
        negative coefficients are treated symmetrically.

    "permutation"
        Use ``sklearn.inspection.permutation_importance``. Works for any
        model that implements ``score()``. Slower — requires one pass over
        the data per feature.

    Parameters
    ----------
    model : MLModel or sklearn estimator
        A fitted model. If it has a ``._estimator`` attribute (MLModel wrapper),
        the underlying estimator is used for ``gain`` / ``coef`` methods.
    X : pd.DataFrame
        Feature matrix used for importance computation. For "permutation",
        this is the evaluation set (not necessarily training data).
    y : pd.Series
        True labels aligned with X. Required for "permutation".
    method : str
        One of "gain", "coef", "permutation". Default "gain".

    Returns
    -------
    dict[str, float]
        {feature_name: importance_score} normalised so values sum to 1.0.
        Returns empty dict if importance cannot be computed.
    """
    feature_names = list(X.columns)
    if not feature_names:
        logger.warning("compute_feature_importance: X has no columns")
        return {}

    # Resolve the underlying sklearn estimator
    estimator = _resolve_estimator(model)

    try:
        if method == "gain":
            return _importance_gain(estimator, feature_names)
        elif method == "coef":
            return _importance_coef(estimator, feature_names)
        elif method == "permutation":
            return _importance_permutation(model, X, y, feature_names)
        else:
            logger.warning(
                "compute_feature_importance: unknown method '%s'. "
                "Falling back to 'gain', then 'coef'.", method
            )
            result = _importance_gain(estimator, feature_names)
            if not result:
                result = _importance_coef(estimator, feature_names)
            return result

    except Exception as exc:
        logger.error("compute_feature_importance failed (method=%s): %s", method, exc)
        return {}


def generate_importance_report(
    model_id: str,
    importance_dict: Dict[str, float],
    top_n: int = 15,
) -> ExplainabilityArtifact:
    """
    Create an ExplainabilityArtifact from a pre-computed importance dict.

    Parameters
    ----------
    model_id : str
    importance_dict : dict[str, float]
        As returned by ``compute_feature_importance``.
    top_n : int
        Number of top features to include in ``top_features_ranked``. Default 15.

    Returns
    -------
    ExplainabilityArtifact
        Populated artifact. If importance_dict is empty, the artifact still
        has valid default fields.
    """
    ranked = rank_features_by_importance(importance_dict, top_n=top_n)
    top_names = [name for name, _ in ranked]

    # Determine importance method heuristic from dict contents
    # (not always possible to infer; default to "gain")
    importance_method = "gain"

    artifact = ExplainabilityArtifact(
        model_id=model_id,
        global_feature_importances=dict(importance_dict),
        top_features_ranked=top_names,
        importance_method=importance_method,
    )
    return artifact


def rank_features_by_importance(
    importance_dict: Dict[str, float],
    top_n: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Return sorted (feature_name, importance_score) pairs in descending order.

    Parameters
    ----------
    importance_dict : dict[str, float]
        Feature name to importance score mapping.
    top_n : int | None
        If given, return only the top_n entries.

    Returns
    -------
    list[tuple[str, float]]
        Sorted by importance score descending.
    """
    if not importance_dict:
        return []

    items = sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        items = items[:top_n]
    return items


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_estimator(model: object) -> object:
    """Return the underlying sklearn estimator from an MLModel wrapper, or the model itself."""
    if hasattr(model, "_estimator"):
        return model._estimator
    return model


def _normalise(raw: np.ndarray) -> np.ndarray:
    """Normalise a 1-D array of non-negative values to sum=1. Returns zeros on empty/NaN."""
    arr = np.asarray(raw, dtype=float)
    arr = np.where(np.isfinite(arr), np.abs(arr), 0.0)
    total = arr.sum()
    if total == 0.0:
        return arr
    return arr / total


def _importance_gain(
    estimator: object,
    feature_names: List[str],
) -> Dict[str, float]:
    """Extract feature importances from tree-model's .feature_importances_ attribute."""
    if not hasattr(estimator, "feature_importances_"):
        return {}

    raw = np.asarray(estimator.feature_importances_, dtype=float)
    if raw.shape[0] != len(feature_names):
        logger.warning(
            "_importance_gain: mismatch between feature_importances_ length (%d) "
            "and feature_names length (%d)",
            raw.shape[0], len(feature_names),
        )
        return {}

    normed = _normalise(raw)
    return {name: float(score) for name, score in zip(feature_names, normed)}


def _importance_coef(
    estimator: object,
    feature_names: List[str],
) -> Dict[str, float]:
    """Extract feature importances from linear model's .coef_ attribute."""
    if not hasattr(estimator, "coef_"):
        return {}

    coef = np.asarray(estimator.coef_, dtype=float)
    # coef_ can be 2-D for multi-class; take mean abs across classes
    if coef.ndim == 2:
        coef = np.abs(coef).mean(axis=0)
    else:
        coef = np.abs(coef)

    if coef.shape[0] != len(feature_names):
        logger.warning(
            "_importance_coef: mismatch between coef_ length (%d) "
            "and feature_names length (%d)",
            coef.shape[0], len(feature_names),
        )
        return {}

    normed = _normalise(coef)
    return {name: float(score) for name, score in zip(feature_names, normed)}


def _importance_permutation(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
) -> Dict[str, float]:
    """Compute permutation importance using sklearn."""
    try:
        from sklearn.inspection import permutation_importance
    except ImportError:
        logger.warning(
            "_importance_permutation: sklearn not available; returning empty dict"
        )
        return {}

    try:
        X_arr = X.fillna(0.0).values
        y_arr = np.asarray(y, dtype=float)

        # Use the public interface (MLModel has predict_proba; sklearn estimator also works)
        result = permutation_importance(
            model,
            X_arr,
            y_arr,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        raw = result.importances_mean

        if raw.shape[0] != len(feature_names):
            logger.warning(
                "_importance_permutation: shape mismatch (%d vs %d)",
                raw.shape[0], len(feature_names),
            )
            return {}

        normed = _normalise(raw)
        return {name: float(score) for name, score in zip(feature_names, normed)}

    except Exception as exc:
        logger.warning("_importance_permutation failed: %s", exc)
        return {}
