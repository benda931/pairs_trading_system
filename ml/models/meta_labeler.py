# -*- coding: utf-8 -*-
"""
ml/models/meta_labeler.py — Meta-Labeling Model
================================================

Meta-labeling (de Prado, 2018): given a rule-based signal, predict whether
to TAKE it, SKIP it, TAKE a smaller position, or wait for confirmation.

Design
------
MetaLabelModel wraps an MLModel and implements MetaLabelProtocol so it can
be plugged directly into SignalQualityEngine:

    from ml.models.meta_labeler import MetaLabelModel
    model = MetaLabelModel.load("models/meta_label_v2.pkl")
    engine = SignalQualityEngine(ml_hook=model)

The model accepts either:
  - SignalFeatures (from core.diagnostics)
  - dict of feature values
  - pd.DataFrame row(s)

All prediction methods are fallback-safe.
The rule-based floor (grade F) is NEVER overridden by this model.
"""

from __future__ import annotations

import logging
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.contracts import (
    MetaLabelAction,
    MLTaskFamily,
    PredictionRecord,
    TrainingRunArtifact,
)
from ml.models.base import MLModel, META_LABEL_FEATURES

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# META LABEL MODEL
# ══════════════════════════════════════════════════════════════════════════════

class MetaLabelModel:
    """
    Meta-labeling model: given a rule-based signal, predict whether to take/skip/resize.

    Implements core.signal_quality.MetaLabelProtocol so it can be plugged directly
    into SignalQualityEngine.

    Parameters
    ----------
    base_model : MLModel, optional
        Underlying classifier. Defaults to LogisticRegressionModel.
    task_family : MLTaskFamily
        Should be META_LABELING.
    label_name : str
        Name of the label this model was trained on (e.g. "meta_take_10d").
    feature_names : list[str], optional
        Feature names expected at inference. Defaults to META_LABEL_FEATURES.
    action_threshold : float
        P(take) above this → TAKE. Default 0.55.
    downsize_threshold : float
        P(take) between downsize_threshold and action_threshold → TAKE_SMALLER. Default 0.45.
    skip_threshold : float
        P(take) below this → SKIP. Default 0.35.
    fallback_action : MetaLabelAction
        Action to return when the model is unavailable. Default TAKE (conservative).
    """

    def __init__(
        self,
        base_model: Optional[MLModel] = None,
        task_family: MLTaskFamily = MLTaskFamily.META_LABELING,
        label_name: str = "meta_take_10d",
        feature_names: Optional[List[str]] = None,
        action_threshold: float = 0.55,
        downsize_threshold: float = 0.45,
        skip_threshold: float = 0.35,
        fallback_action: MetaLabelAction = MetaLabelAction.TAKE,
    ) -> None:
        if base_model is None:
            from ml.models.classifiers import LogisticRegressionModel
            base_model = LogisticRegressionModel(task_family=task_family)

        self._model = base_model
        self._task_family = task_family
        self._label_name = label_name
        self._feature_names = feature_names or META_LABEL_FEATURES
        self._action_threshold = action_threshold
        self._downsize_threshold = downsize_threshold
        self._skip_threshold = skip_threshold
        self._fallback_action = fallback_action
        self._model_id = str(uuid.uuid4())[:12]

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_fitted(self) -> bool:
        return self._model.is_fitted

    # ── Fitting ───────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        train_end: Any = None,
        sample_weight: Optional[np.ndarray] = None,
        calibration_X: Optional[pd.DataFrame] = None,
        calibration_y: Optional[pd.Series] = None,
    ) -> TrainingRunArtifact:
        """
        Fit the meta-label model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix indexed by DatetimeIndex.
        y : pd.Series
            Binary labels: 1 = signal succeeded, 0 = signal failed.
        train_end : str | pd.Timestamp, optional
            Hard cutoff for leakage prevention.
        sample_weight : np.ndarray, optional
        calibration_X : pd.DataFrame, optional
            Held-out calibration features.
        calibration_y : pd.Series, optional

        Returns
        -------
        TrainingRunArtifact
        """
        artifact = self._model.fit(
            X,
            y,
            train_end=train_end,
            sample_weight=sample_weight,
            calibration_X=calibration_X,
            calibration_y=calibration_y,
        )
        logger.info(
            "MetaLabelModel[%s] fitted on %d samples, label=%s",
            self._model_id[:8],
            artifact.n_train_samples,
            self._label_name,
        )
        return artifact

    # ── MetaLabelProtocol interface ───────────────────────────────────────

    def predict_success_probability(self, features: Any) -> float:
        """
        Implements MetaLabelProtocol.predict_success_probability.

        Accepts:
        - SignalFeatures (from core.diagnostics)
        - dict of feature values
        - pd.DataFrame (first row used)

        Returns
        -------
        float in [0, 1], or NaN if model is not fitted.
        """
        if not self._model.is_fitted:
            return float("nan")

        try:
            X = self._coerce_to_frame(features)
            probas = self._model.predict_proba(X)
            return float(probas[0, 1]) if probas.shape[1] > 1 else float(probas[0, 0])
        except Exception as exc:
            logger.warning(
                "MetaLabelModel predict_success_probability failed: %s", exc
            )
            return float("nan")

    def recommend_action(self, features: Any) -> MetaLabelAction:
        """
        Return a structured MetaLabelAction recommendation.

        Decision logic:
          p >= action_threshold               → TAKE
          downsize_threshold <= p < action_threshold  → TAKE_SMALLER
          skip_threshold <= p < downsize_threshold    → WAIT_FOR_CONFIRMATION
          p < skip_threshold                  → SKIP

        Falls back to self._fallback_action if model is not fitted.
        """
        p = self.predict_success_probability(features)

        if np.isnan(p):
            return self._fallback_action

        if p >= self._action_threshold:
            return MetaLabelAction.TAKE
        elif p >= self._downsize_threshold:
            return MetaLabelAction.TAKE_SMALLER
        elif p >= self._skip_threshold:
            return MetaLabelAction.WAIT_FOR_CONFIRMATION
        else:
            return MetaLabelAction.SKIP

    # ── Batch structured predictions ──────────────────────────────────────

    def predict_structured(
        self,
        X: pd.DataFrame,
        entity_ids: Optional[List[str]] = None,
        as_of: Optional[str] = None,
    ) -> List[PredictionRecord]:
        """
        Batch structured predictions with MetaLabelAction populated.

        Parameters
        ----------
        X : pd.DataFrame
        entity_ids : list[str], optional
        as_of : str, optional

        Returns
        -------
        list[PredictionRecord]
        """
        records = self._model.predict_structured(X, as_of=as_of, entity_ids=entity_ids)

        # Enrich each record with the meta_action
        enriched = []
        for rec in records:
            action = self._score_to_action(rec.score)
            enriched.append(PredictionRecord(
                prediction_id=rec.prediction_id,
                model_id=self._model_id,
                model_version=rec.model_version,
                task_family=self._task_family.value,
                entity_id=rec.entity_id,
                prediction_timestamp=rec.prediction_timestamp,
                score=rec.score,
                label=action.value,
                meta_action=action,
                confidence=rec.confidence,
                uncertainty=rec.uncertainty,
                calibration_applied=rec.calibration_applied,
                feature_availability=rec.feature_availability,
                missing_features=rec.missing_features,
                fallback_used=rec.fallback_used,
                fallback_reason=rec.fallback_reason,
                warnings=rec.warnings,
            ))
        return enriched

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        """Persist this MetaLabelModel to disk. Returns absolute path."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("MetaLabelModel[%s] saved to %s", self._model_id[:8], dest)
        return str(dest.resolve())

    @classmethod
    def load(cls, path: str) -> "MetaLabelModel":
        """Load a MetaLabelModel from disk."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected MetaLabelModel, got {type(obj).__name__}")
        logger.info("MetaLabelModel[%s] loaded from %s", obj._model_id[:8], path)
        return obj

    # ── Internal helpers ──────────────────────────────────────────────────

    def _coerce_to_frame(self, features: Any) -> pd.DataFrame:
        """Convert various input types to a single-row pd.DataFrame."""
        if isinstance(features, pd.DataFrame):
            return features

        if isinstance(features, dict):
            row = {k: features.get(k, np.nan) for k in self._feature_names}
            return pd.DataFrame([row])

        # Attempt to handle SignalFeatures dataclass by converting to dict
        if hasattr(features, "__dataclass_fields__") or hasattr(features, "__dict__"):
            try:
                d = {
                    k: getattr(features, k, np.nan)
                    for k in self._feature_names
                }
                return pd.DataFrame([d])
            except Exception:
                pass

        raise TypeError(
            f"Cannot coerce {type(features).__name__} to DataFrame. "
            "Pass a dict, SignalFeatures, or pd.DataFrame."
        )

    def _score_to_action(self, score: float) -> MetaLabelAction:
        """Convert raw score to MetaLabelAction."""
        if np.isnan(score):
            return self._fallback_action
        if score >= self._action_threshold:
            return MetaLabelAction.TAKE
        elif score >= self._downsize_threshold:
            return MetaLabelAction.TAKE_SMALLER
        elif score >= self._skip_threshold:
            return MetaLabelAction.WAIT_FOR_CONFIRMATION
        else:
            return MetaLabelAction.SKIP
