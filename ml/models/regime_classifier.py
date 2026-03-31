# -*- coding: utf-8 -*-
"""
ml/models/regime_classifier.py — Regime Classification Model
=============================================================

Implements RegimeClassifierHookProtocol from core.regime_engine so this model
can be plugged directly into RegimeEngine:

    from ml.models.regime_classifier import RegimeClassificationModel
    model = RegimeClassificationModel.load("models/regime_v1.pkl")
    engine = RegimeEngine(ml_hook=model)

Safety floor doctrine (from CLAUDE.md):
  The RegimeEngine enforces the floor — ML cannot return BROKEN or CRISIS.
  This model nonetheless avoids those labels in its own output as an extra
  defensive layer. Any logic relying solely on this model must not bypass
  the RegimeEngine safety floor.

Label encoding:
  0 → MEAN_REVERTING
  1 → TRENDING
  2 → HIGH_VOL
  3 → CRISIS    (output only if safety floor is disabled)
  4 → BROKEN    (output only if safety floor is disabled)
  5 → UNKNOWN   (fallback)
"""

from __future__ import annotations

import logging
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.contracts import RegimeLabel
from ml.contracts import (
    MLTaskFamily,
    PredictionRecord,
    TrainingRunArtifact,
)
from ml.models.base import MLModel, REGIME_FEATURES

logger = logging.getLogger(__name__)


# ── Label encoding ────────────────────────────────────────────────────────────

REGIME_TO_INT: Dict[RegimeLabel, int] = {
    RegimeLabel.MEAN_REVERTING: 0,
    RegimeLabel.TRENDING: 1,
    RegimeLabel.HIGH_VOL: 2,
    RegimeLabel.CRISIS: 3,
    RegimeLabel.BROKEN: 4,
    RegimeLabel.UNKNOWN: 5,
}

INT_TO_REGIME: Dict[int, RegimeLabel] = {v: k for k, v in REGIME_TO_INT.items()}

# Labels the ML model is allowed to output (safety floor: no BROKEN/CRISIS)
SAFE_OUTPUT_LABELS = {
    0: RegimeLabel.MEAN_REVERTING,
    1: RegimeLabel.TRENDING,
    2: RegimeLabel.HIGH_VOL,
    5: RegimeLabel.UNKNOWN,
}


# ══════════════════════════════════════════════════════════════════════════════
# REGIME CLASSIFICATION MODEL
# ══════════════════════════════════════════════════════════════════════════════

class RegimeClassificationModel:
    """
    Regime classification model implementing RegimeClassifierHookProtocol.

    Designed to be plugged into RegimeEngine via:
        engine = RegimeEngine(ml_hook=RegimeClassificationModel.load(path))

    The model outputs one of: MEAN_REVERTING, TRENDING, HIGH_VOL, UNKNOWN.
    It never outputs BROKEN or CRISIS — those are enforced at the RegimeEngine
    level based on structural rule-based checks. This model adds an extra
    defensive layer by not producing those labels at all.

    Parameters
    ----------
    base_model : MLModel, optional
        Underlying classifier. Defaults to RandomForestModel (multiclass).
    task_family : MLTaskFamily
        Should be REGIME_CLASSIFICATION.
    feature_names : list[str], optional
        Features expected at inference. Defaults to REGIME_FEATURES.
    min_confidence : float
        Minimum predicted probability for the top class to use the ML label.
        Below this threshold, falls back to fallback_regime.
    fallback_regime : RegimeLabel
        Regime to return when model confidence is too low or model is unavailable.
        Defaults to HIGH_VOL (conservative — signals less tradable).
    """

    def __init__(
        self,
        base_model: Optional[MLModel] = None,
        task_family: MLTaskFamily = MLTaskFamily.REGIME_CLASSIFICATION,
        feature_names: Optional[List[str]] = None,
        min_confidence: float = 0.60,
        fallback_regime: RegimeLabel = RegimeLabel.HIGH_VOL,
    ) -> None:
        if base_model is None:
            from ml.models.classifiers import RandomForestModel
            base_model = RandomForestModel(
                task_family=task_family,
                feature_names=feature_names or REGIME_FEATURES,
            )

        self._model = base_model
        self._task_family = task_family
        self._feature_names = feature_names or REGIME_FEATURES
        self._min_confidence = min_confidence
        self._fallback_regime = fallback_regime
        self._model_id = str(uuid.uuid4())[:12]
        self._class_to_label: Dict[int, RegimeLabel] = {}   # Populated after fit

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
        Fit the regime classification model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix indexed by DatetimeIndex.
        y : pd.Series
            Regime labels — either RegimeLabel enum values or integer codes.
            Integers are interpreted as REGIME_TO_INT encoding.
        train_end : str | pd.Timestamp, optional
        sample_weight : np.ndarray, optional
        calibration_X : pd.DataFrame, optional
        calibration_y : pd.Series, optional

        Returns
        -------
        TrainingRunArtifact
        """
        # Normalise y to integer codes
        y_int = self._normalise_labels(y)

        # Build class→label map from training data
        unique_classes = sorted(y_int.unique())
        self._class_to_label = {
            int(c): INT_TO_REGIME.get(int(c), RegimeLabel.UNKNOWN)
            for c in unique_classes
        }

        artifact = self._model.fit(
            X,
            y_int,
            train_end=train_end,
            sample_weight=sample_weight,
            calibration_X=calibration_X,
            calibration_y=calibration_y if calibration_y is None else self._normalise_labels(calibration_y),
        )

        logger.info(
            "RegimeClassificationModel[%s] fitted: %d samples, classes=%s",
            self._model_id[:8],
            artifact.n_train_samples,
            [INT_TO_REGIME.get(c, "?") for c in unique_classes],
        )
        return artifact

    # ── RegimeClassifierHookProtocol interface ────────────────────────────

    def classify(self, features: Any) -> Tuple[RegimeLabel, float]:
        """
        Implements RegimeClassifierHookProtocol.

        Accepts RegimeFeatureSet (from core.regime_engine) or a dict of feature values.
        Returns (RegimeLabel, confidence).

        Safety: never returns BROKEN or CRISIS regardless of model output.
        Falls back to self._fallback_regime if:
        - Model is not fitted
        - Confidence < min_confidence
        - Any error occurs
        """
        if not self._model.is_fitted:
            return self._fallback_regime, 0.0

        try:
            X = self._coerce_to_frame(features)
            probas = self._get_multiclass_proba(X)  # shape (1, n_classes)

            if probas is None or probas.shape[0] == 0:
                return self._fallback_regime, 0.0

            # Find top class
            top_idx = int(np.argmax(probas[0]))
            confidence = float(probas[0, top_idx])

            if confidence < self._min_confidence:
                return self._fallback_regime, confidence

            # Map class index to regime
            # The estimator's classes_ attribute tells us which integer class each column maps to
            regime = self._idx_to_regime(top_idx)

            # Safety floor: never output BROKEN or CRISIS
            if regime in (RegimeLabel.BROKEN, RegimeLabel.CRISIS):
                return self._fallback_regime, confidence

            return regime, confidence

        except Exception as exc:
            logger.warning(
                "RegimeClassificationModel.classify() failed: %s", exc
            )
            return self._fallback_regime, 0.0

    # ── Batch structured predictions ──────────────────────────────────────

    def predict_structured(
        self,
        X: pd.DataFrame,
        entity_ids: Optional[List[str]] = None,
        as_of: Optional[str] = None,
    ) -> List[PredictionRecord]:
        """
        Batch structured predictions with regime label and confidence.
        """
        records = []
        for i in range(len(X)):
            row = X.iloc[[i]]
            regime, confidence = self.classify(row)
            ts = as_of or (str(X.index[i]) if isinstance(X.index[i], pd.Timestamp) else "")
            eid = entity_ids[i] if entity_ids and i < len(entity_ids) else ""

            records.append(PredictionRecord(
                prediction_id=str(uuid.uuid4())[:8],
                model_id=self._model_id,
                task_family=self._task_family.value,
                entity_id=eid,
                prediction_timestamp=ts,
                score=confidence,
                label=regime.value,
                confidence=confidence,
                fallback_used=(regime == self._fallback_regime),
            ))
        return records

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        """Persist this model to disk. Returns absolute path."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("RegimeClassificationModel[%s] saved to %s", self._model_id[:8], dest)
        return str(dest.resolve())

    @classmethod
    def load(cls, path: str) -> "RegimeClassificationModel":
        """Load model from disk."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected RegimeClassificationModel, got {type(obj).__name__}")
        logger.info("RegimeClassificationModel[%s] loaded from %s", obj._model_id[:8], path)
        return obj

    # ── Internal helpers ──────────────────────────────────────────────────

    def _normalise_labels(self, y: pd.Series) -> pd.Series:
        """Convert RegimeLabel enum values to integer codes if needed."""
        def _convert(v: Any) -> int:
            if isinstance(v, RegimeLabel):
                return REGIME_TO_INT.get(v, 5)
            if isinstance(v, str):
                try:
                    return REGIME_TO_INT.get(RegimeLabel(v), 5)
                except ValueError:
                    return 5
            return int(v)

        return y.map(_convert)

    def _coerce_to_frame(self, features: Any) -> pd.DataFrame:
        """Convert various input types to a single-row pd.DataFrame."""
        if isinstance(features, pd.DataFrame):
            return features

        if isinstance(features, dict):
            row = {k: features.get(k, np.nan) for k in self._feature_names}
            return pd.DataFrame([row])

        # Handle RegimeFeatureSet or any dataclass / object with attributes
        if hasattr(features, "__dataclass_fields__") or hasattr(features, "__dict__"):
            try:
                d = {k: getattr(features, k, np.nan) for k in self._feature_names}
                return pd.DataFrame([d])
            except Exception:
                pass

        raise TypeError(
            f"Cannot coerce {type(features).__name__} to DataFrame. "
            "Pass a dict, RegimeFeatureSet, or pd.DataFrame."
        )

    def _get_multiclass_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get multiclass probability output from the base model.
        For binary models, returns a 2-column array.
        """
        estimator = self._model._estimator
        X_filled = X.fillna(0.0).values

        if self._model._calibration_applied and self._model._calibrator is not None:
            raw = self._model._calibrator.predict_proba(X_filled)
        else:
            try:
                raw = estimator.predict_proba(X_filled)
            except Exception as exc:
                logger.warning("predict_proba failed in regime model: %s", exc)
                return None

        return raw

    def _idx_to_regime(self, col_idx: int) -> RegimeLabel:
        """
        Convert probability array column index to RegimeLabel.
        Uses estimator.classes_ if available to map sklearn's internal ordering.
        """
        estimator = self._model._estimator
        if hasattr(estimator, "classes_"):
            classes = list(estimator.classes_)
            if col_idx < len(classes):
                int_class = int(classes[col_idx])
                # Check safe labels first
                if int_class in SAFE_OUTPUT_LABELS:
                    return SAFE_OUTPUT_LABELS[int_class]
                return INT_TO_REGIME.get(int_class, self._fallback_regime)
        # Fallback: treat col_idx directly as integer class
        return SAFE_OUTPUT_LABELS.get(col_idx, self._fallback_regime)
