# -*- coding: utf-8 -*-
"""
ml/models/break_detector.py — Structural Break / Instability Detection Model
=============================================================================

Predicts probability that a pairs relationship is deteriorating or broken.

Use cases:
  - Early warning before classical rules (ADF, correlation drop) detect the break
  - Supplementing regime classification with pair-specific instability signals
  - Risk overlay: elevate exit caution when P(break) is high

The model outputs P(break within label_horizon days) in [0, 1].
It is NOT a substitute for the classical structural break rules in
core/regime_engine.py — it is an additional probabilistic signal.

Key design decisions:
  - Fallback returns fallback_probability (not 0 or 1)
  - All predict methods are exception-safe
  - Works on individual feature dicts, dataclass objects, or pd.DataFrames
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
    MLTaskFamily,
    PredictionRecord,
    TrainingRunArtifact,
)
from ml.models.base import MLModel, BREAK_DETECTION_FEATURES

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# BREAK DETECTION MODEL
# ══════════════════════════════════════════════════════════════════════════════

class BreakDetectionModel:
    """
    Break / instability detection model.

    Predicts probability that a relationship is deteriorating or broken within
    the training label horizon (e.g. "break_20d" = break within 20 days).

    Can serve as an early warning system before classical rules detect the break.

    Parameters
    ----------
    base_model : MLModel, optional
        Underlying classifier. Defaults to RandomForestModel.
    task_family : MLTaskFamily
        Should be BREAK_DETECTION.
    label_name : str
        Name of the label this model was trained on. Default "break_20d".
    feature_names : list[str], optional
        Feature names expected at inference. Defaults to BREAK_DETECTION_FEATURES.
    break_probability_threshold : float
        P(break) above this value → is_break_risk_elevated() returns True.
        Default 0.65.
    fallback_probability : float
        Value returned when the model is unavailable or errors.
        Default 0.20 (slight non-zero prior on break risk).
    """

    def __init__(
        self,
        base_model: Optional[MLModel] = None,
        task_family: MLTaskFamily = MLTaskFamily.BREAK_DETECTION,
        label_name: str = "break_20d",
        feature_names: Optional[List[str]] = None,
        break_probability_threshold: float = 0.65,
        fallback_probability: float = 0.20,
    ) -> None:
        if base_model is None:
            from ml.models.classifiers import RandomForestModel
            base_model = RandomForestModel(
                task_family=task_family,
                feature_names=feature_names or BREAK_DETECTION_FEATURES,
            )

        self._model = base_model
        self._task_family = task_family
        self._label_name = label_name
        self._feature_names = feature_names or BREAK_DETECTION_FEATURES
        self._break_threshold = break_probability_threshold
        self._fallback_probability = fallback_probability
        self._model_id = str(uuid.uuid4())[:12]

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_fitted(self) -> bool:
        return self._model.is_fitted

    @property
    def break_probability_threshold(self) -> float:
        return self._break_threshold

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
        Fit the break detection model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix indexed by DatetimeIndex.
        y : pd.Series
            Binary labels: 1 = break occurred within horizon, 0 = relationship stable.
        train_end : str | pd.Timestamp, optional
        sample_weight : np.ndarray, optional
        calibration_X : pd.DataFrame, optional
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
            "BreakDetectionModel[%s] fitted: %d samples, label=%s",
            self._model_id[:8],
            artifact.n_train_samples,
            self._label_name,
        )
        return artifact

    # ── Primary prediction interface ──────────────────────────────────────

    def predict_break_probability(self, features: Any) -> float:
        """
        Return P(break within label horizon). Fallback-safe.

        Parameters
        ----------
        features : dict | SignalFeatures | pd.DataFrame | RegimeFeatureSet
            Feature values. NaN is acceptable for missing features.

        Returns
        -------
        float in [0, 1].
        Returns fallback_probability if model is not fitted or on any error.
        """
        if not self._model.is_fitted:
            return self._fallback_probability

        try:
            X = self._coerce_to_frame(features)
            probas = self._model.predict_proba(X)
            # Column 1 = probability of break (positive class)
            p_break = float(probas[0, 1]) if probas.shape[1] > 1 else float(probas[0, 0])
            return np.clip(p_break, 0.0, 1.0).item()
        except Exception as exc:
            logger.warning(
                "BreakDetectionModel predict_break_probability failed: %s", exc
            )
            return self._fallback_probability

    def is_break_risk_elevated(self, features: Any) -> bool:
        """
        Return True if P(break) > break_probability_threshold.

        This is the primary boolean signal for downstream consumers
        (risk engine, exit logic, etc.).
        """
        p = self.predict_break_probability(features)
        return p > self._break_threshold

    # ── Batch structured predictions ──────────────────────────────────────

    def predict_structured(
        self,
        X: pd.DataFrame,
        entity_ids: Optional[List[str]] = None,
        as_of: Optional[str] = None,
    ) -> List[PredictionRecord]:
        """
        Batch structured predictions. Includes break risk flag in label field.

        Parameters
        ----------
        X : pd.DataFrame
        entity_ids : list[str], optional
        as_of : str, optional

        Returns
        -------
        list[PredictionRecord]
        """
        if not self._model.is_fitted:
            # Return fallback records for all rows
            records = []
            for i in range(len(X)):
                ts = as_of or (str(X.index[i]) if isinstance(X.index[i], pd.Timestamp) else "")
                eid = entity_ids[i] if entity_ids and i < len(entity_ids) else ""
                records.append(PredictionRecord(
                    prediction_id=str(uuid.uuid4())[:8],
                    model_id=self._model_id,
                    task_family=self._task_family.value,
                    entity_id=eid,
                    prediction_timestamp=ts,
                    score=self._fallback_probability,
                    label="BREAK_RISK_UNKNOWN",
                    confidence=0.0,
                    fallback_used=True,
                    fallback_reason="model_not_fitted",
                ))
            return records

        base_records = self._model.predict_structured(X, as_of=as_of, entity_ids=entity_ids)

        enriched = []
        for rec in base_records:
            p_break = rec.score
            elevated = p_break > self._break_threshold
            label = "BREAK_RISK_ELEVATED" if elevated else "BREAK_RISK_NORMAL"

            enriched.append(PredictionRecord(
                prediction_id=rec.prediction_id,
                model_id=self._model_id,
                model_version=rec.model_version,
                task_family=self._task_family.value,
                entity_id=rec.entity_id,
                prediction_timestamp=rec.prediction_timestamp,
                score=p_break,
                label=label,
                confidence=rec.confidence,
                uncertainty=1.0 - rec.confidence,
                calibration_applied=rec.calibration_applied,
                feature_availability=rec.feature_availability,
                missing_features=rec.missing_features,
                fallback_used=rec.fallback_used,
                fallback_reason=rec.fallback_reason,
                warnings=rec.warnings,
                recommended_use="exit_caution" if elevated else "normal_monitoring",
            ))
        return enriched

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> str:
        """Persist this model to disk. Returns absolute path."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BreakDetectionModel[%s] saved to %s", self._model_id[:8], dest)
        return str(dest.resolve())

    @classmethod
    def load(cls, path: str) -> "BreakDetectionModel":
        """Load model from disk."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected BreakDetectionModel, got {type(obj).__name__}")
        logger.info("BreakDetectionModel[%s] loaded from %s", obj._model_id[:8], path)
        return obj

    # ── Internal helpers ──────────────────────────────────────────────────

    def _coerce_to_frame(self, features: Any) -> pd.DataFrame:
        """Convert various input types to a single-row pd.DataFrame."""
        if isinstance(features, pd.DataFrame):
            return features

        if isinstance(features, dict):
            row = {k: features.get(k, np.nan) for k in self._feature_names}
            return pd.DataFrame([row])

        # Handle dataclass / object with attributes
        if hasattr(features, "__dataclass_fields__") or hasattr(features, "__dict__"):
            try:
                d = {k: getattr(features, k, np.nan) for k in self._feature_names}
                return pd.DataFrame([d])
            except Exception:
                pass

        raise TypeError(
            f"Cannot coerce {type(features).__name__} to DataFrame. "
            "Pass a dict, feature object, or pd.DataFrame."
        )
