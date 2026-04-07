# -*- coding: utf-8 -*-
"""
ml/models/base.py — Extended ML Model Base
===========================================

Extends the existing models.base.BaseModel with:
- calibration support
- structured prediction output (PredictionRecord)
- artifact persistence (save/load)
- model card generation
- fallback policy integration

Design:
  MLModel wraps any sklearn-compatible estimator and provides a uniform interface
  for fitting, predicting, saving, and explaining models on the ML platform.
  All predict methods are fallback-safe: they never raise; they return a neutral
  output on any failure and log the error.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ml.contracts import (
    FallbackPolicy,
    MLTaskFamily,
    ModelCard,
    ModelStatus,
    PredictionRecord,
    TrainingRunArtifact,
    MetaLabelAction,
)

logger = logging.getLogger(__name__)

# ── Feature lists used as defaults ────────────────────────────────────────

META_LABEL_FEATURES: List[str] = [
    "z_score",
    "z_robust",
    "z_percentile",
    "z_velocity",
    "z_acceleration",
    "time_above_threshold_days",
    "residual_slope",
    "spread_vol",
    "corr_20d",
    "corr_63d",
    "corr_drift",
    "coint_score",
    "half_life_days",
    "half_life_change_pct",
    "beta_cv",
    "residual_var_stability",
    "spread_vol_regime",
    "vol_of_vol",
    "mean_reversion_quality",
    "regime_safety",
    "z_persistence",
    "break_risk",
]

REGIME_FEATURES: List[str] = [
    "z_score",
    "z_velocity",
    "z_acceleration",
    "z_persistence",
    "spread_vol",
    "spread_vol_regime",
    "vol_of_vol",
    "corr_20d",
    "corr_63d",
    "corr_252d",
    "corr_drift",
    "half_life_days",
    "beta_cv",
    "break_risk",
    "mean_reversion_quality",
]

BREAK_DETECTION_FEATURES: List[str] = [
    "z_score",
    "z_velocity",
    "z_acceleration",
    "spread_vol",
    "spread_vol_regime",
    "vol_of_vol",
    "corr_20d",
    "corr_drift",
    "half_life_days",
    "half_life_change_pct",
    "beta_cv",
    "residual_var_stability",
    "break_risk",
    "adf_pvalue",
    "hurst_exponent",
]


# ══════════════════════════════════════════════════════════════════════════════
# MLMODEL — EXTENDED MODEL BASE
# ══════════════════════════════════════════════════════════════════════════════

class MLModel:
    """
    Extended model base for the ML platform.

    Wraps sklearn-compatible estimators with:
    - leakage guard on fit()
    - optional calibration wrapper (held-out data only)
    - structured predict() returning PredictionRecord objects
    - artifact save/load via pickle
    - model card generation

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Must implement fit(X, y) and predict_proba(X).
    model_id : str, optional
        Stable identifier for this model. Auto-generated if not provided.
    task_family : MLTaskFamily
        High-level task this model serves.
    feature_names : list[str], optional
        Canonical feature names. Stored for validation on predict.
    calibrate : bool
        Whether to apply calibration when calibration data is provided.
    calibration_method : str
        "isotonic" (default) or "sigmoid" (Platt scaling).
    fallback_policy : FallbackPolicy, optional
        Policy governing fallback behaviour when prediction fails.
    """

    def __init__(
        self,
        estimator: Any,
        model_id: Optional[str] = None,
        task_family: MLTaskFamily = MLTaskFamily.META_LABELING,
        feature_names: Optional[List[str]] = None,
        calibrate: bool = True,
        calibration_method: str = "isotonic",
        fallback_policy: Optional[FallbackPolicy] = None,
    ) -> None:
        self._estimator = estimator
        self._model_id = model_id or str(uuid.uuid4())[:12]
        self._task_family = task_family
        self._feature_names: List[str] = feature_names or []
        self._calibrate = calibrate
        self._calibration_method = calibration_method
        self._fallback_policy = fallback_policy

        self._fitted = False
        self._calibrator: Any = None           # Set after calibration
        self._calibration_applied = False
        self._training_artifact: Optional[TrainingRunArtifact] = None
        self._train_end: Optional[pd.Timestamp] = None
        self._model_class_name = type(estimator).__name__

    # ── Public properties ──────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def task_family(self) -> MLTaskFamily:
        return self._task_family

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def training_artifact(self) -> Optional[TrainingRunArtifact]:
        return self._training_artifact

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
    ) -> "TrainingRunArtifact":
        """
        Fit the model with a leakage guard.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix indexed by DatetimeIndex.
        y : pd.Series
            Labels aligned to X.
        train_end : str | pd.Timestamp, optional
            Hard cutoff. Raises ValueError if X contains rows after this date.
        sample_weight : np.ndarray, optional
            Per-sample weights passed to the estimator's fit method.
        calibration_X : pd.DataFrame, optional
            Held-out calibration features. Must NOT overlap with X.
        calibration_y : pd.Series, optional
            Held-out calibration labels.

        Returns
        -------
        TrainingRunArtifact
            Provenance record for this training run.

        Raises
        ------
        ValueError
            If X contains rows after train_end (leakage guard).
        """
        # Normalise train_end
        if train_end is not None:
            cutoff = pd.Timestamp(train_end)
            future_rows = X.index[X.index > cutoff]
            if len(future_rows) > 0:
                raise ValueError(
                    f"Leakage guard: X contains {len(future_rows)} rows after "
                    f"train_end={train_end}. First offender: {future_rows[0]}"
                )
            self._train_end = cutoff
        else:
            self._train_end = None

        # Align X and y, drop NaNs
        common_idx = X.index.intersection(y.index)
        X_fit = X.loc[common_idx].copy()
        y_fit = y.loc[common_idx].copy()

        # Drop rows where all features are NaN
        valid_mask = ~X_fit.isna().all(axis=1)
        X_fit = X_fit.loc[valid_mask]
        y_fit = y_fit.loc[X_fit.index]

        if len(X_fit) == 0:
            raise ValueError("No training samples after alignment and NaN removal.")

        # Store feature names if not pre-specified
        if not self._feature_names:
            self._feature_names = list(X_fit.columns)

        # Fill remaining NaNs with 0 (median is train-time; 0 is safe fallback)
        X_fit_filled = X_fit.fillna(0.0)
        y_arr = y_fit.values

        # Fit the estimator
        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        self._estimator.fit(X_fit_filled.values, y_arr, **fit_kwargs)
        self._fitted = True

        # Calibration — MUST use held-out data only
        if self._calibrate and calibration_X is not None and calibration_y is not None:
            from ml.models.calibration import CalibratedModelWrapper
            self._calibrator = CalibratedModelWrapper(
                base_estimator=self._estimator,
                method=self._calibration_method,
            )
            calib_X_filled = calibration_X.fillna(0.0)
            self._calibrator.fit_calibration(calib_X_filled, calibration_y)
            self._calibration_applied = True

        # Per-fold IC values for robustness scoring
        # Time-series CV: 5 expanding-window folds on training data
        ic_values_per_fold: List[float] = []
        walk_forward_ic_values: List[float] = []
        try:
            n = len(X_fit_filled)
            if n >= 50:
                n_splits = 5
                fold_size = n // (n_splits + 1)
                for k in range(n_splits):
                    train_end_k = (k + 1) * fold_size
                    val_start_k = train_end_k
                    val_end_k = min(train_end_k + fold_size, n)
                    if val_end_k - val_start_k < 5:
                        continue
                    X_cv_tr = X_fit_filled.values[:train_end_k]
                    y_cv_tr = y_arr[:train_end_k]
                    X_cv_val = X_fit_filled.values[val_start_k:val_end_k]
                    y_cv_val = y_arr[val_start_k:val_end_k]
                    try:
                        import copy
                        cv_est = copy.deepcopy(self._estimator)
                        cv_est.fit(X_cv_tr, y_cv_tr)
                        fold_probas = cv_est.predict_proba(X_cv_val)
                        fold_scores = fold_probas[:, 1] if fold_probas.shape[1] > 1 else fold_probas[:, 0]
                        fold_ic, _ = stats.spearmanr(fold_scores, y_cv_val)
                        if not np.isnan(fold_ic):
                            ic_values_per_fold.append(float(fold_ic))
                    except Exception as cv_exc:
                        logger.debug("CV fold %d IC failed: %s", k, cv_exc)

                # Walk-forward IC: rolling-origin OOS (distinct from CV IC)
                # Each origin trains on all data up to that point, validates on next window.
                wf_step = max(fold_size, 20)
                wf_origins = range(fold_size * 2, n - wf_step, wf_step)
                for origin in wf_origins:
                    X_wf_tr = X_fit_filled.values[:origin]
                    y_wf_tr = y_arr[:origin]
                    X_wf_val = X_fit_filled.values[origin:origin + wf_step]
                    y_wf_val = y_arr[origin:origin + wf_step]
                    if len(y_wf_val) < 5:
                        continue
                    try:
                        import copy
                        wf_est = copy.deepcopy(self._estimator)
                        wf_est.fit(X_wf_tr, y_wf_tr)
                        wf_probas = wf_est.predict_proba(X_wf_val)
                        wf_scores = wf_probas[:, 1] if wf_probas.shape[1] > 1 else wf_probas[:, 0]
                        wf_ic, _ = stats.spearmanr(wf_scores, y_wf_val)
                        if not np.isnan(wf_ic):
                            walk_forward_ic_values.append(float(wf_ic))
                    except Exception as wf_exc:
                        logger.debug("Walk-forward origin %d IC failed: %s", origin, wf_exc)
        except Exception as ic_exc:
            logger.debug("IC computation skipped: %s", ic_exc)

        # Build training artifact
        artifact = self._build_artifact(
            X_fit=X_fit,
            y_fit=y_fit,
            train_end=train_end,
            ic_values_per_fold=ic_values_per_fold,
            walk_forward_ic_values=walk_forward_ic_values,
        )
        self._training_artifact = artifact

        logger.info(
            "MLModel[%s] fitted: %d samples, %d features, calibrated=%s",
            self._model_id[:8],
            len(X_fit),
            X_fit.shape[1],
            self._calibration_applied,
        )
        return artifact

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return calibrated (if calibrator is fitted) or raw probability predictions.

        Returns shape (n_samples, 2) for binary classification.
        Never raises — returns array of 0.5 on any failure.
        """
        if not self._fitted:
            logger.warning("predict_proba called on unfitted model %s", self._model_id[:8])
            return np.full((len(X), 2), 0.5)

        try:
            X_filled = X.fillna(0.0).values

            if self._calibration_applied and self._calibrator is not None:
                return self._calibrator.predict_proba(X_filled)

            raw = self._estimator.predict_proba(X_filled)
            return raw

        except Exception as exc:
            logger.error(
                "predict_proba failed for model %s: %s", self._model_id[:8], exc
            )
            return np.full((len(X), 2), 0.5)

    def predict_structured(
        self,
        X: pd.DataFrame,
        as_of: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
    ) -> List[PredictionRecord]:
        """
        Return a structured PredictionRecord for each row in X.

        Parameters
        ----------
        X : pd.DataFrame
        as_of : str, optional
            ISO timestamp representing the point-in-time for these predictions.
        entity_ids : list[str], optional
            Entity identifiers (pair ids, etc.) matching rows in X.

        Returns
        -------
        list[PredictionRecord]
        """
        probas = self.predict_proba(X)
        fallback_used = not self._fitted
        records = []

        for i, (idx, _) in enumerate(X.iterrows()):
            try:
                proba_positive = float(probas[i, 1]) if probas.shape[1] > 1 else float(probas[i, 0])
                ts = as_of or (str(idx) if isinstance(idx, pd.Timestamp) else "")
                eid = entity_ids[i] if entity_ids and i < len(entity_ids) else ""

                missing = [c for c in self._feature_names if c in X.columns and pd.isna(X.iloc[i][c])]
                availability = "full" if not missing else ("partial" if len(missing) < 3 else "degraded")

                record = PredictionRecord(
                    prediction_id=str(uuid.uuid4())[:8],
                    model_id=self._model_id,
                    model_version="1.0",
                    task_family=self._task_family.value,
                    entity_id=eid,
                    prediction_timestamp=ts,
                    score=proba_positive,
                    confidence=self._estimate_confidence(proba_positive),
                    calibration_applied=self._calibration_applied,
                    feature_availability=availability,
                    missing_features=missing,
                    fallback_used=fallback_used,
                    fallback_reason="model_not_fitted" if fallback_used else "",
                )
                records.append(record)

            except Exception as exc:
                logger.warning(
                    "predict_structured row %d failed: %s", i, exc
                )
                records.append(self._fallback_record(as_of or "", entity_ids[i] if entity_ids and i < len(entity_ids) else ""))

        return records

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Information Coefficient (Spearman rank correlation between predictions and labels).

        Returns NaN if fewer than 5 samples are available.
        """
        if not self._fitted:
            return float("nan")
        try:
            probas = self.predict_proba(X)
            preds = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
            common_idx = X.index.intersection(y.index)
            if len(common_idx) < 5:
                return float("nan")
            pred_series = pd.Series(preds, index=X.index).loc[common_idx]
            label_series = y.loc[common_idx]
            corr, _ = stats.spearmanr(pred_series.values, label_series.values)
            return float(corr) if not np.isnan(corr) else float("nan")
        except Exception as exc:
            logger.warning("score() failed: %s", exc)
            return float("nan")

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Any) -> str:
        """
        Persist this MLModel to disk via pickle.

        Parameters
        ----------
        path : str | Path
            Destination file path. Created if it doesn't exist.

        Returns
        -------
        str
            Absolute path written.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("MLModel[%s] saved to %s", self._model_id[:8], dest)
        return str(dest.resolve())

    @classmethod
    def load(cls, path: Any) -> "MLModel":
        """Load an MLModel artifact from disk."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected {cls.__name__}, got {type(obj).__name__}"
            )
        logger.info("MLModel[%s] loaded from %s", obj._model_id[:8], path)
        return obj

    # ── Model card ────────────────────────────────────────────────────────

    def generate_model_card(self) -> "ModelCard":
        """Generate a ModelCard describing this model."""
        artifact = self._training_artifact
        return ModelCard(
            model_id=self._model_id,
            model_class=self._model_class_name,
            task_family=self._task_family.value,
            version="1.0",
            intended_use=f"ML prediction for {self._task_family.value} tasks",
            out_of_scope_use="Do not use for direct trade execution without portfolio layer approval.",
            feature_groups_used=self._feature_names[:10],
            training_data_summary=(
                f"{artifact.n_train_samples} samples, {artifact.n_features} features"
                if artifact else "Not yet trained"
            ),
            training_period=(
                f"{artifact.train_start} to {artifact.train_end}"
                if artifact else ""
            ),
            known_limitations=[
                "Calibration is only valid within the training distribution.",
                "Regime shifts may degrade performance before retraining.",
            ],
            failure_modes=[
                "Silent degradation when feature distribution shifts.",
                "Returns 0.5 probability when unfitted or on exception.",
            ],
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _estimate_confidence(self, probability: float) -> float:
        """Confidence proxy: distance from 0.5, scaled to [0, 1]."""
        return min(1.0, abs(probability - 0.5) * 2.0)

    def _fallback_record(self, ts: str, entity_id: str) -> PredictionRecord:
        return PredictionRecord(
            prediction_id=str(uuid.uuid4())[:8],
            model_id=self._model_id,
            task_family=self._task_family.value,
            entity_id=entity_id,
            prediction_timestamp=ts,
            score=0.5,
            confidence=0.0,
            fallback_used=True,
            fallback_reason="prediction_error",
        )

    def _build_artifact(
        self,
        X_fit: pd.DataFrame,
        y_fit: pd.Series,
        train_end: Any,
        ic_values_per_fold: Optional[List[float]] = None,
        walk_forward_ic_values: Optional[List[float]] = None,
    ) -> TrainingRunArtifact:
        """Build the TrainingRunArtifact post-fit."""
        feature_hash = hashlib.sha256(
            "|".join(sorted(X_fit.columns)).encode()
        ).hexdigest()[:16]

        train_start_str = ""
        train_end_str = ""

        if isinstance(X_fit.index, pd.DatetimeIndex) and len(X_fit) > 0:
            train_start_str = X_fit.index[0].isoformat()
            if train_end is not None:
                train_end_str = pd.Timestamp(train_end).isoformat()
            else:
                train_end_str = X_fit.index[-1].isoformat()

        # Feature importances if available
        feature_importances: Dict[str, float] = {}
        top_features: List[str] = []
        if hasattr(self._estimator, "feature_importances_"):
            fi = self._estimator.feature_importances_
            cols = list(X_fit.columns)
            feature_importances = {
                c: float(v) for c, v in zip(cols, fi)
            }
            sorted_features = sorted(
                feature_importances.items(), key=lambda kv: kv[1], reverse=True
            )
            top_features = [k for k, _ in sorted_features[:10]]
        elif hasattr(self._estimator, "coef_"):
            coef = np.abs(self._estimator.coef_).flatten()
            cols = list(X_fit.columns)
            feature_importances = {c: float(v) for c, v in zip(cols, coef)}
            sorted_features = sorted(
                feature_importances.items(), key=lambda kv: kv[1], reverse=True
            )
            top_features = [k for k, _ in sorted_features[:10]]

        # Aggregate IC metrics from fold lists
        ic_per_fold: List[float] = ic_values_per_fold or []
        wf_ic_values: List[float] = walk_forward_ic_values or []

        cv_ic_mean = float(np.mean(ic_per_fold)) if ic_per_fold else float("nan")
        wf_ic_mean = (
            float(np.mean([v for v in wf_ic_values if v is not None and v == v]))
            if wf_ic_values else float("nan")
        )

        artifact = TrainingRunArtifact(
            run_id=str(uuid.uuid4())[:8],
            model_id=self._model_id,
            task_family=self._task_family.value,
            feature_hash=feature_hash,
            train_start=train_start_str,
            train_end=train_end_str,
            n_train_samples=len(X_fit),
            n_features=X_fit.shape[1],
            calibration_applied=self._calibration_applied,
            top_features=top_features,
            feature_importances=feature_importances,
            cv_ic_mean=cv_ic_mean,
        )

        # Per-fold IC values for robustness scoring
        # Required by governance engine to compute real robustness_score
        artifact.cv_ic_per_fold = ic_per_fold  # list[float] from CV

        # Walk-forward IC: rolling-origin OOS (distinct from CV IC)
        # This is a genuine sequential test simulating production deployment order.
        if wf_ic_values:
            artifact.walk_forward_ic_mean = wf_ic_mean

        return artifact
