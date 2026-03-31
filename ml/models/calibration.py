# -*- coding: utf-8 -*-
"""
ml/models/calibration.py — Calibration Wrapper and Utilities
=============================================================

Provides:
  CalibratedModelWrapper  — wraps an sklearn estimator with Platt or isotonic calibration
  compute_calibration_metrics — reliability curve, Brier score, ECE
  brier_score             — standalone Brier scorer

Key constraint:
  Calibration MUST always use a held-out calibration set.
  Never fit calibration on the same data used to train the base model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATED MODEL WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class CalibratedModelWrapper:
    """
    Wraps an already-fitted sklearn estimator with probability calibration.

    Calibration must use a HELD-OUT calibration set (never the training set).
    Supports:
      - "isotonic"  — isotonic regression (non-parametric, slower but more flexible)
      - "sigmoid"   — Platt scaling (parametric, faster, works well with SVM-style scores)

    Parameters
    ----------
    base_estimator : sklearn estimator
        Already fitted estimator that exposes predict_proba().
    method : str
        "isotonic" (default) or "sigmoid".
    """

    def __init__(
        self,
        base_estimator: Any,
        method: str = "isotonic",
    ) -> None:
        self._base = base_estimator
        self._method = method
        self._calibrator: Any = None
        self._is_calibrated = False

    def fit_calibration(
        self,
        X_calib: Any,
        y_calib: Any,
    ) -> "CalibratedModelWrapper":
        """
        Fit the calibration layer using held-out data.

        Parameters
        ----------
        X_calib : array-like or pd.DataFrame
            Features from the held-out calibration set.
        y_calib : array-like or pd.Series
            Labels from the held-out calibration set.

        Returns
        -------
        self
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import clone

        # Get raw scores from the base estimator
        if isinstance(X_calib, pd.DataFrame):
            X_arr = X_calib.fillna(0.0).values
        else:
            X_arr = np.asarray(X_calib)
            X_arr = np.where(np.isnan(X_arr), 0.0, X_arr)

        if isinstance(y_calib, pd.Series):
            y_arr = y_calib.values
        else:
            y_arr = np.asarray(y_calib)

        # Use sklearn's calibration infrastructure on held-out scores
        # We do manual isotonic/sigmoid regression on (score, label) pairs
        # to avoid refitting the base estimator
        try:
            raw_proba = self._base.predict_proba(X_arr)
            scores = raw_proba[:, 1] if raw_proba.shape[1] > 1 else raw_proba[:, 0]
        except Exception as exc:
            logger.error("Could not get base probabilities for calibration: %s", exc)
            self._is_calibrated = False
            return self

        if self._method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(scores, y_arr)
            self._calibrator = ir
        elif self._method in ("sigmoid", "platt"):
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1.0, max_iter=1000)
            lr.fit(scores.reshape(-1, 1), y_arr)
            self._calibrator = lr
        else:
            raise ValueError(f"Unknown calibration method: {self._method!r}. Use 'isotonic' or 'sigmoid'.")

        self._is_calibrated = True
        logger.debug(
            "Calibration fitted (%s) on %d samples.", self._method, len(y_arr)
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Return calibrated probability predictions.

        If calibration has not been fitted, returns raw base probabilities.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X.fillna(0.0).values
        else:
            X_arr = np.asarray(X)
            X_arr = np.where(np.isnan(X_arr), 0.0, X_arr)

        try:
            raw_proba = self._base.predict_proba(X_arr)
            scores = raw_proba[:, 1] if raw_proba.shape[1] > 1 else raw_proba[:, 0]
        except Exception as exc:
            logger.error("Base predict_proba failed: %s", exc)
            return np.full((len(X_arr), 2), 0.5)

        if not self._is_calibrated or self._calibrator is None:
            return raw_proba

        try:
            if self._method == "isotonic":
                cal_pos = self._calibrator.predict(scores)
            else:  # sigmoid/Platt
                cal_pos = self._calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]

            # Clip to valid probability range
            cal_pos = np.clip(cal_pos, 1e-6, 1 - 1e-6)
            return np.column_stack([1.0 - cal_pos, cal_pos])

        except Exception as exc:
            logger.warning("Calibration transform failed, using raw proba: %s", exc)
            return raw_proba

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute calibration diagnostics.

    Computes:
    - Brier score
    - Expected calibration error (ECE)
    - Reliability curve data (bucket mean predicted vs mean actual)

    Parameters
    ----------
    y_true : np.ndarray, shape (n,)
        Binary true labels.
    y_pred_proba : np.ndarray, shape (n,) or (n, 2)
        Predicted probabilities. If 2D, column 1 is used.
    n_bins : int
        Number of equal-width probability bins for reliability diagram.

    Returns
    -------
    dict with keys:
      brier_score, ece, bucket_edges, bucket_mean_predicted,
      bucket_mean_actual, bucket_counts
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred_proba)
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 1]

    # Brier score
    bs = brier_score(y_true, y_pred)

    # Reliability curve
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_edges: List[float] = []
    bucket_mean_predicted: List[float] = []
    bucket_mean_actual: List[float] = []
    bucket_counts: List[int] = []
    ece_sum = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi) if i < n_bins - 1 else (y_pred >= lo) & (y_pred <= hi)
        n_bin = int(mask.sum())
        bucket_counts.append(n_bin)
        bucket_edges.append(float((lo + hi) / 2.0))

        if n_bin == 0:
            bucket_mean_predicted.append(float("nan"))
            bucket_mean_actual.append(float("nan"))
        else:
            mean_pred = float(y_pred[mask].mean())
            mean_act = float(y_true[mask].mean())
            bucket_mean_predicted.append(mean_pred)
            bucket_mean_actual.append(mean_act)
            ece_sum += (n_bin / n_total) * abs(mean_pred - mean_act)

    return {
        "brier_score": float(bs),
        "ece": float(ece_sum),
        "bucket_edges": bucket_edges,
        "bucket_mean_predicted": bucket_mean_predicted,
        "bucket_mean_actual": bucket_mean_actual,
        "bucket_counts": bucket_counts,
        "n_samples": int(n_total),
    }


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Brier score = mean squared error of probability predictions.

    Lower is better. 0.25 is the score of always predicting 0.5 for balanced classes.

    Parameters
    ----------
    y_true : np.ndarray
        Binary true labels {0, 1}.
    y_pred : np.ndarray
        Predicted probabilities in [0, 1].

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_pred - y_true) ** 2))
