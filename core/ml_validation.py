# -*- coding: utf-8 -*-
"""
core/ml_validation.py — Financial ML Cross-Validation & Drift Detection
========================================================================

Ported from srv_quant_system (ml_cross_validation.py + ml_adaptive.py).

Features:
- PurgedKFold: sklearn-compatible CV splitter preventing data leakage
- Embargo gap to prevent autocorrelation leakage
- Financial scorers (Sharpe, IC, directional accuracy)
- DriftDetector: monitors model performance for concept drift
- PSI (Population Stability Index) for feature distribution drift
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("core.ml_validation")


# ══════════════════════════════════════════════════════════════════
# Purged K-Fold Cross-Validation (de Prado, 2018)
# ══════════════════════════════════════════════════════════════════

class PurgedKFold:
    """Purged K-Fold CV with embargo gap for financial time series.

    Prevents information leakage by:
    1. Chronological splits (no random shuffling)
    2. Purging: removing training samples that overlap with test
    3. Embargo: adding a gap after each test fold to prevent autocorrelation

    Compatible with sklearn's cross-validation interface.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n = len(X)
        embargo = int(n * self.embargo_pct)
        fold_size = n // self.n_splits

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)

            # training: everything before test_start and after test_end + embargo
            train_before = list(range(0, max(0, test_start - embargo)))
            train_after = list(range(min(n, test_end + embargo), n))
            train_idx = np.array(train_before + train_after)
            test_idx = np.array(range(test_start, test_end))

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def cross_val_score_purged(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    scoring: str = "sharpe",
) -> list[float]:
    """Cross-validation with purged K-fold and financial scorers."""
    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    scores = []

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = _compute_score(y_test, y_pred, scoring)
        scores.append(score)

    return scores


def _compute_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute financial scoring metric."""
    if metric == "sharpe":
        returns = y_true * np.sign(y_pred)
        if returns.std() < 1e-10:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    elif metric == "ic":
        # Information Coefficient (Spearman rank correlation)
        from scipy.stats import spearmanr
        corr, _ = spearmanr(y_true, y_pred)
        return float(corr) if np.isfinite(corr) else 0.0

    elif metric == "accuracy":
        correct = (np.sign(y_true) == np.sign(y_pred)).mean()
        return float(correct)

    elif metric == "mse":
        return -float(np.mean((y_true - y_pred) ** 2))

    return 0.0


# ══════════════════════════════════════════════════════════════════
# Drift Detection
# ══════════════════════════════════════════════════════════════════

@dataclass
class DriftReport:
    """Drift detection report."""
    is_drifting: bool = False
    ic_degraded: bool = False
    hit_rate_collapsed: bool = False
    feature_drift: bool = False
    psi_values: dict = field(default_factory=dict)
    rolling_ic: float = np.nan
    rolling_hit_rate: float = np.nan
    recommendation: str = "CONTINUE"  # CONTINUE, RETRAIN, HALT


class DriftDetector:
    """Monitors model performance and detects concept drift.

    Three-pronged detection:
    1. IC degradation (rolling IC < 0 for N consecutive observations)
    2. Hit-rate collapse (< 50% for N consecutive days)
    3. Feature distribution drift via PSI
    """

    def __init__(
        self,
        ic_window: int = 20,
        ic_threshold: float = 0.0,
        ic_consecutive: int = 5,
        hit_rate_window: int = 20,
        hit_rate_threshold: float = 0.50,
        hit_rate_consecutive: int = 10,
        psi_threshold: float = 0.20,
    ):
        self.ic_window = ic_window
        self.ic_threshold = ic_threshold
        self.ic_consecutive = ic_consecutive
        self.hit_rate_window = hit_rate_window
        self.hit_rate_threshold = hit_rate_threshold
        self.hit_rate_consecutive = hit_rate_consecutive
        self.psi_threshold = psi_threshold

        self._ic_history: list[float] = []
        self._hit_history: list[float] = []

    def update(self, y_true: float, y_pred: float) -> None:
        """Update detector with new observation."""
        # IC approximation (running correlation proxy)
        self._ic_history.append(y_true * np.sign(y_pred))

        # hit rate
        correct = float(np.sign(y_true) == np.sign(y_pred))
        self._hit_history.append(correct)

    def check(
        self,
        *,
        feature_train: np.ndarray | None = None,
        feature_current: np.ndarray | None = None,
    ) -> DriftReport:
        """Check for drift."""
        report = DriftReport()

        # IC check
        if len(self._ic_history) >= self.ic_window:
            recent_ic = np.mean(self._ic_history[-self.ic_window:])
            report.rolling_ic = recent_ic

            # consecutive negative IC
            recent = self._ic_history[-self.ic_consecutive:]
            if len(recent) >= self.ic_consecutive and all(v < self.ic_threshold for v in recent):
                report.ic_degraded = True

        # hit rate check
        if len(self._hit_history) >= self.hit_rate_window:
            recent_hr = np.mean(self._hit_history[-self.hit_rate_window:])
            report.rolling_hit_rate = recent_hr

            recent = self._hit_history[-self.hit_rate_consecutive:]
            if len(recent) >= self.hit_rate_consecutive:
                rolling_hr = np.mean(recent)
                if rolling_hr < self.hit_rate_threshold:
                    report.hit_rate_collapsed = True

        # PSI check
        if feature_train is not None and feature_current is not None:
            psi_vals = check_multi_feature_drift(feature_train, feature_current)
            report.psi_values = psi_vals
            if any(v >= self.psi_threshold for v in psi_vals.values()):
                report.feature_drift = True

        # overall decision
        report.is_drifting = report.ic_degraded or report.hit_rate_collapsed or report.feature_drift

        if report.ic_degraded and report.hit_rate_collapsed:
            report.recommendation = "HALT"
        elif report.is_drifting:
            report.recommendation = "RETRAIN"
        else:
            report.recommendation = "CONTINUE"

        return report

    def reset(self) -> None:
        self._ic_history.clear()
        self._hit_history.clear()


# ── PSI (Population Stability Index) ─────────────────────────────

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Population Stability Index for feature distribution comparison.

    PSI interpretation:
    - < 0.10: stable
    - 0.10-0.20: moderate drift
    - >= 0.20: significant drift
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # remove nans
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    # create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return 0.0

    # bin counts
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # convert to proportions with small epsilon
    eps = 1e-4
    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    # PSI formula
    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return max(0.0, psi)


def check_multi_feature_drift(
    train_features: np.ndarray,
    current_features: np.ndarray,
) -> dict[str, float]:
    """Check PSI across multiple features."""
    if train_features.ndim == 1:
        train_features = train_features.reshape(-1, 1)
    if current_features.ndim == 1:
        current_features = current_features.reshape(-1, 1)

    n_features = min(train_features.shape[1], current_features.shape[1])
    results = {}

    for i in range(n_features):
        psi = compute_psi(train_features[:, i], current_features[:, i])
        results[f"feature_{i}"] = psi

    return results
