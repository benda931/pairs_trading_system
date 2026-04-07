# -*- coding: utf-8 -*-
"""
ml/evaluation/metrics.py — Core Evaluation Metrics
====================================================

Standalone metric functions for evaluating stat-arb ML models.

Covers:
- Information Coefficient (IC) and its t-statistic
- Classification metrics: AUC-ROC, PR-AUC, Brier score, ECE
- Trading-specific: meta-label utility, walk-forward IC, regime-sliced metrics
- Robustness scoring
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd

# sklearn imports are deferred where needed so the module loads even if
# sklearn is not installed (tests can still import the utilities).
try:
    from scipy.stats import spearmanr as _spearmanr
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import (
        roc_auc_score as _roc_auc_score,
        average_precision_score as _avg_precision_score,
        brier_score_loss as _brier_score_loss,
        log_loss as _log_loss,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------

def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Spearman rank correlation between predictions and outcomes.

    Returns NaN if fewer than 3 overlapping observations exist.
    """
    if len(y_true) < 3 or len(y_pred) < 3:
        return float("nan")

    # Align on index
    common = y_true.index.intersection(y_pred.index)
    if len(common) < 3:
        return float("nan")

    yt = y_true.loc[common].values.astype(float)
    yp = y_pred.loc[common].values.astype(float)

    # Remove NaN pairs
    mask = np.isfinite(yt) & np.isfinite(yp)
    if mask.sum() < 3:
        return float("nan")

    if _SCIPY_AVAILABLE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = _spearmanr(yt[mask], yp[mask])
        return float(corr) if np.isfinite(corr) else float("nan")

    # Fallback: numpy rank correlation
    def _rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1, dtype=float)
        return ranks

    r_true = _rank(yt[mask])
    r_pred = _rank(yp[mask])
    n = len(r_true)
    d2 = np.sum((r_true - r_pred) ** 2)
    corr = 1.0 - 6.0 * d2 / (n * (n * n - 1))
    return float(corr)


def ic_t_stat(ic: float, n: int) -> float:
    """
    T-statistic for IC significance test.

    Formula: IC * sqrt(n) / sqrt(1 - IC^2)

    Returns NaN if |IC| >= 1 or n < 2.
    """
    if not np.isfinite(ic) or n < 2:
        return float("nan")
    if abs(ic) >= 1.0:
        return float("nan")
    denom = math.sqrt(1.0 - ic * ic)
    if denom < 1e-12:
        return float("nan")
    return ic * math.sqrt(n) / denom


def ic_with_tstat(
    y_true: "pd.Series",
    y_pred: "pd.Series",
) -> tuple:
    """
    Compute Information Coefficient with t-statistic and sample size.

    An IC without a t-statistic is uninterpretable — IC=0.065 on 20 samples
    (t=0.82, p=0.42) is noise; the same IC on 200 samples (t=2.58, p=0.01)
    is signal. Both numbers MUST always be reported together.

    Parameters
    ----------
    y_true : pd.Series
        Realized outcomes.
    y_pred : pd.Series
        Model predictions / scores.

    Returns
    -------
    tuple : (ic, t_stat, n_samples)
        ic        : Spearman rank correlation (IC).
        t_stat    : t-statistic for the IC (IC / SE, where SE = sqrt((1-IC²)/(N-2))).
        n_samples : Number of valid (non-NaN) pairs used.

    Notes
    -----
    Returns (nan, nan, 0) on any error or if N < 4.
    The t_stat follows a t-distribution with N-2 degrees of freedom under H0: IC=0.
    Minimum meaningful sample: N >= 30 for asymptotic normality.
    """
    try:
        # Align and drop NaN pairs
        combined = pd.DataFrame({"y": y_true, "p": y_pred}).dropna()
        n = len(combined)
        if n < 4:
            return float("nan"), float("nan"), 0

        if _SCIPY_AVAILABLE:
            ic_val, _ = _spearmanr(combined["y"], combined["p"])
        else:
            # Fallback: numpy rank correlation
            def _rank(x: np.ndarray) -> np.ndarray:
                order = np.argsort(x)
                ranks = np.empty_like(order, dtype=float)
                ranks[order] = np.arange(1, len(x) + 1, dtype=float)
                return ranks

            arr_y = combined["y"].values.astype(float)
            arr_p = combined["p"].values.astype(float)
            r_true = _rank(arr_y)
            r_pred = _rank(arr_p)
            d2 = np.sum((r_true - r_pred) ** 2)
            ic_val = 1.0 - 6.0 * d2 / (n * (n * n - 1))

        ic_val = float(ic_val)

        if not (-1.0 <= ic_val <= 1.0):
            return float("nan"), float("nan"), n

        # t-statistic: IC * sqrt(N-2) / sqrt(1 - IC²)
        denom = max(1.0 - ic_val ** 2, 1e-12)
        t_stat = ic_val * ((n - 2) ** 0.5) / (denom ** 0.5)

        return ic_val, float(t_stat), n

    except Exception as exc:
        import logging
        logging.getLogger("ml.evaluation.metrics").debug("ic_with_tstat failed: %s", exc)
        return float("nan"), float("nan"), 0


def ic_is_significant(ic: float, n: int, alpha: float = 0.05) -> bool:
    """
    Return True if the IC is statistically significant at the given level.

    Uses the t-distribution with N-2 degrees of freedom.
    Minimum recommended N: 30.

    Parameters
    ----------
    ic : float
        Information Coefficient value.
    n : int
        Number of samples used to compute IC.
    alpha : float
        Significance level (default 0.05, two-tailed).

    Returns
    -------
    bool : True if |t_stat| >= critical value at alpha level.
    """
    if not math.isfinite(ic) or n < 4:
        return False
    try:
        from scipy.stats import t as t_dist
        denom = max(1.0 - ic ** 2, 1e-12)
        t_stat = ic * ((n - 2) ** 0.5) / (denom ** 0.5)
        # Two-tailed critical value
        critical = t_dist.ppf(1 - alpha / 2, df=n - 2)
        return abs(t_stat) >= critical
    except Exception:
        # Approximate: |IC| * sqrt(N) > 1.96 for alpha≈0.05 (two-tailed)
        return abs(ic) * (n ** 0.5) >= 1.96


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------

def auc_roc(y_true, y_pred_proba) -> float:
    """
    ROC AUC from sklearn.

    Returns NaN if only one class is present or sklearn is unavailable.
    """
    if not _SKLEARN_AVAILABLE:
        return float("nan")

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred_proba, dtype=float)

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    if len(yt) < 2:
        return float("nan")
    if len(np.unique(yt)) < 2:
        return float("nan")

    try:
        return float(_roc_auc_score(yt, yp))
    except Exception:
        return float("nan")


def pr_auc(y_true, y_pred_proba) -> float:
    """
    Precision-Recall AUC (average precision score).

    Returns NaN if only one class is present or sklearn is unavailable.
    """
    if not _SKLEARN_AVAILABLE:
        return float("nan")

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred_proba, dtype=float)

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    if len(yt) < 2:
        return float("nan")
    if len(np.unique(yt)) < 2:
        return float("nan")

    try:
        return float(_avg_precision_score(yt, yp))
    except Exception:
        return float("nan")


def brier_score(y_true, y_pred_proba) -> float:
    """
    Mean squared error of probability predictions (Brier score).

    Lower is better; perfect calibration → 0.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred_proba, dtype=float)

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    if len(yt) == 0:
        return float("nan")

    return float(np.mean((yt - yp) ** 2))


def calibration_error(y_true, y_pred_proba, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    Partitions predictions into n_bins equal-width bins,
    then computes the weighted absolute difference between
    mean predicted probability and observed frequency per bin.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred_proba, dtype=float)

    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]

    n = len(yt)
    if n == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Include right edge only for the last bin
        if hi == 1.0:
            in_bin = (yp >= lo) & (yp <= hi)
        else:
            in_bin = (yp >= lo) & (yp < hi)

        n_bin = in_bin.sum()
        if n_bin == 0:
            continue

        mean_pred = yp[in_bin].mean()
        mean_actual = yt[in_bin].mean()
        ece += (n_bin / n) * abs(mean_pred - mean_actual)

    return float(ece)


# ---------------------------------------------------------------------------
# Meta-Label Utility
# ---------------------------------------------------------------------------

def meta_label_utility(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_signal_precision: float,
) -> dict:
    """
    Compute meta-label utility metrics.

    Parameters
    ----------
    y_true : binary array indicating true reversion success (1) or failure (0)
    y_pred : binary array of model decisions (1 = TAKE, 0 = SKIP)
    raw_signal_precision : precision of the unfiltered signal (baseline)

    Returns
    -------
    dict with keys:
      - filter_precision  : P(success | model says TAKE)
      - filter_recall     : P(model says TAKE | success)
      - filter_improvement: filter_precision - raw_signal_precision
      - coverage          : fraction of signals taken
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    n = len(yt)
    if n == 0:
        return {
            "filter_precision": float("nan"),
            "filter_recall": float("nan"),
            "filter_improvement": float("nan"),
            "coverage": float("nan"),
        }

    taken = yp == 1
    success = yt == 1

    n_taken = taken.sum()
    n_success = success.sum()

    if n_taken == 0:
        filter_precision = float("nan")
    else:
        filter_precision = float((yt[taken] == 1).sum() / n_taken)

    if n_success == 0:
        filter_recall = float("nan")
    else:
        filter_recall = float((yp[success] == 1).sum() / n_success)

    filter_improvement = (
        filter_precision - raw_signal_precision
        if np.isfinite(filter_precision)
        else float("nan")
    )

    coverage = float(n_taken / n)

    return {
        "filter_precision": filter_precision,
        "filter_recall": filter_recall,
        "filter_improvement": filter_improvement,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Walk-Forward IC
# ---------------------------------------------------------------------------

def walk_forward_ic(
    model: Any,
    datasets: list,
) -> dict:
    """
    Compute IC across walk-forward folds.

    Parameters
    ----------
    model : fitted MLModel with predict_proba or predict method
    datasets : list of (X_train, X_test, y_train, y_test) tuples

    Returns
    -------
    dict: {ic_mean, ic_std, ic_positive_rate, ic_values: list}
    """
    ic_values: list[float] = []

    for fold_idx, fold in enumerate(datasets):
        if len(fold) < 4:
            continue

        X_train, X_test, y_train, y_test = fold[0], fold[1], fold[2], fold[3]

        try:
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)
                # Handle multi-output: take positive class column
                if hasattr(y_pred_proba, "ndim") and y_pred_proba.ndim == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                y_pred_series = pd.Series(
                    y_pred_proba,
                    index=X_test.index if hasattr(X_test, "index") else range(len(y_pred_proba)),
                )
            else:
                y_pred_arr = model.predict(X_test)
                y_pred_series = pd.Series(
                    y_pred_arr,
                    index=X_test.index if hasattr(X_test, "index") else range(len(y_pred_arr)),
                )

            if not isinstance(y_test, pd.Series):
                y_test = pd.Series(
                    y_test,
                    index=X_test.index if hasattr(X_test, "index") else range(len(y_test)),
                )

            ic = information_coefficient(y_test, y_pred_series)
            ic_values.append(ic)
        except Exception:
            # Folds that fail contribute NaN
            ic_values.append(float("nan"))

    finite_ic = [v for v in ic_values if np.isfinite(v)]

    if not finite_ic:
        return {
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "ic_positive_rate": float("nan"),
            "ic_values": ic_values,
        }

    ic_arr = np.array(finite_ic)
    return {
        "ic_mean": float(np.mean(ic_arr)),
        "ic_std": float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else 0.0,
        "ic_positive_rate": float(np.mean(ic_arr > 0)),
        "ic_values": ic_values,
    }


# ---------------------------------------------------------------------------
# Regime-Sliced Metrics
# ---------------------------------------------------------------------------

def regime_sliced_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    regime_labels: np.ndarray,
) -> dict:
    """
    Compute AUC/Brier/IC per regime label.

    Returns
    -------
    dict: {regime_name: {auc, brier, n_samples}}
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred_proba, dtype=float)
    rl = np.asarray(regime_labels)

    result: dict = {}

    for regime in np.unique(rl):
        mask = rl == regime
        n_regime = mask.sum()

        if n_regime < 5:
            result[str(regime)] = {
                "auc": float("nan"),
                "brier": float("nan"),
                "n_samples": int(n_regime),
            }
            continue

        yt_r = yt[mask]
        yp_r = yp[mask]

        regime_auc = auc_roc(yt_r, yp_r)
        regime_brier = brier_score(yt_r, yp_r)

        result[str(regime)] = {
            "auc": regime_auc,
            "brier": regime_brier,
            "n_samples": int(n_regime),
        }

    return result


# ---------------------------------------------------------------------------
# Robustness Score
# ---------------------------------------------------------------------------

def robustness_score(ic_values: list) -> float:
    """
    Robustness = fraction of periods with positive IC.

    A model is considered robust if it performs above the naive baseline
    in more than 60% of periods.

    Returns NaN if no finite IC values are provided.
    """
    finite = [v for v in ic_values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(np.array(finite) > 0))
