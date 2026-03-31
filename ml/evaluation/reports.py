# -*- coding: utf-8 -*-
"""
ml/evaluation/reports.py — Model Evaluation Report Generator
=============================================================

Generates structured EvaluationReport and CalibrationReport objects
from trained ML models and labelled datasets.

All methods are fallback-safe: exceptions are caught and logged; NaN is
returned for any metric that cannot be computed rather than crashing the
evaluation pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ml.contracts import (
    CalibrationReport,
    EvaluationReport,
)
from ml.evaluation.metrics import (
    auc_roc,
    brier_score,
    calibration_error,
    ic_t_stat,
    information_coefficient,
    meta_label_utility,
    pr_auc,
    regime_sliced_metrics,
    robustness_score,
    walk_forward_ic,
)

try:
    from sklearn.metrics import (
        accuracy_score as _accuracy_score,
        f1_score as _f1_score,
        log_loss as _log_loss,
        precision_score as _precision_score,
        recall_score as _recall_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

logger = logging.getLogger("ml.evaluation.reports")


def _safe(fn: Callable, *args, **kwargs) -> float:
    """Call fn(*args, **kwargs) and return NaN on any exception."""
    try:
        result = fn(*args, **kwargs)
        return float(result) if result is not None else float("nan")
    except Exception as exc:
        logger.debug("Metric computation failed: %s", exc)
        return float("nan")


def _predict_proba_1d(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Call model.predict_proba(X) and return the positive-class column as 1-D.

    Falls back to model.predict(X) if predict_proba is not available.
    """
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(X)
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 2:
            return arr[:, 1]
        return arr
    raw = model.predict(X)
    return np.asarray(raw, dtype=float)


class ModelEvaluator:
    """
    Generates structured evaluation reports for ML models.

    Usage
    -----
    evaluator = ModelEvaluator()
    report = evaluator.evaluate(model, X_train, y_train, X_test, y_test)
    """

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        regime_labels: Optional[pd.Series] = None,
        raw_signal_precision: Optional[float] = None,
    ) -> EvaluationReport:
        """
        Full evaluation: AUC, Brier, IC, regime-sliced metrics,
        meta-label utility.

        Parameters
        ----------
        model : fitted model with predict / predict_proba
        X_train, y_train : training set (for in-sample diagnostics)
        X_test, y_test : held-out test set (primary evaluation)
        X_val, y_val : optional validation set
        regime_labels : regime label per test observation (index-aligned to y_test)
        raw_signal_precision : baseline precision of the unfiltered signal
        """
        report = EvaluationReport()

        # Identify model ID if available
        if hasattr(model, "metadata") and hasattr(model.metadata, "model_id"):
            report.model_id = str(model.metadata.model_id)
        elif hasattr(model, "model_id"):
            report.model_id = str(model.model_id)

        report.n_samples = len(y_test)

        # -- Predicted probabilities on test set --
        try:
            y_proba_test = _predict_proba_1d(model, X_test)
        except Exception as exc:
            logger.warning("predict_proba failed for test set: %s", exc)
            return report

        yt = np.asarray(y_test, dtype=float)

        # -- Core classification metrics --
        report.auc = _safe(auc_roc, yt, y_proba_test)
        report.pr_auc = _safe(pr_auc, yt, y_proba_test)
        report.brier_score = _safe(brier_score, yt, y_proba_test)

        if _SKLEARN_AVAILABLE:
            mask = np.isfinite(yt) & np.isfinite(y_proba_test)
            if mask.sum() >= 2:
                yt_m = yt[mask]
                yp_m = y_proba_test[mask]
                if len(np.unique(yt_m)) >= 2:
                    report.log_loss = _safe(_log_loss, yt_m, yp_m)

            y_pred_binary = (y_proba_test >= 0.5).astype(int)
            report.accuracy = _safe(_accuracy_score, yt, y_pred_binary)
            report.precision = _safe(_precision_score, yt, y_pred_binary, zero_division=0)
            report.recall = _safe(_recall_score, yt, y_pred_binary, zero_division=0)
            report.f1 = _safe(_f1_score, yt, y_pred_binary, zero_division=0)

        # -- IC --
        y_test_series = (
            y_test if isinstance(y_test, pd.Series)
            else pd.Series(yt, index=X_test.index)
        )
        y_proba_series = pd.Series(y_proba_test, index=X_test.index)
        ic = information_coefficient(y_test_series, y_proba_series)
        report.information_coefficient = ic
        report.ic_t_stat = ic_t_stat(ic, len(yt))

        # -- Regime-sliced metrics --
        if regime_labels is not None:
            try:
                rl = np.asarray(regime_labels.reindex(y_test.index).values
                                if isinstance(regime_labels, pd.Series)
                                else regime_labels,
                                dtype=str)
                regime_met = regime_sliced_metrics(yt, y_proba_test, rl)
                report.regime_metrics = regime_met
            except Exception as exc:
                logger.warning("Regime-sliced metrics failed: %s", exc)

        # -- Meta-label utility --
        if raw_signal_precision is not None:
            try:
                y_pred_binary = (y_proba_test >= 0.5).astype(int)
                utility = meta_label_utility(yt, y_pred_binary, raw_signal_precision)
                report.meta_label_precision = utility.get("filter_precision", float("nan"))
                report.baseline_precision = raw_signal_precision
                report.filter_improvement = utility.get("filter_improvement", float("nan"))
            except Exception as exc:
                logger.warning("Meta-label utility computation failed: %s", exc)

        # -- Class balance --
        unique, counts = np.unique(yt[np.isfinite(yt)], return_counts=True)
        report.class_balance = {str(k): float(v / len(yt)) for k, v in zip(unique, counts)}

        return report

    # ------------------------------------------------------------------
    # Calibration diagnostics
    # ------------------------------------------------------------------

    def evaluate_calibration(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        split: str = "test",
    ) -> CalibrationReport:
        """Generate calibration diagnostics."""
        report = CalibrationReport(split=split)

        if hasattr(model, "metadata") and hasattr(model.metadata, "model_id"):
            report.model_id = str(model.metadata.model_id)

        try:
            y_proba = _predict_proba_1d(model, X)
        except Exception as exc:
            logger.warning("predict_proba failed in calibration eval: %s", exc)
            return report

        yt = np.asarray(y, dtype=float)
        report.n_samples = len(yt)
        report.brier_score_uncalibrated = _safe(brier_score, yt, y_proba)

        n_bins = 10
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bucket_edges: list = []
        bucket_mean_pred: list = []
        bucket_mean_actual: list = []
        bucket_counts: list = []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            in_bin = (y_proba >= lo) & (y_proba < hi) if hi < 1.0 else (y_proba >= lo) & (y_proba <= hi)
            n_bin = in_bin.sum()
            bucket_edges.append(float(lo))
            bucket_counts.append(int(n_bin))
            if n_bin > 0:
                bucket_mean_pred.append(float(y_proba[in_bin].mean()))
                bucket_mean_actual.append(float(yt[in_bin].mean()))
            else:
                bucket_mean_pred.append(float("nan"))
                bucket_mean_actual.append(float("nan"))

        report.bucket_edges = bucket_edges
        report.bucket_mean_predicted = bucket_mean_pred
        report.bucket_mean_actual = bucket_mean_actual
        report.bucket_counts = bucket_counts

        # Calibrated Brier: approximate by isotonic regression if available
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(out_of_bounds="clip")
            valid = np.isfinite(yt) & np.isfinite(y_proba)
            if valid.sum() >= 10:
                y_cal = ir.fit_transform(y_proba[valid], yt[valid])
                report.brier_score_calibrated = _safe(brier_score, yt[valid], y_cal)
                improvement = report.brier_score_uncalibrated - report.brier_score_calibrated
                report.brier_improvement = float(improvement) if np.isfinite(improvement) else float("nan")
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("Isotonic calibration failed: %s", exc)

        return report

    # ------------------------------------------------------------------
    # Baseline comparison
    # ------------------------------------------------------------------

    def compare_to_baseline(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        baseline_score: float = 0.5,
    ) -> dict:
        """
        Compare model to naive baselines.

        Baselines:
        - always-positive classifier (predicts 1 for every sample)
        - random classifier (AUC = 0.5)
        - raw signal (no filter), represented by baseline_score

        Returns improvement summary.
        """
        result: dict = {
            "model_auc": float("nan"),
            "always_positive_precision": float("nan"),
            "random_auc": 0.5,
            "baseline_score": baseline_score,
            "lift_over_random": float("nan"),
            "lift_over_always_positive": float("nan"),
            "lift_over_raw_signal": float("nan"),
        }

        yt = np.asarray(y_test, dtype=float)
        n = len(yt)
        if n == 0:
            return result

        try:
            y_proba = _predict_proba_1d(model, X_test)
        except Exception as exc:
            logger.warning("Baseline comparison predict_proba failed: %s", exc)
            return result

        model_auc = auc_roc(yt, y_proba)
        result["model_auc"] = model_auc

        # Always-positive: just classifies everything as positive
        n_pos = (yt == 1).sum()
        always_pos_precision = float(n_pos / n) if n > 0 else float("nan")
        result["always_positive_precision"] = always_pos_precision

        if np.isfinite(model_auc):
            result["lift_over_random"] = float(model_auc - 0.5)
            if np.isfinite(always_pos_precision):
                model_prec = brier_score(yt, y_proba)
                # Use model AUC vs always-positive precision as proxy
                result["lift_over_always_positive"] = float(model_auc - always_pos_precision)
            result["lift_over_raw_signal"] = float(model_auc - baseline_score)

        return result

    # ------------------------------------------------------------------
    # Walk-forward evaluation
    # ------------------------------------------------------------------

    def walk_forward_evaluation(
        self,
        model_factory: Callable,
        datasets: list,
    ) -> dict:
        """
        Re-train and evaluate on each walk-forward fold.

        Parameters
        ----------
        model_factory : callable() -> MLModel (fresh unfitted model)
        datasets : list of (X_train, X_test, y_train, y_test) tuples

        Returns
        -------
        dict: {ic_values, ic_mean, ic_std, robustness_score, fold_reports}
        """
        ic_values: list[float] = []
        fold_reports: list[dict] = []
        trained_models: list = []

        for fold_idx, fold in enumerate(datasets):
            if len(fold) < 4:
                ic_values.append(float("nan"))
                continue

            X_train, X_test, y_train, y_test = fold[0], fold[1], fold[2], fold[3]
            fold_info: dict = {"fold": fold_idx, "n_train": len(y_train), "n_test": len(y_test)}

            try:
                m = model_factory()
                m.fit(X_train, y_train)
                trained_models.append(m)

                y_proba = _predict_proba_1d(m, X_test)
                yt = np.asarray(y_test, dtype=float)

                fold_auc = auc_roc(yt, y_proba)
                fold_brier = brier_score(yt, y_proba)

                y_proba_series = pd.Series(y_proba, index=X_test.index if hasattr(X_test, "index") else range(len(y_proba)))
                y_test_series = y_test if isinstance(y_test, pd.Series) else pd.Series(yt, index=X_test.index if hasattr(X_test, "index") else range(len(yt)))
                ic = information_coefficient(y_test_series, y_proba_series)

                ic_values.append(ic)
                fold_info.update({"auc": fold_auc, "brier": fold_brier, "ic": ic})

            except Exception as exc:
                logger.warning("Walk-forward fold %d failed: %s", fold_idx, exc)
                ic_values.append(float("nan"))
                fold_info["error"] = str(exc)

            fold_reports.append(fold_info)

        # Compute walk-forward IC across all folds (no re-fit needed for this)
        finite_ic = [v for v in ic_values if np.isfinite(v)]

        ic_mean = float(np.mean(finite_ic)) if finite_ic else float("nan")
        ic_std = float(np.std(finite_ic, ddof=1)) if len(finite_ic) > 1 else 0.0
        rob = robustness_score(ic_values)

        return {
            "ic_values": ic_values,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "robustness_score": rob,
            "fold_reports": fold_reports,
        }

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------

    def generate_summary_text(self, report: EvaluationReport) -> str:
        """Generate human-readable summary of the evaluation report."""
        lines = [
            f"Evaluation Report — model_id={report.model_id or 'N/A'}",
            f"  Split          : {report.split}",
            f"  N samples      : {report.n_samples}",
            f"  AUC-ROC        : {report.auc:.4f}" if np.isfinite(report.auc) else "  AUC-ROC        : N/A",
            f"  PR-AUC         : {report.pr_auc:.4f}" if np.isfinite(report.pr_auc) else "  PR-AUC         : N/A",
            f"  Brier score    : {report.brier_score:.4f}" if np.isfinite(report.brier_score) else "  Brier score    : N/A",
            f"  Log loss       : {report.log_loss:.4f}" if np.isfinite(report.log_loss) else "  Log loss       : N/A",
            f"  IC             : {report.information_coefficient:.4f}" if np.isfinite(report.information_coefficient) else "  IC             : N/A",
            f"  IC t-stat      : {report.ic_t_stat:.2f}" if np.isfinite(report.ic_t_stat) else "  IC t-stat      : N/A",
            f"  Accuracy       : {report.accuracy:.4f}" if np.isfinite(report.accuracy) else "  Accuracy       : N/A",
            f"  Precision      : {report.precision:.4f}" if np.isfinite(report.precision) else "  Precision      : N/A",
            f"  Recall         : {report.recall:.4f}" if np.isfinite(report.recall) else "  Recall         : N/A",
            f"  F1             : {report.f1:.4f}" if np.isfinite(report.f1) else "  F1             : N/A",
        ]

        if report.meta_label_precision is not None and np.isfinite(report.meta_label_precision):
            lines += [
                f"  Meta-label precision : {report.meta_label_precision:.4f}",
                f"  Baseline precision   : {report.baseline_precision:.4f}" if np.isfinite(report.baseline_precision) else "  Baseline precision   : N/A",
                f"  Filter improvement   : {report.filter_improvement:+.4f}" if np.isfinite(report.filter_improvement) else "  Filter improvement   : N/A",
            ]

        if report.regime_metrics:
            lines.append("  Regime-sliced metrics:")
            for regime_name, metrics in report.regime_metrics.items():
                auc_str = f"{metrics.get('auc', float('nan')):.4f}" if np.isfinite(metrics.get("auc", float("nan"))) else "N/A"
                brier_str = f"{metrics.get('brier', float('nan')):.4f}" if np.isfinite(metrics.get("brier", float("nan"))) else "N/A"
                lines.append(f"    {regime_name:20s} : AUC={auc_str}, Brier={brier_str}, n={metrics.get('n_samples', 0)}")

        return "\n".join(lines)
