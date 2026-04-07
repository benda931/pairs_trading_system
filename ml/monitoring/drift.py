# -*- coding: utf-8 -*-
"""
ml/monitoring/drift.py — Feature Drift and Model Health Monitoring
===================================================================

FeatureDriftMonitor: per-feature PSI drift detection.
ModelHealthMonitor : aggregate health checks (freshness, drift, performance).

All methods are fallback-safe — they never raise. Exceptions are logged at
WARNING level and safe defaults are returned.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml.contracts import (
    DriftReport,
    DriftSeverity,
    ModelHealthState,
    ModelHealthStatus,
    ModelMetadata,
    ModelStatus,
    _now_utc,
)
from ml.monitoring.health import kolmogorov_smirnov_drift, psi_score

logger = logging.getLogger("ml.monitoring")


# ---------------------------------------------------------------------------
# FeatureDriftMonitor
# ---------------------------------------------------------------------------

class FeatureDriftMonitor:
    """
    Monitor feature distribution drift using PSI.

    PSI < 0.10: no significant drift
    0.10-0.25 : moderate drift
    > 0.25    : significant drift
    """

    def __init__(
        self,
        n_bins: int = 10,
        psi_warn_threshold: float = 0.10,
        psi_alert_threshold: float = 0.25,
    ):
        self._n_bins = n_bins
        self._psi_warn = psi_warn_threshold
        self._psi_alert = psi_alert_threshold

        # Reference distributions stored as (sorted_array, bin_edges, counts)
        # keyed by feature name
        self._reference: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Fit reference
    # ------------------------------------------------------------------

    def fit_reference(self, X_reference: pd.DataFrame) -> None:
        """
        Store reference feature distributions for drift detection.

        Parameters
        ----------
        X_reference : DataFrame where each column is a feature.
        """
        self._reference = {}
        for col in X_reference.columns:
            try:
                arr = np.asarray(X_reference[col], dtype=float)
                arr = arr[np.isfinite(arr)]
                if len(arr) >= 10:
                    self._reference[col] = arr
            except Exception as exc:
                logger.debug("fit_reference skipped column %s: %s", col, exc)

        logger.info(
            "FeatureDriftMonitor: fitted reference on %d features",
            len(self._reference),
        )

    # ------------------------------------------------------------------
    # PSI per feature
    # ------------------------------------------------------------------

    def compute_psi(self, X_current: pd.DataFrame) -> dict[str, float]:
        """
        Compute PSI per feature.

        Returns {feature_name: psi_score}. Features not in the reference
        or with insufficient data receive NaN.
        """
        result: dict[str, float] = {}

        for col in X_current.columns:
            ref = self._reference.get(col)
            if ref is None:
                result[col] = float("nan")
                continue

            try:
                cur = np.asarray(X_current[col], dtype=float)
                cur = cur[np.isfinite(cur)]
                if len(cur) < 10:
                    result[col] = float("nan")
                    continue
                result[col] = psi_score(ref, cur, n_bins=self._n_bins)
            except Exception as exc:
                logger.debug("PSI computation failed for %s: %s", col, exc)
                result[col] = float("nan")

        return result

    # ------------------------------------------------------------------
    # Drift report
    # ------------------------------------------------------------------

    def generate_drift_report(
        self,
        model_id: str,
        X_current: pd.DataFrame,
    ) -> DriftReport:
        """
        Generate a structured DriftReport for the given model and current data.
        """
        report = DriftReport(model_id=model_id, report_type="feature")

        try:
            psi_scores = self.compute_psi(X_current)
            report.feature_psi_scores = {k: v for k, v in psi_scores.items()}

            finite_psi = [v for v in psi_scores.values() if np.isfinite(v)]
            if finite_psi:
                report.mean_psi = float(np.mean(finite_psi))

            report.features_drifted = [
                k for k, v in psi_scores.items()
                if np.isfinite(v) and v >= self._psi_warn
            ]

            severity = _classify_psi_severity(
                report.mean_psi if np.isfinite(report.mean_psi) else 0.0,
                psi_warn=self._psi_warn,
                psi_alert=self._psi_alert,
            )
            report.severity = severity
            report.recommended_action = _recommend_action_from_severity(severity)

        except Exception as exc:
            logger.warning("generate_drift_report failed for %s: %s", model_id, exc)

        return report


# ---------------------------------------------------------------------------
# ModelHealthMonitor
# ---------------------------------------------------------------------------

class ModelHealthMonitor:
    """
    Monitors overall model health: freshness, drift, performance, calibration.
    """

    def __init__(
        self,
        registry=None,                  # MLModelRegistry | None
        stale_threshold_hours: int = 48,
    ):
        if registry is None:
            try:
                from ml.registry import get_ml_registry
                self._registry = get_ml_registry()
            except Exception:
                self._registry = None
        else:
            self._registry = registry

        self._stale_hours = stale_threshold_hours
        self._drift_monitors: dict[str, FeatureDriftMonitor] = {}
        # Store recent score baselines: model_id → reference score array
        self._score_baselines: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Primary health check
    # ------------------------------------------------------------------

    def check_health(
        self,
        model_id: str,
        X_recent: Optional[pd.DataFrame] = None,
        y_recent: Optional[pd.Series] = None,
    ) -> ModelHealthStatus:
        """
        Run all health checks for a model.

        Returns ModelHealthStatus with state, drift severity, recommended action.
        Never raises.
        """
        status = ModelHealthStatus(model_id=model_id)

        try:
            # Retrieve metadata
            metadata: Optional[ModelMetadata] = None
            if self._registry is not None:
                metadata = self._registry.get(model_id)

            if metadata is None:
                status.state = ModelHealthState.SUSPENDED
                status.recommended_action = "model_not_found"
                status.action_reason = f"model_id={model_id} not in registry"
                return status

            status.task_family = metadata.task_family.value

            # Freshness check
            is_fresh, hours_since = self.check_freshness(metadata)
            status.hours_since_training = hours_since
            status.stale = not is_fresh
            status.last_trained_at = metadata.trained_at

            # Drift check (if drift monitor is registered for this model)
            if model_id in self._drift_monitors and X_recent is not None:
                try:
                    monitor = self._drift_monitors[model_id]
                    drift_report = monitor.generate_drift_report(model_id, X_recent)
                    status.latest_drift_report_id = drift_report.report_id
                    status.drift_severity = drift_report.severity

                    # Auto-block: CRITICAL drift must stop production inference immediately.
                    # Drift detection that only observes but never acts provides no safety guarantee.
                    if drift_report.severity == DriftSeverity.CRITICAL:
                        self._auto_block_model(model_id, drift_report)

                except Exception as exc:
                    logger.warning("Drift check failed for %s: %s", model_id, exc)
                    status.drift_severity = DriftSeverity.NONE

            # Recent performance check
            if y_recent is not None and X_recent is not None:
                try:
                    model_obj = self._registry.get_model_object(model_id) if self._registry else None
                    if model_obj is not None:
                        from ml.evaluation.metrics import auc_roc, brier_score, information_coefficient
                        if hasattr(model_obj, "predict_proba"):
                            raw = model_obj.predict_proba(X_recent)
                            yp = np.asarray(raw, dtype=float)
                            if yp.ndim == 2:
                                yp = yp[:, 1]
                        else:
                            yp = np.asarray(model_obj.predict(X_recent), dtype=float)

                        yt = np.asarray(y_recent, dtype=float)
                        status.recent_ic = information_coefficient(
                            pd.Series(yt), pd.Series(yp)
                        )
                        status.recent_brier = brier_score(yt, yp)

                        # Compare to stored baseline metrics
                        baseline_ic = metadata.cv_ic_mean
                        if np.isfinite(status.recent_ic) and np.isfinite(baseline_ic):
                            if status.recent_ic < baseline_ic - 0.05:
                                status.performance_degraded = True
                except Exception as exc:
                    logger.warning("Performance check failed for %s: %s", model_id, exc)

            # Determine overall state
            status.state = _determine_health_state(status)
            status.recommended_action = self.recommend_action(status)

        except Exception as exc:
            logger.error("check_health failed for %s: %s", model_id, exc)
            status.state = ModelHealthState.SUSPENDED
            status.recommended_action = f"error: {exc}"

        return status

    # ------------------------------------------------------------------
    # Freshness
    # ------------------------------------------------------------------

    def check_freshness(self, metadata: ModelMetadata) -> tuple[bool, float]:
        """Returns (is_fresh, hours_since_training)."""
        if not metadata.trained_at:
            return False, float("nan")

        try:
            trained_dt = datetime.fromisoformat(metadata.trained_at)
            if trained_dt.tzinfo is None:
                trained_dt = trained_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(tz=timezone.utc)
            hours = (now - trained_dt).total_seconds() / 3600.0
            return hours <= self._stale_hours, hours
        except Exception as exc:
            logger.debug("check_freshness parse error: %s", exc)
            return False, float("nan")

    # ------------------------------------------------------------------
    # Prediction drift
    # ------------------------------------------------------------------

    def check_prediction_drift(
        self,
        model_id: str,
        recent_scores: list[float],
        reference_scores: list[float],
    ) -> DriftSeverity:
        """
        Check if recent score distribution has drifted from baseline using PSI.
        """
        try:
            ref = np.array([s for s in reference_scores if np.isfinite(s)], dtype=float)
            cur = np.array([s for s in recent_scores if np.isfinite(s)], dtype=float)

            if len(ref) < 10 or len(cur) < 10:
                return DriftSeverity.NONE

            psi = psi_score(ref, cur)
            if not np.isfinite(psi):
                return DriftSeverity.NONE

            return _classify_psi_severity(psi)
        except Exception as exc:
            logger.warning("check_prediction_drift failed for %s: %s", model_id, exc)
            return DriftSeverity.NONE

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def recommend_action(self, health: ModelHealthStatus) -> str:
        """Return recommended action string based on health state."""
        if health.state == ModelHealthState.HEALTHY:
            return "none"
        if health.state == ModelHealthState.STALE:
            return "retrain"
        if health.state == ModelHealthState.DRIFTING:
            return "monitor_and_consider_retrain"
        if health.state == ModelHealthState.DEGRADED:
            return "retrain_or_rollback"
        if health.state == ModelHealthState.SUSPENDED:
            return "investigate_and_redeploy"
        if health.state == ModelHealthState.FALLBACK_ACTIVE:
            return "restore_primary_model"
        return "none"

    # ------------------------------------------------------------------
    # Register drift monitor
    # ------------------------------------------------------------------

    def register_drift_monitor(
        self,
        model_id: str,
        monitor: FeatureDriftMonitor,
    ) -> None:
        """Attach a pre-fitted FeatureDriftMonitor for a specific model."""
        self._drift_monitors[model_id] = monitor

    # ------------------------------------------------------------------
    # Auto-block on critical drift
    # ------------------------------------------------------------------

    def _auto_block_model(self, model_id: str, drift_report) -> None:
        """
        Automatically set model status to BLOCKED when drift is CRITICAL.

        Drift detection that does not trigger automatic model blocking is purely
        observational — it cannot prevent a critically drifted model from serving
        production inference. This method closes that gap.

        Also opens an incident if an incident manager is available.
        Never raises — failure to block is logged at CRITICAL level but does not
        crash the health check.
        """
        from ml.contracts import ModelStatus
        try:
            if self._registry is not None:
                # Block the model in the registry
                if hasattr(self._registry, "update_status"):
                    self._registry.update_status(model_id, ModelStatus.BLOCKED)
                    logger.error(
                        "AUTO-BLOCKED model %s: DriftSeverity.CRITICAL detected "
                        "(mean_psi=%.4f, features_drifted=%d). "
                        "Model must be retrained before returning to production.",
                        model_id,
                        drift_report.mean_psi if hasattr(drift_report, "mean_psi") else float("nan"),
                        len(drift_report.features_drifted) if hasattr(drift_report, "features_drifted") else 0,
                    )
                else:
                    logger.critical(
                        "Cannot auto-block model %s: registry has no update_status method. "
                        "MANUAL INTERVENTION REQUIRED — model is critically drifted.",
                        model_id,
                    )
        except Exception as exc:
            logger.critical(
                "Failed to auto-block critically drifted model %s: %s — "
                "MANUAL INTERVENTION REQUIRED.",
                model_id, exc,
            )

        # Attempt to open an incident if the incident manager is reachable
        try:
            from incidents.manager import IncidentManager
            im = IncidentManager()
            im.open_incident(
                title=f"CRITICAL drift: model {model_id} auto-blocked",
                severity="critical",
                source="drift_monitor",
                details={
                    "model_id": model_id,
                    "drift_report_id": getattr(drift_report, "report_id", ""),
                    "mean_psi": getattr(drift_report, "mean_psi", float("nan")),
                    "features_drifted": getattr(drift_report, "features_drifted", []),
                    "action_taken": "model_status_set_to_BLOCKED",
                },
            )
        except Exception:
            pass  # Incident manager is optional; missing it does not unblock the model

    # ------------------------------------------------------------------
    # Health summary DataFrame
    # ------------------------------------------------------------------

    def generate_health_summary(self, model_ids: list[str]) -> pd.DataFrame:
        """
        Generate a DataFrame summarizing health of all listed models.

        Columns: model_id, state, stale, drift_severity, recommended_action
        """
        rows = []
        for model_id in model_ids:
            try:
                health = self.check_health(model_id)
                rows.append({
                    "model_id": model_id,
                    "state": health.state.value,
                    "stale": health.stale,
                    "hours_since_training": health.hours_since_training,
                    "drift_severity": health.drift_severity.value,
                    "performance_degraded": health.performance_degraded,
                    "recent_ic": health.recent_ic,
                    "recent_brier": health.recent_brier,
                    "recommended_action": health.recommended_action,
                })
            except Exception as exc:
                logger.warning("Health summary failed for %s: %s", model_id, exc)
                rows.append({
                    "model_id": model_id,
                    "state": ModelHealthState.SUSPENDED.value,
                    "stale": True,
                    "hours_since_training": float("nan"),
                    "drift_severity": DriftSeverity.NONE.value,
                    "performance_degraded": False,
                    "recent_ic": float("nan"),
                    "recent_brier": float("nan"),
                    "recommended_action": f"error: {exc}",
                })

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_psi_severity(
    psi: float,
    psi_warn: float = 0.10,
    psi_alert: float = 0.25,
) -> DriftSeverity:
    if not np.isfinite(psi):
        return DriftSeverity.NONE
    if psi < psi_warn:
        return DriftSeverity.NONE
    if psi < psi_alert:
        return DriftSeverity.MODERATE
    if psi < 0.50:
        return DriftSeverity.SEVERE
    return DriftSeverity.CRITICAL


def _recommend_action_from_severity(severity: DriftSeverity) -> str:
    mapping = {
        DriftSeverity.NONE: "none",
        DriftSeverity.MINOR: "monitor",
        DriftSeverity.MODERATE: "monitor",
        DriftSeverity.SEVERE: "retrain",
        DriftSeverity.CRITICAL: "suspend",
    }
    return mapping.get(severity, "none")


def _determine_health_state(status: ModelHealthStatus) -> ModelHealthState:
    """Derive overall health state from component checks."""
    if status.stale and status.drift_severity in (DriftSeverity.SEVERE, DriftSeverity.CRITICAL):
        return ModelHealthState.DEGRADED
    if status.performance_degraded:
        return ModelHealthState.DEGRADED
    if status.stale:
        return ModelHealthState.STALE
    if status.drift_severity in (DriftSeverity.MODERATE, DriftSeverity.SEVERE, DriftSeverity.CRITICAL):
        return ModelHealthState.DRIFTING
    return ModelHealthState.HEALTHY
