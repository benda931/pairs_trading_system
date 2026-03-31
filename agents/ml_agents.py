# -*- coding: utf-8 -*-
"""
agents/ml_agents.py — ML / Model Agent Implementations
=======================================================

Seven agent classes covering the ML platform lifecycle: feature health,
label governance, model research, meta-labeling, regime modeling, model
risk assessment, and promotion review.

All agents:
  - Subclass BaseAgent (from agents.base)
  - Handle ImportError gracefully with lightweight fallbacks
  - Return a proper dict from _execute() — never None
  - Use uuid.uuid4() for generated IDs
  - Use datetime.utcnow().isoformat() + "Z" for timestamps
  - Are fully type-annotated
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from agents.base import AgentAuditLogger, BaseAgent
from core.contracts import AgentTask


def _utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _new_id() -> str:
    return str(uuid.uuid4())


# ── Governance promotion thresholds ──────────────────────────────
_ML_THRESHOLDS = {
    "ic_min": 0.05,
    "auc_min": 0.55,
    "brier_max": 0.25,
}

_REGIME_THRESHOLDS = {
    "ic_min": 0.07,
    "auc_min": 0.60,
    "brier_max": 0.22,
}

_BREAK_THRESHOLDS = {
    "ic_min": 0.08,
    "auc_min": 0.62,
    "brier_max": 0.20,
}


# ══════════════════════════════════════════════════════════════════
# 1. FeatureStewardAgent
# ══════════════════════════════════════════════════════════════════


class FeatureStewardAgent(BaseAgent):
    """
    Audits feature health for the ML platform.

    Checks NaN rates, constant features, and drift severity by attempting
    to use ``ml.monitoring.drift.FeatureDriftMonitor``. Falls back to simple
    mean/std/NaN checks if the monitor is unavailable.

    Task types
    ----------
    audit_feature_health
        Full health check across listed features.
    check_feature_drift
        Drift-focused check comparing current vs reference distributions.

    Required payload keys
    ---------------------
    feature_names : list[str]

    Optional payload keys
    ---------------------
    prices : pd.DataFrame
    as_of : str  (ISO date)
    reference_data : dict[str, list]  (feature_name → reference values)
    current_data : dict[str, list]    (feature_name → current values)

    Output keys
    -----------
    feature_health : dict[str, dict]
    unhealthy_features : list[str]
    drift_alerts : list[str]
    recommendations : list[str]
    """

    NAME = "feature_steward"
    ALLOWED_TASK_TYPES = {"audit_feature_health", "check_feature_drift"}
    REQUIRED_PAYLOAD_KEYS = {"feature_names"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        feature_names: list[str] = task.payload["feature_names"]
        reference_data: dict[str, list] = task.payload.get("reference_data") or {}
        current_data: dict[str, list] = task.payload.get("current_data") or {}
        as_of = task.payload.get("as_of")

        audit.log(
            f"Auditing {len(feature_names)} features, "
            f"has_reference={bool(reference_data)}, as_of={as_of}"
        )

        # Try FeatureDriftMonitor
        _drift_monitor = None
        try:
            from ml.monitoring.drift import FeatureDriftMonitor
            _drift_monitor = FeatureDriftMonitor()
            audit.log("Using ml.monitoring.drift.FeatureDriftMonitor")
        except ImportError:
            audit.warn("FeatureDriftMonitor unavailable — using lightweight fallback")

        # Try to load FEATURE_REGISTRY for availability checks
        _feature_registry: dict[str, Any] = {}
        try:
            from ml.features.definitions import FEATURE_REGISTRY
            _feature_registry = dict(FEATURE_REGISTRY)
            audit.log(f"Loaded FEATURE_REGISTRY with {len(_feature_registry)} features")
        except ImportError:
            pass

        feature_health: dict[str, dict[str, Any]] = {}
        unhealthy: list[str] = []
        drift_alerts: list[str] = []

        for fname in feature_names:
            available_in_registry = fname in _feature_registry if _feature_registry else None
            nan_rate: float | None = None
            drift_severity: str = "UNKNOWN"
            notes: list[str] = []

            # NaN / constant check from current_data
            if fname in current_data:
                try:
                    import numpy as np
                    vals = np.array(current_data[fname], dtype=float)
                    if len(vals) > 0:
                        nan_rate = float(np.isnan(vals).mean())
                        if nan_rate > 0.10:
                            notes.append(f"high_nan_rate:{nan_rate:.2%}")
                        if len(vals) > 1 and float(np.nanstd(vals)) < 1e-10:
                            notes.append("constant_feature")
                except Exception as exc:
                    notes.append(f"compute_error:{exc}")

            # Drift check
            if fname in current_data and fname in reference_data and _drift_monitor is not None:
                try:
                    report = _drift_monitor.compute_psi(
                        reference=reference_data[fname],
                        current=current_data[fname],
                    )
                    psi = float(report.psi) if hasattr(report, "psi") else 0.0
                    if psi >= 0.25:
                        drift_severity = "HIGH"
                        drift_alerts.append(f"{fname}: PSI={psi:.3f} (HIGH drift)")
                    elif psi >= 0.10:
                        drift_severity = "MEDIUM"
                    else:
                        drift_severity = "LOW"
                except Exception as exc:
                    drift_severity = "ERROR"
                    notes.append(f"drift_check_error:{exc}")
            elif fname in current_data and fname in reference_data:
                # Lightweight: compare means and stds
                try:
                    import numpy as np
                    ref = np.array(reference_data[fname], dtype=float)
                    cur = np.array(current_data[fname], dtype=float)
                    ref_mean, cur_mean = float(np.nanmean(ref)), float(np.nanmean(cur))
                    ref_std = float(np.nanstd(ref)) + 1e-12
                    mean_shift = abs(ref_mean - cur_mean) / ref_std
                    if mean_shift > 2.0:
                        drift_severity = "HIGH"
                        drift_alerts.append(f"{fname}: mean_shift={mean_shift:.2f}σ (HIGH drift)")
                    elif mean_shift > 1.0:
                        drift_severity = "MEDIUM"
                    else:
                        drift_severity = "LOW"
                except Exception:
                    drift_severity = "UNKNOWN"

            is_healthy = (
                (nan_rate is None or nan_rate <= 0.10)
                and drift_severity not in ("HIGH", "ERROR")
                and "constant_feature" not in notes
            )
            if not is_healthy:
                unhealthy.append(fname)

            feature_health[fname] = {
                "available": True if available_in_registry is None else available_in_registry,
                "nan_rate": round(nan_rate, 4) if nan_rate is not None else None,
                "drift_severity": drift_severity,
                "healthy": is_healthy,
                "notes": notes,
            }

        # Recommendations
        recommendations: list[str] = []
        if unhealthy:
            recommendations.append(
                f"{len(unhealthy)} unhealthy features detected — review before model training"
            )
        if drift_alerts:
            recommendations.append(
                f"{len(drift_alerts)} features show drift — consider model retraining"
            )
        not_in_registry = [f for f in feature_names if not feature_health[f]["available"]]
        if not_in_registry:
            recommendations.append(
                f"{len(not_in_registry)} features not found in FEATURE_REGISTRY: {not_in_registry}"
            )

        audit.log(
            f"Feature audit complete: {len(unhealthy)}/{len(feature_names)} unhealthy, "
            f"{len(drift_alerts)} drift alerts"
        )

        return {
            "feature_health": feature_health,
            "unhealthy_features": unhealthy,
            "drift_alerts": drift_alerts,
            "recommendations": recommendations,
        }


# ══════════════════════════════════════════════════════════════════
# 2. LabelGovernanceAgent
# ══════════════════════════════════════════════════════════════════


class LabelGovernanceAgent(BaseAgent):
    """
    Validates ML labels for leakage documentation and governance compliance.

    Checks that all requested labels exist in ``LABEL_REGISTRY`` and that
    each has leakage risks documented. Falls back gracefully if the registry
    is not importable.

    Task types
    ----------
    validate_labels
        Check existence and documentation of requested labels.
    check_label_leakage
        Verify leakage documentation completeness.

    Required payload keys
    ---------------------
    label_names : list[str]

    Optional payload keys
    ---------------------
    metadata : dict  (additional context)

    Output keys
    -----------
    label_audit : dict[str, dict]
    undocumented_labels : list[str]
    warnings : list[str]
    """

    NAME = "label_governance"
    ALLOWED_TASK_TYPES = {"validate_labels", "check_label_leakage"}
    REQUIRED_PAYLOAD_KEYS = {"label_names"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        label_names: list[str] = task.payload["label_names"]
        metadata: dict[str, Any] = task.payload.get("metadata") or {}

        audit.log(f"Auditing {len(label_names)} labels")

        _label_registry: dict[str, Any] = {}
        try:
            from ml.labels.definitions import LABEL_REGISTRY
            _label_registry = dict(LABEL_REGISTRY)
            audit.log(f"Loaded LABEL_REGISTRY with {len(_label_registry)} labels")
        except ImportError:
            audit.warn("LABEL_REGISTRY unavailable — cannot verify label existence")

        label_audit: dict[str, dict[str, Any]] = {}
        undocumented: list[str] = []
        warnings: list[str] = []

        for lname in label_names:
            if _label_registry:
                defn = _label_registry.get(lname)
                exists = defn is not None
            else:
                defn = None
                exists = None  # unknown

            has_leakage_docs = False
            valid_use_cases: list[str] = []
            notes: list[str] = []

            if defn is not None:
                # Check leakage_risks attribute
                lr = getattr(defn, "leakage_risks", None)
                has_leakage_docs = bool(lr)
                if not has_leakage_docs:
                    undocumented.append(lname)
                    notes.append("missing_leakage_documentation")

                # Valid use cases
                vuc = getattr(defn, "valid_use_cases", [])
                valid_use_cases = list(vuc) if vuc else []

                # Horizon check
                horizon = getattr(defn, "horizon_days", None)
                if horizon is not None and horizon <= 0:
                    notes.append(f"invalid_horizon:{horizon}")
                    warnings.append(f"Label '{lname}' has non-positive horizon_days={horizon}")

            elif exists is False:
                notes.append("not_in_LABEL_REGISTRY")
                undocumented.append(lname)
                warnings.append(f"Label '{lname}' not found in LABEL_REGISTRY")
            else:
                notes.append("registry_unavailable")

            label_audit[lname] = {
                "exists": exists,
                "has_leakage_docs": has_leakage_docs,
                "valid_use_cases": valid_use_cases,
                "notes": notes,
            }

        if undocumented:
            warnings.append(
                f"{len(undocumented)} labels lack leakage documentation: {undocumented}. "
                "Add '# LEAKAGE NOTE:' comments to builder methods before training."
            )

        audit.log(
            f"Label governance audit complete: {len(undocumented)} undocumented labels, "
            f"{len(warnings)} warnings"
        )

        return {
            "label_audit": label_audit,
            "undocumented_labels": undocumented,
            "warnings": warnings,
        }


# ══════════════════════════════════════════════════════════════════
# 3. ModelResearchAgent
# ══════════════════════════════════════════════════════════════════


class ModelResearchAgent(BaseAgent):
    """
    Compares and evaluates ML models against governance thresholds.

    Task types
    ----------
    compare_models
        Rank a list of models by composite IC/AUC/Brier score.
    evaluate_model
        Evaluate a single model against governance criteria.

    Required payload keys
    ---------------------
    model_ids : list[str]

    Optional payload keys
    ---------------------
    evaluation_metrics : dict[str, dict]  ({model_id: {ic, auc, brier}})
    baseline_metrics : dict  ({ic, auc, brier})

    Output keys
    -----------
    model_rankings : list[dict]
    best_model_id : str | None
    below_threshold : list[str]
    recommendations : list[str]
    """

    NAME = "model_research"
    ALLOWED_TASK_TYPES = {"compare_models", "evaluate_model"}
    REQUIRED_PAYLOAD_KEYS = {"model_ids"}

    @staticmethod
    def _composite_score(ic: float | None, auc: float | None, brier: float | None) -> float:
        """Compute a [0,1] composite score from IC, AUC-ROC, and Brier."""
        # Normalise each metric to [0,1]; missing metrics contribute 0.
        ic_score = max(0.0, min(1.0, (ic or 0.0) / 0.20)) if ic is not None else 0.0
        auc_score = max(0.0, min(1.0, ((auc or 0.5) - 0.5) / 0.20)) if auc is not None else 0.0
        # For Brier: lower is better; 0.25 is the max acceptable
        brier_score = max(0.0, min(1.0, (0.25 - (brier or 0.25)) / 0.25)) if brier is not None else 0.0
        return 0.40 * ic_score + 0.40 * auc_score + 0.20 * brier_score

    @staticmethod
    def _meets_governance(ic: float | None, auc: float | None, brier: float | None) -> bool:
        """Return True if all available metrics meet base governance thresholds."""
        if ic is not None and ic < _ML_THRESHOLDS["ic_min"]:
            return False
        if auc is not None and auc < _ML_THRESHOLDS["auc_min"]:
            return False
        if brier is not None and brier > _ML_THRESHOLDS["brier_max"]:
            return False
        return True

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        model_ids: list[str] = task.payload["model_ids"]
        eval_metrics: dict[str, dict[str, float]] = task.payload.get("evaluation_metrics") or {}
        baseline: dict[str, float] = task.payload.get("baseline_metrics") or {}

        audit.log(
            f"Evaluating {len(model_ids)} models, "
            f"has_metrics={bool(eval_metrics)}, has_baseline={bool(baseline)}"
        )

        rankings: list[dict[str, Any]] = []
        below_threshold: list[str] = []

        for mid in model_ids:
            m = eval_metrics.get(mid) or {}
            ic = m.get("ic")
            auc = m.get("auc")
            brier = m.get("brier")

            # Try to pull from ML registry if metrics not provided
            if ic is None and auc is None and brier is None:
                try:
                    from ml.registry.registry import get_ml_registry
                    reg = get_ml_registry()
                    mdl = reg.get(mid)
                    if mdl is not None:
                        meta = getattr(mdl, "metadata", None)
                        if meta:
                            ic = getattr(meta, "ic", None)
                            auc = getattr(meta, "auc", None)
                            brier = getattr(meta, "brier", None)
                except Exception:
                    pass

            score = self._composite_score(ic, auc, brier)
            meets_gov = self._meets_governance(ic, auc, brier)
            if not meets_gov:
                below_threshold.append(mid)

            failing_criteria: list[str] = []
            if ic is not None and ic < _ML_THRESHOLDS["ic_min"]:
                failing_criteria.append(f"ic={ic:.4f}<{_ML_THRESHOLDS['ic_min']}")
            if auc is not None and auc < _ML_THRESHOLDS["auc_min"]:
                failing_criteria.append(f"auc={auc:.4f}<{_ML_THRESHOLDS['auc_min']}")
            if brier is not None and brier > _ML_THRESHOLDS["brier_max"]:
                failing_criteria.append(f"brier={brier:.4f}>{_ML_THRESHOLDS['brier_max']}")

            rankings.append(
                {
                    "model_id": mid,
                    "composite_score": round(score, 6),
                    "ic": ic,
                    "auc": auc,
                    "brier": brier,
                    "meets_governance": meets_gov,
                    "failing_criteria": failing_criteria,
                }
            )

        rankings.sort(key=lambda x: -(x["composite_score"] or 0.0))
        best_model_id = rankings[0]["model_id"] if rankings else None

        # Baseline comparison
        recommendations: list[str] = []
        if below_threshold:
            recommendations.append(
                f"{len(below_threshold)} model(s) below governance thresholds: {below_threshold}. "
                "Do not promote these to CHAMPION."
            )
        if rankings and baseline:
            best = rankings[0]
            b_ic = baseline.get("ic", 0.0)
            b_auc = baseline.get("auc", 0.5)
            if best.get("ic") and best["ic"] > b_ic + 0.02:
                recommendations.append(
                    f"Best model '{best_model_id}' outperforms baseline IC by "
                    f"{best['ic'] - b_ic:.4f} — consider shadow deployment."
                )

        if not below_threshold and rankings:
            recommendations.append(
                f"All models meet governance thresholds. Best model: '{best_model_id}'."
            )

        audit.log(
            f"Model research complete: best={best_model_id}, "
            f"below_threshold={below_threshold}"
        )

        return {
            "model_rankings": rankings,
            "best_model_id": best_model_id,
            "below_threshold": below_threshold,
            "recommendations": recommendations,
        }


# ══════════════════════════════════════════════════════════════════
# 4. MetaLabelingAgent
# ══════════════════════════════════════════════════════════════════


class MetaLabelingAgent(BaseAgent):
    """
    Applies meta-labeling to raw signals to filter low-quality entries.

    Attempts to use ``ml.inference.scorer.ModelScorer`` for ML-based
    meta-labeling. Falls back to a conviction × regime-safety heuristic.

    Task types
    ----------
    assess_meta_label
        Assess a batch of signals and recommend TAKE/SKIP/DOWNSIZE.
    batch_meta_label
        Alias for assess_meta_label.

    Required payload keys
    ---------------------
    signals : list[dict]  (each: {pair_id, z_score, regime, conviction})

    Optional payload keys
    ---------------------
    model_id : str

    Output keys
    -----------
    assessments : list[dict]
    take_count : int
    skip_count : int
    downsize_count : int
    """

    NAME = "meta_labeling"
    ALLOWED_TASK_TYPES = {"assess_meta_label", "batch_meta_label"}
    REQUIRED_PAYLOAD_KEYS = {"signals"}

    # Safety factor per regime
    _REGIME_SAFETY: dict[str, float] = {
        "MEAN_REVERTING": 1.0,
        "TRANSITIONAL": 0.70,
        "TRENDING": 0.0,    # veto
        "CRISIS": 0.0,       # veto
        "BROKEN": 0.0,       # veto
        "UNKNOWN": 0.50,
    }

    def _heuristic_probability(
        self, z_score: float, regime: str, conviction: float
    ) -> float:
        """Heuristic meta-label probability from conviction and regime safety."""
        safety = self._REGIME_SAFETY.get(regime, 0.50)
        raw = float(conviction) * safety
        # Boost for strong z-scores
        abs_z = abs(float(z_score))
        z_boost = min(abs_z / 4.0, 0.2)
        return min(raw + z_boost, 1.0)

    @staticmethod
    def _action_from_prob(prob: float) -> str:
        if prob >= 0.65:
            return "TAKE"
        elif prob >= 0.40:
            return "DOWNSIZE"
        else:
            return "SKIP"

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        signals: list[dict[str, Any]] = task.payload["signals"]
        model_id: str | None = task.payload.get("model_id")

        audit.log(f"Meta-labeling {len(signals)} signals, model_id={model_id}")

        _scorer = None
        try:
            from ml.inference.scorer import ModelScorer
            _scorer = ModelScorer()
            if model_id:
                _scorer = ModelScorer(model_id=model_id)
            audit.log("Using ml.inference.scorer.ModelScorer")
        except (ImportError, Exception) as exc:
            audit.warn(f"ModelScorer unavailable ({exc}) — using heuristic fallback")

        assessments: list[dict[str, Any]] = []
        take_count = skip_count = downsize_count = 0

        for sig in signals:
            pair_id = sig.get("pair_id", "unknown")
            z_score = float(sig.get("z_score", 0.0))
            regime = str(sig.get("regime", "UNKNOWN"))
            conviction = float(sig.get("conviction", 0.5))
            fallback_used = False

            if _scorer is not None:
                try:
                    result = _scorer.score(
                        pair_id=pair_id,
                        z_score=z_score,
                        regime=regime,
                        conviction=conviction,
                    )
                    prob = float(result.probability) if hasattr(result, "probability") else 0.5
                    fallback_used = getattr(result, "fallback_triggered", False)
                    if fallback_used:
                        prob = self._heuristic_probability(z_score, regime, conviction)
                except Exception as exc:
                    audit.warn(f"{pair_id}: ModelScorer.score failed ({exc}), using heuristic")
                    prob = self._heuristic_probability(z_score, regime, conviction)
                    fallback_used = True
            else:
                prob = self._heuristic_probability(z_score, regime, conviction)
                fallback_used = True

            action = self._action_from_prob(prob)

            if action == "TAKE":
                take_count += 1
            elif action == "SKIP":
                skip_count += 1
            else:
                downsize_count += 1

            assessments.append(
                {
                    "pair_id": pair_id,
                    "success_probability": round(prob, 6),
                    "action": action,
                    "confidence": round(conviction, 6),
                    "regime": regime,
                    "fallback_used": fallback_used,
                }
            )

        audit.log(
            f"Meta-labeling complete: take={take_count}, "
            f"downsize={downsize_count}, skip={skip_count}"
        )

        return {
            "assessments": assessments,
            "take_count": take_count,
            "skip_count": skip_count,
            "downsize_count": downsize_count,
        }


# ══════════════════════════════════════════════════════════════════
# 5. RegimeModelingAgent
# ══════════════════════════════════════════════════════════════════


class RegimeModelingAgent(BaseAgent):
    """
    Classifies regimes using the ML-based regime model.

    Always enforces the safety floor: BROKEN/CRISIS cannot be overridden
    by ML output if break_confidence > 0.80.

    Task types
    ----------
    classify_regimes_ml
        Classify regimes for a set of entities using the ML model.
    evaluate_regime_model
        Evaluate regime model accuracy on provided labels.

    Required payload keys
    ---------------------
    features : dict | pd.DataFrame

    Optional payload keys
    ---------------------
    model_id : str
    as_of : str  (ISO date)

    Output keys
    -----------
    regime_classifications : list[dict]
    distribution : dict[str, int]
    safety_floor_applied_count : int
    """

    NAME = "regime_modeling"
    ALLOWED_TASK_TYPES = {"classify_regimes_ml", "evaluate_regime_model"}
    REQUIRED_PAYLOAD_KEYS = {"features"}

    # Heuristic regime classification from spread statistics
    @staticmethod
    def _rule_based_regime(
        spread_vals: Any,
        break_confidence: float = 0.0,
    ) -> tuple[str, float]:
        """Return (regime_str, confidence) using rule-based heuristics."""
        if break_confidence > 0.80:
            return "BROKEN", break_confidence
        try:
            import numpy as np
            arr = np.array(spread_vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) < 10:
                return "UNKNOWN", 0.3
            # Rough Hurst
            n = len(arr)
            dy = arr[1:] - arr[:-1]
            acf1 = float(np.corrcoef(dy[:-1], dy[1:])[0, 1]) if len(dy) > 2 else 0.0
            if acf1 < -0.15:
                return "MEAN_REVERTING", min(0.5 + abs(acf1), 0.9)
            elif acf1 > 0.10:
                return "TRENDING", min(0.5 + acf1, 0.9)
            return "TRANSITIONAL", 0.5
        except Exception:
            return "UNKNOWN", 0.3

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        features = task.payload["features"]
        model_id: str | None = task.payload.get("model_id")
        as_of = task.payload.get("as_of")

        audit.log(
            f"Classifying regimes via ML model, model_id={model_id}, as_of={as_of}"
        )

        _scorer = None
        try:
            from ml.inference.scorer import ModelScorer
            _scorer = ModelScorer(model_id=model_id) if model_id else ModelScorer()
            audit.log("Using ml.inference.scorer.ModelScorer for regime classification")
        except (ImportError, Exception) as exc:
            audit.warn(f"ModelScorer unavailable ({exc}) — using rule-based fallback")

        # Normalise features to list of entity dicts
        if isinstance(features, dict) and not any(
            isinstance(v, dict) for v in features.values()
        ):
            # Single entity represented as {feature_name: value}
            entities = [{"entity_id": "default", **features}]
        elif isinstance(features, dict):
            # {entity_id: {feature_name: value}}
            entities = [{"entity_id": k, **v} for k, v in features.items()]
        elif hasattr(features, "iterrows"):
            # DataFrame: each row is an entity
            entities = [{"entity_id": str(idx), **row.to_dict()} for idx, row in features.iterrows()]
        else:
            entities = [{"entity_id": "default"}]

        classifications: list[dict[str, Any]] = []
        safety_floor_count = 0

        for ent in entities:
            entity_id = str(ent.get("entity_id", "unknown"))
            break_conf = float(ent.get("break_confidence", 0.0))
            fallback_used = False

            if _scorer is not None:
                try:
                    result = _scorer.score(**{k: v for k, v in ent.items() if k != "entity_id"})
                    regime_str = str(getattr(result, "regime", "UNKNOWN"))
                    confidence = float(getattr(result, "confidence", 0.5))
                    fallback_used = getattr(result, "fallback_triggered", False)
                    if fallback_used:
                        spread_vals = ent.get("spread_values", [])
                        regime_str, confidence = self._rule_based_regime(spread_vals, break_conf)
                except Exception as exc:
                    audit.warn(f"{entity_id}: ModelScorer failed ({exc}), using rule-based")
                    spread_vals = ent.get("spread_values", [])
                    regime_str, confidence = self._rule_based_regime(spread_vals, break_conf)
                    fallback_used = True
            else:
                spread_vals = ent.get("spread_values", [])
                regime_str, confidence = self._rule_based_regime(spread_vals, break_conf)
                fallback_used = True

            # Enforce safety floor: ML cannot override BROKEN/CRISIS
            if break_conf > 0.80 and regime_str not in ("BROKEN", "CRISIS"):
                regime_str = "BROKEN"
                confidence = break_conf
                safety_floor_count += 1
                audit.warn(
                    f"{entity_id}: safety floor applied — break_confidence={break_conf:.3f}"
                )

            classifications.append(
                {
                    "entity": entity_id,
                    "regime": regime_str,
                    "confidence": round(confidence, 4),
                    "fallback_used": fallback_used,
                }
            )

        # Distribution
        distribution: dict[str, int] = {}
        for c in classifications:
            r = c["regime"]
            distribution[r] = distribution.get(r, 0) + 1

        audit.log(
            f"Regime modeling complete: {distribution}, "
            f"safety_floor_applied={safety_floor_count}"
        )

        return {
            "regime_classifications": classifications,
            "distribution": distribution,
            "safety_floor_applied_count": safety_floor_count,
        }


# ══════════════════════════════════════════════════════════════════
# 6. ModelRiskAgent
# ══════════════════════════════════════════════════════════════════


class ModelRiskAgent(BaseAgent):
    """
    Assesses operational risk of deployed ML models.

    Evaluates staleness, drift severity, and performance degradation.
    Classifies each model as LOW/MEDIUM/HIGH/CRITICAL risk.

    Task types
    ----------
    assess_model_risk
        Full risk assessment across a list of model IDs.
    check_model_staleness
        Focused staleness check only.

    Required payload keys
    ---------------------
    model_ids : list[str]

    Optional payload keys
    ---------------------
    health_status : dict[str, str]     ({model_id: "healthy"|"degraded"|...})
    drift_reports : list[dict]          (list of {model_id, psi_score, ...})
    age_days : dict[str, float]         ({model_id: days_since_training})
    stale_threshold_days : int          (default 90)

    Output keys
    -----------
    risk_assessments : dict[str, dict]
    high_risk_models : list[str]
    recommendations : list[str]
    """

    NAME = "model_risk"
    ALLOWED_TASK_TYPES = {"assess_model_risk", "check_model_staleness"}
    REQUIRED_PAYLOAD_KEYS = {"model_ids"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        model_ids: list[str] = task.payload["model_ids"]
        health_status: dict[str, str] = task.payload.get("health_status") or {}
        drift_reports: list[dict[str, Any]] = task.payload.get("drift_reports") or []
        age_days: dict[str, float] = task.payload.get("age_days") or {}
        stale_threshold: int = int(task.payload.get("stale_threshold_days", 90))

        audit.log(
            f"Assessing risk for {len(model_ids)} models, stale_threshold={stale_threshold}d"
        )

        # Index drift reports by model_id
        drift_by_model: dict[str, dict] = {}
        for dr in drift_reports:
            mid = dr.get("model_id")
            if mid:
                drift_by_model[mid] = dr

        risk_assessments: dict[str, dict[str, Any]] = {}
        high_risk_models: list[str] = []

        for mid in model_ids:
            reasons: list[str] = []
            risk_level = "LOW"

            # Staleness
            age = age_days.get(mid)
            if age is None:
                # Try to get from ML registry
                try:
                    from ml.registry.registry import get_ml_registry
                    reg = get_ml_registry()
                    mdl = reg.get(mid)
                    if mdl is not None:
                        trained_at = getattr(getattr(mdl, "metadata", None), "trained_at", None)
                        if trained_at:
                            try:
                                td = datetime.utcnow() - datetime.fromisoformat(
                                    str(trained_at).replace("Z", "")
                                )
                                age = td.days + td.seconds / 86400.0
                            except Exception:
                                pass
                except Exception:
                    pass

            if age is not None:
                if age > stale_threshold * 2:
                    risk_level = "CRITICAL"
                    reasons.append(f"stale:{age:.0f}d>>{stale_threshold*2}d")
                elif age > stale_threshold:
                    risk_level = max(risk_level, "HIGH", key=lambda x: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(x))
                    reasons.append(f"stale:{age:.0f}d>{stale_threshold}d")
                elif age > stale_threshold * 0.75:
                    if risk_level == "LOW":
                        risk_level = "MEDIUM"
                    reasons.append(f"approaching_stale:{age:.0f}d")

            # Health status
            health = health_status.get(mid, "unknown")
            if health == "critical":
                risk_level = "CRITICAL"
                reasons.append("health_status:critical")
            elif health == "degraded":
                if risk_level in ("LOW", "MEDIUM"):
                    risk_level = "HIGH"
                reasons.append("health_status:degraded")

            # Drift
            dr = drift_by_model.get(mid)
            if dr:
                psi = float(dr.get("psi_score", 0.0))
                if psi >= 0.25:
                    if risk_level == "LOW":
                        risk_level = "HIGH"
                    elif risk_level == "MEDIUM":
                        risk_level = "HIGH"
                    reasons.append(f"drift_psi:{psi:.3f}")
                elif psi >= 0.10:
                    if risk_level == "LOW":
                        risk_level = "MEDIUM"
                    reasons.append(f"moderate_drift_psi:{psi:.3f}")

            # Recommended action
            if risk_level == "CRITICAL":
                action = "IMMEDIATE_RETRAIN_OR_DISABLE"
            elif risk_level == "HIGH":
                action = "SCHEDULE_RETRAIN"
            elif risk_level == "MEDIUM":
                action = "MONITOR_CLOSELY"
            else:
                action = "NO_ACTION"

            if risk_level in ("HIGH", "CRITICAL"):
                high_risk_models.append(mid)

            risk_assessments[mid] = {
                "risk_level": risk_level,
                "reasons": reasons,
                "recommended_action": action,
                "age_days": round(age, 1) if age is not None else None,
                "health_status": health,
            }

        recommendations: list[str] = []
        if high_risk_models:
            recommendations.append(
                f"{len(high_risk_models)} high/critical risk model(s): {high_risk_models}. "
                "Immediate review required."
            )
        n_stale = sum(
            1 for a in risk_assessments.values()
            if any("stale" in r and "approaching" not in r for r in a["reasons"])
        )
        if n_stale:
            recommendations.append(f"{n_stale} stale model(s) detected — trigger retraining pipeline.")

        audit.log(
            f"Model risk assessment complete: {len(high_risk_models)} high-risk models"
        )

        return {
            "risk_assessments": risk_assessments,
            "high_risk_models": high_risk_models,
            "recommendations": recommendations,
        }


# ══════════════════════════════════════════════════════════════════
# 7. PromotionReviewAgent
# ══════════════════════════════════════════════════════════════════


class PromotionReviewAgent(BaseAgent):
    """
    Conducts a governance review for promoting an ML model or other subject.

    Checks promotion criteria (IC/AUC/Brier for ML models) and produces a
    typed ``PromotionReviewRecord``.

    Task types
    ----------
    review_promotion
        Full governance review of a promotion request.
    check_promotion_criteria
        Check only the metrics without creating a formal review record.

    Required payload keys
    ---------------------
    subject_type : str
    subject_id : str
    subject_name : str
    metrics : dict
    current_status : str
    proposed_status : str
    evidence_ids : list[str]

    Output keys
    -----------
    promotion_review : dict
    approved : bool
    criteria_passed : list[str]
    criteria_failed : list[str]
    requires_human_approval : bool
    """

    NAME = "promotion_review"
    ALLOWED_TASK_TYPES = {"review_promotion", "check_promotion_criteria"}
    REQUIRED_PAYLOAD_KEYS = {
        "subject_type", "subject_id", "subject_name",
        "current_status", "proposed_status",
    }

    # Valid promotion transitions
    _VALID_TRANSITIONS: set[tuple[str, str]] = {
        ("CANDIDATE", "CHALLENGER"),
        ("CHALLENGER", "CHAMPION"),
        ("CHAMPION", "RETIRED"),
        ("CANDIDATE", "RETIRED"),
        ("CHALLENGER", "RETIRED"),
        ("DRAFT", "ACTIVE"),
        ("ACTIVE", "DEPRECATED"),
        ("ACTIVE", "SUSPENDED"),
    }

    def _check_metrics(
        self,
        subject_type: str,
        metrics: dict[str, float],
    ) -> tuple[list[str], list[str]]:
        """Return (criteria_passed, criteria_failed)."""
        passed: list[str] = []
        failed: list[str] = []

        stype = subject_type.upper()
        if stype in ("REGIME_MODEL",):
            thresholds = _REGIME_THRESHOLDS
        elif stype in ("BREAK_DETECTOR",):
            thresholds = _BREAK_THRESHOLDS
        else:
            thresholds = _ML_THRESHOLDS

        ic = metrics.get("ic")
        auc = metrics.get("auc")
        brier = metrics.get("brier")

        if ic is not None:
            criterion = f"ic>={thresholds['ic_min']}"
            if float(ic) >= thresholds["ic_min"]:
                passed.append(criterion)
            else:
                failed.append(f"{criterion} (got {ic:.4f})")

        if auc is not None:
            criterion = f"auc>={thresholds['auc_min']}"
            if float(auc) >= thresholds["auc_min"]:
                passed.append(criterion)
            else:
                failed.append(f"{criterion} (got {auc:.4f})")

        if brier is not None:
            criterion = f"brier<={thresholds['brier_max']}"
            if float(brier) <= thresholds["brier_max"]:
                passed.append(criterion)
            else:
                failed.append(f"{criterion} (got {brier:.4f})")

        return passed, failed

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        subject_type: str = task.payload["subject_type"]
        subject_id: str = task.payload["subject_id"]
        subject_name: str = task.payload["subject_name"]
        metrics: dict[str, float] = task.payload.get("metrics") or {}
        current_status: str = task.payload["current_status"]
        proposed_status: str = task.payload["proposed_status"]
        evidence_ids: list[str] = task.payload.get("evidence_ids") or []

        audit.log(
            f"Reviewing promotion: {subject_name} ({subject_type}) "
            f"{current_status} → {proposed_status}"
        )

        # Validate transition
        transition = (current_status.upper(), proposed_status.upper())
        transition_valid = transition in self._VALID_TRANSITIONS

        if not transition_valid:
            audit.warn(f"Invalid transition {transition} — blocking promotion")

        # Check metrics
        criteria_passed, criteria_failed = self._check_metrics(subject_type, metrics)

        # Determine decision
        approved = transition_valid and len(criteria_failed) == 0
        decision = "APPROVED" if approved else ("REJECTED" if criteria_failed else "DEFERRED")

        if not transition_valid:
            criteria_failed.append(f"invalid_transition:{current_status}→{proposed_status}")
            decision = "REJECTED"
            approved = False

        # Human approval required for CHAMPION promotions in production
        requires_human = proposed_status.upper() == "CHAMPION"

        rationale_parts: list[str] = []
        if approved:
            rationale_parts.append(
                f"All {len(criteria_passed)} governance criteria met for "
                f"{subject_type} promotion {current_status}→{proposed_status}."
            )
        else:
            rationale_parts.append(
                f"Promotion blocked: {len(criteria_failed)} criteria failed."
            )
        rationale = " ".join(rationale_parts)

        # Build PromotionReviewRecord
        review_id = _new_id()
        try:
            from governance.contracts import PromotionReviewRecord
            record = PromotionReviewRecord(
                review_id=review_id,
                subject_type=subject_type,
                subject_id=subject_id,
                subject_name=subject_name,
                reviewed_by=self.NAME,
                reviewed_at=_utcnow(),
                current_status=current_status,
                proposed_status=proposed_status,
                evidence_bundle_ids=tuple(evidence_ids),
                criteria_passed=tuple(criteria_passed),
                criteria_failed=tuple(criteria_failed),
                decision=decision,
                rationale=rationale,
                conditions=(),
                approval_request_id=None,
            )
            review_dict = {
                "review_id": record.review_id,
                "subject_type": record.subject_type,
                "subject_id": record.subject_id,
                "subject_name": record.subject_name,
                "reviewed_by": record.reviewed_by,
                "reviewed_at": record.reviewed_at,
                "current_status": record.current_status,
                "proposed_status": record.proposed_status,
                "evidence_bundle_ids": list(record.evidence_bundle_ids),
                "criteria_passed": list(record.criteria_passed),
                "criteria_failed": list(record.criteria_failed),
                "decision": record.decision,
                "rationale": record.rationale,
                "conditions": list(record.conditions),
                "approval_request_id": record.approval_request_id,
            }
        except ImportError:
            review_dict = {
                "review_id": review_id,
                "subject_type": subject_type,
                "subject_id": subject_id,
                "subject_name": subject_name,
                "reviewed_by": self.NAME,
                "reviewed_at": _utcnow(),
                "current_status": current_status,
                "proposed_status": proposed_status,
                "evidence_bundle_ids": evidence_ids,
                "criteria_passed": criteria_passed,
                "criteria_failed": criteria_failed,
                "decision": decision,
                "rationale": rationale,
                "conditions": [],
                "approval_request_id": None,
            }

        audit.log(
            f"Promotion review complete: decision={decision}, "
            f"passed={criteria_passed}, failed={criteria_failed}, "
            f"requires_human={requires_human}"
        )

        return {
            "promotion_review": review_dict,
            "approved": approved,
            "criteria_passed": criteria_passed,
            "criteria_failed": criteria_failed,
            "requires_human_approval": requires_human,
        }
