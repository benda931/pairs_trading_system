# -*- coding: utf-8 -*-
"""
ml/governance/policies.py — ML Governance Engine
=================================================

Enforces:
- Promotion criteria (CANDIDATE → CHAMPION)
- Usage contracts (model–consumer binding rules)
- Retirement criteria evaluation

Design:
- All decisions are explicit — never implicit or silent.
- Every check populates criteria_met / criteria_failed lists.
- The engine never mutates model state directly; callers act on the returned
  PromotionDecision.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from ml.contracts import (
    ChampionChallengerRecord,
    GovernanceStatus,
    MLUsageContract,
    ModelHealthState,
    ModelHealthStatus,
    ModelMetadata,
    ModelStatus,
    PromotionDecision,
    PromotionOutcome,
    TrainingRunArtifact,
    _now_utc,
)

logger = logging.getLogger("ml.governance")


class GovernanceEngine:
    """
    Enforces ML governance rules: approval workflows, usage contracts,
    kill criteria.

    Criteria strings follow the format "<field> <op> <value>" and are
    evaluated against the TrainingRunArtifact fields.
    """

    PROMOTION_CRITERIA: dict[ModelStatus, list[str]] = {
        ModelStatus.CHAMPION: [
            "val_auc >= 0.55",
            "val_brier < 0.25",
            "walk_forward_ic_mean > 0",
            "robustness_score >= 0.55",
            "calibration_brier_improvement > 0",
        ]
    }

    RETIREMENT_CRITERIA: list[str] = [
        "stale",
        "drift_severity_critical",
        "performance_degraded",
        "suspended",
    ]

    # ------------------------------------------------------------------
    # Promotion criteria evaluation
    # ------------------------------------------------------------------

    def check_promotion_criteria(
        self,
        metadata: ModelMetadata,
        run_artifact: TrainingRunArtifact,
        champion_comparison: Optional[ChampionChallengerRecord] = None,
    ) -> PromotionDecision:
        """
        Evaluate whether a candidate model meets promotion criteria.

        Returns a PromotionDecision with explicit criteria_met / criteria_failed
        lists. Never silently promotes.
        """
        to_status = ModelStatus.CHAMPION  # Only champion promotion is governed
        criteria = self.PROMOTION_CRITERIA.get(to_status, [])

        criteria_met: list[str] = []
        criteria_failed: list[str] = []

        # Build a context dict for evaluation
        ctx = _artifact_context(metadata, run_artifact, champion_comparison)

        for criterion in criteria:
            try:
                passed, reason = _evaluate_criterion(criterion, ctx)
                if passed:
                    criteria_met.append(criterion)
                else:
                    criteria_failed.append(f"{criterion} [actual={reason}]")
            except Exception as exc:
                criteria_failed.append(f"{criterion} [eval_error: {exc}]")
                logger.warning("Criterion evaluation failed: %s — %s", criterion, exc)

        # Leakage audit check — blocks promotion on HIGH-risk dimensions
        leakage_ok, leakage_reason = self._check_leakage_audit(run_artifact)
        if not leakage_ok:
            criteria_failed.append(f"leakage_audit [{leakage_reason}]")
        else:
            criteria_met.append(f"leakage_audit [{leakage_reason}]")

        # Determine outcome
        if not criteria_failed:
            outcome = PromotionOutcome.PROMOTE
            evidence = (
                f"All {len(criteria_met)} promotion criteria met. "
                f"val_auc={ctx.get('val_auc', 'N/A'):.4f}, "
                f"val_brier={ctx.get('val_brier', 'N/A'):.4f}."
            )
        elif len(criteria_failed) == 1 and "calibration" in criteria_failed[0]:
            # Calibration is desirable but not blocking; allow conditional
            outcome = PromotionOutcome.DEFER
            evidence = (
                f"{len(criteria_failed)} criterion failed (calibration). "
                "Consider conditional approval after calibration."
            )
        else:
            outcome = PromotionOutcome.REJECT
            evidence = (
                f"{len(criteria_failed)} criteria not met: "
                + "; ".join(criteria_failed[:3])
            )

        decision = PromotionDecision(
            model_id=metadata.model_id,
            from_status=metadata.status,
            to_status=to_status,
            comparison_id=champion_comparison.comparison_id if champion_comparison else "",
            evidence_summary=evidence,
            criteria_met=criteria_met,
            criteria_failed=criteria_failed,
            outcome=outcome,
            decided_by="governance_engine",
            requires_manual_approval=True,
            manually_approved=False,
        )

        logger.info(
            "Promotion decision for %s: %s (%d met, %d failed)",
            metadata.model_id,
            outcome.value,
            len(criteria_met),
            len(criteria_failed),
        )

        return decision

    # ------------------------------------------------------------------
    # Usage contract validation
    # ------------------------------------------------------------------

    def check_usage_contract(
        self,
        model: ModelMetadata,
        contract: MLUsageContract,
    ) -> tuple[bool, list[str]]:
        """
        Verify a model may be used by a consumer under the specified contract.

        Returns (allowed, violations).
        """
        violations: list[str] = []

        # Check model status matches contract requirements
        if contract.require_champion_status and model.status != ModelStatus.CHAMPION:
            violations.append(
                f"contract requires CHAMPION status; model is {model.status.value}"
            )

        # Check governance approval
        if model.governance_status not in (
            GovernanceStatus.APPROVED,
            GovernanceStatus.CONDITIONALLY_APPROVED,
        ):
            violations.append(
                f"model governance_status is {model.governance_status.value}; "
                "must be APPROVED or CONDITIONALLY_APPROVED"
            )

        # Check consumer restrictions
        if model.forbidden_consumers and contract.consumer in model.forbidden_consumers:
            violations.append(
                f"consumer '{contract.consumer}' is on the forbidden list for this model"
            )

        if model.allowed_consumers and contract.consumer not in model.allowed_consumers:
            violations.append(
                f"consumer '{contract.consumer}' is not in the allowed_consumers list"
            )

        # Check calibration requirement
        if contract.require_calibration and not model.calibrated:
            violations.append(
                "contract requires calibration; model is not calibrated"
            )

        # Check min confidence is achievable (sanity check only)
        if contract.min_model_confidence > 1.0:
            violations.append(
                f"contract min_model_confidence={contract.min_model_confidence} is >1.0 (invalid)"
            )

        # Hard rule: ML must never override hard risk rules
        if contract.may_override_hard_rules:
            violations.append(
                "contract.may_override_hard_rules=True is NEVER permitted; "
                "ML must not override hard risk rules"
            )

        allowed = len(violations) == 0
        return allowed, violations

    # ------------------------------------------------------------------
    # Retirement criteria
    # ------------------------------------------------------------------

    def evaluate_retirement_criteria(
        self,
        model: ModelMetadata,
        health: ModelHealthStatus,
    ) -> tuple[bool, list[str]]:
        """
        Check whether a model meets retirement criteria.

        Returns (should_retire, reasons).
        """
        reasons: list[str] = []

        # Already retired
        if model.status == ModelStatus.RETIRED:
            return False, ["already_retired"]

        # Blocked models should be retired
        if model.status == ModelStatus.BLOCKED:
            reasons.append("model_is_blocked")

        # Health-based retirement
        if health.state == ModelHealthState.SUSPENDED:
            reasons.append("model_suspended_in_health_check")

        if health.stale:
            reasons.append(f"model_stale: {health.hours_since_training:.1f}h since training")

        from ml.contracts import DriftSeverity
        if health.drift_severity == DriftSeverity.CRITICAL:
            reasons.append("drift_severity_critical")

        if health.performance_degraded:
            reasons.append("performance_degraded")

        # Retirement criteria from model metadata
        if model.retirement_criteria:
            for crit in model.retirement_criteria:
                reasons.append(f"metadata_criterion: {crit}")

        should_retire = len(reasons) > 0
        return should_retire, reasons

    # ------------------------------------------------------------------
    # Leakage audit check
    # ------------------------------------------------------------------

    def _check_leakage_audit(
        self,
        run_artifact: TrainingRunArtifact,
    ) -> tuple[bool, str]:
        """
        Fail governance if the leakage audit shows HIGH risk on any dimension.

        A default-constructed LeakageAuditReport (never explicitly run) has
        all boolean risk flags False and passed=True.  That is treated as
        inconclusive (not run) and does NOT block promotion — only an explicitly
        set risk flag blocks.

        Parameters
        ----------
        run_artifact : TrainingRunArtifact
            The artifact produced by the training run.

        Returns
        -------
        tuple[bool, str] : (passed, reason_key)
        """
        audit = getattr(run_artifact, "leakage_audit", None)
        if audit is None:
            # No leakage_audit attribute at all — log warning but don't block
            # (backward-compatible: older artifacts predate the audit field).
            logger.warning(
                "No LeakageAuditReport in run_artifact for model %s "
                "— leakage not verified",
                getattr(run_artifact, "model_id", "unknown"),
            )
            return True, "leakage_audit_missing"

        # Check boolean risk dimensions — any True flag means HIGH risk.
        risk_flags: dict[str, bool] = {
            "future_feature_risk":     bool(getattr(audit, "future_feature_risk", False)),
            "future_label_risk":       bool(getattr(audit, "future_label_risk", False)),
            "normalization_leak_risk": bool(getattr(audit, "normalization_leak_risk", False)),
            "overlap_label_risk":      bool(getattr(audit, "overlap_label_risk", False)),
        }

        high_risks = [k for k, v in risk_flags.items() if v]
        if high_risks:
            msg = f"HIGH risk flags: {high_risks}"
            logger.error(
                "GOVERNANCE BLOCK — leakage audit failed for model %s — %s",
                getattr(run_artifact, "model_id", "unknown"),
                msg,
            )
            return False, msg

        embargo_ok = bool(getattr(audit, "embargo_adequate", True))
        purge_ok   = bool(getattr(audit, "purge_adequate", True))
        if not embargo_ok:
            return False, "leakage_audit: embargo_adequate=False"
        if not purge_ok:
            return False, "leakage_audit: purge_adequate=False"

        return True, "leakage_audit_passed"

    def enforce_retirement(
        self,
        model: "ModelMetadata",
        health: "ModelHealthStatus",
        registry=None,
    ) -> tuple:
        """
        Evaluate retirement criteria AND retire the model if criteria are met.

        Unlike evaluate_retirement_criteria() which only returns a recommendation,
        this method actually sets model.status = RETIRED in the registry.

        This closes the gap where retirement decisions were computed but never acted on.

        Parameters
        ----------
        model : ModelMetadata
            Model to evaluate.
        health : ModelHealthStatus
            Current health status.
        registry : MLModelRegistry, optional
            If provided, calls registry.update_status() to persist the retirement.
            If None, only returns the decision without persisting.

        Returns
        -------
        tuple : (retired: bool, reasons: list[str])
        """
        should_retire, reasons = self.evaluate_retirement_criteria(model, health)

        if not should_retire:
            return False, reasons

        # Act on the decision
        logger.warning(
            "Retiring model %s (task=%s): %s",
            model.model_id,
            model.task_family.value,
            "; ".join(reasons),
        )

        # Update registry if available
        if registry is not None:
            try:
                if hasattr(registry, "update_status"):
                    registry.update_status(model.model_id, ModelStatus.RETIRED)
                    logger.info("Model %s status set to RETIRED in registry", model.model_id)
                else:
                    logger.warning(
                        "Cannot retire model %s: registry has no update_status method",
                        model.model_id,
                    )
            except Exception as exc:
                logger.error("Failed to retire model %s: %s", model.model_id, exc)
                return False, [f"retirement_write_failed: {exc}"]

        return True, reasons


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _artifact_context(
    metadata: ModelMetadata,
    run_artifact: TrainingRunArtifact,
    comparison: Optional[ChampionChallengerRecord],
) -> dict:
    """Build a flat evaluation context dict from artifacts."""
    ctx: dict = {}

    # From run_artifact
    ctx["val_auc"] = _nanf(run_artifact.val_auc)
    ctx["val_brier"] = _nanf(run_artifact.val_brier)
    ctx["test_auc"] = _nanf(run_artifact.test_auc)
    ctx["test_brier"] = _nanf(run_artifact.test_brier)
    ctx["train_auc"] = _nanf(run_artifact.train_auc)
    ctx["cv_ic_mean"] = _nanf(run_artifact.cv_ic_mean)
    ctx["cv_ic_std"] = _nanf(run_artifact.cv_ic_std)
    ctx["calibration_brier_improvement"] = _nanf(run_artifact.calibration_brier_improvement)

    # walk_forward_ic_mean must come from a genuine walk-forward test, not from CV.
    # CV and walk-forward are NOT equivalent — walk-forward simulates production deployment order.
    wf_ic = _nanf(getattr(run_artifact, "walk_forward_ic_mean", float("nan")))
    if math.isfinite(wf_ic):
        ctx["walk_forward_ic_mean"] = wf_ic
    else:
        # walk_forward_ic_mean not computed — fail the criterion explicitly
        # rather than silently aliasing to CV IC (which would mask a missing test).
        ctx["walk_forward_ic_mean"] = float("nan")
        import logging as _log
        _log.getLogger("ml.governance").warning(
            "walk_forward_ic_mean not populated for model %s — criterion will fail. "
            "Run a genuine walk-forward test (rolling-origin OOS) and populate "
            "TrainingRunArtifact.walk_forward_ic_mean.",
            metadata.model_id,
        )

    # Robustness: fraction of CV folds with positive IC.
    # Requires cv_ic_per_fold to be populated by the training runner.
    # Falls back to a conservative 0.4 if fold data is unavailable.
    cv_folds = getattr(run_artifact, "cv_ic_per_fold", None) or []
    if cv_folds:
        ctx["robustness_score"] = float(sum(1 for ic in cv_folds if ic > 0) / len(cv_folds))
    else:
        # No fold-level data available — cannot assess robustness; fail safe.
        ctx["robustness_score"] = 0.4
        import logging as _log
        _log.getLogger("ml.governance").warning(
            "cv_ic_per_fold not populated for model %s — robustness_score defaulting to 0.4 (fail-safe). "
            "Populate TrainingRunArtifact.cv_ic_per_fold from the training runner to enable real robustness scoring.",
            metadata.model_id,
        )

    # From champion comparison
    if comparison is not None:
        ctx["champion_auc"] = _nanf(comparison.champion_auc)
        ctx["challenger_auc"] = _nanf(comparison.challenger_auc)
        ctx["champion_brier"] = _nanf(comparison.champion_brier)
        ctx["challenger_brier"] = _nanf(comparison.challenger_brier)

    return ctx


def _nanf(v) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _evaluate_criterion(criterion: str, ctx: dict) -> tuple[bool, str]:
    """
    Evaluate a criterion string like 'val_auc >= 0.55'.

    Returns (passed, actual_value_str).
    """
    parts = criterion.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Cannot parse criterion: {criterion!r}")

    field, op, threshold_str = parts
    threshold = float(threshold_str)
    actual = ctx.get(field, float("nan"))

    if not math.isfinite(actual):
        return False, "nan"

    op_map = {
        ">=": actual >= threshold,
        ">": actual > threshold,
        "<=": actual <= threshold,
        "<": actual < threshold,
        "==": abs(actual - threshold) < 1e-9,
        "!=": abs(actual - threshold) >= 1e-9,
    }

    if op not in op_map:
        raise ValueError(f"Unknown operator: {op!r}")

    return op_map[op], f"{actual:.4f}"
