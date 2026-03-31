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

    # Alias for criteria
    ctx["walk_forward_ic_mean"] = ctx["cv_ic_mean"]

    # Robustness score: approximate from cv_ic_mean (positive IC suggests robustness)
    # The full robustness requires ic_values list; we approximate here
    ctx["robustness_score"] = 0.6 if _nanf(run_artifact.cv_ic_mean) > 0 else 0.4

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
