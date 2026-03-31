# -*- coding: utf-8 -*-
"""
policies/registry.py — Versioned policy definitions, evaluation, and conformance reporting.

PolicyRegistry ships with 4 canonical system-wide policies (8 rules total).
Policies are versioned; only ACTIVE versions are evaluated.
register_policy() supersedes the previous version automatically.
evaluate() runs all active rules against an entity + context dict and returns
a PolicyConformanceReport.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PolicyScope(Enum):
    """Organisational scope that a policy applies to."""
    SYSTEM_WIDE = "SYSTEM_WIDE"
    STRATEGY = "STRATEGY"
    MODEL = "MODEL"
    AGENT = "AGENT"
    DEPLOYMENT = "DEPLOYMENT"
    DATA = "DATA"
    RISK = "RISK"
    COMPLIANCE = "COMPLIANCE"
    GOVERNANCE = "GOVERNANCE"


class PolicyRuleType(Enum):
    """Nature of a policy rule and how it is enforced."""
    HARD_LIMIT = "HARD_LIMIT"           # blocking; violation prevents the action
    SOFT_LIMIT = "SOFT_LIMIT"           # advisory; violation is logged/warned
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"  # requires explicit approval before proceeding
    AUDIT_REQUIRED = "AUDIT_REQUIRED"   # must produce an audit record
    NOTIFY_REQUIRED = "NOTIFY_REQUIRED" # must notify named parties
    PROHIBITED = "PROHIBITED"           # action is categorically disallowed
    MANDATORY = "MANDATORY"             # action must be taken (omission is the violation)


class PolicyLifecycleState(Enum):
    """Lifecycle state of a policy version."""
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"
    SUPERSEDED = "SUPERSEDED"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyRule:
    """
    A single immutable rule within a policy version.

    ``entity_filter`` is a glob-style string used for matching entity
    identifiers (e.g., ``"strategy:*"``, ``"model:regime_*"``).
    ``automated_enforcement`` distinguishes engine-enforced rules from
    advisory/procedural ones.
    """
    rule_id: str
    name: str
    description: str
    rule_type: PolicyRuleType
    scope: PolicyScope
    entity_filter: str              # glob-style filter, e.g. "strategy:*" or "model:regime_*"
    condition: str                  # human-readable condition description
    action_on_violation: str        # what should happen on violation (description)
    severity: str                   # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    automated_enforcement: bool     # True = engine enforces; False = advisory only


@dataclass(frozen=True)
class PolicyVersion:
    """
    An immutable version of a policy document containing one or more rules.

    When a new version is registered, the previous version transitions to
    SUPERSEDED with ``effective_until`` set.
    """
    policy_id: str
    version: int
    rules: tuple[PolicyRule, ...]
    effective_from: datetime
    effective_until: Optional[datetime]
    authored_by: str
    change_summary: str
    approved_by: Optional[str]
    state: PolicyLifecycleState


@dataclass(frozen=True)
class PolicyEvaluationResult:
    """
    Outcome of evaluating a single policy rule against an entity.

    ``blocking`` is True when the rule type is HARD_LIMIT or PROHIBITED
    and the evaluation did not pass — indicating a hard stop.
    """
    policy_id: str
    rule_id: str
    entity_type: str
    entity_id: str
    passed: bool
    rule_type: PolicyRuleType
    violation_description: Optional[str]
    blocking: bool                  # True = hard stop; False = advisory
    evaluated_at: datetime


@dataclass(frozen=True)
class PolicyConformanceReport:
    """
    Summary of all policy rule evaluations for a single entity.

    ``is_conformant`` is True only when ``blocking_violations == 0``.
    """
    entity_type: str
    entity_id: str
    total_rules_evaluated: int
    passed: int
    failed: int
    blocking_violations: int
    advisory_violations: int
    results: tuple[PolicyEvaluationResult, ...]
    is_conformant: bool             # True only if zero blocking violations
    generated_at: datetime


# ---------------------------------------------------------------------------
# PolicyRegistry
# ---------------------------------------------------------------------------

class PolicyRegistry:
    """
    Registry of versioned PolicyVersions with evaluation and conformance reporting.

    Policies are versioned; only ACTIVE versions are evaluated.
    Each policy can have multiple rules; rules are evaluated independently.
    register_policy() supersedes the previous ACTIVE version automatically.

    Thread safety: this class is not internally synchronised.
    """

    def __init__(self) -> None:
        self._policies: dict[str, list[PolicyVersion]] = {}  # policy_id → version history
        self._active: dict[str, PolicyVersion] = {}          # policy_id → current active version
        self._evaluation_log: list[PolicyEvaluationResult] = []
        self._populate_defaults()

    # ------------------------------------------------------------------
    # Default policies
    # ------------------------------------------------------------------

    def _populate_defaults(self) -> None:
        """Register 4 canonical system-wide policies (8 rules total)."""
        now = datetime.utcnow()

        policies_spec: list[tuple[str, str, list[PolicyRule], str]] = [
            (
                "POL-ML-001",
                "ML Model Governance",
                [
                    PolicyRule(
                        "POL-ML-001-R1",
                        "Champion requires governance approval",
                        "Model metrics must meet IC>=0.05, AUC>=0.55, Brier<=0.25 before promotion",
                        PolicyRuleType.HARD_LIMIT,
                        PolicyScope.MODEL,
                        "model:*",
                        "metrics below promotion thresholds",
                        "block promotion",
                        "CRITICAL",
                        True,
                    ),
                    PolicyRule(
                        "POL-ML-001-R2",
                        "ML cannot override risk limits",
                        "ML hooks may not bypass kill-switch or drawdown manager hard vetoes",
                        PolicyRuleType.PROHIBITED,
                        PolicyScope.MODEL,
                        "model:*",
                        "ML override of hard risk rule",
                        "reject override attempt",
                        "CRITICAL",
                        True,
                    ),
                ],
                "ml_governance",
            ),
            (
                "POL-TRADE-001",
                "Trading Activity Controls",
                [
                    PolicyRule(
                        "POL-TRADE-001-R1",
                        "No live orders in BROKEN/CRISIS regime",
                        "Block new entries when regime classification is BROKEN or CRISIS",
                        PolicyRuleType.PROHIBITED,
                        PolicyScope.STRATEGY,
                        "strategy:*",
                        "entry attempted in prohibited regime",
                        "block entry",
                        "CRITICAL",
                        True,
                    ),
                    PolicyRule(
                        "POL-TRADE-001-R2",
                        "Signal quality grade F blocks portfolio routing",
                        "Grade F signals must not be funded by the portfolio layer",
                        PolicyRuleType.PROHIBITED,
                        PolicyScope.STRATEGY,
                        "strategy:*",
                        "grade F signal routed to portfolio",
                        "reject intent",
                        "CRITICAL",
                        True,
                    ),
                ],
                "risk_team",
            ),
            (
                "POL-DEPL-001",
                "Deployment Controls",
                [
                    PolicyRule(
                        "POL-DEPL-001-R1",
                        "Deployed != Activated invariant",
                        "A strategy must be explicitly activated after deployment; deployed state is not active",
                        PolicyRuleType.MANDATORY,
                        PolicyScope.DEPLOYMENT,
                        "deployment:*",
                        "activation without explicit activation step",
                        "block activation",
                        "CRITICAL",
                        True,
                    ),
                    PolicyRule(
                        "POL-DEPL-001-R2",
                        "Live trading requires dual approval",
                        "Two independent approvers are required for any live-capital activation",
                        PolicyRuleType.APPROVAL_REQUIRED,
                        PolicyScope.DEPLOYMENT,
                        "deployment:live_*",
                        "single-approver live activation",
                        "require second approver",
                        "CRITICAL",
                        False,
                    ),
                ],
                "devops",
            ),
            (
                "POL-DATA-001",
                "Data Quality Standards",
                [
                    PolicyRule(
                        "POL-DATA-001-R1",
                        "Price data must be fresh",
                        "Data older than 24h blocks signal computation to prevent stale-data trades",
                        PolicyRuleType.HARD_LIMIT,
                        PolicyScope.DATA,
                        "data:prices_*",
                        "stale data used for signal computation",
                        "block computation",
                        "HIGH",
                        True,
                    ),
                    PolicyRule(
                        "POL-DATA-001-R2",
                        "Train/test boundary is sacred",
                        "Model parameters must never be estimated on test data",
                        PolicyRuleType.PROHIBITED,
                        PolicyScope.MODEL,
                        "model:*",
                        "test data used for parameter estimation",
                        "raise ValueError",
                        "CRITICAL",
                        True,
                    ),
                ],
                "data_team",
            ),
        ]

        for policy_id, _name, rules, author in policies_spec:
            version = PolicyVersion(
                policy_id=policy_id,
                version=1,
                rules=tuple(rules),
                effective_from=now,
                effective_until=None,
                authored_by=author,
                change_summary="Initial version",
                approved_by="system",
                state=PolicyLifecycleState.ACTIVE,
            )
            self._policies[policy_id] = [version]
            self._active[policy_id] = version

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_policy(
        self,
        policy_id: str,
        rules: list[PolicyRule],
        authored_by: str,
        change_summary: str,
        approved_by: Optional[str] = None,
    ) -> PolicyVersion:
        """
        Register a new version of a policy.

        If a previous ACTIVE version exists it is transitioned to
        SUPERSEDED with ``effective_until`` set to now.
        """
        existing = self._policies.get(policy_id, [])
        version_num = len(existing) + 1
        now = datetime.utcnow()

        # Supersede the previous active version
        if policy_id in self._active:
            prev = self._active[policy_id]
            superseded = PolicyVersion(
                policy_id=prev.policy_id,
                version=prev.version,
                rules=prev.rules,
                effective_from=prev.effective_from,
                effective_until=now,
                authored_by=prev.authored_by,
                change_summary=prev.change_summary,
                approved_by=prev.approved_by,
                state=PolicyLifecycleState.SUPERSEDED,
            )
            if existing:
                existing[-1] = superseded
            else:
                existing.append(superseded)

        new_version = PolicyVersion(
            policy_id=policy_id,
            version=version_num,
            rules=tuple(rules),
            effective_from=now,
            effective_until=None,
            authored_by=authored_by,
            change_summary=change_summary,
            approved_by=approved_by,
            state=PolicyLifecycleState.ACTIVE,
        )
        existing.append(new_version)
        self._policies[policy_id] = existing
        self._active[policy_id] = new_version
        return new_version

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        entity_type: str,
        entity_id: str,
        context: dict[str, Any],
    ) -> PolicyConformanceReport:
        """
        Evaluate all active policy rules against an entity and context.

        Context-based rule resolution:
        The *context* dict may supply explicit pass/fail overrides using keys
        of the form ``"rule_<rule_id>_passed"`` (bool).  Absent overrides
        default to passing (True) so that callers only need to supply
        failing signals.

        Scope matching:
        A rule is evaluated when ``rule.scope.value.lower()`` appears in
        ``entity_type.lower()``, or when the scope is SYSTEM_WIDE.
        """
        results: list[PolicyEvaluationResult] = []
        now = datetime.utcnow()

        for policy in self._active.values():
            if policy.state != PolicyLifecycleState.ACTIVE:
                continue
            for rule in policy.rules:
                # Scope matching
                scope_match = (
                    rule.scope == PolicyScope.SYSTEM_WIDE
                    or rule.scope.value.lower() in entity_type.lower()
                )
                if not scope_match:
                    continue

                # Context-driven pass/fail (default: pass)
                override_key = f"rule_{rule.rule_id}_passed"
                passed: bool = bool(context.get(override_key, True))

                violation_desc: Optional[str] = None
                if not passed:
                    violation_desc = (
                        f"Rule '{rule.name}' violated: {rule.condition}"
                    )

                blocking = (
                    not passed
                    and rule.rule_type in (
                        PolicyRuleType.HARD_LIMIT,
                        PolicyRuleType.PROHIBITED,
                    )
                )

                result = PolicyEvaluationResult(
                    policy_id=policy.policy_id,
                    rule_id=rule.rule_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    passed=passed,
                    rule_type=rule.rule_type,
                    violation_description=violation_desc,
                    blocking=blocking,
                    evaluated_at=now,
                )
                results.append(result)
                self._evaluation_log.append(result)

        passed_count = sum(1 for r in results if r.passed)
        failed_count = sum(1 for r in results if not r.passed)
        blocking_count = sum(1 for r in results if r.blocking)
        advisory_count = sum(1 for r in results if not r.passed and not r.blocking)

        return PolicyConformanceReport(
            entity_type=entity_type,
            entity_id=entity_id,
            total_rules_evaluated=len(results),
            passed=passed_count,
            failed=failed_count,
            blocking_violations=blocking_count,
            advisory_violations=advisory_count,
            results=tuple(results),
            is_conformant=blocking_count == 0,
            generated_at=now,
        )

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def suspend_policy(self, policy_id: str, reason: str) -> None:
        """
        Suspend an active policy.

        The suspended version remains in ``_active`` but with state
        SUSPENDED so that evaluate() skips it.
        """
        if policy_id not in self._active:
            return
        p = self._active[policy_id]
        self._active[policy_id] = PolicyVersion(
            policy_id=p.policy_id,
            version=p.version,
            rules=p.rules,
            effective_from=p.effective_from,
            effective_until=p.effective_until,
            authored_by=p.authored_by,
            change_summary=p.change_summary + f" [SUSPENDED: {reason}]",
            approved_by=p.approved_by,
            state=PolicyLifecycleState.SUSPENDED,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_policies(self) -> list[PolicyVersion]:
        """Return all currently ACTIVE policy versions."""
        return [p for p in self._active.values() if p.state == PolicyLifecycleState.ACTIVE]

    def get_policy_history(self, policy_id: str) -> list[PolicyVersion]:
        """Return the full version history for a policy (oldest first)."""
        return list(self._policies.get(policy_id, []))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, int]:
        """Return a snapshot of policy health metrics."""
        active = self.get_active_policies()
        total_rules = sum(len(p.rules) for p in active)
        violations = sum(1 for r in self._evaluation_log if not r.passed)
        return {
            "active_policies": len(active),
            "total_active_rules": total_rules,
            "total_evaluations": len(self._evaluation_log),
            "violations": violations,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[PolicyRegistry] = None


def get_policy_registry() -> PolicyRegistry:
    """Return the process-level singleton PolicyRegistry."""
    global _registry
    if _registry is None:
        _registry = PolicyRegistry()
    return _registry
