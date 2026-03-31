# -*- coding: utf-8 -*-
"""
governance/engine.py — GovernancePolicyEngine
==============================================

Thread-safe engine that evaluates agent actions against registered governance
policies and maintains an append-only violation log.

Five default policies are pre-registered at startup:
  1. NO_RISK_LIMIT_OVERRIDE   — agents may never override hard risk limits
  2. MAX_DELEGATION_DEPTH_3   — agent delegation depth capped at 3
  3. PRODUCTION_HIGH_RISK_APPROVAL — HIGH/SENSITIVE in production requires approval
  4. NO_SILENT_MODEL_RETRAIN  — model retraining without approval is forbidden
  5. AUDIT_TRAIL_REQUIRED     — every material action must carry an audit trail

Emergency disable/reenable lets operations teams instantly block specific agents
pending investigation.

Singleton access via ``get_governance_engine()``.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from governance.contracts import (
    GovernancePolicyVersion,
    GuardrailViolation,
    PolicyCheckResult,
    PolicyStatus,
    PolicyViolationSeverity,
)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _today_iso() -> str:
    """Return current UTC date as ISO 8601 date string."""
    return datetime.now(timezone.utc).date().isoformat()


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════
# GOVERNANCE POLICY ENGINE
# ══════════════════════════════════════════════════════════════════


class GovernancePolicyEngine:
    """Evaluate agent actions against registered governance policies.

    Responsibilities
    ----------------
    - Maintain a registry of ``GovernancePolicyVersion`` objects.
    - Evaluate ``check_policy()`` calls against all ACTIVE policies.
    - Maintain an append-only log of ``GuardrailViolation`` objects.
    - Support emergency disable / reenable of specific agents.
    - Expose hard-coded invariant checks (no risk-limit override, delegation cap).
    - Provide ``get_metrics()`` for monitoring dashboards.

    Thread Safety
    -------------
    All mutations are serialized through a single ``threading.Lock``.

    Parameters
    ----------
    None (all configuration is through ``register_policy()``).
    """

    def __init__(self) -> None:
        self._policies: Dict[str, GovernancePolicyVersion] = {}
        self._violations: List[GuardrailViolation] = []
        self._emergency_disabled: Set[str] = set()
        self._lock = threading.Lock()

        self._register_default_policies()

    # ──────────────────────────────────────────────────────────────
    # POLICY CHECKS
    # ──────────────────────────────────────────────────────────────

    def check_policy(
        self,
        agent_name: str,
        task_type: str,
        action_type: str,
        environment: str,
        risk_class: str,
        task_id: str,
    ) -> PolicyCheckResult:
        """Evaluate all active policies against a proposed agent action.

        Evaluation order:
        1. Emergency-disabled check — immediate EMERGENCY block.
        2. Iterate all ACTIVE policies. Collect failure results.
        3. Return the single worst-severity failure, or a passing INFO result.

        Violations are recorded for every failed policy check.

        Parameters
        ----------
        agent_name : str
            Name of the agent proposing the action.
        task_type : str
            Type of the AgentTask.
        action_type : str
            Semantic action type (e.g. "MODEL_PROMOTE", "TRADE_EXECUTE").
        environment : str
            Target environment (e.g. "production", "staging", "research").
        risk_class : str
            Risk classification of the action.
        task_id : str
            ID of the AgentTask.

        Returns
        -------
        PolicyCheckResult
            The most severe result. ``passed=True`` means all checks passed.
        """
        with self._lock:
            now = _now_iso()

            # 1. Emergency disabled
            if agent_name in self._emergency_disabled:
                violation = GuardrailViolation(
                    violation_id=_new_id(),
                    policy_name="EMERGENCY_DISABLE",
                    agent_name=agent_name,
                    task_id=task_id,
                    workflow_run_id=None,
                    description=(
                        "Agent '{}' is emergency-disabled. All actions blocked.".format(
                            agent_name
                        )
                    ),
                    severity=PolicyViolationSeverity.EMERGENCY,
                    blocked=True,
                    timestamp=now,
                    remediation=(
                        "Contact operations to reenable the agent via "
                        "emergency_reenable()."
                    ),
                )
                self._violations.append(violation)
                return PolicyCheckResult(
                    check_id=_new_id(),
                    policy_name="EMERGENCY_DISABLE",
                    policy_version="1.0",
                    agent_name=agent_name,
                    task_id=task_id,
                    action_type=action_type,
                    passed=False,
                    severity=PolicyViolationSeverity.EMERGENCY,
                    message=(
                        "Agent '{}' is emergency-disabled. Action blocked.".format(
                            agent_name
                        )
                    ),
                    details=("Agent is in emergency-disabled state.",),
                    timestamp=now,
                    remediation_hint="Call emergency_reenable(agent_name, approved_by).",
                )

            # 2. Evaluate all active policies
            failures: List[PolicyCheckResult] = []

            for policy in self._policies.values():
                if policy.status != PolicyStatus.ACTIVE:
                    continue
                result = self._evaluate_policy(
                    policy=policy,
                    agent_name=agent_name,
                    task_type=task_type,
                    action_type=action_type,
                    environment=environment,
                    risk_class=risk_class,
                    task_id=task_id,
                    now=now,
                )
                if not result.passed:
                    failures.append(result)
                    # Record violation
                    self._violations.append(
                        GuardrailViolation(
                            violation_id=_new_id(),
                            policy_name=policy.policy_name,
                            agent_name=agent_name,
                            task_id=task_id,
                            workflow_run_id=None,
                            description=result.message,
                            severity=result.severity,
                            blocked=(
                                result.severity
                                >= PolicyViolationSeverity.VIOLATION
                            ),
                            timestamp=now,
                            remediation=result.remediation_hint,
                        )
                    )

            if not failures:
                return PolicyCheckResult(
                    check_id=_new_id(),
                    policy_name="ALL_POLICIES",
                    policy_version="N/A",
                    agent_name=agent_name,
                    task_id=task_id,
                    action_type=action_type,
                    passed=True,
                    severity=PolicyViolationSeverity.INFO,
                    message="All active governance policies passed.",
                    details=tuple(
                        "Policy '{}' passed.".format(p.policy_name)
                        for p in self._policies.values()
                        if p.status == PolicyStatus.ACTIVE
                    ),
                    timestamp=now,
                )

            # Return worst-severity failure
            _sev_order = [
                PolicyViolationSeverity.EMERGENCY,
                PolicyViolationSeverity.CRITICAL,
                PolicyViolationSeverity.VIOLATION,
                PolicyViolationSeverity.WARNING,
                PolicyViolationSeverity.INFO,
            ]
            for sev in _sev_order:
                for f in failures:
                    if f.severity == sev:
                        return f
            return failures[0]

    def check_may_override_risk_limit(
        self, agent_name: str, task_id: str
    ) -> PolicyCheckResult:
        """Hard invariant: agents may never override hard risk limits.

        This check is unconditional — it always returns passed=False,
        severity=CRITICAL regardless of the calling context.

        Parameters
        ----------
        agent_name : str
        task_id : str

        Returns
        -------
        PolicyCheckResult with passed=False, severity=CRITICAL.
        """
        now = _now_iso()
        with self._lock:
            violation = GuardrailViolation(
                violation_id=_new_id(),
                policy_name="NO_RISK_LIMIT_OVERRIDE",
                agent_name=agent_name,
                task_id=task_id,
                workflow_run_id=None,
                description=(
                    "Agent '{}' attempted to override a hard risk limit. "
                    "This is categorically forbidden.".format(agent_name)
                ),
                severity=PolicyViolationSeverity.CRITICAL,
                blocked=True,
                timestamp=now,
                remediation=(
                    "Hard risk limits are immutable by design. "
                    "No override path exists."
                ),
            )
            self._violations.append(violation)

        return PolicyCheckResult(
            check_id=_new_id(),
            policy_name="NO_RISK_LIMIT_OVERRIDE",
            policy_version="1.0",
            agent_name=agent_name,
            task_id=task_id,
            action_type="RISK_LIMIT_OVERRIDE",
            passed=False,
            severity=PolicyViolationSeverity.CRITICAL,
            message="Agents may never override hard risk limits.",
            details=(
                "Policy NO_RISK_LIMIT_OVERRIDE is an absolute invariant.",
                "No ML output, agent decision, or operator request can bypass this.",
            ),
            timestamp=now,
            remediation_hint=(
                "Hard risk limits are set at system configuration level. "
                "Modify system configuration through the approved change process."
            ),
        )

    def check_delegation_depth(
        self, agent_name: str, depth: int, task_id: str
    ) -> PolicyCheckResult:
        """Verify that the agent delegation depth does not exceed the limit of 3.

        Parameters
        ----------
        agent_name : str
        depth : int
            Current delegation depth (0 = top-level call).
        task_id : str

        Returns
        -------
        PolicyCheckResult
            passed=True if depth <= 3; passed=False with VIOLATION severity otherwise.
        """
        now = _now_iso()
        max_depth = 3

        if depth <= max_depth:
            return PolicyCheckResult(
                check_id=_new_id(),
                policy_name="MAX_DELEGATION_DEPTH_3",
                policy_version="1.0",
                agent_name=agent_name,
                task_id=task_id,
                action_type="DELEGATION",
                passed=True,
                severity=PolicyViolationSeverity.INFO,
                message="Delegation depth {} is within the allowed limit of {}.".format(
                    depth, max_depth
                ),
                details=(
                    "Depth: {}. Max: {}.".format(depth, max_depth),
                ),
                timestamp=now,
            )

        with self._lock:
            violation = GuardrailViolation(
                violation_id=_new_id(),
                policy_name="MAX_DELEGATION_DEPTH_3",
                agent_name=agent_name,
                task_id=task_id,
                workflow_run_id=None,
                description=(
                    "Delegation depth {} exceeds maximum of {}.".format(
                        depth, max_depth
                    )
                ),
                severity=PolicyViolationSeverity.VIOLATION,
                blocked=True,
                timestamp=now,
                remediation=(
                    "Restructure the workflow to reduce delegation nesting. "
                    "Maximum allowed depth is {}.".format(max_depth)
                ),
            )
            self._violations.append(violation)

        return PolicyCheckResult(
            check_id=_new_id(),
            policy_name="MAX_DELEGATION_DEPTH_3",
            policy_version="1.0",
            agent_name=agent_name,
            task_id=task_id,
            action_type="DELEGATION",
            passed=False,
            severity=PolicyViolationSeverity.VIOLATION,
            message="Delegation depth {} exceeds maximum of {}.".format(
                depth, max_depth
            ),
            details=(
                "Depth: {}. Max: {}.".format(depth, max_depth),
                "Exceeding the delegation cap risks unbounded recursion and "
                "loss of audit clarity.",
            ),
            timestamp=now,
            remediation_hint=(
                "Flatten the workflow or introduce an intermediate orchestrator "
                "to keep depth <= {}.".format(max_depth)
            ),
        )

    # ──────────────────────────────────────────────────────────────
    # EMERGENCY CONTROLS
    # ──────────────────────────────────────────────────────────────

    def emergency_disable(self, agent_name: str, reason: str) -> None:
        """Immediately block all actions from the specified agent.

        Emergency disable takes effect on the next ``check_policy()`` call.
        All subsequent checks for this agent will return EMERGENCY severity.

        Parameters
        ----------
        agent_name : str
            The agent to disable.
        reason : str
            Reason for the emergency disable (written to violation log).
        """
        with self._lock:
            self._emergency_disabled.add(agent_name)
            self._violations.append(
                GuardrailViolation(
                    violation_id=_new_id(),
                    policy_name="EMERGENCY_DISABLE",
                    agent_name=agent_name,
                    task_id="SYSTEM",
                    workflow_run_id=None,
                    description=(
                        "Emergency disable applied to '{}'. Reason: {}".format(
                            agent_name, reason
                        )
                    ),
                    severity=PolicyViolationSeverity.EMERGENCY,
                    blocked=True,
                    timestamp=_now_iso(),
                    remediation=(
                        "Call emergency_reenable(agent_name, approved_by) "
                        "after investigation is complete."
                    ),
                )
            )

    def emergency_reenable(self, agent_name: str, approved_by: str) -> None:
        """Re-enable an emergency-disabled agent.

        Parameters
        ----------
        agent_name : str
            The agent to reenable.
        approved_by : str
            Identity of the approver authorizing the reenable.
        """
        with self._lock:
            self._emergency_disabled.discard(agent_name)
            self._violations.append(
                GuardrailViolation(
                    violation_id=_new_id(),
                    policy_name="EMERGENCY_REENABLE",
                    agent_name=agent_name,
                    task_id="SYSTEM",
                    workflow_run_id=None,
                    description=(
                        "Emergency disable lifted for '{}'. Approved by: {}.".format(
                            agent_name, approved_by
                        )
                    ),
                    severity=PolicyViolationSeverity.INFO,
                    blocked=False,
                    timestamp=_now_iso(),
                    remediation="Agent is now active. Monitor closely.",
                )
            )

    def is_emergency_disabled(self, agent_name: str) -> bool:
        """Return True if the agent is currently emergency-disabled.

        Parameters
        ----------
        agent_name : str

        Returns
        -------
        bool
        """
        with self._lock:
            return agent_name in self._emergency_disabled

    # ──────────────────────────────────────────────────────────────
    # POLICY REGISTRY
    # ──────────────────────────────────────────────────────────────

    def register_policy(self, policy: GovernancePolicyVersion) -> None:
        """Register or replace a governance policy.

        If a policy with the same ``policy_id`` already exists, it is
        replaced. The caller is responsible for version management.

        Parameters
        ----------
        policy : GovernancePolicyVersion
        """
        with self._lock:
            self._policies[policy.policy_id] = policy

    def get_violations(
        self,
        agent_name: Optional[str] = None,
        severity: Optional[PolicyViolationSeverity] = None,
    ) -> List[GuardrailViolation]:
        """Return recorded violations, optionally filtered.

        Parameters
        ----------
        agent_name : Optional[str]
            If provided, only return violations for this agent.
        severity : Optional[PolicyViolationSeverity]
            If provided, only return violations at this severity or higher.

        Returns
        -------
        list[GuardrailViolation]
            Violations in chronological order.
        """
        with self._lock:
            results = list(self._violations)

        if agent_name is not None:
            results = [v for v in results if v.agent_name == agent_name]
        if severity is not None:
            results = [v for v in results if v.severity >= severity]
        return results

    # ──────────────────────────────────────────────────────────────
    # METRICS
    # ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for the governance engine.

        Returns
        -------
        dict
            Keys: total_violations, violations_by_severity, blocked_count,
            active_policy_count, emergency_disabled_agents,
            violations_by_policy (top 10).
        """
        with self._lock:
            violations = list(self._violations)
            disabled = set(self._emergency_disabled)
            active_count = sum(
                1
                for p in self._policies.values()
                if p.status == PolicyStatus.ACTIVE
            )

        by_severity: Dict[str, int] = {s.value: 0 for s in PolicyViolationSeverity}
        blocked_count = 0
        by_policy: Dict[str, int] = {}

        for v in violations:
            by_severity[v.severity.value] += 1
            if v.blocked:
                blocked_count += 1
            by_policy[v.policy_name] = by_policy.get(v.policy_name, 0) + 1

        top_policies = sorted(by_policy.items(), key=lambda x: -x[1])[:10]

        return {
            "total_violations": len(violations),
            "violations_by_severity": by_severity,
            "blocked_count": blocked_count,
            "active_policy_count": active_count,
            "emergency_disabled_agents": sorted(disabled),
            "violations_by_policy": dict(top_policies),
        }

    # ──────────────────────────────────────────────────────────────
    # INTERNAL: DEFAULT POLICIES
    # ──────────────────────────────────────────────────────────────

    def _register_default_policies(self) -> None:
        """Register the five mandatory default governance policies."""
        today = _today_iso()

        defaults: List[GovernancePolicyVersion] = [
            GovernancePolicyVersion(
                policy_id="NO_RISK_LIMIT_OVERRIDE",
                policy_name="No Risk Limit Override",
                version="1.0",
                description=(
                    "Agents, ML models, and automated workflows are categorically "
                    "forbidden from overriding hard risk limits. Hard limits exist "
                    "in kill-switch, drawdown manager, and lifecycle state machine. "
                    "No output from any agent or model may bypass these controls."
                ),
                status=PolicyStatus.ACTIVE,
                effective_from=today,
                effective_until=None,
                author="platform",
                approved_by="platform",
                rules=(
                    "RULE-001: MLUsageContract.may_override_risk_limit is always False.",
                    "RULE-002: Kill-switch and drawdown hard vetoes are not bypassable.",
                    "RULE-003: Lifecycle state machine hard transitions cannot be skipped.",
                ),
                risk_class="ALL",
                environment_scope=("production", "staging", "research", "dev"),
                last_reviewed=today,
                next_review=None,
            ),
            GovernancePolicyVersion(
                policy_id="MAX_DELEGATION_DEPTH_3",
                policy_name="Maximum Agent Delegation Depth 3",
                version="1.0",
                description=(
                    "Agent delegation chains may not exceed 3 levels of depth. "
                    "Deeper chains risk unbounded recursion, loss of audit clarity, "
                    "and unpredictable resource consumption. Flat workflows are preferred."
                ),
                status=PolicyStatus.ACTIVE,
                effective_from=today,
                effective_until=None,
                author="platform",
                approved_by="platform",
                rules=(
                    "RULE-010: Delegation depth tracked from originating task.",
                    "RULE-011: Depth > 3 is a VIOLATION and blocks execution.",
                    "RULE-012: Top-level call counts as depth 0.",
                ),
                risk_class="ALL",
                environment_scope=("production", "staging", "research", "dev"),
                last_reviewed=today,
                next_review=None,
            ),
            GovernancePolicyVersion(
                policy_id="PRODUCTION_HIGH_RISK_APPROVAL",
                policy_name="Production High-Risk Action Requires Approval",
                version="1.0",
                description=(
                    "Actions classified as HIGH_RISK or SENSITIVE in the production "
                    "environment must pass through the ApprovalEngine with at least "
                    "POLICY_GATED mode. Automatic execution of high-risk production "
                    "actions is forbidden."
                ),
                status=PolicyStatus.ACTIVE,
                effective_from=today,
                effective_until=None,
                author="platform",
                approved_by="platform",
                rules=(
                    "RULE-020: HIGH_RISK + production → approval required.",
                    "RULE-021: SENSITIVE + any environment → approval required.",
                    "RULE-022: ApprovalMode.AUTOMATIC forbidden for HIGH_RISK in production.",
                ),
                risk_class="HIGH",
                environment_scope=("production",),
                last_reviewed=today,
                next_review=None,
            ),
            GovernancePolicyVersion(
                policy_id="NO_SILENT_MODEL_RETRAIN",
                policy_name="No Silent Model Retraining",
                version="1.0",
                description=(
                    "Model retraining, re-fitting, or parameter updates must not "
                    "occur silently without approval. Every retraining event must be "
                    "registered as an ApprovalRequest and logged in the audit trail. "
                    "Champion promotion requires GovernanceEngine review."
                ),
                status=PolicyStatus.ACTIVE,
                effective_from=today,
                effective_until=None,
                author="platform",
                approved_by="platform",
                rules=(
                    "RULE-030: MODEL_RETRAIN action type requires approval.",
                    "RULE-031: Champion promotion requires GovernanceEngine.check_promotion_criteria().",
                    "RULE-032: Retraining without approval is a VIOLATION.",
                ),
                risk_class="SENSITIVE",
                environment_scope=("production", "staging"),
                last_reviewed=today,
                next_review=None,
            ),
            GovernancePolicyVersion(
                policy_id="AUDIT_TRAIL_REQUIRED",
                policy_name="Audit Trail Required for Material Actions",
                version="1.0",
                description=(
                    "Every material agent action (trade execution, model promotion, "
                    "capital allocation, kill-switch trigger, emergency disable) must "
                    "produce an AuditRecord. Actions without audit trails are a "
                    "VIOLATION and may not be replicated in production."
                ),
                status=PolicyStatus.ACTIVE,
                effective_from=today,
                effective_until=None,
                author="platform",
                approved_by="platform",
                rules=(
                    "RULE-040: All material actions produce AuditRecord via IncidentManager.log_audit().",
                    "RULE-041: AuditRecord.schema_version must be present.",
                    "RULE-042: Missing audit trail is VIOLATION severity.",
                ),
                risk_class="ALL",
                environment_scope=("production", "staging"),
                last_reviewed=today,
                next_review=None,
            ),
        ]

        for policy in defaults:
            self._policies[policy.policy_id] = policy

    # ──────────────────────────────────────────────────────────────
    # INTERNAL: POLICY EVALUATION
    # ──────────────────────────────────────────────────────────────

    def _evaluate_policy(
        self,
        policy: GovernancePolicyVersion,
        agent_name: str,
        task_type: str,
        action_type: str,
        environment: str,
        risk_class: str,
        task_id: str,
        now: str,
    ) -> PolicyCheckResult:
        """Evaluate a single active policy against the action.

        Parameters
        ----------
        policy : GovernancePolicyVersion
        agent_name, task_type, action_type, environment, risk_class, task_id, now : str

        Returns
        -------
        PolicyCheckResult
        """
        pid = policy.policy_id
        env_lower = environment.lower()
        risk_upper = risk_class.upper()

        # Scope check — skip if environment not in scope
        if (
            policy.environment_scope
            and env_lower not in [e.lower() for e in policy.environment_scope]
            and "all" not in [e.lower() for e in policy.environment_scope]
        ):
            return PolicyCheckResult(
                check_id=_new_id(),
                policy_name=policy.policy_name,
                policy_version=policy.version,
                agent_name=agent_name,
                task_id=task_id,
                action_type=action_type,
                passed=True,
                severity=PolicyViolationSeverity.INFO,
                message="Policy '{}' not applicable to environment '{}'.".format(
                    policy.policy_name, environment
                ),
                details=("Environment '{}' out of policy scope.".format(environment),),
                timestamp=now,
            )

        if pid == "NO_RISK_LIMIT_OVERRIDE":
            if action_type.upper() in (
                "RISK_LIMIT_OVERRIDE",
                "KILLSWITCH_BYPASS",
                "DRAWDOWN_BYPASS",
                "LIFECYCLE_FORCE",
            ):
                return PolicyCheckResult(
                    check_id=_new_id(),
                    policy_name=policy.policy_name,
                    policy_version=policy.version,
                    agent_name=agent_name,
                    task_id=task_id,
                    action_type=action_type,
                    passed=False,
                    severity=PolicyViolationSeverity.CRITICAL,
                    message="Action '{}' violates NO_RISK_LIMIT_OVERRIDE.".format(
                        action_type
                    ),
                    details=(policy.rules[0], policy.rules[1]),
                    timestamp=now,
                    remediation_hint=(
                        "Hard risk limits cannot be overridden by any agent action."
                    ),
                )

        elif pid == "PRODUCTION_HIGH_RISK_APPROVAL":
            if env_lower == "production" and risk_upper in ("HIGH", "SENSITIVE", "CRITICAL"):
                if action_type.upper() in ("AUTO_APPROVE", "SKIP_APPROVAL"):
                    return PolicyCheckResult(
                        check_id=_new_id(),
                        policy_name=policy.policy_name,
                        policy_version=policy.version,
                        agent_name=agent_name,
                        task_id=task_id,
                        action_type=action_type,
                        passed=False,
                        severity=PolicyViolationSeverity.VIOLATION,
                        message=(
                            "HIGH/SENSITIVE action in production cannot be auto-approved."
                        ),
                        details=(
                            "Risk class: {}. Environment: {}.".format(
                                risk_class, environment
                            ),
                        ),
                        timestamp=now,
                        remediation_hint=(
                            "Use ApprovalMode.POLICY_GATED or HUMAN_REQUIRED "
                            "for high-risk production actions."
                        ),
                    )

        elif pid == "NO_SILENT_MODEL_RETRAIN":
            if action_type.upper() in (
                "MODEL_RETRAIN",
                "MODEL_REFIT",
                "SILENT_RETRAIN",
                "MODEL_PROMOTE_SILENT",
            ):
                return PolicyCheckResult(
                    check_id=_new_id(),
                    policy_name=policy.policy_name,
                    policy_version=policy.version,
                    agent_name=agent_name,
                    task_id=task_id,
                    action_type=action_type,
                    passed=False,
                    severity=PolicyViolationSeverity.VIOLATION,
                    message=(
                        "Model retraining/promotion requires explicit approval. "
                        "Silent retraining is forbidden."
                    ),
                    details=(policy.rules[0], policy.rules[1]),
                    timestamp=now,
                    remediation_hint=(
                        "Submit an ApprovalRequest before retraining. "
                        "Use GovernanceEngine.check_promotion_criteria() before promoting."
                    ),
                )

        elif pid == "AUDIT_TRAIL_REQUIRED":
            _material_actions = {
                "TRADE_EXECUTE",
                "MODEL_PROMOTE",
                "CAPITAL_ALLOCATE",
                "KILL_SWITCH_TRIGGER",
                "EMERGENCY_DISABLE",
                "POSITION_OVERRIDE",
            }
            if action_type.upper() in _material_actions:
                # This policy is a reminder — always pass here since audit logging
                # is enforced at the IncidentManager layer, not the policy check layer.
                pass

        # Default: pass
        return PolicyCheckResult(
            check_id=_new_id(),
            policy_name=policy.policy_name,
            policy_version=policy.version,
            agent_name=agent_name,
            task_id=task_id,
            action_type=action_type,
            passed=True,
            severity=PolicyViolationSeverity.INFO,
            message="Policy '{}' passed.".format(policy.policy_name),
            details=(),
            timestamp=now,
        )


# ══════════════════════════════════════════════════════════════════
# SINGLETON FACTORY
# ══════════════════════════════════════════════════════════════════

_governance_engine_instance: Optional[GovernancePolicyEngine] = None
_governance_engine_lock = threading.Lock()


def get_governance_engine() -> GovernancePolicyEngine:
    """Return the process-level singleton GovernancePolicyEngine.

    Thread-safe. Instantiates on first call.

    Returns
    -------
    GovernancePolicyEngine
    """
    global _governance_engine_instance
    if _governance_engine_instance is None:
        with _governance_engine_lock:
            if _governance_engine_instance is None:
                _governance_engine_instance = GovernancePolicyEngine()
    return _governance_engine_instance
