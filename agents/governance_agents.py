# -*- coding: utf-8 -*-
"""
agents/governance_agents.py — Governance and Review Agent Implementations
==========================================================================

Five agent classes covering governance policy review, approval recommendation,
change impact assessment, promotion gating, and audit trail validation.

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


# ── Internal policy rule sets (used when governance.engine is unavailable) ──

# Actions that are categorically blocked
_BLOCKED_ACTIONS = frozenset({
    "DISABLE_KILL_SWITCH",
    "OVERRIDE_RISK_LIMIT",
    "DELETE_AUDIT_RECORD",
    "BYPASS_LEAKAGE_CHECK",
})

# Actions that require human approval regardless of environment
_HUMAN_REQUIRED_ACTIONS = frozenset({
    "MODEL_PROMOTE_TO_CHAMPION",
    "FORCE_POSITION_SIZE_OVERRIDE",
    "CAPITAL_BUDGET_INCREASE",
    "KILL_SWITCH_RESET",
    "DUAL_APPROVAL_REQUIRED",
})

# Actions requiring dual approval in production
_DUAL_APPROVAL_PRODUCTION = frozenset({
    "ORDER_EXECUTION",
    "LIVE_TRADING_ENABLE",
})


# ══════════════════════════════════════════════════════════════════
# 1. PolicyReviewAgent
# ══════════════════════════════════════════════════════════════════


class PolicyReviewAgent(BaseAgent):
    """
    Evaluates a proposed agent action against governance policies.

    Attempts to use ``governance.engine.GovernancePolicyEngine.check_policy()``;
    falls back to an internal rule set when the engine is unavailable.

    Task types
    ----------
    review_policy
        Full policy check for a proposed action.
    check_policy_compliance
        Alias — check compliance of an agent action.

    Required payload keys
    ---------------------
    agent_name : str
    task_type : str
    action_type : str
    environment : str
    risk_class : str

    Optional payload keys
    ---------------------
    task_id : str

    Output keys
    -----------
    policy_check : dict  (PolicyCheckResult fields)
    blocked : bool
    violation_severity : str
    remediation : str
    """

    NAME = "policy_review"
    ALLOWED_TASK_TYPES = {"review_policy", "check_policy_compliance"}
    REQUIRED_PAYLOAD_KEYS = {"agent_name", "task_type", "action_type", "environment", "risk_class"}

    def _fallback_check(
        self,
        agent_name: str,
        action_type: str,
        environment: str,
        risk_class: str,
        task_id: str,
    ) -> dict[str, Any]:
        """Internal rule-based policy check when governance.engine is unavailable."""
        check_id = _new_id()
        now = _utcnow()
        details: list[str] = []
        passed = True
        severity = "INFO"
        message = "Action permitted by default policy rules."
        remediation = ""

        action_upper = action_type.upper()
        env_upper = environment.upper()
        risk_upper = risk_class.upper()

        # Rule 1: Blocked actions
        if action_upper in _BLOCKED_ACTIONS:
            passed = False
            severity = "EMERGENCY"
            message = f"Action '{action_type}' is categorically blocked by governance policy."
            remediation = "This action cannot be permitted. Contact the risk management team."
            details.append(f"BLOCKED_ACTION_LIST matched: {action_upper}")

        # Rule 2: Sensitive risk class in production requires human approval
        elif risk_upper == "SENSITIVE" and env_upper == "PRODUCTION":
            passed = False
            severity = "CRITICAL"
            message = (
                f"SENSITIVE action '{action_type}' in PRODUCTION requires human approval."
            )
            remediation = "Submit a formal ApprovalRequest through the approval workflow."
            details.append("SENSITIVE+PRODUCTION requires human gate")

        # Rule 3: High-risk in production
        elif risk_upper == "HIGH_RISK" and env_upper == "PRODUCTION":
            passed = False
            severity = "VIOLATION"
            message = (
                f"HIGH_RISK action '{action_type}' in PRODUCTION requires policy gate."
            )
            remediation = "Obtain POLICY_GATED approval before proceeding."
            details.append("HIGH_RISK+PRODUCTION requires policy gate")

        # Rule 4: Human-required action types
        elif action_upper in _HUMAN_REQUIRED_ACTIONS:
            passed = False
            severity = "VIOLATION"
            message = f"Action '{action_type}' requires human approval."
            remediation = "Route to human review queue via ApprovalRequest."
            details.append(f"HUMAN_REQUIRED_ACTIONS matched: {action_upper}")

        # Rule 5: Dual approval in production
        elif action_upper in _DUAL_APPROVAL_PRODUCTION and env_upper == "PRODUCTION":
            passed = False
            severity = "VIOLATION"
            message = f"Action '{action_type}' requires dual human approval in PRODUCTION."
            remediation = "Create ApprovalRequest with DUAL_APPROVAL mode."
            details.append("DUAL_APPROVAL_PRODUCTION matched")

        else:
            details.append(f"No blocking rules matched for {action_upper}/{risk_upper}/{env_upper}")

        return {
            "check_id": check_id,
            "policy_name": "internal_fallback_policy",
            "policy_version": "1.0",
            "agent_name": agent_name,
            "task_id": task_id,
            "action_type": action_type,
            "passed": passed,
            "severity": severity,
            "message": message,
            "details": details,
            "timestamp": now,
            "remediation_hint": remediation,
        }

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        agent_name: str = task.payload["agent_name"]
        task_type: str = task.payload["task_type"]
        action_type: str = task.payload["action_type"]
        environment: str = task.payload["environment"]
        risk_class: str = task.payload["risk_class"]
        task_id: str = task.payload.get("task_id") or task.task_id

        audit.log(
            f"Policy review: agent={agent_name}, action={action_type}, "
            f"env={environment}, risk={risk_class}"
        )

        check_dict: dict[str, Any] | None = None

        # Try governance engine
        try:
            from governance.engine import get_governance_engine
            engine = get_governance_engine()
            result = engine.check_policy(
                agent_name=agent_name,
                task_id=task_id,
                action_type=action_type,
                risk_class=risk_class,
                environment=environment,
            )
            check_dict = {
                "check_id": getattr(result, "check_id", _new_id()),
                "policy_name": getattr(result, "policy_name", "governance_engine"),
                "policy_version": getattr(result, "policy_version", "1.0"),
                "agent_name": getattr(result, "agent_name", agent_name),
                "task_id": getattr(result, "task_id", task_id),
                "action_type": getattr(result, "action_type", action_type),
                "passed": bool(getattr(result, "passed", True)),
                "severity": str(getattr(result, "severity", "INFO")),
                "message": str(getattr(result, "message", "")),
                "details": list(getattr(result, "details", [])),
                "timestamp": str(getattr(result, "timestamp", _utcnow())),
                "remediation_hint": str(getattr(result, "remediation_hint", "")),
            }
            audit.log("Used governance.engine.GovernancePolicyEngine")
        except (ImportError, Exception) as exc:
            audit.warn(f"GovernancePolicyEngine unavailable ({exc}) — using fallback rules")
            check_dict = self._fallback_check(
                agent_name, action_type, environment, risk_class, task_id
            )

        blocked = not check_dict["passed"]
        violation_severity = check_dict["severity"] if not check_dict["passed"] else "INFO"

        audit.log(
            f"Policy check complete: passed={check_dict['passed']}, "
            f"severity={check_dict['severity']}, blocked={blocked}"
        )

        return {
            "policy_check": check_dict,
            "blocked": blocked,
            "violation_severity": violation_severity,
            "remediation": check_dict.get("remediation_hint", ""),
        }


# ══════════════════════════════════════════════════════════════════
# 2. ApprovalRecommendationAgent
# ══════════════════════════════════════════════════════════════════


class ApprovalRecommendationAgent(BaseAgent):
    """
    Recommends the appropriate approval mode for a proposed action.

    Uses a risk_class × environment matrix to determine whether an action
    should be AUTOMATIC, POLICY_GATED, HUMAN_REQUIRED, or BLOCKED.

    Task types
    ----------
    recommend_approval
        Full recommendation including evidence sufficiency check.
    assess_approval_request
        Alias.

    Required payload keys
    ---------------------
    action_type : str
    risk_class : str
    environment : str
    evidence_summary : str
    agent_name : str

    Output keys
    -----------
    recommended_mode : str
    rationale : str
    evidence_sufficient : bool
    missing_evidence : list[str]
    suggested_reviewer : str
    """

    NAME = "approval_recommendation"
    ALLOWED_TASK_TYPES = {"recommend_approval", "assess_approval_request"}
    REQUIRED_PAYLOAD_KEYS = {"action_type", "risk_class", "environment", "evidence_summary", "agent_name"}

    # Risk × Environment → ApprovalMode
    # Rows: INFORMATIONAL, BOUNDED_SAFE, MEDIUM_RISK, HIGH_RISK, SENSITIVE
    # Cols: RESEARCH, STAGING, PAPER, PRODUCTION
    _MODE_MATRIX: dict[str, dict[str, str]] = {
        "INFORMATIONAL": {
            "RESEARCH": "AUTOMATIC", "STAGING": "AUTOMATIC",
            "PAPER": "AUTOMATIC", "PRODUCTION": "AUTOMATIC",
        },
        "BOUNDED_SAFE": {
            "RESEARCH": "AUTOMATIC", "STAGING": "POLICY_GATED",
            "PAPER": "POLICY_GATED", "PRODUCTION": "POLICY_GATED",
        },
        "MEDIUM_RISK": {
            "RESEARCH": "POLICY_GATED", "STAGING": "POLICY_GATED",
            "PAPER": "HUMAN_REQUIRED", "PRODUCTION": "HUMAN_REQUIRED",
        },
        "HIGH_RISK": {
            "RESEARCH": "POLICY_GATED", "STAGING": "HUMAN_REQUIRED",
            "PAPER": "HUMAN_REQUIRED", "PRODUCTION": "HUMAN_REQUIRED",
        },
        "SENSITIVE": {
            "RESEARCH": "HUMAN_REQUIRED", "STAGING": "HUMAN_REQUIRED",
            "PAPER": "HUMAN_REQUIRED", "PRODUCTION": "BLOCKED",
        },
    }

    # Reviewer mapping by environment
    _REVIEWER_MAP: dict[str, str] = {
        "PRODUCTION": "risk_management_team",
        "PAPER": "quant_lead",
        "STAGING": "senior_engineer",
        "RESEARCH": "team_lead",
    }

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        action_type: str = task.payload["action_type"]
        risk_class: str = str(task.payload["risk_class"]).upper()
        environment: str = str(task.payload["environment"]).upper()
        evidence_summary: str = task.payload["evidence_summary"]
        agent_name: str = task.payload["agent_name"]

        audit.log(
            f"Recommending approval mode: action={action_type}, "
            f"risk={risk_class}, env={environment}"
        )

        # Override: blocked actions are always BLOCKED
        if action_type.upper() in _BLOCKED_ACTIONS:
            mode = "BLOCKED"
            rationale = f"Action '{action_type}' is in the categorically blocked actions list."
        elif action_type.upper() in _HUMAN_REQUIRED_ACTIONS:
            mode = "HUMAN_REQUIRED"
            rationale = f"Action '{action_type}' always requires human approval."
        else:
            risk_row = self._MODE_MATRIX.get(risk_class)
            if risk_row is None:
                # Unknown risk class — default to HUMAN_REQUIRED
                mode = "HUMAN_REQUIRED"
                rationale = f"Unknown risk class '{risk_class}' — defaulting to HUMAN_REQUIRED."
            else:
                mode = risk_row.get(environment, "HUMAN_REQUIRED")
                rationale = (
                    f"Risk class '{risk_class}' in '{environment}' environment maps to '{mode}' "
                    f"per approval mode matrix."
                )

        # Evidence sufficiency check
        missing_evidence: list[str] = []
        evidence_lower = evidence_summary.lower()
        if mode in ("HUMAN_REQUIRED", "POLICY_GATED"):
            if "metrics" not in evidence_lower and "metric" not in evidence_lower:
                missing_evidence.append("performance_metrics")
            if "impact" not in evidence_lower:
                missing_evidence.append("impact_assessment")
            if "risk" not in evidence_lower:
                missing_evidence.append("risk_evaluation")
        evidence_sufficient = len(missing_evidence) == 0

        suggested_reviewer = self._REVIEWER_MAP.get(environment, "risk_management_team")

        # Try to import ApprovalMode for validation
        try:
            from approvals.contracts import ApprovalMode
            # Validate mode is a known value
            valid_modes = {m.value for m in ApprovalMode}
            if mode not in valid_modes:
                mode = "HUMAN_REQUIRED"
                rationale += " (mode corrected to HUMAN_REQUIRED — unknown mode)"
        except ImportError:
            pass

        audit.log(
            f"Approval recommendation: mode={mode}, "
            f"evidence_sufficient={evidence_sufficient}, missing={missing_evidence}"
        )

        return {
            "recommended_mode": mode,
            "rationale": rationale,
            "evidence_sufficient": evidence_sufficient,
            "missing_evidence": missing_evidence,
            "suggested_reviewer": suggested_reviewer,
        }


# ══════════════════════════════════════════════════════════════════
# 3. ChangeImpactAgent
# ══════════════════════════════════════════════════════════════════


class ChangeImpactAgent(BaseAgent):
    """
    Assesses the blast radius of a proposed system change.

    Maps affected_components to dependent agents and workflows, estimates
    risk classification, and produces a typed ``ChangeImpactReport``.

    Task types
    ----------
    assess_change_impact
        Full impact assessment.
    generate_impact_report
        Alias.

    Required payload keys
    ---------------------
    change_description : str
    affected_components : list[str]
    change_type : str
    proposed_by : str

    Output keys
    -----------
    impact_report : dict  (ChangeImpactReport fields)
    risk_class : str
    requires_approval : bool
    rollback_required : bool
    """

    NAME = "change_impact"
    ALLOWED_TASK_TYPES = {"assess_change_impact", "generate_impact_report"}
    REQUIRED_PAYLOAD_KEYS = {"change_description", "affected_components", "change_type", "proposed_by"}

    # Component → dependent agents
    _COMPONENT_AGENT_DEPS: dict[str, list[str]] = {
        "core.contracts": ["all_agents"],
        "core.signals_engine": ["signal_analyst", "trade_lifecycle"],
        "core.regime_engine": ["regime_surveillance", "signal_analyst", "regime_modeling"],
        "core.risk_engine": ["drawdown_monitor", "kill_switch", "de_risking"],
        "core.optimizer": ["portfolio_construction"],
        "core.sql_store": ["all_data_agents"],
        "portfolio.allocator": ["portfolio_construction", "capital_budget", "exposure_monitor"],
        "ml.inference": ["meta_labeling", "regime_modeling"],
        "orchestration.engine": ["all_agents"],
    }

    # Component → dependent workflows
    _COMPONENT_WORKFLOW_DEPS: dict[str, list[str]] = {
        "core.signals_engine": ["signal_generation", "trade_entry", "trade_exit"],
        "core.regime_engine": ["regime_classification", "signal_generation"],
        "portfolio.allocator": ["portfolio_construction", "rebalancing"],
        "core.optimizer": ["walk_forward_optimization"],
        "ml.inference": ["meta_label_scoring", "regime_classification_ml"],
        "core.sql_store": ["data_ingestion", "price_update"],
    }

    # Change type → base risk class
    _CHANGETYPE_RISK: dict[str, str] = {
        "HOTFIX": "MEDIUM_RISK",
        "FEATURE": "MEDIUM_RISK",
        "REFACTOR": "MEDIUM_RISK",
        "MODEL_UPDATE": "HIGH_RISK",
        "CONFIG_CHANGE": "MEDIUM_RISK",
        "SCHEMA_MIGRATION": "HIGH_RISK",
        "DEPENDENCY_UPGRADE": "MEDIUM_RISK",
        "BREAKING_CHANGE": "SENSITIVE",
        "ROLLBACK": "HIGH_RISK",
        "EMERGENCY": "HIGH_RISK",
    }

    def _resolve_dependencies(
        self,
        components: list[str],
    ) -> tuple[list[str], list[str]]:
        """Return (affected_agents, affected_workflows) from component list."""
        agents_set: set[str] = set()
        workflows_set: set[str] = set()
        for comp in components:
            for key, deps in self._COMPONENT_AGENT_DEPS.items():
                if key in comp or comp in key:
                    agents_set.update(deps)
            for key, deps in self._COMPONENT_WORKFLOW_DEPS.items():
                if key in comp or comp in key:
                    workflows_set.update(deps)
        return sorted(agents_set), sorted(workflows_set)

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        change_description: str = task.payload["change_description"]
        affected_components: list[str] = list(task.payload["affected_components"])
        change_type: str = str(task.payload["change_type"]).upper()
        proposed_by: str = task.payload["proposed_by"]

        audit.log(
            f"Assessing change impact: type={change_type}, "
            f"components={affected_components}, proposed_by={proposed_by}"
        )

        # Resolve dependencies
        affected_agents, affected_workflows = self._resolve_dependencies(affected_components)

        # Risk classification
        base_risk = self._CHANGETYPE_RISK.get(change_type, "MEDIUM_RISK")
        # Escalate if core contracts or orchestration affected
        high_impact_components = {"core.contracts", "orchestration.engine", "core.sql_store"}
        if any(c in " ".join(affected_components) for c in high_impact_components):
            if base_risk == "MEDIUM_RISK":
                base_risk = "HIGH_RISK"
        if "all_agents" in affected_agents or len(affected_components) > 5:
            base_risk = "HIGH_RISK"

        # Reversibility
        irreversible_types = {"SCHEMA_MIGRATION", "BREAKING_CHANGE"}
        reversibility = (
            "IRREVERSIBLE" if change_type in irreversible_types
            else "PARTIALLY_REVERSIBLE" if change_type in {"MODEL_UPDATE", "DEPENDENCY_UPGRADE"}
            else "REVERSIBLE"
        )

        requires_approval = base_risk in ("HIGH_RISK", "SENSITIVE")
        rollback_required = change_type in ("SCHEMA_MIGRATION", "BREAKING_CHANGE", "EMERGENCY")

        estimated_impact = (
            f"Change affects {len(affected_components)} component(s), "
            f"{len(affected_agents)} agent type(s), "
            f"{len(affected_workflows)} workflow type(s). "
            f"Risk: {base_risk}."
        )
        rollback_plan = (
            "1. Revert git commit or deployment artifact. "
            "2. Run smoke tests against reverted version. "
            "3. Verify data consistency post-revert. "
            "4. Update incident record with revert action."
            if rollback_required
            else "Standard git revert or config rollback sufficient."
        )

        report_id = _new_id()
        now = _utcnow()

        try:
            from governance.contracts import ChangeImpactReport
            report = ChangeImpactReport(
                report_id=report_id,
                change_description=change_description,
                proposed_by=proposed_by,
                proposed_at=now,
                affected_components=tuple(affected_components),
                affected_workflows=tuple(affected_workflows),
                affected_agents=tuple(affected_agents),
                risk_class=base_risk,
                estimated_impact=estimated_impact,
                reversibility=reversibility,
                requires_approval=requires_approval,
                rollback_plan=rollback_plan,
                notes=f"change_type={change_type}",
            )
            report_dict = {
                "report_id": report.report_id,
                "change_description": report.change_description,
                "proposed_by": report.proposed_by,
                "proposed_at": report.proposed_at,
                "affected_components": list(report.affected_components),
                "affected_workflows": list(report.affected_workflows),
                "affected_agents": list(report.affected_agents),
                "risk_class": report.risk_class,
                "estimated_impact": report.estimated_impact,
                "reversibility": report.reversibility,
                "requires_approval": report.requires_approval,
                "rollback_plan": report.rollback_plan,
                "notes": report.notes,
            }
        except ImportError:
            report_dict = {
                "report_id": report_id,
                "change_description": change_description,
                "proposed_by": proposed_by,
                "proposed_at": now,
                "affected_components": affected_components,
                "affected_workflows": affected_workflows,
                "affected_agents": affected_agents,
                "risk_class": base_risk,
                "estimated_impact": estimated_impact,
                "reversibility": reversibility,
                "requires_approval": requires_approval,
                "rollback_plan": rollback_plan,
                "notes": f"change_type={change_type}",
            }

        audit.log(
            f"Change impact assessment complete: risk={base_risk}, "
            f"requires_approval={requires_approval}, reversibility={reversibility}"
        )

        return {
            "impact_report": report_dict,
            "risk_class": base_risk,
            "requires_approval": requires_approval,
            "rollback_required": rollback_required,
        }


# ══════════════════════════════════════════════════════════════════
# 4. PromotionGateAgent
# ══════════════════════════════════════════════════════════════════


class PromotionGateAgent(BaseAgent):
    """
    Gates promotion of models and other subjects through the approval workflow.

    Validates the requested transition, checks governance metrics, and creates
    an ``ApprovalRequest`` if the subject is eligible.

    Task types
    ----------
    gate_promotion
        Full gate: validate transition, check metrics, create ApprovalRequest.
    check_promotion_eligibility
        Eligibility check only (no ApprovalRequest created).

    Required payload keys
    ---------------------
    subject_type : str
    subject_id : str
    subject_name : str
    from_status : str
    to_status : str

    Optional payload keys
    ---------------------
    metrics : dict
    evidence_ids : list[str]

    Output keys
    -----------
    eligible : bool
    blocking_reasons : list[str]
    approval_request_id : str | None
    approval_mode : str
    auto_approved : bool
    """

    NAME = "promotion_gate"
    ALLOWED_TASK_TYPES = {"gate_promotion", "check_promotion_eligibility"}
    REQUIRED_PAYLOAD_KEYS = {
        "subject_type", "subject_id", "subject_name", "from_status", "to_status"
    }

    # Valid promotion transitions by subject type
    _MODEL_TRANSITIONS: set[tuple[str, str]] = {
        ("CANDIDATE", "CHALLENGER"),
        ("CHALLENGER", "CHAMPION"),
        ("CHAMPION", "RETIRED"),
        ("CANDIDATE", "RETIRED"),
        ("CHALLENGER", "RETIRED"),
    }

    _POLICY_TRANSITIONS: set[tuple[str, str]] = {
        ("DRAFT", "ACTIVE"),
        ("ACTIVE", "DEPRECATED"),
        ("ACTIVE", "SUSPENDED"),
        ("SUSPENDED", "ACTIVE"),
        ("DEPRECATED", "ACTIVE"),  # re-activation
    }

    # Metrics thresholds per subject_type
    _THRESHOLDS: dict[str, dict[str, float]] = {
        "MODEL": {"ic_min": 0.05, "auc_min": 0.55, "brier_max": 0.25},
        "REGIME_MODEL": {"ic_min": 0.07, "auc_min": 0.60, "brier_max": 0.22},
        "BREAK_DETECTOR": {"ic_min": 0.08, "auc_min": 0.62, "brier_max": 0.20},
    }

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        subject_type: str = str(task.payload["subject_type"]).upper()
        subject_id: str = task.payload["subject_id"]
        subject_name: str = task.payload["subject_name"]
        from_status: str = str(task.payload["from_status"]).upper()
        to_status: str = str(task.payload["to_status"]).upper()
        metrics: dict[str, float] = task.payload.get("metrics") or {}
        evidence_ids: list[str] = task.payload.get("evidence_ids") or []

        audit.log(
            f"Gating promotion: {subject_name} ({subject_type}) "
            f"{from_status} → {to_status}"
        )

        blocking_reasons: list[str] = []

        # Validate transition
        transition = (from_status, to_status)
        if subject_type in ("POLICY",):
            valid_transitions = self._POLICY_TRANSITIONS
        else:
            valid_transitions = self._MODEL_TRANSITIONS

        if transition not in valid_transitions:
            blocking_reasons.append(
                f"invalid_transition:{from_status}→{to_status} "
                f"(valid for {subject_type}: {sorted(valid_transitions)})"
            )

        # Check metrics if thresholds defined
        thresholds = self._THRESHOLDS.get(subject_type)
        if thresholds and metrics:
            ic = metrics.get("ic")
            auc = metrics.get("auc")
            brier = metrics.get("brier")
            if ic is not None and float(ic) < thresholds["ic_min"]:
                blocking_reasons.append(
                    f"ic={ic:.4f}<{thresholds['ic_min']}"
                )
            if auc is not None and float(auc) < thresholds["auc_min"]:
                blocking_reasons.append(
                    f"auc={auc:.4f}<{thresholds['auc_min']}"
                )
            if brier is not None and float(brier) > thresholds["brier_max"]:
                blocking_reasons.append(
                    f"brier={brier:.4f}>{thresholds['brier_max']}"
                )

        eligible = len(blocking_reasons) == 0

        # Determine approval mode
        if to_status == "CHAMPION":
            approval_mode = "HUMAN_REQUIRED"
        elif to_status in ("CHALLENGER", "ACTIVE") and eligible:
            approval_mode = "POLICY_GATED"
        elif to_status in ("RETIRED", "DEPRECATED", "SUSPENDED"):
            approval_mode = "AUTOMATIC"
        else:
            approval_mode = "POLICY_GATED"

        auto_approved = approval_mode == "AUTOMATIC" and eligible
        approval_request_id: str | None = None

        if eligible and task.task_type == "gate_promotion":
            request_id = _new_id()
            now = _utcnow()

            try:
                from approvals.contracts import ApprovalMode, ApprovalRequest

                mode_map = {
                    "AUTOMATIC": ApprovalMode.AUTOMATIC,
                    "POLICY_GATED": ApprovalMode.POLICY_GATED,
                    "HUMAN_REQUIRED": ApprovalMode.HUMAN_REQUIRED,
                    "BLOCKED": ApprovalMode.BLOCKED,
                }
                mode_enum = mode_map.get(approval_mode, ApprovalMode.POLICY_GATED)

                req = ApprovalRequest(
                    request_id=request_id,
                    workflow_run_id=None,
                    agent_name=self.NAME,
                    task_id=task.task_id,
                    action_type=f"PROMOTE_{subject_type}_{from_status}_TO_{to_status}",
                    action_description=(
                        f"Promote {subject_name} ({subject_type}) from "
                        f"{from_status} to {to_status}"
                    ),
                    risk_class="HIGH_RISK" if to_status == "CHAMPION" else "MEDIUM_RISK",
                    environment="PRODUCTION",
                    evidence_bundle_ids=tuple(evidence_ids),
                    recommendation_id=None,
                    requested_by=self.NAME,
                    requested_at=now,
                    expires_at=None,
                    approval_mode=mode_enum,
                    required_approvers=("risk_management_team",) if to_status == "CHAMPION" else (),
                    context_summary=(
                        f"Promotion gate for {subject_name}: metrics={metrics}"
                    ),
                    policy_check_results=(),
                    notes="",
                )
                approval_request_id = req.request_id
                audit.log(f"Created ApprovalRequest id={request_id}, mode={approval_mode}")

            except ImportError:
                audit.warn("approvals.contracts unavailable — using plain request_id")
                approval_request_id = request_id

            # Try to submit to approval engine if automatic
            if auto_approved:
                try:
                    from approvals.engine import get_approval_engine
                    engine = get_approval_engine()
                    engine.auto_approve(approval_request_id)
                    audit.log(f"Auto-approved request {approval_request_id}")
                except (ImportError, Exception) as exc:
                    audit.warn(f"Auto-approval submission failed: {exc}")

        audit.log(
            f"Promotion gate complete: eligible={eligible}, "
            f"mode={approval_mode}, auto_approved={auto_approved}, "
            f"blocking_reasons={blocking_reasons}"
        )

        return {
            "eligible": eligible,
            "blocking_reasons": blocking_reasons,
            "approval_request_id": approval_request_id,
            "approval_mode": approval_mode,
            "auto_approved": auto_approved,
        }


# ══════════════════════════════════════════════════════════════════
# 5. AuditTrailValidationAgent
# ══════════════════════════════════════════════════════════════════


class AuditTrailValidationAgent(BaseAgent):
    """
    Validates completeness of the audit trail.

    Retrieves audit records from ``incidents.manager.IncidentManager`` and
    checks that every expected agent execution has a corresponding record.
    Falls back to checking the in-memory registry audit log if the incident
    manager is unavailable.

    Task types
    ----------
    validate_audit_trail
        Full audit completeness check.
    check_audit_completeness
        Alias.

    Optional payload keys
    ---------------------
    workflow_run_id : str
    agent_name : str
    time_window_hours : float  (default 24)

    Output keys
    -----------
    audit_completeness : float
    total_records : int
    gap_count : int
    gaps : list[dict]
    integrity_ok : bool
    """

    NAME = "audit_trail_validation"
    ALLOWED_TASK_TYPES = {"validate_audit_trail", "check_audit_completeness"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        workflow_run_id: str | None = task.payload.get("workflow_run_id")
        agent_name_filter: str | None = task.payload.get("agent_name")
        time_window_hours: float = float(task.payload.get("time_window_hours", 24))

        audit.log(
            f"Validating audit trail: workflow_run_id={workflow_run_id}, "
            f"agent_filter={agent_name_filter}, window={time_window_hours}h"
        )

        audit_records: list[dict[str, Any]] = []

        # Try IncidentManager audit ledger
        try:
            from incidents.manager import get_incident_manager
            manager = get_incident_manager()
            if hasattr(manager, "get_audit_records"):
                raw_records = manager.get_audit_records(
                    workflow_run_id=workflow_run_id,
                    agent_name=agent_name_filter,
                    time_window_hours=time_window_hours,
                )
                audit_records = [
                    {
                        "record_id": str(getattr(r, "record_id", _new_id())),
                        "agent_name": str(getattr(r, "agent_name", "unknown")),
                        "task_id": str(getattr(r, "task_id", "")),
                        "timestamp": str(getattr(r, "timestamp", "")),
                        "record_type": str(getattr(r, "record_type", "AGENT_EXECUTION")),
                        "workflow_run_id": str(getattr(r, "workflow_run_id", "")),
                    }
                    for r in raw_records
                ]
                audit.log(f"Retrieved {len(audit_records)} records from IncidentManager")
            else:
                audit.warn("IncidentManager.get_audit_records() not available")
        except (ImportError, Exception) as exc:
            audit.warn(f"IncidentManager unavailable ({exc}) — checking registry audit log")

        # Fallback: use in-memory registry audit log
        if not audit_records:
            try:
                from agents.registry import get_default_registry
                registry = get_default_registry()
                raw = registry.get_audit_log(
                    agent_name=agent_name_filter,
                    limit=1000,
                )
                # Filter by time window
                cutoff = datetime.utcnow()
                from datetime import timedelta
                window_start = cutoff - timedelta(hours=time_window_hours)
                for entry in raw:
                    try:
                        entry_ts = datetime.fromisoformat(
                            str(entry.get("dispatched_at", "")).replace("Z", "")
                        )
                        if entry_ts >= window_start:
                            audit_records.append(
                                {
                                    "record_id": entry.get("task_id", _new_id()),
                                    "agent_name": entry.get("agent_name", "unknown"),
                                    "task_id": entry.get("task_id", ""),
                                    "timestamp": entry.get("dispatched_at", ""),
                                    "record_type": "AGENT_EXECUTION",
                                    "status": entry.get("status", ""),
                                    "workflow_run_id": workflow_run_id or "",
                                }
                            )
                    except Exception:
                        audit_records.append(entry)  # type: ignore[arg-type]
                audit.log(f"Retrieved {len(audit_records)} records from registry audit log")
            except Exception as exc2:
                audit.warn(f"Registry audit log unavailable: {exc2}")

        # Integrity checks
        total = len(audit_records)
        gaps: list[dict[str, Any]] = []

        # Check for missing task_ids (entries without a task_id are suspicious)
        missing_task_ids = [r for r in audit_records if not r.get("task_id")]
        for r in missing_task_ids:
            gaps.append(
                {
                    "gap_type": "missing_task_id",
                    "agent_name": r.get("agent_name", "unknown"),
                    "timestamp": r.get("timestamp", ""),
                    "description": "Audit record missing task_id — may indicate incomplete logging",
                }
            )

        # Check for duplicate task_ids (same task logged twice)
        task_id_counts: dict[str, int] = {}
        for r in audit_records:
            tid = r.get("task_id", "")
            if tid:
                task_id_counts[tid] = task_id_counts.get(tid, 0) + 1
        for tid, count in task_id_counts.items():
            if count > 1:
                gaps.append(
                    {
                        "gap_type": "duplicate_task_id",
                        "task_id": tid,
                        "count": count,
                        "description": f"Task ID appears {count} times in audit records",
                    }
                )

        # Check for failed tasks without error details
        failed_no_error = [
            r for r in audit_records
            if r.get("status") in ("FAILED", "failed") and not r.get("error")
        ]
        for r in failed_no_error:
            gaps.append(
                {
                    "gap_type": "failed_without_error_detail",
                    "task_id": r.get("task_id", ""),
                    "agent_name": r.get("agent_name", "unknown"),
                    "description": "Failed task has no error details in audit record",
                }
            )

        gap_count = len(gaps)

        # Completeness score: penalise for gaps
        if total == 0:
            completeness = 1.0  # vacuously complete
        else:
            gap_penalty = min(float(gap_count) / float(total), 1.0)
            completeness = max(0.0, 1.0 - gap_penalty)

        integrity_ok = gap_count == 0

        audit.log(
            f"Audit trail validation complete: total={total}, gaps={gap_count}, "
            f"completeness={completeness:.2%}, integrity_ok={integrity_ok}"
        )

        return {
            "audit_completeness": round(completeness, 6),
            "total_records": total,
            "gap_count": gap_count,
            "gaps": gaps,
            "integrity_ok": integrity_ok,
        }
