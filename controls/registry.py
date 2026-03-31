# -*- coding: utf-8 -*-
"""
controls/registry.py — Control definitions, test history, and status tracking.

ControlRegistry ships with 15 canonical system controls pre-registered and
ACTIVE.  Additional controls can be registered at runtime via register().
Test results are appended via record_test(), which auto-updates the control
status based on the outcome.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ControlDomain(Enum):
    """Risk/governance domain that a control belongs to."""
    DATA_QUALITY = "DATA_QUALITY"
    MODEL_RISK = "MODEL_RISK"
    EXECUTION_RISK = "EXECUTION_RISK"
    MARKET_RISK = "MARKET_RISK"
    OPERATIONAL_RISK = "OPERATIONAL_RISK"
    COMPLIANCE = "COMPLIANCE"
    GOVERNANCE = "GOVERNANCE"
    TECHNOLOGY = "TECHNOLOGY"
    HUMAN_OVERSIGHT = "HUMAN_OVERSIGHT"
    SEGREGATION_OF_DUTIES = "SEGREGATION_OF_DUTIES"


class ControlType(Enum):
    """Nature of a control — how it acts against a risk."""
    PREVENTIVE = "PREVENTIVE"       # stops the risk event from occurring
    DETECTIVE = "DETECTIVE"         # identifies the risk event after occurrence
    CORRECTIVE = "CORRECTIVE"       # remedies the consequences after detection
    COMPENSATING = "COMPENSATING"   # mitigates residual risk when primary is unavailable
    DIRECTIVE = "DIRECTIVE"         # establishes expected behaviour (policy, procedure)


class ControlFrequency(Enum):
    """How often a control operates or is assessed."""
    CONTINUOUS = "CONTINUOUS"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUAL = "ANNUAL"
    AD_HOC = "AD_HOC"
    EVENT_DRIVEN = "EVENT_DRIVEN"


class ControlStatus(Enum):
    """Current operational state of a control."""
    DESIGNED = "DESIGNED"           # documented but not yet active
    ACTIVE = "ACTIVE"               # operating as designed
    DEGRADED = "DEGRADED"           # partially failing / reduced effectiveness
    FAILED = "FAILED"               # not operating; risk unmitigated
    RETIRED = "RETIRED"             # decommissioned
    UNDER_REVIEW = "UNDER_REVIEW"   # under assessment for change


class ControlTestResult(Enum):
    """Outcome of a control effectiveness test."""
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControlDefinition:
    """
    Immutable definition of a single internal control.

    ``related_risks`` links this control to risk register entries by ID.
    ``automated`` distinguishes system-enforced controls from procedural ones.
    ``critical`` marks single points of failure when degraded.
    """
    control_id: str
    name: str
    description: str
    domain: ControlDomain
    control_type: ControlType
    frequency: ControlFrequency
    owner: str                          # team/role responsible
    automated: bool                     # True = system-enforced; False = manual/procedural
    critical: bool                      # True = single point of failure when degraded
    related_risks: tuple[str, ...]      # risk register IDs this control mitigates
    implementation_notes: str = ""


@dataclass(frozen=True)
class ControlTestRecord:
    """
    Immutable record of a single control effectiveness test.

    ``evidence_bundle_id`` links to an EvidenceBundle if one was created
    for this test.  ``remediation_deadline`` is only meaningful when
    ``remediation_required=True``.
    """
    test_id: str
    control_id: str
    tested_at: datetime
    tested_by: str
    result: ControlTestResult
    findings: str
    evidence_bundle_id: Optional[str]
    remediation_required: bool
    remediation_deadline: Optional[datetime]
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ControlOwnerRecord:
    """
    Ownership assignment record for a control.

    ``review_cadence_days`` drives the review schedule.
    ``next_review_due`` is set externally by a scheduler or set to None
    when no cadence-based review has been scheduled yet.
    """
    control_id: str
    owner: str
    assigned_at: datetime
    assigned_by: str
    review_cadence_days: int
    last_reviewed: Optional[datetime]
    next_review_due: Optional[datetime]


# ---------------------------------------------------------------------------
# ControlRegistry
# ---------------------------------------------------------------------------

class ControlRegistry:
    """
    Registry of ControlDefinitions with test history and status tracking.

    The registry is pre-populated with 15 canonical system controls on
    construction.  All 15 default controls are ACTIVE.

    Thread safety: this class is not internally synchronised.  Callers that
    share a registry across threads should provide external locking.
    """

    def __init__(self) -> None:
        self._controls: dict[str, ControlDefinition] = {}
        self._statuses: dict[str, ControlStatus] = {}
        self._test_records: dict[str, list[ControlTestRecord]] = {}
        self._owners: dict[str, ControlOwnerRecord] = {}
        self._populate_defaults()

    # ------------------------------------------------------------------
    # Default controls
    # ------------------------------------------------------------------

    def _populate_defaults(self) -> None:
        """Register the 15 canonical system controls as ACTIVE."""
        defaults: list[ControlDefinition] = [
            ControlDefinition(
                "CTRL-DQ-001",
                "Price Data Freshness Check",
                "Ensure price data is updated within 24h before market open",
                ControlDomain.DATA_QUALITY,
                ControlType.PREVENTIVE,
                ControlFrequency.DAILY,
                "data_team",
                True,
                True,
                ("RISK-DATA-001",),
            ),
            ControlDefinition(
                "CTRL-DQ-002",
                "Spread Computation Integrity",
                "Validate spread values are within expected ranges before signal emission",
                ControlDomain.DATA_QUALITY,
                ControlType.DETECTIVE,
                ControlFrequency.CONTINUOUS,
                "quant_team",
                True,
                True,
                ("RISK-DATA-002",),
            ),
            ControlDefinition(
                "CTRL-MR-001",
                "Model Drift Monitoring",
                "Detect feature/concept drift via PSI checks on every inference batch",
                ControlDomain.MODEL_RISK,
                ControlType.DETECTIVE,
                ControlFrequency.DAILY,
                "ml_team",
                True,
                True,
                ("RISK-MODEL-001",),
            ),
            ControlDefinition(
                "CTRL-MR-002",
                "ML Leakage Prevention",
                "Enforce train/test boundary in all model training pipelines",
                ControlDomain.MODEL_RISK,
                ControlType.PREVENTIVE,
                ControlFrequency.EVENT_DRIVEN,
                "ml_team",
                True,
                True,
                ("RISK-MODEL-002",),
            ),
            ControlDefinition(
                "CTRL-MR-003",
                "Champion Promotion Governance",
                "Require IC>=0.05, AUC>=0.55, Brier<=0.25 before champion promotion",
                ControlDomain.MODEL_RISK,
                ControlType.DIRECTIVE,
                ControlFrequency.EVENT_DRIVEN,
                "governance_team",
                True,
                True,
                ("RISK-MODEL-003",),
            ),
            ControlDefinition(
                "CTRL-ER-001",
                "Kill Switch Circuit Breaker",
                "Halt all order flow within 100ms on kill-switch engagement",
                ControlDomain.EXECUTION_RISK,
                ControlType.PREVENTIVE,
                ControlFrequency.CONTINUOUS,
                "trading_ops",
                True,
                True,
                ("RISK-EXEC-001",),
            ),
            ControlDefinition(
                "CTRL-ER-002",
                "Position Reconciliation",
                "Daily reconcile broker positions vs internal ledger; block on critical diffs",
                ControlDomain.EXECUTION_RISK,
                ControlType.DETECTIVE,
                ControlFrequency.DAILY,
                "trading_ops",
                True,
                True,
                ("RISK-EXEC-002",),
            ),
            ControlDefinition(
                "CTRL-ER-003",
                "Hedge Ratio Drift Alert",
                "Alert on hedge ratio drift >10%, block on drift >25%",
                ControlDomain.EXECUTION_RISK,
                ControlType.DETECTIVE,
                ControlFrequency.CONTINUOUS,
                "risk_team",
                True,
                True,
                ("RISK-EXEC-003",),
            ),
            ControlDefinition(
                "CTRL-MKT-001",
                "Regime Classification Guardrail",
                "Block new entries in CRISIS/BROKEN regimes regardless of signal quality",
                ControlDomain.MARKET_RISK,
                ControlType.PREVENTIVE,
                ControlFrequency.CONTINUOUS,
                "risk_team",
                True,
                True,
                ("RISK-MKT-001",),
            ),
            ControlDefinition(
                "CTRL-MKT-002",
                "Portfolio Drawdown Limits",
                "Enforce 4-tier drawdown heat model with automatic de-risking",
                ControlDomain.MARKET_RISK,
                ControlType.PREVENTIVE,
                ControlFrequency.CONTINUOUS,
                "risk_team",
                True,
                True,
                ("RISK-MKT-002",),
            ),
            ControlDefinition(
                "CTRL-OPS-001",
                "Deployment Freeze Enforcement",
                "Block deployments during configured freeze windows",
                ControlDomain.OPERATIONAL_RISK,
                ControlType.PREVENTIVE,
                ControlFrequency.EVENT_DRIVEN,
                "devops",
                True,
                True,
                ("RISK-OPS-001",),
            ),
            ControlDefinition(
                "CTRL-OPS-002",
                "Secrets Never Logged",
                "Enforce SecretReference abstraction; secrets never appear in logs or audit trails",
                ControlDomain.OPERATIONAL_RISK,
                ControlType.PREVENTIVE,
                ControlFrequency.CONTINUOUS,
                "security_team",
                True,
                True,
                ("RISK-OPS-002",),
            ),
            ControlDefinition(
                "CTRL-GOV-001",
                "Dual-Control for Live Activation",
                "Require two approvers for any live-trading activation",
                ControlDomain.GOVERNANCE,
                ControlType.DIRECTIVE,
                ControlFrequency.EVENT_DRIVEN,
                "governance_team",
                False,
                True,
                ("RISK-GOV-001",),
            ),
            ControlDefinition(
                "CTRL-SOD-001",
                "Signal-to-Execution Segregation",
                "Agents proposing signals must not directly submit orders",
                ControlDomain.SEGREGATION_OF_DUTIES,
                ControlType.PREVENTIVE,
                ControlFrequency.CONTINUOUS,
                "governance_team",
                True,
                True,
                ("RISK-SOD-001",),
            ),
            ControlDefinition(
                "CTRL-HO-001",
                "Human Review for P0 Incidents",
                "All P0 incidents require human acknowledgment before resolution",
                ControlDomain.HUMAN_OVERSIGHT,
                ControlType.DIRECTIVE,
                ControlFrequency.EVENT_DRIVEN,
                "incident_team",
                False,
                True,
                ("RISK-INC-001",),
            ),
        ]
        for ctrl in defaults:
            self._controls[ctrl.control_id] = ctrl
            self._statuses[ctrl.control_id] = ControlStatus.ACTIVE
            self._test_records[ctrl.control_id] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        ctrl: ControlDefinition,
        initial_status: ControlStatus = ControlStatus.DESIGNED,
    ) -> None:
        """Register a new control. Existing controls are replaced."""
        self._controls[ctrl.control_id] = ctrl
        self._statuses[ctrl.control_id] = initial_status
        if ctrl.control_id not in self._test_records:
            self._test_records[ctrl.control_id] = []

    # ------------------------------------------------------------------
    # Test recording
    # ------------------------------------------------------------------

    def record_test(
        self,
        control_id: str,
        tested_by: str,
        result: ControlTestResult,
        findings: str,
        evidence_bundle_id: Optional[str] = None,
        remediation_required: bool = False,
        remediation_deadline: Optional[datetime] = None,
    ) -> ControlTestRecord:
        """
        Append a test result to a control's history and auto-update status.

        FAIL  → ControlStatus.FAILED
        PARTIAL → ControlStatus.DEGRADED
        PASS  → ControlStatus.ACTIVE (if previously FAILED or DEGRADED)
        """
        if control_id not in self._controls:
            raise KeyError(f"Control '{control_id}' is not registered")

        record = ControlTestRecord(
            test_id=str(uuid.uuid4()),
            control_id=control_id,
            tested_at=datetime.utcnow(),
            tested_by=tested_by,
            result=result,
            findings=findings,
            evidence_bundle_id=evidence_bundle_id,
            remediation_required=remediation_required,
            remediation_deadline=remediation_deadline,
        )
        self._test_records[control_id].append(record)

        # Auto-update status
        if result == ControlTestResult.FAIL:
            self._statuses[control_id] = ControlStatus.FAILED
        elif result == ControlTestResult.PARTIAL:
            self._statuses[control_id] = ControlStatus.DEGRADED
        elif result == ControlTestResult.PASS:
            if self._statuses[control_id] in (ControlStatus.FAILED, ControlStatus.DEGRADED):
                self._statuses[control_id] = ControlStatus.ACTIVE

        return record

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def update_status(self, control_id: str, status: ControlStatus) -> None:
        """Manually override the status of a registered control."""
        if control_id not in self._controls:
            raise KeyError(f"Control '{control_id}' is not registered")
        self._statuses[control_id] = status

    # ------------------------------------------------------------------
    # Ownership
    # ------------------------------------------------------------------

    def assign_owner(
        self,
        control_id: str,
        owner: str,
        assigned_by: str,
        review_cadence_days: int = 90,
    ) -> ControlOwnerRecord:
        """Assign or reassign ownership of a control."""
        rec = ControlOwnerRecord(
            control_id=control_id,
            owner=owner,
            assigned_at=datetime.utcnow(),
            assigned_by=assigned_by,
            review_cadence_days=review_cadence_days,
            last_reviewed=None,
            next_review_due=None,
        )
        self._owners[control_id] = rec
        return rec

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, control_id: str) -> Optional[ControlDefinition]:
        """Return the ControlDefinition for *control_id*, or None."""
        return self._controls.get(control_id)

    def get_status(self, control_id: str) -> Optional[ControlStatus]:
        """Return the current ControlStatus for *control_id*, or None."""
        return self._statuses.get(control_id)

    def get_test_history(self, control_id: str) -> list[ControlTestRecord]:
        """Return a copy of the test history for *control_id*."""
        return list(self._test_records.get(control_id, []))

    def get_failed_controls(self) -> list[ControlDefinition]:
        """Return all controls with status FAILED."""
        return [
            ctrl
            for ctrl_id, ctrl in self._controls.items()
            if self._statuses.get(ctrl_id) == ControlStatus.FAILED
        ]

    def get_degraded_controls(self) -> list[ControlDefinition]:
        """Return all controls with status DEGRADED."""
        return [
            ctrl
            for ctrl_id, ctrl in self._controls.items()
            if self._statuses.get(ctrl_id) == ControlStatus.DEGRADED
        ]

    def get_critical_failures(self) -> list[ControlDefinition]:
        """Return controls marked critical that are FAILED or DEGRADED."""
        return [
            ctrl
            for ctrl_id, ctrl in self._controls.items()
            if ctrl.critical
            and self._statuses.get(ctrl_id) in (ControlStatus.FAILED, ControlStatus.DEGRADED)
        ]

    def list_controls(
        self,
        domain: Optional[ControlDomain] = None,
    ) -> list[ControlDefinition]:
        """Return all registered controls, optionally filtered by domain."""
        ctrls = list(self._controls.values())
        if domain is not None:
            ctrls = [c for c in ctrls if c.domain == domain]
        return ctrls

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, int]:
        """Return a snapshot of control health metrics."""
        statuses = list(self._statuses.values())
        return {
            "total_controls": len(self._controls),
            "active": sum(1 for s in statuses if s == ControlStatus.ACTIVE),
            "failed": sum(1 for s in statuses if s == ControlStatus.FAILED),
            "degraded": sum(1 for s in statuses if s == ControlStatus.DEGRADED),
            "critical_failures": len(self.get_critical_failures()),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[ControlRegistry] = None


def get_control_registry() -> ControlRegistry:
    """Return the process-level singleton ControlRegistry."""
    global _registry
    if _registry is None:
        _registry = ControlRegistry()
    return _registry
