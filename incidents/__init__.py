# -*- coding: utf-8 -*-
"""
incidents — Incident lifecycle management and audit ledger.

Public API
----------
Contracts
~~~~~~~~~
IncidentRecord          — mutable incident lifecycle record
RunbookReference        — reference to an operational runbook
RemediationPlan         — mutable remediation plan with ordered steps
RemediationStep         — single remediation step (frozen)
PostmortemArtifact      — immutable blameless postmortem document
AuditRecord             — immutable append-only audit ledger entry

Enumerations
~~~~~~~~~~~~
IncidentSeverity        — P0_CRITICAL | P1_HIGH | P2_MEDIUM | P3_LOW | P4_INFO
IncidentStatus          — OPEN | IN_PROGRESS | MONITORING | RESOLVED | CLOSED | POSTMORTEM
AuditRecordType         — AGENT_EXECUTION | POLICY_CHECK | APPROVAL_DECISION |
                          WORKFLOW_TRANSITION | DELEGATION | ESCALATION |
                          EMERGENCY_ACTION | INCIDENT_CREATED | INCIDENT_UPDATED | OVERRIDE

Manager
~~~~~~~
IncidentManager         — thread-safe incident and audit ledger manager
get_incident_manager    — singleton factory
"""

from incidents.contracts import (
    AuditRecord,
    AuditRecordType,
    IncidentRecord,
    IncidentSeverity,
    IncidentStatus,
    PostmortemArtifact,
    RemediationPlan,
    RemediationStep,
    RunbookReference,
)
from incidents.manager import IncidentManager, get_incident_manager

__all__ = [
    # Contracts
    "IncidentRecord",
    "RunbookReference",
    "RemediationPlan",
    "RemediationStep",
    "PostmortemArtifact",
    "AuditRecord",
    # Enumerations
    "IncidentSeverity",
    "IncidentStatus",
    "AuditRecordType",
    # Manager
    "IncidentManager",
    "get_incident_manager",
]
