# -*- coding: utf-8 -*-
"""
approvals — Approval lifecycle management for agent actions.

Public API
----------
Contracts
~~~~~~~~~
ApprovalRequest       — formal action-gate request
ApprovalDecision      — outcome of an adjudicated request
OverrideRecord        — audit trail for policy bypass events
HumanReviewTicket     — queued ticket for human reviewer
EscalationRecord      — escalation from one tier to another

Enumerations
~~~~~~~~~~~~
ApprovalStatus        — PENDING | APPROVED | REJECTED | ESCALATED | EXPIRED | AUTO_APPROVED
ReviewPriority        — ROUTINE | ELEVATED | URGENT | CRITICAL
ApprovalMode          — AUTOMATIC | POLICY_GATED | HUMAN_REQUIRED | DUAL_APPROVAL | BLOCKED

Engine
~~~~~~
ApprovalEngine        — thread-safe adjudication engine
get_approval_engine   — singleton factory
"""

from approvals.contracts import (
    ApprovalDecision,
    ApprovalMode,
    ApprovalRequest,
    ApprovalStatus,
    EscalationRecord,
    HumanReviewTicket,
    OverrideRecord,
    ReviewPriority,
)
from approvals.engine import ApprovalEngine, get_approval_engine

__all__ = [
    # Contracts
    "ApprovalRequest",
    "ApprovalDecision",
    "OverrideRecord",
    "HumanReviewTicket",
    "EscalationRecord",
    # Enumerations
    "ApprovalStatus",
    "ReviewPriority",
    "ApprovalMode",
    # Engine
    "ApprovalEngine",
    "get_approval_engine",
]
