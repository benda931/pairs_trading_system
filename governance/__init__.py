# -*- coding: utf-8 -*-
"""
governance — Governance policy evaluation and guardrail enforcement.

Public API
----------
Contracts
~~~~~~~~~
PolicyCheckResult         — result of a single policy evaluation
GuardrailViolation        — recorded hard/soft guardrail violation
GovernancePolicyVersion   — versioned policy with approval metadata
ArtifactReference         — lightweight artifact reference with provenance
ChangeImpactReport        — structured change impact assessment
PromotionReviewRecord     — governance review for model/policy promotion

Enumerations
~~~~~~~~~~~~
PolicyStatus              — ACTIVE | DEPRECATED | DRAFT | SUSPENDED
PolicyViolationSeverity   — INFO | WARNING | VIOLATION | CRITICAL | EMERGENCY

Engine
~~~~~~
GovernancePolicyEngine    — thread-safe policy evaluation engine
get_governance_engine     — singleton factory
"""

from governance.contracts import (
    ArtifactReference,
    ChangeImpactReport,
    GovernancePolicyVersion,
    GuardrailViolation,
    PolicyCheckResult,
    PolicyStatus,
    PolicyViolationSeverity,
    PromotionReviewRecord,
)
from governance.engine import GovernancePolicyEngine, get_governance_engine

__all__ = [
    # Contracts
    "PolicyCheckResult",
    "GuardrailViolation",
    "GovernancePolicyVersion",
    "ArtifactReference",
    "ChangeImpactReport",
    "PromotionReviewRecord",
    # Enumerations
    "PolicyStatus",
    "PolicyViolationSeverity",
    # Engine
    "GovernancePolicyEngine",
    "get_governance_engine",
]
