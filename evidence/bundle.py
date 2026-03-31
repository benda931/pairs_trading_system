# -*- coding: utf-8 -*-
"""
evidence/bundle.py — Evidence collection, bundling, and completeness reporting.

An EvidenceBundle groups EvidenceItems against a set of EvidenceRequirements
for a single reviewable entity (strategy, model, deployment, policy change).
EvidenceBundleBuilder is a mutable helper for constructing bundles incrementally
before freezing them via build().
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

class EvidenceType(Enum):
    """Classification of evidence artefact kinds."""
    BACKTEST_RESULT = "BACKTEST_RESULT"
    WALK_FORWARD_RESULT = "WALK_FORWARD_RESULT"
    GOVERNANCE_CHECK = "GOVERNANCE_CHECK"
    APPROVAL_DECISION = "APPROVAL_DECISION"
    AUDIT_TRAIL = "AUDIT_TRAIL"
    PERFORMANCE_REPORT = "PERFORMANCE_REPORT"
    DRIFT_REPORT = "DRIFT_REPORT"
    RECONCILIATION_REPORT = "RECONCILIATION_REPORT"
    MANUAL_ATTESTATION = "MANUAL_ATTESTATION"
    REGULATORY_FILING = "REGULATORY_FILING"
    THIRD_PARTY_AUDIT = "THIRD_PARTY_AUDIT"
    PEER_REVIEW = "PEER_REVIEW"
    SYSTEM_LOG_EXPORT = "SYSTEM_LOG_EXPORT"
    MODEL_CARD = "MODEL_CARD"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    COMPLIANCE_SIGN_OFF = "COMPLIANCE_SIGN_OFF"


class EvidenceStatus(Enum):
    """Lifecycle state of a single evidence item."""
    PENDING = "PENDING"
    COLLECTED = "COLLECTED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceItem:
    """
    A single immutable piece of evidence.

    ``tags`` and ``metadata`` use tuples for frozen compatibility.
    Convert ``metadata`` back to a dict with ``dict(item.metadata)``.
    """
    evidence_id: str
    evidence_type: EvidenceType
    status: EvidenceStatus
    title: str
    description: str
    collected_at: datetime
    collected_by: str
    expiry_date: Optional[datetime]
    artifact_ref: str                           # path, URL, or system reference
    checksum: Optional[str]                     # sha256 of artefact if available
    tags: tuple[str, ...]                       # searchable tags
    metadata: tuple[tuple[str, Any], ...]       # frozen-compatible key-value pairs


@dataclass(frozen=True)
class EvidenceRequirement:
    """
    Declares what evidence is needed to satisfy a review gate.

    A *mandatory* requirement is a hard blocker; non-mandatory requirements
    are advisory and appear in the completeness report but do not set
    ``is_complete=False``.
    """
    requirement_id: str
    description: str
    required_evidence_types: tuple[EvidenceType, ...]
    min_count: int      # minimum number of items of any of the required types
    mandatory: bool     # True = hard blocker; False = advisory


@dataclass(frozen=True)
class EvidenceBundle:
    """
    Immutable snapshot of collected evidence items and their requirements
    for a specific entity and review purpose.
    """
    bundle_id: str
    entity_type: str        # "strategy" | "model" | "deployment" | "policy_change"
    entity_id: str
    purpose: str            # "promotion_review" | "compliance_audit" | "incident_postmortem"
    created_at: datetime
    created_by: str
    evidence_items: tuple[EvidenceItem, ...]
    requirements: tuple[EvidenceRequirement, ...]


@dataclass(frozen=True)
class EvidenceCompletenessReport:
    """
    Completeness assessment of an EvidenceBundle against its requirements.

    ``is_complete`` is True only when there are zero missing mandatory
    requirements AND zero expired items.
    """
    bundle_id: str
    is_complete: bool
    total_items: int
    verified_items: int
    missing_requirements: tuple[str, ...]   # requirement_ids that are not satisfied
    expired_items: tuple[str, ...]          # evidence_ids that have expired
    completeness_pct: float                 # verified_items / max(total_items, 1) * 100
    generated_at: datetime


# ---------------------------------------------------------------------------
# EvidenceBundleBuilder
# ---------------------------------------------------------------------------

class EvidenceBundleBuilder:
    """
    Mutable builder for constructing an EvidenceBundle incrementally.

    Usage::

        builder = EvidenceBundleBuilder(
            entity_type="model",
            entity_id="regime_classifier_v3",
            purpose="promotion_review",
            created_by="ml_team",
        )
        builder.add_requirement(
            "Walk-forward backtest required",
            required_types=[EvidenceType.WALK_FORWARD_RESULT],
            mandatory=True,
        )
        item = builder.collect(
            evidence_type=EvidenceType.WALK_FORWARD_RESULT,
            title="WF Backtest v3 — 2024-Q1",
            description="5-fold walk-forward, IC=0.07",
            collected_by="ml_team",
            artifact_ref="s3://ml-artefacts/wf_v3_2024Q1.json",
        )
        bundle = builder.build()
        report = builder.check_completeness()
    """

    def __init__(
        self,
        entity_type: str,
        entity_id: str,
        purpose: str,
        created_by: str,
    ) -> None:
        self.bundle_id: str = str(uuid.uuid4())
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.purpose = purpose
        self.created_by = created_by
        self.created_at: datetime = datetime.utcnow()
        self._items: list[EvidenceItem] = []
        self._requirements: list[EvidenceRequirement] = []

    # ------------------------------------------------------------------
    # Requirements
    # ------------------------------------------------------------------

    def add_requirement(
        self,
        description: str,
        required_types: list[EvidenceType],
        min_count: int = 1,
        mandatory: bool = True,
    ) -> EvidenceRequirement:
        """Declare a new evidence requirement for this bundle."""
        req = EvidenceRequirement(
            requirement_id=str(uuid.uuid4()),
            description=description,
            required_evidence_types=tuple(required_types),
            min_count=min_count,
            mandatory=mandatory,
        )
        self._requirements.append(req)
        return req

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------

    def collect(
        self,
        evidence_type: EvidenceType,
        title: str,
        description: str,
        collected_by: str,
        artifact_ref: str = "",
        expiry_date: Optional[datetime] = None,
        checksum: Optional[str] = None,
        tags: tuple[str, ...] = (),
        metadata: Optional[dict] = None,
    ) -> EvidenceItem:
        """Collect a new evidence item and add it to this bundle."""
        item = EvidenceItem(
            evidence_id=str(uuid.uuid4()),
            evidence_type=evidence_type,
            status=EvidenceStatus.COLLECTED,
            title=title,
            description=description,
            collected_at=datetime.utcnow(),
            collected_by=collected_by,
            expiry_date=expiry_date,
            artifact_ref=artifact_ref,
            checksum=checksum,
            tags=tuple(tags),
            metadata=tuple((k, v) for k, v in (metadata or {}).items()),
        )
        self._items.append(item)
        return item

    # ------------------------------------------------------------------
    # Build / check
    # ------------------------------------------------------------------

    def build(self) -> EvidenceBundle:
        """Freeze the current state into an immutable EvidenceBundle."""
        return EvidenceBundle(
            bundle_id=self.bundle_id,
            entity_type=self.entity_type,
            entity_id=self.entity_id,
            purpose=self.purpose,
            created_at=self.created_at,
            created_by=self.created_by,
            evidence_items=tuple(self._items),
            requirements=tuple(self._requirements),
        )

    def check_completeness(self) -> EvidenceCompletenessReport:
        """
        Evaluate how well the collected evidence satisfies the declared
        requirements.

        An item is considered *active* (eligible to satisfy requirements)
        when its status is COLLECTED or VERIFIED and it has not expired.
        """
        now = datetime.utcnow()
        expired_ids: list[str] = []

        for item in self._items:
            if item.expiry_date is not None and item.expiry_date < now:
                expired_ids.append(item.evidence_id)

        expired_set = set(expired_ids)
        active_items = [
            i for i in self._items
            if i.status in (EvidenceStatus.COLLECTED, EvidenceStatus.VERIFIED)
            and i.evidence_id not in expired_set
        ]

        missing: list[str] = []
        for req in self._requirements:
            count = sum(
                1 for i in active_items
                if i.evidence_type in req.required_evidence_types
            )
            if count < req.min_count:
                missing.append(req.requirement_id)

        mandatory_missing = [
            r.requirement_id
            for r in self._requirements
            if r.mandatory and r.requirement_id in missing
        ]

        total = len(self._items)
        verified_count = len(active_items)
        completeness = verified_count / max(total, 1)

        return EvidenceCompletenessReport(
            bundle_id=self.bundle_id,
            is_complete=len(mandatory_missing) == 0 and len(expired_ids) == 0,
            total_items=total,
            verified_items=verified_count,
            missing_requirements=tuple(missing),
            expired_items=tuple(expired_ids),
            completeness_pct=round(completeness * 100, 1),
            generated_at=now,
        )
