# -*- coding: utf-8 -*-
"""
retention/policy.py
===================
Data retention policy enforcement with legal hold support.

Manages archival schedules, deletion eligibility, and legal holds.
Retention periods are aligned with SEC Rule 17a-4 and FINRA Rule 4511
where applicable; internal policy governs the rest.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


# ── Enums ─────────────────────────────────────────────────────────────────────

class RetentionCategory(Enum):
    AUDIT_TRAIL = "AUDIT_TRAIL"
    TRADE_RECORDS = "TRADE_RECORDS"
    MODEL_ARTIFACTS = "MODEL_ARTIFACTS"
    RESEARCH_DATA = "RESEARCH_DATA"
    OPERATIONAL_LOGS = "OPERATIONAL_LOGS"
    INCIDENT_RECORDS = "INCIDENT_RECORDS"
    COMPLIANCE_RECORDS = "COMPLIANCE_RECORDS"
    ATTESTATIONS = "ATTESTATIONS"
    EXCEPTION_RECORDS = "EXCEPTION_RECORDS"


class ArchiveStatus(Enum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    PENDING_DELETION = "PENDING_DELETION"
    DELETED = "DELETED"
    UNDER_LEGAL_HOLD = "UNDER_LEGAL_HOLD"


# ── Contracts ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetentionPolicy:
    policy_id: str
    category: RetentionCategory
    min_retention_days: int
    max_retention_days: Optional[int]
    description: str
    regulatory_basis: str
    auto_archive_after_days: int
    auto_delete_after_days: Optional[int]


@dataclass(frozen=True)
class ArchiveRecord:
    archive_id: str
    entity_type: str
    entity_id: str
    category: RetentionCategory
    status: ArchiveStatus
    created_at: datetime
    last_accessed: Optional[datetime]
    archive_date: Optional[datetime]
    deletion_eligible_date: Optional[datetime]
    legal_hold: bool
    legal_hold_reason: Optional[str]
    storage_ref: str


@dataclass(frozen=True)
class LegalHold:
    hold_id: str
    reason: str
    imposed_by: str
    imposed_at: datetime
    entity_type: str
    entity_id_pattern: str
    lifted_at: Optional[datetime]
    lifted_by: Optional[str]
    active: bool


# ── Manager ───────────────────────────────────────────────────────────────────

class RetentionManager:
    """
    Data retention policy enforcement with legal hold support.

    Manages archival schedules, deletion eligibility, and legal holds.
    """

    DEFAULT_POLICIES: list[RetentionPolicy] = [
        RetentionPolicy(
            "RET-AUDIT", RetentionCategory.AUDIT_TRAIL,
            2555, None,
            "Audit chain entries", "SEC Rule 17a-4",
            365, None,
        ),
        RetentionPolicy(
            "RET-TRADE", RetentionCategory.TRADE_RECORDS,
            2555, None,
            "All trade and order records", "FINRA Rule 4511",
            180, None,
        ),
        RetentionPolicy(
            "RET-MODEL", RetentionCategory.MODEL_ARTIFACTS,
            1825, None,
            "ML model files and metadata", "Internal Policy",
            90, 2555,
        ),
        RetentionPolicy(
            "RET-RESEARCH", RetentionCategory.RESEARCH_DATA,
            1095, None,
            "Research outputs and backtests", "Internal Policy",
            180, 1825,
        ),
        RetentionPolicy(
            "RET-OPLOG", RetentionCategory.OPERATIONAL_LOGS,
            365, 730,
            "Operational and system logs", "Internal Policy",
            90, 730,
        ),
        RetentionPolicy(
            "RET-INCIDENT", RetentionCategory.INCIDENT_RECORDS,
            1825, None,
            "Incident records and postmortems", "Internal Policy",
            365, None,
        ),
        RetentionPolicy(
            "RET-COMPLIANCE", RetentionCategory.COMPLIANCE_RECORDS,
            2555, None,
            "Compliance reviews and attestations", "SEC Rule 17a-4",
            365, None,
        ),
        RetentionPolicy(
            "RET-ATTEST", RetentionCategory.ATTESTATIONS,
            1825, None,
            "Attestation records", "Internal Policy",
            180, 1825,
        ),
        RetentionPolicy(
            "RET-EXCEPTION", RetentionCategory.EXCEPTION_RECORDS,
            2555, None,
            "Exception requests and waivers", "Internal Policy",
            365, None,
        ),
    ]

    def __init__(self) -> None:
        self._policies: dict[RetentionCategory, RetentionPolicy] = {
            p.category: p for p in self.DEFAULT_POLICIES
        }
        self._records: dict[str, ArchiveRecord] = {}
        self._legal_holds: dict[str, LegalHold] = {}

    def register(
        self,
        entity_type: str,
        entity_id: str,
        category: RetentionCategory,
        storage_ref: str = "",
    ) -> ArchiveRecord:
        """Register a new artifact under the appropriate retention policy."""
        policy = self._policies[category]
        now = datetime.utcnow()
        deletion_eligible: Optional[datetime] = (
            now + timedelta(days=policy.min_retention_days)
            if policy.min_retention_days
            else None
        )

        record = ArchiveRecord(
            archive_id=str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            category=category,
            status=ArchiveStatus.ACTIVE,
            created_at=now,
            last_accessed=now,
            archive_date=None,
            deletion_eligible_date=deletion_eligible,
            legal_hold=False,
            legal_hold_reason=None,
            storage_ref=storage_ref,
        )
        self._records[record.archive_id] = record
        return record

    def update_last_accessed(self, archive_id: str) -> bool:
        """Update the last_accessed timestamp for a record."""
        record = self._records.get(archive_id)
        if not record:
            return False
        self._records[archive_id] = ArchiveRecord(
            archive_id=record.archive_id,
            entity_type=record.entity_type,
            entity_id=record.entity_id,
            category=record.category,
            status=record.status,
            created_at=record.created_at,
            last_accessed=datetime.utcnow(),
            archive_date=record.archive_date,
            deletion_eligible_date=record.deletion_eligible_date,
            legal_hold=record.legal_hold,
            legal_hold_reason=record.legal_hold_reason,
            storage_ref=record.storage_ref,
        )
        return True

    def archive_record(self, archive_id: str) -> bool:
        """Transition a record from ACTIVE to ARCHIVED status."""
        record = self._records.get(archive_id)
        if not record or record.status != ArchiveStatus.ACTIVE:
            return False
        self._records[archive_id] = ArchiveRecord(
            archive_id=record.archive_id,
            entity_type=record.entity_type,
            entity_id=record.entity_id,
            category=record.category,
            status=ArchiveStatus.ARCHIVED,
            created_at=record.created_at,
            last_accessed=record.last_accessed,
            archive_date=datetime.utcnow(),
            deletion_eligible_date=record.deletion_eligible_date,
            legal_hold=record.legal_hold,
            legal_hold_reason=record.legal_hold_reason,
            storage_ref=record.storage_ref,
        )
        return True

    def mark_pending_deletion(self, archive_id: str) -> bool:
        """Mark a record as pending deletion (awaiting physical removal)."""
        record = self._records.get(archive_id)
        if not record or record.legal_hold:
            return False
        self._records[archive_id] = ArchiveRecord(
            archive_id=record.archive_id,
            entity_type=record.entity_type,
            entity_id=record.entity_id,
            category=record.category,
            status=ArchiveStatus.PENDING_DELETION,
            created_at=record.created_at,
            last_accessed=record.last_accessed,
            archive_date=record.archive_date,
            deletion_eligible_date=record.deletion_eligible_date,
            legal_hold=record.legal_hold,
            legal_hold_reason=record.legal_hold_reason,
            storage_ref=record.storage_ref,
        )
        return True

    def impose_legal_hold(
        self,
        reason: str,
        imposed_by: str,
        entity_type: str,
        entity_id_pattern: str,
    ) -> LegalHold:
        """
        Impose a legal hold on all matching records.

        entity_id_pattern supports:
        - Exact match: "trade_001"
        - Wildcard suffix: "trade_*"
        - Global wildcard: "*" (all records of entity_type)
        """
        hold = LegalHold(
            hold_id=str(uuid.uuid4()),
            reason=reason,
            imposed_by=imposed_by,
            imposed_at=datetime.utcnow(),
            entity_type=entity_type,
            entity_id_pattern=entity_id_pattern,
            lifted_at=None,
            lifted_by=None,
            active=True,
        )
        self._legal_holds[hold.hold_id] = hold

        # Apply hold to all matching records
        for archive_id, record in list(self._records.items()):
            if record.entity_type != entity_type:
                continue
            match = (
                entity_id_pattern == "*"
                or record.entity_id == entity_id_pattern
                or (
                    entity_id_pattern.endswith("*")
                    and record.entity_id.startswith(entity_id_pattern.rstrip("*"))
                )
            )
            if match:
                self._records[archive_id] = ArchiveRecord(
                    archive_id=record.archive_id,
                    entity_type=record.entity_type,
                    entity_id=record.entity_id,
                    category=record.category,
                    status=ArchiveStatus.UNDER_LEGAL_HOLD,
                    created_at=record.created_at,
                    last_accessed=record.last_accessed,
                    archive_date=record.archive_date,
                    deletion_eligible_date=record.deletion_eligible_date,
                    legal_hold=True,
                    legal_hold_reason=reason,
                    storage_ref=record.storage_ref,
                )

        return hold

    def lift_legal_hold(self, hold_id: str, lifted_by: str) -> bool:
        """Lift a legal hold. Does not automatically restore records to prior status."""
        hold = self._legal_holds.get(hold_id)
        if not hold:
            return False
        self._legal_holds[hold_id] = LegalHold(
            hold_id=hold.hold_id,
            reason=hold.reason,
            imposed_by=hold.imposed_by,
            imposed_at=hold.imposed_at,
            entity_type=hold.entity_type,
            entity_id_pattern=hold.entity_id_pattern,
            lifted_at=datetime.utcnow(),
            lifted_by=lifted_by,
            active=False,
        )
        return True

    def get_deletion_eligible(self) -> list[ArchiveRecord]:
        """Return records that have passed their deletion_eligible_date and are not on legal hold."""
        now = datetime.utcnow()
        return [
            r for r in self._records.values()
            if not r.legal_hold
            and r.deletion_eligible_date is not None
            and r.deletion_eligible_date <= now
            and r.status == ArchiveStatus.ACTIVE
        ]

    def get_records_by_category(self, category: RetentionCategory) -> list[ArchiveRecord]:
        """Return all archive records for a given category."""
        return [r for r in self._records.values() if r.category == category]

    def get_record(self, archive_id: str) -> Optional[ArchiveRecord]:
        """Look up an archive record by ID."""
        return self._records.get(archive_id)

    def get_active_legal_holds(self) -> list[LegalHold]:
        """Return all currently active legal holds."""
        return [h for h in self._legal_holds.values() if h.active]

    def get_policy(self, category: RetentionCategory) -> Optional[RetentionPolicy]:
        """Look up the retention policy for a category."""
        return self._policies.get(category)

    def get_metrics(self) -> dict:
        """Return aggregate metrics for monitoring dashboards."""
        return {
            "total_records": len(self._records),
            "under_legal_hold": sum(1 for r in self._records.values() if r.legal_hold),
            "deletion_eligible": len(self.get_deletion_eligible()),
            "active_legal_holds": len(self.get_active_legal_holds()),
        }


# ── Singleton accessor ─────────────────────────────────────────────────────────

_manager: Optional[RetentionManager] = None


def get_retention_manager() -> RetentionManager:
    """Return the module-level singleton RetentionManager."""
    global _manager
    if _manager is None:
        _manager = RetentionManager()
    return _manager
