# -*- coding: utf-8 -*-
"""
attestations/engine.py
======================
Manages attestation lifecycle: schedule → notify → collect → store → recertify.

Supports one-time and recurring attestations with automatic re-scheduling.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


# ── Enums ─────────────────────────────────────────────────────────────────────

class AttestationScope(Enum):
    CONTROL_EFFECTIVENESS = "CONTROL_EFFECTIVENESS"
    POLICY_COMPLIANCE = "POLICY_COMPLIANCE"
    DATA_QUALITY = "DATA_QUALITY"
    MODEL_PERFORMANCE = "MODEL_PERFORMANCE"
    OPERATIONAL_READINESS = "OPERATIONAL_READINESS"
    SEGREGATION_OF_DUTIES = "SEGREGATION_OF_DUTIES"
    RISK_LIMIT_ADHERENCE = "RISK_LIMIT_ADHERENCE"


class AttestationStatus(Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    OVERDUE = "OVERDUE"
    WAIVED = "WAIVED"
    WITHDRAWN = "WITHDRAWN"


# ── Contracts ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AttestationRequest:
    request_id: str
    scope: AttestationScope
    title: str
    description: str
    attester: str
    entity_type: str
    entity_id: str
    due_date: datetime
    cadence_days: int
    evidence_bundle_id: Optional[str]
    status: AttestationStatus
    requested_at: datetime
    requested_by: str


@dataclass(frozen=True)
class AttestationRecord:
    record_id: str
    request_id: str
    attester: str
    attested_at: datetime
    scope: AttestationScope
    entity_type: str
    entity_id: str
    attestation_text: str
    qualifications: str
    evidence_bundle_id: Optional[str]
    next_due: Optional[datetime]


# ── Engine ─────────────────────────────────────────────────────────────────────

class AttestationEngine:
    """
    Manages attestation lifecycle: schedule → notify → collect → store → recertify.

    Supports one-time and recurring attestations. Completing a recurring attestation
    automatically schedules the next one.
    """

    def __init__(self) -> None:
        self._requests: dict[str, AttestationRequest] = {}
        self._records: list[AttestationRecord] = []
        self._lock = threading.Lock()

    def request_attestation(
        self,
        scope: AttestationScope,
        title: str,
        description: str,
        attester: str,
        entity_type: str,
        entity_id: str,
        due_date: datetime,
        cadence_days: int = 0,
        evidence_bundle_id: str = None,
        requested_by: str = "system",
    ) -> AttestationRequest:
        """Schedule a new attestation request."""
        req = AttestationRequest(
            request_id=str(uuid.uuid4()),
            scope=scope,
            title=title,
            description=description,
            attester=attester,
            entity_type=entity_type,
            entity_id=entity_id,
            due_date=due_date,
            cadence_days=cadence_days,
            evidence_bundle_id=evidence_bundle_id,
            status=AttestationStatus.PENDING,
            requested_at=datetime.utcnow(),
            requested_by=requested_by,
        )
        with self._lock:
            self._requests[req.request_id] = req
        return req

    def complete_attestation(
        self,
        request_id: str,
        attester: str,
        attestation_text: str,
        qualifications: str = "",
        evidence_bundle_id: str = None,
    ) -> AttestationRecord:
        """
        Record a completed attestation.

        For recurring attestations (cadence_days > 0), automatically schedules
        the next attestation request.
        """
        req = self._requests.get(request_id)
        if not req:
            raise KeyError(f"Attestation request {request_id} not found")

        next_due: Optional[datetime] = None
        if req.cadence_days > 0:
            next_due = datetime.utcnow() + timedelta(days=req.cadence_days)

        record = AttestationRecord(
            record_id=str(uuid.uuid4()),
            request_id=request_id,
            attester=attester,
            attested_at=datetime.utcnow(),
            scope=req.scope,
            entity_type=req.entity_type,
            entity_id=req.entity_id,
            attestation_text=attestation_text,
            qualifications=qualifications,
            evidence_bundle_id=evidence_bundle_id or req.evidence_bundle_id,
            next_due=next_due,
        )

        completed_req = AttestationRequest(
            request_id=req.request_id,
            scope=req.scope,
            title=req.title,
            description=req.description,
            attester=req.attester,
            entity_type=req.entity_type,
            entity_id=req.entity_id,
            due_date=req.due_date,
            cadence_days=req.cadence_days,
            evidence_bundle_id=req.evidence_bundle_id,
            status=AttestationStatus.COMPLETED,
            requested_at=req.requested_at,
            requested_by=req.requested_by,
        )

        with self._lock:
            self._requests[request_id] = completed_req
            self._records.append(record)

            # Auto-schedule next recurring attestation
            if req.cadence_days > 0 and next_due is not None:
                next_req = AttestationRequest(
                    request_id=str(uuid.uuid4()),
                    scope=req.scope,
                    title=req.title,
                    description=req.description,
                    attester=req.attester,
                    entity_type=req.entity_type,
                    entity_id=req.entity_id,
                    due_date=next_due,
                    cadence_days=req.cadence_days,
                    evidence_bundle_id=None,
                    status=AttestationStatus.PENDING,
                    requested_at=datetime.utcnow(),
                    requested_by="system",
                )
                self._requests[next_req.request_id] = next_req

        return record

    def waive_attestation(self, request_id: str, waived_by: str) -> bool:
        """Waive a pending attestation request."""
        req = self._requests.get(request_id)
        if not req:
            return False
        self._requests[request_id] = AttestationRequest(
            request_id=req.request_id,
            scope=req.scope,
            title=req.title,
            description=req.description,
            attester=req.attester,
            entity_type=req.entity_type,
            entity_id=req.entity_id,
            due_date=req.due_date,
            cadence_days=req.cadence_days,
            evidence_bundle_id=req.evidence_bundle_id,
            status=AttestationStatus.WAIVED,
            requested_at=req.requested_at,
            requested_by=req.requested_by,
        )
        return True

    def mark_overdue(self) -> list[str]:
        """Mark all pending requests past their due_date as OVERDUE."""
        now = datetime.utcnow()
        overdue_ids: list[str] = []
        for req_id, req in list(self._requests.items()):
            if req.status == AttestationStatus.PENDING and req.due_date < now:
                self._requests[req_id] = AttestationRequest(
                    request_id=req.request_id,
                    scope=req.scope,
                    title=req.title,
                    description=req.description,
                    attester=req.attester,
                    entity_type=req.entity_type,
                    entity_id=req.entity_id,
                    due_date=req.due_date,
                    cadence_days=req.cadence_days,
                    evidence_bundle_id=req.evidence_bundle_id,
                    status=AttestationStatus.OVERDUE,
                    requested_at=req.requested_at,
                    requested_by=req.requested_by,
                )
                overdue_ids.append(req_id)
        return overdue_ids

    def get_pending(self, attester: str = None) -> list[AttestationRequest]:
        """Return all pending or overdue attestation requests, optionally filtered by attester."""
        pending = [
            r for r in self._requests.values()
            if r.status in (AttestationStatus.PENDING, AttestationStatus.OVERDUE)
        ]
        if attester:
            pending = [r for r in pending if r.attester == attester]
        return pending

    def get_records(
        self,
        entity_type: str = None,
        entity_id: str = None,
    ) -> list[AttestationRecord]:
        """Return completed attestation records, optionally filtered."""
        records = list(self._records)
        if entity_type:
            records = [r for r in records if r.entity_type == entity_type]
        if entity_id:
            records = [r for r in records if r.entity_id == entity_id]
        return records

    def get_request(self, request_id: str) -> Optional[AttestationRequest]:
        """Look up an attestation request by ID."""
        return self._requests.get(request_id)

    def get_metrics(self) -> dict:
        """Return aggregate metrics for monitoring dashboards."""
        statuses = [r.status for r in self._requests.values()]
        return {
            "total_requests": len(self._requests),
            "pending": sum(1 for s in statuses if s == AttestationStatus.PENDING),
            "overdue": sum(1 for s in statuses if s == AttestationStatus.OVERDUE),
            "completed": sum(1 for s in statuses if s == AttestationStatus.COMPLETED),
            "total_records": len(self._records),
        }


# ── Singleton accessor ─────────────────────────────────────────────────────────

_engine: Optional[AttestationEngine] = None


def get_attestation_engine() -> AttestationEngine:
    """Return the module-level singleton AttestationEngine."""
    global _engine
    if _engine is None:
        _engine = AttestationEngine()
    return _engine
