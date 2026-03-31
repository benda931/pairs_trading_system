# -*- coding: utf-8 -*-
"""
exceptions_mgmt/engine.py
=========================
Exception management lifecycle: request → review → approve/reject → monitor → expire/renew.

Tracks active waivers and monitors compensating controls. Uses "exceptions_mgmt" as the
package name to avoid collision with Python's built-in `exceptions` namespace.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ── Enums ─────────────────────────────────────────────────────────────────────

class ExceptionCategory(Enum):
    POLICY_DEVIATION = "POLICY_DEVIATION"
    RISK_LIMIT_BREACH = "RISK_LIMIT_BREACH"
    CONTROL_BYPASS = "CONTROL_BYPASS"
    DATA_QUALITY_WAIVER = "DATA_QUALITY_WAIVER"
    MODEL_DEPLOYMENT = "MODEL_DEPLOYMENT"
    OPERATIONAL_NECESSITY = "OPERATIONAL_NECESSITY"


class ExceptionStatus(Enum):
    PENDING = "PENDING"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    WITHDRAWN = "WITHDRAWN"
    PENDING_RENEWAL = "PENDING_RENEWAL"


class CompensatingControlStatus(Enum):
    PROPOSED = "PROPOSED"
    ACTIVE = "ACTIVE"
    VERIFIED = "VERIFIED"
    EXPIRED = "EXPIRED"
    WITHDRAWN = "WITHDRAWN"


# ── Contracts ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CompensatingControl:
    control_id: str
    description: str
    owner: str
    effective_from: datetime
    effective_until: Optional[datetime]
    status: CompensatingControlStatus
    verification_notes: str = ""


@dataclass(frozen=True)
class ExceptionRequest:
    request_id: str
    category: ExceptionCategory
    title: str
    description: str
    requestor: str
    policy_id: Optional[str]
    rule_id: Optional[str]
    entity_type: str
    entity_id: str
    justification: str
    risk_assessment: str
    compensating_controls: tuple[CompensatingControl, ...]
    requested_at: datetime
    expiry_date: Optional[datetime]
    status: ExceptionStatus
    reviewed_by: Optional[str]
    reviewed_at: Optional[datetime]
    review_notes: str = ""


@dataclass(frozen=True)
class ExceptionWaiver:
    waiver_id: str
    request_id: str
    granted_by: str
    granted_at: datetime
    conditions: tuple[str, ...]
    expiry_date: Optional[datetime]
    renewable: bool
    renewal_lead_days: int
    active: bool


# ── Engine ─────────────────────────────────────────────────────────────────────

class ExceptionEngine:
    """
    Exception management lifecycle: request → review → approve/reject → monitor → expire/renew.

    Tracks active waivers and monitors compensating controls.
    """

    def __init__(self) -> None:
        self._requests: dict[str, ExceptionRequest] = {}
        self._waivers: dict[str, ExceptionWaiver] = {}
        self._waiver_by_request: dict[str, str] = {}
        self._lock = threading.Lock()

    def submit_request(
        self,
        category: ExceptionCategory,
        title: str,
        description: str,
        requestor: str,
        entity_type: str,
        entity_id: str,
        justification: str,
        risk_assessment: str,
        policy_id: str = None,
        rule_id: str = None,
        compensating_controls: list[CompensatingControl] = None,
        expiry_date: datetime = None,
    ) -> ExceptionRequest:
        """Submit a new exception request for review."""
        req = ExceptionRequest(
            request_id=str(uuid.uuid4()),
            category=category,
            title=title,
            description=description,
            requestor=requestor,
            policy_id=policy_id,
            rule_id=rule_id,
            entity_type=entity_type,
            entity_id=entity_id,
            justification=justification,
            risk_assessment=risk_assessment,
            compensating_controls=tuple(compensating_controls or []),
            requested_at=datetime.utcnow(),
            expiry_date=expiry_date,
            status=ExceptionStatus.PENDING,
            reviewed_by=None,
            reviewed_at=None,
        )
        with self._lock:
            self._requests[req.request_id] = req
        return req

    def approve(
        self,
        request_id: str,
        reviewed_by: str,
        conditions: list[str] = None,
        renewable: bool = True,
        renewal_lead_days: int = 30,
        review_notes: str = "",
    ) -> ExceptionWaiver:
        """Approve an exception request and issue a waiver."""
        req = self._requests.get(request_id)
        if not req:
            raise KeyError(f"Request {request_id} not found")

        updated_req = ExceptionRequest(
            request_id=req.request_id,
            category=req.category,
            title=req.title,
            description=req.description,
            requestor=req.requestor,
            policy_id=req.policy_id,
            rule_id=req.rule_id,
            entity_type=req.entity_type,
            entity_id=req.entity_id,
            justification=req.justification,
            risk_assessment=req.risk_assessment,
            compensating_controls=req.compensating_controls,
            requested_at=req.requested_at,
            expiry_date=req.expiry_date,
            status=ExceptionStatus.APPROVED,
            reviewed_by=reviewed_by,
            reviewed_at=datetime.utcnow(),
            review_notes=review_notes,
        )

        waiver = ExceptionWaiver(
            waiver_id=str(uuid.uuid4()),
            request_id=request_id,
            granted_by=reviewed_by,
            granted_at=datetime.utcnow(),
            conditions=tuple(conditions or []),
            expiry_date=req.expiry_date,
            renewable=renewable,
            renewal_lead_days=renewal_lead_days,
            active=True,
        )

        with self._lock:
            self._requests[request_id] = updated_req
            self._waivers[waiver.waiver_id] = waiver
            self._waiver_by_request[request_id] = waiver.waiver_id

        return waiver

    def reject(
        self,
        request_id: str,
        reviewed_by: str,
        review_notes: str,
    ) -> ExceptionRequest:
        """Reject an exception request."""
        req = self._requests.get(request_id)
        if not req:
            raise KeyError(f"Request {request_id} not found")
        updated = ExceptionRequest(
            request_id=req.request_id,
            category=req.category,
            title=req.title,
            description=req.description,
            requestor=req.requestor,
            policy_id=req.policy_id,
            rule_id=req.rule_id,
            entity_type=req.entity_type,
            entity_id=req.entity_id,
            justification=req.justification,
            risk_assessment=req.risk_assessment,
            compensating_controls=req.compensating_controls,
            requested_at=req.requested_at,
            expiry_date=req.expiry_date,
            status=ExceptionStatus.REJECTED,
            reviewed_by=reviewed_by,
            reviewed_at=datetime.utcnow(),
            review_notes=review_notes,
        )
        with self._lock:
            self._requests[request_id] = updated
        return updated

    def withdraw_request(self, request_id: str, withdrawn_by: str) -> ExceptionRequest:
        """Withdraw a pending or under-review exception request."""
        req = self._requests.get(request_id)
        if not req:
            raise KeyError(f"Request {request_id} not found")
        updated = ExceptionRequest(
            request_id=req.request_id,
            category=req.category,
            title=req.title,
            description=req.description,
            requestor=req.requestor,
            policy_id=req.policy_id,
            rule_id=req.rule_id,
            entity_type=req.entity_type,
            entity_id=req.entity_id,
            justification=req.justification,
            risk_assessment=req.risk_assessment,
            compensating_controls=req.compensating_controls,
            requested_at=req.requested_at,
            expiry_date=req.expiry_date,
            status=ExceptionStatus.WITHDRAWN,
            reviewed_by=req.reviewed_by,
            reviewed_at=req.reviewed_at,
            review_notes=req.review_notes,
        )
        with self._lock:
            self._requests[request_id] = updated
        return updated

    def expire_waivers(self) -> list[str]:
        """Check and expire waivers past their expiry_date. Returns list of expired waiver_ids."""
        now = datetime.utcnow()
        expired_ids: list[str] = []
        for waiver_id, waiver in list(self._waivers.items()):
            if waiver.active and waiver.expiry_date and waiver.expiry_date < now:
                self._waivers[waiver_id] = ExceptionWaiver(
                    waiver_id=waiver.waiver_id,
                    request_id=waiver.request_id,
                    granted_by=waiver.granted_by,
                    granted_at=waiver.granted_at,
                    conditions=waiver.conditions,
                    expiry_date=waiver.expiry_date,
                    renewable=waiver.renewable,
                    renewal_lead_days=waiver.renewal_lead_days,
                    active=False,
                )
                expired_ids.append(waiver_id)
        return expired_ids

    def has_active_waiver(
        self,
        entity_type: str,
        entity_id: str,
        rule_id: str = None,
    ) -> bool:
        """Check if entity has an active waiver (optionally for a specific rule)."""
        now = datetime.utcnow()
        for req_id, req in self._requests.items():
            if req.entity_type != entity_type or req.entity_id != entity_id:
                continue
            if req.status != ExceptionStatus.APPROVED:
                continue
            if rule_id and req.rule_id != rule_id:
                continue
            waiver_id = self._waiver_by_request.get(req_id)
            if not waiver_id:
                continue
            waiver = self._waivers[waiver_id]
            if not waiver.active:
                continue
            if waiver.expiry_date and waiver.expiry_date < now:
                continue
            return True
        return False

    def get_active_waivers(self) -> list[ExceptionWaiver]:
        """Return all currently active waivers."""
        return [w for w in self._waivers.values() if w.active]

    def get_pending_requests(self) -> list[ExceptionRequest]:
        """Return all requests in PENDING status."""
        return [r for r in self._requests.values() if r.status == ExceptionStatus.PENDING]

    def get_request(self, request_id: str) -> Optional[ExceptionRequest]:
        """Look up a request by ID."""
        return self._requests.get(request_id)

    def get_waiver(self, waiver_id: str) -> Optional[ExceptionWaiver]:
        """Look up a waiver by ID."""
        return self._waivers.get(waiver_id)

    def get_metrics(self) -> dict:
        """Return aggregate metrics for monitoring dashboards."""
        statuses = [r.status for r in self._requests.values()]
        return {
            "total_requests": len(self._requests),
            "pending": sum(1 for s in statuses if s == ExceptionStatus.PENDING),
            "approved": sum(1 for s in statuses if s == ExceptionStatus.APPROVED),
            "rejected": sum(1 for s in statuses if s == ExceptionStatus.REJECTED),
            "active_waivers": len(self.get_active_waivers()),
        }


# ── Singleton accessor ─────────────────────────────────────────────────────────

_engine: Optional[ExceptionEngine] = None


def get_exception_engine() -> ExceptionEngine:
    """Return the module-level singleton ExceptionEngine."""
    global _engine
    if _engine is None:
        _engine = ExceptionEngine()
    return _engine
