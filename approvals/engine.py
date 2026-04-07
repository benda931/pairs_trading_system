# -*- coding: utf-8 -*-
"""
approvals/engine.py — ApprovalEngine
======================================

Thread-safe, in-memory approval engine responsible for:
- Adjudicating ApprovalRequests based on mode and policy.
- Maintaining records of decisions, tickets, escalations, and overrides.
- Exposing a simple metrics API for monitoring.
- Per-action-type policy function registration (GovernanceRouter integration).
- SLA breach detection for HUMAN_REQUIRED tickets.
- Cancellation of pending requests (conflict resolution support).
- Environment restriction enforcement (LIVE actions use live-only policy fns).

The engine is intentionally stateless with respect to persistence — all
records live in memory. Production deployments should wrap this class with
a persistence adapter (DB-backed store).

Singleton access via `get_approval_engine()`.

Thread safety: all mutable stores are protected by a single `threading.Lock`.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

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


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════


def _now_iso() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════
# APPROVAL ENGINE
# ══════════════════════════════════════════════════════════════════


class ApprovalEngine:
    """Central approval engine for agent action gating.

    Responsibilities
    ----------------
    - Accept ApprovalRequests and route them based on ApprovalMode.
    - Auto-approve AUTOMATIC requests and low-risk RESEARCH environment actions.
    - Return PENDING decisions for HUMAN_REQUIRED requests and create
      HumanReviewTickets for the review queue.
    - Delegate POLICY_GATED requests to an optional policy_engine.
    - Permanently block BLOCKED requests.
    - Provide decide(), get_pending_requests(), and resolve_ticket() APIs.
    - Expose an escalation store via escalate().

    Thread Safety
    -------------
    All store mutations are serialized through a single threading.Lock.
    Callers may safely call any method from multiple threads.

    Parameters
    ----------
    policy_engine : optional
        Any object implementing ``check_policy(...) -> PolicyCheckResult``.
        If None, POLICY_GATED requests fall back to AUTO_APPROVED.
    default_mode : ApprovalMode
        Default mode used when the request does not specify one explicitly.
        Defaults to POLICY_GATED.
    """

    def __init__(
        self,
        policy_engine: Optional[Any] = None,
        default_mode: ApprovalMode = ApprovalMode.POLICY_GATED,
    ) -> None:
        self._policy_engine = policy_engine
        self._default_mode = default_mode

        # In-memory stores
        self._requests: Dict[str, ApprovalRequest] = {}
        self._decisions: Dict[str, ApprovalDecision] = {}
        self._tickets: Dict[str, HumanReviewTicket] = {}
        self._escalations: Dict[str, EscalationRecord] = {}
        self._overrides: Dict[str, OverrideRecord] = {}

        # Per-action-type policy function registry.
        # Maps action_type → Callable[[ApprovalRequest], bool]
        # Each function returns True if the action should be auto-approved.
        self._policy_fns: Dict[str, Callable[[ApprovalRequest], bool]] = {}

        # Reverse index: action_id → request_id for GovernanceRouter lookups.
        self._action_id_to_request_id: Dict[str, str] = {}

        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────
    # REQUEST APPROVAL
    # ──────────────────────────────────────────────────────────────

    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Submit an ApprovalRequest and return an initial decision.

        Routing logic (evaluated in order):
        1. BLOCKED mode → reject immediately, never approve.
        2. AUTOMATIC mode → auto-approve immediately.
        3. RESEARCH environment + LOW risk → auto-approve immediately.
        4. HUMAN_REQUIRED / DUAL_APPROVAL → create HumanReviewTicket,
           return PENDING decision.
        5. POLICY_GATED + policy_engine available → delegate to policy_engine.
        6. POLICY_GATED + no policy_engine → auto-approve (open environment
           without a policy engine is treated as permissive).

        Parameters
        ----------
        request : ApprovalRequest
            The request to adjudicate.

        Returns
        -------
        ApprovalDecision
            The initial decision. Status may be PENDING if human review
            is required; callers must poll or listen for resolution.
        """
        with self._lock:
            self._requests[request.request_id] = request

            mode = request.approval_mode

            # 1. Categorically blocked actions
            if mode == ApprovalMode.BLOCKED:
                decision = ApprovalDecision(
                    decision_id=_new_id(),
                    request_id=request.request_id,
                    decided_at=_now_iso(),
                    decided_by="ApprovalEngine",
                    status=ApprovalStatus.REJECTED,
                    rationale=(
                        "Action type '{}' is categorically blocked by policy. "
                        "No override path exists.".format(request.action_type)
                    ),
                    conditions=(),
                    evidence_reviewed=(),
                    override_used=False,
                )
                self._decisions[request.request_id] = decision
                return decision

            # 2. Fully automatic — no gate required
            if mode == ApprovalMode.AUTOMATIC:
                decision = ApprovalDecision(
                    decision_id=_new_id(),
                    request_id=request.request_id,
                    decided_at=_now_iso(),
                    decided_by="ApprovalEngine",
                    status=ApprovalStatus.AUTO_APPROVED,
                    rationale="Approval mode is AUTOMATIC; action auto-approved.",
                    conditions=(),
                    evidence_reviewed=request.evidence_bundle_ids,
                    override_used=False,
                )
                self._decisions[request.request_id] = decision
                return decision

            # 3. Low-risk research environment — auto-approve
            if (
                request.environment.lower() in ("research", "dev", "local")
                and request.risk_class.upper() == "LOW"
            ):
                decision = ApprovalDecision(
                    decision_id=_new_id(),
                    request_id=request.request_id,
                    decided_at=_now_iso(),
                    decided_by="ApprovalEngine",
                    status=ApprovalStatus.AUTO_APPROVED,
                    rationale=(
                        "Auto-approved: LOW risk action in '{}' environment.".format(
                            request.environment
                        )
                    ),
                    conditions=(),
                    evidence_reviewed=request.evidence_bundle_ids,
                    override_used=False,
                )
                self._decisions[request.request_id] = decision
                return decision

            # 4. Human review required
            if mode in (ApprovalMode.HUMAN_REQUIRED, ApprovalMode.DUAL_APPROVAL):
                priority = self._derive_priority(request)
                ticket = HumanReviewTicket(
                    ticket_id=_new_id(),
                    created_at=_now_iso(),
                    priority=priority,
                    title="Review required: {} by {}".format(
                        request.action_type, request.agent_name
                    ),
                    description=(
                        "{}\n\nContext: {}\nRisk class: {}\nEnvironment: {}\n"
                        "Required approvers: {}".format(
                            request.action_description,
                            request.context_summary,
                            request.risk_class,
                            request.environment,
                            ", ".join(request.required_approvers) or "any",
                        )
                    ),
                    agent_name=request.agent_name,
                    workflow_run_id=request.workflow_run_id,
                    approval_request_id=request.request_id,
                    evidence_bundle_ids=request.evidence_bundle_ids,
                    due_by=request.expires_at,
                    assigned_to=(
                        request.required_approvers[0]
                        if request.required_approvers
                        else None
                    ),
                    status="open",
                )
                self._tickets[ticket.ticket_id] = ticket

                decision = self._create_pending_decision(request)
                self._decisions[request.request_id] = decision
                return decision

            # 4b. Track action_id → request_id for lookup
            if request.task_id:
                self._action_id_to_request_id[request.task_id] = request.request_id

            # 5a. Per-action-type policy function (registered by GovernanceRouter)
            specific_fn_result = self.check_policy_fn(request)
            if mode == ApprovalMode.POLICY_GATED and specific_fn_result is not None:
                if specific_fn_result:
                    decision = ApprovalDecision(
                        decision_id=_new_id(),
                        request_id=request.request_id,
                        decided_at=_now_iso(),
                        decided_by="PolicyFn[{}]".format(request.action_type),
                        status=ApprovalStatus.AUTO_APPROVED,
                        rationale="Per-action-type policy function approved '{}'.".format(
                            request.action_type),
                        conditions=(),
                        evidence_reviewed=request.evidence_bundle_ids,
                        override_used=False,
                    )
                else:
                    # Specific policy rejected → escalate to human review
                    priority = self._derive_priority(request)
                    ticket = HumanReviewTicket(
                        ticket_id=_new_id(),
                        created_at=_now_iso(),
                        priority=priority,
                        title="Policy fn rejected — human review: {}".format(
                            request.action_type),
                        description=(
                            "The per-action-type policy function for '{}' returned False.\n"
                            "Context: {}\nRisk: {}\nEnvironment: {}".format(
                                request.action_type,
                                request.context_summary,
                                request.risk_class,
                                request.environment,
                            )
                        ),
                        agent_name=request.agent_name,
                        workflow_run_id=request.workflow_run_id,
                        approval_request_id=request.request_id,
                        evidence_bundle_ids=request.evidence_bundle_ids,
                        due_by=request.expires_at,
                        assigned_to=None,
                        status="open",
                    )
                    self._tickets[ticket.ticket_id] = ticket
                    decision = self._create_pending_decision(request)
                self._decisions[request.request_id] = decision
                return decision

            # 5b. Policy-gated with an engine
            if mode == ApprovalMode.POLICY_GATED and self._policy_engine is not None:
                try:
                    result = self._policy_engine.check_policy(
                        agent_name=request.agent_name,
                        task_type=request.action_type,
                        action_type=request.action_type,
                        environment=request.environment,
                        risk_class=request.risk_class,
                        task_id=request.task_id,
                    )
                    if result.passed:
                        decision = ApprovalDecision(
                            decision_id=_new_id(),
                            request_id=request.request_id,
                            decided_at=_now_iso(),
                            decided_by="PolicyEngine",
                            status=ApprovalStatus.AUTO_APPROVED,
                            rationale="Policy '{}' passed.".format(result.policy_name),
                            conditions=(),
                            evidence_reviewed=request.evidence_bundle_ids,
                            override_used=False,
                        )
                    else:
                        # Policy failed — escalate to human review
                        priority = self._derive_priority(request)
                        ticket = HumanReviewTicket(
                            ticket_id=_new_id(),
                            created_at=_now_iso(),
                            priority=priority,
                            title="Policy gate failed — human review required: {}".format(
                                request.action_type
                            ),
                            description=(
                                "Policy '{}' failed with severity {}.\n{}\n\n"
                                "Remediation: {}\n\nOriginal request context: {}".format(
                                    result.policy_name,
                                    result.severity,
                                    result.message,
                                    result.remediation_hint,
                                    request.context_summary,
                                )
                            ),
                            agent_name=request.agent_name,
                            workflow_run_id=request.workflow_run_id,
                            approval_request_id=request.request_id,
                            evidence_bundle_ids=request.evidence_bundle_ids,
                            due_by=request.expires_at,
                            assigned_to=None,
                            status="open",
                        )
                        self._tickets[ticket.ticket_id] = ticket
                        decision = self._create_pending_decision(request)
                except Exception as exc:  # noqa: BLE001
                    # Policy engine failure — fail-safe to pending
                    decision = self._create_pending_decision(request)
                    decision = ApprovalDecision(
                        decision_id=decision.decision_id,
                        request_id=decision.request_id,
                        decided_at=decision.decided_at,
                        decided_by=decision.decided_by,
                        status=ApprovalStatus.PENDING,
                        rationale=(
                            "Policy engine raised an exception: {}. "
                            "Defaulting to PENDING for human review.".format(exc)
                        ),
                        conditions=(),
                        evidence_reviewed=(),
                        override_used=False,
                    )
                self._decisions[request.request_id] = decision
                return decision

            # 6. Policy-gated but no engine — open environment, auto-approve
            decision = ApprovalDecision(
                decision_id=_new_id(),
                request_id=request.request_id,
                decided_at=_now_iso(),
                decided_by="ApprovalEngine",
                status=ApprovalStatus.AUTO_APPROVED,
                rationale=(
                    "POLICY_GATED mode but no policy engine configured; "
                    "auto-approved in permissive configuration."
                ),
                conditions=(),
                evidence_reviewed=request.evidence_bundle_ids,
                override_used=False,
            )
            self._decisions[request.request_id] = decision
            return decision

    # ──────────────────────────────────────────────────────────────
    # DECIDE (MANUAL ADJUDICATION)
    # ──────────────────────────────────────────────────────────────

    def decide(
        self,
        request_id: str,
        status: ApprovalStatus,
        decided_by: str,
        rationale: str,
        conditions: Optional[List[str]] = None,
    ) -> ApprovalDecision:
        """Manually adjudicate a pending request.

        Intended for human reviewers or automated policy callbacks to resolve
        requests that are currently in PENDING or ESCALATED state.

        Parameters
        ----------
        request_id : str
            ID of the ApprovalRequest to adjudicate.
        status : ApprovalStatus
            The decision outcome.
        decided_by : str
            Identity of the reviewer.
        rationale : str
            Mandatory explanation for the decision.
        conditions : list[str], optional
            Conditions attached to an APPROVED decision.

        Returns
        -------
        ApprovalDecision

        Raises
        ------
        KeyError
            If request_id does not exist.
        ValueError
            If the request has already been finalized.
        """
        with self._lock:
            if request_id not in self._requests:
                raise KeyError("ApprovalRequest '{}' not found.".format(request_id))

            existing = self._decisions.get(request_id)
            if existing and existing.status not in (
                ApprovalStatus.PENDING,
                ApprovalStatus.ESCALATED,
            ):
                raise ValueError(
                    "Request '{}' already finalized with status '{}'.".format(
                        request_id, existing.status
                    )
                )

            decision = ApprovalDecision(
                decision_id=_new_id(),
                request_id=request_id,
                decided_at=_now_iso(),
                decided_by=decided_by,
                status=status,
                rationale=rationale,
                conditions=tuple(conditions or []),
                evidence_reviewed=self._requests[request_id].evidence_bundle_ids,
                override_used=False,
            )
            self._decisions[request_id] = decision
            return decision

    # ──────────────────────────────────────────────────────────────
    # QUERY METHODS
    # ──────────────────────────────────────────────────────────────

    def expire_stale_requests(self, max_age_hours: float = 48.0) -> int:
        """
        Expire all pending requests older than max_age_hours.

        Requests that sit unreviewed beyond their TTL should auto-expire
        rather than accumulating forever. Expired requests are set to
        REJECTED status with a "timed_out" rationale.

        Called automatically from get_pending_requests() to keep the
        inbox clean.

        Parameters
        ----------
        max_age_hours : float
            Maximum age of a pending request before auto-expiry (default 48h).

        Returns
        -------
        int : Number of requests expired.
        """
        from datetime import datetime, timezone, timedelta

        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=max_age_hours)
        expired_count = 0

        with self._lock:
            for req_id, req in list(self._requests.items()):
                decision = self._decisions.get(req_id)
                if decision is None:
                    continue

                # Only expire PENDING and ESCALATED requests
                try:
                    from approvals.contracts import ApprovalStatus
                    if decision.status not in (
                        ApprovalStatus.PENDING,
                        ApprovalStatus.ESCALATED,
                    ):
                        continue
                except Exception:
                    continue

                # Check age
                try:
                    requested_at_str = getattr(req, "requested_at", None)
                    if not requested_at_str:
                        continue
                    requested_dt = datetime.fromisoformat(requested_at_str)
                    if requested_dt.tzinfo is None:
                        requested_dt = requested_dt.replace(tzinfo=timezone.utc)
                    if requested_dt < cutoff:
                        # Expire by setting to REJECTED with timeout rationale
                        # Use dataclasses.replace to create updated frozen object
                        try:
                            from dataclasses import replace as _replace
                            self._decisions[req_id] = _replace(
                                decision,
                                status=ApprovalStatus.REJECTED,
                                rationale="auto_expired: request exceeded TTL without review",
                                decided_by="system_expiry",
                                decided_at=datetime.now(tz=timezone.utc).isoformat(),
                            )
                            expired_count += 1
                            logger.info(
                                "Auto-expired approval request %s (age: %.1fh > %.1fh TTL)",
                                req_id,
                                (datetime.now(tz=timezone.utc) - requested_dt).total_seconds() / 3600,
                                max_age_hours,
                            )
                        except Exception as exc:
                            logger.debug("Could not expire request %s: %s", req_id, exc)
                except Exception as exc:
                    logger.debug("expire_stale_requests: date parse failed for %s: %s", req_id, exc)

        return expired_count

    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Return all requests whose current decision is PENDING or ESCALATED.

        Returns
        -------
        list[ApprovalRequest]
            Unresolved requests, ordered by request submission time.
        """
        # Auto-expire stale requests before returning the inbox
        self.expire_stale_requests()

        with self._lock:
            pending = []
            for request_id, request in self._requests.items():
                decision = self._decisions.get(request_id)
                if decision is None or decision.status in (
                    ApprovalStatus.PENDING,
                    ApprovalStatus.ESCALATED,
                ):
                    pending.append(request)
            return sorted(pending, key=lambda r: r.requested_at)

    def get_decision(self, request_id: str) -> Optional[ApprovalDecision]:
        """Return the current decision for a request, or None if not yet decided.

        Parameters
        ----------
        request_id : str
            ID of the ApprovalRequest.

        Returns
        -------
        Optional[ApprovalDecision]
        """
        with self._lock:
            return self._decisions.get(request_id)

    def get_open_tickets(self) -> List[HumanReviewTicket]:
        """Return all human review tickets that are open or in_review.

        Returns
        -------
        list[HumanReviewTicket]
            Open tickets, sorted by priority then creation time.
        """
        _priority_order = {
            ReviewPriority.CRITICAL: 0,
            ReviewPriority.URGENT: 1,
            ReviewPriority.ELEVATED: 2,
            ReviewPriority.ROUTINE: 3,
        }
        with self._lock:
            open_tickets = [
                t
                for t in self._tickets.values()
                if t.status in ("open", "in_review")
            ]
            return sorted(
                open_tickets,
                key=lambda t: (_priority_order.get(t.priority, 99), t.created_at),
            )

    # ──────────────────────────────────────────────────────────────
    # TICKET RESOLUTION
    # ──────────────────────────────────────────────────────────────

    def resolve_ticket(
        self,
        ticket_id: str,
        resolution: str,
        resolved_by: str,
    ) -> HumanReviewTicket:
        """Mark a ticket as resolved and close it.

        Parameters
        ----------
        ticket_id : str
            ID of the HumanReviewTicket to resolve.
        resolution : str
            Resolution notes.
        resolved_by : str
            Identity of the reviewer closing the ticket.

        Returns
        -------
        HumanReviewTicket
            The updated ticket (reconstructed frozen dataclass).

        Raises
        ------
        KeyError
            If ticket_id does not exist.
        """
        with self._lock:
            if ticket_id not in self._tickets:
                raise KeyError("Ticket '{}' not found.".format(ticket_id))

            old = self._tickets[ticket_id]
            resolved_at = _now_iso()

            # Reconstruct frozen dataclass with updated fields
            updated = HumanReviewTicket(
                ticket_id=old.ticket_id,
                created_at=old.created_at,
                priority=old.priority,
                title=old.title,
                description=old.description,
                agent_name=old.agent_name,
                workflow_run_id=old.workflow_run_id,
                approval_request_id=old.approval_request_id,
                evidence_bundle_ids=old.evidence_bundle_ids,
                due_by=old.due_by,
                assigned_to=resolved_by,
                status="resolved",
                resolution=resolution,
                resolved_at=resolved_at,
            )
            self._tickets[ticket_id] = updated
            return updated

    # ──────────────────────────────────────────────────────────────
    # ESCALATION
    # ──────────────────────────────────────────────────────────────

    def escalate(self, escalation: EscalationRecord) -> None:
        """Record an escalation event.

        If the escalation references an approval_request_id, the corresponding
        decision is updated to ESCALATED status.

        Parameters
        ----------
        escalation : EscalationRecord
            The escalation record to store.
        """
        with self._lock:
            self._escalations[escalation.escalation_id] = escalation

            # Update linked decision to ESCALATED
            if escalation.approval_request_id:
                existing = self._decisions.get(escalation.approval_request_id)
                if existing and existing.status == ApprovalStatus.PENDING:
                    escalated_decision = ApprovalDecision(
                        decision_id=existing.decision_id,
                        request_id=existing.request_id,
                        decided_at=existing.decided_at,
                        decided_by=existing.decided_by,
                        status=ApprovalStatus.ESCALATED,
                        rationale=(
                            "Escalated to '{}': {}".format(
                                escalation.escalated_to, escalation.reason
                            )
                        ),
                        conditions=existing.conditions,
                        evidence_reviewed=existing.evidence_reviewed,
                        override_used=existing.override_used,
                        notes=existing.notes,
                    )
                    self._decisions[escalation.approval_request_id] = escalated_decision

    # ──────────────────────────────────────────────────────────────
    # METRICS
    # ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for the approval engine.

        Returns
        -------
        dict
            Keys: counts_by_status, total_requests, total_decisions,
            open_ticket_count, total_tickets, escalation_count,
            override_count, avg_decision_time_seconds (from decided
            requests where requested_at and decided_at are both present).
        """
        with self._lock:
            counts: Dict[str, int] = {s.value: 0 for s in ApprovalStatus}
            for decision in self._decisions.values():
                counts[decision.status.value] += 1

            # Average decision time
            decision_times: List[float] = []
            for request_id, decision in self._decisions.items():
                if decision.status in (
                    ApprovalStatus.PENDING,
                    ApprovalStatus.ESCALATED,
                ):
                    continue
                request = self._requests.get(request_id)
                if request:
                    try:
                        req_dt = datetime.fromisoformat(request.requested_at)
                        dec_dt = datetime.fromisoformat(decision.decided_at)
                        delta = (dec_dt - req_dt).total_seconds()
                        if delta >= 0:
                            decision_times.append(delta)
                    except (ValueError, TypeError):
                        pass

            avg_decision_time = (
                sum(decision_times) / len(decision_times) if decision_times else 0.0
            )

            open_tickets = sum(
                1
                for t in self._tickets.values()
                if t.status in ("open", "in_review")
            )

            return {
                "counts_by_status": counts,
                "total_requests": len(self._requests),
                "total_decisions": len(self._decisions),
                "open_ticket_count": open_tickets,
                "total_tickets": len(self._tickets),
                "escalation_count": len(self._escalations),
                "override_count": len(self._overrides),
                "avg_decision_time_seconds": round(avg_decision_time, 2),
            }

    # ──────────────────────────────────────────────────────────────
    # POLICY FUNCTION REGISTRY
    # ──────────────────────────────────────────────────────────────

    def register_policy_function(
        self,
        action_type: str,
        policy_fn: Callable[[ApprovalRequest], bool],
    ) -> None:
        """Register a per-action-type policy function for POLICY_GATED adjudication.

        The function receives the full `ApprovalRequest` and returns True if the
        action should be auto-approved, False if it should escalate to human review.

        Policy functions are evaluated during `request_approval()` before falling
        back to the generic `_policy_engine`. This allows GovernanceRouter to
        register action-specific evidence-based policy checks.

        Parameters
        ----------
        action_type : str
            E.g. "BLOCK_ENTRY", "DELEVERAGE".
        policy_fn : Callable[[ApprovalRequest], bool]
            Should not raise; return False on any uncertainty.
        """
        with self._lock:
            self._policy_fns[action_type] = policy_fn

    def check_policy_fn(self, request: ApprovalRequest) -> Optional[bool]:
        """Evaluate the registered per-action-type policy function if one exists.

        Returns True (auto-approve), False (escalate), or None (no policy function).
        """
        with self._lock:
            fn = self._policy_fns.get(request.action_type)
        if fn is None:
            return None
        try:
            return fn(request)
        except Exception:
            return False  # Fail-closed: policy exception → escalate to human

    # ──────────────────────────────────────────────────────────────
    # CANCELLATION
    # ──────────────────────────────────────────────────────────────

    def cancel_approval(self, request_id: str, reason: str) -> bool:
        """Cancel a pending approval request.

        Used by GovernanceRouter's ConflictResolver to cancel the losing action
        when a higher-priority action wins a conflict.

        Parameters
        ----------
        request_id : str
            ID of the ApprovalRequest to cancel.
        reason : str
            Cancellation reason (recorded in the decision rationale).

        Returns
        -------
        bool
            True if the request existed and was in a cancellable state (PENDING
            or ESCALATED). False otherwise.
        """
        with self._lock:
            if request_id not in self._requests:
                return False
            existing = self._decisions.get(request_id)
            if existing and existing.status not in (
                ApprovalStatus.PENDING, ApprovalStatus.ESCALATED
            ):
                return False

            cancelled_decision = ApprovalDecision(
                decision_id=_new_id(),
                request_id=request_id,
                decided_at=_now_iso(),
                decided_by="ApprovalEngine.cancel_approval",
                status=ApprovalStatus.REJECTED,
                rationale="Cancelled by conflict resolution: {}".format(reason),
                conditions=(),
                evidence_reviewed=(),
                override_used=False,
            )
            self._decisions[request_id] = cancelled_decision

            # Close the linked ticket if any
            for ticket_id, ticket in self._tickets.items():
                if ticket.approval_request_id == request_id and ticket.status == "open":
                    updated = HumanReviewTicket(
                        ticket_id=ticket.ticket_id,
                        created_at=ticket.created_at,
                        priority=ticket.priority,
                        title=ticket.title,
                        description=ticket.description,
                        agent_name=ticket.agent_name,
                        workflow_run_id=ticket.workflow_run_id,
                        approval_request_id=ticket.approval_request_id,
                        evidence_bundle_ids=ticket.evidence_bundle_ids,
                        due_by=ticket.due_by,
                        assigned_to=ticket.assigned_to,
                        status="closed",
                        resolution="Cancelled: {}".format(reason),
                        resolved_at=_now_iso(),
                    )
                    self._tickets[ticket_id] = updated
                    break

        return True

    # ──────────────────────────────────────────────────────────────
    # LOOKUP BY ACTION ID
    # ──────────────────────────────────────────────────────────────

    def get_approval_by_action_id(self, action_id: str) -> Optional[ApprovalRequest]:
        """Return the ApprovalRequest for a given action_id, or None.

        GovernanceRouter uses this to check if an identical action is already
        pending approval before submitting a duplicate request.

        The lookup relies on `task_id` in ApprovalRequest being set to the
        action_id by GovernanceRouter when submitting requests.
        """
        with self._lock:
            request_id = self._action_id_to_request_id.get(action_id)
            if request_id:
                return self._requests.get(request_id)
            # Linear fallback scan (O(n) but request volume is low)
            for req in self._requests.values():
                if req.task_id == action_id:
                    self._action_id_to_request_id[action_id] = req.request_id
                    return req
        return None

    # ──────────────────────────────────────────────────────────────
    # SLA BREACH DETECTION
    # ──────────────────────────────────────────────────────────────

    def get_sla_breached_tickets(
        self,
        as_of: Optional[datetime] = None,
    ) -> List[HumanReviewTicket]:
        """Return all open tickets whose `due_by` deadline has passed.

        GovernanceRouter calls this periodically to detect SLA breaches and
        open P2 incidents for unreviewed human-required actions.

        Parameters
        ----------
        as_of : datetime, optional
            The reference time for deadline comparison. Defaults to now.

        Returns
        -------
        list[HumanReviewTicket]
            Open tickets past their deadline, sorted by due_by ascending
            (most overdue first).
        """
        if as_of is None:
            as_of = datetime.now(timezone.utc)

        with self._lock:
            tickets = list(self._tickets.values())

        breached = []
        for ticket in tickets:
            if ticket.status not in ("open", "in_review"):
                continue
            if ticket.due_by is None:
                continue
            try:
                due = datetime.fromisoformat(ticket.due_by)
                if due.tzinfo is None:
                    due = due.replace(tzinfo=timezone.utc)
                if due < as_of:
                    breached.append(ticket)
            except (ValueError, TypeError):
                continue

        return sorted(
            breached,
            key=lambda t: t.due_by or "",
        )

    def check_environment_restriction(
        self,
        request: ApprovalRequest,
        allowed_environments: Optional[List[str]] = None,
    ) -> bool:
        """Check whether an ApprovalRequest is allowed in its stated environment.

        Used by GovernanceRouter to block LIVE actions from being evaluated by
        research-only policy functions. Returns True if the request is allowed,
        False if it should be rejected due to environment mismatch.

        Parameters
        ----------
        request : ApprovalRequest
        allowed_environments : list[str], optional
            Environments in which this request type is permitted. If None,
            no restriction is applied and True is returned.

        Returns
        -------
        bool
        """
        if allowed_environments is None:
            return True
        return request.environment.lower() in [e.lower() for e in allowed_environments]

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────

    def _create_pending_decision(self, request: ApprovalRequest) -> ApprovalDecision:
        """Create a PENDING decision for a request awaiting human review.

        Parameters
        ----------
        request : ApprovalRequest

        Returns
        -------
        ApprovalDecision with status=PENDING.
        """
        return ApprovalDecision(
            decision_id=_new_id(),
            request_id=request.request_id,
            decided_at=_now_iso(),
            decided_by="ApprovalEngine",
            status=ApprovalStatus.PENDING,
            rationale="Awaiting human review. A review ticket has been created.",
            conditions=(),
            evidence_reviewed=(),
            override_used=False,
        )

    def _derive_priority(self, request: ApprovalRequest) -> ReviewPriority:
        """Derive ticket priority from risk class and environment.

        Parameters
        ----------
        request : ApprovalRequest

        Returns
        -------
        ReviewPriority
        """
        risk = request.risk_class.upper()
        env = request.environment.lower()

        if risk in ("CRITICAL", "SENSITIVE") or env == "production":
            if risk == "CRITICAL":
                return ReviewPriority.CRITICAL
            return ReviewPriority.URGENT
        if risk == "HIGH":
            return ReviewPriority.ELEVATED
        return ReviewPriority.ROUTINE


# ══════════════════════════════════════════════════════════════════
# SINGLETON FACTORY
# ══════════════════════════════════════════════════════════════════

_engine_instance: Optional[ApprovalEngine] = None
_engine_lock = threading.Lock()


def get_approval_engine(
    policy_engine: Optional[Any] = None,
    default_mode: ApprovalMode = ApprovalMode.POLICY_GATED,
) -> ApprovalEngine:
    """Return the process-level singleton ApprovalEngine.

    On first call, instantiates the engine with the provided parameters.
    Subsequent calls return the same instance regardless of arguments.

    Parameters
    ----------
    policy_engine : optional
        Policy engine passed only on the first call.
    default_mode : ApprovalMode
        Default mode passed only on the first call.

    Returns
    -------
    ApprovalEngine
    """
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = ApprovalEngine(
                    policy_engine=policy_engine,
                    default_mode=default_mode,
                )
    return _engine_instance
