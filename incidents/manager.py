# -*- coding: utf-8 -*-
"""
incidents/manager.py — IncidentManager
========================================

Thread-safe, in-memory incident lifecycle manager covering:
- Incident creation and status management with auto-linked runbooks.
- Timeline event appending for chronological audit trails.
- Remediation plan attachment.
- Postmortem artifact creation.
- Append-only audit record ledger (capped at 50,000 records).
- Repeat pattern detection for systemic issue identification.
- Metrics for monitoring dashboards.

Five default runbooks are pre-registered at startup.

Singleton access via ``get_incident_manager()``.

Thread safety: a single ``threading.Lock`` serializes all store mutations.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from incidents.contracts import (
    AuditRecord,
    AuditRecordType,
    IncidentRecord,
    IncidentSeverity,
    IncidentStatus,
    IncidentTriggerSource,
    PostmortemArtifact,
    RemediationPlan,
    RunbookReference,
)


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════


def _now_iso() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _today_iso() -> str:
    """Return current UTC date as ISO 8601 date string."""
    return datetime.now(timezone.utc).date().isoformat()


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


_AUDIT_RECORD_CAP = 250_000  # Increased from 50K for governed action volume


# ══════════════════════════════════════════════════════════════════
# INCIDENT MANAGER
# ══════════════════════════════════════════════════════════════════


class IncidentManager:
    """Central incident lifecycle and audit ledger manager.

    Responsibilities
    ----------------
    - Create, update, and close ``IncidentRecord`` objects.
    - Maintain a chronological timeline per incident.
    - Attach ``RemediationPlan`` objects to incidents.
    - Generate ``PostmortemArtifact`` objects at incident closure.
    - Maintain an append-only ``AuditRecord`` ledger (capped at 50,000).
    - Auto-link relevant runbooks when an incident is created.
    - Detect repeat incident patterns by affected component.
    - Expose ``get_metrics()`` for dashboards.

    Thread Safety
    -------------
    All store mutations are protected by a single ``threading.Lock``.

    Default runbooks pre-registered:
    - ``drift_alert``             — feature drift / data distribution shift
    - ``kill_switch_triggered``   — kill-switch activation response
    - ``model_degraded``          — model performance degradation
    - ``data_integrity_failure``  — data pipeline integrity failure
    - ``agent_stuck``             — agent task stuck / timeout
    """

    def __init__(self) -> None:
        self._incidents: Dict[str, IncidentRecord] = {}
        self._remediations: Dict[str, RemediationPlan] = {}
        self._postmortems: Dict[str, PostmortemArtifact] = {}
        self._audit_records: List[AuditRecord] = []
        self._runbooks: Dict[str, RunbookReference] = {}
        self._lock = threading.Lock()

        self._register_default_runbooks()

    # ──────────────────────────────────────────────────────────────
    # INCIDENT CREATION
    # ──────────────────────────────────────────────────────────────

    def create_incident(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        detected_by: str,
        affected_components: List[str],
        evidence_bundle_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> IncidentRecord:
        """Create a new incident and auto-link applicable runbooks.

        Runbooks are matched by ``applicable_components`` intersection with
        ``affected_components`` and by ``applicable_severity`` matching the
        incident severity.

        Parameters
        ----------
        title : str
            Short incident title.
        description : str
            Full incident description.
        severity : IncidentSeverity
            Severity tier.
        detected_by : str
            Identity of the detector (agent name, alert ID, user).
        affected_components : list[str]
            System components affected.
        evidence_bundle_ids : list[str], optional
            Evidence bundles to attach.
        tags : list[str], optional
            Free-form tags.

        Returns
        -------
        IncidentRecord
        """
        incident_id = _new_id()
        now = _now_iso()

        # Auto-link runbooks
        linked_runbooks = self._match_runbooks(affected_components, severity)

        incident = IncidentRecord(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.OPEN,
            detected_at=now,
            detected_by=detected_by,
            affected_components=list(affected_components),
            affected_agents=[],
            affected_workflows=[],
            evidence_bundle_ids=list(evidence_bundle_ids or []),
            related_alert_ids=[],
            runbook_refs=linked_runbooks,
            timeline=[
                {
                    "ts": now,
                    "actor": detected_by,
                    "action": "INCIDENT_CREATED",
                    "notes": "Incident created with severity {}.".format(
                        severity.value
                    ),
                }
            ],
            tags=list(tags or []),
        )

        with self._lock:
            self._incidents[incident_id] = incident

        # Auto-log audit record
        self.log_audit(
            AuditRecord(
                record_id=_new_id(),
                record_type=AuditRecordType.INCIDENT_CREATED,
                actor=detected_by,
                action="INCIDENT_CREATED",
                subject=incident_id,
                outcome="SUCCESS",
                timestamp=now,
                workflow_run_id=None,
                task_id=None,
                evidence_ids=tuple(evidence_bundle_ids or []),
                policy_check_ids=(),
                details="Incident '{}' created with severity {}.".format(
                    title, severity.value
                ),
            )
        )

        return incident

    # ──────────────────────────────────────────────────────────────
    # STATUS MANAGEMENT
    # ──────────────────────────────────────────────────────────────

    def update_status(
        self,
        incident_id: str,
        status: IncidentStatus,
        actor: str,
        notes: str = "",
    ) -> IncidentRecord:
        """Update the status of an incident and append a timeline event.

        Parameters
        ----------
        incident_id : str
        status : IncidentStatus
        actor : str
        notes : str, optional

        Returns
        -------
        IncidentRecord

        Raises
        ------
        KeyError
            If incident_id does not exist.
        """
        with self._lock:
            incident = self._get_incident_unsafe(incident_id)
            old_status = incident.status
            incident.status = status
            incident.timeline.append(
                {
                    "ts": _now_iso(),
                    "actor": actor,
                    "action": "STATUS_CHANGED",
                    "notes": "Status changed from {} to {}. {}".format(
                        old_status.value, status.value, notes
                    ).strip(),
                }
            )
            return incident

    # ──────────────────────────────────────────────────────────────
    # TIMELINE
    # ──────────────────────────────────────────────────────────────

    def add_timeline_event(
        self,
        incident_id: str,
        actor: str,
        action: str,
        notes: str = "",
    ) -> None:
        """Append an event to the incident timeline.

        Parameters
        ----------
        incident_id : str
        actor : str
        action : str
            Short action description.
        notes : str, optional

        Raises
        ------
        KeyError
            If incident_id does not exist.
        """
        with self._lock:
            incident = self._get_incident_unsafe(incident_id)
            incident.timeline.append(
                {
                    "ts": _now_iso(),
                    "actor": actor,
                    "action": action,
                    "notes": notes,
                }
            )

    # ──────────────────────────────────────────────────────────────
    # REMEDIATION
    # ──────────────────────────────────────────────────────────────

    def attach_remediation(
        self, incident_id: str, plan: RemediationPlan
    ) -> None:
        """Attach a RemediationPlan to an incident.

        Parameters
        ----------
        incident_id : str
        plan : RemediationPlan

        Raises
        ------
        KeyError
            If incident_id does not exist.
        """
        with self._lock:
            self._get_incident_unsafe(incident_id)  # validate exists
            self._remediations[plan.plan_id] = plan
            incident = self._incidents[incident_id]
            incident.timeline.append(
                {
                    "ts": _now_iso(),
                    "actor": plan.created_by,
                    "action": "REMEDIATION_ATTACHED",
                    "notes": "Remediation plan '{}' attached with {} steps.".format(
                        plan.plan_id, len(plan.steps)
                    ),
                }
            )

    # ──────────────────────────────────────────────────────────────
    # CLOSE INCIDENT
    # ──────────────────────────────────────────────────────────────

    def close_incident(
        self, incident_id: str, resolution: str, actor: str
    ) -> IncidentRecord:
        """Mark an incident as CLOSED with a resolution summary.

        Sets ``resolved_at``, ``resolution_summary``, and status to CLOSED.

        Parameters
        ----------
        incident_id : str
        resolution : str
            Description of how the incident was resolved.
        actor : str
            Identity of the person or process closing the incident.

        Returns
        -------
        IncidentRecord

        Raises
        ------
        KeyError
            If incident_id does not exist.
        """
        now = _now_iso()
        with self._lock:
            incident = self._get_incident_unsafe(incident_id)
            incident.status = IncidentStatus.CLOSED
            incident.resolved_at = now
            incident.resolution_summary = resolution
            incident.timeline.append(
                {
                    "ts": now,
                    "actor": actor,
                    "action": "INCIDENT_CLOSED",
                    "notes": resolution,
                }
            )
            return incident

    # ──────────────────────────────────────────────────────────────
    # POSTMORTEM
    # ──────────────────────────────────────────────────────────────

    def create_postmortem(
        self,
        incident_id: str,
        root_cause: str,
        contributing_factors: List[str],
        what_went_well: List[str],
        what_went_wrong: List[str],
        action_items: List[str],
        authored_by: str,
    ) -> PostmortemArtifact:
        """Create a PostmortemArtifact and link it to the incident.

        Also updates incident status to POSTMORTEM.

        Parameters
        ----------
        incident_id : str
        root_cause : str
        contributing_factors : list[str]
        what_went_well : list[str]
        what_went_wrong : list[str]
        action_items : list[str]
        authored_by : str

        Returns
        -------
        PostmortemArtifact

        Raises
        ------
        KeyError
            If incident_id does not exist.
        """
        now = _now_iso()
        with self._lock:
            incident = self._get_incident_unsafe(incident_id)

            postmortem_id = _new_id()
            postmortem = PostmortemArtifact(
                postmortem_id=postmortem_id,
                incident_id=incident_id,
                authored_by=authored_by,
                authored_at=now,
                incident_summary=(
                    "Incident '{}' (severity {}) detected at {}.".format(
                        incident.title,
                        incident.severity.value,
                        incident.detected_at,
                    )
                ),
                timeline_summary=(
                    "{} timeline events recorded. "
                    "Resolved at: {}.".format(
                        len(incident.timeline),
                        incident.resolved_at or "not yet resolved",
                    )
                ),
                root_cause=root_cause,
                contributing_factors=tuple(contributing_factors),
                what_went_well=tuple(what_went_well),
                what_went_wrong=tuple(what_went_wrong),
                action_items=tuple(action_items),
                prevention_recommendations=(),
                related_incident_ids=(),
                schema_version="1.0",
            )

            self._postmortems[postmortem_id] = postmortem
            incident.postmortem_id = postmortem_id
            incident.status = IncidentStatus.POSTMORTEM
            incident.timeline.append(
                {
                    "ts": now,
                    "actor": authored_by,
                    "action": "POSTMORTEM_CREATED",
                    "notes": "Postmortem '{}' authored.".format(postmortem_id),
                }
            )

        return postmortem

    # ──────────────────────────────────────────────────────────────
    # AUDIT LEDGER
    # ──────────────────────────────────────────────────────────────

    def log_audit(self, record: AuditRecord) -> None:
        """Append an AuditRecord to the audit ledger.

        The ledger is capped at 50,000 records. When the cap is reached,
        the oldest 5,000 records are discarded (sliding window).

        Parameters
        ----------
        record : AuditRecord
        """
        with self._lock:
            self._audit_records.append(record)
            if len(self._audit_records) > _AUDIT_RECORD_CAP:
                # Discard oldest 10% to amortize the cost
                discard = _AUDIT_RECORD_CAP // 10
                self._audit_records = self._audit_records[discard:]

    def get_audit_records(
        self,
        actor: Optional[str] = None,
        record_type: Optional[AuditRecordType] = None,
        limit: int = 100,
    ) -> List[AuditRecord]:
        """Return audit records, optionally filtered and limited.

        Records are returned in reverse chronological order (most recent first).

        Parameters
        ----------
        actor : Optional[str]
            If provided, only return records for this actor.
        record_type : Optional[AuditRecordType]
            If provided, only return records of this type.
        limit : int
            Maximum number of records to return (default 100).

        Returns
        -------
        list[AuditRecord]
        """
        with self._lock:
            records = list(reversed(self._audit_records))

        if actor is not None:
            records = [r for r in records if r.actor == actor]
        if record_type is not None:
            records = [r for r in records if r.record_type == record_type]
        return records[:limit]

    # ──────────────────────────────────────────────────────────────
    # QUERY METHODS
    # ──────────────────────────────────────────────────────────────

    def get_incident(self, incident_id: str) -> Optional[IncidentRecord]:
        """Return an IncidentRecord by ID, or None if not found.

        Parameters
        ----------
        incident_id : str

        Returns
        -------
        Optional[IncidentRecord]
        """
        with self._lock:
            return self._incidents.get(incident_id)

    def list_incidents(
        self,
        status: Optional[IncidentStatus] = None,
        severity: Optional[IncidentSeverity] = None,
    ) -> List[IncidentRecord]:
        """Return all incidents, optionally filtered by status and severity.

        Results are sorted by ``detected_at`` descending (most recent first).

        Parameters
        ----------
        status : Optional[IncidentStatus]
        severity : Optional[IncidentSeverity]

        Returns
        -------
        list[IncidentRecord]
        """
        with self._lock:
            incidents = list(self._incidents.values())

        if status is not None:
            incidents = [i for i in incidents if i.status == status]
        if severity is not None:
            incidents = [i for i in incidents if i.severity == severity]

        return sorted(incidents, key=lambda i: i.detected_at, reverse=True)

    # ──────────────────────────────────────────────────────────────
    # REPEAT PATTERN DETECTION
    # ──────────────────────────────────────────────────────────────

    def get_repeat_patterns(self) -> List[Dict[str, Any]]:
        """Identify components that have appeared in more than one incident.

        Returns
        -------
        list[dict]
            Each dict has keys: ``component`` (str), ``incident_count`` (int),
            ``incident_ids`` (list[str]), ``severities`` (list[str]).
            Sorted by ``incident_count`` descending.
        """
        with self._lock:
            incidents = list(self._incidents.values())

        component_map: Dict[str, List[IncidentRecord]] = {}
        for incident in incidents:
            for component in incident.affected_components:
                component_map.setdefault(component, []).append(incident)

        patterns = []
        for component, incs in component_map.items():
            if len(incs) > 1:
                patterns.append(
                    {
                        "component": component,
                        "incident_count": len(incs),
                        "incident_ids": [i.incident_id for i in incs],
                        "severities": [i.severity.value for i in incs],
                    }
                )

        return sorted(patterns, key=lambda p: -p["incident_count"])

    # ──────────────────────────────────────────────────────────────
    # METRICS
    # ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for the incident manager.

        Returns
        -------
        dict
            Keys: counts_by_severity, counts_by_status, open_count,
            resolved_count, total_incidents, total_audit_records,
            avg_resolution_time_hours, repeat_incident_components.
        """
        with self._lock:
            incidents = list(self._incidents.values())
            audit_count = len(self._audit_records)

        by_severity: Dict[str, int] = {s.value: 0 for s in IncidentSeverity}
        by_status: Dict[str, int] = {s.value: 0 for s in IncidentStatus}
        open_count = 0
        resolved_count = 0
        resolution_times: List[float] = []

        for incident in incidents:
            by_severity[incident.severity.value] += 1
            by_status[incident.status.value] += 1

            if incident.status in (IncidentStatus.OPEN, IncidentStatus.IN_PROGRESS):
                open_count += 1
            if incident.status in (
                IncidentStatus.RESOLVED,
                IncidentStatus.CLOSED,
                IncidentStatus.POSTMORTEM,
            ):
                resolved_count += 1

            if incident.resolved_at and incident.detected_at:
                try:
                    detected = datetime.fromisoformat(incident.detected_at)
                    resolved = datetime.fromisoformat(incident.resolved_at)
                    hours = (resolved - detected).total_seconds() / 3600
                    if hours >= 0:
                        resolution_times.append(hours)
                except (ValueError, TypeError):
                    pass

        avg_resolution = (
            sum(resolution_times) / len(resolution_times)
            if resolution_times
            else 0.0
        )

        repeat_patterns = self.get_repeat_patterns()
        repeat_components = [p["component"] for p in repeat_patterns]

        return {
            "counts_by_severity": by_severity,
            "counts_by_status": by_status,
            "open_count": open_count,
            "resolved_count": resolved_count,
            "total_incidents": len(incidents),
            "total_audit_records": audit_count,
            "avg_resolution_time_hours": round(avg_resolution, 2),
            "repeat_incident_components": repeat_components,
        }

    # ──────────────────────────────────────────────────────────────
    # GOVERNANCE INTEGRATION
    # ──────────────────────────────────────────────────────────────

    def open_from_governed_action(
        self,
        record,  # GovernedActionRecord — typed loosely to avoid circular import
        trigger_source: str = IncidentTriggerSource.AGENT_ACTION.value,
    ) -> IncidentRecord:
        """Convenience factory: open an IncidentRecord from a GovernedActionRecord.

        Extracts all relevant fields from the governance record and creates a
        properly-structured incident. This is the canonical way for GovernanceRouter
        to open incidents — it avoids the boilerplate of calling `create_incident()`
        with raw kwargs at every call site.

        Also sets `requires_postmortem=True` and `postmortem_deadline` for P0
        incidents and emergency bypasses.

        Parameters
        ----------
        record : GovernedActionRecord
            The governance record for the executed action.
        trigger_source : str
            IncidentTriggerSource value string. Defaults to "agent_action".

        Returns
        -------
        IncidentRecord
        """
        from incidents.contracts import IncidentSeverity

        # Map governance tier / emergency bypass to severity
        severity_str = getattr(record, "incident_severity", None)
        if severity_str is None:
            # Determine from override_record_id (emergency bypass → P0)
            if getattr(record, "override_record_id", None):
                severity_str = "P0"
            else:
                severity_str = "P2"

        sev_map = {
            "P0": IncidentSeverity.P0_CRITICAL,
            "P1": IncidentSeverity.P1_HIGH,
            "P2": IncidentSeverity.P2_MEDIUM,
            "P3": IncidentSeverity.P3_LOW,
            "P4": IncidentSeverity.P4_INFO,
        }
        severity = sev_map.get(severity_str.upper(), IncidentSeverity.P2_MEDIUM)

        is_emergency_bypass = bool(getattr(record, "override_record_id", None))
        requires_postmortem = (
            severity == IncidentSeverity.P0_CRITICAL or is_emergency_bypass
        )

        now = _now_iso()
        postmortem_deadline = None
        if requires_postmortem:
            deadline_dt = datetime.now(timezone.utc) + timedelta(hours=24)
            postmortem_deadline = deadline_dt.isoformat()

        title = "{} in {} ({})".format(
            getattr(record, "action_type", "UNKNOWN"),
            getattr(record, "environment", "unknown"),
            "EMERGENCY_BYPASS" if is_emergency_bypass else "executed",
        )

        description = (
            "Governed action '{}' executed by agent '{}' in {} environment.\n"
            "Governance tier: {}.\n"
            "Trigger source: {}.\n"
            "Executed: {}. Error: {}.".format(
                getattr(record, "action_type", "UNKNOWN"),
                getattr(record, "source_agent", "unknown"),
                getattr(record, "environment", "unknown"),
                getattr(record, "governance_tier", "unknown"),
                trigger_source,
                getattr(record, "executed", False),
                getattr(record, "execution_error", None) or "none",
            )
        )

        tags = [
            getattr(record, "action_type", ""),
            str(getattr(record, "environment", "")),
            str(getattr(record, "governance_tier", "")),
            trigger_source,
        ]
        if is_emergency_bypass:
            tags.append("emergency_bypass")
        if requires_postmortem:
            tags.append("requires_postmortem")

        incident = self.create_incident(
            title=title,
            description=description,
            severity=severity,
            detected_by="governance_router",
            affected_components=["agent_feedback", "governance_router"],
            evidence_bundle_ids=[getattr(record, "action_id", "")],
            tags=[t for t in tags if t],
        )

        # Populate governance-specific fields
        incident.trigger_source = trigger_source
        incident.governed_action_id = getattr(record, "action_id", None)
        incident.requires_postmortem = requires_postmortem
        incident.postmortem_deadline = postmortem_deadline

        # Auto-link governance runbook
        if "governance_response" in self._runbooks:
            with self._lock:
                if "governance_response" not in incident.runbook_refs:
                    incident.runbook_refs.append("governance_response")

        if is_emergency_bypass and "kill_switch_triggered" in self._runbooks:
            with self._lock:
                if "kill_switch_triggered" not in incident.runbook_refs:
                    incident.runbook_refs.append("kill_switch_triggered")

        return incident

    def link_governed_action(
        self,
        incident_id: str,
        action_id: str,
    ) -> None:
        """Add a bidirectional link between an incident and a GovernedActionRecord.

        Sets `incident.governed_action_id` and appends a timeline event.
        Useful when the incident is created before the action is fully executed,
        or when multiple actions should be linked to the same incident.

        Parameters
        ----------
        incident_id : str
        action_id : str
            GovernedActionRecord.action_id

        Raises
        ------
        KeyError
            If incident_id does not exist.
        """
        with self._lock:
            incident = self._get_incident_unsafe(incident_id)
            incident.governed_action_id = action_id
            incident.timeline.append({
                "ts": _now_iso(),
                "actor": "governance_router",
                "action": "GOVERNED_ACTION_LINKED",
                "notes": "Linked to governed action '{}'.".format(action_id),
            })

    # ──────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────

    def _get_incident_unsafe(self, incident_id: str) -> IncidentRecord:
        """Return incident by ID or raise KeyError. Must be called under lock.

        Parameters
        ----------
        incident_id : str

        Returns
        -------
        IncidentRecord

        Raises
        ------
        KeyError
        """
        incident = self._incidents.get(incident_id)
        if incident is None:
            raise KeyError("Incident '{}' not found.".format(incident_id))
        return incident

    def _match_runbooks(
        self, affected_components: List[str], severity: IncidentSeverity
    ) -> List[str]:
        """Return runbook IDs matching the affected components and severity.

        Called under lock during create_incident.

        Parameters
        ----------
        affected_components : list[str]
        severity : IncidentSeverity

        Returns
        -------
        list[str]
            Matched runbook IDs.
        """
        matched = []
        component_set = {c.lower() for c in affected_components}
        sev_value = severity.value

        for runbook in self._runbooks.values():
            # Check severity match
            sev_match = (
                not runbook.applicable_severity
                or sev_value in runbook.applicable_severity
            )
            # Check component match
            comp_match = (
                not runbook.applicable_components
                or bool(
                    component_set
                    & {c.lower() for c in runbook.applicable_components}
                )
            )
            if sev_match and comp_match:
                matched.append(runbook.runbook_id)

        return matched

    def _register_default_runbooks(self) -> None:
        """Register the five default operational runbooks."""
        today = _today_iso()

        runbooks = [
            RunbookReference(
                runbook_id="drift_alert",
                title="Feature Drift / Data Distribution Shift",
                description=(
                    "Response procedure for alerts indicating feature drift "
                    "or data distribution shift detected by FeatureDriftMonitor."
                ),
                applicable_severity=(
                    IncidentSeverity.P1_HIGH.value,
                    IncidentSeverity.P2_MEDIUM.value,
                    IncidentSeverity.P3_LOW.value,
                ),
                applicable_components=("feature_store", "drift_monitor", "ml_pipeline"),
                steps=(
                    "1. Identify which features have drifted (PSI > threshold).",
                    "2. Check upstream data feeds for anomalies or pipeline failures.",
                    "3. Determine if drift is regime-driven or a data quality issue.",
                    "4. If data quality: escalate to data engineering, halt affected models.",
                    "5. If regime-driven: document in incident, update regime engine baseline.",
                    "6. Re-evaluate model performance on recent data; trigger retraining if PSI > 0.25.",
                    "7. Notify portfolio layer to reduce sizing until drift resolves.",
                    "8. Close incident when drift metrics return to normal range.",
                ),
                url=None,
                version="1.0",
                last_reviewed=today,
            ),
            RunbookReference(
                runbook_id="kill_switch_triggered",
                title="Kill-Switch Triggered",
                description=(
                    "Response procedure when the KillSwitchManager activates "
                    "and halts trading activity."
                ),
                applicable_severity=(
                    IncidentSeverity.P0_CRITICAL.value,
                    IncidentSeverity.P1_HIGH.value,
                ),
                applicable_components=(
                    "kill_switch",
                    "portfolio_allocator",
                    "risk_ops",
                ),
                steps=(
                    "1. Confirm kill-switch mode (SOFT_HALT, HARD_HALT, or LIQUIDATE).",
                    "2. Do NOT reset kill-switch without explicit approval.",
                    "3. Review triggering condition (drawdown, loss limit, manual).",
                    "4. Assess current open positions for immediate risk.",
                    "5. If LIQUIDATE mode: verify orderly position wind-down is proceeding.",
                    "6. Notify trading desk and risk management.",
                    "7. Document root cause in incident timeline.",
                    "8. Submit ChangeImpactReport before resetting the kill-switch.",
                    "9. Obtain ApprovalDecision (HUMAN_REQUIRED) before calling KillSwitchManager.reset().",
                    "10. Monitor closely for 2 hours after reset.",
                ),
                url=None,
                version="1.0",
                last_reviewed=today,
            ),
            RunbookReference(
                runbook_id="model_degraded",
                title="Model Performance Degradation",
                description=(
                    "Response procedure when a model's live performance falls "
                    "below governance thresholds (IC, AUC, Brier)."
                ),
                applicable_severity=(
                    IncidentSeverity.P1_HIGH.value,
                    IncidentSeverity.P2_MEDIUM.value,
                ),
                applicable_components=("ml_registry", "model_scorer", "governance"),
                steps=(
                    "1. Confirm degradation via ModelHealthMonitor.get_health_report().",
                    "2. Check if CHALLENGER model is available; if so, promote to shadow.",
                    "3. Freeze champion model from further inference on affected task family.",
                    "4. Verify fallback_triggered flag is being handled correctly by callers.",
                    "5. Initiate emergency retraining only if approved via ApprovalEngine.",
                    "6. Run LeakageAuditor before retraining to confirm data integrity.",
                    "7. Evaluate retrained model on OOT holdout; check all governance criteria.",
                    "8. Promote to CHALLENGER for shadow evaluation; monitor 5 days.",
                    "9. Promote to CHAMPION after governance approval.",
                    "10. Document in postmortem if degradation caused trading losses.",
                ),
                url=None,
                version="1.0",
                last_reviewed=today,
            ),
            RunbookReference(
                runbook_id="data_integrity_failure",
                title="Data Pipeline Integrity Failure",
                description=(
                    "Response procedure for data feed failures, corrupted price data, "
                    "or missing market data that affects signal generation."
                ),
                applicable_severity=(
                    IncidentSeverity.P0_CRITICAL.value,
                    IncidentSeverity.P1_HIGH.value,
                    IncidentSeverity.P2_MEDIUM.value,
                ),
                applicable_components=(
                    "data_provider",
                    "sql_store",
                    "feature_store",
                    "signal_engine",
                ),
                steps=(
                    "1. Identify which data source failed (IBKR, FMP, Yahoo Finance).",
                    "2. Check DataProvider priority routing; confirm fallback is active.",
                    "3. Assess staleness of cached data.",
                    "4. If stale > 30 minutes: halt signal generation for affected pairs.",
                    "5. Notify data engineering and investigate upstream failure.",
                    "6. Validate data integrity post-recovery using checksums or known-good snapshots.",
                    "7. Replay affected signals on recovered data before resuming.",
                    "8. Verify no stale features were used in open position decisions.",
                    "9. Document data gap in audit trail.",
                ),
                url=None,
                version="1.0",
                last_reviewed=today,
            ),
            RunbookReference(
                runbook_id="agent_stuck",
                title="Agent Task Stuck / Timeout",
                description=(
                    "Response procedure when an agent task is unresponsive, "
                    "stuck in a loop, or has exceeded its execution timeout."
                ),
                applicable_severity=(
                    IncidentSeverity.P2_MEDIUM.value,
                    IncidentSeverity.P3_LOW.value,
                ),
                applicable_components=("agent_registry", "orchestrator", "workflow"),
                steps=(
                    "1. Identify the stuck agent via AgentRegistry.get_status().",
                    "2. Check if the agent is in an infinite delegation loop (depth > 3).",
                    "3. Check for deadlocks via thread dump if available.",
                    "4. If agent holds positions: do NOT forcefully kill; contact risk management first.",
                    "5. Apply emergency_disable(agent_name) via GovernancePolicyEngine.",
                    "6. Recover any pending AgentTask objects from the task queue.",
                    "7. Root-cause the timeout: external API call, DB lock, or logic bug.",
                    "8. Fix root cause; reenable agent via emergency_reenable() with approval.",
                    "9. Replay missed tasks in dependency order.",
                    "10. Add timeout guard to the agent if missing.",
                ),
                url=None,
                version="1.0",
                last_reviewed=today,
            ),
        ]

        # Governance response runbook (Phase 1 addition)
        runbooks.append(RunbookReference(
            runbook_id="governance_response",
            title="Governed Agent Action — Response Protocol",
            description=(
                "Response procedure for incidents triggered by the GovernanceRouter: "
                "executed actions, emergency bypasses, SLA breaches, and precision demotions."
            ),
            applicable_severity=(
                IncidentSeverity.P0_CRITICAL.value,
                IncidentSeverity.P1_HIGH.value,
                IncidentSeverity.P2_MEDIUM.value,
                IncidentSeverity.P3_LOW.value,
            ),
            applicable_components=("agent_feedback", "governance_router", "approvals"),
            steps=(
                "1. Identify the GovernedActionRecord: check incident.governed_action_id "
                "   and retrieve from GovernanceRouter audit ledger.",
                "2. Review action_type, governance_tier, and evidence_snapshot.",
                "3. If EMERGENCY_BYPASS: verify override_record_id is present; "
                "   confirm circuit-breaker trigger was valid; schedule postmortem within 24h.",
                "4. If EXECUTION_FAILURE: check execution_error; determine root cause; "
                "   assess whether rollback is needed (check rollback_handle_id and deadline).",
                "5. If PRECISION_DEMOTION: review ActionPrecisionMetrics for the affected "
                "   action type; do NOT reset demotion without PM sign-off and justification.",
                "6. If SLA_BREACH: check ApprovalEngine.get_sla_breached_tickets(); "
                "   escalate pending tickets; review whether HUMAN_REQUIRED tier is "
                "   correctly calibrated for the approval capacity.",
                "7. To execute rollback: call GovernanceRouter.rollback(handle_id, rollback_fn) "
                "   with an appropriate state-restore function.",
                "8. Document resolution in incident timeline and close.",
            ),
            url=None,
            version="1.0",
            last_reviewed=today,
        ))

        for runbook in runbooks:
            self._runbooks[runbook.runbook_id] = runbook


# ══════════════════════════════════════════════════════════════════
# SINGLETON FACTORY
# ══════════════════════════════════════════════════════════════════

_manager_instance: Optional[IncidentManager] = None
_manager_lock = threading.Lock()


def get_incident_manager() -> IncidentManager:
    """Return the process-level singleton IncidentManager.

    Thread-safe. Instantiates on first call.

    Returns
    -------
    IncidentManager
    """
    global _manager_instance
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = IncidentManager()
    return _manager_instance
