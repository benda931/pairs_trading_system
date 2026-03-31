# -*- coding: utf-8 -*-
"""
reporting/dashboard.py
======================
Aggregates data from all governance subsystems into a unified read-only dashboard.

This is a pure reporting layer — it never mutates state in any subsystem.
All subsystem imports are lazy and wrapped in try/except so the dashboard
remains importable even when individual packages are not yet available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# ── Contracts ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ControlMatrixEntry:
    control_id: str
    control_name: str
    domain: str
    status: str
    last_tested: Optional[datetime]
    last_result: str
    critical: bool
    owner: str


@dataclass(frozen=True)
class GovernanceDashboardSummary:
    generated_at: datetime
    # Controls
    total_controls: int
    active_controls: int
    failed_controls: int
    degraded_controls: int
    critical_failures: int
    # Surveillance
    open_surveillance_events: int
    critical_surveillance_events: int
    open_surveillance_cases: int
    # Exceptions
    active_waivers: int
    pending_exception_requests: int
    # Attestations
    pending_attestations: int
    overdue_attestations: int
    # Policies
    active_policies: int
    total_policy_rules: int
    policy_violations_24h: int
    # Audit
    total_audit_entries: int
    audit_chains: int
    # SoD
    sod_violations: int
    critical_sod_violations: int
    # Overall
    governance_health: str
    top_concerns: tuple[str, ...]
    control_matrix: tuple[ControlMatrixEntry, ...]


# ── Dashboard ─────────────────────────────────────────────────────────────────

class GovernanceDashboard:
    """
    Aggregates data from all governance subsystems into a unified dashboard.

    This is a read-only reporting layer — it never mutates state.
    """

    def generate_summary(self) -> GovernanceDashboardSummary:
        """Collect metrics from all subsystems and produce a summary."""
        concerns: list[str] = []

        # Controls
        try:
            from controls.registry import get_control_registry  # type: ignore[import]
            ctrl_reg = get_control_registry()
            ctrl_metrics = ctrl_reg.get_metrics()
            ctrl_failed = ctrl_metrics.get("failed", 0)
            ctrl_degraded = ctrl_metrics.get("degraded", 0)
            ctrl_critical = ctrl_metrics.get("critical_failures", 0)
            ctrl_total = ctrl_metrics.get("total_controls", 0)
            ctrl_active = ctrl_metrics.get("active", 0)
            if ctrl_critical > 0:
                concerns.append(f"{ctrl_critical} critical control(s) failing")
        except Exception:
            ctrl_failed = ctrl_degraded = ctrl_critical = ctrl_total = ctrl_active = 0

        # Surveillance
        try:
            from surveillance.engine import get_surveillance_engine
            surv = get_surveillance_engine()
            surv_metrics = surv.get_metrics()
            surv_open = surv_metrics.get("open_events", 0)
            surv_critical = surv_metrics.get("critical_open", 0)
            surv_cases = surv_metrics.get("open_cases", 0)
            if surv_critical > 0:
                concerns.append(f"{surv_critical} critical surveillance event(s) open")
        except Exception:
            surv_open = surv_critical = surv_cases = 0

        # Exceptions
        try:
            from exceptions_mgmt.engine import get_exception_engine
            exc = get_exception_engine()
            exc_metrics = exc.get_metrics()
            exc_waivers = exc_metrics.get("active_waivers", 0)
            exc_pending = exc_metrics.get("pending", 0)
        except Exception:
            exc_waivers = exc_pending = 0

        # Attestations
        try:
            from attestations.engine import get_attestation_engine
            att = get_attestation_engine()
            att_metrics = att.get_metrics()
            att_pending = att_metrics.get("pending", 0)
            att_overdue = att_metrics.get("overdue", 0)
            if att_overdue > 0:
                concerns.append(f"{att_overdue} overdue attestation(s)")
        except Exception:
            att_pending = att_overdue = 0

        # Policies
        try:
            from policies.registry import get_policy_registry  # type: ignore[import]
            pol = get_policy_registry()
            pol_metrics = pol.get_metrics()
            pol_active = pol_metrics.get("active_policies", 0)
            pol_rules = pol_metrics.get("total_active_rules", 0)
            pol_violations = pol_metrics.get("violations", 0)
        except Exception:
            pol_active = pol_rules = pol_violations = 0

        # Audit
        try:
            from audit.chain import get_audit_chain_registry  # type: ignore[import]
            audit_reg = get_audit_chain_registry()
            audit_total = audit_reg.total_entries
            audit_chains = len(audit_reg.list_chain_ids())
        except Exception:
            audit_total = audit_chains = 0

        # SoD
        try:
            from operating_model.access import get_access_control_manager
            acm = get_access_control_manager()
            acm_metrics = acm.get_metrics()
            sod_violations = acm_metrics.get("sod_violations", 0)
            sod_critical = acm_metrics.get("critical_violations", 0)
            if sod_critical > 0:
                concerns.append(f"{sod_critical} critical SoD violation(s)")
        except Exception:
            sod_violations = sod_critical = 0

        # Health determination
        if ctrl_critical > 0 or surv_critical > 0 or sod_critical > 0:
            health = "RED"
        elif ctrl_failed > 0 or ctrl_degraded > 0 or att_overdue > 0 or surv_open > 2:
            health = "AMBER"
        else:
            health = "GREEN"

        return GovernanceDashboardSummary(
            generated_at=datetime.utcnow(),
            total_controls=ctrl_total,
            active_controls=ctrl_active,
            failed_controls=ctrl_failed,
            degraded_controls=ctrl_degraded,
            critical_failures=ctrl_critical,
            open_surveillance_events=surv_open,
            critical_surveillance_events=surv_critical,
            open_surveillance_cases=surv_cases,
            active_waivers=exc_waivers,
            pending_exception_requests=exc_pending,
            pending_attestations=att_pending,
            overdue_attestations=att_overdue,
            active_policies=pol_active,
            total_policy_rules=pol_rules,
            policy_violations_24h=pol_violations,
            total_audit_entries=audit_total,
            audit_chains=audit_chains,
            sod_violations=sod_violations,
            critical_sod_violations=sod_critical,
            governance_health=health,
            top_concerns=tuple(concerns[:5]),
            control_matrix=(),  # populated separately via build_control_matrix()
        )

    def build_control_matrix(self) -> tuple[ControlMatrixEntry, ...]:
        """Build the full control matrix from the control registry."""
        try:
            from controls.registry import get_control_registry  # type: ignore[import]
            ctrl_reg = get_control_registry()
            entries: list[ControlMatrixEntry] = []
            for ctrl in ctrl_reg.list_controls():
                status = ctrl_reg.get_status(ctrl.control_id)
                history = ctrl_reg.get_test_history(ctrl.control_id)
                last_test = history[-1] if history else None
                entries.append(ControlMatrixEntry(
                    control_id=ctrl.control_id,
                    control_name=ctrl.name,
                    domain=ctrl.domain.value,
                    status=status.value if status else "UNKNOWN",
                    last_tested=last_test.tested_at if last_test else None,
                    last_result=last_test.result.value if last_test else "UNTESTED",
                    critical=ctrl.critical,
                    owner=ctrl.owner,
                ))
            return tuple(entries)
        except Exception:
            return ()

    def generate_full_summary(self) -> GovernanceDashboardSummary:
        """Generate summary with control matrix populated."""
        summary = self.generate_summary()
        matrix = self.build_control_matrix()
        # Reconstruct with populated control_matrix
        return GovernanceDashboardSummary(
            generated_at=summary.generated_at,
            total_controls=summary.total_controls,
            active_controls=summary.active_controls,
            failed_controls=summary.failed_controls,
            degraded_controls=summary.degraded_controls,
            critical_failures=summary.critical_failures,
            open_surveillance_events=summary.open_surveillance_events,
            critical_surveillance_events=summary.critical_surveillance_events,
            open_surveillance_cases=summary.open_surveillance_cases,
            active_waivers=summary.active_waivers,
            pending_exception_requests=summary.pending_exception_requests,
            pending_attestations=summary.pending_attestations,
            overdue_attestations=summary.overdue_attestations,
            active_policies=summary.active_policies,
            total_policy_rules=summary.total_policy_rules,
            policy_violations_24h=summary.policy_violations_24h,
            total_audit_entries=summary.total_audit_entries,
            audit_chains=summary.audit_chains,
            sod_violations=summary.sod_violations,
            critical_sod_violations=summary.critical_sod_violations,
            governance_health=summary.governance_health,
            top_concerns=summary.top_concerns,
            control_matrix=matrix,
        )


# ── Singleton accessor ─────────────────────────────────────────────────────────

_dashboard: Optional[GovernanceDashboard] = None


def get_governance_dashboard() -> GovernanceDashboard:
    """Return the module-level singleton GovernanceDashboard."""
    global _dashboard
    if _dashboard is None:
        _dashboard = GovernanceDashboard()
    return _dashboard
