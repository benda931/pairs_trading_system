# -*- coding: utf-8 -*-
"""
operating_model/access.py
=========================
RBAC with Segregation-of-Duties (SoD) enforcement for the trading platform.

Manages permission grants, validates role conflicts, and tracks RACI
responsibility assignments.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


# ── Enums ─────────────────────────────────────────────────────────────────────

class AccessRole(Enum):
    QUANT_RESEARCHER = "QUANT_RESEARCHER"
    ML_ENGINEER = "ML_ENGINEER"
    RISK_OFFICER = "RISK_OFFICER"
    TRADING_OPERATOR = "TRADING_OPERATOR"
    GOVERNANCE_OFFICER = "GOVERNANCE_OFFICER"
    DEVOPS_ENGINEER = "DEVOPS_ENGINEER"
    INCIDENT_MANAGER = "INCIDENT_MANAGER"
    SYSTEM_ADMIN = "SYSTEM_ADMIN"
    LIVE_TRADING_APPROVER = "LIVE_TRADING_APPROVER"
    AUDITOR = "AUDITOR"
    COMPLIANCE_OFFICER = "COMPLIANCE_OFFICER"


class PermissionType(Enum):
    READ = "READ"
    WRITE = "WRITE"
    APPROVE = "APPROVE"
    EXECUTE = "EXECUTE"
    EMERGENCY_OVERRIDE = "EMERGENCY_OVERRIDE"
    AUDIT_READ = "AUDIT_READ"


class SodViolationSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ── Contracts ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PermissionGrant:
    grant_id: str
    principal: str
    role: AccessRole
    permissions: tuple[PermissionType, ...]
    resource_scope: str
    granted_by: str
    granted_at: datetime
    expires_at: Optional[datetime]
    active: bool
    notes: str = ""


@dataclass(frozen=True)
class SodRule:
    rule_id: str
    name: str
    description: str
    conflicting_roles: tuple[AccessRole, ...]
    severity: SodViolationSeverity
    automated_enforcement: bool


@dataclass(frozen=True)
class SodViolation:
    violation_id: str
    rule_id: str
    principal: str
    conflicting_roles_held: tuple[AccessRole, ...]
    detected_at: datetime
    severity: SodViolationSeverity
    remediation: str


@dataclass(frozen=True)
class ResponsibilityAssignment:
    """RACI matrix entry for a given process."""
    assignment_id: str
    process: str
    responsible: str
    accountable: str
    consulted: tuple[str, ...]
    informed: tuple[str, ...]
    effective_from: datetime


# ── Manager ───────────────────────────────────────────────────────────────────

class AccessControlManager:
    """
    RBAC with SoD enforcement for the trading platform.
    Manages permission grants, validates role conflicts, and tracks responsibility.
    """

    DEFAULT_SOD_RULES: list[SodRule] = [
        SodRule(
            "SOD-001",
            "Signal-Execution Separation",
            "Agent/role proposing signals cannot also submit orders",
            (AccessRole.QUANT_RESEARCHER, AccessRole.TRADING_OPERATOR),
            SodViolationSeverity.CRITICAL,
            True,
        ),
        SodRule(
            "SOD-002",
            "Model Train-Promote Separation",
            "The same engineer cannot train AND promote to champion without a second approver",
            (AccessRole.ML_ENGINEER, AccessRole.GOVERNANCE_OFFICER),
            SodViolationSeverity.WARNING,
            False,
        ),
        SodRule(
            "SOD-003",
            "Risk-Trading Separation",
            "A risk officer cannot also be a live trading operator",
            (AccessRole.RISK_OFFICER, AccessRole.TRADING_OPERATOR),
            SodViolationSeverity.WARNING,
            True,
        ),
        SodRule(
            "SOD-004",
            "Audit Independence",
            "Auditors cannot have write permissions to audited systems",
            (AccessRole.AUDITOR, AccessRole.SYSTEM_ADMIN),
            SodViolationSeverity.CRITICAL,
            True,
        ),
        SodRule(
            "SOD-005",
            "Governance Neutrality",
            "Governance officers should not have live trading activation rights",
            (AccessRole.GOVERNANCE_OFFICER, AccessRole.LIVE_TRADING_APPROVER),
            SodViolationSeverity.WARNING,
            False,
        ),
    ]

    def __init__(self) -> None:
        self._grants: dict[str, PermissionGrant] = {}
        self._grants_by_principal: dict[str, list[str]] = {}
        self._sod_rules: dict[str, SodRule] = {
            r.rule_id: r for r in self.DEFAULT_SOD_RULES
        }
        self._violations: list[SodViolation] = []
        self._responsibilities: dict[str, ResponsibilityAssignment] = {}

    def grant_role(
        self,
        principal: str,
        role: AccessRole,
        permissions: list[PermissionType],
        granted_by: str,
        resource_scope: str = "*",
        expires_at: datetime = None,
        notes: str = "",
    ) -> PermissionGrant:
        """Grant a role with specified permissions to a principal."""
        grant = PermissionGrant(
            grant_id=str(uuid.uuid4()),
            principal=principal,
            role=role,
            permissions=tuple(permissions),
            resource_scope=resource_scope,
            granted_by=granted_by,
            granted_at=datetime.utcnow(),
            expires_at=expires_at,
            active=True,
            notes=notes,
        )
        self._grants[grant.grant_id] = grant
        self._grants_by_principal.setdefault(principal, []).append(grant.grant_id)

        # Immediate SoD check after granting
        self._check_sod_for_principal(principal)

        return grant

    def revoke_grant(self, grant_id: str) -> bool:
        """Revoke an active permission grant."""
        if grant_id not in self._grants:
            return False
        g = self._grants[grant_id]
        self._grants[grant_id] = PermissionGrant(
            grant_id=g.grant_id,
            principal=g.principal,
            role=g.role,
            permissions=g.permissions,
            resource_scope=g.resource_scope,
            granted_by=g.granted_by,
            granted_at=g.granted_at,
            expires_at=g.expires_at,
            active=False,
            notes=g.notes,
        )
        return True

    def get_roles(self, principal: str) -> list[AccessRole]:
        """Return all currently active roles for a principal."""
        now = datetime.utcnow()
        grant_ids = self._grants_by_principal.get(principal, [])
        roles: list[AccessRole] = []
        for gid in grant_ids:
            g = self._grants[gid]
            if g.active and (g.expires_at is None or g.expires_at > now):
                roles.append(g.role)
        return roles

    def has_permission(
        self,
        principal: str,
        permission: PermissionType,
        resource: str = None,
    ) -> bool:
        """
        Check whether a principal holds a permission, optionally scoped to a resource.
        Resource scope matching is prefix-based when scope ends with '*'.
        """
        now = datetime.utcnow()
        grant_ids = self._grants_by_principal.get(principal, [])
        for gid in grant_ids:
            g = self._grants[gid]
            if not g.active:
                continue
            if g.expires_at and g.expires_at <= now:
                continue
            if permission not in g.permissions:
                continue
            if resource and g.resource_scope != "*":
                if not resource.startswith(g.resource_scope.rstrip("*")):
                    continue
            return True
        return False

    def _check_sod_for_principal(self, principal: str) -> None:
        """Check SoD rules for a single principal and record any violations found."""
        roles = set(self.get_roles(principal))
        for rule in self._sod_rules.values():
            if all(r in roles for r in rule.conflicting_roles):
                violation = SodViolation(
                    violation_id=str(uuid.uuid4()),
                    rule_id=rule.rule_id,
                    principal=principal,
                    conflicting_roles_held=tuple(
                        r for r in rule.conflicting_roles if r in roles
                    ),
                    detected_at=datetime.utcnow(),
                    severity=rule.severity,
                    remediation=(
                        f"Remove one of the conflicting roles from principal '{principal}'"
                    ),
                )
                self._violations.append(violation)

    def check_sod(self) -> list[SodViolation]:
        """Run a full SoD check across all principals. Returns all detected violations."""
        self._violations.clear()
        for principal in list(self._grants_by_principal.keys()):
            self._check_sod_for_principal(principal)
        return list(self._violations)

    def add_sod_rule(self, rule: SodRule) -> None:
        """Register an additional SoD rule."""
        self._sod_rules[rule.rule_id] = rule

    def assign_responsibility(
        self,
        process: str,
        responsible: str,
        accountable: str,
        consulted: list[str] = None,
        informed: list[str] = None,
    ) -> ResponsibilityAssignment:
        """Create or replace the RACI assignment for a process."""
        assignment = ResponsibilityAssignment(
            assignment_id=str(uuid.uuid4()),
            process=process,
            responsible=responsible,
            accountable=accountable,
            consulted=tuple(consulted or []),
            informed=tuple(informed or []),
            effective_from=datetime.utcnow(),
        )
        self._responsibilities[process] = assignment
        return assignment

    def get_responsibility(self, process: str) -> Optional[ResponsibilityAssignment]:
        """Look up the RACI assignment for a process."""
        return self._responsibilities.get(process)

    def get_sod_violations(self) -> list[SodViolation]:
        """Return the current list of recorded SoD violations."""
        return list(self._violations)

    def get_grant(self, grant_id: str) -> Optional[PermissionGrant]:
        """Look up a permission grant by ID."""
        return self._grants.get(grant_id)

    def get_metrics(self) -> dict:
        """Return aggregate metrics for monitoring dashboards."""
        now = datetime.utcnow()
        active_grants = [
            g for g in self._grants.values()
            if g.active and (g.expires_at is None or g.expires_at > now)
        ]
        return {
            "total_grants": len(self._grants),
            "active_grants": len(active_grants),
            "unique_principals": len(self._grants_by_principal),
            "sod_violations": len(self._violations),
            "critical_violations": sum(
                1 for v in self._violations
                if v.severity == SodViolationSeverity.CRITICAL
            ),
        }


# ── Singleton accessor ─────────────────────────────────────────────────────────

_manager: Optional[AccessControlManager] = None


def get_access_control_manager() -> AccessControlManager:
    """Return the module-level singleton AccessControlManager."""
    global _manager
    if _manager is None:
        _manager = AccessControlManager()
    return _manager
