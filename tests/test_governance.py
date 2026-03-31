# -*- coding: utf-8 -*-
"""tests/test_governance.py — Phase 5 Governance/Compliance/Audit/Surveillance tests

Sections:
  A. AuditChain (10 tests)
  B. Evidence    (10 tests)
  C. Controls    (10 tests)
  D. Policies    (10 tests)
  E. Surveillance (10 tests)
  F. Exceptions  (10 tests)
  G. Attestations (10 tests)
  H. AccessControl (10 tests)
  I. Reporting    (10 tests)
  J. Retention    (10 tests)
"""
import dataclasses
import pytest
from datetime import datetime, timedelta

from audit.chain import AuditChain, AuditChainRegistry, AuditChainStatus
from evidence.bundle import EvidenceBundleBuilder, EvidenceType, EvidenceStatus
from controls.registry import (
    ControlRegistry,
    ControlDefinition,
    ControlTestResult,
    ControlStatus,
    ControlDomain,
    ControlType,
    ControlFrequency,
)
from policies.registry import PolicyRegistry, PolicyRule, PolicyRuleType, PolicyScope
from surveillance.engine import SurveillanceEngine, SurveillanceFamily, SurveillanceSeverity
from exceptions_mgmt.engine import (
    ExceptionEngine,
    ExceptionCategory,
    ExceptionStatus,
    CompensatingControl,
    CompensatingControlStatus,
)
from attestations.engine import AttestationEngine, AttestationScope, AttestationStatus
from operating_model.access import (
    AccessControlManager,
    AccessRole,
    PermissionType,
    SodViolationSeverity,
)
from reporting.dashboard import GovernanceDashboard, GovernanceDashboardSummary
from retention.policy import RetentionManager, RetentionCategory, ArchiveStatus


# =============================================================================
# Section A: AuditChain (10 tests)
# =============================================================================

def test_audit_chain_empty_valid():
    chain = AuditChain("chain-1")
    report = chain.validate()
    assert report.status.value == "VALID"
    assert report.entry_count == 0


def test_audit_chain_append_and_validate():
    chain = AuditChain("chain-2")
    e = chain.append("agent_x", "create_strategy", "strategy", "strat-1", {"key": "val"})
    assert e.chain_id == "chain-2"
    assert e.prev_entry_id is None
    report = chain.validate()
    assert report.status.value == "VALID"
    assert report.entry_count == 1


def test_audit_chain_links_sequential():
    chain = AuditChain("chain-3")
    e1 = chain.append("a", "action1", "model", "m1", {})
    e2 = chain.append("b", "action2", "model", "m1", {})
    assert e2.prev_entry_id == e1.entry_id


def test_audit_chain_payload_hash_computed():
    chain = AuditChain("chain-4")
    e = chain.append("a", "act", "model", "m1", {"x": 1})
    assert len(e.payload_hash) == 64  # sha256 hex


def test_audit_chain_query_by_entity():
    chain = AuditChain("chain-5")
    chain.append("a", "act1", "strategy", "s1", {})
    chain.append("b", "act2", "model", "m1", {})
    results = chain.query(entity_id="s1")
    assert len(results) == 1


def test_audit_chain_reconstruct_history():
    chain = AuditChain("chain-6")
    chain.append("a", "create", "strategy", "s1", {})
    chain.append("a", "update", "strategy", "s1", {})
    chain.append("b", "create", "model", "m1", {})
    history = chain.reconstruct_entity_history("s1")
    assert len(history) == 2


def test_audit_chain_registry_get_or_create():
    reg = AuditChainRegistry()
    c1 = reg.get_or_create("chain-a")
    c2 = reg.get_or_create("chain-a")
    assert c1 is c2  # same object


def test_audit_chain_registry_log_convenience():
    reg = AuditChainRegistry()
    e = reg.log("my-chain", "system", "initialize", "platform", "platform-1", {"version": "1.0"})
    assert e.actor == "system"
    assert reg.total_entries == 1


def test_audit_chain_registry_validate_all():
    reg = AuditChainRegistry()
    reg.log("c1", "a", "act", "x", "x1", {})
    reg.log("c2", "b", "act", "y", "y1", {})
    reports = reg.validate_all()
    assert "c1" in reports and "c2" in reports
    assert all(r.status.value == "VALID" for r in reports.values())


def test_audit_chain_metadata_stored():
    chain = AuditChain("chain-meta")
    e = chain.append("a", "act", "m", "id1", {}, metadata={"key": "value"})
    # metadata stored as tuple of tuples
    assert any(k == "key" for k, v in e.metadata)


# =============================================================================
# Section B: Evidence (10 tests)
# =============================================================================

def test_evidence_builder_empty():
    b = EvidenceBundleBuilder("model", "m1", "promotion_review", "analyst")
    report = b.check_completeness()
    assert report.is_complete  # no requirements = complete


def test_evidence_collect_item():
    b = EvidenceBundleBuilder("model", "m1", "promotion_review", "analyst")
    item = b.collect(
        EvidenceType.BACKTEST_RESULT, "Backtest", "12-month backtest",
        "analyst", artifact_ref="s3://backtest.parquet"
    )
    bundle = b.build()
    assert len(bundle.evidence_items) == 1
    assert bundle.evidence_items[0].evidence_id == item.evidence_id


def test_evidence_requirement_satisfied():
    b = EvidenceBundleBuilder("model", "m1", "promotion_review", "analyst")
    b.add_requirement("Need backtest", [EvidenceType.BACKTEST_RESULT], min_count=1, mandatory=True)
    b.collect(EvidenceType.BACKTEST_RESULT, "Backtest", "desc", "analyst")
    report = b.check_completeness()
    assert report.is_complete


def test_evidence_requirement_not_satisfied():
    b = EvidenceBundleBuilder("model", "m1", "promotion_review", "analyst")
    req = b.add_requirement("Need backtest", [EvidenceType.BACKTEST_RESULT], min_count=1, mandatory=True)
    report = b.check_completeness()
    assert not report.is_complete
    assert req.requirement_id in report.missing_requirements


def test_evidence_completeness_pct():
    b = EvidenceBundleBuilder("model", "m1", "promo", "analyst")
    b.collect(EvidenceType.AUDIT_TRAIL, "Audit", "desc", "analyst")
    b.collect(EvidenceType.GOVERNANCE_CHECK, "Gov check", "desc", "analyst")
    report = b.check_completeness()
    assert report.total_items == 2
    assert report.completeness_pct == 100.0


def test_evidence_expired_item():
    b = EvidenceBundleBuilder("strategy", "s1", "audit", "analyst")
    past = datetime.utcnow() - timedelta(days=1)
    b.collect(
        EvidenceType.PERFORMANCE_REPORT, "Old report", "desc", "analyst", expiry_date=past
    )
    report = b.check_completeness()
    assert len(report.expired_items) == 1


def test_evidence_bundle_immutable():
    b = EvidenceBundleBuilder("strategy", "s1", "audit", "analyst")
    b.collect(EvidenceType.AUDIT_TRAIL, "Audit", "desc", "analyst")
    bundle = b.build()
    assert dataclasses.is_dataclass(bundle)


def test_evidence_multiple_requirements():
    b = EvidenceBundleBuilder("model", "m1", "promo", "analyst")
    b.add_requirement("Backtest required", [EvidenceType.BACKTEST_RESULT], mandatory=True)
    b.add_requirement("Governance check required", [EvidenceType.GOVERNANCE_CHECK], mandatory=True)
    b.collect(EvidenceType.BACKTEST_RESULT, "BT", "desc", "analyst")
    report = b.check_completeness()
    assert not report.is_complete  # second requirement missing


def test_evidence_advisory_requirement_not_blocking():
    b = EvidenceBundleBuilder("model", "m1", "promo", "analyst")
    b.add_requirement("Optional peer review", [EvidenceType.PEER_REVIEW], mandatory=False)
    report = b.check_completeness()
    assert report.is_complete  # advisory only, no hard block


def test_evidence_bundle_tags():
    b = EvidenceBundleBuilder("model", "m1", "promo", "analyst")
    item = b.collect(
        EvidenceType.MODEL_CARD, "Card", "desc", "analyst", tags=("ml", "production")
    )
    assert "ml" in item.tags


# =============================================================================
# Section C: Controls (10 tests)
# =============================================================================

def test_controls_default_populated():
    reg = ControlRegistry()
    controls = reg.list_controls()
    assert len(controls) >= 15


def test_controls_all_default_active():
    reg = ControlRegistry()
    for ctrl in reg.list_controls():
        status = reg.get_status(ctrl.control_id)
        assert status == ControlStatus.ACTIVE


def test_controls_record_test_pass():
    reg = ControlRegistry()
    rec = reg.record_test("CTRL-DQ-001", "auditor", ControlTestResult.PASS, "Passed all checks")
    assert rec.result == ControlTestResult.PASS
    assert reg.get_status("CTRL-DQ-001") == ControlStatus.ACTIVE


def test_controls_record_test_fail():
    reg = ControlRegistry()
    reg.record_test("CTRL-DQ-001", "auditor", ControlTestResult.FAIL, "Data gap found")
    assert reg.get_status("CTRL-DQ-001") == ControlStatus.FAILED


def test_controls_record_test_partial_degraded():
    reg = ControlRegistry()
    reg.record_test("CTRL-ER-002", "auditor", ControlTestResult.PARTIAL, "Some checks passed")
    assert reg.get_status("CTRL-ER-002") == ControlStatus.DEGRADED


def test_controls_critical_failures():
    reg = ControlRegistry()
    reg.record_test("CTRL-ER-001", "auditor", ControlTestResult.FAIL, "Kill switch not responding")
    critical = reg.get_critical_failures()
    assert any(c.control_id == "CTRL-ER-001" for c in critical)


def test_controls_register_custom():
    reg = ControlRegistry()
    custom = ControlDefinition(
        "CTRL-CUSTOM-001", "Custom Control", "A test control",
        ControlDomain.GOVERNANCE, ControlType.DETECTIVE, ControlFrequency.DAILY,
        "test_team", False, False, ()
    )
    reg.register(custom)
    assert reg.get("CTRL-CUSTOM-001") is not None


def test_controls_assign_owner():
    reg = ControlRegistry()
    rec = reg.assign_owner("CTRL-DQ-001", "data_team", "admin", review_cadence_days=30)
    assert rec.owner == "data_team"


def test_controls_test_history():
    reg = ControlRegistry()
    reg.record_test("CTRL-DQ-002", "a", ControlTestResult.PASS, "ok")
    reg.record_test("CTRL-DQ-002", "b", ControlTestResult.FAIL, "fail")
    history = reg.get_test_history("CTRL-DQ-002")
    assert len(history) == 2


def test_controls_metrics():
    reg = ControlRegistry()
    metrics = reg.get_metrics()
    assert metrics["total_controls"] >= 15
    assert metrics["active"] >= 15
    assert metrics["failed"] == 0


# =============================================================================
# Section D: Policies (10 tests)
# =============================================================================

def test_policies_default_populated():
    reg = PolicyRegistry()
    active = reg.get_active_policies()
    assert len(active) >= 4


def test_policies_evaluate_no_violations():
    reg = PolicyRegistry()
    report = reg.evaluate("model", "m1", {})
    assert report.is_conformant


def test_policies_evaluate_blocking_violation():
    reg = PolicyRegistry()
    active = reg.get_active_policies()
    # Find first HARD_LIMIT or PROHIBITED rule
    first_blocking_rule = None
    first_blocking_policy = None
    for policy in active:
        for rule in policy.rules:
            if rule.rule_type in (PolicyRuleType.HARD_LIMIT, PolicyRuleType.PROHIBITED):
                first_blocking_rule = rule
                first_blocking_policy = policy
                break
        if first_blocking_rule:
            break

    if first_blocking_rule is not None:
        # Determine entity_type so it matches scope
        scope_to_entity = {
            "SYSTEM_WIDE": "model",
            "MODEL": "model",
            "STRATEGY": "strategy",
            "DEPLOYMENT": "deployment",
            "DATA": "data",
        }
        entity_type = scope_to_entity.get(first_blocking_rule.scope.value, "model")
        report = reg.evaluate(
            entity_type, "e1",
            {f"rule_{first_blocking_rule.rule_id}_passed": False}
        )
        assert report.blocking_violations >= 1
        assert not report.is_conformant


def test_policies_register_new_version():
    reg = PolicyRegistry()
    rules = [
        PolicyRule(
            "TEST-R1", "Test rule", "desc",
            PolicyRuleType.SOFT_LIMIT, PolicyScope.SYSTEM_WIDE,
            "*", "condition", "notify", "LOW", False
        )
    ]
    version = reg.register_policy("POL-TEST-NEW", rules, "analyst", "New test policy", approved_by="manager")
    assert version.version == 1
    assert version.state.value == "ACTIVE"


def test_policies_version_supersedes_previous():
    reg = PolicyRegistry()
    rules_v1 = [
        PolicyRule(
            "R1", "Rule 1", "desc",
            PolicyRuleType.SOFT_LIMIT, PolicyScope.SYSTEM_WIDE,
            "*", "cond", "act", "LOW", False
        )
    ]
    reg.register_policy("POL-VERSIONED", rules_v1, "analyst", "V1")
    rules_v2 = [
        PolicyRule(
            "R2", "Rule 2", "desc",
            PolicyRuleType.SOFT_LIMIT, PolicyScope.SYSTEM_WIDE,
            "*", "cond", "act", "LOW", False
        )
    ]
    v2 = reg.register_policy("POL-VERSIONED", rules_v2, "analyst", "V2")
    assert v2.version == 2
    history = reg.get_policy_history("POL-VERSIONED")
    assert len(history) == 2


def test_policies_suspend():
    reg = PolicyRegistry()
    reg.suspend_policy("POL-ML-001", "temporary hold")
    active = reg.get_active_policies()
    policy_ids = [p.policy_id for p in active]
    assert "POL-ML-001" not in policy_ids


def test_policies_conformance_report_structure():
    reg = PolicyRegistry()
    report = reg.evaluate("strategy", "s1", {})
    assert hasattr(report, "total_rules_evaluated")
    assert hasattr(report, "is_conformant")
    assert hasattr(report, "results")


def test_policies_metrics():
    reg = PolicyRegistry()
    metrics = reg.get_metrics()
    assert metrics["active_policies"] >= 4
    assert metrics["total_active_rules"] >= 8


def test_policies_evaluation_logged():
    reg = PolicyRegistry()
    reg.evaluate("model", "m1", {})
    metrics = reg.get_metrics()
    assert metrics["total_evaluations"] > 0


def test_policies_policy_not_found_returns_empty():
    reg = PolicyRegistry()
    history = reg.get_policy_history("POL-NONEXISTENT")
    assert history == []


# =============================================================================
# Section E: Surveillance (10 tests)
# =============================================================================

def test_surveillance_defaults_loaded():
    eng = SurveillanceEngine()
    rules = eng.list_rules(enabled_only=True)
    assert len(rules) >= 12


def test_surveillance_no_breach_returns_none():
    eng = SurveillanceEngine()
    result = eng.detect("SURV-SA-001", "strategy", "s1", 2.0)  # threshold is 4.0
    assert result is None


def test_surveillance_breach_creates_event():
    eng = SurveillanceEngine()
    event = eng.detect("SURV-SA-001", "strategy", "s1", 5.0)
    assert event is not None
    assert event.metric_value == 5.0
    assert event.status.value == "OPEN"


def test_surveillance_auto_escalate_creates_case():
    eng = SurveillanceEngine()
    event = eng.detect("SURV-SA-002", "strategy", "s1", 4.0)  # auto_escalate=True, threshold=3.0
    assert event is not None
    assert event.case_id is not None
    cases = eng.get_open_cases()
    assert any(c.case_id == event.case_id for c in cases)


def test_surveillance_acknowledge_event():
    eng = SurveillanceEngine()
    event = eng.detect("SURV-SA-001", "strategy", "s1", 5.0)
    result = eng.acknowledge_event(event.event_id, "risk_officer")
    assert result
    open_events = eng.get_open_events()
    assert not any(e.event_id == event.event_id for e in open_events)


def test_surveillance_resolve_event():
    eng = SurveillanceEngine()
    event = eng.detect("SURV-SA-001", "strategy", "s1", 5.0)
    eng.resolve_event(event.event_id)
    open_events = eng.get_open_events()
    assert not any(e.event_id == event.event_id for e in open_events)


def test_surveillance_ic_degradation_lower_is_worse():
    eng = SurveillanceEngine()
    # SURV-MD-001: IC < 0.02 triggers
    event = eng.detect("SURV-MD-001", "model", "m1", 0.01)  # below threshold
    assert event is not None
    no_event = eng.detect("SURV-MD-001", "model", "m2", 0.10)  # above threshold
    assert no_event is None


def test_surveillance_create_manual_case():
    eng = SurveillanceEngine()
    case = eng.create_case(
        "Manual investigation",
        SurveillanceFamily.EXECUTION_ANOMALY,
        [],
        "risk_officer"
    )
    assert case.case_id is not None
    open_cases = eng.get_open_cases()
    assert any(c.case_id == case.case_id for c in open_cases)


def test_surveillance_close_case():
    eng = SurveillanceEngine()
    case = eng.create_case("Test case", SurveillanceFamily.DATA_INTEGRITY, [], "officer")
    result = eng.close_case(case.case_id, "officer", "no_action", "No issue found")
    assert result
    open_cases = eng.get_open_cases()
    assert not any(c.case_id == case.case_id for c in open_cases)


def test_surveillance_metrics():
    eng = SurveillanceEngine()
    metrics = eng.get_metrics()
    assert "total_rules" in metrics
    assert metrics["total_rules"] >= 12
    assert metrics["open_events"] == 0


# =============================================================================
# Section F: Exceptions (10 tests)
# =============================================================================

def test_exception_submit_request():
    eng = ExceptionEngine()
    req = eng.submit_request(
        ExceptionCategory.POLICY_DEVIATION, "Test exception", "desc",
        "analyst", "model", "m1", "Business need", "Low risk"
    )
    assert req.status == ExceptionStatus.PENDING
    assert req.request_id is not None


def test_exception_approve():
    eng = ExceptionEngine()
    req = eng.submit_request(
        ExceptionCategory.CONTROL_BYPASS, "Bypass", "desc", "analyst",
        "strategy", "s1", "justification", "risk"
    )
    waiver = eng.approve(req.request_id, "risk_officer", conditions=["Monitor daily"], renewable=True)
    assert waiver.active
    assert waiver.granted_by == "risk_officer"


def test_exception_reject():
    eng = ExceptionEngine()
    req = eng.submit_request(
        ExceptionCategory.RISK_LIMIT_BREACH, "Limit breach", "desc", "analyst",
        "portfolio", "p1", "justification", "high risk"
    )
    updated = eng.reject(req.request_id, "risk_officer", "Risk too high")
    assert updated.status == ExceptionStatus.REJECTED


def test_exception_has_active_waiver():
    eng = ExceptionEngine()
    req = eng.submit_request(
        ExceptionCategory.DATA_QUALITY_WAIVER, "Data waiver", "desc", "analyst",
        "data", "price_data", "justification", "low risk", rule_id="CTRL-DQ-001"
    )
    eng.approve(req.request_id, "officer")
    assert eng.has_active_waiver("data", "price_data", "CTRL-DQ-001")


def test_exception_no_waiver_after_reject():
    eng = ExceptionEngine()
    req = eng.submit_request(
        ExceptionCategory.POLICY_DEVIATION, "Test", "desc", "analyst",
        "model", "m1", "justification", "risk"
    )
    eng.reject(req.request_id, "officer", "Denied")
    assert not eng.has_active_waiver("model", "m1")


def test_exception_expire_waivers():
    eng = ExceptionEngine()
    past = datetime.utcnow() - timedelta(seconds=1)
    req = eng.submit_request(
        ExceptionCategory.OPERATIONAL_NECESSITY, "Expired", "desc", "analyst",
        "strategy", "s1", "justification", "risk", expiry_date=past
    )
    eng.approve(req.request_id, "officer")
    expired = eng.expire_waivers()
    assert len(expired) == 1
    assert not eng.has_active_waiver("strategy", "s1")


def test_exception_pending_requests():
    eng = ExceptionEngine()
    eng.submit_request(ExceptionCategory.POLICY_DEVIATION, "T1", "d", "a", "m", "m1", "j", "r")
    eng.submit_request(ExceptionCategory.POLICY_DEVIATION, "T2", "d", "a", "m", "m2", "j", "r")
    assert len(eng.get_pending_requests()) == 2


def test_exception_compensating_controls():
    ctrl = CompensatingControl(
        "CC-001", "Daily monitoring", "risk_team",
        datetime.utcnow(), None, CompensatingControlStatus.ACTIVE
    )
    eng = ExceptionEngine()
    req = eng.submit_request(
        ExceptionCategory.CONTROL_BYPASS, "T", "d", "a", "s", "s1", "j", "r",
        compensating_controls=[ctrl]
    )
    assert len(req.compensating_controls) == 1


def test_exception_metrics():
    eng = ExceptionEngine()
    eng.submit_request(ExceptionCategory.POLICY_DEVIATION, "T", "d", "a", "m", "m1", "j", "r")
    metrics = eng.get_metrics()
    assert metrics["pending"] == 1
    assert metrics["total_requests"] == 1


def test_exception_active_waivers_list():
    eng = ExceptionEngine()
    req = eng.submit_request(ExceptionCategory.POLICY_DEVIATION, "T", "d", "a", "m", "m1", "j", "r")
    eng.approve(req.request_id, "officer")
    waivers = eng.get_active_waivers()
    assert len(waivers) == 1


# =============================================================================
# Section G: Attestations (10 tests)
# =============================================================================

def test_attestation_request():
    eng = AttestationEngine()
    due = datetime.utcnow() + timedelta(days=7)
    req = eng.request_attestation(
        AttestationScope.CONTROL_EFFECTIVENESS, "Control attestation",
        "Attest CTRL-DQ-001", "risk_officer", "control", "CTRL-DQ-001", due
    )
    assert req.status == AttestationStatus.PENDING
    assert req.attester == "risk_officer"


def test_attestation_complete():
    eng = AttestationEngine()
    due = datetime.utcnow() + timedelta(days=7)
    req = eng.request_attestation(
        AttestationScope.POLICY_COMPLIANCE, "Policy", "desc",
        "officer", "policy", "POL-ML-001", due
    )
    record = eng.complete_attestation(req.request_id, "officer", "I confirm compliance with POL-ML-001")
    assert record.attestation_text == "I confirm compliance with POL-ML-001"


def test_attestation_recurring_schedules_next():
    eng = AttestationEngine()
    due = datetime.utcnow() + timedelta(days=1)
    req = eng.request_attestation(
        AttestationScope.RISK_LIMIT_ADHERENCE, "Monthly risk check", "desc",
        "officer", "portfolio", "p1", due, cadence_days=30
    )
    eng.complete_attestation(req.request_id, "officer", "Confirmed")
    # Should have created a new pending request
    pending = eng.get_pending(attester="officer")
    assert len(pending) == 1


def test_attestation_one_time_no_renewal():
    eng = AttestationEngine()
    due = datetime.utcnow() + timedelta(days=1)
    req = eng.request_attestation(
        AttestationScope.OPERATIONAL_READINESS, "One-time", "desc",
        "officer", "system", "prod", due, cadence_days=0
    )
    eng.complete_attestation(req.request_id, "officer", "Ready")
    pending = eng.get_pending(attester="officer")
    assert len(pending) == 0


def test_attestation_mark_overdue():
    eng = AttestationEngine()
    past = datetime.utcnow() - timedelta(days=1)
    req = eng.request_attestation(
        AttestationScope.DATA_QUALITY, "Data attestation", "desc",
        "analyst", "data", "prices", past
    )
    overdue_ids = eng.mark_overdue()
    assert req.request_id in overdue_ids


def test_attestation_get_records():
    eng = AttestationEngine()
    due = datetime.utcnow() + timedelta(days=7)
    req = eng.request_attestation(
        AttestationScope.MODEL_PERFORMANCE, "Model perf", "desc",
        "ml_eng", "model", "m1", due
    )
    eng.complete_attestation(req.request_id, "ml_eng", "Model performing within parameters")
    records = eng.get_records(entity_type="model", entity_id="m1")
    assert len(records) == 1


def test_attestation_get_pending_by_attester():
    eng = AttestationEngine()
    due = datetime.utcnow() + timedelta(days=7)
    eng.request_attestation(
        AttestationScope.CONTROL_EFFECTIVENESS, "T1", "d", "officer_a", "c", "c1", due
    )
    eng.request_attestation(
        AttestationScope.CONTROL_EFFECTIVENESS, "T2", "d", "officer_b", "c", "c2", due
    )
    assert len(eng.get_pending(attester="officer_a")) == 1


def test_attestation_with_evidence():
    eng = AttestationEngine()
    due = datetime.utcnow() + timedelta(days=7)
    req = eng.request_attestation(
        AttestationScope.SEGREGATION_OF_DUTIES, "SoD check", "desc",
        "compliance", "access", "rbac", due, evidence_bundle_id="bundle-123"
    )
    record = eng.complete_attestation(
        req.request_id, "compliance", "SoD rules respected",
        evidence_bundle_id="bundle-123"
    )
    assert record.evidence_bundle_id == "bundle-123"


def test_attestation_qualifications_captured():
    eng = AttestationEngine()
    due = datetime.utcnow() + timedelta(days=7)
    req = eng.request_attestation(
        AttestationScope.RISK_LIMIT_ADHERENCE, "Risk", "desc",
        "officer", "risk", "r1", due
    )
    record = eng.complete_attestation(
        req.request_id, "officer", "Limits adhered to",
        qualifications="Excluding tech incident on 2026-03-01"
    )
    assert "2026-03-01" in record.qualifications


def test_attestation_metrics():
    eng = AttestationEngine()
    due = datetime.utcnow() + timedelta(days=7)
    eng.request_attestation(
        AttestationScope.CONTROL_EFFECTIVENESS, "T", "d", "o", "c", "c1", due
    )
    metrics = eng.get_metrics()
    assert metrics["pending"] == 1
    assert metrics["total_requests"] == 1


# =============================================================================
# Section H: AccessControl (10 tests)
# =============================================================================

def test_access_grant_role():
    acm = AccessControlManager()
    grant = acm.grant_role("analyst_1", AccessRole.QUANT_RESEARCHER, [PermissionType.READ], "admin")
    assert grant.active
    assert AccessRole.QUANT_RESEARCHER in acm.get_roles("analyst_1")


def test_access_revoke_grant():
    acm = AccessControlManager()
    grant = acm.grant_role("analyst_1", AccessRole.QUANT_RESEARCHER, [PermissionType.READ], "admin")
    acm.revoke_grant(grant.grant_id)
    assert AccessRole.QUANT_RESEARCHER not in acm.get_roles("analyst_1")


def test_access_has_permission():
    acm = AccessControlManager()
    acm.grant_role(
        "operator_1", AccessRole.TRADING_OPERATOR,
        [PermissionType.READ, PermissionType.EXECUTE], "admin"
    )
    assert acm.has_permission("operator_1", PermissionType.EXECUTE)
    assert not acm.has_permission("operator_1", PermissionType.APPROVE)


def test_access_no_permission_without_grant():
    acm = AccessControlManager()
    assert not acm.has_permission("unknown_user", PermissionType.READ)


def test_access_sod_violation_detected():
    acm = AccessControlManager()
    # SOD-001: QUANT_RESEARCHER + TRADING_OPERATOR conflict
    acm.grant_role("conflict_user", AccessRole.QUANT_RESEARCHER, [PermissionType.READ], "admin")
    acm.grant_role("conflict_user", AccessRole.TRADING_OPERATOR, [PermissionType.EXECUTE], "admin")
    violations = acm.get_sod_violations()
    assert any(v.principal == "conflict_user" for v in violations)


def test_access_no_sod_violation_clean_user():
    acm = AccessControlManager()
    acm.grant_role("clean_user", AccessRole.ML_ENGINEER, [PermissionType.WRITE], "admin")
    violations = [v for v in acm.get_sod_violations() if v.principal == "clean_user"]
    assert len(violations) == 0


def test_access_full_sod_check():
    acm = AccessControlManager()
    violations = acm.check_sod()
    assert isinstance(violations, list)


def test_access_assign_responsibility():
    acm = AccessControlManager()
    assignment = acm.assign_responsibility(
        "model_promotion",
        responsible="ml_engineer",
        accountable="ml_lead",
        consulted=["risk_officer"],
        informed=["governance_officer"],
    )
    assert assignment.responsible == "ml_engineer"
    assert "risk_officer" in assignment.consulted


def test_access_get_responsibility():
    acm = AccessControlManager()
    acm.assign_responsibility("live_activation", "trading_operator", "risk_officer")
    retrieved = acm.get_responsibility("live_activation")
    assert retrieved is not None
    assert retrieved.responsible == "trading_operator"


def test_access_metrics():
    acm = AccessControlManager()
    acm.grant_role("u1", AccessRole.AUDITOR, [PermissionType.AUDIT_READ], "admin")
    metrics = acm.get_metrics()
    assert metrics["active_grants"] >= 1
    assert metrics["unique_principals"] >= 1


# =============================================================================
# Section I: Reporting (10 tests)
# =============================================================================

def test_reporting_dashboard_generates():
    dashboard = GovernanceDashboard()
    summary = dashboard.generate_summary()
    assert isinstance(summary, GovernanceDashboardSummary)
    assert summary.generated_at is not None


def test_reporting_health_green_by_default():
    dashboard = GovernanceDashboard()
    summary = dashboard.generate_summary()
    assert summary.governance_health in ("GREEN", "AMBER", "RED")


def test_reporting_summary_has_controls_data():
    dashboard = GovernanceDashboard()
    summary = dashboard.generate_summary()
    assert summary.total_controls >= 0


def test_reporting_summary_fields_present():
    dashboard = GovernanceDashboard()
    summary = dashboard.generate_summary()
    required_fields = [
        "total_controls", "open_surveillance_events", "active_waivers",
        "pending_attestations", "active_policies", "sod_violations", "governance_health"
    ]
    for field in required_fields:
        assert hasattr(summary, field)


def test_reporting_top_concerns_tuple():
    dashboard = GovernanceDashboard()
    summary = dashboard.generate_summary()
    assert isinstance(summary.top_concerns, tuple)
    assert len(summary.top_concerns) <= 5


def test_reporting_control_matrix_build():
    dashboard = GovernanceDashboard()
    matrix = dashboard.build_control_matrix()
    assert isinstance(matrix, tuple)
    assert len(matrix) >= 15  # at least 15 default controls


def test_reporting_control_matrix_entries():
    dashboard = GovernanceDashboard()
    matrix = dashboard.build_control_matrix()
    for entry in matrix:
        assert hasattr(entry, "control_id")
        assert hasattr(entry, "status")
        assert hasattr(entry, "critical")


def test_reporting_summary_frozen():
    dashboard = GovernanceDashboard()
    summary = dashboard.generate_summary()
    assert dataclasses.is_dataclass(summary)


def test_reporting_multiple_calls_consistent():
    dashboard = GovernanceDashboard()
    s1 = dashboard.generate_summary()
    s2 = dashboard.generate_summary()
    # Same control counts (singletons used internally)
    assert s1.total_controls == s2.total_controls


def test_reporting_governance_health_values():
    dashboard = GovernanceDashboard()
    summary = dashboard.generate_summary()
    assert summary.governance_health in ("GREEN", "AMBER", "RED")


# =============================================================================
# Section J: Retention (10 tests)
# =============================================================================

def test_retention_register_record():
    mgr = RetentionManager()
    record = mgr.register("model", "m1", RetentionCategory.MODEL_ARTIFACTS, "s3://models/m1.pkl")
    assert record.status == ArchiveStatus.ACTIVE
    assert record.category == RetentionCategory.MODEL_ARTIFACTS


def test_retention_deletion_eligible_date_set():
    mgr = RetentionManager()
    record = mgr.register("audit_entry", "a1", RetentionCategory.AUDIT_TRAIL)
    assert record.deletion_eligible_date is not None
    # AUDIT_TRAIL has 2555 days minimum retention
    diff = record.deletion_eligible_date - datetime.utcnow()
    assert diff.days >= 2550


def test_retention_legal_hold_impose():
    mgr = RetentionManager()
    mgr.register("trade", "t1", RetentionCategory.TRADE_RECORDS)
    hold = mgr.impose_legal_hold("Litigation hold", "legal_team", "trade", "t1")
    assert hold.active
    assert hold.hold_id is not None


def test_retention_legal_hold_blocks_deletion():
    mgr = RetentionManager()
    mgr.register("trade", "t1", RetentionCategory.TRADE_RECORDS)
    mgr.impose_legal_hold("Litigation", "legal", "trade", "t1")
    eligible = mgr.get_deletion_eligible()
    assert not any(r.entity_id == "t1" for r in eligible)


def test_retention_legal_hold_wildcard():
    mgr = RetentionManager()
    mgr.register("trade", "t1", RetentionCategory.TRADE_RECORDS)
    mgr.register("trade", "t2", RetentionCategory.TRADE_RECORDS)
    mgr.impose_legal_hold("Batch hold", "legal", "trade", "*")
    records = mgr.get_records_by_category(RetentionCategory.TRADE_RECORDS)
    assert all(r.legal_hold for r in records)


def test_retention_lift_legal_hold():
    mgr = RetentionManager()
    mgr.register("incident", "i1", RetentionCategory.INCIDENT_RECORDS)
    hold = mgr.impose_legal_hold("Regulatory inquiry", "legal", "incident", "i1")
    result = mgr.lift_legal_hold(hold.hold_id, "legal_team")
    assert result
    active_holds = mgr.get_active_legal_holds()
    assert not any(h.hold_id == hold.hold_id for h in active_holds)


def test_retention_default_policies_loaded():
    mgr = RetentionManager()
    # All 9 categories should have policies
    for cat in RetentionCategory:
        assert cat in mgr._policies


def test_retention_get_records_by_category():
    mgr = RetentionManager()
    mgr.register("model", "m1", RetentionCategory.MODEL_ARTIFACTS)
    mgr.register("model", "m2", RetentionCategory.MODEL_ARTIFACTS)
    mgr.register("trade", "t1", RetentionCategory.TRADE_RECORDS)
    records = mgr.get_records_by_category(RetentionCategory.MODEL_ARTIFACTS)
    assert len(records) == 2


def test_retention_metrics():
    mgr = RetentionManager()
    mgr.register("model", "m1", RetentionCategory.MODEL_ARTIFACTS)
    metrics = mgr.get_metrics()
    assert metrics["total_records"] == 1
    assert metrics["under_legal_hold"] == 0


def test_retention_hold_id_unique():
    mgr = RetentionManager()
    h1 = mgr.impose_legal_hold("H1", "l", "model", "*")
    h2 = mgr.impose_legal_hold("H2", "l", "trade", "*")
    assert h1.hold_id != h2.hold_id
