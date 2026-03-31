# Governance Architecture

**Document version:** 1.0
**Last reviewed:** 2026-03-31
**Package root:** `pairs_trading_system/` (10 governance packages)

> **Integration Status: SCAFFOLD**
> Governance infrastructure is implemented and tested (100 tests pass). As of 2026-03-31,
> **no operational decision is gated by any governance check.** Audit chains are empty.
> SurveillanceEngine.detect() is never called from operational code.
> The only partial integration: GovernanceEngine is referenced in ML registry promote()
> but is non-blocking on ImportError.
> See: `docs/INTEGRATION_STATUS.md`, `docs/remediation/remediation_ledger.md:P1-GOV`

---

## 1. Overview

The governance layer is a cross-cutting compliance, audit, and risk-oversight platform
that wraps all trading-system activity. It is not part of the signal or portfolio pipelines;
it observes, records, and can block — but it never trades.

### Purpose

| Goal | Mechanism |
|------|-----------|
| Tamper-evident audit trail | Hash-linked AuditChain entries |
| Evidence gating for promotions | EvidenceBundleBuilder with typed requirements |
| Operational control health | ControlRegistry with test history and status |
| Versioned policy enforcement | PolicyRegistry with conformance reports |
| Real-time anomaly detection | SurveillanceEngine with 12 default rules |
| Formal exception process | ExceptionEngine with waiver lifecycle |
| Recurring accountability | AttestationEngine with cadence scheduling |
| Role-based access control | AccessControlManager with SoD enforcement |
| Unified health dashboard | GovernanceDashboard (read-only aggregator) |
| Data lifecycle governance | RetentionManager with legal hold support |

### Package Map

```
audit/          — AuditChain, AuditChainRegistry
evidence/       — EvidenceBundleBuilder, EvidenceItem, EvidenceRequirement
controls/       — ControlRegistry (15 defaults), ControlTestRecord
policies/       — PolicyRegistry (4 defaults, 8 rules), PolicyConformanceReport
surveillance/   — SurveillanceEngine (12 default rules), SurveillanceEvent, SurveillanceCase
exceptions_mgmt/— ExceptionEngine, ExceptionRequest, ExceptionWaiver, CompensatingControl
attestations/   — AttestationEngine, AttestationRequest, AttestationRecord
operating_model/— AccessControlManager, SodRule, SodViolation, ResponsibilityAssignment
reporting/      — GovernanceDashboard, GovernanceDashboardSummary, ControlMatrixEntry
retention/      — RetentionManager, RetentionPolicy, ArchiveRecord, LegalHold
```

### Design Principles

1. **Immutability first.** Every domain object is a frozen dataclass. Mutation
   happens by constructing a replacement object with updated fields.

2. **Singleton engines, fresh instances for tests.** Each package exposes a
   `get_*()` module-level accessor for production use. Tests instantiate
   classes directly to avoid shared state.

3. **Lazy imports in the dashboard.** `GovernanceDashboard` imports all
   subsystems inside `try/except` blocks, making it importable even when
   individual packages are unavailable.

4. **No circular dependencies.** Governance packages do not import from
   `core/`, `portfolio/`, `agents/`, or `ml/`. They receive typed inputs via
   method parameters.

5. **Never raises in aggregate queries.** Dashboard and metrics methods catch
   all exceptions and return safe zero-defaults rather than propagating.

---

## 2. Audit Chain

### Purpose

An append-only, hash-linked audit log for tracking every significant action
taken on a domain entity. Designed to detect tampering and enable forensic
reconstruction of entity history.

### Key Types

| Type | Role |
|------|------|
| `AuditChainEntry` | Frozen dataclass; one per action |
| `AuditChain` | Ordered list of entries for one logical chain |
| `AuditChainRegistry` | Thread-safe factory and registry of chains |
| `AuditChainValidationReport` | Result of `validate()` walk |
| `AuditChainStatus` | `VALID`, `MISSING_LINKS`, `HASH_MISMATCH`, `TAMPERED`, `UNKNOWN` |

### Hash Linking

Every `AuditChainEntry` stores:
- `payload_hash` — SHA-256 of the JSON-serialised payload dict
- `prev_entry_id` — UUID of the previous entry (None for the first)
- `chain_id` — logical chain identifier

`AuditChain.validate()` walks entries checking that `entry[i].prev_entry_id ==
entry[i-1].entry_id`. A broken link sets `status = MISSING_LINKS`.

### Typical Usage

```python
from audit.chain import AuditChain, AuditChainRegistry

# Per-entity chain (fresh instance for testing; use get_audit_chain_registry() in prod)
chain = AuditChain("strategy-abc")
e1 = chain.append("ml_team", "create", "strategy", "abc", {"version": "v1"})
e2 = chain.append("ml_team", "update", "strategy", "abc", {"version": "v2"})

report = chain.validate()
assert report.status.value == "VALID"

# Reconstruct all changes to entity "abc"
history = chain.reconstruct_entity_history("abc")
```

### Registry Pattern

`AuditChainRegistry.get_or_create(chain_id)` is idempotent — calling it twice
with the same ID returns the same `AuditChain` object (thread-safe). The
convenience method `registry.log(chain_id, actor, action, entity_type,
entity_id, payload)` combines get-or-create and append into a single call.

### Invariants

- Chain IDs are arbitrary strings — typically `"{entity_type}-{entity_id}"`.
- `metadata` is stored as `tuple[tuple[str, Any], ...]` for frozen compatibility.
  Convert back with `dict(entry.metadata)`.
- `AuditChain` is NOT internally thread-safe for writes; use `AuditChainRegistry`
  which holds a lock around chain creation.

---

## 3. Evidence Management

### Purpose

Structured collection and completeness checking of evidence items for review
gates (model promotion, compliance audit, incident postmortem).

### Key Types

| Type | Role |
|------|------|
| `EvidenceItem` | Frozen; one collected piece of evidence |
| `EvidenceRequirement` | Declared requirement for a bundle |
| `EvidenceBundle` | Frozen snapshot of items + requirements |
| `EvidenceBundleBuilder` | Mutable builder; call `build()` to freeze |
| `EvidenceCompletenessReport` | Output of `check_completeness()` |
| `EvidenceType` | 16 values: `BACKTEST_RESULT`, `AUDIT_TRAIL`, `MODEL_CARD`, etc. |
| `EvidenceStatus` | `PENDING`, `COLLECTED`, `VERIFIED`, `REJECTED`, `EXPIRED`, `SUPERSEDED` |

### Builder Pattern

```python
builder = EvidenceBundleBuilder(
    entity_type="model", entity_id="regime_v3",
    purpose="promotion_review", created_by="ml_team"
)

# Declare requirements
req = builder.add_requirement(
    "Walk-forward required",
    required_types=[EvidenceType.WALK_FORWARD_RESULT],
    min_count=1, mandatory=True
)

# Collect evidence
builder.collect(
    EvidenceType.WALK_FORWARD_RESULT, "WF 2024-Q1", "IC=0.07",
    collected_by="ml_team", artifact_ref="s3://wf_v3.json"
)

report = builder.check_completeness()
assert report.is_complete  # True after requirement is satisfied
bundle = builder.build()   # Frozen EvidenceBundle
```

### Completeness Rules

`is_complete = True` only when:
1. Zero mandatory requirements are unsatisfied.
2. Zero collected items are expired.

Non-mandatory requirements appear in `missing_requirements` but do not set
`is_complete = False`.

`completeness_pct` is `verified_items / max(total_items, 1) * 100`.

### Expiry

Items with `expiry_date < now` are added to `expired_items` in the report and
are excluded from satisfying requirements. Items do not auto-change their
`EvidenceStatus` — callers must decide whether to recollect.

---

## 4. Controls Registry

### Purpose

Maintains 15 canonical system controls, records test history, and tracks
operational status for each control.

### Default Controls (15)

| ID | Domain | Name | Critical |
|----|--------|------|---------|
| CTRL-DQ-001 | DATA_QUALITY | Price Data Freshness Check | Yes |
| CTRL-DQ-002 | DATA_QUALITY | Spread Computation Integrity | Yes |
| CTRL-MR-001 | MODEL_RISK | Model Drift Monitoring | Yes |
| CTRL-MR-002 | MODEL_RISK | ML Leakage Prevention | Yes |
| CTRL-MR-003 | MODEL_RISK | Champion Promotion Governance | Yes |
| CTRL-ER-001 | EXECUTION_RISK | Kill Switch Circuit Breaker | Yes |
| CTRL-ER-002 | EXECUTION_RISK | Position Reconciliation | Yes |
| CTRL-ER-003 | EXECUTION_RISK | Hedge Ratio Drift Alert | Yes |
| CTRL-MKT-001 | MARKET_RISK | Regime Classification Guardrail | Yes |
| CTRL-MKT-002 | MARKET_RISK | Portfolio Drawdown Limits | Yes |
| CTRL-OPS-001 | OPERATIONAL_RISK | Deployment Freeze Enforcement | Yes |
| CTRL-OPS-002 | OPERATIONAL_RISK | Secrets Never Logged | Yes |
| CTRL-GOV-001 | GOVERNANCE | Dual-Control for Live Activation | Yes |
| CTRL-SOD-001 | SEGREGATION_OF_DUTIES | Signal-to-Execution Segregation | Yes |
| CTRL-HO-001 | HUMAN_OVERSIGHT | Human Review for P0 Incidents | Yes |

### Status Lifecycle

```
DESIGNED → ACTIVE → DEGRADED → FAILED → ACTIVE (on recovery)
                            ↓
                         RETIRED
```

Status transitions are automatic on `record_test()`:
- `PASS` → `ACTIVE` (if previously FAILED or DEGRADED)
- `PARTIAL` → `DEGRADED`
- `FAIL` → `FAILED`

### Critical Failures

`get_critical_failures()` returns controls marked `critical=True` that are
FAILED or DEGRADED. The governance dashboard reports this as `critical_failures`.
Any non-zero count moves governance health to RED.

### Ownership

`assign_owner(control_id, owner, assigned_by, review_cadence_days)` sets a
`ControlOwnerRecord`. Ownership is optional metadata — it does not affect
status transitions.

---

## 5. Policy Registry

### Purpose

Maintains versioned policy documents. Only ACTIVE versions are evaluated.
Every `evaluate()` call produces a `PolicyConformanceReport` with per-rule
`PolicyEvaluationResult` objects.

### Default Policies (4 policies, 8 rules)

| Policy ID | Policy Name | Rule Count |
|-----------|-------------|-----------|
| POL-ML-001 | ML Model Governance | 2 |
| POL-TRADE-001 | Trading Activity Controls | 2 |
| POL-DEPL-001 | Deployment Controls | 2 |
| POL-DATA-001 | Data Quality Standards | 2 |

### Rule Types and Enforcement

| `PolicyRuleType` | Blocking? | Description |
|-----------------|-----------|-------------|
| `HARD_LIMIT` | Yes | Blocks the action |
| `PROHIBITED` | Yes | Categorically disallows |
| `APPROVAL_REQUIRED` | No | Requires approval before proceeding |
| `AUDIT_REQUIRED` | No | Must produce an audit record |
| `NOTIFY_REQUIRED` | No | Must notify named parties |
| `SOFT_LIMIT` | No | Advisory; logs warning |
| `MANDATORY` | No | Omission is the violation |

`is_conformant = True` iff `blocking_violations == 0`.

### Evaluation

```python
from policies.registry import PolicyRegistry

reg = PolicyRegistry()

# Default: all rules pass unless explicitly overridden
report = reg.evaluate("model", "regime_v3", {})
assert report.is_conformant

# Signal a specific rule violation
report = reg.evaluate("model", "regime_v3", {
    "rule_POL-ML-001-R1_passed": False  # Champion promotion check failed
})
assert not report.is_conformant
assert report.blocking_violations == 1
```

### Versioning

`register_policy(policy_id, rules, authored_by, change_summary)` increments
the version number and transitions the previous ACTIVE version to SUPERSEDED.
`get_policy_history(policy_id)` returns all versions (oldest first).

`suspend_policy(policy_id, reason)` transitions the active version to SUSPENDED
so `evaluate()` skips it without removing the version history.

---

## 6. Surveillance Engine

### Purpose

Automated market and operational anomaly detection. Rules are evaluated against
submitted metric values. Threshold breaches become `SurveillanceEvent` objects.
Related events are grouped into `SurveillanceCase` objects for human review.

### Default Rules (12)

| Rule ID | Family | Threshold | Auto-Escalate |
|---------|--------|-----------|--------------|
| SURV-SA-001 | SPREAD_ANOMALY | z-score >= 4.0 | No |
| SURV-SA-002 | SPREAD_ANOMALY | vol ratio >= 3.0 | Yes |
| SURV-RB-001 | REGIME_BREACH | any CRISIS regime | Yes |
| SURV-RB-002 | REGIME_BREACH | any BROKEN entry | Yes |
| SURV-EA-001 | EXECUTION_ANOMALY | slippage >= 10bps | No |
| SURV-EA-002 | EXECUTION_ANOMALY | order rate >= 10x | Yes |
| SURV-MD-001 | MODEL_DEGRADATION | IC < 0.02 (lower is bad) | No |
| SURV-MD-002 | MODEL_DEGRADATION | PSI >= 0.25 | No |
| SURV-CB-001 | CONCENTRATION_BREACH | sector >= 40% | No |
| SURV-DI-001 | DATA_INTEGRITY | data age >= 48h | Yes |
| SURV-AB-001 | AGENT_BEHAVIOR | error rate >= 20% | No |
| SURV-RL-001 | RISK_LIMIT_APPROACH | drawdown >= 80% of limit | No |

### Detection Logic

```python
eng = SurveillanceEngine()

# Returns None if threshold not breached
event = eng.detect("SURV-SA-001", "strategy", "s1", metric_value=2.0)
assert event is None

# Returns SurveillanceEvent if breached
event = eng.detect("SURV-SA-001", "strategy", "s1", metric_value=5.0)
assert event.status.value == "OPEN"
```

**Special case — lower is bad:** Rules in `lower_is_bad_rules` (currently
`{"SURV-MD-001"}`) trigger when `metric_value < threshold` instead of `>=`.

### Auto-Escalation

Rules with `auto_escalate=True` automatically create a `SurveillanceCase` and
link the triggering event via `case_id`. This is used for critical families
(SPREAD_ANOMALY vol explosion, REGIME_BREACH, EXECUTION_ANOMALY runaway,
DATA_INTEGRITY).

### Event Lifecycle

```
OPEN → ACKNOWLEDGED → UNDER_INVESTIGATION → RESOLVED | FALSE_POSITIVE
```

`acknowledge_event(event_id, acknowledged_by)` removes the event from
`get_open_events()` results. `resolve_event(event_id)` sets `resolved_at`.

### Case Management

Cases can be created manually via `create_case(title, family, event_ids, opened_by)`
or automatically via auto-escalation. `close_case(case_id, closed_by, disposition,
findings)` transitions to `CLOSED_CLEAN` (disposition="no_action") or
`CLOSED_ACTION_TAKEN`.

---

## 7. Exception Management

### Purpose

Formal exception request lifecycle: submit → review → approve/reject →
monitor → expire. Approved exceptions issue `ExceptionWaiver` objects that
can be checked before routing around a control or policy rule.

### Exception Categories

`POLICY_DEVIATION`, `RISK_LIMIT_BREACH`, `CONTROL_BYPASS`, `DATA_QUALITY_WAIVER`,
`MODEL_DEPLOYMENT`, `OPERATIONAL_NECESSITY`.

### Lifecycle

```
PENDING → UNDER_REVIEW → APPROVED → (active waiver)
                       ↓
                    REJECTED
                    WITHDRAWN
```

Waivers have an optional `expiry_date`. `expire_waivers()` must be called
periodically (e.g., daily) to deactivate expired waivers. `has_active_waiver()`
checks for an unexpired approved waiver for a given entity and optional rule ID.

### Compensating Controls

When submitting a bypass exception, callers can attach `CompensatingControl`
objects describing mitigating measures. These are stored on the `ExceptionRequest`
and visible to reviewers.

```python
from exceptions_mgmt.engine import ExceptionEngine, ExceptionCategory, CompensatingControl, CompensatingControlStatus
from datetime import datetime

eng = ExceptionEngine()
ctrl = CompensatingControl(
    "CC-001", "Daily P&L monitoring", "risk_team",
    datetime.utcnow(), None, CompensatingControlStatus.ACTIVE
)
req = eng.submit_request(
    ExceptionCategory.CONTROL_BYPASS, "Bypass CTRL-DQ-001 for restatement",
    "Price data restatement in progress", "quant_team",
    "data", "price_data", "Manual verification in place", "LOW",
    rule_id="CTRL-DQ-001", compensating_controls=[ctrl]
)
waiver = eng.approve(req.request_id, "risk_officer", conditions=["Monitor hourly"])
assert eng.has_active_waiver("data", "price_data", "CTRL-DQ-001")
```

---

## 8. Attestations

### Purpose

Schedule, collect, and store formal attestations from named parties.
Recurring attestations (cadence_days > 0) auto-schedule the next request on
completion.

### Key Types

| Type | Role |
|------|------|
| `AttestationRequest` | Pending or overdue attestation task |
| `AttestationRecord` | Completed, immutable attestation record |
| `AttestationScope` | 7 values: `CONTROL_EFFECTIVENESS`, `POLICY_COMPLIANCE`, etc. |
| `AttestationStatus` | `PENDING`, `COMPLETED`, `OVERDUE`, `WAIVED`, `WITHDRAWN` |

### Recurring Attestations

```python
from attestations.engine import AttestationEngine, AttestationScope
from datetime import datetime, timedelta

eng = AttestationEngine()
due = datetime.utcnow() + timedelta(days=1)

req = eng.request_attestation(
    AttestationScope.CONTROL_EFFECTIVENESS,
    "Monthly CTRL-DQ-001 attestation", "Attest price freshness control",
    attester="data_lead", entity_type="control", entity_id="CTRL-DQ-001",
    due_date=due, cadence_days=30
)

record = eng.complete_attestation(req.request_id, "data_lead", "Control operating as designed")
# A new PENDING request for 30 days hence is now in the engine
assert len(eng.get_pending(attester="data_lead")) == 1
```

### Overdue Detection

`mark_overdue()` transitions all PENDING requests past `due_date` to OVERDUE.
This must be called externally (e.g., by a daily scheduled task or the
monitoring layer). The method returns a list of request IDs marked overdue.

### Evidence Linking

`request_attestation(..., evidence_bundle_id="bundle-uuid")` links an evidence
bundle to the request. Callers can override this on completion via
`complete_attestation(..., evidence_bundle_id="new-bundle")`. The link is
preserved in the `AttestationRecord`.

---

## 9. Operating Model (RBAC + SoD)

### Purpose

Role-based access control with Segregation-of-Duties enforcement and RACI
responsibility assignment.

### Roles

`QUANT_RESEARCHER`, `ML_ENGINEER`, `RISK_OFFICER`, `TRADING_OPERATOR`,
`GOVERNANCE_OFFICER`, `DEVOPS_ENGINEER`, `INCIDENT_MANAGER`, `SYSTEM_ADMIN`,
`LIVE_TRADING_APPROVER`, `AUDITOR`, `COMPLIANCE_OFFICER`.

### SoD Rules (5 defaults)

| Rule ID | Conflicting Roles | Severity |
|---------|-------------------|---------|
| SOD-001 | QUANT_RESEARCHER + TRADING_OPERATOR | CRITICAL |
| SOD-002 | ML_ENGINEER + GOVERNANCE_OFFICER | WARNING |
| SOD-003 | RISK_OFFICER + TRADING_OPERATOR | WARNING |
| SOD-004 | AUDITOR + SYSTEM_ADMIN | CRITICAL |
| SOD-005 | GOVERNANCE_OFFICER + LIVE_TRADING_APPROVER | WARNING |

SoD is checked immediately on every `grant_role()` call. Violations are
recorded in `_violations`. `check_sod()` runs a full sweep across all
principals and resets the violations list.

### RACI Assignments

```python
acm = AccessControlManager()
assignment = acm.assign_responsibility(
    process="live_activation",
    responsible="trading_operator",
    accountable="risk_officer",
    consulted=["governance_officer", "compliance_officer"],
    informed=["cto"],
)
retrieved = acm.get_responsibility("live_activation")
```

### Permission Checking

```python
acm.grant_role("u1", AccessRole.TRADING_OPERATOR,
               [PermissionType.READ, PermissionType.EXECUTE], "admin")
assert acm.has_permission("u1", PermissionType.EXECUTE)
assert not acm.has_permission("u1", PermissionType.APPROVE)
```

Resource scope matching uses prefix logic when `resource_scope` ends with `*`.

---

## 10. Reporting Dashboard

### Purpose

Read-only aggregated health view across all governance subsystems. Never
mutates subsystem state.

### GovernanceDashboardSummary Fields

| Category | Fields |
|----------|--------|
| Controls | `total_controls`, `active_controls`, `failed_controls`, `degraded_controls`, `critical_failures` |
| Surveillance | `open_surveillance_events`, `critical_surveillance_events`, `open_surveillance_cases` |
| Exceptions | `active_waivers`, `pending_exception_requests` |
| Attestations | `pending_attestations`, `overdue_attestations` |
| Policies | `active_policies`, `total_policy_rules`, `policy_violations_24h` |
| Audit | `total_audit_entries`, `audit_chains` |
| SoD | `sod_violations`, `critical_sod_violations` |
| Overall | `governance_health`, `top_concerns`, `control_matrix` |

### Health Determination Logic

| Condition | Health |
|-----------|--------|
| `critical_failures > 0` OR `critical_surveillance_events > 0` OR `critical_sod_violations > 0` | RED |
| `failed_controls > 0` OR `degraded_controls > 0` OR `overdue_attestations > 0` OR `open_surveillance_events > 2` | AMBER |
| None of the above | GREEN |

### Usage

```python
from reporting.dashboard import GovernanceDashboard

dashboard = GovernanceDashboard()
summary = dashboard.generate_summary()

print(f"Health: {summary.governance_health}")
print(f"Critical failures: {summary.critical_failures}")

matrix = dashboard.build_control_matrix()
for entry in matrix:
    print(f"{entry.control_id}: {entry.status} (critical={entry.critical})")
```

`generate_summary()` populates `control_matrix=()` (empty). Use
`generate_full_summary()` if you need the matrix embedded in the summary
object, or call `build_control_matrix()` separately.

### Resilience

All subsystem calls inside `generate_summary()` are wrapped in `try/except
Exception`. If any subsystem fails to import or raises, the corresponding
counters default to zero. The dashboard never propagates subsystem exceptions.

---

## 11. Data Retention

### Purpose

Tracks artifacts under retention policies, manages legal holds, and identifies
records eligible for deletion.

### Default Retention Policies (9)

| Category | Min Retention | Regulatory Basis |
|----------|-------------|-----------------|
| AUDIT_TRAIL | 2555 days (7yr) | SEC Rule 17a-4 |
| TRADE_RECORDS | 2555 days (7yr) | FINRA Rule 4511 |
| MODEL_ARTIFACTS | 1825 days (5yr) | Internal Policy |
| RESEARCH_DATA | 1095 days (3yr) | Internal Policy |
| OPERATIONAL_LOGS | 365 days (1yr) | Internal Policy |
| INCIDENT_RECORDS | 1825 days (5yr) | Internal Policy |
| COMPLIANCE_RECORDS | 2555 days (7yr) | SEC Rule 17a-4 |
| ATTESTATIONS | 1825 days (5yr) | Internal Policy |
| EXCEPTION_RECORDS | 2555 days (7yr) | Internal Policy |

### Archive Record Lifecycle

```
ACTIVE → ARCHIVED → PENDING_DELETION → DELETED
                  ↑
           UNDER_LEGAL_HOLD (can be imposed at any point)
```

`register(entity_type, entity_id, category)` creates an `ArchiveRecord` with
`deletion_eligible_date = now + min_retention_days`.

`get_deletion_eligible()` returns records where:
- `status == ACTIVE`
- `legal_hold == False`
- `deletion_eligible_date <= now`

### Legal Holds

`impose_legal_hold(reason, imposed_by, entity_type, entity_id_pattern)` applies
a hold to all matching records. Patterns:
- `"trade_001"` — exact match
- `"trade_*"` — prefix wildcard
- `"*"` — all records of that `entity_type`

`lift_legal_hold(hold_id, lifted_by)` deactivates the hold object but does NOT
automatically restore record statuses. Records remain at `UNDER_LEGAL_HOLD`
status until the caller decides to transition them.

---

## 12. Governance Lifecycle

### Research Proposal to Postmortem

The full lifecycle for a new strategy or model moves through these governance checkpoints:

```
1. Research Phase
   ├── AuditChain: log candidate generation, validation, spread fitting
   ├── EvidenceBundle: collect backtest, walk-forward, governance check
   └── PolicyRegistry.evaluate(): verify POL-ML-001 compliance

2. Promotion Gate
   ├── EvidenceCompletenessReport.is_complete must be True
   ├── ControlRegistry: CTRL-MR-003 (Champion Governance) must be ACTIVE
   └── GovernanceEngine.check_promotion_criteria() must pass

3. Deployment
   ├── AuditChain: log deployment stage transitions
   ├── PolicyRegistry.evaluate(): verify POL-DEPL-001 (dual approval)
   ├── AttestationEngine.request_attestation(): schedule operational readiness
   └── RetentionManager.register(): register model artifact

4. Live Operation
   ├── SurveillanceEngine.detect(): continuous monitoring
   ├── ExceptionEngine.has_active_waiver(): gate control bypasses
   ├── AttestationEngine.mark_overdue(): daily overdue check
   └── GovernanceDashboard.generate_summary(): periodic health check

5. Incident / Postmortem
   ├── SurveillanceEngine.create_case(): group events
   ├── AuditChain: log all incident actions
   ├── EvidenceBundle: collect evidence for postmortem
   ├── RetentionManager: INCIDENT_RECORDS registered for 5-year retention
   └── AttestationEngine.request_attestation(): post-incident review attestation
```

### Promotion Evidence Checklist

Before calling `GovernanceEngine.check_promotion_criteria()`, callers should
build an evidence bundle with at least:

| EvidenceType | Mandatory | Notes |
|-------------|-----------|-------|
| BACKTEST_RESULT | Yes | Full in-sample backtest |
| WALK_FORWARD_RESULT | Yes | Purged K-fold, IC >= 0.05 |
| GOVERNANCE_CHECK | Yes | GovernanceEngine approval record |
| AUDIT_TRAIL | Yes | Chain ID linking to training run |
| PERFORMANCE_REPORT | No (advisory) | Out-of-sample summary |
| PEER_REVIEW | No (advisory) | Optional second reviewer |

---

## 13. Design Principles

### Immutability

All domain objects are frozen dataclasses. Fields that require a collection
(tags, metadata, related_risks) use `tuple` rather than `list`. This ensures
dataclass hashing and safe sharing across threads.

Mutation is always modelled as "replace the dict entry with a new object".
For example, to update event status in `SurveillanceEngine`:

```python
self._events[event_id] = SurveillanceEvent(
    event_id=e.event_id,
    ...
    status=SurveillanceEventStatus.ACKNOWLEDGED,
    acknowledged_by=acknowledged_by,
    ...
)
```

### Singleton Engines

Each package exposes a `get_*()` accessor that caches a module-level
singleton. This is the correct path for production code.

For tests, instantiate directly:

```python
# Production
from surveillance.engine import get_surveillance_engine
eng = get_surveillance_engine()

# Test (fresh state, no shared singleton)
from surveillance.engine import SurveillanceEngine
eng = SurveillanceEngine()
```

### Lazy Imports

`GovernanceDashboard` uses lazy imports inside `try/except` for every subsystem.
This pattern ensures:
1. The dashboard module is always importable.
2. Missing or partially-installed packages degrade gracefully to zero metrics.
3. No circular import risk at module load time.

Other governance packages do NOT use lazy imports — they have hard dependencies
within the governance layer only.

### No Circular Dependencies

Dependency graph (one-way):

```
reporting/dashboard.py
    → controls/registry.py
    → surveillance/engine.py
    → exceptions_mgmt/engine.py
    → attestations/engine.py
    → policies/registry.py
    → audit/chain.py
    → operating_model/access.py

# None of the above import from core/, portfolio/, agents/, or ml/
```

The governance layer is a leaf layer. Core domain modules import from it
(e.g., to log an audit entry), but the governance packages never import from
core.

### Thread Safety

- `AuditChainRegistry` uses a `threading.Lock` around chain creation.
- `SurveillanceEngine` uses a `threading.Lock` around event creation.
- `ExceptionEngine` and `AttestationEngine` use locks around request mutation.
- `ControlRegistry`, `PolicyRegistry`, `AccessControlManager`, and
  `RetentionManager` are not internally synchronised — use external locking
  for concurrent writes.

### Aggregate Metrics Pattern

All engines expose `get_metrics() -> dict` returning plain `dict[str, int]`.
This is the recommended interface for dashboard and monitoring integrations —
prefer `get_metrics()` over reading internal state directly.
