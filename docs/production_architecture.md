# Production Architecture

> **Integration Status: SCAFFOLD**
> Runtime infrastructure (RuntimeStateManager, ControlPlaneEngine, AlertEngine,
> ReconciliationEngine, DeploymentEngine) is implemented and tested (113 tests pass).
> As of 2026-03-31, **no live or paper trading system exists.** `is_safe_to_trade()`
> is never called from any execution path. Two independent kill-switch systems
> are not yet synchronized (P0-KS).
> See: `docs/INTEGRATION_STATUS.md`, `docs/remediation/remediation_ledger.md:P1-SAFE`

## 1. Production Architecture Overview

The pairs trading system is designed for institutional-grade operation: it can be run in research mode, backtested at high fidelity, shadow-traded, and eventually activated in a live-capital environment. The production layer is the set of subsystems that govern how the system behaves once code leaves development and enters an environment where real capital or regulatory consequences exist.

The production architecture spans seven packages that sit above the domain logic:

| Package | Responsibility |
|---------|---------------|
| `runtime/` | Canonical runtime state: what is active, what is halted, what is throttled. |
| `control_plane/` | Operator control surface: the only legal path to changing runtime state. |
| `monitoring/` | Health checks, heartbeat tracking, tradability gate. |
| `alerts/` | Rule-based alert firing, deduplication, suppression, escalation. |
| `reconciliation/` | Position/order integrity checks, leg-imbalance detection, EOD sign-off. |
| `deployment/` | Release lifecycle, stage transitions, freeze windows, rollback governance. |
| `secrets/` | Secret references and loading without ever persisting secret values. |

These packages are intentionally free of domain logic. They do not compute spreads, manage positions, or generate signals. Their job is to provide the operator and automated systems with a safe, auditable, and reversible way to operate the platform.

### Design Invariants

- **Observed state is separate from desired state.** `RuntimeState` is what the system currently is. `DesiredRuntimeState` is what the operator intends. `RuntimeStateManager.diff()` surfaces drift between the two.
- **Deployed is not activated.** A release can sit in `DEPLOYED` or `VERIFIED` indefinitely without going live. Activation is always an explicit, operator-gated step.
- **Every operator action is recorded.** The control plane never fires-and-forgets. Every `ControlPlaneActionRecord` captures the actor, timestamp, action type, parameters, approval references, and audit trail.
- **The kill switch is a hard gate.** When engaged, `is_safe_to_trade()` returns `False` everywhere — in `RuntimeStateManager`, in `SystemHealthMonitor`, and in `ControlPlaneEngine.run_preflight_checks()`. No domain subsystem can bypass this.
- **Secret values never appear in logs or memory.** `SecretLoader` loads and discards; it never stores. Logging uses only `secret_name`, not the value.

---

## 2. Environment Model and Runtime Modes

### Runtime Modes

`RuntimeMode` (defined in `runtime/contracts.py`) represents the operational posture of the system:

| Mode | Description |
|------|-------------|
| `RESEARCH` | Offline analysis only. No orders, no live feeds. |
| `BACKTEST` | Historical replay. No orders, no live feeds. |
| `PAPER` | Live market data. Simulated order execution. |
| `SHADOW` | Live market data. Signals computed but not sent. |
| `STAGING` | Full end-to-end but against a non-production broker account. |
| `LIVE` | Real capital, real orders. |

The mode is set at `RuntimeStateManager` construction time and is embedded in every `RuntimeState` snapshot. Mode escalation (e.g., from PAPER to LIVE) requires an explicit operator action through the control plane.

### Environment Specifications

`EnvironmentSpec` (defined in `runtime/contracts.py`, instantiated in `runtime/environment.py`) encodes the permissions and constraints for each named deployment environment:

```
research  → no broker orders, no live capital, lowest risk ceiling
backtest  → same as research
paper     → broker orders allowed, no live capital, 100k position limit
shadow    → broker access for market data only, signals not routed
staging   → full access but separate broker account
live      → live capital enabled, kill switch auto-engages on breach
```

Key fields on `EnvironmentSpec`:

- `allow_live_capital` — if False, any attempt to fund real trades is blocked at the control plane level.
- `allow_broker_orders` — if False, order routing is disabled system-wide.
- `max_position_size_usd` — position limit enforced by the control plane before every activation.
- `requires_approval_for` — tuple of action types that cannot proceed without a valid approval ID.
- `risk_class_ceiling` — maximum risk class of operations permitted (e.g., `MEDIUM_RISK` for paper, `HIGH_RISK` for live).
- `kill_switch_auto_engage` — if True, the kill switch engages automatically on certain health check failures.

#### Unknown Environment Fallback

`get_environment_spec(env_name)` falls back to the `"research"` spec for any unrecognized name. This ensures that novel or mistyped environment names always result in the most restrictive set of permissions rather than an error or a permissive default.

### Validating Actions Against the Environment

`validate_environment_action(env, action_type, risk_class)` is the policy enforcement point. It checks three independent gates in order:

1. **Approval gate** — is this action type in `requires_approval_for`?
2. **Risk ceiling gate** — does the risk class of the action exceed the environment ceiling?
3. **Human review gate** — does the action require human review above a threshold?

Each gate returns `(allowed: bool, reason: str)`. The control plane calls this before dispatching any action.

---

## 3. Control Plane Design

The control plane (`control_plane/engine.py`) is the only legal interface through which runtime state can be changed by operators or automated systems. No domain module should mutate `RuntimeStateManager` directly in production.

### Entry Points

All mutations go through `ControlPlaneEngine.execute_action(action: ControlPlaneAction)`. Convenience methods (e.g., `enable_strategy()`, `engage_kill_switch()`) are thin wrappers that construct an action and call `execute_action`.

### Action Types

`ControlPlaneActionType` defines 18 distinct actions:

| Action | Effect |
|--------|--------|
| `ENABLE_STRATEGY` | Activates a strategy in `RuntimeStateManager`. |
| `DISABLE_STRATEGY` | Deactivates a strategy. |
| `ENABLE_MODEL` | Activates a model. |
| `DISABLE_MODEL` | Disables a model. |
| `ENABLE_AGENT` | Activates an agent. |
| `DISABLE_AGENT` | Disables an agent. |
| `SET_THROTTLE` | Sets the global sizing throttle to a `ThrottleLevel`. |
| `SET_EXITS_ONLY` | Switches to exits-only mode (new entries blocked). |
| `ENGAGE_KILL_SWITCH` | Engages the kill switch with reason + actor. |
| `RELEASE_KILL_SWITCH` | Releases the kill switch; requires non-empty approval ID. |
| `TRIGGER_DRAIN` | Begins draining positions in a component. |
| `FORCE_RECONCILE` | Triggers an immediate reconciliation cycle. |
| `PROMOTE_MODEL` | Promotes a model to CHAMPION status. |
| `ALTER_RISK_LIMIT` | Modifies a runtime risk limit. |
| `APPLY_OVERRIDE` | Applies a `RuntimeOverride` to the state. |
| `CLEAR_OVERRIDE` | Removes a `RuntimeOverride`. |
| `UPDATE_CONFIG` | Updates a runtime configuration parameter. |
| `TRIGGER_ROLLBACK` | Initiates a deployment rollback. |

### Approval Gate

Emergency actions (`action.is_emergency = True`) bypass the approval requirement but must carry a non-empty `justification`. All other actions subject to environment approval policy (`requires_approval_for`) require a valid `approval_id` in the action payload.

### Audit Trail

Every action produces a `ControlPlaneActionRecord`. The record stores:

- `action` — the original frozen `ControlPlaneAction`
- `executed_at` — ISO-8601 timestamp
- `succeeded` — whether the action completed successfully
- `error` — error message if succeeded is False
- `audit_trail` — list of free-text entries added during dispatch

Action history is accessible via `get_action_history()` and is the canonical audit log for any compliance or incident review.

### Singleton Access

In production, `get_control_plane()` returns the shared engine instance. Tests that need isolation should construct `ControlPlaneEngine(state_manager=...)` directly without going through the singleton.

---

## 4. Activation and Deactivation Workflow

### Strategy Activation

To activate a strategy in production:

1. Call `ControlPlaneEngine.enable_strategy(strategy_id, approved_by, approval_id)`.
2. The engine checks `validate_environment_action(env, "activate_strategy", ...)`.
3. If the environment requires approval (paper, live), `approval_id` must be non-empty.
4. If permitted, `RuntimeStateManager.activate_strategy(strategy_id)` is called.
5. A `ControlPlaneActionRecord` is emitted.

Deactivation follows the same path via `disable_strategy()`. The control plane does not validate whether the strategy ID exists in any registry — that is the responsibility of the domain layer above.

### Model and Agent Activation

Model and agent activation (`ENABLE_MODEL`, `DISABLE_MODEL`, `ENABLE_AGENT`, `DISABLE_AGENT`) follow the same pattern. In the live environment, `enable_agent` requires approval. In paper and lower environments, agents can be enabled without approval.

### `ActivationStatus` Lifecycle

The `ActivationStatus` enum (in `runtime/contracts.py`) represents where a component is in its activation lifecycle:

```
PENDING_REVIEW → APPROVED → ACTIVATING → ACTIVE
                    ↓                        ↓
                REJECTED              DEACTIVATING → INACTIVE
                                           ↓
                                      SUSPENDED / FAILED
```

`RuntimeStateManager` stores `StrategyActivationRecord`, `ModelActivationRecord`, and `AgentActivationRecord` (all mutable) indexed by component ID. These records are updated in-place under the manager's threading lock.

### Exits-Only Mode

`set_exits_only()` moves the system to a mode where new entries are blocked but existing positions can be closed. Internally it sets the global throttle to `ThrottleLevel.EXITS_ONLY`, which maps to a sizing multiplier of `0.0` for new positions. The portfolio layer must check `ThrottleState.multiplier_for(level)` before sizing any entry.

---

## 5. Monitoring and Observability

`SystemHealthMonitor` (`monitoring/health.py`) aggregates health signals from all platform components into a single `ServiceStatusSummary` on each call to `check_all()`.

### Individual Checks

| Check Method | What It Checks |
|---|---|
| `check_runtime_state()` | Kill switch, throttle level, component halt states. |
| `check_kill_switch()` | Reads kill switch state from `portfolio.risk_ops`. |
| `check_overrides()` | Scans active overrides for near-expiry or expired entries. |
| `check_sql_store()` | Lazy-imports `core.sql_store` and tests connectivity. |
| `check_heartbeats()` | Compares last-seen heartbeat times to `heartbeat_gap_s` threshold. |
| `check_broker()` | Given a `BrokerConnectionStatus`, assesses connection quality. |
| `check_data_feeds()` | Given a list of `MarketDataFeedStatus`, flags stale or disconnected feeds. |

### Severity Model

Each check returns a `HealthStatus` with a `CheckSeverity`:

- `OK` — nominal
- `WARNING` — degraded but tradable
- `CRITICAL` — blocking; sets `safe_to_trade = False`
- `UNKNOWN` — check could not be performed; counts as WARNING in aggregate

Conservative aggregation: any single CRITICAL sets `safe_to_trade = False`. UNKNOWN promotes to WARNING in the summary.

### Tradability Gate

`is_safe_to_trade(env)` is the canonical gate for the domain layer to ask whether it is safe to enter new positions. It independently checks:

1. Kill switch state
2. Runtime state (HALTED components)
3. SQL store connectivity
4. Critical heartbeat gaps

Returns `(bool, List[str])`. The second element is the list of blocking reasons, which should be logged or included in incident records.

### Heartbeat Registry

`register_heartbeat(record: HeartbeatRecord)` stores the most recent heartbeat from each named component. Components that fail to heartbeat within `heartbeat_gap_s` seconds are flagged. In live production, every long-running service (data ingestor, signal engine, order router) must emit heartbeats on a defined cadence.

---

## 6. Alerting and Escalation Philosophy

### Alert Engine

`AlertEngine` (`alerts/engine.py`) is a rule-based alerting system. Operators register `AlertRule` objects that define:

- **Trigger conditions** (described in `description` and `severity`)
- **Deduplication key** (`dedup_key`) — prevents alert storms
- **Suppression window** (`suppression_window_minutes`) — after firing, the same dedup key is suppressed for this duration
- **Auto-resolve timeout** (`auto_resolve_minutes`) — if set, firing alerts self-resolve after this interval
- **Escalation** — whether and when to escalate to a human
- **Runbook URL** — link to remediation instructions

### Default Rules

The engine registers 15+ default rules at construction time covering:

- Kill switch engagement/release
- Reconciliation breaks
- Position limit breaches
- Data feed staleness
- Broker disconnection
- Model drift
- Deployment failures
- Heartbeat gaps
- Risk limit violations

### Firing and Deduplication

`fire(rule_id, source, scope, message, context)` emits an `AlertEvent` if the rule is not currently suppressed for that dedup key. The effective dedup key is `{rule.dedup_key}.{scope}`, allowing the same rule to fire independently per pair, per strategy, or globally.

If the same dedup key is fired within the suppression window, `fire()` returns `None`. Callers must not assume an alert was recorded just because they called `fire()`.

### Acknowledgement

`acknowledge(alert_id, acknowledged_by, notes)` moves an alert to `ACKNOWLEDGED` status and returns an `AlertAcknowledgement` record. This does not silence the alert — it records that a human has seen it. Auto-resolution continues on its normal timer.

### Escalation Philosophy

Alerts escalate when they remain unacknowledged past `escalation_delay_minutes`. Escalation is recorded via `EscalationRecord` but the engine does not send notifications directly — that is delegated to an external notification bus. The engine's job is to produce structured records; delivery is an integration concern.

---

## 7. Incident Management

The production architecture does not include a dedicated incident management package. Incidents are surfaced through the alert engine (`AlertFamily.SYSTEM`, `AlertFamily.RISK`) and captured in operator runbooks (see Section 14). Key patterns:

### Incident Classification

Alerts with `AlertSeverity.EMERGENCY` represent the highest severity class. They correspond to:

- Kill switch engagements
- Position limit hard-breaches
- Broker connectivity loss with open positions
- Reconciliation failures that cannot be auto-resolved

### Incident Response Sequence

1. `AlertEngine` fires an EMERGENCY alert.
2. Operator is notified via external channel (Slack, PagerDuty — not in scope here).
3. Operator reviews `SystemHealthMonitor.check_all()` and `ControlPlaneEngine.get_action_history()`.
4. Operator executes remediation via `ControlPlaneEngine` (e.g., `engage_kill_switch`, `trigger_drain`).
5. All control plane actions during the incident are automatically logged.
6. After resolution, operator calls `acknowledge()` on all EMERGENCY alerts.
7. Operator documents the incident in the runbook for retrospective review.

### Post-Incident Review

The EOD report (`ReconciliationEngine.generate_eod_report()`) captures the day's reconciliation state, trade summary, and any outstanding diffs. It should be retained as an artifact for every trading day.

---

## 8. Deployment and Rollout Governance

### Stage Lifecycle

`DeploymentStage` defines the canonical progression of a release:

```
BUILT → TESTED → PACKAGED → APPROVED → DEPLOYED → VERIFIED → ACTIVATED
                                                                    ↓
                                              CANARIED → EXPANDED → ACTIVATED (loop)
                                                    ↓
                                              ROLLED_BACK → RETIRED
```

Every stage transition is gated by `transition_stage(release_id, new_stage, actor, approval_id)`. Transitions that skip stages raise `ValueError`. The engine maintains a set of valid `(from, to)` pairs.

### Deployed ≠ Activated

This is the most important deployment invariant. A release can sit in `DEPLOYED` or `VERIFIED` indefinitely for smoke testing, canary observation, or operator review. Only an explicit `transition_stage(release_id, ACTIVATED, ...)` call — which requires an approval ID in restricted environments — brings the release live.

### Rollout Strategies

`RolloutPlan` supports four strategies:

- `immediate` — deploy to 100% of traffic at once
- `canary` — deploy to `canary_fraction` of traffic, observe for `observation_window_minutes`, then expand
- `staged` — deploy in ordered steps described in `steps` tuple
- `blue_green` — two parallel deployments; switch traffic atomically

The engine does not execute the rollout itself — that is the responsibility of the deployment system (CI/CD, Kubernetes, etc.). The engine governs the lifecycle and approval state.

### Freeze Windows

`set_freeze(env, frozen, actor, reason)` prevents new DEPLOYED or ACTIVATED transitions for the named environment. This is used for:

- End-of-quarter blackout periods
- Major market events (index rebalances, earnings seasons)
- Incident response (preventing further changes during an active incident)

`is_frozen(env)` can be checked before any deployment operation.

### Artifact Tracking

`DeploymentArtifact` (frozen) captures git SHA, checksum, dependency versions, and target environments for every deployable artifact. The `compatible_schema_versions` field ensures database migrations are applied before a code deployment activates.

---

## 9. Rollback and Recovery

### Rollback Decision

`rollback(release_id, reason, actor, rollback_to_version, approval_id, notes)` does two things:

1. Transitions the release to `ROLLED_BACK`.
2. Returns a `RollbackDecision` record with the categorized `RollbackReason`, timestamps, and whether the rollback was automated.

`RollbackReason` values:

| Reason | Trigger |
|--------|---------|
| `HEALTH_CHECK_FAILED` | Automated post-deploy health check failure |
| `PERFORMANCE_REGRESSION` | Signal quality or Sharpe ratio degradation detected |
| `RECONCILIATION_FAILURE` | Unresolvable position mismatch |
| `OPERATOR_REQUEST` | Manual rollback by operator |
| `POLICY_VIOLATION` | Config or model violates governance policy |
| `INCIDENT_TRIGGERED` | Rollback as part of an active incident response |

### Auto-Rollback

When `RolloutPlan.rollback_on_error = True`, the deployment system should call `rollback()` automatically on health check failure. The engine records `automated=True` in the `RollbackDecision`.

### Recovery Path

After rollback:

1. `ROLLED_BACK` releases can be `RETIRED` via `transition_stage(release_id, RETIRED)`.
2. The operator must diagnose the failure and produce a new release (`BUILT` → ... → `APPROVED`).
3. A new rollout plan is created with the remediated artifact.
4. The freeze window (if set) must be cleared before the next deploy attempt.

---

## 10. Reconciliation and EOD Controls

### Reconciliation Engine

`ReconciliationEngine` (`reconciliation/engine.py`) is the system-of-record check between the platform's internal state and the broker's reported state.

### What Gets Reconciled

| Check | Description |
|-------|-------------|
| Position reconciliation | Quantity and side of every symbol vs broker |
| Order reconciliation | Open order status vs broker |
| Leg imbalance check | Detects one leg of a spread open while the other is flat |
| Hedge ratio drift | Detects when the live hedge ratio diverges from the target |

### Diff Types

`DiffType` (10 values) classifies every detected discrepancy:

```
QUANTITY_MISMATCH, SIDE_MISMATCH, MISSING_INTERNAL, MISSING_BROKER,
ORPHAN_POSITION, LEG_IMBALANCE, HEDGE_RATIO_DRIFT, ORDER_STATUS_MISMATCH,
ORDER_FILL_MISSING, STALE_DATA
```

Diffs with `is_critical=True` block live trading via `is_clean()`.

### Reconciliation Tolerance

The default position quantity tolerance is 2% (configurable). Quantities within this tolerance are considered matched. Large mismatches (beyond tolerance) produce `QUANTITY_MISMATCH` diffs.

### EOD Report

`generate_eod_report(date, env, trade_summary, reconciliation_report)` produces an `EndOfDayReport` that summarizes:

- Total trades, open positions, gross/net PnL
- Orders submitted, filled, rejected
- Incident and alert counts
- Reconciliation status for the day
- Model versions active during the session
- Configuration version

This report should be persisted and reviewed before the next trading session opens.

### Alert Integration

If an `alert_engine` is passed to `ReconciliationEngine`, critical reconciliation diffs automatically fire `RECONCILIATION_BREAK` alerts. This wires the reconciliation engine into the alerting pipeline without requiring the caller to explicitly fire alerts.

---

## 11. Runtime Safety Controls and Kill-Switch Philosophy

### Kill Switch Design

The kill switch is a hard, system-wide halt. It is not a soft throttle — it cannot be partially engaged. When the kill switch is active:

- `RuntimeStateManager.is_safe_to_trade()` returns `(False, [...])` unconditionally.
- `ControlPlaneEngine.run_preflight_checks()` returns a report with `kill_switch_clear = False`.
- `SystemHealthMonitor.is_safe_to_trade()` returns `False`.

The kill switch is engaged by `RuntimeStateManager.engage_kill_switch(reason, actor)` or via `ControlPlaneEngine.engage_kill_switch(...)`. It can only be released via `release_kill_switch(approval_id, released_by)`, and `approval_id` must be non-empty. This prevents automated accidental release — release always requires a documented approval.

### Throttle Levels

`ThrottleLevel` represents a sizing multiplier applied to all new positions:

| Level | Multiplier | Meaning |
|-------|-----------|---------|
| `NONE` | 1.0 | Full sizing |
| `LIGHT` | 0.75 | 25% reduction |
| `MODERATE` | 0.50 | 50% reduction |
| `HEAVY` | 0.25 | 75% reduction |
| `EXITS_ONLY` | 0.0 | No new entries |
| `HALTED` | 0.0 | No activity |

The portfolio layer reads `ThrottleState.multiplier_for(level)` before every sizing decision.

### Component-Level Halts

`RuntimeStateManager.update_component_state(component_id, ServiceState.HALTED)` marks an individual component (e.g., `"broker"`, `"data_feed"`) as halted. `is_safe_to_trade()` checks for any HALTED components and returns `False` if any are found.

### Override System

`RuntimeOverride` allows temporary policy changes (e.g., increase a risk limit for a specific strategy for 4 hours). Overrides carry:

- `override_type` — what parameter is being overridden
- `scope` — global or component-specific
- `expires_at` — optional expiry timestamp
- `is_emergency` — if True, approval is not required (justification is)

`expire_stale_overrides()` should be called periodically to clean up expired entries.

---

## 12. Model and Agent Production Control Rules

### Model Activation Requirements

A model may only be activated in production if:

1. It is registered in `MLModelRegistry` with status `CHAMPION` or `CHALLENGER`.
2. `GovernanceEngine.check_promotion_criteria()` has approved the model.
3. In live/staging environments, the activation requires `approval_id`.

`ControlPlaneEngine.disable_model(model_id, ...)` deactivates a model immediately. The portfolio layer's `ModelScorer` handles fallback — it never exposes a missing model as an error to callers.

### Agent Activation Requirements

Agents must be registered in `AgentRegistry` before they can be enabled. The control plane does not validate registry membership — this is enforced at the domain layer. An agent that is not in the registry but is marked active in `RuntimeStateManager` will silently not be dispatched.

### ML Overrides Are Never Permitted to Override Hard Risk Rules

This is an architectural constant, not a runtime parameter:

- Kill switch, drawdown manager HARD mode, and lifecycle FAILED states cannot be overridden by any ML signal or model output.
- `MLUsageContract.may_override_risk_limit` is always `False`.
- If a model predicts MEAN_REVERTING but `break_confidence > 0.80`, the engine returns BROKEN regardless.

### Shadow Mode

In `RuntimeMode.SHADOW`, the system computes signals and size recommendations but does not route orders. This mode is used for:

- New model validation against live data
- Pre-live agent calibration
- Monitoring new strategy performance without capital risk

The control plane enforces shadow mode by checking `allow_broker_orders` on the environment spec.

---

## 13. Security, Secrets, and Access Model

### Secret References

`SecretReference` (frozen, defined in `secrets/contracts.py`) is the only representation of a secret that may be stored in memory, passed between modules, or logged. It contains:

- `secret_name` — logical name (e.g., `"FMP_API_KEY"`)
- `provider` — one of `"env_var"`, `"config_file"`, `"vault"`, `"aws_ssm"`
- `key_path` — provider-specific path
- `scope` — `"read_only"`, `"trading"`, or `"admin"`
- `rotation_due` / `last_rotated` — rotation policy metadata

The actual secret value is never stored in the reference.

### Secret Loading

`SecretLoader.load(ref)` retrieves the value at runtime and returns it to the caller. It:

- Catches all exceptions and returns `None` on failure (never raises)
- Never logs the value
- Never caches the value

Callers are responsible for deciding what to do when `load()` returns `None` (fail fast, use default, raise, etc.).

### Freshness Validation

`validate_freshness(ref)` returns `(bool, reason_str)`:

- `(True, "No rotation policy configured.")` — no rotation due date
- `(True, "Secret 'X' rotation due in N day(s).")` — fresh
- `(False, "Secret 'X' rotation overdue by N day(s).")` — stale

Production startup scripts should check freshness for all trading-scope secrets.

### Custom Providers

For HashiCorp Vault, AWS SSM, or other backends, register a loader function:

```python
SecretLoader.register_provider("vault", lambda key_path: vault_client.read(key_path))
```

The custom loader must follow the same security contract: never log the value, return `None` on failure.

### Access Scopes

- `read_only` — market data API keys, public data endpoints
- `trading` — broker API keys, FMP API key
- `admin` — database credentials, infrastructure access

Scope is metadata only — `SecretLoader` does not enforce scope. Enforcement is at the caller level.

---

## 14. Operator Dashboard and Runbook Model

### Runbook Conventions

Each automated alert rule should have a corresponding runbook entry. The `runbook_url` field on `AlertRule` links the alert to the procedure. Runbooks should follow this structure:

```
## Alert: <RULE_ID>
### When
Triggered when [condition].

### Impact
[What breaks if this is not addressed.]

### Immediate Action
1. Check SystemHealthMonitor.check_all()
2. Review ControlPlaneEngine.get_action_history() for recent changes
3. Execute [specific control plane action]

### Escalation
If not resolved in N minutes, engage kill switch and page on-call.

### Post-Incident
File an EOD report supplement. Review deployment history for recent changes.
```

### Operator Checklist: Pre-Market Open

1. Run `SystemHealthMonitor.check_all(env="live")` — all components OK.
2. Verify `ReconciliationEngine.is_clean()` for prior day positions.
3. Confirm no freeze windows active via `DeploymentEngine.is_frozen("live")`.
4. Review `AlertEngine.get_active_alerts()` — acknowledge or resolve any carry-over alerts.
5. Confirm desired runtime state matches active state via `RuntimeStateManager.diff()`.
6. Verify all trading-scope secrets are fresh via `SecretLoader.validate_freshness()`.

### Operator Checklist: End of Day

1. Call `ReconciliationEngine.generate_eod_report()` and persist the result.
2. Review `AlertEngine.get_active_alerts()` and close any resolved incidents.
3. Run `RuntimeStateManager.expire_stale_overrides()`.
4. Confirm no unresolved critical diffs via `ReconciliationEngine.is_clean()`.
5. Archive `ControlPlaneEngine.get_action_history()` for the session.

### Dashboard Integration Points

The production layer exposes these surfaces to a monitoring dashboard:

| Surface | Method | Update Frequency |
|---------|--------|-----------------|
| System health | `SystemHealthMonitor.check_all()` | Every 30s |
| Active alerts | `AlertEngine.get_active_alerts()` | Every 30s |
| Runtime state | `RuntimeStateManager.snapshot()` | Every 10s |
| Action history | `ControlPlaneEngine.get_action_history()` | On change |
| Reconciliation status | `ReconciliationEngine.is_clean()` | Every 60s |
| Health metrics | `SystemHealthMonitor.get_metrics()` | Every 60s |

---

## 15. Known Limitations and Roadmap

### Current Limitations

**No external notification delivery.** The alert engine produces structured `AlertEvent` and `EscalationRecord` objects, but does not send messages to Slack, PagerDuty, or email. Integration with an external bus is required for live operations.

**Reconciliation tolerance is static.** The 2% quantity mismatch tolerance is a construction-time constant. Dynamic tolerance (e.g., larger tolerance for illiquid names) is not yet supported.

**Secret provider plugins are in-memory only.** `SecretLoader.register_provider()` stores providers in a class-level dict that is reset on process restart. Production deployments with Vault or AWS SSM should register providers at application startup in a deterministic initialization sequence.

**Freeze windows are not persisted.** `DeploymentEngine.set_freeze()` stores freeze state in memory. Process restarts clear all freeze windows. For production, freeze windows should be backed by a durable store (the `sql_store` or a dedicated config table).

**Control plane does not validate approval IDs against an external approval system.** The `approval_id` field is a string that callers provide. The engine checks that it is non-empty but does not verify it against an approval workflow system. Integration with an approval system (GitHub PR, Jira ticket, internal approval workflow) is a deployment-layer concern.

**SystemHealthMonitor.check_sql_store() triggers duckdb_engine import.** On Python 3.13 with certain numpy builds, importing `duckdb_engine` causes a fatal segfault. The test suite patches this method. In production, this is not an issue because duckdb_engine is loaded at process startup long before the health check runs.

### Roadmap

- **Persistent freeze windows** — store freeze state in `sql_store` for durability across restarts.
- **Webhook/bus integration for alerts** — pluggable delivery backends for `AlertEngine`.
- **Approval workflow integration** — validate `approval_id` against an external system before control plane dispatch.
- **Dynamic reconciliation tolerances** — per-symbol tolerance configuration.
- **Audit log persistence** — write `ControlPlaneActionRecord` to `sql_store` for regulatory retention.
- **Multi-environment control plane** — single operator interface for managing paper and live environments simultaneously.
- **Automated canary health evaluation** — `DeploymentEngine` automatically evaluating canary metrics and calling `rollback()` without operator intervention.
