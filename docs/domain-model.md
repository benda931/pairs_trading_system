# Domain Model Handbook — Pairs Trading System

This handbook defines every major canonical object in the platform. These definitions
are authoritative — when in doubt, this document and `core/contracts.py` are the sources of truth.

> **Rule:** All domain types live in their layer's canonical contracts file.
> Never define domain types in UI, utility, or ad-hoc modules.

---

## Layer Map

```
research/discovery_contracts.py   Research-phase types (instruments, candidates, universes)
core/contracts.py                 Platform-wide types (pairs, spreads, signals, regimes, risk)
core/intents.py                   Signal-layer intent types (entry, exit, hold, suspend, retire)
core/diagnostics.py               Diagnostic/audit artifacts (signal, regime, lifecycle diagnostics)
portfolio/contracts.py            Portfolio-layer types (ranking, sizing, allocation, exposure)
ml/contracts.py                   ML-layer types (features, labels, models, inference, governance)
```

---

## Research Layer Objects

### InstrumentMetadata
**Module:** `research/discovery_contracts.py:127` | **Lifecycle:** Created during universe construction

| Field | Type | Purpose |
|-------|------|---------|
| `symbol` | str | Ticker symbol |
| `sector`, `industry` | str | Classification |
| `avg_dollar_volume` | float | Liquidity measure |
| `history_days` | int | Available history length |
| `missing_data_pct` | float | Data quality measure |
| `is_eligible` | bool | Passes EligibilityFilter |
| `ineligibility_reasons` | list | Why rejected (explicit) |

**Anti-pattern:** Never create instruments without checking eligibility.

### UniverseSnapshot
**Module:** `research/discovery_contracts.py:281` | **Lifecycle:** Versioned, immutable once created

Captures a point-in-time view of eligible instruments. Must respect `train_end`.

### CandidatePair
**Module:** `research/discovery_contracts.py:342` | **Lifecycle:** Created during discovery, fed to validation

| Field | Type | Purpose |
|-------|------|---------|
| `pair_id` | PairId | Canonical pair identifier |
| `discovery_family` | DiscoveryFamily | How it was found (CORRELATION, DISTANCE, CLUSTER, etc.) |
| `discovery_score` | float | Raw discovery score |
| `rejection_reasons` | list | Empty if passed, populated if failed |

**Key doctrine:** High correlation = candidate. Cointegration + stability = tradable.

---

## Core Domain Objects

### PairId
**Module:** `core/contracts.py:272` | **Frozen dataclass**

Stores symbols in **lexicographic order** (sym_x < sym_y). The constructor auto-orders.
Use `PairId("MSFT", "AAPL")` — it becomes `PairId(sym_x="AAPL", sym_y="MSFT")`.

### SpreadDefinition
**Module:** `core/contracts.py:387` | **Lifecycle:** Fitted during research, used in signal generation

| Field | Type | Purpose |
|-------|------|---------|
| `pair_id` | PairId | Which pair |
| `model` | SpreadModel | OLS, ROLLING_OLS, KALMAN, etc. |
| `hedge_ratio` | float | Beta for spread computation |
| `half_life_days` | float | Mean-reversion speed |
| `mean`, `std` | float | Spread distribution parameters |

**Critical:** `compute_spread(lx, ly)` returns **raw spread** (not z-scored).
Use `compute_zscore(lx, ly)` for the normalized z-score.

### RegimeState
**Module:** `core/contracts.py:447`

| Field | Type | Purpose |
|-------|------|---------|
| `label` | RegimeLabel | MEAN_REVERTING, TRENDING, HIGH_VOL, CRISIS, BROKEN, UNKNOWN |
| `confidence` | float | Classification confidence [0, 1] |
| `classified_at` | datetime | When classification was made |

**Safety floor:** ML hooks cannot override BROKEN or CRISIS if `break_confidence > 0.80`.

### TradeSignal
**Module:** `core/contracts.py:468`

| Field | Type | Purpose |
|-------|------|---------|
| `direction` | SignalDirection | LONG_SPREAD, SHORT_SPREAD, FLAT, EXIT |
| `z_score` | float | Current z-score |
| `confidence` | float | Signal conviction [0, 1] |
| `regime` | RegimeLabel | Current market regime |

---

## Intent Objects (Signal Layer Output)

> **Doctrine:** Intents are proposals, never automatic executors. The portfolio layer
> must check `block_reasons`, apply risk limits, and approve before execution.

### EntryIntent
**Module:** `core/intents.py:158`

| Field | Type | Purpose |
|-------|------|---------|
| `pair_id` | PairId | Which pair |
| `direction` | SignalDirection | Long or short spread |
| `z_score` | float | Current z-score at signal time |
| `confidence` | float | Signal conviction [0, 1] |
| `entry_mode` | EntryMode | THRESHOLD_CROSS, CONFIRMED, STAGED, etc. |
| `block_reasons` | list[BlockReason] | Why this should NOT be acted on |

### ExitIntent
**Module:** `core/intents.py:270`

| Field | Type | Purpose |
|-------|------|---------|
| `exit_reasons` | list[ExitReason] | Why exiting (always explicit, never empty) |
| `exit_mode` | ExitMode | IMMEDIATE, PATIENT, TWAP, AGGRESSIVE |
| `is_stop` | property | True if any reason is a stop-loss type |

### SignalDecision
**Module:** `core/intents.py:352` | **Lifecycle:** Output of SignalPipeline

Wraps an intent with regime, quality, lifecycle context. This is what the portfolio layer consumes.

| Field | Type | Purpose |
|-------|------|---------|
| `intent` | BaseIntent | The proposed action |
| `regime_label` | RegimeLabel | Regime at decision time |
| `quality_grade` | SignalQualityGrade | A+ through F |
| `lifecycle_state` | TradeLifecycleState | Current lifecycle position |

---

## Portfolio Layer Objects

### RankedOpportunity
**Module:** `portfolio/contracts.py:278` | **Lifecycle:** Created by OpportunityRanker

7-dimension composite scoring:

| Dimension | Field |
|-----------|-------|
| Signal strength | `signal_strength_score` |
| Signal quality | `signal_quality_score` |
| Regime suitability | `regime_suitability_score` |
| Reversion probability | `reversion_probability` |
| Diversification value | `diversification_value` |
| Stability | `stability_score` |
| Capacity | `capacity_score` |

### AllocationDecision
**Module:** `portfolio/contracts.py:599` | **Lifecycle:** Output of PortfolioAllocator.run_cycle()

| Field | Type | Purpose |
|-------|------|---------|
| `pair_id` | PairId | Which pair |
| `outcome` | AllocationOutcome | FUNDED, REJECTED, DEFERRED |
| `rationale` | AllocationRationale | Why this decision was made |
| `sizing` | SizingDecision | Notional, weight, risk contribution |

### DrawdownState
**Module:** `portfolio/contracts.py:94`

6-level heat state machine: NORMAL -> CAUTIOUS -> THROTTLED -> DEFENSIVE -> RECOVERY_ONLY -> HALTED

### KillSwitchState
**Module:** `portfolio/contracts.py:62`

4 modes: OFF -> SOFT -> REDUCE -> HARD. Escalation-only automatic; recovery requires explicit reset.

---

## ML Layer Objects

> **Integration Status: SCAFFOLD.** Zero models trained. All inference returns neutral 0.5.

### FeatureDefinition
**Module:** `ml/contracts.py:165` | **Frozen**

61 features defined across 6 entity scopes. Each has `lookback_days`, `required_inputs`, `version`.

### ModelMetadata
**Module:** `ml/contracts.py:693`

Tracks model identity, lineage, version, and promotion status (RESEARCH -> CANDIDATE -> CHALLENGER -> CHAMPION -> RETIRED).

### InferenceResult
**Module:** `ml/contracts.py:659`

| Field | Type | Purpose |
|-------|------|---------|
| `score` | float | Model output (default: 0.5 neutral) |
| `fallback_triggered` | bool | True if no model was available |
| `model_id` | str | Which model produced the score |

**Critical:** Always check `fallback_triggered` before acting on the result.

---

## Lifecycle State Machine

**Module:** `core/lifecycle.py` | **13 states, 13+ triggers**

```
NOT_ELIGIBLE ──> WATCHLIST ──> SETUP_FORMING ──> ENTRY_READY ──> PENDING_ENTRY ──> ACTIVE
                    ^                                                                |
                    |                                                                v
                    +──── COOLDOWN <──── PENDING_EXIT <──── EXIT_READY <──── REDUCING
                                                                              ^
                                                                     SCALING_IN

                    Any state ──> SUSPENDED ──> WATCHLIST (auto-resume 30d) or RETIRED
                    Any state ──> RETIRED (terminal)
```

**State timeouts:** SETUP_FORMING (10d), ENTRY_READY (3d), PENDING_ENTRY (1d), SUSPENDED (30d auto-resume).

**Invalid transitions raise `ValueError`.** Always check `sm.can_enter()`, `sm.can_add()`,
`sm.is_position_active()` before calling `transition()`.

---

## Diagnostic / Audit Objects

### SignalAuditRecord
**Module:** `core/diagnostics.py:403`

Full audit trail for one signal decision: pair, spread state, regime, quality grade, intent, block reasons, timestamp.

### PortfolioAuditRecord
**Module:** `portfolio/contracts.py:920`

Full audit trail for one allocation cycle: intents received, rankings, allocations made, constraints checked, capital state.

---

## Deprecated Types

| Old Type | Location | Replacement | Status |
|----------|----------|-------------|--------|
| `ExitReason` (10 values) | `root/backtest.py` | `core/contracts.py:ExitReason` (19 values) | **Removed** (MIG-001) |
| `TradeSide` (SHORT_LONG/LONG_SHORT) | `root/backtest.py` | `core/contracts.py:SignalDirection` | **Deprecated** (MIG-002) |
| `TradeSide` (LONG/SHORT) | `root/trade_logic.py` | `core/contracts.py:SignalDirection` | **Deprecated** (MIG-002) |

See `docs/migration/migration_ledger.md` for sunset conditions.
