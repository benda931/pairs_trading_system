# Production Audit — Pairs Trading System
## "Wide, Controlled, Defensible, Production-Grade"

**Audit Date:** 2026-04-04
**Auditor:** Claude Opus (automated deep scan)
**System:** 363 Python files, ~170K lines, 50 agents, 12 quant engines
**Tests:** 976 passing
**Verdict:** YELLOW — Conditional Launch (3 critical, 4 high issues)

---

## DELIVERABLE 1: Brutally Honest Architectural Review

### What's Strong

| Area | Grade | Evidence |
|------|-------|---------|
| Type safety | A | 97.5% of 315 core functions have return type hints |
| Agent extensibility | A | 50 agents, clean BaseAgent protocol, audit logging |
| Quant engine depth | A | 12 engines with real math (Johansen, GARCH MLE, OU stopping) |
| Portfolio constraints | A- | 4-layer risk model, kill-switch, drawdown deleverage |
| Signal pipeline | B+ | Regime → Threshold → Quality → Intent chain, ML hook |
| Feedback loop | B+ | Agent outputs → 9 action types → auto-execute with alerts |
| Scheduler/automation | B | APScheduler daemon, 4 recurring jobs, Telegram alerts |

### What's Broken

| Area | Grade | Evidence |
|------|-------|---------|
| Streamlit in core/ | F | 50+ st.session_state accesses in core/, breaks CLI/test/API |
| Exception handling | D | 149 bare `except Exception: pass` in core/ |
| Config mutation | D | 12 files write config.json outside config_manager |
| Test coverage | D+ | 68% of core modules have ZERO tests |
| Allocation idempotency | D | Double-call = double capital allocated |
| Dataclass validation | C- | 85% of 190 dataclasses lack `__post_init__` |
| Layer boundaries | C | 16 cross-layer import violations |

### Architecture Reality

```
DESIGNED:                          ACTUAL:
┌────────┐                        ┌────────┐
│  root/  │ ← only UI here       │  root/  │ ← UI + some logic
├────────┤                        ├────────┤
│ agents/ │ ← autonomous          │ agents/ │ ← mostly scaffold
├────────┤                        ├────────┤
│  core/  │ ← pure logic          │  core/  │ ← imports streamlit!
├────────┤                        │         │ ← imports agents!
│portfolio│ ← risk mgmt           │         │ ← imports root/!
├────────┤                        ├────────┤
│   ml/   │ ← ML platform         │   ml/   │ ← complete, unused
├────────┤                        ├────────┤
│ common/ │ ← utilities           │ common/ │ ← clean ✓
└────────┘                        └────────┘
```

---

## DELIVERABLE 2: Weak Points Causing Fragility

### CRITICAL (Production Blockers)

**C1. Streamlit Hardcoded in Core Logic**
- `core/app_context.py`: 30+ `st.session_state` accesses
- `core/alert_bus.py`: 10+ `st.session_state` accesses
- `core/ml_analysis.py`: 5+ `st.session_state` accesses
- **Impact:** Core cannot run outside Streamlit (breaks CLI, tests, API server)
- **Fix:** Introduce `StateProvider` protocol, inject at construction

**C2. Portfolio Allocation Not Idempotent**
- `run_portfolio_allocation_cycle()` has no dedup check
- Called twice → allocates 2× capital
- No `last_allocation_ts` guard
- **Impact:** Scheduler restart mid-pipeline doubles positions
- **Fix:** Add `allocation_batch_id` + dedup check

**C3. Config.json Written by 12 Files**
- Only `common/config_manager.py` should write
- 11 other files write directly → race conditions
- **Impact:** Config corruption, inconsistent state between reads
- **Fix:** Route all writes through config_manager API

### HIGH (Pre-Production Required)

**H1. 149 Silent Exception Handlers**
- Pattern: `except Exception: pass` or `except Exception: logger.debug(...)`
- No error counters, no telemetry, no alerting
- **Impact:** Bugs hide for weeks, impossible to diagnose production issues
- **Fix:** Add `@log_exception` decorator, error counters per module

**H2. 48 Core Modules Without Tests**
- Critical untested: fair_value_engine, risk_analytics, leverage_engine, garch_engine, monte_carlo, correlation_monitor, spread_analytics
- **Impact:** Regression risk on every change
- **Fix:** Add contract tests for each engine (input→output shape validation)

**H3. No Circuit Breaker on Agent Feedback**
- Feedback loop can fire KILL_SWITCH, DELEVERAGE, FORCE_EXIT
- No cool-down window between actions
- No max-actions-per-day limit
- **Impact:** Cascading actions in volatile market
- **Fix:** Add ActionThrottler with cool-down per action type

**H4. No Data Staleness Guard**
- Pipeline runs signals on whatever data is cached
- No check: "is this data from today?"
- **Impact:** Trading on stale prices = wrong signals
- **Fix:** Add freshness check: reject data older than market close

### MEDIUM (Post-Production)

**M1.** 16 cross-layer import violations (core/ → root/, core/ → agents/)
**M2.** 85% of dataclasses lack runtime validation
**M3.** No structured error telemetry (error rates, latency tracking)
**M4.** Agent payloads still partially empty for some agents
**M5.** ML platform (ml/) fully built but inference not in production path

---

## DELIVERABLE 3: Target Architecture

### Principle: Preserve Breadth, Add Control

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 6: UI / REPORTING                                     │
│  root/dashboard.py + 18 modules + Streamlit tabs            │
│  RULE: Only layer that imports streamlit                     │
├─────────────────────────────────────────────────────────────┤
│  LAYER 5: ORCHESTRATION / AUTOMATION                         │
│  orchestrator.py, scheduler_daemon, run_daily_pipeline       │
│  RULE: Calls layers below via interfaces, never direct       │
├─────────────────────────────────────────────────────────────┤
│  LAYER 4: AGENTS / FEEDBACK                                  │
│  50 agents + agent_feedback.py + ActionThrottler            │
│  RULE: Agents return recommendations, never mutate directly  │
│  RULE: Feedback loop has cool-down, max-actions, audit trail │
├─────────────────────────────────────────────────────────────┤
│  LAYER 3: QUANT ENGINES                                      │
│  12 engines (analytics, risk, MC, GARCH, cycle, exit...)    │
│  RULE: Pure functions, no side effects, no state             │
│  RULE: Each declares: inputs, outputs, failure mode          │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2: SIGNAL + PORTFOLIO + RISK                          │
│  signal_pipeline, portfolio/, leverage_engine                │
│  RULE: Idempotent operations with batch_id                   │
│  RULE: Hard risk vetoes NEVER overridden                     │
├─────────────────────────────────────────────────────────────┤
│  LAYER 1: DATA + PERSISTENCE + CONFIG                        │
│  data_loader, sql_store, config_manager                      │
│  RULE: Single writer per resource                            │
│  RULE: Freshness validation on every read                    │
├─────────────────────────────────────────────────────────────┤
│  LAYER 0: CONTRACTS + PROTOCOLS                              │
│  core/contracts.py — THE source of truth                     │
│  RULE: All layers import from here, never reverse            │
└─────────────────────────────────────────────────────────────┘
```

### New Components Needed

| Component | Purpose | Layer |
|-----------|---------|-------|
| `StateProvider` protocol | Abstract st.session_state | L0 (contracts) |
| `ActionThrottler` | Cool-down + max-actions for feedback | L4 |
| `AllocationBatchGuard` | Idempotency for portfolio allocation | L2 |
| `DataFreshnessGuard` | Reject stale price data | L1 |
| `ErrorTelemetry` | Count errors per module, alert on spikes | L5 |
| `ModuleContract` decorator | Declare inputs/outputs/failure mode | L0 |

---

## DELIVERABLE 4: Concrete Refactors by Subsystem

### 4.1 Core State Management (CRITICAL)

**Current:** `core/app_context.py` directly uses `st.session_state`
**Target:** Abstract `StateProvider` protocol

```python
# core/contracts.py — add to existing
class StateProvider(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...
    def has(self, key: str) -> bool: ...

class InMemoryStateProvider:
    """For CLI, tests, API server."""
    def __init__(self):
        self._store: Dict[str, Any] = {}
    def get(self, key, default=None): return self._store.get(key, default)
    def set(self, key, value): self._store[key] = value
    def has(self, key): return key in self._store

class StreamlitStateProvider:
    """For dashboard only."""
    def get(self, key, default=None): return st.session_state.get(key, default)
    def set(self, key, value): st.session_state[key] = value
    def has(self, key): return key in st.session_state
```

### 4.2 Portfolio Allocation (CRITICAL)

**Current:** No dedup, can double-allocate
**Target:** Batch-guarded allocation

```python
# In run_portfolio_allocation_cycle:
batch_id = f"alloc_{datetime.now().strftime('%Y%m%d')}"
if self._last_allocation_batch == batch_id:
    logger.warning("Allocation already ran for batch %s — skipping", batch_id)
    return None
# ... run allocation ...
self._last_allocation_batch = batch_id
```

### 4.3 Agent Feedback (HIGH)

**Current:** No throttle on actions
**Target:** ActionThrottler with cool-down

```python
class ActionThrottler:
    COOL_DOWN = {
        "KILL_SWITCH": 3600,      # 1 hour between kill-switch actions
        "DELEVERAGE": 1800,       # 30 min between deleverage
        "FORCE_EXIT": 300,        # 5 min between exits per pair
        "BLOCK_ENTRY": 600,       # 10 min between blocks
        "RETRAIN_MODEL": 86400,   # 1 per day
        "OPTIMIZE_PARAMS": 86400, # 1 per day
        "UPDATE_CONFIG": 3600,    # 1 per hour
    }
    MAX_ACTIONS_PER_CYCLE = 10    # Hard cap
```

### 4.4 Data Layer (HIGH)

**Current:** No freshness check
**Target:** Guard on every price read

```python
def load_price_data_guarded(symbol, max_staleness_hours=18):
    data = load_price_data(symbol)
    if data.empty:
        raise DataFreshnessError(f"No data for {symbol}")
    last_date = data.index[-1]
    age_hours = (datetime.now() - last_date).total_seconds() / 3600
    if age_hours > max_staleness_hours:
        raise DataFreshnessError(f"{symbol} data is {age_hours:.0f}h old")
    return data
```

### 4.5 Error Handling (HIGH)

**Current:** `except Exception: pass`
**Target:** Structured error logging

```python
# Replace 149 instances of:
except Exception:
    pass

# With:
except Exception as exc:
    _error_counter[module_name] += 1
    logger.warning("%s: %s (count=%d)", module_name, exc, _error_counter[module_name])
    if _error_counter[module_name] > ERROR_THRESHOLD:
        alert_system(module_name, "ERROR_RATE", f"{_error_counter[module_name]} errors")
```

---

## DELIVERABLE 5: Contract Definitions for Core Modules

### Engine Contract Template

Every quant engine MUST declare:

```python
@dataclass(frozen=True)
class EngineContract:
    name: str
    purpose: str
    inputs: Dict[str, str]       # param_name → type description
    outputs: Dict[str, str]      # field_name → type description
    failure_mode: str            # What happens on error
    fallback: str                # Degraded behavior
    owner: str                   # Team/module responsible
    consumers: List[str]         # Who uses this output
    economic_value: str          # Alpha / risk / governance / research
    classification: str          # "PRODUCTION" / "RESEARCH" / "EXPERIMENTAL"
```

### Concrete Contracts

| Engine | Classification | Economic Value | Failure Mode |
|--------|---------------|----------------|--------------|
| spread_analytics | PRODUCTION | Alpha discovery (cointegration validation) | Return quality=F, never crash |
| risk_analytics | PRODUCTION | Risk reduction (VaR/CVaR/DD) | Return neutral risk, alert |
| signal_pipeline | PRODUCTION | Alpha generation (entry/exit signals) | Return FLAT, no positions |
| leverage_engine | PRODUCTION | Risk sizing (Kelly, vol-target) | Return leverage=0.5 (conservative) |
| garch_engine | PRODUCTION | Vol forecasting → sizing quality | Fallback to EWMA → realized vol |
| monte_carlo | RESEARCH | Strategy validation (DSR, confidence) | Skip, log warning |
| cycle_detector | RESEARCH | Lookback optimization | Skip, use default lookback=60 |
| optimal_exit | RESEARCH | Exit timing improvement | Skip, use static exit_z |
| factor_attribution | REPORTING | Performance explanation | Skip, no alpha impact |
| correlation_monitor | PRODUCTION | Pair health monitoring | Alert on failure, don't block |
| universe_scanner | RESEARCH | New pair discovery | Skip, keep existing universe |
| execution_algos | EXPERIMENTAL | Execution quality (not connected to IBKR) | N/A until IBKR wired |

---

## DELIVERABLE 6: Governance Design

### Agent Governance Tiers

| Tier | Who | Can Do | Approval | Cool-Down |
|------|-----|--------|----------|-----------|
| **TIER 0: Read-Only** | 30 agents | Analyze, report, recommend | None | None |
| **TIER 1: Alert** | 8 agents | Emit alerts, publish to bus | None | 5 min |
| **TIER 2: Threshold** | 4 agents | Adjust entry_z, size multiplier | Auto (bounded) | 30 min |
| **TIER 3: Position** | 3 agents | Block entry, force exit, deleverage | Auto + audit | 1 hour |
| **TIER 4: System** | 2 agents | Kill-switch, halt pipeline | Auto + alert + audit | 4 hours |
| **TIER 5: Config** | 3 agents | Retrain model, update config, optimize | Auto + validate + rollback | 24 hours |

### Self-Modifying Component Rules

Any component that changes system behavior MUST:

1. **Log before**: What's the current state?
2. **Validate change**: Is it within bounds?
3. **Apply change**: With rollback capability
4. **Verify after**: Did the change achieve intent?
5. **Alert**: Notify via Telegram + bus
6. **Cool-down**: Wait before next change
7. **Hard vetoes**: Never override kill-switch, drawdown limits, max leverage

### ML Governance

- ML predictions NEVER override risk limits
- `fallback_triggered=True` → system uses deterministic rules
- Model promotion requires: IC≥0.05, AUC≥0.55, Brier≤0.25
- Champion/Challenger pattern with shadow period
- Auto-demote on PSI drift > 0.25

---

## DELIVERABLE 7: Production-Hardening Roadmap

### Phase 1: CRITICAL FIXES (Block Production)

| # | Fix | Files | Risk | Effort |
|---|-----|-------|------|--------|
| 1.1 | StateProvider protocol (decouple Streamlit) | contracts.py, app_context.py, alert_bus.py | HIGH→LOW | 4h |
| 1.2 | Allocation idempotency guard | orchestrator.py | HIGH→LOW | 1h |
| 1.3 | Config write centralization | 12 files | MEDIUM→LOW | 3h |

### Phase 2: HIGH PRIORITY (Pre-Production)

| # | Fix | Files | Risk | Effort |
|---|-----|-------|------|--------|
| 2.1 | ActionThrottler for feedback loop | agent_feedback.py | HIGH→LOW | 2h |
| 2.2 | Data freshness guard | data_loader.py | HIGH→LOW | 1h |
| 2.3 | Replace 149 silent catches with telemetry | 15 core files | MEDIUM→LOW | 6h |
| 2.4 | Add contract tests for 12 engines | tests/ | MEDIUM→LOW | 4h |

### Phase 3: MEDIUM (Post-Launch)

| # | Fix | Files | Risk | Effort |
|---|-----|-------|------|--------|
| 3.1 | Fix 16 cross-layer imports | core/ | LOW | 3h |
| 3.2 | Add `__post_init__` to 160 dataclasses | core/ | LOW | 4h |
| 3.3 | Engine contract declarations | 12 engines | LOW | 2h |
| 3.4 | Integration test: signal→allocation→feedback | tests/ | LOW | 3h |

### Phase 4: EXCELLENCE (Ongoing)

| # | Fix | Impact |
|---|-----|--------|
| 4.1 | Wire IBKR paper trading | Real execution |
| 4.2 | Promote ML models through governance | ML alpha |
| 4.3 | Dashboard error boundary per tab | UI resilience |
| 4.4 | Automated regression detection | Quality gate |

---

## SUMMARY

**The system is WIDE and AMBITIOUS. Good.**

**The system lacks DISCIPLINE and SAFETY. Fixable.**

The 3 critical fixes (StateProvider, idempotency, config centralization) can be done in one session. The 4 high-priority fixes in another. After that, this is a production-grade platform.

**Preserve every engine. Preserve every agent. Add the safety rails.**
