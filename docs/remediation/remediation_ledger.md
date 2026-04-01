# Remediation Ledger — Pairs Trading System
**Version:** 2.0
**Date:** 2026-04-01
**Truth Audit:** Integration reconciliation (2026-04-01)
**Status:** Active

## Status Definitions (v2.0 — honest taxonomy)
- **COMPLETE** = Code runs by default in operational paths, no flag required
- **WIRED (opt-in)** = Code path exists and is tested; requires explicit enablement
- **AVAILABLE** = Function/method defined and tested; never invoked from operational code
- **IN_PROGRESS** = Partially addressed
- **PLANNED** = Not started
- **DEFERRED** = Intentionally postponed
- **DOWNGRADED** = Scaffold; tracked but not blocking

## P0 Findings (Trust-Breaking)

| ID | Title | Severity | Status | Evidence |
|----|-------|----------|--------|----------|
| P0-WF | Calendar WF honestly labeled | P0 | COMPLETE | default=63, floor=max(63,...), docstring warns "NOT true walk-forward" |
| P0-EXEC | bar_lag execution delay | P0 | COMPLETE | default bar_lag=1 (next-bar), pending_action queue |
| P0-KS | Kill-switch bridge | P0 | WIRED (opt-in) | Factory exists but never called from operational code |
| P0-DOCS | Documentation overstatement | P0 | IN_PROGRESS | This v2.0 ledger corrects prior overstatements |

## P1 Findings

| ID | Title | Severity | Status | Evidence |
|----|-------|----------|--------|----------|
| P1-PIPE | Signal pipeline to backtester | P1 | COMPLETE | use_signal_pipeline defaults to True; SignalPipeline.evaluate_bar() is the default entry/exit path |
| P1-PORTINT | Signal-to-portfolio bridge | P1 | AVAILABLE | Defined, tested, called from dashboard only |
| P1-MINOBS | Minimum observation count | P1 | COMPLETE | MIN_OBS=252 in contracts.py |
| P1-SAFE | Runtime safety gating | P1 | AVAILABLE | safety_check callback exists; no caller injects it |
| P1-SURV | Survivorship bias docs | P1 | COMPLETE | Explicit inline comment in universe.py |
| P1-ML | Meta-label ML overlay | P1 | AVAILABLE | Training script tested; no model wired operationally |
| P1-AGENTS | Agent dispatch | P1 | AVAILABLE | 2 agents wired in pipeline; pipeline never called |
| P1-GOV | Governance gate | P1 | WIRED (opt-in) | Called in promote(); non-CRITICAL falls through |
| P1-AUDIT | Audit chains | P1 | DOWNGRADED | Scaffold only |
| P1-SURV2 | Stale data surveillance | P1 | COMPLETE | detect("SURV-DI-001") in load_price_data() |

## P2+ Findings

| ID | Title | Severity | Status |
|----|-------|----------|--------|
| P2-COSTS | Flat cost model | P2 | DEFERRED |
| P2-DUPRANK | Duplicate pair ranking | P2 | PLANNED |
| P2-DUPTHROT | Duplicate throttle | P2 | PLANNED |
| P2-COINT | Coint stability soft-scoring | P2 | PLANNED |
| P2-MLT | ML training script | P2 | COMPLETE |
| P3-SIGMIG | signals_engine.py (2700L) | P3 | PLANNED |
| P3-PARTA | Partial fills not modeled | P3 | DEFERRED |
| P4-BACKUP | Backup files removed | P4 | COMPLETE |

## Residual Risk Register

| ID | Risk | Severity |
|----|------|----------|
| RR-001 | Walk-forward Sharpe subject to overfitting | HIGH |
| RR-002 | No live/paper trading system | HIGH |
| RR-003 | ML models never trained in production | HIGH |
| RR-004 | Safety checks available but not enforced | MEDIUM |
| RR-005 | Audit chains empty | MEDIUM |
| RR-006 | Signal pipeline default=True but backtests may differ from legacy path | LOW |
| RR-007 | Orchestrator pipeline never called | MEDIUM |
