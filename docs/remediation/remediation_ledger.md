# Remediation Ledger — Pairs Trading System
**Version:** 2.1
**Date:** 2026-04-01
**Truth Audit:** Integration reconciliation + fixes (2026-04-01)

## Status Definitions
- **COMPLETE** = Code runs by default in operational paths
- **WIRED (opt-in)** = Code exists, tested, requires explicit enablement
- **AVAILABLE** = Defined and tested; no operational caller yet
- **IN_PROGRESS** = Partially addressed
- **PLANNED** = Not started
- **DEFERRED** = Intentionally postponed
- **DOWNGRADED** = Scaffold; tracked but not blocking

## P0 Findings

| ID | Title | Severity | Status | Evidence |
|----|-------|----------|--------|----------|
| P0-WF | Calendar WF honestly labeled | P0 | COMPLETE | default=63, floor=max(63,...), docstring warns "NOT true walk-forward" |
| P0-EXEC | bar_lag execution delay | P0 | COMPLETE | default bar_lag=1 (next-bar), pending_action queue |
| P0-KS | Kill-switch with control-plane bridge | P0 | WIRED (opt-in) | Factory and callback exist. Orchestrator uses make_kill_switch_manager_with_control_plane() in portfolio cycle. |
| P0-DOCS | Documentation truthfulness | P0 | COMPLETE | Ledger v2.1 reconciled. INTEGRATION_STATUS.md rewritten. All architecture docs have truthfulness banners. |

## P1 Findings

| ID | Title | Severity | Status | Evidence |
|----|-------|----------|--------|----------|
| P1-PIPE | Signal pipeline default in backtester | P1 | COMPLETE | use_signal_pipeline defaults to **True**. Backtester uses evaluate_bar() by default. |
| P1-PORTINT | Portfolio bridge in orchestrator | P1 | COMPLETE | orchestrator.run_portfolio_allocation_cycle() calls bridge_signals_to_allocator(). Dashboard also calls it. |
| P1-MINOBS | Minimum observation count | P1 | COMPLETE | MIN_OBS=252 |
| P1-SAFE | Runtime safety gating | P1 | COMPLETE | Orchestrator injects is_safe_to_trade via safety_check callback. Dashboard uses None (research mode). |
| P1-SURV | Survivorship bias docs | P1 | COMPLETE | Explicit inline comment in universe.py |
| P1-ML | Meta-label ML overlay | P1 | AVAILABLE | Training script tested. ml_quality_hook=None by default. No model wired operationally. |
| P1-AGENTS | Agent dispatch | P1 | COMPLETE | run_daily_pipeline() dispatches 2 agents. Pipeline callable via `python -m core.orchestrator` or start_daemon(). |
| P1-GOV | Governance gate | P1 | WIRED (opt-in) | Called in promote(). CRITICAL raises ValueError. Non-critical falls through. |
| P1-AUDIT | Audit chains | P1 | DOWNGRADED | Scaffold |
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

## Residual Risks

| ID | Risk | Severity |
|----|------|----------|
| RR-001 | Walk-forward Sharpe subject to overfitting | HIGH |
| RR-002 | No live/paper trading system | HIGH |
| RR-003 | ML models never trained in production | HIGH |
| RR-004 | Audit chains empty | MEDIUM |
| RR-005 | 32 of 33 agents remain scaffold | MEDIUM |
