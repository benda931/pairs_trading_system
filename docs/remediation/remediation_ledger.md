# Remediation Ledger — Pairs Trading System
**Version:** 3.0
**Date:** 2026-04-02
**Status:** Active

## Status Definitions
- **COMPLETE** = Operational by default
- **WIRED (opt-in)** = Requires explicit enablement
- **DEFERRED** = Intentionally postponed
- **DOWNGRADED** = Scaffold, not blocking

## P0 Findings (All COMPLETE)

| ID | Title | Status | Evidence |
|----|-------|--------|----------|
| P0-WF | Calendar WF labeled | COMPLETE | 63-day floor, docstring warns |
| P0-EXEC | bar_lag execution delay | COMPLETE | default=1 (next-bar) |
| P0-KS | Kill-switch bridge | COMPLETE | make_kill_switch_manager_with_control_plane() in every cycle |
| P0-DOCS | Documentation truthfulness | COMPLETE | Ledger v3.0, all docs reconciled |

## P1 Findings (All COMPLETE or WIRED)

| ID | Title | Status | Evidence |
|----|-------|--------|----------|
| P1-PIPE | Signal pipeline default | COMPLETE | use_signal_pipeline=True |
| P1-PORTINT | Portfolio bridge | COMPLETE | bridge_signals_to_allocator() in daily pipeline |
| P1-MINOBS | MIN_OBS=252 | COMPLETE | core/contracts.py |
| P1-SAFE | Safety gating | COMPLETE | is_safe_to_trade injected |
| P1-SURV | Survivorship docs | COMPLETE | Explicit comment |
| P1-ML | Meta-label ML overlay | COMPLETE | Trained model at models/meta_label_latest.pkl, auto-loaded by orchestrator |
| P1-AGENTS | All 40 agents dispatchable | COMPLETE | 13 daily auto + 27 on-demand |
| P1-GOV | Governance gate | WIRED (opt-in) | CRITICAL blocks in promote() |
| P1-AUDIT | Audit chains | DOWNGRADED | Scaffold |
| P1-SURV2 | Stale data surveillance | COMPLETE | SURV-DI-001 in load_price_data() |

## P2 Findings

| ID | Title | Status | Evidence |
|----|-------|--------|----------|
| P2-COSTS | Flat cost model | DEFERRED | Acceptable for daily |
| P2-DUPRANK | Duplicate pair ranking | COMPLETE | core/pair_ranking.py deprecated with header |
| P2-DUPTHROT | Duplicate throttle | DEFERRED | Low priority |
| P2-COINT | Hard stability rejection | COMPLETE | stability < 0.15 → hard reject in pair_validator.py |
| P2-MLT | ML training script | COMPLETE | scripts/train_meta_label.py |

## P3+ Findings

| ID | Title | Status |
|----|-------|--------|
| P3-SIGMIG | signals_engine.py role clarified | COMPLETE |
| P3-PARTA | Partial fills | DEFERRED |
| P4-BACKUP | Backup files removed | COMPLETE |

## Residual Risks

| ID | Risk | Severity |
|----|------|----------|
| RR-001 | Walk-forward Sharpe subject to overfitting | HIGH |
| RR-002 | No live/paper trading system | HIGH |
| RR-004 | Audit chains empty | MEDIUM |
