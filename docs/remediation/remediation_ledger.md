# Remediation Ledger — Pairs Trading System
**Version:** 4.0
**Date:** 2026-04-02
**Status:** Active — reconciled against actual code paths

## Status Definitions
- **COMPLETE** = Operational by default in real code paths
- **WIRED (opt-in)** = Requires explicit enablement
- **AVAILABLE** = Defined, tested, callable but not auto-invoked
- **DEFERRED** = Intentionally postponed

## P0 Findings — All COMPLETE

| ID | Title | Status | Evidence |
|----|-------|--------|----------|
| P0-WF | Calendar WF labeled | COMPLETE | 63-day floor, docstring warns |
| P0-EXEC | bar_lag execution delay | COMPLETE | default=1 (next-bar), fallback to Yahoo/FMP |
| P0-KS | Kill-switch bridge | COMPLETE | make_kill_switch_manager_with_control_plane() per cycle |
| P0-DOCS | Documentation truthfulness | COMPLETE | Ledger v4.0 + INTEGRATION_STATUS v3.0 reconciled |

## P1 Findings — All COMPLETE or WIRED

| ID | Title | Status | Evidence |
|----|-------|--------|----------|
| P1-PIPE | Signal pipeline default | COMPLETE | use_signal_pipeline=True in backtester |
| P1-PORTINT | Portfolio bridge | COMPLETE | bridge_signals_to_allocator() in daily pipeline |
| P1-MINOBS | MIN_OBS=252 | COMPLETE | core/contracts.py |
| P1-SAFE | Safety gating | COMPLETE | is_safe_to_trade injected in allocation cycle |
| P1-SURV | Survivorship docs | COMPLETE | Explicit comment in universe.py |
| P1-ML | ML overlay (XGBoost) | COMPLETE | XGBoost loaded as priority 1 (AUC=0.778) |
| P1-AGENTS | 48 agents (13 daily) | COMPLETE | 13 auto-dispatched, 27 on-demand, 8 GPT/auto |
| P1-GOV | Governance gate | WIRED (opt-in) | CRITICAL blocks in promote() |
| P1-AUDIT | Audit chain writer | AVAILABLE | core/audit_writer.py writes to 5 chains |
| P1-SURV2 | Surveillance hook | COMPLETE | SURV-DI-001 in load_price_data() |

## P2+ Findings

| ID | Title | Status |
|----|-------|--------|
| P2-COSTS | Flat cost model | DEFERRED |
| P2-DUPRANK | Deprecated pair_ranking | COMPLETE |
| P2-DUPTHROT | Duplicate throttle | DEFERRED |
| P2-COINT | Hard stability rejection | COMPLETE |
| P2-MLT | ML training scripts | COMPLETE |
| P3-SIGMIG | signals_engine.py role clarified | COMPLETE |
| P3-PARTA | Partial fills | DEFERRED |
| P4-BACKUP | Backup files removed | COMPLETE |

## Residual Risks

| ID | Risk | Severity | Mitigation |
|----|------|----------|-----------|
| RR-001 | Walk-forward validated but in-sample optimization still possible | MEDIUM | WF engine available (DSR=1.000 on tested pairs) |
| RR-002 | No live/paper trading system | HIGH | Position tracker ready; IBKR stub exists |
| RR-003 | Simplified transaction costs (5+2 bps) | MEDIUM | Acceptable for daily stat arb research |
| RR-004 | Audit chains available but not auto-populated in all paths | LOW | core/audit_writer.py ready for integration |
