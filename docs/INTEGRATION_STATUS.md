# Integration Status Register
**Date:** 2026-03-31
**Authoritative source:** This document and CLAUDE.md "Current Integration Status"

This register tracks what is truly operational vs scaffolded infrastructure.

## System A (Operational)

These modules are actively used for backtesting, optimization, and dashboard rendering.

| Module | Description | Status |
|--------|-------------|--------|
| `core/contracts.py` | Domain types, enums, protocols | **Operational** |
| `core/optimization_backtester.py` | Z-score mean-reversion backtester | **Operational** |
| `core/signals_engine.py` | Signal computation (legacy coordinator) | **Operational** (deprecation planned per P3-SIGMIG) |
| `core/pair_ranking.py` | Pair scoring for research | **Operational** (research scripts only) |
| `core/sql_store.py` | DuckDB/SQLite persistence | **Operational** |
| `core/orchestrator.py` | Daily pipeline orchestrator | **Operational** |
| `common/signal_generator.py` | Z-score/Bollinger/RSI computation | **Operational** |
| `common/data_loader.py` | Price data loading and caching | **Operational** |
| `common/fmp_client.py` | FMP API client | **Operational** |
| `research/discovery_pipeline.py` | Pair discovery orchestration | **Operational** |
| `research/pair_validator.py` | Statistical validation | **Operational** |
| `research/spread_constructor.py` | OLS/Rolling OLS/Kalman spreads | **Operational** |
| `root/dashboard.py` | Streamlit dashboard (15 tabs) | **Operational** |
| `root/optimization_tab.py` | Parameter optimization UI | **Operational** |
| `root/backtest.py` | Backtest tab rendering | **Operational** |

## System B (Infrastructure — Individually Tested, Not Integrated with System A)

These modules are professionally designed, comprehensively tested in isolation,
and awaiting integration with System A.

| Module | Description | Tests | Integration Status |
|--------|-------------|-------|--------------------|
| `core/signal_pipeline.py` | Canonical signal pipeline (ADR-006) | Tested + Wired | **Integrated**: backtester calls `evaluate_bar()` when `use_signal_pipeline=True` (P1-PIPE COMPLETE) |
| `core/threshold_engine.py` | Regime-conditioned thresholds | Tested | **Not called** from operational path |
| `core/signal_quality.py` | Signal quality grading (A+ to F) | Tested | **Not called** from operational path |
| `core/regime_engine.py` | Regime classification engine | Tested | **Not called** from operational path |
| `core/lifecycle.py` | Trade lifecycle state machine | Tested | **Not called** from operational path |
| `portfolio/allocator.py` | Portfolio construction engine | 82 tests pass | **Never receives** real signals (P1-PORTINT) |
| `portfolio/ranking.py` | Opportunity ranking (7 dimensions) | Tested | Only called from agents (not System A) |
| `portfolio/risk_ops.py` | Drawdown/kill-switch managers | Tested | Kill-switch bridge incomplete (P0-KS) |
| `ml/features/` | 61 feature definitions | Tested | No feature compute-correctness tests |
| `ml/labels/` | 26 label definitions | Tested | No training pipeline exists |
| `ml/inference/scorer.py` | ModelScorer with fallback | Tested | Always returns neutral 0.5 (no models) |
| `ml/registry/` | Champion/challenger registry | Tested | Governance gate not enforced (P1-GOV) |
| `agents/` | 33 registered agents | 91 tests pass | **Never dispatched** from operational workflow |
| `governance/engine.py` | Governance policy engine | 100 tests pass | **Not enforced** at runtime (P1-GOV) |
| `audit/chain.py` | Hash-linked audit chain | Tested | **All chains empty** (P1-AUDIT) |
| `surveillance/engine.py` | 12 surveillance rules | Tested | **detect() never called** operationally (P1-SURV2) |
| `runtime/state.py` | Runtime state manager | 113 tests pass | **is_safe_to_trade() never called** (P1-SAFE) |
| `control_plane/engine.py` | Control plane operations | Tested | No operational callers |

## Known Backtest Limitations (ADR-007)

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Same-close execution | Sharpe/PnL structurally optimistic | Documented; `bar_lag` param exists but not consumed |
| Calendar-segment "walk-forward" | Not true expanding-window WF | 63-day floor; use `WalkForwardHarness` for rigor |
| Flat cost model | Understates transaction costs | Document as limitation |
| No survivorship filtering | Universe may include delisted stocks | `EligibilityFilter` mitigates partially |

## Cross-References

- **Remediation findings:** `docs/remediation/remediation_ledger.md`
- **Migration status:** `docs/migration/migration_ledger.md`
- **Architecture decisions:** `docs/adr/ADR-001` through `ADR-007`
- **CLAUDE.md:** Full platform handbook with extension guides
