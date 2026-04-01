# Integration Status Register
**Date:** 2026-04-01
**Truth Audit:** Reconciled against actual code paths (not doc claims)

## Status Key
- **Operational** = Runs by default in real code paths
- **Opt-in** = Code exists, tested, requires explicit flag/parameter to activate
- **Available** = Defined and tested; no operational code invokes it
- **Scaffold** = Infrastructure exists; no integration

## System A (Operational by Default)

| Module | Status | Notes |
|--------|--------|-------|
| `core/contracts.py` | **Operational** | Single source of truth for all domain types |
| `core/optimization_backtester.py` | **Operational** | bar_lag=1 by default (P0-EXEC COMPLETE) |
| `core/signals_engine.py` | **Operational** | Legacy coordinator; deprecation planned (P3-SIGMIG) |
| `core/sql_store.py` | **Operational** | DuckDB persistence |
| `core/orchestrator.py` | **Operational** | Pipeline defined, but run_daily_pipeline() has no trigger |
| `common/signal_generator.py` | **Operational** | Z-score computation |
| `common/data_loader.py` | **Operational** | Includes SURV-DI-001 surveillance hook (P1-SURV2 COMPLETE) |
| `common/fmp_client.py` | **Operational** | FMP API connected |
| `research/` | **Operational** | Discovery, validation, spread construction, walk-forward |
| `root/dashboard.py` | **Operational** | 15 tabs, Streamlit UI |
| `root/optimization_tab.py` | **Operational** | 63-day WF floor (P0-WF COMPLETE) |

## Opt-In Integrations (tested, require explicit enablement)

| Module | Flag/Mechanism | Default | Notes |
|--------|---------------|---------|-------|
| `core/signal_pipeline.py` | `use_signal_pipeline=True` in params | **True** | Default changed — SignalPipeline.evaluate_bar() is now the default backtester path (P1-PIPE COMPLETE) |
| `portfolio/risk_ops.py` kill-switch callback | `control_plane_callback=fn` | **None** | Factory exists, never called operationally (P0-KS) |
| `ml/registry/registry.py` governance gate | Governance check in promote() | **Active but advisory** | CRITICAL blocks; non-critical falls through (P1-GOV) |

## Available but Not Called Operationally

| Module | What Exists | Why Not Called |
|--------|-------------|---------------|
| `core/portfolio_bridge.py` | bridge_signals_to_allocator() | Called from dashboard UI only, not backend (P1-PORTINT) |
| `core/portfolio_bridge.py` safety_check | safety_check callback parameter | No caller injects is_safe_to_trade (P1-SAFE) |
| `scripts/train_meta_label.py` | ML training + inference | No model wired to SignalPipeline operationally (P1-ML) |
| `core/orchestrator.py` agent dispatch | 2 agents in run_daily_pipeline() | Pipeline itself has no CLI/cron trigger (P1-AGENTS) |

## Scaffold (Individually Tested, No Operational Integration)

| Module | Tests | Notes |
|--------|-------|-------|
| `core/threshold_engine.py` | Tested | Called only through signal_pipeline (which is opt-in) |
| `core/signal_quality.py` | Tested | Called only through signal_pipeline (which is opt-in) |
| `core/regime_engine.py` | Tested | Called only through signal_pipeline (which is opt-in) |
| `core/lifecycle.py` | Tested | Called only through signal_pipeline (which is opt-in) |
| `portfolio/allocator.py` | 82+ tests | Called only through portfolio_bridge (dashboard manual) |
| `ml/inference/scorer.py` | Tested | Always returns neutral 0.5 (no models trained) |
| `agents/` (33 agents) | 91+ tests | 2 wired to pipeline, but pipeline never triggered |
| `governance/engine.py` | 100 tests | Not enforced at runtime except in promote() |
| `audit/chain.py` | Tested | All chains empty |
| `surveillance/engine.py` | Tested | 12 rules; only SURV-DI-001 called operationally |
| `runtime/state.py` | 113 tests | is_safe_to_trade() defined, never called |
| `control_plane/engine.py` | Tested | No operational callers |

## Backtest Limitations (ADR-007)

| Limitation | Status | Notes |
|------------|--------|-------|
| Execution timing | **FIXED** | bar_lag=1 default (next-bar fill) |
| Calendar WF | **HONESTLY LABELED** | 63-day floor, docstring warns "NOT true walk-forward" |
| Flat cost model | DEFERRED | Acceptable for daily-frequency research |
| Survivorship bias | DOCUMENTED | EligibilityFilter with explicit residual-risk comment |
