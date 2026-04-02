# Integration Status Register
**Date:** 2026-04-01
**Truth Audit:** Reconciled + operational pipeline wired (v2.2)

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
| `core/orchestrator.py` | **Operational** | run_daily_pipeline() wired end-to-end; CLI: `scripts/run_daily_pipeline.py`; daemon: start_daemon() |
| `scripts/run_daily_pipeline.py` | **Operational** | CLI trigger for daily pipeline; supports --daemon, --dry-run, --capital |
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
| `portfolio/risk_ops.py` kill-switch callback | `control_plane_callback=fn` | **Active** | make_kill_switch_manager_with_control_plane() called in every run_portfolio_allocation_cycle() (P0-KS COMPLETE) |
| `ml/registry/registry.py` governance gate | Governance check in promote() | **Active but advisory** | CRITICAL blocks; non-critical falls through (P1-GOV) |

## Available but Not Called Operationally

| Module | What Exists | Why Not Called |
|--------|-------------|---------------|
| `scripts/train_meta_label.py` | ML training + inference | No model wired to SignalPipeline operationally (P1-ML) |

## Now Operational (promoted from Available)

| Module | Evidence | Finding Closed |
|--------|----------|----------------|
| `core/portfolio_bridge.py` bridge_signals_to_allocator() | Called from run_daily_pipeline() via run_portfolio_allocation_cycle() | P1-PORTINT COMPLETE |
| `core/portfolio_bridge.py` safety_check | is_safe_to_trade injected in run_portfolio_allocation_cycle() | P1-SAFE COMPLETE |
| `core/orchestrator.py` agent dispatch | 2 agents dispatched; pipeline callable via CLI + daemon | P1-AGENTS COMPLETE |
| `portfolio/risk_ops.py` kill-switch | make_kill_switch_manager_with_control_plane() called per cycle | P0-KS COMPLETE |

## Scaffold (Individually Tested, No Operational Integration)

| Module | Tests | Notes |
|--------|-------|-------|
| `core/threshold_engine.py` | Tested | Called through signal_pipeline (default backtester path) |
| `core/signal_quality.py` | Tested | Called through signal_pipeline (default backtester path) |
| `core/regime_engine.py` | Tested | Called through signal_pipeline (default backtester path) |
| `core/lifecycle.py` | Tested | Called through signal_pipeline (default backtester path) |
| `portfolio/allocator.py` | 82+ tests | Called through portfolio_bridge (dashboard + daily pipeline) |
| `ml/inference/scorer.py` | Tested | Always returns neutral 0.5 (no models trained) |
| `agents/` (38 of 40 scaffold) | 91+ tests | 2 operational (system_health, data_integrity); 38 registered but never dispatched |
| `governance/engine.py` | 100 tests | Not enforced at runtime except in promote() |
| `audit/chain.py` | Tested | All chains empty |
| `surveillance/engine.py` | Tested | 12 rules; only SURV-DI-001 called operationally |
| `runtime/state.py` | 113 tests | is_safe_to_trade() called in run_portfolio_allocation_cycle() |
| `control_plane/engine.py` | Tested | Called via make_kill_switch_manager_with_control_plane() |

## Backtest Limitations (ADR-007)

| Limitation | Status | Notes |
|------------|--------|-------|
| Execution timing | **FIXED** | bar_lag=1 default (next-bar fill) |
| Calendar WF | **HONESTLY LABELED** | 63-day floor, docstring warns "NOT true walk-forward" |
| Flat cost model | DEFERRED | Acceptable for daily-frequency research |
| Survivorship bias | DOCUMENTED | EligibilityFilter with explicit residual-risk comment |
