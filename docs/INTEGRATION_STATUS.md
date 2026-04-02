# Integration Status Register
**Date:** 2026-04-02
**Version:** 3.0 — Full reconciliation against actual code paths

## Status Key
- **Operational** = Runs by default in real code paths
- **Opt-in** = Requires explicit flag/parameter
- **Available** = Defined and tested; callable but not auto-invoked
- **Scaffold** = Infrastructure exists; no operational integration

## System A — Operational Core

| Module | Status | Evidence |
|--------|--------|----------|
| `core/contracts.py` | **Operational** | Single source of truth for all domain types |
| `core/optimization_backtester.py` | **Operational** | bar_lag=1 default; Yahoo/FMP fallback on SqlStore failure |
| `core/signal_pipeline.py` | **Operational** | Default backtester path (use_signal_pipeline=True) |
| `core/portfolio_bridge.py` | **Operational** | bridge_signals_to_allocator() in every daily pipeline run |
| `core/portfolio_backtester.py` | **Operational** | Kelly sizing, vol targeting, regime-conditional, 28 alpha pairs |
| `core/walk_forward_engine.py` | **Operational** | Expanding-window WF with deflated Sharpe (DSR=1.000) |
| `core/orchestrator.py` | **Operational** | Daily pipeline: signals → agents → allocation → risk |
| `core/position_tracker.py` | **Available** | Position/order tracking ready for live trading |
| `core/attribution.py` | **Available** | Alpha decomposition (pair selection, timing, sizing) |
| `core/alpha_persistence.py` | **Available** | Save/load alpha results to SqlStore with JSON fallback |
| `core/alerts.py` | **Available** | Telegram + console alerts (needs TELEGRAM_BOT_TOKEN in .env) |
| `core/audit_writer.py` | **Available** | Writes to 5 named audit chains (signals, allocations, risk, models, config) |
| `core/signals_engine.py` | **Operational** | Universe batch scanner (legacy, deprecation planned) |
| `core/sql_store.py` | **Operational** | DuckDB persistence |
| `common/signal_generator.py` | **Operational** | Z-score computation library |
| `common/data_loader.py` | **Operational** | Price loading + SURV-DI-001 surveillance hook |
| `common/fmp_client.py` | **Operational** | FMP API (key in .env, not config.json) |
| `common/gpt_client.py` | **Operational** | GPT-4o client with cost tracking ($5/day limit) |

## Scripts — Operational Pipelines

| Script | Status | What It Does |
|--------|--------|-------------|
| `scripts/run_full_alpha.py` | **Operational** | Full alpha: discover→optimize→backtest→filter→portfolio (28 alpha pairs, Sharpe 1.19) |
| `scripts/run_auto_improve.py` | **Operational** | GPT-4o analysis + model retrain + param optimize + config update |
| `scripts/run_backtest.py` | **Operational** | Realistic per-pair + portfolio backtester with SPY benchmark |
| `scripts/run_daily_pipeline.py` | **Operational** | CLI trigger for orchestrator daily pipeline |
| `scripts/train_meta_label.py` | **Operational** | LogReg meta-label training (AUC varies by pair) |
| `scripts/train_xgboost_model.py` | **Operational** | XGBoost meta-label with 40+ features (AUC=0.778 for XLI/XLB) |
| `scripts/run_alpha_pipeline.py` | **Operational** | Discovery + validation + signals (6-stage pipeline) |
| `scripts/setup_scheduler.py` | **Available** | Windows Task Scheduler for 4 recurring tasks |

## Agents — 48 Registered, 13 Daily Auto-Dispatched

| Layer | Agents | Auto-Dispatched | Evidence |
|-------|--------|-----------------|----------|
| Monitoring | SystemHealth, DataIntegrity | 2 (daily) | After health_check, data_refresh |
| Signal | RegimeSurveillance, SignalAnalyst, TradeLifecycle, ExitOversight | 4 (daily) | After compute_signals |
| Risk | ExposureMonitor, DrawdownMonitor, KillSwitch, CapitalBudget, DeRisking, DriftMonitoring, AlertAggregation | 7 (daily) | After portfolio_allocation |
| Research | 11 agents | On-demand | `dispatch_research_agents(symbols)` |
| ML | 7 agents | On-demand | `dispatch_ml_agents()` |
| Governance | 8 agents | On-demand | `dispatch_governance_agents()` |
| GPT Analysis | GPTSignalAdvisor, GPTModelTuner, GPTStrategyResearcher, GPTReportGenerator | On-demand | Via `run_auto_improve.py` |
| Auto-Execution | AutoModelRetrainer, AutoDataRefresher, AutoParameterOptimizer, AutoConfigUpdater | On-demand | Via `run_auto_improve.py` |

## ML Layer

| Component | Status | Evidence |
|-----------|--------|----------|
| XGBoost meta-label | **Operational** | Orchestrator loads xgb_meta_*.pkl as priority 1 (AUC=0.778) |
| LogReg meta-label | **Operational** | Fallback when XGBoost unavailable (meta_label_latest.pkl) |
| 40+ engineered features | **Operational** | scripts/train_xgboost_model.py compute_advanced_features() |
| ModelScorer | **Scaffold** | Always returns neutral 0.5 (not used; direct hook instead) |
| ML governance gate | **Opt-in** | promote() checks governance; CRITICAL blocks |

## Runtime Safety

| Component | Status | Evidence |
|-----------|--------|----------|
| is_safe_to_trade() | **Operational** | Injected in run_portfolio_allocation_cycle() |
| Kill-switch + control-plane | **Operational** | make_kill_switch_manager_with_control_plane() per cycle |
| Regime-conditional sizing | **Operational** | 4 regimes: LOW_VOL(1.3x), NORMAL(1.0x), HIGH_VOL(0.6x), CRISIS(0.2x) |
| Drawdown de-risking | **Operational** | -10% deleverage, -20% halt |
| Vol targeting | **Operational** | 10% annual vol target with dynamic scalar |

## Dashboard

| Tab | Status | Notes |
|-----|--------|-------|
| Dashboard Home | **Operational** | Health, alerts, investment readiness |
| Alpha Performance | **Operational** | Plotly equity curve, drawdown, pair heatmap, trade blotter |
| Pair Analysis | **Operational** | Z-score, correlation, half-life, normalized charts |
| Backtest | **Operational** | Real PnL via Yahoo/FMP fallback |
| Optimization | **Operational** | Optuna-based, 63-day WF floor |
| Macro | **Operational** | FMP data connected |
| Portfolio / Fund View | **Operational** | Alpha pair configs loaded |
| Risk | **Operational** | Auto-loads equity curves from backtests |
| Config | **Operational** | Settings management |
| Agents / Logs | **Operational** | Agent status + system logs |

## Alpha Performance (verified results)

| Metric | Value | Benchmark (SPY) |
|--------|-------|-----------------|
| Portfolio Sharpe | 1.19 | 0.68 |
| CAGR | 12.6% | — |
| Max Drawdown | -8.2% | ~-24.5% |
| Alpha pairs | 28 of 104 | — |
| Walk-forward OOS Sharpe | 1.10 | — |
| Deflated Sharpe Ratio | 1.000 | — |
| Top pair (IWM/SPY) | Sharpe 1.24 | — |

## Backtest Limitations (ADR-007)

| Limitation | Status |
|------------|--------|
| Execution timing | **FIXED** — bar_lag=1 default |
| Calendar WF | **FIXED** — 63-day floor, honest labeling |
| Flat cost model | DEFERRED — 5bps + 2bps market impact |
| Survivorship bias | DOCUMENTED — EligibilityFilter |
