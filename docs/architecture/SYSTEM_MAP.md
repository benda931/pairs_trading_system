# System Map — Pairs Trading System
**Generated:** 2026-04-02
**Method:** Full codebase inspection (191 files, 203,123 lines)

## Top-Level Subsystem Map

```
pairs_trading_system/
│
├── core/          (60 files, 51,930 lines) — Domain logic, engines, contracts
│   ├── contracts.py         895L — THE source of truth for all types
│   ├── signal_pipeline.py   609L — Canonical signal decision engine
│   ├── signals_engine.py   2734L — Universe batch scanner (legacy coordinator)
│   ├── optimization_backtester.py 3108L — Backtester with monkey-patched run()
│   ├── portfolio_backtester.py  465L — Portfolio-level Kelly/vol-target backtest
│   ├── portfolio_bridge.py  260L — Signal → portfolio adapter
│   ├── orchestrator.py     1377L — Daily pipeline + agent dispatch
│   ├── sql_store.py        2950L — DuckDB persistence
│   ├── app_context.py      4358L — Giant app bootstrap (NEEDS REFACTOR)
│   ├── walk_forward_engine.py 318L — Expanding-window WF validation
│   ├── risk_engine.py      2495L — Risk calculations
│   └── ... (49 more files)
│
├── common/        (30 files, 24,238 lines) — Shared utilities, data loading
│   ├── data_loader.py      1213L — Canonical price loader
│   ├── signal_generator.py 1112L — Low-level z-score/Bollinger/RSI library
│   ├── fmp_client.py        551L — FMP API client
│   ├── gpt_client.py        300L — GPT-4o client with cost tracking
│   ├── config_manager.py    601L — Configuration loading
│   ├── helpers.py          1710L — Large utility bag (NEEDS SPLIT)
│   └── ... (24 more files)
│
├── root/          (40 files, 98,217 lines) — Streamlit dashboard + UI
│   ├── dashboard.py       18725L — Main dashboard shell (35 parts!)
│   ├── optimization_tab.py 13191L — Optimization UI
│   ├── portfolio_tab.py    6539L — Portfolio tab
│   ├── alpha_dashboard_tab.py 314L — Alpha performance (Plotly)
│   └── ... (36 more files including backup)
│
├── scripts/       (32 files, 11,297 lines) — CLI tools, pipelines
│   ├── run_full_alpha.py    408L — Full alpha pipeline
│   ├── run_auto_improve.py  298L — GPT auto-improvement
│   ├── run_backtest.py      563L — Realistic backtester
│   ├── train_xgboost_model.py 284L — XGBoost training
│   └── ... (28 more files)
│
├── agents/        (11 files, 7,706 lines) — 48 registered agents
├── research/       (9 files, 5,455 lines) — Discovery, validation, spreads
├── portfolio/      (9 files, 4,280 lines) — Allocation, sizing, risk ops
├── ml/            (25 files, ~8,000 lines) — ML platform (features, models, inference)
├── governance/     (3 files) — Policy engine
├── audit/          (3 files) — Audit chain
├── surveillance/   (3 files) — Surveillance engine
├── runtime/        (3 files) — Runtime state manager
├── control_plane/  (3 files) — Control plane engine
├── orchestration/  (3 files) — Workflow engine
├── monitoring/     (3 files) — Health + workflow monitoring
└── tests/         (19 files, ~10,000 lines) — 899 tests passing
```

## Runtime Flow Summary

```
DAILY PIPELINE (core/orchestrator.py:run_daily_pipeline):
  1. health_check → SystemHealthAgent
  2. data_refresh → DataIntegrityAgent
  3. compute_signals → 4 signal agents (Regime, Analyst, Lifecycle, Exit)
     → _collect_signal_decisions() [loads XGBoost ML model]
     → SignalPipeline.evaluate() per pair
  4. portfolio_allocation → bridge_signals_to_allocator()
     → safety_check=is_safe_to_trade()
     → kill_switch=make_kill_switch_manager_with_control_plane()
     → PortfolioAllocator.run_cycle()
  5. 7 risk agents (Exposure, Drawdown, KillSwitch, Capital, DeRisk, Drift, Alert)
  6. risk_check

ALPHA PIPELINE (scripts/run_full_alpha.py):
  1. Discover → score 104 pairs by composite metric
  2. Validate → ADF + cointegration + Hurst tests
  3. Optimize → Optuna per validated pair (30 trials)
  4. Walk-Forward → 3-fold OOS validation
  5. Backtest → realistic PnL per pair
  6. Filter → keep Sharpe > 0.3
  7. Portfolio → equal-weight Kelly/vol-target backtest
  8. Report → GPT-4o analysis

DASHBOARD (root/dashboard.py → streamlit run):
  → 10 active tabs: Home, Alpha, Pair, Backtest, Optimization,
    Macro, Portfolio, Risk, Config, Agents/Logs
  → 5 hidden tabs: Smart Scan, Matrix, Comparison, Insights, Fair Value
```

## Architectural Layering (as actually implemented)

```
Layer 1: DOMAIN CONTRACTS    core/contracts.py (types, enums)
Layer 2: DATA LOADING        common/data_loader.py, common/fmp_client.py
Layer 3: SIGNAL COMPUTATION  common/signal_generator.py → core/signal_pipeline.py
Layer 4: PORTFOLIO LOGIC     portfolio/allocator.py, core/portfolio_bridge.py
Layer 5: ML OVERLAY          ml/ → XGBoost models via orchestrator
Layer 6: RISK MANAGEMENT     portfolio/risk_ops.py, runtime/state.py
Layer 7: ORCHESTRATION       core/orchestrator.py, monitoring/workflow.py
Layer 8: AGENTS              agents/ (48 agents, 13 auto-dispatched)
Layer 9: PERSISTENCE         core/sql_store.py (DuckDB)
Layer 10: UI                 root/ (Streamlit dashboard)
Layer 11: TOOLING            scripts/ (CLI pipelines)
```
