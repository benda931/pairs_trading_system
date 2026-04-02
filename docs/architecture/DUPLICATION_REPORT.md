# Duplication Report — Pairs Trading System
**Generated:** 2026-04-02

## 1. Duplicate Data Loading Paths

| # | Path | File | Lines | Status |
|---|------|------|-------|--------|
| 1 | `load_price_data()` | `common/data_loader.py:907` | Canonical | **USE THIS** |
| 2 | `load_price_data()` stub | `common/utils.py:221` | RuntimeError fallback | Legacy adapter |
| 3 | Root-level `data_loader.py` | `./data_loader.py:33` | Wrapper | Redundant |
| 4 | `yf_loader.py` | `./datafeed/yf_loader.py:27` | Separate module | Unused |
| 5 | Direct `yf.download()` | `common/data_providers.py:714,723` | Bypasses cache | **FIX** |
| 6 | Direct `yf.download()` | `core/macro_data.py:289` | Macro-specific | Acceptable |

**Problem:** 6 files bypass the canonical loader and hit Yahoo directly.

## 2. Duplicate Backtest Engines

| # | Engine | File | Status |
|---|--------|------|--------|
| 1 | `backtest_pair()` | `scripts/run_backtest.py:39` | **CANONICAL** |
| 2 | `run_backtest()` | `root/backtest.py:2095` | Dashboard variant |
| 3 | `run_backtest()` | `root/backtest_logic.py:678` | **NAME CONFLICT** with #2 |
| 4 | `backtest()` | `core/macro_engine.py:510` | Macro-specific (OK) |
| 5 | `OptimizationBacktester.run()` | `core/optimization_backtester.py` | Delegates to #1 via fallback |
| 6 | `run_portfolio_backtest()` | `core/portfolio_backtester.py` | Portfolio-level (different purpose) |

**Problem:** Two `run_backtest()` in root/ with same name. 5 distinct engines.

## 3. Duplicate Signal Paths

| # | Path | File | Status |
|---|------|------|--------|
| 1 | `SignalPipeline.evaluate()` | `core/signal_pipeline.py` | **CANONICAL** |
| 2 | `compute_universe_signals()` | `core/signals_engine.py` | Universe scanner (different purpose) |
| 3 | `SignalGenerator.generate()` | `common/signal_generator.py` | Helper library |
| 4 | `generate_signal_candidates()` | `common/signal_generator.py:1023` | Research |
| 5 | Orphaned signals | `scripts/run_mini_fund_signals.py:256` | Direct yf.download |

**Z-Score computed in 5+ places:** agents/signal_agents.py, agents/ml_agents.py,
root/backtest.py, ml/features/builder.py (x2), scripts/train_xgboost_model.py

## 4. Duplicate Configuration Loading

| # | Path | File | Status |
|---|------|------|--------|
| 1 | `load_config()` | `common/config_manager.py` | **CANONICAL** |
| 2 | AppContext wrapper | `core/app_context.py:497` | Acceptable |
| 3 | **Direct JSON write** | `agents/auto_agents.py:269` | **DANGEROUS** — bypasses validation |

## 5. Duplicate Optimization Implementations

17 files import optuna with NO factory pattern. 12 `create_study()` calls:
- `agents/auto_agents.py:231`
- `core/full_parameter_optimization.py:425,521`
- `core/optimizer.py:397,404`
- `core/fair_value_optimizer_v2.py:194`
- `core/meta_optimization.py:54`
- `core/walk_forward_engine.py:201`
- `root/optimization_tab.py:4122,4129`
- `scripts/optuna_backtest_search.py:693`
- `scripts/run_full_alpha.py` (2x)

## 6. Suspected Dead Code (Zero Imports, >200 Lines)

| File | Lines | Verdict |
|------|-------|---------|
| `root/dashboard.backup_mojibake.py` | 11,155 | **DELETE** — corrupted backup |
| `root/run_all.py` | 263 | LEGACY — old batch runner |
| `root/live_dash_app.py` | 78 | DEAD |
| `root/run_upgrade.py` | 7 | DEAD |
| `root/test_agent.py` | 12 | DEAD |
| `root/settings.py` | 0 | DEAD |
| `root/visualization.py` | 88 | DEAD |
| `root/dedupe_opt_tab.py` | 86 | One-time utility |
| `core/feature_selection.py` | 35 | DEAD |
| `core/paper_trader.py` | 390 | DEAD — never imported |
| `core/meta_optimization.py` | 122 | DEAD — never imported |

**Total dead code: ~12,236 lines (6% of codebase)**

## 7. Duplicate Class/Type Definitions

| Type | Locations | Problem |
|------|-----------|---------|
| `SignalDecision` | `core/intents.py:352` + `core/signal_pipeline.py:87` | Two definitions |
| `BacktestConfig` | `core/optimization_backtester.py:129` + `root/backtest.py:461` + `root/backtest_logic.py:145` | Three definitions |
| `TradeSide` | `root/backtest.py:371` + `root/trade_logic.py:21` | Two incompatible enums |
