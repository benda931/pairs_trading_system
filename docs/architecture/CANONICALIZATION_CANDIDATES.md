# Canonicalization Candidates — Pairs Trading System
**Generated:** 2026-04-02

## Priority 1: Data Loading Canonicalization

| Current | Candidate | Rationale | Difficulty | Risk |
|---------|-----------|-----------|------------|------|
| 6 data loading paths | `common/data_loader.py:load_price_data()` | Already used by 50+ files; has caching, yfinance MultiIndex fix, surveillance hook | LOW | LOW |
| **Action:** Replace 6 direct `yf.download()` calls with canonical loader | | | |

## Priority 2: Backtest Engine Canonicalization

| Current | Candidate | Rationale | Difficulty | Risk |
|---------|-----------|-----------|------------|------|
| 5 backtest engines | `scripts/run_backtest.py:backtest_pair()` for per-pair, `core/portfolio_backtester.py` for portfolio | Already used by optimization_backtester fallback, alpha pipeline | MEDIUM | LOW |
| **Action:** Rename `root/backtest_logic.py:run_backtest()` to avoid name conflict with `root/backtest.py:run_backtest()` | | |

## Priority 3: Z-Score Computation Canonicalization

| Current | Candidate | Rationale | Difficulty | Risk |
|---------|-----------|-----------|------------|------|
| 5+ z-score implementations | `common/signal_generator.py:zscore_signals()` | Low-level helper; all others should delegate | MEDIUM | LOW |
| **Action:** Create `common.signal_generator.compute_zscore(spread, lookback)` utility and use everywhere | | |

## Priority 4: Configuration Write Canonicalization

| Current | Candidate | Rationale | Difficulty | Risk |
|---------|-----------|-----------|------------|------|
| Direct JSON writes in `auto_agents.py` | `common/config_manager.py:save_config()` | Has validation, backup logic | LOW | MEDIUM |
| **Action:** Replace `json.dump()` in auto_agents with `save_config()` | | |

## Priority 5: Optuna Factory

| Current | Candidate | Rationale | Difficulty | Risk |
|---------|-----------|-----------|------------|------|
| 12 `create_study()` calls | Create `common/optuna_factory.py:create_optuna_study()` | Consistent storage, logging, sampler config | MEDIUM | LOW |
| **Action:** Centralize study creation with default storage path, sampler, pruner | | |

## Priority 6: Dead Code Removal

| Current | Action | Lines Removed |
|---------|--------|---------------|
| `root/dashboard.backup_mojibake.py` | DELETE | 11,155 |
| `root/run_all.py` | DELETE | 263 |
| `root/live_dash_app.py` | DELETE | 78 |
| `root/run_upgrade.py` | DELETE | 7 |
| `root/test_agent.py` | DELETE | 12 |
| `root/settings.py` | DELETE | 0 |
| `root/visualization.py` | DELETE | 88 |
| `core/feature_selection.py` | DELETE | 35 |
| `core/paper_trader.py` | DELETE | 390 |
| `core/meta_optimization.py` | DELETE | 122 |
| **Total** | | **12,150 lines** |

## Priority 7: Type Definition Deduplication

| Duplicate | Canonical | Action |
|-----------|-----------|--------|
| `SignalDecision` in `signal_pipeline.py` | `core/intents.py:SignalDecision` | Use intents.py definition, import in signal_pipeline |
| `BacktestConfig` in 3 files | `core/optimization_backtester.py` | Import from there in root/ files |
| `TradeSide` in 2 files | `core/contracts.py:SignalDirection` | Already deprecated (MIG-002) |

---

## Top 10 Highest-Leverage Architectural Problems

1. **12,150 lines of confirmed dead code** — Pure noise. DELETE immediately.
   - Risk: ZERO. Files have zero imports.
   - Leverage: Removes 6% of codebase. Makes everything clearer.

2. **root/dashboard.py is 18,725 lines** — Unmaintainable monolith.
   - 35 "parts" in one file. Should be split into focused modules.
   - Risk: HIGH (everything imports from it)
   - Leverage: Highest maintenance cost reduction.

3. **core/app_context.py is 4,358 lines** — God object.
   - Bootstraps everything. 30+ try/except blocks that swallow errors.
   - Risk: MEDIUM
   - Leverage: Major debugging/testability improvement.

4. **5 independent z-score implementations** — Correctness risk.
   - Each may have slightly different semantics (window, fillna, etc.)
   - Risk: LOW (research correctness)
   - Leverage: Reproducibility guarantee.

5. **Direct yf.download() bypasses canonical loader** — Cache/rate-limit risk.
   - 6 files bypass common/data_loader.py
   - Risk: LOW
   - Leverage: Eliminates rate limiting issues.

6. **No shared Optuna factory** — Inconsistent optimization.
   - 12 independent create_study() calls with different configs.
   - Risk: LOW
   - Leverage: Consistent experiment tracking.

7. **BacktestConfig defined in 3 files** — Contract divergence risk.
   - core/ and root/ definitions may drift.
   - Risk: MEDIUM
   - Leverage: Single source of truth for backtest config.

8. **auto_agents.py writes config.json directly** — Corruption risk.
   - Bypasses config_manager.py validation.
   - Risk: MEDIUM (auto-improvement changes config)
   - Leverage: Safety of autonomous system.

9. **core/signals_engine.py (2,734 lines) still operational** — Legacy.
   - Used by orchestrator.task_compute_signals(), sql_store, app_context.
   - signal_pipeline.py is canonical but signals_engine is still needed.
   - Risk: LOW (clarified with role header)
   - Leverage: Future migration target.

10. **71 entrypoints in a single repo** — Confusion risk.
    - Many are one-time scripts, research tools, or dead stubs.
    - Risk: LOW
    - Leverage: Contributor clarity.
