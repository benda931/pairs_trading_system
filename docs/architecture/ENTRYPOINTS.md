# Entrypoints — Pairs Trading System
**Generated:** 2026-04-02
**Total entrypoints found:** 71 files with `if __name__ == "__main__"`

## Canonical Entrypoints (Production-Intended)

| Entrypoint | Purpose | Triggers |
|------------|---------|----------|
| `streamlit run root/dashboard.py` | Main dashboard | 10+ tabs, 18,725 lines |
| `python scripts/run_daily_pipeline.py` | Daily pipeline | orchestrator → 13 agents → allocation |
| `python scripts/run_full_alpha.py` | Alpha pipeline | discover → optimize → backtest → filter |
| `python scripts/run_auto_improve.py` | GPT auto-improvement | data refresh → GPT analysis → retrain → optimize |
| `python scripts/run_backtest.py` | Realistic backtester | Per-pair + portfolio backtest |
| `python scripts/train_xgboost_model.py` | ML training | XGBoost meta-label (40+ features) |
| `python scripts/train_meta_label.py` | ML training | LogReg meta-label |
| `python scripts/run_alpha_pipeline.py` | Signal pipeline | discover → validate → signal → rank |
| `python -m uvicorn root.api_server:app` | Fair Value API | FastAPI on port 8000 |
| `python scripts/setup_scheduler.py` | Task Scheduler | Creates 4 Windows scheduled tasks |
| `start_system.bat` | Full system launch | API + Dashboard + Alpha + Auto-improve |

## Research/Diagnostic Entrypoints

| Entrypoint | Purpose | Status |
|------------|---------|--------|
| `scripts/optuna_backtest_search.py` | Optuna parameter search | Research tool |
| `scripts/research_rank_pairs_from_dq.py` | Pair ranking research | Research tool |
| `scripts/ingest_prices_fmp.py` | FMP price ingestion | Data tool |
| `scripts/ingest_prices_for_dq_pairs.py` | DQ pair price ingestion | Data tool |
| `scripts/build_dq_pairs_universe.py` | Universe construction | Research tool |
| `scripts/run_mini_fund_signals.py` | Mini fund signals | Research tool |
| `scripts/run_mini_fund_optimize.py` | Mini fund optimization | Research tool |
| `scripts/run_mini_fund_snapshot.py` | Mini fund snapshot | Research tool |
| `scripts/replay_best_trial.py` | Replay Optuna trial | Debug tool |
| `scripts/maintain_duckdb_cache.py` | DuckDB maintenance | Ops tool |

## Legacy/Deprecated Entrypoints

| Entrypoint | Status | Notes |
|------------|--------|-------|
| `root/run_all.py` | LEGACY | Old batch runner |
| `root/run_upgrade.py` | DEAD | 7 lines, does nothing |
| `root/live_dash_app.py` | DEAD | 78 lines, empty |
| `root/test_agent.py` | DEAD | 12 lines, stub |
| `root/dedupe_opt_tab.py` | UTILITY | One-time dedup tool |
| `root/settings.py` | DEAD | 0 lines |
| `root/visualization.py` | DEAD | 88 lines, minimal |
| `hedge_fund_upgrade_agent.py` | LEGACY | Root-level GPT upgrader |
| `health_check_full_system.py` | LEGACY | Root-level health check |

## Backup File (Should Delete)

| File | Lines | Status |
|------|-------|--------|
| `root/dashboard.backup_mojibake.py` | 11,155 | DELETE — mojibake backup |
