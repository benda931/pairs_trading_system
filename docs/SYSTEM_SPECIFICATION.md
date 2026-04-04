# Pairs Trading System — Complete Technical Specification
## AI-to-AI Transfer Document

> **Purpose:** This document describes the COMPLETE architecture, capabilities, and internals
> of an institutional-grade statistical arbitrage platform. It is designed for transfer between
> AI systems — any AI reading this should understand every component, its purpose, status, and
> how to work with the codebase.

---

## 1. SYSTEM IDENTITY

| Field | Value |
|-------|-------|
| **Name** | Pairs Trading System |
| **Type** | Statistical arbitrage platform for equities pairs trading |
| **Language** | Python 3.13.7 |
| **UI** | Streamlit (port 8501) |
| **Database** | DuckDB (local), SQLAlchemy |
| **Broker** | Interactive Brokers (ib_insync) — paper/live |
| **Data Sources** | FMP API (primary), Yahoo Finance (fallback), IBKR (live) |
| **Scheduling** | APScheduler daemon (4 recurring jobs) |
| **ML** | XGBoost meta-labeling + scikit-learn |
| **Repository** | github.com/benda931/pairs_trading_system |
| **Total Code** | ~170,000 lines across 363 Python files |
| **Tests** | 976 passing (23 test files) |
| **Agents** | 50 registered, autonomous feedback loop |

---

## 2. ARCHITECTURE LAYERS

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 7: DASHBOARD (Streamlit, 18 extracted modules)           │
│  root/dashboard.py (5,886 lines) + 18 dashboard_*.py modules   │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 6: AGENTS (50 agents, feedback loop)                     │
│  agents/ (11 modules) + core/agent_feedback.py                  │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 5: ORCHESTRATION (daily pipeline, scheduler)             │
│  core/orchestrator.py + scripts/run_scheduler_daemon.py         │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4: QUANT ENGINES (12 engines)                            │
│  spread_analytics, risk_analytics, monte_carlo, garch_engine,   │
│  cycle_detector, optimal_exit, factor_attribution,              │
│  correlation_monitor, universe_scanner, execution_algos,        │
│  portfolio_rebalancer, leverage_engine                          │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3: SIGNAL PIPELINE                                       │
│  signal_pipeline.py → regime_engine → threshold_engine →        │
│  signal_quality → intents (EntryIntent/ExitIntent)              │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: PORTFOLIO MANAGEMENT                                  │
│  portfolio/ (allocator, capital, ranking, sizing, exposures,    │
│  risk_ops) + core/portfolio_bridge.py                           │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: DATA & PERSISTENCE                                    │
│  common/data_loader.py, common/data_providers.py,               │
│  core/sql_store.py, common/fmp_client.py                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. DIRECTORY STRUCTURE

```
pairs_trading_system/
├── core/                    # 65 modules, ~58K lines — domain logic
│   ├── contracts.py         # THE source of truth: 14 enums, 19 dataclasses, 6 protocols
│   ├── signal_pipeline.py   # Canonical signal engine (ADR-006)
│   ├── orchestrator.py      # Daily pipeline orchestrator (2,033 lines)
│   ├── agent_feedback.py    # Agent → system action feedback loop
│   ├── spread_analytics.py  # Johansen, EG, ADF/KPSS, Hurst, OU, TLS
│   ├── risk_analytics.py    # VaR (3 methods), CVaR, stress tests
│   ├── monte_carlo.py       # Block bootstrap, DSR, strategy comparison
│   ├── garch_engine.py      # GARCH(1,1) MLE, vol regimes, forecasts
│   ├── cycle_detector.py    # FFT + Haar wavelet + Hilbert phase
│   ├── optimal_exit.py      # OU boundary, composite scoring, trailing stop
│   ├── correlation_monitor.py # CUSUM breaks, pair health, eff-N-bets
│   ├── factor_attribution.py  # Multi-factor OLS, Brinson-Fachler, risk attr
│   ├── universe_scanner.py  # 2-phase discovery, ranked scoring
│   ├── portfolio_rebalancer.py # Risk parity, txn cost opt, execution plan
│   ├── execution_algos.py   # TWAP/VWAP/Iceberg, coordinated pairs exec
│   ├── leverage_engine.py   # Kelly, GARCH vol-target, regime, VIX dampening
│   ├── context_services.py  # Service initialization (extracted from app_context)
│   ├── app_context.py       # Global state (4,124 lines)
│   └── ...                  # 47 more modules
│
├── agents/                  # 11 modules, 50 agents
│   ├── base.py              # BaseAgent, AgentAuditLogger
│   ├── registry.py          # AgentRegistry, dispatch, permissions
│   ├── signal_agents.py     # 4 agents: signal/regime/lifecycle/exit
│   ├── portfolio_agents.py  # 6 agents: allocation/exposure/dd/killswitch
│   ├── research_agents.py   # 8 agents: discovery/validation/spread/regime
│   ├── ml_agents.py         # 7 agents: feature/label/model/meta-label
│   ├── monitoring_agents.py # 7 agents: health/drift/data/incident/alert
│   ├── governance_agents.py # 5 agents: policy/audit/approval/impact
│   ├── gpt_agents.py        # 4 agents: GPT-4o signal/model/strategy/report
│   └── auto_agents.py       # 4 agents: retrain/refresh/optimize/config
│
├── root/                    # 48 modules — Streamlit dashboard
│   ├── dashboard.py         # Main app (5,886 lines, was 18,725)
│   ├── dashboard_*.py       # 18 extracted modules
│   ├── pair_tab.py          # Pair analysis tab
│   ├── optimization_tab.py  # Optimizer tab (13,191 lines)
│   ├── portfolio_tab.py     # Portfolio management
│   ├── risk_tab.py          # Risk monitoring
│   ├── macro_tab.py         # Macro environment
│   └── ...                  # 25 more tab/service modules
│
├── portfolio/               # Portfolio management subsystem
│   ├── allocator.py         # PortfolioAllocator — main cycle engine
│   ├── capital.py           # CapitalManager — pool, sleeves
│   ├── ranking.py           # OpportunityRanker — 7-dimension scoring
│   ├── sizing.py            # SizingEngine — vol-target + scalar stack
│   ├── exposures.py         # ExposureAnalyzer
│   ├── risk_ops.py          # DrawdownManager + KillSwitchManager
│   └── analytics.py         # PortfolioAnalytics
│
├── research/                # Offline research pipeline
│   ├── discovery_pipeline.py
│   ├── pair_validator.py
│   ├── spread_constructor.py  # OLS / Rolling OLS / Kalman
│   ├── stability_analysis.py
│   └── walk_forward.py
│
├── ml/                      # ML platform
│   ├── features/            # 61 feature definitions
│   ├── labels/              # 26 label definitions
│   ├── datasets/            # Temporal splitting, leakage audit
│   ├── models/              # MetaLabel, RegimeClassification, BreakDetection
│   ├── inference/           # ModelScorer (never raises, tiered fallback)
│   ├── registry/            # Champion/Challenger management
│   ├── monitoring/          # PSI drift, model health
│   └── governance/          # Promotion criteria, usage contracts
│
├── common/                  # Shared utilities
│   ├── data_loader.py       # Canonical price loader (Yahoo + FMP + cache)
│   ├── data_providers.py    # Multi-source provider (IBKR, FMP, Yahoo)
│   ├── feature_engineering.py # 50+ feature functions + compute_zscore()
│   ├── config_manager.py    # Config loading, validation, profiles
│   ├── gpt_client.py        # GPT-4o client (cost tracking, rate limiting)
│   ├── optuna_factory.py    # Canonical Optuna study creation
│   └── fmp_client.py        # FMP HTTP client
│
├── scripts/                 # 31 operational scripts
│   ├── run_scheduler_daemon.py  # APScheduler daemon (4 jobs)
│   ├── run_daily_pipeline.py    # CLI pipeline entry
│   ├── run_auto_improve.py      # GPT auto-improvement cycle
│   ├── run_full_alpha.py        # Alpha generation pipeline
│   ├── run_backtest.py          # Realistic backtester
│   ├── train_xgboost_model.py   # XGBoost meta-label training
│   └── setup_scheduler.py       # Windows Task Scheduler
│
├── tests/                   # 23 test files, 976 tests
├── config.json              # Main configuration
├── start_system.bat         # Launch everything
├── stop_system.bat          # Stop everything
└── CLAUDE.md                # AI assistant instructions
```

---

## 4. DAILY PIPELINE (16:15 ET, Automated)

The scheduler daemon runs this pipeline every weekday at 16:15 ET:

```
Step 1: health_check
  → Verify 8 core modules importable
  → Dispatch SystemHealthAgent (WorkflowEngine)

Step 2: data_refresh
  → Download prices for all symbols (Yahoo/FMP)
  → Dispatch DataIntegrityAgent (WorkflowEngine)

Step 3: compute_signals
  → Load prices for all active pairs
  → For each pair:
    ├─ Compute OLS spread + z-score
    ├─ Run SignalPipeline (regime → threshold → quality → intent)
    ├─ Enrich with CycleDetector (dominant period, phase, lookback)
    ├─ Enrich with OptimalExit (OU boundary, dynamic thresholds)
    └─ Enrich with XGBoost ML (meta-label probability)
  → Collect SignalDecisions

Step 4: portfolio_allocation
  → bridge_signals_to_allocator (safety_check + kill-switch)
  → PortfolioRebalancer → execution plan with txn costs

Step 5: signal_agents (with REAL spread data)
  → regime_surveillance → regime_map + shift/broken alerts
  → signal_analyst → quality assessment per pair
  → trade_lifecycle → stale/blocked position alerts
  → exit_oversight → exit/reduce signals

Step 6: risk_agents (with REAL NAV/DD from bus)
  → drawdown_monitor → current_dd_pct, heat_level
  → kill_switch → triggered flag
  → exposure_monitor → violations list
  → capital_budget, derisking, drift_monitoring, alert_aggregation

Step 7: analytics engines
  → risk_analytics → VaR/CVaR/drawdown + alerts on CVaR>3%
  → universe_scanner → new pair discovery + alerts on A+/A
  → correlation_monitor → CUSUM break detection + alerts
  → monte_carlo → Sharpe CI, DSR, P(loss), P(ruin)
  → factor_attribution → alpha/beta/IR vs SPY

Step 8: AGENT FEEDBACK LOOP
  → Read ALL agent outputs
  → Apply rules → generate FeedbackActions
  → Execute: BLOCK_ENTRY / FORCE_EXIT / DELEVERAGE / KILL_SWITCH /
             RETRAIN_MODEL / OPTIMIZE_PARAMS / UPDATE_CONFIG
  → Send Telegram alerts for EMERGENCY/CRITICAL

Step 9: alerts + summary → Telegram + AgentBus
```

---

## 5. QUANT ENGINES — MATHEMATICAL DETAILS

### 5.1 Spread Analytics (core/spread_analytics.py)
- **Engle-Granger**: OLS residuals → ADF test (p<0.05 = cointegrated)
- **Johansen**: Trace statistic on [X,Y] matrix, 90/95/99% critical values
- **ADF**: Augmented Dickey-Fuller with AIC lag selection
- **KPSS**: H0=stationary, reject = non-stationary (complement to ADF)
- **OU Half-Life**: AR(1) fit → θ = -ln(φ) → t½ = ln(2)/θ
- **Hurst**: Rescaled Range (R/S) method, H<0.5 = mean-reverting
- **Variance Ratio**: Lo-MacKinlay VR(q), VR<1 = mean-reverting
- **TLS Hedge Ratio**: SVD-based Deming regression (errors-in-variables)
- **Quality Score**: Weighted composite of stationarity(30%) + MR(30%) + stability(20%) + trading(20%)

### 5.2 Risk Analytics (core/risk_analytics.py)
- **Historical VaR**: Empirical percentile of returns
- **Parametric VaR**: μ + z_α × σ (Gaussian)
- **Cornish-Fisher VaR**: z_cf = z + (z²-1)s/6 + (z³-3z)k/24 - (2z³-5z)s²/36
- **CVaR (ES)**: E[R | R ≤ -VaR] (tail mean)
- **Drawdown**: Peak-to-trough, duration, recovery, Calmar, Sterling
- **Tail Risk**: Skew, kurtosis, Jarque-Bera, Omega, tail ratio
- **6 Stress Scenarios**: 2σ/3σ move, correlation break, liquidity crisis, VIX spike, trending regime

### 5.3 Monte Carlo (core/monte_carlo.py)
- **Block Bootstrap**: Preserves autocorrelation (block_size=21 days)
- **Parametric**: Student-t if |kurtosis|>1, else Gaussian
- **Deflated Sharpe Ratio**: Bailey & Lopez de Prado (2014)
  DSR = Φ((SR - E[max(SR)]) / √((1+½SR²-s·SR+k/4·SR²)/(N-1)))
- **Strategy Comparison**: Paired bootstrap, P(A>B), Holm-Bonferroni

### 5.4 GARCH Engine (core/garch_engine.py)
- **GARCH(1,1)**: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
- **MLE fit**: L-BFGS-B optimizer, constraints α≥0, β≥0, α+β<1
- **EWMA fallback**: σ²_t = λ·σ²_{t-1} + (1-λ)·ε²_t (λ=0.94)
- **Vol regimes**: LOW(<25th pct) / NORMAL / HIGH(>75th) / CRISIS(>95th)
- **Multi-horizon**: σ²_{t+h} = V_L + (α+β)^{h-1}(σ²_{t+1} - V_L)

### 5.5 Cycle Detector (core/cycle_detector.py)
- **FFT**: Hanning window, power spectrum, peak detection (SNR threshold)
- **Wavelet**: Haar basis, 6-level decomposition, energy per scale
- **Phase**: Hilbert transform → instantaneous phase → NEAR_TROUGH/RISING/NEAR_PEAK/FALLING
- **Optimal Lookback**: dominant_period × 1.5

### 5.6 Optimal Exit (core/optimal_exit.py)
- **OU Boundary**: E[z_t] = z_0·exp(-θt), exit when marginal profit from waiting = 0
- **Time-Dependent Threshold**: exit_z rises with holding time (time decay)
- **Composite Score**: z_proximity(35%) + time_pressure(15%) + stop_loss(20%) + max_holding(10%) + profit_taking(10%) + regime_stress(10%)
- **Trailing Stop Optimization**: MC simulation over ATR multipliers × tightening rates

### 5.7 Leverage Engine (core/leverage_engine.py)
- **Kelly**: f* = fraction × (p/a - q/b), default fraction=0.5 (half-Kelly)
- **Vol Target**: leverage = min(target_vol/realized_vol, cap)
- **GARCH Enhanced**: uses forecast vol instead of realized
- **5 Multiplicative Factors**: base × regime × VIX × drawdown × margin
- **Risk Parity**: inverse-vol weighting with per-pair caps

---

## 6. AGENT FEEDBACK LOOP (core/agent_feedback.py)

The feedback loop converts agent outputs into system actions:

### Signal Rules
| Agent Output | Action | Auto-Execute |
|---|---|:---:|
| regime = CRISIS | BLOCK_ENTRY + DELEVERAGE 30% | ✅ |
| regime = BROKEN | BLOCK_ENTRY + DELEVERAGE 50% | ✅ |
| regime = TENSION | ADJUST_THRESHOLD (z×1.3) | ✅ |
| quality = F | BLOCK pair-specific entry | ✅ |
| exit_signals | FORCE_EXIT per pair | ✅ |
| stale > 60 days | FORCE_EXIT | ✅ |

### Risk Rules
| Agent Output | Action | Auto-Execute |
|---|---|:---:|
| drawdown > -10% | DELEVERAGE 50% | ✅ |
| drawdown > -20% | KILL_SWITCH (EXITS_ONLY) | ✅ |
| exposure violation | BLOCK_ENTRY | ✅ |
| kill_switch triggered | HALT_ALL | ✅ |

### Improvement Rules
| Agent Output | Action | Auto-Execute |
|---|---|:---:|
| GPT: retrain | dispatch AutoModelRetrainer | ✅ |
| GPT: optimize | dispatch AutoParameterOptimizer | ✅ |
| optimizer: best_params | dispatch AutoConfigUpdater | ✅ |

---

## 7. CONFIGURATION (config.json)

### Key Strategy Parameters
```json
{
  "strategy": {
    "z_open": 1.52,              // Entry z-score threshold
    "z_close": 1.0,             // Exit z-score threshold
    "max_exposure_per_trade": 0.1,
    "rolling_window": 60,
    "atr_window": 14,
    "use_volatility_adjustment": true
  },
  "filters": {
    "min_correlation": 0.7,
    "min_edge": 1.2,
    "min_half_life": 1,
    "max_half_life": 200
  }
}
```

### Scheduler Configuration
```json
{
  "scheduler_run_time": "16:15",
  "scheduler_timezone": "America/New_York",
  "scheduler_capital": 1000000.0,
  "auto_improve_interval_hours": 6.0,
  "data_refresh_interval_hours": 2.0,
  "scheduler_enabled": true
}
```

---

## 8. HOW TO WORK WITH THIS CODEBASE

### Run the system
```bash
start_system.bat                              # Start API + Dashboard + Scheduler
python scripts/run_scheduler_daemon.py        # Scheduler only
python scripts/run_scheduler_daemon.py --run-now --once  # Run pipeline once
```

### Run tests
```bash
python -m pytest tests/ -x -q                 # All 976 tests
python -m pytest tests/test_quant_engines.py  # Quant engines only
python -m pytest tests/test_agent_feedback.py # Feedback loop only
```

### Key conventions
- **core/contracts.py** is THE source of truth for all types/enums
- **Train/test boundary**: every function takes `train_end` parameter
- **Agents never raise** — return AgentResult with status=FAILED
- **ML never overrides risk** — kill-switch/drawdown are hard vetoes
- **Rejection reasons always explicit** — never fail silently

### Adding new code
- New engine: create in `core/`, add tests, wire into orchestrator
- New agent: subclass BaseAgent, register in registry.py, add test
- New tab: create `root/<name>_tab.py`, add to dashboard.py tab registry
- New feature: add to `common/feature_engineering.py`

---

## 9. KNOWN LIMITATIONS

1. **IBKR not connected** — order router is scaffold, no live execution
2. **dashboard.py still 5,886 lines** — core orchestration can't be further split
3. **optimization_tab.py is 13,191 lines** — needs future splitting
4. **GPT agents disabled by default** — need OPENAI_API_KEY in .env
5. **No intraday data support** — daily bars only
6. **No multi-strategy support** — single pairs-trading strategy

---

## 10. ENVIRONMENT

| Component | Version/Value |
|-----------|---------------|
| Python | 3.13.7 |
| OS | Windows 11 |
| FMP API Key | Configured in .env |
| OpenAI API Key | Configured in .env (optional) |
| Dashboard Port | 8501 |
| API Port | 8000 |
| Database | DuckDB (local file) |
| Scheduler | APScheduler, daily 16:15 ET |

---

*Document generated: April 4, 2026*
*System version: 86 commits, 976 tests, 50 agents*
*Purpose: AI-to-AI transfer of complete system knowledge*
