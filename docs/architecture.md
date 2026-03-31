# Architecture Overview — Pairs Trading System

## System Purpose

A portfolio-level, market-neutral, regime-aware, factor-aware relative-value
statistical arbitrage platform. The primary alpha abstraction is **residual mean
reversion** of cointegrated equity pairs.

---

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
│   Streamlit Dashboard (root/)    Desktop App (root_desktop/)    │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                     Agent Layer (agents/)                        │
│  UniverseDiscovery  PairValidation  SpreadFit  RiskOversight    │
│  Each agent: narrow mandate, typed AgentTask/AgentResult         │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  Research Pipeline (research/)                   │
│     PairValidator → SpreadConstructor → WalkForwardHarness      │
│     ALL parameter estimation uses training data only            │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  Domain Core (core/)                             │
│  contracts.py (types)   signals_engine   optimizer              │
│  fmp_fundamentals       fmp_macro        leverage_engine         │
│  tail_risk              trade_monitor    attribution             │
│  ml_validation          risk_engine      regime_classifier       │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  ML Infrastructure (models/)                     │
│     ModelRegistry   DatasetBuilder   FeatureStore               │
│     Leakage-safe dataset construction + feature engineering      │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  Data Layer (common/)                            │
│     FMPClient (stable-first)   DataProviders (priority routing) │
│     DuckDB/SQLite persistence (core/sql_store.py)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pair Lifecycle State Machine

```
                    ┌──────────────┐
                    │  CANDIDATE   │ ← universe scan produces
                    └──────┬───────┘
                           │ PairValidator.validate()
              ┌────────────┴────────────┐
              │ FAIL                    │ PASS / WARN
              ▼                         ▼
        ┌──────────┐             ┌──────────────┐
        │ REJECTED │             │  VALIDATED   │
        └──────────┘             └──────┬───────┘
                                        │ spread fit + active monitoring
                                        ▼
                                  ┌──────────┐
                                  │  ACTIVE  │◄──────────────────┐
                                  └────┬─────┘                   │ recovered
                                       │                         │
                    ┌──────────────────┼──────────────────────┐  │
                    │                  │                       │  │
                    ▼                  ▼                       ▼  │
              ┌──────────┐      ┌───────────┐          ┌──────────┐
              │ SUSPENDED│      │  PAUSED   │          │  BROKEN  │
              └──────────┘      └───────────┘          └──────────┘
                    │                                        │
                    └────────────────────────────────────────┘
                                        │
                                        ▼
                                  ┌──────────┐
                                  │ RETIRED  │
                                  └──────────┘
```

---

## Research Pipeline

The research pipeline is **strictly offline** and respects a train/test boundary:

```
Prices (full history)
        │
        ├─── [train window] ───► PairValidator.validate()
        │                              │ FAIL → skip
        │                              │ PASS ↓
        │                        SpreadConstructor.fit()
        │                              │
        │                        SpreadDefinition
        │                        (beta, intercept, mean, std)
        │
        ├─── [embargo gap] ──── (no data used)
        │
        └─── [test window] ───► SpreadConstructor.transform()
                                       │
                                  z-score series (test)
                                       │
                                  BacktestEngine
                                       │
                                  FoldResult (PnL, Sharpe, etc.)
```

---

## Spread Construction Methods

| Method | Best for | Adapts to drift |
|--------|----------|----------------|
| Static OLS | Stable structural pairs | No |
| Rolling OLS | Slowly drifting pairs | Partial |
| Kalman Filter | Fast-drifting pairs (ETF vs basket) | Yes |

All methods: parameters estimated from training data only. Test window always
uses the fixed parameters from the last training observation.

---

## Attribution Framework

Each signal/trade gets a **Match Confidence (MC)** score:

```
MC = SDS × (1 - FJS) × (1 - MSS) × (1 - STF)

Where:
  SDS = Signal Definition Score (how clean is the z-score entry?)
  FJS = Fundamental Justification Score (do fundamentals explain the spread?)
  MSS = Macro Sensitivity Score (is the spread macro-driven vs mean-reverting?)
  STF = Short-Term Friction Score (liquidity, bid-ask, market impact)
```

Higher MC = more conviction. MC < 0.3 = avoid trade.

---

## Key Design Decisions

See `docs/adr/` for full Architecture Decision Records.

- **ADR-001**: All domain types in `core/contracts.py` — single source of truth
- **ADR-002**: PairValidator uses explicit hard/soft test doctrine
- **ADR-003**: Spread construction separates fit (training) from transform (any window)
- **ADR-004**: Agents use narrow mandate + typed contracts + audit trails

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data storage | DuckDB (primary), SQLite (Optuna studies) |
| ML optimization | Optuna (multi-objective, persistent) |
| Backtesting | Custom (core/optimization_backtester.py) |
| Dashboard | Streamlit |
| Data provider | FMP (primary), yfinance (fallback), IBKR (live) |
| Statistical tests | statsmodels (ADF, Engle-Granger) |
| Kalman filter | Pure NumPy (no pykalman dependency) |
