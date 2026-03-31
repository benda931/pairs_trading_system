# Pairs Trading System

An institutional-grade **statistical arbitrage research platform** for equities pairs trading.

> **Maturity Level:** Disciplined research platform with comprehensive infrastructure scaffolding.
> Safe for offline research, pair discovery, and exploratory backtesting.
> Not yet safe for live capital deployment, paper trading, or automated execution.

> **Disclaimer:** This repository is for research and education purposes only.
> It is not financial advice. Trading involves substantial risk.

---

## What This System Does

The platform discovers, validates, and backtests mean-reverting equities pairs using
cointegration-based statistical arbitrage. Its north star is a portfolio-level,
market-neutral, regime-aware, factor-aware relative-value system where:

- **Correlation and distance** are used for pair discovery
- **Cointegration, stability, and half-life** are used for validation
- **Residual mean reversion** is the primary alpha abstraction

## What Is Operational Today

| Capability | Status | Module |
|---|---|---|
| Pair discovery (correlation, distance, cluster, cointegration) | **Operational** | `research/` |
| Pair validation (ADF, Hurst, half-life, stability) | **Operational** | `research/` |
| Spread construction (OLS, Rolling OLS, Kalman) | **Operational** | `research/` |
| Z-score backtesting and parameter optimization | **Operational** | `core/`, `root/` |
| Streamlit dashboard (15 tabs) | **Operational** | `root/` |
| FMP / Yahoo Finance data integration | **Operational** | `common/` |

## What Is Infrastructure (Scaffolded, Not Yet Integrated)

| Capability | Status | Module |
|---|---|---|
| Regime-aware signal pipeline | Implemented, not called from backtest path | `core/signal_pipeline.py` |
| Portfolio construction and allocation | Implemented, never receives real signals | `portfolio/` |
| ML platform (features, labels, inference) | Designed and tested, zero models trained | `ml/` |
| Agent orchestration (33 agents) | Registered and tested, never dispatched operationally | `agents/` |
| Governance, audit, surveillance | Implemented, not enforced at runtime | `governance/`, `audit/`, `surveillance/` |
| Runtime control plane | Implemented, no live system exists | `runtime/`, `control_plane/` |

See `CLAUDE.md` "Current Integration Status" for the authoritative status register.

## Architecture Overview

```
pairs_trading_system/
  core/         Domain logic (contracts, backtester, optimizer, signals)
  research/     Offline research pipeline (discovery, validation, spreads, walk-forward)
  portfolio/    Capital allocation, ranking, sizing, risk operations
  ml/           ML infrastructure (features, labels, models, registry, inference)
  agents/       Agent system (33 agents, registry, orchestration)
  common/       Shared utilities (data loading, FMP client, config)
  root/         Streamlit dashboard tabs and CLI
  runtime/      Runtime state management
  control_plane/ Control plane engine
  governance/   Governance policy engine
  audit/        Audit chain infrastructure
  surveillance/ Surveillance engine (12 rules)
  docs/         Architecture docs, ADRs, migration ledger, remediation ledger
  tests/        Test suite (19 modules, 785 tests)
```

## Key Architectural Principle

`core/contracts.py` is the **single source of truth** for all domain types. All enums,
dataclasses, and protocol definitions live there. Import from `core.contracts` everywhere.

## Getting Started

```powershell
# Create and activate virtual environment
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run the dashboard
streamlit run root/dashboard.py

# Run tests
python -m pytest tests/ -v
```

## Documentation

| Document | Audience | Purpose |
|----------|----------|---------|
| `CLAUDE.md` | Contributors, Claude Code | Platform handbook: conventions, extension guides, gotchas |
| `CONTRIBUTING.md` | New contributors | How to extend the platform safely |
| `docs/architecture.md` | All | Platform architecture overview |
| `docs/discovery_methodology.md` | Quant researchers | Research methodology and validation doctrine |
| `docs/signal_architecture.md` | Signal engineers | Signal, regime, lifecycle, quality architecture |
| `docs/portfolio_architecture.md` | Portfolio engineers | Allocation, sizing, risk operations |
| `docs/ml_architecture.md` | ML engineers | ML platform design (scaffold) |
| `docs/agent_architecture.md` | Platform engineers | Agent system design (scaffold) |
| `docs/governance_architecture.md` | Governance stakeholders | Governance, audit, compliance (scaffold) |
| `docs/production_architecture.md` | Operators | Runtime, control plane, monitoring (scaffold) |
| `docs/migration/migration_ledger.md` | Contributors | Deprecated paths and canonical replacements |
| `docs/remediation/remediation_ledger.md` | Reviewers | Known findings with severity and status |
| `docs/adr/` | Architects | 7 Architecture Decision Records |

## Known Limitations

- **Backtest execution timing:** Same-close execution (signals and fills on same bar). See ADR-007.
- **Walk-forward:** Optimization tab uses calendar-segment stability checks, not true walk-forward.
  Use `research/walk_forward.py:WalkForwardHarness` for genuine walk-forward.
- **ML:** Zero models trained. All ML inference returns neutral probability (0.5).
- **Kill-switch:** Two independent subsystems (portfolio and control-plane) not yet synchronized.
- **Governance:** Implemented but not enforced at runtime for operational decisions.

See `docs/remediation/remediation_ledger.md` for the full findings register.
