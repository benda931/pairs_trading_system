# Pairs Trading System

Hedge-fund-grade, modular pairs-trading research and execution platform.
Focus: correlation/cointegration-driven strategies, robust backtesting, optimization,
risk controls, and a Streamlit-based analytics dashboard with optional live trading integration.

> Disclaimer: This repository is for research/education purposes only.
> It is not financial advice. Trading involves substantial risk. Use at your own risk.

---

## Highlights

- **Modular architecture**: separation between research/engines (`core/`), shared infra (`common/`), UI/orchestration (`root/`), and automation (`scripts/`, `agent/`).
- **Backtesting & Optimization**: fund-style metrics, parameter search, batch campaigns, and result persistence.
- **Data → SqlStore**: canonical storage layer for prices, trials, and snapshots (DuckDB/SQLite/SQLAlchemy, depending on config).
- **Live/Paper Trading (optional)**: Interactive Brokers (IBKR) connectivity for market data and execution (when enabled).
- **Dashboards**: Streamlit GUI inspired by professional trading tool UX patterns.

---

## Repository Structure

- `common/` — shared utilities, data helpers, logging, configuration utilities, etc.
- `core/` — quant engines: signals, backtesting, optimization, metrics, risk, macro, ML tooling.
- `root/` — Streamlit UI tabs, CLI entrypoints, orchestration layer (must NOT be imported by `core/`).
- `api/` — API layer (service endpoints / integration surface, if enabled).
- `agent/` — AI/automation agents (upgrade/refactor tools, orchestration helpers).
- `scripts/` — operational scripts (ingestion, campaigns, maintenance, diagnostics).
- `configs/` — configuration templates (JSON/YAML), profiles, examples.
- `studies/` — research notebooks / experiments.
- `tools/` — dev tooling and utilities.

---

## Quickstart

### 1) Create & activate a virtual environment (Windows)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
