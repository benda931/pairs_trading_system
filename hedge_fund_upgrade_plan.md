# Hedge-Fund Upgrade Plan
_Project root: C:\Users\omrib\OneDrive\Desktop\pairs_trading_system_

## Phases
### 1. Architecture & Environment Modes  (`phase_architecture_env`)

Introduce explicit environment modes (dev/research/paper/live) and centralize upgrade orchestration.

**Milestones:**
- Define global upgrade phases and stable ids
- Make upgrade commands deterministic and idempotent across runs

### 2. Data Layer, Backfills & Source-of-Truth  (`phase_data_layer`)

Harden the data ingestion and storage pipeline so all components share a single, auditable source-of-truth for prices, signals, and metadata.

**Milestones:**
- Standardize SqlStore schemas for prices, signals, positions, and experiments with explicit versions and indices
- Unify data access via sql_price_loader and market_data_router, deprecating ad-hoc loaders and scripts
- Implement robust backfill and incremental ingestion from IBKR and Yahoo with gap detection and anomaly logging
- Add data quality scoring and freshness checks surfaced through data_quality and dashboard_service
- Define clear history-length and intraday vs EOD retention policies, enforced at the SqlStore level

### 3. Signals, Portfolio Construction & Risk Overlays  (`phase_signals_portfolio`)

Separate signal generation, portfolio construction, and risk overlays while ensuring shared parameters across research, backtest, and live.

**Milestones:**
- Refine signals_engine and signal_generator to expose a clear Signal API (inputs, outputs, metadata, config snapshot)
- Introduce portfolio construction layer that consumes signals_engine outputs and produces target positions with sizing rules
- Integrate risk_engine and risk_parity as overlays on target portfolios (max leverage, exposure caps, kill-switches, drawdown gates)
- Persist signals, portfolios, and risk states in SqlStore with full metadata (env, config, git_rev, data snapshot)
- Ensure optimization_backtester, optimizer, and live flows reuse the same signal and risk parameters with explicit environment overrides only

### 4. Execution Router, Broker Abstraction & Safety  (`phase_execution_broker`)

Build a robust execution layer that cleanly separates target intent from fill state, supports paper vs live, and enforces capital safety.

**Milestones:**
- Harden ib_order_router into a broker-agnostic ExecutionRouter with a single interface for target positions and orders
- Implement explicit paper vs live broker profiles with circuit breakers, rate-limits, and retry/backoff policies
- Model order lifecycle clearly (submitted, partially filled, filled, rejected, cancelled) and persist all events in SqlStore
- Add safeguards for order sizing, price bounds, and slippage controls, including pre-trade risk checks from risk_engine
- Enable deterministic simulation and replay modes that use the same execution logic on historical data

### 5. Unified Web App, Dashboards & Live Controls  (`phase_web_app_ux`)

Expose a single, coherent Streamlit-based web app with distinct sections for research, optimization, monitoring, and live trading, all backed by the shared engine.

**Milestones:**
- Create a unified entry-point dashboard with clear environment indicator and risk status banner
- Split UI into tabs or sections (Universe, Research, Optimization, Backtests, Live, Monitoring) using shared AppContext
- Refactor existing Streamlit tabs to use ui_helpers, dashboard_service, and dashboard_models instead of ad-hoc logic
- Implement live trading control panel (paper/live) that manages strategies, allocations, and kill-switches with explicit confirmations
- Provide stateful session handling and profile switching so users can move between dev/research/paper/live with consistent behaviour

### 6. Observability, Operations & Code Quality  (`phase_observability_ops`)

Add observability, health checks, and safety rails around automated upgrades.

**Milestones:**
- Introduce health checks after major upgrade phases
- Add snapshotting of code/configs prior to applying upgrades

### 7. Testing, Simulation, Deployment & AI Agents  (`phase_testing_deployment_ai`)

Build realistic tests, simulation paths, and an upgrade agent workflow that supports safe, incremental deployment from dev to live.

**Milestones:**
- Add fast, deterministic tests for SqlStore, signals_engine, risk_engine, execution router, and key dashboard flows
- Implement end-to-end simulation and replay pipelines that run live-like flows on historical data with the same configs
- Define deployment profiles and checklists for moving strategies from dev to research to paper to live with health gates
- Integrate AI-based upgrade agents as first-class tools that propose localized, safe refactors using AppContext and shared services
- Persist config and environment snapshots alongside backtests and live runs to guarantee reproducibility and auditability

## Files & Tasks
### `__backup_dupes\optimization_tab.backup_20251102_131113.py`
- Role: legacy backup of the optimization Streamlit tab; reference for migrating to unified research UI
- Priority: `medium`
- Categories: web_ui, research, backtest, infra
- Phases: phase_web_app_ux, phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Audit differences vs active optimization tab and document which behaviours must be preserved
- Extract any non-duplicated logic into shared research/optimization helpers used by the main UI
- Clearly mark this file as deprecated and exclude it from production entry points
- Plan eventual removal once all logic is moved into shared modules and tested

### `__backup_dupes\optimization_tab.backup_quotes_fix.py`
- Role: experimental backup of optimization tab with quote-handling fixes; source for robust data UI
- Priority: `medium`
- Categories: web_ui, research, backtest, data_ingest
- Phases: phase_data_layer, phase_web_app_ux, phase_testing_deployment_ai

**Tasks:**
- Identify quote-handling fixes that are missing from the main optimization tab or data loaders
- Move any durable fixes into market_data_router/price_loader instead of keeping UI-specific hacks
- Tag this module as backup-only and remove from Streamlit navigation and imports
- Schedule deletion after confirming functionality is covered by core data layer tests

### `check_duckdb.py`
- Role: simple DuckDB connectivity and schema smoke-test script
- Priority: `medium`
- Categories: infra, data_ingest, tests
- Phases: phase_data_layer, phase_observability_ops, phase_testing_deployment_ai

**Tasks:**
- Refactor into a reusable DuckDB health-check function callable from AppContext
- Extend checks to validate core tables, indices, and recent data freshness
- Integrate with observability layer (logs/health status) instead of ad-hoc stdout prints
- Guard script execution with clear environment/profile selection

### `common\__init__.py`
- Role: package initializer for shared common utilities used across the system
- Priority: `low`
- Categories: infra, utils
- Phases: phase_architecture_env

**Tasks:**
- Export only stable public symbols (e.g. helpers, typing_compat) to avoid accidental tight coupling
- Add lightweight package-level docstring describing common module responsibilities
- Ensure no side effects or heavy imports occur at package import time

### `common\advanced_metrics.py`
- Role: library of advanced statistical metrics for pairs signals, risk, and analytics
- Priority: `high`
- Categories: research, backtest, risk, utils
- Phases: phase_signals_portfolio, phase_testing_deployment_ai, phase_architecture_env

**Tasks:**
- Identify core metrics used in live signals and expose a stable, typed API for them
- Add deterministic behaviour (e.g. seeds, numerical tolerances) and basic unit-style doctests
- Optimize hot paths with NumPy vectorization and avoid unnecessary copies for large universes
- Tag pure-analytics functions vs ones that depend on live configuration or state
- Improve error handling and input validation for NaNs, missing data, and misaligned indices

### `common\automl_tools.py`
- Role: AutoML and analytics utilities for model selection, feature ranking, and meta-optimization
- Priority: `medium`
- Categories: research, backtest, infra
- Phases: phase_signals_portfolio, phase_testing_deployment_ai, phase_architecture_env

**Tasks:**
- Standardize configuration via central config_manager instead of inline constants
- Ensure all optimization routines use deterministic seeds from a shared seed hub
- Split heavy training functions from lightweight helpers to keep UI responsive
- Add minimal tests covering a representative AutoML workflow on sample data

### `common\config_manager.py`
- Role: central configuration manager for environments, profiles, and feature flags
- Priority: `high`
- Categories: infra, risk, live_trading, web_ui
- Phases: phase_architecture_env, phase_observability_ops, phase_testing_deployment_ai

**Tasks:**
- Define Pydantic v2 models for global, environment, and strategy-level configs with explicit schemas
- Make environment/profile (dev/research/paper/live) explicit and queryable system-wide
- Add support for config snapshots (including git_rev) to be stored alongside results and runs
- Introduce safe defaults and validation for risk-related parameters and broker settings
- Ensure no direct env-var reads outside this module; expose a single AppContext-friendly API

### `common\data_loader.py`
- Role: legacy-compatible data loading façade bridging various providers and local storage
- Priority: `high`
- Categories: data_ingest, research, backtest, live_trading, infra
- Phases: phase_data_layer, phase_architecture_env, phase_observability_ops

**Tasks:**
- Refactor to delegate actual I/O to market_data_router and sql_price_loader while keeping backward compatibility
- Make history-length, frequency, and asset-universe policies explicit and configurable
- Implement robust gap detection, anomaly logging, and data-quality scoring hooks
- Ensure consistent behaviour across research/backtest/live by parameterizing environment instead of hard-coding
- Add caching and DuckDB/Parquet integration aligned with the central SqlStore

### `common\data_providers.py`
- Role: unified abstraction over external market data vendors and broker feeds
- Priority: `high`
- Categories: data_ingest, live_trading, research, infra
- Phases: phase_data_layer, phase_execution_broker, phase_observability_ops

**Tasks:**
- Define clear provider interfaces (sync/async) for historical and real-time data retrieval
- Centralize vendor-specific configs and credentials via config_manager and AppContext
- Add retry/backoff and error-classification logic with structured logging
- Expose data freshness and latency metrics for observability dashboards
- Ensure behaviour is deterministic in replay/simulation modes vs true live streaming

### `common\feature_engineering.py`
- Role: feature construction module for signals, clustering, and regime detection
- Priority: `medium`
- Categories: research, backtest, risk
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Group features into reusable, documented pipelines suited for pairs trading and macro overlays
- Standardize feature configurations and random seeds via config_manager
- Optimize rolling-window and cross-sectional operations for large universes
- Add basic tests validating feature outputs on synthetic time series

### `common\fundamental_loader.py`
- Role: fundamental and index/ETF data ingestion module supporting factor and macro-aware signals
- Priority: `medium`
- Categories: data_ingest, research, backtest
- Phases: phase_data_layer, phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Align schemas and keys with SqlStore and market_data_router for consistent joins with price data
- Make universe selection, point-in-time handling, and history length explicit and configurable
- Add gap and anomaly checks for fundamentals (e.g. stale or missing filings)
- Expose a thin, typed API used by signal_generator and macro modules

### `common\helpers.py`
- Role: shared helper functions used across the codebase (dates, logging glue, small utilities)
- Priority: `medium`
- Categories: utils, infra
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Identify frequently used helpers and ensure they are side-effect free and well-typed
- Deprecate or relocate any heavy/duplicated logic to more appropriate dedicated modules
- Add small doctest-style examples for non-trivial helpers
- Ensure no environment-specific behaviour leaks into generic helpers

### `common\json_safe.py`
- Role: safe JSON encoding utilities for complex objects (e.g. configs, metrics, model params)
- Priority: `low`
- Categories: utils, infra
- Phases: phase_architecture_env, phase_observability_ops

**Tasks:**
- Guarantee that all core domain objects (configs, signals, portfolios) can be serialized deterministically
- Integrate with logging and artifact-writing helpers for consistent JSON output
- Add small tests for edge cases (NaN, datetimes, decimals)

### `common\live_pair_store.py`
- Role: DuckDB-backed store for live pair profiles, states, and metadata
- Priority: `high`
- Categories: live_trading, risk, data_ingest, infra
- Phases: phase_data_layer, phase_signals_portfolio, phase_execution_broker, phase_observability_ops

**Tasks:**
- Define explicit schemas and indices for live pair profiles, including environment and run identifiers
- Implement transactional updates and safe concurrency patterns for live reads/writes
- Persist signal, risk, and execution metadata to make live decisions fully auditable
- Expose simple health and freshness checks for monitoring dashboards
- Align API and models with live_profiles and core.app_context for consistent access

### `common\live_profiles.py`
- Role: typed data models and contracts for live pair profiles and their risk settings
- Priority: `high`
- Categories: live_trading, risk, infra
- Phases: phase_architecture_env, phase_signals_portfolio, phase_execution_broker

**Tasks:**
- Express profiles as Pydantic models with explicit fields for environment, sizing, and risk limits
- Ensure backward-compatible migration paths when models evolve (versioning, defaults)
- Integrate with config_manager and live_pair_store as the single source of truth for live pairs
- Add validation hooks enforcing basic risk constraints at profile load time
- Document how profiles map to signals, portfolios, and execution engines

### `common\macro_adjustments.py`
- Role: module that applies macro-based adjustments to pair signals and exposures
- Priority: `medium`
- Categories: research, risk, backtest
- Phases: phase_signals_portfolio, phase_observability_ops

**Tasks:**
- Clarify and document adjustment rules (e.g. volatility regimes, macro tilts) as configuration
- Decouple pure macro-signal computation from application to pair portfolios
- Ensure adjustments are applied consistently in backtest and live via shared pipelines
- Add diagnostics to explain how macro adjustments impact sizing and risk metrics

### `common\macro_factors.py`
- Role: ingestion and representation of macro factor time series
- Priority: `medium`
- Categories: data_ingest, research, risk
- Phases: phase_data_layer, phase_signals_portfolio, phase_observability_ops

**Tasks:**
- Standardize macro factor identifiers, frequencies, and calendars with the main price database
- Add quality checks and anomaly detection for macro series (e.g. structural breaks)
- Expose a small API for retrieving aligned macro panels for feature_engineering and macro_sensitivity
- Document configuration for data vendors, history length, and update schedules

### `common\macro_sensitivity.py`
- Role: compute macro sensitivity of pairs/strategies for risk and regime-aware sizing
- Priority: `medium`
- Categories: research, risk, backtest
- Phases: phase_signals_portfolio, phase_observability_ops, phase_testing_deployment_ai

**Tasks:**
- Define robust regression/estimation routines for macro betas with proper diagnostics
- Integrate outputs with risk overlays and reporting (e.g. macro exposure dashboards)
- Ensure sensitivity calculations are reproducible and documented with input configs
- Add tests using synthetic data to validate sign and magnitude of estimated betas

### `common\market_data_router.py`
- Role: smart router that selects and orchestrates market data sources (SQL, vendors, brokers)
- Priority: `high`
- Categories: data_ingest, live_trading, infra
- Phases: phase_data_layer, phase_execution_broker, phase_observability_ops

**Tasks:**
- Design a pluggable routing policy based on environment, asset class, and latency requirements
- Implement unified interfaces for bars, quotes, and snapshots with consistent schemas
- Add caching, batching, and rate-limiting controls for heavy queries and live streams
- Emit structured logs and metrics about data source selection, latency, and failures
- Integrate with sql_price_loader, data_providers, and data_loader as the central data entry point

### `common\matrix_helpers.py`
- Role: matrix and linear-algebra utilities for correlations, covariance, and risk calculations
- Priority: `medium`
- Categories: research, backtest, risk, utils
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Profile and optimize core operations (e.g. covariance, inverses) for large universes
- Ensure numerical stability and robust handling of singular or near-singular matrices
- Provide small, well-documented functions instead of monolithic utilities where possible
- Add targeted tests comparing results to reference implementations (e.g. NumPy/ SciPy)

### `common\portfolio_loader.py`
- Role: loader for live portfolios and equity curves from broker or SqlStore
- Priority: `medium`
- Categories: data_ingest, live_trading, backtest, risk
- Phases: phase_data_layer, phase_execution_broker, phase_observability_ops

**Tasks:**
- Define clear schemas for positions, PnL, and equity curves aligned with dashboard_models
- Support multiple environments (paper/live) and brokers via the central broker abstraction
- Expose convenient accessors for portfolio snapshots used in risk and monitoring UIs
- Add sanity checks for stale or inconsistent portfolio data

### `common\price_loader.py`
- Role: legacy-compatible price loader wrapper that forwards to the new Sql/market data stack
- Priority: `high`
- Categories: data_ingest, research, backtest, live_trading
- Phases: phase_data_layer, phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Implement a thin adapter over market_data_router and sql_price_loader to avoid duplicated logic
- Preserve existing public function signatures while internally delegating to the new stack
- Add environment-aware behaviour (e.g. simulation vs live) driven by config_manager
- Introduce basic caching and validation of returned price series

### `common\risk_Helpers.py`
- Role: risk helpers providing sizing, limits, and overlay plumbing for macro/pairs engines
- Priority: `high`
- Categories: risk, live_trading, backtest
- Phases: phase_signals_portfolio, phase_execution_broker, phase_observability_ops

**Tasks:**
- Clarify and centralize risk constraints (max leverage, exposure per pair, kill-switches)
- Ensure all helpers are parameterized by environment and strategy profile
- Add deterministic risk calculations and unit tests for key sizing and limit functions
- Integrate with live_profiles and dashboard_models to surface risk state in the UI

### `common\signal_generator.py`
- Role: core hedge-fund-grade signal engine for pairs selection, spreads, and trading signals
- Priority: `high`
- Categories: research, backtest, live_trading, risk
- Phases: phase_signals_portfolio, phase_data_layer, phase_testing_deployment_ai

**Tasks:**
- Separate signal computation, portfolio construction, and risk overlays into clear submodules
- Parameterize all behaviour via config_manager and live_profiles, not global variables
- Ensure identical logic can run in backtest, paper, and live modes with environment-aware overrides only where required
- Persist generated signals and metadata (inputs, configs, git_rev) via live_pair_store/SqlStore
- Add hooks for explainability (e.g. feature attributions or diagnostics) for top signals

### `common\sql_price_loader.py`
- Role: high-performance price loader from SqlStore/DuckDB for all environments
- Priority: `high`
- Categories: data_ingest, research, backtest, live_trading
- Phases: phase_data_layer, phase_testing_deployment_ai, phase_observability_ops

**Tasks:**
- Define canonical price table schemas (fields, indices) and enforce them on read
- Support intraday vs end-of-day modes with explicit parameters and defaults
- Add robust gap detection, anomaly flags, and data-quality scores per time series
- Optimize queries for large universes and integrate with market_data_router
- Expose replay/simulation capabilities over historical data for live-like testing

### `common\stat_tests.py`
- Role: statistical tests for time-series characteristics (e.g. stationarity, cointegration)
- Priority: `high`
- Categories: research, backtest, risk
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Standardize test interfaces and return types (p-values, diagnostics) with strong typing
- Ensure tests are numerically robust and deterministic with explicit random seeds where used
- Document recommended defaults for pairs-trading workflows (e.g. Engle–Granger settings)
- Add focused tests that validate behaviour on synthetic stationary and non-stationary series

### `common\test_helpers.py`
- Role: shared helpers for unit and integration tests across modules
- Priority: `low`
- Categories: tests, utils
- Phases: phase_testing_deployment_ai

**Tasks:**
- Provide factories for synthetic time series, portfolios, and configs with deterministic seeds
- Avoid any dependency on live external services; use local fixtures or stubs only
- Document usage patterns for new tests to keep them fast and reliable

### `common\test_matrix_helpers_and_advanced_metrics.py`
- Role: test suite covering matrix_helpers and advanced_metrics functionality
- Priority: `medium`
- Categories: tests, research, backtest
- Phases: phase_testing_deployment_ai

**Tasks:**
- Expand coverage to critical metrics and edge cases relevant for live risk/signals
- Ensure tests run quickly and deterministically with controlled random seeds
- Align assertions with documented contracts in advanced_metrics and matrix_helpers
- Integrate with the central test_helpers for data generation

### `common\typing_compat.py`
- Role: shared typing aliases and compatibility helpers for consistent type hints
- Priority: `low`
- Categories: infra, utils
- Phases: phase_architecture_env

**Tasks:**
- Consolidate common type aliases (PriceFrame, SignalDF, ProfileID, etc.) used across modules
- Ensure no runtime-heavy imports occur at module import time
- Document which aliases are stable and safe for external modules to depend on

### `common\ui_helpers.py`
- Role: unified Streamlit UI helper layer for layout, styling, and state management
- Priority: `high`
- Categories: web_ui, infra
- Phases: phase_web_app_ux, phase_architecture_env, phase_observability_ops

**Tasks:**
- Provide a namespaced key-generation helper to avoid Streamlit key collisions
- Expose standard components for environment banners, risk status, and action confirmations
- Ensure all widgets persist state via st.session_state in a consistent pattern
- Abstract away repetitive layout code (tabs, cards) used across research and live dashboards
- Add lightweight tests or manual check routines for key UI flows

### `common\utils.py`
- Role: general-purpose utilities shared across the system (date/time, paths, small math)
- Priority: `medium`
- Categories: utils, infra
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Audit and de-duplicate functions that overlap with helpers, matrix_helpers, or json_safe
- Add type hints and docstrings to all public functions used outside this module
- Ensure no environment-specific logic or configuration is hard-coded here
- Introduce small tests for non-trivial utility functions

### `common\zoom_storage.py`
- Role: shared Optuna storage and configuration layer for optimization studies
- Priority: `medium`
- Categories: infra, research, backtest
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Standardize Optuna study creation with deterministic seeds and common pruner/sampler configs
- Integrate with SqlStore/DuckDB for persistent study storage and metadata
- Attach config snapshots and git_rev to each study for reproducibility
- Provide helpers to summarize and export study results to dashboards

### `configs\__init__.py`
- Role: namespace package for configuration schemas and presets
- Priority: `low`
- Categories: infra
- Phases: phase_architecture_env

**Tasks:**
- Add a short docstring describing configuration package structure and usage
- Ensure it does not perform any imports with side effects
- Optionally expose stable aliases to core config models defined in config_manager

### `core/dashboard_service.py`
- Role: Orchestrator for dashboards and views over research, backtest, and live trading state using shared services.
- Priority: `high`
- Categories: web_ui, api, research, backtest, live_trading, infra
- Phases: phase_architecture_env, phase_web_app_ux, phase_observability_ops

**Tasks:**
- Unify dashboard endpoints for research, backtest, and live views using AppContext profiles
- Refactor to separate data retrieval from UI rendering, returning pure data models
- Integrate SqlStore, signals_engine, and risk_engine to provide coherent state panels
- Add explicit environment and profile indicators to all views and API methods
- Expose health, data freshness, and risk summary widgets for phase_observability_ops

### `core/data_quality.py`
- Role: Central data quality engine scoring and monitoring market and macro data across the system.
- Priority: `high`
- Categories: data_ingest, infra, research, backtest, live_trading
- Phases: phase_data_layer, phase_observability_ops, phase_testing_deployment_ai

**Tasks:**
- Define canonical data quality metrics and scoring for prices, volumes, and macro series
- Integrate with SqlStore to persist data quality snapshots per symbol, timeframe, and environment
- Implement gap/anomaly detection hooks for ib_data_ingestor and yf_loader
- Expose data quality summaries to dashboard_service and risk_engine
- Make thresholds and remediation policies configurable via params and Pydantic models

### `core/distributions.py`
- Role: Central Optuna distribution and search-space factory for optimization pipelines.
- Priority: `medium`
- Categories: infra, research, backtest, utils
- Phases: phase_architecture_env, phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Standardize distribution creation using ranges and params as single sources of truth
- Ensure deterministic, seeded behavior for all stochastic distributions
- Validate requested ranges and raise structured exceptions for invalid configs
- Expose helper functions used by optimizer, full_parameter_optimization, and meta_optimization

### `core/exceptions.py`
- Role: Shared exception hierarchy for data, optimization, risk, and execution components.
- Priority: `medium`
- Categories: infra, utils
- Phases: phase_architecture_env, phase_observability_ops

**Tasks:**
- Define clear base exception classes for data, config, risk, optimization, and execution domains
- Map external library and broker errors into structured internal exceptions
- Ensure all core modules import and use these exceptions instead of ad-hoc ones
- Attach environment, profile, and run_id metadata to critical exceptions where possible

### `core/fair_value_advisor.py`
- Role: High-level advisor translating fair value outputs into trade recommendations and commentary.
- Priority: `medium`
- Categories: research, backtest, live_trading, risk
- Phases: phase_signals_portfolio, phase_observability_ops

**Tasks:**
- Normalize advisor interface to consume fair_value_engine outputs and market state
- Log rationale and key drivers behind each recommendation for explainability
- Integrate advisor outputs with signals_engine as one input source
- Persist advisor decisions and configs in SqlStore for audit and replay

### `core/fair_value_config.py`
- Role: Configuration layer for the fair value engine, with environment-aware profiles.
- Priority: `medium`
- Categories: infra, research, live_trading
- Phases: phase_architecture_env, phase_signals_portfolio

**Tasks:**
- Refactor configs into Pydantic models with validation and helpful error messages
- Centralize fair value defaults and parameter ranges referencing params and ranges
- Support explicit env/profile overrides for research, paper, and live modes
- Add config snapshot helpers for storing fair value settings alongside runs

### `core/fair_value_engine.py`
- Role: Core fair value computation engine for pairs, feeding signals and risk overlays.
- Priority: `high`
- Categories: research, backtest, live_trading, risk
- Phases: phase_signals_portfolio, phase_data_layer, phase_testing_deployment_ai

**Tasks:**
- Ensure deterministic fair value calculations given inputs and config snapshots
- Integrate with SqlStore for reading prices and persisting fair value results
- Align parameterization with fair_value_config, params, and ranges
- Expose explainability hooks (decomposition of drivers, sensitivities) for dashboards
- Provide a unified API for use by signals_engine, fair_value_advisor, and backtests

### `core/fair_value_optimizer_v2.py`
- Role: Fair value specific optimization routines using Optuna and shared distributions.
- Priority: `medium`
- Categories: research, backtest, infra
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Refactor to use core/distributions for all search spaces and ranges
- Make study seeding and sampler configuration deterministic and profile-driven
- Persist optimization results and configs in SqlStore for later analysis
- Align objective metrics with metrics utilities and optimization_backtester outputs

### `core/feature_selection.py`
- Role: Feature selection utilities for ML-based signal and ranking models.
- Priority: `medium`
- Categories: research, backtest, utils
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Ensure feature selection routines are stateless, typed, and reproducible with seeded randomness
- Integrate with ml_analysis for model-based importance and selection
- Persist selected feature sets and metadata to SqlStore for reproducibility
- Expose simple APIs for optimizer and meta_optimization to reuse

### `core/full_parameter_optimization.py`
- Role: End-to-end optimization orchestration across signals, risk, and execution parameters.
- Priority: `high`
- Categories: research, backtest, infra
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Unify optimization workflow around AppContext, SqlStore, and shared distributions
- Include risk_engine constraints and capital usage in optimization objectives
- Support multi-env profiles (research, paper, live-sim) with consistent configs
- Persist all results, configs, and git_rev snapshots for audit and replay
- Add fast smoke-test path for CI-style validation of key optimization flows

### `core/fv_extensions.py`
- Role: Auxiliary helpers and extensions for the fair value engine and advisor.
- Priority: `medium`
- Categories: research, backtest, utils
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Consolidate fair value helper functions and remove duplicated logic
- Add clear docstrings and type hints for all public helpers
- Guard experimental or heavy features behind explicit feature flags
- Ensure extensions do not bypass core fair_value_engine interfaces

### `core/ib_data_ingestor.py`
- Role: Robust historical and intraday data ingestor from IBKR into the central SqlStore.
- Priority: `high`
- Categories: data_ingest, infra, research, backtest, live_trading
- Phases: phase_data_layer, phase_execution_broker, phase_observability_ops

**Tasks:**
- Implement resilient IBKR data fetching with retries, backoff, and rate limiting
- Standardize schemas and write paths into SqlStore with environment tagging
- Enforce history length, intraday vs EOD policies, and retention rules
- Integrate with data_quality for gap, anomaly, and stale-data detection
- Add a simulation/replay mode that reads from stored data instead of IBKR live

### `core/ib_order_router.py`
- Role: Central IBKR order routing and execution safety layer for paper and live trading.
- Priority: `high`
- Categories: live_trading, risk, infra
- Phases: phase_execution_broker, phase_observability_ops, phase_architecture_env

**Tasks:**
- Separate desired target positions from actual orders, fills, and pending state
- Implement idempotent order submission with retries, backoff, and circuit breakers
- Integrate with risk_engine for pre-trade checks and exposure limits
- Add explicit paper vs live configuration profiles and safety guards
- Persist all order intents, confirmations, and rejections to SqlStore for audit

### `core/index_fundamentals.py`
- Role: Fundamental data engine for indices/ETFs used in macro and pairs analysis.
- Priority: `medium`
- Categories: data_ingest, research, backtest, risk
- Phases: phase_data_layer, phase_signals_portfolio

**Tasks:**
- Align fundamental data schemas and keys with SqlStore conventions
- Implement refresh/TTL policies and backfill logic for missing fundamentals
- Expose factorized fundamentals (value, quality, growth) to signals_engine and pair_ranking
- Persist ingestion metadata and quality scores for audit and diagnostics

### `core/macro_data.py`
- Role: Macro time series ingestion and caching layer feeding macro_engine and features.
- Priority: `medium`
- Categories: data_ingest, infra, research
- Phases: phase_data_layer, phase_observability_ops

**Tasks:**
- Standardize macro series identifiers, frequencies, and schema for SqlStore
- Implement cache/TTL adapters with clear invalidation rules
- Integrate with external macro providers and handle retries/failover
- Record missing, revised, or delayed macro data events for diagnostics

### `core/macro_engine.py`
- Role: Macro profile engine computing macro overlays and regimes for pairs trading.
- Priority: `high`
- Categories: research, backtest, live_trading, risk
- Phases: phase_signals_portfolio, phase_observability_ops, phase_testing_deployment_ai

**Tasks:**
- Define a macro profile model combining macro_data, regimes, and sensitivities
- Integrate macro overlays with signals_engine and risk_engine as risk modifiers
- Support backtest and live evaluation of macro-aware strategies with shared configs
- Persist macro overlay states and decisions into SqlStore for replay and explainability
- Expose macro regime and overlay summaries to dashboard_service

### `core/macro_features.py`
- Role: Feature construction utilities for macro-driven factors used in signals and risk.
- Priority: `medium`
- Categories: research, backtest, risk
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Define a canonical set of macro features derived from macro_data and index_fundamentals
- Ensure feature computation is deterministic and aligned across backtest and live
- Expose configurable feature sets via params and fair_value_config where relevant
- Persist macro feature metadata and transformations for traceability

### `core/meta_optimization.py`
- Role: Meta-optimization utilities for tuning optimization processes and hyper-parameters.
- Priority: `medium`
- Categories: research, backtest, infra
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Leverage central distributions and ranges for meta-parameter search
- Define meta-objectives (robustness, stability across regimes) using metrics utilities
- Ensure deterministic seeding and logging of all meta-optimization runs
- Store meta-optimization results and configs in SqlStore for later analysis

### `core/meta_optimizer.py`
- Role: Orchestrator for running multiple optimizations and aggregating their results.
- Priority: `medium`
- Categories: research, backtest, infra
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Coordinate multiple optimizer/full_parameter_optimization runs under AppContext
- Support profile-based meta-runs (different universes, regimes, risk settings)
- Aggregate and persist summary statistics and best configurations into SqlStore
- Expose a simple interface for agents or CLI tools to trigger meta-optimization jobs

### `core/metrics.py`
- Role: Shared metrics library for performance, risk, and optimization evaluation.
- Priority: `medium`
- Categories: research, backtest, risk, utils
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Consolidate performance and risk metric implementations used across backtests and live
- Ensure metric functions are pure, typed, and consistent in sign/scale conventions
- Add advanced risk-adjusted metrics (Sharpe, Sortino, Calmar, drawdowns, tail risk)
- Include simple doctests or unit-style checks for core calculations

### `core/ml_analysis.py`
- Role: Machine learning analysis toolkit for pairs selection, signals, and diagnostics.
- Priority: `medium`
- Categories: research, backtest, utils
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Modularize heavy ML routines into smaller, testable functions with clear inputs/outputs
- Integrate SHAP or permutation importance with graceful fallbacks when deps missing
- Enforce reproducible ML experiments via centralized random seeds and config snapshots
- Persist key ML artifacts and analysis outputs into SqlStore for dashboards

### `core/optimization_backtester.py`
- Role: Backtesting engine tuned for evaluating optimization candidates under realistic assumptions.
- Priority: `high`
- Categories: backtest, research, risk
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Align backtest flow with signals_engine, risk_engine, and optimizer parameter sets
- Model realistic trading costs, slippage, and execution constraints
- Support walk-forward and rolling-window evaluations with consistent configs
- Persist standardized backtest results (PnL, risk stats, trades) into SqlStore
- Expose summary views consumable by dashboard_service and meta_optimizer

### `core/optimizer.py`
- Role: Core pairs-trading optimizer coordinating signals, parameters, and evaluation.
- Priority: `high`
- Categories: research, backtest, infra
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Centralize optimization entrypoints and remove duplicate or legacy flows
- Use ranges and distributions as the single source of parameter search spaces
- Integrate tightly with optimization_backtester for candidate evaluation
- Persist optimization logs, trials, and best configs using SqlStore and json_safe
- Ensure optimizer behavior is deterministic per profile and seed

### `core/pair_ranking.py`
- Role: Canonical pair scoring and ranking engine used for universe selection and monitoring.
- Priority: `high`
- Categories: research, backtest, live_trading, risk
- Phases: phase_signals_portfolio, phase_data_layer

**Tasks:**
- Define a standard pair score that combines statistical, liquidity, and risk dimensions
- Integrate data_quality, macro_features, and index_fundamentals into ranking inputs
- Persist ranked universes and scores in SqlStore tagged by profile and timestamp
- Expose ranking explainability (top drivers per pair) for dashboards and audits
- Parameterize ranking weights and criteria via params and ranges

### `core/pair_recommender.py`
- Role: Recommendation engine suggesting candidate pairs and parameter presets for research UI.
- Priority: `high`
- Categories: research, web_ui, api, backtest
- Phases: phase_signals_portfolio, phase_web_app_ux

**Tasks:**
- Integrate with pair_ranking and pairs_universe as primary inputs
- Expose a clean API for dashboard_service to request recommendations
- Support filtering by risk profile, sector, liquidity, and environment
- Log and persist recommendation rationales and configs to SqlStore
- Provide hooks to evaluate recommended pairs via optimization_backtester

### `core/pairs_universe.py`
- Role: Single source of truth for pairs universes across research, backtest, and live.
- Priority: `high`
- Categories: infra, data_ingest, research, backtest, live_trading, risk
- Phases: phase_architecture_env, phase_data_layer, phase_signals_portfolio

**Tasks:**
- Define a canonical pairs universe schema compatible with SqlStore and JSON files
- Integrate with SqlStore to persist and version universes across environments
- Support profile-specific universes (dev, research, paper, live) with validation
- Run data_quality checks on universes (liquidity, data availability, symbol validity)
- Track change history and provenance for any universe modification

### `core/params.py`
- Role: Centralized parameter and profile configuration hub for the pairs-trading system.
- Priority: `high`
- Categories: infra, research, backtest, live_trading, risk
- Phases: phase_architecture_env, phase_signals_portfolio

**Tasks:**
- Refactor scattered params into structured Pydantic models keyed by environment/profile
- Link parameter definitions to ranges and defaults used across optimizers and engines
- Support config snapshotting with git_rev and run_id for every major workflow
- Deprecate direct magic numbers elsewhere in favor of params accessors
- Validate parameter compatibility with risk_engine and broker limits

### `core/ranges.py`
- Role: Advanced parameter range utilities powering optimization and configuration.
- Priority: `high`
- Categories: infra, research, backtest, utils
- Phases: phase_architecture_env, phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Standardize parameter range definitions for all optimization-relevant knobs
- Integrate tightly with distributions and params as a single source of truth
- Add validation for bounds, step sizes, and log/linear semantics
- Remove any duplicated or unused range definitions across the codebase
- Support config-driven overrides per environment/profile

### `core/refactor_load_price_data.py`
- Role: One-off refactor tool to migrate legacy price-loading code to the new data layer.
- Priority: `low`
- Categories: utils, other
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Restrict usage to controlled migration workflows and document limitations
- Add safety checks and a dry-run mode before modifying any target files

### `core/refactor_streamlit_query_params.py`
- Role: Automated migration helper for updating deprecated Streamlit query params APIs.
- Priority: `low`
- Categories: utils, web_ui
- Phases: phase_web_app_ux, phase_testing_deployment_ai

**Tasks:**
- Implement idempotent transformation from st.experimental_get_query_params to st.query_params
- Add minimal tests or checks ensuring no behavioral changes in existing tabs

### `core/refactor_streamlit_width.py`
- Role: Migration helper for updating Streamlit width-related arguments across the UI.
- Priority: `low`
- Categories: utils, web_ui
- Phases: phase_web_app_ux, phase_testing_deployment_ai

**Tasks:**
- Automate safe replacement of use_container_width with the new width API
- Provide a dry-run report mode to preview changes before applying them

### `core/regime_classifier.py`
- Role: Regime detection module classifying volatility and market regimes for signals and risk.
- Priority: `medium`
- Categories: research, backtest, risk
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Define a clean interface for regime classification based on price and macro inputs
- Implement an initial volatility-based regime classifier with deterministic behavior
- Integrate regime outputs into signals_engine and macro_engine as optional inputs
- Persist regime labels and configs into SqlStore for backtests and live monitoring

### `core/risk_engine.py`
- Role: Central risk engine enforcing limits, exposures, and kill-switches across the fund.
- Priority: `high`
- Categories: risk, live_trading, backtest, infra
- Phases: phase_signals_portfolio, phase_execution_broker, phase_observability_ops

**Tasks:**
- Define standard risk model interfaces for position, portfolio, and exposure checks
- Unify risk rules and limits across research, paper, and live with explicit overrides
- Integrate with SqlStore and broker router to monitor real positions and orders
- Implement kill-switches, max drawdown gates, and portfolio-level caps
- Log and persist all risk decisions and violations with environment metadata
- Support simulation and replay modes for testing new risk policies

### `core/risk_parity.py`
- Role: Risk-parity utilities for portfolio sizing and allocation within pairs strategies.
- Priority: `high`
- Categories: risk, research, backtest, live_trading
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Implement robust risk-parity sizing algorithms compatible with pairs portfolios
- Expose a clear API used by signals_engine and portfolio constructors
- Support leverage, concentration, and sector exposure constraints
- Add validation and unit-style tests on toy portfolios to verify behavior
- Coordinate with risk_engine to avoid conflicting sizing and limits

### `core/signals_engine.py`
- Role: High-level signals engine generating pair signals and target portfolios from data and configs.
- Priority: `high`
- Categories: research, backtest, live_trading, risk
- Phases: phase_signals_portfolio, phase_execution_broker, phase_testing_deployment_ai

**Tasks:**
- Unify signal generation pipeline (data → features → scores → targets) under a single interface
- Decouple signal computation from execution, outputting explicit target positions and metadata
- Integrate fair_value_engine, macro_engine, regime_classifier, and risk_parity as pluggable components
- Ensure deterministic outputs given inputs, environment, and config snapshot
- Persist signals, targets, and inputs to SqlStore for full traceability across modes
- Support both simulation and live modes using the same core logic with profile-based switches

### `core/sql_store.py`
- Role: Central SQL persistence layer for prices, signals, risk, experiments, and metadata.
- Priority: `high`
- Categories: infra, data_ingest, research, backtest, live_trading
- Phases: phase_architecture_env, phase_data_layer, phase_observability_ops

**Tasks:**
- Finalize schema definitions and typed accessors for all major entities (prices, signals, risk, orders, configs)
- Implement schema versioning and lightweight migrations with clear upgrade paths
- Add health checks, connection management, and error handling for all supported backends
- Optimize indices and partitioning for common access patterns and time-series queries
- Tag all records with environment, profile, run_id, and git_rev where applicable
- Provide simple helper APIs used by dashboard_service, optimizers, and risk_engine

### `core/tab_comparison_matrices.py`
- Role: UI and computation layer for comparison matrices (correlations, metrics) across pairs.
- Priority: `medium`
- Categories: web_ui, research, backtest
- Phases: phase_web_app_ux, phase_signals_portfolio

**Tasks:**
- Refactor matrix computations into pure functions reusable outside Streamlit contexts
- Ensure metrics used here are sourced from advanced_metrics and metrics utilities
- Integrate pairs_universe and SqlStore as data sources instead of ad-hoc loaders
- Add basic caching and performance safeguards for large universes

### `core\__init__.py`
- Role: namespace for core application modules (AppContext, analytics, dashboard models)
- Priority: `low`
- Categories: infra
- Phases: phase_architecture_env

**Tasks:**
- Provide a docstring outlining the responsibilities of the core package
- Re-export only key entry-point abstractions (e.g. AppContext) if needed
- Ensure no heavy computation runs at import time

### `core\analysis_helpers.py`
- Role: advanced analysis utilities for evaluating pairs, strategies, and risk in research/backtests
- Priority: `high`
- Categories: research, backtest, risk
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Align analysis outputs (metrics, tables) with dashboard_models for easy visualization
- Ensure all functions are pure and deterministic given their inputs
- Factor out shared code with advanced_metrics and analytics to avoid duplication
- Add support for environment-aware analysis (e.g. paper vs live history windows)
- Introduce tests for key analysis pipelines using synthetic data

### `core\analytics.py`
- Role: analytics and reporting layer for pairs trading optimization and performance tracking
- Priority: `high`
- Categories: research, backtest, web_ui
- Phases: phase_signals_portfolio, phase_web_app_ux, phase_testing_deployment_ai

**Tasks:**
- Provide high-level reporting functions consumed by Streamlit dashboards and agents
- Integrate with SqlStore to read/write analytics summaries with explicit schemas
- Ensure metrics are consistent across research, backtest, and live equity curves
- Expose configuration-driven report generation (e.g. top-N pairs, risk metrics)
- Add tests or notebook-style validations for core analytics flows

### `core\anomaly_detection.py`
- Role: anomaly detection engine for data, signals, and portfolio behaviour
- Priority: `high`
- Categories: risk, research, backtest, infra
- Phases: phase_observability_ops, phase_data_layer, phase_testing_deployment_ai

**Tasks:**
- Define clear anomaly types (data gaps, outliers, regime breaks, execution issues) with structured outputs
- Integrate anomaly alerts into observability dashboards and logs
- Ensure algorithms run deterministically and are configurable via config_manager
- Provide hooks to tag and persist anomalies in SqlStore for later analysis
- Add tests on synthetic anomalies to validate detection quality and false-positive rates

### `core\app_context.py`
- Role: central application context wiring together configs, SqlStore, brokers, risk engine, and UI
- Priority: `high`
- Categories: infra, live_trading, research, backtest, web_ui, risk
- Phases: phase_architecture_env, phase_execution_broker, phase_observability_ops, phase_testing_deployment_ai

**Tasks:**
- Formalize AppContext as the single entry point for environment/profile selection and dependency wiring
- Integrate config_manager, SqlStore connections, broker router, and risk engine behind typed properties
- Ensure context supports dev/research/paper/live modes with explicit, inspectable state
- Add lightweight health-check methods for core subsystems (data, broker, risk, web)
- Minimize global state and side effects; make context construction deterministic and testable
- Document usage patterns for Streamlit apps, agents, and CLI tools to share this context

### `core\clustering.py`
- Role: clustering utilities for pair universe curation and strategy segmentation
- Priority: `medium`
- Categories: research, backtest
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Standardize inputs/outputs (feature matrices, cluster labels) for downstream use in signal_generator
- Ensure clustering is reproducible via fixed seeds and deterministic algorithms where possible
- Expose configuration for clustering schemes via config_manager
- Add tests verifying basic clustering behaviour on toy data

### `core\dashboard_models.py`
- Role: data models powering dashboards and web UIs (KPIs, tables, health views)
- Priority: `high`
- Categories: web_ui, infra, live_trading, research, risk
- Phases: phase_web_app_ux, phase_observability_ops, phase_architecture_env

**Tasks:**
- Define Pydantic models for dashboard payloads (pairs, portfolios, risk state, health summaries)
- Ensure models encode environment, timestamps, and run identifiers for auditability
- Align schemas with live_pair_store, portfolio_loader, and analytics outputs
- Provide serialization helpers for Streamlit and any future API endpoints
- Add validation tests to catch breaking schema changes early

### `data_loader.py`
- Role: Thin adapter bridging legacy callers to common.data_loader and the SqlStore-based data layer.
- Priority: `medium`
- Categories: data_ingest, infra, research, backtest
- Phases: phase_data_layer, phase_architecture_env

**Tasks:**
- Keep the module as a stable facade delegating to common.data_loader and sql_price_loader
- Ensure all paths go through AppContext and SqlStore where possible for consistency
- Add deprecation warnings or logs for legacy-only usage patterns
- Standardize default history length and frequency arguments across callers

### `datafeed/ib_connection.py`
- Role: Shared IBKR connection manager reused by data ingestion and order routing.
- Priority: `high`
- Categories: infra, data_ingest, live_trading
- Phases: phase_execution_broker, phase_data_layer, phase_observability_ops

**Tasks:**
- Centralize IBKR client creation, reconnect logic, and error handling
- Load credentials and connection settings from secure env/profile configs
- Expose health-check and status APIs for dashboard_service and ops tooling
- Share connection safely between ib_data_ingestor and ib_order_router with clear lifecycle
- Add explicit awareness of paper vs live endpoints and safety limits

### `datafeed/yf_loader.py`
- Role: Lightweight Yahoo Finance-based loader for research and testing data.
- Priority: `medium`
- Categories: data_ingest, research, backtest
- Phases: phase_data_layer, phase_testing_deployment_ai

**Tasks:**
- Wrap yfinance calls with retry, backoff, and simple rate-limiting safeguards
- Normalize outputs into the same schema used by SqlStore price tables
- Integrate basic data_quality checks for gaps and anomalies
- Support optional local caching for offline tests and faster research iterations

### `dedupe_opt_tab.py`
- Role: One-off maintenance tool to deduplicate optimization tab code and reduce dashboard bloat
- Priority: `low`
- Categories: infra, utils
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Refactor script into a small, idempotent CLI utility using the shared AppContext logging
- Document usage and mark as maintenance-only, ensuring it is excluded from main app entry points
- Add a minimal test or dry-run mode to verify changes without overwriting files

### `fix_imports.py`
- Role: Temporary refactoring helper to normalize imports across the project
- Priority: `low`
- Categories: infra, utils
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Limit script scope to a deterministic, dry-runable import normalization pass with backups
- Update it to respect the current package layout and avoid touching generated or backup files
- Mark script as dev-only and wire basic logging for traceability when used

### `gpt_upgrade_agent.py`
- Role: Thin wrapper for AI-based upgrade agent focused on per-file refactors and governance
- Priority: `medium`
- Categories: infra, other
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Implement a minimal UpgradeAgent interface that delegates to hedge_fund_upgrade_agent primitives
- Wire environment-aware safety guards so the agent never touches live-trading configs directly
- Add configuration hooks for rate-limiting, logging, and dry-run modes
- Ensure deterministic behavior by seeding any stochastic components through central config

### `health_check_full_system.py`
- Role: End-to-end system health checker covering data, risk, broker, and web entrypoints
- Priority: `high`
- Categories: infra, tests, risk
- Phases: phase_architecture_env, phase_observability_ops, phase_testing_deployment_ai

**Tasks:**
- Refactor checks into composable functions (data, broker, risk, dashboards, APIs) with clear statuses
- Integrate with AppContext and SqlStore to validate data freshness, schemas, and history length
- Add broker and paper/live profile connectivity checks with safe timeouts and retries
- Emit structured health results (JSON) suitable for dashboards and external monitoring tools
- Introduce a fast smoke-test mode for CI and a deeper diagnostic mode for ops

### `hedge_fund_upgrade_agent.py`
- Role: Central coordinator for stepwise, phase-based upgrades of the codebase toward hedge-fund-grade standards.
- Priority: `medium`
- Categories: infra, other
- Phases: phase_architecture_env, phase_observability_ops

**Tasks:**
- Model upgrade phases and per-file tasks as explicit data structures tied to the global phase ids.
- Add deterministic execution order and idempotency checks so rerunning the agent does not corrupt files.
- Introduce dry-run and snapshot modes that back up modified files and configs before applying changes.
- Integrate with health_check_full_system.py to validate the system after each major upgrade phase.

### `quick_test.py`
- Role: Minimal developer smoke-test script for critical imports and basic flows
- Priority: `low`
- Categories: tests, utils
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Replace ad-hoc imports with a deterministic, fast smoke-test of AppContext and core services
- Ensure script can run from project root and respect active environment/profile settings
- Mark as dev-only and integrate into a basic local pre-commit or CI check

### `root/system_upgrader_agent.py`
- Role: Central log-driven upgrade agent orchestrating phased refactors and code migrations for the whole system.
- Priority: `high`
- Categories: infra, utils, tests
- Phases: phase_architecture_env, phase_observability_ops, phase_testing_deployment_ai

**Tasks:**
- Align agent actions and config with the defined upgrade phases and environment profiles
- Refactor to use AppContext and central config_manager instead of ad-hoc paths or flags
- Add dry-run and rollback metadata so upgrades are idempotent and auditable
- Persist upgrade decisions, affected files, and git_rev into SqlStore or JSON snapshots
- Improve structured logging and error handling for batch and per-file upgrade runs

### `root/tab_comparison_matrices.py`
- Role: Streamlit tab providing HF-grade correlation and comparison matrices for research and monitoring.
- Priority: `high`
- Categories: web_ui, research
- Phases: phase_architecture_env, phase_data_layer, phase_signals_portfolio, phase_web_app_ux

**Tasks:**
- Wire tab into AppContext with explicit environment profile (dev/research/paper/live) indicators
- Route all price and factor data access through SqlStore/market_data_router for consistency
- Reuse shared signal and risk helpers for metrics instead of custom in-tab calculations
- Add caching, progress indicators, and unique key generation for all Streamlit widgets
- Standardize outputs (tables/plots) to feed other tabs and agents via a shared response schema

### `root/test_agent.py`
- Role: Lightweight harness for exercising and validating the visualization agent behaviours.
- Priority: `low`
- Categories: tests, research
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Convert into a small, deterministic test harness that calls VisualizationAgent via a stable API
- Ensure it uses central config/env profiles instead of hard-coded paths or modes
- Mark clearly as non-production and safe to run in any environment

### `root/trade_logic.py`
- Role: Core trade decision module mapping signals and portfolio targets into actionable position intents.
- Priority: `high`
- Categories: live_trading, backtest, risk
- Phases: phase_architecture_env, phase_signals_portfolio, phase_execution_broker, phase_observability_ops

**Tasks:**
- Separate pure signal interpretation (what target to hold) from execution concerns (how to trade)
- Define clear data models for target positions, deltas, and trade intents shared across backtest and live
- Integrate risk overlays (max leverage, exposure caps, kill-switches) using shared risk_helpers
- Ensure all parameters and thresholds are pulled from central, versioned config models
- Add structured logging of every trade decision including inputs, overrides, and environment

### `root/visualization.py`
- Role: Shared visualization toolkit for plots and tables used by research and UI agents.
- Priority: `medium`
- Categories: research, web_ui, utils
- Phases: phase_signals_portfolio, phase_web_app_ux, phase_testing_deployment_ai

**Tasks:**
- Refactor plotting helpers to accept typed data models/DataFrames and avoid direct global state
- Consolidate duplicated plotting logic used by tabs, scripts, and agents into reusable functions
- Add options for environment-aware styling and risk indicators where relevant
- Ensure graceful degradation when optional visualization libraries are missing

### `root/visualization_agent.py`
- Role: Agent that orchestrates creation of rich visual diagnostics for pairs, backtests, and live state.
- Priority: `medium`
- Categories: web_ui, research, infra
- Phases: phase_architecture_env, phase_web_app_ux, phase_testing_deployment_ai

**Tasks:**
- Refactor the agent to depend on AppContext and shared visualization helpers instead of ad-hoc flows
- Make the agent environment-aware (research/paper/live) and restrict data sources accordingly
- Add structured inputs/outputs so other modules can request specific visualizations programmatically
- Improve logging and error handling around heavy plots to avoid crashing UI flows

### `root/volatility.py`
- Role: Utility module for volatility and risk metric calculations reused across signals and portfolio logic.
- Priority: `medium`
- Categories: risk, research, utils
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Align volatility estimators with those in advanced_metrics and risk_helpers to avoid inconsistencies
- Add full type hints and small doctest-style examples for core formulas
- Ensure functions are side-effect-free and safe to use in both backtest and live paths
- Introduce basic unit-style tests for key volatility computations

### `root\__init__.py`
- Role: Root package initializer wiring environment profiles, shared context, and high-level entrypoints
- Priority: `high`
- Categories: infra, web_ui
- Phases: phase_architecture_env, phase_web_app_ux

**Tasks:**
- Expose a small, typed API to create an AppContext based on profile (dev/research/paper/live)
- Remove legacy globals and ensure environment is explicit and inspectable from all root modules
- Centralize project-wide constants (e.g., version, git_rev lookup) behind typed accessors
- Document the root package contract for dashboards, APIs, and scripts

### `root\analysis.py`
- Role: Core research and fund-level analysis engine for ranking pairs, portfolios, and regimes
- Priority: `high`
- Categories: research, backtest, risk
- Phases: phase_data_layer, phase_signals_portfolio, phase_testing_deployment_ai, phase_architecture_env

**Tasks:**
- Refactor heavy functions into smaller, typed units that consume SqlStore/price loaders instead of ad-hoc data
- Align signal and ranking computation with the shared signal_generator and risk overlays
- Persist analysis outputs (rankings, metrics, configs, git_rev) into a central store for traceability
- Ensure all randomness (e.g., sampling, clustering) is seeded via central config for reproducibility
- Add hooks for regime-aware analytics and integration with dashboard views

### `root\api_server.py`
- Role: FastAPI-based fair-value and analytics HTTP API for internal tools and agents
- Priority: `high`
- Categories: api, research, live_trading
- Phases: phase_architecture_env, phase_signals_portfolio, phase_execution_broker, phase_web_app_ux, phase_observability_ops, phase_testing_deployment_ai

**Tasks:**
- Refactor to instantiate a single AppContext per process with explicit profile selection
- Standardize request/response models using Pydantic v2 with clear schemas and validation
- Route price and signal requests through the shared data and signal engines with consistent parameters
- Add observability: structured logging, request timing, and health endpoints for each dependency
- Ensure rate limits, auth hooks, and explicit separation between research and live endpoints

### `root\backtest.py`
- Role: High-level backtesting orchestrator for pairs strategies across universes and regimes
- Priority: `high`
- Categories: backtest, research, risk
- Phases: phase_data_layer, phase_signals_portfolio, phase_execution_broker, phase_testing_deployment_ai, phase_architecture_env

**Tasks:**
- Unify backtest configuration into typed models shared with live trading and paper profiles
- Ensure data access goes through SqlStore/price loader with deterministic seeds and exact history rules
- Delegate strategy logic to reusable components also used by live execution (signals, sizing, risk overlays)
- Persist backtest runs, parameters, and performance metrics for later comparison and replay
- Provide hooks for simulation modes that mimic live-execution constraints and slippage

### `root\backtest_logic.py`
- Role: Core backtest implementation details and shared utilities used by root.backtest
- Priority: `high`
- Categories: backtest, research
- Phases: phase_signals_portfolio, phase_data_layer, phase_testing_deployment_ai

**Tasks:**
- Isolate pure strategy, PnL, and risk computations from I/O for easier reuse and testing
- Align order/position representation with the live execution engine’s data structures
- Support regime-aware and rolling-window backtests consistent with live signal parameters
- Add unit-style tests or doctests for key math functions (returns, drawdowns, risk metrics)

### `root\config_tab.py`
- Role: Streamlit configuration tab for environment, profiles, and global system settings
- Priority: `high`
- Categories: web_ui, infra
- Phases: phase_architecture_env, phase_web_app_ux

**Tasks:**
- Refactor UI to read/write a central Pydantic-based settings model via AppContext
- Make environment/profile selection explicit, clearly labeled, and persisted in session_state
- Guard dangerous options (live trading, leverage caps) behind confirmations and clear warnings
- Add validation feedback for misconfigured paths, brokers, or data sources visible in the UI

### `root\dashboard.backup_mojibake.py`
- Role: Legacy backup of an older dashboard implementation kept only for reference
- Priority: `low`
- Categories: other
- Phases: phase_architecture_env

**Tasks:**
- Mark file as deprecated/backup and exclude it from any runtime imports or tooling
- Document which parts (if any) are still relevant and migrate them into shared modules before removal
- Plan eventual deletion once parity with root.dashboard.py and newer UIs is confirmed

### `root\dashboard.py`
- Role: Main Streamlit dashboard aggregating research, risk, and live-trading controls
- Priority: `high`
- Categories: web_ui, research, live_trading, risk
- Phases: phase_architecture_env, phase_signals_portfolio, phase_execution_broker, phase_web_app_ux, phase_observability_ops

**Tasks:**
- Decompose the monolithic file into smaller, testable sections wired through dashboard_service_factory
- Ensure all tabs use shared services (data, signals, risk, broker) via AppContext, not ad-hoc flows
- Make environment and risk status prominent (profile badges, risk lights, read-only vs live modes)
- Standardize widget keys and state handling using centralized UI helpers to avoid conflicts
- Add lightweight performance and health indicators (latency, data age, broker status) to the layout

### `root\dashboard_home_v2.py`
- Role: Modernized dashboard home providing top-level fund status and navigation hub
- Priority: `high`
- Categories: web_ui, infra
- Phases: phase_architecture_env, phase_web_app_ux, phase_observability_ops

**Tasks:**
- Surface high-level KPIs, environment indicators, and key health metrics via shared services
- Provide clear navigation to research, optimization, risk, and live-trading screens
- Integrate profile-aware views so dev/research/paper/live are visually distinct and safe
- Refactor shared cards and summaries into reusable components backed by AppContext

### `root\dashboard_service_factory.py`
- Role: Factory for dashboard-level services wiring data, signals, risk, and brokers into UI-friendly adapters
- Priority: `high`
- Categories: infra, web_ui
- Phases: phase_architecture_env, phase_web_app_ux, phase_observability_ops

**Tasks:**
- Define clear service interfaces for data access, signal generation, portfolio views, and execution
- Ensure all services are instantiated from AppContext with explicit environment profiles
- Provide lightweight caching and error boundaries suitable for Streamlit reruns
- Document factory usage patterns and migrate dashboard tabs to depend exclusively on it

### `root\dedupe_opt_tab.py`
- Role: Dashboard-specific helper for deduplicating optimization tab UI and logic
- Priority: `medium`
- Categories: web_ui, infra, utils
- Phases: phase_architecture_env, phase_web_app_ux

**Tasks:**
- Factor out common optimization-tab sections into reusable UI components and service calls
- Remove code duplication with root.optimization_tab and backup variants while preserving behavior
- Align any optimization-specific state management with shared Streamlit key conventions
- Add brief documentation of how deduped components are intended to be reused across tabs

### `root\fair_value_api_tab.py`
- Role: Streamlit tab for exploring and validating the fair-value API and advisor logic
- Priority: `high`
- Categories: web_ui, api, research
- Phases: phase_signals_portfolio, phase_web_app_ux, phase_architecture_env

**Tasks:**
- Replace direct data or model access with calls to root.api_server or shared signal services
- Expose configuration controls (universe, parameters) that mirror live API behavior
- Show clear separation between simulation, paper, and live query modes in the UI
- Log queries and responses for debugging and validation of fair-value decisions
- Guard against using dummy data when connected to live profiles

### `root\generate_config.py`
- Role: Utility to generate standard config.json/universe configs for pairs trading experiments
- Priority: `medium`
- Categories: data_ingest, research, infra
- Phases: phase_architecture_env, phase_data_layer, phase_testing_deployment_ai

**Tasks:**
- Refactor to use Pydantic-based config models aligned with root.settings profiles
- Ensure deterministic pair selection logic with explicit random seeds and filters
- Write generated configs into a governed location with version and timestamp metadata
- Add a dry-run/preview mode to inspect output before writing files

### `root\generate_pairs_universe.py`
- Role: Universe builder for generating and maintaining the tradable pairs universe
- Priority: `high`
- Categories: data_ingest, research, backtest
- Phases: phase_data_layer, phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Use centralized data loaders and fundamental/price filters to build a reproducible pairs universe
- Persist universes and selection criteria into SqlStore with history and config snapshots
- Expose options for different environments (research vs live eligible universe) via profiles
- Integrate basic diagnostics (coverage, liquidity, sector balance) into output logs
- Provide an option to run as a simulation or backfill mode for historical analyses

### `root\ibkr_connection.py`
- Role: IBKR connectivity and session management for data and trading, behind a safe abstraction
- Priority: `high`
- Categories: live_trading, data_ingest, infra, api
- Phases: phase_execution_broker, phase_data_layer, phase_observability_ops, phase_architecture_env

**Tasks:**
- Wrap IBKR client initialization in a profile-aware broker adapter with paper/live separation
- Implement robust retry, backoff, and timeout logic with structured error reporting
- Expose a minimal, typed interface for market data and order routing used by higher layers
- Ensure credentials are loaded from secure sources (env/secret manager stub) and never logged
- Add health checks and connectivity tests consumable by dashboards and health_check_full_system

### `root\index_fundamentals_tab.py`
- Role: Streamlit tab for exploring index/ETF fundamentals and their impact on pairs
- Priority: `medium`
- Categories: web_ui, research, data_ingest
- Phases: phase_data_layer, phase_signals_portfolio, phase_web_app_ux

**Tasks:**
- Route data access through the shared fundamental_loader and SqlStore instead of direct sources
- Present fundamentals in a way that ties into pair selection and risk overlays
- Add caching and pagination for large datasets to keep UI responsive
- Highlight environment/profile (research vs live) and disable dummy data in live views

### `root\ingest_universe_from_ib.py`
- Role: Ingestion script/tab for importing instrument universe from IBKR into the central store
- Priority: `high`
- Categories: data_ingest, research, live_trading
- Phases: phase_data_layer, phase_execution_broker, phase_observability_ops

**Tasks:**
- Refactor ingestion logic to use the IBKR broker adapter and SqlStore with explicit schemas
- Add gap detection, anomaly logging, and data-quality scoring during universe import
- Make environment/profile explicit (paper vs live) and safe by default
- Support incremental updates and idempotent re-runs without duplicating instruments

### `root\insights.py`
- Role: Research insights and analytics builder feeding high-level views and reports
- Priority: `medium`
- Categories: research, web_ui
- Phases: phase_signals_portfolio, phase_data_layer, phase_web_app_ux

**Tasks:**
- Consolidate scattered analytics into reusable functions returning typed data models
- Tie insights to persisted backtest/live metrics to make them reproducible and explorable
- Expose a minimal API for dashboards and agents to request top-N pairs, themes, or risks
- Ensure all computations use shared signal/risk engines rather than duplicating logic

### `root\live_dash_app.py`
- Role: Streamlit entrypoint for live trading and monitoring dashboard
- Priority: `high`
- Categories: web_ui, live_trading, risk
- Phases: phase_architecture_env, phase_execution_broker, phase_web_app_ux, phase_observability_ops

**Tasks:**
- Ensure startup builds an AppContext with a mandatory explicit live/paper profile selection
- Limit this app to live-relevant tabs and hide research-only or dummy-data views
- Display clear risk state, broker connectivity, and kill-switch controls at the top level
- Wire in health checks and latency indicators for data and broker flows

### `root\macro_tab.py`
- Role: Macro adjustments and market regime configuration tab influencing signals and risk
- Priority: `high`
- Categories: web_ui, research, risk
- Phases: phase_signals_portfolio, phase_data_layer, phase_web_app_ux

**Tasks:**
- Connect macro sliders and settings directly to the macro_adjustments and macro_sensitivity engines
- Persist macro regimes and parameters with timestamps and profiles for reproducibility
- Differentiate between research tuning and live-approved macro overlays with approvals or locks
- Visualize macro impacts on key pairs and portfolio metrics using shared analytics helpers

### `root\matrix_research_tab.py`
- Role: Matrix-style research tab for cross-sectional pair analytics and selection
- Priority: `high`
- Categories: web_ui, research, backtest
- Phases: phase_data_layer, phase_signals_portfolio, phase_web_app_ux

**Tasks:**
- Standardize matrix computations through advanced_metrics and matrix_helpers modules
- Load data exclusively via SqlStore/market_data_router with clear history-length controls
- Allow exporting selected universes or candidate pairs into configs and backtests
- Optimize performance with caching and efficient queries for large universes
- Make environment/profile explicit and avoid any implicit live-trading actions

### `root\optimization_tab.backup_quotes_fix.py`
- Role: Legacy backup of optimization tab with quote-related fixes
- Priority: `low`
- Categories: other
- Phases: phase_architecture_env

**Tasks:**
- Mark as backup-only and remove any references from dashboards or scripts
- Extract any still-missing bugfixes into shared optimization logic, then plan deprecation
- Exclude file from automated tooling (lint, build) once migration is complete

### `root\optimization_tab.dedup.py`
- Role: Partially deduplicated version of the optimization tab used during refactoring
- Priority: `medium`
- Categories: web_ui, research, backtest
- Phases: phase_signals_portfolio, phase_web_app_ux, phase_testing_deployment_ai

**Tasks:**
- Compare implementation with root.optimization_tab and extract shared components into services/UI helpers
- Remove residual duplication and ensure a single canonical optimization UI remains
- Validate that all optimization workflows use shared backtest/Optuna engines and consistent configs
- Once merged, mark this file as deprecated or fully remove it

### `root\optimization_tab.py`
- Role: Primary optimization Streamlit tab for hyper-parameters, pair selection, and strategy search
- Priority: `high`
- Categories: web_ui, research, backtest
- Phases: phase_signals_portfolio, phase_data_layer, phase_web_app_ux, phase_testing_deployment_ai

**Tasks:**
- Route all optimization runs through a shared Optuna/Zoom engine with deterministic seeds
- Persist optimization studies, configs, and results in SqlStore/Parquet with metadata for replay
- Align objective functions and constraints with live risk and portfolio construction rules
- Separate UI concerns (widgets, layouts) from optimization logic via dashboard_service_factory
- Add clear environment/profile indicators and restrict live-linked parameters without approval

### `root\overview_tab.py`
- Role: Investment-committee style overview tab for fund-level status and KPIs
- Priority: `high`
- Categories: web_ui, risk, live_trading
- Phases: phase_web_app_ux, phase_observability_ops, phase_architecture_env

**Tasks:**
- Aggregate key metrics (AUM, exposure, PnL, drawdowns, crowding) via shared analytics services
- Display environment and profile prominently, including whether views are live or simulated
- Integrate data freshness and health indicators for critical components (data, broker, risk engine)
- Offer drill-down links to risk, portfolio, and pair-level tabs without duplicating logic

### `root\pair_tab.py`
- Role: Pair-level deep-dive dashboard for signals, trades, and live status
- Priority: `high`
- Categories: web_ui, research, live_trading, risk
- Phases: phase_signals_portfolio, phase_data_layer, phase_execution_broker, phase_web_app_ux

**Tasks:**
- Use a unified pair profile model from live_pair_store/live_profiles for all views
- Show signal state, recent trades, and risk overlays derived from shared engines
- Support both research and live modes with clear labeling and read-only safeguards in live
- Integrate controls for paper-trade simulations and live order suggestions while delegating execution
- Ensure widgets and callbacks use stable keys and centralized time-series data loaders

### `root\pairs.py`
- Role: Domain models and utilities for pairs, spreads, and related portfolio constructs
- Priority: `high`
- Categories: research, backtest, risk, infra
- Phases: phase_signals_portfolio, phase_data_layer, phase_execution_broker, phase_architecture_env

**Tasks:**
- Define typed Pydantic models or dataclasses for pair definitions, states, and configs
- Centralize spread/z-score/Ou-process helpers used by backtests, signals, and dashboards
- Align pair identifiers and metadata with SqlStore schemas and live_pair_store profiles
- Add serialization helpers to persist pair states and configs across runs and environments

### `root\portfolio_tab.py`
- Role: Portfolio-level dashboard for positions, exposures, risk, and performance
- Priority: `high`
- Categories: web_ui, risk, live_trading, research
- Phases: phase_signals_portfolio, phase_execution_broker, phase_web_app_ux, phase_observability_ops

**Tasks:**
- Source portfolio holdings and PnL from the shared portfolio_loader and SqlStore
- Visualize risk metrics (VaR/ES, drawdowns, concentration, correlation crowding) using shared risk_helpers
- Support scenario filters (profile, date range, strategy) without duplicating analytics code
- Highlight discrepancies between target and actual exposures and link to execution diagnostics
- Make live vs paper portfolios clearly distinguishable with appropriate safeguards

### `root\risk_tab.py`
- Role: Fund-level risk dashboard focused on real data and live constraints
- Priority: `high`
- Categories: web_ui, risk, live_trading
- Phases: phase_web_app_ux, phase_observability_ops, phase_execution_broker

**Tasks:**
- Integrate portfolio, pair-level, and macro risks into a unified view using shared risk_helpers
- Display real-time or near-real-time metrics (exposures, limits, breaches) with clear thresholds
- Expose safe controls for risk limits and kill-switches, logging all changes with metadata
- Ensure tab is read-only or heavily guarded in live mode and clearly separate research views

### `root\run_all.py`
- Role: Unified entrypoint for launching the full system (dashboards, APIs, services)
- Priority: `high`
- Categories: infra, web_ui
- Phases: phase_architecture_env, phase_web_app_ux, phase_testing_deployment_ai

**Tasks:**
- Convert into a small, explicit launcher that selects environment/profile and orchestrates components
- Support starting subsets (e.g., dashboards only, API only) via CLI flags or config
- Wire structured logging and health checks at startup, failing fast on critical misconfigurations
- Document intended usage patterns for dev/research/paper/live deployments

### `root\run_upgrade.py`
- Role: Entry script for running upgrade and visualization agents in a controlled manner
- Priority: `medium`
- Categories: infra, other
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Refactor to use hedge_fund_upgrade_agent/gpt_upgrade_agent with explicit phase selection
- Add CLI arguments or config for dry-run, scope restriction, and logging verbosity
- Ensure it cannot modify live-trading configs without an explicit override flag
- Log executed upgrades and outcomes into a small audit trail for governance

### `root\run_zoom_campaign.py`
- Role: Coordinator for running Optuna Zoom optimization campaigns over strategies/universes
- Priority: `medium`
- Categories: research, backtest, infra
- Phases: phase_signals_portfolio, phase_testing_deployment_ai, phase_architecture_env

**Tasks:**
- Standardize campaign configuration (search space, objectives, seeds) using Pydantic models
- Route all optimizations through shared backtest and optimization engines for consistency
- Persist campaign results, logs, and config snapshots for later analysis and reproduction
- Provide a simulation vs live-constraints mode to reflect realistic trading limitations

### `root\settings.py`
- Role: Centralized settings and profile management module for dev/research/paper/live environments
- Priority: `high`
- Categories: infra, risk, web_ui
- Phases: phase_architecture_env, phase_web_app_ux, phase_execution_broker, phase_observability_ops

**Tasks:**
- Define Pydantic-based settings models capturing environment, data paths, broker configs, and risk caps
- Implement profile loading/overrides from JSON/YAML/env with clear precedence rules
- Expose helpers for runtime components (dashboards, APIs, backtests) to query active settings
- Track git_rev and config snapshots for each run to support full reproducibility and audits
- Mark dangerous overrides (e.g., disabling risk limits) and require explicit confirmation in UIs/CLIs

### `root\smart_scan_tab.py`
- Role: Smart-scan Streamlit tab for discovering and ranking candidate pairs and strategies
- Priority: `high`
- Categories: web_ui, research, backtest
- Phases: phase_signals_portfolio, phase_data_layer, phase_web_app_ux

**Tasks:**
- Use shared analysis and signal engines to compute candidate scores and diagnostics
- Allow users to filter and export selected pairs into universe configs and backtests
- Ensure scans operate on the central data store with explicit history windows and quality checks
- Optimize performance for large universes via batching, caching, and async-friendly design where possible
- Display profile/environment and prevent accidental live trading actions from this research tab

### `root_desktop/app.py`
- Role: Legacy desktop entry point mirroring the main web app functionality for local workflows.
- Priority: `medium`
- Categories: web_ui, infra
- Phases: phase_architecture_env, phase_web_app_ux

**Tasks:**
- Refactor bootstrap to reuse the same AppContext and config models as the main web application
- Minimize business logic here by delegating to shared services and views
- Make environment profile explicit and visible in the desktop UI state
- Clarify which flows are desktop-only vs shared with the web app

### `root_desktop/views/main_window.py`
- Role: Main desktop window and tab container for research, optimization, and monitoring views.
- Priority: `medium`
- Categories: web_ui, research
- Phases: phase_architecture_env, phase_web_app_ux

**Tasks:**
- Align tabs and layout with the Streamlit-based dashboard structure where feasible
- Ensure each tab uses shared services (data loaders, signal engine, risk engine) rather than custom logic
- Display environment and risk status prominently in the main window
- Introduce consistent key naming and state handling for interactive widgets

### `run_hf_upgrade_batch.py`
- Role: Command-line entry script to run the system upgrade agent in batch mode.
- Priority: `medium`
- Categories: infra, utils
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Wrap system_upgrader_agent with a clear CLI interface and help text
- Wire in environment/profile selection and dry-run options by default
- Add structured logging to summarize outcomes and failures of each batch run
- Ensure script exits with meaningful status codes for CI integration

### `scripts/backtest_pair_from_sql.py`
- Role: Script to run a single-pair backtest using historical data from SqlStore.
- Priority: `medium`
- Categories: backtest, research
- Phases: phase_architecture_env, phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Refactor to use the shared backtest engine and AppContext instead of custom plumbing
- Pull parameters from typed config models and support environment-aware overrides
- Enforce deterministic seeds and persist backtest configs and results to SqlStore
- Harmonize output metrics and artifacts with those used by the web dashboard

### `scripts/build_dq_pairs_universe.py`
- Role: Universe builder that constructs and maintains the dq_pairs research universe in DuckDB/SqlStore.
- Priority: `high`
- Categories: data_ingest, research
- Phases: phase_architecture_env, phase_data_layer, phase_observability_ops

**Tasks:**
- Define and enforce a clear schema for the dq_pairs universe including keys and indices
- Use centralized DuckDB/SqlStore helpers with proper locking and error handling
- Add options for incremental updates, history retention, and data quality scoring
- Log build metadata (config, git_rev, run_id) for each universe snapshot
- Expose reusable functions for other tools to query or extend the dq_pairs universe

### `scripts/build_zoom_backtest_universe.py`
- Role: Prepares a backtest-ready universe tailored for Optuna/Zoom optimization campaigns.
- Priority: `medium`
- Categories: data_ingest, backtest
- Phases: phase_data_layer, phase_testing_deployment_ai

**Tasks:**
- Standardize universe schema and link it to dq_pairs and SqlStore price histories
- Support deterministic sampling and seeding to make optimization studies reproducible
- Record creation metadata and filters used for each universe snapshot
- Expose a callable function that can be reused by other research and CI pipelines

### `scripts/export_zoom_best_params_for_dq_pairs.py`
- Role: Exports best optimization parameters from Zoom/Optuna studies for the dq_pairs universe.
- Priority: `medium`
- Categories: research, backtest
- Phases: phase_data_layer, phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Use the shared zoom_storage/Optuna abstraction instead of local file logic
- Persist exported params into SqlStore or structured JSON with versioning and git_rev
- Ensure mapping between pairs and param sets is explicit and schema-validated
- Add CLI options for environment/profile selection and filters on study names

### `scripts/ingest_ibkr_prices.py`
- Role: Robust price ingestion script pulling historical data from IBKR into central storage.
- Priority: `high`
- Categories: data_ingest, live_trading
- Phases: phase_architecture_env, phase_data_layer, phase_observability_ops

**Tasks:**
- Refactor to use the unified market_data_router and SqlStore schemas for prices
- Implement robust retry/backoff, rate limiting, and safe failure modes for IBKR API calls
- Add gap detection, anomaly logging, and basic data quality scoring per symbol
- Make broker connection and symbols driven by environment-aware config models
- Record ingestion runs (coverage, errors, timestamps) for monitoring dashboards

### `scripts/ingest_prices_for_dq_pairs.py`
- Role: Ingests historical prices for the dq_pairs universe from configured data sources.
- Priority: `high`
- Categories: data_ingest, research
- Phases: phase_architecture_env, phase_data_layer, phase_observability_ops

**Tasks:**
- Route all data access through market_data_router to support multiple vendors consistently
- Ensure writes go to the canonical price tables in DuckDB/SqlStore with clear retention policies
- Implement gap detection, retries, and anomaly flags at the symbol-pair level
- Parameterize universe, date ranges, and vendors via typed, environment-aware configs
- Integrate ingestion run metadata with observability dashboards

### `scripts/maintain_duckdb_cache.py`
- Role: Maintenance script for cache.duckdb handling compaction, retention, and integrity checks.
- Priority: `high`
- Categories: data_ingest, infra
- Phases: phase_data_layer, phase_observability_ops

**Tasks:**
- Centralize DuckDB connection management and locking via shared helpers
- Implement retention policies (by age, size, or tags) for price and signal tables
- Add integrity checks, index validation, and lightweight VACUUM/OPTIMIZE routines
- Expose health metrics (table sizes, fragmentation, last maintenance) for monitoring
- Make critical operations dry-runnable and configurable per environment

### `scripts/optuna_backtest_search.py`
- Role: Runs Optuna-based hyperparameter search over the backtest engine for pairs strategies.
- Priority: `high`
- Categories: backtest, research
- Phases: phase_signals_portfolio, phase_data_layer, phase_testing_deployment_ai

**Tasks:**
- Refactor to call a shared backtest-and-evaluate function with clear typed configs
- Enforce deterministic seeding for Optuna studies and the underlying backtests
- Persist study results, best params, and config snapshots into zoom_storage/SqlStore
- Add pruning and checkpointing options suitable for long-running campaigns
- Expose standardized summaries consumable by the web dashboard and agents

### `scripts/replay_best_trial.py`
- Role: Replays and validates the best Optuna trial for a specific pair using the backtest engine.
- Priority: `high`
- Categories: backtest, research
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Fetch best trial parameters via the shared zoom_storage interface using strong typing
- Run replay using the same backtest engine and data layer as used in the original study
- Persist replay results and diagnostics with explicit links to the originating study/trial
- Ensure deterministic behaviour and consistent metrics vs the original optimization
- Prepare outputs (plots/tables) in a format reusable by UI and reporting tools

### `scripts/research_rank_pairs_from_dq.py`
- Role: Research script ranking pairs from dq_pairs universe based on signals and performance metrics.
- Priority: `high`
- Categories: research, backtest
- Phases: phase_architecture_env, phase_data_layer, phase_signals_portfolio

**Tasks:**
- Use the shared signal engine and portfolio metrics instead of bespoke ranking logic
- Ensure all input data comes from the canonical dq_pairs and price tables in SqlStore
- Persist rankings, configs, and metrics for each run with environment and git_rev tags
- Align ranking outputs with downstream consumers (top-pair selection, dashboards, mini-fund)
- Add basic self-checks for data freshness and completeness before running rankings

### `scripts/run_mini_fund_optimize.py`
- Role: Runs parameter optimization focused on the mini-fund subset of pairs.
- Priority: `medium`
- Categories: backtest, research
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Refactor to reuse the same Optuna/backtest scaffolding as the main optimization scripts
- Drive mini-fund universe selection via config and SqlStore queries instead of hard-coded lists
- Persist mini-fund-specific studies with clear tags and separation from global studies
- Harmonize metrics and outputs with those used for the full-universe campaigns

### `scripts/run_mini_fund_signals.py`
- Role: Generates daily signals for the mini-fund pairs, feeding paper/live workflows.
- Priority: `high`
- Categories: research, live_trading, backtest
- Phases: phase_architecture_env, phase_data_layer, phase_signals_portfolio, phase_observability_ops

**Tasks:**
- Route all calculations through the central signal engine and risk overlays used system-wide
- Make environment profile explicit (research/paper/live) and gate risky actions accordingly
- Persist generated signals and context (configs, data snapshot) into SqlStore for auditability
- Add monitoring hooks to flag missing data, stale prices, or inconsistent exposures
- Ensure deterministic behaviour for research re-runs while supporting live-time stamping

### `scripts/run_mini_fund_snapshot.py`
- Role: Produces a snapshot/scan of mini-fund pairs for daily monitoring and research.
- Priority: `medium`
- Categories: research, web_ui
- Phases: phase_data_layer, phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Use shared data loaders and signal computations to build the snapshot
- Persist snapshot outputs in a standardized schema consumable by dashboards
- Parameterize snapshot scope and thresholds via config models
- Add light self-checks for data freshness and coverage before producing a snapshot

### `scripts/run_zoom_campaign_for_dq_pairs.py`
- Role: Orchestrates multi-study Zoom/Optuna campaigns across the dq_pairs universe.
- Priority: `medium`
- Categories: backtest, research
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Refactor to orchestrate campaigns via zoom_storage and shared optimization helpers
- Support batching, resume-from-checkpoint, and deterministic seeding per campaign
- Tag runs with environment, universe version, and config snapshots for later analysis
- Expose progress and summary stats suitable for monitoring or UI integration

### `scripts/save_zoom_best_params.py`
- Role: Extracts and persists best Zoom/Optuna parameters into canonical storage.
- Priority: `medium`
- Categories: research, infra
- Phases: phase_data_layer, phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Read best params via zoom_storage with proper validation of study and trial IDs
- Save results into SqlStore/JSON with explicit schema and versioning
- Include environment, universe, and metric used for each best-param record
- Ensure idempotent writes so repeated runs do not corrupt existing records

### `scripts/select_top_pairs_from_ranked_csv.py`
- Role: Utility script to select top-ranked pairs from research outputs for further workflows.
- Priority: `medium`
- Categories: research, utils
- Phases: phase_data_layer, phase_signals_portfolio

**Tasks:**
- Treat ranked CSV as a formally defined schema and validate before processing
- Parameterize selection criteria (top-N, filters, risk caps) via config models
- Output selected pairs in formats consumable by backtests, dashboards, and live configs
- Record selection run metadata and links to the source ranking run

### `scripts/use_optuna_best_for_pair.py`
- Role: Applies stored best Optuna parameters to run focused backtests for a chosen pair.
- Priority: `medium`
- Categories: backtest, research
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Load best params using shared zoom_storage and config translation helpers
- Run backtests via the central backtest engine and data layer for consistency
- Emit standardized reports (metrics, plots, configs) for comparison across pairs
- Ensure deterministic runs and clear environment tagging in all outputs

### `sitecustomize.py`
- Role: Project-level bootstrap configuring paths, logging, and environment for all Python processes.
- Priority: `high`
- Categories: infra, utils
- Phases: phase_architecture_env, phase_observability_ops

**Tasks:**
- Centralize environment/profile detection and AppContext initialization in a safe, minimal way
- Set up structured logging defaults, timezone handling, and basic error hooks
- Avoid heavy or side-effectful imports that could break external tools or REPLs
- Document and guard any behaviour that differs between dev/research/paper/live
- Ensure compatibility with test runners and scripts that may bypass full app bootstrap

### `strip_old_actions_block.py`
- Role: Maintenance utility to strip obsolete actions blocks from code files.
- Priority: `low`
- Categories: infra, utils
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Confirm script is clearly marked as a one-off maintenance tool, not part of runtime flows
- Add dry-run and backup options before mutating files
- Align path handling and logging with the system_upgrader_agent where relevant

### `study.py`
- Role: Study orchestration module tying together experiments, backtests, and optimization runs.
- Priority: `medium`
- Categories: research, backtest
- Phases: phase_signals_portfolio, phase_testing_deployment_ai

**Tasks:**
- Refactor to use shared config models and AppContext for all study definitions
- Delegate actual backtest/optimization work to the core engines and scripts
- Persist study metadata, configs, and links to outputs in SqlStore
- Ensure reproducibility by enforcing deterministic seeding and version tagging

### `tools/bootstrap_universe.py`
- Role: Tooling script to initialize or extend the trading/research universe from raw definitions.
- Priority: `medium`
- Categories: data_ingest, infra
- Phases: phase_architecture_env, phase_data_layer

**Tasks:**
- Delegate universe construction to shared universe-building helpers (e.g., dq_pairs builder)
- Support idempotent runs and environment-specific universe variants
- Validate inputs and schemas before inserting into SqlStore/DuckDB
- Add clear logging of created/updated symbols and pairs

### `tools/debug_settings.py`
- Role: Diagnostics tool to inspect configuration, environment, and component health.
- Priority: `high`
- Categories: infra, tests
- Phases: phase_architecture_env, phase_observability_ops, phase_testing_deployment_ai

**Tasks:**
- Connect to AppContext and config_manager to display effective settings and profiles
- Add health checks for SqlStore, DuckDB, broker connectivity, and data sources
- Report discrepancies between expected and actual environments or paths
- Provide machine-readable output (JSON/lines) for CI and monitoring hooks
- Ensure it is safe to run in all environments without triggering trading actions

### `tools/ingest_from_ibkr.py`
- Role: Thin wrapper delegating IBKR historical ingestion to the core ingestion engine.
- Priority: `medium`
- Categories: data_ingest
- Phases: phase_architecture_env, phase_data_layer

**Tasks:**
- Ensure it calls core.ib_data_ingestor or scripts/ingest_ibkr_prices via a stable API
- Expose minimal, clear CLI arguments for universe, dates, and environment profile
- Add basic logging and error reporting consistent with other ingestion tools
- Document that it is a convenience wrapper, not a separate ingestion path

### `tools/migrate_utcnow_to_timezone.py`
- Role: Codebase refactoring tool to replace naive datetime.utcnow usage with timezone-aware calls.
- Priority: `low`
- Categories: infra, utils
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Keep script clearly marked as a development/refactor utility only
- Ensure safe dry-run mode and backups before rewriting files
- Align datetime conventions with global time-handling policies defined in config

### `tools/refactor_use_container_width.py`
- Role: Refactoring tool to migrate Streamlit use_container_width usage to the new width API.
- Priority: `low`
- Categories: infra, web_ui, utils
- Phases: phase_web_app_ux, phase_testing_deployment_ai

**Tasks:**
- Confirm it operates only on project files and respects .gitignore or allowlists
- Provide dry-run and diff output options before applying changes
- Align transformations with the current UI helper patterns and key conventions

### `tools/test_ib_basic.py`
- Role: Connectivity and sanity-check script for IBKR API access.
- Priority: `medium`
- Categories: tests, data_ingest
- Phases: phase_architecture_env, phase_observability_ops, phase_testing_deployment_ai

**Tasks:**
- Ensure it never sends live orders and is clearly scoped to read-only checks
- Drive connection parameters (host, port, clientId, environment) from central config
- Report connectivity, latency, and basic symbol data in a structured, parseable format
- Guard execution with explicit environment/profile flags to avoid misuse

### `validate_project_structure.py`
- Role: Validator enforcing project layout, naming, and cohesion rules across the codebase.
- Priority: `high`
- Categories: infra, tests
- Phases: phase_architecture_env, phase_testing_deployment_ai

**Tasks:**
- Codify expectations about directories, module names, and key shared components
- Add checks for duplicate or dead entry points that bypass AppContext and SqlStore
- Expose output as both human-readable and machine-readable for CI gating
- Integrate with system_upgrader_agent to highlight files needing migration
- Ensure validations are fast enough to run frequently during development

### `view_dq_pairs.py`
- Role: Viewing tool for exploring the dq_pairs universe and its key attributes.
- Priority: `medium`
- Categories: web_ui, research
- Phases: phase_data_layer, phase_signals_portfolio, phase_web_app_ux

**Tasks:**
- Refactor to read dq_pairs from the canonical SqlStore/DuckDB location
- Standardize views and metrics with those used in the main web dashboard
- Add environment/profile awareness and clear indication of data freshness
- Prepare a callable function that UI tabs or agents can reuse for dq_pairs views
