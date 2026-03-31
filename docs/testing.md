# Testing & Quality Gates — Pairs Trading System

## Test Suite Overview

| Metric | Value |
|--------|-------|
| Test files | 19 |
| Total test functions | ~785 |
| Current pass rate | 765 passed, 17 failed (pre-existing remediation gaps) |
| Framework | pytest |
| Run command | `python -m pytest tests/ -v` |

## Test Taxonomy

### 1. Contract Tests
Verify canonical types, enum values, dataclass construction, invariants.

| File | Tests | Covers |
|------|-------|--------|
| `test_contracts.py` | 16 | PairId ordering, ValidationThresholds, MIN_OBS=252, type construction |

### 2. Domain Logic Tests
Verify core research, signal, and portfolio logic.

| File | Tests | Covers |
|------|-------|--------|
| `test_discovery.py` | 38 | Universe building, candidate generation, validation, train_end discipline |
| `test_pair_validator.py` | 10 | ADF, Hurst, half-life, stability checks, rejection reasons |
| `test_spread_constructor.py` | 18 | OLS, Rolling OLS, Kalman spread fitting |
| `test_walk_forward.py` | 17 | WalkForwardHarness expanding-window splits, embargo, purge |
| `test_signal_engine.py` | 86 | ThresholdEngine, RegimeEngine, SignalQualityEngine, lifecycle state machine, full pipeline traces, temporal integrity |
| `test_portfolio.py` | 82 | CapitalManager, OpportunityRanker, SizingEngine, ExposureAnalyzer, PortfolioAllocator, DrawdownManager, KillSwitchManager, full cycle integration |

### 3. Infrastructure Tests
Verify scaffolded infrastructure in isolation.

| File | Tests | Covers |
|------|-------|--------|
| `test_ml_platform.py` | 115 | Features, labels, datasets, leakage auditor, model registry, inference, governance |
| `test_agent_architecture.py` | 91 | All 33 agents — success/failure cases, audit logging, permissions |
| `test_governance.py` | 100 | Audit chains, evidence, controls, policies, surveillance, attestations, RBAC, retention |
| `test_production_ops.py` | 113 | RuntimeStateManager, ControlPlaneEngine, AlertEngine, ReconciliationEngine, DeploymentEngine |

### 4. Remediation Tests
Verify that tracked P0/P1 findings have been fixed.

| File | Tests | Status |
|------|-------|--------|
| `test_remediation.py` | 32 | 19 passing, 13 failing (pre-existing — P0-KS bridge, P1-GOV gate, P1-SURV2 hook not yet implemented) |

### 5. Integration Tests
Verify cross-layer flows.

| File | Section | What It Tests |
|------|---------|--------------|
| `test_signal_engine.py` | TestFullPipeline | data -> regime -> quality -> threshold -> action |
| `test_portfolio.py` | TestPortfolioIntegration | intents -> rank -> size -> allocate -> snapshot |
| `test_production_ops.py` | TestIntegration | control plane -> kill switch -> alert -> reconciliation |

### 6. Other

| File | Tests | Covers |
|------|-------|--------|
| `test_optimizer.py` | 3 | Optimization parameter handling |
| `test_config.py` | 3 | Configuration loading |
| `test_imports.py` | 2 | Import smoke tests |
| `test_risk_engine.py` | 5 | Risk engine calculations |
| `test_sql_store.py` | 5 | SQL persistence (2 failing — `_ensure_writable` missing) |
| `test_bug_fixes.py` | 12 | Signal generator guards, risk breaches (3 failing) |
| `test_orchestrator.py` | 11 | Daily pipeline orchestration |

---

## Known Failing Tests (17 total)

These are **pre-existing** and tracked in the remediation ledger:

| Category | Count | Root Cause | Remediation ID |
|----------|-------|-----------|----------------|
| Kill-switch bridge | 7 | KillSwitchManager constructor doesn't accept `cfg=` kwarg | P0-KS |
| Governance gate | 2 | `promote()` doesn't call `GovernanceEngine.check_policy()` | P1-GOV |
| Surveillance hook | 3 | `_compute_data_age_hours()` not in data_loader | P1-SURV2 |
| Bug fixes | 3 | Signal generator constant-series guard, risk breach interface | Pre-existing |
| SqlStore | 2 | `_ensure_writable()` method missing | Pre-existing |

---

## Quality Gates

### Before Any Merge / Completion

1. **No new test failures:** `python -m pytest tests/ -v` must not introduce regressions
2. **Type construction valid:** Any new dataclass/enum must live in its canonical module
3. **Temporal discipline:** Any function estimating parameters must respect `train_end`
4. **Import boundaries:** `core/` must never import from `root/`, `runtime/`, `governance/`
5. **Rejection reasons explicit:** Validation failures must populate `rejection_reasons`

### When Touching Specific Layers

| Layer Changed | Must Run | Must Also Check |
|--------------|----------|-----------------|
| `core/contracts.py` | `test_contracts.py` | All downstream tests (types are used everywhere) |
| `research/` | `test_discovery.py`, `test_pair_validator.py`, `test_spread_constructor.py` | `test_walk_forward.py` |
| `core/signal_pipeline.py` | `test_signal_engine.py` | `test_remediation.py` P1-PIPE section |
| `portfolio/` | `test_portfolio.py` | Verify `run_cycle()` contract unchanged |
| `ml/` | `test_ml_platform.py` | Verify inference fallback still works |
| `agents/` | `test_agent_architecture.py` | Verify `execute()` never raises |
| `governance/` | `test_governance.py` | Verify audit chain integrity |
| `runtime/` | `test_production_ops.py` | Verify `is_safe_to_trade()` contract unchanged |
| Documentation | N/A | Verify truthfulness against code (no aspirational claims) |

### Temporal Integrity Checks

These tests verify no future data leakage:
- `test_signal_engine.py::TestAntiLeakage::test_regime_features_respect_as_of`
- `test_signal_engine.py::TestAntiLeakage::test_signal_analyst_agent_clips_to_as_of`
- `test_ml_platform.py::TestPointInTimeFeatureBuilder::test_compute_pair_features_clips_to_as_of`
- `test_discovery.py::test_train_end_respected_in_snapshot`

---

## How to Run Tests

```bash
# Full suite
python -m pytest tests/ -v

# Specific layer
python -m pytest tests/test_signal_engine.py -v

# Only remediation checks
python -m pytest tests/test_remediation.py -v

# Quick smoke test (contracts + imports)
python -m pytest tests/test_contracts.py tests/test_imports.py -v
```
