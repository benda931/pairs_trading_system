# Remediation Ledger — Pairs Trading System
**Version:** 1.0
**Date:** 2026-03-31
**Source Review:** Institutional Architecture Review (2026-03-31)
**Status:** Active

## How to Use This Ledger
- Every finding from the institutional review has a stable ID
- Each finding has: severity, root cause, strategy, files, tests, residual risk, status
- Do not close a finding without evidence of fix or explicit downgrade rationale

## Finding Registry

### P0 Findings (Trust-Breaking)

| ID | Title | Severity | Status | Strategy | Evidence Files |
|----|-------|----------|--------|----------|---------------|
| P0-WF | Walk-forward uses calendar segments not true WF | P0 | IN_PROGRESS | Downgrade + min_seg floor | root/optimization_tab.py:7116-7248 |
| P0-EXEC | Execution lag via bar_lag parameter | P0 | COMPLETE | bar_lag consumed in backtester: signals at bar i, fills at bar i+bar_lag via pending_action queue. Default bar_lag=1 (next-bar). bar_lag=0 for legacy same-close. | core/optimization_backtester.py |
| P0-KS | Two disconnected kill-switches with no synchronization | P0 | IN_PROGRESS | Canonicalize via control_plane bridge | portfolio/risk_ops.py, control_plane/engine.py |
| P0-DOCS | Documentation overstates ML integration, agent orchestration, governance enforcement | P0 | IN_PROGRESS | Truthfulness markers in CLAUDE.md + docs | CLAUDE.md, docs/ |

### P1 Findings (Critical Architectural)

| ID | Title | Severity | Status | Strategy | Evidence Files |
|----|-------|----------|--------|----------|---------------|
| P1-PIPE | Canonical signal pipeline wired to backtester | P1 | COMPLETE | SignalPipeline.evaluate_bar() called from backtester when use_signal_pipeline=True | core/signal_pipeline.py, core/optimization_backtester.py |
| P1-PORTINT | Signal-to-portfolio bridge wired | P1 | COMPLETE | core/portfolio_bridge.py extracts EntryIntents from SignalDecisions and feeds them to PortfolioAllocator.run_cycle() | core/portfolio_bridge.py, portfolio/allocator.py |
| P1-MINOBS | Minimum observation count too low (60 days) for AR(1) reliability | P1 | IN_PROGRESS | Raise to 252 | research/pair_validator.py |
| P1-SAFE | Runtime safety gating wired via portfolio bridge | P1 | COMPLETE | bridge_signals_to_allocator() accepts safety_check callback; blocks all entries when unsafe. Architecture boundary preserved — core/ does not import runtime/. | core/portfolio_bridge.py, runtime/state.py |
| P1-SURV | Survivorship bias — no delisted stock filtering documented | P1 | IN_PROGRESS | Document EligibilityFilter as mitigation | research/universe.py |

### P1 Findings (Integration Gaps — Overstated Capabilities)

| ID | Title | Severity | Status | Strategy | Evidence Files |
|----|-------|----------|--------|----------|---------------|
| P1-ML | Meta-label ML overlay: training + inference + pipeline wiring | P1 | COMPLETE | MetaLabelModel training script (scripts/train_meta_label.py), wired to SignalPipeline via ml_quality_hook. Deterministic fallback preserved. 7 integration tests. | scripts/train_meta_label.py, ml/models/meta_labeler.py, core/signal_pipeline.py |
| P1-AGENTS | 2 of 40 agents dispatched via WorkflowEngine | P1 | COMPLETE | run_daily_pipeline() dispatches (1) SystemHealthAgent after health_check and (2) DataIntegrityAgent after data_refresh. Both via WorkflowEngine (monitoring/workflow.py): BOUNDED_SAFE, single-step, no approval gate, typed AgentTask/AgentResult, audit trail, alert bus integration, direct-dispatch fallback. CLI: scripts/run_data_integrity.py. 52 tests (test_agent_dispatch.py). 38 remaining agents scaffold-only. | monitoring/workflow.py, core/orchestrator.py, agents/monitoring_agents.py, tests/test_agent_dispatch.py |
| P1-GOV | Governance never enforced at runtime | P1 | IN_PROGRESS | Wire one gate: model promotion | governance/engine.py, ml/registry/ |
| P1-AUDIT | Audit chains empty for all operational decisions | P1 | DOWNGRADED | Truthful scaffold marking | audit/chain.py |
| P1-SURV2 | SurveillanceEngine never called from operational code | P1 | IN_PROGRESS | Wire one rule: stale data | surveillance/engine.py |

### P2 Findings (Major Weakness)

| ID | Title | Severity | Status | Strategy | Evidence Files |
|----|-------|----------|--------|----------|---------------|
| P0-KS | Kill-switch bridge to control plane | P0 | COMPLETE | KillSwitchManager accepts cfg= and control_plane_callback; callback fires on trigger | portfolio/risk_ops.py |
| P1-GOV | Governance gate on model promotion | P1 | COMPLETE | GovernanceEngine.check_policy() called at module level in promote(); CRITICAL blocks raise ValueError | ml/registry/registry.py |
| P1-SURV2 | Stale data surveillance hook | P1 | COMPLETE | _compute_data_age_hours() + SURV-DI-001 detection in load_price_data(); errors never break loading | common/data_loader.py |
| P2-COSTS | Flat cost model (no volume-based market impact) | P2 | DEFERRED | Document limitation; acceptable for daily | core/optimization_backtester.py |
| P2-DUPRANK | Duplicate pair ranking (core/pair_ranking.py vs portfolio/ranking.py) | P2 | PLANNED | Deprecate core/pair_ranking.py | core/pair_ranking.py |
| P2-DUPTHROT | Duplicate throttle (HeatState vs ThrottleLevel) | P2 | PLANNED | Bridge in P0-KS fix | portfolio/risk_ops.py, control_plane/ |
| P2-COINT | Cointegration stability doesn't hard-reject pairs | P2 | PLANNED | Add hard rejection on low stability score | research/discovery_pipeline.py |
| P2-MLT | Meta-label training script | P2 | COMPLETE | scripts/train_meta_label.py trains meta-label model with point-in-time features and temporal split | scripts/train_meta_label.py |

### P3/P4 Findings

| ID | Title | Severity | Status | Strategy |
|----|-------|----------|--------|----------|
| P3-SIGMIG | signals_engine.py dead coordinator (2700 lines, Hebrew comments) | P3 | PLANNED | Deprecate + redirect to signal_pipeline.py |
| P3-PARTA | Partial fills / bid-ask not modeled | P3 | DEFERRED | Document limitation |
| P4-BACKUP | Backup files in repo (.bak, .gpt_backup) | P4 | DEFERRED | Clean up separately |

## Canonical Ownership Decisions

| Concept | Old Owner(s) | Canonical Owner | Migration Status |
|---------|-------------|-----------------|-----------------|
| Kill-switch state | portfolio/risk_ops.py:KillSwitchManager AND control_plane/engine.py | control_plane/engine.py → propagates to portfolio | Bridge created (P0-KS) |
| Throttle level | portfolio/risk_ops.py:HeatState AND control_plane:ThrottleLevel | control_plane/contracts.py:ThrottleLevel | Bridge planned |
| Signal generation (operational) | common/signal_generator.py | core/signal_pipeline.py (to be created) | In progress |
| Pair ranking (operational) | core/pair_ranking.py | portfolio/ranking.py:OpportunityRanker | core/pair_ranking.py to be deprecated |
| Walk-forward validation | root/optimization_tab.py (calendar WF) | research/walk_forward.py:WalkForwardHarness | Calendar WF downgraded to "stability check" |

## Residual Risk Register

| ID | Risk | Severity | Mitigation | Acceptable Until |
|----|------|----------|-----------|-----------------|
| RR-001 | Walk-forward Sharpe still subject to parameter overfitting | HIGH | Min segment floor 63 days, documented as stability check | Until true WF optimizer is implemented |
| RR-002 | No live/paper trading system | HIGH | All use is backtest/research only | Explicitly documented |
| RR-003 | ML models never trained | HIGH | ML platform marked as scaffold; fallback to neutral (0.5) documented | Until training pipeline exists |
| RR-004 | No runtime safety checks in execution | MEDIUM | Only backtesting; no real capital at risk | Until live system exists |
| RR-005 | Audit chains empty | MEDIUM | No real operational decisions occur outside backtest | Until live system exists |
