# Portfolio Architecture

> **Integration Status: SCAFFOLD**
> The portfolio package (allocator, ranking, sizing, exposure, risk ops) is implemented
> and tested (82 tests pass). As of 2026-03-31, **PortfolioAllocator.run_cycle() never
> receives real signals from System A.** It is only called from `agents/portfolio_agents.py`
> (which itself is never dispatched operationally).
> See: `docs/INTEGRATION_STATUS.md`, `docs/remediation/remediation_ledger.md:P1-PORTINT`

## Overview

The `portfolio/` package is the **capital allocation and risk operating model** — the
layer that sits between the signal engine and the execution layer.

```
Signal Layer  →  [EntryIntent / ExitIntent]
                         ↓
Portfolio Layer:
  opportunity_ranking  →  OpportunitySet (ranked by composite score)
  capital              →  CapitalManager (sleeve budgets, allocation lifecycle)
  sizing               →  SizingEngine (vol-target + conviction + drawdown scalars)
  exposures            →  ExposureAnalyzer (sector/cluster/shared-leg)
  allocator            →  PortfolioAllocator (main engine: rank → size → check → fund)
  risk_ops             →  DrawdownManager + KillSwitchManager
  analytics            →  PortfolioSnapshot + diagnostics
                         ↓
Execution Layer  →  [AllocationDecision → order routing]
```

---

## Core Doctrine

### 1. A good pair is not automatically a good portfolio position
A pair that scores well statistically may still be:
- Crowding a sector or correlation cluster already over-represented in the portfolio
- Sharing a leg (e.g. AAPL) with 4 other active pairs, creating unintended single-name risk
- Unattractive on a risk-adjusted basis vs other currently available opportunities

### 2. Capital is scarce; signals must compete
Every EntryIntent enters the ranking engine and earns a composite score across 7 dimensions.
Capital flows to the highest-ranked fundable opportunities, respecting all constraints.
Unfunded opportunities are recorded with full rationale — not silently dropped.

### 3. Diversification ≠ number of positions
20 pairs all using AAPL as a leg is not diversification. The ExposureAnalyzer
explicitly tracks shared-leg concentration, sector concentration, and cluster
crowding. All three are hard constraints in the allocator.

### 4. Pair-level neutrality ≠ portfolio-level neutrality
Each pair is individually long/short the spread. But if 15 pairs are all long
tech and short financials, the portfolio has significant factor exposure. Factor
betas can be passed to ExposureAnalyzer for portfolio-level factor exposure tracking.

### 5. All allocation decisions must be auditable
Every AllocationDecision — funded OR rejected — carries:
- The ranked opportunity's score decomposition (7 dimensions)
- The sizing scalar stack (conviction, quality, regime, drawdown, vol-target)
- The constraint check result (which rules fired, severity, current vs limit)
- The AllocationRationale (outcome, capital granted, sleeve, decision notes)

---

## Layer Stack

### portfolio/capital.py — CapitalManager

```
Total Capital = Allocated + Reserved + Free

Sleeves = sub-buckets with max_capital_fraction, max_positions, regime/quality filters.

CapitalManager methods:
  allocate(sleeve, pair_id, amount)   → reserves capital
  confirm_fill(pair_id)               → converts reserved → allocated
  release(pair_id)                    → frees capital on close
  can_allocate(sleeve, amount)        → (bool, reason) — non-mutating check
  sleeve_budget(sleeve_name)          → CapitalBudget (current state)
  free_capital_in_sleeve(sleeve)      → float — most restrictive limit
```

**Key invariant:** `free_capital = total - allocated - reserved`. Never drops below 0.

### portfolio/ranking.py — OpportunityRanker

Converts EntryIntent → RankedOpportunity via 7-dimension scoring:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| signal_strength | 0.25 | \|z\| above entry threshold → saturation |
| signal_quality | 0.20 | Quality grade A+→F translated to [0,1] |
| regime_suitability | 0.20 | MEAN_REVERTING=1.0, CRISIS=0.0 |
| reversion_probability | 0.15 | conviction proxy (ML hook optional) |
| diversification_value | 0.10 | marginal diversification vs active pairs |
| stability | 0.05 | half-life–based stability score |
| freshness | 0.05 | signal recency decay |

**Overlap penalty** (applied post-scoring, not part of weights):
- Shared-leg: −0.10 per pair already using the same instrument
- Cluster crowding: −0.15 when cluster concentration exceeds threshold
- Max total penalty: 0.40 (capped)

**Hard blockers** (pair never gets funded regardless of score):
- quality_grade = "F"
- regime = "CRISIS" or "BROKEN"
- skip_recommended = True (signal analyst veto)

**ML hook:** Implement `RankingMLHookProtocol.score(opportunity) → [0,1]`.
ML score blends with rule-based composite at `ml_blend_fraction` (default 0.20).

### portfolio/sizing.py — SizingEngine

Sizing pipeline per position:

```
1. target_leverage = LeverageEngine(realized_vol, drawdown, vix, regime)
2. base_weight = mode-specific (equal_weight | risk_parity | vol_target)
3. × conviction_scalar   — [0, 1.2]  (dead zone + saturation)
4. × quality_scalar      — [0, 1.2]  (A+=1.2, F=0)
5. × regime_scalar       — [0, 1.1]  (MEAN_REVERTING=1.1, CRISIS=0)
6. × drawdown_scalar     — [0, 1]    (from DrawdownState.throttle_factor)
7. Cap to max_single_pair_weight (default 15%)
8. gross_notional = final_weight × total_capital × target_leverage
9. leg split: leg_x = gross / (1 + hedge_ratio), leg_y = gross - leg_x
10. capital_usage = gross_notional × margin_fraction
```

All scalars recorded in `SizingDecision` for full auditability.

### portfolio/exposures.py — ExposureAnalyzer

Computes ExposureSummary from active allocations:
- Gross/net leverage and their limits
- Per-sector and per-cluster concentration fractions
- SharedLegSummary: instrument → n_pairs_using, total/net notional
- ClusterExposureSummary: cluster → fraction_of_portfolio, is_overcrowded
- Factor exposures (optional, requires factor_betas input)

**Concentration flags:**
- `is_dominant`: instrument used in ≥ 3 pairs OR notional ≥ 20% of portfolio
- `is_overcrowded`: cluster fraction > 25% of gross exposure

### portfolio/allocator.py — PortfolioAllocator

Main engine. `run_cycle(intents, **context) → (decisions, diagnostics)`.

Constraint enforcement order (fast-path, most restrictive first):
1. Kill-switch HARD/REDUCE → block all new entries immediately
2. Heat level HALTED/RECOVERY_ONLY → block all new entries
3. Score below min_score_to_fund → UNFUNDED
4. Position count limit → BLOCKED_CAPITAL
5. Per-cycle new-entry limit → QUEUED
6. Sizing → if not executable: UNFUNDED
7. Risk constraints (sector/cluster/leverage/shared-leg) → BLOCKED_RISK
8. Capital availability → BLOCKED_CAPITAL
9. Partial funding (if enabled and > floor) → PARTIAL_FUNDED
10. ✅ Allocate → FUNDED

All rejected allocations are still recorded in the decisions list with full rationale.

**De-risking:** `compute_derisking(active_allocations, drawdown_state) → DeRiskingDecision`
- HALTED: exit everything
- RECOVERY_ONLY: exit bottom 50% by composite score
- DEFENSIVE: reduce all by (1 − throttle_factor)
- THROTTLED: exit bottom 20%, reduce rest by 20%

### portfolio/risk_ops.py — Risk Operating Model

**4-layer model:**

```
Layer 1: Instrument/Spread
  Per-pair stop-loss z-score enforced at signal layer (stop_z in EntryIntent)
  ThresholdEngine handles this independently

Layer 2: Portfolio
  DrawdownManager.update() → heat_level state machine

Layer 3: Drawdown / Degradation (Heat State Machine)
  NORMAL → CAUTIOUS (3% DD)
  → THROTTLED (6% DD or 5 consecutive losses)
  → DEFENSIVE (10% DD or 8% 30-day rolling)
  → RECOVERY_ONLY (15% DD)
  → HALTED (20% DD)

  Transitions: one-way worsening (automatic)
  Recovery: requires attempt_recovery() + holding period + DD improvement

Layer 4: Kill-switch / Governance
  KillSwitchManager:
    OFF → SOFT (12% DD)
    → REDUCE (16% DD)
    → HARD (20% DD, or 5% single-day loss, or 10 consecutive losses)

  Kill-switch escalates only (never de-escalates automatically).
  Reset requires acknowledgment (configurable).
  Manual trigger: KillSwitchManager.trigger_manual(mode, reason)
```

**Interaction:** Kill-switch and heat level are independent mechanisms.
Heat level drives position sizing throttle; kill-switch drives entry/exit blocking.
Both are checked in `run_cycle`.

### portfolio/analytics.py — Portfolio Analytics

Produces:
- `PortfolioSnapshot` — canonical point-in-time state for downstream layers
- `PortfolioDiagnostics` — cycle metrics (funnel counts, capital usage, risk metrics)
- `OpportunityFunnelReport` — conversion rates at each stage
- `PositionAttribution` — key driver and limiting factor per funded position
- `PortfolioAuditRecord` — master audit record for one cycle

---

## Portfolio Agents (agents/portfolio_agents.py)

6 agents wrapping the portfolio construction layer:

| Agent | Task Type | Purpose |
|-------|-----------|---------|
| PortfolioConstructionAgent | run_allocation_cycle | Full cycle: rank → size → check → allocate |
| CapitalBudgetAgent | check_capital_budget | Non-mutating capital availability check |
| ExposureMonitorAgent | compute_exposure | Current exposure + violation flags |
| DrawdownMonitorAgent | update_drawdown | Heat state machine update |
| KillSwitchAgent | evaluate_kill_switch / manual_trigger / reset_kill_switch | Kill-switch management |
| DeRiskingAgent | compute_derisking | De-risking decisions when stressed |

---

## Data Flow

```
EntryIntent (from signal layer)
   │
   ▼
OpportunityRanker.rank(intents, active_pairs, cluster_map, instrument_map)
   │ → OpportunitySet (sorted by composite_score, blockers last)
   │
   ▼
for each fundable opportunity:
   SizingEngine.size(opportunity, total_capital, spread_vol, hedge_ratio, ...)
   │ → SizingDecision (gross/risk notional, leg sizes, scalar audit trail)
   │
   ▼
PortfolioAllocator._check_constraints(opp, sizing, current_exposure, ...)
   │ → RiskConstraintResult (hard/soft violations)
   │
   ▼
CapitalManager.can_allocate(sleeve, capital_usage)
   │ → (bool, reason)
   │
   ▼
CapitalManager.allocate(sleeve, pair_id, capital_usage)
AllocationDecision recorded (approved=True/False, full rationale)
   │
   ▼
PortfolioAnalytics.build_snapshot(capital_mgr, dd_state, ks_state, decisions)
   │ → PortfolioSnapshot (canonical output for execution layer)
   │
   ▼
Execution Layer → order routing
```

---

## Configuration Reference

### AllocatorConfig
```python
AllocatorConfig(
    ranking=RankingConfig(
        weights=RankingWeights(),     # 7-dimension weights (must sum to 1.0)
        regime_scores={...},           # Regime → [0,1]
        ml_enabled=False,              # Enable ML ranking hook
        ml_blend_fraction=0.20,        # ML weight in composite
    ),
    sizing=SizingConfig(
        mode="vol_target",             # "equal_weight" | "risk_parity" | "vol_target"
        target_portfolio_vol=0.12,     # 12% annualised
        max_leverage=2.0,
        max_single_pair_weight=0.15,   # 15% cap per pair
        conviction_scaling_enabled=True,
    ),
    exposure=ExposureConfig(
        max_sector_fraction=0.40,      # 40% sector concentration limit
        max_cluster_fraction=0.25,     # 25% cluster concentration limit
        shared_leg_threshold=3,        # Pairs using same instrument before flagging
        max_gross_leverage=4.0,
    ),
    max_total_positions=30,
    max_new_entries_per_cycle=5,
    min_score_to_fund=0.15,
    allow_partial_funding=True,
    partial_funding_floor=0.50,
)
```

### DrawdownConfig
```python
DrawdownConfig(
    cautious_dd_threshold=0.03,        # 3% → CAUTIOUS
    throttled_dd_threshold=0.06,       # 6% → THROTTLED
    defensive_dd_threshold=0.10,       # 10% → DEFENSIVE
    recovery_only_dd_threshold=0.15,   # 15% → RECOVERY_ONLY
    halted_dd_threshold=0.20,          # 20% → HALTED
    recovery_step_up_threshold=0.025,  # 2.5% improvement needed to step up
    recovery_min_holding_days=5,       # Days at level before recovery attempt
)
```

---

## How to Add a New Sleeve

1. Define the sleeve in your application startup:
   ```python
   sleeve = SleeveDef(
       name="momentum_pairs",
       max_capital_fraction=0.20,
       allowed_regimes=["TRENDING"],  # Only TRENDING regime pairs
       min_quality_grade="B",
   )
   capital_mgr.add_sleeve(sleeve)
   ```
2. The ranker's `_suggest_sleeve()` assigns pairs to sleeves based on regime + quality.
   Override by modifying `OpportunityRanker._suggest_sleeve()` for custom routing.
3. `SleeveDef.is_regime_allowed(regime)` is called by the sleeve-blocking constraint.

## How to Add a New Ranking Dimension

1. Add a field to `RankedOpportunity` in `portfolio/contracts.py`
2. Add the corresponding weight to `RankingWeights` (adjust others to keep sum=1.0)
3. Compute the score in `OpportunityRanker.rank()` (before composite computation)
4. Update the composite formula in `rank()` and `_apply_diversification_scores()`
5. Add test in `tests/test_portfolio.py::TestOpportunityRanker`

## How to Add a New Constraint

1. Add the enum value to `ConstraintType` in `portfolio/contracts.py`
2. Add the check in `PortfolioAllocator._check_constraints()`
3. Decide severity: "HARD" (blocks) or "SOFT" (warns but doesn't block)
4. Add threshold to `ExposureConfig` or `AllocatorConfig` if tunable
5. Add test in `tests/test_portfolio.py::TestPortfolioAllocator`

## Common Gotchas

- `CapitalManager.can_allocate()` checks both pool-level AND sleeve-level limits.
  A sleeve may have room but the portfolio pool may be exhausted (or vice versa).

- Cluster/sector concentration is skipped for `"UNKNOWN"` assignments.
  If you don't provide `cluster_map` or `sector_map`, concentrations are not enforced.
  Always pass cluster/sector metadata in production.

- `SizingDecision.gross_notional` includes both legs. `capital_usage` is the margin
  requirement (gross × margin_fraction). For risk budgeting use `risk_notional` and
  `risk_contribution`.

- `DrawdownManager.attempt_recovery()` must be called explicitly — the state machine
  never upgrades automatically. Typical pattern: call attempt_recovery() at the start
  of each rebalance cycle if conditions are met.

- Kill-switch escalation is one-way per session. If you want to allow de-escalation
  after clearing conditions, call `KillSwitchManager.acknowledge()` then `reset()`.

- `PortfolioConstructionAgent` creates a fresh `CapitalManager` each cycle. It
  restores existing allocations from `active_allocations` payload to track existing
  exposure correctly. Always pass the current `active_allocations` payload.
