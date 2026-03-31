# Signal Architecture — Pairs Trading System

**Version:** 1.0
**Last updated:** 2026-03-31
**Scope:** Signal engine, regime engine, trade lifecycle, threshold engine, signal quality, analytics, and agent layer

> **Integration Status: PARTIALLY INTEGRATED**
> The signal components (ThresholdEngine, RegimeEngine, SignalQualityEngine,
> TradeLifecycleStateMachine) are individually implemented and tested (86 tests pass).
> The canonical `SignalPipeline` (`core/signal_pipeline.py`, ADR-006) chains them
> together but is **not yet called from the backtester or any operational path** (P1-PIPE).
> The operational backtester uses raw z-score thresholds from `common/signal_generator.py`
> without regime/quality gating.
> See: `docs/INTEGRATION_STATUS.md`, `docs/remediation/remediation_ledger.md:P1-PIPE`

---

## Table of Contents

1. [Overview and Philosophy](#1-overview-and-philosophy)
2. [Signal Architecture](#2-signal-architecture)
3. [Regime Methodology](#3-regime-methodology)
4. [Trade Lifecycle State Machine](#4-trade-lifecycle-state-machine)
5. [Entry Methodology](#5-entry-methodology)
6. [Exit Methodology](#6-exit-methodology)
7. [Threshold Methodology](#7-threshold-methodology)
8. [Signal Quality Philosophy](#8-signal-quality-philosophy)
9. [Portfolio, Risk, and Execution Interaction](#9-portfolio-risk-and-execution-interaction)
10. [ML Role and Constraints](#10-ml-role-and-constraints)
11. [Agent Layer](#11-agent-layer)
12. [Known Limitations](#12-known-limitations)

---

## 1. Overview and Philosophy

### North Star

The signal engine's job is to **propose**, never to decide.  A signal is a
structured rationale artifact — it tells the portfolio layer *why* a position
might be warranted, at what z-score level, under what regime, and with what
confidence.  The portfolio layer decides whether to act; execution decides how.

### Core Doctrines

**Regime is first-class.**  The same z-score means completely different things
in a MEAN_REVERTING regime versus a TRENDING or CRISIS regime.  Every signal
evaluation starts with a regime classification.

**Exit logic is as important as entry logic.**  The system defines 22 explicit
`ExitReason` values (mean-reversion completion, time-based, risk-based,
regime-based, opportunity-cost).  Exit is not an afterthought.

**No static-threshold dogma.**  Thresholds adapt to regime and current spread
volatility.  The same pair can have entry_z=2.0 one month and entry_z=2.8 the
next if vol has elevated.  Every threshold decision is fully auditable.

**Signals are proposals, not automatic truths.**  `EntryIntent`, `ExitIntent`,
and other intent objects flow upward.  They carry block reasons, quality grades,
and rationale strings.  Consumers check `block_reasons` before acting.

**Leakage prevention.**  Every feature computation accepts an `as_of` parameter
and clips data strictly to `data[data.index <= as_of]`.  This is not optional.

---

## 2. Signal Architecture

### Layer Stack

```
┌─────────────────────────────────────────────────────────┐
│  Portfolio / Risk Layer  (consumer of intents)          │
├─────────────────────────────────────────────────────────┤
│  Intent Objects  (core/intents.py)                      │
│    EntryIntent · ExitIntent · WatchIntent · HoldIntent  │
│    ReduceIntent · SuspendIntent · RetireIntent          │
├─────────────────────────────────────────────────────────┤
│  Signal Quality  (core/signal_quality.py)               │
│    Grade A+ → F  ·  size multiplier  ·  ML hook        │
├─────────────────────────────────────────────────────────┤
│  Threshold Engine  (core/threshold_engine.py)           │
│    STATIC / VOL_SCALED / REGIME_CONDITIONED             │
├─────────────────────────────────────────────────────────┤
│  Regime Engine  (core/regime_engine.py)                 │
│    Waterfall rules → ML hook → safety floor             │
├─────────────────────────────────────────────────────────┤
│  Spread State  (core/diagnostics.py)                    │
│    z_score · spread_vol · half_life · correlations      │
└─────────────────────────────────────────────────────────┘
```

### Key Files

| File | Role |
|------|------|
| `core/contracts.py` | All enums: `TradeLifecycleState`, `ExitReason`, `BlockReason`, `IntentAction`, `SignalQualityGrade`, `RegimeLabel` |
| `core/intents.py` | Intent dataclasses + `SignalDecision` wrapper |
| `core/diagnostics.py` | `SpreadStateSnapshot`, `SignalDiagnostics`, `ExitDiagnostics`, `SignalAuditRecord` |
| `core/lifecycle.py` | `TradeLifecycleStateMachine`, `LifecycleRegistry`, `CooldownPolicy` |
| `core/threshold_engine.py` | `ThresholdEngine`, `ThresholdSet`, `ThresholdConfig` |
| `core/signal_quality.py` | `SignalQualityEngine`, `SignalQualityScore`, `MetaLabelProtocol` |
| `core/regime_engine.py` | `RegimeEngine`, `RegimeFeatureSet`, `RegimeTradabilityModifiers` |
| `core/signal_analytics.py` | `SignalAnalytics`, `LifecycleAnalytics`, `ExitAnalytics` |
| `agents/signal_agents.py` | Four signal-domain agents |

---

## 3. Regime Methodology

### Why Regime Is First

Mean reversion requires a stable, cointegrated relationship.  Regimes describe
the *current* state of that relationship.  A pair that was profitable in
MEAN_REVERTING regime can destroy capital when the same code runs during a
TRENDING or CRISIS period.

### Classification Waterfall

The `RegimeEngine` evaluates conditions in strict priority order:

```
1. Data quality check
   └─ If insufficient data → UNKNOWN

2. Structural break detection
   └─ break_confidence > 0.80 OR cusum_stat > threshold
   └─ → BROKEN (safety floor; ML cannot override)

3. Crisis detection
   └─ spread_vol_ratio > 4.0 AND rolling_corr < 0.20
   └─ → CRISIS (safety floor; ML cannot override)

4. Trending detection
   └─ z_persistence > 0.90 OR z_mean_shift > 3.0 AND adf_pass_rate < 0.30
   └─ → TRENDING

5. High volatility
   └─ spread_vol_ratio > 1.5
   └─ → HIGH_VOL

6. Mean reverting
   └─ adf_rolling_pass_rate > 0.60 AND z_persistence < 0.50
   └─ → MEAN_REVERTING

7. ML hook (optional)
   └─ If enabled AND confidence ≥ 0.60 AND not BROKEN/CRISIS
   └─ → ML label (with confidence weighting)

8. Default → UNKNOWN
```

### Feature Set (`RegimeFeatureSet`)

| Feature | Meaning |
|---------|---------|
| `spread_vol_ratio` | Current vol / baseline vol (20d / 252d) |
| `z_persistence` | AR(1) coefficient of z-score series (0=white noise, 1=unit root) |
| `z_mean_shift` | Normalised shift in recent z-score mean |
| `rolling_corr` | Recent rolling correlation of the two legs |
| `corr_drift` | Change in rolling correlation (negative = correlation deteriorating) |
| `half_life` | OU half-life estimate |
| `beta_cv` | Coefficient of variation of rolling hedge ratio (stability proxy) |
| `cusum_stat` | CUSUM statistic (structural break indicator) |
| `break_confidence` | Probability of structural break |
| `adf_rolling_pass_rate` | Fraction of rolling windows where ADF rejects unit root |

### Tradability Modifiers

Each regime produces a `RegimeTradabilityModifiers` record:

| Regime | Entry | Add | Suggest Exit | Size |
|--------|-------|-----|-------------|------|
| MEAN_REVERTING | ✓ | ✓ | ✗ | 1.0× |
| HIGH_VOL | ✓ | ✗ | ✗ | 0.7× |
| TRENDING | ✗ | ✗ | ✓ | 0× |
| CRISIS | ✗ | ✗ | ✓ | 0× |
| BROKEN | ✗ | ✗ | ✓ (immediate) | 0× |
| UNKNOWN | ✓ (cautious) | ✗ | ✗ | 0.5× |

---

## 4. Trade Lifecycle State Machine

### 13 States

```
NOT_ELIGIBLE ──[revalidate]──► WATCHLIST ──[signal_forming]──► SETUP_FORMING
                                   ▲                                  │
                                   │                         [entry_ready]
                            [cooldown_expired]                        ▼
                                   │                           ENTRY_READY
                               COOLDOWN ◄──[exit_filled]──     [entry_submitted]
                                                          │           ▼
                              PENDING_EXIT ◄──[exit_sig]─┤    PENDING_ENTRY
                                                          │     [entry_filled]
                              EXIT_READY ◄──[exit_sig]───┤           ▼
                                                          └────── ACTIVE ─────► SCALING_IN
                                                                     │
                                                               [reduce] ▼
                                                                  REDUCING
                                   ┌─────────────────────────────────┘
                          [suspend]│ (from any active state)
                                   ▼
                               SUSPENDED ──[resume]──► (prior state)
                                   │
                               [retire]
                                   ▼
                               RETIRED (terminal)
```

### State Semantics

| State | Meaning | Typical max duration |
|-------|---------|---------------------|
| NOT_ELIGIBLE | Failed validation or not yet assessed | Indefinite |
| WATCHLIST | Validated; monitoring for setup | Indefinite |
| SETUP_FORMING | Signal beginning to form; pre-entry | 5 days |
| ENTRY_READY | z-score at entry threshold; ready to send order | 3 days |
| PENDING_ENTRY | Order submitted; awaiting fill | 2 days |
| ACTIVE | Position on; monitoring | 30 days |
| SCALING_IN | Adding to position | 2 days |
| REDUCING | Partial exit underway | 2 days |
| EXIT_READY | Exit signal confirmed; ready to send | 2 days |
| PENDING_EXIT | Exit order submitted | 2 days |
| COOLDOWN | Post-exit waiting period | 3–14 days |
| SUSPENDED | Manual hold or regime block | 60 days |
| RETIRED | Terminal; no further transitions | — |

### Cooldown Policy

Cooldown duration depends on *why* the trade exited:

| Exit circumstance | Cooldown |
|-------------------|---------|
| Normal (mean reversion complete) | 3 days |
| After adverse excursion stop | 7 days |
| After regime-driven exit | 5 days |
| After structural break | 14 days |

Longer cooldowns after break exits prevent re-entering a pair whose relationship
has just broken down.

### Audit Trail

Every `TradeLifecycleStateMachine.transition()` call appends a
`LifecycleTransitionRecord` to `sm.history` with:
- `from_state`, `to_state`, `trigger`
- `timestamp`
- Optional `rationale` string

---

## 5. Entry Methodology

### Entry Decision Flow

```
1. Is pair in lifecycle state ENTRY_READY?
2. Is regime entry-blocked? (TRENDING, CRISIS, BROKEN → No)
3. Is quality grade ≥ C? (grade F → block)
4. Is |z_score| ≥ entry_z? (threshold from ThresholdEngine)
5. Is |z_score| < stop_z? (don't enter at diverging extreme)
6. Are all block reasons empty? (risk gates, portfolio limits, etc.)
→ If all pass: emit EntryIntent(action=ENTER)
→ Else: emit WatchIntent(action=WATCH, block_reasons=[...])
```

### Entry Signals vs. Re-entry Signals

After cooldown expiry, re-entry uses `re_entry_z` (slightly wider than `entry_z`)
to avoid whipsaw at the same level that triggered the previous exit.  Set
`is_reentry=True` in `ThresholdEngine.compute()`.

### What Triggers `ENTRY_READY`

The lifecycle state machine transitions to `ENTRY_READY` when the upstream
signal engine (signals_engine.py) produces a sufficient conviction score and
the spread z-score crosses `entry_z`.  The transition uses trigger
`TRIGGER_ENTRY_READY`.

---

## 6. Exit Methodology

### 22 ExitReasons (Grouped)

**Mean-reversion completion (normal exits)**
- `MEAN_REVERSION_COMPLETE` — |z| ≤ exit_z; spread converged to mean
- `TARGET_REACHED` — explicit profit target hit
- `PARTIAL_CONVERGENCE` — halfway to mean; staged profit taking
- `DIMINISHING_EDGE` — conviction has decayed substantially

**Time-based exits**
- `TIME_STOP` — maximum holding period exceeded
- `SIGNAL_DECAY` — z-score signal fading without convergence
- `STALE_TRADE` — position stagnant; capital cost exceeds expected edge

**Risk-based exits**
- `ADVERSE_EXCURSION_STOP` — |z| ≥ stop_z; spread moving against position
- `SPREAD_STOP` — absolute spread price level exceeded risk budget
- `VOLATILITY_SPIKE` — vol spike that invalidates position sizing
- `INSTABILITY_TRIGGERED` — rolling ADF pass rate collapsed
- `BREAK_RISK` — CUSUM near structural break threshold

**Regime-based exits**
- `REGIME_FLIP` — regime changed to TRENDING/CRISIS/BROKEN
- `STRUCTURAL_BREAK` — confirmed structural break in cointegration
- `NOISY_REGIME` — persistent HIGH_VOL degrades signal quality
- `EVENT_CONTAMINATION` — earnings/M&A/macro event contaminating spread

**Opportunity-cost exits**
- `CAPITAL_RECYCLING` — better opportunity available; free up capital
- `PORTFOLIO_CONSTRAINT` — portfolio-level limit requires reduction
- `STAGED_PROFIT_TAKING` — planned partial exit at intermediate levels
- `REGIME_WEAKENING` — regime deteriorating but not yet flipped
- `CONFIDENCE_COLLAPSE` — quality grade dropped to D or F post-entry

**Manual/system exits**
- `MANUAL_EXIT` — operator-initiated
- `KILL_SWITCH` — system-wide emergency exit

### Exit Priority

When multiple exit reasons are present simultaneously, priority:
1. `KILL_SWITCH` (immediate, system-wide)
2. `STRUCTURAL_BREAK`, `REGIME_FLIP` (relationship impaired)
3. `ADVERSE_EXCURSION_STOP` (risk breach)
4. `MEAN_REVERSION_COMPLETE`, `TARGET_REACHED` (normal P&L)
5. `TIME_STOP`, `STALE_TRADE` (decay-based)
6. All others

---

## 7. Threshold Methodology

### Three Modes

**STATIC** — fallback; uses `base_entry_z`, `base_exit_z`, `base_stop_z`
from `ThresholdConfig`.  Always available.

**VOLATILITY_SCALED** — when `current_spread_vol / baseline_spread_vol ≠ 1.0`:
```
entry_z = base_entry_z × vol_multiplier
vol_multiplier = 1 + sensitivity × (vol_ratio - 1)
vol_multiplier capped at [0.8, 2.0]
```
Default `vol_scale_only_widens=True` prevents tightening thresholds when vol
is below baseline (conservative).

**REGIME_CONDITIONED** — when the current regime has explicit overrides in
`ThresholdConfig.regime_thresholds`:

| Regime | entry_z | exit_z | stop_z | no_trade_band |
|--------|---------|--------|--------|---------------|
| MEAN_REVERTING | 2.0 | 0.4 | 3.5 | 0.8 |
| HIGH_VOL | 2.5 | 0.6 | 5.0 | 1.2 |
| TRENDING | — | — | — | — (entry_blocked) |
| CRISIS | — | — | — | — (entry_blocked) |
| BROKEN | — | — | — | — (entry_blocked) |

### Hysteresis

`entry_z > exit_z` is always enforced (minimum gap 0.5).  This prevents
whipsaw: once exited near z=0.4, the position won't re-enter until z returns
to 2.0+.

### Half-Life Calibration

Short half-life (< 5 days): exit_z tightened to ≤ 0.2 (fast reverter
overshoots; take profit quickly).

Long half-life (> 60 days): exit_z widened to ≥ 0.8 (slow reverter; hold
through noise before declaring success).

### Invariants

The `ThresholdSet.__post_init__` method enforces:
- `stop_z ≥ entry_z + 1.0`
- `exit_z ≤ entry_z - 0.5`
- `re_entry_z ≥ entry_z`

---

## 8. Signal Quality Philosophy

### Quality Is Not the Same as Signal Strength

Conviction (from `signal_stack.py`) measures *how strong* a signal is.
Quality measures *how trustworthy* the signal is given regime, MR score,
and features.  A high-conviction signal in a broken regime has grade F.

### Grading Formula

```
composite = 0.40 × conviction
          + 0.30 × mr_score
          + 0.20 × regime_safety
          + 0.10 × feature_score

A+  composite ≥ 0.85
A   composite ≥ 0.70
B   composite ≥ 0.55
C   composite ≥ 0.40
D   composite ≥ 0.25
F   composite < 0.25  (or hard veto triggered)
```

### Hard Vetoes (Always Grade F)

- Regime is CRISIS, BROKEN, or TRENDING
- `mr_score < 0.20` (spread not mean-reverting)

No amount of ML confidence can override a hard veto.  Grade F signals are
never sent to the portfolio layer as `EntryIntent`.

### Size Multipliers by Grade

| Grade | Size Multiplier |
|-------|----------------|
| A+    | 1.25×          |
| A     | 1.00×          |
| B     | 0.75×          |
| C     | 0.50×          |
| D     | 0.25×          |
| F     | 0× (blocked)   |

### ML Meta-Labeling

An optional `MetaLabelProtocol` hook can:
- Upgrade a signal grade by at most **one level** (e.g., B → A)
- Never upgrade grade F
- Supply `ml_probability` (probability of successful trade outcome)

This follows de Prado's meta-labeling framework: the primary model says
"direction"; the secondary model says "should we bet on this direction today?"

---

## 9. Portfolio, Risk, and Execution Interaction

### What the Signal Layer Returns

The signal layer emits `SignalDecision` objects:

```python
SignalDecision(
    intent=EntryIntent(pair_id=..., confidence=0.8, rationale="..."),
    regime_label=RegimeLabel.MEAN_REVERTING,
    quality_grade=SignalQualityGrade.A,
    lifecycle_state=TradeLifecycleState.ENTRY_READY,
    conviction=0.82,
)
```

These are **proposals**.  The portfolio layer must:
1. Check `decision.intent.block_reasons` — empty means no signal-level blocks
2. Apply its own position/concentration/volatility limits
3. Apply Kelly/vol-target sizing adjusted by `quality.size_multiplier`
4. Route to execution only after portfolio-level approval

### Signal Layer Does NOT

- Know current portfolio P&L
- Know account risk limits
- Send orders
- Mutate any shared state

All of these concerns live in upstream layers.

---

## 10. ML Role and Constraints

### Where ML Is Hooked In

| Layer | Hook | Constraint |
|-------|------|-----------|
| Regime engine | `RegimeClassifierHookProtocol` | Cannot override BROKEN or CRISIS |
| Signal quality | `MetaLabelProtocol` | Cannot override grade F; at most +1 grade |
| Threshold engine | ML_PREDICTED mode | Requires `ThresholdSet` return type |

### Safety Floor Doctrine

Rule-based safety floors always win over ML:
- BROKEN regime: ML cannot re-classify to MEAN_REVERTING
- CRISIS regime: ML cannot re-classify to anything tradable
- Grade F: ML cannot upgrade to any tradable grade

This is intentional.  ML models are trained on historical data and may have
seen limited CRISIS/BROKEN regimes.  Hard rule-based floors provide
out-of-distribution safety.

### Leakage Protocol

Any ML model used as a hook must be trained with `train_end` strictly earlier
than the evaluation point.  `DatasetBuilder` and `PurgedKFold` in
`core/ml_validation.py` enforce this.

---

## 11. Agent Layer

### Four Signal Agents (`agents/signal_agents.py`)

**`SignalAnalystAgent`** (`signal_analyst.classify`)
- Input: pair_id, spread series, optional prices, as_of, signal_confidence
- Runs: regime auto-detection → threshold computation → quality assessment → action proposal
- Output: `decision` dict, z_score, thresholds, regime_features, warnings

**`RegimeSurveillanceAgent`** (`regime_surveillance.scan`)
- Input: dict of spread series, optional prior_regimes
- Runs: per-pair regime classification, shift detection, broken/crisis flagging
- Output: regime_map, shift_alerts, instability_alerts, broken_alerts, crisis_alert

**`TradeLifecycleAgent`** (`lifecycle.inspect`)
- Input: states dict, optional entry_timestamps, custom timeout thresholds
- Detects: stale states, timeout risks, action_required, blocked transitions
- Output: stale_alerts, blocked_alerts, action_required, timeout_risk, summary

**`ExitOversightAgent`** (`exit_oversight.scan`)
- Input: list of open position dicts (curr_z, entry_z, regime, age, direction, grade)
- Checks: mean-reversion completion, stop levels, regime flips, time stops, quality degradation
- Output: exit_signals, reduce_signals, risk_alerts, clean_holds, summary

### Agent Contract

All agents:
- Accept `AgentTask`, return `AgentResult` (never raise)
- Call `audit.log()` for every significant decision
- Include full `audit_trail` in `AgentResult`
- Have `REQUIRED_PAYLOAD_KEYS` validated before `_execute` is called
- Are stateless — identical inputs produce identical outputs

---

## 12. Known Limitations

### 1. Regime Lag

Rule-based regime classification uses rolling windows (20d, 60d) which
inherently lag real-time regime changes.  A regime flip may not be detected
until several bars after it occurs.  Mitigation: CUSUM statistics provide
a leading indicator but with higher false-positive rates.

### 2. Half-Life Estimation Noise

The AR(1) half-life estimate is sensitive to outliers and regime changes.
A single regime shift can produce a spuriously long half-life.  Always
pair half-life with the ADF rolling pass rate before acting on it.

### 3. Conviction ≠ Quality

`signal_stack.py` conviction is based on distortion × dislocation × MR quality
× regime safety.  Signal quality adds an additional meta-label layer.  The two
can conflict: high conviction in a regime that's deteriorating → low quality
grade.  Always check quality grade; do not route grade F signals regardless of
conviction.

### 4. Static RegimeFeatureSet in Tests

`RegimeFeatureSet` can be constructed directly with arbitrary values for
testing.  Be careful: directly setting `break_confidence=0.99` will always
trigger BROKEN classification regardless of the spread.  Integration tests
should use `build_regime_features()` from actual spread data.

### 5. No Cross-Pair Regime Correlation

Each pair is classified independently.  If 90% of the portfolio is in
HIGH_VOL regime simultaneously, the system will not automatically reduce
exposure at the portfolio level — that responsibility belongs to the portfolio
risk layer.

### 6. Agent Agents Are Not Real-Time

`RegimeSurveillanceAgent` and `ExitOversightAgent` are designed for periodic
(daily or intraday batch) runs, not tick-by-tick streaming.  For real-time
monitoring, the underlying `RegimeEngine` and `ThresholdEngine` can be called
directly.

### 7. ML Hooks Are Optional Stubs

All three ML hook protocols (`RegimeClassifierHookProtocol`,
`MetaLabelProtocol`, ML threshold override) are defined as protocols with
no concrete implementation provided.  Bringing ML models to production
requires training infrastructure beyond this codebase.
