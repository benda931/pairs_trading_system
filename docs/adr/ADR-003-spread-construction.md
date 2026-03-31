# ADR-003: Spread Construction Separation of Fit and Transform

**Status:** Accepted
**Date:** 2026-03-27

## Context

Early spread construction mixed parameter estimation and application in a single
function, making it impossible to apply a model out-of-sample without re-fitting.
This is a fundamental look-ahead bias risk.

## Decision

All spread constructors implement two separate operations:

1. **`fit(price_x, price_y, *, train_end)`** — estimates parameters from training
   data only. Returns a `SpreadDefinition` (immutable, serialisable).

2. **`transform(price_x, price_y, defn)`** — applies a pre-fitted `SpreadDefinition`
   to any time window (train, test, or live). No parameter updates.

The `SpreadDefinition` dataclass captures everything needed for out-of-sample
application: `hedge_ratio`, `intercept`, `mean`, `std`, `model`, `train_start`,
`train_end`, plus model-specific fields (`window`, `kalman_*`).

## Spread Models

Three models are currently supported:

| Model | `fit` cost | Adapts OOS | Best for |
|-------|-----------|-----------|---------|
| `STATIC_OLS` | O(n) OLS | No | Stable pairs |
| `ROLLING_OLS` | O(n²/w) | Partial (applies last beta) | Slowly drifting pairs |
| `KALMAN` | O(n) | No (uses final state, no new updates) | Fast-adapting research |

**Note on Kalman OOS:** The `transform` step applies the *last estimated state* from
training as a fixed beta. True online Kalman (continuously updating in the test window)
would require re-running the filter — which is a form of look-ahead unless carefully
managed with sequential updates. This tradeoff is documented here and should be
revisited if the system moves to live sequential updates.

## Factory Function

`build_spread(pair_id, prices, *, model, train_end, window, ...)` combines fit + transform
into a single call that returns `(SpreadDefinition, z_score_series)`. The z-score covers
the full price history (train + test) but parameters are from training only.

## Consequences

**Positive:**
- No look-ahead bias in walk-forward validation
- SpreadDefinition can be serialised and reloaded for live trading
- Easy to A/B test multiple spread models on the same pair

**Negative:**
- Rolling OLS `transform()` uses a fixed beta (the last rolling estimate), not a
  continuously-updated rolling beta. This is conservative but misses drift post-train.
