# ADR-006: Canonical Signal Generation Pipeline

**Status:** Accepted
**Date:** 2026-03-31
**Deciders:** Principal Architecture Review
**Findings Addressed:** P1-PIPE, P1-PORTINT

## Context

Two incompatible signal generation paths exist:
1. **Old path (operational):** `common/signal_generator.py` → `(z_score, beta, spread)` tuples → used by backtest, optimization, dashboards
2. **New path (infrastructure, unused):** `core/threshold_engine.py` → `core/regime_engine.py` → `core/signal_quality.py` → `core/lifecycle.py` → `EntryIntent` objects

The old path is used. The new path is implemented but never called.

## Decision

Create `core/signal_pipeline.py` as the canonical integration bridge.

`signal_pipeline.py` will:
1. Accept spread z-scores and prices from the old path
2. Pass them through ThresholdEngine, RegimeEngine, SignalQualityEngine, and lifecycle checks
3. Return `EntryIntent` / `ExitIntent` objects that are compatible with `portfolio/allocator.py`

This allows the backtest to adopt the new engines incrementally without breaking old workflows.

## Canonical signal output: `core/intents.py:EntryIntent`
All callers should produce `EntryIntent` / `ExitIntent`. The old `(z, beta, spread)` tuples are deprecated for portfolio routing purposes.

## Migration Notes
- `common/signal_generator.py`: NOT deprecated (still produces z-score inputs)
- `core/signals_engine.py`: To be deprecated once signal_pipeline.py is wired
- `core/signal_pipeline.py`: New canonical integration point
