# ADR-005: Canonical Kill-Switch Architecture

**Status:** Accepted
**Date:** 2026-03-31
**Deciders:** Principal Architecture Review
**Findings Addressed:** P0-KS

## Context

Two independent kill-switch implementations exist with no synchronization:
1. `portfolio/risk_ops.py:KillSwitchManager` — portfolio-layer drawdown-triggered halt
2. `control_plane/engine.py:engage_kill_switch()` — operator-triggered control plane halt

If either fires, the other is not notified. A system with two kill-switches that disagree is more dangerous than one with a single imperfect one.

## Decision

**Canonical owner: `control_plane/engine.py`**

`portfolio/risk_ops.py:KillSwitchManager` is the **trigger detector** — it detects when a kill-switch condition is warranted based on portfolio metrics (drawdown, consecutive losses). When it determines a kill is warranted, it calls `control_plane/engine.py:engage_kill_switch()` as the canonical enforcement point.

`control_plane/engine.py:engage_kill_switch()` propagates to `runtime/state.py:RuntimeStateManager` which is the authoritative runtime state source.

## Implementation Plan
1. Add a `control_plane_callback` parameter to `KillSwitchManager.__init__()`
2. When `check()` detects HARD condition, call the callback if provided
3. The callback should be `control_plane.engage_kill_switch`
4. This preserves backward compatibility: without callback, KillSwitchManager works standalone

## Consequences
- **Positive:** Single canonical runtime state source
- **Positive:** Operator-triggered and system-triggered halts use same enforcement path
- **Negative:** KillSwitchManager needs a dependency on control_plane (acceptable)
- **Residual risk:** If callback is not provided, dual-system problem remains

## Migration Notes
- Old behavior: KillSwitchManager operates standalone (unchanged if no callback provided)
- New behavior: Provide callback at construction time in any live trading context
