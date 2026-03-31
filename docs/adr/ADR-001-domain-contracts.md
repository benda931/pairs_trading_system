# ADR-001: Single Domain Contracts Module

**Status:** Accepted
**Date:** 2026-03-27

## Context

The codebase had domain types scattered across multiple modules with inconsistent
naming (e.g., hedge ratio enums existed in three places, `PairId` was sometimes
a tuple and sometimes a dataclass). This made it impossible to trace what "a pair"
meant from one module to the next.

## Decision

All domain types, enums, and dataclasses are defined in exactly one place:
`core/contracts.py`.

This includes:
- `PairId` — canonical pair identifier (always lexicographically ordered)
- `PairLifecycleState` — state machine enum
- `ValidationResult`, `ValidationTest`, `PairValidationReport`
- `SpreadModel`, `HedgeRatioMethod`, `SpreadDefinition`
- `AgentTask`, `AgentResult`, `AgentStatus`
- `ValidationThresholds` — default test thresholds

No other module may define these types. All imports are `from core.contracts import ...`.

## Consequences

**Positive:**
- One place to look for what any domain concept means
- Type checking works across the entire codebase
- Adding a new lifecycle state automatically propagates everywhere

**Negative:**
- `core/contracts.py` becomes a large file
- Any circular import must be resolved at the contracts level (use `__future__.annotations`)

## Alternatives Considered

- **Protocol interfaces scattered per domain subdomain**: rejected because it creates
  import cycles and makes the type hierarchy invisible.
- **Pydantic models**: deferred — overkill at current stage, would add startup cost.
