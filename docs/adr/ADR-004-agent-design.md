# ADR-004: Agent Design Principles

**Status:** Accepted
**Date:** 2026-03-27

## Context

The system needed a way to orchestrate multi-step research and execution workflows
without creating spaghetti dependencies. Previous code had functions calling each
other across module boundaries with no clear ownership or audit trail.

## Decision

All computational workflows are decomposed into agents with narrow mandates:

### Principles

1. **Narrow mandate**: each agent does exactly one job (universe discovery OR
   pair validation OR spread fitting — never all three).

2. **Typed contracts**: agents accept `AgentTask` and return `AgentResult`.
   No implicit state sharing between agents.

3. **Audit trail**: every agent receives an `AgentAuditLogger`. Every significant
   decision (data access, threshold evaluation, rejection) must be logged.

4. **Fail-safe**: `BaseAgent.execute()` wraps `_execute()` in a try/except.
   Agents never raise — they return `AgentResult(status=FAILED, error=...)`.

5. **Permission system**: `AgentRegistry` tracks permissions per agent.
   `EXECUTION` permission requires explicit grant — research agents are `READ_ONLY`.

### Current Agents

| Agent | Task types | Permission |
|-------|-----------|-----------|
| `UniverseDiscoveryAgent` | `discover_pairs` | READ_ONLY |
| `PairValidationAgent` | `validate_pairs`, `validate_single` | READ_ONLY + RESEARCH |
| `SpreadFitAgent` | `fit_spreads` | READ_ONLY + RESEARCH |

### Routing

`AgentRegistry.dispatch(task)` routes by `task.agent_name`. The registry maintains
a bounded audit log of all dispatched tasks (last 10,000 by default).

## Consequences

**Positive:**
- Clear separation of concerns
- Every execution is auditable
- Easy to add new agents without touching existing code

**Negative:**
- More boilerplate than direct function calls
- Passing `pd.DataFrame` through `AgentTask.payload` is not type-safe
  (payload is `dict[str, Any]`). Future work: use typed payload dataclasses.

## Future: Async/Parallel Dispatch

The registry is currently synchronous. `dispatch_batch()` runs tasks sequentially.
A future upgrade should add async dispatch with `asyncio` or a thread pool,
particularly for parallel pair validation across a large universe.
