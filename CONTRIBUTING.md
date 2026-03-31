# Contributing to Pairs Trading System

This document explains how to extend the platform safely without reintroducing
architectural debt or semantic drift.

---

## Before You Start

1. **Read `CLAUDE.md`** — it is the authoritative platform handbook
2. **Read `docs/migration/migration_ledger.md`** — it lists deprecated paths you must not reuse
3. **Read `docs/remediation/remediation_ledger.md`** — it lists known findings and their status
4. **Run the test suite** before and after your changes: `python -m pytest tests/ -v`

## Repository Structure Rules

| Directory | Purpose | May Import From |
|-----------|---------|-----------------|
| `core/` | Domain logic (no UI, no IO) | `core/`, standard library |
| `research/` | Offline research pipeline | `core/`, `common/` |
| `portfolio/` | Capital allocation, risk | `core/`, `common/` |
| `ml/` | ML infrastructure | `core/`, `common/` |
| `agents/` | Agent system | `core/`, `portfolio/`, `ml/`, `common/` |
| `common/` | Shared utilities | standard library, third-party |
| `root/` | Streamlit UI, CLI | anything (but `core/` must NEVER import from `root/`) |
| `runtime/` | Runtime state | `core/`, `common/` |
| `control_plane/` | Control plane | `runtime/`, `core/`, `common/` |
| `governance/` | Governance engine | `core/`, `common/` |

**Critical rule:** `core/` must never import from `root/`, `runtime/`, `control_plane/`,
`governance/`, `agents/`, or `ml/`. Domain logic is environment-agnostic.

## The Single Source of Truth

`core/contracts.py` is the **canonical type registry**. All domain enums, dataclasses,
and protocols must be defined there. Never define domain types in other files.

**Forbidden:**
```python
# DON'T do this in root/my_tab.py
class ExitReason(str, Enum):  # WRONG — duplicate of core.contracts.ExitReason
    ...
```

**Correct:**
```python
from core.contracts import ExitReason  # Always import from contracts
```

## Train/Test Boundary

Every function that estimates parameters must take `train_end: Optional[datetime]`.
Parameters are **always** estimated from `data[data.index <= train_end]`.
This is non-negotiable for research integrity.

## Deprecated Paths — Do Not Use

| Old Path | Use Instead | Reason |
|----------|------------|--------|
| `root/backtest.py:ExitReason` | `core/contracts.py:ExitReason` | Removed duplicate |
| `secrets/` package | `secrets_mgmt/` package | Shadowed stdlib |
| `root.utils`, `root.data_loader` | `common.utils`, `common.data_loader` | Dead aliases removed |
| `root/backtest.py:TradeSide` | `core/contracts.py:SignalDirection` | Deprecated, sunset pending |

See `docs/migration/migration_ledger.md` for the full registry.

## How to Add New Functionality

CLAUDE.md contains step-by-step guides for adding:

- A new discovery family
- A new validation test
- A new spread model
- A new signal family
- A new regime classifier
- A new threshold scheme
- A new lifecycle rule
- A new portfolio constraint / ranking dimension / sleeve
- A new agent
- A new governance policy
- A new alert rule
- A new ML feature / label / model family

**Every addition requires:**
1. Implementation in the correct canonical module
2. Tests in `tests/`
3. Documentation updates (CLAUDE.md section and/or architecture doc)
4. No introduction of duplicate type definitions

## Walk-Forward Disambiguation

- **Calendar stability check:** `root/optimization_tab.py` — splits by calendar segments
  with 63-day minimum floor. This is NOT true walk-forward.
- **True walk-forward:** `research/walk_forward.py:WalkForwardHarness` — expanding-window
  with purged splits and embargo. Use this for rigorous evaluation.

## Test Requirements

Run the full suite before submitting:

```bash
python -m pytest tests/ -v
```

Current expected results: 785 collected, ~768 passing, ~17 pre-existing failures
(see `docs/remediation/remediation_ledger.md` for tracked failures).

**When you touch a layer, verify its tests specifically:**

| Layer | Test File |
|-------|-----------|
| Core contracts | `tests/test_contracts.py` |
| Discovery/validation | `tests/test_discovery.py`, `tests/test_pair_validator.py` |
| Signal engine | `tests/test_signal_engine.py` |
| Portfolio | `tests/test_portfolio.py` |
| ML platform | `tests/test_ml_platform.py` |
| Agents | `tests/test_agent_architecture.py` |
| Governance | `tests/test_governance.py` |
| Production ops | `tests/test_production_ops.py` |
| Remediation | `tests/test_remediation.py` |

## Code Style

- Python 3.13+
- Type hints on all public functions
- Frozen dataclasses for immutable domain objects
- Explicit rejection reasons (never fail silently)
- English for new code comments (existing Hebrew comments are being translated over time)
- Agent `execute()` must never raise — always return `AgentResult` with status
