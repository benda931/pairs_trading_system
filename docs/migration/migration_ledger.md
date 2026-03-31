# Migration Ledger -- Pairs Trading System
**Version:** 1.0
**Created:** 2026-03-31
**Purpose:** Track every legacy path, deprecation, compatibility shim, and removal across the platform.

## How to Use This Ledger
- Every migrated/deprecated/removed item has a stable ID (MIG-xxx)
- Before adding new code, check whether the concept you need already has a canonical path here
- Do NOT reintroduce removed items without an ADR

## Migration Registry

### MIG-001: ExitReason Duplicate (REMOVED)

| Field | Value |
|-------|-------|
| **Legacy Path** | `root/backtest.py:ExitReason` (10 values: NORMAL, Z_EXIT, Z_STOP, MAX_BARS, ...) |
| **Canonical Replacement** | `core/contracts.py:ExitReason` (19 values, semantically richer) |
| **Classification** | F -- Immediate Removal |
| **Risk if Left** | Audit trail confusion: trade exits classified with incompatible enum depending on code path |
| **Compatibility Strategy** | Full Removal -- enum was defined but never referenced (zero usages of `ExitReason.` in file) |
| **Migration Status** | COMPLETE |
| **Tests Updated** | N/A (no tests referenced the local enum) |
| **Docs Updated** | This ledger |
| **Removal Date** | 2026-03-31 |

---

### MIG-002: TradeSide Duplicate (DEPRECATED, SUNSET PENDING)

| Field | Value |
|-------|-------|
| **Legacy Path 1** | `root/backtest.py:TradeSide` (SHORT_LONG, LONG_SHORT, FLAT) |
| **Legacy Path 2** | `root/trade_logic.py:TradeSide` (LONG, SHORT) |
| **Canonical Replacement** | `core/contracts.py:SignalDirection` (LONG_SPREAD, SHORT_SPREAD, FLAT, EXIT) |
| **Classification** | D -- Deprecated and Restricted |
| **Risk if Left** | Two incompatible TradeSide enums with different semantics; confusion for contributors |
| **Compatibility Strategy** | Freeze and Sunset -- both are used internally in their files (~1-2 references each) |
| **Owner** | Migration backlog |
| **Sunset Condition** | Migrate `root/backtest.py` line ~985 and `root/trade_logic.py` line ~367 to use `SignalDirection`, then remove both |
| **Migration Status** | DEPRECATED (deprecation comment added 2026-03-31) |

---

### MIG-003: Walk-Forward Floor Fix (COMPLETED)

| Field | Value |
|-------|-------|
| **Legacy Behavior** | `optimization_tab.py:6666`: `max(5, min_seg_days)` with default 10 |
| **Canonical Behavior** | `max(63, min_seg_days)` with default 63 (per R-001 / ADR-007) |
| **Classification** | Corrective Fix |
| **Risk if Left** | 5-day WF segments allow parameter overfitting; backtest credibility compromised |
| **Migration Status** | COMPLETE |
| **Tests** | `test_remediation.py::TestR001` should now pass |
| **Docs Updated** | Docstring added to `_run_walkforward_for_params` noting calendar-segment limitation |

---

### MIG-004: Backup Files (REMOVED)

| Field | Value |
|-------|-------|
| **Removed Files** | `common/ui_helpers.py.bak`, `core/sql_store.py.gpt_backup`, `hedge_fund_upgrade_agent.py.gpt_backup`, `root/dashboard.py.bak`, `root/dashboard_home_v2.py.bak`, `root/matrix_research_tab.py.bak`, `root/smart_scan_tab.py.bak`, `root/optimization_tab.dedup.py`, `root/optimization_tab.backup_quotes_fix.py` |
| **Classification** | F -- Immediate Removal |
| **Risk if Left** | Contributor confusion; stale imports in backup files; repo clutter |
| **Migration Status** | COMPLETE |
| **Removal Date** | 2026-03-31 |

---

### MIG-005: root/__init__.py Dead Aliases (REMOVED)

| Field | Value |
|-------|-------|
| **Legacy Path** | `root.__init__.ALIASES` mapping `root.utils` -> `common.utils`, etc. |
| **Canonical Path** | Direct imports: `from common.utils import ...`, `from common.data_loader import ...` |
| **Classification** | F -- Immediate Removal |
| **Risk if Left** | Contributors may believe `root.utils` is a valid import path; adds indirection |
| **Evidence** | Zero imports of `root.utils`, `root.data_loader`, `root.config_manager` found in codebase |
| **Migration Status** | COMPLETE |

---

### MIG-006: secrets/ -> secrets_mgmt/ Rename (COMPLETED, PRIOR SESSION)

| Field | Value |
|-------|-------|
| **Legacy Path** | `secrets/` package (shadowed stdlib `secrets` module) |
| **Canonical Path** | `secrets_mgmt/` package |
| **Classification** | Critical Fix (caused `ImportError: cannot import name randbits` system-wide) |
| **Migration Status** | COMPLETE |
| **Residual Risk** | None -- zero references to old `from secrets.` import path remain |

---

### MIG-007: CLAUDE.md Signal Engine Reference (CORRECTED)

| Field | Value |
|-------|-------|
| **Legacy Guidance** | CLAUDE.md line 321: "Add to `core/signals_engine.py`" |
| **Canonical Guidance** | "Add to `core/signal_pipeline.py`" (per ADR-006) |
| **Classification** | G -- Legacy Truthfulness Hazard |
| **Risk if Left** | Contributors add signal families to the 2726-line dead coordinator instead of the canonical pipeline |
| **Migration Status** | COMPLETE |

---

## Canonical Path Summary

| Concept | Canonical Module | Legacy/Deprecated | Status |
|---------|-----------------|-------------------|--------|
| Domain types, enums | `core/contracts.py` | None (unique) | Canonical |
| Exit reasons | `core/contracts.py:ExitReason` | `root/backtest.py:ExitReason` (REMOVED) | Canonical |
| Trade direction | `core/contracts.py:SignalDirection` | `root/backtest.py:TradeSide`, `root/trade_logic.py:TradeSide` (DEPRECATED) | Migration pending |
| Signal pipeline | `core/signal_pipeline.py` | `core/signals_engine.py` (operational but scheduled for deprecation per P3-SIGMIG) | In transition |
| Pair ranking (research) | `core/pair_ranking.py` | None | Research-only |
| Opportunity ranking (portfolio) | `portfolio/ranking.py` | None | Canonical |
| Walk-forward (true) | `research/walk_forward.py` | `root/optimization_tab.py` calendar segments (DOCUMENTED) | Canonical is research/ |
| Kill-switch (tactical) | `portfolio/risk_ops.py:KillSwitchManager` | None | Canonical for portfolio layer |
| Kill-switch (strategic) | `control_plane/engine.py` | None | Canonical for control plane |
| Secret references | `secrets_mgmt/` | `secrets/` (REMOVED) | Canonical |
| Package aliases | Direct imports (`common.utils`, etc.) | `root.__init__.ALIASES` (REMOVED) | Canonical |

## Residual Legacy Debt

| Item | Location | Why It Remains | Sunset Condition |
|------|----------|----------------|------------------|
| `TradeSide` in `root/backtest.py` | Lines ~371-374 | Used at line ~985 for rendering | Migrate to `SignalDirection` |
| `TradeSide` in `root/trade_logic.py` | Lines ~21-23 | Used at line ~367 for trade execution | Migrate to `SignalDirection` |
| `core/signals_engine.py` | 2726 lines | Used by sql_store, orchestrator, app_context | Per P3-SIGMIG: deprecate after signal_pipeline wiring |
| `core/pair_ranking.py` | 770 lines | Used by 1 research script only | Consider moving to `research/` |
| Hebrew comments | ~64 files | Developer language preference | Translate for international accessibility |
