# -*- coding: utf-8 -*-
"""
health_check_full_system.py — HF-grade End-to-End Health Check
==============================================================

מטרות (ברמת קרן):
- בדיקה מקצה לקצה של wiring בסיסי: AppContext -> SqlStore -> data tables -> service factories
- מינימום side effects: לא פותח חיבורי IBKR, לא מריץ Streamlit, ולא יוצר פעולות מסחר
- דיווח מקצועי: OK/WARN/FAIL/SKIP, עם פירוט, זמן ריצה לכל בדיקה, ויציאה עם קוד סטטוס אופציונלי (strict)

עקרונות:
- CLI mode: לא מייבאים/מריצים root tabs (Streamlit) כדי להימנע מ-warnings ורעש.
- Optional data (signals_universe / bt_runs): WARN אם חסר, לא FAIL (כי זה Phase 2/3).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.app_context import AppContext
from core.sql_store import SqlStore


# =============================================================================
# Models / Status
# =============================================================================

class CheckStatus(str, Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    details: Optional[str] = None
    duration_ms: Optional[int] = None


# =============================================================================
# Output helpers
# =============================================================================

def _section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_result(r: CheckResult) -> None:
    tag = f"{r.status.value:4}"
    dur = f" ({r.duration_ms}ms)" if r.duration_ms is not None else ""
    print(f"[{tag}] {r.name}{dur}")
    if r.details:
        print("  " + r.details.replace("\n", "\n  "))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _run_check(name: str, fn: Callable[[], Tuple[CheckStatus, str]]) -> CheckResult:
    t0 = _now_ms()
    try:
        status, details = fn()
        t1 = _now_ms()
        return CheckResult(name=name, status=status, details=details, duration_ms=(t1 - t0))
    except Exception:
        t1 = _now_ms()
        tb = traceback.format_exc()
        return CheckResult(name=name, status=CheckStatus.FAIL, details=tb, duration_ms=(t1 - t0))


def _is_streamlit_runtime() -> bool:
    """
    True only when running under Streamlit runtime with a valid session context.
    Uses safe_session_state to avoid warnings in CLI.
    """
    try:
        from common.streamlit_guard import safe_session_state
        return safe_session_state() is not None
    except Exception:
        return False


# =============================================================================
# Checks
# =============================================================================

def check_app_context() -> Tuple[CheckStatus, str]:
    app_ctx = AppContext.get_global()

    env = getattr(app_ctx, "environment", getattr(app_ctx.settings, "env", "dev"))
    profile = getattr(app_ctx, "profile", getattr(app_ctx.settings, "profile", "default"))
    run_id = getattr(app_ctx, "run_id", None)
    seed = getattr(app_ctx, "seed", None)

    prov = getattr(app_ctx, "provenance", None)
    git_rev = getattr(prov, "git_rev", None) if prov is not None else None
    created_at = getattr(prov, "created_at", None) if prov is not None else None

    project_root = getattr(app_ctx, "project_root", None)

    details = (
        f"env={env}, profile={profile}, run_id={run_id}, seed={seed}, "
        f"git_rev={git_rev}, created_at={created_at}, project_root={project_root}"
    )
    return CheckStatus.OK, details


def check_sql_store(app_ctx: AppContext) -> Tuple[CheckStatus, str]:
    store = SqlStore.from_settings(app_ctx.settings)
    info = store.describe_engine()

    details_lines = [
        f"engine_url={info.get('engine', {}).get('engine_url')}",
        f"dialect={info.get('engine', {}).get('dialect')}",
        f"default_env={info.get('engine', {}).get('default_env')}",
        f"tables_count={info.get('tables_count')}",
        f"has_prices={info.get('has_prices')}",
        f"has_dq_pairs={info.get('has_dq_pairs')}",
        f"has_kv_store={info.get('has_kv_store')}",
        f"has_dashboard_snapshots={info.get('has_dashboard_snapshots')}",
    ]
    return CheckStatus.OK, "\n".join(details_lines)


def check_prices(store: SqlStore) -> Tuple[CheckStatus, str]:
    env = store.default_env or "dev"
    cov = store.load_prices_coverage_summary(env=env)

    if cov is None or cov.empty:
        return CheckStatus.FAIL, f"no rows in prices for env={env}"

    details_lines = [
        f"env={env}, rows_total={int(cov['n_rows'].sum())}, symbols={len(cov)}",
        str(cov),
    ]
    return CheckStatus.OK, "\n".join(details_lines)


def check_universe(store: SqlStore) -> Tuple[CheckStatus, str]:
    env = store.default_env or "dev"
    try:
        dq_df = store.load_pair_quality(env=env, latest_only=True)
    except TypeError:
        dq_df = store.load_pair_quality()

    if dq_df is None or dq_df.empty:
        return CheckStatus.FAIL, f"no rows in dq_pairs/pairs_quality for env={env}"

    n_rows = len(dq_df)
    n_pairs = len(dq_df["pair"].unique()) if "pair" in dq_df.columns else n_rows

    details = f"env={env}, pairs={n_pairs}, rows={n_rows}\n{dq_df.head(10)}"
    return CheckStatus.OK, details


def check_signals(store: SqlStore) -> Tuple[CheckStatus, str]:
    """
    signals_universe עדיין יכול להיות "לא פרוס" (Phase 2/3) -> WARN ולא FAIL.
    """
    env = store.default_env or "dev"
    try:
        sig_df = store.load_signals_universe(env=env, latest_only=True, limit=100)
    except TypeError:
        sig_df = store.load_signals_universe()

    if sig_df is None or sig_df.empty:
        return CheckStatus.WARN, f"no rows in signals_universe for env={env} (expected in Phase 2/3)"

    sig_df = sig_df.copy()
    if "pair" not in sig_df.columns and {"sym_x", "sym_y"} <= set(sig_df.columns):
        sig_df["pair"] = sig_df["sym_x"].astype(str) + "-" + sig_df["sym_y"].astype(str)

    n_rows = len(sig_df)
    n_pairs = len(sig_df["pair"].unique()) if "pair" in sig_df.columns else n_rows

    sample_cols = [
        c for c in [
            "pair", "sym_x", "sym_y", "profile_name", "env",
            "signal", "zscore", "edge", "meta_edge_hint", "meta_deploy_tier",
        ]
        if c in sig_df.columns
    ]

    payload = sig_df[sample_cols].head(10) if sample_cols else sig_df.head(10)
    details = f"env={env}, signal_rows={n_rows}, pairs_with_signals={n_pairs}\n{payload}"
    return CheckStatus.OK, details


def check_backtest_metrics(store: SqlStore) -> Tuple[CheckStatus, str]:
    """
    bt_runs/pair_backtest_metrics עדיין יכול להיות "לא פרוס" (Phase 2/3) -> WARN ולא FAIL.
    """
    env = store.default_env or "dev"
    df = store.load_pair_backtest_metrics(env=env, latest_only=True)

    if df is None or df.empty:
        return CheckStatus.WARN, f"no backtest metrics found for env={env} (expected in Phase 2/3)"

    details = f"env={env}, rows={len(df)}\n{df.head(10)}"
    return CheckStatus.OK, details


def check_dashboard_service(run_build_context: bool) -> Tuple[CheckStatus, str]:
    """
    CLI mode: ברירת מחדל לא בונים DashboardContext (זה יכול לגרור Streamlit side-effects).
    """
    factory = importlib.import_module("root.dashboard_service_factory")

    create_dashboard_service = getattr(factory, "create_dashboard_service", None)
    build_default_dashboard_context = getattr(factory, "build_default_dashboard_context", None)

    if create_dashboard_service is None:
        return CheckStatus.FAIL, "root.dashboard_service_factory.create_dashboard_service not found"
    if build_default_dashboard_context is None:
        return CheckStatus.FAIL, "root.dashboard_service_factory.build_default_dashboard_context not found"

    svc = create_dashboard_service()

    if not run_build_context:
        return CheckStatus.OK, f"DashboardService={type(svc).__name__} (context build skipped)"

    ctx = build_default_dashboard_context()
    return CheckStatus.OK, f"DashboardService={type(svc).__name__}, DashboardContext={type(ctx).__name__}"


def check_core_modules(import_ui_tabs: bool) -> Tuple[CheckStatus, str]:
    """
    HF-grade module import check:
    - REQUIRED core modules must import successfully.
    - OPTIONAL modules:
        - if module itself is missing -> SKIP
        - if module exists but fails due to missing dependency or runtime error -> FAIL

    CLI default: no UI tabs.
    """
    required: List[str] = [
        "core.app_context",
        "core.sql_store",
        "core.optimization_backtester",
        "core.full_parameter_optimization",
        "core.meta_optimization",
        "core.metrics",
    ]

    optional: List[str] = []
    if import_ui_tabs:
        optional += [
            "root.smart_scan_tab",
            "root.optimization_tab",
            "root.macro_tab",
            "root.risk_tab",
        ]

    loaded: List[str] = []
    failed: List[str] = []
    skipped: List[str] = []

    def _try_import(name: str, is_required: bool) -> None:
        try:
            importlib.import_module(name)
            loaded.append(name)
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", None)
            # If the module itself is missing (not an internal dependency)
            if missing == name:
                if is_required:
                    failed.append(f"{name} (missing)")
                else:
                    skipped.append(f"{name} (not installed)")
            else:
                # internal dependency missing -> fail (this is a real bug)
                failed.append(f"{name} (dependency missing: {missing})")
        except Exception as e:
            failed.append(f"{name} ({type(e).__name__}: {e})")

    for m in required:
        _try_import(m, True)

    for m in optional:
        _try_import(m, False)

    if failed:
        details = "Loaded:\n- " + "\n- ".join(loaded) if loaded else "Loaded: (none)"
        details += "\n\nFailed:\n- " + "\n- ".join(failed)
        if skipped:
            details += "\n\nSkipped:\n- " + "\n- ".join(skipped)
        return CheckStatus.FAIL, details

    details = "Loaded:\n- " + "\n- ".join(loaded) if loaded else "Loaded: (none)"
    if skipped:
        details += "\n\nSkipped:\n- " + "\n- ".join(skipped)
    return CheckStatus.OK, details


def check_ibkr_helper(no_ib: bool) -> Tuple[CheckStatus, str]:
    """
    IBKR helper check:
    - never sends orders
    - in --no-ib mode: SKIP
    - otherwise calls ib_connection_status(None) if present
    """
    if no_ib:
        return CheckStatus.SKIP, "--no-ib enabled"

    mod = importlib.import_module("root.ibkr_connection")
    status_fn = getattr(mod, "ib_connection_status", None)
    if status_fn is None:
        return CheckStatus.WARN, "ib_connection_status not found in root.ibkr_connection"

    # IMPORTANT: pass None to avoid forcing a connection
    status = status_fn(None)
    return CheckStatus.OK, f"status={status}"


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="HF-grade end-to-end health check")
    ap.add_argument("--strict", action="store_true", help="Treat WARN as FAIL (CI mode).")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    ap.add_argument("--no-ib", action="store_true", help="Skip IBKR helper check.")
    ap.add_argument(
        "--import-ui-tabs",
        action="store_true",
        help="Also import Streamlit tabs (root.*_tab). Default is off in CLI.",
    )
    ap.add_argument(
        "--build-dashboard-context",
        action="store_true",
        help="Actually build DashboardContext (may trigger Streamlit side-effects).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    results: List[CheckResult] = []

    # 1) AppContext
    _section("1) AppContext")
    r1 = _run_check("AppContext.get_global()", lambda: check_app_context())
    _print_result(r1)
    results.append(r1)

    if r1.status == CheckStatus.FAIL:
        # cannot proceed safely
        _section("Summary")
        print("\nAppContext failed; stopping early.")
        return 2

    app_ctx = AppContext.get_global()

    # 2) SqlStore
    _section("2) SqlStore & DuckDB")
    r2 = _run_check("SqlStore.from_settings + describe_engine()", lambda: check_sql_store(app_ctx))
    _print_result(r2)
    results.append(r2)

    store: Optional[SqlStore]
    try:
        store = SqlStore.from_settings(app_ctx.settings)
    except Exception:
        store = None

    # 3) Data coverage
    _section("3) Prices / Universe / Signals / Backtest metrics")
    if store is None:
        r_skip = CheckResult(
            name="SqlStore dependent checks",
            status=CheckStatus.FAIL,
            details="SqlStore could not be instantiated; skipping prices/universe/signals/backtest checks.",
        )
        _print_result(r_skip)
        results.append(r_skip)
    else:
        for nm, fn in [
            ("Prices coverage (prices table)", lambda: check_prices(store)),
            ("Universe / dq_pairs (latest_only)", lambda: check_universe(store)),
            ("Signals Universe (signals_universe)", lambda: check_signals(store)),
            ("Backtest metrics (bt_runs / pair_backtest_metrics)", lambda: check_backtest_metrics(store)),
        ]:
            rr = _run_check(nm, fn)
            _print_result(rr)
            results.append(rr)

    # 4) Dashboard + core modules
    _section("4) Dashboard Service & Core Modules")
    # In CLI, default is to only verify factory imports and svc creation, not context build
    run_build_ctx = bool(args.build_dashboard_context) and _is_streamlit_runtime()
    rr_dash = _run_check(
        "DashboardService + DashboardContext",
        lambda: check_dashboard_service(run_build_context=run_build_ctx),
    )
    _print_result(rr_dash)
    results.append(rr_dash)

    rr_core = _run_check(
        "Core modules import",
        lambda: check_core_modules(import_ui_tabs=bool(args.import_ui_tabs) and _is_streamlit_runtime()),
    )
    _print_result(rr_core)
    results.append(rr_core)

    # 5) IBKR helper
    _section("5) IBKR helper (ibkr_connection)")
    rr_ib = _run_check("IBKR helper (ibkr_connection_status)", lambda: check_ibkr_helper(no_ib=bool(args.no_ib)))
    _print_result(rr_ib)
    results.append(rr_ib)

    # Summary
    _section("Summary")
    counts: Dict[CheckStatus, int] = {s: 0 for s in CheckStatus}
    for r in results:
        counts[r.status] += 1

    total = len(results)
    ok = counts[CheckStatus.OK]
    warn = counts[CheckStatus.WARN]
    fail = counts[CheckStatus.FAIL]
    skip = counts[CheckStatus.SKIP]

    print(f"\nChecks: total={total} | OK={ok} | WARN={warn} | FAIL={fail} | SKIP={skip}")

    if args.json:
        payload = {
            "counts": {k.value: v for k, v in counts.items()},
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "duration_ms": r.duration_ms,
                    "details": r.details,
                }
                for r in results
            ],
        }
        print("\n" + json.dumps(payload, ensure_ascii=False, indent=2))

    # Exit code policy
    if fail > 0:
        return 2
    if args.strict and warn > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
