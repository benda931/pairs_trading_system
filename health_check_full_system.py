# -*- coding: utf-8 -*-
"""
health_check_full_system.py — End-to-End Health Check for Omri's Pairs Trading System
=====================================================================================

מה הסקריפט בודק (כמעט end-to-end):
-----------------------------------
1. AppContext:
   - AppContext.get_global()
   - env / profile / project_root

2. SqlStore / DuckDB:
   - SqlStore.from_settings(AppContext.settings)
   - describe_engine() → engine_url / dialect / tables
   - טבלאות בסיס: prices, dq_pairs, kv_store, dashboard_snapshots

3. Prices:
   - load_prices_coverage_summary(env)
   - מדפיס Symbols, n_rows, min_date/max_date

4. Universe / dq_pairs:
   - load_pair_quality(env, latest_only=True)
   - מדפיס מספר הזוגות וטעימה

5. Signals:
   - load_signals_universe(env, latest_only=True, limit=50)
   - אם אין טבלה / אין דאטה → מדווח, אבל לא מפיל את הסקריפט

6. Backtest / Metrics:
   - load_pair_backtest_metrics(env, latest_only=True)
   - בודק אם bt_runs / metrics קיימים

7. Dashboard Service:
   - import core.dashboard_service_factory
   - create_dashboard_service()
   - build_default_dashboard_context()

8. מודולים קריטיים:
   - core.backtesting או core.backtest
   - core.meta_optimization / full_parameter_optimization
   - root.smart_scan_tab / root.backtest_tab / root.optimization_tab (אם קיימים)

9. IBKR helper:
   - import root.ibkr_connection
   - קורא ib_connection_status() אם קיים, *בלי* לשלוח פקודות.

התוצאה:
--------
הסקריפט מדפיס OK/FAIL לכל חלק, וסיפור קצר על מה קורה.
אם משהו חסר (למשל signals_universe או bt_runs) הוא יציין את זה, אבל ימשיך לשאר הבדיקות.
"""

from __future__ import annotations

from dataclasses import dataclass
from pprint import pprint
from typing import Any, List, Optional

import importlib
import traceback

from core.app_context import AppContext
from core.sql_store import SqlStore


# ============================================================================
# Utilities
# ============================================================================

@dataclass
class CheckResult:
    name: str
    ok: bool
    details: Optional[str] = None


def _section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_result(result: CheckResult) -> None:
    status = "OK " if result.ok else "FAIL"
    print(f"[{status}] {result.name}")
    if result.details:
        print("  " + result.details.replace("\n", "\n  "))


# ============================================================================
# Checks
# ============================================================================

def check_app_context() -> CheckResult:
    try:
        app_ctx = AppContext.get_global()
        env = getattr(app_ctx.settings, "env", "dev")
        profile = getattr(app_ctx.settings, "profile", "default")
        root = getattr(app_ctx, "project_root", None)
        details = f"env={env}, profile={profile}, project_root={root}"
        return CheckResult("AppContext.get_global()", True, details)
    except Exception:
        tb = traceback.format_exc()
        return CheckResult("AppContext.get_global()", False, tb)


def check_sql_store(app_ctx: AppContext) -> CheckResult:
    try:
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
        return CheckResult("SqlStore.from_settings + describe_engine()", True, "\n".join(details_lines))
    except Exception:
        tb = traceback.format_exc()
        return CheckResult("SqlStore.from_settings + describe_engine()", False, tb)


def check_prices(store: SqlStore) -> CheckResult:
    try:
        env = store.default_env or "dev"
        cov = store.load_prices_coverage_summary(env=env)
        if cov is None or cov.empty:
            return CheckResult(
                "Prices coverage (prices table)",
                False,
                f"no rows in prices for env={env}",
            )
        details_lines = [f"env={env}, rows_total={int(cov['n_rows'].sum())}, symbols={len(cov)}"]
        details_lines.append(str(cov))
        return CheckResult("Prices coverage (prices table)", True, "\n".join(details_lines))
    except Exception:
        tb = traceback.format_exc()
        return CheckResult("Prices coverage (prices table)", False, tb)


def check_universe(store: SqlStore) -> CheckResult:
    try:
        env = store.default_env or "dev"
        try:
            dq_df = store.load_pair_quality(env=env, latest_only=True)
        except TypeError:
            dq_df = store.load_pair_quality()

        if dq_df is None or dq_df.empty:
            return CheckResult(
                "Universe / dq_pairs",
                False,
                f"no rows in dq_pairs/pairs_quality for env={env}",
            )

        n_rows = len(dq_df)
        if "pair" in dq_df.columns:
            n_pairs = len(dq_df["pair"].unique())
        else:
            n_pairs = n_rows

        details = f"env={env}, pairs={n_pairs}, rows={n_rows}\n" + str(dq_df.head(10))
        return CheckResult("Universe / dq_pairs (latest_only)", True, details)
    except Exception:
        tb = traceback.format_exc()
        return CheckResult("Universe / dq_pairs (latest_only)", False, tb)


def check_signals(store: SqlStore) -> CheckResult:
    """
    בודק אם קיימת טבלת signals_universe ואם יש בה דאטה.
    אם אין — מדווח FALSE, אבל זה לא חייב להיות "באג" (אלא אם ציפית שיהיו אותות).
    """
    try:
        env = store.default_env or "dev"
        try:
            sig_df = store.load_signals_universe(env=env, latest_only=True, limit=100)
        except TypeError:
            sig_df = store.load_signals_universe()

        if sig_df is None or sig_df.empty:
            return CheckResult(
                "Signals Universe (signals_universe)",
                False,
                f"no rows in signals_universe for env={env}",
            )

        sig_df = sig_df.copy()
        if "pair" not in sig_df.columns and {"sym_x", "sym_y"} <= set(sig_df.columns):
            sig_df["pair"] = sig_df["sym_x"].astype(str) + "-" + sig_df["sym_y"].astype(str)

        n_rows = len(sig_df)
        n_pairs = len(sig_df["pair"].unique()) if "pair" in sig_df.columns else n_rows

        details_lines = [f"env={env}, signal_rows={n_rows}, pairs_with_signals={n_pairs}"]
        sample_cols = [
            c
            for c in [
                "pair",
                "sym_x",
                "sym_y",
                "profile_name",
                "env",
                "signal",
                "zscore",
                "edge",
                "meta_edge_hint",
                "meta_deploy_tier",
            ]
            if c in sig_df.columns
        ]
        if sample_cols:
            details_lines.append(str(sig_df[sample_cols].head(10)))
        else:
            details_lines.append(str(sig_df.head(10)))

        return CheckResult("Signals Universe (signals_universe)", True, "\n".join(details_lines))
    except Exception:
        tb = traceback.format_exc()
        return CheckResult("Signals Universe (signals_universe)", False, tb)


def check_backtest_metrics(store: SqlStore) -> CheckResult:
    """
    בודק האם קיימת טבלת bt_runs / metrics דרך load_pair_backtest_metrics.
    """
    try:
        env = store.default_env or "dev"
        df = store.load_pair_backtest_metrics(env=env, latest_only=True)
        if df is None or df.empty:
            return CheckResult(
                "Backtest metrics (bt_runs / pair_backtest_metrics)",
                False,
                f"no backtest metrics found for env={env}",
            )
        details = f"env={env}, rows={len(df)}\n" + str(df.head(10))
        return CheckResult("Backtest metrics (bt_runs / pair_backtest_metrics)", True, details)
    except Exception:
        tb = traceback.format_exc()
        return CheckResult("Backtest metrics (bt_runs / pair_backtest_metrics)", False, tb)


def check_dashboard_service() -> CheckResult:
    """
    בודק שה-DashboardService ו-DashboardContext יכולים להיבנות.
    """
    try:
        factory = importlib.import_module("root.dashboard_service_factory")
        create_dashboard_service = getattr(factory, "create_dashboard_service")
        build_default_dashboard_context = getattr(factory, "build_default_dashboard_context")

        svc = create_dashboard_service()
        ctx = build_default_dashboard_context()

        details = f"DashboardService={type(svc).__name__}, DashboardContext={type(ctx).__name__}"
        return CheckResult("DashboardService + DashboardContext", True, details)
    except Exception:
        tb = traceback.format_exc()
        return CheckResult("DashboardService + DashboardContext", False, tb)


def check_core_modules() -> CheckResult:
    """
    בודק טעינת מודולי ליבה חשובים (בלי להריץ אותם):

    - core.backtesting או core.backtest
    - core.meta_optimization / full_parameter_optimization
    - root.smart_scan_tab / root.backtest / root.optimization_tab (אם קיימים)
    """
    modules_to_try: List[str] = [
        "core.backtesting",
        "core.backtest",
        "core.full_parameter_optimization",
        "core.meta_optimization",
        "core.optimization_backtester",
        "core.ml_analysis",
        "root.smart_scan_tab",
        "root.backtest_tab",
        "root.optimization_tab",
        "root.macro_tab",
        "root.risk_tab",
    ]
    loaded: List[str] = []
    failed: List[str] = []

    for mod_name in modules_to_try:
        try:
            importlib.import_module(mod_name)
            loaded.append(mod_name)
        except ModuleNotFoundError:
            # לא נחשב ככישלון קשה – אולי מודול לא קיים במערכת הנוכחית
            continue
        except Exception:
            failed.append(mod_name)

    if failed:
        details = f"Loaded: {loaded}\nFailed: {failed}"
        return CheckResult("Core modules import", False, details)

    details = f"Loaded: {loaded}"
    return CheckResult("Core modules import", True, details)


def check_ibkr_helper() -> CheckResult:
    """
    בודק חיבור בסיסי ל-helper של IBKR (ibkr_connection).
    לא פותח חיבור חדש בכוח, רק מנסה לקרוא ib_connection_status אם קיים.
    """
    try:
        mod = importlib.import_module("root.ibkr_connection")
        status_fn = getattr(mod, "ib_connection_status", None)
        if status_fn is None:
            return CheckResult(
                "IBKR helper (ibkr_connection)",
                False,
                "ib_connection_status function not found in root.ibkr_connection",
            )
        status = status_fn()
        return CheckResult(
            "IBKR helper (ibkr_connection_status)",
            True,
            f"status={status}",
        )
    except Exception:
        tb = traceback.format_exc()
        return CheckResult("IBKR helper (ibkr_connection_status)", False, tb)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    all_results: List[CheckResult] = []

    _section("1) AppContext")
    r_ctx = check_app_context()
    _print_result(r_ctx)
    all_results.append(r_ctx)

    if not r_ctx.ok:
        print("\n!! AppContext נכשל — שאר הבדיקות אולי לא רלוונטיות, אבל ננסה להמשיך בזהירות.")
        app_ctx = None
    else:
        # נשתמש ב-AppContext המקורי
        app_ctx = AppContext.get_global()

    if app_ctx is None:
        # אין טעם להמשיך לשאר ה-checkים שתלויים ב-AppContext
        print("\nAppContext לא זמין — סיום הסקריפט.")
        return

    _section("2) SqlStore & DuckDB")
    r_store = check_sql_store(app_ctx)
    _print_result(r_store)
    all_results.append(r_store)

    # נבנה SqlStore לעבודה בהמשך (אם זה נכשל – חלק מהבדיקות ייכשלו גם)
    try:
        store = SqlStore.from_settings(app_ctx.settings)
    except Exception:
        store = None

    if store is not None:
        _section("3) Prices / Universe / Signals / Backtest metrics")

        r_prices = check_prices(store)
        _print_result(r_prices)
        all_results.append(r_prices)

        r_universe = check_universe(store)
        _print_result(r_universe)
        all_results.append(r_universe)

        r_signals = check_signals(store)
        _print_result(r_signals)
        all_results.append(r_signals)

        r_bt = check_backtest_metrics(store)
        _print_result(r_bt)
        all_results.append(r_bt)
    else:
        print("\n!! SqlStore לא זמין — מדלגים על בדיקות prices/universe/signals/backtest.")

    _section("4) Dashboard Service & Core Modules")

    r_dash = check_dashboard_service()
    _print_result(r_dash)
    all_results.append(r_dash)

    r_core = check_core_modules()
    _print_result(r_core)
    all_results.append(r_core)

    _section("5) IBKR helper (ibkr_connection)")

    r_ib = check_ibkr_helper()
    _print_result(r_ib)
    all_results.append(r_ib)

    _section("Summary")

    ok_count = sum(1 for r in all_results if r.ok)
    total = len(all_results)
    print(f"\nChecks passed: {ok_count}/{total}")

    if ok_count == total:
        print("\n✅ System looks healthy end-to-end (בהינתן הדאטה הקיימת).")
    else:
        print("\n⚠️ חלק מהבדיקות נכשלו — מומלץ לעבור על ה-FAILים למעלה ולראות מה חסר/שבור.")


if __name__ == "__main__":
    main()
