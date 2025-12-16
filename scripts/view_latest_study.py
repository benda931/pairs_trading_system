#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/view_latest_study.py
============================

Utility script to inspect the *latest* optimisation study (Zoom Campaign)
saved into DuckDB by root.optimization_tab (via save_trials_to_duck).

Design goals
------------
- Robust and explicit.
- Does NOT assume specific columns beyond `study_id`, `datetime_start`, `score`.
- If extra metadata columns exist (pair, sampler, n_trials, timeout_sec) – uses them.
- If they don't exist – degrades gracefully.

Usage
-----
Examples (from project root):

    # 1. View latest study (global)
    python -m scripts.view_latest_study

    # 2. View latest study for a specific pair (only if 'pair' column exists)
    python -m scripts.view_latest_study --pair XLP-XLY

    # 3. Show top 50 trials instead of 20
    python -m scripts.view_latest_study --top 50

    # 4. Order by Sharpe instead of Score
    python -m scripts.view_latest_study --order-by Sharpe
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _resolve_duckdb_path() -> Path:
    """
    Resolve the DuckDB path consistently with SqlStore.

    Priority:
    1. SQL_STORE_URL env var if duckdb:///...
    2. LocalAppData default used by SqlStore.
    3. Fallback: project-local cache.duckdb
    """
    sql_url = os.getenv("SQL_STORE_URL", "")
    if sql_url.startswith("duckdb:///"):
        return Path(sql_url.replace("duckdb:///", "", 1))

    local_app_data = os.getenv("LOCALAPPDATA", "")
    if local_app_data:
        return Path(local_app_data) / "pairs_trading_system" / "cache.duckdb"

    return PROJECT_ROOT / "cache.duckdb"


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class StudyMeta:
    study_id: int
    pair: Optional[str]
    created_at: Optional[pd.Timestamp]
    sampler: Optional[str] = None
    n_trials: Optional[int] = None
    timeout_sec: Optional[int] = None

    @classmethod
    def from_row(cls, row: pd.Series) -> "StudyMeta":
        return cls(
            study_id=int(row["study_id"]),
            pair=str(row["pair"]) if "pair" in row and row["pair"] is not None else None,
            created_at=pd.to_datetime(row["created_at"])
            if "created_at" in row and row["created_at"] is not None
            else None,
            sampler=str(row["sampler"]) if "sampler" in row and row["sampler"] is not None else None,
            n_trials=int(row["n_trials"]) if "n_trials" in row and row["n_trials"] is not None else None,
            timeout_sec=int(row["timeout_sec"]) if "timeout_sec" in row and row["timeout_sec"] is not None else None,
        )


# =============================================================================
# Core logic
# =============================================================================


def _open_duckdb_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    db_path = _resolve_duckdb_path()
    if not db_path.exists():
        raise SystemExit(f"[view_latest_study] DuckDB file not found: {db_path}")
    return duckdb.connect(str(db_path), read_only=read_only)


def _get_trials_columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    info = con.execute("PRAGMA table_info('trials')").df()
    if info.empty:
        raise SystemExit("[view_latest_study] Table 'trials' does not exist in DuckDB.")
    return info["name"].tolist()


def _select_latest_study(
    con: duckdb.DuckDBPyConnection,
    pair: Optional[str] = None,
) -> StudyMeta:
    """
    בוחר את ה-study האחרון על בסיס datetime_start מטבלת trials בדאק-DB.

    לוגיקה:
    - אם יש datetime_start בטבלה → משתמשים בו.
    - אם אין → נופלים חוצה עם הודעה ברורה.
    - בוחרים את study_id עם ה־datetime_start הכי מאוחר (optionally מסונן לפי pair).
    """

    cols = _get_trials_columns(con)
    cols_set = set(cols)

    if "datetime_start" not in cols_set:
        raise SystemExit(
            "[view_latest_study] trials table is missing 'datetime_start' column – "
            "cannot determine latest study."
        )

    has_pair = "pair" in cols_set

    select_parts = [
        "study_id",
        "COUNT(*) AS n_trials",
        "MAX(datetime_start) AS created_at",
    ]
    group_by_parts = ["study_id"]

    if has_pair:
        select_parts.append("any_value(pair) AS pair")
        group_by_parts.append("pair")

    base_sql = f"""
        SELECT
            {", ".join(select_parts)}
        FROM trials
    """

    where_clause = ""
    params: list = []

    if pair is not None:
        if has_pair:
            where_clause = "WHERE pair = ?"
            params.append(pair)
        else:
            print(
                "[view_latest_study] WARNING: 'pair' column not found in trials; "
                "ignoring --pair filter."
            )

    group_by_sql = f"GROUP BY {', '.join(group_by_parts)}"
    order_by_sql = "ORDER BY created_at DESC, study_id DESC"

    meta_sql = f"""
        {base_sql}
        {where_clause}
        {group_by_sql}
        {order_by_sql}
        LIMIT 1
    """

    df_meta = con.execute(meta_sql, params).df()
    if df_meta.empty:
        msg = "[view_latest_study] No trials found in DuckDB 'trials' table"
        if pair:
            msg += f" for pair={pair!r}"
        raise SystemExit(msg)

    row = df_meta.iloc[0]

    # אם StudyMeta.from_row לא דורש sampler/timeout, זה הכי פשוט:
    return StudyMeta.from_row(row)



def _detect_order_column(columns: list[str], requested: Optional[str]) -> str:
    """
    Decide which column to order trials by.
    """
    if requested and requested in columns:
        return requested

    for candidate in ("Score", "Sharpe", "Profit", "score", "value"):
        if candidate in columns:
            return candidate

    # Fallback: first column
    return columns[0]


def load_latest_study_trials(
    pair: Optional[str] = None,
    order_by: Optional[str] = None,
) -> Tuple[StudyMeta, pd.DataFrame, str]:
    """
    High-level API:

    - Connect to DuckDB.
    - Detect latest study for given pair (or globally).
    - Load all trials for that study.
    - Decide order_by column.
    """
    con = _open_duckdb_connection(read_only=True)

    # Sanity: check trials table
    tables = con.execute("PRAGMA show_tables").df()
    if "trials" not in tables["name"].tolist():
        raise SystemExit("[view_latest_study] DuckDB does not contain a 'trials' table.")

    meta = _select_latest_study(con, pair=pair)

    df_trials = con.execute(
        "SELECT * FROM trials WHERE study_id = ?",
        [meta.study_id],
    ).df()

    if df_trials.empty:
        raise SystemExit(
            f"[view_latest_study] No rows found in 'trials' for study_id={meta.study_id}"
        )

    order_col = _detect_order_column(df_trials.columns.tolist(), order_by)
    df_trials = df_trials.sort_values(order_col, ascending=False).reset_index(drop=True)

    return meta, df_trials, order_col


# =============================================================================
# Printing helpers
# =============================================================================


def _print_meta(meta: StudyMeta, order_by: str) -> None:
    print("META:")
    d = asdict(meta)
    if meta.created_at is not None:
        d["created_at"] = meta.created_at.isoformat(sep=" ")
    for k, v in d.items():
        print(f"  {k:12s}: {v}")
    print(f"  {'order_by':12s}: {order_by}")
    print()


def _print_best_trial(df: pd.DataFrame, order_by: str) -> None:
    print("BEST TRIAL (top row after ordering):")
    best = df.iloc[0]

    core_fields = [
        "z_entry",
        "z_exit",
        "lookback",
        "entry_decay",
        "exit_decay",
        order_by,
        "Sharpe",
        "Score",
        "Profit",
        "Drawdown",
    ]
    seen = set()
    for field in core_fields:
        if field in df.columns and field not in seen:
            print(f"  {field:12s}: {best[field]}")
            seen.add(field)
    print()


def _print_head(df: pd.DataFrame, top: int) -> None:
    top = max(1, int(top))
    print(f"HEAD (top {top} rows):")
    print(df.head(top))
    print(f"\n[rows: {len(df)} total]")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="view_latest_study",
        description="Inspect the latest optimisation study stored in DuckDB (Zoom Campaign).",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Optional pair filter, e.g. XLP-XLY (only works if 'pair' column exists).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top rows to display from the study (default: 20).",
    )
    parser.add_argument(
        "--order-by",
        type=str,
        default=None,
        help="Column to order trials by (default: Score→Sharpe→Profit→score/value).",
    )

    args = parser.parse_args()

    meta, df_trials, order_col = load_latest_study_trials(
        pair=args.pair,
        order_by=args.order_by,
    )

    _print_meta(meta, order_by=order_col)
    _print_best_trial(df_trials, order_by=order_col)
    _print_head(df_trials, top=args.top)


if __name__ == "__main__":
    main()
