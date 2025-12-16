# scripts/view_latest_zoom_sqlite.py

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pandas as pd


def trials_to_df(study: optuna.Study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        row = {
            "trial_no": t.number,
            "score": t.value,
        }
        # פרמטרים (z_entry, z_exit, lookback וכו')
        row.update(t.params)

        # user_attrs מה־Backtester (Sharpe, Profit, Drawdown וכו')
        if t.user_attrs:
            row.update(t.user_attrs)

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View latest zoom Optuna study directly from SQLite (no DuckDB)."
    )
    parser.add_argument("--pair", required=True, help="Pair label, e.g. XLP-XLY")
    parser.add_argument(
        "--storage-url",
        default="sqlite:///data/zoom_studies.db",
        help="Optuna storage URL (default: sqlite:///data/zoom_studies.db)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="How many best trials to display (default: 30)",
    )
    parser.add_argument(
        "--order-by",
        default="score",
        help="Column to order by (default: score)",
    )
    args = parser.parse_args()

    pair = args.pair.upper()
    storage_url = args.storage_url

    # השם של ה־study לפי הזום: zoom::<PAIR>::stage0
    study_name = f"zoom::{pair}::stage0"

    print(f"Loading study from Optuna SQLite:")
    print(f"  storage : {storage_url}")
    print(f"  study   : {study_name}")
    print("")

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    df = trials_to_df(study)

    if df.empty:
        print("No COMPLETE trials found in this study.")
        return

    order_col = args.order_by
    if order_col not in df.columns:
        print(f"[WARN] order-by column {order_col!r} not found, falling back to 'score'.")
        order_col = "score"

    df_sorted = df.sort_values(order_col, ascending=False).reset_index(drop=True)

    best = df_sorted.iloc[0]
    print("BEST TRIAL:")
    print(f"  trial_no : {best['trial_no']}")
    print(f"  score    : {best['score']}")
    for k in ("Sharpe", "Profit", "Drawdown"):
        if k in df_sorted.columns:
            print(f"  {k:<8}: {best[k]}")
    print("")

    print(f"HEAD (top {args.top} rows ordered by {order_col}):")
    print(df_sorted.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
