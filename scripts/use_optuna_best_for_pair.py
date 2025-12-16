# -*- coding: utf-8 -*-
"""
scripts/use_optuna_best_for_pair.py — Apply Optuna best params to backtest
===========================================================================

מטרה:
------
1. לקרוא קובץ CSV ש-Optuna ייצר (optuna_backtest_search).
2. לבחור את ה-trial הכי טוב (לפי value, כלומר Sharpe / PnL).
3. להוציא:
   • פקודת CLI מלאה ל-root.backtest עם אותם פרמטרים.
   • Snippet JSON שאפשר להדביק ל-opt_best_params_registry / session_state.

שימוש:
------
python -m scripts.use_optuna_best_for_pair ^
  --pair XLP XLY ^
  --results results/optuna_XLP_XLY_sharpe_core.csv ^
  --start 2018-01-01 ^
  --end 2025-12-05 ^
  --direction maximize ^
  --min-trades 10
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


logger = logging.getLogger("UseOptunaBest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


# מיפוי שמות פרמטרים → דגלי CLI של root.backtest
PARAM_TO_CLI_FLAG: Dict[str, str] = {
    "z_entry": "z_entry",
    "z_exit": "z_exit",
    "lookback": "lookback",
    "atr_window": "atr_window",
    "edge_min": "edge_min",
    "atr_max": "atr_max",
    "corr_min": "corr_min",
    "beta_lo": "beta_min",   # מיוחד: עובר ל-beta_min
    "beta_hi": "beta_max",   # מיוחד: עובר ל-beta_max
    "coint_pmax": "coint_pmax",
    "half_life_max": "half_life_max",
    "notional": "notional",
    "slippage_bps": "slippage_bps",
    "slippage_mode": "slippage_mode",
    "slippage_atr_frac": "slippage_atr_frac",
    "transaction_cost_per_trade": "transaction_cost_per_trade",
    "bar_lag": "bar_lag",
    "max_bars_held": "max_bars_held",
    "z_stop": "z_stop",
    "run_dd_stop_pct": "run_dd_stop_pct",
}


def _load_trials(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path)
    if "state" not in df.columns or "value" not in df.columns:
        raise ValueError("CSV does not look like optuna_backtest_search output (missing 'state' / 'value').")
    return df


def _select_best_trial(
    df: pd.DataFrame,
    *,
    direction: str = "maximize",
    min_trades: int = 0,
) -> pd.Series:
    """בוחר את ה-trial הכי טוב לפי direction ו-min_trades."""
    df = df.copy()

    # רק COMPLETE
    df = df[df["state"] == "COMPLETE"]
    if df.empty:
        raise ValueError("No COMPLETE trials found in CSV.")

    # אם יש metric_Trades → נסנן לפי מינימום טריידים
    trades_col_candidates = [c for c in df.columns if c.lower() in ("metric_trades", "metric_n_trades")]
    if trades_col_candidates and min_trades > 0:
        tcol = trades_col_candidates[0]
        df = df[df[tcol] >= min_trades]
        if df.empty:
            raise ValueError(f"No trials with >= min_trades={min_trades} trades.")

    # בחירה לפי value
    if direction == "maximize":
        best_idx = df["value"].idxmax()
    else:
        best_idx = df["value"].idxmin()
    best = df.loc[best_idx]
    return best


def _extract_params_from_row(row: pd.Series) -> Dict[str, Any]:
    """מוציא את כל param_* → dict {name: value}."""
    params: Dict[str, Any] = {}
    for col in row.index:
        if not col.startswith("param_"):
            continue
        name = col[len("param_") :]
        val = row[col]
        # המרה נעימה לסוגים
        if isinstance(val, float) and np.isnan(val):
            continue
        if isinstance(val, str):
            # מורידים רווחים מיותרים
            params[name] = val.strip()
        else:
            params[name] = val
    return params


def _format_value_for_cli(v: Any) -> str:
    """פורמט אחיד ל-CLI (int/float/str)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):g}"
    return str(v)


def build_cli_command(
    sym_x: str,
    sym_y: str,
    start: str,
    end: str,
    params: Dict[str, Any],
) -> str:
    """
    בונה פקודת CLI ל-root.backtest שמכילה את הפרמטרים מה-Optuna best trial.

    הערות:
    -------
    • משתמש במיפוי PARAM_TO_CLI_FLAG (כולל beta_lo → beta_min, beta_hi → beta_max).
    • מדלג על פרמטרים שאינם מוכרים ל-CLI.
    """
    parts: List[str] = [
        "python",
        "-m",
        "root.backtest",
        sym_x,
        sym_y,
        "--start",
        start,
        "--end",
        end,
    ]

    for name, value in params.items():
        flag_name = PARAM_TO_CLI_FLAG.get(name)
        if not flag_name:
            # פרמטרים שהם לא "core" ל-CLI (למשל gating מתקדמים) מדולגים כאן,
            # אבל עדיין נשמרים ב-JSON של ה-Preset.
            continue
        parts.append(f"--{flag_name}")
        parts.append(_format_value_for_cli(value))

    return " ".join(parts)


def build_registry_snippet(
    pair_label: str,
    params: Dict[str, Any],
    score: float,
    profile: str = "core_sharpe",
) -> str:
    """
    בונה JSON snippet לשימוש ב-opt_best_params_registry / session_state.

    דוגמה:
    -------
    {
      "XLP-XLY": {
        "params": {...},
        "score": 1.23,
        "profile": "core_sharpe",
        "updated_at": "2025-12-09T17:30:00Z"
      }
    }
    """
    payload = {
        pair_label: {
            "params": params,
            "score": float(score),
            "profile": profile,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Use Optuna best trial for a given pair.")
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("SYM_X", "SYM_Y"),
        required=True,
        help="Pair symbols, e.g. XLP XLY",
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to optuna results CSV from scripts.optuna_backtest_search",
    )
    parser.add_argument("--start", type=str, required=True, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument(
        "--direction",
        type=str,
        choices=("maximize", "minimize"),
        default="maximize",
        help="Optuna study direction (usually 'maximize' for Sharpe / PnL)",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=0,
        help="Filter out trials with fewer than this number of trades (0 = no filter).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="core_sharpe",
        help="Profile label to store in registry snippet.",
    )

    args = parser.parse_args(argv)

    sym_x, sym_y = args.pair
    sym_x = sym_x.upper()
    sym_y = sym_y.upper()
    pair_label = f"{sym_x}-{sym_y}"

    results_path = Path(args.results)
    df = _load_trials(results_path)

    logger.info(
        "Loaded %d trials from %s (states=%s)",
        len(df),
        results_path,
        sorted(df["state"].unique().tolist()),
    )

    best = _select_best_trial(
        df,
        direction=args.direction,
        min_trades=args.min_trades,
    )

    params = _extract_params_from_row(best)
    score = float(best["value"])

    logger.info("Best trial: id=%s | value=%s", best.get("trial_id", best.name), score)
    logger.info("Params used (%d): %s", len(params), ", ".join(sorted(params.keys())))

    # ---- CLI command ----
    cli_cmd = build_cli_command(
        sym_x=sym_x,
        sym_y=sym_y,
        start=args.start,
        end=args.end,
        params=params,
    )
    print("\n================ CLI command (root.backtest) ================\n")
    print(cli_cmd)
    print("\n=============================================================\n")

    # ---- Registry snippet ----
    reg_snippet = build_registry_snippet(
        pair_label=pair_label,
        params=params,
        score=score,
        profile=args.profile,
    )
    print("========== opt_best_params_registry snippet (JSON) ==========\n")
    print(reg_snippet)
    print("\n=============================================================\n")


if __name__ == "__main__":
    main()
