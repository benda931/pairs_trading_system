# scripts/backtest_pair_from_sql.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from datetime import date, datetime
from typing import Any, Dict, Optional

from common.config_manager import load_settings
from core.sql_store import SqlStore
from core.optimization_backtester import run_backtest_for_pair  # להתאים לשם האמיתי

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("backtest_pair_from_sql")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | backtest | %(message)s",
    )


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HF-grade backtest for a single pair using SqlStore prices only."
    )

    parser.add_argument("--pair", type=str, help="Pair in format 'XLP-XLY'")
    parser.add_argument("--sym-x", type=str, help="First leg symbol (e.g. XLP)")
    parser.add_argument("--sym-y", type=str, help="Second leg symbol (e.g. XLY)")

    parser.add_argument(
        "--start",
        type=str,
        default="2018-01-01",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD, default: today)",
    )

    # פרמטרים בסיסיים עם אפשרות override מה-CLI
    parser.add_argument("--z-open", type=float, default=2.0)
    parser.add_argument("--z-close", type=float, default=0.5)
    parser.add_argument("--rolling-window", type=int, default=120)
    parser.add_argument("--max-exposure", type=float, default=0.1)

    parser.add_argument(
        "--n-splits",
        type=int,
        default=1,
        help="Number of walk-forward splits (1 = single backtest)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="manual_cli",
        help="Tag for saving results to SqlStore/experiments table",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="If set, save results to SqlStore experiments table",
    )

    return parser.parse_args()


def _resolve_pair(args: argparse.Namespace) -> tuple[str, str]:
    if args.pair:
        parts = args.pair.replace(" ", "").split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid --pair format: {args.pair!r}")
        return parts[0].upper(), parts[1].upper()

    if args.sym_x and args.sym_y:
        return args.sym_x.strip().upper(), args.sym_y.strip().upper()

    raise ValueError("You must provide either --pair XLP-XLY or --sym-x/--sym-y")


def _build_params(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "z_open": args.z_open,
        "z_close": args.z_close,
        "rolling_window": args.rolling_window,
        "max_exposure_per_trade": args.max_exposure,
    }


def main() -> None:
    _setup_logging()
    args = _parse_args()

    sym_x, sym_y = _resolve_pair(args)
    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end) if args.end else date.today()
    params = _build_params(args)

    logger.info("==== Backtest Started ====")
    logger.info("Pair: %s-%s", sym_x, sym_y)
    logger.info("Date range: %s → %s", start_date, end_date)
    logger.info("Params: %s", params)
    logger.info("n_splits=%s | tag=%s | save=%s", args.n_splits, args.tag, args.save)

    settings = load_settings(PROJECT_ROOT)
    store = SqlStore.from_settings(settings)

    # חשוב: run_backtest_for_pair צריך להשתמש אך ורק ב-SqlStore (prices),
    # לא ב-Yahoo. אם עדיין יש שם קריאה ל-yfinance – זה המקום לנתק אותה.
    result = run_backtest_for_pair(
        sym_x=sym_x,
        sym_y=sym_y,
        store=store,
        start_date=start_date,
        end_date=end_date,
        n_splits=args.n_splits,
        params=params,
        tag=args.tag,
        save_to_store=args.save,
    )

    # מצפה ל- dict עם מפתחות סטנדרטיים; תתאים לשמות האמיתיים שלך.
    sharpe = result.get("sharpe")
    pnl = result.get("pnl")
    max_dd = result.get("max_drawdown")
    trades = result.get("n_trades")
    hit_ratio = result.get("hit_ratio")

    logger.info("==== Backtest Summary ====")
    logger.info("Sharpe:        %.4f", sharpe if sharpe is not None else float("nan"))
    logger.info("Total PnL:     %.2f", pnl if pnl is not None else float("nan"))
    logger.info("Max Drawdown:  %.2f", max_dd if max_dd is not None else float("nan"))
    logger.info("Trades:        %s", trades)
    logger.info("Hit ratio:     %.2f", hit_ratio if hit_ratio is not None else float("nan"))

    print("PAIR:", f"{sym_x}-{sym_y}")
    print("PERIOD:", f"{start_date} → {end_date}")
    print("SHARPE:", sharpe)
    print("PNL:", pnl)
    print("MAX_DD:", max_dd)
    print("TRADES:", trades)
    print("HIT_RATIO:", hit_ratio)

    logger.info("==== Backtest Finished ====")


if __name__ == "__main__":
    main()
