# -*- coding: utf-8 -*-
"""
core/alpha_persistence.py — Alpha Results Persistence to SqlStore
=================================================================

Saves alpha pipeline results (pairs, backtests, equity curves) to
DuckDB via SqlStore so they persist across dashboard sessions.

Fixes #10: "Alpha results don't persist to SqlStore"
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def save_alpha_results_to_store(
    alpha_configs: list[dict],
    equity_curve: Optional[pd.Series] = None,
    trade_log: Optional[list[dict]] = None,
) -> bool:
    """
    Save alpha pipeline results to SqlStore for dashboard persistence.

    Saves to tables:
    - alpha_pairs: pair configs with Sharpe, return, params
    - equity_curve: portfolio equity time series
    - trade_log: individual trade records
    """
    try:
        from core.sql_store import SqlStore

        store = SqlStore()

        # Save alpha pair configs
        if alpha_configs:
            df = pd.DataFrame(alpha_configs)
            df["saved_at"] = datetime.now(timezone.utc).isoformat()
            store.save_dataframe("alpha_pairs", df)
            logger.info("Saved %d alpha pairs to SqlStore", len(df))

        # Save equity curve
        if equity_curve is not None and len(equity_curve) > 0:
            eq_df = pd.DataFrame({
                "date": equity_curve.index,
                "equity": equity_curve.values,
                "pnl": equity_curve.diff().fillna(0).values,
            })
            store.save_dataframe("equity_curve", eq_df)
            logger.info("Saved equity curve (%d bars) to SqlStore", len(eq_df))

        # Save trade log
        if trade_log:
            trades_df = pd.DataFrame(trade_log)
            trades_df["saved_at"] = datetime.now(timezone.utc).isoformat()
            store.save_dataframe("trade_log", trades_df)
            logger.info("Saved %d trades to SqlStore", len(trades_df))

        return True

    except Exception as exc:
        logger.warning("Failed to save to SqlStore: %s", exc)

        # Fallback: save to JSON files
        try:
            out_dir = PROJECT_ROOT / "logs" / "alpha_results"
            out_dir.mkdir(parents=True, exist_ok=True)

            if alpha_configs:
                path = out_dir / "alpha_pairs_latest.json"
                path.write_text(json.dumps(alpha_configs, indent=2, default=str))

            if equity_curve is not None:
                eq_path = PROJECT_ROOT / "logs" / "backtests" / "portfolio_equity_latest.csv"
                eq_path.parent.mkdir(parents=True, exist_ok=True)
                equity_curve.to_csv(eq_path)

            logger.info("Saved alpha results to JSON files (SqlStore fallback)")
            return True

        except Exception as exc2:
            logger.error("Failed to save alpha results: %s", exc2)
            return False


def load_alpha_results_from_store() -> dict:
    """Load persisted alpha results from SqlStore or JSON files."""
    result = {"alpha_configs": [], "equity_curve": None, "trade_log": []}

    try:
        from core.sql_store import SqlStore
        store = SqlStore()

        # Try SqlStore first
        try:
            df = store.read_table("alpha_pairs")
            if not df.empty:
                result["alpha_configs"] = df.to_dict("records")
        except Exception:
            pass

        try:
            eq_df = store.read_table("equity_curve")
            if not eq_df.empty and "equity" in eq_df.columns:
                if "date" in eq_df.columns:
                    eq_df["date"] = pd.to_datetime(eq_df["date"])
                    result["equity_curve"] = eq_df.set_index("date")["equity"]
        except Exception:
            pass

        try:
            trades_df = store.read_table("trade_log")
            if not trades_df.empty:
                result["trade_log"] = trades_df.to_dict("records")
        except Exception:
            pass

    except Exception:
        pass

    # Fallback to JSON files
    if not result["alpha_configs"]:
        try:
            path = PROJECT_ROOT / "logs" / "alpha_results" / "alpha_pairs_latest.json"
            if path.exists():
                result["alpha_configs"] = json.loads(path.read_text())
        except Exception:
            pass

    if result["equity_curve"] is None:
        try:
            eq_path = PROJECT_ROOT / "logs" / "backtests" / "portfolio_equity_latest.csv"
            if eq_path.exists():
                df = pd.read_csv(eq_path, index_col=0, parse_dates=True)
                if not df.empty:
                    result["equity_curve"] = df.iloc[:, 0]
        except Exception:
            pass

    return result
