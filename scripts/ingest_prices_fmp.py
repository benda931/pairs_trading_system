#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/ingest_prices_fmp.py — Daily FMP Price Ingestion
=========================================================

Fetches daily EOD prices from FMP for a configurable symbol list,
writes them to the canonical SQL `prices` table, and updates the
`data_freshness` tracking table.

Usage
-----
    # Ingest all DQ pairs universe symbols (default)
    python scripts/ingest_prices_fmp.py

    # Specific symbols
    python scripts/ingest_prices_fmp.py --symbols AAPL MSFT SPY QQQ

    # Full backfill (2 years)
    python scripts/ingest_prices_fmp.py --days 730

    # Dry run (print what would be fetched, don't write to SQL)
    python scripts/ingest_prices_fmp.py --dry-run

    # Refresh even if data is fresh
    python scripts/ingest_prices_fmp.py --force

Environment variables
---------------------
    FMP_API_KEY    Override the API key (also readable from config.json)

Exit codes
----------
    0 — success (all symbols fetched)
    1 — partial success (some symbols failed)
    2 — total failure (no data written)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

# ── path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# IMPORTANT: duckdb_engine must load before pandas to avoid segfault
# on Python 3.13 + DuckDB 1.3.x (see core/sql_store.py header).
try:
    import duckdb_engine  # noqa: F401
except ImportError:
    pass

import pandas as pd  # noqa: E402 — after duckdb_engine

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_prices_fmp")


# ── helpers ────────────────────────────────────────────────────────────────

def _load_dq_symbols() -> List[str]:
    """Load symbol list from the DQ pairs universe CSV, if available."""
    candidates = [
        ROOT / "data" / "dq_pairs_universe.csv",
        ROOT / "data" / "pairs_universe.csv",
        ROOT / "pairs_universe.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                syms: set[str] = set()
                for col in ("sym_x", "sym_y", "symbol", "ticker"):
                    if col in df.columns:
                        syms.update(df[col].dropna().str.upper().tolist())
                if syms:
                    logger.info("Loaded %d symbols from %s", len(syms), path)
                    return sorted(syms)
            except Exception as exc:
                logger.warning("Could not load symbols from %s: %s", path, exc)

    # Fallback: S&P 500 sector ETFs + common benchmarks
    logger.info("No universe CSV found; using default benchmark symbols.")
    return [
        "SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "HYG", "LQD",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU", "XLRE",
        "VNQ", "EEM", "EFA", "ARKK", "ARKG", "ARKW",
    ]


def _check_existing_freshness(sql_store, symbols: List[str], stale_days: int) -> List[str]:
    """Return list of symbols that need re-fetching (stale or missing)."""
    try:
        df = sql_store.get_data_freshness()
        if df.empty:
            return symbols

        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=stale_days)).isoformat()
        fresh = set(
            df.loc[
                (df["provider"] == "fmp") & (df["last_fetch_utc"] >= cutoff),
                "symbol",
            ].tolist()
        )
        stale = [s for s in symbols if s not in fresh]
        if fresh:
            logger.info(
                "Skipping %d already-fresh symbols; fetching %d stale/missing.",
                len(fresh) - len([s for s in symbols if s not in fresh]),
                len(stale),
            )
        return stale
    except Exception as exc:
        logger.warning("Could not check freshness; will fetch all: %s", exc)
        return symbols


# ── main ingestion logic ────────────────────────────────────────────────────

def ingest(
    symbols: List[str],
    *,
    start: str,
    end: str,
    dry_run: bool = False,
    batch_size: int = 20,
    fmp_api_key: Optional[str] = None,
) -> dict:
    """
    Fetch prices from FMP and write to SQL.

    Returns a summary dict with success/failure counts.
    """
    from common.fmp_client import get_fmp_client
    from core.sql_store import SqlStore

    client = get_fmp_client(fmp_api_key)
    sql_store = SqlStore.from_settings({})

    total = len(symbols)
    succeeded: List[str] = []
    failed: List[str] = []

    run_id = str(uuid.uuid4())[:8]
    logger.info(
        "Ingestion run=%s | %d symbols | %s → %s | dry_run=%s",
        run_id, total, start, end, dry_run,
    )

    # Process in batches to avoid overloading the API
    for batch_start in range(0, total, batch_size):
        batch = symbols[batch_start: batch_start + batch_size]
        logger.info(
            "Batch %d-%d / %d: %s",
            batch_start + 1,
            min(batch_start + batch_size, total),
            total,
            batch,
        )

        try:
            df = client.get_batch_prices(batch, start=start, end=end)

            if df.empty:
                logger.warning("FMP returned empty DataFrame for batch %s", batch)
                failed.extend(batch)
                continue

            fetched_syms = df["symbol"].unique().tolist() if "symbol" in df.columns else []
            missing = [s for s in batch if s.upper() not in [x.upper() for x in fetched_syms]]
            if missing:
                logger.warning("FMP returned no data for: %s", missing)
                failed.extend(missing)

            if dry_run:
                logger.info(
                    "[DRY RUN] Would write %d rows for %d symbols.",
                    len(df), len(fetched_syms),
                )
                succeeded.extend(fetched_syms)
                continue

            # Write to SQL
            sql_store._ensure_prices_schema()
            prices_tbl = sql_store._tbl("prices")

            # Normalize column names for SQL table
            write_df = df.copy()
            if "datetime" in write_df.columns and "date" not in write_df.columns:
                write_df = write_df.rename(columns={"datetime": "date"})
            write_df["ts_utc"] = datetime.now(tz=timezone.utc).isoformat()
            write_df["env"] = "prod"

            # Delete then insert (upsert pattern)
            syms_sql = ", ".join(f"'{s}'" for s in fetched_syms)
            del_sql = (
                f"DELETE FROM {prices_tbl} "
                f"WHERE symbol IN ({syms_sql}) "
                f"  AND date >= '{start}' "
                f"  AND date <= '{end}'"
            )
            from sqlalchemy import text
            with sql_store.engine.begin() as conn:
                conn.execute(text(del_sql))

            cols = ["symbol", "date", "open", "high", "low", "close", "volume", "ts_utc", "env"]
            write_df = write_df[[c for c in cols if c in write_df.columns]]
            write_df.to_sql(
                prices_tbl,
                con=sql_store.engine,
                if_exists="append",
                index=False,
            )

            # Update freshness tracking
            for sym in fetched_syms:
                sym_df = df[df["symbol"].str.upper() == sym.upper()]
                dt_col = "datetime" if "datetime" in sym_df.columns else "date"
                newest = str(pd.to_datetime(sym_df[dt_col]).max().date()) if not sym_df.empty else None
                sql_store.update_data_freshness(
                    symbol=sym,
                    provider="fmp",
                    newest_date=newest,
                    row_count=len(sym_df),
                )

            logger.info("Wrote %d rows for: %s", len(write_df), fetched_syms)
            succeeded.extend(fetched_syms)

        except Exception as exc:
            logger.error("Batch failed (%s): %s", batch, exc, exc_info=True)
            failed.extend(batch)

    # Log system event
    summary = {
        "run_id": run_id,
        "total": total,
        "succeeded": len(succeeded),
        "failed": len(failed),
        "failed_symbols": failed,
        "start": start,
        "end": end,
        "dry_run": dry_run,
    }

    if not dry_run:
        try:
            sql_store.log_system_event(
                event_type="price_ingestion",
                message=f"FMP ingestion: {len(succeeded)}/{total} succeeded",
                severity="INFO" if not failed else "WARNING",
                component="ingest_prices_fmp",
                detail=summary,
            )
        except Exception:
            pass

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest daily EOD prices from FMP into SQL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols", nargs="+", metavar="SYM",
        help="Symbols to ingest (default: DQ universe or benchmarks)",
    )
    parser.add_argument(
        "--days", type=int, default=5,
        help="Days of history to fetch (default: 5, use 730 for full backfill)",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date YYYY-MM-DD (overrides --days)",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--stale-days", type=int, default=2,
        help="Re-fetch if data is older than N days (default: 2)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if data is fresh",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be fetched without writing to SQL",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Symbols per API batch (default: 20)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="FMP API key (default: FMP_API_KEY env var or config.json)",
    )

    args = parser.parse_args()

    # Resolve date range
    today = datetime.now().strftime("%Y-%m-%d")
    end_date = args.end or today
    if args.start:
        start_date = args.start
    else:
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    # Resolve symbols
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = _load_dq_symbols()

    if not symbols:
        logger.error("No symbols to ingest.")
        return 2

    # Check freshness (skip fresh symbols unless --force)
    if not args.force and not args.dry_run:
        try:
            from core.sql_store import SqlStore
            sql = SqlStore.from_settings({})
            symbols = _check_existing_freshness(sql, symbols, stale_days=args.stale_days)
        except Exception as exc:
            logger.warning("Freshness check failed, fetching all: %s", exc)

    if not symbols:
        logger.info("All symbols are fresh. Nothing to do. Use --force to override.")
        return 0

    logger.info("Ingesting %d symbols: %s → %s", len(symbols), start_date, end_date)

    summary = ingest(
        symbols,
        start=start_date,
        end=end_date,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        fmp_api_key=args.api_key,
    )

    # Report
    print("\n" + "=" * 60)
    print(f"  Ingestion complete  (run_id={summary['run_id']})")
    print(f"  Symbols: {summary['total']}  |  OK: {summary['succeeded']}  |  Failed: {summary['failed']}")
    if summary["failed_symbols"]:
        print(f"  Failed: {summary['failed_symbols']}")
    print("=" * 60)

    if summary["succeeded"] == 0:
        return 2
    if summary["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
