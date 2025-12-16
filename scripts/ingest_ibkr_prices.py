# scripts/ingest_ibkr_prices.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
from datetime import date, datetime
from typing import List, Tuple, Optional, Any, Dict

# ----------------------------------------------------------------------
# Project root & sys.path
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ----------------------------------------------------------------------
# Imports from project
# ----------------------------------------------------------------------
from common.config_manager import load_settings
from core.sql_store import SqlStore
from core.ib_data_ingestor import IBDataIngestor

logger = logging.getLogger("ingest_ibkr_prices")


# ----------------------------------------------------------------------
# CLI parsing & helpers
# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HF-grade IBKR → SqlStore ingestion for pairs/symbols"
    )

    parser.add_argument(
        "--pair",
        type=str,
        help="Pair in format 'XLP-XLY' (sym_x-sym_y), e.g. --pair XLP-XLY",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Explicit list of symbols, e.g. --symbols XLP XLY SPY",
    )
    parser.add_argument(
        "--from-universe",
        type=str,
        default=None,
        help="Optional universe table name in SqlStore (e.g. 'dq_pairs')",
    )

    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date (YYYY-MM-DD, default: 2015-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: today)",
    )

    parser.add_argument(
        "--bar-size",
        type=str,
        default="1 day",
        help="IB bar size (e.g. '1 day', '1 hour')",
    )
    parser.add_argument(
        "--what",
        type=str,
        default="TRADES",
        help="whatToShow for IB (TRADES / ADJUSTED_LAST / MIDPOINT / ...)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, allow overwriting existing data ranges (implementation-specific).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="If set, pull only missing dates per symbol from SqlStore.prices",
    )

    return parser.parse_args()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | ingest | %(message)s",
    )


def _parse_date(value: Optional[str], default: Optional[date] = None) -> date:
    if not value:
        if default is None:
            raise ValueError("No date value and no default provided")
        return default
    return datetime.strptime(value, "%Y-%m-%d").date()


# ----------------------------------------------------------------------
# Universe / symbols resolution
# ----------------------------------------------------------------------
def _collect_symbols(
    store: SqlStore,
    pair: Optional[str],
    symbols: Optional[List[str]],
    universe_table: Optional[str],
) -> List[str]:
    """
    מחזיר רשימת סימבולים סופית להזרקת דאטה.
    סדר עדיפות:
    1. --symbols
    2. --pair
    3. --from-universe
    """
    # 1) סימבולים מפורשים מה-CLI
    if symbols:
        return sorted({s.strip().upper() for s in symbols if s.strip()})

    # 2) Pair יחיד כמו "BITO-BKCH"
    if pair:
        parts = pair.replace(" ", "").split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: {pair!r} (expected 'XLP-XLY')")
        return sorted({p.strip().upper() for p in parts if p.strip()})

    # 3) Universe מטבלה ב-SqlStore (למשל dq_pairs)
    if universe_table:
        logger.info("Loading symbols from universe table %s ...", universe_table)
        df = store.read_table(universe_table)
        cols = {c.lower(): c for c in df.columns}
        candidates: List[str] = []

        if "sym_x" in cols and "sym_y" in cols:
            candidates.extend(df[cols["sym_x"]].astype(str).tolist())
            candidates.extend(df[cols["sym_y"]].astype(str).tolist())
        elif "symbol" in cols:
            candidates.extend(df[cols["symbol"]].astype(str).tolist())
        else:
            raise RuntimeError(
                f"Universe table {universe_table!r} does not contain sym_x/sym_y/symbol columns"
            )

        uniq = sorted({s.strip().upper() for s in candidates if s})
        logger.info("Universe %s → %d unique symbols", universe_table, len(uniq))
        return uniq

    raise ValueError("You must provide either --symbols, --pair, or --from-universe")


# ----------------------------------------------------------------------
# Incremental range helper
# ----------------------------------------------------------------------
def _resolve_incremental_range(
    store: SqlStore,
    symbol: str,
    start_date: date,
    end_date: date,
) -> Tuple[date, date]:
    """
    אם incremental=True:
        • מנסה לקרוא היסטוריית מחירים מ-SqlStore.load_price_history.
        • אם יש דאטה – מתחיל יום אחרי התאריך האחרון.
        • אם אין דאטה / שגיאה – משתמש ב-start_date המקורי.
    """
    import pandas as pd  # local import כדי לא לזהם את הטופ של הקובץ

    # אם ל-Store אין load_price_history (גרסה ישנה) – נ fallback לטווח מלא
    if not hasattr(store, "load_price_history"):
        return start_date, end_date

    try:
        df = store.load_price_history(symbol)
    except Exception as exc:
        logger.warning(
            "load_price_history(%s) failed (%s) – using full range %s→%s",
            symbol, exc, start_date, end_date,
        )
        return start_date, end_date

    if df is None or df.empty:
        # אין שום דאטה לסימבול → טוענים מההתחלה
        logger.info("No existing price history for %s – using full range %s→%s",
                    symbol, start_date, end_date)
        return start_date, end_date

    # ננסה לחלץ תאריכים בצורה אחידה:
    try:
        if "date" in getattr(df, "columns", []):
            raw_dates = df["date"]
        else:
            raw_dates = df.index

        # הופכים תמיד ל-DatetimeIndex ואז ל-date אמיתי
        dates = pd.to_datetime(raw_dates, errors="coerce").dropna()
        if dates.empty:
            raise ValueError("could not parse any valid dates")

        last_dt = dates.max().date()  # כאן last_dt הוא כבר datetime.date

    except Exception as exc:
        logger.warning(
            "Could not infer last date for %s from existing history (%s) – using full range",
            symbol, exc,
        )
        return start_date, end_date

    logger.info("Existing history for %s up to %s", symbol, last_dt)

    # אם כבר יש דאטה עד אחרי end_date – אין מה למשוך
    if last_dt >= end_date:
        logger.info("Symbol %s already up-to-date until %s (>= %s)",
                    symbol, last_dt, end_date)
        return end_date, end_date

    # מתחילים יום אחרי התאריך האחרון, אבל לא לפני start_date המקורי
    new_start = max(last_dt, start_date)
    new_start = date.fromordinal(new_start.toordinal() + 1)

    logger.info("Incremental range for %s: %s→%s", symbol, new_start, end_date)
    return new_start, end_date


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    _setup_logging()
    args = _parse_args()

    # Load settings from config.json (אם load_settings תומך בזה)
    try:
        settings = load_settings(CONFIG_PATH)
    except TypeError:
        # גרסאות ישנות יותר של load_settings אולי לא מקבלות path
        settings = load_settings()

    # SqlStore לכתיבה (ingestion → read_only=False)
    store = SqlStore.from_settings(settings, read_only=False)

    # IBKR ingestor מרכזי
    ingestor = IBDataIngestor.from_settings(settings=settings, store=store)

    # דגל ib_enable בהגדרות
    if getattr(settings, "ib_enable", True) is False or getattr(
        getattr(settings, "ib", None), "enable", True
    ) is False:
        logger.error("ib_enable is False in settings – aborting ingestion.")
        return

    # טווח תאריכים
    start_date = _parse_date(args.start, default=date(2015, 1, 1))
    end_date = _parse_date(args.end, default=date.today())

    # Universe של סימבולים
    symbols = _collect_symbols(
        store=store,
        pair=args.pair,
        symbols=args.symbols,
        universe_table=args.from_universe,
    )

    logger.info("==== IBKR Ingestion Started ====")
    logger.info("Symbols: %s", ", ".join(symbols))
    logger.info("Date range: %s → %s", start_date, end_date)
    logger.info("Bar size: %s | whatToShow: %s", args.bar_size, args.what)
    logger.info("force=%s | incremental=%s", args.force, args.incremental)

    total_inserted = 0
    total_symbols = 0

    for sym in symbols:
        sym = sym.strip().upper()
        if not sym:
            continue

        s_start, s_end = start_date, end_date
        if args.incremental:
            s_start, s_end = _resolve_incremental_range(store, sym, start_date, end_date)
            if s_start >= s_end:
                logger.info("Skipping %s – no new dates to ingest.", sym)
                continue

        logger.info(
            "Ingesting %s from %s to %s ...",
            sym,
            s_start,
            s_end,
        )
        try:
            # ingestor.ingest_symbol אמורה:
            # - למשוך דאטה מ-IBKR
            # - לשמור ל-SqlStore.prices (דרך SqlStore / דרך ingestor עצמו)
            # - להחזיר כמה שורות נשמרו
            n_rows = ingestor.ingest_symbol(
                symbol=sym,
                start_date=s_start,
                end_date=s_end,
                bar_size=args.bar_size,
                what_to_show=args.what,
                force=args.force,
            )
            n_rows_int = int(n_rows or 0)
            total_inserted += n_rows_int
            total_symbols += 1
            logger.info("Symbol %s → %s new rows", sym, n_rows_int)

        except Exception as exc:
            logger.exception("Failed to ingest %s: %s", sym, exc)

    logger.info(
        "==== IBKR Ingestion Finished | symbols=%s | total new rows: %s ====",
        total_symbols,
        total_inserted,
    )


if __name__ == "__main__":
    main()
