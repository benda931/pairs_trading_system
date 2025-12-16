# -*- coding: utf-8 -*-
"""
root/ingest_universe_from_ib.py — HF-grade Ingestion from IBKR into SqlStore.prices
===================================================================================

תפקיד:
-------
1. לקרוא Universe של זוגות מתוך SqlStore.dq_pairs (מה שיצרת עם generate_pairs_universe).
2. להוציא רשימת סימבולים ייחודיים (sym_x + sym_y).
3. להתחבר ל-IBKR (ib_insync / TWS / Gateway).
4. למשוך היסטוריית מחירים יומית לכל סימבול.
5. לשמור לטבלת prices דרך SqlStore.save_price_history, בצורה idempotent.

עקרונות:
---------
- Data מקורית: IBKR בלבד (לא Yahoo).
- Truth: SqlStore (DuckDB ב-logs/pairs_trading_<env>.duckdb).
- Idempotent: אם --full-refresh → מוחק את הנתונים הקיימים לסימבול לפני כתיבה.
- מדווח: לוגים + קובץ CSV של דוח אינג’סטיון ב-logs/ingestion_reports/.

שימוש:
------
    cd C:\Users\omrib\OneDrive\Desktop\pairs_trading_system

    # אינג'סטיון מלא לכל ה-universe מ-dq_pairs:
    python -m root.ingest_universe_from_ib --env dev --full-refresh

    # רק 20 סימבולים ראשונים (לבדיקה):
    python -m root.ingest_universe_from_ib --env dev --max-symbols 20

    # רק סימבולים ספציפיים:
    python -m root.ingest_universe_from_ib --env dev --only SPY,QQQ,XLY,XLC

    # הרצה בלי כתיבה (בדיקה בלבד):
    python -m root.ingest_universe_from_ib --env dev --dry-run
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Set

import logging

import pandas as pd
from ib_insync import IB, Stock, util  # ודא ש-ib_insync מותקן

from core.sql_store import SqlStore

# =====================
# Logging בסיסי לסקריפט
# =====================

LOGGER = logging.getLogger("IngestionIBKR")

if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# =====================
# קונפיגורציית אינג'סטיון
# =====================

@dataclass
class IngestionConfig:
    env: str = "dev"
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497        # 7497 = paper, 7496 = live
    ib_client_id: int = 42
    duration_str: str = "5 Y"  # כמה היסטוריה למשוך
    bar_size: str = "1 day"
    what_to_show: str = "TRADES"
    max_symbols: Optional[int] = None
    only_symbols: Optional[List[str]] = None
    full_refresh: bool = False  # אם True → מוחק prices עבור כל סימבול לפני כתיבה
    dry_run: bool = False       # אם True → לא כותב ל-SqlStore
    section: str = "prices_ingestion"  # metadata.section


# =====================
# עזר: SqlStore + Universe
# =====================

def get_sql_store(env: str) -> SqlStore:
    """
    בונה SqlStore דרך from_settings, כמו בשאר המערכת.
    """
    settings_dummy = SimpleNamespace(env=env)
    store = SqlStore.from_settings(settings_dummy)
    return store


def get_universe_symbols(store: SqlStore, cfg: IngestionConfig) -> List[str]:
    """
    מוציא רשימת סימבולים ייחודיים מתוך dq_pairs (sym_x/sym_y).
    אם הועברו only_symbols בקונפיג → יסנן בהתאם.
    """
    dq = store.load_pair_quality(env=cfg.env, latest_only=True)
    if dq.empty:
        LOGGER.warning(
            "dq_pairs ריק — כנראה שצריך להריץ generate_pairs_universe קודם. "
            "לא נבצע אינג'סטיון."
        )
        return []

    cols = [c for c in ("sym_x", "sym_y") if c in dq.columns]
    if not cols:
        LOGGER.warning("dq_pairs נטול עמודות sym_x/sym_y — לא ניתן להוציא universe.")
        return []

    syms: Set[str] = set()
    for c in cols:
        syms.update(dq[c].dropna().astype(str).tolist())

    symbols = sorted(syms)

    # סינון לפי only_symbols אם ביקשו
    if cfg.only_symbols:
        wanted = {s.strip().upper() for s in cfg.only_symbols}
        symbols = [s for s in symbols if s.upper() in wanted]

    # max_symbols להגבלת גודל (לבדיקות)
    if cfg.max_symbols is not None and cfg.max_symbols > 0:
        symbols = symbols[: cfg.max_symbols]

    LOGGER.info("Universe resolved: %d symbols from dq_pairs.", len(symbols))
    return symbols


# =====================
# עזר: IBKR
# =====================

def connect_ib(cfg: IngestionConfig) -> IB:
    ib = IB()
    LOGGER.info(
        "Connecting to IBKR at %s:%s (clientId=%s)...",
        cfg.ib_host,
        cfg.ib_port,
        cfg.ib_client_id,
    )
    ib.connect(cfg.ib_host, cfg.ib_port, clientId=cfg.ib_client_id)
    LOGGER.info("Connected to IBKR.")
    return ib


def fetch_ib_history(
    ib: IB,
    symbol: str,
    cfg: IngestionConfig,
) -> pd.DataFrame:
    """
    מושך היסטוריית מחירים יומית מ-IBKR לסימבול בודד.
    מחזיר DataFrame עם index=date ועמודות OHLCV.
    """
    contract = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=cfg.duration_str,
        barSizeSetting=cfg.bar_size,
        whatToShow=cfg.what_to_show,
        useRTH=True,
        formatDate=1,
        keepUpToDate=False,
    )

    if not bars:
        raise ValueError(f"No historical bars returned from IB for {symbol}")

    df = util.df(bars)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

    # נשמור רק את מה שמעניין אותנו: open/high/low/close/volume
    wanted_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in wanted_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns from IB for {symbol}: {missing}")

    return df[wanted_cols]


# =====================
# עזר: SqlStore prices (truncate + save + coverage)
# =====================

def truncate_symbol_prices(store: SqlStore, symbol: str, env: str) -> None:
    """
    מוחק נתוני prices קיימים עבור symbol+env כדי שהאינג'סטיון יהיה idempotent.
    """
    tbl = store._tbl("prices")  # שימוש ב-prefix של SqlStore
    sql = f"DELETE FROM {tbl} WHERE symbol = :sym AND (env = :env OR env IS NULL)"
    try:
        with store.engine.begin() as conn:
            conn.execute(
                # type: ignore[arg-type]
                util.text(sql),  # משתמשים ב-sqlalchemy.text דרך ib_insync.util.text? עדיף לא.
            )
    except Exception as exc:
        # fallback ללא util.text
        from sqlalchemy import text as sqla_text
        with store.engine.begin() as conn:
            conn.execute(sqla_text(sql), {"sym": symbol, "env": env})
        LOGGER.warning("truncate_symbol_prices(%s) — util.text failed, used sqlalchemy.text: %s", symbol, exc)


# =====================
# אינג'סטיון לסימבול אחד
# =====================

def ingest_symbol(
    ib: IB,
    store: SqlStore,
    symbol: str,
    cfg: IngestionConfig,
) -> Dict[str, Any]:
    """
    מריץ אינג'סטיון לסימבול אחד ומחזיר dict קטן של תוצאות (ל-report).
    """
    started_at = datetime.utcnow()
    status = "ok"
    rows = 0
    error: Optional[str] = None

    try:
        df_sym = fetch_ib_history(ib, symbol, cfg)
        rows = len(df_sym)

        LOGGER.info(
            "[%s] Fetched %d rows from IBKR (duration=%s, bar_size=%s)",
            symbol,
            rows,
            cfg.duration_str,
            cfg.bar_size,
        )

        if cfg.dry_run:
            LOGGER.info("[%s] dry-run=True → לא כותב ל-SqlStore.", symbol)
        else:
            if cfg.full_refresh:
                # מוחק את כל השורות הקיימות לסימבול+env לפני כתיבה
                try:
                    tbl = store._tbl("prices")
                    from sqlalchemy import text as sqla_text
                    with store.engine.begin() as conn:
                        conn.execute(
                            sqla_text(
                                f"DELETE FROM {tbl} WHERE symbol = :sym AND (env = :env OR env IS NULL)"
                            ),
                            {"sym": symbol, "env": cfg.env},
                        )
                    LOGGER.info("[%s] Existing prices rows deleted for env=%s.", symbol, cfg.env)
                except Exception as exc:
                    LOGGER.warning("[%s] Failed to delete existing price rows: %s", symbol, exc)

            store.save_price_history(symbol, df_sym, env=cfg.env)
            LOGGER.info("[%s] Saved %d price rows into SqlStore.prices", symbol, rows)

    except Exception as exc:
        status = "error"
        error = str(exc)
        LOGGER.warning("[%s] ingestion failed: %s", symbol, exc)

    finished_at = datetime.utcnow()
    return {
        "symbol": symbol,
        "status": status,
        "rows": rows,
        "error": error,
        "started_at_utc": started_at.isoformat() + "Z",
        "finished_at_utc": finished_at.isoformat() + "Z",
        "env": cfg.env,
    }


# =====================
# CLI parsing
# =====================

def parse_args() -> IngestionConfig:
    parser = argparse.ArgumentParser(
        description="HF-grade IBKR → SqlStore.prices ingestion script"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="dev",
        help="Environment name (dev/paper/live). Default: dev",
    )
    parser.add_argument(
        "--ib-host",
        type=str,
        default="127.0.0.1",
        help="IBKR host (TWS/Gateway). Default: 127.0.0.1",
    )
    parser.add_argument(
        "--ib-port",
        type=int,
        default=7497,
        help="IBKR port (7497=paper, 7496=live). Default: 7497",
    )
    parser.add_argument(
        "--ib-client-id",
        type=int,
        default=42,
        help="IBKR clientId for this script. Default: 42",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default="5 Y",
        help="IB durationStr, e.g. '1 Y', '5 Y', '365 D'. Default: '5 Y'",
    )
    parser.add_argument(
        "--bar-size",
        type=str,
        default="1 day",
        help="IB barSizeSetting, e.g. '1 day', '1 hour'. Default: '1 day'",
    )
    parser.add_argument(
        "--what-to-show",
        type=str,
        default="TRADES",
        help="IB whatToShow, e.g. 'TRADES', 'MIDPOINT'. Default: 'TRADES'",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Limit number of symbols from dq_pairs (for testing).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of symbols to ingest (overrides dq_pairs).",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Delete existing price rows for each symbol+env before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run ingestion but do NOT write to SqlStore.",
    )

    args = parser.parse_args()

    only_symbols: Optional[List[str]] = None
    if args.only:
        only_symbols = [s.strip() for s in args.only.split(",") if s.strip()]

    cfg = IngestionConfig(
        env=args.env,
        ib_host=args.ib_host,
        ib_port=args.ib_port,
        ib_client_id=args.ib_client_id,
        duration_str=args.duration,
        bar_size=args.bar_size,
        what_to_show=args.what_to_show,
        max_symbols=args.max_symbols,
        only_symbols=only_symbols,
        full_refresh=bool(args.full_refresh),
        dry_run=bool(args.dry_run),
    )
    return cfg


# =====================
# MAIN
# =====================

def main() -> None:
    cfg = parse_args()
    LOGGER.info("Starting IBKR ingestion (env=%s, full_refresh=%s, dry_run=%s)",
                cfg.env, cfg.full_refresh, cfg.dry_run)

    # 1. SqlStore
    store = get_sql_store(cfg.env)

    # 2. Universe
    symbols = get_universe_symbols(store, cfg)
    if not symbols:
        LOGGER.warning("No symbols to ingest. Exiting.")
        return

    # 3. IBKR
    ib = connect_ib(cfg)

    # 4. לולאת סימבולים + דוח
    results: List[Dict[str, Any]] = []
    try:
        for sym in symbols:
            LOGGER.info("==== Ingesting %s ====", sym)
            res = ingest_symbol(ib, store, sym, cfg)
            results.append(res)
    finally:
        ib.disconnect()
        LOGGER.info("Disconnected from IBKR.")

    # 5. דוח ingestion ל-CSV
    if results:
        df_report = pd.DataFrame(results)
        logs_dir = Path("logs") / "ingestion_reports"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = logs_dir / f"ib_ingestion_{cfg.env}_{ts}.csv"
        df_report.to_csv(report_path, index=False, encoding="utf-8")
        ok = sum(1 for r in results if r["status"] == "ok")
        err = sum(1 for r in results if r["status"] != "ok")
        LOGGER.info(
            "Ingestion completed: success=%d, errors=%d. Report: %s",
            ok,
            err,
            report_path,
        )
    else:
        LOGGER.info("Ingestion finished with no results (no symbols processed).")


if __name__ == "__main__":
    main()
