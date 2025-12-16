# -*- coding: utf-8 -*-
"""
scripts/ingest_prices_for_dq_pairs.py — Price Ingestion for dq_pairs (HF-grade v3, IBKR-ready, no Yahoo)
=========================================================================================================

תפקידים:
---------
1. לקרוא את יוניברס הזוגות מטבלת dq_pairs ב-DuckDB (cache.duckdb).
2. לגזור רשימת סימבולים ייחודיים (sym_x/sym_y), עם אפשרות להגבלה/סינון.
3. למשוך היסטוריית מחירים לכל הסימבולים בטווח תאריכים *מספק אמיתי* (IBKR / ספק אחר ברמה דומה).
4. ליצור/לעדכן טבלת prices ב-DB עם חוזה יציב:

       prices(
           symbol VARCHAR,
           ts     TIMESTAMP,
           open   DOUBLE,
           high   DOUBLE,
           low    DOUBLE,
           close  DOUBLE,
           volume DOUBLE
       )

5. לשמש כחוליית Data → DuckDB/SqlStore לכל:
   - מחקר
   - Backtests / Optimization
   - Dashboard / Agents
   - Risk / Monitoring

עקרונות HF-grade:
------------------
- אין תלות ב-Yahoo Finance (yfinance) — deliberately removed.
- Provider ברירת מחדל: "ibkr" (או ספק פרימיום אחר שתממש).
- הספק מחובר בשכבה אחת (_download_prices_ibkr), כדי שלא תצטרך לגעת בכל שאר המערכת.
- סקריפט CLI ידידותי עם:
   * start/end
   * limit על מספר סימבולים
   * force_rebuild
   * dry-run
   * לוג מפורט לפני / אחרי

TODO לחיבור אמיתי:
-------------------
- לממש את _download_prices_ibkr כך ש:
   * משתמש במודול IBKR שלך (למשל root.ibkr_connection / core.ib_data_ingestor).
   * מחזיר DataFrame במבנה הסטנדרטי: symbol, ts, open, high, low, close, volume.
- (אופציונלי) לחבר את DuckDB ל-SqlStore (אם SqlStore כבר משתמש באותו קובץ cache.duckdb,
  אין צורך בשינוי – זה כבר מקור הנתונים שלו).
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import duckdb
import pandas as pd
from ib_insync import IB, Stock, util
# אם יש לך מודול IBKR פנימי, תעדכן כאן:
# from root.ibkr_connection import get_ibkr_client
# from ib_insync import IB, util, Stock


# ========= Logging =========

logger = logging.getLogger("PriceIngest")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
)
logger.addHandler(_handler)

def _connect_ibkr() -> IB:
    """
    מחבר ל-IBKR (TWS / Gateway) בצורה יציבה, עם לוגים ברורים.

    קונפיג:
    -------
    - IBKR_HOST (env)   ברירת מחדל: 127.0.0.1
    - IBKR_PORT (env)   ברירת מחדל: 7497 (TWS paper) / 7496 (Live) – תעדכן בהתאם.
    - IBKR_CLIENT_ID    ברירת מחדל: 1

    אם החיבור נכשל → זורק חריגה.
    """
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PORT", "7497"))
    client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))

    logger.info("[IBKR] Connecting to IBKR at %s:%s (clientId=%s)...", host, port, client_id)
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=10)
    except Exception as exc:  # noqa: BLE001
        logger.error("[IBKR] Failed to connect to IBKR: %r", exc, exc_info=True)
        raise

    if not ib.isConnected():
        raise RuntimeError("IBKR connection failed (ib.isConnected() is False).")

    logger.info("[IBKR] Connected successfully.")
    return ib

# ========= Config dataclass =========


@dataclass
class IngestConfig:
    duckdb_path: Path
    dq_table: str = "dq_pairs"
    prices_table: str = "prices"
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    force_rebuild: bool = False  # אם True → מוחקים ומבנים מחדש את prices
    provider: str = "ibkr"       # ברירת מחדל: IBKR (אין יותר yahoo).
    max_symbols: Optional[int] = None  # הגבלת כמות סימבולים (לצורך בדיקות / SEV)
    dry_run: bool = False                 # אם True → לא מכניס ל-DB, רק מדמה.


# ========= Helpers: Paths & Universe =========


def _resolve_default_duckdb_path() -> Path:
    """
    קובע את מיקום קובץ DuckDB בצורה אחידה לכל המערכת.

    סדר עדיפויות:
    1. משתנה סביבה PAIRS_TRADING_CACHE_DB
    2. LOCALAPPDATA\\pairs_trading_system\\cache.duckdb
    """
    env_path = os.getenv("PAIRS_TRADING_CACHE_DB")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        logger.info("Using PAIRS_TRADING_CACHE_DB=%s", path)
        return path

    local_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    path = (local_dir / "pairs_trading_system" / "cache.duckdb").resolve()
    logger.info("Using default LOCALAPPDATA DB path: %s", path)
    return path


def _load_universe_symbols(
    conn: duckdb.DuckDBPyConnection,
    cfg: IngestConfig,
) -> List[str]:
    """
    קורא את dq_pairs ומחזיר רשימת סימבולים ייחודיים (sym_x, sym_y).

    אפשרויות:
    ----------
    - משתמש בעמודות sym_x, sym_y.
    - מאפשר הגבלה לפי cfg.max_symbols (לפי סדר שיחזור מה-DB).
    """
    logger.info("Loading universe from table '%s'...", cfg.dq_table)
    df = conn.execute(f"SELECT sym_x, sym_y FROM {cfg.dq_table}").fetchdf()

    if df.empty:
        raise RuntimeError(
            f"Table '{cfg.dq_table}' is empty – no pairs to ingest prices for."
        )

    symbols: set[str] = set()
    for _, row in df.iterrows():
        sx = str(row["sym_x"]).strip()
        sy = str(row["sym_y"]).strip()
        if sx:
            symbols.add(sx)
        if sy:
            symbols.add(sy)

    symbols_list = sorted(symbols)
    if cfg.max_symbols is not None:
        symbols_list = symbols_list[: cfg.max_symbols]

    logger.info("Universe has %d unique symbols (max_symbols=%s).", len(symbols_list), cfg.max_symbols)
    return symbols_list


# ========= Provider: IBKR skeleton =========


def _download_prices_ibkr(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    מוריד מחירים מ-IBKR ומחזיר DataFrame סטנדרטי:

        columns: symbol, ts, open, high, low, close, volume
        ts      = datetime64[ns]
        symbol  = str

    שימוש:
    -------
    - מושך ברים יומיים (1 day) מסוג TRADES.
    - מסנן לתאריכים [start_date, end_date].
    - מדלג על סימבולים בעייתיים, משאיר לוג מפורט.

    הערות:
    -------
    - מניח US equities (SMART / USD). אם יש לך Universe אחר, תצטרך להתאים.
    - לפני הרצה יש לוודא ש-TWS / Gateway פתוחים וה-API מאופשר.
    """
    # ניקוי וייחוד רשימת הסימבולים
    symbols_norm = [str(s).strip().upper() for s in symbols if s and str(s).strip()]
    symbols_norm = list(dict.fromkeys(symbols_norm))  # שומר סדר + מסיר כפילויות

    if not symbols_norm:
        logger.warning("[IBKR] No symbols provided to _download_prices_ibkr.")
        return _empty_prices_df()

    # Parse תאריכים לסינון פנימי
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    ib = _connect_ibkr()

    frames: List[pd.DataFrame] = []
    n_total = len(symbols_norm)

    try:
        for i, sym in enumerate(symbols_norm, start=1):
            try:
                logger.info("[IBKR] (%d/%d) %s", i, n_total, sym)

                # חוזה בסיסי למניות US (SMART/ USD). תתאים אם יש לך הגדרות אחרות.
                contract = Stock(sym, "SMART", "USD")

                # בקשת ברים היסטוריים
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_ts.strftime("%Y%m%d %H:%M:%S"),
                    durationStr="10 Y",
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )

                if not bars:
                    logger.warning("[IBKR] %s — no bars returned, skipping.", sym)
                    continue

                df = util.df(bars)

                # מנרמלים שמות עמודות
                if "date" in df.columns:
                    df = df.rename(columns={"date": "ts"})
                elif "barTime" in df.columns:
                    df = df.rename(columns={"barTime": "ts"})

                rename_map = {}
                for cand in ("open", "Open"):
                    if cand in df.columns:
                        rename_map[cand] = "open"
                for cand in ("high", "High"):
                    if cand in df.columns:
                        rename_map[cand] = "high"
                for cand in ("low", "Low"):
                    if cand in df.columns:
                        rename_map[cand] = "low"
                for cand in ("close", "Close"):
                    if cand in df.columns:
                        rename_map[cand] = "close"
                for cand in ("volume", "Volume"):
                    if cand in df.columns:
                        rename_map[cand] = "volume"

                if rename_map:
                    df = df.rename(columns=rename_map)

                required_cols = {"ts", "close"}
                if not required_cols.issubset(df.columns):
                    logger.warning(
                        "[IBKR] %s — missing required cols %s (has: %s). Skipping symbol.",
                        sym,
                        sorted(list(required_cols - set(df.columns))),
                        list(df.columns),
                    )
                    continue

                # סינון טווח תאריכים
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
                df = df.dropna(subset=["ts"])
                df = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)]

                if df.empty:
                    logger.warning("[IBKR] %s — no data in requested date range, skipping.", sym)
                    continue

                # בחירת עמודות
                keep_cols = ["ts", "close"]
                for c in ("open", "high", "low", "volume"):
                    if c in df.columns:
                        keep_cols.append(c)

                df = df[keep_cols].copy()
                df["symbol"] = sym

                frames.append(df)

            except Exception as exc:  # noqa: BLE001
                logger.exception("[IBKR] %s — failed to download: %r", sym, exc)
                continue
    finally:
        try:
            ib.disconnect()
            logger.info("[IBKR] Disconnected.")
        except Exception:
            pass

    if not frames:
        logger.error(
            "[IBKR] No valid price data downloaded for any symbol. Returning empty DataFrame."
        )
        return _empty_prices_df()

    full = pd.concat(frames, ignore_index=True)

    # טיפוסים וניקוי כפילויות
    full["ts"] = pd.to_datetime(full["ts"], errors="coerce")
    full = full.dropna(subset=["ts"])
    full["symbol"] = full["symbol"].astype(str)

    full = (
        full.drop_duplicates(subset=["symbol", "ts"])
        .sort_values(["symbol", "ts"])
        .reset_index(drop=True)
    )

    logger.info(
        "[IBKR] Finished download: %d rows for %d symbols.",
        len(full),
        full["symbol"].nunique(),
    )

    # אם משום מה אין 'close' בשלב הזה (לא צפוי)
    if "close" not in full.columns:
        logger.error(
            "[IBKR] Final dataframe has no 'close' column. Columns: %s",
            list(full.columns),
        )
        return _empty_prices_df()

    return full


def _empty_prices_df() -> pd.DataFrame:
    """
    מחזיר DataFrame ריק במבנה הצפוי.
    """
    return pd.DataFrame(
        columns=["symbol", "ts", "open", "high", "low", "close", "volume"]
    )


# ========= DuckDB: prices table management =========


def _ensure_prices_table(
    conn: duckdb.DuckDBPyConnection,
    cfg: IngestConfig,
    force_rebuild: bool,
) -> None:
    """
    דואג שטבלת prices קיימת ובמצב תקין.
    אם force_rebuild=True → מוחק ובונה מחדש.
    """
    if force_rebuild:
        logger.info("Force rebuild: dropping table '%s' if exists.", cfg.prices_table)
        conn.execute(f"DROP TABLE IF EXISTS {cfg.prices_table}")

    # יוצרים אם לא קיימת
    logger.info("Ensuring prices table '%s' exists...", cfg.prices_table)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {cfg.prices_table} (
            symbol VARCHAR,
            ts     TIMESTAMP,
            open   DOUBLE,
            high   DOUBLE,
            low    DOUBLE,
            close  DOUBLE,
            volume DOUBLE
        )
        """
    )


def _load_prices_for_existing(
    conn: duckdb.DuckDBPyConnection,
    cfg: IngestConfig,
) -> int:
    """
    מחזיר כמה שורות כבר יש בטבלת prices (לצורכי לוג בלבד).
    """
    try:
        n = conn.execute(f"SELECT COUNT(*) FROM {cfg.prices_table}").fetchone()[0]
        return int(n)
    except Exception:
        return 0


def _insert_prices(
    conn: duckdb.DuckDBPyConnection,
    cfg: IngestConfig,
    prices_df: pd.DataFrame,
) -> None:
    """
    מכניס את מחירי הסגירה לטבלה.

    כרגע:
    -----
    - מוחק כפילויות ברמת symbol, ts (בפנדס).
    - מוסיף באמצעות DuckDB INSERT עם רשימת עמודות מפורשת, כדי לתמוך
      בטבלאות prices רחבות (יותר מ-7 עמודות).
    - אם תרצה Upsert אמיתי, אפשר ליישם בהמשך.
    """
    if prices_df.empty:
        logger.warning("prices_df is empty, nothing to insert.")
        return

    # ניקוי כפילויות לפני הכנסת הנתונים
    prices_df = (
        prices_df.drop_duplicates(subset=["symbol", "ts"])
        .sort_values(["symbol", "ts"])
        .reset_index(drop=True)
    )

    logger.info(
        "Inserting %d rows into '%s' for %d symbols...",
        len(prices_df),
        cfg.prices_table,
        prices_df["symbol"].nunique(),
    )

    # רישום טבלת ביניים ב-duckdb
    conn.register("prices_tmp", prices_df)

    # שים לב: רשימת שמות העמודות בצד שמאל !
    conn.execute(
        f"""
        INSERT INTO {cfg.prices_table} (symbol, ts, open, high, low, close, volume)
        SELECT symbol, ts, open, high, low, close, volume
        FROM prices_tmp
        """
    )
    conn.unregister("prices_tmp")


# ========= Main ingest routine =========


def run_ingest(cfg: IngestConfig) -> None:
    """
    רוטינת ingest מלאה ברמת קרן גידור:

    1. פתיחת DuckDB (cache.duckdb).
    2. טעינת dq_pairs → רשימת סימבולים ייחודיים.
    3. יצירת/וידוא טבלת prices.
    4. משיכת מחירים מ-Provider (IBKR).
    5. הכנסת הנתונים לטבלה (אלא אם dry_run=True).
    """
    logger.info("===== Price Ingestion for dq_pairs started =====")
    logger.info("DuckDB path: %s", cfg.duckdb_path)
    logger.info("dq_table: %s | prices_table: %s", cfg.dq_table, cfg.prices_table)
    logger.info("Date range: %s → %s", cfg.start_date, cfg.end_date)
    logger.info(
        "Provider: %s | force_rebuild=%s | max_symbols=%s | dry_run=%s",
        cfg.provider,
        cfg.force_rebuild,
        cfg.max_symbols,
        cfg.dry_run,
    )

    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(cfg.duckdb_path))

    try:
        symbols = _load_universe_symbols(conn, cfg)
        _ensure_prices_table(conn, cfg, force_rebuild=cfg.force_rebuild)

        existing_rows = _load_prices_for_existing(conn, cfg)
        logger.info("Current prices table row count: %d", existing_rows)

        provider = cfg.provider.lower()

        if provider == "ibkr":
            prices_df = _download_prices_ibkr(symbols, cfg.start_date, cfg.end_date)
        else:
            # משדרים באופן מפורש ש-Yahoo לא נתמך יותר
            raise NotImplementedError(
                f"Provider '{cfg.provider}' is not implemented. "
                f"Yahoo has been intentionally disabled; please use 'ibkr' and wire _download_prices_ibkr."
            )

        if cfg.dry_run:
            logger.info(
                "[DRY-RUN] Not inserting into '%s'. Downloaded %d rows for %d symbols.",
                cfg.prices_table,
                len(prices_df),
                prices_df["symbol"].nunique() if not prices_df.empty else 0,
            )
        else:
            _insert_prices(conn, cfg, prices_df)
            final_rows = _load_prices_for_existing(conn, cfg)
            logger.info(
                "Final prices table row count: %d (delta=%d).",
                final_rows,
                final_rows - existing_rows,
            )

        logger.info("===== Price Ingestion finished (provider=%s) =====", provider)
    finally:
        conn.close()
        logger.info("Closed DuckDB connection.")


# ========= CLI =========


def parse_args() -> IngestConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest historical prices for dq_pairs universe into DuckDB (prices table), "
            "HF-grade, non-Yahoo (IBKR-ready)."
        )
    )
    parser.add_argument(
        "--duckdb-path",
        type=str,
        default=None,
        help=(
            "Optional override for DuckDB path. "
            "Default: PAIRS_TRADING_CACHE_DB or LOCALAPPDATA\\pairs_trading_system\\cache.duckdb."
        ),
    )
    parser.add_argument(
        "--dq-table",
        type=str,
        default="dq_pairs",
        help="Name of the dq_pairs universe table.",
    )
    parser.add_argument(
        "--prices-table",
        type=str,
        default="prices",
        help="Name of the prices table to create/fill.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Start date for price history (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="End date for price history (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Drop and recreate prices table before inserting.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ibkr",
        help="Data provider to use (default: 'ibkr'; 'yahoo' is deliberately not supported).",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="Optional limit on number of unique symbols from dq_pairs (for testing or controlled runs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do everything except the final INSERT into the prices table.",
    )

    args = parser.parse_args()

    duckdb_path = (
        Path(args.duckdb_path).expanduser().resolve()
        if args.duckdb_path
        else _resolve_default_duckdb_path()
    )

    return IngestConfig(
        duckdb_path=duckdb_path,
        dq_table=args.dq_table,
        prices_table=args.prices_table,
        start_date=args.start_date,
        end_date=args.end_date,
        force_rebuild=bool(args.force_rebuild),
        provider=args.provider.lower(),
        max_symbols=args.max_symbols,
        dry_run=bool(args.dry_run),
    )


def main() -> int:
    cfg = parse_args()
    try:
        run_ingest(cfg)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Price ingestion failed: %r", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
