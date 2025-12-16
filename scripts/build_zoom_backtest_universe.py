# -*- coding: utf-8 -*-
"""
scripts/build_zoom_backtest_universe.py — Zoom Backtest Universe Builder (HF-grade v1)
======================================================================================

מטרה:
------
1. לטעון את pairs_universe_ranked.csv (תוצאה של research_rank_pairs_from_dq.py).
2. לבחור:
   - 5 הזוגות עם pair_score הכי גבוה.
   - ועוד 5 זוגות ידניים שהוגדרו מראש:
       XLP-XLY, XLB-XLI, XLY-XLC, QQQ-QQQE, SPY-RSP
3. למזג (אם קיים) את data/zoom_best_params.csv עבור אותם זוגות.
4. לשמור:
   - data/zoom_backtest_universe.csv (לשימוש סקריפטים / דשבורד).
   - טבלה zoom_backtest_universe בתוך cache.duckdb.

חשוב:
------
הקובץ הזה **לא מריץ עדיין Backtest**.
הוא מכין "יוניברס backtest מה-zoom" מסודר וניתן להרחבה.
בשלב הבא נוכל לבנות עליו סקריפט שממש קורא Backtester.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import duckdb  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    duckdb = None  # type: ignore[assignment]


# ======================================================================================
# Paths & logging
# ======================================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RANKED_CSV = PROJECT_ROOT / "pairs_universe_ranked.csv"
DEFAULT_ZOOM_PARAMS_CSV = PROJECT_ROOT / "data" / "zoom_best_params.csv"
DEFAULT_OUTPUT_UNIVERSE_CSV = PROJECT_ROOT / "data" / "zoom_backtest_universe.csv"
DEFAULT_DUCKDB_PATH = Path(
    os.getenv(
        "PAIRS_TRADING_CACHE_DB",
        str(
            Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            / "pairs_trading_system"
            / "cache.duckdb"
        ),
    )
).resolve()

logger = logging.getLogger("ZoomBacktestUniverse")


# ======================================================================================
# Config
# ======================================================================================

@dataclass
class ZoomUniverseConfig:
    ranked_csv: Path = DEFAULT_RANKED_CSV
    zoom_params_csv: Path = DEFAULT_ZOOM_PARAMS_CSV
    output_universe_csv: Path = DEFAULT_OUTPUT_UNIVERSE_CSV
    duckdb_path: Path = DEFAULT_DUCKDB_PATH
    duckdb_table: str = "zoom_backtest_universe"

    top_ranked: int = 5  # כמה מהטופ נכניס
    manual_pairs: Tuple[Tuple[str, str], ...] = (
        ("XLP", "XLY"),
        ("XLB", "XLI"),
        ("XLY", "XLC"),
        ("QQQ", "QQQE"),
        ("SPY", "RSP"),
    )


# ======================================================================================
# Helpers
# ======================================================================================

def _load_ranked_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ranked CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Ranked CSV '{path}' is empty.")

    # נוודא שיש לנו sym_x/sym_y ו-pair_score
    cols = {c.lower(): c for c in df.columns}
    if "sym_x" not in cols or "sym_y" not in cols:
        raise KeyError(
            f"Ranked CSV '{path}' must contain columns sym_x and sym_y "
            f"(found: {list(df.columns)!r})"
        )
    if "pair_score" not in cols:
        raise KeyError(
            f"Ranked CSV '{path}' must contain column pair_score "
            f"(found: {list(df.columns)!r})"
        )

    # ננרמל טיפה
    df[cols["sym_x"]] = df[cols["sym_x"]].astype(str).str.strip()
    df[cols["sym_y"]] = df[cols["sym_y"]].astype(str).str.strip()

    return df


def _select_top_ranked(df_ranked: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df_ranked.copy()

    # מעדיפים רק זוגות is_viable==True אם העמודה קיימת
    if "is_viable" in df.columns:
        df = df[df["is_viable"].astype(bool)]

    df = df.sort_values("pair_score", ascending=False)
    return df.head(n).copy()


def _select_manual_pairs(df_ranked: pd.DataFrame, manual_pairs: Tuple[Tuple[str, str], ...]) -> pd.DataFrame:
    df = df_ranked.copy()

    rows: List[dict] = []
    for sx, sy in manual_pairs:
        mask = (df["sym_x"] == sx) & (df["sym_y"] == sy)
        if mask.any():
            # משתמשים בשורה "הטובה ביותר" אם יש כפילויות
            best_row = df[mask].sort_values("pair_score", ascending=False).iloc[0]
            rows.append(best_row.to_dict())
            logger.info("[ZoomUniverse] Manual pair %s-%s found in ranked CSV.", sx, sy)
        else:
            # ניצור שורה "רזה" רק עם הסמלים, והכל NaN
            logger.warning(
                "[ZoomUniverse] Manual pair %s-%s NOT found in ranked CSV — creating stub row.",
                sx,
                sy,
            )
            rows.append(
                {
                    "sym_x": sx,
                    "sym_y": sy,
                    "pair_score": np.nan,
                    "pair_label": "",
                    "is_viable": False,
                }
            )
    return pd.DataFrame(rows)


def _merge_zoom_params(df: pd.DataFrame, zoom_params_csv: Path) -> pd.DataFrame:
    """
    מחבר zoom_best_params.csv אם קיים:
    - מצופה שהקובץ מכיל sym_x, sym_y או pair.
    - אם אין קובץ/עמודות — מחזיר את df המקורי עם עמודה has_zoom_params=False.
    """
    df_out = df.copy()
    df_out["has_zoom_params"] = False

    if not zoom_params_csv.exists():
        logger.warning(
            "[ZoomUniverse] zoom_best_params.csv not found at %s — skipping params merge.",
            zoom_params_csv,
        )
        return df_out

    zoom = pd.read_csv(zoom_params_csv)
    if zoom.empty:
        logger.warning(
            "[ZoomUniverse] zoom_best_params.csv exists but is empty — skipping params merge."
        )
        return df_out

    cols = {c.lower(): c for c in zoom.columns}

    # ננסה קודם sym_x/sym_y ישירות
    if "sym_x" in cols and "sym_y" in cols:
        zx = cols["sym_x"]
        zy = cols["sym_y"]
        zoom[zx] = zoom[zx].astype(str).str.strip()
        zoom[zy] = zoom[zy].astype(str).str.strip()
        join_cols = ["sym_x", "sym_y"]
        zoom = zoom.rename(columns={zx: "sym_x", zy: "sym_y"})
    elif "pair" in cols:
        # pair בפורמט "AAA-BBB"
        pcol = cols["pair"]
        tmp = zoom[pcol].astype(str).str.strip().str.replace(" ", "")
        split = tmp.str.split("-", n=1, expand=True)
        zoom["sym_x"] = split[0]
        zoom["sym_y"] = split[1]
        join_cols = ["sym_x", "sym_y"]
    else:
        logger.warning(
            "[ZoomUniverse] zoom_best_params.csv does not contain sym_x/sym_y or pair — skipping params merge."
        )
        return df_out

    # נשמור את כל שאר העמודות כ-param_*
    meta_cols = {"sym_x", "sym_y"}
    param_cols = [c for c in zoom.columns if c not in meta_cols]

    # כדי לא לזהם את השמות המקוריים, נוסיף prefix param_
    rename_params = {c: f"param_{c}" for c in param_cols}
    zoom_params = zoom[["sym_x", "sym_y"] + param_cols].rename(columns=rename_params)

    merged = df_out.merge(zoom_params, on=join_cols, how="left", validate="m:1")
    merged["has_zoom_params"] = merged.filter(like="param_").notna().any(axis=1)

    return merged


def _write_universe_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("[ZoomUniverse] Saved zoom backtest universe CSV to: %s", path)


def _write_duckdb(df: pd.DataFrame, db_path: Path, table: str) -> None:
    if duckdb is None:
        logger.warning(
            "[ZoomUniverse] duckdb is not installed; skipping DuckDB write."
        )
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.register("zoom_df", df)
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM zoom_df")
        logger.info(
            "[ZoomUniverse] Wrote DuckDB table '%s' (%d rows) into %s",
            table,
            len(df),
            db_path,
        )
    finally:
        con.close()


# ======================================================================================
# Main
# ======================================================================================

def run_zoom_universe_builder(cfg: ZoomUniverseConfig) -> pd.DataFrame:
    logger.info("===== Zoom Backtest Universe Builder started =====")
    logger.info("Ranked CSV: %s", cfg.ranked_csv)
    logger.info("Zoom params CSV: %s", cfg.zoom_params_csv)
    logger.info("DuckDB: %s | table: %s", cfg.duckdb_path, cfg.duckdb_table)
    logger.info("top_ranked=%d | manual_pairs=%r", cfg.top_ranked, cfg.manual_pairs)

    df_ranked = _load_ranked_universe(cfg.ranked_csv)

    # 1) 5 הזוגות הכי טובים
    top_df = _select_top_ranked(df_ranked, n=cfg.top_ranked)
    logger.info("[ZoomUniverse] Selected %d top-ranked pairs.", len(top_df))

    # 2) 5 זוגות ידניים
    manual_df = _select_manual_pairs(df_ranked, cfg.manual_pairs)
    logger.info("[ZoomUniverse] Prepared %d manual pairs.", len(manual_df))

    # 3) מאחדים ומורידים כפילויות
    combined = pd.concat([top_df, manual_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sym_x", "sym_y"], keep="first")
    combined.reset_index(drop=True, inplace=True)
    logger.info("[ZoomUniverse] Combined universe has %d unique pairs.", len(combined))

    # 4) מיזוג zoom_best_params.csv (אם קיים)
    combined = _merge_zoom_params(combined, cfg.zoom_params_csv)
    n_with_params = int(combined["has_zoom_params"].sum())
    logger.info(
        "[ZoomUniverse] %d / %d pairs have zoom params.",
        n_with_params,
        len(combined),
    )

    # 5) כתיבה ל-CSV ו-DuckDB
    _write_universe_csv(combined, cfg.output_universe_csv)
    _write_duckdb(combined, cfg.duckdb_path, cfg.duckdb_table)

    logger.info("===== Zoom Backtest Universe Builder finished successfully =====")
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a 10-pair zoom backtest universe (top-ranked + manual)."
    )
    parser.add_argument(
        "--ranked-csv",
        type=str,
        default=str(DEFAULT_RANKED_CSV),
        help="Path to pairs_universe_ranked.csv (default: project_root/pairs_universe_ranked.csv).",
    )
    parser.add_argument(
        "--zoom-params-csv",
        type=str,
        default=str(DEFAULT_ZOOM_PARAMS_CSV),
        help="Path to zoom_best_params.csv (default: data/zoom_best_params.csv).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_UNIVERSE_CSV),
        help="Output CSV for zoom backtest universe (default: data/zoom_backtest_universe.csv).",
    )
    parser.add_argument(
        "--duckdb-path",
        type=str,
        default=str(DEFAULT_DUCKDB_PATH),
        help="DuckDB cache path (default: LOCALAPPDATA/pairs_trading_system/cache.duckdb or PAIRS_TRADING_CACHE_DB).",
    )
    parser.add_argument(
        "--duckdb-table",
        type=str,
        default="zoom_backtest_universe",
        help="DuckDB table name for zoom backtest universe (default: zoom_backtest_universe).",
    )
    parser.add_argument(
        "--top-ranked",
        type=int,
        default=5,
        help="How many top-ranked pairs to include (default: 5).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    cfg = ZoomUniverseConfig(
        ranked_csv=Path(args.ranked_csv).resolve(),
        zoom_params_csv=Path(args.zoom_params_csv).resolve(),
        output_universe_csv=Path(args.output_csv).resolve(),
        duckdb_path=Path(args.duckdb_path).resolve(),
        duckdb_table=args.duckdb_table,
        top_ranked=args.top_ranked,
    )

    try:
        run_zoom_universe_builder(cfg)
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Zoom backtest universe build failed: %r", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
