# -*- coding: utf-8 -*-
"""
scripts/select_top_pairs_from_ranked_csv.py — Top Pairs Selector (HF-grade v3)
==============================================================================

תפקידים:
--------
1. לקרוא קובץ ranked universe שנוצר ע"י:
       scripts/research_rank_pairs_from_dq.py
   (בד"כ: pairs_universe_ranked.csv).

2. לסנן ולדרג זוגות לפי:
   - pair_score  (כמו שמחושב ב-core.pair_ranking; סקאלה ~ [-2..+2])
   - pair_label  (A+/A/A-/B+/... אם קיים)
   - ספי corr / half_life / spread_vol / n_obs
   - סינון clone-like (SPY/VOO וכו') אם רוצים.

3. לייצר:
   - pairs_universe_selected.csv       — sym_x,sym_y ליוניברס של המערכת.
   - pairs_universe_selected_full.csv  — דו"ח מחקר מלא עם כל המדדים.
   - טבלת dq_pairs ב-DuckDB cache.duckdb (sym_x,sym_y).

הערות חשובות:
--------------
- הסקריפט *לא* מחשב מחדש pair_score — הוא סומך על core.pair_ranking.
- ברירת המחדל:
    • משתמש ב-pair_score כמו שהוא.
    • מסנן לפי min_score >= 0.0  (כל מה שחיובי).
    • אפשר להחליש/להחמיר דרך CLI: --min-score, --min-label וכו'.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import duckdb  # optional
except Exception:  # noqa: BLE001
    duckdb = None


# =========================
# Helpers
# =========================


def _utc_ts() -> str:
    """UTC timestamp for backups / temp names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_backup(path: Path) -> Optional[Path]:
    """
    אם הקובץ קיים → ניצור backup עם חותמת זמן.
    אחרת → נחזיר None.
    """
    if not path.exists():
        return None
    ts = _utc_ts()
    bak = path.with_name(f"{path.stem}.bak-{ts}{path.suffix}")
    bak.parent.mkdir(parents=True, exist_ok=True)
    path.replace(bak)
    return bak


def _label_rank(label: str) -> int:
    """
    ממפה pair_label לטווח מספרי (A+ הכי טוב, D הכי גרוע).
    אם אין label או לא מוכר → 0.
    """
    if not label:
        return 0
    label = str(label).strip().upper()
    order = ["D", "C", "B-", "B", "B+", "A-", "A", "A+"]
    mapping: Dict[str, int] = {lab: i for i, lab in enumerate(order)}
    return mapping.get(label, 0)


def _default_duckdb_path() -> Path:
    """
    ברירת מחדל ל-cache.duckdb אם לא בא מה-CLI.
    """
    import os

    env_path = os.getenv("PAIRS_TRADING_CACHE_DB")
    if env_path:
        return Path(env_path).expanduser().resolve()

    local_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return (local_dir / "pairs_trading_system" / "cache.duckdb").resolve()


# =========================
# Config dataclass
# =========================


@dataclass
class SelectionConfig:
    ranked_csv: Path
    top_n: int = 150

    # סינון לפי score/label
    min_score: float = 0.0          # pair_score >= min_score
    min_label: str = "C"            # מינימום pair_label (C/B-/B/A-...)

    # ספי מבנה
    min_n_obs: int = 200            # n_obs מינימלי (אם קיים)
    max_corr: float = 0.999         # תקרת correlation רכה
    max_half_life: float = 300.0    # Half-life מקסימלי
    min_spread_vol: float = 0.0     # תנודתיות מינימלית בספרד

    # Clone-like filter
    allow_clones: bool = False
    clone_corr: float = 0.995       # אם corr מעל זה + Sharpe עומד בתנאי → clone-like
    clone_max_abs_sharpe: float = 0.10

    # DuckDB integration
    duckdb_path: Optional[Path] = None
    duckdb_table: str = "dq_pairs"

    # Outputs
    universe_csv: Path = Path("pairs_universe_selected.csv")
    report_csv: Path = Path("pairs_universe_selected_full.csv")


# =========================
# Core logic
# =========================


def _load_ranked_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ranked CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Ranked CSV '{path}' is empty.")

    # נמפה עמודות בסיסיות אם צריך
    cols = {str(c).lower().strip(): c for c in df.columns}

    # sym_x / sym_y
    if "sym_x" not in cols or "sym_y" not in cols:
        # ננסה symbol_1 / symbol_2
        s1 = cols.get("symbol_1")
        s2 = cols.get("symbol_2")
        if not s1 or not s2:
            raise RuntimeError(
                f"Ranked CSV '{path}' must contain sym_x/sym_y or symbol_1/symbol_2."
            )
        df = df.rename(columns={s1: "sym_x", s2: "sym_y"})
    else:
        df = df.rename(columns={cols["sym_x"]: "sym_x", cols["sym_y"]: "sym_y"})

    # pair_score
    if "pair_score" not in cols:
        raise RuntimeError(
            f"Ranked CSV '{path}' does not contain 'pair_score'. "
            "Make sure you ran scripts/research_rank_pairs_from_dq.py (v1+)."
        )

    # pair_label (לא חובה אבל מומלץ)
    if "pair_label" not in cols:
        # נוסיף None, רק כדי שנוכל לחיות עם זה
        df["pair_label"] = None

    # ניקוי שמות סימבולים
    df["sym_x"] = df["sym_x"].astype(str).str.strip()
    df["sym_y"] = df["sym_y"].astype(str).str.strip()

    return df


def _enrich_label_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pair_label"] = df["pair_label"].astype(str).replace({"nan": ""})
    df["label_rank"] = df["pair_label"].apply(_label_rank).astype(int)
    return df


def _detect_clone_like(df: pd.DataFrame, cfg: SelectionConfig) -> pd.Series:
    """
    אם אין is_clone_like מ-core.pair_ranking, נוסיף הערכה פשוטה:
    - corr גבוה מאוד
    - ו־Sharpe כמעט 0 → כנראה "אותו נכס".
    """
    if "corr" not in df.columns:
        return pd.Series(False, index=df.index)

    corr = pd.to_numeric(df["corr"], errors="coerce").fillna(0.0)
    sharpe = pd.to_numeric(df.get("spread_sharpe", 0.0), errors="coerce").fillna(0.0)

    clone_mask = (corr >= cfg.clone_corr) & (sharpe.abs() <= cfg.clone_max_abs_sharpe)
    return clone_mask


def _apply_filters(df: pd.DataFrame, cfg: SelectionConfig) -> pd.DataFrame:
    df = df.copy()

    # ניקוי אינפים/נאן
    df = df.replace([np.inf, -np.inf], np.nan)

    # pair_score -> float
    df["pair_score"] = pd.to_numeric(df["pair_score"], errors="coerce")

    # label_rank
    df = _enrich_label_rank(df)

    # n_obs
    if "n_obs" in df.columns:
        df["n_obs"] = pd.to_numeric(df["n_obs"], errors="coerce")

    # corr / half_life / spread_vol
    for col in ("corr", "half_life", "spread_vol"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # is_clone_like: אם אין עמודה, נבנה אחת
    if "is_clone_like" not in df.columns:
        df["is_clone_like"] = _detect_clone_like(df, cfg).astype(bool)
    else:
        df["is_clone_like"] = df["is_clone_like"].fillna(False).astype(bool)

    # 1. Hard filter: score
    df = df[df["pair_score"].notna()]
    df = df[df["pair_score"] >= cfg.min_score]

    # 2. Hard filter: label
    min_label_rank = _label_rank(cfg.min_label)
    df = df[df["label_rank"] >= min_label_rank]

    # 3. Hard filter: n_obs
    if "n_obs" in df.columns:
        df = df[df["n_obs"].fillna(0) >= cfg.min_n_obs]

    # 4. Hard filter: corr / half_life / spread_vol
    if "corr" in df.columns:
        df = df[df["corr"].fillna(0.0) <= cfg.max_corr]
    if "half_life" in df.columns:
        df = df[df["half_life"].fillna(cfg.max_half_life) <= cfg.max_half_life]
    if "spread_vol" in df.columns:
        df = df[df["spread_vol"].fillna(0.0) >= cfg.min_spread_vol]

    # 5. Clone filter
    if not cfg.allow_clones:
        df = df[~df["is_clone_like"]]

    # 6. לא רוצים (sym_x == sym_y)
    df = df[df["sym_x"] != df["sym_y"]]

    return df


def _dedupe_unordered(df: pd.DataFrame) -> pd.DataFrame:
    """
    אם מופיעים גם (A,B) וגם (B,A) - נשאיר רק את זה עם ה-score הגבוה.
    """
    if df.empty:
        return df

    df = df.copy()
    key = df.apply(
        lambda r: "||".join(sorted([str(r["sym_x"]), str(r["sym_y"])])),
        axis=1,
    )
    df["_unordered_key"] = key
    df = df.sort_values("pair_score", ascending=False)
    df = df.drop_duplicates(subset=["_unordered_key"], keep="first")
    df = df.drop(columns=["_unordered_key"])
    return df


def _select_top(df: pd.DataFrame, cfg: SelectionConfig) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.sort_values("pair_score", ascending=False)
    if cfg.top_n > 0:
        df = df.head(cfg.top_n)
    df = df.reset_index(drop=True)
    return df


def _write_universe_csv(df: pd.DataFrame, cfg: SelectionConfig) -> None:
    uni = df[["sym_x", "sym_y"]].drop_duplicates().reset_index(drop=True)

    out_path = cfg.universe_csv.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _safe_backup(out_path)  # אם כבר יש כזה → גיבוי
    uni.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[OK] Universe CSV: {out_path} ({len(uni)} rows)")


def _write_report_csv(df: pd.DataFrame, cfg: SelectionConfig) -> None:
    out_path = cfg.report_csv.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _safe_backup(out_path)
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[OK] Report CSV:   {out_path} ({len(df)} rows)")


def _write_duckdb(df: pd.DataFrame, cfg: SelectionConfig) -> None:
    if cfg.duckdb_path is None and duckdb is None:
        print("[WARN] DuckDB not available and no path given; skipping dq_pairs.")
        return

    if duckdb is None:
        raise RuntimeError(
            "duckdb is not installed (pip install duckdb) or not importable."
        )

    db_path = cfg.duckdb_path or _default_duckdb_path()
    db_path = db_path.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(db_path))
    try:
        # נרשום DF עם sym_x,sym_y בלבד
        uni = df[["sym_x", "sym_y"]].drop_duplicates().reset_index(drop=True)
        conn.register("pairs_df", uni)

        conn.execute(f"DROP TABLE IF EXISTS {cfg.duckdb_table}")
        conn.execute(
            f"""
            CREATE TABLE {cfg.duckdb_table} AS
            SELECT sym_x, sym_y
            FROM pairs_df
            """
        )
        n = conn.execute(
            f"SELECT COUNT(*) FROM {cfg.duckdb_table}"
        ).fetchone()[0]
        print(
            f"[OK] Wrote DuckDB table '{cfg.duckdb_table}' into: {db_path} "
            f"({n} rows)."
        )
    finally:
        conn.close()


# =========================
# CLI
# =========================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select top tradable pairs from ranked CSV and publish universe + dq_pairs."
    )
    parser.add_argument(
        "--ranked-csv",
        type=str,
        default="pairs_universe_ranked.csv",
        help="Ranked pairs CSV from research_rank_pairs_from_dq.py (default: pairs_universe_ranked.csv).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=150,
        help="Maximum number of pairs to select (default: 150).",
    )
    parser.add_argument(
        "--universe-csv",
        type=str,
        default="pairs_universe_selected.csv",
        help="Output CSV for universe (sym_x,sym_y).",
    )
    parser.add_argument(
        "--report-csv",
        type=str,
        default="pairs_universe_selected_full.csv",
        help="Output CSV for detailed report.",
    )
    parser.add_argument(
        "--duckdb-path",
        type=str,
        default="",
        help="DuckDB cache path (default: LOCALAPPDATA\\pairs_trading_system\\cache.duckdb or PAIRS_TRADING_CACHE_DB).",
    )
    parser.add_argument(
        "--duckdb-table",
        type=str,
        default="dq_pairs",
        help="DuckDB table name for pairs universe (default: dq_pairs).",
    )

    # Score / label filters
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum pair_score to keep (new scale ~[-2..+2], default: 0.0).",
    )
    parser.add_argument(
        "--min-label",
        type=str,
        default="C",
        help="Minimum pair_label: D/C/B-/B/B+/A-/A/A+ (default: C).",
    )

    # Structural filters
    parser.add_argument(
        "--min-n-obs",
        type=int,
        default=200,
        help="Minimum n_obs per pair (default: 200).",
    )
    parser.add_argument(
        "--max-corr",
        type=float,
        default=0.999,
        help="Maximum allowed correlation (default: 0.999).",
    )
    parser.add_argument(
        "--max-half-life",
        type=float,
        default=300.0,
        help="Maximum allowed half-life in days (default: 300).",
    )
    parser.add_argument(
        "--min-spread-vol",
        type=float,
        default=0.0,
        help="Minimum required spread_vol (default: 0.0).",
    )

    # Clone control
    parser.add_argument(
        "--allow-clones",
        action="store_true",
        help="Allow clone-like pairs (default: False → filter them out).",
    )

    args = parser.parse_args()

    ranked_csv = Path(args.ranked_csv).expanduser().resolve()
    universe_csv = Path(args.universe_csv).expanduser()
    report_csv = Path(args.report_csv).expanduser()
    duckdb_path = Path(args.duckdb_path).expanduser().resolve() if args.duckdb_path else None

    cfg = SelectionConfig(
        ranked_csv=ranked_csv,
        top_n=args.top,
        min_score=args.min_score,
        min_label=args.min_label,
        min_n_obs=args.min_n_obs,
        max_corr=args.max_corr,
        max_half_life=args.max_half_life,
        min_spread_vol=args.min_spread_vol,
        allow_clones=args.allow_clones,
        duckdb_path=duckdb_path,
        duckdb_table=args.duckdb_table,
        universe_csv=universe_csv,
        report_csv=report_csv,
    )

    print(f"[Info] Ranked CSV: {cfg.ranked_csv}")
    print(f"[Info] top={cfg.top_n}, min_score={cfg.min_score}, min_label={cfg.min_label}, "
          f"min_n_obs={cfg.min_n_obs}, max_corr={cfg.max_corr}, "
          f"max_half_life={cfg.max_half_life}, min_spread_vol={cfg.min_spread_vol}, "
          f"allow_clones={cfg.allow_clones}")
    print(f"[Info] DuckDB path: {cfg.duckdb_path or _default_duckdb_path()}")

    df = _load_ranked_df(cfg.ranked_csv)
    df = _apply_filters(df, cfg)
    df = _dedupe_unordered(df)
    df = _select_top(df, cfg)

    print(f"[OK] Selected {len(df)} pairs after filters and ranking.")

    if df.empty:
        print("[WARN] No pairs passed the filters. "
              "Try lowering --min-score or --min-label, or relaxing other thresholds.")
        return 0

    _write_universe_csv(df, cfg)
    _write_report_csv(df, cfg)

    # כתיבה ל-DuckDB
    _write_duckdb(df, cfg)

    # הדפסת טופ 10
    preview_cols = [
        "sym_x",
        "sym_y",
        "pair_score",
        "pair_label",
        "corr",
        "half_life",
        "spread_sharpe",
        "spread_sortino",
        "spread_vol",
        "spread_max_dd",
        "is_clone_like",
    ]
    existing = [c for c in preview_cols if c in df.columns]
    print("\nTop 10 preview:")
    print(df[existing].head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
