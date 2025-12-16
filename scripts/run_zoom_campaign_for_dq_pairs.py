# -*- coding: utf-8 -*-
"""
scripts/run_zoom_campaign_for_dq_pairs.py
=========================================

Batch ZoomCampaign runner ברמת קרן גידור:

- טוען את ה-Universe מטבלת dq_pairs (DuckDB / SqlStore דרך SQLAlchemy).
- מריץ את ZoomCampaign דרך ה-CLI הקיים של root/optimization_tab
  לכל זוג או לזוג בודד – בדיוק כמו שאתה מריץ ידנית.

איך זה עובד בפועל:
-------------------
1. טעינת רשימת זוגות:
   • ברירת מחדל: duckdb:///cache.duckdb, טבלת dq_pairs.
   • אפשר לסנן לפי WHERE, להגביל כמות, או לבחור זוג אחד ידנית.

2. בניית פקודת CLI:
   • python -m root.optimization_tab zoom-campaign --pair XLY-XLP ...
   • מעביר sampler, n_trials, timeout, zoom_stages, study_prefix.

3. ריצה בזוגות:
   • לכל זוג → subprocess.call(cmd) → שומר את כל הלוגים כמו היום.

שימי 💡:
--------
אם ב-root/optimization_tab ה-CLI שלך שונה קצת (למשל subcommand אחר,
או flags אחרים) – תשנה רק את build_cmd למטה שיתאים בדיוק לפרמטרים שלך.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sqlalchemy import create_engine


# =========================
# Section 1: Data structures
# =========================

@dataclass
class Pair:
    sym_x: str
    sym_y: str

    @property
    def pair_str(self) -> str:
        return f"{self.sym_x}-{self.sym_y}"


# =========================
# Section 2: Universe loader
# =========================

DEFAULT_SQL_URL: str = "duckdb:///C:/Users/omrib/AppData/Local/pairs_trading_system/cache.duckdb"
DEFAULT_DQ_TABLE: str = "dq_pairs"


def load_pairs_from_sql(
    sql_url: str,
    table: str = DEFAULT_DQ_TABLE,
    where: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Pair]:
    """
    טוען sym_x, sym_y מטבלת dq_pairs (או טבלה אחרת) ומחזיר רשימת Pair.

    Parameters
    ----------
    sql_url : str
        SQLAlchemy URL, למשל: "duckdb:///cache.duckdb".
    table : str
        שם הטבלה (ברירת מחדל: dq_pairs).
    where : str, optional
        ביטוי WHERE חופשי, למשל "score >= 0.5 AND is_active = TRUE".
    limit : int, optional
        הגבלת מספר הזוגות.

    Returns
    -------
    List[Pair]
    """
    engine = create_engine(sql_url)

    query = f"SELECT sym_x, sym_y FROM {table}"
    if where:
        query += f" WHERE {where}"
    # אם יש לך עמודת score / rank ניתן לסדר כאן:
    query += " ORDER BY sym_x, sym_y"

    if limit is not None:
        query += f" LIMIT {limit}"

    df = pd.read_sql(query, engine)

    pairs: List[Pair] = [
        Pair(sym_x=row.sym_x, sym_y=row.sym_y)
        for row in df.itertuples(index=False)
    ]
    return pairs


# =========================
# Section 3: Command builder
# =========================

def build_zoom_campaign_cmd(
    pair: Pair,
    *,
    sampler: str,
    n_trials: int,
    timeout_sec: int,
    zoom_stages: int,
    study_prefix: str,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """
    בונה את פקודת ה-CLI ל-ZoomCampaign עבור זוג ספציפי.

    ❗ חשוב:
    אם ה-CLI שלך ב-root/optimization_tab שונה (שם subcommand, שמות flags),
    תעדכן כאן שיתאים 1:1 למה שאתה מריץ ידנית היום.
    """
    cmd: List[str] = [
        sys.executable,
        "-m",
        "root.optimization_tab",   # אם במקום זה יש לך קובץ, אפשר לשנות ל: "root.optimization_tab_cli"
        "zoom-campaign",           # subcommand לדוגמה – תעדכן אם שונה
        "--pair",
        pair.pair_str,
        "--sampler",
        sampler,
        "--n-trials",
        str(n_trials),
        "--timeout-sec",
        str(timeout_sec),
        "--zoom-stages",
        str(zoom_stages),
        "--study-prefix",
        study_prefix,
    ]

    if extra_args:
        cmd.extend(extra_args)

    return cmd


def run_for_pair(pair: Pair, args: argparse.Namespace) -> int:
    """
    מריץ ZoomCampaign עבור זוג אחד ומדפיס header יפה ללוג.
    """
    cmd = build_zoom_campaign_cmd(
        pair,
        sampler=args.sampler,
        n_trials=args.n_trials,
        timeout_sec=args.timeout_sec,
        zoom_stages=args.zoom_stages,
        study_prefix=args.study_prefix,
        extra_args=args.extra_opt_args,
    )

    banner = f"[ZoomCampaign] Running for {pair.pair_str}"
    print("=" * 80)
    print(banner)
    print("-" * 80)
    print("CMD:", " ".join(cmd))
    print("=" * 80, flush=True)

    return subprocess.call(cmd)


# =========================
# Section 4: CLI
# =========================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_zoom_campaign_for_dq_pairs",
        description=(
            "מריץ ZoomCampaign (דרך root/optimization_tab) עבור זוג יחיד "
            "או עבור Universe מתוך dq_pairs."
        ),
    )

    # ---- Universe / selection ----
    parser.add_argument(
        "--sql-url",
        type=str,
        default=DEFAULT_SQL_URL,
        help=(
            f"SQLAlchemy URL ל-SqlStore/DuckDB (ברירת מחדל: {DEFAULT_SQL_URL}). "
            "למשל: duckdb:///cache.duckdb"
        ),
    )
    parser.add_argument(
        "--table",
        type=str,
        default=DEFAULT_DQ_TABLE,
        help=f"שם טבלת הזוגות (ברירת מחדל: {DEFAULT_DQ_TABLE}).",
    )
    parser.add_argument(
        "--where",
        type=str,
        default=None,
        help=(
            "תנאי WHERE ל-SQL, למשל: \"score >= 0.5 AND is_active = TRUE\". "
            "אם לא ניתן – לוקח את כל הטבלה."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="הגבלת מספר הזוגות מה-universe. שימושי לבדיקות (למשל 5).",
    )

    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help=(
            "אם אתה רוצה להריץ על זוג אחד בלבד, ציין בפורמט SYM_X-SYM_Y "
            "(למשל XLY-XLP). אם לא מצוין – נלקח ה-universe מהטבלה."
        ),
    )

    # ---- Optimization / ZoomCampaign params ----
    parser.add_argument(
        "--sampler",
        type=str,
        default="TPE",
        help="שם הסמפלר ל-ZoomCampaign/Optuna (ברירת מחדל: TPE).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="מספר טריילים לכל ZoomCampaign (ברירת מחדל: 50).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=600,
        help="Timeout בשניות לכל זוג (ברירת מחדל: 600 = 10 דקות).",
    )
    parser.add_argument(
        "--zoom-stages",
        type=int,
        default=3,
        help="מספר שלבי zoom באלגוריתם ZoomCampaign (ברירת מחדל: 3).",
    )
    parser.add_argument(
        "--study-prefix",
        type=str,
        default="zoom",
        help="Prefix לשם ה-study ב-Optuna/SqlStore (ברירת מחדל: 'zoom').",
    )

    parser.add_argument(
        "--extra-opt-args",
        nargs="*",
        default=None,
        help=(
            "פרמטרים נוספים שתעביר כמו שהם ל-root/optimization_tab, לדוגמה:\n"
            "--extra-opt-args --risk-profile aggressive --min-sharpe 1.0"
        ),
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # ---- בחירת universe ----
    if args.pair:
        # מצב זוג יחיד ידני, למשל: --pair XLY-XLP
        try:
            sym_x, sym_y = args.pair.split("-", 1)
        except ValueError:
            print("❌ --pair חייב להיות בפורמט SYM_X-SYM_Y, לדוגמה: XLY-XLP")
            return 1

        pairs = [Pair(sym_x=sym_x.strip(), sym_y=sym_y.strip())]
    else:
        # מצב universe מטבלת dq_pairs
        print(
            f"[Universe] Loading pairs from {args.sql_url} table={args.table} "
            f"where={args.where!r} limit={args.limit!r}"
        )
        pairs = load_pairs_from_sql(
            sql_url=args.sql_url,
            table=args.table,
            where=args.where,
            limit=args.limit,
        )
        print(f"[Universe] Loaded {len(pairs)} pairs.")

        if not pairs:
            print("⚠️  לא נמצאו זוגות להרצה. בדוק את ה-SQL / WHERE / טבלה.")
            return 1

    # ---- ריצה לכל זוג ----
    exit_code = 0
    for idx, pair in enumerate(pairs, start=1):
        print(f"\n>>> [{idx}/{len(pairs)}] {pair.pair_str}")
        code = run_for_pair(pair, args)
        if code != 0:
            print(f"⚠️  ZoomCampaign נכשל עבור {pair.pair_str} עם קוד יציאה {code}")
            exit_code = code  # שומר את האחרון הבעייתי

    print("\n✅ Batch ZoomCampaign run finished.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
