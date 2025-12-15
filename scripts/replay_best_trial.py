# -*- coding: utf-8 -*-
"""
scripts/replay_best_trial.py — הרצת Backtest מקצועית ל-best Optuna trial עבור זוג ספציפי
=========================================================================================

שימוש בסיסי:
-------------
    python -m scripts.replay_best_trial --pair BITO-BKCH

עם חלון תאריכים:
-----------------
    python -m scripts.replay_best_trial \
        --pair BITO-BKCH \
        --start 2018-01-01 \
        --end 2025-12-05

מה הסקריפט עושה בפועל:
-----------------------
1. מתחבר ל-SqlStore (DuckDB / כל Engine אחר) במצב read_only.
2. מנסה למצוא את ה-trial הכי טוב עבור הזוג:
   • קודם בכיוון שניתן (למשל "XLP-XLY")
   • אם אין — מנסה בכיוון ההפוך ("XLY-XLP") ומעדכן בהתאם.
3. שולף מהטבלה trials:
   • study_id, trial_no, score, state, sampler, n_trials, timeout_sec
   • params_json → dict "params"
   • perf_json   → dict "perf" (Sharpe/Profit/Drawdown/Score וכו')
4. מדפיס סיכום של ה-best trial.
5. בונה BacktestConfig לזוג (כולל start/end אם ניתנו).
6. מקים OptimizationBacktester עם אותם params בדיוק.
7. מריץ את ה-pipeline המקצועי (דרך helper _bt_run_professional),
   ואם אין — נופל חזרה ל-run().
8. מציג:
   • performance dict מה-backtest האמיתי
   • performance meta מה-Optuna (אם קיים)
   • כמה פרמטרים מרכזיים (z_entry, z_exit, lookback, half_life, hedge_ratio וכו' אם קיימים).
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from core.sql_store import SqlStore
from core.optimization_backtester import OptimizationBacktester, BacktestConfig

logger = logging.getLogger("replay_best_trial")


# ============================================================================
# עזר: פרשנות תאריכים מה-CLI
# ============================================================================


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay best Optuna trial for a given pair using real price data."
    )

    p.add_argument(
        "--pair",
        required=True,
        help='זוג בפורמט "SYM1-SYM2" למשל "BITO-BKCH" / "XLY-XLP"',
    )

    p.add_argument(
        "--start",
        type=_parse_date,
        default=None,
        help="תאריך התחלה YYYY-MM-DD (אופציונלי, אם לא – ה-Backtester יחליט לבד).",
    )
    p.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="תאריך סיום YYYY-MM-DD (אופציונלי, אם לא – עד היום/עד סוף הדאטה).",
    )

    p.add_argument(
        "--only-complete",
        action="store_true",
        help="לסנן רק trials במצב COMPLETE (אם state נשמר). ברירת מחדל: לא מסנן.",
    )

    p.add_argument(
        "--limit",
        type=int,
        default=1,
        help="כמה trials להביא מסודרים לפי score (ברירת מחדל: 1 – הכי טוב בלבד).",
    )

    p.add_argument(
        "--engine-url",
        default=None,
        help=(
            "Engine URL ל-SqlStore. אם לא מצוין: "
            "SqlStore.from_settings ינסה SQL_STORE_URL / config.json / ברירת מחדל."
        ),
    )

    p.add_argument(
        "--verbose",
        action="store_true",
        help="אם מצוין – מדפיס יותר פרטים (params מלאים וכו').",
    )

    return p.parse_args()


# ============================================================================
# עזר: בחירה חכמה של pair (עם reverse) ושליפת ה-best trials
# ============================================================================


def _resolve_pair_with_reverse(
    store: SqlStore,
    pair: str,
    *,
    limit: int,
    only_complete: bool,
) -> Tuple[str, pd.DataFrame]:
    """
    מנסה למצוא trials עבור pair כמו שהוא, ואם אין – עבור ההיפוך.

    מחזיר:
        (pair_used, df_best)
    כאשר:
        pair_used = הזוג שבפועל נמצאו עבורו trials.
        df_best   = DataFrame מהפונקציה get_best_trials_for_pair.
    """
    sym_a, sym_b = pair.split("-")

    df_direct = store.get_best_trials_for_pair(
        pair,
        limit=limit,
        only_complete=only_complete,
    )
    if not df_direct.empty:
        logger.info(
            "get_best_trials_for_pair: using direct pair=%s (found %s trials, top score=%s)",
            pair,
            len(df_direct),
            float(df_direct["score"].max()) if "score" in df_direct.columns else None,
        )
        return pair, df_direct

    rev_pair = f"{sym_b}-{sym_a}"
    df_rev = store.get_best_trials_for_pair(
        rev_pair,
        limit=limit,
        only_complete=only_complete,
    )
    if not df_rev.empty:
        logger.info(
            "get_best_trials_for_pair: direct pair=%s empty, but reverse=%s has %s trials (top score=%s)",
            pair,
            rev_pair,
            len(df_rev),
            float(df_rev["score"].max()) if "score" in df_rev.columns else None,
        )
        print(
            f"ℹ️ לא נמצאו trials עבור {pair}, "
            f"אבל כן נמצאו עבור {rev_pair} – נשתמש בהם."
        )
        return rev_pair, df_rev

    logger.info(
        "get_best_trials_for_pair: no trials for pair=%s or reverse=%s",
        pair,
        rev_pair,
    )
    return pair, df_direct  # df_direct ריק


# ============================================================================
# עזר: בניית BacktestConfig אמיתית לזוג ולחלון זמנים
# ============================================================================


def build_backtest_config(
    pair: str,
    start: Optional[date],
    end: Optional[date],
) -> BacktestConfig:
    """
    בונה BacktestConfig מקצועי עבור זוג נתון, לשימוש ב-replay_best_trial.

    מה היא עושה בפועל:
    -------------------
    1. מנקה ומפרק את ה-pair ל-symbol_a / symbol_b (ב-uppercase).
    2. מוודאת ש-start/end תקינים (אם ניתנו).
    3. מחשבת min_history_days באופן דינמי לפי חלון התאריכים (אם קיים),
       עם ברירת מחדל 120 ימים.
    4. מגדירה פרמטרי מסחר/ריסק סטנדרטיים:
       • initial_capital, max_gross_exposure, target_gross_leverage
       • commission_bps, slippage_bps, lot_size, וכו'.
    5. מכריחה data_source="SQL" כדי למשוך מחירים מ-SqlStore (IBKR/duckdb),
       ולא מ-AUTO / Yahoo וכד'.

    הערה:
    -----
    • אם תרחיב את BacktestConfig בעתיד (strategy/profile וכו') – זה המקום
      להזריק את השדות החדשים.
    """

    # --- 1. פירוק וניקוי הזוג ---
    if "-" not in pair:
        raise ValueError(f"pair חייב להיות בפורמט 'SYM1-SYM2', קיבלתי: {pair!r}")

    sym_a_raw, sym_b_raw = pair.split("-", 1)
    sym_a = sym_a_raw.strip().upper()
    sym_b = sym_b_raw.strip().upper()

    if not sym_a or not sym_b:
        raise ValueError(f"pair לא תקין (symbol ריק): {pair!r}")

    # --- 2. ולידציה על תאריכים ---
    if start and end and start > end:
        raise ValueError(f"start ({start}) מאוחר מ-end ({end}) עבור pair={pair}")

    # --- 3. חישוב min_history_days חכם ---
    # אם יש חלון מוגדר → נגדיר מינימום כחלק מהחלון (רבע ממנו, תחום בין 60–365)
    # אחרת → ברירת מחדל 120 ימים.
    if start and end:
        span_days = (end - start).days
        if span_days > 0:
            min_history_days = max(60, min(365, span_days // 4))
        else:
            min_history_days = 120
    else:
        min_history_days = 120

    # --- 4. בניית ה-BacktestConfig עם ערכים מקצועיים ---
    cfg = BacktestConfig(
        symbol_a=sym_a,
        symbol_b=sym_b,
        # חלון התאריכים – אם None, ה-Backtester יכול לבחור "כל ההיסטוריה"
        start=start,
        end=end,

        # דרישת היסטוריה מינימלית (ימנע מאיתנו להריץ על 10 ימים בטעות)
        min_history_days=min_history_days,

        # פרמטרים פיננסיים בסיסיים (תואם למה שכבר ראינו אצלך ב-asdict)
        initial_capital=100_000.0,
        max_gross_exposure=300_000.0,   # 3x הון ברוטו
        target_gross_leverage=1.0,
        risk_free_rate=0.02,            # 2% שנתי, אות סטנדרטי

        commission_bps=1.0,             # 1bp לכל צד
        slippage_bps=2.0,               # 2bp הנחה סולידית
        lot_size=1.0,

        dollar_neutral=True,
        beta_neutral=False,
        rebalance_days=0,               # אין rebal יזום, אלא לפי סיגנל

        bar_freq="D",                   # Daily bars

        # 🔥 חשוב – דוחף את ה-Backtester להשתמש ב-SqlStore ולא ב-AUTO/Yahoo
        data_source="SQL",

        # מטא־דאטה שימושי ל-debug / לוגים / Dashboard
        extra={
            "source": "replay_best_trial",
            "pair": pair,
            "start_cli": start.isoformat() if start else None,
            "end_cli": end.isoformat() if end else None,
            "min_history_days_auto": min_history_days,
        },
    )

    logger.info(
        "build_backtest_config: pair=%s | start=%s | end=%s | "
        "min_history_days=%s | data_source=%s",
        pair,
        start,
        end,
        min_history_days,
        getattr(cfg, "data_source", "N/A"),
    )

    return cfg



# ============================================================================
# עזר: הרצת pipeline מקצועי בצורה גמישה
# ============================================================================


def run_professional_backtest(bt: OptimizationBacktester) -> Dict[str, float]:
    """
    מריץ את הפייפליין המקצועי של OptimizationBacktester.

    כיום OptimizationBacktester.run כבר מחובר ל- _bt_run_professional_plus,
    כך שקריאה אחת ל-run() נותנת את כל הפיצ'רים:
    - טעינת מחירים (SqlStore / utils / yfinance)
    - ספרד, Hedge-Ratio, Vol Targeting, Regimes
    - סימולציית Equity + מדדי סיכון
    - דיאגנוסטיקות מתקדמות (stationarity, scenarios וכו')
    """
    return bt.run()


# ============================================================================
# עזר: הדפסות יפות
# ============================================================================


def _print_best_trial_header(row: pd.Series, pair_used: str) -> None:
    print("=== Best trial from DuckDB ===")
    print(f"Pair:        {pair_used}")
    print(f"Study ID:    {int(row['study_id'])}")
    print(f"Trial no.:   {int(row['trial_no'])}")
    score = float(row["score"]) if "score" in row and row["score"] is not None else float("nan")
    print(f"Score:       {score:.6f}")
    print(f"State:       {row.get('state')}")
    print(f"Sampler:     {row.get('sampler')}")
    print(f"n_trials:    {row.get('n_trials')}")
    print(f"timeout_sec: {row.get('timeout_sec')}")
    print()


def _print_backtest_config(cfg: BacktestConfig) -> None:
    print("=== BacktestConfig (חלקי) ===")
    try:
        print(asdict(cfg))
    except Exception:
        print(cfg)
    print()


def _print_params_preview(params: Dict[str, Any]) -> None:
    print("=== פרמטרים עיקריים (אם קיימים) ===")
    keys_of_interest = [
        "z_entry",
        "z_exit",
        "lookback",
        "half_life",
        "mean_reversion_speed",
        "hurst",
        "hedge_ratio",
        "ADF_pval",
        "adf_tstat",
        "beta_OLS",
        "beta_kalman_vol",
        "spread_mean",
        "spread_std",
    ]
    for k in keys_of_interest:
        if k in params:
            print(f"{k:22s}: {params.get(k)}")
    print()


def _print_perf_dict(title: str, perf: Dict[str, Any]) -> None:
    print(title)
    for k in sorted(perf.keys()):
        print(f"{k:12s}: {perf[k]}")
    print()


# ============================================================================
# main
# ============================================================================


def main() -> int:
    args = _parse_args()

    # 1. SqlStore (read_only, לא נוגעים בסכמות)
    settings: Dict[str, Any] = {}
    if args.engine_url:
        settings["engine_url"] = args.engine_url
    elif "SQL_STORE_URL" in os.environ:
        # נותנים ל-SqlStore.from_settings לטפל ב-SQL_STORE_URL בעצמו
        settings = {}

    store = SqlStore.from_settings(settings, read_only=True)

    # 2. להביא best trials (עם reverse אם צריך)
    pair_used, df_best = _resolve_pair_with_reverse(
        store=store,
        pair=args.pair,
        limit=args.limit,
        only_complete=args.only_complete,
    )

    if df_best.empty:
        print(
            f"⚠️ אין רשומות ב-trials עבור pair={args.pair} "
            f"(וגם לא עבור הכיוון ההפוך)."
        )
        return 1

    # משתמשים בשורה הראשונה (הכי טובה לפי score)
    row = df_best.iloc[0]
    best_params: Dict[str, Any] = row.get("params") or {}
    best_perf_meta: Dict[str, Any] = row.get("perf") or {}

    _print_best_trial_header(row, pair_used)

    # 3. BacktestConfig לזוג+חלון (עם השמות start/end שהמחלקה מכירה)
    cfg = build_backtest_config(pair_used, args.start, args.end)
    _print_backtest_config(cfg)

    if best_params:
        _print_params_preview(best_params)
    else:
        print("⚠️ לא נמצאו params מפוענחים (עמודת 'params' ריקה).")
        print()


    # 4. להקים את ה-OptimizationBacktester עם כל ההגדרות מה-cfg + הפרמטרים
    sym_a, sym_b = pair_used.split("-")

    # כל מה ששייך ל-BacktestConfig / engine (לא רוצים שידרס מה-params)
    config_kwargs = dict(
        start=cfg.start,
        end=cfg.end,
        min_history_days=cfg.min_history_days,
        initial_capital=cfg.initial_capital,
        max_gross_exposure=cfg.max_gross_exposure,
        risk_free_rate=cfg.risk_free_rate,
        commission_bps=cfg.commission_bps,
        slippage_bps=cfg.slippage_bps,
        dollar_neutral=cfg.dollar_neutral,
        beta_neutral=cfg.beta_neutral,
        rebalance_days=cfg.rebalance_days,
        bar_freq=cfg.bar_freq,
        data_source=cfg.data_source,
    )

    # מסננים מתוך best_params כל מה שכבר מופיע ב-config_kwargs
    params_clean = {
        k: v
        for k, v in (best_params or {}).items()
        if k not in config_kwargs
    }

    bt = OptimizationBacktester(
        sym_a,
        sym_b,
        **config_kwargs,
        **params_clean,
    )


    print("=== Running professional backtest... ===")
    try:
        perf_dict = run_professional_backtest(bt)
    except Exception as exc:
        logger.exception("Professional backtest failed: %s", exc)
        print(f"❌ Professional backtest failed: {exc}")
        return 2

    print()
    _print_perf_dict("=== Backtest performance dict ===", perf_dict)

    # 5. אופציונלי: להשוות לפורמט מה-Optuna (perf meta)
    if best_perf_meta:
        print()
        _print_perf_dict("=== Perf meta from Optuna (אם קיים) ===", best_perf_meta)

    # 6. אם המשתמש ביקש verbose — להדפיס את כל הפרמטרים
    if args.verbose and best_params:
        print("\n=== כל הפרמטרים (debug) ===")
        for k in sorted(best_params.keys()):
            print(f"{k:32s}: {best_params[k]}")

    return 0


if __name__ == "__main__":
    # לוג בסיסי נוח לסקריפט CLI
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | replay_best_trial | %(message)s",
    )
    raise SystemExit(main())
