# -*- coding: utf-8 -*-
"""
scripts/run_mini_fund_snapshot.py — Mini-Fund Scanner (HF-grade)
=================================================================

כלי "קרן גידור" שמבצע סריקה מעל תוצאות האופטימיזציה (DuckDB)
ובונה תמונת מצב ברמת Mini-Fund:

1. טעינת universe:
   - ברירת מחדל: רשימת זוגות ידנית (MINI_FUND_PAIRS).
   - אופציה: טעינת כל הזוגות שיש להם studies ב-DuckDB (--from-db).

2. איסוף תוצאות:
   - לוקח מספר studies אחרונים לכל זוג (ברירת מחדל: 3).
   - טוען את כל ה-trials מה-DuckDB (load_trials_from_duck).
   - משתמש בכל המטריקות הקיימות:
       Sharpe, Profit, Drawdown, Score, ParamScore, DSR, wf_robust_penalty, ...

3. חישובי HF-grade לכל זוג:
   - n_studies, n_trials_total
   - best_score, best_sharpe
   - median_sharpe, sharpe_p10/p90
   - dd_p90 (כמעט Worst-Case), dd_median
   - median_ParamScore (אם קיים)
   - median_DSR (אם קיים)
   - quality_label: "CORE" / "SATELLITE" / "WATCHLIST"
   - suggested_weight (0–1) + notional (לפי budget)

4. יצוא:
   - snapshot.csv — טבלת זוגות + סטטיסטיקות + משקלות.
   - snapshot_params.json — best_params לכל זוג (by Score & by Sharpe).
   - הדפסה יפה לקונסול.

הרצה:
    python scripts/run_mini_fund_snapshot.py
    python scripts/run_mini_fund_snapshot.py --from-db --budget 250000 --min-sharpe 0.7
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import sys
import argparse
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# שימוש בפונקציות שכבר קיימות ב-root.optimization_tab
from root.optimization_tab import (  # type: ignore
    load_trials_from_duck,
    list_studies_for_pair,
    list_pairs_in_db,
    METRIC_KEYS,
)

# =========================
# 1. הגדרות ברירת מחדל
# =========================

# Universe ידני — אפשר לשחק כאן חופשי
MINI_FUND_PAIRS: List[Tuple[str, str]] = [
    ("SPY", "QQQ"),
    ("XLY", "XLP"),
    ("ITOT", "VTI"),
    ("DIA", "SPY"),
    ("HYG", "SPY"),
]

DEFAULT_OUTPUT_DIR = Path("mini_fund_reports").resolve()
DEFAULT_MAX_STUDIES_PER_PAIR = 3


@dataclass
class PairBestParams:
    by_score: Dict[str, Any]
    by_sharpe: Dict[str, Any]


@dataclass
class PairStats:
    pair_label: str
    n_studies: int
    n_trials: int
    best_score: Optional[float]
    best_sharpe: Optional[float]
    sharpe_med: Optional[float]
    sharpe_p10: Optional[float]
    sharpe_p90: Optional[float]
    dd_med: Optional[float]
    dd_p90: Optional[float]
    param_score_med: Optional[float]
    dsr_med: Optional[float]
    wf_robust_med: Optional[float]
    quality_label: str
    weight: float
    notional: float
    best_params: PairBestParams


# =========================
# 2. Helpers
# =========================

def _pair_label(sym1: str, sym2: str) -> str:
    return f"{sym1.strip()}-{sym2.strip()}"


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _split_params_vs_metrics(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    מזהה עמודות פרמטרים לעומת מטריקות באמצעות METRIC_KEYS
    ושמות ידועים (Score, Sharpe, Drawdown וכו').
    """
    metric_like = set(
        c
        for c in df.columns
        if METRIC_KEYS.get(str(c).lower())
        or c in (
            "Score",
            "Sharpe",
            "Profit",
            "Drawdown",
            "DSR",
            "p_overfit",
            "wf_robust_penalty",
            "Pair",
            "study_id",
            "trial_no",
        )
    )
    param_cols = [c for c in df.columns if c not in metric_like]
    metric_cols = [c for c in df.columns if c in metric_like]
    return param_cols, metric_cols


def load_trials_for_pair(
    pair_label: str,
    max_studies: int = DEFAULT_MAX_STUDIES_PER_PAIR,
) -> pd.DataFrame:
    """
    טוען את כל ה-trials מהמספר האחרון של studies עבור זוג נתון.

    מחזיר DataFrame משולב עם עמודת study_id.
    """
    studies_df = list_studies_for_pair(pair_label, limit=max_studies)
    if studies_df is None or studies_df.empty:
        return pd.DataFrame()

    studies_df = studies_df.sort_values("created_at", ascending=False)
    studies = studies_df["study_id"].astype(int).tolist()

    parts: List[pd.DataFrame] = []
    for sid in studies:
        dft = load_trials_from_duck(int(sid))
        if dft is None or dft.empty:
            continue
        dft = dft.copy()
        dft["study_id"] = int(sid)
        parts.append(dft)

    if not parts:
        return pd.DataFrame()

    df_all = pd.concat(parts, ignore_index=True)
    return df_all


def compute_pair_stats(
    pair_label: str,
    df_all: pd.DataFrame,
    *,
    budget: float = 100_000.0,
    min_sharpe_core: float = 1.0,
    min_sharpe_satellite: float = 0.5,
) -> PairStats:
    """
    מחשב סטטיסטיקות HF-grade לזוג אחד בהתבסס על df_all של כל ה-trials.

    עושה:
    - זיהוי פרמטרים ומטריקות
    - חישוב best/median/quantiles
    - איכות (quality_label)
    - משקל ונוטיונל
    - פרמטרים הכי טובים (by Score & by Sharpe)
    """
    if df_all is None or df_all.empty:
        return PairStats(
            pair_label=pair_label,
            n_studies=0,
            n_trials=0,
            best_score=None,
            best_sharpe=None,
            sharpe_med=None,
            sharpe_p10=None,
            sharpe_p90=None,
            dd_med=None,
            dd_p90=None,
            param_score_med=None,
            dsr_med=None,
            wf_robust_med=None,
            quality_label="EMPTY",
            weight=0.0,
            notional=0.0,
            best_params=PairBestParams(by_score={}, by_sharpe={}),
        )

    df = df_all.copy()
    n_trials = len(df)
    n_studies = int(df["study_id"].nunique()) if "study_id" in df.columns else 1

    # המרת מטריקות לעמודות מספריות
    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[col], errors="coerce")

    sc = _num("Score")
    sh = _num("Sharpe")
    dd = _num("Drawdown").abs()
    ps = _num("ParamScore") if "ParamScore" in df.columns else pd.Series(dtype=float)
    dsr = _num("DSR") if "DSR" in df.columns else pd.Series(dtype=float)
    wf_rob = _num("wf_robust_penalty") if "wf_robust_penalty" in df.columns else pd.Series(dtype=float)

    best_score = _safe_float(sc.max()) if not sc.empty else None
    best_sharpe = _safe_float(sh.max()) if not sh.empty else None

    sharpe_med = _safe_float(sh.median()) if not sh.empty else None
    sharpe_p10 = _safe_float(sh.quantile(0.10)) if not sh.empty else None
    sharpe_p90 = _safe_float(sh.quantile(0.90)) if not sh.empty else None

    dd_med = _safe_float(dd.median()) if not dd.empty else None
    dd_p90 = _safe_float(dd.quantile(0.90)) if not dd.empty else None

    param_score_med = _safe_float(ps.median()) if not ps.empty else None
    dsr_med = _safe_float(dsr.median()) if not dsr.empty else None
    wf_robust_med = _safe_float(wf_rob.median()) if not wf_rob.empty else None

    # איכות: CORE / SATELLITE / WATCHLIST
    quality_label = "WATCHLIST"
    if sharpe_med is not None:
        if sharpe_med >= min_sharpe_core:
            quality_label = "CORE"
        elif sharpe_med >= min_sharpe_satellite:
            quality_label = "SATELLITE"
        else:
            quality_label = "WATCHLIST"

    # בחירת best_params לפי Score / Sharpe
    param_cols, metric_cols = _split_params_vs_metrics(df)

    best_by_score: Dict[str, Any] = {}
    best_by_sharpe: Dict[str, Any] = {}

    if not df.empty and param_cols:
        # לפי Score
        if "Score" in df.columns:
            df_sc = df.sort_values("Score", ascending=False)
            row = df_sc.iloc[0]
            best_by_score = {c: row[c] for c in param_cols}

        # לפי Sharpe
        if "Sharpe" in df.columns:
            df_sh = df.sort_values("Sharpe", ascending=False)
            row2 = df_sh.iloc[0]
            best_by_sharpe = {c: row2[c] for c in param_cols}

    # הצעת משקל: שילוב Sharpe ו-Score
    # (אפשר לשחק בזה בעתיד)
    weight_raw = 0.0
    if best_score is not None and best_sharpe is not None:
        # נרמול גס: Score ~ 0-3, Sharpe ~ 0-3
        weight_raw = max(0.0, best_sharpe) + max(0.0, best_score)
    elif best_sharpe is not None:
        weight_raw = max(0.0, best_sharpe)
    elif best_score is not None:
        weight_raw = max(0.0, best_score)

    # ניקוי קטן — זה עדיין רק פרופורציה
    weight = float(weight_raw)

    # notional יחושב בשלב מאוחר יותר אחרי שנראה את כל הזוגות
    notional = 0.0

    return PairStats(
        pair_label=pair_label,
        n_studies=n_studies,
        n_trials=n_trials,
        best_score=best_score,
        best_sharpe=best_sharpe,
        sharpe_med=sharpe_med,
        sharpe_p10=sharpe_p10,
        sharpe_p90=sharpe_p90,
        dd_med=dd_med,
        dd_p90=dd_p90,
        param_score_med=param_score_med,
        dsr_med=dsr_med,
        wf_robust_med=wf_robust_med,
        quality_label=quality_label,
        weight=weight,
        notional=notional,
        best_params=PairBestParams(by_score=best_by_score, by_sharpe=best_by_sharpe),
    )


def build_mini_fund_snapshot(
    pairs: Sequence[Tuple[str, str]],
    *,
    budget: float = 100_000.0,
    max_studies_per_pair: int = DEFAULT_MAX_STUDIES_PER_PAIR,
    min_sharpe_filter: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, PairBestParams]]:
    """
    רץ על כל הזוגות, מחשב stats, מנרמל משקלות, ומחזיר:

    df_snapshot:
        Pair | n_studies | n_trials | Best_Score | Best_Sharpe | Sharpe_med | ... | Quality | Weight | Notional

    best_params_map:
        dict: pair_label → PairBestParams
    """
    stats_list: List[PairStats] = []

    for sym1, sym2 in pairs:
        pair_label = _pair_label(sym1, sym2)
        df_all = load_trials_for_pair(pair_label, max_studies=max_studies_per_pair)
        stats = compute_pair_stats(
            pair_label,
            df_all,
            budget=budget,
        )
        # פילטר לפי Sharpe median, אם ביקשת
        if stats.sharpe_med is not None and stats.sharpe_med < min_sharpe_filter:
            continue
        stats_list.append(stats)

    if not stats_list:
        return pd.DataFrame(), {}

    # נרמול משקלות לזוגות
    weights_raw = np.array([max(s.weight, 0.0) for s in stats_list], dtype=float)
    if not np.any(weights_raw > 0):
        weights_raw[:] = 1.0
    weights_norm = weights_raw / weights_raw.sum()

    for idx, s in enumerate(stats_list):
        s.weight = float(weights_norm[idx])
        s.notional = float(weights_norm[idx] * budget)

    # בניית DataFrame
    rows: List[Dict[str, Any]] = []
    best_params_map: Dict[str, PairBestParams] = {}

    for s in stats_list:
        best_params_map[s.pair_label] = s.best_params
        rows.append(
            {
                "Pair": s.pair_label,
                "n_studies": s.n_studies,
                "n_trials": s.n_trials,
                "Best_Score": s.best_score,
                "Best_Sharpe": s.best_sharpe,
                "Sharpe_med": s.sharpe_med,
                "Sharpe_p10": s.sharpe_p10,
                "Sharpe_p90": s.sharpe_p90,
                "DD_med": s.dd_med,
                "DD_p90": s.dd_p90,
                "ParamScore_med": s.param_score_med,
                "DSR_med": s.dsr_med,
                "WF_robust_med": s.wf_robust_med,
                "Quality": s.quality_label,
                "Weight": s.weight,
                "Notional": s.notional,
            }
        )

    df_snapshot = pd.DataFrame(rows)
    # ממיינים לפי Quality ואז לפי Best_Score
    sort_cols = ["Quality", "Best_Score"]
    df_snapshot = df_snapshot.sort_values(sort_cols, ascending=[True, False]).reset_index(drop=True)
    return df_snapshot, best_params_map


# =========================
# 3. CLI & Main
# =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mini-Fund Snapshot (HF-grade)")
    p.add_argument(
        "--from-db",
        action="store_true",
        help="Use all pairs from DuckDB (list_pairs_in_db) instead of MINI_FUND_PAIRS",
    )
    p.add_argument(
        "--limit-pairs",
        type=int,
        default=0,
        help="Limit number of pairs from DB (0 = no limit)",
    )
    p.add_argument(
        "--budget",
        type=float,
        default=100_000.0,
        help="Total budget for sleeve (used to compute Notional per pair)",
    )
    p.add_argument(
        "--min-sharpe",
        type=float,
        default=0.0,
        help="Minimum median Sharpe to keep a pair in the snapshot",
    )
    p.add_argument(
        "--max-studies",
        type=int,
        default=DEFAULT_MAX_STUDIES_PER_PAIR,
        help="Number of latest studies to aggregate per pair",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for snapshot files",
    )
    return p.parse_args()


def build_universe(from_db: bool, limit_pairs: int) -> List[Tuple[str, str]]:
    if not from_db:
        return list(MINI_FUND_PAIRS)

    # טעינת זוגות מ-DuckDB
    pairs_labels = list_pairs_in_db(limit=200 if limit_pairs <= 0 else limit_pairs)
    out: List[Tuple[str, str]] = []
    for pl in pairs_labels:
        if "-" not in pl:
            continue
        a, b = pl.split("-", 1)
        a, b = a.strip(), b.strip()
        if a and b:
            out.append((a, b))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = build_universe(args.from_db, args.limit_pairs)
    if not universe:
        print("No pairs in universe (check MINI_FUND_PAIRS or DuckDB).")
        return

    print("=== Mini-Fund Snapshot (HF-grade) ===")
    print(f"Universe size: {len(universe)} pairs")
    print(f"Budget: {args.budget:,.0f} USD")
    print(f"Max studies per pair: {args.max_studies}")
    if args.min_sharpe > 0:
        print(f"Sharpe median filter: ≥ {args.min_sharpe:.2f}")
    print("Collecting data from DuckDB...")

    df_snapshot, best_params_map = build_mini_fund_snapshot(
        universe,
        budget=args.budget,
        max_studies_per_pair=args.max_studies,
        min_sharpe_filter=args.min_sharpe,
    )

    if df_snapshot.empty:
        print("No data found for the given universe/filters.")
        return

    # הדפסה יפה
    with pd.option_context("display.max_colwidth", 80, "display.width", 180):
        print("\n--- Snapshot ---")
        print(df_snapshot.to_string(index=False))

    # שמירת snapshot.csv
    csv_path = out_dir / "mini_fund_snapshot.csv"
    df_snapshot.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nSnapshot saved to: {csv_path}")

    # שמירת best_params כ-JSON
    params_payload: Dict[str, Dict[str, Any]] = {}
    for pair_label, bp in best_params_map.items():
        params_payload[pair_label] = {
            "best_by_score": bp.by_score,
            "best_by_sharpe": bp.by_sharpe,
        }
    json_path = out_dir / "mini_fund_best_params.json"
    json_path.write_text(json.dumps(params_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best-params map saved to: {json_path}")


if __name__ == "__main__":
    main()
