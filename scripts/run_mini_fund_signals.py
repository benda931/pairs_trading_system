# -*- coding: utf-8 -*-
"""
scripts/run_mini_fund_signals.py — Mini-Fund Daily Signals (HF-grade + ParamScore)
==================================================================================

מטרות:
-------
1. לקרוא:
   - mini_fund_reports/mini_fund_snapshot.csv  (תמונת מצב ביצועים + Notional לזוג)
   - mini_fund_reports/mini_fund_best_params.json (best params לכל זוג)

2. לייצר טבלת סיגנלים ברמת Mini-Fund:
   Pair | Sym_A | Sym_B | Signal | Side_A | Qty_A | Side_B | Qty_B | Price_A | Price_B | Z |
   Notional | ParamScore | PerfNorm | ParamNorm | CombinedScore | Comment

3. ParamScore:
   - מחושב לפי core.params.score_params_dict(params)
   - משקף את איכות סט הפרמטרים לפי הציונים/צבעים/משקולות ב-params.py.

4. CombinedScore:
   - ציון משוכלל של:
       • ביצועים היסטוריים (Best_Score או Best_Sharpe מה-snapshot)
       • איכות פרמטרים (ParamScore)
   - ניתן לשנות את המשקולות W_PERF ו-W_PARAM לפי טעם.

הערה:
------
בגרסה הזו מקור מחירים הוא yfinance (לצורך ריצה מהירה).
במערכת הסופית מומלץ להחליף למשיכה מ-SqlStore/IBKR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import datetime as dt
import json
import math

import numpy as np
import pandas as pd

# ---- ParamScore מתוך core.params ----
try:
    from core.params import score_params_dict as params_score_params_dict  # type: ignore
except Exception:
    params_score_params_dict = None  # type: ignore

# ---- מקור מחירים זמני (yfinance) ----
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None  # type: ignore


# ========= קונפיג ברירת מחדל =========

REPORTS_DIR = Path("mini_fund_reports").resolve()
SNAPSHOT_CSV = REPORTS_DIR / "mini_fund_snapshot.csv"
BEST_PARAMS_JSON = REPORTS_DIR / "mini_fund_best_params.json"

DEFAULT_LOOKBACK_DAYS = 60    # אם ב-best_params אין lookback
MIN_STD_EPS = 1e-6            # כדי לא לחלק ב-0


@dataclass
class PairSignal:
    pair_label: str
    sym_a: str
    sym_b: str
    signal: str           # "ENTER", "EXIT", "HOLD", "SKIP"
    side_a: str           # "LONG"/"SHORT"/"-"
    side_b: str           # "LONG"/"SHORT"/"-"
    qty_a: int
    qty_b: int
    price_a: Optional[float]
    price_b: Optional[float]
    z_value: Optional[float]
    notional: float
    comment: str

    # ציון פרמטרים לפי params.py
    param_score: Optional[float] = None     # 0–1
    combined_score: Optional[float] = None  # ימולא בשלב טבלת הסיגנלים


# ========= 1. קריאת snapshot + params =========

def load_snapshot_and_params() -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    if not SNAPSHOT_CSV.exists():
        raise FileNotFoundError(f"Snapshot CSV not found: {SNAPSHOT_CSV}")
    if not BEST_PARAMS_JSON.exists():
        raise FileNotFoundError(f"Best-params JSON not found: {BEST_PARAMS_JSON}")

    df_snap = pd.read_csv(SNAPSHOT_CSV)
    params_text = BEST_PARAMS_JSON.read_text(encoding="utf-8")
    params_dict: Dict[str, Dict[str, Any]] = json.loads(params_text) or {}

    return df_snap, params_dict


# ========= 2. פונקציות מחירים (זמנית: yfinance) =========

def get_price_series(symbol: str, lookback_days: int = 120) -> pd.Series:
    """
    מחזיר סדרת מחירי Adj Close ל-symbol מסוים מתוך yfinance.
    בגרסה סופית מומלץ להחליף למקור דאטה פנימי (SqlStore/IBKR).
    """
    if yf is None:
        raise RuntimeError(
            "yfinance is not installed; cannot fetch prices. "
            "Install with: pip install yfinance"
        )

    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days * 2)  # *2 לכיסוי סופי שבוע
    data = yf.download(symbol, start=start, end=end)
    if data.empty or "Adj Close" not in data.columns:
        raise RuntimeError(f"No price data for symbol {symbol}")
    s = data["Adj Close"].dropna()
    return s


def get_last_lookback_window(symbol: str, lookback_days: int) -> pd.Series:
    """
    מחזיר חלון מחירים אחרון באורך lookback_days (או פחות אם אין מספיק דאטה).
    """
    s = get_price_series(symbol, lookback_days=lookback_days)
    if s.empty:
        raise RuntimeError(f"No price history for {symbol}")
    return s.tail(lookback_days)


# ========= 3. חישוב Spread ו-Z =========

def compute_spread_and_z(
    prices_a: pd.Series,
    prices_b: pd.Series,
    params: Dict[str, Any],
) -> Tuple[float, float, float]:
    """
    מחשב:
        spread_t = P_A - hedge_ratio * P_B
        mean_spread, std_spread (לפי lookback)
        z_t = (spread_t - mean) / std

    פרמטרים רלוונטיים:
        - lookback
        - hedge_ratio (אם קיים; אחרת 1.0)
    """
    df = pd.concat(
        [prices_a.rename("A"), prices_b.rename("B")],
        axis=1,
    ).dropna()
    if df.empty:
        raise RuntimeError("No overlapping price history for the pair")

    hedge_ratio = float(params.get("hedge_ratio", 1.0) or 1.0)
    spread = df["A"] - hedge_ratio * df["B"]

    mean_spread = float(spread.mean())
    std_spread = float(spread.std(ddof=0) or 0.0)
    if std_spread <= 0:
        std_spread = MIN_STD_EPS

    spread_t = float(spread.iloc[-1])
    z_t = (spread_t - mean_spread) / std_spread

    return spread_t, mean_spread, z_t


# ========= 4. החלטת סיגנל לפי Z והפרמטרים =========

def decide_signal(
    z_value: float,
    params: Dict[str, Any],
) -> Tuple[str, str, str, str]:
    """
    מחליט על מצב:
        signal: "ENTER"/"EXIT"/"HOLD"
        side_A/side_B: LONG / SHORT / "-"

    לוגיקה:
        z_entry = params["z_entry"] / "z_open" / 2.0 (ברירת מחדל)
        z_exit  = params["z_exit"] / "z_close" / 0.5

        אם |Z| < z_exit  → EXIT (אין פוזיציה)
        אם Z > z_entry   → ENTER: A SHORT, B LONG
        אם Z < -z_entry  → ENTER: A LONG,  B SHORT
        אחרת             → HOLD
    """
    z_entry = float(params.get("z_entry", params.get("z_open", 2.0)) or 2.0)
    z_exit = float(params.get("z_exit", params.get("z_close", 0.5)) or 0.5)

    abs_z = abs(z_value)

    if abs_z < z_exit:
        return "EXIT", "-", "-", f"Z={z_value:.2f} inside exit band (|Z| < z_exit={z_exit:.2f})"

    if z_value > z_entry:
        return "ENTER", "SHORT", "LONG", f"Z={z_value:.2f} > z_entry={z_entry:.2f} (A expensive vs B)"

    if z_value < -z_entry:
        return "ENTER", "LONG", "SHORT", f"Z={z_value:.2f} < -z_entry={z_entry:.2f} (A cheap vs B)"

    return "HOLD", "-", "-", (
        f"Z={z_value:.2f} in mid-band "
        f"(z_exit={z_exit:.2f} ≤ |Z| ≤ z_entry={z_entry:.2f})"
    )


# ========= 5. חישוב כמות מכל leg =========

def compute_leg_quantities(
    notional: float,
    price_a: Optional[float],
    price_b: Optional[float],
    signal: str,
    side_a: str,
    side_b: str,
    *,
    leg_weight_a: float = 0.5,
    leg_weight_b: float = 0.5,
) -> Tuple[int, int]:
    """
    מחשב כמה יחידות מכל leg:

        notional_total → notional_a / notional_b
        Qty = floor(notional_leg / price_leg)

    אם signal != "ENTER" → 0 יחידות.
    """
    if signal != "ENTER":
        return 0, 0
    if notional <= 0:
        return 0, 0

    notional_a = notional * float(leg_weight_a)
    notional_b = notional * float(leg_weight_b)

    qty_a = 0
    qty_b = 0

    if price_a and price_a > 0 and side_a in ("LONG", "SHORT"):
        qty_a = int(math.floor(notional_a / price_a))
    if price_b and price_b > 0 and side_b in ("LONG", "SHORT"):
        qty_b = int(math.floor(notional_b / price_b))

    return qty_a, qty_b


# ========= 6. סיגנל לזוג אחד =========

def compute_signal_for_pair(
    sym_a: str,
    sym_b: str,
    pair_label: str,
    notional: float,
    params_map: Dict[str, Any],
) -> PairSignal:
    """
    חישוב סיגנל לזוג:

    1. בחירת סט פרמטרים:
        - קודם best_by_score מה-JSON.
        - אם ריק → best_by_sharpe.
    2. חישוב ParamScore לפי params.py.
    3. משיכת מחירים אחרונים והערכת Z.
    4. החלטת סיגנל + צדדים.
    5. חישוב כמות מכל leg.
    """
    best_by_score = params_map.get("best_by_score") or {}
    best_by_sharpe = params_map.get("best_by_sharpe") or {}
    params = best_by_score if best_by_score else best_by_sharpe

    if not params:
        return PairSignal(
            pair_label=pair_label,
            sym_a=sym_a,
            sym_b=sym_b,
            signal="SKIP",
            side_a="-",
            side_b="-",
            qty_a=0,
            qty_b=0,
            price_a=None,
            price_b=None,
            z_value=None,
            notional=notional,
            comment="No best_params defined for this pair",
            param_score=None,
            combined_score=None,
        )

    # ParamScore לפי core.params
    param_score: Optional[float] = None
    if params_score_params_dict is not None:
        try:
            ps = params_score_params_dict(params)  # dict עם total_score וכו'
            param_score = float(ps.get("total_score", 0.0))
        except Exception:
            param_score = None

    lookback = int(params.get("lookback", DEFAULT_LOOKBACK_DAYS) or DEFAULT_LOOKBACK_DAYS)

    try:
        prices_a = get_last_lookback_window(sym_a, lookback)
        prices_b = get_last_lookback_window(sym_b, lookback)
        price_a = float(prices_a.iloc[-1])
        price_b = float(prices_b.iloc[-1])

        spread_t, mean_spread, z_t = compute_spread_and_z(prices_a, prices_b, params)
        signal, side_a, side_b, reason = decide_signal(z_t, params)
        qty_a, qty_b = compute_leg_quantities(
            notional=notional,
            price_a=price_a,
            price_b=price_b,
            signal=signal,
            side_a=side_a,
            side_b=side_b,
        )

        comment = (
            f"{reason} | P_A={price_a:.2f}, P_B={price_b:.2f}, "
            f"spread={spread_t:.4f}, mean={mean_spread:.4f}"
        )

        return PairSignal(
            pair_label=pair_label,
            sym_a=sym_a,
            sym_b=sym_b,
            signal=signal,
            side_a=side_a,
            side_b=side_b,
            qty_a=qty_a,
            qty_b=qty_b,
            price_a=price_a,
            price_b=price_b,
            z_value=z_t,
            notional=notional,
            comment=comment,
            param_score=param_score,
            combined_score=None,
        )
    except Exception as e:
        return PairSignal(
            pair_label=pair_label,
            sym_a=sym_a,
            sym_b=sym_b,
            signal="SKIP",
            side_a="-",
            side_b="-",
            qty_a=0,
            qty_b=0,
            price_a=None,
            price_b=None,
            z_value=None,
            notional=notional,
            comment=f"Error computing signal: {e}",
            param_score=param_score,
            combined_score=None,
        )


# ========= 7. בניית טבלת סיגנלים לכל המיני-פאנד =========

def build_signals_table() -> pd.DataFrame:
    df_snap, params_map = load_snapshot_and_params()

    if df_snap.empty:
        raise RuntimeError("Snapshot CSV is empty — run run_mini_fund_snapshot.py first.")

    rows: List[Dict[str, Any]] = []

    for _, row in df_snap.iterrows():
        pair_label = str(row["Pair"])
        if "-" not in pair_label:
            continue

        sym_a, sym_b = pair_label.split("-", 1)
        sym_a = sym_a.strip()
        sym_b = sym_b.strip()

        notional = float(row.get("Notional", 0.0) or 0.0)

        best_params_entry = params_map.get(pair_label)
        if best_params_entry is None:
            sig = PairSignal(
                pair_label=pair_label,
                sym_a=sym_a,
                sym_b=sym_b,
                signal="SKIP",
                side_a="-",
                side_b="-",
                qty_a=0,
                qty_b=0,
                price_a=None,
                price_b=None,
                z_value=None,
                notional=notional,
                comment="Pair not found in mini_fund_best_params.json",
                param_score=None,
                combined_score=None,
            )
        else:
            sig = compute_signal_for_pair(
                sym_a=sym_a,
                sym_b=sym_b,
                pair_label=pair_label,
                notional=notional,
                params_map=best_params_entry,
            )

        rows.append(
            {
                "Pair": sig.pair_label,
                "Sym_A": sig.sym_a,
                "Sym_B": sig.sym_b,
                "Signal": sig.signal,
                "Side_A": sig.side_a,
                "Qty_A": sig.qty_a,
                "Side_B": sig.side_b,
                "Qty_B": sig.qty_b,
                "Price_A": sig.price_a,
                "Price_B": sig.price_b,
                "Z": sig.z_value,
                "Notional": sig.notional,
                "ParamScore": sig.param_score,
                "Best_Score": row.get("Best_Score"),
                "Best_Sharpe": row.get("Best_Sharpe"),
                "Comment": sig.comment,
            }
        )

    df_signals = pd.DataFrame(rows)

    # ========= 8. חישוב CombinedScore =========

    if not df_signals.empty:
        # נרמול ביצועים (עדיף Best_Score, אחרת Best_Sharpe)
        perf_col = None
        if "Best_Score" in df_signals.columns:
            perf_col = "Best_Score"
        elif "Best_Sharpe" in df_signals.columns:
            perf_col = "Best_Sharpe"

        if perf_col is not None:
            perf = pd.to_numeric(df_signals[perf_col], errors="coerce")
            perf_min = perf.min()
            perf_max = perf.max()
            perf_span = (perf_max - perf_min) if perf_max > perf_min else 1.0
            df_signals["PerfNorm"] = (perf - perf_min) / perf_span
        else:
            df_signals["PerfNorm"] = np.nan

        # נרמול ParamScore
        if "ParamScore" in df_signals.columns:
            ps = pd.to_numeric(df_signals["ParamScore"], errors="coerce")
            if ps.notna().any():
                ps_min = ps.min()
                ps_max = ps.max()
                ps_span = (ps_max - ps_min) if ps_max > ps_min else 1.0
                df_signals["ParamNorm"] = (ps - ps_min) / ps_span
            else:
                df_signals["ParamNorm"] = np.nan
        else:
            df_signals["ParamNorm"] = np.nan

        # משקולות לציון המשוכלל — אפשר לשחק כאן
        W_PERF = 0.6   # כמה משקל לתת לביצועים היסטוריים
        W_PARAM = 0.4  # כמה משקל לתת לאיכות הפרמטרים

        df_signals["CombinedScore"] = (
            W_PERF * df_signals["PerfNorm"].fillna(0.0)
            + W_PARAM * df_signals["ParamNorm"].fillna(0.0)
        )

    return df_signals


# ========= 9. main =========

def main() -> None:
    print("=== Mini-Fund Daily Signals (with ParamScore & CombinedScore) ===")

    df = build_signals_table()
    if df.empty:
        print("No signals (empty table). Check snapshot/params.")
        return

    # הדפסה יפה למסך
    with pd.option_context("display.max_colwidth", 120, "display.width", 220):
        print(df.to_string(index=False))

    # שמירה ל-CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "mini_fund_signals.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSignals saved to: {out_path}")


if __name__ == "__main__":
    main()
