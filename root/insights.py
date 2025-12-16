# -*- coding: utf-8 -*-
"""
insights.py — טאב תובנות (Streamlit) — גרסת Hedge-Fund (v4)
-------------------------------------------------------------
יכולות מרכזיות:
- טעינת לוגים חכמה (logs/ או נתיב מותאם / קונטקסט מה־dashboard)
- נרמול עמודות גם עם mojibake ושמות מעורבבים
- פילטרים מתקדמים (זוג, טווחי רווח, תאריך, כיוון, k-sigma, משך עסקה, קבוצה, notional)
- KPI-ים מקצועיים:
    * WinRate, Profit Factor, Sharpe per trade
    * Skew/Kurtosis, Tail-Ratio, Expectancy, Payoff
    * Max Drawdown על עקומת ההון + MAR, Drawdown per pair
- ויזואליזציות Plotly:
    * Top/Bottom 10 pairs
    * Box Plot per pair
    * Heatmap PnL
    * Monthly PnL + Pair×Month
    * Histogram, Equity Curve, Benchmarks (Top-N vs All)
    * Drill-down לפי זוג, פילוח לפי משך / צד / קבוצה
- ML Bridge:
    * חיבור ל-core.ml_analysis — תצוגת ML summary מהטאב Optimisation
- Fair Value Scanner:
    * חיבור ל-FairValueEngine על prices_wide + Universe של זוגות
- AI Agents:
    * System Upgrader Agent — קורא קוד/לוגים ומציע שדרוגים
    * Visualization Agent — מייצר גרפי Plotly משודרגים מקובצי CSV
- GPT:
    * ניתוח טקסטואלי של Top pairs, תובנות ובדיקות אמפיריות עתידיות
- Export:
    * CSV/Excel/Markdown, כולל Config-friendly snapshot
- עמיד לשגיאות: אין קבצים / עמודות חסרות / בלי OpenAI — הטאב ממשיך לעבוד.

שדרוגים מרכזיים בגרסה v4:
--------------------------
1. טיפול מלא ב-FutureWarning של pandas (groupby עם observed=False).
2. Helper לאריזת DataFrame ל-Arrow/Streamlit ללא אזהרות (column names → str).
3. בלוק KPIs משודרג עם Tail-Ratio, Expectancy, Payoff, Gross P/L, Quantiles.
4. פילטר נוסף לפי Notional (אם קיים), עם טווח דינמי.
5. פילוח לפי משך עסקה (duration_days) ל-Bins עם טבלה וגרף.
6. Benchmarks: Equity All vs Top-N + MoM stats.
7. Drill-down לזוג ספציפי עם Equity שלו.
8. Excel Export עם כמה sheets (KPIs / Logs / Monthly / Pair×Month).
9. Markdown Report עם KPIs + head של הלוג.
10. שימוש ב-make_arrow_safe + המרת שמות עמודות ל-str לכל ה-dataframes שמוצגים ב-UI.
11. Universe FairValue Scanner עם מסננים נוספים (z_abs / net_edge / quality_weight).
12. מגבלת max_pairs לניתוח כדי לא להעמיס על ה-UI.
13. System Summary קטן עם טווח תאריכים ונתוני יסוד.
14. טיפול בחריגים (Anomalies) לפי kσ עם דגלים ברורים.
15. תמיכה מלאה בקונטקסט גלובלי מה-dashboard (start/end/paths).
16. שילוב ML bridge, Agents ו-GPT בלי לשבור את זרימת ה-Insights.
17. קיטוע מודולרי: פונקציות ברורות (normalize, filters, KPIs, export, agents).
18. שימוש עקבי ב-try/except סביב חלקים כבדים כדי לא להפיל את הטאב.
19. אפשרות להגביל את מספר הזוגות לניתוח (max_pairs).
20. הוספת __all__ נקי: רק render_insights_tab נחשף.

"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from root.system_upgrader_agent import run_upgrader, FixReport  # type: ignore
from root.visualization_agent import VisualizationAgent  # type: ignore
from core.ml_analysis import render_ml_bridge_panel
from core.fair_value_engine import FairValueEngine, Config  # type: ignore

# ===== OpenAI (אופציונלי) =====
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = lambda *a, **k: None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

load_dotenv()

logger = logging.getLogger("insights")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [insights] — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ===== הגדרות ותוויות =====
CANON_COLS: Dict[str, set[str]] = {
    # canonical name -> possible variants (עברית, אנגלית, ו-mojibake שכיח)
    "pair": {"צמד", "זוג", "pair", "׳–׳•׳’"},
    "pnl": {"רווח ($)", "PnL ($)", "PnL", "pnl", "תשואה", "׳¨׳•׳•׳— ($)"},
    "created": {"כניסה", "תאריך", "date", "created", "׳›׳ ׳™׳¡׳”"},
    "finished": {"יציאה", "סיום", "finished", "׳™׳¦׳™׳׳”"},
    # אופציונלי
    "bars": {"bars_held", "משך ברים", "bars", "bars held"},
    "side": {"כיוון", "side", "direction"},
    "group": {"קבוצה", "sector", "group", "bucket"},
}

HEB: Dict[str, str] = {
    "title": "🔍 תובנות מערכת — לוגים וניתוחים",
    "no_logs": "לא נמצאו קובצי לוג בתיקייה שנבחרה",
    "missing_pnl": "העמודה 'רווח ($)' לא נמצאה ולא זוהתה בשמות חלופיים — בדקו את קבצי הלוג.",
    "stats_header": "📈 סיכום סטטיסטי",
    "box_title": "📦 פיזור רווחים לפי זוג",
    "heatmap_title": "🔥 Heatmap — רווח מצטבר לפי זוג",
    "monthly_title": "🗓️ רווח חודשי",
    "pair_month_title": "🧩 מטריצת זוג × חודש",
    "dist_title": "📊 התפלגות רווחים (Histogram)",
    "eq_title": "📈 עקומת הון מצטברת",
    "top_title": "🏆 Top/Bottom זוגות",
    "drill_title": "🔎 Drill-down לפי זוג",
    "gpt_title": "🧠 ניתוח GPT לתובנות טבלאיות",
    "gpt_btn": "צור תובנות חכמות",
    "gpt_error": "שגיאה בגישה ל-GPT",
    "save_md": "📝 שמירת דוח Markdown",
    "saved_ok": "דוח נשמר ל-logs/insights_report.md",
}

# ===== מודל עזר ל-KPI =====
@dataclass
class KPIs:
    trades: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    median_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = float("nan")
    sharpe_per_trade: float = 0.0
    skew: float = 0.0
    kurt: float = 0.0
    max_drawdown: float = 0.0
    mar_ratio: float = float("nan")
    best_trade: float = 0.0
    worst_trade: float = 0.0
    tail_ratio: float = float("nan")  # |avg top 5%| / |avg bottom 5%|


# ===== עזר: Max Drawdown על סדרת PnL מצטברת =====
def _max_drawdown(equity: np.ndarray) -> Tuple[float, Optional[int], Optional[int]]:
    if equity.size == 0:
        return 0.0, None, None
    peak = equity[0]
    max_dd = 0.0
    peak_idx = 0
    dd_start = 0
    dd_end = 0
    for i, v in enumerate(equity):
        if v > peak:
            peak = v
            peak_idx = i
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
            dd_start = peak_idx
            dd_end = i
    return float(max_dd), dd_start, dd_end


# ===== Arrow-friendly helper (שמות עמודות / ערכים) =====
def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    הופך DataFrame לידידותי ל-PyArrow/Streamlit:
    - דואג שעמודות בעייתיות כמו 'Value' ו-'default' יהיו string.
    """
    df2 = df.copy()
    for col in ("Value", "default"):
        if col in df2.columns:
            df2[col] = df2[col].astype("string")
    return df2


def _arrow_friendly_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper לשימוש לפני st.dataframe:
    - ממיר שמות עמודות ל-str (הימנעות מ-mixed-type column names).
    - משתמש ב-make_arrow_safe לעמודות ספציפיות.
    """
    df2 = df.copy()
    try:
        df2.columns = df2.columns.map(str)
    except Exception:
        pass
    df2 = make_arrow_safe(df2)
    return df2


# ===== נרמול שמות עמודות =====
def _normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}

    def find_any(candidates: set[str]) -> Optional[str]:
        # חיפוש מדויק ואז lowercase
        for c in df.columns:
            if c in candidates:
                return c
        lowers = {x.lower(): x for x in candidates}
        for c in df.columns:
            if str(c).lower().strip() in lowers:
                return c
        return None

    df = df.copy()
    for canon, variants in CANON_COLS.items():
        col = find_any(variants)
        if col is not None:
            mapping[canon] = col

    # עמודת משך בימים
    if "finished" in mapping and "created" in mapping and "duration_days" not in df.columns:
        try:
            df[mapping["created"]] = pd.to_datetime(df[mapping["created"]], errors="coerce")
            df[mapping["finished"]] = pd.to_datetime(df[mapping["finished"]], errors="coerce")
            df["duration_days"] = (df[mapping["finished"]] - df[mapping["created"]]).dt.days.clip(lower=1)
        except Exception:
            df["duration_days"] = 1
    elif "duration_days" not in df.columns:
        df["duration_days"] = 1

    # ודא שה-PnL מספרי והסר NaN
    pnl_col = mapping.get("pnl")
    if pnl_col is not None:
        df[pnl_col] = pd.to_numeric(df[pnl_col], errors="coerce")
        df = df.dropna(subset=[pnl_col])

    return df, mapping


# ===== קריאת קבצים בבטחה =====
@st.cache_data(show_spinner=False)
def load_logs_dir(path: str | Path = "logs", pattern: str = "_log.csv") -> Tuple[pd.DataFrame, Dict[str, str]]:
    base = Path(path)
    if not base.is_dir():
        return pd.DataFrame(), {}

    frames: List[pd.DataFrame] = []
    for file in base.iterdir():
        if not file.is_file():
            continue
        if pattern and not str(file.name).endswith(pattern):
            continue
        last_err: Optional[Exception] = None
        for enc in ("utf-8", "utf-8-sig", "cp1255", "latin-1"):
            try:
                df = pd.read_csv(file, encoding=enc)
                df["__PAIR_FILE__"] = file.stem.replace("_log", "")
                frames.append(df)
                break
            except Exception as e:  # pragma: no cover
                last_err = e
                continue
        else:
            logger.warning("Failed to read %s: %s", file, last_err)

    if not frames:
        return pd.DataFrame(), {}

    df = pd.concat(frames, ignore_index=True)
    df, mapping = _normalize_columns(df)
    return df, mapping


# ===== KPI =====
def _compute_kpis(df: pd.DataFrame, pnl_col: str, created_col: Optional[str]) -> KPIs:
    if df.empty:
        return KPIs()

    pnls = df[pnl_col].astype(float).to_numpy()
    trades = int(len(pnls))
    if trades == 0:
        return KPIs()

    total = float(pnls.sum())
    avg = float(pnls.mean())
    med = float(np.median(pnls))
    wins_mask = pnls > 0
    losses_mask = pnls < 0
    wins = pnls[wins_mask]
    losses = pnls[losses_mask]

    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss_abs = float(-losses.sum()) if losses.size else 0.0
    profit_factor = float(gross_profit / gross_loss_abs) if gross_loss_abs > 0 else float("nan")

    std = float(pnls.std(ddof=0))
    sharpe = float((pnls.mean() / std * sqrt(trades))) if trades > 1 and std > 0 else 0.0
    win_rate = float(wins_mask.sum() / trades)

    best = float(pnls.max())
    worst = float(pnls.min())

    # Moments
    mean = pnls.mean()
    std_m = pnls.std(ddof=0)
    if std_m > 0:
        skew = float(((pnls - mean) ** 3).mean() / (std_m ** 3))
        kurt = float(((pnls - mean) ** 4).mean() / (std_m ** 4))
    else:
        skew = kurt = 0.0

    # Tail ratio (avg top 5% / avg bottom 5%)
    q95 = np.quantile(pnls, 0.95)
    q05 = np.quantile(pnls, 0.05)
    tail_up = pnls[pnls >= q95]
    tail_down = pnls[pnls <= q05]
    avg_tail_up = float(tail_up.mean()) if tail_up.size else float("nan")
    avg_tail_down_abs = float(abs(tail_down.mean())) if tail_down.size else float("nan")
    tail_ratio = (
        float(avg_tail_up / avg_tail_down_abs)
        if not np.isnan(avg_tail_up) and not np.isnan(avg_tail_down_abs) and avg_tail_down_abs > 0
        else float("nan")
    )

    # Max Drawdown על עקומת הון כרונולוגית
    if created_col and created_col in df.columns:
        eq = df.copy()
        eq["_ts"] = pd.to_datetime(eq[created_col], errors="coerce")
        eq = eq.dropna(subset=["_ts"]).sort_values("_ts")
    else:
        eq = df.copy()
        eq["_ts"] = pd.RangeIndex(len(eq))
        eq = eq.sort_values("_ts")

    equity_curve = eq[pnl_col].astype(float).cumsum().to_numpy()
    max_dd, _, _ = _max_drawdown(equity_curve)
    mar = float(total / max_dd) if max_dd > 0 else float("nan")

    return KPIs(
        trades=trades,
        total_pnl=total,
        avg_pnl=avg,
        median_pnl=med,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_per_trade=sharpe,
        skew=skew,
        kurt=kurt,
        max_drawdown=max_dd,
        mar_ratio=mar,
        best_trade=best,
        worst_trade=worst,
        tail_ratio=tail_ratio,
    )


def _kpi_block(
    df: pd.DataFrame,
    pair_col: str,
    pnl_col: str,
    mapping: Dict[str, str],
) -> Tuple[KPIs, pd.DataFrame]:
    """
    בלוק KPI ברמת קרן:
    - KPIs גלובליים על כל הטריידים.
    - סטטיסטיקות מתקדמות (Expectancy, Payoff, Gross P/L, quantiles, Tail-Ratio).
    - טבלת ביצועים לפי זוג (כולל win rate, expectancy, Sharpe per pair).
    - התפלגות טריידים ו-PnL מצטבר.
    """

    created_col = mapping.get("created")
    k = _compute_kpis(df, pnl_col, created_col)

    # --- בסיס: סדרת PnL נקייה ---
    pnl = df[pnl_col].replace([np.inf, -np.inf], np.nan).dropna()
    n_trades = int(len(pnl))
    if n_trades == 0:
        st.info("אין טריידים להצגה.")
        return k, pd.DataFrame()

    # wins / losses
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0  # שלילי
    avg_win = float(wins.mean()) if len(wins) else np.nan
    avg_loss = float(losses.mean()) if len(losses) else np.nan
    loss_abs = abs(avg_loss) if not np.isnan(avg_loss) else np.nan
    payoff = np.nan if np.isnan(avg_win) or np.isnan(loss_abs) or loss_abs == 0 else avg_win / loss_abs

    loss_rate = 1.0 - k.win_rate
    expectancy = (
        k.win_rate * avg_win + loss_rate * avg_loss
        if not np.isnan(avg_win) and not np.isnan(avg_loss)
        else np.nan
    )

    pnl_std = float(pnl.std(ddof=1)) if n_trades > 1 else 0.0
    q05 = float(pnl.quantile(0.05))
    q50 = float(pnl.quantile(0.50))
    q95 = float(pnl.quantile(0.95))
    cum_pnl = pnl.cumsum()

    # =========================
    # שורה 1 – KPIs ראשיים
    # =========================
    c1, c2, c3 = st.columns(3)
    c1.metric("עסקאות", f"{k.trades}")
    c2.metric("סה\"כ רווח", f"{k.total_pnl:,.2f}")
    c3.metric("Win Rate", f"{k.win_rate*100:.1f}%")

    c4, c5, c6 = st.columns(3)
    pf_str = "—" if np.isnan(k.profit_factor) else f"{k.profit_factor:.2f}"
    mar_str = "—" if np.isnan(k.mar_ratio) else f"{k.mar_ratio:.2f}"
    tail_str = "—" if np.isnan(k.tail_ratio) else f"{k.tail_ratio:.2f}"
    c4.metric("Profit Factor", pf_str)
    c5.metric("Sharpe (per trade)", f"{k.sharpe_per_trade:.2f}")
    c6.metric("Tail Ratio (Top 5% / Bottom 5%)", tail_str)

    # =========================
    # שורה 2 – Expectancy & Risk
    # =========================
    r1, r2, r3 = st.columns(3)
    r1.metric("PnL ממוצע / טרייד", f"{k.avg_pnl:.2f}")
    r2.metric("Max Drawdown (PnL)", f"{k.max_drawdown:,.2f}")
    r3.metric("סטיית תקן טריידים", f"{pnl_std:.2f}")

    r4, r5, r6 = st.columns(3)
    exp_str = "—" if np.isnan(expectancy) else f"{expectancy:.2f}"
    payoff_str = "—" if np.isnan(payoff) else f"{payoff:.2f}"
    mar_disp = "—" if np.isnan(k.mar_ratio) else f"{k.mar_ratio:.2f}"
    r4.metric("Expectancy / טרייד", exp_str)
    r5.metric("Payoff Ratio (Avg Win / Avg Loss)", payoff_str)
    r6.metric("MAR (PnL / Max DD)", mar_disp)

    # =========================
    # Moments & Quantiles
    # =========================
    with st.expander("Moments, Quantiles & Distribution", expanded=False):
        d1, d2, d3 = st.columns(3)
        d1.metric("Skew", f"{k.skew:.2f}")
        d2.metric("Kurtosis", f"{k.kurt:.2f}")
        d3.metric("Median PnL", f"{q50:.2f}")

        q1, q2, q3 = st.columns(3)
        q1.metric("5% Quantile (גרוע)", f"{q05:.2f}")
        q2.metric("95% Quantile (מצוין)", f"{q95:.2f}")
        q3.metric("Gross Profit / Loss", f"{gross_profit:,.0f} / {gross_loss:,.0f}")

        # התפלגות + PnL מצטבר (אופציונלי, לא כבד)
        dist_col, cum_col = st.columns(2)
        try:
            # Histogram גס של PnL
            hist_df = (
                pnl.to_frame("PnL")
                .assign(bin=lambda s: pd.cut(s["PnL"], bins=20))
                .groupby("bin", observed=False)["PnL"]
                .count()
                .reset_index()
            )
            hist_df["bin_center"] = hist_df["bin"].apply(
                lambda b: 0.5 * (b.left + b.right) if b is not None else np.nan
            )
            hist_plot = hist_df.set_index("bin_center")["PnL"]
            dist_col.markdown("**התפלגות טריידים (count per bin)**")
            dist_col.bar_chart(hist_plot)
        except Exception:
            dist_col.info("לא הצלחתי להציג היסטוגרמה (בעיה בדאטה או באינדקס).")

        try:
            cum_col.markdown("**PnL מצטבר לפי סדר טריידים**")
            cum_col.line_chart(cum_pnl.reset_index(drop=True))
        except Exception:
            cum_col.info("לא הצלחתי להציג PnL מצטבר.")

    # =========================
    # KPIs לפי זוג
    # =========================
    g = df.groupby(pair_col, observed=False)[pnl_col]
    stats = g.agg(["count", "mean", "std", "sum"]).reset_index()
    stats.rename(
        columns={
            "count": "עסקאות",
            "mean": "רווח ממוצע",
            "std": "סטיית תקן",
            "sum": 'סה"כ רווח',
        },
        inplace=True,
    )

    # Win rate ו-Expectancy לכל זוג
    try:
        def _per_pair_stats(sub: pd.Series) -> Dict[str, float]:
            s = sub.replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) == 0:
                return {"WinRate": np.nan, "Expectancy": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}

            wins_p = s[s > 0]
            losses_p = s[s < 0]
            win_rate_p = 0.0 if len(s) == 0 else len(wins_p) / len(s)
            loss_rate_p = 1.0 - win_rate_p

            avg_win_p = wins_p.mean() if len(wins_p) else np.nan
            avg_loss_p = losses_p.mean() if len(losses_p) else np.nan
            exp_p = (
                win_rate_p * avg_win_p + loss_rate_p * avg_loss_p
                if not np.isnan(avg_win_p) and not np.isnan(avg_loss_p)
                else np.nan
            )
            std_p = s.std(ddof=1) if len(s) > 1 else np.nan
            sharpe_p = (
                np.sqrt(len(s)) * s.mean() / std_p
                if std_p not in (0, np.nan) and not np.isnan(std_p)
                else np.nan
            )
            eq = s.cumsum().to_numpy()
            max_dd_p, _, _ = _max_drawdown(eq)

            return {
                "WinRate": win_rate_p * 100.0,
                "Expectancy": exp_p,
                "Sharpe": sharpe_p,
                "MaxDD": max_dd_p,
            }

        per_pair_extra = g.apply(_per_pair_stats).apply(pd.Series).reset_index(drop=True)
        stats = pd.concat([stats, per_pair_extra], axis=1)
        stats["WinRate"] = stats["WinRate"].round(1)
        stats["Expectancy"] = stats["Expectancy"].round(2)
        stats["Sharpe"] = stats["Sharpe"].round(2)
        stats["MaxDD"] = stats["MaxDD"].round(2)
    except Exception:
        # אם משהו נכשל – לא מפריע לשאר הטבלה
        pass

    st.markdown("#### 📊 ביצועים לפי זוג")
    st.dataframe(_arrow_friendly_df(stats), width="stretch")
    return k, stats

# ===== עזרי פילטרים =====
def _sidebar_filters(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    pair_col: str,
    pnl_col: str,
    *,
    global_start: Optional[pd.Timestamp] = None,
    global_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Sidebar Filters — ליבה של חיתוך הלוגים.
    כל פילטר מייצר mask משלו, ובסוף הכל משולב ל-final_mask אחד.

    שדרוגים מרכזיים:
    -----------------
    - פרופיל סינון (Conservative / Balanced / Aggressive).
    - Quick timeframe presets (All / 30 / 90 / 180 / 365 ימים).
    - Outcome filter: כל הטריידים / רק רווחיים / רק הפסדיים / קרובים לאפס.
    - k-sigma clipping + מידע כמה אחוז נשמר.
    - פילטר לפי משך עסקה, כיוון, קבוצה, Notional.
    - Min trades per pair (פילטר לזוגות עם מספיק דגימות).
    - Filter summary שנשמר ל-session_state לשימוש בדוחות.
    """

    st.sidebar.header("🧰 סינון מתקדם")
    st.sidebar.caption("התאם פילטרים, בדוק כמה דאטה נשאר, ורק אז נתח את התוצאות.")

    # === נתוני בסיס כלליים על הדאטה ===
    pnl_clean = df[pnl_col].replace([np.inf, -np.inf], np.nan).dropna()
    n_rows_total = int(len(df))
    n_trades_total = int(len(pnl_clean))
    q05 = float(pnl_clean.quantile(0.05)) if n_trades_total > 0 else 0.0
    q95 = float(pnl_clean.quantile(0.95)) if n_trades_total > 0 else 0.0

    # ---------- פרופיל סינון כללי ----------
    st.sidebar.markdown("### ⚙️ פרופיל סינון")
    profile = st.sidebar.radio(
        "Profile",
        options=["Conservative", "Balanced", "Aggressive"],
        index=1,
        horizontal=True,
        key="trade_filter_profile",
    )

    # Presets עבור k-sigma + quantiles
    if profile == "Conservative":
        default_k_sigma = 1.5
        default_pnl_q_low, default_pnl_q_high = 0.05, 0.95
    elif profile == "Aggressive":
        default_k_sigma = 0.0
        default_pnl_q_low, default_pnl_q_high = 0.00, 1.00
    else:  # Balanced
        default_k_sigma = 1.0
        default_pnl_q_low, default_pnl_q_high = 0.02, 0.98

    # ===============================
    # 1) Universe / Pairs
    # ===============================
    st.sidebar.markdown("### 🧭 יקום וזוגות")

    unique_pairs = sorted(df[pair_col].dropna().astype(str).unique().tolist())[:2000]
    default_n = min(20, len(unique_pairs))

    st.sidebar.caption(f"יש {len(unique_pairs):,} זוגות ייחודיים בדאטה (מוגבל ל-2000 בתצוגה).")
    sel_pairs = st.sidebar.multiselect(
        "בחר זוגות לניתוח",
        options=unique_pairs,
        default=unique_pairs[: default_n],
        key="filter_pairs",
    )
    mask_pair = df[pair_col].astype(str).isin(sel_pairs) if sel_pairs else pd.Series(True, index=df.index)

    # אפשרות להגדיר מינימום עסקאות לזוג
    st.sidebar.markdown("### 📉 מינימום עסקאות לזוג")
    min_trades_per_pair = st.sidebar.number_input(
        "מינ' עסקאות לזוג (לסינון רעש סטטיסטי)",
        min_value=1,
        max_value=1000,
        value=3,
        step=1,
        key="min_trades_pair",
    )
    if min_trades_per_pair > 1:
        pair_counts = df.groupby(pair_col).size()
        eligible_pairs = pair_counts[pair_counts >= min_trades_per_pair].index
        mask_min_trades = df[pair_col].isin(eligible_pairs)
    else:
        mask_min_trades = pd.Series(True, index=df.index)

    # ===============================
    # 2) PnL & Risk Filters
    # ===============================
    st.sidebar.markdown("### 💰 רווח/הפסד וחריגים")

    if n_trades_total > 0:
        min_pnl, max_pnl = float(pnl_clean.min()), float(pnl_clean.max())
    else:
        min_pnl, max_pnl = 0.0, 0.0

    pnl_profile = st.sidebar.selectbox(
        "פרופיל טווח PnL",
        options=[
            "כל הטווח",
            "ללא קצוות (5%-95%)",
            "ממוקד רווחים (0–95%)",
            "ממוקד הפסדים (5%–0)",
        ],
        key="pnl_profile_mode",
    )

    # נגזור טווח default לפי הפרופיל
    if pnl_profile == "כל הטווח":
        default_low, default_high = min_pnl, max_pnl
    elif pnl_profile == "ללא קצוות (5%-95%)":
        default_low, default_high = q05, q95
    elif pnl_profile == "ממוקד רווחים (0–95%)":
        default_low, default_high = min(0.0, min_pnl), q95
    else:  # "ממוקד הפסדים (5%–0)"
        default_low, default_high = q05, max(0.0, max_pnl)

    pnl_range = st.sidebar.slider(
        "טווח רווח/הפסד (PnL per trade)",
        min_value=float(min(min_pnl, default_low)),
        max_value=float(max(max_pnl, default_high)),
        value=(float(default_low), float(default_high)),
        key="pnl_range",
    )
    mask_pnl = (df[pnl_col] >= pnl_range[0]) & (df[pnl_col] <= pnl_range[1])

    # Outcome filter — Winners / Losers / Near-zero
    outcome_mode = st.sidebar.selectbox(
        "Outcome Filter",
        options=["כל הטריידים", "רק רווחיים", "רק הפסדיים", "קרובים לאפס"],
        key="pnl_outcome_mode",
    )

    if outcome_mode == "רק רווחיים":
        mask_outcome = df[pnl_col] > 0
    elif outcome_mode == "רק הפסדיים":
        mask_outcome = df[pnl_col] < 0
    elif outcome_mode == "קרובים לאפס":
        # סף "קרוב לאפס" יחסית לסטיית תקן; אם אין סטייה → סף קבוע קטן
        if n_trades_total > 1:
            s = float(pnl_clean.std(ddof=0))
            eps = s * 0.1 if s > 0 else 1e-6
        else:
            eps = 1e-6
        mask_outcome = df[pnl_col].between(-eps, eps)
    else:
        mask_outcome = pd.Series(True, index=df.index)

    # דילוג חריגים (k-sigma) – לפי הפרופיל
    k_sigma = st.sidebar.slider(
        "הסר חריגים (kσ סביב הממוצע)",
        min_value=0.0,
        max_value=5.0,
        value=float(default_k_sigma),
        step=0.5,
        key="pnl_k_sigma",
    )
    if k_sigma > 0 and n_trades_total > 1:
        m, s = pnl_clean.mean(), pnl_clean.std(ddof=0)
        if s > 0:
            mask_ks = (df[pnl_col] >= m - k_sigma * s) & (df[pnl_col] <= m + k_sigma * s)
            # מידע: כמה אחוז מהטריידים בתוך הטווח הזה
            frac_inside = float(((pnl_clean >= m - k_sigma * s) & (pnl_clean <= m + k_sigma * s)).mean())
            st.sidebar.write(f"• כ-{frac_inside*100:.1f}% מהטריידים בתוך ±{k_sigma}σ")
        else:
            mask_ks = pd.Series(True, index=df.index)
    else:
        mask_ks = pd.Series(True, index=df.index)

    # ===============================
    # 3) Time Window (created) + Quick timeframe presets
    # ===============================
    if "created" in mapping:
        st.sidebar.markdown("### ⏱ חלון זמן")
        col = mapping["created"]
        dates = pd.to_datetime(df[col], errors="coerce")
        dmin = dates.min()
        dmax = dates.max()

        if pd.notna(dmin) and pd.notna(dmax):
            if global_start is None:
                global_start = dmin
            if global_end is None:
                global_end = dmax

            st.sidebar.caption(
                f"טווח מלא בדאטה: {global_start.date()} → {global_end.date()}"
            )

            # Quick timeframe presets
            tf_mode = st.sidebar.selectbox(
                "Quick timeframe",
                options=["כל התקופה", "30 ימים אחרונים", "90 ימים אחרונים", "180 ימים אחרונים", "365 ימים אחרונים"],
                index=0,
                key="trade_tf_mode",
            )

            # שליטת המשתמש בתאריכים ספציפיים
            start, end = st.sidebar.date_input(
                "טווח תאריכים (כניסת טרייד)",
                value=(global_start.date(), global_end.date()),
                key="trade_date_range",
            )
            mask_date = (dates.dt.date >= start) & (dates.dt.date <= end)

            # זמן נוסף לפי quick timeframe
            if tf_mode != "כל התקופה":
                days_lookup = {
                    "30 ימים אחרונים": 30,
                    "90 ימים אחרונים": 90,
                    "180 ימים אחרונים": 180,
                    "365 ימים אחרונים": 365,
                }
                days_back = days_lookup.get(tf_mode, None)
                if days_back is not None:
                    tf_start_ts = dmax - pd.Timedelta(days=days_back)
                    mask_tf = dates >= tf_start_ts
                    mask_date = mask_date & mask_tf
        else:
            mask_date = pd.Series(True, index=df.index)
    else:
        mask_date = pd.Series(True, index=df.index)
        tf_mode = "n/a"
        start = end = None  # על מנת לא לשבור filter_summary

    # ===============================
    # 4) Trade properties (משך, כיוון, קבוצה, Notional)
    # ===============================
    st.sidebar.markdown("### 🎯 מאפייני עסקה")

    # משך עסקה (ימים)
    if "duration_days" in df.columns:
        dmin = int(max(1, float(df["duration_days"].min())))
        dmax = int(float(df["duration_days"].max()))
        if dmin > dmax:
            dmin, dmax = 1, max(dmin, dmax)
        dur_rng = st.sidebar.slider(
            "משך עסקה (ימים)",
            min_value=dmin,
            max_value=max(dmin, dmax),
            value=(dmin, max(dmin, dmax)),
            key="duration_range",
        )
        dur_arr = df["duration_days"].astype(float)
        mask_dur = (dur_arr >= dur_rng[0]) & (dur_arr <= dur_rng[1])
    else:
        mask_dur = pd.Series(True, index=df.index)
        dur_rng = None

    # כיוון עסקה (אם קיים)
    if "side" in mapping and mapping["side"] in df.columns:
        sides = sorted(df[mapping["side"]].dropna().astype(str).unique().tolist())
        if sides:
            sel_sides = st.sidebar.multiselect(
                "כיוון עסקה",
                options=sides,
                default=sides,
                key="filter_sides",
            )
            mask_side = df[mapping["side"]].astype(str).isin(sel_sides) if sel_sides else pd.Series(True, index=df.index)
        else:
            mask_side = pd.Series(True, index=df.index)
            sel_sides = []
    else:
        mask_side = pd.Series(True, index=df.index)
        sel_sides = []

    # קבוצה / סקטור (אם קיים)
    if "group" in mapping and mapping["group"] in df.columns:
        groups = sorted(df[mapping["group"]].dropna().astype(str).unique().tolist())
        if groups:
            sel_groups = st.sidebar.multiselect(
                "קבוצות / סקטורים",
                options=groups,
                default=groups,
                key="filter_groups",
            )
            mask_group = (
                df[mapping["group"]].astype(str).isin(sel_groups)
                if sel_groups
                else pd.Series(True, index=df.index)
            )
        else:
            mask_group = pd.Series(True, index=df.index)
            sel_groups = []
    else:
        mask_group = pd.Series(True, index=df.index)
        sel_groups = []

    # Notional (אם קיים)
    if "notional" in df.columns:
        st.sidebar.markdown("### 💼 גודל עסקה (Notional)")
        notional_clean = df["notional"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(notional_clean) > 0:
            n_min, n_max = float(notional_clean.min()), float(notional_clean.max())
            n_rng = st.sidebar.slider(
                "טווח Notional",
                min_value=float(n_min),
                max_value=float(n_max),
                value=(float(n_min), float(n_max)),
                key="notional_range",
            )
            mask_notional = (df["notional"] >= n_rng[0]) & (df["notional"] <= n_rng[1])
        else:
            mask_notional = pd.Series(True, index=df.index)
            n_rng = None
    else:
        mask_notional = pd.Series(True, index=df.index)
        n_rng = None

    # ===============================
    # 5) Mask סופי + סיכום
    # ===============================
    final_mask = (
        mask_pair
        & mask_min_trades
        & mask_pnl
        & mask_outcome
        & mask_ks
        & mask_date
        & mask_dur
        & mask_side
        & mask_group
        & mask_notional
    )

    dff = df[final_mask].copy()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 סיכום סינון")
    st.sidebar.write(f"• רשומות לפני סינון: **{n_rows_total:,}**")
    st.sidebar.write(f"• רשומות אחרי סינון: **{len(dff):,}**")
    st.sidebar.write(f"• טריידים עם PnL תקין לפני סינון: **{n_trades_total:,}**")

    # בניית Filter Summary ושמירתו ל-session_state (לטובת דוחות)
    try:
        filter_summary: Dict[str, Any] = {
            "profile": profile,
            "selected_pairs_count": len(sel_pairs) if sel_pairs else "ALL",
            "min_trades_per_pair": int(min_trades_per_pair),
            "pnl_profile": pnl_profile,
            "pnl_range": (float(pnl_range[0]), float(pnl_range[1])),
            "outcome_mode": outcome_mode,
            "k_sigma": float(k_sigma),
            "timeframe_mode": tf_mode if "tf_mode" in locals() else "n/a",
            "date_start": str(start) if "start" in locals() and start is not None else None,
            "date_end": str(end) if "end" in locals() and end is not None else None,
            "duration_range": dur_rng,
            "sides": sel_sides,
            "groups": sel_groups,
            "notional_range": n_rng,
            "rows_before": n_rows_total,
            "rows_after": int(len(dff)),
            "trades_before_pnl_clean": n_trades_total,
        }
        st.session_state["insights_filter_summary"] = filter_summary
    except Exception:
        # לא חובה, רק Nice-to-have
        pass

    return dff


# ===== עזר: שמירת דוח Markdown =====
def _save_markdown_report(
    stats: pd.DataFrame,
    dff: pd.DataFrame,
    pair_col: str,
    pnl_col: str,
    path: str = "logs/insights_report.md",
) -> str:
    """
    מייצר דוח Markdown:
    - טבלת KPIs לפי זוג.
    - Filter summary (אם קיים).
    - Head של הלוג המסונן (50 רשומות).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.StringIO()
    buf.write("# Insights Report\n\n")

    # Filter summary (אם קיים)
    fs = None
    try:
        fs = st.session_state.get("insights_filter_summary")
    except Exception:
        fs = None

    if isinstance(fs, dict) and fs:
        buf.write("## Filter Summary\n\n")
        try:
            fs_df = pd.DataFrame([fs])
            buf.write(_arrow_friendly_df(fs_df).to_markdown(index=False))
            buf.write("\n\n")
        except Exception:
            buf.write(str(fs) + "\n\n")

    buf.write("## KPIs by Pair\n\n")
    buf.write(_arrow_friendly_df(stats).to_markdown(index=False))

    buf.write("\n\n## Filtered Log (head)\n\n")
    buf.write(_arrow_friendly_df(dff.head(50)).to_markdown(index=False))

    with open(path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    return path


# ===== עזרי אנליזה מתקדמים =====
def _label_anomalies(df: pd.DataFrame, pnl_col: str, k: float = 3.0) -> pd.DataFrame:
    """
    מסמן חריגים לפי kσ סביב הממוצע:
    - מוסיף עמודה _zscore (Z-Score של ה-PnL).
    - מוסיף עמודה _anomaly (True/False).
    """
    d = df.copy()
    try:
        col = d[pnl_col].astype(float)
        m, s = col.mean(), col.std(ddof=0)
        if s and s > 0:
            z = (col - m) / s
            d["_zscore"] = z
            d["_anomaly"] = z.abs() > k
        else:
            d["_zscore"] = 0.0
            d["_anomaly"] = False
    except Exception:
        d["_zscore"] = 0.0
        d["_anomaly"] = False
    return d


def _duration_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    יוצרת עמודת duration_bin בקטגוריות:
        1-2, 3-6, 7-13, 14-29, 30-59, 60-89, 90-179, 180-364, 365+
    ומוודאת שזו Categorical ordered (לטובת plotting יפה).
    """
    d = df.copy()
    if "duration_days" not in d.columns:
        d["duration_days"] = 1
    try:
        bins = [1, 3, 7, 14, 30, 60, 90, 180, 365, np.inf]
        labels = ["1-2", "3-6", "7-13", "14-29", "30-59", "60-89", "90-179", "180-364", "365+"]
        d["duration_bin"] = pd.cut(d["duration_days"].astype(float), bins=bins, labels=labels, right=False)
        d["duration_bin"] = pd.Categorical(d["duration_bin"], categories=labels, ordered=True)
    except Exception:
        d["duration_bin"] = pd.Categorical(["unknown"] * len(d), categories=["unknown"], ordered=True)
    return d


def _make_excel_bytes(
    stats: pd.DataFrame,
    dff: pd.DataFrame,
    monthly: Optional[pd.DataFrame] = None,
    pair_month: Optional[pd.DataFrame] = None,
) -> Optional[bytes]:
    """
    Excel Export:
    - Sheet "KPIs_by_Pair"
    - Sheet "Filtered_Log"
    - Sheet "Monthly" (אם קיים)
    - Sheet "Pair_by_Month" (אם קיים)
    - Sheet "Filters" (אם יש filter_summary ב-session_state)
    """
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
            _arrow_friendly_df(stats).to_excel(wr, index=False, sheet_name="KPIs_by_Pair")
            _arrow_friendly_df(dff).to_excel(wr, index=False, sheet_name="Filtered_Log")
            if monthly is not None and not monthly.empty:
                _arrow_friendly_df(monthly).to_excel(wr, index=False, sheet_name="Monthly")
            if pair_month is not None and not pair_month.empty:
                _arrow_friendly_df(pair_month).to_excel(wr, sheet_name="Pair_by_Month")

            # Filter summary sheet
            try:
                fs = st.session_state.get("insights_filter_summary")
                if isinstance(fs, dict) and fs:
                    fs_df = _arrow_friendly_df(pd.DataFrame([fs]))
                    fs_df.to_excel(wr, index=False, sheet_name="Filters")
            except Exception:
                pass

        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None


# ===== עזר: בחירת תיקיית לוגים מתוך ctx / controls =====
def _resolve_logs_dir(
    default: str,
    ctx: Optional[Dict[str, Any]] = None,
    controls: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Resolve logs directory from:
    - controls['logs_dir' / 'logs_path' / 'logs_root']
    - ctx['logs_dir'] or ctx['paths']['logs_dir'/'logs_path']
    - אחרת: default

    בנוסף:
    - תומך בהרחבת ~ (home).
    - תומך ב-expansion של משתני סביבה (למשל $LOGS_DIR).
    """
    candidate: Optional[str] = None

    if controls:
        for key in ("logs_dir", "logs_path", "logs_root"):
            val = controls.get(key)
            if isinstance(val, str) and val:
                candidate = val
                break

    if candidate is None and ctx:
        if isinstance(ctx, dict):
            val = ctx.get("logs_dir")
            if isinstance(val, str) and val:
                candidate = val
            else:
                paths = ctx.get("paths")
                if isinstance(paths, dict):
                    for key in ("logs_dir", "logs_path"):
                        v = paths.get(key)
                        if isinstance(v, str) and v:
                            candidate = v
                            break

    if candidate is None:
        candidate = default

    # הרחבת ~ ומשתני סביבה
    try:
        candidate = os.path.expandvars(os.path.expanduser(candidate))
    except Exception:
        pass

    return candidate

# ===== טאב ראשי =====
def render_insights_tab(
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    *,
    shap_top: Optional[int] = None,      # לעתיד: top features ל-ML bridge
    max_pairs: Optional[int] = None,     # מגבלה על מספר זוגות לניתוח
    ctx: Optional[Dict[str, Any]] = None,
    **controls: Any,
) -> None:
    """
    טאב תובנות — גרסה תואמת dashboard (HF-grade).

    זרימת עבודה:
    -------------
    1. Resolve logs_dir + pattern מה-ctx / controls / UI.
    2. load_logs_dir → df_logs + mapping (pair/pnl/created/finished/side/group).
    3. פילטרים מתקדמים (sidebar) → dff (Filtered log).
    4. System Summary קטן על הדאטה אחרי פילטרים.
    5. KPIs גלובליים + KPIs per pair (כולל Tail-Ratio וכו').
    6. ויזואליזציות: Top/Bottom, Box, Heatmap, Monthly, Pair×Month, Histogram, Equity.
    7. פילוחים Side / Group, Anomalies, Duration bins, Benchmarks.
    8. Drill-down לזוג ספציפי.
    9. Export: CSV/Excel/Markdown.
    10. ML bridge, Fair Value Scanner, GPT insights, AI Agents.

    כל החלקים עטופים ב-try/except כדי שהטאב לא יקרוס על שגיאה נקודתית.
    """

    # ===== 0. Header & Date context =====
    st.header(HEB["title"])

    # פרשנות של start/end כתאריכי pandas (לשימוש ב-_sidebar_filters)
    global_start_ts: Optional[pd.Timestamp] = None
    global_end_ts: Optional[pd.Timestamp] = None
    try:
        if start_date is not None:
            global_start_ts = pd.to_datetime(start_date)
        if end_date is not None:
            global_end_ts = pd.to_datetime(end_date)
    except Exception:
        global_start_ts = global_end_ts = None

    # ===== 1. Data source controls (logs dir, pattern) =====
    default_logs_dir = _resolve_logs_dir("logs", ctx=ctx, controls=controls)
    with st.expander("מקור נתונים", expanded=False):
        colp1, colp2 = st.columns(2)
        with colp1:
            logs_dir = st.text_input(
                "תיקיית לוגים",
                value=st.session_state.get("ins_path", default_logs_dir),
                key="ins_path_main",
            )
        with colp2:
            pattern = st.text_input(
                "תבנית קובץ",
                value="_log.csv",
                help="סינון לפי סיומת / תבנית (למשל _log.csv)",
            )
        st.caption("שנה והרץ כדי לטעון מחדש את הלוגים.")

    # ===== 2. Load logs & canonicalize columns =====
    df_logs, mapping = load_logs_dir(logs_dir, pattern)
    if df_logs.empty:
        st.warning(HEB["no_logs"])
        return

    pair_col = mapping.get("pair") or "__PAIR_FILE__"
    pnl_col = mapping.get("pnl")
    if pnl_col is None:
        st.error(HEB["missing_pnl"])
        return

    # מגבלה על מספר זוגות לניתוח (שיפור ביצועים + UX)
    if max_pairs is not None and max_pairs > 0:
        try:
            top_pairs_for_slice = (
                df_logs.groupby(pair_col, observed=False)[pnl_col]
                .sum()
                .sort_values(ascending=False)
                .head(int(max_pairs))
                .index.tolist()
            )
            df_logs = df_logs[df_logs[pair_col].isin(top_pairs_for_slice)].copy()
        except Exception as e:
            st.caption(f"max_pairs slice skipped: {e}")

    # ===== 3. Advanced filters (sidebar) =====
    dff = _sidebar_filters(
        df_logs,
        mapping,
        pair_col,
        pnl_col,
        global_start=global_start_ts,
        global_end=global_end_ts,
    )
    if dff.empty:
        st.info("אחרי הפילטרים לא נשארו נתונים להצגה.")
        return

    # ===== 4. System Summary (metadata) =====
    with st.expander("📊 System Summary (Logs)", expanded=False):
        try:
            created_col = mapping.get("created")
            dt_min = dt_max = None
            if created_col and created_col in dff.columns:
                dt = pd.to_datetime(dff[created_col], errors="coerce")
                dt_min, dt_max = dt.min(), dt.max()
            summary_payload = {
                "rows_filtered": int(len(dff)),
                "unique_pairs": int(dff[pair_col].nunique()),
                "logs_dir": str(logs_dir),
                "pattern": pattern,
                "date_min": str(dt_min) if dt_min is not None else None,
                "date_max": str(dt_max) if dt_max is not None else None,
            }
            st.json(summary_payload)
        except Exception as e:
            st.caption(f"System summary skipped: {e}")

    # ===== 5. KPIs =====
    st.subheader(HEB["stats_header"])
    kpi_obj, stats = _kpi_block(dff, pair_col, pnl_col, mapping)

    # ===== 6. Top / Bottom 10 pairs =====
    st.subheader(HEB["top_title"])
    try:
        by_pair = (
            dff.groupby(pair_col, observed=False)[pnl_col]
            .sum()
            .sort_values(ascending=False)
        )
        top = by_pair.head(10)
        bottom = by_pair.tail(10)

        col_t, col_b = st.columns(2)
        with col_t:
            try:
                fig_top = px.bar(top.reset_index(), x=pair_col, y=pnl_col, title="Top 10 pairs by PnL")
                st.plotly_chart(fig_top, width="stretch")
            except Exception:
                st.info("Top-10 chart failed (data issue).")
        with col_b:
            try:
                fig_bottom = px.bar(bottom.reset_index(), x=pair_col, y=pnl_col, title="Bottom 10 pairs by PnL")
                st.plotly_chart(fig_bottom, width="stretch")
            except Exception:
                st.info("Bottom-10 chart failed (data issue).")
    except Exception as e:
        st.info(f"Top/Bottom section skipped: {e}")
        by_pair = dff.groupby(pair_col, observed=False)[pnl_col].sum()

    # ===== 7. ML Summary Bridge =====
    st.subheader("🤖 ML Summary (מערכת)")
    try:
        render_ml_bridge_panel()
    except Exception as e:
        st.caption(f"ML bridge unavailable: {e}")

    # ===== 8. Fair Value Scanner (Universe) =====
    st.subheader("⚖️ Fair Value Scanner — Universe")
    with st.expander("⚖️ Fair Value Engine — Universe Scanner", expanded=False):
        st.caption(
            "מריץ FairValueEngine על Universe של זוגות, על בסיס מחירי מעבר (wide prices CSV). "
            "מטרת הפאנל: לזהות זוגות mispriced / mean-reverting עם איכות גבוהה."
        )

        # 8.1 Prices wide CSV
        prices_file = st.file_uploader(
            "Prices wide CSV (תאריכים בשורה הראשונה, עמודות = סימבולים)",
            type=["csv"],
            key="fv_prices_csv",
        )

        prices_wide: Optional[pd.DataFrame]
        prices_wide = None
        if prices_file is not None:
            try:
                df_prices = pd.read_csv(prices_file)
                # זיהוי עמודת תאריך
                date_col = None
                for c in df_prices.columns:
                    if str(c).lower() in {"date", "datetime", "time"}:
                        date_col = c
                        break
                if date_col is None:
                    df_prices.iloc[:, 0] = pd.to_datetime(df_prices.iloc[:, 0], errors="coerce")
                    prices_wide = (
                        df_prices.set_index(df_prices.columns[0])
                        .select_dtypes(include=[float, int])
                        .astype(float)
                    )
                else:
                    df_prices[date_col] = pd.to_datetime(df_prices[date_col], errors="coerce")
                    prices_wide = (
                        df_prices.set_index(date_col)
                        .select_dtypes(include=[float, int])
                        .astype(float)
                    )

                st.caption(f"prices_wide shape = {prices_wide.shape[0]} rows × {prices_wide.shape[1]} symbols")
                st.dataframe(_arrow_friendly_df(prices_wide.tail(5)), width="stretch", height=200)
            except Exception as e:
                st.error(f"טעינת prices_wide נכשלה: {e}")
                prices_wide = None

        # 8.2 Pairs Y:X input
        pairs_raw = st.text_area(
            "Pairs (Y:X per line)",
            value="XLY:XLP\nQQQ:IWM",
            key="fv_pairs_raw",
            help="פורמט: Y:X בכל שורה (למשל XLY:XLP)",
        )

        pairs_list: List[Tuple[str, str]] = []
        for ln in pairs_raw.splitlines():
            ln = ln.strip()
            if not ln or ":" not in ln:
                continue
            y, x = ln.split(":", 1)
            y, x = y.strip(), x.strip()
            if y and x:
                pairs_list.append((y, x))

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            window = st.number_input("חלון עיקרי (window)", min_value=60, max_value=252 * 3, value=252, step=10)
            z_in = st.number_input("סף כניסה (z_in)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
            z_out = st.number_input("סף יציאה (z_out)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        with col_cfg2:
            target_vol = st.number_input("יעד תנודתיות שנתי לזוג (%)", min_value=2.0, max_value=30.0, value=10.0, step=0.5)
            kelly_frac = st.number_input("Kelly fraction", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
            max_lev = st.number_input("Max leverage", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

        run_fv = st.button("🚀 Run FairValueEngine on Universe", key="fv_run_universe")

        if run_fv:
            if prices_wide is None:
                st.error("חסר prices_wide – העלה קובץ מחירים רחב (CSV).")
            elif not pairs_list:
                st.error("לא הוגדר אף זוג (Y:X).")
            else:
                try:
                    cfg = Config()
                    cfg.window = int(window)
                    cfg.z_in = float(z_in)
                    cfg.z_out = float(z_out)
                    cfg.target_vol_ann = float(target_vol / 100.0)  # אחוז → fraction
                    cfg.kelly_fraction = float(kelly_frac)
                    cfg.max_leverage = float(max_lev)
                    cfg.pairs = pairs_list

                    eng = FairValueEngine(config=cfg)
                    with st.spinner("מריץ FairValueEngine על כל הזוגות..."):
                        res_fv = eng.run(prices_wide=prices_wide, pairs=pairs_list)

                    if res_fv is None or res_fv.empty:
                        st.info("FairValueEngine לא החזיר תוצאות (בדוק מחירים/זוגות).")
                    else:
                        # נתמקד ב-window העיקרי או ב-ensemble אם קיים
                        df_view = res_fv.copy()
                        if (df_view["window"] == -1).any():
                            df_view_main = df_view[df_view["window"] == -1].copy()
                        else:
                            df_view_main = df_view[df_view["window"] == cfg.window].copy()

                        st.caption(
                            f"Total rows: {len(df_view)} | Main view (window={cfg.window} / ensemble): "
                            f"{len(df_view_main)} rows."
                        )

                        # סינון בסיסי לפי z/edge/quality
                        col_f1, col_f2, col_f3 = st.columns(3)
                        with col_f1:
                            z_abs_min = st.number_input(
                                "מינימום |zscore|", min_value=0.0, max_value=10.0, value=1.0, step=0.1
                            )
                        with col_f2:
                            edge_min = st.number_input(
                                "מינימום net_edge_z", min_value=-5.0, max_value=5.0, value=-0.5, step=0.1
                            )
                        with col_f3:
                            q_min = st.number_input(
                                "מינימום quality_weight", min_value=0.0, max_value=1.0, value=0.0, step=0.05
                            )

                        df_filt = df_view_main.copy()
                        if "zscore" in df_filt.columns:
                            df_filt = df_filt[df_filt["zscore"].abs() >= z_abs_min]
                        if "net_edge_z" in df_filt.columns:
                            df_filt = df_filt[df_filt["net_edge_z"] >= edge_min]
                        if "quality_weight" in df_filt.columns:
                            df_filt = df_filt[df_filt["quality_weight"] >= q_min]

                        # מיון לפי ז/edge או SR_net אם יש
                        sort_options = []
                        for c in ("zscore", "vol_adj_mispricing", "net_edge_z", "sr_net", "psr_net", "dsr_net"):
                            if c in df_filt.columns:
                                sort_options.append(c)
                        sort_by = st.selectbox(
                            "מיין לפי",
                            options=sort_options or ["zscore"],
                            index=0,
                            key="fv_sort_by",
                        )
                        ascending = st.checkbox("מיון עולה", value=False, key="fv_sort_asc")
                        df_filt = df_filt.sort_values(sort_by, ascending=ascending)

                        st.dataframe(
                            _arrow_friendly_df(df_filt),
                            width="stretch",
                            height=400,
                        )

                        # הורדה
                        st.download_button(
                            "💾 הורד FairValueEngine universe results (CSV)",
                            data=df_filt.to_csv(index=False).encode("utf-8"),
                            file_name="fair_value_universe_scanner.csv",
                            mime="text/csv",
                            key="fv_universe_dl",
                        )

                except Exception as e:
                    st.error(f"FairValueEngine universe run failed: {e}")

    # ===== 9. Box Plot =====
    st.subheader(HEB["box_title"])
    try:
        fig_box = px.box(dff, x=pair_col, y=pnl_col, points="all")
        st.plotly_chart(fig_box, width="stretch")
    except Exception as e:
        st.info(f"Box plot skipped: {e}")

    # ===== 10. Heatmap רווח מצטבר =====
    st.subheader(HEB["heatmap_title"])
    try:
        pivot = dff.pivot_table(values=pnl_col, index=pair_col, aggfunc="sum").fillna(0)
        pivot["log"] = np.log(np.abs(pivot[pnl_col]) + 1)
        fig_hm = px.imshow(pivot[["log"]].T, color_continuous_scale="RdBu", aspect="auto", text_auto=True)
        st.plotly_chart(fig_hm, width="stretch")
    except Exception as e:
        st.info(f"Heatmap skipped: {e}")

    # ===== 11. רווח חודשי + Pair×Month =====
    created_col = mapping.get("created")
    monthly = pd.DataFrame()
    pair_month = pd.DataFrame()

    st.subheader(HEB["monthly_title"])
    try:
        if created_col is not None:
            tmp = dff.copy()
            tmp[created_col] = pd.to_datetime(tmp[created_col], errors="coerce")
            tmp = tmp.dropna(subset=[created_col])
            if not tmp.empty:
                tmp["חודש"] = tmp[created_col].dt.to_period("M").astype(str)
                monthly = tmp.groupby("חודש", observed=False)[pnl_col].sum().reset_index()
                fig_line = px.line(monthly, x="חודש", y=pnl_col)
                st.plotly_chart(fig_line, width="stretch")
            else:
                st.info("אין תאריכי כניסה תקינים לחישוב חודשי.")
        else:
            st.info("עמודת תאריך כניסה לא זוהתה — דילוג על גרף חודשי.")
    except Exception as e:
        st.info(f"Monthly aggregation skipped: {e}")

    st.subheader(HEB["pair_month_title"])
    try:
        if created_col is not None and not dff.empty:
            tmp2 = dff.copy()
            tmp2[created_col] = pd.to_datetime(tmp2[created_col], errors="coerce")
            tmp2 = tmp2.dropna(subset=[created_col])
            if not tmp2.empty:
                tmp2["חודש"] = tmp2[created_col].dt.to_period("M").astype(str)
                pm = (
                    tmp2.groupby(["חודש", pair_col], observed=False)[pnl_col]
                    .sum()
                    .unstack(fill_value=0)
                )
                pair_month = pm
                fig_pm = px.imshow(pm.T, aspect="auto", color_continuous_scale="Bluered")
                st.plotly_chart(fig_pm, width="stretch")
    except Exception as e:
        st.info(f"Pair×Month skipped: {e}")

    # ===== 12. התפלגות רווחים =====
    st.subheader(HEB["dist_title"])
    try:
        fig_hist = px.histogram(dff, x=pnl_col, nbins=50)
        st.plotly_chart(fig_hist, width="stretch")
    except Exception as e:
        st.info(f"Histogram skipped: {e}")

    # ===== 13. עקומת הון =====
    st.subheader(HEB["eq_title"])
    try:
        if created_col:
            eq = dff.copy()
            eq["_ts"] = pd.to_datetime(eq[created_col], errors="coerce")
        else:
            eq = dff.copy()
            eq["_ts"] = pd.RangeIndex(len(eq))
        eq = eq.sort_values("_ts")
        eq["cum"] = eq[pnl_col].astype(float).cumsum()
        fig_eq = px.line(eq, x="_ts", y="cum")
        st.plotly_chart(fig_eq, width="stretch")
    except Exception as e:
        st.info(f"Equity curve skipped: {e}")

    # ===== 14. פילוח לפי צד / קבוצה (Side / Group) =====
    with st.expander("🧩 פילוח לפי כיוון / קבוצה", expanded=False):
        try:
            if "side" in mapping and mapping["side"] in dff.columns:
                side_counts = (
                    dff.groupby(mapping["side"], observed=False)[pnl_col]
                    .agg(["count", "mean", "sum"])
                    .reset_index()
                )
                st.markdown("**Side breakdown**")
                st.dataframe(_arrow_friendly_df(side_counts), width="stretch")
                try:
                    fig_side = px.bar(side_counts, x=mapping["side"], y="sum", title="PnL by Side")
                    st.plotly_chart(fig_side, width="stretch")
                except Exception:
                    pass
            else:
                st.caption("No 'side' column found for breakdown.")

            if "group" in mapping and mapping["group"] in dff.columns:
                grp_stats = (
                    dff.groupby(mapping["group"], observed=False)[pnl_col]
                    .agg(["count", "mean", "sum"])
                    .reset_index()
                )
                st.markdown("**Group / Sector breakdown**")
                st.dataframe(_arrow_friendly_df(grp_stats), width="stretch")
                try:
                    fig_grp = px.bar(grp_stats, x=mapping["group"], y="sum", title="PnL by Group/Sector")
                    st.plotly_chart(fig_grp, width="stretch")
                except Exception:
                    pass
        except Exception as e:
            st.caption(f"Side/Group breakdown skipped: {e}")

    # ===== 15. אנומליות (חריגים) =====
    st.subheader("⚠️ חריגים (Anomalies)")
    try:
        k_anom = st.slider("סף kσ לחריג", min_value=1.0, max_value=6.0, value=3.0, step=0.5, key="ins_k_anom")
        d_anom = _label_anomalies(dff, pnl_col, k=k_anom)
        only_anom = st.checkbox("הצג רק חריגים", value=False, key="ins_only_anom")
        view = d_anom[d_anom["_anomaly"]] if only_anom else d_anom
        if view["_anomaly"].any():
            st.dataframe(_arrow_friendly_df(view[view["_anomaly"]]), width="stretch", height=220)
        else:
            st.info("לא נמצאו חריגים עבור הסף הנוכחי.")
    except Exception as e:
        st.info(f"Anomalies skipped: {e}")

    # ===== 16. פילוח לפי משך עסקה (Bins) =====
    st.subheader("⏱️ פילוח לפי משך (Bins)")
    try:
        d_bins = _duration_bins(dff)
        grp = (
            d_bins.groupby("duration_bin", observed=False)[pnl_col]
            .agg(["count", "mean", "sum"])
            .reset_index()
        )
        grp.columns = ["משך", "עסקאות", "רווח ממוצע", "סה\"כ רווח"]
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.dataframe(_arrow_friendly_df(grp), width="stretch")
        with col_b2:
            fig_bins = px.bar(grp, x="משך", y="רווח ממוצע")
            st.plotly_chart(fig_bins, width="stretch")
    except Exception as e:
        st.info(f"Duration bins skipped: {e}")

    # ===== 17. Benchmarks: Top-N מול הכל + MoM =====
    st.subheader("🏁 Benchmarks")
    try:
        topN = st.number_input("Top-N להשוואה", min_value=1, max_value=100, value=5, step=1, key="ins_topn")
        by_pair_all = (
            dff.groupby(pair_col, observed=False)[pnl_col]
            .sum()
            .sort_values(ascending=False)
        )
        sel = set(by_pair_all.head(int(topN)).index)
        if created_col:
            dd = dff.copy()
            dd["_ts"] = pd.to_datetime(dd[created_col], errors="coerce")
        else:
            dd = dff.copy()
            dd["_ts"] = pd.RangeIndex(len(dd))
        dd = dd.sort_values("_ts")
        dd["is_top"] = dd[pair_col].isin(sel)

        eq_all = dd[pnl_col].astype(float).cumsum()
        eq_top = dd.loc[dd["is_top"], pnl_col].astype(float).cumsum()

        fig_bm = go.Figure()
        fig_bm.add_trace(go.Scatter(x=dd["_ts"], y=eq_all, name="All", mode="lines"))
        fig_bm.add_trace(
            go.Scatter(
                x=dd.loc[dd["is_top"], "_ts"],
                y=eq_top,
                name=f"Top {int(topN)}",
                mode="lines",
            )
        )
        fig_bm.update_layout(title="Equity: All vs Top-N")
        st.plotly_chart(fig_bm, width="stretch")

        # MoM deltas (אם יש תאריך)
        if created_col:
            mon = dd.dropna(subset=["_ts"]).copy()
            mon["M"] = mon["_ts"].dt.to_period("M").astype(str)
            mom = mon.groupby("M", observed=False)[pnl_col].sum().reset_index()
            mom["pct_change"] = mom[pnl_col].pct_change()
            st.write("שינויים חודשיים (MoM):")
            st.dataframe(_arrow_friendly_df(mom.tail(12)), width="stretch")
    except Exception as e:
        st.info(f"Benchmarks skipped: {e}")

    # ===== 18. Drill-down לפי זוג =====
    st.subheader(HEB["drill_title"])
    try:
        pairs_sorted = by_pair.index.tolist()
        sel_pair = st.selectbox("בחר זוג", options=pairs_sorted[:500]) if pairs_sorted else None
        if sel_pair:
            dpp = dff[dff[pair_col] == sel_pair].copy()
            st.write(f"{sel_pair} — {len(dpp)} עסקאות")
            st.dataframe(_arrow_friendly_df(dpp), width="stretch", height=240)
            if created_col:
                dpp["_ts"] = pd.to_datetime(dpp[created_col], errors="coerce")
            else:
                dpp["_ts"] = pd.RangeIndex(len(dpp))
            dpp = dpp.sort_values("_ts")
            dpp["cum"] = dpp[pnl_col].astype(float).cumsum()
            fig_pp = px.line(dpp, x="_ts", y="cum", title=f"Equity — {sel_pair}")
            st.plotly_chart(fig_pp, width="stretch")
    except Exception as e:
        st.info(f"Drill-down skipped: {e}")

    # ===== 19. הורדות =====
    with st.expander("הורדות", expanded=False):
        st.download_button(
            "📥 הורד סיכומים (CSV)",
            data=stats.to_csv(index=False).encode("utf-8"),
            file_name="insights_summaries.csv",
        )
        st.download_button(
            "📥 הורד את כל הלוג המסונן (CSV)",
            data=dff.to_csv(index=False).encode("utf-8"),
            file_name="insights_filtered_logs.csv",
        )

        # יצוא Excel (multi-sheet)
        try:
            xbytes = _make_excel_bytes(stats, dff, monthly, pair_month)
            if xbytes:
                st.download_button(
                    "📊 Export Excel (KPIs, Logs, Monthly, Pair×Month)",
                    data=xbytes,
                    file_name="insights_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.caption("לא ניתן להפיק קובץ Excel (חסר מנוע כתיבה).")
        except Exception as e:
            st.caption(f"Excel export skipped: {e}")

        if st.button(HEB["save_md"], key="ins_save_md"):
            try:
                path_md = _save_markdown_report(stats, dff, pair_col, pnl_col)
                st.success(f"{HEB['saved_ok']} — {path_md}")
            except Exception as e:
                st.warning(f"Save report failed: {e}")

    # ===== 20. GPT ניתוח טקסטואלי (אופציונלי) =====
    st.subheader(HEB["gpt_title"])
    use_gpt = st.toggle("הפעל GPT (דורש OPENAI_API_KEY)", value=False)
    if use_gpt:
        # הגנה למקרה שעמודת 'סה\"כ רווח' לא קיימת (ניקח פשוט לפי sum של PnL)
        try:
            if 'סה"כ רווח' in stats.columns:
                key_col = 'סה"כ רווח'
            else:
                key_col = pnl_col
                if key_col not in stats.columns:
                    # fallback: ננסה לבנות sum לפי pair_col
                    stats_sum = dff.groupby(pair_col, observed=False)[pnl_col].sum().reset_index()
                    stats_sum.columns = [pair_col, key_col]
                    stats = stats.merge(stats_sum, on=pair_col, how="left")

            top_pairs = (
                stats.sort_values(key_col, ascending=False)[pair_col]
                .head(3)
                .astype(str)
                .tolist()
            )
        except Exception:
            top_pairs = dff.groupby(pair_col, observed=False)[pnl_col].sum().sort_values(ascending=False).head(3).index.tolist()

        prompt = f"""
את/ה אנליסט/ית קורלציות וזוגות. נתח/י את שלושת הזוגות המובילים בלוגים שלנו,
מצא/י תכונות משותפות, התנהגות סטטיסטית חוזרת, ומצבי שוק שמחזקים/מחלישים את האדג'.
הזוגות: {top_pairs}.
תן/י 5 תובנות קונקרטיות + 3 בדיקות אמפיריות עתידיות להוכחת/הפרכת ההשערות.
"""
        if st.button(HEB["gpt_btn"], key="insights_gpt_btn"):
            api_key = os.getenv("OPENAI_API_KEY")
            if OpenAI is None or not api_key:
                st.error(f"{HEB['gpt_error']}: חסר OpenAI SDK או מפתח.")
            else:
                try:
                    client = OpenAI(api_key=api_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a financial analyst specializing in pairs trading.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                    )
                    st.markdown(resp.choices[0].message.content)
                except Exception as e:
                    st.error(f"{HEB['gpt_error']}: {e}")

    # ===== 21. AI Agents: System Upgrader & Visualization =====
    st.subheader("🤖 AI Agents – System Upgrader & Visualization")

    # --- System Upgrader Agent ---
    with st.expander("🛠 System Upgrader Agent – הצעות לשדרוג הקוד", expanded=False):
        st.write(
            "הסוכן קורא לוגים וקבצי קוד ברפו, מאתר Patterns של בעיות מוכרות, "
            "ומחזיר דוח מפורט עם הצעות תיקון (עם אפשרות dry-run)."
        )

        repo_root_str = st.text_input(
            "נתיב ל-Repo Root",
            value=str(Path(".").resolve()),
            key="agent_repo_root",
        )
        log_path_str = st.text_input(
            "נתיב לקובץ לוג לניתוח (אופציונלי)",
            value="logs/dashboard_audit.log",
            key="agent_log_path",
        )
        dry_run = st.checkbox(
            "Dry Run בלבד (לא לשמור שינויים)",
            value=True,
            key="agent_dry_run",
        )

        if st.button("🚀 הרץ System Upgrader Agent", key="run_system_upgrader"):
            repo_root = Path(repo_root_str).resolve()
            log_path = Path(log_path_str).resolve() if log_path_str else None

            try:
                report: FixReport = run_upgrader(repo_root, log_path, dry_run=dry_run)
                st.success("הסוכן רץ בהצלחה. דוח למטה:")
                st.json(report.to_dict() if hasattr(report, "to_dict") else report.__dict__)
            except Exception as e:
                st.error(f"System Upgrader נכשל: {e}")

    # --- Visualization Agent ---
    with st.expander("📊 Visualization Agent – גרפים חכמים מהלוגים", expanded=False):
        st.write(
            "הסוכן מייצר גרפי Plotly משודרגים עבור זוגות / לוגים, "
            "כולל Theme כהה/בהיר, חיווי אנומליות, ותצוגת ספרד/מחירים."
        )

        log_csv_path = st.text_input(
            "CSV של לוג/דאטה (למשל logs/META-MSFT_log.csv)",
            value="",
            key="viz_agent_csv",
        )
        dark_mode = st.checkbox("Dark Mode", value=True, key="viz_agent_dark")

        if st.button("🎨 צור גרף משודרג", key="run_viz_agent") and log_csv_path:
            try:
                df_viz = pd.read_csv(log_csv_path)
                agent = VisualizationAgent(dark_mode=dark_mode)
                fig = agent.generate_enhanced_plot(df_viz, pair_name=log_csv_path)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"VisualizationAgent נכשל: {e}")


__all__ = ["render_insights_tab"]
