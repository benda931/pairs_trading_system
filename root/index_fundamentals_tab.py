# -*- coding: utf-8 -*-
"""
root/index_fundamentals_tab.py — Index Fundamentals & Spread Tab (HF-grade)
===========================================================================

טאב ניתוח פנדומנטלי למדדים/מניות, עם דגש על:
1. השוואה פנדומנטלית X מול Y:
   - ∆PE, ∆PB, ∆ROE, ∆Growth וכו' – במונחי אחוזים.
   - ניסוח מפורש של הפערים ("PE של X עלה 10%, של Y 2% → +8% לטובת X").

2. קישור בין שינויי פנדומנטל לבין שינויי הספרד במחיר:
   - Fundamental vs Spread Matrix:
       ∆Fundamental, ∆Spread, Corr, Beta, Same-direction flags.
   - גילוי alignment (השוק מיישר קו עם הפנדומנטל) ודיסוננס (השוק נגד).

3. מפעל רעיונות לזוגות (Pair Idea Factory):
   - סריקה אוטומטית של כל זוגות המדדים ב-Universe.
   - איתור זוגות בהם:
       * הפנדומנטל זז חזק, הספרד לא → פוטנציאל שהספרד יתיישר.
       * הספרד זז חזק, הפנדומנטל לא → פוטנציאל mean-reversion.

4. ניתוח עומק למדד בודד (Single Index Deep Dive):
   - history של Value/Quality/Growth/Composite.
   - multi-horizon percentiles (1Y/3Y/5Y/10Y).
   - פירוק תשואה (Earnings vs Re-rating vs Dividends).

5. Reports & Export:
   - דוחות Markdown למדד ו-Universe.
   - Export JSON/CSV ל-Universe ולרעיונות זוגות.

חיבור ל-App Context / טאבים אחרים
---------------------------------
הטאב מקבל פרמטר `app_ctx` (אופציונלי), ומשתלב איתו כך:
- אם `app_ctx` הוא dict:
    app_ctx["fundamentals"] = {
        "last_universe": [...],
        "last_universe_meta": {...},
        "last_pair_ideas": [...],
        "last_focus_symbol": "SPY",
    }
- טאבים אחרים (pairs/macro) יכולים לקרוא מה־app_ctx כדי:
    * להשתמש ב-last_pair_ideas בתור קלט אוטומטי.
    * לעבוד על אותו symbol שנבחר ב־Deep Dive.

אם אין app_ctx – הטאב עדיין עובד, ומשתמש רק ב-st.session_state.

20 רעיונות חדשים שמורחבים בתוך העיצוב (חלקם ממומשים, חלקם hooks):
-----------------------------------------------------------------
1. שמירת Universe אחרון ב-app_ctx לשימוש חוזר לטאבים אחרים.
2. שמירת רעיונות זוגות אחרונים ב-app_ctx["fundamentals"]["last_pair_ideas"].
3. סימון symbol בפוקוס (Deep Dive) ב-app_ctx["fundamentals"]["last_focus_symbol"].
4. הוספת "Profile name" ל-Universe כדי שטאבים אחרים ידעו איזה universe רץ.
5. אפשרות לתת לטאב הזוגות לקחת default זוג מתוך last_pair_ideas.
6. אפשרות למאקרו-tab לדעת איזה Universe רץ ולבנות עליו Regime view.
7. אפשרות לשמור snapshot של Universe (shape, snapshot_date) ב-app_ctx.
8. Export JSON ל-Universe ולמדד בודד – לחיבור ל-Agent חיצוני.
9. תמיכה ב-price loader גנרי (data_loader או utils) בלי תלות קשיחה.
10. Filter פנימי לספים (min_abs_rel_diff, min_abs_spread) קל לכוונון.
11. חישוב Fundamental lead ו-Spread lead כסקורים מספריים.
12. שימוש ב-corr/beta כדי להבין עד כמה הספרד "מונע" ע"י הפנדומנטל.
13. Summaries של momentum/stability לכל Universe.
14. אחידות לשפה: כל תצוגה מבוססת על אותם שדות Composite/Value/Quality/Growth.
15. Debug & Coverage View שמאפשר לבדוק למה משהו לא נכנס ל-Universe.
16. החזקת Pair Ideas גם ב-session_state כ-DataFrame מלא.
17. הכנת Hooks להמשך – שליחת events ל-app_ctx (למשל "open_pair_tab" בעתיד).
18. הגדרה פשוטה מצד dashboard – רק import + קריאת render_index_fundamentals_tab(ctx).
19. מבנה קובץ מסודר לפי חלקים לוגיים – Universe / Pair / Matrix / Deep Dive / Reports.
20. קוד נקי ללא כפילויות פונקציות, בלי הגדרת render_* יותר מפעם אחת.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import json
import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common.helpers import get_logger
from common.fundamental_loader import load_index_fundamentals

try:
    from common.data_loader import load_price_data as _load_price_data
except Exception:
    try:
        from common.utils import load_price_data as _load_price_data  # type: ignore
    except Exception:
        _load_price_data = None  # type: ignore

from core.index_fundamentals import (
    ScoreEngineConfig,
    UniverseScoresSummary,
    score_universe_fundamentals,
    get_universe_score_snapshot,
    select_top_n_by_score,
    select_bottom_n_by_score,
    bucket_scores_into_quantiles,
    summarize_universe_momentum,
    summarize_universe_stability,
    score_index_fundamentals_df,
    compute_fundamental_percentiles_for_symbol,
    decompose_return_into_earnings_and_rerating,
    export_index_scores_to_dict,
    export_universe_scores_to_dict,
    generate_index_markdown_report,
    generate_universe_markdown_report,
)

logger = get_logger("root.index_fundamentals_tab")


# ============================================================
# Config dataclasses
# ============================================================

@dataclass
class UniverseProfile:
    name: str
    description: str
    symbols: List[str]


@dataclass
class TabUIConfig:
    advanced_mode: bool = False
    show_momentum: bool = True
    show_stability: bool = True
    table_density: str = "normal"  # "compact" | "normal" | "spacious"
    top_n_default: int = 10
    bucket_quantiles_default: int = 5


@dataclass
class UniverseRunConfig:
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    snapshot_date_policy: str = "latest"  # "latest" | "common" | "custom:YYYY-MM-DD"
    min_rows_threshold: int = 4
    min_non_nan_ratio: float = 0.5
    debug_log_level: int = 0


@dataclass
class IndexFundamentalsTabConfig:
    universe_profiles: List[UniverseProfile] = field(default_factory=list)
    ui: TabUIConfig = field(default_factory=TabUIConfig)
    engine: ScoreEngineConfig = field(default_factory=ScoreEngineConfig)
    run: UniverseRunConfig = field(default_factory=UniverseRunConfig)


# ============================================================
# Defaults / Profiles
# ============================================================

def _build_default_universe_profiles() -> List[UniverseProfile]:
    profiles: List[UniverseProfile] = []

    profiles.append(
        UniverseProfile(
            name="Omri Default",
            description="מדדי ליבה + EM + סקטורים עיקריים",
            symbols=[
                "SPY",
                "QQQ",
                "IWM",
                "EEM",
                "EFA",
                "EWJ",
                "XLK",
                "XLF",
                "XLE",
                "VNQ",
            ],
        )
    )

    profiles.append(
        UniverseProfile(
            name="US Core",
            description="מדדי ליבה בארה\"ב",
            symbols=["SPY", "QQQ", "IWM", "DIA", "VTI"],
        )
    )

    profiles.append(
        UniverseProfile(
            name="Global Core",
            description="מדדי עולם (US + Developed + EM)",
            symbols=["SPY", "EFA", "EWJ", "EEM", "VT"],
        )
    )

    profiles.append(
        UniverseProfile(
            name="US Sectors",
            description="סקטורי S&P 500",
            symbols=["XLK", "XLF", "XLE", "XLY", "XLP", "XLV", "XLI", "XLB", "XLU", "XLRE"],
        )
    )

    return profiles


def _build_default_tab_config() -> IndexFundamentalsTabConfig:
    profiles = _build_default_universe_profiles()
    ui_cfg = TabUIConfig()
    engine_cfg = ScoreEngineConfig()
    run_cfg = UniverseRunConfig()
    return IndexFundamentalsTabConfig(
        universe_profiles=profiles,
        ui=ui_cfg,
        engine=engine_cfg,
        run=run_cfg,
    )


# ============================================================
# Session-state helpers
# ============================================================

_TAB_STATE_KEY = "index_fundamentals_tab_state"


def _get_tab_config_from_session() -> IndexFundamentalsTabConfig:
    if _TAB_STATE_KEY not in st.session_state:
        st.session_state[_TAB_STATE_KEY] = _build_default_tab_config()
    cfg = st.session_state[_TAB_STATE_KEY]
    if not isinstance(cfg, IndexFundamentalsTabConfig):
        cfg = _build_default_tab_config()
        st.session_state[_TAB_STATE_KEY] = cfg
    return cfg


def _save_tab_config_to_session(cfg: IndexFundamentalsTabConfig) -> None:
    st.session_state[_TAB_STATE_KEY] = cfg


# ============================================================
# Generic helpers
# ============================================================

def _ensure_datetime_index_local(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for cand in ("date", "period_end", "report_date", "as_of_date"):
        if cand in df.columns:
            df = df.copy()
            df[cand] = pd.to_datetime(df[cand])
            df = df.set_index(cand)
            return df.sort_index()
    return df


def _safe_series_local(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

def _safe_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not (v == v):  # NaN
            return None
        return v
    except Exception:
        return None

def push_fundamentals_metrics_to_ctx(
    *,
    valuation_score: Any,
    fund_score: Any,
    sharpe_60d: Any = None,
    max_dd_60d: Any = None,
    ctx_key: str = "fundamentals_metrics",
) -> None:
    """
    שומר מדדים פנדומנטליים רלוונטיים ל-Tab Comparison בתוך st.session_state[ctx_key].

    Sharpe / Max DD אופציונליים – אפשר להעביר None/לא להעביר בכלל.
    """
    metrics: Dict[str, float] = {}

    vs = _safe_float_or_none(valuation_score)
    if vs is not None:
        metrics["valuation_score"] = vs
        metrics["fundamental_score"] = vs
        metrics["value_score"] = vs

    fs = _safe_float_or_none(fund_score)
    if fs is not None:
        metrics["fund_score"] = fs
        metrics["tab_score"] = fs

    sh = _safe_float_or_none(sharpe_60d)
    if sh is not None:
        metrics["sharpe_60d"] = sh
        metrics["fund_sharpe_60d"] = sh

    dd = _safe_float_or_none(max_dd_60d)
    if dd is not None:
        metrics["max_dd_60d"] = dd
        metrics["fund_max_dd_60d"] = dd

    st.session_state[ctx_key] = metrics

def _load_price_history(
    symbol: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.Series:
    if _load_price_data is None:
        raise RuntimeError("לא נמצא load_price_data (common.data_loader/common.utils).")
    df = _load_price_data(symbol)
    if df is None or df.empty:
        return pd.Series(dtype=float)

    close_col = None
    for c in df.columns:
        if c.lower() == "close":
            close_col = c
            break
    if close_col is None:
        raise RuntimeError(f"לא מצאתי עמודת close ל-{symbol}")

    s = pd.to_numeric(df[close_col], errors="coerce")
    s.index = pd.to_datetime(df.index)
    s = s.dropna().sort_index()

    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index <= end]
    return s


def _align_two_series(
    s1: pd.Series,
    s2: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    if s1.empty or s2.empty:
        return s1.iloc[0:0], s2.iloc[0:0]
    idx = s1.index.intersection(s2.index)
    if idx.empty:
        return s1.iloc[0:0], s2.iloc[0:0]
    return s1.loc[idx], s2.loc[idx]

def _classify_field_theme(field: str) -> str:
    """
    מסווג field לקטגוריה לוגית: value / quality / growth / other.
    """
    f = field.lower()
    if any(tok in f for tok in ["pe", "pb", "earnings_yield", "fcf_yield", "dividend"]):
        return "value"
    if any(tok in f for tok in ["roe", "roa", "roic", "margin", "coverage"]):
        return "quality"
    if any(tok in f for tok in ["growth", "eps_growth", "revenue_growth"]):
        return "growth"
    return "other"

def _safe_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not (v == v):  # NaN
            return None
        return v
    except Exception:
        return None

def push_fundamentals_metrics_to_ctx(
    *,
    valuation_score: Any,
    fund_score: Any,
    sharpe_60d: Any = None,
    max_dd_60d: Any = None,
    ctx_key: str = "fundamentals_metrics",
) -> None:
    """
    שומר מדדים פנדומנטליים רלוונטיים ל-Tab Comparison בתוך st.session_state[ctx_key].

    משתדל לשמור כמה aliases כדי ש-Tab Comparison Lab יוכל למצוא אותם:
      - valuation_score / fundamental_score / value_score
      - fund_score / tab_score
      - sharpe_60d / fund_sharpe_60d
      - max_dd_60d / fund_max_dd_60d
    """
    metrics: Dict[str, float] = {}

    vs = _safe_float_or_none(valuation_score)
    if vs is not None:
        metrics["valuation_score"] = vs
        metrics["fundamental_score"] = vs
        metrics["value_score"] = vs

    fs = _safe_float_or_none(fund_score)
    if fs is not None:
        metrics["fund_score"] = fs
        metrics["tab_score"] = fs

    sh = _safe_float_or_none(sharpe_60d)
    if sh is not None:
        metrics["sharpe_60d"] = sh
        metrics["fund_sharpe_60d"] = sh

    dd = _safe_float_or_none(max_dd_60d)
    if dd is not None:
        metrics["max_dd_60d"] = dd
        metrics["fund_max_dd_60d"] = dd

    st.session_state[ctx_key] = metrics
# ============================================================
# Universe selector
# ============================================================

def _render_universe_selector(cfg: IndexFundamentalsTabConfig) -> List[str]:
    profiles = cfg.universe_profiles
    profile_names = [p.name for p in profiles]

    st.markdown("#### 🌍 בחירת Universe למדדים")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        selected_name = st.selectbox(
            "בחר פרופיל",
            options=profile_names,
            index=0,
        )

    selected_profile = next((p for p in profiles if p.name == selected_name), profiles[0])

    with col_right:
        st.caption(selected_profile.description)
        text = st.text_area(
            "ערוך/עדכן סימולים (מופרדים בפסיק)",
            value=", ".join(selected_profile.symbols),
            height=80,
        )

    universe = [s.strip().upper() for s in text.split(",") if s.strip()]
    return universe


# ============================================================
# Universe overview (scores, top/bottom, buckets)
# ============================================================

def _run_universe_scores(
    cfg: IndexFundamentalsTabConfig,
    universe: list[str],
) -> UniverseScoresSummary:
    summary = score_universe_fundamentals(
        universe,
        engine_cfg=cfg.engine,
        allow_remote=True,
        universe_min_rows_threshold=cfg.run.min_rows_threshold,
        universe_min_non_nan_ratio=cfg.run.min_non_nan_ratio,
        snapshot_date_policy=cfg.run.snapshot_date_policy,
        universe_debug_log=cfg.run.debug_log_level,
    )
    return summary


def _render_universe_overview_section(
    cfg: IndexFundamentalsTabConfig,
    universe: list[str],
) -> UniverseScoresSummary:
    st.markdown("#### 📊 Universe Overview – Value / Quality / Growth / Composite")

    with st.spinner("מריץ מנוע פנדומנטל לכל המדדים ב-Universe..."):
        summary = _run_universe_scores(cfg, universe)

    panel = summary.panel
    if panel is None or panel.empty:
        st.warning("לא התקבלו נתונים עבור ה-Universe. בדוק דאטה פנדומנטלי.")
        st.json(summary.coverage_info)
        return summary

    snapshot_df = get_universe_score_snapshot(
        summary,
        include_momentum=cfg.ui.show_momentum,
        include_stability=cfg.ui.show_stability,
    )

    # בחירת score לשימוש ב-Top/Bottom + מצב ניתוח מהיר
    score_cols = [c for c in ["composite_score", "value_score", "quality_score", "growth_score"] if c in snapshot_df.columns]
    if not score_cols:
        score_cols = ["composite_score"] if "composite_score" in snapshot_df.columns else []

    st.markdown("##### בחירת מצב ניתוח + score לרנקינג")

    # מצבי ניתוח מהירים → לעמודת score
    mode_to_col = {}
    if "composite_score" in score_cols:
        mode_to_col["Composite – משקול כללי"] = "composite_score"
    if "value_score" in score_cols:
        mode_to_col["Value – זול/יקר"] = "value_score"
    if "quality_score" in score_cols:
        mode_to_col["Quality – איכות מאזנים/רווח"] = "quality_score"
    if "growth_score" in score_cols:
        mode_to_col["Growth – צמיחה"] = "growth_score"

    col_mode, col_sel_score = st.columns([1.2, 1.8])

    # בחירת mode מהיר
    with col_mode:
        mode_label = st.radio(
            "מצב ניתוח מהיר",
            options=list(mode_to_col.keys()),
            index=0,
            key="universe_score_mode",
        )

    # מה mode אומר בפועל – איזו עמודת score להדגיש כברירת מחדל
    default_col = mode_to_col.get(mode_label)
    if default_col not in score_cols and score_cols:
        default_col = score_cols[0]

    with col_sel_score:
        selected_score_col = st.selectbox(
            "עמודת Score ל-Top/Bottom (ניתן לעקוף את ה-mode)",
            options=score_cols,
            index=score_cols.index(default_col) if default_col in score_cols else 0,
            key="universe_score_col",
        )


    st.success(f"Universe panel: {panel.shape[0]} רשומות, {panel.index.get_level_values('symbol').nunique()} סימולים.")

    # KPIs
    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.metric("מספר מדדים ב-Universe", snapshot_df.shape[0])
    with col_k2:
        if "composite_score" in snapshot_df.columns and not snapshot_df.empty:
            st.metric("Composite ממוצע", f"{snapshot_df['composite_score'].mean():.1f}")
    with col_k3:
        if "composite_score" in snapshot_df.columns and not snapshot_df.empty:
            best_sym = snapshot_df["composite_score"].idxmax()
            best_val = snapshot_df["composite_score"].max()
            st.metric("מדד מוביל", f"{best_sym} ({best_val:.1f})")

    st.markdown("##### טבלת Snapshot – ציונים לכל המדדים")
    st.dataframe(snapshot_df.round(2), use_container_width=True)
    # Heatmap בסיסי של Scores
    score_cols_for_heatmap = [c for c in ["value_score", "quality_score", "growth_score", "composite_score"] if c in snapshot_df.columns]
    if score_cols_for_heatmap:
        st.markdown("##### Heatmap – ערכי Scores לכל המדדים")
        df_hm = snapshot_df[score_cols_for_heatmap].copy()
        df_hm = df_hm.sort_values(selected_score_col, ascending=False)  # לפי score שנבחר
        fig_hm = px.imshow(
            df_hm.T,  # ציר Y = metric, X = symbol
            labels={"x": "Symbol", "y": "Score type", "color": "Score"},
            x=df_hm.index,
            y=df_hm.columns,
            aspect="auto",
        )
        st.plotly_chart(fig_hm, use_container_width=True)


    # Top / Bottom לפי ה-score שנבחר
    top_n = min(cfg.ui.top_n_default, max(1, snapshot_df.shape[0]))
    col_top, col_bottom = st.columns(2)

    with col_top:
        st.markdown(f"**🏆 Top N לפי {selected_score_col}**")
        top_df = select_top_n_by_score(
            snapshot_df,
            score_column=selected_score_col,
            n=top_n,
        )
        if not top_df.empty:
            st.dataframe(
                top_df[[selected_score_col]].round(2),
                use_container_width=True,
            )
        else:
            st.info("אין נתונים.")

    with col_bottom:
        st.markdown(f"**📉 Bottom N לפי {selected_score_col}**")
        bottom_df = select_bottom_n_by_score(
            snapshot_df,
            score_column=selected_score_col,
            n=top_n,
        )
        if not bottom_df.empty:
            st.dataframe(
                bottom_df[[selected_score_col]].round(2),
                use_container_width=True,
            )
        else:
            st.info("אין נתונים.")

    # Buckets distribution לפי ה-score שנבחר
    st.markdown(f"##### חלוקה ל-Buckets (Quintiles) לפי {selected_score_col}")
    buckets = bucket_scores_into_quantiles(
        snapshot_df,
        score_column=selected_score_col,
        quantiles=cfg.ui.bucket_quantiles_default,
        top_bucket_label="Top",
        bottom_bucket_label="Bottom",
    )
    if not buckets.empty:
        counts = buckets.value_counts().to_frame("count")
        st.dataframe(counts, use_container_width=True)
    else:
        st.info("לא ניתן לבנות Buckets (מעט מדי נתונים).")

    # Momentum / Stability summaries
    if cfg.ui.show_momentum:
        st.markdown("##### סיכום מומנטום ציונים ב-Universe (תאריך אחרון)")
        mom_summary = summarize_universe_momentum(panel)
        if not mom_summary.empty:
            st.dataframe(mom_summary.round(3), use_container_width=True)
        else:
            st.info("אין מספיק נתוני מומנטום.")

    if cfg.ui.show_stability:
        st.markdown("##### סיכום יציבות ציונים ב-Universe (תאריך אחרון)")
        stab_summary = summarize_universe_stability(panel)
        if not stab_summary.empty:
            st.dataframe(stab_summary.round(3), use_container_width=True)
        else:
            st.info("אין מספיק נתוני יציבות.")

    return summary


# ============================================================
# Pair fundamentals comparison (X vs Y)
# ============================================================

def _compute_pair_fundamental_diff(
    symbol_x: str,
    symbol_y: str,
    *,
    fields: List[str],
    lookback_days: int = 365,
) -> pd.DataFrame:
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=lookback_days)

    df_x = load_index_fundamentals(
        symbol_x,
        start=start_date.date(),
        end=end_date.date(),
        fields=fields,
        allow_remote=True,
        force_refresh=False,
        require_fresh_local=False,
        allow_partial=True,
    )
    df_y = load_index_fundamentals(
        symbol_y,
        start=start_date.date(),
        end=end_date.date(),
        fields=fields,
        allow_remote=True,
        force_refresh=False,
        require_fresh_local=False,
        allow_partial=True,
    )

    df_x = _ensure_datetime_index_local(df_x)
    df_y = _ensure_datetime_index_local(df_y)

    if not df_x.empty and not df_y.empty:
        idx = df_x.index.intersection(df_y.index)
        if not idx.empty:
            df_x = df_x.loc[idx]
            df_y = df_y.loc[idx]

    rows: List[Dict[str, Any]] = []
    for field in fields:
        sx = _safe_series_local(df_x, field).dropna()
        sy = _safe_series_local(df_y, field).dropna()
        if sx.empty or sy.empty:
            continue
        idx = sx.index.intersection(sy.index)
        if idx.empty:
            continue
        sx = sx.loc[idx]
        sy = sy.loc[idx]
        first_x, last_x = float(sx.iloc[0]), float(sx.iloc[-1])
        first_y, last_y = float(sy.iloc[0]), float(sy.iloc[-1])
        if first_x == 0 or first_y == 0:
            continue
        rel_x = last_x / first_x - 1.0
        rel_y = last_y / first_y - 1.0
        diff = rel_x - rel_y
        rows.append(
            {
                "field": field,
                "value_x_last": last_x,
                "value_y_last": last_y,
                "rel_change_x": rel_x,
                "rel_change_y": rel_y,
                "rel_change_diff": diff,
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("field")
    return out.sort_values("rel_change_diff", ascending=False)


def _render_pair_fundamental_comparison_section(
    universe: list[str],
) -> None:
    st.markdown("#### 🔗 השוואת פנדומנטל – X מול Y")

    col_pair, col_params = st.columns([2, 1])

    with col_pair:
        sym_x = st.selectbox("בחר X", options=universe, index=0, key="pair_x")
        sym_y = st.selectbox("בחר Y", options=universe, index=min(1, len(universe) - 1), key="pair_y")

    with col_params:
        lookback_days = st.number_input(
            "טווח בדיקה (ימים)",
            min_value=90,
            max_value=3650,
            value=365,
            step=30,
            key="pair_lookback",
        )
        default_fields = ["pe", "pb", "dividend_yield", "roe", "net_margin", "eps_growth_3y", "revenue_growth_3y"]
        fields_text = st.text_area(
            "שדות להשוואה (מופרדים בפסיק)",
            value=", ".join(default_fields),
            height=70,
            key="pair_fields",
        )
        fields = [f.strip().lower() for f in fields_text.split(",") if f.strip()]

    if st.button("🔍 הרץ השוואת פנדומנטל X מול Y", key="btn_pair_fundamentals"):
        with st.spinner("מחשב השוואת פנדומנטל לזוג..."):
            df_pair = _compute_pair_fundamental_diff(
                sym_x,
                sym_y,
                fields=fields,
                lookback_days=int(lookback_days),
            )

        if df_pair.empty:
            st.warning("לא הצלחנו להפיק השוואה – אין מספיק דאטה לזוג.")
            return

        st.markdown(
            f"**X = {sym_x.upper()}, Y = {sym_y.upper()}, טווח ≈ {int(lookback_days/365)} שנים**"
        )

        df_show = df_pair.copy()
        for col in ["rel_change_x", "rel_change_y", "rel_change_diff"]:
            df_show[col] = (df_show[col] * 100.0).round(2)
        df_show = df_show.rename(
            columns={
                "rel_change_x": f"∆% {sym_x.upper()}",
                "rel_change_y": f"∆% {sym_y.upper()}",
                "rel_change_diff": "פער ∆% (X - Y)",
            }
        )
        st.dataframe(df_show, use_container_width=True)

        st.markdown("##### 🗣 סיכום מילולי קצר")
        top_n = min(5, df_pair.shape[0])
        df_top = df_pair.sort_values("rel_change_diff", ascending=False).head(top_n)
        for field, row in df_top.iterrows():
            rx = row["rel_change_x"] * 100.0
            ry = row["rel_change_y"] * 100.0
            diff = row["rel_change_diff"] * 100.0
            direction = "יותר" if diff >= 0 else "פחות"
            st.write(
                f"- ב־**{field}**: {sym_x.upper()} עלה בכ־**{rx:.1f}%**, "
                f"{sym_y.upper()} עלה בכ־**{ry:.1f}%** → {sym_x.upper()} "
                f"השתפר ב־**{abs(diff):.1f}% {direction}** מ־{sym_y.upper()}."
            )

        # גרף bar של פערים פנדומנטליים (∆% X - Y)
        st.markdown("##### גרף ∆% (X - Y) לכל metric")
        df_bar = df_pair.copy()
        df_bar["rel_change_diff_pct"] = df_bar["rel_change_diff"] * 100.0
        df_bar_plot = df_bar.reset_index()[["field", "rel_change_diff_pct"]]
        fig_bar = px.bar(
            df_bar_plot,
            x="field",
            y="rel_change_diff_pct",
            title=f"פערי ∆% פנדומנטל לטובת {sym_x.upper()} מול {sym_y.upper()}",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================
# Fundamental vs Spread Matrix (∆Fundamentals vs ∆Spread)
# ============================================================

def _compute_fundamental_vs_spread_matrix(
    symbol_x: str,
    symbol_y: str,
    *,
    fields: List[str],
    lookback_days: int = 365,
) -> pd.DataFrame:
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=lookback_days)

    df_x = load_index_fundamentals(
        symbol_x,
        start=start_date.date(),
        end=end_date.date(),
        fields=fields,
        allow_remote=True,
        force_refresh=False,
        require_fresh_local=False,
        allow_partial=True,
    )
    df_y = load_index_fundamentals(
        symbol_y,
        start=start_date.date(),
        end=end_date.date(),
        fields=fields,
        allow_remote=True,
        force_refresh=False,
        require_fresh_local=False,
        allow_partial=True,
    )
    df_x = _ensure_datetime_index_local(df_x)
    df_y = _ensure_datetime_index_local(df_y)

    px_x = _load_price_history(symbol_x, start=start_date, end=end_date)
    px_y = _load_price_history(symbol_y, start=start_date, end=end_date)
    px_x, px_y = _align_two_series(px_x, px_y)
    if px_x.empty or px_y.empty:
        return pd.DataFrame()

    spread = np.log(px_x / px_y)
    spread_change = float(spread.iloc[-1] - spread.iloc[0])

    rows: List[Dict[str, Any]] = []

    for field in fields:
        sx = _safe_series_local(df_x, field).dropna()
        sy = _safe_series_local(df_y, field).dropna()
        if sx.empty or sy.empty:
            continue
        idx = sx.index.intersection(sy.index).intersection(spread.index)
        if idx.empty:
            continue

        sx = sx.loc[idx]
        sy = sy.loc[idx]
        sp = spread.loc[idx]

        diff_series = sx - sy
        diff_chg = diff_series.diff()
        spread_chg = sp.diff()

        # קורלציה ובטא
        corr = diff_chg.corr(spread_chg)

        beta = np.nan
        x = diff_chg.dropna()
        y = spread_chg.dropna()
        cidx = x.index.intersection(y.index)
        if not cidx.empty and x.loc[cidx].std(ddof=0) > 0:
            xc = x.loc[cidx] - x.loc[cidx].mean()
            yc = y.loc[cidx] - y.loc[cidx].mean()
            beta = float((xc * yc).mean() / (xc.var(ddof=0)))

        first_diff = float(diff_series.iloc[0])
        last_diff = float(diff_series.iloc[-1])
        if first_diff != 0:
            rel_diff = last_diff / first_diff - 1.0
        else:
            rel_diff = np.nan

        same_dir = None
        if not np.isnan(rel_diff):
            same_dir = np.sign(rel_diff) == np.sign(spread_change)

        rows.append(
            {
                "field": field,
                "rel_change_diff": rel_diff,
                "spread_change": spread_change,
                "corr_metric_vs_spread": corr,
                "beta_metric_vs_spread": beta,
                "same_direction": same_dir,
            }
        )

    if not rows:
        return pd.DataFrame()

    df_mat = pd.DataFrame(rows).set_index("field")

    # signal_score – מדד פשוט עד כמה הפנדומנטל והספרד לא מסכימים:
    # ערך גבוה = פער גדול + כיוון מנוגד.
    same = df_mat["same_direction"]
    rel_abs = df_mat["rel_change_diff"].abs()
    spread_abs = df_mat["spread_change"].abs()

    df_mat["dislocation_score"] = rel_abs * spread_abs * np.where(same == False, 1.0, 0.2)

    return df_mat



def _render_fundamental_vs_spread_section(
    universe: list[str],
) -> None:
    st.markdown("#### 📐 Fundamental vs Spread – ∆פנדומנטל מול ∆ספרד")

    col_pair, col_params = st.columns([2, 1])

    with col_pair:
        sym_x = st.selectbox("X (לספרד)", options=universe, index=0, key="spread_x")
        sym_y = st.selectbox("Y (לספרד)", options=universe, index=min(1, len(universe) - 1), key="spread_y")

    with col_params:
        lookback_days = st.number_input(
            "טווח (ימים)", 90, 3650, 365, 30, key="spread_lookback_days"
        )
        default_fields = ["pe", "pb", "dividend_yield", "roe", "net_margin", "eps_growth_3y"]
        fields_text = st.text_area(
            "שדות פנדומנטליים (מופרדים בפסיק)",
            value=", ".join(default_fields),
            height=70,
            key="spread_fields",
        )
        fields = [f.strip().lower() for f in fields_text.split(",") if f.strip()]

    if st.button("📊 הרץ Fundamental vs Spread Matrix", key="btn_fund_vs_spread"):
        with st.spinner("מחשב מטריצת ∆Fundamentals מול ∆Spread..."):
            try:
                matrix = _compute_fundamental_vs_spread_matrix(
                    sym_x,
                    sym_y,
                    fields=fields,
                    lookback_days=int(lookback_days),
                )
            except Exception as exc:
                logger.exception("Failed fund vs spread matrix: %s", exc)
                st.error(f"❗ שגיאה: {exc}")
                return

        if matrix.empty:
            st.warning("לא התקבלו נתונים – בדוק דאטה פנדומנטלי ומחיר.")
            return

        show = matrix.copy()
        if "rel_change_diff" in show.columns:
            show["rel_change_diff_pct"] = (show["rel_change_diff"] * 100.0).round(2)
        show["spread_change_pct"] = (show["spread_change"] * 100.0).round(2)

        st.markdown(
            f"**X = {sym_x.upper()}, Y = {sym_y.upper()}, טווח ≈ {int(lookback_days/365)} שנים**"
        )
        st.dataframe(show.round(4), use_container_width=True)

        # הדגשת דיסוננסים – השורות עם dislocation_score גבוה
        if "dislocation_score" in matrix.columns:
            st.markdown("##### דיסוננסים בולטים (פנדומנטל מול ספרד)")
            df_dis = matrix.copy()
            df_dis = df_dis.sort_values("dislocation_score", ascending=False).head(5)
            if not df_dis.empty:
                show_dis = df_dis[["rel_change_diff", "spread_change", "dislocation_score"]].copy()
                show_dis["rel_change_diff_pct"] = (show_dis["rel_change_diff"] * 100.0).round(2)
                show_dis["spread_change_pct"] = (show_dis["spread_change"] * 100.0).round(2)
                st.dataframe(show_dis.round(4), use_container_width=True)

        st.markdown("##### 🧭 פירוש:")
        st.write(
            "- rel_change_diff>0 + spread_change>0 → הפנדומנטל זז לטובת X והספרד נפתח לטובת X.\n"
            "- rel_change_diff>0 + spread_change<0 → פנדומנטל לטובת X אבל השוק זז לטובת Y (דיסוננס).\n"
            "- corr_metric_vs_spread גבוהה → לאורך התקופה השינויים בפנדומנטל ובספרד היו מתואמים.\n"
            "- beta_metric_vs_spread חיובית גבוהה → שינוי בפנדומנטל דחף את הספרד לאותו כיוון."
        )


# ============================================================
# Pair Idea Factory & Divergence Scanner
# ============================================================

def _generate_pair_universe(universe: list[str], max_pairs: int | None = None) -> list[tuple[str, str]]:
    pairs = list(itertools.combinations(sorted(set(universe)), 2))
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def _scan_pair_divergences_in_universe(
    universe: list[str],
    *,
    fields: List[str],
    lookback_days: int = 365,
    max_pairs: int | None = 100,
    min_abs_rel_diff: float = 0.05,   # 5% פער פנדומנטל מינימלי
    min_abs_spread: float = 0.0,
) -> pd.DataFrame:
    pairs = _generate_pair_universe(universe, max_pairs=max_pairs)
    rows: List[Dict[str, Any]] = []

    for (sym_x, sym_y) in pairs:
        try:
            matrix = _compute_fundamental_vs_spread_matrix(
                sym_x,
                sym_y,
                fields=fields,
                lookback_days=lookback_days,
            )
        except Exception as exc:
            logger.warning("pair scan failed for (%s,%s): %s", sym_x, sym_y, exc)
            continue

        if matrix is None or matrix.empty:
            continue

        for field, row in matrix.iterrows():
            rel_diff = float(row.get("rel_change_diff", np.nan))
            spread_chg = float(row.get("spread_change", np.nan))
            if np.isnan(rel_diff) or np.isnan(spread_chg):
                continue

            if abs(rel_diff) < min_abs_rel_diff and abs(spread_chg) < min_abs_spread:
                continue

            fd_lead = abs(rel_diff) / (1.0 + abs(spread_chg))
            sp_lead = abs(spread_chg) / (1.0 + abs(rel_diff))
            same_dir = row.get("same_direction", None)

            rows.append(
                {
                    "symbol_x": sym_x,
                    "symbol_y": sym_y,
                    "pair": f"{sym_x}/{sym_y}",
                    "field": field,
                    "rel_change_diff": rel_diff,
                    "spread_change": spread_chg,
                    "fundamental_lead_score": fd_lead,
                    "spread_lead_score": sp_lead,
                    "corr_metric_vs_spread": float(row.get("corr_metric_vs_spread", np.nan)),
                    "beta_metric_vs_spread": float(row.get("beta_metric_vs_spread", np.nan)),
                    "same_direction": same_dir,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["rel_change_diff_pct"] = df["rel_change_diff"] * 100.0
    df["spread_change_pct"] = df["spread_change"] * 100.0
    df["rel_change_diff_z"] = (df["rel_change_diff"] - df["rel_change_diff"].mean()) / (df["rel_change_diff"].std(ddof=0) or 1e-9)
    return df


def _render_pair_idea_factory_section(
    universe: list[str],
    app_ctx: Optional[Any] = None,
) -> None:
    st.markdown("#### 🧠 מפעל רעיונות לזוגות – Fundamental vs Spread Divergences")

    if len(universe) < 2:
        st.info("צריך לפחות שני סימולים ב-Universe בשביל לבנות רעיונות לזוגות.")
        return

    col_cfg1, col_cfg2 = st.columns([2, 1])

    with col_cfg1:
        fields_default = ["pe", "pb", "dividend_yield", "roe", "eps_growth_3y"]
        fields_text = st.text_area(
            "שדות פנדומנטליים לסריקה",
            value=", ".join(fields_default),
            height=70,
            key="idea_fields",
        )
        fields = [f.strip().lower() for f in fields_text.split(",") if f.strip()]

        lookback_days = st.number_input(
            "טווח בדיקה (ימים)",
            min_value=90,
            max_value=3650,
            value=365,
            step=30,
            key="idea_lookback",
        )

    with col_cfg2:
        idea_mode = st.selectbox(
            "מצב רעיונות",
            options=[
                "Fundamental leads (פנדומנטל זז, ספרד לא)",
                "Spread leads (ספרד זז, פנדומנטל לא)",
            ],
            key="idea_mode",
        )

        idea_theme = st.radio(
            "Theme פנדומנטלי",
            options=["All", "Value", "Quality", "Growth"],
            index=0,
            key="idea_theme",
            help="סנן רעיונות לפי סוג פנדומנטל: Value / Quality / Growth.",
        )

        top_n = st.number_input(
            "כמה רעיונות להציג",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="idea_top_n",
        )
        min_rel_pct = st.number_input(
            "מינימום ∆% פנדומנטל (abs)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
            key="idea_min_rel_pct",
        )
        min_spread_pct = st.number_input(
            "מינימום ∆% Spread (abs) (למצב Spread leads)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key="idea_min_spread_pct",
        )

    if st.button("🔎 סרוק רעיונות לזוגות בכל ה-Universe", key="btn_scan_pairs"):
        with st.spinner("סורק זוגות... זה יכול לקחת קצת זמן ב-Universe גדול."):
            df_pairs = _scan_pair_divergences_in_universe(
                universe,
                fields=fields,
                lookback_days=int(lookback_days),
                max_pairs=200,
                min_abs_rel_diff=min_rel_pct / 100.0,
                min_abs_spread=min_spread_pct / 100.0 if "Spread leads" in idea_mode else 0.0,
            )

        if df_pairs.empty:
            st.warning("לא נמצאו זוגות מעניינים לפי הספים שנבחרו.")
            return

        # שמירה ל-session / app_ctx לטאבים אחרים
        st.session_state["last_pair_ideas_df"] = df_pairs.copy()

        if isinstance(app_ctx, dict):
            app_ctx.setdefault("fundamentals", {})
            app_ctx["fundamentals"]["last_pair_ideas"] = df_pairs.head(200).to_dict(orient="records")
            app_ctx["fundamentals"]["last_pair_ideas_meta"] = {
                "universe_size": len(universe),
                "pairs_scanned": int(df_pairs["pair"].nunique()),
                "fields_used": fields,
                "lookback_days": int(lookback_days),
                "idea_mode": idea_mode,
            }
        # סינון לפי Theme
        if idea_theme != "All":
            theme_map = df_pairs["field"].map(_classify_field_theme)
            if idea_theme == "Value":
                df_pairs = df_pairs[theme_map == "value"]
            elif idea_theme == "Quality":
                df_pairs = df_pairs[theme_map == "quality"]
            elif idea_theme == "Growth":
                df_pairs = df_pairs[theme_map == "growth"]

            if df_pairs.empty:
                st.warning(f"לא נמצאו רעיונות מתאימים ל-Theme {idea_theme}. נסה לשנות סף או Theme.")
                return

        # בחירת score לפי מצב
        if idea_mode.startswith("Fundamental"):
            score_col = "fundamental_lead_score"
            sort_asc = False
            st.markdown("מציג זוגות בהם ∆Fundamental גדול יחסית ו-∆Spread קטן – השוק עוד לא יישר קו.")
        else:
            score_col = "spread_lead_score"
            sort_asc = False
            st.markdown("מציג זוגות בהם ∆Spread גדול יחסית ו-∆Fundamental קטן – אולי הספרד הגזים.")

        df_show = df_pairs.sort_values(score_col, ascending=sort_asc).head(int(top_n))

        # Heatmap – כמה רעיונות לכל field וכיוון
        st.markdown("##### התפלגות רעיונות לפי field וכיוון (אותו כיוון / מנוגד)")
        df_hm = df_pairs.copy()
        df_hm["direction"] = np.where(df_hm["same_direction"] == True, "Same", np.where(df_hm["same_direction"] == False, "Opposite", "Unknown"))
        counts = df_hm.groupby(["field", "direction"]).size().reset_index(name="count")
        if not counts.empty:
            fig_hm = px.density_heatmap(
                counts,
                x="field",
                y="direction",
                z="count",
                color_continuous_scale="Blues",
                title="התפלגות רעיונות לפי field וכיוון",
            )
            st.plotly_chart(fig_hm, use_container_width=True)

        cols = [
            "pair",
            "field",
            "rel_change_diff_pct",
            "spread_change_pct",
            "fundamental_lead_score",
            "spread_lead_score",
            "corr_metric_vs_spread",
            "beta_metric_vs_spread",
            "same_direction",
        ]
        cols = [c for c in cols if c in df_show.columns]

        st.dataframe(df_show[cols].round(3), use_container_width=True)

        st.markdown("##### 🗣 פירוש קצר לכל רעיון")
        for _, row in df_show.iterrows():
            pair = row["pair"]
            field = row["field"]
            relp = row["rel_change_diff_pct"]
            sprp = row["spread_change_pct"]
            same = row["same_direction"]
            mode_str = "פנדומנטל" if idea_mode.startswith("Fundamental") else "ספרד"

            s_dir = ""
            if same is True:
                s_dir = " (אותו כיוון)"
            elif same is False:
                s_dir = " (כיוונים מנוגדים)"

            st.write(
                f"- **{pair} – {field}**: ∆פנדומנטל ≈ {relp:.1f}%, ∆Spread ≈ {sprp:.1f}% "
                f"[{mode_str}]{s_dir}"
            )


# ============================================================
# Single Index Deep Dive
# ============================================================

def _pick_single_index_symbol(universe: list[str]) -> str:
    col1, col2 = st.columns([2, 1])
    with col1:
        sym = st.selectbox(
            "בחר מדד/מניה ל-Dive",
            options=universe,
            index=0,
            key="deep_symbol",
        )
    with col2:
        sym_manual = st.text_input(
            "או הזן סימול ידנית",
            value="",
            key="deep_symbol_manual",
        ).strip().upper()
        if sym_manual:
            sym = sym_manual
    return sym


def _load_index_scores_and_price(
    symbol: str,
    cfg: IndexFundamentalsTabConfig,
    lookback_days: int,
) -> tuple[pd.DataFrame, pd.Series]:
    end_ts = pd.Timestamp.today().normalize()
    start_ts = end_ts - pd.Timedelta(days=lookback_days)

    df_scores = score_index_fundamentals_df(
        symbol,
        start=start_ts.date(),
        end=end_ts.date(),
        engine_cfg=cfg.engine,
        allow_remote=True,
    )
    df_scores = _ensure_datetime_index_local(df_scores)

    px = pd.Series(dtype=float)
    if _load_price_data is not None:
        try:
            px = _load_price_history(symbol, start=start_ts, end=end_ts)
        except Exception as exc:
            logger.warning("deep dive price load failed for %s: %s", symbol, exc)

    return df_scores, px


def _render_single_index_scores_chart(
    symbol: str,
    df_scores: pd.DataFrame,
) -> None:
    if df_scores is None or df_scores.empty:
        st.info("אין מספיק נתונים להצגת היסטוריית scores.")
        return

    st.markdown("##### גרף היסטורי – Value / Quality / Growth / Composite")

    cols = [c for c in ["value_score", "quality_score", "growth_score", "composite_score"] if c in df_scores.columns]
    if not cols:
        st.info("אין עמודות score מתאימות.")
        return

    df_plot = df_scores[cols].copy().sort_index()

    fig = px.line(
        df_plot,
        x=df_plot.index,
        y=cols,
        labels={"value": "Score", "index": "Date"},
        title=f"Scores History – {symbol}",
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)


def _render_single_index_percentiles(symbol: str) -> None:
    st.markdown("##### Percentiles רב־אופקיים (1Y/3Y/5Y/10Y)")

    default_fields = ["pe", "pb", "dividend_yield", "roe", "net_margin"]
    fields_text = st.text_area(
        "מטריקות לניתוח אחוזונים",
        value=", ".join(default_fields),
        height=70,
        key="deep_fields_pct",
    )
    fields = [f.strip().lower() for f in fields_text.split(",") if f.strip()]

    if st.button("📈 חשב אחוזונים רב־אופקיים", key="btn_deep_pct"):
        with st.spinner("מחשב אחוזונים..."):
            pct_df = compute_fundamental_percentiles_for_symbol(
                symbol,
                fields=fields,
                horizons_days={"1y": 365, "3y": 365 * 3, "5y": 365 * 5, "10y": 365 * 10},
                allow_remote=True,
            )
        if pct_df.empty:
            st.warning("לא נמצאו מספיק נתונים לחישוב אחוזונים.")
            return
        st.dataframe(pct_df.round(1), use_container_width=True)


def _render_single_index_price_vs_metric(
    symbol: str,
    df_scores: pd.DataFrame,
    metric_field: str,
) -> None:
    st.markdown("##### השוואת metric פנדומנטלי מול Composite Score")

    if metric_field not in df_scores.columns:
        st.info(f"לא נמצא השדה {metric_field} בנתונים.")
        return

    df_plot = df_scores[[metric_field, "composite_score"]].copy().dropna(how="all")
    if df_plot.empty:
        st.info("אין מספיק נתונים להשוואה.")
        return

    df_long = df_plot.reset_index().melt(
        id_vars=["index"],
        value_vars=[metric_field, "composite_score"],
        var_name="series",
        value_name="value",
    )
    df_long = df_long.rename(columns={"index": "date"})

    fig = px.line(
        df_long,
        x="date",
        y="value",
        color="series",
        title=f"{symbol} – {metric_field} vs Composite Score",
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)


def _render_single_index_return_decomposition(
    symbol: str,
    bench_symbol: Optional[str],
    lookback_days: int,
) -> None:
    st.markdown("##### פירוק תשואה מקורב (Earnings vs Rerating vs Dividends)")

    end_ts = pd.Timestamp.today().normalize()
    start_ts = end_ts - pd.Timedelta(days=lookback_days)

    try:
        px_sym = _load_price_history(symbol, start=start_ts, end=end_ts)
    except Exception:
        px_sym = pd.Series(dtype=float)

    df_fund = load_index_fundamentals(
        symbol,
        start=start_ts.date(),
        end=end_ts.date(),
        fields=["eps", "eps_ttm", "dividends", "dividend"],
        allow_remote=True,
        force_refresh=False,
        require_fresh_local=False,
        allow_partial=True,
    )
    df_fund = _ensure_datetime_index_local(df_fund)

    eps_col = None
    for c in ["eps", "eps_ttm"]:
        if c in df_fund.columns:
            eps_col = c
            break

    if eps_col is None or px_sym.empty:
        st.info("אין מספיק נתוני EPS או מחיר כדי לבצע פירוק תשואה.")
        return

    eps = _safe_series_local(df_fund, eps_col).dropna()

    div = None
    for c in ["dividends", "dividend"]:
        if c in df_fund.columns:
            div = _safe_series_local(df_fund, c)
            break

    deco = decompose_return_into_earnings_and_rerating(
        price=px_sym,
        eps=eps,
        dividends=div,
    )

    st.write(f"**מדד:** {symbol} | טווח ≈ {int(lookback_days/365)} שנים")

    df_deco = pd.DataFrame(
        {
            "component": [
                "Total return",
                "Earnings contribution",
                "Re-rating contribution",
                "Dividend contribution",
            ],
            "value": [
                deco.get("total_return", np.nan),
                deco.get("earnings_contribution", np.nan),
                deco.get("rerating_contribution", np.nan),
                deco.get("dividend_contribution", np.nan),
            ],
        }
    )
    df_deco["value_pct"] = (df_deco["value"] * 100.0).round(2)
    st.dataframe(df_deco, use_container_width=True)

    if bench_symbol:
        st.caption(f"(Hook) אפשר בהמשך להשוות decomposition מול Benchmark {bench_symbol}.")


def _render_single_index_text_summary(
    symbol: str,
    df_scores: pd.DataFrame,
) -> None:
    if df_scores is None or df_scores.empty:
        return

    last_row = df_scores.sort_index().iloc[-1]
    vs = float(last_row.get("value_score", np.nan))
    qs = float(last_row.get("quality_score", np.nan))
    gs = float(last_row.get("growth_score", np.nan))
    cs = float(last_row.get("composite_score", np.nan))

    st.markdown("##### 🗣 סיכום מילולי על המדד")

    def bucket(score: float) -> str:
        if np.isnan(score):
            return "לא מוגדר"
        if score >= 80:
            return "גבוה מאוד"
        if score >= 60:
            return "גבוה"
        if score >= 40:
            return "בינוני"
        if score >= 20:
            return "נמוך"
        return "נמוך מאוד"

    st.write(
        f"- **Value:** {vs:.1f} – {bucket(vs)}.\n"
        f"- **Quality:** {qs:.1f} – {bucket(qs)}.\n"
        f"- **Growth:** {gs:.1f} – {bucket(gs)}.\n"
        f"- **Composite:** {cs:.1f} – {bucket(cs)}.\n"
    )
    st.write(
        "באופן כללי: Composite גבוה עם Value גבוה ו-Quality סביר → מועמד ל-Overweight.\n"
        "Composite נמוך עם Value נמוך ו-Quality חלש → מועמד ל-Underweight או הימנעות."
    )


def _render_single_index_deep_dive_section(
    cfg: IndexFundamentalsTabConfig,
    universe: list[str],
    app_ctx: Optional[Any] = None,
) -> None:
    st.markdown("### 🔎 ניתוח עומק – מדד בודד")

    if not universe:
        st.info("אין Universe מוגדר.")
        return

    col_sel, col_cfg = st.columns([2, 1])

    with col_sel:
        symbol = _pick_single_index_symbol(universe)

    with col_cfg:
        lookback = st.slider(
            "טווח היסטוריה (ימים)",
            min_value=90,
            max_value=3650,
            value=730,
            step=90,
            key="deep_lookback",
        )
        metric_field = st.text_input(
            "metric להשוואה מול Composite (למשל pe / pb / roe)",
            value="pe",
            key="deep_metric",
        ).strip().lower()
        bench_symbol = st.text_input(
            "Benchmark לפירוק תשואה (אופציונלי)",
            value="",
            key="deep_bench",
        ).strip().upper() or None

    if st.button("🔬 הרץ ניתוח עומק למדד", key="btn_deep_dive"):
        with st.spinner("טוען scores + מחיר + fundamentals..."):
            df_scores, px = _load_index_scores_and_price(symbol, cfg, lookback)

        if df_scores.empty:
            st.warning("לא נמצאו scores למדד. בדוק דאטה פנדומנטלי.")
            return

        # שמירת focus symbol ב-app_ctx
        if isinstance(app_ctx, dict):
            app_ctx.setdefault("fundamentals", {})
            app_ctx["fundamentals"]["last_focus_symbol"] = symbol

        st.markdown(f"#### {symbol} – Scores & Fundamentals")

        _render_single_index_scores_chart(symbol, df_scores)

        st.markdown("---")
        _render_single_index_percentiles(symbol)

        st.markdown("---")
        _render_single_index_price_vs_metric(symbol, df_scores, metric_field)

        st.markdown("---")
        _render_single_index_return_decomposition(symbol, bench_symbol, lookback)

        st.markdown("---")
        _render_single_index_text_summary(symbol, df_scores)


# ============================================================
# Reports / Export / Coverage
# ============================================================

def _download_button_for_text(
    label: str,
    text: str,
    file_name: str,
    mime: str = "text/plain",
    key: Optional[str] = None,
) -> None:
    if not text:
        return
    data = text.encode("utf-8")
    st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        key=key,
    )


def _download_button_for_json(
    label: str,
    obj: dict,
    file_name: str,
    key: Optional[str] = None,
) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime="application/json",
        key=key,
    )


def _download_button_for_csv(
    label: str,
    df: pd.DataFrame,
    file_name: str,
    key: Optional[str] = None,
) -> None:
    if df is None or df.empty:
        return
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        key=key,
    )


def _render_reports_and_export_section(
    cfg: IndexFundamentalsTabConfig,
    universe: list[str],
    summary: UniverseScoresSummary,
) -> None:
    st.markdown("### 📑 דוחות ו-Export")

    panel = summary.panel
    if panel is None or panel.empty:
        st.info("אין Panel מריצת ה-Universe – הרץ קודם את הניתוח.")
        return

    snapshot_df = get_universe_score_snapshot(
        summary,
        include_momentum=cfg.ui.show_momentum,
        include_stability=cfg.ui.show_stability,
    )

    # Universe Report
    with st.expander("🌎 דוח Universe (Markdown)", expanded=False):
        if st.button("📝 צור דוח Universe", key="btn_universe_report"):
            with st.spinner("בונה דוח Universe..."):
                md_universe = generate_universe_markdown_report(summary, top_n=cfg.ui.top_n_default)
            st.markdown("#### תצוגת הדוח (Markdown)")
            st.markdown(md_universe)
            _download_button_for_text(
                "⬇️ הורד כ־universe_report.md",
                md_universe,
                file_name="universe_fundamentals_report.md",
                mime="text/markdown",
                key="dl_universe_md",
            )

    # Index Report
    with st.expander("📈 דוח למדד בודד (Markdown)", expanded=False):
        if snapshot_df.empty:
            st.info("אין snapshot זמין. הרץ ניתוח Universe קודם.")
        else:
            sym_list = list(snapshot_df.index)
            col_sel, col_btn = st.columns([2, 1])
            with col_sel:
                sym = st.selectbox(
                    "בחר מדד לדוח",
                    options=sym_list,
                    key="report_index_symbol",
                )
            with col_btn:
                if st.button("📝 צור דוח למדד", key="btn_index_report"):
                    with st.spinner("בונה דוח למדד..."):
                        md_index = generate_index_markdown_report(sym, engine_cfg=cfg.engine)
                    st.markdown(f"#### דוח למדד {sym}")
                    st.markdown(md_index)
                    _download_button_for_text(
                        f"⬇️ הורד כ-{sym}_fundamentals.md",
                        md_index,
                        file_name=f"{sym}_fundamentals_report.md",
                        mime="text/markdown",
                        key="dl_index_md",
                    )

    # JSON export
    with st.expander("📦 Export JSON (Universe + Index)", expanded=False):
        col_univ_json, col_idx_json = st.columns(2)

        with col_univ_json:
            univ_payload = export_universe_scores_to_dict(summary)
            st.caption("Universe JSON Snapshot (מקוצר)")
            st.json({k: v for k, v in univ_payload.items() if k != "symbols"})
            _download_button_for_json(
                "⬇️ הורד Universe JSON מלא",
                univ_payload,
                file_name="universe_fundamentals_snapshot.json",
                key="dl_universe_json",
            )

        with col_idx_json:
            if snapshot_df.empty:
                st.info("אין snapshot למדדים.")
            else:
                sym = st.selectbox(
                    "בחר מדד ל-JSON",
                    options=list(snapshot_df.index),
                    key="json_index_symbol",
                )
                index_scores_df = score_index_fundamentals_df(sym, engine_cfg=cfg.engine)
                idx_payload = export_index_scores_to_dict(
                    type("Tmp", (), {"symbol": sym, "df": index_scores_df})(),  # IndexScoreSeries זמני
                    latest_only=True,
                )
                st.caption("Index JSON Snapshot")
                st.json(idx_payload)
                _download_button_for_json(
                    f"⬇️ הורד JSON עבור {sym}",
                    idx_payload,
                    file_name=f"{sym}_fundamentals_snapshot.json",
                    key="dl_index_json",
                )

    # CSV export for pair ideas
    with st.expander("🔁 Export לרעיונות זוגות (CSV)", expanded=False):
        df_ideas: Optional[pd.DataFrame] = st.session_state.get("last_pair_ideas_df")
        if df_ideas is None or df_ideas.empty:
            st.info("אין טבלת רעיונות זוגות זמינה מהריצה הנוכחית.")
        else:
            st.dataframe(df_ideas.head(20).round(3), use_container_width=True)
            _download_button_for_csv(
                "⬇️ הורד את כל רעיונות הזוגות כ-CSV",
                df_ideas,
                file_name="pair_ideas_fundamentals_spread.csv",
                key="dl_pair_ideas_csv",
            )

    # Debug / Coverage
    with st.expander("🧪 Debug & Coverage", expanded=False):
        st.caption("Coverage info (מגיע מ-score_universe_fundamentals):")
        st.json(summary.coverage_info)
        st.caption("Panel shape:")
        st.write(summary.panel.shape)


# ============================================================
# Main tab function – with app_ctx integration
# ============================================================

def render_index_fundamentals_tab(app_ctx: Optional[Any] = None) -> None:
    """
    טאב מלא:
    1. Universe overview (scores).
    2. Pair fundamentals X/Y.
    3. Fundamental vs Spread matrix.
    4. Pair idea factory.
    5. Single Index Deep Dive.
    6. Reports & Export.

    app_ctx:
        - אם dict → נכתוב לתוכו תחת app_ctx["fundamentals"]:
            * last_universe
            * last_universe_meta
            * last_pair_ideas
            * last_focus_symbol
        - אם None → נעבוד רק עם session_state.
    """
    st.markdown("### 🧮 ניתוח פנדומנטלי + ספרד למדדים")

    cfg = _get_tab_config_from_session()

    # הגדרות תצוגה
    with st.expander("⚙️ הגדרות תצוגה", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            cfg.ui.advanced_mode = st.checkbox("מצב מתקדם", value=cfg.ui.advanced_mode)
        with col2:
            cfg.ui.show_momentum = st.checkbox("הצג מומנטום ציונים", value=cfg.ui.show_momentum)
            cfg.ui.show_stability = st.checkbox("הצג יציבות ציונים", value=cfg.ui.show_stability)
        with col3:
            cfg.ui.table_density = st.selectbox(
                "צפיפות טבלאות",
                options=["compact", "normal", "spacious"],
                index=["compact", "normal", "spacious"].index(cfg.ui.table_density),
            )

    _save_tab_config_to_session(cfg)

    universe = _render_universe_selector(cfg)
    if not universe:
        st.warning("לא נבחרו סימולים ב-Universe.")
        return

    run_btn = st.button("🚀 הרץ ניתוח Universe + השוואות + רעיונות + דיפ־דייב + דוחות", type="primary")
    if not run_btn:
        st.info("בחר Universe ולחץ על הכפתור.")
        return

    # 1. Universe Overview
    summary = _render_universe_overview_section(cfg, universe)

    # ---- הזנת מדדי Fundamentals ל-Tab Comparison (fundamentals_metrics) ----
    try:
        snapshot_df = get_universe_score_snapshot(
            summary,
            include_momentum=cfg.ui.show_momentum,
            include_stability=cfg.ui.show_stability,
        )

        if snapshot_df is not None and not snapshot_df.empty:
            valuation_score = None
            fund_total_score = None

            if "value_score" in snapshot_df.columns:
                valuation_score = float(snapshot_df["value_score"].mean())

            if "composite_score" in snapshot_df.columns:
                fund_total_score = float(snapshot_df["composite_score"].mean())

            if valuation_score is not None and fund_total_score is not None:
                push_fundamentals_metrics_to_ctx(
                    valuation_score=valuation_score,   # 0..100 בערך – ציון Value ממוצע
                    fund_score=fund_total_score,       # 0..100 – Composite ממוצע
                    # כרגע אין לך Sharpe/DD ברמת Fundamentals, אז נשאיר None
                    sharpe_60d=None,
                    max_dd_60d=None,
                )
    except Exception as exc:
        logger.debug("push_fundamentals_metrics_to_ctx failed (non-fatal): %s", exc)

    # שמירה ל-app_ctx
    if isinstance(app_ctx, dict):
        app_ctx.setdefault("fundamentals", {})
        app_ctx["fundamentals"]["last_universe"] = universe
        app_ctx["fundamentals"]["last_universe_meta"] = {
            "panel_shape": summary.panel.shape if summary.panel is not None else None,
            "snapshot_date": summary.snapshot_date.isoformat() if summary.snapshot_date is not None else None,
        }

    # 2. Pair X/Y fundamentals
    st.markdown("---")
    _render_pair_fundamental_comparison_section(universe)

    # 3. Fundamental vs Spread matrix
    st.markdown("---")
    _render_fundamental_vs_spread_section(universe)

    # 4. Pair idea factory
    st.markdown("---")
    _render_pair_idea_factory_section(universe, app_ctx=app_ctx)

    # 5. Single Index Deep Dive
    st.markdown("---")
    _render_single_index_deep_dive_section(cfg, universe, app_ctx=app_ctx)

    # 6. Reports & Export
    st.markdown("---")
    _render_reports_and_export_section(cfg, universe, summary)
