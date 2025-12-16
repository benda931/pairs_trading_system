# -*- coding: utf-8 -*-
"""pair_tab.py — ניתוח זוג ברמת קרן גידור (Part 1/6)
=======================================================

שלב 1/6 — שלד מקצועי מורחב:

בשלב הזה הטאב נותן:
- בחירת זוג וטעינת מחירי *legs* דרך MarketDataRouter / load_price_data.
- חישוב ספרד, Z-Score, קורלציה סטטית, Half-Life, Realized Vol 20d.
- כרטיסי KPI בסיסיים אך מקצועיים.
- גרף מחירים מנורמלים לשני הנכסים.
- גרף Spread לאורך זמן.
- גרף Z-Score עם אזורי כניסה/יציאה בסיסיים.
- טבלת "Spread Diagnostics" עם סטטיסטיקות עומק לספרד ו-Z.
- טבלת "Legs Summary" לתשואה/תנודתיות/Sharpe נאיבי לכל leg.

השלבים הבאים (2–6) יוסיפו:
- טסטים סטטיסטיים (ADF, KPSS, Engle-Granger, Ljung-Box וכו').
- ניתוח Mean-Reversion רב-אופקי (HL short/med/long, Hurst) מתקדם יותר.
- Trade Analytics מה-Backtest, Regime Analysis, תרחישי "מה אם" ופורטליו.
"""
from __future__ import annotations

import logging
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import math

# מודולים פנימיים
from common.utils import (
    calculate_beta,
    calculate_correlation,
    calculate_zscore,
    calculate_half_life,
    load_price_data,  # חשוב: להשתמש באותו loader כמו הדשבורד
)


from dataclasses import asdict  # אם עדיין לא מיובא

try:
    from core.pair_recommender import recommend_pair  # type: ignore
except Exception:
    # fallback: אם המודול לא קיים / יש בו שגיאה, לא מפילים את כל הטאב
    def recommend_pair(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError("core.pair_recommender.recommend_pair is unavailable")

from core.fair_value_engine import FairValueEngine, Config as FairValueConfig  # type: ignore

# ניווט בין טאבים (אם קיים בדשבורד)
try:
    from root.dashboard import set_nav_target  # type: ignore[import]
except Exception:  # pragma: no cover
    set_nav_target = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ==================== Helper Functions ====================

def _extract_symbols_from_pair(pair_obj: Any) -> Tuple[str, str]:
    """
    מקבל אובייקט "זוג" בכל פורמט סביר,
    ומחזיר (sym_x, sym_y) כמחרוזות.

    תומך:
    - dict עם key "symbols": {'symbols': ['XLY', 'XLC']}
    - dict עם keys 'sym_x'/'sym_y'
    - list/tuple באורך 2
    - מחרוזת בפורמטים כמו "XLY/XLC" או "XLY,XLC"
    """
    # מחרוזת "XLY/XLC", "XLY,XLC", "XLY-XLC", "XLY|XLC", "XLY:XLC"
    if isinstance(pair_obj, str):
        s = pair_obj.strip()
        for sep in ("/", ",", "-", "|", ":"):
            if sep in s:
                a, b = s.split(sep, 1)
                return a.strip(), b.strip()


    # רשימה / טופל
    if isinstance(pair_obj, (list, tuple)) and len(pair_obj) == 2:
        return str(pair_obj[0]), str(pair_obj[1])

    # מחרוזת "XLY/XLC" או "XLY,XLC"
    if isinstance(pair_obj, str):
        if "/" in pair_obj:
            a, b = pair_obj.split("/", 1)
            return a.strip(), b.strip()
        if "," in pair_obj:
            a, b = pair_obj.split(",", 1)
            return a.strip(), b.strip()

    raise ValueError(f"Cannot extract symbols from pair object: {pair_obj!r}")


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    """קורלציה בטוחה בין שני וקטורים מספריים (לא על Datetime/index)."""
    a_arr = np.asarray(a.values, dtype="float64")
    b_arr = np.asarray(b.values, dtype="float64")

    if a_arr.size < 2 or b_arr.size < 2:
        return float("nan")

    try:
        return float(np.corrcoef(a_arr, b_arr)[0, 1])
    except Exception:
        return float("nan")
    
def _get_price_series(df: pd.DataFrame, field: str = "Close") -> pd.Series:
    """Extract a clean price series from a price DataFrame.

    עדיף להשתמש בעמודות סטנדרטיות (Close / close / Adj Close וכו'),
    ואם אין – נבחר את העמודה הנומרית הראשונה.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    candidates = [field, field.lower(), "Adj Close", "adj_close", "close"]
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")

    # numeric fallback (למשל אם זה DataFrame גולמי מ-IBKR)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        return pd.to_numeric(df[num_cols[0]], errors="coerce")

    # fallback אחרון: העמודה הראשונה
    return pd.to_numeric(df.iloc[:, 0], errors="coerce")


def _fetch_leg_history(symbol: str, start_date: date, end_date: date, refresh: bool) -> pd.DataFrame:
    """טוען היסטוריית מחירים ל-symbol יחיד.

    ניסיון ראשון: MarketDataRouter (IBKR / ספקים אחרים).
    אם אין Router / אין דאטה → fallback ל-load_price_data ההיסטורי.
    """
    router = st.session_state.get("md_router")
    preferred_source = st.session_state.get("data_source_preferred")

    # 1) ניסיון דרך ה-Router (IBKR וכו')
    if router is not None and not refresh:
        try:
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())
            res = router.get_history(  # type: ignore[attr-defined]
                symbols=[symbol],
                start=start_dt,
                end=end_dt,
                period=None,
                bar_size="1d",
                preferred_source=preferred_source,
                require_non_empty=False,
            )
            df = res.df
            if df is not None and not df.empty:
                if "symbol" in df.columns:
                    df = df[df["symbol"] == symbol]
                if "datetime" in df.columns:
                    df = df.set_index("datetime")
                return df.sort_index()
        except Exception as exc:  # best-effort
            logger.warning("Router-based history for %s failed: %s", symbol, exc)

    # 2) fallback: load_price_data ההיסטורי (לצמדים, או אם Router נכשל)
    try:
        # כמו בדשבורד – קודם בלי תאריכים
        df = load_price_data(symbol)
    except TypeError:
        # אם החתימה אצלך דורשת start/end
        try:
            df = load_price_data(symbol, start_date=start_date, end_date=end_date)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("load_price_data(%s, with dates) failed: %s", symbol, exc)
            return pd.DataFrame()
    except Exception as exc:
        logger.warning("load_price_data(%s) failed: %s", symbol, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    try:
        df = df.copy()
        idx = pd.to_datetime(df.index)
        mask = (idx >= pd.Timestamp(start_date)) & (idx <= pd.Timestamp(end_date))
        return df.loc[mask]
    except Exception:
        # אם חיתוך התאריכים נכשל – נחזיר את הדאטה כמו שהוא
        return df


def _split_pair_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    """נסה לפרק label של זוג ("XLY-XLP", "XLY/XLP" וכו') לשני סימבולים.

    תומך במפרידים: | / \\ : -
    """
    seps = ("|", "/", "\\", ":", "-")
    for sep in seps:
        if sep in label:
            left, right = label.split(sep, 1)
            left, right = left.strip(), right.strip()
            if left and right:
                return left, right
    return None, None


def _pick_default_pair(pairs: List[str]) -> Optional[str]:
    """בחר pair דיפולטי לפי selected_pair ב-session אם קיים."""
    if not pairs:
        return None
    sel = st.session_state.get("selected_pair")
    if isinstance(sel, str) and sel in pairs:
        return sel
    return pairs[0]


def _compute_half_life(spread: pd.Series) -> float:
    """עטיפה בטוחה לחישוב Half-Life (אם הפונקציה קיימת)."""
    try:
        return float(calculate_half_life(spread.dropna()))
    except Exception:
        return float("nan")


def _compute_realized_vol(series: pd.Series, window: int = 20) -> float:
    """חישוב סטיית תקן של תשואות לוג (Realized Vol) על חלון נתון."""
    if series is None or series.empty or window <= 1:
        return float("nan")
    s = series.dropna().astype(float)
    if len(s) <= window:
        return float("nan")
    r = np.log(s / s.shift(1)).dropna()
    r = r.tail(window)
    if r.empty:
        return float("nan")
    return float(r.std(ddof=0))

@st.cache_resource
def _get_fair_value_engine() -> FairValueEngine:
    """
    יוצר ומחזיר מופע יחיד של FairValueEngine (עם cache של Streamlit),
    כדי לא לאתחל את המנוע בכל רענון.
    """
    try:
        cfg = FairValueConfig()  # נטען קונפיג בסיסי (או מ-params אם קיים)
        return FairValueEngine(config=cfg)
    except Exception as exc:
        logger.warning("FairValueEngine init failed, falling back to default config: %s", exc)
        return FairValueEngine()


def _compute_pair_fair_value_row(
    sym_y: str,
    sym_x: str,
    s_y: pd.Series,
    s_x: pd.Series,
    *,
    cfg_overrides: Optional[Dict[str, Any]] = None,
) -> Optional[pd.Series]:
    """
    מחשב Fair Value לזוג אחד על בסיס סדרות מחירים (s_y, s_x),
    ומחזיר את השורה המועדפת (ensemble window=-1 אם קיים, אחרת window הראשי).

    sym_y, sym_x:
        הסימבולים של ה-legs (Y,X) כפי שאתה רוצה לנתח (סדר רק לצורך label).
    s_y, s_x:
        Series של מחירי הסגירה (או מה שאתה משתמש בו בטאב).
    cfg_overrides:
        מילון אופציונלי לצורך override של פרמטרים (window, z_in, z_out וכו').
    """
    if s_y is None or s_x is None or s_y.empty or s_x.empty:
        return None

    # בניית prices_wide קטן מהסדרות הקיימות בטאב
    prices_wide = pd.concat(
        [
            pd.Series(s_y, name=sym_y).astype(float),
            pd.Series(s_x, name=sym_x).astype(float),
        ],
        axis=1,
    ).dropna()

    if prices_wide.empty:
        return None

    eng_base = _get_fair_value_engine()

    # מייצרים קונפיג "קופי" עם overrides רלוונטיים
    cfg_dict = dict(eng_base.config.__dict__)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            if k in cfg_dict:
                cfg_dict[k] = v
    cfg = FairValueConfig(**cfg_dict)
    eng_local = FairValueEngine(config=cfg)

    try:
        res = eng_local.run(prices_wide=prices_wide, pairs=[(sym_y, sym_x)])
    except Exception as exc:
        logger.warning("FairValueEngine.run failed for %s-%s: %s", sym_y, sym_x, exc)
        return None

    if res is None or res.empty:
        return None

    # אם יש שורת ensemble (window == -1) – נעדיף אותה; אחרת window ראשי; אחרת אחרונה
    try:
        if "window" in res.columns and (res["window"] == -1).any():
            row = res.loc[res["window"] == -1].iloc[0]
        elif "window" in res.columns and (res["window"] == cfg.window).any():
            row = res.loc[res["window"] == cfg.window].iloc[-1]
        else:
            row = res.iloc[-1]
    except Exception:
        row = res.iloc[-1]

    row.name = f"{sym_y}-{sym_x}"
    return row


def _render_pair_fair_value_section(
    sym_x: str,
    sym_y: str,
    s1: pd.Series,
    s2: pd.Series,
    *,
    ctx: Optional[Dict[str, Any]] = None,
    table_height: int = 320,
) -> None:
    """
    מציג פאנל Fair Value & Mispricing לזוג נוכחי, על בסיס FairValueEngine.

    sym_x, sym_y:
        סימבולים כפי שנבחרו בטאב (X/Y).
    s1, s2:
        סדרות המחיר שכבר נטענו (מתאימות ל-sym_x/sym_y).
    """
    st.subheader("⚖️ Fair Value & Mispricing (FairValueEngine)")

    # מסתנכרן עם ספי Z מהמערכת (טאב Optimization / Config)
    cfg_overrides: Dict[str, Any] = {}
    try:
        z_in = float(st.session_state.get("entry_z", st.session_state.get("pair_entry_z", 1.5)))
        z_out = float(st.session_state.get("exit_z", st.session_state.get("pair_exit_z", 0.5)))
        cfg_overrides["z_in"] = z_in
        cfg_overrides["z_out"] = z_out
    except Exception:
        z_in = 1.5
        z_out = 0.5

    # חשוב: כאן אני בוחר sym_y כ-Y ו-sym_x כ-X, אבל זה רק ל-label;
    # אם אתה רוצה לשמור על אותו סדר כמו קודם, אפשר להעביר (sym_x, sym_y, s1, s2) ל־compute.
    row = _compute_pair_fair_value_row(sym_y=sym_y, sym_x=sym_x, s_y=s2, s_x=s1, cfg_overrides=cfg_overrides)
    if row is None:
        st.info("לא נמצאה שורת Fair Value תקפה לזוג (בדוק מחירים/היסטוריה).")
        return

    def g(name: str, default: float = float("nan")) -> float:
        try:
            v = row.get(name, default)
            return float(v) if v is not None else default
        except Exception:
            return default

    z = g("zscore")
    mis = g("mispricing")
    vmis = g("vol_adj_mispricing")
    y_fair = g("y_fair")
    hl = g("halflife")
    qw = g("quality_weight")
    sr_net = g("sr_net")
    psr_net = g("psr_net")
    dsr_net = g("dsr_net")
    net_edge_z = g("net_edge_z")
    tgt_units = g("target_pos_units")
    rv = g("realized_vol")
    rsharpe = g("rolling_sharpe")

    action = str(row.get("action", "") or "flat")
    is_coint = bool(row.get("is_coint", False))
    adf_p = g("adf_p")
    res_adf_p = g("residual_adf_p")
    hurst = g("hurst")

    # כרטיסי KPI
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Z-Score (FairValueEngine)", f"{z:.2f}")
        st.metric("Mispricing", f"{mis:.3f}")
    with c2:
        st.metric("Vol-Adj Mispricing", f"{vmis:.3f}")
        st.metric("Net Edge (Z)", f"{net_edge_z:.2f}")
    with c3:
        st.metric("Half-Life (ימים)", "∞" if not np.isfinite(hl) else f"{hl:.1f}")
        st.metric("Quality Weight", f"{qw:.2f}")
    with c4:
        st.metric("SR Net", f"{sr_net:.2f}")
        st.metric("PSR / DSR", f"{psr_net:.2f} / {dsr_net:.2f}")

    c5, c6 = st.columns(2)
    with c5:
        st.metric("Target Position (Units)", f"{tgt_units:.1f}")
    with c6:
        st.metric("Y Fair Value", f"{y_fair:.3f}")

    # סטטיסטיקה מתקדמת
    with st.expander("🔍 סטטיסטיקה מתקדמת (Coint / ADF / Hurst / Vol)", expanded=False):
        st.markdown(
            f"- **is_coint:** `{is_coint}`  \n"
            f"- **ADF p-value (spread):** `{adf_p:.4f}`  \n"
            f"- **Residual ADF p-value:** `{res_adf_p:.4f}`  \n"
            f"- **Hurst:** `{hurst:.3f}`  \n"
            f"- **Realized Vol (spread):** `{rv:.4f}`  \n"
            f"- **Rolling Sharpe (spread):** `{rsharpe:.3f}`"
        )

    st.markdown(f"**Action (Engine):** `{action}`")
    reason = str(row.get("reason", "") or "")
    if reason:
        st.caption(f"Reason: {reason}")

    # שורה מלאה (חלון שנבחר) – אפשר להראות כטבלה קצרה
    with st.expander("📋 שורת Fair-Value שנבחרה", expanded=False):
        try:
            df_row = row.to_frame().T
            st.dataframe(df_row, width="stretch")
        except Exception:
            st.info("לא ניתן להציג טבלה לשורה הנוכחית (row→DataFrame).")


def _describe_spread(
    spread: pd.Series,
    z_series: pd.Series,
    entry_z: float,
    exit_z: float,
) -> pd.DataFrame:
    """בונה טבלת דיאגנוסטיקה מורחבת לספרד ול-Z.

    כולל:
    - mean/median/std/skew/kurtosis/min/max/IQR של הספרד
    - סטטיסטיקות Z: mean/std/skew/kurtosis
    - אחוז ימים ב-|Z|>1/2/3
    - אחוז ימים ב-Entry / Exit / Neutral
    - אחוז זמן בצד +Z / בצד -Z (אסימטריה)
    """
    s = spread.dropna().astype(float)
    z = z_series.dropna().astype(float)

    if s.empty or z.empty:
        return pd.DataFrame()

    desc: Dict[str, float] = {}

    # ספרד
    q1, q3 = np.nanpercentile(s, [25, 75])
    iqr = q3 - q1

    desc["N"] = float(len(s))
    desc["Spread Mean"] = float(s.mean())
    desc["Spread Median"] = float(s.median())
    desc["Spread Std"] = float(s.std(ddof=1))
    desc["Spread Skew"] = float(s.skew())
    desc["Spread Kurtosis"] = float(s.kurtosis())
    desc["Spread Min"] = float(s.min())
    desc["Spread Max"] = float(s.max())
    desc["Spread IQR"] = float(iqr)

    # סטטיסטיקות Z
    z1, z3 = np.nanpercentile(z, [25, 75])
    z_iqr = z3 - z1
    desc["Z Mean"] = float(z.mean())
    desc["Z Median"] = float(z.median())
    desc["Z Std"] = float(z.std(ddof=1))
    desc["Z Skew"] = float(z.skew())
    desc["Z Kurtosis"] = float(z.kurtosis())
    desc["Z IQR"] = float(z_iqr)

    # אזורי Z
    z_abs = z.abs()
    n = float(len(z_abs))
    desc["Pct |Z|>1"] = float((z_abs > 1.0).sum() / n * 100.0)
    desc["Pct |Z|>2"] = float((z_abs > 2.0).sum() / n * 100.0)
    desc["Pct |Z|>3"] = float((z_abs > 3.0).sum() / n * 100.0)

    in_entry = (z_abs >= abs(entry_z)).sum()
    in_exit = (z_abs <= abs(exit_z)).sum()
    neutral = len(z_abs) - in_entry - in_exit

    desc["Days in Entry Zone"] = float(in_entry)
    desc["Days in Exit Zone"] = float(in_exit)
    desc["Days Neutral"] = float(neutral)

    desc["Pct Entry Zone"] = float(in_entry / n * 100.0)
    desc["Pct Exit Zone"] = float(in_exit / n * 100.0)
    desc["Pct Neutral"] = float(neutral / n * 100.0)

    # אסימטריה: כמה זמן בצד +Z וכמה בצד -Z
    pos_side = (z > 0).sum()
    neg_side = (z < 0).sum()
    desc["Pct Z>0"] = float(pos_side / n * 100.0)
    desc["Pct Z<0"] = float(neg_side / n * 100.0)

    df = (
        pd.Series(desc)
        .to_frame(name="Spread / Z Diagnostics")
        .reset_index(names=["Metric"])
    )
    return df


def _compute_leg_stats(series: pd.Series) -> Dict[str, float]:
    """חישוב תשואה, Vol, Sharpe נאיבי, Max Drawdown ו-CAGR ל-leg נתון.

    מתייחס לסדרה כאל מחירי יום-יומיים.
    """
    s = series.dropna().astype(float)
    if len(s) < 5:
        return {
            "Total Return": float("nan"),
            "Vol 20d": float("nan"),
            "Vol 60d": float("nan"),
            "Vol 120d": float("nan"),
            "Sharpe naive": float("nan"),
            "Max Drawdown": float("nan"),
            "CAGR": float("nan"),
        }

    total_ret = float(s.iloc[-1] / s.iloc[0] - 1.0)

    def _vol_w(w: int) -> float:
        return _compute_realized_vol(s, window=w)

    vol20 = _vol_w(20)
    vol60 = _vol_w(60)
    vol120 = _vol_w(120)

    # Sharpe נאיבי על בסיס תשואות יומיות (252 ימי מסחר)
    r = np.log(s / s.shift(1)).dropna()
    if r.empty or float(r.std(ddof=1)) == 0.0:
        sharpe = float("nan")
    else:
        mu = float(r.mean())
        sigma = float(r.std(ddof=1))
        sharpe = float((mu / sigma) * np.sqrt(252.0))

    # Max Drawdown
    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd = (cum / roll_max - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    max_dd = float(dd.min()) if not dd.empty else float("nan")

    # CAGR
    try:
        days = (s.index[-1] - s.index[0]).days or 1
        years = days / 365.25
        cagr = (s.iloc[-1] / max(s.iloc[0], 1e-9)) ** (1.0 / years) - 1.0
        cagr = float(cagr)
    except Exception:
        cagr = float("nan")

    return {
        "Total Return": total_ret,
        "Vol 20d": vol20,
        "Vol 60d": vol60,
        "Vol 120d": vol120,
        "Sharpe naive": sharpe,
        "Max Drawdown": max_dd,
        "CAGR": cagr,
    }


def _build_legs_summary(sym_x: str, s1: pd.Series, sym_y: str, s2: pd.Series) -> pd.DataFrame:
    """בונה טבלת סיכום לשני ה-legs."""
    stats_x = _compute_leg_stats(s1)
    stats_y = _compute_leg_stats(s2)

    df = pd.DataFrame.from_dict(
        {
            sym_x: stats_x,
            sym_y: stats_y,
        },
        orient="index",
    )
    df.index.name = "Symbol"
    return df.reset_index()

def _load_pair_prices_for_research(sym_a: str, sym_b: str, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.Series, pd.Series]:
    """
    Helper קטן לטעינת מחירים לזוג, לטובת pair_recommender.
    אפשר להתאים אותו ל-MarketDataRouter / IBKR / Yahoo לפי מה שאתה משתמש.
    """
    from common.utils import load_price_data  # או ה-router שלך

    # דוגמה: load_price_data מחזיר DataFrame עם עמודות הסימבולים
    df = load_price_data(symbols=[sym_a, sym_b], start_date=start, end_date=end, freq="D")
    df = df.sort_index()
    if sym_a not in df.columns or sym_b not in df.columns:
        raise ValueError(f"Missing columns {sym_a}/{sym_b} in price data.")

    price_a = pd.to_numeric(df[sym_a], errors="coerce").dropna()
    price_b = pd.to_numeric(df[sym_b], errors="coerce").dropna()
    return price_a, price_b

def _zscore_series(x: pd.Series, window: int) -> pd.Series:
    """
    Z-Score מתגלגל לסדרה:

        Z_t = (x_t - mean_window) / std_window
    """
    x = pd.to_numeric(x, errors="coerce")
    mu = x.rolling(window).mean()
    sigma = x.rolling(window).std(ddof=0)
    z = (x - mu) / sigma
    return z

def _pairs_from_app_ctx(ctx: Optional[Dict[str, Any]]) -> List[str]:
    """
    בונה רשימת pairs (מחרוזות) מתוך app_ctx["fundamentals"]["last_pair_ideas"],
    אם קיימים רעיונות זוגות מהטאב הפנדומנטלי.

    כל pair יהיה בצורה 'X/Y'.
    """
    if not isinstance(ctx, dict):
        return []

    fund_ctx = ctx.get("fundamentals", {})
    ideas = fund_ctx.get("last_pair_ideas", [])
    if not ideas:
        return []

    pairs_set = set()
    for row in ideas:
        pair_label = row.get("pair")
        if isinstance(pair_label, str) and "/" in pair_label:
            pairs_set.add(pair_label)
        else:
            # fallback: לנסות לבנות מ-symbol_x/symbol_y
            sx = row.get("symbol_x")
            sy = row.get("symbol_y")
            if isinstance(sx, str) and isinstance(sy, str):
                pairs_set.add(f"{sx}/{sy}")

    return sorted(pairs_set)

def _pick_default_pair_from_focus(
    pairs: List[str],
    ctx: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    בוחר pair דיפולטי לפי last_focus_symbol מה-ctx, אם קיים.

    מחפש first pair שבו הסימבול מופיע כ-X או Y.
    pairs הם מחרוזות כמו 'X/Y' או 'X-Y'.
    """
    if not isinstance(ctx, dict) or not pairs:
        return None

    fund_ctx = ctx.get("fundamentals", {})
    focus = fund_ctx.get("last_focus_symbol")
    if not isinstance(focus, str) or not focus:
        return None
    focus = focus.upper()

    candidates = []
    for p in pairs:
        try:
            sx, sy = _extract_symbols_from_pair(p)
        except Exception:
            sx, sy = _split_pair_label(str(p))
        sx = (sx or "").upper()
        sy = (sy or "").upper()
        if focus in (sx, sy):
            candidates.append(p)

    if candidates:
        return candidates[0]
    return None

# ==================== Main Tab Renderer (Part 1) ====================


def _render_pair_tab_core(
    pairs: List[str],
    config: Dict[str, Any],
    start_date: date,
    end_date: date,
    ctx: Optional[Dict[str, Any]] = None,
    **controls: Any,
) -> None:
    """
    טאב ניתוח זוג (Pair Analysis) — גרסה מורחבת (HF-grade, Tabs)

    מה הטאב נותן:
    --------------
    1. בסיס:
       - בחירת זוג (עם סנכרון ל-wrapper: active_pair/scope/compare_pairs).
       - טעינת מחירים לשני ה-legs דרך MarketDataRouter / load_price_data.
       - חישוב Spread, Z-Score, קורלציה סטטית, Half-Life, Realized Vol 20d.
       - כרטיסי KPI בסיסיים + Go/No-Go summary.
       - גרף מחירים מנורמלים, גרף Spread, גרף Z-Score.
       - Data-quality summary לזוג.

    2. Stats & Regimes:
       - אבחון סטטיסטי מתקדם (ADF / KPSS / Engle-Granger / Ljung-Box / ACF/PACF / Histogram Z).
       - Mean-Reversion & Regime Analysis (Multi-HL, Hurst, Regime Summary).
       - Macro overlay בסיסי אם ctx מכיל מידע רלוונטי.

    3. Trades:
       - Trade Analytics & Backtest Distribution (אם יש תוצאות Backtest ב-session).
       - רמז ל-Drift בין Backtest ל-live (אם קיים מידע).

    4. Scenarios:
       - Scenario Analysis (What-If) על הספרד והמחירים.

    5. Report:
       - Pair Report להורדה (Markdown) שמרכז את כל התמונה.
    """

    # RTL כלל-מערכתי לטאב
    st.markdown(
        """
        <style>
        .stApp { direction: rtl; }
        .stMarkdown, .stMetric, .stDataFrame, .stPlotlyChart { text-align: right; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # פרמטרים מה-controls (מגיעים מה-dashboard הראשי)
    entry_z = float(controls.get("entry_z", config.get("z_entry", 2.0)))
    exit_z = float(controls.get("exit_z", config.get("z_exit", 0.5)))
    ma_win = int(controls.get("ma_win", config.get("ma_win", 20)))

    profile = str(controls.get("profile", config.get("profile", "research"))).lower()
    analysis_mode = controls.get("analysis_mode", "מחקר")
    time_preset = controls.get("time_preset", None)
    scope = controls.get("scope", "זוג נבחר בלבד")
    active_pair_from_controls = controls.get("active_pair")
    compare_pairs = controls.get("compare_pairs") or []

    st.header("🔍 ניתוח זוג מתקדם — בסיס מורחב")

    # ---- 0. אם לא הגיעו pairs מבחוץ – ניסיון אוטומטי מה-ctx ----
    if not pairs:
        auto_pairs = _pairs_from_app_ctx(ctx)
        if auto_pairs:
            pairs = auto_pairs
            st.info("משתמש ברעיונות מהטאב הפנדומנטלי בתור pairs לניתוח.")
        else:
            st.warning("אין זוגות זמינים לניתוח (לא הגיעו מבחוץ, ואין pair ideas מהטאב הפנדומנטלי).")
            return

    # ---- 1. מייצר label קריא לכל אובייקט זוג (dict / list / str ...) ----
    def _make_pair_label_local(p: Any) -> str:
        try:
            if isinstance(p, dict):
                if "pair" in p:
                    return str(p["pair"])
                if "symbols" in p and isinstance(p["symbols"], (list, tuple)) and len(p["symbols"]) == 2:
                    return f"{p['symbols'][0]}-{p['symbols'][1]}"
                if "sym_x" in p and "sym_y" in p:
                    return f"{p['sym_x']}-{p['sym_y']}"
            if isinstance(p, (list, tuple)) and len(p) == 2:
                return f"{p[0]}-{p[1]}"
            # fallback – מחרוזת פשוטה
            return str(p)
        except Exception:
            return str(p)

    labels = [_make_pair_label_local(p) for p in pairs]

    # ---- 2. בחירת זוג לפי:
    #   a) active_pair מה-controls (wrapper)
    #   b) selected_pair מה-session
    #   c) focus מה-ctx
    #   d) הראשון ברשימה
    default_label = None
    if isinstance(active_pair_from_controls, str):
        # אם ה-wrapper בחר זוג – ננסה להשתמש בו
        default_label = active_pair_from_controls.replace("/", "-")

    if not default_label:
        default_label = st.session_state.get("selected_pair")

    if isinstance(default_label, str) and default_label in labels:
        default_index = labels.index(default_label)
    else:
        focused_pair_label = _pick_default_pair_from_focus(labels, ctx)
        if isinstance(focused_pair_label, str) and focused_pair_label in labels:
            default_index = labels.index(focused_pair_label)
        else:
            default_index = 0

    selected_label = st.selectbox(
        "🎯 בחר זוג לניתוח",
        options=labels,
        index=default_index,
        key="pair_tab_selected_pair",
    )
    st.session_state["selected_pair"] = selected_label

    # מוצאים את האובייקט המקורי של הזוג לפי ה-label
    try:
        pair_idx = labels.index(selected_label)
        pair_obj = pairs[pair_idx]
    except ValueError:
        pair_obj = selected_label

    # ננסה לחלץ sym_x / sym_y מהאובייקט
    try:
        sym_x, sym_y = _extract_symbols_from_pair(pair_obj)
    except Exception:
        sym_x, sym_y = _split_pair_label(str(selected_label))

    if not (sym_x and sym_y):
        st.error("לא הצלחתי לפרק את הזוג לשני סימבולים (sym_x/sym_y). בדוק את הפורמט ב-universe/load_pairs.")
        return

    # ---- 3. טעינת דאטה ----
    refresh = st.sidebar.checkbox(
        "🔄 רענן נתונים לזוג הנוכחי", value=False, key="pair_tab_refresh"
    )

    df_x = _fetch_leg_history(sym_x, start_date, end_date, refresh=refresh)
    df_y = _fetch_leg_history(sym_y, start_date, end_date, refresh=refresh)

    if df_x.empty or df_y.empty:
        st.error("לא נמצאו נתוני מחיר לאחד מהסימבולים.")
        return

    s1 = _get_price_series(df_x)
    s2 = _get_price_series(df_y)
    s1, s2 = s1.align(s2, join="inner")

    if len(s1) < ma_win or len(s2) < ma_win:
        st.warning("אין מספיק היסטוריה עבור חלון החישוב.")
        return

    # ---- 4. חישובים סטטיסטיים בסיסיים ----
    try:
        spread = s1 - s2
        z_series = _zscore_series(spread, ma_win)
    except Exception as exc:
        logger.exception("Z-score computation failed: %s", exc)
        st.error(f"חישוב Z-Score נכשל: {exc}")
        return

    # מתאם סטטי
    try:
        corr_static = float(calculate_correlation(s1, s2))
    except Exception:
        corr_static = _safe_corr(s1, s2)

    # Half-Life & Volatility
    hl = _compute_half_life(spread)
    vol_20 = _compute_realized_vol(spread, window=20)

    # KPI נגזרים
    z_clean = z_series.dropna()
    z_last = float(z_clean.iloc[-1]) if not z_clean.empty else float("nan")
    z_prev = float(z_clean.iloc[-2]) if len(z_clean) > 1 else z_last

    z_delta: Optional[str] = None
    if not (np.isnan(z_last) or np.isnan(z_prev)):
        z_delta = f"{(z_last - z_prev):+.2f}"

    # ---- 5. מבנה הטאב: תתי־טאבים ----
    tab_overview, tab_stats, tab_trades, tab_scenarios, tab_report = st.tabs(
        ["🔎 Overview", "📊 Stats & Regimes", "💼 Trades", "🧪 Scenarios", "🧾 Report"]
    )

    # =========================
    # TAB 1 — OVERVIEW
    # =========================
    with tab_overview:
        st.subheader(f"📌 Overview — {sym_x} / {sym_y}")

        # 5.1 Fair Value Engine — Mispricing מתקדם (אם קיים)
        try:
            table_height = int(st.session_state.get("_opt_table_height", 320))
        except Exception:
            table_height = 320

        try:
            _render_pair_fair_value_section(
                sym_x=sym_x,
                sym_y=sym_y,
                s1=s1,
                s2=s2,
                ctx=ctx,
                table_height=table_height,
            )
        except Exception as e:
            logger.debug("FairValue section failed for %s-%s: %s", sym_x, sym_y, e)

        # 5.2 כרטיסי KPI בסיסיים
        st.markdown("#### 📊 מדדי מצב נוכחי")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Z-Score נוכחי", f"{z_last:.2f}", delta=z_delta)
        c2.metric("מתאם (סטטי)", f"{corr_static:.2f}")
        c3.metric("Half-Life (ימים)", "N/A" if np.isnan(hl) else f"{hl:.1f}")
        c4.metric("Realized Vol 20d", "N/A" if np.isnan(vol_20) else f"{vol_20:.3f}")

        # 5.3 Go / No-Go Summary קצר
        go_no_go = "🔄 ניטרלי"
        reasons: List[str] = []
        if not np.isnan(z_last):
            if abs(z_last) >= entry_z:
                go_no_go = "✅ הזדמנות (Z בקצה band)"
                reasons.append("Z-score קרוב/מעבר לרמת כניסה")
            elif abs(z_last) <= exit_z:
                go_no_go = "⚪ באמצע טווח (לא קצה Z)"
                reasons.append("Z-score קרוב לרמת יציאה / מרכז")
        if corr_static < 0.2:
            go_no_go = "⚠️ קורלציה חלשה"
            reasons.append("מתאם סטטי נמוך מ-0.2")
        if hl > 120 or np.isnan(hl):
            reasons.append("Half-Life ארוך / לא יציב")

        st.markdown(f"**Decision snapshot:** {go_no_go}")
        if reasons:
            st.caption("סיבות עיקריות: " + " | ".join(reasons))

        # 5.4 Data Quality לזוג
        st.markdown("#### 🧾 Data Quality לנכסים")
        dq_cols = ["symbol", "len", "start", "end", "missing_pct"]
        try:
            cov_info = []
            for sym, s in ((sym_x, s1), (sym_y, s2)):
                s_clean = s.dropna()
                cov_info.append(
                    {
                        "symbol": sym,
                        "len": int(len(s_clean)),
                        "start": s_clean.index.min() if len(s_clean) else None,
                        "end": s_clean.index.max() if len(s_clean) else None,
                        "missing_pct": float(
                            100.0 * (1.0 - len(s_clean) / max(len(s), 1))
                        ),
                    }
                )
            dq_df = pd.DataFrame(cov_info, columns=dq_cols)
            st.dataframe(dq_df, width = "stretch")
        except Exception:
            st.caption("לא הצלחתי לחשב Data Quality (בעיה באינדקס/תאריכים).")

        # 5.5 גרף מחירים מנורמלים לשני הנכסים
        st.markdown("#### 📈 מחירי הנכסים (מנורמלים ל-1 בתחילת התקופה)")
        try:
            base_x = s1.iloc[0]
            base_y = s2.iloc[0]
            norm_x = s1 / base_x if base_x not in (0, np.nan) else s1
            norm_y = s2 / base_y if base_y not in (0, np.nan) else s2

            fig_prices = go.Figure()
            fig_prices.add_trace(
                go.Scatter(x=norm_x.index, y=norm_x, name=f"{sym_x} (נורמליזציה)")
            )
            fig_prices.add_trace(
                go.Scatter(x=norm_y.index, y=norm_y, name=f"{sym_y} (נורמליזציה)")
            )
            fig_prices.update_layout(
                xaxis_title="תאריך",
                yaxis_title="מחיר מנורמל",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_prices, width = "stretch")
        except Exception as exc:
            st.caption(f"גרף מחירים מנורמלים לא זמין כרגע: {exc}")

        # 5.6 גרף Spread
        st.markdown("#### 📉 גרף Spread")
        try:
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread"))
            fig_spread.update_layout(
                xaxis_title="תאריך",
                yaxis_title="Spread",
            )
            st.plotly_chart(fig_spread, width = "stretch")
        except Exception as exc:
            st.caption(f"גרף Spread לא זמין כרגע: {exc}")

        # 5.7 גרף Z-Score לאורך זמן
        st.markdown("#### 📉 Z-Score לאורך זמן")
        try:
            df_z = pd.DataFrame({"Z": z_series}).dropna()
            df_z["Upper"] = entry_z
            df_z["Lower"] = -entry_z
            df_z["Exit"] = exit_z

            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(x=df_z.index, y=df_z["Z"], name="Z-Score"))
            fig_z.add_trace(
                go.Scatter(
                    x=df_z.index,
                    y=df_z["Upper"],
                    name="Entry +Z",
                    line=dict(dash="dash"),
                )
            )
            fig_z.add_trace(
                go.Scatter(
                    x=df_z.index,
                    y=df_z["Lower"],
                    name="Entry -Z",
                    line=dict(dash="dash"),
                )
            )
            fig_z.add_trace(
                go.Scatter(
                    x=df_z.index,
                    y=df_z["Exit"],
                    name="Exit Z",
                    line=dict(dash="dot"),
                )
            )
            fig_z.update_layout(
                xaxis_title="תאריך",
                yaxis_title="Z-Score",
            )
            st.plotly_chart(fig_z, width = "stretch")
        except Exception as exc:
            st.caption(f"Z-Score chart unavailable: {exc}")

        # 5.8 Spread Diagnostics + Legs Summary
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**📊 Spread Diagnostics**")
            diag_df = _describe_spread(spread, z_series, entry_z=entry_z, exit_z=exit_z)
            if diag_df.empty:
                st.caption("לא ניתן לחשב סטטיסטיקות ספרד (נתונים חסרים).")
            else:
                def _fmt(v: Any) -> Any:
                    if isinstance(v, (int, float)) and abs(v) < 1e6:
                        return round(float(v), 4)
                    return v

                diag_display = diag_df.copy()
                diag_display["Spread / Z Diagnostics"] = diag_display["Spread / Z Diagnostics"].map(_fmt)
                st.dataframe(diag_display, width = "stretch")

        with col_right:
            st.markdown("**📈 Legs Summary**")
            legs_df = _build_legs_summary(sym_x, s1, sym_y, s2)
            if legs_df.empty:
                st.caption("לא ניתן לחשב סטטיסטיקות ל-legs.")
            else:
                st.dataframe(legs_df, width = "stretch")

        # 5.9 Compare mode – טבלת השוואה מול זוגות נוספים
        if compare_pairs:
            st.markdown("#### 📊 השוואת הזוג לזוגות נוספים (Compare mode)")
            rows = []
            for cp in compare_pairs:
                try:
                    sx, sy = _split_pair_label(str(cp))
                    if not (sx and sy):
                        continue
                    df_cx = _fetch_leg_history(sx, start_date, end_date, refresh=False)
                    df_cy = _fetch_leg_history(sy, start_date, end_date, refresh=False)
                    if df_cx.empty or df_cy.empty:
                        continue
                    cx = _get_price_series(df_cx)
                    cy = _get_price_series(df_cy)
                    cx, cy = cx.align(cy, join="inner")
                    sp = cx - cy
                    zs = _zscore_series(sp, ma_win)
                    corr_cp = _safe_corr(cx, cy)
                    hl_cp = _compute_half_life(sp)
                    vol_cp = _compute_realized_vol(sp, window=20)
                    z_last_cp = float(zs.dropna().iloc[-1]) if not zs.dropna().empty else np.nan

                    rows.append(
                        {
                            "Pair": cp,
                            "Corr": corr_cp,
                            "HL": hl_cp,
                            "Vol20": vol_cp,
                            "Z_last": z_last_cp,
                        }
                    )
                except Exception:
                    continue

            if rows:
                cmp_df = pd.DataFrame(rows)
                st.dataframe(cmp_df.round(3), width = "stretch")
            else:
                st.caption("לא הצלחתי לחשב השוואה לזוגות הנוספים (נתוני מחיר חסרים או בעייתיים).")

        # 5.10 המלצת גודל פוזיציה Heuristic (Risk Suggestion)
        st.markdown("#### 💡 הצעה לגודל פוזיציה (Rule-of-Thumb)")
        try:
            # הנחה: רוצים לסכן כ-1% מההון בטרייד, ללא מידע על ההון עצמו → גודל יחסי
            risk_per_trade = 0.01
            # אם יש HL ו-Vol20 – נשתמש בהם כדי להעריך גודל יחסי
            if not np.isnan(vol_20) and vol_20 > 0 and not np.isnan(hl) and hl > 0:
                # sqrt(HL) אינטואיטיבי: ככל ש-HL גדול, נרצה להקטין גודל
                size_suggestion = risk_per_trade / (vol_20 * (hl ** 0.5))
                st.caption(
                    f"גודל יחסי מוצע (Unitless): ≈ `{size_suggestion:.3f}` "
                    f"(מבוסס על Vol20 ו-HL; דורש התאמה להון שלך ולמגבלות סיכון בפועל)."
                )
            else:
                st.caption("לא ניתן לחשב הצעת גודל פוזיציה (Vol/HL לא זמינים).")
        except Exception:
            st.caption("לא הצלחתי להפיק הצעת גודל פוזיציה (בעיה בחישוב).")

    # =========================
    # TAB 2 — Stats & Regimes
    # =========================
    with tab_stats:
        st.subheader("📊 אבחון סטטיסטי ומשטרי שוק")

        # אבחון סטטיסטי מתקדם
        _render_advanced_diagnostics(spread, z_series, s1, s2, entry_z, exit_z)

        # Mean-Reversion & Regime Analysis
        _render_mean_reversion_and_regime(
            spread=spread,
            corr_static=corr_static,
            vol_20=vol_20,
            hl_windows=[20, 60, 120],
        )

        # נתונים שנשתמש בהם לדוח (וגם לדיאגנוסטיקת Regime)
        hl_df_for_report = _compute_multihorizon_half_life(spread, [20, 60, 120])
        hurst_val = _compute_hurst(spread)
        regime_df = _build_regime_summary(corr_static, vol_20, hl_df_for_report, hurst_val)
        try:
            regime_label = (
                regime_df.loc[regime_df["Metric"] == "Regime (Corr/Vol)", "Value"]
                .astype(str)
                .iloc[0]
            )
        except Exception:
            regime_label = None

        # Macro overlay (אם ctx מכיל מידע)
        st.markdown("#### 🌐 Macro overlay (אם זמין בקונטקסט)")
        try:
            macro_info = None
            if isinstance(ctx, dict):
                macro_info = ctx.get("macro_regime") or ctx.get("macro_context")
            if macro_info is not None:
                st.json(macro_info)
            else:
                st.caption("אין Macro context זמין לזוג זה (אפשר לחבר בעתיד ל-Macro Engine).")
        except Exception:
            st.caption("לא הצלחתי למשוך מידע מאקרו מהקונטקסט.")

    # =========================
    # TAB 3 — Trades
    # =========================
    with tab_trades:
        st.subheader("💼 Trade Analytics & Backtest Distribution")
        backtest_result = st.session_state.get("pair_backtest_result")
        if backtest_result is not None:
            trades_df = _extract_trades_df(backtest_result)
            trade_stats_df = _compute_trade_stats(trades_df) if not trades_df.empty else pd.DataFrame()
        else:
            trades_df = pd.DataFrame()
            trade_stats_df = pd.DataFrame()

        _render_trade_analytics(backtest_result)

        # רמז ל-Drift (בינוני/גבוה) – אם יש מידע נוסף ב-ctx
        if backtest_result is not None and isinstance(ctx, dict):
            st.markdown("#### 🧭 Drift hint (Backtest vs Live)")
            try:
                live_metrics = ctx.get("live_pair_metrics", {}).get(f"{sym_x}/{sym_y}")
                bt_sharpe = getattr(backtest_result, "sharpe", None)
                live_sharpe = None
                if isinstance(live_metrics, dict):
                    live_sharpe = live_metrics.get("sharpe")

                if bt_sharpe is not None and live_sharpe is not None:
                    drift = float(live_sharpe) - float(bt_sharpe)
                    st.caption(
                        f"Sharpe live ≈ {live_sharpe:.2f}, Sharpe backtest ≈ {bt_sharpe:.2f}, drift ≈ {drift:+.2f}"
                    )
                    if abs(drift) > 0.5:
                        st.warning("קיים פער משמעותי בין Backtest ל-live. מומלץ לבדוק את הזוג לעומק.")
                else:
                    st.caption("אין מספיק מידע למסך drift בין Backtest ל-live.")
            except Exception:
                st.caption("לא הצלחתי לחשב drift בין Backtest ל-live.")

    # =========================
    # TAB 4 — Scenarios
    # =========================
    with tab_scenarios:
        st.subheader("🧪 Scenario Analysis (What-If)")
        _render_scenario_analysis(
            spread=spread,
            z_series=z_series,
            s1=s1,
            s2=s2,
            entry_z=entry_z,
            exit_z=exit_z,
        )

        # כרגע לא מחזירים טבלאות תרחישים לדוח (אפשר להרחיב בהמשך)
        mr_scenarios_df = None
        shock_scenarios_df = None

    # =========================
    # TAB 5 — Report
    # =========================
    with tab_report:
        st.subheader("🧾 Pair Report להורדה")
        # כדי שהדוח יעבוד, נוודא שיש לנו את האובייקטים שהוגדרו בטאבים הקודמים
        try:
            hl_df_for_report  # noqa: F821
        except NameError:
            # אם משום מה לא רץ ה-Stats tab (Streamlit בכל מקרה מריץ הכל, אבל להיות דפנסיבי)
            hl_df_for_report = _compute_multihorizon_half_life(spread, [20, 60, 120])
            hurst_val = _compute_hurst(spread)
            regime_df = _build_regime_summary(corr_static, vol_20, hl_df_for_report, hurst_val)
            try:
                regime_label = (
                    regime_df.loc[regime_df["Metric"] == "Regime (Corr/Vol)", "Value"]
                    .astype(str)
                    .iloc[0]
                )
            except Exception:
                regime_label = None

        # אם לא הגיעו trade_stats_df / mr_scenarios_df / shock_scenarios_df – נגדיר כברירת מחדל
        try:
            trade_stats_df  # noqa: F821
        except NameError:
            trade_stats_df = pd.DataFrame()
        try:
            mr_scenarios_df  # noqa: F821
        except NameError:
            mr_scenarios_df = None
        try:
            shock_scenarios_df  # noqa: F821
        except NameError:
            shock_scenarios_df = None
        try:
            diag_df  # noqa: F821
        except NameError:
            diag_df = _describe_spread(spread, z_series, entry_z=entry_z, exit_z=exit_z)
        try:
            legs_df  # noqa: F821
        except NameError:
            legs_df = _build_legs_summary(sym_x, s1, sym_y, s2)

        _render_pair_report_section(
            sym_x=sym_x,
            sym_y=sym_y,
            date_start=start_date,
            date_end=end_date,
            z_last=z_last,
            corr_static=corr_static,
            hl=hl,
            vol_20=vol_20,
            hurst_val=hurst_val,
            regime_label=regime_label,
            spread_diag=diag_df,
            legs_summary=legs_df,
            regime_df=regime_df,
            trade_stats_df=trade_stats_df,
            mr_scenarios_df=mr_scenarios_df,
            shock_scenarios_df=shock_scenarios_df,
        )

# ==================== Adapter לדשבורד החדש (router) ====================

def render_pair_tab(
    app_ctx: Any,
    feature_flags: Dict[str, Any],
    nav_payload: Optional[Dict[str, Any]] = None,
) -> None:
    """
    עטיפה שמתאימה לחתימה שהדשבורד החדש מצפה לה:
        render_pair_tab(app_ctx, feature_flags, nav_payload)

    HF-grade Router לתוך _render_pair_tab_core:
    -------------------------------------------
    - אוסף רשימת זוגות ממספר מקורות (nav_payload, config, ctx, app_ctx).
    - קובע טווח תאריכים לפי base_dashboard_context + nav_payload + presets.
    - מאפשר לבחור זוג פעיל + מצב עבודה (מחקר / ניטור / לפני ביצוע).
    - תומך ב-Compare mode (עד 3 זוגות לניתוח משווה).
    - מציג Header עם env/profile/timespan/source/מספר זוגות.
    - מוסיף כפתורי Next/Prev לזוג, כפתורי "שלח לאופטימיזציה" / "שלח למאקרו".
    - שומר snapshot של הניתוח בסשן עבור Agents/Insights וכו'.
    """

    # --- 1) config בסיסי ---
    try:
        from common.config_manager import load_config  # כמו ב-config_tab.py

        config: Dict[str, Any] = load_config()
    except Exception:
        config = {}

    # --- 2) טווח תאריכים בסיסי (ברירת מחדל שנה אחורה) ---
    base_ctx = st.session_state.get("base_dashboard_context")
    today = date.today()
    start_date: date = today.replace(year=today.year - 1)
    end_date: date = today

    # א. אם base_ctx (DashboardContext או dict) מכיל start/end – נכבד אותו
    try:
        if hasattr(base_ctx, "start_date"):
            sd = getattr(base_ctx, "start_date")
            if isinstance(sd, date):
                start_date = sd
        if hasattr(base_ctx, "end_date"):
            ed = getattr(base_ctx, "end_date")
            if isinstance(ed, date):
                end_date = ed

        if isinstance(base_ctx, dict):
            sd = base_ctx.get("start_date")
            ed = base_ctx.get("end_date")
            if isinstance(sd, date):
                start_date = sd
            if isinstance(ed, date):
                end_date = ed
    except Exception:
        pass

    # ב. nav_payload יכול לדרוס תאריכים (למשל מה-Home / Matrix / Macro)
    nav_source = None
    if isinstance(nav_payload, dict):
        nav_source = nav_payload.get("source") or nav_payload.get("source_tab")
        try:
            nav_start = nav_payload.get("start_date")
            nav_end = nav_payload.get("end_date")
            if isinstance(nav_start, date):
                start_date = nav_start
            if isinstance(nav_end, date):
                end_date = nav_end
        except Exception:
            pass

    # --- 3) pairs: בנייה חכמה לפי סדר עדיפויות ---

    pairs_candidates: List[str] = []

    # 3.0 – זוג אחרון שנבחר בטאב (נוחות)
    last_pair = st.session_state.get("pair_tab_last_pair")
    if isinstance(last_pair, str) and last_pair.strip():
        pairs_candidates.append(last_pair.strip())

    # 3.1 – nav_payload: pair / pairs / symbols / universe_pairs
    if isinstance(nav_payload, dict):
        # pair יחיד כמחרוזת "AAA/BBB"
        p = nav_payload.get("pair")
        if isinstance(p, str) and p.strip():
            pairs_candidates.append(p.strip())

        # רשימת pairs מוכנה
        pl = nav_payload.get("pairs")
        if isinstance(pl, (list, tuple)):
            for x in pl:
                if isinstance(x, str) and x.strip():
                    pairs_candidates.append(x.strip())

        # symbols → pair (למשל ["XLY", "XLP"])
        symbols = nav_payload.get("symbols")
        if isinstance(symbols, (list, tuple)) and len(symbols) == 2:
            s0, s1 = str(symbols[0]).strip(), str(symbols[1]).strip()
            if s0 and s1:
                pairs_candidates.append(f"{s0}/{s1}")

        # universe_pairs (מטריקס / smart scan)
        universe_pairs = nav_payload.get("universe_pairs")
        if isinstance(universe_pairs, (list, tuple)):
            for x in universe_pairs:
                if isinstance(x, str) and x.strip():
                    pairs_candidates.append(x.strip())

    # 3.2 – מתוך config (keys נפוצים)
    if not pairs_candidates:
        for key in ("pairs", "pairs_universe", "ranked_pairs"):
            val = config.get(key)
            if isinstance(val, list) and val:
                for x in val:
                    if isinstance(x, str) and x.strip():
                        pairs_candidates.append(x.strip())
                break

    # 3.3 – מתוך ctx בסשן (ideas מהטאבים האחרים, universe של המערכת)
    if not pairs_candidates:
        ctx_dict = st.session_state.get("ctx")
        if isinstance(ctx_dict, dict):
            try:
                pairs_candidates = _pairs_from_app_ctx(ctx_dict)
            except Exception:
                pairs_candidates = []

    # 3.4 – מתוך app_ctx עצמו (אם מכיל universe / ranked_pairs על האובייקט)
    if not pairs_candidates and app_ctx is not None:
        try:
            if hasattr(app_ctx, "universe") and isinstance(app_ctx.universe, (list, tuple)):
                for x in app_ctx.universe:
                    if isinstance(x, str) and x.strip():
                        pairs_candidates.append(x.strip())
            if hasattr(app_ctx, "ranked_pairs") and isinstance(app_ctx.ranked_pairs, (list, tuple)):
                for x in app_ctx.ranked_pairs:
                    if isinstance(x, str) and x.strip():
                        pairs_candidates.append(x.strip())
        except Exception:
            pass

    # ניקוי כפילויות וריקים
    clean_pairs: List[str] = []
    seen = set()
    for p in pairs_candidates:
        if not isinstance(p, str):
            continue
        ps = p.strip()
        if not ps or ps in seen:
            continue
        clean_pairs.append(ps)
        seen.add(ps)

    pairs = clean_pairs

    if not pairs:
        st.error("⚠️ אין זוגות זמינים לניתוח — לא נמצאו pairs ב-nav_payload, בקונפיג, ב-ctx או ב-app_ctx.")
        return

    # נשמור את הראשון כ"זוג אחרון" לנוחות
    st.session_state["pair_tab_last_pair"] = pairs[0]

    # --- 4) Header קטן בטאב עצמו (לא sidebar) ---

    env = feature_flags.get("env") or config.get("env") or "dev"
    profile = (feature_flags.get("profile") or config.get("profile") or "research").lower()

    st.markdown("### 🧪 ניתוח זוג ברמת קרן")

    hdr_col1, hdr_col2, hdr_col3 = st.columns([2, 2, 2])

    with hdr_col1:
        st.markdown(f"**Environment:** `{env}`  |  **Profile:** `{profile}`")
        if nav_source:
            st.caption(f"Nav source: `{nav_source}`")

    with hdr_col2:
        st.markdown(
            f"**טווח תאריכים (לפני presets):** {start_date.isoformat()} → {end_date.isoformat()}"
        )
        st.caption("ניתן לשנות Preset בסרגל הצד.")

    with hdr_col3:
        st.markdown(f"**מספר זוגות זמינים:** `{len(pairs)}`")
        st.caption("בחר זוג פעיל וגם Compare mode אם תרצה.")

    st.markdown("---")

    # --- 5) Sidebar – מצבי עבודה, טווח זמן, בחירת זוג + Compare mode ---

    st.sidebar.markdown("### 🧪 Pair Tab — מצב עבודה וזוגות")

    # 5.1 מצב עבודה (מחקר / ניטור / לפני ביצוע)
    analysis_mode = st.sidebar.radio(
        "מצב עבודה",
        options=["מחקר", "ניטור", "לפני ביצוע"],
        index=0,
        horizontal=False,
        key="pair_tab_mode",
    )

    # 5.2 Presets לטווח זמן
    time_preset = st.sidebar.selectbox(
        "טווח זמן",
        options=["YTD", "1y", "3y", "All (מה-Context)", "Custom (מה-Context בלבד)"],
        index=1,
        key="pair_tab_time_preset",
    )

    # מיישמים presets על בסיס end_date (או היום)
    effective_end = end_date or today
    if time_preset == "YTD":
        start_date = date(effective_end.year, 1, 1)
    elif time_preset == "1y":
        start_date = effective_end - timedelta(days=365)
    elif time_preset == "3y":
        start_date = effective_end - timedelta(days=3 * 365)
    elif time_preset == "All (מה-Context)":
        # נשאיר את start_date/end_date לפי base_ctx/nav_payload
        pass
    else:  # "Custom (מה-Context בלבד)" – כאן לא משנים כלום, מאפשר טווח שונה רק דרך context חיצוני
        pass

    # 5.3 בחירת זוג פעיל
    try:
        default_idx = (
            pairs.index(st.session_state.get("pair_tab_last_pair"))
            if st.session_state.get("pair_tab_last_pair") in pairs
            else 0
        )
    except ValueError:
        default_idx = 0

    active_pair = st.sidebar.selectbox(
        "זוג לניתוח",
        options=pairs,
        index=default_idx,
        key="pair_tab_active_pair",
    )

    # 5.4 Compare mode – לבחור עד 3 זוגות להשוואה
    compare_pairs: List[str] = []
    if len(pairs) > 1:
        st.sidebar.markdown("### 📊 Compare mode")
        compare_pairs = st.sidebar.multiselect(
            "בחר עד 3 זוגות להשוואה",
            options=[p for p in pairs if p != active_pair],
            default=[],
            max_selections=3,
            key="pair_tab_compare_pairs",
        )

    # 5.5 Scope – זוג בודד או כל הזוגות
    analysis_scope = st.sidebar.radio(
        "Scope",
        options=["זוג נבחר בלבד", "כל הזוגות (סיכומים/Rankings)"],
        index=0,
        horizontal=True,
        key="pair_tab_scope",
    )

    if analysis_scope == "זוג נבחר בלבד":
        pairs_for_core = [active_pair]
    else:
        pairs_for_core = pairs

    # --- 6) כפתורי ניווט Next / Prev לזוג הפעיל (בחלק העליון, מתחת ל-Header) ---

    nav_col_left, nav_col_mid, nav_col_right = st.columns([1, 2, 1])
    with nav_col_left:
        if st.button("⬅ זוג קודם", key="pair_tab_prev_pair"):
            idx = pairs.index(active_pair)
            new_idx = (idx - 1) % len(pairs)
            st.session_state["pair_tab_active_pair"] = pairs[new_idx]
            st.session_state["pair_tab_last_pair"] = pairs[new_idx]
            st.experimental_rerun()
    with nav_col_mid:
        st.markdown(f"**זוג פעיל:** `{active_pair}`")
        if compare_pairs:
            st.caption("Compare mode: " + ", ".join(compare_pairs))
    with nav_col_right:
        if st.button("זוג הבא ➡", key="pair_tab_next_pair"):
            idx = pairs.index(active_pair)
            new_idx = (idx + 1) % len(pairs)
            st.session_state["pair_tab_active_pair"] = pairs[new_idx]
            st.session_state["pair_tab_last_pair"] = pairs[new_idx]
            st.experimental_rerun()

    st.markdown("---")

    # --- 7) פרמטרי כניסה/יציאה בסיסיים לפי profile/mode (לשימוש בליבה) ---

    # התאמת defaults לפי profile + מצב עבודה
    if profile == "conservative" or analysis_mode == "לפני ביצוע":
        default_entry_z = config.get("z_entry", 2.5)
        default_exit_z = config.get("z_exit", 0.5)
    elif profile == "aggressive":
        default_entry_z = config.get("z_entry", 1.5)
        default_exit_z = config.get("z_exit", 0.0)
    else:  # research / balanced
        default_entry_z = config.get("z_entry", 2.0)
        default_exit_z = config.get("z_exit", 0.5)

    entry_z = st.sidebar.number_input(
        "Z-score כניסה",
        value=float(default_entry_z),
        step=0.1,
        key="pair_tab_entry_z",
    )
    exit_z = st.sidebar.number_input(
        "Z-score יציאה",
        value=float(default_exit_z),
        step=0.1,
        key="pair_tab_exit_z",
    )
    ma_win = st.sidebar.number_input(
        "חלון ממוצע נע לספרד (ימים)",
        value=int(config.get("ma_win", 20)),
        min_value=5,
        max_value=252,
        step=1,
        key="pair_tab_ma_win",
    )

    # --- 8) כפתורים לשליחת הזוג לטאבים אחרים (Optimization / Macro / וכו') ---

    st.sidebar.markdown("### 🔗 ניווט לטאבים אחרים")

    if set_nav_target is not None:
        # Optimization
        if st.sidebar.button("🚀 שלח הזוג לאופטימיזציה", key="pair_tab_send_to_opt"):
            payload = {
                "pairs": [active_pair],
                "start_date": start_date,
                "end_date": end_date,
                "source": "pair_tab",
            }
            try:
                set_nav_target("optimization", payload)
                st.sidebar.success("הוגדר nav_target לטאב האופטימיזציה.")
            except Exception as exc:
                st.sidebar.warning(f"לא הצלחתי להגדיר nav_target לאופטימיזציה: {exc}")

        # Macro
        if st.sidebar.button("🌐 הצג הזוג מול מאקרו", key="pair_tab_send_to_macro"):
            payload = {
                "pair": active_pair,
                "start_date": start_date,
                "end_date": end_date,
                "source": "pair_tab",
            }
            try:
                set_nav_target("macro", payload)
                st.sidebar.success("הוגדר nav_target לטאב המאקרו.")
            except Exception as exc:
                st.sidebar.warning(f"לא הצלחתי להגדיר nav_target למאקרו: {exc}")

    # --- 9) Health בסיסי מה-ctx (אם יש) + snapshot לסוכנים ---

    ctx_session = st.session_state.get("ctx")
    if isinstance(ctx_session, dict):
        ctx_for_core = ctx_session
        # נסה למשוך מידע quality אם קיים
        health = ctx_session.get("data_quality", {}).get(active_pair)
    else:
        ctx_for_core = None
        health = None

    if health is not None:
        try:
            st.info(
                f"Data health לזוג {active_pair}: "
                f"len={health.get('len')}, start={health.get('start')}, end={health.get('end')}, "
                f"missing={health.get('missing_pct'):.2f}%"
            )
        except Exception:
            pass

    # שמירת snapshot בסיסי ל-Agents/Insights
    st.session_state["pair_tab_last_snapshot"] = {
        "pair": active_pair,
        "compare_pairs": compare_pairs,
        "scope": analysis_scope,
        "env": env,
        "profile": profile,
        "mode": analysis_mode,
        "start_date": start_date,
        "end_date": end_date,
        "nav_source": nav_source,
    }

    # --- 10) controls שנעביר לליבה (אפשר להרחיב בעתיד בלי לשבור) ---
    controls: Dict[str, Any] = {
        "entry_z": float(entry_z),
        "exit_z": float(exit_z),
        "ma_win": int(ma_win),
        "profile": profile,
        "env": env,
        "active_pair": active_pair,
        "scope": analysis_scope,
        "analysis_mode": analysis_mode,
        "time_preset": time_preset,
        "compare_pairs": compare_pairs,
        "nav_payload": nav_payload,
        "source": "pair_tab",
    }

    # --- 11) קריאה לפונקציית הליבה (Backwards compatible) ---
    _render_pair_tab_core(
        pairs=pairs_for_core,
        config=config,
        start_date=start_date,
        end_date=end_date,
        ctx=ctx_for_core,
        **controls,
    )

# ==================== Part 2 — Advanced Statistical Diagnostics ====================

# Optional statsmodels / arch imports (אבחון סטטיסטי מתקדם ברמת קרן)
try:  # pragma: no cover - כל החבילה אופציונלית
    import statsmodels.api as sm  # type: ignore[import]
    from statsmodels.tsa.stattools import (  # type: ignore[import]
        adfuller,
        kpss,
        coint,
        acf,
        pacf,
    )
    from statsmodels.stats.diagnostic import (  # type: ignore[import]
        acorr_ljungbox,
        het_arch,
    )
    from statsmodels.tsa.vector_ar.vecm import coint_johansen  # type: ignore[import]
    from statsmodels.tsa.seasonal import STL  # type: ignore[import]
except Exception:  # אם statsmodels לא מותקן — נבטל בעדינות
    sm = None  # type: ignore[assignment]
    adfuller = None  # type: ignore[assignment]
    kpss = None  # type: ignore[assignment]
    coint = None  # type: ignore[assignment]
    acf = None  # type: ignore[assignment]
    pacf = None  # type: ignore[assignment]
    acorr_ljungbox = None  # type: ignore[assignment]
    het_arch = None  # type: ignore[assignment]
    coint_johansen = None  # type: ignore[assignment]
    STL = None  # type: ignore[assignment]

# Optional ARCH/unit-root imports (adf/kpss/pp משודרגים)
try:  # pragma: no cover - אופציונלי
    from arch.unitroot import (  # type: ignore[import]
        ADF as ARCH_ADF,
        KPSS as ARCH_KPSS,
        PhillipsPerron,
    )
except Exception:
    ARCH_ADF = None  # type: ignore[assignment]
    ARCH_KPSS = None  # type: ignore[assignment]
    PhillipsPerron = None  # type: ignore[assignment]

def _safe_test_available() -> bool:
    """בודק אם statsmodels זמינה. אם לא — נחזיר False ונציג הודעה רכה."""
    return adfuller is not None and kpss is not None


def _run_stationarity_tests(
    spread: pd.Series,
    s1: pd.Series,
    s2: pd.Series,
    signif: float = 0.05,
) -> pd.DataFrame:
    """מריץ טסטי stationarity / cointegration בסיסיים על הספרד וה-legs.

    כולל:
    - ADF על הספרד
    - KPSS על הספרד
    - Engle-Granger (coint) בין s1 ו-s2
    """
    if not _safe_test_available():
        return pd.DataFrame(
            [
                {
                    "Test": "ADF / KPSS / Engle-Granger",
                    "Statistic": np.nan,
                    "p-value": np.nan,
                    "Conclusion": "statsmodels לא מותקן — אין טסטים זמינים",
                }
            ]
        )

    s = spread.dropna().astype(float)
    x = s1.dropna().astype(float)
    y = s2.dropna().astype(float)

    rows = []

    # --- ADF ---
    try:
        adf_res = adfuller(s, autolag="AIC")
        adf_stat, adf_p = float(adf_res[0]), float(adf_res[1])
        concl = "Stationary (דוחה H0)" if adf_p < signif else "Non-stationary (לא דוחה H0)"
        rows.append(
            {
                "Test": "ADF (Spread)",
                "Statistic": adf_stat,
                "p-value": adf_p,
                "Conclusion": concl,
            }
        )
    except Exception as exc:
        rows.append(
            {
                "Test": "ADF (Spread)",
                "Statistic": np.nan,
                "p-value": np.nan,
                "Conclusion": f"Failed: {exc}",
            }
        )

    # --- KPSS ---
    try:
        kpss_res = kpss(s, regression="c", nlags="auto")
        kpss_stat, kpss_p = float(kpss_res[0]), float(kpss_res[1])
        # ב-KPSS H0 היא Stationarity, אז p-value קטן → Non-stationary
        concl = "Non-stationary (דוחה H0)" if kpss_p < signif else "Stationary (לא דוחה H0)"
        rows.append(
            {
                "Test": "KPSS (Spread)",
                "Statistic": kpss_stat,
                "p-value": kpss_p,
                "Conclusion": concl,
            }
        )
    except Exception as exc:
        rows.append(
            {
                "Test": "KPSS (Spread)",
                "Statistic": np.nan,
                "p-value": np.nan,
                "Conclusion": f"Failed: {exc}",
            }
        )

    # --- Engle-Granger Cointegration ---
    try:
        # coint מחזיר (stat, pvalue, crit_vals)
        c_stat, c_p, _ = coint(x.align(y, join="inner")[0], x.align(y, join="inner")[1])
        concl = "Cointegrated (דוחה H0)" if c_p < signif else "Not Cointegrated (לא דוחה H0)"
        rows.append(
            {
                "Test": "Engle-Granger (X,Y)",
                "Statistic": float(c_stat),
                "p-value": float(c_p),
                "Conclusion": concl,
            }
        )
    except Exception as exc:
        rows.append(
            {
                "Test": "Engle-Granger (X,Y)",
                "Statistic": np.nan,
                "p-value": np.nan,
                "Conclusion": f"Failed: {exc}",
            }
        )

    return pd.DataFrame(rows)


def _run_ljung_box_on_z(z_series: pd.Series, lags: int = 10, signif: float = 0.05) -> pd.DataFrame:
    """מבחן Ljung-Box על Z לבדיקת אוטוקורלציה (האם השארית 'לבנה')."""
    if acorr_ljungbox is None:
        return pd.DataFrame(
            [
                {
                    "Lag": np.nan,
                    "LB Stat": np.nan,
                    "p-value": np.nan,
                    "Conclusion": "statsmodels לא מותקן — אין טסט Ljung-Box",
                }
            ]
        )

    z = z_series.dropna().astype(float)
    if z.empty:
        return pd.DataFrame()

    try:
        res = acorr_ljungbox(z, lags=lags, return_df=True)
        rows = []
        for lag, row in res.iterrows():
            p = float(row["lb_pvalue"])
            concl = "Reject H0 (אוטוקורלציה קיימת)" if p < signif else "Do not reject H0 (שארית לבנה)"
            rows.append(
                {
                    "Lag": int(row["lag"]),
                    "LB Stat": float(row["lb_stat"]),
                    "p-value": p,
                    "Conclusion": concl,
                }
            )
        return pd.DataFrame(rows)
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "Lag": np.nan,
                    "LB Stat": np.nan,
                    "p-value": np.nan,
                    "Conclusion": f"Failed: {exc}",
                }
            ]
        )


def _build_acf_pacf(z_series: pd.Series, nlags: int = 20) -> Tuple[pd.Series, pd.Series]:
    """חישוב ACF/PACF בסיסיים ל-Z (אם statsmodels קיים)."""
    z = z_series.dropna().astype(float)
    if z.empty or acf is None or pacf is None:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    try:
        acf_vals = acf(z, nlags=nlags, fft=True)
        pacf_vals = pacf(z, nlags=nlags, method="ywunbiased")
        idx = list(range(len(acf_vals)))
        return pd.Series(acf_vals, index=idx), pd.Series(pacf_vals, index=idx)
    except Exception:
        return pd.Series(dtype=float), pd.Series(dtype=float)


def _plot_z_hist_with_normal(z_series: pd.Series) -> go.Figure:
    """היסטוגרמת Z + עקומת נורמל סטנדרטי לצורך בדיקת צורת התפלגות."""
    z = z_series.dropna().astype(float)
    if z.empty:
        return go.Figure()

    mean = float(z.mean())
    std = float(z.std(ddof=1)) or 1.0

    # בניית בנינים להיסטוגרמה
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=z.values,
            nbinsx=40,
            histnorm="probability",
            name="Z Histogram",
        )
    )

    # עקומת נורמל מקורבת על אותו טווח
    x_vals = np.linspace(z.min(), z.max(), 200)
    pdf_vals = (
        1.0
        / (std * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((x_vals - mean) / std) ** 2)
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=pdf_vals * (z.size * (x_vals[1] - x_vals[0])),  # scaling גס לאותה סקאלה
            mode="lines",
            name="Normal PDF (scaled)",
        )
    )

    fig.update_layout(
        xaxis_title="Z",
        yaxis_title="Probability / Density (scaled)",
        title="Z Distribution vs Normal Approximation",
    )
    return fig


def _render_advanced_diagnostics(
    spread: pd.Series,
    z_series: pd.Series,
    s1: pd.Series,
    s2: pd.Series,
    entry_z: float,
    exit_z: float,
) -> None:
    """UI של חלק 2: אבחון סטטיסטי מתקדם לספרד ול-Z."""
    st.subheader("🔬 חלק 2 — אבחון סטטיסטי מתקדם (Spread & Residuals)")

    # --- טסטים סטטיסטיים לספרד ולצמד ---
    st.markdown("**🧪 טסטי Stationarity / Cointegration**")
    tests_df = _run_stationarity_tests(spread, s1, s2)
    if tests_df.empty:
        st.caption("לא ניתן להריץ טסטים (נתונים חסרים או חוסר חבילות).")
    else:
        # עיגול עדין לקריאות
        display_df = tests_df.copy()
        for col in ["Statistic", "p-value"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(4)
        st.dataframe(display_df, width="stretch")

    # --- Ljung-Box + ACF/PACF ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📉 Ljung-Box על Z (אוטוקורלציה של השארית)**")
        lb_df = _run_ljung_box_on_z(z_series)
        if lb_df.empty:
            st.caption("לא ניתן לחשב Ljung-Box (נתונים חסרים או חוסר חבילה).")
        else:
            lb_disp = lb_df.copy()
            lb_disp["LB Stat"] = lb_disp["LB Stat"].astype(float).round(3)
            lb_disp["p-value"] = lb_disp["p-value"].astype(float).round(4)
            st.dataframe(lb_disp, width="stretch")

    with col2:
        st.markdown("**📊 ACF / PACF ל-Z**")
        acf_vals, pacf_vals = _build_acf_pacf(z_series)
        if acf_vals.empty or pacf_vals.empty:
            st.caption("לא ניתן לחשב ACF/PACF (נתונים חסרים או חוסר חבילה).")
        else:
            fig_acf = go.Figure()
            fig_acf.add_trace(
                go.Bar(x=list(acf_vals.index), y=acf_vals.values, name="ACF")
            )
            fig_acf.update_layout(
                xaxis_title="Lag",
                yaxis_title="Correlation",
                title="ACF of Z",
            )
            st.plotly_chart(fig_acf, width="stretch")

            fig_pacf = go.Figure()
            fig_pacf.add_trace(
                go.Bar(x=list(pacf_vals.index), y=pacf_vals.values, name="PACF")
            )
            fig_pacf.update_layout(
                xaxis_title="Lag",
                yaxis_title="Partial Correlation",
                title="PACF of Z",
            )
            st.plotly_chart(fig_pacf, width="stretch")

    # --- היסטוגרמת Z + Normal ---
    st.markdown("**📈 התפלגות Z (Histogram + Normal Overlay)**")
    fig_hist = _plot_z_hist_with_normal(z_series)
    if fig_hist.data:
        st.plotly_chart(fig_hist, width="stretch")
    else:
        st.caption("לא ניתן להציג היסטוגרמת Z (אין מספיק נתונים).")


# ==================== Part 3 — Mean-Reversion & Regime Analysis ====================

"""
חלק 3 — ניתוח Mean-Reversion ו-Regime Analysis לזוג

החלק הזה מוסיף שכבה ברמת קרן גידור מעל המדדים הבסיסיים:

- חישוב Half-Life על פני כמה אופקי זמן (short / medium / long) עם פרשנות מילולית.
- חישוב Hurst exponent על הספרד + סיווג איכותי של סוג התהליך (Mean-Reverting / Random / Trending).
- Regime Analysis לפי קורלציה ותנודתיות (Low/High Vol × Low/High Corr) עם תיאור מילולי ברור.
- טבלת HL מרובת אופקים + טבלת Regime Summary סינתטית.
- גרף עמודות להשוואת Half-Life בין אופקי הזמן.
- כרטיסי KPI קטנים שנותנים "צילום מצב" על הזוג מבחינת Mean-Reversion.

שימוש מיועד מתוך render_pair_tab (בקובץ הראשי):

    hl_windows = [20, 60, 120]
    _render_mean_reversion_and_regime(
        spread=spread,
        corr_static=corr_static,
        vol_20=vol_20,
        hl_windows=hl_windows,
    )
"""

try:
    from common.utils import hurst_exponent as _hurst_exponent  # type: ignore[attr-defined]
except Exception:  # fallback אם אין מימוש במערכת

    def _hurst_exponent(_series: pd.Series) -> float:  # type: ignore[override]
        """Fallback: אם אין Hurst במערכת, נחזיר NaN."""
        return float("nan")


def _compute_multihorizon_half_life(spread: pd.Series, windows: List[int]) -> pd.DataFrame:
    """חישוב Half-Life על חלונות שונים (short / medium / long).

    לדוגמה: [20, 60, 120]. עבור כל חלון:
    - חותכים את הספרד לקטע האחרון באורך החלון.
    - מחשבים עליו Half-Life.
    - מחשבים HL/Window כדי להעריך את מהירות ה-Mean-Reversion ביחס לאורך החלון.
    - מוסיפים פרשנות מילולית.
    """
    s = spread.dropna().astype(float)
    rows: List[Dict[str, Any]] = []

    for w in windows:
        if len(s) < max(w, 20):
            hl_val = float("nan")
        else:
            try:
                sub = s.tail(w)
                hl_val = float(calculate_half_life(sub))
            except Exception:
                hl_val = float("nan")

        ratio = float(hl_val / w) if (not np.isnan(hl_val) and w > 0) else float("nan")

        if np.isnan(hl_val):
            regime = "לא ידוע (אין מספיק נתונים / כישלון חישוב)"
        elif hl_val < 0:
            regime = "Mean-Reversion לא יציב (HL שלילי)"
        elif hl_val < 0.5 * w:
            regime = "Mean-Reversion מהיר (HL קצר משמעותית מהחלון)"
        elif hl_val <= 1.5 * w:
            regime = "Mean-Reversion בינוני (HL קרוב לאורך החלון)"
        else:
            regime = "Mean-Reversion איטי / חלש (HL ארוך מהחלון)"

        rows.append(
            {
                "Window (ימים)": int(w),
                "Half-Life": hl_val,
                "HL / Window": ratio,
                "Interpretation": regime,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Half-Life"] = df["Half-Life"].astype(float).round(2)
        df["HL / Window"] = df["HL / Window"].astype(float).round(2)
    return df


def _compute_hurst(spread: pd.Series) -> float:
    """עטיפה בטוחה ל-Hurst exponent על הספרד."""
    s = spread.dropna().astype(float)
    if len(s) < 80:  # צריך מספיק נקודות כדי שההערכה תהיה הגיונית
        return float("nan")
    try:
        return float(_hurst_exponent(s))
    except Exception:
        return float("nan")


def _classify_hurst(h: float) -> str:
    """תיאור מילולי ל-Hurst: האם הספרד באמת Mean-Reverting, רנדומי או טרנדי."""
    if np.isnan(h):
        return "לא ידוע (חוסר נתונים / כישלון חישוב)"
    if h < 0.35:
        return "Mean-Reverting חזק (H < 0.35) — ספרד מאוד 'קפיצי' לממוצע"
    if h < 0.5:
        return "Mean-Reverting מתון (0.35 ≤ H < 0.5) — התנהגות סטטיסטית טובה למסחר זוגי"
    if h <= 0.65:
        return "דמוי Random Walk (0.5 ≤ H ≤ 0.65) — אין יתרון ברור ל-Mean-Reversion"
    return "Trending / Persistent (H > 0.65) — התנהגות טרנדית, פחות אידיאלית לספרד קלאסי"


def _classify_regime(
    corr: float,
    vol: float,
    corr_thr_high: float = 0.8,
    corr_thr_low: float = 0.5,
    vol_low: float = 0.01,
    vol_high: float = 0.03,
) -> str:
    """סיווג רג'ים לפי קורלציה ותנודתיות.

    - Low Vol / High Corr → סביבה אידיאלית לזוגיות סטטיסטית.
    - High Vol / High Corr → סביבה מעניינת אך מסוכנת (Tail Risk גבוה).
    - High Vol / Low Corr → מסוכן, קשר חלש.
    - Low Corr / Anything → בדרך כלל להימנע.
    """
    if np.isnan(corr) or np.isnan(vol):
        return "לא ידוע (חוסר נתונים)"

    if corr >= corr_thr_high and vol <= vol_low:
        return "Low Vol / High Corr — אידיאלי לזוג סטטיסטי קלאסי"
    if corr >= corr_thr_high and vol > vol_low:
        return "High Vol / High Corr — פוטנציאל גבוה אך סיכון זנבות גבוה"
    if corr_thr_low <= corr < corr_thr_high and vol <= vol_high:
        return "Mid Corr / Mid Vol — אפשרי, אבל דורש פילטרים מחמירים יותר"
    if corr < corr_thr_low and vol >= vol_high:
        return "High Vol / Low Corr — קשר חלש ורועש, עדיף להתרחק"
    if corr < corr_thr_low:
        return "Low Corr — לא מתאים לזוגיות סטטיסטית קלאסית"
    return "Regime ביניים — נדרש שיקול דעת נוסף (אולי להקטין סייז)"


def _build_regime_summary(
    corr_static: float,
    vol_20: float,
    hl_df: pd.DataFrame,
    hurst: float,
) -> pd.DataFrame:
    """טבלת Regime Summary שמרכזת את מצב הזוג כרגע.

    - Static Correlation
    - Realized Vol 20d
    - Regime (Corr/Vol)
    - Hurst Exponent + פרשנות
    - Typical Half-Life (מהחלון האמצעי)
    """
    regime_label = _classify_regime(corr_static, vol_20)
    hurst_label = _classify_hurst(hurst)

    if not hl_df.empty:
        mid_idx = len(hl_df) // 2
        hl_mid = float(hl_df.iloc[mid_idx]["Half-Life"])
        hl_mid_int = str(hl_df.iloc[mid_idx]["Interpretation"])
    else:
        hl_mid = float("nan")
        hl_mid_int = "לא ידוע"

    rows = [
        {"Metric": "Static Correlation", "Value": round(corr_static, 3) if not np.isnan(corr_static) else np.nan},
        {"Metric": "Realized Vol 20d", "Value": round(vol_20, 4) if not np.isnan(vol_20) else np.nan},
        {"Metric": "Regime (Corr/Vol)", "Value": regime_label},
        {"Metric": "Hurst Exponent", "Value": round(hurst, 3) if not np.isnan(hurst) else np.nan},
        {"Metric": "Hurst Interpretation", "Value": hurst_label},
        {"Metric": "Typical Half-Life", "Value": hl_mid},
        {"Metric": "HL Interpretation", "Value": hl_mid_int},
    ]

    return pd.DataFrame(rows)


def _plot_hl_bar_chart(hl_df: pd.DataFrame) -> go.Figure:
    """גרף עמודות להשוואת Half-Life בין אופקי הזמן."""
    fig = go.Figure()
    if hl_df.empty:
        return fig

    fig.add_trace(
        go.Bar(
            x=hl_df["Window (ימים)"].astype(str),
            y=hl_df["Half-Life"],
            name="Half-Life",
        )
    )
    fig.update_layout(
        xaxis_title="חלון (ימים)",
        yaxis_title="Half-Life (ימים)",
        title="השוואת Half-Life לפי חלון זמן",
    )
    return fig


def _render_mean_reversion_and_regime(
    spread: pd.Series,
    corr_static: float,
    vol_20: float,
    hl_windows: Optional[List[int]] = None,
) -> None:
    """UI של חלק 3: ניתוח Mean-Reversion ו-Regime Analysis לזוג.

    hl_windows: רשימת חלונות (ברירת מחדל: [20, 60, 120]).
    """
    if hl_windows is None:
        hl_windows = [20, 60, 120]

    st.subheader("🌀 חלק 3 — Mean-Reversion & Regime Analysis")

    # ===== Half-Life מרובה אופקים =====
    st.markdown("**⏱ Half-Life על פני אופקי זמן שונים**")
    hl_df = _compute_multihorizon_half_life(spread, hl_windows)
    if hl_df.empty:
        st.caption("לא ניתן לחשב Half-Life (נתונים חסרים או חלונות גדולים מדי).")
    else:
        st.dataframe(hl_df, width="stretch")
        fig_hl = _plot_hl_bar_chart(hl_df)
        if fig_hl.data:
            st.plotly_chart(fig_hl, width="stretch")

    # ===== Hurst & Regime Summary =====
    st.markdown("**🌐 Hurst & Regime Summary**")
    hurst_val = _compute_hurst(spread)
    regime_df = _build_regime_summary(corr_static, vol_20, hl_df, hurst_val)
    st.dataframe(regime_df, width="stretch")

    # כרטיסי KPI קטנים לתחושת "מצב הזוג" מבחינת Mean-Reversion
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Hurst Exponent",
        "N/A" if np.isnan(hurst_val) else f"{hurst_val:.3f}",
        help="<0.5 → Mean-Reverting, ≈0.5 → Random, >0.5 → Trending",
    )
    c2.metric(
        "Static Corr",
        "N/A" if np.isnan(corr_static) else f"{corr_static:.3f}",
    )
    c3.metric(
        "Realized Vol 20d",
        "N/A" if np.isnan(vol_20) else f"{vol_20:.4f}",
    )

    # סיכום טקסטואלי קצר בעברית למצב הזוג
    st.markdown("**📌 סיכום מילולי קצר:**")
    bullets: List[str] = []

    if not np.isnan(hurst_val):
        bullets.append(f"• Hurst ≈ {hurst_val:.2f} — {_classify_hurst(hurst_val)}")
    else:
        bullets.append("• לא ניתן היה לחשב Hurst בצורה אמינה.")

    if not np.isnan(corr_static):
        if corr_static >= 0.8:
            corr_comment = "גבוהה ומתאימה למסחר זוגי"
        elif corr_static >= 0.6:
            corr_comment = "בינונית — כדאי לשלב פילטרים נוספים"
        else:
            corr_comment = "נמוכה — מתאים רק לסטרטגיות נישתיות מאוד"
        bullets.append(f"• קורלציה סטטית ≈ {corr_static:.2f} — {corr_comment}")
    else:
        bullets.append("• לא ניתן היה לחשב קורלציה בצורה אמינה.")

    if not hl_df.empty:
        hl_text = ", ".join(
            (
                f"HL({int(row['Window (ימים)'])}d)≈{row['Half-Life']:.1f}"
                if not np.isnan(row["Half-Life"])
                else f"HL({int(row['Window (ימים)'])}d) לא ידוע"
            )
            for _, row in hl_df.iterrows()
        )
        bullets.append(f"• Half-Life לפי חלונות שונים: {hl_text}.")

    st.markdown("\n".join(bullets))

# ==================== Part 4 — Trade Analytics & Backtest Distribution ====================

"""
חלק 4 — ניתוח טריידים ו-Backtest לזוג יחיד

החלק הזה מטפל באבחון ביצועי האסטרטגיה על הזוג:

- טבלת סטטיסטיקות טריידים:
  #Trades, Win Rate, Avg Win / Avg Loss, Max Win / Loss, Profit Factor,
  Avg Holding Period, Max Holding Period, Time in Market (אם ניתן לחשב).
- גרף היסטוגרמת PnL per trade.
- גרף Cumulative PnL (מתוך equity_curve אם קיים, או מצטבר מה-trades).
- גרף MAE/MFE (אם שדות כאלה קיימים ב-DataFrame של הטריידים).
"""

def _extract_trades_df(backtest_result: Any) -> pd.DataFrame:
    """מוציא DataFrame של טריידים מתוך backtest_result בפורמטים נפוצים.

    תומך בכמה אפשרויות:
    - dict עם מפתח 'trades'
    - אובייקט עם attribute בשם trades
    - כבר DataFrame
    """
    if backtest_result is None:
        return pd.DataFrame()

    if isinstance(backtest_result, pd.DataFrame):
        return backtest_result

    if isinstance(backtest_result, dict):
        trades = backtest_result.get("trades")
        if isinstance(trades, pd.DataFrame):
            return trades
        # לפעמים זה רשימת dicts
        if isinstance(trades, list):
            try:
                return pd.DataFrame(trades)
            except Exception:
                return pd.DataFrame()

    # אובייקט עם attribute
    trades_attr = getattr(backtest_result, "trades", None)
    if isinstance(trades_attr, pd.DataFrame):
        return trades_attr

    return pd.DataFrame()


def _extract_equity_curve(backtest_result: Any) -> pd.Series:
    """מוציא equity_curve / pnl_curve אם קיימים, אחרת בונה מצטבר מ-pnl טריידים."""
    if backtest_result is None:
        return pd.Series(dtype=float)

    # dict-style
    if isinstance(backtest_result, dict):
        for key in ("equity_curve", "pnl_curve", "pnl_series"):
            val = backtest_result.get(key)
            if isinstance(val, (pd.Series, pd.DataFrame)):
                if isinstance(val, pd.DataFrame):
                    # ניקח את העמודה הראשונה האקטיבית
                    num_cols = val.select_dtypes(include=[np.number]).columns
                    if len(num_cols):
                        return pd.to_numeric(val[num_cols[0]], errors="coerce")
                else:
                    return pd.to_numeric(val, errors="coerce")

    # attribute-style
    for attr in ("equity_curve", "pnl_curve", "pnl_series"):
        v_attr = getattr(backtest_result, attr, None)
        if isinstance(v_attr, (pd.Series, pd.DataFrame)):
            if isinstance(v_attr, pd.DataFrame):
                num_cols = v_attr.select_dtypes(include=[np.number]).columns
                if len(num_cols):
                    return pd.to_numeric(v_attr[num_cols[0]], errors="coerce")
            else:
                return pd.to_numeric(v_attr, errors="coerce")

    # fallback: נבנה מצטבר מה-trades לפי exit_time אם יש, אחרת לפי אינדקס
    trades = _extract_trades_df(backtest_result)
    if trades.empty or "pnl" not in trades.columns:
        return pd.Series(dtype=float)

    df = trades.copy()
    if "exit_time" in df.columns:
        idx = pd.to_datetime(df["exit_time"], errors="coerce")
    elif "close_time" in df.columns:
        idx = pd.to_datetime(df["close_time"], errors="coerce")
    else:
        idx = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)

    pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
    eq = pnl.cumsum()
    eq.index = idx
    return eq.sort_index()


def _compute_trade_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """חישוב סטטיסטיקות עיקריות לטריידים."""
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return pd.DataFrame(
            [
                {"Metric": "Trades Available", "Value": 0},
                {"Metric": "Win Rate", "Value": np.nan},
                {"Metric": "Avg Win", "Value": np.nan},
                {"Metric": "Avg Loss", "Value": np.nan},
                {"Metric": "Max Win", "Value": np.nan},
                {"Metric": "Max Loss", "Value": np.nan},
                {"Metric": "Profit Factor", "Value": np.nan},
                {"Metric": "Avg Holding (days)", "Value": np.nan},
                {"Metric": "Max Holding (days)", "Value": np.nan},
                {"Metric": "Time in Market (%)", "Value": np.nan},
            ]
        )

    df = trades.copy()
    pnl = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)

    n_trades = len(df)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = float(len(wins) / n_trades * 100.0) if n_trades > 0 else np.nan
    avg_win = float(wins.mean()) if len(wins) else np.nan
    avg_loss = float(losses.mean()) if len(losses) else np.nan
    max_win = float(wins.max()) if len(wins) else np.nan
    max_loss = float(losses.min()) if len(losses) else np.nan

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else np.nan

    # Holding period (ימים) אם אפשר לחשב
    holding_days = None
    if "entry_time" in df.columns and "exit_time" in df.columns:
        try:
            entry_t = pd.to_datetime(df["entry_time"], errors="coerce")
            exit_t = pd.to_datetime(df["exit_time"], errors="coerce")
            holding = (exit_t - entry_t).dt.total_seconds() / (3600 * 24.0)
            holding_days = holding.replace([np.inf, -np.inf], np.nan).dropna()
        except Exception:
            holding_days = None

    if holding_days is not None and not holding_days.empty:
        avg_hold = float(holding_days.mean())
        max_hold = float(holding_days.max())
    else:
        avg_hold = np.nan
        max_hold = np.nan

    # Time in Market (%): אם יש לנו תקופה משוערת
    time_in_market = np.nan
    if holding_days is not None and not holding_days.empty:
        # Guestimate: סך ימי חשיפה / (טווח תאריכים כולל)
        try:
            total_exposure_days = float(holding_days.sum())
            if "entry_time" in df.columns and "exit_time" in df.columns:
                start_all = pd.to_datetime(df["entry_time"], errors="coerce").min()
                end_all = pd.to_datetime(df["exit_time"], errors="coerce").max()
                total_days = (end_all - start_all).total_seconds() / (3600 * 24.0)
                if total_days > 0:
                    time_in_market = float(total_exposure_days / total_days * 100.0)
        except Exception:
            time_in_market = np.nan

    rows = [
        {"Metric": "Trades Available", "Value": n_trades},
        {"Metric": "Win Rate (%)", "Value": win_rate},
        {"Metric": "Avg Win", "Value": avg_win},
        {"Metric": "Avg Loss", "Value": avg_loss},
        {"Metric": "Max Win", "Value": max_win},
        {"Metric": "Max Loss", "Value": max_loss},
        {"Metric": "Profit Factor", "Value": profit_factor},
        {"Metric": "Avg Holding (days)", "Value": avg_hold},
        {"Metric": "Max Holding (days)", "Value": max_hold},
        {"Metric": "Time in Market (%)", "Value": time_in_market},
    ]
    stats_df = pd.DataFrame(rows)
    return stats_df


def _plot_trade_pnl_histogram(trades: pd.DataFrame) -> go.Figure:
    """היסטוגרמת PnL per trade."""
    fig = go.Figure()
    if trades is None or trades.empty or "pnl" not in trades.columns:
        return fig

    pnl = pd.to_numeric(trades["pnl"], errors="coerce").dropna()
    if pnl.empty:
        return fig

    fig.add_trace(
        go.Histogram(
            x=pnl.values,
            nbinsx=40,
            name="PnL per Trade",
        )
    )
    fig.update_layout(
        xaxis_title="PnL per Trade",
        yaxis_title="Count",
        title="Trade PnL Distribution",
    )
    return fig


def _plot_cumulative_pnl(equity_curve: pd.Series) -> go.Figure:
    """גרף PnL מצטבר (או equity curve) לאורך זמן."""
    fig = go.Figure()
    if equity_curve is None or equity_curve.empty:
        return fig

    s = pd.to_numeric(equity_curve, errors="coerce").dropna()
    if s.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name="Cumulative PnL",
        )
    )
    fig.update_layout(
        xaxis_title="תאריך / אינדקס",
        yaxis_title="PnL מצטבר",
        title="Cumulative PnL — זוג בודד",
    )
    return fig


def _plot_mae_mfe_scatter(trades: pd.DataFrame) -> Optional[go.Figure]:
    """גרף MAE/MFE אם יש עמודות מתאימות ב-DataFrame.

    מצפה לעמודות:
    - 'mae' — Maximum Adverse Excursion
    - 'mfe' — Maximum Favourable Excursion
    """
    if trades is None or trades.empty:
        return None
    if "mae" not in trades.columns or "mfe" not in trades.columns:
        return None

    try:
        mae = pd.to_numeric(trades["mae"], errors="coerce")
        mfe = pd.to_numeric(trades["mfe"], errors="coerce")
        pnl = pd.to_numeric(trades["pnl"], errors="coerce") if "pnl" in trades.columns else None
    except Exception:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mae,
            y=mfe,
            mode="markers",
            name="MAE vs MFE",
            text=None if pnl is None else [f"PnL={val:.2f}" for val in pnl],
        )
    )
    fig.update_layout(
        xaxis_title="MAE",
        yaxis_title="MFE",
        title="MAE / MFE Scatter",
    )
    return fig


def _render_trade_analytics(backtest_result: Any) -> None:
    """UI של חלק 4: ניתוח טריידים ו-Backtest distribution לזוג.

    backtest_result אמור להגיע מ-run_backtest או ממודול backtesting אחר.
    """
    st.subheader("📊 חלק 4 — Trade Analytics & Backtest Distribution")

    trades = _extract_trades_df(backtest_result)
    equity_curve = _extract_equity_curve(backtest_result)

    # ===== טבלת סטטיסטיקות טריידים =====
    st.markdown("**📋 סטטיסטיקות טריידים**")
    stats_df = _compute_trade_stats(trades)
    if stats_df.empty:
        st.caption("אין טריידים זמינים לניתוח (Backtest לא החזיר נתוני טריידים).")
    else:
        display_df = stats_df.copy()
        # פורמט עדין
        def _fmt_val(v: Any) -> Any:
            if isinstance(v, (int, float)) and abs(v) < 1e7:
                return round(float(v), 4)
            return v

        display_df["Value"] = display_df["Value"].map(_fmt_val)
        st.dataframe(display_df, width="stretch")

    # ===== גרפי טריידים =====
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📉 Trade PnL Distribution**")
        fig_hist = _plot_trade_pnl_histogram(trades)
        if fig_hist.data:
            st.plotly_chart(fig_hist, width="stretch")
        else:
            st.caption("לא ניתן לבנות היסטוגרמת PnL (אין מספיק נתונים).")

    with col2:
        st.markdown("**📈 Cumulative PnL**")
        fig_cum = _plot_cumulative_pnl(equity_curve)
        if fig_cum.data:
            st.plotly_chart(fig_cum, width="stretch")
        else:
            st.caption("לא ניתן לבנות עקומת PnL מצטבר (אין equity curve זמין).")

    # ===== MAE/MFE =====
    st.markdown("**📌 MAE / MFE (אם קיים ב-Backtest)**")
    fig_mae_mfe = _plot_mae_mfe_scatter(trades)
    if fig_mae_mfe is not None and fig_mae_mfe.data:
        st.plotly_chart(fig_mae_mfe, width="stretch")
    else:
        st.caption("לא נמצאו עמודות MAE/MFE בנתוני הטריידים — מדלג על הגרף.")

# ==================== Part 5 — Scenario Analysis (What-If) ====================

"""
חלק 5 — ניתוח תרחישים (What-If) לזוג ברמת קרן גידור

מה החלק הזה נותן:

1. תרחישי חזרה לממוצע (Mean-Reversion Scenarios)
   - Z נוכחי → Z יעד: 0, Exit Z, וגם Z יעד מותאם אישית.
   - חישוב Target Spread מתוך μ, σ של הספרד.
   - ΔSpread, PnL לפוזיציית LONG SPREAD ו-SHORT SPREAD.
   - PnL ביחס להון שהוגדר (Capital) → PnL % Capital.

2. תרחישי Shock במחירי ה-legs (Equity Shocks)
   - קומבינציות של ±5% / ±10% על X ו-Y בנפרד וביחד.
   - חישוב Spread חדש, ΔSpread, Z חדש (בקירוב), PnL long/short.

3. UI מקצועי:
   - קלט Units (כמה יחידות ספרד / פוזיציה יחסית).
   - קלט Capital ו-Risk % לראיית PnL כאחוז מההון.
   - טבלאות תרחישים + כותרת עליונה עם התרחיש המרכזי (Main Scenario).

שימוש מתוך render_pair_tab (דוגמה):

    _render_scenario_analysis(
        spread=spread,
        z_series=z_series,
        s1=s1,
        s2=s2,
        entry_z=entry_z,
        exit_z=exit_z,
    )
"""

def _compute_spread_mu_sigma(spread: pd.Series) -> Tuple[float, float]:
    """מחזיר (mean, std) של הספרד לצורך חישובי Z-יעד."""
    s = spread.dropna().astype(float)
    if s.empty:
        return float("nan"), float("nan")
    mu = float(s.mean())
    sigma = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
    return mu, sigma


def _build_mean_reversion_scenarios(
    spread: pd.Series,
    z_series: pd.Series,
    entry_z: float,
    exit_z: float,
    units: float,
    capital: float,
    target_z_custom: Optional[float] = None,
) -> pd.DataFrame:
    """בונה טבלת תרחישים לחזרה לממוצע/יעדי Z שונים.

    target_z_custom: ערך Z מותאם אישית שהמשתמש בוחר (יכול להיות None).
    units: כמה יחידות ספרד הפוזיציה מחזיקה (scale יחסי).
    capital: הון שיוקצה לזוג (לחישוב PnL % Capital).
    """
    s = spread.dropna().astype(float)
    z = z_series.dropna().astype(float)

    if s.empty or z.empty:
        return pd.DataFrame()

    mu, sigma = _compute_spread_mu_sigma(s)
    if np.isnan(mu) or np.isnan(sigma) or sigma == 0.0:
        return pd.DataFrame()

    spread_curr = float(s.iloc[-1])
    z_curr = float(z.iloc[-1])

    # רשימת Z יעדים
    targets: List[float] = [0.0, float(exit_z), -float(exit_z)]
    # תיקון חלקי לכיוון הכניסה (0.5 * entry_z באותו סימן)
    targets.append(0.5 * float(entry_z) * np.sign(z_curr))
    # יעד מותאם אישית אם המשתמש הגדיר
    if target_z_custom is not None:
        targets.append(float(target_z_custom))

    # לנקות כפילויות תוך שמירת סדר
    seen = set()
    target_z_list: List[float] = []
    for t in targets:
        key = round(t, 4)
        if key not in seen:
            seen.add(key)
            target_z_list.append(t)

    rows: List[Dict[str, Any]] = []
    for z_tgt in target_z_list:
        spread_tgt = float(mu + z_tgt * sigma)
        delta_spread = spread_tgt - spread_curr

        pnl_long = delta_spread * units
        pnl_short = -delta_spread * units

        pnl_long_pct = float(pnl_long / capital * 100.0) if capital > 0 else np.nan
        pnl_short_pct = float(pnl_short / capital * 100.0) if capital > 0 else np.nan

        rows.append(
            {
                "Scenario": f"Z: {z_curr:.2f} → {z_tgt:.2f}",
                "Current Z": z_curr,
                "Target Z": z_tgt,
                "Current Spread": spread_curr,
                "Target Spread": spread_tgt,
                "ΔSpread": delta_spread,
                "PnL (Long Spread)": pnl_long,
                "PnL% Capital (Long)": pnl_long_pct,
                "PnL (Short Spread)": pnl_short,
                "PnL% Capital (Short)": pnl_short_pct,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        for col in [
            "Current Spread",
            "Target Spread",
            "ΔSpread",
            "PnL (Long Spread)",
            "PnL (Short Spread)",
            "PnL% Capital (Long)",
            "PnL% Capital (Short)",
        ]:
            df[col] = df[col].astype(float).round(4)
        df["Current Z"] = df["Current Z"].astype(float).round(3)
        df["Target Z"] = df["Target Z"].astype(float).round(3)
    return df


def _build_equity_shock_scenarios(
    s1: pd.Series,
    s2: pd.Series,
    spread: pd.Series,
    units: float,
    capital: float,
    shocks: Optional[List[Tuple[str, float, float]]] = None,
) -> pd.DataFrame:
    """בונה טבלת תרחישי Shock במחירי X/Y.

    shocks: רשימה של (שם_תרחיש, x_mult, y_mult) למשל:
        ("X -5%", 0.95, 1.0), ("Y -5%", 1.0, 0.95) וכו'.
    units: יחידות ספרד.
    capital: הון לשקלול PnL כ-% מההון.
    """
    s1c = s1.dropna().astype(float)
    s2c = s2.dropna().astype(float)
    sp = spread.dropna().astype(float)

    if s1c.empty or s2c.empty or sp.empty:
        return pd.DataFrame()

    p1 = float(s1c.iloc[-1])
    p2 = float(s2c.iloc[-1])
    spread_curr = float(sp.iloc[-1])

    mu, sigma = _compute_spread_mu_sigma(sp)

    if shocks is None:
        shocks = [
            ("X -5%", 0.95, 1.00),
            ("X +5%", 1.05, 1.00),
            ("Y -5%", 1.00, 0.95),
            ("Y +5%", 1.00, 1.05),
            ("X -5%, Y -5%", 0.95, 0.95),
            ("X -10%", 0.90, 1.00),
            ("Y -10%", 1.00, 0.90),
            ("X -10%, Y -10%", 0.90, 0.90),
        ]

    rows: List[Dict[str, Any]] = []
    for name, x_mult, y_mult in shocks:
        p1_new = p1 * x_mult
        p2_new = p2 * y_mult
        spread_new = p1_new - p2_new
        delta_spread = spread_new - spread_curr

        pnl_long = delta_spread * units
        pnl_short = -delta_spread * units
        pnl_long_pct = float(pnl_long / capital * 100.0) if capital > 0 else np.nan
        pnl_short_pct = float(pnl_short / capital * 100.0) if capital > 0 else np.nan

        if not np.isnan(mu) and not np.isnan(sigma) and sigma != 0.0:
            z_new = float((spread_new - mu) / sigma)
        else:
            z_new = float("nan")

        rows.append(
            {
                "Shock Scenario": name,
                "X_mult": x_mult,
                "Y_mult": y_mult,
                "New Price X": p1_new,
                "New Price Y": p2_new,
                "New Spread": spread_new,
                "ΔSpread": delta_spread,
                "Z_new (approx)": z_new,
                "PnL (Long Spread)": pnl_long,
                "PnL% Capital (Long)": pnl_long_pct,
                "PnL (Short Spread)": pnl_short,
                "PnL% Capital (Short)": pnl_short_pct,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        for col in [
            "New Price X",
            "New Price Y",
            "New Spread",
            "ΔSpread",
            "Z_new (approx)",
            "PnL (Long Spread)",
            "PnL% Capital (Long)",
            "PnL (Short Spread)",
            "PnL% Capital (Short)",
        ]:
            df[col] = df[col].astype(float).round(4)
    return df


def _render_scenario_analysis(
    spread: pd.Series,
    z_series: pd.Series,
    s1: pd.Series,
    s2: pd.Series,
    entry_z: float,
    exit_z: float,
) -> None:
    """UI של חלק 5: ניתוח תרחישים (Mean-Reversion + Shocks)."""
    st.subheader("📈 חלק 5 — Scenario Analysis (What-If)")

    s = spread.dropna().astype(float)
    z = z_series.dropna().astype(float)
    if s.empty or z.empty:
        st.caption("לא ניתן לבצע ניתוח תרחישים — אין מספיק דאטה לספרד/Z.")
        return

    # ==== שליטת משתמש ====
    with st.expander("⚙️ הגדרות תרחישים", expanded=True):
        col_conf1, col_conf2, col_conf3 = st.columns(3)
        with col_conf1:
            units = st.number_input(
                "Units (גודל פוזיציה יחסית על הספרד)",
                min_value=0.0,
                value=1.0,
                step=1.0,
                help="כמה 'יחידות ספרד' אתה מניח שהפוזיציה מחזיקה. "
                     "זה scale יחסי — רק לצורך חישוב PnL.",
                key="scenario_units",
            )
        with col_conf2:
            capital = st.number_input(
                "Capital שהוקצה לזוג ($)",
                min_value=0.0,
                value=100_000.0,
                step=10_000.0,
                help="משמש לצורך חישוב PnL כאחוז מההון המוקצה לזוג.",
                key="scenario_capital",
            )
        with col_conf3:
            target_z_custom = st.number_input(
                "Z יעד מותאם אישית (אופציונלי)",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.25,
                help="אם אינך רוצה להשתמש בו — סמן למטה שלא להשתמש בערך הזה.",
                key="scenario_custom_z",
            )
        use_custom = st.checkbox(
            "השתמש בערך Z המותאם אישית בתרחישים",
            value=False,
            key="scenario_use_custom_z",
        )
        target_z_arg: Optional[float] = target_z_custom if use_custom else None

    # ==== תרחישי Mean-Reversion ====
    st.markdown("### 🔄 Mean-Reversion Scenarios (Z → Target)")

    mr_df = _build_mean_reversion_scenarios(
        spread=spread,
        z_series=z_series,
        entry_z=entry_z,
        exit_z=exit_z,
        units=units,
        capital=capital,
        target_z_custom=target_z_arg,
    )

    if mr_df.empty:
        st.caption("לא ניתן לחשב תרחישי חזרה לממוצע (חוסר נתונים / σ=0).")
    else:
        # נבחר את התרחיש המרכזי כ-Z→0
        main_row = mr_df.loc[mr_df["Target Z"].sub(0.0).abs().idxmin()]
        pnl_long_main = float(main_row["PnL (Long Spread)"])
        pnl_long_pct_main = float(main_row["PnL% Capital (Long)"])

        c1, c2 = st.columns(2)
        c1.metric(
            "PnL Long Spread — Z→0",
            f"{pnl_long_main:,.2f} $",
            delta=f"{pnl_long_pct_main:+.2f}%" if not np.isnan(pnl_long_pct_main) else None,
        )
        c2.metric(
            "Current Z",
            f"{float(main_row['Current Z']):.2f}",
            help="Z הנוכחי של הספרד לפי ההיסטוריה וציון זד נוכחי.",
        )

        st.dataframe(mr_df, width="stretch")

    # ==== תרחישי Shock במחירים ====
    st.markdown("### ⚡ Equity Shock Scenarios (X/Y ±%)")

    shocks_df = _build_equity_shock_scenarios(
        s1=s1,
        s2=s2,
        spread=spread,
        units=units,
        capital=capital,
        shocks=None,  # ברירת מחדל פנימית
    )

    if shocks_df.empty:
        st.caption("לא ניתן לחשב תרחישי Shock — חסר דאטה למחירי X/Y או לספרד.")
    else:
        st.dataframe(shocks_df, width="stretch")

    # טקסט סיכום קטן
    st.markdown("**📌 הערות מקצועיות לתרחישים:**")
    comments: List[str] = []
    s_last = float(s.iloc[-1])
    z_last = float(z.iloc[-1])

    comments.append(f"• Spread נוכחי ≈ {s_last:.4f}, Z נוכחי ≈ {z_last:.2f}.")
    if capital > 0 and units > 0:
        comments.append(
            f"• עבור Units={units:.0f} והון מוקצה של ≈{capital:,.0f}$, כל שינוי של 1 יחידת Spread "
            f"מתורגם ל-PnL של ≈{units:,.0f}$, כלומר ≈{units / capital * 100:.2f}% מההון."
        )
    if not mr_df.empty:
        best_row = mr_df.iloc[mr_df["PnL (Long Spread)"].idxmax()]
        comments.append(
            f"• בתרחיש החיובי ביותר ללונג ספרד (מבחינת Z→יעד), PnL≈{best_row['PnL (Long Spread)']:.2f}$ "
            f"(≈{best_row['PnL% Capital (Long)']:.2f}% מההון)."
        )
    if not shocks_df.empty:
        worst_shock = shocks_df.iloc[shocks_df["PnL (Long Spread)"].idxmin()]
        comments.append(
            f"• תרחיש ה-Shock הכי שלילי ללונג ספרד: {worst_shock['Shock Scenario']} "
            f"עם PnL≈{worst_shock['PnL (Long Spread)']:.2f}$ "
            f"(≈{worst_shock['PnL% Capital (Long)']:.2f}% מההון)."
        )

    st.markdown("\n".join(comments))

# ==================== Part 6 — Portfolio View & Pair Report ====================

"""
חלק 6 — אינטגרציית פורטפוליו + דוח Pair Report להורדה

מה החלק הזה נותן:

1. Portfolio View (אופציונלי):
   - אם יש לך equity curve של הזוג + סדרת תשואות של פורטפוליו / Benchmark,
     נקבל:
       * קורלציה בין הזוג לפורטפוליו.
       * תרומה לסיכון (Variance Contribution) בקירוב.
       * משקל מומלץ לזוג (לפי יעד Risk Budget פשוט).

2. Pair Report (Markdown):
   - דוח טקסטואלי מסודר שמרכז:
       * נתוני בסיס (סימבולים, טווח תאריכים)
       * KPIs מרכזיים (Z, corr, HL, Hurst, Vol וכו')
       * Spread Diagnostics
       * Mean-Reversion & Regime Summary
       * Trade Analytics Summary
       * Scenario Analysis Summary
   - כפתור להורדת pair_report_{sym_x}_{sym_y}.md
"""


def _compute_portfolio_correlation_and_risk(
    pair_equity: Optional[pd.Series],
    portfolio_returns: Optional[pd.Series],
    pair_weight: float,
) -> Dict[str, float]:
    """חישוב קורלציה ותרומה לסיכון של הזוג בפורטפוליו (בקירוב)."""
    out = {
        "pair_vol": float("nan"),
        "portfolio_vol": float("nan"),
        "corr_pair_portfolio": float("nan"),
        "risk_contribution": float("nan"),
    }

    if pair_equity is None or portfolio_returns is None:
        return out

    # תשואות הזוג מתוך equity curve (אם היא בסדרה של רווח מצטבר)
    s = pd.to_numeric(pair_equity, errors="coerce").dropna()
    if s.empty:
        return out

    pair_ret = s.pct_change().dropna()
    port_ret = pd.to_numeric(portfolio_returns, errors="coerce").dropna()

    pair_ret, port_ret = pair_ret.align(port_ret, join="inner")
    if pair_ret.empty or port_ret.empty:
        return out

    pair_vol = float(pair_ret.std(ddof=1))
    port_vol = float(port_ret.std(ddof=1))
    corr = float(np.corrcoef(pair_ret, port_ret)[0, 1])

    # תרומה לסיכון (approx: weight * corr * pair_vol / port_vol)
    if port_vol > 0:
        risk_contrib = float(pair_weight * corr * pair_vol / port_vol)
    else:
        risk_contrib = float("nan")

    out.update(
        {
            "pair_vol": pair_vol,
            "portfolio_vol": port_vol,
            "corr_pair_portfolio": corr,
            "risk_contribution": risk_contrib,
        }
    )
    return out


def _render_portfolio_view_for_pair(
    sym_x: str,
    sym_y: str,
    pair_equity: Optional[pd.Series],
    portfolio_returns: Optional[pd.Series],
    pair_weight_current: float,
    target_risk_budget: float = 0.05,
) -> None:
    """UI קטן של Portfolio View לזוג.

    target_risk_budget: כמה מרכיב הסיכון הכולל אתה מוכן לתת לזוג (למשל 5%).
    """
    st.subheader("🏦 חלק 6 — Portfolio View (זוג בתוך הפורטפוליו)")

    if pair_equity is None or portfolio_returns is None:
        st.caption(
            "לא הועברו סדרות פורטפוליו / equity curve — "
            "אפשר יהיה לחבר את זה בהמשך דרך מודול הפורטפוליו."
        )
        return

    metrics = _compute_portfolio_correlation_and_risk(
        pair_equity=pair_equity,
        portfolio_returns=portfolio_returns,
        pair_weight=pair_weight_current,
    )

    corr = metrics["corr_pair_portfolio"]
    pair_vol = metrics["pair_vol"]
    port_vol = metrics["portfolio_vol"]
    risk_contrib = metrics["risk_contribution"]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Corr(pair, portfolio)",
        "N/A" if np.isnan(corr) else f"{corr:.3f}",
    )
    c2.metric(
        "Pair Vol (daily)",
        "N/A" if np.isnan(pair_vol) else f"{pair_vol:.4f}",
    )
    c3.metric(
        "Risk Contribution (approx)",
        "N/A" if np.isnan(risk_contrib) else f"{risk_contrib:.3f}",
        help="בקירוב: weight × corr × pair_vol / portfolio_vol",
    )

    # המלצת משקל פשוטה לפי Risk Budget
    st.markdown("**🎯 המלצת משקל פשוטה לזוג בפורטפוליו**")
    if np.isnan(pair_vol) or np.isnan(port_vol) or port_vol == 0:
        st.caption("לא ניתן לחשב המלצת משקל (חוסר נתונים / σ פורטפוליו=0).")
        return

    # Approx: רוצים risk_contribution ≈ target_risk_budget
    # rc ≈ w * corr * pair_vol / port_vol → w_target ≈ target * port_vol / (corr * pair_vol)
    if np.isnan(corr) or corr == 0:
        st.caption("הקורלציה בין הזוג לפורטפוליו אפסית / לא ידועה — קשה להעריך משקל אופטימלי.")
        return

    w_target = target_risk_budget * port_vol / (corr * pair_vol)
    w_target = float(w_target)

    col_w1, col_w2 = st.columns(2)
    col_w1.metric(
        "Current Weight (guess)",
        f"{pair_weight_current:.3f}",
    )
    col_w2.metric(
        "Target Weight (Risk Budget)",
        f"{w_target:.3f}",
        help=f"מבוסס על תקציב סיכון יעד ≈ {target_risk_budget:.2f}",
    )

    st.caption(
        "הערה: זה חישוב מאוד גס. "
        "במציאות תרצה לשלב גם מגבלות רגולטוריות, קורלציה עם שאר הזוגות, "
        "וגם בדיקת drawdown ו-Stress Tests מערכתיים."
    )


# ---------- Pair Report (Markdown) ----------

def _format_pct(x: Optional[float]) -> str:
    if x is None or np.isnan(x):
        return "N/A"
    return f"{x:.2f}%"


def _format_float(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _build_pair_markdown_report(
    sym_x: str,
    sym_y: str,
    date_start: Optional[date],
    date_end: Optional[date],
    # KPIs בסיסיים
    z_last: float,
    corr_static: float,
    hl: float,
    vol_20: float,
    hurst_val: Optional[float] = None,
    regime_label: Optional[str] = None,
    # טבלאות DataFrame שכבר קיימות בטאב:
    spread_diag: Optional[pd.DataFrame] = None,
    legs_summary: Optional[pd.DataFrame] = None,
    regime_df: Optional[pd.DataFrame] = None,
    trade_stats_df: Optional[pd.DataFrame] = None,
    mr_scenarios_df: Optional[pd.DataFrame] = None,
    shock_scenarios_df: Optional[pd.DataFrame] = None,
) -> str:
    """בונה Markdown report מסודר לזוג.

    מטרה: טקסט שקל לשמור כ-.md ולהמיר ל-PDF בהמשך אם תרצה.
    """

    lines: List[str] = []

    # כותרת
    lines.append(f"# Pair Report — {sym_x} vs {sym_y}")
    lines.append("")
    if date_start and date_end:
        lines.append(f"**Period:** {date_start.isoformat()} → {date_end.isoformat()}")
    lines.append("")

    # סקשן KPIs
    lines.append("## 1. Core KPIs")
    lines.append("")
    lines.append(f"- Latest Z-Score: **{_format_float(z_last, 2)}**")
    lines.append(f"- Static Correlation: **{_format_float(corr_static, 3)}**")
    lines.append(f"- Half-Life (days): **{_format_float(hl, 2)}**")
    lines.append(f"- Realized Vol (20d): **{_format_float(vol_20, 4)}**")

    if hurst_val is not None:
        lines.append(f"- Hurst Exponent: **{_format_float(hurst_val, 3)}**")
    if regime_label:
        lines.append(f"- Regime (Corr/Vol): **{regime_label}**")

    lines.append("")

    # Spread Diagnostics
    lines.append("## 2. Spread Diagnostics")
    lines.append("")
    if spread_diag is None or spread_diag.empty:
        lines.append("_No spread diagnostics available._")
    else:
        lines.append("Key statistics for the spread and Z-distribution:")
        lines.append("")
        for _, row in spread_diag.iterrows():
            metric = str(row.get("Metric", ""))
            value = row.iloc[1]
            lines.append(f"- **{metric}:** {_format_float(value, 4)}")
    lines.append("")

    # Legs Summary
    lines.append("## 3. Legs Summary")
    lines.append("")
    if legs_summary is None or legs_summary.empty:
        lines.append("_No legs summary available._")
    else:
        lines.append("Performance summary for each leg:")
        lines.append("")
        # נסכם רק כמה מדדים מרכזיים
        for _, row in legs_summary.iterrows():
            sym = row.get("Symbol", "N/A")
            tr = row.get("Total Return", np.nan)
            vol20 = row.get("Vol 20d", np.nan)
            sharpe = row.get("Sharpe naive", np.nan)
            lines.append(
                f"- **{sym}:** Total Return={_format_pct(tr * 100 if not np.isnan(tr) else np.nan)}, "
                f"Vol20={_format_float(vol20, 4)}, Sharpe={_format_float(sharpe, 2)}"
            )
    lines.append("")

    # Regime & Mean-Reversion
    lines.append("## 4. Mean-Reversion & Regime Analysis")
    lines.append("")
    if regime_df is None or regime_df.empty:
        lines.append("_No regime summary available._")
    else:
        for _, row in regime_df.iterrows():
            metric = str(row.get("Metric", ""))
            value = row.get("Value")
            lines.append(f"- **{metric}:** {value}")
    lines.append("")

    # Trade Analytics
    lines.append("## 5. Trade Analytics")
    lines.append("")
    if trade_stats_df is None or trade_stats_df.empty:
        lines.append("_No trade statistics available (no backtest or no trades)._")
    else:
        # נבחר כמה שדות עיקריים
        important_metrics = {
            "Trades Available",
            "Win Rate (%)",
            "Avg Win",
            "Avg Loss",
            "Max Win",
            "Max Loss",
            "Profit Factor",
            "Avg Holding (days)",
            "Max Holding (days)",
            "Time in Market (%)",
        }
        for _, row in trade_stats_df.iterrows():
            metric = str(row.get("Metric", ""))
            value = row.get("Value")
            if important_metrics and metric not in important_metrics:
                continue
            if "Rate" in metric or "Time in Market" in metric:
                txt_val = _format_pct(value)
            else:
                txt_val = _format_float(value, 4)
            lines.append(f"- **{metric}:** {txt_val}")
    lines.append("")

    # Scenario Analysis
    lines.append("## 6. Scenario Analysis")
    lines.append("")
    # Mean-Reversion scenarios
    lines.append("### 6.1 Mean-Reversion Scenarios (Z → Target)")
    if mr_scenarios_df is None or mr_scenarios_df.empty:
        lines.append("_No mean-reversion scenarios computed._")
    else:
        for _, row in mr_scenarios_df.iterrows():
            scen = row.get("Scenario", "")
            pnl_long = row.get("PnL (Long Spread)", np.nan)
            pnl_long_pct = row.get("PnL% Capital (Long)", np.nan)
            lines.append(
                f"- **{scen}:** PnL Long ≈ {_format_float(pnl_long, 2)}$ "
                f"({_format_pct(pnl_long_pct)} of allocated capital)"
            )
    lines.append("")
    # Shock scenarios
    lines.append("### 6.2 Equity Shock Scenarios (X/Y ±%)")
    if shock_scenarios_df is None or shock_scenarios_df.empty:
        lines.append("_No shock scenarios computed._")
    else:
        for _, row in shock_scenarios_df.iterrows():
            name = row.get("Shock Scenario", "")
            pnl_long = row.get("PnL (Long Spread)", np.nan)
            pnl_long_pct = row.get("PnL% Capital (Long)", np.nan)
            z_new = row.get("Z_new (approx)", np.nan)
            lines.append(
                f"- **{name}:** Z_new≈{_format_float(z_new, 2)}, "
                f"PnL Long≈{_format_float(pnl_long, 2)}$ "
                f"({_format_pct(pnl_long_pct)} of allocated capital)"
            )
    lines.append("")

    # סיכום
    lines.append("## 7. Overall Assessment")
    lines.append("")
    lines.append(
        "This pair report provides a holistic view of the statistical behaviour, "
        "trading performance and scenario sensitivity of the pair. "
        "Use it as a building block in your portfolio construction and risk budgeting process."
    )
    lines.append("")

    return "\n".join(lines)


def _render_pair_report_section(
    sym_x: str,
    sym_y: str,
    date_start: Optional[date],
    date_end: Optional[date],
    # נתונים שנאספו בטאב:
    z_last: float,
    corr_static: float,
    hl: float,
    vol_20: float,
    hurst_val: Optional[float],
    regime_label: Optional[str],
    spread_diag: Optional[pd.DataFrame],
    legs_summary: Optional[pd.DataFrame],
    regime_df: Optional[pd.DataFrame],
    trade_stats_df: Optional[pd.DataFrame],
    mr_scenarios_df: Optional[pd.DataFrame],
    shock_scenarios_df: Optional[pd.DataFrame],
) -> None:
    """UI של יצירת Pair Report והורדה כקובץ Markdown."""
    st.subheader("📄 Pair Report — יצוא דוח לזוג")

    report_md = _build_pair_markdown_report(
        sym_x=sym_x,
        sym_y=sym_y,
        date_start=date_start,
        date_end=date_end,
        z_last=z_last,
        corr_static=corr_static,
        hl=hl,
        vol_20=vol_20,
        hurst_val=hurst_val,
        regime_label=regime_label,
        spread_diag=spread_diag,
        legs_summary=legs_summary,
        regime_df=regime_df,
        trade_stats_df=trade_stats_df,
        mr_scenarios_df=mr_scenarios_df,
        shock_scenarios_df=shock_scenarios_df,
    )

    st.markdown("**תצוגה מקדימה (Markdown):**")
    st.code(report_md, language="markdown")

    file_name = f"pair_report_{sym_x}_{sym_y}.md"
    st.download_button(
        label="💾 הורד Pair Report (.md)",
        data=report_md.encode("utf-8"),
        file_name=file_name,
        mime="text/markdown",
        key="pair_report_download_btn",
    )
