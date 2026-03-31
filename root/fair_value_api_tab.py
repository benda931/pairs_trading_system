# -*- coding: utf-8 -*-
"""
fair_value_api_tab.py — Fair Value Engine / Optimizer / Advisor API Lab (v2)
=============================================================================

טאב Streamlit מבודד שעובד רק מול ה-HTTP API המקומי שלך (root/api_server.py).

מצבים:
- ⚙️ Engine (/engine/run)   — מריץ את FairValueEngine ומראה תוצאות + גרפים.
- 🧠 Advisor (/advisor/run) — מריץ Engine בפנים, מנתח את ה-universe ומחזיר עצות לשיפור.
- 🧬 Optimizer (/optimizer/run) — מריץ optimize_fair_value ומציג את bestParams.

מקורות דאטה:
- Demo data (סימולציה מובנית).
- העלאת CSV (תאריכים + טיקרים).
- JSON גולמי.
- טעינת JSON שמור (payload previously downloaded).

תוספות בגרסה זו:
- כפתור Download JSON לכל payload (Engine / Advisor / Optimizer).
- אפשרות לטעון payload מקובץ JSON.
- היסטוריית ריצות Advisor + השוואת שתי ריצות (KPIs + הבדל עצות).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import itertools
import json

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from root.optimization_tab import api_optimize_pair, api_optimize_pairs_batch
try:
    from core.app_context import AppContext
except Exception:
    AppContext = None  # fallback אם אין / בזמן בדיקות

import logging

logger = logging.getLogger(__name__)

# =====================================
#  הגדרות בסיס ל-API
# =====================================
def _get_app_ctx() -> Any:
    """
    מנסה להחזיר AppContext אחד:

    סדר עדיפויות:
    1. st.session_state["app_ctx"] אם קיים.
    2. AppContext.get_global() אם הפונקציה קיימת.
    3. אחרת → None.
    """
    # 1. מה-session_state (dashoard.py בד"כ שם שם את ה-app_ctx)
    ctx = None
    try:
        ctx = st.session_state.get("app_ctx")
    except Exception:
        ctx = None

    # 2. AppContext.get_global() אם עדיין אין
    if ctx is None and AppContext is not None:
        try:
            if hasattr(AppContext, "get_global"):
                ctx = AppContext.get_global()
        except Exception:
            ctx = None

    return ctx


def _get_fair_value_config():
    """
    מנסה להחזיר FairValueAPIConfig מתוך ה-AppContext.
    מחזיר None אם אין או אם disabled.
    """
    ctx = _get_app_ctx()
    if ctx is None:
        return None

    cfg = getattr(ctx, "fair_value_api", None)
    # יכול להיות שזה dict או Pydantic; נוודא שהוא לא ריק
    if cfg is None:
        return None

    try:
        # אם זה מודל Pydantic — יש לו dict / model_dump; אם dataclass — נניח שיש attr.
        is_enabled = getattr(cfg, "is_enabled", getattr(cfg, "enabled", False))
        if not is_enabled:
            return None
    except Exception:
        return None

    return cfg

DEFAULT_API_BASE = "http://localhost:8000"

def _resolve_api_base() -> str:
    """
    מחזיר את בסיס ה-API לשירות ה-Fair Value.

    אם יש secrets.toml עם FAIR_VALUE_API_URL → ישתמש בו.
    אם אין secrets בכלל → נשתמש ב-DEFAULT_API_BASE בלי להפיל את הטאב.
    """
    try:
        return st.secrets.get("FAIR_VALUE_API_URL", DEFAULT_API_BASE)
    except StreamlitSecretNotFoundError:
        logger.info(
            "FairValue: no secrets.toml found; using DEFAULT_API_BASE=%s",
            DEFAULT_API_BASE,
        )
        return DEFAULT_API_BASE
    except Exception as exc:
        logger.warning(
            "FairValue: failed to read FAIR_VALUE_API_URL from secrets (%s); "
            "using DEFAULT_API_BASE=%s",
            exc,
            DEFAULT_API_BASE,
        )
        return DEFAULT_API_BASE
# =====================================
#  Helpers — HTTP
# =====================================

def _get_json(path: str, timeout: int = 5) -> Dict[str, Any]:
    """
    GET JSON עם שימוש ב-FairValueAPIConfig אם זמין (timeouts / headers / TLS).
    """
    base = _resolve_api_base()
    url = f"{base.rstrip('/')}/{path.lstrip('/')}"

    cfg = _get_fair_value_config()
    req_kwargs: Dict[str, Any] = {}

    if cfg is not None:
        # נשתמש ב-timeout הכללי / verify / headers
        try:
            req_kwargs.update(cfg.as_requests_kwargs())
        except Exception:
            # fallback: נבנה ידנית
            req_kwargs["timeout"] = getattr(cfg, "total_timeout_sec", timeout) or timeout
            req_kwargs["headers"] = getattr(cfg, "headers", {}) or {}
            req_kwargs["verify"] = getattr(cfg, "verify_tls", True)
    else:
        req_kwargs["timeout"] = timeout

    resp = requests.get(url, **req_kwargs)
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {data}")
    return data



def _post_json(path: str, payload: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    base = _resolve_api_base()
    url = f"{base.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {data}")
    return data



def _check_health() -> Tuple[bool, Dict[str, Any]]:
    try:
        data = _get_json("/health", timeout=3)
        return True, data
    except Exception as e:
        return False, {"error": str(e)}


# =====================================
#  Helpers — Data construction
# =====================================

def _build_demo_prices(n_days: int = 252) -> Dict[str, Any]:
    """
    בונה price matrix פשוטה (3 טיקרים, N ימים) לסביבת דמו.

    - שלושה סימבולים: SPY, QQQ, IWM.
    - מהלך רנדומלי עם קורלציה גבוהה בין SPY/QQQ, קצת שונה ל-IWM.
    """
    end = datetime.now(timezone.utc).date()
    dates = [end - timedelta(days=i) for i in range(n_days)]
    dates = sorted(dates)

    rng = np.random.default_rng(42)
    base_r = rng.normal(0, 0.01, size=n_days)
    r_spy = base_r
    r_qqq = base_r + rng.normal(0, 0.004, size=n_days)
    r_iwm = base_r * 0.7 + rng.normal(0, 0.008, size=n_days)

    p_spy = 100 * np.exp(np.cumsum(r_spy))
    p_qqq = 100 * np.exp(np.cumsum(r_qqq))
    p_iwm = 100 * np.exp(np.cumsum(r_iwm))

    time_index = [datetime.combine(d, datetime.min.time()) for d in dates]

    prices_wide = {
        "SPY": p_spy.round(4).tolist(),
        "QQQ": p_qqq.round(4).tolist(),
        "IWM": p_iwm.round(4).tolist(),
    }

    pairs = [
        ["SPY", "QQQ"],
        ["SPY", "IWM"],
        ["QQQ", "IWM"],
    ]

    return {
        "timeIndex": time_index,
        "pricesWide": prices_wide,
        "pairs": pairs,
    }


def _build_from_csv(uploaded_file, key_prefix: str) -> Optional[Dict[str, Any]]:
    """
    קריאת CSV מהמשתמש:
    - עמודת תאריך (נבחרת על ידו)
    - עמודות מחירים (טיקרים)
    - בחירת זוגות לניתוח
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"קריאת ה-CSV נכשלה: {e}")
        return None

    if df.empty:
        st.error("קובץ ה-CSV ריק.")
        return None

    st.markdown("**תצוגה מקדימה של הדאטה:**")
    st.dataframe(df.head(), use_container_width=True)

    cols = list(df.columns)
    date_col = st.selectbox(
        "עמודת תאריך:",
        options=cols,
        index=0,
        key=f"{key_prefix}_date_col",
    )

    value_cols = [c for c in cols if c != date_col]
    if not value_cols:
        st.error("לא נמצאו עמודות מחירים (חוץ מעמודת התאריך).")
        return None

    selected_symbols = st.multiselect(
        "בחר טיקרים (עמודות מחירים):",
        options=value_cols,
        default=value_cols[: min(5, len(value_cols))],
        key=f"{key_prefix}_symbols",
    )

    if len(selected_symbols) < 2:
        st.info("צריך לפחות 2 טיקרים כדי לבנות זוגות.")
        return None

    all_pairs = list(itertools.combinations(selected_symbols, 2))
    default_pairs = all_pairs[: min(10, len(all_pairs))]

    selected_pairs = st.multiselect(
        "בחר זוגות לניתוח:",
        options=[f"{a}/{b}" for (a, b) in all_pairs],
        default=[f"{a}/{b}" for (a, b) in default_pairs],
        key=f"{key_prefix}_pairs",
    )

    if not selected_pairs:
        st.info("המערכת לא תריץ כלום בלי לפחות זוג אחד.")
        return None

    try:
        idx = pd.to_datetime(df[date_col], errors="coerce")
    except Exception as e:
        st.error(f"כשל בהמרת עמודת התאריך ל-datetime: {e}")
        return None

    mask = idx.notna()
    if not mask.any():
        st.error("עמודת התאריך לא הצליחה לעבור המרה ל-datetime.")
        return None

    df = df.loc[mask].reset_index(drop=True)
    idx = idx.loc[mask]

    prices_wide: Dict[str, List[float]] = {}
    for sym in selected_symbols:
        try:
            series = pd.to_numeric(df[sym], errors="coerce")
        except Exception:
            continue
        prices_wide[sym] = series.ffill().bfill().tolist()

    time_index = [dt.to_pydatetime() for dt in idx]

    pairs: List[List[str]] = []
    for s in selected_pairs:
        a, b = s.split("/")
        pairs.append([a.strip(), b.strip()])

    return {
        "timeIndex": time_index,
        "pricesWide": prices_wide,
        "pairs": pairs,
    }


def _config_overrides_ui(key_prefix: str) -> Dict[str, Any]:
    """
    UI בסיסי ל-configOverrides ל-Engine.
    מחזיר מילון שמתאים ל-EngineConfigOverridesModel ב-API.
    """
    st.markdown("#### ⚙️ הגדרות Engine (configOverrides)")

    col1, col2, col3 = st.columns(3)

    with col1:
        window = st.number_input(
            "window (חישוב fair value)",
            min_value=20,
            max_value=252 * 3,
            value=126,
            step=10,
            key=f"{key_prefix}_window",
        )
        min_overlap = st.number_input(
            "min_overlap",
            min_value=20,
            max_value=int(window),
            value=min(60, int(window)),
            step=5,
            key=f"{key_prefix}_min_overlap",
        )
        log_mode = st.checkbox(
            "log_mode (log prices)",
            value=True,
            key=f"{key_prefix}_log_mode",
        )

    with col2:
        z_in = st.number_input(
            "z_in (כניסה לפוזיציה)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            key=f"{key_prefix}_z_in",
        )
        z_out = st.number_input(
            "z_out (יציאה מפוזיציה)",
            min_value=0.5,
            max_value=6.0,
            value=0.5,
            step=0.1,
            key=f"{key_prefix}_z_out",
        )
        use_winsor = st.checkbox(
            "use_winsor_for_z",
            value=True,
            key=f"{key_prefix}_use_winsor",
        )

    with col3:
        target_vol = st.number_input(
            "target_vol_ann (יעד תנודתיות שנתית)",
            min_value=0.01,
            max_value=1.0,
            value=0.15,
            step=0.01,
            key=f"{key_prefix}_target_vol",
        )
        kelly_fraction = st.number_input(
            "kelly_fraction",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key=f"{key_prefix}_kelly_fraction",
        )
        max_leverage = st.number_input(
            "max_leverage",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.5,
            key=f"{key_prefix}_max_leverage",
        )

    overrides: Dict[str, Any] = {
        "window": int(window),
        "min_overlap": int(min_overlap),
        "log_mode": bool(log_mode),
        "z_in": float(z_in),
        "z_out": float(z_out),
        "use_winsor_for_z": bool(use_winsor),
        "target_vol_ann": float(target_vol),
        "kelly_fraction": float(kelly_fraction),
        "max_leverage": float(max_leverage),
    }
    return overrides


def _normalize_pair_field(rows: List[Dict[str, Any]]) -> None:
    """
    מנרמל את השדה pair לתצורה "A/B" כדי שיהיה נוח להציג ב-DataFrame.
    עובד in-place.
    """
    for r in rows:
        pair_val = r.get("pair")
        if isinstance(pair_val, dict) and "__root__" in pair_val:
            root = r["pair"]["__root__"]
            if isinstance(root, (list, tuple)) and len(root) == 2:
                r["pair"] = f"{root[0]}/{root[1]}"
        elif isinstance(pair_val, (list, tuple)) and len(pair_val) == 2:
            r["pair"] = f"{pair_val[0]}/{pair_val[1]}"


def _payload_download_ui(payload: Dict[str, Any], label: str, key_prefix: str) -> None:
    """
    מציג כפתור Download ל-payload JSON.
    """
    b = json.dumps(payload, indent=2, default=str).encode("utf-8")
    st.download_button(
        label=label,
        data=b,
        file_name=f"{key_prefix}_payload_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key=f"{key_prefix}_download_btn",
    )


def _payload_upload_json_ui(key_prefix: str) -> Optional[Dict[str, Any]]:
    """
    מאפשר לטעון payload מ-JSON שמור (מה-Download).
    """
    uploaded = st.file_uploader(
        "או טען payload מקובץ JSON שמור:",
        type=["json"],
        key=f"{key_prefix}_upload_json",
    )
    if uploaded is None:
        return None

    try:
        raw = uploaded.read().decode("utf-8")
        data = json.loads(raw)
        st.success("טענת payload מ-JSON.")
        with st.expander("🔍 JSON טעון", expanded=False):
            st.json(data)
        return data
    except Exception as e:
        st.error(f"כשל בקריאת JSON: {e}")
        return None


# =====================================
#  Analytics helpers (Engine results)
# =====================================

# All 26 engine output columns with labels and descriptions
_ENGINE_COLS_META: Dict[str, Dict[str, str]] = {
    "pair":             {"label": "Pair",          "group": "id"},
    "window":           {"label": "Window",        "group": "config"},
    "action":           {"label": "Action",        "group": "signal"},
    "mispricing":       {"label": "Mispricing",    "group": "edge"},
    "vol_adj_mispricing": {"label": "Vol-adj Misprice", "group": "edge"},
    "zscore":           {"label": "Z-score",       "group": "signal"},
    "band_p95":         {"label": "Band p95",      "group": "band"},
    "band_upper":       {"label": "Band upper",    "group": "band"},
    "band_lower":       {"label": "Band lower",    "group": "band"},
    "halflife":         {"label": "Half-life (d)", "group": "quality"},
    "rolling_corr":     {"label": "Rolling Corr",  "group": "quality"},
    "distance_corr":    {"label": "Dist Corr",     "group": "quality"},
    "adf_p":            {"label": "ADF p-val",     "group": "stat"},
    "residual_adf_p":   {"label": "Resid ADF p",  "group": "stat"},
    "is_coint":         {"label": "Cointegrated",  "group": "stat"},
    "y_fair":           {"label": "Fair Value Y",  "group": "valuation"},
    "target_pos_units": {"label": "Target Pos",   "group": "sizing"},
    "cost_spread_units": {"label": "Cost Spread",  "group": "sizing"},
    "rp_weight":        {"label": "RP Weight",     "group": "sizing"},
    "sr_net":           {"label": "Sharpe net",    "group": "perf"},
    "psr_net":          {"label": "PSR net",       "group": "perf"},
    "dsr_net":          {"label": "DSR net",       "group": "perf"},
    "avg_hold_days":    {"label": "Avg hold (d)",  "group": "perf"},
    "turnover_est":     {"label": "Turnover est",  "group": "perf"},
    "net_edge_z":       {"label": "Net edge Z",    "group": "edge"},
    "reason":           {"label": "Reason",        "group": "signal"},
}

_STAT_QUALITY_THRESHOLDS = {
    "adf_p":            {"good": 0.05,  "warn": 0.10,  "direction": "lower"},
    "residual_adf_p":   {"good": 0.05,  "warn": 0.10,  "direction": "lower"},
    "rolling_corr":     {"good": 0.65,  "warn": 0.50,  "direction": "higher"},
    "distance_corr":    {"good": 0.50,  "warn": 0.35,  "direction": "higher"},
    "halflife":         {"good_lo": 2,  "good_hi": 60, "warn_hi": 120, "direction": "range"},
    "net_edge_z":       {"good": 1.0,   "warn": 0.0,   "direction": "higher"},
    "dsr_net":          {"good": 0.5,   "warn": 0.0,   "direction": "higher"},
}


def _stat_status(col: str, val: float) -> str:
    """Returns 'good', 'warn', or 'bad' for a given metric value."""
    t = _STAT_QUALITY_THRESHOLDS.get(col)
    if t is None or val is None or (isinstance(val, float) and np.isnan(val)):
        return "neutral"
    d = t.get("direction", "higher")
    if d == "range":
        if t["good_lo"] <= val <= t["good_hi"]:
            return "good"
        if val <= t.get("warn_hi", 120):
            return "warn"
        return "bad"
    if d == "higher":
        if val >= t["good"]:
            return "good"
        if val >= t["warn"]:
            return "warn"
        return "bad"
    # lower is better
    if val <= t["good"]:
        return "good"
    if val <= t["warn"]:
        return "warn"
    return "bad"


def _status_color(status: str) -> str:
    return {"good": "#4CAF50", "warn": "#FF9800", "bad": "#F44336", "neutral": "#9E9E9E"}.get(status, "#9E9E9E")


def _render_statistical_scorecard(df: pd.DataFrame) -> None:
    """
    Renders a per-pair statistical quality scorecard.
    Covers ADF p-val, half-life, rolling corr, distance corr, cointegration,
    net edge Z, and the action signal for each pair.
    """
    st.markdown("### 🧪 Statistical Quality Scorecard")
    st.caption(
        "Green = passes quality threshold · Orange = marginal · Red = fails. "
        "Half-life target: 2–60 days (mean-reversion). ADF p-val < 0.05 preferred."
    )

    score_cols = [c for c in ["adf_p", "residual_adf_p", "rolling_corr", "distance_corr",
                               "halflife", "net_edge_z", "dsr_net", "is_coint", "action"] if c in df.columns]
    if not score_cols:
        st.info("No quality-score columns available from the engine response.")
        return

    # Colour-coded aggregate per pair
    try:
        import plotly.graph_objects as _go_sc
        stat_dim_cols = [c for c in ["adf_p", "residual_adf_p", "rolling_corr", "distance_corr", "halflife", "net_edge_z", "dsr_net"] if c in df.columns]

        pairs_list = df["pair"].tolist() if "pair" in df.columns else [str(i) for i in df.index]
        n_pairs = len(pairs_list)

        # Build score matrix
        z_vals = []
        hover_texts = []
        for col in stat_dim_cols:
            col_z = []
            col_h = []
            for _, row in df.iterrows():
                val = row.get(col)
                try:
                    val_f = float(val)
                except Exception:
                    val_f = float("nan")
                status = _stat_status(col, val_f)
                score = {"good": 1.0, "warn": 0.5, "bad": 0.0, "neutral": 0.5}.get(status, 0.5)
                col_z.append(score)
                col_h.append(f"{col}={val_f:.4f} [{status}]" if not np.isnan(val_f) else f"{col}=N/A")
            z_vals.append(col_z)
            hover_texts.append(col_h)

        col_labels = [_ENGINE_COLS_META.get(c, {}).get("label", c) for c in stat_dim_cols]
        pair_labels = pairs_list[:min(25, n_pairs)]
        z_trimmed = [row[:len(pair_labels)] for row in z_vals]
        hover_trimmed = [row[:len(pair_labels)] for row in hover_texts]

        fig_hm = _go_sc.Figure(data=_go_sc.Heatmap(
            z=z_trimmed,
            x=pair_labels,
            y=col_labels,
            colorscale="RdYlGn",
            zmin=0, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z_trimmed],
            texttemplate="%{text}",
            hovertext=hover_trimmed,
            hoverinfo="text",
            showscale=True,
            colorbar=dict(title="Quality", tickvals=[0, 0.5, 1], ticktext=["Fail", "Warn", "Pass"]),
        ))
        fig_hm.update_layout(
            title="Statistical Quality Matrix (1=Pass, 0.5=Marginal, 0=Fail)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ECEFF1"), height=max(250, len(stat_dim_cols) * 45 + 80),
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    except Exception as _hm_err:
        st.caption(f"Quality heatmap unavailable: {_hm_err}")

    # Metric summary table with colour flags
    try:
        summary_rows = []
        for col in [c for c in ["adf_p", "residual_adf_p", "rolling_corr", "distance_corr",
                                  "halflife", "net_edge_z", "dsr_net"] if c in df.columns]:
            series = pd.to_numeric(df[col], errors="coerce")
            n_good = int(series.apply(lambda v: _stat_status(col, v) == "good").sum())
            n_warn = int(series.apply(lambda v: _stat_status(col, v) == "warn").sum())
            n_bad  = int(series.apply(lambda v: _stat_status(col, v) == "bad").sum())
            summary_rows.append({
                "Metric": _ENGINE_COLS_META.get(col, {}).get("label", col),
                "Mean": round(float(series.mean()), 4) if series.notna().any() else None,
                "Median": round(float(series.median()), 4) if series.notna().any() else None,
                "Min": round(float(series.min()), 4) if series.notna().any() else None,
                "Max": round(float(series.max()), 4) if series.notna().any() else None,
                "Pass": n_good,
                "Warn": n_warn,
                "Fail": n_bad,
            })
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            st.markdown("**Quality metric summary across all pairs:**")
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
    except Exception as _sum_err:
        st.caption(f"Summary table unavailable: {_sum_err}")

    # Cointegration + action breakdown
    col_coint, col_act = st.columns(2)
    with col_coint:
        if "is_coint" in df.columns:
            try:
                coint_counts = df["is_coint"].value_counts()
                n_coint = int(coint_counts.get(True, 0))
                n_not = int(coint_counts.get(False, 0))
                st.metric("Cointegrated pairs", f"{n_coint} / {n_coint + n_not}", delta=f"{n_coint/(n_coint+n_not)*100:.0f}%" if (n_coint+n_not) > 0 else "0%")
            except Exception:
                pass
    with col_act:
        if "action" in df.columns:
            try:
                act_counts = df["action"].value_counts().reset_index()
                act_counts.columns = ["Action", "Count"]
                st.markdown("**Signal actions:**")
                st.dataframe(act_counts, use_container_width=True, hide_index=True)
            except Exception:
                pass


def _render_engine_multi_charts(df: pd.DataFrame) -> None:
    """
    Multi-panel analytics: Z-score vs mispricing scatter, performance bar,
    rolling-corr vs half-life scatter, and full metrics heatmap.
    """
    import plotly.graph_objects as _go_mc
    import plotly.express as _px_mc

    st.markdown("### 📈 Advanced Analytics — All Engine Dimensions")

    tab_scatter, tab_bar, tab_heatmap, tab_perf = st.tabs([
        "🔵 Z vs Edge", "📊 Top-N Bar", "🟩 Metrics Heatmap", "📉 Performance"
    ])

    # --- Tab 1: Z-score vs net_edge_z scatter ---
    with tab_scatter:
        try:
            z_col  = next((c for c in ["zscore", "vol_adj_mispricing"] if c in df.columns), None)
            e_col  = next((c for c in ["net_edge_z", "mispricing"] if c in df.columns), None)
            if z_col and e_col:
                df_sc = df[["pair", z_col, e_col]].copy()
                df_sc[z_col] = pd.to_numeric(df_sc[z_col], errors="coerce")
                df_sc[e_col] = pd.to_numeric(df_sc[e_col], errors="coerce")
                df_sc = df_sc.dropna()

                color_col = "action" if "action" in df.columns else None
                df_sc2 = df_sc.copy()
                if color_col:
                    df_sc2[color_col] = df[color_col].values[:len(df_sc2)]

                fig_sc = _px_mc.scatter(
                    df_sc2, x=z_col, y=e_col,
                    hover_data=["pair"],
                    color=color_col,
                    labels={z_col: _ENGINE_COLS_META.get(z_col, {}).get("label", z_col),
                            e_col: _ENGINE_COLS_META.get(e_col, {}).get("label", e_col)},
                    title=f"{_ENGINE_COLS_META.get(z_col,{}).get('label',z_col)} vs {_ENGINE_COLS_META.get(e_col,{}).get('label',e_col)}",
                    template="plotly_dark",
                )
                fig_sc.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_sc.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_sc.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sc, use_container_width=True)

                # Correlation between z-score and edge
                corr_ze = df_sc[[z_col, e_col]].corr().iloc[0, 1]
                st.caption(f"Pearson r({z_col}, {e_col}) = {corr_ze:.3f}")
            else:
                st.info("Z-score and edge columns not found in engine output.")

            # Second scatter: rolling_corr vs halflife
            if "rolling_corr" in df.columns and "halflife" in df.columns:
                st.markdown("**Rolling Correlation vs Half-Life**")
                df_hl = df[["pair", "rolling_corr", "halflife"]].copy()
                df_hl["rolling_corr"] = pd.to_numeric(df_hl["rolling_corr"], errors="coerce")
                df_hl["halflife"] = pd.to_numeric(df_hl["halflife"], errors="coerce")
                df_hl = df_hl.dropna()
                fig_hl = _px_mc.scatter(
                    df_hl, x="halflife", y="rolling_corr",
                    hover_data=["pair"],
                    title="Mean-Reversion Quality: Half-Life vs Rolling Correlation",
                    labels={"halflife": "Half-life (days)", "rolling_corr": "Rolling Corr"},
                    template="plotly_dark",
                    color_discrete_sequence=["#42A5F5"],
                )
                # Add quality zones
                fig_hl.add_vrect(x0=2, x1=60, fillcolor="rgba(76,175,80,0.07)", line_width=0, annotation_text="Optimal HL zone")
                fig_hl.add_hrect(y0=0.60, y1=1.0, fillcolor="rgba(76,175,80,0.07)", line_width=0)
                fig_hl.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hl, use_container_width=True)
        except Exception as _sc_err:
            st.caption(f"Scatter chart unavailable: {_sc_err}")

    # --- Tab 2: Top-N bar chart ---
    with tab_bar:
        try:
            perf_options = [c for c in ["dsr_net", "psr_net", "sr_net", "net_edge_z"] if c in df.columns]
            if perf_options:
                bar_metric = st.selectbox("Metric for Top-N bar:", options=perf_options, key="fv_bar_metric")
                top_n_bar = st.slider("Top-N pairs:", 5, min(50, len(df)), min(15, len(df)), key="fv_bar_topn")
                df_bar = df[["pair", bar_metric]].copy()
                df_bar[bar_metric] = pd.to_numeric(df_bar[bar_metric], errors="coerce")
                df_bar = df_bar.dropna().sort_values(bar_metric, ascending=False).head(top_n_bar)

                colors = ["#4CAF50" if v > 0 else "#F44336" for v in df_bar[bar_metric]]
                fig_bar = _go_mc.Figure(_go_mc.Bar(
                    x=df_bar["pair"], y=df_bar[bar_metric],
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in df_bar[bar_metric]],
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    title=f"Top-{top_n_bar} Pairs by {_ENGINE_COLS_META.get(bar_metric,{}).get('label',bar_metric)}",
                    height=420, xaxis_tickangle=-35,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ECEFF1"),
                )
                fig_bar.add_hline(y=0, line_color="white", line_dash="dot", opacity=0.4)
                st.plotly_chart(fig_bar, use_container_width=True)

                # Show also worst pairs
                df_worst = df[["pair", bar_metric]].copy()
                df_worst[bar_metric] = pd.to_numeric(df_worst[bar_metric], errors="coerce")
                df_worst = df_worst.dropna().sort_values(bar_metric, ascending=True).head(5)
                if not df_worst.empty:
                    st.markdown("**Bottom 5 pairs:**")
                    st.dataframe(df_worst, use_container_width=True, hide_index=True)
            else:
                st.info("No performance columns found.")
        except Exception as _bar_err:
            st.caption(f"Bar chart unavailable: {_bar_err}")

    # --- Tab 3: Full metrics heatmap ---
    with tab_heatmap:
        try:
            num_cols = [c for c in df.columns if c not in ("pair", "action", "reason", "window", "is_coint") and
                        pd.api.types.is_numeric_dtype(df[c])]
            if num_cols and "pair" in df.columns:
                hm_cols = st.multiselect(
                    "Columns to include in heatmap:",
                    options=num_cols,
                    default=num_cols[:min(10, len(num_cols))],
                    key="fv_hm_cols",
                )
                if hm_cols:
                    df_hm = df[["pair"] + hm_cols].set_index("pair")
                    df_hm = df_hm.apply(pd.to_numeric, errors="coerce")
                    # Normalize per column for visual comparison
                    df_norm = df_hm.copy()
                    for c in hm_cols:
                        col_min, col_max = df_hm[c].min(), df_hm[c].max()
                        rng = col_max - col_min
                        if rng > 0:
                            df_norm[c] = (df_hm[c] - col_min) / rng
                        else:
                            df_norm[c] = 0.5

                    fig_hm2 = _go_mc.Figure(_go_mc.Heatmap(
                        z=df_norm.T.values.tolist(),
                        x=df_norm.index.tolist(),
                        y=[_ENGINE_COLS_META.get(c, {}).get("label", c) for c in df_norm.columns],
                        colorscale="RdYlGn", zmin=0, zmax=1,
                        text=[[f"{df_hm.iloc[ci][ri]:.3f}" for ci in range(len(df_hm))] for ri, _ in enumerate(df_norm.columns)],
                        texttemplate="%{text}",
                        hoverinfo="text",
                        showscale=True,
                        colorbar=dict(title="Norm."),
                    ))
                    fig_hm2.update_layout(
                        title="All Metrics Heatmap (column-normalized 0→1)",
                        height=max(300, len(hm_cols) * 30 + 100),
                        xaxis_tickangle=-30,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ECEFF1"),
                    )
                    st.plotly_chart(fig_hm2, use_container_width=True)
                    st.caption("Raw values shown in cells. Colors are column-normalized for cross-pair comparison.")
            else:
                st.info("No numeric columns available for heatmap.")
        except Exception as _hm2_err:
            st.caption(f"Metrics heatmap unavailable: {_hm2_err}")

    # --- Tab 4: Performance distribution ---
    with tab_perf:
        try:
            perf_cols = [c for c in ["sr_net", "psr_net", "dsr_net", "turnover_est", "avg_hold_days", "rp_weight"] if c in df.columns]
            if perf_cols:
                chosen = st.multiselect("Metrics to plot:", perf_cols, default=perf_cols[:min(3, len(perf_cols))], key="fv_perf_cols")
                for pc in chosen:
                    series = pd.to_numeric(df[pc], errors="coerce").dropna()
                    if series.empty:
                        continue
                    fig_hist = _go_mc.Figure()
                    fig_hist.add_trace(_go_mc.Histogram(
                        x=series, nbinsx=20,
                        name=_ENGINE_COLS_META.get(pc, {}).get("label", pc),
                        marker_color="#42A5F5", opacity=0.75,
                    ))
                    fig_hist.add_vline(x=float(series.mean()), line_color="#FF9800", line_dash="dash",
                                       annotation_text=f"Mean={series.mean():.3f}", annotation_position="top right")
                    fig_hist.update_layout(
                        title=f"Distribution: {_ENGINE_COLS_META.get(pc,{}).get('label',pc)}",
                        height=300, bargap=0.05,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ECEFF1"),
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No performance columns found.")
        except Exception as _perf_err:
            st.caption(f"Performance charts unavailable: {_perf_err}")


def _render_sensitivity_sweep(df: pd.DataFrame) -> None:
    """
    Sensitivity analysis: show how engine output KPIs vary across pairs
    for different z_in / z_out / window settings already embedded in the results.
    When multiple windows are present (batch), plots window vs KPI.
    Also provides a manual parameter exploration UI.
    """
    st.markdown("### 🔭 Sensitivity Analysis")
    st.caption("Explore how key metrics vary with z-threshold and window settings across the universe.")

    try:
        import plotly.graph_objects as _go_sens

        # If engine returned multi-window results, use them
        if "window" in df.columns and df["window"].nunique() > 1:
            st.markdown("**Multi-window results detected — plotting KPI vs window:**")
            kpi_sens = st.selectbox("KPI:", [c for c in ["dsr_net", "net_edge_z", "sr_net"] if c in df.columns], key="fv_sens_kpi")
            df_ws = df[["window", kpi_sens]].copy()
            df_ws[kpi_sens] = pd.to_numeric(df_ws[kpi_sens], errors="coerce")
            df_ws_agg = df_ws.groupby("window")[kpi_sens].agg(["mean", "median", "std"]).reset_index()
            fig_w = _go_sens.Figure()
            fig_w.add_trace(_go_sens.Scatter(
                x=df_ws_agg["window"], y=df_ws_agg["mean"],
                name="Mean", mode="lines+markers", line=dict(color="#42A5F5"),
            ))
            fig_w.add_trace(_go_sens.Scatter(
                x=df_ws_agg["window"], y=df_ws_agg["median"],
                name="Median", mode="lines+markers", line=dict(color="#FF9800", dash="dash"),
            ))
            fig_w.update_layout(
                title=f"{kpi_sens} vs Window size", height=350,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ECEFF1"),
            )
            st.plotly_chart(fig_w, use_container_width=True)
        else:
            # Cross-section sensitivity: show range of KPI across pairs + basic stats
            st.markdown("**Cross-sectional KPI sensitivity (single window run):**")
            kpi_list = [c for c in ["dsr_net", "net_edge_z", "sr_net", "halflife", "rolling_corr"] if c in df.columns]
            if kpi_list:
                df_kpis = df[kpi_list].apply(pd.to_numeric, errors="coerce")
                stats = df_kpis.describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]].T
                stats.index = [_ENGINE_COLS_META.get(c, {}).get("label", c) for c in stats.index]
                stats = stats.round(4)
                st.dataframe(stats, use_container_width=True)

                # Boxplot per KPI
                fig_box = _go_sens.Figure()
                for kpi_c in kpi_list:
                    series = pd.to_numeric(df[kpi_c], errors="coerce").dropna()
                    fig_box.add_trace(_go_sens.Box(
                        y=series, name=_ENGINE_COLS_META.get(kpi_c, {}).get("label", kpi_c),
                        boxmean="sd", marker_color="#42A5F5", jitter=0.3, pointpos=-1.5,
                    ))
                fig_box.update_layout(
                    title="KPI distribution across pairs", height=380,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ECEFF1"), showlegend=False,
                )
                st.plotly_chart(fig_box, use_container_width=True)

        # Manual z_in / z_out exploration (informational — shows where pairs would rank)
        st.markdown("**Manual threshold exploration:**")
        st.caption("Select z_in threshold to see how many pairs currently have |zscore| > z_in (entry signal).")
        if "zscore" in df.columns:
            z_thresh = st.slider("z_in threshold:", 0.5, 4.0, 2.0, 0.25, key="fv_sens_zthresh")
            z_vals = pd.to_numeric(df["zscore"], errors="coerce").abs()
            n_active = int((z_vals >= z_thresh).sum())
            n_total = int(z_vals.notna().sum())
            pct = n_active / n_total * 100 if n_total > 0 else 0.0
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Pairs with |Z| ≥ z_in", f"{n_active} / {n_total}")
            col_b.metric("Active fraction", f"{pct:.1f}%")
            if "net_edge_z" in df.columns:
                mask = z_vals >= z_thresh
                avg_edge = float(pd.to_numeric(df.loc[mask, "net_edge_z"], errors="coerce").mean()) if mask.any() else float("nan")
                col_c.metric("Avg net_edge_z (active)", f"{avg_edge:.3f}" if not np.isnan(avg_edge) else "N/A")
        else:
            st.info("zscore column not in engine output.")
    except Exception as _sens_err:
        st.caption(f"Sensitivity analysis unavailable: {_sens_err}")


def _render_data_quality_validator(payload: Dict[str, Any]) -> None:
    """
    Validates the input price data for gaps, stale prices, short history,
    and missing symbols — before running the engine.
    """
    st.markdown("### 🔍 Data Quality Validator")
    prices_wide: Dict[str, Any] = payload.get("pricesWide", {})
    time_index: List[Any] = payload.get("timeIndex", [])

    if not prices_wide or not time_index:
        st.info("No data to validate — load a dataset first.")
        return

    try:
        n_dates = len(time_index)
        st.markdown(f"**Time index:** {n_dates} rows · First: `{time_index[0]}` · Last: `{time_index[-1]}`")

        rows = []
        for sym, vals in prices_wide.items():
            arr = np.array(vals, dtype=float)
            n_nan  = int(np.isnan(arr).sum())
            n_zero = int((arr == 0).sum())
            n_neg  = int((arr < 0).sum())
            pct_miss = round(n_nan / len(arr) * 100, 2) if len(arr) > 0 else 0.0
            # stale detection: runs of identical prices
            if len(arr) > 1:
                diffs = np.diff(arr)
                max_stale_run = int(max(
                    (sum(1 for _ in g) + 1 for k, g in itertools.groupby(diffs) if k == 0),
                    default=0
                ))
            else:
                max_stale_run = 0

            status = "✅ OK"
            if pct_miss > 10:
                status = "❌ High missing"
            elif pct_miss > 2:
                status = "⚠️ Some gaps"
            elif max_stale_run > 5:
                status = "⚠️ Stale prices"
            elif n_zero > 0 or n_neg > 0:
                status = "⚠️ Zero/neg prices"

            rows.append({
                "Symbol": sym,
                "N obs": len(arr),
                "Missing %": pct_miss,
                "Zero": n_zero,
                "Negative": n_neg,
                "Max stale run": max_stale_run,
                "Status": status,
            })

        df_dq = pd.DataFrame(rows)
        st.dataframe(df_dq, use_container_width=True, hide_index=True)

        n_ok  = (df_dq["Status"] == "✅ OK").sum()
        n_warn = df_dq["Status"].str.startswith("⚠️").sum()
        n_err  = df_dq["Status"].str.startswith("❌").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Symbols OK", f"{n_ok} / {len(rows)}")
        c2.metric("Warnings", n_warn, delta=f"-{n_warn}" if n_warn else None, delta_color="inverse")
        c3.metric("Errors", n_err, delta=f"-{n_err}" if n_err else None, delta_color="inverse")

        # Date gap detection
        try:
            idx_parsed = pd.to_datetime(time_index, errors="coerce")
            idx_clean = idx_parsed.dropna().sort_values()
            if len(idx_clean) > 1:
                deltas = idx_clean.diff().dropna()
                max_gap = deltas.max()
                median_gap = deltas.median()
                if max_gap > pd.Timedelta(days=5):
                    st.warning(f"Date gap detected: max gap = {max_gap.days} days (median = {median_gap.days}d). Check for weekends/holidays or missing data.")
                else:
                    st.success(f"Date continuity OK · max gap: {max_gap.days}d · median: {median_gap.days}d")
        except Exception:
            pass

        # CSV export
        dq_csv = df_dq.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export data quality report (CSV)", dq_csv, "data_quality.csv", "text/csv",
                           key="fv_dq_download")
    except Exception as _dq_err:
        st.caption(f"Data quality validator error: {_dq_err}")


def _render_per_pair_detail(df: pd.DataFrame) -> None:
    """
    Per-pair expandable detail cards with all 26 engine output columns,
    colour-coded by quality thresholds.
    """
    st.markdown("### 🔎 Per-Pair Detail Cards")
    st.caption("Click a pair to see all engine output columns with quality annotations.")

    if df.empty or "pair" not in df.columns:
        st.info("No pair data available.")
        return

    pairs = df["pair"].tolist()
    selected_pairs = st.multiselect(
        "Select pairs to inspect:",
        options=pairs,
        default=pairs[:min(3, len(pairs))],
        key="fv_per_pair_select",
    )

    for pair_id in selected_pairs:
        row = df[df["pair"] == pair_id].iloc[0].to_dict()
        with st.expander(f"🔬 {pair_id}", expanded=True):
            group_order = ["signal", "stat", "quality", "edge", "band", "valuation", "sizing", "perf", "config", "id"]
            groups: Dict[str, List[tuple]] = {}
            for col, val in row.items():
                meta = _ENGINE_COLS_META.get(col, {})
                grp = meta.get("group", "other")
                groups.setdefault(grp, []).append((col, meta.get("label", col), val))

            for grp in group_order + ["other"]:
                fields = groups.get(grp, [])
                if not fields:
                    continue
                st.markdown(f"**{grp.upper()}**")
                ncols = 3
                field_chunks = [fields[i:i+ncols] for i in range(0, len(fields), ncols)]
                for chunk in field_chunks:
                    cols_ui = st.columns(ncols)
                    for ci, (col, label, val) in enumerate(chunk):
                        with cols_ui[ci]:
                            try:
                                val_f = float(val)
                                status = _stat_status(col, val_f)
                                color = _status_color(status)
                                st.markdown(
                                    f"<span style='color:{color};font-size:0.82em;'>{label}</span><br>"
                                    f"<span style='font-size:1.0em;font-weight:600;'>{val_f:.4f}</span>",
                                    unsafe_allow_html=True,
                                )
                            except (TypeError, ValueError):
                                st.markdown(
                                    f"<span style='color:#9E9E9E;font-size:0.82em;'>{label}</span><br>"
                                    f"<span style='font-size:1.0em;'>{val}</span>",
                                    unsafe_allow_html=True,
                                )

    # CSV export for all pairs
    try:
        csv_all = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export all pairs (CSV)", csv_all, "engine_results.csv", "text/csv",
                           key="fv_all_csv_export")
    except Exception:
        pass


# =====================================
#  Engine Section
# =====================================

def _render_engine_section() -> None:
    st.subheader("⚙️ Engine – /engine/run")

    dataset_mode = st.radio(
        "מקור דאטה ל-Engine:",
        options=["Demo", "Upload CSV", "Raw JSON", "Load saved JSON"],
        horizontal=True,
        key="fv_engine_dataset_mode",
    )

    payload: Optional[Dict[str, Any]] = None

    if dataset_mode == "Demo":
        base = _build_demo_prices()
        overrides = _config_overrides_ui("engine_demo")
        payload = {
            "timeIndex": [dt.isoformat() for dt in base["timeIndex"]],
            "pricesWide": base["pricesWide"],
            "pairs": base["pairs"],
            "configOverrides": overrides,
        }
        with st.expander("🔍 Preview payload (Demo + overrides)", expanded=False):
            st.json(payload)
        _payload_download_ui(payload, "⬇️ הורד Engine payload (Demo)", "engine_demo")

    elif dataset_mode == "Upload CSV":
        uploaded = st.file_uploader(
            "העלה קובץ מחירים בפורמט CSV (Date + טיקרים):",
            type=["csv"],
            key="fv_engine_csv",
        )
        if uploaded is not None:
            base = _build_from_csv(uploaded, key_prefix="engine_csv")
            if base is not None:
                overrides = _config_overrides_ui("engine_csv")
                payload = {
                    "timeIndex": [dt.isoformat() for dt in base["timeIndex"]],
                    "pricesWide": base["pricesWide"],
                    "pairs": base["pairs"],
                    "configOverrides": overrides,
                }
                with st.expander("🔍 Preview payload (CSV + overrides)", expanded=False):
                    st.json(payload)
                _payload_download_ui(payload, "⬇️ הורד Engine payload (CSV)", "engine_csv")
        else:
            st.info("העלה קובץ CSV כדי לבנות payload.")

    elif dataset_mode == "Load saved JSON":
        payload = _payload_upload_json_ui("engine_load")
    else:  # Raw JSON
        raw = st.text_area(
            "Engine request JSON (timeIndex, pricesWide, pairs, configOverrides)",
            value=json.dumps(
                {
                    "timeIndex": [],
                    "pricesWide": {},
                    "pairs": [],
                    "configOverrides": {},
                },
                indent=2,
                default=str,
            ),
            height=260,
            key="fv_engine_raw_json",
        )
        try:
            payload = json.loads(raw)
        except Exception:
            st.error("JSON לא תקין, תקן לפני שליחה.")
            payload = None

    if payload is None:
        st.info("אין payload לטאב Fair Value (לא נשלח זוג/Universe מהטאבים האחרים). "
                "אפשר לבחור זוג/Universe מתוך ה-UI של הטאב.")
        return  # שוב, return במקום st.stop()

    # Data quality validation (before sending to engine)
    with st.expander("🔍 Data Quality Validator (pre-flight)", expanded=False):
        _render_data_quality_validator(payload)

    if st.button("🚀 הרץ /engine/run", type="primary", key="fv_engine_send"):
        with st.spinner("מריץ את FairValueEngine דרך ה-API..."):
            try:
                data = _post_json("/engine/run", payload)
            except Exception as e:
                st.error(f"שגיאה בקריאת ה-API: {e}")
                return

        st.success("Engine החזיר תשובה.")

        meta = data.get("meta", {})
        rows = data.get("rows", [])

        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            st.markdown("**Meta**")
            st.json(meta)
        with col_meta2:
            st.metric("מספר זוגות", len(rows))
        with col_meta3:
            st.caption("הנתונים מטופלים בצד ה-Engine, זה רק UI לקריאה.")

        if not rows:
            st.info("לא חזרו rows מה-Engine.")
            return

        _normalize_pair_field(rows)
        df = pd.DataFrame(rows)

        st.markdown("### 📋 תוצאות גולמיות")
        st.dataframe(df, use_container_width=True)

        st.markdown("### 📊 ניתוח מהיר של universe")

        metrics_cols = st.multiselect(
            "בחר עמודות איכות/סיכון להצגה כגרפים:",
            options=[c for c in df.columns if c in ["dsr_net", "psr_net", "sr_net", "net_edge_z", "turnover_est", "avg_hold_days"]],
            default=[c for c in ["dsr_net", "net_edge_z"] if c in df.columns],
            key="fv_engine_metric_cols",
        )

        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.metric("Pairs count", len(df))
        with col_kpi2:
            if "dsr_net" in df.columns:
                frac_good = float((df["dsr_net"] > 0).mean() * 100.0)
                st.metric("% עם DSR>0", f"{frac_good:.1f}%")
            elif "sr_net" in df.columns:
                frac_good = float((df["sr_net"] > 0).mean() * 100.0)
                st.metric("% עם Sharpe>0", f"{frac_good:.1f}%")
            else:
                st.metric("% pairs with edge>0", "N/A")
        with col_kpi3:
            if "rp_weight" in df.columns:
                w = df["rp_weight"].abs().fillna(0.0)
                s = float(w.sum()) or 1.0
                top5 = float(w.sort_values(ascending=False).head(5).sum() / s * 100)
                st.metric("ריכוז Top-5", f"{top5:.1f}%")
            else:
                st.metric("ריכוז Top-5", "N/A")

        if metrics_cols:
            target_metric = st.selectbox(
                "מדד עיקרי לניתוח:",
                options=metrics_cols,
                index=0,
                key="fv_engine_target_metric",
            )
            st.markdown(f"#### התפלגות {target_metric}")
            fig = px.histogram(df, x=target_metric, nbins=30)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"#### Top pairs לפי {target_metric}")
            top_n = st.slider("כמה זוגות להציג (Top-N):", min_value=5, max_value=50, value=15, step=5, key="fv_engine_top_n")
            df_top = df.sort_values(by=target_metric, ascending=False).head(top_n)
            st.dataframe(df_top, use_container_width=True)
        else:
            st.info("בחר לפחות מדד אחד כדי לראות גרפים ו-Top pairs.")

        st.divider()

        # ── New professional analytics sections ──────────────────────────
        _render_statistical_scorecard(df)

        st.divider()
        _render_engine_multi_charts(df)

        st.divider()
        _render_sensitivity_sweep(df)

        st.divider()
        with st.expander("🔎 Per-Pair Detail Cards", expanded=False):
            _render_per_pair_detail(df)


# =====================================
#  Advisor Section (עם השוואת ריצות)
# =====================================

def _init_advisor_run_store() -> None:
    """
    מוודא שיש מקום בהיסטוריה של ריצות Advisor ב-session_state.
    """
    if "fv_advisor_runs" not in st.session_state:
        st.session_state["fv_advisor_runs"] = []  # list of dicts


def _store_advisor_run(summary: Dict[str, Any], advice: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    """
    שומר ריצה אחת של Advisor להיסטוריה (עד 10 ריצות אחרונות).
    """
    _init_advisor_run_store()
    runs: List[Dict[str, Any]] = st.session_state["fv_advisor_runs"]

    run_id = f"run_{len(runs) + 1}"
    entry = {
        "id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "advice": advice,
        "payload": payload,
    }
    runs.append(entry)
    if len(runs) > 10:
        del runs[0]
    st.session_state["fv_advisor_runs"] = runs


def _render_advisor_compare_ui() -> None:
    """
    מאפשר לבחור שתי ריצות Advisor ולהשוות ביניהן:
    - KPIs מרכזיים
    - הבדל בעצות (IDs שנוספו/נעלמו)
    """
    _init_advisor_run_store()
    runs: List[Dict[str, Any]] = st.session_state["fv_advisor_runs"]

    if len(runs) < 2:
        st.info("כדי להשוות ריצות צריך לפחות 2 ריצות Advisor. תריץ כמה פעמים עם פרמטרים שונים.")
        return

    st.markdown("### 🔁 השוואת ריצות Advisor")

    options = [f"{r['id']} — {r['timestamp']}" for r in runs]
    col_a, col_b = st.columns(2)
    with col_a:
        sel_a = st.selectbox("ריצה A:", options=options, index=len(options) - 2, key="fv_adv_cmp_a")
    with col_b:
        sel_b = st.selectbox("ריצה B:", options=options, index=len(options) - 1, key="fv_adv_cmp_b")

    if sel_a == sel_b:
        st.warning("בחר שתי ריצות שונות להשוואה.")
        return

    run_a = runs[options.index(sel_a)]
    run_b = runs[options.index(sel_b)]

    sum_a = run_a["summary"]
    sum_b = run_b["summary"]

    st.markdown("#### KPIs עיקריים (A vs B)")
    kpi_keys = [
        ("n_pairs", "מספר זוגות"),
        ("frac_good_pairs", "% זוגות עם edge>0"),
        ("concentration_top5_pct", "ריכוז Top-5"),
        ("avg_halflife", "Halflife ממוצע"),
        ("avg_turnover_est", "Turnover ממוצע"),
        ("avg_edge_z", "Edge ממוצע (net_edge_z)"),
    ]

    for key, label in kpi_keys:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**{label} — A**")
            st.write(sum_a.get(key))
        with col2:
            st.markdown(f"**{label} — B**")
            st.write(sum_b.get(key))
        with col3:
            try:
                va = float(sum_a.get(key)) if sum_a.get(key) is not None else None
                vb = float(sum_b.get(key)) if sum_b.get(key) is not None else None
                if va is not None and vb is not None:
                    diff = vb - va
                    st.write(f"Δ {diff:+.3f}")
                else:
                    st.write("Δ N/A")
            except Exception:
                st.write("Δ N/A")

    st.markdown("#### שינוי בעצות (Advice delta)")

    ids_a = {a["id"] for a in run_a["advice"]}
    ids_b = {b["id"] for b in run_b["advice"]}

    added = ids_b - ids_a
    removed = ids_a - ids_b
    unchanged = ids_a & ids_b

    col_add, col_rem, col_same = st.columns(3)
    with col_add:
        st.markdown("**עצות חדשות (ב-B ולא ב-A):**")
        if not added:
            st.write("אין.")
        else:
            for adv_id in sorted(added):
                st.write(f"- {adv_id}")
    with col_rem:
        st.markdown("**עצות שנעלמו (ב-A ולא ב-B):**")
        if not removed:
            st.write("אין.")
        else:
            for adv_id in sorted(removed):
                st.write(f"- {adv_id}")
    with col_same:
        st.markdown("**עצות משותפות:**")
        if not unchanged:
            st.write("אין.")
        else:
            for adv_id in sorted(unchanged):
                st.write(f"- {adv_id}")


def _render_advisor_section() -> None:
    st.subheader("🧠 Advisor – /advisor/run")

    st.caption("ה-Advisor מריץ את ה-Engine בפנים, מנתח את ה-universe ונותן עצות פרמטריות לשיפור.")

    dataset_mode = st.radio(
        "מקור דאטה ל-Advisor:",
        options=["Demo", "Upload CSV", "Raw JSON", "Load saved JSON"],
        horizontal=True,
        key="fv_advisor_dataset_mode",
    )

    payload: Optional[Dict[str, Any]] = None

    if dataset_mode == "Demo":
        base = _build_demo_prices()
        overrides = _config_overrides_ui("advisor_demo")
        payload = {
            "timeIndex": [dt.isoformat() for dt in base["timeIndex"]],
            "pricesWide": base["pricesWide"],
            "pairs": base["pairs"],
            "configOverrides": overrides,
        }
        with st.expander("🔍 Preview payload (Demo + overrides)", expanded=False):
            st.json(payload)
        _payload_download_ui(payload, "⬇️ הורד Advisor payload (Demo)", "advisor_demo")

    elif dataset_mode == "Upload CSV":
        uploaded = st.file_uploader(
            "העלה קובץ מחירים בפורמט CSV (Date + טיקרים):",
            type=["csv"],
            key="fv_advisor_csv",
        )
        if uploaded is not None:
            base = _build_from_csv(uploaded, key_prefix="advisor_csv")
            if base is not None:
                overrides = _config_overrides_ui("advisor_csv")
                payload = {
                    "timeIndex": [dt.isoformat() for dt in base["timeIndex"]],
                    "pricesWide": base["pricesWide"],
                    "pairs": base["pairs"],
                    "configOverrides": overrides,
                }
                with st.expander("🔍 Preview payload (CSV + overrides)", expanded=False):
                    st.json(payload)
                _payload_download_ui(payload, "⬇️ הורד Advisor payload (CSV)", "advisor_csv")
        else:
            st.info("העלה קובץ CSV כדי לבנות payload.")

    elif dataset_mode == "Load saved JSON":
        payload = _payload_upload_json_ui("advisor_load")
    else:
        raw = st.text_area(
            "Advisor request JSON (כמו EngineRunRequest)",
            value=json.dumps(
                {
                    "timeIndex": [],
                    "pricesWide": {},
                    "pairs": [],
                    "configOverrides": {},
                },
                indent=2,
                default=str,
            ),
            height=260,
            key="fv_advisor_raw_json",
        )
        try:
            payload = json.loads(raw)
        except Exception:
            st.error("JSON לא תקין, תקן לפני שליחה.")
            payload = None

    if payload is None:
        st.info("אין payload לטאב Fair Value (לא נשלח זוג/Universe מהטאבים האחרים). "
                "אפשר לבחור זוג/Universe מתוך ה-UI של הטאב.")
        return  # שוב, return במקום st.stop()

    if st.button("🧠 הרץ /advisor/run", type="primary", key="fv_advisor_send"):
        with st.spinner("מריץ Advisor (Engine + ניתוח) דרך ה-API..."):
            try:
                data = _post_json("/advisor/run", payload)
            except Exception as e:
                st.error(f"שגיאה בקריאת ה-API: {e}")
                return

        st.success("Advisor החזיר תשובה.")

        summary = data.get("summary", {})
        advice = data.get("advice", [])

        _store_advisor_run(summary, advice, payload)

        st.markdown("### 📊 Summary — תמונת מצב של ה-universe")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("מספר זוגות", summary.get("n_pairs", 0))
        with col2:
            frac_good = summary.get("frac_good_pairs")
            st.metric("% זוגות עם edge>0", f"{frac_good:.1f}%" if frac_good is not None else "N/A")
        with col3:
            c5 = summary.get("concentration_top5_pct")
            st.metric("ריכוז Top-5", f"{c5:.1f}%" if c5 is not None else "N/A")
        with col4:
            hl = summary.get("avg_halflife")
            st.metric("Halflife ממוצע (ימים)", f"{hl:.1f}" if hl is not None else "N/A")

        with st.expander("🔍 Summary full JSON", expanded=False):
            st.json(summary)

        st.markdown("### 📋 עצות לשיפור (Advisor)")

        if not advice:
            st.info("ה-Advisor לא החזיר עצות. ייתכן שה-universe נראה סביר לפי ההיגיון הנוכחי.")
        else:
            def _sev_emoji(sev: str) -> str:
                sev = (sev or "").lower()
                if sev == "critical":
                    return "🛑"
                if sev == "warning":
                    return "⚠️"
                return "ℹ️"

            for item in advice:
                sev = item.get("severity", "info")
                emoji = _sev_emoji(sev)
                header = f"{emoji} [{sev.upper()}] {item.get('category', '')} — {item.get('id', '')}"
                with st.expander(header, expanded=(sev in {"critical", "warning"})):
                    st.markdown(f"**Message:** {item.get('message','')}")
                    st.markdown(f"**Rationale:** {item.get('rationale','')}")
                    suggested = item.get("suggested_changes", {}) or {}
                    if suggested:
                        st.markdown("**Suggested changes:**")
                        for k, v in suggested.items():
                            st.markdown(f"- **{k}**: {v}")

    # בלוק השוואת ריצות
    with st.expander("🔁 השוואת ריצות Advisor (לפני/אחרי שינוי פרמטרים)", expanded=False):
        _render_advisor_compare_ui()

def _render_fv_local_single_opt_section() -> None:
    """
    מריץ api_optimize_pair על זוג אחד מתוך הקונטקסט של Fair Value,
    בלי שימוש ב-HTTP API, אלא ישירות במנוע האופטימיזציה הפנימי.
    """
    st.markdown("### 🔬 Local Pairs Optimiser (Single Pair, HF-grade)")

    # ננסה להשתמש בזוג שנבחר ב-dashboard (אם יש)
    sel_pair = st.session_state.get("selected_pair", "")
    default_sym1, default_sym2 = "SPY", "QQQ"

    if isinstance(sel_pair, str) and sel_pair:
        for sep in ("|", "/", "\\", ":", "-"):
            if sep in sel_pair:
                a, b = sel_pair.split(sep, 1)
                default_sym1, default_sym2 = a.strip(), b.strip()
                break

    col1, col2, col3 = st.columns([1.2, 1.2, 1.0])
    with col1:
        sym1 = st.text_input("Symbol A (local optimiser)", value=default_sym1, key="fv_loc_sym1")
    with col2:
        sym2 = st.text_input("Symbol B (local optimiser)", value=default_sym2, key="fv_loc_sym2")
    with col3:
        n_trials = st.number_input(
            "n_trials",
            min_value=20,
            max_value=1000,
            value=150,
            step=10,
            key="fv_loc_n_trials",
        )

    timeout_min = st.number_input(
        "Timeout (minutes)",
        min_value=1,
        max_value=60,
        value=10,
        step=1,
        key="fv_loc_timeout",
    )

    st.markdown("**Metric weights (local optimisation)**")
    cws1, cws2, cws3, cws4 = st.columns(4)
    with cws1:
        w_sh = cws1.number_input("W Sharpe", 0.0, 1.0, 0.3, 0.05, key="fv_loc_w_sharpe")
    with cws2:
        w_pf = cws2.number_input("W Profit", 0.0, 1.0, 0.4, 0.05, key="fv_loc_w_profit")
    with cws3:
        w_dd = cws3.number_input("W Drawdown", 0.0, 1.0, 0.2, 0.05, key="fv_loc_w_dd")
    with cws4:
        w_sr = cws4.number_input("W Sortino", 0.0, 1.0, 0.1, 0.05, key="fv_loc_w_sortino")

    weights = {
        "Sharpe": float(w_sh),
        "Profit": float(w_pf),
        "Drawdown": float(w_dd),
        "Sortino": float(w_sr),
    }

    run_btn = st.button("🚀 Run local optimisation for this pair", key="fv_loc_run_single")

    if not run_btn:
        return

    with st.spinner(f"Optimising {sym1}-{sym2} via api_optimize_pair…"):
        try:
            df_res, meta = api_optimize_pair(
                sym1,
                sym2,
                ranges=None,     # כרגע נותנים לו לבנות לפי profile דיפולט
                weights=weights,
                n_trials=int(n_trials),
                timeout_min=int(timeout_min),
                direction="maximize",
                sampler_name="TPE",
                pruner_name="median",
                profile="default",
                multi_objective=False,
                objective_metrics=None,
                param_mapping=None,
            )
        except Exception as _opt_err:
            st.error(f"Local optimisation failed: {_opt_err}")
            logger.exception("api_optimize_pair raised for %s-%s", sym1, sym2)
            return

    st.markdown("#### 🧾 Meta (local optimiser)")
    st.json(meta)

    if df_res is None or df_res.empty:
        st.warning("Local optimisation produced no rows for this pair.")
        return

    st.markdown("#### 🏅 Top 10 configs (local optimiser)")
    top10 = df_res.copy()
    if "Score" in top10.columns:
        top10 = top10.sort_values("Score", ascending=False).head(10)
    else:
        top10 = top10.head(10)

    st.dataframe(top10, use_container_width=True)

    # שומרים סיכום ל-session לטובת סוכן/טאב אחר
    try:
        st.session_state.setdefault("fv_local_opt_single", {})
        st.session_state["fv_local_opt_single"][f"{sym1}-{sym2}"] = {
            "meta": meta,
            "top_params": top10.to_dict(orient="records"),
        }
    except Exception:
        pass


def _render_fv_local_batch_opt_section() -> None:
    """
    מריץ api_optimize_pairs_batch על רשימת זוגות מתוך ה-Fair Value Lab.
    """
    st.markdown("### 🧪 Local Batch Optimiser (Pairs universe)")

    st.caption("הזן רשימת זוגות (או השתמש בברירת מחדל) להרצת אופטימיזציה פנימית על כמה pairs ביחד.")

    default_pairs_list = st.session_state.get("fv_universe_pairs", ["SPY-QQQ", "SPY-IWM"])
    default_text = "\n".join(default_pairs_list)

    pairs_text = st.text_area(
        "Pairs (one per line, format: SYM1-SYM2)",
        value=default_text,
        key="fv_loc_batch_pairs_text",
    )

    pairs: List[Tuple[str, str]] = []
    for ln in pairs_text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        sep_used = None
        for sep in ("|", "/", "\\", ":", "-"):
            if sep in ln:
                sep_used = sep
                break
        if not sep_used:
            continue
        a, b = ln.split(sep_used, 1)
        a, b = a.strip(), b.strip()
        if a and b:
            pairs.append((a, b))

    if not pairs:
        st.info("Add at least one pair to run batch optimisation.")
        return

    st.write("Pairs to optimise:", ", ".join([f"{a}-{b}" for a, b in pairs]))

    n_trials = st.number_input(
        "n_trials per pair",
        min_value=20,
        max_value=1000,
        value=80,
        step=10,
        key="fv_loc_batch_n_trials",
    )
    timeout_min = st.number_input(
        "Timeout per pair (minutes)",
        min_value=1,
        max_value=60,
        value=6,
        step=1,
        key="fv_loc_batch_timeout",
    )

    run_batch_btn = st.button("🚀 Run local batch optimisation", key="fv_loc_batch_run")

    if not run_batch_btn:
        return

    with st.spinner(f"Running local batch optimisation for {len(pairs)} pairs…"):
        df_batch, meta_batch = api_optimize_pairs_batch(
            pairs,
            ranges=None,
            weights=None,     # ה-API כבר מטפל ב-fallbacks חכמים
            n_trials=int(n_trials),
            timeout_min=int(timeout_min),
            direction="maximize",
            sampler_name="TPE",
            pruner_name="median",
            profile="default",
            multi_objective=False,
            objective_metrics=None,
            param_mapping=None,
        )

    st.markdown("#### 🧾 Batch meta (local optimiser)")
    st.json(
        {
            "status": meta_batch.get("status"),
            "duration_sec": meta_batch.get("duration_sec"),
            "n_pairs": meta_batch.get("n_pairs"),
        }
    )

    if df_batch is None or df_batch.empty:
        st.warning("Batch optimisation produced no rows.")
        return

    per_pair = meta_batch.get("per_pair", {})
    summary_rows = []
    for label, info in per_pair.items():
        summary_rows.append(
            {
                "Pair": label,
                "Best Score": info.get("best_score"),
                "Best Sharpe": info.get("best_sharpe"),
                "Rows": info.get("rows"),
            }
        )
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        st.markdown("#### 📊 Per-pair summary (local optimiser)")
        st.dataframe(df_summary, use_container_width=True)

    st.markdown("#### 🔎 Full batch df (top 200 rows)")
    st.dataframe(df_batch.head(200), use_container_width=True)

    # שמירה ל-session לשימוש עתידי
    st.session_state["fv_local_batch_opt_df"] = df_batch
    st.session_state["fv_local_batch_opt_meta"] = meta_batch

# =====================================
#  Optimizer Section
# =====================================

def _render_optimizer_section() -> None:
    st.subheader("🧬 Optimizer – /optimizer/run")

    # ----- Local HF-grade optimiser using internal engine -----
    with st.expander("🔬 Local Pairs Optimiser (internal engine)", expanded=False):
        _render_fv_local_single_opt_section()

    with st.expander("🧪 Local Batch Optimiser (internal engine)", expanded=False):
        _render_fv_local_batch_opt_section()

    st.markdown("---")
    st.markdown("### 🌐 Remote Optimizer API – /optimizer/run")

    mode = st.radio(
        "מקור דאטה ל-Optimizer:",
        options=["Demo", "Upload CSV", "Raw JSON", "Load saved JSON"],
        horizontal=True,
        key="fv_opt_dataset_mode",
    )

    payload: Optional[Dict[str, Any]] = None

    if mode == "Demo":
        base = _build_demo_prices()
        st.markdown("#### ⚙️ OptConfig (Demo)")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_trials = st.number_input(
                "n_trials",
                min_value=5,
                max_value=500,
                value=20,
                step=5,
                key="fv_opt_demo_n_trials",
            )
            timeout_sec = st.number_input(
                "timeout_sec",
                min_value=5,
                max_value=600,
                value=60,
                step=5,
                key="fv_opt_demo_timeout",
            )
        with col2:
            target = st.selectbox(
                "target",
                options=["dsr_net", "psr_net", "sr_net"],
                index=0,
                key="fv_opt_demo_target",
            )
            use_ensemble = st.checkbox(
                "use_ensemble",
                value=False,
                key="fv_opt_demo_use_ensemble",
            )
        with col3:
            n_folds = st.number_input(
                "n_folds",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                key="fv_opt_demo_n_folds",
            )
            test_frac = st.slider(
                "test_frac",
                min_value=0.05,
                max_value=0.5,
                value=0.2,
                step=0.05,
                key="fv_opt_demo_test_frac",
            )

        opt_config = {
            "n_trials": int(n_trials),
            "timeout_sec": int(timeout_sec),
            "sampler": "tpe",
            "seed": 42,
            "pruner": "median",
            "target": target,
            "use_ensemble": bool(use_ensemble),
            "agg": "median",
            "trim_alpha": 0.1,
            "penalty_turnover": 0.0,
            "min_avg_hold": 0.0,
            "n_folds": int(n_folds),
            "test_frac": float(test_frac),
            "purge_frac": 0.02,
        }

        payload = {
            "timeIndex": [dt.isoformat() for dt in base["timeIndex"]],
            "pricesWide": base["pricesWide"],
            "pairs": base["pairs"],
            "optConfig": opt_config,
        }

        with st.expander("🔍 Preview payload (Demo OptConfig)", expanded=False):
            st.json(payload)
        _payload_download_ui(payload, "⬇️ הורד Optimizer payload (Demo)", "opt_demo")

    elif mode == "Upload CSV":
        uploaded = st.file_uploader(
            "העלה קובץ CSV (Date + טיקרים):",
            type=["csv"],
            key="fv_opt_csv",
        )
        if uploaded is not None:
            base = _build_from_csv(uploaded, key_prefix="opt_csv")
            if base is not None:
                st.markdown("#### ⚙️ OptConfig (CSV)")
                col1, col2 = st.columns(2)
                with col1:
                    n_trials = st.number_input(
                        "n_trials",
                        min_value=5,
                        max_value=500,
                        value=30,
                        step=5,
                        key="fv_opt_csv_n_trials",
                    )
                with col2:
                    target = st.selectbox(
                        "target",
                        options=["dsr_net", "psr_net", "sr_net"],
                        index=0,
                        key="fv_opt_csv_target",
                    )

                opt_config = {
                    "n_trials": int(n_trials),
                    "target": target,
                }

                payload = {
                    "timeIndex": [dt.isoformat() for dt in base["timeIndex"]],
                    "pricesWide": base["pricesWide"],
                    "pairs": base["pairs"],
                    "optConfig": opt_config,
                }
                with st.expander("🔍 Preview payload (CSV OptConfig)", expanded=False):
                    st.json(payload)
                _payload_download_ui(payload, "⬇️ הורד Optimizer payload (CSV)", "opt_csv")
        else:
            st.info("העלה CSV כדי להריץ אופטימיזציה.")
    elif mode == "Load saved JSON":
        payload = _payload_upload_json_ui("opt_load")
    else:
        raw = st.text_area(
            "Optimizer request JSON (timeIndex, pricesWide, pairs, optConfig, ...)",
            value=json.dumps(
                {
                    "timeIndex": [],
                    "pricesWide": {},
                    "pairs": [],
                    "optConfig": {
                        "n_trials": 10,
                        "target": "dsr_net",
                    },
                },
                indent=2,
                default=str,
            ),
            height=260,
            key="fv_opt_raw_json",
        )
        try:
            payload = json.loads(raw)
        except Exception:
            st.error("JSON לא תקין, תקן לפני שליחה.")
            payload = None

    if payload is None:
        st.info("אין payload לטאב Fair Value (לא נשלח זוג/Universe מהטאבים האחרים). "
                "אפשר לבחור זוג/Universe מתוך ה-UI של הטאב.")
        return  # שוב, return במקום st.stop()

    if st.button("🚀 הרץ /optimizer/run", type="primary", key="fv_opt_send"):
        with st.spinner("מריץ optimize_fair_value דרך ה-API..."):
            try:
                data = _post_json("/optimizer/run", payload)
            except Exception as e:
                st.error(f"שגיאה בקריאת ה-API: {e}")
                return

        st.success("Optimizer החזיר תשובה.")

        st.markdown("### 🏆 תוצאות אופטימיזציה")
        st.json(data)

        best_params = data.get("bestParams")
        if best_params is not None:
            st.markdown("#### 📋 bestParams כטבלה")
            df_params = pd.DataFrame(
                [{"param": k, "value": v} for k, v in best_params.items()]
            )
            st.dataframe(df_params, use_container_width=True)


# =====================================
#  Entry point for dashboard
# =====================================

def render_fair_value_api_tab() -> None:
    """
    טאב מבודד – לא נוגע בשום State גלובלי של הדשבורד.
    מיועד לבדיקה/משחק עם Fair Value API בלבד.
    """
    st.markdown(
        """
<div style="
    background:linear-gradient(90deg,#1A237E 0%,#283593 100%);
    border-radius:10px;padding:14px 20px;margin-bottom:14px;
    box-shadow:0 2px 8px rgba(26,35,126,0.22);
">
    <div style="font-size:1.15rem;font-weight:800;color:white;letter-spacing:-0.2px;">
        ⚖️ Fair Value API Lab
    </div>
    <div style="font-size:0.76rem;color:rgba(255,255,255,0.76);margin-top:3px;">
        Isolated sandbox · API testing · Universe analysis · Advisor recommendations · Run comparison
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.caption(
        "This tab interacts only with the local API server (root/api_server.py) via HTTP. "
        "No global config, data, or tables are modified."
    )

    cfg = _get_fair_value_config()
    current_base = _resolve_api_base()

    st.write(f"**API base:** `{current_base}`")

    if cfg is not None:
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            st.markdown("**FairValueAPIConfig**")
            st.write(f"Profile: `{getattr(cfg, 'profile', 'unknown')}`")
            st.write(f"Enabled: `{getattr(cfg, 'is_enabled', getattr(cfg, 'enabled', False))}`")
            st.write(f"Connect timeout: `{getattr(cfg, 'connect_timeout_sec', None)}` sec")
            st.write(f"Read timeout: `{getattr(cfg, 'read_timeout_sec', None)}` sec")
        with col_cfg2:
            st.markdown("**Rate & Logging**")
            st.write(f"Max concurrent requests: `{getattr(cfg, 'max_concurrent_requests', None)}`")
            st.write(f"Max requests/minute: `{getattr(cfg, 'max_requests_per_minute', None)}`")
            st.write(f"log_requests: `{getattr(cfg, 'log_requests', None)}`")
            st.write(f"log_payloads: `{getattr(cfg, 'log_payloads', None)}`")
    else:
        st.info(
            "לא נמצא FairValueAPIConfig פעיל בקונטקסט. "
            "הטאב משתמש ב-FAIR_VALUE_API_URL (secrets) או בברירת מחדל לוקאלית."
        )

    ok, health = _check_health()
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        if ok:
            st.success("API Online ✅")
        else:
            st.error("API Offline ❌ — ודא ש-uvicorn של api_server.py רץ.")
    with col_h2:
        st.markdown("**/health response:**")
        st.json(health)

    mode = st.radio(
        "בחר מצב עבודה:",
        options=["⚙️ Engine", "🧠 Advisor", "🧬 Optimizer"],
        horizontal=True,
        key="fv_mode",
    )

    if mode.startswith("⚙️"):
        _render_engine_section()
    elif mode.startswith("🧠"):
        _render_advisor_section()
    else:
        _render_optimizer_section()


__all__ = ["render_fair_value_api_tab"]
