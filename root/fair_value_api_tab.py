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

from datetime import datetime, timedelta
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
    end = datetime.utcnow().date()
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
        prices_wide[sym] = series.fillna(method="ffill").fillna(method="bfill").tolist()

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
        file_name=f"{key_prefix}_payload_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
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
        "timestamp": datetime.utcnow().isoformat(),
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

        best_params = data.get("bestParams") or {}
        if best_params:
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
    st.markdown("## 🧬 Fair Value API Lab — Sandbox מבודד")
    st.caption(
        "הטאב הזה עובד רק מול ה-API המקומי (root/api_server.py) דרך HTTP. "
        "הוא לא משנה שום דבר בקונפיג, דאטה או טבלאות אחרות."
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
