# -*- coding: utf-8 -*-
"""
overview_tab.py — Investment Committee Overview (HF-grade)
==========================================================

טאב "סקירה כוללת" סופר-חשוב לפני הכנסת כסף אמיתי:

מה הוא עושה:
-------------
1. Pair Snapshot (מיקרו):
   - מציג את הזוג הנבחר (selected_pair).
   - מסכם PairDiagnostics: Cointegration, Halflife, Hurst, Corr, Beta, Vol.
   - מציג סטטוס איכות סטטיסטית (Mean-reversion / Trending / Unclear).

2. Recommender vs Optimization vs Backtest (מזעור פערים):
   - שמושך:
       * pair_rec_params / pair_recommendations (Pair Tab)
       * opt_df / opt_df_batch (Optimization Tab)
       * bt_last_result (Backtest Tab)
   - בונה טבלת השוואת פרמטרים (Recommender / Opt Best / Last BT).
   - מציג KPIs מרכזיים מכל צד.

3. Portfolio & Risk Snapshot (מאקרו-תיק):
   - משתמש ב-session_state:
       * portfolio_snapshot (אם קיים).
       * risk_limits (יעדי סיכון / תקרות).
   - מראה:
       * חשיפה נוכחית לגזור/סקטור (אם קיים).
       * VaR / ES / MaxDD על הפורטפוליו.
       * האם השטחה/הוספת הזוג תחרוג מגבולות הסיכון.

4. Macro Overlay (מצב שוק):
   - קורא macro_snapshot מה-session אם קיים:
       * regime_name (Risk-on/off, Inflation, Recession).
       * risk_index, vol_regime, credit_spread_level, וכו'.
   - מציג האם האסטרטגיה (mean-reversion pairs) "עם" או "נגד" המאקרו.

5. Fair Value Advisor Snapshot (אם קיים):
   - אם fair value advisor רץ בעבר ושמרת לתוך:
       * st.session_state["fv_advisor_runs"]
     הטאב ייקח את הריצה האחרונה וימשוך summary + 3 עצות חזקות.

6. Investment Readiness Verdict:
   - מסכם את כל הנ"ל ל:
       * ✅ GO
       * ⚠️ CAUTION
       * ❌ NO-GO
   - נותן Check-list מילולי מה צריך לשפר לפני כניסה.

⚠️ חשוב:
- הטאב **לא מריץ אופטימיזציה/בק-טסט/אדווייזר** בעצמו.
- הוא רק קורא מה שכבר שמור ב-st.session_state.
- אם משהו חסר — הוא פשוט מסביר מה חסר, בלי לשבור כלום.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:  # Plotly אופציונלי (אם קיים)
    import plotly.express as px  # type: ignore
except Exception:  # pragma: no cover
    px = None  # type: ignore

JSONDict = Dict[str, Any]


# =========================
#  Session access helpers
# =========================

def _get_selected_pair_key() -> Optional[str]:
    """
    מחזיר מחרוזת זוג נבחר, למשל 'SPY-QQQ'.

    Pair Tab צפוי לשמור:
    - st.session_state["selected_pair"] = "SYM1-SYM2"
    """
    val = st.session_state.get("selected_pair")
    return str(val) if isinstance(val, str) and val.strip() else None


def _get_pair_containers(pair_key: str) -> Tuple[JSONDict, JSONDict, JSONDict]:
    """
    מחזיר:
    - diagnostics dict מתוך pair_diagnostics[pair_key] אם קיים.
    - recommendation dict מתוך pair_recommendations[pair_key] אם קיים.
    - rec_params dict מתוך pair_rec_params[pair_key] אם קיים.
    """
    diags_map = st.session_state.get("pair_diagnostics", {}) or {}
    recs_map = st.session_state.get("pair_recommendations", {}) or {}
    recp_map = st.session_state.get("pair_rec_params", {}) or {}

    diag = diags_map.get(pair_key) or {}
    rec = recs_map.get(pair_key) or {}
    rec_params = recp_map.get(pair_key) or {}

    if diag and not isinstance(diag, dict):
        try:
            if is_dataclass(diag):
                diag = asdict(diag)
            else:
                diag = dict(diag)
        except Exception:
            diag = {}

    return diag, rec, rec_params


def _get_opt_df() -> Optional[pd.DataFrame]:
    """
    מחזיר DataFrame של אופטימיזציה:
    - opt_df אם קיים.
    - אחרת opt_df_batch אם קיים.
    """
    df = st.session_state.get("opt_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df

    df_batch = st.session_state.get("opt_df_batch")
    if isinstance(df_batch, pd.DataFrame) and not df_batch.empty:
        return df_batch

    return None


def _get_bt_last_result_dict() -> Optional[JSONDict]:
    """
    מחזיר את תוצאת הבק-טסט האחרון כ-dict אם אפשר.

    Backtest Tab צפוי לשמור:
    - st.session_state["bt_last_result"] = BacktestResult
    """
    res = st.session_state.get("bt_last_result")
    if res is None:
        return None

    if hasattr(res, "to_dict"):
        try:
            return res.to_dict()  # type: ignore[no-any-return]
        except Exception:
            pass

    if is_dataclass(res):
        try:
            return asdict(res)
        except Exception:
            pass

    if isinstance(res, dict):
        return res

    try:
        return dict(vars(res))
    except Exception:
        return None


def _get_portfolio_snapshot() -> Optional[JSONDict]:
    """
    מחזיר תמונת מצב פורטפוליו אם קיימת:
    צפוי מבנה כללי כמו:
    {
      "total_equity": ...,
      "cash": ...,
      "gross_exposure": ...,
      "net_exposure": ...,
      "var_95": ...,
      "es_97": ...,
      "sector_exposure": {...},
      "pair_exposure": {...},
      ...
    }
    """
    snap = st.session_state.get("portfolio_snapshot")
    if isinstance(snap, dict):
        return snap
    return None


def _get_risk_limits() -> Optional[JSONDict]:
    """
    מחזיר הגדרות גבולות סיכון אם קיימות, למשל:
    {
      "max_gross_exposure": 2.0,
      "max_var_95": 0.05,
      "max_single_pair_risk": 0.02,
      ...
    }
    """
    limits = st.session_state.get("risk_limits")
    if isinstance(limits, dict):
        return limits
    return None


def _get_macro_snapshot() -> Optional[JSONDict]:
    """
    מחזיר macro_snapshot אם קיים, למשל:
    {
      "regime_name": "Risk-on",
      "risk_index": 0.7,
      "vol_regime": "low",
      "credit_spread_level": "tight",
      "comment": "Equities supported by loose financial conditions",
      ...
    }
    """
    snap = st.session_state.get("macro_snapshot")
    if isinstance(snap, dict):
        return snap
    return None


def _get_latest_fv_advisor() -> Optional[JSONDict]:
    """
    אם טאב Fair Value API שמר ריצות Advisor:
    st.session_state["fv_advisor_runs"] = [ { summary, advice, ... }, ... ]

    נחזיר את האחרונה.
    """
    runs = st.session_state.get("fv_advisor_runs")
    if isinstance(runs, list) and runs:
        last = runs[-1]
        if isinstance(last, dict) and "summary" in last:
            return last
    return None


# =========================
#  Comparison helpers
# =========================

def _pick_metric(metrics: JSONDict, candidates: List[str]) -> Tuple[Optional[str], Optional[float]]:
    """
    מחפש מדד לפי רשימת עדיפויות של שמות (case-insensitive).
    מחזיר (שם אמיתי, ערך) אם נמצא.
    """
    if not isinstance(metrics, dict):
        return None, None

    lower_map = {k.lower(): k for k in metrics.keys()}
    for name in candidates:
        key = lower_map.get(name.lower())
        if key is not None:
            try:
                return key, float(metrics[key])
            except Exception:
                return key, None
    return None, None


def _extract_best_opt_row_for_pair(df: pd.DataFrame, pair_key: str) -> Optional[pd.Series]:
    """
    מחפש את השורה הטובה ביותר ב-opt_df עבור זוג ספציפי.
    """
    if df is None or df.empty:
        return None

    df_local = df.copy()
    key = pair_key

    for col in ("Pair", "pair"):
        if col in df_local.columns:
            sub = df_local[df_local[col].astype(str) == key]
            if not sub.empty:
                df_local = sub
                break

    if "symbol_1" in df_local.columns and "symbol_2" in df_local.columns:
        merged_key = df_local["symbol_1"].astype(str) + "-" + df_local["symbol_2"].astype(str)
        mask = merged_key == key
        if mask.any():
            df_local = df_local[mask]

    if df_local.empty:
        return None

    score_cols_pref = [
        "Score",
        "score",
        "Sharpe",
        "sharpe",
        "sr_net",
        "dsr_net",
        "psr_net",
        "objective",
    ]
    score_col = None
    lower_cols = {c.lower(): c for c in df_local.columns}
    for name in score_cols_pref:
        if name.lower() in lower_cols:
            score_col = lower_cols[name.lower()]
            break

    if score_col is None:
        return df_local.iloc[0]

    return df_local.sort_values(by=score_col, ascending=False).iloc[0]


def _build_param_comparison_table(
    rec_params: JSONDict,
    opt_row: Optional[pd.Series],
    bt_config: Optional[JSONDict],
) -> Optional[pd.DataFrame]:
    """
    משווה פרמטרים:
    - Pair Recommender (rec_params)
    - Optimization Best (opt_row)
    - Last Backtest Config (bt_config)
    """
    if not rec_params and opt_row is None and not bt_config:
        return None

    opt_dict: JSONDict = {}
    if opt_row is not None:
        try:
            opt_dict = opt_row.to_dict()
        except Exception:
            opt_dict = {}

    bt_cfg = bt_config or {}

    keys: List[str] = sorted(
        set(rec_params.keys()) | set(opt_dict.keys()) | set(bt_cfg.keys())
    )

    rows = []
    for k in keys:
        rec_v = rec_params.get(k)
        opt_v = opt_dict.get(k)
        bt_v = bt_cfg.get(k)

        try:
            dv_opt_rec = float(opt_v) - float(rec_v)  # type: ignore[arg-type]
        except Exception:
            dv_opt_rec = np.nan

        try:
            dv_bt_opt = float(bt_v) - float(opt_v)  # type: ignore[arg-type]
        except Exception:
            dv_bt_opt = np.nan

        rows.append(
            {
                "Parameter": k,
                "Pair Rec.": rec_v,
                "Opt Best": opt_v,
                "Last BT": bt_v,
                "Δ Opt - Rec": dv_opt_rec,
                "Δ BT - Opt": dv_bt_opt,
            }
        )

    return pd.DataFrame(rows)


# =========================
#  Readiness classification
# =========================

def _classify_pair_quality(diag: JSONDict) -> str:
    """
    מחלק איכות סטטיסטית של הזוג:

    מחזיר אחד מ:
    - "good"
    - "caution"
    - "bad"
    """
    coint_p = diag.get("coint_pvalue")
    hl = diag.get("halflife")
    hurst = diag.get("hurst")

    # Cointegration
    try:
        cp = float(coint_p) if coint_p is not None else None
    except Exception:
        cp = None

    # Halflife
    try:
        hl_f = float(hl) if hl is not None else None
    except Exception:
        hl_f = None

    # Hurst
    try:
        h_f = float(hurst) if hurst is not None else None
    except Exception:
        h_f = None

    good_points = 0
    bad_points = 0

    if cp is not None:
        if cp < 0.05:
            good_points += 1
        elif cp > 0.15:
            bad_points += 1

    if hl_f is not None:
        if 1 <= hl_f <= 60:
            good_points += 1
        else:
            bad_points += 1

    if h_f is not None:
        if h_f < 0.4:
            good_points += 1
        elif h_f > 0.6:
            bad_points += 1

    if bad_points >= 2:
        return "bad"
    if good_points >= 2 and bad_points == 0:
        return "good"
    return "caution"


def _classify_backtest_quality(metrics: JSONDict) -> str:
    """
    מסווג איכות בק-טסט (Sharpe / MaxDD / #Trades).
    """
    _, sharpe = _pick_metric(metrics, ["Sharpe"])
    _, maxdd = _pick_metric(metrics, ["MaxDD", "MaxDrawdown"])
    _, trades = _pick_metric(metrics, ["Trades", "NTrades", "TradeCount"])

    good_points = 0
    bad_points = 0

    if sharpe is not None:
        if sharpe >= 2.0:
            good_points += 1
        elif sharpe < 1.0:
            bad_points += 1

    if maxdd is not None:
        try:
            if maxdd <= -5:  # כבר באחוזים שליליים?
                good_points += 1
            elif maxdd <= -20:
                bad_points += 1
        except Exception:
            pass

    if trades is not None:
        if trades >= 50:
            good_points += 1
        elif trades < 20:
            bad_points += 1

    if bad_points >= 2:
        return "bad"
    if good_points >= 2 and bad_points == 0:
        return "good"
    return "caution"


def _classify_portfolio_risk(port_snap: Optional[JSONDict], limits: Optional[JSONDict]) -> str:
    """
    מסווג את מצב הסיכון של הפורטפוליו:
    - "good" / "caution" / "bad"
    """
    if port_snap is None or limits is None:
        return "caution"

    ge = port_snap.get("gross_exposure")
    var_95 = port_snap.get("var_95")
    max_ge = limits.get("max_gross_exposure")
    max_var = limits.get("max_var_95")

    bad_points = 0
    good_points = 0

    try:
        if ge is not None and max_ge is not None:
            r = float(ge) / float(max_ge)
            if r > 1.1:
                bad_points += 1
            elif r < 0.8:
                good_points += 1
    except Exception:
        pass

    try:
        if var_95 is not None and max_var is not None:
            r = float(var_95) / float(max_var)
            if r > 1.1:
                bad_points += 1
            elif r < 0.8:
                good_points += 1
    except Exception:
        pass

    if bad_points >= 2:
        return "bad"
    if good_points >= 1 and bad_points == 0:
        return "good"
    return "caution"


def _classify_macro_regime(macro: Optional[JSONDict]) -> str:
    """
    מסווג מצב מאקרו לכיוון אסטרטגיית mean-reversion pairs:
    - "good" / "caution" / "bad"
    """
    if macro is None:
        return "caution"

    regime = str(macro.get("regime_name", "")).lower()
    vol_regime = str(macro.get("vol_regime", "")).lower()

    if "risk-on" in regime and "low" in vol_regime:
        return "good"
    if "stress" in regime or "crisis" in regime:
        return "bad"
    if "risk-off" in regime and "high" in vol_regime:
        return "caution"
    return "caution"


def _overall_verdict(
    pair_q: str,
    bt_q: str,
    port_q: str,
    macro_q: str,
) -> str:
    """
    מכלול סיווגים → GO / CAUTION / NO-GO.
    """
    states = [pair_q, bt_q, port_q, macro_q]
    if "bad" in states:
        return "NO-GO"
    if all(s == "good" for s in states):
        return "GO"
    return "CAUTION"


# =========================
#  Main render function
# =========================

def render_overview_tab() -> None:
    """
    טאב הסיכום המרכזי לפני השקעה.

    לא מריץ כלום בעצמו — רק קורא:
    - Pair Tab (diagnostics, rec_params)
    - Optimization Tab (opt_df / opt_df_batch)
    - Backtest Tab (bt_last_result)
    - Portfolio Tab (portfolio_snapshot, risk_limits)
    - Macro Tab (macro_snapshot)
    - Fair Value Advisor (fv_advisor_runs) אם קיים
    """
    st.markdown("### 📊 סקירה כוללת (Overview) — Investment Committee View")
    st.caption(
        "מסך אחד שמרכז את האבחון לזוג הנבחר, תוצאות האופטימיזציה והבק-טסט, "
        "מצב הפורטפוליו והמאקרו, לפני הכנסת כסף אמיתי."
    )

    ctx_dict = st.session_state.get("ctx", {}) or {}
    with st.expander("🧬 Context (ctx) מהטאבים השונים", expanded=False):
        if ctx_dict:
            st.json(ctx_dict)
        else:
            st.info("עדיין לא נוצר ctx (למשל מטאבי Backtest / Optimization / Overview).")

    # ---------- 1. Pair Diagnostics ----------
    st.markdown("### 1️⃣ זוג נבחר ו-Health סטטיסטי")

    pair_key = _get_selected_pair_key()
    if not pair_key:
        st.info("לא נמצא זוג נבחר. עבור קודם לטאב **🧪 ניתוח זוג** ובחר זוג.")
        return

    st.markdown(f"**זוג נוכחי:** `{pair_key}`")

    diag, rec, rec_params = _get_pair_containers(pair_key)

    col_a, col_b = st.columns(2)
    with col_a:
        if diag:
            st.markdown("**PairDiagnostics — מדדים עיקריים**")
            diag_df = pd.DataFrame([diag])
            cols_pref = [
                "n_obs",
                "coint_pvalue",
                "adf_pvalue",
                "hurst",
                "halflife",
                "corr",
                "beta",
                "spread_vol",
            ]
            cols_show = [c for c in cols_pref if c in diag_df.columns]
            if cols_show:
                st.dataframe(diag_df[cols_show], width = "stretch")
            else:
                st.dataframe(diag_df, width = "stretch")
        else:
            st.info("אין PairDiagnostics שמורים לזוג הזה (עדיין).")

    with col_b:
        if rec:
            st.markdown("**המלצות Recommender לזוג**")
            rec_show = {k: rec[k] for k in sorted(rec.keys())}
            st.json(rec_show)
        else:
            st.info("אין pair_recommendations לזוג הזה — תריץ ניתוח בטאב Pair.")

    with st.expander("📋 Full PairDiagnostics (raw)", expanded=False):
        if diag:
            st.json(diag)
        else:
            st.info("אין אבחון מלא לזוג הזה.")

    pair_quality = _classify_pair_quality(diag)
    if pair_quality == "good":
        st.success("Pair statistical quality: GOOD — Mean-reversion profile looks solid.")
    elif pair_quality == "bad":
        st.error("Pair statistical quality: BAD — האיתותים הסטטיסטיים חלשים / לא עקביים.")
    else:
        st.warning("Pair statistical quality: CAUTION — יש סימנים מעורבים, שווה להעמיק לפני השקעה כבדה.")

    # ---------- 2. Recommender vs Optimization vs Backtest ----------
    st.markdown("### 2️⃣ Recommender vs Optimization vs Backtest")

    opt_df = _get_opt_df()
    bt_res_dict = _get_bt_last_result_dict()

    if opt_df is None:
        st.info("לא נמצאו תוצאות אופטימיזציה (opt_df/opt_df_batch). עבור לטאב **⚙️ אופטימיזציה** והרץ ריצה.")
    if bt_res_dict is None:
        st.info("לא נמצא בק-טסט אחרון (bt_last_result). עבור לטאב **📈 Backtest** והרץ ריצה.")

    opt_row = _extract_best_opt_row_for_pair(opt_df, pair_key) if opt_df is not None else None
    bt_cfg: Optional[JSONDict] = None
    bt_metrics: JSONDict = {}

    if bt_res_dict is not None:
        bt_cfg = bt_res_dict.get("config") or {}
        bt_metrics = bt_res_dict.get("metrics") or {}

    col1, col2, col3 = st.columns(3)
    with col1:
        if opt_row is not None:
            st.markdown("**Opt. Best Row — KPIs**")
            s = opt_row
            kpi_candidates = [
                "Score",
                "Sharpe",
                "Calmar",
                "CAGR",
                "Profit",
                "Trades",
            ]
            kpis = {}
            for name in kpi_candidates:
                for c in s.index:
                    if c.lower() == name.lower():
                        kpis[name] = s[c]
                        break
            if kpis:
                st.dataframe(pd.DataFrame([kpis]), width = "stretch")
            else:
                st.dataframe(pd.DataFrame([s.to_dict()]), width = "stretch")
        else:
            st.info("אין שורת אופטימיזציה מתאימה לזוג הזה ב-opt_df.")

    with col2:
        if bt_metrics:
            st.markdown("**Backtest אחרון — KPIs**")
            main_keys = [
                "Sharpe",
                "Sortino",
                "Calmar",
                "CAGR",
                "MaxDD",
                "Trades",
                "PnL",
            ]
            m_show = {k: bt_metrics.get(k) for k in main_keys if k in bt_metrics}
            if not m_show:
                m_show = bt_metrics
            st.dataframe(pd.DataFrame([m_show]), width = "stretch")
        else:
            st.info("אין מדדי Backtest זמינים.")

    with col3:
        if rec_params:
            st.markdown("**Pair Rec. params**")
            st.dataframe(pd.DataFrame([rec_params]), width = "stretch")
        else:
            st.info("אין pair_rec_params לזוג הזה.")

    comp_df = _build_param_comparison_table(rec_params, opt_row, bt_cfg)
    if comp_df is not None:
        st.markdown("#### 🔍 השוואת פרמטרים (Recommender / Optimization / Backtest)")
        st.dataframe(comp_df, width = "stretch")

        # אם Plotly קיים — נציג deviation plot
        if px is not None:
            df_dev = comp_df[["Parameter", "Δ Opt - Rec", "Δ BT - Opt"]].copy()
            df_long = df_dev.melt(id_vars="Parameter", var_name="Delta", value_name="Value")
            fig = px.bar(
                df_long,
                x="Parameter",
                y="Value",
                color="Delta",
                title="סטיות פרמטרים בין Recommender / Opt / Backtest",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, width = "stretch")
    else:
        st.info("לא הצלחנו לבנות טבלת השוואת פרמטרים — חסרים פרמטרים חופפים בין המקורות.")

    bt_quality = _classify_backtest_quality(bt_metrics) if bt_metrics else "caution"
    if bt_quality == "good":
        st.success("Backtest quality: GOOD — היסטורית, הזוג התנהג יפה תחת הפרמטרים שנבדקו.")
    elif bt_quality == "bad":
        st.error("Backtest quality: BAD — ביצועים היסטוריים חלשים / תנודתיות גבוהה מדי.")
    else:
        st.warning("Backtest quality: CAUTION — התמונה ההיסטורית לא חד-משמעית, צריך זהירות.")

    # ---------- 3. Portfolio & Risk ----------
    st.markdown("### 3️⃣ מצב פורטפוליו וסיכון (Portfolio & Risk)")

    port_snap = _get_portfolio_snapshot()
    risk_limits = _get_risk_limits()

    colp1, colp2 = st.columns(2)
    with colp1:
        if port_snap:
            st.markdown("**Portfolio snapshot**")
            keys_pref = [
                "total_equity",
                "cash",
                "gross_exposure",
                "net_exposure",
                "var_95",
                "es_97",
                "n_positions",
            ]
            show = {k: port_snap.get(k) for k in keys_pref if k in port_snap}
            if not show:
                show = port_snap
            st.dataframe(pd.DataFrame([show]), width = "stretch")
        else:
            st.info("אין portfolio_snapshot ב-session. טאב Portfolio כנראה לא רץ עדיין.")

    with colp2:
        if risk_limits:
            st.markdown("**Risk limits (יעדי סיכון / תקרות)**")
            st.json(risk_limits)
        else:
            st.info("אין risk_limits ב-session. אפשר להגדיר אותם בקובץ קונפיג / טאב Risk Engine עתידי.")

    if port_snap and risk_limits:
        port_quality = _classify_portfolio_risk(port_snap, risk_limits)
    else:
        port_quality = "caution"

    if port_quality == "good":
        st.success("Portfolio risk state: GOOD — מתחת לגבולות הסיכון שהוגדרו.")
    elif port_quality == "bad":
        st.error("Portfolio risk state: BAD — חלק מהגבולות (VaR / חשיפה) נראים חריגים יחסית ל-limits.")
    else:
        st.warning("Portfolio risk state: CAUTION — חסר מידע/גבולות או שהפורטלופיו קרוב לגבולות הסיכון.")

    # אם יש חשיפה לזוג הזה בפורטפוליו — נדווח
    if port_snap:
        pair_expo = port_snap.get("pair_exposure") or {}
        if isinstance(pair_expo, dict) and pair_key in pair_expo:
            st.markdown("**חשיפה קיימת לזוג הזה בפורטפוליו**")
            st.json({pair_key: pair_expo[pair_key]})

    # ---------- 4. Macro Overlay ----------
    st.markdown("### 4️⃣ מצב מאקרו ביחס לאסטרטגיה")

    macro_snap = _get_macro_snapshot()
    if macro_snap:
        colm1, colm2 = st.columns([2, 1])
        with colm1:
            st.markdown("**Macro snapshot**")
            st.json(macro_snap)
        with colm2:
            regime = str(macro_snap.get("regime_name", "N/A"))
            vol_regime = str(macro_snap.get("vol_regime", "N/A"))
            st.metric("Regime", regime)
            st.metric("Vol regime", vol_regime)
    else:
        st.info("אין macro_snapshot ב-session. טאב Macro כנראה לא רץ עדיין.")

    macro_quality = _classify_macro_regime(macro_snap)

    if macro_quality == "good":
        st.success("Macro regime: GOOD — תנאי שוק סבירים למסחר זוגי mean-reversion.")
    elif macro_quality == "bad":
        st.error("Macro regime: BAD — תנאי שוק קיצוניים / משבריים, זהירות גבוהה לפני הגדלת חשיפה.")
    else:
        st.warning("Macro regime: CAUTION — תמונת המאקרו לא חד-משמעית או לא מוגדרת.")

    # ---------- 5. Fair Value Advisor Snapshot (אם קיים) ----------
    st.markdown("### 5️⃣ Fair Value Advisor (אם רץ)")

    fv_run = _get_latest_fv_advisor()
    if fv_run:
        summary = fv_run.get("summary", {})
        advice = fv_run.get("advice", []) or []
        st.markdown("**Advisor summary (אחרון שרץ)**")
        st.json(summary)

        st.markdown("**Top 3 עצות אחרונות**")
        # ניקח עד 3 עצות הכי חמורות (critical > warning > info)
        def _sev_rank(sev: str) -> int:
            s = (sev or "").lower()
            if s == "critical":
                return 0
            if s == "warning":
                return 1
            return 2

        sorted_adv = sorted(advice, key=lambda a: _sev_rank(a.get("severity", "info")))
        for item in sorted_adv[:3]:
            sev = item.get("severity", "info").upper()
            cat = item.get("category", "")
            msg = item.get("message", "")
            rat = item.get("rationale", "")
            st.markdown(f"- **[{sev}] {cat}** — {msg}")
            if rat:
                st.caption(rat)
    else:
        st.info("לא נמצאה ריצת Advisor אחרונה. אפשר להריץ בטאב **🧬 Fair Value API**.")

    # ---------- 6. Investment Readiness Verdict ----------
    st.markdown("### 6️⃣ החלטה לפני השקעה (Investment Readiness)")

    verdict = _overall_verdict(
        pair_q=pair_quality,
        bt_q=bt_quality,
        port_q=port_quality,
        macro_q=macro_quality,
    )

    if verdict == "GO":
        st.success("✅ Overall verdict: **GO** — התנאים נראים טובים יחסית מבחינת סטטיסטיקה, Backtest, סיכון ומאקרו.")
    elif verdict == "NO-GO":
        st.error("❌ Overall verdict: **NO-GO** — יש לפחות תחום אחד שנראה בעייתי ברמה גבוהה.")
    else:
        st.warning("⚠️ Overall verdict: **CAUTION** — יש שילוב של נקודות טובות וחלשות, צריך זהירות/כיוונון נוסף.")

    bullets: List[str] = []

    if pair_quality == "bad":
        bullets.append("• לשקול לבחור זוג אחר או להקשיח פילטרים סטטיסטיים (cointegration / halflife / Hurst).")
    elif pair_quality == "caution":
        bullets.append("• לעבור שוב על PairDiagnostics ולוודא שהזוג באמת מתנהג mean-reverting ולא טרנדי.")

    if bt_quality == "bad":
        bullets.append("• לבצע backtest עמוק יותר / Walk-forward, או לעדכן פרמטרים בהתאם ל-Opt best.")
    elif bt_quality == "caution":
        bullets.append("• לחזור לטאב Backtest ולבחון sensitivity (כמה הרגישות לפרמטרים גבוהה).")

    if port_quality == "bad":
        bullets.append("• לצמצם חשיפה בפוזיציות קיימות או להעלות מרחק ביטחון מגבולות VaR/חשיפה לפני פתיחת הזוג.")
    elif port_quality == "caution":
        bullets.append("• לוודא שהזוג לא מגדיל חשיפה לסקטור/מדינה כבר עמוסים בתיק.")

    if macro_quality == "bad":
        bullets.append("• להתחשב בכך שהמאקרו עוין ל-mean-reversion pairs (מומלץ להקטין sizing או לדחות כניסה).")
    elif macro_quality == "caution":
        bullets.append("• לשקול התאמת sizing/thresholds (z_in/z_out) לתנאי המאקרו הנוכחיים.")

    if fv_run is not None:
        bullets.append("• לעבור על עצות ה-Advisor ולראות אילו פרמטרים Structural כדאי לשנות (window, z_in/z_out, וכו').")

    if not bullets:
        bullets.append("• עדיין אין מספיק מידע מכל הטאבים כדי לקבל החלטה חזקה — להשלים נתונים חסרים (Opt / BT / Macro / Portfolio).")

    st.markdown("**Check-list פרקטי:**")
    for b in bullets:
        st.write(b)

    st.caption(
        "הטאב הזה הוא מסך החלטה ברמת קרן: הוא לא מחליף שיקול דעת, "
        "אבל מרכז עבורך את כל ההיבטים החשובים לפני הכנסת כסף אמיתי."
    )


__all__ = ["render_overview_tab"]
