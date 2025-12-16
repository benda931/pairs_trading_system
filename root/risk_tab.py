# -*- coding: utf-8 -*-
"""
root/risk_tab.py — Fund-Level Risk Dashboard (HF-grade, Real Data Only, v2.5)
==============================================================================

טאב Streamlit ברמת קרן גידור שמחובר ישירות ל-core.risk_engine:

- משתמש ב: RiskLimits, risk_assessment_to_dashboard_dict.
- מנתח היסטוריית Equity/PnL *אמיתית בלבד*.
- מציג:
    1. Overall Risk Score + Scaling factor + Risk Label.
    2. Risk Goals vs Realized (Vol / DD / Sharpe).
    3. Risk Timeline (rolling Vol / DD) – אם מחושב במנוע.
    4. Bucket Risks & Bucket Scaling suggestions – אם קיימים.
    5. Kill-Switch status + Severity + Reason.
    6. Summary טקסטואלי מהמנוע.
    7. Data Quality & Length.
    8. Macro-Risk Overlay (אינטגרציה עם Macro Tab).
    9. Smart Scan Risk Overlay (אינטגרציה עם Smart Scan Tab).
    10. What-if Scaling — סימולציה פשוטה לשינוי סיכון תחת סקיילינג אחר.
    11. Risk Assessment History — השוואת הערכות סיכון בין ריצות.

אין שום דמו:
-------------
אם אין היסטוריה אמיתית (Session / SqlStore) → מציג הודעת תצורה ולא ממציא נתונים.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np

from core.app_context import AppContext
from core.risk_engine import (
    RiskLimits,
    risk_assessment_to_dashboard_dict,
)


# ============================================================
# 1) Data loading — Session / SqlStore
# ============================================================


def _get_sql_store(app_ctx: AppContext):
    """מוציא SqlStore מתוך AppContext, אם קיים."""
    store = getattr(app_ctx, "sql_store", None)
    if store is not None:
        return store

    services = getattr(app_ctx, "services", None)
    if isinstance(services, dict):
        return services.get("sql_store")

    return None


def _load_history_from_session() -> Optional[pd.DataFrame]:
    """
    מנסה לטעון היסטוריית Equity/PnL מתוך session_state:

    סדר עדיפויות:
        1) risk_hist_df
        2) backtest_history_df
        3) portfolio_history_df
    """
    try:
        sess = st.session_state
    except Exception:
        return None

    for key in ("risk_hist_df", "backtest_history_df", "portfolio_history_df"):
        df = sess.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df

    return None


def _load_history_from_sql_store(app_ctx: AppContext) -> Optional[pd.DataFrame]:
    """
    Hook לטעינת היסטוריה מ-SqlStore:

    מניח טבלה equity_curve עם שדות:
        date, equity, pnl

    אפשר להחליף/להתאים לסכמה שלך.
    """
    store = _get_sql_store(app_ctx)
    if store is None:
        return None

    try:
        df = store.read_table("equity_curve")
    except Exception:
        return None

    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("ts") or cols.get("timestamp")
    equity_col = cols.get("equity") or cols.get("total_equity") or cols.get("nav")
    pnl_col = cols.get("pnl") or cols.get("pl") or cols.get("profit")

    if not date_col or not equity_col:
        return None

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col]),
            "Equity": pd.to_numeric(df[equity_col], errors="coerce"),
        }
    )

    if pnl_col:
        out["PnL"] = pd.to_numeric(df[pnl_col], errors="coerce")
    else:
        # אם אין PnL מפורש – ננסה לגזור משינויי Equity
        out["PnL"] = out["Equity"].diff().fillna(0.0)

    out = out.dropna(subset=["date", "Equity"])
    return out if not out.empty else None


def _load_history_for_risk(app_ctx: AppContext) -> Optional[pd.DataFrame]:
    """
    מקור יחיד להיסטוריית Equity/PnL עבור ה-Risk Engine.

    לוגיקה:
    --------
    1. session_state (risk_hist_df / backtest_history_df / portfolio_history_df)
    2. SqlStore (equity_curve) — אם תגדיר טבלה כזו.
    3. אם אין – מחזיר None, וה-UI יסביר שאין דאטה (בלי דמו).
    """
    df = _load_history_from_session()
    if df is not None and not df.empty:
        return df

    df = _load_history_from_sql_store(app_ctx)
    if df is not None and not df.empty:
        return df

    return None


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    מנרמל את היסטוריית הסיכון:
    - מבטיח עמודות date / Equity / PnL.
    - ממיין לפי תאריך.
    - מסיר כפילויות.
    """
    out = df.copy()
    cols = {c.lower(): c for c in out.columns}
    date_col = cols.get("date") or cols.get("ts") or cols.get("timestamp")
    equity_col = cols.get("equity") or cols.get("total_equity") or cols.get("nav")
    pnl_col = cols.get("pnl") or cols.get("pl") or cols.get("profit")

    if date_col and date_col != "date":
        out.rename(columns={date_col: "date"}, inplace=True)
    if equity_col and equity_col != "Equity":
        out.rename(columns={equity_col: "Equity"}, inplace=True)
    if pnl_col and pnl_col != "PnL":
        out.rename(columns={pnl_col: "PnL"}, inplace=True)

    if "date" not in out.columns or "Equity" not in out.columns:
        return out

    out["date"] = pd.to_datetime(out["date"])
    out = out.dropna(subset=["date", "Equity"])
    out = out.sort_values("date").drop_duplicates(subset=["date"])

    if "PnL" not in out.columns:
        out["PnL"] = out["Equity"].diff().fillna(0.0)

    return out


# ============================================================
# 2) Panels — תתי-רכיבים של הטאב
# ============================================================


def _render_equity_preview(hist: pd.DataFrame) -> None:
    """תצוגת Equity Curve + Data Quality ב-expander."""
    with st.expander("📈 Equity curve & Data Quality", expanded=False):
        df = hist.copy()
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
        except Exception:
            pass

        try:
            eq = df.set_index("date")["Equity"]
            st.line_chart(eq)
        except Exception:
            st.caption("לא ניתן להציג Equity curve (בעיה בעמודות / אינדקס).")

        # Data quality & span
        try:
            n_rows = len(df)
            start = df["date"].min()
            end = df["date"].max()
            days_span = (end - start).days if pd.notna(start) and pd.notna(end) else None
            pnl_nan_ratio = float(df["PnL"].isna().mean()) if "PnL" in df.columns else None

            info_rows = []
            info_rows.append({"metric": "N rows", "value": n_rows})
            if days_span is not None:
                info_rows.append({"metric": "Span (days)", "value": days_span})
            if pnl_nan_ratio is not None:
                info_rows.append({"metric": "PnL NaN ratio", "value": f"{pnl_nan_ratio:.1%}"})

            st.caption("Data quality summary:")
            st.table(pd.DataFrame(info_rows))

            if n_rows < 60:
                st.warning("⚠️ פחות מ-60 תצפיות — הערכת הסיכון עלולה להיות לא יציבה.")
            elif n_rows < 200:
                st.info("ℹ️ היסטוריית הסיכון יחסית קצרה (פחות מ-200 תצפיות).")
        except Exception:
            pass


def _render_pnl_distribution(hist: pd.DataFrame) -> None:
    """פאנל: התפלגות PnL + מדדי Tail Risk פשוטים."""
    with st.expander("📊 PnL Distribution & Tail Risk", expanded=False):
        if "PnL" not in hist.columns:
            st.caption("אין עמודת PnL בהיסטוריה — לא ניתן להציג התפלגות PnL.")
            return

        pnl = pd.to_numeric(hist["PnL"], errors="coerce").dropna()
        if pnl.empty:
            st.caption("עמודת PnL ריקה / לא תקינה.")
            return

        # סטטיסטיקות בסיסיות
        mean = float(pnl.mean())
        std = float(pnl.std(ddof=1))
        p5 = float(np.percentile(pnl, 5))
        p1 = float(np.percentile(pnl, 1))

        rows = [
            {"metric": "Mean PnL", "value": f"{mean:.4f}"},
            {"metric": "Std PnL", "value": f"{std:.4f}"},
            {"metric": "5% worst (≈VaR)", "value": f"{p5:.4f}"},
            {"metric": "1% worst (≈Tail)", "value": f"{p1:.4f}"},
        ]
        st.table(pd.DataFrame(rows))

        # Histogram
        try:
            hist_counts, bin_edges = np.histogram(pnl, bins=30)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            df_hist = pd.DataFrame(
                {
                    "bin_center": bin_centers,
                    "count": hist_counts,
                }
            ).set_index("bin_center")
            st.bar_chart(df_hist)
        except Exception:
            st.caption("לא ניתן לחשב histogram על PnL (בעיה בנתונים).")


def _render_overall_risk_header(risk_dict: Dict[str, Any]) -> None:
    """Header עיקרי של מצב הסיכון + המלצות גסות."""
    overall_score = risk_dict.get("overall_risk_score")
    risk_label = risk_dict.get("risk_label")
    scaling_factor = risk_dict.get("scaling_factor")
    reward = risk_dict.get("reward", {}) or {}
    kill = risk_dict.get("kill_switch", {}) or {}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Overall Risk Score",
            value=f"{overall_score:.1f}" if overall_score is not None else "N/A",
            help="ציון 0–100 של מצב הסיכון הכולל לפי מנוע הסיכון.",
        )
        st.caption(f"Risk label: `{risk_label or 'N/A'}`")

    with col2:
        st.metric(
            label="Suggested Scaling (gross)",
            value=f"{scaling_factor:.2f}x" if scaling_factor is not None else "N/A",
            help="פקטור מומלץ למכפלת gross exposure לפי מצב הסיכון.",
        )
        if reward:
            rr_score = reward.get("risk_reward_score")
            st.caption(f"Risk/Reward score: `{rr_score}`")

    with col3:
        ks_enabled = bool(kill.get("enabled"))
        ks_mode = kill.get("mode", "N/A")
        severity = kill.get("severity_score", None)

        st.metric(
            label="Kill-Switch",
            value="ON" if ks_enabled else "OFF",
            help=f"Mode: {ks_mode}",
        )
        if severity is not None:
            st.caption(f"Kill severity: `{severity:.2f}`")

    # פרשנות גסה
    if overall_score is not None:
        if overall_score >= 80:
            st.success("✅ רמת הסיכון ביחס ליעדים מעולה – אפשר לשקול העלאת חשיפה זהירה.")
        elif overall_score >= 60:
            st.info("ℹ️ רמת הסיכון סבירה – אין צורך בפעולות דרמטיות כרגע.")
        elif overall_score >= 40:
            st.warning("⚠️ רמת סיכון בינונית-גבוהה – כדאי לבדוק Buckets וחשיפה סקטוריאלית.")
        else:
            st.error("🛑 רמת סיכון גבוהה מאוד – מומלץ לשקול הורדת חשיפה או הפעלת Kill-Switch.")

    st.markdown("---")


def _render_goals_vs_realized(risk_dict: Dict[str, Any]) -> None:
    """טבלת יעד מול ביצוע: Vol / DD / Sharpe."""
    st.markdown("### 🎯 Risk Goals vs Realized")

    goals_gap = risk_dict.get("goals_gap", {}) or {}
    goals = goals_gap.get("goals", {}) or {}
    realized = goals_gap.get("realized", {}) or {}
    gaps = goals_gap.get("gaps", {}) or {}

    rows = [
        {
            "Metric": "Vol (annual)",
            "Target": goals.get("target_vol_pct"),
            "Realized": realized.get("realized_vol_pct"),
            "Gap": gaps.get("vol_gap"),
        },
        {
            "Metric": "Max Drawdown",
            "Target": goals.get("target_dd_pct"),
            "Realized": realized.get("realized_max_dd_pct"),
            "Gap": gaps.get("dd_gap"),
        },
        {
            "Metric": "Sharpe",
            "Target": f"[{goals.get('target_sharpe_min')}, {goals.get('target_sharpe_max')}]",
            "Realized": realized.get("realized_sharpe"),
            "Gap": gaps.get("sharpe_gap"),
        },
    ]
    df_goals = pd.DataFrame(rows)

    def _fmt(x):
        if x is None:
            return ""
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    df_show = df_goals.copy()
    for col in ("Target", "Realized", "Gap"):
        df_show[col] = df_show[col].apply(_fmt)

    st.dataframe(df_show, width="stretch", height=200)
    st.markdown("---")


def _render_risk_timeline(risk_dict: Dict[str, Any]) -> None:
    """Risk Timeline: rolling Vol / DD לאורך זמן (אם קיים)."""
    st.markdown("### ⏱ Risk Timeline (rolling Vol / DD)")

    tl_df = risk_dict.get("risk_timeline")
    if not isinstance(tl_df, pd.DataFrame) or tl_df.empty:
        st.caption("אין Risk Timeline מחושב (risk_timeline ריק או לא הוחזר מהמנוע).")
        return

    tl = tl_df.copy()
    if "date" in tl.columns:
        tl["date"] = pd.to_datetime(tl["date"])
        tl.set_index("date", inplace=True)

    col_vol, col_dd = st.columns(2)

    with col_vol:
        st.caption("Rolling Vol (annualized)")
        if "rolling_vol_pct" in tl.columns:
            st.line_chart(tl["rolling_vol_pct"])
        else:
            st.caption("אין עמודה rolling_vol_pct ב-risk_timeline.")

    with col_dd:
        st.caption("Rolling Drawdown")
        if "rolling_dd_pct" in tl.columns:
            st.line_chart(tl["rolling_dd_pct"])
        else:
            st.caption("אין עמודה rolling_dd_pct ב-risk_timeline.")

    st.markdown("---")


def _render_buckets_and_scaling(risk_dict: Dict[str, Any]) -> None:
    """Bucket Risks + Scaling Suggestions (אם קיימים)."""
    st.markdown("### 🧩 Bucket Risks & Scaling Suggestions")

    bucket_risks = risk_dict.get("bucket_risks", []) or []
    bucket_scaling = risk_dict.get("bucket_scaling", []) or []

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("#### Top Bucket Risks")
        if bucket_risks:
            df_br = pd.DataFrame(bucket_risks)
            sort_col = None
            for cand in ("contribution_to_risk", "risk_score", "vol_pct"):
                if cand in df_br.columns:
                    sort_col = cand
                    break
            if sort_col:
                df_br = df_br.sort_values(sort_col, ascending=False)
            st.dataframe(df_br, width="stretch", height=260)
        else:
            st.caption("אין bucket_risks ב-risk_dict (כנראה לא הועבר bucket_col).")

    with col_b2:
        st.markdown("#### Bucket Scaling")
        if bucket_scaling:
            df_bs = pd.DataFrame(bucket_scaling)
            st.dataframe(df_bs, width="stretch", height=260)
        else:
            st.caption("אין bucket_scaling מחושב (או שאין bucket data).")

    st.markdown("---")


def _render_kill_switch_details(risk_dict: Dict[str, Any]) -> None:
    """Kill-Switch: מצב, סיבה, breached limits."""
    st.markdown("### ⛔ Kill-Switch Details")

    kill = risk_dict.get("kill_switch", {}) or {}
    if not kill:
        st.caption("לא התקבל מידע Kill-Switch מהמנוע (kill_switch ריק).")
        return

    ks_enabled = bool(kill.get("enabled"))
    ks_mode = kill.get("mode", "N/A")
    severity = kill.get("severity_score", None)
    reason = kill.get("reason") or kill.get("reason_text")
    breached_limits = kill.get("breached_limits", [])

    col_k1, col_k2 = st.columns(2)

    with col_k1:
        st.write(f"**Enabled:** `{ks_enabled}`  •  **Mode:** `{ks_mode}`")
        if severity is not None:
            st.write(f"**Severity score:** `{severity:.2f}`")
        st.write(f"**Trigger source:** `{kill.get('trigger_source', 'unknown')}`")

    with col_k2:
        if reason:
            st.write("**Reason:**")
            st.write(reason)
        if breached_limits:
            st.write("**Breached limits:**")
            for b in breached_limits:
                st.write(f"- {b}")

    st.markdown("---")


def _render_risk_summary(risk_dict: Dict[str, Any]) -> None:
    """Summary טקסטואלי מלא מהמנוע."""
    st.markdown("### 🧾 Risk Summary (from engine)")

    summary = risk_dict.get("summary", {}) or {}
    if not summary:
        st.caption("אין summary ב-risk_dict (summary ריק).")
        return

    with st.expander("הצג Summary מלא", expanded=False):
        st.json(summary)


# ============================================================
# 2.1) Macro-Risk Panel — אינטגרציה עם המאקרו
# ============================================================


def _render_macro_risk_panel() -> None:
    """
    פאנל מאקרו לטאב הריסק — משתמש במידע מה-Macro Tab:

    session_state keys (אם קיימים):
    - macro_regime_label
    - macro_regime_snapshot (dict עם risk_on / inflation / growth)
    - macro_factor_summary_text
    - macro_risk_alert
    - macro_risk_budget_hint
    """
    st.markdown("### 🌐 Macro Overlay (from Macro Tab)")

    sess = st.session_state
    macro_regime_label = sess.get("macro_regime_label")
    macro_regime_snapshot = sess.get("macro_regime_snapshot")
    macro_factor_summary = sess.get("macro_factor_summary_text")
    macro_risk_alert = bool(sess.get("macro_risk_alert", False))
    macro_risk_budget_hint = sess.get("macro_risk_budget_hint", None)

    if not (macro_regime_label or macro_regime_snapshot or macro_factor_summary or macro_risk_budget_hint):
        st.caption("לא נמצא מידע מאקרו ב-session_state. הרץ קודם את Macro Tab.")
        st.markdown("---")
        return

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        if macro_regime_label:
            st.metric("Macro Regime", macro_regime_label)
        if macro_risk_budget_hint is not None:
            try:
                st.metric("Macro Risk Budget Hint", f"{float(macro_risk_budget_hint):.2f}x")
            except Exception:
                st.metric("Macro Risk Budget Hint", str(macro_risk_budget_hint))

        if macro_risk_alert:
            st.error("Macro Risk Alert: מצב Risk-Off/Stress לפי המאקרו.")

    with col_m2:
        if isinstance(macro_regime_snapshot, dict):
            st.caption("Regime snapshot (risk_on / inflation / growth):")
            st.json(macro_regime_snapshot)
        if macro_factor_summary:
            st.caption("Macro factor summary:")
            st.markdown(macro_factor_summary)

    st.markdown("---")


# ============================================================
# 2.2) Smart Scan Risk Panel — אינטגרציה עם Smart Scan
# ============================================================


def _render_smart_scan_risk_panel() -> None:
    """
    פאנל שמראה איך Smart Scan תורם לתמונת הסיכון:

    משתמש ב:
    - smart_scan_results (DataFrame של df_scan)
    - smart_scan_last_meta
    - smart_scan_macro_overlay_pairs (multipliers/include/scores)
    """
    st.markdown("### 🧪 Smart Scan Risk Overlay")

    sess = st.session_state
    df_scan = sess.get("smart_scan_results")
    meta = sess.get("smart_scan_last_meta", {}) or {}
    overlay = sess.get("smart_scan_macro_overlay_pairs", {}) or {}

    if not isinstance(df_scan, pd.DataFrame) or df_scan.empty:
        st.caption("אין smart_scan_results ב-session_state. הרץ קודם את Smart Scan Tab.")
        st.markdown("---")
        return

    ranking_metric = meta.get("ranking_metric", "smart_score_total")
    if ranking_metric not in df_scan.columns:
        ranking_metric = "smart_score_total"

    df_view = df_scan.sort_values(ranking_metric, ascending=False).copy()

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.caption(f"Top 20 pairs by `{ranking_metric}` (Smart Scan):")
        cols_to_show = ["pair", ranking_metric]
        if "smart_score_with_macro" in df_view.columns:
            cols_to_show.append("smart_score_with_macro")
        if "stability_ratio" in df_view.columns:
            cols_to_show.append("stability_ratio")
        st.dataframe(df_view.head(20)[cols_to_show], width="stretch", height=260)

    with col_s2:
        st.caption("Macro overlay usage on pairs:")
        if overlay:
            try:
                n_pairs_overlay = len(overlay.get("multipliers", {}))
                n_excluded = sum(1 for v in overlay.get("filters", {}).values() if not v)
            except Exception:
                n_pairs_overlay, n_excluded = None, None

            rows = []
            rows.append({"metric": "#pairs in overlay", "value": n_pairs_overlay})
            rows.append({"metric": "#pairs excluded by macro", "value": n_excluded})
            df_overlay_info = pd.DataFrame(rows)
            st.table(df_overlay_info)
        else:
            st.caption("לא נמצא smart_scan_macro_overlay_pairs – אולי המאקרו כובה בסריקה האחרונה.")

    st.markdown("---")


# ============================================================
# 2.3) What-if Scaling — סימולציה פשוטה
# ============================================================


def _render_what_if_scaling(risk_dict: Dict[str, Any]) -> None:
    """
    What-if Scaling — מציע סקיילינג אחר על בסיס ההערכה הנוכחית:

    הנחות פשוטות:
    - Vol נמדד בקירוב ∝ scaling_factor.
    - Max DD נמדד בקירוב ∝ scaling_factor.
    זה לא תחליף לסימולציה אמיתית, אבל נותן אינדיקציה מהירה.
    """
    st.markdown("### 🎚 What-if Scaling Simulation")

    goals_gap = risk_dict.get("goals_gap", {}) or {}
    realized = goals_gap.get("realized", {}) or {}
    vol_real = realized.get("realized_vol_pct")
    dd_real = realized.get("realized_max_dd_pct")

    if vol_real is None and dd_real is None:
        st.caption("אין מספיק מידע על Vol/DD כדי לבצע סימולציה (GoalsGap.realized ריק).")
        st.markdown("---")
        return

    base_scaling = risk_dict.get("scaling_factor") or 1.0
    scale = st.slider(
        "Gross exposure scaling (what-if)",
        min_value=0.1,
        max_value=2.5,
        value=float(base_scaling),
        step=0.05,
        help="מניח שסיכון (Vol/DD) גדל/קטן לינארית עם המכפיל.",
        key="risk_whatif_scaling",
    )

    rows = []
    if vol_real is not None:
        vol_scaled = float(vol_real) * (scale / float(base_scaling or 1.0))
        rows.append({"metric": "Vol (realized, scaled)", "value": f"{vol_scaled:.2f}%"})
    if dd_real is not None:
        dd_scaled = float(dd_real) * (scale / float(base_scaling or 1.0))
        rows.append({"metric": "Max DD (realized, scaled)", "value": f"{dd_scaled:.2f}%"})

    if rows:
        st.table(pd.DataFrame(rows))
    else:
        st.caption("לא הצלחנו לחשב Vol/DD scaled.")

    st.markdown("---")


# ============================================================
# 3) Tab Renderer — הפונקציה הראשית
# ============================================================


def render_risk_tab(
    app_ctx: AppContext,
    feature_flags: Dict[str, Any],
    nav_payload: Optional[Dict[str, Any]] = None,
) -> None:
    """
    נקודת כניסה רשמית לטאב 'risk' בדשבורד.

    זרימה:
    -------
    1. טוענים היסטוריה *אמיתית* (Session / SqlStore) ומנרמלים.
    2. מציגים Equity preview + Data Quality + PnL Distribution.
    3. בונים RiskLimits בסיסי (ניתן להחליף מקובץ קונפיג בעתיד).
    4. מריצים risk_assessment_to_dashboard_dict.
    5. מציגים:
        • Overall Risk Header.
        • Goals vs Realized.
        • Risk Timeline.
        • Bucket risks & scaling.
        • Kill-switch details.
        • Macro-Risk panel (אם קיים מידע מאקרו).
        • Smart Scan Risk overlay (אם קיים smart_scan_results).
        • What-if Scaling Simulation.
        • Summary מלא.
    """

    st.title("⚠️ Fund-Level Risk Dashboard (Real Data, HF-grade)")

    st.markdown(
        """
        הטאב הזה נותן **תמונת מצב סיכון ברמת הקרן** על בסיס דאטה אמיתי בלבד:

        - Overall Risk Score + Scaling factor.  
        - Risk Goals מול ביצועים בפועל (Vol / DD / Sharpe).  
        - Risk Timeline (Vol / DD גלגלי) אם קיים.  
        - Bucket risks + הצעות Scaling (אם הועבר bucket_col למנוע).  
        - Kill-Switch status + Reason + breached limits.  
        - Macro overlay + Smart Scan overlay לחיבור עם שאר הטאבים במערכת.  

        אין כאן שום סימולציה או דמו – הכל יוצא מהיסטוריית Equity/PnL אמיתית
        מה-Session או מה-SqlStore.
        """
    )

    # 1) היסטוריית Equity/PnL
    hist_raw = _load_history_for_risk(app_ctx)
    if hist_raw is None or hist_raw.empty:
        st.error(
            "לא נמצאה היסטוריית Equity/PnL לניתוח.\n\n"
            "שים DataFrame ב-session_state תחת אחד המפתחות:\n"
            " • risk_hist_df\n"
            " • backtest_history_df\n"
            " • portfolio_history_df\n"
            "או הטען טבלה מתאימה ל-SqlStore (למשל equity_curve) ועדכן את _load_history_from_sql_store."
        )
        return

    df = _normalize_history(hist_raw)
    if "date" not in df.columns or "Equity" not in df.columns:
        st.error(
            "היסטוריית הסיכון חייבת לכלול לפחות עמודות 'date' ו-'Equity' לאחר הנרמול.\n"
            "עדכן את מבנה ה-DataFrame שאתה שם ב-session_state / SqlStore."
        )
        return

    # 1.1 בחירת חלון זמן מתוך היסטוריה
    with st.expander("📅 Time window for risk analysis", expanded=False):
        try:
            all_dates = df["date"].sort_values()
            min_d = all_dates.min()
            max_d = all_dates.max()
            date_range = st.slider(
                "בחר חלון זמן לניתוח",
                min_value=min_d.to_pydatetime(),
                max_value=max_d.to_pydatetime(),
                value=(min_d.to_pydatetime(), max_d.to_pydatetime()),
            )
            start_sel, end_sel = date_range
            mask = (df["date"] >= start_sel) & (df["date"] <= end_sel)
            df = df.loc[mask].copy()
        except Exception:
            pass

    # 1.2 Preview + Data Quality + PnL Distribution
    _render_equity_preview(df)
    _render_pnl_distribution(df)

    # 2) RiskLimits בסיסיים — ניתן להחליף בקונפיג עתידי
    base_limits = RiskLimits()

    # 3) מריצים Risk Assessment
    try:
        risk_dict = risk_assessment_to_dashboard_dict(
            hist=df,
            base_limits=base_limits,
            equity_col="Equity",
            pnl_col="PnL" if "PnL" in df.columns else None,
            date_col="date",
            bucket_col=None,          # תוכל להעביר כאן שם עמודה של bucket/strategy/desk אם יש
            realized_sharpe=None,
        )
    except Exception as e:
        st.error("כשל בהרצת risk_assessment_to_dashboard_dict על הדאטה.")
        st.caption(str(e))
        return

    # נשמור ל-session לשימוש בטאבים אחרים / agents + היסטוריית Risk Assessments
    st.session_state["risk_assessment_bundle"] = risk_dict
    try:
        hist_list = st.session_state.get("risk_assessment_history", [])
        if not isinstance(hist_list, list):
            hist_list = []
        hist_meta = {
            "ts_utc": pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z"),
            "overall_risk_score": risk_dict.get("overall_risk_score"),
            "risk_label": risk_dict.get("risk_label"),
            "scaling_factor": risk_dict.get("scaling_factor"),
        }
        hist_list.append(hist_meta)
        st.session_state["risk_assessment_history"] = hist_list[-50:]
    except Exception:
        pass

    # 4) Panels עיקריים
    _render_overall_risk_header(risk_dict)
    _render_goals_vs_realized(risk_dict)
    _render_risk_timeline(risk_dict)
    _render_buckets_and_scaling(risk_dict)
    _render_kill_switch_details(risk_dict)

    # 5) Macro-Risk Integration (אם קיים מידע)
    _render_macro_risk_panel()

    # 6) Smart Scan Risk Overlay (אם קיים מידע)
    _render_smart_scan_risk_panel()

    # 7) What-if Scaling Simulation
    _render_what_if_scaling(risk_dict)

    # 8) Summary מלא
    _render_risk_summary(risk_dict)

    # 9) Risk Assessment History — השוואה בין ריצות
    with st.expander("📆 Risk Assessment History (this session)", expanded=False):
        hist_list = st.session_state.get("risk_assessment_history", [])
        if not hist_list:
            st.caption("לא בוצעו עדיין הערכות סיכון לשמירה בהיסטוריה.")
        else:
            df_hist = pd.DataFrame(hist_list)
            st.dataframe(df_hist, width="stretch", height=260)

    st.caption(
        "Risk Tab (HF-grade): מחובר ישירות ל-core.risk_engine, עובד רק על דאטה אמיתי.\n"
        "אינטגרציית המאקרו וה-Smart Scan מאפשרת לחבר בין מצב השוק, המערכות והקרן לתמונת סיכון אחת."
    )
