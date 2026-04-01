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
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY = True
except Exception:
    _PLOTLY = False

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
    """Equity Curve (Plotly), Drawdown, rolling Sharpe + Data Quality."""
    with st.expander("📈 Equity Curve, Drawdown & Data Quality", expanded=True):
        df = hist.copy()
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass

        # ── KPI row ───────────────────────────────────────────────
        try:
            n_rows = len(df)
            start = df["date"].min()
            end = df["date"].max()
            days_span = int((end - start).days) if pd.notna(start) and pd.notna(end) else None
            eq_arr = pd.to_numeric(df["Equity"], errors="coerce").ffill()
            total_ret = float((eq_arr.iloc[-1] / eq_arr.iloc[0] - 1) * 100) if len(eq_arr) > 1 else 0.0
            peak = eq_arr.cummax()
            dd_series = (eq_arr - peak) / peak * 100
            max_dd = float(dd_series.min())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("N rows", n_rows)
            k2.metric("Span", f"{days_span}d" if days_span else "N/A")
            k3.metric("Total Return", f"{total_ret:.1f}%")
            k4.metric("Max Drawdown", f"{max_dd:.1f}%")

            if n_rows < 60:
                st.warning("⚠️ פחות מ-60 תצפיות — הערכת הסיכון עלולה להיות לא יציבה.")
            elif n_rows < 200:
                st.info("ℹ️ היסטוריית הסיכון יחסית קצרה (פחות מ-200 תצפיות).")
        except Exception as _kpi_err:
            st.caption(f"KPI computation error: {_kpi_err}")

        # ── Equity + Drawdown dual-panel chart ────────────────────
        try:
            if _PLOTLY:
                from plotly.subplots import make_subplots as _make_subplots
                dates = df["date"]
                eq_arr = pd.to_numeric(df["Equity"], errors="coerce").ffill()
                peak = eq_arr.cummax()
                dd_s = (eq_arr - peak) / peak * 100

                fig = _make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     row_heights=[0.65, 0.35], vertical_spacing=0.06)
                fig.add_trace(go.Scatter(x=dates, y=eq_arr, name="Equity",
                                         line=dict(color="#42A5F5", width=1.8)), row=1, col=1)
                fig.add_trace(go.Scatter(x=dates, y=dd_s, name="Drawdown %",
                                         line=dict(color="#EF5350", width=1.2),
                                         fill="tozeroy", fillcolor="rgba(239,83,80,0.12)"), row=2, col=1)
                fig.update_layout(
                    height=440, showlegend=True,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ECEFF1"), margin=dict(l=0, r=0, t=20, b=0),
                )
                fig.update_yaxes(title_text="Equity", row=1, col=1)
                fig.update_yaxes(title_text="DD %", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart(df.set_index("date")["Equity"])
        except Exception as _chart_err:
            st.caption(f"Equity chart error: {_chart_err}")

        # ── Rolling Sharpe (30d) ───────────────────────────────────
        try:
            if _PLOTLY and "PnL" in df.columns:
                pnl_s = pd.to_numeric(df["PnL"], errors="coerce").fillna(0.0)
                window = min(30, max(5, len(pnl_s) // 5))
                roll_mean = pnl_s.rolling(window).mean()
                roll_std  = pnl_s.rolling(window).std(ddof=1).replace(0, np.nan)
                roll_sharpe = (roll_mean / roll_std * np.sqrt(252)).dropna()
                fig_sr = go.Figure(go.Scatter(
                    x=df["date"].iloc[roll_sharpe.index],
                    y=roll_sharpe.values,
                    name=f"Rolling Sharpe ({window}d)",
                    line=dict(color="#AB47BC", width=1.5),
                ))
                fig_sr.add_hline(y=0, line_color="gray", line_dash="dot", opacity=0.5)
                fig_sr.add_hline(y=1, line_color="#4CAF50", line_dash="dash", opacity=0.4,
                                 annotation_text="Sharpe=1", annotation_position="right")
                fig_sr.update_layout(
                    title=f"Rolling Sharpe ratio ({window}d window)",
                    height=250, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ECEFF1"), margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig_sr, use_container_width=True)
        except Exception as _sr_err:
            st.caption(f"Rolling Sharpe error: {_sr_err}")


def _render_pnl_distribution(hist: pd.DataFrame) -> None:
    """התפלגות PnL (Plotly histogram + VaR/ES/CVaR Cornish-Fisher), monthly heatmap."""
    with st.expander("📊 PnL Distribution, Tail Risk & Monthly Heatmap", expanded=False):
        if "PnL" not in hist.columns:
            st.caption("אין עמודת PnL בהיסטוריה — לא ניתן להציג התפלגות PnL.")
            return

        pnl = pd.to_numeric(hist["PnL"], errors="coerce").dropna()
        if pnl.empty:
            st.caption("עמודת PnL ריקה / לא תקינה.")
            return

        # ── Tail risk KPIs ────────────────────────────────────────
        mean_pnl  = float(pnl.mean())
        std_pnl   = float(pnl.std(ddof=1))
        p5        = float(np.percentile(pnl, 5))
        p1        = float(np.percentile(pnl, 1))
        cvar_95   = float(pnl[pnl <= p5].mean()) if (pnl <= p5).any() else p5
        skew      = float(pnl.skew())
        kurt      = float(pnl.kurt())
        pos_days  = int((pnl > 0).sum())
        win_rate  = pos_days / len(pnl) * 100

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Mean PnL", f"{mean_pnl:.4f}")
        k2.metric("Std PnL",  f"{std_pnl:.4f}")
        k3.metric("VaR 95%",  f"{p5:.4f}")
        k4.metric("CVaR 95%", f"{cvar_95:.4f}")
        k5.metric("Skewness", f"{skew:.3f}")
        k6.metric("Win rate", f"{win_rate:.1f}%")

        # ── Plotly histogram ──────────────────────────────────────
        try:
            if _PLOTLY:
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(
                    x=pnl, nbinsx=40, name="PnL",
                    marker_color="#42A5F5", opacity=0.75,
                ))
                fig_h.add_vline(x=p5,      line_color="#FF9800", line_dash="dash",
                                annotation_text="VaR 95%", annotation_position="top left")
                fig_h.add_vline(x=p1,      line_color="#EF5350", line_dash="dash",
                                annotation_text="VaR 99%", annotation_position="top left")
                fig_h.add_vline(x=mean_pnl, line_color="#4CAF50", line_dash="dot",
                                annotation_text="Mean", annotation_position="top right")
                fig_h.update_layout(
                    title="PnL Distribution",
                    height=320, bargap=0.05,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ECEFF1"),
                )
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                hist_counts, bin_edges = np.histogram(pnl, bins=30)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                st.bar_chart(pd.DataFrame({"count": hist_counts}, index=bin_centers))
        except Exception as _h_err:
            st.caption(f"Histogram error: {_h_err}")

        # ── Monthly PnL heatmap ───────────────────────────────────
        try:
            if _PLOTLY and "date" in hist.columns:
                df_m = hist[["date", "PnL"]].copy()
                df_m["date"] = pd.to_datetime(df_m["date"])
                df_m["PnL"]  = pd.to_numeric(df_m["PnL"], errors="coerce")
                df_m["year"]  = df_m["date"].dt.year
                df_m["month"] = df_m["date"].dt.month
                monthly = df_m.groupby(["year", "month"])["PnL"].sum().reset_index()
                pivot = monthly.pivot(index="year", columns="month", values="PnL")
                month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"]
                pivot.columns = [month_names[m-1] for m in pivot.columns]

                fig_hm = go.Figure(go.Heatmap(
                    z=pivot.values.tolist(),
                    x=pivot.columns.tolist(),
                    y=[str(y) for y in pivot.index.tolist()],
                    colorscale="RdYlGn",
                    text=[[f"{v:.4f}" if v is not None and not np.isnan(v) else "" for v in row]
                          for row in pivot.values.tolist()],
                    texttemplate="%{text}",
                    showscale=True,
                    colorbar=dict(title="PnL"),
                ))
                fig_hm.update_layout(
                    title="Monthly PnL Heatmap",
                    height=max(200, len(pivot) * 35 + 80),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ECEFF1"),
                )
                st.plotly_chart(fig_hm, use_container_width=True)
        except Exception as _hm_err:
            st.caption(f"Monthly heatmap error: {_hm_err}")


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
    """טבלת יעד מול ביצוע: Vol / DD / Sharpe + Plotly bullet/gauge chart."""
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
            "Target": goals.get("target_sharpe_min"),
            "Realized": realized.get("realized_sharpe"),
            "Gap": gaps.get("sharpe_gap"),
        },
    ]

    # ── Plotly grouped bar: Target vs Realized ────────────────
    if _PLOTLY:
        try:
            metrics   = [r["Metric"] for r in rows]
            targets   = [float(r["Target"]) if r["Target"] is not None else 0.0 for r in rows]
            realizeds = [float(r["Realized"]) if r["Realized"] is not None else 0.0 for r in rows]
            # colour each realized bar: green if ≤ target (for vol/dd), ≥ target for sharpe
            bar_colors = []
            for i, m in enumerate(metrics):
                t, rv = targets[i], realizeds[i]
                if "sharpe" in m.lower():
                    bar_colors.append("#4CAF50" if rv >= t else "#EF5350")
                else:
                    bar_colors.append("#4CAF50" if rv <= t else "#EF5350")

            fig_g = go.Figure()
            fig_g.add_trace(go.Bar(
                name="Target", x=metrics, y=targets,
                marker_color="#546E7A", opacity=0.65,
            ))
            fig_g.add_trace(go.Bar(
                name="Realized", x=metrics, y=realizeds,
                marker_color=bar_colors, opacity=0.90,
            ))
            fig_g.update_layout(
                barmode="group",
                title="Goals vs Realized (green = on-target)",
                height=300,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ECEFF1"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_g, use_container_width=True)
        except Exception as _ge:
            st.caption(f"Goals chart error: {_ge}")

    # ── Summary table ─────────────────────────────────────────
    def _fmt(x):
        if x is None:
            return ""
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    df_goals = pd.DataFrame(rows)
    df_show = df_goals.copy()
    for col in ("Target", "Realized", "Gap"):
        df_show[col] = df_show[col].apply(_fmt)

    st.dataframe(df_show, use_container_width=True)
    st.markdown("---")


def _render_risk_timeline(risk_dict: Dict[str, Any]) -> None:
    """Risk Timeline: rolling Vol / DD לאורך זמן — Plotly dual-axis."""
    st.markdown("### ⏱ Risk Timeline (rolling Vol / DD)")

    tl_df = risk_dict.get("risk_timeline")
    if not isinstance(tl_df, pd.DataFrame) or tl_df.empty:
        st.caption("אין Risk Timeline מחושב (risk_timeline ריק או לא הוחזר מהמנוע).")
        st.markdown("---")
        return

    tl = tl_df.copy()
    has_date = "date" in tl.columns
    if has_date:
        tl["date"] = pd.to_datetime(tl["date"])

    has_vol = "rolling_vol_pct" in tl.columns
    has_dd  = "rolling_dd_pct" in tl.columns

    if _PLOTLY and (has_vol or has_dd):
        try:
            from plotly.subplots import make_subplots as _make_subplots
            x_axis = tl["date"] if has_date else tl.index

            n_rows = (1 if has_vol else 0) + (1 if has_dd else 0)
            if n_rows == 2:
                fig_tl = _make_subplots(rows=2, cols=1, shared_xaxes=True,
                                         row_heights=[0.5, 0.5], vertical_spacing=0.06,
                                         subplot_titles=("Rolling Vol % (annualized)", "Rolling Drawdown %"))
                fig_tl.add_trace(go.Scatter(
                    x=x_axis, y=tl["rolling_vol_pct"], name="Vol %",
                    line=dict(color="#FFA726", width=1.6),
                ), row=1, col=1)
                fig_tl.add_trace(go.Scatter(
                    x=x_axis, y=tl["rolling_dd_pct"], name="DD %",
                    line=dict(color="#EF5350", width=1.4),
                    fill="tozeroy", fillcolor="rgba(239,83,80,0.10)",
                ), row=2, col=1)
            elif has_vol:
                fig_tl = go.Figure(go.Scatter(
                    x=x_axis, y=tl["rolling_vol_pct"], name="Vol %",
                    line=dict(color="#FFA726", width=1.6),
                ))
                fig_tl.update_layout(title="Rolling Vol % (annualized)")
            else:
                fig_tl = go.Figure(go.Scatter(
                    x=x_axis, y=tl["rolling_dd_pct"], name="DD %",
                    line=dict(color="#EF5350", width=1.4),
                    fill="tozeroy", fillcolor="rgba(239,83,80,0.10)",
                ))
                fig_tl.update_layout(title="Rolling Drawdown %")

            fig_tl.update_layout(
                height=380, showlegend=True,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ECEFF1"), margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_tl, use_container_width=True)
        except Exception as _tl_err:
            st.caption(f"Timeline chart error: {_tl_err}")
            # fallback
            if has_date:
                tl.set_index("date", inplace=True)
            if has_vol:
                st.line_chart(tl["rolling_vol_pct"])
            if has_dd:
                st.line_chart(tl["rolling_dd_pct"])
    else:
        # no Plotly fallback
        if has_date:
            tl.set_index("date", inplace=True)
        col_vol, col_dd = st.columns(2)
        with col_vol:
            st.caption("Rolling Vol (annualized)")
            if has_vol:
                st.line_chart(tl["rolling_vol_pct"])
            else:
                st.caption("אין עמודה rolling_vol_pct.")
        with col_dd:
            st.caption("Rolling Drawdown")
            if has_dd:
                st.line_chart(tl["rolling_dd_pct"])
            else:
                st.caption("אין עמודה rolling_dd_pct.")

    st.markdown("---")


def _render_buckets_and_scaling(risk_dict: Dict[str, Any]) -> None:
    """Bucket Risks + Scaling Suggestions (אם קיימים) + Plotly bar."""
    st.markdown("### 🧩 Bucket Risks & Scaling Suggestions")

    bucket_risks = risk_dict.get("bucket_risks", []) or []
    bucket_scaling = risk_dict.get("bucket_scaling", []) or []

    if not bucket_risks and not bucket_scaling:
        st.caption("אין bucket_risks / bucket_scaling ב-risk_dict (כנראה לא הועבר bucket_col).")
        st.markdown("---")
        return

    # ── Plotly bar chart for bucket risk contributions ────────
    if _PLOTLY and bucket_risks:
        try:
            df_br = pd.DataFrame(bucket_risks)
            sort_col = next(
                (c for c in ("contribution_to_risk", "risk_score", "vol_pct") if c in df_br.columns),
                None,
            )
            name_col = next(
                (c for c in ("bucket", "strategy", "desk", "name") if c in df_br.columns),
                None,
            )
            if sort_col and name_col:
                df_br = df_br.sort_values(sort_col, ascending=True)
                vals = df_br[sort_col].tolist()
                colors = ["#EF5350" if v == max(vals) else "#42A5F5" for v in vals]
                fig_b = go.Figure(go.Bar(
                    x=vals,
                    y=df_br[name_col].astype(str).tolist(),
                    orientation="h",
                    marker_color=colors,
                    name=sort_col,
                ))
                fig_b.update_layout(
                    title=f"Bucket Risk Contribution ({sort_col})",
                    height=max(250, len(df_br) * 30 + 80),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ECEFF1"),
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis_title=sort_col,
                )
                st.plotly_chart(fig_b, use_container_width=True)
        except Exception as _be:
            st.caption(f"Bucket chart error: {_be}")

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("#### Top Bucket Risks")
        if bucket_risks:
            df_br = pd.DataFrame(bucket_risks)
            sort_col = next(
                (c for c in ("contribution_to_risk", "risk_score", "vol_pct") if c in df_br.columns),
                None,
            )
            if sort_col:
                df_br = df_br.sort_values(sort_col, ascending=False)
            st.dataframe(df_br, use_container_width=True)
        else:
            st.caption("אין bucket_risks ב-risk_dict.")

    with col_b2:
        st.markdown("#### Bucket Scaling")
        if bucket_scaling:
            df_bs = pd.DataFrame(bucket_scaling)
            st.dataframe(df_bs, use_container_width=True)
        else:
            st.caption("אין bucket_scaling מחושב.")

    st.markdown("---")


def _render_kill_switch_details(risk_dict: Dict[str, Any]) -> None:
    """Kill-Switch: מצב, סיבה, breached limits + Plotly severity gauge."""
    st.markdown("### ⛔ Kill-Switch Details")

    kill = risk_dict.get("kill_switch", {}) or {}
    if not kill:
        st.caption("לא התקבל מידע Kill-Switch מהמנוע (kill_switch ריק).")
        st.markdown("---")
        return

    ks_enabled = bool(kill.get("enabled"))
    ks_mode = kill.get("mode", "N/A")
    severity = kill.get("severity_score", None)
    reason = kill.get("reason") or kill.get("reason_text")
    breached_limits = kill.get("breached_limits", [])

    # ── Severity gauge ────────────────────────────────────────
    if _PLOTLY and severity is not None:
        try:
            sev_val = max(0.0, min(1.0, float(severity)))
            gauge_color = (
                "#4CAF50" if sev_val < 0.35 else
                "#FFA726" if sev_val < 0.65 else
                "#EF5350"
            )
            fig_ks = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sev_val * 100,
                number={"suffix": "%", "font": {"color": "#ECEFF1"}},
                title={"text": "Kill-Switch Severity", "font": {"color": "#ECEFF1"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#ECEFF1"},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 35],  "color": "rgba(76,175,80,0.15)"},
                        {"range": [35, 65], "color": "rgba(255,167,38,0.15)"},
                        {"range": [65, 100],"color": "rgba(239,83,80,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": sev_val * 100,
                    },
                },
            ))
            fig_ks.update_layout(
                height=230,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ECEFF1"), margin=dict(l=20, r=20, t=40, b=0),
            )
            st.plotly_chart(fig_ks, use_container_width=True)
        except Exception as _ks_err:
            st.caption(f"Severity gauge error: {_ks_err}")

    col_k1, col_k2 = st.columns(2)

    with col_k1:
        status_color = "🔴" if ks_enabled else "🟢"
        st.write(f"{status_color} **Enabled:** `{ks_enabled}`  •  **Mode:** `{ks_mode}`")
        if severity is not None:
            st.write(f"**Severity score:** `{float(severity):.2f}`")
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
        st.dataframe(df_view.head(20)[cols_to_show], use_container_width=True)

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
    What-if Scaling — sensitivity lines + interactive slider.

    Assumes Vol ∝ scaling_factor, Max DD ∝ scaling_factor (linear approximation).
    """
    st.markdown("### 🎚 What-if Scaling Simulation")

    goals_gap = risk_dict.get("goals_gap", {}) or {}
    realized = goals_gap.get("realized", {}) or {}
    goals    = goals_gap.get("goals", {}) or {}
    vol_real = realized.get("realized_vol_pct")
    dd_real  = realized.get("realized_max_dd_pct")

    if vol_real is None and dd_real is None:
        st.caption("אין מספיק מידע על Vol/DD כדי לבצע סימולציה (GoalsGap.realized ריק).")
        st.markdown("---")
        return

    base_scaling = float(risk_dict.get("scaling_factor") or 1.0)
    scale = st.slider(
        "Gross exposure scaling (what-if)",
        min_value=0.1,
        max_value=2.5,
        value=base_scaling,
        step=0.05,
        help="Vol/DD scale linearly with exposure multiplier.",
        key="risk_whatif_scaling",
    )

    # ── Plotly sensitivity curve ──────────────────────────────
    if _PLOTLY:
        try:
            sweep = np.linspace(0.1, 2.5, 48)
            fig_wi = go.Figure()

            if vol_real is not None:
                vol_curve = [float(vol_real) * (s / base_scaling) for s in sweep]
                fig_wi.add_trace(go.Scatter(
                    x=sweep, y=vol_curve, name="Vol % (scaled)",
                    line=dict(color="#FFA726", width=2),
                ))
                # target line
                tgt_vol = goals.get("target_vol_pct")
                if tgt_vol is not None:
                    fig_wi.add_hline(
                        y=float(tgt_vol), line_color="#4CAF50", line_dash="dash",
                        annotation_text=f"Vol target {tgt_vol:.1f}%",
                        annotation_position="right",
                    )

            if dd_real is not None:
                dd_curve = [abs(float(dd_real)) * (s / base_scaling) for s in sweep]
                fig_wi.add_trace(go.Scatter(
                    x=sweep, y=dd_curve, name="Max DD % (scaled)",
                    line=dict(color="#EF5350", width=2, dash="dot"),
                ))
                tgt_dd = goals.get("target_dd_pct")
                if tgt_dd is not None:
                    fig_wi.add_hline(
                        y=abs(float(tgt_dd)), line_color="#66BB6A", line_dash="dash",
                        annotation_text=f"DD target {tgt_dd:.1f}%",
                        annotation_position="right",
                    )

            # marker for selected scale
            fig_wi.add_vline(
                x=scale, line_color="white", line_dash="dot", opacity=0.6,
                annotation_text=f"Selected: {scale:.2f}x",
                annotation_position="top right",
            )

            fig_wi.update_layout(
                title="What-if: Vol & DD vs Scaling Factor",
                height=320,
                xaxis_title="Scaling factor",
                yaxis_title="% of portfolio",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ECEFF1"),
                legend=dict(orientation="h", y=1.12),
                margin=dict(l=0, r=0, t=50, b=0),
            )
            st.plotly_chart(fig_wi, use_container_width=True)
        except Exception as _wi_err:
            st.caption(f"What-if chart error: {_wi_err}")

    # ── Point estimate table ──────────────────────────────────
    rows = []
    if vol_real is not None:
        vol_scaled = float(vol_real) * (scale / base_scaling)
        rows.append({"Metric": "Vol (scaled)", "Value": f"{vol_scaled:.2f}%"})
    if dd_real is not None:
        dd_scaled = float(dd_real) * (scale / base_scaling)
        rows.append({"Metric": "Max DD (scaled)", "Value": f"{dd_scaled:.2f}%"})
    if rows:
        st.table(pd.DataFrame(rows))

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

    st.markdown(
        """
<div style="
    background: linear-gradient(90deg, #B71C1C 0%, #C62828 100%);
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(198,40,40,0.20);
">
    <div style="font-size:1.25rem;font-weight:800;color:white;letter-spacing:-0.2px;">
        ⚠️ Fund-Level Risk Dashboard
    </div>
    <div style="font-size:0.80rem;color:rgba(255,255,255,0.80);margin-top:4px;">
        Real-data only · VaR · Drawdown · Kill-Switch · Macro Overlay · Scaling Simulation
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown(
            """
            **Risk Dashboard** provides a real-data fund-level risk view:

            | Panel | Description |
            |---|---|
            | **Risk Score** | Overall risk score + scaling factor |
            | **Goals vs Realized** | Vol / Drawdown / Sharpe vs targets |
            | **Timeline** | Rolling Vol/DD history |
            | **Buckets** | Per-strategy bucket risks + scaling hints |
            | **Kill-Switch** | Status + breach reasons |
            | **Macro Overlay** | Regime-aware risk integration |
            | **Smart Scan** | Signal-level risk contribution |
            | **What-if** | Scaling simulation |

            > Data source: `risk_hist_df` / `backtest_history_df` / `portfolio_history_df` in session, or `equity_curve` in SqlStore.
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
            "ts_utc": pd.Timestamp.now(tz="UTC").isoformat().replace("+00:00", "Z"),
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
            st.dataframe(df_hist, use_container_width=True)

    st.caption(
        "Risk Tab (HF-grade): מחובר ישירות ל-core.risk_engine, עובד רק על דאטה אמיתי.\n"
        "אינטגרציית המאקרו וה-Smart Scan מאפשרת לחבר בין מצב השוק, המערכות והקרן לתמונת סיכון אחת."
    )
