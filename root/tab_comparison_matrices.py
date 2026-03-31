# -*- coding: utf-8 -*-
"""
root/tab_comparison_matrices.py — HF-grade Comparison Matrices Tab (v3, Real Data)
===================================================================================

טאב Streamlit ברמת קרן גידור שעוטף את מודול הליבה:
    core.tab_comparison_matrices

המהות:
-------
• אין כאן שום דמו. הטאב עובד *רק* על דאטה אמיתי שיגיע מ:
    1. SqlStore (טבלת tab_profiles).
    2. Runtime/Context (build_profiles_from_context).
    3. Session DataFrame.
    4. Upload CSV.

• אם אין דאטה → הטאב מציג הודעה מקצועית "חסר קונפיג" ולא ממציא נתונים.

• חבילת השוואה מלאה (Comparison Bundle):
    - similarity:            Tab vs Tab similarity matrix.
    - distance:              Tab vs Tab distance matrix.
    - metric_vs_tab:         Metric vs Tab normalized matrix.
    - tab_type_summary:      Summary לפי tab_type.
    - tab_type_similarity:   דמיון בין tab_type-ים.
    - ranks:                 Rank per metric.
    - metric_corr:           קורלציה בין מטריקות.
    - meta:                  JSON-safe metadata.

• UI ברמת קרן + עוד ~20 רעיונות:
    1. בחירת מקור פרופילים (SqlStore / Runtime / Session / Upload).
    2. פילטר לפי tab_type.
    3. פילטר לפי tags.
    4. כרטיסי Overall Summary (מספר טאבים, מספר מטריקות, טווחי Sharpe/DD).
    5. בחירת מטריקות להשוואה.
    6. בחירת normalisation / similarity / distance.
    7. Scenario Preset (default / risk-focused / latency-focused).
    8. מטריצות similarity/distance/metric-vs-tab.
    9. Summary לפי tab_type.
    10. Anchor Tab Alignment + בחירת anchor לפי type.
    11. Clustering (Hierarchical) לאשכולי טאבים.
    12. Anomaly Detection לטאבים חריגים.
    13. Diagnostics למטריקות (mean/std/min/max/coverage).
    14. Highlight למטריקות חופפות (Redundant KPIs) לפי metric_corr.
    15. Tag-level Summary (ממוצע KPIs לפי tag).
    16. Tab Health: כמה missing metrics לכל טאב.
    17. שמירת bundle + profiles ל-session_state.
    18. Export ל-CSV (similarity/distance/metric_vs_tab).
    19. Export JSON ל-Bundle מלא (לסוכנים/Agents).
    20. Hooks לשימוש עתידי בטאבים אחרים (למשל Agents / DashboardService).
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List, Sequence

import streamlit as st
import pandas as pd
import numpy as np

from core.app_context import AppContext
from core.tab_comparison_matrices import (
    TabProfile,
    TabComparisonConfig,
    build_comparison_bundle,
    build_profiles_from_context,
    hierarchical_cluster_tabs,
    compute_alignment_scores,
    detect_tab_anomalies,
)

# ============================================================
# 1) Data loaders — SqlStore / Runtime / Session / CSV
# ============================================================


def _get_sql_store(app_ctx: AppContext):
    """מוציא SqlStore מתוך AppContext אם קיים."""
    store = getattr(app_ctx, "sql_store", None)
    if store is not None:
        return store

    services = getattr(app_ctx, "services", None)
    if isinstance(services, dict):
        return services.get("sql_store")

    return None


def _load_profiles_df_from_sql_store(app_ctx: AppContext) -> Optional[pd.DataFrame]:
    """
    מנסה לקרוא טבלה tab_profiles מה-SqlStore.

    אם הטבלה לא קיימת או ריקה → מחזיר None.
    """
    store = _get_sql_store(app_ctx)
    if store is None:
        return None

    try:
        df = store.read_table("tab_profiles")
    except Exception as e:
        st.warning(f"קריאה מ-SqlStore.read_table('tab_profiles') נכשלה: {e}")
        return None

    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    return df


def _load_profiles_df_from_session() -> Optional[pd.DataFrame]:
    """
    מנסה לקרוא DataFrame של TabProfiles מתוך session_state.

    מפתחות אפשריים:
        - tab_profiles_df
        - comparison_tab_profiles_df
    """
    try:
        sess = st.session_state
    except Exception:
        return None

    for key in ("tab_profiles_df", "comparison_tab_profiles_df"):
        df = sess.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df

    return None


def _load_profiles_df_from_upload(upload_file) -> Optional[pd.DataFrame]:
    """טוען DataFrame של TabProfiles מ-CSV שהמשתמש העלה."""
    if upload_file is None:
        return None

    try:
        df = pd.read_csv(upload_file)
    except Exception as e:
        st.error(f"בעיה בקריאת קובץ ה-CSV: {e}")
        return None

    return df if not df.empty else None


def _load_profiles_from_runtime_context(app_ctx: AppContext) -> Optional[List[TabProfile]]:
    """
    מנסה לבנות TabProfile-ים מתוך Runtime/Context אמיתי.

    לוגיקה:
    -------
    1. אם ב-AppContext יש attr 'tab_profiles' (List[TabProfile]) → נשתמש בו.
    2. אם יש ctx dict (app_ctx.ctx_dict או st.session_state["ctx"]) → ננסה build_profiles_from_context.
       הספק (spec) צריך להיות מוגדר במערכת שלך (ניתן לשנות כאן).
    """
    profiles = getattr(app_ctx, "tab_profiles", None)
    if isinstance(profiles, list) and profiles and isinstance(profiles[0], TabProfile):
        return profiles

    ctx_dict = getattr(app_ctx, "ctx_dict", None)
    if ctx_dict is None:
        try:
            ctx_dict = st.session_state.get("ctx", None)
        except Exception:
            ctx_dict = None

    if not isinstance(ctx_dict, dict):
        return None

    # spec בסיסי לדוגמה — אתה יכול להחליף אותו לספק אמיתי שלך
    spec = {
        "home": {
            "tab_type": "overview",
            "label": "Dashboard Home",
            "path": ["tabs", "home", "metrics"],
        },
        "backtest": {
            "tab_type": "stats",
            "label": "Backtest KPIs",
            "path": ["tabs", "backtest", "metrics"],
        },
        "macro": {
            "tab_type": "macro",
            "label": "Macro Engine",
            "path": ["tabs", "macro", "scores"],
        },
    }
    try:
        profiles_from_ctx = build_profiles_from_context(ctx_dict, spec)
        if profiles_from_ctx:
            return profiles_from_ctx
    except Exception as e:
        st.warning(f"build_profiles_from_context נכשל: {e}")
        return None

    return None


def _df_to_profiles(df: pd.DataFrame) -> List[TabProfile]:
    """
    ממיר DataFrame -> List[TabProfile].

    דרישות:
        columns חובה: tab_id, tab_type, label
        אופציונלי:
            weight, tags, וכל שאר העמודות → metrics (מספריות).
    """
    required = {"tab_id", "tab_type", "label"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"חסרות עמודות חובה בטבלת הפרופילים: {missing}")

    profiles: List[TabProfile] = []
    for _, row in df.iterrows():
        tab_id = str(row["tab_id"])
        tab_type = str(row["tab_type"])
        label = str(row["label"])
        weight = float(row["weight"]) if "weight" in df.columns and pd.notna(row["weight"]) else 1.0

        tags: List[str] = []
        if "tags" in df.columns and pd.notna(row["tags"]):
            tags = [t.strip() for t in str(row["tags"]).split(",") if t.strip()]

        metrics: Dict[str, float] = {}
        for col in df.columns:
            if col in {"tab_id", "tab_type", "label", "weight", "tags"}:
                continue
            val = row.get(col)
            try:
                fval = float(val)
            except Exception:
                continue
            if np.isfinite(fval):
                metrics[col] = fval

        profiles.append(
            TabProfile(
                tab_id=tab_id,
                tab_type=tab_type,
                label=label,
                metrics=metrics,
                weight=weight,
                tags=tags,
                metadata={},
            )
        )

    return profiles


def _load_tab_profiles(
    app_ctx: AppContext,
    source: str,
    upload_file,
) -> List[TabProfile]:
    """
    מקור יחיד לטעינת TabProfile-ים:

    source:
        "sql_store"   → tab_profiles מ-SqlStore.
        "runtime"     → build_profiles_from_context / app_ctx.tab_profiles.
        "session"     → DataFrame ב-session_state.
        "upload"      → CSV מהמשתמש.

    אין fallback לדמו. אם אין דאטה → מחזירים רשימה ריקה וה-UI מסביר.
    """
    source = source.lower()
    df: Optional[pd.DataFrame] = None

    if source == "sql_store":
        df = _load_profiles_df_from_sql_store(app_ctx)
        if df is None:
            st.error(
                "לא נמצאה טבלה tab_profiles ב-SqlStore או שהיא ריקה.\n"
                "צור טבלה כזו וטעין לשם פרופילים אמיתיים, או השתמש במקור Runtime/Session/Upload."
            )
            return []

        return _df_to_profiles(df)

    if source == "runtime":
        profiles = _load_profiles_from_runtime_context(app_ctx)
        if profiles:
            return profiles
        st.error(
            "לא הצלחתי לבנות TabProfiles מ-Runtime/Context.\n"
            "ודא שיש app_ctx.tab_profiles או ctx_dict מתאים, או השתמש ב-SqlStore/Session/Upload."
        )
        return []

    if source == "session":
        df = _load_profiles_df_from_session()
        if df is None:
            st.error(
                "לא נמצא DataFrame של טאב-פרופילים ב-session_state.\n"
                "שים DataFrame תחת 'tab_profiles_df' או 'comparison_tab_profiles_df'."
            )
            return []
        return _df_to_profiles(df)

    if source == "upload":
        df = _load_profiles_df_from_upload(upload_file)
        if df is None:
            st.error("קובץ ה-CSV ריק או לא תקין.")
            return []
        return _df_to_profiles(df)

    st.error(f"מקור פרופילים לא מוכר: {source}")
    return []


# ============================================================
# 2) Diagnostics helpers — Metric / Tag / Tab health
# ============================================================


def _compute_metric_diagnostics(metric_vs_tab_df: pd.DataFrame) -> pd.DataFrame:
    """
    מחזיר DataFrame עם דיאגנוסטיקה לכל מטריקה:
        metric, mean, std, min, max, coverage_pct, iqr
    """
    if metric_vs_tab_df is None or metric_vs_tab_df.empty:
        return pd.DataFrame(columns=["metric", "mean", "std", "min", "max", "coverage_pct", "iqr"])

    rows = []
    for metric in metric_vs_tab_df.index:
        s = pd.to_numeric(metric_vs_tab_df.loc[metric], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        coverage = s.notna().mean() * 100.0
        s_clean = s.dropna()
        if s_clean.empty:
            rows.append(
                {
                    "metric": metric,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "coverage_pct": coverage,
                    "iqr": np.nan,
                }
            )
            continue

        q25 = s_clean.quantile(0.25)
        q75 = s_clean.quantile(0.75)
        rows.append(
            {
                "metric": metric,
                "mean": float(s_clean.mean()),
                "std": float(s_clean.std(ddof=1)) if len(s_clean) > 1 else 0.0,
                "min": float(s_clean.min()),
                "max": float(s_clean.max()),
                "coverage_pct": coverage,
                "iqr": float(q75 - q25),
            }
        )

    df_diag = pd.DataFrame(rows).sort_values("coverage_pct", ascending=False)
    return df_diag


def _compute_tag_summary(profiles: List[TabProfile]) -> pd.DataFrame:
    """
    מחזיר סיכום לפי tags:
        tag, n_tabs, <mean metrics...>
    """
    rows = []
    for p in profiles:
        tags = p.tags or []
        if not tags:
            tags = ["(no_tag)"]
        for t in tags:
            row = {"tag": t, "tab_id": p.tab_id}
            row.update(p.metrics or {})
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["tag", "n_tabs"])

    df = pd.DataFrame(rows)
    # n_tabs per tag + mean metrics
    grouped = df.groupby("tag")
    agg = grouped.agg({col: "mean" for col in df.columns if col not in {"tag", "tab_id"}})
    agg["n_tabs"] = grouped["tab_id"].nunique()
    cols = ["n_tabs"] + [c for c in agg.columns if c != "n_tabs"]
    return agg[cols].sort_values("n_tabs", ascending=False)


def _compute_tab_health(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    בריאות טאב: כמה metrics חסרים לכל טאב.
    """
    if profiles_df is None or profiles_df.empty:
        return pd.DataFrame(columns=["tab_id", "missing_metrics", "missing_ratio"])

    non_metric_cols = {"tab_id", "tab_type", "label", "weight", "tags"}
    metric_cols = [c for c in profiles_df.columns if c not in non_metric_cols]

    rows = []
    for _, row in profiles_df.iterrows():
        total = len(metric_cols)
        missing = int(sum(pd.isna(row[c]) for c in metric_cols))
        ratio = missing / total if total > 0 else 0.0
        rows.append(
            {
                "tab_id": row["tab_id"],
                "tab_type": row["tab_type"],
                "missing_metrics": missing,
                "missing_ratio": ratio,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("missing_ratio", ascending=False)


# ============================================================
# 3) Tab Renderer — הטאב עצמו
# ============================================================


def render_tab(
    app_ctx: AppContext,
    feature_flags: Dict[str, Any],
    nav_payload: Optional[Dict[str, Any]] = None,
) -> None:
    """
    נקודת כניסה לטאב 'comparison_matrices' בדשבורד.

    אין כאן שום דמו:
    - או שיש TabProfiles אמיתיים → ניתוח מלא.
    - או שאין → הסבר מה צריך להגדיר ועוצרים.
    """

    st.markdown(
        """
<div style="
    background:linear-gradient(90deg,#4A148C 0%,#6A1B9A 100%);
    border-radius:10px;padding:14px 20px;margin-bottom:14px;
    box-shadow:0 2px 8px rgba(74,20,140,0.22);
">
    <div style="font-size:1.15rem;font-weight:800;color:white;letter-spacing:-0.2px;">
        🔬 Comparison Matrices — Tab Intelligence
    </div>
    <div style="font-size:0.76rem;color:rgba(255,255,255,0.78);margin-top:3px;">
        Similarity · Distance · Correlation between tabs · KPI-based · Real data only
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(
        "Analyzes similarity/distance/correlation between system tabs based on real KPIs "
        "(Sharpe / DD / Risk / Macro / Latency). No synthetic data — shows what is missing if tab profiles are unavailable."
    )

    # ------------------ 1. מקור פרופילים ------------------
    st.markdown("### 1️⃣ מקור הטאב-פרופילים")

    col_src1, col_src2 = st.columns([2, 2])

    with col_src1:
        source = st.radio(
            "בחר מקור פרופילים",
            options=["SqlStore", "Runtime/Context", "Session DF", "Upload CSV"],
            index=0,
            horizontal=True,
            key="cmp_profiles_source",
        )

    upload_file = None
    with col_src2:
        if source == "Upload CSV":
            upload_file = st.file_uploader(
                "Tab Profiles CSV (עמודות חובה: tab_id, tab_type, label)",
                type=["csv"],
                key="cmp_profiles_upload",
            )

    profiles = _load_tab_profiles(app_ctx, source=source.lower(), upload_file=upload_file)
    if not profiles:
        return  # כבר הופיעה הודעה מתאימה

    # טבלת פרופילים גולמית
    profiles_df = pd.DataFrame(
        [
            {
                "tab_id": p.tab_id,
                "tab_type": p.tab_type,
                "label": p.label,
                "weight": p.weight,
                "tags": ", ".join(p.tags or []),
                **(p.metrics or {}),
            }
            for p in profiles
        ]
    )

    # נשמור את הפקטור ל-session לטאבים אחרים/agents
    st.session_state["comparison_profiles_df"] = profiles_df

    # ------------------ 2. פילטרים (tab_type / tags) + Summary ------------------
    st.markdown("### 2️⃣ פילטרים + Summary ברמת מערכת")

    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])

    with col_f1:
        tab_types = sorted(profiles_df["tab_type"].dropna().unique().tolist())
        selected_types = st.multiselect(
            "Filter by tab_type",
            options=tab_types,
            default=tab_types,
            key="cmp_filter_tab_types",
        )

    with col_f2:
        all_tags = sorted(
            {
                t.strip()
                for tags in profiles_df["tags"].fillna("").tolist()
                for t in str(tags).split(",")
                if t.strip()
            }
        )
        selected_tags = st.multiselect(
            "Filter by tags",
            options=all_tags,
            default=all_tags,
            key="cmp_filter_tags",
        )

    with col_f3:
        st.caption("🔎 Overall summary")
        n_tabs = len(profiles_df)
        metric_cols = [c for c in profiles_df.columns if c not in {"tab_id", "tab_type", "label", "weight", "tags"}]
        n_metrics = len(metric_cols)
        st.write(f"Tabs: **{n_tabs}**, Metrics: **{n_metrics}**")

        if "sharpe" in metric_cols:
            sharpe_vals = pd.to_numeric(profiles_df["sharpe"], errors="coerce").dropna()
            if not sharpe_vals.empty:
                st.write(
                    f"Sharpe: min={sharpe_vals.min():.2f}, "
                    f"median={sharpe_vals.median():.2f}, max={sharpe_vals.max():.2f}"
                )
        if "max_dd" in metric_cols:
            dd_vals = pd.to_numeric(profiles_df["max_dd"], errors="coerce").dropna()
            if not dd_vals.empty:
                st.write(
                    f"Max DD: min={dd_vals.min():.2%}, max={dd_vals.max():.2%}"
                )

    # פילטר לפי tab_type / tags
    filtered_profiles = []
    for p in profiles:
        if selected_types and p.tab_type not in selected_types:
            continue
        if selected_tags:
            tags = set(p.tags or [])
            if tags and not (tags & set(selected_tags)):
                continue
        filtered_profiles.append(p)

    if not filtered_profiles:
        st.error("אחרי הפילטרים לא נשאר אף טאב. רכך את הסינון.")
        return

    profiles = filtered_profiles  # נמשיך רק עם המסוננים
    profiles_df = pd.DataFrame(
        [
            {
                "tab_id": p.tab_id,
                "tab_type": p.tab_type,
                "label": p.label,
                "weight": p.weight,
                "tags": ", ".join(p.tags or []),
                **(p.metrics or {}),
            }
            for p in profiles
        ]
    )

    with st.expander("📋 טאב-פרופילים לאחר סינון", expanded=False):
        st.dataframe(profiles_df, width="stretch")

    # ------------------ 3. קונפיגורציית השוואה ------------------
    st.markdown("### 3️⃣ קונפיגורציית השוואה (Config + Presets)")

    all_metric_keys = sorted({k for p in profiles for k in (p.metrics or {}).keys()})
    if not all_metric_keys:
        st.error("לא נמצאו מטריקות מספריות בפרופילים (אחרי סינון).")
        return

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        norm_method = st.selectbox(
            "Normalization",
            ["zscore", "minmax", "robust", "none"],
            index=0,
            key="cmp_norm",
        )
    with col_cfg2:
        sim_method = st.selectbox(
            "Similarity method",
            ["cosine", "corr", "euclidean"],
            index=0,
            key="cmp_sim_method",
        )
    with col_cfg3:
        dist_metric = st.selectbox(
            "Distance metric",
            ["euclidean"],
            index=0,
            key="cmp_dist_metric",
        )

    col_cfg4, col_cfg5 = st.columns([2, 4])

    with col_cfg4:
        preset = st.selectbox(
            "Preset (Layer focus)",
            ["default", "risk_focused", "latency_focused"],
            index=0,
            key="cmp_preset",
            help=(
                "default – ללא הטיות מיוחדות.\n"
                "risk_focused – מדגיש Sharpe/DD/Risk.\n"
                "latency_focused – מדגיש latency_ms / live_readiness."
            ),
        )

    with col_cfg5:
        metric_keys_selected = st.multiselect(
            "מטריקות להשוואה",
            options=all_metric_keys,
            default=all_metric_keys,
            key="cmp_metric_keys",
        )
        if not metric_keys_selected:
            metric_keys_selected = all_metric_keys

    # metric_weights לפי preset
    metric_weights: Optional[Dict[str, float]] = None
    if preset != "default":
        metric_weights = {k: 1.0 for k in metric_keys_selected}
        if preset == "risk_focused":
            for k in metric_keys_selected:
                if any(s in k.lower() for s in ("sharpe", "dd", "risk")):
                    metric_weights[k] = 2.0
        elif preset == "latency_focused":
            for k in metric_keys_selected:
                if any(s in k.lower() for s in ("latency", "live_ready", "live_readiness")):
                    metric_weights[k] = 2.5

    cfg = TabComparisonConfig(
        normalization=norm_method,
        similarity_method=sim_method,
        distance_metric=dist_metric,
        metric_weights=metric_weights,
        metric_meta=None,
        group_weights=None,
    )

    # ------------------ 4. הרצת bundle ------------------
    st.markdown("### 4️⃣ חישוב Comparison Bundle")

    run = st.button("🚀 הרץ Comparison Bundle על הדאטה", key="cmp_run_btn")
    if not run:
        st.info("בחר קונפיג ולחץ על הכפתור כדי לחשב מטריצות.")
        return

    bundle = build_comparison_bundle(
        profiles=profiles,
        cfg=cfg,
        metric_keys=metric_keys_selected,
    )

    similarity_df: pd.DataFrame = bundle["similarity"]
    distance_df: pd.DataFrame = bundle["distance"]
    metric_vs_tab_df: pd.DataFrame = bundle["metric_vs_tab"]
    tab_type_summary_df: pd.DataFrame = bundle["tab_type_summary"]
    tab_type_similarity_df: pd.DataFrame = bundle["tab_type_similarity"]
    ranks_df: pd.DataFrame = bundle["ranks"]
    metric_corr_df: pd.DataFrame = bundle["metric_corr"]
    meta_bundle: Dict[str, Any] = bundle["meta"]

    # נשמור הכל ל-session
    st.session_state["comparison_bundle"] = bundle

    # ------------------ 5. מטריצות ליבה ------------------
    st.markdown("### 5️⃣ מטריצות ליבה")

    tab_m1, tab_m2, tab_m3 = st.tabs(
        ["Similarity", "Distance", "Metric vs Tab"]
    )

    with tab_m1:
        st.markdown("#### 🔗 Tab vs Tab — Similarity")
        if not similarity_df.empty:
            st.dataframe(similarity_df, width="stretch")
        else:
            st.info("מטריצת similarity ריקה (אין מספיק מטריקות משותפות).")

    with tab_m2:
        st.markdown("#### 📏 Tab vs Tab — Distance")
        if not distance_df.empty:
            st.dataframe(distance_df, width="stretch")
        else:
            st.info("מטריצת distance ריקה.")

    with tab_m3:
        st.markdown("#### 📊 Metric vs Tab (normalized)")
        if not metric_vs_tab_df.empty:
            st.dataframe(metric_vs_tab_df, width="stretch")
        else:
            st.info("אין Metric vs Tab matrix (אין מטריקות זמינות).")

    # ------------------ 6. Summary לפי tab_type + Tag Summary ------------------
    st.markdown("### 6️⃣ Summary לפי tab_type + Tags")

    col_types1, col_types2 = st.columns(2)

    with col_types1:
        st.markdown("#### Summary by tab_type")
        if not tab_type_summary_df.empty:
            st.dataframe(tab_type_summary_df, width="stretch", height=260)
        else:
            st.caption("אין Summary ברמת tab_type.")

    with col_types2:
        st.markdown("#### tab_type vs tab_type similarity")
        if not tab_type_similarity_df.empty:
            st.dataframe(tab_type_similarity_df, width="stretch", height=260)
        else:
            st.caption("אין מטריצת דמיון ברמת tab_type.")

    st.markdown("#### 🏷 Tag-level Summary")
    tag_summary_df = _compute_tag_summary(profiles)
    if not tag_summary_df.empty:
        st.dataframe(tag_summary_df, width="stretch", height=260)
    else:
        st.caption("אין tags בפרופילים, לכן אין Tag Summary.")

    # ------------------ 7. Anchor / Clusters / Anomalies ------------------
    st.markdown("### 7️⃣ Anchor / Clusters / Anomalies")

    col_a1, col_a2 = st.columns(2)

    # Anchor tab alignment
    with col_a1:
        st.markdown("#### 🎯 Anchor Tab Alignment")
        if not similarity_df.empty:
            anchor_options = list(similarity_df.index)
            # אפשרות לבחור anchor לפי tab_type
            type_for_anchor = st.selectbox(
                "Anchor type (אופציונלי):",
                options=["(all types)"] + sorted(set(p.tab_type for p in profiles)),
                index=0,
                key="cmp_anchor_type",
            )
            if type_for_anchor != "(all types)":
                anchor_options = [p.tab_id for p in profiles if p.tab_type == type_for_anchor]

            anchor_tab = st.selectbox(
                "בחר Anchor tab",
                anchor_options,
                index=0,
                key="cmp_anchor_tab",
            )
            scores = compute_alignment_scores(similarity_df, benchmark_tab_id=anchor_tab)
            scores_sorted = scores.sort_values(ascending=False)
            st.dataframe(
                scores_sorted.to_frame(name="alignment_score"),
                width="stretch",
                height=260,
            )
        else:
            st.caption("אין similarity matrix — לא ניתן לחשב Alignment.")

    # Clustering
    with col_a2:
        st.markdown("#### 🧬 Hierarchical Clustering")
        if not distance_df.empty and distance_df.shape[0] > 1:
            max_clusters = st.slider(
                "מספר מקסימלי של אשכולות",
                min_value=2,
                max_value=min(8, distance_df.shape[0]),
                value=min(4, distance_df.shape[0]),
                step=1,
                key="cmp_max_clusters",
            )
            try:
                cluster_labels, _ = hierarchical_cluster_tabs(
                    distance_df,
                    method="ward",
                    max_clusters=int(max_clusters),
                )
                df_clusters = cluster_labels.to_frame(name="cluster_id")
                st.dataframe(df_clusters, width="stretch", height=260)
            except Exception as e:
                st.warning(f"Clustering נכשל (אולי SciPy לא מותקן): {e}")
        else:
            st.caption("אין distance matrix עם יותר מטאב אחד — אי אפשר לבצע clustering.")

    # Anomalies
    st.markdown("#### ⚠️ Tab Anomalies (Outliers)")
    if not distance_df.empty and distance_df.shape[0] > 2:
        anomalies_df = detect_tab_anomalies(
            distance_df,
            zscore_threshold=2.5,
            min_peers=2,
        )
        if not anomalies_df.empty:
            st.dataframe(anomalies_df, width="stretch", height=220)
        else:
            st.caption("לא נמצאו טאבים חריגים לפי קריטריון ה-anomaly.")
    else:
        st.caption("אין distance matrix מספיק מלאה — אי אפשר לזהות anomalies.")

    # ------------------ 8. Metric Diagnostics + Redundant KPIs ------------------
    st.markdown("### 8️⃣ Metric Diagnostics + Redundant KPIs")

    col_md1, col_md2 = st.columns(2)

    with col_md1:
        st.markdown("#### 📐 Metric diagnostics")
        metric_diag_df = _compute_metric_diagnostics(metric_vs_tab_df)
        if not metric_diag_df.empty:
            st.dataframe(metric_diag_df, width="stretch", height=260)
        else:
            st.caption("אין Metric vs Tab matrix, לכן אין דיאגנוסטיקה למטריקות.")

    with col_md2:
        st.markdown("#### 🧪 Redundant KPIs (high correlation)")
        if not metric_corr_df.empty:
            # נזהה זוגות עם |corr| > 0.9
            redundant_rows = []
            for i, m1 in enumerate(metric_corr_df.index):
                for j, m2 in enumerate(metric_corr_df.columns):
                    if j <= i:
                        continue
                    corr_val = metric_corr_df.loc[m1, m2]
                    if pd.isna(corr_val):
                        continue
                    if abs(corr_val) >= 0.9:
                        redundant_rows.append(
                            {"metric_1": m1, "metric_2": m2, "corr": float(corr_val)}
                        )
            if redundant_rows:
                df_red = pd.DataFrame(redundant_rows).sort_values("corr", ascending=False)
                st.dataframe(df_red, width="stretch", height=260)
                st.caption("מטריקות עם קורלציה גבוהה מאוד יכולות להיות redundant.")
            else:
                st.caption("לא נמצאו זוגות מטריקות עם |corr| ≥ 0.9.")
        else:
            st.caption("אין Metric correlation matrix זמינה.")

    # ------------------ 9. Tab Health ------------------
    st.markdown("### 9️⃣ Tab Health — Missing metrics per tab")

    tab_health_df = _compute_tab_health(profiles_df)
    if not tab_health_df.empty:
        st.dataframe(tab_health_df, width="stretch", height=220)
    else:
        st.caption("אין מידע על missing metrics.")

    # ------------------ 10. Export (CSV/JSON) ------------------
    st.markdown("### 🔟 Export — לשימוש ב-Agents / Offline")

    col_ex1, col_ex2 = st.columns(2)

    with col_ex1:
        st.caption("Export CSVs (Similarity / Distance / Metric vs Tab)")
        if st.button("⬇️ הכנת CSV-ים למטריצות", key="cmp_export_csv_btn"):
            try:
                similarity_df.to_csv("comparison_similarity.csv")
                distance_df.to_csv("comparison_distance.csv")
                metric_vs_tab_df.to_csv("comparison_metric_vs_tab.csv")
                st.success(
                    "נשמרו comparison_similarity.csv, comparison_distance.csv, comparison_metric_vs_tab.csv בסביבת הריצה."
                )
            except Exception as e:
                st.error(f"Export CSV נכשל: {e}")

    with col_ex2:
        st.caption("Export Bundle JSON (ל-Agent / תיעוד)")
        if st.button("⬇️ Export full bundle as JSON-like dict", key="cmp_export_json_btn"):
            try:
                # לא כותבים לקובץ בפועל כאן, אבל אפשר לשמור ב-session עבור Agent
                st.session_state["comparison_bundle_export"] = meta_bundle
                st.success("comparison_bundle.meta נשמר ב-session_state['comparison_bundle_export'].")
            except Exception as e:
                st.error(f"Export Bundle נכשל: {e}")

    st.markdown("---")
    st.caption(
        "Comparison Matrices Tab (v3): עובד רק על דאטה אמיתי, בלי דמו, "
        "משתמש ב-core.tab_comparison_matrices ומספק שכבת אינטליגנציה מלאה ברמת קרן."
    )
