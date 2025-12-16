# -*- coding: utf-8 -*-
"""
config_tab.py — טאב Streamlit לניהול קובצי קונפיג + DB Health (HF-grade)
========================================================================

מה הטאב הזה יודע לעשות:
------------------------
1. Config Editor (מיני-מערכת ניהול קונפיגים):
   - טעינה של config.json או פרופילים מתיקיית configs/
   - עריכת JSON גולמי בצורה בטוחה (עם json_safe)
   - שמירה על הקובץ הראשי / שמירה כפרופיל חדש
   - מחיקת פרופילים מיותרים
   - הצגת "סיכום מהיר" של הקונפיג (מספר זוגות, גרסה, פרמטרים חשובים)
   - אפשרות להוריד את הקונפיג כקובץ (Download)

2. DB Health (מיני-טאב בתוך הטאב):
   - שימוש ב-SqlStore.from_settings(AppContext.settings)
   - תצוגת engine_url / dialect / env / מספר טבלאות
   - כיסוי מחירים לפי symbol (n_rows, min_date, max_date)
   - השגיאה האחרונה מ-SqlStore (get_last_error)

הטאב מחולק למיני-טאבים:
    • "Config Editor" — כל ניהול הקונפיגים.
    • "DB Health" — מצב ה-SqlStore וה-DB (DuckDB).

מיועד להשתלב בדשבורד הראשי (dashboard.py) כטאב config.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from common.config_manager import load_config, save_config, list_configs
from common.json_safe import make_json_safe
from core.app_context import AppContext
from core.sql_store import SqlStore

logger = logging.getLogger(__name__)
if not logger.handlers:
    # לא עושים basicConfig בתוך ספרייה – זה תפקיד ה־dashboard הראשי
    logger.addHandler(logging.NullHandler())

CONFIGS_DIR = Path("configs")


# =============================================================================
# DB Health Panel (mini-tab)
# =============================================================================
def render_db_health_panel(app_ctx: AppContext) -> None:
    """
    פאנל קטן שמציג מצב DB (DuckDB / SqlStore) ברמת קרן.

    מיועד להתלבש כמיני-טאב בתוך טאב אחר.
    """
    # SqlStore לפי ה-settings של האפליקציה
    store = SqlStore.from_settings(app_ctx.settings)

    # תיאור engine
    info = store.describe_engine()
    engine = info.get("engine", {})
    tables_count = info.get("tables_count", 0)

    # כיסוי מחירים
    env = engine.get("default_env") or "dev"
    coverage_df = store.load_prices_coverage_summary(env=env)

    # שגיאה אחרונה אם הייתה
    last_err = store.get_last_error()

    st.subheader("🗄️ Database / SqlStore Health", anchor=False)

    # שורה עליונה – פרטי Engine + קונטקסט
    col0, col1, col2, col3 = st.columns([1.6, 1.1, 1.1, 1.2])
    with col0:
        st.markdown("**Engine URL**")
        st.code(str(engine.get("engine_url", "")), language="text")
    with col1:
        st.markdown("**Dialect / Env**")
        st.metric("Dialect", engine.get("dialect", "?"))
        st.metric("Env", engine.get("default_env", "?"))
    with col2:
        st.markdown("**Tables**")
        st.metric("Tables", tables_count)
        st.metric("Has prices", "✅" if info.get("has_prices") else "❌")
    with col3:
        st.markdown("**Core Tables**")
        st.caption("kv_store / dq_pairs / snapshots")
        st.write(
            f"kv_store: {'✅' if info.get('has_kv_store') else '❌'}\n"
            f"dq_pairs: {'✅' if info.get('has_dq_pairs') else '❌'}\n"
            f"snapshots: {'✅' if info.get('has_dashboard_snapshots') else '❌'}"
        )

    st.markdown("---")

    # 📈 כיסוי מחירים
    st.markdown("### 📈 Prices Coverage by Symbol")
    if coverage_df is None or coverage_df.empty:
        st.info("אין נתוני prices בטבלה כרגע (או שאין התאמה ל-env).")
    else:
        coverage_df = coverage_df.copy()
        coverage_df["min_date"] = coverage_df["min_date"].astype(str)
        coverage_df["max_date"] = coverage_df["max_date"].astype(str)

        min_global = coverage_df["min_date"].min()
        max_global = coverage_df["max_date"].max()
        total_rows = int(coverage_df["n_rows"].sum())

        k_col1, k_col2, k_col3 = st.columns(3)
        with k_col1:
            st.metric("Symbols", len(coverage_df))
        with k_col2:
            st.metric("Total rows", total_rows)
        with k_col3:
            st.metric("Date range", f"{min_global} → {max_global}")

        st.dataframe(
            coverage_df,
            use_container_width=True,
        )

    st.markdown("---")

    # 🧬 Universe / dq_pairs summary
    st.markdown("### 🧬 Universe / Pairs Quality (dq_pairs)")

    try:
        dq_df = store.load_pair_quality(env=env, latest_only=True)
    except TypeError:
        # במקרה ויש חתימות ישנות איפשהו – fallback לקריאה הפשוטה
        dq_df = store.load_pair_quality()

    if dq_df is None or dq_df.empty:
        st.info("לא נמצאו רשומות ב־dq_pairs / pairs_quality עבור ה-env הנוכחי.")
    else:
        dq_df = dq_df.copy()
        n_pairs = len(dq_df["pair"].unique()) if "pair" in dq_df.columns else len(dq_df)

        last_ts = None
        if "ts_utc" in dq_df.columns:
            try:
                last_ts = str(max(dq_df["ts_utc"]))
            except Exception:
                last_ts = None

        col_u1, col_u2, col_u3 = st.columns(3)
        with col_u1:
            st.metric("Pairs (latest)", n_pairs)
        with col_u2:
            st.metric("Rows", len(dq_df))
        with col_u3:
            st.metric("Last ts_utc", last_ts or "N/A")

        preferred_cols = [
            "pair",
            "sym_x",
            "sym_y",
            "profile",
            "env",
            "section",
            "score",
            "z_open",
            "z_close",
        ]
        cols = [c for c in preferred_cols if c in dq_df.columns]
        if not cols:
            cols = list(dq_df.columns)  # fallback

        st.dataframe(
            dq_df[cols].head(100),
            use_container_width=True,
        )

    st.markdown("---")

    # 🔔 Signals Universe — end-to-end: prices → dq_pairs → signals
    st.markdown("### 🔔 Signals Universe (signals_universe)")

    # אפשר בעתיד להוסיף כאן selectbox לפרופיל, בינתיים נשתמש ב-All
    try:
        sig_df = store.load_signals_universe(
            env=env,
            latest_only=True,
            limit=500,
        )
    except TypeError:
        # במקרה שיש חתימה ישנה
        sig_df = store.load_signals_universe()

    if sig_df is None or sig_df.empty:
        st.info("לא נמצאו אותות ב־signals_universe עבור ה-env הנוכחי.")
    else:
        sig_df = sig_df.copy()

        # פיצוח pair אם אין, לפי sym_x/sym_y
        if "pair" not in sig_df.columns and {"sym_x", "sym_y"} <= set(sig_df.columns):
            sig_df["pair"] = (
                sig_df["sym_x"].astype(str) + "-" + sig_df["sym_y"].astype(str)
            )

        if "pair" in sig_df.columns:
            n_signal_pairs = len(sig_df["pair"].unique())
        else:
            n_signal_pairs = len(sig_df)

        n_rows = len(sig_df)

        # מעט אינדיקציות מעניינות: deploy_tier / edge_hint אם קיימים
        deploy_counts = None
        if "meta_deploy_tier" in sig_df.columns:
            deploy_counts = sig_df["meta_deploy_tier"].value_counts()
        elif "deploy_tier" in sig_df.columns:
            deploy_counts = sig_df["deploy_tier"].value_counts()

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Signal rows (latest)", n_rows)
        with col_s2:
            st.metric("Pairs with signals", n_signal_pairs)
        with col_s3:
            if deploy_counts is not None and not deploy_counts.empty:
                top_tier = deploy_counts.index[0]
                st.metric("Top deploy tier", f"{top_tier} ({int(deploy_counts.iloc[0])})")
            else:
                st.metric("Deploy tiers", "N/A")

        preferred_sig_cols = [
            "pair",
            "sym_x",
            "sym_y",
            "profile_name",
            "env",
            "signal",
            "zscore",
            "edge",
            "meta_edge_hint",
            "meta_deploy_tier",
        ]
        sig_cols = [c for c in preferred_sig_cols if c in sig_df.columns]
        if not sig_cols:
            sig_cols = list(sig_df.columns)

        st.dataframe(
            sig_df[sig_cols].head(100),
            use_container_width=True,
        )

    st.markdown("---")

    # ⚠️ שגיאה אחרונה
    st.markdown("### ⚠️ Last SqlStore Error")
    if last_err:
        st.error(last_err)
    else:
        st.success("אין שגיאה אחרונה מדווחת מ-SqlStore (ה-DB נראה בריא).")

# =============================================================================
# Helpers — Config logic
# =============================================================================
def _safe_dumps(cfg: Dict[str, Any]) -> str:
    """המרת קונפיג ל־JSON בצורה בטוחה להצגה/עריכה."""
    try:
        return json.dumps(make_json_safe(cfg), ensure_ascii=False, indent=2)
    except Exception:  # pragma: no cover - הגנה כללית
        logger.exception("Failed to dump config as JSON")
        return "{}"


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    """ניסיון לפענח JSON מהטקסט; מחזיר None אם יש שגיאה ומציג הודעה ב־UI."""
    text = (text or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        st.error(f"שגיאת JSON: {e}")
        logger.warning("JSON decode error in config editor: %s", e)
        return None


def _normalize_profile_name(raw: str) -> Optional[str]:
    """ניקוי שם קובץ פרופיל והוספת סיומת .json במידת הצורך."""
    raw = (raw or "").strip()
    if not raw:
        return None
    if not raw.endswith(".json"):
        raw = raw + ".json"
    return raw


def _get_nested(cfg: Dict[str, Any], *path: str) -> Any:
    """שליפת ערכים מרובי־רמות (cfg[a][b][c]) בצורה בטוחה."""
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _render_config_summary(cfg: Dict[str, Any]) -> None:
    """סיכום מהיר של הקונפיג ברמת 'קרן גידור' (KPIים בסיסיים)."""
    meta = cfg.get("metadata", {}) if isinstance(cfg.get("metadata"), dict) else {}

    # מספר זוגות (אם קיים)
    pairs = cfg.get("pairs")
    num_pairs = len(pairs) if isinstance(pairs, list) else None

    # פרמטרים נפוצים – מנסים גם בשורש וגם תחת strategy/risk
    z_open = _get_nested(cfg, "strategy", "z_open") or cfg.get("z_open")
    z_close = _get_nested(cfg, "strategy", "z_close") or cfg.get("z_close")
    max_exp = (
        _get_nested(cfg, "risk", "max_exposure_per_trade")
        or cfg.get("max_exposure_per_trade")
    )
    max_trades = (
        _get_nested(cfg, "risk", "max_open_trades")
        or cfg.get("max_open_trades")
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("מספר זוגות בקובץ", num_pairs if num_pairs is not None else "לא מוגדר")
    with col2:
        st.metric("גרסת קובץ", meta.get("version", "N/A"))
    with col3:
        st.metric("יוצר / מקור", meta.get("author", "N/A"))

    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Z פתיחה (z_open)", z_open if z_open is not None else "N/A")
    with col5:
        st.metric("Z סגירה (z_close)", z_close if z_close is not None else "N/A")
    with col6:
        st.metric(
            "Max exposure per trade",
            max_exp if max_exp is not None else "N/A",
        )

    col7, col8, _ = st.columns(3)
    with col7:
        st.metric(
            "Max open trades",
            max_trades if max_trades is not None else "N/A",
        )
    with col8:
        st.metric("Pairs section?", "✓" if isinstance(pairs, list) else "✗")

    st.markdown("---")
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        st.markdown("#### 🔧 Strategy")
        strategy = cfg.get("strategy", {})
        if isinstance(strategy, dict) and strategy:
            st.json(make_json_safe(strategy))
        else:
            st.caption("אין אובייקט strategy מפורש בקובץ.")

    with sub_col2:
        st.markdown("#### 🛡 Risk / Filters / Metadata")
        risk = cfg.get("risk", {})
        filters = cfg.get("filters", {})
        st.caption("Risk:")
        if isinstance(risk, dict) and risk:
            st.json(make_json_safe(risk))
        else:
            st.caption("אין אובייקט risk מפורש.")

        st.caption("Filters:")
        if isinstance(filters, dict) and filters:
            st.json(make_json_safe(filters))
        else:
            st.caption("אין אובייקט filters מפורש.")

        st.caption("Metadata:")
        if isinstance(meta, dict) and meta:
            st.json(make_json_safe(meta))
        else:
            st.caption("אין metadata מפורט.")


# =============================================================================
# Internal render: Config Editor content (for mini-tab)
# =============================================================================
def _render_config_editor(app_ctx: Optional[AppContext]) -> None:
    """
    מימוש מלא של לוגיקת עריכת הקונפיגים — נקרא מתוך מיני-טאב "Config Editor".
    """
    st.caption(
        "כאן תוכל לטעון, לערוך ולשמור את קובץ ה־config הראשי (config.json) "
        "ואת כל הפרופילים בתיקיית configs/ ברמת בקרת־גרסה ידנית."
    )

    # --- רשימת קבצים זמינים ---
    profiles: List[str] = list_configs() or []
    options: List[str] = ["config.json"] + profiles

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_config = st.selectbox(
            "בחר קובץ קונפיג לטעינה",
            options=options,
            index=0,
            key="cfg_select",
        )
    with col_sel2:
        if selected_config == "config.json":
            st.info("עריכת הקובץ הראשי של הדשבורד.")
        else:
            st.success(f"עובד על פרופיל: {selected_config}")

    # --- טעינת הקונפיג מה־config_manager ---
    cfg: Dict[str, Any] = load_config(selected_config)

    # --- סיכום מהיר ---
    with st.expander("📊 סיכום מהיר של הקונפיג הנוכחי", expanded=False):
        _render_config_summary(cfg)

    # --- כפתור הורדה (Download) ---
    download_col, _ = st.columns([1, 3])
    with download_col:
        safe_bytes = _safe_dumps(cfg).encode("utf-8")
        st.download_button(
            label="⬇️ הורדת הקונפיג כקובץ JSON",
            data=safe_bytes,
            file_name=selected_config,
            mime="application/json",
        )

    st.subheader("✏️ עריכת JSON גולמי")

    # טקסט לעריכה – משויכים למפתח לפי הקובץ הנבחר כדי לא לערבב בין פרופילים
    editor_key = f"config_json_{selected_config}"
    default_text = _safe_dumps(cfg)
    json_text = st.text_area(
        "הגדרות במבנה JSON (ניתן לעריכה מלאה):",
        value=default_text,
        height=420,
        key=editor_key,
    )

    # אינדיקציה לשינויים (unsaved changes)
    is_modified = json_text.strip() != default_text.strip()
    if is_modified:
        st.warning("יש שינויים שלא נשמרו בקונפיג הנוכחי.", icon="⚠️")
    else:
        st.caption("אין שינויים ביחס לקובץ הטעון מהדיסק.")

    # --- פעולות שמירה / שמירה כחדש ---
    col_save1, col_save2 = st.columns([2, 2])
    with col_save1:
        if st.button("💾 שמירה על הקובץ הנוכחי", key="btn_save_current"):
            parsed = _parse_json(json_text)
            if parsed is not None:
                # אם מדובר ב־config.json – file_name=None → שמירה לנתיב הראשי
                file_name: Optional[str] = (
                    None if selected_config == "config.json" else selected_config
                )
                saved_path = save_config(parsed, file_name)
                st.success(f"הקונפיג נשמר בהצלחה כ־{saved_path}")
                logger.info("Config saved: %s", saved_path)

    with col_save2:
        new_profile_name = st.text_input(
            "שם פרופיל חדש (ללא או עם ‎.json‎):",
            key="cfg_new_profile_name",
            placeholder="my_profile.json",
        )
        if st.button("📁 שמירה כפרופיל חדש", key="btn_save_as_new"):
            parsed = _parse_json(json_text)
            if parsed is not None:
                target_name = _normalize_profile_name(new_profile_name)
                saved_path = save_config(parsed, target_name)
                st.success(f"פרופיל חדש נשמר כ־{saved_path}")
                logger.info("Config profile saved: %s", saved_path)

    # --- פעולות תחזוקה: מחיקה / רענון / רשימת פרופילים --- 
    st.markdown("---")
    col_maint1, col_maint2, col_maint3 = st.columns(3)

    with col_maint1:
        if st.button("↩ טעינת הקובץ מחדש מהדיסק", key="btn_reload_cfg"):
            st.experimental_rerun()

    with col_maint2:
        if selected_config != "config.json":
            if st.button("🗑 מחיקת הפרופיל הנוכחי", key="btn_delete_profile"):
                cfg_path = CONFIGS_DIR / selected_config
                if cfg_path.exists():
                    try:
                        cfg_path.unlink()
                        st.success(f"הפרופיל {selected_config} נמחק מתיקיית configs/.")
                        logger.info("Config profile deleted: %s", cfg_path)
                        st.experimental_rerun()
                    except Exception as e:  # pragma: no cover
                        logger.exception("Failed to delete profile %s", cfg_path)
                        st.error(f"שגיאה במחיקת הפרופיל: {e}")
                else:
                    st.warning(f"הקובץ {cfg_path} לא נמצא בדיסק.")

    with col_maint3:
        with st.expander("📚 פרופילים זמינים", expanded=False):
            profiles_now = list_configs() or []
            if profiles_now:
                st.write("פרופילים בתיקיית configs/:")
                for name in profiles_now:
                    path = CONFIGS_DIR / name
                    details: List[str] = []
                    if path.exists():
                        size = path.stat().st_size
                        details.append(f"גודל: {size} bytes")
                    st.write(
                        f"• **{name}**"
                        + (f" — {', '.join(details)}" if details else "")
                    )
            else:
                st.info("אין עדיין פרופילים שמורים בתיקיית configs/. שמור אחד חדש כדי להתחיל.")


# =============================================================================
# Main render function (with mini-tabs)
# =============================================================================
def render_config_tab(app_ctx: Optional[AppContext] = None) -> None:
    """
    פונקציית הרינדור הראשית של טאב הקונפיג.

    app_ctx:
        אם מועבר מה-dashboard (מומלץ), משתמשים בו.
        אם לא — ננסה AppContext.get_global().
    """
    if app_ctx is None:
        try:
            app_ctx = AppContext.get_global()
        except Exception:
            app_ctx = None  # נסבול גם מצב שאין AppContext גלובלי

    st.header("🧾 ניהול קונפיגים (הגדרות מערכת)")

    if app_ctx is not None:
        with st.expander("📌 App / Env context", expanded=False):
            env_str = getattr(app_ctx.settings, "env", "dev")
            profile_str = getattr(app_ctx.settings, "profile", "default")

            # project_root יכול להיות שדה ב-app_ctx, או ב-settings, או בכלל לא קיים
            project_root = getattr(app_ctx, "project_root", None)
            if project_root is None:
                project_root = getattr(app_ctx.settings, "project_root", None)

            st.write(
                f"**Env**: `{env_str}` • "
                f"**Profile**: `{profile_str}`"
            )
            if project_root is not None:
                st.caption(f"Project root: `{project_root}`")
            else:
                st.caption("Project root לא מוגדר ב-AppContext (לא קריטי).")
    else:
        st.caption("AppContext לא זמין — עובדים במצב סטנדרטי של ניהול קונפיגים.")

    # מיני-טאבים בתוך הטאב: Config Editor / DB Health
    tab_editor, tab_db_health = st.tabs(["Config Editor", "DB Health"])

    with tab_editor:
        _render_config_editor(app_ctx)

    with tab_db_health:
        if app_ctx is None:
            st.error("AppContext לא זמין — לא ניתן להציג DB Health בלי קונטקסט אפליקציה.")
        else:
            render_db_health_panel(app_ctx)
