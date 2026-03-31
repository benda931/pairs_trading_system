# -*- coding: utf-8 -*-
"""
root/macro_tab.py — טאב התאמות מאקרו ומצבי שוק (Tab 8)
========================================================

טאב המאקרו הוא שכבת-על על כל המערכת, ומחבר בין:
- מצב מאקרו נוכחי (Snapshot + Regimes)
- התאמות גלובליות לתיק (Exposure / Filters / Caps)
- "DNA מאקרו" לזוג בודד
- ציר זמן של משטרים ושוקים (Regime Timeline + Macro Shocks)
- כלי UX מקצועיים (Presets, Overlays, Backtests)

שכבות הלוגיקה (High-Level Design)
----------------------------------
1. Base Layer — Global Macro Context
   - בחירת טווח תאריכים (Date Range) למאקרו, ברירת מחדל: שנה אחורה.
   - בחירת תדירות מאקרו: D / W / M (יומי / שבועי / חודשי).
   - בחירת סביבת עבודה: Backtest / Live (לטובת חיבור ל־IBKR / DuckDB / SQL).
   - טעינת:
       * MacroBundle (מודל ישן): CPI, Unemployment, Policy Rate, Yield Curve,
         PMI, Credit Spread, Oil, DXY, VIX וכו'.
       * macro_factors (מודל חדש): rate_short / rate_long / slopes / VIX term structure /
         credit_spread / risk_on_proxy / FX וכו' דרך:
             - load_macro_factors(...)
             - add_derived_factors(...)
   - בניית MacroSnapshot:
       * build_macro_snapshot(macro_df, cfg_factors)
       * summarize_macro_snapshot(snapshot) → משפט "מצב מאקרו" קצר.
       * regimes + group_regimes + summary_label נשמרים ל־session_state.

2. Portfolio Macro Adjustments — התאמות גלובליות לתיק
   - compute_adjustments(pairs_df, bundle, cfg_macro) → AdjustmentResult:
       * exposure_multiplier – מקדם גלובלי לתיק.
       * pair_adjustments – pair_id → multiplier.
       * filters – pair_id → include True/False.
       * pair_scores – Macro Fit לכל זוג (0–100).
       * caps_hints – רמזי Caps לפי סקטור/אזור/מטבע.
       * meta – תאריך, משטר, mean_score וכו'.
   - התוצאה נשמרת ב־session_state (מאפשר גישה מטאבים אחרים וממנוע האקזקיושן).

3. Pair Macro DNA — רגישות מאקרו לזוג בודד
   - בחירת זוג מ־pairs_df["pair_id"] + בחירת שיטת Spread:
       * log_spread / diff / ratio.
   - בניית spread_series מתוך prices_wide (sym_a, sym_b).
   - יישור macro_df לסדרה והפקת Regime DF:
       * build_macro_regime_series(macro_df) → rates_regime / curve_regime / vol_regime וכו'.
   - compute_pair_macro_sensitivity(spread_series, macro_df, regime_df, cfg_sens) →
       * PairMacroSensitivity:
           - exposures: factor → MacroExposure(beta, tstat, pvalue).
           - regime_perf: regime → RegimePerformance(mean, vol, sharpe, hit_ratio, n_obs).
           - overall_score (0–100).
           - summary_text – תיאור טקסטואלי איכותי.
       * build_exposures_table(...) / build_regime_performance_table(...) לנראות.

4. Regime Timeline & Macro Shocks
   - build_macro_regime_series(macro_df) → Regime DF לאורך הזמן.
   - Heatmap / Timeline:
       * factorize labels → codes
       * plotly heatmap: X=time, Y=regime_type, color=code.
   - detect_macro_shocks(macro_df) → DataFrame עם:
       * has_shock, shock_factors, severity וכו'.
   - UX:
       * טבלאות Tail, כפתור הורדה CSV, ותמיכה למחקר וביצועי תיק.

5. Advanced UX & Macro Overlays
   - Feature engineering ל־macro_factors:
       * Z-window, momentum windows, volatility windows, EWMA, lags, YoY.
   - Presets:
       * Risk-On Friendly / Risk-Off Hedge / Balanced Regime.
       * שמירה/טעינה של MacroConfig / MacroFactorConfig / PairMacroSensitivityConfig כ־JSON.
   - Risk Controls:
       * sizing: invvol / ERC, cov_window, cov_shrink.
       * Sector / Region / Currency caps.
   - Macro Overlay:
       * שילוב Macro Fit / Sensitivity / Caps לכל זוג בטבלה אחת (“Overlay View”).

6. Integration & State Contract
   - פונקציה ציבורית אחת בלבד:
       * render(pairs_df: pd.DataFrame, cfg: MacroConfig | None, bundle: MacroBundle | None)
         → AdjustmentResult.
   - חוזה state מול שאר המערכת (session_state keys):
       * "macro_adjustments_result" / "macro_tab_result"
       * "macro_meta"
       * "macro_profile"
       * "macro_regimes_prob"
       * "macro_features_df"
       * "macro_shocks_df"
       * "macro_tab_bundle" / "macro_tab_cfg" / "macro_tab_cfg_factors" / "macro_tab_macro_df"

הקובץ הנוכחי אחראי אך ורק על **UI + חיבור לוגיקה**. כל החישובים הסטטיסטיים/מאקרו
נשארים במודולים: common.macro_adjustments, core.macro_engine, core.macro_data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)
from datetime import datetime, timezone

from uuid import uuid4

import pandas as pd
import streamlit as st

# Optional plotting (Heatmaps, timelines, PNG export)
try:
    import plotly.express as px  # type: ignore
    import plotly.io as pio  # type: ignore

    _HAS_PX = True
except Exception:  # noqa: BLE001
    _HAS_PX = False
    pio = None  # type: ignore[assignment]

# Optional YAML export (תמיכה בהורדת קונפיג/תוצאה כ-YAML)
try:
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # noqa: BLE001
    _HAS_YAML = False

# מנוע המאקרו המרכזי — כל החישובים האמיתיים (אין לוגיקה כפולה כאן)
# מאקרו מערכתית / התאמות לתיק
from common.macro_adjustments import (
    MacroConfig,
    MacroFactorConfig,
    MacroBundle,
    AdjustmentResult,
    load_macro_bundle,
    compute_adjustments,
    render_streamlit_ui,
)

# פקטורי מאקרו / Regimes / Snapshot (אם יש לך common/macro_factors.py)
from common.macro_factors import (
    load_macro_factors,
    add_derived_factors,
    build_macro_snapshot,
    summarize_macro_snapshot,
    build_macro_regime_series,
    detect_macro_shocks,
)

# רגישות לזוג (pair-level macro sensitivity)
from common.macro_sensitivity import (
    PairMacroSensitivityConfig,
    compute_pair_macro_sensitivity,
    build_exposures_table,          # טבלת betas לפקטורים
    build_regime_performance_table, # ביצועים לפי משטרים לזוג
)


LOGGER = logging.getLogger("root.macro_tab")

# מפתח בסיס לטאב; ממנו נגזרים כל המפתחות ל-Streamlit (widgets, flags, state)
TAB_KEY = "macro_root_tab8"


# ---------------------------------------------------------------------------
# חוזה state ומפתחות session_state
# ---------------------------------------------------------------------------


class MacroStateKeys:
    """קבועי מפתחות session_state לטאב המאקרו.

    חשוב:
    - מונע "magic strings" מפוזרים בקוד.
    - מאפשר לשאר הטאבים/מודולים לדעת מה בדיוק נשמר תחת איזה מפתח.
    """

    RESULT: str = "macro_adjustments_result"
    RESULT_ALT: str = "macro_tab_result"

    META: str = "macro_meta"

    PROFILE: str = "macro_profile"
    REGIMES_PROB: str = "macro_regimes_prob"
    FEATURES_DF: str = "macro_features_df"
    SHOCKS_DF: str = "macro_shocks_df"

    BUNDLE: str = "macro_tab_bundle"
    CFG: str = "macro_tab_cfg"
    FACTOR_CFG: str = "macro_tab_cfg_factors"
    MACRO_DF: str = "macro_tab_macro_df"


JSONDict = Dict[str, Any]
MacroFreq = Literal["D", "W", "M"]
MacroEnv = Literal["backtest", "live"]


class MacroMetaPayload(TypedDict, total=False):
    """מבנה המטא־דאטה שנשמר מן ה־AdjustmentResult לצורך דוחות ו־UX."""

    ts: str
    apply_mode: str
    exposure_multiplier: float
    pairs: int
    included: int
    regime_label: str
    mean_score: float


def keygen(namespace: str, *parts: object) -> str:
    """יוצר מפתח ייחודי עקבי ל־Streamlit.

    Parameters
    ----------
    namespace:
        מרחב שמות בסיסי (למשל TAB_KEY) כדי להימנע מהתנגשויות עם טאבים אחרים.
    *parts:
        חלקים נוספים שמזהים את הרכיב (section, widget, sub-key...).

    Returns
    -------
    str
        מחרוזת מפתח אחידה בסגנון: ``"{namespace}.part1.part2"``.
    """
    tokens = [str(namespace)] + [str(p) for p in parts if p is not None]
    return ".".join(tokens)


def get_flag(name: str, default: bool = False) -> bool:
    """קורא דגל תכונה מ־session_state (או ברירת מחדל).

    Notes
    -----
    - כל הדגלים נשמרים תחת המפתח: ``feature_flag.{name}``.
    - מאפשר שליטה גלובלית על UX (הורדות, YAML, ניסויים, debug וכו').
    """
    key = f"feature_flag.{name}"
    val = st.session_state.get(key, default)
    return bool(val)


def set_flag(name: str, value: bool) -> None:
    """מעדכן דגל תכונה ב־session_state."""
    st.session_state[f"feature_flag.{name}"] = bool(value)


def get_session_uid() -> str:
    """מחזיר מזהה ייחודי לריצה הנוכחית של הטאב.

    משמש ל:
    - יצירת מפתחות יציבים עבור render_streamlit_ui ומודולים אחרים.
    - מניעת StreamlitDuplicateElementKey כשיש מספר טאבים/אפליקציות.
    """
    uid = st.session_state.get("session_uid")
    if not isinstance(uid, str) or not uid:
        uid = str(uuid4())
        st.session_state["session_uid"] = uid
    return uid


def state_get(key: str, default: Any = None) -> Any:
    """Wrapper קטן ל־session_state.get — לשיפור קריאות הקוד."""
    return st.session_state.get(key, default)


def state_set(key: str, value: Any) -> None:
    """Wrapper קטן ל־session_state[key] = value — אחיד בכל הטאב."""
    st.session_state[key] = value


@dataclass
class MacroWorkspace:
    """תצורת סביבת העבודה (שכבת בסיס) של טאב המאקרו.

    Attributes
    ----------
    start_date:
        תחילת טווח התאריכים למאקרו (אופציונלי, לפי בחירת המשתמש).
    end_date:
        סוף טווח התאריכים למאקרו.
    freq:
        תדירות מאקרו: "D" / "W" / "M".
    env:
        סביבת עבודה: "backtest" / "live".
    """

    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    freq: MacroFreq = "D"
    env: MacroEnv = "backtest"


@dataclass
class MacroTabState:
    """State לוגי מרוכז לטאב המאקרו (ללא אובדן תאימות ל-session_state).

    שימוש אופייני (בחלקים הבאים):
    -------------------------------
    1. יצירת State בתחילת render():
           ws = MacroWorkspace(...)
           state = MacroTabState(workspace=ws, macro_cfg=cfg, ...)

    2. מילוי שדות מחושבים בהמשך:
           state.macro_df = macro_df
           state.adjustment_result = result
           state.macro_snapshot = snapshot

    3. סנכרון עם session_state (שקוף לשאר הטאבים):
           state_set(MacroStateKeys.RESULT, state.adjustment_result)
           state_set(MacroStateKeys.MACRO_DF, state.macro_df)
    """

    workspace: MacroWorkspace
    macro_cfg: MacroConfig
    factor_cfg: Optional[MacroFactorConfig] = None
    pair_sens_cfg: Optional[PairMacroSensitivityConfig] = None
    bundle: Optional[MacroBundle] = None

    # תוצרים מחושבים (ימולאו בחלקים 2–5)
    adjustment_result: Optional[AdjustmentResult] = None
    macro_snapshot: Optional[JSONDict] = None
    macro_df: Optional[pd.DataFrame] = None
    macro_factors_df: Optional[pd.DataFrame] = None
    macro_regime_df: Optional[pd.DataFrame] = None
    macro_regimes_prob: Optional[pd.DataFrame] = None
    macro_features_df: Optional[pd.DataFrame] = None
    macro_shocks_df: Optional[pd.DataFrame] = None
    macro_profile: Optional[Dict[str, JSONDict]] = None

# ---------------------------------------------------------------------------
# Feature Flags, Debug Tools, ולידציה, Meta & KPIs, Live Health
# ---------------------------------------------------------------------------

class MacroDebugPayload(TypedDict, total=False):
    """Payload ל־Debug Panels (לשימוש פנימי בטאב המאקרו)."""

    snapshot: JSONDict
    meta: MacroMetaPayload
    overlay: Dict[str, JSONDict]


# ===========================================================================
# Feature Flags / ניסויים / Debug
# ===========================================================================


def feature_toggles_ui(namespace: str = TAB_KEY) -> None:
    """UI לניהול דגלי תכונה (Feature Flags) רלוונטיים לטאב המאקרו.

    דגלים קיימים
    -------------
    - macro_download:
        מאפשר כפתור הורדת תוצאת המאקרו (AdjustmentResult + meta) כ-JSON.
    - export_yaml:
        מאפשר הורדת תוצאה גם בפורמט YAML (אם PyYAML מותקן).
    - macro_debug:
        מציג תיבות Debug עם JSON גולמי (snapshot / meta / overlay).
    - macro_show_regimes_table:
        כאשר פעיל, יופיעו גם טבלאות Regimes מפורטות בנוסף ל-Heatmap.
    - macro_warn_pairs_quality:
        כאשר פעיל, יוצגו אזהרות איכות על pairs_df (עמודות חסרות וכו').
    - macro_show_source_summary:
        כאשר פעיל, מוצגת טבלת סיכום של מקורות הדאטה (IBKR / YF / DuckDB / SQL).
    """
    with st.expander("⚙️ מאפייני טאב (Feature Flags)", expanded=False):
        dl = st.checkbox(
            "אפשר הורדת תוצאה (JSON)",
            value=get_flag("macro_download", False),
            key=keygen(namespace, "ff", "download"),
            help="כאשר פעיל, יוצג כפתור הורדה של תוצאת המאקרו בסוף הטאב.",
        )
        set_flag("macro_download", dl)

        yml = st.checkbox(
            "אפשר הורדת YAML",
            value=get_flag("export_yaml", False),
            key=keygen(namespace, "ff", "yaml"),
            help="דורש את הספרייה 'pyyaml'. יוסיף כפתור הורדת YAML.",
        )
        set_flag("export_yaml", yml)

        dbg = st.checkbox(
            "מצב Debug (הצג JSON גולמי)",
            value=get_flag("macro_debug", False),
            key=keygen(namespace, "ff", "debug"),
            help="יוסיף תיבות מידע עם JSON גולמי (snapshot/meta/overlay) למפתחים.",
        )
        set_flag("macro_debug", dbg)

        show_reg_tbl = st.checkbox(
            "הצג גם טבלת Regimes מפורטת",
            value=get_flag("macro_show_regimes_table", True),
            key=keygen(namespace, "ff", "reg_tbl"),
            help="כאשר פעיל — בנוסף ל-Heatmap יוצגו גם טבלאות Regime DF.",
        )
        set_flag("macro_show_regimes_table", show_reg_tbl)

        warn_pairs = st.checkbox(
            "הצג אזהרות איכות על pairs_df",
            value=get_flag("macro_warn_pairs_quality", True),
            key=keygen(namespace, "ff", "warn_pairs"),
            help="מתריע כאשר חסרות עמודות מאקרו חשובות ביקום הזוגות.",
        )
        set_flag("macro_warn_pairs_quality", warn_pairs)

        show_source_summary = st.checkbox(
            "הצג סיכום מקורות דאטה (IBKR/YF/DuckDB/SQL)",
            value=get_flag("macro_show_source_summary", True),
            key=keygen(namespace, "ff", "src_summary"),
            help="מציג טבלת Summary על מקורות המאקרו הפעילים.",
        )
        set_flag("macro_show_source_summary", show_source_summary)

    _render_flags_badge_row()


def _render_flags_badge_row() -> None:
    """שורת Badges קטנה שמראה בזריזות אילו דגלים פעילים."""
    flags = {
        "JSON": get_flag("macro_download", False),
        "YAML": get_flag("export_yaml", False) and _HAS_YAML,
        "Debug": get_flag("macro_debug", False),
        "Regimes Tbl": get_flag("macro_show_regimes_table", False),
    }
    parts: List[str] = []
    for name, active in flags.items():
        color = "#16a34a" if active else "#4b5563"
        parts.append(
            "<span style='display:inline-block;margin-right:4px;"
            f"padding:2px 8px;border-radius:999px;background:{color};"
            "color:white;font-size:11px;'>"
            f"{name}"
            "</span>",
        )
    if parts:
        st.markdown(
            "<div style='margin-bottom:6px;'>" + "".join(parts) + "</div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# ולידציה של pairs_df + עזרי DataFrame
# ===========================================================================


def _validate_pairs_df(pairs_df: pd.DataFrame) -> List[str]:
    """בודק שעמודות בסיס קיימות ומחזיר רשימת אזהרות (אם יש).

    בדיקות
    -------
    1. pair_id (חובה) — מזהה ייחודי לזוג, משמש כ-key לכל המערכות.
    2. עמודות מאקרו מומלצות (לא חובה, אך מחזקות את שכבת המאקרו):
       - sector / industry
       - region / country
       - currency
       - macro_bucket (למשל: Growth, Value, Duration, Cyclical)
       - macro_sensitivity (מידע קודם על רגישות מאקרו, אם קיים).
    3. בדיקת כפילויות:
       - pair_id כפולים.
    4. בדיקת כיסוי:
       - כמה אחוז מהזוגות חסרים להם שדות מאקרו "טובים שיהיו".
    """
    warnings: List[str] = []

    if pairs_df is None or pairs_df.empty:
        warnings.append("pairs_df ריק — הטאב יפעל במצב תצוגה בלבד.")
        return warnings

    cols = set(pairs_df.columns)

    # 1. חובה: pair_id
    required = {"pair_id"}
    missing = required - cols
    if missing:
        warnings.append(f"חסרות עמודות נדרשות ב-pairs_df: {sorted(missing)}")

    # 2. עמודות מאקרו מומלצות
    recommended_sets = [
        {"sector"},
        {"region", "country"},
        {"currency"},
        {"macro_bucket"},
        {"macro_sensitivity"},
    ]
    for rec in recommended_sets:
        if not rec.issubset(cols):
            warnings.append(
                f"עמודות מאקרו מומלצות חסרות (לא חובה אבל עדיף): {sorted(rec)}",
            )

    # 3. כפילויות pair_id
    if "pair_id" in cols:
        dup_mask = pairs_df["pair_id"].duplicated(keep=False)
        if bool(dup_mask.any()):
            n_dup = int(dup_mask.sum())
            warnings.append(
                f"נמצאו {n_dup} רשומות עם pair_id כפולים — כדאי לנקות לפני שימוש אמיתי.",
            )

    # 4. כיסוי שדות מאקרו "טובים שיהיו"
    coverage_checks = [
        ("sector", "כיסוי סקטור"),
        ("region", "כיסוי region"),
        ("currency", "כיסוי מטבע"),
        ("macro_sensitivity", "כיסוי macro_sensitivity"),
    ]
    for col_name, label in coverage_checks:
        if col_name in cols:
            non_null = pairs_df[col_name].notnull().sum()
            total = len(pairs_df)
            if total > 0:
                coverage = 100.0 * non_null / total
                if coverage < 60.0:
                    warnings.append(
                        f"{label} נמוך ({coverage:.1f}%) — מומלץ להשלים נתונים לניתוח מאקרו איכותי.",
                    )

    return warnings


def _ensure_pairs_df(pairs_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """דואג שתמיד יהיה DataFrame 'חוקי' לטאב.

    - אם pairs_df None או ריק → יוחזר DF ריק עם pair_id (ליציבות ה-UI).
    - אם חסרה עמודת pair_id → נייצר אותה מתוך index.
    """
    if pairs_df is None or pairs_df.empty:
        return pd.DataFrame({"pair_id": pd.Index([], name="pair_id")})
    if "pair_id" not in pairs_df.columns:
        df = pairs_df.copy()
        df["pair_id"] = df.index.astype(str)
        return df
    return pairs_df

def _pairs_list_to_df_for_tab(pairs: List[Any]) -> pd.DataFrame:
    """
    ממיר את ה-list שמגיע מה-dashboard.get_pairs(...)
    ל-DataFrame עם pair_id ועמודות sym_x/sym_y בסיסיות.

    זה מאפשר לטאב המאקרו לעבוד עם ה-API החדש של הדשבורד,
    שבו המעבר הוא pairs בתור list (str/dict/tuple), ולא DataFrame.
    """
    # אם כבר קיבלנו DataFrame – רק נדאג שהוא תקין
    if isinstance(pairs, pd.DataFrame):
        return _ensure_pairs_df(pairs)

    rows: List[Dict[str, Any]] = []
    for p in pairs:
        pair_id: str
        sym_x: Optional[str] = None
        sym_y: Optional[str] = None

        if isinstance(p, dict):
            sym_x = p.get("sym_x") or p.get("a") or p.get("symbol_a")
            sym_y = p.get("sym_y") or p.get("b") or p.get("symbol_b")
            pair_id = (
                p.get("pair_id")
                or p.get("pair")
                or (f"{sym_x}-{sym_y}" if sym_x and sym_y else str(p))
            )
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            sym_x, sym_y = str(p[0]), str(p[1])
            pair_id = f"{sym_x}-{sym_y}"
        elif isinstance(p, str):
            s = p.strip()
            # מפרידים נפוצים בין זוגות
            for sep in ("-", "/", "|", ":", ","):
                if sep in s:
                    a, b = s.split(sep, 1)
                    sym_x, sym_y = a.strip(), b.strip()
                    break
            if sym_x is None or sym_y is None:
                # אם לא זיהינו מפריד – נניח סימבול בודד
                sym_x = sym_y = s
            pair_id = f"{sym_x}-{sym_y}"
        else:
            sym_x = sym_y = str(p)
            pair_id = sym_x

        rows.append(
            {
                "pair_id": str(pair_id),
                "sym_x": sym_x,
                "sym_y": sym_y,
            },
        )

    df = pd.DataFrame(rows)
    return _ensure_pairs_df(df)

def _render_pairs_warnings(pairs_df: pd.DataFrame) -> None:
    """מציג אזהרות איכות על pairs_df אם feature-flag רלוונטי פעיל."""
    if not get_flag("macro_warn_pairs_quality", True):
        return
    msgs = _validate_pairs_df(pairs_df)
    for msg in msgs:
        st.info(f"ℹ️ {msg}")


# ===========================================================================
# Meta stamping (macro_meta) + KPIs עליונים + עזרי זמן/הסבר
# ===========================================================================


def _stamp_result(result: AdjustmentResult, cfg: MacroConfig) -> None:
    """שומר Meta מרוכז על תוצאת המאקרו ב-session_state.

    נשמרים:
    --------
    - ts:        חותמת זמן UTC ISO.
    - apply_mode:מצב יישום (למשל 'hybrid' / 'caps_only' ... מתוך MacroConfig).
    - exposure_multiplier:מקדם חשיפה גלובלי.
    - pairs:     מספר זוגות שיש להם pair_adjustments.
    - included:  כמה זוגות מסומנים כ-include=True.
    - regime_label: תג משטר מאקרו מחושב (risk_on / risk_off / stagflation וכו').
    - mean_score: ציון ממוצע (אם קיים ב-result.meta).

    נשמר תחת:
    - MacroStateKeys.META  (modern)
    - "macro_meta"         (תאימות לאחור)
    """
    try:
        from datetime import datetime as _dt

        # ניסיון ראשון: אם כבר יש label בנוי
        regime_label: Optional[str] = getattr(result, "regime_label", None)

        # ניסיון שני: נבנה label מתוך regime_snapshot (risk_on / growth / inflation)
        if not regime_label and getattr(result, "regime_snapshot", None) is not None:
            snap = getattr(result, "regime_snapshot") or {}
            try:
                r = float(snap.get("risk_on", 0.0))
                g = float(snap.get("growth", 0.0))
                i = float(snap.get("inflation", 0.0))
            except Exception:  # noqa: BLE001
                r, g, i = 0.0, 0.0, 0.0

            if r < -0.5 or (g < -0.4 and i < 0):
                regime_label = "risk_off"
            elif i > 0.4 and g < 0:
                regime_label = "stagflation"
            elif i > 0 and g > 0.2:
                regime_label = "reflation"
            elif g < -0.2 and r <= 0.2:
                regime_label = "slowdown"
            else:
                regime_label = "risk_on" if r > 0 else "neutral"

        meta: MacroMetaPayload = {
            "ts": _dt.utcnow().isoformat(timespec="seconds") + "Z",
            "apply_mode": str(getattr(cfg, "apply_mode", "hybrid")),
            "exposure_multiplier": float(getattr(result, "exposure_multiplier", 1.0)),
            "pairs": int(len(getattr(result, "pair_adjustments", {}) or {})),
            "included": int(
                sum(1 for v in (getattr(result, "filters", {}) or {}).values() if v),
            ),
        }
        if regime_label is not None:
            meta["regime_label"] = str(regime_label)

        meta_from_result = getattr(result, "meta", {}) or {}
        if "mean_score" in meta_from_result:
            try:
                meta["mean_score"] = float(meta_from_result["mean_score"])
            except Exception:  # noqa: BLE001
                pass

        # שמירה ב-session_state לפי החוזה
        state_set(MacroStateKeys.META, meta)
        # תאימות לאחור לשאר הקוד במערכת
        st.session_state["macro_meta"] = dict(meta)
    except Exception:  # noqa: BLE001
        LOGGER.debug("_stamp_result: skip meta save", exc_info=True)


def _explain_regime_label(label: str) -> str:
    """הסבר טקסטואלי קצר ל-regime_label (להצגה מתחת ל-KPIs)."""
    mapping = {
        "risk_on": "סביבה תומכת סיכון — מניות ואסטרטגיות פרו-סיכון בד\"כ נהנות.",
        "risk_off": "סביבת Risk-off — בריחה מסיכון, העדפת מקלטים בטוחים.",
        "stagflation": "אינפלציה גבוהה + צמיחה חלשה — סביבת מאקרו קשה לנכסי סיכון.",
        "reflation": "חזרה לצמיחה עם אינפלציה מתונה — טוב לרוב הנכסים הריאליים.",
        "slowdown": "האטה בצמיחה, עדיין לא משבר — כדאי להיזהר במינוף וחשיפה סקטוריאלית.",
        "neutral": "סביבה נייטרלית — ללא הטיה חזקה לכיוון Risk-on/Risk-off.",
    }
    return mapping.get(
        label,
        "סביבת משטר מאקרו כללית ללא תיוג ספציפי.",
    )


def _warn_if_meta_stale(meta: Mapping[str, Any], max_age_minutes: int = 15) -> None:
    """מזהיר אם ה-meta ישן מדי (למשל טאב פתוח שעות בלי רענון)."""
    try:
        from datetime import datetime as _dt

        ts_str = str(meta.get("ts", ""))
        if not ts_str:
            return
        ts = _dt.fromisoformat(ts_str.replace("Z", ""))
        age_min = (_dt.utcnow() - ts).total_seconds() / 60.0
        if age_min > max_age_minutes:
            st.warning(
                f"התוצאה חישובית ישנה יחסית ({age_min:.1f} דקות) — "
                "שקול לרענן את חישובי המאקרו.",
            )
    except Exception:  # noqa: BLE001
        return


def _render_top_kpis(meta: Mapping[str, Any]) -> None:
    """מציג KPIs עליונים על בסיס meta שנשמר מ-_stamp_result.

    KPIs:
    -----
    1. זמן ריצה (ts)
    2. apply_mode (מצב יישום)
    3. exposure_multiplier (× חשיפה)
    4. זוגות כלולים מתוך כלל הזוגות (included / pairs)
    5. mean_score + Badge איכות ציון (גבוה / בינוני / נמוך)
    """
    if not meta:
        st.info("עדיין לא חושב Macro Adjustment — הרץ חישוב כדי לקבל KPIs.")
        return

    ts = str(meta.get("ts", "N/A"))
    apply_mode = str(meta.get("apply_mode", "hybrid"))
    exposure = float(meta.get("exposure_multiplier", 1.0))
    included = int(meta.get("included", 0))
    pairs = int(meta.get("pairs", 0))
    mean_score = meta.get("mean_score", None)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🕒 זמן ריצה", ts)
    with c2:
        st.metric("מצב יישום", apply_mode)
    with c3:
        st.metric("מקדם חשיפה", f"× {exposure:.2f}")
    with c4:
        st.metric("זוגות כלולים", f"{included}/{pairs}")
    with c5:
        if mean_score is not None:
            try:
                msf = float(mean_score)
                st.metric("ציון ממוצע", f"{msf:.1f}")
                if msf >= 60:
                    color, label = "#16a34a", "גבוה"
                elif msf >= 40:
                    color, label = "#f59e0b", "בינוני"
                else:
                    color, label = "#dc2626", "נמוך"
                st.markdown(
                    "<span style='display:inline-block;padding:4px 10px;"
                    "border-radius:999px;background:{};color:white;font-weight:600;'>"
                    "איכות ציון: {}</span>".format(color, label),
                    unsafe_allow_html=True,
                )
            except Exception:  # noqa: BLE001
                st.metric("ציון ממוצע", "N/A")
        else:
            st.metric("ציון ממוצע", "N/A")

    regime_label = meta.get("regime_label")
    if regime_label:
        st.caption(f"משטר נוכחי: **{regime_label}** — {_explain_regime_label(str(regime_label))}")

    _warn_if_meta_stale(meta)

    # ---- דחיפת מדדי מאקרו ל-Tab Comparison (macro_metrics) ----
    try:
        macro_meta = meta or {}
        mean_score = _safe_float_or_none(macro_meta.get("mean_score"))
        regime_label = str(macro_meta.get("regime_label", "") or "")

        # הופכים regime_label למעין רגישות 0..1
        regime_map = {
            "risk_off": 0.1,
            "stagflation": 0.2,
            "slowdown": 0.3,
            "neutral": 0.5,
            "reflation": 0.7,
            "risk_on": 0.9,
        }
        macro_sensitivity = regime_map.get(regime_label, 0.5)

        macro_score = mean_score if mean_score is not None else 50.0

        push_macro_metrics_to_ctx(
            macro_sensitivity=macro_sensitivity,
            macro_score=macro_score,
            # אין לך כרגע max_dd_60d/vol_60d גלובליים, אז נשאיר None
            max_dd_60d=None,
            vol_60d=None,
        )
    except Exception:
        # לא מפילים את הטאב אם משהו קטן נכשל — פשוט לא נעדכן macro_metrics
        LOGGER.debug("push_macro_metrics_to_ctx failed (non-fatal)", exc_info=True)

def _render_risk_profile_banner() -> None:
    """
    מציג Banner קטן של 'מצב סיכון גלובלי' (risk_profile_from_macro)
    שהגיע מה-dashboard / render_macro_tab (risk_mode).

    זה מאפשר לראות במבט אחד:
      - defensive / stress → מצב הגנתי / משברי
      - normal            → רגיל
      - offensive         → אגרסיבי
    """
    risk_profile = state_get("risk_profile_from_macro", None)
    if not risk_profile:
        return

    rp = str(risk_profile).lower()
    if rp in {"defensive", "stress"}:
        msg = f"מצב סיכון גלובלי: **{risk_profile}** — פרופיל הגנתי / משברי, רצוי להיזהר במינוף וחשיפה."
        try:
            st.error(msg)
        except Exception:
            st.write(msg)
    elif rp in {"offensive", "aggressive"}:
        msg = f"מצב סיכון גלובלי: **{risk_profile}** — פרופיל התקפי, מתאים לתנאים נוחים אך עם סיכון מוגבר."
        try:
            st.warning(msg)
        except Exception:
            st.write(msg)
    else:
        msg = f"מצב סיכון גלובלי: **{risk_profile}** — פרופיל רגיל/נייטרלי."
        try:
            st.info(msg)
        except Exception:
            st.write(msg)

# ===========================================================================
# Live Health: חיווי על מקורות דאטה ומצב Client
# ===========================================================================


def _render_live_health(cfg: MacroConfig) -> None:
    """מציג Badges של בריאות שכבות Live לפי המקורות/הרשאות הנוכחיים.

    בדיקות:
    --------
    - Client:ON    → cfg.use_data_client
    - Live        → cfg.data_client_live
    - IBKR        → האם יש token ב-session_state + מקור ibkr: כלשהו.
    - YF          → האם יש מקור שמתחיל ב-yf:
    - DuckDB      → האם יש מקור שמתחיל ב-duckdb:
    - SQL         → האם יש מקור שמתחיל ב-sql:

    המקורות נלקחים מ:
    - cfg.sources (אם קיים שדה כזה ב-MacroConfig).
    - session_state["macro_sources_overrides"] (שנוספו מתוך הטאב).
    """
    try:
        overrides = state_get("macro_sources_overrides", {}) or {}
        cfg_sources = getattr(cfg, "sources", {}) or {}
        vals: List[str] = []
        vals.extend(str(v) for v in cfg_sources.values())
        vals.extend(str(v) for v in overrides.values())

        def has_prefix(prefix: str) -> bool:
            return any(isinstance(v, str) and v.startswith(prefix) for v in vals)

        ibkr_ok = bool(state_get(keygen(TAB_KEY, "ibkr", "token"))) and has_prefix("ibkr:")
        yf_ok = has_prefix("yf:")
        duck_ok = has_prefix("duckdb:")
        sql_ok = has_prefix("sql:")

        client_on = bool(getattr(cfg, "use_data_client", False))
        client_live = bool(getattr(cfg, "data_client_live", False))

        def badge(txt: str, ok: bool) -> str:
            color = "#16a34a" if ok else "#dc2626"
            suffix = " ✓" if ok else " ✕"
            return (
                "<span style='display:inline-block;margin-right:6px;"
                "padding:2px 8px;border-radius:999px;background:{};"
                "color:white;font-size:12px'>".format(color)
                + txt
                + suffix
                + "</span>"
            )

        html = (
            badge("Client:ON", client_on)
            + badge("Live", client_live)
            + badge("IBKR", ibkr_ok)
            + badge("YF", yf_ok)
            + badge("DuckDB", duck_ok)
            + badge("SQL", sql_ok)
        )
        st.markdown(html, unsafe_allow_html=True)

        if get_flag("macro_show_source_summary", True):
            _render_source_summary(cfg, overrides)
    except Exception:  # noqa: BLE001
        # עדיף לא להפיל את הטאב בגלל אינדיקטור ויזואלי
        LOGGER.debug("_render_live_health failed", exc_info=True)


def _render_source_summary(
    cfg: MacroConfig,
    overrides: Optional[Mapping[str, str]] = None,
) -> None:
    """טבלת Summary קטנה של מקורות דאטה מאקרו (IBKR / YF / DuckDB / SQL)."""
    try:
        sources = getattr(cfg, "sources", {}) or {}
        overrides = overrides or {}
        all_sources: Dict[str, str] = {}
        all_sources.update({str(k): str(v) for k, v in sources.items()})
        all_sources.update({str(k): str(v) for k, v in overrides.items()})

        rows: List[Dict[str, Any]] = []
        for name, uri in all_sources.items():
            if ":" in uri:
                prefix, rest = uri.split(":", 1)
            else:
                prefix, rest = "unknown", uri
            rows.append(
                {
                    "logical_name": name,
                    "kind": prefix,
                    "uri": uri,
                    "detail": rest,
                },
            )
        if not rows:
            st.caption("אין מקורות מאקרו מוגדרים כרגע (cfg.sources ו־overrides ריקים).")
            return
        df = pd.DataFrame(rows)
        df_summary = (
            df.groupby("kind", as_index=False)["logical_name"]
            .count()
            .rename(columns={"logical_name": "count"})
        )
        with st.expander("🔎 סיכום מקורות מאקרו (by kind)", expanded=False):
            st.dataframe(df_summary, use_container_width=True)
            show_full = st.toggle(
                "הצג טבלת מקורות מלאה",
                value=False,
                key=keygen(TAB_KEY, "src", "full"),
            )
            if show_full:
                st.dataframe(df, use_container_width=True)
    except Exception:  # noqa: BLE001
        LOGGER.debug("_render_source_summary failed", exc_info=True)


# ===========================================================================
# Debug Panels (נשתמש בהם בחלק render אם macro_debug=True)
# ===========================================================================


def _render_debug_panels(
    snapshot: Optional[JSONDict] = None,
    overlay: Optional[Dict[str, JSONDict]] = None,
) -> None:
    """מציג Debug Panels אם דגל macro_debug פעיל.

    Parameters
    ----------
    snapshot:
        ה-MacroSnapshot האחרון (אם זמין).
    overlay:
        macro_profile / overlay של משקולות ופעולות לפי זוג (אם זמין).
    """
    if not get_flag("macro_debug", False):
        return

    meta = state_get(MacroStateKeys.META, {})
    with st.expander("🐞 Macro Debug — Meta & Snapshot", expanded=False):
        st.markdown("**Meta (macro_meta):**")
        st.json(meta or {}, expanded=False)

        if snapshot is not None:
            st.markdown("**Macro Snapshot (last computation):**")
            st.json(snapshot, expanded=False)

    if overlay:
        with st.expander("🐞 Macro Debug — Overlay / Profile", expanded=False):
            st.json(overlay, expanded=False)
# ===========================================================================
# סיום חלק 2: עזרי State, Feature Flags, ולידציה, Meta & KPIs, Live Health
# ===========================================================================

# ---------------------------------------------------------------------------
# Base Layer — Workspace controls, Macro data loading, Snapshot
# ---------------------------------------------------------------------------


def _default_workspace_dates() -> Tuple[pd.Timestamp, pd.Timestamp]:
    """טווח ברירת מחדל לשכבת המאקרו: שנה אחורה מהיום (normalized)."""
    today = pd.Timestamp.today().normalize()
    start = today - pd.Timedelta(days=365)
    return start, today


def _workspace_sidebar(namespace: str = TAB_KEY) -> MacroWorkspace:
    """Sidebar גלובלי לשכבת המאקרו (Date Range / Freq / Env).

    זהו ה-"קונטרול טאוור" של כל הטאב:
    - כל מה שקשור לתקופה, תדירות וסביבת עבודה — מוגדר כאן.
    - כל שאר השכבות (Adjustments / DNA / Regimes / Shocks) נשענות עליו.
    """
    with st.sidebar:
        st.subheader("🌍 שכבת בסיס מאקרו — טווח ותצורה")

        # ברירת מחדל: שנה אחורה
        default_start, default_end = _default_workspace_dates()
        date_key = keygen(namespace, "ws", "date_range")

        date_val = st.date_input(
            "טווח תאריכים למאקרו",
            value=(default_start.date(), default_end.date()),
            key=date_key,
            help="הטווח שבו ינותחו הפקטורים המאקרו (Snapshot / Regimes / Backtest).",
        )

        if isinstance(date_val, (list, tuple)) and len(date_val) == 2:
            start_d, end_d = date_val
        else:
            start_d, end_d = default_start.date(), default_end.date()

        start_ts = pd.to_datetime(start_d)
        end_ts = pd.to_datetime(end_d)

        freq = st.selectbox(
            "תדירות מאקרו",
            options=["D", "W", "M"],
            index=0,
            key=keygen(namespace, "ws", "freq"),
            format_func=lambda x: {
                "D": "יומי (D)",
                "W": "שבועי (W)",
                "M": "חודשי (M)",
            }.get(x, str(x)),
            help="קובע כיצד macro_df יירסםפל (D/W/M).",
        )

        env = st.radio(
            "סביבת עבודה",
            options=["backtest", "live"],
            index=0,
            key=keygen(namespace, "ws", "env"),
            format_func=lambda x: "Backtest" if x == "backtest" else "Live (Data Client / IBKR)",
            help="כרגע אינפורמטיבי; בהמשך ישפיע על מקורות/פרמטרים במנוע המאקרו.",
        )

        st.caption(
            "📝 ההגדרות כאן משפיעות על כל שכבות הטאב: Snapshot, Regimes, התאמות לתיק ו-DNA לזוג.",
        )

    ws = MacroWorkspace(
        start_date=start_ts,
        end_date=end_ts,
        freq=freq,  # type: ignore[arg-type]
        env=env,    # type: ignore[arg-type]
    )

    # שמירה ל-session_state למודולים אחרים (למשל tabs אחרים או ה-Executor)
    state_set("macro_workspace", {
        "start": str(ws.start_date),
        "end": str(ws.end_date),
        "freq": ws.freq,
        "env": ws.env,
    })
    return ws


def _init_macro_configs(
    cfg: Optional[MacroConfig],
    factor_cfg: Optional[MacroFactorConfig],
    pair_sens_cfg: Optional[PairMacroSensitivityConfig],
) -> Tuple[MacroConfig, Optional[MacroFactorConfig], Optional[PairMacroSensitivityConfig]]:
    """מאתחל MacroConfig / MacroFactorConfig / PairMacroSensitivityConfig.

    סדר העדיפויות:
    1. אם הועבר אובייקט בפועל לפונקציה — משתמשים בו.
    2. אחרת, אם קיים אובייקט ב-session_state (למשל מה-Config Tab) — נטען ממנו.
    3. אחרת, נייצר קונפיג חדש (ברירת מחדל).

    כך הטאב "מתחבר" אוטומטית להגדרות קיימות במערכת, אבל גם עובד standalone.
    """
    # MacroConfig
    if cfg is None:
        cfg_from_state = state_get(MacroStateKeys.CFG)
        if isinstance(cfg_from_state, MacroConfig):
            cfg = cfg_from_state
        else:
            cfg = MacroConfig()  # type: ignore[call-arg]

    # MacroFactorConfig
    if factor_cfg is None:
        cfg_f_from_state = state_get(MacroStateKeys.FACTOR_CFG)
        if isinstance(cfg_f_from_state, MacroFactorConfig):
            factor_cfg = cfg_f_from_state
        else:
            try:
                factor_cfg = MacroFactorConfig()  # type: ignore[call-arg]
            except Exception:  # noqa: BLE001
                factor_cfg = None

    # PairMacroSensitivityConfig
    if pair_sens_cfg is None:
        cfg_sens_state = state_get("macro_pair_sens_cfg")
        if isinstance(cfg_sens_state, PairMacroSensitivityConfig):
            pair_sens_cfg = cfg_sens_state
        else:
            try:
                pair_sens_cfg = PairMacroSensitivityConfig()  # type: ignore[call-arg]
            except Exception:  # noqa: BLE001
                pair_sens_cfg = None

    # נשמור את הקונפיגים גם ב-session_state לטובת טאב/מודולים אחרים
    state_set(MacroStateKeys.CFG, cfg)
    if factor_cfg is not None:
        state_set(MacroStateKeys.FACTOR_CFG, factor_cfg)
    if pair_sens_cfg is not None:
        state_set("macro_pair_sens_cfg", pair_sens_cfg)

    return cfg, factor_cfg, pair_sens_cfg


def _normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """מוודא שה-index של df הוא DatetimeIndex ממוין (ל-resample נקי)."""
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _slice_to_workspace(df: pd.DataFrame, ws: MacroWorkspace) -> pd.DataFrame:
    """גוזר את macro_df לטווח ה-Workspace (start/end)."""
    if df.empty:
        return df
    df = _normalize_datetime_index(df)
    if ws.start_date is not None:
        df = df[df.index >= ws.start_date]
    if ws.end_date is not None:
        df = df[df.index <= ws.end_date]
    return df


def _resample_macro_df(df: pd.DataFrame, freq: MacroFreq) -> pd.DataFrame:
    """ריסמפל של macro_df לפי תדירות המאקרו שנבחרה (D/W/M)."""
    df = _normalize_datetime_index(df)
    if df.empty:
        return df

    if freq == "D":
        return df

    rule = "W-FRI" if freq == "W" else "M"
    try:
        df_res = df.resample(rule).last()
    except Exception:  # noqa: BLE001
        # fallback פשוט אם resample לא זמין (למשל index לא רציף)
        return df
    return df_res.dropna(how="all")


def _maybe_align_macro_to_universe(
    macro_df: pd.DataFrame,
    *,
    prices_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """מנסה ליישר את macro_df ליקום המחירים/זמנים של המערכת.

    לוגיקה:
    -------
    1. אם נשלח prices_df → נשתמש ב-index שלו.
    2. אחרת, ננסה למצוא prices_wide / returns_wide ב-session_state.
    3. אם אין כלום — מחזירים את macro_df כמו שהוא.

    המטרה: macro_df מיושר לחותמת הזמן של המערכת (למשל trading days אמיתיים).
    """
    if macro_df is None or macro_df.empty:
        return macro_df

    macro_df = _normalize_datetime_index(macro_df)

    if prices_df is None:
        prices_df = state_get("prices_wide")
        if not isinstance(prices_df, pd.DataFrame) or prices_df.empty:
            prices_df = state_get("returns_wide")
            if not isinstance(prices_df, pd.DataFrame) or prices_df.empty:
                return macro_df

    prices_df = _normalize_datetime_index(prices_df)
    common_index = macro_df.index.intersection(prices_df.index)
    if len(common_index) == 0:
        # אם אין חיתוך — עדיף לא "לשבור" את המאקרו, רק נתריע ברמת לוג
        LOGGER.debug("no common index between macro_df and prices universe — using macro_df as-is")
        return macro_df
    return macro_df.loc[common_index]


def _describe_macro_df(macro_df: Optional[pd.DataFrame]) -> None:
    """תצוגת מידע מהירה על macro_df לצרכי UX/Debug."""
    if macro_df is None or macro_df.empty:
        st.caption("macro_df: (ריק) — אין נתוני מאקרו תקינים לתקופה/תדירות שנבחרה.")
        return

    n_rows, n_cols = macro_df.shape
    start, end = macro_df.index.min(), macro_df.index.max()
    st.caption(
        f"macro_df: **{n_rows:,}** תצפיות × **{n_cols:,}** פקטורים | "
        f"טווח: **{start.date()} → {end.date()}**",
    )


def _load_macro_data_for_workspace(
    state: MacroTabState,
    *,
    pairs_df: Optional[pd.DataFrame] = None,
) -> MacroTabState:
    """טוען את מקורות המאקרו (Bundle ישן + פקטורים חדשים) לפי ה-Workspace.

    לוגיקה:
    -------
    1. MacroBundle (ישן):
        - load_macro_bundle(cfg) → MacroBundle
        - נשמר ב-session_state תחת MacroStateKeys.BUNDLE.
    2. macro_factors (חדש):
        - ננסה קריאות שונות ל-load_macro_factors:
            * load_macro_factors(cfg, factor_cfg=factor_cfg)
            * load_macro_factors(cfg)
            * load_macro_factors()
        - add_derived_factors(df, factor_cfg) → העשרת פקטורים (אם factor_cfg קיים).
        - slice_to_workspace לפי start/end.
        - יישור לאינדקס של prices_wide (אם קיים).
        - resample לפי state.workspace.freq.
        - נשמר כ-state.macro_df + state.macro_factors_df + MacroStateKeys.MACRO_DF.
    """
    cfg = state.macro_cfg
    ws = state.workspace

    # 1) Bundle ישן — תמיד ננסה לטעון (עם fallback ל-Bundle ריק)
    bundle = state.bundle
    if bundle is None:
        try:
            bundle = load_macro_bundle(cfg)
        except Exception as e:  # noqa: BLE001
            LOGGER.error("כשל בטעינת MacroBundle: %s", e)
            try:
                bundle = MacroBundle({})  # type: ignore[call-arg]
            except Exception:
                bundle = None
    state.bundle = bundle
    state_set(MacroStateKeys.BUNDLE, bundle)

    # 2) macro_factors החדש
    macro_df: Optional[pd.DataFrame] = None
    # ננסה כמה חתימות נפוצות כדי להיות עמידים לשינויים במודול המאקרו
    if macro_df is None:
        try:
            macro_df = load_macro_factors(cfg, state.factor_cfg)  # type: ignore[call-arg]
        except TypeError:
            try:
                macro_df = load_macro_factors(cfg)  # type: ignore[call-arg]
            except Exception:  # noqa: BLE001
                macro_df = None
        except Exception:
            macro_df = None

    if macro_df is None:
        try:
            macro_df = load_macro_factors()  # type: ignore[call-arg]
        except Exception:  # noqa: BLE001
            macro_df = None

    if macro_df is not None:
        # enrich / derived factors
        try:
            if state.factor_cfg is not None:
                macro_df = add_derived_factors(macro_df, state.factor_cfg)  # type: ignore[arg-type]
            else:
                macro_df = add_derived_factors(macro_df)  # type: ignore[call-arg]
        except TypeError:
            # ייתכן שהפונקציה לא מקבלת factor_cfg — נתעלם בשקט
            LOGGER.debug("add_derived_factors called without factor_cfg")
        except Exception as e:  # noqa: BLE001
            LOGGER.debug("add_derived_factors failed (non-fatal): %s", e)

        macro_df = _slice_to_workspace(macro_df, ws)
        macro_df = _maybe_align_macro_to_universe(
            macro_df,
            prices_df=state_get("prices_wide") if pairs_df is None else state_get("prices_wide"),
        )
        macro_df = _resample_macro_df(macro_df, ws.freq)

        state.macro_df = macro_df
        state.macro_factors_df = macro_df
        state_set(MacroStateKeys.MACRO_DF, macro_df)
    else:
        state.macro_df = None
        state.macro_factors_df = None

    return state


def _extract_macro_key_metrics(snapshot: JSONDict) -> List[Dict[str, Any]]:
    """מוציא 'Key Metrics' מתוך snapshot לפי שמות פקטורים מקובלים.

    מחפש ערכים ב:
    -------------
    - last_values
    - zscores

    ומחזיר רשימה של:
    { "label": ..., "value": ..., "z": ..., "key": ... }
    """
    if not snapshot:
        return []

    last_vals = snapshot.get("last_values", {}) or {}
    zscores = snapshot.get("zscores", {}) or {}

    # פקטורים טיפוסיים — אפשר להרחיב בהמשך
    canonical_order = [
        ("policy_rate", "ריבית מדיניות"),
        ("rate_short", "ריבית קצרה"),
        ("rate_long", "ריבית ארוכה"),
        ("slope_10y_3m", "עקום 10Y-3M"),
        ("slope_10y_2y", "עקום 10Y-2Y"),
        ("vix", "VIX (שוק אופציות)"),
        ("vix_term_1_0", "VIX Term (1-0)"),
        ("credit_spread", "Credit Spread"),
        ("dxy", "מדד דולר (DXY)"),
        ("risk_on_proxy", "Risk-On Proxy"),
    ]

    key_metrics: List[Dict[str, Any]] = []
    for key, label in canonical_order:
        if key in last_vals or key in zscores:
            val = last_vals.get(key, None)
            z = zscores.get(key, None)
            try:
                val_f = float(val) if val is not None else None
            except Exception:
                val_f = None
            try:
                z_f = float(z) if z is not None else None
            except Exception:
                z_f = None
            key_metrics.append(
                {
                    "key": key,
                    "label": label,
                    "value": val_f,
                    "z": z_f,
                },
            )
    return key_metrics


def _render_macro_snapshot_section(state: MacroTabState) -> None:
    """מציג את 'מצב המאקרו הנוכחי' (Snapshot) + כפתור הורדה + Key Metrics.

    משתמש ב:
    --------
    - build_macro_snapshot(macro_df, factor_cfg/ cfg)
    - summarize_macro_snapshot(snapshot)

    מציג:
    -----
    - משפט סיכום קצר.
    - KPIs לפי קבוצות משטר (rates / curve / vol / credit / fx / risk).
    - "Key Macro Metrics" עם ערכים אחרונים ו-Z-score.
    - Expander עם snapshot מלא.
    - כפתור הורדה כ-JSON.
    - Debug Panels אם macro_debug=True.
    """
    st.subheader("📊 מצב מאקרו נוכחי")

    ws = state.workspace
    macro_df = state.macro_df

    if ws.start_date is not None and ws.end_date is not None:
        st.caption(
            f"טווח ניתוח מאקרו: **{ws.start_date.date()} → {ws.end_date.date()}** | "
            f"תדירות: **{ws.freq}** | סביבת עבודה: **{ws.env}**",
        )

    _describe_macro_df(macro_df)

    if macro_df is None or macro_df.empty:
        st.info(
            "לא נמצאו נתוני מאקרו מהמודול החדש (macro_factors) לתקופה הנבחרת.\n"
            "עדיין ניתן להשתמש ב-MacroBundle הישן לתיקון חשיפה דרך מנוע המאקרו.",
        )
        return

    # בניית snapshot
    snapshot: Optional[JSONDict] = None
    try:
        try:
            snapshot = build_macro_snapshot(macro_df, state.factor_cfg)  # type: ignore[arg-type]
        except TypeError:
            # חתימה חלופית: אולי נדרש רק macro_df
            snapshot = build_macro_snapshot(macro_df)  # type: ignore[call-arg]
    except Exception as e:  # noqa: BLE001
        LOGGER.error("build_macro_snapshot failed: %s", e)
        st.warning("כשל בבניית Macro Snapshot — מוצג רק מידע בסיסי על המאקרו.")
        snapshot = None

    state.macro_snapshot = snapshot
    if snapshot is not None:
        state_set("macro_snapshot", snapshot)

    # סיכום טקסטואלי
    summary_text = ""
    if snapshot is not None:
        try:
            summary_text = summarize_macro_snapshot(snapshot)
        except Exception as e:  # noqa: BLE001
            LOGGER.debug("summarize_macro_snapshot failed: %s", e)

    if summary_text:
        st.markdown(f"**סיכום מצב מאקרו:** {summary_text}")
    else:
        st.markdown("**סיכום מצב מאקרו:** (לא זמין — ייתכן שחסרים פקטורים או snapshot חלקי)")

    # KPIs לפי קבוצות (rates / curve / vol / credit / fx / risk)
    group_regimes = (snapshot or {}).get("group_regimes", {}) if snapshot else {}
    if isinstance(group_regimes, dict) and group_regimes:
        # ננסה להציג בסדר קבוע, ואם יש עוד קבוצות נוספות — נוסיף בסוף
        preferred_order = ["rates", "curve", "vol", "credit", "fx", "risk"]
        items_ordered: List[Tuple[str, Any]] = []
        seen = set()
        for k in preferred_order:
            if k in group_regimes:
                items_ordered.append((k, group_regimes[k]))
                seen.add(k)
        for k, v in group_regimes.items():
            if k not in seen:
                items_ordered.append((k, v))

        n_cols = min(len(items_ordered), 6) or 1
        cols = st.columns(n_cols)
        for idx, (group_name, label) in enumerate(items_ordered):
            col = cols[idx % n_cols]
            with col:
                st.metric(str(group_name), str(label))

    # Key Macro Metrics (last_values + z-scores) — 2 שורות של KPIs
    if snapshot is not None:
        metrics = _extract_macro_key_metrics(snapshot)
        if metrics:
            st.markdown("**Key Macro Metrics (Last Value + Z):**")
            # נחלק לבלוקים של עד 3 KPIs בשורה
            chunk_size = 3
            for i in range(0, len(metrics), chunk_size):
                row = metrics[i : i + chunk_size]
                cols = st.columns(len(row))
                for c, m in zip(cols, row):
                    val_str = "N/A" if m["value"] is None else f"{m['value']:.2f}"
                    z_str = "" if m["z"] is None else f" (Z={m['z']:.2f})"
                    with c:
                        st.metric(m["label"], f"{val_str}{z_str}")

    # ערכים אחרונים + Z-scores ב-expander
    with st.expander("🔍 פירוט Snapshot (last values / z-scores / regimes)", expanded=False):
        if snapshot is not None:
            last_vals = snapshot.get("last_values", {})
            zscores = snapshot.get("zscores", {})
            regimes = snapshot.get("regimes", {})

            if last_vals or zscores or regimes:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Last Values**")
                    st.json(last_vals or {}, expanded=False)
                with c2:
                    st.markdown("**Z-scores**")
                    st.json(zscores or {}, expanded=False)
                with c3:
                    st.markdown("**Regimes (by factor)**")
                    st.json(regimes or {}, expanded=False)
                st.markdown("---")

            st.markdown("**Snapshot מלא (JSON):**")
            st.json(snapshot, expanded=False)
        else:
            st.info("אין Snapshot תקין להצגה.")

        # כפתור הורדה כ-JSON
        if snapshot is not None:
            snap_payload = {
                "workspace": {
                    "start": str(ws.start_date) if ws.start_date is not None else None,
                    "end": str(ws.end_date) if ws.end_date is not None else None,
                    "freq": ws.freq,
                    "env": ws.env,
                },
                "snapshot": snapshot,
            }
            st.download_button(
                label="⬇️ הורד Macro Snapshot (JSON)",
                data=json.dumps(snap_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="macro_snapshot.json",
                mime="application/json",
                key=keygen(TAB_KEY, "snapshot", "dl"),
            )

    # Debug Panels אם הדגל פעיל
    _render_debug_panels(snapshot=snapshot, overlay=state.macro_profile or {})

# ---------------------------------------------------------------------------
# Sidebar Controls — Presets, Feature Engineering, Risk, Regime Model, IBKR
# ---------------------------------------------------------------------------


# ===========================
# Presets — שמירה/טעינה
# ===========================


def _collect_ui_cfg_from_state(namespace: str = TAB_KEY) -> Dict[str, Any]:
    """אוסף את ערכי ה־UI מתוך session_state כדי לשמור Preset.

    לא כולל את ה-Workspace (טווח תאריכים/תדירות/סביבה) — זה נשמר בנפרד.
    מה כן נאסף:
    - פרמטרי תאימות/מודל:
        macro_model, lookback, ewma_lambda, min_regime
    - שליטת סיכון גלובלית:
        target_vol, max_pair_weight, sector_cap
    - ציון Macro Fit:
        score_filter_threshold, score_gain_floor, score_weight_gain, caps_mode
    - Paper mode:
        paper_mode
    """
    keys = {
        "model": keygen(namespace, "macro_model"),
        "lookback": keygen(namespace, "lookback"),
        "ewma_lambda": keygen(namespace, "ewma"),
        "min_regime": keygen(namespace, "min_regime"),
        "target_vol": keygen(namespace, "target_vol"),
        "max_pair_weight": keygen(namespace, "max_pair_w"),
        "sector_cap": keygen(namespace, "sector_cap"),
        "score_filter_threshold": keygen(namespace, "score_thr"),
        "score_gain_floor": keygen(namespace, "score_floor"),
        "score_weight_gain": keygen(namespace, "score_gain"),
        "caps_mode": keygen(namespace, "caps_mode"),
        "paper_mode": keygen(namespace, "paper"),
    }
    out: Dict[str, Any] = {}
    for name, k in keys.items():
        out[name] = state_get(k)
    return out

def _get_builtin_macro_presets() -> Dict[str, Dict[str, Any]]:
    """
    Presets מובנים לטאב המאקרו.

    המפתחות תואמים ל-UI cfg שנאסף ב-_collect_ui_cfg_from_state /
    ומיושם דרך _apply_ui_preset:
        - model
        - lookback
        - ewma_lambda
        - min_regime
        - target_vol
        - max_pair_weight
        - sector_cap
        - score_filter_threshold
        - score_gain_floor
        - score_weight_gain
        - caps_mode
        - paper_mode
    """
    return {
        "base": {
            "model": "HMM",
            "lookback": 365,
            "ewma_lambda": 0.94,
            "min_regime": 15,
            "target_vol": 15.0,
            "max_pair_weight": 5.0,
            "sector_cap": 20.0,
            "score_filter_threshold": 30.0,
            "score_gain_floor": 60.0,
            "score_weight_gain": 0.15,
            "caps_mode": "hint",
            "paper_mode": True,
        },
        "risk_off_defensive": {
            "model": "HMM",
            "lookback": 365,
            "ewma_lambda": 0.97,
            "min_regime": 20,
            "target_vol": 10.0,
            "max_pair_weight": 3.0,
            "sector_cap": 15.0,
            "score_filter_threshold": 40.0,
            "score_gain_floor": 55.0,
            "score_weight_gain": 0.10,
            "caps_mode": "clip",
            "paper_mode": True,
        },
        "risk_on_aggressive": {
            "model": "k-means",
            "lookback": 240,
            "ewma_lambda": 0.90,
            "min_regime": 10,
            "target_vol": 25.0,
            "max_pair_weight": 8.0,
            "sector_cap": 30.0,
            "score_filter_threshold": 20.0,
            "score_gain_floor": 70.0,
            "score_weight_gain": 0.20,
            "caps_mode": "hint",
            "paper_mode": False,
        },
        "balanced_regime": {
            "model": "HSMM",
            "lookback": 360,
            "ewma_lambda": 0.95,
            "min_regime": 15,
            "target_vol": 17.0,
            "max_pair_weight": 6.0,
            "sector_cap": 25.0,
            "score_filter_threshold": 30.0,
            "score_gain_floor": 65.0,
            "score_weight_gain": 0.15,
            "caps_mode": "clip",
            "paper_mode": True,
        },
        "macro_research_deep": {
            "model": "XGBoost",
            "lookback": 720,
            "ewma_lambda": 0.97,
            "min_regime": 20,
            "target_vol": 12.0,
            "max_pair_weight": 4.0,
            "sector_cap": 20.0,
            "score_filter_threshold": 25.0,
            "score_gain_floor": 60.0,
            "score_weight_gain": 0.12,
            "caps_mode": "hint",
            "paper_mode": True,
        },
    }

def _apply_ui_preset(preset: Dict[str, Any], namespace: str = TAB_KEY) -> None:
    """מיישם Preset ערכים ל־session_state כך שהווידג'טים יתעדכנו.

    Notes
    -----
    - לא מבצע שום חישוב — רק מעדכן session_state.
    - render() אחרי זה מפעיל את החישוב מחדש (st.rerun).
    """
    mapping = {
        "model": keygen(namespace, "macro_model"),
        "lookback": keygen(namespace, "lookback"),
        "ewma_lambda": keygen(namespace, "ewma"),
        "min_regime": keygen(namespace, "min_regime"),
        "target_vol": keygen(namespace, "target_vol"),
        "max_pair_weight": keygen(namespace, "max_pair_w"),
        "sector_cap": keygen(namespace, "sector_cap"),
        "score_filter_threshold": keygen(namespace, "score_thr"),
        "score_gain_floor": keygen(namespace, "score_floor"),
        "score_weight_gain": keygen(namespace, "score_gain"),
        "caps_mode": keygen(namespace, "caps_mode"),
        "paper_mode": keygen(namespace, "paper"),
    }
    for name, val in preset.items():
        if name in mapping:
            state_set(mapping[name], val)


def preset_ui_controls(namespace: str = TAB_KEY) -> None:
    """UI לטעינת/שמירת פרופיל פרמטרים של הטאב (Preset).

    עכשיו כולל גם:
    ---------------
    - טעינה אוטומטית של Preset מובנה לפי macro_preset_selected
      שמגיע מה-dashboard / render_macro_tab.
    - רשימת Presets מובנים (base / risk_off_defensive / risk_on_aggressive /
      balanced_regime / macro_research_deep) עם כפתור טעינה מה-UI.
    - עדיין תומך בשמירה/טעינה של Preset כ-JSON כמו קודם.
    """
    builtin_presets = _get_builtin_macro_presets()

    with st.sidebar.expander("💾 Presets (שמירה/טעינה)", expanded=False):
        # --- 1) טעינה אוטומטית לפי macro_preset_selected מה-router/dashboard ---
        macro_preset_selected = state_get("macro_preset_selected", None)
        applied_name_key = keygen(namespace, "preset", "applied_name")
        applied_name = state_get(applied_name_key, None)

        if (
            isinstance(macro_preset_selected, str)
            and macro_preset_selected in builtin_presets
            and applied_name != macro_preset_selected
        ):
            # נטען Preset מובנה שנשלח מה-dashboard (nav_payload.macro_preset)
            preset_data = builtin_presets[macro_preset_selected]
            _apply_ui_preset(preset_data, namespace)
            state_set(applied_name_key, macro_preset_selected)
            st.info(
                f"נטען Preset מאקרו מהדשבורד: **{macro_preset_selected}**",
                icon="✅" if hasattr(st, "info") else None,
            )

        # --- 2) Presets מובנים לבחירה ידנית מה-UI ---
        if builtin_presets:
            st.markdown("**Presets מובנים (Macro):**")
            preset_names = list(builtin_presets.keys())
            select_label = "בחר Preset מובנה"
            chosen = st.selectbox(
                select_label,
                options=["<None>"] + preset_names,
                index=0,
                key=keygen(namespace, "preset", "builtin_select"),
            )
            if chosen != "<None>":
                if st.button(
                    "טען Preset מובנה",
                    key=keygen(namespace, "preset", "load_builtin"),
                ):
                    _apply_ui_preset(builtin_presets[chosen], namespace)
                    state_set("macro_preset_selected", chosen)
                    state_set(applied_name_key, chosen)
                    st.success(f"Preset מובנה **{chosen}** נטען והוחל על ה-UI.")

        st.markdown("---")

        # --- 3) שמירת הפרופיל הנוכחי כ-JSON (כמו קודם) ---
        if st.button("שמור פרופיל נוכחי", key=keygen(namespace, "preset", "save")):
            cfg = _collect_ui_cfg_from_state(namespace)
            st.download_button(
                label="⬇️ הורד Preset (JSON)",
                data=json.dumps(cfg, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="macro_preset.json",
                mime="application/json",
                key=keygen(namespace, "preset", "dl"),
            )

        # --- 4) טעינת Preset חיצוני מ-JSON (כמו קודם) ---
        up = st.file_uploader(
            "טען Preset (JSON)",
            type=["json"],
            key=keygen(namespace, "preset", "upl"),
        )
        if up is not None:
            try:
                data = json.loads(up.read().decode("utf-8"))
                if isinstance(data, dict):
                    _apply_ui_preset(data, namespace)
                    st.success("Preset נטען מקובץ JSON — מרענן את הטאב…")
                    # נעדכן גם את applied_name כדי שלא ידרס ע"י macro_preset_selected
                    state_set(applied_name_key, "custom_from_file")
                    st.rerun()
                else:
                    st.warning("קובץ Preset לא תקין (לא אובייקט JSON).")
            except Exception as e:  # noqa: BLE001
                st.error(f"טעינת Preset נכשלה: {e}")

# ===========================
# Feature Engineering לפקטורי מאקרו
# ===========================


def _parse_int_csv(s: str) -> List[int]:
    s = (s or "").strip()
    return [int(x.strip()) for x in s.split(",") if x.strip()] if s else []


def _feature_params_ui(namespace: str = TAB_KEY) -> Dict[str, Any]:
    """UI מתקדם לפיצ'רים על פקטורי מאקרו; מוחזר למנוע דרך cfg['feature_params'].

    פיצ'רים נתמכים:
    ----------------
    - Z-scores (חלון + winsorization)
    - Momentum windows (על רמות / תשואות)
    - Volatility windows
    - Rolling means
    - EWMA spans
    - Rank windows
    - Lags
    - YoY (שינוי שנה אחורה)
    - Seasonal dummies (חודשים/רבעונים)
    - Standardization (z-score כולל)
    - Prefix לשמות פיצ'רים
    - NaN policy (ffill / bfill / zero / drop)

    בנוסף:
    - include_raw_factors: האם להשאיר גם את הפקטורים המקוריים.
    - drop_low_quality: אפשרות לנקות פקטורים עם איכות סטטיסטית נמוכה (לשימוש פנימי במנוע).
    """
    with st.sidebar.expander("🧪 Feature engineering (מאקרו)", expanded=False):
        st.caption("פיצ'רים מחושבים על אינדיקטורי מאקרו (ולא על ספרדי זוגות).")

        z_window = st.number_input(
            "Z window (ימים)",
            min_value=5,
            max_value=730,
            value=int(state_get(keygen(namespace, "feat", "zwin"), 60) or 60),
            step=5,
            key=keygen(namespace, "feat", "zwin"),
        )
        z_winsor = st.number_input(
            "Z winsor (סטיות תקן)",
            min_value=0.5,
            max_value=10.0,
            value=float(state_get(keygen(namespace, "feat", "zw"), 3.0) or 3.0),
            step=0.5,
            key=keygen(namespace, "feat", "zw"),
        )

        mom_windows = st.text_input(
            "Momentum windows (csv)",
            value=str(state_get(keygen(namespace, "feat", "mom"), "20,60")),
            key=keygen(namespace, "feat", "mom"),
            help="למשל: 20,60,120",
        )
        vol_windows = st.text_input(
            "Vol windows (csv)",
            value=str(state_get(keygen(namespace, "feat", "vol"), "20")),
            key=keygen(namespace, "feat", "vol"),
        )
        mean_windows = st.text_input(
            "Rolling mean windows (csv)",
            value=str(state_get(keygen(namespace, "feat", "mean"), "")),
            key=keygen(namespace, "feat", "mean"),
        )
        ewma_spans = st.text_input(
            "EWMA spans (csv)",
            value=str(state_get(keygen(namespace, "feat", "ewma"), "")),
            key=keygen(namespace, "feat", "ewma"),
        )
        rank_windows = st.text_input(
            "Rank windows (csv)",
            value=str(state_get(keygen(namespace, "feat", "rank"), "")),
            key=keygen(namespace, "feat", "rank"),
        )
        lags = st.text_input(
            "Lags (csv)",
            value=str(state_get(keygen(namespace, "feat", "lags"), "1,5,20")),
            key=keygen(namespace, "feat", "lags"),
        )

        yoy_window = st.number_input(
            "YoY window (ימים)",
            min_value=60,
            max_value=1260,
            value=int(state_get(keygen(namespace, "feat", "yoy"), 252) or 252),
            step=12,
            key=keygen(namespace, "feat", "yoy"),
        )

        vol_on = st.selectbox(
            "Vol on",
            options=["returns", "levels"],
            index=0,
            key=keygen(namespace, "feat", "volon"),
        )
        mean_on = st.selectbox(
            "Mean on",
            options=["levels", "returns"],
            index=0,
            key=keygen(namespace, "feat", "meanon"),
        )
        ewma_on = st.selectbox(
            "EWMA on",
            options=["levels", "returns"],
            index=0,
            key=keygen(namespace, "feat", "ewmaon"),
        )

        seasonal_dummies = st.checkbox(
            "Seasonal dummies (חודשים/רבעונים)",
            value=bool(state_get(keygen(namespace, "feat", "season"), False)),
            key=keygen(namespace, "feat", "season"),
        )
        quality_window = st.number_input(
            "Quality window (אופציונלי)",
            min_value=0,
            max_value=365,
            value=int(state_get(keygen(namespace, "feat", "qual"), 60) or 60),
            step=5,
            key=keygen(namespace, "feat", "qual"),
            help="0=כבוי; >0 יאפשר מדידת איכות לפקטורים לאורך חלון זה.",
        )
        standardize = st.checkbox(
            "Standardize features (z-score כולל)",
            value=bool(state_get(keygen(namespace, "feat", "std"), False)),
            key=keygen(namespace, "feat", "std"),
        )
        include_raw = st.checkbox(
            "השאר גם פקטורים מקוריים (raw)",
            value=bool(state_get(keygen(namespace, "feat", "raw"), True)),
            key=keygen(namespace, "feat", "raw"),
        )
        drop_low_quality = st.checkbox(
            "נקה פקטורים עם איכות סטטיסטית נמוכה",
            value=bool(state_get(keygen(namespace, "feat", "drop_lowq"), False)),
            key=keygen(namespace, "feat", "drop_lowq"),
        )

        prefix = st.text_input(
            "Prefix לשמות פיצ'רים",
            value=str(state_get(keygen(namespace, "feat", "prefix"), "macro_")),
            key=keygen(namespace, "feat", "prefix"),
        )
        dropna_strategy = st.selectbox(
            "NaN policy",
            options=["ffill", "bfill", "zero", "drop"],
            index=0,
            key=keygen(namespace, "feat", "nan"),
        )

    params = {
        "z_window": int(z_window),
        "z_winsor": float(z_winsor),
        "mom_windows": _parse_int_csv(mom_windows),
        "vol_windows": _parse_int_csv(vol_windows),
        "rolling_mean_windows": _parse_int_csv(mean_windows),
        "ewma_spans": _parse_int_csv(ewma_spans),
        "rank_windows": _parse_int_csv(rank_windows),
        "lags": _parse_int_csv(lags),
        "yoy_window": int(yoy_window),
        "vol_on": vol_on,
        "mean_on": mean_on,
        "ewma_on": ewma_on,
        "seasonal_dummies": bool(seasonal_dummies),
        "quality_window": int(quality_window),
        "standardize": bool(standardize),
        "include_raw_factors": bool(include_raw),
        "drop_low_quality": bool(drop_low_quality),
        "prefix": prefix,
        "dropna_strategy": dropna_strategy,
    }
    # נשמור גם עותק אחרון לצרכי Debug / השוואה
    state_set(keygen(namespace, "feat", "last_params"), dict(params))
    return params


# ===========================
# Risk Controls (sizing & caps)
# ===========================


def _parse_caps_text(s: str) -> Dict[str, float]:
    """ממיר טקסט בסגנון 'Tech:0.25, Energy:0.2' למילון {שם: cap}."""
    out: Dict[str, float] = {}
    s = (s or "").strip()
    if not s:
        return out
    for part in s.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                out[str(k).strip()] = float(v.strip())
            except Exception:  # noqa: BLE001
                continue
    return out


def _macro_risk_controls_ui(namespace: str = TAB_KEY) -> Dict[str, Any]:
    """UI לשכבת סיכון גלובלית: sizing / covariance / caps + Risk Presets."""
    with st.sidebar.expander("🛡️ Risk Controls (sizing & caps)", expanded=False):
        risk_sizing = st.selectbox(
            "Sizing method",
            ["invvol", "erc"],
            index=0,
            key=keygen(namespace, "risk", "sizing"),
            help="invvol = 1/σ; erc = Equal Risk Contribution.",
        )
        cov_window = st.number_input(
            "Cov window (ימים)",
            min_value=20,
            max_value=252,
            value=int(state_get(keygen(namespace, "risk", "covwin"), 60) or 60),
            step=5,
            key=keygen(namespace, "risk", "covwin"),
        )
        cov_shrink = st.number_input(
            "Cov shrink (0..1)",
            min_value=0.0,
            max_value=1.0,
            value=float(state_get(keygen(namespace, "risk", "shrink"), 0.1) or 0.1),
            step=0.05,
            key=keygen(namespace, "risk", "shrink"),
        )

        st.caption("Caps בפורמט: Tech:0.25, Energy:0.2 … (ערכים בין 0 ל-1)")

        sector_caps_txt = st.text_input(
            "Sector caps",
            value=str(state_get(keygen(namespace, "risk", "sectorcaps"), "")),
            key=keygen(namespace, "risk", "sectorcaps"),
        )
        region_caps_txt = st.text_input(
            "Region caps",
            value=str(state_get(keygen(namespace, "risk", "regioncaps"), "")),
            key=keygen(namespace, "risk", "regioncaps"),
        )
        currency_caps_txt = st.text_input(
            "Currency caps",
            value=str(state_get(keygen(namespace, "risk", "currencycaps"), "")),
            key=keygen(namespace, "risk", "currencycaps"),
        )

        # שמירת קפס דיקטיים ב-session_state (כמו בגרסה הישנה)
        sector_caps_dict = _parse_caps_text(sector_caps_txt)
        region_caps_dict = _parse_caps_text(region_caps_txt)
        currency_caps_dict = _parse_caps_text(currency_caps_txt)

        state_set(keygen(namespace, "risk", "sectorcaps_dict"), sector_caps_dict)
        state_set(keygen(namespace, "risk", "regioncaps_dict"), region_caps_dict)
        state_set(keygen(namespace, "risk", "currencycaps_dict"), currency_caps_dict)

        st.divider()
        # Risk Presets (save/load)
        risk_preset = {
            "risk_sizing": risk_sizing,
            "cov_window": int(cov_window),
            "cov_shrink": float(cov_shrink),
            "sector_caps": sector_caps_txt,
            "region_caps": region_caps_txt,
            "currency_caps": currency_caps_txt,
        }
        colp1, colp2 = st.columns(2)
        with colp1:
            st.download_button(
                label="⬇️ הורד Risk Preset (JSON)",
                data=json.dumps(risk_preset, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="macro_risk_preset.json",
                mime="application/json",
                key=keygen(namespace, "risk", "dl"),
            )
        with colp2:
            f_risk = st.file_uploader(
                "טען Risk Preset (JSON)",
                type=["json"],
                key=keygen(namespace, "risk", "upl"),
            )
            if f_risk is not None:
                try:
                    data = json.loads(f_risk.read().decode("utf-8"))
                    if isinstance(data, dict):
                        if "risk_sizing" in data:
                            state_set(keygen(namespace, "risk", "sizing"), str(data["risk_sizing"]))
                        if "cov_window" in data:
                            state_set(keygen(namespace, "risk", "covwin"), int(data["cov_window"]))
                        if "cov_shrink" in data:
                            state_set(keygen(namespace, "risk", "shrink"), float(data["cov_shrink"]))
                        if "sector_caps" in data:
                            state_set(keygen(namespace, "risk", "sectorcaps"), str(data["sector_caps"]))
                        if "region_caps" in data:
                            state_set(keygen(namespace, "risk", "regioncaps"), str(data["region_caps"]))
                        if "currency_caps" in data:
                            state_set(keygen(namespace, "risk", "currencycaps"), str(data["currency_caps"]))
                        st.success("Risk Preset נטען — מרענן את הטאב…")
                        st.rerun()
                    else:
                        st.warning("קובץ Risk Preset לא תקין.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"טעינת Risk Preset נכשלה: {e}")

    return {
        "risk_sizing": risk_sizing,
        "cov_window": int(cov_window),
        "cov_shrink": float(cov_shrink),
        "sector_caps_dict": sector_caps_dict,
        "region_caps_dict": region_caps_dict,
        "currency_caps_dict": currency_caps_dict,
    }


# ===========================
# Regime Model UI (HMM / KMeans / XGB)
# ===========================


def _macro_regime_model_ui(namespace: str = TAB_KEY) -> Dict[str, Any]:
    """UI להגדרת מודל משטרים (Regime model) ל-Profile/Backtest."""
    with st.sidebar.expander("🧠 Regime model (HMM / KMeans / XGB)", expanded=False):
        rm_mode = st.selectbox(
            "Mode",
            ["none", "hmm", "kmeans", "xgb"],
            index=0,
            key=keygen(namespace, "rm", "mode"),
        )
        rm_states = st.number_input(
            "States (k)",
            min_value=2,
            max_value=6,
            value=int(state_get(keygen(namespace, "rm", "k"), 3) or 3),
            step=1,
            key=keygen(namespace, "rm", "k"),
        )
        rm_pca = st.number_input(
            "PCA components (0=off)",
            min_value=0,
            max_value=16,
            value=int(state_get(keygen(namespace, "rm", "pca"), 3) or 3),
            step=1,
            key=keygen(namespace, "rm", "pca"),
        )
        rm_smooth = st.number_input(
            "Smooth window",
            min_value=0,
            max_value=50,
            value=int(state_get(keygen(namespace, "rm", "smooth"), 5) or 5),
            step=1,
            key=keygen(namespace, "rm", "smooth"),
        )
        rm_hyst = st.number_input(
            "Hysteresis (ticks)",
            min_value=0,
            max_value=50,
            value=int(state_get(keygen(namespace, "rm", "hyst"), 3) or 3),
            step=1,
            key=keygen(namespace, "rm", "hyst"),
        )
        rm_temp = st.number_input(
            "Temp scale (<=1=sharp)",
            min_value=0.1,
            max_value=5.0,
            value=float(state_get(keygen(namespace, "rm", "temp"), 1.0) or 1.0),
            step=0.1,
            key=keygen(namespace, "rm", "temp"),
        )
        rm_prior = st.text_input(
            "Class prior (csv, optional)",
            value=str(state_get(keygen(namespace, "rm", "prior"), "")),
            key=keygen(namespace, "rm", "prior"),
            help="לדוגמה: 0.2,0.5,0.3 כאשר הסכום ≈ 1.",
        )

    # נבנה מבנה cfg קטן לשימוש ב-compute_profile / backtest
    prior_list: Optional[List[float]] = None
    if rm_prior.strip():
        try:
            prior_list = [float(x.strip()) for x in rm_prior.split(",") if x.strip()]
        except Exception:  # noqa: BLE001
            prior_list = None

    return {
        "mode": rm_mode,
        "n_states": int(rm_states),
        "pca_components": int(rm_pca),
        "smooth_window": int(rm_smooth),
        "hysteresis": int(rm_hyst),
        "temp_scale": float(rm_temp),
        "class_prior": prior_list,
    }


# ===========================
# IBKR Client Portal — מקורות Live
# ===========================


def _macro_ibkr_portal_ui(namespace: str = TAB_KEY) -> None:
    """UI לניהול חיבור IBKR Client Portal והגדרת מקורות live לנתוני מאקרו."""
    with st.sidebar.expander("🔐 IBKR Client Portal", expanded=False):
        ibkr_token = st.text_input(
            "Portal access token",
            type="password",
            value=str(state_get(keygen(namespace, "ibkr", "token"), "")),
            key=keygen(namespace, "ibkr", "token_input"),
            help="הדבק כאן טוקן גישה מ-Client Portal. נשמר בזיכרון ריצה בלבד.",
        )
        if ibkr_token:
            state_set(keygen(namespace, "ibkr", "token"), ibkr_token)

        ibkr_base = st.text_input(
            "Base URL",
            value=str(state_get(keygen(namespace, "ibkr", "base"), "https://localhost:5000")),
            key=keygen(namespace, "ibkr", "base_input"),
            help="למשל: https://localhost:5000 (Client Portal)",
        )
        if ibkr_base:
            state_set(keygen(namespace, "ibkr", "base"), ibkr_base)

        auto_live = st.toggle(
            "הפעל Live אוטומטית אם יש טוקן",
            value=bool(state_get(keygen(namespace, "ibkr", "autolive"), True)),
            key=keygen(namespace, "ibkr", "autolive"),
        )

        st.divider()
        st.markdown("**מקורות מאקרו לדוגמה (דרך IBKR):**")
        col1, col2, col3 = st.columns([1.2, 1, 1])
        with col1:
            src_key = st.text_input(
                "שם מקור (לוגי)",
                value=str(state_get(keygen(namespace, "ibkr", "src_key"), "vix_live")),
                key=keygen(namespace, "ibkr", "src_key"),
            )
        with col2:
            conid = st.text_input(
                "CONID",
                value=str(state_get(keygen(namespace, "ibkr", "conid"), "12222083")),
                key=keygen(namespace, "ibkr", "conid"),
            )
        with col3:
            field = st.text_input(
                "FIELD",
                value=str(state_get(keygen(namespace, "ibkr", "field"), "31")),
                key=keygen(namespace, "ibkr", "field"),
            )

        if st.button("➕ הוסף מקור IBKR", key=keygen(namespace, "ibkr", "add_src")):
            uri = f"ibkr:conid={conid}?field={field}"
            overrides = state_get("macro_sources_overrides", {}) or {}
            overrides[str(src_key)] = uri
            state_set("macro_sources_overrides", overrides)
            st.success(f"נוסף מקור: {src_key} → {uri}")

        overrides = state_get("macro_sources_overrides", {})
        if overrides:
            st.caption("מקורות שנוספו בטאב זה")
            st.json(overrides)
            if st.button("נקה Overrides", key=keygen(namespace, "ibkr", "clr")):
                state_set("macro_sources_overrides", {})
                st.success("ה־Overrides נוקו.")

        st.divider()
        # Ping חיבור IBKR
        if st.button("🔎 בדוק חיבור IBKR", key=keygen(namespace, "ibkr", "ping")):
            try:
                from core.macro_data import MacroDataClient  # type: ignore

                token_val = state_get(keygen(namespace, "ibkr", "token"))
                base_val = state_get(keygen(namespace, "ibkr", "base"), "https://localhost:5000")
                if not token_val:
                    st.error("חסר Portal access token — הדבק טוקן לפני בדיקה.")
                else:
                    client = MacroDataClient(
                        sources={"ping": f"ibkr:conid={conid}?field={field}"},
                        allow_ibkr=True,
                        ibkr_token=token_val,
                        ibkr_base_url=base_val,
                        allow_yf=False,
                        allow_duckdb=False,
                        allow_sql=False,
                    )
                    with st.spinner("בודק חיבור ל-IBKR…"):
                        df_ping = client.get("ping", live=True)
                    if isinstance(df_ping, pd.DataFrame) and not df_ping.empty:
                        val = df_ping.get("value")
                        if val is not None and not val.empty:
                            _ = float(val.iloc[-1])
                            st.success("✅ חיבור תקין וקיבלנו ערך.")
                            st.dataframe(df_ping.tail(1), use_container_width=True)
                        else:
                            st.warning("התקבלה תשובה אך ללא ערך תקין.")
                    else:
                        st.warning("לא התקבלה תשובת DataFrame תקפה מ-IBKR.")
            except Exception as e:  # noqa: BLE001
                LOGGER.exception("IBKR ping failed: %s", e)
                st.error(f"❌ כשל בבדיקת חיבור: {e}")

        # Auto-live אינפורמטיבי (העדפות) — את העדכון ל-MacroConfig נעשה ב-render()
        state_set(keygen(namespace, "ibkr", "auto_live_flag"), bool(auto_live))


# ===========================
# Sidebar מרכזי לפרמטרי MacroConfig
# ===========================


def _render_sidebar(namespace: str = TAB_KEY) -> Dict[str, Any]:
    """Sidebar לפרמטרים המרכזיים של MacroConfig (ללא Workspace).

    מה מוגדר כאן:
    --------------
    - macro_model: סוג מודל המשטר (HMM/HSMM/k-means/XGBoost) — אינפורמטיבי למנוע.
    - lookback: חלון רגישות לזוג (ימים).
    - ewma_lambda: פרמטר החלקה.
    - min_regime: אורך מינימלי למשטר (ימים).
    - target_vol: יעד תנודתיות לתיק (%).
    - max_pair_weight: משקל מקסימלי לזוג (%).
    - sector_cap: תקרת סקטור כללית (%).
    - score_filter_threshold: סף סינון לפי Macro Fit (0–100).
    - score_gain_floor: רצפת ציון שמעליה מגבירים משקל (0–100).
    - score_weight_gain: עוצמת תגבור משקל (0.00–0.50).
    - caps_mode: hint / clip (רק רמזים או חיתוך בפועל).
    - paper_mode: האם הטאב פועל ב-Paper Mode בלבד.

    החזרת הערכים:
    --------------
    הפונקציה מחזירה מילון ui_cfg; Workspace (date_range/freq/env) נלקח
    מהפונקציה _workspace_sidebar ושמור ב-session_state.
    """
    with st.sidebar:
        st.subheader("⚙️ התאמות מאקרו כלכליות — תצורה")

        model = st.selectbox(
            "בחירת מודל משטרים",
            options=["HMM", "HSMM", "k-means", "XGBoost"],
            index=0,
            key=keygen(namespace, "macro_model"),
        )

        lookback = st.slider(
            "Lookback לרגישות/פרופיל (ימים)",
            min_value=60,
            max_value=720,
            value=int(state_get(keygen(namespace, "lookback"), 365) or 365),
            step=15,
            key=keygen(namespace, "lookback"),
        )
        ewma_lambda = st.slider(
            "EWMA λ",
            min_value=0.80,
            max_value=0.995,
            value=float(state_get(keygen(namespace, "ewma"), 0.94) or 0.94),
            step=0.005,
            key=keygen(namespace, "ewma"),
        )
        min_regime = st.slider(
            "משך מינימלי למשטר (ימים)",
            min_value=5,
            max_value=60,
            value=int(state_get(keygen(namespace, "min_regime"), 15) or 15),
            step=1,
            key=keygen(namespace, "min_regime"),
        )

        target_vol = st.slider(
            "יעד תנודתיות (%)",
            min_value=5.0,
            max_value=40.0,
            value=float(state_get(keygen(namespace, "target_vol"), 15.0) or 15.0),
            step=0.5,
            key=keygen(namespace, "target_vol"),
        )
        max_pair_weight = st.slider(
            "מקס' משקל לזוג (%)",
            min_value=1.0,
            max_value=20.0,
            value=float(state_get(keygen(namespace, "max_pair_w"), 5.0) or 5.0),
            step=0.5,
            key=keygen(namespace, "max_pair_w"),
        )
        sector_cap = st.slider(
            "תקרת סקטור כללית (%)",
            min_value=5.0,
            max_value=50.0,
            value=float(state_get(keygen(namespace, "sector_cap"), 20.0) or 20.0),
            step=1.0,
            key=keygen(namespace, "sector_cap"),
        )

        st.markdown("**סינון/תגבור לפי Macro Fit**")
        score_filter_threshold = st.slider(
            "סף סינון לפי ציון (0–100)",
            min_value=0.0,
            max_value=100.0,
            value=float(state_get(keygen(namespace, "score_thr"), 30.0) or 30.0),
            step=1.0,
            key=keygen(namespace, "score_thr"),
        )
        score_gain_floor = st.slider(
            "רצפת תגבור משקל (0–100)",
            min_value=0.0,
            max_value=100.0,
            value=float(state_get(keygen(namespace, "score_floor"), 60.0) or 60.0),
            step=1.0,
            key=keygen(namespace, "score_floor"),
        )
        score_weight_gain = st.slider(
            "עוצמת תגבור משקל (0.00–0.50)",
            min_value=0.0,
            max_value=0.5,
            value=float(state_get(keygen(namespace, "score_gain"), 0.15) or 0.15),
            step=0.01,
            key=keygen(namespace, "score_gain"),
        )
        caps_mode = st.selectbox(
            "מצב Caps (רמז/חיתוך)",
            options=["hint", "clip"],
            index=0,
            key=keygen(namespace, "caps_mode"),
        )

        paper_mode = st.toggle(
            "Paper Mode (ללא שליחה ל-Executor)",
            value=bool(state_get(keygen(namespace, "paper"), True)),
            key=keygen(namespace, "paper"),
        )

        st.divider()
        if st.button("איפוס טאב (מאקרו בלבד)", key=keygen(namespace, "reset")):
            for k in list(st.session_state.keys()):
                if str(k).startswith("macro_") or str(k).startswith("feature_flag.") or str(k).startswith("risk."):
                    del st.session_state[k]
            st.success("טאב המאקרו אופס — מרענן…")
            st.rerun()

    # Workspace (תאריכים/תדירות/סביבה) כבר שמור ב-session_state ע"י _workspace_sidebar
    ws_state = state_get("macro_workspace", {})
    date_range_val: Optional[Tuple[Any, Any]] = None
    try:
        start = ws_state.get("start")
        end = ws_state.get("end")
        if start and end:
            date_range_val = (pd.to_datetime(start), pd.to_datetime(end))
    except Exception:  # noqa: BLE001
        date_range_val = None

    return {
        "date_range": date_range_val,
        "model": model,
        "lookback": lookback,
        "ewma_lambda": ewma_lambda,
        "min_regime": min_regime,
        "target_vol": target_vol,
        "max_pair_weight": max_pair_weight,
        "sector_cap": sector_cap,
        "score_filter_threshold": score_filter_threshold,
        "score_gain_floor": score_gain_floor,
        "score_weight_gain": score_weight_gain,
        "caps_mode": caps_mode,
        "paper_mode": paper_mode,
    }

# ---------------------------------------------------------------------------
# Visuals & Analytics — Regime Timeline, Features Heatmap, Overlay, Shocks, DNA
# ---------------------------------------------------------------------------


# ===========================
# Regime Timeline (probabilities)
# ===========================


def _render_regime_timeline(regimes_prob: Optional[pd.DataFrame]) -> None:
    """תרשים הסתברויות משטרים (אם plotly זמין)."""
    if regimes_prob is None or regimes_prob.empty:
        st.info("אין נתוני הסתברויות משטר להצגה.")
        return
    if not _HAS_PX:
        st.info("Plotly לא מותקן — דלג על התרשים.")
        return

    df = regimes_prob.copy()
    df = _normalize_datetime_index(df)
    df_reset = df.reset_index()
    x_col = df.index.name or "date"

    fig = px.area(
        df_reset,
        x=x_col,
        y=df.columns,
        title="Regime Probabilities Over Time",
    )
    st.plotly_chart(fig, use_container_width=True)

def _regimes_tools(regimes_prob: Optional[pd.DataFrame]) -> None:
    """כלי עזר להסתברויות משטרים: הורדה וסטטיסטיקה קצרה."""
    if regimes_prob is None or regimes_prob.empty:
        return

    st.caption(f"דגימות: {len(regimes_prob):,}")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="⬇️ הורד Regimes (CSV)",
            data=regimes_prob.to_csv(index=True).encode("utf-8"),
            file_name="macro_regimes_prob.csv",
            mime="text/csv",
            key=keygen(TAB_KEY, "reg", "dl_csv"),
        )
    with c2:
        cols_payload = {"columns": list(regimes_prob.columns)}
        st.download_button(
            label="⬇️ הורד רשימת עמודות (JSON)",
            data=json.dumps(cols_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="macro_regimes_columns.json",
            mime="application/json",
            key=keygen(TAB_KEY, "reg", "dl_cols"),
        )

    if get_flag("macro_show_regimes_table", True):
        show_tbl = st.toggle(
            "הצג טבלת Regimes (תצוגה)",
            value=False,
            key=keygen(TAB_KEY, "reg", "show_tbl"),
        )
        if show_tbl:
            n = st.number_input(
                "שורות אחרונות לתצוגה",
                min_value=50,
                max_value=2000,
                value=300,
                step=50,
                key=keygen(TAB_KEY, "reg", "nsample"),
            )
            st.dataframe(regimes_prob.tail(int(n)), use_container_width=True)


# ===========================
# Macro Features Heatmap
# ===========================


def _render_macro_heatmap(features_df: Optional[pd.DataFrame]) -> Optional[object]:
    """Heatmap לפיצ'רי מאקרו (z-score) אם אפשר. מחזיר את ה-Figure אם זמין."""
    if features_df is None or features_df.empty:
        st.info("אין פיצ'רים להצגה.")
        return None
    if not _HAS_PX:
        st.info("Plotly לא מותקן — דלג על ההיטמאפ.")
        return None

    df = features_df.copy()
    df = _normalize_datetime_index(df)
    df = df.tail(200)

    fig = px.imshow(
        df.T,
        aspect="auto",
        origin="lower",
        labels=dict(x="זמן", y="פיצ'רים", color="value"),
        title="Macro Features Heatmap (Tail 200)",
    )
    st.plotly_chart(fig, use_container_width=True)
    return fig


def _features_tools(
    features_df: Optional[pd.DataFrame],
    params: Optional[Dict[str, Any]] = None,
    fig: Optional[object] = None,
) -> None:
    """כלי עזר לפיצ'רים: הורדות, עמודות, PNG להיטמאפ, ותצוגת טבלה."""
    if features_df is None or features_df.empty:
        return

    n_rows, n_cols = features_df.shape
    st.caption(
        f"סה\"כ פיצ'רים: {n_cols} עמודות × {n_rows} שורות "
        "(מוצגות 200 אחרונות בהיטמאפ).",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label="⬇️ הורד Features (CSV)",
            data=features_df.to_csv(index=True).encode("utf-8"),
            file_name="macro_features_summary.csv",
            mime="text/csv",
            key=keygen(TAB_KEY, "feat", "dl_csv"),
        )
    with c2:
        cols_payload = {"columns": list(features_df.columns)}
        st.download_button(
            label="⬇️ הורד רשימת עמודות (JSON)",
            data=json.dumps(cols_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="macro_features_columns.json",
            mime="application/json",
            key=keygen(TAB_KEY, "feat", "dl_cols"),
        )
    with c3:
        if fig is not None and pio is not None:
            export_png = st.toggle(
                "ייצא Heatmap כ-PNG",
                value=False,
                key=keygen(TAB_KEY, "feat", "pngtg"),
            )
            if export_png:
                try:
                    png_bytes = pio.to_image(fig, format="png", scale=2)
                    st.download_button(
                        label="⬇️ הורד Heatmap (PNG)",
                        data=png_bytes,
                        file_name="macro_features_heatmap.png",
                        mime="image/png",
                        key=keygen(TAB_KEY, "feat", "dl_png"),
                    )
                except Exception:
                    st.info("כדי לייצא PNG יש להתקין את 'kaleido'.")

    show_tbl = st.toggle(
        "הצג טבלת Features (תצוגה)",
        value=False,
        key=keygen(TAB_KEY, "feat", "show_tbl"),
    )
    if show_tbl:
        sample = st.number_input(
            "שורות אחרונות לתצוגה",
            min_value=10,
            max_value=2000,
            value=200,
            step=10,
            key=keygen(TAB_KEY, "feat", "nsample"),
        )
        st.dataframe(features_df.tail(int(sample)), use_container_width=True)

    if params is not None and get_flag("macro_debug", False):
        with st.expander("🐞 Feature Params (Debug)", expanded=False):
            st.json(params, expanded=False)


# ===========================
# Pair Macro Fit & Overlay Tables
# ===========================


def _render_pair_fit_table(profile: Optional[Dict[str, Any]]) -> None:
    """טבלת דירוג זוגים לפי Macro Fit (אם קיים פרופיל)."""
    if not profile:
        st.info("טרם חושב Macro Profile — לחץ 'חשב Macro Profile'.")
        return

    rows: List[Dict[str, Any]] = []
    for pid, row in profile.items():
        rows.append(
            {
                "pair_id": pid,
                "macro_fit_score": row.get("macro_fit_score"),
                "weight_adj": row.get("weight_adj"),
                "action": row.get("action"),
                "reason": row.get("reason"),
            },
        )

    if not rows:
        st.info("אין נתונים להצגה ב-Macro Profile.")
        return

    df = pd.DataFrame(rows).sort_values(
        ["macro_fit_score"],
        ascending=False,
        na_position="last",
    )
    st.dataframe(df.head(50), use_container_width=True)


def _render_overlay_table(profile: Optional[Dict[str, Any]]) -> None:
    """טבלת Overlay מלאה: include / caps / weight_adj / score / action."""
    if not profile:
        st.info("אין Overlay להצגה. לחץ 'חשב Macro Profile'.")
        return

    rows: List[Dict[str, Any]] = []
    for pid, row in profile.items():
        rows.append(
            {
                "pair_id": pid,
                "include": row.get("include", True),
                "weight_adj": row.get("weight_adj"),
                "macro_fit_score": row.get("macro_fit_score"),
                "cap_applied": row.get("cap_applied", False),
                "cap_value": row.get("cap_value"),
                "action": row.get("action"),
            },
        )

    df = pd.DataFrame(rows).sort_values(
        ["include", "macro_fit_score"],
        ascending=[False, False],
        na_position="last",
    )
    st.dataframe(df, use_container_width=True)


def _overlay_downloads(profile: Optional[Dict[str, Any]]) -> None:
    """כפתורי הורדה ל-Overlay (JSON/CSV) אם קיים."""
    if not isinstance(profile, dict) or not profile:
        return

    st.download_button(
        label="⬇️ הורד Overlay (JSON)",
        data=json.dumps(profile, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="macro_overlay.json",
        mime="application/json",
        key=keygen(TAB_KEY, "overlay", "json"),
    )
    df = pd.DataFrame(
        [
            {"pair_id": k, **(v or {})}
            for k, v in profile.items()
        ],
    )
    st.download_button(
        label="⬇️ הורד Overlay (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="macro_overlay.csv",
        mime="text/csv",
        key=keygen(TAB_KEY, "overlay", "csv"),
    )


# ===========================
# Macro Shocks Section
# ===========================


def _render_macro_shocks_section(state: MacroTabState) -> None:
    """מחשב ומציג Macro Shocks (אם הפונקציה זמינה)."""
    st.subheader("⚡ Macro Shocks")

    macro_df = state.macro_df or state_get(MacroStateKeys.MACRO_DF)
    if macro_df is None or not isinstance(macro_df, pd.DataFrame) or macro_df.empty:
        st.info("אין macro_df זמין לחישוב שוקים.")
        return

    shocks_df: Optional[pd.DataFrame] = None
    try:
        shocks_df = detect_macro_shocks(macro_df)  # type: ignore[call-arg]
    except TypeError:
        # ייתכן שהפונקציה דורשת פרמטרים נוספים — במקרה כזה המשתמש יעדכן במודול המאקרו.
        LOGGER.debug("detect_macro_shocks signature mismatch (TypeError)", exc_info=True)
    except Exception as e:  # noqa: BLE001
        LOGGER.debug("detect_macro_shocks failed (non-fatal): %s", e)
        shocks_df = None

    if shocks_df is None or shocks_df.empty:
        st.info("לא זוהו שוקים משמעותיים בפקטורי המאקרו (או שהפונקציה מחזירה ריק).")
        return

    shocks_df = _normalize_datetime_index(shocks_df)
    state.macro_shocks_df = shocks_df
    state_set(MacroStateKeys.SHOCKS_DF, shocks_df)

    st.caption(f"מספר שוקים מזוהים: {len(shocks_df):,}")
    st.dataframe(shocks_df.tail(50), use_container_width=True)

    st.download_button(
        label="⬇️ הורד Macro Shocks (CSV)",
        data=shocks_df.to_csv(index=True).encode("utf-8"),
        file_name="macro_shocks.csv",
        mime="text/csv",
        key=keygen(TAB_KEY, "shocks", "dl"),
    )


# ===========================
# Pair Macro DNA (רגישות לזוג נבחר)
# ===========================


def _infer_pair_symbols(row: pd.Series) -> Optional[Tuple[str, str]]:
    """מנסה להסיק את סימבולי הזוג (A/B) מתוך שדהי pairs_df שונים."""
    candidates = [
        ("sym_a", "sym_b"),
        ("sym_x", "sym_y"),
        ("a", "b"),
        ("leg_a", "leg_b"),
    ]
    for c1, c2 in candidates:
        if c1 in row.index and c2 in row.index:
            sym1 = str(row[c1])
            sym2 = str(row[c2])
            if sym1 and sym2:
                return sym1, sym2
    return None


def _build_spread_series(
    pair_row: pd.Series,
    prices_wide: pd.DataFrame,
    method: str,
) -> Optional[pd.Series]:
    """בונה סדרת spread לזוג לפי שיטת החישוב שנבחרה."""
    syms = _infer_pair_symbols(pair_row)
    if syms is None:
        return None

    a, b = syms
    if a not in prices_wide.columns or b not in prices_wide.columns:
        return None

    pa = prices_wide[a].astype(float)
    pb = prices_wide[b].astype(float)
    if method == "log_spread":
        spread = (pa.apply(lambda x: float("nan") if x <= 0 else x)).pipe(lambda s: s.apply(pd.np.log)) - (
            pb.apply(lambda x: float("nan") if x <= 0 else x)
        ).pipe(lambda s: s.apply(pd.np.log))
    elif method == "ratio":
        spread = pa / pb
    else:  # diff
        spread = pa - pb

    spread = spread.dropna()
    spread.name = f"{a}_{b}_{method}"
    return spread


def _render_pair_macro_dna_section(state: MacroTabState, pairs_df: pd.DataFrame) -> None:
    """מציג את רגישות המאקרו ('DNA מאקרו') לזוג נבחר."""
    st.subheader("🧬 DNA מאקרו לזוג נבחר")

    if pairs_df is None or pairs_df.empty:
        st.info("אין pairs_df זמין — לא ניתן לחשב DNA לזוג.")
        return

    # בחירת זוג
    pair_ids = list(pairs_df["pair_id"].astype(str).unique()) if "pair_id" in pairs_df.columns else []
    if not pair_ids:
        st.info("לא נמצאה עמודת pair_id ב-pairs_df.")
        return

    selected_pair = st.selectbox(
        "בחר זוג לניתוח מאקרו",
        options=pair_ids,
        index=0,
        key=keygen(TAB_KEY, "dna", "pair"),
    )

    spread_method = st.selectbox(
        "שיטת ספרד",
        options=["log_spread", "diff", "ratio"],
        index=0,
        key=keygen(TAB_KEY, "dna", "method"),
    )

    run_dna = st.button(
        "חשב DNA מאקרו לזוג",
        key=keygen(TAB_KEY, "dna", "run"),
    )

    if not run_dna:
        st.caption("בחר זוג ולחץ על 'חשב DNA מאקרו לזוג' כדי לקבל רגישות מפורטת.")
        return

    prices_wide = state_get("prices_wide")
    if not isinstance(prices_wide, pd.DataFrame) or prices_wide.empty:
        st.error("prices_wide לא זמין ב-session_state — הטאב המאקרו צריך גישה למחירי הזוגות.")
        return

    try:
        idx = pairs_df["pair_id"].astype(str) == str(selected_pair)
        row = pairs_df.loc[idx].iloc[0]
    except Exception:  # noqa: BLE001
        st.error("לא הצלחתי למצוא את הזוג המתאים ב-pairs_df.")
        return

    spread = _build_spread_series(row, prices_wide, spread_method)
    if spread is None or spread.empty:
        st.error("לא הצלחתי לבנות סדרת spread לזוג (בדוק עמודות סימבולים ו-prices_wide).")
        return

    macro_df = state.macro_df or state_get(MacroStateKeys.MACRO_DF)
    if macro_df is None or not isinstance(macro_df, pd.DataFrame) or macro_df.empty:
        st.error("macro_df לא זמין — לא ניתן לחשב רגישות מאקרו לזוג.")
        return

    try:
        regime_df = build_macro_regime_series(macro_df)  # type: ignore[call-arg]
    except Exception as e:  # noqa: BLE001
        LOGGER.debug("build_macro_regime_series failed: %s", e)
        st.error("כשל בבניית Regime DF — לא ניתן להשלים את ניתוח ה-DNA.")
        return

    try:
        cfg_sens = state.pair_sens_cfg
        if cfg_sens is not None:
            sens = compute_pair_macro_sensitivity(
                spread_series=spread,
                macro_df=macro_df,
                regime_df=regime_df,
                cfg=cfg_sens,  # type: ignore[arg-type]
            )
        else:
            sens = compute_pair_macro_sensitivity(
                spread_series=spread,
                macro_df=macro_df,
                regime_df=regime_df,
            )
    except TypeError:
        # חתימת פונקציה שונה — ננסה קריאה פשוטה
        sens = compute_pair_macro_sensitivity(spread, macro_df, regime_df)  # type: ignore[call-arg]
    except Exception as e:  # noqa: BLE001
        LOGGER.exception("compute_pair_macro_sensitivity failed: %s", e)
        st.error("כשל בחישוב DNA מאקרו לזוג — בדוק את מודול המקרו.")
        return

    # הצגה
    summary_text = getattr(sens, "summary_text", "")
    overall_score = getattr(sens, "overall_score", None)

    top_cols = st.columns(2)
    with top_cols[0]:
        st.markdown(f"**זוג:** `{selected_pair}` — שיטת Spread: `{spread_method}`")
        if summary_text:
            st.markdown(f"**סיכום איכותי:** {summary_text}")
    with top_cols[1]:
        if overall_score is not None:
            try:
                score_f = float(overall_score)
                st.metric("Overall Macro Sensitivity Score", f"{score_f:.1f}/100")
            except Exception:  # noqa: BLE001
                pass

    # טבלת exposures
    exposures = getattr(sens, "exposures", None)
    if exposures:
        try:
            exp_df = build_exposures_table(exposures)  # type: ignore[call-arg]
        except Exception:
            # ננסה לבנות ידנית
            rows: List[Dict[str, Any]] = []
            for fac, obj in exposures.items():
                rows.append(
                    {
                        "factor": fac,
                        "beta": getattr(obj, "beta", None),
                        "tstat": getattr(obj, "tstat", None),
                        "pvalue": getattr(obj, "pvalue", None),
                    },
                )
            exp_df = pd.DataFrame(rows)
        st.markdown("**רגישות לפקטורי מאקרו (Betas / t-stat / p-value):**")
        st.dataframe(exp_df, use_container_width=True)
    else:
        st.info("לא נמצאה טבלת exposures ב-PairMacroSensitivity.")

    # טבלת ביצועים לפי משטרים
    regime_perf = getattr(sens, "regime_perf", None)
    if regime_perf:
        try:
            reg_df = build_regime_performance_table(regime_perf)  # type: ignore[call-arg]
        except Exception:
            rows = []
            for rkey, obj in regime_perf.items():
                rows.append(
                    {
                        "regime_key": rkey,
                        "label": getattr(obj, "label", rkey),
                        "n_obs": getattr(obj, "n_obs", None),
                        "mean": getattr(obj, "mean", None),
                        "vol": getattr(obj, "vol", None),
                        "sharpe": getattr(obj, "sharpe", None),
                        "hit_ratio": getattr(obj, "hit_ratio", None),
                    },
                )
            reg_df = pd.DataFrame(rows)
        st.markdown("**ביצועים לפי משטרים (mean/vol/sharpe/hit_ratio):**")
        st.dataframe(reg_df, use_container_width=True)
    else:
        st.info("לא נמצאה טבלת ביצועים לפי משטר ב-PairMacroSensitivity.")

    # הורדה כ-JSON
    try:
        if hasattr(sens, "to_dict"):
            payload = sens.to_dict()  # type: ignore[call-arg]
        else:
            payload = {
                "overall_score": overall_score,
                "summary_text": summary_text,
                "exposures": getattr(sens, "exposures", {}),
                "regime_perf": getattr(sens, "regime_perf", {}),
            }
        st.download_button(
            label="⬇️ הורד Pair Macro DNA (JSON)",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"pair_macro_dna_{selected_pair}.json",
            mime="application/json",
            key=keygen(TAB_KEY, "dna", "dl"),
        )
    except Exception:  # noqa: BLE001
        pass
# ---------------------------------------------------------------------------
# Orchestration — הפונקציה הראשית render()
# ---------------------------------------------------------------------------


def _sync_cfg_with_ui(
    cfg: MacroConfig,
    ui_cfg: Dict[str, Any],
    risk_cfg: Dict[str, Any],
) -> MacroConfig:
    """מעביר ערכי UI אל MacroConfig (score thresholds, caps, risk, sources, IBKR)."""
    try:
        # ספי Macro Fit
        cfg.score_filter_threshold = float(
            ui_cfg.get("score_filter_threshold", getattr(cfg, "score_filter_threshold", 30.0)),
        )
        cfg.score_gain_floor = float(
            ui_cfg.get("score_gain_floor", getattr(cfg, "score_gain_floor", 60.0)),
        )
        cfg.score_weight_gain = float(
            ui_cfg.get("score_weight_gain", getattr(cfg, "score_weight_gain", 0.15)),
        )
        cfg.caps_mode = str(ui_cfg.get("caps_mode", getattr(cfg, "caps_mode", "hint")))

        # Risk sizing / covariance
        try:
            cfg.risk_sizing = str(
                risk_cfg.get("risk_sizing", getattr(cfg, "risk_sizing", "invvol")),
            )  # type: ignore[attr-defined]
            cfg.cov_window = int(
                risk_cfg.get("cov_window", getattr(cfg, "cov_window", 60)),
            )  # type: ignore[attr-defined]
            cfg.cov_shrink = float(
                risk_cfg.get("cov_shrink", getattr(cfg, "cov_shrink", 0.1)),
            )  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            LOGGER.debug("risk sizing sync skipped", exc_info=True)

        # Caps לפי סקטור/אזור/מטבע
        sc = risk_cfg.get("sector_caps_dict", {}) or {}
        rc = risk_cfg.get("region_caps_dict", {}) or {}
        cc = risk_cfg.get("currency_caps_dict", {}) or {}
        try:
            if sc:
                cfg.sector_caps = dict(sc)  # type: ignore[attr-defined]
            if rc:
                cfg.region_caps = dict(rc)  # type: ignore[attr-defined]
            if cc:
                cfg.currency_caps = dict(cc)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            LOGGER.debug("caps sync skipped", exc_info=True)

        # IBKR auto-live
        token_val = state_get(keygen(TAB_KEY, "ibkr", "token"))
        base_val = state_get(
            keygen(TAB_KEY, "ibkr", "base"),
            "https://localhost:5000",
        )
        auto_live_flag = bool(state_get(keygen(TAB_KEY, "ibkr", "auto_live_flag"), True))
        if token_val and auto_live_flag:
            cfg.use_data_client = True
            cfg.data_client_live = True
            cfg.data_client_allow_ibkr = True  # type: ignore[attr-defined]
            cfg.ibkr_token = token_val  # type: ignore[attr-defined]
            cfg.ibkr_base_url = base_val  # type: ignore[attr-defined]

        # מקורות שנוספו מתוך הטאב
        overrides = state_get("macro_sources_overrides", {})
        if isinstance(overrides, dict) and overrides:
            try:
                if not hasattr(cfg, "sources") or getattr(cfg, "sources") is None:
                    cfg.sources = {}  # type: ignore[attr-defined]
                cfg.sources.update({str(k): str(v) for k, v in overrides.items()})  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                LOGGER.debug("cfg.sources sync failed", exc_info=True)
    except Exception:  # noqa: BLE001
        LOGGER.debug("cfg sync skipped", exc_info=True)

    return cfg


def render(
    pairs_df: Optional[pd.DataFrame] = None,
    cfg: Optional[MacroConfig] = None,
    bundle: Optional[MacroBundle] = None,
) -> AdjustmentResult:
    """ממשק הקריאה לדשבורד. מחזיר AdjustmentResult לשאר המערכת.

    Parameters
    ----------
    pairs_df : Optional[pd.DataFrame]
        טבלת הזוגות של המערכת. אם ריקה, יוצג UI ללא נתונים, אך עדיין ניתן
        לראות Snapshot, Regimes, Shocks וכו'.
    cfg : Optional[MacroConfig]
        אובייקט קונפיגורציה להתאמות מאקרו. אם None, ייווצר חדש עם ברירות מחדל
        (או ייטען מה-session_state אם קיים).
    bundle : Optional[MacroBundle]
        חבילת נתוני מאקרו טעונה מראש. אם None, תיטען לפי cfg דרך load_macro_bundle.

    Returns
    -------
    AdjustmentResult
        תוצאת התאמות המאקרו (exposure_multiplier / pair_adjustments / filters / ...).
    """

    # --- Macro context & panels מה-router / dashboard ----------------------
    macro_ctx_from_nav = state_get("macro_ctx_from_nav", {}) or {}
    if not isinstance(macro_ctx_from_nav, Mapping):
        macro_ctx_from_nav = {}

    macro_panels_flags = _get_macro_panels_flags()
    read_only_flag = bool(state_get("macro_read_only", False))

    risk_profile_from_macro = (
        state_get("risk_profile_from_macro", None)
        or macro_ctx_from_nav.get("risk_mode")
    )
    macro_preset_selected = (
        state_get("macro_preset_selected", None)
        or macro_ctx_from_nav.get("macro_preset")
    )

    # focus_pair_id ל-DNA (ה-router כבר מכניס ל-key המתאים)
    focus_pair_id = state_get(keygen(TAB_KEY, "dna", "pair"), None)

    # פונקציה קטנה שנותנת flag לכל סקשן לפי macro_panels_flags
    def _panel_flag(name: str, default: bool = True) -> bool:
        if not macro_panels_flags:
            return default
        try:
            return bool(macro_panels_flags.get(name, default))
        except Exception:  # noqa: BLE001
            return default

    # מה להציג בפועל (אפשר לכוון מה-dashboard דרך macro_panels)
    show_regimes_section = _panel_flag("regimes", True)
    show_features_section = _panel_flag("macro_surprises", True)
    show_overlay_section = _panel_flag("fair_value", True)
    show_shocks_section = _panel_flag("liquidity", True)
    show_dna_section = True  # כרגע תמיד מציגים DNA; אפשר לקשור לפאנל אם תרצה
    show_profile_section = not read_only_flag
    show_backtest_section = not read_only_flag

    # נשמור קצת info ל-debug (לא חובה)
    try:
        LOGGER.debug(
            "macro_tab.render | read_only=%s, risk_profile_from_macro=%s, "
            "preset=%s, focus_pair_id=%s, panels=%s",
            read_only_flag,
            risk_profile_from_macro,
            macro_preset_selected,
            focus_pair_id,
            list(macro_panels_flags.keys()),
        )
    except Exception:  # noqa: BLE001
        pass

    # --- Professional tab header -------------------------------------------
    st.markdown(
        """
<div style="
    background:linear-gradient(90deg,#1B5E20 0%,#2E7D32 100%);
    border-radius:10px;padding:14px 20px;margin-bottom:14px;
    box-shadow:0 2px 8px rgba(46,125,50,0.20);
    display:flex;align-items:center;justify-content:space-between;
">
    <div>
        <div style="font-size:1.20rem;font-weight:800;color:white;letter-spacing:-0.2px;">
            🌍 Macro &amp; Regime Dashboard
        </div>
        <div style="font-size:0.78rem;color:rgba(255,255,255,0.78);margin-top:3px;">
            Regime classification · Factor analysis · Pair overlays · Risk budget hints
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # --- בסיס: pairs_df + ולידציה -----------------------------------------
    pairs_df = _ensure_pairs_df(pairs_df)
    _render_pairs_warnings(pairs_df)

    # --- Feature Flags + Workspace + Presets + IBKR + Feature Params + Risk + Regime Model + Sidebar ---
    feature_toggles_ui(TAB_KEY)
    preset_ui_controls(TAB_KEY)

    workspace = _workspace_sidebar(TAB_KEY)

    # אתחול קונפיגים (MacroConfig + FactorConfig + SensitivityConfig)
    cfg, factor_cfg, pair_sens_cfg = _init_macro_configs(cfg, None, None)

    # IBKR + מקורות live
    _macro_ibkr_portal_ui(TAB_KEY)

    # Feature engineering על פקטורי מאקרו
    feature_params = _feature_params_ui(TAB_KEY)

    # שכבת סיכון (sizing & caps)
    risk_cfg = _macro_risk_controls_ui(TAB_KEY)

    # מודל משטרים (HMM/KMeans/XGB)
    regime_model_cfg = _macro_regime_model_ui(TAB_KEY)

    # Sidebar מרכזי – פרמטרי MacroConfig
    ui_cfg = _render_sidebar(TAB_KEY)

    # סנכרון הפרמטרים אל cfg
    cfg = _sync_cfg_with_ui(cfg, ui_cfg, risk_cfg)

    # --- יצירת State וטענת דאטה מאקרו -------------------------------------
    state = MacroTabState(
        workspace=workspace,
        macro_cfg=cfg,
        factor_cfg=factor_cfg,
        pair_sens_cfg=pair_sens_cfg,
        bundle=bundle,
    )

    # נעביר לתוך state גם מה שכבר קיים ב-session (למשכיות בין ריצות)
    state.macro_profile = state_get(MacroStateKeys.PROFILE, None)
    state.macro_regimes_prob = state_get(MacroStateKeys.REGIMES_PROB, None)
    state.macro_features_df = state_get(MacroStateKeys.FEATURES_DF, None)
    state.macro_shocks_df = state_get(MacroStateKeys.SHOCKS_DF, None)

    state = _load_macro_data_for_workspace(state, pairs_df=pairs_df)

    # --- שכבת Snapshot על המאקרו ------------------------------------------
    _render_macro_snapshot_section(state)

    # --- חישוב התאמות מאקרו לתיק (AdjustmentResult) -----------------------
    session_uid = get_session_uid()
    key = keygen(TAB_KEY, session_uid)

    if state.bundle is None:
        # אם מסיבה כלשהי bundle עדיין None, ננסה שוב לטעון
        try:
            state.bundle = load_macro_bundle(cfg)
            state_set(MacroStateKeys.BUNDLE, state.bundle)
        except Exception as e:  # noqa: BLE001
            LOGGER.error("טעינת MacroBundle נכשלה פעם שנייה: %s", e)

    try:
        with st.spinner("מחשב התאמות מאקרו לתיק…"):
            # תאימות מלאה למנוע הישן: render_streamlit_ui עושה את כל העבודה הכבדה
            result = render_streamlit_ui(
                pairs_df=pairs_df,
                cfg=cfg,
                bundle=state.bundle,
                key=key,
            )
    except Exception as e:  # noqa: BLE001
        LOGGER.exception("שגיאה בתוך render_streamlit_ui: %s", e)
        st.error("❌ שגיאה בהצגת טאב המאקרו. מוצג מצב נייטרלי.")
        # מצב ניטרלי כתוצאה חלופית אם ה-UI קרס
        result = AdjustmentResult(
            exposure_multiplier=1.0,
            pair_adjustments={},
            filters={},
            regime_snapshot=None,
        )

    # שמירת התוצאה ב-session_state וב-state
    state.adjustment_result = result
    state_set(MacroStateKeys.RESULT, result)
    st.session_state["macro_tab_result"] = result  # תאימות לשם הישן
    st.session_state["macro_adjustments_result"] = result

    # Meta + KPIs
    _stamp_result(result, cfg)
    meta = state_get(MacroStateKeys.META, {})
    _render_top_kpis(meta)
    _render_risk_profile_banner()

    # 🔄 דחיפת מדדי Macro + Risk ל-Analysis / Tab Comparison
    try:
        # מאקרו (רגישות גלובלית + mean_score)
        mean_score = _safe_float_or_none(meta.get("mean_score"))
        regime_label = str(meta.get("regime_label", "") or "")

        regime_map = {
            "risk_off": 0.1,
            "stagflation": 0.2,
            "slowdown": 0.3,
            "neutral": 0.5,
            "reflation": 0.7,
            "risk_on": 0.9,
        }
        macro_sensitivity = regime_map.get(regime_label, 0.5)
        macro_score = mean_score if mean_score is not None else 50.0

        push_macro_metrics_to_ctx(
            macro_sensitivity=macro_sensitivity,
            macro_score=macro_score,
        )

        # Risk metrics גלובליים
        push_risk_metrics_to_ctx(result, meta)
    except Exception:
        LOGGER.debug("push_macro/risk metrics to ctx failed (non-fatal)", exc_info=True)

    _render_live_health(cfg)

    # ----------------------------------------------------------------------
    # Backtest / Profile — קבצים, כפתורים ותוצאות
    # ----------------------------------------------------------------------
    if show_backtest_section:
        with st.expander("Backtest — טען קבצים", expanded=False):
            ret_file = st.file_uploader(
                "Returns (CSV/Parquet)",
                type=["csv", "parquet"],
                key=keygen(TAB_KEY, "upl", "returns"),
            )
            pairs_file_bt = st.file_uploader(
                "Pairs Universe (CSV/Parquet)",
                type=["csv", "parquet"],
                key=keygen(TAB_KEY, "upl", "pairs"),
            )
    else:
        ret_file = None
        pairs_file_bt = None

    cols_actions = st.columns(3)
    with cols_actions[0]:
        if show_profile_section:
            run_profile = st.button(
                "חשב Macro Profile",
                key=keygen(TAB_KEY, "act", "compute"),
            )
        else:
            st.button(
                "חשב Macro Profile",
                key=keygen(TAB_KEY, "act", "compute_disabled"),
                disabled=True,
            )
            run_profile = False
    with cols_actions[1]:
        if show_backtest_section:
            run_bt = st.button(
                "Backtest",
                key=keygen(TAB_KEY, "act", "bt"),
            )
        else:
            st.button(
                "Backtest",
                key=keygen(TAB_KEY, "act", "bt_disabled"),
                disabled=True,
            )
            run_bt = False
    with cols_actions[2]:
        if read_only_flag:
            st.caption("מצב Read-only: חישובים כבדים (Profile/Backtest) נעולים מהדשבורד.")
        else:
            st.write("")

    # --- Macro Profile ----------------------------------------------------
    if run_profile:
        try:
            from core.macro_engine import compute_profile  # type: ignore

            cfg_for_profile: Dict[str, Any] = dict(ui_cfg)
            cfg_for_profile["feature_params"] = dict(feature_params or {})

            if regime_model_cfg.get("mode") != "none":
                cfg_for_profile["regime_model"] = dict(regime_model_cfg)

            prof = compute_profile(pairs_df=pairs_df, cfg=cfg_for_profile)
            profile = getattr(prof, "profile", getattr(prof, "profile", {}))
            regimes_prob = getattr(prof, "regimes_prob", None)
            features_df = getattr(prof, "features_summary", None)

            state.macro_profile = profile
            state.macro_regimes_prob = regimes_prob
            state.macro_features_df = features_df

            state_set(MacroStateKeys.PROFILE, profile)
            state_set(MacroStateKeys.REGIMES_PROB, regimes_prob)
            state_set(MacroStateKeys.FEATURES_DF, features_df)
            st.success("Macro Profile חושב בהצלחה.")
        except Exception as e:  # noqa: BLE001
            LOGGER.exception("compute_profile failure: %s", e)
            st.error("כשל בחישוב Macro Profile — ודא שקבצי core קיימים.")

    # --- Backtest ---------------------------------------------------------
    if run_bt:
        try:
            from core.macro_engine import (  # type: ignore
                backtest as macro_backtest,
                run_backtest_with_files,
            )

            with st.spinner("מריץ Backtest מאקרו…"):
                if ret_file is not None:
                    cfg_bt: Dict[str, Any] = dict(ui_cfg)
                    cfg_bt["feature_params"] = dict(feature_params or {})
                    if regime_model_cfg.get("mode") != "none":
                        cfg_bt["regime_model"] = dict(regime_model_cfg)
                    # המנוע בד"כ מצפה ל-max_pair_weight בפורמט fraction ולא אחוז
                    if isinstance(cfg_bt.get("max_pair_weight"), (int, float)):
                        cfg_bt["max_pair_weight"] = float(cfg_bt["max_pair_weight"]) / 100.0

                    res_bt = run_backtest_with_files(
                        returns_obj=ret_file.getvalue(),
                        returns_name=str(ret_file.name),
                        pairs_obj=(
                            pairs_file_bt.getvalue()
                            if pairs_file_bt is not None
                            else None
                        ),
                        pairs_name=(
                            str(pairs_file_bt.name)
                            if pairs_file_bt is not None
                            else None
                        ),
                        cfg=cfg_bt,
                    )
                else:
                    res_bt = macro_backtest(cfg=dict(ui_cfg))

            st.success("Backtest הושלם")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Sharpe", f"{getattr(res_bt, 'sharpe', 0.0):.2f}")
            c2.metric("Sortino", f"{getattr(res_bt, 'sortino', 0.0):.2f}")
            c3.metric("CAGR", f"{getattr(res_bt, 'cagr', 0.0):.2%}")
            c4.metric("Max DD", f"{getattr(res_bt, 'max_dd', 0.0):.2%}")
            c5.metric("Uplift vs EW", f"{getattr(res_bt, 'uplift_vs_base', 0.0):.2%}")

            bt_payload = {
                "sharpe": float(getattr(res_bt, "sharpe", 0.0)),
                "sortino": float(getattr(res_bt, "sortino", 0.0)),
                "cagr": float(getattr(res_bt, "cagr", 0.0)),
                "max_dd": float(getattr(res_bt, "max_dd", 0.0)),
                "uplift_vs_base": float(getattr(res_bt, "uplift_vs_base", 0.0)),
            }
            json_bytes = json.dumps(
                bt_payload,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8")
            st.download_button(
                label="⬇️ הורד Backtest (JSON)",
                data=json_bytes,
                file_name="macro_backtest_result.json",
                mime="application/json",
                key=keygen(TAB_KEY, "bt", "dl_json"),
            )
            bt_df = pd.DataFrame([bt_payload])
            st.download_button(
                label="⬇️ הורד Backtest (CSV)",
                data=bt_df.to_csv(index=False).encode("utf-8"),
                file_name="macro_backtest_result.csv",
                mime="text/csv",
                key=keygen(TAB_KEY, "bt", "dl_csv"),
            )
        except Exception as e:  # noqa: BLE001
            LOGGER.exception("macro backtest failure: %s", e)
            st.error("כשל בהרצת Backtest — ודא שקבצי core קיימים.")

    # ----------------------------------------------------------------------
    # Visuals & Analytics — Regimes, Features, Overlay, Shocks, DNA
    # ----------------------------------------------------------------------
    if show_regimes_section:
        st.subheader("ציר זמן משטרים")
        rp = state.macro_regimes_prob or state_get(MacroStateKeys.REGIMES_PROB)
        _render_regime_timeline(rp)
        _regimes_tools(rp)
    else:
        rp = None  # ליתר ביטחון

    if show_features_section:
        st.subheader("מפת חום — פיצ'רי מאקרו")
        feat_df = state.macro_features_df or state_get(MacroStateKeys.FEATURES_DF)
        feat_fig = _render_macro_heatmap(feat_df)
        _features_tools(feat_df, params=feature_params, fig=feat_fig)

    if show_overlay_section:
        st.subheader("דירוג זוגות לפי Macro Fit")
        profile = state.macro_profile or state_get(MacroStateKeys.PROFILE)
        _render_pair_fit_table(profile)

        st.subheader("Overlay — משקולות ופעולות")
        _render_overlay_table(profile)
        _overlay_downloads(profile)
    else:
        profile = state.macro_profile or state_get(MacroStateKeys.PROFILE)

    # Macro Shocks
    if show_shocks_section:
        _render_macro_shocks_section(state)

    # DNA לזוג נבחר
    if show_dna_section:
        _render_pair_macro_dna_section(state, pairs_df)

    # ----------------------------------------------------------------------
    # הורדות של AdjustmentResult (JSON/YAML) לפי Feature Flags
    # ----------------------------------------------------------------------
    if get_flag("macro_download", default=False):
        try:
            snapshot = getattr(result, "regime_snapshot", None)
            if snapshot is not None:
                try:
                    snapshot = {
                        k: float(v)
                        for k, v in dict(snapshot).items()
                    }
                except Exception:  # noqa: BLE001
                    snapshot = dict(snapshot)  # type: ignore[arg-type]

            payload = {
                "meta": state_get(MacroStateKeys.META, {}),
                "result": {
                    "exposure_multiplier": float(
                        getattr(result, "exposure_multiplier", 1.0),
                    ),
                    "pair_adjustments": getattr(result, "pair_adjustments", {}),
                    "filters": getattr(result, "filters", {}),
                    "regime_snapshot": snapshot or {},
                    "pair_scores": getattr(result, "pair_scores", {}),
                    "caps_hints": getattr(result, "caps_hints", {}),
                },
            }
            st.download_button(
                label="⬇️ הורד תוצאת מאקרו (JSON)",
                data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="macro_adjustments_result.json",
                mime="application/json",
                key=keygen(TAB_KEY, "download"),
            )

            caps = getattr(result, "caps_hints", {})
            if isinstance(caps, dict) and caps:
                st.download_button(
                    label="⬇️ הורד Caps Hints (JSON)",
                    data=json.dumps(caps, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="macro_caps_hints.json",
                    mime="application/json",
                    key=keygen(TAB_KEY, "download", "caps"),
                )
        except Exception:  # noqa: BLE001
            LOGGER.debug("macro_download disabled due to error", exc_info=True)

    if get_flag("export_yaml", default=False) and _HAS_YAML:
        try:
            payload_yaml = {
                "meta": state_get(MacroStateKeys.META, {}),
                "result": {
                    "exposure_multiplier": float(
                        getattr(result, "exposure_multiplier", 1.0),
                    ),
                    "pair_adjustments": getattr(result, "pair_adjustments", {}),
                    "filters": getattr(result, "filters", {}),
                },
            }
            st.download_button(
                label="⬇️ הורד תוצאת מאקרו (YAML)",
                data=yaml.safe_dump(payload_yaml, allow_unicode=True).encode("utf-8"),
                file_name="macro_adjustments_result.yaml",
                mime="text/yaml",
                key=keygen(TAB_KEY, "download", "yaml"),
            )
        except Exception:  # noqa: BLE001
            LOGGER.debug("macro_yaml disabled due to error", exc_info=True)

    return result

def render_macro_tab(*args: Any, **kwargs: Any) -> None:
    """
    render_macro_tab — Universal Router Wrapper לטאב המאקרו (HF-grade, v5+)
    =======================================================================

    מה הפונקציה הזו עושה בפועל (גבוה):
    -----------------------------------
    1. Router חכם:
       • תומך בקריאה החדשה (Dashboard v4):
            render_macro_tab(app_ctx, feature_flags, nav_payload=None)
       • תומך בקריאה הישנה (legacy):
            render_macro_tab(pairs, config, ctx=None, ctrl_macro=None)

    2. ניהול קונטקסט מאקרו:
       • שומר nav_payload / macro_ctx / macro_panels ב-session_state.
       • יוצר macro_run_id + macro_entry_meta + macro_entry_telemetry.
       • מכוון Workspace:
            - env (backtest/live)
            - date_range לפי date_window (1Y/3Y/5Y/10Y/MAX).
       • מקבע דגלים חכמים:
            - macro_debug (ui_mode="power")
            - macro_show_regimes_table (macro_profile="risk")
            - macro_warn_pairs_quality / macro_show_source_summary (risk_mode=defensive/stress)
            - macro_read_only / risk_profile_from_macro / macro_preset_selected.

    3. שכבת Macro Factors (רעיון 1–3):
       • בונה DataFrame של פקטורי מאקרו מתוך MacroBundle.
       • מייצר MacroSnapshot (מצב מאקרו: rates/curve/vol/credit/fx/risk + risk_on_score).
       • מזהה shocks בפקטורים (detect_macro_shocks):
            - multi-window (למשל 1/5/20 ימים).
            - מתייג direction וחומרה.
       • שומר ב-session_state:
            - macro_factor_snapshot      (dict)
            - macro_factor_summary_text  (string קצר)
            - macro_factor_risk_on_score (float)
            - macro_factor_extreme_keys  (list)
            - macro_factor_shocks_meta   (summary ל־Risk/Portfolio/Agents).

    4. שכבת Macro Adjustments לזוגות (רעיון 4–7):
       • אם יש pairs_df ולא read_only:
            Regime → Exposure Multiplier → Pair Overlays:
            - macro_multiplier לכל זוג.
            - macro_include (פילטר).
            - macro_score (Macro Fit).
            - caps_hints (תקרות סקטור/אזור/מטבע).
       • שומר את overlay ב-session_state:
            - macro_pair_multipliers
            - macro_pair_filters
            - macro_pair_scores
            - macro_pair_caps_hints
            - macro_regime_snapshot / macro_regime_label
            - macro_risk_alert (לריסק אנג'ין).

    5. שכבת Meta ל-Universe / Risk Budget / Agents (רעיון 8–10):
       • macro_universe_meta:
            - n_pairs, universe_hash (hash של pair_idים).
       • macro_risk_budget_hint:
            - המלצה פשוטה ל-risk budget לפי exposure_multiplier + risk_mode.
       • macro_focus_pair_overlay:
            - overlay ממוקד לזוג אחד (focus_pair_id) לטובת DNA/Pair Tab/Agents.

    6. בסוף:
       • קוראת לפונקציה render(pairs_df, cfg, bundle) – מנוע המאקרו הפנימי
         שבונה את ה-UI של הטאב.
    """

    from datetime import date, datetime, timedelta  # שים לב: בלי timezone כדי לא לשבור

    # Helper פנימי: המרה מ-MacroBundle ל-DataFrame של פקטורים
    def _bundle_to_factor_df(bundle: MacroBundle) -> pd.DataFrame:
        series_map: Dict[str, pd.Series] = {}
        data = getattr(bundle, "data", {}) or {}
        for key, md in data.items():
            s = getattr(md, "series", None)
            if s is None or s.empty:
                continue
            s = s.copy()
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index, errors="coerce")
            series_map[key] = s.sort_index().astype(float)
        if not series_map:
            return pd.DataFrame()
        df = pd.DataFrame(series_map).sort_index()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        return df

    # ==========================
    # 1) ניסיון: סגנון חדש (app_ctx, feature_flags, nav_payload)
    # ==========================
    app_ctx = kwargs.get("app_ctx")
    feature_flags = kwargs.get("feature_flags")
    nav_payload = kwargs.get("nav_payload")

    # נסיון לחלץ app_ctx מתוך args אם לא הגיע ב-kwargs
    if app_ctx is None and args:
        candidate = args[0]
        if hasattr(candidate, "settings") or hasattr(candidate, "project_root"):
            app_ctx = candidate
            if len(args) > 1:
                feature_flags = args[1]
            if len(args) > 2:
                nav_payload = args[2]

    # סגנון חדש (Dashboard v4)
    if app_ctx is not None:
        # ---------- 1.A שמירת payloadים ב-session_state ----------
        macro_ctx: Dict[str, Any] = {}
        macro_panels: Dict[str, Any] = {}

        if isinstance(nav_payload, Mapping):
            try:
                state_set("macro_nav_payload", dict(nav_payload))
            except Exception:
                st.session_state["macro_nav_payload"] = dict(nav_payload)

            maybe_ctx = nav_payload.get("macro_ctx")
            if isinstance(maybe_ctx, Mapping):
                macro_ctx = dict(maybe_ctx)
                try:
                    state_set("macro_ctx_from_nav", macro_ctx)
                except Exception:
                    st.session_state["macro_ctx_from_nav"] = macro_ctx

            maybe_panels = nav_payload.get("macro_panels")
            if isinstance(maybe_panels, Mapping):
                macro_panels = dict(maybe_panels)
                try:
                    state_set("macro_panels_requested", macro_panels)
                except Exception:
                    st.session_state["macro_panels_requested"] = macro_panels
        else:
            nav_payload = {}
            macro_ctx = {}
            macro_panels = {}

        # ---------- 1.B run_id + Telemetry בסיסית ----------
        try:
            macro_run_id = str(uuid4())
            state_set("macro_run_id", macro_run_id)
        except Exception:
            macro_run_id = "unknown"

        env_from_ctx = macro_ctx.get("env") if isinstance(macro_ctx, Mapping) else None
        profile_from_ctx = macro_ctx.get("profile") if isinstance(macro_ctx, Mapping) else None

        if isinstance(feature_flags, Mapping):
            env_from_ctx = env_from_ctx or feature_flags.get("env")
            profile_from_ctx = profile_from_ctx or feature_flags.get("profile")

        ui_mode = macro_ctx.get("ui_mode") if isinstance(macro_ctx, Mapping) else None
        macro_profile = macro_ctx.get("macro_profile") if isinstance(macro_ctx, Mapping) else None
        risk_mode = macro_ctx.get("risk_mode") if isinstance(macro_ctx, Mapping) else None
        layout_profile = macro_ctx.get("layout_profile") if isinstance(macro_ctx, Mapping) else None
        default_view = (
            nav_payload.get("default_view")
            if isinstance(nav_payload, Mapping) and "default_view" in nav_payload
            else macro_ctx.get("default_view")
            if isinstance(macro_ctx, Mapping)
            else None
        )
        overlay_focus = (
            nav_payload.get("overlay_focus")
            if isinstance(nav_payload, Mapping) and "overlay_focus" in nav_payload
            else macro_ctx.get("overlay_focus")
            if isinstance(macro_ctx, Mapping)
            else None
        )
        date_window = (
            nav_payload.get("date_window")
            if isinstance(nav_payload, Mapping) and "date_window" in nav_payload
            else macro_ctx.get("date_window")
            if isinstance(macro_ctx, Mapping)
            else None
        )
        macro_preset = (
            nav_payload.get("macro_preset")
            if isinstance(nav_payload, Mapping) and "macro_preset" in nav_payload
            else macro_ctx.get("macro_preset")
            if isinstance(macro_ctx, Mapping)
            else None
        )
        focus_pair_id = (
            nav_payload.get("focus_pair_id")
            if isinstance(nav_payload, Mapping)
            else None
        )
        read_only_flag = bool(
            (isinstance(nav_payload, Mapping) and nav_payload.get("read_only"))
            or (isinstance(macro_ctx, Mapping) and macro_ctx.get("read_only"))
        )

        # Telemetry לקונטקסט כניסה (תיקון timezone: utcnow() + "Z")
        entry_meta = {
            "run_id": macro_run_id,
            "ts_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "env": env_from_ctx,
            "profile": profile_from_ctx,
            "macro_profile": macro_profile,
            "risk_mode": risk_mode,
            "ui_mode": ui_mode,
            "layout_profile": layout_profile,
            "default_view": default_view,
            "overlay_focus": overlay_focus,
            "date_window": date_window,
            "macro_preset": macro_preset,
            "panels_requested": list(macro_panels.keys()) if macro_panels else [],
        }
        try:
            state_set("macro_entry_meta", entry_meta)
        except Exception:
            st.session_state["macro_entry_meta"] = entry_meta

        try:
            state_set(
                "macro_entry_telemetry",
                {
                    "run_id": macro_run_id,
                    "env": env_from_ctx,
                    "profile": profile_from_ctx,
                    "macro_profile": macro_profile,
                    "risk_mode": risk_mode,
                    "dt_utc": entry_meta["ts_utc"],
                },
            )
        except Exception:
            pass

        # ---------- 1.C כיוון Workspace (env / date_range) ----------
        env_sidebar = "backtest"
        if str(env_from_ctx).lower() in {"prod", "live", "production"}:
            env_sidebar = "live"
        try:
            st.session_state[keygen(TAB_KEY, "ws", "env")] = env_sidebar
        except Exception:
            pass

        days_map = {"1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5, "10Y": 365 * 10}
        if isinstance(date_window, str):
            dw = date_window.upper()
            if dw in days_map:
                end_d = date.today()
                start_d = end_d - timedelta(days=days_map[dw])
            elif dw == "MAX":
                end_d = date.today()
                start_d = end_d - timedelta(days=365 * 15)
            else:
                start_d = end_d = None

            if start_d is not None and end_d is not None:
                try:
                    st.session_state[keygen(TAB_KEY, "ws", "date_range")] = (start_d, end_d)
                except Exception:
                    pass

        # ---------- 1.D דגלים חכמים (Debug / Regimes / Source Summary) ----------
        if ui_mode == "power":
            set_flag("macro_debug", True)
        if macro_profile == "risk":
            set_flag("macro_show_regimes_table", True)
        if risk_mode in {"defensive", "stress"}:
            set_flag("macro_warn_pairs_quality", True)
            set_flag("macro_show_source_summary", True)

        try:
            state_set("macro_read_only", bool(read_only_flag))
        except Exception:
            st.session_state["macro_read_only"] = bool(read_only_flag)

        if risk_mode:
            try:
                state_set("risk_profile_from_macro", str(risk_mode))
            except Exception:
                st.session_state["risk_profile_from_macro"] = str(risk_mode)

        if macro_preset:
            try:
                state_set("macro_preset_selected", str(macro_preset))
            except Exception:
                st.session_state["macro_preset_selected"] = str(macro_preset)

        if focus_pair_id:
            try:
                st.session_state[keygen(TAB_KEY, "dna", "pair")] = str(focus_pair_id)
            except Exception:
                pass

        should_autorefresh = False
        if env_sidebar == "live":
            try:
                if isinstance(feature_flags, Mapping) and feature_flags.get("market_data"):
                    should_autorefresh = True
            except Exception:
                pass
        try:
            state_set("macro_should_autorefresh", bool(should_autorefresh))
        except Exception:
            st.session_state["macro_should_autorefresh"] = bool(should_autorefresh)

        # ---------- 1.E Logging ----------
        LOGGER.info(
            "macro_tab.render_macro_tab[new] | run_id=%s | env=%s, profile=%s, macro_profile=%s, "
            "risk_mode=%s, ui_mode=%s, layout=%s, date_window=%s, preset=%s, read_only=%s",
            macro_run_id,
            env_from_ctx,
            profile_from_ctx,
            macro_profile,
            risk_mode,
            ui_mode,
            layout_profile,
            date_window,
            macro_preset,
            read_only_flag,
        )

        if macro_panels:
            try:
                active_panels = [k for k, v in macro_panels.items() if v]
            except Exception:
                active_panels = []
            LOGGER.debug(
                "macro_tab.render_macro_tab[new] | requested panels=%s",
                active_panels,
            )

        # ---------- 1.F מציאת pairs_df ממקורות שונים ----------
        pairs_df: Optional[pd.DataFrame] = None

        if isinstance(nav_payload, Mapping):
            maybe_df = nav_payload.get("pairs_df")
            if isinstance(maybe_df, pd.DataFrame):
                pairs_df = maybe_df

        if pairs_df is None and isinstance(nav_payload, Mapping):
            maybe_pairs = nav_payload.get("pairs")
            if isinstance(maybe_pairs, list):
                pairs_df = _pairs_list_to_df_for_tab(maybe_pairs)

        if pairs_df is None and hasattr(app_ctx, "pairs_df"):
            maybe = getattr(app_ctx, "pairs_df")
            if isinstance(maybe, pd.DataFrame):
                pairs_df = maybe

        if pairs_df is None and hasattr(app_ctx, "universe"):
            maybe = getattr(app_ctx, "universe")
            if isinstance(maybe, pd.DataFrame):
                pairs_df = maybe

        if pairs_df is None:
            maybe = state_get("pairs_df")
            if isinstance(maybe, pd.DataFrame):
                pairs_df = maybe

        if pairs_df is None:
            LOGGER.debug(
                "macro_tab.render_macro_tab[new] | run_id=%s | pairs_df not found → running in 'macro-only' mode.",
                macro_run_id,
            )
        else:
            LOGGER.debug(
                "macro_tab.render_macro_tab[new] | run_id=%s | using pairs_df with %s rows.",
                macro_run_id,
                len(pairs_df),
            )

        # ---------- 1.G בניית MacroConfig + MacroBundle ----------
        settings = getattr(app_ctx, "settings", None)
        if settings is not None and hasattr(settings, "macro"):
            macro_cfg: MacroConfig = getattr(settings, "macro")  # type: ignore[assignment]
        else:
            macro_cfg = MacroConfig()

        if macro_profile and hasattr(macro_cfg, "macro_profile"):
            macro_cfg.macro_profile = macro_profile  # type: ignore[attr-defined]

        bundle: MacroBundle = load_macro_bundle(macro_cfg)

        # ---------- 1.H שכבת Macro Factors (Snapshot + Shocks) ----------
        factor_df = _bundle_to_factor_df(bundle)
        factor_snapshot = None
        factor_summary_text = None
        factor_shocks_meta: Dict[str, Any] = {}

        if not factor_df.empty:
            try:
                factor_snapshot = build_macro_snapshot(factor_df, cfg=None)  # מ-macro_factors
                factor_summary_text = summarize_macro_snapshot(factor_snapshot)
            except Exception:
                factor_snapshot = None
                factor_summary_text = None

            try:
                shocks_df = detect_macro_shocks(
                    factor_df,
                    cfg=None,
                    windows=(1, 5, 20),
                    z_threshold=3.0,
                    max_events_per_factor=50,
                )
            except Exception:
                shocks_df = pd.DataFrame()

            # שמירת snapshot + summary + shocks ב-session_state
            if factor_snapshot is not None:
                snap_dict = factor_snapshot.to_dict()
                try:
                    state_set("macro_factor_snapshot", snap_dict)
                except Exception:
                    st.session_state["macro_factor_snapshot"] = snap_dict

                try:
                    state_set("macro_factor_summary_text", factor_summary_text or "")
                except Exception:
                    st.session_state["macro_factor_summary_text"] = factor_summary_text or ""

                # רעיון: רמת risk-on אוהדת / עוינת לכל המערכת
                try:
                    risk_on_score = snap_dict.get("risk_on_score", 0.0)
                except Exception:
                    risk_on_score = 0.0
                try:
                    state_set("macro_factor_risk_on_score", float(risk_on_score))
                except Exception:
                    st.session_state["macro_factor_risk_on_score"] = float(risk_on_score)

                # פקטורים קיצוניים (לבלון/באנר)
                extreme_factors = snap_dict.get("extreme_factors") or []
                try:
                    state_set("macro_factor_extreme_keys", list(extreme_factors))
                except Exception:
                    st.session_state["macro_factor_extreme_keys"] = list(extreme_factors)

            if shocks_df is not None and not shocks_df.empty:
                # Summary קצר של shocks לפי group/פקטור
                try:
                    last_ts = shocks_df.index.max()
                    top_events = shocks_df.sort_values("severity_rank", ascending=False).head(20)
                    factor_shocks_meta = {
                        "last_ts": last_ts.isoformat(),
                        "n_events": int(len(shocks_df)),
                        "top_factors": list(top_events["factor_key"].value_counts().head(5).index),
                    }
                    state_set("macro_factor_shocks_meta", factor_shocks_meta)
                except Exception:
                    st.session_state["macro_factor_shocks_meta"] = factor_shocks_meta

        # ---------- 1.I שכבת Macro Adjustments לזוגות ----------
        adj = None
        if pairs_df is not None and not read_only_flag:
            try:
                pairs_df = pairs_df.copy()
                adj = compute_adjustments(pairs_df, bundle, macro_cfg)
            except Exception:
                LOGGER.exception("macro_tab.render_macro_tab[new]: compute_adjustments failed")
                adj = None

        # שמירת תוצאות המאקרו ב-session_state לשימוש בטאבים אחרים
        if adj is not None:
            try:
                scores = list(adj.pair_scores.values()) if adj.pair_scores else []
                mean_score = sum(scores) / len(scores) if scores else None
            except Exception:
                mean_score = None

            try:
                state_set(
                    "macro_last_adjustments",
                    {
                        "run_id": macro_run_id,
                        "ts_utc": entry_meta["ts_utc"],
                        "regime_label": adj.regime_label,
                        "exposure_multiplier": adj.exposure_multiplier,
                        "n_pairs": len(adj.pair_adjustments),
                        "included": int(sum(1 for v in adj.filters.values() if v)),
                        "mean_macro_score": mean_score,
                    },
                )
            except Exception:
                st.session_state["macro_last_adjustments"] = {
                    "run_id": macro_run_id,
                    "regime_label": adj.regime_label,
                    "exposure_multiplier": adj.exposure_multiplier,
                }

            # overlay ברמת זוג
            try:
                state_set("macro_pair_multipliers", adj.pair_adjustments)
                state_set("macro_pair_filters", adj.filters)
                state_set("macro_pair_scores", adj.pair_scores)
                state_set("macro_pair_caps_hints", adj.caps_hints)
            except Exception:
                st.session_state["macro_pair_multipliers"] = adj.pair_adjustments
                st.session_state["macro_pair_filters"] = adj.filters
                st.session_state["macro_pair_scores"] = adj.pair_scores
                st.session_state["macro_pair_caps_hints"] = adj.caps_hints

            # snapshot של Regime
            if adj.regime_snapshot is not None:
                try:
                    state_set("macro_regime_snapshot", adj.regime_snapshot.to_dict())
                except Exception:
                    st.session_state["macro_regime_snapshot"] = adj.regime_snapshot.to_dict()
            if adj.regime_label is not None:
                try:
                    state_set("macro_regime_label", adj.regime_label)
                except Exception:
                    st.session_state["macro_regime_label"] = adj.regime_label

            # רעיון: macro_risk_alert לריסק אנג'ין
            risk_alert = False
            try:
                if adj.exposure_multiplier is not None and adj.exposure_multiplier < 0.8:
                    risk_alert = True
                if str(adj.regime_label or "").startswith("risk_off"):
                    risk_alert = True
            except Exception:
                pass
            try:
                state_set("macro_risk_alert", bool(risk_alert))
            except Exception:
                st.session_state["macro_risk_alert"] = bool(risk_alert)

            # meta על ה-universe (גודל ו-hash לפי pair_id)
            try:
                if pairs_df is not None:
                    if "pair_id" in pairs_df.columns:
                        ids = list(pairs_df["pair_id"].astype(str))
                    else:
                        ids = [f"{row.get('a','?')}-{row.get('b','?')}" for _, row in pairs_df.iterrows()]
                    universe_hash = hash(tuple(ids))
                    universe_meta = {
                        "run_id": macro_run_id,
                        "n_pairs": len(ids),
                        "universe_hash": universe_hash,
                    }
                    state_set("macro_universe_meta", universe_meta)
            except Exception:
                pass

            # focus_pair_id → overlay ממוקד לזוג אחד
            if focus_pair_id is not None:
                pid = str(focus_pair_id)
                focused = {
                    "pair_id": pid,
                    "macro_multiplier": adj.pair_adjustments.get(pid),
                    "macro_include": adj.filters.get(pid),
                    "macro_score": adj.pair_scores.get(pid),
                    "caps_hint": adj.caps_hints.get(pid),
                }
                try:
                    state_set("macro_focus_pair_overlay", focused)
                except Exception:
                    st.session_state["macro_focus_pair_overlay"] = focused

            # רעיון: macro_risk_budget_hint – המלצה לגודל סיכון (0–1)
            try:
                exp_mult = float(adj.exposure_multiplier or 1.0)
            except Exception:
                exp_mult = 1.0
            base_budget = 1.0
            if risk_mode in {"defensive", "stress"}:
                base_budget = 0.7
            elif risk_mode in {"aggressive"}:
                base_budget = 1.3
            risk_budget_hint = max(0.3, min(1.5, base_budget * exp_mult))
            try:
                state_set("macro_risk_budget_hint", risk_budget_hint)
            except Exception:
                st.session_state["macro_risk_budget_hint"] = risk_budget_hint

        # גם אם אין pairs_df או read_only, עדיין קוראים ל-render עם cfg+bundle
        try:
            render(pairs_df=pairs_df, cfg=macro_cfg, bundle=bundle)
        except TypeError:
            render(pairs_df=pairs_df, cfg=None, bundle=None)

        LOGGER.info(
            "macro_tab.render_macro_tab[new] | run_id=%s | done",
            macro_run_id,
        )
        return

    # ==========================
    # 2) Legacy style (pairs, config, ctx=None, ctrl_macro=None)
    # ==========================
    if not args:
        LOGGER.error(
            "render_macro_tab called without app_ctx or pairs (no arguments); running macro-only.",
        )
        render(pairs_df=None, cfg=None, bundle=None)
        return

    pairs = args[0]
    try:
        pairs_df_legacy = _pairs_list_to_df_for_tab(pairs)
    except Exception:
        LOGGER.exception(
            "render_macro_tab[legacy]: failed to convert pairs→pairs_df; falling back to empty universe.",
        )
        pairs_df_legacy = None

    LOGGER.info(
        "macro_tab.render_macro_tab[legacy] called with %s pairs",
        len(pairs) if isinstance(pairs, Sequence) else "unknown",
    )

    # גם במצב legacy – נריץ שכבת מאקרו בסיסית אם יש pairs_df
    if pairs_df_legacy is not None and len(pairs_df_legacy) > 0:
        legacy_run_id = f"legacy-{uuid4()}"
        macro_cfg = MacroConfig()
        bundle = load_macro_bundle(macro_cfg)
        try:
            adj = compute_adjustments(pairs_df_legacy, bundle, macro_cfg)
            state_set(
                "macro_last_adjustments_legacy",
                {
                    "run_id": legacy_run_id,
                    "regime_label": adj.regime_label,
                    "exposure_multiplier": adj.exposure_multiplier,
                    "n_pairs": len(adj.pair_adjustments),
                },
            )
        except Exception:
            LOGGER.exception("macro_tab.render_macro_tab[legacy]: compute_adjustments failed")
            bundle = None
            macro_cfg = None  # type: ignore[assignment]
        render(pairs_df=pairs_df_legacy, cfg=macro_cfg, bundle=bundle)
    else:
        render(pairs_df=None, cfg=None, bundle=None)


def _safe_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if not (v == v):  # NaN check
            return None
        return v
    except Exception:
        return None

def push_macro_metrics_to_ctx(
    *,
    macro_sensitivity: Any,
    macro_score: Any,
    max_dd_60d: Any = None,
    vol_60d: Any = None,
    ctx_key: str = "macro_metrics",
) -> None:
    """
    שומר מדדי מאקרו רלוונטיים ל-Tab Comparison בתוך st.session_state[ctx_key].

    מפתחי ה-keys מותאמים ל-render_tab_comparison_lab:
      - "macro_sensitivity" / "macro_regime_score"
      - "macro_score" / "tab_score"
      - "max_dd_60d" / "macro_max_dd"
      - "vol_60d" / "macro_vol_60d"
    """
    metrics: Dict[str, float] = {}


    ms = _safe_float_or_none(macro_sensitivity)
    if ms is not None:
        metrics["macro_sensitivity"] = ms
        metrics["macro_regime_score"] = ms  # alias אופציונלי

    sc = _safe_float_or_none(macro_score)
    if sc is not None:
        metrics["macro_score"] = sc
        metrics["tab_score"] = sc  # alias כללי

    dd = _safe_float_or_none(max_dd_60d)
    if dd is not None:
        metrics["max_dd_60d"] = dd
        metrics["macro_max_dd"] = dd

    vol = _safe_float_or_none(vol_60d)
    if vol is not None:
        metrics["vol_60d"] = vol
        metrics["macro_vol_60d"] = vol

    st.session_state[ctx_key] = metrics

def push_risk_metrics_to_ctx(
    result: AdjustmentResult,
    meta: Mapping[str, Any],
    ctx_key: str = "risk_metrics",
) -> None:
    """
    שומר מדדי סיכון גלובליים ל-Tab Comparison בתוך st.session_state[ctx_key].

    מבוסס על:
      - result.exposure_multiplier
      - meta['pairs'] / meta['included'] מ-_stamp_result
      - risk_profile_from_macro (אם קיים ב-session_state)
    """
    metrics: Dict[str, float] = {}

    exp_mult = _safe_float_or_none(getattr(result, "exposure_multiplier", 1.0))
    if exp_mult is None:
        exp_mult = 1.0

    pairs = _safe_float_or_none(meta.get("pairs"))
    included = _safe_float_or_none(meta.get("included"))

    if pairs is not None and pairs > 0 and included is not None:
        inclusion_ratio = max(0.0, min(1.0, included / pairs))
    else:
        inclusion_ratio = 1.0

    # risk_score פשוט: כמה חשופים × כמה זוגות משחקים בפנים
    risk_score = 100.0 * inclusion_ratio * min(exp_mult, 2.0) / 2.0

    metrics["risk_exposure"] = float(exp_mult)
    metrics["risk_inclusion_ratio"] = float(inclusion_ratio)
    metrics["risk_score"] = float(risk_score)

    # ננסה למשוך גם risk_profile מהמאקרו (אם הגיע מה-dashboard)
    risk_profile = state_get("risk_profile_from_macro", None)
    if risk_profile is not None:
        metrics["risk_profile"] = float("nan")  # רק placeholder מספרי
        # נשמור גם את הטקסט במפתח נפרד
        try:
            # st.session_state מאפשר גם non-float values, אז נשים dict מורכב
            pass
        except Exception:
            pass

    st.session_state[ctx_key] = metrics
    # בנוסף, נשמור טקסטואלי נפרד אם רוצים:
    if risk_profile is not None:
        st.session_state[f"{ctx_key}_label"] = str(risk_profile)

def _get_macro_panels_flags() -> Dict[str, bool]:
    """
    מחזיר dict של פאנלים מבוקשים מה-dashboard (אם קיים),
    לפי מה שה-router כתב ב-session_state["macro_panels_requested"].
    אם אין כלום – מחזיר dict ריק, והטאב מתנהג לפי ברירת המחדל שלו.
    """
    panels_req = state_get("macro_panels_requested", None)
    if isinstance(panels_req, Mapping):
        try:
            return {str(k): bool(v) for k, v in panels_req.items()}
        except Exception:  # noqa: BLE001
            return {}
    return {}

# ---------------------------------------------------------------------------
# תאימות ל-pd.np (לשימוש לוגאריתמי בתוך _build_spread_series אם צריך)
# ---------------------------------------------------------------------------
try:  # התאמה לאחור במקרה שמשתמשים ב-pd.np.log
    import numpy as _np  # type: ignore

    if not hasattr(pd, "np"):
        pd.np = _np  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass


__all__ = ["render", "render_macro_tab"]

