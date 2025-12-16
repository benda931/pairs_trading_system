# -*- coding: utf-8 -*-
"""
common/macro_adjustments.py — מודול התאמות מאקרו (לשימוש מהטאב ב-root)
---------------------------------------------------------------------
מטרת הקובץ: שכבת מאקרו אחת שמבצעת:
1) טעינת נתוני מאקרו (CSV/מקורות אחרים — כרגע local:*.csv; ניתן להרחבה),
2) בניית מדדי Regime (Risk-On / Inflation / Growth) חלקים,
3) חישוב מקדמי התאמה גלובליים וברמת זוגות (multipliers + filters),
4) API ידידותי ל-UI (`render_streamlit_ui`) ללא כפילות לוגיקה.

הקובץ כתוב כך שהטאב (`root/macro_tab.py`) רק מייבא וקורא, ללא לוגיקה כפולה.
שומר על חתימות יציבות ותלוי אופציונלית ב-Pydantic ו-Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd

# ===== Optional dependencies (graceful) =====
try:  # Pydantic v2 (רצוי)
    from pydantic import BaseModel, Field, ConfigDict  # type: ignore
    _HAS_PYDANTIC = True
except Exception:  # noqa: BLE001
    BaseModel = object  # type: ignore
    Field = lambda default=None, **_: default  # type: ignore
    ConfigDict = dict  # type: ignore
    _HAS_PYDANTIC = False

try:  # Streamlit ל-UI בלבד
    import streamlit as st  # type: ignore
    _HAS_STREAMLIT = True
except Exception:  # noqa: BLE001
    _HAS_STREAMLIT = False

# ===== Logging =====
logger = logging.getLogger("common.macro_adjustments")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# =====================================================================
#                           קונפיגורציה
# =====================================================================
class _PydModel(BaseModel):
    if _HAS_PYDANTIC:
        model_config = ConfigDict(validate_assignment=True, extra="ignore")  # type: ignore


# =====================================================================
#                          Macro Layer Config
# =====================================================================

class MacroConfig(_PydModel if _HAS_PYDANTIC else object):
    """הגדרות שכבת המאקרו.

    אם Pydantic זמין — נקבל ולידציה; אחרת זו מעטפת נתונים פשוטה.
    שכבה זו מייצגת את ה-**מקרו גלובלי**: אילו אינדיקטורים פעילים,
    פרמטרי חישוב, מגבלות סקטור/אזור/מטבע, וכו'.
    """

    # האם שכבת המאקרו פעילה בכלל
    enabled: bool = Field(True, description="הפעלת שכבת המאקרו")

    # פרופיל מאקרו ברירת מחדל (סגנון פעולה)
    macro_profile: Literal[
        "default",
        "defensive",
        "balanced",
        "aggressive",
        "crisis",
        "reflation",
    ] = Field("default", description="פרופיל מאקרו כללי (משפיע על פרשנות הפקטורים והרגישות)")

    # אופן גילוי המשטרים (רק לוגיקה חיצונית משתמשת בזה)
    regime_detection_mode: Literal[
        "simple_scores",
        "pca",
        "ml_classifier",
    ] = Field(
        "simple_scores",
        description="שיטת גילוי משטרים: על בסיס ציונים פשוטים / PCA / מודל ML חיצוני",
    )

    # חלון ברירת מחדל לכל החישובים
    lookback_days: int = Field(365, description="חלון ברירת מחדל לחישובים (ימים)")

    # ------------------------------------------------------------
    # בחירת אינדיקטורים (toggle) — מה בכלל נכנס ל-Macro Bundle
    # ------------------------------------------------------------
    use_cpi: bool = Field(True)
    use_unemployment: bool = Field(True)
    use_policy_rate: bool = Field(True)
    use_yield_curve: bool = Field(True)
    use_pmi: bool = Field(True)
    use_credit_spread: bool = Field(True)
    use_oil: bool = Field(True)
    use_dxy: bool = Field(True)
    use_vix: bool = Field(True)

    # ------------------------------------------------------------
    # פרמטרי החלקה/ספים
    # ------------------------------------------------------------
    regime_smoothing_days: int = Field(21, description="מס' ימים להחלקת ציוני משטר (rolling mean)")
    exposure_multiplier_bounds: Tuple[float, float] = Field(
        (0.5, 1.5),
        description="גבולות מכפיל חשיפה שמאקרו יכול להחיל (min,max)",
    )
    filter_threshold: float = Field(
        0.0,
        description="סף סינון לזוגות רגישים מאוד: מתחת לסף אפשר לסנן זוג או להוריד חשיפה",
    )

    # ------------------------------------------------------------
    # אופן היישום למערכת (איך המאקרו מתערב)
    # ------------------------------------------------------------
    apply_mode: Literal["exposure_only", "filter_only", "hybrid"] = Field(
        "hybrid",
        description="exposure_only: משנה חשיפות; filter_only: משמש כסינון; hybrid: גם וגם",
    )

    # ------------------------------------------------------------
    # רגישות סקטוריאלית (hintים ל-risk engine / portfolio)
    # ------------------------------------------------------------
    sector_sensitivity: Dict[str, float] = Field(
        {
            "Tech": 0.9,
            "Financials": 0.7,
            "Energy": 1.0,
            "Consumer": 0.6,
            "Industrials": 0.5,
            "Utilities": 0.4,
            "HealthCare": 0.5,
            "Materials": 0.6,
            "RealEstate": 0.5,
            "Communication": 0.7,
        },
        description="רגישות סקטוריאלית לשינויי משטר (משמש כרמז ל-Risk/Portfolio)",
    )

    # ------------------------------------------------------------
    # מקורות נתונים לוגיים לכל אינדיקטור
    # ------------------------------------------------------------
    sources: Dict[str, str] = Field(
        {
            "cpi": "local:cpi.csv",
            "unemployment": "local:unemployment.csv",
            "policy_rate": "local:policy_rate.csv",
            "yield_curve": "local:yc_10y_2y.csv",
            "pmi": "local:pmi.csv",
            "credit_spread": "local:hy_oas.csv",
            "oil": "local:brent.csv",
            "dxy": "local:dxy.csv",
            "vix": "local:vix.csv",
        },
        description="מפה: key לוגי → מקור נתונים (local:/duckdb:/sql:/etc.)",
    )

    # ------------------------------------------------------------
    # פרמטרים מתקדמים ברמת התיק
    # ------------------------------------------------------------
    volatility_target: float = Field(
        0.15,
        description="יעד תנודתיות שנתי (לדוגמה 0.15=15%) — רלוונטי לרובד ה-Risk/Portfolio",
    )
    cvar_limit: float = Field(
        0.10,
        description="מגבלת CVaR ברמת התיק (יחסי) — לוגיקה חיצונית יכולה להשתמש בזה כגבול",
    )
    max_pair_weight: float = Field(
        0.05,
        description="מקסימום משקל לזוג ביחס לתיק (יחסי, 0..1)",
    )
    sector_caps: Dict[str, float] = Field(
        {},
        description="תקרות חשיפה לסקטורים (0..1)",
    )
    region_caps: Dict[str, float] = Field(
        {},
        description="תקרות חשיפה לאזורים (0..1)",
    )
    currency_caps: Dict[str, float] = Field(
        {},
        description="תקרות חשיפה למטבעות (0..1)",
    )

    hysteresis_days: int = Field(
        5,
        description="היסטרזיס לשינוי משטר/פעולה (מס' ימים לפחות לפני שמחליפים החלטה)",
    )
    min_regime_duration: int = Field(
        10,
        description="משך מינימלי למשטר לפני שניתן לעבור למשטר אחר",
    )
    shock_sensitivity: float = Field(
        1.0,
        description="רגישות פונקציית tanh לשוקי-הלם / שינויים חדים במדדים",
    )
    kill_switch_drawdown: float = Field(
        0.20,
        description="כיבוי מוחלט (hint) אם DD מצטבר חוצה סף זה — מימוש ברובד Risk",
    )
    liquidity_min_adv: float = Field(
        1e6,
        description="מינימום ADV למסחר (hint ל-screening של זוגות)",
    )

    data_lag_policy: Literal["truncate", "ffill", "bfill"] = Field(
        "ffill",
        description="מדיניות לטיפול בפערי נתונים (חסרים) במקרו",
    )
    pca_components: int = Field(
        0,
        description="מס' רכיבי PCA לחישוב פקטורי מקרו סינתטיים (0=כבוי)",
    )
    zscore_min_periods: int = Field(
        60,
        description="חלון מינימלי לחישוב Z-score (מספר נקודות לפחות ברולינג)",
    )
    winsor_limit: float = Field(
        3.0,
        description="גזירת קצוות לחישוב Z-score (כמה סטיות תקן לחיתוך קיצוניות)",
    )

    # משקלי ערוצי משטר (risk_on, inflation, growth) להשפעה על חשיפה
    regime_weights: Tuple[float, float, float] = Field(
        (0.5, -0.2, 0.4),
        description="משקלי (risk_on, inflation, growth) לחשיפה מאקרו",
    )
    sensitivity_default: float = Field(
        0.6,
        description="רגישות ברירת מחדל לזוג אם לא סופקה רגישות אחרת",
    )

    # ------------------------------------------------------------
    # per-factor overrides לחישובי MacroFactorConfig
    # ------------------------------------------------------------
    factor_weights: Dict[str, float] = Field(
        {},
        description="override למשקל פקטור (weight) לפי key, למשל {'vix': 1.5}",
    )
    factor_signs: Dict[str, int] = Field(
        {},
        description="override לסימן (+1/-1/0) לפי key, למשל {'dxy': -1}",
    )
    factor_categories: Dict[str, str] = Field(
        {},
        description="override לקטגוריה לוגית לפי key, אם רוצים להחליף את ברירת המחדל",
    )

    # ------------------------------------------------------------
    # טעינת מקורות מורחבים
    # ------------------------------------------------------------
    allow_duckdb: bool = Field(
        True,
        description="אפשר טעינת duckdb:* אם duckdb מותקן",
    )
    allow_sql: bool = Field(
        False,
        description="אפשר טעינת sql:* (מומלץ False כברירת מחדל)",
    )
    duckdb_safe_readonly: bool = Field(
        True,
        description="פתח חיבור DuckDB לקריאה בלבד אם נתמך",
    )

    # Caps hints mode: רק מחזיר רמזים, לא משנה multipliers כאן
    caps_mode: Literal["hint", "clip"] = Field(
        "hint",
        description="'hint' מחזיר מכפיל תקרה, 'clip' יחיל חיתוך עדין בצד המאקרו עצמו",
    )

    # ------------------------------------------------------------
    # Live data client (core.macro_data)
    # ------------------------------------------------------------
    use_data_client: bool = Field(
        False,
        description="השתמש ב-core.macro_data.MacroDataClient לטעינת מקורות במקום CSV בלבד",
    )
    data_client_live: bool = Field(
        False,
        description="עקוף Cache והבא נתונים חיים בכל קריאה (יקר יותר)",
    )
    data_client_interval: Optional[str] = Field(
        None,
        description="Interval ל-yfinance (למשל '1d','1h','5m')",
    )
    data_client_period: Optional[str] = Field(
        None,
        description="Period ל-yfinance (למשל '1y','5y','max')",
    )
    data_client_allow_yf: bool = Field(
        True,
        description="אפשר yfinance בלקוח הדאטה",
    )
    data_client_allow_duckdb: bool = Field(
        True,
        description="אפשר duckdb בלקוח הדאטה",
    )
    data_client_allow_sql: bool = Field(
        False,
        description="אפשר sql בלקוח הדאטה",
    )
    # IBKR Client Portal
    data_client_allow_ibkr: bool = Field(
        True,
        description="אפשר IBKR בלקוח הדאטה",
    )
    ibkr_token: Optional[str] = Field(
        None,
        description="Portal access token (נשמר בזמן ריצה בלבד)",
    )
    ibkr_base_url: str = Field(
        "https://localhost:5000",
        description="IBKR Client Portal base URL",
    )


# =====================================================================
#                     Per-factor configuration (MacroFactorConfig)
# =====================================================================

class MacroFactorConfig(_PydModel if _HAS_PYDANTIC else object):
    """
    קונפיגורציה לפקטור מאקרו יחיד (CPI, Unemployment, VIX, וכו').

    זה משלים את MacroConfig (השכבה הגלובלית) ומייצג יחידה אחת שניתן
    להציג ב-UI, לנתח, ולהשתמש בה להסקת משטר/חשיפה.
    """

    key: str                                   # מפתח לוגי: "cpi", "unemployment", "vix"...
    name: str                                  # שם להצגה ב-UI

    enabled: bool = True                       # האם הפקטור נכנס לחישובים
    category: Literal[
        "inflation",
        "growth",
        "policy",
        "rates",
        "credit",
        "fx",
        "vol",
        "commodity",
        "other",
    ] = "other"

    # sign: איך ערך גבוה משפיע על risk-on/risk-off
    # +1: ערך גבוה = risk-on, -1: ערך גבוה = risk-off, 0: נייטרלי / mixed
    sign: int = 0

    # weight: משקל יחסי בחישובי חשיפה / ציון משטר
    weight: float = 1.0

    # source: מתוך MacroConfig.sources (זיהוי מקור)
    source: str = ""

    # overrides לוגיים מה-MacroConfig (אם None → נלקחים מהקונפיג הגלובלי)
    lookback_days: Optional[int] = None
    smoothing_days: Optional[int] = None
    zscore_min_periods: Optional[int] = None
    winsor_limit: Optional[float] = None


def build_default_macro_factors(cfg: MacroConfig) -> Dict[str, MacroFactorConfig]:
    """
    בונה אוסף פקטורים (MacroFactorConfig) מתוך MacroConfig אחד.

    משתמש ב:
      - toggles (use_cpi, use_unemployment, ...)
      - cfg.sources[key] לכל פקטור
      - factor_weights / factor_signs / factor_categories כ-overrides
      - lookback_days, regime_smoothing_days, zscore_min_periods, winsor_limit כערכי ברירת מחדל
    """
    factors: Dict[str, MacroFactorConfig] = {}

    def _mk(
        key: str,
        name: str,
        enabled_flag: bool,
        default_category: str,
        default_sign: int,
        default_weight: float = 1.0,
    ) -> None:
        src = cfg.sources.get(key)
        if not enabled_flag or not src:
            return

        category_str = cfg.factor_categories.get(key, default_category)
        # cast לקטגוריה חוקית; אם לא קיים — "other"
        if category_str not in {
            "inflation",
            "growth",
            "policy",
            "rates",
            "credit",
            "fx",
            "vol",
            "commodity",
            "other",
        }:
            category_str = "other"

        sign = cfg.factor_signs.get(key, default_sign)
        weight = cfg.factor_weights.get(key, default_weight)

        factors[key] = MacroFactorConfig(
            key=key,
            name=name,
            enabled=True,
            category=category_str,        # type: ignore[arg-type]
            sign=sign,
            weight=weight,
            source=src,
            lookback_days=cfg.lookback_days,
            smoothing_days=cfg.regime_smoothing_days,
            zscore_min_periods=cfg.zscore_min_periods,
            winsor_limit=cfg.winsor_limit,
        )

    # הגדרת פקטורים סטנדרטיים
    _mk("cpi",           "CPI Inflation",        cfg.use_cpi,           "inflation", -1)
    _mk("unemployment",  "Unemployment Rate",    cfg.use_unemployment,  "growth",    -1)
    _mk("policy_rate",   "Policy Rate",          cfg.use_policy_rate,   "policy",    -1)
    _mk("yield_curve",   "Yield Curve 10Y-2Y",   cfg.use_yield_curve,   "rates",     -1)
    _mk("pmi",           "PMI",                  cfg.use_pmi,           "growth",     1)
    _mk("credit_spread", "HY Credit Spread",     cfg.use_credit_spread, "credit",    -1)
    _mk("oil",           "Oil (Brent)",          cfg.use_oil,           "commodity",  1)
    _mk("dxy",           "DXY (USD)",            cfg.use_dxy,           "fx",        -1)
    _mk("vix",           "VIX",                  cfg.use_vix,           "vol",       -1)

    return factors


# =====================================================================
#                          טעינת נתונים
# =====================================================================
class MacroData:
    """עטיפת Series מסוג תאריך→ערך."""
    def __init__(self, name: str, series: pd.Series):
        self.name = name
        s = series.copy()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        self.series = s.sort_index().astype(float)

    def last(self, n: int = 1) -> pd.Series:
        return self.series.tail(n)


class MacroBundle:
    """אוסף סידרות מאקרו לשימוש החישובים."""
    def __init__(self, data: Dict[str, MacroData]):
        self.data = data

    def get(self, key: str) -> Optional[MacroData]:
        return self.data.get(key)

    def available(self) -> List[str]:
        return list(self.data.keys())


def _load_series_from_local_csv(path: str, value_col: Optional[str] = None) -> pd.Series:
    df = pd.read_csv(path)
    # איתור עמודת תאריך
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    if value_col is None:
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not num_cols:
            raise ValueError(f"No numeric column found in {path}")
        value_col = num_cols[0]
    return df[value_col].astype(float).rename(value_col)


def _load_macro_source(uri: str) -> pd.Series:
    """טוען מקור לפי פרוטוקול לוגי.

    נתמך:
      - local:PATH.csv
      - duckdb:DB_PATH|TABLE  (דורש duckdb מותקן; קריאה בלבד)
      - sql:CONNSTR|QUERY     (כבוי כברירת מחדל; שימוש זהיר)
    """
    if uri.startswith("local:"):
        rel = uri.split("local:", 1)[1]
        return _load_series_from_local_csv(rel)

    if uri.startswith("duckdb:"):
        try:
            import duckdb  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ImportError("duckdb לא מותקן בסביבה") from e
        spec = uri.split("duckdb:", 1)[1]
        if "|" not in spec:
            raise ValueError("duckdb URI חייב לכלול פורמט: duckdb:DB_PATH|TABLE")
        db_path, table = spec.split("|", 1)
        con = duckdb.connect(database=db_path)
        try:
            df = con.execute(f"SELECT * FROM {table}").fetchdf()
        finally:
            con.close()
        date_col = None
        for c in df.columns:
            if str(c).lower() in ("date", "time", "timestamp"):
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not num_cols:
            raise ValueError("duckdb source must include a numeric column")
        return df[num_cols[0]].astype(float)

    if uri.startswith("sql:"):
        # סכימה בסיסית: sql:CONNSTR|QUERY — אחריות המשתמש לבחור שאילתא שמחזירה date,value
        spec = uri.split("sql:", 1)[1]
        if "|" not in spec:
            raise ValueError("sql URI חייב לכלול פורמט: sql:CONNSTR|QUERY")
        conn_str, query = spec.split("|", 1)
        try:
            import sqlalchemy as sa  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ImportError("sqlalchemy לא מותקן בסביבה") from e
        engine = sa.create_engine(conn_str)
        try:
            df = pd.read_sql_query(query, engine)
        finally:
            engine.dispose()
        date_col = None
        for c in df.columns:
            if "date" in str(c).lower() or "time" in str(c).lower():
                date_col = c
                break
        if date_col is None:
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not num_cols:
            raise ValueError("sql source must include a numeric column")
        return df[num_cols[0]].astype(float)

    raise NotImplementedError(f"Unsupported macro source uri: {uri}")


def _maybe_client(cfg: MacroConfig):
    """בונה MacroDataClient אם זמין ומאופשר ב-cfg; אחרת מחזיר None."""
    if not getattr(cfg, "use_data_client", False):
        return None
    try:
        from core.macro_data import MacroDataClient  # type: ignore
    except Exception:
        logger.debug("MacroDataClient לא זמין — נופל חן לטעינה הישנה")
        return None
    return MacroDataClient(
        sources=dict(cfg.sources),
        allow_duckdb=bool(getattr(cfg, "data_client_allow_duckdb", True)),
        allow_sql=bool(getattr(cfg, "data_client_allow_sql", False)),
        allow_yf=bool(getattr(cfg, "data_client_allow_yf", True)),
        allow_ibkr=bool(getattr(cfg, "data_client_allow_ibkr", True)),
        ibkr_token=getattr(cfg, "ibkr_token", None),
        ibkr_base_url=str(getattr(cfg, "ibkr_base_url", "https://localhost:5000")),
    )


def _load_series_with_cfg(uri: str, cfg: MacroConfig) -> pd.Series:
    """טוען Series לפי uri עם עדיפות ללקוח הדאטה החי (אם מאופשר)."""
    cli = _maybe_client(cfg)
    if cli is not None:
        try:
            # אם ה-URI מוגדר ב-sources, נשתמש ב-id הלוגי; אחרת ננסה קריאת ad-hoc
            # כאן נעשה ניסיון ad-hoc: אם זה uri מסוג yf:/duckdb:/sql:/local: הלקוח יטפל.
            df = cli.get(
                indicator_id="adhoc",  # לא נדרש בפועל — הלקוח משתמש ב-uri פנימי
                start=None,
                end=None,
                freq="D",
                live=bool(getattr(cfg, "data_client_live", False)),
                interval=getattr(cfg, "data_client_interval", None),
                period=getattr(cfg, "data_client_period", None),
            )
            # אם חזר DataFrame תקין
            if isinstance(df, pd.DataFrame) and not df.empty and "value" in df.columns:
                return df["value"].astype(float)
        except Exception:
            # ניפול חזרה למטען הישן
            pass
    # fallback ישן
    return _load_macro_source(uri)


def load_macro_bundle(cfg: MacroConfig) -> MacroBundle:
    selected: Dict[str, str] = {}
    if cfg.use_cpi:
        selected["cpi"] = cfg.sources.get("cpi", "")
    if cfg.use_unemployment:
        selected["unemployment"] = cfg.sources.get("unemployment", "")
    if cfg.use_policy_rate:
        selected["policy_rate"] = cfg.sources.get("policy_rate", "")
    if cfg.use_yield_curve:
        selected["yield_curve"] = cfg.sources.get("yield_curve", "")
    if cfg.use_pmi:
        selected["pmi"] = cfg.sources.get("pmi", "")
    if cfg.use_credit_spread:
        selected["credit_spread"] = cfg.sources.get("credit_spread", "")
    if cfg.use_oil:
        selected["oil"] = cfg.sources.get("oil", "")
    if cfg.use_dxy:
        selected["dxy"] = cfg.sources.get("dxy", "")
    if cfg.use_vix:
        selected["vix"] = cfg.sources.get("vix", "")

    data: Dict[str, MacroData] = {}
    for key, uri in selected.items():
        if not uri:
            logger.warning("מקור מאקרו חסר עבור %s — מדלג", key)
            continue
        try:
            ser = _load_series_with_cfg(uri, cfg)
            data[key] = MacroData(key, ser)
        except Exception as e:  # noqa: BLE001
            logger.error("שגיאה בטעינת %s (%s): %s", key, uri, e)
    return MacroBundle(data)


# =====================================================================
#                         Regime Indicators
# =====================================================================

def _zscore(x: pd.Series, winsor: float, min_periods: int) -> pd.Series:
    x = x.copy().astype(float)
    z = x.rolling(min_periods, min_periods=min_periods).apply(
        lambda s: (s.iloc[-1] - s.mean()) / (s.std(ddof=0) + 1e-9), raw=False
    )
    return z.clip(-winsor, winsor)


def build_regime_indicators(bundle: MacroBundle, cfg: MacroConfig) -> pd.DataFrame:
    """מחזיר DataFrame עם עמודות risk_on / inflation / growth (מוחלקים)."""
    frames: List[pd.Series] = []
    idx: Optional[pd.DatetimeIndex] = None

    def add(name: str, ser: Optional[MacroData], transform) -> None:
        nonlocal idx
        if ser is None:
            return
        base = ser.series
        if idx is None:
            idx = base.index
        z = transform(base).reindex(idx).fillna(0.0)
        frames.append(z.rename(name))

    vix = bundle.get("vix")
    add("risk_on_vix", vix, lambda s: -_zscore(s, cfg.winsor_limit, cfg.zscore_min_periods))  # VIX גבוה → Risk-Off

    cr = bundle.get("credit_spread")
    add("risk_on_credit", cr, lambda s: -_zscore(s, cfg.winsor_limit, cfg.zscore_min_periods))  # HY-OAS גבוה → Risk-Off

    yc = bundle.get("yield_curve")
    add("risk_on_yc", yc, lambda s: _zscore(s, cfg.winsor_limit, cfg.zscore_min_periods))      # עקום תלול → Risk-On

    oil = bundle.get("oil")
    add("inflation_oil", oil, lambda s: _zscore(s, cfg.winsor_limit, cfg.zscore_min_periods))   # עליית נפט → אינפלציוני

    cpi = bundle.get("cpi")
    add("inflation_cpi", cpi, lambda s: _zscore(s.pct_change().fillna(0.0), cfg.winsor_limit, cfg.zscore_min_periods))

    dxy = bundle.get("dxy")
    add("risk_on_dxy", dxy, lambda s: -_zscore(s, cfg.winsor_limit, cfg.zscore_min_periods))
    add("inflation_dxy", dxy, lambda s: -_zscore(s, cfg.winsor_limit, cfg.zscore_min_periods))

    pmi = bundle.get("pmi")
    add("growth_pmi", pmi, lambda s: _zscore(s, cfg.winsor_limit, cfg.zscore_min_periods))

    un = bundle.get("unemployment")
    add("growth_unemp", un, lambda s: -_zscore(s, cfg.winsor_limit, cfg.zscore_min_periods))

    pr = bundle.get("policy_rate")
    add("growth_policy_rate", pr, lambda s: -_zscore(s.pct_change().fillna(0.0), cfg.winsor_limit, cfg.zscore_min_periods))
    add("inflation_policy_rate", pr, lambda s: _zscore(s.pct_change().fillna(0.0), cfg.winsor_limit, cfg.zscore_min_periods))

    if not frames:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))

    M = pd.concat(frames, axis=1)
    regime = pd.DataFrame(index=M.index)
    regime["risk_on"] = M[[c for c in M.columns if c.startswith("risk_on_")]].mean(axis=1)
    regime["inflation"] = M[[c for c in M.columns if c.startswith("inflation_")]].mean(axis=1)
    regime["growth"] = M[[c for c in M.columns if c.startswith("growth_")]].mean(axis=1)

    if cfg.regime_smoothing_days > 1:
        regime = regime.rolling(cfg.regime_smoothing_days, min_periods=1).mean()
    return regime.fillna(0.0)


# =====================================================================
#                      התאמות חשיפה/פילטר לזוגות
# =====================================================================

def _regime_bucket(regime: pd.Series) -> str:
    """ממיר snapshot רציף לתג משטר פשוט לשימוש ב-hysteresis/min-duration.
    כללים בסיסיים (היוריסטיקה מתונה):
    - Risk-Off חזק: risk_on < -0.5 או (growth < -0.4 ו-inflation < 0)
    - Stagflation: inflation > 0.4 ו-growth < 0
    - Reflation: inflation > 0 ו-growth > 0.2
    - Slowdown: growth < -0.2 ו-risk_on <= 0.2
    - אחרת: Risk-On/Neutral לפי risk_on
    """
    r = float(regime.get("risk_on", 0.0))
    g = float(regime.get("growth", 0.0))
    i = float(regime.get("inflation", 0.0))
    if r < -0.5 or (g < -0.4 and i < 0):
        return "risk_off"
    if i > 0.4 and g < 0:
        return "stagflation"
    if i > 0 and g > 0.2:
        return "reflation"
    if g < -0.2 and r <= 0.2:
        return "slowdown"
    return "risk_on" if r > 0 else "neutral"


def _gate_regime_change(current: pd.Series, bucket: str, cfg: MacroConfig) -> pd.Series:
    """אוכף min_regime_duration באמצעות session_state (אם קיים Streamlit).
    אם השינוי מוקדם מדי — נשיב את ה-snapshot הקודם; אחרת נעדכן ונקבל את הנוכחי.
    """
    if not _HAS_STREAMLIT:
        return current
    ss = st.session_state
    prev = ss.get("macro_prev_regime_state")
    if not prev:
        ss["macro_prev_regime_state"] = {"bucket": bucket, "since": 0, "snapshot": current}
        return current
    if prev["bucket"] == bucket:
        prev["since"] = int(prev.get("since", 0)) + 1
        prev["snapshot"] = current
        ss["macro_prev_regime_state"] = prev
        return current
    if int(prev.get("since", 0)) < int(getattr(cfg, "min_regime_duration", 0)):
        return prev.get("snapshot", current)
    ss["macro_prev_regime_state"] = {"bucket": bucket, "since": 0, "snapshot": current}
    return current


def _apply_hysteresis_filters(
    pair_filt: Dict[str, bool], regime: pd.Series, cfg: MacroConfig
) -> Dict[str, bool]:
    """מרכך היפוכים תכופים בהחלטות include ע""י היסטוריית מצב ב-session_state.
    אם אין Streamlit — מחזיר את המפה כפי שהיא.
    שינוי include מאושר רק אם חלפו לפחות `cfg.hysteresis_days` טיקים מאז ההיפוך הקודם.
    """
    if not _HAS_STREAMLIT:
        return pair_filt
    ss = st.session_state
    hist: Dict[str, Dict[str, int | bool]] = ss.get("macro_pair_hysteresis", {})
    out: Dict[str, bool] = {}
    min_ticks = int(getattr(cfg, "hysteresis_days", 0))
    for pid, inc in pair_filt.items():
        state = hist.get(pid)
        if state is None:
            hist[pid] = {"include": inc, "since": 0}
            out[pid] = inc
            continue
        if bool(state.get("include")) == bool(inc):
            state["since"] = int(state.get("since", 0)) + 1
            out[pid] = inc
        else:
            if int(state.get("since", 0)) >= min_ticks:
                state["include"] = inc
                state["since"] = 0
                out[pid] = inc
            else:
                out[pid] = bool(state.get("include"))
                state["since"] = int(state.get("since", 0)) + 1
        hist[pid] = state
    ss["macro_pair_hysteresis"] = hist
    return out
@dataclass
class AdjustmentResult:
    """תוצאת התאמה לשילוב במערכת.

    Attributes
    ----------
    exposure_multiplier: float
        מקדם חשיפה גלובלי (לכלל המערכת)
    pair_adjustments: Dict[str, float]
        מקדמי חשיפה ברמת זוג
    filters: Dict[str, bool]
        פילטר הכללה/סינון לזוגות
    regime_snapshot: Optional[pd.Series]
        שורת Regime אחרונה לחיווי UI
    """
    exposure_multiplier: float
    pair_adjustments: Dict[str, float] = field(default_factory=dict)
    filters: Dict[str, bool] = field(default_factory=dict)
    regime_snapshot: Optional[pd.Series] = None
    regime_label: Optional[str] = None
    pair_scores: Dict[str, float] = field(default_factory=dict)
    caps_hints: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


def compute_exposure_multiplier(regime: pd.Series, cfg: MacroConfig) -> float:
    """ממיר את משולש ה-Regime למקדם חשיפה רציף בטווח הנתון.

    משתמש במשקלים מ-`cfg.regime_weights` וב-`cfg.shock_sensitivity`.
    """
    lo, hi = cfg.exposure_multiplier_bounds
    w_risk, w_infl, w_grow = cfg.regime_weights
    score = w_risk * regime.get("risk_on", 0.0) + w_infl * regime.get("inflation", 0.0) + w_grow * regime.get("growth", 0.0)
    score = np.tanh(score * cfg.shock_sensitivity)
    score = (score + 1.0) / 2.0  # → [0,1]
    mult = lo + score * (hi - lo)
    return float(np.clip(mult, lo, hi))


def _infer_sector_sensitivity(pair_row: pd.Series, sector_map: Dict[str, float]) -> float:
    for col in ("sector_a", "sector_b", "sector", "category"):
        if col in pair_row and isinstance(pair_row[col], str):
            sectors = [s.strip() for s in pair_row[col].split("/")]
            vals = [sector_map.get(s, 0.6) for s in sectors]
            if vals:
                return float(np.mean(vals))
    return 0.6


def adjust_pairs(
    pairs_df: pd.DataFrame,
    regime: pd.Series,
    cfg: MacroConfig,
    sensitivity_col: str = "macro_sensitivity",
    id_col: str = "pair_id",
) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """מחשב מקדמי התאמה ופילטרים לכל זוג.

    pairs_df מצופה לכלול מזהה ייחודי (pair_id) ומידע סקטוריאלי אם קיים.
    אם קיימת עמודת sensitivity_col — נשתמש בה; אחרת נשער לפי סקטור.
    """
    out_mult: Dict[str, float] = {}
    out_filt: Dict[str, bool] = {}

    infl = float(regime.get("inflation", 0.0))
    risk_on = float(regime.get("risk_on", 0.0))
    growth = float(regime.get("growth", 0.0))

    for _, row in pairs_df.iterrows():
        pid = str(row.get(id_col, f"{row.get('a','?')}-{row.get('b','?')}"))
        sens = float(row.get(sensitivity_col, np.nan))
        if np.isnan(sens):
            sens = _infer_sector_sensitivity(row, cfg.sector_sensitivity)

        base = 1.0 + 0.25 * (risk_on + 0.5 * growth) - 0.15 * infl
        base *= 1.0 + 0.2 * (sens - cfg.sensitivity_default)

        include = True
        if cfg.apply_mode in ("filter_only", "hybrid"):
            if (growth < -0.8 or risk_on < -0.8) and sens > (0.85 + cfg.filter_threshold):
                include = False

        out_mult[pid] = float(np.clip(base, 0.5, 1.8))
        out_filt[pid] = include

    return out_mult, out_filt


def _pair_macro_fit_score(risk_on: float, growth: float, inflation: float, sens: float) -> float:
    """ציון Macro Fit ∈ [0,100] לפי snapshot פשוט ורגישות זוג.
    היגיון בסיסי: חיובי עם risk_on/growth, שלילי עם inflation גבוהה; מותאם לרגישות.
    """
    raw = 0.6 * risk_on + 0.5 * growth - 0.3 * inflation
    raw *= 1.0 + 0.5 * (sens - 0.6)
    score01 = (np.tanh(raw) + 1.0) / 2.0
    return float(np.clip(score01 * 100.0, 0.0, 100.0))


def compute_adjustments(
    pairs_df: pd.DataFrame,
    bundle: MacroBundle,
    cfg: MacroConfig,
) -> AdjustmentResult:
    """נתיב מלא: Regime → Multipliers/Filters → תוצאה אחת ליתר המערכת."""
    regime_df = build_regime_indicators(bundle, cfg)
    if regime_df.empty:
        logger.warning("Regime DF ריק — מחזיר תוצאה נייטרלית")
        last = pd.Series({"risk_on": 0.0, "inflation": 0.0, "growth": 0.0}, name=pd.Timestamp.utcnow())
        return AdjustmentResult(1.0, {}, {}, last)

    last = regime_df.iloc[-1]
    # אכיפת משך מינימלי למשטר (אם אפשר)
    bucket = _regime_bucket(last)
    last = _gate_regime_change(last, bucket, cfg)
    exp_mult = compute_exposure_multiplier(last, cfg)
    pair_mult, pair_filt = adjust_pairs(pairs_df, last, cfg)

    # היסטרזיס להחלטות include כדי למנוע היפוכים מהירים
    pair_filt = _apply_hysteresis_filters(pair_filt, last, cfg)

    # חישוב ציוני Macro Fit לכל זוג (לא משנה את החתימות הקיימות)
    pair_scores: Dict[str, float] = {}
    infl = float(last.get("inflation", 0.0))
    risk_on = float(last.get("risk_on", 0.0))
    growth = float(last.get("growth", 0.0))
    for _, row in pairs_df.iterrows():
        pid = str(row.get("pair_id", f"{row.get('a','?')}-{row.get('b','?')}"))
        sens = row.get("macro_sensitivity", np.nan)
        if np.isnan(sens):
            sens = _infer_sector_sensitivity(row, cfg.sector_sensitivity)
        pair_scores[pid] = _pair_macro_fit_score(risk_on, growth, infl, float(sens))

    if cfg.apply_mode == "exposure_only":
        pair_filt = {k: True for k in pair_filt}
    elif cfg.apply_mode == "filter_only":
        pair_mult = {k: 1.0 for k in pair_mult}

    # חישוב Caps Hints
    caps_hints: Dict[str, float] = {}
    if getattr(cfg, "sector_caps", None) or getattr(cfg, "region_caps", None) or getattr(cfg, "currency_caps", None):
        for _, row in pairs_df.iterrows():
            pid = str(row.get("pair_id", f"{row.get('a','?')}-{row.get('b','?')}"))
            caps = []
            for col in ("sector", "sector_a", "sector_b"):
                val = row.get(col)
                if isinstance(val, str) and val in cfg.sector_caps:
                    caps.append(float(cfg.sector_caps[val]))
            reg = row.get("region")
            if isinstance(reg, str) and reg in cfg.region_caps:
                caps.append(float(cfg.region_caps[reg]))
            ccy = row.get("currency")
            if isinstance(ccy, str) and ccy in cfg.currency_caps:
                caps.append(float(cfg.currency_caps[ccy]))
            caps_hints[pid] = float(min(caps)) if caps else 1.0
            if getattr(cfg, "caps_mode", "hint") == "clip" and caps:
                pair_mult[pid] = min(pair_mult.get(pid, 1.0), caps_hints[pid])

    # מטא לתיעוד מהיר
    meta = {
        "ts": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
        "regime_label": bucket,
        "pairs": len(pair_mult),
        "included": int(sum(1 for v in pair_filt.values() if v)),
        "mean_score": float(np.mean(list(pair_scores.values()))) if pair_scores else None,
    }

    return AdjustmentResult(exp_mult, pair_mult, pair_filt, last, _regime_bucket(last), pair_scores, caps_hints, meta)


# =====================================================================
#                        Streamlit UI (אופציונלי)
# =====================================================================

def render_streamlit_ui(
    pairs_df: pd.DataFrame,
    cfg: Optional[MacroConfig] = None,
    bundle: Optional[MacroBundle] = None,
    key: str = "macro_tab",
) -> AdjustmentResult:
    """UI לטאב המאקרו — הטאב ב-root קורא רק לפונקציה זו.

    הערה: השימוש ב-Streamlit אופציונלי; אם לא מותקן, ייזרק RuntimeError.
    """
    if not _HAS_STREAMLIT:
        raise RuntimeError("Streamlit אינו מותקן בסביבה זו")

    st = __import__("streamlit")  # import מקומי למניעת עומס כאשר לא בשימוש
    st.header("⚙️ התאמות מאקרו לתיק / לזוגות")

    if cfg is None:
        cfg = MacroConfig()  # type: ignore[arg-type]

    with st.expander("הגדרות מקור נתונים", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            cfg.use_cpi = st.checkbox("CPI", value=cfg.use_cpi, key=f"{key}_cpi")
            cfg.use_unemployment = st.checkbox("אבטלה", value=cfg.use_unemployment, key=f"{key}_unemp")
            cfg.use_policy_rate = st.checkbox("ריבית", value=cfg.use_policy_rate, key=f"{key}_rate")
            cfg.use_yield_curve = st.checkbox("עקום 10Y-2Y", value=cfg.use_yield_curve, key=f"{key}_yc")
        with cols[1]:
            cfg.use_pmi = st.checkbox("PMI", value=cfg.use_pmi, key=f"{key}_pmi")
            cfg.use_credit_spread = st.checkbox("Credit Spread", value=cfg.use_credit_spread, key=f"{key}_cs")
            cfg.use_oil = st.checkbox("נפט", value=cfg.use_oil, key=f"{key}_oil")
        with cols[2]:
            cfg.use_dxy = st.checkbox("DXY", value=cfg.use_dxy, key=f"{key}_dxy")
            cfg.use_vix = st.checkbox("VIX", value=cfg.use_vix, key=f"{key}_vix")
            cfg.regime_smoothing_days = int(
                st.slider("החלקת Regime (ימים)", 1, 60, cfg.regime_smoothing_days, key=f"{key}_smooth")
            )

    with st.expander("אסטרטגיית יישום", expanded=True):
        cfg.apply_mode = st.selectbox(
            "מצב יישום",
            options=["exposure_only", "filter_only", "hybrid"],
            index=["exposure_only", "filter_only", "hybrid"].index(cfg.apply_mode),
            key=f"{key}_applymode",
        )
        bounds = st.slider(
            "טווח מקדם חשיפה גלובלי",
            0.2,
            2.5,
            cfg.exposure_multiplier_bounds,
            step=0.05,
            key=f"{key}_bounds",
        )
        cfg.exposure_multiplier_bounds = (float(bounds[0]), float(bounds[1]))
        cfg.filter_threshold = float(
            st.slider("סף פילטר לזוגות רגישים מאוד", 0.0, 0.5, float(cfg.filter_threshold), step=0.01, key=f"{key}_flt")
        )

    # התאמות מתקדמות
    with st.expander("התאמות מתקדמות", expanded=False):
        cfg.winsor_limit = float(
            st.slider("Winsor limit (Z-score)", 1.0, 6.0, float(cfg.winsor_limit), step=0.1, key=f"{key}_winsor")
        )
        cfg.zscore_min_periods = int(
            st.slider("חלון מינימלי ל-Z-score", 20, 252, int(cfg.zscore_min_periods), step=5, key=f"{key}_zmin")
        )
        cfg.shock_sensitivity = float(
            st.slider("רגישות להלם (tanh)", 0.2, 3.0, float(cfg.shock_sensitivity), step=0.1, key=f"{key}_shock")
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            rw1 = st.slider("משקל Risk-On", -1.0, 1.0, float(cfg.regime_weights[0]), step=0.05, key=f"{key}_w_risk")
        with c2:
            rw2 = st.slider("משקל Inflation", -1.0, 1.0, float(cfg.regime_weights[1]), step=0.05, key=f"{key}_w_infl")
        with c3:
            rw3 = st.slider("משקל Growth", -1.0, 1.0, float(cfg.regime_weights[2]), step=0.05, key=f"{key}_w_grow")
        cfg.regime_weights = (float(rw1), float(rw2), float(rw3))
        c4, c5 = st.columns(2)
        with c4:
            cfg.hysteresis_days = int(st.slider("היסטרזיס (ימים)", 0, 30, int(cfg.hysteresis_days), key=f"{key}_hyst"))
        with c5:
            cfg.min_regime_duration = int(st.slider("משך מינ' למשטר", 0, 60, int(cfg.min_regime_duration), key=f"{key}_mindur"))

    # בקרות סיכון
    with st.expander("בקרות סיכון", expanded=False):
        cfg.volatility_target = float(
            st.slider("יעד תנודתיות (%)", 1.0, 50.0, float(cfg.volatility_target * 100.0), step=0.5, key=f"{key}_tvol")
        ) / 100.0
        cfg.cvar_limit = float(
            st.slider("מגבלת CVaR (%)", 1.0, 50.0, float(cfg.cvar_limit * 100.0), step=0.5, key=f"{key}_cvar")
        ) / 100.0
        cfg.max_pair_weight = float(
            st.slider("מקס' משקל לזוג (%)", 0.5, 25.0, float(cfg.max_pair_weight * 100.0), step=0.5, key=f"{key}_mpw")
        ) / 100.0

    # טעינת נתונים
    if bundle is None:
        bundle = load_macro_bundle(cfg)

    # חישוב
    result = compute_adjustments(pairs_df, bundle, cfg)

    # תצוגה
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("מקדם חשיפה גלובלי", f"× {result.exposure_multiplier:.2f}")
    with c2:
        st.metric("Risk-On", f"{result.regime_snapshot.get('risk_on', 0.0):+.2f}")
    with c3:
        st.metric("Growth", f"{result.regime_snapshot.get('growth', 0.0):+.2f}")
    st.metric("Inflation", f"{result.regime_snapshot.get('inflation', 0.0):+.2f}")

    st.subheader("התאמות ברמת זוג")
    df_rows = {
        "pair_id": list(result.pair_adjustments.keys()),
        "multiplier": list(result.pair_adjustments.values()),
        "include": [result.filters[k] for k in result.pair_adjustments.keys()],
    }
    if result.pair_scores:
        df_rows["macro_fit_score"] = [result.pair_scores.get(k) for k in result.pair_adjustments.keys()]
    table = (
        pd.DataFrame(df_rows)
        .sort_values(["macro_fit_score" if "macro_fit_score" in df_rows else "multiplier"], ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(table, width="stretch")

    st.caption("הערה: הלוגיקה כאן — הטאב ב-root רק קורא לפונקציה הזו ומציג.")

    return result


# =====================================================================
#                         Public API
# =====================================================================
__all__ = [
    "MacroConfig",
    "MacroBundle",
    "MacroData",
    "load_macro_bundle",
    "build_regime_indicators",
    "compute_exposure_multiplier",
    "adjust_pairs",
    "compute_adjustments",
    "AdjustmentResult",
    "render_streamlit_ui",
]
