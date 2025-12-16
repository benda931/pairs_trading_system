# -*- coding: utf-8 -*-
"""
common/macro_factors.py — Macro factor loader & snapshot (HF-grade v1)
======================================================================

תפקיד הקובץ:
------------
1. לספק API אחיד להבאת פקטורים מאקרו (ריביות, עקום, מט"ח, וולאטיליות, קרדיט, קומודיטיז).
2. לתת "סנפשוט מאקרו" נוכחי (מצב שוק) עם תיוג רג'ימי בסיסי.
3. להיות שכבת לוגיקה **ללא UI** — טאב המאקרו ישתמש בו מאוחר יותר.

הקובץ *לא* מחליט מאיפה מגיע הדאטה (yfinance / IBKR / DuckDB / CSV) —
הוא מצפה או ל-DataFrame מוכן, או ל-loader חיצוני:
    price_loader(symbol: str) -> pd.Series | pd.DataFrame

"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Literal

import numpy as np
import pandas as pd


# ==========================
# Part 1 — Models & Config
# ==========================


@dataclass
class MacroFactorSpec:
    """
    הגדרה של פקטור מאקרו יחיד.

    name      : שם לוגי פנימי (e.g. "rate_10y", "slope_10y_2y").
    symbol    : טיקר / מזהה מקור (e.g. "^TNX", "DGS10", "DXY").
    field     : שם העמודה אם loader מחזיר DataFrame (ברירת מחדל: "close").
    transform : "level" | "diff" | "log" | "pct" — איך להפוך את הסדרה.
    scale     : כפל סקלרי (e.g. 0.01 כדי להפוך bps ל-%).
    group     : קבוצת פקטור ("rates","fx","vol","credit","commodity","other").
    """

    name: str
    symbol: str
    field: str = "close"
    transform: str = "level"  # "level" | "diff" | "log" | "pct"
    scale: float = 1.0
    group: str = "other"

def _default_rate_short() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="rate_short",
        symbol="^IRX",
        field="Close",
        transform="level",
        scale=0.01,
        group="rates",
    )


def _default_rate_long() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="rate_long",
        symbol="^TNX",
        field="Close",
        transform="level",
        scale=0.01,
        group="rates",
    )


def _default_rate_2y() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="rate_2y",
        symbol="^UST2Y",
        field="Close",
        transform="level",
        scale=0.01,
        group="rates",
    )


def _default_rate_5y() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="rate_5y",
        symbol="^UST5Y",
        field="Close",
        transform="level",
        scale=0.01,
        group="rates",
    )


def _default_dxy() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="dxy",
        symbol="DXY",
        field="Close",
        transform="level",
        scale=1.0,
        group="fx",
    )


def _default_vix() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="vix",
        symbol="^VIX",
        field="Close",
        transform="level",
        scale=1.0,
        group="vol",
    )


def _default_move() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="move",
        symbol="MOVE",
        field="Close",
        transform="level",
        scale=1.0,
        group="vol",
    )


def _default_vix_f1() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="vix_f1",
        symbol="VX1!",
        field="Close",
        transform="level",
        scale=1.0,
        group="vol",
    )


def _default_vix_f2() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="vix_f2",
        symbol="VX2!",
        field="Close",
        transform="level",
        scale=1.0,
        group="vol",
    )


def _default_credit_ig() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="credit_ig",
        symbol="LQD",
        field="Close",
        transform="pct",
        scale=1.0,
        group="credit",
    )


def _default_credit_hy() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="credit_hy",
        symbol="HYG",
        field="Close",
        transform="pct",
        scale=1.0,
        group="credit",
    )


def _default_oil() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="oil",
        symbol="CL=F",
        field="Close",
        transform="pct",
        scale=1.0,
        group="commodity",
    )


def _default_gold() -> MacroFactorSpec:
    return MacroFactorSpec(
        name="gold",
        symbol="GC=F",
        field="Close",
        transform="pct",
        scale=1.0,
        group="commodity",
    )

@dataclass
class MacroFactorConfig:
    """
    קונפיג כולל של פקטורי מאקרו.
    אפשר להרחיב/לשנות בפועל לפי Universe/Niche שלך.
    """

    # Rates (ניתן להתאים לסימבולים האמיתיים שלך)
    rate_short: MacroFactorSpec = field(default_factory=_default_rate_short)
    rate_long: MacroFactorSpec = field(default_factory=_default_rate_long)
    rate_2y: MacroFactorSpec = field(default_factory=_default_rate_2y)
    rate_5y: MacroFactorSpec = field(default_factory=_default_rate_5y)

    # FX / Dollar
    dxy: MacroFactorSpec = field(default_factory=_default_dxy)

    # Volatility
    vix: MacroFactorSpec = field(default_factory=_default_vix)
    move: MacroFactorSpec = field(default_factory=_default_move)
    vix_f1: MacroFactorSpec = field(default_factory=_default_vix_f1)
    vix_f2: MacroFactorSpec = field(default_factory=_default_vix_f2)

    # Credit proxies
    credit_ig: MacroFactorSpec = field(default_factory=_default_credit_ig)
    credit_hy: MacroFactorSpec = field(default_factory=_default_credit_hy)

    # Commodities
    oil: MacroFactorSpec = field(default_factory=_default_oil)
    gold: MacroFactorSpec = field(default_factory=_default_gold)

    def all_specs(self) -> List[MacroFactorSpec]:
        return [v for v in self.__dict__.values() if isinstance(v, MacroFactorSpec)]

@dataclass
class MacroSnapshot:
    """
    סנפשוט מאקרו נוכחי — ערכים + Z + תיוג רג'ימי ברמת פקטור/קבוצה.

    last_values      : value האחרון של כל פקטור.
    zscores          : Z של הפקטור (על פני ההיסטוריה הנתונה).
    regimes          : label ברמת factor ("rates_high","usd_strong","high_vol"...).
    group_regimes    : label ברמת group (e.g. "rates:high_rising", "vol:low").
    summary_label    : תיאור קצר למעלה ("High rates, inverted curve, risk-off").
    as_of            : זמן/תאריך התצפית האחרון.
    risk_on_score    : סיגנל רציף risk-on/risk-off (חיובי → risk-on, שלילי → risk-off).
    extreme_factors  : רשימת פקטורים שנמצאים ברמות קיצוניות (לפי Z).
    """

    last_values: Dict[str, float]
    zscores: Dict[str, float]
    regimes: Dict[str, str]
    group_regimes: Dict[str, str]
    summary_label: str
    as_of: Optional[pd.Timestamp]
    risk_on_score: float = 0.0
    extreme_factors: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MacroFactorStats:
    """
    סטטיסטיקות בסיסיות לפקטור מאקרו בודד.

    name            : שם הצגתי.
    key             : מפתח לוגי ("rate_short","dxy"...).
    group           : קבוצת פקטור ("rates","fx","vol","credit","commodity","other").
    mean            : ממוצע היסטורי.
    std             : סטיית תקן.
    p10/p50/p90     : אחוזון 10/50/90.
    current         : הערך האחרון.
    zscore_current  : Z-score של הערך האחרון מול ההיסטוריה.
    is_extreme      : האם הזזה קיצונית (|z|>z_extreme).
    n_obs           : מספר תצפיות.
    """

    name: str
    key: str
    group: str
    mean: float
    std: float
    p10: float
    p50: float
    p90: float
    current: float
    zscore_current: float
    is_extreme: bool
    n_obs: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MacroShockEvent:
    """
    אירוע Shocks של פקטור מאקרו.

    factor_name     : שם הפקטור (למשל "VIX").
    factor_key      : מפתח לוגי ("vix","credit_spread"...).
    group           : קבוצת פקטור ("rates","fx","vol","credit","commodity","other").
    ts              : timestamp האירוע.
    window_days     : על פני כמה ימים נמדדה הקפיצה.
    change          : שינוי (Δ) באותו חלון.
    zscore_change   : Z-score של השינוי.
    direction       : "up" | "down".
    severity_rank   : דירוג חומרה (0..1) יחסית לשאר shocks באותו פקטור/חלון.
    """

    factor_name: str
    factor_key: str
    group: str
    ts: pd.Timestamp
    window_days: int
    change: float
    zscore_change: float
    direction: Literal["up", "down"]
    severity_rank: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d

# ===============================
# Part 2 — Loading & Processing
# ===============================


def _apply_transform(series: pd.Series, method: str, scale: float) -> pd.Series:
    """
    מיישם transform+scale על סדרת מחירים.

    method:
        - "level": ללא שינוי (מלבד scale).
        - "diff":  ΔX_t = X_t - X_{t-1}
        - "log" :  Δlog(X_t) = log(X_t) - log(X_{t-1})
        - "pct" :  ΔX_t / X_{t-1}
    scale:
        - כפול סקלרי (למשל 0.01 כדי להפוך bps ל-%).
    """
    s = series.sort_index().astype(float)
    if method == "diff":
        s = s.diff()
    elif method == "log":
        s = np.log(s).diff()
    elif method == "pct":
        s = s.pct_change()
    # "level" → אין שינוי נוסף
    return s * float(scale)


def _normalize_macro_df(
    df: pd.DataFrame,
    freq: str = "B",
    forward_fill: bool = True,
) -> pd.DataFrame:
    """
    מנרמל DataFrame של פקטורי מאקרו:

    - ממיר אינדקס ל-DateTimeIndex.
    - ממיין לפי תאריך.
    - מריסמפל בתדירות נתונה (ברירת מחדל Business Days).
    - מבצע forward-fill (אם ביקשת).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.sort_index()
    if freq:
        df = df.resample(freq).last()

    if forward_fill:
        df = df.ffill()

    return df


def _winsorize_z(
    df: pd.DataFrame,
    z_max: float = 5.0,
) -> pd.DataFrame:
    """
    Winsorization פשוט לפי Z-score על כל פקטור:

    - מחשב Z לכל עמודה.
    - מחתך ערכים |Z|>z_max לגבול הקרוב.
    """
    if df.empty:
        return df

    means = df.mean()
    stds = df.std().replace(0, np.nan)

    def _clip_col(col: pd.Series) -> pd.Series:
        mu = means.get(col.name, 0.0)
        sigma = stds.get(col.name, np.nan)
        if not np.isfinite(sigma) or sigma <= 0:
            return col
        z = (col - mu) / sigma
        z_clipped = z.clip(-z_max, z_max)
        return mu + z_clipped * sigma

    return df.apply(_clip_col)


def load_macro_factors(
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: Optional[MacroFactorConfig] = None,
    price_loader: Optional[Callable[[str], Any]] = None,
    raw_df: Optional[pd.DataFrame] = None,
    *,
    align_to_index: Optional[pd.DatetimeIndex] = None,
    freq: str = "B",
    forward_fill: bool = True,
    winsorize_z: Optional[float] = None,
) -> pd.DataFrame:
    """
    טוען פקטורי מאקרו לטווח [start, end] ומבצע עיבודים בסיסיים.

    אפשרויות קלט:
    -------------
    raw_df:
        DataFrame קיים עם עמודות = שמות פקטורים (rate_short, dxy וכו').
        הקוד ינקה אינדקס, יריסמפל ויבצע מילוי/חיתוך לטווח.
    price_loader:
        פונקציה חיצונית שמחזירה מחירים לפי סימבול:
            price_loader(symbol: str) -> Series | DataFrame

        - DataFrame:
            משתמשים ב-field שצוין ב-MacroFactorSpec (או בעמודה הראשונה).
        - Series:
            נתייחס אליה כסדרה יחידה.

    פרמטרים נוספים:
    ----------------
    align_to_index:
        אם לא None: מיישרים את הפקטורים לפי אינדקס נתון (למשל index של prices_wide של הזוגות).
        שימושי כדי לחסוך חוסר תאימות תאריכים בין מאקרו לזוגות.
    freq:
        תדירות ריסמפלינג (ברירת מחדל "B" – ימי מסחר).
    forward_fill:
        אם True: מבצע forward-fill אחרי ריסמפלינג (מומלץ למאקרו).
    winsorize_z:
        אם לא None: מבצע Winsorization לפי Z־score על כל factor עם הסף z_max הזה.

    מחזיר:
    -------
    DataFrame:
        index = DateTimeIndex מטוהר,
        columns = שמות factor (cfg.all_specs().name + נגזרות).
    """
    cfg = cfg or MacroFactorConfig()

    # 1) Daload raw_df or via price_loader
    if raw_df is not None and not raw_df.empty:
        df = raw_df.copy()
        df.index = pd.to_datetime(df.index)
    else:
        if price_loader is None:
            raise ValueError("load_macro_factors: must provide either raw_df or price_loader")

        series_map: Dict[str, pd.Series] = {}
        for spec in cfg.all_specs():
            try:
                data = price_loader(spec.symbol)
                if isinstance(data, pd.DataFrame):
                    if spec.field in data.columns:
                        s = data[spec.field]
                    else:
                        s = data.iloc[:, 0]
                else:
                    s = pd.Series(data)
                s.index = pd.to_datetime(s.index, errors="coerce")
                s = s.loc[(s.index >= start) & (s.index <= end)]
                if s.empty:
                    continue
                series_map[spec.name] = _apply_transform(s, spec.transform, spec.scale)
            except Exception:
                # לא רוצים שהטאב יקרוס בגלל פקטור אחד – פשוט נדלג עליו
                continue

        df = pd.DataFrame(series_map)

    # 2) Normalization: sort, resample, fill
    df = df.loc[(df.index >= start) & (df.index <= end)]
    df = _normalize_macro_df(df, freq=freq, forward_fill=forward_fill)

    # 3) Derived factors (slope, credit spreads, proxies)
    df = add_derived_factors(df, cfg=cfg)

    # 4) Winsorization לפי Z (אופציונלי)
    if winsorize_z is not None and winsorize_z > 0:
        df = _winsorize_z(df, z_max=float(winsorize_z))

    # 5) Alignment to external index (למשל index של prices_wide)
    if align_to_index is not None:
        idx = pd.to_datetime(align_to_index)
        df = df.reindex(idx)
        if forward_fill:
            df = df.ffill()

    return df


def add_derived_factors(
    macro_df: pd.DataFrame,
    cfg: Optional[MacroFactorConfig] = None,
) -> pd.DataFrame:
    """
    מוסיף פקטורים נגזרים (slope, ספreads, proxies) ל-DataFrame קיים.

    דוגמאות:
    ---------
    - slope_10y_3m  = rate_long - rate_short
    - slope_10y_2y  = rate_long - rate_2y
    - slope_5y_2y   = rate_5y   - rate_2y
    - curve_chg_21d = שינוי בסלופ ב-21 ימים
    - credit_spread = credit_hy - credit_ig
    - risk_on_proxy = פונקציה של credit_spread ו-VIX
    - vix_term_1_0  = חוזה חזית / ספוט - 1 (contango/backwardation)
    """
    cfg = cfg or MacroFactorConfig()
    df = macro_df.copy().sort_index()

    cols = set(df.columns)

    # עקומי תשואות: סלוופים שונים
    if {"rate_long", "rate_short"} <= cols:
        df["slope_10y_3m"] = df["rate_long"] - df["rate_short"]

    if {"rate_long", "rate_2y"} <= cols:
        df["slope_10y_2y"] = df["rate_long"] - df["rate_2y"]

    if {"rate_5y", "rate_2y"} <= cols:
        df["slope_5y_2y"] = df["rate_5y"] - df["rate_2y"]

    # שינוי סלופ לאורך תקופה (למשל חודש ~ 21 ימי מסחר)
    for col in ("slope_10y_3m", "slope_10y_2y", "slope_5y_2y"):
        if col in df.columns:
            s = df[col]
            df[col + "_chg_21d"] = s - s.shift(21)

    # credit spread
    if {"credit_hy", "credit_ig"} <= cols:
        df["credit_spread"] = df["credit_hy"] - df["credit_ig"]

    # risk-on proxy פשוט:
    #   - credit_spread עולה → stress
    #   - vix עולה → stress
    #   הופכים סימן כדי לקבל "risk_on" כששני אלה יורדים
    if "credit_spread" in df.columns and "vix" in cols:
        cs = df["credit_spread"]
        vix = df["vix"]
        cs_n = (cs - cs.mean()) / cs.std().replace(0, np.nan)
        vix_n = (vix - vix.mean()) / vix.std().replace(0, np.nan)
        risk_on = -(cs_n.fillna(0.0) + vix_n.fillna(0.0))
        df["risk_on_proxy"] = risk_on

    # מבנה עקום ה-VIX (חוזים על ה-VIX):
    # vix_term_1_0 = VX1 / VIX - 1 (contango>0, backwardation<0)
    # vix_term_2_0 = VX2 / VIX - 1
    if {"vix", "vix_f1"} <= cols:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["vix_term_1_0"] = df["vix_f1"] / df["vix"] - 1.0
    if {"vix", "vix_f2"} <= cols:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["vix_term_2_0"] = df["vix_f2"] / df["vix"] - 1.0

    # נגזרת על term-structure: שינוי ב-contango/backwardation
    for col in ("vix_term_1_0", "vix_term_2_0"):
        if col in df.columns:
            df[col + "_chg_21d"] = df[col] - df[col].shift(21)

    return df

def compute_factor_stats(
    macro_df: pd.DataFrame,
    cfg: Optional[MacroFactorConfig] = None,
    lookback_days: Optional[int] = None,
    as_of: Optional[pd.Timestamp] = None,
    z_extreme: float = 2.5,
) -> Dict[str, MacroFactorStats]:
    """
    מחשב סטטיסטיקות לכל פקטור על חלון זמן:

    • lookback_days=None → כל ההיסטוריה.
    • lookback_days>0   → חיתוך ל-[as_of - lookback_days, as_of].

    מחזיר dict: factor_key → MacroFactorStats.
    """
    cfg = cfg or MacroFactorConfig()
    macro_df = macro_df.sort_index()

    if macro_df.empty:
        return {}

    if as_of is None:
        as_of = macro_df.index.max()

    if lookback_days is not None:
        start = as_of - pd.Timedelta(days=lookback_days)
        macro_df = macro_df[(macro_df.index >= start) & (macro_df.index <= as_of)]

    if macro_df.empty:
        return {}

    # map name→spec כדי לדעת group ו-description
    spec_by_name: Dict[str, MacroFactorSpec] = {s.name: s for s in cfg.all_specs()}

    stats: Dict[str, MacroFactorStats] = {}
    for col in macro_df.columns:
        s = macro_df[col].dropna().astype(float)
        if s.empty:
            continue

        spec = spec_by_name.get(col, MacroFactorSpec(name=col, symbol=col))
        cur = float(s.iloc[-1])
        m = float(s.mean())
        std = float(s.std(ddof=1))
        p10 = float(np.percentile(s, 10))
        p50 = float(np.percentile(s, 50))
        p90 = float(np.percentile(s, 90))
        if std > 0:
            z = (cur - m) / std
        else:
            z = 0.0
        extreme = abs(z) >= z_extreme

        stats[col] = MacroFactorStats(
            name=spec.name,
            key=spec.name,
            group=spec.group,
            mean=m,
            std=std,
            p10=p10,
            p50=p50,
            p90=p90,
            current=cur,
            zscore_current=float(z),
            is_extreme=extreme,
            n_obs=int(len(s)),
        )

    return stats

# =====================================
# Part 3 — Snapshots & Regime labelling
# =====================================


def _simple_regime_label(value: float, z: float, group: str) -> str:
    """
    תיוג רג'ימי בסיסי לפי value ו-Z של הפקטור.

    - group="rates":  rates_high / rates_low / rates_mid
    - group="fx":     usd_strong / usd_weak / usd_neutral
    - group="vol":    high_vol / low_vol / mid_vol
    - group="credit": credit_stress / credit_easy / credit_neutral
    - group="commodity": כרגע neutral (ניתן להרחיב בהמשך)
    """
    if group == "rates":
        if z > 1.0:
            return "rates_high"
        if z < -1.0:
            return "rates_low"
        return "rates_mid"

    if group == "fx":
        if z > 1.0:
            return "usd_strong"
        if z < -1.0:
            return "usd_weak"
        return "usd_neutral"

    if group == "vol":
        # כאן גם value חשוב (למשל VIX מעל 25)
        if value >= 25:
            return "high_vol"
        if value <= 15:
            return "low_vol"
        return "mid_vol"

    if group == "credit":
        if z > 1.0:
            return "credit_stress"
        if z < -1.0:
            return "credit_easy"
        return "credit_neutral"

    if group == "commodity":
        # אפשר להרחיב כאן לשדות כמו oil_high / gold_risk_off וכו'
        return "neutral"

    return "neutral"


def _classify_curve_regime(macro_df: pd.DataFrame) -> str:
    """
    סיווג משטר עקום ריביות (curve regime) לפי סלופים ו/או שינויי סלופ:

    - inverted: slope < 0 באופן מובהק
    - steepener: slope חיובי וגדל (chg>0)
    - flattener: slope חיובי אבל קטן ו/או יורד
    - neutral: אחרת
    """
    candidates = []
    for col in ("slope_10y_3m", "slope_10y_2y", "slope_5y_2y"):
        if col in macro_df.columns:
            s = macro_df[col].dropna()
            if s.empty:
                continue
            last_val = float(s.iloc[-1])
            if len(s) > 5:
                delta = float(s.iloc[-1] - s.iloc[-5])
            else:
                delta = 0.0
            candidates.append((col, last_val, delta))

    if not candidates:
        return "curve_unknown"

    # נבחר את הסלופ הארוך ביותר הזמין להחלטה (10y_3m>10y_2y>5y_2y)
    pref_order = ["slope_10y_3m", "slope_10y_2y", "slope_5y_2y"]
    chosen = None
    for name in pref_order:
        for cand in candidates:
            if cand[0] == name:
                chosen = cand
                break
        if chosen is not None:
            break
    if chosen is None:
        chosen = candidates[0]

    _, last_val, delta = chosen

    if last_val < 0:
        return "curve_inverted"
    if last_val > 0 and delta > 0:
        return "curve_steepener"
    if last_val > 0 and delta < 0:
        return "curve_flattener"
    return "curve_neutral"


def _compute_risk_on_score(macro_df: pd.DataFrame) -> float:
    """
    מחשב risk-on score פשוט, על בסיס risk_on_proxy אם קיים,
    או על בסיס שילוב של credit_spread, vix, ו-term structure של חוזי VIX.

    ערך חיובי → risk-on, ערך שלילי → risk-off.
    """
    df = macro_df.sort_index()

    if "risk_on_proxy" in df.columns:
        s = df["risk_on_proxy"].dropna()
        if not s.empty:
            window = min(len(s), 20)
            return float(s.iloc[-window:].mean())

    cs = df["credit_spread"].dropna() if "credit_spread" in df.columns else pd.Series(dtype=float)
    vx = df["vix"].dropna() if "vix" in df.columns else pd.Series(dtype=float)
    ts = df["vix_term_1_0"].dropna() if "vix_term_1_0" in df.columns else pd.Series(dtype=float)

    if cs.empty or vx.empty:
        return 0.0

    cs_z = (cs - cs.mean()) / cs.std().replace(0, np.nan)
    vx_z = (vx - vx.mean()) / vx.std().replace(0, np.nan)

    cs_z = cs_z.fillna(0.0)
    vx_z = vx_z.fillna(0.0)

    # risk_on_proxy בסיסי: -(credit_spread_z + vix_z)
    ro = -(cs_z + vx_z)

    # אם יש term structure — backwardation (ts<0) מחריף risk_off
    if not ts.empty:
        ts_z = (ts - ts.mean()) / ts.std().replace(0, np.nan)
        ts_z = ts_z.fillna(0.0)
        # backwardation (ts<0) → מוסיפים משקל שלילי
        adj = -ts_z.clip(upper=0)  # רק החלק השלילי מחריף risk_off
        ro = ro + adj.reindex_like(ro).fillna(0.0)

    window = min(len(ro), 20)
    return float(ro.iloc[-window:].mean())


def build_macro_snapshot(
    macro_df: pd.DataFrame,
    cfg: Optional[MacroFactorConfig] = None,
) -> MacroSnapshot:
    """
    מקבל DataFrame של מאקרו → מחזיר MacroSnapshot (ערכים, Z, תיוגים, summary).

    macro_df:
        - index: DateTimeIndex.
        - columns: factor names (MacroFactorSpec.name + נגזרות).

    מה שנכנס ל-MacroSnapshot:
        - last_values      : dict factor → value אחרון.
        - zscores          : dict factor → z-score על פני כל ההיסטוריה.
        - regimes          : dict factor → label (rates_high / usd_strong וכו').
        - group_regimes    : dict group  → label (rates_high / usd_strong / credit_stress / ...).
        - summary_label    : תיאור קצר ברמת מאקרו (“high rates, high vol, credit stress, usd_strong”).
        - risk_on_score    : סיגנל רציף risk-on/risk-off.
        - extreme_factors  : רשימת פקטורים קיצוניים לפי z-score.
    """
    cfg = cfg or MacroFactorConfig()
    macro_df = macro_df.sort_index()

    if macro_df.empty:
        return MacroSnapshot(
            last_values={},
            zscores={},
            regimes={},
            group_regimes={},
            summary_label="macro_no_data",
            as_of=None,
            risk_on_score=0.0,
            extreme_factors=(),
        )

    # 1) ערכים אחרונים
    last_row = macro_df.iloc[-1]
    last_values = {
        c: float(last_row[c])
        for c in macro_df.columns
        if np.isfinite(last_row.get(c, np.nan))
    }

    # 2) Z-score פשוט על פני כל ההיסטוריה
    means = macro_df.mean()
    stds = macro_df.std().replace(0, np.nan)
    zscores: Dict[str, float] = {}
    for c in macro_df.columns:
        if c not in last_values:
            continue
        mu = float(means.get(c, np.nan))
        sigma = float(stds.get(c, np.nan))
        if np.isfinite(sigma) and sigma > 0:
            zscores[c] = (last_values[c] - mu) / sigma
        else:
            zscores[c] = 0.0

    # 3) סיווג ברמת factor + aggregation ברמת group
    regimes: Dict[str, str] = {}
    group_regimes_raw: Dict[str, List[str]] = {}

    for spec in cfg.all_specs():
        name = spec.name
        if name not in last_values:
            continue
        val = float(last_values[name])
        z = float(zscores.get(name, 0.0))
        lab = _simple_regime_label(val, z, spec.group)
        regimes[name] = lab
        group_regimes_raw.setdefault(spec.group, []).append(lab)

    # 4) סיווג ברמת group (rates / fx / vol / credit / commodity)
    group_regimes: Dict[str, str] = {}
    for g, labs in group_regimes_raw.items():
        joined = " ".join(labs)
        if g == "rates":
            if "rates_high" in joined:
                group_regimes[g] = "rates_high"
            elif "rates_low" in joined:
                group_regimes[g] = "rates_low"
            else:
                group_regimes[g] = "rates_mid"
        elif g == "fx":
            if "usd_strong" in joined:
                group_regimes[g] = "usd_strong"
            elif "usd_weak" in joined:
                group_regimes[g] = "usd_weak"
            else:
                group_regimes[g] = "usd_neutral"
        elif g == "vol":
            if "high_vol" in joined:
                group_regimes[g] = "high_vol"
            elif "low_vol" in joined:
                group_regimes[g] = "low_vol"
            else:
                group_regimes[g] = "mid_vol"
        elif g == "credit":
            if "credit_stress" in joined:
                group_regimes[g] = "credit_stress"
            elif "credit_easy" in joined:
                group_regimes[g] = "credit_easy"
            else:
                group_regimes[g] = "credit_neutral"
        else:
            group_regimes[g] = labs[0] if labs else "neutral"

    # 5) Curve regime
    curve_regime = _classify_curve_regime(macro_df)
    group_regimes["curve"] = curve_regime

    # 6) Risk-on score (רציף)
    risk_on_score = _compute_risk_on_score(macro_df)

    if risk_on_score > 0.5:
        risk_on_label = "risk_on_strong"
    elif risk_on_score > 0.1:
        risk_on_label = "risk_on"
    elif risk_on_score < -0.5:
        risk_on_label = "risk_off_strong"
    elif risk_on_score < -0.1:
        risk_on_label = "risk_off"
    else:
        risk_on_label = "risk_neutral"
    group_regimes["risk"] = risk_on_label

    # 7) פקטורים קיצוניים (לפי Z)
    extreme_keys = [k for k, z in zscores.items() if abs(z) >= 2.5]

    # 8) Summary label טקסטואלי קצר
    summary_parts: List[str] = []

    rates_lab = group_regimes.get("rates")
    curve_lab = group_regimes.get("curve")
    vol_lab = group_regimes.get("vol")
    credit_lab = group_regimes.get("credit")
    fx_lab = group_regimes.get("fx")

    if rates_lab:
        summary_parts.append(rates_lab.replace("_", " "))
    if curve_lab and curve_lab != "curve_unknown":
        summary_parts.append(curve_lab.replace("_", " "))
    if vol_lab:
        summary_parts.append(vol_lab.replace("_", " "))
    if credit_lab:
        summary_parts.append(credit_lab.replace("_", " "))
    if fx_lab:
        summary_parts.append(fx_lab.replace("_", " "))
    summary_parts.append(risk_on_label.replace("_", " "))

    summary_label = ", ".join(summary_parts) if summary_parts else "macro_neutral"

    return MacroSnapshot(
        last_values=last_values,
        zscores=zscores,
        regimes=regimes,
        group_regimes=group_regimes,
        summary_label=summary_label,
        as_of=macro_df.index[-1],
        risk_on_score=float(risk_on_score),
        extreme_factors=tuple(extreme_keys),
    )



def build_macro_regime_series(
    macro_df: pd.DataFrame,
    cfg: Optional[MacroFactorConfig] = None,
) -> pd.DataFrame:
    """בונה סדרת משטרים לאורך זמן, לא רק Snapshot יחיד.

    מחזיר DataFrame:
        index  = תאריכים
        columns = [
            "rates_regime",
            "fx_regime",
            "vol_regime",
            "credit_regime",
            "curve_regime",
            "risk_regime",
        ]

    כך ניתן:
    - לצייר timeline של משטרי מאקרו בטאב המאקרו.
    - לחתוך ביצועי זוגות לפי משטרים (למשל רק כשהעקום inverted)."""
    cfg = cfg or MacroFactorConfig()
    macro_df = macro_df.sort_index()

    if macro_df.empty:
        return pd.DataFrame(
            columns=[
                "rates_regime",
                "fx_regime",
                "vol_regime",
                "credit_regime",
                "curve_regime",
                "risk_regime",
            ]
        )

    rows: List[Dict[str, str]] = []
    idx: List[pd.Timestamp] = []

    for ts, _ in macro_df.iterrows():
        hist = macro_df.loc[:ts]
        snap = build_macro_snapshot(hist, cfg=cfg)
        idx.append(ts)
        rows.append(
            {
                "rates_regime": snap.group_regimes.get("rates", "rates_mid"),
                "fx_regime": snap.group_regimes.get("fx", "usd_neutral"),
                "vol_regime": snap.group_regimes.get("vol", "mid_vol"),
                "credit_regime": snap.group_regimes.get("credit", "credit_neutral"),
                "curve_regime": snap.group_regimes.get("curve", "curve_neutral"),
                "risk_regime": snap.group_regimes.get("risk", "risk_neutral"),
            }
        )

    reg_df = pd.DataFrame(rows, index=pd.to_datetime(idx))
    return reg_df.sort_index()


def summarize_macro_snapshot(snapshot: MacroSnapshot) -> str:
    """יוצר תיאור טקסטואלי קריא של מצב המאקרו מסנפשוט אחד.

    שימושי לטאב המאקרו כדי להציג משפט אחד קצר, לדוגמה:
    "ריביות גבוהות, עקום inverted, וולאטים גבוה, קרדיט במתח, מצב risk-off".
    """
    if snapshot.as_of is None:
        return "מאקרו: אין נתונים זמינים."

    parts: List[str] = []
    gr = snapshot.group_regimes

    rates_lab = gr.get("rates")
    curve_lab = gr.get("curve")
    vol_lab = gr.get("vol")
    credit_lab = gr.get("credit")
    fx_lab = gr.get("fx")
    risk_lab = gr.get("risk")

    if rates_lab == "rates_high":
        parts.append("ריביות גבוהות")
    elif rates_lab == "rates_low":
        parts.append("ריביות נמוכות")
    elif rates_lab:
        parts.append("ריביות בינוניות")

    if curve_lab == "curve_inverted":
        parts.append("עקום הפוך (inverted)")
    elif curve_lab == "curve_steepener":
        parts.append("עקום מתלבלב (steepener)")
    elif curve_lab == "curve_flattener":
        parts.append("עקום משתטח (flattener)")

    if vol_lab == "high_vol":
        parts.append("וולאטיליות גבוהה")
    elif vol_lab == "low_vol":
        parts.append("וולאטיליות נמוכה")

    if credit_lab == "credit_stress":
        parts.append("מתח בשוק האשראי")
    elif credit_lab == "credit_easy":
        parts.append("קרדיט משוחרר")

    if fx_lab == "usd_strong":
        parts.append("דולר חזק")
    elif fx_lab == "usd_weak":
        parts.append("דולר חלש")

    if risk_lab == "risk_on_strong":
        parts.append("Risk-on חזק")
    elif risk_lab == "risk_on":
        parts.append("Risk-on")
    elif risk_lab == "risk_off_strong":
        parts.append("Risk-off חזק")
    elif risk_lab == "risk_off":
        parts.append("Risk-off")

    if not parts:
        return "מאקרו: מצב נייטרלי."

    return "מאקרו: " + ", ".join(parts)

def detect_macro_shocks(
    macro_df: pd.DataFrame,
    cfg: Optional[MacroFactorConfig] = None,
    windows: Tuple[int, ...] = (1, 5, 20),
    z_threshold: float = 3.0,
    max_events_per_factor: int = 50,
) -> pd.DataFrame:
    """
    מזהה Macro shocks בפקטורים, על בסיס macro_df.

    macro_df:
        index  = DateTimeIndex
        columns = factor names (למשל: rate_short, vix, credit_spread, ...)

    הלוגיקה:
    --------
    לכל פקטור ולכל window ב-windows:
        1. מחשבים diff: ΔX_t = X_t - X_{t-window}
        2. מחשבים Z-score של השינוי (על פני כל ההיסטוריה).
        3. מאתרים תאריכים שבהם |Z| >= z_threshold.
        4. מתייגים כיוון (up/down) וחומרה יחסית (severity_rank ∈ (0,1]).

    מחזיר:
    -------
    DataFrame עם index=ts ועמודות:
        factor_name, factor_key, group, window_days, change, zscore_change,
        direction, severity_rank
    """
    cfg = cfg or MacroFactorConfig()
    macro_df = macro_df.sort_index()

    if macro_df.empty:
        return pd.DataFrame(
            columns=[
                "factor_name",
                "factor_key",
                "group",
                "window_days",
                "change",
                "zscore_change",
                "direction",
                "severity_rank",
            ]
        ).set_index(pd.DatetimeIndex([], name="ts"))

    spec_by_name: Dict[str, MacroFactorSpec] = {s.name: s for s in cfg.all_specs()}
    events: List[MacroShockEvent] = []

    for name in macro_df.columns:
        series = macro_df[name].dropna().astype(float)
        if series.empty:
            continue

        spec = spec_by_name.get(name, MacroFactorSpec(name=name, symbol=name))

        for window in windows:
            if window <= 0 or len(series) <= window:
                continue

            diff = series - series.shift(window)
            diff = diff.dropna()
            std = diff.std(ddof=1)
            if std == 0 or np.isnan(std):
                continue

            z = diff / std
            hits = z[np.abs(z) >= z_threshold]
            if hits.empty:
                continue

            # דירוג חומרה לפי |z|
            hits_abs = hits.abs().sort_values(ascending=False)
            if max_events_per_factor is not None and len(hits_abs) > max_events_per_factor:
                hits_abs = hits_abs.iloc[:max_events_per_factor]

            max_abs = float(hits_abs.iloc[0])
            for ts, val in hits_abs.items():
                direction: Literal["up", "down"] = "up" if val > 0 else "down"
                change = float(diff.loc[ts])
                severity = float(abs(val) / max_abs) if max_abs > 0 else 1.0
                events.append(
                    MacroShockEvent(
                        factor_name=spec.name,
                        factor_key=spec.name,
                        group=spec.group,
                        ts=pd.to_datetime(ts),
                        window_days=int(window),
                        change=change,
                        zscore_change=float(val),
                        direction=direction,
                        severity_rank=severity,
                    )
                )

    if not events:
        return pd.DataFrame(
            columns=[
                "factor_name",
                "factor_key",
                "group",
                "window_days",
                "change",
                "zscore_change",
                "direction",
                "severity_rank",
            ]
        ).set_index(pd.DatetimeIndex([], name="ts"))

    rows = []
    for ev in events:
        d = ev.to_dict()
        ts = pd.to_datetime(d.pop("ts"))
        d["ts"] = ts
        rows.append(d)

    df = pd.DataFrame(rows).set_index("ts").sort_index()
    return df


__all__ = [
    "MacroFactorSpec",
    "MacroFactorConfig",
    "MacroSnapshot",
    "MacroFactorStats",
    "MacroShockEvent",
    "load_macro_factors",
    "add_derived_factors",
    "compute_factor_stats",
    "build_macro_snapshot",
    "build_macro_regime_series",
    "summarize_macro_snapshot",
    "detect_macro_shocks",
]


