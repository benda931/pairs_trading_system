# -*- coding: utf-8 -*-
"""
core/signals_engine.py — High-Level Signals Engine for Pairs Trading
====================================================================

שכבת-על מעל common.signal_generator.SignalGenerator:

- עבודה ברמת זוג (PairSignal) וברמת יקום (UniverseSignals).
- פרופילים לוגיים של אותות (SignalProfile) – mean_reversion / short_term / vol_arb / slow_mean_reversion.
- אינטגרציה קלה עם לוגיקת MR (Z/β/HL), איכות (quality_score) ו־edge.

שימו לב:
- מנוע הסיגנלים הטכני (Z-score, Bollinger, RSI, CUSUM וכו') יושב ב־common.signal_generator.
- כאן אנחנו מגדירים את ה-*שימוש* בסיגנלים ברמת קרן (scans, dashboards, universe).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path   
import logging
import json  
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
JSONDict = Dict[str, Any]

# פונקציות טעינת מחירים (נגדיר חתימות, המימוש בפועל מגיע מבחוץ)
PriceLoaderFn = Callable[[str, date, date], pd.DataFrame]
PairLegsLoaderFn = Callable[[Any, date, date], Tuple[pd.Series, pd.Series, str]]  # (s1, s2, label)


# ========= Pair-level models =========

@dataclass
class PairSignal:
    """
    סיכום אות ברמת זוג בודד.

    idea:
    - זה מה שהטאבים Dashboard / Smart Scan / Pair Analysis צריכים כדי להחליט "טופ |Z|", איכות, Edge וכו'.
    """

    # זיהוי
    pair_label: str
    sym_x: str
    sym_y: str

    # ליבה סטטיסטית
    beta: float
    z: float
    abs_z: float
    corr: float
    half_life: Optional[float]  # ימים
    points: int                 # מספר תצפיות אמיתי

    # Multi-Z (למשל z_10, z_30, z_60)
    multi_z: JSONDict = field(default_factory=dict)

    # איכות / Edge
    quality_score: Optional[float] = None
    edge: Optional[float] = None

    # מידע על פרופיל
    profile_name: str = "mean_reversion"
    z_window: int = 30
    beta_window: int = 60

    # סטטוס
    status: str = "ok"          # "ok" / "skipped" / "error"
    reason: str = ""            # "no_data" / "insufficient_points" / "exception:..."

    # מטא (לשימוש חופשי: sectors, cluster, regime, וכו')
    meta: JSONDict = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        d = asdict(self)
        return d


@dataclass
class UniverseSignals:
    """
    תוצאות ברמת universe:
    - signals_df: DataFrame עם אותות לזוגות שעברו
    - diag_df: DataFrame עם סטטוס לכל זוג (כולל כאלו שנפלו)
    """

    signals_df: pd.DataFrame
    diagnostics_df: pd.DataFrame

    def to_dict(self) -> JSONDict:
        return {
            "signals": self.signals_df.to_dict(orient="records"),
            "diagnostics": self.diagnostics_df.to_dict(orient="records"),
        }


# ========= Signal Profile Model =========

@dataclass
class SignalProfile:
    """
    פרופיל אותות לוגי (logical profile) – איך רוצים להסתכל על Mean-Reversion / Short-Term / Vol-Arb וכו'.

    שדות:
    - name: שם לוגי ("mean_reversion", "short_term", "vol_arb", "slow_mean_reversion"...)
    - description: תיאור מילולי – משמש ל-UI / דוחות.
    - z_window: חלון ראשי ל-Z-score על spread.
    - multi_z_windows: חלונות נוספים לניתוח רב-טווח (short/medium/long).
    - min_points: כמה נקודות מינימום כדי לסמוך על האות.
    - beta_window: חלון לחישוב β מתגלגל.
    - hl_min / hl_max: טווח Half-Life "בריא" לפרופיל.
    - min_corr: מינימום קורלציה כדי להיכנס בכלל למשחק.
    - score_weights: משקולות לסקור מורכב (corr / half_life / |Z| / quality / edge).
    - enabled_signals: אילו משפחות סיגנלים מתוך SignalGenerator להפעיל (zscore, spread, rsi, corr, coint...).
    - extras: מקום להרחבות (Vol focus, execution filters וכו').
    """

    name: str
    description: str = ""

    # חלונות ועוצמות
    z_window: int = 30
    multi_z_windows: List[int] = field(default_factory=list)
    beta_window: Optional[int] = None
    min_points: int = 60

    # תנאים סטטיסטיים
    hl_min: Optional[float] = None
    hl_max: Optional[float] = None
    min_corr: Optional[float] = None

    # משקולות לציון מורכב
    weight_corr: float = 0.3
    weight_half_life: float = 0.2
    weight_abs_z: float = 0.3
    weight_quality: float = 0.2
    weight_edge: float = 0.0

    # משפחות סיגנלים שנרצה להפעיל מתוך common.signal_generator
    enabled_signals: List[str] = field(
        default_factory=lambda: [
            "zscore",
            "spread",
            "correlation",
            "rolling_cointegration",
            "adf",
            "rsi",
        ]
    )

    # מקום להרחבות
    extras: JSONDict = field(default_factory=dict)

    def normalized_weights(self) -> JSONDict:
        """
        מחזיר משקולות מנורמלות (סכום = 1) לסקור.
        """
        comps = {
            "corr": max(self.weight_corr, 0.0),
            "half_life": max(self.weight_half_life, 0.0),
            "abs_z": max(self.weight_abs_z, 0.0),
            "quality": max(self.weight_quality, 0.0),
            "edge": max(self.weight_edge, 0.0),
        }
        s = sum(comps.values())
        if s <= 0:
            return {k: 0.0 for k in comps}
        return {k: v / s for k, v in comps.items()}

    def to_dict(self) -> JSONDict:
        d = asdict(self)
        d["score_weights_normalized"] = self.normalized_weights()
        return d


# ========= פרופילים מוכנים (bridge לפרופילים שהגדרת בדשבורד) =========

SIGNAL_PROFILES: Dict[str, SignalProfile] = {
    "mean_reversion": SignalProfile(
        name="mean_reversion",
        description="Mean-Reversion קלאסי לטווח בינוני (30 ימים).",
        z_window=30,
        multi_z_windows=[10, 60],
        min_points=60,
        beta_window=60,
        hl_min=3,
        hl_max=40,
        min_corr=0.5,
        weight_corr=0.30,
        weight_half_life=0.20,
        weight_abs_z=0.30,
        weight_quality=0.20,
        weight_edge=0.0,
        enabled_signals=[
            "zscore",
            "spread",
            "correlation",
            "rolling_cointegration",
            "adf",
        ],
    ),
    "short_term": SignalProfile(
        name="short_term",
        description="Short-Term MR – טווח קצר, תגובתי (10 ימים).",
        z_window=10,
        multi_z_windows=[5, 20],
        min_points=30,
        beta_window=30,
        hl_min=1,
        hl_max=20,
        min_corr=0.3,
        weight_corr=0.25,
        weight_half_life=0.15,
        weight_abs_z=0.40,
        weight_quality=0.20,
        enabled_signals=[
            "zscore",
            "spread",
            "rsi",
            "correlation",
        ],
    ),
    "vol_arb": SignalProfile(
        name="vol_arb",
        description="Volatility Arbitrage – פוקוס על תנודתיות וספרד.",
        z_window=20,
        multi_z_windows=[10, 40],
        min_points=40,
        beta_window=40,
        hl_min=2,
        hl_max=60,
        min_corr=0.2,
        weight_corr=0.20,
        weight_half_life=0.20,
        weight_abs_z=0.30,
        weight_quality=0.20,
        weight_edge=0.10,
        enabled_signals=[
            "spread",
            "zscore",
            "correlation",
        ],
        extras={"focus_vol": True},
    ),
    "slow_mean_reversion": SignalProfile(
        name="slow_mean_reversion",
        description="Mean-Reversion איטי לטווח ארוך (60 ימים).",
        z_window=60,
        multi_z_windows=[30, 120],
        min_points=120,
        beta_window=120,
        hl_min=10,
        hl_max=120,
        min_corr=0.6,
        weight_corr=0.35,
        weight_half_life=0.25,
        weight_abs_z=0.20,
        weight_quality=0.20,
        enabled_signals=[
            "spread",
            "zscore",
            "rolling_cointegration",
            "adf",
        ],
    ),
}


def get_signal_profile(name: str) -> SignalProfile:
    """
    מחזיר SignalProfile לפי שם, או mean_reversion אם לא קיים.
    """
    key = (name or "mean_reversion").strip().lower()
    if key not in SIGNAL_PROFILES:
        key = "mean_reversion"
    return SIGNAL_PROFILES[key]

# ========= Bridge ל-common.signal_generator + עזרי Z/β/HL =========

from common.signal_generator import (
    SignalGenerator,
    ZScoreConfig,
    BollingerConfig,
    CUSUMConfig,
    RSIConfig,
    CorrelationConfig,
    CointegrationConfig,
    ADFConfig,
    SpreadConfig,
)

# אופציונלי – פונקציות חכמות מ-common.utils (אם קיימות)
try:  # pragma: no cover
    from common.utils import (
        calculate_half_life as _calculate_half_life,
        calculate_correlation as _calculate_correlation,
        evaluate_pair_quality as _evaluate_pair_quality,
    )
except Exception:  # pragma: no cover
    _calculate_half_life = None
    _calculate_correlation = None
    _evaluate_pair_quality = None


def _calc_zscore_series(x: pd.Series, window: int) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std(ddof=0)
    return (x - mu) / sd


def _last_non_nan(s: pd.Series, default: float = np.nan) -> float:
    try:
        return float(s.dropna().iloc[-1])
    except Exception:
        return float(default)


def _fit_beta(s1: pd.Series, s2: pd.Series) -> float:
    """
    β גלובלי פשוט בין שני נכסים:
    - רגרסיה ליניארית Y ~ βX
    או fallback לקורלציה ויחס סטיות תקן.
    """
    s1 = pd.to_numeric(s1, errors="coerce").dropna()
    s2 = pd.to_numeric(s2, errors="coerce").dropna()
    s1, s2 = s1.align(s2, join="inner")
    if len(s1) < 10:
        return 1.0
    try:
        x = s2.values
        y = s1.values
        # y = a + βx → ניקח רק β
        beta, _ = np.polyfit(x, y, 1)
        return float(beta)
    except Exception:
        try:
            corr = np.corrcoef(s1.values, s2.values)[0, 1]
            std2 = np.std(s2.values)
            if std2 == 0:
                return 1.0
            return float(corr * np.std(s1.values) / std2)
        except Exception:
            return 1.0


def _rolling_beta(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    """
    β מתגלגל (for diagnostics / stability metrics).
    """
    s1 = pd.to_numeric(s1, errors="coerce")
    s2 = pd.to_numeric(s2, errors="coerce")
    cov = s1.rolling(window).cov(s2)
    var = s2.rolling(window).var(ddof=0)
    beta = cov / var
    return beta.ffill()


def _compute_spread_and_beta_series(
    s1: pd.Series,
    s2: pd.Series,
    beta_window: int,
) -> Tuple[pd.Series, pd.Series, float, float]:
    """
    מחזיר:
    - beta_series (rolling)
    - spread series לפי β האחרון
    - beta_last
    - beta_stability_metric (סטיית תקן נורמליזציה של β מתגלגל)
    """
    beta_series = _rolling_beta(s1, s2, beta_window)
    beta_last = _last_non_nan(beta_series, default=1.0)
    spread = s1 - beta_last * s2

    # מדד יציבות β – רעיון מתקדם: std(β) / |β_last|
    try:
        beta_std = float(beta_series.dropna().std(ddof=0))
        beta_stability = float(beta_std / max(abs(beta_last), 1e-6))
    except Exception:
        beta_stability = np.nan

    return beta_series, spread, float(beta_last), beta_stability


# ========= בניית SignalGenerator מתוך SignalProfile =========

def build_signal_generator_for_profile(
    profile: SignalProfile,
    *,
    base_config: Optional[Dict[str, Any]] = None,
) -> SignalGenerator:
    """
    בונה SignalGenerator מתוך SignalProfile:

    רעיונות מתקדמים כאן:
    1. התאמת זמנים (windows) לכל משפחת סיגנלים לפי z_window של הפרופיל.
    2. כיבוי משפחות סיגנלים שלא רלוונטיות לפרופיל (enabled_signals).
    3. אפשרות להרחבה דרך profile.extras בעתיד (למשל התאמת thresholds).

    החזרה היא SignalGenerator מוכן להרגיץ generate(price_x, series2=price_y).
    """
    if base_config is None:
        cfg = SignalGenerator.default_config()
    else:
        # עותק עמוק כדי שלא ילכלך את הבסיס
        cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in base_config.items()}

    # התאמת חלונות:
    # zscore / spread / correlation / rolling_cointegration / adf / rsi
    zwin = int(profile.z_window)
    for name in ("zscore", "spread", "correlation", "rolling_cointegration", "adf", "rsi"):
        if name not in cfg:
            cfg[name] = {}
        if "window" in cfg[name]:
            cfg[name]["window"] = zwin
        else:
            cfg[name]["window"] = zwin

    # כיבוי/הדלקה של משפחות לפי enabled_signals
    enabled_set = set(profile.enabled_signals or [])
    families = {
        "zscore": ZScoreConfig,
        "bollinger": BollingerConfig,
        "cusum": CUSUMConfig,
        "rsi": RSIConfig,
        "correlation": CorrelationConfig,
        "rolling_cointegration": CointegrationConfig,
        "adf": ADFConfig,
        "spread": SpreadConfig,
    }
    for key, cls in families.items():
        sub_cfg = cfg.get(key, {})
        if key in enabled_set:
            sub_cfg.setdefault("enabled", True)
        else:
            sub_cfg["enabled"] = False
        cfg[key] = sub_cfg

    # יצירת האובייקט
    sg = SignalGenerator(cfg)
    return sg


# ========= PairSignal Engine (חישוב אות לזוג אחד) =========

def compute_pair_signal(
    pair_obj: Any,
    start_date: date,
    end_date: date,
    *,
    profile_name: str = "mean_reversion",
    pair_legs_loader: PairLegsLoaderFn,
    base_signal_config: Optional[Dict[str, Any]] = None,
) -> Optional[PairSignal]:
    """
    מחשב PairSignal אחד ברמת קרן, על בסיס SignalProfile + SignalGenerator.

    רעיונות מתקדמים מוטמעים:
    1. שימוש ב-SignalGenerator כדי למדוד:
       - consensus_entry האחרון
       - צפיפות אותות (signal_density_252)
       - זמן מאז אות אחרון (time_since_last_entry)
    2. שימוש ב-β מתגלגל ליציבות β (beta_stability_metric).
    3. שימוש Half-Life + Corr כדי לקבוע fragility flags.
    4. ניצול Multi-Z ל-"z_consistency_score" (האם זמנים שונים מסכימים על כיוון).
    5. ADF/Cointegration מתוך הסיגנלים → stationarity_strength / cointegration_strength.
    6. בניית quality_score מתוך evaluate_pair_quality (אם קיים) + מדדים נוספים.
    7. חישוב deploy_tier (A/B/C) לפי איכות, Corr, HL.
    8. edge_hint – הערכת Edge גסה לפי Z, Corr, HL, Quality.
    9. שמירת meta עשיר (execution_intensity_hint, flags, וכו').
    10. התמודדות אלגנטית עם מצבי skip/error עם reason מפורט.
    """
    profile = get_signal_profile(profile_name)

    # ---------- טעינת המחירים לזוג ----------
    try:
        s1, s2, label = pair_legs_loader(pair_obj, start_date, end_date)
    except Exception as e:
        logger.warning("pair_legs_loader failed for %r: %s", pair_obj, e)
        return PairSignal(
            pair_label=str(pair_obj),
            sym_x="",
            sym_y="",
            beta=1.0,
            z=np.nan,
            abs_z=np.nan,
            corr=np.nan,
            half_life=None,
            points=0,
            status="error",
            reason=f"pair_legs_loader_failed: {e}",
        )

    if s1 is None or s2 is None or s1.empty or s2.empty:
        return PairSignal(
            pair_label=str(pair_obj),
            sym_x="",
            sym_y="",
            beta=1.0,
            z=np.nan,
            abs_z=np.nan,
            corr=np.nan,
            half_life=None,
            points=0,
            status="skipped",
            reason="no_data",
        )

    # ננסה להסיק סימבולים מתוך ה-index / name
    sym_x = str(getattr(s1, "name", "X"))
    sym_y = str(getattr(s2, "name", "Y"))

    # יישור
    s1, s2 = s1.align(s2, join="inner")
    points = int(len(s1))
    if points < max(profile.min_points, profile.z_window + 5):
        return PairSignal(
            pair_label=label,
            sym_x=sym_x,
            sym_y=sym_y,
            beta=1.0,
            z=np.nan,
            abs_z=np.nan,
            corr=np.nan,
            half_life=None,
            points=points,
            status="skipped",
            reason="insufficient_points",
        )

    # ---------- β ו-spread + β-stability ----------
    beta_window = profile.beta_window or max(10, min(120, profile.z_window))
    beta_series, spread, beta_last, beta_stability = _compute_spread_and_beta_series(
        s1,
        s2,
        beta_window=beta_window,
    )

    # ---------- Z-Score ----------
    z_main = _calc_zscore_series(spread, profile.z_window)
    z_last = _last_non_nan(z_main)
    abs_z = abs(z_last)

    # Multi-Z
    multi_z: JSONDict = {}
    for w in profile.multi_z_windows or []:
        try:
            w_int = int(w)
            if w_int <= 0:
                continue
            z_tmp = _calc_zscore_series(spread, w_int)
            multi_z[f"z_w{w_int}"] = _last_non_nan(z_tmp)
        except Exception:
            continue

    # Z-consistency score – האם כיוון Z דומה בכל החלונות
    z_vals = [z_last] + [v for v in multi_z.values() if isinstance(v, (int, float))]
    z_consistency_score = 1.0
    try:
        if len(z_vals) >= 2:
            signs = [np.sign(z) for z in z_vals if not np.isnan(z)]
            if signs:
                same_sign = (np.array(signs) == np.sign(z_last)).mean()
                z_consistency_score = float(same_sign)  # 0–1
    except Exception:
        z_consistency_score = np.nan

    # ---------- Corr ----------
    try:
        if _calculate_correlation is not None:
            corr = float(_calculate_correlation(s1, s2))
        else:
            corr = float(np.corrcoef(s1.values, s2.values)[0, 1])
    except Exception:
        corr = float("nan")

    # ---------- Half-Life ----------
    try:
        if _calculate_half_life is not None:
            hl_raw = _calculate_half_life(spread)
            half_life = float(hl_raw) if hl_raw is not None else None
        else:
            half_life = None
    except Exception:
        half_life = None

    # ---------- Quality Score ----------
    quality_score = None
    try:
        if _evaluate_pair_quality is not None:
            quality_score = float(_evaluate_pair_quality(s1, s2))
    except Exception:
        quality_score = None

    # ---------- SignalGenerator & Execution Metrics ----------
    sg = build_signal_generator_for_profile(profile, base_config=base_signal_config)
    sig_df = sg.generate(s1, series2=s2)
    consensus = sg.aggregate_signals(sig_df)

    # direction & time since last entry
    last_entry_side = 0
    time_since_last_entry = None
    signal_density_252 = None
    try:
        entry_series = consensus.get("entry")
        if entry_series is not None and not entry_series.empty:
            non_zero_entries = entry_series[entry_series != 0]
            if not non_zero_entries.empty:
                last_entry_side = int(non_zero_entries.iloc[-1])
                if isinstance(entry_series.index, pd.DatetimeIndex):
                    last_ts = non_zero_entries.index[-1]
                    last_ts = pd.to_datetime(last_ts)
                    end_ts = pd.to_datetime(entry_series.index[-1])
                    delta_days = (end_ts - last_ts).days
                    time_since_last_entry = int(delta_days)
            # צפיפות אותות – scaled ל-252 ימי מסחר
            total_entries = int((entry_series != 0).sum())
            n_days = max(1, len(entry_series))
            signal_density_252 = float(total_entries / n_days * 252.0)
    except Exception:
        pass

    # ---------- Stationarity / Cointegration Strength ----------
    stationarity_strength = None
    cointegration_strength = None
    try:
        adf_cols = [c for c in sig_df.columns if c.startswith("adf_p_value")]
        if adf_cols:
            p_adf = _last_non_nan(sig_df[adf_cols[0]])
            stationarity_strength = float(1.0 - min(max(p_adf, 0.0), 1.0)) if not np.isnan(p_adf) else None
    except Exception:
        stationarity_strength = None

    try:
        coint_cols = [c for c in sig_df.columns if c.startswith("coint_p_value")]
        if coint_cols:
            p_coint = _last_non_nan(sig_df[coint_cols[0]])
            cointegration_strength = float(1.0 - min(max(p_coint, 0.0), 1.0)) if not np.isnan(p_coint) else None
    except Exception:
        cointegration_strength = None

    # ---------- Flags & Tiering ----------
    fragility_flags: List[str] = []

    # Corr
    if profile.min_corr is not None and not np.isnan(corr):
        if corr < profile.min_corr:
            fragility_flags.append("low_corr_for_profile")

    # Half-Life
    if half_life is not None:
        if profile.hl_min is not None and half_life < profile.hl_min:
            fragility_flags.append("hl_too_short")
        if profile.hl_max is not None and half_life > profile.hl_max:
            fragility_flags.append("hl_too_long")

    # β stability
    if not np.isnan(beta_stability) and beta_stability > 0.5:
        fragility_flags.append("beta_unstable")

    # Z extremes
    if abs_z >= 4.0:
        fragility_flags.append("extreme_z")

    # Deployment tier A/B/C
    deploy_tier = "C"
    try:
        q = quality_score if quality_score is not None else 0.0
        hl_ok = (
            (half_life is not None)
            and (profile.hl_min is None or half_life >= profile.hl_min)
            and (profile.hl_max is None or half_life <= profile.hl_max)
        )
        corr_ok = (not np.isnan(corr)) and (profile.min_corr is None or corr >= profile.min_corr)

        if q >= 70 and hl_ok and corr_ok and not fragility_flags:
            deploy_tier = "A"
        elif q >= 40 and corr_ok:
            deploy_tier = "B"
        else:
            deploy_tier = "C"
    except Exception:
        deploy_tier = "C"

    # ---------- Edge hint ----------
    edge_hint = None
    try:
        # Edge גס: יותר טוב אם |Z| גבוה, corr גבוה, hl בינוני, quality גבוה
        score_z = min(abs_z / 3.0, 2.0)         # cap
        score_corr = max(0.0, (corr + 1.0) / 2.0) if not np.isnan(corr) else 0.0
        score_hl = 0.0
        if half_life is not None and profile.hl_min and profile.hl_max:
            center = (profile.hl_min + profile.hl_max) / 2.0
            # penalize distance from center
            score_hl = max(0.0, 1.0 - abs(half_life - center) / center)
        score_q = (quality_score or 0.0) / 100.0
        edge_hint = float(score_z * 0.4 + score_corr * 0.2 + score_hl * 0.2 + score_q * 0.2)
    except Exception:
        edge_hint = None

    # ---------- בניית PairSignal ----------
    meta: JSONDict = {
        "profile": profile.to_dict(),
        "beta_stability": beta_stability,
        "z_consistency_score": z_consistency_score,
        "stationarity_strength": stationarity_strength,
        "cointegration_strength": cointegration_strength,
        "last_entry_side": last_entry_side,
        "time_since_last_entry": time_since_last_entry,
        "signal_density_252": signal_density_252,
        "fragility_flags": fragility_flags,
        "deploy_tier": deploy_tier,
        "edge_hint": edge_hint,
    }

    ps = PairSignal(
        pair_label=label,
        sym_x=sym_x,
        sym_y=sym_y,
        beta=float(beta_last),
        z=float(z_last),
        abs_z=float(abs_z),
        corr=float(corr),
        half_life=half_life,
        points=points,
        multi_z=multi_z,
        quality_score=quality_score,
        edge=None,  # edge אמיתי/כלכלי – אפשר לחבר בהמשך ל-trade_logic / backtest
        profile_name=profile.name,
        z_window=int(profile.z_window),
        beta_window=int(beta_window),
        status="ok",
        reason="",
        meta=meta,
    )
    return ps

# ========= Part 3 — Universe Signals Engine (HF-grade, משודרג מאוד) =========

from concurrent.futures import ThreadPoolExecutor, as_completed


def _score_pair_signal(sig: PairSignal, profile: SignalProfile) -> float:
    """
    ציון מורכב לזוג בודד (0–1+).

    משלב:
    - |Z|
    - Corr
    - Half-Life fit לפרופיל
    - Quality Score
    - Edge Hint
    - Z-consistency
    - Stationarity & Cointegration strength
    - ענישת fragility flags / β instability / extreme Z
    """
    if sig.status != "ok":
        return 0.0

    w = profile.normalized_weights()

    # |Z| – ננרמל על 0–4 סטיות תקן
    abs_z = float(abs(sig.z)) if not np.isnan(sig.z) else 0.0
    score_abs_z = min(abs_z / 4.0, 1.0)

    # Corr
    if np.isnan(sig.corr):
        score_corr = 0.0
    else:
        score_corr = max(0.0, (sig.corr + 1.0) / 2.0)

    # HL fit – HL קרוב ל-center של הטווח
    score_hl = 0.0
    hl = sig.half_life
    if hl is not None and profile.hl_min is not None and profile.hl_max is not None:
        center = (profile.hl_min + profile.hl_max) / 2.0
        if center > 0:
            score_hl = max(0.0, 1.0 - abs(hl - center) / center)

    # Quality
    q = sig.quality_score if sig.quality_score is not None else 0.0
    score_q = max(0.0, min(q / 100.0, 1.0))

    # Edge hint
    edge_hint = sig.meta.get("edge_hint")
    if edge_hint is None or not isinstance(edge_hint, (float, int)):
        score_edge = 0.0
    else:
        score_edge = max(0.0, min(float(edge_hint), 1.5)) / 1.5  # cap 1.5

    # Stationarity & Cointegration
    st_strength = sig.meta.get("stationarity_strength")
    co_strength = sig.meta.get("cointegration_strength")
    try:
        st_strength = float(st_strength) if st_strength is not None else 0.0
    except Exception:
        st_strength = 0.0
    try:
        co_strength = float(co_strength) if co_strength is not None else 0.0
    except Exception:
        co_strength = 0.0

    # Z-consistency
    z_consistency = sig.meta.get("z_consistency_score")
    if z_consistency is None or not isinstance(z_consistency, (float, int)):
        z_consistency = 1.0
    else:
        z_consistency = max(0.0, min(float(z_consistency), 1.0))

    # בסיס הציון
    raw = (
        w["abs_z"] * score_abs_z
        + w["corr"] * score_corr
        + w["half_life"] * score_hl
        + w["quality"] * score_q
        + w["edge"] * score_edge
    )

    # בונוס קטן אם יש stationarity/cointegration חזקות
    raw *= (0.9 + 0.1 * st_strength)   # 0.9–1.0
    raw *= (0.9 + 0.1 * co_strength)

    # ענישה לפי fragility flags / β instability / extreme_z
    fragility_flags = sig.meta.get("fragility_flags", []) or []
    if fragility_flags:
        raw *= 0.9 ** len(fragility_flags)

    # β stability penalty
    beta_stab = sig.meta.get("beta_stability")
    try:
        beta_stab = float(beta_stab) if beta_stab is not None else 0.0
    except Exception:
        beta_stab = 0.0
    if beta_stab > 0.5:
        raw *= 0.8
    elif beta_stab > 0.2:
        raw *= 0.9

    # Consistency
    raw *= float(z_consistency)

    return float(max(0.0, raw))


def compute_pair_signal_with_reason(
    pair_obj: Any,
    start_date: date,
    end_date: date,
    *,
    profile_name: str = "mean_reversion",
    pair_legs_loader: PairLegsLoaderFn,
    base_signal_config: Optional[Dict[str, Any]] = None,
) -> JSONDict:
    """
    עטיפה ידידותית – תמיד מחזירה dict עם:
    - pair_label, sym_x, sym_y (אם אפשר)
    - status: "ok"/"skipped"/"error"
    - reason: טקסט קצר ברור
    - שדות ה-PairSignal אם הכל תקין
    """
    try:
        sig = compute_pair_signal(
            pair_obj,
            start_date,
            end_date,
            profile_name=profile_name,
            pair_legs_loader=pair_legs_loader,
            base_signal_config=base_signal_config,
        )
    except Exception as e:
        logger.warning("compute_pair_signal failed for %r: %s", pair_obj, e)
        return {
            "pair_label": str(pair_obj),
            "sym_x": "",
            "sym_y": "",
            "status": "error",
            "reason": f"exception:{e}",
        }

    if sig is None:
        return {
            "pair_label": str(pair_obj),
            "sym_x": "",
            "sym_y": "",
            "status": "error",
            "reason": "returned_none",
        }

    d = sig.to_dict()
    d.setdefault("status", "ok")
    d.setdefault("reason", "")
    return d


def _execution_intensity_label_from_meta(meta: JSONDict) -> str:
    """
    Execution intensity לפי signal_density_252:
    - < 5  → 'low'
    - < 25 → 'medium'
    - אחרת → 'high'
    """
    try:
        val = meta.get("signal_density_252")
        x = float(val)
    except Exception:
        return "unknown"

    if x < 5:
        return "low"
    if x < 25:
        return "medium"
    return "high"


def compute_universe_signals(
    pairs: List[Any],
    start_date: date,
    end_date: date,
    *,
    profile_name: str = "mean_reversion",
    pair_legs_loader: PairLegsLoaderFn,
    base_signal_config: Optional[Dict[str, Any]] = None,
    max_workers: int = 8,
    top_n: Optional[int] = None,
    min_score: float = 0.0,
    on_pair_meta: Optional[Callable[[Any, JSONDict], JSONDict]] = None,
    scoring_override: Optional[Callable[[PairSignal, SignalProfile], float]] = None,
) -> UniverseSignals:
    """
    מנוע אותות ברמת universe (HF-grade):

    Features:
    ---------
    1. ThreadPoolExecutor – חישוב מקבילי לזוגות.
    2. scoring per pair לפי _score_pair_signal או scoring_override מותאם.
    3. תיוג execution_intensity (low/medium/high).
    4. סינון לפי min_score + top_n.
    5. on_pair_meta hook – enrich meta (sector/cluster/macro).
    6. Universe normalizations: z_norm / corr_norm / hl_norm.
    7. Ranking: score_rank / score_pct_rank.
    8. anomaly flags (z_anomaly_flag).
    9. core_pairs (high-confidence pairs).
    10. universe diagnostics – לכל זוג status + reason + score.
    """
    if not pairs:
        empty_df = pd.DataFrame()
        return UniverseSignals(empty_df, empty_df)

    profile = get_signal_profile(profile_name)
    results: List[JSONDict] = []

    # --- חישוב לכל זוג (מקבילי) ---
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                compute_pair_signal_with_reason,
                p,
                start_date,
                end_date,
                profile_name=profile_name,
                pair_legs_loader=pair_legs_loader,
                base_signal_config=base_signal_config,
            ): p
            for p in pairs
        }
        for fut in as_completed(futures):
            out = fut.result()
            if isinstance(out, dict):
                results.append(out)

    if not results:
        empty_df = pd.DataFrame()
        return UniverseSignals(empty_df, empty_df)

    df_diag = pd.DataFrame(results)

    # נפריד בין ok לבין לא
    df_ok = df_diag[df_diag["status"] == "ok"].copy()
    df_not_ok = df_diag[df_diag["status"] != "ok"].copy()

    if df_ok.empty:
        return UniverseSignals(pd.DataFrame(), df_diag)

    # --- scoring לזוגות תקינים ---
    scores: List[float] = []
    ps_list: List[PairSignal] = []

    for i, row in df_ok.iterrows():
        try:
            ps = PairSignal(
                pair_label=row.get("pair_label", ""),
                sym_x=row.get("sym_x", ""),
                sym_y=row.get("sym_y", ""),
                beta=float(row.get("beta", 1.0)),
                z=float(row.get("z", np.nan)),
                abs_z=float(row.get("abs_z", np.nan)),
                corr=float(row.get("corr", np.nan)),
                half_life=row.get("half_life"),
                points=int(row.get("points", 0)),
                multi_z=row.get("multi_z") or {},
                quality_score=row.get("quality_score"),
                edge=row.get("edge"),
                profile_name=row.get("profile_name", profile.name),
                z_window=int(row.get("z_window", profile.z_window)),
                beta_window=int(row.get("beta_window", profile.beta_window or profile.z_window)),
                status=row.get("status", "ok"),
                reason=row.get("reason", ""),
                meta=row.get("meta") or {},
            )
            ps_list.append(ps)

            if scoring_override is not None:
                s = float(scoring_override(ps, profile))
            else:
                s = float(_score_pair_signal(ps, profile))
        except Exception:
            ps_list.append(ps)
            s = 0.0
        scores.append(float(s))

    df_ok["score"] = scores

    # min_score filter
    if min_score > 0.0:
        df_ok = df_ok[df_ok["score"] >= min_score]

    if df_ok.empty:
        return UniverseSignals(pd.DataFrame(), df_diag)

    # --- Normalizations & ranks ---
    def _norm_series(s: pd.Series) -> pd.Series:
        s_num = pd.to_numeric(s, errors="coerce")
        mn, mx = s_num.min(), s_num.max()
        if np.isnan(mn) or np.isnan(mx) or mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s_num - mn) / (mx - mn)

    df_ok["z_norm"] = _norm_series(df_ok["abs_z"])
    df_ok["corr_norm"] = _norm_series(df_ok["corr"])
    df_ok["hl_norm"] = _norm_series(df_ok["half_life"])

    df_ok["score_rank"] = df_ok["score"].rank(ascending=False, method="min")
    df_ok["score_pct_rank"] = (len(df_ok) - df_ok["score_rank"]) / max(len(df_ok) - 1, 1)

    # --- anomaly flags על |Z| ---
    try:
        z_mean = float(df_ok["abs_z"].mean())
        z_std = float(df_ok["abs_z"].std(ddof=0))
        if z_std > 0:
            df_ok["z_anomaly_flag"] = (df_ok["abs_z"] > z_mean + 3 * z_std).map(
                lambda v: "extreme_z" if v else "none"
            )
        else:
            df_ok["z_anomaly_flag"] = "none"
    except Exception:
        df_ok["z_anomaly_flag"] = "none"

    # --- execution_intensity מתוך meta.signal_density_252 ---
    exec_labels: List[str] = []
    exec_vals: List[float] = []
    for meta in df_ok["meta"]:
        if isinstance(meta, dict):
            exec_labels.append(_execution_intensity_label_from_meta(meta))
            try:
                exec_vals.append(float(meta.get("signal_density_252", np.nan)))
            except Exception:
                exec_vals.append(np.nan)
        else:
            exec_labels.append("unknown")
            exec_vals.append(np.nan)
    df_ok["execution_intensity"] = exec_labels
    df_ok["execution_density"] = exec_vals

    # core_pairs: top 20% לפי score_pct_rank
    df_ok["is_core_pair"] = df_ok["score_pct_rank"] >= 0.8

    # --- enrich meta מבחוץ (on_pair_meta) ---
    if on_pair_meta is not None:
        new_meta: List[JSONDict] = []
        for p_label, meta in zip(df_ok["pair_label"], df_ok["meta"]):
            m = meta or {}
            try:
                enriched = on_pair_meta(p_label, m)
                new_meta.append(enriched if isinstance(enriched, dict) else m)
            except Exception:
                new_meta.append(m)
        df_ok["meta"] = new_meta

    # מיון סופי
    df_ok = df_ok.sort_values(["score", "abs_z", "points"], ascending=[False, False, False])

    # top_n
    if top_n is not None:
        df_ok = df_ok.head(int(top_n))

    df_signals = df_ok.reset_index(drop=True)

    # diagnostics – לכל הזוגות, כולל score=0 ל-not_ok
    df_not_ok["score"] = 0.0
    df_diag_final = pd.concat([df_signals, df_not_ok], ignore_index=True, sort=False)

    return UniverseSignals(signals_df=df_signals, diagnostics_df=df_diag_final)


def compute_universe_signals_with_diagnostics(
    pairs: List[Any],
    start_date: date,
    end_date: date,
    *,
    profile_name: str = "mean_reversion",
    pair_legs_loader: PairLegsLoaderFn,
    base_signal_config: Optional[Dict[str, Any]] = None,
    max_workers: int = 8,
    top_n: Optional[int] = None,
    min_score: float = 0.0,
    on_pair_meta: Optional[Callable[[Any, JSONDict], JSONDict]] = None,
    scoring_override: Optional[Callable[[PairSignal, SignalProfile], float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    עטיפה שמחזירה:
    - signals_df
    - diagnostics_df
    """
    uni = compute_universe_signals(
        pairs,
        start_date,
        end_date,
        profile_name=profile_name,
        pair_legs_loader=pair_legs_loader,
        base_signal_config=base_signal_config,
        max_workers=max_workers,
        top_n=top_n,
        min_score=min_score,
        on_pair_meta=on_pair_meta,
        scoring_override=scoring_override,
    )
    return uni.signals_df, uni.diagnostics_df


def compute_universe_signals_profile(
    profile_name: str,
    pairs: List[Any],
    start_date: date,
    end_date: date,
    *,
    pair_legs_loader: PairLegsLoaderFn,
    base_signal_config: Optional[Dict[str, Any]] = None,
    max_workers: int = 8,
    top_n: Optional[int] = None,
    min_score: float = 0.0,
) -> pd.DataFrame:
    """
    עטיפה נוחה: מחזיר רק signals_df עבור profile נתון.
    """
    uni = compute_universe_signals(
        pairs,
        start_date,
        end_date,
        profile_name=profile_name,
        pair_legs_loader=pair_legs_loader,
        base_signal_config=base_signal_config,
        max_workers=max_workers,
        top_n=top_n,
        min_score=min_score,
    )
    return uni.signals_df


def summarize_universe_signals(signals_df: pd.DataFrame) -> JSONDict:
    """
    סיכום מהיר ל-universe signals:

    מחזיר:
    - count_ok
    - avg_abs_z / max_abs_z
    - avg_corr
    - hl_median
    - avg_score
    - tier_counts (A/B/C/unknown)
    - core_pairs_count
    - execution_intensity_counts
    - z_anomaly_count
    """
    if signals_df is None or signals_df.empty:
        return {
            "count_ok": 0,
            "avg_abs_z": None,
            "max_abs_z": None,
            "avg_corr": None,
            "hl_median": None,
            "avg_score": None,
            "tier_counts": {},
            "core_pairs_count": 0,
            "execution_intensity_counts": {},
            "z_anomaly_count": 0,
        }

    df = signals_df.copy()
    out: JSONDict = {}
    out["count_ok"] = int(len(df))

    if "abs_z" in df.columns:
        out["avg_abs_z"] = float(df["abs_z"].mean())
        out["max_abs_z"] = float(df["abs_z"].max())
    else:
        out["avg_abs_z"] = out["max_abs_z"] = None

    if "corr" in df.columns:
        out["avg_corr"] = float(df["corr"].mean())
    else:
        out["avg_corr"] = None

    if "half_life" in df.columns:
        out["hl_median"] = float(df["half_life"].median())
    else:
        out["hl_median"] = None

    if "score" in df.columns:
        out["avg_score"] = float(df["score"].mean())
    else:
        out["avg_score"] = None

    # tier_counts מתוך meta
    tier_counts: Dict[str, int] = {}
    if "meta" in df.columns:
        for meta in df["meta"]:
            if isinstance(meta, dict):
                t = meta.get("deploy_tier", "unknown")
            else:
                t = "unknown"
            tier_counts[t] = tier_counts.get(t, 0) + 1
    out["tier_counts"] = tier_counts

    # core pairs
    if "is_core_pair" in df.columns:
        out["core_pairs_count"] = int(df["is_core_pair"].sum())
    else:
        out["core_pairs_count"] = 0

    # execution intensity counts
    exec_counts: Dict[str, int] = {}
    if "execution_intensity" in df.columns:
        for v in df["execution_intensity"]:
            v = str(v or "unknown")
            exec_counts[v] = exec_counts.get(v, 0) + 1
    out["execution_intensity_counts"] = exec_counts

    # z anomaly count
    z_anom_count = 0
    if "z_anomaly_flag" in df.columns:
        z_anom_count = int((df["z_anomaly_flag"] == "extreme_z").sum())
    out["z_anomaly_count"] = z_anom_count

    return out

# ========= Part 4 — Dashboard / Scan / Matrix & Advanced Wrappers =========

def compute_dashboard_signals(
    pairs: List[Any],
    start_date: date,
    end_date: date,
    *,
    profile_name: str = "mean_reversion",
    pair_legs_loader: PairLegsLoaderFn,
    base_signal_config: Optional[Dict[str, Any]] = None,
    top_n: int = 15,
    min_points: int = 40,
    min_abs_z: float = 0.0,
    min_quality: float = 0.0,
    min_corr: Optional[float] = None,
) -> Tuple[pd.DataFrame, JSONDict]:
    """
    עטיפה ייעודית ל-Dashboard Home:
    - מחשבת universe signals לפי profile.
    - מסננת לפי points / |Z| / Quality / Corr.
    - מחזירה:
        df_top: top-N אותות
        summary: dict מסכם (summarize_universe_signals).
    """
    df_signals, _ = compute_universe_signals_with_diagnostics(
        pairs=pairs,
        start_date=start_date,
        end_date=end_date,
        profile_name=profile_name,
        pair_legs_loader=pair_legs_loader,
        base_signal_config=base_signal_config,
        max_workers=8,
        top_n=None,
        min_score=0.0,
    )

    if df_signals is None or df_signals.empty:
        return pd.DataFrame(), summarize_universe_signals(df_signals)

    profile = get_signal_profile(profile_name)
    corr_thr = min_corr if min_corr is not None else profile.min_corr

    df = df_signals.copy()

    if "points" in df.columns:
        df = df[df["points"] >= int(min_points)]
    if "abs_z" in df.columns and min_abs_z > 0.0:
        df = df[df["abs_z"].abs() >= float(min_abs_z)]
    if "quality_score" in df.columns and min_quality > 0.0:
        df = df[df["quality_score"].fillna(-1.0) >= float(min_quality)]
    if "corr" in df.columns and corr_thr is not None:
        df = df[df["corr"].fillna(-1.0) >= float(corr_thr)]

    if df.empty:
        return pd.DataFrame(), summarize_universe_signals(df_signals)

    if "score" in df.columns:
        df = df.sort_values(["score", "abs_z", "points"], ascending=[False, False, False])
    elif "abs_z" in df.columns:
        df = df.sort_values("abs_z", ascending=False)

    df_top = df.head(int(top_n)).reset_index(drop=True)
    summary = summarize_universe_signals(df_top)
    return df_top, summary


def compute_smart_scan_signals(
    pairs: List[Any],
    start_date: date,
    end_date: date,
    *,
    profile_name: str = "mean_reversion",
    pair_legs_loader: PairLegsLoaderFn,
    base_signal_config: Optional[Dict[str, Any]] = None,
    universe_n: Optional[int] = None,
    min_corr: float = 0.3,
    min_quality: float = 0.0,
    min_abs_z: float = 1.0,
    max_half_life: Optional[float] = None,
    max_z_cap: float = 6.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    עטיפה ל-"Smart Scan":

    - חישוב signals לכל ה-universe.
    - פילטרים: Corr, Quality, |Z|, Half-Life.
    - חיתוך חריגים לפי max_z_cap.
    - universe_n: אם מוגדר, לוקח top-N אחרי המיון.
    """
    df_signals, df_diag = compute_universe_signals_with_diagnostics(
        pairs=pairs,
        start_date=start_date,
        end_date=end_date,
        profile_name=profile_name,
        pair_legs_loader=pair_legs_loader,
        base_signal_config=base_signal_config,
        max_workers=8,
        top_n=None,
        min_score=0.0,
    )

    if df_signals is None or df_signals.empty:
        return pd.DataFrame(), df_diag

    df = df_signals.copy()

    if "corr" in df.columns:
        df = df[df["corr"].fillna(-1.0) >= float(min_corr)]

    if "quality_score" in df.columns and min_quality > 0.0:
        df = df[df["quality_score"].fillna(-1.0) >= float(min_quality)]

    if "abs_z" in df.columns and min_abs_z > 0.0:
        df = df[df["abs_z"].abs() >= float(min_abs_z)]

    if max_half_life is not None and "half_life" in df.columns:
        df["half_life"] = pd.to_numeric(df["half_life"], errors="coerce")
        df = df[df["half_life"].fillna(float("inf")) <= float(max_half_life)]

    if "abs_z" in df.columns and max_z_cap is not None:
        df = df[df["abs_z"].abs() <= float(max_z_cap)]

    if df.empty:
        return df, df_diag

    if universe_n is not None:
        if "score" in df.columns:
            df = df.sort_values(["score", "abs_z", "points"], ascending=[False, False, False])
        elif "abs_z" in df.columns:
            df = df.sort_values("abs_z", ascending=False)
        df = df.head(int(universe_n))

    df_scan = df.reset_index(drop=True)
    return df_scan, df_diag


def signals_to_matrix(
    df_signals: pd.DataFrame,
    *,
    value_col: str = "abs_z",
    index_col: str = "sym_x",
    columns_col: str = "sym_y",
    aggregate: str = "mean",
) -> pd.DataFrame:
    """
    ממפה signals_df למטריצה (ל-Matrix / Heatmap):

    - value_col: abs_z / corr / score / ...
    - aggregate: mean / max / min.
    """
    if df_signals is None or df_signals.empty:
        return pd.DataFrame()

    df = df_signals.copy()
    if not {value_col, index_col, columns_col}.issubset(df.columns):
        return pd.DataFrame()

    if aggregate == "mean":
        piv = df.groupby([index_col, columns_col])[value_col].mean().unstack(columns_col)
    elif aggregate == "max":
        piv = df.groupby([index_col, columns_col])[value_col].max().unstack(columns_col)
    else:
        piv = df.groupby([index_col, columns_col])[value_col].min().unstack(columns_col)

    return piv.sort_index().sort_index(axis=1)


def attach_signals_to_universe_meta(
    df_signals: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    pair_col_signals: str = "pair_label",
    pair_col_meta: str = "Pair",
) -> pd.DataFrame:
    """
    מחבר signals_df ל-DataFrame של metadata (sector/region/cluster/...).
    """
    if df_signals is None or df_signals.empty or meta_df is None or meta_df.empty:
        return pd.DataFrame()

    ds = df_signals.copy()
    dm = meta_df.copy()

    if pair_col_signals not in ds.columns or pair_col_meta not in dm.columns:
        return ds

    dm = dm.rename(columns={pair_col_meta: pair_col_signals})
    return ds.merge(dm, on=pair_col_signals, how="left")


def signals_to_watchlist(
    df_signals: pd.DataFrame,
    *,
    tier: str = "A",
    min_score: float = 0.5,
    max_pairs: int = 50,
) -> List[str]:
    """
    מחלץ רשימת Watchlist מתוך signals_df:
    - רק deploy_tier מסוים (A/B).
    - score מעל min_score.
    """
    if df_signals is None or df_signals.empty:
        return []

    df = df_signals.copy()
    if "meta" not in df.columns:
        return []

    tiers: List[str] = []
    for meta in df["meta"]:
        if isinstance(meta, dict):
            tiers.append(str(meta.get("deploy_tier", "unknown")))
        else:
            tiers.append("unknown")
    df["deploy_tier"] = tiers

    mask = (df["deploy_tier"] == tier) & (
        pd.to_numeric(df["score"], errors="coerce").fillna(0.0) >= float(min_score)
    )
    df = df[mask]

    df = df.sort_values(["score", "abs_z", "points"], ascending=[False, False, False])
    return df["pair_label"].head(int(max_pairs)).astype(str).tolist()


def signals_to_pair_flags(
    df_signals: pd.DataFrame,
    *,
    min_score_favorite: float = 0.7,
    max_tier_for_avoid: str = "C",
) -> JSONDict:
    """
    ממיר signals_df למבנה pair_flags:
      {
        "favorite": [...],
        "avoid": [...],
        "core": [...],
      }
    """
    flags: JSONDict = {"favorite": [], "avoid": [], "core": []}
    if df_signals is None or df_signals.empty:
        return flags

    df = df_signals.copy()
    if "meta" not in df.columns:
        return flags

    tiers: List[str] = []
    frags: List[List[str]] = []
    for meta in df["meta"]:
        if isinstance(meta, dict):
            tiers.append(str(meta.get("deploy_tier", "unknown")))
            frags.append(list(meta.get("fragility_flags", []) or []))
        else:
            tiers.append("unknown")
            frags.append([])
    df["deploy_tier"] = tiers
    df["fragility_flags"] = frags

    mask_fav = (
        (pd.to_numeric(df["score"], errors="coerce").fillna(0.0) >= float(min_score_favorite))
        & (df["deploy_tier"].isin(["A", "B"]))
    )
    flags["favorite"] = (
        df.loc[mask_fav, "pair_label"].astype(str).dropna().unique().tolist()
    )

    if "is_core_pair" in df.columns:
        flags["core"] = (
            df.loc[df["is_core_pair"].fillna(False), "pair_label"]
            .astype(str)
            .dropna()
            .unique()
            .tolist()
        )

    avoid_mask = (df["deploy_tier"].isin([max_tier_for_avoid, "unknown"])) | (
        df["fragility_flags"].apply(lambda fl: "extreme_z" in fl or "beta_unstable" in fl)
    )
    flags["avoid"] = (
        df.loc[avoid_mask, "pair_label"].astype(str).dropna().unique().tolist()
    )

    return flags


# ========= רעיון 1 – השוואת פרופילים על אותו Universe =========

def compare_profiles_for_universe(
    profile_names: List[str],
    pairs: List[Any],
    start_date: date,
    end_date: date,
    *,
    pair_legs_loader: PairLegsLoaderFn,
    base_signal_config: Optional[Dict[str, Any]] = None,
    max_workers: int = 8,
) -> Dict[str, pd.DataFrame]:
    """
    מריץ כמה profiles על אותו universe ומחזיר:
    {profile_name: df_signals}
    מאפשר לראות איך universe נראה בפרופילים שונים (MR/Short-Term/Vol-Arb...).
    """
    out: Dict[str, pd.DataFrame] = {}
    for pname in profile_names:
        df_signals = compute_universe_signals_profile(
            profile_name=pname,
            pairs=pairs,
            start_date=start_date,
            end_date=end_date,
            pair_legs_loader=pair_legs_loader,
            base_signal_config=base_signal_config,
            max_workers=max_workers,
            top_n=None,
            min_score=0.0,
        )
        out[pname] = df_signals
    return out


# ========= רעיון 2 – שינוי מאז ריצה קודמת (Delta Signals) =========

def diff_universe_signals(
    df_new: pd.DataFrame,
    df_old: pd.DataFrame,
    *,
    key_col: str = "pair_label",
    value_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    נותן Delta בין שני DataFrames של signals:
    - שימושי ל-"מה השתנה מהריצה הקודמת לדשבורד?"
    """
    if df_new is None or df_new.empty:
        return pd.DataFrame()
    if df_old is None or df_old.empty:
        return df_new.copy()

    if value_cols is None:
        value_cols = ["score", "abs_z", "corr", "half_life"]

    n = df_new.copy()
    o = df_old.copy()
    n = n[[key_col] + [c for c in value_cols if c in n.columns]]
    o = o[[key_col] + [c for c in value_cols if c in o.columns]]

    merged = n.merge(o, on=key_col, how="left", suffixes=("_new", "_old"))
    for c in value_cols:
        c_new = f"{c}_new"
        c_old = f"{c}_old"
        if c_new in merged.columns and c_old in merged.columns:
            merged[f"{c}_delta"] = merged[c_new] - merged[c_old]
    return merged


# ========= רעיון 3 – חלוקה לפי Execution Window (Intraday / Swing / Position) =========

def classify_execution_window_from_meta(meta: JSONDict) -> str:
    """
    מתייג חלון ביצוע (execution window) לפי:
    - signal_density_252
    - half_life (אם קיים)
    """
    try:
        density = float(meta.get("signal_density_252", np.nan))
    except Exception:
        density = np.nan

    hl = meta.get("half_life")
    try:
        hl = float(hl) if hl is not None else np.nan
    except Exception:
        hl = np.nan

    # מאוד גס – אפשר לחדד:
    if not np.isnan(density) and density > 100:
        return "intraday"
    if not np.isnan(hl):
        if hl <= 5:
            return "short_term"
        if hl <= 30:
            return "swing"
        return "position"
    return "unknown"


def tag_execution_window(df_signals: pd.DataFrame) -> pd.DataFrame:
    """
    מוסיף execution_window לכל זוג (intraday/short_term/swing/position).
    """
    if df_signals is None or df_signals.empty:
        return df_signals

    df = df_signals.copy()
    if "meta" not in df.columns:
        df["execution_window"] = "unknown"
        return df

    labels: List[str] = []
    for meta in df["meta"]:
        if isinstance(meta, dict):
            labels.append(classify_execution_window_from_meta({**meta, "half_life": meta.get("half_life")}))
        else:
            labels.append("unknown")
    df["execution_window"] = labels
    return df


# ========= רעיון 4 – יצירת backtest jobs מתוך signals =========

def signals_to_backtest_jobs(
    df_signals: pd.DataFrame,
    *,
    profile_name: str = "mean_reversion",
    max_jobs: int = 50,
) -> List[JSONDict]:
    """
    ממיר signals לזוגות jobs לבק-טסט:

    כל job:
      { "pair_label": ..., "profile_name": ..., "tier": ..., "score": ... }
    """
    if df_signals is None or df_signals.empty:
        return []

    df = df_signals.copy()
    if "score" not in df.columns:
        return []

    tiers = []
    for meta in df["meta"]:
        if isinstance(meta, dict):
            tiers.append(str(meta.get("deploy_tier", "unknown")))
        else:
            tiers.append("unknown")
    df["deploy_tier"] = tiers

    df = df.sort_values(["score", "abs_z", "points"], ascending=[False, False, False])
    jobs: List[JSONDict] = []
    for _, row in df.head(int(max_jobs)).iterrows():
        jobs.append(
            {
                "pair_label": str(row.get("pair_label")),
                "profile_name": profile_name,
                "tier": row.get("deploy_tier"),
                "score": float(row.get("score", 0.0)),
            }
        )
    return jobs


# ========= רעיון 5 – חיבור signals ל-RiskEngine (זיהוי pairs שבריריים לשיערוך סיכון) =========

def extract_fragile_pairs_for_risk(
    df_signals: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    מחלץ רשימות של זוגות שבריריים:
    - extreme_z_pairs
    - beta_unstable_pairs
    - low_corr_pairs
    """
    out = {
        "extreme_z_pairs": [],
        "beta_unstable_pairs": [],
        "low_corr_pairs": [],
    }
    if df_signals is None or df_signals.empty or "meta" not in df_signals.columns:
        return out

    ez: List[str] = []
    bu: List[str] = []
    lc: List[str] = []

    profile = get_signal_profile("mean_reversion")  # baseline ל-min_corr
    min_corr = profile.min_corr or 0.5

    for _, row in df_signals.iterrows():
        label = str(row.get("pair_label"))
        meta = row.get("meta") or {}
        if not isinstance(meta, dict):
            continue

        frags = meta.get("fragility_flags", []) or []
        if "extreme_z" in frags:
            ez.append(label)
        if "beta_unstable" in frags:
            bu.append(label)

        corr_val = row.get("corr")
        try:
            c = float(corr_val)
            if c < min_corr:
                lc.append(label)
        except Exception:
            pass

    out["extreme_z_pairs"] = sorted(set(ez))
    out["beta_unstable_pairs"] = sorted(set(bu))
    out["low_corr_pairs"] = sorted(set(lc))
    return out


# ========= רעיון 6 – גרסת "Simple View" לדשבורד =========

def build_simple_signals_view(df_signals: pd.DataFrame) -> pd.DataFrame:
    """
    יוצר תצוגה קומפקטית ונקייה ל-"מצב יומי" בדשבורד:
    עמודות:
      - pair_label
      - z
      - abs_z
      - corr
      - half_life
      - score
      - deploy_tier
    """
    if df_signals is None or df_signals.empty:
        return pd.DataFrame()

    df = df_signals.copy()
    cols = ["pair_label", "z", "abs_z", "corr", "half_life", "score"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    tiers = []
    for meta in df["meta"]:
        if isinstance(meta, dict):
            tiers.append(meta.get("deploy_tier", "unknown"))
        else:
            tiers.append("unknown")
    df["deploy_tier"] = tiers

    return df[cols + ["deploy_tier"]].sort_values("score", ascending=False).reset_index(drop=True)


# ========= רעיון 7 – Signals Heatmap Data (למספר מדדים) =========

def build_signals_heatmap_data(
    df_signals: pd.DataFrame,
    *,
    metrics: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    בונה heatmap matrices עבור מספר metrics (למשל abs_z, corr, score).
    """
    if df_signals is None or df_signals.empty:
        return {}

    if metrics is None:
        metrics = ["abs_z", "corr", "score"]

    out: Dict[str, pd.DataFrame] = {}
    for m in metrics:
        if m not in df_signals.columns:
            continue
        mat = signals_to_matrix(
            df_signals,
            value_col=m,
            index_col="sym_x",
            columns_col="sym_y",
            aggregate="mean",
        )
        out[m] = mat
    return out


# ========= רעיון 8 – Signals Summary ל-Matrix Tab =========

def build_matrix_signals_summary(
    df_signals: pd.DataFrame,
) -> JSONDict:
    """
    סיכום קטן לטאב מטריצה:
    - num_pairs
    - avg_abs_z / max_abs_z
    - avg_corr
    - median_half_life
    - num_high_z (למשל |Z|>=2.5)
    """
    if df_signals is None or df_signals.empty:
        return {
            "num_pairs": 0,
            "avg_abs_z": None,
            "max_abs_z": None,
            "avg_corr": None,
            "median_half_life": None,
            "num_high_z": 0,
        }

    df = df_signals.copy()
    out: JSONDict = {}
    out["num_pairs"] = int(len(df))

    if "abs_z" in df.columns:
        out["avg_abs_z"] = float(df["abs_z"].mean())
        out["max_abs_z"] = float(df["abs_z"].max())
        out["num_high_z"] = int((df["abs_z"].abs() >= 2.5).sum())
    else:
        out["avg_abs_z"] = out["max_abs_z"] = None
        out["num_high_z"] = 0

    if "corr" in df.columns:
        out["avg_corr"] = float(df["corr"].mean())
    else:
        out["avg_corr"] = None

    if "half_life" in df.columns:
        out["median_half_life"] = float(df["half_life"].median())
    else:
        out["median_half_life"] = None

    return out


# ========= רעיון 9 – Multi-Profile Delta עבור זוג בודד =========

def compare_pair_across_profiles(
    pair_obj: Any,
    start_date: date,
    end_date: date,
    *,
    profile_names: List[str],
    pair_legs_loader: PairLegsLoaderFn,
    base_signal_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    משווה איך אותו pair נראה בפרופילים שונים:
    מחזיר DataFrame עם שורה לכל profile:
      - profile_name
      - z / abs_z / corr / half_life / score_approx
    """
    rows: List[JSONDict] = []
    for pname in profile_names:
        try:
            sig = compute_pair_signal(
                pair_obj,
                start_date,
                end_date,
                profile_name=pname,
                pair_legs_loader=pair_legs_loader,
                base_signal_config=base_signal_config,
            )
        except Exception as e:
            logger.warning("compare_pair_across_profiles: %s failed: %s", pname, e)
            continue

        if sig is None or sig.status != "ok":
            continue

        prof = get_signal_profile(pname)
        score_approx = _score_pair_signal(sig, prof)
        rows.append(
            {
                "profile_name": pname,
                "z": sig.z,
                "abs_z": sig.abs_z,
                "corr": sig.corr,
                "half_life": sig.half_life,
                "score_approx": score_approx,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score_approx", ascending=False).reset_index(drop=True)


# ========= רעיון 10 – Signals → JSON snapshot ל-AppContext או ל-Export =========

def signals_to_snapshot(
    df_signals: pd.DataFrame,
    *,
    top_n: int = 50,
) -> JSONDict:
    """
    יוצר snapshot קומפקטי של signals:
    - top_n rows as records
    - summary
    שימושי ל:
      - AppContext snapshots
      - Export Center
      - Agents
    """
    if df_signals is None or df_signals.empty:
        return {
            "signals_top": [],
            "summary": summarize_universe_signals(df_signals),
        }

    df = df_signals.copy()
    df = df.sort_values("score", ascending=False) if "score" in df.columns else df
    df_top = df.head(int(top_n))

    return {
        "signals_top": df_top.to_dict(orient="records"),
        "summary": summarize_universe_signals(df),
    }

# ========= Part 5 — AppContext / Experiments / Reporting Integration (HF-grade) =========

import json
from typing import Any

# ננסה להתחבר ל-core.app_context אם זמין, אבל לא נדרוש את זה כתלות קשיחה
try:  # pragma: no cover
    from core.app_context import (
        register_experiment_run as _ctx_register_experiment_run,
        ctx_to_markdown as _ctx_to_markdown,
    )
except Exception:  # pragma: no cover
    _ctx_register_experiment_run = None  # type: ignore
    _ctx_to_markdown = None  # type: ignore


def signals_to_ctx_features(
    df_signals: pd.DataFrame,
    *,
    profile_name: str,
) -> JSONDict:
    """
    בונה וקטור פיצ'רים קומפקטי מה-signals_df לצורך ctx.state_features / Insights / RL.

    כולל:
    - summary מהיר (summarize_universe_signals)
    - מדדי פיזור score / abs_z
    - counts לפי tiers / execution_intensity / anomaly
    """
    summary = summarize_universe_signals(df_signals)

    features: JSONDict = {
        "profile_name": profile_name,
        "count_ok": summary["count_ok"],
        "avg_abs_z": summary["avg_abs_z"],
        "max_abs_z": summary["max_abs_z"],
        "avg_corr": summary["avg_corr"],
        "hl_median": summary["hl_median"],
        "avg_score": summary["avg_score"],
        "core_pairs_count": summary["core_pairs_count"],
        "z_anomaly_count": summary["z_anomaly_count"],
    }

    # סטיות / skewness של score ותפלגות הערכים
    if df_signals is not None and not df_signals.empty and "score" in df_signals.columns:
        s = pd.to_numeric(df_signals["score"], errors="coerce").dropna()
        if not s.empty:
            features["score_std"] = float(s.std(ddof=0))
            features["score_skew"] = float(s.skew())
        else:
            features["score_std"] = None
            features["score_skew"] = None
    else:
        features["score_std"] = None
        features["score_skew"] = None

    # פיזור half-life
    if df_signals is not None and not df_signals.empty and "half_life" in df_signals.columns:
        hl = pd.to_numeric(df_signals["half_life"], errors="coerce").dropna()
        if not hl.empty:
            features["hl_min"] = float(hl.min())
            features["hl_max"] = float(hl.max())
        else:
            features["hl_min"] = None
            features["hl_max"] = None
    else:
        features["hl_min"] = None
        features["hl_max"] = None

    # tier_counts
    tier_counts = summary.get("tier_counts", {}) or {}
    for tier_name, cnt in tier_counts.items():
        features[f"tier_{tier_name}_count"] = int(cnt)

    # execution_intensity_counts
    exec_counts = summary.get("execution_intensity_counts", {}) or {}
    for k, cnt in exec_counts.items():
        features[f"exec_intensity_{k}_count"] = int(cnt)

    return features


def attach_signals_features_to_ctx(
    ctx: Any,
    df_signals: pd.DataFrame,
    *,
    profile_name: str,
    slot: str = "signals",
    snapshot_top_n: int = 50,
) -> Any:
    """
    מעדכן AppContext (אם הוא אובייקט ctx "אמיתי") עם:

    - ctx.state_features[slot] ← signals_to_ctx_features(...)
    - ctx.controls["signals_snapshot"][slot] ← snapshot קומפקטי (top_n)
    - ctx.tags מתעשר ב-tags רלוונטיים (לדוגמה: signals_profile_..., signals_core_pairs>0)

    הפונקציה חוזרת עם ctx (אותו אובייקט, אחרי עדכון).
    """
    if ctx is None:
        return ctx

    # state_features
    features = signals_to_ctx_features(df_signals, profile_name=profile_name)

    try:
        state_features = getattr(ctx, "state_features", None)
        if not isinstance(state_features, dict):
            state_features = {}
        state_features[f"{slot}:{profile_name}"] = features
        setattr(ctx, "state_features", state_features)
    except Exception:
        pass

    # snapshot בתוך controls (לשימוש בטאבים אחרים / Export Center)
    try:
        controls = getattr(ctx, "controls", None)
        if not isinstance(controls, dict):
            controls = {}
        snapshots = controls.get("signals_snapshot", {})
        if not isinstance(snapshots, dict):
            snapshots = {}
        snapshots[f"{slot}:{profile_name}"] = signals_to_snapshot(df_signals, top_n=snapshot_top_n)
        controls["signals_snapshot"] = snapshots
        setattr(ctx, "controls", controls)
    except Exception:
        pass

    # tags
    try:
        tags = getattr(ctx, "tags", None)
        if not isinstance(tags, dict):
            tags = {}
        tags["signals_profile"] = profile_name
        if features.get("core_pairs_count", 0) > 0:
            tags["signals_core_pairs"] = "yes"
        else:
            tags["signals_core_pairs"] = "no"
        setattr(ctx, "tags", tags)
    except Exception:
        pass

    return ctx


def register_signals_experiment(
    ctx: Any,
    df_signals: pd.DataFrame,
    *,
    profile_name: str,
    extra_meta: Optional[JSONDict] = None,
) -> None:
    """
    רושם ריצת "experiment" של signals ל-AppContext experiment log (אם זמין):

    • משתמש register_experiment_run מה-app_context אם קיים.
    • kpis בסיסיים מה-signals summary.
    """
    if _ctx_register_experiment_run is None:
        return

    summary = summarize_universe_signals(df_signals)
    kpis = {
        "total_pairs": summary["count_ok"],
        "avg_abs_z": summary["avg_abs_z"],
        "max_abs_z": summary["max_abs_z"],
        "avg_corr": summary["avg_corr"],
        "median_half_life": summary["hl_median"],
        "core_pairs_count": summary["core_pairs_count"],
        "z_anomaly_count": summary["z_anomaly_count"],
    }

    meta = dict(extra_meta or {})
    meta["profile_name"] = profile_name

    try:
        _ctx_register_experiment_run(ctx, kpis=kpis, extra_meta=meta)
    except Exception as e:
        logger.warning("register_signals_experiment failed: %s", e)


def build_signals_markdown_report(
    ctx: Any,
    df_signals: pd.DataFrame,
    *,
    profile_name: str,
) -> str:
    """
    בונה דוח Markdown על signals + context (אם ctx_to_markdown זמין):

    • חלק ראשון: ctx (אם אפשר).
    • חלק שני: Summary על universe signals.
    • חלק שלישי: טבלה קטנה של top-20 signals.
    """
    parts: List[str] = []

    # חלק 1 – AppContext (אם אפשר)
    if _ctx_to_markdown is not None and ctx is not None:
        try:
            parts.append(_ctx_to_markdown(ctx))
        except Exception:
            parts.append(f"# Context Report\n\nRun ID: `{getattr(ctx, 'run_id', 'N/A')}`")
    else:
        parts.append(f"# Context Report\n\nRun ID: `{getattr(ctx, 'run_id', 'N/A')}`")

    parts.append("\n---\n")
    parts.append(f"## Signals Report — Profile `{profile_name}`\n")

    # Summary
    summary = summarize_universe_signals(df_signals)
    parts.append("### Summary")
    parts.append("")
    parts.append(f"- Number of pairs with signals: **{summary['count_ok']}**")
    if summary["avg_abs_z"] is not None:
        parts.append(f"- Average |Z|: **{summary['avg_abs_z']:.2f}**, Max |Z|: **{summary['max_abs_z']:.2f}**")
    if summary["avg_corr"] is not None:
        parts.append(f"- Average corr: **{summary['avg_corr']:.3f}**")
    if summary["hl_median"] is not None:
        parts.append(f"- Median Half-Life: **{summary['hl_median']:.1f}** days")
    parts.append(f"- Core pairs count: **{summary['core_pairs_count']}**")
    parts.append(f"- Extreme Z anomalies: **{summary['z_anomaly_count']}**")
    parts.append("")

    # tier breakdown
    if summary["tier_counts"]:
        parts.append("**Tier breakdown:**")
        for t, cnt in summary["tier_counts"].items():
            parts.append(f"- Tier `{t}`: {cnt} pairs")
        parts.append("")

    # Top table (top-20)
    if df_signals is not None and not df_signals.empty:
        df = df_signals.copy()
        if "score" in df.columns:
            df = df.sort_values("score", ascending=False)
        df_top = df.head(20)[["pair_label", "z", "abs_z", "corr", "half_life", "score"]].copy()
        df_top = df_top.fillna("N/A")

        parts.append("### Top 20 Signals")
        parts.append("")
        parts.append("```text")
        parts.append(df_top.to_string(index=False))
        parts.append("```")

    return "\n".join(parts)


def filter_signals_for_execution_mode(
    df_signals: pd.DataFrame,
    *,
    mode: str = "live",
    max_pairs: int = 50,
) -> pd.DataFrame:
    """
    מסנן signals לפי מצב execution:

    - mode="live":
        * רק tier A/B.
        * אין fragility קשה (extreme_z/beta_unstable).
    - mode="paper":
        * אפשר גם tier C.
    - mode="research":
        * הכל, אבל מסומן מה מסוכן.
    """
    if df_signals is None or df_signals.empty:
        return pd.DataFrame()

    df = df_signals.copy()
    if "meta" not in df.columns:
        return df.head(int(max_pairs))

    tiers: List[str] = []
    flags: List[List[str]] = []
    for meta in df["meta"]:
        if isinstance(meta, dict):
            tiers.append(str(meta.get("deploy_tier", "unknown")))
            flags.append(list(meta.get("fragility_flags", []) or []))
        else:
            tiers.append("unknown")
            flags.append([])
    df["deploy_tier"] = tiers
    df["fragility_flags"] = flags

    if mode == "live":
        mask = (
            df["deploy_tier"].isin(["A", "B"])
            & ~df["fragility_flags"].apply(lambda fl: "extreme_z" in fl or "beta_unstable" in fl)
        )
        df = df[mask]
    elif mode == "paper":
        # פחות הקפדה – רק להימנע ממקרים מאוד קיצוניים
        mask = ~df["fragility_flags"].apply(lambda fl: "extreme_z" in fl and "beta_unstable" in fl)
        df = df[mask]
    else:  # research
        # לא מסננים, רק מסמנים קולונה "risk_flag"
        risk_flag = []
        for fl in df["fragility_flags"]:
            if "extreme_z" in fl and "beta_unstable" in fl:
                risk_flag.append("high")
            elif "extreme_z" in fl or "beta_unstable" in fl:
                risk_flag.append("medium")
            else:
                risk_flag.append("low")
        df["execution_risk_flag"] = risk_flag

    if "score" in df.columns:
        df = df.sort_values("score", ascending=False)
    return df.head(int(max_pairs))


def signals_quality_diagnostics(
    df_diag: pd.DataFrame,
) -> JSONDict:
    """
    דיאגנוסטיקה על universe diagnostics:
    - מספר זוגות "skipped" לפי reason
    - מספר errors לפי reason
    """
    if df_diag is None or df_diag.empty:
        return {"skipped": {}, "errors": {}}

    df = df_diag.copy()

    skipped = df[df["status"] == "skipped"]
    errors = df[df["status"] == "error"]

    skipped_counts: Dict[str, int] = {}
    errors_counts: Dict[str, int] = {}

    if not skipped.empty:
        skipped_counts = (
            skipped["reason"].astype(str).value_counts().to_dict()
        )
    if not errors.empty:
        errors_counts = (
            errors["reason"].astype(str).value_counts().to_dict()
        )

    return {
        "skipped": skipped_counts,
        "errors": errors_counts,
    }


def signals_profile_comparison_summary(
    profiles_to_signals: Dict[str, pd.DataFrame],
) -> JSONDict:
    """
    סיכום השוואתי בין פרופילים שונים על אותו universe:

    מחזיר עבור כל profile:
    - num_pairs
    - avg_abs_z
    - avg_corr
    - median_half_life
    - fraction_high_quality (quality_score>=70)
    """
    summary: JSONDict = {}
    for pname, df in profiles_to_signals.items():
        if df is None or df.empty:
            summary[pname] = {
                "num_pairs": 0,
                "avg_abs_z": None,
                "avg_corr": None,
                "median_half_life": None,
                "fraction_high_quality": None,
            }
            continue

        s: JSONDict = {}
        s["num_pairs"] = int(len(df))
        if "abs_z" in df.columns:
            s["avg_abs_z"] = float(df["abs_z"].mean())
        else:
            s["avg_abs_z"] = None

        if "corr" in df.columns:
            s["avg_corr"] = float(df["corr"].mean())
        else:
            s["avg_corr"] = None

        if "half_life" in df.columns:
            s["median_half_life"] = float(df["half_life"].median())
        else:
            s["median_half_life"] = None

        if "quality_score" in df.columns:
            qs = pd.to_numeric(df["quality_score"], errors="coerce").dropna()
            if not qs.empty:
                s["fraction_high_quality"] = float((qs >= 70.0).mean())
            else:
                s["fraction_high_quality"] = None
        else:
            s["fraction_high_quality"] = None

        summary[pname] = s

    return summary

# ========= Part 6 — ML Datasets, Optimisation Hooks, Debug & Public API =========

def signals_to_ml_dataset(
    df_signals: pd.DataFrame,
    backtest_df: Optional[pd.DataFrame] = None,
    *,
    pair_col_signals: str = "pair_label",
    pair_col_bt: str = "Pair",
    target_col_bt: str = "Sharpe",
    extra_targets: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
    """
    בונה מערך ML (X, y) מאוסף signals + תוצאות Backtest (אם זמינות):

    X:
      - abs_z, corr, half_life, quality_score, score, z_norm, corr_norm, hl_norm, ...
      - features מתוך meta (z_consistency_score, stationarity_strength, cointegration_strength, edge_hint, ...)

    y:
      - מטרת יעד אחת (לדוגמה Sharpe מתוך backtest_df).

    extra_targets:
      - רשימת עמודות נוספות מה-backtest להחזיר בנפרד (לניתוח/ריבוי יעדים).

    מחזיר:
      - X  (DataFrame)
      - y  (Series)
      - Y_extra (DataFrame או None)
    """
    if df_signals is None or df_signals.empty:
        return pd.DataFrame(), pd.Series(dtype=float), None

    df = df_signals.copy()

    # אם יש backtest_df – נחבר אותו לפי pair
    if backtest_df is not None and not backtest_df.empty:
        bt = backtest_df.copy()
        if pair_col_bt not in bt.columns:
            # לא ניתן לחבר
            bt = None
        else:
            bt = bt.copy()
            bt[pair_col_bt] = bt[pair_col_bt].astype(str)
            df[pair_col_signals] = df[pair_col_signals].astype(str)
            # ניקח מטריקות ממוצעות per pair
            agg_cols = [target_col_bt] + (extra_targets or [])
            for c in agg_cols:
                if c not in bt.columns:
                    bt[c] = np.nan
            bt_agg = bt.groupby(pair_col_bt)[agg_cols].mean().reset_index()
            bt_agg = bt_agg.rename(columns={pair_col_bt: pair_col_signals})
            df = df.merge(bt_agg, on=pair_col_signals, how="left")
    else:
        bt = None

    # בניית feature set
    base_cols = [
        "abs_z",
        "corr",
        "half_life",
        "points",
        "score",
        "z_norm",
        "corr_norm",
        "hl_norm",
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan

    feat_df: pd.DataFrame = df[base_cols].copy()

    # meta-derived features (אם קיימת עמודת meta)
    if "meta" in df.columns:
        meta_rows = df["meta"].apply(lambda x: x if isinstance(x, dict) else {})
        meta_df = pd.json_normalize(meta_rows)
        # בוחרים כמה פיצ'רים שימושיים אם יש
        meta_cols_pref = [
            "z_consistency_score",
            "stationarity_strength",
            "cointegration_strength",
            "beta_stability",
            "signal_density_252",
            "edge_hint",
        ]
        for mc in meta_cols_pref:
            if mc in meta_df.columns:
                feat_df[mc] = pd.to_numeric(meta_df[mc], errors="coerce")
        # אפשר להוסיף עוד לפי הצורך
    # y (יעד) – target_col_bt אם קיים
    if backtest_df is not None and target_col_bt in df.columns:
        y = pd.to_numeric(df[target_col_bt], errors="coerce")
    else:
        y = pd.Series(dtype=float)

    # extra targets
    Y_extra = None
    if backtest_df is not None and extra_targets:
        cols_exist = [c for c in extra_targets if c in df.columns]
        if cols_exist:
            Y_extra = df[cols_exist].apply(pd.to_numeric, errors="coerce")

    # ניקוי NaN (אפשר להחליף ב-0/ממוצע – כאן נשאיר NaN, ל-ML חיצוני)
    return feat_df, y, Y_extra


def suggest_param_ranges_from_signals(
    df_signals: pd.DataFrame,
    *,
    z_window_current: int,
) -> Dict[str, Dict[str, float]]:
    """
    רעיון: להשתמש ב-signals כדי להציע טווחי פרמטרים לאופטימיזציה (Optuna וכו'):

    מדוע?
    - אם HL רוב הזמן קצרה → כדאי לאפשר קרובים יותר ל-window קטן.
    - אם Corr גבוה → אפשר לצמצם טווח min_corr.
    - אם abs_z עושה peaks גבוהים מדי → יש לחשוב אולי להוריד entry_z.

    מחזיר dict בסגנון:
      {
        "entry_z": {"min": ..., "max": ...},
        "exit_z":  {"min": ..., "max": ...},
        "z_window": {"min": ..., "max": ...},
        "hl_min": {"min": ..., "max": ...},
        "hl_max": {"min": ..., "max": ...},
      }
    """
    ranges: Dict[str, Dict[str, float]] = {}

    if df_signals is None or df_signals.empty:
        return ranges

    df = df_signals.copy()

    # HL stats
    hl = pd.to_numeric(df.get("half_life", pd.Series(dtype=float)), errors="coerce").dropna()
    if not hl.empty:
        hl_median = float(hl.median())
        hl_min = float(hl.quantile(0.1))
        hl_max = float(hl.quantile(0.9))
    else:
        hl_median = float(z_window_current)
        hl_min = max(1.0, float(z_window_current) / 3)
        hl_max = float(z_window_current) * 3

    # abs_z stats
    absz = pd.to_numeric(df.get("abs_z", pd.Series(dtype=float)), errors="coerce").dropna()
    if not absz.empty:
        absz_med = float(absz.median())
        absz_80 = float(absz.quantile(0.8))
        absz_95 = float(absz.quantile(0.95))
    else:
        absz_med, absz_80, absz_95 = 1.0, 2.0, 3.0

    # Corr stats
    corr = pd.to_numeric(df.get("corr", pd.Series(dtype=float)), errors="coerce").dropna()
    if not corr.empty:
        corr_median = float(corr.median())
    else:
        corr_median = 0.5

    # הצעות:
    ranges["z_window"] = {
        "min": max(5.0, hl_min / 2),
        "max": max(hl_max * 1.5, float(z_window_current)),
    }
    ranges["entry_z"] = {
        "min": max(0.5, absz_med * 0.7),
        "max": max(2.0, absz_95 * 1.2),
    }
    ranges["exit_z"] = {
        "min": 0.0,
        "max": max(1.5, absz_med),
    }
    ranges["hl_min"] = {
        "min": max(1.0, hl_min * 0.5),
        "max": max(2.0, hl_min * 1.1),
    }
    ranges["hl_max"] = {
        "min": max(5.0, hl_median),
        "max": max(hl_max * 1.2, hl_median * 2.0),
    }
    ranges["min_corr"] = {
        "min": min(0.0, corr_median - 0.2),
        "max": min(0.9, corr_median + 0.2),
    }

    return ranges


def debug_pair_signal(
    sig: PairSignal,
    *,
    include_meta: bool = True,
) -> str:
    """
    מחזיר מחרוזת טקסט קצרה לאבחון PairSignal:

    - pair_label
    - z / corr / HL / score_approx
    - flags / tier
    """
    profile = get_signal_profile(sig.profile_name)
    score_approx = _score_pair_signal(sig, profile)
    lines = [
        f"Pair: {sig.pair_label} ({sig.sym_x}-{sig.sym_y})",
        f"z={sig.z:.2f}, |z|={sig.abs_z:.2f}, corr={sig.corr:.3f}, HL={sig.half_life}",
        f"score≈{score_approx:.3f}, points={sig.points}, status={sig.status}",
    ]
    if include_meta:
        deploy_tier = sig.meta.get("deploy_tier", "unknown")
        flags = sig.meta.get("fragility_flags", [])
        lines.append(f"tier={deploy_tier}, fragility_flags={flags}")
    return "\n".join(lines)


def debug_universe_signals(
    df_signals: pd.DataFrame,
    *,
    max_rows: int = 10,
) -> str:
    """
    Text debug של universe signals:

    - Summary קצר (summarize_universe_signals)
    - כמה שורות מה-top.
    """
    summary = summarize_universe_signals(df_signals)
    lines = [
        f"Universe Signals: {summary['count_ok']} pairs",
        f"avg_abs_z={summary['avg_abs_z']}, max_abs_z={summary['max_abs_z']}",
        f"avg_corr={summary['avg_corr']}, hl_median={summary['hl_median']}",
        f"avg_score={summary['avg_score']}, core_pairs={summary['core_pairs_count']}",
        f"z_anomalies={summary['z_anomaly_count']}",
        "",
        "Top rows:",
    ]
    if df_signals is not None and not df_signals.empty:
        df = df_signals.copy()
        if "score" in df.columns:
            df = df.sort_values("score", ascending=False)
        df_top = df.head(int(max_rows))[["pair_label", "z", "abs_z", "corr", "half_life", "score"]].fillna("N/A")
        lines.append(df_top.to_string(index=False))
    else:
        lines.append("(no data)")
    return "\n".join(lines)


def export_signals_to_json(
    df_signals: pd.DataFrame,
    *,
    path: Optional[Path] = None,
    top_n: int = 100,
) -> Optional[Path]:
    """
    כותב signals ל-JSON (top_n בלבד) כדי לטעון אותם חיצונית/ל-Agent.

    כל רשומה כוללת:
    - fields מה-signals
    - meta (כ-dict)
    """
    if df_signals is None or df_signals.empty:
        return None

    try:
        base_dir = PROJECT_ROOT / "logs" / "signals_snapshots"
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        base_dir = PROJECT_ROOT

    if path is None:
        fname = f"signals_snapshot_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        path = base_dir / fname

    df = df_signals.copy()
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False)
    df_top = df.head(int(top_n))

    # להפוך ל-list של dictים JSON-safe
    records: List[JSONDict] = []
    for _, row in df_top.iterrows():
        rec: JSONDict = {}
        for col, val in row.items():
            if col == "meta":
                if isinstance(val, dict):
                    rec["meta"] = val
                else:
                    rec["meta"] = {}
            else:
                # types בסיסיים בלבד
                if isinstance(val, (float, int, str, bool)) or val is None:
                    rec[col] = val
                else:
                    rec[col] = str(val)
        records.append(rec)

    payload = {
        "ts": datetime.now(timezone.utc)().isoformat() + "Z",
        "count": len(records),
        "signals": records,
    }

    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
    except Exception as e:
        logger.warning("export_signals_to_json failed: %s", e)
        return None


# ========= Public API for core/signals_engine =========

__all__ = [
    # Models
    "SignalProfile",
    "PairSignal",
    "UniverseSignals",
    # Profiles
    "SIGNAL_PROFILES",
    "get_signal_profile",
    # Pair-level
    "compute_pair_signal",
    "compute_pair_signal_with_reason",
    # Universe-level
    "compute_universe_signals",
    "compute_universe_signals_with_diagnostics",
    "compute_universe_signals_profile",
    "summarize_universe_signals",
    # Dashboard / Scan / Matrix wrappers
    "compute_dashboard_signals",
    "compute_smart_scan_signals",
    "signals_to_matrix",
    "attach_signals_to_universe_meta",
    "signals_to_watchlist",
    "signals_to_pair_flags",
    "compare_profiles_for_universe",
    "diff_universe_signals",
    "compare_pair_across_profiles",
    "build_simple_signals_view",
    "build_signals_heatmap_data",
    "build_matrix_signals_summary",
    "signals_to_snapshot",
    # ML / Experiments / Debug
    "signals_to_ml_dataset",
    "suggest_param_ranges_from_signals",
    "signals_to_ctx_features",
    "attach_signals_features_to_ctx",
    "register_signals_experiment",
    "build_signals_markdown_report",
    "filter_signals_for_execution_mode",
    "signals_quality_diagnostics",
    "signals_profile_comparison_summary",
    "debug_pair_signal",
    "debug_universe_signals",
    "export_signals_to_json",
    "signals_to_backtest_jobs",
    "extract_fragile_pairs_for_risk",
    "tag_execution_window",
]
