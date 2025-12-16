# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import logging
import json   
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
# שורש הפרויקט — נשתמש בו בשביל risk bundles
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ========= Type aliases =========

JSONDict = Dict[str, Any]
Float = Optional[float]


# ========= RiskLimits (כולל דינמיקה, פרופילים ויעדים) =========

@dataclass
class RiskLimits:
    """
    מגבלות סיכון בסיסיות + דינמיות.
    זה ה-"contract" בין הקרן לבין כל מנועי המסחר/אסטרטגיות.
    """

    # מגבלות בסיסיות (closing)
    max_daily_loss_pct: float = 0.03        # הפסד יומי יחסי (3%)
    max_weekly_loss_pct: float = 0.06       # הפסד שבועי (לוגי)
    max_monthly_loss_pct: float = 0.12      # הפסד חודשי (לוגי)

    # מגבלות תוך-יומיות / ממונפות
    max_intraday_loss_pct: float = 0.05     # הפסד תוך־יומי (ביחס להון)
    max_gross_leverage: float = 3.0         # מינוף כולל מותר
    max_net_leverage: float = 2.0           # מינוף נטו מותר

    # Drawdown היסטורי
    max_drawdown_pct: float = 0.20          # DD מקסימלי "קשיח" (20%)

    # מקרו / תנודתיות
    vix_kill_threshold: float = 35.0
    anomaly_trigger_min_frac: float = 0.20  # אחוז ימים אנומליים (60 יום אחרונים) שמדליק חוק

    # גבולות תחתונים – כדי שהידוק דינמי לא יחרבש את הלוגיקה
    min_gross_leverage: float = 1.0         # פחות מזה = אין מסחר אמיתי
    min_daily_loss_pct: float = 0.01        # פחות מזה = אי אפשר כמעט לזוז

    # פרופיל לוגי: neutral / risk_on / risk_off / crisis
    profile_name: str = "neutral"

    # פקטור הידוק/הקלה (0.5–1.5)
    tightened_factor: float = 1.0

    # יעדים (לא מגבלות) – לצורך Risk Goals
    target_vol_pct: float = 0.15            # סטיית תקן שנתית רצויה
    target_dd_pct: float = 0.15            # DD "רצוי" (לא קשיח)
    target_sharpe_min: float = 1.0
    target_sharpe_max: float = 2.5

    # Risk Budget per bucket (למשל per strategy/cluster)
    # דוגמה: {"MR_core": 0.4, "trend": 0.3, "carry": 0.3}
    risk_budget_per_bucket: JSONDict = field(default_factory=dict)

    # overrides לפי משטרים (אופציונלי – משמש לתצוגה/חישוב דינמי)
    # דוגמה: {"crisis": {"max_gross_leverage": 1.5, "max_drawdown_pct": 0.15}}
    regime_overrides: JSONDict = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return asdict(self)


# ========= RiskScenario (ל-Scenario Deck) =========

@dataclass
class RiskScenario:
    """
    תרחיש מאקרו/שוק ל-What-If:

    דוגמאות:
    - Mild Correction (SPY -5%)
    - Shock (SPY -10%)
    - Crisis (SPY -20%)
    - Rates Shock (TLT -10%)
    - FX Shock (DXY +5%)
    """

    name: str
    spy_shock_pct: float
    vix_target: Optional[float] = None
    description: str = ""

    # אופציונלי – shocks נוספים, למשל {"TLT": -0.1, "HYG": -0.08}
    extra_shocks: JSONDict = field(default_factory=dict)

    # משטר מאקרו משוער שבו התרחיש מתרחש ("risk_off", "crisis"...)
    assumed_macro_regime: Optional[str] = None

    # דירוג חומרה (0–100)
    severity_score: float = 50.0

    # אפשרות להגדיר הערות על בקשת פעולה לוגית (למשל "switch_to_risk_off")
    suggested_profile: Optional[str] = None

    def to_dict(self) -> JSONDict:
        return asdict(self)


# ========= BucketRisk (תרומת סיכון לפי Bucket) =========

@dataclass
class BucketRisk:
    """
    סיכונים ל-Bucket בודד (Strategy / Sector / Cluster / Regime):

    - total_pnl: רווח/הפסד מצטבר
    - vol_pct: סטיית תקן משוערת (שנתית) של PnL / Equity
    - max_dd_pct: drawdown מקסימלי (ביחס ליקום/הון)
    - contribution_to_risk: אחוז תרומה לסיכון הכולל (למשל לפי VaR/ES, normalized)
    """

    name: str
    total_pnl: float
    vol_pct: float
    max_dd_pct: float
    contribution_to_risk: float

    # מטריקות נוספות
    sharpe: Optional[float] = None
    beta_to_spy: Optional[float] = None
    var_95_pct: Optional[float] = None
    es_95_pct: Optional[float] = None

    # חשיפות
    gross_exposure: Optional[float] = None
    net_exposure: Optional[float] = None

    # דיאגנוסטיקה
    sample_size: Optional[int] = None            # מספר תצפיות
    worst_pnl: Optional[float] = None            # PnL גרוע ביותר
    worst_dd_pct: Optional[float] = None         # DD גרוע ביותר בתקופה

    def to_dict(self) -> JSONDict:
        return asdict(self)


# ========= RiskState (State מלא של הקרן) =========

@dataclass
class RiskState:
    """
    מצב סיכון נוכחי (state) של הקרן:

    - PnL/Equity: daily / weekly / monthly / YTD
    - Drawdown: max_dd_pct
    - Leverage/Vol: gross/net leverage, realized_vol
    - Market risk: VIX, macro_regime, matrix_crowded_regime
    - Anomalies: anomaly_score/flag
    - Tail / Stability / Liquidity: tail_risk_score, wf_stability_score, liquidity_stress_score
    - bucket_risk_summary: תמצית per bucket (למשל Strategy / Cluster)
    """

    # Equity & PnL
    equity_today: Float = None
    equity_yday: Float = None
    daily_pnl_abs: Float = None
    daily_pnl_pct: Float = None
    weekly_pnl_pct: Float = None
    monthly_pnl_pct: Float = None
    ytd_pnl_pct: Float = None

    # Drawdown
    max_dd_pct: Float = None          # מתוך היסטוריית equity

    # Leverage & Vol
    gross_leverage: Float = None
    net_leverage: Float = None
    target_leverage: Float = None     # אם מוגדר יעד
    realized_vol_pct: Float = None    # סטיית תקן משוערת (שנתית) על PnL/Equity

    # Market risk
    vix: Float = None
    macro_regime: Optional[str] = None            # "risk_on" / "risk_off" / ...
    matrix_crowded_regime: Optional[str] = None   # "Highly Crowded" / "Crowded" / ...

    # Anomaly detection
    anomaly_score: Float = None       # fraction of anomaly days in last window
    anomaly_flag: Optional[bool] = None

    # Tail / Stability / Liquidity
    tail_risk_score: Float = None         # score 0–∞ (גבוה → מסוכן)
    wf_stability_score: Float = None      # יציבות WF (0–1)
    liquidity_stress_score: Float = None  # רגישות למחסור נזילות (0–1+)

    # Bucket-level summary (זיקוק של רשימת BucketRisk):
    # לדוגמה:
    # {
    #   "top_risk_buckets": [{"name":...,"contribution_to_risk":...}, ...],
    #   "total_buckets": 5,
    # }
    bucket_risk_summary: JSONDict = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return asdict(self)


# ========= RiskEvent (יומן אירועי סיכון) =========

@dataclass
class RiskEvent:
    """
    אירוע סיכון לוגי:

    - event_type: "kill_switch_trigger" / "limit_breach" / "scenario_test" / "liquidity_stress" / ...
    - category: "risk" / "data" / "research" / "execution"
    - severity: 0–100
    - message: טקסט קריא
    - meta: פרטים נוספים (run_id, scenario_name, bucket, ...)
    """

    event_type: str
    category: str
    severity: float
    message: str
    ts_utc: str = field(default_factory=lambda: datetime.now(timezone.utc)().isoformat() + "Z")
    run_id: Optional[str] = None
    meta: JSONDict = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> JSONDict:
        return asdict(self)


# ========= KillSwitchDecision =========

@dataclass
class KillSwitchDecision:
    """
    החלטת Kill-Switch:

    modes:
    - "off"    – אין פעולה
    - "soft"   – אין פתיחת פוזיציות חדשות
    - "reduce" – הפחתת מינוף / Exposure
    - "hard"   – עצירה מלאה
    """

    enabled: bool
    mode: str
    reason: str
    triggered_rules: List[str]
    recommended_actions: List[str]
    state: RiskState
    limits: RiskLimits
    severity_score: float           # 0–100
    time_to_recheck_sec: int = 300  # תוך כמה זמן לבדוק מחדש

    # scale factor להצעה ל-Executor (idea 18)
    scaling_factor: float = 1.0

    def to_dict(self) -> JSONDict:
        data = asdict(self)
        data["state"] = self.state.to_dict()
        data["limits"] = self.limits.to_dict()
        return data


# ========= StressResult (תוצאה של תרחיש / shock) =========

@dataclass
class StressResult:
    """
    תוצאה של סימולציית סטרס:

    - scenario_name
    - equity_today, equity_after
    - approx_pnl_change (abs / pct)
    - approx_dd_effect_pct
    - worst_buckets: רשימת BucketRisk בעייתיים במיוחד
    - risk_score_delta: השינוי ב-Coherent Risk Score
    """

    scenario_name: str
    equity_today: Float
    equity_after: Float
    approx_pnl_change: Float
    approx_pnl_change_pct: Float
    approx_dd_effect_pct: Float
    risk_score_delta: Float = 0.0
    worst_buckets: List[BucketRisk] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        d = asdict(self)
        d["worst_buckets"] = [b.to_dict() for b in self.worst_buckets]
        return d


# ========= Preset scenarios (Deck בסיסי) =========

SCENARIOS: List[RiskScenario] = [
    RiskScenario(
        name="Mild Correction (SPY -5%)",
        spy_shock_pct=-0.05,
        vix_target=25.0,
        description="תיקון מתון בשוק, תנודתיות עולה אך לא משבר.",
        assumed_macro_regime="risk_off",
        severity_score=30.0,
        suggested_profile="risk_off",
    ),
    RiskScenario(
        name="Shock (SPY -10%)",
        spy_shock_pct=-0.10,
        vix_target=35.0,
        description="זעזוע חריף, בדיקת עמידות DD ונזילות.",
        assumed_macro_regime="risk_off",
        severity_score=60.0,
        suggested_profile="risk_off",
    ),
    RiskScenario(
        name="Crisis (SPY -20%)",
        spy_shock_pct=-0.20,
        vix_target=45.0,
        description="תרחיש משברי – Worst-Case לקרן.",
        assumed_macro_regime="crisis",
        severity_score=90.0,
        suggested_profile="crisis",
    ),
    RiskScenario(
        name="Rates Shock (TLT -10%)",
        spy_shock_pct=0.0,
        vix_target=None,
        description="שוק ריביות זז חזק – משפיע על duration/rate-sensitive strategies.",
        extra_shocks={"TLT": -0.10},
        assumed_macro_regime="risk_off",
        severity_score=55.0,
        suggested_profile="risk_off",
    ),
    RiskScenario(
        name="FX Shock (DXY +5%)",
        spy_shock_pct=0.0,
        vix_target=None,
        description="זעזוע בדולר – רלוונטי ל-EM / carry / cross-asset.",
        extra_shocks={"DXY": 0.05},
        assumed_macro_regime="neutral",
        severity_score=45.0,
        suggested_profile="neutral",
    ),
]

# ========= Equity, PnL & Drawdown estimation =========

def _compute_equity_and_drawdown(
    hist: pd.DataFrame,
    *,
    equity_col: str = "Equity",
    pnl_col: str = "PnL",
    date_col: Optional[str] = None,
) -> Tuple[pd.Series, Optional[pd.Series], Optional[pd.Series]]:
    """
    מחלץ Equity curve ו-DD מתוך טבלת history.

    לוגיקה:
    - אם יש עמודת Equity → משתמש בה.
    - אחרת, אם יש PnL → מבנה Equity בהינתן Equity התחלתי (או 0).
    - אם יש date_col → ממיין לפיה.
    """
    if hist is None or hist.empty:
        return pd.Series(dtype=float), None, None

    df = hist.copy()

    # אינדקס זמן
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
        idx = df[date_col]
    else:
        idx = None

    # Equity
    if equity_col in df.columns:
        eq = pd.to_numeric(df[equity_col], errors="coerce").dropna()
        if idx is not None and len(eq) == len(idx):
            eq.index = idx
    elif pnl_col in df.columns:
        pnl = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
        eq = pnl.cumsum()
        if idx is not None and len(eq) == len(idx):
            eq.index = idx
    else:
        return pd.Series(dtype=float), None, None

    if eq.empty:
        return pd.Series(dtype=float), None, None

    peak = eq.cummax()
    dd = eq - peak
    dd_pct = dd / peak.replace(0, np.nan)
    return eq, dd, dd_pct


def _multi_horizon_pnl(eq: pd.Series) -> Dict[str, Optional[float]]:
    """
    מחושב daily / weekly / monthly / YTD PnL% על בסיס Equity curve.
    """
    out: Dict[str, Optional[float]] = {
        "daily_pnl_pct": None,
        "weekly_pnl_pct": None,
        "monthly_pnl_pct": None,
        "ytd_pnl_pct": None,
    }
    if eq is None or eq.empty:
        return out

    eq = eq.dropna()
    if len(eq) < 2:
        return out

    # Daily
    try:
        out["daily_pnl_pct"] = float(eq.iloc[-1] / eq.iloc[-2] - 1.0)
    except Exception:
        pass

    # Weekly (~5 ימים אחורה)
    try:
        if len(eq) >= 6:
            out["weekly_pnl_pct"] = float(eq.iloc[-1] / eq.iloc[-6] - 1.0)
    except Exception:
        pass

    # Monthly (~21 ימי מסחר)
    try:
        if len(eq) >= 22:
            out["monthly_pnl_pct"] = float(eq.iloc[-1] / eq.iloc[-22] - 1.0)
    except Exception:
        pass

    # YTD – לפי שנה קלנדרית (אם יש תאריכים)
    try:
        idx = pd.to_datetime(eq.index)
        year_now = idx[-1].year
        ytd_mask = idx.year == year_now
        eq_ytd = eq[ytd_mask]
        if len(eq_ytd) >= 2:
            out["ytd_pnl_pct"] = float(eq_ytd.iloc[-1] / eq_ytd.iloc[0] - 1.0)
    except Exception:
        pass

    return out


def _estimate_realized_vol_from_pnl(
    hist: pd.DataFrame,
    *,
    pnl_col: str = "PnL",
    date_col: Optional[str] = None,
    annualization_factor: int = 252,
) -> Optional[float]:
    """
    סטיית תקן שנתית משוערת מה-PnL (idea: realized_vol_pct).
    """
    if hist is None or hist.empty or pnl_col not in hist.columns:
        return None
    df = hist.copy()
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

    pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna()
    if pnl.empty:
        return None

    # אם יש Equity, אפשר לחשב תשואות; אחרת מניחים PnL על הון נומינלי (פשוט):
    try:
        vol = float(pnl.std(ddof=1))
        if vol == 0:
            return 0.0
        return float(vol * np.sqrt(annualization_factor))
    except Exception:
        return None


# ========= Bucket risk estimation (רעיון 4, 8) =========

def build_bucket_risk_from_history(
    hist: pd.DataFrame,
    *,
    bucket_col: str,
    pnl_col: str = "PnL",
    date_col: Optional[str] = None,
) -> Tuple[List[BucketRisk], JSONDict]:
    """
    בונה BucketRisk לכל bucket (Strategy/Sector/Cluster וכו') מתוך history.
    מחזיר:
    - רשימת BucketRisk
    - summary dict (לשילוב ב-RiskState.bucket_risk_summary)
    """
    if hist is None or hist.empty or bucket_col not in hist.columns or pnl_col not in hist.columns:
        return [], {}

    df = hist.copy()

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)

    df[pnl_col] = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)

    # קבוצות לפי bucket
    groups = df.groupby(bucket_col)
    buckets: List[BucketRisk] = []

    # קודם כל PnL לכל bucket
    total_pnl_per_bucket = groups[pnl_col].sum()
    # סטיית תקן ו-DD לכל bucket
    for name, g in groups:
        pnl_series = g[pnl_col]

        total_pnl = float(pnl_series.sum())
        vol_daily = float(pnl_series.std(ddof=1)) if len(pnl_series) > 1 else 0.0
        vol_annual = vol_daily * np.sqrt(252)

        # DD יחסי – אם יש date_col, מבנה Equity עבור bucket
        try:
            eq_bucket, dd_b, dd_pct_b = _compute_equity_and_drawdown(
                g,
                equity_col="Equity",  # אם קיים
                pnl_col=pnl_col,
                date_col=date_col,
            )
            if dd_pct_b is not None and not dd_pct_b.empty:
                max_dd_pct = float(dd_pct_b.min())
            else:
                max_dd_pct = 0.0
        except Exception:
            max_dd_pct = 0.0

        # Sharpe גס לפי PnL (ללא risk-free)
        sharpe = None
        try:
            if len(pnl_series) > 1:
                mean_pnl = float(pnl_series.mean())
                if vol_daily > 0:
                    sharpe = float((mean_pnl / vol_daily) * np.sqrt(252))
        except Exception:
            sharpe = None

        # worst PnL & DD
        try:
            worst_pnl = float(pnl_series.min())
        except Exception:
            worst_pnl = None

        worst_dd_pct = max_dd_pct

        # placeholders ל-beta / VaR / ES – ניתן לפתח בהמשך
        beta_to_spy = None
        var_95_pct = None
        es_95_pct = None

        bucket = BucketRisk(
            name=str(name),
            total_pnl=total_pnl,
            vol_pct=float(vol_annual),
            max_dd_pct=max_dd_pct,
            contribution_to_risk=0.0,  # נחשב בהמשך
            sharpe=sharpe,
            beta_to_spy=beta_to_spy,
            var_95_pct=var_95_pct,
            es_95_pct=es_95_pct,
            gross_exposure=None,
            net_exposure=None,
            sample_size=int(len(pnl_series)),
            worst_pnl=worst_pnl,
            worst_dd_pct=worst_dd_pct,
        )
        buckets.append(bucket)

    # חישוב contribution_to_risk לפי vol או DD (נבחר vol כבסיס)
    vols = np.array([b.vol_pct for b in buckets], dtype=float)
    if np.any(vols > 0):
        w = vols / vols.sum()
        for b, w_i in zip(buckets, w):
            b.contribution_to_risk = float(w_i)
    else:
        # fallback: לפי abs total_pnl
        total_abs_pnl = np.sum([abs(b.total_pnl) for b in buckets])
        if total_abs_pnl > 0:
            for b in buckets:
                b.contribution_to_risk = float(abs(b.total_pnl) / total_abs_pnl)
        else:
            for b in buckets:
                b.contribution_to_risk = 0.0

    # summary קטן לתוך RiskState.bucket_risk_summary
    summary: JSONDict = {}
    try:
        top = sorted(
            buckets,
            key=lambda x: x.contribution_to_risk,
            reverse=True,
        )
        summary = {
            "total_buckets": len(buckets),
            "top_risk_buckets": [
                {
                    "name": b.name,
                    "contribution_to_risk": b.contribution_to_risk,
                    "vol_pct": b.vol_pct,
                    "max_dd_pct": b.max_dd_pct,
                    "sharpe": b.sharpe,
                }
                for b in top[:10]
            ],
        }
    except Exception:
        summary = {"total_buckets": len(buckets)}

    return buckets, summary


# ========= RiskState estimation (רעיון 3 + tail/stability/liquidity hooks) =========

def estimate_risk_state_from_history(
    hist: pd.DataFrame,
    *,
    equity_col: str = "Equity",
    pnl_col: str = "PnL",
    date_col: Optional[str] = None,
    vix_value: Optional[float] = None,
    macro_regime: Optional[str] = None,
    matrix_crowded_regime: Optional[str] = None,
    anomaly_series: Optional[pd.Series] = None,
    tail_risk_score: Optional[float] = None,
    wf_stability_score: Optional[float] = None,
    liquidity_stress_score: Optional[float] = None,
    bucket_col: Optional[str] = None,
) -> RiskState:
    """
    בונה RiskState מתוך היסטוריית backtest/portfolio:
    - Equity, multi-horizon PnL, DD
    - realized_vol
    - Anomaly / VIX / macro / matrix
    - Bucket risk summary (אופציונלי – אם יש bucket_col)
    """
    equity, dd, dd_pct = _compute_equity_and_drawdown(
        hist,
        equity_col=equity_col,
        pnl_col=pnl_col,
        date_col=date_col,
    )

    equity_today = equity_yday = daily_pnl_abs = daily_pnl_pct = None
    weekly_pnl_pct = monthly_pnl_pct = ytd_pnl_pct = None
    max_dd_pct = None

    if equity is not None and not equity.empty:
        eq_vals = equity.dropna()
        if len(eq_vals) >= 2:
            equity_today = float(eq_vals.iloc[-1])
            equity_yday = float(eq_vals.iloc[-2])
            daily_pnl_abs = equity_today - equity_yday
            if equity_yday != 0:
                daily_pnl_pct = float(equity_today / equity_yday - 1.0)

        mh = _multi_horizon_pnl(eq_vals)
        weekly_pnl_pct = mh["weekly_pnl_pct"]
        monthly_pnl_pct = mh["monthly_pnl_pct"]
        ytd_pnl_pct = mh["ytd_pnl_pct"]

    if dd_pct is not None and not dd_pct.dropna().empty:
        max_dd_pct = float(dd_pct.min())

    realized_vol_pct = _estimate_realized_vol_from_pnl(
        hist,
        pnl_col=pnl_col,
        date_col=date_col,
        annualization_factor=252,
    )

    anomaly_score = None
    anomaly_flag = None
    if anomaly_series is not None and not anomaly_series.empty:
        s = anomaly_series.astype(bool)
        if len(s) > 0:
            anomaly_score = float(s.tail(60).mean())
            anomaly_flag = bool(s.iloc[-1])

    bucket_summary: JSONDict = {}
    if bucket_col is not None and bucket_col in hist.columns:
        _, bucket_summary = build_bucket_risk_from_history(
            hist,
            bucket_col=bucket_col,
            pnl_col=pnl_col,
            date_col=date_col,
        )

    state = RiskState(
        equity_today=equity_today,
        equity_yday=equity_yday,
        daily_pnl_abs=daily_pnl_abs,
        daily_pnl_pct=daily_pnl_pct,
        weekly_pnl_pct=weekly_pnl_pct,
        monthly_pnl_pct=monthly_pnl_pct,
        ytd_pnl_pct=ytd_pnl_pct,
        max_dd_pct=max_dd_pct,
        gross_leverage=None,      # ימולא מבחוץ (מתוך פוזיציות חיות)
        net_leverage=None,
        target_leverage=None,
        realized_vol_pct=realized_vol_pct,
        vix=vix_value,
        macro_regime=macro_regime,
        matrix_crowded_regime=matrix_crowded_regime,
        anomaly_score=anomaly_score,
        anomaly_flag=anomaly_flag,
        tail_risk_score=tail_risk_score,
        wf_stability_score=wf_stability_score,
        liquidity_stress_score=liquidity_stress_score,
        bucket_risk_summary=bucket_summary,
    )
    return state
# ========= Dynamic Risk Limits, Risk Score & Kill-Switch (חלק 3) =========

def get_dynamic_risk_limits(
    base: RiskLimits,
    state: Optional[RiskState] = None,
    *,
    recent_breach_count: int = 0,
) -> RiskLimits:
    """
    מחזיר RiskLimits דינמי לפי:
    - macro_regime (risk_on / risk_off / crisis / neutral)
    - matrix_crowded_regime (Highly Crowded / Crowded / Fragmented / ...)
    - vix
    - recent_breach_count (כמה הפרות חוקים לאחרונה)

    זה מיישם את רעיונות 1+2: Dynamic Limits + Adaptive Thresholds.
    """
    limits = RiskLimits(**asdict(base))

    mr = (state.macro_regime or "").lower() if state and state.macro_regime else ""
    cr = (state.matrix_crowded_regime or "").lower() if state and state.matrix_crowded_regime else ""
    v = float(state.vix) if state and state.vix is not None else None

    # === התאמת פרופיל לפי משטר מאקרו ===
    if "crisis" in mr:
        limits.profile_name = "crisis"
        limits.max_daily_loss_pct = max(limits.min_daily_loss_pct, base.max_daily_loss_pct * 0.5)
        limits.max_weekly_loss_pct = base.max_weekly_loss_pct * 0.7
        limits.max_monthly_loss_pct = base.max_monthly_loss_pct * 0.7
        limits.max_drawdown_pct = base.max_drawdown_pct * 0.7
        limits.max_gross_leverage = max(base.min_gross_leverage, base.max_gross_leverage * 0.5)
        limits.max_net_leverage = max(base.min_gross_leverage, base.max_net_leverage * 0.5)
        limits.tightened_factor = 0.6
    elif "risk_off" in mr or "risk-off" in mr:
        limits.profile_name = "risk_off"
        limits.max_daily_loss_pct = max(limits.min_daily_loss_pct, base.max_daily_loss_pct * 0.7)
        limits.max_weekly_loss_pct = base.max_weekly_loss_pct * 0.8
        limits.max_monthly_loss_pct = base.max_monthly_loss_pct * 0.85
        limits.max_drawdown_pct = base.max_drawdown_pct * 0.8
        limits.max_gross_leverage = max(base.min_gross_leverage, base.max_gross_leverage * 0.7)
        limits.max_net_leverage = max(base.min_gross_leverage, base.max_net_leverage * 0.7)
        limits.tightened_factor = 0.8
    elif "risk_on" in mr:
        limits.profile_name = "risk_on"
        limits.max_daily_loss_pct = base.max_daily_loss_pct * 1.1
        limits.max_weekly_loss_pct = base.max_weekly_loss_pct * 1.05
        limits.max_monthly_loss_pct = base.max_monthly_loss_pct * 1.05
        limits.max_drawdown_pct = base.max_drawdown_pct * 1.0
        limits.max_gross_leverage = base.max_gross_leverage * 1.1
        limits.max_net_leverage = base.max_net_leverage * 1.1
        limits.tightened_factor = 1.0
    else:
        limits.profile_name = "neutral"
        limits.tightened_factor = 1.0

    # === התאמה לפי Crowdedness במטריצה ===
    if "highly" in cr:
        limits.max_gross_leverage = max(base.min_gross_leverage, limits.max_gross_leverage * 0.7)
        limits.max_net_leverage = max(base.min_gross_leverage, limits.max_net_leverage * 0.7)
        limits.max_daily_loss_pct = max(limits.min_daily_loss_pct, limits.max_daily_loss_pct * 0.7)
    elif "crowded" in cr:
        limits.max_gross_leverage = max(base.min_gross_leverage, limits.max_gross_leverage * 0.85)
        limits.max_net_leverage = max(base.min_gross_leverage, limits.max_net_leverage * 0.85)

    # === התאמה לפי VIX ===
    if v is not None:
        if v >= 40:
            limits.max_daily_loss_pct = max(limits.min_daily_loss_pct, limits.max_daily_loss_pct * 0.6)
            limits.max_weekly_loss_pct *= 0.7
            limits.max_monthly_loss_pct *= 0.7
            limits.max_drawdown_pct *= 0.7
            limits.max_gross_leverage = max(base.min_gross_leverage, limits.max_gross_leverage * 0.5)
            limits.max_net_leverage = max(base.min_gross_leverage, limits.max_net_leverage * 0.5)
        elif v >= 25:
            limits.max_daily_loss_pct = max(limits.min_daily_loss_pct, limits.max_daily_loss_pct * 0.8)
            limits.max_weekly_loss_pct *= 0.85
            limits.max_monthly_loss_pct *= 0.9
            limits.max_gross_leverage = max(base.min_gross_leverage, limits.max_gross_leverage * 0.8)
            limits.max_net_leverage = max(base.min_gross_leverage, limits.max_net_leverage * 0.8)

    # === התאמה לפי מספר הפרות חוקים אחרונות ===
    if recent_breach_count > 0:
        factor = max(0.4, 1.0 - 0.1 * recent_breach_count)
        limits.max_daily_loss_pct = max(limits.min_daily_loss_pct, limits.max_daily_loss_pct * factor)
        limits.max_weekly_loss_pct *= factor
        limits.max_monthly_loss_pct *= factor
        limits.max_drawdown_pct *= factor
        limits.max_gross_leverage = max(base.min_gross_leverage, limits.max_gross_leverage * factor)
        limits.max_net_leverage = max(base.min_gross_leverage, limits.max_net_leverage * factor)
        limits.tightened_factor *= factor

    return limits


# ========= Coherent Risk Score (idea 13) =========

def compute_overall_risk_score(
    state: RiskState,
    limits: RiskLimits,
    *,
    bucket_risks: Optional[List[BucketRisk]] = None,
) -> float:
    """
    מחשב ציון סיכון כולל 0–100 (גבוה = מצב בריא יותר).

    לוקח בחשבון:
    - DD מול מגבלה
    - PnL יומי/שבועי/חודשי/שנתי
    - VIX
    - tail_risk_score, wf_stability_score, liquidity_stress_score
    - bucket_risk (ריכוזיות סיכון)
    """
    score = 80.0

    # DD
    if state.max_dd_pct is not None and limits.max_drawdown_pct > 0:
        dd_ratio = abs(state.max_dd_pct) / limits.max_drawdown_pct
        if dd_ratio > 1.0:
            score -= 25.0
        elif dd_ratio > 0.7:
            score -= 15.0
        elif dd_ratio > 0.4:
            score -= 5.0

    # daily PnL
    if state.daily_pnl_pct is not None:
        dp = float(state.daily_pnl_pct)
        if dp < 0:
            score -= min(10.0, abs(dp) * 200.0)

    # weekly / monthly / ytd drift
    for key, weight in [
        ("weekly_pnl_pct", 4.0),
        ("monthly_pnl_pct", 3.0),
        ("ytd_pnl_pct", 2.0),
    ]:
        val = getattr(state, key, None)
        if val is not None:
            try:
                v = float(val)
                if v < 0:
                    score -= min(weight, abs(v) * 100.0)
            except Exception:
                pass

    # VIX
    if state.vix is not None:
        vx = float(state.vix)
        if vx >= 40:
            score -= 20.0
        elif vx >= 30:
            score -= 12.0
        elif vx >= 25:
            score -= 6.0

    # tail risk
    if state.tail_risk_score is not None:
        tr = max(0.0, float(state.tail_risk_score))
        score -= min(20.0, tr)

    # WF stability (גבוה = טוב)
    if state.wf_stability_score is not None:
        stbl = float(state.wf_stability_score)
        if stbl < 0.3:
            score -= 15.0
        elif stbl < 0.6:
            score -= 7.0
        else:
            score += min(5.0, (stbl - 0.6) * 20.0)

    # liquidity stress (גבוה = רע)
    if state.liquidity_stress_score is not None:
        ls = max(0.0, float(state.liquidity_stress_score))
        score -= min(15.0, ls * 20.0)

    # ריכוזיות סיכון ב-buckets (idea 4, 8, 9)
    if bucket_risks:
        contribs = np.array([b.contribution_to_risk for b in bucket_risks], dtype=float)
        if contribs.size > 0:
            # מדד ריכוזיות פשוט – Herfindahl-like
            hhi = float(np.sum(contribs**2))
            if hhi > 0.5:
                score -= 15.0
            elif hhi > 0.3:
                score -= 8.0

    return float(max(0.0, min(100.0, score)))


# ========= Scaling rule (idea 18) =========

def compute_scaling_factor_from_risk(
    risk_score: float,
    *,
    min_scale: float = 0.2,
    max_scale: float = 1.5,
) -> float:
    """
    ממפה risk_score (0–100) ל-scale factor (0.2–1.5 למשל):
    - risk_score גבוה → scale קרוב ל-1.0–1.5 (אפשר להגדיל קצת)
    - risk_score נמוך → scale קרוב ל-0.2–0.5 (להוריד מינוף/חשיפה)
    """
    rs = float(max(0.0, min(100.0, risk_score)))

    # ננרמל ל-[0,1]
    x = rs / 100.0  # 0 = הכי מסוכן, 1 = הכי בטוח

    # map [0,1] to [min_scale, max_scale]
    scale = min_scale + (max_scale - min_scale) * x
    return float(max(min_scale, min(max_scale, scale)))


# ========= Kill Switch evaluation & modes (ideas 2, 10, 11) =========

def evaluate_kill_switch(
    state: RiskState,
    limits: RiskLimits,
    *,
    recent_breach_count: int = 0,
) -> KillSwitchDecision:
    """
    מחליט האם להפעיל Kill-Switch ובאיזה mode:
    - "off": אין פעולה
    - "soft": אין פתיחת פוזיציות חדשות
    - "reduce": הפחתת מינוף/חשיפה
    - "hard": עצירה מלאה

    מחזיר גם severity_score (0–100) + scaling_factor.
    """
    rules: List[str] = []
    actions: List[str] = []
    severity = 0.0

    # dynamic limits לפי state
    dyn_limits = get_dynamic_risk_limits(limits, state, recent_breach_count=recent_breach_count)

    # daily loss rule
    if state.daily_pnl_pct is not None and state.daily_pnl_pct <= -dyn_limits.max_daily_loss_pct:
        rules.append(
            f"daily_loss {state.daily_pnl_pct*100:.2f}% ≤ -{dyn_limits.max_daily_loss_pct*100:.2f}%"
        )
        severity += 25.0
        actions.append("Stop opening new positions today; consider cutting largest losers.")

    # weekly loss rule
    if state.weekly_pnl_pct is not None and state.weekly_pnl_pct <= -dyn_limits.max_weekly_loss_pct:
        rules.append(
            f"weekly_loss {state.weekly_pnl_pct*100:.2f}% ≤ -{dyn_limits.max_weekly_loss_pct*100:.2f}%"
        )
        severity += 15.0
        actions.append("Review weekly performance; reduce risk on weak strategies.")

    # monthly loss rule
    if state.monthly_pnl_pct is not None and state.monthly_pnl_pct <= -dyn_limits.max_monthly_loss_pct:
        rules.append(
            f"monthly_loss {state.monthly_pnl_pct*100:.2f}% ≤ -{dyn_limits.max_monthly_loss_pct*100:.2f}%"
        )
        severity += 15.0
        actions.append("Consider rebalancing or pausing aggressive profiles.")

    # drawdown rule
    if state.max_dd_pct is not None and state.max_dd_pct <= -dyn_limits.max_drawdown_pct:
        rules.append(
            f"max_dd {state.max_dd_pct*100:.2f}% ≤ -{dyn_limits.max_drawdown_pct*100:.2f}%"
        )
        severity += 30.0
        actions.append("Reduce portfolio gross exposure; review worst buckets / pairs.")

    # leverage rules
    if state.gross_leverage is not None and state.gross_leverage > dyn_limits.max_gross_leverage:
        rules.append(
            f"gross_leverage {state.gross_leverage:.2f}x > {dyn_limits.max_gross_leverage:.2f}x"
        )
        severity += 20.0
        actions.append("Close or scale down leveraged positions to get within gross leverage limit.")

    if state.net_leverage is not None and state.net_leverage > dyn_limits.max_net_leverage:
        rules.append(
            f"net_leverage {state.net_leverage:.2f}x > {dyn_limits.max_net_leverage:.2f}x"
        )
        severity += 10.0
        actions.append("Reduce directional exposure to meet net leverage constraints.")

    # VIX rule
    if state.vix is not None and state.vix >= dyn_limits.vix_kill_threshold:
        rules.append(
            f"vix {state.vix:.2f} ≥ {dyn_limits.vix_kill_threshold:.2f}"
        )
        severity += 20.0
        actions.append("High volatility regime – consider Risk-Off profile and tighter sizing.")

    # anomaly rule
    if (
        state.anomaly_flag is True
        and state.anomaly_score is not None
        and state.anomaly_score >= dyn_limits.anomaly_trigger_min_frac
    ):
        rules.append(
            f"anomaly_freq {state.anomaly_score*100:.1f}% ≥ {dyn_limits.anomaly_trigger_min_frac*100:.1f}%"
        )
        severity += 15.0
        actions.append("Frequent anomalies in returns – freeze new risk, investigate anomalies / data.")

    # tail risk / liquidity sensitivity
    if state.tail_risk_score is not None and state.tail_risk_score > 0:
        severity += min(15.0, float(state.tail_risk_score))
    if state.liquidity_stress_score is not None and state.liquidity_stress_score > 0:
        severity += min(10.0, float(state.liquidity_stress_score) * 10.0)

    # adaptive boost לפי כמות הפרות קודמות
    severity += min(20.0, recent_breach_count * 5.0)

    # נורמליזציה ל-0–100
    severity = float(max(0.0, min(100.0, severity)))

    # קביעת mode לפי severity (Graceful Degradation)
    mode = "off"
    enabled = False
    if severity >= 70.0:
        mode = "hard"
        enabled = True
    elif severity >= 45.0:
        mode = "reduce"
        enabled = True
    elif severity >= 25.0:
        mode = "soft"
        enabled = True

    if not enabled:
        reason = "No kill-switch rules triggered."
        actions = ["Continue trading with monitoring."]
    else:
        reason = "; ".join(rules)

    # Overall risk score (לשימוש בשאר המערכת)
    risk_score = compute_overall_risk_score(state, dyn_limits)

    # Scaling factor (idea 18): ככל שהסיכון גבוה יותר → scale נמוך יותר
    scaling_factor = compute_scaling_factor_from_risk(risk_score)

    # זמן עד בדיקה חוזרת (idea 12 – Risk Timeline)
    if mode == "hard":
        recheck = 1800  # 30 דקות
    elif mode == "reduce":
        recheck = 900   # 15 דקות
    elif mode == "soft":
        recheck = 300   # 5 דקות
    else:
        recheck = 600   # 10 דקות ברירת מחדל

    return KillSwitchDecision(
        enabled=enabled,
        mode=mode,
        reason=reason,
        triggered_rules=rules,
        recommended_actions=actions,
        state=state,
        limits=dyn_limits,
        severity_score=severity,
        time_to_recheck_sec=recheck,
        scaling_factor=scaling_factor,
    )

# ========= Scenario / Stress Testing & Advanced Risk Utilities (חלק 4 מורחב) =========

def simulate_equity_scenario(
    hist: pd.DataFrame,
    scenario: RiskScenario,
    *,
    equity_col: str = "Equity",
    pnl_col: str = "PnL",
    date_col: Optional[str] = None,
    portfolio_beta_to_spy: float = 1.0,
    bucket_risks: Optional[List[BucketRisk]] = None,
    limits: Optional[RiskLimits] = None,
) -> Optional[StressResult]:
    """
    סימולציית סטרס גסה על Equity (רעיון 5, 7, 15, 17 + הרחבות):

    - Shock ב-SPY → משפיע על Equity לפי beta (portfolio_beta_to_spy).
    - extra_shocks (TLT, DXY, וכו') יכולים להכנס בהמשך לתמחור מפורט יותר.
    - מחזיר:
        * equity_after
        * approx_pnl_change (abs / pct)
        * approx_dd_effect_pct
        * risk_score_delta (אם limits ניתנו)
        * worst_buckets (לפי contribution_to_risk)
    """
    equity, dd, dd_pct = _compute_equity_and_drawdown(
        hist,
        equity_col=equity_col,
        pnl_col=pnl_col,
        date_col=date_col,
    )
    if equity is None or equity.empty:
        return None

    equity_today = float(equity.dropna().iloc[-1])

    # shock ברמת הקרן = beta * shock SPY
    shock_spy = float(scenario.spy_shock_pct or 0.0)
    effective_shock = shock_spy * float(portfolio_beta_to_spy)

    # TODO: בעתיד – להכניס השפעה של extra_shocks לפי רגישות buckets לנכסים ספציפיים.
    equity_after = equity_today * (1.0 + effective_shock)
    approx_pnl_change = equity_after - equity_today
    approx_pnl_change_pct = approx_pnl_change / equity_today if equity_today != 0 else 0.0

    # השפעת shock על DD (נאיבי – משפרים/מחריפים DD באותה מידה)
    approx_dd_effect_pct = 0.0
    if dd_pct is not None and not dd_pct.dropna().empty:
        current_dd_pct = float(dd_pct.min())
        approx_dd_effect_pct = current_dd_pct + effective_shock

    # worst buckets (אם יש)
    worst_buckets: List[BucketRisk] = []
    if bucket_risks:
        worst_buckets = sorted(
            bucket_risks,
            key=lambda b: b.contribution_to_risk,
            reverse=True,
        )[:5]

    # שינוי בציון סיכון (אם יש limits)
    risk_score_delta = 0.0
    if limits is not None:
        state_before = estimate_risk_state_from_history(
            hist,
            equity_col=equity_col,
            pnl_col=pnl_col,
            date_col=date_col,
        )
        score_before = compute_overall_risk_score(state_before, limits, bucket_risks=bucket_risks)

        state_after = state_before
        state_after.equity_today = equity_after
        state_after.max_dd_pct = (state_before.max_dd_pct or 0.0) + effective_shock
        score_after = compute_overall_risk_score(state_after, limits, bucket_risks=bucket_risks)
        risk_score_delta = float(score_after - score_before)

    return StressResult(
        scenario_name=scenario.name,
        equity_today=equity_today,
        equity_after=equity_after,
        approx_pnl_change=approx_pnl_change,
        approx_pnl_change_pct=approx_pnl_change_pct,
        approx_dd_effect_pct=approx_dd_effect_pct,
        risk_score_delta=risk_score_delta,
        worst_buckets=worst_buckets,
    )


def compute_risk_sensitivity_curve(
    hist: pd.DataFrame,
    limits: RiskLimits,
    *,
    shock_grid: List[float],
    equity_col: str = "Equity",
    pnl_col: str = "PnL",
    date_col: Optional[str] = None,
    portfolio_beta_to_spy: float = 1.0,
    bucket_risks: Optional[List[BucketRisk]] = None,
) -> pd.DataFrame:
    """
    Risk Sensitivity Map (רעיון 17):
    מחזיר טבלה: shock → equity_after, pnl_change, risk_score_after, risk_score_delta.
    """
    rows: List[Dict[str, Any]] = []
    equity, dd, dd_pct = _compute_equity_and_drawdown(
        hist,
        equity_col=equity_col,
        pnl_col=pnl_col,
        date_col=date_col,
    )
    if equity is None or equity.empty:
        return pd.DataFrame()

    equity_today = float(equity.dropna().iloc[-1])

    base_state = estimate_risk_state_from_history(
        hist,
        equity_col=equity_col,
        pnl_col=pnl_col,
        date_col=date_col,
    )
    base_score = compute_overall_risk_score(base_state, limits, bucket_risks=bucket_risks)

    for s in shock_grid:
        eff_shock = float(s) * float(portfolio_beta_to_spy)
        eq_after = equity_today * (1.0 + eff_shock)
        pnl_change = eq_after - equity_today
        pnl_change_pct = pnl_change / equity_today if equity_today != 0 else 0.0

        st_after = base_state
        st_after.equity_today = eq_after
        st_after.max_dd_pct = (base_state.max_dd_pct or 0.0) + eff_shock
        score_after = compute_overall_risk_score(st_after, limits, bucket_risks=bucket_risks)

        rows.append(
            {
                "shock": s,
                "equity_after": eq_after,
                "pnl_change": pnl_change,
                "pnl_change_pct": pnl_change_pct,
                "risk_score_after": score_after,
                "risk_score_delta": score_after - base_score,
            }
        )

    return pd.DataFrame(rows)


# ========= Risk Event Registry (idea 14) =========

_RISK_EVENTS: List[RiskEvent] = []


def register_risk_event(
    event_type: str,
    category: str,
    severity: float,
    message: str,
    *,
    run_id: Optional[str] = None,
    meta: Optional[JSONDict] = None,
) -> RiskEvent:
    """
    רושם RiskEvent ברמת מודול (ללא תלות ב-UI).
    """
    ev = RiskEvent(
        event_type=event_type,
        category=category,
        severity=float(severity),
        message=message,
        run_id=run_id,
        meta=dict(meta or {}),
    )
    _RISK_EVENTS.append(ev)
    if len(_RISK_EVENTS) > 1000:
        del _RISK_EVENTS[: len(_RISK_EVENTS) - 1000]
    return ev


def get_risk_events() -> List[RiskEvent]:
    return list(_RISK_EVENTS)


def get_risk_events_dataframe(limit: Optional[int] = None) -> pd.DataFrame:
    if not _RISK_EVENTS:
        return pd.DataFrame()
    events = _RISK_EVENTS[-limit:] if limit is not None else _RISK_EVENTS
    return pd.DataFrame([e.to_dict() for e in events])


def clear_risk_events() -> None:
    _RISK_EVENTS.clear()


# ========= Risk Budget Monitor (idea 9) =========

def compute_risk_budget_usage(
    buckets: List[BucketRisk],
    limits: RiskLimits,
) -> pd.DataFrame:
    """
    מחשב שימוש ב-Risk Budget לכל bucket:
    - risk_budget: מה הוקצה בקונפיג (0–1)
    - contribution_to_risk: מה קורה בפועל
    - budget_usage: contrib / budget
    """
    if not buckets:
        return pd.DataFrame()

    data: List[Dict[str, Any]] = []
    rb_map = limits.risk_budget_per_bucket or {}

    for b in buckets:
        budget = rb_map.get(b.name)
        usage = None
        if budget is not None and budget > 0:
            usage = float(b.contribution_to_risk / budget)
        data.append(
            {
                "bucket": b.name,
                "risk_budget": budget,
                "contribution_to_risk": b.contribution_to_risk,
                "budget_usage": usage,
                "vol_pct": b.vol_pct,
                "max_dd_pct": b.max_dd_pct,
                "sharpe": b.sharpe,
            }
        )

    df = pd.DataFrame(data)
    return df.sort_values("contribution_to_risk", ascending=False)


# ========= Risk Timeline Analytics (idea 12) =========

def compute_risk_timeline(
    equity: pd.Series,
    *,
    window_days: int = 60,
) -> pd.DataFrame:
    """
    בונה "Risk Timeline" מאקוויטי:
    - rolling_dd_pct
    - rolling_vol_pct (על תשואות)
    """
    if equity is None or equity.empty:
        return pd.DataFrame()

    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if eq.empty:
        return pd.DataFrame()

    rets = eq.pct_change().dropna()
    if rets.empty:
        return pd.DataFrame()

    window = max(5, window_days)

    vol = rets.rolling(window).std(ddof=1) * np.sqrt(252)
    vol.name = "rolling_vol_pct"

    peak = eq.rolling(window).max()
    dd = eq - peak
    dd_pct = dd / peak.replace(0, np.nan)
    dd_pct.name = "rolling_dd_pct"

    df_tl = pd.concat([vol, dd_pct], axis=1).dropna(how="all")
    return df_tl


# ========= רעיון חדש 1 – Scenario Deck Runner =========

def run_scenario_deck(
    hist: pd.DataFrame,
    limits: RiskLimits,
    *,
    equity_col: str = "Equity",
    pnl_col: str = "PnL",
    date_col: Optional[str] = None,
    portfolio_beta_to_spy: float = 1.0,
    bucket_risks: Optional[List[BucketRisk]] = None,
) -> List[StressResult]:
    """
    מריץ את כל SCENARIOS על היסטוריה נתונה ומחזיר רשימת StressResult.
    """
    results: List[StressResult] = []
    for sc in SCENARIOS:
        try:
            res = simulate_equity_scenario(
                hist,
                sc,
                equity_col=equity_col,
                pnl_col=pnl_col,
                date_col=date_col,
                portfolio_beta_to_spy=portfolio_beta_to_spy,
                bucket_risks=bucket_risks,
                limits=limits,
            )
            if res is not None:
                results.append(res)
        except Exception as e:
            logger.warning("Scenario '%s' failed: %s", sc.name, e)
    return results


def scenarios_to_dataframe(results: List[StressResult]) -> pd.DataFrame:
    """
    הופך רשימת StressResult ל-DataFrame נוח.
    """
    if not results:
        return pd.DataFrame()
    rows = []
    for r in results:
        d = r.to_dict()
        wb = d.pop("worst_buckets", [])
        d["worst_buckets_names"] = [b.get("name") for b in wb]
        rows.append(d)
    return pd.DataFrame(rows)


# ========= רעיון חדש 2 – Scenario Ranking =========

def rank_scenarios_by_risk(
    results: List[StressResult],
    *,
    sort_by: str = "risk_score_delta",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    מדרג תרחישים לפי impact (risk_score_delta / DD impact / PnL impact).
    """
    df = scenarios_to_dataframe(results)
    if df.empty:
        return df

    if sort_by not in df.columns:
        # fallback ל-risk_score_delta אם קיים
        if "risk_score_delta" in df.columns:
            sort_by = "risk_score_delta"
        elif "approx_dd_effect_pct" in df.columns:
            sort_by = "approx_dd_effect_pct"
            ascending = True  # DD שלילי יותר = גרוע
        else:
            return df

    return df.sort_values(sort_by, ascending=ascending)


# ========= רעיון חדש 3 – What-if Limits Change (Idea 20) =========

def simulate_limits_change_what_if(
    hist: pd.DataFrame,
    old_limits: RiskLimits,
    new_limits: RiskLimits,
    *,
    equity_col: str = "Equity",
    pnl_col: str = "PnL",
    date_col: Optional[str] = None,
) -> JSONDict:
    """
    בודק: האם ההיסטוריה שהייתה עד עכשיו היתה מפעילה Kill-Switch תחת limits חדשים?

    מחזיר dict עם:
    - would_trigger_old: bool
    - would_trigger_new: bool
    - severity_old / severity_new
    """
    state = estimate_risk_state_from_history(
        hist,
        equity_col=equity_col,
        pnl_col=pnl_col,
        date_col=date_col,
    )
    # כאן אפשר להעביר recent_breach_count=0 לצורך סימולציה
    dec_old = evaluate_kill_switch(state, old_limits, recent_breach_count=0)
    dec_new = evaluate_kill_switch(state, new_limits, recent_breach_count=0)

    return {
        "would_trigger_old": dec_old.enabled,
        "would_trigger_new": dec_new.enabled,
        "severity_old": dec_old.severity_score,
        "severity_new": dec_new.severity_score,
        "mode_old": dec_old.mode,
        "mode_new": dec_new.mode,
    }


# ========= רעיון חדש 4 – Risk Goals Monitor =========

def compute_risk_goals_gap(
    *,
    state: RiskState,
    limits: RiskLimits,
    realized_sharpe: Optional[float] = None,
) -> JSONDict:
    """
    מחשב "gap" בין ביצועי הקרן לבין היעדים המוצהרים ב-RiskLimits.
    """
    goals = {
        "target_vol_pct": limits.target_vol_pct,
        "target_dd_pct": limits.target_dd_pct,
        "target_sharpe_min": limits.target_sharpe_min,
        "target_sharpe_max": limits.target_sharpe_max,
    }

    realized = {
        "realized_vol_pct": state.realized_vol_pct,
        "realized_max_dd_pct": state.max_dd_pct,
        "realized_sharpe": realized_sharpe,
    }

    # gaps
    gaps: JSONDict = {}
    if state.realized_vol_pct is not None:
        gaps["vol_gap"] = float(state.realized_vol_pct - limits.target_vol_pct)
    else:
        gaps["vol_gap"] = None

    if state.max_dd_pct is not None:
        gaps["dd_gap"] = float(abs(state.max_dd_pct) - limits.target_dd_pct)
    else:
        gaps["dd_gap"] = None

    if realized_sharpe is not None:
        rs = float(realized_sharpe)
        gaps["sharpe_gap_min"] = float(rs - limits.target_sharpe_min)
        gaps["sharpe_gap_max"] = float(limits.target_sharpe_max - rs)
    else:
        gaps["sharpe_gap_min"] = None
        gaps["sharpe_gap_max"] = None

    return {
        "goals": goals,
        "realized": realized,
        "gaps": gaps,
    }


# ========= רעיון חדש 5 – Liquidity Stress Profile =========

def compute_liquidity_stress_profile(
    buckets: List[BucketRisk],
    *,
    liquidity_factor: float = 0.5,
) -> JSONDict:
    """
    מעריך רגישות לנזילות: אם liquidity יורד ל-liquidity_factor (למשל 0.5 = 50%),
    כמה "כואב" להקטין פוזיציות.

    מחזיר dict עם:
    - stressed_exposure_per_bucket (יחסי)
    - liquidity_stress_score (0–1+)
    """
    if not buckets:
        return {"stressed_exposure_per_bucket": {}, "liquidity_stress_score": None}

    # נניח שעלות "יציאה" פרופורציונלית ל-contribution_to_risk / liquidity_factor
    stressed: Dict[str, float] = {}
    total_cost = 0.0
    for b in buckets:
        cost = b.contribution_to_risk / max(liquidity_factor, 1e-6)
        stressed[b.name] = float(cost)
        total_cost += cost

    # מנרמלים ל-1
    if total_cost > 0:
        for k in list(stressed.keys()):
            stressed[k] = stressed[k] / total_cost

    # score: ככל שכל המאסה על מספר קטן של buckets → גבוה
    vals = np.array(list(stressed.values()), dtype=float)
    if vals.size > 0:
        hhi = float(np.sum(vals**2))
    else:
        hhi = 0.0

    return {
        "stressed_exposure_per_bucket": stressed,
        "liquidity_stress_score": hhi,  # 0–1 (גבוה = ריכוזיות גבוהה יותר)
    }


# ========= רעיון חדש 6 – Risk Dashboard Summary =========

def build_risk_dashboard_summary(
    state: RiskState,
    limits: RiskLimits,
    decision: Optional[KillSwitchDecision] = None,
    *,
    realized_sharpe: Optional[float] = None,
    bucket_risks: Optional[List[BucketRisk]] = None,
) -> JSONDict:
    """
    בונה summary אחד מרוכז לרמת Dashboard:
    - ציון סיכון כולל
    - מצב Kill-Switch
    - סטטוס יעדים
    - buckets מסוכנים
    """
    risk_score = compute_overall_risk_score(state, limits, bucket_risks=bucket_risks)
    scaling_factor = compute_scaling_factor_from_risk(risk_score)

    goals = compute_risk_goals_gap(
        state=state,
        limits=limits,
        realized_sharpe=realized_sharpe,
    )

    worst_buckets = []
    if bucket_risks:
        worst_buckets = sorted(
            bucket_risks,
            key=lambda b: b.contribution_to_risk,
            reverse=True,
        )[:5]
        worst_buckets = [b.to_dict() for b in worst_buckets]

    kill_info = None
    if decision is not None:
        kill_info = decision.to_dict()

    return {
        "risk_score": risk_score,
        "scaling_factor": scaling_factor,
        "kill_switch": kill_info,
        "goals": goals,
        "worst_buckets": worst_buckets,
        "state": state.to_dict(),
        "limits": limits.to_dict(),
    }
# ========= High-level Risk Assessment Pipeline (Part 5) =========

def run_risk_assessment(
    hist: pd.DataFrame,
    base_limits: RiskLimits,
    *,
    equity_col: str = "Equity",
    pnl_col: str = "PnL",
    date_col: Optional[str] = None,
    vix_value: Optional[float] = None,
    macro_regime: Optional[str] = None,
    matrix_crowded_regime: Optional[str] = None,
    anomaly_series: Optional[pd.Series] = None,
    tail_risk_score: Optional[float] = None,
    wf_stability_score: Optional[float] = None,
    liquidity_stress_score: Optional[float] = None,
    bucket_col: Optional[str] = None,
    portfolio_beta_to_spy: float = 1.0,
    recent_breach_count: int = 0,
    realized_sharpe: Optional[float] = None,
) -> JSONDict:
    """
    Pipeline אחד שמקבל history + base_limits ומחזיר:
      - state: RiskState
      - dynamic_limits: RiskLimits
      - kill_switch: KillSwitchDecision
      - overall_risk_score
      - scaling_factor
      - risk_goals_gap
      - risk_timeline (df קטן)
    """
    # --- Build state ---
    state = estimate_risk_state_from_history(
        hist,
        equity_col=equity_col,
        pnl_col=pnl_col,
        date_col=date_col,
        vix_value=vix_value,
        macro_regime=macro_regime,
        matrix_crowded_regime=matrix_crowded_regime,
        anomaly_series=anomaly_series,
        tail_risk_score=tail_risk_score,
        wf_stability_score=wf_stability_score,
        liquidity_stress_score=liquidity_stress_score,
        bucket_col=bucket_col,
    )

    # Bucket risks (אם ביקשנו)
    bucket_risks: List[BucketRisk] = []
    if bucket_col is not None and bucket_col in hist.columns:
        br, _ = build_bucket_risk_from_history(
            hist,
            bucket_col=bucket_col,
            pnl_col=pnl_col,
            date_col=date_col,
        )
        bucket_risks = br

    # --- Dynamic limits & kill switch ---
    dyn_limits = get_dynamic_risk_limits(base_limits, state, recent_breach_count=recent_breach_count)
    decision = evaluate_kill_switch(
        state=state,
        limits=base_limits,  # משתמש בבסיס, אבל בפנים נשען על dyn_limits
        recent_breach_count=recent_breach_count,
    )

    # --- Risk score & scaling ---
    overall_risk_score = compute_overall_risk_score(state, dyn_limits, bucket_risks=bucket_risks)
    scaling_factor = compute_scaling_factor_from_risk(overall_risk_score)

    # --- Risk goals gap ---
    goals_gap = compute_risk_goals_gap(
        state=state,
        limits=dyn_limits,
        realized_sharpe=realized_sharpe,
    )

    # --- Risk timeline (אם יש Equity) ---
    equity, _, _ = _compute_equity_and_drawdown(
        hist,
        equity_col=equity_col,
        pnl_col=pnl_col,
        date_col=date_col,
    )
    risk_timeline_df = compute_risk_timeline(equity) if equity is not None and not equity.empty else pd.DataFrame()

    # Summary object עבור Dashboard/Insights
    summary = build_risk_dashboard_summary(
        state=state,
        limits=dyn_limits,
        decision=decision,
        realized_sharpe=realized_sharpe,
        bucket_risks=bucket_risks,
    )

    return {
        "state": state,
        "dynamic_limits": dyn_limits,
        "kill_switch": decision,
        "overall_risk_score": overall_risk_score,
        "scaling_factor": scaling_factor,
        "goals_gap": goals_gap,
        "risk_timeline": risk_timeline_df,
        "bucket_risks": bucket_risks,
        "summary": summary,
    }


# ========= RL/Ranking Oriented Utilities =========

def compute_risk_reward_from_assessment(
    assessment: JSONDict,
    *,
    weight_risk_score: float = 2.0,
    weight_sharpe: float = 1.0,
    weight_dd: float = 1.0,
) -> float:
    """
    פונקציית reward לרמת "הערכת סיכון":
    משלב:
      - overall_risk_score (גבוה = טוב)
      - realized_sharpe (אם קיים)
      - max_dd_pct (קנס)
    """
    rs = float(assessment.get("overall_risk_score", 0.0))
    state: RiskState = assessment.get("state")
    realized_sharpe = assessment.get("summary", {}).get("goals", {}).get("realized", {}).get("realized_sharpe")

    reward = 0.0
    # Risk score: ככל שגבוה יותר – reward עולה
    reward += weight_risk_score * (rs / 100.0)

    # Sharpe: מוסיף עוד
    if realized_sharpe is not None:
        try:
            s = float(realized_sharpe)
            reward += weight_sharpe * s
        except Exception:
            pass

    # DD: קנס
    if isinstance(state, RiskState) and state.max_dd_pct is not None:
        try:
            dd = float(state.max_dd_pct)
            if dd < 0:
                reward += -weight_dd * abs(dd)  # dd שלילי → קנס
        except Exception:
            pass

    return float(reward)


def label_risk_profile(
    state: RiskState,
    limits: RiskLimits,
    *,
    overall_risk_score: Optional[float] = None,
) -> str:
    """
    מחזיר label קצר ל"רמת הסיכון" (לשימוש ב-UI/Insights):
    - "Safe"
    - "Caution"
    - "High-Risk"
    - "Critical"
    """
    if overall_risk_score is None:
        overall_risk_score = compute_overall_risk_score(state, limits)

    rs = float(overall_risk_score)
    if rs >= 80:
        return "Safe"
    if rs >= 60:
        return "Caution"
    if rs >= 40:
        return "High-Risk"
    return "Critical"


# ========= Export/Serialization helpers =========

def risk_state_to_dict(state: RiskState) -> JSONDict:
    """ממיר RiskState ל-dict JSON-safe."""
    return state.to_dict()


def risk_limits_to_dict(limits: RiskLimits) -> JSONDict:
    """ממיר RiskLimits ל-dict JSON-safe."""
    return limits.to_dict()


def stress_results_to_dataframe(results: List[StressResult]) -> pd.DataFrame:
    """
    הפיכת רשימת StressResult ל-DataFrame, כולל worst_buckets בשמות בלבד.
    """
    if not results:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for r in results:
        d = r.to_dict()
        wb = d.pop("worst_buckets", [])
        d["worst_buckets_names"] = [b.get("name") for b in wb]
        rows.append(d)
    return pd.DataFrame(rows)


# ========= Bucket Scaling Suggestions (Scaling by Risk, idea 18 extended) =========

def compute_bucket_scaling_suggestions(
    bucket_risks: List[BucketRisk],
    *,
    overall_scaling_factor: float,
    max_bucket_scale: float = 1.2,
    min_bucket_scale: float = 0.3,
) -> pd.DataFrame:
    """
    מספק המלצת scaling לכל bucket לפי:
    - overall_scaling_factor
    - contribution_to_risk (יותר גבוה → scale למטה)
    - Sharpe (יותר גבוה → scale למעלה, עד גבול)
    """
    if not bucket_risks:
        return pd.DataFrame()

    data: List[Dict[str, Any]] = []
    for b in bucket_risks:
        base = overall_scaling_factor

        # penalty לפי contribution_to_risk
        c = float(b.contribution_to_risk or 0.0)
        if c > 0.3:
            base *= 0.7
        elif c > 0.2:
            base *= 0.85

        # בונוס לפי Sharpe (אם חיובי)
        if b.sharpe is not None:
            try:
                s = float(b.sharpe)
                if s > 1.5:
                    base *= 1.1
                elif s < 0.5:
                    base *= 0.9
            except Exception:
                pass

        scale = max(min_bucket_scale, min(max_bucket_scale, base))
        data.append(
            {
                "bucket": b.name,
                "contribution_to_risk": b.contribution_to_risk,
                "sharpe": b.sharpe,
                "suggested_scale": scale,
            }
        )

    df = pd.DataFrame(data)
    return df.sort_values("suggested_scale", ascending=False)


# ========= Risk Assessment Convenience Wrapper =========

def risk_assessment_to_dashboard_dict(
    hist: pd.DataFrame,
    base_limits: RiskLimits,
    *,
    equity_col: str = "Equity",
    pnl_col: str = "PnL",
    date_col: Optional[str] = None,
    vix_value: Optional[float] = None,
    macro_regime: Optional[str] = None,
    matrix_crowded_regime: Optional[str] = None,
    anomaly_series: Optional[pd.Series] = None,
    tail_risk_score: Optional[float] = None,
    wf_stability_score: Optional[float] = None,
    liquidity_stress_score: Optional[float] = None,
    bucket_col: Optional[str] = None,
    portfolio_beta_to_spy: float = 1.0,
    recent_breach_count: int = 0,
    realized_sharpe: Optional[float] = None,
) -> JSONDict:
    """
    עטיפה נוחה ל-UI:
    מריצה run_risk_assessment ומחזירה dict JSON-safe עם כל מה שצריך ל-Dashboard.
    """
    assess = run_risk_assessment(
        hist,
        base_limits,
        equity_col=equity_col,
        pnl_col=pnl_col,
        date_col=date_col,
        vix_value=vix_value,
        macro_regime=macro_regime,
        matrix_crowded_regime=matrix_crowded_regime,
        anomaly_series=anomaly_series,
        tail_risk_score=tail_risk_score,
        wf_stability_score=wf_stability_score,
        liquidity_stress_score=liquidity_stress_score,
        bucket_col=bucket_col,
        portfolio_beta_to_spy=portfolio_beta_to_spy,
        recent_breach_count=recent_breach_count,
        realized_sharpe=realized_sharpe,
    )

    state: RiskState = assess["state"]
    limits: RiskLimits = assess["dynamic_limits"]
    decision: KillSwitchDecision = assess["kill_switch"]
    bucket_risks: List[BucketRisk] = assess["bucket_risks"]
    overall_risk_score: float = assess["overall_risk_score"]
    scaling_factor: float = assess["scaling_factor"]
    summary: JSONDict = assess["summary"]

    label = label_risk_profile(state, limits, overall_risk_score=overall_risk_score)
    reward = compute_risk_reward_from_assessment(assess)

    bucket_scaling_df = compute_bucket_scaling_suggestions(
        bucket_risks=bucket_risks,
        overall_scaling_factor=scaling_factor,
    )

    return {
        "state": state.to_dict(),
        "limits": limits.to_dict(),
        "kill_switch": decision.to_dict(),
        "overall_risk_score": overall_risk_score,
        "risk_label": label,
        "scaling_factor": scaling_factor,
        "reward": reward,
        "goals_gap": assess["goals_gap"],
        "risk_timeline": assess["risk_timeline"],
        "bucket_risks": [b.to_dict() for b in bucket_risks],
        "bucket_scaling": bucket_scaling_df.to_dict(orient="records"),
        "summary": summary,
    }

# ========= Part 6 — AppContext Integration, Breach Stats, Risk Bundles & Checklists =========

try:
    # אינטגרציה עם AppContext (אם זמין)
    from core.app_context import (
        AppContext,
        apply_policies_to_ctx,
        compute_live_readiness,
        ctx_to_markdown as _ctx_to_markdown,
    )
except Exception:  # pragma: no cover
    AppContext = None                # type: ignore
    apply_policies_to_ctx = None     # type: ignore
    compute_live_readiness = None    # type: ignore
    _ctx_to_markdown = None          # type: ignore


# ========= Breach statistics (להזנת recent_breach_count / dashboards) =========

def compute_recent_breach_stats(
    events_df: pd.DataFrame,
    *,
    window_minutes: int = 60,
    now_utc: Optional[datetime] = None,
) -> JSONDict:
    """
    מחשב סטטיסטיקה בסיסית מה-RiskEvent log:
    - breach_count: כמה limit_breach-ים בחלון
    - kill_switch_triggers: כמה Kill-Switch הופעל
    - avg_severity: חומרה ממוצעת
    """
    if events_df is None or events_df.empty:
        return {
            "breach_count": 0,
            "kill_switch_triggers": 0,
            "avg_severity": 0.0,
        }

    df = events_df.copy()
    if "ts_utc" not in df.columns:
        return {
            "breach_count": 0,
            "kill_switch_triggers": 0,
            "avg_severity": 0.0,
        }

    now = now_utc or datetime.now(timezone.utc)()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce")
    df = df.dropna(subset=["ts_utc"])

    cutoff = now - timedelta(minutes=window_minutes)
    recent = df[df["ts_utc"] >= cutoff]
    if recent.empty:
        return {
            "breach_count": 0,
            "kill_switch_triggers": 0,
            "avg_severity": 0.0,
        }

    etype = recent["event_type"].astype(str)

    mask_breach = etype.str.contains("breach", case=False)
    mask_kill = etype.str.contains("kill_switch_trigger", case=False)

    breach_count = int(mask_breach.sum())
    kill_triggers = int(mask_kill.sum())

    if "severity" in recent.columns:
        avg_sev = float(pd.to_numeric(recent["severity"], errors="coerce").dropna().mean())
    else:
        avg_sev = 0.0

    return {
        "breach_count": breach_count,
        "kill_switch_triggers": kill_triggers,
        "avg_severity": avg_sev,
    }


# ========= רעיון חדש 1 — יצירת RiskEvent מתוך KillSwitchDecision =========

def emit_risk_events_from_decision(
    decision: KillSwitchDecision,
    *,
    run_id: Optional[str] = None,
) -> Optional[RiskEvent]:
    """
    יוצר RiskEvent מסוג 'kill_switch_trigger' אם ה-Kill-Switch הופעל.
    """
    if not decision.enabled:
        return None

    msg = f"Kill-Switch mode={decision.mode}, severity={decision.severity_score:.1f}, reason={decision.reason}"
    meta = {
        "mode": decision.mode,
        "severity_score": decision.severity_score,
        "triggered_rules": decision.triggered_rules,
        "recommended_actions": decision.recommended_actions,
    }
    ev = register_risk_event(
        event_type="kill_switch_trigger",
        category="risk",
        severity=decision.severity_score,
        message=msg,
        run_id=run_id,
        meta=meta,
    )
    return ev



# ========= AppContext ↔ RiskEngine Integration =========

def update_app_context_with_risk(
    ctx: Any,
    hist: pd.DataFrame,
    base_limits: RiskLimits,
    *,
    equity_col: str = "Equity",
    pnl_col: str = "PnL",
    date_col: Optional[str] = None,
    vix_value: Optional[float] = None,
    macro_regime: Optional[str] = None,
    matrix_crowded_regime: Optional[str] = None,
    anomaly_series: Optional[pd.Series] = None,
    tail_risk_score: Optional[float] = None,
    wf_stability_score: Optional[float] = None,
    liquidity_stress_score: Optional[float] = None,
    bucket_col: Optional[str] = None,
    realized_sharpe: Optional[float] = None,
    recent_breach_df: Optional[pd.DataFrame] = None,
    prev_live_readiness: Optional[JSONDict] = None,
) -> Tuple[Any, JSONDict]:
    """
    מחבר RiskEngine ל-AppContext:

    - מריץ run_risk_assessment על history.
    - מעדכן ctx דרך apply_policies_to_ctx (risk/data/research).
    - מחשב live_readiness ומוסיף tags מתאימים.
    - מחזיר:
        ctx מעודכן,
        risk_dashboard_dict לשימוש ב-UI / דוחות.
    """
    # fallback אם אין אינטגרציה מלא
    if AppContext is None or apply_policies_to_ctx is None:
        assessment = run_risk_assessment(
            hist,
            base_limits,
            equity_col=equity_col,
            pnl_col=pnl_col,
            date_col=date_col,
            vix_value=vix_value,
            macro_regime=macro_regime,
            matrix_crowded_regime=matrix_crowded_regime,
            anomaly_series=anomaly_series,
            tail_risk_score=tail_risk_score,
            wf_stability_score=wf_stability_score,
            liquidity_stress_score=liquidity_stress_score,
            bucket_col=bucket_col,
            realized_sharpe=realized_sharpe,
        )
        dashboard_dict = risk_assessment_to_dashboard_dict(
            hist,
            base_limits,
            equity_col=equity_col,
            pnl_col=pnl_col,
            date_col=date_col,
            vix_value=vix_value,
            macro_regime=macro_regime,
            matrix_crowded_regime=matrix_crowded_regime,
            anomaly_series=anomaly_series,
            tail_risk_score=tail_risk_score,
            wf_stability_score=wf_stability_score,
            liquidity_stress_score=liquidity_stress_score,
            bucket_col=bucket_col,
            realized_sharpe=realized_sharpe,
        )
        return ctx, dashboard_dict

    # breach stats → recent_breach_count
    recent_breach_count = 0
    if recent_breach_df is not None and not recent_breach_df.empty:
        stats = compute_recent_breach_stats(recent_breach_df)
        recent_breach_count = int(stats.get("breach_count", 0))

    assessment = run_risk_assessment(
        hist,
        base_limits,
        equity_col=equity_col,
        pnl_col=pnl_col,
        date_col=date_col,
        vix_value=vix_value,
        macro_regime=macro_regime,
        matrix_crowded_regime=matrix_crowded_regime,
        anomaly_series=anomaly_series,
        tail_risk_score=tail_risk_score,
        wf_stability_score=wf_stability_score,
        liquidity_stress_score=liquidity_stress_score,
        bucket_col=bucket_col,
        recent_breach_count=recent_breach_count,
        realized_sharpe=realized_sharpe,
    )

    state: RiskState = assessment["state"]
    dyn_limits: RiskLimits = assessment["dynamic_limits"]
    decision: KillSwitchDecision = assessment["kill_switch"]

    # נבנה risk_state dict ל-apply_policies_to_ctx
    risk_state_dict: JSONDict = {
        "daily_pnl_pct": state.daily_pnl_pct,
        "max_dd_pct": state.max_dd_pct,
        "gross_leverage": state.gross_leverage,
        "vix": state.vix,
    }
    risk_limits_dict: JSONDict = {
        "max_daily_loss_pct": dyn_limits.max_daily_loss_pct,
        "max_drawdown_pct": dyn_limits.max_drawdown_pct,
        "max_gross_leverage": dyn_limits.max_gross_leverage,
        "vix_kill_threshold": dyn_limits.vix_kill_threshold,
        "anomaly_trigger_min_frac": dyn_limits.anomaly_trigger_min_frac,
    }

    # data_quality – hook ל-core/data_quality (כרגע None, ניתן להזין מבחוץ)
    data_quality_dict: Optional[JSONDict] = None

    # research stats (num_trades/period/num_pairs)
    num_trades = int(len(hist)) if hist is not None else None
    period_days = None
    if hist is not None and not hist.empty and date_col and date_col in hist.columns:
        try:
            idx = pd.to_datetime(hist[date_col], errors="coerce").dropna()
            if not idx.empty:
                period_days = int((idx.max() - idx.min()).days)
        except Exception:
            period_days = None
    num_pairs = None  # אפשר לשלב מתוך ctx.pairs אם תרצה

    research_stats = {
        "num_trades": num_trades,
        "period_days": period_days,
        "num_pairs": num_pairs,
    }

    # החלת policy engine על ctx
    ctx = apply_policies_to_ctx(
        ctx,
        risk_state=risk_state_dict,
        risk_limits=risk_limits_dict,
        data_quality=data_quality_dict,
        research_stats=research_stats,
    )

    # live_readiness + tagging + RiskEvent על שינוי מצב (רעיון חדש 2)
    live_info: Optional[JSONDict] = None
    if compute_live_readiness is not None:
        live_info = compute_live_readiness(ctx)
        try:
            is_ready = bool(live_info.get("live_ready", False))
            ctx.tags.setdefault("live_ready", "yes" if is_ready else "no")
            ctx.tags["risk_profile_label"] = label_risk_profile(
                state,
                dyn_limits,
                overall_risk_score=assessment["overall_risk_score"],
            )
            # אם היה prev_live_readiness ואנחנו עוברים מ-True ל-False → RiskEvent
            if prev_live_readiness is not None:
                prev_ready = bool(prev_live_readiness.get("live_ready", False))
                if prev_ready and not is_ready:
                    register_risk_event(
                        event_type="live_readiness_drop",
                        category="risk",
                        severity=80.0,
                        message="Live readiness changed from READY to NOT_READY",
                        run_id=ctx.run_id,
                        meta={"prev": prev_live_readiness, "current": live_info},
                    )
        except Exception:
            pass

    dashboard_dict = risk_assessment_to_dashboard_dict(
        hist,
        base_limits,
        equity_col=equity_col,
        pnl_col=pnl_col,
        date_col=date_col,
        vix_value=vix_value,
        macro_regime=macro_regime,
        matrix_crowded_regime=matrix_crowded_regime,
        anomaly_series=anomaly_series,
        tail_risk_score=tail_risk_score,
        wf_stability_score=wf_stability_score,
        liquidity_stress_score=liquidity_stress_score,
        bucket_col=bucket_col,
        realized_sharpe=realized_sharpe,
    )

    if live_info is not None:
        dashboard_dict["live_readiness"] = live_info

    # RiskEvent מתוך החלטת Kill-Switch (רעיון חדש 1)
    try:
        emit_risk_events_from_decision(decision, run_id=ctx.run_id)
    except Exception:
        pass

    return ctx, dashboard_dict


# ========= רעיון חדש 3 — Checklists טקסטואליים (ל-UI / דוחות) =========

def build_risk_checklist_items(
    ctx: Any,
    risk_dashboard: JSONDict,
) -> List[str]:
    """
    יוצר רשימת Bullet Points של "מה לעשות עכשיו":
    - בהתאם ל-Kill-Switch
    - בהתאם ל-risk_score ול-goals_gap
    """
    items: List[str] = []

    try:
        rs = float(risk_dashboard.get("overall_risk_score", 0.0))
        label = risk_dashboard.get("risk_label")
        ks = risk_dashboard.get("kill_switch") or {}
        goals_gap = risk_dashboard.get("goals_gap") or {}
        gaps = goals_gap.get("gaps", {})

        # מצב Kill-Switch
        if ks.get("enabled"):
            mode = ks.get("mode")
            items.append(f"Kill-Switch במצב `{mode}` – להפסיק פתיחת פוזיציות חדשות ולהקטין מינוף בהדרגה.")
        else:
            items.append("Kill-Switch כבוי – אבל יש לבצע בדיקת סיכון יומית רגילה.")

        # Risk score
        if rs < 60:
            items.append("ציון סיכון כללי מתחת ל-60 – לבדוק אסטרטגיות חלשות ו-buckets עם תרומת סיכון גבוהה.")
        elif rs < 80:
            items.append("ציון סיכון באמצע – להימנע מהגדלת מינוף עד לרגיעה במדדים.")

        # DD gap
        dd_gap = gaps.get("dd_gap")
        if dd_gap is not None and isinstance(dd_gap, (int, float)) and dd_gap > 0:
            items.append("DD בפועל גבוה מהיעד – לשקול צמצום exposure באסטרטגיות עם DD קיצוני.")

        # Vol gap
        vol_gap = gaps.get("vol_gap")
        if vol_gap is not None and isinstance(vol_gap, (int, float)):
            if vol_gap > 0.0:
                items.append("Volatility בפועל מעל היעד – להקטין מינוף / חשיפה במודלים תנודתיים.")
            elif vol_gap < -0.05:
                items.append("Volatility בפועל מתחת ליעד – אפשר לשקול הגדלת חשיפה מבוקרת.")

    except Exception:
        pass

    return items


# ========= רעיון חדש 4 — כתיבת Risk Bundle לדיסק =========

def write_risk_bundle_to_disk(
    ctx: Any,
    risk_dashboard: JSONDict,
    *,
    path: Optional[Path] = None,
) -> Optional[Path]:
    """
    כותב bundle קטן (JSON) עם:
    - ctx.short_summary
    - risk_dashboard summary
    - timestamp
    """
    try:
        base_dir = PROJECT_ROOT / "logs" / "risk_bundles"
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        base_dir = PROJECT_ROOT

    if path is None:
        fname = f"risk_bundle_{getattr(ctx, 'run_id', 'unknown')}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        path = base_dir / fname

    try:
        payload: JSONDict = {
            "ts": datetime.now(timezone.utc)().isoformat() + "Z",
            "run_id": getattr(ctx, "run_id", None),
            "ctx_summary": getattr(ctx, "short_summary", lambda: {} )(),
            "risk_dashboard": risk_dashboard,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path
    except Exception:
        return None


# ========= רעיון חדש 5 — טעינת Risk Bundle חזרה לניתוח =========

def load_risk_bundle_from_disk(path: Path) -> Optional[JSONDict]:
    """
    טוען risk bundle (ctx_summary + risk_dashboard) מקובץ JSON.
    """
    if not path.exists():
        return None
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        doc = json.loads(txt)
        return doc if isinstance(doc, dict) else None
    except Exception:
        return None


# ========= Markdown Risk Report (משולב עם AppContext) =========

def build_risk_markdown_report_for_ctx(
    ctx: Any,
    risk_dashboard: JSONDict,
) -> str:
    """
    דוח Markdown משולב:
    - חלק AppContext (דרך ctx_to_markdown אם זמין)
    - חלק Risk Engine: Risk Score, Kill-Switch, Goals, Buckets, Checklist
    """
    parts: List[str] = []

    # --- חלק 1: הקונטקסט עצמו ---
    if _ctx_to_markdown is not None:
        try:
            parts.append(_ctx_to_markdown(ctx, kpis=risk_dashboard.get("summary", {}).get("goals", {})))
        except Exception:
            parts.append(f"# Context Report\n\nRun ID: `{ctx.run_id}`")
    else:
        parts.append(f"# Context Report\n\nRun ID: `{ctx.run_id}`")

    parts.append("\n---\n")
    parts.append("## Risk Engine Summary")

    try:
        rs = risk_dashboard.get("overall_risk_score")
        label = risk_dashboard.get("risk_label")
        scaling = risk_dashboard.get("scaling_factor")
        parts.append(f"- **Overall Risk Score:** `{rs:.1f}`" if rs is not None else "- Overall Risk Score: N/A")
        parts.append(f"- **Risk Label:** `{label}`")
        parts.append(f"- **Suggested Scaling Factor:** `{scaling:.2f}`" if scaling is not None else "- Scaling Factor: N/A")
    except Exception:
        pass

    # Goals gap
    goals_gap = risk_dashboard.get("goals_gap") or {}
    if goals_gap:
        parts.append("\n### Risk Goals vs Reality")
        try:
            goals = goals_gap.get("goals", {})
            realized = goals_gap.get("realized", {})
            gaps = goals_gap.get("gaps", {})
            parts.append("- Target Vol (%): `{}` vs Realized: `{}`".format(
                goals.get("target_vol_pct"), realized.get("realized_vol_pct")
            ))
            parts.append("- Target DD (%): `{}` vs Realized: `{}`".format(
                goals.get("target_dd_pct"), realized.get("realized_max_dd_pct")
            ))
            parts.append("- Target Sharpe Range: `[{}, {}]` vs Realized: `{}`".format(
                goals.get("target_sharpe_min"), goals.get("target_sharpe_max"), realized.get("realized_sharpe")
            ))
            parts.append("- Vol Gap: `{}`".format(gaps.get("vol_gap")))
            parts.append("- DD Gap: `{}`".format(gaps.get("dd_gap")))
        except Exception:
            pass

    # Worst buckets
    wb = risk_dashboard.get("worst_buckets") or []
    if wb:
        parts.append("\n### Top Risk Buckets")
        for b in wb:
            try:
                parts.append(
                    "- `{name}` | contrib={contribution_to_risk:.3f} | vol={vol_pct:.2f} | DD={max_dd_pct:.2f}".format(
                        name=b.get("name"),
                        contribution_to_risk=float(b.get("contribution_to_risk", 0.0)),
                        vol_pct=float(b.get("vol_pct", 0.0)),
                        max_dd_pct=float(b.get("max_dd_pct", 0.0)),
                    )
                )
            except Exception:
                continue

    # Kill-Switch
    ks = risk_dashboard.get("kill_switch") or {}
    if ks:
        parts.append("\n### Kill-Switch")
        try:
            parts.append(f"- Enabled: `{ks.get('enabled')}`")
            parts.append(f"- Mode: `{ks.get('mode')}`")
            parts.append(f"- Severity: `{ks.get('severity_score')}`")
            reason = ks.get("reason")
            if reason:
                parts.append(f"- Reason: `{reason}`")
        except Exception:
            pass

    # Checklist
    checklist = build_risk_checklist_items(ctx, risk_dashboard)
    if checklist:
        parts.append("\n### Action Checklist")
        for item in checklist:
            parts.append(f"- {item}")

    return "\n".join(parts)
