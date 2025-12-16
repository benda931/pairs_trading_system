# -*- coding: utf-8 -*-
"""
core/stat_tests.py — סטטיסטיקות לזיהוי תכונות סדרות זמן
======================================================

מבחני סטציונריות וקו-אינטגרציה ברמת "קרן גידור":

- ADF (Augmented Dickey–Fuller)
- KPSS
- Phillips–Perron
- Cointegration (Engle–Granger)
- Zivot–Andrews (structural break)
- Random-walk simulation סביב ADF
- Hurst exponent (דרך common.utils או fallback פנימי)
- Rolling ADF / KPSS / Hurst
- StatDiagnostics: דיאגנוסטיקה מלאה לסדרה/זוג
- multi_stat_diagnostics: דיאגנוסטיקה לכל ה-DataFrame
- compute_stat_quality_score: ציון איכות סטטיסטי לספראד של זוג
- CLI: הרצת דיאגנוסטיקה על CSV → JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from common.json_safe import make_json_safe, json_default as _json_default

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
if not logger.handlers:
    # לא מגדירים basicConfig בספרייה – משאירים למערכת הראשית
    logger.addHandler(logging.NullHandler())

# ----------------------------------------------------------------------------
# Globals / Settings
# ----------------------------------------------------------------------------

MIN_SERIES_LENGTH: int = int(os.getenv("MIN_SERIES_LENGTH", "20"))


def _check_length(series: pd.Series, name: str = "series") -> bool:
    """
    בדיקת אורך מינימלי לסדרה, כדי לא להריץ מבחנים על מעט מדי תצפיות.
    """
    n = series.dropna().size
    if n < MIN_SERIES_LENGTH:
        logger.warning(
            "Series '%s' too short (%d < %d) – skipping some tests.",
            name,
            n,
            MIN_SERIES_LENGTH,
        )
        return False
    return True


# ----------------------------------------------------------------------------
# יחידות בסיס – מבחנים בודדים
# ----------------------------------------------------------------------------

def adf_full(
    series: pd.Series,
    maxlag: Optional[int] = None,
    regression: str = "c",
    autolag: str = "AIC",
) -> Dict[str, Any]:
    """
    ADF מלא עם החזרת כל הפרטים הדרושים לדיאגנוסטיקה.

    Parameters
    ----------
    series : pd.Series
        סדרת הזמן (רצוי מחירי לוג או ספראד).
    maxlag : int, optional
        מספר לאגים מקסימלי; אם None משתמש בברירת מחדל של statsmodels.
    regression : {'c', 'ct', 'ctt', 'nc'}
        סוג הרגרסיה.
    autolag : {'AIC', 'BIC', 't-stat', None}
        קריטריון בחירת לאג.

    Returns
    -------
    dict
        {
          'test_statistic', 'p_value', 'used_lag', 'nobs',
          'critical_values', 'ic_best', 'regression', 'autolag'
        }
    """
    s = series.dropna()
    if not _check_length(s):
        return {"error": "series too short"}

    try:
        stat, pval, usedlag, nobs, crit, icbest = adfuller(
            s, maxlag=maxlag, regression=regression, autolag=autolag
        )
        return {
            "test_statistic": float(stat),
            "p_value": float(pval),
            "used_lag": int(usedlag),
            "nobs": int(nobs),
            "critical_values": {k: float(v) for k, v in crit.items()},
            "ic_best": float(icbest),
            "regression": regression,
            "autolag": autolag,
        }
    except Exception as e:  # pragma: no cover
        logger.exception("ADF failed: %s", e)
        return {"error": str(e)}


def kpss_test(
    series: pd.Series,
    regression: str = "c",
    nlags: str | int = "auto",
) -> Dict[str, Any]:
    """
    מבחן KPSS – בודק Stationarity עם null של סטציונריות.

    Parameters
    ----------
    series : pd.Series
    regression : {'c', 'ct'}
        'c' – רמת ממוצע קבועה; 'ct' – ממוצע + טרנד.
    nlags : {'auto', int}
        מספר לאגים להפחתת אוטוקורלציה.

    Returns
    -------
    dict
        { 'test_statistic', 'p_value', 'lags', 'critical_values',
          'regression', 'nlags' }
    """
    from statsmodels.tsa.stattools import kpss  # import מקומי

    s = series.dropna()
    if not _check_length(s):
        return {"error": "series too short"}

    try:
        stat, pval, lags, crit = kpss(s, regression=regression, nlags=nlags)
        return {
            "test_statistic": float(stat),
            "p_value": float(pval),
            "lags": int(lags),
            "critical_values": {k: float(v) for k, v in crit.items()},
            "regression": regression,
            "nlags": nlags,
        }
    except Exception as e:  # pragma: no cover
        logger.exception("KPSS failed: %s", e)
        return {"error": str(e)}


def phillips_perron_test(
    series: pd.Series,
    lags: Optional[int] = None,
    trend: str = "ct",
) -> Dict[str, Any]:
    """
    מבחן Phillips–Perron – אלטרנטיבה ל-ADF עם טיפול שונה באוטוקורלציה.

    Parameters
    ----------
    series : pd.Series
    lags : int, optional
        מספר לאגים לשימוש; אם None, statsmodels בוחרת אוטומטית.
    trend : {'c', 'ct'}
        'c' – intercept; 'ct' – intercept + trend.

    Returns
    -------
    dict
        { 'test_statistic', 'p_value', 'critical_values',
          'lags', 'trend' }
    """
    from statsmodels.tsa.stattools import phillips_perron

    s = series.dropna()
    if not _check_length(s):
        return {"error": "series too short"}

    try:
        stat, pval, crit = phillips_perron(s, lags=lags, trend=trend)
        return {
            "test_statistic": float(stat),
            "p_value": float(pval),
            "critical_values": {k: float(v) for k, v in crit.items()},
            "lags": lags,
            "trend": trend,
        }
    except Exception as e:  # pragma: no cover
        logger.exception("Phillips–Perron failed: %s", e)
        return {"error": str(e)}


def cointegration_full(
    series1: pd.Series,
    series2: pd.Series,
    trend: str = "c",
) -> Dict[str, Any]:
    """
    מבחן קו-אינטגרציה (Engle–Granger) מלא.

    Parameters
    ----------
    series1, series2 : pd.Series
        שתי סדרות המחירים (רצוי על אותה אינדקס זמן).
    trend : {'c', 'ct'}
        סוג הטרנד במבחן.

    Returns
    -------
    dict
        { 'test_statistic', 'p_value', 'critical_values', 'trend' }
    """
    s1 = series1.dropna()
    s2 = series2.dropna()
    if not _check_length(s1, "series1") or not _check_length(s2, "series2"):
        return {"error": "series too short"}

    try:
        stat, pval, crit = coint(s1, s2, trend=trend)
        return {
            "test_statistic": float(stat),
            "p_value": float(pval),
            "critical_values": {k: float(v) for k, v in crit.items()},
            "trend": trend,
        }
    except Exception as e:  # pragma: no cover
        logger.exception("Cointegration test failed: %s", e)
        return {"error": str(e)}


def zivot_andrews_test(
    series: pd.Series,
    trim: float = 0.15,
    maxlag: Optional[int] = None,
    regression: str = "c",
) -> Dict[str, Any]:
    """
    מבחן Zivot–Andrews – בודק סטציונריות עם נקודת שבירה אנדוגנית.

    Parameters
    ----------
    series : pd.Series
    trim : float
        כמה מהקצוות לחתוך כדי לחפש נקודת שבירה.
    maxlag : int, optional
    regression : {'c', 'ct', 'ctt'}

    Returns
    -------
    dict
        { 'test_statistic', 'p_value', 'critical_values',
          'breakpoint', 'breakpoint_loc', 'regression' }
    """
    from statsmodels.tsa.stattools import zivot_andrews

    s = series.dropna()
    if not _check_length(s):
        return {"error": "series too short"}

    try:
        stat, pval, crit, bp = zivot_andrews(
            s, trim=trim, maxlag=maxlag, regression=regression
        )
        bp = int(bp)
        bp_loc = s.index[bp] if hasattr(s.index, "__getitem__") else bp
        return {
            "test_statistic": float(stat),
            "p_value": float(pval),
            "critical_values": {k: float(v) for k, v in crit.items()},
            "breakpoint": bp,
            "breakpoint_loc": bp_loc,
            "regression": regression,
        }
    except Exception as e:  # pragma: no cover
        logger.exception("Zivot–Andrews failed: %s", e)
        return {"error": str(e)}


def random_walk_test(
    series: pd.Series,
    n_sims: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    store_sim_stats: bool = False,
) -> Dict[str, Any]:
    """
    בדיקת "כמה ADF של הסדרה קיצוני לעומת Random Walk".

    הרעיון: מחשבים ADF לסדרה האמיתית, מדמים n_sims מסלולי Random Walk
    באותו אורך, ומסתכלים על החלק היחסי של סימולציות שבהן סטטיסטיקת ה-ADF
    יותר שלילית (stationary יותר) מהמקור.

    Returns
    -------
    dict
        {
          'base_stat', 'sim_fraction', 'alpha',
          'reject_random_walk', 'n_sims',
          'simulated_stats' (אופציונלי)
        }
    """
    s = series.dropna()
    if not _check_length(s):
        return {"error": "series too short"}

    if seed is not None:
        np.random.seed(seed)

    try:
        base_stat = float(adfuller(s)[0])
    except Exception as e:  # pragma: no cover
        logger.exception("ADF in random_walk_test failed: %s", e)
        return {"error": f"ADF failed: {e}"}

    n = s.size
    sims = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        rw = pd.Series(np.random.randn(n)).cumsum()
        sims[i] = adfuller(rw)[0]

    # סטטיסטיקה שלילית יותר → stationarity חזקה יותר
    frac = float(np.mean(sims < base_stat))
    reject = frac < alpha

    out: Dict[str, Any] = {
        "base_stat": base_stat,
        "sim_fraction": frac,
        "alpha": float(alpha),
        "reject_random_walk": bool(reject),
        "n_sims": int(n_sims),
    }
    if store_sim_stats:
        out["simulated_stats"] = sims
    return out


def plot_breakpoint(series: pd.Series, breakpoint: int) -> None:
    """
    ציור נקודת שבירה לפי תוצאת Zivot–Andrews.
    """
    import matplotlib.pyplot as plt  # import מקומי

    plt.figure(figsize=(10, 4))
    plt.plot(series, label="Series")
    plt.axvline(x=breakpoint, color="red", linestyle="--", label="Break Point")
    plt.title("Zivot–Andrews Breakpoint")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_random_walk_simulation(
    series: pd.Series,
    n_sims: int = 500,
    bins: int = 30,
) -> None:
    """
    המחשת התפלגות סטטיסטיקת ADF של Random Walk לעומת הסדרה המקורית.
    """
    import matplotlib.pyplot as plt

    s = series.dropna()
    if not _check_length(s):
        logger.warning("Series too short for plot_random_walk_simulation")
        return

    try:
        base_stat = float(adfuller(s)[0])
    except Exception as e:  # pragma: no cover
        logger.exception("ADF failed in plot_random_walk_simulation: %s", e)
        return

    n = s.size
    sims = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        rw = pd.Series(np.random.randn(n)).cumsum()
        sims[i] = adfuller(rw)[0]

    plt.figure(figsize=(8, 5))
    plt.hist(sims, bins=bins, edgecolor="black")
    plt.axvline(
        base_stat,
        color="red",
        linestyle="--",
        label=f"Original Series Stat: {base_stat:.2f}",
    )
    plt.title("Random Walk ADF Statistic Distribution")
    plt.xlabel("ADF Statistic")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Helper: קו-אינטגרציה בינארית
# ----------------------------------------------------------------------------

def is_cointegrated(
    series1: pd.Series,
    series2: pd.Series,
    threshold: float = 0.05,
) -> bool:
    """
    החזרת True/False לפי p-value של מבחן קו-אינטגרציה.
    """
    result = cointegration_full(series1, series2)
    return bool(result.get("p_value", 1.0) < threshold)


# ----------------------------------------------------------------------------
# Hurst helper
# ----------------------------------------------------------------------------

def _compute_hurst(series: pd.Series) -> Optional[float]:
    """
    ניסיון להשתמש ב-hurst_exponent מהמודול המשותף.
    אם אינו זמין – fallback לאומדן פשוט.
    """
    s = series.dropna()
    if not _check_length(s):
        return None

    # קודם לנסות את המימוש מהמערכת שלך
    try:
        from common.utils import hurst_exponent  # type: ignore
        return float(hurst_exponent(s))
    except Exception:
        pass

    # Fallback פשוט (R/S)
    try:
        ts = s.values.astype(float)
        n = ts.size
        if n < MIN_SERIES_LENGTH:
            return None
        mean_ts = ts.mean()
        dev = ts - mean_ts
        cum_dev = np.cumsum(dev)
        r = np.max(cum_dev) - np.min(cum_dev)
        s_std = np.std(ts)
        if s_std == 0:
            return None
        rs = r / s_std
        return float(np.log(rs) / np.log(n))
    except Exception:
        return None


# ----------------------------------------------------------------------------
# High-level diagnostics API
# ----------------------------------------------------------------------------

class StatDiagnostics:
    """
    מחלקה מרכזית לדיאגנוסטיקה של סדרת זמן אחת (ואופציונלית שנייה לקו-אינטגרציה).

    שימוש בסיסי
    -----------
        diag = StatDiagnostics(spread, price_x)
        report = diag.run_all()
        df = diag.to_dataframe()
    """

    def __init__(
        self,
        series: pd.Series,
        series2: Optional[pd.Series] = None,
        name: Optional[str] = None,
        min_length: Optional[int] = None,
    ) -> None:
        self.series = series
        self.series2 = series2
        self.name = name or getattr(series, "name", "series")
        self.min_length = min_length or MIN_SERIES_LENGTH

    def _ok(self) -> bool:
        return _check_length(self.series, self.name)

    def run_all(self, tests: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """
        מריץ את כל המבחנים המרכזיים ומחזיר dict מסודר.

        Parameters
        ----------
        tests : sequence[str], optional
            אם הועבר – יריץ רק את המבחנים ברשימה
            (למשל: ['adf', 'kpss', 'pp']).

        Returns
        -------
        dict
            {
              'meta': {...},
              'adf': {...} | error,
              'kpss_level': {...},
              'kpss_trend': {...},
              'pp': {...},
              'zivot_andrews': {...},
              'random_walk': {...},
              'hurst': float | None,
              'coint': {...} | error (אם series2 קיים),
              'is_coint': bool | None
            }
        """
        s = self.series
        meta = {
            "name": self.name,
            "length": int(s.dropna().size),
            "min_length_required": int(self.min_length),
            "n_nan": int(s.isna().sum()),
        }

        def _want(key: str) -> bool:
            return tests is None or key in tests

        out: Dict[str, Any] = {"meta": meta}

        if _want("adf"):
            out["adf"] = adf_full(s)
        if _want("kpss"):
            out["kpss_level"] = kpss_test(s, regression="c")
            out["kpss_trend"] = kpss_test(s, regression="ct")
        if _want("pp"):
            out["pp"] = phillips_perron_test(s)
        if _want("zivot_andrews"):
            out["zivot_andrews"] = zivot_andrews_test(s)
        if _want("random_walk"):
            out["random_walk"] = random_walk_test(s, n_sims=500, alpha=0.05)
        if _want("hurst"):
            out["hurst"] = _compute_hurst(s)

        if self.series2 is not None and _want("coint"):
            out["coint"] = cointegration_full(self.series, self.series2)
            out["is_coint"] = is_cointegrated(self.series, self.series2)
        else:
            out["coint"] = None
            out["is_coint"] = None

        return out

    def summary(self) -> str:
        """
        סיכום טקסטואלי קצר – מתאים ללוגים / debug.
        """
        r = self.run_all()
        lines = [f"Diagnostics for {self.name}"]

        adf_res = r.get("adf", {})
        if isinstance(adf_res, dict) and "p_value" in adf_res:
            if adf_res["p_value"] < 0.05:
                lines.append("• ADF: series appears STATIONARY (p < 0.05).")
            else:
                lines.append("• ADF: series NON-stationary (p ≥ 0.05).")
        else:
            lines.append("• ADF: not available.")

        hurst_val = r.get("hurst")
        if isinstance(hurst_val, (float, int)):
            lines.append(f"• Hurst exponent: {hurst_val:.3f}.")
        else:
            lines.append("• Hurst exponent: N/A.")

        kpss_lvl = r.get("kpss_level", {})
        kpss_tr = r.get("kpss_trend", {})
        lvl_p = kpss_lvl.get("p_value")
        tr_p = kpss_tr.get("p_value")
        if isinstance(lvl_p, (float, int)):
            if lvl_p < 0.05:
                lines.append("• KPSS (level): rejects stationarity (p < 0.05).")
            else:
                lines.append("• KPSS (level): does NOT reject stationarity.")
        if isinstance(tr_p, (float, int)):
            if tr_p < 0.05:
                lines.append("• KPSS (trend): rejects trend-stationarity (p < 0.05).")
            else:
                lines.append("• KPSS (trend): does NOT reject trend-stationarity.")

        if self.series2 is not None:
            is_c = r.get("is_coint")
            if is_c is True:
                lines.append("• Cointegration: series ARE cointegrated (p < 0.05).")
            elif is_c is False:
                lines.append("• Cointegration: NO cointegration detected.")
            else:
                lines.append("• Cointegration: N/A.")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """
        המרה של תוצאות run_all ל-DataFrame חתוך:
        עמודות: [series, test, metric, value]
        """
        report = self.run_all()
        rows: List[Dict[str, Any]] = []

        series_name = self.name
        for test_name, result in report.items():
            if test_name == "meta":
                for k, v in result.items():
                    rows.append(
                        {
                            "series": series_name,
                            "test": "meta",
                            "metric": k,
                            "value": v,
                        }
                    )
                continue

            if isinstance(result, dict):
                for key, value in result.items():
                    rows.append(
                        {
                            "series": series_name,
                            "test": test_name,
                            "metric": key,
                            "value": value,
                        }
                    )
            else:
                rows.append(
                    {
                        "series": series_name,
                        "test": test_name,
                        "metric": "value",
                        "value": result,
                    }
                )

        return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Batch diagnostics על DataFrame שלם
# ----------------------------------------------------------------------------

def multi_stat_diagnostics(
    df: pd.DataFrame,
    df2: Optional[pd.DataFrame] = None,
    tests: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    מריץ StatDiagnostics לכל עמודה ב-df, ובמידה ו-df2 סופק – באותה עמודה
    משתמש בה כ-series2 לצורך קו-אינטגרציה.

    Parameters
    ----------
    df : pd.DataFrame
        נתוני הבסיס (למשל ספראד/מחירים של זוגות).
    df2 : pd.DataFrame, optional
        DataFrame שני באותה צורה (לא חובה).
    tests : sequence[str], optional
        רשימת מבחנים להכללה (כמו ב-StatDiagnostics.run_all).

    Returns
    -------
    pd.DataFrame
        טבלת דיאגנוסטיקה מאוחדת לכל העמודות.
    """
    combined: List[pd.DataFrame] = []

    for col in df.columns:
        s1 = df[col]
        s2 = df2[col] if df2 is not None and col in df2.columns else None
        diag = StatDiagnostics(s1, s2, name=str(col))
        # tests כרגע לא מפלטר ברמת DF – אפשר להרחיב אם תרצה
        df_out = diag.to_dataframe()
        combined.append(df_out)

    if not combined:
        return pd.DataFrame(columns=["series", "test", "metric", "value"])

    return pd.concat(combined, ignore_index=True)


# ----------------------------------------------------------------------------
# Rolling diagnostics — איך הסטציונריות משתנה לאורך הזמן
# ----------------------------------------------------------------------------

def _iter_rolling_windows(n: int, window: int, step: int) -> List[Tuple[int, int]]:
    """
    מחזיר זוגות (start, end) לחלונות רולינג.
    end הוא אינדקס *סגור* (כלומר כולל).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if n < window:
        return []

    out: List[Tuple[int, int]] = []
    for start in range(0, n - window + 1, step):
        end = start + window - 1
        out.append((start, end))
    return out


def rolling_adf(
    series: pd.Series,
    window: int,
    step: int = 1,
    regression: str = "c",
    autolag: str = "AIC",
) -> pd.DataFrame:
    """
    ADF rolling לאורך הסדרה.

    Parameters
    ----------
    series : pd.Series
    window : int
        גודל חלון (מספר תצפיות).
    step : int
        כמה להתקדם בכל פעם.
    regression, autolag :
        עוברים ישירות ל-adf_full.

    Returns
    -------
    pd.DataFrame
        עמודות:
            - 'end_index' (לייבל מהאינדקס של הסדרה)
            - 'end_pos' (מיקום אינדקס אינטגרלי)
            - 'p_value'
            - 'test_statistic'
            - 'used_lag'
    """
    s = series.dropna()
    n = len(s)
    if n < window or window < MIN_SERIES_LENGTH:
        logger.warning(
            "rolling_adf: window (%d) or series length (%d) too small (min=%d)",
            window,
            n,
            MIN_SERIES_LENGTH,
        )
        return pd.DataFrame(
            columns=["end_index", "end_pos", "p_value", "test_statistic", "used_lag"]
        )

    rows: List[Dict[str, Any]] = []
    idx = s.index
    for start, end in _iter_rolling_windows(n, window, step):
        seg = s.iloc[start : end + 1]
        res = adf_full(seg, regression=regression, autolag=autolag)
        if "error" in res:
            pval = np.nan
            stat = np.nan
            used_lag = np.nan
        else:
            pval = res.get("p_value", np.nan)
            stat = res.get("test_statistic", np.nan)
            used_lag = res.get("used_lag", np.nan)

        rows.append(
            {
                "end_index": idx[end],
                "end_pos": end,
                "p_value": pval,
                "test_statistic": stat,
                "used_lag": used_lag,
            }
        )

    return pd.DataFrame(rows)


def rolling_kpss(
    series: pd.Series,
    window: int,
    step: int = 1,
    nlags: str | int = "auto",
) -> pd.DataFrame:
    """
    KPSS rolling לאורך הסדרה (גם level וגם trend).

    Returns
    -------
    pd.DataFrame
        עמודות:
            - 'end_index'
            - 'end_pos'
            - 'p_value_level'
            - 'p_value_trend'
    """
    s = series.dropna()
    n = len(s)
    if n < window or window < MIN_SERIES_LENGTH:
        logger.warning(
            "rolling_kpss: window (%d) or series length (%d) too small (min=%d)",
            window,
            n,
            MIN_SERIES_LENGTH,
        )
        return pd.DataFrame(
            columns=["end_index", "end_pos", "p_value_level", "p_value_trend"]
        )

    rows: List[Dict[str, Any]] = []
    idx = s.index
    for start, end in _iter_rolling_windows(n, window, step):
        seg = s.iloc[start : end + 1]
        lvl = kpss_test(seg, regression="c", nlags=nlags)
        trd = kpss_test(seg, regression="ct", nlags=nlags)

        p_lvl = lvl.get("p_value", np.nan) if isinstance(lvl, dict) else np.nan
        p_trd = trd.get("p_value", np.nan) if isinstance(trd, dict) else np.nan

        rows.append(
            {
                "end_index": idx[end],
                "end_pos": end,
                "p_value_level": p_lvl,
                "p_value_trend": p_trd,
            }
        )

    return pd.DataFrame(rows)


def rolling_hurst(
    series: pd.Series,
    window: int,
    step: int = 1,
) -> pd.DataFrame:
    """
    Hurst rolling לאורך הסדרה – לראות אם התהליך נע יותר לכיוון
    Mean-Reversion / Trending לאורך זמן.

    Returns
    -------
    pd.DataFrame
        עמודות:
            - 'end_index'
            - 'end_pos'
            - 'hurst'
    """
    s = series.dropna()
    n = len(s)
    if n < window or window < MIN_SERIES_LENGTH:
        logger.warning(
            "rolling_hurst: window (%d) or series length (%d) too small (min=%d)",
            window,
            n,
            MIN_SERIES_LENGTH,
        )
        return pd.DataFrame(columns=["end_index", "end_pos", "hurst"])

    rows: List[Dict[str, Any]] = []
    idx = s.index
    for start, end in _iter_rolling_windows(n, window, step):
        seg = s.iloc[start : end + 1]
        h = _compute_hurst(seg)
        rows.append(
            {
                "end_index": idx[end],
                "end_pos": end,
                "hurst": h,
            }
        )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# High-level convenience: ציון איכות סטטיסטי לזוג
# ----------------------------------------------------------------------------

def compute_stat_quality_score(
    spread: pd.Series,
    secondary: Optional[pd.Series] = None,
) -> float:
    """
    מחשב ציון איכות סטטיסטי לספראד של זוג, ברמת "קרן גידור".

    הרעיון:
    - ADF p-value נמוך → טוב (סטציונריות)
    - KPSS level p-value גבוה → טוב (לא דוחה סטציונריות)
    - Hurst בסביבות 0.3–0.7 → טוב (Mean-Reverting סביר, לא רעש ולא טרנד חזק מדי)
    - Cointegration (אם secondary סופק) → בונוס

    הציון לא "מדעי מושלם", אלא scale נוח בערך 0–4,
    שאפשר להכניס לנוסחת הדירוג של הזוג.
    """
    s = spread.dropna()
    if not _check_length(s, "spread"):
        return 0.0

    diag = StatDiagnostics(s, series2=secondary, name="spread")
    r = diag.run_all(tests=["adf", "kpss", "hurst", "coint"])

    # --- ADF ---
    adf_p = 1.0
    adf_res = r.get("adf", {})
    if isinstance(adf_res, dict):
        adf_p = float(adf_res.get("p_value", 1.0) or 1.0)
    adf_p = min(max(adf_p, 0.0), 1.0)
    # ככל שה-p נמוך יותר → יותר נקודות (עד ~1.5)
    adf_score = max(0.0, 1.5 - adf_p * 3.0)

    # --- KPSS (level) ---
    kpss_level_p = 0.0
    kpss_level = r.get("kpss_level", {})
    if isinstance(kpss_level, dict):
        kpss_level_p = float(kpss_level.get("p_value", 0.0) or 0.0)
    kpss_level_p = min(max(kpss_level_p, 0.0), 1.0)
    # p גבוה → יותר נקודות (עד 1.0)
    kpss_score = kpss_level_p * 1.0

    # --- Hurst ---
    hurst_val = r.get("hurst", None)
    hurst_score = 0.0
    if isinstance(hurst_val, (float, int)):
        h = float(hurst_val)
        # הכי טוב סביב 0.5; נותן ניקוד לפי מרחק מ-0.5
        # אם Hurst בין 0.3 ל-0.7 → נקודות חזקות
        dist = min(abs(h - 0.5), 0.5)  # מרחק מקסימלי 0.5
        hurst_score = max(0.0, 1.5 - dist * 3.0)  # בערך 0–1.5

    # --- Cointegration (אם יש) ---
    coint_bonus = 0.0
    is_c = r.get("is_coint", None)
    if isinstance(is_c, bool) and is_c:
        coint_bonus = 0.5

    total = adf_score + kpss_score + hurst_score + coint_bonus
    return float(total)


# ----------------------------------------------------------------------------
# CLI / API Interface
# ----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    CLI להרצת דיאגנוסטיקה על קובץ CSV אחד (ואופציונלית שני) והוצאת JSON.
    """
    parser = argparse.ArgumentParser(
        description="Run statistical diagnostics on time series data (pairs-trading grade)."
    )
    parser.add_argument("file1", help="Path to CSV file for primary series")
    parser.add_argument(
        "--file2",
        help="Optional CSV file for secondary series (same columns)",
        default=None,
    )
    parser.add_argument(
        "--output",
        help="Output JSON file for results (if not given, prints to stdout)",
        default=None,
    )
    args = parser.parse_args(argv)

    df1 = pd.read_csv(args.file1, index_col=0, parse_dates=True)
    df2 = None
    if args.file2:
        df2 = pd.read_csv(args.file2, index_col=0, parse_dates=True)

    results: Dict[str, Any] = {}
    for col in df1.columns:
        s1 = df1[col]
        s2 = df2[col] if df2 is not None and col in df2.columns else None
        diag = StatDiagnostics(s1, s2, name=str(col))
        results[col] = diag.run_all()

    safe = make_json_safe(results)
    output_str = json.dumps(
        safe,
        default=_json_default,
        ensure_ascii=False,
        indent=2,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str)
        print(f"Results written to {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":  # pragma: no cover
    main()
