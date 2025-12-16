# -*- coding: utf-8 -*-
"""
core/macro_engine.py — Macro profile & overlay backtest engine (HF-grade)
========================================================================

תפקיד הקובץ
-----------
שכבת Engine בין טאב המאקרו / ה-root לבין שכבת החישוב המאקרו (common.macro_adjustments)
ושכבת ניהול הסיכונים (common.risk_Helpers).

הקובץ נותן שני APIs עיקריים (ללא תלות ב-Streamlit):

Public API
----------
1. compute_profile(pairs_df: pd.DataFrame, cfg: dict) -> MacroProfile
   - בונה פרופיל מאקרו לכל pair_id ביקום:
     * macro_fit_score
     * weight_adj (מכפיל חשיפה)
     * action (Neutral/Strengthen/Weaken/Pause)
     * include (True/False)
     * cap_applied / cap_value
   - מחזיר גם:
     * regimes_prob: one-hot/soft allocation ל־regimes (risk_on/off, reflation וכו').
     * features_summary: סיכום פיצ'רים מאקרו (אם זמין core.macro_features).
     * meta: מידע עזר (פרמטרים, timestamps, feature_params וכו').

2. backtest(cfg: dict) -> BacktestResult
   - מריץ backtest overlay מאקרו־+־Risk engine על DataFrame תשואות pairs:
     * cfg["returns_df"]: DataFrame, index=date, columns=pair_id, values=returns.
     * cfg["pairs_df"]: DataFrame עם עמודת pair_id (אופציונלי, משמש caps וכו').
   - השכבה הזאת:
     * מחשבת MacroProfile (overlay multipliers + filters).
     * מחשבת משקולות בסיס (invvol / risk-parity / equal).
     * מיישמת multipliers, group caps, normalize.
     * מיישמת DrawdownGate (kill-switch) אם זמין.
     * מחזירה מדדי ביצוע + אובייקטים (equity, weights וכו').

קונפיגים עיקריים ב-cfg (לא חובה, עם defaults סבירים)
---------------------------------------------------
- seed: int (ברירת מחדל 42).
- risk_sizing: str ("invvol" / "erc" / "equal" ... בהתאם ל-risk_Helpers).
- target_vol / volatility_target: float (למשל 0.15).
- cov_window: int (ברירת מחדל 60).
- cov_shrink: float (ברירת מחדל 0.1).
- max_pair_weight: float (ברירת מחדל 0.1).
- enable_macro_overlay: bool (ברירת מחדל True).
- kill_switch_drawdown: float (threshold ל-DrawdownGate, למשל 0.2 ל-20%).
- kill_cooloff: int (מספר ימים "קירור" אחרי טריגר gate).
- risk_free: float (שיעור שנתי ל-Sharpe/Sortino, ברירת מחדל 0.0).

Caps לפי קבוצות (לא חובה, אם קיימת עמודה מתאימה ב-pairs_df)
------------------------------------------------------------
- sector_caps: dict[str, float]
- region_caps: dict[str, float]
- currency_caps: dict[str, float]

פונקציות עזר בקובץ
-------------------
- parse_returns_df(...)     – קריאת קובץ תשואות CSV/Parquet ל-DataFrame תקני.
- parse_pairs_df(...)       – קריאת קובץ pairs ויצירת pair_id אם חסר.
- run_backtest_with_files(...) – wrapper נוח ל-files -> backtest(cfg).

הקובץ Designed להיות ללא תלות ב-Streamlit וניתן לשימוש גם ב-CLI/Unit Tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Union
import logging

import numpy as np
import pandas as pd

# =============================================================================
# Imports: risk sizing & metrics (helpers)
# =============================================================================
try:
    from common.risk_Helpers import (
        target_weights_from_returns,
        apply_multipliers,
        portfolio_metrics,
        apply_caps,
        normalize_weights,
    )
    try:
        from common.risk_Helpers import DrawdownGate
    except Exception:  # DrawdownGate לא חובה
        DrawdownGate = None  # type: ignore[attr-defined]
except Exception:
    # מאפשר למנוע לקרוס גם אם risk_Helpers לא קיים (למשל בזמן פיתוח)
    target_weights_from_returns = None  # type: ignore[assignment]
    apply_multipliers = None  # type: ignore[assignment]
    portfolio_metrics = None  # type: ignore[assignment]
    apply_caps = None  # type: ignore[assignment]
    normalize_weights = None  # type: ignore[assignment]
    DrawdownGate = None  # type: ignore[assignment]

# =============================================================================
# Logger
# =============================================================================
LOGGER = logging.getLogger("core.macro_engine")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


# =============================================================================
# Data structures (public)
# =============================================================================
@dataclass
class MacroProfile:
    """
    אובייקט פרופיל מאקרו ברמת universe.

    Attributes
    ----------
    profile:
        dict[pair_id, dict] עם מפתחות עיקריים:
            - macro_fit_score: float | None
            - weight_adj: float (מכפיל משקל בסיסי)
            - action: str ("Neutral", "Strengthen", "Weaken", "Pause")
            - reason: str
            - include: bool
            - cap_applied: bool
            - cap_value: float | None

    regimes_prob:
        DataFrame עם one-hot (או soft) ל-regimes:
        עמודות: ["risk_on","risk_off","stagflation","reflation","slowdown","neutral"]

    features_summary:
        DataFrame מסוכם של פיצ'רי מאקרו (אם זמין core.macro_features).

    meta:
        dict עם פרטי עזר (פרמטרים, timestamps, feature_params וכו').
    """

    profile: Dict[str, Dict[str, Any]]
    regimes_prob: Optional[pd.DataFrame] = None
    features_summary: Optional[pd.DataFrame] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class BacktestResult:
    """
    תוצאת Backtest של שכבת המאקרו+Risk Engine.

    המדדים מחושבים על סדרת התשואות של הפורטפוליו אחרי overlay + gates.

    Attributes (מדדים עיקריים)
    --------------------------
    sharpe: float
    sortino: float
    cagr: float
    max_dd: float
    uplift_vs_base: float
        Annualized excess return מול baseline equal-weight.
    ann_vol: float
    var: float
    es: float

    Attributes נוספים (עבור דשבורד/ניתוח מתקדם)
    ------------------------------------------
    equity_curve: pd.Series | None
        עקומת equity (starting at 1.0).

    returns: pd.Series | None
        סדרת תשואות יומית/תקופתית אחרי overlay + gates.

    weights: pd.Series | None
        משקולות סופיות של pairs בפורטפוליו (אחרי multipliers + caps + normalize).

    profile: dict | None
        ההעתק המלא של MacroProfile.profile ששימש ב-backtest.

    regimes_prob: pd.DataFrame | None
        עותק של MacroProfile.regimes_prob (אם קיים).

    meta: dict | None
        מטא-דאטה (מספר pairs, תאריכים, שיטת sizing וכו').
    """

    sharpe: float = 0.0
    sortino: float = 0.0
    cagr: float = 0.0
    max_dd: float = 0.0
    uplift_vs_base: float = 0.0
    ann_vol: float = 0.0
    var: float = 0.0
    es: float = 0.0

    equity_curve: Optional[pd.Series] = None
    returns: Optional[pd.Series] = None
    weights: Optional[pd.Series] = None
    profile: Optional[Dict[str, Dict[str, Any]]] = None
    regimes_prob: Optional[pd.DataFrame] = None
    meta: Optional[Dict[str, Any]] = None


# =============================================================================
# Internal helpers: Regime labeling
# =============================================================================
def _label_bucket(r: float, g: float, i: float) -> str:
    """
    מפה פשוטה מ-risk_on/growth/inflation לנקודת משטר (regime) משוקללת.

    r : סיגנל "risk-on" (לדוגמה מ-credit spreads, VIX וכו')
    g : growth
    i : inflation
    """
    if r < -0.5 or (g < -0.4 and i < 0):
        return "risk_off"
    if i > 0.4 and g < 0:
        return "stagflation"
    if i > 0 and g > 0.2:
        return "reflation"
    if g < -0.2 and r <= 0.2:
        return "slowdown"
    return "risk_on" if r > 0 else "neutral"


def _regimes_prob_from_df(regime_df: pd.DataFrame) -> pd.DataFrame:
    """
    ממיר DataFrame של סיגנלי risk_on/growth/inflation ל-one-hot/soft regime probs.

    regime_df:
        index=date, columns כוללות לפחות:
        - "risk_on"
        - "growth"
        - "inflation"

    מחזיר DataFrame:
        index=date, columns=regime labels (rolling mean על 5 ימים להחלקה).
    """
    if regime_df.empty:
        return regime_df.copy()

    labels = ["risk_on", "risk_off", "stagflation", "reflation", "slowdown", "neutral"]
    rows: List[Dict[str, Any]] = []
    for ts, row in regime_df.iterrows():
        r = float(row.get("risk_on", 0.0))
        g = float(row.get("growth", 0.0))
        i = float(row.get("inflation", 0.0))
        lab = _label_bucket(r, g, i)
        o = {l: 0.0 for l in labels}
        o[lab] = 1.0
        o["date"] = ts
        rows.append(o)

    out = pd.DataFrame(rows).set_index("date").sort_index()
    # החלקה קלה (rolling mean) כדי להקטין רעש
    out = out.rolling(5, min_periods=1).mean()
    return out


# =============================================================================
# Fallback using common.macro_adjustments
# =============================================================================
def _try_adjustments(
    pairs_df: pd.DataFrame, cfg_dict: Dict[str, Any]
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Optional[pd.DataFrame],
    Dict[str, Any],
    Optional[pd.DataFrame],
]:
    """
    Fallback/primary path: שימוש ב-common.macro_adjustments לחישוב scores/filters/caps.

    Returns
    -------
    profile: Dict[str, Dict[str, Any]]
    regimes_prob: Optional[pd.DataFrame]
    meta: Dict[str, Any]
    features_summary: Optional[pd.DataFrame]
    """
    try:
        from common.macro_adjustments import (  # type: ignore
            MacroConfig,
            load_macro_bundle,
            compute_adjustments,
            build_regime_indicators,
        )
        # optional features builder
        try:
            from core.macro_features import build_features  # type: ignore
        except Exception:
            build_features = None  # type: ignore[assignment]
    except Exception as e:  # noqa: BLE001
        LOGGER.warning("common.macro_adjustments לא זמין, fallback ניטרלי: %s", e)
        prof: Dict[str, Dict[str, Any]] = {}
        for _, row in pairs_df.iterrows():
            pid = str(row.get("pair_id", f"{row.get('a', '?')}-{row.get('b', '?')}"))
            prof[pid] = {
                "macro_fit_score": None,
                "weight_adj": 1.0,
                "action": "Pause",
                "reason": "macro_adjustments unavailable",
                "include": True,
                "cap_applied": False,
                "cap_value": None,
            }
        return prof, None, {}, None

    # בניית config מ-cfg_dict (רק שדות רלוונטיים)
    cfg = MacroConfig()  # type: ignore[call-arg]
    for k in (
        "score_filter_threshold",
        "score_gain_floor",
        "score_weight_gain",
        "caps_mode",
        "exposure_multiplier_bounds",
        "regime_smoothing_days",
        "sector_caps",
        "region_caps",
        "currency_caps",
    ):
        if k in cfg_dict:
            try:
                setattr(cfg, k, cfg_dict[k])
            except Exception:
                pass

    bundle = load_macro_bundle(cfg)
    result = compute_adjustments(pairs_df, bundle, cfg)

    exp_mult = float(getattr(result, "exposure_multiplier", 1.0))
    scores = getattr(result, "pair_scores", {}) or {}
    mults = getattr(result, "pair_adjustments", {}) or {}
    filters = getattr(result, "filters", {}) or {}
    caps_hints = getattr(result, "caps_hints", {}) or {}
    meta = getattr(result, "meta", {}) or {}

    # הזרקת feature_params למטא במקרה ויש ב-cfg
    if "feature_params" in cfg_dict:
        try:
            meta["feature_params"] = dict(cfg_dict["feature_params"])
        except Exception:
            pass

    profile: Dict[str, Dict[str, Any]] = {}
    for _, row in pairs_df.iterrows():
        pid = str(row.get("pair_id", f"{row.get('a', '?')}-{row.get('b', '?')}"))
        sc = scores.get(pid)
        w = mults.get(pid, 1.0) * exp_mult
        inc = bool(filters.get(pid, True))
        cap = float(caps_hints.get(pid, 1.0))
        cap_applied = False
        if cap < 1.0:
            w = min(w, cap)
            cap_applied = True

        # קביעת action / reason
        if not inc:
            action, reason = "Pause", "Filtered by macro layer"
        elif sc is not None and sc < 30:
            action, reason = "Weaken", "Score below threshold"
        elif sc is not None and sc >= 70:
            action, reason = "Strengthen", "High macro alignment"
        else:
            action, reason = "Neutral", "Baseline"

        profile[pid] = {
            "macro_fit_score": sc,
            "weight_adj": float(np.clip(w, 0.2, 2.0)),
            "action": action,
            "reason": reason,
            "include": inc,
            "cap_applied": cap_applied,
            "cap_value": cap if cap_applied else None,
        }

    # Regime probabilities
    try:
        regime_df = build_regime_indicators(bundle, cfg)
        regimes_prob = _regimes_prob_from_df(regime_df)
    except Exception:
        regimes_prob = None

    # Features summary (optional, אם build_features זמין)
    features_summary: Optional[pd.DataFrame] = None
    try:
        if "bundle" in locals() and build_features is not None:
            df_map = {
                k: v.series.to_frame(name="value")
                for k, v in getattr(bundle, "data", {}).items()
            }
            if df_map:
                feat_params = dict(cfg_dict.get("feature_params", {}))
                features_summary = build_features(df_map, params=feat_params)
    except Exception as e:
        LOGGER.warning("features build failed: %s", e)

    return profile, regimes_prob, meta, features_summary


# =============================================================================
# Public API: compute_profile
# =============================================================================
def compute_profile(pairs_df: pd.DataFrame, cfg: Dict[str, Any]) -> MacroProfile:
    """
    מחשב MacroProfile עבור universe של pairs.

    Strategy:
    ---------
    1. בעתיד: שימוש ב-core/macro_* (מודלים מותאמים אישית של עומרי).
    2. כרגע: שימוש ב-common.macro_adjustments כ-overlay שכבת מאקרו.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        חייבת לכלול לפחות עמודת "pair_id" (או a/b ליצירת pair_id).
    cfg : dict
        קונפיג כללי (עובר ל-MacroConfig דרך _try_adjustments).

    Returns
    -------
    MacroProfile
    """
    np.random.seed(int(cfg.get("seed", 42)))

    # הבטחת pair_id
    if "pair_id" not in pairs_df.columns:
        if "a" in pairs_df.columns and "b" in pairs_df.columns:
            pairs_df = pairs_df.copy()
            pairs_df["pair_id"] = (
                pairs_df["a"].astype(str) + "-" + pairs_df["b"].astype(str)
            )
        else:
            pairs_df = pairs_df.copy()
            pairs_df["pair_id"] = [
                f"pair_{i}" for i in range(len(pairs_df.index))
            ]

    profile, regimes_prob, meta, feats = _try_adjustments(pairs_df, cfg)
    return MacroProfile(
        profile=profile,
        regimes_prob=regimes_prob,
        features_summary=feats,
        meta=meta,
    )


# =============================================================================
# Internal helpers: stats
# =============================================================================
def _ann_factor(index: pd.DatetimeIndex) -> float:
    """
    קובע פקטור שנתי לפי התדירות הממוצעת של ה-index.

    - תדירות יומית ≈ 252
    - שבועית ≈ 52
    - חודשית ≈ 12
    """
    if len(index) < 2:
        return 252.0

    days = (index[-1] - index[0]).days
    if days <= 0:
        return 252.0

    avg_step = days / max(len(index) - 1, 1)
    if avg_step <= 1.5:
        return 252.0
    if avg_step <= 7.5:
        return 52.0
    if avg_step <= 30.5:
        return 12.0
    # fallback: scale לפי גס
    return max(252.0 / avg_step, 1.0)


def _drawdown_stats(series: pd.Series) -> Tuple[float, float, pd.Series]:
    """
    מחשב (max_dd, cagr, equity_curve) עבור סדרת תשואות תקופתית.

    Returns
    -------
    max_dd : float
        drawdown מקסימלי (חיובי, למשל 0.25 עבור 25%).
    cagr : float
    equity_curve : pd.Series
    """
    if series.empty:
        return 0.0, 0.0, pd.Series(dtype=float)

    equity = (1.0 + series.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min()) * -1.0
    total_return = float(equity.iloc[-1] - 1.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    return max_dd, cagr, equity


def _safe_std(x: pd.Series) -> float:
    return float(x.std(ddof=0)) if len(x) > 1 else 0.0


# =============================================================================
# Public API: backtest
# =============================================================================
def backtest(cfg: Dict[str, Any]) -> BacktestResult:
    """
    מריץ backtest overlay מאקרו על universe pairs.

    Expected cfg keys
    -----------------
    - "returns_df": pd.DataFrame
        index=date, columns=pair_id, values=returns.
    - "pairs_df": pd.DataFrame (אופציונלי, אך מומלץ)
        כולל pair_id ועמודות sector/region/currency וכו'.

    אם returns_df לא קיים:
        - מחזיר הערכה גסה על בסיס MacroProfile בלבד (לשימוש כ-sanity check).
    """
    np.random.seed(int(cfg.get("seed", 42)))

    returns_df = cfg.get("returns_df")
    pairs_df = cfg.get("pairs_df")

    # -------------------------------------------------------------------------
    # Case 1: יש returns_df => Backtest מלא
    # -------------------------------------------------------------------------
    if returns_df is not None and isinstance(returns_df, pd.DataFrame):
        # אם pairs_df חסר – נבנה מינימלי מתוך העמודות
        if pairs_df is None or not isinstance(pairs_df, pd.DataFrame):
            pairs_df = pd.DataFrame({"pair_id": list(returns_df.columns)})

        # Macro profile (overlay)
        prof = compute_profile(pairs_df, cfg)
        overlay = prof.profile  # dict[pair_id]-> dict(...)

        enable_overlay = bool(cfg.get("enable_macro_overlay", True))

        # multipliers: include=False -> 0
        multipliers: Dict[str, float] = {}
        for pid in returns_df.columns:
            if (pid in overlay) and enable_overlay:
                row = overlay[pid]
                mult = float(row.get("weight_adj", 1.0))
                if not row.get("include", True):
                    mult = 0.0
            else:
                mult = 1.0
            multipliers[pid] = mult

        # פילטר universe לפי multipliers>0
        cols = [c for c in returns_df.columns if multipliers.get(c, 0.0) > 0.0]
        if not cols:
            LOGGER.warning("backtest: no active pairs after overlay filtering.")
            return BacktestResult(profile=overlay, regimes_prob=prof.regimes_prob)

        rets = returns_df[cols].dropna(how="all").fillna(0.0)
        if rets.empty:
            LOGGER.warning("backtest: returns_df after cleaning is empty.")
            return BacktestResult(profile=overlay, regimes_prob=prof.regimes_prob)

        # Risk sizing parameters
        method = str(cfg.get("risk_sizing", "invvol")).lower()
        tgt_vol = float(cfg.get("target_vol", cfg.get("volatility_target", 0.15)))
        cov_window = int(cfg.get("cov_window", 60))
        shrink = float(cfg.get("cov_shrink", 0.1))
        max_pair_weight = float(cfg.get("max_pair_weight", 0.1))

        # Base weights (risk engine) – אם אין risk_Helpers, נ fallback ל-EW
        base_w: Optional[pd.Series]
        if target_weights_from_returns is not None:
            try:
                base_w_obj = target_weights_from_returns(
                    rets,
                    target_vol=tgt_vol,
                    method=method,
                    cov_window=cov_window,
                    shrink=shrink,
                    max_leverage=1.0,
                    floor=0.0,
                    cap=max_pair_weight,
                )
                # תמיכה גם במקרה ש-target_weights_from_returns מחזיר DataFrame (time-varying)
                if isinstance(base_w_obj, pd.DataFrame):
                    base_w = base_w_obj.iloc[-1].astype(float)
                else:
                    base_w = base_w_obj.astype(float)  # type: ignore[assignment]
            except Exception as e:
                LOGGER.warning(
                    "target_weights_from_returns failed (%s), falling back to equal-weight.",
                    e,
                )
                base_w = None
        else:
            base_w = None

        if base_w is None or base_w.empty:
            base_w = pd.Series(1.0 / len(cols), index=cols, dtype=float)

        # החלת multipliers
        if apply_multipliers is not None:
            try:
                w = apply_multipliers(
                    base_w,
                    multipliers,
                    cap=max_pair_weight,
                    floor=0.0,
                    renorm=True,
                )
            except Exception as e:
                LOGGER.warning(
                    "apply_multipliers failed (%s), falling back to manual scaling.",
                    e,
                )
                w = base_w.copy()
                for k, m in multipliers.items():
                    if k in w.index:
                        w.loc[k] = min(
                            max(w.loc[k] * m, 0.0),
                            max_pair_weight,
                        )
                s = w.sum()
                if s > 0:
                    w = w / s
        else:
            # Fallback: scaling ידני
            w = base_w.copy()
            for k, m in multipliers.items():
                if k in w.index:
                    w.loc[k] = min(
                        max(w.loc[k] * m, 0.0),
                        max_pair_weight,
                    )
            s = w.sum()
            if s > 0:
                w = w / s

        # Group caps (sector/region/currency) אם זמינים
        try:
            if apply_caps is not None and isinstance(pairs_df, pd.DataFrame):

                def _apply_group_caps(
                    w_ser: pd.Series,
                    caps_map: Optional[Dict[str, float]],
                    col_name: str,
                ) -> pd.Series:
                    if not caps_map:
                        return w_ser
                    if "pair_id" not in pairs_df.columns or col_name not in pairs_df.columns:
                        return w_ser
                    groups = (
                        pairs_df.set_index("pair_id")[col_name].astype(str).to_dict()
                    )
                    return apply_caps(w_ser, caps_map, groups)

                w = _apply_group_caps(w, cfg.get("sector_caps"), "sector")
                w = _apply_group_caps(w, cfg.get("region_caps"), "region")
                w = _apply_group_caps(w, cfg.get("currency_caps"), "currency")
                if normalize_weights is not None:
                    w = normalize_weights(w)
                else:
                    s = w.sum()
                    if s > 0:
                        w = w / s
        except Exception as e:
            LOGGER.warning("group caps application failed: %s", e)

        # יישור לפי עמודות rets
        w = w.reindex(rets.columns).fillna(0.0)
        if w.sum() <= 0:
            LOGGER.warning("backtest: final weights sum to zero.")
            return BacktestResult(profile=overlay, regimes_prob=prof.regimes_prob)

        # Portfolio returns *לפני* Kill switch
        port = pd.Series(
            rets.values.dot(w.values),
            index=rets.index,
            name="portfolio_ret",
        )

        # Optional kill-switch via DrawdownGate
        ks = float(cfg.get("kill_switch_drawdown", 0.0) or 0.0)
        cool = int(cfg.get("kill_cooloff", 5))
        if ks > 0.0 and DrawdownGate is not None:
            try:
                gate = DrawdownGate(threshold=ks, cooloff=cool)
                eq = 1.0
                gated_vals: List[float] = []
                cool_ctr = 0
                for r in port.values:
                    if cool_ctr > 0:
                        gated_vals.append(0.0)
                        eq *= 1.0
                        cool_ctr -= 1
                        gate.update(eq)
                        continue
                    eq *= 1.0 + float(r)
                    tripped, _ = gate.update(eq)
                    gated_vals.append(float(r))
                    if tripped:
                        cool_ctr = cool
                port = pd.Series(gated_vals, index=port.index, name="portfolio_ret")
            except Exception as e:
                LOGGER.warning("DrawdownGate failed, using ungated returns: %s", e)

        # Risk-free rate (annual)
        rf_ann = float(cfg.get("risk_free", 0.0))

        # Metrics
        ann = _ann_factor(port.index)
        mu_gross = float(port.mean()) * ann
        sd = _safe_std(port) * np.sqrt(ann)
        neg = port[port < 0]
        sd_down = (
            float(neg.std(ddof=0)) * np.sqrt(ann) if len(neg) > 1 else 0.0
        )

        # Net excess מול risk-free
        mu = mu_gross - rf_ann

        sharpe = (mu / sd) if sd > 1e-12 else 0.0
        sortino = (mu / sd_down) if sd_down > 1e-12 else 0.0
        max_dd, cagr, equity = _drawdown_stats(port)

        # uplift מול EW baseline (ללא overlay ו-caps)
        ew = np.ones(len(rets.columns), dtype=float) / float(len(rets.columns))
        base = pd.Series(
            rets.values.dot(ew),
            index=rets.index,
            name="baseline_ret",
        )
        uplift = mu_gross - float(base.mean()) * ann

        # extra risk metrics (אם portfolio_metrics זמין)
        ann_vol = float(sd)
        var_val = 0.0
        es_val = 0.0
        try:
            if portfolio_metrics is not None:
                m = portfolio_metrics(w, rets)
                ann_vol = float(m.get("ann_vol", ann_vol))
                var_val = float(m.get("var", 0.0))
                es_val = float(m.get("es", 0.0))
        except Exception as e:
            LOGGER.warning("portfolio_metrics failed: %s", e)

        meta: Dict[str, Any] = {
            "n_pairs": int(len(w)),
            "start": port.index[0] if len(port) else None,
            "end": port.index[-1] if len(port) else None,
            "risk_sizing": method,
            "target_vol": tgt_vol,
            "cov_window": cov_window,
            "cov_shrink": shrink,
            "max_pair_weight": max_pair_weight,
            "kill_switch_drawdown": ks,
            "kill_cooloff": cool,
            "risk_free": rf_ann,
        }
        # מיזוג מטא מה-MacroProfile אם קיים
        if prof.meta:
            meta["macro_meta"] = prof.meta

        return BacktestResult(
            sharpe=float(sharpe),
            sortino=float(sortino),
            cagr=float(cagr),
            max_dd=float(max_dd),
            uplift_vs_base=float(uplift),
            ann_vol=ann_vol,
            var=var_val,
            es=es_val,
            equity_curve=equity,
            returns=port,
            weights=w,
            profile=overlay,
            regimes_prob=prof.regimes_prob,
            meta=meta,
        )

    # -------------------------------------------------------------------------
    # Case 2: אין returns_df – הערכה היברידית על בסיס MacroProfile בלבד
    # -------------------------------------------------------------------------
    pairs_df = cfg.get("pairs_df")
    if isinstance(pairs_df, pd.DataFrame) and not pairs_df.empty:
        prof = compute_profile(pairs_df, cfg)
        inc = sum(1 for v in prof.profile.values() if v.get("include", True))
        n = max(len(prof.profile), 1)
        mean_w = float(
            np.mean([v.get("weight_adj", 1.0) for v in prof.profile.values()])
        )
        # אומדן גס: "Sharpe" פיקטיבי כדי לקבל סיגנל איכות
        sharpe_proxy = (inc / n) * (mean_w)
        return BacktestResult(
            sharpe=float(sharpe_proxy),
            sortino=sharpe_proxy * 0.9,
            cagr=sharpe_proxy * 0.1,
            max_dd=0.1,
            uplift_vs_base=sharpe_proxy * 0.05,
            profile=prof.profile,
            regimes_prob=prof.regimes_prob,
            meta={"mode": "profile_only"},
        )

    # אין נתונים – מחזירים אובייקט ריק
    return BacktestResult(meta={"mode": "empty"})


# =============================================================================
# File parsing helpers (optional)
# =============================================================================
def _detect_fmt_from_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    lname = str(name).lower()
    if lname.endswith(".parquet") or lname.endswith(".pq"):
        return "parquet"
    if lname.endswith(".csv") or lname.endswith(".txt"):
        return "csv"
    return None


def parse_returns_df(obj: Any, name: Optional[str] = None) -> pd.DataFrame:
    """
    ממיר input ל-DataFrame תשואות סטנדרטי: index=date, columns=pair_id, values=float.

    Parameters
    ----------
    obj : Any
        Path / file-like / bytes / buffer / DataFrame / dict-like.
    name : Optional[str]
        שם קובץ (לזיהוי פורמט CSV/Parquet).

    Returns
    -------
    pd.DataFrame
    """
    # אם כבר DataFrame – ננקה ונתאים
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        fmt = _detect_fmt_from_name(name)
        try:
            if fmt == "parquet":
                import io

                if isinstance(obj, (bytes, bytearray)):
                    df = pd.read_parquet(io.BytesIO(obj))
                else:
                    df = pd.read_parquet(obj)
            else:  # CSV (default)
                import io

                if isinstance(obj, (bytes, bytearray)):
                    df = pd.read_csv(io.BytesIO(obj))
                else:
                    df = pd.read_csv(obj)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"failed to read returns file: {e}")

    # זיהוי עמודת תאריך
    date_col: Optional[str] = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("date", "time", "timestamp", "dt"):
            date_col = c
            break
    if date_col is None:
        # מניחים שהעמודה הראשונה היא תאריך
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # המרה לערכים נומריים וניקוי NaN
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0.0)

    return df


def parse_pairs_df(obj: Any, name: Optional[str] = None) -> pd.DataFrame:
    """
    ממיר input ל-DataFrame pairs עם עמודת pair_id.

    Parameters
    ----------
    obj : Any
        Path / file-like / bytes / buffer / DataFrame / dict-like.
    name : Optional[str]
        שם קובץ (לזיהוי פורמט CSV/Parquet).

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        fmt = _detect_fmt_from_name(name)
        try:
            if fmt == "parquet":
                import io

                if isinstance(obj, (bytes, bytearray)):
                    df = pd.read_parquet(io.BytesIO(obj))
                else:
                    df = pd.read_parquet(obj)
            else:
                import io

                if isinstance(obj, (bytes, bytearray)):
                    df = pd.read_csv(io.BytesIO(obj))
                else:
                    df = pd.read_csv(obj)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"failed to read pairs file: {e}")

    if "pair_id" not in df.columns:
        # מנסים לבנות מ-a/b
        if "a" in df.columns and "b" in df.columns:
            df["pair_id"] = df["a"].astype(str) + "-" + df["b"].astype(str)
        else:
            # fallback: pair_0, pair_1,...
            df["pair_id"] = [f"pair_{i}" for i in range(len(df))]

    return df


def run_backtest_with_files(
    returns_obj: Any,
    pairs_obj: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
    returns_name: Optional[str] = None,
    pairs_name: Optional[str] = None,
) -> BacktestResult:
    """
    Utility: קורא קבצי Returns/Pairs ומריץ backtest עליהם.

    Parameters
    ----------
    returns_obj : Any
        קובץ תשואות (path/bytes/DataFrame וכו').
    pairs_obj : Any, optional
        קובץ pairs (אם None – בונה pair_id מהעמודות של returns_df בלבד).
    cfg : dict, optional
        קונפיג עבור backtest.
    returns_name, pairs_name : str, optional
        שמות קבצים (לזיהוי CSV/Parquet).

    Returns
    -------
    BacktestResult
    """
    cfg = dict(cfg or {})
    returns_df = parse_returns_df(returns_obj, name=returns_name)
    pairs_df = (
        parse_pairs_df(pairs_obj, name=pairs_name) if pairs_obj is not None else None
    )
    cfg["returns_df"] = returns_df
    if pairs_df is not None:
        cfg["pairs_df"] = pairs_df
    return backtest(cfg)


__all__ = [
    "MacroProfile",
    "BacktestResult",
    "compute_profile",
    "backtest",
    "parse_returns_df",
    "parse_pairs_df",
    "run_backtest_with_files",
]
