from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Literal, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    # לא מגדירים basicConfig בספרייה – משאירים למערכת הראשית
    logger.addHandler(logging.NullHandler())


# ============================================================================
# Enums & Data Models
# ============================================================================


class TradeSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class TradeAction(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    HOLD = "hold"
    FORCE_CLOSE = "force_close"  # hard-stop / risk control


@dataclass(frozen=True)
class ZScoreRules:
    """
    חוקים לפתיחת/סגירת עסקה לפי Z-score.

    Parameters
    ----------
    open_threshold : float
        רמת כניסה בסיסית (abs(z) >= open_threshold).
    close_threshold : float
        רמת יציאה בסיסית (abs(z) <= close_threshold).
    hard_stop : Optional[float]
        רמת Z קיצונית שמחייבת סגירה מיידית (למשל 4.0).
    hysteresis : float
        היסטרזיס בין כניסה ליציאה – כדי למנוע "ריצודים" מיותרים סביב הסף.
        לדוגמה: open=2.0, close=0.5 → hysteresis=1.5.
    """

    open_threshold: float
    close_threshold: float
    hard_stop: Optional[float] = None
    hysteresis: float = 0.0


@dataclass(frozen=True)
class EdgeConfig:
    """
    קונפיגורציה לחישוב Edge.

    Parameters
    ----------
    vol_floor : float
        מינימום ל-volatility_ratio כדי להימנע מחלוקה במספר קטן מדי.
    max_edge : Optional[float]
        תקרת Edge (אם None – אין תקרה).
    round_to : int
        מספר ספרות אחרי הנקודה לעיגול.
    scaling : float
        מקדם סקלה בסיסי (להביא את ה-Edge לסקאלה של 0–10 / 0–100).
    """

    vol_floor: float = 1e-4
    max_edge: Optional[float] = None
    round_to: int = 2
    scaling: float = 1.0


@dataclass(frozen=True)
class RegimeProfile:
    """
    פרופיל סגנון / רמת סיכון של האסטרטגיה.

    Parameters
    ----------
    name : str
        שם הפרופיל (לוגים / UI).
    open_multiplier : float
        מכפיל על open_threshold לפי משטר (regime).
    close_multiplier : float
        מכפיל על close_threshold לפי משטר.
    edge_min : float
        מינימום Edge הנדרש לכניסה.
    """

    name: str
    open_multiplier: float = 1.0
    close_multiplier: float = 1.0
    edge_min: float = 0.0


@dataclass
class TradeDecision:
    """
    תוצאת החלטה למסחר: האם לפתוח / לסגור / להחזיק, ומה הסיבה.

    Attributes
    ----------
    action : TradeAction
        מה לבצע (OPEN / CLOSE / HOLD / FORCE_CLOSE).
    side : Optional[TradeSide]
        LONG / SHORT כאשר פותחים עסקה. None אם אין שינוי.
    reason : str
        תיאור קצר להסבר (לוג/דיבאג).
    score : float
        ציון איכות החלטה (למשל Edge).
    meta : Dict[str, Any]
        מידע נוסף (zscore, edge, thresholds וכו').
    """

    action: TradeAction
    side: Optional[TradeSide]
    reason: str
    score: float
    meta: Dict[str, Any]


# ============================================================================
# Core Helpers – basic Z rules
# ============================================================================


def should_open_trade(zscore: float, rules: ZScoreRules | float) -> bool:
    """
    האם תנאי יסוד לפתיחת עסקה מתקיים לפי Z-score (בלי Edge / משטר).

    - אם rules הוא float → מתייחס אליו כ-open_threshold.
    - אם זה ZScoreRules → משתמש ב-open_threshold.
    """
    if isinstance(rules, (float, int)):
        thr = float(rules)
    else:
        thr = float(rules.open_threshold)

    decision = abs(zscore) >= thr
    logger.debug("should_open_trade[z=%.3f, thr=%.3f] -> %s", zscore, thr, decision)
    return decision


def should_close_trade(zscore: float, rules: ZScoreRules | float) -> bool:
    """
    האם תנאי יציאה בסיסי מתקיים לפי Z-score (בלי משטר / Edge).

    - אם rules הוא float → מתייחס אליו כ-close_threshold.
    - אם זה ZScoreRules → משתמש ב-close_threshold ו-hard_stop אם קיים.
    """
    if isinstance(rules, (float, int)):
        close_thr = float(rules)
        decision = abs(zscore) <= close_thr
        logger.debug("should_close_trade[z=%.3f, close=%.3f] -> %s", zscore, close_thr, decision)
        return decision

    # hard-stop
    if rules.hard_stop is not None and abs(zscore) >= float(rules.hard_stop):
        logger.debug(
            "should_close_trade[z=%.3f] -> FORCE_CLOSE (hard_stop=%.3f)",
            zscore,
            rules.hard_stop,
        )
        return True

    close_thr = float(rules.close_threshold)
    decision = abs(zscore) <= close_thr
    logger.debug(
        "should_close_trade[z=%.3f, close_thr=%.3f] -> %s",
        zscore,
        close_thr,
        decision,
    )
    return decision


# ============================================================================
# Edge computation & regimes
# ============================================================================


def evaluate_edge(
    zscore: float,
    volatility_ratio: float,
    cfg: EdgeConfig | None = None,
) -> float:
    """
    מחשב Edge "נורמלי" על בסיס Z-score ותנודתיות יחסית.

    רעיון בסיסי:
    ------------
    - abs(z) גדול → סטייה חזקה.
    - volatility_ratio קטן → תנודתיות נוכחית נמוכה → Edge גבוה.
    - volatility_ratio גדול → שוק "פרוע" → Edge קטן יותר (אות פחות נקי).

    Edge בסיסי = scaling * abs(z) / max(volatility_ratio, vol_floor).

    אם max_edge לא None → מגבילים לתקרה.
    """
    cfg = cfg or EdgeConfig()
    vol = float(volatility_ratio)

    if not np.isfinite(vol) or vol <= 0:
        logger.debug("evaluate_edge: invalid vol_ratio=%r -> edge=0.0", volatility_ratio)
        return 0.0

    eff_vol = max(vol, cfg.vol_floor)
    raw = cfg.scaling * (abs(float(zscore)) / eff_vol)

    if cfg.max_edge is not None:
        raw = min(raw, float(cfg.max_edge))

    edge = round(float(raw), cfg.round_to)
    logger.debug(
        "evaluate_edge[z=%.3f, vol=%.6f, eff_vol=%.6f, scaling=%.3f] -> edge=%.3f",
        zscore,
        vol,
        eff_vol,
        cfg.scaling,
        edge,
    )
    return edge


def detect_regime(
    realized_vol: float,
    long_term_vol: float,
    threshold_low: float = 0.8,
    threshold_high: float = 1.2,
) -> Literal["calm", "normal", "stressed"]:
    """
    קובע "משטר תנודתיות" בסיסי לפי יחס vol_current / vol_long_term.

    Returns
    -------
    "calm"    : תנודתיות נוכחית קטנה באופן משמעותי מהארוך.
    "normal"  : סביב הנורמל.
    "stressed": גבוהה מאוד.
    """
    if long_term_vol <= 0 or not np.isfinite(long_term_vol):
        return "normal"

    ratio = realized_vol / long_term_vol

    if ratio <= threshold_low:
        return "calm"
    if ratio >= threshold_high:
        return "stressed"
    return "normal"


def get_regime_profile(name: Literal["defensive", "balanced", "aggressive"]) -> RegimeProfile:
    """
    פרופילי ברירת-מחדל למשקיע:

    - defensive : דורש Edge גבוה יותר ו-Z פתיחה חזק יותר.
    - balanced  : ברירת מחדל.
    - aggressive: thresholds נמוכים יותר, נכנסים בקלות.
    """
    if name == "defensive":
        return RegimeProfile(
            name="defensive",
            open_multiplier=1.2,
            close_multiplier=1.0,
            edge_min=2.0,
        )
    if name == "aggressive":
        return RegimeProfile(
            name="aggressive",
            open_multiplier=0.8,
            close_multiplier=1.0,
            edge_min=0.5,
        )
    return RegimeProfile(
        name="balanced",
        open_multiplier=1.0,
        close_multiplier=1.0,
        edge_min=1.0,
    )


# ============================================================================
# High-level decision engine
# ============================================================================


def decide_trade_action(
    zscore: float,
    in_position: bool,
    side: Optional[TradeSide],
    rules: ZScoreRules,
    edge: float,
    regime_profile: RegimeProfile,
) -> TradeDecision:
    """
    מוח החלטה מלא עבור עסקה על בסיס Z-score + Edge + Regime Profile.

    Parameters
    ----------
    zscore : float
        ערך Z נוכחי של הספרד.
    in_position : bool
        האם יש כבר עסקה פתוחה.
    side : Optional[TradeSide]
        כיוון עסקה קיים (אם יש): LONG / SHORT.
    rules : ZScoreRules
        חוקים בסיסיים לכניסה/יציאה לפי Z.
    edge : float
        Edge מחושב (evaluate_edge).
    regime_profile : RegimeProfile
        פרופיל סגנון (defensive/balanced/aggressive).

    Returns
    -------
    TradeDecision
        החלטה מלאה (action, side, reason, score, meta).
    """
    z = float(zscore)
    abs_z = abs(z)

    # התאמת thresholds למשטר
    open_thr = rules.open_threshold * regime_profile.open_multiplier
    close_thr = rules.close_threshold * regime_profile.close_multiplier

    # hard stop
    if in_position and rules.hard_stop is not None and abs_z >= rules.hard_stop:
        reason = f"Hard stop triggered: |Z|={abs_z:.3f} ≥ {rules.hard_stop:.3f}"
        return TradeDecision(
            action=TradeAction.FORCE_CLOSE,
            side=side,
            reason=reason,
            score=edge,
            meta={
                "z": z,
                "abs_z": abs_z,
                "open_thr": open_thr,
                "close_thr": close_thr,
                "hard_stop": rules.hard_stop,
                "edge": edge,
                "regime": regime_profile.name,
            },
        )

    # אין עסקה פתוחה → בודקים כניסה
    if not in_position:
        # דרישת מינימום Edge
        if edge < regime_profile.edge_min:
            reason = f"Edge too low ({edge:.2f} < {regime_profile.edge_min:.2f}), skip"
            return TradeDecision(
                action=TradeAction.HOLD,
                side=None,
                reason=reason,
                score=edge,
                meta={"z": z, "abs_z": abs_z, "edge": edge, "regime": regime_profile.name},
            )

        if abs_z >= open_thr:
            # כיוון – אם z חיובי → spread>0 → SHORT spread (לונג leg זול / שורט leg יקר)
            # אם z שלילי → LONG spread
            new_side = TradeSide.SHORT if z > 0 else TradeSide.LONG
            reason = f"Open {new_side.value} — |Z|={abs_z:.3f} ≥ {open_thr:.3f}, edge={edge:.2f}"
            return TradeDecision(
                action=TradeAction.OPEN,
                side=new_side,
                reason=reason,
                score=edge,
                meta={
                    "z": z,
                    "abs_z": abs_z,
                    "open_thr": open_thr,
                    "close_thr": close_thr,
                    "edge": edge,
                    "regime": regime_profile.name,
                },
            )

        reason = f"No entry — |Z|={abs_z:.3f} < {open_thr:.3f} or edge<min"
        return TradeDecision(
            action=TradeAction.HOLD,
            side=None,
            reason=reason,
            score=edge,
            meta={"z": z, "abs_z": abs_z, "open_thr": open_thr, "edge": edge, "regime": regime_profile.name},
        )

    # יש עסקה פתוחה → בודקים יציאה
    # אם הוגדר hysteresis – מוודאים שאנחנו לא סוגרים מהר מדי
    effective_close_thr = close_thr
    if rules.hysteresis > 0:
        # למשל: לא נסגור אם עדיין קרובים מאוד ל-open_thr
        effective_close_thr = min(close_thr, max(0.0, open_thr - rules.hysteresis))

    if abs_z <= effective_close_thr:
        reason = f"Close position — |Z|={abs_z:.3f} ≤ {effective_close_thr:.3f}"
        return TradeDecision(
            action=TradeAction.CLOSE,
            side=side,
            reason=reason,
            score=edge,
            meta={
                "z": z,
                "abs_z": abs_z,
                "open_thr": open_thr,
                "close_thr": close_thr,
                "effective_close_thr": effective_close_thr,
                "edge": edge,
                "regime": regime_profile.name,
            },
        )

    # אחרת מחזיקים
    reason = f"Hold — |Z|={abs_z:.3f} > {effective_close_thr:.3f}, edge={edge:.2f}"
    return TradeDecision(
        action=TradeAction.HOLD,
        side=side,
        reason=reason,
        score=edge,
        meta={
            "z": z,
            "abs_z": abs_z,
            "open_thr": open_thr,
            "close_thr": close_thr,
            "effective_close_thr": effective_close_thr,
            "edge": edge,
            "regime": regime_profile.name,
        },
    )


# ============================================================================
# Utilities – position sizing & serialization
# ============================================================================


def compute_position_size(
    edge: float,
    capital: float,
    risk_per_trade: float,
    price: float,
    vol_estimate: Optional[float] = None,
    max_leverage: float = 3.0,
) -> int:
    """
    מחשב גודל פוזיציה מומלץ (ביחידות), על בסיס Edge, הון, סיכון ו-vol.

    הרעיון:
    --------
    - risk_dollars = capital * risk_per_trade
    - per_unit_risk ≈ price * max(vol_estimate, 1%–2%)
    - units = min( risk_dollars / per_unit_risk, capital * max_leverage / price )
    """
    if capital <= 0 or price <= 0 or risk_per_trade <= 0:
        return 0

    risk_dollars = capital * risk_per_trade

    vol = float(vol_estimate) if vol_estimate is not None else 0.02
    vol = max(vol, 0.005)  # לא פחות מ-0.5% תנודתיות
    per_unit_risk = price * vol

    raw_units = risk_dollars / per_unit_risk

    # משקל לפי Edge (Edge גבוה → מעט יותר אגרסיבי, Edge נמוך → מקטין)
    edge_factor = max(0.0, min(edge / 5.0, 2.0)) if np.isfinite(edge) else 1.0
    units = raw_units * edge_factor

    # מגבלת מינוף בסיסית
    max_units_by_leverage = (capital * max_leverage) / price
    units = min(units, max_units_by_leverage)

    return int(max(0, np.floor(units)))


def trade_decision_to_dict(decision: TradeDecision) -> Dict[str, Any]:
    """
    ממיר TradeDecision ל-dict נוח ללוגים / JSON.
    """
    return {
        "action": decision.action.value,
        "side": decision.side.value if decision.side is not None else None,
        "reason": decision.reason,
        "score": float(decision.score),
        "meta": decision.meta,
    }


# ============================================================================
# Quick self-test (אופציונלי)
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

    rules = ZScoreRules(open_threshold=2.0, close_threshold=0.5, hard_stop=4.0, hysteresis=1.0)
    regime = get_regime_profile("balanced")

    z = 2.7
    vol_ratio = 0.8
    edge_val = evaluate_edge(z, vol_ratio, EdgeConfig(scaling=1.0, max_edge=10.0))

    dec = decide_trade_action(
        zscore=z,
        in_position=False,
        side=None,
        rules=rules,
        edge=edge_val,
        regime_profile=regime,
    )
    print("Decision:", trade_decision_to_dict(dec))

    size = compute_position_size(
        edge=edge_val,
        capital=100_000,
        risk_per_trade=0.01,
        price=100.0,
        vol_estimate=0.02,
    )
    print("Suggested units:", size)
