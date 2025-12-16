# -*- coding: utf-8 -*-
"""
core/ib_order_router.py — IB Order Router (HF-grade Skeleton)
=============================================================

שכבת Orders מעל IBKR, יושבת מעל root/ibkr_connection.py.

עקרונות:
---------
- הפרדה חדה בין:
    • חיבור ל-IB (ibkr_connection.get_ib_instance)
    • לוגיקת הזמנה (מסחר זוגות / חיסול פוזיציות / סנכרון פוזיציות)
    • לוגיקת סיכון (pre_trade_check) — hook להרחבה.
- שימוש ב-ib_insync אם זמין, אך המודול לא "מת" אם אין אותו (פשוט לא יוכל לשגר Orders).

המודול הזה הוא שלד בטוח יחסית:
- לא עושה לוגיקה מורכבת של sizing / הגנות — זה נשאר לריסק שלך.
- כן מרכז את כל ה-IB orders במקום אחד, וקל לשים עליו Checks, Logs ו-Kill-Switchים.

שימוש בסיסי:
-------------
    from core.ib_order_router import IBOrderRouter, PairOrderLeg, PairOrderRequest

    router = IBOrderRouter(settings=app_ctx.settings.as_dict())

    req = PairOrderRequest(
        pair_id="SPY-QQQ",
        legs=[
            PairOrderLeg(symbol="SPY", action="BUY", quantity=50),
            PairOrderLeg(symbol="QQQ", action="SELL", quantity=50),
        ],
    )

    result = router.submit_pair_order(req)

    if result.success:
        print("Orders sent:", result.order_ids)
    else:
        print("Error:", result.error)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Mapping, Tuple

import logging

from root.ibkr_connection import (
    get_ib_instance,
    ib_connection_status,
    IBConnectionManager,
)

if TYPE_CHECKING:  # רק בשביל type hints, לא בזמן ריצה
    from ib_insync import IB as IBType, Stock, Forex, Contract, Order  # pragma: no cover
else:
    IBType = Any  # בזמן ריצה לא נכפה ib_insync


logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [ib_order_router] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# =========================
# Lazy imports of ib_insync
# =========================

_TRADE_CLASSES: Dict[str, Any] = {}
_TRADE_IMPORT_ERROR: Optional[BaseException] = None


def _import_trade_classes() -> Dict[str, Any]:
    """
    Lazy-import של המחלקות הנפוצות מה-ib_insync:
      - Stock, Forex
      - Order, MarketOrder, LimitOrder

    מחזיר dict עם השמות, או {} אם ib_insync לא זמין.

    התנהגות חשובה:
    ---------------
    - אם ib_insync *לא מותקן* → שומרים את השגיאה ב-_TRADE_IMPORT_ERROR, מחזירים {}.
    - אם יש RuntimeError על "no current event loop" בזמן import →
      לא נחשב ככשל סופי, נרשום WARNING וננסה שוב מאוחר יותר.
    """
    global _TRADE_CLASSES, _TRADE_IMPORT_ERROR

    # כבר נטען בהצלחה בעבר
    if _TRADE_CLASSES:
        return _TRADE_CLASSES

    # אם כבר זיהינו שחסר ib_insync (ImportError אמיתי) – אין למה לנסות שוב
    if isinstance(_TRADE_IMPORT_ERROR, ImportError):
        return {}

    try:
        from ib_insync import (  # type: ignore
            Stock,
            Forex,
            Order,
            MarketOrder,
            LimitOrder,
        )

        _TRADE_CLASSES = {
            "Stock": Stock,
            "Forex": Forex,
            "Order": Order,
            "MarketOrder": MarketOrder,
            "LimitOrder": LimitOrder,
        }
        logger.info("ib_insync trade classes imported successfully")
        return _TRADE_CLASSES

    except ImportError as e:
        # ספרייה לא מותקנת – זו שגיאה "אמיתית"
        _TRADE_IMPORT_ERROR = e
        logger.warning("ib_insync is not installed — order routing will be unavailable.")
        return {}

    except Exception as e:
        msg = str(e).lower()

        # מקרה מיוחד: אין event loop עדיין ב-thread הזה (classic Streamlit+eventkit)
        if isinstance(e, RuntimeError) and "no current event loop" in msg:
            logger.warning(
                "ib_insync trade classes not imported yet (no current event loop); "
                "will retry later when an event loop is available."
            )
            # חשוב: *לא* לשמור את e ב-_TRADE_IMPORT_ERROR,
            # כדי שנוכל לנסות שוב אחרי ש-ibkr_connection ירים event loop.
            return {}

        # כל שאר השגיאות → נשמור ונראה stacktrace פעם אחת
        _TRADE_IMPORT_ERROR = e
        logger.exception("Failed to import ib_insync trade classes: %s", e)
        return {}

# =========================
# Dataclasses for requests
# =========================

@dataclass
class PairOrderLeg:
    """
    רגל בודדת ב-Order של זוג.

    Parameters
    ----------
    symbol : str
        סימבול (למשל "SPY", "QQQ").
    action : str
        BUY / SELL.
    quantity : float
        כמות (shares / יחידות).
    sec_type : str
        סוג נכס: STK / FX / FUT / OPT... (ברירת מחדל STK).
    currency : str
        מטבע (ברירת מחדל USD).
    exchange : str
        בורסה לוגית, לדוגמה "SMART".
    """
    symbol: str
    action: str  # "BUY" / "SELL"
    quantity: float
    sec_type: str = "STK"
    currency: str = "USD"
    exchange: str = "SMART"
    primary_exchange: Optional[str] = None


@dataclass
class PairOrderRequest:
    """
    בקשה למסחר זוגי (שתי רגליים או יותר).

    Parameters
    ----------
    pair_id : str
        מזהה לוגי לזוג (SPY-QQQ, AAPL-MSFT וכו').
    legs : List[PairOrderLeg]
        רשימת הרגליים (לרוב שתיים: long / short).
    order_type : str
        MKT / LMT (מימוש ראשון — רק MKT/LMT).
    limit_price : Optional[float]
        מחיר Limit (אם order_type="LMT").
    time_in_force : str
        DAY / GTC / IOC...
    account : Optional[str]
        חשבון IB ספציפי (אם יש לך שכבת account routing).
    allow_partial : bool
        האם מותר חלק מהפקודות לעבור (True) או הכל/כלום (False — סיבוב עתידי).
    tags : Dict[str, Any]
        שדה חופשי למטא־דאטה (strategy, run_id, וכו').
    """
    pair_id: str
    legs: List[PairOrderLeg]
    order_type: str = "MKT"
    limit_price: Optional[float] = None
    time_in_force: str = "DAY"
    account: Optional[str] = None
    allow_partial: bool = True
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairOrderResult:
    """
    תוצאת ניסיון מסחר זוגי.

    Attributes
    ----------
    success : bool
        האם המסחר הוגדר כ"מוצלח" (כלומר — לפחות הוזנו Orders ל-IB).
    order_ids : List[int]
        מזהי Orders שחזרו מ-IB.
    error : Optional[str]
        הודעת שגיאה אנושית (אם הייתה).
    details : Dict[str, Any]
        מטא־דאטה נוסף (כמו statuses, fills ידועים וכו').
    """
    success: bool
    order_ids: List[int] = field(default_factory=list)
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


# =========================
# Risk / Pre-trade hook
# =========================

def pre_trade_check(
    request: PairOrderRequest,
    *,
    max_notional_per_leg: Optional[float] = None,
    max_quantity_per_leg: Optional[float] = None,
    allowed_symbols: Optional[Sequence[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Hook פשוט ל-check לפני שליחת Orders ל-IB.

    לא מממש Risk Engine אמיתי — רק מעטפת שתוכל להרחיב:
      - max_notional_per_leg: לא עובר את גודל הפוזיציה בדולרים (approx).
      - max_quantity_per_leg: לא עובר כמות יחידות.
      - allowed_symbols: רשימת סימבולים מותרת (white-list).

    מחזיר:
      (True, None)  → אם אין בעיה.
      (False, "סיבה") → אם לא עובר את הבדיקה.
    """
    # בדיקת סימבול ברשימת ה-allowed
    if allowed_symbols is not None:
        allowed_set = {s.upper() for s in allowed_symbols}
        for leg in request.legs:
            if leg.symbol.upper() not in allowed_set:
                return False, f"Symbol {leg.symbol} not in allowed_symbols"

    # בדיקת כמות
    if max_quantity_per_leg is not None:
        for leg in request.legs:
            if abs(leg.quantity) > max_quantity_per_leg:
                return False, f"Quantity {leg.quantity} for {leg.symbol} exceeds max_quantity_per_leg"

    # Notional per leg — אין לנו מחיר חי כאן, זה Hook עתידי.
    # אפשר להכניס פה בדיקה נגד fair_value או last_price מהמערכת שלך.

    return True, None


# =========================
# IBOrderRouter
# =========================

class IBOrderRouter:
    """
    Router למסחר מול IBKR (מעל ib_insync).

    מאפיינים:
    ----------
    - יושב מעל get_ib_instance(...).
    - תומך ב-singleton או context מקומי (לפי design של ibkr_connection).
    - מרכז:
        • submit_pair_order(...)
        • close_pair_positions(...)
        • sync_positions_from_ib(...)
        • cancel_all_open_orders(...)
    """

    def __init__(
        self,
        *,
        settings: Optional[Mapping[str, Any]] = None,
        use_singleton: bool = True,
        profile: Optional[str] = None,
        readonly: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        settings : Mapping | None
            dict עם הגדרות (למשל app_ctx.settings.as_dict()).
        use_singleton : bool
            האם להשתמש ב-singleton מהמודול ibkr_connection.
        profile : str | None
            "paper" / "live" — אם None, נקבע לפי settings/ENV.
        readonly : bool
            אם True, מבקש חיבור readOnly (לקריאת דאטה בלבד).
        """
        self.settings = settings
        self.use_singleton = use_singleton
        self.profile = profile
        self.readonly = readonly

        self._classes = _import_trade_classes()

    # ---- Internal helpers ----

    def _get_ib(self) -> Optional[IBType]:
        """שולף IB instance דרך ibkr_connection."""
        ib = get_ib_instance(
            readonly=self.readonly,
            use_singleton=self.use_singleton,
            profile=self.profile,
            settings=self.settings,
        )
        if ib is None:
            logger.warning("IBOrderRouter: get_ib_instance returned None (no connection).")
        else:
            status = ib_connection_status(ib)
            logger.debug("IBOrderRouter: ib_connection_status = %r", status)
        return ib

    def _make_contract(self, leg: PairOrderLeg) -> Any:
        """בונה Contract לפי סוג רגל (בינתיים STK ו-FX)."""
        if not self._classes:
            raise RuntimeError("ib_insync trade classes are unavailable")

        if leg.sec_type.upper() == "STK":
            Stock = self._classes["Stock"]
            if leg.primary_exchange:
                return Stock(
                    leg.symbol,
                    leg.exchange,
                    leg.currency,
                    primaryExchange=leg.primary_exchange,
                )
            return Stock(leg.symbol, leg.exchange, leg.currency)

        if leg.sec_type.upper() == "FX":
            Forex = self._classes["Forex"]
            return Forex(f"{leg.symbol}{leg.currency}")

        # לעתיד: FUT / OPT / CFD וכו'.
        raise NotImplementedError(f"Unsupported sec_type={leg.sec_type}")

    def _make_order(self, leg: PairOrderLeg, req: PairOrderRequest) -> Any:
        """בונה Order (Market/Limit) לפי PairOrderRequest."""
        if not self._classes:
            raise RuntimeError("ib_insync trade classes are unavailable")

        action = leg.action.upper()
        qty = abs(leg.quantity)

        if req.order_type.upper() == "LMT":
            LimitOrder = self._classes["LimitOrder"]
            if req.limit_price is None:
                raise ValueError("limit_price must be set when order_type='LMT'")
            return LimitOrder(
                action=action,
                totalQuantity=qty,
                lmtPrice=float(req.limit_price),
                tif=req.time_in_force,
            )
        # ברירת מחדל: Market
        MarketOrder = self._classes["MarketOrder"]
        return MarketOrder(action=action, totalQuantity=qty, tif=req.time_in_force)

    # ---- Public methods ----

    def submit_pair_order(
        self,
        request: PairOrderRequest,
        *,
        risk_params: Optional[Dict[str, Any]] = None,
    ) -> PairOrderResult:
        """
        שולח Orders למסחר זוגי (רגל/יים) דרך IB.

        risk_params (אופציונלי) יכול להכיל:
          - max_notional_per_leg
          - max_quantity_per_leg
          - allowed_symbols

        מחזיר PairOrderResult עם הצלחה/שגיאה ומזהי Orders.
        """
        risk_params = risk_params or {}
        ok, reason = pre_trade_check(
            request,
            max_notional_per_leg=risk_params.get("max_notional_per_leg"),
            max_quantity_per_leg=risk_params.get("max_quantity_per_leg"),
            allowed_symbols=risk_params.get("allowed_symbols"),
        )
        if not ok:
            msg = f"pre_trade_check failed: {reason}"
            logger.warning("submit_pair_order(%s) blocked by risk_check: %s", request.pair_id, reason)
            return PairOrderResult(success=False, error=msg)

        if self.readonly:
            msg = "IBOrderRouter is in readonly mode — cannot submit orders."
            logger.warning(msg)
            return PairOrderResult(success=False, error=msg)

        ib = self._get_ib()
        if ib is None:
            return PairOrderResult(success=False, error="IB connection is unavailable")

        order_ids: List[int] = []
        details: Dict[str, Any] = {"pair_id": request.pair_id, "legs": []}

        for leg in request.legs:
            try:
                contract = self._make_contract(leg)
                order = self._make_order(leg, request)
                if request.account:
                    try:
                        order.account = request.account  # type: ignore[attr-defined]
                    except Exception:
                        logger.debug("Could not set order.account")

                logger.info(
                    "Placing order: pair=%s, symbol=%s, action=%s, qty=%.2f, type=%s, tif=%s",
                    request.pair_id,
                    leg.symbol,
                    leg.action,
                    leg.quantity,
                    request.order_type,
                    request.time_in_force,
                )

                trade = ib.placeOrder(contract, order)  # type: ignore[attr-defined]
                oid = getattr(order, "orderId", None)
                if oid is not None:
                    order_ids.append(int(oid))
                details["legs"].append(
                    {
                        "symbol": leg.symbol,
                        "action": leg.action,
                        "quantity": leg.quantity,
                        "orderId": oid,
                    }
                )
            except Exception as e:
                logger.exception("Error while placing order for leg %s: %s", leg, e)
                if not request.allow_partial:
                    return PairOrderResult(
                        success=False,
                        order_ids=order_ids,
                        error=f"Failed to place order for leg {leg.symbol}: {e}",
                        details=details,
                    )

        success = len(order_ids) > 0
        return PairOrderResult(
            success=success,
            order_ids=order_ids,
            error=None if success else "No orders placed",
            details=details,
        )

    def sync_positions_from_ib(self) -> List[Dict[str, Any]]:
        """
        סנכרון פוזיציות מ-IB → רשימת dictים.

        כל dict כולל:
          - account, symbol, secType, position, avgCost, currency
        """
        ib = self._get_ib()
        if ib is None:
            return []

        try:
            positions = ib.positions()  # type: ignore[attr-defined]
        except Exception as e:
            logger.exception("sync_positions_from_ib failed: %s", e)
            return []

        rows: List[Dict[str, Any]] = []
        for pos in positions:
            try:
                # pos: Position(account, contract, position, avgCost)
                contract = getattr(pos, "contract", None)
                rows.append(
                    {
                        "account": getattr(pos, "account", None),
                        "symbol": getattr(contract, "symbol", None),
                        "secType": getattr(contract, "secType", None),
                        "currency": getattr(contract, "currency", None),
                        "position": getattr(pos, "position", None),
                        "avgCost": getattr(pos, "avgCost", None),
                    }
                )
            except Exception:
                continue

        return rows

    def close_pair_positions(
        self,
        pair_id: str,
        sym_a: str,
        sym_b: str,
        *,
        account: Optional[str] = None,
        time_in_force: str = "DAY",
    ) -> PairOrderResult:
        """
        ניסיון "חיסול" פוזיציות לזוג מסוים (sym_a/sym_b) ע"י שליחת פקודות הפוכות
        עבור הפוזיציות הנוכחיות (אם קיימות).

        **חשוב**:
        ---------
        - זו פונקציה בסיסית, לא מחליפה Risk Engine מלא.
        - מניחה STK / SMART / USD (אפשר להרחיב).
        """
        positions = self.sync_positions_from_ib()
        if not positions:
            return PairOrderResult(success=True, order_ids=[], error=None, details={"note": "No positions"})

        legs: List[PairOrderLeg] = []
        for row in positions:
            sym = str(row.get("symbol") or "")
            if sym.upper() not in {sym_a.upper(), sym_b.upper()}:
                continue
            pos = float(row.get("position") or 0.0)
            if abs(pos) < 1e-8:
                continue
            action = "SELL" if pos > 0 else "BUY"
            legs.append(
                PairOrderLeg(
                    symbol=sym,
                    action=action,
                    quantity=abs(pos),
                    sec_type=str(row.get("secType") or "STK"),
                    currency=str(row.get("currency") or "USD"),
                )
            )

        if not legs:
            return PairOrderResult(
                success=True,
                order_ids=[],
                error=None,
                details={"note": "No positions to close for given pair"},
            )

        req = PairOrderRequest(
            pair_id=pair_id,
            legs=legs,
            order_type="MKT",
            time_in_force=time_in_force,
            account=account,
            allow_partial=True,
            tags={"action": "close_pair_positions"},
        )
        return self.submit_pair_order(req)

    def cancel_all_open_orders(self) -> bool:
        """
        ביטול כל ה-Open Orders ב-IB (אזהרה: משפיע על כל המערכת, לא רק על הזוג שלך).

        מיועד יותר למצבי חירום / Kill-switch.
        """
        ib = self._get_ib()
        if ib is None:
            return False
        try:
            logger.warning("Cancelling all open orders on IB (global).")
            ib.reqGlobalCancel()  # type: ignore[attr-defined]
            return True
        except Exception as e:
            logger.exception("cancel_all_open_orders failed: %s", e)
            return False
