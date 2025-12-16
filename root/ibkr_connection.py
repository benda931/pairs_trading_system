# -*- coding: utf-8 -*-
"""
root/ibkr_connection.py — IBKR connection helper (Streamlit-safe, HF-grade v4)
------------------------------------------------------------------------------

תפקיד הקובץ:
============

1. לולאת asyncio & ib_insync:
   - לדאוג שתהיה לולאת asyncio תקינה לפני שימוש ב-ib_insync / eventkit.
   - להיערך לסביבות שונות:
       * Streamlit (ScriptRunner.scriptThread)
       * Jupyter / IPython
       * סקריפטים רגילים / CLI
   - לייבא את IB מתוך ib_insync בצורה עצלה ובטוחה.

2. חיבור ל-IBKR ברמת "קרן גידור":
   - get_ib_instance(...)
       * חיבור עם ברירות מחדל מ-ENV (IB_HOST, IB_PORT, IB_CLIENT_ID, IB_MODE).
       * תמיכה בפרופיל: "paper" / "live".
       * תמיכה ב-singleton (משותף למודול).
       * readonly=True למצב קריאה בלבד (לניתוח, ללא פקודות).
   - disconnect_ib(...)
       * ניתוק בטוח מחיבור קיים.
   - ib_connection_status(...)
       * סטטוס חיבור בסיסי + נתונים שימושיים לדשבורד.

3. הרחבות (HF-grade helpers):
   - configure_ib_from_settings(settings: Mapping) → Dict[str, Any]
       * חילוץ host/port/client_id/profile מתוך dict קונפיגורציה (config.json / AppContext).
   - reset_ib_singleton() → None
       * ניקוי singleton ברמת המודול (למשל בשינוי פרופיל/סביבה).
   - IBConnectionManager (context manager)
       * שימוש "זמני" ב-IB ללא singleton, כולל ניתוק אוטומטי.
   - ib_health_check(ib: IBType) → Dict[str, Any]
       * בדיקת בריאות בסיסית (מחזיר info: זמן שרת, פרופיל, isConnected).

שילוב עם Streamlit:
====================
- Streamlit מריץ קוד בתוך thread "ScriptRunner.scriptThread" ללא event loop ברירת מחדל.
- eventkit (שבשימוש ib_insync) מנסה לקרוא event loop בזמן import.
- לכן, לפני import של ib_insync אנחנו דואגים שיהיה loop:
    _ensure_event_loop()

הקובץ "שקט" אם ib_insync לא מותקן:
- כל הפונקציות יחזירו None או סטטוס ידידותי,
- והדשבורד יכול להציג הודעה, אבל המערכת לא תתרסק.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, Dict, Mapping

if TYPE_CHECKING:
    # רק בזמן type-checking נייבא את הטיפוס האמיתי
    from ib_insync import IB as IBType  # pragma: no cover
else:
    IBType = Any  # בזמן ריצה זה יהיה Any – לא שוברים כלום אם ib_insync לא מותקן


logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [ibkr_connection] %(message)s")
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ===================== Event Loop Helpers =====================

def _loop_debug_context() -> str:
    """מחזיר מידע קצר על מצב ה-loop לצורך לוגים / debug."""
    try:
        running_loop = asyncio.get_running_loop()
        running = True
        loop_id = id(running_loop)
    except RuntimeError:
        running = False
        loop_id = None
    return f"running={running}, loop_id={loop_id}, policy={type(asyncio.get_event_loop_policy()).__name__}"


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    """Ensure the current thread has a valid asyncio event loop.

    חשוב במיוחד כשמייבאים ib_insync/eventkit מתוך Streamlit / Jupyter,
    אחרת מתקבלת השגיאה:
        RuntimeError: There is no current event loop in thread 'ScriptRunner.scriptThread'
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Existing event loop is closed")
        return loop
    except RuntimeError:
        # אין loop פעיל ל-thread הנוכחי – ניצור חדש
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Created new asyncio event loop for IBKR integration (%s)", _loop_debug_context())
        return loop


# ===================== ib_insync Import (lazy-safe) =====================

_IB_CLASS: Optional[Any] = None
_IB_IMPORT_ERROR: Optional[BaseException] = None


def _import_ib_class() -> Optional[Any]:
    """ייבוא עצלן של IB מתוך ib_insync, עם טיפול ב-event loop ולוגים.

    מחזיר את המחלקה IB אם הצליח, אחרת None.
    """
    global _IB_CLASS, _IB_IMPORT_ERROR

    if _IB_CLASS is not None:
        return _IB_CLASS
    if _IB_IMPORT_ERROR is not None:
        # כבר ניסינו ונכשלנו – לא ננסה כל פעם מחדש
        return None

    try:
        _ensure_event_loop()
        from ib_insync import IB as _IB  # type: ignore
        _IB_CLASS = _IB
        logger.info("ib_insync imported successfully")
        return _IB_CLASS
    except ImportError as e:
        _IB_IMPORT_ERROR = e
        logger.warning("ib_insync is not installed. IBKR connection will be unavailable.")
    except Exception as e:  # catch eventkit errors וכו'
        _IB_IMPORT_ERROR = e
        logger.exception("Failed to import ib_insync: %s", e)
    return None


# ===================== Settings / Profiles Helpers =====================

@dataclass
class IBConnectionSettings:
    """תצורת חיבור IBKR (ללא תלות ב-ib_insync עצמו)."""

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    profile: str = "paper"  # "paper" / "live"
    readonly: bool = False

    @classmethod
    def from_env(cls) -> "IBConnectionSettings":
        """בניית תצורה מ-ENV (עם ברירות מחדל)."""
        mode = os.getenv("IB_MODE", "paper").lower()
        host = os.getenv("IB_HOST", "127.0.0.1")

        def _get_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except Exception:
                return default

        if mode in {"live", "prod", "production"}:
            default_port = 7496
            profile = "live"
        else:
            default_port = 7497
            profile = "paper"

        port = _get_int("IB_PORT", default_port)
        client_id = _get_int("IB_CLIENT_ID", 1)

        return cls(
            host=host,
            port=port,
            client_id=client_id,
            profile=profile,
            readonly=False,
        )


def _get_env_default_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def configure_ib_from_settings(settings: Mapping[str, Any]) -> IBConnectionSettings:
    """
    גוזר הגדרות חיבור IB מתוך dict כללי (למשל config.json / AppContext.settings).

    שמות אפשריים:
      - "ib_host" / "IB_HOST"
      - "ib_port" / "IB_PORT"
      - "ib_client_id" / "IB_CLIENT_ID"
      - "ib_profile" / "IB_MODE"   ("paper" / "live")
      - "ib_readonly"

    אם לא נמצא ערך מתאים, נ fallback ל-IBConnectionSettings.from_env().
    """
    base = IBConnectionSettings.from_env()

    def _get(key1: str, key2: str | None = None, default: Any | None = None) -> Any:
        if key1 in settings:
            return settings[key1]
        if key2 is not None and key2 in settings:
            return settings[key2]
        return default

    host = _get("ib_host", "IB_HOST", base.host) or base.host
    port = _get("ib_port", "IB_PORT", base.port) or base.port
    client_id = _get("ib_client_id", "IB_CLIENT_ID", base.client_id) or base.client_id
    profile = _get("ib_profile", "IB_MODE", base.profile) or base.profile
    readonly = bool(_get("ib_readonly", None, False))

    try:
        port = int(port)
    except Exception:
        port = base.port

    try:
        client_id = int(client_id)
    except Exception:
        client_id = base.client_id

    profile_str = str(profile).lower()
    if profile_str in {"live", "prod", "production"}:
        profile_str = "live"
    else:
        profile_str = "paper"

    return IBConnectionSettings(
        host=str(host),
        port=int(port),
        client_id=int(client_id),
        profile=profile_str,
        readonly=readonly,
    )


# ===================== Singleton storage (optional) =====================

_IB_SINGLETON: Optional[IBType] = None  # type: ignore[assignment]


def reset_ib_singleton() -> None:
    """מאפס את ה-singleton (למשל כאשר מחליפים פרופיל/סביבה)."""
    global _IB_SINGLETON
    try:
        if _IB_SINGLETON is not None and getattr(_IB_SINGLETON, "isConnected", None):
            if _IB_SINGLETON.isConnected():  # type: ignore[attr-defined]
                logger.info("reset_ib_singleton: disconnecting existing singleton IB instance")
                _IB_SINGLETON.disconnect()  # type: ignore[attr-defined]
    except Exception:
        logger.debug("reset_ib_singleton: error while disconnecting existing singleton", exc_info=True)
    _IB_SINGLETON = None


# ===================== Public API =====================

def get_ib_instance(
    host: Optional[str] = None,
    port: Optional[int] = None,
    client_id: Optional[int] = None,
    *,
    readonly: bool = False,
    use_singleton: bool = True,
    profile: Optional[str] = None,
    settings: Optional[Mapping[str, Any]] = None,
) -> Optional[IBType]:
    """Create and connect an IB instance.

    Parameters
    ----------
    host : str | None
        IB Gateway / TWS host (default from env IB_HOST or "127.0.0.1").
    port : int | None
        IB Gateway / TWS port (default from env IB_PORT or 7497; 7496 ל-live).
    client_id : int | None
        IB client id (default from env IB_CLIENT_ID or 1 – צריך להיות ייחודי לכל חיבור).
    readonly : bool
        אם True – מפעיל ib.readOnly = True (לשימוש רק לקריאת דאטה).
    use_singleton : bool
        אם True – משתמש ב-instance יחיד ברמת המודול (שימושי ל-Streamlit).
    profile : str | None
        "paper" / "live" – אם מועבר, גובר על IB_MODE / ib_profile ב-settings.
    settings : Mapping[str, Any] | None
        dict עם הגדרות אפליקציה/קונפיג אשר מכיל שדות ib_host/ib_port/... (לא חובה).

    Returns
    -------
    IB | None
        אובייקט IB מחובר, או None אם ib_insync לא מותקן / החיבור נכשל.
    """
    global _IB_SINGLETON

    IBClass = _import_ib_class()
    if IBClass is None:
        logger.warning("get_ib_instance called but ib_insync is unavailable.")
        return None

    # --- Resolve settings ---
    if settings is not None:
        conn_settings = configure_ib_from_settings(settings)
    else:
        conn_settings = IBConnectionSettings.from_env()

    # override from explicit params
    if host is not None:
        conn_settings.host = host
    if port is not None:
        conn_settings.port = int(port)
    if client_id is not None:
        conn_settings.client_id = int(client_id)
    if profile is not None:
        conn_settings.profile = str(profile).lower()

    # override readonly
    if readonly:
        conn_settings.readonly = True

    # env-dependent default port (אם המשתמש לא כפה)
    if profile is not None:
        if conn_settings.profile == "live" and "IB_PORT" not in os.environ and port is None:
            conn_settings.port = 7496
        elif conn_settings.profile == "paper" and "IB_PORT" not in os.environ and port is None:
            conn_settings.port = 7497

    # ודא שיש loop פעיל לפני connect
    _ensure_event_loop()

    # אם singleton מופעל ויש לנו instance מחובר – החזר אותו
    try:
        if use_singleton and _IB_SINGLETON is not None and _IB_SINGLETON.isConnected():  # type: ignore[attr-defined]
            return _IB_SINGLETON  # type: ignore[return-value]
    except Exception:
        _IB_SINGLETON = None

    # ניצור instance חדש
    ib: IBType = IBClass()  # type: ignore[call-arg]
    try:
        logger.info(
            "Connecting to IBKR at %s:%s (client_id=%s, profile=%s, readonly=%s)",
            conn_settings.host,
            conn_settings.port,
            conn_settings.client_id,
            conn_settings.profile,
            conn_settings.readonly,
        )
        ib.connect(conn_settings.host, conn_settings.port, clientId=conn_settings.client_id)  # type: ignore[attr-defined]
        if conn_settings.readonly:
            try:
                ib.readOnly = True  # type: ignore[attr-defined]
            except Exception:
                pass
        if not ib.isConnected():  # type: ignore[attr-defined]
            logger.warning("IBKR connection attempt finished but isConnected() is False")
            try:
                ib.disconnect()  # type: ignore[attr-defined]
            except Exception:
                pass
            return None
        logger.info("IBKR connection established successfully.")

        # ננסה לשמור מידע על הפרופיל בתוך האובייקט (לא חובה, אבל נוח לבדיקות)
        try:
            setattr(ib, "_hf_profile", conn_settings.profile)
        except Exception:
            pass

        if use_singleton:
            _IB_SINGLETON = ib
        return ib
    except Exception as e:
        logger.exception("Failed to connect to IBKR: %s", e)
        try:
            ib.disconnect()  # type: ignore[attr-defined]
        except Exception:
            pass
        return None


def disconnect_ib(ib: Optional[Any]) -> None:
    """ניתוק חיבור IB בצורה בטוחה (אם קיים)."""
    if ib is None:
        return
    try:
        if getattr(ib, "isConnected", None) and ib.isConnected():  # type: ignore[attr-defined]
            logger.info("Disconnecting IBKR instance")
            ib.disconnect()  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Error while disconnecting IBKR: %s", e)


def ib_connection_status(ib: Optional[Any]) -> Dict[str, Any]:
    """החזרת סטטוס חיבור IBKR כמילון – לשימוש בדשבורד / לוגים.

    מחזיר מידע בסיסי:
      - connected: bool
      - note: str (אם לא מחובר / אין ib_insync)
      - client_id / host / port (אם זמין)
      - profile (אם נשמר ב-ib._hf_profile)
    """
    if ib is None:
        return {"connected": False, "note": "No IB instance"}

    try:
        connected = bool(getattr(ib, "isConnected", lambda: False)())  # type: ignore[attr-defined]
    except Exception:
        connected = False

    status: Dict[str, Any] = {"connected": connected}
    try:
        if connected and hasattr(ib, "clientId"):
            status["client_id"] = getattr(ib, "clientId", None)
        if connected and hasattr(ib, "host") and hasattr(ib, "port"):
            status["host"] = getattr(ib, "host", None)
            status["port"] = getattr(ib, "port", None)
        # פרופיל לוגי (paper/live) אם יש
        if hasattr(ib, "_hf_profile"):
            status["profile"] = getattr(ib, "_hf_profile", None)
    except Exception:
        pass
    return status


# ===================== Health Check & Context Manager =====================

def ib_health_check(ib: Optional[Any]) -> Dict[str, Any]:
    """
    בדיקת בריאות בסיסית של חיבור IBKR.

    מחזיר:
      - connected: bool
      - server_time: str | None
      - profile: str | None
      - error: str | None
    """
    info: Dict[str, Any] = {
        "connected": False,
        "server_time": None,
        "profile": None,
        "error": None,
    }
    if ib is None:
        info["error"] = "No IB instance"
        return info

    try:
        connected = bool(getattr(ib, "isConnected", lambda: False)())  # type: ignore[attr-defined]
        info["connected"] = connected
        if hasattr(ib, "_hf_profile"):
            info["profile"] = getattr(ib, "_hf_profile", None)

        if not connected:
            info["error"] = "IB not connected"
            return info

        # ping קל: reqCurrentTime
        try:
            current_time = getattr(ib, "reqCurrentTime", lambda: None)()  # type: ignore[attr-defined]
            info["server_time"] = str(current_time) if current_time is not None else None
        except Exception as e:
            info["error"] = f"reqCurrentTime failed: {e}"
    except Exception as e:
        info["error"] = str(e)
    return info


class IBConnectionManager:
    """
    Context manager לחיבור IB זמני (ללא singleton), ברמת "קרן גידור":

        with IBConnectionManager(profile="paper", readonly=True) as ib:
            if ib is not None:
                # ... עבודה עם IB ...

    - לא תלוי ב-singleton (use_singleton=False).
    - מחבר ב-__enter__ ומנתק ב-__exit__ (אם התחבר).
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        *,
        readonly: bool = False,
        profile: Optional[str] = None,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self._readonly = readonly
        self._profile = profile
        self._settings = settings
        self._ib: Optional[IBType] = None

    def __enter__(self) -> Optional[IBType]:  # type: ignore[override]
        self._ib = get_ib_instance(
            host=self._host,
            port=self._port,
            client_id=self._client_id,
            readonly=self._readonly,
            use_singleton=False,
            profile=self._profile,
            settings=self._settings,
        )
        return self._ib

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        if self._ib is not None:
            try:
                disconnect_ib(self._ib)
            finally:
                self._ib = None


__all__ = [
    "get_ib_instance",
    "disconnect_ib",
    "ib_connection_status",
    "ib_health_check",
    "IBConnectionManager",
    "configure_ib_from_settings",
    "reset_ib_singleton",
]
