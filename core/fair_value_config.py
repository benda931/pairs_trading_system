# -*- coding: utf-8 -*-
"""
core/fair_value_config.py — Fair Value API configuration (HF-grade)
===================================================================

תפקיד המודול
-------------
לתת מודל קונפיגורציה עשיר ל-Fair Value API (מקומי או חיצוני):

- שליטה האם ה-API מופעל בכלל (enabled/profile).
- כתובת בסיס (base_url) + api_key / auth.
- הגדרות רשת:
    * timeouts (connect/read/total)
    * retries + backoff
    * verify_tls
    * rate-limit מותאם.
- התנהגות לוגים:
    * log_requests / log_payloads / log_responses.
- יצירת kwargs מוכנים ל:
    * requests / httpx / כל client אחר.

שימוש טיפוסי
------------
1. מתוך ENV בלבד:

    >>> from core.fair_value_config import FairValueAPIConfig
    >>> cfg = FairValueAPIConfig.from_env()
    >>> if cfg.is_enabled:
    ...     do_request(cfg)

2. מתוך AppContext.settings (dict) + ENV fallback:

    >>> cfg = FairValueAPIConfig.from_settings(app_ctx.settings, env_prefix="FV_API_")

המודל נועד להיות "data only" — לא מבצע HTTP, רק מספק את כל
המידע הדרוש כדי להשתמש בו בצורה עקבית בכל המערכת.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import os

from pydantic import BaseModel, Field, validator


class FairValueAPIConfig(BaseModel):
    """
    קונפיגורציה מלאה ל-Fair Value API.

    שדות עיקריים:
    --------------
    enabled:
        דגל כללי — האם להשתמש ב-API בכלל.

    profile:
        "disabled" — לא להשתמש ב-API, נקודה.
        "local"    — שרת לוקאלי (למשל FastAPI ב-uvicorn).
        "remote"   — שירות חיצוני בענן.
        "mock"     — מצב בדיקות/סימולציה, בלי קריאת HTTP אמיתית.

    base_url:
        כתובת בסיס ל-API, למשל "http://localhost:8000" או "https://api.my-fv.com".

    api_key:
        מחרוזת API key (אם יש). אפשר לשים ריק אם אין צורך.

    headers:
        כותרות ברירת מחדל שנשלחות בכל בקשה (למשל Authorization / X-Client-ID).

    timeouts:
        connect_timeout_sec, read_timeout_sec, total_timeout_sec.

    retry:
        max_retries, backoff_factor (במקרה של כשלי רשת זמניים).

    tls:
        verify_tls — האם לוודא תעודת TLS (מומלץ True ל-remote).

    limits:
        max_concurrent_requests — כמה בקשות במקביל מותרות ל-client הזה.
        max_requests_per_minute — קירוב פשוט ל-rate-limit פר קליינט.

    logging:
        log_requests   — לוג לשורת בקשה (method,url,status).
        log_payloads   — לוג מלא לגוף request/response (זהיר בפרודקשן).
        log_failures   — לוג שגיאות בלבד.
    """

    # מצב כללי
    enabled: bool = Field(
        False,
        description="האם להשתמש ב-Fair Value API בכלל."
    )
    profile: str = Field(
        "disabled",
        description="פרופיל: disabled | local | remote | mock",
    )

    # פרטי חיבור בסיסיים
    base_url: str = Field(
        "",
        description="Base URL של ה-Fair Value API (למשל 'http://localhost:8000').",
    )
    api_key: str = Field(
        "",
        description="API key אם קיים (אופציונלי).",
    )

    # HTTP headers
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="כותרות ברירת מחדל לבקשות HTTP.",
    )

    # Timeouts (בשניות)
    connect_timeout_sec: float = Field(
        5.0,
        ge=0.1,
        description="Timeout לחיבור (connect).",
    )
    read_timeout_sec: float = Field(
        30.0,
        ge=0.1,
        description="Timeout לקריאה (read).",
    )
    total_timeout_sec: float = Field(
        60.0,
        ge=0.1,
        description="Timeout כולל לבקשה (connect+read).",
    )

    # Retries
    max_retries: int = Field(
        3,
        ge=0,
        description="כמה ניסיונות חוזרים על כשלי רשת זמניים.",
    )
    backoff_factor: float = Field(
        0.3,
        ge=0.0,
        description="מקדם backoff בין retries (למשל ב-requests.adapters.HTTPAdapter).",
    )

    # TLS / אבטחה
    verify_tls: bool = Field(
        True,
        description="האם לוודא תעודת TLS (מומלץ True ל-remote).",
    )

    # Rate limiting / concurrency
    max_concurrent_requests: int = Field(
        8,
        ge=1,
        description="מכסימום בקשות שמותר לשלוח במקביל (per client).",
    )
    max_requests_per_minute: int = Field(
        300,
        ge=1,
        description="Rate-limit מקורב (לא נאכף כאן, רק מידע למימושים).",
    )

    # Logging flags
    log_requests: bool = Field(
        True,
        description="האם לוג שורה לכל בקשה (method, url, status).",
    )
    log_payloads: bool = Field(
        False,
        description="האם ללוג גוף מלא של request/response (זהיר בפרודקשן).",
    )
    log_failures: bool = Field(
        True,
        description="האם ללוג רק כשלי רשת/שגיאות.",
    )

    class Config:
        # מאפשר שימוש ב-BaseModel עם הגדרות קצת יותר גמישות
        validate_assignment = True

    # ------------------------------------------------------------------
    # Properties נוחים
    # ------------------------------------------------------------------

    @property
    def is_configured(self) -> bool:
        """
        האם ה-config מוגדר בצורה סבירה:
        - enabled == True
        - base_url לא ריק
        """
        return bool(self.enabled and self.base_url)

    @property
    def is_enabled(self) -> bool:
        """
        alias ל-is_configured עבור קריאות קצרות.
        """
        return self.is_configured

    @property
    def is_local(self) -> bool:
        return self.profile == "local"

    @property
    def is_remote(self) -> bool:
        return self.profile == "remote"

    @property
    def is_mock(self) -> bool:
        return self.profile == "mock"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @validator("profile")
    def _validate_profile(cls, v: str) -> str:
        v = (v or "").lower()
        allowed = {"disabled", "local", "remote", "mock"}
        if v not in allowed:
            raise ValueError(f"profile must be one of {allowed}, got {v!r}")
        return v

    @validator("base_url")
    def _strip_base_url(cls, v: str) -> str:
        v = (v or "").strip()
        # מסירים '/', אבל לא אם זו רק הסלאש
        if v.endswith("/") and len(v) > 1:
            v = v[:-1]
        return v

    @validator("total_timeout_sec")
    def _validate_total_timeout(cls, v: float, values: Dict[str, Any]) -> float:
        """
        total_timeout חייב להיות לפחות כמו connect_timeout + read_timeout בערך גס.
        אם לא — נעלה אותו מעט.
        """
        conn = float(values.get("connect_timeout_sec", 0.0))
        read = float(values.get("read_timeout_sec", 0.0))
        min_total = max(conn + read, 0.1)
        if v < min_total:
            # לא נזרוק שגיאה, רק נעדכן כלפי מעלה בעדינות
            return min_total
        return v

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, prefix: str = "FV_API_") -> "FairValueAPIConfig":
        """
        טעינת קונפיגורציה מ־ENV בלבד.

        ENV variables (ברירת מחדל, ניתן לשנות prefix):
        ------------------------------------------------
        - FV_API_ENABLED          ("1"/"true"/"yes" → True)
        - FV_API_PROFILE          ("disabled"/"local"/"remote"/"mock")
        - FV_API_KEY              (api_key)
        - FV_API_BASE_URL         (base_url)
        - FV_API_CONNECT_TIMEOUT  (float)
        - FV_API_READ_TIMEOUT     (float)
        - FV_API_TOTAL_TIMEOUT    (float)
        - FV_API_MAX_RETRIES      (int)
        - FV_API_BACKOFF_FACTOR   (float)
        - FV_API_VERIFY_TLS       ("1"/"true"/"yes")
        - FV_API_MAX_CONCURRENT   (int)
        - FV_API_MAX_PER_MINUTE   (int)
        - FV_API_LOG_REQUESTS     ("1"/"true"/"yes")
        - FV_API_LOG_PAYLOADS     ("1"/"true"/"yes")
        - FV_API_LOG_FAILURES     ("1"/"true"/"yes")
        """
        def _b(name: str, default: bool) -> bool:
            val = os.getenv(prefix + name, "")
            if not val:
                return default
            return val.strip().lower() in {"1", "true", "yes", "on"}

        def _i(name: str, default: int) -> int:
            val = os.getenv(prefix + name, "")
            try:
                return int(val)
            except Exception:
                return default

        def _f(name: str, default: float) -> float:
            val = os.getenv(prefix + name, "")
            try:
                return float(val)
            except Exception:
                return default

        enabled = _b("ENABLED", False)
        profile = os.getenv(prefix + "PROFILE", "disabled").strip().lower() or "disabled"
        api_key = os.getenv(prefix + "KEY", "") or ""
        base_url = os.getenv(prefix + "BASE_URL", "") or ""

        connect_timeout = _f("CONNECT_TIMEOUT", 5.0)
        read_timeout = _f("READ_TIMEOUT", 30.0)
        total_timeout = _f("TOTAL_TIMEOUT", 60.0)
        max_retries = _i("MAX_RETRIES", 3)
        backoff = _f("BACKOFF_FACTOR", 0.3)
        verify_tls = _b("VERIFY_TLS", True)
        max_conc = _i("MAX_CONCURRENT", 8)
        max_per_min = _i("MAX_PER_MINUTE", 300)

        log_requests = _b("LOG_REQUESTS", True)
        log_payloads = _b("LOG_PAYLOADS", False)
        log_failures = _b("LOG_FAILURES", True)

        # Headers בסיסיים (אם יש API key נכניס Authorization)
        headers: Dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        return cls(
            enabled=enabled,
            profile=profile,
            api_key=api_key,
            base_url=base_url,
            headers=headers,
            connect_timeout_sec=connect_timeout,
            read_timeout_sec=read_timeout,
            total_timeout_sec=total_timeout,
            max_retries=max_retries,
            backoff_factor=backoff,
            verify_tls=verify_tls,
            max_concurrent_requests=max_conc,
            max_requests_per_minute=max_per_min,
            log_requests=log_requests,
            log_payloads=log_payloads,
            log_failures=log_failures,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        *,
        key: str = "fair_value_api",
        env_prefix: str = "FV_API_",
    ) -> "FairValueAPIConfig":
        """
        בנייה מתוך settings (למשל AppContext.settings) + fallback ל-ENV.

        מבנה מצופה ב-settings (לדוגמה):
        --------------------------------
        settings["fair_value_api"] = {
            "enabled": true,
            "profile": "local",
            "base_url": "http://localhost:8000",
            "api_key": "xxxxx",
            "connect_timeout_sec": 3.0,
            "read_timeout_sec": 20.0,
            "total_timeout_sec": 30.0,
            "max_retries": 2,
            "verify_tls": false,
            "log_requests": true,
            "log_payloads": false,
        }

        הערכים מה-settings ינצחו ערכי ENV במקרה של התנגשות.
        """
        base_cfg = cls.from_env(prefix=env_prefix)
        raw = settings.get(key, {}) or {}
        if not isinstance(raw, Mapping):
            # אם המבנה לא נכון — פשוט נחזור ל-base_cfg
            return base_cfg

        # merge: base_env ← settings override
        data = base_cfg.dict()
        for k, v in raw.items():
            if k in data:
                data[k] = v

        try:
            return cls(**data)
        except Exception:
            # אם משהו לא תקין בקונפיג — נעדיף לחזור ל-ENV בלבד
            return base_cfg

    # ------------------------------------------------------------------
    # Helpers ל-clients (requests/httpx)
    # ------------------------------------------------------------------

    def as_requests_kwargs(self) -> Dict[str, Any]:
        """
        מחזיר kwargs טיפוסיים ל-requests (או wrap דומה):

        {
          "timeout": total_timeout_sec,
          "headers": headers,
          "verify": verify_tls,
        }

        את ניהול ה-retries/adapter/RateLimit נשאיר לשכבה שמעל.
        """
        timeout = self.total_timeout_sec or max(
            self.connect_timeout_sec + self.read_timeout_sec,
            0.1,
        )
        return {
            "timeout": timeout,
            "headers": dict(self.headers),
            "verify": self.verify_tls,
        }

    def as_httpx_kwargs(self) -> Dict[str, Any]:
        """
        מחזיר kwargs טיפוסיים ל-httpx.AsyncClient או httpx.Client:

        {
          "base_url": base_url,
          "headers": headers,
          "timeout": {
              "connect": connect_timeout_sec,
              "read": read_timeout_sec,
              "write": read_timeout_sec,
              "pool": total_timeout_sec,
          },
          "verify": verify_tls,
        }
        """
        return {
            "base_url": self.base_url,
            "headers": dict(self.headers),
            "timeout": {
                "connect": self.connect_timeout_sec,
                "read": self.read_timeout_sec,
                "write": self.read_timeout_sec,
                "pool": self.total_timeout_sec,
            },
            "verify": self.verify_tls,
        }
