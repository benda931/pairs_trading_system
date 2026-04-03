# -*- coding: utf-8 -*-
"""
root/dashboard_url_router.py — URL query params bridge
========================================================

Extracted from dashboard.py Part 17/35.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

# Import deps from dashboard at call time to avoid circular imports
DashboardRuntime = Any
FeatureFlags = Dict[str, Any]
AppContext = Any
TabKey = str
EnvName = str
ProfileName = str
NavPayload = Optional[Dict[str, Any]]

# Part 17/35 – URL query params bridge (env/profile + deep-link navigation)
# =====================

# הרעיון:
# --------
# מאפשר:
#   1. שליטה על env/profile דרך ה-URL:
#        ?env=live&profile=trading
#   2. Deep-link לטאבים עם payload:
#        ?tab=backtest&pair=AAPL/MSFT&preset=smoke&mode=wf
#   3. ניווט ראשוני "חכם" – query params → nav_target → Router.
#
# זה קריטי כדי:
#   - לשתף לינקים מדויקים (למשל למקרו/ריסק/Backtest מסוים).
#   - לאפשר אינטגרציה עם Desktop/Agents שמייצרים URLs לטאבים ספציפיים.

SESSION_KEY_QUERY_INTENT: str = "_dashboard_query_intent"


def _get_query_params() -> Mapping[str, List[str]]:
    """
    עטיפה בטוחה סביב st.query_params (API החדש).

    מחזירה:
        Mapping[str, List[str]]

    הערות:
    -------
    - st.query_params הוא אובייקט דמוי-dict.
    - כדי לשמור תאימות לחתימה הישנה (dict של key → List[str]),
      נשתמש ב-.to_dict() אם קיים, או נמיר ידנית.
    """
    try:
        qp = st.query_params  # QueryParamsProxy / dict-like

        # API הרשמי – מחזיר dict[str, List[str]]
        to_dict = getattr(qp, "to_dict", None)
        if callable(to_dict):
            data = to_dict()
            if isinstance(data, Mapping):
                return data  # type: ignore[return-value]

        # fallback: qp כבר dict / mapping, אבל הערכים יכולים להיות str ולא list
        if isinstance(qp, Mapping):
            out: Dict[str, List[str]] = {}
            for k, v in qp.items():
                # st.query_params מחזיר או str אחד או list של str
                if isinstance(v, list):
                    out[str(k)] = [str(x) for x in v]
                elif v is None:
                    out[str(k)] = []
                else:
                    out[str(k)] = [str(v)]
            return out

        return {}
    except Exception as exc:  # pragma: no cover
        logger.debug("st.query_params failed in _get_query_params: %s", exc)
        return {}


def _first_query_value(
    params: Mapping[str, Sequence[str]],
    key: str,
) -> Optional[str]:
    """
    מחזיר את הערך הראשון של פרמטר query נתון, אם קיים ולא ריק.

    לדוגמה:
        params = {"env": ["live"], "pair": ["AAPL/MSFT"]}
        _first_query_value(params, "env") → "live"
    """
    try:
        values = params.get(key)
    except Exception:
        return None

    if not values:
        return None

    try:
        first = values[0]
    except Exception:
        return None

    if first is None:
        return None

    s = str(first).strip()
    return s or None


def parse_dashboard_query_params() -> Dict[str, Any]:
    """
    מפרש את query params של הדשבורד למבנה נוח:

    תומך בפרמטרים:
    ---------------
    - env:        env נדרש (dev/live/paper/research/backtest/...)
    - profile:    profile נדרש (trading/research/risk/macro/monitoring)
    - tab:        טאב יעד (home/backtest/risk/macro/...)
    - pair:       זוג (למשל "AAPL/MSFT")
    - preset:     preset עבור Backtest/Scan (למשל "smoke")
    - mode:       מצב Backtest (למשל "wf", "single")
    - portfolio:  מזהה פורטפוליו ("default", "fund_core" וכו')
    - view:       תת-תצוגה (למשל "limits", "overview")
    - macro_view: תצוגת מקרו (למשל "regimes", "indicators")

    הפלט:
    -----
    dict עם מפתחות:
        "env", "profile", "tab", "pair", "preset", "mode",
        "portfolio_id", "view", "macro_view"
    """
    params = _get_query_params()

    env_raw = _first_query_value(params, "env")
    profile_raw = _first_query_value(params, "profile")
    tab_raw = _first_query_value(params, "tab")

    pair_raw = _first_query_value(params, "pair")
    preset_raw = _first_query_value(params, "preset")
    mode_raw = _first_query_value(params, "mode")
    portfolio_raw = _first_query_value(params, "portfolio")
    view_raw = _first_query_value(params, "view")
    macro_view_raw = _first_query_value(params, "macro_view")

    parsed: Dict[str, Any] = {
        "env": env_raw,
        "profile": profile_raw,
        "tab": tab_raw,
        "pair": pair_raw,
        "preset": preset_raw,
        "mode": mode_raw,
        "portfolio_id": portfolio_raw,
        "view": view_raw,
        "macro_view": macro_view_raw,
    }

    # אם אין אף פרמטר משמעותי – נחזיר dict ריק כדי שלא נדרוס כלום.
    if not any(parsed.values()):
        return {}

    logger.info("Dashboard query params parsed: %s", parsed)
    return parsed


def _apply_query_env_profile_overrides(parsed: Mapping[str, Any]) -> None:
    """
    מיישם env/profile מה-URL אל session_state לפני יצירת ה-Runtime.

    אסטרטגיה:
    ----------
    - אם parsed["env"] קיים → מנרמל לערך EnvName ומשתמש בו כ-session_state["env"].
    - אם parsed["profile"] קיים → מנרמל לערך ProfileName ומשתמש בו כ-session_state["profile"].

    בכך, detect_env_profile(app_ctx) יראה את הערכים האלה בעדיפות ראשונה.
    """
    raw_env = parsed.get("env")
    raw_profile = parsed.get("profile")

    if raw_env:
        env_norm = _normalize_env(str(raw_env))
        try:
            st.session_state[SESSION_KEY_ENV] = env_norm
            st.session_state["env"] = env_norm
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to apply query env override to session_state: %s", exc)
        else:
            logger.info("Applied env override from query params: env=%s", env_norm)

    if raw_profile:
        profile_norm = _normalize_profile(str(raw_profile))
        try:
            st.session_state[SESSION_KEY_PROFILE] = profile_norm
            st.session_state["profile"] = profile_norm
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Failed to apply query profile override to session_state: %s", exc
            )
        else:
            logger.info(
                "Applied profile override from query params: profile=%s", profile_norm
            )


def _build_query_nav_payload(parsed: Mapping[str, Any]) -> Optional[NavPayload]:
    """
    בונה NavPayload מתוך הפרמטרים שקיבלנו מה-URL.

    לוגיקה:
    -------
    - אם אין אף שדה "עסקי" (pair/preset/mode/portfolio_id/view/macro_view) → מחזיר None.
    - אחרת → מחזיר dict עם המפתחות הלא-ריקים.
    """
    keys = ("pair", "preset", "mode", "portfolio_id", "view", "macro_view")
    payload: NavPayload = {}

    for key in keys:
        val = parsed.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if not s:
            continue
        payload[key] = s

    if not payload:
        return None

    return payload


def _store_query_nav_intent(parsed: Mapping[str, Any]) -> None:
    """
    שומר Intent לניווט ראשוני מתוך ה-URL ב-session_state:

    מבנה:
        SESSION_KEY_QUERY_INTENT = {
            "tab_key": "<tab_key>",
            "payload": {...} or None,
            "applied": bool  # האם כבר יוצר nav_target בפועל
        }

    חשוב:
    -----
    * איננו מתקשרים כאן ישירות עם set_nav_target, משום שעדיין אין לנו
      Runtime/TabRegistry – יתכן שהטאב לא זמין בפרופיל הנוכחי.
    * היישום בפועל ל-nav_target יתרחש אחרי יצירת DashboardRuntime.
    """
    tab_raw = parsed.get("tab")
    if not tab_raw:
        return

    tab_key = str(tab_raw).strip()
    if not tab_key:
        return

    payload = _build_query_nav_payload(parsed)
    intent = {
        "tab_key": tab_key,
        "payload": payload,
        "applied": False,
    }

    try:
        st.session_state[SESSION_KEY_QUERY_INTENT] = intent
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to store query nav intent in session_state: %s", exc)
        return

    logger.info(
        "Stored query nav intent: tab_key=%s, payload_keys=%s",
        tab_key,
        list(payload.keys()) if isinstance(payload, Mapping) else None,
    )


def apply_query_params_pre_runtime(app_ctx: "AppContext") -> Dict[str, Any]:
    """
    נקודת הכניסה הראשית לעיבוד Query Params לפני יצירת ה-Runtime:

    Flow:
    -----
    1. parse_dashboard_query_params() – פענוח מלא.
    2. אם אין פרמטרים רלוונטיים → החזרה {} (ולא נעשה כלום).
    3. יישום env/profile (אם קיימים) אל session_state.
    4. שמירת Intent ל-nav_target (אם tab הוגדר ב-URL).

    החזרה:
    ------
    dict parsed – כדי שאם תרצה בעתיד להשתמש במידע הזה ישירות
    (למשל להצגה ב-UI), הוא יהיה זמין.
    """
    parsed = parse_dashboard_query_params()
    if not parsed:
        return {}

    _apply_query_env_profile_overrides(parsed)
    _store_query_nav_intent(parsed)

    return parsed


def apply_query_nav_target_if_needed(runtime: DashboardRuntime) -> None:
    """
    מממש את Intent הניווט שנשמר מ-URL (אם קיים) ל-nav_target בפועל.

    Flow:
    -----
    1. קורא SESSION_KEY_QUERY_INTENT.
    2. אם intent["applied"] == True → לא עושה כלום.
    3. אם tab_key לא קיים ב-runtime.tab_registry → מתעלם (לוג אזהרה).
    4. אם הטאב לא enabled עבור profile נוכחי → מתעלם (לוג אזהרה).
    5. אחרת → קורא set_nav_target(tab_key, payload), מסמן applied=True.

    היתרון:
    --------
    * Separation of concerns:
        - Part 17 מטפל רק ב-query → intent → nav_target.
        - Part 15 (router) ממשיך לעבוד עם nav_target הרגיל (__consume_nav_target).
    """
    try:
        intent_raw = st.session_state.get(SESSION_KEY_QUERY_INTENT)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to access SESSION_KEY_QUERY_INTENT: %s", exc)
        return

    if not isinstance(intent_raw, Mapping):
        return

    if intent_raw.get("applied"):
        return

    tab_key = str(intent_raw.get("tab_key") or "").strip()
    if not tab_key:
        return

    payload = intent_raw.get("payload")
    if payload is not None and not isinstance(payload, Mapping):
        # נשמור רק Mapping; אם זה משהו אחר – נדחוס תחת "value"
        payload = {"value": payload}

    # בדיקה מול TabRegistry
    meta = runtime.tab_registry.get(tab_key)
    if meta is None:
        logger.warning(
            "Query nav intent refers to unknown tab_key='%s'; ignoring.", tab_key
        )
        # נסמן כ-applied כדי למנוע לופים אינסופיים
        intent_raw = dict(intent_raw)
        intent_raw["applied"] = True
        try:
            st.session_state[SESSION_KEY_QUERY_INTENT] = intent_raw
        except Exception:
            pass
        return

    # בדיקה אם הטאב enabled עבור הפרופיל הנוכחי
    if not _is_tab_enabled_for_profile(meta, runtime.profile):
        logger.warning(
            "Query nav intent tab_key='%s' is not enabled for profile='%s'; ignoring.",
            tab_key,
            runtime.profile,
        )
        intent_raw = dict(intent_raw)
        intent_raw["applied"] = True
        try:
            st.session_state[SESSION_KEY_QUERY_INTENT] = intent_raw
        except Exception:
            pass
        return

    # אם עברנו את הבדיקות – ניצור nav_target בפועל
    set_nav_target(tab_key, payload if isinstance(payload, Mapping) else None)

    intent_raw = dict(intent_raw)
    intent_raw["applied"] = True
    try:
        st.session_state[SESSION_KEY_QUERY_INTENT] = intent_raw
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to mark query nav intent as applied: %s", exc)

    logger.info(
        "Query nav intent applied as nav_target: tab_key=%s, payload_keys=%s",
        tab_key,
        list(payload.keys()) if isinstance(payload, Mapping) else None,
    )


# עדכון __all__ עבור חלק 17
try:
    __all__ += [
        "SESSION_KEY_QUERY_INTENT",
        "parse_dashboard_query_params",
        "apply_query_params_pre_runtime",
        "apply_query_nav_target_if_needed",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "SESSION_KEY_QUERY_INTENT",
        "parse_dashboard_query_params",
        "apply_query_params_pre_runtime",
        "apply_query_nav_target_if_needed",
    ]

# =====================
