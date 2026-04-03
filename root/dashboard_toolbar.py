# -*- coding: utf-8 -*-
"""
root/dashboard_toolbar.py — Dashboard toolbar
===============================================

Extracted from dashboard.py Part 22/35.
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

# Part 22/35 – Dashboard toolbar (HF-grade controls & personalization UI)
# =====================

def _render_dashboard_toolbar(
    runtime: DashboardRuntime,
    prefs: UserDashboardPrefs,
) -> None:
    """
    🎛 Dashboard Toolbar – שכבת שליטה אישית ברמת קרן.

    כעת כולל גם:
    - שליטה מוגבלת ב-env/profile דרך ה-UI:
        * env: רק אם PAIRS_ALLOW_ENV_SWITCH_FROM_UI מאפשר.
        * env="live": רק אם PAIRS_ALLOW_LIVE_ENV_FROM_UI מאפשר.
        * profile: תמיד ניתן לשנות (trading/research/risk/macro/monitoring/default).
    - עדיין לא משנה קונפיג גלובלי – רק את הסשן הנוכחי (st.session_state).
    """
    ff = runtime.feature_flags
    env = runtime.env
    profile = runtime.profile
    run_id = runtime.run_id

    live_actions = bool(ff.get("enable_live_trading_actions", False))
    experiment_mode = bool(ff.get("enable_experiment_mode", False))

    dense_layout = bool(prefs.dense_layout)
    debug_allowed = env in ("dev", "research", "test")
    current_debug = bool(ff.get("show_debug_info", False))

    # env/profile switching permissions
    raw_allow_env = os.getenv("PAIRS_ALLOW_ENV_SWITCH_FROM_UI")
    allow_env_switch = bool(raw_allow_env) and raw_allow_env.strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    raw_allow_live = os.getenv("PAIRS_ALLOW_LIVE_ENV_FROM_UI")
    allow_live_from_ui = bool(raw_allow_live) and raw_allow_live.strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    toolbar = st.container()
    with toolbar:
        st.markdown("#### 🎛 Dashboard controls")

        col_main, col_layout, col_debug = st.columns([2.2, 1.3, 1.3])

        # ---------- צד שמאל: Context, env/profile & modes ----------
        with col_main:
            st.markdown(
                f"- **Env:** `{env}`  •  **Profile:** `{profile}`  •  "
                f"**Run ID:** `{run_id}`"
            )
            st.caption(
                f"Live actions: "
                f"{'✅ Enabled' if live_actions else '⭕ Disabled'}  •  "
                f"Experiment mode: "
                f"{'🧪 On' if experiment_mode else 'Off'}"
            )

            # env/profile selectors (guarded)
            if allow_env_switch:
                env_options = list(_KNOWN_ENVS)
                try:
                    current_idx = env_options.index(env)
                except ValueError:
                    current_idx = 0

                new_env = st.selectbox(
                    "Env (UI override)",
                    options=env_options,
                    index=current_idx,
                    key=make_widget_key("toolbar", "env", run_id),
                    help=(
                        "שינוי env ברמת Session בלבד. "
                        "LIVE יתאפשר רק אם PAIRS_ALLOW_LIVE_ENV_FROM_UI מאופשר."
                    ),
                )

                if new_env != env:
                    # לא מאפשרים LIVE אם הדגל לא מוגדר
                    if new_env == "live" and not allow_live_from_ui:
                        st.warning(
                            "Env='live' לא ניתן לשינוי מה-UI – "
                            "אפשר רק אם PAIRS_ALLOW_LIVE_ENV_FROM_UI מאופשר."
                        )
                    else:
                        logger.info(
                            "Toolbar env switch requested: %s → %s (profile=%s)",
                            env,
                            new_env,
                            profile,
                        )
                        try:
                            st.session_state[SESSION_KEY_ENV] = new_env
                            st.session_state["env"] = new_env
                        except Exception as exc:  # pragma: no cover
                            logger.error(
                                "Failed to apply env switch to session_state: %s", exc
                            )
                        else:
                            try:
                                st.rerun()
                            except Exception as exc:  # pragma: no cover
                                logger.debug("st.rerun after env switch failed: %s", exc)

            # Profile selector – תמיד אפשר לשנות (ברמת Session)
            try:
                profile_options = list(_KNOWN_PROFILES)
                current_p_idx = profile_options.index(profile)
            except ValueError:
                profile_options = list(_KNOWN_PROFILES)
                current_p_idx = 0

            new_profile = st.selectbox(
                "Profile (UI preference)",
                options=profile_options,
                index=current_p_idx,
                key=make_widget_key("toolbar", "profile", run_id),
                help="שינוי פרופיל ברמת Session (trading/research/risk/macro/monitoring/default).",
            )

            if new_profile != profile:
                logger.info(
                    "Toolbar profile switch requested: %s → %s (env=%s)",
                    profile,
                    new_profile,
                    env,
                )
                try:
                    st.session_state[SESSION_KEY_PROFILE] = new_profile
                    st.session_state["profile"] = new_profile
                except Exception as exc:  # pragma: no cover
                    logger.error(
                        "Failed to apply profile switch to session_state: %s", exc
                    )
                else:
                    try:
                        st.rerun()
                    except Exception as exc:  # pragma: no cover
                        logger.debug("st.rerun after profile switch failed: %s", exc)

            if prefs.preferred_benchmark:
                st.caption(
                    f"Preferred benchmark: `{prefs.preferred_benchmark}` "
                    f"(משפיע על ניתוחים/דוחות ברמת ברירת מחדל)."
                )

        # ---------- אמצע: Layout / UX ----------
        with col_layout:
            st.markdown("**Layout & UX**")
            dense_new = st.checkbox(
                "Dense layout",
                value=dense_layout,
                key=make_widget_key("toolbar", "dense_layout", run_id),
                help=(
                    "מצמצם מרווחים ומסתיר חלק מהכותרות – "
                    "מותאם למשתמשים מתקדמים שרוצים לראות יותר דאטה על המסך."
                ),
            )

            # עדכון Prefs אם יש שינוי
            if dense_new != dense_layout:
                prefs.dense_layout = dense_new
                try:
                    st.session_state[SESSION_KEY_USER_PREFS] = prefs
                except Exception:  # pragma: no cover
                    pass
                logger.info(
                    "UserDashboardPrefs: dense_layout changed to %s for user_key=%s",
                    dense_new,
                    prefs.user_key,
                )

            st.caption(
                f"Nav history limit: {prefs.max_nav_history} events "
                "(ניתן לשנות בקונפיג/Prefs בעתיד)."
            )

        # ---------- ימין: Debug mode ----------
        with col_debug:
            st.markdown("**Debug & Telemetry**")

            if debug_allowed:
                debug_new = st.checkbox(
                    "Debug mode",
                    value=current_debug,
                    key=make_widget_key("toolbar", "debug_mode", run_id),
                    help=(
                        "מציג פאנלי Debug, stacktraces, Telemetry וזמני ריצה. "
                        "מומלץ רק למצב פיתוח/מחקר."
                    ),
                )

                if debug_new != current_debug:
                    # עדכון FeatureFlags
                    ff["show_debug_info"] = debug_new
                    # עדכון Prefs
                    prefs.show_debug_by_default = debug_new
                    try:
                        st.session_state[SESSION_KEY_USER_PREFS] = prefs
                    except Exception:  # pragma: no cover
                        pass

                    logger.info(
                        "Debug mode toggled to %s (env=%s, profile=%s, user_key=%s)",
                        debug_new,
                        env,
                        profile,
                        prefs.user_key,
                    )

                st.caption(
                    "זמין בסביבות dev/research/test בלבד. "
                    "ב-LIVE לא ניתן להפעיל debug מה-UI."
                )
            else:
                st.caption(
                    "Debug mode נעול בסביבה זו.  "
                    "הפעל דרך config/envvars בלבד."
                )


# -------------------------
# Shell wrapper – גרסה מעודכנת עם Toolbar
# -------------------------


# עדכון __all__ עבור חלק 22
try:
    __all__ += [
        "_render_dashboard_toolbar",
        "render_dashboard_shell",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_render_dashboard_toolbar",
        "render_dashboard_shell",
    ]
# =====================
