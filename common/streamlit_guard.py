# -*- coding: utf-8 -*-
"""
common/streamlit_guard.py
------------------------
Safe Streamlit accessors that do not spam warnings in bare/CLI mode.
"""

from __future__ import annotations

import logging
from typing import Any, Optional


_ST_CTX_LOGGER_NAME = "streamlit.runtime.scriptrunner_utils.script_run_context"


def safe_session_state() -> Optional[Any]:
    """
    Return st.session_state only when running under Streamlit.

    In bare/CLI mode, Streamlit's get_script_run_ctx() emits a WARNING.
    We temporarily suppress that specific logger to avoid noisy tooling output.
    """
    try:
        import streamlit as st  # type: ignore
        from streamlit.runtime.scriptrunner_utils.script_run_context import (  # type: ignore
            get_script_run_ctx,
        )
    except Exception:
        return None

    ctx_logger = logging.getLogger(_ST_CTX_LOGGER_NAME)
    old_level = ctx_logger.level
    try:
        # suppress only the ctx warning in bare mode
        ctx_logger.setLevel(logging.ERROR)
        ctx = get_script_run_ctx()
    finally:
        ctx_logger.setLevel(old_level)

    if ctx is None:
        return None

    try:
        return st.session_state
    except Exception:
        return None
