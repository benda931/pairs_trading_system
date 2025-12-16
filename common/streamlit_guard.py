# common/streamlit_guard.py
from __future__ import annotations

from typing import Any, Optional

def is_streamlit_runtime() -> bool:
    try:
        import streamlit as st  # type: ignore
        # הגנה: בקונטקסט אמיתי יש ScriptRunContext
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        return get_script_run_ctx() is not None
    except Exception:
        return False

def safe_session_state() -> Optional[Any]:
    if not is_streamlit_runtime():
        return None
    try:
        import streamlit as st  # type: ignore
        return st.session_state
    except Exception:
        return None
