# -*- coding: utf-8 -*-
"""
root/dashboard_agents_tab.py — Advanced Agents Tab
=====================================================

Extracted from dashboard.py Part 30/35.

Contains the agents control center with agent dispatch,
action routing, saved views management, and status display.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# Import deps
try:
    from root.dashboard import (
        _log_tab_entry,
        _find_module,
        _find_tab_function_in_module,
        _invoke_tab_function,
        ensure_dashboard_runtime,
        export_dashboard_state_for_agents,
        TAB_KEY_SMART_SCAN,
        TAB_KEY_PAIR,
        TAB_KEY_BACKTEST,
        TAB_KEY_RISK,
        TAB_KEY_MACRO,
        TAB_KEY_MATRIX,
        TAB_KEY_COMPARISON_MATRICES,
        TAB_KEY_PORTFOLIO,
        TAB_KEY_FAIR_VALUE,
        TAB_KEY_LOGS,
    )
except ImportError:
    _log_tab_entry = lambda *a, **kw: None
    _find_module = lambda *a: None
    _find_tab_function_in_module = lambda *a: None
    _invoke_tab_function = lambda *a, **kw: None
    ensure_dashboard_runtime = lambda ctx: None
    export_dashboard_state_for_agents = lambda runtime: {}
    TAB_KEY_SMART_SCAN = "smart_scan"
    TAB_KEY_PAIR = "pair"
    TAB_KEY_BACKTEST = "backtest"
    TAB_KEY_RISK = "risk"
    TAB_KEY_MACRO = "macro"
    TAB_KEY_MATRIX = "matrix"
    TAB_KEY_COMPARISON_MATRICES = "comparison_matrices"
    TAB_KEY_PORTFOLIO = "portfolio"
    TAB_KEY_FAIR_VALUE = "fair_value"
    TAB_KEY_LOGS = "logs"

try:
    from root.dashboard_health import compute_dashboard_health, check_dashboard_ready
except ImportError:
    compute_dashboard_health = lambda runtime: None
    check_dashboard_ready = lambda ctx: {}

try:
    from root.dashboard_metrics import build_dashboard_overview_metrics, dashboard_overview_metrics_to_dict
except ImportError:
    build_dashboard_overview_metrics = lambda runtime: []
    dashboard_overview_metrics_to_dict = lambda m: []

try:
    from root.dashboard_integrations import (
        handle_agent_action, handle_agent_actions_batch,
        get_agent_actions_history_tail,
        list_saved_views, add_saved_view_from_runtime,
        find_saved_view_by_name, apply_saved_view,
        export_saved_views_for_agents,
    )
except ImportError:
    handle_agent_action = lambda *a, **kw: {}
    handle_agent_actions_batch = lambda *a, **kw: []
    get_agent_actions_history_tail = lambda limit=10: []
    list_saved_views = lambda: []
    add_saved_view_from_runtime = lambda *a, **kw: None
    find_saved_view_by_name = lambda name: None
    apply_saved_view = lambda *a, **kw: {}
    export_saved_views_for_agents = lambda runtime: {}

try:
    from root.dashboard_alerts_bus import render_dashboard_alert_center
except ImportError:
    render_dashboard_alert_center = lambda *a, **kw: None

# Forward refs
DashboardRuntime = Any
FeatureFlags = Dict[str, Any]
NavPayload = Optional[Dict[str, Any]]
AppContext = Any
TAB_KEY_AGENTS = "agents"

# Part 30/35 – Advanced Agents Tab (HF-grade AI control center)
# =====================

def _render_agents_internal_fallback(
    runtime: DashboardRuntime,
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:
    """
    מימוש פנימי מתקדם לטאב 🤖 Agents, במקרה שאין מודול root.agents_tab ייעודי.

    מה הטאב הזה נותן:
    ------------------
    1. **Context Panel** – מצב המערכת בעיניים של סוכן:
       - env/profile/run_id
       - health (severity/score/can_trade/can_backtest)
       - capabilities/domains
       - overview metrics summary

    2. **Quick Actions** – כפתורים שמפעילים handle_agent_action:
       - Open tabs (Backtest / Risk / Macro / Pair / Matrix / FairValue / Portfolio).
       - Run backtest for selected pair.
       - Snapshot (ל-SqlStore אם זמין).
       - Push state to Desktop (אם יש desktop_integration).
       - Save view (layout).

    3. **What should the agent care about?** – המלצות מתוך DashboardHealth:
       - issues / warnings / recommended_actions.

    4. **Saved Views** – יצירה, בחירה, והצגה של views (layouts).

    5. **Agent Context & Actions history** – context מלא + tail של פעולות Agents.
    """
    ff = feature_flags
    env = runtime.env
    profile = runtime.profile

    # ========= חלק 1 – Context overview (Health + capabilities + overview) =========

    st.markdown(
        """
<div style="
    background:linear-gradient(90deg,#0D47A1 0%,#1565C0 100%);
    border-radius:10px;padding:14px 20px;margin-bottom:14px;
    box-shadow:0 2px 8px rgba(13,71,161,0.22);
">
    <div style="font-size:1.15rem;font-weight:800;color:white;letter-spacing:-0.2px;">
        🤖 Agent Control Center
    </div>
    <div style="font-size:0.76rem;color:rgba(255,255,255,0.78);margin-top:3px;">
        AI supervision · Quick actions · System health · Saved views · Automation history
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    col_ctx_left, col_ctx_mid, col_ctx_right = st.columns([1.5, 1.5, 1.5])

    # ---- Health & meta ----
    health = compute_dashboard_health(runtime)
    with col_ctx_left:
        st.markdown("**Runtime & Health**")
        st.write(f"Env / Profile: `{env}` / `{profile}`")
        st.write(f"Run ID: `{runtime.run_id}`")
        st.write(
            f"Health: `{health.severity}`  •  "
            f"score≈`{health.score:.1f}`  •  "
            f"ready=`{health.ready}`"
        )
        st.caption(
            f"can_trade=`{health.can_trade}`  •  "
            f"can_backtest=`{health.can_backtest}`  •  "
            f"can_optimize=`{health.can_optimize}`  •  "
            f"can_monitor=`{health.can_monitor}`"
        )

    # ---- Capabilities / domains ----
    with col_ctx_mid:
        st.markdown("**Capabilities**")
        caps = runtime.capabilities
        cap_rows = [{"capability": k, "enabled": bool(v)} for k, v in caps.items()]
        if cap_rows:
            df_caps = pd.DataFrame(cap_rows)
            df_caps["status"] = df_caps["enabled"].map(lambda x: "✅" if x else "⭕")
            df_caps.set_index("capability", inplace=True)
            st.dataframe(df_caps, use_container_width=True, height=210)
        else:
            st.caption("No capabilities detected.")

    with col_ctx_right:
        st.markdown("**Overview snapshot**")
        overview_metrics = build_dashboard_overview_metrics(runtime)
        metrics_dicts = dashboard_overview_metrics_to_dict(overview_metrics)
        st.caption(f"Overview metrics: `{len(metrics_dicts)}`")

        # נציג רק summary קטן (מפתחות/קטגוריות)
        if metrics_dicts:
            sample = metrics_dicts[:6]
            st.json(
                [
                    {
                        "key": m["key"],
                        "category": m.get("category"),
                        "level": m.get("level"),
                        "value": m.get("value"),
                    }
                    for m in sample
                ]
            )
        else:
            st.caption("No overview metrics yet.")

    st.markdown("---")

    # ========= חלק 2 – Quick AI-style actions (navigation / backtest / snapshot / desktop) =========

    st.markdown("#### 2️⃣ Quick AI-style actions")

    col_open, col_bt, col_misc = st.columns([1.5, 1.5, 1.5])

    # 2.1 Open tabs
    with col_open:
        st.markdown("**Open / focus tabs**")

        # קונפיג טאב יעד
        tab_options = {
            "Smart Scan": TAB_KEY_SMART_SCAN,
            "Pair Analysis": TAB_KEY_PAIR,
            "Backtest": TAB_KEY_BACKTEST,
            "Risk": TAB_KEY_RISK,
            "Macro": TAB_KEY_MACRO,
            "Matrix": TAB_KEY_MATRIX,
            "Comparison Matrices": TAB_KEY_COMPARISON_MATRICES,
            "Portfolio": TAB_KEY_PORTFOLIO,
            "Fair Value": TAB_KEY_FAIR_VALUE,
            "Logs / System": TAB_KEY_LOGS,
        }

        target_label = st.selectbox(
            "בחר טאב לפתיחה (Navigation):",
            options=list(tab_options.keys()),
            key="agents_open_tab_select",
        )
        target_key = tab_options[target_label]

        if st.button("🎯 Open selected tab", key="agents_open_tab_btn"):
            res = handle_agent_action(
                runtime,
                {"source": "agents_tab", "action": "open_tab", "tab_key": target_key},
            )
            if res.get("status") == "ok":
                st.success(f"Tab `{target_key}` will be focused on next run.")
            else:
                st.warning(res)

    # 2.2 Backtest for pair
    with col_bt:
        st.markdown("**Backtest for pair**")

        pair_str = st.text_input(
            "Pair (symbol1/symbol2)",
            key="agents_bt_pair",
            placeholder="AAPL/MSFT",
        )
        preset = st.selectbox(
            "Preset",
            options=["smoke", "default", "deep", "wf"],
            index=0,
            key="agents_bt_preset",
        )
        mode = st.selectbox(
            "Mode",
            options=["single", "wf"],
            index=1,
            key="agents_bt_mode",
        )

        if st.button("🚀 Run backtest for pair", key="agents_bt_run"):
            pair_clean = pair_str.strip()
            if not pair_clean or "/" not in pair_clean:
                st.error("Please provide pair in format `SYMBOL1/SYMBOL2`.")
            else:
                res = handle_agent_action(
                    runtime,
                    {
                        "source": "agents_tab",
                        "action": "run_backtest_for_pair",
                        "tab_key": TAB_KEY_BACKTEST,
                        "payload": {
                            "pair": pair_clean,
                            "preset": preset,
                            "mode": mode,
                        },
                    },
                )
                if res.get("status") == "ok":
                    st.success(
                        f"Backtest nav_target set for pair `{pair_clean}` "
                        f"(preset={preset}, mode={mode})."
                    )
                else:
                    st.warning(res)

    # 2.3 Snapshot / Desktop push / headless test
    with col_misc:
        st.markdown("**Snapshot & Desktop & Headless test**")

        if st.button("💾 Snapshot dashboard", key="agents_snapshot"):
            res = handle_agent_action(
                runtime,
                {"source": "agents_tab", "action": "snapshot"},
            )
            if res.get("status") == "ok":
                st.success("Snapshot saved (SqlStore + session).")
            else:
                st.warning(res)

        if runtime.capabilities.get("desktop_integration", False):
            if st.button("🖥 Push state to Desktop", key="agents_push_desktop"):
                res = handle_agent_action(
                    runtime,
                    {"source": "agents_tab", "action": "push_to_desktop"},
                )
                if res.get("status") == "ok":
                    st.success("Dashboard state pushed to Desktop.")
                else:
                    st.warning(res)
        else:
            st.caption("Desktop integration is not enabled for this runtime.")

        # Headless test snapshot – בודק שהמערכת ready בלי UI
        if st.button("🧪 Run headless readiness check", key="agents_headless_check"):
            ready, info = check_dashboard_ready(runtime.app_ctx)  # type: ignore[arg-type]
            if ready:
                st.success("Headless readiness check: READY")
            else:
                st.error("Headless readiness check: NOT READY")
            st.json(info)

    st.markdown("---")

    # ========= חלק 3 – Health-driven recommendations (What should the agent do next?) =========

    st.markdown("#### 3️⃣ Health-driven recommendations")

    col_issues, col_reco = st.columns([1.6, 1.4])

    with col_issues:
        st.markdown("**Issues & warnings**")
        if health.issues or health.warnings:
            issues_list = [
                f"❌ {msg}" for msg in health.issues
            ] + [
                f"⚠️ {msg}" for msg in health.warnings
            ]
            for line in issues_list:
                st.write(line)
        else:
            st.caption("No issues/warnings reported by DashboardHealth.")

    with col_reco:
        st.markdown("**Suggested actions for Agent**")
        if health.recommended_actions:
            for rec in health.recommended_actions:
                st.write(f"👉 {rec}")
        else:
            st.caption("No specific recommendations – system looks healthy.")

    st.markdown("---")

    # ========= חלק 4 – Saved Views management =========

    st.markdown("#### 4️⃣ Saved views & layouts")

    views = list_saved_views()
    cols_views = st.columns([1.4, 1.3, 1.3])

    with cols_views[0]:
        st.markdown("**Create new view**")
        new_view_name = st.text_input(
            "View name",
            key="agents_new_view_name",
            placeholder="Morning monitor / Risk overview / ...",
        )
        new_view_tags = st.text_input(
            "Tags (comma-separated)",
            key="agents_new_view_tags",
            placeholder="monitoring, risk, live",
        )
        new_view_notes = st.text_area(
            "Notes (optional)",
            key="agents_new_view_notes",
            height=80,
        )

        if st.button("📌 Save current layout as view", key="agents_save_view"):
            tags_list = [
                t.strip()
                for t in new_view_tags.split(",")
                if t.strip()
            ] if new_view_tags else []
            view = add_saved_view_from_runtime(
                runtime,
                name=new_view_name or None,
                notes=new_view_notes or None,
                tags=tags_list,
            )
            st.success(f"Saved view `{view.name}`.")

    with cols_views[1]:
        st.markdown("**Apply existing view**")
        if views:
            view_names = [v.name for v in views]
            selected_view_name = st.selectbox(
                "Choose a view",
                options=view_names,
                key="agents_apply_view_select",
            )
            if st.button("🎯 Apply view", key="agents_apply_view_btn"):
                v = find_saved_view_by_name(selected_view_name)
                if v is None:
                    st.error("Selected view not found.")
                else:
                    res = apply_saved_view(runtime, v)
                    if res.get("status") == "ok":
                        st.success(
                            f"View `{v.name}` applied – target tab `{res.get('applied_tab')}` will focus next run."
                        )
                    else:
                        st.warning(res)
        else:
            st.caption("No saved views yet – create one on the left.")

    with cols_views[2]:
        st.markdown("**Views snapshot for agents**")
        views_export = export_saved_views_for_agents(runtime)
        st.json(views_export)

    st.markdown("---")

    # ========= חלק 5 – Agent context & actions history =========

    st.markdown("#### 5️⃣ Agent context & actions history")

    col_ctx, col_hist = st.columns([1.4, 1.6])

    with col_ctx:
        st.markdown("**Agent context payload**")
        # נעדכן context טרי (בלתי תלוי ב-shell)
        agent_ctx = export_dashboard_state_for_agents(runtime)
        st.json(agent_ctx)

    with col_hist:
        st.markdown("**Agent actions history (tail)**")
        history_tail = get_agent_actions_history_tail(limit=50)
        if not history_tail:
            st.caption("No agent actions recorded yet.")
        else:
            df_hist = pd.DataFrame(history_tail)
            st.dataframe(df_hist, use_container_width=True)

    st.markdown("---")

    # ========= חלק 6 – System alert center =========
    st.markdown("#### 6️⃣ System alert center")
    try:
        render_dashboard_alert_center(max_items=20)
    except Exception as _ae:
        st.caption(f"Alert center unavailable: {_ae}")

    st.markdown("---")

    # ========= חלק 7 – Data feed quality overview =========
    st.markdown("#### 7️⃣ Data feed quality")
    try:
        _caps7 = getattr(runtime, "capabilities", {}) or {}
        _feed_rows = []
        for _src_name, _label in [
            ("sql_store", "SQL store"),
            ("backtester", "Backtester"),
            ("optimizer", "Optimizer / Meta-optimizer"),
            ("risk_engine", "Risk engine"),
            ("macro_engine", "Macro engine"),
            ("live_data", "Live data feed"),
            ("broker", "Broker connectivity"),
            ("ibkr", "IBKR"),
            ("fmp", "FMP"),
            ("yfinance", "Yahoo Finance"),
        ]:
            _on = bool(_caps7.get(_src_name, False))
            _feed_rows.append({
                "source": _label,
                "status": "✅ Online" if _on else "⭕ Offline / N/A",
                "capability_key": _src_name,
            })
        if _feed_rows:
            _df_feed = pd.DataFrame(_feed_rows).set_index("source")
            st.dataframe(_df_feed, use_container_width=True)
            _n_online = sum(1 for r in _feed_rows if "Online" in r["status"])
            st.caption(f"{_n_online}/{len(_feed_rows)} data sources online")
        else:
            st.caption("No capability information available.")
    except Exception as _dfe:
        st.caption(f"Data feed quality check failed: {_dfe}")

    st.markdown("---")

    # ========= חלק 8 – Batch action runner =========
    st.markdown("#### 8️⃣ Batch action runner")
    st.caption(
        "Dispatch multiple agent actions in one shot — JSON array of action dicts. "
        "Each item must have at minimum `{\"action\": \"...\", \"source\": \"batch\"}`."
    )
    try:
        import json as _bjson
        _default_batch = (
            '[\n'
            '  {"action": "snapshot", "source": "agents_batch"},\n'
            '  {"action": "open_tab", "tab_key": "home", "source": "agents_batch"}\n'
            ']'
        )
        _batch_json = st.text_area(
            "Batch actions (JSON array)",
            value=_default_batch,
            height=130,
            key="agents_batch_runner_json",
        )
        if st.button("▶ Run batch", key="agents_batch_run_btn"):
            try:
                _batch_actions = _bjson.loads(_batch_json)
                if not isinstance(_batch_actions, list):
                    st.error("Input must be a JSON **array** (list) of action objects.")
                else:
                    _batch_results = handle_agent_actions_batch(runtime, _batch_actions)
                    _n_ok = sum(1 for r in _batch_results if isinstance(r, dict) and r.get("status") == "ok")
                    st.success(
                        f"Batch completed: {len(_batch_results)} actions · {_n_ok} succeeded."
                    )
                    st.json(_batch_results)
            except Exception as _berr:
                st.error(f"Batch failed: {_berr}")
    except Exception as _brun_err:
        st.caption(f"Batch runner unavailable: {_brun_err}")


# -------------------------
# override: render_agents_tab – keep external, then fallback
# -------------------------

def render_agents_tab(
    app_ctx: "AppContext",
    feature_flags: FeatureFlags,
    nav_payload: Optional[NavPayload] = None,
) -> None:  # type: ignore[override]
    """
    🤖 Agents Tab – גרסה משודרגת:

    סדר עדיפות:
    -----------
    1. ניסיון להריץ מודול ייעודי (agents_tab) – כדי לשמור על תאימות מלא.
    2. אם אין מודול / פונקציה מתאימה:
        → fallback פנימי עשיר (_render_agents_internal_fallback) על בסיס DashboardRuntime.

    יתרונות ה-fallback:
    --------------------
    - נותן Agent Control Center מלא גם בלי לבנות agents_tab.py.
    - משתמש בכל התשתיות שכבר בנינו:
        * DashboardRuntime
        * DashboardHealth + Overview + Home Context
        * handle_agent_action / batch
        * Saved Views
        * Desktop bridge
        * nav_target
    """
    _log_tab_entry(TAB_KEY_AGENTS, feature_flags, nav_payload)

    # 1) ניסיון להריץ מודול ייעודי (אם קיים)
    module = _find_module(
        (
            "agents_tab",
            "root.agents_tab",
        )
    )
    if module is not None:
        fn = _find_tab_function_in_module(
            module,
            (
                "render_agents_tab",
                "render_tab",
            ),
        )
        if fn is not None:
            try:
                _invoke_tab_function("agents", fn, app_ctx, feature_flags, nav_payload)
                return
            except Exception as exc:  # pragma: no cover
                logger.error(
                    "External agents_tab renderer raised %s – falling back to internal view.",
                    exc,
                    exc_info=True,
                )

    # 2) fallback פנימי – DashboardRuntime-based
    runtime = ensure_dashboard_runtime(app_ctx)
    _render_agents_internal_fallback(runtime, feature_flags, nav_payload)


# עדכון __all__ עבור חלק 30
try:
    __all__ += [
        "_render_agents_internal_fallback",
        "render_agents_tab",
    ]
except NameError:  # pragma: no cover
    __all__ = [
        "_render_agents_internal_fallback",
        "render_agents_tab",
    ]

# =====================
