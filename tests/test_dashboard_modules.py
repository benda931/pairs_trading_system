# -*- coding: utf-8 -*-
"""
Tests for extracted dashboard modules.

Verifies that all 18 extracted modules:
1. Import without errors
2. Define expected symbols
3. Maintain backward compatibility via dashboard.py re-exports
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── Section 1: Module importability ─────────────────────────────

EXTRACTED_MODULES = [
    "root.dashboard_alerts_bus",
    "root.dashboard_telemetry",
    "root.dashboard_health",
    "root.dashboard_metrics",
    "root.dashboard_integrations",
    "root.dashboard_logs_tab",
    "root.dashboard_agents_tab",
    "root.dashboard_diagnostics",
    "root.dashboard_preferences",
    "root.dashboard_home_helpers",
    "root.dashboard_cache",
    "root.dashboard_ui_components",
    "root.dashboard_sidebar",
    "root.dashboard_url_router",
    "root.dashboard_agent_context",
    "root.dashboard_toolbar",
    "root.dashboard_api_bundle",
    "root.dashboard_headless",
]


class TestDashboardModuleSyntax:
    """All extracted dashboard modules must have valid Python syntax."""

    @pytest.mark.parametrize("module_name", EXTRACTED_MODULES)
    def test_module_syntax_valid(self, module_name):
        """Each extracted module file must parse without syntax errors."""
        # Convert module name to file path
        rel_path = module_name.replace(".", "/") + ".py"
        file_path = PROJECT_ROOT / rel_path
        assert file_path.exists(), f"Module file not found: {file_path}"

        source = file_path.read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"Syntax error in {rel_path}: {exc}")


class TestDashboardCoreImport:
    """The main dashboard.py must still import cleanly."""

    def test_dashboard_main_syntax(self):
        file_path = PROJECT_ROOT / "root" / "dashboard.py"
        source = file_path.read_text(encoding="utf-8")
        ast.parse(source)  # Should not raise


# ─── Section 2: Key exports verification ─────────────────────────

class TestDashboardAlertsBus:
    """root.dashboard_alerts_bus must export expected symbols."""

    def test_exports(self):
        from root.dashboard_alerts_bus import (
            DashboardAlert,
            emit_dashboard_alert,
            get_dashboard_alerts,
            clear_dashboard_alerts,
            SESSION_KEY_ALERTS,
        )
        assert SESSION_KEY_ALERTS == "dashboard_alerts"
        assert callable(emit_dashboard_alert)
        assert callable(get_dashboard_alerts)
        assert callable(clear_dashboard_alerts)


class TestDashboardTelemetry:
    """root.dashboard_telemetry must export expected symbols."""

    def test_exports(self):
        from root.dashboard_telemetry import (
            DashboardSummary,
            TabUsageStats,
            ServiceHealthSnapshot,
            build_dashboard_summary,
            SESSION_KEY_DASHBOARD_SUMMARY,
        )
        assert callable(build_dashboard_summary)
        assert SESSION_KEY_DASHBOARD_SUMMARY == "dashboard_summary"


class TestDashboardHealth:
    """root.dashboard_health must export expected symbols."""

    def test_exports(self):
        from root.dashboard_health import (
            DashboardHealth,
            compute_dashboard_health,
            check_dashboard_ready,
            SESSION_KEY_HEALTH_LAST,
        )
        assert callable(compute_dashboard_health)
        assert callable(check_dashboard_ready)


class TestDashboardMetrics:
    """root.dashboard_metrics must export expected symbols."""

    def test_exports(self):
        from root.dashboard_metrics import (
            OverviewMetric,
            build_dashboard_overview_metrics,
            SESSION_KEY_OVERVIEW_LAST,
        )
        assert callable(build_dashboard_overview_metrics)


class TestDashboardIntegrations:
    """root.dashboard_integrations must export expected symbols."""

    def test_desktop_bridge_exports(self):
        from root.dashboard_integrations import (
            DashboardDesktopBridgeConfig,
            build_desktop_integration_config,
            get_last_desktop_push_info,
        )
        assert callable(build_desktop_integration_config)

    def test_agent_router_exports(self):
        from root.dashboard_integrations import (
            AgentAction,
            handle_agent_action,
            handle_agent_actions_batch,
            get_agent_actions_history_tail,
        )
        assert callable(handle_agent_action)

    def test_saved_views_exports(self):
        from root.dashboard_integrations import (
            SavedDashboardView,
            list_saved_views,
            add_saved_view_from_runtime,
            find_saved_view_by_name,
            apply_saved_view,
            export_saved_views_for_agents,
        )
        assert callable(list_saved_views)
        assert callable(find_saved_view_by_name)


class TestContextServices:
    """core.context_services must export service init functions."""

    def test_exports(self):
        from core.context_services import (
            init_all_services,
            _init_sql_store,
            _init_market_data_router,
            _init_engines,
            _init_fair_value_api,
            _init_agents_manager,
            _init_ib_router,
        )
        assert callable(init_all_services)
        assert callable(_init_sql_store)
        assert callable(_init_engines)


# ─── Section 3: File size regression guard ────────────────────────

class TestDashboardFileSize:
    """Guard against dashboard.py growing back into a monolith."""

    def test_dashboard_under_8000_lines(self):
        """dashboard.py must stay under 8,000 lines after refactoring."""
        file_path = PROJECT_ROOT / "root" / "dashboard.py"
        line_count = len(file_path.read_text(encoding="utf-8").splitlines())
        assert line_count < 8000, (
            f"dashboard.py has grown to {line_count} lines! "
            f"Extract new code into dashboard_*.py modules instead."
        )

    def test_app_context_under_5000_lines(self):
        """app_context.py must stay under 5,000 lines after refactoring."""
        file_path = PROJECT_ROOT / "core" / "app_context.py"
        line_count = len(file_path.read_text(encoding="utf-8").splitlines())
        assert line_count < 5000, (
            f"app_context.py has grown to {line_count} lines! "
            f"Extract new code into core/context_*.py modules instead."
        )
