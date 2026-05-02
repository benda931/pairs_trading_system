from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

from core.state_provider import InMemoryStateProvider


def test_inmemory_state_provider_get_set_has_delete() -> None:
    provider = InMemoryStateProvider()

    assert provider.get("missing") is None
    assert provider.get("missing", 123) == 123
    assert provider.has("answer") is False

    provider.set("answer", 42)

    assert provider.has("answer") is True
    assert provider.get("answer") == 42

    provider.delete("answer")

    assert provider.has("answer") is False
    assert provider.get("answer") is None


def test_importing_app_context_does_not_require_streamlit_runtime() -> None:
    mod = importlib.import_module("core.app_context")
    assert hasattr(mod, "AppContext")
    assert hasattr(mod, "get_app_context")


def test_app_context_get_global_works_with_inmemory_provider(monkeypatch) -> None:
    app_context = importlib.import_module("core.app_context")
    provider = InMemoryStateProvider()

    monkeypatch.setattr(app_context.AppContext, "_GLOBAL_CTX", None)
    monkeypatch.setattr(app_context.AppContext, "init_services", lambda self: None)
    app_context.set_state_provider(provider)

    ctx = app_context.AppContext.get_global()

    assert isinstance(ctx, app_context.AppContext)
    assert provider.get("app_ctx") is ctx
    assert ctx.section == "dashboard"


def test_alert_bus_uses_injected_provider() -> None:
    alert_bus = importlib.import_module("core.alert_bus")
    provider = InMemoryStateProvider()
    alert_bus.set_state_provider(provider)
    alert_bus.clear_dashboard_alerts()

    alert = alert_bus.emit_dashboard_alert("warning", "test", "hello")

    assert alert.message == "hello"
    alerts = alert_bus.get_dashboard_alerts()
    assert len(alerts) == 1
    assert provider.get("dashboard_alerts")[0]["message"] == "hello"


def test_core_has_no_top_level_streamlit_imports() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for path in (repo_root / "core").rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped in {"import streamlit as st", "from streamlit import"} or stripped.startswith("from streamlit import "):
                offenders.append(f"{path}:{idx}:{stripped}")
    assert offenders == []


def test_ml_analysis_summary_uses_injected_provider() -> None:
    ml_analysis = importlib.import_module("core.ml_analysis")
    provider = InMemoryStateProvider()
    ml_analysis.set_state_provider(provider)

    summary = {"config": {"model": "xgb"}, "metrics": {"rmse": 1.23}}
    ml_analysis.publish_ml_summary_to_session(summary, key="ml_summary_test")

    assert provider.get("ml_summary_test") == summary


def test_macro_adjustments_uses_injected_provider_for_regime_memory() -> None:
    macro_adjustments = importlib.import_module("common.macro_adjustments")
    provider = InMemoryStateProvider()
    macro_adjustments.set_state_provider(provider)

    cfg = macro_adjustments.MacroConfig()
    cfg.min_regime_duration = 2
    current = pd.Series({"risk_on": 0.3, "growth": 0.2, "inflation": 0.1})

    first = macro_adjustments._gate_regime_change(current, "risk_on", cfg)
    second = macro_adjustments._gate_regime_change(current, "risk_on", cfg)

    assert first.equals(current)
    assert second.equals(current)
    stored = provider.get("macro_prev_regime_state")
    assert stored["bucket"] == "risk_on"
    assert stored["since"] == 1


def test_macro_adjustments_uses_injected_provider_for_hysteresis() -> None:
    macro_adjustments = importlib.import_module("common.macro_adjustments")
    provider = InMemoryStateProvider()
    macro_adjustments.set_state_provider(provider)

    cfg = macro_adjustments.MacroConfig()
    cfg.hysteresis_days = 2
    regime = pd.Series({"risk_on": -0.4, "growth": -0.2, "inflation": 0.3})

    first = macro_adjustments._apply_hysteresis_filters({"SPY/QQQ": True}, regime, cfg)
    second = macro_adjustments._apply_hysteresis_filters({"SPY/QQQ": False}, regime, cfg)

    assert first["SPY/QQQ"] is True
    assert second["SPY/QQQ"] is True
    stored = provider.get("macro_pair_hysteresis")
    assert stored["SPY/QQQ"]["include"] is True
