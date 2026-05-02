from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pandas as pd
from types import SimpleNamespace


def _stale_price_frame() -> pd.DataFrame:
    end_ts = datetime.now(timezone.utc) - timedelta(days=400)
    index = pd.date_range(end=end_ts, periods=260, freq="B", tz="UTC")
    return pd.DataFrame({"close": [100.0 + i * 0.1 for i in range(len(index))]}, index=index)


def _fresh_price_frame() -> pd.DataFrame:
    end_ts = datetime.now(timezone.utc)
    index = pd.date_range(end=end_ts, periods=260, freq="B", tz="UTC")
    return pd.DataFrame({"close": [100.0 + i * 0.1 for i in range(len(index))]}, index=index)


def test_data_freshness_check_fails_and_blocks_compute_signals(monkeypatch, tmp_path):
    from core.agent_bus import AgentBus
    from core.orchestrator import PairsOrchestrator

    monkeypatch.setattr(PairsOrchestrator, "_run_startup_reconciliation", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_startup_health_check", lambda self: None)
    monkeypatch.setattr("common.config_manager.load_config", lambda: {"pairs": ["SPY/QQQ"]})
    monkeypatch.setattr("core.orchestrator._get_configured_pairs", lambda cfg=None: [("SPY", "QQQ")])
    monkeypatch.setattr("common.data_loader.load_price_data", lambda *args, **kwargs: _stale_price_frame())
    monkeypatch.setattr(
        "core.orchestrator.validate_pair_frames",
        lambda *args, **kwargs: {
            "ok": False,
            "reason": "stale_data",
            "x": {"ok": False},
            "y": {"ok": False},
        },
    )

    orch = PairsOrchestrator()
    orch.bus = AgentBus(path=tmp_path / "bus.json")

    freshness_result = orch._execute_task(orch.tasks["data_freshness_check"])

    assert freshness_result.status == "failed"
    assert freshness_result.output["reason"] == "all_pairs_stale_or_invalid"
    assert freshness_result.output["pairs_failed"] == 1
    latest = orch.bus.latest("data_freshness")
    assert latest is not None
    assert latest["payload"]["status"] == "failed"

    compute_called = {"value": False}

    def _should_not_run(**kwargs):
        compute_called["value"] = True
        return {"status": "ok"}

    orch.tasks["compute_signals"].func = _should_not_run
    compute_result = orch._execute_task(orch.tasks["compute_signals"])

    assert compute_result.status == "skipped"
    assert compute_result.output["reason"] == "all_pairs_stale_or_invalid"
    assert compute_result.error == "dep:data_freshness_check failed:all_pairs_stale_or_invalid"
    assert compute_called["value"] is False


def test_run_daily_pipeline_all_pairs_stale_skips_compute_signals_and_allocation(monkeypatch, tmp_path):
    from core.agent_bus import AgentBus
    from core.agent_bus import TaskResult
    from core.orchestrator import PairsOrchestrator

    monkeypatch.setattr(PairsOrchestrator, "_run_startup_reconciliation", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_startup_health_check", lambda self: None)
    monkeypatch.setattr("common.config_manager.load_config", lambda: {"pairs": ["SPY/QQQ"]})
    monkeypatch.setattr("core.orchestrator._get_configured_pairs", lambda cfg=None: [("SPY", "QQQ")])
    monkeypatch.setattr("common.data_loader.load_price_data", lambda *args, **kwargs: _stale_price_frame())
    monkeypatch.setattr(
        "core.orchestrator.validate_pair_frames",
        lambda *args, **kwargs: {
            "ok": False,
            "reason": "stale_data",
            "x": {"ok": False},
            "y": {"ok": False},
        },
    )
    monkeypatch.setattr(PairsOrchestrator, "run_agent_system_health_check", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "run_agent_data_integrity_check", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_risk_analytics", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_universe_scan", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_correlation_monitor", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_feedback_loop", lambda self, results: None)
    monkeypatch.setattr(PairsOrchestrator, "run_agent_training_cycle", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "run_cfa_research_cycle", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "run_portfolio_monitor_cycle", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_send_pipeline_alerts", lambda self, results, signal_decisions: None)
    monkeypatch.setattr(PairsOrchestrator, "_dispatch_signal_agents", lambda self: [])
    monkeypatch.setattr(PairsOrchestrator, "_dispatch_risk_agents", lambda self: [])

    compute_called = {"value": False}
    alloc_called = {"value": False}

    def _fake_compute(**kwargs):
        compute_called["value"] = True
        return {"status": "ok"}

    def _fake_allocate(*args, **kwargs):
        alloc_called["value"] = True
        return TaskResult(task_name="portfolio_allocation", status="success")

    orch = PairsOrchestrator()
    orch.bus = AgentBus(path=tmp_path / "bus.json")
    orch.tasks["compute_signals"].func = _fake_compute
    monkeypatch.setattr(orch, "run_portfolio_allocation_cycle", _fake_allocate)

    results = orch.run_daily_pipeline()
    result_map = {r.task_name: r for r in results if getattr(r, "task_name", None)}

    assert compute_called["value"] is False
    assert alloc_called["value"] is False
    assert result_map["data_freshness_check"].status == "failed"
    assert result_map["compute_signals"].status == "skipped"
    assert "portfolio_allocation" not in result_map
    latest = orch.bus.latest("data_freshness")
    assert latest is not None
    assert latest["payload"]["status"] == "failed"
    assert latest["payload"]["reason"] == "all_pairs_stale_or_invalid"


def test_run_daily_pipeline_partial_freshness_uses_only_passed_pairs(monkeypatch, tmp_path):
    from core.agent_bus import AgentBus
    from core.agent_bus import TaskResult
    from core.orchestrator import PairsOrchestrator

    monkeypatch.setattr(PairsOrchestrator, "_run_startup_reconciliation", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_startup_health_check", lambda self: None)
    monkeypatch.setattr(
        "common.config_manager.load_config",
        lambda: {"pairs": ["SPY/QQQ", "IWM/SPY"], "scheduler_capital": 1000000.0},
    )
    monkeypatch.setattr(
        "core.orchestrator._get_configured_pairs",
        lambda cfg=None: [("SPY", "QQQ"), ("IWM", "SPY")],
    )
    monkeypatch.setattr(PairsOrchestrator, "run_agent_system_health_check", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "run_agent_data_integrity_check", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_risk_analytics", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_universe_scan", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_correlation_monitor", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_feedback_loop", lambda self, results: None)
    monkeypatch.setattr(PairsOrchestrator, "run_agent_training_cycle", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "run_cfa_research_cycle", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "run_portfolio_monitor_cycle", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_send_pipeline_alerts", lambda self, results, signal_decisions: None)
    monkeypatch.setattr(PairsOrchestrator, "_dispatch_signal_agents", lambda self: [])
    monkeypatch.setattr(PairsOrchestrator, "_dispatch_risk_agents", lambda self: [])

    def _fake_load_price_data(symbol, *args, **kwargs):
        if symbol == "IWM":
            return _stale_price_frame()
        return _fresh_price_frame()

    monkeypatch.setattr("common.data_loader.load_price_data", _fake_load_price_data)
    monkeypatch.setattr(
        "core.orchestrator.validate_pair_frames",
        lambda sym_x, df_x, sym_y, df_y, **kwargs: {
            "ok": (sym_x, sym_y) == ("SPY", "QQQ"),
            "reason": "ok" if (sym_x, sym_y) == ("SPY", "QQQ") else "stale_data",
            "x": {"ok": (sym_x, sym_y) == ("SPY", "QQQ")},
            "y": {"ok": (sym_x, sym_y) == ("SPY", "QQQ")},
        },
    )

    captured_pairs = {"compute": None, "collect": None, "alloc": None}

    def _fake_compute(**kwargs):
        orchestrator = kwargs.get("orchestrator")
        captured_pairs["compute"] = list(getattr(orchestrator, "_fresh_pairs_override") or [])
        return {
            "status": "ok",
            "pairs_requested": len(captured_pairs["compute"]),
            "pairs_computed": len(captured_pairs["compute"]),
        }

    def _fake_collect(self, pairs=None, lookback_days=252):
        resolved = list(pairs) if pairs is not None else list(self._get_active_pairs())
        captured_pairs["collect"] = resolved
        return [SimpleNamespace(pair_id=f"{resolved[0][0]}/{resolved[0][1]}")]

    def _fake_allocate(signal_decisions=None, capital=0.0, allocation_batch_id=None):
        captured_pairs["alloc"] = list(signal_decisions or [])
        return TaskResult(task_name="portfolio_allocation", status="success")

    orch = PairsOrchestrator()
    orch.bus = AgentBus(path=tmp_path / "bus.json")
    orch.tasks["compute_signals"].func = _fake_compute
    monkeypatch.setattr(PairsOrchestrator, "_collect_signal_decisions", _fake_collect)
    monkeypatch.setattr(orch, "run_portfolio_allocation_cycle", _fake_allocate)

    results = orch.run_daily_pipeline()
    result_map = {r.task_name: r for r in results if getattr(r, "task_name", None)}

    assert result_map["data_freshness_check"].status == "success"
    assert result_map["data_freshness_check"].output["status"] == "partial"
    assert captured_pairs["compute"] == [("SPY", "QQQ")]
    assert captured_pairs["collect"] == [("SPY", "QQQ")]
    assert len(captured_pairs["alloc"]) == 1
    assert captured_pairs["alloc"][0].pair_id == "SPY/QQQ"
    latest = orch.bus.latest("data_freshness")
    assert latest is not None
    assert latest["payload"]["status"] == "partial"
    assert latest["payload"]["passed_pairs"] == ["SPY/QQQ"]


def test_run_daily_pipeline_all_pass_continues_normally(monkeypatch, tmp_path):
    from core.agent_bus import AgentBus
    from core.agent_bus import TaskResult
    from core.orchestrator import PairsOrchestrator

    monkeypatch.setattr(PairsOrchestrator, "_run_startup_reconciliation", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_startup_health_check", lambda self: None)
    monkeypatch.setattr(
        "common.config_manager.load_config",
        lambda: {"pairs": ["SPY/QQQ"], "scheduler_capital": 1000000.0},
    )
    monkeypatch.setattr("core.orchestrator._get_configured_pairs", lambda cfg=None: [("SPY", "QQQ")])
    monkeypatch.setattr("common.data_loader.load_price_data", lambda *args, **kwargs: _fresh_price_frame())
    monkeypatch.setattr(
        "core.orchestrator.validate_pair_frames",
        lambda *args, **kwargs: {
            "ok": True,
            "reason": "ok",
            "x": {"ok": True},
            "y": {"ok": True},
        },
    )
    monkeypatch.setattr(PairsOrchestrator, "run_agent_system_health_check", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "run_agent_data_integrity_check", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_risk_analytics", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_universe_scan", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_correlation_monitor", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_feedback_loop", lambda self, results: None)
    monkeypatch.setattr(PairsOrchestrator, "run_agent_training_cycle", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "run_cfa_research_cycle", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "run_portfolio_monitor_cycle", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_send_pipeline_alerts", lambda self, results, signal_decisions: None)
    monkeypatch.setattr(PairsOrchestrator, "_dispatch_signal_agents", lambda self: [])
    monkeypatch.setattr(PairsOrchestrator, "_dispatch_risk_agents", lambda self: [])

    captured = {"compute": False, "collect": False, "alloc": False}

    def _fake_compute(**kwargs):
        captured["compute"] = True
        return {"status": "ok", "pairs_requested": 1, "pairs_computed": 1}

    def _fake_collect(self, pairs=None, lookback_days=252):
        captured["collect"] = True
        return [SimpleNamespace(pair_id="SPY/QQQ")]

    def _fake_allocate(signal_decisions=None, capital=0.0, allocation_batch_id=None):
        captured["alloc"] = True
        return TaskResult(task_name="portfolio_allocation", status="success")

    orch = PairsOrchestrator()
    orch.bus = AgentBus(path=tmp_path / "bus.json")
    orch.tasks["compute_signals"].func = _fake_compute
    monkeypatch.setattr(PairsOrchestrator, "_collect_signal_decisions", _fake_collect)
    monkeypatch.setattr(orch, "run_portfolio_allocation_cycle", _fake_allocate)

    results = orch.run_daily_pipeline()
    result_map = {r.task_name: r for r in results if getattr(r, "task_name", None)}

    assert result_map["data_freshness_check"].status == "success"
    assert result_map["data_freshness_check"].output["status"] == "ok"
    assert captured == {"compute": True, "collect": True, "alloc": True}
