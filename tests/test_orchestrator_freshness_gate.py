from __future__ import annotations

import pandas as pd


def _stale_price_frame() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=260, freq="B", tz="UTC")
    return pd.DataFrame({"close": [100.0 + i * 0.1 for i in range(len(index))]}, index=index)


def test_data_freshness_check_fails_and_blocks_compute_signals(monkeypatch, tmp_path):
    from core.agent_bus import AgentBus
    from core.orchestrator import PairsOrchestrator

    monkeypatch.setattr(PairsOrchestrator, "_run_startup_reconciliation", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_startup_health_check", lambda self: None)
    monkeypatch.setattr("common.config_manager.load_config", lambda: {"pairs": ["SPY/QQQ"]})
    monkeypatch.setattr("core.orchestrator._get_configured_pairs", lambda cfg=None: [("SPY", "QQQ")])
    monkeypatch.setattr("common.data_loader.load_price_data", lambda *args, **kwargs: _stale_price_frame())

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
