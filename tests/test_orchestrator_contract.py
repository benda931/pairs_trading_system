from __future__ import annotations


def test_validate_pipeline_contract_default_is_ok(monkeypatch):
    from core.orchestrator import PairsOrchestrator

    monkeypatch.setattr(PairsOrchestrator, "_run_startup_reconciliation", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_startup_health_check", lambda self: None)

    orch = PairsOrchestrator()
    contract = orch.validate_pipeline_contract()

    assert contract["ok"] is True
    assert contract["errors"] == []
    assert contract["dependencies"]["compute_signals"] == ["data_freshness_check"]


def test_validate_pipeline_contract_fails_when_compute_dependency_removed(monkeypatch):
    from core.orchestrator import PairsOrchestrator

    monkeypatch.setattr(PairsOrchestrator, "_run_startup_reconciliation", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_startup_health_check", lambda self: None)

    orch = PairsOrchestrator()
    orch.tasks["compute_signals"].depends_on = []

    contract = orch.validate_pipeline_contract()

    assert contract["ok"] is False
    assert any("compute_signals must depend on data_freshness_check" in err for err in contract["errors"])


def test_validate_pipeline_contract_fails_when_freshness_task_missing(monkeypatch):
    from core.orchestrator import PairsOrchestrator

    monkeypatch.setattr(PairsOrchestrator, "_run_startup_reconciliation", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_startup_health_check", lambda self: None)

    orch = PairsOrchestrator()
    orch.tasks.pop("data_freshness_check", None)

    contract = orch.validate_pipeline_contract()

    assert contract["ok"] is False
    assert any("missing required tasks" in err and "data_freshness_check" in err for err in contract["errors"])
