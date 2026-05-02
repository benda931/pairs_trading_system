from __future__ import annotations

from types import SimpleNamespace


def test_feedback_loop_defaults_to_dry_run_and_skips_action_execution(monkeypatch, tmp_path):
    from core.orchestrator import PairsOrchestrator

    created = {"dry_run": None, "execute_args": None}

    class _StubLoop:
        def __init__(self, dry_run=False, **kwargs):
            created["dry_run"] = dry_run

        def process_agent_results(self, results):
            return [
                SimpleNamespace(
                    action_type="BLOCK_ENTRY",
                    target="SPY/QQQ",
                    severity="WARNING",
                    executed=False,
                )
            ]

        def execute_actions(self, actions):
            created["execute_args"] = list(actions)
            return SimpleNamespace(
                n_actions_generated=len(actions),
                n_actions_executed=0,
                n_actions_blocked=0,
                n_actions_throttled=0,
                system_state_changes={},
                actions=list(actions),
                throttled_actions=[],
            )

    monkeypatch.setattr(PairsOrchestrator, "_run_startup_reconciliation", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_startup_health_check", lambda self: None)
    monkeypatch.setattr("core.agent_feedback.AgentFeedbackLoop", _StubLoop)
    monkeypatch.setattr(
        "common.config_manager.load_config",
        lambda: {
            "strategy": {"dry_run": True},
            "execution": {
                "allow_live_orders": False,
                "allow_agent_actions": False,
                "paper_only": True,
            },
        },
    )

    orch = PairsOrchestrator()
    orch.bus.path = tmp_path / "bus.json"
    summary = orch._run_feedback_loop([SimpleNamespace(task_name="agent_x", status="success", output={})])

    assert created["dry_run"] is True
    assert created["execute_args"] == []
    latest = orch.bus.latest("feedback_loop")
    assert latest is not None
    assert latest["payload"]["execution_mode"]["dry_run"] is True
    assert latest["payload"]["execution_mode"]["allow_agent_actions"] is False
    assert summary is not None


def test_feedback_loop_throttles_actions_before_execution(monkeypatch, tmp_path):
    from core.orchestrator import PairsOrchestrator

    created = {"dry_run": None, "execute_args": None}

    class _StubLoop:
        def __init__(self, dry_run=False, **kwargs):
            created["dry_run"] = dry_run

        def process_agent_results(self, results):
            return [
                SimpleNamespace(
                    action_type="KILL_SWITCH",
                    target="system",
                    severity="EMERGENCY",
                    executed=False,
                )
            ]

        def execute_actions(self, actions):
            created["execute_args"] = list(actions)
            return SimpleNamespace(
                n_actions_generated=len(actions),
                n_actions_executed=0,
                n_actions_blocked=0,
                n_actions_throttled=0,
                system_state_changes={},
                actions=list(actions),
                throttled_actions=[],
            )

    class _StubThrottler:
        def allow(self, action_type, key=None):
            return False

        def mark(self, action_type, key=None):
            return None

    monkeypatch.setattr(PairsOrchestrator, "_run_startup_reconciliation", lambda self: None)
    monkeypatch.setattr(PairsOrchestrator, "_run_startup_health_check", lambda self: None)
    monkeypatch.setattr("core.agent_feedback.AgentFeedbackLoop", _StubLoop)
    monkeypatch.setattr("core.action_throttler.ActionThrottler", lambda: _StubThrottler())
    monkeypatch.setattr(
        "common.config_manager.load_config",
        lambda: {
            "strategy": {"dry_run": False},
            "execution": {
                "allow_live_orders": True,
                "allow_agent_actions": True,
                "paper_only": False,
            },
            "ib": {"enabled": True},
        },
    )

    orch = PairsOrchestrator()
    orch.bus.path = tmp_path / "bus.json"
    summary = orch._run_feedback_loop([SimpleNamespace(task_name="agent_x", status="success", output={})])

    assert created["dry_run"] is False
    assert created["execute_args"] == []
    assert summary is not None
    assert summary.n_actions_throttled == 1
    assert len(summary.throttled_actions) == 1
    latest = orch.bus.latest("feedback_loop")
    assert latest is not None
    assert len(latest["payload"]["throttled_actions"]) == 1
    assert latest["payload"]["throttled_actions"][0]["type"] == "KILL_SWITCH"
