from __future__ import annotations

from core.execution_safety import get_execution_mode


def test_execution_safety_default_config_is_dry_run():
    mode = get_execution_mode({})

    assert mode["dry_run"] is True
    assert mode["allow_live_orders"] is False
    assert mode["allow_agent_actions"] is False


def test_execution_safety_strategy_dry_run_false_alone_is_still_dry_run():
    mode = get_execution_mode({"strategy": {"dry_run": False}})

    assert mode["dry_run"] is True


def test_execution_safety_allow_live_orders_without_ib_is_still_dry_run():
    mode = get_execution_mode(
        {
            "strategy": {"dry_run": False},
            "execution": {"allow_live_orders": True, "paper_only": False},
        }
    )

    assert mode["dry_run"] is True
    assert mode["allow_live_orders"] is False


def test_execution_safety_all_live_gates_true_disables_dry_run():
    mode = get_execution_mode(
        {
            "strategy": {"dry_run": False},
            "execution": {
                "allow_live_orders": True,
                "allow_agent_actions": True,
                "paper_only": False,
            },
            "ib": {"enabled": True},
        }
    )

    assert mode["dry_run"] is False
    assert mode["allow_live_orders"] is True
    assert mode["allow_agent_actions"] is True
