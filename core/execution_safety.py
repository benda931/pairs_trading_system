from __future__ import annotations

from typing import Any


def get_execution_mode(config: dict | None = None) -> dict[str, Any]:
    if config is None:
        from common.config_manager import load_config

        config = load_config()

    cfg = dict(config or {})
    strategy_dry_run = bool((cfg.get("strategy") or {}).get("dry_run", True))
    execution = dict(cfg.get("execution") or {})
    ib_enabled = bool(cfg.get("ib_enable", False) or (cfg.get("ib") or {}).get("enabled", False))
    allow_live_orders = bool(execution.get("allow_live_orders", False))
    allow_agent_actions = bool(execution.get("allow_agent_actions", False))
    paper_only = bool(execution.get("paper_only", True))

    effective_dry_run = not (
        strategy_dry_run is False
        and allow_live_orders is True
        and ib_enabled is True
        and paper_only is False
    )

    return {
        "dry_run": effective_dry_run,
        "allow_agent_actions": allow_agent_actions and not effective_dry_run,
        "allow_live_orders": allow_live_orders and not effective_dry_run,
        "paper_only": paper_only,
        "ib_enabled": ib_enabled,
    }
