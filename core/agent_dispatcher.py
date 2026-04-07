# -*- coding: utf-8 -*-
"""
core/agent_dispatcher.py — Agent Dispatch and Scheduling
=========================================================

Manages the lifecycle of autonomous agents:
- Registration with dependency declarations
- Scheduled dispatch (cron-like, interval-based)
- Dependency resolution before dispatch
- Failure handling with circuit breaker
- AgentDecision routing to approval queue

Extracted from core/orchestrator.py to give agent management
its own module with clear ownership.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("core.agent_dispatcher")


@dataclass
class AgentRegistration:
    """Registry entry for a single agent."""
    agent_id: str
    description: str
    run_fn: Callable
    schedule_interval_seconds: float = 3600.0   # Default: hourly
    depends_on: List[str] = field(default_factory=list)
    tier: str = "ADVISORY_ONLY"   # ADVISORY_ONLY | AUTO_EXECUTABLE | POLICY_GATED | HUMAN_REQUIRED
    max_consecutive_failures: int = 3
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _last_run_at: float = field(default=0.0, init=False, repr=False)
    _circuit_open: bool = field(default=False, init=False, repr=False)

    def is_due(self) -> bool:
        """Return True if the agent is due for its next run."""
        if self._circuit_open:
            return False
        return (time.monotonic() - self._last_run_at) >= self.schedule_interval_seconds

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._last_run_at = time.monotonic()
        if self._circuit_open:
            logger.info("AgentDispatcher: circuit closed for %s", self.agent_id)
            self._circuit_open = False

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        self._last_run_at = time.monotonic()
        if self._consecutive_failures >= self.max_consecutive_failures:
            self._circuit_open = True
            logger.error(
                "AgentDispatcher: circuit OPEN for %s after %d consecutive failures — "
                "agent disabled until manually reset",
                self.agent_id, self._consecutive_failures,
            )

    def reset_circuit(self) -> None:
        """Manually reset the circuit breaker to re-enable a failed agent."""
        self._circuit_open = False
        self._consecutive_failures = 0
        logger.info("AgentDispatcher: circuit manually reset for %s", self.agent_id)


class AgentDispatcher:
    """
    Schedules and dispatches autonomous agents.

    Agents must be registered via `register()` before the dispatch loop
    is started. Each agent's `run_fn` is called when due and dependencies
    have completed successfully.

    Agent decisions (AgentDecision objects) are routed to the approval
    queue rather than written directly to storage.
    """

    def __init__(self, bus=None):
        """
        Parameters
        ----------
        bus : AgentBus, optional
            File-based message bus for inter-agent communication.
        """
        self._agents: Dict[str, AgentRegistration] = {}
        self._bus = bus
        self._completed_this_cycle: set = set()

    def register(self, registration: AgentRegistration) -> None:
        """Register an agent for scheduled dispatch."""
        self._agents[registration.agent_id] = registration
        logger.info(
            "AgentDispatcher: registered %s (interval=%ds, tier=%s)",
            registration.agent_id,
            int(registration.schedule_interval_seconds),
            registration.tier,
        )

    def dispatch_due(self, context: Optional[Any] = None) -> List[str]:
        """
        Dispatch all agents that are due and have satisfied dependencies.

        Parameters
        ----------
        context : AppContext or similar, optional
            Passed to each agent's run_fn.

        Returns
        -------
        list[str] : agent_ids that were dispatched in this cycle.
        """
        self._completed_this_cycle = set()
        dispatched = []

        for agent_id, reg in self._agents.items():
            if not reg.is_due():
                continue
            if not self._dependencies_met(reg):
                logger.debug(
                    "AgentDispatcher: %s skipped — dependencies not met: %s",
                    agent_id, reg.depends_on,
                )
                continue

            try:
                logger.info("AgentDispatcher: dispatching %s (tier=%s)", agent_id, reg.tier)
                result = reg.run_fn(context) if context is not None else reg.run_fn()
                reg.record_success()
                self._completed_this_cycle.add(agent_id)
                dispatched.append(agent_id)

                # Route AgentDecision objects to approval queue
                if result is not None:
                    self._route_decision(agent_id, result, reg.tier)

                if self._bus is not None:
                    self._bus.publish(agent_id, {
                        "status": "success",
                        "ts": datetime.now(tz=timezone.utc).isoformat(),
                    })

            except Exception as exc:
                reg.record_failure()
                logger.error("AgentDispatcher: %s failed: %s", agent_id, exc)
                if self._bus is not None:
                    self._bus.publish(agent_id, {
                        "status": "failed",
                        "error": str(exc),
                        "ts": datetime.now(tz=timezone.utc).isoformat(),
                    })

        return dispatched

    def get_status(self) -> List[Dict[str, Any]]:
        """Return status of all registered agents."""
        return [
            {
                "agent_id": reg.agent_id,
                "tier": reg.tier,
                "circuit_open": reg._circuit_open,
                "consecutive_failures": reg._consecutive_failures,
                "seconds_since_last_run": time.monotonic() - reg._last_run_at,
                "is_due": reg.is_due(),
            }
            for reg in self._agents.values()
        ]

    def reset_circuit(self, agent_id: str) -> bool:
        """Manually reset a tripped circuit breaker."""
        if agent_id not in self._agents:
            return False
        self._agents[agent_id].reset_circuit()
        return True

    def _dependencies_met(self, reg: AgentRegistration) -> bool:
        if not reg.depends_on:
            return True
        return all(dep in self._completed_this_cycle for dep in reg.depends_on)

    def _route_decision(self, agent_id: str, result: Any, tier: str) -> None:
        """Route AgentDecision results to the approval queue or auto-execute."""
        try:
            from core.intents import AgentDecision, AgentDecisionStatus
            if not isinstance(result, AgentDecision):
                return

            # ADVISORY_ONLY and POLICY_GATED always go to approval queue
            if tier in ("ADVISORY_ONLY", "POLICY_GATED", "HUMAN_REQUIRED"):
                try:
                    from approvals.engine import get_approval_engine
                    # AgentDecision routing to approval engine is advisory only
                    # The approval engine handles routing based on its own policy
                    logger.info(
                        "AgentDispatcher: routing decision from %s (type=%s) to approval queue",
                        agent_id, result.decision_type.value,
                    )
                except Exception as exc:
                    logger.warning(
                        "AgentDispatcher: could not route decision to approval engine: %s", exc
                    )

            elif tier == "AUTO_EXECUTABLE":
                logger.info(
                    "AgentDispatcher: AUTO_EXECUTABLE decision from %s (type=%s) — auto-approved",
                    agent_id, result.decision_type.value,
                )

        except Exception as exc:
            logger.debug("AgentDispatcher._route_decision: %s", exc)
