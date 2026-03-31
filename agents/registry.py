# -*- coding: utf-8 -*-
"""
agents/registry.py — Agent Registry
====================================

Central registry that:
  1. Registers available agents by name
  2. Routes AgentTask objects to the correct agent
  3. Enforces permission scopes (read-only vs full-access agents)
  4. Persists an audit log of every executed task

Usage:
    registry = AgentRegistry()
    registry.register(UniverseDiscoveryAgent())
    registry.register(PairValidationAgent())

    task = AgentTask(task_id="...", agent_name="universe_discovery", ...)
    result = registry.dispatch(task)
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

from core.contracts import AgentResult, AgentStatus, AgentTask
from agents.base import BaseAgent

logger = logging.getLogger("agents.registry")


# ── Permission system ─────────────────────────────────────────────

class AgentPermission:
    """Permission scope for an agent."""
    READ_ONLY = "read_only"        # Can read data, never write
    RESEARCH = "research"          # Can write research artefacts
    EXECUTION = "execution"        # Can submit orders (requires explicit grant)
    ADMIN = "admin"                # Can modify system configuration


# ── Registry ──────────────────────────────────────────────────────

class AgentRegistry:
    """
    Central registry for agent discovery and dispatch.

    Thread-safe: multiple agents can execute concurrently.
    Maintains a bounded in-memory audit log (last N executions).
    """

    def __init__(self, max_audit_log: int = 10_000):
        self._agents: dict[str, BaseAgent] = {}
        self._permissions: dict[str, set[str]] = defaultdict(set)
        self._audit_log: list[dict[str, Any]] = []
        self._max_audit_log = max_audit_log
        self._lock = threading.Lock()

    def register(
        self,
        agent: BaseAgent,
        permissions: Optional[set[str]] = None,
    ) -> None:
        """
        Register an agent.

        Parameters
        ----------
        agent : BaseAgent instance
        permissions : set of AgentPermission constants (default: READ_ONLY)
        """
        name = agent.NAME
        with self._lock:
            if name in self._agents:
                logger.warning("Agent '%s' is already registered — overwriting", name)
            self._agents[name] = agent
            self._permissions[name] = permissions or {AgentPermission.READ_ONLY}
            logger.info(
                "Registered agent '%s' with permissions: %s",
                name, self._permissions[name],
            )

    def unregister(self, name: str) -> None:
        """Remove a registered agent."""
        with self._lock:
            self._agents.pop(name, None)
            self._permissions.pop(name, None)
            logger.info("Unregistered agent '%s'", name)

    def dispatch(self, task: AgentTask) -> AgentResult:
        """
        Route an AgentTask to the appropriate registered agent.

        Returns AgentResult in all cases — never raises.
        Appends to audit log regardless of success/failure.
        """
        agent = self._agents.get(task.agent_name)

        if agent is None:
            error_msg = (
                f"No agent registered for '{task.agent_name}'. "
                f"Available: {list(self._agents.keys())}"
            )
            logger.error(error_msg)
            result = AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status=AgentStatus.FAILED,
                output={},
                error=error_msg,
                audit_trail=[error_msg],
            )
        else:
            result = agent.execute(task)

        self._append_audit(task, result)
        return result

    def dispatch_batch(self, tasks: list[AgentTask]) -> list[AgentResult]:
        """Dispatch multiple tasks sequentially."""
        return [self.dispatch(t) for t in tasks]

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)

    def list_agents(self) -> list[dict[str, Any]]:
        """Return a summary of all registered agents."""
        with self._lock:
            return [
                {
                    "name": name,
                    "class": type(agent).__name__,
                    "allowed_task_types": list(agent.ALLOWED_TASK_TYPES),
                    "permissions": list(self._permissions[name]),
                }
                for name, agent in self._agents.items()
            ]

    def has_permission(self, agent_name: str, permission: str) -> bool:
        return permission in self._permissions.get(agent_name, set())

    def get_audit_log(
        self,
        agent_name: Optional[str] = None,
        status: Optional[AgentStatus] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit log with optional filters."""
        with self._lock:
            entries = list(self._audit_log)

        if agent_name:
            entries = [e for e in entries if e["agent_name"] == agent_name]
        if status:
            entries = [e for e in entries if e["status"] == status.value]

        return entries[-limit:]

    def _append_audit(self, task: AgentTask, result: AgentResult) -> None:
        entry = {
            "task_id": task.task_id,
            "agent_name": task.agent_name,
            "task_type": task.task_type,
            "status": result.status.value,
            "duration_seconds": result.duration_ms / 1000.0 if result.duration_ms else 0.0,
            "error": result.error,
            "dispatched_at": datetime.utcnow().isoformat(),
            "n_audit_entries": len(result.audit_trail),
        }
        with self._lock:
            self._audit_log.append(entry)
            if len(self._audit_log) > self._max_audit_log:
                # Trim oldest entries
                self._audit_log = self._audit_log[-self._max_audit_log:]


# ── Default registry singleton ────────────────────────────────────

_default_registry: Optional[AgentRegistry] = None
_registry_lock = threading.Lock()


def get_default_registry() -> AgentRegistry:
    """Get or create the default global agent registry."""
    global _default_registry
    with _registry_lock:
        if _default_registry is None:
            _default_registry = AgentRegistry()
            _populate_default_registry(_default_registry)
    return _default_registry


def _populate_default_registry(registry: AgentRegistry) -> None:
    """Register all built-in agents with appropriate permissions."""
    # ── Core base agents (from agents.base) ───────────────────────
    from agents.base import (
        PairValidationAgent,
        SpreadFitAgent,
        UniverseDiscoveryAgent,
    )

    registry.register(
        UniverseDiscoveryAgent(),
        permissions={AgentPermission.READ_ONLY},
    )
    registry.register(
        PairValidationAgent(),
        permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
    )
    registry.register(
        SpreadFitAgent(),
        permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
    )

    # ── Signal agents (from agents.signal_agents) ─────────────────
    try:
        from agents.signal_agents import (
            ExitOversightAgent,
            RegimeSurveillanceAgent,
            SignalAnalystAgent,
            TradeLifecycleAgent,
        )
        registry.register(
            SignalAnalystAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            RegimeSurveillanceAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
        registry.register(
            TradeLifecycleAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            ExitOversightAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
    except ImportError as exc:
        logger.warning("Could not register signal_agents: %s", exc)

    # ── Portfolio agents (from agents.portfolio_agents) ───────────
    try:
        from agents.portfolio_agents import (
            CapitalBudgetAgent,
            DeRiskingAgent,
            DrawdownMonitorAgent,
            ExposureMonitorAgent,
            KillSwitchAgent,
            PortfolioConstructionAgent,
        )
        registry.register(
            PortfolioConstructionAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            CapitalBudgetAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            ExposureMonitorAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
        registry.register(
            DrawdownMonitorAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
        registry.register(
            KillSwitchAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
        registry.register(
            DeRiskingAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
    except ImportError as exc:
        logger.warning("Could not register portfolio_agents: %s", exc)

    # ── Research agents (from agents.research_agents) ─────────────
    try:
        from agents.research_agents import (
            CandidateDiscoveryAgent,
            ExperimentCoordinatorAgent,
            RegimeResearchAgent,
            RelationshipValidationAgent,
            ResearchSummarizationAgent,
            SignalResearchAgent,
            SpreadSpecificationAgent,
            UniverseCuratorAgent,
        )
        registry.register(
            UniverseCuratorAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            CandidateDiscoveryAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            RelationshipValidationAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            SpreadSpecificationAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            RegimeResearchAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            SignalResearchAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            ExperimentCoordinatorAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            ResearchSummarizationAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
    except ImportError as exc:
        logger.warning("Could not register research_agents: %s", exc)

    # ── ML agents (from agents.ml_agents) ─────────────────────────
    try:
        from agents.ml_agents import (
            FeatureStewardAgent,
            LabelGovernanceAgent,
            MetaLabelingAgent,
            ModelResearchAgent,
            ModelRiskAgent,
            PromotionReviewAgent,
            RegimeModelingAgent,
        )
        registry.register(
            FeatureStewardAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            LabelGovernanceAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            ModelResearchAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            MetaLabelingAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            RegimeModelingAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            ModelRiskAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            PromotionReviewAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
    except ImportError as exc:
        logger.warning("Could not register ml_agents: %s", exc)

    # ── Monitoring agents (from agents.monitoring_agents) ─────────
    try:
        from agents.monitoring_agents import (
            AlertAggregationAgent,
            DataIntegrityAgent,
            DriftMonitoringAgent,
            IncidentTriageAgent,
            OrchestrationReliabilityAgent,
            PostmortemDraftingAgent,
            SystemHealthAgent,
        )
        registry.register(
            SystemHealthAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
        registry.register(
            DriftMonitoringAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
        registry.register(
            DataIntegrityAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
        registry.register(
            OrchestrationReliabilityAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
        registry.register(
            IncidentTriageAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
        registry.register(
            PostmortemDraftingAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
        registry.register(
            AlertAggregationAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
    except ImportError as exc:
        logger.warning("Could not register monitoring_agents: %s", exc)

    # ── Governance agents (from agents.governance_agents) ─────────
    try:
        from agents.governance_agents import (
            ApprovalRecommendationAgent,
            AuditTrailValidationAgent,
            ChangeImpactAgent,
            PolicyReviewAgent,
            PromotionGateAgent,
        )
        registry.register(
            PolicyReviewAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
        registry.register(
            AuditTrailValidationAgent(),
            permissions={AgentPermission.READ_ONLY},
        )
        registry.register(
            ApprovalRecommendationAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            ChangeImpactAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
        registry.register(
            PromotionGateAgent(),
            permissions={AgentPermission.READ_ONLY, AgentPermission.RESEARCH},
        )
    except ImportError as exc:
        logger.warning("Could not register governance_agents: %s", exc)

    logger.info("Default registry populated with %d agents", len(registry._agents))
