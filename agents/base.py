# -*- coding: utf-8 -*-
"""
agents/base.py — Agent Protocol, Base Class, and Audit Logging
==============================================================

Defines the contract that every agent in this system must satisfy:
  - Typed AgentTask / AgentResult (imported from core.contracts)
  - BaseAgent abstract class with built-in audit trail and error handling
  - AgentAuditLogger: structured logging for every agent action

Design principles:
  - Narrow mandate: each agent does exactly one job
  - Explicit audit trail: every significant decision is logged
  - No implicit side-effects: agents return results, never modify shared state directly
  - Fail-safe: exceptions are caught, converted to FAILED AgentResult with details

Usage:
    class MyAgent(BaseAgent):
        NAME = "my_agent"
        ALLOWED_TASK_TYPES = {"do_thing"}

        def _execute(self, task: AgentTask) -> dict:
            ...
            return {"result": "value"}
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar

from core.contracts import AgentResult, AgentStatus, AgentTask

logger = logging.getLogger("agents.base")


# ── Audit Logger ──────────────────────────────────────────────────

class AgentAuditLogger:
    """
    Structured audit trail builder for agent execution.

    Each agent receives one per task. Records a chronological log of
    every significant decision, data access, or state change.
    """

    def __init__(self, task_id: str, agent_name: str):
        self.task_id = task_id
        self.agent_name = agent_name
        self._entries: list[str] = []
        self._start_time = time.monotonic()

    def log(self, message: str, level: str = "INFO") -> None:
        elapsed = time.monotonic() - self._start_time
        entry = f"[{elapsed:.3f}s] [{level}] {message}"
        self._entries.append(entry)
        if level == "ERROR":
            logger.error("%s/%s: %s", self.agent_name, self.task_id, message)
        elif level == "WARNING":
            logger.warning("%s/%s: %s", self.agent_name, self.task_id, message)
        else:
            logger.debug("%s/%s: %s", self.agent_name, self.task_id, message)

    def warn(self, message: str) -> None:
        self.log(message, level="WARNING")

    def error(self, message: str) -> None:
        self.log(message, level="ERROR")

    def entries(self) -> list[str]:
        return list(self._entries)

    def summary(self) -> str:
        elapsed = time.monotonic() - self._start_time
        return f"{self.agent_name}/{self.task_id}: {len(self._entries)} entries in {elapsed:.2f}s"


# ── Base Agent ────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base for all agents in this system.

    Subclasses must define:
      - NAME: str — unique agent identifier
      - ALLOWED_TASK_TYPES: set[str] — task types this agent handles
      - _execute(task, audit) -> dict — the actual implementation

    The base class provides:
      - Input validation (task type, required payload keys)
      - Automatic audit trail creation and capture
      - Exception isolation: all errors become FAILED AgentResult
      - Timing and metadata on every result
    """

    NAME: ClassVar[str] = "base_agent"
    ALLOWED_TASK_TYPES: ClassVar[set[str]] = set()
    REQUIRED_PAYLOAD_KEYS: ClassVar[set[str]] = set()

    def __init__(self):
        self._log = logging.getLogger(f"agents.{self.NAME}")

    def execute(self, task: AgentTask) -> AgentResult:
        """
        Execute a task. Returns AgentResult in all cases (never raises).

        Validates task type and payload keys, runs _execute(), handles exceptions.
        """
        audit = AgentAuditLogger(task.task_id, self.NAME)
        start = time.monotonic()

        # Validate task type
        if self.ALLOWED_TASK_TYPES and task.task_type not in self.ALLOWED_TASK_TYPES:
            audit.error(
                f"Task type '{task.task_type}' not in allowed types: {self.ALLOWED_TASK_TYPES}"
            )
            return AgentResult(
                task_id=task.task_id,
                agent_name=self.NAME,
                status=AgentStatus.FAILED,
                output={},
                error=f"Unsupported task type: {task.task_type}",
                audit_trail=audit.entries(),
                duration_ms=(time.monotonic() - start) * 1000,
            )

        # Validate required payload keys
        missing = self.REQUIRED_PAYLOAD_KEYS - set(task.payload.keys())
        if missing:
            audit.error(f"Missing required payload keys: {missing}")
            return AgentResult(
                task_id=task.task_id,
                agent_name=self.NAME,
                status=AgentStatus.FAILED,
                output={},
                error=f"Missing payload keys: {missing}",
                audit_trail=audit.entries(),
                duration_ms=(time.monotonic() - start) * 1000,
            )

        audit.log(f"Starting task type='{task.task_type}' payload_keys={list(task.payload.keys())}")

        try:
            output = self._execute(task, audit)
            elapsed = time.monotonic() - start
            audit.log(f"Completed in {elapsed:.2f}s, output_keys={list(output.keys()) if output else []}")

            return AgentResult(
                task_id=task.task_id,
                agent_name=self.NAME,
                status=AgentStatus.COMPLETED,
                output=output or {},
                audit_trail=audit.entries(),
                duration_ms=elapsed * 1000,
            )

        except Exception as exc:
            elapsed = time.monotonic() - start
            tb = traceback.format_exc()
            audit.error(f"Exception: {exc}\n{tb}")
            self._log.exception("Agent %s failed on task %s", self.NAME, task.task_id)

            return AgentResult(
                task_id=task.task_id,
                agent_name=self.NAME,
                status=AgentStatus.FAILED,
                output={},
                error=str(exc),
                audit_trail=audit.entries(),
                duration_ms=elapsed * 1000,
            )

    @abstractmethod
    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        """
        Implement agent logic here.

        Parameters
        ----------
        task : AgentTask
        audit : AgentAuditLogger — call audit.log() for every significant decision

        Returns
        -------
        dict — output payload (will be attached to AgentResult.output)

        Raises
        ------
        Any exception — will be caught by execute() and converted to FAILED result.
        """

    def create_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        *,
        priority: int = 0,
    ) -> AgentTask:
        """Convenience: create a properly-typed task for this agent."""
        return AgentTask(
            task_id=str(uuid.uuid4()),
            agent_name=self.NAME,
            task_type=task_type,
            payload=payload,
            priority=priority,
            created_at=datetime.utcnow(),
        )

    @property
    def name(self) -> str:
        return self.NAME


# ── Concrete agent stubs ──────────────────────────────────────────

class UniverseDiscoveryAgent(BaseAgent):
    """
    Discovers candidate pairs from a universe of symbols.

    Task type: "discover_pairs"
    Payload:
      - symbols: list[str]
      - prices: pd.DataFrame (or path to data)
      - max_pairs: int (optional, default 100)
      - min_correlation: float (optional, default 0.6)

    Output:
      - candidate_pairs: list of {sym_x, sym_y, correlation, distance}
      - n_candidates: int
    """
    NAME = "universe_discovery"
    ALLOWED_TASK_TYPES = {"discover_pairs"}
    REQUIRED_PAYLOAD_KEYS = {"symbols"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict:
        import pandas as pd
        from itertools import combinations

        symbols: list[str] = task.payload["symbols"]
        prices: pd.DataFrame = task.payload.get("prices", pd.DataFrame())
        max_pairs: int = task.payload.get("max_pairs", 100)
        min_corr: float = task.payload.get("min_correlation", 0.60)

        audit.log(f"Discovering pairs from {len(symbols)} symbols, min_corr={min_corr}")

        if prices.empty:
            audit.warn("No prices provided — returning empty candidate list")
            return {"candidate_pairs": [], "n_candidates": 0}

        # Filter to available columns
        available = [s for s in symbols if s in prices.columns]
        audit.log(f"{len(available)}/{len(symbols)} symbols have price data")

        if len(available) < 2:
            return {"candidate_pairs": [], "n_candidates": 0}

        # Compute returns correlation
        returns = prices[available].pct_change().dropna()
        corr = returns.corr()

        candidates = []
        for sx, sy in combinations(available, 2):
            c = corr.loc[sx, sy]
            if abs(c) >= min_corr:
                candidates.append({
                    "sym_x": min(sx, sy),
                    "sym_y": max(sx, sy),
                    "correlation": float(c),
                })

        # Sort by correlation descending, cap at max_pairs
        candidates.sort(key=lambda x: -abs(x["correlation"]))
        candidates = candidates[:max_pairs]

        audit.log(f"Found {len(candidates)} candidates (capped at {max_pairs})")
        return {"candidate_pairs": candidates, "n_candidates": len(candidates)}


class PairValidationAgent(BaseAgent):
    """
    Validates candidate pairs using statistical tests.

    Task type: "validate_pairs"
    Payload:
      - pair_ids: list of {sym_x, sym_y} dicts or PairId objects
      - prices: pd.DataFrame
      - train_end: str (ISO date, optional)

    Output:
      - validated: list of validation report dicts
      - n_passed: int
      - n_failed: int
      - n_warned: int
    """
    NAME = "pair_validation"
    ALLOWED_TASK_TYPES = {"validate_pairs", "validate_single"}
    REQUIRED_PAYLOAD_KEYS = {"prices"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict:
        import pandas as pd
        from core.contracts import PairId
        from research.pair_validator import PairValidator

        prices: pd.DataFrame = task.payload["prices"]
        raw_pairs = task.payload.get("pair_ids", [])
        train_end = task.payload.get("train_end")

        validator = PairValidator()

        # Normalise pair IDs
        pair_ids = []
        for p in raw_pairs:
            if isinstance(p, PairId):
                pair_ids.append(p)
            elif isinstance(p, dict):
                pair_ids.append(PairId(p["sym_x"], p["sym_y"]))
            else:
                audit.warn(f"Unrecognised pair format: {p}")

        audit.log(f"Validating {len(pair_ids)} pairs, train_end={train_end}")

        reports = validator.validate_batch(pair_ids, prices, train_end=train_end)

        n_passed = sum(1 for r in reports if r.result.value == "PASS")
        n_failed = sum(1 for r in reports if r.result.value == "FAIL")
        n_warned = sum(1 for r in reports if r.result.value == "WARN")

        audit.log(f"Results: {n_passed} PASS, {n_warned} WARN, {n_failed} FAIL")

        return {
            "validated": [r.__dict__ for r in reports],
            "n_passed": n_passed,
            "n_failed": n_failed,
            "n_warned": n_warned,
        }


class SpreadFitAgent(BaseAgent):
    """
    Fits spread definitions for validated pairs.

    Task type: "fit_spreads"
    Payload:
      - pair_ids: list of PairId or dicts
      - prices: pd.DataFrame
      - model: str (STATIC_OLS | ROLLING_OLS | KALMAN)
      - train_end: str ISO date (optional)
      - window: int (optional, default 60)

    Output:
      - spread_definitions: list of SpreadDefinition dicts
      - n_fitted: int
      - n_failed: int
    """
    NAME = "spread_fit"
    ALLOWED_TASK_TYPES = {"fit_spreads"}
    REQUIRED_PAYLOAD_KEYS = {"prices", "pair_ids"}

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict:
        import pandas as pd
        from core.contracts import PairId, SpreadModel
        from research.spread_constructor import build_spread

        prices: pd.DataFrame = task.payload["prices"]
        raw_pairs = task.payload["pair_ids"]
        model_str: str = task.payload.get("model", "STATIC_OLS").upper()
        train_end = task.payload.get("train_end")
        window: int = task.payload.get("window", 60)

        try:
            model = SpreadModel(model_str)
        except ValueError:
            raise ValueError(f"Invalid spread model: {model_str}")

        pair_ids = []
        for p in raw_pairs:
            if isinstance(p, PairId):
                pair_ids.append(p)
            elif isinstance(p, dict):
                pair_ids.append(PairId(p["sym_x"], p["sym_y"]))

        audit.log(f"Fitting {len(pair_ids)} pairs, model={model.value}, window={window}")

        definitions = []
        n_failed = 0
        for pid in pair_ids:
            try:
                defn, _ = build_spread(
                    pid, prices,
                    model=model,
                    train_end=train_end,
                    window=window,
                )
                definitions.append(defn.__dict__)
                audit.log(f"{pid.label}: beta={defn.hedge_ratio:.4f}")
            except Exception as exc:
                n_failed += 1
                audit.warn(f"{pid.label}: fit failed — {exc}")

        audit.log(f"Fitted {len(definitions)}, failed {n_failed}")
        return {
            "spread_definitions": definitions,
            "n_fitted": len(definitions),
            "n_failed": n_failed,
        }
