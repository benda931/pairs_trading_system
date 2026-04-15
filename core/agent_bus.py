# -*- coding: utf-8 -*-
"""
core/agent_bus.py — File-Based Agent Communication Bus
=======================================================

Extracted from orchestrator.py to break circular dependency (AP-4):
  orchestrator → agent_feedback → orchestrator (for bus access)

Now: orchestrator → agent_bus ← agent_feedback (both import bus, not each other)

Also exports TaskResult for shared use across orchestration + feedback layers.

Usage:
    from core.agent_bus import AgentBus, TaskResult

    bus = AgentBus()
    bus.publish("my_agent", {"key": "value"})
    latest = bus.latest("my_agent")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
AGENT_BUS_PATH = LOGS_DIR / "agent_bus.json"


class AgentBus:
    """File-based message bus for agent communication (no external deps)."""

    def __init__(self, path: Path = AGENT_BUS_PATH, max_history: int = 20):
        self.path = path
        self.max_history = max_history
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, list]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save(self, data: Dict[str, list]) -> None:
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def publish(self, agent: str, payload: Dict[str, Any]) -> None:
        data = self._load()
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        if agent not in data:
            data[agent] = []
        data[agent].append(entry)
        data[agent] = data[agent][-self.max_history:]
        self._save(data)

    def latest(self, agent: str) -> Optional[Dict[str, Any]]:
        data = self._load()
        entries = data.get(agent, [])
        return entries[-1] if entries else None

    def history(self, agent: str, n: int = 10) -> List[Dict[str, Any]]:
        data = self._load()
        return data.get(agent, [])[-n:]


@dataclass
class TaskResult:
    """Result from a single task execution."""
    task_name: str
    status: str  # "success" | "failed" | "skipped"
    duration_sec: float = 0.0
    error: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
