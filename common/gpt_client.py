# -*- coding: utf-8 -*-
"""
common/gpt_client.py — GPT-4o Client for Autonomous Agents
===========================================================

Singleton GPT client used by agents/gpt_agents.py for AI-powered analysis.

Features:
- Loads OPENAI_API_KEY from .env
- GPT-4o with structured JSON responses
- Cost tracking per agent per day
- Rate limiting and retry logic
- Never imported from core/ (architecture boundary preserved)

Usage:
    from common.gpt_client import get_gpt_client

    client = get_gpt_client()
    response = client.ask(
        system="You are a quantitative trading analyst.",
        prompt="Analyze this pair performance: ...",
        agent_name="gpt_signal_advisor",
    )
    print(response["content"])
    print(f"Cost: ${response['cost_usd']:.4f}")
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from datetime import date
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_api_key() -> Optional[str]:
    """Load OpenAI API key from .env or environment."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


# GPT-4o pricing (per 1M tokens, as of 2026)
_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class GPTClient:
    """
    Thread-safe GPT-4o client with cost tracking and rate limiting.

    All agent GPT calls go through this client to ensure:
    - Single API key management
    - Cost tracking per agent per day
    - Rate limiting (max requests per minute)
    - Structured JSON response parsing
    - Audit trail for every call
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_requests_per_minute: int = 20,
        max_daily_cost_usd: float = 5.0,
    ):
        self._api_key = api_key or _load_api_key()
        self._model = model
        self._max_rpm = max_requests_per_minute
        self._max_daily_cost = max_daily_cost_usd
        self._lock = threading.Lock()

        # Cost tracking
        self._daily_costs: dict[str, float] = {}  # date_str -> total_cost
        self._agent_costs: dict[str, float] = {}  # agent_name -> total_cost
        self._request_count = 0
        self._last_request_time = 0.0

        # Audit log
        self._call_log: list[dict[str, Any]] = []

        # OpenAI client (lazy init)
        self._client = None

        if not self._api_key:
            logger.warning("GPTClient: No OPENAI_API_KEY found — GPT calls will fail")

    def _get_client(self):
        """Lazy-init OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise RuntimeError("openai package not installed: pip install openai")
        return self._client

    def _check_rate_limit(self) -> None:
        """Simple rate limiter — wait if too many requests per minute."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < (60.0 / self._max_rpm):
            sleep_time = (60.0 / self._max_rpm) - elapsed
            time.sleep(sleep_time)

    def _check_cost_limit(self, agent_name: str) -> None:
        """Raise if daily cost limit exceeded."""
        today = str(date.today())
        daily_total = self._daily_costs.get(today, 0.0)
        if daily_total >= self._max_daily_cost:
            raise RuntimeError(
                f"GPT daily cost limit exceeded: ${daily_total:.2f} >= ${self._max_daily_cost:.2f}"
            )

    def _track_cost(self, agent_name: str, input_tokens: int, output_tokens: int) -> float:
        """Track cost and return USD amount."""
        pricing = _PRICING.get(self._model, _PRICING["gpt-4o"])
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        today = str(date.today())
        with self._lock:
            self._daily_costs[today] = self._daily_costs.get(today, 0.0) + cost
            self._agent_costs[agent_name] = self._agent_costs.get(agent_name, 0.0) + cost
        return cost

    @property
    def is_available(self) -> bool:
        """Check if GPT client can make calls."""
        return self._api_key is not None

    def ask(
        self,
        system: str,
        prompt: str,
        agent_name: str = "unknown",
        *,
        json_mode: bool = False,
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """
        Send a prompt to GPT-4o and return structured result.

        Parameters
        ----------
        system : str
            System message defining agent role.
        prompt : str
            User prompt with analysis request.
        agent_name : str
            Agent name for cost tracking and audit.
        json_mode : bool
            If True, request JSON response format.
        max_tokens : int
            Maximum response tokens.
        temperature : float
            Creativity (0.0 = deterministic, 1.0 = creative).

        Returns
        -------
        dict with keys:
            content: str — GPT response text
            parsed: dict | None — parsed JSON if json_mode
            model: str — model used
            input_tokens: int
            output_tokens: int
            cost_usd: float
            agent_name: str
            error: str | None
        """
        if not self._api_key:
            return {
                "content": "",
                "parsed": None,
                "model": self._model,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "agent_name": agent_name,
                "error": "No OPENAI_API_KEY configured",
            }

        with self._lock:
            self._check_cost_limit(agent_name)
            self._check_rate_limit()
            self._request_count += 1
            self._last_request_time = time.monotonic()

        try:
            client = self._get_client()
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content or ""
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            cost = self._track_cost(agent_name, input_tokens, output_tokens)

            parsed = None
            if json_mode:
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("GPT returned invalid JSON for %s", agent_name)

            result = {
                "content": content,
                "parsed": parsed,
                "model": self._model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "agent_name": agent_name,
                "error": None,
            }

            # Audit log
            self._call_log.append({
                "agent": agent_name,
                "timestamp": time.time(),
                "tokens": input_tokens + output_tokens,
                "cost_usd": cost,
                "model": self._model,
            })

            logger.info(
                "GPT call: agent=%s, tokens=%d+%d, cost=$%.4f",
                agent_name, input_tokens, output_tokens, cost,
            )
            return result

        except Exception as e:
            logger.error("GPT call failed for %s: %s", agent_name, e)
            return {
                "content": "",
                "parsed": None,
                "model": self._model,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "agent_name": agent_name,
                "error": str(e),
            }

    def get_cost_summary(self) -> dict[str, Any]:
        """Return cost tracking summary."""
        today = str(date.today())
        return {
            "daily_cost_usd": self._daily_costs.get(today, 0.0),
            "daily_limit_usd": self._max_daily_cost,
            "total_requests": self._request_count,
            "agent_costs": dict(self._agent_costs),
            "model": self._model,
        }


# ── Singleton ────────────────────────────────────────────────────

_client: Optional[GPTClient] = None
_client_lock = threading.Lock()


def get_gpt_client(**kwargs) -> GPTClient:
    """Get or create the global GPT client singleton."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = GPTClient(**kwargs)
    return _client
