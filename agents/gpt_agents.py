# -*- coding: utf-8 -*-
"""
agents/gpt_agents.py — GPT-4o Powered Analysis Agents
======================================================

Agents that use GPT-4o for intelligent analysis and recommendations.
These agents are READ_ONLY + RESEARCH — they analyze and recommend
but never execute changes directly.

Agents:
- GPTSignalAdvisor: Analyzes pair performance, recommends parameter changes
- GPTModelTuner: Analyzes ML model metrics, recommends retraining strategy
- GPTStrategyResearcher: Discovers new pair opportunities via AI analysis
- GPTReportGenerator: Generates daily improvement reports
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agents.base import BaseAgent, AgentAuditLogger
from core.contracts import AgentTask

logger = logging.getLogger(__name__)


def _get_gpt():
    """Lazy import to avoid circular deps."""
    from common.gpt_client import get_gpt_client
    return get_gpt_client()


class GPTSignalAdvisor(BaseAgent):
    """
    Uses GPT-4o to analyze pair signal performance and recommend
    parameter adjustments (z_open, z_close, stop_z, lookback, etc.)

    Task types: gpt_signal_advisor.analyze
    Payload: pairs_summary (dict of pair metrics)
    """

    NAME = "gpt_signal_advisor"
    ALLOWED_TASK_TYPES = {"gpt_signal_advisor.analyze"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        gpt = _get_gpt()
        if not gpt.is_available:
            audit.warn("GPT not available — returning empty recommendations")
            return {"recommendations": [], "error": "GPT not configured"}

        pairs_summary = task.payload.get("pairs_summary", {})
        audit.log(f"Analyzing {len(pairs_summary)} pairs for signal optimization")

        response = gpt.ask(
            system=(
                "You are a quantitative trading analyst specializing in statistical "
                "arbitrage pairs trading. Analyze the pair performance data and suggest "
                "specific parameter improvements. Respond in JSON with keys: "
                "recommendations (list of {pair, parameter, current_value, suggested_value, rationale}), "
                "summary (string)."
            ),
            prompt=f"Pair performance data:\n{json.dumps(pairs_summary, indent=2, default=str)[:3000]}",
            agent_name=self.NAME,
            json_mode=True,
            temperature=0.2,
        )

        if response["error"]:
            audit.warn(f"GPT error: {response['error']}")
            return {"recommendations": [], "error": response["error"],
                    "cost_usd": response["cost_usd"]}

        parsed = response.get("parsed") or {}
        recs = parsed.get("recommendations", [])
        audit.log(f"GPT produced {len(recs)} recommendations (cost: ${response['cost_usd']:.4f})")

        return {
            "recommendations": recs,
            "summary": parsed.get("summary", ""),
            "cost_usd": response["cost_usd"],
            "model": response["model"],
        }


class GPTModelTuner(BaseAgent):
    """
    Uses GPT-4o to analyze ML model metrics and recommend retraining.

    Task types: gpt_model_tuner.analyze
    Payload: model_metrics (dict with AUC, Brier, etc.)
    """

    NAME = "gpt_model_tuner"
    ALLOWED_TASK_TYPES = {"gpt_model_tuner.analyze"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        gpt = _get_gpt()
        if not gpt.is_available:
            audit.warn("GPT not available")
            return {"should_retrain": False, "error": "GPT not configured"}

        metrics = task.payload.get("model_metrics", {})
        audit.log(f"Analyzing model metrics: {metrics}")

        response = gpt.ask(
            system=(
                "You are an ML engineer specializing in meta-labeling models for pairs trading. "
                "Analyze the model performance metrics and decide if retraining is needed. "
                "Respond in JSON: {should_retrain: bool, reason: str, "
                "suggested_changes: [{parameter, value, rationale}], priority: 'high'|'medium'|'low'}"
            ),
            prompt=f"Current model metrics:\n{json.dumps(metrics, indent=2, default=str)}",
            agent_name=self.NAME,
            json_mode=True,
            temperature=0.1,
        )

        parsed = response.get("parsed") or {}
        should_retrain = parsed.get("should_retrain", False)
        audit.log(f"GPT recommendation: retrain={should_retrain}, priority={parsed.get('priority', '?')}")

        return {
            "should_retrain": should_retrain,
            "reason": parsed.get("reason", ""),
            "suggested_changes": parsed.get("suggested_changes", []),
            "priority": parsed.get("priority", "low"),
            "cost_usd": response["cost_usd"],
        }


class GPTStrategyResearcher(BaseAgent):
    """
    Uses GPT-4o to research new pair trading opportunities.

    Task types: gpt_strategy_researcher.research
    Payload: universe_summary, market_context
    """

    NAME = "gpt_strategy_researcher"
    ALLOWED_TASK_TYPES = {"gpt_strategy_researcher.research"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        gpt = _get_gpt()
        if not gpt.is_available:
            return {"ideas": [], "error": "GPT not configured"}

        universe = task.payload.get("universe_summary", {})
        context = task.payload.get("market_context", "")
        audit.log("Researching new pair opportunities")

        response = gpt.ask(
            system=(
                "You are a senior quant researcher in statistical arbitrage. "
                "Given the current universe of pairs and market context, suggest "
                "new pair trading ideas or improvements to existing strategies. "
                "Respond in JSON: {ideas: [{pair, thesis, expected_edge, risk, "
                "validation_steps}], market_outlook: str}"
            ),
            prompt=(
                f"Current universe: {json.dumps(universe, default=str)[:2000]}\n"
                f"Market context: {context[:1000]}"
            ),
            agent_name=self.NAME,
            json_mode=True,
            temperature=0.4,
        )

        parsed = response.get("parsed") or {}
        ideas = parsed.get("ideas", [])
        audit.log(f"GPT generated {len(ideas)} research ideas")

        return {
            "ideas": ideas,
            "market_outlook": parsed.get("market_outlook", ""),
            "cost_usd": response["cost_usd"],
        }


class GPTReportGenerator(BaseAgent):
    """
    Generates daily improvement reports summarizing all agent activity.

    Task types: gpt_report_generator.generate
    Payload: agent_results (list of recent agent outputs)
    """

    NAME = "gpt_report_generator"
    ALLOWED_TASK_TYPES = {"gpt_report_generator.generate"}
    REQUIRED_PAYLOAD_KEYS: set[str] = set()

    def _execute(self, task: AgentTask, audit: AgentAuditLogger) -> dict[str, Any]:
        gpt = _get_gpt()
        if not gpt.is_available:
            return {"report": "GPT not available", "error": "GPT not configured"}

        results = task.payload.get("agent_results", [])
        audit.log(f"Generating report from {len(results)} agent results")

        results_summary = json.dumps(results, default=str)[:4000]

        response = gpt.ask(
            system=(
                "You are a trading desk operations manager. Generate a concise daily "
                "report summarizing autonomous agent activity. Include: key findings, "
                "actions taken, model updates, parameter changes, data quality issues, "
                "and recommendations for human review. Format as markdown."
            ),
            prompt=f"Agent activity summary:\n{results_summary}",
            agent_name=self.NAME,
            json_mode=False,
            max_tokens=3000,
            temperature=0.3,
        )

        audit.log(f"Report generated ({len(response['content'])} chars)")

        return {
            "report": response["content"],
            "cost_usd": response["cost_usd"],
        }
