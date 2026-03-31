# -*- coding: utf-8 -*-
"""
orchestration/engine.py — WorkflowEngine
==========================================

Executes WorkflowDefinition objects as tracked WorkflowRun instances.

Features
--------
* Dependency-aware topological step ordering
* Per-step timeout enforcement (wall-clock, checked between retries)
* Bounded retries per step
* Approval gate integration via an injected callback
* Artefact collection from step outputs
* Replay from a prior WorkflowRun (skips already-completed steps)
* Graceful cancellation via threading.Event
* Full WorkflowOutcome produced on terminal state
* In-memory run store (``dict[run_id, WorkflowRun]``)

No external dependencies beyond stdlib.  Import numpy / pandas only if
pre-built workflow factories require them for payload construction.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.contracts import AgentStatus, AgentTask
from orchestration.contracts import (
    DelegationRecord,
    EnvironmentClass,
    FailureClass,
    FailureRecord,
    RiskClass,
    WorkflowDefinition,
    WorkflowOutcome,
    WorkflowRun,
    WorkflowStatus,
    WorkflowStep,
    WorkflowStepRun,
    WorkflowStepStatus,
    WorkflowTransition,
)

logger = logging.getLogger("orchestration.engine")


# ── Helpers ───────────────────────────────────────────────────────


def _utc_now() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _elapsed_ms(start: float) -> float:
    """Return milliseconds elapsed since *start* (``time.monotonic()`` value)."""
    return (time.monotonic() - start) * 1_000.0


def _topological_sort(steps: tuple[WorkflowStep, ...]) -> list[WorkflowStep]:
    """
    Return *steps* in a valid execution order that respects all ``depends_on``
    constraints (Kahn's algorithm).

    Raises
    ------
    ValueError
        If a dependency cycle is detected or an unknown step_id is referenced.
    """
    step_map: dict[str, WorkflowStep] = {s.step_id: s for s in steps}
    unknown_refs = [
        (s.step_id, dep)
        for s in steps
        for dep in s.depends_on
        if dep not in step_map
    ]
    if unknown_refs:
        raise ValueError(
            f"WorkflowDefinition references unknown step IDs: {unknown_refs}"
        )

    # in-degree map
    in_degree: dict[str, int] = {s.step_id: 0 for s in steps}
    dependents: dict[str, list[str]] = {s.step_id: [] for s in steps}
    for s in steps:
        for dep in s.depends_on:
            in_degree[s.step_id] += 1
            dependents[dep].append(s.step_id)

    queue: list[str] = [sid for sid, deg in in_degree.items() if deg == 0]
    ordered: list[WorkflowStep] = []

    while queue:
        sid = queue.pop(0)
        ordered.append(step_map[sid])
        for child in dependents[sid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(ordered) != len(steps):
        cycle_nodes = [sid for sid, deg in in_degree.items() if deg > 0]
        raise ValueError(
            f"Dependency cycle detected in WorkflowDefinition involving "
            f"steps: {cycle_nodes}"
        )

    return ordered


def _build_failure_record(
    *,
    agent_name: str,
    task_id: str,
    workflow_run_id: Optional[str],
    failure_class: FailureClass,
    message: str,
    is_retryable: bool,
    retry_count: int,
    escalated: bool,
) -> FailureRecord:
    return FailureRecord(
        failure_id=str(uuid.uuid4()),
        agent_name=agent_name,
        task_id=task_id,
        workflow_run_id=workflow_run_id,
        failure_class=failure_class,
        message=message,
        is_retryable=is_retryable,
        retry_count=retry_count,
        escalated=escalated,
        timestamp=_utc_now(),
    )


# ── WorkflowEngine ────────────────────────────────────────────────


class WorkflowEngine:
    """
    Executes WorkflowDefinitions as WorkflowRun instances.

    Parameters
    ----------
    registry : AgentRegistry
        The agent registry used to dispatch AgentTask objects.
    approval_callback : callable(step, workflow_run) -> bool, optional
        Called when a step requires approval before or after execution.
        Should return True (approved) or False (rejected).
        If None, approval is auto-granted in RESEARCH / STAGING environments
        and rejected in PAPER / PRODUCTION.
    artifact_store : callable(run_id, step_id, artifacts: dict) -> list[str], optional
        If provided, called after each step completes to persist artefacts.
        Must return a list of artefact IDs.
    environment : EnvironmentClass
        The deployment environment.  Controls default approval behaviour.
    """

    def __init__(
        self,
        registry: Any,                              # agents.registry.AgentRegistry
        approval_callback: Optional[Callable] = None,
        artifact_store: Optional[Callable] = None,
        environment: EnvironmentClass = EnvironmentClass.RESEARCH,
    ) -> None:
        self._registry = registry
        self._approval_callback = approval_callback
        self._artifact_store = artifact_store
        self._environment = environment

        # run_id -> WorkflowRun (in-memory)
        self._runs: dict[str, WorkflowRun] = {}
        # run_id -> threading.Event (for cancellation)
        self._cancel_events: dict[str, threading.Event] = {}
        # run_id -> list[FailureRecord]
        self._failure_records: dict[str, list[FailureRecord]] = {}
        # run_id -> list[DelegationRecord]
        self._delegation_records: dict[str, list[DelegationRecord]] = {}

        self._lock = threading.Lock()

        logger.info(
            "WorkflowEngine initialised (environment=%s)", environment.value
        )

    # ── Public API ────────────────────────────────────────────────

    def run(
        self,
        definition: WorkflowDefinition,
        trigger_payload: dict,
        triggered_by: str = "manual",
        replay_of: Optional[str] = None,
    ) -> WorkflowRun:
        """
        Execute *definition* synchronously and return the completed WorkflowRun.

        Parameters
        ----------
        definition : WorkflowDefinition
        trigger_payload : dict
            Arbitrary input data forwarded to the first step's agent payload.
        triggered_by : str
            One of: ``"schedule"`` | ``"event"`` | ``"manual"`` |
            ``"agent"`` | ``"incident"``
        replay_of : str, optional
            If provided, the run_id of a prior run to replay from.  Steps
            that already completed in the prior run are skipped.

        Returns
        -------
        WorkflowRun
            The completed (or failed / cancelled) run object.

        Raises
        ------
        ValueError
            If *definition* has a dependency cycle or references unknown steps.
        RuntimeError
            If *replay_of* refers to an unknown run.
        """
        run_id = str(uuid.uuid4())
        cancel_event = threading.Event()

        wf_run = WorkflowRun(
            run_id=run_id,
            workflow_id=definition.workflow_id,
            workflow_name=definition.name,
            status=WorkflowStatus.RUNNING,
            environment=self._environment,
            triggered_by=triggered_by,
            trigger_payload=trigger_payload,
            started_at=_utc_now(),
            replay_of_run_id=replay_of,
        )

        with self._lock:
            self._runs[run_id] = wf_run
            self._cancel_events[run_id] = cancel_event
            self._failure_records[run_id] = []
            self._delegation_records[run_id] = []

        logger.info(
            "WorkflowRun %s started: workflow=%s triggered_by=%s",
            run_id, definition.workflow_id, triggered_by,
        )

        # Validate definition and determine step execution order
        try:
            ordered_steps = _topological_sort(definition.steps)
        except ValueError as exc:
            self._finalise_run(
                wf_run,
                WorkflowStatus.FAILED,
                error=f"WorkflowDefinition validation failed: {exc}",
            )
            return wf_run

        # Determine which steps to skip when replaying
        skip_step_ids: set[str] = set()
        if replay_of is not None:
            prior_run = self.get_run(replay_of)
            if prior_run is None:
                self._finalise_run(
                    wf_run,
                    WorkflowStatus.FAILED,
                    error=f"Replay target run_id={replay_of!r} not found",
                )
                return wf_run
            # Skip steps that completed successfully in the prior run
            for sr in prior_run.step_runs:
                if sr.status == WorkflowStepStatus.COMPLETED:
                    skip_step_ids.add(sr.step_id)
            logger.info(
                "Replay mode: skipping %d already-completed steps from run %s",
                len(skip_step_ids), replay_of,
            )

        wall_clock_start = time.monotonic()
        completed_step_ids: set[str] = set(skip_step_ids)

        # Pre-populate skipped step runs for replayed steps
        for step in ordered_steps:
            if step.step_id in skip_step_ids:
                sr = WorkflowStepRun(
                    step_run_id=str(uuid.uuid4()),
                    step_id=step.step_id,
                    workflow_run_id=run_id,
                    status=WorkflowStepStatus.SKIPPED,
                    started_at=None,
                    completed_at=_utc_now(),
                    agent_task_id=None,
                    agent_result_status=None,
                    output_summary="Skipped — completed in prior run (replay)",
                )
                wf_run.step_runs.append(sr)

        # Execute steps
        abort_workflow = False

        for step in ordered_steps:
            if step.step_id in skip_step_ids:
                continue  # already recorded as SKIPPED above

            # Check cancellation
            if cancel_event.is_set():
                logger.info(
                    "WorkflowRun %s: cancellation requested before step %s",
                    run_id, step.step_id,
                )
                self._skip_remaining_steps(
                    wf_run, ordered_steps, completed_step_ids, skip_step_ids
                )
                self._finalise_run(wf_run, WorkflowStatus.CANCELLED)
                return wf_run

            # Check workflow-level wall-clock timeout
            if _elapsed_ms(wall_clock_start) / 1_000.0 > definition.max_duration_seconds:
                logger.warning(
                    "WorkflowRun %s exceeded max_duration_seconds=%.1f",
                    run_id, definition.max_duration_seconds,
                )
                self._skip_remaining_steps(
                    wf_run, ordered_steps, completed_step_ids, skip_step_ids
                )
                self._finalise_run(
                    wf_run,
                    WorkflowStatus.FAILED,
                    error=(
                        f"Workflow exceeded max_duration_seconds="
                        f"{definition.max_duration_seconds:.1f}"
                    ),
                )
                return wf_run

            if abort_workflow:
                self._record_step_skipped(wf_run, step, run_id)
                continue

            # Check that all dependencies completed successfully
            unmet_deps = [
                dep for dep in step.depends_on
                if dep not in completed_step_ids
            ]
            if unmet_deps:
                logger.warning(
                    "WorkflowRun %s: step %s has unmet dependencies %s — skipping",
                    run_id, step.step_id, unmet_deps,
                )
                self._record_step_skipped(
                    wf_run, step, run_id,
                    reason=f"Unmet dependencies: {unmet_deps}",
                )
                continue

            # Execute the step (with retries)
            step_run, step_succeeded = self._execute_step(
                step=step,
                wf_run=wf_run,
                trigger_payload=trigger_payload,
                cancel_event=cancel_event,
            )
            wf_run.step_runs.append(step_run)

            if step_succeeded:
                completed_step_ids.add(step.step_id)
                # Collect artefact IDs
                wf_run.artifact_ids.extend(step_run.artifact_ids)
            else:
                # Step failed — apply on_failure policy
                if step.on_failure == "fail_workflow":
                    logger.error(
                        "WorkflowRun %s: step %s failed with on_failure=fail_workflow — aborting",
                        run_id, step.step_id,
                    )
                    abort_workflow = True
                elif step.on_failure == "skip":
                    logger.warning(
                        "WorkflowRun %s: step %s failed — skipping (on_failure=skip)",
                        run_id, step.step_id,
                    )
                    # Treat as non-blocking; do NOT add to completed_step_ids
                elif step.on_failure == "escalate":
                    logger.warning(
                        "WorkflowRun %s: step %s failed — escalating (on_failure=escalate)",
                        run_id, step.step_id,
                    )
                    # Record escalation; abort if in a restrictive environment
                    if self._environment in (
                        EnvironmentClass.PAPER, EnvironmentClass.PRODUCTION
                    ):
                        abort_workflow = True
                elif step.on_failure == "retry":
                    # Retries already exhausted in _execute_step; treat as fail_workflow
                    logger.error(
                        "WorkflowRun %s: step %s retry exhausted — aborting",
                        run_id, step.step_id,
                    )
                    abort_workflow = True

        if abort_workflow:
            self._finalise_run(
                wf_run,
                WorkflowStatus.FAILED,
                error="One or more steps failed with on_failure=fail_workflow",
            )
        else:
            self._finalise_run(wf_run, WorkflowStatus.COMPLETED)

        return wf_run

    def cancel(self, run_id: str) -> bool:
        """
        Request cancellation of a running workflow.

        Returns True if the cancel signal was set, False if the run was not
        found or was already in a terminal state.
        """
        with self._lock:
            event = self._cancel_events.get(run_id)
            run = self._runs.get(run_id)

        if event is None or run is None:
            logger.warning("cancel(): run_id=%s not found", run_id)
            return False

        if run.status not in (WorkflowStatus.RUNNING, WorkflowStatus.PAUSED):
            logger.info(
                "cancel(): run_id=%s already in terminal state %s",
                run_id, run.status.value,
            )
            return False

        event.set()
        logger.info("cancel(): cancellation signal sent to run_id=%s", run_id)
        return True

    def get_run(self, run_id: str) -> Optional[WorkflowRun]:
        """Return the WorkflowRun for *run_id*, or None if not found."""
        return self._runs.get(run_id)

    def list_runs(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
    ) -> list[WorkflowRun]:
        """
        Return all stored WorkflowRuns, optionally filtered by workflow_id
        and / or status.
        """
        with self._lock:
            runs = list(self._runs.values())

        if workflow_id is not None:
            runs = [r for r in runs if r.workflow_id == workflow_id]
        if status is not None:
            runs = [r for r in runs if r.status == status]
        return runs

    def get_outcome(self, run_id: str) -> Optional[WorkflowOutcome]:
        """
        Build and return a WorkflowOutcome for a terminal run.

        Returns None if the run is not found or is still in progress.
        """
        run = self.get_run(run_id)
        if run is None or run.status == WorkflowStatus.RUNNING:
            return None

        steps_completed = sum(
            1 for sr in run.step_runs
            if sr.status == WorkflowStepStatus.COMPLETED
        )
        steps_failed = sum(
            1 for sr in run.step_runs
            if sr.status == WorkflowStepStatus.FAILED
        )
        steps_skipped = sum(
            1 for sr in run.step_runs
            if sr.status == WorkflowStepStatus.SKIPPED
        )

        started_dt = datetime.fromisoformat(run.started_at)
        completed_str = run.completed_at or _utc_now()
        completed_dt = datetime.fromisoformat(completed_str)
        total_ms = (completed_dt - started_dt).total_seconds() * 1_000.0

        failure_records = self._failure_records.get(run_id, [])
        escalation_count = sum(1 for fr in failure_records if fr.escalated)

        return WorkflowOutcome(
            run_id=run_id,
            workflow_id=run.workflow_id,
            status=run.status,
            completed_at=completed_str,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            steps_skipped=steps_skipped,
            total_duration_ms=total_ms,
            artifact_count=len(run.artifact_ids),
            recommendation_count=len(run.recommendation_ids),
            approval_count=len(run.approval_request_ids),
            escalation_count=escalation_count,
            notes=run.notes,
        )

    def get_health_metrics(self) -> dict:
        """
        Return engine-level observability metrics.

        Returns a dict with keys:
          - ``total_runs`` (int)
          - ``runs_by_status`` (dict[str, int])
          - ``total_failures`` (int)
          - ``total_delegations`` (int)
          - ``environment`` (str)
          - ``registered_agents`` (list[str])
        """
        with self._lock:
            runs = list(self._runs.values())
            all_failures: list[FailureRecord] = []
            for fr_list in self._failure_records.values():
                all_failures.extend(fr_list)
            all_delegations: list[DelegationRecord] = []
            for dr_list in self._delegation_records.values():
                all_delegations.extend(dr_list)

        runs_by_status: dict[str, int] = {}
        for run in runs:
            key = run.status.value
            runs_by_status[key] = runs_by_status.get(key, 0) + 1

        registered = (
            [a["name"] for a in self._registry.list_agents()]
            if hasattr(self._registry, "list_agents")
            else []
        )

        return {
            "total_runs": len(runs),
            "runs_by_status": runs_by_status,
            "total_failures": len(all_failures),
            "total_delegations": len(all_delegations),
            "environment": self._environment.value,
            "registered_agents": registered,
        }

    # ── Internal step execution ───────────────────────────────────

    def _execute_step(
        self,
        step: WorkflowStep,
        wf_run: WorkflowRun,
        trigger_payload: dict,
        cancel_event: threading.Event,
    ) -> tuple[WorkflowStepRun, bool]:
        """
        Execute a single step (with retries).

        Returns ``(WorkflowStepRun, succeeded: bool)``.
        """
        step_run = WorkflowStepRun(
            step_run_id=str(uuid.uuid4()),
            step_id=step.step_id,
            workflow_run_id=wf_run.run_id,
            status=WorkflowStepStatus.PENDING,
            started_at=None,
            completed_at=None,
            agent_task_id=None,
            agent_result_status=None,
        )

        for attempt in range(step.retry_max + 1):
            if cancel_event.is_set():
                step_run.status = WorkflowStepStatus.FAILED
                step_run.error = "Cancelled before step could complete"
                step_run.completed_at = _utc_now()
                return step_run, False

            if attempt > 0:
                step_run.status = WorkflowStepStatus.RETRYING
                step_run.retry_count = attempt
                logger.info(
                    "WorkflowRun %s: retrying step %s (attempt %d/%d)",
                    wf_run.run_id, step.step_id, attempt, step.retry_max,
                )

            # Pre-execution approval gate
            if step.requires_approval_before:
                approved = self._request_approval(
                    step=step, wf_run=wf_run, gate="before"
                )
                wf_run.approval_request_ids.append(
                    f"{wf_run.run_id}:{step.step_id}:before:{attempt}"
                )
                if not approved:
                    logger.warning(
                        "WorkflowRun %s: step %s rejected at pre-execution approval gate",
                        wf_run.run_id, step.step_id,
                    )
                    failure = _build_failure_record(
                        agent_name=step.agent_name,
                        task_id=step_run.step_run_id,
                        workflow_run_id=wf_run.run_id,
                        failure_class=FailureClass.APPROVAL_REJECTED,
                        message=f"Step {step.step_id!r} rejected at pre-execution gate",
                        is_retryable=False,
                        retry_count=attempt,
                        escalated=False,
                    )
                    self._failure_records[wf_run.run_id].append(failure)
                    step_run.status = WorkflowStepStatus.FAILED
                    step_run.error = failure.message
                    step_run.completed_at = _utc_now()
                    return step_run, False

            # Build and dispatch the agent task
            task_id = str(uuid.uuid4())
            step_run.started_at = step_run.started_at or _utc_now()
            step_run.status = WorkflowStepStatus.RUNNING
            step_run.agent_task_id = task_id

            task = AgentTask(
                task_id=task_id,
                agent_name=step.agent_name,
                task_type=step.task_type,
                payload={**trigger_payload},
                requires_approval=step.requires_approval_before,
                correlation_id=wf_run.run_id,
            )

            logger.debug(
                "WorkflowRun %s: dispatching step %s -> agent %s (task %s)",
                wf_run.run_id, step.step_id, step.agent_name, task_id,
            )

            # Enforce per-step timeout via a result container + thread
            result_container: list = []
            error_container: list = []

            def _run_agent() -> None:
                try:
                    result = self._registry.dispatch(task)
                    result_container.append(result)
                except Exception as exc:  # pragma: no cover — defensive
                    error_container.append(exc)

            agent_thread = threading.Thread(target=_run_agent, daemon=True)
            agent_thread.start()
            agent_thread.join(timeout=step.timeout_seconds)

            if agent_thread.is_alive():
                # Timed out — the thread will eventually complete but we move on
                logger.warning(
                    "WorkflowRun %s: step %s timed out after %.1fs",
                    wf_run.run_id, step.step_id, step.timeout_seconds,
                )
                failure = _build_failure_record(
                    agent_name=step.agent_name,
                    task_id=task_id,
                    workflow_run_id=wf_run.run_id,
                    failure_class=FailureClass.TIMEOUT,
                    message=(
                        f"Step {step.step_id!r} timed out after "
                        f"{step.timeout_seconds:.1f}s"
                    ),
                    is_retryable=True,
                    retry_count=attempt,
                    escalated=False,
                )
                self._failure_records[wf_run.run_id].append(failure)
                step_run.error = failure.message
                # Try again if retries remain
                continue

            if error_container:
                # Unexpected exception from the dispatch layer
                exc = error_container[0]
                failure = _build_failure_record(
                    agent_name=step.agent_name,
                    task_id=task_id,
                    workflow_run_id=wf_run.run_id,
                    failure_class=FailureClass.DEPENDENCY_FAILURE,
                    message=f"Dispatch exception: {exc}",
                    is_retryable=True,
                    retry_count=attempt,
                    escalated=False,
                )
                self._failure_records[wf_run.run_id].append(failure)
                step_run.error = failure.message
                continue

            if not result_container:
                # Should not happen, but guard defensively
                failure = _build_failure_record(
                    agent_name=step.agent_name,
                    task_id=task_id,
                    workflow_run_id=wf_run.run_id,
                    failure_class=FailureClass.DEPENDENCY_FAILURE,
                    message="Agent thread completed but produced no result",
                    is_retryable=True,
                    retry_count=attempt,
                    escalated=False,
                )
                self._failure_records[wf_run.run_id].append(failure)
                step_run.error = failure.message
                continue

            agent_result = result_container[0]
            step_run.agent_result_status = agent_result.status.value

            if agent_result.status != AgentStatus.COMPLETED:
                failure = _build_failure_record(
                    agent_name=step.agent_name,
                    task_id=task_id,
                    workflow_run_id=wf_run.run_id,
                    failure_class=FailureClass.DEPENDENCY_FAILURE,
                    message=(
                        f"Agent returned status={agent_result.status.value}: "
                        f"{agent_result.error or '(no error message)'}"
                    ),
                    is_retryable=True,
                    retry_count=attempt,
                    escalated=False,
                )
                self._failure_records[wf_run.run_id].append(failure)
                step_run.error = failure.message
                continue

            # Step succeeded — collect artefacts
            artifact_ids = self._collect_artifacts(
                wf_run=wf_run,
                step=step,
                output=agent_result.output,
            )
            step_run.artifact_ids.extend(artifact_ids)

            # Build output summary
            output_keys = list(agent_result.output.keys()) if agent_result.output else []
            step_run.output_summary = (
                f"Completed: {len(output_keys)} output keys"
                + (f"; artifacts={len(artifact_ids)}" if artifact_ids else "")
            )

            # Post-execution approval gate
            if step.requires_approval_after:
                approved = self._request_approval(
                    step=step, wf_run=wf_run, gate="after"
                )
                wf_run.approval_request_ids.append(
                    f"{wf_run.run_id}:{step.step_id}:after:{attempt}"
                )
                if not approved:
                    logger.warning(
                        "WorkflowRun %s: step %s rejected at post-execution gate",
                        wf_run.run_id, step.step_id,
                    )
                    failure = _build_failure_record(
                        agent_name=step.agent_name,
                        task_id=task_id,
                        workflow_run_id=wf_run.run_id,
                        failure_class=FailureClass.APPROVAL_REJECTED,
                        message=f"Step {step.step_id!r} rejected at post-execution gate",
                        is_retryable=False,
                        retry_count=attempt,
                        escalated=False,
                    )
                    self._failure_records[wf_run.run_id].append(failure)
                    step_run.status = WorkflowStepStatus.FAILED
                    step_run.error = failure.message
                    step_run.completed_at = _utc_now()
                    return step_run, False

            # All good
            step_run.status = WorkflowStepStatus.COMPLETED
            step_run.completed_at = _utc_now()
            logger.info(
                "WorkflowRun %s: step %s COMPLETED (attempt %d)",
                wf_run.run_id, step.step_id, attempt,
            )
            return step_run, True

        # All retries exhausted
        failure = _build_failure_record(
            agent_name=step.agent_name,
            task_id=step_run.agent_task_id or step_run.step_run_id,
            workflow_run_id=wf_run.run_id,
            failure_class=FailureClass.RETRY_EXHAUSTED,
            message=(
                f"Step {step.step_id!r} exhausted {step.retry_max} retries: "
                f"{step_run.error or '(unknown error)'}"
            ),
            is_retryable=False,
            retry_count=step.retry_max,
            escalated=step.on_failure == "escalate",
        )
        self._failure_records[wf_run.run_id].append(failure)
        step_run.status = WorkflowStepStatus.FAILED
        step_run.error = failure.message
        step_run.completed_at = _utc_now()
        logger.error(
            "WorkflowRun %s: step %s FAILED after %d retries",
            wf_run.run_id, step.step_id, step.retry_max,
        )
        return step_run, False

    # ── Approval helpers ──────────────────────────────────────────

    def _request_approval(
        self,
        step: WorkflowStep,
        wf_run: WorkflowRun,
        gate: str,
    ) -> bool:
        """
        Request approval for a step gate.

        Invokes ``approval_callback(step, wf_run, gate)`` if set.
        Falls back to auto-approve in RESEARCH / STAGING and auto-reject
        in PAPER / PRODUCTION for HIGH_RISK+ operations.
        """
        if self._approval_callback is not None:
            try:
                result = self._approval_callback(step, wf_run, gate)
                logger.info(
                    "WorkflowRun %s: approval callback returned %s for step %s (%s gate)",
                    wf_run.run_id, result, step.step_id, gate,
                )
                return bool(result)
            except Exception as exc:  # pragma: no cover — defensive
                logger.error(
                    "WorkflowRun %s: approval callback raised %s — defaulting to reject",
                    wf_run.run_id, exc,
                )
                return False

        # Default behaviour: auto-approve in research/staging
        if self._environment in (
            EnvironmentClass.RESEARCH, EnvironmentClass.STAGING
        ):
            logger.info(
                "WorkflowRun %s: auto-approving step %s (%s gate) in %s environment",
                wf_run.run_id, step.step_id, gate, self._environment.value,
            )
            return True

        # In PAPER / PRODUCTION without a callback, reject anything above BOUNDED_SAFE
        if step.risk_class in (RiskClass.HIGH_RISK, RiskClass.SENSITIVE):
            logger.warning(
                "WorkflowRun %s: auto-rejecting HIGH_RISK/SENSITIVE step %s in %s "
                "(no approval_callback set)",
                wf_run.run_id, step.step_id, self._environment.value,
            )
            return False

        # Medium risk and below: auto-approve even in paper/production
        logger.info(
            "WorkflowRun %s: auto-approving step %s (%s) in %s",
            wf_run.run_id, step.step_id, step.risk_class.value, self._environment.value,
        )
        return True

    # ── Artefact helpers ──────────────────────────────────────────

    def _collect_artifacts(
        self,
        wf_run: WorkflowRun,
        step: WorkflowStep,
        output: dict,
    ) -> list[str]:
        """
        Persist step output via the artefact store if one is configured.

        Returns a list of artefact IDs produced.
        """
        if self._artifact_store is None or not output:
            return []
        try:
            ids = self._artifact_store(wf_run.run_id, step.step_id, output)
            return list(ids) if ids else []
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "WorkflowRun %s: artifact_store raised %s for step %s — ignoring",
                wf_run.run_id, exc, step.step_id,
            )
            return []

    # ── Terminal-state helpers ────────────────────────────────────

    def _finalise_run(
        self,
        wf_run: WorkflowRun,
        status: WorkflowStatus,
        error: Optional[str] = None,
    ) -> None:
        """Stamp the WorkflowRun with a terminal status and completion timestamp."""
        wf_run.status = status
        wf_run.completed_at = _utc_now()
        if error:
            wf_run.error = error
        logger.info(
            "WorkflowRun %s finalised: status=%s error=%r",
            wf_run.run_id, status.value, error,
        )

    def _record_step_skipped(
        self,
        wf_run: WorkflowRun,
        step: WorkflowStep,
        run_id: str,
        reason: str = "Skipped due to upstream failure",
    ) -> None:
        """Append a SKIPPED WorkflowStepRun to *wf_run*."""
        sr = WorkflowStepRun(
            step_run_id=str(uuid.uuid4()),
            step_id=step.step_id,
            workflow_run_id=run_id,
            status=WorkflowStepStatus.SKIPPED,
            started_at=None,
            completed_at=_utc_now(),
            agent_task_id=None,
            agent_result_status=None,
            output_summary=reason,
        )
        wf_run.step_runs.append(sr)

    def _skip_remaining_steps(
        self,
        wf_run: WorkflowRun,
        ordered_steps: list[WorkflowStep],
        completed_step_ids: set[str],
        skip_step_ids: set[str],
    ) -> None:
        """Record SKIPPED entries for all steps not yet executed."""
        executed_ids = {sr.step_id for sr in wf_run.step_runs}
        for step in ordered_steps:
            if step.step_id not in executed_ids:
                self._record_step_skipped(
                    wf_run, step, wf_run.run_id,
                    reason="Skipped — workflow cancelled or timed out",
                )


# ══════════════════════════════════════════════════════════════════
# PRE-BUILT WORKFLOW FACTORIES
# ══════════════════════════════════════════════════════════════════


def build_research_discovery_workflow() -> WorkflowDefinition:
    """
    Build a pre-wired research discovery workflow.

    Step order
    ----------
    1. ``universe_refresh``   — Universe Refresh (UniverseDiscoveryAgent)
    2. ``candidate_discovery`` — Candidate Discovery (depends on universe_refresh)
    3. ``pair_validation``    — Pair Validation (depends on candidate_discovery)
    4. ``spread_fit``         — Spread Fitting (depends on pair_validation)

    All steps are BOUNDED_SAFE, require no approval gates, and use
    ``on_failure="skip"`` so a single bad pair does not abort the pipeline.

    Returns
    -------
    WorkflowDefinition
    """
    steps = (
        WorkflowStep(
            step_id="universe_refresh",
            name="Universe Refresh",
            agent_name="universe_discovery",
            task_type="discover_pairs",
            depends_on=(),
            timeout_seconds=120.0,
            retry_max=2,
            risk_class=RiskClass.BOUNDED_SAFE,
            requires_approval_before=False,
            requires_approval_after=False,
            on_failure="fail_workflow",
            notes="Discovers candidate pairs from the configured symbol universe.",
        ),
        WorkflowStep(
            step_id="candidate_discovery",
            name="Candidate Discovery",
            agent_name="universe_discovery",
            task_type="discover_pairs",
            depends_on=("universe_refresh",),
            timeout_seconds=180.0,
            retry_max=1,
            risk_class=RiskClass.BOUNDED_SAFE,
            requires_approval_before=False,
            requires_approval_after=False,
            on_failure="skip",
            notes="Re-runs discovery with refined correlation thresholds.",
        ),
        WorkflowStep(
            step_id="pair_validation",
            name="Pair Validation",
            agent_name="pair_validation",
            task_type="validate_pairs",
            depends_on=("candidate_discovery",),
            timeout_seconds=300.0,
            retry_max=1,
            risk_class=RiskClass.BOUNDED_SAFE,
            requires_approval_before=False,
            requires_approval_after=False,
            on_failure="skip",
            notes="Statistical validation: ADF, cointegration, half-life.",
        ),
        WorkflowStep(
            step_id="spread_fit",
            name="Spread Fitting",
            agent_name="spread_fit",
            task_type="fit_spreads",
            depends_on=("pair_validation",),
            timeout_seconds=300.0,
            retry_max=1,
            risk_class=RiskClass.BOUNDED_SAFE,
            requires_approval_before=False,
            requires_approval_after=False,
            on_failure="skip",
            notes="OLS / Rolling-OLS / Kalman spread construction for validated pairs.",
        ),
    )

    transitions = (
        WorkflowTransition(
            from_step_id="universe_refresh",
            to_step_id="candidate_discovery",
            condition="on_success",
        ),
        WorkflowTransition(
            from_step_id="candidate_discovery",
            to_step_id="pair_validation",
            condition="on_success",
        ),
        WorkflowTransition(
            from_step_id="pair_validation",
            to_step_id="spread_fit",
            condition="on_success",
        ),
        WorkflowTransition(
            from_step_id="spread_fit",
            to_step_id=None,
            condition="always",
        ),
    )

    return WorkflowDefinition(
        workflow_id="research_discovery_v1",
        name="Research Discovery Pipeline",
        description=(
            "Universe Refresh → Candidate Discovery → Pair Validation → Spread Fit. "
            "Produces validated SpreadDefinitions ready for signal generation."
        ),
        version="1.0.0",
        steps=steps,
        transitions=transitions,
        entry_condition="Scheduled daily after market close, or triggered manually.",
        termination_condition=(
            "Spread Fit step completes (success or skip) or Universe Refresh fails."
        ),
        environment_class=EnvironmentClass.RESEARCH,
        risk_class=RiskClass.BOUNDED_SAFE,
        max_duration_seconds=1200.0,
        idempotent=True,
        replayable=True,
        owner="research",
        tags=("discovery", "validation", "spread_fitting", "scheduled"),
    )


def build_model_promotion_workflow() -> WorkflowDefinition:
    """
    Build a pre-wired model promotion workflow.

    Step order
    ----------
    1. ``model_eval``         — Model Evaluation (INFORMATIONAL)
    2. ``evidence_bundle``    — Evidence Bundle Assembly (INFORMATIONAL)
    3. ``approval_gate``      — Human Approval Gate (MEDIUM_RISK, approval required)
    4. ``promote_model``      — Promote Model to CHAMPION (HIGH_RISK, approval required)

    The approval gate step uses ``requires_approval_before=True`` so a human
    must explicitly approve before promotion proceeds.

    Returns
    -------
    WorkflowDefinition
    """
    steps = (
        WorkflowStep(
            step_id="model_eval",
            name="Model Evaluation",
            agent_name="pair_validation",       # Re-used for demonstration
            task_type="validate_pairs",
            depends_on=(),
            timeout_seconds=300.0,
            retry_max=2,
            risk_class=RiskClass.INFORMATIONAL,
            requires_approval_before=False,
            requires_approval_after=False,
            on_failure="fail_workflow",
            notes=(
                "Evaluate candidate model on out-of-time data. "
                "Produces validation metrics and evidence bundle."
            ),
        ),
        WorkflowStep(
            step_id="evidence_bundle",
            name="Evidence Bundle Assembly",
            agent_name="pair_validation",
            task_type="validate_pairs",
            depends_on=("model_eval",),
            timeout_seconds=120.0,
            retry_max=1,
            risk_class=RiskClass.INFORMATIONAL,
            requires_approval_before=False,
            requires_approval_after=False,
            on_failure="escalate",
            notes="Assembles structured evidence bundle from evaluation outputs.",
        ),
        WorkflowStep(
            step_id="approval_gate",
            name="Human Approval Gate",
            agent_name="pair_validation",       # Placeholder — approval is engine-level
            task_type="validate_pairs",
            depends_on=("evidence_bundle",),
            timeout_seconds=86400.0,            # 24 h — wait for human
            retry_max=0,
            risk_class=RiskClass.MEDIUM_RISK,
            requires_approval_before=True,
            requires_approval_after=False,
            on_failure="fail_workflow",
            notes=(
                "Blocks until a human reviewer approves the evidence bundle. "
                "Rejection aborts the workflow."
            ),
        ),
        WorkflowStep(
            step_id="promote_model",
            name="Promote Model to Champion",
            agent_name="pair_validation",       # Placeholder
            task_type="validate_pairs",
            depends_on=("approval_gate",),
            timeout_seconds=60.0,
            retry_max=1,
            risk_class=RiskClass.HIGH_RISK,
            requires_approval_before=True,
            requires_approval_after=True,
            on_failure="escalate",
            notes=(
                "Promotes the candidate model to CHAMPION in the ML registry. "
                "Previous CHAMPION is automatically demoted to RETIRED."
            ),
        ),
    )

    transitions = (
        WorkflowTransition(
            from_step_id="model_eval",
            to_step_id="evidence_bundle",
            condition="on_success",
        ),
        WorkflowTransition(
            from_step_id="evidence_bundle",
            to_step_id="approval_gate",
            condition="on_success",
        ),
        WorkflowTransition(
            from_step_id="approval_gate",
            to_step_id="promote_model",
            condition="on_approval",
        ),
        WorkflowTransition(
            from_step_id="approval_gate",
            to_step_id=None,
            condition="on_rejection",
            notes="Workflow terminates if approval is rejected.",
        ),
        WorkflowTransition(
            from_step_id="promote_model",
            to_step_id=None,
            condition="always",
        ),
    )

    return WorkflowDefinition(
        workflow_id="model_promotion_v1",
        name="Model Promotion Workflow",
        description=(
            "Model Eval → Evidence Bundle → Approval Gate → Promote. "
            "Governs promotion of a challenger model to production CHAMPION."
        ),
        version="1.0.0",
        steps=steps,
        transitions=transitions,
        entry_condition=(
            "Triggered when a challenger model passes out-of-time evaluation "
            "criteria defined in GovernanceEngine."
        ),
        termination_condition=(
            "Model promoted to CHAMPION or approval rejected."
        ),
        environment_class=EnvironmentClass.STAGING,
        risk_class=RiskClass.HIGH_RISK,
        max_duration_seconds=90000.0,           # 25 h to allow for human review
        idempotent=False,
        replayable=True,
        owner="ml_platform",
        tags=("model_governance", "promotion", "approval_required"),
    )


def build_drift_alert_workflow() -> WorkflowDefinition:
    """
    Build a pre-wired drift alert workflow.

    Step order
    ----------
    1. ``drift_check``        — Feature Drift Check (INFORMATIONAL)
    2. ``risk_assessment``    — Risk Assessment (BOUNDED_SAFE)
    3. ``incident_creation``  — Incident Creation (MEDIUM_RISK)
    4. ``human_review``       — Human Review Gate (HIGH_RISK, approval required)

    Returns
    -------
    WorkflowDefinition
    """
    steps = (
        WorkflowStep(
            step_id="drift_check",
            name="Feature Drift Check",
            agent_name="universe_discovery",    # Placeholder
            task_type="discover_pairs",
            depends_on=(),
            timeout_seconds=120.0,
            retry_max=2,
            risk_class=RiskClass.INFORMATIONAL,
            requires_approval_before=False,
            requires_approval_after=False,
            on_failure="fail_workflow",
            notes=(
                "Computes PSI for all active model features against the reference "
                "distribution.  Flags features above the drift threshold."
            ),
        ),
        WorkflowStep(
            step_id="risk_assessment",
            name="Risk Assessment",
            agent_name="pair_validation",
            task_type="validate_pairs",
            depends_on=("drift_check",),
            timeout_seconds=120.0,
            retry_max=1,
            risk_class=RiskClass.BOUNDED_SAFE,
            requires_approval_before=False,
            requires_approval_after=False,
            on_failure="escalate",
            notes=(
                "Assesses trading risk given detected drift. "
                "May recommend de-risking or position freeze."
            ),
        ),
        WorkflowStep(
            step_id="incident_creation",
            name="Incident Creation",
            agent_name="pair_validation",
            task_type="validate_pairs",
            depends_on=("risk_assessment",),
            timeout_seconds=60.0,
            retry_max=1,
            risk_class=RiskClass.MEDIUM_RISK,
            requires_approval_before=False,
            requires_approval_after=False,
            on_failure="escalate",
            notes="Creates a structured incident record in the incident log.",
        ),
        WorkflowStep(
            step_id="human_review",
            name="Human Review Gate",
            agent_name="pair_validation",
            task_type="validate_pairs",
            depends_on=("incident_creation",),
            timeout_seconds=86400.0,            # 24 h for human response
            retry_max=0,
            risk_class=RiskClass.HIGH_RISK,
            requires_approval_before=True,
            requires_approval_after=False,
            on_failure="fail_workflow",
            notes=(
                "Blocks until a human reviewer acknowledges the drift incident. "
                "In PRODUCTION, unacknowledged incidents after 24h should trigger "
                "an automatic kill-switch escalation."
            ),
        ),
    )

    transitions = (
        WorkflowTransition(
            from_step_id="drift_check",
            to_step_id="risk_assessment",
            condition="on_success",
        ),
        WorkflowTransition(
            from_step_id="risk_assessment",
            to_step_id="incident_creation",
            condition="on_success",
        ),
        WorkflowTransition(
            from_step_id="incident_creation",
            to_step_id="human_review",
            condition="on_success",
        ),
        WorkflowTransition(
            from_step_id="human_review",
            to_step_id=None,
            condition="always",
        ),
    )

    return WorkflowDefinition(
        workflow_id="drift_alert_v1",
        name="Drift Alert Workflow",
        description=(
            "Drift Check → Risk Assessment → Incident Creation → Human Review. "
            "Triggered when FeatureDriftMonitor detects PSI above threshold."
        ),
        version="1.0.0",
        steps=steps,
        transitions=transitions,
        entry_condition=(
            "Triggered by FeatureDriftMonitor when PSI for any active feature "
            "exceeds the configured alert threshold."
        ),
        termination_condition=(
            "Human reviewer acknowledges the incident, or 24h timeout escalates to "
            "kill-switch."
        ),
        environment_class=EnvironmentClass.PRODUCTION,
        risk_class=RiskClass.HIGH_RISK,
        max_duration_seconds=90000.0,
        idempotent=False,
        replayable=False,
        owner="ml_platform",
        tags=("drift_detection", "incident", "human_review", "alert"),
    )
