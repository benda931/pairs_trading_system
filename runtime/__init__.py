# -*- coding: utf-8 -*-
"""
runtime — Runtime State and Environment Management
====================================================

Public API for the runtime package.  Import domain types, the state manager,
and environment utilities from here rather than from sub-modules directly.

Example::

    from runtime import (
        RuntimeMode, ServiceState, ThrottleLevel,
        StrategyActivationRecord, RuntimeState,
        RuntimeStateManager, get_runtime_state_manager,
        get_environment_spec, validate_environment_action,
        ENVIRONMENT_SPECS,
    )
"""

from runtime.contracts import (
    ActivationDecision,
    ActivationRequest,
    ActivationStatus,
    AgentActivationRecord,
    DesiredRuntimeState,
    DrainState,
    EnvironmentRestrictionType,
    EnvironmentSpec,
    LiveTradingReadinessReport,
    ModelActivationRecord,
    OverrideApprovalRecord,
    PauseState,
    PreflightCheckReport,
    RuntimeConfigSnapshot,
    RuntimeMode,
    RuntimeOverride,
    RuntimeState,
    RuntimeTransition,
    ServiceState,
    StrategyActivationRecord,
    ThrottleLevel,
)

from runtime.state import (
    RuntimeStateManager,
    get_runtime_state_manager,
)

from runtime.environment import (
    ENVIRONMENT_SPECS,
    allows_broker_orders,
    get_environment_spec,
    list_environments,
    requires_live_capital,
    validate_environment_action,
)

__all__ = [
    # Enums
    "RuntimeMode",
    "ServiceState",
    "ActivationStatus",
    "ThrottleLevel",
    "EnvironmentRestrictionType",
    # Core domain types
    "EnvironmentSpec",
    "RuntimeTransition",
    "ActivationRequest",
    "ActivationDecision",
    "StrategyActivationRecord",
    "ModelActivationRecord",
    "AgentActivationRecord",
    "RuntimeOverride",
    "OverrideApprovalRecord",
    "RuntimeState",
    "DesiredRuntimeState",
    "PauseState",
    "DrainState",
    "RuntimeConfigSnapshot",
    "LiveTradingReadinessReport",
    "PreflightCheckReport",
    # State manager
    "RuntimeStateManager",
    "get_runtime_state_manager",
    # Environment helpers
    "ENVIRONMENT_SPECS",
    "get_environment_spec",
    "validate_environment_action",
    "list_environments",
    "requires_live_capital",
    "allows_broker_orders",
]
