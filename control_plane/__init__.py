# -*- coding: utf-8 -*-
"""
control_plane — Operator Control Surface
==========================================

Public API for the control_plane package.

Example::

    from control_plane import (
        ControlPlaneEngine, get_control_plane,
        ControlPlaneAction, ControlPlaneActionType,
        KillSwitchState, ThrottleState,
        OperatorActionRecord, HeartbeatRecord,
    )

    cp = get_control_plane()
    record = cp.engage_kill_switch("circuit breaker triggered", "risk-system")
"""

from control_plane.contracts import (
    ControlPlaneAction,
    ControlPlaneActionRecord,
    ControlPlaneActionType,
    HeartbeatRecord,
    KillSwitchState,
    OperatorActionRecord,
    ThrottleState,
)

from control_plane.engine import (
    ControlPlaneEngine,
    get_control_plane,
)

__all__ = [
    # Enums
    "ControlPlaneActionType",
    # Domain types
    "ControlPlaneAction",
    "ControlPlaneActionRecord",
    "KillSwitchState",
    "ThrottleState",
    "OperatorActionRecord",
    "HeartbeatRecord",
    # Engine
    "ControlPlaneEngine",
    "get_control_plane",
]
