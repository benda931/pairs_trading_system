# -*- coding: utf-8 -*-
"""
deployment — Deployment lifecycle management package.

Public re-exports for convenient top-level imports.
"""

from deployment.contracts import (
    ConfigVersionRecord,
    DeploymentArtifact,
    DeploymentStage,
    ReleaseRecord,
    RollbackDecision,
    RollbackReason,
    RolloutPlan,
)
from deployment.engine import DeploymentEngine, get_deployment_engine

__all__ = [
    # Enums
    "DeploymentStage",
    "RollbackReason",
    # Value objects
    "ConfigVersionRecord",
    "DeploymentArtifact",
    "ReleaseRecord",
    "RollbackDecision",
    "RolloutPlan",
    # Engine
    "DeploymentEngine",
    "get_deployment_engine",
]
