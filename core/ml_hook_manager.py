# -*- coding: utf-8 -*-
"""
core/ml_hook_manager.py — ML Hook Lifecycle Manager
====================================================

Manages the lifecycle of the ML hook object used by the signal pipeline.

Responsibilities:
- Load the ML hook via the formal ML governance layer ONLY
- Cache the hook with TTL-based refresh
- Expose hook status for health monitoring
- Never load ungoverned pickle files

Extracted from core/orchestrator.py to give this critical governance
concern its own module with clear ownership and testability.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("core.ml_hook_manager")

# Default TTL for the cached ML hook (seconds)
# Re-checks the registry every 30 minutes in case a model was promoted/blocked
_DEFAULT_HOOK_TTL_SECONDS = 1800


class MLHookManager:
    """
    Manages the ML hook used for signal quality scoring.

    The hook is loaded from the formal ML registry ONLY. Pickle file
    fallbacks are explicitly prohibited — they bypass governance controls.

    Parameters
    ----------
    hook_ttl_seconds : int
        How long to cache the hook before re-checking the registry.
        Default: 1800 seconds (30 minutes).
    """

    def __init__(self, hook_ttl_seconds: int = _DEFAULT_HOOK_TTL_SECONDS):
        self._ttl = hook_ttl_seconds
        self._hook: Optional[Any] = None
        self._loaded_at: float = 0.0
        self._hook_model_id: str = ""
        self._hook_status: str = "not_loaded"

    @property
    def hook(self) -> Optional[Any]:
        """Return the current ML hook, refreshing if TTL has expired."""
        if self._is_stale():
            self._refresh()
        return self._hook

    @property
    def model_id(self) -> str:
        """Model ID of the currently loaded hook (empty if no hook)."""
        return self._hook_model_id

    @property
    def status(self) -> str:
        """Status string for health monitoring."""
        return self._hook_status

    def force_refresh(self) -> None:
        """Force an immediate hook refresh from the registry."""
        self._loaded_at = 0.0
        self._refresh()

    def _is_stale(self) -> bool:
        return (time.monotonic() - self._loaded_at) > self._ttl

    def _refresh(self) -> None:
        """Reload the ML hook from the formal ML governance layer."""
        self._hook = self._load_governed_hook()
        self._loaded_at = time.monotonic()

    def _load_governed_hook(self) -> Optional[Any]:
        """
        Load ML hook via ModelScorer (formal inference layer) ONLY.

        No pickle files. No ad-hoc fallbacks. If no governed model is
        available, return None — the pipeline uses deterministic scoring.
        """
        try:
            from ml.inference.scorer import ModelScorer
            from ml.contracts import InferenceRequest, MLTaskFamily, ModelStatus

            scorer = ModelScorer()
            model = scorer.get_model_for_task(MLTaskFamily.META_LABELING)

            if model is None:
                self._hook_status = "no_champion"
                self._hook_model_id = ""
                logger.info(
                    "MLHookManager: no CHAMPION model for META_LABELING — "
                    "using deterministic quality fallback"
                )
                return None

            # Verify champion status
            metadata = getattr(model, "metadata", None)
            if metadata is not None:
                status = getattr(metadata, "status", None)
                if status != ModelStatus.CHAMPION:
                    self._hook_status = f"not_champion:{status}"
                    self._hook_model_id = getattr(metadata, "model_id", "")
                    logger.warning(
                        "MLHookManager: model %s is %s (not CHAMPION) — skipping",
                        self._hook_model_id, status,
                    )
                    return None
                self._hook_model_id = getattr(metadata, "model_id", "")

            self._hook_status = "champion_loaded"

            class _ScorerHook:
                def __init__(self, _scorer, _model_id):
                    self._scorer = _scorer
                    self.model_id = _model_id
                    self.model_status = ModelStatus.CHAMPION

                def predict_success_probability(self, feats):
                    try:
                        req = InferenceRequest(
                            entity_id="pipeline",
                            task_family=MLTaskFamily.META_LABELING,
                            features=feats if isinstance(feats, dict) else {},
                            allow_fallback=False,
                        )
                        result = self._scorer.score(req)
                        return result.score if not result.fallback_used else float("nan")
                    except Exception:
                        return float("nan")

                def score(self, opportunity) -> float:
                    """For portfolio ranking compatibility."""
                    return float("nan")

            logger.info(
                "MLHookManager: loaded CHAMPION model %s for META_LABELING",
                self._hook_model_id,
            )
            return _ScorerHook(scorer, self._hook_model_id)

        except Exception as exc:
            self._hook_status = f"load_error:{exc}"
            self._hook_model_id = ""
            logger.debug("MLHookManager: ModelScorer unavailable: %s", exc)
            return None
