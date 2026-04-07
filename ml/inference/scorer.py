# -*- coding: utf-8 -*-
"""
ml/inference/scorer.py — Inference Layer
=========================================

Handles:
- Model loading from registry
- Feature validation at inference time
- Fallback logic when model is unavailable / stale / has insufficient features
- Structured prediction output (InferenceResult)
- TTL-based caching of model objects
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml.contracts import (
    FallbackPolicy,
    InferenceRequest,
    InferenceResult,
    MetaLabelAction,
    MLTaskFamily,
    ModelStatus,
    _now_utc,
)

logger = logging.getLogger("ml.inference.scorer")


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

class _CacheEntry:
    __slots__ = ("model", "loaded_at")

    def __init__(self, model: Any):
        self.model = model
        self.loaded_at: float = time.monotonic()

    def is_fresh(self, ttl_seconds: int) -> bool:
        return (time.monotonic() - self.loaded_at) < ttl_seconds


# ---------------------------------------------------------------------------
# ModelScorer
# ---------------------------------------------------------------------------

class ModelScorer:
    """
    Inference layer for the ML platform.

    Handles model loading, feature validation, fallback logic, and structured
    prediction output. Never raises — all errors are captured in InferenceResult.
    """

    def __init__(
        self,
        registry=None,          # MLModelRegistry | None
        fallback_policy: Optional[FallbackPolicy] = None,
        cache_ttl_seconds: int = 300,
    ):
        # Import lazily to avoid circular imports
        if registry is None:
            try:
                from ml.registry import get_ml_registry
                self._registry = get_ml_registry()
            except Exception:
                self._registry = None
        else:
            self._registry = registry

        self._fallback_policy = fallback_policy or FallbackPolicy()
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, _CacheEntry] = {}

    # ------------------------------------------------------------------
    # Single inference
    # ------------------------------------------------------------------

    def score(self, request: InferenceRequest) -> InferenceResult:
        """
        Execute a single inference request.

        Returns a fallback InferenceResult if the model is unavailable,
        stale, or missing too many features. Never raises.
        """
        try:
            return self._score_internal(request)
        except Exception as exc:
            logger.error("Unexpected error in score(): %s", exc, exc_info=True)
            return self._apply_fallback(request, f"unexpected_error: {exc}")

    def _score_internal(self, request: InferenceRequest) -> InferenceResult:
        # 1. Resolve model
        model = self.get_model_for_task(request.task_family)
        if model is None:
            if request.allow_fallback:
                return self._apply_fallback(request, "no_model_available")
            return _error_result(request, "no_model_available")

        # 1b. In-training-window guard: if as_of is within the model's training window,
        # the model has implicitly seen this data during training — reject the request.
        metadata = getattr(model, "metadata", None)
        model_id = _get_model_id(model)
        if request.as_of and metadata is not None and hasattr(metadata, "trained_until") and metadata.trained_until:
            try:
                from datetime import datetime, timezone as _tz
                req_dt = datetime.fromisoformat(request.as_of)
                until_dt = datetime.fromisoformat(metadata.trained_until)
                if req_dt.tzinfo is None:
                    req_dt = req_dt.replace(tzinfo=_tz.utc)
                if until_dt.tzinfo is None:
                    until_dt = until_dt.replace(tzinfo=_tz.utc)
                if req_dt <= until_dt:
                    logger.warning(
                        "Inference blocked: as_of=%s is within training window "
                        "(trained_until=%s) for model %s. "
                        "This would constitute lookahead bias.",
                        request.as_of, metadata.trained_until, model_id,
                    )
                    return self._apply_fallback(
                        request,
                        f"in_training_window: as_of={request.as_of} <= trained_until={metadata.trained_until}",
                    )
            except Exception as _exc:
                logger.debug("trained_until check skipped: %s", _exc)

        # 2. Validate features
        required_features = _get_feature_names(model)
        ok, missing = self._validate_features(
            request.features,
            required_features,
            missing_threshold=self._fallback_policy.missing_feature_threshold,
        )
        feature_availability = "full" if not missing else (
            "partial" if len(missing) < len(required_features) * 0.5 else "degraded"
        )

        if not ok:
            if request.allow_fallback or self._fallback_policy.trigger_on_missing_features:
                return self._apply_fallback(
                    request,
                    f"insufficient_features: {missing[:5]}",
                )
            return _error_result(request, f"missing_features: {missing}")

        # 3. Build feature vector
        warnings: list[str] = []
        X = _build_feature_df(request.features, required_features, warnings)

        # 4. Predict
        try:
            if hasattr(model, "predict_proba"):
                raw = model.predict_proba(X)
                arr = np.asarray(raw, dtype=float)
                score_val = float(arr[0, 1] if arr.ndim == 2 else arr[0])
            else:
                raw = model.predict(X)
                score_val = float(np.asarray(raw, dtype=float).ravel()[0])
        except Exception as exc:
            logger.warning("predict failed for request %s: %s", request.request_id, exc)
            if request.allow_fallback:
                return self._apply_fallback(request, f"predict_error: {exc}")
            return _error_result(request, f"predict_error: {exc}")

        # 5. Meta action
        meta_action: Optional[MetaLabelAction] = None
        if request.task_family == MLTaskFamily.META_LABELING:
            meta_action = MetaLabelAction.TAKE if score_val >= 0.5 else MetaLabelAction.SKIP

        # 6. Confidence — use score distance from 0.5 as proxy
        confidence = float(2.0 * abs(score_val - 0.5))

        return InferenceResult(
            request_id=request.request_id,
            model_id=model_id,
            task_family=request.task_family.value,
            entity_id=request.entity_id,
            score=score_val,
            label=str(int(score_val >= 0.5)),
            meta_action=meta_action,
            confidence=confidence,
            feature_availability=feature_availability,
            missing_features=missing,
            fallback_used=False,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def score_batch(
        self,
        requests: list[InferenceRequest],
        model_id: Optional[str] = None,
    ) -> list[InferenceResult]:
        """
        Batch scoring. Falls back gracefully per-request on failure.
        """
        results: list[InferenceResult] = []

        for req in requests:
            try:
                results.append(self.score(req))
            except Exception as exc:
                logger.error("Batch score failed for request %s: %s", req.request_id, exc)
                results.append(self._apply_fallback(req, f"batch_error: {exc}"))

        return results

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def get_model_for_task(
        self,
        task_family: MLTaskFamily,
        prefer_status: ModelStatus = ModelStatus.CHAMPION,
    ) -> Optional[Any]:
        """
        Load the best available model for a task family from the registry.

        Tries CHAMPION first, then falls back to CHALLENGER, then CANDIDATE.
        Caches the model object for cache_ttl_seconds.
        """
        if self._registry is None:
            return None

        # Build cache key
        cache_key = f"{task_family.value}:{prefer_status.value}"
        if cache_key in self._cache and self._cache[cache_key].is_fresh(self._cache_ttl):
            return self._cache[cache_key].model

        # Resolve from registry
        model_obj = None
        fallback_order = [prefer_status]
        if prefer_status != ModelStatus.CHAMPION:
            fallback_order.append(ModelStatus.CHAMPION)
        # Append CHALLENGER if not already present; CANDIDATE is intentionally excluded —
        # candidate models have not passed governance review and must not serve production inference.
        if ModelStatus.CHALLENGER not in fallback_order:
            fallback_order.append(ModelStatus.CHALLENGER)

        for status in fallback_order:
            try:
                if status == ModelStatus.CHAMPION:
                    meta = self._registry.get_champion(task_family)
                elif status == ModelStatus.CHALLENGER:
                    challengers = self._registry.get_challengers(task_family)
                    meta = challengers[0] if challengers else None
                else:
                    # Should not be reached with CANDIDATE excluded, but guard anyway.
                    meta = None

                if meta is not None:
                    obj = self._registry.get_model_object(meta.model_id)
                    if obj is not None:
                        model_obj = obj
                        self._cache[cache_key] = _CacheEntry(model_obj)
                        break
            except Exception as exc:
                logger.debug("Registry lookup failed for %s/%s: %s", task_family, status, exc)

        return model_obj

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _apply_fallback(
        self,
        request: InferenceRequest,
        reason: str,
    ) -> InferenceResult:
        """Return a neutral/safe fallback InferenceResult."""
        neutral = self._fallback_policy.neutral_score
        meta_action = self._fallback_policy.fallback_meta_action

        if self._fallback_policy.log_fallback:
            logger.info(
                "Fallback used for request %s (entity=%s, reason=%s)",
                request.request_id,
                request.entity_id,
                reason,
            )

        return InferenceResult(
            request_id=request.request_id,
            model_id="fallback",
            task_family=request.task_family.value,
            entity_id=request.entity_id,
            score=neutral,
            label=None,
            meta_action=meta_action,
            confidence=0.0,
            feature_availability="degraded",
            missing_features=[],
            fallback_used=True,
            fallback_reason=reason,
            warnings=[f"fallback_active: {reason}"],
        )

    # ------------------------------------------------------------------
    # Feature validation
    # ------------------------------------------------------------------

    def _validate_features(
        self,
        features: dict,
        required_features: list[str],
        missing_threshold: float = 0.20,
    ) -> tuple[bool, list[str]]:
        """
        Check feature availability.

        Returns (ok, missing_list).
        ok is True if the fraction of missing features is below missing_threshold.
        """
        if not required_features:
            return True, []

        missing = [f for f in required_features if f not in features or features[f] != features[f]]

        fraction_missing = len(missing) / len(required_features)
        ok = fraction_missing <= missing_threshold
        return ok, missing

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate_cache(self, task_family: Optional[MLTaskFamily] = None) -> None:
        """Invalidate cached model objects."""
        if task_family is None:
            self._cache.clear()
        else:
            keys_to_remove = [k for k in self._cache if k.startswith(task_family.value + ":")]
            for k in keys_to_remove:
                del self._cache[k]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model_id(model: Any) -> str:
    if hasattr(model, "metadata") and hasattr(model.metadata, "model_id"):
        return str(model.metadata.model_id)
    if hasattr(model, "model_id"):
        return str(model.model_id)
    return "unknown"


def _get_feature_names(model: Any) -> list[str]:
    """Extract expected feature names from model if available."""
    for attr in ("feature_names_in_", "feature_names", "feature_columns"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            if val is not None:
                return list(val)
    return []


def _build_feature_df(
    features: dict,
    required_features: list[str],
    warnings: list[str],
) -> pd.DataFrame:
    """
    Build a single-row DataFrame from a feature dict.

    Missing required features are filled with 0.0 with a warning recorded.
    """
    row: dict[str, float] = {}
    for feat in (required_features if required_features else list(features.keys())):
        val = features.get(feat, float("nan"))
        if not np.isfinite(val):
            row[feat] = 0.0
            warnings.append(f"feature_imputed_zero: {feat}")
        else:
            row[feat] = float(val)

    # Also include any extra features not in required list
    for k, v in features.items():
        if k not in row:
            row[k] = float(v) if np.isfinite(float(v)) else 0.0

    return pd.DataFrame([row])


def _error_result(request: InferenceRequest, reason: str) -> InferenceResult:
    return InferenceResult(
        request_id=request.request_id,
        model_id="",
        task_family=request.task_family.value,
        entity_id=request.entity_id,
        score=float("nan"),
        fallback_used=False,
        fallback_reason=reason,
        warnings=[reason],
    )
