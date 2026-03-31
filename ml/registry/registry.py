# -*- coding: utf-8 -*-
"""
ml/registry/registry.py — Extended ML Model Registry with Governance
=====================================================================

Extends the existing models.base.ModelRegistry with:
- MLTaskFamily-aware queries
- Champion / challenger tracking
- Governance status management
- ModelCard storage
- Promotion workflow with explicit PromotionDecision
- Persistent JSON storage so registrations survive process restarts
- Thread-safe operations
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml.contracts import (
    ChampionChallengerRecord,
    MLTaskFamily,
    ModelCard,
    ModelMetadata,
    ModelStatus,
    PromotionDecision,
    PromotionOutcome,
    TrainingRunArtifact,
    _now_utc,
)
from ml.evaluation.metrics import auc_roc, brier_score, information_coefficient, regime_sliced_metrics

try:
    from governance.engine import get_governance_engine as _get_governance_engine
except ImportError:
    _get_governance_engine = None

# Module-level reference for patching
get_governance_engine = _get_governance_engine

logger = logging.getLogger("ml.registry")


def _isnan(v: Any) -> bool:
    try:
        return float(v) != float(v)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class MLModelRegistry:
    """
    Extended model registry for the ML platform.

    Thread-safe. Optionally persisted to JSON.
    """

    def __init__(self, storage_path: Optional[str | Path] = None):
        self._lock = threading.Lock()
        self._models: Dict[str, ModelMetadata] = {}
        self._run_artifacts: Dict[str, TrainingRunArtifact] = {}
        self._model_cards: Dict[str, ModelCard] = {}
        # model_id → actual fitted model object (in-memory only)
        self._model_objects: Dict[str, Any] = {}

        self._storage_path: Optional[Path] = (
            Path(storage_path) if storage_path is not None else None
        )

        if self._storage_path is not None and self._storage_path.exists():
            try:
                self.load()
            except Exception as exc:
                logger.warning("Failed to load registry from %s: %s", self._storage_path, exc)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model: Any,
        metadata: ModelMetadata,
        run_artifact: Optional[TrainingRunArtifact] = None,
        model_card: Optional[ModelCard] = None,
    ) -> str:
        """
        Register a model. Returns model_id. Thread-safe.

        Parameters
        ----------
        model : fitted model object (stored in-memory, not serialized)
        metadata : ModelMetadata describing the model
        run_artifact : optional TrainingRunArtifact
        model_card : optional ModelCard (required for CHAMPION promotion)
        """
        with self._lock:
            model_id = metadata.model_id
            self._models[model_id] = metadata
            self._model_objects[model_id] = model

            if run_artifact is not None:
                self._run_artifacts[model_id] = run_artifact

            if model_card is not None:
                self._model_cards[model_id] = model_card

        logger.info(
            "Registered model %s (family=%s, status=%s)",
            model_id,
            metadata.task_family.value,
            metadata.status.value,
        )

        if self._storage_path is not None:
            self._safe_save()

        return model_id

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """Return metadata for a model_id, or None if not found."""
        with self._lock:
            return self._models.get(model_id)

    def get_model_object(self, model_id: str) -> Optional[Any]:
        """Return the in-memory model object for model_id."""
        with self._lock:
            return self._model_objects.get(model_id)

    def get_run_artifact(self, model_id: str) -> Optional[TrainingRunArtifact]:
        with self._lock:
            return self._run_artifacts.get(model_id)

    def get_model_card(self, model_id: str) -> Optional[ModelCard]:
        with self._lock:
            return self._model_cards.get(model_id)

    def get_champion(self, task_family: MLTaskFamily) -> Optional[ModelMetadata]:
        """Return the current champion model for a task family."""
        with self._lock:
            champions = [
                m for m in self._models.values()
                if m.task_family == task_family and m.status == ModelStatus.CHAMPION
            ]
        if not champions:
            return None
        # If multiple champions (shouldn't happen), return most recently trained
        return max(champions, key=lambda m: m.trained_at or "")

    def get_challengers(self, task_family: MLTaskFamily) -> List[ModelMetadata]:
        """Return all challenger models for a task family."""
        with self._lock:
            return [
                m for m in self._models.values()
                if m.task_family == task_family and m.status == ModelStatus.CHALLENGER
            ]

    def list_models(
        self,
        task_family: Optional[MLTaskFamily] = None,
        status: Optional[ModelStatus] = None,
    ) -> List[ModelMetadata]:
        """List models, optionally filtered by task family and/or status."""
        with self._lock:
            models = list(self._models.values())

        if task_family is not None:
            models = [m for m in models if m.task_family == task_family]
        if status is not None:
            models = [m for m in models if m.status == status]

        return sorted(models, key=lambda m: m.trained_at or "", reverse=True)

    # ------------------------------------------------------------------
    # Promotion workflow
    # ------------------------------------------------------------------

    def promote(
        self,
        model_id: str,
        to_status: ModelStatus,
        decision: PromotionDecision,
    ) -> bool:
        """
        Promote or demote a model.

        Governance rules enforced:
        - Promoting to CHAMPION requires PromotionOutcome.PROMOTE and
          decision.manually_approved == True (or requires_manual_approval == False).
        - If another CHAMPION exists for the same task family, it is demoted
          to CHALLENGER automatically.

        Returns False if governance rules block the promotion.
        """
        # Governance gate (P1-GOV): Check governance policy before champion promotion
        # Finding: docs/remediation/remediation_ledger.md:P1-GOV
        if to_status == ModelStatus.CHAMPION:
            try:
                if get_governance_engine is not None:
                    gov_engine = get_governance_engine()
                    check = gov_engine.check_policy(
                        agent_name="ml_registry",
                        task_type="model_promotion",
                        action_type="MODEL_PROMOTE_TO_CHAMPION",
                        environment="production",
                        risk_class="HIGH",
                        task_id=f"promote:{model_id}",
                    )
                    if check is not None and not check.passed and check.severity.value in ("CRITICAL", "EMERGENCY"):
                        raise ValueError(
                            f"Governance policy blocked champion promotion for model {model_id}: "
                            f"{check.message}. "
                            f"Use GovernanceEngine to review. Finding: P1-GOV"
                        )
                else:
                    logger.warning(
                        "Governance engine not available — skipping policy check for model %s promotion",
                        model_id,
                    )
            except ValueError:
                raise  # Re-raise governance blocks
            except Exception as e:
                logger.warning(
                    "Governance check failed for model promotion (non-blocking): %s", e
                )

        with self._lock:
            metadata = self._models.get(model_id)
            if metadata is None:
                logger.warning("promote(): model_id %s not found", model_id)
                return False

            # Governance gate for CHAMPION
            if to_status == ModelStatus.CHAMPION:
                if decision.outcome != PromotionOutcome.PROMOTE:
                    logger.info(
                        "Promotion blocked for %s: outcome=%s (must be PROMOTE)",
                        model_id,
                        decision.outcome.value,
                    )
                    return False
                if decision.requires_manual_approval and not decision.manually_approved:
                    logger.info(
                        "Promotion blocked for %s: requires manual approval", model_id
                    )
                    return False

                # Demote existing champion for this task family
                for mid, m in self._models.items():
                    if (
                        mid != model_id
                        and m.task_family == metadata.task_family
                        and m.status == ModelStatus.CHAMPION
                    ):
                        # Create updated metadata with CHALLENGER status
                        self._models[mid] = _update_metadata_status(m, ModelStatus.CHALLENGER)
                        logger.info("Demoted previous champion %s to CHALLENGER", mid)

            # Apply the status change
            promoted_at = _now_utc() if to_status == ModelStatus.CHAMPION else ""
            retired_at = _now_utc() if to_status == ModelStatus.RETIRED else ""
            self._models[model_id] = _update_metadata_status(
                metadata, to_status, promoted_at=promoted_at, retired_at=retired_at
            )
            logger.info(
                "Promoted model %s: %s → %s",
                model_id,
                metadata.status.value,
                to_status.value,
            )

        if self._storage_path is not None:
            self._safe_save()

        return True

    # ------------------------------------------------------------------
    # Champion/Challenger comparison
    # ------------------------------------------------------------------

    def compare_champion_challenger(
        self,
        champion_id: str,
        challenger_id: str,
        eval_X: pd.DataFrame,
        eval_y: pd.Series,
        regime_labels: Optional[pd.Series] = None,
    ) -> ChampionChallengerRecord:
        """
        Run a champion/challenger comparison and return a structured record.
        """
        record = ChampionChallengerRecord(
            champion_model_id=champion_id,
            challenger_model_id=challenger_id,
        )

        with self._lock:
            champ_meta = self._models.get(champion_id)
            chall_meta = self._models.get(challenger_id)
            champ_model = self._model_objects.get(champion_id)
            chall_model = self._model_objects.get(challenger_id)

        if champ_meta is not None:
            record.task_family = champ_meta.task_family.value

        yt = np.asarray(eval_y, dtype=float)

        def _score_model(m: Any) -> tuple[float, float, float]:
            """Returns (auc, brier, ic)."""
            if m is None:
                return float("nan"), float("nan"), float("nan")
            try:
                if hasattr(m, "predict_proba"):
                    raw = m.predict_proba(eval_X)
                    yp = np.asarray(raw, dtype=float)
                    if yp.ndim == 2:
                        yp = yp[:, 1]
                else:
                    yp = np.asarray(m.predict(eval_X), dtype=float)

                model_auc = auc_roc(yt, yp)
                model_brier = brier_score(yt, yp)
                yp_series = pd.Series(yp, index=eval_X.index)
                y_series = eval_y if isinstance(eval_y, pd.Series) else pd.Series(yt, index=eval_X.index)
                ic = information_coefficient(y_series, yp_series)
                return model_auc, model_brier, ic
            except Exception as exc:
                logger.warning("Scoring failed for model: %s", exc)
                return float("nan"), float("nan"), float("nan")

        record.champion_auc, record.champion_brier, record.champion_ic = _score_model(champ_model)
        record.challenger_auc, record.challenger_brier, record.challenger_ic = _score_model(chall_model)

        # Regime comparison
        if regime_labels is not None:
            try:
                rl = np.asarray(
                    regime_labels.reindex(eval_y.index).values
                    if isinstance(regime_labels, pd.Series)
                    else regime_labels,
                    dtype=str,
                )
                if champ_model is not None and chall_model is not None:
                    champ_yp = _get_proba(champ_model, eval_X)
                    chall_yp = _get_proba(chall_model, eval_X)
                    if champ_yp is not None and chall_yp is not None:
                        champ_regime = regime_sliced_metrics(yt, champ_yp, rl)
                        chall_regime = regime_sliced_metrics(yt, chall_yp, rl)
                        comparison: dict = {}
                        for regime in set(list(champ_regime.keys()) + list(chall_regime.keys())):
                            comparison[regime] = {
                                "champion_auc": champ_regime.get(regime, {}).get("auc", float("nan")),
                                "challenger_auc": chall_regime.get(regime, {}).get("auc", float("nan")),
                                "champion_brier": champ_regime.get(regime, {}).get("brier", float("nan")),
                                "challenger_brier": chall_regime.get(regime, {}).get("brier", float("nan")),
                            }
                        record.regime_comparison = comparison
                        record.n_regimes_tested = len(comparison)
            except Exception as exc:
                logger.warning("Regime comparison failed: %s", exc)

        # Determine outcome
        if (
            np.isfinite(record.challenger_auc)
            and np.isfinite(record.champion_auc)
            and record.challenger_auc > record.champion_auc + 0.01
            and np.isfinite(record.challenger_brier)
            and np.isfinite(record.champion_brier)
            and record.challenger_brier < record.champion_brier - 0.005
        ):
            record.outcome = PromotionOutcome.PROMOTE
            record.decision_rationale = (
                f"Challenger AUC {record.challenger_auc:.4f} > Champion AUC "
                f"{record.champion_auc:.4f} by >0.01 and lower Brier."
            )
        else:
            record.outcome = PromotionOutcome.REJECT
            record.decision_rationale = (
                "Challenger did not meet minimum improvement thresholds "
                "(AUC lift > 0.01 and Brier reduction > 0.005)."
            )

        return record

    # ------------------------------------------------------------------
    # Deprecation
    # ------------------------------------------------------------------

    def deprecate(self, model_id: str, reason: str = "") -> None:
        """Mark a model as RETIRED."""
        with self._lock:
            metadata = self._models.get(model_id)
            if metadata is None:
                logger.warning("deprecate(): model_id %s not found", model_id)
                return
            self._models[model_id] = _update_metadata_status(
                metadata, ModelStatus.RETIRED, retired_at=_now_utc()
            )

        if reason:
            logger.info("Deprecated model %s: %s", model_id, reason)
        else:
            logger.info("Deprecated model %s", model_id)

        if self._storage_path is not None:
            self._safe_save()

    # ------------------------------------------------------------------
    # DataFrame view
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return a summary DataFrame of all registered models."""
        with self._lock:
            models = list(self._models.values())

        if not models:
            return pd.DataFrame()

        rows = []
        for m in models:
            rows.append({
                "model_id": m.model_id,
                "task_family": m.task_family.value,
                "model_class": m.model_class,
                "version": m.version,
                "status": m.status.value,
                "governance_status": m.governance_status.value,
                "val_auc": m.val_auc,
                "val_brier": m.val_brier,
                "cv_ic_mean": m.cv_ic_mean,
                "trained_at": m.trained_at,
                "promoted_at": m.promoted_at,
                "retired_at": m.retired_at,
            })

        return pd.DataFrame(rows).set_index("model_id")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist registry metadata to JSON file (if storage_path is set)."""
        if self._storage_path is None:
            return

        with self._lock:
            data = {
                "models": {
                    mid: _metadata_to_dict(m)
                    for mid, m in self._models.items()
                },
                "run_artifacts": {
                    mid: _run_artifact_to_dict(a)
                    for mid, a in self._run_artifacts.items()
                },
                "model_cards": {
                    mid: _model_card_to_dict(c)
                    for mid, c in self._model_cards.items()
                },
                "saved_at": _now_utc(),
            }

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

        logger.debug("Registry saved to %s", self._storage_path)

    def load(self) -> None:
        """Load registry metadata from JSON file."""
        if self._storage_path is None or not self._storage_path.exists():
            return

        with open(self._storage_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        with self._lock:
            for mid, d in data.get("models", {}).items():
                try:
                    self._models[mid] = _metadata_from_dict(d)
                except Exception as exc:
                    logger.warning("Failed to deserialize model %s: %s", mid, exc)

            for mid, d in data.get("run_artifacts", {}).items():
                try:
                    self._run_artifacts[mid] = _run_artifact_from_dict(d)
                except Exception as exc:
                    logger.warning("Failed to deserialize run artifact %s: %s", mid, exc)

            for mid, d in data.get("model_cards", {}).items():
                try:
                    self._model_cards[mid] = _model_card_from_dict(d)
                except Exception as exc:
                    logger.warning("Failed to deserialize model card %s: %s", mid, exc)

        logger.info("Registry loaded from %s (%d models)", self._storage_path, len(self._models))

    def _safe_save(self) -> None:
        try:
            self.save()
        except Exception as exc:
            logger.warning("Auto-save failed: %s", exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_registry_instance: Optional[MLModelRegistry] = None
_registry_lock = threading.Lock()


def get_ml_registry(storage_path: Optional[str | Path] = None) -> MLModelRegistry:
    """Return the process-wide singleton ML registry."""
    global _registry_instance
    with _registry_lock:
        if _registry_instance is None:
            _registry_instance = MLModelRegistry(storage_path=storage_path)
        elif storage_path is not None and _registry_instance._storage_path is None:
            # Lazily attach storage path if not yet set
            _registry_instance._storage_path = Path(storage_path)
    return _registry_instance


# ---------------------------------------------------------------------------
# Internal helpers — dataclass ↔ dict (shallow, avoids heavy dependencies)
# ---------------------------------------------------------------------------

def _update_metadata_status(
    m: ModelMetadata,
    new_status: ModelStatus,
    promoted_at: str = "",
    retired_at: str = "",
) -> ModelMetadata:
    """Return a new ModelMetadata with updated status fields."""
    from dataclasses import replace
    kwargs: dict = {"status": new_status}
    if promoted_at:
        kwargs["promoted_at"] = promoted_at
    if retired_at:
        kwargs["retired_at"] = retired_at
    return replace(m, **kwargs)


def _metadata_to_dict(m: ModelMetadata) -> dict:
    from dataclasses import asdict
    d = asdict(m)
    # Serialize enums
    d["task_family"] = m.task_family.value
    d["status"] = m.status.value
    d["governance_status"] = m.governance_status.value
    # Replace NaN with None for JSON serialization
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}


def _metadata_from_dict(d: dict) -> ModelMetadata:
    from ml.contracts import GovernanceStatus
    d = dict(d)
    d["task_family"] = MLTaskFamily(d["task_family"])
    d["status"] = ModelStatus(d["status"])
    d["governance_status"] = GovernanceStatus(d.get("governance_status", "pending_review"))
    # Restore None → NaN for float fields
    float_fields = {
        "val_auc", "val_brier", "cv_ic_mean", "cv_ic_std",
        "calibration_brier",
    }
    for f in float_fields:
        if f in d and d[f] is None:
            d[f] = float("nan")
    return ModelMetadata(**d)


def _run_artifact_to_dict(a: TrainingRunArtifact) -> dict:
    from dataclasses import asdict
    d = asdict(a)
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}


def _run_artifact_from_dict(d: dict) -> TrainingRunArtifact:
    d = dict(d)
    float_fields = {
        "train_auc", "val_auc", "test_auc",
        "train_brier", "val_brier", "test_brier",
        "cv_ic_mean", "cv_ic_std", "calibration_brier_improvement",
    }
    for f in float_fields:
        if f in d and d[f] is None:
            d[f] = float("nan")
    return TrainingRunArtifact(**d)


def _model_card_to_dict(c: ModelCard) -> dict:
    from dataclasses import asdict
    d = asdict(c)
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}


def _model_card_from_dict(d: dict) -> ModelCard:
    d = dict(d)
    if d.get("primary_metric_value") is None:
        d["primary_metric_value"] = float("nan")
    return ModelCard(**d)


def _get_proba(model: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Helper to extract 1D predicted probabilities from a model."""
    try:
        if hasattr(model, "predict_proba"):
            raw = np.asarray(model.predict_proba(X), dtype=float)
            return raw[:, 1] if raw.ndim == 2 else raw
        return np.asarray(model.predict(X), dtype=float)
    except Exception:
        return None
