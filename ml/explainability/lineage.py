# -*- coding: utf-8 -*-
"""
ml/explainability/lineage.py — Feature Lineage Tracking
=========================================================

Tracks the complete lineage of features used in ML models.

Answers questions like:
- Which features were used by model X?
- How were they computed?
- Which feature set version was active?
- What raw data was required?
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ml.contracts import FeatureLineageRecord
from ml.features.definitions import FEATURE_REGISTRY

logger = logging.getLogger("ml.explainability.lineage")


class FeatureLineageTracker:
    """
    Tracks the complete lineage of features used in ML models.

    Answers questions like:
    - Which features were used by model X?
    - How were they computed?
    - Which feature set version was active?
    - What raw data was required?

    Thread-safe for read operations; writes should be done during training only.
    """

    def __init__(self) -> None:
        # model_id -> list of FeatureLineageRecord
        self._records: Dict[str, List[FeatureLineageRecord]] = {}

    # ------------------------------------------------------------------
    # Core recording
    # ------------------------------------------------------------------

    def record_training_lineage(
        self,
        model_id: str,
        feature_names: List[str],
        feature_set_id: str,
        train_start: str,
        train_end: str,
        dataset_snapshot_id: str,
    ) -> List[FeatureLineageRecord]:
        """
        Create FeatureLineageRecords for a training run.

        One record is created per feature. Records are stored internally
        and also returned for the caller to persist or attach to a
        TrainingRunArtifact.

        Parameters
        ----------
        model_id : str
            Identifier of the model being trained.
        feature_names : list[str]
            Names of all features used during training.
        feature_set_id : str
            Identifier of the FeatureSetVersion used.
        train_start : str
            ISO-8601 timestamp of training data start.
        train_end : str
            ISO-8601 timestamp of training data end (hard boundary).
        dataset_snapshot_id : str
            Identifier of the DatasetSnapshot this lineage corresponds to.

        Returns
        -------
        list[FeatureLineageRecord]
            One record per feature, populated from FEATURE_REGISTRY where available.
        """
        records: List[FeatureLineageRecord] = []
        now = datetime.now(tz=timezone.utc).isoformat()

        for name in feature_names:
            defn = FEATURE_REGISTRY.get(name)

            inputs_used: List[str] = []
            transform_steps: List[str] = []
            version = "1.0"

            if defn is not None:
                inputs_used = list(defn.required_inputs)
                version = defn.version
                transform_steps = [defn.notes] if defn.notes else []
                output_shape = "scalar"
            else:
                output_shape = "unknown"

            record = FeatureLineageRecord(
                feature_name=name,
                feature_version=version,
                entity_id=dataset_snapshot_id,
                as_of=train_end,
                inputs_used=inputs_used,
                lookback_start=train_start,
                lookback_end=train_end,
                transform_steps=transform_steps,
                output_shape=output_shape,
                null_fraction=0.0,
                recorded_at=now,
            )
            records.append(record)

        # Store under model_id (append if model already has records from prior runs)
        if model_id not in self._records:
            self._records[model_id] = []
        self._records[model_id].extend(records)

        logger.info(
            "FeatureLineageTracker: recorded %d feature records for model %s",
            len(records),
            model_id,
        )
        return records

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_model_lineage(self, model_id: str) -> List[FeatureLineageRecord]:
        """
        Retrieve all lineage records for a model.

        Returns an empty list if no records are found.
        """
        return list(self._records.get(model_id, []))

    def list_model_ids(self) -> List[str]:
        """Return all model IDs that have lineage records."""
        return list(self._records.keys())

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_lineage_report(self, model_id: str) -> dict:
        """
        Human-readable lineage report for a model.

        Returns a dict with:
        - model_id
        - n_features
        - features: list of {name, version, inputs, steps}
        - all_raw_inputs: deduplicated set of raw data series required
        - coverage: fraction of features found in FEATURE_REGISTRY
        """
        records = self.get_model_lineage(model_id)
        if not records:
            return {
                "model_id": model_id,
                "n_features": 0,
                "features": [],
                "all_raw_inputs": [],
                "coverage": 0.0,
                "warning": "No lineage records found for this model.",
            }

        all_inputs: set[str] = set()
        registered_count = 0
        features_info = []

        for rec in records:
            all_inputs.update(rec.inputs_used)
            in_registry = rec.feature_name in FEATURE_REGISTRY
            if in_registry:
                registered_count += 1

            features_info.append({
                "name": rec.feature_name,
                "version": rec.feature_version,
                "inputs": list(rec.inputs_used),
                "transform_steps": rec.transform_steps,
                "lookback_start": rec.lookback_start,
                "lookback_end": rec.lookback_end,
                "in_registry": in_registry,
            })

        coverage = registered_count / len(records) if records else 0.0

        return {
            "model_id": model_id,
            "n_features": len(records),
            "features": features_info,
            "all_raw_inputs": sorted(all_inputs),
            "coverage": coverage,
        }

    # ------------------------------------------------------------------
    # Compatibility check
    # ------------------------------------------------------------------

    def check_feature_compatibility(
        self,
        model_feature_names: List[str],
        available_feature_names: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Check if the features required by a model are available.

        Parameters
        ----------
        model_feature_names : list[str]
            Features the model was trained on / expects at inference.
        available_feature_names : list[str]
            Features currently available in the inference environment.

        Returns
        -------
        (ok, missing) : (bool, list[str])
            ok is True if all required features are available.
            missing lists any feature in model_feature_names not in available.
        """
        available_set = set(available_feature_names)
        missing = [f for f in model_feature_names if f not in available_set]
        ok = len(missing) == 0
        return ok, missing
