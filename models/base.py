# -*- coding: utf-8 -*-
"""
models/base.py — Model Protocol, Metadata, and Registry
========================================================

Defines the contract that every ML model in this system must satisfy:
  - ModelProtocol: fit/predict/score interface
  - ModelMetadata: full provenance record per trained model
  - ModelRegistry: stores, retrieves, and versions trained models

Design principles:
  - Every model is tagged with the exact dataset it was trained on
  - No model is used without its training provenance being inspectable
  - Leakage prevention is enforced at the registry level (train_end check)
  - Models are versioned and can be compared across parameter configs
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger("models.base")


# ── Model Protocol ────────────────────────────────────────────────

@runtime_checkable
class ModelProtocol(Protocol):
    """
    Interface that every model in this system must satisfy.

    Enforces:
    - fit() takes X, y with an explicit train_end cutoff
    - predict() returns pd.Series indexed to match X
    - score() returns a float metric (higher = better)
    - metadata is always accessible
    """

    @property
    def metadata(self) -> "ModelMetadata":
        ...

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        train_end: Optional[datetime] = None,
        sample_weights: Optional[pd.Series] = None,
    ) -> "ModelProtocol":
        ...

    def predict(self, X: pd.DataFrame) -> pd.Series:
        ...

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        ...


# ── Model Metadata ────────────────────────────────────────────────

@dataclass
class ModelMetadata:
    """
    Full provenance record for a trained model.

    Stored alongside every model in the registry. Provides:
    - What was trained (model class, hyperparameters)
    - When it was trained (timestamp, train window)
    - What data it was trained on (feature hash, n_samples)
    - How well it performed (validation metrics by fold)
    """
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_class: str = ""
    model_version: str = "1.0"
    pair_label: str = ""          # e.g. "AAPL/MSFT"
    feature_names: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Training provenance
    train_start: Optional[datetime] = None
    train_end: Optional[datetime] = None
    n_samples: int = 0
    n_features: int = 0
    feature_hash: str = ""        # SHA256 of feature names + dtypes

    # Validation metrics (from purged K-fold CV)
    cv_sharpe_mean: float = np.nan
    cv_sharpe_std: float = np.nan
    cv_ic_mean: float = np.nan    # Information Coefficient
    cv_ic_std: float = np.nan
    cv_n_folds: int = 0

    # Drift tracking
    last_evaluated_at: Optional[datetime] = None
    psi_score: float = np.nan     # Population Stability Index vs training
    ic_degraded: bool = False
    drift_detected: bool = False

    # Lifecycle
    trained_at: datetime = field(default_factory=datetime.utcnow)
    deprecated_at: Optional[datetime] = None
    is_active: bool = True
    notes: str = ""

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
            elif isinstance(v, np.floating):
                d[k] = float(v)
            else:
                d[k] = v
        return d

    @classmethod
    def _hash_features(cls, feature_names: list[str]) -> str:
        content = json.dumps(sorted(feature_names)).encode()
        return hashlib.sha256(content).hexdigest()[:16]


# ── Base Model ────────────────────────────────────────────────────

class BaseModel:
    """
    Abstract base for all ML models in this system.

    Subclasses must implement:
      - _fit_impl(X_train, y_train, **kwargs) -> self
      - _predict_impl(X) -> np.ndarray

    The base class provides:
      - train_end guard: raises if X contains dates > train_end
      - Automatic metadata population
      - score() with pluggable metric
    """

    MODEL_CLASS: str = "base"

    def __init__(self, hyperparameters: Optional[dict] = None):
        self._hyperparameters = hyperparameters or {}
        self._metadata = ModelMetadata(
            model_class=self.MODEL_CLASS,
            hyperparameters=dict(self._hyperparameters),
        )
        self._fitted = False

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        train_end: Optional[datetime] = None,
        sample_weights: Optional[pd.Series] = None,
        pair_label: str = "",
    ) -> "BaseModel":
        """Fit with leakage guard and automatic metadata capture."""
        # Leakage guard
        if train_end is not None:
            cutoff = pd.Timestamp(train_end)
            future_rows = X.index[X.index > cutoff]
            if len(future_rows) > 0:
                raise ValueError(
                    f"Leakage guard: X contains {len(future_rows)} rows after "
                    f"train_end={train_end}. First offender: {future_rows[0]}"
                )

        # Align X and y
        common_idx = X.index.intersection(y.index)
        X_train = X.loc[common_idx].dropna()
        y_train = y.loc[X_train.index]

        if len(X_train) == 0:
            raise ValueError("No training samples after alignment and dropna")

        # Populate metadata
        self._metadata.pair_label = pair_label
        self._metadata.train_start = X_train.index[0].to_pydatetime() if hasattr(X_train.index[0], 'to_pydatetime') else X_train.index[0]
        self._metadata.train_end = train_end or (X_train.index[-1].to_pydatetime() if hasattr(X_train.index[-1], 'to_pydatetime') else X_train.index[-1])
        self._metadata.n_samples = len(X_train)
        self._metadata.n_features = X_train.shape[1]
        self._metadata.feature_names = list(X_train.columns)
        self._metadata.feature_hash = ModelMetadata._hash_features(self._metadata.feature_names)

        self._fit_impl(X_train, y_train, sample_weights=sample_weights)
        self._fitted = True
        logger.debug(
            "Fitted %s on %d samples, %d features",
            self.MODEL_CLASS, len(X_train), X_train.shape[1],
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        raw = self._predict_impl(X.dropna())
        return pd.Series(raw, index=X.dropna().index, name="prediction")

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Information Coefficient (rank correlation) between predictions and labels."""
        preds = self.predict(X)
        common = preds.index.intersection(y.index)
        if len(common) < 5:
            return np.nan
        return float(preds.loc[common].corr(y.loc[common], method="spearman"))

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: Optional[pd.Series],
    ) -> None:
        raise NotImplementedError

    def _predict_impl(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# ── Model Registry ────────────────────────────────────────────────

class ModelRegistry:
    """
    Stores and retrieves trained models with full provenance.

    In-memory for now. Extend with DuckDB/SQLite persistence as needed.
    """

    def __init__(self):
        self._models: dict[str, BaseModel] = {}          # model_id -> model
        self._by_pair: dict[str, list[str]] = {}          # pair_label -> [model_ids]
        self._lock = threading.Lock()

    def register(self, model: BaseModel) -> str:
        """Store a trained model. Returns model_id."""
        if not model._fitted:
            raise ValueError("Cannot register unfitted model")

        mid = model.metadata.model_id
        pair = model.metadata.pair_label or "global"

        with self._lock:
            self._models[mid] = model
            if pair not in self._by_pair:
                self._by_pair[pair] = []
            self._by_pair[pair].append(mid)

        logger.info(
            "Registered model %s for pair '%s' (n_samples=%d)",
            mid[:8], pair, model.metadata.n_samples,
        )
        return mid

    def get(self, model_id: str) -> Optional[BaseModel]:
        return self._models.get(model_id)

    def get_latest_for_pair(self, pair_label: str) -> Optional[BaseModel]:
        """Return the most recently trained model for a pair."""
        with self._lock:
            ids = self._by_pair.get(pair_label, [])
        if not ids:
            return None
        # Sort by trained_at
        models = [self._models[mid] for mid in ids if mid in self._models]
        models.sort(key=lambda m: m.metadata.trained_at, reverse=True)
        return models[0] if models else None

    def list_models(
        self,
        pair_label: Optional[str] = None,
        active_only: bool = True,
    ) -> list[ModelMetadata]:
        with self._lock:
            models = list(self._models.values())

        if pair_label:
            models = [m for m in models if m.metadata.pair_label == pair_label]
        if active_only:
            models = [m for m in models if m.metadata.is_active]

        return [m.metadata for m in models]

    def deprecate(self, model_id: str, reason: str = "") -> None:
        """Mark a model as inactive."""
        model = self._models.get(model_id)
        if model:
            model.metadata.is_active = False
            model.metadata.deprecated_at = datetime.utcnow()
            model.metadata.notes = reason
            logger.info("Deprecated model %s: %s", model_id[:8], reason)

    def to_dataframe(self) -> pd.DataFrame:
        """Summary DataFrame of all registered models."""
        rows = [m.metadata.to_dict() for m in self._models.values()]
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Singleton ─────────────────────────────────────────────────────

_default_model_registry: Optional[ModelRegistry] = None
_mr_lock = threading.Lock()


def get_model_registry() -> ModelRegistry:
    global _default_model_registry
    with _mr_lock:
        if _default_model_registry is None:
            _default_model_registry = ModelRegistry()
    return _default_model_registry
