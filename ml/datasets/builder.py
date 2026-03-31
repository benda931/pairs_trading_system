# -*- coding: utf-8 -*-
"""
ml/datasets/builder.py — Extended ML Dataset Builder
======================================================

Wraps the existing models.dataset_builder.DatasetBuilder with:
  - explicit LeakageAuditReport generation
  - DatasetSnapshot creation (immutable training artifact)
  - multiple label support (returns y as DataFrame, not Series)
  - feature set version tracking
  - walk-forward dataset generation for time-series CV

Entry point for all offline ML training data preparation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.contracts import (
    DatasetSnapshot,
    DatasetSplitPlan,
    FeatureSetVersion,
    LeakageAuditReport,
    SplitPolicy,
)
from ml.datasets.leakage import LeakageAuditor
from ml.datasets.splits import TemporalSplitter
from ml.labels.builder import LabelBuilder
from ml.labels.definitions import LABEL_REGISTRY

logger = logging.getLogger("ml.datasets.builder")


class MLDatasetBuilder:
    """
    Extended dataset builder for the ML platform.

    Wraps models.dataset_builder.DatasetBuilder, extending it with:
    - explicit LeakageAuditReport generation
    - DatasetSnapshot creation (immutable artifact record)
    - multiple label support
    - feature set version tracking

    All temporal boundaries are treated as strict hard cutoffs.
    The train_end parameter is the single most important argument —
    it governs the feature/label isolation boundary.
    """

    def __init__(
        self,
        purge_days: int = 5,
        embargo_days: int = 10,
    ) -> None:
        self._purge_days = purge_days
        self._embargo_days = embargo_days
        self._splitter = TemporalSplitter(purge_days=purge_days, embargo_days=embargo_days)
        self._auditor = LeakageAuditor()
        self._label_builder = LabelBuilder()

    # ------------------------------------------------------------------
    # Primary build method
    # ------------------------------------------------------------------

    def build(
        self,
        pair_id: str,
        px: pd.Series,
        py: pd.Series,
        z: pd.Series,
        feature_set: FeatureSetVersion,
        label_names: List[str],
        train_end: str,
        purge_days: Optional[int] = None,
        embargo_days: Optional[int] = None,
        split_policy: SplitPolicy = SplitPolicy.CHRONOLOGICAL,
        spread: Optional[pd.Series] = None,
        hedge_ratio: Optional[pd.Series] = None,
        adf_pvalues: Optional[pd.Series] = None,
    ) -> Tuple[
        DatasetSnapshot,
        pd.DataFrame,   # X_train
        pd.DataFrame,   # X_test
        pd.DataFrame,   # y_train (multi-label)
        pd.DataFrame,   # y_test  (multi-label)
    ]:
        """
        Build a complete ML dataset.

        Parameters
        ----------
        pair_id:      identifier string for the pair (e.g. "AAPL_MSFT")
        px, py:       price series for each leg
        z:            z-score spread series
        feature_set:  FeatureSetVersion describing which features to compute
        label_names:  list of label names from LABEL_REGISTRY to compute
        train_end:    ISO date string — hard boundary for feature computation
        purge_days:   override instance default if provided
        embargo_days: override instance default if provided
        split_policy: chronological (default), walk-forward, or rolling-origin
        spread:       optional spread series for instability labels
        hedge_ratio:  optional hedge ratio series for HR instability label
        adf_pvalues:  optional ADF p-value series for persistence labels

        Returns
        -------
        (snapshot, X_train, X_test, y_train, y_test)

        snapshot is an immutable artifact record; store it alongside the model.
        """
        purge = purge_days if purge_days is not None else self._purge_days
        embargo = embargo_days if embargo_days is not None else self._embargo_days

        train_end_ts = pd.Timestamp(train_end)

        # ------------------------------------------------------------------
        # Step 1: Build features (past-only, clipped to train_end for training)
        # ------------------------------------------------------------------
        X = self._build_features(px, py, z, feature_set, train_end_ts)

        # ------------------------------------------------------------------
        # Step 2: Build labels (forward-looking, offline training only)
        # ------------------------------------------------------------------
        all_labels_df = self._label_builder.build_all_labels(
            z=z,
            spread=spread,
            hedge_ratio=hedge_ratio,
            adf_pvalues=adf_pvalues,
        )

        # Validate requested label names
        valid_labels = [n for n in label_names if n in all_labels_df.columns]
        missing = [n for n in label_names if n not in all_labels_df.columns]
        if missing:
            logger.warning("Labels not available (missing inputs?): %s", missing)
        if not valid_labels:
            raise ValueError(f"None of the requested labels could be computed: {label_names}")

        Y = all_labels_df[valid_labels]

        # ------------------------------------------------------------------
        # Step 3: Align X and Y on common index
        # ------------------------------------------------------------------
        common_idx = X.index.intersection(Y.index)
        X = X.loc[common_idx]
        Y = Y.loc[common_idx]

        # ------------------------------------------------------------------
        # Step 4: Generate temporal split plan
        # ------------------------------------------------------------------
        splitter = TemporalSplitter(purge_days=purge, embargo_days=embargo)
        if split_policy == SplitPolicy.CHRONOLOGICAL:
            plan = splitter.chronological_split(common_idx)
        elif split_policy == SplitPolicy.WALK_FORWARD:
            plans = splitter.walk_forward_splits(common_idx)
            plan = plans[-1]  # Use last fold for single-build; use build_walk_forward for all
        elif split_policy == SplitPolicy.ROLLING_ORIGIN:
            plans = splitter.rolling_origin_splits(common_idx)
            plan = plans[-1]
        else:
            plan = splitter.chronological_split(common_idx)

        # ------------------------------------------------------------------
        # Step 5: Apply split to X and Y
        # ------------------------------------------------------------------
        X_train, X_test, y_train, y_test = self._apply_split(X, Y, plan)

        # Drop rows with all-NaN labels in train set
        y_train_clean = y_train.dropna(how="all")
        X_train = X_train.loc[y_train_clean.index]
        y_train = y_train_clean

        # ------------------------------------------------------------------
        # Step 6: Leakage audit
        # ------------------------------------------------------------------
        # Use first non-empty label column for the single-label audit
        primary_label = valid_labels[0]
        max_horizon = max(
            (LABEL_REGISTRY[n].horizon_days for n in valid_labels if n in LABEL_REGISTRY),
            default=10,
        )

        audit: LeakageAuditReport = self._auditor.audit(
            X_train=X_train,
            y_train=y_train[primary_label] if primary_label in y_train.columns else y_train.iloc[:, 0],
            X_test=X_test,
            y_test=y_test[primary_label] if primary_label in y_test.columns else y_test.iloc[:, 0],
            train_end=train_end_ts,
            test_start=pd.Timestamp(plan.test_start) if plan.test_start else train_end_ts,
            label_horizon_days=max_horizon,
            embargo_days=embargo,
        )

        if not audit.passed:
            logger.error(
                "Leakage audit FAILED for pair %s: %s", pair_id, audit.violations
            )

        # ------------------------------------------------------------------
        # Step 7: Build DatasetSnapshot
        # ------------------------------------------------------------------
        # Class balance for primary binary label
        class_balance: dict = {}
        if primary_label in y_train.columns:
            vc = y_train[primary_label].value_counts(normalize=True)
            class_balance = {str(k): float(v) for k, v in vc.items()}

        null_profile = {
            col: float(X_train[col].isna().mean())
            for col in X_train.columns
        }

        snapshot = DatasetSnapshot(
            feature_set_id=feature_set.feature_set_id,
            feature_hash=feature_set.feature_hash(),
            label_name=",".join(valid_labels),
            universe_snapshot_id=pair_id,
            time_start=common_idx[0].isoformat() if len(common_idx) > 0 else "",
            time_end=common_idx[-1].isoformat() if len(common_idx) > 0 else "",
            train_rows=len(X_train),
            validation_rows=0,
            test_rows=len(X_test),
            n_features=len(X_train.columns),
            n_entities=1,
            class_balance_train=class_balance,
            null_profile=null_profile,
            leakage_audit_passed=audit.passed,
            leakage_audit_warnings=audit.warnings + audit.violations,
        )

        logger.info(
            "Dataset built for %s: train=%d, test=%d, features=%d, labels=%s, audit=%s",
            pair_id, len(X_train), len(X_test), len(X_train.columns),
            valid_labels, "PASS" if audit.passed else "FAIL",
        )

        return snapshot, X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Walk-forward dataset sequence
    # ------------------------------------------------------------------

    def build_walk_forward_datasets(
        self,
        pair_id: str,
        px: pd.Series,
        py: pd.Series,
        z: pd.Series,
        feature_set: FeatureSetVersion,
        label_name: str,
        n_folds: int = 5,
        spread: Optional[pd.Series] = None,
        hedge_ratio: Optional[pd.Series] = None,
        adf_pvalues: Optional[pd.Series] = None,
    ) -> List[
        Tuple[
            DatasetSnapshot,
            pd.DataFrame,   # X_train
            pd.DataFrame,   # X_test
            pd.Series,      # y_train
            pd.Series,      # y_test
        ]
    ]:
        """
        Generate a sequence of walk-forward datasets for time-series cross-validation.

        Returns one (snapshot, X_train, X_test, y_train, y_test) tuple per fold.
        Each fold uses an expanding train window.

        The caller is responsible for fitting and evaluating a model on each fold
        independently. Never concatenate folds for a single fit.
        """
        # Build full feature and label matrices first
        X = self._build_features(px, py, z, feature_set, train_end_ts=None)
        all_labels_df = self._label_builder.build_all_labels(
            z=z,
            spread=spread,
            hedge_ratio=hedge_ratio,
            adf_pvalues=adf_pvalues,
        )

        if label_name not in all_labels_df.columns:
            raise ValueError(
                f"Label '{label_name}' not available. "
                f"Available: {list(all_labels_df.columns)}"
            )

        Y = all_labels_df[[label_name]]
        common_idx = X.index.intersection(Y.index)
        X = X.loc[common_idx]
        Y = Y.loc[common_idx]

        # Generate walk-forward fold plans
        splitter = TemporalSplitter(
            purge_days=self._purge_days,
            embargo_days=self._embargo_days,
        )
        fold_plans = splitter.walk_forward_splits(common_idx, n_folds=n_folds)

        results = []
        for fold_plan in fold_plans:
            X_train, X_test, y_full_train, y_full_test = self._apply_split(X, Y, fold_plan)

            y_train_s = y_full_train[label_name].dropna()
            X_train = X_train.loc[y_train_s.index]
            y_test_s = y_full_test[label_name]

            max_horizon = (
                LABEL_REGISTRY[label_name].horizon_days
                if label_name in LABEL_REGISTRY
                else 10
            )

            train_end_ts = pd.Timestamp(fold_plan.train_end)
            audit = self._auditor.audit(
                X_train=X_train,
                y_train=y_train_s,
                X_test=X_test,
                y_test=y_test_s,
                train_end=train_end_ts,
                test_start=pd.Timestamp(fold_plan.test_start),
                label_horizon_days=max_horizon,
                embargo_days=self._embargo_days,
            )

            class_balance: dict = {}
            vc = y_train_s.value_counts(normalize=True)
            class_balance = {str(k): float(v) for k, v in vc.items()}

            snapshot = DatasetSnapshot(
                feature_set_id=feature_set.feature_set_id,
                feature_hash=feature_set.feature_hash(),
                label_name=label_name,
                universe_snapshot_id=pair_id,
                time_start=fold_plan.train_start,
                time_end=fold_plan.test_end,
                train_rows=len(X_train),
                test_rows=len(X_test),
                n_features=len(X_train.columns),
                n_entities=1,
                class_balance_train=class_balance,
                leakage_audit_passed=audit.passed,
                leakage_audit_warnings=audit.warnings + audit.violations,
                notes=f"walk_forward fold {fold_plan.fold_index}/{fold_plan.n_folds}",
            )

            results.append((snapshot, X_train, X_test, y_train_s, y_test_s))

        logger.info(
            "Built %d walk-forward folds for pair %s, label %s",
            len(results), pair_id, label_name,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(
        self,
        px: pd.Series,
        py: pd.Series,
        z: pd.Series,
        feature_set: FeatureSetVersion,
        train_end_ts: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        """
        Build the feature matrix from raw price and z-score data.

        Attempts to use ml.features.builder.PointInTimeFeatureBuilder if available.
        Falls back to a minimal baseline feature set if the full builder is not wired.

        All features are clipped to train_end_ts for leakage safety.
        """
        try:
            from ml.features.builder import PointInTimeFeatureBuilder
            pit_builder = PointInTimeFeatureBuilder()
            # PointInTimeFeatureBuilder expects as_of for point-in-time computation;
            # for offline batch use, compute over the full series and clip afterward.
            X = pit_builder.build_dataframe(
                px=px, py=py, z=z,
                feature_names=feature_set.feature_names or None,
            )
        except Exception as exc:
            logger.warning(
                "PointInTimeFeatureBuilder unavailable (%s); using baseline features", exc
            )
            X = self._baseline_features(px, py, z)

        if train_end_ts is not None:
            X = X[X.index <= train_end_ts]

        return X

    @staticmethod
    def _baseline_features(
        px: pd.Series,
        py: pd.Series,
        z: pd.Series,
    ) -> pd.DataFrame:
        """
        Minimal baseline feature set used when full feature builder is unavailable.

        Provides enough signal for basic meta-labeling and regime tasks.
        All features use only backward-looking windows (no lookahead).
        """
        df = pd.DataFrame(index=z.index)
        df["z_score"] = z
        df["z_abs"] = z.abs()
        df["z_lag1"] = z.shift(1)
        df["z_ma5"] = z.rolling(5, min_periods=2).mean()
        df["z_std20"] = z.rolling(20, min_periods=5).std()
        df["z_momentum5"] = z - z.shift(5)
        # Log returns
        if len(px) > 1:
            df["ret_x_1d"] = np.log(px / px.shift(1))
            df["ret_y_1d"] = np.log(py / py.shift(1))
            df["ret_x_5d"] = np.log(px / px.shift(5))
            df["ret_y_5d"] = np.log(py / py.shift(5))
        # Volatility
        if len(px) > 20:
            log_rx = np.log(px / px.shift(1))
            log_ry = np.log(py / py.shift(1))
            df["vol_x_20d"] = log_rx.rolling(20, min_periods=10).std() * np.sqrt(252)
            df["vol_y_20d"] = log_ry.rolling(20, min_periods=10).std() * np.sqrt(252)
        # Rolling correlation
        if len(px) > 20:
            df["corr_20d"] = px.rolling(20, min_periods=10).corr(py)
        return df.dropna(how="all")

    @staticmethod
    def _apply_split(
        X: pd.DataFrame,
        Y: pd.DataFrame,
        plan: DatasetSplitPlan,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply a DatasetSplitPlan to feature and label DataFrames.

        Returns (X_train, X_test, y_train, y_test).
        Validation split is not returned here — use test split for held-out evaluation.
        """
        train_end = pd.Timestamp(plan.train_end)
        test_start = pd.Timestamp(plan.test_start)
        test_end = pd.Timestamp(plan.test_end)

        train_mask = X.index <= train_end
        test_mask = (X.index >= test_start) & (X.index <= test_end)

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = Y.loc[Y.index[Y.index <= train_end]]
        y_test = Y.loc[Y.index[(Y.index >= test_start) & (Y.index <= test_end)]]

        # Align X and y within each split
        train_common = X_train.index.intersection(y_train.index)
        test_common = X_test.index.intersection(y_test.index)

        return (
            X_train.loc[train_common],
            X_test.loc[test_common],
            y_train.loc[train_common],
            y_test.loc[test_common],
        )
