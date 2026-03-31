# -*- coding: utf-8 -*-
"""
ml/datasets/splits.py — Temporal Splitting Utilities
======================================================

Leakage-safe temporal train/validation/test split generation.

All splitters enforce a purge window (removes samples near fold boundaries)
and an embargo window (skips additional days to prevent label contamination).

Supported strategies:
  - Chronological: single 60/20/20 split
  - Walk-forward: expanding train window, fixed test window
  - Rolling-origin: fixed-width train window, rolling forward
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import List

import pandas as pd

from ml.contracts import DatasetSplitPlan, SplitPolicy

logger = logging.getLogger("ml.datasets.splits")


class TemporalSplitter:
    """
    Generates leakage-safe temporal train/validation/test splits.

    Supports: chronological, walk-forward (expanding), and rolling-origin.
    All splits enforce purge_days and embargo_days gaps at each boundary.

    purge_days: rows within this many days of any boundary are excluded from both
                adjacent windows. Prevents information leakage via overlapping labels.
    embargo_days: additional buffer after each training window before test begins.
                  Accounts for label horizon contamination.
    """

    def __init__(
        self,
        purge_days: int = 5,
        embargo_days: int = 10,
    ) -> None:
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    # ------------------------------------------------------------------
    # Chronological split
    # ------------------------------------------------------------------

    def chronological_split(
        self,
        index: pd.DatetimeIndex,
        train_frac: float = 0.60,
        val_frac: float = 0.20,
    ) -> DatasetSplitPlan:
        """
        Single chronological split: train | (purge+embargo) | val | (purge+embargo) | test.

        The purge/embargo buffers are carved out of the boundaries; they are not
        counted toward any split. Rows in the buffer zones are simply excluded from
        training and evaluation.

        Returns a DatasetSplitPlan with start/end timestamps for each split.
        """
        if train_frac + val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0 to leave room for test")
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("index must be pd.DatetimeIndex")
        if len(index) < 50:
            raise ValueError(f"Index too short ({len(index)}) for chronological split")

        index = index.sort_values()
        n = len(index)
        buffer_days = self.purge_days + self.embargo_days

        # Raw boundary indices (before buffer carve-out)
        train_end_idx = int(n * train_frac)
        val_end_idx = int(n * (train_frac + val_frac))

        train_end_ts = index[train_end_idx]
        val_end_ts = index[val_end_idx]

        # Apply buffer: val starts after embargo
        val_start_ts = train_end_ts + timedelta(days=buffer_days)
        test_start_ts = val_end_ts + timedelta(days=buffer_days)

        # Guard: ensure val_start <= val_end and test_start <= last index
        if val_start_ts >= val_end_ts:
            logger.warning(
                "Buffer (%d days) is wider than validation window; shrinking buffers", buffer_days
            )
            val_start_ts = train_end_ts + timedelta(days=1)
        if test_start_ts > index[-1]:
            logger.warning("Buffer pushes test_start beyond available data")
            test_start_ts = val_end_ts + timedelta(days=1)

        plan = DatasetSplitPlan(
            split_policy=SplitPolicy.CHRONOLOGICAL,
            train_start=index[0].isoformat(),
            train_end=train_end_ts.isoformat(),
            validation_start=val_start_ts.isoformat(),
            validation_end=val_end_ts.isoformat(),
            test_start=test_start_ts.isoformat(),
            test_end=index[-1].isoformat(),
            purge_days=self.purge_days,
            embargo_days=self.embargo_days,
            n_folds=1,
            fold_index=0,
        )
        logger.debug(
            "Chronological split: train=%s→%s, val=%s→%s, test=%s→%s",
            plan.train_start, plan.train_end,
            plan.validation_start, plan.validation_end,
            plan.test_start, plan.test_end,
        )
        return plan

    # ------------------------------------------------------------------
    # Walk-forward splits (expanding train window)
    # ------------------------------------------------------------------

    def walk_forward_splits(
        self,
        index: pd.DatetimeIndex,
        n_folds: int = 5,
        min_train_periods: int = 252,
        test_periods: int = 63,
    ) -> List[DatasetSplitPlan]:
        """
        Walk-forward expanding-window splits.

        Each fold:
          - Train: index[0] → fold boundary (expanding)
          - Test: fold boundary + embargo → fold boundary + embargo + test_periods

        Returns a list of DatasetSplitPlan, one per fold.
        """
        index = index.sort_values()
        n = len(index)
        buffer = self.purge_days + self.embargo_days
        min_required = min_train_periods + n_folds * (test_periods + buffer)

        if n < min_required:
            raise ValueError(
                f"Index length {n} too short for walk-forward with n_folds={n_folds}, "
                f"min_train_periods={min_train_periods}, test_periods={test_periods}. "
                f"Need >= {min_required}"
            )

        plans: List[DatasetSplitPlan] = []
        # Compute fold boundaries: split remaining data after initial train into n_folds
        available_after_train = n - min_train_periods
        fold_step = available_after_train // n_folds

        for fold in range(n_folds):
            train_end_idx = min_train_periods + fold * fold_step
            test_start_idx = train_end_idx + buffer
            test_end_idx = min(test_start_idx + test_periods, n - 1)

            if test_start_idx >= n or test_end_idx <= test_start_idx:
                logger.warning("Fold %d: insufficient data; skipping", fold)
                continue

            plan = DatasetSplitPlan(
                split_policy=SplitPolicy.WALK_FORWARD,
                train_start=index[0].isoformat(),
                train_end=index[train_end_idx].isoformat(),
                validation_start="",
                validation_end="",
                test_start=index[test_start_idx].isoformat(),
                test_end=index[test_end_idx].isoformat(),
                purge_days=self.purge_days,
                embargo_days=self.embargo_days,
                n_folds=n_folds,
                fold_index=fold,
            )
            plans.append(plan)

        if not plans:
            raise ValueError("No valid walk-forward folds could be generated")

        logger.debug("Generated %d walk-forward folds", len(plans))
        return plans

    # ------------------------------------------------------------------
    # Rolling-origin splits (fixed-width train window)
    # ------------------------------------------------------------------

    def rolling_origin_splits(
        self,
        index: pd.DatetimeIndex,
        initial_train_periods: int = 504,
        test_periods: int = 63,
        step_periods: int = 21,
    ) -> List[DatasetSplitPlan]:
        """
        Rolling-origin splits with a fixed-width train window.

        Each fold:
          - Train: train_start → train_end (window of initial_train_periods)
          - Test: train_end + embargo → train_end + embargo + test_periods
          - Then slide forward by step_periods

        Returns a list of DatasetSplitPlan, one per step.
        """
        index = index.sort_values()
        n = len(index)
        buffer = self.purge_days + self.embargo_days
        min_length = initial_train_periods + buffer + test_periods

        if n < min_length:
            raise ValueError(
                f"Index length {n} < minimum required {min_length} for rolling-origin splits"
            )

        plans: List[DatasetSplitPlan] = []
        fold = 0
        train_start_idx = 0

        while True:
            train_end_idx = train_start_idx + initial_train_periods
            test_start_idx = train_end_idx + buffer
            test_end_idx = test_start_idx + test_periods

            if test_end_idx > n - 1:
                break

            plan = DatasetSplitPlan(
                split_policy=SplitPolicy.ROLLING_ORIGIN,
                train_start=index[train_start_idx].isoformat(),
                train_end=index[train_end_idx].isoformat(),
                validation_start="",
                validation_end="",
                test_start=index[test_start_idx].isoformat(),
                test_end=index[test_end_idx].isoformat(),
                purge_days=self.purge_days,
                embargo_days=self.embargo_days,
                n_folds=0,   # filled in below
                fold_index=fold,
            )
            plans.append(plan)
            train_start_idx += step_periods
            fold += 1

        # Back-fill n_folds now that we know total
        total = len(plans)
        plans = [
            DatasetSplitPlan(
                split_id=p.split_id,
                dataset_def_id=p.dataset_def_id,
                split_policy=p.split_policy,
                train_start=p.train_start,
                train_end=p.train_end,
                validation_start=p.validation_start,
                validation_end=p.validation_end,
                test_start=p.test_start,
                test_end=p.test_end,
                purge_days=p.purge_days,
                embargo_days=p.embargo_days,
                n_folds=total,
                fold_index=p.fold_index,
                created_at=p.created_at,
            )
            for p in plans
        ]

        if not plans:
            raise ValueError("No valid rolling-origin folds could be generated")

        logger.debug("Generated %d rolling-origin folds", len(plans))
        return plans
