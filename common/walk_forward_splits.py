# -*- coding: utf-8 -*-
"""
common/walk_forward_splits.py — Walk-Forward Split Generation
================================================================

Canonical utility for generating walk-forward train/test splits with
purge + embargo for leakage-free validation.

Extracted from root/analysis.py to break core/ → root/ import violation (AP-2).

Design (de Prado / Bailey):
- Expanding-window splits (train grows over time)
- Purge: remove label-horizon overlap at training boundary
- Embargo: gap between train_end and test_start to prevent regime leakage

Usage:
    from common.walk_forward_splits import get_walk_forward_splits

    splits = get_walk_forward_splits(
        index=prices.index,
        n_splits=5,
        test_days=252,
        min_train_days=504,
        embargo_days=20,
    )
    for train_idx, test_idx in splits:
        ...
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardSplit:
    """One walk-forward fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_days: int

    def train_mask(self, index: pd.Index) -> np.ndarray:
        """Boolean mask for training rows."""
        return (index >= self.train_start) & (index <= self.train_end)

    def test_mask(self, index: pd.Index) -> np.ndarray:
        """Boolean mask for test rows."""
        return (index >= self.test_start) & (index <= self.test_end)


def get_walk_forward_splits(
    index: Union[pd.Index, pd.DatetimeIndex, Sequence],
    n_splits: int = 5,
    test_days: int = 252,
    min_train_days: int = 504,
    embargo_days: int = 20,
    purge_days: int = 0,
) -> List[WalkForwardSplit]:
    """
    Generate expanding-window walk-forward splits with purge + embargo.

    Parameters
    ----------
    index : pd.DatetimeIndex or sequence
        Time index of the full dataset.
    n_splits : int
        Number of folds to produce.
    test_days : int
        Size of each test window (trading days).
    min_train_days : int
        Minimum training window (first fold's train size).
    embargo_days : int
        Gap between train_end and test_start (prevents regime leakage).
    purge_days : int
        Extra days trimmed from end of train (for labels that look forward).

    Returns
    -------
    List[WalkForwardSplit]
        Ordered list of folds. Empty list on error (never raises).
    """
    try:
        if not isinstance(index, pd.DatetimeIndex):
            index = pd.DatetimeIndex(index)
        n = len(index)

        required = min_train_days + embargo_days + n_splits * test_days
        if n < required:
            logger.warning(
                "get_walk_forward_splits: insufficient data (%d < %d required)",
                n, required,
            )
            return []

        splits: List[WalkForwardSplit] = []
        for fold in range(n_splits):
            # Expanding train: ends later for each fold
            train_end_pos = min_train_days + fold * test_days - 1
            purge_end_pos = max(0, train_end_pos - purge_days)
            test_start_pos = train_end_pos + embargo_days + 1
            test_end_pos = min(n - 1, test_start_pos + test_days - 1)

            if test_start_pos >= n or test_end_pos <= test_start_pos:
                break

            splits.append(WalkForwardSplit(
                fold_id=fold,
                train_start=index[0],
                train_end=index[purge_end_pos],
                test_start=index[test_start_pos],
                test_end=index[test_end_pos],
                embargo_days=embargo_days,
            ))

        return splits
    except Exception as exc:
        logger.warning("get_walk_forward_splits failed: %s", exc)
        return []


# Backward-compatible alias: returns tuple of (train_idx, test_idx) arrays
def get_walk_forward_splits_as_indices(
    index: Union[pd.Index, pd.DatetimeIndex],
    **kwargs,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compat shim: returns indexer tuples instead of WalkForwardSplit objects."""
    idx = pd.DatetimeIndex(index) if not isinstance(index, pd.DatetimeIndex) else index
    folds = get_walk_forward_splits(idx, **kwargs)
    out = []
    for f in folds:
        train_mask = f.train_mask(idx)
        test_mask = f.test_mask(idx)
        out.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return out
