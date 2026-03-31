# -*- coding: utf-8 -*-
"""
models/dataset_builder.py — Leakage-Safe Dataset Construction
=============================================================

Constructs ML training datasets for pairs trading signals with
strict temporal isolation:
  - Features are always computed from data available at time T
  - Labels (forward returns) are always from T+1 onwards
  - Purge window: remove samples within N days of each fold boundary
  - Embargo window: skip N additional days after the purge

The output is a DatasetBundle — a self-describing container that
records exactly what was used to build each dataset, so models
trained from it carry full provenance.

Usage:
    builder = DatasetBuilder(horizon=5, purge_days=10, embargo_days=5)
    bundle = builder.build(pair_id, prices, z_scores, spread_defn)
    X_train, y_train = bundle.train_split(train_end)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from core.contracts import PairId, SpreadDefinition

logger = logging.getLogger("models.dataset_builder")


# ── DatasetBundle ─────────────────────────────────────────────────

@dataclass
class DatasetBundle:
    """
    Self-describing ML dataset for a single pair.

    Contains features (X), labels (y), and full provenance metadata.
    Provides leakage-safe train/test splits.
    """
    pair_id: PairId
    X: pd.DataFrame                # Feature matrix, DatetimeIndex
    y: pd.Series                   # Forward return labels, DatetimeIndex

    # Provenance
    feature_names: list[str] = field(default_factory=list)
    label_name: str = "fwd_return"
    horizon_days: int = 5
    purge_days: int = 10
    embargo_days: int = 5
    built_at: datetime = field(default_factory=datetime.utcnow)
    n_raw_samples: int = 0
    n_clean_samples: int = 0
    spread_model: str = ""

    def train_split(
        self,
        train_end: datetime,
        *,
        purge: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return (X_train, y_train) strictly before train_end.

        With purge=True, removes samples within purge_days of train_end
        to prevent label leakage across the fold boundary.
        """
        cutoff = pd.Timestamp(train_end)
        mask = self.X.index < cutoff

        if purge:
            purge_start = cutoff - pd.Timedelta(days=self.purge_days)
            mask &= ~((self.X.index >= purge_start) & (self.X.index < cutoff))

        X_tr = self.X.loc[mask]
        y_tr = self.y.loc[X_tr.index]
        return X_tr, y_tr

    def test_split(
        self,
        test_start: datetime,
        test_end: Optional[datetime] = None,
        *,
        apply_embargo: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Return (X_test, y_test) in [test_start + embargo, test_end].

        apply_embargo=True skips embargo_days after test_start.
        """
        start = pd.Timestamp(test_start)
        if apply_embargo:
            start = start + pd.Timedelta(days=self.embargo_days)

        mask = self.X.index >= start
        if test_end is not None:
            mask &= self.X.index <= pd.Timestamp(test_end)

        X_te = self.X.loc[mask]
        y_te = self.y.loc[X_te.index]
        return X_te, y_te

    def summary(self) -> dict:
        return {
            "pair_label": self.pair_id.label,
            "n_samples": len(self.X),
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "horizon_days": self.horizon_days,
            "date_range": f"{self.X.index[0].date()} → {self.X.index[-1].date()}" if len(self.X) > 0 else "empty",
            "label_coverage": float((~self.y.isna()).mean()),
            "spread_model": self.spread_model,
            "built_at": self.built_at.isoformat(),
        }


# ── Feature definitions ───────────────────────────────────────────

def _compute_z_features(z: pd.Series, windows: list[int] = (5, 10, 20, 60)) -> pd.DataFrame:
    """Z-score based features: level, momentum, volatility, mean-reversion speed."""
    feats = {}
    feats["z"] = z
    feats["z_abs"] = z.abs()
    feats["z_sign"] = np.sign(z)

    for w in windows:
        feats[f"z_mean_{w}d"] = z.rolling(w).mean()
        feats[f"z_std_{w}d"] = z.rolling(w).std()
        feats[f"z_momentum_{w}d"] = z - z.shift(w)

    # AR(1) coefficient (rolling) — proxy for mean-reversion speed
    feats["z_ar1_20d"] = z.rolling(20).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else np.nan,
        raw=True,
    )

    # Crossing-zero frequency (how often z crosses zero = mean-reverting signal)
    feats["z_cross_zero_20d"] = (np.sign(z) != np.sign(z.shift(1))).rolling(20).mean()

    return pd.DataFrame(feats)


def _compute_return_features(
    px: pd.Series,
    py: pd.Series,
    windows: list[int] = (5, 10, 20),
) -> pd.DataFrame:
    """Return-based features for each leg and their spread."""
    feats = {}
    rx = np.log(px).diff()
    ry = np.log(py).diff()
    spread_ret = rx - ry

    feats["ret_x_1d"] = rx
    feats["ret_y_1d"] = ry
    feats["spread_ret_1d"] = spread_ret

    for w in windows:
        feats[f"ret_x_{w}d"] = rx.rolling(w).sum()
        feats[f"ret_y_{w}d"] = ry.rolling(w).sum()
        feats[f"vol_x_{w}d"] = rx.rolling(w).std()
        feats[f"vol_y_{w}d"] = ry.rolling(w).std()
        feats[f"corr_{w}d"] = rx.rolling(w).corr(ry)

    # Relative volume/momentum
    feats["rel_momentum_20d"] = feats["ret_x_20d"] - feats["ret_y_20d"]
    feats["vol_ratio_20d"] = feats["vol_x_20d"] / (feats["vol_y_20d"] + 1e-8)

    return pd.DataFrame(feats)


def _compute_spread_features(spread_raw: pd.Series) -> pd.DataFrame:
    """Raw spread (not normalised) features for stationarity/trend signals."""
    feats = {}
    feats["spread_raw"] = spread_raw
    feats["spread_diff"] = spread_raw.diff()
    feats["spread_acc"] = spread_raw.diff().diff()

    # Bollinger band position
    for w in (20, 60):
        mu = spread_raw.rolling(w).mean()
        sigma = spread_raw.rolling(w).std().clip(lower=1e-8)
        feats[f"bb_position_{w}d"] = (spread_raw - mu) / sigma

    return pd.DataFrame(feats)


def _compute_forward_return(
    z: pd.Series,
    horizon: int,
    *,
    method: str = "z_reversion",
) -> pd.Series:
    """
    Compute forward label.

    method="z_reversion": label = -(z[t+horizon] - z[t])
      Positive label = z moved toward zero = profitable mean reversion.

    method="return": label = raw forward price return of spread.
    """
    if method == "z_reversion":
        # Forward z relative to current: if we short the spread (short when z>0),
        # we profit when z falls, so label = z[t] - z[t+horizon]
        return (z - z.shift(-horizon)).rename("fwd_z_reversion")
    else:
        raise ValueError(f"Unknown label method: {method}")


# ── DatasetBuilder ────────────────────────────────────────────────

class DatasetBuilder:
    """
    Constructs leakage-safe ML datasets for pairs trading.

    For each timestamp T, features use data up to and including T.
    Labels use data strictly after T (shifted by horizon).
    """

    def __init__(
        self,
        horizon: int = 5,
        purge_days: int = 10,
        embargo_days: int = 5,
        z_windows: tuple = (5, 10, 20, 60),
        return_windows: tuple = (5, 10, 20),
        label_method: str = "z_reversion",
        min_samples: int = 100,
    ):
        self.horizon = horizon
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.z_windows = list(z_windows)
        self.return_windows = list(return_windows)
        self.label_method = label_method
        self.min_samples = min_samples

    def build(
        self,
        pair_id: PairId,
        prices: pd.DataFrame,
        z_scores: pd.Series,
        spread_defn: Optional[SpreadDefinition] = None,
        *,
        train_end: Optional[datetime] = None,
    ) -> DatasetBundle:
        """
        Build a complete DatasetBundle.

        Parameters
        ----------
        pair_id : PairId
        prices : pd.DataFrame with sym_x and sym_y columns
        z_scores : pd.Series of spread z-scores (DatetimeIndex)
        spread_defn : SpreadDefinition (for raw spread computation)
        train_end : if set, only include data up to this date

        Returns
        -------
        DatasetBundle
        """
        sym_x, sym_y = pair_id.sym_x, pair_id.sym_y
        if sym_x not in prices.columns or sym_y not in prices.columns:
            raise ValueError(f"prices must contain {sym_x} and {sym_y}")

        px = prices[sym_x]
        py = prices[sym_y]

        # Apply train_end cutoff
        if train_end is not None:
            cutoff = pd.Timestamp(train_end)
            # Include horizon extra days to compute labels right up to train_end
            extended_end = cutoff + pd.Timedelta(days=self.horizon + 5)
            px = px[px.index <= extended_end]
            py = py[py.index <= extended_end]
            z_scores = z_scores[z_scores.index <= extended_end]

        # Compute features
        feature_frames = [
            _compute_z_features(z_scores, windows=self.z_windows),
            _compute_return_features(px, py, windows=self.return_windows),
        ]

        # Raw spread features if we have a definition
        if spread_defn is not None:
            import numpy as np
            lx = np.log(px.clip(lower=1e-8))
            ly = np.log(py.clip(lower=1e-8))
            raw_spread = lx - spread_defn.hedge_ratio * ly - spread_defn.intercept
            feature_frames.append(_compute_spread_features(raw_spread))

        X = pd.concat(feature_frames, axis=1)

        # Compute labels
        y = _compute_forward_return(z_scores, self.horizon, method=self.label_method)

        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Apply hard train_end cutoff to features (labels need the lookahead)
        if train_end is not None:
            X = X[X.index <= pd.Timestamp(train_end)]
            y = y[y.index <= pd.Timestamp(train_end)]

        n_raw = len(X)

        # Drop rows where label is NaN (last `horizon` rows)
        valid_mask = ~y.isna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        n_clean = len(X)

        if n_clean < self.min_samples:
            logger.warning(
                "%s: only %d clean samples (min=%d)",
                pair_id.label, n_clean, self.min_samples,
            )

        bundle = DatasetBundle(
            pair_id=pair_id,
            X=X,
            y=y,
            feature_names=list(X.columns),
            label_name=f"fwd_z_reversion_{self.horizon}d",
            horizon_days=self.horizon,
            purge_days=self.purge_days,
            embargo_days=self.embargo_days,
            n_raw_samples=n_raw,
            n_clean_samples=n_clean,
            spread_model=spread_defn.model.value if spread_defn else "unknown",
        )

        logger.debug(
            "%s: built dataset, %d samples, %d features, horizon=%d",
            pair_id.label, n_clean, X.shape[1], self.horizon,
        )
        return bundle

    def build_for_universe(
        self,
        pair_ids: list[PairId],
        prices: pd.DataFrame,
        z_scores_map: dict[str, pd.Series],
        spread_defn_map: Optional[dict[str, SpreadDefinition]] = None,
        *,
        train_end: Optional[datetime] = None,
    ) -> dict[str, DatasetBundle]:
        """
        Build datasets for multiple pairs.

        Parameters
        ----------
        z_scores_map : {pair_label: z_series}
        spread_defn_map : {pair_label: SpreadDefinition} (optional)
        """
        bundles = {}
        for pid in pair_ids:
            z = z_scores_map.get(pid.label)
            if z is None:
                logger.warning("No z-scores for %s — skipping", pid.label)
                continue
            defn = (spread_defn_map or {}).get(pid.label)
            try:
                bundle = self.build(pid, prices, z, defn, train_end=train_end)
                bundles[pid.label] = bundle
            except Exception as exc:
                logger.error("Failed to build dataset for %s: %s", pid.label, exc)
        return bundles
