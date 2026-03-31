# -*- coding: utf-8 -*-
"""
models/feature_store.py — Structured Feature Definitions
=========================================================

Provides:
  - FeatureDefinition: named, typed, documented feature spec
  - FeatureGroup: logical collection of related features
  - FeatureStore: registry mapping names to computation functions

This allows consistent feature engineering across:
  - Walk-forward backtests
  - Live signal generation
  - Model comparison experiments

All feature computations are stateless pure functions that take
a pd.DataFrame and return a pd.Series (or pd.DataFrame for groups).

Usage:
    store = FeatureStore.default()
    X = store.compute(["z_level", "z_momentum_20d", "vol_ratio_20d"], prices, z)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("models.feature_store")


# ── Feature Definition ────────────────────────────────────────────

@dataclass
class FeatureDefinition:
    """
    Describes a single feature: name, type, description, compute function.
    """
    name: str
    description: str
    category: str          # "spread", "return", "volatility", "regime", "macro"
    lookback_days: int     # minimum days of history needed to compute this feature
    compute_fn: Callable   # fn(context: FeatureContext) -> pd.Series
    normalise: bool = True # whether to z-normalise this feature before use in models

    def __repr__(self) -> str:
        return f"FeatureDefinition(name={self.name!r}, category={self.category!r}, lookback={self.lookback_days})"


@dataclass
class FeatureContext:
    """Input context passed to all feature compute functions."""
    z: pd.Series             # Spread z-score
    px: pd.Series            # Price series for sym_x
    py: pd.Series            # Price series for sym_y
    extra: dict = field(default_factory=dict)  # Optional: macro, regime data


# ── Built-in feature implementations ─────────────────────────────

def _z_level(ctx: FeatureContext) -> pd.Series:
    return ctx.z.rename("z_level")

def _z_abs(ctx: FeatureContext) -> pd.Series:
    return ctx.z.abs().rename("z_abs")

def _z_sign(ctx: FeatureContext) -> pd.Series:
    return np.sign(ctx.z).rename("z_sign")

def _z_mean_5d(ctx: FeatureContext) -> pd.Series:
    return ctx.z.rolling(5).mean().rename("z_mean_5d")

def _z_mean_20d(ctx: FeatureContext) -> pd.Series:
    return ctx.z.rolling(20).mean().rename("z_mean_20d")

def _z_std_20d(ctx: FeatureContext) -> pd.Series:
    return ctx.z.rolling(20).std().rename("z_std_20d")

def _z_momentum_5d(ctx: FeatureContext) -> pd.Series:
    return (ctx.z - ctx.z.shift(5)).rename("z_momentum_5d")

def _z_momentum_20d(ctx: FeatureContext) -> pd.Series:
    return (ctx.z - ctx.z.shift(20)).rename("z_momentum_20d")

def _z_ar1_20d(ctx: FeatureContext) -> pd.Series:
    return ctx.z.rolling(20).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else np.nan,
        raw=True,
    ).rename("z_ar1_20d")

def _z_cross_zero_20d(ctx: FeatureContext) -> pd.Series:
    return (np.sign(ctx.z) != np.sign(ctx.z.shift(1))).rolling(20).mean().rename("z_cross_zero_20d")

def _ret_x_1d(ctx: FeatureContext) -> pd.Series:
    return np.log(ctx.px).diff().rename("ret_x_1d")

def _ret_y_1d(ctx: FeatureContext) -> pd.Series:
    return np.log(ctx.py).diff().rename("ret_y_1d")

def _spread_ret_1d(ctx: FeatureContext) -> pd.Series:
    return (np.log(ctx.px).diff() - np.log(ctx.py).diff()).rename("spread_ret_1d")

def _vol_x_20d(ctx: FeatureContext) -> pd.Series:
    return np.log(ctx.px).diff().rolling(20).std().rename("vol_x_20d")

def _vol_y_20d(ctx: FeatureContext) -> pd.Series:
    return np.log(ctx.py).diff().rolling(20).std().rename("vol_y_20d")

def _vol_ratio_20d(ctx: FeatureContext) -> pd.Series:
    vx = np.log(ctx.px).diff().rolling(20).std()
    vy = np.log(ctx.py).diff().rolling(20).std().clip(lower=1e-8)
    return (vx / vy).rename("vol_ratio_20d")

def _corr_20d(ctx: FeatureContext) -> pd.Series:
    rx = np.log(ctx.px).diff()
    ry = np.log(ctx.py).diff()
    return rx.rolling(20).corr(ry).rename("corr_20d")

def _corr_60d(ctx: FeatureContext) -> pd.Series:
    rx = np.log(ctx.px).diff()
    ry = np.log(ctx.py).diff()
    return rx.rolling(60).corr(ry).rename("corr_60d")

def _rel_momentum_20d(ctx: FeatureContext) -> pd.Series:
    rx = np.log(ctx.px).diff().rolling(20).sum()
    ry = np.log(ctx.py).diff().rolling(20).sum()
    return (rx - ry).rename("rel_momentum_20d")

def _bb_position_20d(ctx: FeatureContext) -> pd.Series:
    mu = ctx.z.rolling(20).mean()
    sigma = ctx.z.rolling(20).std().clip(lower=1e-8)
    return ((ctx.z - mu) / sigma).rename("bb_position_20d")

def _bb_position_60d(ctx: FeatureContext) -> pd.Series:
    mu = ctx.z.rolling(60).mean()
    sigma = ctx.z.rolling(60).std().clip(lower=1e-8)
    return ((ctx.z - mu) / sigma).rename("bb_position_60d")


# ── Feature Store ─────────────────────────────────────────────────

_BUILTIN_FEATURES: list[FeatureDefinition] = [
    FeatureDefinition("z_level", "Current z-score level", "spread", 1, _z_level),
    FeatureDefinition("z_abs", "Absolute z-score", "spread", 1, _z_abs),
    FeatureDefinition("z_sign", "Sign of z-score", "spread", 1, _z_sign, normalise=False),
    FeatureDefinition("z_mean_5d", "5-day rolling mean of z", "spread", 5, _z_mean_5d),
    FeatureDefinition("z_mean_20d", "20-day rolling mean of z", "spread", 20, _z_mean_20d),
    FeatureDefinition("z_std_20d", "20-day rolling std of z", "spread", 20, _z_std_20d),
    FeatureDefinition("z_momentum_5d", "z[t] - z[t-5]", "spread", 5, _z_momentum_5d),
    FeatureDefinition("z_momentum_20d", "z[t] - z[t-20]", "spread", 20, _z_momentum_20d),
    FeatureDefinition("z_ar1_20d", "AR(1) coefficient (rolling 20d)", "spread", 20, _z_ar1_20d),
    FeatureDefinition("z_cross_zero_20d", "Zero-crossing frequency (20d)", "spread", 20, _z_cross_zero_20d),
    FeatureDefinition("ret_x_1d", "1-day log return of sym_x", "return", 1, _ret_x_1d),
    FeatureDefinition("ret_y_1d", "1-day log return of sym_y", "return", 1, _ret_y_1d),
    FeatureDefinition("spread_ret_1d", "1-day spread return (ret_x - ret_y)", "return", 1, _spread_ret_1d),
    FeatureDefinition("vol_x_20d", "20-day realised vol of sym_x", "volatility", 20, _vol_x_20d),
    FeatureDefinition("vol_y_20d", "20-day realised vol of sym_y", "volatility", 20, _vol_y_20d),
    FeatureDefinition("vol_ratio_20d", "Vol(x)/Vol(y) ratio", "volatility", 20, _vol_ratio_20d),
    FeatureDefinition("corr_20d", "20-day return correlation", "return", 20, _corr_20d),
    FeatureDefinition("corr_60d", "60-day return correlation", "return", 60, _corr_60d),
    FeatureDefinition("rel_momentum_20d", "20-day relative momentum (x vs y)", "return", 20, _rel_momentum_20d),
    FeatureDefinition("bb_position_20d", "Bollinger-band position (20d)", "spread", 20, _bb_position_20d),
    FeatureDefinition("bb_position_60d", "Bollinger-band position (60d)", "spread", 60, _bb_position_60d),
]


class FeatureStore:
    """
    Registry of named feature definitions with batch compute support.
    """

    def __init__(self):
        self._features: dict[str, FeatureDefinition] = {}

    def register(self, feat: FeatureDefinition) -> None:
        self._features[feat.name] = feat

    def get(self, name: str) -> Optional[FeatureDefinition]:
        return self._features.get(name)

    def list_features(self, category: Optional[str] = None) -> list[FeatureDefinition]:
        feats = list(self._features.values())
        if category:
            feats = [f for f in feats if f.category == category]
        return sorted(feats, key=lambda f: f.name)

    def compute_one(
        self,
        name: str,
        ctx: FeatureContext,
    ) -> pd.Series:
        """Compute a single named feature."""
        feat = self._features.get(name)
        if feat is None:
            raise KeyError(f"Feature '{name}' not found in store")
        return feat.compute_fn(ctx)

    def compute(
        self,
        names: list[str],
        ctx: FeatureContext,
        *,
        drop_na_rows: bool = False,
    ) -> pd.DataFrame:
        """
        Compute multiple named features into a DataFrame.

        Parameters
        ----------
        names : list of feature names to compute
        ctx : FeatureContext
        drop_na_rows : if True, drop rows where any feature is NaN

        Returns
        -------
        pd.DataFrame with one column per feature, DatetimeIndex
        """
        series_list = []
        for name in names:
            try:
                s = self.compute_one(name, ctx)
                series_list.append(s)
            except Exception as exc:
                logger.warning("Failed to compute feature '%s': %s", name, exc)
                # Insert NaN column so downstream code doesn't silently drop the feature
                series_list.append(pd.Series(np.nan, index=ctx.z.index, name=name))

        X = pd.concat(series_list, axis=1)

        if drop_na_rows:
            X = X.dropna()

        return X

    def compute_all(
        self,
        ctx: FeatureContext,
        *,
        categories: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Compute all registered features (optionally filtered by category)."""
        names = [
            f.name for f in self.list_features()
            if categories is None or f.category in categories
        ]
        return self.compute(names, ctx)

    @classmethod
    def default(cls) -> "FeatureStore":
        """Return a store pre-loaded with all built-in features."""
        store = cls()
        for feat in _BUILTIN_FEATURES:
            store.register(feat)
        return store

    def required_lookback(self, names: list[str]) -> int:
        """Return the minimum history needed to compute a feature set."""
        return max(
            (self._features[n].lookback_days for n in names if n in self._features),
            default=0,
        )
