# -*- coding: utf-8 -*-
"""
ml/features/builder.py — Point-in-Time Feature Builder
=======================================================

PointInTimeFeatureBuilder is the live/online feature computation path.
It computes any registered feature at a specific point in time without
ever touching data beyond `as_of`.

This is intentionally separate from DatasetBuilder (offline training),
which builds feature matrices across many timestamps in one pass.

Design rules:
  - Every compute_* method clips data to `as_of` FIRST, before any math.
  - Insufficient lookback → feature is omitted with a warning (no NaN bleed).
  - No method raises — all errors are caught and surfaced as warnings in the
    returned FeatureSnapshot.  The caller checks warnings, not exceptions.
  - All arithmetic uses numpy/pandas; scipy only for skew/kurtosis.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ml.contracts import (
    EntityScope,
    FeatureCategory,
    FeatureDefinition,
    FeatureGroup,
    FeatureSnapshot,
)
from ml.features.definitions import FEATURE_GROUPS, FEATURE_REGISTRY

logger = logging.getLogger("ml.features.builder")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ANNUALISE = math.sqrt(252)
_ENTRY_Z_THRESHOLD = 2.0  # used for pair_entry_attempts_5d


def _to_datetime(ts: Union[datetime, str]) -> datetime:
    """Coerce str/datetime to timezone-aware datetime (UTC)."""
    if isinstance(ts, str):
        dt = pd.Timestamp(ts)
    else:
        dt = pd.Timestamp(ts)
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return dt.to_pydatetime()


def _clip(series: pd.Series, as_of: datetime) -> pd.Series:
    """Return series with all data after as_of removed."""
    if series.empty:
        return series
    idx = series.index
    if hasattr(idx, "tz") and idx.tz is None:
        # Localise naive index to UTC so comparison works
        idx = idx.tz_localize("UTC")
        series = series.copy()
        series.index = idx
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    return series[series.index <= as_of_ts]


def _log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def _ar1_coef(series: pd.Series) -> float:
    """OLS AR(1) coefficient.  Returns NaN if insufficient data."""
    s = series.dropna()
    if len(s) < 4:
        return float("nan")
    y = s.values[1:]
    x = s.values[:-1]
    xm = x - x.mean()
    denom = float(np.dot(xm, xm))
    if denom == 0:
        return float("nan")
    return float(np.dot(xm, y - y.mean()) / denom)


def _ols_slope(y: np.ndarray) -> float:
    """Slope of OLS(y ~ t) where t = [0,1,...,n-1]."""
    n = len(y)
    if n < 2:
        return float("nan")
    t = np.arange(n, dtype=float)
    t -= t.mean()
    denom = float(np.dot(t, t))
    if denom == 0:
        return float("nan")
    return float(np.dot(t, y - y.mean()) / denom)


def _half_life_from_ar1(ar1: float) -> float:
    """Half-life in days from AR(1) coefficient."""
    if np.isnan(ar1) or ar1 >= 1.0 or ar1 <= 0.0:
        return float("nan")
    return float(-math.log(2) / math.log(ar1))


def _percentile_rank(value: float, series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float((s < value).sum() / len(s))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PointInTimeFeatureBuilder:
    """
    Computes features at a specific point in time, ensuring no future data is used.

    This is the live/online feature computation path (parallel to DatasetBuilder
    which is for offline training).

    Parameters
    ----------
    feature_store : optional
        An existing models.feature_store.FeatureStore instance.  If supplied,
        its compute_fn callables are available as a fallback for any registered
        feature not implemented natively in this builder.  When None, only the
        natively implemented features are available.
    """

    def __init__(self, feature_store=None) -> None:
        self._feature_store = feature_store
        self._registry: Dict[str, FeatureDefinition] = FEATURE_REGISTRY
        self._groups: Dict[str, FeatureGroup] = FEATURE_GROUPS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_pair_features(
        self,
        pair_id: str,
        px: pd.Series,
        py: pd.Series,
        z: pd.Series,
        as_of: Union[datetime, str],
        feature_names: Optional[List[str]] = None,
        volume_x: Optional[pd.Series] = None,
        volume_y: Optional[pd.Series] = None,
    ) -> FeatureSnapshot:
        """
        Compute pair-level and regime-level features at `as_of`.

        Parameters
        ----------
        pair_id : str
            Identifier for the pair (used as entity_id in snapshot).
        px, py : pd.Series
            Price series for each leg (datetime index).
        z : pd.Series
            Pre-computed z-score series.
        as_of : datetime | str
            All data is clipped to this timestamp before computation.
        feature_names : list[str] | None
            Subset of features to compute.  None = all pair + regime features.
        volume_x, volume_y : pd.Series | None
            Volume series for each leg, used for reg_liquidity_proxy.
        """
        as_of_dt = _to_datetime(as_of)
        warnings: List[str] = []
        values: Dict[str, float] = {}

        # Clip to as_of
        px_c = _clip(px, as_of_dt).dropna()
        py_c = _clip(py, as_of_dt).dropna()
        z_c = _clip(z, as_of_dt).dropna()

        # Determine which pair + regime features to compute
        if feature_names is None:
            target_names = [
                n for n, d in self._registry.items()
                if d.entity_scope in (EntityScope.PAIR, EntityScope.REGIME)
            ]
        else:
            target_names = feature_names

        for name in target_names:
            defn = self._registry.get(name)
            if defn is None:
                warnings.append(f"Feature '{name}' not in registry — skipped")
                continue
            if defn.entity_scope not in (EntityScope.PAIR, EntityScope.REGIME):
                continue
            try:
                val = self._compute_pair_single(
                    name, defn, px_c, py_c, z_c,
                    volume_x=volume_x, volume_y=volume_y,
                    as_of=as_of_dt, warnings=warnings,
                )
                if val is not None:
                    values[name] = float(val)
            except Exception as exc:
                warnings.append(f"Error computing '{name}': {exc}")

        null_count = len(target_names) - len(values)
        return FeatureSnapshot(
            entity_id=pair_id,
            as_of=pd.Timestamp(as_of_dt).isoformat(),
            feature_values=values,
            null_count=max(null_count, 0),
            warnings=warnings,
        )

    def compute_instrument_features(
        self,
        symbol: str,
        prices: pd.Series,
        volume: Optional[pd.Series],
        as_of: Union[datetime, str],
        feature_names: Optional[List[str]] = None,
    ) -> FeatureSnapshot:
        """
        Compute instrument-level features at `as_of`.

        Parameters
        ----------
        symbol : str
            Instrument identifier used as entity_id.
        prices : pd.Series
            Close price series with datetime index.
        volume : pd.Series | None
            Volume series; needed for inst_volume_ratio.
        as_of : datetime | str
            All data clipped to this timestamp.
        feature_names : list[str] | None
            None = all instrument features.
        """
        as_of_dt = _to_datetime(as_of)
        warnings: List[str] = []
        values: Dict[str, float] = {}

        px_c = _clip(prices, as_of_dt).dropna()
        vol_c = _clip(volume, as_of_dt).dropna() if volume is not None else pd.Series(dtype=float)

        if feature_names is None:
            target_names = [
                n for n, d in self._registry.items()
                if d.entity_scope == EntityScope.INSTRUMENT
            ]
        else:
            target_names = feature_names

        for name in target_names:
            defn = self._registry.get(name)
            if defn is None:
                warnings.append(f"Feature '{name}' not in registry — skipped")
                continue
            if defn.entity_scope != EntityScope.INSTRUMENT:
                continue
            try:
                val = self._compute_instrument_single(
                    name, defn, px_c, vol_c, warnings=warnings
                )
                if val is not None:
                    values[name] = float(val)
            except Exception as exc:
                warnings.append(f"Error computing '{name}': {exc}")

        null_count = len(target_names) - len(values)
        return FeatureSnapshot(
            entity_id=symbol,
            as_of=pd.Timestamp(as_of_dt).isoformat(),
            feature_values=values,
            null_count=max(null_count, 0),
            warnings=warnings,
        )

    def compute_signal_features(
        self,
        pair_id: str,
        z: pd.Series,
        entry_z: float,
        entry_timestamp: Union[datetime, str],
        as_of: Union[datetime, str],
        trade_outcomes: Optional[List[float]] = None,
    ) -> FeatureSnapshot:
        """
        Compute signal-context features at `as_of`.

        Parameters
        ----------
        pair_id : str
            Pair identifier.
        z : pd.Series
            Z-score series with datetime index.
        entry_z : float
            The z-score value at the moment entry threshold was breached.
        entry_timestamp : datetime | str
            When the entry threshold was first breached.
        as_of : datetime | str
            Current timestamp — data is clipped here.
        trade_outcomes : list[float] | None
            Recent P&L outcomes for sig_prior_failed_5d (negative = failure).
        """
        as_of_dt = _to_datetime(as_of)
        entry_dt = _to_datetime(entry_timestamp)
        warnings: List[str] = []
        values: Dict[str, float] = {}

        z_c = _clip(z, as_of_dt).dropna()

        target_names = [
            n for n, d in self._registry.items()
            if d.entity_scope == EntityScope.SIGNAL
        ]

        for name in target_names:
            try:
                val = self._compute_signal_single(
                    name, z_c, entry_z, entry_dt, as_of_dt,
                    trade_outcomes=trade_outcomes,
                    warnings=warnings,
                )
                if val is not None:
                    values[name] = float(val)
            except Exception as exc:
                warnings.append(f"Error computing '{name}': {exc}")

        null_count = len(target_names) - len(values)
        return FeatureSnapshot(
            entity_id=pair_id,
            as_of=pd.Timestamp(as_of_dt).isoformat(),
            feature_values=values,
            null_count=max(null_count, 0),
            warnings=warnings,
        )

    def build_feature_vector(
        self,
        snapshots: List[FeatureSnapshot],
        feature_names: List[str],
    ) -> Dict[str, float]:
        """
        Merge multiple FeatureSnapshots into a single flat feature vector.

        Later snapshots in the list take precedence for duplicate feature names.

        Parameters
        ----------
        snapshots : list[FeatureSnapshot]
            Any mix of pair, instrument, signal, portfolio, or execution snapshots.
        feature_names : list[str]
            The ordered feature names to include in the output vector.
            Features missing from all snapshots are omitted from the result.

        Returns
        -------
        dict[str, float]
            Mapping of feature_name → value for all available features.
        """
        merged: Dict[str, float] = {}
        for snap in snapshots:
            merged.update(snap.feature_values)
        return {k: merged[k] for k in feature_names if k in merged}

    def validate_feature_availability(
        self,
        feature_vector: Dict[str, float],
        required_features: List[str],
        missing_threshold: float = 0.20,
    ) -> Tuple[bool, List[str]]:
        """
        Check that sufficient features are available for model inference.

        Parameters
        ----------
        feature_vector : dict[str, float]
            The computed feature vector (from build_feature_vector).
        required_features : list[str]
            The full list of features expected by the model.
        missing_threshold : float
            Maximum acceptable fraction of missing features (default 0.20 = 20%).

        Returns
        -------
        (ok, missing_list)
            ok : bool — True if missing fraction ≤ missing_threshold.
            missing_list : list[str] — names of missing features.
        """
        missing = [f for f in required_features if f not in feature_vector]
        if not required_features:
            return True, []
        fraction_missing = len(missing) / len(required_features)
        ok = fraction_missing <= missing_threshold
        return ok, missing

    # ------------------------------------------------------------------
    # Internal per-feature compute helpers
    # ------------------------------------------------------------------

    def _compute_instrument_single(
        self,
        name: str,
        defn: FeatureDefinition,
        px: pd.Series,
        volume: pd.Series,
        warnings: List[str],
    ) -> Optional[float]:
        """Compute a single instrument-level feature.  Returns None if insufficient data."""
        lb = defn.lookback_days

        if name == "inst_ret_1d":
            if len(px) < 2:
                warnings.append(f"{name}: insufficient data (need 2, have {len(px)})")
                return None
            return float(np.log(px.iloc[-1] / px.iloc[-2]))

        if name == "inst_ret_5d":
            if len(px) < 6:
                warnings.append(f"{name}: insufficient data (need 6, have {len(px)})")
                return None
            return float(np.log(px.iloc[-1] / px.iloc[-6]))

        if name == "inst_ret_20d":
            if len(px) < 21:
                warnings.append(f"{name}: insufficient data (need 21, have {len(px)})")
                return None
            return float(np.log(px.iloc[-1] / px.iloc[-21]))

        if name == "inst_vol_20d":
            if len(px) < 21:
                warnings.append(f"{name}: need 21 bars, have {len(px)}")
                return None
            rets = _log_returns(px.iloc[-21:])
            return float(rets.std() * _ANNUALISE)

        if name == "inst_vol_60d":
            if len(px) < 61:
                warnings.append(f"{name}: need 61 bars, have {len(px)}")
                return None
            rets = _log_returns(px.iloc[-61:])
            return float(rets.std() * _ANNUALISE)

        if name == "inst_vol_ratio":
            if len(px) < 61:
                warnings.append(f"{name}: need 61 bars, have {len(px)}")
                return None
            v20 = _log_returns(px.iloc[-21:]).std() * _ANNUALISE
            v60 = _log_returns(px.iloc[-61:]).std() * _ANNUALISE
            if v60 == 0:
                return float("nan")
            return float(v20 / v60)

        if name == "inst_mom_20d":
            if len(px) < 21:
                warnings.append(f"{name}: need 21 bars, have {len(px)}")
                return None
            ret = float(np.log(px.iloc[-1] / px.iloc[-21]))
            vol = _log_returns(px.iloc[-21:]).std() * _ANNUALISE
            if vol == 0:
                return float("nan")
            return float(ret / vol)

        if name == "inst_mom_60d":
            if len(px) < 61:
                warnings.append(f"{name}: need 61 bars, have {len(px)}")
                return None
            ret = float(np.log(px.iloc[-1] / px.iloc[-61]))
            vol = _log_returns(px.iloc[-61:]).std() * _ANNUALISE
            if vol == 0:
                return float("nan")
            return float(ret / vol)

        if name == "inst_gap_5d":
            if len(px) < 5:
                warnings.append(f"{name}: need 5 bars, have {len(px)}")
                return None
            w = px.iloc[-5:]
            mean_px = float(w.mean())
            if mean_px == 0:
                return float("nan")
            return float((w.max() - w.min()) / mean_px)

        if name == "inst_volume_ratio":
            if volume is None or len(volume) < 20:
                warnings.append(f"{name}: need 20 volume bars")
                return None
            recent = float(volume.iloc[-5:].mean()) if len(volume) >= 5 else float(volume.mean())
            baseline = float(volume.iloc[-20:].mean())
            if baseline == 0:
                return float("nan")
            return float(recent / baseline)

        warnings.append(f"Instrument feature '{name}' has no native implementation")
        return None

    def _compute_pair_single(
        self,
        name: str,
        defn: FeatureDefinition,
        px: pd.Series,
        py: pd.Series,
        z: pd.Series,
        volume_x: Optional[pd.Series],
        volume_y: Optional[pd.Series],
        as_of: datetime,
        warnings: List[str],
    ) -> Optional[float]:
        """Compute a single pair or regime feature.  Returns None if insufficient data."""

        # ── Z-score features ────────────────────────────────────────────

        if name == "pair_z":
            if z.empty:
                return None
            return float(z.iloc[-1])

        if name == "pair_z_abs":
            if z.empty:
                return None
            return abs(float(z.iloc[-1]))

        if name == "pair_z_mean_5d":
            if len(z) < 5:
                warnings.append(f"{name}: need 5 bars, have {len(z)}")
                return None
            return float(z.iloc[-5:].mean())

        if name == "pair_z_mean_20d":
            if len(z) < 20:
                warnings.append(f"{name}: need 20 bars, have {len(z)}")
                return None
            return float(z.iloc[-20:].mean())

        if name == "pair_z_std_20d":
            if len(z) < 20:
                warnings.append(f"{name}: need 20 bars, have {len(z)}")
                return None
            return float(z.iloc[-20:].std())

        if name == "pair_z_mom_5d":
            if len(z) < 6:
                warnings.append(f"{name}: need 6 bars, have {len(z)}")
                return None
            return float(z.iloc[-1] - z.iloc[-6])

        if name == "pair_z_mom_20d":
            if len(z) < 21:
                warnings.append(f"{name}: need 21 bars, have {len(z)}")
                return None
            return float(z.iloc[-1] - z.iloc[-21])

        if name == "pair_z_ar1":
            if len(z) < 22:
                warnings.append(f"{name}: need 22 bars, have {len(z)}")
                return None
            return _ar1_coef(z.iloc[-22:])

        if name == "pair_z_cross_zero_20d":
            if len(z) < 20:
                warnings.append(f"{name}: need 20 bars, have {len(z)}")
                return None
            w = z.iloc[-20:].values
            crossings = int(np.sum(np.diff(np.sign(w)) != 0))
            return float(crossings / 20.0)

        if name == "pair_bb_pos_20d":
            if len(z) < 20:
                warnings.append(f"{name}: need 20 bars, have {len(z)}")
                return None
            w = z.iloc[-20:]
            std = float(w.std())
            if std == 0:
                return float("nan")
            return float((z.iloc[-1] - w.mean()) / std)

        if name == "pair_bb_pos_60d":
            if len(z) < 60:
                warnings.append(f"{name}: need 60 bars, have {len(z)}")
                return None
            w = z.iloc[-60:]
            std = float(w.std())
            if std == 0:
                return float("nan")
            return float((z.iloc[-1] - w.mean()) / std)

        if name == "pair_div_speed":
            if len(z) < 4:
                warnings.append(f"{name}: need 4 bars, have {len(z)}")
                return None
            return float(np.diff(z.iloc[-4:].values).mean())

        if name == "pair_entry_attempts_5d":
            if len(z) < 6:
                warnings.append(f"{name}: need 6 bars, have {len(z)}")
                return None
            abs_z = z.iloc[-6:].abs().values
            # Count upward crossings of _ENTRY_Z_THRESHOLD
            crossings = int(np.sum(
                (abs_z[1:] >= _ENTRY_Z_THRESHOLD) & (abs_z[:-1] < _ENTRY_Z_THRESHOLD)
            ))
            return float(crossings)

        if name == "pair_residual_skew":
            if len(z) < 20:
                warnings.append(f"{name}: need 20 bars, have {len(z)}")
                return None
            from scipy.stats import skew
            return float(skew(z.iloc[-20:].values))

        if name == "pair_residual_kurt":
            if len(z) < 20:
                warnings.append(f"{name}: need 20 bars, have {len(z)}")
                return None
            from scipy.stats import kurtosis
            return float(kurtosis(z.iloc[-20:].values))

        if name == "pair_rel_momentum":
            if len(z) < 21:
                warnings.append(f"{name}: need 21 bars, have {len(z)}")
                return None
            dz = float(z.iloc[-1] - z.iloc[-21])
            spread_vol = _log_returns(pd.Series(z.iloc[-21:].values)).std()
            if spread_vol == 0 or np.isnan(spread_vol):
                return float("nan")
            return float(dz / spread_vol)

        # ── Correlation features ─────────────────────────────────────────

        if name in ("pair_corr_20d", "pair_corr_60d", "pair_corr_trend"):
            window_20 = 21
            window_60 = 61
            if name == "pair_corr_20d":
                if len(px) < window_20 or len(py) < window_20:
                    warnings.append(f"{name}: need {window_20} bars")
                    return None
                ret_x = _log_returns(px.iloc[-window_20:])
                ret_y = _log_returns(py.iloc[-window_20:])
                aligned = ret_x.align(ret_y, join="inner")[0].dropna()
                if len(aligned) < 5:
                    return float("nan")
                rx = _log_returns(px.iloc[-window_20:]).reindex(aligned.index)
                ry = _log_returns(py.iloc[-window_20:]).reindex(aligned.index)
                return float(rx.corr(ry))
            if name == "pair_corr_60d":
                if len(px) < window_60 or len(py) < window_60:
                    warnings.append(f"{name}: need {window_60} bars")
                    return None
                rx = _log_returns(px.iloc[-window_60:])
                ry = _log_returns(py.iloc[-window_60:])
                return float(rx.corr(ry))
            if name == "pair_corr_trend":
                if len(px) < window_60 or len(py) < window_60:
                    warnings.append(f"{name}: need {window_60} bars")
                    return None
                rx = _log_returns(px)
                ry = _log_returns(py)
                c20 = rx.iloc[-20:].corr(ry.iloc[-20:]) if len(rx) >= 20 else float("nan")
                c60 = rx.iloc[-60:].corr(ry.iloc[-60:]) if len(rx) >= 60 else float("nan")
                if np.isnan(c20) or np.isnan(c60):
                    return float("nan")
                return float(c20 - c60)

        # ── Stability / hedge ratio features ─────────────────────────────

        if name == "pair_hedge_ratio":
            if len(px) < 60 or len(py) < 60:
                warnings.append(f"{name}: need 60 bars")
                return None
            px_w = px.iloc[-60:].values.astype(float)
            py_w = py.iloc[-60:].values.astype(float)
            X = np.column_stack([np.ones(len(px_w)), px_w])
            try:
                beta = np.linalg.lstsq(X, py_w, rcond=None)[0][1]
            except Exception:
                return float("nan")
            return float(beta)

        if name == "pair_hr_stability":
            # Compute rolling 20d hedge ratios across the 60d window and measure CV
            if len(px) < 60 or len(py) < 60:
                warnings.append(f"{name}: need 60 bars")
                return None
            hrlist: List[float] = []
            n = min(len(px), len(py), 60)
            px_w = px.iloc[-n:].values.astype(float)
            py_w = py.iloc[-n:].values.astype(float)
            step = 5
            w = 20
            for i in range(0, n - w + 1, step):
                xi = px_w[i: i + w]
                yi = py_w[i: i + w]
                Xm = np.column_stack([np.ones(w), xi])
                try:
                    b = np.linalg.lstsq(Xm, yi, rcond=None)[0][1]
                    hrlist.append(b)
                except Exception:
                    pass
            if len(hrlist) < 2:
                return float("nan")
            hr_arr = np.array(hrlist)
            mean_hr = abs(float(np.mean(hr_arr)))
            if mean_hr == 0:
                return float("nan")
            return float(np.std(hr_arr) / mean_hr)

        if name == "pair_half_life":
            if len(z) < 60:
                warnings.append(f"{name}: need 60 bars, have {len(z)}")
                return None
            ar1 = _ar1_coef(z.iloc[-60:])
            return _half_life_from_ar1(ar1)

        if name == "pair_half_life_ratio":
            if len(z) < 60:
                warnings.append(f"{name}: need 60 bars, have {len(z)}")
                return None
            ar1 = _ar1_coef(z.iloc[-60:])
            hl = _half_life_from_ar1(ar1)
            if np.isnan(hl):
                return float("nan")
            return float(hl / 30.0)

        # ── Spread vol features ───────────────────────────────────────────

        if name == "pair_spread_vol_20d":
            if len(z) < 21:
                warnings.append(f"{name}: need 21 bars, have {len(z)}")
                return None
            return float(np.diff(z.iloc[-21:].values).std())

        if name == "pair_spread_vol_ratio":
            if len(z) < 61:
                warnings.append(f"{name}: need 61 bars, have {len(z)}")
                return None
            v20 = float(np.diff(z.iloc[-21:].values).std())
            v60 = float(np.diff(z.iloc[-61:].values).std())
            if v60 == 0:
                return float("nan")
            return float(v20 / v60)

        # ── Regime features ───────────────────────────────────────────────

        if name == "reg_vol_regime":
            w20, w120 = 21, 121
            need = max(w20, w120)
            if len(px) < need or len(py) < need:
                warnings.append(f"{name}: need {need} bars")
                return None
            v20_x = _log_returns(px.iloc[-w20:]).std() * _ANNUALISE
            v20_y = _log_returns(py.iloc[-w20:]).std() * _ANNUALISE
            v120_x = _log_returns(px.iloc[-w120:]).std() * _ANNUALISE
            v120_y = _log_returns(py.iloc[-w120:]).std() * _ANNUALISE
            recent = (v20_x + v20_y) / 2.0
            base = (v120_x + v120_y) / 2.0
            if base == 0:
                return float("nan")
            return float(recent / base)

        if name == "reg_vol_of_vol":
            if len(px) < 60 or len(py) < 60:
                warnings.append(f"{name}: need 60 bars")
                return None
            # Rolling 20d vol of x over last 60 bars
            n = 60
            vols = []
            for i in range(n - 20):
                v = _log_returns(px.iloc[i: i + 21]).std() * _ANNUALISE
                vols.append(v)
            return float(np.std(vols)) if vols else float("nan")

        if name == "reg_trend_slope":
            if len(z) < 21:
                warnings.append(f"{name}: need 21 bars, have {len(z)}")
                return None
            w = z.iloc[-20:].values.astype(float)
            std_z = float(np.std(w))
            if std_z == 0:
                return 0.0
            slope = _ols_slope(w)
            return float(slope / std_z)

        if name == "reg_mean_reversion_quality":
            if len(z) < 61:
                warnings.append(f"{name}: need 61 bars, have {len(z)}")
                return None
            return _ar1_coef(z.iloc[-61:])

        if name == "reg_spread_persistence":
            if len(z) < 62:
                warnings.append(f"{name}: need 62 bars, have {len(z)}")
                return None
            w = z.iloc[-60:].values.astype(float)
            s = pd.Series(w)
            return float(s.corr(s.shift(1)))

        if name == "reg_break_indicator":
            if len(z) < 60:
                warnings.append(f"{name}: need 60 bars, have {len(z)}")
                return None
            w = z.iloc[-60:].values.astype(float)
            mean_w = w.mean()
            demeaned = w - mean_w
            cusum = np.cumsum(demeaned)
            std_w = w.std()
            if std_w == 0:
                return 0.0
            # Normalised CUSUM range
            cusum_range = float((cusum.max() - cusum.min()) / (std_w * math.sqrt(len(w))))
            return cusum_range

        if name == "reg_market_stress":
            w20 = 21
            if len(px) < w20 or len(py) < w20:
                warnings.append(f"{name}: need {w20} bars")
                return None
            v_x = _log_returns(px.iloc[-w20:]).std() * _ANNUALISE
            v_y = _log_returns(py.iloc[-w20:]).std() * _ANNUALISE
            return float(abs(v_x - v_y))

        if name == "reg_liquidity_proxy":
            if volume_x is None or volume_y is None:
                warnings.append(f"{name}: volume_x/volume_y required")
                return None
            vx = _clip(volume_x, as_of).dropna()
            vy = _clip(volume_y, as_of).dropna()
            if len(vx) < 5 or len(vy) < 5:
                warnings.append(f"{name}: need 5 volume bars")
                return None
            recent = float(vx.iloc[-5:].mean() + vy.iloc[-5:].mean())
            base_x = float(vx.iloc[-60:].mean()) if len(vx) >= 60 else float(vx.mean())
            base_y = float(vy.iloc[-60:].mean()) if len(vy) >= 60 else float(vy.mean())
            base = base_x + base_y
            if base == 0:
                return float("nan")
            return float(recent / base)

        warnings.append(f"Pair/regime feature '{name}' has no native implementation")
        return None

    def _compute_signal_single(
        self,
        name: str,
        z: pd.Series,
        entry_z: float,
        entry_dt: datetime,
        as_of: datetime,
        trade_outcomes: Optional[List[float]],
        warnings: List[str],
    ) -> Optional[float]:
        """Compute a single signal-context feature."""

        if name == "sig_z_at_entry":
            return float(entry_z)

        if name == "sig_time_since_breach":
            delta = pd.Timestamp(as_of) - pd.Timestamp(entry_dt)
            return float(delta.days)

        if name == "sig_time_since_prior_cross":
            if len(z) < 2:
                warnings.append(f"{name}: need at least 2 z bars")
                return None
            # Find last zero crossing before as_of
            vals = z.values.astype(float)
            signs = np.sign(vals)
            changes = np.where(np.diff(signs) != 0)[0]
            if len(changes) == 0:
                # No crossing found in window; return full window length
                return float(len(z))
            last_cross_idx = int(changes[-1])
            last_cross_ts = z.index[last_cross_idx]
            if hasattr(last_cross_ts, "tzinfo") and last_cross_ts.tzinfo is None:
                last_cross_ts = last_cross_ts.tz_localize("UTC")
            as_of_ts = pd.Timestamp(as_of)
            if as_of_ts.tzinfo is None:
                as_of_ts = as_of_ts.tz_localize("UTC")
            delta = as_of_ts - last_cross_ts
            return float(max(delta.days, 0))

        if name == "sig_confirmation_bars":
            if z.empty:
                return 0.0
            entry_ts = pd.Timestamp(entry_dt)
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            z_since = z[z.index >= entry_ts]
            if z_since.empty:
                return 0.0
            # Count consecutive bars from entry where |z| >= threshold
            above = (z_since.abs() >= _ENTRY_Z_THRESHOLD).values
            count = 0
            for v in above:
                if v:
                    count += 1
                else:
                    break
            return float(count)

        if name == "sig_prior_failed_5d":
            if trade_outcomes is None or len(trade_outcomes) == 0:
                return 0.0
            return float(1 if any(o < 0 for o in trade_outcomes[-5:]) else 0)

        if name == "sig_entry_z_percentile":
            if len(z) < 60:
                warnings.append(f"{name}: need 60 bars for percentile")
                return None
            return _percentile_rank(abs(entry_z), z.iloc[-60:].abs())

        if name == "sig_spread_accel":
            # Second difference of z near entry
            entry_ts = pd.Timestamp(entry_dt)
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            # Use last 5 bars before / at entry
            z_window = z[z.index <= entry_ts].iloc[-5:]
            if len(z_window) < 3:
                warnings.append(f"{name}: need 3 bars around entry")
                return None
            d2 = float(np.diff(np.diff(z_window.values)).mean())
            return d2

        warnings.append(f"Signal feature '{name}' has no native implementation")
        return None
