# -*- coding: utf-8 -*-
"""
ml/labels/builder.py — Label Builder
======================================

Generates labels from price/spread data for ML training.

CRITICAL: Label generation uses FUTURE data by design — it is for offline
training pipelines ONLY. The train_end boundary enforced by DatasetBuilder
ensures that features use only past data; labels deliberately look forward.

Never import or call LabelBuilder from live signal generation code.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ml.labels.builder")


class LabelBuilder:
    """
    Generates labels from price/spread data for ML training.

    CRITICAL: Label generation uses FUTURE data by design (offline training only).
    The train_end boundary enforced by DatasetBuilder ensures features use only
    past data. Labels deliberately look forward — this is intentional and correct.

    All methods return pd.Series with the same index as the input, with NaN for
    rows that have insufficient future data to compute the label.
    """

    # ------------------------------------------------------------------
    # A. Reversion success labels
    # ------------------------------------------------------------------

    def build_reversion_label(
        self,
        z: pd.Series,
        horizon: int,
        entry_threshold: float = 2.0,
        target_fraction: float = 0.5,
    ) -> pd.Series:
        """
        Binary label: did z-score revert to target_fraction * entry_z within horizon days?

        Returns 1 if successful reversion, 0 if not, NaN if insufficient future data.

        LEAKAGE NOTE: This method accesses future values of z.
        Only call from offline label-generation pipelines. Never use in features.
        """
        results = pd.Series(np.nan, index=z.index, dtype=float)
        z_arr = z.values
        n = len(z_arr)

        for i in range(n - 1):
            if abs(z_arr[i]) < entry_threshold or np.isnan(z_arr[i]):
                continue
            end = min(i + horizon, n)
            if end - i < horizon:
                # Insufficient future data — leave as NaN
                continue
            entry_sign = np.sign(z_arr[i])
            target = target_fraction * abs(z_arr[i])
            # Check if forward window crosses from current side toward zero
            future_z = z_arr[i + 1: end + 1]
            # Reversion: opposite sign or abs(z) <= target
            crossed = any(
                (entry_sign * fz <= target) for fz in future_z if not np.isnan(fz)
            )
            results.iloc[i] = 1.0 if crossed else 0.0

        return results

    def build_profit_target_label(
        self,
        z: pd.Series,
        horizon: int,
        entry_threshold: float = 2.0,
        target_fraction: float = 0.80,
    ) -> pd.Series:
        """
        Binary label: did z-score hit target_fraction * entry_z within horizon days?
        Stricter than reversion — 80% of the way to zero counts as hitting target.

        LEAKAGE NOTE: Accesses future z values. Offline training only.
        """
        return self.build_reversion_label(
            z=z,
            horizon=horizon,
            entry_threshold=entry_threshold,
            target_fraction=1.0 - target_fraction,
        )

    def build_profitable_net_label(
        self,
        z: pd.Series,
        horizon: int,
        entry_threshold: float = 2.0,
        cost_bps: float = 30.0,
        spread_vol: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Binary label: was net P&L > 0 after transaction costs within horizon days?

        P&L proxy = z[t] - z[t+horizon] (for short-spread trades), adjusted for cost_bps.
        spread_vol scales the bps cost into z-score units if provided.

        LEAKAGE NOTE: Accesses z[t+horizon]. Offline training only.
        """
        results = pd.Series(np.nan, index=z.index, dtype=float)
        z_arr = z.values
        n = len(z_arr)

        cost_z = cost_bps / 10_000.0  # crude cost in z-score units if no vol provided

        for i in range(n):
            if abs(z_arr[i]) < entry_threshold or np.isnan(z_arr[i]):
                continue
            end = i + horizon
            if end >= n:
                continue
            if np.isnan(z_arr[end]):
                continue

            entry_sign = np.sign(z_arr[i])
            gross_pnl_z = entry_sign * (z_arr[i] - z_arr[end])
            cost = cost_z
            if spread_vol is not None and not np.isnan(spread_vol.iloc[i]):
                cost = cost_bps / 10_000.0 / spread_vol.iloc[i]
            results.iloc[i] = 1.0 if gross_pnl_z > cost else 0.0

        return results

    def build_path_quality_label(
        self,
        z: pd.Series,
        horizon: int,
        entry_threshold: float = 2.0,
        mae_sigma_limit: float = 1.5,
    ) -> pd.Series:
        """
        Binary label: was max adverse excursion (MAE) < mae_sigma_limit * sigma within horizon?

        MAE measured in absolute z-score units from entry level.

        LEAKAGE NOTE: Accesses entire forward path up to horizon. Offline training only.
        """
        results = pd.Series(np.nan, index=z.index, dtype=float)
        z_arr = z.values
        n = len(z_arr)

        for i in range(n):
            if abs(z_arr[i]) < entry_threshold or np.isnan(z_arr[i]):
                continue
            end = min(i + horizon, n)
            if end - i < horizon:
                continue
            future_z = z_arr[i + 1: end + 1]
            entry_sign = np.sign(z_arr[i])
            # Adverse direction = same sign as entry (spread going further from zero)
            adverse = [fz for fz in future_z if not np.isnan(fz) and entry_sign * fz > entry_sign * z_arr[i]]
            max_adverse = max((entry_sign * fz - entry_sign * z_arr[i] for fz in adverse), default=0.0)
            results.iloc[i] = 1.0 if max_adverse < mae_sigma_limit else 0.0

        return results

    # ------------------------------------------------------------------
    # B. Meta-labeling
    # ------------------------------------------------------------------

    def build_meta_take_label(
        self,
        z: pd.Series,
        horizon: int,
        entry_threshold: float = 2.0,
        cost_bps: float = 30.0,
    ) -> pd.Series:
        """
        Binary label for meta-labeling: 1 = take the signal, 0 = skip.
        Based on whether the trade was net profitable after costs.

        LEAKAGE NOTE: Accesses z[t+horizon]. Offline training only.
        """
        return self.build_profitable_net_label(
            z=z,
            horizon=horizon,
            entry_threshold=entry_threshold,
            cost_bps=cost_bps,
        )

    def build_meta_delay_label(
        self,
        z: pd.Series,
        horizon: int = 10,
        delay_bars: int = 2,
        entry_threshold: float = 2.0,
        cost_bps: float = 30.0,
    ) -> pd.Series:
        """
        Binary: was delayed entry (by delay_bars) more profitable than immediate entry?

        LEAKAGE NOTE: Compares t vs t+delay outcomes. Offline training only.
        """
        results = pd.Series(np.nan, index=z.index, dtype=float)
        z_arr = z.values
        n = len(z_arr)
        cost_z = cost_bps / 10_000.0

        for i in range(n):
            if abs(z_arr[i]) < entry_threshold or np.isnan(z_arr[i]):
                continue
            i_delayed = i + delay_bars
            end_immediate = i + horizon
            end_delayed = i_delayed + horizon
            if end_delayed >= n:
                continue

            entry_sign = np.sign(z_arr[i])
            pnl_immediate = entry_sign * (z_arr[i] - z_arr[end_immediate]) - cost_z
            pnl_delayed = entry_sign * (z_arr[i_delayed] - z_arr[end_delayed]) - cost_z
            results.iloc[i] = 1.0 if pnl_delayed > pnl_immediate else 0.0

        return results

    def build_meta_size_full_label(
        self,
        z: pd.Series,
        horizon: int = 10,
        entry_threshold: float = 2.0,
        cost_bps: float = 30.0,
        mae_sigma_limit: float = 1.5,
    ) -> pd.Series:
        """
        Binary: was position worth full size? (profitable_net AND path_quality)

        LEAKAGE NOTE: Combines two forward-looking labels. Offline training only.
        """
        profitable = self.build_profitable_net_label(z, horizon, entry_threshold, cost_bps)
        path_ok = self.build_path_quality_label(z, horizon, entry_threshold, mae_sigma_limit)
        result = profitable.where(profitable.notna() & path_ok.notna())
        result = (profitable == 1.0) & (path_ok == 1.0)
        return result.astype(float).where(profitable.notna() & path_ok.notna())

    # ------------------------------------------------------------------
    # C. Relationship persistence
    # ------------------------------------------------------------------

    def build_persistence_label(
        self,
        adf_pvalues: pd.Series,
        horizon: int,
        max_pvalue: float = 0.10,
    ) -> pd.Series:
        """
        Binary: did ADF p-value remain < max_pvalue over the forward horizon?

        LEAKAGE NOTE: Accesses adf_pvalues[t+1..t+horizon]. Offline training only.
        """
        results = pd.Series(np.nan, index=adf_pvalues.index, dtype=float)
        arr = adf_pvalues.values
        n = len(arr)

        for i in range(n):
            end = min(i + horizon, n)
            if end - i < horizon:
                continue
            future = arr[i + 1: end + 1]
            valid = future[~np.isnan(future)]
            if len(valid) == 0:
                continue
            results.iloc[i] = 1.0 if np.all(valid < max_pvalue) else 0.0

        return results

    def build_half_life_stable_label(
        self,
        half_life: pd.Series,
        horizon: int = 30,
        change_multiple: float = 2.0,
    ) -> pd.Series:
        """
        Binary: did half-life stay within change_multiple * current value for horizon days?

        LEAKAGE NOTE: Accesses future half_life values. Offline training only.
        """
        results = pd.Series(np.nan, index=half_life.index, dtype=float)
        arr = half_life.values
        n = len(arr)

        for i in range(n):
            if np.isnan(arr[i]) or arr[i] <= 0:
                continue
            end = min(i + horizon, n)
            if end - i < horizon:
                continue
            future = arr[i + 1: end + 1]
            valid = future[~np.isnan(future)]
            if len(valid) == 0:
                continue
            max_ratio = np.max(np.abs(valid) / arr[i])
            results.iloc[i] = 1.0 if max_ratio <= change_multiple else 0.0

        return results

    def build_coint_strength_label(
        self,
        adf_pvalues: pd.Series,
        horizon: int = 30,
    ) -> pd.Series:
        """
        Continuous: mean cointegration strength (1 - p-value) over next horizon days.

        LEAKAGE NOTE: Accesses forward adf_pvalues. Offline training only.
        """
        results = pd.Series(np.nan, index=adf_pvalues.index, dtype=float)
        arr = adf_pvalues.values
        n = len(arr)

        for i in range(n):
            end = min(i + horizon, n)
            if end - i < horizon:
                continue
            future = arr[i + 1: end + 1]
            valid = future[~np.isnan(future)]
            if len(valid) == 0:
                continue
            results.iloc[i] = float(np.mean(1.0 - valid))

        return results

    # ------------------------------------------------------------------
    # D. Break / instability
    # ------------------------------------------------------------------

    def build_break_label(
        self,
        spread: pd.Series,
        horizon: int,
        cusum_threshold: float = 3.0,
    ) -> pd.Series:
        """
        Binary: did a structural break occur within horizon days?
        Uses simple CUSUM statistic to detect level/variance shift.

        LEAKAGE NOTE: Scans forward window of spread. Offline training only.
        """
        results = pd.Series(np.nan, index=spread.index, dtype=float)
        arr = spread.values
        n = len(arr)

        for i in range(n):
            end = min(i + horizon, n)
            if end - i < horizon:
                continue
            future = arr[i + 1: end + 1]
            valid = future[~np.isnan(future)]
            if len(valid) < 3:
                continue
            # Simple CUSUM: cumulative sum of deviations from initial mean
            base_mean = arr[i] if not np.isnan(arr[i]) else np.nanmean(arr[max(0, i - 20): i + 1])
            deviations = valid - base_mean
            cusum = np.cumsum(deviations - np.mean(deviations))
            # Normalize by std dev of the window
            std = np.std(deviations) if np.std(deviations) > 1e-10 else 1.0
            cusum_stat = np.max(np.abs(cusum)) / (std * np.sqrt(len(valid)))
            results.iloc[i] = 1.0 if cusum_stat > cusum_threshold else 0.0

        return results

    def build_hr_instability_label(
        self,
        hedge_ratio: pd.Series,
        horizon: int,
        change_threshold: float = 0.30,
    ) -> pd.Series:
        """
        Binary: did hedge ratio change by > change_threshold fraction within horizon days?

        LEAKAGE NOTE: Accesses future hedge_ratio values. Offline training only.
        """
        results = pd.Series(np.nan, index=hedge_ratio.index, dtype=float)
        arr = hedge_ratio.values
        n = len(arr)

        for i in range(n):
            if np.isnan(arr[i]) or abs(arr[i]) < 1e-10:
                continue
            end = min(i + horizon, n)
            if end - i < horizon:
                continue
            future = arr[i + 1: end + 1]
            valid = future[~np.isnan(future)]
            if len(valid) == 0:
                continue
            max_change = np.max(np.abs(valid - arr[i])) / abs(arr[i])
            results.iloc[i] = 1.0 if max_change > change_threshold else 0.0

        return results

    def build_residual_var_spike_label(
        self,
        spread: pd.Series,
        horizon: int = 10,
        var_multiple: float = 2.0,
        lookback: int = 20,
    ) -> pd.Series:
        """
        Binary: did residual variance increase > var_multiple within horizon days?

        LEAKAGE NOTE: Compares forward variance to backward variance. Offline training only.
        """
        results = pd.Series(np.nan, index=spread.index, dtype=float)
        arr = spread.values
        n = len(arr)

        for i in range(lookback, n):
            end = min(i + horizon, n)
            if end - i < horizon:
                continue
            base_var = np.var(arr[i - lookback: i])
            if base_var < 1e-12:
                continue
            future = arr[i + 1: end + 1]
            valid = future[~np.isnan(future)]
            if len(valid) < 3:
                continue
            fwd_var = np.var(valid)
            results.iloc[i] = 1.0 if fwd_var / base_var > var_multiple else 0.0

        return results

    # ------------------------------------------------------------------
    # E. Holding-time labels
    # ------------------------------------------------------------------

    def build_holding_time_label(
        self,
        z: pd.Series,
        entry_threshold: float = 2.0,
        target_z: float = 0.5,
        max_horizon: int = 60,
    ) -> pd.Series:
        """
        Continuous: days until z crosses target_z after entry.
        Censored at max_horizon (returns max_horizon if no crossing).

        LEAKAGE NOTE: Scans up to max_horizon days forward. Offline training only.
        """
        results = pd.Series(np.nan, index=z.index, dtype=float)
        z_arr = z.values
        n = len(z_arr)

        for i in range(n):
            if abs(z_arr[i]) < entry_threshold or np.isnan(z_arr[i]):
                continue
            entry_sign = np.sign(z_arr[i])
            crossed = max_horizon  # censored default
            for k in range(1, max_horizon + 1):
                if i + k >= n:
                    break
                fz = z_arr[i + k]
                if np.isnan(fz):
                    continue
                if entry_sign * fz <= target_z:
                    crossed = k
                    break
            results.iloc[i] = float(crossed)

        return results

    def build_time_to_stop_label(
        self,
        z: pd.Series,
        entry_threshold: float = 2.0,
        stop_z: float = 3.5,
        max_horizon: int = 60,
    ) -> pd.Series:
        """
        Continuous: days until z hits stop_z level after entry.
        Censored at max_horizon if stop not hit.

        LEAKAGE NOTE: Scans up to max_horizon forward. Offline training only.
        """
        results = pd.Series(np.nan, index=z.index, dtype=float)
        z_arr = z.values
        n = len(z_arr)

        for i in range(n):
            if abs(z_arr[i]) < entry_threshold or np.isnan(z_arr[i]):
                continue
            entry_sign = np.sign(z_arr[i])
            hit = max_horizon
            for k in range(1, max_horizon + 1):
                if i + k >= n:
                    break
                fz = z_arr[i + k]
                if np.isnan(fz):
                    continue
                if entry_sign * fz >= stop_z:
                    hit = k
                    break
            results.iloc[i] = float(hit)

        return results

    # ------------------------------------------------------------------
    # F. Threshold labels
    # ------------------------------------------------------------------

    def build_optimal_entry_z_bucket_label(
        self,
        z: pd.Series,
        horizon: int = 20,
        cost_bps: float = 30.0,
        z_bands: tuple = (2.0, 2.5, 3.0),
    ) -> pd.Series:
        """
        Ordinal: which entry z-band performed best?
        0 = 2.0-2.5, 1 = 2.5-3.0, 2 = 3.0+

        LEAKAGE NOTE: Forward P&L comparison across z-bands. Offline training only.
        """
        results = pd.Series(np.nan, index=z.index, dtype=float)
        z_arr = z.values
        n = len(z_arr)
        cost_z = cost_bps / 10_000.0

        for i in range(n):
            abs_z = abs(z_arr[i])
            if np.isnan(z_arr[i]) or abs_z < z_bands[0]:
                continue
            end = i + horizon
            if end >= n or np.isnan(z_arr[end]):
                continue
            entry_sign = np.sign(z_arr[i])
            pnl = entry_sign * (z_arr[i] - z_arr[end]) - cost_z
            if pnl <= 0:
                continue
            # Assign bucket based on entry z magnitude
            if abs_z < z_bands[1]:
                results.iloc[i] = 0.0
            elif abs_z < z_bands[2]:
                results.iloc[i] = 1.0
            else:
                results.iloc[i] = 2.0

        return results

    # ------------------------------------------------------------------
    # Master builder
    # ------------------------------------------------------------------

    def build_all_labels(
        self,
        z: pd.Series,
        spread: Optional[pd.Series] = None,
        hedge_ratio: Optional[pd.Series] = None,
        adf_pvalues: Optional[pd.Series] = None,
        train_end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Build all available labels and return as DataFrame.

        train_end: ISO date string — clip label computation to this date.
        The clipping prevents the label series from including dates beyond
        which features are available, maintaining the train/test boundary.

        LEAKAGE NOTE: This method intentionally accesses future data.
        Only call from offline label-generation pipelines.

        Returns: DataFrame with label names as columns, DatetimeIndex.
        """
        if train_end is not None:
            cutoff = pd.Timestamp(train_end)
            z = z[z.index <= cutoff]
            if spread is not None:
                spread = spread[spread.index <= cutoff]
            if hedge_ratio is not None:
                hedge_ratio = hedge_ratio[hedge_ratio.index <= cutoff]
            if adf_pvalues is not None:
                adf_pvalues = adf_pvalues[adf_pvalues.index <= cutoff]

        labels: dict[str, pd.Series] = {}

        # A. Reversion success
        labels["reversion_5d"] = self.build_reversion_label(z, 5)
        labels["reversion_10d"] = self.build_reversion_label(z, 10)
        labels["reversion_20d"] = self.build_reversion_label(z, 20)
        labels["profit_target_5d"] = self.build_profit_target_label(z, 5)
        labels["profitable_net_10d"] = self.build_profitable_net_label(z, 10)
        labels["path_quality_10d"] = self.build_path_quality_label(z, 10)

        # B. Meta-labeling
        labels["meta_take_5d"] = self.build_meta_take_label(z, 5)
        labels["meta_take_10d"] = self.build_meta_take_label(z, 10)
        labels["meta_size_full_10d"] = self.build_meta_size_full_label(z, 10)
        labels["meta_delay_recommended"] = self.build_meta_delay_label(z, 10)

        # C. Relationship persistence (requires adf_pvalues)
        if adf_pvalues is not None:
            labels["persistence_30d"] = self.build_persistence_label(adf_pvalues, 30)
            labels["persistence_60d"] = self.build_persistence_label(adf_pvalues, 60)
            labels["coint_strength_30d"] = self.build_coint_strength_label(adf_pvalues, 30)
        else:
            logger.debug("adf_pvalues not provided; skipping persistence labels")

        # D. Break / instability (requires spread and/or hedge_ratio)
        if spread is not None:
            labels["break_5d"] = self.build_break_label(spread, 5)
            labels["break_20d"] = self.build_break_label(spread, 20)
            labels["residual_var_spike_10d"] = self.build_residual_var_spike_label(spread, 10)

        if hedge_ratio is not None:
            labels["hr_instability_10d"] = self.build_hr_instability_label(hedge_ratio, 10)
            if spread is not None:
                hl = self._estimate_half_life(spread)
                labels["half_life_stable_30d"] = self.build_half_life_stable_label(hl, 30)

        # E. Holding-time
        labels["time_to_mean_reversion"] = self.build_holding_time_label(z)
        labels["time_to_stop"] = self.build_time_to_stop_label(z)

        # F. Threshold
        labels["optimal_entry_z_bucket"] = self.build_optimal_entry_z_bucket_label(z)

        return pd.DataFrame(labels, index=z.index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_half_life(spread: pd.Series, window: int = 30) -> pd.Series:
        """
        Estimate rolling half-life from an OU process fitted to spread.
        Uses the lag-1 autocorrelation proxy: hl = -ln(2) / ln(rho).
        """
        hl = pd.Series(np.nan, index=spread.index)
        arr = spread.values
        n = len(arr)
        for i in range(window, n):
            seg = arr[i - window: i]
            seg = seg[~np.isnan(seg)]
            if len(seg) < 10:
                continue
            rho = np.corrcoef(seg[:-1], seg[1:])[0, 1]
            if rho <= 0 or rho >= 1:
                continue
            hl.iloc[i] = -np.log(2) / np.log(rho)
        return hl
