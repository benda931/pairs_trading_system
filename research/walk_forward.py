# -*- coding: utf-8 -*-
"""
research/walk_forward.py — Walk-Forward Research Harness
=========================================================

Implements a rigorous walk-forward validation framework for pairs:
  1. Purged time-series splits with configurable embargo gaps
  2. Per-fold pair validation + spread fitting (training data only)
  3. Out-of-sample z-score generation on test window
  4. Simple mean-reversion backtest on the test z-series
  5. Aggregation of fold results into ExperimentResult

Design principles:
  - Zero look-ahead: all parameters estimated strictly from training data
  - Explicit embargo: N-day gap between train end and test start
  - Each fold is independent — results aggregated after all folds complete
  - Returns rich diagnostics for regime analysis and hyper-parameter search

Typical usage:
    harness = WalkForwardHarness(n_splits=5, test_days=252, embargo_days=20)
    result = harness.run(pair_id, prices, model=SpreadModel.KALMAN)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from core.contracts import (
    PairId,
    PairLifecycleState,
    PairValidationReport,
    SpreadDefinition,
    SpreadModel,
    ValidationResult,
)
from research.pair_validator import PairValidator
from research.spread_constructor import build_spread

logger = logging.getLogger("research.walk_forward")


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Results for a single walk-forward fold."""
    fold_idx: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Validation
    validation: PairValidationReport

    # Spread
    spread_defn: Optional[SpreadDefinition] = None
    train_z: Optional[pd.Series] = None
    test_z: Optional[pd.Series] = None

    # Backtest metrics (on test window only)
    n_trades: int = 0
    pnl: float = 0.0
    sharpe: float = np.nan
    max_drawdown: float = np.nan
    win_rate: float = np.nan
    avg_holding_days: float = np.nan

    # Spread quality
    spread_mean: float = np.nan
    spread_std: float = np.nan
    spread_autocorr: float = np.nan     # AR(1) — low = fast mean reversion
    test_adf_pvalue: float = np.nan    # ADF on test spread

    # Status
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "fold_idx": self.fold_idx,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "validation_result": self.validation.result.value if self.validation else None,
            "n_trades": self.n_trades,
            "pnl": self.pnl,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "avg_holding_days": self.avg_holding_days,
            "spread_autocorr": self.spread_autocorr,
            "test_adf_pvalue": self.test_adf_pvalue,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class ExperimentResult:
    """Aggregated walk-forward experiment results for a pair."""
    pair_id: PairId
    model: SpreadModel

    # Config
    n_splits: int
    test_days: int
    embargo_days: int

    # Per-fold
    folds: list[FoldResult] = field(default_factory=list)

    # Aggregate metrics (across valid folds only)
    avg_sharpe: float = np.nan
    avg_pnl: float = np.nan
    avg_win_rate: float = np.nan
    avg_max_drawdown: float = np.nan
    validation_pass_rate: float = np.nan  # % folds that passed validation
    consistent_sharpe: float = np.nan     # std of fold Sharpes (lower = more consistent)

    # Regime-conditional breakdown
    regime_conditional_sharpe: dict[str, float] = field(default_factory=dict)
    regime_dependent: bool = False        # True = strategy only works in MEAN_REVERTING

    # Overall verdict
    viable: bool = False
    viability_reason: str = ""
    warnings: list[str] = field(default_factory=list)

    # Metadata
    run_at: datetime = field(default_factory=datetime.utcnow)
    total_price_days: int = 0

    def valid_folds(self) -> list[FoldResult]:
        return [f for f in self.folds if not f.skipped and not np.isnan(f.sharpe)]

    def to_summary_dict(self) -> dict:
        return {
            "pair_label": self.pair_id.label,
            "model": self.model.value,
            "n_splits": self.n_splits,
            "test_days": self.test_days,
            "embargo_days": self.embargo_days,
            "avg_sharpe": self.avg_sharpe,
            "avg_pnl": self.avg_pnl,
            "avg_win_rate": self.avg_win_rate,
            "avg_max_drawdown": self.avg_max_drawdown,
            "validation_pass_rate": self.validation_pass_rate,
            "consistent_sharpe": self.consistent_sharpe,
            "regime_conditional_sharpe": self.regime_conditional_sharpe,
            "regime_dependent": self.regime_dependent,
            "viable": self.viable,
            "viability_reason": self.viability_reason,
            "warnings": self.warnings,
            "run_at": self.run_at,
            "total_price_days": self.total_price_days,
        }

    def to_fold_df(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.folds])


# ── Simple mean-reversion backtest ────────────────────────────────

@dataclass
class _Trade:
    entry_date: datetime
    entry_z: float
    direction: int  # +1 = long spread, -1 = short spread
    exit_date: Optional[datetime] = None
    exit_z: Optional[float] = None
    pnl: float = 0.0
    holding_days: int = 0


def _run_simple_backtest(
    z: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss: float = 4.0,
    max_holding_days: int = 60,
) -> dict:
    """
    Simple threshold mean-reversion backtest on z-score series.

    Entry: |z| > entry_threshold
    Exit:  |z| < exit_threshold, OR |z| > stop_loss, OR max holding exceeded
    PnL is measured in z-score units (normalised).

    Returns summary metrics dict.
    """
    trades: list[_Trade] = []
    current_trade: Optional[_Trade] = None

    for i, (date, z_val) in enumerate(z.items()):
        if np.isnan(z_val):
            continue

        if current_trade is None:
            # Check for entry
            if z_val > entry_threshold:
                current_trade = _Trade(
                    entry_date=date,
                    entry_z=z_val,
                    direction=-1,  # short spread when z > +threshold
                )
            elif z_val < -entry_threshold:
                current_trade = _Trade(
                    entry_date=date,
                    entry_z=z_val,
                    direction=+1,  # long spread when z < -threshold
                )
        else:
            holding = i - z.index.get_loc(current_trade.entry_date)  # type: ignore[arg-type]
            abs_z = abs(z_val)

            should_exit = (
                abs_z < exit_threshold
                or abs_z > stop_loss
                or holding >= max_holding_days
            )

            if should_exit:
                current_trade.exit_date = date
                current_trade.exit_z = z_val
                # PnL = direction * (entry_z - exit_z)  [in z-units]
                current_trade.pnl = current_trade.direction * (
                    current_trade.entry_z - z_val
                )
                current_trade.holding_days = holding
                trades.append(current_trade)
                current_trade = None

    # Close any open trade at end
    if current_trade is not None and len(z) > 0:
        last_date = z.index[-1]
        last_z = z.iloc[-1]
        current_trade.exit_date = last_date
        current_trade.exit_z = last_z
        current_trade.pnl = current_trade.direction * (current_trade.entry_z - last_z)
        current_trade.holding_days = len(z) - z.index.get_loc(current_trade.entry_date)  # type: ignore[arg-type]
        trades.append(current_trade)

    if not trades:
        return {
            "n_trades": 0, "pnl": 0.0, "sharpe": np.nan,
            "max_drawdown": np.nan, "win_rate": np.nan, "avg_holding_days": np.nan,
        }

    pnls = [t.pnl for t in trades]
    cumulative = np.cumsum(pnls)

    # Sharpe (annualised from per-trade returns)
    # Use sqrt(252) for annualization — sqrt(n_trades) was wrong (not annualised)
    n_trades = len(pnls)
    if n_trades >= 2:
        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1))
        sharpe = avg_pnl / max(std_pnl, 1e-8) * np.sqrt(252)
    else:
        avg_pnl = float(np.mean(pnls)) if pnls else 0.0
        sharpe = 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(np.max(drawdowns))

    return {
        "n_trades": len(trades),
        "pnl": float(np.sum(pnls)),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(np.mean([p > 0 for p in pnls])),
        "avg_holding_days": float(np.mean([t.holding_days for t in trades])),
    }


# ── Regime-segmented validation helpers ──────────────────────────

def _segment_by_regime(
    fold_results: list[dict],
    regime_key: str = "dominant_regime",
) -> dict[str, list[dict]]:
    """Group fold results by dominant regime during test period."""
    segmented: dict[str, list[dict]] = {}
    for fold in fold_results:
        regime = fold.get(regime_key, "UNKNOWN")
        if regime not in segmented:
            segmented[regime] = []
        segmented[regime].append(fold)
    return segmented


def _compute_regime_conditional_sharpe(
    fold_results: list[dict],
) -> dict[str, float]:
    """
    Compute average OOS Sharpe separately for each regime.
    Returns dict: regime_label → avg_sharpe.
    """
    segmented = _segment_by_regime(fold_results)
    regime_sharpes: dict[str, float] = {}
    for regime, folds in segmented.items():
        sharpes = [f.get("oos_sharpe", 0.0) for f in folds if f.get("oos_sharpe") is not None]
        if sharpes:
            regime_sharpes[regime] = float(np.mean(sharpes))
    return regime_sharpes


# ── Walk-forward split generator ──────────────────────────────────

def _generate_splits(
    index: pd.DatetimeIndex,
    n_splits: int,
    test_days: int,
    min_train_days: int,
    embargo_days: int,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """
    Generate (train_start, train_end, test_start, test_end) tuples.

    Uses an expanding-window design:
    - test windows are non-overlapping and contiguous (no gaps between folds)
    - training window grows with each fold
    - embargo_days gap inserted between train_end and test_start

    Returns list of (train_start, train_end, test_start, test_end).
    """
    if len(index) == 0:
        return []

    total_days = len(index)
    required_days = min_train_days + n_splits * test_days + (n_splits - 1) * embargo_days
    if total_days < required_days:
        logger.warning(
            "Insufficient data: %d days available, %d required for %d splits",
            total_days, required_days, n_splits,
        )
        # Reduce splits if needed
        n_splits = max(1, (total_days - min_train_days) // (test_days + embargo_days))
        if n_splits == 0:
            return []

    splits = []
    # Work backwards from the end: last fold has test_end = last date
    fold_boundaries = []
    end_idx = total_days
    for _ in range(n_splits):
        test_end_idx = end_idx
        test_start_idx = test_end_idx - test_days
        if test_start_idx <= min_train_days:
            break
        fold_boundaries.append((test_start_idx, test_end_idx))
        end_idx = test_start_idx - embargo_days

    fold_boundaries.reverse()

    train_start_idx = 0
    for test_start_idx, test_end_idx in fold_boundaries:
        train_end_idx = test_start_idx - embargo_days

        if train_end_idx - train_start_idx < min_train_days:
            continue

        splits.append((
            index[train_start_idx].to_pydatetime(),
            index[train_end_idx - 1].to_pydatetime(),
            index[test_start_idx].to_pydatetime(),
            index[min(test_end_idx - 1, total_days - 1)].to_pydatetime(),
        ))

    return splits


# ── Walk-Forward Harness ──────────────────────────────────────────

class WalkForwardHarness:
    """
    Walk-forward validation harness for pairs trading research.

    For each fold:
    1. Fit PairValidator on training data → skip fold if hard validation fails
    2. Fit SpreadConstructor on training data → SpreadDefinition
    3. Apply SpreadDefinition to test data → z-score series
    4. Run simple mean-reversion backtest on test z-scores
    5. Compute spread diagnostics on test window

    All fitting uses strictly training data. Test window is never touched during fitting.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_days: int = 252,
        min_train_days: int = 504,
        embargo_days: int = 20,
        # Backtest parameters
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        max_holding_days: int = 60,
        # Validation strictness
        validator: Optional[PairValidator] = None,
        skip_fold_on_validation_fail: bool = True,
    ):
        self.n_splits = n_splits
        self.test_days = test_days
        self.min_train_days = min_train_days
        self.embargo_days = embargo_days
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.max_holding_days = max_holding_days
        self.validator = validator or PairValidator()
        self.skip_fold_on_validation_fail = skip_fold_on_validation_fail

    def run(
        self,
        pair_id: PairId,
        prices: pd.DataFrame,
        *,
        model: SpreadModel = SpreadModel.STATIC_OLS,
        window: int = 60,
        use_log_prices: bool = True,
        kalman_obs_cov: float = 0.001,
        kalman_trans_cov: float = 0.0001,
    ) -> ExperimentResult:
        """
        Run full walk-forward experiment for a pair.

        Parameters
        ----------
        pair_id : PairId
        prices : pd.DataFrame
            Must contain columns for both pair legs.
        model : SpreadModel
            Spread construction method.
        window : int
            Rolling window for spread normalisation (also used by RollingOLS).

        Returns
        -------
        ExperimentResult with fold-level and aggregated metrics.
        """
        sym_x, sym_y = pair_id.sym_x, pair_id.sym_y
        if sym_x not in prices.columns or sym_y not in prices.columns:
            raise ValueError(f"prices must contain {sym_x} and {sym_y}")

        # Align and clean
        px = prices[[sym_x, sym_y]].dropna()
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()

        result = ExperimentResult(
            pair_id=pair_id,
            model=model,
            n_splits=self.n_splits,
            test_days=self.test_days,
            embargo_days=self.embargo_days,
            total_price_days=len(px),
        )

        splits = _generate_splits(
            px.index,
            n_splits=self.n_splits,
            test_days=self.test_days,
            min_train_days=self.min_train_days,
            embargo_days=self.embargo_days,
        )

        if not splits:
            result.viability_reason = (
                f"Insufficient data: {len(px)} days, need "
                f"{self.min_train_days + self.n_splits * self.test_days}"
            )
            logger.warning("%s: %s", pair_id.label, result.viability_reason)
            return result

        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
            fold = self._run_fold(
                fold_idx=fold_idx,
                pair_id=pair_id,
                prices=px,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                model=model,
                window=window,
                use_log_prices=use_log_prices,
                kalman_obs_cov=kalman_obs_cov,
                kalman_trans_cov=kalman_trans_cov,
            )
            result.folds.append(fold)

        self._aggregate(result)
        return result

    def _run_fold(
        self,
        *,
        fold_idx: int,
        pair_id: PairId,
        prices: pd.DataFrame,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        model: SpreadModel,
        window: int,
        use_log_prices: bool,
        kalman_obs_cov: float,
        kalman_trans_cov: float,
    ) -> FoldResult:
        """Execute one walk-forward fold."""
        logger.debug(
            "Fold %d: train [%s → %s], embargo, test [%s → %s]",
            fold_idx, train_start.date(), train_end.date(),
            test_start.date(), test_end.date(),
        )

        # Extract train slice
        train_prices = prices.loc[
            (prices.index >= pd.Timestamp(train_start))
            & (prices.index <= pd.Timestamp(train_end))
        ]

        # Step 1: Validate on training data
        validation = self.validator.validate(
            pair_id, train_prices, train_end=train_end
        )

        fold = FoldResult(
            fold_idx=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            validation=validation,
        )

        if self.skip_fold_on_validation_fail and validation.result == ValidationResult.FAIL:
            fold.skipped = True
            fold.skip_reason = "; ".join(validation.rejection_reasons) or "FAIL"
            logger.debug(
                "Fold %d skipped: %s", fold_idx, fold.skip_reason
            )
            return fold

        # Step 2: Fit spread on training data
        try:
            defn, train_z = build_spread(
                pair_id=pair_id,
                prices=train_prices,
                model=model,
                train_end=train_end,
                window=window,
                use_log_prices=use_log_prices,
                kalman_obs_cov=kalman_obs_cov,
                kalman_trans_cov=kalman_trans_cov,
            )
        except Exception as exc:
            fold.skipped = True
            fold.skip_reason = f"spread_fit_failed: {exc}"
            logger.warning("Fold %d spread fit failed: %s", fold_idx, exc)
            return fold

        fold.spread_defn = defn
        fold.train_z = train_z
        fold.spread_mean = defn.mean
        fold.spread_std = defn.std

        # Step 3: Apply spread to test data (no re-fitting)
        test_prices = prices.loc[
            (prices.index >= pd.Timestamp(test_start))
            & (prices.index <= pd.Timestamp(test_end))
        ]

        if len(test_prices) < 20:
            fold.skipped = True
            fold.skip_reason = f"insufficient_test_data: {len(test_prices)} rows"
            return fold

        try:
            from research.spread_constructor import (
                KalmanConstructor,
                RollingOLSConstructor,
                StaticOLSConstructor,
            )
            sym_x, sym_y = pair_id.sym_x, pair_id.sym_y
            px_test = test_prices[sym_x]
            py_test = test_prices[sym_y]

            if model == SpreadModel.STATIC_OLS:
                ctor = StaticOLSConstructor(use_log_prices=use_log_prices)
                test_z = ctor.transform(px_test, py_test, defn)
            elif model == SpreadModel.ROLLING_OLS:
                ctor = RollingOLSConstructor(window=window, use_log_prices=use_log_prices)
                test_z = ctor.transform(px_test, py_test, defn, rolling_window=window)
            elif model == SpreadModel.KALMAN:
                ctor = KalmanConstructor(
                    observation_cov=kalman_obs_cov,
                    transition_cov=kalman_trans_cov,
                    use_log_prices=use_log_prices,
                    initial_beta=defn.hedge_ratio,
                )
                test_z = ctor.transform(px_test, py_test, defn, rolling_window=30)
            else:
                raise ValueError(f"Unsupported model: {model}")
        except Exception as exc:
            fold.skipped = True
            fold.skip_reason = f"transform_failed: {exc}"
            logger.warning("Fold %d transform failed: %s", fold_idx, exc)
            return fold

        fold.test_z = test_z

        # Step 4: Spread diagnostics on test window
        test_raw = test_z.dropna()
        if len(test_raw) > 5:
            fold.spread_autocorr = float(test_raw.autocorr(lag=1))
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_stat, adf_p, *_ = adfuller(test_raw.values, maxlag=5, autolag=None)
                fold.test_adf_pvalue = float(adf_p)
            except Exception:
                pass

        # Step 5: Backtest on test z-scores
        bt = _run_simple_backtest(
            test_z,
            entry_threshold=self.entry_z,
            exit_threshold=self.exit_z,
            stop_loss=self.stop_z,
            max_holding_days=self.max_holding_days,
        )

        fold.n_trades = bt["n_trades"]
        fold.pnl = bt["pnl"]
        fold.sharpe = bt["sharpe"]
        fold.max_drawdown = bt["max_drawdown"]
        fold.win_rate = bt["win_rate"]
        fold.avg_holding_days = bt["avg_holding_days"]

        return fold

    def _aggregate(self, result: ExperimentResult) -> None:
        """Compute aggregate metrics from fold results."""
        valid = result.valid_folds()

        all_folds = [f for f in result.folds if not f.skipped]
        n_total = len(result.folds)
        n_passed = sum(
            1 for f in result.folds
            if not f.skipped and f.validation.result != ValidationResult.FAIL
        )

        result.validation_pass_rate = n_passed / n_total if n_total > 0 else 0.0

        if not valid:
            result.viability_reason = "No valid folds produced a backtest"
            return

        sharpes = [f.sharpe for f in valid]
        result.avg_sharpe = float(np.mean(sharpes))
        result.avg_pnl = float(np.mean([f.pnl for f in valid]))
        result.avg_win_rate = float(np.nanmean([f.win_rate for f in valid if not np.isnan(f.win_rate)]))
        result.avg_max_drawdown = float(np.nanmean([f.max_drawdown for f in valid if not np.isnan(f.max_drawdown)]))
        result.consistent_sharpe = float(np.std(sharpes)) if len(sharpes) > 1 else 0.0

        # Viability: require avg_sharpe > 0.5, pass_rate > 50%, >= 2 valid folds
        if (
            result.avg_sharpe > 0.5
            and result.validation_pass_rate >= 0.5
            and len(valid) >= 2
        ):
            result.viable = True
            result.viability_reason = (
                f"avg_sharpe={result.avg_sharpe:.2f}, "
                f"pass_rate={result.validation_pass_rate:.0%}, "
                f"valid_folds={len(valid)}"
            )
        else:
            reasons = []
            if result.avg_sharpe <= 0.5:
                reasons.append(f"avg_sharpe={result.avg_sharpe:.2f}<=0.5")
            if result.validation_pass_rate < 0.5:
                reasons.append(f"pass_rate={result.validation_pass_rate:.0%}<50%")
            if len(valid) < 2:
                reasons.append(f"only {len(valid)} valid fold(s)")
            result.viability_reason = "; ".join(reasons)

        # ── Regime-segmented Sharpe breakdown ────────────────────────
        # Build fold dicts compatible with _compute_regime_conditional_sharpe.
        # The dominant_regime is sourced from the validation report's stability
        # regime label (computed on the training window); oos_sharpe maps to the
        # fold's backtest Sharpe on the held-out test window.
        fold_dicts_for_regime: list[dict] = []
        for f in valid:
            dominant_regime = "UNKNOWN"
            if f.validation is not None:
                # PairValidationReport may expose regime_label via its stability report
                dominant_regime = getattr(f.validation, "regime_label", "UNKNOWN") or "UNKNOWN"
            fold_dicts_for_regime.append({
                "dominant_regime": dominant_regime,
                "oos_sharpe": f.sharpe if not np.isnan(f.sharpe) else None,
            })

        regime_sharpes = _compute_regime_conditional_sharpe(fold_dicts_for_regime)
        result.regime_conditional_sharpe = regime_sharpes

        # Flag regime-dependent strategies: performs well only in MEAN_REVERTING
        if regime_sharpes:
            mr_sharpe = regime_sharpes.get("MEAN_REVERTING", 0.0)
            other_sharpes = [v for k, v in regime_sharpes.items() if k != "MEAN_REVERTING"]
            avg_other = float(np.mean(other_sharpes)) if other_sharpes else 0.0
            result.regime_dependent = (mr_sharpe > 0.5) and (avg_other < 0.1)
            if result.regime_dependent:
                result.warnings = result.warnings + [
                    f"Strategy appears regime-dependent: MR Sharpe={mr_sharpe:.2f}, Other={avg_other:.2f}"
                ]
                logger.warning(
                    "%s: regime-dependent — MR Sharpe=%.2f, other avg=%.2f",
                    result.pair_id.label, mr_sharpe, avg_other,
                )


# ── Batch runner ──────────────────────────────────────────────────

def run_batch_walk_forward(
    pair_ids: list[PairId],
    prices: pd.DataFrame,
    *,
    model: SpreadModel = SpreadModel.STATIC_OLS,
    n_splits: int = 5,
    test_days: int = 252,
    min_train_days: int = 504,
    embargo_days: int = 20,
    window: int = 60,
    max_workers: int = 1,
) -> list[ExperimentResult]:
    """
    Run walk-forward experiments for a list of pairs.

    Parameters
    ----------
    pair_ids : list of PairId
    prices : pd.DataFrame with all required columns
    model : SpreadModel
    n_splits, test_days, min_train_days, embargo_days, window : harness config
    max_workers : if > 1, run pairs in parallel using ThreadPoolExecutor

    Returns
    -------
    List of ExperimentResult, one per pair.
    """
    harness = WalkForwardHarness(
        n_splits=n_splits,
        test_days=test_days,
        min_train_days=min_train_days,
        embargo_days=embargo_days,
    )

    results: list[ExperimentResult] = []

    if max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for pid in pair_ids:
                fut = executor.submit(
                    harness.run, pid, prices,
                    model=model, window=window
                )
                futures[fut] = pid

            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as exc:
                    logger.error("Walk-forward failed for %s: %s", pid.label, exc)
    else:
        for pid in pair_ids:
            try:
                res = harness.run(pid, prices, model=model, window=window)
                results.append(res)
                logger.info(
                    "%s: viable=%s, avg_sharpe=%.2f, valid_folds=%d/%d",
                    pid.label, res.viable,
                    res.avg_sharpe if not np.isnan(res.avg_sharpe) else -999,
                    len(res.valid_folds()), res.n_splits,
                )
            except Exception as exc:
                logger.error("Walk-forward failed for %s: %s", pid.label, exc)

    return results


def summarise_experiments(results: list[ExperimentResult]) -> pd.DataFrame:
    """Convert a list of ExperimentResult into a summary DataFrame."""
    rows = [r.to_summary_dict() for r in results]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("avg_sharpe", ascending=False).reset_index(drop=True)
    return df
