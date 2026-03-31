#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train_meta_label.py — Meta-Label Model Training Script
==============================================================

Trains a meta-labeling model that predicts whether a rule-based signal
should be TAKEN or SKIPPED, based on recent spread/regime/signal features.

This is the first real ML training pipeline in the platform (P2-MLT).

Usage:
    python scripts/train_meta_label.py --pair AAPL-MSFT --lookback 504

    # Or for a batch of pairs:
    python scripts/train_meta_label.py --all-pairs --lookback 504

The trained model can be plugged into the SignalPipeline:
    pipeline = SignalPipeline(pair_id=pair_id, ml_quality_hook=meta_model)

Doctrine (from CLAUDE.md):
    - ML is an overlay, not the foundation
    - Every ML decision has a deterministic fallback
    - Point-in-time correctness is non-negotiable
    - Labels use future data BY DESIGN (LEAKAGE NOTE)
    - Champion promotion requires governance approval
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.meta_labeler import MetaLabelModel
from ml.models.base import META_LABEL_FEATURES
from ml.labels.builder import LabelBuilder
from ml.contracts import (
    MLTaskFamily, ModelStatus, ModelMetadata,
    TrainingRunArtifact, PromotionOutcome,
)

logger = logging.getLogger(__name__)


def compute_features_for_pair(
    px: pd.Series,
    py: pd.Series,
    z: pd.Series,
    spread: pd.Series,
    *,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute meta-label features from price/spread data.

    This function computes features that are available at time T using
    only data up to T (point-in-time safe).  It parallels the definitions
    in ml/features/definitions.py but is self-contained for training.

    Parameters
    ----------
    px, py : pd.Series
        Price series for leg X and leg Y.
    z : pd.Series
        Z-score spread series.
    spread : pd.Series
        Raw spread series.
    feature_names : list[str], optional
        Which features to compute.  Defaults to META_LABEL_FEATURES.

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by date.
    """
    features = pd.DataFrame(index=z.index)

    # Z-score features (pair-level)
    features["z"] = z
    features["z_abs"] = z.abs()
    features["z_mean_5d"] = z.rolling(5, min_periods=3).mean()
    features["z_mean_20d"] = z.rolling(20, min_periods=10).mean()
    features["z_std_5d"] = z.rolling(5, min_periods=3).std()
    features["z_std_20d"] = z.rolling(20, min_periods=10).std()
    features["z_mom_5d"] = z.diff(5)
    features["z_mom_20d"] = z.diff(20)
    features["z_ar1"] = z.shift(1)

    # Crossing zero count in last 20 days
    sign_changes = (z * z.shift(1) < 0).astype(float)
    features["z_cross_zero_20d"] = sign_changes.rolling(20, min_periods=10).sum()

    # Bollinger position
    bb_mean = z.rolling(20, min_periods=10).mean()
    bb_std = z.rolling(20, min_periods=10).std().replace(0, np.nan)
    features["bb_pos_20d"] = (z - bb_mean) / bb_std
    features["bb_pos_20d"] = features["bb_pos_20d"].fillna(0.0)

    # Divergence speed
    features["div_speed"] = z.diff(1).abs()

    # Returns features (instrument-level, using leg X as proxy)
    ret_x = px.pct_change()
    features["ret_1d"] = ret_x
    features["ret_5d"] = px.pct_change(5)
    features["ret_20d"] = px.pct_change(20)
    features["vol_20d"] = ret_x.rolling(20, min_periods=10).std() * np.sqrt(252)
    features["vol_60d"] = ret_x.rolling(60, min_periods=30).std() * np.sqrt(252)
    vol_20 = features["vol_20d"].replace(0, np.nan)
    vol_60 = features["vol_60d"].replace(0, np.nan)
    features["vol_ratio"] = vol_20 / vol_60
    features["vol_ratio"] = features["vol_ratio"].fillna(1.0)

    # Correlation features
    features["corr_20d"] = px.rolling(20, min_periods=15).corr(py)
    features["corr_60d"] = px.rolling(60, min_periods=30).corr(py)

    # Spread volatility
    spread_ret = spread.pct_change().replace([np.inf, -np.inf], np.nan)
    features["spread_vol_20d"] = spread_ret.rolling(20, min_periods=10).std()

    # Half-life proxy (via AR(1) coefficient)
    z_lag = z.shift(1)
    rolling_corr = z.rolling(60, min_periods=30).corr(z_lag)
    features["half_life"] = -np.log(2) / np.log(rolling_corr.abs().clip(0.01, 0.99))
    features["half_life"] = features["half_life"].clip(1, 200).fillna(20.0)

    # Mean reversion quality proxy
    features["mr_quality"] = (1.0 - rolling_corr.abs()).clip(0, 1).fillna(0.5)

    # Signal context features
    features["z_at_entry"] = z  # Current z (proxy for entry z)
    features["entry_z_percentile"] = z.rolling(252, min_periods=60).rank(pct=True)

    # Fill remaining NaN with 0.0 for model robustness
    features = features.fillna(0.0)

    # Select only requested features (with fallback for missing)
    if feature_names:
        available = [f for f in feature_names if f in features.columns]
        missing = [f for f in feature_names if f not in features.columns]
        if missing:
            for m in missing:
                features[m] = 0.0
            logger.info(
                "Features %s not computable — filled with 0.0",
                missing[:5],
            )
        features = features[feature_names]

    return features


def train_meta_label_model(
    px: pd.Series,
    py: pd.Series,
    z: pd.Series,
    spread: pd.Series,
    *,
    train_end: str | pd.Timestamp,
    label_horizon: int = 10,
    cost_bps: float = 30.0,
    entry_threshold: float = 2.0,
    pair_id: str = "UNKNOWN",
) -> tuple[MetaLabelModel, TrainingRunArtifact, dict]:
    """
    Train a meta-label model on a single pair's data.

    Parameters
    ----------
    px, py : pd.Series
        Price series for legs X and Y.
    z : pd.Series
        Z-score spread series.
    spread : pd.Series
        Raw spread series.
    train_end : str or pd.Timestamp
        Hard cutoff — features computed from data <= train_end,
        labels use data AFTER train_end (by design, not leakage).
    label_horizon : int
        Days forward for label computation (default 10).
    cost_bps : float
        Round-trip cost for profitability label (default 30 bps).
    pair_id : str
        Identifier for logging.

    Returns
    -------
    tuple[MetaLabelModel, TrainingRunArtifact, dict]
        Trained model, training artifact, evaluation metrics dict.
    """
    train_end_ts = pd.Timestamp(train_end)

    # 1. Compute features (point-in-time safe)
    feature_names = list(META_LABEL_FEATURES)
    X_all = compute_features_for_pair(
        px, py, z, spread, feature_names=feature_names,
    )

    # 2. Compute labels (uses future data — offline training only)
    label_builder = LabelBuilder()
    y_all = label_builder.build_meta_take_label(
        z=z,
        horizon=label_horizon,
        entry_threshold=entry_threshold,
        cost_bps=cost_bps,
    )

    # 3. Align features and labels
    common_idx = X_all.index.intersection(y_all.dropna().index)
    X_all = X_all.loc[common_idx]
    y_all = y_all.loc[common_idx]

    # 4. Split train/test at train_end (temporal split)
    train_mask = X_all.index <= train_end_ts
    test_mask = X_all.index > train_end_ts

    X_train = X_all.loc[train_mask]
    y_train = y_all.loc[train_mask]
    X_test = X_all.loc[test_mask]
    y_test = y_all.loc[test_mask]

    if len(X_train) < 50:
        raise ValueError(
            f"Insufficient training data: {len(X_train)} samples "
            f"(need >= 50) for pair {pair_id}"
        )

    logger.info(
        "Training meta-label model for %s: %d train, %d test samples",
        pair_id, len(X_train), len(X_test),
    )

    # 5. Train model
    model = MetaLabelModel(
        label_name=f"meta_take_{label_horizon}d",
        feature_names=feature_names,
    )
    artifact = model.fit(
        X_train, y_train,
        train_end=train_end_ts,
    )

    # 6. Evaluate on test set
    metrics = {}
    if len(X_test) > 10:
        try:
            from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

            y_pred_proba = model._model.predict_proba(X_test)
            if y_pred_proba.shape[1] > 1:
                y_scores = y_pred_proba[:, 1]
            else:
                y_scores = y_pred_proba[:, 0]

            metrics["val_auc"] = float(roc_auc_score(y_test, y_scores))
            metrics["val_brier"] = float(brier_score_loss(y_test, y_scores))
            metrics["val_logloss"] = float(log_loss(y_test, y_scores))
            metrics["n_train"] = len(X_train)
            metrics["n_test"] = len(X_test)
            metrics["label_rate"] = float(y_train.mean())
            metrics["test_label_rate"] = float(y_test.mean())

            logger.info(
                "Meta-label evaluation for %s: AUC=%.3f, Brier=%.3f, LogLoss=%.3f",
                pair_id, metrics["val_auc"], metrics["val_brier"],
                metrics["val_logloss"],
            )
        except Exception as e:
            logger.warning("Evaluation failed for %s: %s", pair_id, e)
            metrics["eval_error"] = str(e)
    else:
        logger.warning(
            "Insufficient test data for %s: %d samples", pair_id, len(X_test),
        )
        metrics["n_test"] = len(X_test)

    return model, artifact, metrics


def main():
    """CLI entry point for meta-label training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train meta-labeling model for signal filtering",
    )
    parser.add_argument(
        "--pair", type=str, default="AAPL-MSFT",
        help="Pair to train on (e.g. 'AAPL-MSFT')",
    )
    parser.add_argument(
        "--lookback", type=int, default=504,
        help="Days of history to use (default 504 = 2 years)",
    )
    parser.add_argument(
        "--horizon", type=int, default=10,
        help="Label horizon in trading days (default 10)",
    )
    parser.add_argument(
        "--cost-bps", type=float, default=30.0,
        help="Round-trip cost in basis points (default 30)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.8,
        help="Fraction of data for training (default 0.8)",
    )
    parser.add_argument(
        "--save-path", type=str, default=None,
        help="Where to save the model (default: models/meta_label_{pair}.pkl)",
    )
    args = parser.parse_args()

    sym_a, sym_b = args.pair.split("-", 1)

    # Load price data
    try:
        from common.data_loader import load_price_data
        df_a = load_price_data(sym_a)
        df_b = load_price_data(sym_b)
    except Exception as e:
        logger.error("Failed to load data for %s: %s", args.pair, e)
        sys.exit(1)

    if df_a.empty or df_b.empty:
        logger.error("Empty data for %s or %s", sym_a, sym_b)
        sys.exit(1)

    # Extract close prices
    px = df_a["close"].tail(args.lookback)
    py = df_b["close"].tail(args.lookback)

    # Align
    common = px.index.intersection(py.index)
    px = px.loc[common]
    py = py.loc[common]

    # Compute spread and z-score
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    X_ols = add_constant(px.values)
    result = OLS(py.values, X_ols).fit()
    beta = result.params[1]
    alpha = result.params[0]

    spread = py - beta * px - alpha
    lookback = 60
    mu = spread.rolling(lookback, min_periods=30).mean()
    sigma = spread.rolling(lookback, min_periods=30).std().replace(0, np.nan)
    z = ((spread - mu) / sigma).fillna(0.0)

    # Train/test split
    train_end_idx = int(len(px) * args.train_frac)
    train_end = str(px.index[train_end_idx].date())

    logger.info(
        "Training meta-label for %s: %d observations, train_end=%s",
        args.pair, len(px), train_end,
    )

    model, artifact, metrics = train_meta_label_model(
        px=px, py=py, z=z, spread=spread,
        train_end=train_end,
        label_horizon=args.horizon,
        cost_bps=args.cost_bps,
        pair_id=args.pair,
    )

    # Save model
    save_path = args.save_path or f"models/meta_label_{args.pair}.pkl"
    saved = model.save(save_path)
    logger.info("Model saved to %s", saved)

    # Print results
    print(f"\n{'='*60}")
    print(f"Meta-Label Training Complete: {args.pair}")
    print(f"{'='*60}")
    print(f"  Train samples: {metrics.get('n_train', '?')}")
    print(f"  Test samples:  {metrics.get('n_test', '?')}")
    print(f"  Label rate:    {metrics.get('label_rate', '?'):.3f}")
    if "val_auc" in metrics:
        print(f"  AUC:           {metrics['val_auc']:.3f}")
        print(f"  Brier:         {metrics['val_brier']:.3f}")
        print(f"  LogLoss:       {metrics['val_logloss']:.3f}")
    print(f"  Model saved:   {saved}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
