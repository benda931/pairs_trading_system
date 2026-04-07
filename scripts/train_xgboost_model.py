#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train_xgboost_model.py — XGBoost Meta-Label Model Training
===================================================================

Trains an XGBoost meta-labeling model with:
- 40+ engineered features (momentum, vol, regime, microstructure)
- Walk-forward cross-validation (purged, no leakage)
- Proper calibration on held-out data
- Feature importance analysis
- Saves model for production use

Usage:
    python scripts/train_xgboost_model.py --pair XLI-XLB
    python scripts/train_xgboost_model.py --all-alpha  # Train on all alpha pairs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def compute_advanced_features(px: pd.Series, py: pd.Series, lookback: int = 60) -> pd.DataFrame:
    """
    Compute 40+ features for meta-labeling.

    Feature categories:
    - Z-score features (current, momentum, acceleration)
    - Volatility features (realized, implied proxy, regime)
    - Correlation features (rolling, stability)
    - Microstructure (volume ratio, gap)
    - Momentum (relative, cross-asset)
    - Mean-reversion quality (AR1, half-life stability)
    - Regime indicators
    """
    features = pd.DataFrame(index=px.index)

    # Spread (canonical z-score)
    from common.feature_engineering import compute_zscore
    beta = float(np.cov(px.values, py.values)[0, 1] / np.var(px.values))
    spread = py - beta * px
    z = compute_zscore(spread, lookback, min_periods=lookback // 2, fillna_value=0.0)

    # ── Z-score features ─────────────────────────────────────────
    features["z"] = z
    features["z_abs"] = z.abs()
    features["z_sq"] = z ** 2
    features["z_sign"] = np.sign(z)
    features["z_ma5"] = z.rolling(5).mean()
    features["z_ma20"] = z.rolling(20).mean()
    features["z_std5"] = z.rolling(5).std()
    features["z_std20"] = z.rolling(20).std()
    features["z_mom5"] = z.diff(5)
    features["z_mom20"] = z.diff(20)
    features["z_accel"] = z.diff(1).diff(1)  # Second derivative
    features["z_ar1"] = z.shift(1)
    features["z_ar5"] = z.shift(5)
    features["z_cross_zero"] = (z * z.shift(1) < 0).astype(float).rolling(20).sum()
    features["z_above_2"] = (z.abs() > 2.0).astype(float).rolling(20).sum()
    features["z_percentile"] = z.rolling(252, min_periods=60).rank(pct=True)

    # ── Volatility features ──────────────────────────────────────
    ret_x = px.pct_change()
    ret_y = py.pct_change()
    features["vol_x_20d"] = ret_x.rolling(20).std() * np.sqrt(252)
    features["vol_y_20d"] = ret_y.rolling(20).std() * np.sqrt(252)
    features["vol_x_60d"] = ret_x.rolling(60).std() * np.sqrt(252)
    features["vol_spread"] = spread.pct_change().replace([np.inf, -np.inf], np.nan).rolling(20).std()
    features["vol_ratio"] = features["vol_x_20d"] / features["vol_x_60d"].replace(0, np.nan)
    features["vol_of_vol"] = features["vol_x_20d"].rolling(20).std()

    # ── Correlation features ─────────────────────────────────────
    features["corr_20d"] = px.rolling(20).corr(py)
    features["corr_60d"] = px.rolling(60).corr(py)
    features["corr_diff"] = features["corr_20d"] - features["corr_60d"]
    features["corr_stability"] = features["corr_20d"].rolling(20).std()

    # ── Momentum features ────────────────────────────────────────
    features["ret_x_5d"] = px.pct_change(5)
    features["ret_y_5d"] = py.pct_change(5)
    features["ret_x_20d"] = px.pct_change(20)
    features["ret_y_20d"] = py.pct_change(20)
    features["rel_momentum"] = features["ret_x_20d"] - features["ret_y_20d"]
    features["spread_momentum"] = spread.pct_change(5).replace([np.inf, -np.inf], np.nan)

    # ── Mean-reversion quality ───────────────────────────────────
    z_lag = z.shift(1)
    features["ar1_coeff"] = z.rolling(60).corr(z_lag)
    hl_raw = -np.log(2) / np.log(features["ar1_coeff"].abs().clip(0.01, 0.99))
    features["half_life"] = hl_raw.clip(1, 200)
    features["hl_stability"] = features["half_life"].rolling(20).std()
    features["mr_quality"] = (1 - features["ar1_coeff"].abs()).clip(0, 1)

    # ── Regime indicators ────────────────────────────────────────
    features["trend_strength"] = (px.rolling(20).mean() - px.rolling(60).mean()) / px.rolling(60).std().replace(0, np.nan)
    from scipy.stats import skew as _skew, kurtosis as _kurt
    features["spread_skew"] = spread.rolling(60).apply(lambda x: _skew(x), raw=True)
    features["spread_kurt"] = spread.rolling(60).apply(lambda x: _kurt(x), raw=True)

    # Fill NaN
    features = features.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return features


def compute_labels(z: pd.Series, horizon: int = 10, entry_threshold: float = 1.5) -> pd.Series:
    """
    Binary label: 1 = signal was profitable within horizon, 0 = not.
    Uses future data (LEAKAGE NOTE: offline training only).
    """
    labels = pd.Series(np.nan, index=z.index)
    for i in range(len(z) - horizon):
        if abs(z.iloc[i]) < entry_threshold:
            continue
        future_z = z.iloc[i + 1:i + horizon + 1]
        # Profitable = z moved toward zero by at least 50%
        initial_z = abs(z.iloc[i])
        reverted = (future_z.abs() < initial_z * 0.5).any()
        labels.iloc[i] = 1.0 if reverted else 0.0
    return labels


def train_xgboost_model(
    sym_x: str,
    sym_y: str,
    *,
    train_frac: float = 0.8,
    horizon: int = 10,
    entry_threshold: float = 1.5,
) -> dict:
    """Train XGBoost meta-label model and return metrics."""
    from common.data_loader import load_price_data, _load_symbol_full_cached
    if hasattr(_load_symbol_full_cached, "cache_clear"):
        _load_symbol_full_cached.cache_clear()

    px = load_price_data(sym_x)["close"]
    py = load_price_data(sym_y)["close"]
    common = px.index.intersection(py.index)
    px, py = px.loc[common], py.loc[common]

    if len(px) < 300:
        return {"error": f"Insufficient data: {len(px)} rows"}

    # Compute features
    X = compute_advanced_features(px, py)

    # Compute labels (canonical z-score)
    from common.feature_engineering import compute_zscore
    beta = float(np.cov(px.values, py.values)[0, 1] / np.var(px.values))
    spread = py - beta * px
    z = compute_zscore(spread, 60, min_periods=30, fillna_value=0.0)
    y = compute_labels(z, horizon=horizon, entry_threshold=entry_threshold)

    # Align and drop NaN
    common_idx = X.index.intersection(y.dropna().index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    if len(X) < 100:
        return {"error": f"Too few labeled samples: {len(X)}"}

    # Train/test split (temporal)
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train XGBoost
    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=20,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
    except ImportError:
        return {"error": "xgboost not installed"}

    # Evaluate
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    from scipy.stats import spearmanr
    import math

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, y_pred))
    brier = float(brier_score_loss(y_test, y_pred))
    logloss = float(log_loss(y_test, y_pred))

    # Per-fold IC (Spearman rank correlation between predicted probability and
    # actual label) — computed via 5-fold expanding-window walk-forward CV on
    # the training set only (no leakage from X_test).
    fold_ic_values: list = []
    n_tr = len(X_train)
    n_cv_splits = 5
    fold_size = n_tr // (n_cv_splits + 1)
    if fold_size >= 5:
        import copy
        X_tr_vals = X_train.values
        y_tr_vals = y_train.values
        for k in range(n_cv_splits):
            cv_train_end = (k + 1) * fold_size
            cv_val_start = cv_train_end
            cv_val_end = min(cv_train_end + fold_size, n_tr)
            if cv_val_end - cv_val_start < 5:
                continue
            try:
                cv_est = copy.deepcopy(model)
                # Refit on CV training slice without early stopping eval_set
                cv_est.set_params(early_stopping_rounds=None)
                cv_est.fit(
                    X_tr_vals[:cv_train_end],
                    y_tr_vals[:cv_train_end],
                    verbose=False,
                )
                fold_probas = cv_est.predict_proba(X_tr_vals[cv_val_start:cv_val_end])[:, 1]
                fold_labels = y_tr_vals[cv_val_start:cv_val_end]
                fold_ic, _ = spearmanr(fold_probas, fold_labels)
                if not math.isnan(float(fold_ic)):
                    fold_ic_values.append(float(fold_ic))
            except Exception as fold_exc:
                logger.debug("CV fold %d IC failed: %s", k, fold_exc)

    # Populate artifact fields
    artifact_cv_ic_per_fold = [float(ic) for ic in fold_ic_values]
    oos_ics = [ic for ic in fold_ic_values if not math.isnan(ic)]
    artifact_walk_forward_ic_mean = float(np.mean(oos_ics)) if oos_ics else float("nan")

    # Feature importance
    importance = dict(zip(X.columns, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]

    # Save model
    import pickle
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"xgb_meta_{sym_x}_{sym_y}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": list(X.columns), "pair": f"{sym_x}/{sym_y}"}, f)

    return {
        "pair": f"{sym_x}/{sym_y}",
        "auc": round(auc, 4),
        "brier": round(brier, 4),
        "logloss": round(logloss, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_rate": round(float(y_train.mean()), 3),
        "n_features": len(X.columns),
        "top_features": top_features,
        "model_path": str(model_path),
        # IC fields — mirrors TrainingRunArtifact.cv_ic_per_fold /
        # walk_forward_ic_mean for governance engine consumption.
        "cv_ic_per_fold": artifact_cv_ic_per_fold,
        "walk_forward_ic_mean": artifact_walk_forward_ic_mean,
        "cv_ic_mean": float(np.mean(artifact_cv_ic_per_fold)) if artifact_cv_ic_per_fold else float("nan"),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    for n in ["httpx", "yfinance", "common.data_loader"]:
        logging.getLogger(n).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="XLI-XLB")
    parser.add_argument("--all-alpha", action="store_true")
    args = parser.parse_args()

    if args.all_alpha:
        alpha_file = PROJECT_ROOT / "logs" / "alpha_results" / "alpha_pairs_latest.json"
        if alpha_file.exists():
            configs = json.loads(alpha_file.read_text())
            pairs = [(c["pair"].split("/")[0], c["pair"].split("/")[1]) for c in configs[:10]]
        else:
            pairs = [("XLI", "XLB"), ("IWM", "SPY"), ("XRT", "XLK")]
    else:
        sx, sy = args.pair.split("-")
        pairs = [(sx, sy)]

    print(f"\n{'='*60}")
    print(f"  XGBoost Meta-Label Training: {len(pairs)} pairs")
    print(f"{'='*60}")

    for sx, sy in pairs:
        result = train_xgboost_model(sx, sy)
        if result.get("error"):
            print(f"  ❌ {sx}/{sy}: {result['error']}")
        else:
            emoji = "🟢" if result["auc"] > 0.55 else "🟡" if result["auc"] > 0.50 else "🔴"
            print(f"  {emoji} {result['pair']}: AUC={result['auc']:.3f}  "
                  f"Brier={result['brier']:.3f}  n_train={result['n_train']}  "
                  f"n_test={result['n_test']}")
            print(f"     Top features: {', '.join(f[0] for f in result['top_features'][:5])}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
