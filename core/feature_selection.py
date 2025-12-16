# core/feature_selection.py
"""
Feature Selection for Pairs Trading Optimization
Combines correlation-based filtering and SHAP importance fallback.
"""
import pandas as pd
import numpy as np

def select_features(opt_results: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Select optimal features/parameters using statistics and fallback SHAP importance.
    Args:
        opt_results: DataFrame, output of run_optimization (with all params/metrics).
        config: dict, pipeline config.
    Returns:
        DataFrame of selected optimal parameter sets/features.
    """
    if opt_results.empty or 'sharpe' not in opt_results.columns:
        return pd.DataFrame()  # fallback for empty results

    # 1. Correlation filter (remove features with high pairwise corr)
    param_cols = [c for c in opt_results.columns if c not in ['sharpe','return','drawdown','win_rate','trial_id']]
    corr = opt_results[param_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > 0.95)]
    selected = opt_results.drop(columns=drop_cols)

    # 2. Filter top-N by Sharpe
    top_n = config.get('feature_selection', {}).get('top_n', 20)
    selected = selected.sort_values('sharpe', ascending=False).head(top_n)

    # 3. (Optional) SHAP importance fallback ג€“ for now, just output sorted table
    # If you add SHAP, you can enrich this here.

    return selected.reset_index(drop=True)
