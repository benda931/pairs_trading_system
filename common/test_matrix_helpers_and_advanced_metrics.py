import pytest
import numpy as np
import pandas as pd

from common import matrix_helpers, advanced_metrics
from common.json_safe import make_json_safe, json_default as _json_default

# --- Dynamic GPU/CPU backend selection ---
# Attempt to use CuPy if available and GPU driver compatible;
# otherwise fall back to NumPy automatically.
try:
    import cupy as cp
    # quick test allocation
    _ = cp.zeros(1)
    matrix_helpers.xp = cp
    advanced_metrics.xp = cp
except Exception:
    import numpy as np
    matrix_helpers.xp = np
    advanced_metrics.xp = np  # ensure advanced metrics also use CPU backend

# For parallel execution, switch to threads to avoid pickling local functions
import concurrent.futures
matrix_helpers._PoolExecutor = concurrent.futures.ThreadPoolExecutor


# --- Tests for matrix_helpers.py ---

def test_ensure_matrix_series_valid():
    mats = [np.eye(2), [[1, 2], [3, 4]], pd.DataFrame([[5,6],[7,8]]).values]
    series = matrix_helpers.ensure_matrix_series(mats)
    assert isinstance(series, pd.Series)
    assert all(isinstance(m, np.ndarray) for m in series)
    assert series.iloc[0].shape == (2,2)


def test_ensure_matrix_series_invalid_shape():
    with pytest.raises(ValueError):
        matrix_helpers.ensure_matrix_series([np.eye(2), np.zeros((3,3))])


def test_apply_matrix_series_and_parallel():
    s = pd.Series([np.ones((2,2)), np.eye(2)])
    serial = matrix_helpers.apply_matrix_series(s, lambda m: m.sum())
    parallel = matrix_helpers.apply_matrix_series_parallel(s, lambda m: m.sum())
    assert serial.tolist() == [4,2]
    assert parallel.tolist() == [4,2]


def test_rolling_matrix():
    s = pd.Series([np.eye(2)*i for i in range(1,5)])
    out = matrix_helpers.rolling_matrix(s, window=2, fn=lambda x: x.sum(axis=2))
    assert np.isnan(out.iloc[0])
    expected = np.eye(2)*3
    assert np.allclose(out.iloc[1], expected)


def test_matrix_correlation_and_covariance():
    s = pd.Series([np.eye(2), np.ones((2,2))])
    corr = matrix_helpers.matrix_correlation(s)
    cov = matrix_helpers.matrix_covariance(s)
    assert np.allclose(corr.iloc[0], np.eye(2))
    assert np.allclose(cov.iloc[1], np.zeros((2,2)))


def test_matrix_pca_and_eigendecompose():
    mat = np.array([[2,0],[0,1]])
    w, v = matrix_helpers.matrix_eigendecompose(mat)
    assert set(np.round(w,3)) == set([1.0,2.0])
    s = pd.Series([mat, mat])
    df = matrix_helpers.matrix_pca(s, n_components=1)
    assert df.shape == (2,2)


def test_broadcast_and_persist_tensor_series(tmp_path):
    s = pd.Series([np.eye(2), np.eye(2)])
    b = matrix_helpers.broadcast_to_series(s, (2,3))
    assert b.iloc[0].shape == (2,3)
    path = tmp_path / 't.npz'
    matrix_helpers.save_tensor_series(s, str(path))
    loaded = matrix_helpers.load_tensor_series(str(path))
    assert np.allclose(loaded.iloc[1], np.eye(2))


def test_robust_cov_and_outliers():
    mat = np.vstack([np.ones(5), np.arange(5)])
    s = pd.Series([mat])
    rc = matrix_helpers.robust_cov(s)
    assert isinstance(rc.iloc[0], np.ndarray)
    mask = matrix_helpers.detect_outliers(s, z_thresh=0.1).iloc[0]
    assert mask.dtype == bool


def test_kernel_pca_and_pairwise_distances():
    mat = np.random.RandomState(0).rand(5,3)
    s = pd.Series([mat])
    kp = matrix_helpers.kernel_pca(s, n_components=2)
    assert kp.shape == (1, 6)
    pdist = matrix_helpers.pairwise_distances(s, metric='euclidean').iloc[0]
    assert pdist.shape == (5,5)

# --- Tests for advanced_metrics.py ---

@pytest.mark.skipif(not hasattr(advanced_metrics, 'rolling_covariance'), reason="advanced_metrics not available")
def test_rolling_covariance():
    s = pd.Series([np.eye(2)*i for i in range(1,5)])
    out = advanced_metrics.rolling_covariance(s, window=2)
    assert np.isnan(out.iloc[0])
    assert isinstance(out.iloc[1], np.ndarray)

@pytest.mark.skipif(not hasattr(advanced_metrics, 'dynamic_time_warping'), reason="advanced_metrics not available")
def test_dynamic_time_warping():
    mat = np.column_stack((np.arange(5), np.arange(5)[::-1]))
    s = pd.Series([mat])
    dist = advanced_metrics.dynamic_time_warping(s)
    assert isinstance(dist.iloc[0], float)

@pytest.mark.skipif(not hasattr(advanced_metrics, 'distance_correlation'), reason="advanced_metrics not available")
def test_distance_correlation():
    mat = np.column_stack((np.arange(5), np.arange(5)))
    s = pd.Series([mat])
    dc = advanced_metrics.distance_correlation(s)
    assert dc.iloc[0] == pytest.approx(1.0)

@pytest.mark.skipif(not hasattr(advanced_metrics, 'spectral_analysis'), reason="advanced_metrics not available")
def test_spectral_analysis():
    s = pd.Series([np.eye(3)])
    spec = advanced_metrics.spectral_analysis(s)
    assert isinstance(spec.iloc[0], np.ndarray)

@pytest.mark.skipif(not hasattr(advanced_metrics, 'mahalanobis_distance'), reason="advanced_metrics not available")
def test_mahalanobis_distance():
    mat = np.eye(3)
    s = pd.Series([mat])
    md = advanced_metrics.mahalanobis_distance(s)
    assert md.iloc[0].shape == (3,3)

@pytest.mark.skipif(not hasattr(advanced_metrics, 'rolling_spectral_analysis'), reason="advanced_metrics not available")
def test_rolling_spectral_analysis():
    s = pd.Series([np.eye(3), np.eye(3)])
    out = advanced_metrics.rolling_spectral_analysis(s, window=2)
    assert isinstance(out.iloc[1], np.ndarray)

@pytest.mark.skipif(not hasattr(advanced_metrics, 'rolling_mahalanobis_distance'), reason="advanced_metrics not available")
def test_rolling_mahalanobis_distance():
    s = pd.Series([np.eye(3), np.eye(3)])
    out = advanced_metrics.rolling_mahalanobis_distance(s, window=2)
    assert isinstance(out.iloc[1], np.ndarray)

@pytest.mark.skipif(not hasattr(advanced_metrics, 'rolling_distance_correlation'), reason="advanced_metrics not available")
def test_rolling_distance_correlation():
    s = pd.Series([np.column_stack((np.arange(5), np.arange(5))) for _ in range(2)])
    out = advanced_metrics.rolling_distance_correlation(s, window=2)
    assert isinstance(out.iloc[1], float)

# --- Tests for additional advanced_metrics functions ---

@pytest.mark.skipif(not hasattr(advanced_metrics, 'order_flow_imbalance'), reason="advanced_metrics missing order_flow_imbalance")
def test_order_flow_imbalance():
    import pandas as pd, numpy as np
    df_l1 = pd.DataFrame({'bid_size':[1,2,3,4],'ask_size':[4,3,2,1]})
    result = advanced_metrics.order_flow_imbalance(df_l1, window=2)
    assert isinstance(result, pd.Series)
    # check last value
    assert result.iloc[-1] == pytest.approx(((4-1)+(3-2))/((4+1)/2))

@pytest.mark.skipif(not hasattr(advanced_metrics, 'bid_ask_spread_pct'), reason="advanced_metrics missing bid_ask_spread_pct")
def test_bid_ask_spread_pct():
    import pandas as pd, numpy as np
    df_l1 = pd.DataFrame({'bid':[9,8],'ask':[11,12]})
    result = advanced_metrics.bid_ask_spread_pct(df_l1)
    assert np.allclose(result, [(11-9)/10, (12-8)/10])

@pytest.mark.skipif(not hasattr(advanced_metrics, 'amihud_illiq'), reason="advanced_metrics missing amihud_illiq")
def test_amihud_illiq():
    import pandas as pd
    close = pd.Series([100, 102, 104, 103])
    volume = pd.Series([10,10,10,10])
    result = advanced_metrics.amihud_illiq(close, volume, window=2)
    assert isinstance(result, pd.Series)
    assert result.iloc[1] == pytest.approx(abs(0.02)/10)

@pytest.mark.skipif(not hasattr(advanced_metrics, 'intraday_vol_ratio'), reason="advanced_metrics missing intraday_vol_ratio")
def test_intraday_vol_ratio():
    import pandas as pd
    min_ret = pd.Series([0.0, 0.1, -0.1, 0.2])
    daily_ret = pd.Series([0.0, 0.05, -0.05, 0.1])
    result = advanced_metrics.intraday_vol_ratio(min_ret, daily_ret, window=2)
    assert isinstance(result, pd.Series)
    assert len(result) == 4

@pytest.mark.skipif(not hasattr(advanced_metrics, 'autocorr_lag1'), reason="advanced_metrics missing autocorr_lag1")
def test_autocorr_lag1():
    import pandas as pd
    s = pd.Series([1,2,3,4,5,6])
    result = advanced_metrics.autocorr_lag1(s, window=3)
    assert isinstance(result, pd.Series)
    assert result.iloc[0:2].isna().all()

@pytest.mark.skipif(not hasattr(advanced_metrics, 'drawdown_half_life'), reason="advanced_metrics missing drawdown_half_life")
def test_drawdown_half_life():
    import pandas as pd
    s = pd.Series([1,0.5,0.75,1.0])
    result = advanced_metrics.drawdown_half_life(s)
    assert isinstance(result, pd.Series)
    assert result.iloc[-1] >= 1

@pytest.mark.skipif(not hasattr(advanced_metrics, 'beta_market_dynamic'), reason="advanced_metrics missing beta_market_dynamic")
def test_beta_market_dynamic():
    import pandas as pd
    asset = pd.Series([0,1,2,3,4])
    market = pd.Series([0,1,2,3,4])
    result = advanced_metrics.beta_market_dynamic(asset, market, window=3)
    assert (result.dropna() == 1).all()

@pytest.mark.skipif(not hasattr(advanced_metrics, 'news_sentiment_score'), reason="advanced_metrics missing news_sentiment_score")
def test_news_sentiment_score():
    import pandas as pd
    headlines = pd.Series(["good", "bad", None])
    result = advanced_metrics.news_sentiment_score(headlines)
    assert isinstance(result, pd.Series)
    assert -1.0 <= result.min() <= result.max() <= 1.0

@pytest.mark.skipif(not hasattr(advanced_metrics, 'macro_regime_classifier'), reason="advanced_metrics missing macro_regime_classifier")
def test_macro_regime_classifier():
    import pandas as pd
    df = pd.DataFrame({'gdp_nowcast':[1,-1], 'cpi_surprise':[0,1], 'unemployment':[4,6]})
    result = advanced_metrics.macro_regime_classifier(df)
    assert set(result.tolist()) <= {"growth","inflation","stagflation","recession","neutral"}

@pytest.mark.skipif(not hasattr(advanced_metrics, 'garch_volatility'), reason="advanced_metrics missing garch_volatility")
def test_garch_volatility_import():
    import pytest, sys
    sys.modules["arch"] = None
    with pytest.raises(ImportError):
        import common.helpers as h; h.garch_volatility([0.1,0.2,0.1])


