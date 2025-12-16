# test_helpers.py

import numpy as np
import pandas as pd
import pytest

from common.helpers import (
from common.json_safe import make_json_safe, json_default as _json_default
    rolling_sharpe_ratio,
    garch_volatility,
    risk_metrics,
    gpu_accelerate,
    summarize_series,
    read_json,
    write_json,
    flatten_dict,
    unflatten_dict,
    deep_merge,
    load_dotenv,
)

def test_rolling_sharpe_ratio():
    data = np.random.normal(0, 0.01, size=100)
    result = rolling_sharpe_ratio(data, window=20)
    assert len(result) == len(data)
    assert np.isnan(result[:19]).all()
    assert np.isfinite(result[20:]).any()

def test_garch_volatility():
    data = np.random.normal(0, 0.01, size=100)
    try:
        result = garch_volatility(data)
        assert len(result) == len(data)
        assert np.isfinite(result[10:]).any()
    except ImportError:
        pytest.skip("arch package not installed")

def test_risk_metrics():
    data = np.random.normal(0, 0.01, size=100)
    res = risk_metrics(data)
    assert {'VaR', 'CVaR', 'max_drawdown', 'ulcer_index'} <= set(res.keys())
    for val in res.values():
        assert isinstance(val, (float, np.ndarray))

def test_gpu_accelerate_cpu():
    @gpu_accelerate
    def square(x):
        return x ** 2

    x = np.arange(1000)
    result = square(x)
    assert np.allclose(result, x ** 2)

def test_summarize_series():
    mats = pd.Series([np.eye(2), np.array([[1, 2], [3, 4]])])
    summary = summarize_series(mats, n_pca=1)
    assert isinstance(summary, dict)
    assert 'correlation' in summary

def test_flatten_unflatten():
    d = {'a': {'b': 1, 'c': {'d': 2}}}
    flat = flatten_dict(d)
    assert flat == {'a.b': 1, 'a.c.d': 2}
    unflat = unflatten_dict(flat)
    assert unflat == d

def test_deep_merge():
    d1 = {'a': {'b': 1}}
    d2 = {'a': {'c': 2}}
    merged = deep_merge(d1.copy(), d2)
    assert merged == {'a': {'b': 1, 'c': 2}}

def test_json_io(tmp_path):
    obj = {'a': 1, 'b': [1, 2, 3]}
    path = tmp_path / "test.json"
    write_json(obj, path)
    loaded = read_json(path)
    assert loaded == obj

def test_load_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_VAR=123\n")
    result = load_dotenv(env_file)
    assert result.get("TEST_VAR") == "123"

# Cross-file interaction (integration-level)
def test_summarize_with_metrics():
    mats = pd.Series([np.random.normal(size=(10, 2)) for _ in range(5)])
    summary = summarize_series(mats, n_pca=2)
    # Should include fields computed by matrix_helpers and advanced_metrics
    keys = ['correlation', 'covariance', 'pca', 'cointegration_engle', 'cointegration_johansen', 'mahalanobis']
    for key in keys:
        assert key in summary

import subprocess
import sys

def test_cli_summarize(tmp_path):
    # Create dummy npz file
    test_data = {str(i): np.eye(2) for i in range(3)}
    npz_path = tmp_path / "test_data.npz"
    np.savez_compressed(npz_path, **test_data)

    # Run the CLI summarize command
    result = subprocess.run([
        sys.executable,
        "-m", "common.helpers",  # assuming the CLI is defined in helpers.py as __main__
        "summarize",
        "--input", str(npz_path),
        "--n-pca", "1"
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert "correlation" in result.stdout or "covariance" in result.stdout

def test_cli_hash_file(tmp_path):
    file = tmp_path / "file.txt"
    file.write_text("hello world")

    result = subprocess.run([
        sys.executable,
        "-m", "common.helpers",
        "hash-file",
        str(file),
        "--algorithm", "sha256"
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert len(result.stdout.strip()) >= 64  # SHA256 hex length

def test_cli_humanize_bytes():
    result = subprocess.run([
        sys.executable,
        "-m", "common.helpers",
        "humanize-bytes",
        "2048"
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert "2.0 KB" in result.stdout or "2.0ג€¯KB" in result.stdout

def test_cli_load_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MY_TEST_VAR=456")

    result = subprocess.run([
        sys.executable,
        "-m", "common.helpers",
        "load-dotenv",
        "--path", str(env_file),
        "--override"
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert "MY_TEST_VAR" in result.stdout or "456" in result.stdout

def test_retry_decorator():
    from common.helpers import retry

    call_counter = {"count": 0}

    @retry(tries=3, delay=0.01)
    def sometimes_fail():
        call_counter["count"] += 1
        if call_counter["count"] < 2:
            raise ValueError("fail once")
        return "success"

    assert sometimes_fail() == "success"


def test_async_timeout():
    import asyncio
    from common.helpers import async_timeout

    async def slow():
        await asyncio.sleep(0.05)
        return "done"

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(async_timeout(slow(), timeout=0.01))

    result = asyncio.run(async_timeout(slow(), timeout=0.1))
    assert result == "done"

def test_end_to_end_simulation(tmp_path):
    from common.helpers import write_json, read_json, summarize_series, risk_metrics

    # Simulate full flow: Generate ג†’ Save ג†’ Load ג†’ Analyze
    matrices = pd.Series([np.random.normal(size=(50, 3)) for _ in range(5)])

    # Save raw matrices to JSON
    raw_data = {str(i): m.tolist() for i, m in enumerate(matrices)}
    data_path = tmp_path / "matrices.json"
    write_json(raw_data, data_path)

    # Load them back
    loaded_data = read_json(data_path)
    loaded_series = pd.Series([np.array(loaded_data[str(i)]) for i in range(len(loaded_data))])

    # Run summarize and risk metrics
    summary = summarize_series(loaded_series, n_pca=2)
    returns = np.random.normal(0, 0.02, size=100)
    risks = risk_metrics(returns)

    # Save results to JSON
    results_path = tmp_path / "results.json"
    write_json({"summary": summary, "risk": risks}, results_path)

    # Re-load and verify keys exist
    final = read_json(results_path)
    assert "summary" in final and "risk" in final
    assert "correlation" in final["summary"]
    assert "VaR" in final["risk"]

def test_read_json_invalid(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("not a json")
    with pytest.raises(Exception):
        _ = read_json(path)

def test_garch_import_fail(monkeypatch):
    monkeypatch.setitem(sys.modules, "arch", None)
    from common import helpers
    with pytest.raises(ImportError):
        helpers.garch_volatility(np.random.normal(0, 0.01, size=100))

def test_gpu_accelerate_no_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)

    @gpu_accelerate
    def identity(x):
        return x

    arr = np.arange(1000)
    result = identity(arr)
    assert np.all(result == arr)

def test_invalid_npz(tmp_path):
    path = tmp_path / "bad.npz"
    path.write_bytes(b"not npz format")
    with pytest.raises(Exception):
        np.load(path)

def test_summarize_empty_series():
    empty = pd.Series([], dtype=object)
    with pytest.raises(ValueError):
        summarize_series(empty)


# Run with: pytest test_helpers.py



