# test_helpers.py
# NOTE: This file has known issues with function references.
# Tests reference functions that may not exist in common.helpers.
# Move to tests/ directory and fix imports before using.

import numpy as np
import pandas as pd
import pytest

from common.helpers import (
    read_json,
    write_json,
    flatten_dict,
    unflatten_dict,
    deep_merge,
)
from common.json_safe import make_json_safe, json_default as _json_default


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


def test_read_json_invalid(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("not a json")
    with pytest.raises(Exception):
        _ = read_json(path)
