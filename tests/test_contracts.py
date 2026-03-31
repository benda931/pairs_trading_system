# -*- coding: utf-8 -*-
"""
tests/test_contracts.py — Tests for core/contracts.py
"""
from __future__ import annotations

import pytest
import numpy as np

from core.contracts import (
    HedgeRatioMethod,
    PairId,
    PairLifecycleState,
    SpreadDefinition,
    SpreadModel,
    ValidationResult,
)


class TestPairId:
    def test_canonical_order_already_sorted(self):
        pid = PairId("AAPL", "MSFT")
        assert pid.sym_x == "AAPL"
        assert pid.sym_y == "MSFT"

    def test_canonical_order_auto_sorted(self):
        pid = PairId("MSFT", "AAPL")
        assert pid.sym_x == "AAPL"
        assert pid.sym_y == "MSFT"

    def test_label(self):
        pid = PairId("AAPL", "MSFT")
        assert pid.label == "AAPL/MSFT"

    def test_equality_regardless_of_construction_order(self):
        p1 = PairId("AAPL", "MSFT")
        p2 = PairId("MSFT", "AAPL")
        assert p1 == p2

    def test_hashable(self):
        p1 = PairId("AAPL", "MSFT")
        p2 = PairId("MSFT", "AAPL")
        s = {p1, p2}
        assert len(s) == 1

    def test_from_label(self):
        pid = PairId.from_label("AAPL/MSFT")
        assert pid.sym_x == "AAPL"
        assert pid.sym_y == "MSFT"

    def test_from_label_auto_sorts(self):
        pid = PairId.from_label("MSFT/AAPL")
        assert pid.sym_x == "AAPL"

    def test_from_label_invalid(self):
        with pytest.raises(ValueError):
            PairId.from_label("AAPL")

    def test_contains(self):
        pid = PairId("AAPL", "MSFT")
        assert "AAPL" in pid
        assert "MSFT" in pid
        assert "GOOG" not in pid

    def test_as_tuple(self):
        pid = PairId("AAPL", "MSFT")
        assert pid.as_tuple() == ("AAPL", "MSFT")


class TestSpreadDefinition:
    def _make_defn(self, beta=0.8, intercept=0.1, mean=0.0, std=1.0):
        from datetime import datetime
        return SpreadDefinition(
            pair_id=PairId("AAPL", "MSFT"),
            model=SpreadModel.STATIC_OLS,
            hedge_ratio=beta,
            hedge_ratio_method=HedgeRatioMethod.OLS,
            intercept=intercept,
            mean=mean,
            std=std,
            estimated_at=datetime.utcnow(),
            train_start=datetime(2022, 1, 1),
            train_end=datetime(2023, 1, 1),
        )

    def test_compute_spread(self):
        import numpy as np
        import pandas as pd
        defn = self._make_defn(beta=1.0, intercept=0.0)
        dates = pd.bdate_range("2023-01-01", periods=3)
        lx = pd.Series([2.0, 3.0, 4.0], index=dates)
        ly = pd.Series([2.0, 3.0, 4.0], index=dates)
        spread = defn.compute_spread(lx, ly)
        np.testing.assert_array_almost_equal(spread.values, [0.0, 0.0, 0.0])

    def test_compute_zscore_uses_mean_std(self):
        import numpy as np
        import pandas as pd
        defn = self._make_defn(beta=1.0, intercept=0.0, mean=1.0, std=2.0)
        dates = pd.bdate_range("2023-01-01", periods=1)
        lx = pd.Series([3.0], index=dates)
        ly = pd.Series([2.0], index=dates)
        # raw spread = 3 - 1*2 - 0 = 1.0; z = (1.0 - 1.0) / 2.0 = 0.0
        z = defn.compute_zscore(lx, ly)
        np.testing.assert_array_almost_equal(z.values, [0.0])

    def test_constructor_enforces_std_floor(self):
        """Constructors (not the dataclass) enforce the std floor via max(std, 1e-8)."""
        from research.spread_constructor import StaticOLSConstructor
        import pandas as pd
        dates = pd.bdate_range("2021-01-01", periods=100)
        # All-same prices → zero spread std
        px = pd.Series([100.0] * 100, index=dates, name="X")
        py = pd.Series([100.0] * 100, index=dates, name="Y")
        ctor = StaticOLSConstructor()
        defn = ctor.fit(px, py, pair_id=PairId("X", "Y"))
        assert defn.std >= 1e-8


class TestEnums:
    def test_pair_lifecycle_states(self):
        states = [s.value for s in PairLifecycleState]
        assert "CANDIDATE" in states
        assert "VALIDATED" in states
        assert "ACTIVE" in states
        assert "REJECTED" in states
        assert "RETIRED" in states

    def test_validation_result_ordering(self):
        assert ValidationResult.PASS.value == "PASS"
        assert ValidationResult.FAIL.value == "FAIL"
        assert ValidationResult.WARN.value == "WARN"

    def test_spread_model_values(self):
        assert SpreadModel.STATIC_OLS.value == "STATIC_OLS"
        assert SpreadModel.ROLLING_OLS.value == "ROLLING_OLS"
        assert SpreadModel.KALMAN.value == "KALMAN"
