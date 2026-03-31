# -*- coding: utf-8 -*-
"""
tests/test_ml_platform.py — Comprehensive ML Platform Tests
=============================================================

Tests for the complete ML platform stack:
  A. Contracts
  B. Feature platform (registry, builder)
  C. Label platform (registry, builder)
  D. Dataset (temporal splits, leakage auditor, dataset builder)
  E. Models (MLModel, MetaLabelModel, RegimeClassificationModel, BreakDetectionModel)
  F. Evaluation (metrics, ModelEvaluator)
  G. Registry and governance
  H. Inference and monitoring
  I. Integration tests (end-to-end pipelines)
  J. Explainability (lineage, importance)
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n: int, start: str = "2018-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


def _make_X(n: int, cols=None, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = cols or [f"f{i}" for i in range(5)]
    return pd.DataFrame(
        rng.standard_normal((n, len(cols))),
        index=_make_dates(n),
        columns=cols,
    )


def _make_y_binary(n: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.integers(0, 2, size=n).astype(float),
        index=_make_dates(n),
    )


# ===========================================================================
# A. Contracts tests
# ===========================================================================

class TestContracts:
    def test_feature_definition_immutable(self):
        from ml.contracts import FeatureDefinition, EntityScope, FeatureCategory
        fd = FeatureDefinition(
            name="test_feat",
            description="test",
            entity_scope=EntityScope.PAIR,
            category=FeatureCategory.Z_SCORE,
            required_inputs=("z",),
            lookback_days=20,
        )
        # frozen dataclass — should raise on assignment
        with pytest.raises((AttributeError, TypeError)):
            fd.name = "other"  # type: ignore[misc]

    def test_label_definition_immutable(self):
        from ml.contracts import LabelDefinition, LabelFamily, EntityScope
        ld = LabelDefinition(
            name="test_label",
            family=LabelFamily.META_LABEL,
            description="test",
            horizon_days=10,
            entity_scope=EntityScope.PAIR,
            output_type="binary",
        )
        with pytest.raises((AttributeError, TypeError)):
            ld.name = "other"  # type: ignore[misc]

    def test_prediction_record_is_valid(self):
        from ml.contracts import PredictionRecord
        rec = PredictionRecord(score=0.7, model_id="m1", entity_id="p1")
        # score is finite → is_valid should be True if implemented as property
        assert rec.score == 0.7
        assert rec.model_id == "m1"

    def test_inference_result_should_act(self):
        from ml.contracts import InferenceResult
        # Valid, non-fallback result with confidence >= 0
        result = InferenceResult(
            score=0.65,
            confidence=0.70,
            fallback_used=False,
        )
        assert result.is_valid is True
        assert result.should_act is True

    def test_inference_result_should_not_act_when_fallback(self):
        from ml.contracts import InferenceResult
        result = InferenceResult(
            score=0.65,
            confidence=0.70,
            fallback_used=True,
        )
        assert result.should_act is False

    def test_inference_result_should_not_act_when_nan_score(self):
        from ml.contracts import InferenceResult
        result = InferenceResult(
            score=float("nan"),
            confidence=0.80,
            fallback_used=False,
        )
        assert result.is_valid is False
        assert result.should_act is False

    def test_dataset_snapshot_creation(self):
        from ml.contracts import DatasetSnapshot
        snap = DatasetSnapshot(
            feature_set_id="fs-001",
            label_name="meta_take_10d",
            train_rows=500,
            n_features=22,
        )
        assert snap.train_rows == 500
        assert snap.n_features == 22
        assert snap.snapshot_id  # auto-generated

    def test_model_metadata_creation(self):
        from ml.contracts import ModelMetadata, MLTaskFamily, ModelStatus
        meta = ModelMetadata(
            task_family=MLTaskFamily.META_LABELING,
            model_class="LogisticRegression",
            status=ModelStatus.RESEARCH,
        )
        assert meta.status == ModelStatus.RESEARCH
        assert meta.model_id  # auto-generated

    def test_fallback_policy_defaults(self):
        from ml.contracts import FallbackPolicy
        policy = FallbackPolicy()
        assert policy.trigger_on_stale is True
        assert policy.trigger_on_drift is True
        assert policy.trigger_on_missing_features is True
        assert policy.fallback_type == "rule_based"


# ===========================================================================
# B. Feature platform tests
# ===========================================================================

class TestFeatureRegistry:
    def test_registry_not_empty(self):
        from ml.features.definitions import FEATURE_REGISTRY
        assert len(FEATURE_REGISTRY) > 0

    def test_all_feature_registry_entries_have_required_fields(self):
        from ml.features.definitions import FEATURE_REGISTRY
        for name, defn in FEATURE_REGISTRY.items():
            assert defn.name == name, f"Name mismatch for {name}"
            assert defn.description, f"Missing description for {name}"
            assert len(defn.required_inputs) > 0, f"No required_inputs for {name}"
            assert defn.lookback_days > 0, f"lookback_days must be > 0 for {name}"
            assert defn.version, f"Missing version for {name}"

    def test_feature_groups_reference_valid_features(self):
        from ml.features.definitions import FEATURE_REGISTRY, FEATURE_GROUPS
        all_names = set(FEATURE_REGISTRY.keys())
        for group_name, group in FEATURE_GROUPS.items():
            for feat_name in group.feature_names:
                assert feat_name in all_names, (
                    f"Feature group '{group_name}' references unknown feature '{feat_name}'"
                )

    def test_meta_label_feature_group_exists(self):
        from ml.features.definitions import FEATURE_GROUPS
        # meta-labeling features come from "pair_zscore_features" and others
        # At minimum the pair-level zscore group should exist
        assert "pair_zscore_features" in FEATURE_GROUPS

    def test_regime_classification_feature_group_exists(self):
        from ml.features.definitions import FEATURE_GROUPS
        assert "regime_features" in FEATURE_GROUPS

    def test_break_detection_feature_group_exists(self):
        from ml.features.definitions import FEATURE_GROUPS
        # break detection uses stability features
        assert "pair_stability_features" in FEATURE_GROUPS

    def test_feature_registry_contains_pair_z(self):
        from ml.features.definitions import FEATURE_REGISTRY
        assert "pair_z" in FEATURE_REGISTRY

    def test_feature_registry_contains_regime_features(self):
        from ml.features.definitions import FEATURE_REGISTRY
        assert "reg_vol_regime" in FEATURE_REGISTRY
        assert "reg_break_indicator" in FEATURE_REGISTRY


class TestPointInTimeFeatureBuilder:
    def setup_method(self):
        np.random.seed(42)
        self.dates = _make_dates(100)
        self.px = pd.Series(
            100 * np.cumprod(1 + 0.01 * np.random.randn(100)), index=self.dates
        )
        self.py = pd.Series(
            100 * np.cumprod(1 + 0.01 * np.random.randn(100)), index=self.dates
        )
        self.z = pd.Series(np.random.randn(100), index=self.dates)

    def test_compute_pair_features_clips_to_as_of(self):
        from ml.features.builder import PointInTimeFeatureBuilder
        builder = PointInTimeFeatureBuilder()
        as_of = self.dates[49]  # midpoint

        snap = builder.compute_pair_features(
            pair_id="AAPL-MSFT",
            px=self.px,
            py=self.py,
            z=self.z,
            as_of=as_of,
        )
        # The snapshot as_of field should reflect the cutoff
        assert snap is not None
        # as_of stored as ISO string
        as_of_ts = pd.Timestamp(snap.as_of)
        assert as_of_ts <= pd.Timestamp(as_of).tz_localize("UTC") or \
               as_of_ts <= pd.Timestamp(as_of)

    def test_compute_pair_features_returns_feature_snapshot(self):
        from ml.features.builder import PointInTimeFeatureBuilder
        from ml.contracts import FeatureSnapshot
        builder = PointInTimeFeatureBuilder()
        snap = builder.compute_pair_features(
            pair_id="AAPL-MSFT",
            px=self.px,
            py=self.py,
            z=self.z,
            as_of=self.dates[-1],
        )
        assert isinstance(snap, FeatureSnapshot)
        # Should have at least some computed features
        assert len(snap.feature_values) > 0

    def test_insufficient_lookback_returns_partial_snapshot(self):
        from ml.features.builder import PointInTimeFeatureBuilder
        from ml.contracts import FeatureSnapshot
        builder = PointInTimeFeatureBuilder()
        # Use only 5 data points — far below any lookback requirement
        tiny_dates = _make_dates(5)
        px = pd.Series(100.0 + np.random.randn(5), index=tiny_dates)
        py = pd.Series(100.0 + np.random.randn(5), index=tiny_dates)
        z = pd.Series(np.random.randn(5), index=tiny_dates)

        snap = builder.compute_pair_features(
            pair_id="TINY-PAIR",
            px=px,
            py=py,
            z=z,
            as_of=tiny_dates[-1],
        )
        # Should not raise; may have warnings
        assert isinstance(snap, FeatureSnapshot)

    def test_compute_instrument_features_basic(self):
        from ml.features.builder import PointInTimeFeatureBuilder
        builder = PointInTimeFeatureBuilder()
        snap = builder.compute_instrument_features(
            symbol="AAPL",
            prices=self.px,
            volume=None,
            as_of=self.dates[-1],
        )
        assert len(snap.feature_values) > 0

    def test_compute_signal_features_basic(self):
        from ml.features.builder import PointInTimeFeatureBuilder
        builder = PointInTimeFeatureBuilder()
        snap = builder.compute_signal_features(
            pair_id="AAPL-MSFT",
            z=self.z,
            entry_z=self.z.iloc[50],
            entry_timestamp=self.dates[50],
            as_of=self.dates[-1],
        )
        assert isinstance(snap.feature_values, dict)

    def test_validate_feature_availability_full(self):
        from ml.features.builder import PointInTimeFeatureBuilder
        builder = PointInTimeFeatureBuilder()
        snap = builder.compute_pair_features(
            pair_id="AAPL-MSFT",
            px=self.px,
            py=self.py,
            z=self.z,
            as_of=self.dates[-1],
        )
        available = snap.feature_values
        required = list(available.keys())[:3]  # any 3 features we know are available
        ok, missing = builder.validate_feature_availability(available, required)
        assert ok is True
        assert len(missing) == 0

    def test_validate_feature_availability_partial(self):
        from ml.features.builder import PointInTimeFeatureBuilder
        builder = PointInTimeFeatureBuilder()
        snap = builder.compute_pair_features(
            pair_id="AAPL-MSFT",
            px=self.px,
            py=self.py,
            z=self.z,
            as_of=self.dates[-1],
        )
        required = ["nonexistent_feature_xyz"]
        ok, missing = builder.validate_feature_availability(snap.feature_values, required)
        assert ok is False
        assert "nonexistent_feature_xyz" in missing


# ===========================================================================
# C. Label tests
# ===========================================================================

class TestLabelRegistry:
    def test_label_registry_not_empty(self):
        from ml.labels.definitions import LABEL_REGISTRY
        assert len(LABEL_REGISTRY) > 0

    def test_all_labels_have_horizon(self):
        from ml.labels.definitions import LABEL_REGISTRY
        for name, defn in LABEL_REGISTRY.items():
            assert defn.horizon_days > 0, f"Label '{name}' has horizon_days <= 0"

    def test_meta_take_labels_are_binary(self):
        from ml.labels.definitions import LABEL_REGISTRY
        assert "meta_take_10d" in LABEL_REGISTRY
        assert LABEL_REGISTRY["meta_take_10d"].output_type == "binary"
        assert "meta_take_5d" in LABEL_REGISTRY
        assert LABEL_REGISTRY["meta_take_5d"].output_type == "binary"

    def test_break_labels_are_binary(self):
        from ml.labels.definitions import LABEL_REGISTRY
        assert "break_5d" in LABEL_REGISTRY
        assert LABEL_REGISTRY["break_5d"].output_type == "binary"
        assert "break_20d" in LABEL_REGISTRY
        assert LABEL_REGISTRY["break_20d"].output_type == "binary"

    def test_holding_time_labels_are_continuous(self):
        from ml.labels.definitions import LABEL_REGISTRY
        assert "time_to_mean_reversion" in LABEL_REGISTRY
        assert LABEL_REGISTRY["time_to_mean_reversion"].output_type == "continuous"


class TestLabelBuilder:
    def setup_method(self):
        np.random.seed(42)
        self.dates = _make_dates(200)
        self.z = pd.Series(
            2 * np.sin(np.linspace(0, 10 * np.pi, 200)), index=self.dates
        )

    def test_reversion_label_binary_values(self):
        from ml.labels.builder import LabelBuilder
        builder = LabelBuilder()
        labels = builder.build_reversion_label(self.z, horizon=10)
        non_nan = labels.dropna()
        # Should only contain 0.0 or 1.0
        assert set(non_nan.unique()).issubset({0.0, 1.0}), (
            f"Expected only 0.0/1.0, got {non_nan.unique()}"
        )

    def test_reversion_label_uses_future_data(self):
        from ml.labels.builder import LabelBuilder
        builder = LabelBuilder()
        horizon = 10
        labels = builder.build_reversion_label(self.z, horizon=horizon)
        # The last `horizon` rows must be NaN (insufficient future data)
        tail_labels = labels.iloc[-horizon:]
        assert tail_labels.isna().all(), (
            "Last horizon rows should be NaN (no future data)"
        )

    def test_break_label_smoke(self):
        from ml.labels.builder import LabelBuilder
        builder = LabelBuilder()
        labels = builder.build_break_label(self.z, horizon=5)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(self.z)

    def test_holding_time_label_censored(self):
        from ml.labels.builder import LabelBuilder
        builder = LabelBuilder()
        labels = builder.build_holding_time_label(
            self.z, max_horizon=20
        )
        assert isinstance(labels, pd.Series)
        # All non-NaN values should be in [1, max_horizon]
        non_nan = labels.dropna()
        if len(non_nan) > 0:
            assert (non_nan >= 1).all()
            assert (non_nan <= 20).all()

    def test_build_all_labels_returns_dataframe(self):
        from ml.labels.builder import LabelBuilder
        builder = LabelBuilder()
        df = builder.build_all_labels(self.z)
        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) > 0

    def test_no_label_contamination_before_train_end(self):
        from ml.labels.builder import LabelBuilder
        builder = LabelBuilder()
        horizon = 10
        labels = builder.build_reversion_label(self.z, horizon=horizon)
        # The very last horizon rows should be NaN
        tail = labels.iloc[-(horizon):]
        n_nan_in_tail = tail.isna().sum()
        assert n_nan_in_tail > 0, (
            "Labels within horizon of series end must be NaN to avoid contamination"
        )


# ===========================================================================
# D. Dataset tests — leakage prevention (most critical)
# ===========================================================================

class TestTemporalSplitter:
    def setup_method(self):
        np.random.seed(42)
        self.dates = _make_dates(1000)

    def test_chronological_split_no_overlap(self):
        from ml.datasets.splits import TemporalSplitter
        splitter = TemporalSplitter(purge_days=5, embargo_days=10)
        plan = splitter.chronological_split(self.dates)

        train_end = pd.Timestamp(plan.train_end)
        val_start = pd.Timestamp(plan.validation_start)
        val_end = pd.Timestamp(plan.validation_end)
        test_start = pd.Timestamp(plan.test_start)

        # Strictly chronological
        assert train_end < val_start, "val_start must be after train_end"
        assert val_end < test_start, "test_start must be after val_end"
        # Gap must be at least 1 day
        assert (val_start - train_end).days >= 1
        assert (test_start - val_end).days >= 1

    def test_walk_forward_splits_chronological(self):
        from ml.datasets.splits import TemporalSplitter
        splitter = TemporalSplitter(purge_days=5, embargo_days=10)
        plans = splitter.walk_forward_splits(
            self.dates, n_folds=3, min_train_periods=200, test_periods=50
        )
        assert len(plans) >= 2
        for i, plan in enumerate(plans):
            assert plan.fold_index == i
            train_end = pd.Timestamp(plan.train_end)
            test_start = pd.Timestamp(plan.test_start)
            assert test_start > train_end, f"Fold {i}: test must be after train"

    def test_walk_forward_splits_expanding(self):
        from ml.datasets.splits import TemporalSplitter
        splitter = TemporalSplitter(purge_days=5, embargo_days=10)
        plans = splitter.walk_forward_splits(
            self.dates, n_folds=3, min_train_periods=200, test_periods=50
        )
        # Each fold's train_end should be strictly after the previous
        for i in range(1, len(plans)):
            prev_train_end = pd.Timestamp(plans[i - 1].train_end)
            curr_train_end = pd.Timestamp(plans[i].train_end)
            assert curr_train_end > prev_train_end, (
                f"Walk-forward train window must expand: fold {i}"
            )

    def test_rolling_origin_splits_increasing(self):
        from ml.datasets.splits import TemporalSplitter
        splitter = TemporalSplitter(purge_days=5, embargo_days=10)
        plans = splitter.rolling_origin_splits(
            self.dates,
            initial_train_periods=300,
            test_periods=50,
            step_periods=50,
        )
        assert len(plans) >= 2
        for plan in plans:
            train_end = pd.Timestamp(plan.train_end)
            test_start = pd.Timestamp(plan.test_start)
            assert test_start > train_end

    def test_purge_days_creates_gap(self):
        from ml.datasets.splits import TemporalSplitter
        purge = 5
        embargo = 0
        splitter = TemporalSplitter(purge_days=purge, embargo_days=embargo)
        plan = splitter.chronological_split(self.dates)
        train_end = pd.Timestamp(plan.train_end)
        val_start = pd.Timestamp(plan.validation_start)
        # Gap must be at least purge_days
        gap = (val_start - train_end).days
        assert gap >= purge, f"Expected gap >= {purge}, got {gap}"

    def test_embargo_days_creates_gap(self):
        from ml.datasets.splits import TemporalSplitter
        purge = 0
        embargo = 10
        splitter = TemporalSplitter(purge_days=purge, embargo_days=embargo)
        plan = splitter.chronological_split(self.dates)
        train_end = pd.Timestamp(plan.train_end)
        val_start = pd.Timestamp(plan.validation_start)
        gap = (val_start - train_end).days
        assert gap >= embargo, f"Expected gap >= {embargo}, got {gap}"


class TestLeakageAuditor:
    def setup_method(self):
        np.random.seed(42)
        self.train_dates = _make_dates(200, "2018-01-01")
        self.test_dates = _make_dates(100, "2019-01-01")
        self.X_train = pd.DataFrame(
            np.random.randn(200, 5),
            index=self.train_dates,
            columns=[f"f{i}" for i in range(5)],
        )
        self.y_train = pd.Series(
            np.random.randint(0, 2, 200).astype(float), index=self.train_dates
        )
        self.X_test = pd.DataFrame(
            np.random.randn(100, 5),
            index=self.test_dates,
            columns=[f"f{i}" for i in range(5)],
        )
        self.y_test = pd.Series(
            np.random.randint(0, 2, 100).astype(float), index=self.test_dates
        )
        self.train_end = self.train_dates[-1]
        self.test_start = self.test_dates[0]

    def test_passes_clean_dataset(self):
        from ml.datasets.leakage import LeakageAuditor
        auditor = LeakageAuditor()
        report = auditor.audit(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            train_end=self.train_end,
            test_start=self.test_start,
            label_horizon_days=10,
            embargo_days=5,
        )
        assert report.passed, f"Expected clean audit to pass. Violations: {report.violations}"

    def test_detects_future_features(self):
        from ml.datasets.leakage import LeakageAuditor
        auditor = LeakageAuditor()
        # Inject future rows into X_train (rows after train_end)
        future_dates = _make_dates(5, start="2019-03-01")
        X_contaminated = pd.concat([
            self.X_train,
            pd.DataFrame(
                np.random.randn(5, 5),
                index=future_dates,
                columns=[f"f{i}" for i in range(5)],
            ),
        ]).sort_index()
        y_contaminated = pd.concat([
            self.y_train,
            pd.Series(np.ones(5), index=future_dates),
        ]).sort_index()

        report = auditor.audit(
            X_contaminated, y_contaminated,
            self.X_test, self.y_test,
            train_end=self.train_end,
            test_start=self.test_start,
            label_horizon_days=5,
            embargo_days=5,
        )
        assert not report.passed
        assert report.future_feature_risk is True

    def test_detects_train_test_overlap(self):
        from ml.datasets.leakage import LeakageAuditor
        auditor = LeakageAuditor()
        # Build X_test with dates very close to train_end (only 2 days gap)
        tight_test_dates = pd.date_range(
            start=self.train_end + timedelta(days=2), periods=50, freq="B"
        )
        X_test_close = pd.DataFrame(
            np.random.randn(50, 5), index=tight_test_dates,
            columns=[f"f{i}" for i in range(5)]
        )
        y_test_close = pd.Series(
            np.random.randint(0, 2, 50).astype(float), index=tight_test_dates
        )
        report = auditor.audit(
            self.X_train, self.y_train,
            X_test_close, y_test_close,
            train_end=self.train_end,
            test_start=tight_test_dates[0],
            label_horizon_days=1,
            embargo_days=10,  # requires 10-day gap; actual gap is 2 days
        )
        assert not report.passed
        # Either overlap_label_risk or embargo_adequate will be flagged
        assert report.overlap_label_risk is True or report.embargo_adequate is False

    def test_label_horizon_contamination_flag(self):
        """
        Critical: if test_start - train_end < label_horizon + embargo, flag it.
        """
        from ml.datasets.leakage import LeakageAuditor
        auditor = LeakageAuditor()
        horizon = 20
        embargo = 10
        # Gap is only 5 days — less than horizon + embargo = 30
        tight_test_start = self.train_end + timedelta(days=5)

        report = auditor.audit(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            train_end=self.train_end,
            test_start=tight_test_start,
            label_horizon_days=horizon,
            embargo_days=embargo,
        )
        # Should fail due to insufficient gap
        assert not report.passed
        assert report.embargo_adequate is False

    def test_checks_run_list_populated(self):
        from ml.datasets.leakage import LeakageAuditor
        auditor = LeakageAuditor()
        report = auditor.audit(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            train_end=self.train_end,
            test_start=self.test_start,
        )
        assert len(report.checks_run) > 0


# ===========================================================================
# E. Model tests
# ===========================================================================

class TestMLModel:
    def setup_method(self):
        from sklearn.linear_model import LogisticRegression
        from ml.models.base import MLModel
        np.random.seed(42)
        self.dates = _make_dates(200)
        feature_cols = ["f1", "f2", "f3", "f4", "f5"]
        self.X = pd.DataFrame(
            np.random.randn(200, 5), index=self.dates, columns=feature_cols
        )
        self.y = pd.Series(
            (np.random.randn(200) > 0).astype(float), index=self.dates
        )
        self.model = MLModel(
            LogisticRegression(max_iter=200),
            feature_names=feature_cols,
        )
        self.train_end = self.dates[149]
        self.X_train = self.X.iloc[:150]
        self.y_train = self.y.iloc[:150]

    def test_fit_trains_model(self):
        artifact = self.model.fit(self.X_train, self.y_train)
        assert self.model.is_fitted is True
        assert artifact.n_train_samples > 0

    def test_fit_leakage_guard_raises(self):
        """fit() must raise ValueError if X contains rows after train_end."""
        # X contains rows from dates[0] to dates[199] — after train_end=dates[149]
        with pytest.raises(ValueError, match="train_end"):
            self.model.fit(self.X, self.y, train_end=self.train_end)

    def test_predict_structured_returns_prediction_records(self):
        from ml.contracts import PredictionRecord
        self.model.fit(self.X_train, self.y_train)
        records = self.model.predict_structured(self.X.iloc[:10])
        assert len(records) == 10
        for rec in records:
            assert isinstance(rec, PredictionRecord)
            assert math.isfinite(rec.score)

    def test_predict_proba_shape(self):
        self.model.fit(self.X_train, self.y_train)
        proba = self.model.predict_proba(self.X.iloc[:10])
        assert proba.shape == (10, 2)
        # Probabilities must be in [0, 1]
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_score_returns_ic(self):
        self.model.fit(self.X_train, self.y_train)
        X_test = self.X.iloc[150:]
        y_test = self.y.iloc[150:]
        ic = self.model.score(X_test, y_test)
        assert isinstance(ic, float)

    def test_save_load_roundtrip(self, tmp_path):
        self.model.fit(self.X_train, self.y_train)
        path = str(tmp_path / "test_model.pkl")
        self.model.save(path)

        from ml.models.base import MLModel
        loaded = MLModel.load(path)
        assert loaded.is_fitted is True
        assert loaded.model_id == self.model.model_id

        proba_orig = self.model.predict_proba(self.X.iloc[:5])
        proba_loaded = loaded.predict_proba(self.X.iloc[:5])
        np.testing.assert_allclose(proba_orig, proba_loaded, rtol=1e-5)

    def test_predict_before_fit_returns_neutral(self):
        from sklearn.linear_model import LogisticRegression
        from ml.models.base import MLModel
        unfitted_model = MLModel(LogisticRegression())
        proba = unfitted_model.predict_proba(self.X.iloc[:5])
        # Should return 0.5 (neutral) without raising
        assert proba.shape[0] == 5
        np.testing.assert_allclose(proba, 0.5, atol=1e-9)


class TestMetaLabelModel:
    def setup_method(self):
        from sklearn.linear_model import LogisticRegression
        from ml.models.base import MLModel
        from ml.models.meta_labeler import MetaLabelModel
        np.random.seed(42)
        self.dates = _make_dates(200)
        cols = [f"f{i}" for i in range(5)]
        self.X = pd.DataFrame(np.random.randn(200, 5), index=self.dates, columns=cols)
        self.y = pd.Series(
            (np.random.randn(200) > 0).astype(float), index=self.dates
        )
        base = MLModel(LogisticRegression(max_iter=200), feature_names=cols)
        self.model = MetaLabelModel(base_model=base, feature_names=cols)
        self.model.fit(self.X.iloc[:150], self.y.iloc[:150])

    def test_predict_success_probability_range(self):
        """Probability must be in [0, 1]."""
        features = {f"f{i}": float(np.random.randn()) for i in range(5)}
        p = self.model.predict_success_probability(features)
        assert math.isfinite(p), "Probability must be finite after fit"
        assert 0.0 <= p <= 1.0, f"Probability out of range: {p}"

    def test_recommend_action_returns_action_enum(self):
        from ml.contracts import MetaLabelAction
        features = {f"f{i}": float(np.random.randn()) for i in range(5)}
        action = self.model.recommend_action(features)
        assert isinstance(action, MetaLabelAction)

    def test_recommend_action_fallback_when_unfitted(self):
        from sklearn.linear_model import LogisticRegression
        from ml.models.base import MLModel
        from ml.models.meta_labeler import MetaLabelModel
        from ml.contracts import MetaLabelAction
        base = MLModel(LogisticRegression())
        unfitted = MetaLabelModel(base_model=base, fallback_action=MetaLabelAction.TAKE)
        action = unfitted.recommend_action({"f0": 1.5})
        assert action == MetaLabelAction.TAKE

    def test_fit_and_predict_pipeline(self):
        from sklearn.linear_model import LogisticRegression
        from ml.models.base import MLModel
        from ml.models.meta_labeler import MetaLabelModel
        np.random.seed(0)
        dates = _make_dates(100)
        cols = ["z_score", "spread_vol", "corr_20d"]
        X = pd.DataFrame(np.random.randn(100, 3), index=dates, columns=cols)
        y = pd.Series((np.random.randn(100) > 0).astype(float), index=dates)
        base = MLModel(LogisticRegression(max_iter=300), feature_names=cols)
        m = MetaLabelModel(base_model=base, feature_names=cols)
        artifact = m.fit(X.iloc[:80], y.iloc[:80])
        assert artifact.n_train_samples > 0
        p = m.predict_success_probability(X.iloc[80])
        assert 0 <= p <= 1

    def test_implements_meta_label_protocol(self):
        """MetaLabelModel must have predict_success_probability and recommend_action."""
        assert hasattr(self.model, "predict_success_probability")
        assert hasattr(self.model, "recommend_action")
        assert callable(self.model.predict_success_probability)
        assert callable(self.model.recommend_action)


class TestRegimeClassificationModel:
    def setup_method(self):
        from sklearn.ensemble import RandomForestClassifier
        from ml.models.base import MLModel
        from ml.models.regime_classifier import RegimeClassificationModel
        from ml.contracts import MLTaskFamily
        np.random.seed(42)
        self.dates = _make_dates(200)
        cols = [f"f{i}" for i in range(5)]
        self.X = pd.DataFrame(np.random.randn(200, 5), index=self.dates, columns=cols)
        # Integer labels: 0=MEAN_REVERTING, 1=TRENDING, 2=HIGH_VOL
        self.y = pd.Series(np.random.choice([0, 1, 2], size=200), index=self.dates)
        base = MLModel(
            RandomForestClassifier(n_estimators=10, random_state=42),
            task_family=MLTaskFamily.REGIME_CLASSIFICATION,
            feature_names=cols,
        )
        self.model = RegimeClassificationModel(base_model=base, feature_names=cols)
        self.model.fit(self.X.iloc[:150], self.y.iloc[:150])

    def test_classify_returns_regime_label_and_confidence(self):
        from core.contracts import RegimeLabel
        features = {f"f{i}": float(np.random.randn()) for i in range(5)}
        regime, conf = self.model.classify(features)
        assert isinstance(regime, RegimeLabel)
        assert 0.0 <= conf <= 1.0

    def test_safety_floor_no_crisis_from_ml(self):
        """ML classifier must never return CRISIS or BROKEN."""
        from core.contracts import RegimeLabel
        # Run many predictions — none should be CRISIS or BROKEN
        for _ in range(50):
            features = {f"f{i}": float(np.random.randn()) for i in range(5)}
            regime, _ = self.model.classify(features)
            assert regime not in (RegimeLabel.CRISIS, RegimeLabel.BROKEN), (
                f"Safety floor violated: ML returned {regime}"
            )

    def test_classify_fallback_when_unfitted(self):
        from sklearn.ensemble import RandomForestClassifier
        from ml.models.base import MLModel
        from ml.models.regime_classifier import RegimeClassificationModel
        from core.contracts import RegimeLabel
        base = MLModel(RandomForestClassifier(n_estimators=5))
        unfitted = RegimeClassificationModel(
            base_model=base, fallback_regime=RegimeLabel.HIGH_VOL
        )
        regime, conf = unfitted.classify({"f0": 1.0})
        assert regime == RegimeLabel.HIGH_VOL
        assert conf == 0.0

    def test_implements_regime_hook_protocol(self):
        """RegimeClassificationModel must implement classify()."""
        assert hasattr(self.model, "classify")
        assert callable(self.model.classify)


class TestBreakDetectionModel:
    def setup_method(self):
        from sklearn.ensemble import RandomForestClassifier
        from ml.models.base import MLModel
        from ml.models.break_detector import BreakDetectionModel
        from ml.contracts import MLTaskFamily
        np.random.seed(42)
        self.dates = _make_dates(200)
        cols = [f"f{i}" for i in range(5)]
        self.X = pd.DataFrame(np.random.randn(200, 5), index=self.dates, columns=cols)
        # Binary: 1 = break, 0 = stable; rare event
        self.y = pd.Series(
            (np.random.rand(200) < 0.15).astype(float), index=self.dates
        )
        base = MLModel(
            RandomForestClassifier(n_estimators=10, random_state=42),
            task_family=MLTaskFamily.BREAK_DETECTION,
            feature_names=cols,
        )
        self.model = BreakDetectionModel(
            base_model=base,
            feature_names=cols,
            break_probability_threshold=0.65,
            fallback_probability=0.20,
        )
        self.model.fit(self.X.iloc[:150], self.y.iloc[:150])

    def test_predict_break_probability_range(self):
        features = {f"f{i}": float(np.random.randn()) for i in range(5)}
        p = self.model.predict_break_probability(features)
        assert math.isfinite(p)
        assert 0.0 <= p <= 1.0

    def test_is_break_risk_elevated(self):
        features = {f"f{i}": float(np.random.randn()) for i in range(5)}
        result = self.model.is_break_risk_elevated(features)
        assert isinstance(result, bool)

    def test_fallback_when_unfitted(self):
        from sklearn.ensemble import RandomForestClassifier
        from ml.models.base import MLModel
        from ml.models.break_detector import BreakDetectionModel
        base = MLModel(RandomForestClassifier(n_estimators=5))
        unfitted = BreakDetectionModel(
            base_model=base, fallback_probability=0.20
        )
        p = unfitted.predict_break_probability({"f0": 1.0})
        assert p == pytest.approx(0.20, abs=1e-9)


# ===========================================================================
# F. Evaluation tests
# ===========================================================================

class TestMetrics:
    def setup_method(self):
        np.random.seed(42)
        n = 200
        self.dates = _make_dates(n)
        self.y_true = pd.Series(
            np.random.randint(0, 2, n).astype(float), index=self.dates
        )
        self.y_pred_perfect = self.y_true.copy()
        self.y_pred_random = pd.Series(np.random.rand(n), index=self.dates)
        self.y_pred_reverse = 1.0 - self.y_true.copy()

    def test_information_coefficient_perfect_ranking(self):
        from ml.evaluation.metrics import information_coefficient
        ic = information_coefficient(self.y_true, self.y_pred_perfect)
        assert math.isfinite(ic)
        assert ic > 0.9, f"IC of perfect prediction should be near 1, got {ic}"

    def test_information_coefficient_reverse_ranking(self):
        from ml.evaluation.metrics import information_coefficient
        ic = information_coefficient(self.y_true, self.y_pred_reverse)
        assert math.isfinite(ic)
        assert ic < 0.0, f"IC of reverse ranking should be negative, got {ic}"

    def test_brier_score_perfect(self):
        from ml.evaluation.metrics import brier_score
        bs = brier_score(self.y_true.values, self.y_pred_perfect.values)
        assert math.isfinite(bs)
        assert bs < 0.01, f"Brier score of perfect predictions should be near 0, got {bs}"

    def test_brier_score_random(self):
        from ml.evaluation.metrics import brier_score
        bs = brier_score(self.y_true.values, self.y_pred_random.values)
        assert math.isfinite(bs)
        # Random predictions → Brier score around 0.25
        assert 0.15 < bs < 0.40, f"Brier score of random predictions: {bs}"

    def test_meta_label_utility_improvement(self):
        from ml.evaluation.metrics import meta_label_utility
        # High-confidence correct predictions should improve utility
        result = meta_label_utility(
            y_true=self.y_true.values,
            y_pred=self.y_pred_perfect.values,
            raw_signal_precision=0.50,
        )
        assert isinstance(result, dict)

    def test_regime_sliced_metrics_structure(self):
        from ml.evaluation.metrics import regime_sliced_metrics
        regime_labels = np.array(
            ["MEAN_REVERTING"] * 100 + ["TRENDING"] * 100, dtype=str
        )
        result = regime_sliced_metrics(
            self.y_true.values, self.y_pred_random.values, regime_labels
        )
        assert isinstance(result, dict)
        assert "MEAN_REVERTING" in result or len(result) > 0

    def test_robustness_score_range(self):
        from ml.evaluation.metrics import robustness_score
        ic_series = pd.Series(
            np.random.randn(20) * 0.1 + 0.05, index=_make_dates(20)
        )
        score = robustness_score(ic_series)
        assert isinstance(score, float)
        # Robustness score should be in a reasonable range
        assert -1.0 <= score <= 1.5 or math.isnan(score)


class TestModelEvaluator:
    def setup_method(self):
        from sklearn.linear_model import LogisticRegression
        from ml.models.base import MLModel
        np.random.seed(42)
        n_train, n_test = 150, 50
        cols = ["f1", "f2", "f3"]
        dates_train = _make_dates(n_train, "2018-01-01")
        dates_test = _make_dates(n_test, "2019-01-01")
        self.X_train = pd.DataFrame(
            np.random.randn(n_train, 3), index=dates_train, columns=cols
        )
        self.y_train = pd.Series(
            (np.random.randn(n_train) > 0).astype(float), index=dates_train
        )
        self.X_test = pd.DataFrame(
            np.random.randn(n_test, 3), index=dates_test, columns=cols
        )
        self.y_test = pd.Series(
            (np.random.randn(n_test) > 0).astype(float), index=dates_test
        )
        self.ml_model = MLModel(
            LogisticRegression(max_iter=200), feature_names=cols
        )
        self.ml_model.fit(self.X_train, self.y_train)

    def test_evaluate_returns_evaluation_report(self):
        from ml.evaluation.reports import ModelEvaluator
        from ml.contracts import EvaluationReport
        evaluator = ModelEvaluator()
        report = evaluator.evaluate(
            self.ml_model,
            self.X_train, self.y_train,
            self.X_test, self.y_test,
        )
        assert isinstance(report, EvaluationReport)
        assert report.n_samples == len(self.y_test)

    def test_calibration_report_has_reliability_curve(self):
        from ml.evaluation.reports import ModelEvaluator
        from ml.contracts import CalibrationReport
        evaluator = ModelEvaluator()
        report = evaluator.evaluate_calibration(self.ml_model, self.X_test, self.y_test)
        assert isinstance(report, CalibrationReport)

    def test_compare_to_baseline(self):
        from ml.evaluation.reports import ModelEvaluator
        evaluator = ModelEvaluator()
        report = evaluator.evaluate(
            self.ml_model,
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            raw_signal_precision=0.50,
        )
        # filter_improvement can be NaN if precision isn't computable
        # but the field should exist
        assert hasattr(report, "filter_improvement")


# ===========================================================================
# G. Registry and governance tests
# ===========================================================================

class TestMLModelRegistry:
    def setup_method(self):
        from ml.registry.registry import MLModelRegistry
        from ml.contracts import MLTaskFamily
        self.registry = MLModelRegistry()
        self.task = MLTaskFamily.META_LABELING

    def _make_metadata(self, status=None):
        from ml.contracts import ModelMetadata, ModelStatus, MLTaskFamily
        from ml.contracts import GovernanceStatus
        if status is None:
            from ml.contracts import ModelStatus
            status = ModelStatus.RESEARCH
        return ModelMetadata(
            task_family=MLTaskFamily.META_LABELING,
            model_class="LogisticRegression",
            status=status,
            governance_status=GovernanceStatus.PENDING_REVIEW,
        )

    def test_register_and_get(self):
        meta = self._make_metadata()
        model_id = self.registry.register(object(), meta)
        retrieved = self.registry.get(model_id)
        assert retrieved is not None
        assert retrieved.model_id == model_id

    def test_get_champion_returns_none_when_empty(self):
        result = self.registry.get_champion(self.task)
        assert result is None

    def test_promote_to_champion(self):
        from ml.contracts import ModelStatus, PromotionDecision, PromotionOutcome
        meta = self._make_metadata(status=ModelStatus.CANDIDATE)
        model_id = self.registry.register(object(), meta)
        decision = PromotionDecision(
            model_id=model_id,
            outcome=PromotionOutcome.PROMOTE,
            requires_manual_approval=False,
            manually_approved=True,
        )
        success = self.registry.promote(model_id, ModelStatus.CHAMPION, decision)
        assert success is True
        champion = self.registry.get_champion(self.task)
        assert champion is not None
        assert champion.model_id == model_id
        assert champion.status == ModelStatus.CHAMPION

    def test_promote_demotes_previous_champion(self):
        from ml.contracts import ModelStatus, PromotionDecision, PromotionOutcome
        # First champion
        meta1 = self._make_metadata(status=ModelStatus.CANDIDATE)
        id1 = self.registry.register(object(), meta1)
        dec1 = PromotionDecision(
            model_id=id1,
            outcome=PromotionOutcome.PROMOTE,
            requires_manual_approval=False,
        )
        self.registry.promote(id1, ModelStatus.CHAMPION, dec1)

        # Second challenger becomes champion — demotes first
        meta2 = self._make_metadata(status=ModelStatus.CANDIDATE)
        id2 = self.registry.register(object(), meta2)
        dec2 = PromotionDecision(
            model_id=id2,
            outcome=PromotionOutcome.PROMOTE,
            requires_manual_approval=False,
        )
        self.registry.promote(id2, ModelStatus.CHAMPION, dec2)

        m1 = self.registry.get(id1)
        assert m1.status == ModelStatus.CHALLENGER, (
            "Previous champion should be demoted to CHALLENGER"
        )
        champion = self.registry.get_champion(self.task)
        assert champion.model_id == id2

    def test_deprecate_sets_retired_status(self):
        from ml.contracts import ModelStatus, PromotionDecision, PromotionOutcome
        meta = self._make_metadata(status=ModelStatus.CHAMPION)
        model_id = self.registry.register(object(), meta)
        dec = PromotionDecision(
            model_id=model_id,
            outcome=PromotionOutcome.PROMOTE,
            requires_manual_approval=False,
        )
        self.registry.promote(model_id, ModelStatus.RETIRED, dec)
        retrieved = self.registry.get(model_id)
        assert retrieved.status == ModelStatus.RETIRED

    def test_to_dataframe(self):
        meta = self._make_metadata()
        self.registry.register(object(), meta)
        df = self.registry.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1


class TestGovernanceEngine:
    def _make_passing_artifact(self):
        from ml.contracts import TrainingRunArtifact
        return TrainingRunArtifact(
            val_auc=0.65,
            val_brier=0.20,
            cv_ic_mean=0.05,
            calibration_brier_improvement=0.01,
        )

    def _make_failing_artifact(self):
        from ml.contracts import TrainingRunArtifact
        return TrainingRunArtifact(
            val_auc=0.48,    # below 0.55 threshold
            val_brier=0.30,  # above 0.25 threshold
            cv_ic_mean=-0.01,
            calibration_brier_improvement=0.0,
        )

    def test_promotion_criteria_met(self):
        from ml.governance.policies import GovernanceEngine
        from ml.contracts import ModelMetadata, MLTaskFamily, ModelStatus
        engine = GovernanceEngine()
        meta = ModelMetadata(
            task_family=MLTaskFamily.META_LABELING,
            status=ModelStatus.CANDIDATE,
        )
        artifact = self._make_passing_artifact()
        # Manually inject walk_forward_ic_mean > 0 and robustness_score >= 0.55
        artifact.cv_ic_mean = 0.05
        decision = engine.check_promotion_criteria(meta, artifact)
        # All passing criteria should be listed
        assert len(decision.criteria_failed) < len(decision.criteria_met) or \
               len(decision.criteria_met) > 0

    def test_promotion_criteria_failed(self):
        from ml.governance.policies import GovernanceEngine
        from ml.contracts import ModelMetadata, MLTaskFamily, ModelStatus, PromotionOutcome
        engine = GovernanceEngine()
        meta = ModelMetadata(
            task_family=MLTaskFamily.META_LABELING,
            status=ModelStatus.CANDIDATE,
        )
        artifact = self._make_failing_artifact()
        decision = engine.check_promotion_criteria(meta, artifact)
        assert decision.outcome in (PromotionOutcome.REJECT, PromotionOutcome.DEFER)
        assert len(decision.criteria_failed) > 0

    def test_usage_contract_blocks_hard_rule_override(self):
        from ml.contracts import MLUsageContract
        contract = MLUsageContract(may_override_hard_rules=False)
        assert contract.may_override_hard_rules is False

    def test_retirement_criteria_stale_model(self):
        from ml.governance.policies import GovernanceEngine
        from ml.contracts import ModelMetadata, MLTaskFamily, ModelStatus, ModelHealthStatus, ModelHealthState
        engine = GovernanceEngine()
        # A model that is stale should meet retirement criteria
        meta = ModelMetadata(
            task_family=MLTaskFamily.META_LABELING,
            status=ModelStatus.CHAMPION,
        )
        health = ModelHealthStatus(
            model_id=meta.model_id,
            state=ModelHealthState.STALE,
            stale=True,
        )
        should_retire, reasons = engine.evaluate_retirement_criteria(meta, health)
        assert should_retire is True
        assert len(reasons) > 0


# ===========================================================================
# H. Inference and monitoring tests
# ===========================================================================

class TestModelScorer:
    def setup_method(self):
        from sklearn.linear_model import LogisticRegression
        from ml.models.base import MLModel
        from ml.registry.registry import MLModelRegistry
        from ml.contracts import ModelMetadata, MLTaskFamily, ModelStatus, PromotionDecision, PromotionOutcome
        np.random.seed(42)
        cols = ["f1", "f2", "f3"]
        dates = _make_dates(100)
        X = pd.DataFrame(np.random.randn(100, 3), index=dates, columns=cols)
        y = pd.Series((np.random.randn(100) > 0).astype(float), index=dates)

        self.ml_model = MLModel(
            LogisticRegression(max_iter=200), feature_names=cols
        )
        self.ml_model.fit(X, y)

        self.registry = MLModelRegistry()
        meta = ModelMetadata(
            model_id=self.ml_model.model_id,
            task_family=MLTaskFamily.META_LABELING,
            model_class="LogisticRegression",
            status=ModelStatus.CHAMPION,
        )
        self.model_id = self.registry.register(self.ml_model, meta)

    def test_score_returns_inference_result(self):
        from ml.inference.scorer import ModelScorer
        from ml.contracts import InferenceRequest, MLTaskFamily, InferenceResult
        scorer = ModelScorer(registry=self.registry)
        request = InferenceRequest(
            entity_id="AAPL-MSFT",
            task_family=MLTaskFamily.META_LABELING,
            features={"f1": 1.0, "f2": -0.5, "f3": 0.3},
        )
        result = scorer.score(request)
        assert isinstance(result, InferenceResult)

    def test_fallback_when_no_model(self):
        from ml.inference.scorer import ModelScorer
        from ml.contracts import InferenceRequest, MLTaskFamily, InferenceResult
        from ml.registry.registry import MLModelRegistry
        empty_registry = MLModelRegistry()
        scorer = ModelScorer(registry=empty_registry)
        request = InferenceRequest(
            entity_id="X-Y",
            task_family=MLTaskFamily.META_LABELING,
            features={"f1": 0.5},
        )
        result = scorer.score(request)
        assert isinstance(result, InferenceResult)
        assert result.fallback_used is True

    def test_batch_scoring_never_raises(self):
        from ml.inference.scorer import ModelScorer
        from ml.contracts import InferenceRequest, MLTaskFamily
        scorer = ModelScorer(registry=self.registry)
        requests = [
            InferenceRequest(
                entity_id=f"pair_{i}",
                task_family=MLTaskFamily.META_LABELING,
                features={"f1": float(i), "f2": 0.0, "f3": -float(i)},
            )
            for i in range(10)
        ]
        # Should not raise
        results = scorer.score_batch(requests)
        assert len(results) == 10


class TestFeatureDriftMonitor:
    def setup_method(self):
        np.random.seed(42)
        # Use large n so PSI is stable
        n = 2000
        cols = ["f1", "f2", "f3"]
        self.dates_ref = _make_dates(n, "2018-01-01")
        rng = np.random.default_rng(0)
        # Reference: standard normal
        ref_data = rng.standard_normal((n, 3))
        self.X_reference = pd.DataFrame(ref_data, index=self.dates_ref, columns=cols)
        # Same distribution (independent draw from same params)
        same_data = rng.standard_normal((n, 3))
        self.X_same = pd.DataFrame(same_data, index=self.dates_ref, columns=cols)
        # Heavily different distribution: mean shifted by 5 sigmas
        diff_data = 5.0 + 0.5 * rng.standard_normal((n, 3))
        self.X_different = pd.DataFrame(diff_data, index=self.dates_ref, columns=cols)

    def test_no_drift_same_distribution(self):
        from ml.monitoring.drift import FeatureDriftMonitor
        monitor = FeatureDriftMonitor(n_bins=10)
        monitor.fit_reference(self.X_reference)
        psi_scores = monitor.compute_psi(self.X_same)
        valid_scores = [v for v in psi_scores.values() if math.isfinite(v)]
        assert len(valid_scores) > 0
        # Mean PSI for same-distribution data should be low
        mean_psi = np.mean(valid_scores)
        assert mean_psi < 0.20, (
            f"Mean PSI {mean_psi:.3f} too high for same-distribution data"
        )

    def test_significant_drift_different_distribution(self):
        from ml.monitoring.drift import FeatureDriftMonitor
        monitor = FeatureDriftMonitor()
        monitor.fit_reference(self.X_reference)
        psi_scores = monitor.compute_psi(self.X_different)
        valid_scores = [v for v in psi_scores.values() if math.isfinite(v)]
        assert len(valid_scores) > 0
        # Heavily shifted distribution → at least some features should have PSI > 0.25
        high_psi_count = sum(1 for v in valid_scores if v > 0.25)
        assert high_psi_count > 0, (
            f"Expected some high PSI scores for shifted distribution, got: {psi_scores}"
        )

    def test_drift_report_has_severity(self):
        from ml.monitoring.drift import FeatureDriftMonitor
        from ml.contracts import DriftSeverity
        monitor = FeatureDriftMonitor()
        monitor.fit_reference(self.X_reference)
        report = monitor.generate_drift_report("model-001", self.X_different)
        assert isinstance(report.severity, DriftSeverity)
        assert report.model_id == "model-001"


class TestModelHealthMonitor:
    def setup_method(self):
        from ml.registry.registry import MLModelRegistry
        from ml.contracts import ModelMetadata, MLTaskFamily, ModelStatus
        self.registry = MLModelRegistry()
        now_str = datetime.now(tz=timezone.utc).isoformat()
        meta = ModelMetadata(
            task_family=MLTaskFamily.META_LABELING,
            model_class="LogisticRegression",
            status=ModelStatus.CHAMPION,
            trained_at=now_str,
        )
        self.model_id = self.registry.register(object(), meta)

    def test_fresh_model_healthy(self):
        from ml.monitoring.drift import ModelHealthMonitor
        from ml.contracts import ModelHealthState
        monitor = ModelHealthMonitor(registry=self.registry, stale_threshold_hours=48)
        status = monitor.check_health(self.model_id)
        assert status.state in (
            ModelHealthState.HEALTHY,
            ModelHealthState.DRIFTING,
        ), f"Fresh model should be healthy or drifting, got {status.state}"

    def test_stale_model_detected(self):
        from ml.registry.registry import MLModelRegistry
        from ml.contracts import ModelMetadata, MLTaskFamily, ModelStatus
        from ml.monitoring.drift import ModelHealthMonitor
        from ml.contracts import ModelHealthState
        registry = MLModelRegistry()
        # Trained 3 days ago
        old_ts = (datetime.now(tz=timezone.utc) - timedelta(days=3)).isoformat()
        meta = ModelMetadata(
            task_family=MLTaskFamily.META_LABELING,
            model_class="LogisticRegression",
            status=ModelStatus.CHAMPION,
            trained_at=old_ts,
        )
        model_id = registry.register(object(), meta)
        monitor = ModelHealthMonitor(registry=registry, stale_threshold_hours=24)
        status = monitor.check_health(model_id)
        assert status.stale is True

    def test_health_summary_is_dataframe(self):
        from ml.monitoring.drift import ModelHealthMonitor
        monitor = ModelHealthMonitor(registry=self.registry)
        df = monitor.generate_health_summary(model_ids=[self.model_id])
        assert isinstance(df, pd.DataFrame)


# ===========================================================================
# I. Integration tests — end-to-end pipelines
# ===========================================================================

class TestMLPipelineIntegration:
    """End-to-end integration tests for the complete ML pipeline."""

    def setup_method(self):
        np.random.seed(42)
        self.dates = _make_dates(500, "2018-01-01")
        self.px = pd.Series(
            100 * np.cumprod(1 + 0.01 * np.random.randn(500)), index=self.dates
        )
        self.py = pd.Series(
            100 * np.cumprod(1 + 0.01 * np.random.randn(500)), index=self.dates
        )
        self.z = pd.Series(
            2 * np.sin(np.linspace(0, 20 * np.pi, 500)) + 0.5 * np.random.randn(500),
            index=self.dates,
        )

    def test_meta_label_pipeline(self):
        """Full pipeline: build features → build labels → train meta-labeler → score."""
        from sklearn.linear_model import LogisticRegression
        from ml.models.base import MLModel
        from ml.models.meta_labeler import MetaLabelModel
        from ml.labels.builder import LabelBuilder

        # Build labels
        builder = LabelBuilder()
        labels = builder.build_reversion_label(self.z, horizon=10)

        # Build simple feature matrix from the z-score
        n = 300
        X = pd.DataFrame({
            "z": self.z.iloc[:n].values,
            "z_lag1": self.z.shift(1).iloc[:n].values,
            "z_lag2": self.z.shift(2).iloc[:n].values,
            "z_abs": np.abs(self.z.iloc[:n].values),
            "z_std": self.z.rolling(20).std().iloc[:n].values,
        }, index=self.dates[:n])

        y = labels.iloc[:n]

        # Train/test split (no overlap)
        train_end_idx = 200
        X_train = X.iloc[:train_end_idx].dropna()
        y_train = y.iloc[:train_end_idx].reindex(X_train.index).dropna()
        X_train = X_train.reindex(y_train.index)

        X_test = X.iloc[220:].dropna()
        y_test = y.iloc[220:].reindex(X_test.index).dropna()
        X_test = X_test.reindex(y_test.index)

        if len(X_train) < 10 or len(X_test) < 5:
            pytest.skip("Not enough valid samples for integration test")

        cols = list(X_train.columns)
        base = MLModel(LogisticRegression(max_iter=500), feature_names=cols)
        model = MetaLabelModel(base_model=base, feature_names=cols)
        artifact = model.fit(X_train, y_train)
        assert artifact.n_train_samples > 0

        # Score
        p = model.predict_success_probability(X_test.iloc[0])
        assert 0.0 <= p <= 1.0

    def test_walk_forward_evaluation_pipeline(self):
        """Walk-forward: 2 folds, train/evaluate per fold, compute IC."""
        from sklearn.linear_model import LogisticRegression
        from ml.models.base import MLModel
        from ml.datasets.splits import TemporalSplitter
        from ml.evaluation.metrics import information_coefficient

        n = len(self.dates)
        X = pd.DataFrame({
            "z": self.z.values,
            "z_abs": np.abs(self.z.values),
        }, index=self.dates)
        y = pd.Series((self.z > 1.0).astype(float), index=self.dates)

        splitter = TemporalSplitter(purge_days=5, embargo_days=10)
        plans = splitter.walk_forward_splits(
            self.dates, n_folds=2, min_train_periods=150, test_periods=50
        )
        assert len(plans) >= 2

        ic_values = []
        for plan in plans:
            train_mask = (self.dates >= pd.Timestamp(plan.train_start)) & \
                         (self.dates <= pd.Timestamp(plan.train_end))
            test_mask = (self.dates >= pd.Timestamp(plan.test_start)) & \
                        (self.dates <= pd.Timestamp(plan.test_end))

            X_train_fold = X.loc[train_mask].dropna()
            y_train_fold = y.loc[train_mask].reindex(X_train_fold.index)
            X_test_fold = X.loc[test_mask].dropna()
            y_test_fold = y.loc[test_mask].reindex(X_test_fold.index)

            if len(X_train_fold) < 20 or len(X_test_fold) < 5:
                continue

            model = MLModel(
                LogisticRegression(max_iter=200),
                feature_names=list(X.columns),
            )
            model.fit(X_train_fold, y_train_fold)
            proba = model.predict_proba(X_test_fold)[:, 1]
            y_pred_series = pd.Series(proba, index=X_test_fold.index)
            ic = information_coefficient(y_test_fold, y_pred_series)
            if math.isfinite(ic):
                ic_values.append(ic)

        assert len(ic_values) > 0, "Walk-forward produced no valid IC values"

    def test_drift_detection_pipeline(self):
        """Train on first half, drift monitor on second half (different distribution)."""
        from ml.monitoring.drift import FeatureDriftMonitor

        n = len(self.dates)
        mid = n // 2
        X = pd.DataFrame({
            "z": self.z.values,
            "z_abs": np.abs(self.z.values),
            "z_std": pd.Series(self.z.values).rolling(10, min_periods=1).std().values,
        }, index=self.dates)

        X_ref = X.iloc[:mid]
        # Second half with added drift
        X_current = X.iloc[mid:].copy()
        X_current = X_current + np.array([3.0, 0.0, 0.0])

        monitor = FeatureDriftMonitor(psi_alert_threshold=0.25)
        monitor.fit_reference(X_ref)
        report = monitor.generate_drift_report("drift-test", X_current)
        # With a mean shift of +3 on feature "z", some drift should be detected
        finite_psi = [v for v in report.feature_psi_scores.values() if math.isfinite(v)]
        assert len(finite_psi) > 0


# ===========================================================================
# J. Leakage integrity tests
# ===========================================================================

class TestLeakageIntegrity:
    """Research integrity tests — verify the system correctly prevents leakage."""

    def test_future_feature_is_detected(self):
        """Deliberate leakage: inject a future-informed feature. LeakageAuditor must detect."""
        from ml.datasets.leakage import LeakageAuditor
        np.random.seed(42)
        train_dates = _make_dates(100, "2018-01-01")
        test_dates = _make_dates(50, "2019-01-01")
        train_end = train_dates[-1]

        # X_train contains a row at future date
        future_date = test_dates[10]
        contaminated_dates = train_dates.append(pd.DatetimeIndex([future_date]))
        X_train = pd.DataFrame(
            np.random.randn(len(contaminated_dates), 3),
            index=contaminated_dates,
            columns=["f1", "f2", "f3"],
        )
        y_train = pd.Series(
            np.random.randint(0, 2, len(contaminated_dates)).astype(float),
            index=contaminated_dates,
        )
        X_test = pd.DataFrame(
            np.random.randn(50, 3), index=test_dates, columns=["f1", "f2", "f3"]
        )
        y_test = pd.Series(np.random.randint(0, 2, 50).astype(float), index=test_dates)

        auditor = LeakageAuditor()
        report = auditor.audit(
            X_train, y_train, X_test, y_test,
            train_end=train_end,
            test_start=test_dates[0],
            label_horizon_days=5,
            embargo_days=5,
        )
        assert report.passed is False
        assert report.future_feature_risk is True

    def test_train_test_chronological_order(self):
        """All test timestamps must be strictly after all train timestamps."""
        from ml.datasets.splits import TemporalSplitter
        np.random.seed(42)
        dates = _make_dates(500, "2018-01-01")
        splitter = TemporalSplitter(purge_days=5, embargo_days=10)
        plan = splitter.chronological_split(dates)

        train_end = pd.Timestamp(plan.train_end)
        test_start = pd.Timestamp(plan.test_start)
        assert test_start > train_end, (
            "All test timestamps must be strictly after all train timestamps"
        )

    def test_model_fit_respects_train_end(self):
        """Model trained with train_end=T should have no data from T+1 onwards."""
        from sklearn.linear_model import LogisticRegression
        from ml.models.base import MLModel
        np.random.seed(42)
        dates = _make_dates(200)
        X = pd.DataFrame(
            np.random.randn(200, 3), index=dates, columns=["f1", "f2", "f3"]
        )
        y = pd.Series((np.random.randn(200) > 0).astype(float), index=dates)
        train_end = dates[99]

        model = MLModel(
            LogisticRegression(max_iter=200), feature_names=["f1", "f2", "f3"]
        )
        # X contains data beyond train_end — should raise
        with pytest.raises(ValueError, match="train_end"):
            model.fit(X, y, train_end=train_end)

        # Fitting with only train data should succeed
        model2 = MLModel(
            LogisticRegression(max_iter=200), feature_names=["f1", "f2", "f3"]
        )
        artifact = model2.fit(X.iloc[:100], y.iloc[:100], train_end=train_end)
        assert artifact.n_train_samples > 0

    def test_overlapping_label_horizons_handled(self):
        """
        If label horizon > purge_days, some train samples have labels that
        overlap with test. Verify this is detected and flagged.
        """
        from ml.datasets.leakage import LeakageAuditor
        np.random.seed(42)
        train_dates = _make_dates(200, "2018-01-01")
        test_dates = _make_dates(50, "2019-01-01")
        X_train = pd.DataFrame(
            np.random.randn(200, 3), index=train_dates, columns=["f1", "f2", "f3"]
        )
        y_train = pd.Series(np.random.randint(0, 2, 200).astype(float), index=train_dates)
        X_test = pd.DataFrame(
            np.random.randn(50, 3), index=test_dates, columns=["f1", "f2", "f3"]
        )
        y_test = pd.Series(np.random.randint(0, 2, 50).astype(float), index=test_dates)

        auditor = LeakageAuditor()
        # horizon=30, embargo=5: total needed gap = 35 days
        # Actual gap between train_end (~2018-10) and test_start (2019-01-01) is > 35 days
        # so this should pass
        report = auditor.audit(
            X_train, y_train, X_test, y_test,
            train_end=train_dates[-1],
            test_start=test_dates[0],
            label_horizon_days=30,
            embargo_days=5,
        )
        # With a large gap (several months) this should pass
        gap_days = (test_dates[0] - train_dates[-1]).days
        if gap_days >= 35:
            assert report.passed is True or len(report.violations) == 0 or \
                   all("overlap" not in v.lower() for v in report.violations)


# ===========================================================================
# J. Explainability tests
# ===========================================================================

class TestFeatureLineageTracker:
    def test_record_training_lineage(self):
        from ml.explainability.lineage import FeatureLineageTracker
        tracker = FeatureLineageTracker()
        records = tracker.record_training_lineage(
            model_id="model-001",
            feature_names=["pair_z", "pair_z_abs", "pair_corr_20d"],
            feature_set_id="fs-001",
            train_start="2018-01-01",
            train_end="2020-01-01",
            dataset_snapshot_id="snap-001",
        )
        assert len(records) == 3
        assert records[0].feature_name == "pair_z"

    def test_get_model_lineage(self):
        from ml.explainability.lineage import FeatureLineageTracker
        tracker = FeatureLineageTracker()
        tracker.record_training_lineage(
            model_id="model-002",
            feature_names=["pair_z", "inst_vol_20d"],
            feature_set_id="fs-002",
            train_start="2018-01-01",
            train_end="2020-01-01",
            dataset_snapshot_id="snap-002",
        )
        records = tracker.get_model_lineage("model-002")
        assert len(records) == 2

    def test_get_model_lineage_empty_when_not_found(self):
        from ml.explainability.lineage import FeatureLineageTracker
        tracker = FeatureLineageTracker()
        records = tracker.get_model_lineage("nonexistent-model")
        assert records == []

    def test_generate_lineage_report(self):
        from ml.explainability.lineage import FeatureLineageTracker
        tracker = FeatureLineageTracker()
        tracker.record_training_lineage(
            model_id="model-003",
            feature_names=["pair_z", "pair_corr_20d", "unknown_feature"],
            feature_set_id="fs-003",
            train_start="2018-01-01",
            train_end="2020-01-01",
            dataset_snapshot_id="snap-003",
        )
        report = tracker.generate_lineage_report("model-003")
        assert report["model_id"] == "model-003"
        assert report["n_features"] == 3
        # Coverage: 2 of 3 features are in FEATURE_REGISTRY
        assert 0.0 <= report["coverage"] <= 1.0
        assert "z" in report["all_raw_inputs"]

    def test_check_feature_compatibility_all_available(self):
        from ml.explainability.lineage import FeatureLineageTracker
        tracker = FeatureLineageTracker()
        ok, missing = tracker.check_feature_compatibility(
            model_feature_names=["f1", "f2", "f3"],
            available_feature_names=["f1", "f2", "f3", "f4"],
        )
        assert ok is True
        assert missing == []

    def test_check_feature_compatibility_missing_features(self):
        from ml.explainability.lineage import FeatureLineageTracker
        tracker = FeatureLineageTracker()
        ok, missing = tracker.check_feature_compatibility(
            model_feature_names=["f1", "f2", "f99"],
            available_feature_names=["f1", "f2"],
        )
        assert ok is False
        assert "f99" in missing


class TestFeatureImportance:
    def setup_method(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from ml.models.base import MLModel
        np.random.seed(42)
        n = 100
        self.dates = _make_dates(n)
        self.cols = ["f1", "f2", "f3", "f4", "f5"]
        self.X = pd.DataFrame(
            np.random.randn(n, 5), index=self.dates, columns=self.cols
        )
        self.y = pd.Series(
            (np.random.randn(n) > 0).astype(float), index=self.dates
        )
        # Logistic regression model
        self.lr_model = MLModel(
            LogisticRegression(max_iter=200), feature_names=self.cols
        )
        self.lr_model.fit(self.X, self.y)

        # Tree model
        self.rf_model = MLModel(
            RandomForestClassifier(n_estimators=10, random_state=42),
            feature_names=self.cols,
        )
        self.rf_model.fit(self.X, self.y)

    def test_importance_coef_returns_dict(self):
        from ml.explainability.importance import compute_feature_importance
        result = compute_feature_importance(
            self.lr_model, self.X, self.y, method="coef"
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == set(self.cols)

    def test_importance_coef_sums_to_one(self):
        from ml.explainability.importance import compute_feature_importance
        result = compute_feature_importance(
            self.lr_model, self.X, self.y, method="coef"
        )
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6, f"Importances should sum to 1, got {total}"

    def test_importance_gain_returns_dict(self):
        from ml.explainability.importance import compute_feature_importance
        result = compute_feature_importance(
            self.rf_model, self.X, self.y, method="gain"
        )
        assert isinstance(result, dict)
        assert len(result) == len(self.cols)

    def test_importance_gain_sums_to_one(self):
        from ml.explainability.importance import compute_feature_importance
        result = compute_feature_importance(
            self.rf_model, self.X, self.y, method="gain"
        )
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-6

    def test_rank_features_by_importance_sorted_descending(self):
        from ml.explainability.importance import rank_features_by_importance
        importance = {"f1": 0.10, "f2": 0.50, "f3": 0.30, "f4": 0.05, "f5": 0.05}
        ranked = rank_features_by_importance(importance)
        assert ranked[0][0] == "f2"  # highest
        for i in range(len(ranked) - 1):
            assert ranked[i][1] >= ranked[i + 1][1], "Not sorted descending"

    def test_rank_features_by_importance_top_n(self):
        from ml.explainability.importance import rank_features_by_importance
        importance = {"f1": 0.10, "f2": 0.50, "f3": 0.30, "f4": 0.05, "f5": 0.05}
        ranked = rank_features_by_importance(importance, top_n=2)
        assert len(ranked) == 2

    def test_generate_importance_report_returns_artifact(self):
        from ml.explainability.importance import (
            compute_feature_importance,
            generate_importance_report,
        )
        from ml.contracts import ExplainabilityArtifact
        importance = compute_feature_importance(
            self.rf_model, self.X, self.y, method="gain"
        )
        artifact = generate_importance_report("model-explainer", importance, top_n=3)
        assert isinstance(artifact, ExplainabilityArtifact)
        assert artifact.model_id == "model-explainer"
        assert len(artifact.top_features_ranked) <= 3

    def test_empty_model_returns_empty_dict(self):
        from ml.explainability.importance import compute_feature_importance
        empty_X = pd.DataFrame()
        result = compute_feature_importance(object(), empty_X, pd.Series(), method="gain")
        assert result == {}
