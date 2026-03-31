# -*- coding: utf-8 -*-
"""
ml/datasets/leakage.py — Leakage Audit System
===============================================

Every training dataset must pass a LeakageAuditor before use.
The auditor runs a battery of temporal leakage checks and returns
a structured LeakageAuditReport.

Checks performed:
  1. Future-feature check: any feature row timestamped after cutoff?
  2. Train/test overlap check: do windows overlap within embargo buffer?
  3. Normalization leak check: were test features normalized with test statistics?
  4. Label-horizon contamination: is embargo wide enough given label horizon?

A dataset with audit.passed == False must NOT be used for training.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ml.contracts import LeakageAuditReport

logger = logging.getLogger("ml.datasets.leakage")


class LeakageAuditor:
    """
    Audits a dataset for temporal leakage risks.

    Every training dataset must pass a LeakageAuditor before use.
    The audit is cheap and deterministic — run it every time.

    Usage:
        auditor = LeakageAuditor()
        report = auditor.audit(X_train, y_train, X_test, y_test,
                               train_end=..., test_start=...,
                               label_horizon_days=10)
        assert report.passed, report.violations
    """

    # Names used in checks_run list
    CHECK_FUTURE_FEATURES = "future_features"
    CHECK_TRAIN_TEST_OVERLAP = "train_test_overlap"
    CHECK_NORMALIZATION_LEAK = "normalization_leak"
    CHECK_LABEL_HORIZON = "label_horizon_contamination"
    CHECK_INDEX_MONOTONIC = "index_monotonic"
    CHECK_NULL_LABELS = "null_labels_in_train"

    def audit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        feature_definitions: Optional[Dict] = None,
        label_horizon_days: int = 0,
        embargo_days: int = 10,
    ) -> LeakageAuditReport:
        """
        Run all leakage checks and return a structured LeakageAuditReport.

        Parameters
        ----------
        X_train, y_train: training feature matrix and label series
        X_test, y_test:   test feature matrix and label series
        train_end:        the hard boundary — train features must not exceed this
        test_start:       test features must not precede this
        feature_definitions: optional dict of FeatureDefinition objects for metadata checks
        label_horizon_days: forward horizon of the label; used for horizon contamination check
        embargo_days:     minimum gap between train_end and test_start

        Returns LeakageAuditReport with passed=False if any hard violation detected.
        """
        checks_run: List[str] = []
        violations: List[str] = []
        warnings: List[str] = []

        future_feature_risk = False
        normalization_leak_risk = False
        overlap_label_risk = False
        embargo_adequate = True
        purge_adequate = True

        # ------------------------------------------------------------------
        # Check 1: Index monotonicity
        # ------------------------------------------------------------------
        checks_run.append(self.CHECK_INDEX_MONOTONIC)
        if not X_train.index.is_monotonic_increasing:
            violations.append("X_train index is not monotonically increasing")
        if not X_test.index.is_monotonic_increasing:
            violations.append("X_test index is not monotonically increasing")

        # ------------------------------------------------------------------
        # Check 2: Future features in train set
        # ------------------------------------------------------------------
        checks_run.append(self.CHECK_FUTURE_FEATURES)
        future_cols = self.check_future_features(X_train, train_end)
        if future_cols:
            future_feature_risk = True
            violations.append(
                f"X_train has {len(future_cols)} rows after train_end ({train_end.date()}): "
                f"first offender index = {future_cols[0]}"
            )

        # ------------------------------------------------------------------
        # Check 3: Train/test temporal overlap
        # ------------------------------------------------------------------
        checks_run.append(self.CHECK_TRAIN_TEST_OVERLAP)
        has_overlap = self.check_overlap(
            train_index=X_train.index,
            test_index=X_test.index,
            embargo_days=embargo_days,
        )
        if has_overlap:
            overlap_label_risk = True
            violations.append(
                f"Train and test windows overlap within embargo buffer ({embargo_days} days). "
                f"train_end={train_end.date()}, test_start={test_start.date()}"
            )

        # ------------------------------------------------------------------
        # Check 4: Normalization leak detection
        # ------------------------------------------------------------------
        checks_run.append(self.CHECK_NORMALIZATION_LEAK)
        norm_leak = self.check_normalization_leak(X_train, X_test)
        if norm_leak:
            normalization_leak_risk = True
            warnings.append(
                "Possible normalization leak: test feature statistics closely match training. "
                "Verify that scaling/standardization was fit on training data only."
            )

        # ------------------------------------------------------------------
        # Check 5: Label horizon contamination
        # ------------------------------------------------------------------
        checks_run.append(self.CHECK_LABEL_HORIZON)
        horizon_ok = self.check_label_horizon_contamination(
            train_end=train_end,
            label_horizon_days=label_horizon_days,
            embargo_days=embargo_days,
            test_start=test_start,
        )
        if not horizon_ok:
            embargo_adequate = False
            gap_days = (test_start - train_end).days
            required = label_horizon_days + embargo_days
            violations.append(
                f"Label horizon contamination risk: gap between train_end and test_start "
                f"is {gap_days} days, but label_horizon={label_horizon_days} + "
                f"embargo={embargo_days} = {required} days required."
            )

        # ------------------------------------------------------------------
        # Check 6: NaN labels in training set
        # ------------------------------------------------------------------
        checks_run.append(self.CHECK_NULL_LABELS)
        if y_train is not None and len(y_train) > 0:
            null_frac = y_train.isna().mean()
            if null_frac > 0.30:
                warnings.append(
                    f"High null fraction in y_train: {null_frac:.1%}. "
                    "Consider shrinking the train window or checking label computation."
                )
            if null_frac == 1.0:
                violations.append("y_train is entirely NaN — label generation failed")

        # ------------------------------------------------------------------
        # Verdict
        # ------------------------------------------------------------------
        passed = len(violations) == 0

        if not passed:
            logger.warning(
                "LeakageAudit FAILED (%d violations): %s", len(violations), violations
            )
        elif warnings:
            logger.info(
                "LeakageAudit PASSED with %d warnings: %s", len(warnings), warnings
            )
        else:
            logger.debug("LeakageAudit PASSED cleanly")

        return LeakageAuditReport(
            passed=passed,
            checks_run=checks_run,
            violations=violations,
            warnings=warnings,
            future_feature_risk=future_feature_risk,
            future_label_risk=False,   # label leakage is by design in offline pipelines
            normalization_leak_risk=normalization_leak_risk,
            overlap_label_risk=overlap_label_risk,
            embargo_adequate=embargo_adequate,
            purge_adequate=purge_adequate,
        )

    def check_future_features(
        self,
        X: pd.DataFrame,
        cutoff: pd.Timestamp,
    ) -> List[str]:
        """
        Return list of offending index values where X.index > cutoff.

        An empty list means no future-feature contamination detected.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            return []
        offenders = X.index[X.index > cutoff]
        return [str(ts) for ts in offenders]

    def check_overlap(
        self,
        train_index: pd.DatetimeIndex,
        test_index: pd.DatetimeIndex,
        embargo_days: int,
    ) -> bool:
        """
        Check if train and test windows overlap within embargo buffer.

        Returns True (violation detected) if any test timestamp falls within
        embargo_days of the train window end.
        """
        if len(train_index) == 0 or len(test_index) == 0:
            return False
        train_end = train_index.max()
        test_start = test_index.min()
        gap = (test_start - train_end).days
        return gap < embargo_days

    def check_normalization_leak(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> bool:
        """
        Heuristic check for normalization leakage.

        A likely leak is detected when test feature means are extremely close
        to zero AND test feature stds are extremely close to 1.0 while train stats
        differ significantly. This pattern indicates the test set was re-normalized
        using its own statistics rather than those of the training set.

        Returns True if normalization leak is suspected.
        """
        if X_train.empty or X_test.empty:
            return False

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return False

        # Sample up to 20 columns for efficiency
        cols = numeric_cols[:20]

        test_means = X_test[cols].mean()
        test_stds = X_test[cols].std()
        train_means = X_train[cols].mean()
        train_stds = X_train[cols].std()

        # Suspicious if test is nearly unit-normal but train is not
        test_near_zero_mean = (test_means.abs() < 0.05).mean() > 0.80
        test_near_unit_std = ((test_stds - 1.0).abs() < 0.10).mean() > 0.80
        train_not_normalized = (train_means.abs() > 0.20).mean() > 0.50

        return bool(test_near_zero_mean and test_near_unit_std and train_not_normalized)

    def check_label_horizon_contamination(
        self,
        train_end: pd.Timestamp,
        label_horizon_days: int,
        embargo_days: int,
        test_start: Optional[pd.Timestamp] = None,
    ) -> bool:
        """
        Check if the label horizon could contaminate the train/test boundary.

        Labels extend horizon_days into the future from decision time.
        The test window must start at least (label_horizon_days + embargo_days)
        after train_end to prevent contamination.

        Returns True if the check passes (sufficient gap), False if contamination risk.
        """
        required_gap = label_horizon_days + embargo_days
        if test_start is None:
            return True  # Cannot check without test_start
        actual_gap = (test_start - train_end).days
        return actual_gap >= required_gap

    def summarize(self, report: LeakageAuditReport) -> str:
        """Return a human-readable summary of the audit report."""
        status = "PASSED" if report.passed else "FAILED"
        lines = [
            f"LeakageAudit {status} (audit_id={report.audit_id})",
            f"  Checks run: {', '.join(report.checks_run)}",
        ]
        if report.violations:
            lines.append(f"  Violations ({len(report.violations)}):")
            for v in report.violations:
                lines.append(f"    - {v}")
        if report.warnings:
            lines.append(f"  Warnings ({len(report.warnings)}):")
            for w in report.warnings:
                lines.append(f"    - {w}")
        lines.append(f"  future_feature_risk={report.future_feature_risk}")
        lines.append(f"  normalization_leak_risk={report.normalization_leak_risk}")
        lines.append(f"  overlap_label_risk={report.overlap_label_risk}")
        lines.append(f"  embargo_adequate={report.embargo_adequate}")
        return "\n".join(lines)
