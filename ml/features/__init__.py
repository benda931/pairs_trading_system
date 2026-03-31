# -*- coding: utf-8 -*-
"""
ml/features/__init__.py — Feature Platform Public API
=====================================================

Exports the feature registry, feature group index, and the
point-in-time feature builder for both live signal generation
and offline training pipelines.
"""

from ml.features.definitions import FEATURE_REGISTRY, FEATURE_GROUPS
from ml.features.builder import PointInTimeFeatureBuilder

__all__ = [
    "FEATURE_REGISTRY",
    "FEATURE_GROUPS",
    "PointInTimeFeatureBuilder",
]
