# -*- coding: utf-8 -*-
"""
ml/labels/__init__.py — Label Platform Public API
==================================================

Exports the label registry and the LabelBuilder for offline training pipelines.
Labels are computed from FUTURE data by design and must only be used
in the training pipeline — never in live feature computation.
"""

from ml.labels.definitions import LABEL_REGISTRY
from ml.labels.builder import LabelBuilder

__all__ = [
    "LABEL_REGISTRY",
    "LabelBuilder",
]
