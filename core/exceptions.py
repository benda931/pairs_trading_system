# core/exceptions.py
"""
Custom Exception Classes for the Pairs Trading System
Encapsulates validation, configuration, optimization, and signal generation errors
"""

class SignalGeneratorError(Exception):
    """Raised when signal generator fails to initialize or generate signals."""
    pass

class SeriesValidationError(Exception):
    """Raised when a time series fails validation or has missing/invalid structure."""
    pass

class OptimizationError(Exception):
    """Raised when optimization process encounters a critical error or misconfiguration."""
    pass

class ConfigurationError(Exception):
    """Raised when the config file is missing fields or contains invalid values."""
    pass

class DataLoadingError(Exception):
    """Raised when price/hedge data fails to load or is malformed."""
    pass

class FeatureSelectionError(Exception):
    """Raised when feature selection fails due to instability, data sparsity, or scoring logic issues."""
    pass

class MetaOptimizationError(Exception):
    """Raised when meta-optimization fails to converge or returns invalid parameters."""
    pass

class VisualizationError(Exception):
    """Raised when plot generation or rendering fails (e.g. due to missing data or malformed input)."""
    pass

