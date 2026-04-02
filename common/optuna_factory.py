"""
common/optuna_factory.py — Canonical Optuna study factory.

Centralises create_study() calls to ensure consistent configuration
across all optimization workflows: sampler, pruner, storage, logging.

Usage
-----
    from common.optuna_factory import create_optuna_study

    study = create_optuna_study(
        study_name="alpha_XLI_XLB",
        direction="maximize",
    )
    study.optimize(objective, n_trials=50)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Default storage directory
_DEFAULT_STORAGE_DIR = Path("optuna_studies")

# Default sampler/pruner configuration
_DEFAULT_STARTUP_TRIALS = 10
_DEFAULT_MULTIVARIATE = True


def create_optuna_study(
    study_name: str | None = None,
    *,
    direction: str = "maximize",
    directions: Sequence[str] | None = None,
    storage: str | None = None,
    load_if_exists: bool = True,
    sampler: Any | None = None,
    pruner: Any | None = None,
    use_default_sampler: bool = True,
    use_default_pruner: bool = True,
    n_startup_trials: int = _DEFAULT_STARTUP_TRIALS,
    **kwargs: Any,
) -> Any:
    """
    Create an Optuna study with consistent defaults.

    Parameters
    ----------
    study_name : str | None
        Human-readable study name. None = Optuna auto-generates.
    direction : str
        "maximize" or "minimize". Ignored if *directions* is set.
    directions : Sequence[str] | None
        For multi-objective; overrides *direction*.
    storage : str | None
        Optuna storage URL. None = in-memory.
    load_if_exists : bool
        If True, load existing study with same name from storage.
    sampler : Any | None
        Custom sampler. If None and use_default_sampler=True, creates TPESampler.
    pruner : Any | None
        Custom pruner. If None and use_default_pruner=True, creates MedianPruner.
    use_default_sampler : bool
        Create default TPESampler if no sampler provided.
    use_default_pruner : bool
        Create default MedianPruner if no pruner provided.
    n_startup_trials : int
        Startup trials for default TPESampler.

    Returns
    -------
    optuna.Study
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "optuna is required for optimization. Install with: pip install optuna"
        )

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Default sampler: TPESampler with multivariate
    if sampler is None and use_default_sampler:
        sampler = optuna.samplers.TPESampler(
            multivariate=_DEFAULT_MULTIVARIATE,
            n_startup_trials=n_startup_trials,
        )

    # Default pruner: MedianPruner
    if pruner is None and use_default_pruner:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(5, n_startup_trials // 2),
        )

    # Build create_study kwargs
    study_kwargs: dict[str, Any] = {
        "load_if_exists": load_if_exists,
    }

    if study_name:
        study_kwargs["study_name"] = study_name
    if storage:
        study_kwargs["storage"] = storage
    if sampler:
        study_kwargs["sampler"] = sampler
    if pruner:
        study_kwargs["pruner"] = pruner

    # Direction(s)
    if directions is not None:
        study_kwargs["directions"] = list(directions)
    else:
        study_kwargs["direction"] = direction

    # Merge any extra kwargs
    study_kwargs.update(kwargs)

    study = optuna.create_study(**study_kwargs)

    logger.info(
        "Optuna study created: name=%s direction=%s storage=%s",
        study_name or "(auto)",
        directions or direction,
        storage or "in-memory",
    )

    return study
