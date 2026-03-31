# -*- coding: utf-8 -*-
"""
params.py (root shim)
---------------------
Backward-compat shim for legacy imports: `import params` / `from params import ...`.

Canonical implementation lives in `core.params`.
"""

from __future__ import annotations

from core.params import *  # noqa: F401,F403

# Legacy helper expected by older scripts (e.g. full_parameter_optimization)
# If the canonical module doesn't expose it, we provide a safe no-op that keeps behavior reasonable.
try:
    freeze  # type: ignore[name-defined]
except Exception:
    from typing import Iterable, List, Any
    from core.params import ParamSpec  # type: ignore

    def freeze(specs: Iterable[ParamSpec], **_defaults: Any) -> List[ParamSpec]:
        """
        Legacy compatibility: returns specs as-is.
        (Older scripts pass DEFAULT_VALUES here; using them to literally "freeze"
        would collapse the search space and break optimization.)
        """
        return list(specs)
