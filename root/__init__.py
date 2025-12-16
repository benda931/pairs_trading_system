# root/__init__.py — package bootstrapper (v2)
"""root package initializer.

This module makes it easier to write imports in the app by:

- Allowing `from root.utils import ...` to transparently resolve to
  :mod:`common.utils`.
- Doing the same for :mod:`common.data_loader` and :mod:`common.config_manager`.
- Avoiding duplicate module objects: if a module already exists in
  :data:`sys.modules`, the same instance is reused.

The goal is to keep imports clean and robust even when running the project
from different working directories (IDE, CLI, Streamlit, tests, etc.).
"""
from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Dict, Iterable

# Mapping of logical aliases under `root.` → real modules under `common.`
#
# Example:
#     ALIASES = {"utils": "common.utils"}
#     import root.utils  → actually loads common.utils
#
ALIASES: Dict[str, str] = {
    "utils": "common.utils",
    "data_loader": "common.data_loader",
    "config_manager": "common.config_manager",
}

__all__ = list(ALIASES.keys())


def _ensure_alias(alias: str, target_path: str) -> ModuleType:
    """Ensure that ``root.<alias>`` points to the target module.

    If ``target_path`` is already imported, we reuse the same module object.
    Otherwise we import it once and register it under both names:

    - ``common.<name>`` (its real home)
    - ``root.<alias>``  (our convenience alias)
    """
    # Import or reuse the *real* target module
    target_mod = sys.modules.get(target_path)
    if target_mod is None:
        target_mod = import_module(target_path)

    if not isinstance(target_mod, ModuleType):  # defensive
        raise TypeError(f"Target {target_path!r} is not a module: {type(target_mod)!r}")

    # Register an alias under root.<alias> if not already present
    alias_name = f"root.{alias}"
    sys.modules.setdefault(alias_name, target_mod)
    return target_mod


def bootstrap_root_aliases(names: Iterable[str] | None = None) -> None:
    """Populate ``sys.modules`` with ``root.<alias>`` → ``common.<...>`` mappings.

    Parameters
    ----------
    names : Iterable[str] | None
        Optional subset of aliases to register. If ``None``, all aliases in
        :data:`ALIASES` are processed.
    """
    if names is None:
        to_register = ALIASES.items()
    else:
        to_register = ((n, ALIASES[n]) for n in names if n in ALIASES)

    for alias, target_path in to_register:
        try:
            _ensure_alias(alias, target_path)
        except Exception as exc:  # best-effort only
            # We avoid raising at import time; callers can still import common.*
            # directly if needed.
            sys.stderr.write(
                f"[root.__init__] Failed to alias root.{alias} → {target_path}: {exc}\n"
            )


# Run bootstrap at import time for the default aliases.
bootstrap_root_aliases()
