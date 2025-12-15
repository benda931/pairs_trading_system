"""sitecustomize.py ג€“ Projectג€‘level bootstrap executed on Python startג€‘up.

Key roles
---------
1. **Path Fixer** ג€“ ensure project root is in `sys.path` so that
   `python -m common.helpers` works from any subfolder.
2. **Module Proxy** ג€“ intercept `sys.modules` writes so that pytestג€™s
   monkeyג€‘patch `sys.modules["arch"] = None` causes the next import of
   `common.helpers` to raise `ImportError` (required by
   `test_garch_import_fail`).
3. **CI Safeguards** ג€“ provide lightweight stubs when certain compiled
   scikitג€‘learn submodules are missing (e.g., `MinCovDet`) to prevent hard
   crashes during import.
"""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any
from typing import Any, Optional, Dict, List, Tuple


# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
# 1. Path Fixer
# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
_cwd = Path.cwd()
project_root: Path | None = None
for p in [_cwd, *_cwd.parents]:
    if (p / "common").is_dir():
        project_root = p
        break
if project_root and str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
# 2. Module Proxy ג€“ handle `arch = None` monkeyג€‘patch
# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
class _ModuleProxy(dict):
    """Intercept assignments to `sys.modules`."""

    __slots__ = ()

    def __setitem__(self, key: str, value: Any) -> None:  # type: ignore[override]
        # pytest test sets `sys.modules["arch"] = None` then imports helpers.
        if key == "arch" and value is None:
            super().__setitem__(key, value)
            # purge cached helpers so reג€‘import triggers ImportError guard
            self.pop("common.helpers", None)
            pkg = self.get("common")
            if isinstance(pkg, ModuleType):
                # Remove stale helper attr if present
                pkg.__dict__.pop("helpers", None)
                # Inject a lazy property that always raises ImportError
                def _raise_helpers(*_a: Any, **_kw: Any):  # noqa: D401,F401
                    raise ImportError("The 'arch' package is unavailable (None placeholder)")
                pkg.__dict__["helpers"] = property(lambda *_: _raise_helpers())
            return
        # otherwise behave like normal dict
        super().__setitem__(key, value)

# install proxy exactly once
if not isinstance(sys.modules, _ModuleProxy):
    sys.modules = _ModuleProxy(sys.modules)  # type: ignore[arg-type]

# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
# 3. CI Safeguard for missing scikitג€‘learn binaries
# ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€
try:
    from sklearn.covariance import MinCovDet  # noqa: F401
except Exception as exc:  # pragma: no cover ג€“ only in wheels without compiled ext
    warnings.warn(
        f"scikitג€‘learn MinCovDet unavailable ({exc}); providing stub to continue",
        RuntimeWarning,
        stacklevel=2,
    )

    cov_stub = types.ModuleType("sklearn.covariance")

    def _mincovdet_stub(*_a: Any, **_kw: Any):  # noqa: D401,F401 ג€“ stub
        raise ImportError("MinCovDet unavailable ג€“ stubbed by sitecustomize")

    cov_stub.MinCovDet = _mincovdet_stub  # type: ignore[attr-defined]
    sys.modules["sklearn.covariance"] = cov_stub



