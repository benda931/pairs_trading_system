# -*- coding: utf-8 -*-
"""
ranges.py (root shim)
---------------------
Backward-compat shim for legacy imports: `from ranges import RangeManager`.

Canonical implementation lives in `core.ranges`.
"""

from core.ranges import *  # noqa: F401,F403
