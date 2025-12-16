# -*- coding: utf-8 -*-
"""
common/typing_compat.py — Typing compatibility & shared aliases
===============================================================

מטרות המודול
------------
- נקודת אמת אחת לכל טיפוסי ה־typing החוזרים במערכת.
- תאימות בין גרסאות פייתון שונות (collections.abc vs typing).
- טיפוסים נוחים ל־JSON, Path, Numpy/Pandas.
- שימוש ב־make_json_safe כדי להבטיח אובייקטים ניתנים לסיריאליזציית JSON.

שימוש מומלץ
-----------
    from common.typing_compat import (
        Sequence, Mapping, MutableMapping, Iterable,
        JSONValue, JSONLike, StrPath, NDArray, PDDataFrame, PDSeries,
        to_jsonable, json_default,
    )
"""

from __future__ import annotations

from typing import (
    Any,
    TYPE_CHECKING,
    Dict,
    Hashable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from common.json_safe import make_json_safe, json_default as _json_default

# ============================================================================
# collections.abc / typing compatibility
# ============================================================================

try:
    # מודרני – עדיף collections.abc
    from collections.abc import Sequence, Mapping, MutableMapping, Iterable
except Exception:
    # fallback אם משהו מוזר בסביבה
    from typing import Sequence, Mapping, MutableMapping, Iterable  # type: ignore[misc]


# ============================================================================
# JSON typing helpers
# ============================================================================

JSONScalar = Union[str, int, float, bool, None]

# ערך JSON רקורסיבי (סקאלר / רשימה / dict מקונן)
JSONValue = Union[
    JSONScalar,
    Sequence["JSONValue"],
    Mapping[str, "JSONValue"],
]

# משהו "דמוי JSON" לפני ניקוי
JSONLike = Union[
    Mapping[str, Any],
    Sequence[Any],
]

T = TypeVar("T")


def to_jsonable(obj: Any) -> JSONValue:
    """
    הפיכת אובייקט לצורה בטוחה ל־JSON באמצעות make_json_safe.

    Returns
    -------
    JSONValue
        ייצוג נקי שניתן לסיריאליזציה ע"י json.dumps.
    """
    safe = make_json_safe(obj)
    return safe  # type: ignore[return-value]


# ============================================================================
# Path helpers
# ============================================================================

try:
    from os import PathLike
except Exception:  # pragma: no cover
    class PathLike:  # type: ignore[override]
        """Fallback placeholder אם os.PathLike לא קיים."""
        pass

StrPath = Union[str, "PathLike[str]"]


# ============================================================================
# Numpy / Pandas typing (מותאם ל-Pylance / mypy)
# ============================================================================

if TYPE_CHECKING:
    # בזמן type-check – משתמשים בטיפוסים האמיתיים
    import numpy as np  # type: ignore[import]
    from numpy.typing import NDArray  # type: ignore[import]

    import pandas as pd  # type: ignore[import]
    PDDataFrame = pd.DataFrame
    PDSeries = pd.Series

else:
    # בזמן ריצה – מנסים לייבא בעדינות, בלי לשבור אם חסר
    try:  # pragma: no cover
        import numpy as np  # type: ignore[import]
    except Exception:  # pragma: no cover
        np = object  # type: ignore[assignment]

    class NDArray:  # type: ignore[override]
        """
        Runtime placeholder עבור numpy.typing.NDArray.

        בפועל, ה־type checker רואה את ההגדרה ב־TYPE_CHECKING למעלה,
        אז כאן זה רק כדי שלא יהיו ImportError בזמן ריצה.
        """
        pass

    try:  # pragma: no cover
        import pandas as pd  # type: ignore[import]
        PDDataFrame = pd.DataFrame
        PDSeries = pd.Series
    except Exception:  # pragma: no cover
        pd = object  # type: ignore[assignment]
        PDDataFrame = object
        PDSeries = object


# ============================================================================
# Misc helpers & aliases
# ============================================================================

HashableDict = Mapping[Hashable, Any]
StrDict = Mapping[str, Any]
StrAnyMapping = Mapping[str, Any]
StrAnyMutableMapping = MutableMapping[str, Any]


def json_default(obj: Any) -> Any:
    """
    Wrapper סביב `_json_default` כדי לחשוף אותו ממקום אחד בלבד.

    ניתן להשתמש כ־default=json_default ב־json.dumps.
    """
    return _json_default(obj)


__all__ = [
    # collections-like
    "Sequence",
    "Mapping",
    "MutableMapping",
    "Iterable",
    # JSON
    "JSONScalar",
    "JSONValue",
    "JSONLike",
    "to_jsonable",
    "json_default",
    # Path
    "StrPath",
    # Numpy / Pandas
    "NDArray",
    "PDDataFrame",
    "PDSeries",
    # Misc
    "HashableDict",
    "StrDict",
    "StrAnyMapping",
    "StrAnyMutableMapping",
    # generics
    "T",
]
