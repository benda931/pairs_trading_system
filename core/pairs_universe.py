# -*- coding: utf-8 -*-
"""
core/pairs_universe.py — Unified management for pairs JSON universes (HF-grade)
===============================================================================

מודול מרכזי לניהול "Universe" של זוגות (Pairs) בקבצי JSON:

פורמט(ים) נתמכים ב-JSON
------------------------
המודול יודע לקרוא מספר פורמטים נפוצים, כולם מנורמלים ל-PairRecord:

1) פורמט בסיסי (היסטורי במערכת שלך):
    [
      { "symbols": ["XLY", "XLC"] },
      { "symbols": ["QQQ", "SPY"] },
      ...
    ]

2) פורמטים גמישים יותר:
    [
      { "sym_x": "XLY", "sym_y": "XLP", "category": "sector", "enabled": true,
        "tags": ["us", "core"], "score_hint": 0.83 },
      { "pair": "QQQ-SPY", "category": "index", "enabled": false },
      ...
    ]

יכולות עיקריות
---------------
- טעינת קובץ אחד או כמה קבצים (עם merge ו-dedup אוטומטי).
- נירמול symbols (strip + upper) וכלל key סימטרי (XLY/XLP == XLP/XLY).
- הזרקת קטגוריה אוטומטית לפי טיקר (sector / index / fixed_income / commodity / crypto / other).
- מודל PairRecord עשיר:
    * symbol_1, symbol_2
    * category, source, enabled
    * tags (tuple[str])
    * region (us / eu / apac / em / other)
    * score_hint / priority / notes (אופציונלי לתיעדוף)
- פילטרים:
    * לפי סימבול, קטגוריה, enabled, tag, רשימת סימבולים, רשימת זוגות.
- describe_universe:
    * total
    * by_category
    * by_source
    * by_tag
    * symbol_usage (top symbols, כמה פעמים כל סימבול מופיע בזוגות)
- API שמירה חזרה:
    * save_pairs_universe(...) — כתיבה ל-JSON בצורה נקייה, כולל metadata.

הערה:
------
אין שום "state" חבוי — הכל חוזר בהחזרי פונקציות. השמירה לקובץ היא פעולה מפורשת.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from collections import Counter

from common.helpers import read_json  # read_json כבר קיים אצלך ב-helpers
try:
    # שמירה ל-JSON (אם קיימת במערכת שלך)
    from common.helpers import write_json  # type: ignore
except Exception:  # pragma: no cover
    write_json = None  # type: ignore


# =============================================================================
# Dataclass לזוגים עשירים (PairRecord)
# =============================================================================


@dataclass(frozen=True)
class PairRecord:
    """
    ייצוג עשיר לזוג:

    Attributes
    ----------
    symbol_1, symbol_2 : str
        שמות הסימבולים (בדרך כלל ETFs / סקטורים / אינדקסים / נכסים אחרים).
    category : str
        קטגוריית על בסיסית:
            "sector"       — סקטור (XLY, XLP, XLE...)
            "index"        — מדדי מניות רחבים (SPY, QQQ, DIA...)
            "fixed_income" — אג"ח/קרן מחקה אג"ח (LQD, HYG...)
            "commodity"    — סחורות/זהב/אנרגיה (GDX, XLE, OIH...)
            "crypto"       — קרנות/ETN קריפטו (IBIT, ETHA, BITO...)
            "other"        — כל השאר.
    source : str | None
        מאיזה קובץ / מקור הגיע הזוג (לוגית בלבד, לעקיבות).
    enabled : bool
        האם הזוג פעיל (True) או כבוי (False). פילטרים כברירת מחדל עובדים רק על enabled.
    tags : tuple[str, ...]
        תוויות חופשיות (למשל ("us", "sector", "core", "largecap")).
    region : str | None
        איזור גיאוגרפי משוער ("us" / "eu" / "apac" / "em" / "other").
    score_hint : float | None
        רמז/הערכת ציון (0–1 או כל סקאלה אחרת) לשימוש באופטימיזציה/סינון.
        לא חובה — אם חסר, המערכת תתייחס כ-"unknown".
    priority : int | None
        עדיפות משוערת (1=גבוה, 10=נמוך). שימושי לרשימות Top.
    notes : str | None
        הערות חופשיות (למשל "core book", "experimental", "to_review").
    """

    symbol_1: str
    symbol_2: str
    category: str = "other"
    source: Optional[str] = None
    enabled: bool = True
    tags: Tuple[str, ...] = ()

    # שדות מטא נוספים (לא חובה)
    region: Optional[str] = None
    score_hint: Optional[float] = None
    priority: Optional[int] = None
    notes: Optional[str] = None

    # -------------------------
    # Helper properties / APIs
    # -------------------------

    @property
    def key(self) -> Tuple[str, str]:
        """
        Key קנוני לזוג (sorted), כדי לאכוף סימטריה:
        ("XLY","XLP") ו-("XLP","XLY") יקבלו את אותו key.
        """
        a, b = sorted((self.symbol_1, self.symbol_2))
        return (a, b)

    def to_tuple(self) -> Tuple[str, str]:
        """
        מחזיר (symbol_1, symbol_2) בסדר המוגדר (לא בהכרח ממויין).
        """
        return (self.symbol_1, self.symbol_2)

    def to_dict(self) -> Dict[str, Any]:
        """
        ממיר את הרשומה ל-dict (נוח לשמירה/לוג/JSON).
        """
        return asdict(self)


# =============================================================================
# בסיס מידע: סימבול → קטגוריה / region
# =============================================================================

_SYMBOL_CATEGORY: Dict[str, str] = {
    # US sector ETFs (SPDR)
    "XLY": "sector",
    "XLC": "sector",
    "XLP": "sector",
    "XLI": "sector",
    "XLK": "sector",
    "XLV": "sector",
    "XLB": "sector",
    "XLRE": "sector",
    "XRT": "sector",
    "XLU": "sector",
    "XBI": "sector",
    "XHB": "sector",
    "XME": "sector",
    "XOP": "sector",
    "XLF": "sector",

    "KBE": "sector",
    "VGT": "sector",
    "VHT": "sector",
    "VPU": "sector",
    "VNQ": "sector",
    "IBB": "sector",

    # Broad equity indexes
    "QQQ": "index",
    "SPY": "index",
    "DIA": "index",
    "IVV": "index",
    "VOOG": "index",
    "VTI": "index",
    "ITOT": "index",
    "VEA": "index",
    "IEFA": "index",
    "VEU": "index",
    "VWO": "index",
    "EEM": "index",

    # Fixed income
    "LQD": "fixed_income",
    "IGIB": "fixed_income",
    "HYG": "fixed_income",
    "JNK": "fixed_income",
    "TIP": "fixed_income",
    "IEF": "fixed_income",
    "BND": "fixed_income",
    "AGG": "fixed_income",
    "MBB": "fixed_income",
    "TLT": "fixed_income",
    "SHY": "fixed_income",

    # Commodities / metals / energy proxies
    "GDX": "commodity",
    "GDXJ": "commodity",
    "PALL": "commodity",
    "OIH": "commodity",
    "XLE": "commodity",  # energy sector ETF, משמש גם כ-proxy לסחורות
    "GLD": "commodity",
    "SLV": "commodity",
    "USO": "commodity",

    # Crypto / crypto-proxies
    "IBIT": "crypto",   # Bitcoin ETF
    "BITO": "crypto",
    "ETHA": "crypto",   # Ethereum ETF/ETN כלשהו
}


_SYMBOL_REGION: Dict[str, str] = {
    # US
    "XLY": "us", "XLC": "us", "XLP": "us", "XLI": "us", "XLK": "us", "XLV": "us",
    "XLB": "us", "XLRE": "us", "XRT": "us", "XLU": "us",
    "QQQ": "us", "SPY": "us", "DIA": "us", "IVV": "us", "VTI": "us", "ITOT": "us",
    "LQD": "us", "HYG": "us", "JNK": "us", "TLT": "us", "SHY": "us",
    "IBIT": "us", "BITO": "us", "ETHA": "us",

    # Europe / global ישירים
    "VEA": "global",
    "VEU": "global",
    "IEFA": "global",
    "VWO": "em",
    "EEM": "em",
}


# =============================================================================
# Normalization helpers
# =============================================================================

def _normalize_symbol(sym: str) -> str:
    """ניקוי בסיסי לסימבול: strip + upper."""
    return str(sym).strip().upper()


def _normalize_tags(raw_tags: Any) -> Tuple[str, ...]:
    """
    מנרמל tags לכלל:
        - iterable / רשימה / tuple של מחרוזות → tuple[str.lower()].
        - ערכים לא-תקינים → נתעלם.
    """
    if raw_tags is None:
        return ()
    out: List[str] = []
    if isinstance(raw_tags, (list, tuple, set)):
        candidates = raw_tags
    else:
        # במקרה ומישהו הכניס "equity,us,core"
        candidates = str(raw_tags).split(",")

    for t in candidates:
        s = str(t).strip()
        if not s:
            continue
        out.append(s.lower())
    return tuple(out)


def _infer_category(sym1: str, sym2: str) -> str:
    """הסקת קטגוריה בסיסית לפי שני הסימבולים (מפה בסיסית + fallbacks)."""
    s1 = _normalize_symbol(sym1)
    s2 = _normalize_symbol(sym2)
    k1 = _SYMBOL_CATEGORY.get(s1, "other")
    k2 = _SYMBOL_CATEGORY.get(s2, "other")
    if k1 == k2:
        return k1
    # אם אחת הקטגוריות index/sector והשנייה other → עדיין נעדיף את הלא-other
    favored = {"sector", "index", "fixed_income", "commodity", "crypto"}
    if k1 in favored and k2 == "other":
        return k1
    if k2 in favored and k1 == "other":
        return k2
    return "other"


def _infer_region(sym1: str, sym2: str) -> Optional[str]:
    """הסקת region משוערת לפי הסימבולים (אם לא ניתן – מחזירים None)."""
    s1 = _normalize_symbol(sym1)
    s2 = _normalize_symbol(sym2)
    r1 = _SYMBOL_REGION.get(s1)
    r2 = _SYMBOL_REGION.get(s2)
    if r1 and r2:
        if r1 == r2:
            return r1
        # אם שונים – נעדיף global/em אם קיימים
        for candidate in ("global", "us", "eu", "apac", "em"):
            if r1 == candidate or r2 == candidate:
                return candidate
    return r1 or r2


# =============================================================================
# Loading single file → List[PairRecord]
# =============================================================================

def _extract_symbols_from_item(item: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """
    מנסה לחלץ (sym1, sym2) מתוך רשומת JSON אחת בפורמטים שונים:

        {"symbols": ["XLY","XLC"]}
        {"sym_x": "XLY", "sym_y": "XLC"}
        {"pair": "XLY-XLC"}  או  {"pair": "XLY/XLC"}

    מחזיר None אם אין זוג תקין.
    """
    # פורמט 1: symbols = ["A", "B"]
    syms = item.get("symbols")
    if isinstance(syms, (list, tuple)) and len(syms) == 2:
        a, b = syms
        return _normalize_symbol(a), _normalize_symbol(b)

    # פורמט 2: sym_x / sym_y
    sx = item.get("sym_x") or item.get("symbol_x")
    sy = item.get("sym_y") or item.get("symbol_y")
    if sx and sy:
        return _normalize_symbol(str(sx)), _normalize_symbol(str(sy))

    # פורמט 3: pair="XLY-XLC" או "XLY/XLC"
    pair_str = item.get("pair") or item.get("pair_id")
    if isinstance(pair_str, str) and ("-" in pair_str or "/" in pair_str or "_" in pair_str):
        for sep in ("-", "/", "_"):
            if sep in pair_str:
                a, b = pair_str.split(sep, 1)
                return _normalize_symbol(a), _normalize_symbol(b)

    return None


def _load_pairs_file(path: Path, source_label: Optional[str] = None) -> List[PairRecord]:
    """
    טוען קובץ JSON בפורמט "Universe זוגות" ומחזיר רשימת PairRecord.

    Parameters
    ----------
    path : Path
        נתיב לקובץ JSON.
    source_label : str | None
        תווית מקור שתירשם בשדה `source` (ברירת מחדל — שם הקובץ עצמו).

    Returns
    -------
    List[PairRecord]
    """
    raw = read_json(path)
    if not isinstance(raw, list):
        raise TypeError(f"{path} must contain a JSON list, got {type(raw)}")

    src = source_label or path.name

    records: List[PairRecord] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue

        syms = _extract_symbols_from_item(item)
        if not syms:
            continue

        sa, sb = syms
        if not sa or not sb or sa == sb:
            continue

        # category
        category = str(item.get("category") or _infer_category(sa, sb))

        # enabled
        enabled_val = item.get("enabled", True)
        enabled = bool(enabled_val)

        # tags
        tags = _normalize_tags(item.get("tags"))

        # region
        region = item.get("region")
        if region is None:
            region = _infer_region(sa, sb)
        if isinstance(region, str):
            region = region.lower()

        # score_hint / priority / notes (אם קיימים באובייקט)
        score_hint = item.get("score_hint")
        try:
            score_hint_f: Optional[float]
            score_hint_f = float(score_hint) if score_hint is not None else None
        except Exception:
            score_hint_f = None

        priority = item.get("priority")
        try:
            priority_i: Optional[int]
            priority_i = int(priority) if priority is not None else None
        except Exception:
            priority_i = None

        notes = item.get("notes")
        if notes is not None:
            notes = str(notes)

        rec = PairRecord(
            symbol_1=sa,
            symbol_2=sb,
            category=category,
            source=src,
            enabled=enabled,
            tags=tags,
            region=region,
            score_hint=score_hint_f,
            priority=priority_i,
            notes=notes,
        )
        records.append(rec)

    return records


def _dedup_records(records: Iterable[PairRecord]) -> List[PairRecord]:
    """
    מסיר כפילויות לפי PairRecord.key (זוג sorted).

    אם יש כמה רשומות עם אותו key (מכמה קבצים), נשמור רק את *הראשונה*.
    מי שרוצה מיזוג חכם של tags וכו' יכול להשתמש ב-helper ייעודי (בעתיד).
    """
    seen = set()
    out: List[PairRecord] = []
    for rec in records:
        key = rec.key
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


# =============================================================================
# Public API — טעינת זוגות
# =============================================================================

def load_pairs_universe(
    base_dir: Path | None = None,
    filenames: Sequence[str] | None = None,
) -> List[PairRecord]:
    """
    טוען אחד או יותר קבצי JSON של זוגות, מאחד ומסיר כפילויות.

    Parameters
    ----------
    base_dir : Path | None
        תיקייה שבה יושבים הקבצים. ברירת מחדל:
            project_root/config  (כלומר core/../config).
    filenames : sequence[str] | None
        רשימת שמות קבצים (יחסית ל-base_dir).
        אם None → ברירת מחדל ["pairs_universe.json"] בלבד (שימור התנהגות קיימת).

    Returns
    -------
    List[PairRecord]
        רשימת PairRecord מאוחדת, מנורמלת ובלי כפילויות.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent / "config"

    if filenames is None:
        filenames = ["pairs_universe.json"]

    all_records: List[PairRecord] = []

    for name in filenames:
        path = base_dir / name
        if not path.exists():
            continue
        recs = _load_pairs_file(path, source_label=name)
        all_records.extend(recs)

    return _dedup_records(all_records)


def save_pairs_universe(
    records: Sequence[PairRecord],
    path: Path,
    *,
    pretty: bool = True,
) -> None:
    """
    שומר רשימת PairRecord לקובץ JSON.

    הפורמט הנכתב:
        [
          {
            "symbols": ["XLY", "XLC"],
            "category": "sector",
            "enabled": true,
            "tags": ["us", "core"],
            "region": "us",
            "score_hint": 0.83,
            "priority": 1,
            "source": "pairs_universe.json",
            "notes": "optional"
          },
          ...
        ]

    אם write_json קיים ב-common.helpers → נשתמש בו;
    אחרת נ fallback ל-path.write_text(JSON).
    """
    import json

    data: List[Dict[str, Any]] = []
    for r in records:
        d = r.to_dict()
        # נשמור בפורמט "symbols" + metadata סביב
        symbols = [d.pop("symbol_1"), d.pop("symbol_2")]
        item: Dict[str, Any] = {"symbols": symbols}
        # ננקה tags ריקות/None
        if d.get("tags"):
            item["tags"] = list(d["tags"])
        # שער שדות אחרים
        for key in ("category", "source", "enabled", "region", "score_hint", "priority", "notes"):
            if d.get(key) is not None:
                item[key] = d[key]
        data.append(item)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if write_json is not None:
        write_json(path, data)  # type: ignore[arg-type]
    else:  # pragma: no cover
        path.write_text(
            json.dumps(data, indent=2 if pretty else None, sort_keys=False),
            encoding="utf-8",
        )


def as_symbol_pairs(
    records: Sequence[PairRecord],
    *,
    only_enabled: bool = True,
) -> List[Tuple[str, str]]:
    """
    ממיר רשימת PairRecord ל-[("XLY","XLC"), ...].

    Parameters
    ----------
    records : sequence[PairRecord]
    only_enabled : bool, default True
        אם True → יחזיר רק זוגות enabled=True.

    Returns
    -------
    List[Tuple[str, str]]
    """
    out: List[Tuple[str, str]] = []
    for r in records:
        if only_enabled and not r.enabled:
            continue
        out.append(r.to_tuple())
    return out


# =============================================================================
# Filters
# =============================================================================

def filter_by_symbol(
    records: Sequence[PairRecord],
    symbol: str,
) -> List[PairRecord]:
    """
    מחזיר את כל הזוגות שבהם symbol מופיע כ-symbol_1 או symbol_2.
    """
    s = _normalize_symbol(symbol)
    return [r for r in records if r.symbol_1 == s or r.symbol_2 == s]


def filter_by_category(
    records: Sequence[PairRecord],
    category: str,
) -> List[PairRecord]:
    """מסנן את הזוגות לרק אלה שהקטגוריה שלהם == category."""
    cat = str(category)
    return [r for r in records if r.category == cat]


def filter_enabled(
    records: Sequence[PairRecord],
    enabled: bool = True,
) -> List[PairRecord]:
    """מסנן לפי enabled: True/False."""
    return [r for r in records if r.enabled == enabled]


def filter_by_tag(
    records: Sequence[PairRecord],
    tag: str,
) -> List[PairRecord]:
    """
    מסנן לזוגות שה-tag המבוקש נמצא ב-tags.

    tag מושווה ברמת lower-case.
    """
    t = str(tag).strip().lower()
    if not t:
        return list(records)
    return [r for r in records if t in (tt.lower() for tt in r.tags)]


def filter_by_any_symbol_in(
    records: Sequence[PairRecord],
    symbols: Iterable[str],
) -> List[PairRecord]:
    """
    מחזיר זוגות שבהם לפחות אחד מהסימבולים נמצא בסט שניתן.

    שימושי למשל:
        filter_by_any_symbol_in(records, {"SPY","QQQ","XLY"})
    """
    norm = {_normalize_symbol(s) for s in symbols}
    return [
        r for r in records
        if r.symbol_1 in norm or r.symbol_2 in norm
    ]


def filter_by_pair_list(
    records: Sequence[PairRecord],
    pairs: Iterable[Tuple[str, str]],
) -> List[PairRecord]:
    """
    מסנן לזוגות שה-key שלהם מופיע ברשימת pairs שניתנה.

    pairs יכול להיות:
        [("XLY","XLP"), ("QQQ","SPY"), ...]
    הסדר אינו משנה (נשתמש ב-Key הממויין).
    """
    key_set = {tuple(sorted((_normalize_symbol(a), _normalize_symbol(b)))) for a, b in pairs}
    return [r for r in records if r.key in key_set]


# =============================================================================
# Debug & analytics helpers
# =============================================================================

def describe_universe(
    records: Sequence[PairRecord],
) -> Dict[str, Any]:
    """
    מחזיר summary מהיר על Universe הזוגות:

        {
          "total": int,
          "by_category": {"sector": X, "index": Y, ...},
          "by_source": {"pairs_universe.json": Z, ...},
          "by_tag": {"us": ..., "core": ...},
          "symbol_usage": {
              "XLY": 4,
              "SPY": 10,
              ...
          }
        }

    הערה:
    -----
    keys הקיימים היסטורית ("total", "by_category", "by_source")
    נשמרו לצורך תאימות. פשוט הוספנו מידע נוסף.
    """
    total = len(records)
    by_category: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    tag_counter: Counter[str] = Counter()
    symbol_counter: Counter[str] = Counter()

    for r in records:
        by_category[r.category] = by_category.get(r.category, 0) + 1

        src = r.source or "<unknown>"
        by_source[src] = by_source.get(src, 0) + 1

        for t in r.tags:
            if t:
                tag_counter[t.lower()] += 1

        symbol_counter[r.symbol_1] += 1
        symbol_counter[r.symbol_2] += 1

    by_tag = dict(tag_counter)
    symbol_usage = dict(symbol_counter)

    return {
        "total": total,
        "by_category": by_category,
        "by_source": by_source,
        "by_tag": by_tag,
        "symbol_usage": symbol_usage,
    }
