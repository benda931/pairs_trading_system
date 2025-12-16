# -*- coding: utf-8 -*-
"""
pair.py — Robust pairs universe loader (HF‑grade, v2)
------------------------------------------------------------
Purpose
- Load a *pairs universe* from `config.json` and/or external files (CSV/XLSX/HTTP URLs), with
  flexible schemas, validation, dedup, mapping, and filters.
- Return both a convenient list of `PairSpec` and a pandas DataFrame + rich metadata.
- Provide export helpers and cached I/O for performance.

What’s new in v2
- Multiple sources: `pairs_file` (single) **or** `pairs_files` (list, glob‑friendly). HTTP URLs supported for CSV.
- Env expansion & relative paths resolved from config dir.
- Optional symbol mapping (e.g., broker/API aliases) via `symbols_map` in config.
- Whitelist/blacklist filters; required tags; minimum score; drop reasons report.
- Stronger schema inference for A/B legs; support `pair` column with "A/B" literal.
- Caching: file/URL reads & config parse via `@st.cache_data`.
- Summary stats + easy export (`export_pairs_df`).
- Strict vs. permissive modes.

Config contract (config.json)
-----------------------------
Keys (all optional unless stated):
- `pairs_file`: str — path/URL to CSV/XLSX containing pairs (lower priority than `pairs_files` if both set)
- `pairs_files`: list[str] — multiple files or globs; merged together
- `pairs`: list[object] — inline items, e.g. {"symbols": ["QQQ","SOXX"], "weight_x": 1.0, ...}
- `symbols_map`: dict[str,str] — map raw→canonical tickers (applied after uppercase/trim)
- `allow_directional`: bool — default False; if True, (A,B) ≠ (B,A)
- `filters`: object with optional keys:
    - `whitelist`: list[str] — keep only pairs whose BOTH legs are in this set
    - `blacklist`: list[str] — drop any pair containing a leg in this set
    - `required_tags`: list[str] — pair must contain ALL these tags (case‑insensitive)
    - `min_score`: float — keep only pairs with score ≥ min_score
- `defaults`: object with optional keys: `weight_x`, `weight_y`

Usage
-----
    from pair import (
        load_pairs, load_pairs_df, load_pairs_universe,
        export_pairs_df, PairsMeta
    )
    pairs = load_pairs()                   # → List[PairSpec]
    pairs_df, meta = load_pairs_universe() # → (DataFrame, PairsMeta)

Notes
- Direction is *not* preserved by default: pair (A,B) ≡ (B,A). Set `allow_directional=True` to keep order.
- All tickers are upper‑cased and trimmed, then mapped via `symbols_map` if provided.
- Invalid/empty pairs are dropped with a reason; a drop report is returned in meta.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import os
import re

import pandas as pd
import streamlit as st

# ===================== Data Model =====================
@dataclass(frozen=True)
class PairSpec:
    A: str
    B: str
    weight_x: float = 1.0
    weight_y: float = -1.0
    score: Optional[float] = None
    tags: Tuple[str, ...] = ()
    source: str = "config"

    def key(self, *, allow_directional: bool = False) -> Tuple[str, str]:
        if allow_directional:
            return (self.A, self.B)
        return tuple(sorted((self.A, self.B)))  # type: ignore[return-value]

@dataclass
class PairsMeta:
    source_count: int
    pair_count_raw: int
    pair_count_clean: int
    dropped: List[Dict[str, Any]]
    sources: List[str]

# ===================== Helpers =====================
_POSS_A = ("symbol_x", "a", "leg_a", "sym1", "asset_a", "left", "x")
_POSS_B = ("symbol_y", "b", "leg_b", "sym2", "asset_b", "right", "y")

_DEF_COLS = {
    "weight_x": 1.0,
    "weight_y": -1.0,
    "score": None,
    "tags": None,
}

_DEF_CONFIG_PATH = "config.json"

# ---- caching wrappers ----
@st.cache_data(show_spinner=False)
def _cached_read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _cached_read_csv_url(url: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(url)
    except Exception:
        return None

# ---- utils ----
def _expand_env(s: str) -> str:
    return os.path.expanduser(os.path.expandvars(s))

def _make_abs(path_like: str, base_dir: Path) -> Path:
    p = Path(_expand_env(path_like))
    return p if p.is_absolute() else (base_dir / p).resolve()

# Try to parse tags that might be comma/semicolon separated
def _parse_tags(v: Any) -> Tuple[str, ...]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return tuple()
    if isinstance(v, (list, tuple)):
        return tuple(str(x).strip() for x in v if str(x).strip())
    s = str(v).replace(";", ",")
    return tuple(t.strip() for t in s.split(",") if t.strip())

# Normalize column names to lower
def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

# Attempt to detect A/B columns; also support single `pair` column like "QQQ/SOXX"
def _detect_leg_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = list(df.columns)
    if "pair" in cols:
        return None, None  # handled separately
    a = next((c for c in cols if c in _POSS_A), None)
    b = next((c for c in cols if c in _POSS_B), None)
    if not a or not b:
        raise ValueError("Could not detect pair columns (expected e.g., symbol_x/symbol_y or 'pair').")
    return a, b

def _canonicalize_symbol(s: Any) -> str:
    return str(s).strip().upper()

@st.cache_data(show_spinner=False)
def _read_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        st.error(f"⚠️ config not found: {p}")
        return {}
    try:
        txt = _cached_read_text(str(p))
        if txt is None:
            raise FileNotFoundError()
        return json.loads(txt)
    except Exception as e:
        st.error(f"⚠️ failed to parse config: {e}")
        return {}

@st.cache_data(show_spinner=False)
def _read_pairs_any(path_or_url: str) -> pd.DataFrame:
    """Read CSV/XLSX/local or HTTP CSV; returns lowercase columns."""
    s = path_or_url.strip()
    if re.match(r"^https?://", s, flags=re.IGNORECASE):
        df = _cached_read_csv_url(s)
        if df is None:
            raise ValueError(f"failed reading URL: {s}")
        return _lower_cols(df)
    p = Path(_expand_env(s))
    if not p.exists():
        raise FileNotFoundError(f"pairs_file not found: {p}")
    try:
        if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
            df = pd.read_excel(p)
        else:
            try:
                df = pd.read_csv(p)
            except UnicodeDecodeError:
                df = pd.read_csv(p, encoding="cp1255")
    except Exception as e:
        raise ValueError(f"failed reading pairs_file: {e}")
    return _lower_cols(df)

# Validate raw row → PairSpec (or None)
def _row_to_pair(row: pd.Series, a_col: Optional[str], b_col: Optional[str], *,
                 allow_directional: bool, source: str, defaults: Dict[str, Any]) -> Tuple[Optional[PairSpec], Optional[str]]:
    try:
        if a_col is None and b_col is None and "pair" in row.index:
            toks = str(row["pair"]).replace("\\", "/").split("/")
            if len(toks) >= 2:
                A = _canonicalize_symbol(toks[0])
                B = _canonicalize_symbol(toks[1])
            else:
                return None, "bad_pair_literal"
        else:
            A = _canonicalize_symbol(row[a_col]) if (a_col and a_col in row) else ""
            B = _canonicalize_symbol(row[b_col]) if (b_col and b_col in row) else ""
        if not A or not B:
            return None, "missing_leg"
        if A == B:
            return None, "same_legs"
        wx = float(row.get("weight_x", defaults.get("weight_x", _DEF_COLS["weight_x"])))
        wy = float(row.get("weight_y", defaults.get("weight_y", _DEF_COLS["weight_y"])))
        sc_raw = row.get("score", _DEF_COLS["score"])
        sc = None if sc_raw is None or (isinstance(sc_raw, float) and pd.isna(sc_raw)) else float(sc_raw)
        tg = _parse_tags(row.get("tags", _DEF_COLS["tags"]))
        return PairSpec(A=A, B=B, weight_x=float(wx), weight_y=float(wy), score=sc, tags=tg, source=source), None
    except Exception as e:
        return None, f"row_parse_error: {e}"

# Deduplicate and filter invalid
def _dedup_pairs(pairs: Iterable[PairSpec], *, allow_directional: bool) -> List[PairSpec]:
    seen: set[Tuple[str, str]] = set()
    out: List[PairSpec] = []
    for p in pairs:
        if not isinstance(p, PairSpec):
            continue
        if not p.A or not p.B or p.A == p.B:
            continue
        k = p.key(allow_directional=allow_directional)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out

# Apply symbol mapping
def _apply_symbols_map(pairs: List[PairSpec], symbols_map: Optional[Dict[str, str]]) -> List[PairSpec]:
    if not symbols_map:
        return pairs
    mapped = []
    for p in pairs:
        A = symbols_map.get(p.A, p.A)
        B = symbols_map.get(p.B, p.B)
        mapped.append(PairSpec(A=A, B=B, weight_x=p.weight_x, weight_y=p.weight_y, score=p.score, tags=p.tags, source=p.source))
    return mapped

# Filters
def _apply_filters(df: pd.DataFrame, cfg_filters: Optional[Dict[str, Any]]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    dropped: List[Dict[str, Any]] = []
    if df.empty:
        return df, dropped
    keep = pd.Series(True, index=df.index)
    if cfg_filters:
        wl = set([str(s).upper().strip() for s in cfg_filters.get("whitelist", [])])
        bl = set([str(s).upper().strip() for s in cfg_filters.get("blacklist", [])])
        req_tags = set([str(s).lower().strip() for s in cfg_filters.get("required_tags", [])])
        min_score = cfg_filters.get("min_score")
        if wl:
            wmask = df.apply(lambda r: (r["A"] in wl) and (r["B"] in wl), axis=1)
            dropped += [{"pair": f"{r.A}/{r.B}", "reason": "whitelist_exclude"} for _, r in df[~wmask].iterrows()]
            keep &= wmask
        if bl:
            bmask = df.apply(lambda r: (r["A"] in bl) or (r["B"] in bl), axis=1)
            dropped += [{"pair": f"{r.A}/{r.B}", "reason": "blacklist"} for _, r in df[bmask].iterrows()]
            keep &= ~bmask
        if req_tags:
            tmask = df["tags"].apply(lambda s: req_tags.issubset(set(str(s).lower().split(","))) if pd.notna(s) and str(s).strip() else False)
            dropped += [{"pair": f"{r.A}/{r.B}", "reason": "missing_required_tags"} for _, r in df[~tmask].iterrows()]
            keep &= tmask
        if isinstance(min_score, (int, float)):
            smask = df["score"].fillna(float('-inf')) >= float(min_score)
            dropped += [{"pair": f"{r.A}/{r.B}", "reason": "score_below_min"} for _, r in df[~smask].iterrows()]
            keep &= smask
    kept_df = df[keep].reset_index(drop=True)
    return kept_df, dropped

# ===================== Public API =====================
@st.cache_data(show_spinner=False)
def load_pairs(config_path: str = _DEF_CONFIG_PATH, *, allow_directional: Optional[bool] = None, strict: bool = False) -> List[PairSpec]:
    """Load pairs universe from config + files/URLs + inline.

    Priority: merge(`pairs_files`, `pairs_file`, `pairs`) → normalize → map → dedup → filters.
    `allow_directional`: if None → read from config; else override.
    `strict`: when True, raises on schema issues; otherwise logs and continues.
    """
    cfg = _read_config(config_path)
    base_dir = Path(config_path).resolve().parent

    defaults_cfg = cfg.get("defaults", {}) if isinstance(cfg.get("defaults"), dict) else {}
    allow_dir_cfg = bool(cfg.get("allow_directional", False))
    allow_directional = allow_dir_cfg if allow_directional is None else allow_directional

    # Collect sources
    sources: List[str] = []
    files: List[str] = []
    if isinstance(cfg.get("pairs_files"), list):
        for item in cfg["pairs_files"]:
            if any(ch in str(item) for ch in "*?["):
                # glob relative to config
                for p in (base_dir.glob(item) if not Path(str(item)).is_absolute() else Path("/").glob(item)):
                    files.append(str(p))
            else:
                files.append(str(_make_abs(str(item), base_dir)))
    elif isinstance(cfg.get("pairs_file"), str):
        files.append(str(_make_abs(cfg["pairs_file"], base_dir)))

    inline = cfg.get("pairs", [])
    symbols_map = cfg.get("symbols_map") if isinstance(cfg.get("symbols_map"), dict) else None

    # Read rows → PairSpec
    pairs: List[PairSpec] = []
    dropped: List[Dict[str, Any]] = []

    # File(s)/URL(s)
    for fp in files:
        try:
            df = _read_pairs_any(fp)
            a_col, b_col = _detect_leg_cols(df)
            for _, row in df.iterrows():
                p, reason = _row_to_pair(row, a_col, b_col, allow_directional=allow_directional, source=str(fp), defaults=defaults_cfg)
                if p is not None:
                    pairs.append(p)
                else:
                    dropped.append({"pair": None, "source": str(fp), "reason": reason})
            sources.append(str(fp))
        except Exception as e:
            msg = f"⚠️ {e}"
            if strict:
                raise
            st.error(msg)

    # Inline
    if isinstance(inline, list):
        for it in inline:
            try:
                if isinstance(it, dict):
                    if "symbols" in it and isinstance(it["symbols"], (list, tuple)) and len(it["symbols"]) >= 2:
                        A, B = _canonicalize_symbol(it["symbols"][0]), _canonicalize_symbol(it["symbols"][1])
                    else:
                        A = _canonicalize_symbol(it.get("A") or it.get("a") or it.get("symbol_x") or it.get("sym1"))
                        B = _canonicalize_symbol(it.get("B") or it.get("b") or it.get("symbol_y") or it.get("sym2"))
                    if A and B and A != B:
                        pairs.append(PairSpec(
                            A=A,
                            B=B,
                            weight_x=float(it.get("weight_x", defaults_cfg.get("weight_x", 1.0))),
                            weight_y=float(it.get("weight_y", defaults_cfg.get("weight_y", -1.0))),
                            score=(None if it.get("score") is None else float(it.get("score"))),
                            tags=_parse_tags(it.get("tags")),
                            source="inline",
                        ))
                    else:
                        dropped.append({"pair": None, "source": "inline", "reason": "missing_or_same_legs"})
            except Exception as e:
                if strict:
                    raise
                st.error(f"⚠️ bad inline pair item: {e}")
    elif inline:
        st.error("⚠️ `pairs` must be a list in config.json")

    pair_count_raw = len(pairs)

    # Mapping + dedup
    pairs = _apply_symbols_map(pairs, symbols_map)
    pairs = _dedup_pairs(pairs, allow_directional=allow_directional)

    # To DataFrame for filters/stats/export
    df = pd.DataFrame([
        {
            "A": p.A,
            "B": p.B,
            "weight_x": p.weight_x,
            "weight_y": p.weight_y,
            "score": p.score,
            "tags": ",".join(p.tags),
            "source": p.source,
        }
        for p in pairs
    ])

    # Filters
    df_filtered, dropped2 = _apply_filters(df, cfg.get("filters"))
    dropped += dropped2

    meta = PairsMeta(
        source_count=len(sources) + (1 if isinstance(inline, list) and inline else 0),
        pair_count_raw=pair_count_raw,
        pair_count_clean=int(df_filtered.shape[0]),
        dropped=dropped,
        sources=sources + (["inline"] if isinstance(inline, list) and inline else []),
    )

    # Return PairSpec list matching filtered DF
    if df_filtered.empty:
        st.warning("ℹ️ No pairs found after filters. Adjust your config.")
        return []
    out_pairs = [
        PairSpec(A=row.A, B=row.B, weight_x=float(row.weight_x), weight_y=float(row.weight_y),
                 score=(None if pd.isna(row.score) else float(row.score)),
                 tags=tuple(str(row.tags).split(",")) if str(row.tags).strip() else tuple(),
                 source=str(row.source))
        for row in df_filtered.itertuples(index=False)
    ]
    return out_pairs


@st.cache_data(show_spinner=False)
def load_pairs_df(config_path: str = _DEF_CONFIG_PATH, *, allow_directional: Optional[bool] = None, strict: bool = False) -> pd.DataFrame:
    """Same as `load_pairs` but returns a DataFrame."""
    pairs = load_pairs(config_path=config_path, allow_directional=allow_directional, strict=strict)
    if not pairs:
        return pd.DataFrame(columns=["A","B","weight_x","weight_y","score","tags","source"])
    return pd.DataFrame([
        {
            "A": p.A,
            "B": p.B,
            "weight_x": p.weight_x,
            "weight_y": p.weight_y,
            "score": p.score,
            "tags": ",".join(p.tags),
            "source": p.source,
        }
        for p in pairs
    ])


@st.cache_data(show_spinner=False)
def load_pairs_universe(config_path: str = _DEF_CONFIG_PATH, *, allow_directional: Optional[bool] = None, strict: bool = False) -> Tuple[pd.DataFrame, PairsMeta]:
    """Return (DataFrame, Meta). Useful when you want the drop report/sources."""
    cfg = _read_config(config_path)
    # Use loader to honor filters/mapping/dedup
    pairs = load_pairs(config_path=config_path, allow_directional=allow_directional, strict=strict)
    df = pd.DataFrame([
        {"A": p.A, "B": p.B, "weight_x": p.weight_x, "weight_y": p.weight_y, "score": p.score, "tags": ",".join(p.tags), "source": p.source}
        for p in pairs
    ])
    # Build meta synchronously here as well
    base_sources = []
    if isinstance(cfg.get("pairs_files"), list) and cfg.get("pairs_files"):
        base_sources += [str(s) for s in cfg["pairs_files"]]
    elif isinstance(cfg.get("pairs_file"), str):
        base_sources.append(str(cfg["pairs_file"]))
    meta = PairsMeta(
        source_count=len(base_sources) + (1 if isinstance(cfg.get("pairs"), list) and cfg.get("pairs") else 0),
        pair_count_raw=int(df.shape[0]),
        pair_count_clean=int(df.shape[0]),  # already filtered
        dropped=[],  # drops already accounted for in the inner call
        sources=base_sources + (["inline"] if isinstance(cfg.get("pairs"), list) and cfg.get("pairs") else []),
    )
    return df, meta


def export_pairs_df(df: pd.DataFrame, path: str | os.PathLike[str]) -> bool:
    """Export a pairs DataFrame to CSV (UTF‑8). Returns True on success."""
    try:
        p = Path(_expand_env(str(path)))
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False, encoding="utf-8")
        return True
    except Exception as e:
        st.error(f"⚠️ export failed: {e}")
        return False


# ===================== Integration helpers (Ready-to-Trade tab) =====================
@st.cache_data(show_spinner=False)
def get_universe_symbols(config_path: str = _DEF_CONFIG_PATH, *, allow_directional: Optional[bool] = None, strict: bool = False) -> List[str]:
    """Return a **sorted unique** list of symbols appearing in A/B legs **after** filters/mapping/dedup.
    Use this to feed your prices loader / scan tab.
    """
    df = load_pairs_df(config_path=config_path, allow_directional=allow_directional, strict=strict)
    if df.empty:
        return []
    syms = set(df["A"].astype(str).str.upper().str.strip()) | set(df["B"].astype(str).str.upper().str.strip())
    return sorted(s for s in syms if s)


def sync_universe_to_session_state(config_path: str = _DEF_CONFIG_PATH, *, key: str = "universe", allow_directional: Optional[bool] = None, strict: bool = False) -> List[str]:
    """Load universe from config and store to st.session_state[key]. Returns the list."""
    syms = get_universe_symbols(config_path=config_path, allow_directional=allow_directional, strict=strict)
    st.session_state[key] = syms
    return syms

# ===================== Tiny UI helper (optional) =====================
if __name__ == "__main__":  # quick manual test in Streamlit REPL
    st.title("Pairs Loader — Quick Test (v2)")
    cfg_path = st.text_input("config.json path", value=_DEF_CONFIG_PATH, help="Paths & env are resolved from this file's directory")
    directional = st.checkbox("Allow directional duplicates (A/B ≠ B/A)", value=False)
    strict = st.checkbox("Strict mode (raise on errors)", value=False)

    df, meta = load_pairs_universe(cfg_path, allow_directional=directional, strict=strict)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.dataframe(df, use_container_width=True)
    with c2:
        st.markdown("**Summary**")
        st.write({
            "sources": meta.sources,
            "source_count": meta.source_count,
            "pairs_clean": meta.pair_count_clean,
        })
    if st.button("Export CSV", key="pairs_export"):
        ok = export_pairs_df(df, "./exports/pairs_export.csv")
        if ok:
            st.success("Exported → ./exports/pairs_export.csv")
