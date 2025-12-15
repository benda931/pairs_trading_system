#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re, sys
from pathlib import Path

# ------- Patterns -------
DUP_BLOCKS = [
    r"(?ms)^\s*#\s*Use Top-?K quantile suggestion[\s\S]*?(?=^\s*#|\Z)",
    r"(?ms)^\s*#\s*Use Top-?K Quantile Ranges as Active[\s\S]*?(?=^\s*#|\Z)",
    r"(?ms)^\s*#\s*Correlation Heatmap[\s\S]*?(?=^\s*#|\Z)",
    r"(?ms)^\s*#\s*Contour surface\s*\(mean Score\)[\s\S]*?(?=^\s*#|\Z)",
    r"(?ms)^\s*#\s*Pareto scatter[\s\S]*?(?=^\s*#|\Z)",
    r"(?ms)^\s*#\s*Distributions\s*$[\s\S]*?(?=^\s*#|\Z)",
]

SIDEBAR_PARAM_MAPPING = r"""(?ms)^\s*with\s+st\.sidebar:\s*[\s\S]*?with\s+st\.expander\("Param Mapping \(UI ג†’ Backtester\)".*?st\.success\("Mapping cleared"\)\s*\)\s*"""

# st.dataframe(...) without height=
DATAFRAME_NO_HEIGHT = re.compile(r"""st\.dataframe\(\s*([^\)]*?)\)""", re.M | re.S)

def add_height(m):
    args = m.group(1)
    if "height=" in args:
        return m.group(0)
    # append height=TABLE_HEIGHT before the closing )
    if args.strip().endswith(","):
        new_args = args + " height=TABLE_HEIGHT"
    else:
        new_args = args + (", " if args.strip() else "") + "height=TABLE_HEIGHT"
    return f"st.dataframe({new_args})"

def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_opt_tab_v2.py root/optimization_tab.py")
        sys.exit(2)

    p = Path(sys.argv[1])
    src = p.read_text(encoding="utf-8", errors="replace")
    out = src
    removed = 0

    # Remove duplicate late blocks
    for pat in DUP_BLOCKS:
        out, n = re.subn(pat, "", out)
        removed += n

    # Remove sidebar Param Mapping expander if present (safe if not found)
    out, n_sidebar = re.subn(SIDEBAR_PARAM_MAPPING, "", out)

    # Add height=TABLE_HEIGHT to st.dataframe calls missing it
    out = DATAFRAME_NO_HEIGHT.sub(add_height, out)

    if out != src:
        p.with_suffix(p.suffix + ".bak").write_text(src, encoding="utf-8")
        p.write_text(out, encoding="utf-8")
        print("[OK] Updated", p)
        print("    - duplicate blocks removed:", removed)
        print("    - sidebar Param Mapping removed:", n_sidebar)
        print("    - dataframe height normalized.")
        print("Backup:", p.with_suffix(p.suffix + ".bak"))
    else:
        print("[OK] No changes were necessary.")

if __name__ == "__main__":
    main()
