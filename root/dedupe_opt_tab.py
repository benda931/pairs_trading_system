# dedupe_opt_tab.py
from __future__ import annotations
import sys, re, io, pathlib

# ׳₪׳•׳ ׳§׳¦׳™׳•׳×/׳‘׳׳•׳§׳™׳ ׳©׳™׳“׳•׳¢׳™׳ ׳©׳›׳₪׳•׳׳™׳ ׳׳¢׳™׳×׳™׳ ׳‘׳§׳•׳‘׳¥
KNOWN_FUNCS = [
    # config & storage
    "load_settings", "_default_db_path", "get_duck", "get_ro_duck", "_DuckProxy",
    "_ensure_duck_schema", "make_json_safe", "save_trials_to_duck",
    "list_pairs_in_db", "list_studies_for_pair", "load_trials_from_duck",
    # metrics & ranges
    "_norm_fallback", "_score_fallback", "get_default_param_ranges", "run_backtest",
    # ui helpers
    "_render_profile_sidebar", "_sidebar_common", "_recommend_params",
    "_suggest_ranges_from_df", "_suggest_ranges_from_cluster",
    # optuna
    "_build_objective", "_optuna_optimize",
    # dq
    "_dq_coerce_numeric", "_dq_hash_params_row", "_dq_basic_clean",
    # misc helpers
    "_validate_ranges", "_apply_param_mapping", "_normalize_weights",
    "_log", "_get_logs", "_validate_for_optuna", "_bt_known_kwargs", "_sanitize_bt_kwargs",
    # ui entry
    "render_optimization_tab",
]

DEF_RE = re.compile(r"^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)

def split_by_defs(text: str):
    """ ׳׳₪׳¦׳ ׳׳§׳•׳‘׳™׳•׳× ׳׳₪׳™ def/class, ׳›׳•׳׳ ׳₪׳¨׳•׳׳•׳’ ׳׳₪׳ ׳™ ׳”׳”׳’׳“׳¨׳” ׳”׳¨׳׳©׳•׳ ׳”. """
    parts = []
    last = 0
    for m in DEF_RE.finditer(text):
        if m.start() > last:
            parts.append(("other", None, text[last:m.start()]))
        parts.append(("def", m.group(2), None))  # ׳ ׳—׳׳™׳£ ׳׳×׳•׳›׳ ׳׳—׳¨ ׳›׳
        last = m.start()
    if last < len(text):
        parts.append(("other", None, text[last:]))
    # ׳¢׳›׳©׳™׳• ׳ ׳‘׳ ׳” ׳§׳˜׳¢׳™׳ ׳›׳ ׳©׳›׳ "def" ׳™׳§׳‘׳ ׳׳× ׳”׳’׳•׳£ ׳¢׳“ ׳”׳”׳’׳“׳¨׳” ׳”׳‘׳׳”
    merged = []
    i = 0
    while i < len(parts):
        kind, name, payload = parts[i]
        if kind == "def":
            # ׳’׳•׳£ ׳”׳”׳’׳“׳¨׳” ׳ ׳׳¦׳ ׳‘׳§׳˜׳¢ ׳”׳‘׳ ׳׳¡׳•׳’ "other" (׳׳• ׳¨׳™׳§ ׳׳ EOF)
            body = parts[i+1][2] if i+1 < len(parts) and parts[i+1][0]=="other" else ""
            merged.append(("def", name, body))
            i += 2
        else:
            merged.append(parts[i])
            i += 1
    return merged

def dedupe(text: str) -> str:
    chunks = split_by_defs(text)
    seen = set()
    out = io.StringIO()
    for kind, name, payload in chunks:
        if kind == "other":
            out.write(payload)
        else:
            header_match = re.search(rf"^(def|class)\s+{re.escape(name)}\s*\(.*$", payload, re.M)
            header = header_match.group(0) if header_match else f"def {name}(...):"
            full = payload
            # ׳”׳׳ ׳׳©׳׳•׳¨?
            if name in KNOWN_FUNCS:
                if name in seen:
                    # ׳“׳™׳׳•׳’ ׳¢׳ ׳¢׳•׳×׳§ ׳©׳ ׳™+
                    continue
                seen.add(name)
            # ׳›׳×׳™׳‘׳”: ׳”׳”׳’׳“׳¨׳” + ׳”׳’׳•׳£ ׳©׳׳”
            out.write(full)
    return out.getvalue()

def main():
    if len(sys.argv) != 3:
        print("Usage: python dedupe_opt_tab.py <input.py> <output.py>")
        sys.exit(1)
    src = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    dedup = dedupe(src)
    pathlib.Path(sys.argv[2]).write_text(dedup, encoding="utf-8")
    print("Wrote deduped file to:", sys.argv[2])

if __name__ == "__main__":
    main()
