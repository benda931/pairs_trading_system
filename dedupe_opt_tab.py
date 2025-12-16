from __future__ import annotations
import sys, re, io, pathlib

# פונקציות/בלוקים הידועים שנוטים להכפל – נשמור רק את ההופעה הראשונה שלהם
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

# מאתר כותרות def/class בתחילת שורה
DEF_RE = re.compile(r"^(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(.*?$", re.M)

def split_defs(text: str):
    """מפצל את הטקסט לרשימת [(kind,name,chunk)] כך שכל הגדרה מקבלת את הגוף עד ההגדרה הבאה."""
    matches = list(DEF_RE.finditer(text))
    parts = []
    last = 0
    for i, m in enumerate(matches):
        # פרולוג עד להגדרה
        if m.start() > last:
            parts.append(("other", None, text[last:m.start()]))
        name = m.group(2)
        # גוף ההגדרה עד ההגדרה הבאה או סוף קובץ
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        parts.append(("def", name, text[m.start():end]))
        last = end
    if last < len(text):
        parts.append(("other", None, text[last:]))
    return parts

def dedupe(text: str) -> str:
    chunks = split_defs(text)
    seen = set()
    out = io.StringIO()
    for kind, name, chunk in chunks:
        if kind == "other":
            out.write(chunk)
            continue
        # kind == "def"
        if name in KNOWN_FUNCS:
            if name in seen:
                # דילוג על עותק שני+
                continue
            seen.add(name)
        out.write(chunk)
    return out.getvalue()

def main():
    if len(sys.argv) != 3:
        print("Usage: python dedupe_opt_tab.py <input.py> <output.py>")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[2]
    src = pathlib.Path(inp).read_text(encoding="utf-8")
    out = dedupe(src)
    pathlib.Path(outp).write_text(out, encoding="utf-8")
    print("Wrote deduped file to:", outp)

if __name__ == "__main__":
    main()
