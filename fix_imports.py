# fix_imports.py  ג€“ run once from project root
import re
from pathlib import Path

ROOT = Path(__file__).parent        # ג†’ pairs_trading_system/
PATTERN = re.compile(
    r'from\\s+(utils|data_loader|config_manager)\\s+import'
)

def patch_file(py: Path) -> int:
    txt = py.read_text(encoding='utf-8')
    new_txt, n = PATTERN.subn(
        lambda m: f'from common.{m.group(1)} import', txt
    )
    if n:
        py.write_text(new_txt, encoding='utf-8')
    return n

count = 0
for file in ROOT.rglob('*.py'):
    if '__pycache__' in file.parts or file.parts[0].startswith('.venv'):
        continue
    count += patch_file(file)

print(f"ג”ן¸  Updated {count} import statements.")


