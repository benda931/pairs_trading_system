import os
from pathlib import Path

REQUIRED_MODULES = {
    'common': ['helpers.py', 'signal_generator.py', '__init__.py'],
    'core': [
        'optimizer.py', 'feature_selection.py', 'clustering.py',
        'meta_optimizer.py', 'optimization_backtester.py', '__init__.py'
    ],
    'root': ['visualization.py', 'optimization_tab.py', '__init__.py'],
    'configs': [],  # ׳¨׳§ ׳׳•׳•׳“׳ ׳©׳§׳™׳™׳׳×
}

def check_structure(base_path: str = '.'):
    base = Path(base_path).resolve()
    missing = []

    print(f"\nנ“¦ Checking project structure in: {base}\n")

    for folder, files in REQUIRED_MODULES.items():
        folder_path = base / folder
        if not folder_path.exists():
            print(f"ג Missing folder: {folder}")
            missing.append(folder)
            continue
        else:
            print(f"ג… Folder exists: {folder}")

        for f in files:
            file_path = folder_path / f
            if not file_path.exists():
                print(f"   ג ן¸ Missing file: {f}")
                missing.append(str(file_path))
            else:
                print(f"   ג… Found file: {f}")

    # extra: validate __init__.py in every folder
    for folder in REQUIRED_MODULES.keys():
        init_path = base / folder / '__init__.py'
        if not init_path.exists():
            print(f"ג ן¸  Missing __init__.py in {folder}")
            missing.append(str(init_path))

    if not missing:
        print("\nנ‰ All required folders and files are present!\n")
    else:
        print(f"\nנ¨ Issues detected in {len(missing)} item(s).\n")

if __name__ == "__main__":
    check_structure()


