from __future__ import annotations

import json
from pathlib import Path

import common.config_manager as config_manager


def _rebind_config_paths(project_root: Path) -> None:
    config_manager.PROJECT_ROOT = project_root
    config_manager.CONFIG_PATH = project_root / "config.json"
    config_manager.CONFIG_DIR = project_root / "configs"
    config_manager.load_settings.cache_clear()


def test_resolve_config_path_prefers_existing_profile_in_configs(tmp_path) -> None:
    _rebind_config_paths(tmp_path)
    config_manager.ensure_config_dir()
    profile_path = config_manager.CONFIG_DIR / "research.json"
    profile_path.write_text("{}", encoding="utf-8")

    assert config_manager.resolve_config_path("config.json") == config_manager.CONFIG_PATH
    assert config_manager.resolve_config_path("research.json") == profile_path
    assert config_manager.resolve_config_path(Path("configs") / "research.json") == profile_path


def test_save_config_profile_writes_under_configs(tmp_path) -> None:
    _rebind_config_paths(tmp_path)

    saved_path = config_manager.save_config_profile({"theme": "dark"}, profile="night.json")

    assert saved_path == tmp_path / "configs" / "night.json"
    assert saved_path.exists()
    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    assert payload["theme"] == "dark"


def test_mutate_config_creates_backup_and_updates_file(tmp_path) -> None:
    _rebind_config_paths(tmp_path)
    config_manager.CONFIG_PATH.write_text(json.dumps({"theme": "light"}), encoding="utf-8")

    saved_path = config_manager.mutate_config(
        lambda cfg: {**cfg, "theme": "dark", "default_tab": "Ops"},
        backup=True,
        backup_label="test_backup",
    )

    assert saved_path == config_manager.CONFIG_PATH
    updated = json.loads(config_manager.CONFIG_PATH.read_text(encoding="utf-8"))
    assert updated["theme"] == "dark"
    assert updated["default_tab"] == "Ops"

    backups = list((tmp_path / "configs").glob("config.test_backup.*.json"))
    assert backups
    original = json.loads(backups[0].read_text(encoding="utf-8"))
    assert original["theme"] == "light"
