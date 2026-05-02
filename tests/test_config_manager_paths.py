from __future__ import annotations

from pathlib import Path

import common.config_manager as config_manager


def test_config_path_name_is_config_json():
    assert config_manager.CONFIG_PATH.name == "config.json"


def test_config_path_parent_is_repo_root_not_common():
    repo_root = Path(__file__).resolve().parent.parent
    assert config_manager.CONFIG_PATH.parent == repo_root
    assert config_manager.CONFIG_PATH.parent.name != "common"


def test_no_default_config_path_under_common():
    assert config_manager.CONFIG_PATH != repo_root_common_path()


def repo_root_common_path() -> Path:
    return Path(__file__).resolve().parent.parent / "common" / "config.json"
