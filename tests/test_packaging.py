from pathlib import Path

import yaml

from muon_tomo.cli import DEFAULT_CONFIG, _config


def test_packaged_default_config_matches_repository_copy():
    repository_config = Path(__file__).parents[1] / "configs" / "default.yaml"
    assert DEFAULT_CONFIG.is_file()
    assert yaml.safe_load(DEFAULT_CONFIG.read_text()) == yaml.safe_load(
        repository_config.read_text()
    )


def test_default_config_loads_without_source_tree_lookup():
    config = _config(None)
    assert config["geometry"]["shape"] == [64, 64, 64]
    assert config["geometry"]["axis_order"] == "xyz"
