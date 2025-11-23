from __future__ import annotations

from pathlib import Path

import pytest

from plaintext_sims.config import load_config


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("simpy:\n  seed: 123\n  replications: 10\n", encoding="utf-8")

    cfg = load_config(cfg_path)

    assert cfg["simpy"]["seed"] == 123
    assert cfg["simpy"]["replications"] == 10


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)
