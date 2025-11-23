"""Configuration loading utilities for plaintext-sims experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "experiments.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the YAML configuration file into a plain dictionary."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


__all__ = ["load_config", "DEFAULT_CONFIG_PATH", "PROJECT_ROOT"]
