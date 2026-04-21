"""YAML loading helpers."""

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {resolved}")

    return data
