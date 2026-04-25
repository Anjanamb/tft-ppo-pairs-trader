"""
Configuration loader — reads config.yaml and provides typed access.
"""

from pathlib import Path
from typing import Any

import yaml


_CONFIG_CACHE: dict | None = None
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE

    config_path = Path(path) if path else _CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if path is None:
        _CONFIG_CACHE = config
    return config


def get_tickers(asset_class: str | None = None) -> dict[str, list[str]]:
    """Get tickers grouped by asset class, or for a specific class."""
    cfg = load_config()
    sources = cfg["data"]["sources"]

    result = {}
    for cls_name, cls_cfg in sources.items():
        if cls_name == "macro":
            continue
        if asset_class and cls_name != asset_class:
            continue
        result[cls_name] = cls_cfg["tickers"]

    return result


def get_all_tickers_flat() -> list[str]:
    """Get a flat list of all tradeable tickers."""
    groups = get_tickers()
    return [t for tickers in groups.values() for t in tickers]
