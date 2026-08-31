from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> Dict[str, Any]:
    config_path = project_root() / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
