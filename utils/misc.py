from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


def deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if (
            key in dst
            and isinstance(dst[key], dict)
            and isinstance(value, Mapping)
        ):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {p}")
    return data
