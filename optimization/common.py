from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.example.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    if yaml is None:
        raise RuntimeError("PyYAML is required: pip install pyyaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path.resolve())
    return config


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path, default: Any = None) -> Any:
    json_path = Path(path)
    if not json_path.exists():
        return default
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = json_path.with_name(f".{json_path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(json_path)


def sanitize_label(text: str) -> str:
    safe = text.encode("ascii", "ignore").decode("ascii")
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in safe)
    safe = safe.strip("._-")
    return safe or "sequence"


def campaign_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["runs_dir"]) / str(config["campaign_name"])
