from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.kvresearch.yaml"


@dataclass(frozen=True)
class Config:
    path: Path
    data: dict[str, Any]

    @property
    def project_root(self) -> Path:
        return Path(self.data["project"]["root"]).resolve()

    @property
    def runs_dir(self) -> Path:
        return Path(self.data["project"]["runs_dir"]).resolve()

    @property
    def catalog_path(self) -> Path:
        return Path(self.data["project"]["artifact_catalog"]).resolve()

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"missing config section: {name}")
        return value


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path or DEFAULT_CONFIG).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError(f"invalid configuration: {config_path}")
    return Config(config_path, data)

