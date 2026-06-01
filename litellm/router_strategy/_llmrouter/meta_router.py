from __future__ import annotations

from typing import Any

import yaml


class MetaRouter:
    def __init__(self, model: Any, yaml_path: str, resources: Any = None) -> None:
        self.model = model
        self.yaml_path = yaml_path
        self.resources = resources
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self.cfg = cfg
        self.metric_weights = cfg.get("metric_weights", {}) if isinstance(cfg, dict) else {}
