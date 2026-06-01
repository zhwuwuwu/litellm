from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def load_jsonl(path: str) -> Optional[List[Dict[str, Any]]]:
    if not os.path.exists(path):
        return None
    try:
        rows: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    except Exception:
        return None
