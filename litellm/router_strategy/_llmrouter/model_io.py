from __future__ import annotations

import pickle
from typing import Any


def _require_pkl(path: str) -> None:
    if not path.endswith(".pkl"):
        raise ValueError("Only .pkl files are supported")


def save_model(model: Any, path: str) -> None:
    _require_pkl(path)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> Any:
    _require_pkl(path)
    with open(path, "rb") as f:
        return pickle.load(f)
