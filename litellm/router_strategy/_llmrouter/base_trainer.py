from __future__ import annotations

from typing import Any


class BaseTrainer:
    def __init__(self, router: Any, optimizer: Any, device: Any) -> None:
        self.router = router
        self.optimizer = optimizer
        self.device = device
