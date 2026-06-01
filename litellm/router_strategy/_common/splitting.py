"""
Common stratified splitting utilities.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def split_stratified_two_way(
    data: List[Dict],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Stratified two-way split by (dataset x intent).

    Returns (train, test).
    """
    rng = np.random.default_rng(seed)
    train, test = [], []
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for item in data:
        key = (item.get("dataset", "default"), item.get("intent", "general"))
        groups[key].append(item)

    for (ds, intent), items in sorted(groups.items()):
        n = len(items)
        shuffled = [items[i] for i in rng.permutation(n)]
        n_train = max(1, round(n * train_ratio))
        train.extend(shuffled[:n_train])
        test.extend(shuffled[n_train:])
        logger.debug(
            f"  [{ds}/{intent}] n={n} train={n_train} test={n - n_train}"
        )

    return train, test


def split_stratified_three_way(
    data: List[Dict],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified three-way split by (dataset x intent).

    Returns (train, val, test).
    """
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for item in data:
        key = (item.get("dataset", "default"), item.get("intent", "general"))
        groups[key].append(item)

    for (ds, intent), items in sorted(groups.items()):
        n = len(items)
        shuffled = [items[i] for i in rng.permutation(n)]
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train: n_train + n_val])
        test.extend(shuffled[n_train + n_val:])
        logger.debug(
            f"  [{ds}/{intent}] n={n} train={n_train} val={n_val} test={n - n_train - n_val}"
        )

    return train, val, test
