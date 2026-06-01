"""
Common data loading and normalization utilities.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def normalise_rows(rows: List[Dict], router_name: str = "Router") -> List[Dict]:
    """Convert raw JSONL rows into uniform per-query dicts.

    Auto-detects format from the first row:
    - "model_name" present → long format
    - otherwise → wide format
    """
    if not rows:
        return []
    if "model_name" in rows[0]:
        logger.info(f"  Detected long format ({len(rows)} rows).")
        return pivot_long_rows(rows)
    else:
        logger.info(f"  Detected wide format ({len(rows)} rows).")
        return normalise_wide_rows(rows)


def pivot_long_rows(rows: List[Dict]) -> List[Dict]:
    """Pivot long-format rows (one row per query x model) into per-query dicts."""
    by_query: Dict[str, Dict] = {}
    for row in rows:
        q = str(row.get("query", "")).strip()
        if not q:
            continue
        if q not in by_query:
            ds = str(row.get("dataset") or row.get("task_name") or "default")
            by_query[q] = {"query": q, "records": {}, "usages": {}, "dataset": ds}
        model = str(row.get("model_name", "unknown"))
        perf = row.get("performance")
        cost = row.get("cost")
        by_query[q]["records"][model] = float(perf) if perf is not None else 0.0
        if cost is not None:
            by_query[q]["usages"][model] = {"cost": float(cost)}

    result = list(by_query.values())
    logger.info(f"  Pivoted to {len(result)} unique queries.")
    return result


def normalise_wide_rows(rows: List[Dict]) -> List[Dict]:
    """Normalise wide-format rows."""
    out = []
    for i, row in enumerate(rows):
        q = str(row.get("query", "")).strip()
        if not q or "records" not in row:
            logger.debug(f"Row {i}: missing query/records — skipping.")
            continue
        records = {}
        for model, score in row["records"].items():
            if score is None:
                records[model] = 0.0
            elif isinstance(score, bool):
                records[model] = 1.0 if score else 0.0
            else:
                records[model] = float(score)
        out.append({
            "query": q,
            "records": records,
            "usages": row.get("usages", {}),
            "dataset": str(row.get("dataset") or row.get("task_name") or "default"),
        })
    return out


def pivot_df(df: "pd.DataFrame", router_name: str = "Router") -> List[Dict]:
    """Convert a pandas DataFrame into per-query dicts.

    Required columns: query, model_name, performance.
    Optional columns: cost, dataset, task_name.
    """
    required = {"query", "model_name", "performance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{router_name}] Routing DataFrame missing columns: {missing}. "
            f"Got: {list(df.columns)}"
        )
    has_cost = "cost" in df.columns
    has_dataset = "dataset" in df.columns
    has_task = "task_name" in df.columns

    by_query: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        q = str(row["query"]).strip()
        if not q:
            continue
        if q not in by_query:
            ds = (
                str(row["dataset"]) if has_dataset else
                str(row["task_name"]) if has_task else "default"
            )
            by_query[q] = {"query": q, "records": {}, "usages": {}, "dataset": ds}
        model = str(row["model_name"])
        by_query[q]["records"][model] = float(row["performance"])
        if has_cost and row["cost"] is not None:
            by_query[q]["usages"][model] = {"cost": float(row["cost"])}

    result = list(by_query.values())
    logger.info(f"  Pivoted DataFrame to {len(result)} unique queries.")
    return result


def load_all_routing_data(
    cfg: Dict[str, Any],
    resolve_path_fn: Callable[[str], str],
    router_instance: Any,
    router_name: str = "Router",
) -> List[Dict]:
    """Load ALL routing data into one unified list before any splitting.

    Priority:
    1. data_path.routing_data_all — single unified file
    2. Merge routing_data_train + routing_data_test DataFrames

    Parameters
    ----------
    cfg : dict
        The router's configuration dict.
    resolve_path_fn : callable
        Function to resolve relative paths.
    router_instance : object
        The router instance (to access DataLoader-attached DataFrames).
    router_name : str
        Name for logging.
    """
    from litellm.router_strategy._llmrouter import load_jsonl

    data_path = cfg.get("data_path", {})

    all_key = data_path.get("routing_data_all", "")
    all_path = resolve_path_fn(all_key) if all_key else None

    if all_path and os.path.exists(all_path):
        logger.info(f"[{router_name}] Loading unified data: {all_path}")
        raw = load_jsonl(all_path) or []
        if raw:
            return normalise_rows(raw, router_name)
        logger.warning(
            f"[{router_name}] routing_data_all exists but is empty: {all_path}"
        )

    frames = []
    for attr in ("routing_data_train", "routing_data_test"):
        df = getattr(router_instance, attr, None)
        if df is not None and len(df) > 0:
            frames.append(df)

    if frames:
        import pandas as pd
        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            f"[{router_name}] Merging {len(frames)} DataFrame(s) "
            f"({len(combined)} total rows) into unified pool."
        )
        return pivot_df(combined, router_name)

    logger.warning(f"[{router_name}] No data found at init.")
    return []
