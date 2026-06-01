"""
Common baseline computation and Pareto analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def find_pareto_optimal_models(
    per_model: Dict[str, Dict[str, float]],
) -> List[str]:
    """Find Pareto-optimal models (non-dominated in accuracy-cost space).

    A model *i* is dominated if another model *j* exists such that *j* has
    accuracy >= *i* **and** avg_cost <= *i*, with at least one strict
    inequality.  Non-dominated models form the empirical Pareto frontier.

    Parameters
    ----------
    per_model : dict[str, dict]
        Mapping of model name -> {"accuracy": float, "avg_cost": float}.

    Returns
    -------
    list[str]
        Names of Pareto-optimal models, ordered by ascending cost.
    """
    candidates = [
        (name, stats["accuracy"], stats["avg_cost"])
        for name, stats in per_model.items()
        if stats.get("accuracy") is not None and stats.get("avg_cost") is not None
    ]

    pareto: List[str] = []
    for i, (name_i, acc_i, cost_i) in enumerate(candidates):
        dominated = False
        for j, (name_j, acc_j, cost_j) in enumerate(candidates):
            if i == j:
                continue
            if (acc_j >= acc_i and cost_j <= cost_i
                    and (acc_j > acc_i or cost_j < cost_i)):
                dominated = True
                break
        if not dominated:
            pareto.append(name_i)

    cost_lookup = {name: stats["avg_cost"] for name, stats in per_model.items()}
    pareto.sort(key=lambda m: cost_lookup.get(m, 0.0))
    return pareto


def compute_baselines(
    data: List[Dict],
    available_models: List[str],
    router_name: str = "Router",
) -> Dict[str, Any]:
    """Compute reference baselines over *data*.

    Returns a dict with: oracle, best_single, worst_single, cheapest_single,
    random, per_model, complementarity, pareto_optimal.
    """
    if not data:
        return {}

    models = list(available_models)
    if not models:
        model_set: List[str] = []
        for item in data:
            for m in item.get("records", {}).keys():
                if m not in model_set:
                    model_set.append(m)
        models = model_set

    n = len(data)

    model_correct: Dict[str, int] = {m: 0 for m in models}
    model_cost: Dict[str, float] = {m: 0.0 for m in models}
    model_count: Dict[str, int] = {m: 0 for m in models}

    for item in data:
        records = item.get("records", {})
        usages = item.get("usages", {})
        for m in models:
            if m not in records:
                continue
            model_count[m] += 1
            if float(records.get(m, 0.0)) > 0.5:
                model_correct[m] += 1
            if isinstance(usages.get(m), dict):
                model_cost[m] += float(usages[m].get("cost", 0.0))

    per_model: Dict[str, Dict] = {}
    for m in models:
        cnt = model_count[m] or 1
        per_model[m] = {
            "accuracy": model_correct[m] / cnt,
            "avg_cost": model_cost[m] / cnt,
        }

    oracle_correct = 0
    oracle_total = 0.0
    for item in data:
        records = item.get("records", {})
        usages = item.get("usages", {})
        candidates_map = {
            m: (
                float(records.get(m, 0.0)),
                -float(usages[m].get("cost", 0.0))
                if isinstance(usages.get(m), dict) else 0.0,
            )
            for m in models if m in records
        }
        if not candidates_map:
            continue
        best_m = max(candidates_map, key=lambda m: (candidates_map[m][0], candidates_map[m][1]))
        score = float(records.get(best_m, 0.0))
        cost = (
            float(usages[best_m].get("cost", 0.0))
            if isinstance(usages.get(best_m), dict) else 0.0
        )
        if score > 0.5:
            oracle_correct += 1
        oracle_total += cost

    oracle = {
        "accuracy": oracle_correct / n,
        "avg_cost": oracle_total / n,
    }

    valid = [m for m in models if model_count[m] > 0]
    if not valid:
        return {"oracle": oracle, "per_model": per_model}

    best_m = max(valid, key=lambda m: per_model[m]["accuracy"])
    best_single = {
        "model": best_m,
        "accuracy": per_model[best_m]["accuracy"],
        "avg_cost": per_model[best_m]["avg_cost"],
    }

    worst_m = min(valid, key=lambda m: per_model[m]["accuracy"])
    worst_single = {
        "model": worst_m,
        "accuracy": per_model[worst_m]["accuracy"],
        "avg_cost": per_model[worst_m]["avg_cost"],
    }

    cheapest_m = min(valid, key=lambda m: per_model[m]["avg_cost"])
    cheapest_single = {
        "model": cheapest_m,
        "accuracy": per_model[cheapest_m]["accuracy"],
        "avg_cost": per_model[cheapest_m]["avg_cost"],
    }

    random_acc = float(np.mean([per_model[m]["accuracy"] for m in valid]))
    random_cost = float(np.mean([per_model[m]["avg_cost"] for m in valid]))
    random_baseline = {"accuracy": random_acc, "avg_cost": random_cost}

    complementarity = oracle["accuracy"] - best_single["accuracy"]

    pareto_optimal = find_pareto_optimal_models(per_model)
    logger.info(
        f"[{router_name}] Pareto-optimal models "
        f"({len(pareto_optimal)}): {pareto_optimal}"
    )

    return {
        "oracle": oracle,
        "best_single": best_single,
        "worst_single": worst_single,
        "cheapest_single": cheapest_single,
        "random": random_baseline,
        "per_model": per_model,
        "complementarity": complementarity,
        "pareto_optimal": pareto_optimal,
    }
