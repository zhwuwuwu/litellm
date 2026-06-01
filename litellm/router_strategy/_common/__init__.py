"""Frozen fork of SuperClaw common router utilities.

Source path: integrations/llmrouter/custom_routers/common/
SuperClaw source SHA: e1737666604b0c0d67b6c5be171da9b234d18512
Policy: frozen fork; keep copied module bodies byte-identical except the two
intentional rewrites: data_utils.py line 145 uses vendored
_llmrouter.load_jsonl, and intent_classifier.py line 39 switches to the sibling
intent_cache import.
"""

from .data_utils import (
    normalise_rows,
    pivot_long_rows,
    normalise_wide_rows,
    pivot_df,
    load_all_routing_data,
)
from .baselines import compute_baselines, find_pareto_optimal_models
from .splitting import split_stratified_two_way, split_stratified_three_way
from .embedding_cache import EmbeddingCache
from .intent_cache import IntentCache
from .intent_classifier import IntentClassifier, DEFAULT_INTENT_CLASSES
from .probe_selector import (
    ProbeSelector,
    get_selected_queries,
    select_probe_set_d_optimal,
    select_probe_set_kmeans,
    select_optimal_k_for_probes,
    find_optimal_k_silhouette,
    find_optimal_k_elbow,
    find_optimal_k_gap_statistic,
    find_optimal_k_combined,
)

__all__ = [
    "normalise_rows",
    "pivot_long_rows",
    "normalise_wide_rows",
    "pivot_df",
    "load_all_routing_data",
    "compute_baselines",
    "find_pareto_optimal_models",
    "split_stratified_two_way",
    "split_stratified_three_way",
    "EmbeddingCache",
    "IntentCache",
    "IntentClassifier",
    "DEFAULT_INTENT_CLASSES",
    "ProbeSelector",
    "get_selected_queries",
    "select_probe_set_d_optimal",
    "select_probe_set_kmeans",
    "select_optimal_k_for_probes",
    "find_optimal_k_silhouette",
    "find_optimal_k_elbow",
    "find_optimal_k_gap_statistic",
    "find_optimal_k_combined",
]
