"""
LLMRecommender — probe_selector.py
====================================
Probe set selection for cold-start LLM evaluation.

Supports two methods:
1. D-optimal design - maximizes determinant of X^T X for the probe set
2. K-means cluster centroids - selects points closest to cluster centers

Both methods operate in the embedding space to maximize diversity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)




def get_selected_queries(
    selected_indices: np.ndarray,
    routing_data_all: List[Dict],
) -> List[Dict]:
    """
    Map selected probe indices to their corresponding entries in routing_data_all.

    The returned list preserves the same dict structure as the original input
    data, i.e. each entry has the form::

        {
            "query":   str,
            "records": {model_name: score, ...},
            "usages":  {model_name: {...}, ...},
            "dataset": str,
        }

    Parameters
    ----------
    selected_indices : np.ndarray, shape (probe_size,)
        Indices into ``routing_data_all`` as returned by a probe-selection
        function or ``ProbeSelector.fit()``.
    routing_data_all : list of dict
        The full pool of query dicts in the normalised wide format.

    Returns
    -------
    list of dict
        Selected query dicts in the same format as ``routing_data_all``,
        ordered by ``selected_indices``.
    """
    n = len(routing_data_all)
    selected_queries: List[Dict] = []
    for idx in selected_indices:
        i = int(idx)
        if i < 0 or i >= n:
            raise IndexError(
                f"selected_indices contains out-of-range index {i} "
                f"(routing_data_all has {n} entries)."
            )
        selected_queries.append(routing_data_all[i])
    return selected_queries


def select_probe_set_d_optimal(
    E: np.ndarray,
    probe_size: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select probe set using D-optimal design.

    D-optimal design selects the subset of points that maximizes the
    determinant of (X^T X) where X is the design matrix of selected points.
    This corresponds to maximizing the volume of the ellipsoid spanned by
    the selected points, which is a principled way to ensure diversity.

    Parameters
    ----------
    E : np.ndarray, shape (n_samples, embed_dim)
        All candidate embedding vectors.
    probe_size : int
        Number of probe points to select.
    seed : int
        Random seed for tie-breaking.

    Returns
    -------
    tuple of (selected_indices, selected_embeddings)
        selected_indices : np.ndarray, shape (probe_size,)
            Indices of selected points in E.
        selected_embeddings : np.ndarray, shape (probe_size, embed_dim)
            The selected embedding vectors.
    """
    n_samples, embed_dim = E.shape

    if probe_size >= n_samples:
        logger.warning(
            f"probe_size ({probe_size}) >= n_samples ({n_samples}), "
            "returning all points as probes."
        )
        indices = np.arange(n_samples)
        return indices, E[indices]

    if probe_size <= 0:
        raise ValueError(f"probe_size must be positive, got {probe_size}")

    rng = np.random.default_rng(seed)

    selected_indices: List[int] = []
    remaining_mask = np.ones(n_samples, dtype=bool)

    first_idx = rng.integers(n_samples)
    selected_indices.append(first_idx)
    remaining_mask[first_idx] = False

    E_selected = [E[first_idx]]
    E_remaining = E[remaining_mask]
    remaining_indices = np.where(remaining_mask)[0]

    # Maintain XtX incrementally; small ridge term keeps it non-singular from
    # the first step (single-point XtX is rank-1).
    _eps = 1e-8 * embed_dim
    XtX = E[first_idx][np.newaxis, :].T @ E[first_idx][np.newaxis, :] + _eps * np.eye(embed_dim)

    for _ in range(probe_size - 1):
        if len(E_remaining) == 0:
            break

        # Score all remaining candidates at once via the matrix determinant lemma:
        #   det(XtX + e eᵀ) = det(XtX) * (1 + eᵀ XtX⁻¹ e)
        # det(XtX) is constant across candidates so scores ∝ (1 + leverage_i).
        # One batched solve: V = XtX⁻¹ @ E_remaining.T  →  O(d² · n_remaining)
        # instead of O(d³) det per candidate.
        try:
            V = np.linalg.solve(XtX, E_remaining.T)          # (d, n_remaining)
            leverage = np.einsum("ij,ji->i", E_remaining, V)  # diag of E_remaining @ V
            scores = np.maximum(1.0 + leverage, 0.0)
        except np.linalg.LinAlgError:
            scores = np.zeros(len(E_remaining))

        if scores.sum() == 0:
            remaining_idx = rng.integers(len(E_remaining))
        else:
            scores_normalized = scores / (scores.sum() + 1e-12)
            remaining_idx = rng.choice(
                len(E_remaining), p=scores_normalized
            )

        chosen_global_idx = remaining_indices[remaining_idx]
        selected_indices.append(int(chosen_global_idx))
        remaining_mask[chosen_global_idx] = False

        chosen_e = E[chosen_global_idx]
        # Rank-1 update: XtX += e eᵀ
        XtX = XtX + np.outer(chosen_e, chosen_e)

        E_selected.append(chosen_e)
        E_remaining = E[remaining_mask]
        remaining_indices = np.where(remaining_mask)[0]

    selected_indices = np.array(selected_indices, dtype=int)
    selected_embeddings = E[selected_indices]

    logger.info(
        f"[ProbeSelector] D-optimal: selected {len(selected_indices)} probes "
        f"from {n_samples} candidates"
    )

    return selected_indices, selected_embeddings


def select_probe_set_kmeans(
    E: np.ndarray,
    probe_size: int,
    seed: int = 42,
    n_init: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select probe set using K-means cluster centroids.

    Fits K-means with K = probe_size and selects the point closest to
    each centroid. This ensures diversity by capturing different regions
    of the embedding space.

    Parameters
    ----------
    E : np.ndarray, shape (n_samples, embed_dim)
        All candidate embedding vectors.
    probe_size : int
        Number of probe points to select (K for K-means).
    seed : int
        Random seed for K-means.
    n_init : int
        Number of initializations for K-means.

    Returns
    -------
    tuple of (selected_indices, selected_embeddings)
        selected_indices : np.ndarray, shape (probe_size,)
            Indices of selected points in E.
        selected_embeddings : np.ndarray, shape (probe_size, embed_dim)
            The selected embedding vectors (centroid representatives).
    """
    n_samples, embed_dim = E.shape

    if probe_size >= n_samples:
        logger.warning(
            f"probe_size ({probe_size}) >= n_samples ({n_samples}), "
            "returning all points as probes."
        )
        indices = np.arange(n_samples)
        return indices, E[indices]

    if probe_size <= 0:
        raise ValueError(f"probe_size must be positive, got {probe_size}")

    k = min(probe_size, n_samples)

    E_gpu = np.asarray(E, dtype=np.float32)
    kmeans = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=n_init,
    )
    labels = kmeans.fit_predict(E_gpu)
    centroids = kmeans.cluster_centers_

    # Compute distances from every point to every centroid: (n_samples, k)
    dists_all = np.linalg.norm(
        E[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
    )

    selected_indices = np.full(k, -1, dtype=int)
    selected_embeddings = []

    # Greedy assignment: repeatedly pick the globally closest (point, centroid)
    # pair, marking each point and centroid as used to avoid duplicates.
    remaining_dists = dists_all.copy()
    for _ in range(k):
        flat_idx = int(np.argmin(remaining_dists))
        point_idx, centroid_idx = np.unravel_index(flat_idx, remaining_dists.shape)
        selected_indices[centroid_idx] = int(point_idx)
        selected_embeddings.append(E[point_idx])
        # Remove this point and centroid from future consideration
        remaining_dists[point_idx, :] = np.inf
        remaining_dists[:, centroid_idx] = np.inf

    selected_embeddings = np.array(selected_embeddings)

    logger.info(
        f"[ProbeSelector] K-means: selected {len(selected_indices)} probes "
        f"from {n_samples} candidates (K={k})"
    )

    return selected_indices, selected_embeddings


def find_optimal_k_elbow(
    E: np.ndarray,
    min_k: int = 5,
    max_k: int = 50,
    seed: int = 42,
) -> Tuple[int, Dict[str, Any]]:
    """
    Find optimal number of clusters using the Elbow Method.

    Parameters
    ----------
    E : np.ndarray, shape (n_samples, embed_dim)
        Candidate embedding vectors.
    min_k : int
        Minimum K to try.
    max_k : int
        Maximum K to try.
    seed : int
        Random seed.

    Returns
    -------
    tuple of (optimal_k, metrics)
        optimal_k : int
        metrics : dict with inertia values and detected elbow point
    """
    n_samples = len(E)

    min_k = max(2, min(min_k, n_samples - 1))
    max_k = min(max(max_k, min_k), n_samples - 1)

    E_gpu = np.asarray(E, dtype=np.float32)
    k_range = list(range(min_k, max_k + 1))
    inertia_values = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=5)
        kmeans.fit(E_gpu)
        inertia_values.append(float(kmeans.inertia_))

    k_array = np.array(k_range)
    inertia_array = np.array(inertia_values)

    inertia_normalized = (inertia_array - inertia_array.min()) / (
        inertia_array.max() - inertia_array.min() + 1e-12
    )

    if len(k_range) > 2:
        first_derivative = np.gradient(inertia_normalized, k_array)
        second_derivative = np.gradient(first_derivative, k_array)
        elbow_idx = np.argmax(-second_derivative)
        elbow_k = k_range[elbow_idx]
    else:
        elbow_k = min_k

    metrics = {
        "k_range": k_range,
        "inertia_values": inertia_values,
        "elbow_k": elbow_k,
    }

    logger.info(
        f"[ProbeSelector] Optimal K={elbow_k} via elbow method "
        f"(inertia range: {min(inertia_values):.2f} - {max(inertia_values):.2f})"
    )

    return elbow_k, metrics


def find_optimal_k_silhouette(
    E: np.ndarray,
    min_k: int = 5,
    max_k: int = 50,
    seed: int = 42,
) -> Tuple[int, Dict[str, Any]]:
    """
    Find optimal number of clusters using Silhouette Analysis.

    Parameters
    ----------
    E : np.ndarray, shape (n_samples, embed_dim)
        Candidate embedding vectors.
    min_k : int
        Minimum K to try.
    max_k : int
        Maximum K to try.
    seed : int
        Random seed.

    Returns
    -------
    tuple of (optimal_k, metrics)
        optimal_k : int
        metrics : dict with silhouette scores for each K
    """
    from sklearn.metrics import silhouette_score

    n_samples = len(E)

    min_k = max(2, min(min_k, n_samples - 1))
    max_k = min(max(max_k, min_k), n_samples - 1)

    E_gpu = np.asarray(E, dtype=np.float32)
    k_range = list(range(min_k, max_k + 1))
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=5)
        labels = kmeans.fit_predict(E_gpu)
        score = silhouette_score(E_gpu, labels)
        silhouette_scores.append(float(score))
        logger.debug(f"  K={k}: silhouette={score:.4f}")

    best_idx = int(np.argmax(silhouette_scores))
    optimal_k = k_range[best_idx]

    metrics = {
        "k_range": k_range,
        "silhouette_scores": silhouette_scores,
        "best_k": optimal_k,
        "best_silhouette": float(max(silhouette_scores)),
    }

    logger.info(
        f"[ProbeSelector] Optimal K={optimal_k} via silhouette "
        f"(score={max(silhouette_scores):.4f})"
    )

    return optimal_k, metrics


def find_optimal_k_gap_statistic(
    E: np.ndarray,
    min_k: int = 5,
    max_k: int = 50,
    n_reference: int = 5,
    seed: int = 42,
) -> Tuple[int, Dict[str, Any]]:
    """
    Find optimal number of clusters using the Gap Statistic.

    Parameters
    ----------
    E : np.ndarray, shape (n_samples, embed_dim)
        Candidate embedding vectors.
    min_k : int
        Minimum K to try.
    max_k : int
        Maximum K to try.
    n_reference : int
        Number of reference datasets for gap statistic computation.
    seed : int
        Random seed.

    Returns
    -------
    tuple of (optimal_k, metrics)
        optimal_k : int
        metrics : dict with gap statistics for each K
    """
    n_samples = len(E)

    min_k = max(2, min(min_k, n_samples - 1))
    max_k = min(max(max_k, min_k), n_samples - 1)

    E_gpu = np.asarray(E, dtype=np.float32)
    k_range = list(range(min_k, max_k + 1))

    log_w_values = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=3)
        labels = kmeans.fit_predict(E_gpu)
        centers = kmeans.cluster_centers_

        centers_work = np.asarray(centers, dtype=np.float32)
        distances = np.sqrt(
            np.sum((E_gpu[:, np.newaxis, :] - centers_work[np.newaxis, :, :]) ** 2, axis=2)
        )
        min_distances = distances.min(axis=1)
        w_k = float(np.sum(min_distances ** 2))
        log_w_values.append(np.log(w_k + 1e-12))

    rng = np.random.default_rng(seed)
    X_min = E.min(axis=0)
    X_max = E.max(axis=0)

    gap_statistics = []
    se_values = []

    for i, k in enumerate(k_range):
        ref_log_w = []

        for b in range(n_reference):
            ref_X = rng.uniform(X_min, X_max, size=(len(E), E.shape[1]))
            ref_kmeans = KMeans(n_clusters=k, random_state=seed + b, n_init=3)
            ref_kmeans.fit(ref_X)
            ref_centers = ref_kmeans.cluster_centers_
            ref_distances = np.sqrt(
                np.sum((ref_X[:, np.newaxis, :] - ref_centers[np.newaxis, :, :]) ** 2, axis=2)
            )
            ref_min_distances = ref_distances.min(axis=1)
            w_k_ref = float(np.sum(ref_min_distances ** 2))
            ref_log_w.append(np.log(w_k_ref + 1e-12))

        gap = np.mean(ref_log_w) - log_w_values[i]
        gap_statistics.append(gap)
        se_k = float(np.std(ref_log_w) * np.sqrt(1.0 + 1.0 / n_reference))
        se_values.append(se_k)

    gap_array = np.array(gap_statistics)
    se_array = np.array(se_values)

    optimal_k = k_range[-1]  # default to max
    if len(k_range) > 1:
        for i in range(len(k_range) - 1):
            if gap_array[i] >= gap_array[i + 1] - se_array[i + 1]:
                optimal_k = k_range[i]
                break

    metrics = {
        "k_range": k_range,
        "gap_statistics": [float(g) for g in gap_statistics],
        "log_w_values": [float(w) for w in log_w_values],
        "optimal_k": optimal_k,
        "best_gap": float(max(gap_statistics)),
    }

    logger.info(
        f"[ProbeSelector] Optimal K={optimal_k} via gap statistic "
        f"(best gap={max(gap_statistics):.4f})"
    )

    return optimal_k, metrics


def find_optimal_k_combined(
    E: np.ndarray,
    min_k: int = 5,
    max_k: int = 50,
    methods: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[int, Dict[str, Any]]:
    """
    Find optimal number of clusters using multiple methods and voting.

    Combines silhouette, elbow, and gap statistic to find a robust K.

    Parameters
    ----------
    E : np.ndarray, shape (n_samples, embed_dim)
        Candidate embedding vectors.
    min_k : int
        Minimum K to try.
    max_k : int
        Maximum K to try.
    methods : list of str, optional
        Methods to use: "silhouette", "elbow", "gap". Defaults to all three.
    seed : int
        Random seed.

    Returns
    -------
    tuple of (optimal_k, metrics)
        optimal_k : int
        metrics : dict with all individual method results and voting results
    """
    if methods is None:
        methods = ["silhouette", "elbow", "gap"]

    all_results = {}

    if "silhouette" in methods:
        sil_k, sil_metrics = find_optimal_k_silhouette(E, min_k, max_k, seed)
        all_results["silhouette"] = {"optimal_k": sil_k, "metrics": sil_metrics}

    if "elbow" in methods:
        elbow_k, elbow_metrics = find_optimal_k_elbow(E, min_k, max_k, seed)
        all_results["elbow"] = {"optimal_k": elbow_k, "metrics": elbow_metrics}

    if "gap" in methods:
        gap_k, gap_metrics = find_optimal_k_gap_statistic(E, min_k, max_k, seed=seed)
        all_results["gap"] = {"optimal_k": gap_k, "metrics": gap_metrics}

    k_votes = {}
    for method, result in all_results.items():
        k = result["optimal_k"]
        k_votes[k] = k_votes.get(k, 0) + 1

    max_votes = max(k_votes.values())
    candidates = [k for k, v in k_votes.items() if v == max_votes]

    k_range = list(range(min_k, max_k + 1))
    if len(candidates) > 1:
        median_k = int(np.median(candidates))
        optimal_k = min(candidates, key=lambda x: abs(x - median_k))
    else:
        optimal_k = candidates[0]

    metrics = {
        "individual_results": all_results,
        "k_votes": k_votes,
        "optimal_k": optimal_k,
        "methods_used": list(all_results.keys()),
    }

    logger.info(
        f"[ProbeSelector] Optimal K={optimal_k} via combined voting "
        f"(votes: {k_votes}, methods: {list(all_results.keys())})"
    )

    return optimal_k, metrics


def select_optimal_k_for_probes(
    E: np.ndarray,
    min_k: int = 5,
    max_k: int = 50,
    method: str = "combined",
    seed: int = 42,
) -> Tuple[int, Dict[str, Any]]:
    """
    Find optimal number of probes K using various methods.

    Parameters
    ----------
    E : np.ndarray, shape (n_samples, embed_dim)
        Candidate embedding vectors.
    min_k : int
        Minimum K to try.
    max_k : int
        Maximum K to try.
    method : str
        Method to use: "silhouette", "elbow", "gap", or "combined" (default).
        - "silhouette": maximizes silhouette score
        - "elbow": uses second derivative of inertia curve
        - "gap": uses gap statistic
        - "combined": voting across all three methods
    seed : int
        Random seed.

    Returns
    -------
    tuple of (optimal_k, metrics)
        optimal_k : int
        metrics : dict with method-specific results
    """
    method = method.lower().strip()

    if method == "silhouette":
        return find_optimal_k_silhouette(E, min_k, max_k, seed)
    elif method == "elbow":
        return find_optimal_k_elbow(E, min_k, max_k, seed)
    elif method == "gap":
        return find_optimal_k_gap_statistic(E, min_k, max_k, seed=seed)
    elif method == "combined":
        return find_optimal_k_combined(E, min_k, max_k, seed=seed)
    else:
        logger.warning(
            f"[ProbeSelector] Unknown method '{method}', using 'combined'."
        )
        return find_optimal_k_combined(E, min_k, max_k, seed=seed)


class ProbeSelector:
    """
    Unified probe selector supporting multiple methods.

    Parameters
    ----------
    method : str
        Method to use for probe selection: "d_optimal" or "kmeans".
    probe_size : int
        Number of probes to select. If None, optimal K is automatically found.
    seed : int
        Random seed for reproducibility.
    min_k : int
        Minimum K to try when finding optimal K. Default: 5.
    max_k : int
        Maximum K to try when finding optimal K. Default: 50.
    optimal_k_method : str
        Method to find optimal K when probe_size is None:
        - "silhouette": maximizes silhouette score
        - "elbow": uses second derivative of inertia curve
        - "gap": uses gap statistic
        - "combined": voting across all three methods (default)
    """

    def __init__(
        self,
        method: str = "kmeans",
        probe_size: Optional[int] = None,
        seed: int = 42,
        min_k: int = 5,
        max_k: int = 50,
        optimal_k_method: str = "combined",
    ) -> None:
        self.method = method.lower()
        self.probe_size = probe_size
        self.seed = seed
        self.min_k = min_k
        self.max_k = max_k
        self.optimal_k_method = optimal_k_method.lower()

        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_embeddings_: Optional[np.ndarray] = None
        self.selected_queries_: Optional[List[Dict]] = None
        self.optimal_k_: Optional[int] = None
        self.optimal_k_metrics_: Optional[Dict[str, Any]] = None

    def fit(
        self,
        E: np.ndarray,
        probe_size: Optional[int] = None,
        routing_data_all: Optional[List[Dict]] = None,
    ) -> "ProbeSelector":
        """
        Select probe set from embedding matrix E.

        Parameters
        ----------
        E : np.ndarray, shape (n_samples, embed_dim)
            Candidate embedding vectors.
        probe_size : int, optional
            Override instance probe_size if provided.
        routing_data_all : list of dict, optional
            The full pool of query dicts in the normalised wide format
            (same length as E).  When provided, ``selected_queries_`` is
            populated with the selected entries in the same format as the
            original input data.

        Returns
        -------
        self
        """
        k = probe_size if probe_size is not None else self.probe_size

        if k is None:
            logger.info(
                f"[ProbeSelector] Finding optimal K via {self.optimal_k_method} "
                f"(range: {self.min_k}-{self.max_k})..."
            )
            k, self.optimal_k_metrics_ = select_optimal_k_for_probes(
                E,
                min_k=self.min_k,
                max_k=self.max_k,
                method=self.optimal_k_method,
                seed=self.seed,
            )
            self.optimal_k_ = k

        if self.method == "d_optimal":
            self.selected_indices_, self.selected_embeddings_ = select_probe_set_d_optimal(
                E, probe_size=k, seed=self.seed
            )
        elif self.method == "kmeans":
            self.selected_indices_, self.selected_embeddings_ = select_probe_set_kmeans(
                E, probe_size=k, seed=self.seed
            )
        else:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                f"Supported: 'd_optimal', 'kmeans'"
            )

        if routing_data_all is not None:
            self.selected_queries_ = get_selected_queries(
                self.selected_indices_, routing_data_all
            )
            logger.info(
                f"[ProbeSelector] Mapped {len(self.selected_queries_)} probe indices "
                "to query dicts from routing_data_all."
            )

        logger.info(
            f"[ProbeSelector] Selected {len(self.selected_indices_)} probes "
            f"using {self.method} (K={k or self.optimal_k_})"
        )

        return self

    def transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return selected indices and embeddings.

        Returns
        -------
        tuple of (indices, embeddings)
        """
        if self.selected_indices_ is None:
            raise RuntimeError("Must call fit() before transform()")
        return self.selected_indices_, self.selected_embeddings_

    def fit_transform(
        self,
        E: np.ndarray,
        probe_size: Optional[int] = None,
        routing_data_all: Optional[List[Dict]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and return selected probes in one call.

        Parameters
        ----------
        E : np.ndarray, shape (n_samples, embed_dim)
            Candidate embedding vectors.
        probe_size : int, optional
            Override instance probe_size if provided.
        routing_data_all : list of dict, optional
            When provided, ``selected_queries_`` is populated with the
            selected query dicts in the same format as the original input.

        Returns
        -------
        tuple of (indices, embeddings)
        """
        self.fit(E, probe_size, routing_data_all=routing_data_all)
        return self.transform()

    def save_selected_queries(self, path: str) -> None:
        """
        Save selected queries to a pickle file.

        Parameters
        ----------
        path : str
            Path to save the selected queries (should have .pkl extension).
        """
        import pickle
        from pathlib import Path

        if self.selected_queries_ is None:
            raise RuntimeError(
                "Must call fit() with routing_data_all before saving selected queries."
            )

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(self.selected_queries_, f)

        logger.info(
            f"[ProbeSelector] Saved {len(self.selected_queries_)} selected queries to {path}"
        )

    def load_selected_queries(self, path: str) -> None:
        """
        Load selected queries from a pickle file.

        Parameters
        ----------
        path : str
            Path to load the selected queries from.
        """
        import pickle
        from pathlib import Path

        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Selected queries file not found: {path}")

        with open(file_path, "rb") as f:
            self.selected_queries_ = pickle.load(f)

        logger.info(
            f"[ProbeSelector] Loaded {len(self.selected_queries_)} selected queries from {path}"
        )
