"""
LatentFactorRouter — router.py
==============================
Ridge-regression-based LLM router.

Uses query embeddings to predict performance and cost for each LLM via:
    E @ Lp = P  (performance)
    E @ Lc = C  (cost)

where E is the query representation matrix.

Features:
- 5-fold cross-validation during training
- Ridge/Tikhonov regularization
- Handles missing values via weighted least squares
- Probe set selection for cold-start new LLMs
- PCA for dimensionality reduction (optional)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from litellm.router_strategy._llmrouter import MetaRouter
from litellm.router_strategy._llmrouter import save_model, load_model

from litellm.router_strategy._common import (
    find_pareto_optimal_models,
    load_all_routing_data,
    split_stratified_two_way,
    compute_baselines as _compute_baselines_common,
)
from litellm.router_strategy._common.data_utils import (
    normalise_rows as _normalise_rows_impl,
    pivot_long_rows as _pivot_long_rows_impl,
    normalise_wide_rows as _normalise_wide_rows_impl,
    pivot_df as _pivot_df_impl,
)

from litellm.router_strategy._common.embedding_cache import EmbeddingCache
from litellm.router_strategy._common.intent_classifier import IntentClassifier

if TYPE_CHECKING:
    from .latent_factor_model import (
        LatentFactorModel,
        LatentFactorModelWithCV,
        ResidualBoostedLatentFactorModel,
    )

logger = logging.getLogger(__name__)


@dataclass
class LatentFactorConfig:
    latent_dim: int = 32
    alpha: float = 1.0
    n_folds: int = 5
    probe_size: Optional[int] = None
    probe_method: str = "kmeans"
    enable_pca: bool = False
    pca_n_components: float = 0.99
    optimal_k_method: str = "combined"
    optimal_k_min: int = 5
    optimal_k_max: int = 50
    enable_residual_boost: bool = False
    boost_rounds: int = 100
    boost_max_depth: int = 3
    boost_lr: float = 0.1
    boost_min_samples_leaf: int = 20
    embedding_cache: Optional[bool] = None


class LatentFactorRouter(MetaRouter):
    """
    LatentFactorRouter — Ridge-regression-based LLM router.

    Uses query embeddings to predict performance and cost for each LLM via:
        E @ Lp = P  (performance)
        E @ Lc = C  (cost)

    where E is the query representation matrix.

    Parameters
    ----------
    yaml_path : str
        Path to configuration YAML.
    """

    def __init__(self, yaml_path: str) -> None:
        super().__init__(model=None, yaml_path=yaml_path)

        cfg = self.cfg

        log_cfg = cfg.get("logging", {})
        log_level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
        root_logger = logging.getLogger("latentfactorrouter")
        if not root_logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(log_level)
            formatter = logging.Formatter(
                "%(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
            root_logger.setLevel(log_level)

        emb_cfg = cfg.get("embedding", {})
        dataset_name = cfg.get("dataset", {}).get("name", "default")
        api_key = (emb_cfg.get("api_key") or "").strip()
        if api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = (os.getenv(env_var) or "").strip()
            if not api_key:
                raise ValueError(
                    f"api_key is empty after env-var substitution (${{{env_var}}} resolved to empty/unset)"
                )
        elif not api_key:
            api_key = (os.environ.get("LOCAL_EMBEDDING_API_KEY") or "").strip()

        base_url = (emb_cfg.get("base_url") or "").strip()
        if base_url.startswith("${") and base_url.endswith("}"):
            env_var = base_url[2:-1]
            base_url = (os.getenv(env_var) or "").strip()
            if not base_url:
                raise ValueError(
                    f"base_url is empty after env-var substitution (${{{env_var}}} resolved to empty/unset)"
                )
        elif not base_url:
            base_url = (os.environ.get("LOCAL_EMBEDDING_API_BASE") or "").strip()
            if not base_url:
                base_url = "http://127.0.0.1:18104/v1"

        model_name = (emb_cfg.get("model_name") or "").strip()
        if model_name.startswith("${") and model_name.endswith("}"):
            env_var = model_name[2:-1]
            model_name = (os.getenv(env_var) or "").strip()
            if not model_name:
                model_name = "text-embedding-3-large"
        elif not model_name:
            model_name = (os.environ.get("LOCAL_EMBEDDING_MODEL") or "").strip()
            if not model_name:
                model_name = "text-embedding-3-large"
        cache_dir = emb_cfg.get("cache_dir", ".cache/latentfactorrouter_embeddings")

        self.enable_pca = bool(emb_cfg.get("enable_pca", False))
        self.pca_n_components = float(emb_cfg.get("pca_n_components", 0.99))
        pca_model_path = emb_cfg.get("pca_model_path", "")

        self._max_tokens = int(emb_cfg.get("max_tokens", 7500))
        self._truncate_buffer = float(emb_cfg.get("truncate_buffer", 0.9))
        self._truncate_threshold = int(self._max_tokens * self._truncate_buffer)

        resolved_pca_model_path = None
        if self.enable_pca and pca_model_path:
            if os.path.isabs(pca_model_path):
                resolved_pca_model_path = pca_model_path
            else:
                project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../..")
                )
                resolved_pca_model_path = os.path.join(project_root, pca_model_path)

        self._emb_cfg = emb_cfg
        self._dataset_name = dataset_name
        self._api_key = api_key
        self._base_url = base_url
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._resolved_pca_model_path = resolved_pca_model_path

        self.embedder = None
        self.intent_clf = None
        self._intent_enabled = True
        self._intent_classes = None
        ic_cfg = cfg.get("intent_classifier", {})
        if self._intent_enabled:
            classes_path = ic_cfg.get("classes_path")
            if classes_path and os.path.exists(classes_path):
                self._intent_classes = json.loads(Path(classes_path).read_text())

        lf_cfg = cfg.get("latent_factor", {})
        rt_cfg = cfg.get("routing", {})
        mt_cfg = cfg.get("metric", {}).get("weights", {})

        self.latent_dim = int(lf_cfg.get("latent_dim", 32))
        self.alpha = float(lf_cfg.get("alpha", 1.0))
        self.n_folds = int(lf_cfg.get("n_folds", 5))
        probe_size_raw = lf_cfg.get("probe_size", None)
        self.probe_size = int(probe_size_raw) if probe_size_raw is not None else None
        self.probe_method = str(lf_cfg.get("probe_method", "kmeans")).lower()
        self.seed = int(lf_cfg.get("seed", 42))
        self._train_val_ratio = float(lf_cfg.get("train_val_ratio", 0.9))

        optimal_k_cfg = lf_cfg.get("optimal_k", {})
        self.optimal_k_method = str(optimal_k_cfg.get("method", "combined")).lower()
        self.optimal_k_min = int(optimal_k_cfg.get("min_k", 5))
        self.optimal_k_max = int(optimal_k_cfg.get("max_k", 50))

        self.alpha_P_range = lf_cfg.get("alpha_P_range")
        self.alpha_C_range = lf_cfg.get("alpha_C_range")
        self.normalize_acc = bool(lf_cfg.get("normalize_acc", True))
        self.embed_augment = bool(lf_cfg.get("embed_augment", False))
        self.embed_noise_std = float(lf_cfg.get("embed_noise_std", 0.01))
        self.enable_residual_boost = bool(lf_cfg.get("enable_residual_boost", False))
        self.boost_rounds = int(lf_cfg.get("boost_rounds", 100))
        self.boost_max_depth = int(lf_cfg.get("boost_max_depth", 3))
        self.boost_lr = float(lf_cfg.get("boost_lr", 0.1))
        self.boost_min_samples_leaf = int(lf_cfg.get("boost_min_samples_leaf", 20))

        self.top_k = int(rt_cfg.get("top_k", 1))
        self.excluded_models = list(rt_cfg.get("excluded_models", []))
        self.min_acc_thresh = float(rt_cfg.get("min_accuracy_threshold", 0.0))
        self.budget_limit = rt_cfg.get("budget_limit")

        vb_cfg = cfg.get("verbose", {})
        self._verbose_enabled = bool(vb_cfg.get("enabled", False))

        self._perf_weight = float(mt_cfg.get("performance", 0.7))
        self._cost_weight = 1.0 - self._perf_weight
        logger.info(
            f"[LatentFactorRouter] perf_weight={self._perf_weight}, "
            f"cost_weight={self._cost_weight}"
        )

        self.all_routing_data: List[Dict] = self._load_all_data()
        logger.info(
            f"[LatentFactorRouter] Total unified data: {len(self.all_routing_data)} queries."
        )

        self._latent_model: Optional[Union["LatentFactorModel", "LatentFactorModelWithCV"]] = None
        self._available_models: List[str] = []
        self._is_trained = False

        self._train_data: List[Dict] = []
        self._val_data: List[Dict] = []
        self._test_data: List[Dict] = []

        self._P_matrix: Optional[np.ndarray] = None
        self._C_matrix: Optional[np.ndarray] = None
        self._E_matrix: Optional[np.ndarray] = None

        self._probe_indices: Optional[np.ndarray] = None
        self._probe_embeddings: Optional[np.ndarray] = None

        load_path = self._resolve_path(
            cfg.get("model_path", {}).get("load_model_path", "")
        )
        if load_path and os.path.exists(load_path):
            self.load_artefacts(load_path)
        else:
            logger.info(
                "[LatentFactorRouter] No pre-trained artefacts found. "
                "Run LatentFactorTrainer.train() first."
            )

        logger.info("[LatentFactorRouter] Initialized.")

    def _load_all_data(self) -> List[Dict]:
        return load_all_routing_data(
            cfg=self.cfg,
            resolve_path_fn=self._resolve_path,
            router_instance=self,
            router_name="LatentFactorRouter",
        )

    def _normalise_rows(self, rows: List[Dict]) -> List[Dict]:
        return _normalise_rows_impl(rows, "LatentFactorRouter")

    def _pivot_long_rows(self, rows: List[Dict]) -> List[Dict]:
        return _pivot_long_rows_impl(rows)

    def _normalise_wide_rows(self, rows: List[Dict]) -> List[Dict]:
        return _normalise_wide_rows_impl(rows)

    def _pivot_df(self, df: Any) -> List[Dict]:
        return _pivot_df_impl(df, "LatentFactorRouter")

    def _ensure_embedder_for_training(self) -> None:
        """Lazily initialize embedder and intent classifier only when needed for training."""
        if self.embedder is not None:
            return
        logger.info("[LatentFactorRouter] Initializing embedder for training (cache enabled)...")
        self.embedder = EmbeddingCache(
            base_url=self._base_url,
            api_key=self._api_key,
            model_name=self._model_name,
            dataset_name=self._dataset_name,
            cache_dir=self._cache_dir,
            enable_pca=self.enable_pca,
            pca_n_components=self.pca_n_components,
            max_tokens=self._truncate_threshold,
            batch_size=int(self._emb_cfg.get("batch_size", 64)),
            pca_model_path=self._resolved_pca_model_path,
            skip_cache=False,
        )
        logger.info(f"[LatentFactorRouter] Embedding cache -> {self.embedder.base_db_path}")
        if self._intent_enabled:
            self.intent_clf = IntentClassifier(
                embedding_cache=self.embedder,
                intent_classes=self._intent_classes,
            )
            logger.info(f"[LatentFactorRouter] Intent classes: {self.intent_clf.labels}")

    def prepare_data(
        self, raw_data: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Full pre-processing:
        1. Collect available models.
        2. Precompute ALL embeddings (one-shot, cache-first).
        3. Classify ALL intents (vectorised, before any split).
        4. Stratified 80/20 split by (dataset x intent).
           80% for train+val (5-fold CV), 20% for test only.
        5. Extract P (performance) and C (cost) matrices.
        """
        self._ensure_embedder_for_training()

        all_models: List[str] = []
        for item in raw_data:
            for m in item.get("records", {}).keys():
                if m not in all_models:
                    all_models.append(m)
        self._available_models = [m for m in all_models if m not in self.excluded_models]
        logger.info(
            f"[LatentFactorRouter] Models ({len(self._available_models)}): "
            f"{self._available_models}"
        )

        all_queries = [d["query"] for d in raw_data]
        logger.info(
            f"\n[LatentFactorRouter] Precomputing {len(all_queries)} embeddings "
            f"(cache-first, PCA={self.enable_pca})..."
        )
        self.embedder.precompute(all_queries, desc="Embedding all queries")

        logger.info("[LatentFactorRouter] Classifying intents...")
        intents = self._classify_intents(all_queries)
        for item, intent in zip(raw_data, intents):
            item["intent"] = intent
        logger.info(f"  Intent distribution: {dict(Counter(intents))}")

        train_data, test_data = self._split_stratified(raw_data)
        logger.info(
            f"[LatentFactorRouter] Split -> "
            f"train+val={len(train_data)}  test={len(test_data)}"
        )

        self._train_data = train_data
        self._test_data = test_data
        self._val_data = []

        return train_data, test_data

    def _split_stratified(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        return split_stratified_two_way(data, train_ratio=self._train_val_ratio, seed=self.seed)

    def extract_matrices(
        self, train_data: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Extract P (performance), C (cost), and E (embedding) matrices.

        Returns
        -------
        tuple of (E, P, C, model_names)
            E : np.ndarray, shape (n_instances, embed_dim or pca_dim)
            P : np.ndarray, shape (n_instances, n_llms)
            C : np.ndarray, shape (n_instances, n_llms)
            model_names : list of str
        """
        n_instances = len(train_data)
        n_llms = len(self._available_models)
        model_names = list(self._available_models)

        queries = [d["query"] for d in train_data]
        E = self._embed(queries)
        logger.info(
            f"[LatentFactorRouter] E matrix shape: {E.shape} "
            f"(PCA={self.enable_pca})"
        )

        model_to_idx = {m: j for j, m in enumerate(model_names)}
        P = np.full((n_instances, n_llms), np.nan, dtype=np.float64)
        C = np.full((n_instances, n_llms), np.nan, dtype=np.float64)

        for i, item in enumerate(train_data):
            records = item.get("records", {})
            usages = item.get("usages", {})
            if records:
                perf_entries = [
                    (model_to_idx[m], float(v)) for m, v in records.items() if m in model_to_idx
                ]
                if perf_entries:
                    model_indices, perf_values = zip(*perf_entries)
                    P[i, list(model_indices)] = perf_values
            if usages:
                cost_entries = [
                    (model_to_idx[m], float(u.get("cost", 0.0)))
                    for m, u in usages.items() if isinstance(u, dict) and m in model_to_idx
                ]
                if cost_entries:
                    model_indices, cost_values = zip(*cost_entries)
                    C[i, list(model_indices)] = cost_values

        logger.info(
            f"[LatentFactorRouter] P matrix: shape={P.shape}, "
            f"non-NaN={np.sum(~np.isnan(P))}/{P.size}"
        )
        logger.info(
            f"[LatentFactorRouter] C matrix: shape={C.shape}, "
            f"non-NaN={np.sum(~np.isnan(C))}/{C.size}"
        )

        self._E_matrix = E
        self._P_matrix = P
        self._C_matrix = C

        return E, P, C, model_names

    def fit_latent_model(
        self,
        E: np.ndarray,
        P: np.ndarray,
        C: np.ndarray,
    ) -> Union["LatentFactorModel", "LatentFactorModelWithCV", "ResidualBoostedLatentFactorModel"]:
        """
        Fit the latent factor model.

        Uses LatentFactorModelWithCV when n_folds > 1 for hyperparameter
        selection via cross-validation; otherwise uses LatentFactorModel
        with no CV.

        When enable_residual_boost is True, wraps the base model with
        ResidualBoostedLatentFactorModel for two-stage prediction:
        Stage 1: Linear latent factor model
        Stage 2: Gradient boosting on residuals

        The parameter search uses a two-phase coarse-to-fine approach:
        Phase 1 — log-spaced coarse grid over d × alpha
        Phase 2 — ternary search on log scale within best region

        Parameters
        ----------
        E : np.ndarray, shape (n_instances, embed_dim or pca_dim)
        P : np.ndarray, shape (n_instances, n_llms)
        C : np.ndarray, shape (n_instances, n_llms)

        Returns
        -------
        Union[LatentFactorModel, LatentFactorModelWithCV, ResidualBoostedLatentFactorModel]
        """
        from .latent_factor_model import LatentFactorModel, LatentFactorModelWithCV, ResidualBoostedLatentFactorModel

        if self.n_folds > 1:
            d_min_val = min(8, self.latent_dim)
            d_max_val = min(E.shape[1], np.linalg.matrix_rank(E))
            if d_max_val > d_min_val:
                n_d = min(20, d_max_val - d_min_val + 1)
                latent_dim_range = self._log_spaced_ints(d_min_val, d_max_val, n_d)
            else:
                latent_dim_range = [d_min_val]
            alpha_range = list(np.logspace(-4, 2, 9))

            alpha_P_range = getattr(self, 'alpha_P_range', None) or alpha_range
            alpha_C_range = getattr(self, 'alpha_C_range', None) or alpha_range

            base_model = LatentFactorModelWithCV(
                latent_dim_range=latent_dim_range,
                alpha_range=alpha_range,
                alpha_P_range=alpha_P_range,
                alpha_C_range=alpha_C_range,
                n_folds=self.n_folds,
                d_min=d_min_val,
                d_max=d_max_val,
                n_coarse=len(latent_dim_range),
                ternary_tol=3,
                normalize_acc=getattr(self, 'normalize_acc', True),
                embed_augment=getattr(self, 'embed_augment', False),
                embed_noise_std=getattr(self, 'embed_noise_std', 0.01),
            )

            logger.info(
                f"[LatentFactorRouter] Fitting latent model with CV: "
                f"latent_dim_range={latent_dim_range}, alpha_P_range={alpha_P_range}, alpha_C_range={alpha_C_range}, "
                f"n_folds={self.n_folds}, normalize_acc={self.normalize_acc}"
            )
        else:
            base_model = LatentFactorModel(
                latent_dim=self.latent_dim,
                alpha=self.alpha,
                n_folds=self.n_folds,
                normalize_acc=getattr(self, 'normalize_acc', True),
                embed_augment=getattr(self, 'embed_augment', False),
                embed_noise_std=getattr(self, 'embed_noise_std', 0.01),
            )

            logger.info(
                f"[LatentFactorRouter] Fitting latent model without CV: "
                f"latent_dim={self.latent_dim}, alpha={self.alpha}, n_folds={self.n_folds}"
            )

        if getattr(self, 'enable_residual_boost', False):
            model = ResidualBoostedLatentFactorModel(
                base_model=base_model,
                boost_rounds_per_llm=getattr(self, 'boost_rounds', 100),
                boost_max_depth=getattr(self, 'boost_max_depth', 3),
                boost_lr=getattr(self, 'boost_lr', 0.1),
                boost_min_samples_leaf=getattr(self, 'boost_min_samples_leaf', 20),
                random_state=getattr(self, 'seed', 42),
            )
            logger.info(
                f"[LatentFactorRouter] Residual boosting enabled: "
                f"rounds={getattr(self, 'boost_rounds', 100)}, "
                f"max_depth={getattr(self, 'boost_max_depth', 3)}, "
                f"lr={getattr(self, 'boost_lr', 0.1)}"
            )
        else:
            model = base_model

        model.fit(
            E, P, C,
            perf_weight=self._perf_weight,
            cost_weight=self._cost_weight,
        )

        self._latent_model = model
        self._is_trained = True

        return model

    def _log_spaced_ints(self, lo: int, hi: int, n: int) -> List[int]:
        """Return n unique integers log-spaced between lo and hi (inclusive)."""
        if lo >= hi:
            return [lo]
        raw = np.unique(
            np.round(np.exp(np.linspace(np.log(lo), np.log(hi), n))).astype(int)
        )
        raw = np.clip(raw, lo, hi)
        return sorted(set(raw.tolist()))

    def select_probe_set(self, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select probe set for cold-start new LLM evaluation.

        Parameters
        ----------
        E : np.ndarray, shape (n_instances, embed_dim or pca_dim)
            All candidate embeddings.

        Returns
        -------
        tuple of (probe_indices, probe_embeddings)
        """
        from litellm.router_strategy._common.probe_selector import ProbeSelector

        selector = ProbeSelector(
            method=self.probe_method,
            probe_size=self.probe_size,
            seed=self.seed,
            min_k=self.optimal_k_min,
            max_k=self.optimal_k_max,
            optimal_k_method=self.optimal_k_method,
        )

        probe_indices, probe_embeddings = selector.fit_transform(E)

        self._probe_indices = probe_indices
        self._probe_embeddings = probe_embeddings

        if self.probe_size is None:
            logger.info(
                f"[LatentFactorRouter] Probe set auto-selected: "
                f"method={self.probe_method}, optimal_k={selector.optimal_k_}, "
                f"metrics_keys={list(selector.optimal_k_metrics_.keys()) if selector.optimal_k_metrics_ else 'N/A'}"
            )
        else:
            logger.info(
                f"[LatentFactorRouter] Probe set selected: "
                f"method={self.probe_method}, size={len(probe_indices)}"
            )

        return probe_indices, probe_embeddings

    def _ensure_embedder_for_inference(self) -> None:
        """Lazily initialize embedder for inference with caching disabled."""
        if self.embedder is not None:
            return
        logger.info("[LatentFactorRouter] Initializing embedder for inference (cache disabled)...")
        logger.info(f"[LatentFactorRouter] Embedder config: base_url={self._base_url}")
        self.embedder = EmbeddingCache(
            base_url=self._base_url,
            api_key=self._api_key,
            model_name=self._model_name,
            dataset_name=self._dataset_name,
            cache_dir=self._cache_dir,
            enable_pca=self.enable_pca,
            pca_n_components=self.pca_n_components,
            max_tokens=self._truncate_threshold,
            batch_size=int(self._emb_cfg.get("batch_size", 64)),
            pca_model_path=self._resolved_pca_model_path,
            skip_cache=True,
        )

    def route_single(self, query_input: Dict[str, Any]) -> Dict[str, Any]:
        """Route a single query."""
        self._require_trained()
        if self.embedder is None:
            self._ensure_embedder_for_inference()
        query = (
            query_input["query"]
            if isinstance(query_input, dict)
            else str(query_input)
        )
        models = self._route_query(query)
        return {
            "query": query,
            "predicted_llm": models[0][0] if models else None,
            "predicted_llm_name": models[0][0] if models else None,
            "selected_models": [m[0] for m in models] if models else [],
            "balance_scores": [m[1] for m in models] if models else [],
        }

    def route_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Route a batch of queries."""
        self._require_trained()
        if self.embedder is None:
            self._ensure_embedder_for_inference()
        queries = [
            item["query"] if isinstance(item, dict) else str(item)
            for item in batch
        ]
        results = []
        for q in queries:
            models = self._route_query(q)
            results.append({
                "query": q,
                "predicted_llm": models[0][0] if models else None,
                "predicted_llm_name": models[0][0] if models else None,
                "selected_models": [m[0] for m in models] if models else [],
                "balance_scores": [m[1] for m in models] if models else [],
            })
        return results

    def compute_metrics(self, outputs: List[Dict], batch: List[Dict]) -> dict:
        """Compute routing accuracy and average cost."""
        correct = 0
        total_cost = 0.0
        n = len(batch)
        llm_request_counts: Dict[str, int] = {}
        for out, item in zip(outputs, batch):
            records = item.get("records", {})
            usages = item.get("usages", {})
            chosen = out.get("predicted_llm")
            correct += records.get(chosen, 0.0)
            if chosen and isinstance(usages.get(chosen), dict):
                total_cost += float(usages[chosen].get("cost", 0.0))
            if chosen:
                llm_request_counts[chosen] = llm_request_counts.get(chosen, 0) + 1
        return {
            "accuracy": correct / n if n > 0 else 0.0,
            "avg_cost": total_cost / n if n > 0 else 0.0,
            "total": n,
            "llm_request_counts": llm_request_counts,
        }

    def compute_metrics_from_chosen(
        self,
        chosen_models: np.ndarray,
        batch: List[Dict],
    ) -> dict:
        """
        Compute routing accuracy and average cost from chosen model indices.

        Parameters
        ----------
        chosen_models : np.ndarray, shape (n_queries,)
            Selected model index per query.
        batch : list[dict]
            Test data items with 'records' (ground truth) and 'usages' (costs).

        Returns
        -------
        dict
            accuracy, avg_cost, total, llm_request_counts
        """
        correct = 0
        total_cost = 0.0
        n = len(batch)
        llm_request_counts: Dict[str, int] = {}
        for i, item in enumerate(batch):
            records = item.get("records", {})
            usages = item.get("usages", {})
            chosen_idx = int(chosen_models[i])
            if 0 <= chosen_idx < len(self._available_models):
                chosen = self._available_models[chosen_idx]
            else:
                chosen = None
            if chosen:
                correct += records.get(chosen, 0.0)
                if isinstance(usages.get(chosen), dict):
                    total_cost += float(usages[chosen].get("cost", 0.0))
                llm_request_counts[chosen] = llm_request_counts.get(chosen, 0) + 1
        return {
            "accuracy": correct / n if n > 0 else 0.0,
            "avg_cost": total_cost / n if n > 0 else 0.0,
            "total": n,
            "llm_request_counts": llm_request_counts,
        }

    def predict_for_queries(
        self, queries: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict performance and cost for queries.

        Parameters
        ----------
        queries : list of str

        Returns
        -------
        tuple of (P_pred, C_pred)
            P_pred : np.ndarray, shape (n_queries, n_llms)
            C_pred : np.ndarray, shape (n_queries, n_llms)
        """
        self._require_trained()
        if self.embedder is None:
            self._ensure_embedder_for_inference()
        E = self._embed(queries)
        return self._latent_model.predict(E)

    def recommend_for_queries(
        self, queries: List[str]
    ) -> List[Tuple[int, float]]:
        """
        Get recommendations for queries.

        Returns
        -------
        list of (llm_id, balance_score) tuples
        """
        self._require_trained()
        if self.embedder is None:
            self._ensure_embedder_for_inference()
        E = self._embed(queries)
        return self._latent_model.recommend(E)

    def _classify_intent(self, query: str) -> str:
        return (
            self.intent_clf.classify(query)
            if self.intent_clf
            else "general"
        )

    def _classify_intents(self, queries: List[str]) -> List[str]:
        return (
            self.intent_clf.classify_batch(queries)
            if self.intent_clf
            else ["general"] * len(queries)
        )

    def _embed(self, queries: List[str]) -> np.ndarray:
        return np.array(self.embedder.batch(queries), dtype=np.float32)

    def _score_models(
        self,
        P_pred: np.ndarray,
        C_pred: np.ndarray,
        perf_weight: float,
        cost_weight: float,
    ) -> np.ndarray:
        """
        Score all models for all queries using pre-computed P/C predictions.

        Balance score = perf_weight * norm_perf - cost_weight * norm_cost

        Parameters
        ----------
        P_pred : np.ndarray, shape (n_queries, n_models)
        C_pred : np.ndarray, shape (n_queries, n_models)
        perf_weight : float
        cost_weight : float

        Returns
        -------
        np.ndarray, shape (n_queries, n_models)
            balance scores per query per model
        """
        p_max = P_pred.max(axis=1, keepdims=True) + 1e-12
        c_max = C_pred.max(axis=1, keepdims=True) + 1e-12
        p_norm = P_pred / p_max
        c_norm = C_pred / c_max

        balance = perf_weight * p_norm - cost_weight * c_norm

        if self.budget_limit is not None:
            balance = np.where(C_pred > self.budget_limit, -np.inf, balance)

        for j in range(balance.shape[1]):
            if self.min_acc_thresh > 0:
                p_col_norm = P_pred[:, j] / p_max[:, 0]
                balance[:, j] = np.where(p_col_norm < self.min_acc_thresh, -np.inf, balance[:, j])

        return balance

    def _route_query(self, query: str) -> List[Tuple[str, float]]:
        """
        Route query using the latent factor model.

        Balance score = perf_weight * norm_perf - cost_weight * norm_cost
        """
        E = self._embed([query])
        P_pred, C_pred = self._latent_model.predict(E)

        balance = self._score_models(P_pred, C_pred, self._perf_weight, self._cost_weight)[0]

        n_return = min(self.top_k, len(balance))
        top_indices = np.argsort(balance)[::-1][:n_return]

        results = []
        for idx in top_indices:
            score = float(balance[idx])
            if score != -np.inf:
                results.append((self._available_models[idx], score))
        return results

    def _require_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "[LatentFactorRouter] Not trained. "
                "Run LatentFactorTrainer.train() first."
            )

    def _resolve_path(self, rel_or_abs: str) -> str:
        if not rel_or_abs:
            return ""
        if os.path.isabs(rel_or_abs):
            return rel_or_abs
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../..")
        )
        return os.path.join(project_root, rel_or_abs)

    def evaluate(self, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive evaluation on test_data.
        """
        self._require_trained()
        self._ensure_embedder_for_training()
        queries = [d["query"] for d in test_data]
        outputs = [self.route_single({"query": q}) for q in queries]

        metrics = self.compute_metrics(outputs, test_data)

        baselines = self.compute_baselines(test_data)
        metrics["baselines"] = baselines

        router_acc = metrics["accuracy"]
        router_cost = metrics["avg_cost"]

        oracle_acc = baselines.get("oracle", {}).get("accuracy", 1.0)
        best_acc = baselines.get("best_single", {}).get("accuracy", 0.0)
        best_cost = baselines.get("best_single", {}).get("avg_cost", 0.0)
        random_acc = baselines.get("random", {}).get("accuracy", 0.0)

        metrics["oracle_gap"] = float(oracle_acc - router_acc)
        metrics["oracle_efficiency"] = (
            float(router_acc / oracle_acc) if oracle_acc > 0 else 0.0
        )
        metrics["vs_best_single"] = float(router_acc - best_acc)
        metrics["vs_random"] = float(router_acc - random_acc)
        metrics["cost_savings_vs_best"] = (
            float((best_cost - router_cost) / best_cost) if best_cost > 0 else 0.0
        )
        metrics["quality_cost_score"] = float(
            self._perf_weight * router_acc - self._cost_weight * router_cost
        )

        # Pareto analysis: include router as a candidate alongside base models
        pareto_candidates = dict(baselines.get("per_model", {}))
        pareto_candidates["LatentFactorRouter"] = {
            "accuracy": router_acc,
            "avg_cost": router_cost,
        }
        metrics["pareto_optimal_all"] = find_pareto_optimal_models(pareto_candidates)
        metrics["pareto_optimal_base"] = baselines.get("pareto_optimal", [])

        return metrics

    def verbose_eval(self, test_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Per-query evaluation comparing predicted vs ground truth performance and cost.

        Prints detailed per-query results and final statistics.

        Parameters
        ----------
        test_data : list of dict, optional
            Test data to evaluate on. If None, uses self._test_data.

        Returns
        -------
        dict
            Statistics including MAE, RMSE, MSE for performance and cost predictions.
        """
        self._require_trained()
        self._ensure_embedder_for_training()

        if test_data is None:
            test_data = self._test_data

        if not test_data:
            logger.warning("[LatentFactorRouter] No test data available for verbose_eval.")
            return {}

        queries = [d["query"] for d in test_data]
        model_names = self._available_models

        P_pred, C_pred = self.predict_for_queries(queries)

        perf_errors = []
        cost_errors = []
        perf_sq_errors = []
        cost_sq_errors = []

        logger.info("\n" + "=" * 80)
        logger.info("  VERBOSE EVALUATION — Per-Query Predicted vs Ground Truth")
        logger.info("=" * 80)

        for i, (item, p_pred, c_pred) in enumerate(zip(test_data, P_pred, C_pred)):
            query = item["query"]
            records = item.get("records", {})
            usages = item.get("usages", {})

            query_disp = query[:80] + "..." if len(query) > 80 else query

            logger.info(f"{'Model':<35} {'Pred Perf':>12} {'GT Perf':>12} {'Pred Cost':>12} {'GT Cost':>12}")
            logger.info(f"{'-'*35} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

            for j, model in enumerate(model_names):
                gt_perf = records.get(model, np.nan)
                gt_cost = None
                if model in usages and isinstance(usages[model], dict):
                    gt_cost = usages[model].get("cost")

                pred_perf = float(p_pred[j])
                pred_cost = float(c_pred[j])
                pred_perf = 1.0 if pred_perf > 0.5 else (0.5 if pred_perf == 0.5 else 0.0)

                perf_err = None
                cost_err = None
                if not np.isnan(gt_perf):
                    perf_err = pred_perf - gt_perf
                    perf_errors.append(perf_err)
                    perf_sq_errors.append(perf_err ** 2)

                if gt_cost is not None:
                    cost_err = pred_cost - gt_cost
                    cost_errors.append(cost_err)
                    cost_sq_errors.append(cost_err ** 2)

                gt_perf_str = f"{gt_perf:.4f}" if not np.isnan(gt_perf) else "N/A"
                gt_cost_str = f"{gt_cost:.6f}" if gt_cost is not None else "N/A"

                logger.info(
                    f"{model:<35} {pred_perf:>12.4f} {gt_perf_str:>12} "
                    f"{pred_cost:>12.6f} {gt_cost_str:>12}"
                )

        n_perf = len(perf_errors)
        n_cost = len(cost_errors)

        logger.info("\n" + "=" * 80)
        logger.info("  VERBOSE EVALUATION — Final Statistics")
        logger.info("=" * 80)

        stats = {}

        if n_perf > 0:
            perf_mae = float(np.mean(np.abs(perf_errors)))
            perf_rmse = float(np.sqrt(np.mean(perf_sq_errors)))
            perf_mse = float(np.mean(perf_sq_errors))
            logger.info(f"\n  Performance Prediction (n={n_perf}):")
            logger.info(f"    MAE  : {perf_mae:.4f}")
            logger.info(f"    RMSE : {perf_rmse:.4f}")
            logger.info(f"    MSE  : {perf_mse:.4f}")

            stats["performance_mae"] = perf_mae
            stats["performance_rmse"] = perf_rmse
            stats["performance_mse"] = perf_mse
        else:
            logger.info("\n  Performance Prediction: No valid ground truth found.")

        if n_cost > 0:
            cost_mae = float(np.mean(np.abs(cost_errors)))
            cost_rmse = float(np.sqrt(np.mean(cost_sq_errors)))
            cost_mse = float(np.mean(cost_sq_errors))
            logger.info(f"\n  Cost Prediction (n={n_cost}):")
            logger.info(f"    MAE  : {cost_mae:.6f}")
            logger.info(f"    RMSE : {cost_rmse:.6f}")
            logger.info(f"    MSE  : {cost_mse:.8f}")
            stats["cost_mae"] = cost_mae
            stats["cost_rmse"] = cost_rmse
            stats["cost_mse"] = cost_mse
        else:
            logger.info("\n  Cost Prediction: No valid ground truth found.")

        if n_perf > 0 and n_cost > 0:
            combined_mae = (perf_mae + cost_mae) / 2
            logger.info(f"\n  Combined MAE (perf + cost): {combined_mae:.4f}")
            stats["combined_mae"] = combined_mae

        logger.info("\n" + "=" * 80)

        return stats

    def compute_baselines(self, data: List[Dict]) -> Dict[str, Any]:
        return _compute_baselines_common(data, self._available_models, "LatentFactorRouter")

    def save_artefacts(self, path: str) -> None:
        """Persist latent model and data splits."""
        bundle = {
            "latent_model": self._latent_model,
            "available_models": self._available_models,
            "E_matrix": self._E_matrix,
            "P_matrix": self._P_matrix,
            "C_matrix": self._C_matrix,
            "probe_indices": self._probe_indices,
            "probe_embeddings": self._probe_embeddings,
            "perf_weight": self._perf_weight,
            "cost_weight": self._cost_weight,
            "latent_dim": self.latent_dim,
            "alpha": self.alpha,
            "n_folds": self.n_folds,
        }
        save_model(bundle, path)
        logger.info(f"[LatentFactorRouter] Artefacts saved -> {path}")

        data_path = path.replace(".pkl", "_data.pkl").replace(".pt", "_data.pkl")
        data_bundle = {
            "train_data": self._train_data,
            "val_data": self._val_data,
            "test_data": self._test_data,
        }
        save_model(data_bundle, data_path)
        logger.info(f"[LatentFactorRouter] Data splits saved -> {data_path}")

    def load_artefacts(self, path: str) -> None:
        """Restore artefacts."""
        # Pickle references the old module path 'latentfactorrouter'; make it resolvable.
        _custom_routers_dir = str(Path(__file__).resolve().parent.parent)
        _added = _custom_routers_dir not in sys.path
        if _added:
            sys.path.insert(0, _custom_routers_dir)
        try:
            bundle = load_model(path)
        finally:
            if _added:
                sys.path.remove(_custom_routers_dir)
        self._latent_model = bundle["latent_model"]
        self._available_models = bundle["available_models"]
        self._E_matrix = bundle.get("E_matrix")
        self._P_matrix = bundle.get("P_matrix")
        self._C_matrix = bundle.get("C_matrix")
        self._probe_indices = bundle.get("probe_indices")
        self._probe_embeddings = bundle.get("probe_embeddings")
        self._perf_weight = bundle.get("perf_weight", 0.7)
        self._cost_weight = bundle.get("cost_weight", 0.3)
        self.latent_dim = bundle.get("latent_dim", 32)
        self.alpha = bundle.get("alpha", 1.0)
        self.n_folds = bundle.get("n_folds", 5)
        self._is_trained = True
        logger.info(
            f"[LatentFactorRouter] Artefacts loaded from {path} "
            f"(models: {len(self._available_models)})"
        )

        data_path = path.replace(".pkl", "_data.pkl").replace(".pt", "_data.pkl")
        if os.path.exists(data_path):
            data_bundle = load_model(data_path)
            self._train_data = data_bundle.get("train_data", [])
            self._val_data = data_bundle.get("val_data", [])
            self._test_data = data_bundle.get("test_data", [])
            logger.info(
                f"[LatentFactorRouter] Data splits loaded from {data_path} "
                f"(train: {len(self._train_data)}, val: {len(self._val_data)}, test: {len(self._test_data)})"
            )
        else:
            self._train_data = []
            self._val_data = []
            self._test_data = []
            logger.warning(
                f"[LatentFactorRouter] Data splits file not found: {data_path}"
            )
