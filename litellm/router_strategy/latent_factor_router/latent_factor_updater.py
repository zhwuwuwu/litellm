"""
LatentFactorRouter — latent_factor_updater.py
=============================================
Incremental model-pool management for :class:`LatentFactorRouter`.

Problem
-------
After a LatentFactorRouter is trained, its latent factor model is fixed.
When a new LLM becomes available, we need to learn its latent representations
(Lp_new, Lc_new) without retraining the entire model.

Solution
--------
Use the probe set selected during training to evaluate the new LLM.
The probe set maximizes embedding diversity, so evaluating on it gives us
the most informative signal for learning the new LLM's latent representation.

We solve:
    E_probe @ Lp_new = P_probe_new  (performance on probe set)
    E_probe @ Lc_new = C_probe_new  (cost on probe set)

using the same ridge regression approach as the main model.

Typical usage
-------------
::

    from latentfactorrouter import LatentFactorRouter, LatentFactorUpdater

    router  = LatentFactorRouter("config_latent_factor.yaml")
    trainer = LatentFactorTrainer(router)
    trainer.train()

    updater = LatentFactorUpdater(router)

    probe_results = [
        {"query": "...", "performance": 0.9, "cost": 0.003},
        ...
    ]
    summary = updater.add_model(
        model_name="gpt-5",
        probe_results=probe_results,
        save_path="latent_factor_artefacts.pkl",
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg

if TYPE_CHECKING:
    from .router import LatentFactorRouter

logger = logging.getLogger(__name__)


class LatentFactorUpdater:
    """
    Incrementally extend a trained :class:`LatentFactorRouter` with new LLMs.

    Uses the probe set to learn new LLM latent representations via ridge regression.

    Parameters
    ----------
    router : LatentFactorRouter
        A trained router instance.
    """

    def __init__(self, router: "LatentFactorRouter") -> None:
        self.router = router

    def add_model(
        self,
        model_name: str,
        probe_results: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a single new LLM using probe set evaluation results.

        Parameters
        ----------
        model_name : str
            Identifier for the new model.
        probe_results : list[dict]
            Evaluation results for the new model on the probe set.
            Each dict should have:
                - "query": str
                - "performance": float (0-1 scale)
                - "cost": float
            These queries should match the probe set queries selected during training.
        save_path : str, optional
            If given, the updated artefact is persisted immediately.

        Returns
        -------
        dict
            Update summary with model_name, n_probes, and success status.
        """
        router = self.router
        router._require_trained()

        if not model_name:
            raise ValueError("model_name must be a non-empty string.")

        if model_name in router._available_models:
            raise ValueError(
                f"Model '{model_name}' is already in the pool. "
                f"Use remove_model() first if you want to replace it."
            )

        if not probe_results:
            raise ValueError(
                f"probe_results for '{model_name}' is empty. "
                "Provide evaluation results on at least one probe query."
            )

        probe_indices = router._probe_indices
        probe_embeddings = router._probe_embeddings

        if probe_embeddings is None or probe_indices is None:
            raise RuntimeError(
                "Probe set not available. The router may not have been trained "
                "with probe selection enabled, or the artefact was loaded from "
                "an older version without probe information."
            )

        probe_query_map: Dict[str, Tuple[float, float]] = {}
        for item in probe_results:
            q = item.get("query", "")
            perf = float(item.get("performance", 0.0))
            cost = float(item.get("cost", 0.0))
            probe_query_map[q] = (perf, cost)

        E_probe_list: List[np.ndarray] = []
        P_probe_list: List[float] = []
        C_probe_list: List[float] = []

        for i, idx in enumerate(probe_indices):
            train_item = router._train_data[idx]
            q = train_item["query"]
            if q in probe_query_map:
                perf, cost = probe_query_map[q]
            else:
                logger.warning(
                    f"Probe query at index {idx} not found in probe_results. "
                    f"Query: {q[:50]}... Skipping."
                )
                continue
            E_probe_list.append(probe_embeddings[i])
            P_probe_list.append(perf)
            C_probe_list.append(cost)

        if len(E_probe_list) < 3:
            raise ValueError(
                f"Only {len(E_probe_list)} probe queries matched. "
                "Need at least 3 for ridge regression."
            )

        E_probe = np.array(E_probe_list, dtype=np.float64)
        P_probe = np.array(P_probe_list, dtype=np.float64)
        C_probe = np.array(C_probe_list, dtype=np.float64)

        logger.info(
            f"[LatentFactorUpdater] Learning latent representation for '{model_name}' "
            f"using {len(E_probe)} probe points."
        )

        alpha = router.alpha

        # The trained model uses bias-augmented embeddings (E_aug = [E | 1]).
        # We must match that augmentation here so Lp_new/Lc_new have the same
        # first dimension as model.Lp/model.Lc (embed_dim + 1, not embed_dim).
        n_p = E_probe.shape[0]
        E_probe_aug = np.hstack([E_probe, np.ones((n_p, 1), dtype=E_probe.dtype)])

        E_T_E = E_probe_aug.T @ E_probe_aug + alpha * np.eye(E_probe_aug.shape[1])

        Lp_new = linalg.solve(E_T_E, E_probe_aug.T @ P_probe)
        Lc_new = linalg.solve(E_T_E, E_probe_aug.T @ C_probe)

        Lp_new = Lp_new.reshape(-1, 1)
        Lc_new = Lc_new.reshape(-1, 1)

        model = router._latent_model

        Lp_old = model.Lp
        Lc_old = model.Lc

        model.Lp = np.concatenate([Lp_old, Lp_new], axis=1)
        model.Lc = np.concatenate([Lc_old, Lc_new], axis=1)

        router._available_models.append(model_name)

        logger.info(
            f"[LatentFactorUpdater] '{model_name}' added to pool. "
            f"Lp new shape: {model.Lp.shape}, Lc new shape: {model.Lc.shape}"
        )

        if save_path:
            router.save_artefacts(save_path)
            logger.info(f"[LatentFactorUpdater] Artefacts saved to {save_path}")

        return {
            "model_name": model_name,
            "n_probes_matched": len(E_probe_list),
            "n_probes_total": len(probe_indices),
            "Lp_shape": model.Lp.shape,
            "Lc_shape": model.Lc.shape,
            "saved": save_path is not None,
        }

    def add_model_by_embedding(
        self,
        model_name: str,
        eval_data: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a new LLM when full probe evaluation is not available.

        For each eval_data item, we embed the query and build the latent
        representation using only the queries that are available.

        Parameters
        ----------
        model_name : str
            Identifier for the new model.
        eval_data : list[dict]
            Evaluation results with "query", "performance", "cost" keys.
        save_path : str, optional

        Returns
        -------
        dict
        """
        router = self.router
        router._require_trained()

        if not model_name:
            raise ValueError("model_name must be a non-empty string.")

        if model_name in router._available_models:
            raise ValueError(
                f"Model '{model_name}' is already in the pool."
            )

        if not eval_data:
            raise ValueError("eval_data is empty.")

        eval_lookup: Dict[str, Tuple[float, float]] = {}
        for item in eval_data:
            q = item.get("query", "")
            perf = float(item.get("performance", 0.0))
            cost = float(item.get("cost", 0.0))
            eval_lookup[q] = (perf, cost)

        router.embedder.precompute(list(eval_lookup.keys()), desc="Embedding eval queries")

        E_list: List[np.ndarray] = []
        P_list: List[float] = []
        C_list: List[float] = []

        for q, (perf, cost) in eval_lookup.items():
            emb = router.embedder.batch([q])
            if emb and len(emb) > 0:
                E_list.append(np.array(emb[0], dtype=np.float64))
                P_list.append(perf)
                C_list.append(cost)

        if len(E_list) < 3:
            raise ValueError(
                f"Only {len(E_list)} eval queries embedded successfully. "
                "Need at least 3 for ridge regression."
            )

        E_eval = np.array(E_list, dtype=np.float64)
        P_eval = np.array(P_list, dtype=np.float64)
        C_eval = np.array(C_list, dtype=np.float64)

        alpha = router.alpha

        # Match the bias augmentation used during training.
        n_e = E_eval.shape[0]
        E_eval_aug = np.hstack([E_eval, np.ones((n_e, 1), dtype=E_eval.dtype)])

        E_T_E = E_eval_aug.T @ E_eval_aug + alpha * np.eye(E_eval_aug.shape[1])

        Lp_new = linalg.solve(E_T_E, E_eval_aug.T @ P_eval)
        Lc_new = linalg.solve(E_T_E, E_eval_aug.T @ C_eval)

        Lp_new = Lp_new.reshape(-1, 1)
        Lc_new = Lc_new.reshape(-1, 1)

        model = router._latent_model
        model.Lp = np.concatenate([model.Lp, Lp_new], axis=1)
        model.Lc = np.concatenate([model.Lc, Lc_new], axis=1)

        router._available_models.append(model_name)

        logger.info(
            f"[LatentFactorUpdater] '{model_name}' added via embedding path. "
            f"n_eval={len(E_list)}"
        )

        if save_path:
            router.save_artefacts(save_path)

        return {
            "model_name": model_name,
            "n_eval_queries": len(E_list),
            "Lp_shape": model.Lp.shape,
            "Lc_shape": model.Lc.shape,
            "saved": save_path is not None,
        }

    def remove_model(
        self,
        model_name: str,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Remove an LLM from the routing pool.

        Parameters
        ----------
        model_name : str
        save_path : str, optional

        Returns
        -------
        dict
        """
        router = self.router
        router._require_trained()

        if model_name not in router._available_models:
            raise ValueError(
                f"Model '{model_name}' is not in the pool. "
                f"Available: {router._available_models}"
            )

        model_idx = router._available_models.index(model_name)

        model = router._latent_model
        model.Lp = np.delete(model.Lp, model_idx, axis=1)
        model.Lc = np.delete(model.Lc, model_idx, axis=1)

        router._available_models.remove(model_name)

        logger.info(f"[LatentFactorUpdater] '{model_name}' removed from pool.")

        if save_path:
            router.save_artefacts(save_path)

        return {
            "model_name": model_name,
            "Lp_shape": model.Lp.shape,
            "Lc_shape": model.Lc.shape,
            "saved": save_path is not None,
        }

    def update_model(
        self,
        model_name: str,
        probe_results: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing model's latent representation with new evaluation data.

        Parameters
        ----------
        model_name : str
        probe_results : list[dict]
        save_path : str, optional

        Returns
        -------
        dict
        """
        router = self.router
        router._require_trained()

        if model_name not in router._available_models:
            raise ValueError(f"Model '{model_name}' is not in the pool.")

        probe_indices = router._probe_indices
        probe_embeddings = router._probe_embeddings

        if probe_embeddings is None:
            raise RuntimeError("Probe set not available.")

        probe_query_map: Dict[str, Tuple[float, float]] = {}
        for item in probe_results:
            q = item.get("query", "")
            perf = float(item.get("performance", 0.0))
            cost = float(item.get("cost", 0.0))
            probe_query_map[q] = (perf, cost)

        E_probe_list: List[np.ndarray] = []
        P_probe_list: List[float] = []
        C_probe_list: List[float] = []

        for i, idx in enumerate(probe_indices):
            train_item = router._train_data[idx]
            q = train_item["query"]
            if q in probe_query_map:
                perf, cost = probe_query_map[q]
                E_probe_list.append(probe_embeddings[i])
                P_probe_list.append(perf)
                C_probe_list.append(cost)

        if len(E_probe_list) < 3:
            raise ValueError(
                f"Only {len(E_probe_list)} probe queries matched. "
                "Need at least 3 for ridge regression."
            )

        E_probe = np.array(E_probe_list, dtype=np.float64)
        P_probe = np.array(P_probe_list, dtype=np.float64)
        C_probe = np.array(C_probe_list, dtype=np.float64)

        model_idx = router._available_models.index(model_name)
        alpha = router.alpha
        E_T_E = E_probe.T @ E_probe + alpha * np.eye(E_probe.shape[1])

        Lp_new = linalg.solve(E_T_E, E_probe.T @ P_probe)
        Lc_new = linalg.solve(E_T_E, E_probe.T @ C_probe)

        model = router._latent_model
        model.Lp[:, model_idx] = Lp_new.reshape(-1)
        model.Lc[:, model_idx] = Lc_new.reshape(-1)

        logger.info(
            f"[LatentFactorUpdater] '{model_name}' updated with new probe data."
        )

        if save_path:
            router.save_artefacts(save_path)

        return {
            "model_name": model_name,
            "n_probes_matched": len(E_probe_list),
            "Lp_shape": model.Lp.shape,
            "Lc_shape": model.Lc.shape,
            "saved": save_path is not None,
        }
