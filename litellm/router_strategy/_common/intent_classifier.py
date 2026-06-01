"""
Intent Classifier
=================
Zero-shot intent classifier that categorises an incoming query into one of
several high-level intent classes *before* it reaches the cluster router.

Strategy
--------
Each intent class is represented by a small set of natural-language *prototype
sentences*.  At initialisation time the prototypes are embedded and averaged to
produce a single centroid per class.  At inference time the query embedding is
compared (cosine similarity) against every centroid and the closest class is
returned.

Because the prototypes are embedded with the *same* model already used for
clustering, no additional API calls or model downloads are needed.

Classes (configurable)
----------------------
- ``math``         : arithmetic, algebra, calculus, statistics …
- ``coding``       : programming, debugging, algorithms, data structures …
- ``science``      : physics, chemistry, biology, engineering …
- ``reasoning``    : logic puzzles, causal inference, common-sense …
- ``factual_qa``   : knowledge look-up, definitions, who/what/when/where …
- ``language``     : translation, grammar, writing, summarisation …
- ``general``      : fallback for queries that don't fit any class above
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import logging

from .intent_cache import IntentCache

logger = logging.getLogger(__name__)


__all__ = ["IntentClassifier", "DEFAULT_INTENT_CLASSES"]

# ---------------------------------------------------------------------------
# Default intent taxonomy
# ---------------------------------------------------------------------------

DEFAULT_INTENT_CLASSES: Dict[str, List[str]] = {
    "affection": [
        "Predict the emotion of this sentence. 'I got the job! I start on Monday!'",
        "Predict the emotion of this sentence. 'You never listen to me when I'm trying to explain something!'",
        "Predict the emotion of this sentence. 'Wait, so the plane leaves at 8 AM tomorrow, not PM?'",
        "Predict the emotion of this sentence. 'I just can not believe they canceled the show. I loved it so much.'",
        "What type of emotion is in this sentence? 'Ugh, this milk smells like it went bad three weeks ago.'",
        "What type of emotion is in this sentence? 'Could you pass me the salt?'",
        "What type of emotion is in this sentence? 'Did you hear that noise? I think someone is in the house.'",
        "Analysis the emotion of this sentence. 'Ross, we were on a break!'",
        "Analysis the emotion of this sentence. 'I just spilled coffee all over my brand-new laptop.'",
        "Analysis the emotion of this sentence. 'The quarterly report is due on Friday afternoon.'",
    ],
    "math": [
        "Solve this calculus problem.",
        "Compute the integral of the function.",
        "What is the derivative of x squared?",
        "Find the roots of the quadratic equation.",
        "Calculate the probability of rolling two sixes.",
        "Prove that the sum of angles in a triangle is 180 degrees.",
        "Evaluate this algebraic expression step by step.",
        "Solve the system of linear equations.",
        "What is the eigenvalue of this matrix?",
        "Simplify the following mathematical expression.",
    ],
    "coding": [
        "Write a Python function that sorts a list.",
        "Debug this JavaScript code.",
        "Write a function that takes an integer $n$ and returns the sum of all numbers from 1 to $n$.",
        "Explain how recursion works in programming.",
        "What is the time complexity of merge sort?",
        "def reverse_vowels(s: str) -> str: Reverse only the vowels of the string and return it",
        "Fix the bug in this SQL query.",
        "Write unit tests for this class.",
        "How do I use Docker containers?",
        "Implement a graph traversal algorithm.",
    ],
    "science": [
        "Explain Newton's second law of motion.",
        "What is the difference between DNA and RNA?",
        "How does nuclear fission work?",
        "Describe the process of photosynthesis.",
        "What is the Heisenberg uncertainty principle?",
        "Explain the theory of relativity.",
        "What causes tectonic plate movement?",
        "How are vaccines developed?",
        "What is the chemical formula for glucose?",
        "Explain the concept of entropy in thermodynamics.",
    ],
    "reasoning": [
        "If all A are B and some B are C, what can we conclude?",
        "Identify the logical fallacy in this argument.",
        "What would happen if we removed all regulations?",
        "Analyse the cause-and-effect relationship here.",
        "Solve this lateral thinking puzzle.",
        "Is this argument valid or invalid? Justify your answer.",
        "Compare and contrast these two positions.",
        "Evaluate the credibility of this claim.",
        "Given these premises, what follows?",
        "What are the implications of this decision?",
    ],
    "factual_qa": [
        "Who invented the telephone?",
        "When did World War II end?",
        "What is the capital of Australia?",
        "How many planets are in the solar system?",
        "What is the boiling point of water?",
        "Name the bones in the human hand.",
        "What is the GDP of Japan?",
        "Define the term 'inflation'.",
        "Who wrote Pride and Prejudice?",
        "What is the speed of light?",
    ],
    "language": [
        "Translate this sentence into French.",
        "Summarise this article in three sentences.",
        "Correct the grammar in this paragraph.",
        "Write a professional email declining the offer.",
        "What is the difference between 'affect' and 'effect'?",
        "Paraphrase the following passage.",
        "Identify the literary devices in this poem.",
        "Write a short story in the style of Hemingway.",
        "Improve the clarity of this business report.",
        "What does this idiom mean?",
    ],
    "general": [
        "Tell me something interesting.",
        "What are your thoughts on this topic?",
        "Give me advice on how to handle this situation.",
        "Help me brainstorm ideas for my project.",
        "What should I consider when making this decision?",
        "Can you help me plan my day?",
        "What are the pros and cons of this approach?",
        "I need help with something.",
        "What do you recommend?",
        "Explain this concept to me.",
    ],
}


class IntentClassifier:
    """
    Zero-shot intent classifier backed by an ``EmbeddingCache``.

    Parameters
    ----------
    embedding_cache : EmbeddingCache
        Shared embedding cache; prototypes are stored in the same DB.
    intent_classes : dict[str, list[str]], optional
        Mapping from class label to prototype sentences.  Defaults to
        :data:`DEFAULT_INTENT_CLASSES`.
    prototype_cache_path : str | Path, optional
        JSON file path for persisting computed class centroids so they
        survive process restarts.  Defaults to
        ``<embedding_cache.base_db_path>.intent_centroids.json``.
    intent_cache_dir : str | Path, optional
        Directory to store the intent classification cache. Defaults to
        the same directory as the centroids file.
    skip_intent_cache : bool, optional
        If True, disable the intent classification cache. Defaults to False.
    """

    def __init__(
        self,
        embedding_cache,
        intent_classes: Optional[Dict[str, List[str]]] = None,
        prototype_cache_path: Optional[str | os.PathLike] = None,
        intent_cache_dir: Optional[str | os.PathLike] = None,
        skip_intent_cache: bool = False,
    ) -> None:
        self._cache = embedding_cache
        self._classes = intent_classes or DEFAULT_INTENT_CLASSES
        self._labels: List[str] = list(self._classes.keys())
        self._skip_intent_cache = skip_intent_cache

        if prototype_cache_path is None:
            prototype_cache_path = Path(
                str(embedding_cache.base_db_path) + ".intent_centroids.json"
            )
        self._centroid_path = Path(prototype_cache_path)

        self._centroids: Optional[np.ndarray] = None  # shape (n_classes, dim)
        self._build_centroids()

        self._intent_cache: Optional[IntentCache] = None
        if not self._skip_intent_cache:
            self._intent_cache = IntentCache(
                centroids_path=self._centroid_path,
                intent_labels=self._labels,
                cache_dir=intent_cache_dir,
            )
            logger.info(
                f"IntentClassifier: using intent cache at {self._intent_cache._db_path} "
                f"(key={self._intent_cache.cache_key})"
            )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def classify(self, query: str) -> str:
        """Return the most likely intent class label for *query*."""
        label, _ = self.classify_with_score(query)
        return label

    def classify_with_score(self, query: str) -> Tuple[str, float]:
        """Return ``(label, cosine_similarity_score)`` for *query*."""
        if self._centroids is None:
            return "general", 0.0
        q_emb = np.array(self._cache.get(query, True), dtype=np.float32)
        q_emb /= np.linalg.norm(q_emb) + 1e-12
        sims = self._centroids @ q_emb
        best_idx = int(np.argmax(sims))
        return self._labels[best_idx], float(sims[best_idx])

    def classify_batch(self, queries: List[str]) -> List[str]:
        """Classify a batch of queries efficiently, with caching."""
        if self._centroids is None:
            return ["general"] * len(queries)

        if self._intent_cache is not None:
            cached = self._intent_cache.get_many(queries)
            cached_results: List[Optional[str]] = [cached.get(i) for i in range(len(queries))]
            miss_indices = [i for i, r in enumerate(cached_results) if r is None]

            if not miss_indices:
                return cached_results

            miss_queries = [queries[i] for i in miss_indices]
            miss_embeddings = np.array(self._cache.batch(miss_queries, True), dtype=np.float32)
            norms = np.linalg.norm(miss_embeddings, axis=1, keepdims=True) + 1e-12
            miss_embeddings /= norms
            sims = miss_embeddings @ self._centroids.T
            best_indices = np.argmax(sims, axis=1)
            miss_labels = [self._labels[i] for i in best_indices]

            self._intent_cache.put_many(miss_queries, miss_labels)

            results = cached_results[:]
            for idx, label in zip(miss_indices, miss_labels):
                results[idx] = label
            return results

        embeddings = np.array(self._cache.batch(queries, True), dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings /= norms
        sims = embeddings @ self._centroids.T
        best_indices = np.argmax(sims, axis=1)
        return list(self._labels[best_indices])

    @property
    def labels(self) -> List[str]:
        """All intent class labels in order."""
        return list(self._labels)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_centroids(self) -> None:
        """Load centroids from disk or compute them from prototypes."""
        if self._centroid_path.exists():
            try:
                data = json.loads(self._centroid_path.read_text())
                if data.get("labels") == self._labels:
                    self._centroids = np.array(data["centroids"], dtype=np.float32)
                    logger.info(
                        f"IntentClassifier: loaded {len(self._labels)} centroids from {self._centroid_path}"
                    )
                    return
            except Exception as e:
                logger.warning(
                    f"IntentClassifier: failed to load cached centroids ({e}); recomputing."
                )

        self._compute_and_save_centroids()

    def _compute_and_save_centroids(self) -> None:
        """Embed all prototypes, average per class, L2-normalise, and save."""
        logger.info(
            f"IntentClassifier: computing centroids for {len(self._labels)} classes …"
        )
        centroids = []

        for label in self._labels:
            prototypes = self._classes[label]
            embeddings = self._cache.batch(prototypes, True)
            arr = np.array(embeddings, dtype=np.float32)
            centroid = arr.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            centroids.append(centroid.tolist())

        self._centroids = np.array(centroids, dtype=np.float32)

        # Persist to disk
        try:
            payload = {"labels": self._labels, "centroids": centroids}
            self._centroid_path.write_text(json.dumps(payload))
            logger.info(f"IntentClassifier: centroids saved to {self._centroid_path}")
        except Exception as e:
            logger.warning(f"IntentClassifier: could not save centroids ({e})")
