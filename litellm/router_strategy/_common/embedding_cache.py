"""
Embedding Cache
===============
Persistent SQLite-backed embedding cache, namespaced by (dataset, embedding_model).

Key design:
- One DB file per (dataset_slug, model_slug) pair → experiments on the same
  dataset + model reuse cached vectors without any API calls.
- Batch precompute: call `precompute()` once at experiment start to fill the
  cache for the entire dataset; subsequent lookups are purely local.
- Thread-safe WAL mode; write-lock for single-row inserts.
- Optional PCA-based dimensionality reduction for faster routing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)
from openai import APIError, OpenAI, RateLimitError
from tqdm import tqdm
import tiktoken

if TYPE_CHECKING:
    import numpy as np

__all__ = ["EmbeddingCache"]


def _slugify(name: str) -> str:
    """Convert arbitrary string to a safe filename component."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:80]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class EmbeddingCache:
    """
    Persistent, dataset-aware embedding cache with optional PCA reduction.

    Parameters
    ----------
    base_url : str
        OpenAI-compatible embedding API base URL.
    api_key : str
        API key for the embedding service.
    model_name : str
        Embedding model identifier (used as part of the cache key).
    dataset_name : str
        Logical dataset identifier (used to namespace the cache file).
        Two runs on the *same* dataset + model share a single DB file and
        therefore share all cached vectors.
    base_db_path : str | Path
        Base SQLite database path for full-size embeddings.
    pca_db_path : str | Path | None
        SQLite database path for PCA-reduced embeddings. If None, PCA is disabled.
    pca_n_components : int
        Number of PCA components to reduce embeddings to. Ignored if PCA disabled.
    enable_pca : bool
        Whether to enable PCA-based embedding reduction.
    max_retries : int
        Number of API retry attempts on transient errors.
    initial_delay : float
        Initial back-off delay (seconds); doubles on each retry.
    batch_size : int
        Maximum texts sent in a single API call when precomputing.
    max_tokens: int
        Maximum number of tokens that a LLM can take as input
    pca_model_path : str | Path
        Path to save/load the trained PCA model. Only used when enable_pca=True.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:9301/v1",
        api_key: str = "unused",
        model_name: str = "kalm-embedding-multilingual-mini-instruct-v2.5-q8_0",
        dataset_name: str = "default",
        cache_dir: str | os.PathLike = ".cache/embeddings",
        pca_n_components: float = 0.99,
        enable_pca: bool = False,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        batch_size: int = 64,
        max_tokens: int = 1024,
        pca_model_path: str | os.PathLike | None = None,
        skip_cache: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.batch_size = batch_size
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.max_tokens = max_tokens
        self.skip_cache = skip_cache

        # ── Namespaced cache file ──────────────────────────────────────────
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)

        # Base cache configuration
        base_db_name = f"{_slugify(dataset_name)}__{_slugify(model_name)}.db"
        self.base_db_path = cache_root / base_db_name

        # PCA configuration
        self.enable_pca = enable_pca
        self.pca_n_components = pca_n_components
        pca_db_name = f"{_slugify(dataset_name)}__{_slugify(model_name)}_pca.db"
        self.pca_db_path = cache_root / pca_db_name
        self.pca_model_path = Path(pca_model_path) if pca_model_path else None
        self.db_path = self.pca_db_path if self.enable_pca else self.base_db_path

        # Initialize PCA model (loaded or new)
        self._pca_model = None
        self._pca_fitted = False

        if enable_pca:
            logger.info(f"PCA enabled: {pca_n_components} components")
            logger.info(f"PCA DB path: {self.pca_db_path}")
            logger.info(f"PCA model path: {self.pca_model_path}")

            # Ensure PCA database directory exists
            self.pca_db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_pca_db()

        self._w_lock = threading.Lock()
        self._pca_w_lock = threading.Lock() if enable_pca else None
        self._init_base_db()

        if enable_pca and self.pca_model_path:
            self._load_pca_model()

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def get(self, text: str, is_intent: bool = False) -> List[float]:
        """Return the embedding for *text* (from cache or API)."""
        h = _hash(text)

        if is_intent or not self.enable_pca:
            if self.skip_cache:
                return self._fetch_and_store_base([(text, h)])[0]
            cached = self._select_base(h)
            if cached is not None:
                return cached
            return self._fetch_and_store_base([(text, h)])[0]
        else:
            if self.skip_cache:
                emb = self._fetch_and_store_base([(text, h)])[0]
                reduced = self._transform_pca(emb)
                return reduced
            cached = self._select_pca(h)
            if cached is not None:
                return cached

            full_emb = self._select_base(h)
            if full_emb is not None:
                reduced = self._transform_pca(full_emb)
                self._insert_pca(h, reduced, text)
                return reduced

            emb = self._fetch_and_store_base([(text, h)])[0]
            reduced = self._transform_pca(emb)
            self._insert_pca(h, reduced, text)
            return reduced

    def batch(self, texts: List[str], is_intent: bool = False) -> List[List[float]]:
        """Return embeddings for *texts* in order (cache-first, API for misses)."""
        import numpy as np

        hashes = [_hash(t) for t in texts]

        if not self.skip_cache and not is_intent and not self.enable_pca:
            npy_emb = self._load_from_npy(hashes, is_pca=False)
            if npy_emb is not None:
                return npy_emb.tolist()

        results: Dict[int, List[float]] = {}
        miss_indices: List[int] = []

        if self.skip_cache:
            miss_indices = list(range(len(texts)))
            base_misses = [(i, texts[i], hashes[i]) for i in miss_indices]
        elif is_intent or not self.enable_pca:
            cached = self._select_many_base(hashes)
            for i, h in enumerate(hashes):
                emb = cached.get(h)
                if emb is not None:
                    results[i] = emb
                else:
                    miss_indices.append(i)
            base_misses = [(i, texts[i], hashes[i]) for i in miss_indices]
        else:
            cached = self._select_many_pca(hashes)
            for i, h in enumerate(hashes):
                emb = cached.get(h)
                if emb is not None:
                    results[i] = emb
                else:
                    miss_indices.append(i)

            base_misses = []
            for i in miss_indices:
                h = hashes[i]
                full_emb = self._select_base(h)
                if full_emb is not None:
                    reduced = self._transform_pca(full_emb)
                    results[i] = reduced
                    self._insert_pca(h, reduced, texts[i])
                else:
                    base_misses.append((i, texts[i], h))

        if base_misses:
            fetch_pairs = [(t, h) for _, t, h in base_misses]
            fetched = self._fetch_and_store_base(fetch_pairs)

            for (i, text, h), emb in zip(base_misses, fetched):
                if is_intent or not self.enable_pca:
                    results[i] = emb
                else:
                    reduced = self._transform_pca(emb)
                    results[i] = reduced
                    self._insert_pca(h, reduced, text)

            if not is_intent and not self.enable_pca:
                miss_hashes = [h for _, _, h in base_misses]
                miss_embs = np.array(fetched, dtype=np.float32)
                hash_to_text = {h: texts[i] for i, h in enumerate(hashes)}
                self._append_to_npy(
                    [hash_to_text[h] for h in miss_hashes],
                    miss_hashes,
                    miss_embs,
                    is_pca=False,
                )

        return [results[i] for i in range(len(texts))]

    def precompute(
        self, texts: List[str], desc: str = "Precomputing embeddings"
    ) -> None:
        """
        Fetch and cache embeddings for *all* texts at experiment start.

        If PCA is enabled, this will:
        1. Ensure all full-size embeddings are in the base cache
        2. Fit PCA on all available embeddings (if not already fitted)
        3. Generate reduced embeddings and store in PCA cache

        Parameters
        ----------
        texts : list[str]
            All query texts for the current experiment.
        desc : str
            Progress-bar label.
        """
        import numpy as np

        hashes = [_hash(t) for t in texts]

        if self.enable_pca:
            # Check if PCA model exists and is fitted
            if not self._pca_fitted:
                logger.info("PCA model not found, initiating precompute phase...")
                self._precompute_with_pca(texts, hashes, desc)
            else:
                # Apply transform to all texts using existing PCA model
                self._apply_pca_to_texts(texts, hashes, desc)
        else:
            # No PCA: standard precompute on base cache
            cached = self._select_many_base(hashes)
            misses = [
                (t, h) for t, h in zip(texts, hashes) if cached.get(h) is None
            ]

            if not misses:
                logger.info(
                    "precompute: all embeddings already cached — skipping API calls."
                )
                self._build_npy_index(hashes)
                return

            logger.info(
                f"precompute: fetching {len(misses)} / {len(texts)} embeddings from API …"
            )
            for start in tqdm(range(0, len(misses), self.batch_size), desc=desc):
                chunk = misses[start : start + self.batch_size]
                self._fetch_and_store_base(chunk)

            self._build_npy_index(hashes)

    def _precompute_with_pca(
        self, texts: List[str], hashes: List[str], desc: str
    ) -> None:
        """Precompute phase with PCA: fit on all available embeddings."""
        # First, ensure all embeddings exist in base cache
        base_misses = [
            (t, h) for t, h in zip(texts, hashes) if self._select_base(h) is None
        ]

        if base_misses:
            logger.info(f"fetching {len(base_misses)} new embeddings from API...")
            for start in tqdm(range(0, len(base_misses), self.batch_size), desc=desc):
                chunk = base_misses[start : start + self.batch_size]
                self._fetch_and_store_base(chunk)

        # Now load all embeddings from base cache and fit PCA
        logger.info("loading all embeddings from base cache to fit PCA model...")
        all_embeddings = self._get_all_embeddings_from_base()

        if not all_embeddings:
            logger.warning("no embeddings found in base cache for PCA fitting")
            return

        # Fit PCA on all embeddings
        import numpy as np
        from sklearn.decomposition import PCA

        emb_array = np.array([emb for _, emb in all_embeddings])

        self._pca_model = PCA(n_components=self.pca_n_components, random_state=42)
        logger.info(
            f"fitting PCA on {len(all_embeddings)} embeddings "
            f"(input_dim={emb_array.shape[1]}, n_components={self.pca_n_components})..."
        )
        reduced_embeddings = self._pca_model.fit_transform(emb_array)
        self._pca_fitted = True
        logger.info(
            f"PCA fitted: reduced to {self._pca_model.n_components_} dimensions"
        )

        # Build hash_to_index mapping from all_embeddings to avoid duplicate-hash issues
        # all_embeddings is [(hash, emb), ...] in same order as reduced_embeddings
        hash_to_idx = {hash_: idx for idx, (hash_, _) in enumerate(all_embeddings)}

        # populate PCA cache with reduced embeddings
        logger.info("populating PCA cache with reduced embeddings...")
        records = []
        for text, h in zip(texts, hashes):
            if h in hash_to_idx:
                reduced = reduced_embeddings[hash_to_idx[h]].tolist()
                records.append((h, json.dumps(reduced), text))

        # Insert all records
        self._insert_many_pca(records)

        # Save PCA model
        if self.pca_model_path:
            self._save_pca_model()
            logger.info(f"PCA model saved to {self.pca_model_path}")

        logger.info("PCA precompute complete")

    def _apply_pca_to_texts(
        self, texts: List[str], hashes: List[str], desc: str
    ) -> None:
        """Apply existing PCA transform to all texts."""
        import numpy as np

        emb_by_hash = {}
        all_hashes = []
        base_misses = []

        # Load embeddings from base cache; collect misses for API fetch
        for t, h in zip(texts, hashes):
            full_emb = self._select_base(h)
            if full_emb is not None:
                emb_by_hash[h] = full_emb
                all_hashes.append(h)
            else:
                base_misses.append((t, h))

        if base_misses:
            logger.info(
                f"_apply_pca_to_texts: fetching {len(base_misses)} embeddings "
                f"missing from base cache..."
            )
            for start in range(0, len(base_misses), self.batch_size):
                chunk = base_misses[start : start + self.batch_size]
                fetched = self._fetch_and_store_base(chunk)
                for (t, h), emb in zip(chunk, fetched):
                    emb_by_hash[h] = emb
                    all_hashes.append(h)

        if not emb_by_hash:
            logger.warning("no embeddings found for PCA transform")
            return

        # Transform embeddings
        emb_array = np.array([emb_by_hash[h] for h in all_hashes])
        reduced_embeddings = self._pca_model.transform(emb_array)

        hash_to_text = {h: t for t, h in zip(texts, hashes)}
        records = []
        for h, reduced in zip(all_hashes, reduced_embeddings):
            records.append((h, json.dumps(reduced.tolist()), hash_to_text.get(h, "")))

        self._insert_many_pca(records)

        logger.info(f"PCA transform applied to {len(records)} embeddings")

    def cache_stats(self) -> Dict[str, Any]:
        """Return basic cache statistics."""
        with contextlib.closing(self._connect_base()) as conn:
            (total,) = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()

        stats = {
            "cached_entries": total,
            "db_path": str(self.base_db_path),
            "pca_enabled": self.enable_pca,
        }

        if self.enable_pca and self.pca_db_path:
            with contextlib.closing(self._connect_pca()) as conn:
                (pca_total,) = conn.execute(
                    "SELECT COUNT(*) FROM embeddings"
                ).fetchone()
            stats["pca_entries"] = pca_total
            stats["pca_db_path"] = str(self.pca_db_path)

        return stats

    # ──────────────────────────────────────────────────────────────────────
    # Base database helpers (full-size embeddings)
    # ──────────────────────────────────────────────────────────────────────

    def _init_base_db(self) -> None:
        with contextlib.closing(self._connect_base()) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash  TEXT PRIMARY KEY,
                    model      TEXT NOT NULL,
                    dataset    TEXT NOT NULL,
                    embedding  TEXT NOT NULL,
                    text       TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model);")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dataset ON embeddings(dataset);"
            )

    def _connect_base(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.base_db_path),
            timeout=30,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _select_base(self, text_hash: str) -> Optional[List[float]]:
        with contextlib.closing(self._connect_base()) as conn:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE text_hash=? AND model=? AND dataset=?",
                (text_hash, self.model_name, self.dataset_name),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _select_many_base(self, text_hashes: List[str]) -> Dict[str, Optional[List[float]]]:
        """Batch-select embeddings by hash. Returns {hash: embedding or None}."""
        if not text_hashes:
            return {}
        placeholders = ",".join("?" * len(text_hashes))
        with contextlib.closing(self._connect_base()) as conn:
            rows = conn.execute(
                f"SELECT text_hash, embedding FROM embeddings WHERE text_hash IN ({placeholders}) AND model=? AND dataset=?",
                text_hashes + [self.model_name, self.dataset_name],
            ).fetchall()
        result = {h: None for h in text_hashes}
        for row in rows:
            result[row[0]] = json.loads(row[1])
        return result

    def _insert_base(self, text_hash: str, embedding: List[float], text: str) -> None:
        with self._w_lock, contextlib.closing(self._connect_base()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, model, dataset, embedding, text) VALUES (?, ?, ?, ?, ?)",
                (
                    text_hash,
                    self.model_name,
                    self.dataset_name,
                    json.dumps(embedding),
                    text,
                ),
            )

    def _insert_many_base(self, records: List[Tuple[str, str, str, str, str]]) -> None:
        """records: list of (text_hash, model, dataset, embedding_json, text)"""
        with self._w_lock, contextlib.closing(self._connect_base()) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings "
                "(text_hash, model, dataset, embedding, text) VALUES (?,?,?,?,?)",
                records,
            )

    def _get_all_embeddings_from_base(self) -> List[Tuple[str, List[float]]]:
        """Load all embeddings from base cache for PCA fitting."""
        with contextlib.closing(self._connect_base()) as conn:
            rows = conn.execute(
                "SELECT text_hash, embedding FROM embeddings WHERE model=? AND dataset=?",
                (self.model_name, self.dataset_name),
            ).fetchall()

        return [(row[0], json.loads(row[1])) for row in rows]

    # ──────────────────────────────────────────────────────────────────────
    # PCA database helpers (reduced embeddings)
    # ──────────────────────────────────────────────────────────────────────

    def _init_pca_db(self) -> None:
        """Initialize PCA cache database if not exists."""
        with contextlib.closing(self._connect_pca()) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash  TEXT PRIMARY KEY,
                    embedding  TEXT NOT NULL,
                    text       TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pca_hash ON embeddings(text_hash);"
            )

    def _connect_pca(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.pca_db_path),
            timeout=30,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _select_pca(self, text_hash: str) -> Optional[List[float]]:
        with contextlib.closing(self._connect_pca()) as conn:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE text_hash=?",
                (text_hash,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _select_many_pca(self, text_hashes: List[str]) -> Dict[str, Optional[List[float]]]:
        """Batch-select PCA embeddings by hash. Returns {hash: embedding or None}."""
        if not text_hashes:
            return {}
        placeholders = ",".join("?" * len(text_hashes))
        with contextlib.closing(self._connect_pca()) as conn:
            rows = conn.execute(
                f"SELECT text_hash, embedding FROM embeddings WHERE text_hash IN ({placeholders})",
                text_hashes,
            ).fetchall()
        result = {h: None for h in text_hashes}
        for row in rows:
            result[row[0]] = json.loads(row[1])
        return result

    def _insert_pca(self, text_hash: str, embedding: List[float], text: str) -> None:
        with self._pca_w_lock, contextlib.closing(self._connect_pca()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, embedding, text) VALUES (?, ?, ?)",
                (text_hash, json.dumps(embedding), text),
            )

    def _insert_many_pca(self, records: List[Tuple[str, str, str]]) -> None:
        """records: list of (text_hash, embedding_json, text)"""
        with self._pca_w_lock, contextlib.closing(self._connect_pca()) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (text_hash, embedding, text) VALUES (?, ?, ?)",
                records,
            )

    # ──────────────────────────────────────────────────────────────────────
    # PCA model management
    # ──────────────────────────────────────────────────────────────────────

    def _load_pca_model(self) -> None:
        """Load pre-trained PCA model from disk."""
        if not self.pca_model_path or not self.pca_model_path.exists():
            logger.warning(
                f"PCA model not found at {self.pca_model_path}, will fit on first use"
            )
            return

        try:
            import joblib

            self._pca_model = joblib.load(str(self.pca_model_path))
            self._pca_fitted = True
            logger.info(f"Loaded PCA model from {self.pca_model_path}")
        except Exception as e:
            logger.error(f"Failed to load PCA model: {e}")

    def _save_pca_model(self) -> None:
        """Save trained PCA model to disk."""
        if not self._pca_model or not self.pca_model_path:
            return

        try:
            import joblib

            self.pca_model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._pca_model, str(self.pca_model_path))
            logger.info(f"Saved PCA model to {self.pca_model_path}")
        except Exception as e:
            logger.error(f"Failed to save PCA model: {e}")

    def _transform_pca(self, embedding: List[float]) -> List[float]:
        """Apply PCA transform to a single embedding."""
        if not self._pca_fitted or self._pca_model is None:
            raise RuntimeError("PCA model not fitted. Call precompute() first.")

        import numpy as np

        emb_array = np.array([embedding])
        reduced = self._pca_model.transform(emb_array)[0].tolist()
        return reduced

    # ──────────────────────────────────────────────────────────────────────
    # Numpy fast-path (avoids JSON-in-SQLite overhead for large datasets)
    # ──────────────────────────────────────────────────────────────────────

    def _npy_path(self, is_pca: bool = False) -> Path:
        """Path to the .npy file storing all embeddings as a 2-D ndarray."""
        if is_pca:
            return self.pca_db_path.with_suffix(".npy")
        return self.base_db_path.with_suffix(".npy")

    def _idx_path(self, is_pca: bool = False) -> Path:
        """Path to the .json index mapping hash -> row index in the .npy file."""
        if is_pca:
            stem = self.pca_db_path.stem
            return self.pca_db_path.parent / f"{stem}_idx.json"
        stem = self.base_db_path.stem
        return self.base_db_path.parent / f"{stem}_idx.json"

    def _build_npy_index(self, hashes: List[str]) -> None:
        """Dump all cached embeddings to a single .npy file with hash->index map."""
        import numpy as np

        npy_path = self._npy_path(self.enable_pca)
        idx_path = self._idx_path(self.enable_pca)

        if npy_path.exists() and idx_path.exists():
            return

        if self.enable_pca:
            with contextlib.closing(self._connect_pca()) as conn:
                rows = conn.execute(
                    "SELECT text_hash, embedding FROM embeddings"
                ).fetchall()
        else:
            with contextlib.closing(self._connect_base()) as conn:
                rows = conn.execute(
                    "SELECT text_hash, embedding FROM embeddings WHERE model=? AND dataset=?",
                    (self.model_name, self.dataset_name),
                ).fetchall()

        if not rows:
            return

        embeddings = np.array([json.loads(r[1]) for r in rows], dtype=np.float32)
        hash_list = [r[0] for r in rows]
        hash_to_idx = {h: i for i, h in enumerate(hash_list)}

        np.save(npy_path, embeddings)
        with open(idx_path, "w") as f:
            json.dump(hash_to_idx, f)
        logger.info(f"Built numpy cache: {npy_path} ({embeddings.shape})")

    def _load_from_npy(
        self, hashes: List[str], is_pca: bool = False
    ) -> Optional[np.ndarray]:
        """Load embeddings directly from .npy file. Returns None if not available."""
        import numpy as np

        npy_path = self._npy_path(is_pca)
        idx_path = self._idx_path(is_pca)

        if not npy_path.exists() or not idx_path.exists():
            return None

        try:
            embeddings = np.load(npy_path, mmap_mode="r")
            with open(idx_path) as f:
                hash_to_idx = json.load(f)

            indices = [hash_to_idx[h] for h in hashes if h in hash_to_idx]
            if len(indices) != len(hashes):
                return None

            return embeddings[indices].astype(np.float32)
        except Exception:
            return None

    def _append_to_npy(
        self, texts: List[str], hashes: List[str], embeddings: np.ndarray, is_pca: bool
    ) -> None:
        """Append new embeddings to an existing .npy file and update index."""
        import numpy as np

        npy_path = self._npy_path(is_pca)
        idx_path = self._idx_path(is_pca)

        if npy_path.exists():
            existing = np.load(npy_path, mmap_mode="r")
            all_emb = np.vstack([existing, embeddings])
        else:
            all_emb = embeddings

        np.save(npy_path, all_emb)

        hash_to_idx = {}
        if idx_path.exists():
            with open(idx_path) as f:
                hash_to_idx = json.load(f)

        offset = len(all_emb) - len(embeddings)
        for i, h in enumerate(hashes):
            hash_to_idx[h] = offset + i

        with open(idx_path, "w") as f:
            json.dump(hash_to_idx, f)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within max_tokens."""
        tokens = self._tokenizer.encode(text)
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens]
        return self._tokenizer.decode(tokens)

    def _fetch_and_store_base(self, text_hash_pairs: List[Tuple[str, str]]) -> List[List[float]]:
        """Call the API for *text_hash_pairs* = [(text, hash), …]; store & return."""
        texts = [self._truncate_text(t) for t, _ in text_hash_pairs]
        delay = self.initial_delay

        for attempt in range(self.max_retries):
            try:
                rsp = self._client.embeddings.create(
                    input=texts, model=self.model_name, extra_body={"truncate": True}
                )
                embeddings = [record.embedding for record in rsp.data]
                db_rows = [
                    (h, self.model_name, self.dataset_name, json.dumps(emb), t)
                    for (t, h), emb in zip(text_hash_pairs, embeddings)
                ]
                self._insert_many_base(db_rows)
                return embeddings

            except RateLimitError:
                logger.warning(
                    f"Rate limited (attempt {attempt+1}/{self.max_retries}). Retry in {delay:.1f}s"
                )
            except APIError as e:
                logger.warning(
                    f"API error (attempt {attempt+1}/{self.max_retries}): {e}. Retry in {delay:.1f}s"
                )
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

            time.sleep(delay)
            delay *= 2

        raise RuntimeError(
            f"Failed to fetch embeddings after {self.max_retries} retries."
        )
