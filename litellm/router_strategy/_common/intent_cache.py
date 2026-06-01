"""
Intent Cache
============
Persistent SQLite-backed cache for intent classification results.

Key design:
- One DB file per (dataset, model, centroids_identifier) → ensures correct
  invalidation when embedding model or intent classes change.
- Caches (query_hash → intent_label) mappings to avoid re-running
  cosine-similarity classification when the same queries are processed
  with the same intent configuration.
- Thread-safe WAL mode.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

__all__ = ["IntentCache"]


def _compute_cache_key(centroids_path: str | os.PathLike, intent_labels: List[str]) -> str:
    """Compute a hash that uniquely identifies the intent classification configuration."""
    path_str = str(centroids_path)
    labels_str = "|".join(sorted(intent_labels))
    combined = f"{path_str}||{labels_str}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


class IntentCache:
    """
    Persistent SQLite-backed cache for intent classification results.

    Parameters
    ----------
    centroids_path : str | Path
        Path to the intent centroids JSON file. Used as part of the cache key.
    intent_labels : list[str]
        List of intent class labels. Used as part of the cache key.
    cache_dir : str | Path, optional
        Directory to store the cache DB. Defaults to same directory as centroids_path.
    """

    def __init__(
        self,
        centroids_path: str | os.PathLike,
        intent_labels: List[str],
        cache_dir: Optional[str | os.PathLike] = None,
    ) -> None:
        self._centroids_path = Path(centroids_path)
        self._intent_labels = sorted(intent_labels)
        self._cache_key = _compute_cache_key(self._centroids_path, self._intent_labels)

        if cache_dir is None:
            cache_dir = self._centroids_path.parent
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        db_name = f"intent_cache_{self._cache_key}.db"
        self._db_path = self._cache_dir / db_name

        self._lock = threading.Lock()
        self._init_db()

    @property
    def cache_key(self) -> str:
        """The cache key identifier for this configuration."""
        return self._cache_key

    def get(self, query: str) -> Optional[str]:
        """
        Retrieve cached intent for a single query.

        Parameters
        ----------
        query : str
            The query text.

        Returns
        -------
        str or None
            Cached intent label, or None if not found.
        """
        h = hashlib.sha256(query.encode()).hexdigest()
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT intent_label FROM intent_cache WHERE query_hash=? AND cache_key=?",
                (h, self._cache_key),
            ).fetchone()
        return row[0] if row else None

    def get_many(self, queries: List[str]) -> Dict[int, Optional[str]]:
        """
        Retrieve cached intents for multiple queries.

        Parameters
        ----------
        queries : list[str]
            Query texts.

        Returns
        -------
        dict[int, Optional[str]]
            Mapping from query index to cached intent (or None if not cached).
        """
        if not queries:
            return {}
        hashes = [hashlib.sha256(q.encode()).hexdigest() for q in queries]
        placeholders = ",".join("?" * len(hashes))
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT query_hash, intent_label FROM intent_cache WHERE query_hash IN ({placeholders}) AND cache_key=?",
                hashes + [self._cache_key],
            ).fetchall()
        result: Dict[int, Optional[str]] = {i: None for i in range(len(queries))}
        hash_to_idx = {h: i for i, h in enumerate(hashes)}
        for row in rows:
            idx = hash_to_idx.get(row[0])
            if idx is not None:
                result[idx] = row[1]
        return result

    def put(self, query: str, intent_label: str) -> None:
        """
        Cache the intent for a single query.

        Parameters
        ----------
        query : str
            The query text.
        intent_label : str
            The intent classification result.
        """
        h = hashlib.sha256(query.encode()).hexdigest()
        with self._lock, contextlib.closing(self._connect()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO intent_cache (query_hash, cache_key, query_text, intent_label) VALUES (?, ?, ?, ?)",
                (h, self._cache_key, query, intent_label),
            )

    def put_many(self, queries: List[str], intent_labels: List[str]) -> None:
        """
        Cache intents for multiple queries at once.

        Parameters
        ----------
        queries : list[str]
            Query texts.
        intent_labels : list[str]
            Corresponding intent classification results.
        """
        if not queries:
            return
        records = []
        for q, label in zip(queries, intent_labels):
            h = hashlib.sha256(q.encode()).hexdigest()
            records.append((h, self._cache_key, q, label))
        with self._lock, contextlib.closing(self._connect()) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO intent_cache (query_hash, cache_key, query_text, intent_label) VALUES (?, ?, ?, ?)",
                records,
            )

    def clear(self) -> None:
        """Remove all entries for this configuration from the cache."""
        with self._lock, contextlib.closing(self._connect()) as conn:
            conn.execute("DELETE FROM intent_cache WHERE cache_key=?", (self._cache_key,))
        logger.info(f"[IntentCache] Cleared cache: {self._db_path}")

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with contextlib.closing(self._connect()) as conn:
            (total,) = conn.execute(
                "SELECT COUNT(*) FROM intent_cache WHERE cache_key=?", (self._cache_key,)
            ).fetchone()
        return {
            "cached_entries": total,
            "cache_key": self._cache_key,
            "db_path": str(self._db_path),
        }

    def _init_db(self) -> None:
        with contextlib.closing(self._connect()) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS intent_cache (
                    query_hash   TEXT NOT NULL,
                    cache_key    TEXT NOT NULL,
                    query_text   TEXT,
                    intent_label TEXT NOT NULL,
                    PRIMARY KEY (query_hash, cache_key)
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_key ON intent_cache(cache_key);"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn
