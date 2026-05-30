"""
Response caching — stores previous results to avoid redundant computation.

Domain-agnostic. Uses SQLite for persistence across sessions.
Supports exact match and fuzzy matching via normalized queries.

Usage:
    cache = Cache()
    cache.put("Add types to Button", "interface ButtonProps { ... }")
    result = cache.get("Add types to Button")  # returns cached response
    result = cache.get("add types to button")  # also matches (normalized)
"""

import time
import hashlib
import sqlite3
import re
from pathlib import Path
from typing import Optional


class Cache:
    def __init__(self, db_path: Optional[str] = None, ttl_seconds: int = 86400 * 30):
        """
        Args:
            db_path: Path to SQLite database. Defaults to data/cache.db
            ttl_seconds: Time-to-live for cache entries. Default 30 days.
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "cache.db")

        self.db_path = db_path
        self.ttl = ttl_seconds
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS cache (
                query_hash TEXT PRIMARY KEY,
                query_text TEXT,
                query_normalized TEXT,
                response TEXT,
                model_id TEXT,
                created REAL,
                last_accessed REAL,
                hit_count INTEGER DEFAULT 0,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_cache_normalized
                ON cache(query_normalized);
            CREATE INDEX IF NOT EXISTS idx_cache_created
                ON cache(created);
        """)
        self.db.commit()

    def _normalize(self, query: str) -> str:
        """Normalize a query for fuzzy matching.
        Lowercases, strips extra whitespace, removes punctuation."""
        query = query.lower().strip()
        query = re.sub(r'\s+', ' ', query)
        query = re.sub(r'[^\w\s]', '', query)
        return query

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """Look up a cached response. Tries exact match, then normalized match."""
        now = time.time()

        # Exact match
        query_hash = self._hash(query)
        row = self.db.execute(
            "SELECT response, created FROM cache WHERE query_hash = ?",
            (query_hash,)
        ).fetchone()

        if row and (now - row["created"]) < self.ttl:
            self.db.execute(
                "UPDATE cache SET hit_count = hit_count + 1, last_accessed = ? WHERE query_hash = ?",
                (now, query_hash)
            )
            self.db.commit()
            return row["response"]

        # Normalized match
        normalized = self._normalize(query)
        row = self.db.execute(
            "SELECT query_hash, response, created FROM cache WHERE query_normalized = ?",
            (normalized,)
        ).fetchone()

        if row and (now - row["created"]) < self.ttl:
            self.db.execute(
                "UPDATE cache SET hit_count = hit_count + 1, last_accessed = ? WHERE query_hash = ?",
                (now, row["query_hash"])
            )
            self.db.commit()
            return row["response"]

        return None

    def put(self, query: str, response: str, model_id: str = "", metadata: str = ""):
        """Store a response in the cache."""
        query_hash = self._hash(query)
        normalized = self._normalize(query)
        now = time.time()

        self.db.execute("""
            INSERT OR REPLACE INTO cache
            (query_hash, query_text, query_normalized, response, model_id,
             created, last_accessed, hit_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
        """, (query_hash, query[:1000], normalized, response[:5000],
              model_id, now, now, metadata))
        self.db.commit()

    def invalidate(self, query: str):
        """Remove a specific entry from the cache."""
        query_hash = self._hash(query)
        self.db.execute("DELETE FROM cache WHERE query_hash = ?", (query_hash,))
        self.db.commit()

    def clear_expired(self):
        """Remove entries older than TTL."""
        cutoff = time.time() - self.ttl
        deleted = self.db.execute(
            "DELETE FROM cache WHERE created < ?", (cutoff,)
        ).rowcount
        self.db.commit()
        return deleted

    def clear_all(self):
        """Clear the entire cache."""
        self.db.execute("DELETE FROM cache")
        self.db.commit()

    def stats(self) -> dict:
        """Get cache statistics."""
        row = self.db.execute("""
            SELECT COUNT(*) as total,
                   SUM(hit_count) as total_hits,
                   AVG(hit_count) as avg_hits
            FROM cache
        """).fetchone()

        return {
            "total_entries": row["total"],
            "total_hits": row["total_hits"] or 0,
            "avg_hits_per_entry": round(row["avg_hits"] or 0, 1),
            "db_path": self.db_path,
        }
