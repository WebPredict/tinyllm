"""
Structured long-term memory module.

Tracks durable state over time — user patterns, past errors, session history,
and any domain-specific facts. Unlike RAG (which retrieves document chunks),
structured memory stores typed records in SQLite.

Works for any domain — the schema is configurable.

Usage:
    memory = StructuredMemory()
    memory.record_interaction("generate_component", input="...", output="...", success=True)
    memory.record_error("type_error", details="...")
    recent = memory.get_recent_errors(limit=5)
    context = memory.get_context_for_prompt()  # compact summary for injection
"""

import json
import time
import sqlite3
from pathlib import Path
from typing import Optional


class StructuredMemory:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "memory.db")

        self.db_path = db_path
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                task_type TEXT,
                input_text TEXT,
                output_text TEXT,
                success INTEGER,
                validator_passed INTEGER,
                was_repaired INTEGER,
                latency_ms REAL,
                modules_used TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                error_type TEXT,
                details TEXT,
                task_type TEXT,
                resolved INTEGER DEFAULT 0,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT UNIQUE,
                frequency INTEGER DEFAULT 1,
                last_seen REAL,
                success_rate REAL DEFAULT 0,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started REAL,
                ended REAL,
                interaction_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS key_value (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated REAL
            );

            CREATE INDEX IF NOT EXISTS idx_interactions_task ON interactions(task_type);
            CREATE INDEX IF NOT EXISTS idx_interactions_time ON interactions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_errors_type ON errors(error_type);
            CREATE INDEX IF NOT EXISTS idx_patterns_name ON patterns(pattern_name);
        """)
        self.db.commit()

    # ─── Recording ────────────────────────────

    def record_interaction(self, task_type: str, input_text: str = "",
                          output_text: str = "", success: bool = True,
                          validator_passed: bool = True, was_repaired: bool = False,
                          latency_ms: float = 0, modules_used: list = None,
                          metadata: dict = None):
        """Record a complete interaction."""
        self.db.execute("""
            INSERT INTO interactions
            (timestamp, task_type, input_text, output_text, success,
             validator_passed, was_repaired, latency_ms, modules_used, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), task_type, input_text[:1000], output_text[:2000],
            int(success), int(validator_passed), int(was_repaired),
            latency_ms, json.dumps(modules_used or []),
            json.dumps(metadata or {}),
        ))
        self.db.commit()

        # Update pattern tracking
        self._update_pattern(task_type, success)

    def record_error(self, error_type: str, details: str = "",
                    task_type: str = "", metadata: dict = None):
        """Record an error for tracking recurring issues."""
        self.db.execute("""
            INSERT INTO errors (timestamp, error_type, details, task_type, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (time.time(), error_type, details[:1000], task_type,
              json.dumps(metadata or {})))
        self.db.commit()

    def _update_pattern(self, pattern_name: str, success: bool):
        """Update pattern frequency and success rate."""
        existing = self.db.execute(
            "SELECT frequency, success_rate FROM patterns WHERE pattern_name = ?",
            (pattern_name,)
        ).fetchone()

        if existing:
            freq = existing["frequency"] + 1
            rate = (existing["success_rate"] * existing["frequency"] + int(success)) / freq
            self.db.execute("""
                UPDATE patterns SET frequency = ?, success_rate = ?, last_seen = ?
                WHERE pattern_name = ?
            """, (freq, rate, time.time(), pattern_name))
        else:
            self.db.execute("""
                INSERT INTO patterns (pattern_name, frequency, last_seen, success_rate)
                VALUES (?, 1, ?, ?)
            """, (pattern_name, time.time(), float(success)))
        self.db.commit()

    def set_value(self, key: str, value: str):
        """Store a key-value pair (preferences, settings, etc.)."""
        self.db.execute("""
            INSERT OR REPLACE INTO key_value (key, value, updated)
            VALUES (?, ?, ?)
        """, (key, value, time.time()))
        self.db.commit()

    def get_value(self, key: str, default: str = None) -> Optional[str]:
        """Retrieve a stored value."""
        row = self.db.execute(
            "SELECT value FROM key_value WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    # ─── Querying ─────────────────────────────

    def get_recent_interactions(self, limit: int = 10, task_type: str = None) -> list:
        """Get recent interactions, optionally filtered by task type."""
        if task_type:
            rows = self.db.execute("""
                SELECT * FROM interactions WHERE task_type = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (task_type, limit)).fetchall()
        else:
            rows = self.db.execute("""
                SELECT * FROM interactions ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_recent_errors(self, limit: int = 10, error_type: str = None) -> list:
        """Get recent errors."""
        if error_type:
            rows = self.db.execute("""
                SELECT * FROM errors WHERE error_type = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (error_type, limit)).fetchall()
        else:
            rows = self.db.execute("""
                SELECT * FROM errors ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_common_errors(self, limit: int = 5) -> list:
        """Get most frequent error types."""
        rows = self.db.execute("""
            SELECT error_type, COUNT(*) as count,
                   MAX(timestamp) as last_seen
            FROM errors
            GROUP BY error_type
            ORDER BY count DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_patterns(self, limit: int = 10) -> list:
        """Get tracked patterns sorted by frequency."""
        rows = self.db.execute("""
            SELECT * FROM patterns ORDER BY frequency DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_success_rate(self, task_type: str = None) -> float:
        """Get overall or per-task success rate."""
        if task_type:
            row = self.db.execute("""
                SELECT AVG(success) as rate FROM interactions WHERE task_type = ?
            """, (task_type,)).fetchone()
        else:
            row = self.db.execute(
                "SELECT AVG(success) as rate FROM interactions"
            ).fetchone()
        return row["rate"] if row and row["rate"] is not None else 0.0

    # ─── Context for prompts ──────────────────

    def get_context_for_prompt(self, max_chars: int = 500) -> str:
        """Build a compact context string to inject into model prompts.
        Summarizes recent history, common errors, and patterns."""
        parts = []

        # Common errors
        errors = self.get_common_errors(3)
        if errors:
            error_strs = [f"{e['error_type']} ({e['count']}x)" for e in errors]
            parts.append(f"Common errors: {', '.join(error_strs)}")

        # Success rates by task
        patterns = self.get_patterns(5)
        if patterns:
            pattern_strs = [
                f"{p['pattern_name']}: {p['success_rate']*100:.0f}% ({p['frequency']}x)"
                for p in patterns
            ]
            parts.append(f"Task history: {', '.join(pattern_strs)}")

        # Recent failures
        recent_errors = self.get_recent_errors(3)
        if recent_errors:
            parts.append("Recent errors: " + "; ".join(
                e["details"][:80] for e in recent_errors
            ))

        context = "\n".join(parts)
        return context[:max_chars] if context else ""

    # ─── Stats ────────────────────────────────

    def stats(self) -> dict:
        """Get memory statistics."""
        return {
            "total_interactions": self.db.execute("SELECT COUNT(*) FROM interactions").fetchone()[0],
            "total_errors": self.db.execute("SELECT COUNT(*) FROM errors").fetchone()[0],
            "total_patterns": self.db.execute("SELECT COUNT(*) FROM patterns").fetchone()[0],
            "overall_success_rate": self.get_success_rate(),
            "db_path": self.db_path,
        }
