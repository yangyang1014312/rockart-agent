"""SQLite memory for reviewed rock art detection cases."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_PATH = "rockart_agent_memory.sqlite3"


class RockArtMemory:
    def __init__(self, db_path: str | Path = DEFAULT_MEMORY_PATH) -> None:
        self.db_path = Path(db_path)
        self._memory_conn: sqlite3.Connection | None = None
        if str(db_path) == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:")
        elif self.db_path.parent != Path("."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    user_query TEXT,
                    class_counts_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    analysis TEXT,
                    human_feedback_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def load_recent(self, limit: int = 5) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT image_path, user_query, class_counts_json, summary, analysis,
                       human_feedback_json, created_at
                FROM cases
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        memories = []
        for row in rows:
            memories.append(
                {
                    "image_path": row[0],
                    "user_query": row[1],
                    "class_counts": json.loads(row[2]),
                    "summary": row[3],
                    "analysis": row[4],
                    "human_feedback": json.loads(row[5]) if row[5] else None,
                    "created_at": row[6],
                }
            )
        return memories

    def save_case(
        self,
        image_path: str,
        user_query: str,
        detection_result: dict[str, Any],
        analysis: str,
        human_feedback: str | dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cases (
                    image_path, user_query, class_counts_json, summary, analysis,
                    human_feedback_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    image_path,
                    user_query,
                    json.dumps(detection_result.get("class_counts", {}), ensure_ascii=False),
                    detection_result.get("summary", ""),
                    analysis,
                    json.dumps(human_feedback, ensure_ascii=False) if human_feedback else None,
                ),
            )
