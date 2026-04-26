"""Database query tool for the Agentic RL environment."""

from __future__ import annotations

import logging
import sqlite3
import re
from typing import Any

from .base import BaseTool

logger = logging.getLogger(__name__)


class DatabaseQueryTool(BaseTool):
    """Execute SQL queries against a SQLite database.

    For training, we use an in-memory or file-based SQLite database
    pre-populated with task-specific data.  Production deployments can swap
    in a real database connector.
    """

    name = "database_query"
    description = (
        "Execute a read-only SQL query against the task database "
        "and return the results as a list of rows."
    )
    parameters = {
        "sql": {
            "type": "string",
            "description": "The SQL query to execute (SELECT only)",
        },
    }

    def __init__(self, db_path: str = ":memory:", max_rows: int = 100):
        self.db_path = db_path
        self.max_rows = max_rows
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def execute(self, sql: str, **kwargs: Any) -> list[dict]:
        # Safety: only allow SELECT and WITH ... SELECT statements
        sql_stripped = sql.strip().upper()
        if not (sql_stripped.startswith("SELECT") or sql_stripped.startswith("WITH")):
            raise ValueError("Only SELECT queries are allowed")

        # Block dangerous keywords
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "EXEC"]
        for kw in dangerous:
            if re.search(rf"\b{kw}\b", sql_stripped):
                raise ValueError(f"Forbidden SQL keyword: {kw}")

        conn = self._get_connection()
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchmany(self.max_rows)
        return [dict(zip(columns, row)) for row in rows]

    def load_schema(self, schema_sql: str) -> None:
        """Load a schema into the database (for training setup)."""
        conn = self._get_connection()
        conn.executescript(schema_sql)
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
