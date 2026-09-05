"""
Deterministic, offline tool implementations for the sql_generation domain.

All three tools operate against the fixture SQLite database shipped at
domains/sql_generation/db/shop.db (built by db/build_db.py). They are
read-only: run_query rejects anything that isn't a single SELECT statement,
so exploration can never mutate the fixture.

Convention (relied on by the agent runner, not by metaagent/evaluation.py):
TOOLS maps each tool name declared in tools.json to a callable that accepts
the tool's declared parameters as keyword arguments and returns a
JSON-serializable value.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "db" / "shop.db"
MAX_ROWS = 20


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def list_tables() -> dict:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return {"tables": [r["name"] for r in rows]}
    finally:
        conn.close()


def describe_table(table_name: str) -> dict:
    conn = _connect()
    try:
        rows = conn.execute(f"PRAGMA table_info({table_name!r})").fetchall()
        if not rows:
            return {"error": f"no such table: {table_name}"}
        columns = [{"name": r["name"], "type": r["type"]} for r in rows]
        return {"table": table_name, "columns": columns}
    finally:
        conn.close()


def run_query(sql: str) -> dict:
    stripped = sql.strip().rstrip(";").strip()
    if not stripped.lower().startswith("select"):
        return {"error": "only single SELECT statements are allowed"}
    if ";" in stripped:
        return {"error": "only a single statement is allowed"}

    conn = _connect()
    try:
        cursor = conn.execute(stripped)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(MAX_ROWS)
        return {
            "columns": columns,
            "rows": [list(row) for row in rows],
            "truncated": len(rows) == MAX_ROWS,
        }
    except sqlite3.Error as e:
        return {"error": str(e)}
    finally:
        conn.close()


TOOLS = {
    "list_tables": list_tables,
    "describe_table": describe_table,
    "run_query": run_query,
}
