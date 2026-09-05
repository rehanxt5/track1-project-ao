"""
Objective, deterministic evaluator for sql_generation.

The agent's SQL is EXECUTED against the real fixture database (db/shop.db)
and the result set is compared to a gold result set that was itself
produced by executing task["gold"]["sql"] against the same fixture (see
tasks/dev|test/*.json -- each task stores the literal expected rows, not
just the gold SQL, so evaluation never needs to re-run gold SQL at score
time and can't drift if the fixture changes). No string matching against
the query text and no LLM judging -- only what the query actually returns.

Rows are compared as a multiset of tuples (order-insensitive) unless the
task sets "order_matters": true, in which case exact ordered equality is
required to pass. Floats are rounded to 2 decimals on both sides before
comparison to avoid spurious float-precision mismatches.
score = row-level F1 between the actual and expected multisets of rows
(1.0 only on an exact multiset match); `passed` additionally requires
correct ordering when the task demands it.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

from metaagent.evaluation import Score

DB_PATH = Path(__file__).parent / "db" / "shop.db"

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_sql(output: str) -> tuple[str | None, str | None]:
    text = output.strip()
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"could not parse output as JSON: {e}"
    if not isinstance(parsed, dict) or "sql" not in parsed:
        return None, "parsed JSON object has no 'sql' key"
    sql = parsed["sql"]
    if not isinstance(sql, str) or not sql.strip():
        return None, "'sql' value is not a non-empty string"
    return sql, None


def _normalize_row(row) -> tuple:
    normalized = []
    for value in row:
        if isinstance(value, float):
            normalized.append(round(value, 2))
        else:
            normalized.append(value)
    return tuple(normalized)


def _execute(sql: str) -> tuple[list[tuple] | None, str | None]:
    stripped = sql.strip().rstrip(";").strip()
    if not stripped.lower().startswith("select"):
        return None, "only a single SELECT statement is allowed"
    if ";" in stripped:
        return None, "only a single statement is allowed"

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cursor = conn.execute(stripped)
        rows = [_normalize_row(row) for row in cursor.fetchall()]
        return rows, None
    except sqlite3.Error as e:
        return None, str(e)
    finally:
        conn.close()


def _row_f1(expected: list[tuple], actual: list[tuple]) -> float:
    if not expected and not actual:
        return 1.0
    expected_counts, actual_counts = Counter(expected), Counter(actual)
    overlap = sum((expected_counts & actual_counts).values())
    precision = overlap / len(actual) if actual else 0.0
    recall = overlap / len(expected) if expected else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class SqlGenerationEvaluator:
    def evaluate(self, task: dict, output: str) -> Score:
        gold = task["gold"]
        expected_rows = [_normalize_row(row) for row in gold["expected_rows"]]
        order_matters = gold.get("order_matters", False)

        sql, parse_error = _extract_sql(output)
        if sql is None:
            details = {
                "parse_error": parse_error,
                "sql_actual": None,
                "sql_expected": gold["sql"],
                "execution_error": None,
                "expected_row_count": len(expected_rows),
                "actual_row_count": 0,
            }
            return Score(score=0.0, passed=False, details=details)

        actual_rows, exec_error = _execute(sql)
        if exec_error is not None:
            details = {
                "parse_error": None,
                "sql_actual": sql,
                "sql_expected": gold["sql"],
                "execution_error": exec_error,
                "expected_row_count": len(expected_rows),
                "actual_row_count": 0,
            }
            return Score(score=0.0, passed=False, details=details)

        row_score = _row_f1(expected_rows, actual_rows)
        expected_counts, actual_counts = Counter(expected_rows), Counter(actual_rows)
        missing_rows = list((expected_counts - actual_counts).elements())
        extra_rows = list((actual_counts - expected_counts).elements())

        order_correct = (actual_rows == expected_rows) if order_matters else True
        exact_multiset_match = not missing_rows and not extra_rows
        passed = exact_multiset_match and order_correct
        # right rows, wrong order on a task that requires ORDER BY: still informative
        # partial credit, but must not read as a perfect score.
        score = row_score if order_correct else min(row_score, 0.75)

        details = {
            "parse_error": None,
            "sql_actual": sql,
            "sql_expected": gold["sql"],
            "execution_error": None,
            "columns_expected": gold.get("columns"),
            "expected_row_count": len(expected_rows),
            "actual_row_count": len(actual_rows),
            "missing_rows": missing_rows,
            "extra_rows": extra_rows,
            "order_matters": order_matters,
            "order_correct": order_correct,
        }
        return Score(score=score, passed=passed, details=details)


EVALUATOR = SqlGenerationEvaluator()
