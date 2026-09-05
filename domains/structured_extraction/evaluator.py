"""
Objective, deterministic evaluator for structured_extraction.

Scoring is pure field-level comparison against gold labels -- no LLM
judging. score = fraction of the seven schema fields that match gold
exactly (free-text fields are compared case-insensitively and trimmed;
enum/id fields must match exactly). `passed` requires every REQUIRED
field correct, with at most one optional-field miss.
"""

from __future__ import annotations

import json
import re

from metaagent.evaluation import Score

FIELDS = [
    "customer_name",
    "order_id",
    "product",
    "issue_type",
    "priority",
    "requested_action",
    "contact_method",
]
REQUIRED_FIELDS = ["customer_name", "issue_type", "priority", "requested_action"]
FREE_TEXT_FIELDS = {"customer_name", "product", "order_id"}

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(output: str) -> tuple[dict | None, str | None]:
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
        return None, str(e)
    if not isinstance(parsed, dict):
        return None, "top-level JSON value is not an object"
    return parsed, None


def _normalize(value):
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _field_matches(field_name: str, expected, actual) -> bool:
    if expected is None:
        return actual is None
    if field_name in FREE_TEXT_FIELDS:
        return _normalize(expected) == _normalize(actual)
    return expected == actual


class StructuredExtractionEvaluator:
    def evaluate(self, task: dict, output: str) -> Score:
        gold = task["gold"]
        candidate, parse_error = _extract_json(output)

        if candidate is None:
            details = {
                "parse_error": parse_error,
                "fields": {
                    f: {"expected": gold.get(f), "actual": None, "correct": False} for f in FIELDS
                },
                "missing_fields": list(FIELDS),
                "correct_count": 0,
                "total_fields": len(FIELDS),
                "required_fields_correct": 0,
                "required_fields_total": len(REQUIRED_FIELDS),
            }
            return Score(score=0.0, passed=False, details=details)

        field_results = {}
        missing_fields = []
        correct_count = 0
        required_correct = 0

        for field_name in FIELDS:
            expected = gold.get(field_name)
            present = field_name in candidate
            actual = candidate.get(field_name)
            correct = present and _field_matches(field_name, expected, actual)

            field_results[field_name] = {"expected": expected, "actual": actual, "correct": correct}
            if not present:
                missing_fields.append(field_name)
            if correct:
                correct_count += 1
                if field_name in REQUIRED_FIELDS:
                    required_correct += 1

        score = correct_count / len(FIELDS)
        passed = required_correct == len(REQUIRED_FIELDS) and correct_count >= len(FIELDS) - 1

        details = {
            "parse_error": None,
            "fields": field_results,
            "missing_fields": missing_fields,
            "correct_count": correct_count,
            "total_fields": len(FIELDS),
            "required_fields_correct": required_correct,
            "required_fields_total": len(REQUIRED_FIELDS),
        }
        return Score(score=score, passed=passed, details=details)


EVALUATOR = StructuredExtractionEvaluator()
