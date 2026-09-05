"""
Objective, deterministic evaluator for multi_step_tools.

The agent runner is expected to hand this evaluator the agent's final text
output, which per goal.md must be a single JSON object of the form
{"tool_calls": [{"tool": ..., "args": {...}}, ...], "answer": ...}. That
convention -- rather than a separate transcript object -- is what lets this
evaluator check both the final answer AND the tool-call sequence using only
the `output: str` the shared Evaluator protocol provides.

score = 0.5 * answer_score + 0.5 * tool_score, where:
  - answer_score is 1.0/0.0 for scalar answers (numeric tolerance 1e-2,
    case-insensitive/trimmed string match), or the fraction of matching
    sub-fields for multi-part (dict) answers.
  - tool_score = 0.7 * (matched required calls / total required calls)
               + 0.3 * (satisfied ordering constraints / total constraints)
    A "required call" is matched if some entry in the agent's tool_calls
    list has the same tool name and its args are a superset match of the
    required args (extra args are fine; wrong or missing required args are
    not). No LLM judging anywhere -- pure structural/numeric comparison.
"""

from __future__ import annotations

import json
import re

from metaagent.evaluation import Score

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)

_NUMERIC_TOLERANCE = 1e-2


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


def _values_match(expected, actual) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            return abs(float(expected) - float(actual)) <= _NUMERIC_TOLERANCE
        return False
    if isinstance(expected, str):
        if not isinstance(actual, str):
            return False
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


def _call_matches(required: dict, actual_call) -> bool:
    if not isinstance(actual_call, dict):
        return False
    if actual_call.get("tool") != required["tool"]:
        return False
    actual_args = actual_call.get("args")
    if not isinstance(actual_args, dict):
        return False
    for key, expected_value in required.get("args", {}).items():
        if key not in actual_args or not _values_match(expected_value, actual_args[key]):
            return False
    return True


def _score_answer(expected, actual) -> tuple[float, dict]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0.0, {"expected": expected, "actual": actual, "error": "answer is not an object"}
        field_results = {}
        correct = 0
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            ok = key in actual and _values_match(expected_value, actual_value)
            field_results[key] = {"expected": expected_value, "actual": actual_value, "correct": ok}
            correct += ok
        return correct / len(expected), field_results
    ok = _values_match(expected, actual)
    return (1.0 if ok else 0.0), {"expected": expected, "actual": actual, "correct": ok}


class MultiStepToolsEvaluator:
    def evaluate(self, task: dict, output: str) -> Score:
        gold = task["gold"]
        parsed, parse_error = _extract_json(output)

        if parsed is None:
            return Score(
                score=0.0,
                passed=False,
                details={"json_parse_error": parse_error, "answer": None, "tool_calls_made": None},
            )

        tool_calls = parsed.get("tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = []

        answer_score, answer_details = _score_answer(gold["answer"], parsed.get("answer"))

        required_calls = gold.get("required_calls", [])
        matched_flags = []
        for required in required_calls:
            matched_flags.append(any(_call_matches(required, c) for c in tool_calls))
        matched_count = sum(matched_flags)
        missing_calls = [required_calls[i] for i, ok in enumerate(matched_flags) if not ok]
        call_fraction = (matched_count / len(required_calls)) if required_calls else 1.0

        first_index: dict[str, int] = {}
        for i, call in enumerate(tool_calls):
            if isinstance(call, dict):
                name = call.get("tool")
                if name is not None and name not in first_index:
                    first_index[name] = i

        order_constraints = gold.get("order_constraints", [])
        order_violations = []
        for constraint in order_constraints:
            before, after = constraint["before"], constraint["after"]
            if before not in first_index or after not in first_index:
                order_violations.append(constraint)
            elif first_index[before] >= first_index[after]:
                order_violations.append(constraint)
        order_fraction = (
            (len(order_constraints) - len(order_violations)) / len(order_constraints)
            if order_constraints
            else 1.0
        )

        tool_score = 0.7 * call_fraction + 0.3 * order_fraction
        score = 0.5 * answer_score + 0.5 * tool_score
        passed = answer_score == 1.0 and not missing_calls and not order_violations

        details = {
            "json_parse_error": None,
            "answer": answer_details,
            "answer_score": answer_score,
            "tool_calls_made": tool_calls,
            "required_calls": required_calls,
            "matched_required_calls": matched_count,
            "total_required_calls": len(required_calls),
            "missing_calls": missing_calls,
            "order_constraints": order_constraints,
            "order_violations": order_violations,
            "tool_score": tool_score,
        }
        return Score(score=score, passed=passed, details=details)


EVALUATOR = MultiStepToolsEvaluator()
