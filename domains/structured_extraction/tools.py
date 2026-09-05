"""
Deterministic, offline tool implementations for the structured_extraction
domain.

Convention (relied on by the agent runner, not by metaagent/evaluation.py):
TOOLS maps each tool name declared in tools.json to a callable that accepts
the tool's declared parameters as keyword arguments and returns a
JSON-serializable value.
"""

from __future__ import annotations

import json

SCHEMA = {
    "customer_name": {"type": "string"},
    "order_id": {"type": ["string", "null"], "pattern": "ORD-\\d+"},
    "product": {"type": ["string", "null"]},
    "issue_type": {
        "type": "string",
        "enum": ["defective", "wrong_item", "missing_item", "late_delivery", "billing", "other"],
    },
    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
    "requested_action": {
        "type": "string",
        "enum": ["refund", "replacement", "repair", "information", "cancellation"],
    },
    "contact_method": {"type": ["string", "null"], "enum": ["email", "phone", "chat", None]},
}

REQUIRED_FIELDS = ["customer_name", "issue_type", "priority", "requested_action"]


def get_schema() -> dict:
    return {"fields": SCHEMA, "required": REQUIRED_FIELDS}


def validate_extraction(candidate_json: str) -> dict:
    try:
        candidate = json.loads(candidate_json)
    except json.JSONDecodeError as e:
        return {"valid": False, "errors": {"_json": f"invalid JSON: {e}"}}

    if not isinstance(candidate, dict):
        return {"valid": False, "errors": {"_json": "top-level value must be a JSON object"}}

    errors: dict[str, str] = {}

    for field_name in REQUIRED_FIELDS:
        if field_name not in candidate or candidate[field_name] in (None, ""):
            errors[field_name] = "missing"

    for field_name, spec in SCHEMA.items():
        if field_name not in candidate or field_name in errors:
            continue
        value = candidate[field_name]
        enum = spec.get("enum")
        if enum is not None and value not in enum:
            errors[field_name] = f"must be one of {enum}"

    return {"valid": len(errors) == 0, "errors": errors}


TOOLS = {
    "get_schema": get_schema,
    "validate_extraction": validate_extraction,
}
