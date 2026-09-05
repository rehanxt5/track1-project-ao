"""Regenerate agent_spec.schema.json from the pydantic model.

Run after any change to models.py:
    python -m metaagent.spec.export_schema
"""

from __future__ import annotations

import json
from pathlib import Path

from metaagent.spec.models import AgentSpec

SCHEMA_PATH = Path(__file__).with_name("agent_spec.schema.json")


def export() -> Path:
    schema = AgentSpec.model_json_schema()
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2) + "\n")
    return SCHEMA_PATH


if __name__ == "__main__":
    path = export()
    print(f"wrote {path}")
