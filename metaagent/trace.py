"""Structured execution traces for AgentSpec runs.

Every LLM call, tool call, and critic pass the interpreter (metaagent.runner)
makes becomes one TraceStep, in the exact order it happened, so a downstream
failure analyzer can reconstruct *why* a run succeeded or failed without
re-executing anything. Traces are persisted as JSONL: one run (task_id +
its full step list + aggregate metrics) per line.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TraceStep:
    index: int
    role: str
    kind: str  # "llm" | "tool" | "critic"
    input: Any
    output: Any
    tool_name: str | None
    error: str | None
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    latency_ms: float


def run_result_to_dict(run_result: Any) -> dict:
    """Convert a RunResult (metaagent.runner.RunResult) into a JSON-safe dict.

    RunResult, TraceStep, and Score are all plain dataclasses, so
    dataclasses.asdict recurses through the nested `steps` list and `score`
    for us.
    """
    return asdict(run_result)


def append_trace_jsonl(path: str | Path, run_result: Any) -> None:
    """Append one run's full trace as a single JSON line to `path`.

    Creates parent directories as needed. Safe to call repeatedly against
    the same path to accumulate a batch's worth of traces.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(run_result_to_dict(run_result), default=str)
    with path.open("a") as f:
        f.write(line + "\n")


def read_trace_jsonl(path: str | Path) -> list[dict]:
    """Read back a JSONL trace file written by append_trace_jsonl."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
