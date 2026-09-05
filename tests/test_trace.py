from __future__ import annotations

from dataclasses import dataclass, field

from metaagent.trace import TraceStep, append_trace_jsonl, read_trace_jsonl


def _sample_step(index: int = 0) -> TraceStep:
    return TraceStep(
        index=index,
        role="solver",
        kind="llm",
        input=[{"role": "user", "content": "hi"}],
        output="hello",
        tool_name=None,
        error=None,
        input_tokens=10,
        output_tokens=5,
        reasoning_tokens=0,
        latency_ms=12.5,
    )


@dataclass
class _FakeRunResult:
    """Mimics metaagent.runner.RunResult's shape (a plain dataclass, same as
    the real thing) without importing runner, so this test stays focused
    purely on trace.py's (de)serialization via dataclasses.asdict."""

    task_id: str = "t1"
    output: str = "42"
    score: object = None
    steps: list = field(default_factory=lambda: [_sample_step(0), _sample_step(1)])
    error: object = None
    input_tokens: int = 20
    output_tokens: int = 10
    reasoning_tokens: int = 0
    total_latency_ms: float = 25.0
    step_count: int = 2
    tool_call_count: int = 0


def test_append_and_read_trace_jsonl_roundtrip(tmp_path):
    path = tmp_path / "traces" / "run.jsonl"
    run1 = _FakeRunResult()
    run2 = _FakeRunResult()
    run2.task_id = "t2"

    append_trace_jsonl(path, run1)
    append_trace_jsonl(path, run2)

    lines = read_trace_jsonl(path)
    assert len(lines) == 2
    assert lines[0]["task_id"] == "t1"
    assert lines[1]["task_id"] == "t2"
    assert len(lines[0]["steps"]) == 2
    assert lines[0]["steps"][0]["role"] == "solver"
    assert lines[0]["steps"][0]["kind"] == "llm"
    assert lines[0]["steps"][0]["input_tokens"] == 10


def test_append_trace_jsonl_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "run.jsonl"
    assert not path.parent.exists()
    append_trace_jsonl(path, _FakeRunResult())
    assert path.exists()


def test_read_trace_jsonl_missing_file_returns_empty(tmp_path):
    assert read_trace_jsonl(tmp_path / "nope.jsonl") == []


def test_trace_step_carries_error_and_tool_name():
    step = TraceStep(
        index=3,
        role="worker",
        kind="tool",
        input={"order_id": "ORD-1"},
        output=None,
        tool_name="get_order",
        error="RuntimeError: boom",
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        latency_ms=1.2,
    )
    assert step.kind == "tool"
    assert step.tool_name == "get_order"
    assert step.error == "RuntimeError: boom"
