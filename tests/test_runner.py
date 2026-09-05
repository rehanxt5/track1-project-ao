from __future__ import annotations

import json
from collections import defaultdict

import pytest

from metaagent.runner import BatchResult, RunResult, run_batch, run_spec
from metaagent.spec.models import AgentSpec
from tests.fakes import FakeGateway, valid_spec_dict


def _spec(**overrides) -> AgentSpec:
    d = valid_spec_dict(**{k: v for k, v in overrides.items() if k in {"name", "design_philosophy", "tools", "goal"}})
    if "max_steps" in overrides:
        d["orchestration"]["max_steps"] = overrides["max_steps"]
    if "budget_tokens" in overrides:
        d["orchestration"]["budget_tokens"] = overrides["budget_tokens"]
    if "max_retries" in overrides:
        d["roles"][0]["retry"]["max_retries"] = overrides["max_retries"]
    if "on_failure" in overrides:
        d["roles"][0]["retry"]["on_failure"] = overrides["on_failure"]
    if "backoff" in overrides:
        d["roles"][0]["retry"]["backoff"] = overrides["backoff"]
    return AgentSpec.model_validate(d)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_spec_happy_path_no_tools():
    spec = _spec(tools=[])
    gateway = FakeGateway([json.dumps({"action": "final_answer", "final_output": "42"})])
    task = {"id": "t1", "question": "what is 6*7?"}

    result = await run_spec(spec, task, tools=[], complete_fn=gateway.complete)

    assert isinstance(result, RunResult)
    assert result.error is None
    assert result.score is None  # run_spec never grades -- that's run_batch's job
    assert result.output == "42"
    assert result.task_id == "t1"
    assert result.step_count == 1
    assert result.tool_call_count == 0
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.kind == "llm"
    assert step.role == "solver"
    assert step.error is None
    assert result.input_tokens == 10
    assert result.output_tokens == 10
    assert result.reasoning_tokens == 0
    assert result.total_latency_ms == pytest.approx(step.latency_ms)


@pytest.mark.asyncio
async def test_run_spec_executes_real_tool_calls():
    spec = _spec(tools=["lookup"])
    responses = [
        json.dumps({"action": "tool_call", "tool": "lookup", "tool_args": {"x": 21}}),
        json.dumps({"action": "final_answer", "final_output": "42"}),
    ]
    gateway = FakeGateway(responses)
    tools = [
        {
            "name": "lookup",
            "description": "doubles a number",
            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "fn": lambda x: {"value": x * 2},
        }
    ]
    task = {"id": "t1", "question": "double 21"}

    result = await run_spec(spec, task, tools=tools, complete_fn=gateway.complete)

    assert result.error is None
    assert result.output == "42"
    assert result.tool_call_count == 1
    assert result.step_count == 3
    kinds = [s.kind for s in result.steps]
    assert kinds == ["llm", "tool", "llm"]
    tool_step = result.steps[1]
    assert tool_step.tool_name == "lookup"
    assert tool_step.input == {"x": 21}
    assert tool_step.output == {"value": 42}
    assert tool_step.error is None


@pytest.mark.asyncio
async def test_run_spec_rejects_tool_not_named_by_role():
    # role.tools=["lookup"] but the model asks for a different tool name --
    # the interpreter must not execute it.
    spec = _spec(tools=["lookup"])
    responses = [
        json.dumps({"action": "tool_call", "tool": "delete_everything", "tool_args": {}}),
        json.dumps({"action": "final_answer", "final_output": "gave up"}),
    ]
    gateway = FakeGateway(responses)
    called = {"n": 0}

    def _dangerous(**kwargs):
        called["n"] += 1
        return {"ok": True}

    tools = [
        {"name": "lookup", "description": "", "parameters": {}, "fn": lambda: {}},
        {"name": "delete_everything", "description": "", "parameters": {}, "fn": _dangerous},
    ]
    task = {"id": "t1", "question": "q"}

    result = await run_spec(spec, task, tools=tools, complete_fn=gateway.complete)

    assert called["n"] == 0
    tool_step = next(s for s in result.steps if s.kind == "tool")
    assert tool_step.error is not None
    assert "not available" in tool_step.error


# ---------------------------------------------------------------------------
# tool error + retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_error_is_retried_and_recovers():
    spec = _spec(tools=["flaky"], max_retries=1, backoff="none")
    responses = [
        json.dumps({"action": "tool_call", "tool": "flaky", "tool_args": {}}),
        json.dumps({"action": "final_answer", "final_output": "done"}),
    ]
    gateway = FakeGateway(responses)

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient network blip")
        return {"ok": True}

    tools = [{"name": "flaky", "description": "", "parameters": {}, "fn": flaky}]
    task = {"id": "t1", "question": "q"}

    result = await run_spec(spec, task, tools=tools, complete_fn=gateway.complete)

    assert result.error is None
    assert result.output == "done"
    assert calls["n"] == 2
    tool_steps = [s for s in result.steps if s.kind == "tool"]
    assert len(tool_steps) == 2, "both the failed attempt and the successful retry must appear in the trace"
    assert tool_steps[0].error == "RuntimeError: transient network blip"
    assert tool_steps[0].output is None
    assert tool_steps[1].error is None
    assert tool_steps[1].output == {"ok": True}
    assert result.tool_call_count == 2


@pytest.mark.asyncio
async def test_tool_error_exhausts_retries_and_aborts():
    spec = _spec(tools=["always_broken"], max_retries=1, backoff="none", on_failure="abort")
    responses = [json.dumps({"action": "tool_call", "tool": "always_broken", "tool_args": {}})]
    gateway = FakeGateway(responses)

    def always_broken():
        raise RuntimeError("still broken")

    tools = [{"name": "always_broken", "description": "", "parameters": {}, "fn": always_broken}]
    task = {"id": "t1", "question": "q"}

    result = await run_spec(spec, task, tools=tools, complete_fn=gateway.complete)

    assert result.error is not None
    assert "always_broken" in result.error
    assert result.score is not None
    assert result.score.score == 0.0
    assert result.score.passed is False
    # exactly max_retries + 1 attempts
    tool_steps = [s for s in result.steps if s.kind == "tool"]
    assert len(tool_steps) == 2
    assert all(s.error is not None for s in tool_steps)


@pytest.mark.asyncio
async def test_data_shaped_tool_error_is_not_retried():
    # domains report failed lookups as {"error": ...} *data*, not an
    # exception -- the interpreter must hand that straight back to the model
    # rather than burning retries on it.
    spec = _spec(tools=["lookup"], max_retries=3)
    responses = [
        json.dumps({"action": "tool_call", "tool": "lookup", "tool_args": {"id": "bad"}}),
        json.dumps({"action": "final_answer", "final_output": "no such id"}),
    ]
    gateway = FakeGateway(responses)
    calls = {"n": 0}

    def lookup(id):
        calls["n"] += 1
        return {"error": f"no such id: {id}"}

    tools = [{"name": "lookup", "description": "", "parameters": {}, "fn": lookup}]
    task = {"id": "t1", "question": "q"}

    result = await run_spec(spec, task, tools=tools, complete_fn=gateway.complete)

    assert calls["n"] == 1  # not retried
    assert result.error is None
    tool_step = next(s for s in result.steps if s.kind == "tool")
    assert tool_step.error == "no such id: bad"
    assert tool_step.output == {"error": "no such id: bad"}


# ---------------------------------------------------------------------------
# budget + step limits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exhaustion_sets_error_and_zero_score():
    spec = _spec(tools=[], budget_tokens=5)  # FakeCompletion costs 10+10=20 tokens
    gateway = FakeGateway([json.dumps({"action": "final_answer", "final_output": "done"})])
    task = {"id": "t1", "question": "q"}

    result = await run_spec(spec, task, tools=[], complete_fn=gateway.complete)

    assert result.error is not None
    assert "budget_tokens" in result.error
    assert result.score is not None
    assert result.score.score == 0.0
    assert result.score.passed is False


@pytest.mark.asyncio
async def test_max_steps_exhaustion_never_raises():
    spec = _spec(tools=["lookup"], max_steps=2)
    # the model always asks to call a tool and never finishes
    responses = [json.dumps({"action": "tool_call", "tool": "lookup", "tool_args": {}}) for _ in range(10)]
    gateway = FakeGateway(responses)
    tools = [{"name": "lookup", "description": "", "parameters": {}, "fn": lambda: {"ok": True}}]
    task = {"id": "t1", "question": "q"}

    result = await run_spec(spec, task, tools=tools, complete_fn=gateway.complete)

    assert result.error is not None
    assert "max_steps" in result.error
    assert result.step_count <= 2
    assert result.score.score == 0.0


@pytest.mark.asyncio
async def test_run_spec_never_raises_on_gateway_exception():
    spec = _spec(tools=[], max_retries=0, on_failure="abort")

    async def _boom(*args, **kwargs):
        raise RuntimeError("provider is down")

    task = {"id": "t1", "question": "q"}
    result = await run_spec(spec, task, tools=[], complete_fn=_boom)

    assert result.error is not None
    assert result.score.score == 0.0


# ---------------------------------------------------------------------------
# run_batch: aggregation + reliability (score_stddev across k repeats)
# ---------------------------------------------------------------------------


def _flip_priority(priority: str) -> str:
    return "high" if priority != "high" else "low"


def _structured_extraction_scripted_complete():
    """Alternates between a fully-correct extraction and one with a wrong
    (but validly-typed) required field, per task, so score varies across
    repeats -- deterministically, regardless of concurrent call ordering,
    since it derives the answer from the embedded task JSON rather than
    tracking call order globally."""
    seen = defaultdict(int)

    async def _complete(layer, messages, *, response_format=None, max_tokens=None, temperature=None):
        content = messages[-1]["content"]
        task = json.loads(content.split("Task:\n", 1)[1])
        gold = dict(task["gold"])
        seen[task["id"]] += 1
        if seen[task["id"]] % 2 == 0:
            gold["priority"] = _flip_priority(gold["priority"])
        payload = {"action": "final_answer", "final_output": json.dumps(gold)}
        from tests.fakes import FakeCompletion

        return FakeCompletion(text=json.dumps(payload))

    return _complete


@pytest.mark.asyncio
async def test_run_batch_computes_score_stddev_across_k_repeats():
    spec = _spec(name="extractor", tools=["get_schema", "validate_extraction"], goal="Extract fields.")
    complete_fn = _structured_extraction_scripted_complete()

    batch = await run_batch(
        spec, domain="structured_extraction", split="dev", k=4, concurrency=3, complete_fn=complete_fn
    )

    assert isinstance(batch, BatchResult)
    assert batch.domain == "structured_extraction"
    assert batch.split == "dev"
    assert batch.k == 4
    assert batch.spec_version == 1

    num_dev_tasks = len(list((__import__("pathlib").Path("domains/structured_extraction/tasks/dev")).glob("*.json")))
    assert len(batch.runs) == num_dev_tasks * 4

    # exactly half of every task's repeats are deliberately wrong on a
    # required field, so this must show up as real, nonzero reliability
    # variance -- not a constant.
    assert batch.score_stddev > 0.0
    assert 0.0 < batch.mean_score < 1.0
    assert batch.pass_rate == pytest.approx(0.5)
    assert all(r.score is not None for r in batch.runs)
    assert all(r.error is None for r in batch.runs)
    assert batch.mean_input_tokens > 0
    assert batch.mean_steps == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_run_batch_never_raises_when_every_run_fails():
    spec = _spec(tools=[], budget_tokens=1)  # guarantees every run aborts on budget
    gateway = FakeGateway(
        [json.dumps({"action": "final_answer", "final_output": "x"})] * 30
    )

    batch = await run_batch(
        spec, domain="structured_extraction", split="dev", k=2, concurrency=4, complete_fn=gateway.complete
    )

    assert batch.mean_score == 0.0
    assert batch.pass_rate == 0.0
    assert all(r.error is not None for r in batch.runs)


@pytest.mark.asyncio
async def test_run_batch_persists_traces_as_jsonl(tmp_path):
    spec = _spec(tools=[])
    gateway = FakeGateway([json.dumps({"action": "final_answer", "final_output": "x"})] * 30)
    trace_path = tmp_path / "traces.jsonl"

    batch = await run_batch(
        spec,
        domain="structured_extraction",
        split="dev",
        k=1,
        concurrency=2,
        complete_fn=gateway.complete,
        trace_path=trace_path,
    )

    from metaagent.trace import read_trace_jsonl

    lines = read_trace_jsonl(trace_path)
    assert len(lines) == len(batch.runs)
    assert {line["task_id"] for line in lines} == {r.task_id for r in batch.runs}


# ---------------------------------------------------------------------------
# default wiring: real gateway under GATEWAY_MODE=fake (no complete_fn override)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_spec_default_gateway_fake_mode_smoke(monkeypatch):
    monkeypatch.setenv("GATEWAY_MODE", "fake")
    spec = _spec(tools=[])
    task = {"id": "t1", "question": "what is 6*7?"}

    result = await run_spec(spec, task, tools=[])  # no complete_fn -> real metaagent.gateway.complete

    assert result.error is None
    assert result.step_count == 1
    assert isinstance(result.output, str) and result.output
