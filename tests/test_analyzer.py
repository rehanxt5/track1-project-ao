import asyncio
from types import SimpleNamespace

from metaagent.analyzer import FailureMode, analyze, classify_run
from metaagent.evaluation import Score
from tests.fakes import FakeGateway


def _run(*, score=None, error=None, steps=None, step_count=1, task_id="t1"):
    """A duck-typed RunResult stand-in -- analyzer must work against any
    object exposing this shape, not just metaagent.runner.RunResult."""
    return SimpleNamespace(
        task_id=task_id, score=score, error=error, steps=steps or [], step_count=step_count,
    )


def test_classify_run_returns_none_when_passed():
    run = _run(score=Score(score=1.0, passed=True, details={}))
    assert classify_run(run) is None


def test_classify_run_malformed_output_from_parse_error():
    run = _run(score=Score(score=0.0, passed=False, details={"parse_error": "bad json"}))
    mode, evidence = classify_run(run)
    assert mode is FailureMode.MALFORMED_OUTPUT
    assert "bad json" in evidence


def test_classify_run_malformed_output_from_json_parse_error_key():
    run = _run(score=Score(score=0.0, passed=False, details={"json_parse_error": "oops"}))
    mode, _ = classify_run(run)
    assert mode is FailureMode.MALFORMED_OUTPUT


def test_classify_run_tool_call_errored_from_execution_error():
    run = _run(score=Score(score=0.0, passed=False, details={"execution_error": "no such column"}))
    mode, evidence = classify_run(run)
    assert mode is FailureMode.TOOL_CALL_ERRORED
    assert "no such column" in evidence


def test_classify_run_tool_call_errored_from_step_error_takes_priority():
    run = _run(
        score=Score(score=0.0, passed=False, details={"parse_error": "bad json"}),
        steps=[{"index": 2, "tool": "run_query", "error": "syntax error"}],
    )
    mode, evidence = classify_run(run)
    assert mode is FailureMode.TOOL_CALL_ERRORED
    assert "run_query" in evidence and "syntax error" in evidence


def test_classify_run_missing_context_when_no_tool_calls_attempted():
    run = _run(
        score=Score(
            score=0.0,
            passed=False,
            details={"missing_calls": [{"tool": "get_order"}], "tool_calls_made": []},
        )
    )
    mode, _ = classify_run(run)
    assert mode is FailureMode.MISSING_CONTEXT


def test_classify_run_wrong_tool_chosen_when_some_tool_calls_attempted():
    run = _run(
        score=Score(
            score=0.3,
            passed=False,
            details={
                "missing_calls": [{"tool": "get_employee"}],
                "tool_calls_made": [{"tool": "get_order", "args": {}}],
            },
        )
    )
    mode, _ = classify_run(run)
    assert mode is FailureMode.WRONG_TOOL_CHOSEN


def test_classify_run_instruction_ignored_from_order_violations():
    run = _run(
        score=Score(
            score=0.5, passed=False, details={"order_violations": [{"before": "a", "after": "b"}]},
        )
    )
    mode, _ = classify_run(run)
    assert mode is FailureMode.INSTRUCTION_IGNORED


def test_classify_run_missing_context_from_missing_fields():
    run = _run(score=Score(score=0.5, passed=False, details={"missing_fields": ["priority"]}))
    mode, _ = classify_run(run)
    assert mode is FailureMode.MISSING_CONTEXT


def test_classify_run_reasoning_budget_exhausted():
    run = _run(score=Score(score=0.2, passed=False, details={}), step_count=10)
    mode, evidence = classify_run(run, max_steps=10)
    assert mode is FailureMode.REASONING_BUDGET_EXHAUSTED
    assert "10" in evidence


def test_classify_run_runner_error_takes_priority_over_score():
    run = _run(score=Score(score=1.0, passed=True, details={}), error="gateway timeout")
    mode, evidence = classify_run(run)
    assert mode is FailureMode.RUNNER_ERROR
    assert "gateway timeout" in evidence


def test_classify_run_low_quality_fallback():
    run = _run(score=Score(score=0.6, passed=False, details={}))
    mode, _ = classify_run(run)
    assert mode is FailureMode.LOW_QUALITY_OUTPUT


def test_analyze_aggregates_mechanically_and_makes_one_meta_call():
    runs = [
        _run(task_id="a", score=Score(score=0.0, passed=False, details={"parse_error": "x"})),
        _run(task_id="b", score=Score(score=0.0, passed=False, details={"parse_error": "y"})),
        _run(task_id="c", score=Score(score=0.5, passed=False, details={"missing_fields": ["p"]})),
        _run(task_id="d", score=Score(score=1.0, passed=True, details={})),
    ]
    batch = SimpleNamespace(
        domain="structured_extraction", split="dev", runs=runs,
        mean_score=0.375, pass_rate=0.25, mean_output_tokens=42.0, mean_latency_ms=7.0, mean_steps=1.0,
    )
    gateway = FakeGateway(["Dominant issue: malformed JSON output."])

    diagnosis = asyncio.run(analyze(batch, complete_fn=gateway.complete))

    assert gateway.call_count == 1, "must aggregate first and interpret with a single meta call"
    assert diagnosis.total_runs == 4
    assert diagnosis.failed_runs == 3
    assert diagnosis.dominant_mode == FailureMode.MALFORMED_OUTPUT.value
    assert diagnosis.distribution[FailureMode.MALFORMED_OUTPUT.value] == 2
    assert diagnosis.distribution[FailureMode.MISSING_CONTEXT.value] == 1
    assert diagnosis.interpretation == "Dominant issue: malformed JSON output."
    assert diagnosis.mean_score == 0.375
    # the passing run must not appear as cited evidence
    assert all(i.task_id != "d" for i in diagnosis.instances)


def test_analyze_skips_meta_call_when_nothing_failed():
    runs = [_run(task_id="a", score=Score(score=1.0, passed=True, details={}))]
    batch = SimpleNamespace(
        domain="sql_generation", split="dev", runs=runs,
        mean_score=1.0, pass_rate=1.0, mean_output_tokens=0.0, mean_latency_ms=0.0, mean_steps=0.0,
    )
    gateway = FakeGateway([])  # would raise if analyze() called it
    diagnosis = asyncio.run(analyze(batch, complete_fn=gateway.complete))
    assert gateway.call_count == 0
    assert diagnosis.dominant_mode is None
    assert diagnosis.total_runs == 1
    assert diagnosis.failed_runs == 0
