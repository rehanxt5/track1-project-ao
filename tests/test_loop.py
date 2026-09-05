import asyncio
import json
from types import SimpleNamespace

import metaagent.architect as architect_module
import metaagent.loop as loop_module
import metaagent.optimizer as optimizer_module
from metaagent.archive import Archive
from metaagent.evaluation import Score
from metaagent.runner import BatchResult
from metaagent.spec.models import AgentSpec
from tests.fakes import SAMPLE_TOOLS, FakeGateway, valid_spec_dict


def _fake_seed(monkeypatch, n=1):
    seed = AgentSpec.model_validate(valid_spec_dict())

    async def fake_generate(goal, tools, n_seeds=3, archive=None, *, complete_fn=None, max_repair_attempts=2):
        return [seed.model_copy(deep=True) for _ in range(n)]

    monkeypatch.setattr(architect_module, "generate", fake_generate)
    return seed


def _run(task_id, *, passed, details):
    return SimpleNamespace(
        task_id=task_id,
        score=Score(score=0.9 if passed else 0.0, passed=passed, details=details),
        error=None,
        steps=[],
        step_count=1,
    )


def _make_fake_run_batch():
    """DEV and TEST are made deliberately distinguishable -- DEV runs fail
    with a malformed-output signal, TEST runs pass cleanly -- so any
    contamination (test data reaching the optimizer, or dev data being
    reported as test) is immediately visible in the assertions."""
    dev_run = _run("dev-1", passed=False, details={"parse_error": "bad json"})
    test_run = _run("test-1", passed=True, details={})

    async def fake_run_batch(spec, domain, split, k=3, concurrency=4):
        if split == "dev":
            return BatchResult(
                spec_version=spec.version, domain=domain, split="dev", k=k,
                mean_score=0.9, pass_rate=0.0, score_stddev=0.0,
                mean_input_tokens=10.0, mean_output_tokens=20.0, mean_reasoning_tokens=0.0,
                mean_latency_ms=5.0, mean_steps=1.0, runs=[dev_run],
            )
        return BatchResult(
            spec_version=spec.version, domain=domain, split="test", k=k,
            mean_score=0.1, pass_rate=1.0, score_stddev=0.0,
            mean_input_tokens=10.0, mean_output_tokens=20.0, mean_reasoning_tokens=0.0,
            mean_latency_ms=5.0, mean_steps=1.0, runs=[test_run],
        )

    return fake_run_batch


def test_optimizer_only_ever_sees_dev_derived_diagnoses(tmp_path, monkeypatch):
    _fake_seed(monkeypatch)
    monkeypatch.setattr(loop_module, "run_batch", _make_fake_run_batch())

    captured_diagnoses = []
    real_optimize = optimizer_module.optimize

    def spy_optimize(spec, diagnosis, tools, **kwargs):
        captured_diagnoses.append(diagnosis)
        return real_optimize(spec, diagnosis, tools, **kwargs)

    monkeypatch.setattr(loop_module, "optimize", spy_optimize)

    archive = Archive(tmp_path / "archive.jsonl")
    results_path = tmp_path / "results.jsonl"
    gateway = FakeGateway(["dev diagnosis text"] * 10)

    result = asyncio.run(
        loop_module.run_loop(
            "sql_generation",
            iterations=2,
            k=1,
            n_seeds=1,
            archive=archive,
            results_path=results_path,
            complete_fn=gateway.complete,
        )
    )

    assert captured_diagnoses, "optimize() was never called"
    for diagnosis in captured_diagnoses:
        # This is the core regression: every diagnosis handed to the
        # optimizer must be built from split="dev" with dev's numbers --
        # never test's -- regardless of what TEST scored that iteration.
        assert diagnosis.split == "dev"
        assert diagnosis.mean_score == 0.9
        assert diagnosis.total_runs == 1
        assert diagnosis.dominant_mode == "malformed_output"
        assert all(inst.task_id != "test-1" for inst in diagnosis.instances)

    # best spec is chosen by DEV score (0.9), never by TEST score (0.1)
    assert result.best_dev_score == 0.9

    # but TEST is still recorded per iteration, purely for reporting
    for record in result.iterations:
        assert record.dev_mean_score == 0.9
        assert record.test_mean_score == 0.1


def test_jsonl_and_archive_are_written(tmp_path, monkeypatch):
    _fake_seed(monkeypatch)
    monkeypatch.setattr(loop_module, "run_batch", _make_fake_run_batch())

    archive = Archive(tmp_path / "archive.jsonl")
    results_path = tmp_path / "results.jsonl"
    gateway = FakeGateway(["diag"] * 10)

    result = asyncio.run(
        loop_module.run_loop(
            "sql_generation",
            iterations=2,
            k=1,
            n_seeds=1,
            archive=archive,
            results_path=results_path,
            complete_fn=gateway.complete,
        )
    )

    lines = results_path.read_text().strip().splitlines()
    assert len(lines) == len(result.iterations) == 2
    for line in lines:
        record = json.loads(line)
        assert set(record) >= {
            "iteration", "spec_version", "dev_mean_score", "test_mean_score",
            "dominant_failure_mode", "mutation_rationale",
        }

    entries = archive.load()
    # 1 seed archived during selection + 1 per loop iteration
    assert len(entries) == 1 + len(result.iterations)
    assert all(e.domain == "sql_generation" for e in entries)
    assert any("malformed_output" in e.failure_modes for e in entries)


def test_plateau_stops_before_exhausting_iteration_budget(tmp_path, monkeypatch):
    _fake_seed(monkeypatch)
    monkeypatch.setattr(loop_module, "run_batch", _make_fake_run_batch())

    archive = Archive(tmp_path / "archive.jsonl")
    results_path = tmp_path / "results.jsonl"
    gateway = FakeGateway(["diag"] * 50)

    result = asyncio.run(
        loop_module.run_loop(
            "sql_generation",
            iterations=20,
            k=1,
            n_seeds=1,
            archive=archive,
            results_path=results_path,
            plateau_patience=2,
            complete_fn=gateway.complete,
        )
    )
    # dev score never improves past the first iteration under the fake
    # run_batch, so the plateau should cut this off well short of 20
    assert len(result.iterations) < 20
