"""EXECUTE stage: run an AgentSpec against a domain/split and score it.

PLACEHOLDER: a runner worker owns this module and is landing a real
interpreter (multi-role orchestration walk, tool execution, memory) in
parallel. This is a local stub matching that worker's documented contract
exactly (RunResult / BatchResult / run_batch signature) so the ANALYZE and
IMPROVE stages can be built and tested offline against it. It currently
does a single-shot call to the entry role with no tool-calling loop -- good
enough to exercise the analyzer/optimizer/loop plumbing under
GATEWAY_MODE=fake, not a claim about eval quality.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from metaagent.evaluation import Evaluator, Score, load_evaluator, load_tasks
from metaagent.gateway import complete
from metaagent.spec.models import AgentSpec, Role


@dataclass
class RunResult:
    task_id: str
    output: str
    score: Score
    steps: list[dict[str, Any]]
    error: Optional[str]
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_latency_ms: float
    step_count: int
    tool_call_count: int


@dataclass
class BatchResult:
    spec_version: str
    domain: str
    split: str
    k: int
    mean_score: float
    pass_rate: float
    score_stddev: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_reasoning_tokens: float
    mean_latency_ms: float
    mean_steps: float
    runs: list[RunResult] = field(default_factory=list)


def _entry_role(spec: AgentSpec) -> Role:
    for role in spec.roles:
        if role.name == spec.orchestration.entry_role:
            return role
    raise ValueError(f"entry_role {spec.orchestration.entry_role!r} not found in spec roles")


def _task_prompt(task: dict[str, Any]) -> str:
    return task.get("question") or task.get("text") or json.dumps(task)


async def _run_single(spec: AgentSpec, task: dict[str, Any], evaluator: Evaluator) -> RunResult:
    role = _entry_role(spec)
    task_id = task.get("id", "unknown")
    messages = [
        {"role": "system", "content": role.system_prompt},
        {"role": "user", "content": _task_prompt(task)},
    ]

    try:
        completion = await complete(
            "worker", messages, max_tokens=role.max_tokens, temperature=role.temperature,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a RUNNER_ERROR failure mode
        return RunResult(
            task_id=task_id,
            output="",
            score=Score(score=0.0, passed=False, details={"runner_error": str(exc)}),
            steps=[],
            error=str(exc),
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            total_latency_ms=0.0,
            step_count=0,
            tool_call_count=0,
        )

    score = evaluator.evaluate(task, completion.text)
    step = {
        "index": 0,
        "role": role.name,
        "type": "final_output",
        "content": completion.text,
        "tool": None,
        "tool_args": None,
        "tool_result": None,
        "error": None,
    }
    return RunResult(
        task_id=task_id,
        output=completion.text,
        score=score,
        steps=[step],
        error=None,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        reasoning_tokens=0,
        total_latency_ms=completion.latency_ms,
        step_count=1,
        tool_call_count=0,
    )


async def run_batch(
    spec: AgentSpec, domain: str, split: str, k: int = 3, concurrency: int = 4,
) -> BatchResult:
    tasks = load_tasks(domain, split)
    evaluator = load_evaluator(domain)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _bounded(task: dict[str, Any]) -> RunResult:
        async with semaphore:
            return await _run_single(spec, task, evaluator)

    jobs = [_bounded(task) for task in tasks for _ in range(k)]
    runs: list[RunResult] = list(await asyncio.gather(*jobs)) if jobs else []

    n = len(runs)
    scores = [r.score.score for r in runs]
    mean_score = sum(scores) / n if n else 0.0
    pass_rate = sum(1 for r in runs if r.score.passed) / n if n else 0.0
    variance = sum((s - mean_score) ** 2 for s in scores) / n if n else 0.0

    def _mean(attr: str) -> float:
        return sum(getattr(r, attr) for r in runs) / n if n else 0.0

    return BatchResult(
        spec_version=spec.version,
        domain=domain,
        split=split,
        k=k,
        mean_score=mean_score,
        pass_rate=pass_rate,
        score_stddev=variance**0.5,
        mean_input_tokens=_mean("input_tokens"),
        mean_output_tokens=_mean("output_tokens"),
        mean_reasoning_tokens=_mean("reasoning_tokens"),
        mean_latency_ms=_mean("total_latency_ms"),
        mean_steps=_mean("step_count"),
        runs=runs,
    )
