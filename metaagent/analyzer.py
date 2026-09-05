"""ANALYZE stage: diagnose WHY a batch of runs failed.

Two passes, deliberately in this order:

1. MECHANICAL (this module, no LLM): classify every failed run into a fixed
   taxonomy by reading `Score.details` (the evaluator's own structured
   breakdown -- parse errors, execution errors, missing/extra tool calls,
   ordering violations, missing fields) and the run's trace steps. This is
   cheap, deterministic, and reproducible.
2. INTERPRETIVE (one meta-layer call): hand the *aggregated counts and
   cited evidence* -- never raw transcripts -- to the meta layer and ask it
   to narrate the dominant failure mode in plain language. One call per
   batch, not one call per run.

Never skip straight to "ask an LLM what went wrong": that reads the same
transcripts every time with no memory of frequency, is expensive, and is
far less reliable than aggregate-then-interpret.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol


class FailureMode(str, Enum):
    """Fixed taxonomy. Extend here, not by inventing ad-hoc strings elsewhere."""

    MALFORMED_OUTPUT = "malformed_output"
    WRONG_TOOL_CHOSEN = "wrong_tool_chosen"
    TOOL_CALL_ERRORED = "tool_call_errored"
    MISSING_CONTEXT = "missing_context"
    REASONING_BUDGET_EXHAUSTED = "reasoning_budget_exhausted"
    INSTRUCTION_IGNORED = "instruction_ignored"
    LOW_QUALITY_OUTPUT = "low_quality_output"
    RUNNER_ERROR = "runner_error"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class FailureInstance:
    task_id: str
    mode: FailureMode
    evidence: str


@dataclass
class Diagnosis:
    domain: str
    split: str
    total_runs: int
    failed_runs: int
    distribution: dict[str, int]
    dominant_mode: Optional[str]
    instances: list[FailureInstance]
    interpretation: str
    mean_score: float = 0.0
    pass_rate: float = 0.0
    mean_output_tokens: float = 0.0
    mean_latency_ms: float = 0.0
    mean_steps: float = 0.0


class Completion(Protocol):
    text: str


class CompleteFn(Protocol):
    async def __call__(
        self,
        layer: str,
        messages: list[dict[str, str]],
        *,
        response_format: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Completion: ...


def _step_get(step: Any, key: str, default: Any = None) -> Any:
    if isinstance(step, dict):
        return step.get(key, default)
    return getattr(step, key, default)


def classify_run(run: Any, *, max_steps: Optional[int] = None) -> Optional[tuple[FailureMode, str]]:
    """Mechanically classify one run. Returns None if the run did not fail.

    `run` is duck-typed against the RunResult contract (task_id, output,
    score, steps, error, step_count, ...) so this works whether the caller
    passes real runner.RunResult instances or lightweight test doubles.
    """
    error = getattr(run, "error", None)
    if error:
        return FailureMode.RUNNER_ERROR, f"run error: {error}"

    for step in getattr(run, "steps", None) or []:
        step_error = _step_get(step, "error")
        if step_error:
            tool = _step_get(step, "tool", "?")
            idx = _step_get(step, "index", "?")
            return FailureMode.TOOL_CALL_ERRORED, f"step {idx} (tool={tool!r}) errored: {step_error}"

    score = getattr(run, "score", None)
    if score is None:
        return FailureMode.RUNNER_ERROR, "no score recorded for this run"

    if getattr(score, "passed", True):
        return None

    details = getattr(score, "details", None) or {}

    parse_error = details.get("parse_error") or details.get("json_parse_error")
    if parse_error:
        return FailureMode.MALFORMED_OUTPUT, f"output failed to parse: {parse_error}"

    execution_error = details.get("execution_error")
    if execution_error:
        return FailureMode.TOOL_CALL_ERRORED, f"output executed but errored: {execution_error}"

    step_count = getattr(run, "step_count", None)
    if max_steps is not None and step_count is not None and step_count >= max_steps:
        return (
            FailureMode.REASONING_BUDGET_EXHAUSTED,
            f"step_count {step_count} reached orchestration max_steps {max_steps}",
        )

    order_violations = details.get("order_violations")
    if order_violations:
        return FailureMode.INSTRUCTION_IGNORED, f"ordering constraints violated: {order_violations}"

    missing_calls = details.get("missing_calls")
    if missing_calls:
        tool_calls_made = details.get("tool_calls_made") or []
        if not tool_calls_made:
            return (
                FailureMode.MISSING_CONTEXT,
                f"no tool calls attempted; required calls never made: {missing_calls}",
            )
        return (
            FailureMode.WRONG_TOOL_CHOSEN,
            f"tool calls made but required calls unmet: {missing_calls}",
        )

    missing_fields = details.get("missing_fields")
    if missing_fields:
        return FailureMode.MISSING_CONTEXT, f"output missing required fields: {missing_fields}"

    return (
        FailureMode.LOW_QUALITY_OUTPUT,
        f"score={getattr(score, 'score', None)} below passing with no structural signal in details",
    )


def _select_examples(instances: list[FailureInstance], dominant: Optional[str], limit: int) -> list[FailureInstance]:
    dominant_first = sorted(instances, key=lambda i: i.mode.value != dominant)
    return dominant_first[:limit]


def _build_interpretation_prompt(
    domain: str,
    split: str,
    total_runs: int,
    distribution: Counter,
    examples: list[FailureInstance],
) -> list[dict[str, str]]:
    dist_lines = "\n".join(f"- {mode}: {count}" for mode, count in distribution.most_common()) or "(no failures)"
    evidence_lines = "\n".join(f"- [{i.mode.value}] {i.task_id}: {i.evidence}" for i in examples) or "(none)"
    system = (
        "You are the interpretive step of a meta-agent's failure analyzer. "
        "You are given MECHANICALLY computed failure-mode counts and a small "
        "set of cited evidence lines for one eval batch -- not raw transcripts. "
        "Write a concise (3-6 sentence) diagnosis naming the dominant failure "
        "mode, explaining plausibly why it is happening based on the cited "
        "evidence, and noting anything a spec mutation could plausibly fix. "
        "Do not invent evidence beyond what is given."
    )
    user = (
        f"Domain: {domain}\nSplit: {split}\nTotal runs: {total_runs}\n\n"
        f"Failure mode distribution:\n{dist_lines}\n\nCited evidence:\n{evidence_lines}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def analyze(
    batch: Any,
    *,
    complete_fn: Optional[CompleteFn] = None,
    max_steps: Optional[int] = None,
    max_examples: int = 8,
) -> Diagnosis:
    """Diagnose a BatchResult-like object (spec_version, domain, split, runs, ...)."""
    if complete_fn is None:
        from metaagent.gateway import complete as complete_fn  # noqa: PLC0415

    runs = list(getattr(batch, "runs", None) or [])
    instances: list[FailureInstance] = []
    for run in runs:
        result = classify_run(run, max_steps=max_steps)
        if result is None:
            continue
        mode, evidence = result
        instances.append(FailureInstance(task_id=str(getattr(run, "task_id", "?")), mode=mode, evidence=evidence))

    distribution = Counter(i.mode.value for i in instances)
    dominant = distribution.most_common(1)[0][0] if distribution else None
    examples = _select_examples(instances, dominant, max_examples)

    domain = getattr(batch, "domain", "")
    split = getattr(batch, "split", "")
    if not instances:
        interpretation = "No failures observed in this batch; nothing to diagnose."
    else:
        messages = _build_interpretation_prompt(domain, split, len(runs), distribution, examples)
        completion = await complete_fn("meta", messages, temperature=0.2)
        interpretation = completion.text

    return Diagnosis(
        domain=domain,
        split=split,
        total_runs=len(runs),
        failed_runs=len(instances),
        distribution=dict(distribution),
        dominant_mode=dominant,
        instances=examples,
        interpretation=interpretation,
        mean_score=getattr(batch, "mean_score", 0.0),
        pass_rate=getattr(batch, "pass_rate", 0.0),
        mean_output_tokens=getattr(batch, "mean_output_tokens", 0.0),
        mean_latency_ms=getattr(batch, "mean_latency_ms", 0.0),
        mean_steps=getattr(batch, "mean_steps", 0.0),
    )
