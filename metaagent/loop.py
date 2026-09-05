"""IMPROVE loop: the closed cycle that makes this a meta-agent instead of a
one-shot generator.

    architect.generate seeds -> run_batch(DEV) -> analyze -> optimize -> repeat

Scientific-integrity rule, non-negotiable: the optimizer only ever sees
diagnoses built from the DEV split. TEST is run once per iteration purely
for reporting (it goes straight into the JSONL record, never into
`analyze()` or `optimize()`). "Best spec" is always chosen by DEV score.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from metaagent import architect
from metaagent.analyzer import Diagnosis, analyze, classify_run
from metaagent.archive import Archive
from metaagent.evaluation import DOMAINS_DIR, load_tools
from metaagent.optimizer import optimize
from metaagent.runner import BatchResult, run_batch
from metaagent.spec.models import AgentSpec

DEFAULT_RESULTS_DIR = Path("results")


@dataclass
class IterationRecord:
    iteration: int
    spec_id: str
    spec_version: str
    dev_mean_score: float
    dev_pass_rate: float
    dev_score_stddev: float
    test_mean_score: float
    test_pass_rate: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_reasoning_tokens: float
    mean_latency_ms: float
    mean_steps: float
    dominant_failure_mode: Optional[str]
    mutation_rationale: str
    is_best_so_far: bool
    timestamp: float


@dataclass
class LoopResult:
    domain: str
    iterations: list[IterationRecord]
    best_spec: AgentSpec
    best_dev_score: float
    results_path: str
    archive_path: str


def _load_goal_text(domain: str) -> str:
    goal_path = DOMAINS_DIR / domain / "goal.md"
    if goal_path.is_file():
        return goal_path.read_text()
    return f"Solve {domain} tasks correctly and efficiently."


def _append_jsonl(path: Path, record: IterationRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def _archive_attempt(
    archive: Archive,
    *,
    goal: str,
    domain: str,
    spec: AgentSpec,
    dev_batch: BatchResult,
    test_batch: Optional[BatchResult],
    max_steps: Optional[int],
) -> list[str]:
    failure_modes = sorted(
        {
            result[0].value
            for run in dev_batch.runs
            if (result := classify_run(run, max_steps=max_steps)) is not None
        }
    )
    scores: dict[str, float] = {
        "dev_mean_score": dev_batch.mean_score,
        "dev_pass_rate": dev_batch.pass_rate,
    }
    if test_batch is not None:
        scores["test_mean_score"] = test_batch.mean_score
        scores["test_pass_rate"] = test_batch.pass_rate
    archive.add(
        goal=goal,
        domain=domain,
        spec=spec.model_dump(mode="json"),
        scores=scores,
        failure_modes=failure_modes,
    )
    return failure_modes


async def run_loop(
    domain: str,
    *,
    iterations: int = 5,
    k: int = 3,
    n_seeds: int = 3,
    concurrency: int = 4,
    goal: Optional[str] = None,
    archive: Optional[Archive] = None,
    results_path: Optional[Path] = None,
    plateau_patience: int = 2,
    plateau_epsilon: float = 0.01,
    complete_fn: Any = None,
) -> LoopResult:
    """Run the ANALYZE/IMPROVE cycle for `domain` up to `iterations` times,
    optimizing against the DEV split and scoring TEST once per iteration
    purely for reporting. Stops early on a DEV-score plateau."""
    goal_text = goal or _load_goal_text(domain)
    tools = load_tools(domain)
    archive = archive if archive is not None else Archive(Path(f"archive_{domain}.jsonl"))
    results_path = results_path or (DEFAULT_RESULTS_DIR / f"{domain}.jsonl")

    seeds = await architect.generate(
        goal_text, tools, n_seeds=n_seeds, archive=archive, complete_fn=complete_fn,
    )
    if not seeds:
        raise ValueError("architect.generate returned no viable seeds")

    for seed in seeds:
        if seed.id is None:
            seed.id = str(uuid.uuid4())

    best_spec: Optional[AgentSpec] = None
    best_dev_score = float("-inf")
    for seed in seeds:
        dev_batch = await run_batch(seed, domain, "dev", k=k, concurrency=concurrency, complete_fn=complete_fn)
        _archive_attempt(
            archive, goal=goal_text, domain=domain, spec=seed,
            dev_batch=dev_batch, test_batch=None, max_steps=seed.orchestration.max_steps,
        )
        if dev_batch.mean_score > best_dev_score:
            best_dev_score, best_spec = dev_batch.mean_score, seed

    assert best_spec is not None
    candidate = best_spec
    records: list[IterationRecord] = []
    plateau_count = 0

    for i in range(iterations):
        dev_batch = await run_batch(candidate, domain, "dev", k=k, concurrency=concurrency, complete_fn=complete_fn)

        # DEV feeds the optimizer. TEST below is reporting-only and must never
        # be passed to analyze() or optimize().
        diagnosis: Diagnosis = await analyze(
            dev_batch, complete_fn=complete_fn, max_steps=candidate.orchestration.max_steps,
        )

        test_batch = await run_batch(candidate, domain, "test", k=k, concurrency=concurrency, complete_fn=complete_fn)

        _archive_attempt(
            archive, goal=goal_text, domain=domain, spec=candidate,
            dev_batch=dev_batch, test_batch=test_batch, max_steps=candidate.orchestration.max_steps,
        )

        previous_best = best_dev_score
        is_best = dev_batch.mean_score > best_dev_score
        if is_best:
            best_dev_score, best_spec = dev_batch.mean_score, candidate

        mutation = optimize(candidate, diagnosis, tools)

        record = IterationRecord(
            iteration=i,
            spec_id=candidate.id or "",
            spec_version=candidate.version,
            dev_mean_score=dev_batch.mean_score,
            dev_pass_rate=dev_batch.pass_rate,
            dev_score_stddev=dev_batch.score_stddev,
            test_mean_score=test_batch.mean_score,
            test_pass_rate=test_batch.pass_rate,
            mean_input_tokens=dev_batch.mean_input_tokens,
            mean_output_tokens=dev_batch.mean_output_tokens,
            mean_reasoning_tokens=dev_batch.mean_reasoning_tokens,
            mean_latency_ms=dev_batch.mean_latency_ms,
            mean_steps=dev_batch.mean_steps,
            dominant_failure_mode=diagnosis.dominant_mode,
            mutation_rationale=mutation.rationale,
            is_best_so_far=is_best,
            timestamp=time.time(),
        )
        records.append(record)
        _append_jsonl(results_path, record)

        if is_best and best_dev_score - previous_best > plateau_epsilon:
            plateau_count = 0
        else:
            plateau_count += 1
        if plateau_count >= plateau_patience:
            break

        mutation.spec.id = str(uuid.uuid4())
        candidate = mutation.spec

    assert best_spec is not None
    return LoopResult(
        domain=domain,
        iterations=records,
        best_spec=best_spec,
        best_dev_score=best_dev_score,
        results_path=str(results_path),
        archive_path=str(archive.path),
    )
