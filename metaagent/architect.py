"""The architect: goal + tools -> a handful of verified AgentSpec seeds.

Five steps, in order:
  PERCEIVE   - read the goal and the tools actually available.
  RETRIEVE   - pull similar past specs and their failure modes from the archive.
  SELECT     - pick a design philosophy per seed from a fixed library.
  SYNTHESIZE - one meta-layer call, constrained JSON, N diverse seeds at once.
  VERIFY     - schema + semantic checks, with a bounded repair-and-retry loop.

`complete` and the archive/gateway shapes are owned by other workers; this
module only depends on their call signatures (injected as `complete_fn`),
so it has zero import-time dependency on modules that don't exist yet.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Protocol, Union

from pydantic import ValidationError

from metaagent.archive import Archive, ArchiveEntry
from metaagent.config import architect_reasoning_default
from metaagent.gateway import GatewayReasoningBudgetError
from metaagent.spec.models import SPEC_VERSION, AgentSpec, DesignPhilosophy
from metaagent.spec.verify import VerificationError, verify_spec

logger = logging.getLogger("metaagent.architect")

# SYNTHESIZE and repair calls are the largest outputs in the system: a
# reasoning model thinks silently for a while, then emits either N full
# multi-role AgentSpec JSON objects at once (synthesis) or one (repair).
# Relying on the gateway's global DEFAULT_MAX_TOKENS truncates them (that's
# the bug this sizing fixes) so these calls always pass an explicit budget.
#
# Measured live against glm-4-7-flash/TensorMux, architect.generate(n_seeds=3):
#   max_tokens=2048  -> FAILS, truncated at 2130 reasoning tokens, 0 output
#   max_tokens=8192  -> ok, 1705 reasoning + 3227 output tokens
#   max_tokens=16384 -> ok, 3420 reasoning + 1154 output tokens
# SYNTHESIS_REASONING_RESERVE sits above the largest reasoning spend
# observed (3420) with headroom for slower runs; SYNTHESIS_TOKENS_PER_SEED
# covers one multi-role AgentSpec JSON object's worth of output.
SYNTHESIS_REASONING_RESERVE = 4096
SYNTHESIS_TOKENS_PER_SEED = 2048

# If a synthesis/repair call still exhausts its budget on reasoning alone
# (GatewayReasoningBudgetError), retry exactly once with a bigger budget
# rather than dropping the seed. Capped so a pathological prompt can't spiral
# into an expensive loop.
BUDGET_RETRY_MULTIPLIER = 2
BUDGET_RETRY_CAP = 32768


def _synthesis_max_tokens(n_seeds: int) -> int:
    return SYNTHESIS_REASONING_RESERVE + SYNTHESIS_TOKENS_PER_SEED * max(1, n_seeds)


REPAIR_MAX_TOKENS = _synthesis_max_tokens(1)


class ArchitectError(RuntimeError):
    """Raised when synthesis or repair cannot produce any usable spec."""


class Completion(Protocol):
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    provider: str


class CompleteFn(Protocol):
    async def __call__(
        self,
        layer: str,
        messages: list[dict[str, str]],
        *,
        response_format: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[Union[bool, str]] = None,
    ) -> Completion: ...


async def _complete_with_budget_retry(
    complete_fn: CompleteFn,
    layer: str,
    messages: list[dict[str, str]],
    *,
    response_format: Optional[dict[str, Any]],
    max_tokens: int,
    temperature: Optional[float],
    reasoning: Optional[Union[bool, str]],
) -> Completion:
    """Call complete_fn with the given budget; if a reasoning model exhausts
    it purely on chain-of-thought, retry exactly once with a larger budget
    before letting the error propagate. See GatewayReasoningBudgetError."""
    try:
        return await complete_fn(
            layer,
            messages,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning=reasoning,
        )
    except GatewayReasoningBudgetError as exc:
        retry_tokens = min(max_tokens * BUDGET_RETRY_MULTIPLIER, BUDGET_RETRY_CAP)
        if retry_tokens <= max_tokens:
            raise
        logger.warning(
            "architect retrying %s call with larger budget after reasoning "
            "exhaustion: %d -> %d max_tokens (%s)",
            layer,
            max_tokens,
            retry_tokens,
            exc,
        )
        return await complete_fn(
            layer,
            messages,
            response_format=response_format,
            max_tokens=retry_tokens,
            temperature=temperature,
            reasoning=reasoning,
        )


PHILOSOPHY_LIBRARY: dict[str, dict[str, str]] = {
    "react": {
        "topology": "single_agent",
        "summary": (
            "Interleave reasoning and tool calls in a single loop "
            "(Thought/Action/Observation) until the goal is satisfied."
        ),
    },
    "plan_execute": {
        "topology": "supervisor_workers",
        "summary": (
            "A planner role decomposes the goal into an ordered task list, "
            "then an executor role carries out each task with tools."
        ),
    },
    "plan_critic_reflect": {
        "topology": "pipeline",
        "summary": (
            "A planner drafts a solution, a critic role reviews it against the "
            "goal, and the plan is revised iteratively before finalizing."
        ),
    },
}


async def generate(
    goal: str,
    tools: list[dict[str, Any]],
    n_seeds: int = 3,
    archive: Optional[Archive] = None,
    *,
    complete_fn: Optional[CompleteFn] = None,
    max_repair_attempts: int = 2,
    reasoning: Optional[Union[bool, str]] = None,
) -> list[AgentSpec]:
    if not goal or not goal.strip():
        raise ValueError("goal must not be blank")

    if complete_fn is None:
        from metaagent.gateway import complete as complete_fn  # noqa: PLC0415

    if reasoning is None:
        reasoning = architect_reasoning_default()

    # RETRIEVE
    retrieved = archive.retrieve_similar(goal, tools, k=3) if archive is not None else []
    failure_modes = _collect_failure_modes(retrieved)

    # SELECT
    philosophies = _select_philosophies(n_seeds, retrieved)

    # SYNTHESIZE
    messages = _build_synthesis_prompt(goal, tools, philosophies, failure_modes)
    completion = await _complete_with_budget_retry(
        complete_fn,
        "meta",
        messages,
        response_format={"type": "json_object"},
        max_tokens=_synthesis_max_tokens(len(philosophies)),
        temperature=0.9,
        reasoning=reasoning,
    )
    raw_seeds = _parse_seeds(completion.text)
    raw_seeds = raw_seeds[: len(philosophies)]

    # VERIFY, with repair-and-retry
    specs: list[AgentSpec] = []
    for i, philosophy in enumerate(philosophies):
        raw = raw_seeds[i] if i < len(raw_seeds) else {}
        spec = await _verify_and_repair(
            raw, tools, complete_fn, philosophy, max_repair_attempts, reasoning
        )
        if spec is not None:
            specs.append(spec)

    if not specs:
        raise ArchitectError("all synthesized seeds failed verification and repair")
    return specs


def _collect_failure_modes(retrieved: list[ArchiveEntry], limit: int = 5) -> list[str]:
    seen: list[str] = []
    for entry in retrieved:
        for mode in entry.failure_modes:
            if mode not in seen:
                seen.append(mode)
    return seen[:limit]


def _select_philosophies(
    n_seeds: int, retrieved: list[ArchiveEntry]
) -> list[DesignPhilosophy]:
    library = list(PHILOSOPHY_LIBRARY.keys())
    n = max(1, n_seeds)

    poor: set[str] = set()
    for entry in retrieved:
        phil = entry.spec.get("design_philosophy")
        if phil and entry.scores.get("accuracy", 1.0) < 0.4:
            poor.add(phil)
    ordered = sorted(library, key=lambda p: p in poor)

    selected: list[DesignPhilosophy] = []
    i = 0
    while len(selected) < n:
        selected.append(DesignPhilosophy(ordered[i % len(ordered)]))
        i += 1
    return selected


def _build_synthesis_prompt(
    goal: str,
    tools: list[dict[str, Any]],
    philosophies: list[DesignPhilosophy],
    failure_modes: list[str],
) -> list[dict[str, str]]:
    schema = AgentSpec.model_json_schema()
    tool_lines = "\n".join(
        f"- {t['name']}: {t.get('description', '')}" if isinstance(t, dict) else f"- {t}"
        for t in tools
    ) or "(no tools available)"
    seed_lines = "\n".join(
        f"{i + 1}. design_philosophy={p.value} -- {PHILOSOPHY_LIBRARY[p.value]['summary']}"
        for i, p in enumerate(philosophies)
    )
    failure_block = (
        "Known failure modes from similar past attempts (avoid repeating these):\n"
        + "\n".join(f"- {f}" for f in failure_modes)
        if failure_modes
        else "No prior attempts on record for this goal."
    )
    system = (
        "You are the SYNTHESIZE step of a meta-agent architect. Given a goal and a "
        "set of available tools, emit a single JSON object of the form "
        f'{{"seeds": [...]}} containing exactly {len(philosophies)} AgentSpec '
        "objects, one per requested design philosophy below, each validating "
        f"against this JSON Schema:\n{json.dumps(schema)}\n"
        "Only reference tools from the provided list in each role's `tools`. "
        "Make the seeds genuinely different from each other in orchestration "
        "topology and role structure, not merely reworded."
    )
    user = (
        f"Goal: {goal}\n\nAvailable tools:\n{tool_lines}\n\n"
        f"Required seeds:\n{seed_lines}\n\n{failure_block}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ArchitectError(
            f"could not parse JSON from meta-layer response: {e}"
        ) from e


def _parse_seeds(text: str) -> list[dict[str, Any]]:
    payload = _extract_json(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("seeds", "specs", "agents"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise ArchitectError(f"synthesis response was not a list of seeds: {text!r}")


def _coerce_philosophy(candidate: dict[str, Any], philosophy: DesignPhilosophy) -> dict[str, Any]:
    candidate = dict(candidate)
    valid = {p.value for p in DesignPhilosophy}
    if candidate.get("design_philosophy") not in valid:
        candidate["design_philosophy"] = philosophy.value
    candidate.setdefault("version", SPEC_VERSION)
    return candidate


def _format_pydantic_errors(e: ValidationError) -> str:
    lines = []
    for err in e.errors():
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"{loc}: {err['msg']}")
    return "; ".join(lines)


def _format_verification_errors(errors: list[VerificationError]) -> str:
    return "; ".join(f"[{e.code}] {e.path}: {e.message}" for e in errors)


async def _repair(
    complete_fn: CompleteFn,
    candidate: dict[str, Any],
    errors_desc: str,
    philosophy: DesignPhilosophy,
    reasoning: Optional[Union[bool, str]],
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are the VERIFY/repair step of a meta-agent architect. The "
                "following AgentSpec JSON failed validation. Return corrected "
                "JSON for a single AgentSpec object only (no wrapper, no prose) "
                "that fixes every listed problem and keeps "
                f"design_philosophy={philosophy.value}."
            ),
        },
        {
            "role": "user",
            "content": f"Spec:\n{json.dumps(candidate)}\n\nProblems:\n{errors_desc}",
        },
    ]
    completion = await _complete_with_budget_retry(
        complete_fn,
        "meta",
        messages,
        response_format={"type": "json_object"},
        max_tokens=REPAIR_MAX_TOKENS,
        temperature=0.2,
        reasoning=reasoning,
    )
    payload = _extract_json(completion.text)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    return payload


async def _verify_and_repair(
    raw: dict[str, Any],
    tools: list[dict[str, Any]],
    complete_fn: CompleteFn,
    philosophy: DesignPhilosophy,
    max_repair_attempts: int,
    reasoning: Optional[Union[bool, str]] = None,
) -> Optional[AgentSpec]:
    candidate = raw
    errors_desc = "no candidate was synthesized for this seed"
    for attempt in range(max_repair_attempts + 1):
        candidate = _coerce_philosophy(candidate, philosophy)
        try:
            spec = AgentSpec.model_validate(candidate)
        except ValidationError as e:
            errors_desc = _format_pydantic_errors(e)
        else:
            result = verify_spec(spec, tools)
            if result.ok:
                return spec
            errors_desc = _format_verification_errors(result.errors)

        if attempt == max_repair_attempts:
            return None
        candidate = await _repair(complete_fn, candidate, errors_desc, philosophy, reasoning)
    return None
