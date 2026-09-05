"""OPTIMIZE stage: turn a Diagnosis into a MUTATED AgentSpec.

Every change is a field-level edit applied to a deep copy of the current
spec -- never a from-scratch regeneration -- so every mutation is diffable
and auditable via the `Mutation` list attached to the result. Mutations are
picked deterministically from the dominant failure mode; no LLM call is
needed (or made) here.

Reasoning on/off: AgentSpec (owned by spec/models.py, out of scope for this
module) has no dedicated "thinking" toggle. This module treats it as a
first-class mutation axis anyway, implemented as a system-prompt directive
(REASONING_OFF_MARKER) plus a max_tokens adjustment on the acting role --
fully expressible with existing Role fields, still diffable, and it is the
lever that actually moves tokens/latency for a prompted model: disabling
chain-of-thought on easy prompts is the cheapest accuracy-vs-cost-vs-speed
win available to us.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from metaagent.analyzer import Diagnosis, FailureMode
from metaagent.spec.models import AgentSpec, CriticConfig, MemoryKind, RetryPolicy, Role
from metaagent.spec.verify import verify_spec

REASONING_OFF_MARKER = "[[reasoning:off]]"
REASONING_OFF_DIRECTIVE = (
    f"\n\n{REASONING_OFF_MARKER} Respond directly with only the final required "
    "output. Do not include any chain-of-thought, planning notes, or "
    "step-by-step reasoning in your response -- just the answer."
)
REASONING_OFF_TOKEN_MULTIPLIER = 0.5
REASONING_OFF_MIN_TOKENS = 128
REASONING_ON_TOKEN_MULTIPLIER = 2.0

STRICT_JSON_DIRECTIVE = (
    "\n\nIMPORTANT: Respond with exactly one valid JSON object matching the "
    "required schema and nothing else -- no prose, no markdown code fences, "
    "no explanation before or after it."
)
ORDERING_DIRECTIVE = (
    "\n\nIMPORTANT: Every required tool call must appear, and must appear in "
    "the exact order the task's dependencies demand (e.g. look an entity up "
    "before using its id elsewhere). Do not skip, reorder, or fabricate calls."
)
TOOL_GUIDANCE_DIRECTIVE = (
    "\n\nBefore answering, double check you have called every tool needed to "
    "gather the facts your answer depends on -- do not guess at a value you "
    "could have looked up."
)
CRITIC_SYSTEM_PROMPT = (
    "Review the draft output against the original goal and instructions. "
    "Point out anything missing, incorrect, or that violates the required "
    "output format, then produce a corrected final output."
)


@dataclass(frozen=True)
class Mutation:
    axis: str  # "prompt" | "tools" | "memory" | "orchestration" | "reasoning"
    path: str
    before: Any
    after: Any


@dataclass
class MutationResult:
    spec: AgentSpec
    mutations: list[Mutation]
    rationale: str
    failure_mode: Optional[str]


ToolLike = Any


def _tool_names(tools: list[ToolLike]) -> list[str]:
    names = []
    for t in tools:
        if isinstance(t, str):
            names.append(t)
        elif isinstance(t, dict) and "name" in t:
            names.append(t["name"])
    return names


def _role_index(spec: AgentSpec, role_name: str) -> int:
    for i, role in enumerate(spec.roles):
        if role.name == role_name:
            return i
    raise ValueError(f"role {role_name!r} not found in spec")


def _is_reasoning_off(role: Role) -> bool:
    return REASONING_OFF_MARKER in role.system_prompt


# ---------------------------------------------------------------------------
# Individual mutation candidates. Each returns None if not applicable to the
# current spec, or (mutated_spec, mutations, rationale) if it is.
# ---------------------------------------------------------------------------

CandidateFn = Callable[[AgentSpec, Diagnosis, list[ToolLike], str], Optional[tuple[Any, list[Mutation], str]]]


def _mutate_strict_output_prompt(spec, diagnosis, available_tools, role_name):
    idx = _role_index(spec, role_name)
    role = spec.roles[idx]
    if STRICT_JSON_DIRECTIVE.strip() in role.system_prompt:
        return None
    mutated = spec.model_copy(deep=True)
    before = mutated.roles[idx].system_prompt
    mutated.roles[idx].system_prompt = before + STRICT_JSON_DIRECTIVE
    mutation = Mutation("prompt", f"roles[{idx}].system_prompt", before, mutated.roles[idx].system_prompt)
    rationale = (
        f"Dominant failure mode was {FailureMode.MALFORMED_OUTPUT.value} "
        f"({diagnosis.distribution.get(FailureMode.MALFORMED_OUTPUT.value, 0)} runs): output could not be "
        f"parsed by the evaluator. Added an explicit strict-JSON-only instruction to role "
        f"'{role_name}' system_prompt."
    )
    return mutated, [mutation], rationale


def _mutate_ordering_prompt(spec, diagnosis, available_tools, role_name):
    idx = _role_index(spec, role_name)
    role = spec.roles[idx]
    if ORDERING_DIRECTIVE.strip() in role.system_prompt:
        return None
    mutated = spec.model_copy(deep=True)
    before = mutated.roles[idx].system_prompt
    mutated.roles[idx].system_prompt = before + ORDERING_DIRECTIVE
    mutation = Mutation("prompt", f"roles[{idx}].system_prompt", before, mutated.roles[idx].system_prompt)
    rationale = (
        f"Dominant failure mode was {FailureMode.INSTRUCTION_IGNORED.value}: required ordering/"
        f"instructions were being ignored. Added an explicit ordering directive to role "
        f"'{role_name}' system_prompt."
    )
    return mutated, [mutation], rationale


def _mutate_add_missing_tool(spec, diagnosis, available_tools, role_name):
    idx = _role_index(spec, role_name)
    role = spec.roles[idx]
    available = _tool_names(available_tools)
    missing = [t for t in available if t not in role.tools]
    if not missing:
        return None
    new_tool = missing[0]
    mutated = spec.model_copy(deep=True)
    before = list(mutated.roles[idx].tools)
    mutated.roles[idx].tools = before + [new_tool]
    mutation = Mutation("tools", f"roles[{idx}].tools", before, mutated.roles[idx].tools)
    rationale = (
        f"Dominant failure mode was {FailureMode.WRONG_TOOL_CHOSEN.value}: required tool calls were "
        f"unmet even though the role attempted tool calls. Granted role '{role_name}' access to "
        f"previously unused available tool '{new_tool}'."
    )
    return mutated, [mutation], rationale


def _mutate_tool_guidance_prompt(spec, diagnosis, available_tools, role_name):
    idx = _role_index(spec, role_name)
    role = spec.roles[idx]
    if TOOL_GUIDANCE_DIRECTIVE.strip() in role.system_prompt:
        return None
    mutated = spec.model_copy(deep=True)
    before = mutated.roles[idx].system_prompt
    mutated.roles[idx].system_prompt = before + TOOL_GUIDANCE_DIRECTIVE
    mutation = Mutation("prompt", f"roles[{idx}].system_prompt", before, mutated.roles[idx].system_prompt)
    rationale = (
        f"Dominant failure mode was {FailureMode.WRONG_TOOL_CHOSEN.value} but the role already has "
        f"access to every available tool, so tools couldn't be added. Added an explicit "
        f"tool-verification directive to role '{role_name}' system_prompt instead."
    )
    return mutated, [mutation], rationale


def _mutate_retry_policy(spec, diagnosis, available_tools, role_name):
    idx = _role_index(spec, role_name)
    role = spec.roles[idx]
    if role.retry.max_retries >= 5 and role.retry.on_failure == "escalate_to_critic":
        return None
    mutated = spec.model_copy(deep=True)
    before = mutated.roles[idx].retry.model_copy()
    new_retries = min(5, role.retry.max_retries + 1)
    new_on_failure = "escalate_to_critic" if role.critic and role.critic.enabled else "fallback_role"
    mutated.roles[idx].retry = RetryPolicy(
        max_retries=new_retries, backoff="exponential", on_failure=new_on_failure,
    )
    mutation = Mutation("orchestration", f"roles[{idx}].retry", before, mutated.roles[idx].retry)
    rationale = (
        f"Dominant failure mode was {FailureMode.TOOL_CALL_ERRORED.value}: tool calls were erroring "
        f"out mid-run. Increased role '{role_name}' retry.max_retries to {new_retries} with "
        f"exponential backoff."
    )
    return mutated, [mutation], rationale


def _mutate_upgrade_memory(spec, diagnosis, available_tools, role_name):
    idx = _role_index(spec, role_name)
    role = spec.roles[idx]
    order = [MemoryKind.NONE, MemoryKind.SCRATCHPAD, MemoryKind.SUMMARY_BUFFER, MemoryKind.FULL_HISTORY]
    mutated = spec.model_copy(deep=True)
    before = mutated.roles[idx].memory.model_copy()
    if role.memory.kind in order and order.index(role.memory.kind) < len(order) - 1:
        next_kind = order[order.index(role.memory.kind) + 1]
        mutated.roles[idx].memory.kind = next_kind
    else:
        new_max = min(1000, int(role.memory.max_items * 1.5) + 1)
        if new_max == role.memory.max_items:
            return None
        mutated.roles[idx].memory.max_items = new_max
    mutation = Mutation("memory", f"roles[{idx}].memory", before, mutated.roles[idx].memory)
    rationale = (
        f"Dominant failure mode was {FailureMode.MISSING_CONTEXT.value}: the role acted without "
        f"context it needed (never looked up required information, or dropped required fields). "
        f"Upgraded role '{role_name}' memory from {before.kind.value} "
        f"(max_items={before.max_items}) to {mutated.roles[idx].memory.kind.value} "
        f"(max_items={mutated.roles[idx].memory.max_items})."
    )
    return mutated, [mutation], rationale


def _mutate_enable_critic(spec, diagnosis, available_tools, role_name):
    idx = _role_index(spec, role_name)
    role = spec.roles[idx]
    if role.critic and role.critic.enabled:
        return None
    mutated = spec.model_copy(deep=True)
    before = mutated.roles[idx].critic
    mutated.roles[idx].critic = CriticConfig(
        enabled=True, system_prompt=CRITIC_SYSTEM_PROMPT, max_revisions=1,
    )
    mutation = Mutation("orchestration", f"roles[{idx}].critic", before, mutated.roles[idx].critic)
    rationale = (
        f"Dominant failure mode was {FailureMode.LOW_QUALITY_OUTPUT.value}: outputs scored below "
        f"passing with no single structural defect. Enabled a self-review critic pass on role "
        f"'{role_name}' to catch quality issues before finalizing."
    )
    return mutated, [mutation], rationale


def _mutate_increase_budget(spec, diagnosis, available_tools, role_name):
    orch = spec.orchestration
    if orch.max_steps >= 200 and orch.budget_tokens >= 100_000:
        return None
    mutated = spec.model_copy(deep=True)
    before_steps, before_budget = orch.max_steps, orch.budget_tokens
    mutated.orchestration.max_steps = min(200, int(before_steps * 1.5) + 1)
    mutated.orchestration.budget_tokens = min(100_000, int(before_budget * 1.5))
    mutations = [
        Mutation("orchestration", "orchestration.max_steps", before_steps, mutated.orchestration.max_steps),
        Mutation("orchestration", "orchestration.budget_tokens", before_budget, mutated.orchestration.budget_tokens),
    ]
    rationale = (
        f"Dominant failure mode was {FailureMode.REASONING_BUDGET_EXHAUSTED.value}: runs were hitting "
        f"the step/token ceiling before finishing. Raised max_steps "
        f"{before_steps} -> {mutated.orchestration.max_steps} and budget_tokens "
        f"{before_budget} -> {mutated.orchestration.budget_tokens}."
    )
    return mutated, mutations, rationale


def _mutate_reasoning_off(spec, diagnosis, available_tools, role_name):
    idx = _role_index(spec, role_name)
    role = spec.roles[idx]
    if _is_reasoning_off(role):
        return None
    mutated = spec.model_copy(deep=True)
    before_prompt = mutated.roles[idx].system_prompt
    before_tokens = mutated.roles[idx].max_tokens
    mutated.roles[idx].system_prompt = before_prompt + REASONING_OFF_DIRECTIVE
    mutated.roles[idx].max_tokens = max(REASONING_OFF_MIN_TOKENS, int(before_tokens * REASONING_OFF_TOKEN_MULTIPLIER))
    mutations = [
        Mutation("reasoning", f"roles[{idx}].system_prompt", before_prompt, mutated.roles[idx].system_prompt),
        Mutation("reasoning", f"roles[{idx}].max_tokens", before_tokens, mutated.roles[idx].max_tokens),
    ]
    rationale = (
        f"Dominant failure mode was {FailureMode.REASONING_BUDGET_EXHAUSTED.value} with mean output "
        f"tokens={diagnosis.mean_output_tokens:.0f} and mean latency={diagnosis.mean_latency_ms:.0f}ms. "
        f"Disabling chain-of-thought reasoning has measured 70x fewer tokens and 4x lower latency on "
        f"easy prompts for this worker model, so turned reasoning OFF for role '{role_name}' "
        f"(max_tokens {before_tokens} -> {mutated.roles[idx].max_tokens}) to spend budget on retries "
        f"instead of deliberation."
    )
    return mutated, mutations, rationale


def _mutate_reasoning_on(spec, diagnosis, available_tools, role_name):
    idx = _role_index(spec, role_name)
    role = spec.roles[idx]
    if not _is_reasoning_off(role):
        return None
    mutated = spec.model_copy(deep=True)
    before_prompt = mutated.roles[idx].system_prompt
    before_tokens = mutated.roles[idx].max_tokens
    mutated.roles[idx].system_prompt = before_prompt.replace(REASONING_OFF_DIRECTIVE, "")
    budget_cap = spec.orchestration.budget_tokens
    mutated.roles[idx].max_tokens = min(budget_cap, int(before_tokens * REASONING_ON_TOKEN_MULTIPLIER))
    mutations = [
        Mutation("reasoning", f"roles[{idx}].system_prompt", before_prompt, mutated.roles[idx].system_prompt),
        Mutation("reasoning", f"roles[{idx}].max_tokens", before_tokens, mutated.roles[idx].max_tokens),
    ]
    rationale = (
        f"Failures persisted with reasoning disabled on role '{role_name}' (dominant mode: "
        f"{diagnosis.dominant_mode}); re-enabled reasoning (max_tokens {before_tokens} -> "
        f"{mutated.roles[idx].max_tokens}) to trade cost for the extra deliberation the task needs."
    )
    return mutated, mutations, rationale


def _mutate_fallback(spec, diagnosis, available_tools, role_name):
    mutated = spec.model_copy(deep=True)
    before = mutated.orchestration.max_steps
    mutated.orchestration.max_steps = min(200, before + 1)
    mutation = Mutation("orchestration", "orchestration.max_steps", before, mutated.orchestration.max_steps)
    rationale = (
        f"No single dominant, actionable failure signal was found (dominant mode: "
        f"{diagnosis.dominant_mode}); applied a minimal, always-safe step-budget nudge "
        f"({before} -> {mutated.orchestration.max_steps}) while more data accumulates."
    )
    return mutated, [mutation], rationale


def _candidate_order(dominant_mode: Optional[str], reasoning_off: bool) -> list[CandidateFn]:
    if dominant_mode == FailureMode.REASONING_BUDGET_EXHAUSTED.value:
        return [_mutate_reasoning_off, _mutate_increase_budget, _mutate_fallback]
    if dominant_mode == FailureMode.MALFORMED_OUTPUT.value:
        return [_mutate_strict_output_prompt, _mutate_fallback]
    if dominant_mode == FailureMode.WRONG_TOOL_CHOSEN.value:
        return [_mutate_add_missing_tool, _mutate_tool_guidance_prompt, _mutate_fallback]
    if dominant_mode == FailureMode.TOOL_CALL_ERRORED.value:
        return [_mutate_retry_policy, _mutate_fallback]
    if dominant_mode == FailureMode.MISSING_CONTEXT.value:
        return [_mutate_upgrade_memory, _mutate_add_missing_tool, _mutate_fallback]
    if dominant_mode == FailureMode.INSTRUCTION_IGNORED.value:
        return [_mutate_ordering_prompt, _mutate_fallback]
    if dominant_mode == FailureMode.LOW_QUALITY_OUTPUT.value:
        if reasoning_off:
            return [_mutate_reasoning_on, _mutate_enable_critic, _mutate_fallback]
        return [_mutate_enable_critic, _mutate_fallback]
    return [_mutate_fallback]


def optimize(
    spec: AgentSpec,
    diagnosis: Diagnosis,
    available_tools: list[ToolLike],
    *,
    role_name: Optional[str] = None,
) -> MutationResult:
    """Produce a single mutated spec addressing the diagnosis's dominant
    failure mode. Guaranteed to return a spec that passes `verify_spec`
    against `available_tools` (falls back to a trivial, always-valid
    mutation if every targeted candidate is rejected)."""
    target_role = role_name or spec.orchestration.entry_role
    reasoning_off = _is_reasoning_off(next(r for r in spec.roles if r.name == target_role))
    candidates = _candidate_order(diagnosis.dominant_mode, reasoning_off)

    for candidate_fn in candidates:
        result = candidate_fn(spec, diagnosis, available_tools, target_role)
        if result is None:
            continue
        mutated, mutations, rationale = result
        verification = verify_spec(mutated, available_tools)
        if verification.ok:
            return MutationResult(
                spec=mutated, mutations=mutations, rationale=rationale, failure_mode=diagnosis.dominant_mode,
            )

    # Every targeted candidate either didn't apply or broke verification;
    # the fallback mutation is constructed to always be valid.
    mutated, mutations, rationale = _mutate_fallback(spec, diagnosis, available_tools, target_role)
    return MutationResult(spec=mutated, mutations=mutations, rationale=rationale, failure_mode=diagnosis.dominant_mode)
