"""The RUN stage: a fixed interpreter that executes any valid AgentSpec.

Specs are data; this module is the only thing that turns them into actual
LLM calls, tool calls, and scores. It never `eval`s or `exec`s anything a
model produces -- model output is only ever `json.loads`'d.

How a spec is executed
-----------------------
The interpreter walks `spec.orchestration.steps` starting at `entry_role`,
running one role's turn at a time and following `OrchestrationStep.next` to
hand off between roles (a role with no matching step, or an empty `next`,
is terminal). Within a role's turn it runs a ReAct-style loop: ask the
worker layer for a single JSON decision -- either
`{"action": "tool_call", "tool": ..., "tool_args": {...}}` or
`{"action": "final_answer", "final_output": <text>}` -- execute a real tool
call when asked, and feed the result back, until the role produces a final
answer. `final_output` is opaque free text: it is exactly what the task's
own output-format instructions require (e.g. `{"sql": "..."}"`), graded
verbatim by the domain evaluator; the action/tool_call wrapper around it is
purely an interpreter<->model protocol.

If `role.critic.enabled`, a critic pass reviews the draft and can request
up to `critic.max_revisions` text-only revisions before the role's turn is
considered done.

`role.memory.kind` controls how much prior scratchpad (thought/action/
observation) history is included in each subsequent call: `none` includes
none, `full_history` includes everything, `scratchpad`/`vector_recall` keep
the most recent `max_items` entries (no vector store is available to this
stage, so vector_recall is approximated as recency), and `summary_buffer`
collapses everything past `summarize_after` (or `max_items`) entries into
one condensed line instead of dropping it or re-summarizing it with a
costly extra LLM call.

`role.retry` governs both malformed/failed LLM calls and tool calls that
raise (tools that return `{"error": ...}` as *data* are not retried here --
that's the model's job to notice and react to, per each domain's own tool
convention). Once retries are exhausted, `on_failure` determines what
happens: `abort` stops the run, `fallback_role` ends this role's turn early
so the orchestration graph can hand off to whatever's next (or finish),
and `escalate_to_critic` asks the role's critic for one best-effort piece
of guidance and retries the role's turn incorporating it.

`orchestration.max_steps` bounds the total number of TraceStep entries of
any kind (llm/tool/critic) the run may produce; `orchestration.budget_tokens`
bounds total real token usage (input+output+reasoning, summed from the
gateway's own `Completion`, never estimated). Exceeding either aborts the
run with `error` set and a forced score of 0 -- never an exception.

Grading is deliberately NOT this module's job: run_spec has no evaluator,
so a successful RunResult always carries `score=None`. run_batch is the one
that knows the domain, so it loads that domain's evaluator once and scores
each RunResult after the fact (except for runs that already failed, whose
score is forced to 0 regardless of what evaluating empty/garbage output
would produce).
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Union

from metaagent.evaluation import DOMAINS_DIR, Score, load_evaluator, load_tasks, load_tools
from metaagent.spec.models import AgentSpec, CriticConfig, MemoryKind, MemoryStrategy, Role
from metaagent.trace import TraceStep, append_trace_jsonl

_RUN_TIMEOUT_SECONDS = 300.0
_BACKOFF_BASE_SECONDS = 0.01  # kept tiny so retry/backoff paths stay fast under test

_ROLE_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "description": "'tool_call' or 'final_answer'"},
        "tool": {"type": "string"},
        "tool_args": {"type": "object"},
        "final_output": {"type": "string"},
    },
    "required": ["action"],
}

_CRITIC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "description": "'accept' or 'revise'"},
        "feedback": {"type": "string"},
    },
    "required": ["verdict"],
}


# ---------------------------------------------------------------------------
# Gateway completion protocol (duck-typed; mirrors architect.py's CompleteFn
# so both modules can be handed the same real or fake `complete`).
# ---------------------------------------------------------------------------


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
    ) -> Completion: ...


# ---------------------------------------------------------------------------
# Public result types (the hard contract)
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    task_id: str
    output: str
    score: Optional[Score]
    steps: list[TraceStep]
    error: Optional[str]
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_latency_ms: float
    step_count: int
    tool_call_count: int


@dataclass
class BatchResult:
    spec_version: int
    domain: str
    split: str
    k: int
    mean_score: float
    pass_rate: float
    score_stddev: float  # mean of per-task stddev across k repeats -- the RELIABILITY metric
    mean_input_tokens: float
    mean_output_tokens: float
    mean_reasoning_tokens: float
    mean_latency_ms: float
    mean_steps: float
    runs: list[RunResult]


# ---------------------------------------------------------------------------
# Internal run state
# ---------------------------------------------------------------------------


@dataclass
class _RunCtx:
    spec: AgentSpec
    task: dict
    tools_by_name: dict[str, dict]
    complete_fn: CompleteFn
    steps: list[TraceStep] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    error: Optional[str] = None

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens + self.total_reasoning_tokens

    @property
    def budget_exhausted(self) -> bool:
        return self.total_tokens >= self.spec.orchestration.budget_tokens

    @property
    def steps_exhausted(self) -> bool:
        return self.step_count >= self.spec.orchestration.max_steps

    def record(self, step: TraceStep) -> None:
        self.steps.append(step)
        self.total_input_tokens += step.input_tokens
        self.total_output_tokens += step.output_tokens
        self.total_reasoning_tokens += step.reasoning_tokens


# ---------------------------------------------------------------------------
# run_spec: execute one spec on one task
# ---------------------------------------------------------------------------


async def run_spec(
    spec: AgentSpec,
    task: dict,
    tools: list[dict],
    *,
    complete_fn: Optional[CompleteFn] = None,
) -> RunResult:
    if complete_fn is None:
        from metaagent.gateway import complete as complete_fn  # noqa: PLC0415

    tools_by_name = {t["name"]: t for t in tools if "name" in t}
    ctx = _RunCtx(spec=spec, task=task, tools_by_name=tools_by_name, complete_fn=complete_fn)

    output = ""
    error: Optional[str] = None
    try:
        output = await asyncio.wait_for(_walk_orchestration(ctx), timeout=_RUN_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        error = f"run timed out after {_RUN_TIMEOUT_SECONDS}s"
    except Exception as e:  # noqa: BLE001 - a spec crashing must never abort the batch
        error = f"unhandled error: {e!r}"

    if error is None and ctx.error is not None:
        error = ctx.error

    budget = spec.orchestration.budget_tokens
    if error is None and ctx.total_tokens > budget:
        error = f"budget_tokens ({budget}) exceeded (used {ctx.total_tokens})"

    total_latency_ms = sum(s.latency_ms for s in ctx.steps)
    tool_call_count = sum(1 for s in ctx.steps if s.kind == "tool")
    score = Score(score=0.0, passed=False, details={"error": error}) if error else None

    return RunResult(
        task_id=str(task.get("id", "unknown")),
        output=output,
        score=score,
        steps=ctx.steps,
        error=error,
        input_tokens=ctx.total_input_tokens,
        output_tokens=ctx.total_output_tokens,
        reasoning_tokens=ctx.total_reasoning_tokens,
        total_latency_ms=total_latency_ms,
        step_count=ctx.step_count,
        tool_call_count=tool_call_count,
    )


# ---------------------------------------------------------------------------
# Orchestration graph walk
# ---------------------------------------------------------------------------


async def _walk_orchestration(ctx: _RunCtx) -> str:
    orch = ctx.spec.orchestration
    steps_by_role = {s.role: s for s in orch.steps}
    roles_by_name = {r.name: r for r in ctx.spec.roles}

    role_name: Optional[str] = orch.entry_role
    last_output = ""
    hops = 0
    max_hops = max(len(ctx.spec.roles) * 4, 8)  # guards against a cycle in orchestration.steps

    while role_name is not None:
        hops += 1
        if hops > max_hops:
            ctx.error = ctx.error or "orchestration graph exceeded maximum role hops (possible cycle)"
            break

        role = roles_by_name.get(role_name)
        if role is None:
            ctx.error = ctx.error or f"orchestration references unknown role {role_name!r}"
            break

        last_output = await _run_role_turn(ctx, role, last_output)
        if ctx.error:
            break

        step = steps_by_role.get(role_name)
        nxt = step.next if step else []
        role_name = nxt[0] if nxt else None

    return last_output


async def _run_role_turn(ctx: _RunCtx, role: Role, incoming: str) -> str:
    history: list[dict[str, str]] = []
    draft = ""

    while True:
        if ctx.error:
            return draft
        if ctx.steps_exhausted:
            ctx.error = f"max_steps ({ctx.spec.orchestration.max_steps}) exceeded"
            return draft
        if ctx.budget_exhausted:
            ctx.error = f"budget_tokens ({ctx.spec.orchestration.budget_tokens}) exceeded"
            return draft

        messages = _build_messages(ctx, role, incoming, history)
        completion, llm_error = await _call_llm_with_retry(ctx, role, messages, kind="llm")
        if completion is None:
            if ctx.error:
                return draft
            should_continue = await _handle_on_failure(
                ctx, role, f"llm call failed after retries: {llm_error}", history
            )
            if not should_continue:
                return draft
            continue

        action = _parse_action(completion.text)
        history.append({"type": "llm", "content": completion.text})

        if action.get("action") != "tool_call":
            draft = str(action.get("final_output", completion.text))
            break

        tool_name = action.get("tool")
        tool_args = action.get("tool_args")
        tool_args = tool_args if isinstance(tool_args, dict) else {}

        tool_def = ctx.tools_by_name.get(tool_name) if isinstance(tool_name, str) else None
        allowed = isinstance(tool_name, str) and tool_name in role.tools
        fn = tool_def.get("fn") if tool_def else None

        if not allowed or fn is None:
            reason = (
                f"tool {tool_name!r} not available to role {role.name!r}"
                if not allowed
                else f"tool {tool_name!r} has no registered implementation"
            )
            ctx.record(
                TraceStep(
                    index=ctx.step_count,
                    role=role.name,
                    kind="tool",
                    input=tool_args,
                    output=None,
                    tool_name=tool_name if isinstance(tool_name, str) else None,
                    error=reason,
                    input_tokens=0,
                    output_tokens=0,
                    reasoning_tokens=0,
                    latency_ms=0.0,
                )
            )
            history.append({"type": "tool", "content": f"error: {reason}"})
            should_continue = await _handle_on_failure(ctx, role, reason, history)
            if not should_continue:
                return draft
            continue

        result, tool_error = await _call_tool_with_retry(ctx, role, tool_name, fn, tool_args)
        if tool_error is not None:
            if ctx.error:
                return draft
            history.append({"type": "tool", "content": f"error: {tool_error}"})
            should_continue = await _handle_on_failure(ctx, role, f"tool {tool_name!r} failed: {tool_error}", history)
            if not should_continue:
                return draft
            continue

        history.append(
            {"type": "tool", "content": json.dumps({"tool": tool_name, "args": tool_args, "result": result})}
        )

    if role.critic and role.critic.enabled:
        draft = await _run_critic_loop(ctx, role, incoming, history, draft)

    return draft


async def _handle_on_failure(ctx: _RunCtx, role: Role, detail: str, history: list[dict[str, str]]) -> bool:
    """Apply role.retry.on_failure once retries are exhausted.

    Returns True if the role's turn should retry from the top (a fresh LLM
    decision), False if it should stop now. Only the "abort" path sets
    ctx.error -- "fallback_role" ends the turn quietly so orchestration can
    hand off to whatever role is next (or finish).
    """
    policy = role.retry.on_failure
    if policy == "escalate_to_critic" and role.critic and role.critic.enabled:
        feedback = await _run_critic_feedback(ctx, role, detail)
        history.append({"type": "critic", "content": feedback})
        return not (ctx.error or ctx.steps_exhausted or ctx.budget_exhausted)
    if policy == "fallback_role":
        return False
    ctx.error = f"role {role.name!r} failed: {detail}"
    return False


# ---------------------------------------------------------------------------
# LLM + tool calls with retry/backoff
# ---------------------------------------------------------------------------


async def _backoff(kind: str, attempt: int) -> None:
    if kind == "fixed":
        await asyncio.sleep(_BACKOFF_BASE_SECONDS)
    elif kind == "exponential":
        await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))


async def _call_llm_with_retry(
    ctx: _RunCtx, role: Role, messages: list[dict[str, str]], *, kind: str
) -> tuple[Optional[Completion], Optional[str]]:
    policy = role.retry
    last_error: Optional[str] = None
    schema = _CRITIC_SCHEMA if kind == "critic" else _ROLE_ACTION_SCHEMA

    for attempt in range(policy.max_retries + 1):
        if ctx.steps_exhausted:
            ctx.error = f"max_steps ({ctx.spec.orchestration.max_steps}) exceeded"
            return None, ctx.error

        start = time.perf_counter()
        try:
            completion = await ctx.complete_fn(
                "worker",
                messages,
                response_format={"type": "json_object", "schema": schema},
                max_tokens=role.max_tokens,
                temperature=role.temperature,
            )
        except Exception as e:  # noqa: BLE001 - gateway/provider failure, not a bug
            latency_ms = (time.perf_counter() - start) * 1000
            last_error = str(e)
            ctx.record(
                TraceStep(
                    index=ctx.step_count,
                    role=role.name,
                    kind=kind,
                    input=messages,
                    output=None,
                    tool_name=None,
                    error=last_error,
                    input_tokens=0,
                    output_tokens=0,
                    reasoning_tokens=0,
                    latency_ms=latency_ms,
                )
            )
        else:
            ctx.record(
                TraceStep(
                    index=ctx.step_count,
                    role=role.name,
                    kind=kind,
                    input=messages,
                    output=completion.text,
                    tool_name=None,
                    error=None,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    reasoning_tokens=getattr(completion, "reasoning_tokens", 0) or 0,
                    latency_ms=completion.latency_ms,
                )
            )
            return completion, None

        if attempt < policy.max_retries:
            await _backoff(policy.backoff, attempt)

    return None, last_error


async def _call_tool_with_retry(
    ctx: _RunCtx, role: Role, tool_name: str, fn: Any, tool_args: dict
) -> tuple[Any, Optional[str]]:
    policy = role.retry
    last_error: Optional[str] = None

    for attempt in range(policy.max_retries + 1):
        if ctx.steps_exhausted:
            ctx.error = f"max_steps ({ctx.spec.orchestration.max_steps}) exceeded"
            return None, ctx.error

        start = time.perf_counter()
        try:
            result = fn(**tool_args)
        except Exception as e:  # noqa: BLE001 - a flaky/broken tool, not an interpreter bug
            latency_ms = (time.perf_counter() - start) * 1000
            last_error = f"{type(e).__name__}: {e}"
            ctx.record(
                TraceStep(
                    index=ctx.step_count,
                    role=role.name,
                    kind="tool",
                    input=tool_args,
                    output=None,
                    tool_name=tool_name,
                    error=last_error,
                    input_tokens=0,
                    output_tokens=0,
                    reasoning_tokens=0,
                    latency_ms=latency_ms,
                )
            )
        else:
            latency_ms = (time.perf_counter() - start) * 1000
            # Domains report failed lookups as {"error": ...} data rather than
            # raising -- that's a normal observation for the model to react
            # to, not a retryable interpreter-level failure.
            data_error = result.get("error") if isinstance(result, dict) else None
            ctx.record(
                TraceStep(
                    index=ctx.step_count,
                    role=role.name,
                    kind="tool",
                    input=tool_args,
                    output=result,
                    tool_name=tool_name,
                    error=data_error,
                    input_tokens=0,
                    output_tokens=0,
                    reasoning_tokens=0,
                    latency_ms=latency_ms,
                )
            )
            return result, None

        if attempt < policy.max_retries:
            await _backoff(policy.backoff, attempt)

    return None, last_error


async def _run_critic_feedback(ctx: _RunCtx, role: Role, detail: str) -> str:
    """Best-effort, single-attempt critic call used by on_failure=escalate_to_critic.

    Deliberately not wrapped in its own retry loop -- this is already the
    fallback path for an exhausted retry budget.
    """
    if ctx.steps_exhausted or ctx.budget_exhausted:
        return "(no budget left to consult the critic)"

    critic_cfg = role.critic
    messages = [
        {
            "role": "system",
            "content": (critic_cfg.system_prompt if critic_cfg else "You give short corrective guidance."),
        },
        {"role": "user", "content": f"The role {role.name!r} just failed: {detail}\nSuggest how to proceed."},
    ]
    start = time.perf_counter()
    try:
        completion = await ctx.complete_fn("worker", messages, max_tokens=role.max_tokens, temperature=role.temperature)
    except Exception as e:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000
        ctx.record(
            TraceStep(
                index=ctx.step_count,
                role=role.name,
                kind="critic",
                input=messages,
                output=None,
                tool_name=None,
                error=str(e),
                input_tokens=0,
                output_tokens=0,
                reasoning_tokens=0,
                latency_ms=latency_ms,
            )
        )
        return f"(critic escalation failed: {e})"

    ctx.record(
        TraceStep(
            index=ctx.step_count,
            role=role.name,
            kind="critic",
            input=messages,
            output=completion.text,
            tool_name=None,
            error=None,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            reasoning_tokens=0,
            latency_ms=completion.latency_ms,
        )
    )
    return completion.text


# ---------------------------------------------------------------------------
# Critic accept/revise loop (runs after a role produces a draft)
# ---------------------------------------------------------------------------


async def _run_critic_loop(
    ctx: _RunCtx, role: Role, incoming: str, history: list[dict[str, str]], draft: str
) -> str:
    critic_cfg = role.critic
    assert critic_cfg is not None
    revisions = 0

    while revisions <= critic_cfg.max_revisions:
        if ctx.error or ctx.steps_exhausted or ctx.budget_exhausted:
            break

        messages = _build_critic_review_messages(role, critic_cfg, draft)
        completion, _ = await _call_llm_with_retry(ctx, role, messages, kind="critic")
        if completion is None:
            break

        verdict = _parse_critic(completion.text)
        history.append({"type": "critic", "content": completion.text})

        if verdict.get("verdict") != "revise" or revisions >= critic_cfg.max_revisions:
            break

        feedback = str(verdict.get("feedback", ""))
        history.append({"type": "critic_feedback", "content": feedback})
        revisions += 1
        draft = await _produce_revision(ctx, role, incoming, history)

    return draft


async def _produce_revision(ctx: _RunCtx, role: Role, incoming: str, history: list[dict[str, str]]) -> str:
    messages = _build_messages(ctx, role, incoming, history)
    messages.append(
        {
            "role": "user",
            "content": (
                "Incorporate the critic feedback above and reply again with the same "
                '{"action": "final_answer", "final_output": <text>} JSON format.'
            ),
        }
    )
    completion, _ = await _call_llm_with_retry(ctx, role, messages, kind="llm")
    if completion is None:
        return ""
    action = _parse_action(completion.text)
    return str(action.get("final_output", completion.text))


# ---------------------------------------------------------------------------
# Prompt construction, memory rendering, output parsing
# ---------------------------------------------------------------------------


def _build_messages(ctx: _RunCtx, role: Role, incoming: str, history: list[dict[str, str]]) -> list[dict[str, str]]:
    tool_names = [n for n in role.tools if n in ctx.tools_by_name]
    tool_lines = (
        "\n".join(
            f"- {n}: {ctx.tools_by_name[n].get('description', '')} "
            f"params={json.dumps(ctx.tools_by_name[n].get('parameters', {}))}"
            for n in tool_names
        )
        or "(no tools available to this role)"
    )
    system = (
        f"{role.system_prompt}\n\n"
        "You are one role inside a larger agent system. On every turn, reply with a single JSON "
        'object of the form {"action": "tool_call", "tool": <name>, "tool_args": {...}} to call '
        'one of your tools, or {"action": "final_answer", "final_output": <text>} once you are '
        "done -- final_output must be the literal text answer the task's own instructions "
        "require, formatted exactly as specified there (it is graded verbatim, this wrapper is "
        "only how you talk to the interpreter).\n\n"
        f"Tools available to you:\n{tool_lines}"
    )
    task_text = ctx.task.get("question") or ctx.task.get("prompt") or json.dumps(ctx.task)

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if incoming:
        messages.append({"role": "user", "content": f"Input from the previous step:\n{incoming}"})
    messages.append({"role": "user", "content": f"Task:\n{task_text}"})
    messages.extend(_render_history(role.memory, history))
    return messages


def _render_history(memory: MemoryStrategy, history: list[dict[str, str]]) -> list[dict[str, str]]:
    if memory.kind == MemoryKind.NONE or not history:
        return []

    if memory.kind == MemoryKind.FULL_HISTORY:
        entries = history
    elif memory.kind == MemoryKind.SUMMARY_BUFFER:
        limit = memory.summarize_after or memory.max_items
        if limit and len(history) > limit:
            older, recent = history[:-limit], history[-limit:]
            summary = "; ".join(f"[{e['type']}] {str(e['content'])[:120]}" for e in older)
            entries = [{"type": "summary", "content": f"Earlier steps (summarized): {summary}"}] + recent
        else:
            entries = history
    else:  # SCRATCHPAD, VECTOR_RECALL -- no vector store here, so recency-bounded like a scratchpad
        entries = history[-memory.max_items :]

    lines = "\n".join(f"[{e['type']}] {e['content']}" for e in entries)
    return [{"role": "user", "content": f"Scratchpad so far:\n{lines}"}]


def _build_critic_review_messages(role: Role, critic_cfg: CriticConfig, draft: str) -> list[dict[str, str]]:
    system = (
        f"{critic_cfg.system_prompt}\n\n"
        "Review the draft output below against the acceptance criteria. Reply with a single JSON "
        'object {"verdict": "accept"} or {"verdict": "revise", "feedback": <what to fix>}.\n\n'
        f"Acceptance criteria: {critic_cfg.acceptance_criteria or '(use your judgement)'}"
    )
    user = f"Role being reviewed: {role.name}\n\nDraft output:\n{draft}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_action(text: str) -> dict[str, Any]:
    parsed = _safe_json_object(text)
    return parsed if parsed is not None else {"action": "final_answer", "final_output": text}


def _parse_critic(text: str) -> dict[str, Any]:
    parsed = _safe_json_object(text)
    return parsed if parsed is not None else {"verdict": "accept"}


def _safe_json_object(text: str) -> Optional[dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# run_batch: k repeats x every task in a domain/split
# ---------------------------------------------------------------------------


def _spec_version_int(spec: AgentSpec) -> int:
    try:
        return int(float(spec.version))
    except (TypeError, ValueError):
        return 0


def _load_domain_tool_fns(domain: str) -> dict[str, Any]:
    """Load the domain's TOOLS dict of callables (a convention documented in
    each domains/<domain>/tools.py, not part of metaagent.evaluation)."""
    path = DOMAINS_DIR / domain / "tools.py"
    if not path.exists():
        return {}
    module_spec = importlib.util.spec_from_file_location(f"domains.{domain}.tools", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, "TOOLS", {})


async def run_batch(
    spec: AgentSpec,
    domain: str,
    split: str,
    k: int = 3,
    concurrency: int = 4,
    *,
    complete_fn: Optional[CompleteFn] = None,
    trace_path: Optional[Union[str, Path]] = None,
) -> BatchResult:
    if complete_fn is None:
        from metaagent.gateway import complete as complete_fn  # noqa: PLC0415

    tasks = load_tasks(domain, split)
    evaluator = load_evaluator(domain)
    tool_schemas = load_tools(domain)
    tool_fns = _load_domain_tool_fns(domain)
    tools = [{**schema, "fn": tool_fns.get(schema["name"])} for schema in tool_schemas]

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _run_one(task: dict) -> RunResult:
        async with semaphore:
            result = await run_spec(spec, task, tools, complete_fn=complete_fn)
        if result.error is None:
            try:
                result.score = evaluator.evaluate(task, result.output)
            except Exception as e:  # noqa: BLE001 - a broken evaluator must not abort the batch
                result.error = f"evaluator crashed: {e!r}"
                result.score = Score(score=0.0, passed=False, details={"error": result.error})
        if trace_path is not None:
            append_trace_jsonl(trace_path, result)
        return result

    all_results = await asyncio.gather(*(_run_one(task) for task in tasks for _ in range(k)))

    runs_by_task: dict[str, list[RunResult]] = {}
    for r in all_results:
        runs_by_task.setdefault(r.task_id, []).append(r)
    per_task_stddev = [statistics.pstdev([r.score.score for r in group]) for group in runs_by_task.values()]

    scores = [r.score.score for r in all_results]
    passed_flags = [bool(r.score and r.score.passed) for r in all_results]

    return BatchResult(
        spec_version=_spec_version_int(spec),
        domain=domain,
        split=split,
        k=k,
        mean_score=statistics.fmean(scores),
        pass_rate=sum(passed_flags) / len(passed_flags),
        score_stddev=statistics.fmean(per_task_stddev) if per_task_stddev else 0.0,
        mean_input_tokens=statistics.fmean(r.input_tokens for r in all_results),
        mean_output_tokens=statistics.fmean(r.output_tokens for r in all_results),
        mean_reasoning_tokens=statistics.fmean(r.reasoning_tokens for r in all_results),
        mean_latency_ms=statistics.fmean(r.total_latency_ms for r in all_results),
        mean_steps=statistics.fmean(r.step_count for r in all_results),
        runs=all_results,
    )
