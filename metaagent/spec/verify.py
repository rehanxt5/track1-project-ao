"""Self-verification for AgentSpec: schema checks (pydantic) plus the
semantic checks pydantic can't express because they need the runtime
context (which tools actually exist) or cross-field graph reasoning.

Execution budget is the scarcest resource in this system, so a rejected
spec here is one that never burns an eval run. Errors are structured
(code + path + message) so the architect's repair loop can hand them
straight back to the meta-layer as feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

from metaagent.spec.models import AgentSpec


@dataclass(frozen=True)
class VerificationError:
    code: str
    path: str
    message: str


@dataclass
class VerificationResult:
    errors: list[VerificationError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok


ToolLike = Union[str, dict[str, Any]]


def _tool_names(tools: list[ToolLike]) -> set[str]:
    names = set()
    for t in tools:
        if isinstance(t, str):
            names.add(t)
        elif isinstance(t, dict) and "name" in t:
            names.add(t["name"])
    return names


def verify_spec(spec: AgentSpec, available_tools: list[ToolLike]) -> VerificationResult:
    errors: list[VerificationError] = []
    tool_names = _tool_names(available_tools)
    role_names = {r.name for r in spec.roles}

    for i, role in enumerate(spec.roles):
        path = f"roles[{i}]"
        if not role.system_prompt.strip():
            errors.append(
                VerificationError(
                    "empty_prompt", f"{path}.system_prompt",
                    f"role '{role.name}' has a blank system_prompt",
                )
            )
        if role.critic and role.critic.enabled and not (
            role.critic.system_prompt and role.critic.system_prompt.strip()
        ):
            errors.append(
                VerificationError(
                    "empty_prompt", f"{path}.critic.system_prompt",
                    f"role '{role.name}' enables a critic with a blank system_prompt",
                )
            )
        unknown_tools = [t for t in role.tools if t not in tool_names]
        if unknown_tools:
            errors.append(
                VerificationError(
                    "unknown_tool", f"{path}.tools",
                    f"role '{role.name}' references tools not in the provided "
                    f"tool list: {sorted(unknown_tools)}",
                )
            )
        if role.max_tokens > spec.orchestration.budget_tokens:
            errors.append(
                VerificationError(
                    "budget_too_small", "orchestration.budget_tokens",
                    f"role '{role.name}' max_tokens ({role.max_tokens}) exceeds "
                    f"the total orchestration budget_tokens "
                    f"({spec.orchestration.budget_tokens})",
                )
            )

    if spec.orchestration.entry_role not in role_names:
        errors.append(
            VerificationError(
                "invalid_entry_role", "orchestration.entry_role",
                f"entry_role '{spec.orchestration.entry_role}' is not a defined role",
            )
        )

    adjacency: dict[str, set[str]] = {name: set() for name in role_names}
    for i, step in enumerate(spec.orchestration.steps):
        path = f"orchestration.steps[{i}]"
        if step.role not in role_names:
            errors.append(
                VerificationError(
                    "unknown_role_in_step", f"{path}.role",
                    f"step references undefined role '{step.role}'",
                )
            )
        for n in step.next:
            if n not in role_names:
                errors.append(
                    VerificationError(
                        "unknown_role_in_step", f"{path}.next",
                        f"step hands off to undefined role '{n}'",
                    )
                )
            elif step.role in adjacency:
                adjacency[step.role].add(n)

    if spec.orchestration.entry_role in role_names:
        reachable = {spec.orchestration.entry_role}
        frontier = [spec.orchestration.entry_role]
        while frontier:
            current = frontier.pop()
            for nxt in adjacency.get(current, ()):
                if nxt not in reachable:
                    reachable.add(nxt)
                    frontier.append(nxt)
        orphaned = sorted(role_names - reachable)
        if orphaned:
            errors.append(
                VerificationError(
                    "orphaned_role", "orchestration.steps",
                    f"roles are defined but unreachable from entry_role "
                    f"'{spec.orchestration.entry_role}': {orphaned}",
                )
            )

    return VerificationResult(errors=errors)
