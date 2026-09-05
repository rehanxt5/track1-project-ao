"""Declarative Agent Spec schema.

The architect emits one of these as JSON; a fixed interpreter (owned
elsewhere) executes it. Nothing in this module ever generates or runs
code — mutating a spec is how behaviour changes, along all four axes the
hackathon names: prompts, tools, memory, orchestration.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SPEC_VERSION = "1.0"


class DesignPhilosophy(str, Enum):
    """Fixed library the architect's SELECT step chooses from."""

    REACT = "react"
    PLAN_EXECUTE = "plan_execute"
    PLAN_CRITIC_REFLECT = "plan_critic_reflect"


class Topology(str, Enum):
    SINGLE_AGENT = "single_agent"
    SUPERVISOR_WORKERS = "supervisor_workers"
    PIPELINE = "pipeline"


class MemoryKind(str, Enum):
    NONE = "none"
    SCRATCHPAD = "scratchpad"
    SUMMARY_BUFFER = "summary_buffer"
    FULL_HISTORY = "full_history"
    VECTOR_RECALL = "vector_recall"


class MemoryStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: MemoryKind = MemoryKind.SCRATCHPAD
    max_items: int = Field(default=20, ge=1, le=1000)
    # Only meaningful for summary_buffer; interpreter ignores it otherwise.
    summarize_after: Optional[int] = Field(default=None, ge=1)


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(default=1, ge=0, le=5)
    backoff: Literal["none", "fixed", "exponential"] = "none"
    on_failure: Literal["abort", "fallback_role", "escalate_to_critic"] = "abort"


class CriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    system_prompt: Optional[str] = None
    max_revisions: int = Field(default=1, ge=0, le=5)
    acceptance_criteria: Optional[str] = None

    @model_validator(mode="after")
    def _prompt_required_if_enabled(self) -> "CriticConfig":
        if self.enabled and not (self.system_prompt and self.system_prompt.strip()):
            raise ValueError(
                "critic.system_prompt must be non-empty when critic.enabled is true"
            )
        return self


class Role(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)
    memory: MemoryStrategy = Field(default_factory=MemoryStrategy)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    critic: Optional[CriticConfig] = None
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def _name_not_blank(self) -> "Role":
        if not self.name.strip():
            raise ValueError("role name must not be blank")
        return self


class OrchestrationStep(BaseModel):
    """One node in the orchestration graph.

    `role` is the role that acts at this step; `next` lists role names it
    may hand off to (empty means terminal). The interpreter walks this
    graph starting at `Orchestration.entry_role`.
    """

    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1)
    next: list[str] = Field(default_factory=list)
    condition: Optional[str] = None


class Orchestration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topology: Topology
    entry_role: str = Field(min_length=1)
    steps: list[OrchestrationStep] = Field(default_factory=list)
    max_steps: int = Field(default=10, ge=1, le=200)
    budget_tokens: int = Field(default=20000, ge=1)


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["1.0"] = SPEC_VERSION
    id: Optional[str] = None
    goal: str = Field(min_length=1)
    design_philosophy: DesignPhilosophy
    orchestration: Orchestration
    roles: list[Role] = Field(min_length=1)

    @model_validator(mode="after")
    def _role_names_unique(self) -> "AgentSpec":
        names = [r.name for r in self.roles]
        if len(names) != len(set(names)):
            raise ValueError("role names must be unique within a spec")
        return self
