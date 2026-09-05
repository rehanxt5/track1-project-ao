"""Fake gateway + fixtures so metaagent tests run fully offline, with no
API keys, matching the fixed `complete(layer, messages, ...) -> Completion`
interface other workers own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class FakeCompletion:
    text: str
    input_tokens: int = 10
    output_tokens: int = 10
    latency_ms: float = 1.0
    model: str = "fake-model"
    provider: str = "fake"


class FakeGateway:
    """Scripted `complete` fn: each call pops the next canned response.

    A response can be a plain string (the completion text) or a callable
    that receives (layer, messages, kwargs) and returns text, for tests
    that need to react to what was actually asked.
    """

    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        layer: str,
        messages: list[dict[str, str]],
        *,
        response_format: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> FakeCompletion:
        self.calls.append(
            {
                "layer": layer,
                "messages": messages,
                "response_format": response_format,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self._responses:
            raise AssertionError("FakeGateway ran out of scripted responses")
        response = self._responses.pop(0)
        if callable(response):
            response = response(layer, messages)
        if not isinstance(response, str):
            response = json.dumps(response)
        return FakeCompletion(text=response)

    @property
    def call_count(self) -> int:
        return len(self.calls)


SAMPLE_TOOLS = [
    {"name": "search", "description": "Search the web for a query."},
    {"name": "calculator", "description": "Evaluate a math expression."},
]


def valid_spec_dict(
    *,
    name: str = "solver",
    design_philosophy: str = "react",
    tools: Optional[list[str]] = None,
    goal: str = "Answer a trivia question using search and a calculator.",
) -> dict[str, Any]:
    tools = tools if tools is not None else ["search", "calculator"]
    return {
        "version": "1.0",
        "goal": goal,
        "design_philosophy": design_philosophy,
        "orchestration": {
            "topology": "single_agent",
            "entry_role": name,
            "steps": [],
            "max_steps": 10,
            "budget_tokens": 20000,
        },
        "roles": [
            {
                "name": name,
                "system_prompt": "You solve the goal using the available tools.",
                "tools": tools,
                "memory": {"kind": "scratchpad", "max_items": 20},
                "retry": {"max_retries": 1, "backoff": "none", "on_failure": "abort"},
                "max_tokens": 1024,
                "temperature": 0.2,
            }
        ],
    }
