import asyncio
import json

import pytest

from metaagent import architect
from metaagent.archive import Archive
from metaagent.architect import ArchitectError, generate
from metaagent.gateway import GatewayReasoningBudgetError
from tests.fakes import FakeCompletion, SAMPLE_TOOLS, FakeGateway, valid_spec_dict


def test_generate_rejects_blank_goal():
    gateway = FakeGateway([])
    with pytest.raises(ValueError):
        asyncio.run(generate("   ", SAMPLE_TOOLS, complete_fn=gateway.complete))


def test_generate_returns_diverse_seeds():
    seeds_payload = {
        "seeds": [
            valid_spec_dict(design_philosophy="react"),
            valid_spec_dict(design_philosophy="plan_execute"),
            valid_spec_dict(design_philosophy="plan_critic_reflect"),
        ]
    }
    gateway = FakeGateway([seeds_payload])
    specs = asyncio.run(
        generate("Answer trivia", SAMPLE_TOOLS, n_seeds=3, complete_fn=gateway.complete)
    )
    assert len(specs) == 3
    assert {s.design_philosophy.value for s in specs} == {
        "react",
        "plan_execute",
        "plan_critic_reflect",
    }
    assert gateway.call_count == 1


def test_generate_works_with_cold_archive(tmp_path):
    archive = Archive(tmp_path / "cold.jsonl")
    seeds_payload = {
        "seeds": [
            valid_spec_dict(design_philosophy="react"),
            valid_spec_dict(design_philosophy="plan_execute"),
        ]
    }
    gateway = FakeGateway([seeds_payload])
    specs = asyncio.run(
        generate(
            "g", SAMPLE_TOOLS, n_seeds=2, archive=archive, complete_fn=gateway.complete
        )
    )
    assert len(specs) == 2


def test_generate_repairs_seed_with_unknown_tool():
    broken = valid_spec_dict(design_philosophy="react", tools=["search", "time_machine"])
    fixed = valid_spec_dict(design_philosophy="react", tools=["search"])
    good = valid_spec_dict(design_philosophy="plan_execute")
    gateway = FakeGateway(
        [
            {"seeds": [broken, good]},  # synthesize
            fixed,  # repair for seed 0
        ]
    )
    specs = asyncio.run(
        generate("Answer trivia", SAMPLE_TOOLS, n_seeds=2, complete_fn=gateway.complete)
    )
    assert len(specs) == 2
    assert gateway.call_count == 2
    assert {s.design_philosophy.value for s in specs} == {"react", "plan_execute"}
    react_spec = next(s for s in specs if s.design_philosophy.value == "react")
    assert react_spec.roles[0].tools == ["search"]


def test_generate_drops_seed_that_never_repairs():
    def always_broken(layer, messages):
        return valid_spec_dict(design_philosophy="react", tools=["search", "time_machine"])

    good = valid_spec_dict(design_philosophy="plan_execute")
    gateway = FakeGateway(
        [
            {
                "seeds": [
                    valid_spec_dict(design_philosophy="react", tools=["search", "time_machine"]),
                    good,
                ]
            },
            always_broken,  # repair attempt 1 for seed 0
            always_broken,  # repair attempt 2 for seed 0
        ]
    )
    specs = asyncio.run(
        generate(
            "g",
            SAMPLE_TOOLS,
            n_seeds=2,
            complete_fn=gateway.complete,
            max_repair_attempts=2,
        )
    )
    assert len(specs) == 1
    assert specs[0].design_philosophy.value == "plan_execute"
    assert gateway.call_count == 3


def test_generate_raises_when_all_seeds_fail():
    def always_broken(layer, messages):
        return valid_spec_dict(tools=["search", "time_machine"])

    gateway = FakeGateway(
        [
            {
                "seeds": [
                    valid_spec_dict(tools=["search", "time_machine"]),
                    valid_spec_dict(tools=["search", "time_machine"]),
                ]
            },
            always_broken,
            always_broken,
            always_broken,
            always_broken,
        ]
    )
    with pytest.raises(ArchitectError):
        asyncio.run(
            generate(
                "g",
                SAMPLE_TOOLS,
                n_seeds=2,
                complete_fn=gateway.complete,
                max_repair_attempts=2,
            )
        )


def test_generate_includes_archive_failure_modes_in_prompt(tmp_path):
    archive = Archive(tmp_path / "archive.jsonl")
    archive.add(
        goal="Answer trivia using search",
        domain="trivia",
        spec=valid_spec_dict(),
        scores={"accuracy": 0.9},
        failure_modes=["hallucinated an answer without calling search"],
    )
    seeds_payload = {
        "seeds": [
            valid_spec_dict(design_philosophy="react"),
            valid_spec_dict(design_philosophy="plan_execute"),
        ]
    }
    gateway = FakeGateway([seeds_payload])
    asyncio.run(
        generate(
            "Answer trivia using search",
            SAMPLE_TOOLS,
            n_seeds=2,
            archive=archive,
            complete_fn=gateway.complete,
        )
    )
    sent_messages = gateway.calls[0]["messages"]
    combined = " ".join(m["content"] for m in sent_messages)
    assert "hallucinated an answer without calling search" in combined


# ---------------------------------------------------------------------------
# Explicit token budgets for synthesis/repair (never rely on the gateway's
# global default -- that's what let the GENERATE stage ship 100% broken).
# ---------------------------------------------------------------------------

def test_generate_passes_explicit_max_tokens_sized_for_synthesis():
    seeds_payload = {
        "seeds": [
            valid_spec_dict(design_philosophy="react"),
            valid_spec_dict(design_philosophy="plan_execute"),
            valid_spec_dict(design_philosophy="plan_critic_reflect"),
        ]
    }
    gateway = FakeGateway([seeds_payload])
    asyncio.run(generate("g", SAMPLE_TOOLS, n_seeds=3, complete_fn=gateway.complete))

    assert gateway.calls[0]["max_tokens"] == architect._synthesis_max_tokens(3)
    assert gateway.calls[0]["max_tokens"] > 2048


def test_repair_passes_explicit_max_tokens():
    broken = valid_spec_dict(design_philosophy="react", tools=["search", "time_machine"])
    fixed = valid_spec_dict(design_philosophy="react", tools=["search"])
    gateway = FakeGateway([{"seeds": [broken]}, fixed])
    asyncio.run(generate("g", SAMPLE_TOOLS, n_seeds=1, complete_fn=gateway.complete))

    assert gateway.calls[1]["max_tokens"] == architect.REPAIR_MAX_TOKENS
    assert gateway.calls[1]["max_tokens"] > 2048


# ---------------------------------------------------------------------------
# reasoning parameter: threaded through to complete_fn, defaulted from
# config, overridable per call.
# ---------------------------------------------------------------------------

def test_generate_uses_reasoning_default_from_config(monkeypatch):
    monkeypatch.setattr(architect, "architect_reasoning_default", lambda: "low")
    seeds_payload = {"seeds": [valid_spec_dict(design_philosophy="react")]}
    gateway = FakeGateway([seeds_payload])

    asyncio.run(generate("g", SAMPLE_TOOLS, n_seeds=1, complete_fn=gateway.complete))

    assert gateway.calls[0]["reasoning"] == "low"


def test_generate_explicit_reasoning_overrides_config_default(monkeypatch):
    monkeypatch.setattr(architect, "architect_reasoning_default", lambda: "low")
    seeds_payload = {"seeds": [valid_spec_dict(design_philosophy="react")]}
    gateway = FakeGateway([seeds_payload])

    asyncio.run(
        generate("g", SAMPLE_TOOLS, n_seeds=1, complete_fn=gateway.complete, reasoning=False)
    )

    assert gateway.calls[0]["reasoning"] is False


# ---------------------------------------------------------------------------
# Self-healing retry: a reasoning-budget-exhaustion error on synthesis or
# repair gets exactly one retry at a larger budget before giving up.
# ---------------------------------------------------------------------------

def test_generate_retries_once_on_reasoning_budget_exhaustion():
    seeds_payload = {"seeds": [valid_spec_dict(design_philosophy="react")]}
    calls = []

    async def flaky_complete(layer, messages, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise GatewayReasoningBudgetError("reasoning budget exhausted: boom")
        return FakeCompletion(text=json.dumps(seeds_payload))

    specs = asyncio.run(
        generate("g", SAMPLE_TOOLS, n_seeds=1, complete_fn=flaky_complete)
    )

    assert len(specs) == 1
    assert len(calls) == 2
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]


def test_generate_gives_up_after_one_retry_still_exhausted():
    calls = []

    async def always_exhausted(layer, messages, **kwargs):
        calls.append(kwargs)
        raise GatewayReasoningBudgetError("reasoning budget exhausted: boom")

    with pytest.raises(GatewayReasoningBudgetError):
        asyncio.run(generate("g", SAMPLE_TOOLS, n_seeds=1, complete_fn=always_exhausted))

    assert len(calls) == 2
